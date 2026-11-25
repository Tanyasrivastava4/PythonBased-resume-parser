"""
server.py
=========
Production FastAPI Server for ATS Resume Parser Engine

Key Capabilities:
  1. Lifespan Startup: Pre-loads Surya OCR predictors into RAM ONCE at boot time
     (eliminates 30-45s cold starts permanently). Skipped when OCR_ENGINE=tesseract.
  2. Fast-Path Execution: Digital PDFs/DOCXs process synchronously (< 0.1s).
  3. Asynchronous Background Execution: Scanned/Image PDFs return HTTP 202
     Accepted immediately + job_id, processing heavy OCR in background tasks so
     the CRM frontend never freezes or times out.
  4. Batch Upload Support: Accepts multi-file list uploads (100+ files) in one API call.

OCR engine switch:
    OCR_ENGINE=surya      (default) Surya primary, tesseract fallback
    OCR_ENGINE=tesseract  tesseract only (about 35x faster on CPU, no model preload)

Run Server:
    ./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
    OCR_ENGINE=tesseract ./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
# Cap PyTorch & OpenMP CPU threads to 2 to prevent RAM/CPU spikes that cause VSCode/OS OOM killer to close applications
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

import uuid
import logging
import threading
import asyncio
import concurrent.futures
from typing import List, Optional, Dict, Any

from qdrant_sync import sync_resume_to_qdrant, check_candidate_conflict, execute_candidate_resolution
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from run_pipeline import process_resume, EXTRACTED_TEXT_DIR, SEGMENTED_DIR, FLAT_JSON_DIR


logger = logging.getLogger("ats_server")
logging.basicConfig(level=logging.INFO)

import gc

# In-memory storage for candidate conflicts awaiting user resolution
PENDING_CONFLICTS: Dict[str, Dict[str, Any]] = {}

# "surya" (default) or "tesseract" -- must match the switch in ingestion/ocr_reader.py
#when we were using surya
#OCR_ENGINE = os.environ.get("OCR_ENGINE", "surya").lower()

#when are using tesseract
OCR_ENGINE = os.environ.get("OCR_ENGINE", "tesseract").lower()

# Temporary directory for API file uploads (isolated from resumes/ benchmark directory)
UPLOAD_TEMP_DIR = Path("tmp_uploads")
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store for async background OCR jobs with Bounded LRU Storage Cap
MAX_JOBS = 50
JOBS = {}


def _purge_oldest_jobs_if_needed():
    """Keeps the in-memory JOBS storage bounded under MAX_JOBS to prevent RAM leaks."""
    if len(JOBS) >= MAX_JOBS:
        # Purge oldest completed or failed jobs first
        completed = [k for k, v in JOBS.items() if v.get("status") in ("completed", "failed")]
        if completed:
            del JOBS[completed[0]]
        else:
            del JOBS[next(iter(JOBS))]
        gc.collect()


# ── Bulk Batch Processing Manager (100-200 files) ───────────────────────────
MAX_BATCHES = 20
BATCH_STORE = {}
BATCH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class BatchJob:
    def __init__(self, batch_id: str, total_files: int):
        self.batch_id = batch_id
        self.total = total_files
        self.completed = 0
        self.failed = 0
        self.status = "processing"  # "processing" -> "completed"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at = None
        self.candidates = []
        self.lock = threading.Lock()


def _purge_oldest_batches_if_needed():
    """Keeps the in-memory BATCH_STORE bounded under MAX_BATCHES to prevent RAM leaks."""
    if len(BATCH_STORE) >= MAX_BATCHES:
        completed = [k for k, v in BATCH_STORE.items() if v.status == "completed"]
        if completed:
            del BATCH_STORE[completed[0]]
        else:
            del BATCH_STORE[next(iter(BATCH_STORE))]
        gc.collect()


def _process_batch_item(batch: BatchJob, file_path: Path, original_filename: str, duplicate_policy: str = "prompt"):
    """Processes 1 resume inside the ThreadPoolExecutor and thread-safely updates batch."""
    try:
        logger.info(f"⏳ [BATCH {batch.batch_id}] Processing: {original_filename} (duplicate_policy={duplicate_policy})")
        parsed_record = process_resume(str(file_path), verbose=False)
        parsed_record["file_name"] = original_filename
        parsed_record["source_file"] = original_filename
        parsed_record["batch_status"] = "completed"

        # Read file bytes for Qdrant sync before unlinking
        file_bytes = None
        try:
            with open(file_path, "rb") as fp:
                file_bytes = fp.read()
        except Exception:
            pass

        # Candidate Conflict Detection & Resolution Strategy
        try:
            existing_id, old_record, diff = check_candidate_conflict(parsed_record)

            if existing_id is None:
                # 1. Completely new candidate: index directly!
                point_id = execute_candidate_resolution("keep_both", parsed_record, original_filename, file_bytes=file_bytes)
                parsed_record["id"] = point_id
                parsed_record["conflict"] = {
                    "is_conflict": False,
                    "status": "new_candidate",
                    "point_id": point_id
                }
                logger.info(f"✅ [NEW-CANDIDATE] Created {original_filename} -> Point ID: {point_id}")

            elif diff and diff.get("is_identical"):
                # 2. 100% Identical resume: skip duplicate insert
                parsed_record["id"] = existing_id
                parsed_record["conflict"] = {
                    "is_conflict": False,
                    "status": "identical",
                    "existing_point_id": existing_id,
                    "point_id": existing_id
                }
                logger.info(f"✓ [IDENTICAL] Skipped duplicate {original_filename} -> matches Point #{existing_id}")

            else:
                # 3. Existing profile matched with differences!
                if duplicate_policy == "prompt":
                    conflict_id = f"conf_{uuid.uuid4().hex[:10]}"
                    parsed_record["id"] = existing_id
                    PENDING_CONFLICTS[conflict_id] = {
                        "conflict_id": conflict_id,
                        "parsed_record": parsed_record,
                        "filename": original_filename,
                        "file_bytes": file_bytes,
                        "existing_point_id": existing_id,
                        "existing_candidate": {
                            "id": existing_id,
                            "name": old_record.get("name"),
                            "email": old_record.get("email"),
                            "mobile_number": old_record.get("mobile_number"),
                            "total_experience": old_record.get("total_experience"),
                            "skills": old_record.get("skills", []),
                            "file_name": old_record.get("file_name"),
                        },
                        "diff": diff
                    }
                    parsed_record["conflict"] = {
                        "is_conflict": True,
                        "conflict_id": conflict_id,
                        "existing_point_id": existing_id,
                        "existing_candidate": PENDING_CONFLICTS[conflict_id]["existing_candidate"],
                        "diff": diff,
                        "status": "pending_resolution"
                    }
                    logger.info(f"⚠️ [CONFLICT-DETECTED] Candidate {original_filename} matches Point #{existing_id}. Awaiting user resolution.")

                elif duplicate_policy in ("override", "keep_both", "keep_existing"):
                    point_id = execute_candidate_resolution(
                        duplicate_policy,
                        parsed_record,
                        original_filename,
                        existing_point_id=existing_id,
                        file_bytes=file_bytes
                    )
                    parsed_record["id"] = point_id
                    parsed_record["conflict"] = {
                        "is_conflict": False,
                        "status": "auto_resolved",
                        "action": duplicate_policy,
                        "point_id": point_id
                    }
                    logger.info(f"✅ [AUTO-RESOLVED-{duplicate_policy.upper()}] {original_filename} -> Point ID: {point_id}")

        except Exception as q_err:
            logger.warning(f"⚠️ [CONFLICT-CHECK] Failed during identity check for {original_filename}: {q_err}", exc_info=True)
            parsed_record["conflict"] = {"is_conflict": False, "status": "error", "error": str(q_err)}

        with batch.lock:
            batch.candidates.append(parsed_record)
            batch.completed += 1
            if (batch.completed + batch.failed) >= batch.total:
                batch.status = "completed"
                batch.completed_at = datetime.now(timezone.utc).isoformat()
                logger.info(f"✅ [BATCH {batch.batch_id}] All {batch.total} resumes processed successfully!")
    except Exception as e:
        logger.error(f"❌ [BATCH {batch.batch_id}] Failed to parse {original_filename}: {e}")
        with batch.lock:
            batch.candidates.append({
                "id": uuid.uuid4().hex[:8],
                "name": "Failed to parse",
                "source_file": original_filename,
                "batch_status": "failed",
                "error": str(e)
            })
            batch.failed += 1
            if (batch.completed + batch.failed) >= batch.total:
                batch.status = "completed"
                batch.completed_at = datetime.now(timezone.utc).isoformat()
    finally:
        try:
            file_path.unlink(missing_ok=True)
            intermediate_flat = FLAT_JSON_DIR / f"1_{file_path.stem}_flat.json"
            intermediate_flat.unlink(missing_ok=True)
        except Exception:
            pass
        gc.collect()



# ── Lifespan Startup Manager: Pre-load Models into RAM ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server startup & shutdown handler. Pre-loads heavy PyTorch models
    (Surya OCR) into system RAM once on boot -- unless OCR_ENGINE=tesseract,
    in which case no model is needed and the preload is skipped.
    """
    logger.info(f"🚀 Starting ATS Resume Parser FastAPI Server... (OCR_ENGINE={OCR_ENGINE})")
    if OCR_ENGINE == "tesseract":
        logger.info("[SERVER] OCR_ENGINE=tesseract: skipping Surya model preload.")
    else:
        try:
            from ingestion.ocr_reader import _load_surya_predictors
            logger.info("[SERVER] Pre-loading Surya OCR models into RAM (first time load)...")
            _load_surya_predictors()
            logger.info("✅ [SERVER] Surya OCR models pre-loaded in memory. Zero cold start active.")
        except Exception as e:
            logger.warning(f"⚠️ [SERVER] Could not pre-load Surya OCR models on startup: {e}")

    yield  # Server runs and handles requests here

    logger.info("🛑 Shutting down ATS Resume Parser FastAPI Server...")


app = FastAPI(
    title="ATS Resume Parser Microservice",
    description="Enterprise ATS Resume Parser API with Persistent Model Caching & Async OCR Queue",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for CRM frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper Utilities ─────────────────────────────────────────────────────────
def _is_scanned_pdf(file_path: str) -> bool:
    """
    Fast helper to check if a PDF is an image-based scanned file.
    If cumulative native text across pages <= 100 chars, it requires OCR.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        return True

    if ext != ".pdf":
        return False

    try:
        doc = fitz.open(file_path)
        total_text = ""
        for page in doc:
            total_text += page.get_text()
            if len(total_text.strip()) > 100:
                doc.close()
                return False
        doc.close()
        return len(total_text.strip()) <= 100
    except Exception:
        return False


import threading

_ocr_lock = threading.Lock()


def _run_background_parsing(job_id: str, file_path: str):
    """Executes heavy background OCR parsing sequentially (one job at a time) to prevent RAM spikes."""
    with _ocr_lock:
        logger.info(f"⏳ [ASYNC JOB {job_id}] Starting background OCR parsing for: {file_path}")
        try:
            result = process_resume(file_path, verbose=False)
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = result
            JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"✅ [ASYNC JOB {job_id}] Background OCR completed successfully!")

            # Auto-sync parsed candidate to shared Qdrant Vector DB
            try:
                raw_bytes = None
                p_file = Path(file_path)
                if p_file.exists():
                    raw_bytes = p_file.read_bytes()
                orig_name = JOBS[job_id].get("source_file") or p_file.name
                result["file_name"] = orig_name
                result["source_file"] = orig_name
                sync_resume_to_qdrant(result, orig_name, raw_bytes)
            except Exception as qe:
                logger.warning(f"Failed to sync async resume to Qdrant: {qe}")
        except Exception as e:
            logger.error(f"❌ [ASYNC JOB {job_id}] Background OCR failed: {e}")
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        finally:
            # Auto-delete temporary upload file from disk after background task completes
            try:
                p_f = Path(file_path)
                p_f.unlink(missing_ok=True)
                intermediate_flat = FLAT_JSON_DIR / f"1_{p_f.stem}_flat.json"
                intermediate_flat.unlink(missing_ok=True)
            except Exception:
                pass
            gc.collect()


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint confirming server operational status."""
    surya_cached = False
    try:
        from ingestion.ocr_reader import _surya_predictors
        surya_cached = _surya_predictors is not None
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "ATS Resume Parser Microservice",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ocr_engine": OCR_ENGINE,
        "models_cached_in_ram": {
            "surya_ocr": surya_cached,
        },
        "active_background_jobs": sum(1 for j in JOBS.values() if j["status"] == "processing"),
        "total_jobs_in_ram": len(JOBS),
        "active_batches": sum(1 for b in BATCH_STORE.values() if b.status == "processing"),
        "total_batches_in_ram": len(BATCH_STORE),
    }


@app.post("/api/v1/parse-resume", tags=["Parsing"])
async def parse_resume_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    duplicate_policy: str = Query("prompt", description="Duplicate conflict resolution policy: 'prompt', 'override', 'keep_both', or 'keep_existing'.")
):
    """
    Parses a single uploaded resume file (.pdf, .docx, .doc, .png, .jpg).

    - **Digital PDFs / DOCXs**: Returned synchronously (< 0.1s).
    - **Scanned / Image PDFs**: Accepted immediately (HTTP 202), processed in
      background task so the CRM user interface never freezes.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_id = str(uuid.uuid4())[:8]
    sanitized_filename = f"{file_id}_{Path(file.filename).name}"
    save_path = UPLOAD_TEMP_DIR / sanitized_filename

    # Save uploaded file
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    is_scanned = _is_scanned_pdf(str(save_path))

    if not is_scanned:
        # Fast-Path: Digital text document -> process synchronously (< 0.1s)
        try:
            result = process_resume(str(save_path), verbose=False)
            result["file_name"] = file.filename
            result["source_file"] = file.filename

            # Candidate Conflict Detection & Resolution Strategy
            try:
                existing_id, old_record, diff = check_candidate_conflict(result)

                if existing_id is None:
                    point_id = execute_candidate_resolution("keep_both", result, file.filename, file_bytes=contents)
                    result["id"] = point_id
                    result["conflict"] = {"is_conflict": False, "status": "new_candidate", "point_id": point_id}
                elif diff and diff.get("is_identical"):
                    result["id"] = existing_id
                    result["conflict"] = {"is_conflict": False, "status": "identical", "existing_point_id": existing_id, "point_id": existing_id}
                else:
                    if duplicate_policy == "prompt":
                        conflict_id = f"conf_{uuid.uuid4().hex[:10]}"
                        result["id"] = existing_id
                        PENDING_CONFLICTS[conflict_id] = {
                            "conflict_id": conflict_id,
                            "parsed_record": result,
                            "filename": file.filename,
                            "file_bytes": contents,
                            "existing_point_id": existing_id,
                            "existing_candidate": {
                                "id": existing_id,
                                "name": old_record.get("name"),
                                "email": old_record.get("email"),
                                "mobile_number": old_record.get("mobile_number"),
                                "total_experience": old_record.get("total_experience"),
                                "skills": old_record.get("skills", []),
                                "file_name": old_record.get("file_name"),
                            },
                            "diff": diff
                        }
                        result["conflict"] = {
                            "is_conflict": True,
                            "conflict_id": conflict_id,
                            "existing_point_id": existing_id,
                            "existing_candidate": PENDING_CONFLICTS[conflict_id]["existing_candidate"],
                            "diff": diff,
                            "status": "pending_resolution"
                        }
                    else:
                        point_id = execute_candidate_resolution(duplicate_policy, result, file.filename, existing_point_id=existing_id, file_bytes=contents)
                        result["id"] = point_id
                        result["conflict"] = {"is_conflict": False, "status": "auto_resolved", "action": duplicate_policy, "point_id": point_id}
            except Exception as q_err:
                logger.warning(f"Error checking conflict for {file.filename}: {q_err}")
                result["conflict"] = {"is_conflict": False, "status": "error", "error": str(q_err)}

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "completed",
                    "execution_mode": "fast_path_digital",
                    "source_file": file.filename,
                    "parsed_data": result,
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")
        finally:
            # Auto-delete temporary upload file after fast-path parsing completes
            save_path.unlink(missing_ok=True)
            intermediate_flat = FLAT_JSON_DIR / f"1_{save_path.stem}_flat.json"
            intermediate_flat.unlink(missing_ok=True)
            gc.collect()

    else:
        # Async-Path: Scanned/Image document -> queue in background task
        _purge_oldest_jobs_if_needed()
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "source_file": file.filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None,
        }

        background_tasks.add_task(_run_background_parsing, job_id, str(save_path))

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "processing",
                "execution_mode": "async_background_ocr",
                "job_id": job_id,
                "message": "Scanned document queued for background OCR. CRM frontend will not freeze.",
                "status_url": f"/api/v1/job/{job_id}",
                "source_file": file.filename,
            }
        )


@app.get("/api/v1/job/{job_id}", tags=["Parsing"])
def get_job_status(job_id: str):
    """Returns the status and output of a background OCR job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")

    job_info = JOBS[job_id]
    return {
        "job_id": job_info["job_id"],
        "status": job_info["status"],
        "source_file": job_info["source_file"],
        "created_at": job_info["created_at"],
        "completed_at": job_info.get("completed_at"),
        "result": job_info.get("result"),
        "error": job_info.get("error"),
    }


@app.post("/api/v1/parse-batch", tags=["Batch Parsing"])
async def parse_batch_endpoint(
    files: List[UploadFile] = File(...),
    wait: bool = Query(True, description="If true (default), waits for all resumes to be parsed and returns the candidate JSONs directly in this response. If false, returns 202 Accepted with batch_id for async polling."),
    duplicate_policy: str = Query("prompt", description="Duplicate conflict resolution policy: 'prompt', 'override', 'keep_both', or 'keep_existing'.")
):
    """
    Accepts 100-200+ resume files in a single API call.
    Files are processed concurrently by a bounded 4-worker pool.
    
    - Default (wait=True): Returns HTTP 200 OK directly with all parsed candidate JSONs.
    - Async mode (wait=False): Returns HTTP 202 Accepted immediately with batch_id for background polling.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    _purge_oldest_batches_if_needed()

    batch_id = f"batch_{uuid.uuid4().hex[:10]}"
    batch_dir = UPLOAD_TEMP_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch = BatchJob(batch_id=batch_id, total_files=len(files))
    BATCH_STORE[batch_id] = batch

    logger.info(f"🚀 [BATCH {batch_id}] Received {len(files)} files for bulk processing (wait={wait}, duplicate_policy={duplicate_policy}).")

    futures = []
    for file in files:
        sanitized_filename = f"{uuid.uuid4().hex[:6]}_{Path(file.filename).name}"
        save_path = batch_dir / sanitized_filename

        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        fut = BATCH_EXECUTOR.submit(_process_batch_item, batch, save_path, file.filename, duplicate_policy)
        futures.append(fut)

    if wait:
        # Await completion of all worker threads without blocking FastAPI's async event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, concurrent.futures.wait, futures)

        # Cleanup batch directory if empty
        try:
            batch_dir.rmdir()
        except Exception:
            pass

        with batch.lock:
            processed = batch.completed + batch.failed
            pct = round((processed / batch.total) * 100, 1) if batch.total > 0 else 100.0
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "batch_id": batch.batch_id,
                    "status": batch.status,
                    "total": batch.total,
                    "completed": batch.completed,
                    "failed": batch.failed,
                    "percentage": pct,
                    "created_at": batch.created_at,
                    "completed_at": batch.completed_at,
                    "candidates": list(batch.candidates),
                }
            )

    # Async polling mode (wait=False)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "processing",
            "batch_id": batch_id,
            "total_files": len(files),
            "status_url": f"/api/v1/batch/{batch_id}",
            "message": f"{len(files)} resumes queued for background processing with 4 workers.",
        }
    )


@app.post("/api/v1/resolve-conflict", tags=["Conflict Resolution"])
def resolve_conflict_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Resolves duplicate candidate conflicts according to user choice:
    - keep_existing: Discards new resume, preserves existing DB record.
    - keep_both: Saves new resume under a brand-new Point ID (both coexist).
    - override: Overwrites existing record with new resume data & vector.

    Accepts:
    {
       "conflict_id": "conf_xyz",
       "action": "keep_existing" | "keep_both" | "override"
    }
    OR bulk format:
    {
       "bulk_action": "override" | "keep_both" | "keep_existing",
       "conflict_ids": ["conf_1", "conf_2", ...]
    }
    OR list:
    {
       "resolutions": [
          {"conflict_id": "conf_1", "action": "override"},
          {"conflict_id": "conf_2", "action": "keep_both"}
       ]
    }
    """
    resolutions = []
    bulk_action = payload.get("bulk_action")
    conflict_ids = payload.get("conflict_ids")

    if bulk_action and conflict_ids:
        for cid in conflict_ids:
            resolutions.append({"conflict_id": cid, "action": bulk_action})
    elif "resolutions" in payload:
        resolutions = payload.get("resolutions") or []
    elif "conflict_id" in payload and "action" in payload:
        resolutions.append(payload)

    results = []
    for item in resolutions:
        cid = item.get("conflict_id")
        action = item.get("action", "keep_existing")

        c_data = PENDING_CONFLICTS.pop(cid, None)
        if c_data:
            parsed_record = c_data["parsed_record"]
            filename = c_data["filename"]
            file_bytes = c_data.get("file_bytes")
            existing_point_id = c_data.get("existing_point_id")
        else:
            parsed_record = item.get("parsed_record")
            filename = item.get("filename") or (parsed_record.get("file_name") if parsed_record else "resume.pdf")
            file_bytes = None
            existing_point_id = item.get("existing_point_id")

        if not parsed_record:
            continue

        point_id = execute_candidate_resolution(
            action=action,
            parsed_record=parsed_record,
            filename=filename,
            existing_point_id=existing_point_id,
            file_bytes=file_bytes
        )
        results.append({
            "conflict_id": cid,
            "action": action,
            "point_id": point_id,
            "filename": filename,
            "status": "resolved"
        })

    logger.info(f"✅ [CONFLICT-RESOLVED] Processed {len(results)} candidate conflict resolutions.")
    return {
        "status": "success",
        "resolved_count": len(results),
        "results": results
    }


@app.get("/api/v1/batch/{batch_id}", tags=["Batch Parsing"])
def get_batch_status(batch_id: str):
    """Returns the live progress and cumulative candidate results of a batch."""
    if batch_id not in BATCH_STORE:
        raise HTTPException(status_code=404, detail="Batch ID not found")

    batch = BATCH_STORE[batch_id]
    with batch.lock:
        processed = batch.completed + batch.failed
        pct = round((processed / batch.total) * 100, 1) if batch.total > 0 else 100.0
        return {
            "batch_id": batch.batch_id,
            "status": batch.status,
            "total": batch.total,
            "completed": batch.completed,
            "failed": batch.failed,
            "percentage": pct,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "candidates": list(batch.candidates),
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)







##commenting this to see how tesseract will works instead of surya ocr
#"""
#server.py
#=========
#Production FastAPI Server for ATS Resume Parser Engine
#
#Key Capabilities:
#  1. Lifespan Startup: Pre-loads Surya OCR predictors into RAM ONCE at boot time
#     (eliminates 30-45s cold starts permanently).
#  2. Fast-Path Execution: Digital PDFs/DOCXs process synchronously (< 0.1s).
#  3. Asynchronous Background Execution: Scanned/Image PDFs return HTTP 202
#     Accepted immediately + job_id, processing heavy OCR in background tasks so
#     the CRM frontend never freezes or times out.
#  4. Batch Upload Support: Accepts multi-file list uploads (100+ files) in one API call.
#
#Run Server:
#    ./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
#"""
#
#import os
## Cap PyTorch & OpenMP CPU threads to 2 to prevent RAM/CPU spikes that cause VSCode/OS OOM killer to close applications
##os.environ["OMP_NUM_THREADS"] = "2"
##os.environ["MKL_NUM_THREADS"] = "2"
##os.environ["OPENBLAS_NUM_THREADS"] = "2"
##try:
##    import torch
##    torch.set_num_threads(2)
##except Exception:
##    pass
#
#
#os.environ["OMP_NUM_THREADS"] = "6"
#os.environ["MKL_NUM_THREADS"] = "6"
#os.environ["OPENBLAS_NUM_THREADS"] = "6"
#try:
#    import torch
#    torch.set_num_threads(6)
#except Exception:
#    pass
#
#import uuid
#import logging
#from typing import List, Optional
#from pathlib import Path
#from datetime import datetime, timezone
#from contextlib import asynccontextmanager
#
#import fitz  # PyMuPDF
#from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query, status
#from fastapi.middleware.cors import CORSMiddleware
#from fastapi.responses import JSONResponse
#
#from run_pipeline import process_resume, EXTRACTED_TEXT_DIR, SEGMENTED_DIR, FLAT_JSON_DIR
#
#
#logger = logging.getLogger("ats_server")
#logging.basicConfig(level=logging.INFO)
#
#import gc
#
## Temporary directory for API file uploads (isolated from resumes/ benchmark directory)
#UPLOAD_TEMP_DIR = Path("tmp_uploads")
#UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
#
## In-memory store for async background OCR jobs with Bounded LRU Storage Cap
#MAX_JOBS = 50
#JOBS = {}
#
#
#def _purge_oldest_jobs_if_needed():
#    """Keeps the in-memory JOBS storage bounded under MAX_JOBS to prevent RAM leaks."""
#    if len(JOBS) >= MAX_JOBS:
#        # Purge oldest completed or failed jobs first
#        completed = [k for k, v in JOBS.items() if v.get("status") in ("completed", "failed")]
#        if completed:
#            del JOBS[completed[0]]
#        else:
#            del JOBS[next(iter(JOBS))]
#        gc.collect()
#
#
## ── Lifespan Startup Manager: Pre-load Models into RAM ─────────────────────────
#@asynccontextmanager
#async def lifespan(app: FastAPI):
#    """
#    Server startup & shutdown handler. Pre-loads heavy PyTorch models
#    (Surya OCR, Skills Taxonomy) into system RAM once on boot.
#    """
#    logger.info("🚀 Starting ATS Resume Parser FastAPI Server...")
#    try:
#        from ingestion.ocr_reader import _load_surya_predictors
#        logger.info("[SERVER] Pre-loading Surya OCR models into RAM (first time load)...")
#        _load_surya_predictors()
#        logger.info("✅ [SERVER] Surya OCR models pre-loaded in memory. Zero cold start active.")
#    except Exception as e:
#        logger.warning(f"⚠️ [SERVER] Could not pre-load Surya OCR models on startup: {e}")
#    
#    yield  # Server runs and handles requests here
#    
#    logger.info("🛑 Shutting down ATS Resume Parser FastAPI Server...")
#
#
#app = FastAPI(
#    title="ATS Resume Parser Microservice",
#    description="Enterprise ATS Resume Parser API with Persistent Model Caching & Async OCR Queue",
#    version="2.0.0",
#    lifespan=lifespan,
#)
#
## Enable CORS for CRM frontend integration
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)
#
#
## ── Helper Utilities ─────────────────────────────────────────────────────────
#def _is_scanned_pdf(file_path: str) -> bool:
#    """
#    Fast helper to check if a PDF is an image-based scanned file.
#    If cumulative native text across pages <= 100 chars, it requires OCR.
#    """
#    path = Path(file_path)
#    ext = path.suffix.lower()
#    
#    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
#        return True
#        
#    if ext != ".pdf":
#        return False
#
#    try:
#        doc = fitz.open(file_path)
#        total_text = ""
#        for page in doc:
#            total_text += page.get_text()
#            if len(total_text.strip()) > 100:
#                doc.close()
#                return False
#        doc.close()
#        return len(total_text.strip()) <= 100
#    except Exception:
#        return False
#
#
#import threading
#
#_ocr_lock = threading.Lock()
#
#
#def _run_background_parsing(job_id: str, file_path: str):
#    """Executes heavy background OCR parsing sequentially (one job at a time) to prevent RAM spikes."""
#    with _ocr_lock:
#        logger.info(f"⏳ [ASYNC JOB {job_id}] Starting background OCR parsing for: {file_path}")
#        try:
#            result = process_resume(file_path, verbose=False)
#            JOBS[job_id]["status"] = "completed"
#            JOBS[job_id]["result"] = result
#            JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
#            logger.info(f"✅ [ASYNC JOB {job_id}] Background OCR completed successfully!")
#        except Exception as e:
#            logger.error(f"❌ [ASYNC JOB {job_id}] Background OCR failed: {e}")
#            JOBS[job_id]["status"] = "failed"
#            JOBS[job_id]["error"] = str(e)
#            JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
#        finally:
#            # Auto-delete temporary upload file from disk after background task completes
#            try:
#                Path(file_path).unlink(missing_ok=True)
#            except Exception:
#                pass
#            gc.collect()
#
#
## ── API Endpoints ────────────────────────────────────────────────────────────
#
#@app.get("/health", tags=["Health"])
#def health_check():
#    """Health check endpoint confirming server operational status."""
#    surya_cached = False
#    try:
#        from ingestion.ocr_reader import _surya_predictors
#        surya_cached = _surya_predictors is not None
#    except Exception:
#        pass
#
#    return {
#        "status": "healthy",
#        "service": "ATS Resume Parser Microservice",
#        "timestamp": datetime.now(timezone.utc).isoformat(),
#        "models_cached_in_ram": {
#            "surya_ocr": surya_cached,
#        },
#        "active_background_jobs": sum(1 for j in JOBS.values() if j["status"] == "processing"),
#        "total_jobs_in_ram": len(JOBS),
#    }
#
#
#@app.post("/api/v1/parse-resume", tags=["Parsing"])
#async def parse_resume_endpoint(
#    background_tasks: BackgroundTasks,
#    file: UploadFile = File(...),
#):
#    """
#    Parses a single uploaded resume file (.pdf, .docx, .doc, .png, .jpg).
#    
#    - **Digital PDFs / DOCXs**: Returned synchronously (< 0.1s).
#    - **Scanned / Image PDFs**: Accepted immediately (HTTP 202), processed in
#      background task so the CRM user interface never freezes.
#    """
#    if not file.filename:
#        raise HTTPException(status_code=400, detail="No file provided")
#
#    file_id = str(uuid.uuid4())[:8]
#    sanitized_filename = f"{file_id}_{Path(file.filename).name}"
#    save_path = UPLOAD_TEMP_DIR / sanitized_filename
#
#    # Save uploaded file
#    contents = await file.read()
#    with open(save_path, "wb") as f:
#        f.write(contents)
#
#    is_scanned = _is_scanned_pdf(str(save_path))
#
#    if not is_scanned:
#        # Fast-Path: Digital text document -> process synchronously (< 0.1s)
#        try:
#            result = process_resume(str(save_path), verbose=False)
#            return JSONResponse(
#                status_code=status.HTTP_200_OK,
#                content={
#                    "status": "completed",
#                    "execution_mode": "fast_path_digital",
#                    "source_file": file.filename,
#                    "parsed_data": result,
#                }
#            )
#        except Exception as e:
#            raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")
#        finally:
#            # Auto-delete temporary upload file after fast-path parsing completes
#            save_path.unlink(missing_ok=True)
#            gc.collect()
#
#    else:
#        # Async-Path: Scanned/Image document -> queue in background task
#        _purge_oldest_jobs_if_needed()
#        job_id = f"job_{uuid.uuid4().hex[:12]}"
#        JOBS[job_id] = {
#            "job_id": job_id,
#            "status": "processing",
#            "source_file": file.filename,
#            "created_at": datetime.now(timezone.utc).isoformat(),
#            "result": None,
#            "error": None,
#        }
#
#        background_tasks.add_task(_run_background_parsing, job_id, str(save_path))
#
#        return JSONResponse(
#            status_code=status.HTTP_202_ACCEPTED,
#            content={
#                "status": "processing",
#                "execution_mode": "async_background_ocr",
#                "job_id": job_id,
#                "message": "Scanned document queued for background OCR. CRM frontend will not freeze.",
#                "status_url": f"/api/v1/job/{job_id}",
#                "source_file": file.filename,
#            }
#        )
#
#
#@app.get("/api/v1/job/{job_id}", tags=["Parsing"])
#def get_job_status(job_id: str):
#    """Returns the status and output of a background OCR job."""
#    if job_id not in JOBS:
#        raise HTTPException(status_code=404, detail="Job ID not found")
#        
#    job_info = JOBS[job_id]
#    return {
#        "job_id": job_info["job_id"],
#        "status": job_info["status"],
#        "source_file": job_info["source_file"],
#        "created_at": job_info["created_at"],
#        "completed_at": job_info.get("completed_at"),
#        "result": job_info.get("result"),
#        "error": job_info.get("error"),
#    }
#
#
#@app.post("/api/v1/parse-batch", tags=["Batch Parsing"])
#async def parse_batch_endpoint(
#    background_tasks: BackgroundTasks,
#    files: List[UploadFile] = File(...),
#):
#    """
#    Accepts multiple resume files (e.g. 100+ files) in a single API call.
#    Digital files return results immediately; scanned files are queued.
#    """
#    if not files:
#        raise HTTPException(status_code=400, detail="No files provided")
#
#    summary_results = []
#    digital_count = 0
#    scanned_count = 0
#
#    for file in files:
#        file_id = str(uuid.uuid4())[:8]
#        sanitized_filename = f"{file_id}_{Path(file.filename).name}"
#        save_path = UPLOAD_TEMP_DIR / sanitized_filename
#
#        contents = await file.read()
#        with open(save_path, "wb") as f:
#            f.write(contents)
#
#        is_scanned = _is_scanned_pdf(str(save_path))
#
#        if not is_scanned:
#            try:
#                res = process_resume(str(save_path), verbose=False)
#                digital_count += 1
#                summary_results.append({
#                    "file": file.filename,
#                    "mode": "digital_immediate",
#                    "status": "completed",
#                    "data": res,
#                })
#            except Exception as e:
#                summary_results.append({
#                    "file": file.filename,
#                    "mode": "digital_immediate",
#                    "status": "failed",
#                    "error": str(e),
#                })
#        else:
#            job_id = f"job_{uuid.uuid4().hex[:12]}"
#            JOBS[job_id] = {
#                "job_id": job_id,
#                "status": "processing",
#                "source_file": file.filename,
#                "created_at": datetime.now(timezone.utc).isoformat(),
#                "result": None,
#                "error": None,
#            }
#            background_tasks.add_task(_run_background_parsing, job_id, str(save_path))
#            scanned_count += 1
#            summary_results.append({
#                "file": file.filename,
#                "mode": "async_background_ocr",
#                "status": "processing",
#                "job_id": job_id,
#            })
#
#    return {
#        "total_files": len(files),
#        "digital_immediate_completed": digital_count,
#        "scanned_background_queued": scanned_count,
#        "results": summary_results,
#    }
#
#
#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
#