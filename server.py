"""
server.py
=========
Production FastAPI Server for ATS Resume Parser Engine

Key Capabilities:
  1. Lifespan Startup: Pre-loads Surya OCR predictors into RAM ONCE at boot time
     (eliminates 30-45s cold starts permanently).
  2. Fast-Path Execution: Digital PDFs/DOCXs process synchronously (< 0.1s).
  3. Asynchronous Background Execution: Scanned/Image PDFs return HTTP 202
     Accepted immediately + job_id, processing heavy OCR in background tasks so
     the CRM frontend never freezes or times out.
  4. Batch Upload Support: Accepts multi-file list uploads (100+ files) in one API call.

Run Server:
    ./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
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
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from run_pipeline import process_resume, EXTRACTED_TEXT_DIR, SEGMENTED_DIR, FLAT_JSON_DIR


logger = logging.getLogger("ats_server")
logging.basicConfig(level=logging.INFO)

import gc

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


# ── Lifespan Startup Manager: Pre-load Models into RAM ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server startup & shutdown handler. Pre-loads heavy PyTorch models
    (Surya OCR, Skills Taxonomy) into system RAM once on boot.
    """
    logger.info("🚀 Starting ATS Resume Parser FastAPI Server...")
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
        except Exception as e:
            logger.error(f"❌ [ASYNC JOB {job_id}] Background OCR failed: {e}")
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        finally:
            # Auto-delete temporary upload file from disk after background task completes
            try:
                Path(file_path).unlink(missing_ok=True)
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
        "models_cached_in_ram": {
            "surya_ocr": surya_cached,
        },
        "active_background_jobs": sum(1 for j in JOBS.values() if j["status"] == "processing"),
        "total_jobs_in_ram": len(JOBS),
    }


@app.post("/api/v1/parse-resume", tags=["Parsing"])
async def parse_resume_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
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
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Accepts multiple resume files (e.g. 100+ files) in a single API call.
    Digital files return results immediately; scanned files are queued.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    summary_results = []
    digital_count = 0
    scanned_count = 0

    for file in files:
        file_id = str(uuid.uuid4())[:8]
        sanitized_filename = f"{file_id}_{Path(file.filename).name}"
        save_path = UPLOAD_TEMP_DIR / sanitized_filename

        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        is_scanned = _is_scanned_pdf(str(save_path))

        if not is_scanned:
            try:
                res = process_resume(str(save_path), verbose=False)
                digital_count += 1
                summary_results.append({
                    "file": file.filename,
                    "mode": "digital_immediate",
                    "status": "completed",
                    "data": res,
                })
            except Exception as e:
                summary_results.append({
                    "file": file.filename,
                    "mode": "digital_immediate",
                    "status": "failed",
                    "error": str(e),
                })
        else:
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
            scanned_count += 1
            summary_results.append({
                "file": file.filename,
                "mode": "async_background_ocr",
                "status": "processing",
                "job_id": job_id,
            })

    return {
        "total_files": len(files),
        "digital_immediate_completed": digital_count,
        "scanned_background_queued": scanned_count,
        "results": summary_results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
