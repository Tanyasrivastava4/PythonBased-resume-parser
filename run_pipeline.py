"""
run_pipeline.py
===============
100% Pure Python-based ATS Resume Parser Pipeline.
Zero LLM dependencies, sub-second execution on CPU.

Executes all pipeline layers:
  Layer 0: Ingestion (read PDF/DOCX/DOC/Image/TXT + Tesseract OCR for scanned files)
  Layer 1: Section Segmentation
  Layer 2 & 3: Extraction & Assembly (Contacts, Education, Experience, Skills, Certs, Achievements, Projects)
  Layer 4: Schema Flattening (final display JSON output with sequential counting IDs)

CLI Usage:
    # Process a single resume with ID 1:
    python run_pipeline.py "path/to/resume.pdf" --id 1

    # Process all resumes in the OneDrive folder:
    python run_pipeline.py --all

    # Process first 10 resumes for testing:
    python run_pipeline.py --all --limit 10
"""

import os
import sys
import gc
import glob
import time
import json
import hashlib
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure current directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestion import read_resume_file
from segmenter import split_into_sections
from layer3.flatten import flatten_resume_record
from candidate_diff import consolidate_duplicate_candidates


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESUME_DIR = BASE_DIR / "resumes"
EXTRACTED_TEXT_DIR = BASE_DIR / "extracted_text"
SEGMENTED_DIR = BASE_DIR / "segmented_text"
FLAT_JSON_DIR = BASE_DIR / "flat_json"


def compute_file_md5(file_path: Path) -> str:
    """Computes MD5 hash of a file efficiently using 64KB chunks."""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def deduplicate_files_by_md5(files: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Identifies exact byte-for-byte duplicate resume files using MD5 checksum.
    Prefers cleaner file names (avoiding '(1)', '-1', 'copy').
    Returns (unique_file_paths, list_of_(duplicate_name, original_name)).
    """
    seen_hashes = {}
    duplicates = []

    def _is_copy_name(name: str) -> bool:
        lower = name.lower()
        return "(1)" in lower or "(2)" in lower or "-1." in lower or "copy" in lower

    for f_str in files:
        p = Path(f_str)
        h = compute_file_md5(p)
        if not h:
            seen_hashes[f_str] = p
            continue

        if h in seen_hashes:
            existing_p = seen_hashes[h]
            # If current file has cleaner name than the existing one, prefer current
            if _is_copy_name(existing_p.name) and not _is_copy_name(p.name):
                duplicates.append((existing_p.name, p.name))
                seen_hashes[h] = p
            else:
                duplicates.append((p.name, existing_p.name))
        else:
            seen_hashes[h] = p

    unique_files = [str(p) for p in seen_hashes.values()]
    return unique_files, duplicates


def _save_raw_text(file_path: str, text: str, output_dir: Path, count_id: str) -> Path:
    """Saves Layer 0 extracted raw text to extracted_text/ folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    out_path = output_dir / f"{count_id}_{stem}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def _save_segmented_text(file_path: str, sections: list, output_dir: Path, count_id: str) -> Path:
    """Saves Layer 1 segmented section output to segmented_text/ folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    out_path = output_dir / f"{count_id}_{stem}_segmented.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for section in sections:
            f.write(f"\n{'─' * 60}\n")
            f.write(
                f"LABEL: {section.label}   "
                f"CONFIDENCE: {round(section.confidence, 2)}   "
                f"START LINE: {section.start_line}\n"
            )
            f.write(f"{'─' * 60}\n")
            f.write(section.raw_text)
            f.write("\n")

        labels_found = [s.label for s in sections]
        f.write(f"\n{'─' * 60}\n")
        f.write(f"[Found {len(sections)} sections total: {labels_found}]\n")

    return out_path


def process_resume(
    file_path: str,
    resume_id: int | str = 1,
    output_dir: str = "flat_json",
    verbose: bool = True
) -> dict:
    """
    Processes a single resume file through the entire ATS parser pipeline.
    Assigns a sequential counting number ID (e.g. 1, 2, 3...).
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    count_id = str(resume_id)

    if verbose:
        print(f"\n{'═' * 65}")
        print(f"PROCESSING RESUME [ID: {count_id}] : {input_path.name}")
        print(f"{'═' * 65}")

    t_start = time.perf_counter()

    # Layer 0: Ingestion (Text extraction + Tesseract OCR if scanned)
    if input_path.suffix.lower() == ".txt":
        raw_text = input_path.read_text(encoding="utf-8")
    else:
        raw_text = read_resume_file(str(input_path))

    t_ingest = time.perf_counter()
    ingest_duration = t_ingest - t_start

    extracted_txt_path = _save_raw_text(str(input_path), raw_text, EXTRACTED_TEXT_DIR, count_id)
    if verbose:
        print(f"✓ Layer 0 (Ingestion/OCR) : Extracted {len(raw_text):,} chars in {ingest_duration:.2f}s -> {extracted_txt_path}")

    # Layer 1: Section Segmentation
    t1 = time.perf_counter()
    sections = split_into_sections(raw_text)
    t_segment = time.perf_counter() - t1
    segmented_txt_path = _save_segmented_text(str(input_path), sections, SEGMENTED_DIR, count_id)
    if verbose:
        labels = [s.label for s in sections]
        print(f"✓ Layer 1 (Segmentation)  : Found {len(sections)} sections in {t_segment:.2f}s -> {segmented_txt_path}")

    # Layer 2, 3 & 4: Extraction, Assembly & Flattening
    t2 = time.perf_counter()
    out_dir_path = BASE_DIR / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    flat_record = flatten_resume_record(str(segmented_txt_path))
    t_extract = time.perf_counter() - t2

    # Insert counting number ID at top of record
    flat_record = {
        "id": int(count_id) if count_id.isdigit() else count_id,
        "file_name": input_path.name,
        "file_md5": compute_file_md5(input_path),
        **flat_record
    }

    t_total = time.perf_counter() - t_start
    flat_record["parsing_time_seconds"] = round(t_total, 2)

    flat_json_path = out_dir_path / f"{count_id}_{input_path.stem}_flat.json"

    # Automatically purge any older file for the same resume if the ID prefix shifted
    for old_f in out_dir_path.glob(f"*_{input_path.stem}_flat.json"):
        if old_f.name != flat_json_path.name:
            try:
                old_f.unlink()
            except Exception:
                pass

    with open(flat_json_path, "w", encoding="utf-8") as f:
        json.dump(flat_record, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✓ Layer 2-4 (Extract/Flat): Generated in {t_extract:.2f}s -> {flat_json_path}")
        print(f"⚡ TOTAL EXECUTION TIME   : {t_total:.2f} seconds")
        print(f"{'═' * 65}\n")

    return flat_record


def _worker_process_resume(item: tuple) -> tuple:
    """Helper worker for process_all execution: (index, file_path, output_dir)."""
    idx, file_path, output_dir = item
    try:
        t0 = time.perf_counter()
        process_resume(file_path, resume_id=idx, output_dir=output_dir, verbose=False)
        dur = time.perf_counter() - t0
        return (idx, file_path, True, dur, "")
    except Exception as e:
        return (idx, file_path, False, 0.0, str(e))


def process_all(
    input_dir: str = DEFAULT_RESUME_DIR,
    output_dir: str = "flat_json",
    limit: int = None,
    parallel: bool = True,
    workers: int = None
) -> list[dict]:
    """
    Processes all resume files in a folder through the pure Python ATS parser.
    Assigns sequential counting numbers (1, 2, 3...).
    """
    folder = Path(input_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    supported_extensions = ["*.pdf", "*.docx", "*.doc", "*.png", "*.jpg", "*.jpeg", "*.jfif", "*.txt"]
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(str(folder / ext)))
        files.extend(glob.glob(str(folder / ext.upper())))

    files = sorted(list(set(files)))
    total_raw_files = len(files)

    # Layer 0: File-Level MD5 Deduplication Check
    files, duplicates = deduplicate_files_by_md5(files)
    if duplicates:
        print(f"\nℹ️  [MD5 Deduplication] Filtered out {len(duplicates)} byte-for-byte duplicate resume(s) from {total_raw_files} files:")
        for dup, orig in duplicates[:6]:
            print(f"   • Skipped '{dup}' (identical copy of '{orig}')")
        if len(duplicates) > 6:
            print(f"   ... and {len(duplicates) - 6} more duplicate file copies skipped.")

    if limit and limit > 0:
        files = files[:limit]

    total_files = len(files)
    if not files:
        print(f"No supported resume files found in {input_dir}/")
        return []

    num_workers = workers or min(os.cpu_count() or 4, 8)
    work_items = [(idx, f, output_dir) for idx, f in enumerate(files, 1)]

    print(f"\n=======================================================")
    print(f" 🚀 PURE PYTHON RESUME EXTRACTION PIPELINE")
    print(f" Input Directory:    {input_dir}")
    print(f" Target Resumes:     {total_files} files")
    print(f" Sequential ID Mode: Counting Numbers (1 to {total_files})")
    print(f" Output Flat JSON:   {output_dir}/")
    print(f" Parallel CPU Cores: {num_workers}")
    print(f"=======================================================\n")

    start_time = time.time()
    success_count = 0
    fail_count = 0
    failures = []

    if parallel and num_workers > 1 and total_files > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_map = {
                executor.submit(_worker_process_resume, item): item
                for item in work_items
            }
            completed = 0
            for future in as_completed(future_map):
                completed += 1
                idx, f_path, success, dur, err = future.result()
                stem = Path(f_path).name
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_files - completed) / rate if rate > 0 else 0

                if success:
                    success_count += 1
                    print(f"✓ [{completed}/{total_files}] ID #{idx}: {stem} ({dur:.2f}s) | Remaining ~{remaining:.1f}s")
                else:
                    fail_count += 1
                    failures.append((idx, stem, err))
                    print(f"❌ [{completed}/{total_files}] ID #{idx}: {stem} FAILED: {err}")
    else:
        # Sequential execution
        for idx, f in enumerate(files, 1):
            try:
                process_resume(f, resume_id=idx, output_dir=output_dir, verbose=True)
                success_count += 1
            except Exception as e:
                print(f"❌ ERROR processing ID #{idx} ({f}): {e}")
                fail_count += 1
            finally:
                gc.collect()

    total_elapsed = time.time() - start_time
    avg_rate = total_elapsed / max(1, success_count)

    # Layer 1 Candidate Deduplication & Smart Merge:
    # Identifies candidates across multiple parsed resumes by Email/Mobile Number,
    # merges new skills and experiences, and removes redundant records.
    consolidated_count = consolidate_duplicate_candidates(output_dir)
    final_unique_count = len(list(Path(output_dir).glob("*.json")))

    print(f"\n{'═' * 65}")
    print(f"🎉 PIPELINE EXECUTION COMPLETE")
    print(f" Total Processed:    {total_files} resumes")
    print(f" Succeeded:          {success_count}")
    print(f" Failed:             {fail_count}")
    if consolidated_count > 0:
        print(f" Smart-Merged Dups:  {consolidated_count} duplicate candidate resume(s)")
        print(f" Unique Candidates:  {final_unique_count} distinct candidate profiles")
    print(f" Total Elapsed Time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} mins)")
    print(f" Average Speed:      {avg_rate:.2f}s per resume")
    print(f" Segmented Texts:    {SEGMENTED_DIR}/")
    print(f" Final Flat JSONs:   {output_dir}/")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Python ATS Resume Parser Pipeline")
    parser.add_argument("file", nargs="?", default=None, help="Path to single resume file")
    parser.add_argument("--id", type=int, default=1, help="Counting ID number for single resume (default: 1)")
    parser.add_argument("--all", action="store_true", help="Process all resumes in batch")
    parser.add_argument("--input", type=str, default=DEFAULT_RESUME_DIR, help="Resume input directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of resumes to process")
    parser.add_argument("--output", type=str, default="flat_json", help="Output directory for flat JSONs")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    args = parser.parse_args()

    if args.file:
        process_resume(args.file, resume_id=args.id, output_dir=args.output, verbose=True)
    elif args.all or not args.file:
        process_all(
            input_dir=args.input,
            output_dir=args.output,
            limit=args.limit,
            workers=args.workers
        )
