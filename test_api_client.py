"""
test_api_client.py
==================
Batch Testing Client for ATS Resume Parser FastAPI Server

Sends files from `resumes/` folder directly to the running FastAPI server
endpoint to verify real-time processing, model persistence, and batch response speed.

Usage:
    # Test single file against live server:
    python test_api_client.py "resumes/Akash Resume - Developer.pdf"

    # Test all resumes in resumes/ directory:
    python test_api_client.py --all

    # Test batch payload endpoint with 10 files:
    python test_api_client.py --batch
"""

import sys
import glob
import time
import argparse
import requests
from pathlib import Path


SERVER_URL = "http://127.0.0.1:8000"


def check_server_health() -> bool:
    """Checks if the FastAPI server is running."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Server Health: OK")
            print(f"   Service       : {data.get('service')}")
            print(f"   Models Cached : {data.get('models_cached_in_ram')}")
            return True
    except Exception:
        pass
    print(f"❌ Server is NOT running at {SERVER_URL}.")
    print(f"   Please start the server first using:")
    print(f"   ./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000")
    return False


def test_single_file(file_path: str):
    """Sends a single resume file to the server."""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\n[CLIENT] Sending file: {path.name}")
    start_time = time.time()

    with open(path, "rb") as f:
        files = {"file": (path.name, f, "application/octet-stream")}
        response = requests.post(f"{SERVER_URL}/api/v1/parse-resume", files=files)

    elapsed = time.time() - start_time
    print(f"[CLIENT] Status Code : {response.status_code}")
    print(f"[CLIENT] Elapsed Time: {elapsed:.2f} seconds")

    if response.status_code in (200, 202):
        data = response.json()
        print(f"[CLIENT] Mode        : {data.get('execution_mode')}")
        print(f"[CLIENT] Status      : {data.get('status')}")
        if data.get("parsed_data"):
            parsed = data["parsed_data"]
            print(f"   Candidate Name : {parsed.get('name')}")
            print(f"   Candidate Email: {parsed.get('email')}")
            print(f"   Skills Extracted: {len(parsed.get('skills', []))} skills")
        elif data.get("job_id"):
            print(f"   Background Job ID: {data.get('job_id')}")
            print(f"   Status URL       : {data.get('status_url')}")
    else:
        print(f"[CLIENT] Response Error: {response.text}")


def test_all_files(input_dir: str = "resumes"):
    """Sends all resumes from input_dir to the server one after another."""
    folder = Path(input_dir)
    files = sorted(glob.glob(str(folder / "*.pdf")) + glob.glob(str(folder / "*.docx")))

    if not files:
        print(f"No files found in {input_dir}/")
        return

    print(f"\n🚀 Testing server with {len(files)} resume(s) from '{input_dir}/'...")
    start_total = time.time()

    for idx, f in enumerate(files, 1):
        print(f"\n--- File [{idx}/{len(files)}] ---")
        test_single_file(f)

    total_elapsed = time.time() - start_total
    print(f"\n{'═' * 60}")
    print(f"ALL {len(files)} FILES TESTED in {total_elapsed:.2f} seconds!")
    print(f"Average time per file: {total_elapsed / len(files):.2f}s")
    print(f"{'═' * 60}\n")


def test_batch_payload(input_dir: str = "resumes", count: int = 5):
    """Tests the /api/v1/parse-batch multi-file upload endpoint."""
    folder = Path(input_dir)
    files = sorted(glob.glob(str(folder / "*.pdf")) + glob.glob(str(folder / "*.docx")))[:count]

    if not files:
        print(f"No files found in {input_dir}/")
        return

    print(f"\n🚀 Sending batch request of {len(files)} files to /api/v1/parse-batch...")
    file_tuples = []
    handles = []

    for fpath in files:
        p = Path(fpath)
        h = open(p, "rb")
        handles.append(h)
        file_tuples.append(("files", (p.name, h, "application/octet-stream")))

    start_time = time.time()
    try:
        response = requests.post(f"{SERVER_URL}/api/v1/parse-batch", files=file_tuples)
        elapsed = time.time() - start_time

        print(f"[BATCH CLIENT] Status Code : {response.status_code}")
        print(f"[BATCH CLIENT] Elapsed Time: {elapsed:.2f} seconds")

        if response.status_code == 200:
            data = response.json()
            print(f"   Total Files          : {data.get('total_files')}")
            print(f"   Digital Immediate    : {data.get('digital_immediate_completed')}")
            print(f"   Scanned Queued Async : {data.get('scanned_background_queued')}")
    finally:
        for h in handles:
            h.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test client for ATS Resume Parser FastAPI Server.")
    parser.add_argument("file", nargs="?", help="Single file to send to live server.")
    parser.add_argument("--all", action="store_true", help="Test all resumes in resumes/ directory.")
    parser.add_argument("--batch", action="store_true", help="Test batch upload endpoint with 5 resumes.")

    args = parser.parse_args()

    if not check_server_health():
        sys.exit(1)

    if args.all:
        test_all_files()
    elif args.batch:
        test_batch_payload()
    elif args.file:
        test_single_file(args.file)
    else:
        print("Usage: python test_api_client.py [file | --all | --batch]")
