"""
test_flatten.py
Runs the flat schema over every *_segmented.txt file and saves each
resume's result as its OWN .json file into flat_json/ — one file per
resume, same pattern as experience_text/, education_text/, etc.

Usage:
    # Run on ALL resumes:
    python test_flatten.py

    # Run on ONE specific resume:
    python test_flatten.py segmented_text/Adarsh_Bhagat_Resume_segmented.txt
"""

import sys
import json
import glob
from pathlib import Path

from layer3.flatten import flatten_resume_record

SEGMENTED_DIR = Path("segmented_text")
OUTPUT_DIR = Path("flat_json")


def output_filename(segmented_filepath: str) -> str:
    base = Path(segmented_filepath).stem
    if base.endswith("_segmented"):
        base = base[: -len("_segmented")]
    return f"{base}_flat.json"


def process_file(file_path: str):
    print(f"\n{'═' * 65}")
    print(f"FILE : {Path(file_path).name}")
    print(f"{'═' * 65}")

    flat = flatten_resume_record(file_path)
    print(json.dumps(flat, indent=2, ensure_ascii=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / output_filename(file_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved flat JSON to: {out_path}]")
    return flat


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob(str(SEGMENTED_DIR / "*_segmented.txt")))
        if not files:
            print(f"No *_segmented.txt files found in {SEGMENTED_DIR}/")

    all_results = []
    for fp in files:
        result = process_file(fp)
        all_results.append(result)

    print(f"\n{'═' * 65}")
    print(f"Processed {len(all_results)} resume(s). Saved individually to {OUTPUT_DIR}/")
    print(f"{'═' * 65}")



















#this will save all resume o/p in 1 json file
#"""
#test_flatten.py
#Runs the flat schema over every *_segmented.txt file and saves the
#results as a JSON array (one dict per resume) into flat_json/.
#"""
#
#import sys
#import json
#import glob
#from pathlib import Path
#
#from layer3.flatten import flatten_resume_record
#
#SEGMENTED_DIR = Path("segmented_text")
#OUTPUT_DIR = Path("flat_json")
#
#
#def process_all(files):
#    results = []
#    for fp in files:
#        flat = flatten_resume_record(fp)
#        results.append(flat)
#        print(json.dumps(flat, indent=2, ensure_ascii=False))
#        print()
#
#    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#    out_path = OUTPUT_DIR / "all_resumes_flat.json"
#    with open(out_path, "w", encoding="utf-8") as f:
#        json.dump(results, f, indent=2, ensure_ascii=False)
#
#    print(f"[Saved {len(results)} resume(s) to: {out_path}]")
#
#
#if __name__ == "__main__":
#    if len(sys.argv) > 1:
#        files = sys.argv[1:]
#    else:
#        files = sorted(glob.glob(str(SEGMENTED_DIR / "*_segmented.txt")))
#        if not files:
#            print(f"No *_segmented.txt files found in {SEGMENTED_DIR}/")
#
#    process_all(files)
#    