#!/usr/bin/env python3
"""
reflatten_all.py
================
Batch re-flattens all candidate records in ats_parser/flat_json/ using the updated
Layer 3 pipeline and skills taxonomy.

Applies:
1. Leading bullet stripping and title sanitization.
2. Dash-level validation (e.g. Consultant-3).
3. Structured work_experience preservation with responsibility & project bullets.
4. Updated skills taxonomy (e.g. 'Internal Audit').
"""

import os
import sys
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from layer3.flatten import flatten_resume_record, _sanitize_title, _sanitize_company

FLAT_JSON_DIR = BASE_DIR / "flat_json"
SEG_DIR = BASE_DIR / "segmented_text"


def find_segmented_file(flat_stem: str, file_name: str) -> Path | None:
    # 1. Exact stem match
    c1 = SEG_DIR / f"{flat_stem}_segmented.txt"
    if c1.exists():
        return c1

    # 2. Match by file_name stem
    if file_name:
        fname_stem = Path(file_name).stem
        candidates = list(SEG_DIR.glob(f"*{fname_stem}*_segmented.txt"))
        if candidates:
            return candidates[0]

    # 3. Match without leading ID in flat_stem
    core = "_".join(flat_stem.split("_")[1:])
    if core:
        candidates = list(SEG_DIR.glob(f"*{core}*_segmented.txt"))
        if candidates:
            return candidates[0]

    return None


def main():
    print(f"\n{'═' * 65}")
    print(f"🔄 BATCH RE-FLATTENING ALL CANDIDATE RECORDS IN FLAT_JSON")
    print(f"{'═' * 65}")
    print(f"• Target Directory    : {FLAT_JSON_DIR}")
    print(f"• Segmented Directory : {SEG_DIR}")
    print(f"{'═' * 65}\n")

    flat_files = sorted(
        FLAT_JSON_DIR.glob("*.json"),
        key=lambda p: int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else 999999
    )
    total_files = len(flat_files)
    print(f"Found {total_files} flat json files.")

    t0 = time.perf_counter()
    reflattened_count = 0
    in_place_cleaned_count = 0
    work_exp_count = 0

    for idx, p in enumerate(flat_files, 1):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            flat_stem = p.stem.replace("_flat", "")
            file_name = data.get("file_name", "")

            seg_path = find_segmented_file(flat_stem, file_name)

            if seg_path and seg_path.exists():
                new_flat = flatten_resume_record(str(seg_path))
                # Preserve identity and metadata
                for k in ["id", "file_name", "file_md5", "source_file", "raw_text_length", "ingest_time"]:
                    if k in data and data[k] is not None:
                        new_flat[k] = data[k]

                # If name or contact was in data but missing or less confident in new_flat, preserve data
                if (not new_flat.get("name") or new_flat.get("name") == "Unknown") and data.get("name"):
                    new_flat["name"] = data["name"]
                if not new_flat.get("email") and data.get("email"):
                    new_flat["email"] = data["email"]
                if not new_flat.get("mobile_number") and data.get("mobile_number"):
                    new_flat["mobile_number"] = data["mobile_number"]

                with open(p, "w", encoding="utf-8") as f:
                    json.dump(new_flat, f, indent=2, ensure_ascii=False)

                reflattened_count += 1
                if new_flat.get("work_experience"):
                    work_exp_count += 1
            else:
                # In-place sanitization for any missing segmented file
                desigs = data.get("designation")
                if desigs and isinstance(desigs, list):
                    cleaned = []
                    for d in desigs:
                        cd = _sanitize_title(d)
                        if cd and cd not in cleaned:
                            cleaned.append(cd)
                    data["designation"] = cleaned

                with open(p, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                in_place_cleaned_count += 1

            if idx % 50 == 0 or idx == total_files:
                print(f"[{idx}/{total_files}] Processed ({reflattened_count} re-flattened, {work_exp_count} with structured work_exp)")

        except Exception as e:
            print(f"Error processing {p.name}: {e}")

    duration = time.perf_counter() - t0
    print(f"\n{'═' * 65}")
    print(f"✓ FINISHED IN {duration:.2f}s ({total_files / max(duration, 0.01):.1f} files/sec)")
    print(f"• Successfully re-flattened from segmented: {reflattened_count}")
    print(f"• In-place sanitized: {in_place_cleaned_count}")
    print(f"• Candidates with full structured work_experience: {work_exp_count}")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
