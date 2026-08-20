"""
test_layer3.py
━━━━━━━━━━━━━━
Layer 3 test — runs every *_segmented.txt file in segmented_text/
through build_resume_record() and saves one structured .json per
resume into structured_json/, plus a human-readable .txt summary
(same "save both a machine format and a readable report" pattern
every other layer's test script already follows).

Usage:
    # Run on ALL resumes:
    python test_layer3.py

    # Run on ONE specific resume:
    python test_layer3.py segmented_text/Adarsh_Bhagat_Resume_segmented.txt
"""

import sys
import json
import glob
from pathlib import Path

from layer3 import build_resume_record

SEGMENTED_DIR = Path("segmented_text")
JSON_OUTPUT_DIR = Path("structured_json")
REPORT_OUTPUT_DIR = Path("structured_json_reports")


def _fmt(value):
    return value if value is not None else "None"


def build_readable_report(record: dict) -> str:
    lines = []
    meta = record["meta"]
    lines.append("=" * 70)
    lines.append(f"FILE: {meta['source_file']}")
    lines.append("=" * 70)
    lines.append(f"Sections found : {meta['sections_found']}")
    if meta["schema_errors"]:
        lines.append(f"SCHEMA ERRORS  : {meta['schema_errors']}")
    lines.append(f"Overall conf.  : {record['_confidence']['overall']}")
    lines.append(f"Needs review   : {record['_needs_review']}")
    lines.append("")

    c = record["contact"]
    lines.append("CONTACT")
    lines.append("-" * 70)
    for field_name, field_data in c.items():
        lines.append(f"  {field_name:<10}: {_fmt(field_data['value'])}  "
                      f"(conf {field_data['confidence']}, {field_data['method']})")
    lines.append("")

    if record["summary"]:
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(record["summary"])
        lines.append("")

    sk = record["skills"]
    lines.append(f"SKILLS ({len(sk['items'])}) — confidence {sk['_confidence']}")
    lines.append("-" * 70)
    lines.append(", ".join(sk["items"]) if sk["items"] else "None found")
    lines.append("")

    exp = record["experience"]
    te = exp["total_experience"]
    lines.append(f"EXPERIENCE ({len(exp['jobs'])} jobs, "
                  f"{len(exp['early_career'])} early_career entries) — "
                  f"confidence {exp['_confidence']}")
    lines.append(f"Total experience: {te['total_years']} years "
                  f"({te['total_months']} months, "
                  f"{te['overlapping_periods_merged']} overlaps merged)")
    lines.append("-" * 70)
    for i, job in enumerate(exp["jobs"], 1):
        lines.append(f"  {i}. {_fmt(job['role'])} @ {_fmt(job['company'])} "
                      f"({_fmt(job['start'])} - {_fmt(job['end'])}) "
                      f"[conf {job['confidence']}, review={job['needs_review']}]")
    lines.append("")

    edu = record["education"]
    lines.append(f"EDUCATION ({len(edu['entries'])}) — confidence {edu['_confidence']}")
    lines.append("-" * 70)
    for i, e in enumerate(edu["entries"], 1):
        lines.append(f"  {i}. {_fmt(e['degree_name'])} — {_fmt(e['institution'])} "
                      f"({_fmt(e['year'])}) [conf {e['_confidence']}, "
                      f"review={e['_needs_review']}]")
    lines.append("")

    cert = record["certifications"]
    lines.append(f"CERTIFICATIONS ({len(cert['items'])}) — confidence {cert['_confidence']}")
    lines.append("-" * 70)
    for i, c2 in enumerate(cert["items"], 1):
        lines.append(f"  {i}. {_fmt(c2['name'])} — {_fmt(c2['issuer'])} "
                      f"({_fmt(c2['year'])}) [conf {c2['confidence']}]")
    lines.append("")

    ach = record["achievements"]
    if ach["items"]:
        lines.append(f"ACHIEVEMENTS ({len(ach['items'])}) — confidence {ach['_confidence']}")
        lines.append("-" * 70)
        for a in ach["items"]:
            lines.append(f"  • {a}")
        lines.append("")

    rq = record["_review_queue"]
    lines.append(f"REVIEW QUEUE ({len(rq)} flagged fields)")
    lines.append("-" * 70)
    if rq:
        for item in rq:
            lines.append(f"  {item['field']:<35} conf={item['confidence']}  reason={item['reason']}")
    else:
        lines.append("  (nothing flagged)")
    lines.append("")

    return "\n".join(lines)


def process_file(file_path: str):
    print(f"\n{'═' * 65}")
    print(f"FILE : {Path(file_path).name}")
    print(f"{'═' * 65}")

    record = build_resume_record(file_path)

    JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(file_path).stem
    if original_name.endswith("_segmented"):
        original_name = original_name[: -len("_segmented")]

    json_path = JSON_OUTPUT_DIR / f"{original_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    report = build_readable_report(record)
    print(report)

    report_path = REPORT_OUTPUT_DIR / f"{original_name}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[Saved JSON to: {json_path}]")
    print(f"[Saved report to: {report_path}]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            process_file(path)
    else:
        files = sorted(glob.glob(str(SEGMENTED_DIR / "*_segmented.txt")))
        if not files:
            print(f"No *_segmented.txt files found in {SEGMENTED_DIR}/")
            print("Run python test_segmenter.py first to generate them.")
        for f in files:
            process_file(f)

        print(f"\n{'═' * 65}")
        print(f"Processed {len(files)} resume(s).")
        print(f"{'═' * 65}")