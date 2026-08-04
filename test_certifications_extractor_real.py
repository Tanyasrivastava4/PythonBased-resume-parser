"""
Certifications & Achievements Extractor — Real test
Reads every *_segmented.txt file in segmented_text/, pulls out the sections
labeled "certifications" and "achievements" (if present), runs them through
extractors/certifications.py, prints a report to console, and saves that
same report to certifications_text/ — one .txt per resume.

Run from the ats_parser/ root: python test_certifications_extractor_real.py
"""

import os
import re
import glob

from extractors.certifications import extract_certifications

SEGMENTED_DIR = "segmented_text"
OUTPUT_DIR = "certifications_text"

SECTION_RE = re.compile(
    r"LABEL:\s*(?P<label>\w+).*?\n─+\n(?P<body>.*?)(?=\n─+\nLABEL:|\Z)",
    re.DOTALL
)


def extract_section(full_text: str, label: str) -> str:
    for m in SECTION_RE.finditer(full_text):
        if m.group("label").strip().lower() == label:
            return m.group("body").strip()
    return ""


def build_report(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    cert_text = extract_section(full_text, "certifications")
    # Fall back to a "custom" section if your segmenter routes training /
    # extra credentials there instead of a dedicated "achievements" label
    achievement_text = extract_section(full_text, "achievements") or extract_section(full_text, "custom")

    lines = []
    lines.append("=" * 70)
    lines.append(f"FILE: {os.path.basename(filepath)}")
    lines.append("=" * 70)

    if not cert_text and not achievement_text:
        lines.append("No 'certifications' or 'achievements' section found in this file.\n")
        return "\n".join(lines)

    if cert_text:
        lines.append("RAW CERTIFICATIONS SECTION")
        lines.append("-" * 70)
        lines.append(cert_text)
        lines.append("")

    if achievement_text:
        lines.append("RAW ACHIEVEMENTS / CUSTOM SECTION")
        lines.append("-" * 70)
        lines.append(achievement_text)
        lines.append("")

    result = extract_certifications(cert_text, achievement_text)

    lines.append("EXTRACTION RESULTS")
    lines.append("-" * 70)
    lines.append(f"Certifications found : {len(result['certifications'])}")
    lines.append(f"Achievements found   : {len(result['achievements'])}")
    lines.append(f"Confidence           : {result['_confidence']}")
    lines.append(f"Needs review         : {result['_needs_review']}")
    lines.append("")

    for i, c in enumerate(result["certifications"], 1):
        lines.append(f"Cert {i}")
        lines.append(f"  Name         : {c['name']}")
        lines.append(f"  Issuer       : {c['issuer']}")
        lines.append(f"  Year         : {c['year']}")
        lines.append(f"  Confidence   : {c['confidence']}")
        lines.append(f"  Needs Review : {c['needs_review']}")
        lines.append("")

    if result["achievements"]:
        lines.append("Achievements")
        for a in result["achievements"]:
            lines.append(f"  • {a}")
        lines.append("")

    return "\n".join(lines)


def output_filename(segmented_filepath: str) -> str:
    base = os.path.basename(segmented_filepath).replace("_segmented.txt", "")
    return f"{base}_certifications.txt"


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SEGMENTED_DIR, "*_segmented.txt")))
    if not files:
        print(f"No files found in {SEGMENTED_DIR}/")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fp in files:
        report = build_report(fp)
        print(report)

        out_path = os.path.join(OUTPUT_DIR, output_filename(fp))
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(report)

    print(f"\nSaved {len(files)} report(s) to {OUTPUT_DIR}/")