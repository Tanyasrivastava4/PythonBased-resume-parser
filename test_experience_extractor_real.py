"""
Experience Extractor — Real test
Reads every *_segmented.txt file in segmented_text/, pulls out the section
labeled "experience", runs it through extractors/experience.py, prints a
report to console, and saves that same report to experience_text/ —
one .txt per resume, matching the segmented_text/ output convention.

Run from the ats_parser/ root: python test_experience_extractor_real.py

FIX: extract_section() used to return on the FIRST regex match whose
label matched, via `finditer` + an early `return` inside the loop.
Confirmed on a real resume (Abhinav Ashish): his experience content is
split across TWO separate "LABEL: experience" blocks in the segmented
file (a short "AMTAG Global..." bullet with no date, then the real job
history further down, separated by a "languages" block in between).
The early-return meant only the first, non-dated block ever reached
extract_experience() -- the real job history (NPCI, Grant Thornton, AAA
Technologies, Focus Forensics) was silently discarded every time,
producing "Entries found: 0" even though the extractor itself works
correctly once given the full text.

extract_section() now collects EVERY matching block and joins them with
"\\n\\n", the same join convention section_splitter.py's own
get_section_text() already uses for exactly this situation -- so
multiple same-labeled sections in one resume (which split_into_sections()
legitimately produces sometimes, e.g. a name/header line resetting
detection mid-document) are combined instead of only the first one
surviving.
"""

import os
import re
import glob

from extractors.experience import extract_experience

SEGMENTED_DIR = "segmented_text"
OUTPUT_DIR = "experience_text"

# Matches a labeled section block like:
# ────────────────
# LABEL: experience   CONFIDENCE: 0.95   START LINE: 12
# ────────────────
# <content...>
SECTION_RE = re.compile(
    r"LABEL:\s*(?P<label>\w+).*?\n─+\n(?P<body>.*?)(?=\n─+\nLABEL:|\Z)",
    re.DOTALL
)


def extract_section(full_text: str, label: str) -> str:
    """
    Collects the raw text of EVERY section block matching `label` in the
    segmented file and joins them with a blank line between each --
    NOT just the first one. A resume can legitimately have the same
    section label appear more than once (confirmed on a real resume,
    Abhinav Ashish: two separate "experience" blocks), and dropping
    everything after the first occurrence silently discards real content.
    """
    matches = [
        m.group("body").strip()
        for m in SECTION_RE.finditer(full_text)
        if m.group("label").strip().lower() == label
    ]
    return "\n\n".join(matches)


def build_report(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    exp_text = extract_section(full_text, "experience")

    lines = []
    lines.append("=" * 70)
    lines.append(f"FILE: {os.path.basename(filepath)}")
    lines.append("=" * 70)

    if not exp_text:
        lines.append("No 'experience' section found in this file.\n")
        return "\n".join(lines)

    lines.append("RAW EXPERIENCE SECTION (first 300 chars)")
    lines.append("-" * 70)
    lines.append(exp_text[:300] + ("..." if len(exp_text) > 300 else ""))
    lines.append("")

    result = extract_experience(exp_text)
    jobs = result["experience"]

    lines.append("EXTRACTION RESULTS")
    lines.append("-" * 70)
    lines.append(f"Entries found  : {len(jobs)}")
    lines.append(f"Confidence     : {result['_confidence']}")
    lines.append(f"Needs review   : {result['_needs_review']}")
    te = result["_total_experience"]
    lines.append(f"Total exp      : {te['total_years']} years ({te['total_months']} months, "
                  f"{te['overlapping_periods_merged']} overlapping periods merged)")
    lines.append("")

    for i, job in enumerate(jobs, 1):
        lines.append(f"Entry {i}")
        lines.append(f"  Role         : {job['role']}")
        lines.append(f"  Company      : {job['company']}")
        lines.append(f"  Location     : {job['location']}")
        lines.append(f"  Start - End  : {job['start']} - {job['end']}")
        lines.append(f"  Duration     : {job['duration_months']} months")
        lines.append(f"  Bullets      : {len(job['bullets'])}")
        for b in job["bullets"]:
            lines.append(f"    • {b}")
        if job["sub_sections"]:
            lines.append(f"  Sub-sections : {job['sub_sections']}")
        lines.append(f"  Method       : {job['method']}")
        lines.append(f"  Confidence   : {job['confidence']}")
        lines.append(f"  Needs Review : {job['needs_review']}")
        lines.append("")

    return "\n".join(lines)


def output_filename(segmented_filepath: str) -> str:
    base = os.path.basename(segmented_filepath)
    base = base.replace("_segmented.txt", "")
    return f"{base}_experience.txt"


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


















# just commenting to work with two experience
#"""
#Experience Extractor — Real test
#Reads every *_segmented.txt file in segmented_text/, pulls out the section
#labeled "experience", runs it through extractors/experience.py, prints a
#report to console, and saves that same report to experience_text/ —
#one .txt per resume, matching the segmented_text/ output convention.
#
#Run from the ats_parser/ root: python test_experience_extractor_real.py
#"""
#
#import os
#import re
#import glob
#
#from extractors.experience import extract_experience
#
#SEGMENTED_DIR = "segmented_text"
#OUTPUT_DIR = "experience_text"
#
## Matches a labeled section block like:
## ────────────────
## LABEL: experience   CONFIDENCE: 0.95   START LINE: 12
## ────────────────
## <content...>
#SECTION_RE = re.compile(
#    r"LABEL:\s*(?P<label>\w+).*?\n─+\n(?P<body>.*?)(?=\n─+\nLABEL:|\Z)",
#    re.DOTALL
#)
#
#
#def extract_section(full_text: str, label: str) -> str:
#    for m in SECTION_RE.finditer(full_text):
#        if m.group("label").strip().lower() == label:
#            return m.group("body").strip()
#    return ""
#
#
#def build_report(filepath: str) -> str:
#    with open(filepath, "r", encoding="utf-8") as f:
#        full_text = f.read()
#
#    exp_text = extract_section(full_text, "experience")
#
#    lines = []
#    lines.append("=" * 70)
#    lines.append(f"FILE: {os.path.basename(filepath)}")
#    lines.append("=" * 70)
#
#    if not exp_text:
#        lines.append("No 'experience' section found in this file.\n")
#        return "\n".join(lines)
#
#    lines.append("RAW EXPERIENCE SECTION (first 300 chars)")
#    lines.append("-" * 70)
#    lines.append(exp_text[:300] + ("..." if len(exp_text) > 300 else ""))
#    lines.append("")
#
#    result = extract_experience(exp_text)
#    jobs = result["experience"]
#
#    lines.append("EXTRACTION RESULTS")
#    lines.append("-" * 70)
#    lines.append(f"Entries found  : {len(jobs)}")
#    lines.append(f"Confidence     : {result['_confidence']}")
#    lines.append(f"Needs review   : {result['_needs_review']}")
#    te = result["_total_experience"]
#    lines.append(f"Total exp      : {te['total_years']} years ({te['total_months']} months, "
#                  f"{te['overlapping_periods_merged']} overlapping periods merged)")
#    lines.append("")
#
#    for i, job in enumerate(jobs, 1):
#        lines.append(f"Entry {i}")
#        lines.append(f"  Role         : {job['role']}")
#        lines.append(f"  Company      : {job['company']}")
#        lines.append(f"  Location     : {job['location']}")
#        lines.append(f"  Start - End  : {job['start']} - {job['end']}")
#        lines.append(f"  Duration     : {job['duration_months']} months")
#        lines.append(f"  Bullets      : {len(job['bullets'])}")
#        for b in job["bullets"]:
#            lines.append(f"    • {b}")
#        if job["sub_sections"]:
#            lines.append(f"  Sub-sections : {job['sub_sections']}")
#        lines.append(f"  Method       : {job['method']}")
#        lines.append(f"  Confidence   : {job['confidence']}")
#        lines.append(f"  Needs Review : {job['needs_review']}")
#        lines.append("")
#
#    return "\n".join(lines)
#
#
#def output_filename(segmented_filepath: str) -> str:
#    base = os.path.basename(segmented_filepath)
#    base = base.replace("_segmented.txt", "")
#    return f"{base}_experience.txt"
#
#
#if __name__ == "__main__":
#    files = sorted(glob.glob(os.path.join(SEGMENTED_DIR, "*_segmented.txt")))
#    if not files:
#        print(f"No files found in {SEGMENTED_DIR}/")
#
#    os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#    for fp in files:
#        report = build_report(fp)
#        print(report)
#
#        out_path = os.path.join(OUTPUT_DIR, output_filename(fp))
#        with open(out_path, "w", encoding="utf-8") as out_f:
#            out_f.write(report)
#
#    print(f"\nSaved {len(files)} report(s) to {OUTPUT_DIR}/")
#







#"""
#Experience Extractor — Real test
#Reads every *_segmented.txt file in segmented_text/, pulls out the section
#labeled "experience", runs it through extractors/experience.py, and prints
#a report in the same style as your education/skills extractor tests.
#
#Run from the ats_parser/ root: python test_experience_extractor_real.py
#"""
#
#import os
#import re
#import glob
#
#from extractors.experience import extract_experience
#
#SEGMENTED_DIR = "segmented_text"
#
## Matches a labeled section block like:
## ────────────────
## LABEL: experience   CONFIDENCE: 0.95   START LINE: 12
## ────────────────
## <content...>
#SECTION_RE = re.compile(
#    r"LABEL:\s*(?P<label>\w+).*?\n─+\n(?P<body>.*?)(?=\n─+\nLABEL:|\Z)",
#    re.DOTALL
#)
#
#
#def extract_section(full_text: str, label: str) -> str:
#    for m in SECTION_RE.finditer(full_text):
#        if m.group("label").strip().lower() == label:
#            return m.group("body").strip()
#    return ""
#
#
#def print_report(filepath: str):
#    with open(filepath, "r", encoding="utf-8") as f:
#        full_text = f.read()
#
#    exp_text = extract_section(full_text, "experience")
#
#    print("=" * 70)
#    print(f"FILE: {os.path.basename(filepath)}")
#    print("=" * 70)
#
#    if not exp_text:
#        print("No 'experience' section found in this file.\n")
#        return
#
#    print("RAW EXPERIENCE SECTION (first 300 chars)")
#    print("-" * 70)
#    print(exp_text[:300] + ("..." if len(exp_text) > 300 else ""))
#    print()
#
#    result = extract_experience(exp_text)
#    jobs = result["experience"]
#
#    print("EXTRACTION RESULTS")
#    print("-" * 70)
#    print(f"Entries found  : {len(jobs)}")
#    print(f"Confidence     : {result['_confidence']}")
#    print(f"Needs review   : {result['_needs_review']}")
#    te = result["_total_experience"]
#    print(f"Total exp      : {te['total_years']} years ({te['total_months']} months, "
#          f"{te['overlapping_periods_merged']} overlapping periods merged)")
#    print()
#
#    for i, job in enumerate(jobs, 1):
#        print(f"Entry {i}")
#        print(f"  Role         : {job['role']}")
#        print(f"  Company      : {job['company']}")
#        print(f"  Location     : {job['location']}")
#        print(f"  Start - End  : {job['start']} - {job['end']}")
#        print(f"  Duration     : {job['duration_months']} months")
#        print(f"  Bullets      : {len(job['bullets'])}")
#        for b in job["bullets"][:2]:
#            print(f"    • {b[:80]}{'...' if len(b) > 80 else ''}")
#        if job["sub_sections"]:
#            print(f"  Sub-sections : {job['sub_sections']}")
#        print(f"  Method       : {job['method']}")
#        print(f"  Confidence   : {job['confidence']}")
#        print(f"  Needs Review : {job['needs_review']}")
#        print()
#
#
#if __name__ == "__main__":
#    files = sorted(glob.glob(os.path.join(SEGMENTED_DIR, "*_segmented.txt")))
#    if not files:
#        print(f"No files found in {SEGMENTED_DIR}/")
#    for fp in files:
#        print_report(fp)
#












#"""
#Experience Extractor — Real test
#Reads every *_segmented.txt file in segmented_text/, pulls out the section
#labeled "experience", runs it through extractors/experience.py, and prints
#a report in the same style as your education/skills extractor tests.
#
#Run from the ats_parser/ root: python test_experience_extractor_real.py
#"""
#
#import os
#import re
#import glob
#
#from extractors.experience import extract_experience
#
#SEGMENTED_DIR = "segmented_text"
#
## Matches a labeled section block like:
## ────────────────
## LABEL: experience   CONFIDENCE: 0.95   START LINE: 12
## ────────────────
## <content...>
#SECTION_RE = re.compile(
#    r"LABEL:\s*(?P<label>\w+).*?\n─+\n(?P<body>.*?)(?=\n─+\nLABEL:|\Z)",
#    re.DOTALL
#)
#
#
#def extract_section(full_text: str, label: str) -> str:
#    for m in SECTION_RE.finditer(full_text):
#        if m.group("label").strip().lower() == label:
#            return m.group("body").strip()
#    return ""
#
#
#def print_report(filepath: str):
#    with open(filepath, "r", encoding="utf-8") as f:
#        full_text = f.read()
#
#    exp_text = extract_section(full_text, "experience")
#
#    print("=" * 70)
#    print(f"FILE: {os.path.basename(filepath)}")
#    print("=" * 70)
#
#    if not exp_text:
#        print("No 'experience' section found in this file.\n")
#        return
#
#    print("RAW EXPERIENCE SECTION (first 300 chars)")
#    print("-" * 70)
#    print(exp_text[:300] + ("..." if len(exp_text) > 300 else ""))
#    print()
#
#    result = extract_experience(exp_text)
#    jobs = result["experience"]
#
#    print("EXTRACTION RESULTS")
#    print("-" * 70)
#    print(f"Entries found  : {len(jobs)}")
#    print(f"Confidence     : {result['_confidence']}")
#    print(f"Needs review   : {result['_needs_review']}")
#    print()
#
#    for i, job in enumerate(jobs, 1):
#        print(f"Entry {i}")
#        print(f"  Role         : {job['role']}")
#        print(f"  Company      : {job['company']}")
#        print(f"  Location     : {job['location']}")
#        print(f"  Start - End  : {job['start']} - {job['end']}")
#        print(f"  Duration     : {job['duration_months']} months")
#        print(f"  Bullets      : {len(job['bullets'])}")
#        for b in job["bullets"][:2]:
#            print(f"    • {b[:80]}{'...' if len(b) > 80 else ''}")
#        if job["sub_sections"]:
#            print(f"  Sub-sections : {job['sub_sections']}")
#        print(f"  Method       : {job['method']}")
#        print(f"  Confidence   : {job['confidence']}")
#        print(f"  Needs Review : {job['needs_review']}")
#        print()
#
#
#if __name__ == "__main__":
#    files = sorted(glob.glob(os.path.join(SEGMENTED_DIR, "*_segmented.txt")))
#    if not files:
#        print(f"No files found in {SEGMENTED_DIR}/")
#    for fp in files:
#        print_report(fp)