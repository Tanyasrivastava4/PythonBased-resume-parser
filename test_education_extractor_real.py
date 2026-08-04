"""
Layer 2 test — Education Extractor (real data)
Reads segmented resumes from segmented_text/, pulls out the 'education'
section, runs extractors/education.py, prints the results, and saves
them under education_text/.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extractors.education import extract_education

SEGMENTED_DIR = Path("segmented_text")
OUTPUT_DIR = Path("education_text")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Matches a "LABEL: xxx   CONFIDENCE: 0.95   START LINE: 94" header block,
# followed by a dashed line, followed by the section content.
SECTION_RE = re.compile(
    r'LABEL:\s*(\w+)\s+CONFIDENCE:\s*([\d.]+)\s+START LINE:\s*(\d+)\s*\n'
    r'[─\-]+\n'
    r'(.*?)'
    r'(?=\n[─\-]+\s*\nLABEL:|\n\[Found|\Z)',
    re.DOTALL
)


def get_section(segmented_text: str, label: str) -> str:
    """
    Returns the concatenated text of every section matching `label`.
    """
    chunks = []

    for match in SECTION_RE.finditer(segmented_text):
        if match.group(1).lower() == label.lower():
            chunks.append(match.group(4).strip())

    return "\n\n".join(chunks)


def save_output(filepath: Path,
                education_text: str,
                result: dict,
                elapsed_ms: float):
    """
    Saves the extraction results into education_text/.
    """

    output_file = OUTPUT_DIR / (
        filepath.stem.replace("_segmented", "") + "_education.txt"
    )

    entries = result["education"]

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write(f"FILE: {filepath.name}\n")
        f.write("=" * 70 + "\n\n")

        f.write("RAW EDUCATION SECTION\n")
        f.write("-" * 70 + "\n")

        if education_text:
            f.write(education_text)
        else:
            f.write("No education section found.")

        f.write("\n\n")

        f.write("EXTRACTION RESULTS\n")
        f.write("-" * 70 + "\n")

        f.write(f"Entries found : {len(entries)}\n")
        f.write(f"Confidence    : {result['_confidence']}\n")
        f.write(f"Time          : {elapsed_ms:.2f} ms\n\n")

        if not entries:
            f.write("No education entries extracted.\n")
            return

        for i, entry in enumerate(entries, start=1):

            f.write(f"Entry {i}\n")
            f.write(f"Degree       : {entry['degree']}\n")
            f.write(f"Degree Name  : {entry['degree_name']}\n")
            f.write(f"Degree Type  : {entry['degree_type']}\n")
            f.write(f"Institution  : {entry['institution']}\n")
            f.write(f"Year         : {entry['year']}\n")
            f.write(f"GPA          : {entry['gpa']}\n")
            f.write(f"Needs Review : {entry['_needs_review']}\n")
            f.write(f"Confidence   : {entry['_confidence']}\n")
            f.write("\n")


def run_on_file(filepath: Path):

    text = filepath.read_text(encoding="utf-8")

    education_text = get_section(text, "education")

    print("─" * 64)
    print(f"EDUCATION EXTRACTED FROM: {filepath}")
    print("─" * 64)

    if not education_text:

        print("No 'education' section found.\n")

        save_output(
            filepath,
            education_text,
            {
                "education": [],
                "_confidence": 0.0
            },
            0.0
        )

        return

    print("\n── Education Section Text (raw, as segmented) ──\n")
    print(education_text)

    start = time.perf_counter()

    result = extract_education(education_text)

    elapsed_ms = (time.perf_counter() - start) * 1000

    entries = result["education"]

    print("\n── Extraction Results ──")

    print(f"Entries found  : {len(entries)}")
    print(f"Confidence     : {result['_confidence']}")
    print(f"Time           : {elapsed_ms:.2f} ms\n")

    for i, entry in enumerate(entries, start=1):

        print(f"Entry {i}")
        print(f"  Degree       : {entry['degree']}")
        print(f"  Degree Name  : {entry['degree_name']}")
        print(f"  Degree Type  : {entry['degree_type']}")
        print(f"  Institution  : {entry['institution']}")
        print(f"  Year         : {entry['year']}")
        print(f"  GPA          : {entry['gpa']}")
        print(f"  Needs Review : {entry['_needs_review']}")
        print(f"  Confidence   : {entry['_confidence']}")
        print()

    # Save results to file
    save_output(
        filepath,
        education_text,
        result,
        elapsed_ms
    )

    print(f"Saved results → {OUTPUT_DIR / (filepath.stem.replace('_segmented', '') + '_education.txt')}")
    print()


def main():

    if not SEGMENTED_DIR.exists():
        print(f"'{SEGMENTED_DIR}' not found — run this from the ats_parser/ root.")
        return

    files = sorted(SEGMENTED_DIR.glob("*_segmented.txt"))

    if not files:
        print(f"No *_segmented.txt files found in {SEGMENTED_DIR}/")
        return

    print(f"Found {len(files)} segmented resumes.\n")

    for file in files:
        run_on_file(file)


if __name__ == "__main__":
    main()













#"""
#Layer 2 test — Education Extractor (real data)
#Reads segmented resumes from segmented_text/, pulls out the 'education'
#section, runs extractors/education.py, and prints results in the same
#style as test_skills_extractor_real.py / the contact extractor output.
#"""
#
#import re
#import sys
#import time
#from pathlib import Path
#
#sys.path.insert(0, str(Path(__file__).parent))
#from extractors.education import extract_education
#
#SEGMENTED_DIR = Path("segmented_text")
#
## Matches a "LABEL: xxx   CONFIDENCE: 0.95   START LINE: 94" header block,
## followed by a dashed line, followed by the section content, up to the
## next dashed-line/LABEL header or end of file.
#SECTION_RE = re.compile(
#    r'LABEL:\s*(\w+)\s+CONFIDENCE:\s*([\d.]+)\s+START LINE:\s*(\d+)\s*\n'
#    r'[─\-]+\n'
#    r'(.*?)'
#    r'(?=\n[─\-]+\s*\nLABEL:|\n\[Found|\Z)',
#    re.DOTALL
#)
#
#
#def get_section(segmented_text: str, label: str) -> str:
#    """Returns the concatenated text of every section matching `label`
#    (a resume can have more than one 'skills' block, for example)."""
#    chunks = []
#    for match in SECTION_RE.finditer(segmented_text):
#        if match.group(1).lower() == label.lower():
#            chunks.append(match.group(4).strip())
#    return "\n\n".join(chunks)
#
#
#def run_on_file(filepath: Path):
#    text = filepath.read_text(encoding="utf-8")
#    education_text = get_section(text, "education")
#
#    print("─" * 64)
#    print(f"EDUCATION EXTRACTED FROM: {filepath}")
#    print("─" * 64)
#
#    if not education_text:
#        print("  No 'education' section found in this file.\n")
#        return
#
#    print("\n── Education Section Text (raw, as segmented) ──")
#    print(education_text)
#
#    start = time.perf_counter()
#    result = extract_education(education_text)
#    elapsed_ms = (time.perf_counter() - start) * 1000
#
#    entries = result["education"]
#    print(f"\n── Extraction Results ──")
#    print(f"Entries found  : {len(entries)}")
#    print(f"Confidence     : {result['_confidence']}")
#    print(f"Time           : {elapsed_ms:.2f} ms\n")
#
#    for i, entry in enumerate(entries, 1):
#        print(f"  Entry {i}")
#        print(f"    degree       : {entry['degree']}")
#        print(f"    degree_name  : {entry['degree_name']}")
#        print(f"    degree_type  : {entry['degree_type']}")
#        print(f"    institution  : {entry['institution']}")
#        print(f"    year         : {entry['year']}")
#        print(f"    gpa          : {entry['gpa']}")
#        print(f"    needs_review : {entry['_needs_review']}")
#        print(f"    confidence   : {entry['_confidence']}")
#        print()
#
#
#def main():
#    if not SEGMENTED_DIR.exists():
#        print(f"'{SEGMENTED_DIR}' not found — run this from the ats_parser/ root.")
#        return
#
#    files = sorted(SEGMENTED_DIR.glob("*_segmented.txt"))
#    if not files:
#        print(f"No *_segmented.txt files found in {SEGMENTED_DIR}/")
#        return
#
#    for f in files:
#        run_on_file(f)
#
#
#if __name__ == "__main__":
#    main()
#    