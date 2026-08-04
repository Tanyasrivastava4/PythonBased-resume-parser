"""
test_contact_extractor.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Batch test for Layer 2's contact extractor.

Runs extract_contact() against EVERY *_segmented.txt file in
segmented_text/ (Layer 1's output folder), and saves one result file
per resume into contact_extracted_text/.

Usage:
    # Run on ALL segmented resumes:
    python test_contact_extractor.py

    # Run on ONE specific file:
    python test_contact_extractor.py segmented_text/Abhay_Awasthi_..._segmented.txt
"""

import sys
import glob
from pathlib import Path

from extractors.contact_extractor import extract_contact

# ── Folder where Layer 1's segmented output lives ──
INPUT_DIR = Path("segmented_text")

# ── Folder where this script's output will be saved ──
OUTPUT_DIR = Path("contact_extracted_text")


def save_contact_output(file_path: str, contact: dict) -> Path:
    """
    Saves one resume's extracted contact info to contact_extracted_text/,
    using the same base name as the input file (minus "_segmented",
    plus "_contact.txt").

    e.g. segmented_text/Abhay_Awasthi_..._segmented.txt
         -> contact_extracted_text/Abhay_Awasthi_..._contact.txt
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(file_path).stem
    if original_name.endswith("_segmented"):
        original_name = original_name[: -len("_segmented")]
    output_path = OUTPUT_DIR / f"{original_name}_contact.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'─' * 60}\n")
        f.write(f"CONTACT INFO EXTRACTED FROM: {file_path}\n")
        f.write(f"{'─' * 60}\n")
        for field_name, field_data in contact.items():
            f.write(f"\n{field_name.upper()}\n")
            f.write(f"  value:       {field_data['value']}\n")
            f.write(f"  confidence:  {field_data['confidence']}\n")
            f.write(f"  method:      {field_data['method']}\n")

    return output_path


def process_file(file_path: str):
    print(f"\n{'═' * 65}")
    print(f"FILE : {Path(file_path).name}")
    print(f"{'═' * 65}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    contact = extract_contact(text)

    # ── Print to terminal ──
    print(f"\n{'─' * 60}")
    print(f"CONTACT INFO EXTRACTED FROM: {file_path}")
    print(f"{'─' * 60}")
    for field_name, field_data in contact.items():
        print(f"\n{field_name.upper()}")
        print(f"  value:       {field_data['value']}")
        print(f"  confidence:  {field_data['confidence']}")
        print(f"  method:      {field_data['method']}")

    # ── Save to contact_extracted_text/ ──
    saved_path = save_contact_output(file_path, contact)
    print(f"\n[Saved contact info to: {saved_path}]")


if __name__ == "__main__":
    # If a specific file is passed → run on that file only
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            process_file(path)
    else:
        # Otherwise run on ALL *_segmented.txt files in segmented_text/
        files = sorted(glob.glob(str(INPUT_DIR / "*_segmented.txt")))
        if not files:
            print(f"No *_segmented.txt files found in {INPUT_DIR}/ folder.")
            print("Run python test_segmenter.py first to generate them.")
        for f in files:
            process_file(f)



##working changing just to save the o/p under a folder
#"""
#test_contact_extractor.py
#A throwaway test script to manually check Layer 2's contact extractor
#against the .txt files saved from Layer 0 (in extracted_text/).
#
#Usage:
#    python test_contact_extractor.py
#"""
#
#from extractors.contact_extractor import extract_contact
#
#
## Change this to point at whichever saved .txt file you want to test.
##FILE_TO_TEST = "extracted_text/Anshika_Singh_Investment_Analytics_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Ankush_Bais_AI_Solution_Architect_Intileo.txt"
#FILE_TO_TEST = "extracted_text/Madan_Chawla_Investment_Analytics_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Ankush_Bais_AI_Solution_Architect_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Ankush_Bais_AI_Solution_Architect_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Ankush_Bais_AI_Solution_Architect_Intileo.txt"
#
#
#def main():
#    with open(FILE_TO_TEST, "r", encoding="utf-8") as f:
#        text = f.read()
#
#    contact = extract_contact(text)
#
#    print(f"\n{'─' * 60}")
#    print(f"CONTACT INFO EXTRACTED FROM: {FILE_TO_TEST}")
#    print(f"{'─' * 60}")
#
#    for field_name, field_data in contact.items():
#        print(f"\n{field_name.upper()}")
#        print(f"  value:       {field_data['value']}")
#        print(f"  confidence:  {field_data['confidence']}")
#        print(f"  method:      {field_data['method']}")
#
#
#if __name__ == "__main__":
#    main()























## changing for advanced version
#"""
#test_contact_extractor.py
#A throwaway test script to manually check Layer 2's contact extractor
#against the .txt files saved from Layer 0 (in extracted_text/).
#
#Usage:
#    python test_contact_extractor.py
#"""
#
#from extractors import extract_contact_info
#
#
## Change this to point at whichever saved .txt file you want to test.
#FILE_TO_TEST = "extracted_text/Anshika_Singh_Investment_Analytics_Intileo.txt"
#
#
#def main():
#    with open(FILE_TO_TEST, "r", encoding="utf-8") as f:
#        text = f.read()
#
#    contact = extract_contact_info(text)
#
#    print(f"\n{'─' * 60}")
#    print(f"CONTACT INFO EXTRACTED FROM: {FILE_TO_TEST}")
#    print(f"{'─' * 60}")
#
#    for field_name, field_data in contact.items():
#        print(f"\n{field_name.upper()}")
#        print(f"  value:       {field_data['value']}")
#        print(f"  confidence:  {field_data['confidence']}")
#        print(f"  method:      {field_data['method']}")
#        if len(field_data["all_matches"]) > 1:
#            print(f"  ALL matches: {field_data['all_matches']}")
#
#
#if __name__ == "__main__":
#    main()