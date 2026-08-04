"""
test_segmenter.py
━━━━━━━━━━━━━━━━━
Tests Layer 1 (section segmenter) against ALL .txt files in extracted_text/.
Saves the segmented output for each resume into segmented_text/ folder.

Usage:
    # Run on ALL resumes:
    python test_segmenter.py

    # Run on ONE specific resume:
    python test_segmenter.py extracted_text/Ankush_Bais_AI_Solution_Architect_Intileo.txt
"""
#python build_skills_taxonomy.py --onet "Software Skills.txt" --output data/skills_taxonomy.json
#python test_segmenter.py extracted_text/Ankush_Bais_AI_Solution_Architect_Intileo.txt


import sys
import glob
from pathlib import Path
from segmenter import split_into_sections, get_section_text


# ── Folder where segmented output will be saved ──
OUTPUT_DIR = Path("segmented_text")


def save_segmented_output(file_path: str, sections: list) -> Path:
    """
    Saves the segmented section output to segmented_text/ folder.
    Each resume gets its own .txt file with sections clearly labelled.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(file_path).stem
    output_path   = OUTPUT_DIR / f"{original_name}_segmented.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for section in sections:
            f.write(f"\n{'─' * 60}\n")
            f.write(f"LABEL: {section.label}   "
                    f"CONFIDENCE: {round(section.confidence, 2)}   "
                    f"START LINE: {section.start_line}\n")
            f.write(f"{'─' * 60}\n")
            f.write(section.raw_text)
            f.write("\n")

        labels_found = [s.label for s in sections]
        f.write(f"\n{'─' * 60}\n")
        f.write(f"[Found {len(sections)} sections total: {labels_found}]\n")

    return output_path


def process_file(file_path: str):
    print(f"\n{'═' * 65}")
    print(f"FILE : {Path(file_path).name}")
    print(f"{'═' * 65}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = split_into_sections(text)

    # ── Print to terminal ──
    for section in sections:
        print(f"\n{'─' * 60}")
        print(f"LABEL: {section.label}   "
              f"CONFIDENCE: {round(section.confidence, 2)}   "
              f"START LINE: {section.start_line}")
        print(f"{'─' * 60}")
        print(section.raw_text)

    labels_found = [s.label for s in sections]
    print(f"\n{'─' * 60}")
    print(f"[Found {len(sections)} sections total: {labels_found}]")

    experience_text = get_section_text(sections, "experience")
    if experience_text:
        print(f"[Combined 'experience' text length: {len(experience_text)} characters]")

    # ── Save to segmented_text/ ──
    saved_path = save_segmented_output(file_path, sections)
    print(f"[Saved segmented output to: {saved_path}]")


if __name__ == "__main__":
    # If a specific file is passed → run on that file only
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            process_file(path)
    else:
        # Otherwise run on ALL .txt files in extracted_text/
        files = sorted(glob.glob("extracted_text/*.txt"))
        if not files:
            print("No .txt files found in extracted_text/ folder.")
            print("Run python test_manager.py first to generate them.")
        for f in files:
            process_file(f)






####just changing for to save the output in a structured way
#"""
#test_segmenter.py
#A throwaway test script to manually check Layer 1's section segmenter
#against the .txt files saved from Layer 0 (in extracted_text/).
#
#Usage:
#    python test_segmenter.py
#
#Note: this version's split_into_sections() returns a LIST of Section
#objects (not a plain dictionary like an earlier draft did). Each Section
#has .label, .raw_text, .start_line, and .confidence -- the confidence
#score tells us how sure the segmenter is about each section boundary,
#which matters for later layers (low-confidence sections can be flagged
#for review instead of trusted blindly).
#"""
#
#from segmenter import split_into_sections, get_section_text
#
#
## Change this to point at whichever saved .txt file you want to test.
##FILE_TO_TEST = "extracted_text/Anshika_Singh_Investment_Analytics_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Madan_Chawla_Investment_Analytics_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Ankush_Bais_AI_Solution_Architect_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Shruti_Pandey_Investment_Analytics_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Rishabh_Rai_Assistant_Manager-Cyber_Security_SMFG-Intileo.txt"
##FILE_TO_TEST = "extracted_text/Umesh_Sihag_.Net_Core_Developer_Intileo.txt"
#
## Change this line in test_segmenter.py one at a time:
##FILE_TO_TEST = "extracted_text/Rishabh_Rai_Assistant_Manager-Cyber_Security_SMFG-Intileo.txt"
##FILE_TO_TEST = "extracted_text/Satyadip_Ray_Sr._Business_Development_Executive_Intileo.txt"
#
##FILE_TO_TEST = "extracted_text/Madan_Chawla_Investment_Analytics_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Aarti Suranje_Vice President- Business Solutions Group (Oracle Fusion)_SMFG-Intileo.txt"
#
##FILE_TO_TEST = "extracted_text/Aastha_main (1) (1).txt"
#
#FILE_TO_TEST = "extracted_text/Abhinav Srivastav_Mobile App Developer_GT.txt"
#
#def main():
#    with open(FILE_TO_TEST, "r", encoding="utf-8") as f:
#        text = f.read()
#
#    sections = split_into_sections(text)
#
#    for section in sections:
#        print(f"\n{'─' * 60}")
#        print(f"LABEL: {section.label}   "
#              f"CONFIDENCE: {round(section.confidence, 2)}   "
#              f"START LINE: {section.start_line}")
#        print(f"{'─' * 60}")
#        print(section.raw_text)
#
#    print(f"\n{'─' * 60}")
#    labels_found = [s.label for s in sections]
#    print(f"[Found {len(sections)} sections total: {labels_found}]")
#
#    # Example of using get_section_text() to fetch one section by name,
#    # combining duplicates if the same label appeared more than once.
#    experience_text = get_section_text(sections, "experience")
#    if experience_text:
#        print(f"\n[Combined 'experience' text length: {len(experience_text)} characters]")
#
#
#if __name__ == "__main__":
#    main()























##worked with our version but change for advanced version
#"""
#test_segmenter.py
#A throwaway test script (just like test_manager.py) to manually check
#Layer 1's section segmenter against the .txt files we already saved
#from Layer 0 (Layer 0's output lives in the extracted_text/ folder).
#
#Usage:
#    python test_segmenter.py
#"""
#
#from segmenter import split_into_sections
#
#
## Change this to point at whichever saved .txt file you want to test.
##FILE_TO_TEST = "extracted_text/Anshika_Singh_Investment_Analytics_Intileo.txt"
##FILE_TO_TEST = "extracted_text/Madan_Chawla_Investment_Analytics_Intileo.txt"
#FILE_TO_TEST = "extracted_text/Shruti_Pandey_Investment_Analytics_Intileo.txt"
#
#
#def main():
#    with open(FILE_TO_TEST, "r", encoding="utf-8") as f:
#        text = f.read()
#
#    sections = split_into_sections(text)
#
#    for section_name, content in sections.items():
#        print(f"\n{'─' * 60}")
#        print(f"SECTION: {section_name}")
#        print(f"{'─' * 60}")
#        print(content)
#
#    print(f"\n{'─' * 60}")
#    print(f"[Found {len(sections)} sections: {list(sections.keys())}]")
#    print(f"{'─' * 60}")
#
#
#if __name__ == "__main__":
#    main()