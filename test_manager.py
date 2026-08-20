"""
Test runner — run this to show your manager the output.
Usage: python test_manager.py
Put your PDFs in the resumes/ folder.

This now also SAVES each resume's extracted text into the
extracted_text/ folder, as a .txt file, so Layer 1 (the section
segmenter) has real saved text to work on instead of text that
only ever appeared in the terminal and then disappeared.
"""

import sys
import glob
from pathlib import Path
from ingestion import read_resume_file


# Folder where we will save every resume's extracted text.
# Path(...) just represents a folder location; nothing is created yet.
OUTPUT_DIR = Path("extracted_text")


def save_extracted_text(file_path: str, text: str):
    """
    Saves the extracted text to a .txt file inside extracted_text/,
    using the same name as the original resume file (but ending in .txt
    instead of .pdf/.docx/etc).
    """
    # mkdir = "make directory" (make a folder).
    # parents=True means "also create any missing parent folders along the way".
    # exist_ok=True means "don't raise an error if the folder already exists".
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Path(file_path).stem gives just the filename without its extension.
    # e.g. "resumes/Anshika_Singh.pdf" -> stem is "Anshika_Singh"
    original_name = Path(file_path).stem
    output_path = OUTPUT_DIR / f"{original_name}.txt"

    # "w" means "open this file for writing" (creates it if it doesn't
    # exist, overwrites it if it does). encoding="utf-8" makes sure
    # special characters (like the – dash, or ’ apostrophe we saw
    # earlier) are saved correctly instead of causing errors.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_path


def test_file(file_path: str):
    print(f"\n{'═' * 70}")
    print(f"FILE: {Path(file_path).name}")
    print(f"{'═' * 70}")

    try:
        text = read_resume_file(file_path)
        print(text)
        print(f"\n[Total characters extracted: {len(text):,}]")

        saved_path = save_extracted_text(file_path, text)
        print(f"[Saved extracted text to: {saved_path}]")
    except Exception as e:
        print(f"ERROR: {e}")

    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            test_file(path)
    else:
        #files = sorted(
        #    glob.glob("resumes/*.pdf")  +
        #    glob.glob("resumes/*.docx") +
        #    glob.glob("resumes/*.jpg")  +
        #    glob.glob("resumes/*.png")
        #)
        files = sorted(
            glob.glob("resumes/*.pdf")  +
            glob.glob("resumes/*.docx") +
            glob.glob("resumes/*.doc")  +  
            glob.glob("resumes/*.jpg")  +
            glob.glob("resumes/*.png")
        )
        if not files:
            print("No files found in resumes/ folder.")
            print("Usage: python test_manager.py path/to/resume.pdf")
        for f in files:
            test_file(f)























#"""
#Test runner — run this to show your manager the output.
#Usage: python test_manager.py
#Put your PDFs in the resumes/ folder.
#"""
#
#import sys
#import glob
#from pathlib import Path
#from ingestion import read_resume_file
#
#
#def test_file(file_path: str):
#    print(f"\n{'═' * 70}")
#    print(f"FILE: {Path(file_path).name}")
#    print(f"{'═' * 70}")
#
#    try:
#        text = read_resume_file(file_path)
#        print(text)
#        print(f"\n[Total characters extracted: {len(text):,}]")
#    except Exception as e:
#        print(f"ERROR: {e}")
#
#    print(f"{'═' * 70}\n")
#
#
#if __name__ == "__main__":
#    if len(sys.argv) > 1:
#        for path in sys.argv[1:]:
#            test_file(path)
#    else:
#        files = sorted(
#            glob.glob("resumes/*.pdf")  +
#            glob.glob("resumes/*.docx") +
#            glob.glob("resumes/*.jpg")  +
#            glob.glob("resumes/*.png")
#        )
#        if not files:
#            print("No files found in resumes/ folder.")
#            print("Usage: python test_manager.py path/to/resume.pdf")
#        for f in files:
#            test_file(f)























#"""
#Test runner — run this to show your manager the output.
#Usage: python test_manager.py
#Put your PDFs in the resumes/ folder.
#"""
#
#import sys
#import glob
#from pathlib import Path
#from ingestion import read_resume_file
#
#
#def test_file(file_path: str):
#    print(f"\n{'═' * 70}")
#    print(f"FILE: {Path(file_path).name}")
#    print(f"{'═' * 70}")
#
#    try:
#        text = read_resume_file(file_path)
#        print(text)
#        print(f"\n[Total characters extracted: {len(text):,}]")
#    except Exception as e:
#        print(f"ERROR: {e}")
#
#    print(f"{'═' * 70}\n")
#
#
#if __name__ == "__main__":
#    if len(sys.argv) > 1:
#        for path in sys.argv[1:]:
#            test_file(path)
#    else:
#        files = sorted(
#            glob.glob("resumes/*.pdf")  +
#            glob.glob("resumes/*.docx") +
#            glob.glob("resumes/*.jpg")  +
#            glob.glob("resumes/*.png")
#        )
#        if not files:
#            print("No files found in resumes/ folder.")
#            print("Usage: python test_manager.py path/to/resume.pdf")
#        for f in files:
#            test_file(f)


















#"""
#Test runner — run this to show your manager the output.
#Usage: python test_manager.py
#Put your PDFs in the resumes/ folder.
#"""
#
#import sys
#import glob
#from pathlib import Path
#from ingestion import read_resume_file
#
#def test_file(file_path: str):
#    print(f"\n{'═'*70}")
#    print(f"FILE: {Path(file_path).name}")
#    print(f"{'═'*70}")
#
#    try:
#        text = read_resume_file(file_path)
#        print(text)
#        print(f"\n[Total characters extracted: {len(text):,}]")
#    except Exception as e:
#        print(f"ERROR: {e}")
#
#    print(f"{'═'*70}\n")
#
#
#if __name__ == "__main__":
#    if len(sys.argv) > 1:
#        for path in sys.argv[1:]:
#            test_file(path)
#    else:
#        files = sorted(
#            glob.glob("resumes/*.pdf") +
#            glob.glob("resumes/*.docx") +
#            glob.glob("resumes/*.jpg") +
#            glob.glob("resumes/*.png")
#        )
#        if not files:
#            print("No files found in resumes/ folder.")
#            print("Usage: python test_manager.py path/to/resume.pdf")
#        for f in files:
#            test_file(f)