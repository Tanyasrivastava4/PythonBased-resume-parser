"""
check_ocr_setup.py
━━━━━━━━━━━━━━━━━━━
Run this FIRST, before touching a real scanned resume, to catch broken
dependencies fast instead of debugging blind through a 30-second model load.

Usage:
    python check_ocr_setup.py
    python check_ocr_setup.py path/to/scanned_resume.pdf   # also runs
                                                             # per-page
                                                             # classification
                                                             # (no OCR yet,
                                                             # just tells you
                                                             # what pdf_reader
                                                             # WOULD do)
"""

import shutil
import sys


def check_tesseract():
    print("── Tesseract (system binary + pytesseract) ──")
    binary = shutil.which("tesseract")
    if not binary:
        print("  ✗ tesseract binary NOT found on PATH.")
        print("    pytesseract is just a wrapper — it needs the actual")
        print("    tesseract-ocr program installed separately.")
        print("    Install: sudo apt-get install tesseract-ocr  (Linux)")
        print("             brew install tesseract              (Mac)")
        return False
    print(f"  ✓ tesseract binary found: {binary}")

    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"  ✓ pytesseract importable, tesseract version: {version}")
        return True
    except ImportError:
        print("  ✗ pytesseract not installed. pip install pytesseract")
        return False
    except Exception as e:
        print(f"  ✗ pytesseract found tesseract but errored: {e}")
        return False


def check_surya():
    print("\n── Surya OCR ──")
    try:
        import surya
        version = getattr(surya, "__version__", "unknown")
        print(f"  surya-ocr installed, version: {version}")
    except ImportError:
        print("  surya-ocr not installed at all.")
        print("  (Fine if you're going with the pytesseract-only path.)")
        return False

    try:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        print("  ✓ Predictor-class API (surya-ocr 0.14.x, matches your")
        print("    current ocr_reader.py) is importable.")
        return True
    except ImportError as e:
        print(f"  ✗ Predictor-class API NOT importable: {e}")
        print("    Your installed surya-ocr version doesn't match the API")
        print("    ocr_reader.py targets. Run: pip show surya-ocr")
        print("    and tell Claude the exact version — Surya has changed")
        print("    its API at least 3 times across releases, so ocr_reader.py")
        print("    needs to target whatever's actually installed.")
        print("    Meanwhile the pytesseract fallback in ocr_reader.py will")
        print("    handle scanned pages automatically.")
        return False


def check_pdf_libs():
    print("\n── PDF libraries ──")
    ok = True
    try:
        import fitz
        print(f"  ✓ PyMuPDF (fitz) importable, version: {fitz.__doc__ or fitz.VersionBind}")
    except ImportError:
        print("  ✗ PyMuPDF not installed. pip install PyMuPDF")
        ok = False
    try:
        import pdfplumber
        print(f"  ✓ pdfplumber importable")
    except ImportError:
        print("  ✗ pdfplumber not installed. pip install pdfplumber")
        ok = False
    try:
        from PIL import Image
        print(f"  ✓ Pillow importable")
    except ImportError:
        print("  ✗ Pillow not installed. pip install Pillow")
        ok = False
    return ok


def classify_pdf_pages(pdf_path: str):
    print(f"\n── Per-page classification for: {pdf_path} ──")
    print("(This mirrors exactly what pdf_reader.read_text_pdf() decides")
    print(" to do with each page — no OCR is actually run here yet.)\n")
    try:
        import fitz
        import pdfplumber
        from ingestion.ocr_region import classify_page
    except ImportError as e:
        print(f"  Could not import required modules: {e}")
        print("  Run this script from your ats_parser/ project root.")
        return

    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as fdoc:
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            fpage = fdoc[i]
            char_count = len(fpage.get_text().strip())
            words = []
            if char_count >= 20:
                words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False)
            classification = classify_page(fpage, native_words=words)
            print(f"  Page {i + 1}: char_count={char_count:>5}  →  classified as: {classification}")


if __name__ == "__main__":
    tesseract_ok = check_tesseract()
    surya_ok = check_surya()
    libs_ok = check_pdf_libs()

    print("\n" + "═" * 60)
    if not tesseract_ok:
        print("Fix tesseract first — both the hybrid-region OCR (already")
        print("working in your pipeline) and any pytesseract fallback")
        print("depend on it.")
    elif not surya_ok:
        print("Surya's API is broken for your code. Either pin")
        print("surya-ocr==0.6.13, or switch to the pytesseract full-page")
        print("OCR module — recommended, since tesseract already works")
        print("and you avoid the version-pinning trap entirely.")
    else:
        print("Core OCR dependencies look fine.")
    print("═" * 60)

    if len(sys.argv) > 1:
        classify_pdf_pages(sys.argv[1])
    else:
        print("\n(Pass a PDF path as an argument to also see per-page")
        print(" classification: python check_ocr_setup.py resumes/foo.pdf)")