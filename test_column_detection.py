"""
Quick test: verifies the header-aware two-column retry,
WITHOUT running Surya OCR. Uses pdfplumber native words only.
"""
import pdfplumber
from ingestion.pdf_reader import _is_true_two_column

PDF_PATH = "resumes/Rishi Bakshi_Full stack web security patches_Intileo_page-0001.pdf"

def word_dict(w):
    return {"text": w["text"], "x0": w["x0"], "x1": w["x1"],
            "top": w["top"], "bottom": w["bottom"]}

with pdfplumber.open(PDF_PATH) as pdf:
    for page_num, page in enumerate(pdf.pages):
        raw_words = page.extract_words(x_tolerance=3, y_tolerance=3)
        words = [word_dict(w) for w in raw_words]
        page_width = page.width

        print(f"\n{'='*60}")
        print(f"PAGE {page_num+1}  (width={page_width:.1f}, total words={len(words)})")
        print(f"{'='*60}")

        r1 = _is_true_two_column(words, page_width, header_aware=False)
        print(f"Pass 1 (original)     -> {'TWO-COLUMN DETECTED' if r1 else 'NOT detected'}")

        r2 = _is_true_two_column(words, page_width, header_aware=True)
        print(f"Pass 2 (header-aware) -> {'TWO-COLUMN DETECTED' if r2 else 'NOT detected'}")

        if r2 and not r1:
            left, right = r2
            print(f"\n  FIX FIRED: left={len(left)} words, right={len(right)} words")
            lsorted = sorted(left,  key=lambda w: (w['top'], w['x0']))
            rsorted = sorted(right, key=lambda w: (w['top'], w['x0']))
            print(f"  First 5 LEFT  words: {[w['text'] for w in lsorted[:5]]}")
            print(f"  First 5 RIGHT words: {[w['text'] for w in rsorted[:5]]}")
        elif r1:
            left, right = r1
            print(f"  Pass1 OK: left={len(left)}, right={len(right)} words")

print("\nDone.")
