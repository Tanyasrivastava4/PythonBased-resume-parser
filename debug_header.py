"""
debug_header.py — Run this on the problem resume to trace where the header goes.

Usage:
    python debug_header.py resumes/Aman_Chauhan*.pdf

This will show exactly:
  1. What words pdfplumber extracts (first 20, with coordinates)
  2. What classify_page() returns
  3. Whether a header gap is detected
  4. What image blocks exist on page 1
  5. Whether sidebar/two-column detection fires
  6. What the final text looks like (first 500 chars)
"""

import sys
import pdfplumber
import fitz
from pathlib import Path
from collections import Counter


def debug_pdf(pdf_path: str):
    print(f"\n{'═' * 70}")
    print(f"DEBUGGING: {Path(pdf_path).name}")
    print(f"{'═' * 70}")

    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as fdoc:
        page = pdf.pages[0]
        fpage = fdoc[0]

        # ── 1. Basic page info ──
        print(f"\n[1] PAGE INFO")
        print(f"    Page size: {page.width:.1f} x {page.height:.1f} pt")
        print(f"    fitz char count: {len(fpage.get_text().strip())}")
        print(f"    fitz first 300 chars:")
        fitz_text = fpage.get_text().strip()[:300]
        for line in fitz_text.split("\n"):
            print(f"      | {line}")

        # ── 2. pdfplumber words ──
        words = page.extract_words(
            x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False,
        )
        print(f"\n[2] PDFPLUMBER WORDS (first 30 of {len(words)} total)")
        for i, w in enumerate(words[:30]):
            print(f"    [{i:3d}] x0={w['x0']:6.1f}  top={w['top']:6.1f}  "
                  f"bottom={w['bottom']:6.1f}  text='{w['text']}'")

        # ── 3. Header gap check ──
        if words:
            first_top = min(w["top"] for w in words)
            gap = first_top - fpage.rect.y0
            print(f"\n[3] HEADER GAP CHECK")
            print(f"    Page top (y0): {fpage.rect.y0:.1f}")
            print(f"    First word top: {first_top:.1f}")
            print(f"    Gap: {gap:.1f} pt  (threshold: 100 pt)")
            print(f"    Header gap detected: {'YES' if gap > 100 else 'NO'}")

        # ── 4. Image blocks ──
        blocks = fpage.get_text("dict")["blocks"]
        img_blocks = [b for b in blocks if b.get("type") == 1]
        print(f"\n[4] IMAGE BLOCKS on page 1: {len(img_blocks)}")
        for i, b in enumerate(img_blocks):
            x0, y0, x1, y1 = b["bbox"]
            w, h = x1 - x0, y1 - y0
            area_ratio = (w * h) / (page.width * page.height)
            print(f"    [{i}] bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})  "
                  f"size={w:.0f}x{h:.0f}  area_ratio={area_ratio:.2%}")

        # ── 5. classify_page ──
        from ingestion.ocr_region import classify_page, get_ocr_worthy_image_blocks
        classification = classify_page(fpage, native_words=words)
        print(f"\n[5] CLASSIFY PAGE: '{classification}'")

        ocr_worthy = get_ocr_worthy_image_blocks(fpage, native_words=words)
        print(f"    OCR-worthy image blocks: {len(ocr_worthy)}")
        for i, rect in enumerate(ocr_worthy):
            print(f"    [{i}] {rect}")

        # ── 6. Layout detection ──
        from ingestion.pdf_reader import _find_sidebar_split, _is_true_two_column

        sidebar_split = _find_sidebar_split(words, page.width)
        print(f"\n[6] LAYOUT DETECTION")
        print(f"    Sidebar split: {sidebar_split}")

        if not sidebar_split:
            two_col = _is_true_two_column(words, page.width)
            print(f"    Two-column: {'YES' if two_col else 'NO'}")
            if two_col:
                left, right = two_col
                print(f"    Left words: {len(left)}, Right words: {len(right)}")

        # ── 7. X-position distribution ──
        print(f"\n[7] X0 POSITION DISTRIBUTION (top 10)")
        x0_counter = Counter(round(w["x0"]) for w in words)
        for x0, count in x0_counter.most_common(10):
            print(f"    x0={x0:4d}  count={count}")

        # ── 8. Words in top 150pt ──
        top_words = [w for w in words if w["top"] < 150]
        print(f"\n[8] WORDS IN TOP 150pt: {len(top_words)}")
        for w in sorted(top_words, key=lambda w: (w["top"], w["x0"])):
            print(f"    top={w['top']:6.1f}  x0={w['x0']:6.1f}  text='{w['text']}'")

        # ── 9. Final extracted text (first 500 chars) ──
        from ingestion.pdf_reader import read_text_pdf
        final_text = read_text_pdf(pdf_path)
        print(f"\n[9] FINAL read_text_pdf() OUTPUT (first 500 chars):")
        print(f"{'─' * 50}")
        print(final_text[:500])
        print(f"{'─' * 50}")
        print(f"    Total length: {len(final_text)}")

    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        debug_pdf(path)