"""
PDF Reader — Layer 0

Reads a PDF file and returns clean text.
Per PAGE (not per document), classifies the page as one of:
 "scanned" -> essentially no extractable text; OCR'd via Surya
 "hybrid"  -> has real text AND embedded image regions with text
              (e.g. a contact bar rendered as one flattened graphic);
              OCR's just those regions and merges the recognised
              words into the same word stream native text uses
 "text"    -> normal page, handled exactly as before

REVISION (earlier) — replaced the old whole-document
is_scanned_pdf() gate with per-page classification. That old check
combined the first 3 pages' text and returned one True/False for the
entire file, so it could not handle:
 - a document where page 1 is a scanned image and page 2 is real text
   (or vice versa)
 - a page that is mostly real text but has one embedded image
   (typically a contact bar / icon+label row) that also carries text

Confirmed on a real resume (Adarsh Bhagat): the entire phone/email/
LinkedIn row was a single flattened raster image sitting between the
name and the "Summary" heading. pdfplumber's raw page.chars had zero
characters there, and PyMuPDF's get_text("dict") showed it as an
image block (type=1), not a text block -- so no text-extraction
approach could have recovered it without OCR. classify_page() detects
this ("hybrid"), get_ocr_worthy_image_blocks() finds the image
region, and ocr_region_to_words() OCRs just that crop and returns
pdfplumber-shaped word dicts that get merged into `words` BEFORE
layout detection runs -- so the recovered text flows through the
existing sidebar/two-column/single-column logic unchanged and lands
in the correct reading-order position.

Layout handling within a "text" or "hybrid" page (unchanged from
before): sidebar layout, true two-column layout, single column
fallback, in that order.

MERGE NOTE (carried over from the previous revision, still applies):
 Two-column pages split into a `primary_parts` stream (left column /
 main narrative, contiguous across page breaks) and a
 `secondary_parts` stream (right column short lists), joined
 primary-then-secondary once at the very end. See the docstrings on
 _is_true_two_column() and read_text_pdf() below for the full
 reasoning; unchanged from the prior version of this file.

HEADER-GAP FIX (earlier):
 Many designed/styled resume templates (Canva, Zety, resume-builder
 PDFs) render the header area (name, title, contact row) as a single
 flattened raster image, as vector-path outlines (drawn shapes, not
 character data), or inside annotation/form layers that pdfplumber's
 content-stream reader never touches. In all three cases,
 pdfplumber.extract_words() returns zero words for that region, and
 the page is classified "text" (because the body below has plenty of
 real text). The header simply vanishes from the output.

 Fixed by detecting the "header gap": if the first native word on a
 page starts significantly below the page top (>100pt, roughly 1.4
 inches), there is likely an unextracted header region above it. That
 gap is OCR'd with pytesseract (PSM=4, single column with variable
 text sizes — correct for a multi-line header with large name +
 smaller subtitle + small contact row) and the recovered words are
 prepended to the word list before layout detection runs, so they
 flow through the existing sidebar/two-column/single-column logic
 unchanged.

 This is format-agnostic: it works regardless of WHY the header text
 is missing (image, vector paths, annotations, form objects), because
 pytesseract renders the visual page region to a raster and reads
 whatever is visible there.

 REQUIRES ocr_region.py's ocr_region_to_words() to accept a `psm`
 keyword argument (added specifically to support this call, which
 needs PSM 4 for a multi-line variable-size header, unlike this
 function's other call site below which uses the default PSM 6 for
 single-line contact bars). If you see
 "TypeError: unexpected keyword argument 'psm'" here, ocr_region.py
 needs that parameter added -- it is NOT optional for this fix to work.

PAGE-DEDUP FIX (earlier):
 Some resume-builder PDFs include duplicate pages (e.g. a "display"
 page and a "print" page with identical content). Both pages get
 processed, producing double output. Fixed by deduplicating
 consecutive identical entries in primary_parts / secondary_parts
 before joining.

NOTE ON SCANNED-PAGE OCR: this file is intentionally NOT shared with
the OCR modules (ocr_reader.py / ocr_reader_pytesseract.py). Scanned-
page layout reconstruction has its own separate, independent copy of
this logic in ingestion/ocr_layout_reconstruction.py, which is free
to diverge and be tuned for OCR-specific quirks (e.g. tightly-packed
scanned tables needed a different heading-merge tolerance than resume
sidebars do) WITHOUT any risk of that tuning affecting this file's
already-proven behavior on native PDF/DOCX text.

NOTE ON THE PRE-OCR IMAGE QUALITY GATE (blur rejection / deskew /
brightness correction, added in ocr_reader.py and
ocr_reader_pytesseract.py): this file does NOT need any changes for
that either. This file never touches image pixels itself -- it only
calls read_with_surya_pages() (imported below) and receives back a
plain {page_index: text} dict. If a scanned page fails the blur gate,
read_with_surya_pages() raises ImageQualityError, which is not caught
anywhere in this file, so it propagates naturally up through
read_text_pdf() -> read_pdf() -> read_resume_file() to the caller
(test_manager.py already prints any such exception via its existing
`except Exception as e: print(f"ERROR: {e}")`, so no changes are
needed there either). This holds true regardless of which OCR engine
import is active below.
"""

import re
import pdfplumber
import fitz  # PyMuPDF
from pathlib import Path
from collections import Counter

from ingestion.ocr_region import (
    classify_page,
    get_ocr_worthy_image_blocks,
    ocr_region_to_words,
)

# ── OCR ENGINE SELECTION ──
# Production default: Surya primary, with its own automatic fallback to
# pytesseract on failure (see ocr_reader.py's read_with_surya_pages()).
from ingestion.ocr_reader import read_with_surya_pages

# TESTING OVERRIDE: to force pytesseract only (bypassing Surya entirely,
# e.g. to compare output quality between engines), comment out the
# import above and uncomment the one below instead. Note this is a
# manual, all-or-nothing override -- it does NOT go through Surya's own
# try-Surya-then-fall-back-to-tesseract logic, it skips Surya entirely.
# The pre-OCR quality gate (blur/deskew/brightness) still runs correctly
# either way, since both ocr_reader.py and ocr_reader_pytesseract.py
# call it identically.
# from ingestion.ocr_reader_pytesseract import read_with_surya_pages


# ──────────────────────────────────────────────────────────────
# HEADER-GAP DETECTION
# ──────────────────────────────────────────────────────────────

def _detect_header_gap(words: list, page_rect, min_gap: float = 100.0):
    """
    Checks whether native words start well below the top of the page,
    indicating an unextracted header region (image, vector paths, or
    annotation-layer text that pdfplumber cannot see).

    Returns a fitz.Rect covering the gap (full page width, from page
    top to the first word's vertical position) if a gap is detected,
    or None if no gap exists.

    min_gap: minimum vertical distance (in PDF points, 72pt = 1 inch)
      between the page top and the first word's top to trigger header
      gap detection. Default 100pt (~1.4 inches) avoids false positives
      from normal top margins (typically 36–72pt) while catching
      designed-template headers that take up 1.5–3 inches at the top.

    The returned Rect is passed to ocr_region_to_words() which renders
    that region at high resolution and OCRs it. This works regardless
    of WHY the text is unextractable — the OCR sees whatever is
    visually rendered in that region.
    """
    if not words:
        return None

    first_word_top = min(w["top"] for w in words)
    gap = first_word_top - page_rect.y0

    if gap > min_gap:
        return fitz.Rect(
            page_rect.x0,       # full page width (left edge)
            page_rect.y0,       # page top
            page_rect.x1,       # full page width (right edge)
            first_word_top      # stop at first native word
        )
    return None


# ──────────────────────────────────────────────────────────────
# PAGE-DEDUP HELPER
# ──────────────────────────────────────────────────────────────

def _dedup_consecutive(parts: list) -> list:
    """
    Removes consecutive identical entries from a list of page text
    contributions. Handles the case where a resume-builder PDF has
    duplicate pages (e.g. display + print copies) that would otherwise
    produce double output.

    Uses stripped-text comparison to ignore minor whitespace differences.
    """
    if len(parts) <= 1:
        return parts
    result = [parts[0]]
    for p in parts[1:]:
        if p.strip() != result[-1].strip():
            result.append(p)
    return result


# ──────────────────────────────────────────────────────────────
# SIDEBAR LAYOUT DETECTION
# ──────────────────────────────────────────────────────────────

def _find_sidebar_split(words: list, page_width: float):
    """
    Detects if a page has a sidebar layout — a narrow left column
    containing ONLY section headings, and a wide right column with content.
    Returns the x-coordinate of the split point if sidebar detected,
    or None if not a sidebar layout.
    """
    if not words:
        return None

    x0_counter = Counter(round(w["x0"]) for w in words)
    content_start_x = None
    best_count = 0

    for x0, count in x0_counter.items():
        if 80 < x0 < page_width * 0.45 and count > best_count:
            best_count = count
            content_start_x = x0

    if content_start_x is None or best_count < 3:
        return None

    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]

    if not sidebar_candidates:
        return None
    if len(sidebar_candidates) > 25:
        return None

    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
    heading_boundary = sidebar_x0s[-1]
    for i in range(len(sidebar_x0s) - 1):
        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
        if inner_gap > 30:
            heading_boundary = sidebar_x0s[i]
            break

    gap = content_start_x - heading_boundary
    if gap < 30:
        return None

    split = (heading_boundary + content_start_x) / 2
    return split


def _merge_sidebar_with_content(sidebar_words: list, content_words: list) -> str:
    """
    Merges sidebar heading words with content words by vertical position.
    """
    heading_lines = {}
    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))

    current_top = None
    current_words = []
    for word in sorted_sidebar:
        if current_top is None:
            current_top = word["top"]
            current_words = [word]
        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 1.5:
            current_words.append(word)
            current_top = word["top"]
        else:
            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
            heading_lines[round(current_top)] = heading_text
            current_top = word["top"]
            current_words = [word]
    if current_words:
        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
        heading_lines[round(current_top)] = heading_text

    content_line_list = []
    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))

    current_top = None
    current_words = []
    for word in sorted_content:
        if current_top is None:
            current_top = word["top"]
            current_words = [word]
        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
            current_words.append(word)
        else:
            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
            content_line_list.append((round(current_top), line_text))
            current_top = word["top"]
            current_words = [word]
    if current_words:
        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
        content_line_list.append((round(current_top), line_text))

    result_lines = []
    used_headings = set()

    for content_top, content_text in content_line_list:
        for heading_top, heading_text in sorted(heading_lines.items()):
            if heading_top in used_headings:
                continue
            if abs(heading_top - content_top) <= 20:
                result_lines.append(heading_text)
                used_headings.add(heading_top)
                break
        result_lines.append(content_text)

    for heading_top, heading_text in sorted(heading_lines.items()):
        if heading_top not in used_headings:
            result_lines.append(heading_text)

    return "\n".join(result_lines)


# ──────────────────────────────────────────────────────────────
# TWO-COLUMN LAYOUT DETECTION
# ──────────────────────────────────────────────────────────────

def _find_column_split(words: list, page_width: float):
    """
    Finds the actual column split point for a two-column layout, using
    the gap in x0 start positions rather than assuming the page midpoint.
    """
    if not words:
        return None

    x0_counter = Counter(round(w["x0"]) for w in words)
    all_x0 = sorted(x0_counter.keys())

    if len(all_x0) < 2:
        return None

    best_split = None
    best_score = 0

    for i in range(len(all_x0) - 1):
        gap = all_x0[i + 1] - all_x0[i]
        if gap < 15:
            continue

        split_x = (all_x0[i] + all_x0[i + 1]) / 2
        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])

        if left_count < 20 or right_count < 20:
            continue

        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
            continue

        balance = min(left_count, right_count) / max(left_count, right_count)
        score = gap * balance

        if score > best_score:
            best_score = score
            best_split = split_x

    if best_split is None:
        return None

    left_words = [w for w in words if w["x1"] <= best_split]
    right_words = [w for w in words if w["x0"] > best_split]
    return best_split, left_words, right_words


def _is_true_two_column(words: list, page_width: float):
    """
    Determines whether a page genuinely has a two-column layout, AND
    returns the correctly split word groups if so.
    Requires: both sides substantial (>=30 words), near-empty gutter
    (<5 straddling words), right column spans >=15% of left column's
    vertical height.
    """
    result = _find_column_split(words, page_width)
    if result is None:
        return None

    split_x, left_words, right_words = result

    if len(left_words) < 30 or len(right_words) < 30:
        return None

    gutter_words = [
        w for w in words
        if w["x0"] < split_x - 2 and w["x1"] > split_x + 2
    ]
    if len(gutter_words) > 5:
        return None

    if right_words and left_words:
        right_top = min(w["top"] for w in right_words)
        right_bottom = max(w["bottom"] for w in right_words)
        right_span = right_bottom - right_top

        left_top = min(w["top"] for w in left_words)
        left_bottom = max(w["bottom"] for w in left_words)
        left_span = left_bottom - left_top

        if left_span > 0 and right_span / left_span < 0.15:
            return None

    return left_words, right_words


# ──────────────────────────────────────────────────────────────
# LINE RECONSTRUCTION
# ──────────────────────────────────────────────────────────────

def _extract_words_to_lines(words: list) -> str:
    """
    Reconstructs text lines from word dicts, grouping by vertical (top)
    position and joining with a single space. Works identically whether
    a word came from pdfplumber's native extraction or from
    ocr_region_to_words() -- both use the same dict shape.
    """
    if not words:
        return ""

    lines: list = []
    current_line: list = []
    current_top = None

    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))

    for word in sorted_words:
        if current_top is None:
            current_top = word["top"]
            current_line = [word]
            continue

        line_tolerance = max(word["height"], 1) * 0.5
        if abs(word["top"] - current_top) <= line_tolerance:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
            current_top = word["top"]

    if current_line:
        lines.append(current_line)

    text_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["x0"])
        parts = [w["text"] for w in line_sorted]
        text_lines.append(" ".join(parts))

    return "\n".join(text_lines)


# ──────────────────────────────────────────────────────────────
# MAIN PDF TEXT READER
# ──────────────────────────────────────────────────────────────

def read_text_pdf(pdf_path: str) -> str:
    """
    Reads a PDF page by page.
    PASS 1: classify every page ("scanned" / "hybrid" / "text").
    Collect the indices of "scanned" pages.
    PASS 1.5: if any pages were classified "scanned", batch-OCR just
    those pages in ONE call via read_with_surya_pages(), so
    model loading happens at most once per document regardless of
    how many scanned pages it has. Build a {page_index: text} map.
    PASS 2: walk the pages again in order. For each page:
    - "scanned": use the pre-computed OCR text from the map.
    - "hybrid": extract native words, OCR the flagged image regions,
    merge the OCR'd words into the same words list, then run the
    normal sidebar/two-column/single-column detection on the
    combined set.
    - "text": unchanged, normal word extraction + layout detection.

    - Header gap detection: after extracting words for "text" and
      "hybrid" pages, checks if native text starts well below the page
      top. If so, OCRs the gap region to recover unextracted header
      content (name, contact info rendered as images or vector paths).
    - Page deduplication: removes consecutive identical page
      contributions from primary_parts and secondary_parts before
      joining, fixing double output from PDFs with duplicate pages.

    Two-column pages still split into primary_parts (left column, main
    narrative, contiguous across page breaks) and secondary_parts
    (right column short lists), joined primary-then-secondary once at
    the very end -- unchanged from the prior revision.

    If any "scanned" page fails the pre-OCR quality gate (see
    ocr_reader.py / ocr_reader_pytesseract.py), read_with_surya_pages()
    below raises ImageQualityError, which is intentionally NOT caught
    here -- it propagates straight up to read_pdf()'s caller.
    """
    primary_parts = []
    secondary_parts = []

    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as fdoc:
        page_native_words = []
        classifications = []

        for page_index in range(len(pdf.pages)):
            page = pdf.pages[page_index]
            fpage = fdoc[page_index]

            char_count = len(fpage.get_text().strip())
            if char_count < 20:
                page_native_words.append([])
                classifications.append("scanned")
                continue

            words = page.extract_words(
                x_tolerance=1.5,
                y_tolerance=3,
                keep_blank_chars=False,
            )
            page_native_words.append(words)
            classifications.append(classify_page(fpage, native_words=words))

        scanned_indices = [i for i, c in enumerate(classifications) if c == "scanned"]

        scanned_text_map = {}
        if scanned_indices:
            scanned_text_map = read_with_surya_pages(pdf_path, scanned_indices)

        for page_index, page in enumerate(pdf.pages):
            classification = classifications[page_index]
            page_width = page.width

            if classification == "scanned":
                text = scanned_text_map.get(page_index, "")
                if text.strip():
                    primary_parts.append(text.strip())
                continue

            words = page_native_words[page_index]

            # fpage needed for ALL non-scanned pages (not just "hybrid"),
            # since header-gap detection below runs on "text" pages too.
            fpage = fdoc[page_index]

            if classification == "hybrid":
                for bbox in get_ocr_worthy_image_blocks(fpage, native_words=words):
                    ocr_words = ocr_region_to_words(fpage, bbox)
                    words = words + ocr_words

            # Header gap detection: if native words start well below the
            # page top, the gap likely contains an unextracted header
            # (image, vector paths, or annotation-layer text). OCR that
            # region to recover it. Uses PSM=4 (single column, variable
            # text sizes) instead of the default PSM=6 (uniform text
            # block), because header regions typically have a large
            # name, smaller subtitle, and small contact text -- three
            # very different text sizes.
            header_gap = _detect_header_gap(words, fpage.rect)
            if header_gap:
                header_words = ocr_region_to_words(
                    fpage, header_gap, psm=4
                )
                words = header_words + words

            if not words:
                continue

            sidebar_split = _find_sidebar_split(words, page_width)
            if sidebar_split:
                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
                content_words = [w for w in words if w["x0"] > sidebar_split]
                text = _merge_sidebar_with_content(sidebar_words, content_words)
                if text.strip():
                    primary_parts.append(text.strip())
                continue

            two_col_result = _is_true_two_column(words, page_width)
            if two_col_result:
                left_words, right_words = two_col_result
                left_text = _extract_words_to_lines(left_words)
                right_text = _extract_words_to_lines(right_words)
                if left_text.strip():
                    primary_parts.append(left_text.strip())
                if right_text.strip():
                    secondary_parts.append(right_text.strip())
                continue

            text = _extract_words_to_lines(words)
            if text.strip():
                primary_parts.append(text.strip())

    # Deduplicate consecutive identical page contributions -- some
    # resume-builder PDFs include duplicate pages (e.g. a "display"
    # page and a "print" page with identical content), which would
    # otherwise produce double output.
    primary_parts = _dedup_consecutive(primary_parts)
    secondary_parts = _dedup_consecutive(secondary_parts)

    primary_text = "\n\n".join(primary_parts)
    secondary_text = "\n\n".join(secondary_parts)

    if secondary_text:
        return primary_text + "\n\n" + secondary_text
    return primary_text


def read_pdf(pdf_path: str) -> str:
    """
    Main entry point for Layer 0's PDF handling.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    return read_text_pdf(pdf_path)










#commenting code after coming from home-
#"""
#PDF Reader — Layer 0
#
#Reads a PDF file and returns clean text.
#Per PAGE (not per document), classifies the page as one of:
# "scanned" -> essentially no extractable text; OCR'd via Surya
# "hybrid"  -> has real text AND embedded image regions with text
#              (e.g. a contact bar rendered as one flattened graphic);
#              OCR's just those regions and merges the recognised
#              words into the same word stream native text uses
# "text"    -> normal page, handled exactly as before
#
#REVISION (earlier) — replaced the old whole-document
#is_scanned_pdf() gate with per-page classification. That old check
#combined the first 3 pages' text and returned one True/False for the
#entire file, so it could not handle:
# - a document where page 1 is a scanned image and page 2 is real text
#   (or vice versa)
# - a page that is mostly real text but has one embedded image
#   (typically a contact bar / icon+label row) that also carries text
#
#Confirmed on a real resume (Adarsh Bhagat): the entire phone/email/
#LinkedIn row was a single flattened raster image sitting between the
#name and the "Summary" heading. pdfplumber's raw page.chars had zero
#characters there, and PyMuPDF's get_text("dict") showed it as an
#image block (type=1), not a text block -- so no text-extraction
#approach could have recovered it without OCR. classify_page() detects
#this ("hybrid"), get_ocr_worthy_image_blocks() finds the image
#region, and ocr_region_to_words() OCRs just that crop and returns
#pdfplumber-shaped word dicts that get merged into `words` BEFORE
#layout detection runs -- so the recovered text flows through the
#existing sidebar/two-column/single-column logic unchanged and lands
#in the correct reading-order position.
#
#Layout handling within a "text" or "hybrid" page (unchanged from
#before): sidebar layout, true two-column layout, single column
#fallback, in that order.
#
#MERGE NOTE (carried over from the previous revision, still applies):
# Two-column pages split into a `primary_parts` stream (left column /
# main narrative, contiguous across page breaks) and a
# `secondary_parts` stream (right column short lists), joined
# primary-then-secondary once at the very end. See the docstrings on
# _is_true_two_column() and read_text_pdf() below for the full
# reasoning; unchanged from the prior version of this file.
#
#HEADER-GAP FIX (earlier):
# Many designed/styled resume templates (Canva, Zety, resume-builder
# PDFs) render the header area (name, title, contact row) as a single
# flattened raster image, as vector-path outlines (drawn shapes, not
# character data), or inside annotation/form layers that pdfplumber's
# content-stream reader never touches. In all three cases,
# pdfplumber.extract_words() returns zero words for that region, and
# the page is classified "text" (because the body below has plenty of
# real text). The header simply vanishes from the output.
#
# Fixed by detecting the "header gap": if the first native word on a
# page starts significantly below the page top (>100pt, roughly 1.4
# inches), there is likely an unextracted header region above it. That
# gap is OCR'd with pytesseract (PSM=4, single column with variable
# text sizes — correct for a multi-line header with large name +
# smaller subtitle + small contact row) and the recovered words are
# prepended to the word list before layout detection runs, so they
# flow through the existing sidebar/two-column/single-column logic
# unchanged.
#
# This is format-agnostic: it works regardless of WHY the header text
# is missing (image, vector paths, annotations, form objects), because
# pytesseract renders the visual page region to a raster and reads
# whatever is visible there.
#
# REQUIRES ocr_region.py's ocr_region_to_words() to accept a `psm`
# keyword argument (added specifically to support this call, which
# needs PSM 4 for a multi-line variable-size header, unlike this
# function's other call site below which uses the default PSM 6 for
# single-line contact bars). If you see
# "TypeError: unexpected keyword argument 'psm'" here, ocr_region.py
# needs that parameter added -- it is NOT optional for this fix to work.
#
#PAGE-DEDUP FIX (earlier):
# Some resume-builder PDFs include duplicate pages (e.g. a "display"
# page and a "print" page with identical content). Both pages get
# processed, producing double output. Fixed by deduplicating
# consecutive identical entries in primary_parts / secondary_parts
# before joining.
#
#NOTE ON SCANNED-PAGE OCR: this file is intentionally NOT shared with
#the OCR modules (ocr_reader.py / ocr_reader_pytesseract.py). Scanned-
#page layout reconstruction has its own separate, independent copy of
#this logic in ingestion/ocr_layout_reconstruction.py, which is free
#to diverge and be tuned for OCR-specific quirks (e.g. tightly-packed
#scanned tables needed a different heading-merge tolerance than resume
#sidebars do) WITHOUT any risk of that tuning affecting this file's
#already-proven behavior on native PDF/DOCX text.
#
#NOTE ON THE PRE-OCR IMAGE QUALITY GATE (blur rejection / deskew /
#brightness correction, added in ocr_reader.py and
#ocr_reader_pytesseract.py): this file does NOT need any changes for
#that either. This file never touches image pixels itself -- it only
#calls read_with_surya_pages() (imported below) and receives back a
#plain {page_index: text} dict. If a scanned page fails the blur gate,
#read_with_surya_pages() raises ImageQualityError, which is not caught
#anywhere in this file, so it propagates naturally up through
#read_text_pdf() -> read_pdf() -> read_resume_file() to the caller
#(test_manager.py already prints any such exception via its existing
#`except Exception as e: print(f"ERROR: {e}")`, so no changes are
#needed there either). This holds true regardless of which OCR engine
#import is active below.
#"""
#
#import re
#import pdfplumber
#import fitz  # PyMuPDF
#from pathlib import Path
#from collections import Counter
#
#from ingestion.ocr_region import (
#    classify_page,
#    get_ocr_worthy_image_blocks,
#    ocr_region_to_words,
#)
#
## ── OCR ENGINE SELECTION ──
## Production default: Surya primary, with its own automatic fallback to
## pytesseract on failure (see ocr_reader.py's read_with_surya_pages()).
#from ingestion.ocr_reader import read_with_surya_pages
#
## TESTING OVERRIDE: to force pytesseract only (bypassing Surya entirely,
## e.g. to compare output quality between engines), comment out the
## import above and uncomment the one below instead. Note this is a
## manual, all-or-nothing override -- it does NOT go through Surya's own
## try-Surya-then-fall-back-to-tesseract logic, it skips Surya entirely.
## The pre-OCR quality gate (blur/deskew/brightness) still runs correctly
## either way, since both ocr_reader.py and ocr_reader_pytesseract.py
## call it identically.
## from ingestion.ocr_reader_pytesseract import read_with_surya_pages
#
#
## ──────────────────────────────────────────────────────────────
## HEADER-GAP DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _detect_header_gap(words: list, page_rect, min_gap: float = 100.0):
#    """
#    Checks whether native words start well below the top of the page,
#    indicating an unextracted header region (image, vector paths, or
#    annotation-layer text that pdfplumber cannot see).
#
#    Returns a fitz.Rect covering the gap (full page width, from page
#    top to the first word's vertical position) if a gap is detected,
#    or None if no gap exists.
#
#    min_gap: minimum vertical distance (in PDF points, 72pt = 1 inch)
#      between the page top and the first word's top to trigger header
#      gap detection. Default 100pt (~1.4 inches) avoids false positives
#      from normal top margins (typically 36–72pt) while catching
#      designed-template headers that take up 1.5–3 inches at the top.
#
#    The returned Rect is passed to ocr_region_to_words() which renders
#    that region at high resolution and OCRs it. This works regardless
#    of WHY the text is unextractable — the OCR sees whatever is
#    visually rendered in that region.
#    """
#    if not words:
#        return None
#
#    first_word_top = min(w["top"] for w in words)
#    gap = first_word_top - page_rect.y0
#
#    if gap > min_gap:
#        return fitz.Rect(
#            page_rect.x0,       # full page width (left edge)
#            page_rect.y0,       # page top
#            page_rect.x1,       # full page width (right edge)
#            first_word_top      # stop at first native word
#        )
#    return None
#
#
## ──────────────────────────────────────────────────────────────
## PAGE-DEDUP HELPER
## ──────────────────────────────────────────────────────────────
#
#def _dedup_consecutive(parts: list) -> list:
#    """
#    Removes consecutive identical entries from a list of page text
#    contributions. Handles the case where a resume-builder PDF has
#    duplicate pages (e.g. display + print copies) that would otherwise
#    produce double output.
#
#    Uses stripped-text comparison to ignore minor whitespace differences.
#    """
#    if len(parts) <= 1:
#        return parts
#    result = [parts[0]]
#    for p in parts[1:]:
#        if p.strip() != result[-1].strip():
#            result.append(p)
#    return result
#
#
## ──────────────────────────────────────────────────────────────
## SIDEBAR LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_sidebar_split(words: list, page_width: float):
#    """
#    Detects if a page has a sidebar layout — a narrow left column
#    containing ONLY section headings, and a wide right column with content.
#    Returns the x-coordinate of the split point if sidebar detected,
#    or None if not a sidebar layout.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    content_start_x = None
#    best_count = 0
#
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    if content_start_x is None or best_count < 3:
#        return None
#
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#    if len(sidebar_candidates) > 25:
#        return None
#
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#    heading_boundary = sidebar_x0s[-1]
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list) -> str:
#    """
#    Merges sidebar heading words with content words by vertical position.
#    """
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 1.5:
#            current_words.append(word)
#            current_top = word["top"]
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#        result_lines.append(content_text)
#
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
## ──────────────────────────────────────────────────────────────
## TWO-COLUMN LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_column_split(words: list, page_width: float):
#    """
#    Finds the actual column split point for a two-column layout, using
#    the gap in x0 start positions rather than assuming the page midpoint.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    best_split = None
#    best_score = 0
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        if left_count < 20 or right_count < 20:
#            continue
#
#        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
#            continue
#
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = gap * balance
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#    return best_split, left_words, right_words
#
#
#def _is_true_two_column(words: list, page_width: float):
#    """
#    Determines whether a page genuinely has a two-column layout, AND
#    returns the correctly split word groups if so.
#    Requires: both sides substantial (>=30 words), near-empty gutter
#    (<5 straddling words), right column spans >=15% of left column's
#    vertical height.
#    """
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return None
#
#    gutter_words = [
#        w for w in words
#        if w["x0"] < split_x - 2 and w["x1"] > split_x + 2
#    ]
#    if len(gutter_words) > 5:
#        return None
#
#    if right_words and left_words:
#        right_top = min(w["top"] for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span = right_bottom - right_top
#
#        left_top = min(w["top"] for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.15:
#            return None
#
#    return left_words, right_words
#
#
## ──────────────────────────────────────────────────────────────
## LINE RECONSTRUCTION
## ──────────────────────────────────────────────────────────────
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from word dicts, grouping by vertical (top)
#    position and joining with a single space. Works identically whether
#    a word came from pdfplumber's native extraction or from
#    ocr_region_to_words() -- both use the same dict shape.
#    """
#    if not words:
#        return ""
#
#    lines: list = []
#    current_line: list = []
#    current_top = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
#
#    if current_line:
#        lines.append(current_line)
#
#    text_lines = []
#    for line in lines:
#        line_sorted = sorted(line, key=lambda w: w["x0"])
#        parts = [w["text"] for w in line_sorted]
#        text_lines.append(" ".join(parts))
#
#    return "\n".join(text_lines)
#
#
## ──────────────────────────────────────────────────────────────
## MAIN PDF TEXT READER
## ──────────────────────────────────────────────────────────────
#
#def read_text_pdf(pdf_path: str) -> str:
#    """
#    Reads a PDF page by page.
#    PASS 1: classify every page ("scanned" / "hybrid" / "text").
#    Collect the indices of "scanned" pages.
#    PASS 1.5: if any pages were classified "scanned", batch-OCR just
#    those pages in ONE call via read_with_surya_pages(), so
#    model loading happens at most once per document regardless of
#    how many scanned pages it has. Build a {page_index: text} map.
#    PASS 2: walk the pages again in order. For each page:
#    - "scanned": use the pre-computed OCR text from the map.
#    - "hybrid": extract native words, OCR the flagged image regions,
#    merge the OCR'd words into the same words list, then run the
#    normal sidebar/two-column/single-column detection on the
#    combined set.
#    - "text": unchanged, normal word extraction + layout detection.
#
#    - Header gap detection: after extracting words for "text" and
#      "hybrid" pages, checks if native text starts well below the page
#      top. If so, OCRs the gap region to recover unextracted header
#      content (name, contact info rendered as images or vector paths).
#    - Page deduplication: removes consecutive identical page
#      contributions from primary_parts and secondary_parts before
#      joining, fixing double output from PDFs with duplicate pages.
#
#    Two-column pages still split into primary_parts (left column, main
#    narrative, contiguous across page breaks) and secondary_parts
#    (right column short lists), joined primary-then-secondary once at
#    the very end -- unchanged from the prior revision.
#
#    If any "scanned" page fails the pre-OCR quality gate (see
#    ocr_reader.py / ocr_reader_pytesseract.py), read_with_surya_pages()
#    below raises ImageQualityError, which is intentionally NOT caught
#    here -- it propagates straight up to read_pdf()'s caller.
#    """
#    primary_parts = []
#    secondary_parts = []
#
#    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as fdoc:
#        page_native_words = []
#        classifications = []
#
#        for page_index in range(len(pdf.pages)):
#            page = pdf.pages[page_index]
#            fpage = fdoc[page_index]
#
#            char_count = len(fpage.get_text().strip())
#            if char_count < 20:
#                page_native_words.append([])
#                classifications.append("scanned")
#                continue
#
#            words = page.extract_words(
#                x_tolerance=1.5,
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#            page_native_words.append(words)
#            classifications.append(classify_page(fpage, native_words=words))
#
#        scanned_indices = [i for i, c in enumerate(classifications) if c == "scanned"]
#
#        scanned_text_map = {}
#        if scanned_indices:
#            scanned_text_map = read_with_surya_pages(pdf_path, scanned_indices)
#
#        for page_index, page in enumerate(pdf.pages):
#            classification = classifications[page_index]
#            page_width = page.width
#
#            if classification == "scanned":
#                text = scanned_text_map.get(page_index, "")
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            words = page_native_words[page_index]
#
#            # fpage needed for ALL non-scanned pages (not just "hybrid"),
#            # since header-gap detection below runs on "text" pages too.
#            fpage = fdoc[page_index]
#
#            if classification == "hybrid":
#                for bbox in get_ocr_worthy_image_blocks(fpage, native_words=words):
#                    ocr_words = ocr_region_to_words(fpage, bbox)
#                    words = words + ocr_words
#
#            # Header gap detection: if native words start well below the
#            # page top, the gap likely contains an unextracted header
#            # (image, vector paths, or annotation-layer text). OCR that
#            # region to recover it. Uses PSM=4 (single column, variable
#            # text sizes) instead of the default PSM=6 (uniform text
#            # block), because header regions typically have a large
#            # name, smaller subtitle, and small contact text -- three
#            # very different text sizes.
#            header_gap = _detect_header_gap(words, fpage.rect)
#            if header_gap:
#                header_words = ocr_region_to_words(
#                    fpage, header_gap, psm=4
#                )
#                words = header_words + words
#
#            if not words:
#                continue
#
#            sidebar_split = _find_sidebar_split(words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#                content_words = [w for w in words if w["x0"] > sidebar_split]
#                text = _merge_sidebar_with_content(sidebar_words, content_words)
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            two_col_result = _is_true_two_column(words, page_width)
#            if two_col_result:
#                left_words, right_words = two_col_result
#                left_text = _extract_words_to_lines(left_words)
#                right_text = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    primary_parts.append(left_text.strip())
#                if right_text.strip():
#                    secondary_parts.append(right_text.strip())
#                continue
#
#            text = _extract_words_to_lines(words)
#            if text.strip():
#                primary_parts.append(text.strip())
#
#    # Deduplicate consecutive identical page contributions -- some
#    # resume-builder PDFs include duplicate pages (e.g. a "display"
#    # page and a "print" page with identical content), which would
#    # otherwise produce double output.
#    primary_parts = _dedup_consecutive(primary_parts)
#    secondary_parts = _dedup_consecutive(secondary_parts)
#
#    primary_text = "\n\n".join(primary_parts)
#    secondary_text = "\n\n".join(secondary_parts)
#
#    if secondary_text:
#        return primary_text + "\n\n" + secondary_text
#    return primary_text
#
#
#def read_pdf(pdf_path: str) -> str:
#    """
#    Main entry point for Layer 0's PDF handling.
#    """
#    path = Path(pdf_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {pdf_path}")
#    if path.suffix.lower() != ".pdf":
#        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
#
#    return read_text_pdf(pdf_path)
#
#
#
#








##commenting the code after coming from home-
#"""
#PDF Reader — Layer 0
#
#Reads a PDF file and returns clean text.
#Per PAGE (not per document), classifies the page as one of:
# "scanned" -> essentially no extractable text; OCR'd via Surya
# "hybrid"  -> has real text AND embedded image regions with text
#              (e.g. a contact bar rendered as one flattened graphic);
#              OCR's just those regions and merges the recognised
#              words into the same word stream native text uses
# "text"    -> normal page, handled exactly as before
#
#REVISION (this version) — replaced the old whole-document
#is_scanned_pdf() gate with per-page classification. That old check
#combined the first 3 pages' text and returned one True/False for the
#entire file, so it could not handle:
# - a document where page 1 is a scanned image and page 2 is real text
#   (or vice versa)
# - a page that is mostly real text but has one embedded image
#   (typically a contact bar / icon+label row) that also carries text
#
#Confirmed on a real resume (Adarsh Bhagat): the entire phone/email/
#LinkedIn row was a single flattened raster image sitting between the
#name and the "Summary" heading. pdfplumber's raw page.chars had zero
#characters there, and PyMuPDF's get_text("dict") showed it as an
#image block (type=1), not a text block -- so no text-extraction
#approach could have recovered it without OCR. classify_page() detects
#this ("hybrid"), get_ocr_worthy_image_blocks() finds the image
#region, and ocr_region_to_words() OCRs just that crop and returns
#pdfplumber-shaped word dicts that get merged into `words` BEFORE
#layout detection runs -- so the recovered text flows through the
#existing sidebar/two-column/single-column logic unchanged and lands
#in the correct reading-order position.
#
#Layout handling within a "text" or "hybrid" page (unchanged from
#before): sidebar layout, true two-column layout, single column
#fallback, in that order.
#
#MERGE NOTE (carried over from the previous revision, still applies):
# Two-column pages split into a `primary_parts` stream (left column /
# main narrative, contiguous across page breaks) and a
# `secondary_parts` stream (right column short lists), joined
# primary-then-secondary once at the very end. See the docstrings on
# _is_true_two_column() and read_text_pdf() below for the full
# reasoning; unchanged from the prior version of this file.
#
#HEADER-GAP FIX (this version):
# Many designed/styled resume templates (Canva, Zety, resume-builder
# PDFs) render the header area (name, title, contact row) as a single
# flattened raster image, as vector-path outlines (drawn shapes, not
# character data), or inside annotation/form layers that pdfplumber's
# content-stream reader never touches. In all three cases,
# pdfplumber.extract_words() returns zero words for that region, and
# the page is classified "text" (because the body below has plenty of
# real text). The header simply vanishes from the output.
#
# Fixed by detecting the "header gap": if the first native word on a
# page starts significantly below the page top (>100pt, roughly 1.4
# inches), there is likely an unextracted header region above it. That
# gap is OCR'd with pytesseract (PSM=4, single column with variable
# text sizes — correct for a multi-line header with large name +
# smaller subtitle + small contact row) and the recovered words are
# prepended to the word list before layout detection runs, so they
# flow through the existing sidebar/two-column/single-column logic
# unchanged.
#
# This is format-agnostic: it works regardless of WHY the header text
# is missing (image, vector paths, annotations, form objects), because
# pytesseract renders the visual page region to a raster and reads
# whatever is visible there.
#
#PAGE-DEDUP FIX (this version):
# Some resume-builder PDFs include duplicate pages (e.g. a "display"
# page and a "print" page with identical content). Both pages get
# processed, producing double output. Fixed by deduplicating
# consecutive identical entries in primary_parts / secondary_parts
# before joining.
#"""
#
#import re
#import pdfplumber
#import fitz  # PyMuPDF
#from pathlib import Path
#from collections import Counter
#
#from ingestion.ocr_region import (
#    classify_page,
#    get_ocr_worthy_image_blocks,
#    ocr_region_to_words,
#)
#from ingestion.ocr_reader import read_with_surya_pages
##commented the above one and written the below just to make the fallback at pytesseract temporarily instead of using surya just to see the o/p quality.
#from ingestion.ocr_reader_pytesseract import read_with_surya_pages
#
#
## ──────────────────────────────────────────────────────────────
## HEADER-GAP DETECTION (NEW)
## ──────────────────────────────────────────────────────────────
#
#def _detect_header_gap(words: list, page_rect, min_gap: float = 100.0):
#    """
#    Checks whether native words start well below the top of the page,
#    indicating an unextracted header region (image, vector paths, or
#    annotation-layer text that pdfplumber cannot see).
#
#    Returns a fitz.Rect covering the gap (full page width, from page
#    top to the first word's vertical position) if a gap is detected,
#    or None if no gap exists.
#
#    min_gap: minimum vertical distance (in PDF points, 72pt = 1 inch)
#      between the page top and the first word's top to trigger header
#      gap detection. Default 100pt (~1.4 inches) avoids false positives
#      from normal top margins (typically 36–72pt) while catching
#      designed-template headers that take up 1.5–3 inches at the top.
#
#    The returned Rect is passed to ocr_region_to_words() which renders
#    that region at high resolution and OCRs it. This works regardless
#    of WHY the text is unextractable — the OCR sees whatever is
#    visually rendered in that region.
#    """
#    if not words:
#        return None
#
#    first_word_top = min(w["top"] for w in words)
#    gap = first_word_top - page_rect.y0
#
#    if gap > min_gap:
#        return fitz.Rect(
#            page_rect.x0,       # full page width (left edge)
#            page_rect.y0,       # page top
#            page_rect.x1,       # full page width (right edge)
#            first_word_top      # stop at first native word
#        )
#    return None
#
#
## ──────────────────────────────────────────────────────────────
## PAGE-DEDUP HELPER (NEW)
## ──────────────────────────────────────────────────────────────
#
#def _dedup_consecutive(parts: list) -> list:
#    """
#    Removes consecutive identical entries from a list of page text
#    contributions. Handles the case where a resume-builder PDF has
#    duplicate pages (e.g. display + print copies) that would otherwise
#    produce double output.
#
#    Uses stripped-text comparison to ignore minor whitespace differences.
#    """
#    if len(parts) <= 1:
#        return parts
#    result = [parts[0]]
#    for p in parts[1:]:
#        if p.strip() != result[-1].strip():
#            result.append(p)
#    return result
#
#
## ──────────────────────────────────────────────────────────────
## SIDEBAR LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_sidebar_split(words: list, page_width: float):
#    """
#    Detects if a page has a sidebar layout — a narrow left column
#    containing ONLY section headings, and a wide right column with content.
#    Returns the x-coordinate of the split point if sidebar detected,
#    or None if not a sidebar layout.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    content_start_x = None
#    best_count = 0
#
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    if content_start_x is None or best_count < 3:
#        return None
#
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#    if len(sidebar_candidates) > 25:
#        return None
#
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#    heading_boundary = sidebar_x0s[-1]
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list) -> str:
#    """
#    Merges sidebar heading words with content words by vertical position.
#    """
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 1.5:
#            current_words.append(word)
#            current_top = word["top"]
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#        result_lines.append(content_text)
#
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
## ──────────────────────────────────────────────────────────────
## TWO-COLUMN LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_column_split(words: list, page_width: float):
#    """
#    Finds the actual column split point for a two-column layout, using
#    the gap in x0 start positions rather than assuming the page midpoint.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    best_split = None
#    best_score = 0
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        if left_count < 20 or right_count < 20:
#            continue
#
#        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
#            continue
#
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = gap * balance
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#    return best_split, left_words, right_words
#
#
#def _is_true_two_column(words: list, page_width: float):
#    """
#    Determines whether a page genuinely has a two-column layout, AND
#    returns the correctly split word groups if so.
#    Requires: both sides substantial (>=30 words), near-empty gutter
#    (<5 straddling words), right column spans >=15% of left column's
#    vertical height.
#    """
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return None
#
#    gutter_words = [
#        w for w in words
#        if w["x0"] < split_x - 2 and w["x1"] > split_x + 2
#    ]
#    if len(gutter_words) > 5:
#        return None
#
#    if right_words and left_words:
#        right_top = min(w["top"] for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span = right_bottom - right_top
#
#        left_top = min(w["top"] for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.15:
#            return None
#
#    return left_words, right_words
#
#
## ──────────────────────────────────────────────────────────────
## LINE RECONSTRUCTION
## ──────────────────────────────────────────────────────────────
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from word dicts, grouping by vertical (top)
#    position and joining with a single space. Works identically whether
#    a word came from pdfplumber's native extraction or from
#    ocr_region_to_words() -- both use the same dict shape.
#    """
#    if not words:
#        return ""
#
#    lines: list = []
#    current_line: list = []
#    current_top = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
#
#    if current_line:
#        lines.append(current_line)
#
#    text_lines = []
#    for line in lines:
#        line_sorted = sorted(line, key=lambda w: w["x0"])
#        parts = [w["text"] for w in line_sorted]
#        text_lines.append(" ".join(parts))
#
#    return "\n".join(text_lines)
#
#
## ──────────────────────────────────────────────────────────────
## MAIN PDF TEXT READER
## ──────────────────────────────────────────────────────────────
#
#def read_text_pdf(pdf_path: str) -> str:
#    """
#    Reads a PDF page by page.
#    PASS 1: classify every page ("scanned" / "hybrid" / "text").
#    Collect the indices of "scanned" pages.
#    PASS 1.5: if any pages were classified "scanned", batch-OCR just
#    those pages in ONE Surya call via read_with_surya_pages(), so
#    model loading happens at most once per document regardless of
#    how many scanned pages it has. Build a {page_index: text} map.
#    PASS 2: walk the pages again in order. For each page:
#    - "scanned": use the pre-computed OCR text from the map.
#    - "hybrid": extract native words, OCR the flagged image regions,
#    merge the OCR'd words into the same words list, then run the
#    normal sidebar/two-column/single-column detection on the
#    combined set.
#    - "text": unchanged, normal word extraction + layout detection.
#
#    NEW in this version:
#    - Header gap detection: after extracting words for "text" and
#      "hybrid" pages, checks if native text starts well below the page
#      top. If so, OCRs the gap region to recover unextracted header
#      content (name, contact info rendered as images or vector paths).
#    - Page deduplication: removes consecutive identical page
#      contributions from primary_parts and secondary_parts before
#      joining, fixing double output from PDFs with duplicate pages.
#
#    Two-column pages still split into primary_parts (left column, main
#    narrative, contiguous across page breaks) and secondary_parts
#    (right column short lists), joined primary-then-secondary once at
#    the very end -- unchanged from the prior revision.
#    """
#    primary_parts = []
#    secondary_parts = []
#
#    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as fdoc:
#        page_native_words = []
#        classifications = []
#
#        for page_index in range(len(pdf.pages)):
#            page = pdf.pages[page_index]
#            fpage = fdoc[page_index]
#
#            char_count = len(fpage.get_text().strip())
#            if char_count < 20:
#                page_native_words.append([])
#                classifications.append("scanned")
#                continue
#
#            words = page.extract_words(
#                x_tolerance=1.5,
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#            page_native_words.append(words)
#            classifications.append(classify_page(fpage, native_words=words))
#
#        scanned_indices = [i for i, c in enumerate(classifications) if c == "scanned"]
#
#        scanned_text_map = {}
#        if scanned_indices:
#            scanned_text_map = read_with_surya_pages(pdf_path, scanned_indices)
#
#        for page_index, page in enumerate(pdf.pages):
#            classification = classifications[page_index]
#            page_width = page.width
#
#            if classification == "scanned":
#                text = scanned_text_map.get(page_index, "")
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            words = page_native_words[page_index]
#
#            # ── CHANGED: get fpage for ALL non-scanned pages (was only
#            #    set for "hybrid" before). Needed for header gap detection
#            #    on "text" pages as well.
#            fpage = fdoc[page_index]
#
#            if classification == "hybrid":
#                for bbox in get_ocr_worthy_image_blocks(fpage, native_words=words):
#                    ocr_words = ocr_region_to_words(fpage, bbox)
#                    words = words + ocr_words
#
#            # ── NEW: Header gap detection ──
#            # If native words start well below the page top, the gap
#            # likely contains an unextracted header (image, vector paths,
#            # or annotation-layer text). OCR that region to recover it.
#            # Uses PSM=4 (single column, variable text sizes) instead of
#            # the default PSM=6 (uniform text block), because header
#            # regions typically have a large name, smaller subtitle, and
#            # small contact text — three very different sizes.
#            header_gap = _detect_header_gap(words, fpage.rect)
#            if header_gap:
#                header_words = ocr_region_to_words(
#                    fpage, header_gap, psm=4
#                )
#                words = header_words + words
#
#            if not words:
#                continue
#
#            sidebar_split = _find_sidebar_split(words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#                content_words = [w for w in words if w["x0"] > sidebar_split]
#                text = _merge_sidebar_with_content(sidebar_words, content_words)
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            two_col_result = _is_true_two_column(words, page_width)
#            if two_col_result:
#                left_words, right_words = two_col_result
#                left_text = _extract_words_to_lines(left_words)
#                right_text = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    primary_parts.append(left_text.strip())
#                if right_text.strip():
#                    secondary_parts.append(right_text.strip())
#                continue
#
#            text = _extract_words_to_lines(words)
#            if text.strip():
#                primary_parts.append(text.strip())
#
#    # ── NEW: Deduplicate consecutive identical page contributions ──
#    # Some resume-builder PDFs include duplicate pages (e.g. a "display"
#    # page and a "print" page with identical content), which would
#    # otherwise produce double output.
#    primary_parts = _dedup_consecutive(primary_parts)
#    secondary_parts = _dedup_consecutive(secondary_parts)
#
#    primary_text = "\n\n".join(primary_parts)
#    secondary_text = "\n\n".join(secondary_parts)
#
#    if secondary_text:
#        return primary_text + "\n\n" + secondary_text
#    return primary_text
#
#
#def read_pdf(pdf_path: str) -> str:
#    """
#    Main entry point for Layer 0's PDF handling.
#    """
#    path = Path(pdf_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {pdf_path}")
#    if path.suffix.lower() != ".pdf":
#        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
#
#    return read_text_pdf(pdf_path)
#






# worked one just commenting to see the code given by antigravity works or not
##writing again after the mess
#
#"""
#PDF Reader — Layer 0
#
#Reads a PDF file and returns clean text.
#Per PAGE (not per document), classifies the page as one of:
# "scanned" -> essentially no extractable text; OCR'd via Surya
# "hybrid"  -> has real text AND embedded image regions with text
#              (e.g. a contact bar rendered as one flattened graphic);
#              OCR's just those regions and merges the recognised
#              words into the same word stream native text uses
# "text"    -> normal page, handled exactly as before
#
#REVISION (this version) — replaced the old whole-document
#is_scanned_pdf() gate with per-page classification. That old check
#combined the first 3 pages' text and returned one True/False for the
#entire file, so it could not handle:
# - a document where page 1 is a scanned image and page 2 is real text
#   (or vice versa)
# - a page that is mostly real text but has one embedded image
#   (typically a contact bar / icon+label row) that also carries text
#
#Confirmed on a real resume (Adarsh Bhagat): the entire phone/email/
#LinkedIn row was a single flattened raster image sitting between the
#name and the "Summary" heading. pdfplumber's raw page.chars had zero
#characters there, and PyMuPDF's get_text("dict") showed it as an
#image block (type=1), not a text block -- so no text-extraction
#approach could have recovered it without OCR. classify_page() detects
#this ("hybrid"), get_ocr_worthy_image_blocks() finds the image
#region, and ocr_region_to_words() OCRs just that crop and returns
#pdfplumber-shaped word dicts that get merged into `words` BEFORE
#layout detection runs -- so the recovered text flows through the
#existing sidebar/two-column/single-column logic unchanged and lands
#in the correct reading-order position.
#
#Layout handling within a "text" or "hybrid" page (unchanged from
#before): sidebar layout, true two-column layout, single column
#fallback, in that order.
#
#MERGE NOTE (carried over from the previous revision, still applies):
# Two-column pages split into a `primary_parts` stream (left column /
# main narrative, contiguous across page breaks) and a
# `secondary_parts` stream (right column short lists), joined
# primary-then-secondary once at the very end. See the docstrings on
# _is_true_two_column() and read_text_pdf() below for the full
# reasoning; unchanged from the prior version of this file.
#"""
#
#import re
#import pdfplumber
#import fitz  # PyMuPDF
#from pathlib import Path
#from collections import Counter
##from ingestion.scanned_pdf_reader import read_scanned_pdf
#
#from ingestion.ocr_region import (
#    classify_page,
#    get_ocr_worthy_image_blocks,
#    ocr_region_to_words,
#)
#from ingestion.ocr_reader import read_with_surya_pages
#
#
## ──────────────────────────────────────────────────────────────
## SIDEBAR LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_sidebar_split(words: list, page_width: float):
#    """
#    Detects if a page has a sidebar layout — a narrow left column
#    containing ONLY section headings, and a wide right column with content.
#    Returns the x-coordinate of the split point if sidebar detected,
#    or None if not a sidebar layout.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    content_start_x = None
#    best_count = 0
#
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    if content_start_x is None or best_count < 3:
#        return None
#
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#    if len(sidebar_candidates) > 25:
#        return None
#
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#    heading_boundary = sidebar_x0s[-1]
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list) -> str:
#    """
#    Merges sidebar heading words with content words by vertical position.
#    """
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 1.5:
#            current_words.append(word)
#            current_top = word["top"]
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#        result_lines.append(content_text)
#
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
## ──────────────────────────────────────────────────────────────
## TWO-COLUMN LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_column_split(words: list, page_width: float):
#    """
#    Finds the actual column split point for a two-column layout, using
#    the gap in x0 start positions rather than assuming the page midpoint.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    best_split = None
#    best_score = 0
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        if left_count < 20 or right_count < 20:
#            continue
#
#        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
#            continue
#
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = gap * balance
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#    return best_split, left_words, right_words
#
#
#def _is_true_two_column(words: list, page_width: float):
#    """
#    Determines whether a page genuinely has a two-column layout, AND
#    returns the correctly split word groups if so.
#    Requires: both sides substantial (>=30 words), near-empty gutter
#    (<5 straddling words), right column spans >=15% of left column's
#    vertical height.
#    """
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return None
#
#    gutter_words = [
#        w for w in words
#        if w["x0"] < split_x - 2 and w["x1"] > split_x + 2
#    ]
#    if len(gutter_words) > 5:
#        return None
#
#    if right_words and left_words:
#        right_top = min(w["top"] for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span = right_bottom - right_top
#
#        left_top = min(w["top"] for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.15:
#            return None
#
#    return left_words, right_words
#
#
## ──────────────────────────────────────────────────────────────
## LINE RECONSTRUCTION
## ──────────────────────────────────────────────────────────────
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from word dicts, grouping by vertical (top)
#    position and joining with a single space. Works identically whether
#    a word came from pdfplumber's native extraction or from
#    ocr_region_to_words() -- both use the same dict shape.
#    """
#    if not words:
#        return ""
#
#    lines: list = []
#    current_line: list = []
#    current_top = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
#
#    if current_line:
#        lines.append(current_line)
#
#    text_lines = []
#    for line in lines:
#        line_sorted = sorted(line, key=lambda w: w["x0"])
#        parts = [w["text"] for w in line_sorted]
#        text_lines.append(" ".join(parts))
#
#    return "\n".join(text_lines)
#
#
## ──────────────────────────────────────────────────────────────
## MAIN PDF TEXT READER
## ──────────────────────────────────────────────────────────────
#
#def read_text_pdf(pdf_path: str) -> str:
#    """
#    Reads a PDF page by page.
#    PASS 1: classify every page ("scanned" / "hybrid" / "text").
#    Collect the indices of "scanned" pages.
#    PASS 1.5: if any pages were classified "scanned", batch-OCR just
#    those pages in ONE Surya call via read_with_surya_pages(), so
#    model loading happens at most once per document regardless of
#    how many scanned pages it has. Build a {page_index: text} map.
#    PASS 2: walk the pages again in order. For each page:
#    - "scanned": use the pre-computed OCR text from the map.
#    - "hybrid": extract native words, OCR the flagged image regions,
#    merge the OCR'd words into the same words list, then run the
#    normal sidebar/two-column/single-column detection on the
#    combined set.
#    - "text": unchanged, normal word extraction + layout detection.
#    Two-column pages still split into primary_parts (left column, main
#    narrative, contiguous across page breaks) and secondary_parts
#    (right column short lists), joined primary-then-secondary once at
#    the very end -- unchanged from the prior revision.
#    """
#    primary_parts = []
#    secondary_parts = []
#
#    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as fdoc:
#        page_native_words = []
#        classifications = []
#
#        for page_index in range(len(pdf.pages)):
#            page = pdf.pages[page_index]
#            fpage = fdoc[page_index]
#
#            char_count = len(fpage.get_text().strip())
#            if char_count < 20:
#                page_native_words.append([])
#                classifications.append("scanned")
#                continue
#
#            words = page.extract_words(
#                x_tolerance=1.5,
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#            page_native_words.append(words)
#            classifications.append(classify_page(fpage, native_words=words))
#
#        scanned_indices = [i for i, c in enumerate(classifications) if c == "scanned"]
#
#        #commenting just to make that scaned image code work-
#        scanned_text_map = {}
#        if scanned_indices:
#            scanned_text_map = read_with_surya_pages(pdf_path, scanned_indices)
##
#        ## NEW
#        #scanned_text_map = {}
#        #if scanned_indices:
#        #    scanned_results = read_scanned_pdf(pdf_path, scanned_indices)
#        #    # read_scanned_pdf returns per-page diagnostics (which OCR
#        #    # engine actually ran, and why Surya fell back to Tesseract
#        #    # if it did) alongside the text -- see that module's
#        #    # docstring. Only the text is needed here; the engine info
#        #    # is already printed by read_scanned_pdf() as it runs.
#        #    scanned_text_map = {
#        #        idx: info["text"] for idx, info in scanned_results.items()
#        #    }
#
#        for page_index, page in enumerate(pdf.pages):
#            classification = classifications[page_index]
#            page_width = page.width
#
#            if classification == "scanned":
#                text = scanned_text_map.get(page_index, "")
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            words = page_native_words[page_index]
#
#            if classification == "hybrid":
#                fpage = fdoc[page_index]
#                for bbox in get_ocr_worthy_image_blocks(fpage, native_words=words):
#                    ocr_words = ocr_region_to_words(fpage, bbox)
#                    words = words + ocr_words
#
#            if not words:
#                continue
#
#            sidebar_split = _find_sidebar_split(words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#                content_words = [w for w in words if w["x0"] > sidebar_split]
#                text = _merge_sidebar_with_content(sidebar_words, content_words)
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            two_col_result = _is_true_two_column(words, page_width)
#            if two_col_result:
#                left_words, right_words = two_col_result
#                left_text = _extract_words_to_lines(left_words)
#                right_text = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    primary_parts.append(left_text.strip())
#                if right_text.strip():
#                    secondary_parts.append(right_text.strip())
#                continue
#
#            text = _extract_words_to_lines(words)
#            if text.strip():
#                primary_parts.append(text.strip())
#
#    primary_text = "\n\n".join(primary_parts)
#    secondary_text = "\n\n".join(secondary_parts)
#
#    if secondary_text:
#        return primary_text + "\n\n" + secondary_text
#    return primary_text
#
#
#def read_pdf(pdf_path: str) -> str:
#    """
#    Main entry point for Layer 0's PDF handling.
#    """
#    path = Path(pdf_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {pdf_path}")
#    if path.suffix.lower() != ".pdf":
#        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
#
#    return read_text_pdf(pdf_path)
#
#







##first one  -worked but for one-two resume write the updated code just above
#"""
#PDF Reader — Layer 0
#Reads a PDF file and returns clean text.
#
#Per PAGE (not per document), classifies the page as one of:
#  "scanned" -> essentially no extractable text; OCR'd via Surya
#  "hybrid"  -> has real text AND embedded image regions with text
#               (e.g. a contact bar rendered as one flattened graphic);
#               OCR's just those regions and merges the recognised
#               words into the same word stream native text uses
#  "text"    -> normal page, handled exactly as before
#
#REVISION (this version) — replaced the old whole-document
#is_scanned_pdf() gate with per-page classification. That old check
#combined the first 3 pages' text and returned one True/False for the
#entire file, so it could not handle:
#  - a document where page 1 is a scanned image and page 2 is real text
#    (or vice versa)
#  - a page that is mostly real text but has one embedded image
#    (typically a contact bar / icon+label row) that also carries text
#
#Confirmed on a real resume (Adarsh Bhagat): the entire phone/email/
#LinkedIn row was a single flattened raster image sitting between the
#name and the "Summary" heading. pdfplumber's raw page.chars had zero
#characters there, and PyMuPDF's get_text("dict") showed it as an
#image block (type=1), not a text block -- so no text-extraction
#approach could have recovered it without OCR. classify_page() detects
#this ("hybrid"), get_ocr_worthy_image_blocks() finds the image
#region, and ocr_region_to_words() OCRs just that crop and returns
#pdfplumber-shaped word dicts that get merged into `words` BEFORE
#layout detection runs -- so the recovered text flows through the
#existing sidebar/two-column/single-column logic unchanged and lands
#in the correct reading-order position.
#
#Layout handling within a "text" or "hybrid" page (unchanged from
#before): sidebar layout, true two-column layout, single column
#fallback, in that order.
#
#MERGE NOTE (carried over from the previous revision, still applies):
#  Two-column pages split into a `primary_parts` stream (left column /
#  main narrative, contiguous across page breaks) and a
#  `secondary_parts` stream (right column short lists), joined
#  primary-then-secondary once at the very end. See the docstrings on
#  _is_true_two_column() and read_text_pdf() below for the full
#  reasoning; unchanged from the prior version of this file.
#"""
#
#import re
#import pdfplumber
#import fitz  # PyMuPDF
#from pathlib import Path
#from collections import Counter
#
#from ingestion.ocr_region import (
#    classify_page,
#    get_ocr_worthy_image_blocks,
#    ocr_region_to_words,
#)
#from ingestion.ocr_reader import read_with_surya_pages
#
#
## ──────────────────────────────────────────────────────────────
## SIDEBAR LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_sidebar_split(words: list, page_width: float):
#    """
#    Detects if a page has a sidebar layout — a narrow left column
#    containing ONLY section headings, and a wide right column with content.
#    Returns the x-coordinate of the split point if sidebar detected,
#    or None if not a sidebar layout.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#
#    content_start_x = None
#    best_count = 0
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    if content_start_x is None or best_count < 3:
#        return None
#
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#    if len(sidebar_candidates)  >25:
#        return None
#
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#    heading_boundary = sidebar_x0s[-1]
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list) -> str:
#    """
#    Merges sidebar heading words with content words by vertical position.
#    """
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 1.5:
#            current_words.append(word)
#            current_top = word["top"]
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#        result_lines.append(content_text)
#
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
## ──────────────────────────────────────────────────────────────
## TWO-COLUMN LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _is_full_width_line(line_words: list, page_width: float, split_x: float) -> bool:
#    if not line_words:
#        return False
#    min_x = min(w["x0"] for w in line_words)
#    max_x = max(w["x1"] for w in line_words)
#    span = max_x - min_x
#    if span > page_width * 0.50 and min_x < split_x - 20 and max_x > split_x + 20:
#        return True
#    return False
#
#
#def _has_large_vertical_gap_in_body(words: list, page_height: float, max_gap: float = 350.0) -> bool:
#    body_words = [w for w in words if w["top"] > page_height * 0.18]
#    if len(body_words) < 2:
#        return False
#    sorted_y = sorted(set(round(w["top"]) for w in body_words))
#    for i in range(len(sorted_y) - 1):
#        if sorted_y[i + 1] - sorted_y[i] > max_gap:
#            return True
#    return False
#
#
#def _find_column_split(words: list, page_width: float):
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    lines_map = {}
#    for w in words:
#        y = round(w["top"] / 4) * 4
#        lines_map.setdefault(y, []).append(w)
#
#    best_split = None
#    best_score = -1
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        if left_count < 3 or right_count < 3:
#            continue
#
#        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
#            continue
#
#        non_full_width_words = []
#        for y, line_w in lines_map.items():
#            if not _is_full_width_line(line_w, page_width, split_x):
#                non_full_width_words.extend(line_w)
#
#        gutter_words = [
#            w for w in non_full_width_words
#            if w["x0"] < split_x - 3 and w["x1"] > split_x + 3
#        ]
#
#        if len(gutter_words) > 6:
#            continue
#
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = (gap ** 1.5) * (balance ** 0.5)
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#
#    return best_split, left_words, right_words
#
#
#def _is_true_two_column(words: list, page_width: float, page_height: float = 842.0):
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    if len(left_words) < 3 or len(right_words) < 3:
#        return None
#
#    if _has_large_vertical_gap_in_body(left_words, page_height, max_gap=350.0) or \
#       _has_large_vertical_gap_in_body(right_words, page_height, max_gap=350.0):
#        return None
#
#    return left_words, right_words
#
#
## ──────────────────────────────────────────────────────────────
## LINE RECONSTRUCTION
## ──────────────────────────────────────────────────────────────
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from word dicts, grouping by vertical (top)
#    position and joining with a single space. Works identically whether
#    a word came from pdfplumber's native extraction or from
#    ocr_region_to_words() -- both use the same dict shape.
#    """
#    if not words:
#        return ""
#
#    lines: list = []
#    current_line: list = []
#    current_top = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
#
#    if current_line:
#        lines.append(current_line)
#
#    text_lines = []
#    for line in lines:
#        line_sorted = sorted(line, key=lambda w: w["x0"])
#        current_chunk = []
#        for w in line_sorted:
#            if not current_chunk:
#                current_chunk.append(w)
#            else:
#                prev_w = current_chunk[-1]
#                if w["x0"] - prev_w["x1"] > 35.0:
#                    text_lines.append(" ".join(cw["text"] for cw in current_chunk))
#                    current_chunk = [w]
#                else:
#                    current_chunk.append(w)
#        if current_chunk:
#            text_lines.append(" ".join(cw["text"] for cw in current_chunk))
#
#    return "\n".join(text_lines)
#
#
## ──────────────────────────────────────────────────────────────
## MAIN PDF TEXT READER
## ──────────────────────────────────────────────────────────────
#
#_KNOWN_SECTION_HEADINGS = {
#    "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT HISTORY", "WORK HISTORY",
#    "EDUCATION", "SKILLS", "TECHNICAL SKILLS", "SUMMARY", "PROFESSIONAL SUMMARY",
#    "PROFILE", "OBJECTIVE", "PROJECTS", "CERTIFICATIONS", "LANGUAGES", "INTERESTS"
#}
#
#
#def _find_header_bottom_y(words: list, page_height: float) -> float:
#    for w in words:
#        if w["top"] < page_height * 0.35:
#            txt = w["text"].upper().strip(":")
#            if txt in _KNOWN_SECTION_HEADINGS:
#                if w["top"] < 50.0:
#                    return 0.0
#                return max(0.0, w["top"] - 5.0)
#    return page_height * 0.12
#
#
#def read_text_pdf(pdf_path: str) -> str:
#    """
#    Reads text from a PDF file using layout-aware page reconstruction.
#    """
#    primary_parts = []
#    secondary_parts = []
#
#    with pdfplumber.open(pdf_path) as pdf:
#        fdoc = fitz.open(pdf_path)
#        classifications = []
#        page_native_words = []
#        for page_index in range(len(pdf.pages)):
#            page = pdf.pages[page_index]
#            fpage = fdoc[page_index]
#            words = page.extract_words(
#                x_tolerance=1.5,
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#            page_native_words.append(words)
#            classifications.append(classify_page(fpage, native_words=words))
#
#        scanned_indices = [i for i, c in enumerate(classifications) if c == "scanned"]
#        scanned_text_map = {}
#        if scanned_indices:
#            scanned_text_map = read_with_surya_pages(pdf_path, scanned_indices)
#
#        for page_index, page in enumerate(pdf.pages):
#            classification = classifications[page_index]
#            page_width = page.width
#            page_height = page.height
#
#            if classification == "scanned":
#                text = scanned_text_map.get(page_index, "")
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            words = page_native_words[page_index]
#
#            if classification == "hybrid":
#                fpage = fdoc[page_index]
#                for bbox in get_ocr_worthy_image_blocks(fpage, native_words=words):
#                    ocr_words = ocr_region_to_words(fpage, bbox)
#                    words = words + ocr_words
#
#            if not words:
#                continue
#
#            # ── ZONED PAGE PROCESSING ──
#            # 1. Isolate top header zone (Name, Contact, Summary at top)
#            header_y = _find_header_bottom_y(words, page_height)
#            header_words = [w for w in words if w["top"] < header_y]
#            body_words = [w for w in words if w["top"] >= header_y]
#
#            header_text = _extract_words_to_lines(header_words)
#
#            if not body_words:
#                if header_text.strip():
#                    primary_parts.append(header_text.strip())
#                continue
#
#            # 2. Process body words for layout (Sidebar / Two-column / Single-column)
#            sidebar_split = _find_sidebar_split(body_words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in body_words if w["x0"] < sidebar_split]
#                content_words = [w for w in body_words if w["x0"] > sidebar_split]
#                body_text = _merge_sidebar_with_content(sidebar_words, content_words)
#            else:
#                two_col_result = _is_true_two_column(body_words, page_width, page_height)
#                if two_col_result:
#                    left_words, right_words = two_col_result
#                    left_text = _extract_words_to_lines(left_words)
#                    right_text = _extract_words_to_lines(right_words)
#                    
#                    right_upper = right_text.upper()
#                    left_upper = left_text.upper()
#                    exp_keywords = ["EXPERIENCE", "WORK HISTORY", "EMPLOYMENT HISTORY", "PROFESSIONAL EXPERIENCE"]
#                    right_has_exp = any(h in right_upper for h in exp_keywords)
#                    left_has_exp = any(h in left_upper for h in exp_keywords)
#                    
#                    if right_has_exp and not left_has_exp:
#                        body_text = right_text.strip() + "\n\n" + left_text.strip()
#                    else:
#                        body_text = left_text.strip() + "\n\n" + right_text.strip()
#                else:
#                    body_text = _extract_words_to_lines(body_words)
#
#            page_parts = [p.strip() for p in [header_text, body_text] if p.strip()]
#            if page_parts:
#                primary_parts.append("\n\n".join(page_parts))
#
#    primary_text = "\n\n".join(primary_parts)
#    secondary_text = "\n\n".join(secondary_parts)
#
#    if secondary_text:
#        return primary_text + "\n\n" + secondary_text
#    return primary_text
#
#
#def read_pdf(pdf_path: str) -> str:
#    """
#    Main entry point for Layer 0's PDF handling.
#
#    NOTE: the old whole-document is_scanned_pdf() short-circuit and the
#    _split_merged_headings()/_join_split_headings() post-processing
#    safety net that used to live here have both been REMOVED from this
#    file. Reason: ingestion/__init__.py's normalise_text() already runs
#    the same two functions (identical _SIDEBAR_HEADINGS list) as part
#    of read_resume_file()'s pipeline for every file type, not just PDF.
#    Keeping both copies risked them drifting apart silently. If you'd
#    rather keep this file fully standalone/independently testable,
#    it's safe to re-add them here too since both are idempotent -- just
#    keep the two _SIDEBAR_HEADINGS lists in sync if so.
#    """
#    path = Path(pdf_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {pdf_path}")
#    if path.suffix.lower() != ".pdf":
#        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
#
#    return read_text_pdf(pdf_path)
#

#"""
#PDF Reader — Layer 0
#"""
#
#import re
#import statistics
#import pdfplumber
#import fitz  # PyMuPDF
#from pathlib import Path
#from collections import Counter
#
#from ingestion.ocr_region import (
#    classify_page,
#    get_ocr_worthy_image_blocks,
#    ocr_region_to_words,
#)
#from ingestion.ocr_reader import read_with_surya_pages
#
#
#def _find_sidebar_split(words: list, page_width: float):
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#
#    content_start_x = None
#    best_count = 0
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    if content_start_x is None or best_count < 3:
#        return None
#
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#    if len(sidebar_candidates) > 25:
#        return None
#
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#    heading_boundary = sidebar_x0s[-1]
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list) -> str:
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 1.5:
#            current_words.append(word)
#            current_top = word["top"]
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#        result_lines.append(content_text)
#
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
#def _find_column_split(words: list, page_width: float):
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    best_split = None
#    best_score = 0
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        if left_count < 20 or right_count < 20:
#            continue
#        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
#            continue
#
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = gap * balance
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#
#    return best_split, left_words, right_words
#
#
#def _group_words_into_rows(words: list) -> list:
#    """
#    Groups words into visual rows by vertical position -- same clustering
#    rule _extract_words_to_lines() uses, but returns the row groups
#    themselves instead of joined text, since the row-aware column split
#    below needs to inspect each row individually.
#    """
#    if not words:
#        return []
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#    rows, current_row, current_top = [], [], None
#    for w in sorted_words:
#        if current_top is None:
#            current_top, current_row = w["top"], [w]
#        elif abs(w["top"] - current_top) <= max(w["height"], 1) * 0.5:
#            current_row.append(w)
#        else:
#            rows.append(current_row)
#            current_row, current_top = [w], w["top"]
#    if current_row:
#        rows.append(current_row)
#    return rows
#
#
#def _split_rows_by_column(words: list, split_x: float, min_gap: float = 30, heading_size_ratio: float = 1.4):
#    """
#    Row-aware column split. Confirmed necessary on a real resume (Sahith
#    Saraswathi): the page has a full-width name banner and a full-width
#    summary paragraph ABOVE a genuine two-column body (education/skills/
#    certifications on the left, experience/projects on the right). The
#    old whole-page gutter check (reject the whole page if more than 5
#    words anywhere straddle split_x) rejected this page outright.
#
#    Two things happen, in order, per row:
#
#    1. Look for the row's own largest internal gap between consecutive
#       words; if it's substantial (>= min_gap), split there. Ordinary
#       word-to-word spacing within a sentence is typically under 15pt,
#       so a 30pt+ gap is unambiguously a real column boundary on its own
#       merits -- no need to check proximity to the page-wide split_x
#       estimate (confirmed necessary: two genuine column gaps measured
#       176pt and 203pt wide, with midpoints comfortably outside a
#       +/-50pt window around split_x, which an earlier, stricter version
#       of this fix wrongly rejected).
#
#    2. If no such gap exists, the row has no internal break to split on
#       -- either because it's short content confined entirely to one
#       side (nothing to split against), or because it's a full-width
#       CENTERED heading (common in template resumes: a "SUMMARY" or
#       "EDUCATION" divider bar centered across the page, so its x0 can
#       land past split_x purely because centering pushed it there, not
#       because it's genuinely right-column content). These two cases are
#       distinguished by font size: confirmed by testing that section
#       headings render at ~1.8x the page's median word height while
#       ordinary one-sided content (e.g. a lone job-title line) renders
#       at only ~1.2x -- comfortably separated by the heading_size_ratio
#       threshold. A row whose words are heading-sized goes to `primary`
#       as a full-width line regardless of which side it nominally falls
#       on; otherwise it's routed by which side it actually sits on.
#    """
#    median_height = statistics.median(w["height"] for w in words) if words else 1
#    rows = _group_words_into_rows(words)
#    primary_words, secondary_words = [], []
#    for row in rows:
#        row_sorted = sorted(row, key=lambda w: w["x0"])
#        local_split, best_gap = None, 0
#        for i in range(len(row_sorted) - 1):
#            gap = row_sorted[i + 1]["x0"] - row_sorted[i]["x1"]
#            if gap >= min_gap and gap > best_gap:
#                best_gap = gap
#                local_split = (row_sorted[i]["x1"] + row_sorted[i + 1]["x0"]) / 2
#
#        if local_split is not None:
#            primary_words.extend(w for w in row_sorted if w["x1"] <= local_split)
#            secondary_words.extend(w for w in row_sorted if w["x0"] >= local_split)
#        else:
#            row_avg_height = sum(w["height"] for w in row_sorted) / len(row_sorted)
#            if median_height > 0 and row_avg_height / median_height >= heading_size_ratio:
#                primary_words.extend(row_sorted)
#                continue
#            row_min_x0 = row_sorted[0]["x0"]
#            row_max_x1 = row_sorted[-1]["x1"]
#            if row_min_x0 >= split_x:
#                secondary_words.extend(row_sorted)
#            elif row_max_x1 <= split_x:
#                primary_words.extend(row_sorted)
#            else:
#                primary_words.extend(row_sorted)
#    return primary_words, secondary_words
#
#
#def _is_true_two_column(words: list, page_width: float):
#    """
#    Determines whether a page genuinely has a two-column layout, AND
#    returns the correctly split word groups if so.
#
#    Qualification (unchanged from before): both naive left/right sides
#    from _find_column_split must be substantial (>=30 words each), and
#    the right column must span a reasonable share of the left column's
#    vertical height. This step still uses the simple, naive left_words/
#    right_words split -- it's just a sanity check that a real two-column
#    region exists on this page at all, not the final content split.
#
#    The ACTUAL content split (this revision) uses _split_rows_by_column()
#    instead of the old blanket gutter-word rejection -- see that
#    function's docstring for why: it lets a page mix full-width bands
#    (header, summary) with a genuine two-column body, instead of
#    rejecting the whole page because full-width prose has words
#    straddling the split point.
#    """
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return None
#
#    if right_words and left_words:
#        right_top = min(w["top"] for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span = right_bottom - right_top
#
#        left_top = min(w["top"] for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.15:
#            return None
#
#    primary_words, secondary_words = _split_rows_by_column(words, split_x)
#    if len(primary_words) < 30 or len(secondary_words) < 15:
#        return None
#
#    return primary_words, secondary_words
#
#
#def _extract_words_to_lines(words: list) -> str:
#    if not words:
#        return ""
#
#    lines = []
#    current_line = []
#    current_top = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
#
#    if current_line:
#        lines.append(current_line)
#
#    text_lines = []
#    for line in lines:
#        line_sorted = sorted(line, key=lambda w: w["x0"])
#        parts = [w["text"] for w in line_sorted]
#        text_lines.append(" ".join(parts))
#
#    return "\n".join(text_lines)
#
#
#def read_text_pdf(pdf_path: str) -> str:
#    primary_parts = []
#    secondary_parts = []
#
#    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as fdoc:
#        page_native_words = []
#        classifications = []
#        for page_index in range(len(pdf.pages)):
#            page = pdf.pages[page_index]
#            fpage = fdoc[page_index]
#
#            char_count = len(fpage.get_text().strip())
#            if char_count < 20:
#                page_native_words.append([])
#                classifications.append("scanned")
#                continue
#
#            words = page.extract_words(
#                x_tolerance=1.5,
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#            page_native_words.append(words)
#            classifications.append(classify_page(fpage, native_words=words))
#
#        scanned_indices = [i for i, c in enumerate(classifications) if c == "scanned"]
#
#        scanned_text_map = {}
#        if scanned_indices:
#            scanned_text_map = read_with_surya_pages(pdf_path, scanned_indices)
#
#        for page_index, page in enumerate(pdf.pages):
#            classification = classifications[page_index]
#            page_width = page.width
#
#            if classification == "scanned":
#                text = scanned_text_map.get(page_index, "")
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            words = page_native_words[page_index]
#
#            if classification == "hybrid":
#                fpage = fdoc[page_index]
#                for bbox in get_ocr_worthy_image_blocks(fpage, native_words=words):
#                    ocr_words = ocr_region_to_words(fpage, bbox)
#                    words = words + ocr_words
#
#            if not words:
#                continue
#
#            sidebar_split = _find_sidebar_split(words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#                content_words = [w for w in words if w["x0"] > sidebar_split]
#                text = _merge_sidebar_with_content(sidebar_words, content_words)
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            two_col_result = _is_true_two_column(words, page_width)
#            if two_col_result:
#                left_words, right_words = two_col_result
#                left_text = _extract_words_to_lines(left_words)
#                right_text = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    primary_parts.append(left_text.strip())
#                if right_text.strip():
#                    secondary_parts.append(right_text.strip())
#                continue
#
#            text = _extract_words_to_lines(words)
#            if text.strip():
#                primary_parts.append(text.strip())
#
#    primary_text = "\n\n".join(primary_parts)
#    secondary_text = "\n\n".join(secondary_parts)
#
#    if secondary_text:
#        return primary_text + "\n\n" + secondary_text
#    return primary_text
#
#
#def read_pdf(pdf_path: str) -> str:
#    path = Path(pdf_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {pdf_path}")
#    if path.suffix.lower() != ".pdf":
#        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
#
#    return read_text_pdf(pdf_path)








#"""
#PDF Reader — Layer 0
#"""
#
#import re
#import statistics
#import pdfplumber
#import fitz  # PyMuPDF
#from pathlib import Path
#from collections import Counter
#
#from ingestion.ocr_region import (
#    classify_page,
#    get_ocr_worthy_image_blocks,
#    ocr_region_to_words,
#)
#from ingestion.ocr_reader import read_with_surya_pages
#
#
#def _find_sidebar_split(words: list, page_width: float):
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#
#    content_start_x = None
#    best_count = 0
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    if content_start_x is None or best_count < 3:
#        return None
#
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#    if len(sidebar_candidates) > 25:
#        return None
#
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#    heading_boundary = sidebar_x0s[-1]
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list) -> str:
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 1.5:
#            current_words.append(word)
#            current_top = word["top"]
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#        result_lines.append(content_text)
#
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
#def _find_column_split(words: list, page_width: float):
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    best_split = None
#    best_score = 0
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        if left_count < 20 or right_count < 20:
#            continue
#        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
#            continue
#
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = gap * balance
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#
#    return best_split, left_words, right_words
#
#
#def _group_words_into_rows(words: list) -> list:
#    """
#    Groups words into visual rows by vertical position -- same clustering
#    rule _extract_words_to_lines() uses, but returns the row groups
#    themselves instead of joined text, since the row-aware column split
#    below needs to inspect each row individually.
#    """
#    if not words:
#        return []
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#    rows, current_row, current_top = [], [], None
#    for w in sorted_words:
#        if current_top is None:
#            current_top, current_row = w["top"], [w]
#        elif abs(w["top"] - current_top) <= max(w["height"], 1) * 0.5:
#            current_row.append(w)
#        else:
#            rows.append(current_row)
#            current_row, current_top = [w], w["top"]
#    if current_row:
#        rows.append(current_row)
#    return rows
#
#
#def _split_rows_by_column(words: list, split_x: float, min_gap: float = 30, heading_size_ratio: float = 1.4):
#    """
#    Row-aware column split. Confirmed necessary on a real resume (Sahith
#    Saraswathi): the page has a full-width name banner and a full-width
#    summary paragraph ABOVE a genuine two-column body (education/skills/
#    certifications on the left, experience/projects on the right). The
#    old whole-page gutter check (reject the whole page if more than 5
#    words anywhere straddle split_x) rejected this page outright.
#
#    Two things happen, in order, per row:
#
#    1. Look for the row's own largest internal gap between consecutive
#       words; if it's substantial (>= min_gap), split there. Ordinary
#       word-to-word spacing within a sentence is typically under 15pt,
#       so a 30pt+ gap is unambiguously a real column boundary on its own
#       merits -- no need to check proximity to the page-wide split_x
#       estimate (confirmed necessary: two genuine column gaps measured
#       176pt and 203pt wide, with midpoints comfortably outside a
#       +/-50pt window around split_x, which an earlier, stricter version
#       of this fix wrongly rejected).
#
#    2. If no such gap exists, the row has no internal break to split on
#       -- either because it's short content confined entirely to one
#       side (nothing to split against), or because it's a full-width
#       CENTERED heading (common in template resumes: a "SUMMARY" or
#       "EDUCATION" divider bar centered across the page, so its x0 can
#       land past split_x purely because centering pushed it there, not
#       because it's genuinely right-column content). These two cases are
#       distinguished by font size: confirmed by testing that section
#       headings render at ~1.8x the page's median word height while
#       ordinary one-sided content (e.g. a lone job-title line) renders
#       at only ~1.2x -- comfortably separated by the heading_size_ratio
#       threshold. A row whose words are heading-sized goes to `primary`
#       as a full-width line regardless of which side it nominally falls
#       on; otherwise it's routed by which side it actually sits on.
#    """
#    median_height = statistics.median(w["height"] for w in words) if words else 1
#    rows = _group_words_into_rows(words)
#    primary_words, secondary_words = [], []
#    for row in rows:
#        row_sorted = sorted(row, key=lambda w: w["x0"])
#        local_split, best_gap = None, 0
#        for i in range(len(row_sorted) - 1):
#            gap = row_sorted[i + 1]["x0"] - row_sorted[i]["x1"]
#            if gap >= min_gap and gap > best_gap:
#                best_gap = gap
#                local_split = (row_sorted[i]["x1"] + row_sorted[i + 1]["x0"]) / 2
#
#        if local_split is not None:
#            primary_words.extend(w for w in row_sorted if w["x1"] <= local_split)
#            secondary_words.extend(w for w in row_sorted if w["x0"] >= local_split)
#        else:
#            row_avg_height = sum(w["height"] for w in row_sorted) / len(row_sorted)
#            if median_height > 0 and row_avg_height / median_height >= heading_size_ratio:
#                primary_words.extend(row_sorted)
#                continue
#            row_min_x0 = row_sorted[0]["x0"]
#            row_max_x1 = row_sorted[-1]["x1"]
#            if row_min_x0 >= split_x:
#                secondary_words.extend(row_sorted)
#            elif row_max_x1 <= split_x:
#                primary_words.extend(row_sorted)
#            else:
#                primary_words.extend(row_sorted)
#    return primary_words, secondary_words
#
#
#def _is_true_two_column(words: list, page_width: float):
#    """
#    Determines whether a page genuinely has a two-column layout, AND
#    returns the correctly split word groups if so.
#
#    Qualification (unchanged from before): both naive left/right sides
#    from _find_column_split must be substantial (>=30 words each), and
#    the right column must span a reasonable share of the left column's
#    vertical height. This step still uses the simple, naive left_words/
#    right_words split -- it's just a sanity check that a real two-column
#    region exists on this page at all, not the final content split.
#
#    The ACTUAL content split (this revision) uses _split_rows_by_column()
#    instead of the old blanket gutter-word rejection -- see that
#    function's docstring for why: it lets a page mix full-width bands
#    (header, summary) with a genuine two-column body, instead of
#    rejecting the whole page because full-width prose has words
#    straddling the split point.
#    """
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return None
#
#    if right_words and left_words:
#        right_top = min(w["top"] for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span = right_bottom - right_top
#
#        left_top = min(w["top"] for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.15:
#            return None
#
#    primary_words, secondary_words = _split_rows_by_column(words, split_x)
#    if len(primary_words) < 30 or len(secondary_words) < 15:
#        return None
#
#    return primary_words, secondary_words
#
#
#def _extract_words_to_lines(words: list) -> str:
#    if not words:
#        return ""
#
#    lines = []
#    current_line = []
#    current_top = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
#
#    if current_line:
#        lines.append(current_line)
#
#    text_lines = []
#    for line in lines:
#        line_sorted = sorted(line, key=lambda w: w["x0"])
#        parts = [w["text"] for w in line_sorted]
#        text_lines.append(" ".join(parts))
#
#    return "\n".join(text_lines)
#
#
#def read_text_pdf(pdf_path: str) -> str:
#    primary_parts = []
#    secondary_parts = []
#
#    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as fdoc:
#        page_native_words = []
#        classifications = []
#        for page_index in range(len(pdf.pages)):
#            page = pdf.pages[page_index]
#            fpage = fdoc[page_index]
#
#            char_count = len(fpage.get_text().strip())
#            if char_count < 20:
#                page_native_words.append([])
#                classifications.append("scanned")
#                continue
#
#            words = page.extract_words(
#                x_tolerance=1.5,
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#            page_native_words.append(words)
#            classifications.append(classify_page(fpage, native_words=words))
#
#        scanned_indices = [i for i, c in enumerate(classifications) if c == "scanned"]
#
#        scanned_text_map = {}
#        if scanned_indices:
#            scanned_text_map = read_with_surya_pages(pdf_path, scanned_indices)
#
#        for page_index, page in enumerate(pdf.pages):
#            classification = classifications[page_index]
#            page_width = page.width
#
#            if classification == "scanned":
#                text = scanned_text_map.get(page_index, "")
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            words = page_native_words[page_index]
#
#            if classification == "hybrid":
#                fpage = fdoc[page_index]
#                for bbox in get_ocr_worthy_image_blocks(fpage, native_words=words):
#                    ocr_words = ocr_region_to_words(fpage, bbox)
#                    words = words + ocr_words
#
#            if not words:
#                continue
#
#            sidebar_split = _find_sidebar_split(words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#                content_words = [w for w in words if w["x0"] > sidebar_split]
#                text = _merge_sidebar_with_content(sidebar_words, content_words)
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            two_col_result = _is_true_two_column(words, page_width)
#            if two_col_result:
#                left_words, right_words = two_col_result
#                left_text = _extract_words_to_lines(left_words)
#                right_text = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    primary_parts.append(left_text.strip())
#                if right_text.strip():
#                    secondary_parts.append(right_text.strip())
#                continue
#
#            text = _extract_words_to_lines(words)
#            if text.strip():
#                primary_parts.append(text.strip())
#
#    primary_text = "\n\n".join(primary_parts)
#    secondary_text = "\n\n".join(secondary_parts)
#
#    if secondary_text:
#        return primary_text + "\n\n" + secondary_text
#    return primary_text
#
#
#def read_pdf(pdf_path: str) -> str:
#    path = Path(pdf_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {pdf_path}")
#    if path.suffix.lower() != ".pdf":
#        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
#
#    return read_text_pdf(pdf_path)
#



















##final one - added two just above
###this is working now also, changing just to see for resume abhinav awasthi-
##
###worked
#"""
#PDF Reader — Layer 0
#Reads a PDF file and returns clean text.
#Automatically detects if the PDF is scanned (image-based) or text-based.
#
#Handles three layout types:
#  1. Single column   — standard reading order
#  2. True two-column — two independent content columns side by side
#  3. Sidebar layout  — narrow heading column on left, wide content column on right
#
#MERGE NOTE (this revision):
#  This file merges two parallel branches of pdf_reader.py that had drifted
#  apart while handling different resume shapes:
#
#  Branch A (original) added the STRICT two-column gutter/height-span check
#  in _is_true_two_column() — required to stop single-column resumes with
#  right-aligned dates/locations from being mis-split down the middle.
#
#  Branch B (newer) added:
#    - Sidebar layout detection + merging (_find_sidebar_split,
#      _merge_sidebar_with_content) for resumes with a narrow heading
#      column and a wide content column.
#    - A smarter two-column SPLIT POINT finder (_find_column_split) that
#      locates the actual gap between columns instead of always assuming
#      the page midpoint.
#    - Post-processing safety nets for sidebar headings that slip past
#      detection (_split_merged_headings, _join_split_headings).
#
#  THE CONFLICT: Branch B's _is_true_two_column() had quietly DROPPED
#  Branch A's gutter check and loosened the height-span ratio (0.6 -> 0.25)
#  while it was being rewritten to use _find_column_split(). Taking Branch
#  B as-is would have reintroduced the false-positive two-column splits
#  that Branch A was specifically built to prevent.
#
#  THE FIX: _is_true_two_column() below uses Branch B's _find_column_split()
#  to locate the real gap (better than assuming the midpoint), but then
#  re-applies Branch A's full strictness on top of that result:
#    - both columns must be substantial (>= 30 words — Branch A's threshold,
#      slightly stricter than Branch B's 20)
#    - the gutter (words straddling the split point) must be nearly empty
#    - the right column must span >= 60% of the left column's height
#  This keeps single-column resumes from being falsely split while still
#  benefiting from the smarter, gap-based split point.
#
#  Processing order per page: sidebar check first, then two-column check,
#  then single-column fallback. This matters because a sidebar layout
#  (narrow heading column + wide content column) can otherwise be
#  misread as a lopsided two-column page.
#"""
#
#import re
#import pdfplumber
#import fitz  # PyMuPDF
#from pathlib import Path
#from collections import Counter
#
#
#def is_scanned_pdf(pdf_path: str) -> bool:
#    """
#    Detects if a PDF is scanned (image-only) or has embedded text.
#    A scanned PDF will have almost no extractable text.
#    """
#    doc = fitz.open(pdf_path)
#    total_text = ""
#    # Check only the first 3 pages — enough to decide
#    for page_num in range(min(3, len(doc))):
#        page = doc[page_num]
#        total_text += page.get_text()
#    doc.close()
#
#    # If we extracted fewer than 100 characters, it is likely scanned
#    return len(total_text.strip()) < 100
#
#
## ──────────────────────────────────────────────────────────────
## SIDEBAR LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_sidebar_split(words: list, page_width: float):
#    """
#    Detects if a page has a sidebar layout — a narrow left column
#    containing ONLY section headings, and a wide right column with content.
#
#    Returns the x-coordinate of the split point if sidebar detected,
#    or None if not a sidebar layout.
#
#    Strategy:
#    We look for a consistent "content start" x-position — the x coordinate
#    where the main body text always begins. In a sidebar resume, content
#    ALWAYS starts at the same x position (e.g. x=166 for Abhay, x=167 for
#    Aarti). The sidebar headings always start at a much smaller x (e.g. x=54).
#
#    If we find that most content lines start at a consistent x position,
#    and there are heading-like words consistently to the LEFT of that
#    position, it is a sidebar layout.
#    """
#    if not words:
#        return None
#
#    # Find the most common x0 starting position of words
#    # In a sidebar resume this is where all content lines start consistently.
#    x0_counter = Counter(round(w["x0"]) for w in words)
#
#    # Find content start x — the x position where main body text begins.
#    #
#    # Strategy: find the most frequent x0 position that:
#    # 1. Is in the left half of page (not too far right)
#    # 2. Is not too close to the left edge (not a heading position)
#    # 3. Has the most words starting there (dominant content column)
#    #
#    # We look for the most frequent x0 in the range [80, 45% of page width].
#    # This skips sidebar heading positions (usually x < 80) and finds
#    # the content column start.
#    content_start_x = None
#    best_count = 0
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    # Content column must have at least 3 lines starting there
#    if content_start_x is None or best_count < 3:
#        return None
#
#    # Find words that start BEFORE content_start_x — these are sidebar words
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#
#    # Sidebar candidates must be few (just section headings)
#    if len(sidebar_candidates) > 25:
#        return None
#
#    # Find the natural heading cluster within sidebar candidates.
#    # Sidebar words like "PROFILE", "SUMMARY", "EXPERIENCE" cluster at
#    # small x values (e.g. x=54-108). Sometimes wrapped content lines
#    # also start at slightly smaller x than content_start_x (e.g. x=154).
#    # We find the heading cluster by looking for a natural gap in x0
#    # positions within sidebar candidates — and take everything BEFORE
#    # that gap as the true heading column.
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#
#    # Find the biggest gap within sidebar candidate x0 positions
#    heading_boundary = sidebar_x0s[-1]  # default: all sidebar words
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:  # significant gap within sidebar itself
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    # Gap from heading cluster to content start must be meaningful
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    # Confirmed sidebar — split at midpoint
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list) -> str:
#    """
#    Merges sidebar heading words with content words by vertical position.
#
#    In a sidebar layout, section headings sit in a narrow left column
#    and content sits in a wide right column. We reconstruct the correct
#    reading order by inserting each heading before the content line
#    that starts at the same vertical position.
#
#    e.g.:
#      sidebar:  top=207 "PERSONAL SUMMARY"
#      content:  top=208 "Senior Oracle ERP consultant..."
#    Result:
#      PERSONAL SUMMARY
#      Senior Oracle ERP consultant...
#    """
#    # Group sidebar words into heading lines by vertical position
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 1.5:
#            # Use 1.5x height tolerance (not 0.5x) for sidebar headings.
#            # Sidebar headings like "PROFESSIONAL\nSUMMARY" can span
#            # two lines with a gap of ~12pts — larger than body text spacing.
#            current_words.append(word)
#            current_top = word["top"]  # update to latest top
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    # Build content lines with vertical positions
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    # Merge: insert heading before the content line at the same vertical position
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        # Find sidebar heading closest to this content line (within 20 pts)
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#
#        result_lines.append(content_text)
#
#    # Add any headings that didn't match content lines
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
## ──────────────────────────────────────────────────────────────
## TWO-COLUMN LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_column_split(words: list, page_width: float):
#    """
#    Finds the actual column split point for a two-column layout.
#
#    Instead of always using the page midpoint, we look for the gap
#    in x0 start positions that best divides words into two groups.
#
#    Returns (split_x, left_words, right_words) or None if no split found.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    # Find the gap that best splits words into two substantial groups
#    best_split = None
#    best_score = 0
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        # Both sides must be substantial
#        if left_count < 20 or right_count < 20:
#            continue
#
#        # Split must be in the middle 40-75% of the page
#        # (avoids detecting margins as columns)
#        if split_x < page_width * 0.35 or split_x > page_width * 0.80:
#            continue
#
#        # Score = gap size × balance between columns
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = gap * balance
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#
#    return best_split, left_words, right_words
#
#
#def _is_true_two_column(words: list, page_width: float):
#    """
#    Determines whether a page genuinely has a two-column layout, AND
#    returns the correctly split word groups if so.
#
#    A real two-column layout has ALL of these properties:
#      1. Many words on both the left half AND the right half
#         (>= 30 words each side).
#      2. A clear gutter (gap) between the columns — very few words
#         straddle the split point (< 5 words crossing it).
#      3. The right-column words span at least 60% of the left column's
#         vertical height (not just a few floating labels like dates
#         or locations).
#
#    This restores the strict gutter + height-span checks that an
#    earlier revision of this file's two-column logic had (correctly)
#    enforced to stop single-column resumes with right-aligned dates/
#    contact info from being incorrectly split down the middle —
#    while still using _find_column_split()'s smarter gap-based split
#    point instead of always assuming the page midpoint.
#
#    Returns (left_words, right_words) if a genuine two-column layout
#    is confirmed, or None otherwise.
#    """
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    # Condition 1 — both halves must be substantial
#    if len(left_words) < 30 or len(right_words) < 30:
#        return None
#
#    # Condition 2 — gutter must be nearly empty (< 5 words crossing the
#    # split point). A word "straddles" the split if it starts before the
#    # split AND ends after it — i.e. the split point passes through the
#    # word itself, not just near it.
#    #
#    # NOTE: left_words/right_words from _find_column_split() already
#    # partition ALL of `words` (left_words = x1 <= split_x, right_words =
#    # x0 > split_x), so a word can never be "in neither" — checking
#    # membership in both lists would always be empty and silently defeat
#    # this check. We instead re-scan the ORIGINAL word boxes directly:
#    # a word straddles the gutter if its own x0 falls before split_x and
#    # its own x1 falls after split_x.
#    gutter_words = [
#        w for w in words
#        if w["x0"] < split_x - 2 and w["x1"] > split_x + 2
#    ]
#    if len(gutter_words) > 5:
#        return None
#
#    # Condition 3 — sanity check on height-span ratio. This is now a much
#    # looser backstop than before (was >= 0.6), because Condition 1's
#    # >=30-words-per-side check already does the real work of rejecting
#    # false positives like a handful of floating right-aligned date/
#    # location labels (those never reach 30 words on their own). The old
#    # 0.6 threshold additionally rejected a common GENUINE layout: a
#    # short right-column list (e.g. "Core Competencies", ~60 words)
#    # sitting next to a long left-column experience narrative that fills
#    # most of the page — confirmed on a real resume where the right
#    # column's height-span ratio was 0.35 despite being a legitimate
#    # two-column page. We keep a very loose floor (0.15) purely as a
#    # backstop against the rare case where >=30 "right words" are one
#    # wide repeated artifact rather than real list content.
#    if right_words and left_words:
#        right_top = min(w["top"] for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span = right_bottom - right_top
#
#        left_top = min(w["top"] for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.15:
#            return None
#
#    return left_words, right_words
#
#
## ──────────────────────────────────────────────────────────────
## LINE RECONSTRUCTION
## ──────────────────────────────────────────────────────────────
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from a list of pdfplumber word dicts, grouping
#    words into lines by their vertical (top) position and inserting a
#    single space between words on the same line.
#
#    Why this exists:
#      `page.extract_text()` and `page.extract_words()` decide where a
#      "word" ends and a "space" begins using x_tolerance — the gap (in
#      PDF points) between two characters. A FLAT x_tolerance value (e.g.
#      3 or 5) cannot work correctly across an entire resume, because the
#      real inter-word gap in points is proportional to font size: a large
#      bold heading and small body text do not share the same natural gap
#      width. A flat tolerance that's "tuned" for body text will swallow
#      real spaces in headings (producing "SeniorManager"), and a flat
#      tolerance tuned for headings will insert false spaces in dense body
#      text. This was the actual bug behind the previous garbled output —
#      not the column-splitting logic, which is handled separately.
#
#      Here we bypass that issue: we already have each word as a separate
#      object (with x0, x1, top), so we just need to decide, for each pair
#      of horizontally-adjacent words, whether to join them with a space.
#      We rely on pdfplumber's own word segmentation (already correct,
#      per-character) and simply join every pair of separate words with
#      exactly one space.
#    """
#    if not words:
#        return ""
#
#    # Group words into lines using their vertical center position.
#    # Words on the same printed line will have nearly identical "top".
#    lines: list = []
#    current_line: list = []
#    current_top = None
#
#    # Sort primarily by vertical position, then horizontal position
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        # Same line if vertical position is close (within half the word's height)
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
#
#    if current_line:
#        lines.append(current_line)
#
#    # Within each line, re-sort by x0 (left to right) and join with a
#    # space between every pair of words.
#    text_lines = []
#    for line in lines:
#        line_sorted = sorted(line, key=lambda w: w["x0"])
#        parts = [w["text"] for w in line_sorted]
#        text_lines.append(" ".join(parts))
#
#    return "\n".join(text_lines)
#
#
## ──────────────────────────────────────────────────────────────
## MAIN PDF TEXT READER
## ──────────────────────────────────────────────────────────────
#
#def read_text_pdf(pdf_path: str) -> str:
#    """
#    Reads a normal (text-based) PDF using pdfplumber.
#    Handles single-column, two-column, and sidebar layouts.
#
#    Per page, layout is checked in this order:
#      1. Sidebar layout   (narrow heading column + wide content column)
#      2. True two-column  (two independent content columns)
#      3. Single column     (fallback)
#
#    Sidebar is checked FIRST because a sidebar layout can otherwise be
#    misread as a lopsided two-column page.
#
#    PAGE-BOUNDARY FIX (this revision):
#      The previous version appended each page's reconstructed text to a
#      single `full_text` list, page by page, in page order. This broke
#      on a real resume shaped like this:
#
#        Page 1: LEFT column = name/title/summary/experience (long,
#                 continues onto page 2). RIGHT column = contact info +
#                 skills + education (short, ends partway down page 1).
#        Page 2: continuation of the LEFT column's experience narrative
#                 (the bullet that was cut off at the bottom of page 1),
#                 plus its own short RIGHT column ("Core Competencies").
#
#      Appending whole-page blocks in page order meant page 1's RIGHT
#      column (contact info, skills, education) got inserted as a block
#      immediately after page 1's LEFT column -- but BEFORE page 2's
#      continuation of that same LEFT-column sentence. The result: the
#      candidate's phone number and email appeared to be stuck in the
#      middle of an experience bullet, splitting "Developed incentive
#      calculation dashboards to track employee" (end of page 1) from
#      "incentive payouts, achievements..." (start of page 2), with the
#      entire contact/skills/education block wedged between them.
#
#      THE FIX: instead of one `full_text` list, we keep TWO separate
#      streams across the WHOLE document:
#        - `primary_parts`   -- left-column / main-narrative content
#                               from two-column pages, plus all single-
#                               column page content (since on a single-
#                               column page there's no second stream to
#                               separate it from).
#        - `secondary_parts` -- right-column content from two-column
#                               pages only (contact info, skills lists,
#                               "Core Competencies", etc.)
#      Sidebar-layout pages produce one already-merged string (heading +
#      content interleaved by design) and go straight into
#      `primary_parts`, since that merge is a deliberate vertical-
#      position match, not a column split.
#
#      Both streams are joined PRIMARY-then-SECONDARY only ONCE, at the
#      very end, after every page has been processed. This guarantees a
#      multi-page left-column narrative stays fully contiguous (page 1's
#      cut-off sentence is immediately followed by page 2's continuation
#      in the primary stream), and the secondary stream's short lists
#      come after the complete narrative instead of interrupting it.
#
#      This does mean secondary-stream content (contact/skills/
#      competencies) ends up positioned after the FULL primary narrative
#      rather than near where it visually sits on the page. That is a
#      deliberate trade-off: Layer 1 (the section segmenter) groups text
#      by section heading, not by physical position, so a clean
#      "all primary content, then all secondary content" ordering is
#      easier for it to segment correctly than text that alternates
#      between columns at every page break.
#    """
#    primary_parts = []
#    secondary_parts = []
#
#    with pdfplumber.open(pdf_path) as pdf:
#        for page in pdf.pages:
#            page_width = page.width
#
#            words = page.extract_words(
#                x_tolerance=1.5,   # tight — relies on per-char positions, not gap-as-space heuristic
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#
#            if not words:
#                continue
#
#            # ── 1. Sidebar layout ──
#            # Already a single merged stream (heading + content matched
#            # by vertical position) — goes straight into primary.
#            sidebar_split = _find_sidebar_split(words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#                content_words = [w for w in words if w["x0"] > sidebar_split]
#                text = _merge_sidebar_with_content(sidebar_words, content_words)
#                if text.strip():
#                    primary_parts.append(text.strip())
#                continue
#
#            # ── 2. True two-column layout ──
#            # Left column -> primary stream (main narrative, may
#            # continue across a page break). Right column -> secondary
#            # stream (short lists like skills/competencies/contact info).
#            two_col_result = _is_true_two_column(words, page_width)
#            if two_col_result:
#                left_words, right_words = two_col_result
#                left_text = _extract_words_to_lines(left_words)
#                right_text = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    primary_parts.append(left_text.strip())
#                if right_text.strip():
#                    secondary_parts.append(right_text.strip())
#                continue
#
#            # ── 3. Single column (fallback) ──
#            # No second stream on this page — goes straight into primary
#            # so it stays contiguous with the main narrative around it.
#            text = _extract_words_to_lines(words)
#            if text.strip():
#                primary_parts.append(text.strip())
#
#    # Join PRIMARY (main narrative, contiguous across page breaks) first,
#    # then SECONDARY (short right-column lists) — once, at the very end.
#    primary_text = "\n\n".join(primary_parts)
#    secondary_text = "\n\n".join(secondary_parts)
#
#    if secondary_text:
#        return primary_text + "\n\n" + secondary_text
#    return primary_text
#
#
## ──────────────────────────────────────────────────────────────
## SIDEBAR HEADING SPLITTER
## Safety net for sidebar-layout pages where _find_sidebar_split
## doesn't trigger (e.g. page 3 of Abhay's resume).
## Splits lines where a known heading is merged with content.
## e.g. "CERTIFICATION Certificate on PHP..."
##   -> "CERTIFICATION\nCertificate on PHP..."
## ──────────────────────────────────────────────────────────────
#_SIDEBAR_HEADINGS = [
#    # Multi-word headings first
#    "PROFESSIONAL SUMMARY", "PERSONAL SUMMARY", "PERSONAL DETAILS",
#    "CORE COMPETENCIES", "PROFILE SUMMARY", "LIVE PROJECTS",
#    "WORK EXPERIENCE", "WORK HISTORY", "EMPLOYMENT HISTORY",
#    "KEY SKILLS", "IT SKILLS", "TECHNICAL SKILLS", "LANGUAGE SKILLS",
#    "ROLES & RESPONSIBILITIES", "ROLES RESPONSIBILITIES",
#    "CUSTOM SECTION", "ACADEMIC BACKGROUND",
#    # Single-word headings
#    "CERTIFICATION", "CERTIFICATIONS", "EDUCATION", "EXPERIENCE",
#    "DECLARATION", "SUMMARY", "OBJECTIVE", "PROFILE",
#    "SKILLS", "PROJECTS", "ACHIEVEMENTS", "AWARDS",
#    "LANGUAGES", "INTERESTS", "HOBBIES", "REFERENCES",
#    "PERSONAL",
#]
#
#
#def _split_merged_headings(text: str) -> str:
#    """
#    Splits lines where a sidebar section heading is merged with content.
#    Only runs when _find_sidebar_split misses a page (safety net).
#    """
#    lines = text.split("\n")
#    result = []
#
#    for line in lines:
#        stripped = line.strip()
#        split_done = False
#
#        for heading in _SIDEBAR_HEADINGS:
#            pattern = re.compile(
#                rf"^({re.escape(heading)})\s+([A-Za-z0-9][^\n]{{3,}})$",
#                re.IGNORECASE
#            )
#            match = pattern.match(stripped)
#            if match:
#                result.append(match.group(1))
#                result.append(match.group(2))
#                split_done = True
#                break
#
#        if not split_done:
#            result.append(line)
#
#    return "\n".join(result)
#
#
#def _join_split_headings(text: str) -> str:
#    """
#    Joins consecutive single-word lines that form a known multi-word heading.
#    e.g. "PROFILE" on line N and "SUMMARY" on line N+1
#      -> "PROFILE SUMMARY"
#    """
#    multi_word_headings = [h for h in _SIDEBAR_HEADINGS if " " in h]
#    lines = text.split("\n")
#    result = []
#    i = 0
#    while i < len(lines):
#        line = lines[i].strip()
#        joined = False
#        if i + 1 < len(lines):
#            next_line = lines[i + 1].strip()
#            combined = line + " " + next_line
#            for heading in multi_word_headings:
#                if combined.upper() == heading.upper():
#                    result.append(heading)
#                    i += 2
#                    joined = True
#                    break
#        if not joined:
#            result.append(lines[i])
#            i += 1
#    return "\n".join(result)
#
#
#def read_pdf(pdf_path: str) -> str:
#    """
#    Main entry point. Decides which method to use based on PDF type.
#    Applies sidebar heading splitting as a post-processing safety net.
#    """
#    path = Path(pdf_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {pdf_path}")
#    if path.suffix.lower() != ".pdf":
#        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
#
#    if is_scanned_pdf(pdf_path):
#        from ingestion.ocr_reader import read_with_surya
#        print(f"[INFO] Scanned PDF detected. Using Surya OCR for: {pdf_path}")
#        text = read_with_surya(pdf_path)
#    else:
#        print(f"[INFO] Text PDF detected. Using pdfplumber for: {pdf_path}")
#        text = read_text_pdf(pdf_path)
#
#    # Post-processing safety net for sidebar layouts
#    text = _split_merged_headings(text)
#    text = _join_split_headings(text)
#    return text
#


















##just for updating the code with merge codes
#"""
#PDF Reader — Layer 0
#Reads a PDF file and returns clean text.
#Automatically detects if the PDF is scanned (image-based) or text-based.
#"""
#
#import pdfplumber
#import fitz  # PyMuPDF
#from pathlib import Path
#
#
#def is_scanned_pdf(pdf_path: str) -> bool:
#    """
#    Detects if a PDF is scanned (image-only) or has embedded text.
#    A scanned PDF will have almost no extractable text.
#    """
#    doc = fitz.open(pdf_path)
#    total_text = ""
#    # Check only the first 3 pages — enough to decide
#    for page_num in range(min(3, len(doc))):
#        page = doc[page_num]
#        total_text += page.get_text()
#    doc.close()
#
#    # If we extracted fewer than 100 characters, it is likely scanned
#    return len(total_text.strip()) < 100
#
#
#def _is_true_two_column(words: list, mid_x: float) -> bool:
#    """
#    Determines whether a page genuinely has a two-column layout.
#
#    A real two-column layout has ALL of these properties:
#      1. Many words on both the left half AND the right half.
#      2. A clear gutter (gap) between the columns — very few words
#         straddle the midpoint.
#      3. The right-column words span the full height of the page
#         (not just a few floating labels like dates or locations).
#
#    The old code only checked condition 1 (> 10 words on each side),
#    which caused single-column resumes with right-aligned dates or
#    contact info to be incorrectly split down the middle.
#    """
#    left_words  = [w for w in words if w["x1"] < mid_x - 20]
#    right_words = [w for w in words if w["x0"] > mid_x + 20]
#    gutter_words = [w for w in words if w["x0"] < mid_x + 20 and w["x1"] > mid_x - 20]
#
#    # Condition 1 — both halves must be substantial
#    if len(left_words) < 30 or len(right_words) < 30:
#        return False
#
#    # Condition 2 — gutter must be nearly empty (< 5 words crossing mid)
#    if len(gutter_words) > 5:
#        return False
#
#    # Condition 3 — right column must span at least 60 % of the page height,
#    # not just a handful of floating labels
#    if right_words:
#        right_top    = min(w["top"]    for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span   = right_bottom - right_top
#
#        left_top    = min(w["top"]    for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span   = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.6:
#            return False
#
#    return True
#
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from a list of pdfplumber word dicts, grouping
#    words into lines by their vertical (top) position and inserting a
#    single space between words on the same line.
#
#    Why this exists:
#      `page.extract_text()` and `page.extract_words()` decide where a
#      "word" ends and a "space" begins using x_tolerance — the gap (in
#      PDF points) between two characters. A FLAT x_tolerance value (e.g.
#      3 or 5) cannot work correctly across an entire resume, because the
#      real inter-word gap in points is proportional to font size: a large
#      bold heading and small body text do not share the same natural gap
#      width. A flat tolerance that's "tuned" for body text will swallow
#      real spaces in headings (producing "SeniorManager"), and a flat
#      tolerance tuned for headings will insert false spaces in dense body
#      text. This was the actual bug behind the previous garbled output —
#      not the column-splitting logic, which is now fixed separately.
#
#      Here we bypass that issue: we already have each word as a separate
#      object (with x0, x1, top), so we just need to decide, for each pair
#      of horizontally-adjacent words, whether to join them with a space.
#      We use a tolerance proportional to font size (height) instead of a
#      flat pixel value, mirroring what pdfplumber's own x_tolerance_ratio
#      does internally — but applied between WORDS (which are already
#      correctly segmented) rather than between characters.
#    """
#    if not words:
#        return ""
#
#    # Group words into lines using their vertical center position.
#    # Words on the same printed line will have nearly identical "top".
#    lines: list = []
#    current_line: list = []
#    current_top = None
#
#    # Sort primarily by vertical position, then horizontal position
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        # Same line if vertical position is close (within half the word's height)
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
#
#    if current_line:
#        lines.append(current_line)
#
#    # Within each line, re-sort by x0 (left to right) and join with a
#    # space whenever the gap between words is non-trivial.
#    text_lines = []
#    for line in lines:
#        line_sorted = sorted(line, key=lambda w: w["x0"])
#        parts = [line_sorted[0]["text"]]
#        for prev, curr in zip(line_sorted, line_sorted[1:]):
#            parts.append(curr["text"])
#        text_lines.append(" ".join(parts))
#
#    return "\n".join(text_lines)
#
#
#def read_text_pdf(pdf_path: str) -> str:
#    """
#    Reads a normal (text-based) PDF using pdfplumber.
#
#    FIX 1 — Two-column false positives (already applied):
#      The old code used `len(left_words) > 10 and len(right_words) > 10`
#      to decide if a page was two-column. This threshold was far too
#      loose and caused single-column resumes to be incorrectly split
#      down the middle, scrambling reading order. _is_true_two_column()
#      now requires volume + a clear gutter + matching vertical span
#      before treating a page as two-column.
#
#    FIX 2 — Missing spaces between words ("SeniorManager", "GoldmanSachs"):
#      The previous attempt raised the flat x_tolerance to 5, which is
#      still a single fixed value applied uniformly across every font
#      size on the page. Headings (larger font) and body text (smaller
#      font) do not share the same natural inter-word gap in PDF points,
#      so no single flat value is correct everywhere — this is why some
#      lines merged and others didn't.
#
#      The real fix: use extract_words() (which segments words correctly
#      using per-character analysis) and then reconstruct each line
#      ourselves with _extract_words_to_lines(), inserting exactly one
#      space between every pair of words pdfplumber already identified
#      as separate. This sidesteps the flat-tolerance problem entirely,
#      since we no longer ask pdfplumber to also decide spacing for
#      multi-word text blocks — only word boundaries, which it is
#      already good at.
#    """
#    full_text = []
#
#    with pdfplumber.open(pdf_path) as pdf:
#        for page in pdf.pages:
#            page_width = page.width
#            mid_x      = page_width / 2
#
#            words = page.extract_words(
#                x_tolerance=1.5,   # tight — relies on per-char positions, not gap-as-space heuristic
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#
#            if not words:
#                continue
#
#            if _is_true_two_column(words, mid_x):
#                left_words  = [w for w in words if w["x1"] <= mid_x]
#                right_words = [w for w in words if w["x0"] >  mid_x]
#                left_text   = _extract_words_to_lines(left_words)
#                right_text  = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    full_text.append(left_text.strip())
#                if right_text.strip():
#                    full_text.append(right_text.strip())
#            else:
#                text = _extract_words_to_lines(words)
#                if text.strip():
#                    full_text.append(text.strip())
#
#    return "\n\n".join(full_text)
#
#
#def read_pdf(pdf_path: str) -> str:
#    """
#    Main entry point. Decides which method to use based on PDF type.
#    """
#    path = Path(pdf_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {pdf_path}")
#    if path.suffix.lower() != ".pdf":
#        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
#
#    if is_scanned_pdf(pdf_path):
#        from ingestion.ocr_reader import read_with_surya
#        print(f"[INFO] Scanned PDF detected. Using Surya OCR for: {pdf_path}")
#        return read_with_surya(pdf_path)
#    else:
#        print(f"[INFO] Text PDF detected. Using pdfplumber for: {pdf_path}")
#        return read_text_pdf(pdf_path)
#
