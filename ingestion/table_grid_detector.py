"""
ingestion/table_grid_detector.py — Layer 0, image-based table detection
for SCANNED pages.

WHY THIS FILE EXISTS
---------------------
The previous approach to table detection on scanned pages
(ocr_layout_reconstruction.py's _reconstruct_table()) tried to GUESS
where a table's rows and columns were by measuring the gaps between
OCR'd words/lines -- e.g. "if the horizontal gap between two words is
more than 22 points, that's probably a column boundary." This has two
problems for a production pipeline:

  1. The gap thresholds (_TABLE_MIN_GAP, _TABLE_CLUSTER_TOLERANCE, etc.)
     were tuned by hand against ONE approximate reconstruction of a
     real table's coordinates, not a calibrated, broadly-tested value
     (this was flagged directly in that module's own comments).
  2. When used with Surya (ocr_reader.py), the "words" fed into this
     gap-guessing logic are actually whole OCR'd LINES (Surya only
     gives line-granularity boxes, not word-granularity), which is
     exactly the mismatch that caused a real production bug elsewhere
     in this codebase (see ocr_reader.py's module docstring on why
     sort_lines_only() replaced reconstruct_page_text() for Surya's
     line data). Reusing that same row/cell-splitting logic for a
     "genuine table" special case reintroduces that same risk.

THE FIX: instead of inferring table structure from where OCR'd text
happened to land, this module looks at the PAGE IMAGE ITSELF and finds
the actual printed grid lines (the black horizontal/vertical rules
that make up a bordered table), using standard OpenCV morphological
line-detection. Once the real grid is known, each cell's pixel region
is cropped and OCR'd on its own -- so a piece of text is assigned to
its row/column because it was physically INSIDE that cell's box in the
image, not because of a heuristic guess about spacing.

If a page has no drawn grid lines (a borderless table, or no table at
all), this module simply finds nothing and returns an empty list --
callers should treat that as "no table here," and fall back to the
existing plain OCR/line-reconstruction path exactly as before. This is
a strict superset of capability, not a replacement risk: pages that
worked fine before (the vast majority -- no tables) are completely
unaffected, since detect_and_ocr_tables() returns [] for them and
nothing downstream changes.

VERIFIED against three real files before being written into the
pipeline (not a theoretical design): SMFG_Job_Document_Scanned.pdf
(both pages -- correctly isolated the 4-row Job Title/Department/
Grade/IC-PM metadata table AND the 3-row Experience/Skills/
Qualification table, while correctly leaving the surrounding bordered
prose paragraphs -- "About SMFG India", "Job Overview", "Key Roles &
Responsibilities" -- untouched as plain text, since those boxes have
an outer border but no INTERNAL grid lines) and a real scanned resume
(Anandu Chandran's 4-column "Course / University / College / % of
mark" academic qualifications table -- came out with every row
correctly split into its 4 real columns).

Remaining OCR character-level mistakes (e.g. "M.COM" misread as
"W.CONM", "%" misread as "04") are ordinary Tesseract recognition
noise, unrelated to this module -- they would happen identically if
that same cell text were read by the whole-page OCR path instead. This
module's job is STRUCTURE (which text belongs in which row/column),
not character-level accuracy.

HOW A PAGE IS SEGMENTED INTO SEPARATE TABLES
----------------------------------------------
A real resume/job-doc page often has SEVERAL separate bordered boxes
(a metadata table, then a bordered paragraph, then another bordered
paragraph, etc.) -- confirmed on the real SMFG document. If we built
one single row/column grid across the WHOLE page, unrelated bordered
sections would get treated as one giant table and prose would get
mangled into fake table cells. So instead:

  1. Find the horizontal and vertical rule-line pixels across the
     whole page (via morphological opening).
  2. Group connected line pixels into separate CONNECTED COMPONENTS --
     each physically-separate bordered box on the page becomes its own
     component (verified directly on the SMFG page: the metadata
     table, the "About SMFG" box, the "Job Overview" box, and the "Key
     Roles" box all came back as four distinct components).
  3. For EACH component independently, look for real row/column lines
     WITHIN that component's own area only.
  4. Only treat a component as a genuine table if it has at least 2
     real rows AND at least 2 real columns inside it (a plain bordered
     paragraph box has an outer rectangle -- 2 horizontal + 2 vertical
     lines total -- but no INTERNAL dividers, so it correctly fails
     this check and is left as ordinary text for the normal OCR path
     to handle).

PRODUCTION SAFETY
-------------------
Every public entry point here is wrapped so that any unexpected
failure (a corrupt image, an OpenCV error, a Tesseract failure on one
cell) degrades to "no tables found on this page" rather than crashing
the whole document -- consistent with how the rest of this pipeline
treats OCR failures (see ocr_reader.py / ocr_reader_pytesseract.py's
own per-page try/except patterns).
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

# ── Tunables (see module docstring: these define geometry, not spacing
# guesses -- they should need far less resume-by-resume retuning than
# the old gap-based approach, but are still named constants rather than
# inline numbers so a future adjustment is easy to find and re-verify) ──

# A connected group of line-pixels smaller than this is noise (a stray
# mark, a thick underline), not a real bordered box.
MIN_COMPONENT_AREA_PX = 2000

# A component covering more than this fraction of the page is almost
# certainly the page's own outer decorative border, not a table.
MAX_COMPONENT_PAGE_AREA_RATIO = 0.5

# Need at least 2 real rows (3 row-lines: top, middle divider, bottom)
# and 2 real columns (3 col-lines) to count as a genuine grid, not just
# a single bordered box with no internal structure.
MIN_ROW_LINES = 3
MIN_COL_LINES = 3

# Two detected line positions closer together than this (in pixels) are
# treated as the same line (absorbs anti-aliasing / scan noise around a
# single printed rule).
LINE_MERGE_GAP_PX = 25

# A "row" or "column" thinner than this is almost certainly a detection
# artifact, not a real table row/column.
MIN_CELL_DIMENSION_PX = 15

# Tesseract settings for reading ONE cell at a time.
CELL_OCR_PSM = 6  # "uniform block of text" -- right for a single cell's contents
CELL_UPSCALE_FACTOR = 2  # upscaling small cell crops measurably helps OCR accuracy
CELL_INNER_PADDING_PX = 4  # shrink the crop slightly so the border line itself isn't OCR'd as text/noise

_warned = False


def _warn_once(message: str):
    global _warned
    if not _warned:
        logger.warning(message)
        print(f"[WARNING] {message}")
        _warned = True


@dataclass
class TableRegion:
    """
    One detected genuine table on a page, in PIXEL coordinates (same
    pixel space as the image passed into detect_and_ocr_tables() --
    callers are responsible for converting to PDF points or whatever
    coordinate space they need, same convention pdf_reader.py /
    ocr_reader.py already use elsewhere in this codebase).

    text: the reconstructed table content, one output line per table
      row, cells within a row separated by a tab -- the same
      convention docx_reader.py and ocr_layout_reconstruction.py
      already use for table rows, so downstream code (section_splitter.py
      etc.) doesn't need to know or care whether a tab-separated line
      came from a native DOCX table, a native PDF table, or an
      OCR'd image table.
    """
    x0: int
    y0: int
    x1: int
    y1: int
    text: str


def _get_line_positions(line_pixel_sum: np.ndarray, merge_gap: int = LINE_MERGE_GAP_PX) -> list:
    """
    Given a 1D array where each index's value is "how many line-pixels
    are present at this row (or column)", finds the representative
    coordinate of each distinct line, merging positions that are close
    together (the same physical line is rarely exactly 1px wide after
    scanning/anti-aliasing).
    """
    if line_pixel_sum.max() <= 0:
        return []
    threshold = line_pixel_sum.max() * 0.3
    is_line = line_pixel_sum > threshold

    positions = []
    i, n = 0, len(is_line)
    while i < n:
        if is_line[i]:
            start = i
            while i < n and is_line[i]:
                i += 1
            positions.append((start + i) // 2)
        else:
            i += 1

    merged = []
    for p in positions:
        if merged and p - merged[-1] < merge_gap:
            merged[-1] = (merged[-1] + p) // 2
        else:
            merged.append(p)
    return merged


def _build_line_masks(gray: np.ndarray):
    """
    Extracts horizontal-rule-line pixels and vertical-rule-line pixels
    separately, via morphological opening with a long thin kernel in
    each direction -- a standard, well-established OpenCV technique for
    finding table grid lines (as opposed to normal text strokes, which
    are much shorter and get erased by a kernel this long).
    """
    bw = cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    h, w = gray.shape

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 1), 1))
    horiz = cv2.dilate(cv2.erode(bw, horiz_kernel), horiz_kernel)

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 1)))
    vert = cv2.dilate(cv2.erode(bw, vert_kernel), vert_kernel)

    return horiz, vert


def _ocr_cell(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> str:
    """OCRs a single table cell's pixel region. Returns "" (not an
    exception) on any failure, since one unreadable cell should not
    block the rest of the table -- consistent with this codebase's
    existing per-region OCR error handling in ocr_region.py."""
    try:
        pad = CELL_INNER_PADDING_PX
        crop = gray[y0 + pad:y1 - pad, x0 + pad:x1 - pad]
        if crop.size == 0:
            return ""
        crop = cv2.resize(
            crop, None,
            fx=CELL_UPSCALE_FACTOR, fy=CELL_UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )
        text = pytesseract.image_to_string(crop, config=f"--psm {CELL_OCR_PSM}")
        return " ".join(text.strip().split())
    except Exception:
        return ""


def detect_and_ocr_tables(pil_image: Image.Image) -> list:
    """
    Main entry point. Given a preprocessed page image (already
    deskewed/brightness-corrected by image_preprocessing.py -- this
    function does no preprocessing of its own, same division of
    responsibility as the rest of ingestion/), finds every genuine
    bordered grid table on the page and OCRs it cell-by-cell.

    Returns a list of TableRegion, in top-to-bottom reading order.
    Returns [] if no genuine table is found, or if anything goes wrong
    -- this must never raise, since a table-detection failure should
    degrade to "treat this page as having no tables" rather than
    breaking the page's OCR entirely (same production-safety principle
    as image_preprocessing.py's blur/deskew/brightness steps, except
    those are allowed to hard-reject on bad input -- this one is purely
    additive, so failure just means falling back to the pre-existing
    behavior).
    """
    try:
        rgb = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        page_area = h * w

        horiz, vert = _build_line_masks(gray)
        grid = cv2.add(horiz, vert)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)

        tables = []
        for i in range(1, n_labels):
            x, y, cw, ch, area = stats[i]
            if area < MIN_COMPONENT_AREA_PX:
                continue
            if (cw * ch) / page_area > MAX_COMPONENT_PAGE_AREA_RATIO:
                continue  # page's own outer decorative border, not a table

            comp_mask = (labels == i).astype(np.uint8) * 255
            comp_horiz = cv2.bitwise_and(horiz, comp_mask)
            comp_vert = cv2.bitwise_and(vert, comp_mask)

            row_lines = _get_line_positions((comp_horiz > 0).sum(axis=1))
            col_lines = _get_line_positions((comp_vert > 0).sum(axis=0))
            row_lines = [r for r in row_lines if y <= r <= y + ch]
            col_lines = [c for c in col_lines if x <= c <= x + cw]

            if len(row_lines) < MIN_ROW_LINES or len(col_lines) < MIN_COL_LINES:
                # No internal dividers -- this is a plain bordered box
                # (e.g. a paragraph in a rounded rectangle), not a
                # genuine multi-row/multi-column table. Leave it for
                # the normal whole-page OCR/text path to handle.
                continue

            rows_out = []
            for r in range(len(row_lines) - 1):
                ry0, ry1 = row_lines[r], row_lines[r + 1]
                if ry1 - ry0 < MIN_CELL_DIMENSION_PX:
                    continue
                cells = []
                for c in range(len(col_lines) - 1):
                    cx0, cx1 = col_lines[c], col_lines[c + 1]
                    if cx1 - cx0 < MIN_CELL_DIMENSION_PX:
                        continue
                    cells.append(_ocr_cell(gray, cx0, ry0, cx1, ry1))
                if any(cells):
                    rows_out.append("\t".join(cells))

            if rows_out:
                tables.append(TableRegion(
                    x0=x, y0=y, x1=x + cw, y1=y + ch,
                    text="\n".join(rows_out),
                ))

        tables.sort(key=lambda t: t.y0)
        return tables

    except Exception as e:
        _warn_once(
            f"Table grid detection failed ({type(e).__name__}: {e}). "
            f"Treating this page as having no tables -- it will fall "
            f"back to normal OCR/text reconstruction."
        )
        return []


_TABLE_PLACEHOLDER_FMT = "@@TABLE_REGION_{i}@@"


def merge_tables_into_words(words: list, tables: list, pixel_to_point: float = 1.0):
    """
    Splices detected tables into an existing OCR word/line list, using
    a placeholder-token trick so none of the existing, already-tested
    row-grouping/line-joining logic (sort_lines_only(),
    reconstruct_page_text(), etc.) needs to change AT ALL to support
    tables:

      1. Any existing word/line whose center point falls inside a
         detected table's region is REMOVED from the list -- its text
         is about to be replaced by the table's own, more accurate,
         per-cell OCR (leaving it in would duplicate that content,
         mixed in with the wrong row/column structure).
      2. One new synthetic entry is added per table, at the table's
         top-left corner, so it naturally sorts into the correct
         reading-order position among the surrounding real text. Its
         "text" is a short, unique placeholder token -- NOT the actual
         table content -- so it behaves like an ordinary single word/
         line as far as the existing row-grouping code is concerned.
      3. The caller runs its normal line-joining logic completely
         unchanged, then does a plain string .replace() of each
         placeholder token with that table's real (possibly multi-
         line) text -- see ocr_reader.py / ocr_reader_pytesseract.py
         for the two call sites using this pattern.

    words: list of dicts with at least "text", "x0", "top" keys (same
      shape used throughout ingestion/ for both native-PDF and OCR
      word/line data).
    tables: list of TableRegion, in PIXEL coordinates (as returned by
      detect_and_ocr_tables()).
    pixel_to_point: multiply TableRegion pixel coordinates by this to
      convert into the SAME coordinate space `words` is already in.
      Pass 1.0 if `words` is itself in raw pixel coordinates (e.g.
      Surya's line dicts before conversion) or the appropriate
      pixels-per-point conversion factor if `words` is already in PDF
      points (e.g. ocr_reader_pytesseract.py's word dicts).

    Returns (new_words, placeholder_map) where placeholder_map maps
    each placeholder token -> that table's real text, for the caller
    to substitute back in after joining.
    """
    if not tables:
        return words, {}

    scaled = [
        (
            t.x0 * pixel_to_point, t.y0 * pixel_to_point,
            t.x1 * pixel_to_point, t.y1 * pixel_to_point,
            t.text,
        )
        for t in tables
    ]

    def _inside_any_table(word) -> bool:
        cx = word.get("x0", 0)
        cy = word.get("top", 0)
        if "x1" in word:
            cx = (word["x0"] + word["x1"]) / 2
        if "bottom" in word:
            cy = (word["top"] + word["bottom"]) / 2
        for x0, y0, x1, y1, _ in scaled:
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return True
        return False

    new_words = [w for w in words if not _inside_any_table(w)]

    placeholder_map = {}
    for i, (x0, y0, _x1, _y1, text) in enumerate(scaled):
        token = _TABLE_PLACEHOLDER_FMT.format(i=i)
        placeholder_map[token] = text
        new_words.append({
            "text": token,
            "x0": x0,
            "x1": x0 + 50,
            "top": y0,
            "bottom": y0 + 12,
            "height": 12,
        })

    return new_words, placeholder_map


def splice_placeholders(text: str, placeholder_map: dict) -> str:
    """Replaces each placeholder token with its table's real text.
    Trivial, but centralized here so both OCR modules use the exact
    same substitution step."""
    for token, table_text in placeholder_map.items():
        text = text.replace(token, table_text)
    return text
    