"""
ocr_reader_pytesseract.py — Layer 0, full-page OCR (scanned pages)

Drop-in replacement for read_with_surya_pages() in ocr_reader.py. Same
signature, same return shape ({page_index: text}), so pdf_reader.py needs
exactly ONE line changed to switch backends:

    # from ingestion.ocr_reader import read_with_surya_pages
    from ingestion.ocr_reader_pytesseract import read_with_surya_pages

Why this exists instead of fixing ocr_reader.py's Surya calls:
  Your ocr_reader.py imports (surya.model.detection.model.load_model,
  surya.ocr.run_ocr, etc.) match surya-ocr==0.6.13's API exactly. The
  current PyPI release (0.22.1) removed all of that in favor of a
  VLM-based RecognitionPredictor/DetectionPredictor setup that expects a
  running model server -- `pip install surya-ocr` today gives you a
  library your existing code cannot import at all.

  Rather than pin an old dependency indefinitely, this reuses
  pytesseract -- which you already have working correctly in
  ocr_region.py for hybrid-page regions -- for full scanned pages too.
  One OCR engine total, no extra model download, verified directly
  against real scanned resumes (Priya Sharma, Amol Jadhav) before this
  was written.

── REVISION — word-level extraction + layout reconstruction ──
Confirmed on a real scanned job document (SMFG_Job_Document_Scanned.pdf):
the previous version of this file called pytesseract.image_to_string(),
which returns a single flat string using TESSERACT'S OWN internal line
ordering -- no word positions are kept, so there was no way to detect or
fix a scrambled table/sidebar layout afterward. A 2-column label/value
table ("Job Title" | "Asst. Manager, Cyber Defence") came out with rows
in the wrong order because tesseract's own reading order doesn't
understand table columns.

This version instead calls pytesseract.image_to_data() (word-level
bounding boxes -- the same call ocr_region.py already uses successfully
for hybrid-page image regions), converts each word's pixel bbox into PDF
point coordinates, and runs the resulting word list through
ocr_layout_reconstruction.reconstruct_page_text() -- the SAME sidebar /
two-column / single-column logic pdf_reader.py already uses for native
PDF text. Verified directly against the SMFG table's real OCR'd
coordinates: this correctly reconstructs
    Job Title
    Asst. Manager, Cyber Defence
    Department
    InfoSec
    ...
instead of the old scrambled order.

PRODUCTION SAFETY: this is wrapped in a per-page try/except. If word-
level extraction or layout reconstruction fails or raises for ANY
reason, that single page falls back to the OLD plain image_to_string()
behavior -- it does not raise, and it does not affect any other page or
any other file in a batch run. A warning is logged (once per process,
not once per page, to avoid spamming logs across a large batch) the
first time this fallback triggers, so you can find out it happened
without your terminal filling up across 60+ resumes.

Known quality quirks to expect (seen on real test files, not
hypothetical, unchanged from before this revision):
  - "AI" frequently reads as "Al" (capital I / lowercase l confusion).
    Downstream cleanup can regex-fix this near known tech terms if it
    matters for skill extraction.
  - "•" bullets occasionally misread as a bare "e" at line-start. Not
    caught by section_splitter's _LEADING_BULLET_RE (which expects
    real bullet glyphs), so these lines just won't get bullet
    normalization -- harmless, but flagging so it's not a surprise
    when spot-checking output.

── REVISION — pre-OCR image quality gate ──
Added the same preprocessing step ocr_reader.py (Surya) now uses:
ingestion/image_preprocessing.py's preprocess_scanned_image() runs on
every rendered page BEFORE any OCR happens (word-level or plain-string
fallback), applying:

  1. Blur rejection (raises ImageQualityError) -- see that module's
     calibration table. IMPORTANT: this file renders at 300 DPI while
     ocr_reader.py's Surya path renders at 200 DPI. Raw Laplacian
     variance is resolution-dependent -- confirmed directly: the same
     physical blur scored 10.18 at 200 DPI but only 6.77 at 300 DPI,
     nearly a 2x difference. image_preprocessing.compute_blur_score()
     normalizes to a fixed reference width before scoring specifically
     so ONE threshold (MIN_BLUR_VARIANCE) is valid for both this file's
     300 DPI render and ocr_reader.py's 200 DPI render -- don't bypass
     that normalization or re-derive a separate threshold for this file.
  2. Deskew -- corrects page rotation before word-level OCR runs. This
     matters even more here than for the old plain-string path: a
     rotated page feeds bad x0/top coordinates into
     reconstruct_page_text(), which would compound the exact table-
     scrambling problem this revision's layout fix was built to solve
     in the first place.
  3. Brightness correction -- inverts moderately dark whole-page scans.

Preprocessing happens ONCE per page, before the try/except that guards
layout reconstruction -- NOT inside _ocr_page_with_layout() or
_ocr_page_plain_fallback() individually. This is deliberate: those two
functions now both take an already-preprocessed PIL Image (not a raw
fitz page) precisely so a genuine ImageQualityError propagates straight
up to the caller as a hard rejection, rather than being caught by the
layout-reconstruction try/except and silently retried via
_ocr_page_plain_fallback() -- which cannot recover a blurry image any
better than the word-level path can, so retrying it would just waste
time before failing the same way, or worse, silently return low-
confidence output instead of a clear rejection.

── REVISION (this version) — image-based table grid detection ──
Previously, table structure on this path came from
ocr_layout_reconstruction.py's _reconstruct_table(), which guessed row/
column boundaries from gaps between tesseract's word-level bounding
boxes (a real technique, but tuned against only one hand-estimated
reconstruction of a real table's coordinates -- see that module's own
docstring, which flags _TABLE_MIN_GAP as "a starting point, not a
calibrated constant").

This revision adds ingestion/table_grid_detector.py's
detect_and_ocr_tables() as a FIRST pass, run directly on the page image
before any word-level OCR happens. It finds a table's actual printed
grid lines (real black rules) via OpenCV, and OCRs each cell on its
own -- structure comes from physical pixels, not from guessing at word
spacing. Verified directly against SMFG_Job_Document_Scanned.pdf and a
real scanned resume's 4-column academic-qualifications table -- see
table_grid_detector.py's module docstring for full verification notes.

Any tesseract word whose center falls inside a detected table region is
removed from the word list before reconstruct_page_text() runs (its
text is about to be replaced by the table's own, more accurate,
per-cell OCR), and the table's real text is spliced back into the final
output at the correct position. Because those words are removed first,
_reconstruct_table() inside reconstruct_page_text() naturally has
nothing left to (mis)detect in that region -- it remains in place,
unmodified, purely as a fallback for a borderless table that
detect_and_ocr_tables() didn't find (no drawn grid lines to detect),
which is a reasonable belt-and-suspenders safety net rather than the
sole/primary table-detection strategy it was before.

Pages with no detected table (the large majority) are completely
unaffected: detect_and_ocr_tables() returns [], the word list is
unchanged, and everything proceeds exactly as it did before this
revision.
"""

import logging
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps

from ingestion.ocr_layout_reconstruction import reconstruct_page_text
# NOTE: every call to reconstruct_page_text() below passes
# skip_table_detection=True, because this module ALWAYS runs
# table_grid_detector.py's image-based table detection first (see
# _ocr_page_with_layout() and the standalone-image branch of
# read_with_surya() below). Letting reconstruct_page_text() ALSO run
# its own internal word-gap-based table guess on top of that is not
# just redundant -- it actively corrupts borderless two-column layouts
# by mistaking the column gap for a table-column gap (confirmed on a
# real resume; see ocr_layout_reconstruction.py's module docstring).
from ingestion.image_preprocessing import preprocess_scanned_image, ImageQualityError
from ingestion.table_grid_detector import (
    detect_and_ocr_tables,
    merge_tables_into_words,
    splice_placeholders,
)
# adding code from antigravity
from ingestion.ocr_output_quality import check_not_degenerate, DegenerateOCROutputError

logger = logging.getLogger(__name__)

# 300 DPI balances OCR accuracy against render/OCR time. Confirmed
# sufficient on real resume scans; bump higher only if you see garbled
# small print (e.g. footnote-sized certifications lists). NOTE: this
# differs from ocr_reader.py's Surya path (200 DPI) -- see the module
# docstring's REVISION note on why the blur quality gate is resolution-
# normalized so one threshold works correctly for both.
OCR_DPI = 300
OCR_PSM = 3  # "fully automatic page segmentation" -- right default for
             # a whole page with mixed paragraphs/headings/lists, unlike
             # ocr_region.py's PSM 6 which assumes ONE uniform text block
             # (fine for a single-line contact bar, wrong for a full page)

# Converts a pixel coordinate rendered at OCR_DPI back into PDF points
# (72 points per inch, always -- this has nothing to do with OCR_DPI
# itself, it's just the PDF unit definition). Used to build word dicts
# in the same coordinate space pdfplumber's extract_words() already
# uses, so reconstruct_page_text() doesn't need to know or care whether
# a word came from native PDF text or from OCR.
_PDF_POINTS_PER_INCH = 72
_PIXEL_TO_POINT = _PDF_POINTS_PER_INCH / OCR_DPI

_word_extraction_warned = False


def _warn_once(message: str):
    global _word_extraction_warned
    if not _word_extraction_warned:
        logger.warning(message)
        print(f"[WARNING] {message}")
        _word_extraction_warned = True


def _render_page(fitz_page, dpi: int = OCR_DPI) -> Image.Image:
    zoom = dpi / 72
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return ImageOps.expand(img, border=30, fill="white")


def _extract_words_from_image(img: Image.Image) -> list:
    """
    Runs word-level OCR via pytesseract.image_to_data() and returns a
    list of word dicts in PDF point coordinates: {"text","x0","x1",
    "top","bottom","height"} -- the same shape pdfplumber's
    extract_words() and ocr_region.py's ocr_region_to_words() both use,
    so reconstruct_page_text() can treat them identically.

    img is expected to already be preprocessed (blur-gated, deskewed,
    brightness-corrected) by the caller -- see read_with_surya_pages().
    """
    # Confirmed on the SMFG table: text sitting on a colored (green)
    # table-cell background is missed almost entirely by tesseract on
    # the raw RGB render -- the labels ("Job Title", "Department",
    # "Grade", "IC/PM") simply don't come back as detected words at
    # all. Converting to grayscale first fixes this; ocr_region.py
    # already does the same thing for hybrid-page image regions for
    # the same reason (see its own comment: "notably improves
    # confidence on small embedded-image text").
    img = ImageOps.grayscale(img)
    data = pytesseract.image_to_data(img, config=f"--psm {OCR_PSM}",
                                      output_type=pytesseract.Output.DICT)
    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = (data["left"][i], data["top"][i],
                      data["width"][i], data["height"][i])
        words.append({
            "text": text,
            "x0": x * _PIXEL_TO_POINT,
            "x1": (x + w) * _PIXEL_TO_POINT,
            "top": y * _PIXEL_TO_POINT,
            "bottom": (y + h) * _PIXEL_TO_POINT,
            "height": h * _PIXEL_TO_POINT,
        })
    return words


def _ocr_page_with_layout(img: Image.Image, page_width_pts: float) -> str:
    """
    Table detection -> word-level OCR -> layout reconstruction, for one
    already-rendered-and-preprocessed page image. Raises on any
    failure -- caller (read_with_surya_pages) catches and falls back to
    the old plain-text behavior for just this page.

    See module docstring's REVISION note: detect_and_ocr_tables() runs
    on the image directly (finds real printed grid lines), independent
    of tesseract's word boxes. Any word landing inside a detected table
    is removed before reconstruct_page_text() runs, and the table's own
    per-cell OCR text is spliced back in afterward at the correct
    position.
    """
    tables = detect_and_ocr_tables(img)
    words = _extract_words_from_image(img)

    if tables:
        # IMPORTANT: tables are in raw PIXEL coordinates (as returned by
        # detect_and_ocr_tables(), which works directly on the rendered
        # image before any point-conversion happens), but `words` here
        # has ALREADY been converted into PDF points by
        # _extract_words_from_image(). pixel_to_point=_PIXEL_TO_POINT
        # converts the table bboxes into that same point-space so the
        # two line up correctly -- passing 1.0 here would silently
        # misplace every detected table (a real bug caught during
        # review, not a hypothetical one -- worth remembering if this
        # function is ever refactored).
        words, placeholder_map = merge_tables_into_words(
            words, tables, pixel_to_point=_PIXEL_TO_POINT
        )
    else:
        placeholder_map = {}

    text = reconstruct_page_text(words, page_width_pts, skip_table_detection=True)
    return splice_placeholders(text, placeholder_map)


def _ocr_page_plain_fallback(img: Image.Image) -> str:
    """The old, pre-revision behavior: whole-page OCR with no layout
    awareness at all. Used only when _ocr_page_with_layout() fails.
    Takes an already-rendered-and-preprocessed image, same as the
    layout-aware path -- NOT a raw fitz page, so this never re-triggers
    the blur/deskew/brightness gate a second time for the same page."""
    return pytesseract.image_to_string(img, config=f"--psm {OCR_PSM}").strip()


def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
    """
    Same contract as ocr_reader.read_with_surya_pages(): OCRs only the
    given 0-based page indices of a PDF, returns {page_index: text}.
    Name kept identical so pdf_reader.py's call site doesn't need to
    change beyond the import line.

    Each page is rendered and preprocessed (blur gate + deskew +
    brightness) ONCE, before layout reconstruction is attempted. A
    genuine ImageQualityError from that step is NOT caught here -- it
    propagates straight up to the caller as a hard rejection (see
    module docstring). Only layout-reconstruction failures (a different,
    non-quality-related problem) fall back to plain OCR for that page.
    """
    if not page_indices:
        return {}

    results = {}
    with fitz.open(pdf_path) as doc:
        for page_index in page_indices:
            page = doc[page_index]
            img = _render_page(page)
            # Raises ImageQualityError on genuinely bad pages -- deliberately
            # NOT inside the try/except below, so it propagates as a hard
            # rejection instead of triggering the plain-OCR fallback.
            img = preprocess_scanned_image(img, context=f"page {page_index + 1} of {pdf_path}")
            page_width_pts = page.rect.width
            try:
                text = _ocr_page_with_layout(img, page_width_pts)
            except Exception as e:
                _warn_once(
                    f"Word-level layout reconstruction failed on at least one "
                    f"page ({type(e).__name__}: {e}). Falling back to plain "
                    f"whole-page OCR for any page where this happens -- "
                    f"other pages/files are unaffected."
                )
                text = _ocr_page_plain_fallback(img)
            # adding code from antigravity
            # Last-resort quality gate: tesseract is the final engine,
            # so degenerate output here is a genuine hard rejection.
            # DegenerateOCROutputError propagates uncaught -- never
            # swallowed by the layout-reconstruction try/except above
            # (that block only guards HOW text is reconstructed, not
            # WHETHER the final text is trustworthy).
            check_not_degenerate(text, context=f"page {page_index + 1} of {pdf_path}")
            results[page_index] = text

    return results


def read_with_surya(file_path: str) -> str:
    """
    Same contract as ocr_reader.read_with_surya(): whole-file OCR for a
    scanned PDF or a standalone image file, joined into one string.
    Kept for parity in case anything calls the non-paged version.

    Same preprocessing-before-layout-reconstruction ordering as
    read_with_surya_pages() above, for the same reason: a real
    ImageQualityError must propagate as a hard rejection, not get
    caught by the layout-reconstruction fallback.
    """
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        with fitz.open(file_path) as doc:
            texts = []
            for page in doc:
                img = _render_page(page)
                img = preprocess_scanned_image(img, context=file_path)
                try:
                    _pg_text = _ocr_page_with_layout(img, page.rect.width)
                except Exception as e:
                    _warn_once(
                        f"Word-level layout reconstruction failed on at least "
                        f"one page ({type(e).__name__}: {e}). Falling back to "
                        f"plain whole-page OCR for any page where this happens."
                    )
                    _pg_text = _ocr_page_plain_fallback(img)
                # adding code from antigravity
                # Last-resort quality gate per page.
                check_not_degenerate(_pg_text, context=file_path)
                texts.append(_pg_text)
    elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
        img = Image.open(file_path).convert("RGB")
        img = preprocess_scanned_image(img, context=file_path)
        # Standalone images have no fitz page / PDF point space to
        # convert into -- pixel coordinates ARE the coordinate space
        # here, which is fine since reconstruct_page_text() only cares
        # about relative positions, not absolute units.
        try:
            tables = detect_and_ocr_tables(img)
            gray = ImageOps.grayscale(img)
            data = pytesseract.image_to_data(gray, config=f"--psm {OCR_PSM}",
                                              output_type=pytesseract.Output.DICT)
            words = []
            n = len(data["text"])
            for i in range(n):
                t = data["text"][i].strip()
                if not t:
                    continue
                x, y, w, h = (data["left"][i], data["top"][i],
                              data["width"][i], data["height"][i])
                words.append({"text": t, "x0": x, "x1": x + w,
                              "top": y, "bottom": y + h, "height": h})
            if tables:
                # Here words ARE still in raw pixel coordinates (no PDF-point
                # conversion for standalone images -- see comment above), so
                # table pixel coordinates need no scaling either.
                words, placeholder_map = merge_tables_into_words(
                    words, tables, pixel_to_point=1.0
                )
            else:
                placeholder_map = {}
            page_text = reconstruct_page_text(words, img.width, skip_table_detection=True)
            texts = [splice_placeholders(page_text, placeholder_map)]
        except Exception as e:
            _warn_once(
                f"Word-level layout reconstruction failed on standalone "
                f"image ({type(e).__name__}: {e}). Falling back to plain OCR."
            )
            texts = [pytesseract.image_to_string(img, config=f"--psm {OCR_PSM}").strip()]
        # adding code from antigravity
        # Last-resort quality gate for standalone image.
        check_not_degenerate(texts[0], context=file_path)
    else:
        raise ValueError(f"Unsupported file type for OCR: {path.suffix}")

    return "\n\n".join(t for t in texts if t)


if __name__ == "__main__":
    import sys
    result = read_with_surya(sys.argv[1])
    print(result)















#"""
#ocr_reader_pytesseract.py — Layer 0, full-page OCR (scanned pages)
#
#Drop-in replacement for read_with_surya_pages() in ocr_reader.py. Same
#signature, same return shape ({page_index: text}), so pdf_reader.py needs
#exactly ONE line changed to switch backends:
#
#    # from ingestion.ocr_reader import read_with_surya_pages
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages
#
#Why this exists instead of fixing ocr_reader.py's Surya calls:
#  Your ocr_reader.py imports (surya.model.detection.model.load_model,
#  surya.ocr.run_ocr, etc.) match surya-ocr==0.6.13's API exactly. The
#  current PyPI release (0.22.1) removed all of that in favor of a
#  VLM-based RecognitionPredictor/DetectionPredictor setup that expects a
#  running model server -- `pip install surya-ocr` today gives you a
#  library your existing code cannot import at all.
#
#  Rather than pin an old dependency indefinitely, this reuses
#  pytesseract -- which you already have working correctly in
#  ocr_region.py for hybrid-page regions -- for full scanned pages too.
#  One OCR engine total, no extra model download, verified directly
#  against real scanned resumes (Priya Sharma, Amol Jadhav) before this
#  was written.
#
#── REVISION — word-level extraction + layout reconstruction ──
#Confirmed on a real scanned job document (SMFG_Job_Document_Scanned.pdf):
#the previous version of this file called pytesseract.image_to_string(),
#which returns a single flat string using TESSERACT'S OWN internal line
#ordering -- no word positions are kept, so there was no way to detect or
#fix a scrambled table/sidebar layout afterward. A 2-column label/value
#table ("Job Title" | "Asst. Manager, Cyber Defence") came out with rows
#in the wrong order because tesseract's own reading order doesn't
#understand table columns.
#
#This version instead calls pytesseract.image_to_data() (word-level
#bounding boxes -- the same call ocr_region.py already uses successfully
#for hybrid-page image regions), converts each word's pixel bbox into PDF
#point coordinates, and runs the resulting word list through
#ocr_layout_reconstruction.reconstruct_page_text() -- the SAME sidebar /
#two-column / single-column logic pdf_reader.py already uses for native
#PDF text. Verified directly against the SMFG table's real OCR'd
#coordinates: this correctly reconstructs
#    Job Title
#    Asst. Manager, Cyber Defence
#    Department
#    InfoSec
#    ...
#instead of the old scrambled order.
#
#PRODUCTION SAFETY: this is wrapped in a per-page try/except. If word-
#level extraction or layout reconstruction fails or raises for ANY
#reason, that single page falls back to the OLD plain image_to_string()
#behavior -- it does not raise, and it does not affect any other page or
#any other file in a batch run. A warning is logged (once per process,
#not once per page, to avoid spamming logs across a large batch) the
#first time this fallback triggers, so you can find out it happened
#without your terminal filling up across 60+ resumes.
#
#Known quality quirks to expect (seen on real test files, not
#hypothetical, unchanged from before this revision):
#  - "AI" frequently reads as "Al" (capital I / lowercase l confusion).
#    Downstream cleanup can regex-fix this near known tech terms if it
#    matters for skill extraction.
#  - "•" bullets occasionally misread as a bare "e" at line-start. Not
#    caught by section_splitter's _LEADING_BULLET_RE (which expects
#    real bullet glyphs), so these lines just won't get bullet
#    normalization -- harmless, but flagging so it's not a surprise
#    when spot-checking output.
#
#── REVISION — pre-OCR image quality gate ──
#Added the same preprocessing step ocr_reader.py (Surya) now uses:
#ingestion/image_preprocessing.py's preprocess_scanned_image() runs on
#every rendered page BEFORE any OCR happens (word-level or plain-string
#fallback), applying:
#
#  1. Blur rejection (raises ImageQualityError) -- see that module's
#     calibration table. IMPORTANT: this file renders at 300 DPI while
#     ocr_reader.py's Surya path renders at 200 DPI. Raw Laplacian
#     variance is resolution-dependent -- confirmed directly: the same
#     physical blur scored 10.18 at 200 DPI but only 6.77 at 300 DPI,
#     nearly a 2x difference. image_preprocessing.compute_blur_score()
#     normalizes to a fixed reference width before scoring specifically
#     so ONE threshold (MIN_BLUR_VARIANCE) is valid for both this file's
#     300 DPI render and ocr_reader.py's 200 DPI render -- don't bypass
#     that normalization or re-derive a separate threshold for this file.
#  2. Deskew -- corrects page rotation before word-level OCR runs. This
#     matters even more here than for the old plain-string path: a
#     rotated page feeds bad x0/top coordinates into
#     reconstruct_page_text(), which would compound the exact table-
#     scrambling problem this revision's layout fix was built to solve
#     in the first place.
#  3. Brightness correction -- inverts moderately dark whole-page scans.
#
#Preprocessing happens ONCE per page, before the try/except that guards
#layout reconstruction -- NOT inside _ocr_page_with_layout() or
#_ocr_page_plain_fallback() individually. This is deliberate: those two
#functions now both take an already-preprocessed PIL Image (not a raw
#fitz page) precisely so a genuine ImageQualityError propagates straight
#up to the caller as a hard rejection, rather than being caught by the
#layout-reconstruction try/except and silently retried via
#_ocr_page_plain_fallback() -- which cannot recover a blurry image any
#better than the word-level path can, so retrying it would just waste
#time before failing the same way, or worse, silently return low-
#confidence output instead of a clear rejection.
#
#── REVISION (this version) — image-based table grid detection ──
#Previously, table structure on this path came from
#ocr_layout_reconstruction.py's _reconstruct_table(), which guessed row/
#column boundaries from gaps between tesseract's word-level bounding
#boxes (a real technique, but tuned against only one hand-estimated
#reconstruction of a real table's coordinates -- see that module's own
#docstring, which flags _TABLE_MIN_GAP as "a starting point, not a
#calibrated constant").
#
#This revision adds ingestion/table_grid_detector.py's
#detect_and_ocr_tables() as a FIRST pass, run directly on the page image
#before any word-level OCR happens. It finds a table's actual printed
#grid lines (real black rules) via OpenCV, and OCRs each cell on its
#own -- structure comes from physical pixels, not from guessing at word
#spacing. Verified directly against SMFG_Job_Document_Scanned.pdf and a
#real scanned resume's 4-column academic-qualifications table -- see
#table_grid_detector.py's module docstring for full verification notes.
#
#Any tesseract word whose center falls inside a detected table region is
#removed from the word list before reconstruct_page_text() runs (its
#text is about to be replaced by the table's own, more accurate,
#per-cell OCR), and the table's real text is spliced back into the final
#output at the correct position. Because those words are removed first,
#_reconstruct_table() inside reconstruct_page_text() naturally has
#nothing left to (mis)detect in that region -- it remains in place,
#unmodified, purely as a fallback for a borderless table that
#detect_and_ocr_tables() didn't find (no drawn grid lines to detect),
#which is a reasonable belt-and-suspenders safety net rather than the
#sole/primary table-detection strategy it was before.
#
#Pages with no detected table (the large majority) are completely
#unaffected: detect_and_ocr_tables() returns [], the word list is
#unchanged, and everything proceeds exactly as it did before this
#revision.
#"""
#
#import logging
#from pathlib import Path
#import fitz  # PyMuPDF
#import pytesseract
#from PIL import Image, ImageOps
#
#from ingestion.ocr_layout_reconstruction import reconstruct_page_text
#from ingestion.image_preprocessing import preprocess_scanned_image, ImageQualityError
#from ingestion.table_grid_detector import (
#    detect_and_ocr_tables,
#    merge_tables_into_words,
#    splice_placeholders,
#)
#
#logger = logging.getLogger(__name__)
#
## 300 DPI balances OCR accuracy against render/OCR time. Confirmed
## sufficient on real resume scans; bump higher only if you see garbled
## small print (e.g. footnote-sized certifications lists). NOTE: this
## differs from ocr_reader.py's Surya path (200 DPI) -- see the module
## docstring's REVISION note on why the blur quality gate is resolution-
## normalized so one threshold works correctly for both.
#OCR_DPI = 300
#OCR_PSM = 3  # "fully automatic page segmentation" -- right default for
#             # a whole page with mixed paragraphs/headings/lists, unlike
#             # ocr_region.py's PSM 6 which assumes ONE uniform text block
#             # (fine for a single-line contact bar, wrong for a full page)
#
## Converts a pixel coordinate rendered at OCR_DPI back into PDF points
## (72 points per inch, always -- this has nothing to do with OCR_DPI
## itself, it's just the PDF unit definition). Used to build word dicts
## in the same coordinate space pdfplumber's extract_words() already
## uses, so reconstruct_page_text() doesn't need to know or care whether
## a word came from native PDF text or from OCR.
#_PDF_POINTS_PER_INCH = 72
#_PIXEL_TO_POINT = _PDF_POINTS_PER_INCH / OCR_DPI
#
#_word_extraction_warned = False
#
#
#def _warn_once(message: str):
#    global _word_extraction_warned
#    if not _word_extraction_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _word_extraction_warned = True
#
#
#def _render_page(fitz_page, dpi: int = OCR_DPI) -> Image.Image:
#    zoom = dpi / 72
#    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
#    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
#
#
#def _extract_words_from_image(img: Image.Image) -> list:
#    """
#    Runs word-level OCR via pytesseract.image_to_data() and returns a
#    list of word dicts in PDF point coordinates: {"text","x0","x1",
#    "top","bottom","height"} -- the same shape pdfplumber's
#    extract_words() and ocr_region.py's ocr_region_to_words() both use,
#    so reconstruct_page_text() can treat them identically.
#
#    img is expected to already be preprocessed (blur-gated, deskewed,
#    brightness-corrected) by the caller -- see read_with_surya_pages().
#    """
#    # Confirmed on the SMFG table: text sitting on a colored (green)
#    # table-cell background is missed almost entirely by tesseract on
#    # the raw RGB render -- the labels ("Job Title", "Department",
#    # "Grade", "IC/PM") simply don't come back as detected words at
#    # all. Converting to grayscale first fixes this; ocr_region.py
#    # already does the same thing for hybrid-page image regions for
#    # the same reason (see its own comment: "notably improves
#    # confidence on small embedded-image text").
#    img = ImageOps.grayscale(img)
#    data = pytesseract.image_to_data(img, config=f"--psm {OCR_PSM}",
#                                      output_type=pytesseract.Output.DICT)
#    words = []
#    n = len(data["text"])
#    for i in range(n):
#        text = data["text"][i].strip()
#        if not text:
#            continue
#        x, y, w, h = (data["left"][i], data["top"][i],
#                      data["width"][i], data["height"][i])
#        words.append({
#            "text": text,
#            "x0": x * _PIXEL_TO_POINT,
#            "x1": (x + w) * _PIXEL_TO_POINT,
#            "top": y * _PIXEL_TO_POINT,
#            "bottom": (y + h) * _PIXEL_TO_POINT,
#            "height": h * _PIXEL_TO_POINT,
#        })
#    return words
#
#
#def _ocr_page_with_layout(img: Image.Image, page_width_pts: float) -> str:
#    """
#    Table detection -> word-level OCR -> layout reconstruction, for one
#    already-rendered-and-preprocessed page image. Raises on any
#    failure -- caller (read_with_surya_pages) catches and falls back to
#    the old plain-text behavior for just this page.
#
#    See module docstring's REVISION note: detect_and_ocr_tables() runs
#    on the image directly (finds real printed grid lines), independent
#    of tesseract's word boxes. Any word landing inside a detected table
#    is removed before reconstruct_page_text() runs, and the table's own
#    per-cell OCR text is spliced back in afterward at the correct
#    position.
#    """
#    tables = detect_and_ocr_tables(img)
#    words = _extract_words_from_image(img)
#
#    if tables:
#        # IMPORTANT: tables are in raw PIXEL coordinates (as returned by
#        # detect_and_ocr_tables(), which works directly on the rendered
#        # image before any point-conversion happens), but `words` here
#        # has ALREADY been converted into PDF points by
#        # _extract_words_from_image(). pixel_to_point=_PIXEL_TO_POINT
#        # converts the table bboxes into that same point-space so the
#        # two line up correctly -- passing 1.0 here would silently
#        # misplace every detected table (a real bug caught during
#        # review, not a hypothetical one -- worth remembering if this
#        # function is ever refactored).
#        words, placeholder_map = merge_tables_into_words(
#            words, tables, pixel_to_point=_PIXEL_TO_POINT
#        )
#    else:
#        placeholder_map = {}
#
#    text = reconstruct_page_text(words, page_width_pts)
#    return splice_placeholders(text, placeholder_map)
#
#
#def _ocr_page_plain_fallback(img: Image.Image) -> str:
#    """The old, pre-revision behavior: whole-page OCR with no layout
#    awareness at all. Used only when _ocr_page_with_layout() fails.
#    Takes an already-rendered-and-preprocessed image, same as the
#    layout-aware path -- NOT a raw fitz page, so this never re-triggers
#    the blur/deskew/brightness gate a second time for the same page."""
#    return pytesseract.image_to_string(img, config=f"--psm {OCR_PSM}").strip()
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    Same contract as ocr_reader.read_with_surya_pages(): OCRs only the
#    given 0-based page indices of a PDF, returns {page_index: text}.
#    Name kept identical so pdf_reader.py's call site doesn't need to
#    change beyond the import line.
#
#    Each page is rendered and preprocessed (blur gate + deskew +
#    brightness) ONCE, before layout reconstruction is attempted. A
#    genuine ImageQualityError from that step is NOT caught here -- it
#    propagates straight up to the caller as a hard rejection (see
#    module docstring). Only layout-reconstruction failures (a different,
#    non-quality-related problem) fall back to plain OCR for that page.
#    """
#    if not page_indices:
#        return {}
#
#    results = {}
#    with fitz.open(pdf_path) as doc:
#        for page_index in page_indices:
#            page = doc[page_index]
#            img = _render_page(page)
#            # Raises ImageQualityError on genuinely bad pages -- deliberately
#            # NOT inside the try/except below, so it propagates as a hard
#            # rejection instead of triggering the plain-OCR fallback.
#            img = preprocess_scanned_image(img, context=f"page {page_index + 1} of {pdf_path}")
#            page_width_pts = page.rect.width
#            try:
#                text = _ocr_page_with_layout(img, page_width_pts)
#            except Exception as e:
#                _warn_once(
#                    f"Word-level layout reconstruction failed on at least one "
#                    f"page ({type(e).__name__}: {e}). Falling back to plain "
#                    f"whole-page OCR for any page where this happens -- "
#                    f"other pages/files are unaffected."
#                )
#                text = _ocr_page_plain_fallback(img)
#            results[page_index] = text
#
#    return results
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Same contract as ocr_reader.read_with_surya(): whole-file OCR for a
#    scanned PDF or a standalone image file, joined into one string.
#    Kept for parity in case anything calls the non-paged version.
#
#    Same preprocessing-before-layout-reconstruction ordering as
#    read_with_surya_pages() above, for the same reason: a real
#    ImageQualityError must propagate as a hard rejection, not get
#    caught by the layout-reconstruction fallback.
#    """
#    path = Path(file_path)
#
#    if path.suffix.lower() == ".pdf":
#        with fitz.open(file_path) as doc:
#            texts = []
#            for page in doc:
#                img = _render_page(page)
#                img = preprocess_scanned_image(img, context=file_path)
#                try:
#                    texts.append(_ocr_page_with_layout(img, page.rect.width))
#                except Exception as e:
#                    _warn_once(
#                        f"Word-level layout reconstruction failed on at least "
#                        f"one page ({type(e).__name__}: {e}). Falling back to "
#                        f"plain whole-page OCR for any page where this happens."
#                    )
#                    texts.append(_ocr_page_plain_fallback(img))
#    elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#        img = Image.open(file_path).convert("RGB")
#        img = preprocess_scanned_image(img, context=file_path)
#        # Standalone images have no fitz page / PDF point space to
#        # convert into -- pixel coordinates ARE the coordinate space
#        # here, which is fine since reconstruct_page_text() only cares
#        # about relative positions, not absolute units.
#        try:
#            tables = detect_and_ocr_tables(img)
#            gray = ImageOps.grayscale(img)
#            data = pytesseract.image_to_data(gray, config=f"--psm {OCR_PSM}",
#                                              output_type=pytesseract.Output.DICT)
#            words = []
#            n = len(data["text"])
#            for i in range(n):
#                t = data["text"][i].strip()
#                if not t:
#                    continue
#                x, y, w, h = (data["left"][i], data["top"][i],
#                              data["width"][i], data["height"][i])
#                words.append({"text": t, "x0": x, "x1": x + w,
#                              "top": y, "bottom": y + h, "height": h})
#            if tables:
#                # Here words ARE still in raw pixel coordinates (no PDF-point
#                # conversion for standalone images -- see comment above), so
#                # table pixel coordinates need no scaling either.
#                words, placeholder_map = merge_tables_into_words(
#                    words, tables, pixel_to_point=1.0
#                )
#            else:
#                placeholder_map = {}
#            page_text = reconstruct_page_text(words, img.width)
#            texts = [splice_placeholders(page_text, placeholder_map)]
#        except Exception as e:
#            _warn_once(
#                f"Word-level layout reconstruction failed on standalone "
#                f"image ({type(e).__name__}: {e}). Falling back to plain OCR."
#            )
#            texts = [pytesseract.image_to_string(img, config=f"--psm {OCR_PSM}").strip()]
#    else:
#        raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#
#    return "\n\n".join(t for t in texts if t)
#
#
#if __name__ == "__main__":
#    import sys
#    result = read_with_surya(sys.argv[1])
#    print(result)
#








#commenting this to check the code pasted above after coming from home works or not-
#"""
#ocr_reader_pytesseract.py — Layer 0, full-page OCR (scanned pages)
#
#Drop-in replacement for read_with_surya_pages() in ocr_reader.py. Same
#signature, same return shape ({page_index: text}), so pdf_reader.py needs
#exactly ONE line changed to switch backends:
#
#    # from ingestion.ocr_reader import read_with_surya_pages
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages
#
#Why this exists instead of fixing ocr_reader.py's Surya calls:
#  Your ocr_reader.py imports (surya.model.detection.model.load_model,
#  surya.ocr.run_ocr, etc.) match surya-ocr==0.6.13's API exactly. The
#  current PyPI release (0.22.1) removed all of that in favor of a
#  VLM-based RecognitionPredictor/DetectionPredictor setup that expects a
#  running model server -- `pip install surya-ocr` today gives you a
#  library your existing code cannot import at all.
#
#  Rather than pin an old dependency indefinitely, this reuses
#  pytesseract -- which you already have working correctly in
#  ocr_region.py for hybrid-page regions -- for full scanned pages too.
#  One OCR engine total, no extra model download, verified directly
#  against real scanned resumes (Priya Sharma, Amol Jadhav) before this
#  was written.
#
#Known quality quirks to expect (seen on real test files, not
#hypothetical):
#  - "AI" frequently reads as "Al" (capital I / lowercase l confusion).
#    Downstream cleanup can regex-fix this near known tech terms if it
#    matters for skill extraction.
#  - "•" bullets occasionally misread as a bare "e" at line-start. Not
#    caught by section_splitter's _LEADING_BULLET_RE (which expects
#    real bullet glyphs), so these lines just won't get bullet
#    normalization -- harmless, but flagging so it's not a surprise
#    when spot-checking output.
#"""
#
#from pathlib import Path
#import fitz  # PyMuPDF
#import pytesseract
#from PIL import Image
#
## 300 DPI balances OCR accuracy against render/OCR time. Confirmed
## sufficient on real resume scans; bump higher only if you see garbled
## small print (e.g. footnote-sized certifications lists).
#OCR_DPI = 300
#OCR_PSM = 3  # "fully automatic page segmentation" -- right default for
#             # a whole page with mixed paragraphs/headings/lists, unlike
#             # ocr_region.py's PSM 6 which assumes ONE uniform text block
#             # (fine for a single-line contact bar, wrong for a full page)
#
#
#def _render_page(fitz_page, dpi: int = OCR_DPI) -> Image.Image:
#    zoom = dpi / 72
#    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
#    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    Same contract as ocr_reader.read_with_surya_pages(): OCRs only the
#    given 0-based page indices of a PDF, returns {page_index: text}.
#    Name kept identical so pdf_reader.py's call site doesn't need to
#    change beyond the import line.
#    """
#    if not page_indices:
#        return {}
#
#    results = {}
#    with fitz.open(pdf_path) as doc:
#        for page_index in page_indices:
#            page = doc[page_index]
#            img = _render_page(page)
#            text = pytesseract.image_to_string(img, config=f"--psm {OCR_PSM}")
#            results[page_index] = text.strip()
#
#    return results
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Same contract as ocr_reader.read_with_surya(): whole-file OCR for a
#    scanned PDF or a standalone image file, joined into one string.
#    Kept for parity in case anything calls the non-paged version.
#    """
#    path = Path(file_path)
#
#    if path.suffix.lower() == ".pdf":
#        with fitz.open(file_path) as doc:
#            texts = [
#                pytesseract.image_to_string(_render_page(doc[i]), config=f"--psm {OCR_PSM}").strip()
#                for i in range(len(doc))
#            ]
#    elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#        img = Image.open(file_path).convert("RGB")
#        texts = [pytesseract.image_to_string(img, config=f"--psm {OCR_PSM}").strip()]
#    else:
#        raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#
#    return "\n\n".join(t for t in texts if t)
#
#
#if __name__ == "__main__":
#    import sys
#    result = read_with_surya(sys.argv[1])
#    print(result)
#
#