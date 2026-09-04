"""
ocr_reader.py — Layer 0, full-page OCR (scanned pages)

Surya is the PRIMARY OCR engine. pytesseract is an automatic FALLBACK,
used only if Surya can't be imported, can't load its models, or throws
during inference. You don't need to choose one manually -- this module
tries Surya first every time and only drops to tesseract if that fails.

TARGETS: surya-ocr==0.14.7 (confirmed via `pip show surya-ocr` against
the actual installed environment -- see version history below for why
this matters more than it sounds like it should).

── Three different Surya APIs found across versions, confirmed by
   downloading and inspecting the actual PyPI packages, not guessing ──

1. surya-ocr==0.6.13 -- what the ORIGINAL ocr_reader.py was written
   against: standalone functions, `surya.model.detection.model.load_model`,
   `surya.model.recognition.model.load_model`, `surya.ocr.run_ocr(images,
   langs, det_model, det_processor, rec_model, rec_processor)`. Even in
   this exact version the original code had a wrong import path
   (`surya.model.detection.PROCESSOR.load_processor` -- that function
   actually lives in `surya.model.detection.MODEL`, confirmed against
   the real file contents).

2. surya-ocr==0.14.7 -- what's ACTUALLY INSTALLED in your venv (confirmed
   via your `pip show surya-ocr` output). No `surya.model` package at
   all anymore -- detection and recognition are now predictor CLASSES:
   `surya.detection.DetectionPredictor`, `surya.recognition.RecognitionPredictor`.
   No `langs` parameter either (dropped in this generation). Call shape:

       det_predictor = DetectionPredictor()
       rec_predictor = RecognitionPredictor()
       results = rec_predictor(images, det_predictor=det_predictor)

   This is the version this file is now written against.

3. surya-ocr==0.22.x (current PyPI default if you `pip install surya-ocr`
   with no version pin) -- a further VLM-based rewrite requiring a
   running model server. Different again from both of the above.

The lesson: don't assume a Surya upgrade/reinstall keeps the same call
shape. If you ever bump this dependency, re-check `pip show surya-ocr`
and re-verify the import shape before assuming this file still works --
it's changed shape at least 3 times.

── Fallback behaviour ──
If Surya's imports fail, its models fail to load, or the predictor call
throws for any reason, this module logs a warning once and falls back
to ocr_reader_pytesseract.py for that call. The fallback is all-or-
nothing per batch (not per-page) -- if Surya breaks partway through a
multi-page document, the whole document's scanned pages go to
tesseract rather than trying to reconcile a half-Surya, half-tesseract
result.

── Model caching ──
Your original _load_surya_models() reloaded all 4 models on every call
to read_with_surya_pages(). Across a batch of 60+ resumes that's 60+
model loads. Predictors are now cached at module level after first load.

── REVISION — broadened LaTeX/math markup cleanup ──
Confirmed on a real scanned job description (SMFG_Job_Document_Scanned
.pdf): the previous _clean_surya_text() only stripped the specific
<math>\bullet</math> pattern plus a handful of inline formatting tags
(b/i/u/del/mark/sup/sub/br). That's too narrow. Surya's format-aware
recognition model also wraps STACKED/logo-style text -- e.g. a company
logo rendered as "SMFG" on one line with "IndiaCredit" in smaller type
underneath it -- in full LaTeX subscript/font markup that looks nothing
like the bullet case:

    \mathop{\hbox{\rm SMFG}}_{\hbox{\scriptsize \rm Indiacredit}}

_strip_latex_markup() below is a general-purpose peel that handles this
and the old bullet case with one shared implementation. This part WAS
verified directly (tested against the exact leaked string above and
the pre-existing bullet-math case).

── REVISION — layout-aware page reconstruction (SUPERSEDED, see below) ──
An earlier version of this file tried reusing reconstruct_page_text()
(the word-level sidebar/two-column-aware function) for Surya's output,
by treating each Surya TextLine as one "word". That was flagged at the
time as unverified against a real Surya model. It has now BEEN tested
against real Surya output, and it actively broke things -- see the
next REVISION note below. Do not reintroduce reconstruct_page_text()
here without re-reading that note.

── REVISION — switched to sort_lines_only(), NOT
   reconstruct_page_text() ──
Confirmed on a real scanned resume (Anandu Chandran,
PDF_image2pdf_20260821121805.pdf): reconstruct_page_text() badly
scrambled a completely ordinary SINGLE-COLUMN resume. Two different,
unrelated lines (e.g. two lines of a center-aligned address block) got
merged onto one output line and reordered left-to-right, because
Surya provides LINE-granularity bounding boxes (one box per whole
line), not word-granularity -- and reconstruct_page_text()'s merging
and sidebar/two-column logic is calibrated for word-level data, where
merging nearby-top fragments into one line is the whole point. Applied
to already-whole-line data, that same merging step can wrongly fuse
two DIFFERENT lines together whenever their measured "top" values
happen to land close to each other, which is easy to hit with center-
aligned text (every line has a different x0) or a short heading next
to a wrapped bullet line. Reproduced exactly against this real
resume's garbled output using plausible coordinates for the two lines
that got fused -- see ocr_layout_reconstruction.py's module docstring
for the full trace.

Fixed by switching to ocr_layout_reconstruction.sort_lines_only():
sorts Surya's lines by (top, x0) and emits one output line per input
line, with NO merging and NO sidebar/two-column detection at all.
Trade-off, accepted deliberately: a short heading can occasionally
land one line off from its ideal position if Surya's box for it is
very slightly mismeasured (cosmetic). In exchange, two different
lines' text can never be fused together again, which was the actually
severe failure mode breaking ordinary single-column resumes in
production. This gives up any "smart" sidebar/table reordering for a
scanned sidebar-style document processed via Surya specifically (that
capability was never verified working in the first place) in favor of
guaranteed correctness on the much more common single-column case.

reconstruct_page_text() itself is UNCHANGED and still correct --
ocr_reader_pytesseract.py's tesseract fallback path still uses it,
because tesseract's image_to_data() provides genuine word-level
fragments, which is what that function's thresholds and tolerances
are calibrated for.

── REVISION — pre-OCR image quality gate ──
Added a preprocessing step (ingestion/image_preprocessing.py) that runs
on every rendered/loaded image BEFORE it reaches Surya:

  1. Blur rejection -- raises ImageQualityError for pages too blurry to
     trust. Calibrated via a controlled blur ladder on a real resume:
     sentences stay grammatically intact but silently wrong fields
     (emails, dates, proper nouns) start appearing well before the
     image looks like obvious garbage -- so this is a hard reject, not
     a warning. The blur score is resolution-normalized (see that
     module's _BLUR_REFERENCE_WIDTH) so the same threshold is valid
     whether an image was rendered at this file's 200 DPI or the
     tesseract fallback's 300 DPI.
  2. Deskew -- corrects page rotation. Justified by a real finding on
     this project: a 4.5-degree rotation caused a section heading to be
     sorted into the middle of an unrelated bullet sentence, and
     dropped a line entirely, because reading-order reconstruction
     sorts by y-coordinate and a rotated page's "same visual line" has
     different y-coordinates on the left vs. right edge of the page.
  3. Brightness correction -- inverts pages that are moderately dark
     overall (a real underexposed photo).

ImageQualityError must propagate all the way up to the caller as a
genuine rejection -- it is explicitly re-raised BEFORE the generic
Surya-failure except-block below, so a blurry page is rejected outright
rather than silently retried with tesseract (tesseract cannot recover
detail a blur destroyed any better than Surya can).

── REVISION (this version) — image-based table grid detection ──
Replaced the previous in-file attempt to detect tables from Surya's
LINE-granularity boxes (which called ocr_layout_reconstruction.py's
_reconstruct_table() by pretending each Surya TextLine was a single
"word" -- see _text_line_to_word_dict(), now removed). That approach
reintroduced the exact line-fusion risk the "switched to
sort_lines_only()" revision above was written to eliminate: two
genuinely different lines landing close together vertically could get
merged into one fake table "row" and have their text sliced apart at
the wrong point. It was also never verified against real Surya output
before being written.

Replaced with ingestion/table_grid_detector.py's
detect_and_ocr_tables(), which finds table structure from the PAGE
IMAGE's actual printed grid lines (real black rules), independent of
Surya's boxes entirely, and OCRs each cell on its own. This is a
strict improvement, not a tuning tweak: table structure now comes from
physical pixels, not from guessing at Surya's line positions. Verified
directly against SMFG_Job_Document_Scanned.pdf (both pages) and
Anandu Chandran's real scanned resume (4-column academic-qualifications
table) -- see table_grid_detector.py's module docstring for the full
verification notes.

Tables are detected and OCR'd BEFORE Surya even runs (grid detection
only needs the image, not Surya's output). Any Surya line whose center
falls inside a detected table region is dropped from the line list (its
text is about to be replaced by the table's own, more accurate, per-
cell OCR) and the table's real text is spliced back into the final
output at the correct reading-order position -- see
table_grid_detector.py's merge_tables_into_words()/splice_placeholders()
docstrings for exactly how.

Pages with no detected table (the large majority) are completely
unaffected: detect_and_ocr_tables() returns [], merge_tables_into_words()
returns the original line list unchanged, and everything proceeds
exactly as it did before this revision.
"""

import logging
import re
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

from ingestion.ocr_layout_reconstruction import sort_lines_only
from ingestion.image_preprocessing import preprocess_scanned_image, ImageQualityError
from ingestion.table_grid_detector import (
    detect_and_ocr_tables,
    merge_tables_into_words,
    splice_placeholders,
)

logger = logging.getLogger(__name__)

_MATH_BULLET_RE = re.compile(r"<math[^>]*>\s*\\bullet\s*</math>", re.IGNORECASE)
_MATH_TAG_RE = re.compile(r"</?math[^>]*>", re.IGNORECASE)
_BARE_BULLET_CMD_RE = re.compile(r"\\bullet")
_FORMAT_TAG_RE = re.compile(r"</?(b|i|u|del|mark|sup|sub|br)\b[^>]*>", re.IGNORECASE)
_LONE_BULLET_GLYPH_LINE_RE = re.compile(r"^[\u0660\u06F0.]$")

# ── LaTeX/math markup cleanup (see REVISION note above) ──
_MATH_BLOCK_RE = re.compile(r"<math[^>]*>(.*?)</math>", re.IGNORECASE | re.DOTALL)
_LATEX_BRACE_CMD_RE = re.compile(r"\\[a-zA-Z]+\{([^{}]*)\}")
_LATEX_SCRIPT_RE = re.compile(r"[_^]\{([^{}]*)\}")
_LATEX_NOARG_CMD_RE = re.compile(
    r"\\(rm|bf|it|tt|sf|em|scriptsize|scriptstyle|displaystyle|textstyle|"
    r"normalsize|tiny|small|footnotesize|large|Large|LARGE|huge|Huge|"
    r"mathop|mathrm|mathbf|mathit)\b"
)
_LATEX_STRAY_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_LATEX_STRAY_BRACE_RE = re.compile(r"[{}]")


def _strip_latex_markup(text: str) -> str:
    """
    General-purpose LaTeX/math markup peel. Order matters:
      1. Unwrap <math>...</math> tags, keeping inner content.
      2. Repeatedly peel \\command{content} and _{content}/^{content}
         groups from the INSIDE OUT (handles nesting like
         \\mathop{\\hbox{\\rm SMFG}}, resolving one layer per loop
         iteration until stable).
      3. Turn \\bullet into a real "•" (before step 4's generic sweep
         would otherwise just delete it).
      4. Drop bare style commands with no braces left (\\rm,
         \\scriptsize, \\mathbf, etc.).
      5. Sweep up anything left over: stray \\commands with no braces,
         and any orphaned { or }.
      6. Collapse extra whitespace this stripping tends to leave.
    """
    text = _MATH_BLOCK_RE.sub(r"\1", text)
    prev = None
    while prev != text:
        prev = text
        text = _LATEX_BRACE_CMD_RE.sub(r"\1", text)
        text = _LATEX_SCRIPT_RE.sub(r"\1", text)
    text = _BARE_BULLET_CMD_RE.sub("•", text)
    text = _LATEX_NOARG_CMD_RE.sub("", text)
    text = _LATEX_STRAY_CMD_RE.sub("", text)
    text = _LATEX_STRAY_BRACE_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _clean_surya_text(text: str) -> str:
    """
    surya-ocr 0.14.7's recognition model is format-aware, not plain OCR:
    it wraps text it judged bold in <b>...</b>, tags bullet-point dots
    as <math display="inline">\\bullet</math>, and -- confirmed on a
    real scanned job description -- also wraps stacked/logo-style text
    in full LaTeX subscript markup. See _strip_latex_markup() above.

    Separately, the same model sometimes misreads the bullet glyph
    entirely as some other small-dot-shaped character -- an Arabic-Indic
    digit zero (٠, U+0660), Extended Arabic-Indic digit zero (۰,
    U+06F0), or even a plain period (.). These show up as a lone
    character alone on its own line where a bullet should be.

    This normalizes all of the above down to a plain "•" and strips
    other inline formatting tags to their plain text content, so
    section_splitter.py sees ordinary text instead of markup it was
    never built to parse.
    """
    text = _strip_latex_markup(text)
    text = _FORMAT_TAG_RE.sub("", text)
    cleaned_lines = [
        "•" if _LONE_BULLET_GLYPH_LINE_RE.match(line.strip()) else line
        for line in text.split("\n")
    ]
    return "\n".join(cleaned_lines)

# ── Surya availability check + predictor cache (populated lazily, once) ──
_surya_predictors = None   # (det_predictor, rec_predictor) once loaded
_surya_import_failed = False
_surya_warned = False
_bbox_warned = False

# Surya renders at 200 DPI (see pdf_to_images below). Used to convert a
# TextLine's bbox (pixel coordinates) into PDF points, same coordinate
# space pdfplumber's extract_words() and the tesseract-fallback path
# both use.
OCR_DPI = 200
_PDF_POINTS_PER_INCH = 72
_PIXEL_TO_POINT = _PDF_POINTS_PER_INCH / OCR_DPI

# Set to True (or set env var DEBUG_SURYA=1) to print every detected
# line's real (top, x0, confidence, text) BEFORE reconstruction runs --
# turn this on and re-run against a resume showing a heading-ordering
# problem (like "Career Objective:" landing after its own first
# content line) to see Surya's ACTUAL coordinates for that page,
# instead of guessing. Paste that debug output back for diagnosis.
import os
DEBUG_SURYA_COORDS = os.environ.get("DEBUG_SURYA", "0") == "1"


def _warn_once(message: str):
    global _surya_warned
    if not _surya_warned:
        logger.warning(message)
        print(f"[WARNING] {message}")
        _surya_warned = True


def _warn_bbox_once(message: str):
    global _bbox_warned
    if not _bbox_warned:
        logger.warning(message)
        print(f"[WARNING] {message}")
        _bbox_warned = True


def _load_surya_predictors():
    """Loads and caches Surya's detection + recognition predictors.
    Raises on any failure -- callers must catch and fall back to
    pytesseract."""
    global _surya_predictors
    if _surya_predictors is not None:
        return _surya_predictors

    # surya-ocr==0.14.7 API: predictor classes, not standalone
    # load_model()/load_processor() functions -- see module docstring.
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor

    print("[INFO] Loading Surya OCR predictors (first call only; cached after)...")
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    _surya_predictors = (det_predictor, rec_predictor)
    return _surya_predictors


def _text_line_to_line_dict(line) -> dict:
    """
    Converts one Surya TextLine into the dict shape sort_lines_only()
    expects: {"text", "x0", "top"}. Raises AttributeError if `.bbox`
    isn't present in the shape expected -- caller catches this per
    page (see PRODUCTION SAFETY note below).
    """
    x0, top, x1, bottom = line.bbox
    return {
        "text": line.text,
        "x0": x0 * _PIXEL_TO_POINT,
        "top": top * _PIXEL_TO_POINT,
    }


def _run_surya_on_images(images: list) -> list:
    """Raises on any failure -- callers must catch and fall back."""
    det_predictor, rec_predictor = _load_surya_predictors()

    # 0.14.7 dropped the explicit `langs` parameter entirely -- passing
    # it raises TypeError. Detection is supplied via det_predictor, and
    # recognition runs language-agnostic OCR by default.
    results = rec_predictor(images, det_predictor=det_predictor)

    all_text = []
    for page_idx, page_result in enumerate(results):
        confident_lines = [line for line in page_result.text_lines if line.confidence > 0.3]

        if DEBUG_SURYA_COORDS:
            print(f"[DEBUG] Surya line coordinates for page {page_idx + 1}:")
            for line in confident_lines:
                x0, top, x1, bottom = line.bbox
                print(f"  top={top * _PIXEL_TO_POINT:7.1f}  x0={x0 * _PIXEL_TO_POINT:7.1f}  "
                      f"conf={line.confidence:.2f}  text={line.text!r}")

        # ── Image-based table detection (see REVISION note in module
        # docstring) -- runs on the page IMAGE directly, independent of
        # Surya's line boxes entirely, so it carries none of the
        # line-fusion risk the old _reconstruct_table()-on-Surya-lines
        # approach had. Returns [] for the large majority of pages that
        # have no genuine bordered table, in which case everything
        # below behaves exactly as before this revision.
        tables = detect_and_ocr_tables(images[page_idx])

        try:
            line_dicts = [_text_line_to_line_dict(line) for line in confident_lines]
            if tables:
                line_dicts, placeholder_map = merge_tables_into_words(
                    line_dicts, tables, pixel_to_point=_PIXEL_TO_POINT
                )
            else:
                placeholder_map = {}
            page_text = sort_lines_only(line_dicts)
            if not page_text.strip():
                raise ValueError("reconstruction produced empty text")
        except Exception as e:
            _warn_bbox_once(
                f"Surya line-position reconstruction failed ({type(e).__name__}: {e}). "
                f"Falling back to plain engine-order line join for any page "
                f"where this happens -- other pages/files are unaffected. "
                f"This means Surya's TextLine shape in your installed version "
                f"may not expose .bbox as expected (see ocr_reader.py's "
                f"module docstring)."
            )
            page_text = "\n".join(line.text for line in confident_lines)
            placeholder_map = {}

        page_text = _clean_surya_text(page_text)
        # Splice real table text in AFTER _clean_surya_text() runs, so
        # the LaTeX/formatting cleanup (aimed at Surya's own markup
        # quirks) never has a chance to touch text that came from our
        # own separate cell-by-cell Tesseract OCR -- that text was never
        # wrapped in Surya's markup in the first place.
        page_text = splice_placeholders(page_text, placeholder_map)
        all_text.append(page_text)
    return all_text


def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
    """Converts pages of a PDF into PIL Images at 200 DPI (good balance
    of Surya accuracy vs. speed). page_indices=None converts every page.

    Each rendered page goes through preprocess_scanned_image() (blur
    gate + deskew + brightness correction) before being handed back to
    the caller. A page that fails the blur gate raises
    ImageQualityError here -- read_with_surya_pages() below is
    responsible for NOT swallowing that into the generic
    tesseract-fallback path."""
    doc = fitz.open(pdf_path)
    images = []
    indices = page_indices if page_indices is not None else range(len(doc))
    for page_num in indices:
        page = doc[page_num]
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = preprocess_scanned_image(img, context=f"page {page_num + 1} of {pdf_path}")
        images.append(img)
    doc.close()
    return images


def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
    """
    OCRs the given 0-based page indices, Surya first, tesseract fallback
    on any failure. Returns {page_index: text}. This is the function
    pdf_reader.py calls for pages classified "scanned".

    ImageQualityError (raised by pdf_to_images()'s preprocessing step)
    is deliberately re-raised BEFORE the generic except-block below, so
    a blur rejection is never mistaken for "Surya failed, try tesseract
    instead" -- tesseract cannot recover detail a blur destroyed any
    better than Surya can.
    """
    if not page_indices:
        return {}

    global _surya_import_failed
    if not _surya_import_failed:
        try:
            images = pdf_to_images(pdf_path, page_indices=page_indices)
            texts = _run_surya_on_images(images)
            return dict(zip(page_indices, texts))
        except ImageQualityError:
            raise
        except Exception as e:
            _surya_import_failed = True
            _warn_once(
                f"Surya OCR failed ({type(e).__name__}: {e}). "
                f"Falling back to pytesseract for this and all subsequent pages. "
                f"Check `pip show surya-ocr` is exactly 0.6.13, and that model "
                f"downloads aren't being blocked by network/firewall."
            )

    from ingestion.ocr_reader_pytesseract import read_with_surya_pages as tesseract_fallback
    return tesseract_fallback(pdf_path, page_indices)


def read_with_surya(file_path: str) -> str:
    """
    Whole-file OCR for a standalone image or a fully-scanned PDF,
    joined into one string. Surya first, tesseract fallback on failure.

    Standalone image uploads (.jpg/.png/etc.) also go through
    preprocess_scanned_image() before OCR, same as PDF pages. See
    read_with_surya_pages() docstring for why ImageQualityError is
    re-raised before the generic fallback except-block.
    """
    global _surya_import_failed
    path = Path(file_path)

    if not _surya_import_failed:
        try:
            if path.suffix.lower() == ".pdf":
                images = pdf_to_images(file_path)
            elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
                img = Image.open(file_path).convert("RGB")
                img = preprocess_scanned_image(img, context=file_path)
                images = [img]
            else:
                raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
            all_text = _run_surya_on_images(images)
            return "\n\n".join(all_text)
        except ImageQualityError:
            raise
        except Exception as e:
            _surya_import_failed = True
            _warn_once(
                f"Surya OCR failed ({type(e).__name__}: {e}). "
                f"Falling back to pytesseract."
            )

    from ingestion.ocr_reader_pytesseract import read_with_surya as tesseract_fallback
    return tesseract_fallback(file_path)


if __name__ == "__main__":
    import sys
    print(read_with_surya(sys.argv[1]))
















#"""
#ocr_reader.py — Layer 0, full-page OCR (scanned pages)
#
#Surya is the PRIMARY OCR engine. pytesseract is an automatic FALLBACK,
#used only if Surya can't be imported, can't load its models, or throws
#during inference. You don't need to choose one manually -- this module
#tries Surya first every time and only drops to tesseract if that fails.
#
#TARGETS: surya-ocr==0.14.7 (confirmed via `pip show surya-ocr` against
#the actual installed environment -- see version history below for why
#this matters more than it sounds like it should).
#
#── Three different Surya APIs found across versions, confirmed by
#   downloading and inspecting the actual PyPI packages, not guessing ──
#
#1. surya-ocr==0.6.13 -- what the ORIGINAL ocr_reader.py was written
#   against: standalone functions, `surya.model.detection.model.load_model`,
#   `surya.model.recognition.model.load_model`, `surya.ocr.run_ocr(images,
#   langs, det_model, det_processor, rec_model, rec_processor)`. Even in
#   this exact version the original code had a wrong import path
#   (`surya.model.detection.PROCESSOR.load_processor` -- that function
#   actually lives in `surya.model.detection.MODEL`, confirmed against
#   the real file contents).
#
#2. surya-ocr==0.14.7 -- what's ACTUALLY INSTALLED in your venv (confirmed
#   via your `pip show surya-ocr` output). No `surya.model` package at
#   all anymore -- detection and recognition are now predictor CLASSES:
#   `surya.detection.DetectionPredictor`, `surya.recognition.RecognitionPredictor`.
#   No `langs` parameter either (dropped in this generation). Call shape:
#
#       det_predictor = DetectionPredictor()
#       rec_predictor = RecognitionPredictor()
#       results = rec_predictor(images, det_predictor=det_predictor)
#
#   This is the version this file is now written against.
#
#3. surya-ocr==0.22.x (current PyPI default if you `pip install surya-ocr`
#   with no version pin) -- a further VLM-based rewrite requiring a
#   running model server. Different again from both of the above.
#
#The lesson: don't assume a Surya upgrade/reinstall keeps the same call
#shape. If you ever bump this dependency, re-check `pip show surya-ocr`
#and re-verify the import shape before assuming this file still works --
#it's changed shape at least 3 times.
#
#── Fallback behaviour ──
#If Surya's imports fail, its models fail to load, or the predictor call
#throws for any reason, this module logs a warning once and falls back
#to ocr_reader_pytesseract.py for that call. The fallback is all-or-
#nothing per batch (not per-page) -- if Surya breaks partway through a
#multi-page document, the whole document's scanned pages go to
#tesseract rather than trying to reconcile a half-Surya, half-tesseract
#result.
#
#── Model caching ──
#Your original _load_surya_models() reloaded all 4 models on every call
#to read_with_surya_pages(). Across a batch of 60+ resumes that's 60+
#model loads. Predictors are now cached at module level after first load.
#
#── REVISION — broadened LaTeX/math markup cleanup ──
#Confirmed on a real scanned job description (SMFG_Job_Document_Scanned
#.pdf): the previous _clean_surya_text() only stripped the specific
#<math>\bullet</math> pattern plus a handful of inline formatting tags
#(b/i/u/del/mark/sup/sub/br). That's too narrow. Surya's format-aware
#recognition model also wraps STACKED/logo-style text -- e.g. a company
#logo rendered as "SMFG" on one line with "IndiaCredit" in smaller type
#underneath it -- in full LaTeX subscript/font markup that looks nothing
#like the bullet case:
#
#    \mathop{\hbox{\rm SMFG}}_{\hbox{\scriptsize \rm Indiacredit}}
#
#_strip_latex_markup() below is a general-purpose peel that handles this
#and the old bullet case with one shared implementation. This part WAS
#verified directly (tested against the exact leaked string above and
#the pre-existing bullet-math case).
#
#── REVISION — layout-aware page reconstruction (SUPERSEDED, see below) ──
#An earlier version of this file tried reusing reconstruct_page_text()
#(the word-level sidebar/two-column-aware function) for Surya's output,
#by treating each Surya TextLine as one "word". That was flagged at the
#time as unverified against a real Surya model. It has now BEEN tested
#against real Surya output, and it actively broke things -- see the
#next REVISION note below. Do not reintroduce reconstruct_page_text()
#here without re-reading that note.
#
#── REVISION — switched to sort_lines_only(), NOT
#   reconstruct_page_text() ──
#Confirmed on a real scanned resume (Anandu Chandran,
#PDF_image2pdf_20260821121805.pdf): reconstruct_page_text() badly
#scrambled a completely ordinary SINGLE-COLUMN resume. Two different,
#unrelated lines (e.g. two lines of a center-aligned address block) got
#merged onto one output line and reordered left-to-right, because
#Surya provides LINE-granularity bounding boxes (one box per whole
#line), not word-granularity -- and reconstruct_page_text()'s merging
#and sidebar/two-column logic is calibrated for word-level data, where
#merging nearby-top fragments into one line is the whole point. Applied
#to already-whole-line data, that same merging step can wrongly fuse
#two DIFFERENT lines together whenever their measured "top" values
#happen to land close to each other, which is easy to hit with center-
#aligned text (every line has a different x0) or a short heading next
#to a wrapped bullet line. Reproduced exactly against this real
#resume's garbled output using plausible coordinates for the two lines
#that got fused -- see ocr_layout_reconstruction.py's module docstring
#for the full trace.
#
#Fixed by switching to ocr_layout_reconstruction.sort_lines_only():
#sorts Surya's lines by (top, x0) and emits one output line per input
#line, with NO merging and NO sidebar/two-column detection at all.
#Trade-off, accepted deliberately: a short heading can occasionally
#land one line off from its ideal position if Surya's box for it is
#very slightly mismeasured (cosmetic). In exchange, two different
#lines' text can never be fused together again, which was the actually
#severe failure mode breaking ordinary single-column resumes in
#production. This gives up any "smart" sidebar/table reordering for a
#scanned sidebar-style document processed via Surya specifically (that
#capability was never verified working in the first place) in favor of
#guaranteed correctness on the much more common single-column case.
#
#reconstruct_page_text() itself is UNCHANGED and still correct --
#ocr_reader_pytesseract.py's tesseract fallback path still uses it,
#because tesseract's image_to_data() provides genuine word-level
#fragments, which is what that function's thresholds and tolerances
#are calibrated for.
#
#── REVISION — pre-OCR image quality gate ──
#Added a preprocessing step (ingestion/image_preprocessing.py) that runs
#on every rendered/loaded image BEFORE it reaches Surya:
#
#  1. Blur rejection -- raises ImageQualityError for pages too blurry to
#     trust. Calibrated via a controlled blur ladder on a real resume:
#     sentences stay grammatically intact but silently wrong fields
#     (emails, dates, proper nouns) start appearing well before the
#     image looks like obvious garbage -- so this is a hard reject, not
#     a warning. The blur score is resolution-normalized (see that
#     module's _BLUR_REFERENCE_WIDTH) so the same threshold is valid
#     whether an image was rendered at this file's 200 DPI or the
#     tesseract fallback's 300 DPI.
#  2. Deskew -- corrects page rotation. Justified by a real finding on
#     this project: a 4.5-degree rotation caused a section heading to be
#     sorted into the middle of an unrelated bullet sentence, and
#     dropped a line entirely, because reading-order reconstruction
#     sorts by y-coordinate and a rotated page's "same visual line" has
#     different y-coordinates on the left vs. right edge of the page.
#  3. Brightness correction -- inverts pages that are moderately dark
#     overall (a real underexposed photo).
#
#ImageQualityError must propagate all the way up to the caller as a
#genuine rejection -- it is explicitly re-raised BEFORE the generic
#Surya-failure except-block below, so a blurry page is rejected outright
#rather than silently retried with tesseract (tesseract cannot recover
#detail a blur destroyed any better than Surya can).
#
#── REVISION (this version) — image-based table grid detection ──
#Replaced the previous in-file attempt to detect tables from Surya's
#LINE-granularity boxes (which called ocr_layout_reconstruction.py's
#_reconstruct_table() by pretending each Surya TextLine was a single
#"word" -- see _text_line_to_word_dict(), now removed). That approach
#reintroduced the exact line-fusion risk the "switched to
#sort_lines_only()" revision above was written to eliminate: two
#genuinely different lines landing close together vertically could get
#merged into one fake table "row" and have their text sliced apart at
#the wrong point. It was also never verified against real Surya output
#before being written.
#
#Replaced with ingestion/table_grid_detector.py's
#detect_and_ocr_tables(), which finds table structure from the PAGE
#IMAGE's actual printed grid lines (real black rules), independent of
#Surya's boxes entirely, and OCRs each cell on its own. This is a
#strict improvement, not a tuning tweak: table structure now comes from
#physical pixels, not from guessing at Surya's line positions. Verified
#directly against SMFG_Job_Document_Scanned.pdf (both pages) and
#Anandu Chandran's real scanned resume (4-column academic-qualifications
#table) -- see table_grid_detector.py's module docstring for the full
#verification notes.
#
#Tables are detected and OCR'd BEFORE Surya even runs (grid detection
#only needs the image, not Surya's output). Any Surya line whose center
#falls inside a detected table region is dropped from the line list (its
#text is about to be replaced by the table's own, more accurate, per-
#cell OCR) and the table's real text is spliced back into the final
#output at the correct reading-order position -- see
#table_grid_detector.py's merge_tables_into_words()/splice_placeholders()
#docstrings for exactly how.
#
#Pages with no detected table (the large majority) are completely
#unaffected: detect_and_ocr_tables() returns [], merge_tables_into_words()
#returns the original line list unchanged, and everything proceeds
#exactly as it did before this revision.
#"""
#
#import logging
#import re
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF
#
#from ingestion.ocr_layout_reconstruction import sort_lines_only
#from ingestion.image_preprocessing import preprocess_scanned_image, ImageQualityError
#from ingestion.table_grid_detector import (
#    detect_and_ocr_tables,
#    merge_tables_into_words,
#    splice_placeholders,
#)
#
#logger = logging.getLogger(__name__)
#
#_MATH_BULLET_RE = re.compile(r"<math[^>]*>\s*\\bullet\s*</math>", re.IGNORECASE)
#_MATH_TAG_RE = re.compile(r"</?math[^>]*>", re.IGNORECASE)
#_BARE_BULLET_CMD_RE = re.compile(r"\\bullet")
#_FORMAT_TAG_RE = re.compile(r"</?(b|i|u|del|mark|sup|sub|br)\b[^>]*>", re.IGNORECASE)
#_LONE_BULLET_GLYPH_LINE_RE = re.compile(r"^[\u0660\u06F0.]$")
#
## ── LaTeX/math markup cleanup (see REVISION note above) ──
#_MATH_BLOCK_RE = re.compile(r"<math[^>]*>(.*?)</math>", re.IGNORECASE | re.DOTALL)
#_LATEX_BRACE_CMD_RE = re.compile(r"\\[a-zA-Z]+\{([^{}]*)\}")
#_LATEX_SCRIPT_RE = re.compile(r"[_^]\{([^{}]*)\}")
#_LATEX_NOARG_CMD_RE = re.compile(
#    r"\\(rm|bf|it|tt|sf|em|scriptsize|scriptstyle|displaystyle|textstyle|"
#    r"normalsize|tiny|small|footnotesize|large|Large|LARGE|huge|Huge|"
#    r"mathop|mathrm|mathbf|mathit)\b"
#)
#_LATEX_STRAY_CMD_RE = re.compile(r"\\[a-zA-Z]+")
#_LATEX_STRAY_BRACE_RE = re.compile(r"[{}]")
#
#
#def _strip_latex_markup(text: str) -> str:
#    """
#    General-purpose LaTeX/math markup peel. Order matters:
#      1. Unwrap <math>...</math> tags, keeping inner content.
#      2. Repeatedly peel \\command{content} and _{content}/^{content}
#         groups from the INSIDE OUT (handles nesting like
#         \\mathop{\\hbox{\\rm SMFG}}, resolving one layer per loop
#         iteration until stable).
#      3. Turn \\bullet into a real "•" (before step 4's generic sweep
#         would otherwise just delete it).
#      4. Drop bare style commands with no braces left (\\rm,
#         \\scriptsize, \\mathbf, etc.).
#      5. Sweep up anything left over: stray \\commands with no braces,
#         and any orphaned { or }.
#      6. Collapse extra whitespace this stripping tends to leave.
#    """
#    text = _MATH_BLOCK_RE.sub(r"\1", text)
#    prev = None
#    while prev != text:
#        prev = text
#        text = _LATEX_BRACE_CMD_RE.sub(r"\1", text)
#        text = _LATEX_SCRIPT_RE.sub(r"\1", text)
#    text = _BARE_BULLET_CMD_RE.sub("•", text)
#    text = _LATEX_NOARG_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_BRACE_RE.sub("", text)
#    text = re.sub(r"[ \t]{2,}", " ", text)
#    return text
#
#
#def _clean_surya_text(text: str) -> str:
#    """
#    surya-ocr 0.14.7's recognition model is format-aware, not plain OCR:
#    it wraps text it judged bold in <b>...</b>, tags bullet-point dots
#    as <math display="inline">\\bullet</math>, and -- confirmed on a
#    real scanned job description -- also wraps stacked/logo-style text
#    in full LaTeX subscript markup. See _strip_latex_markup() above.
#
#    Separately, the same model sometimes misreads the bullet glyph
#    entirely as some other small-dot-shaped character -- an Arabic-Indic
#    digit zero (٠, U+0660), Extended Arabic-Indic digit zero (۰,
#    U+06F0), or even a plain period (.). These show up as a lone
#    character alone on its own line where a bullet should be.
#
#    This normalizes all of the above down to a plain "•" and strips
#    other inline formatting tags to their plain text content, so
#    section_splitter.py sees ordinary text instead of markup it was
#    never built to parse.
#    """
#    text = _strip_latex_markup(text)
#    text = _FORMAT_TAG_RE.sub("", text)
#    cleaned_lines = [
#        "•" if _LONE_BULLET_GLYPH_LINE_RE.match(line.strip()) else line
#        for line in text.split("\n")
#    ]
#    return "\n".join(cleaned_lines)
#
## ── Surya availability check + predictor cache (populated lazily, once) ──
#_surya_predictors = None   # (det_predictor, rec_predictor) once loaded
#_surya_import_failed = False
#_surya_warned = False
#_bbox_warned = False
#
## Surya renders at 200 DPI (see pdf_to_images below). Used to convert a
## TextLine's bbox (pixel coordinates) into PDF points, same coordinate
## space pdfplumber's extract_words() and the tesseract-fallback path
## both use.
#OCR_DPI = 200
#_PDF_POINTS_PER_INCH = 72
#_PIXEL_TO_POINT = _PDF_POINTS_PER_INCH / OCR_DPI
#
## Set to True (or set env var DEBUG_SURYA=1) to print every detected
## line's real (top, x0, confidence, text) BEFORE reconstruction runs --
## turn this on and re-run against a resume showing a heading-ordering
## problem (like "Career Objective:" landing after its own first
## content line) to see Surya's ACTUAL coordinates for that page,
## instead of guessing. Paste that debug output back for diagnosis.
#import os
#DEBUG_SURYA_COORDS = os.environ.get("DEBUG_SURYA", "0") == "1"
#
#
#def _warn_once(message: str):
#    global _surya_warned
#    if not _surya_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _surya_warned = True
#
#
#def _warn_bbox_once(message: str):
#    global _bbox_warned
#    if not _bbox_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _bbox_warned = True
#
#
#def _load_surya_predictors():
#    """Loads and caches Surya's detection + recognition predictors.
#    Raises on any failure -- callers must catch and fall back to
#    pytesseract."""
#    global _surya_predictors
#    if _surya_predictors is not None:
#        return _surya_predictors
#
#    # surya-ocr==0.14.7 API: predictor classes, not standalone
#    # load_model()/load_processor() functions -- see module docstring.
#    from surya.detection import DetectionPredictor
#    from surya.recognition import RecognitionPredictor
#
#    print("[INFO] Loading Surya OCR predictors (first call only; cached after)...")
#    det_predictor = DetectionPredictor()
#    rec_predictor = RecognitionPredictor()
#
#    _surya_predictors = (det_predictor, rec_predictor)
#    return _surya_predictors
#
#
#def _text_line_to_line_dict(line) -> dict:
#    """
#    Converts one Surya TextLine into the dict shape sort_lines_only()
#    expects: {"text", "x0", "top"}. Raises AttributeError if `.bbox`
#    isn't present in the shape expected -- caller catches this per
#    page (see PRODUCTION SAFETY note below).
#    """
#    x0, top, x1, bottom = line.bbox
#    return {
#        "text": line.text,
#        "x0": x0 * _PIXEL_TO_POINT,
#        "top": top * _PIXEL_TO_POINT,
#    }
#
#
#def _run_surya_on_images(images: list) -> list:
#    """Raises on any failure -- callers must catch and fall back."""
#    det_predictor, rec_predictor = _load_surya_predictors()
#
#    # 0.14.7 dropped the explicit `langs` parameter entirely -- passing
#    # it raises TypeError. Detection is supplied via det_predictor, and
#    # recognition runs language-agnostic OCR by default.
#    results = rec_predictor(images, det_predictor=det_predictor)
#
#    all_text = []
#    for page_idx, page_result in enumerate(results):
#        confident_lines = [line for line in page_result.text_lines if line.confidence > 0.3]
#
#        if DEBUG_SURYA_COORDS:
#            print(f"[DEBUG] Surya line coordinates for page {page_idx + 1}:")
#            for line in confident_lines:
#                x0, top, x1, bottom = line.bbox
#                print(f"  top={top * _PIXEL_TO_POINT:7.1f}  x0={x0 * _PIXEL_TO_POINT:7.1f}  "
#                      f"conf={line.confidence:.2f}  text={line.text!r}")
#
#        # ── Image-based table detection (see REVISION note in module
#        # docstring) -- runs on the page IMAGE directly, independent of
#        # Surya's line boxes entirely, so it carries none of the
#        # line-fusion risk the old _reconstruct_table()-on-Surya-lines
#        # approach had. Returns [] for the large majority of pages that
#        # have no genuine bordered table, in which case everything
#        # below behaves exactly as before this revision.
#        tables = detect_and_ocr_tables(images[page_idx])
#
#        try:
#            line_dicts = [_text_line_to_line_dict(line) for line in confident_lines]
#            if tables:
#                line_dicts, placeholder_map = merge_tables_into_words(
#                    line_dicts, tables, pixel_to_point=_PIXEL_TO_POINT
#                )
#            else:
#                placeholder_map = {}
#            page_text = sort_lines_only(line_dicts)
#            if not page_text.strip():
#                raise ValueError("reconstruction produced empty text")
#        except Exception as e:
#            _warn_bbox_once(
#                f"Surya line-position reconstruction failed ({type(e).__name__}: {e}). "
#                f"Falling back to plain engine-order line join for any page "
#                f"where this happens -- other pages/files are unaffected. "
#                f"This means Surya's TextLine shape in your installed version "
#                f"may not expose .bbox as expected (see ocr_reader.py's "
#                f"module docstring)."
#            )
#            page_text = "\n".join(line.text for line in confident_lines)
#            placeholder_map = {}
#
#        page_text = _clean_surya_text(page_text)
#        # Splice real table text in AFTER _clean_surya_text() runs, so
#        # the LaTeX/formatting cleanup (aimed at Surya's own markup
#        # quirks) never has a chance to touch text that came from our
#        # own separate cell-by-cell Tesseract OCR -- that text was never
#        # wrapped in Surya's markup in the first place.
#        page_text = splice_placeholders(page_text, placeholder_map)
#        all_text.append(page_text)
#    return all_text
#
#
#def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
#    """Converts pages of a PDF into PIL Images at 200 DPI (good balance
#    of Surya accuracy vs. speed). page_indices=None converts every page.
#
#    Each rendered page goes through preprocess_scanned_image() (blur
#    gate + deskew + brightness correction) before being handed back to
#    the caller. A page that fails the blur gate raises
#    ImageQualityError here -- read_with_surya_pages() below is
#    responsible for NOT swallowing that into the generic
#    tesseract-fallback path."""
#    doc = fitz.open(pdf_path)
#    images = []
#    indices = page_indices if page_indices is not None else range(len(doc))
#    for page_num in indices:
#        page = doc[page_num]
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        img = preprocess_scanned_image(img, context=f"page {page_num + 1} of {pdf_path}")
#        images.append(img)
#    doc.close()
#    return images
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    OCRs the given 0-based page indices, Surya first, tesseract fallback
#    on any failure. Returns {page_index: text}. This is the function
#    pdf_reader.py calls for pages classified "scanned".
#
#    ImageQualityError (raised by pdf_to_images()'s preprocessing step)
#    is deliberately re-raised BEFORE the generic except-block below, so
#    a blur rejection is never mistaken for "Surya failed, try tesseract
#    instead" -- tesseract cannot recover detail a blur destroyed any
#    better than Surya can.
#    """
#    if not page_indices:
#        return {}
#
#    global _surya_import_failed
#    if not _surya_import_failed:
#        try:
#            images = pdf_to_images(pdf_path, page_indices=page_indices)
#            texts = _run_surya_on_images(images)
#            return dict(zip(page_indices, texts))
#        except ImageQualityError:
#            raise
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract for this and all subsequent pages. "
#                f"Check `pip show surya-ocr` is exactly 0.6.13, and that model "
#                f"downloads aren't being blocked by network/firewall."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages as tesseract_fallback
#    return tesseract_fallback(pdf_path, page_indices)
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Whole-file OCR for a standalone image or a fully-scanned PDF,
#    joined into one string. Surya first, tesseract fallback on failure.
#
#    Standalone image uploads (.jpg/.png/etc.) also go through
#    preprocess_scanned_image() before OCR, same as PDF pages. See
#    read_with_surya_pages() docstring for why ImageQualityError is
#    re-raised before the generic fallback except-block.
#    """
#    global _surya_import_failed
#    path = Path(file_path)
#
#    if not _surya_import_failed:
#        try:
#            if path.suffix.lower() == ".pdf":
#                images = pdf_to_images(file_path)
#            elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#                img = Image.open(file_path).convert("RGB")
#                img = preprocess_scanned_image(img, context=file_path)
#                images = [img]
#            else:
#                raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#            all_text = _run_surya_on_images(images)
#            return "\n\n".join(all_text)
#        except ImageQualityError:
#            raise
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya as tesseract_fallback
#    return tesseract_fallback(file_path)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_with_surya(sys.argv[1]))
#
#









#"""
#ocr_reader.py — Layer 0, full-page OCR (scanned pages)
#
#Surya is the PRIMARY OCR engine. pytesseract is an automatic FALLBACK,
#used only if Surya can't be imported, can't load its models, or throws
#during inference. You don't need to choose one manually -- this module
#tries Surya first every time and only drops to tesseract if that fails.
#
#TARGETS: surya-ocr==0.14.7 (confirmed via `pip show surya-ocr` against
#the actual installed environment -- see version history below for why
#this matters more than it sounds like it should).
#
#── Three different Surya APIs found across versions, confirmed by
#   downloading and inspecting the actual PyPI packages, not guessing ──
#
#1. surya-ocr==0.6.13 -- what the ORIGINAL ocr_reader.py was written
#   against: standalone functions, `surya.model.detection.model.load_model`,
#   `surya.model.recognition.model.load_model`, `surya.ocr.run_ocr(images,
#   langs, det_model, det_processor, rec_model, rec_processor)`. Even in
#   this exact version the original code had a wrong import path
#   (`surya.model.detection.PROCESSOR.load_processor` -- that function
#   actually lives in `surya.model.detection.MODEL`, confirmed against
#   the real file contents).
#
#2. surya-ocr==0.14.7 -- what's ACTUALLY INSTALLED in your venv (confirmed
#   via your `pip show surya-ocr` output). No `surya.model` package at
#   all anymore -- detection and recognition are now predictor CLASSES:
#   `surya.detection.DetectionPredictor`, `surya.recognition.RecognitionPredictor`.
#   No `langs` parameter either (dropped in this generation). Call shape:
#
#       det_predictor = DetectionPredictor()
#       rec_predictor = RecognitionPredictor()
#       results = rec_predictor(images, det_predictor=det_predictor)
#
#   This is the version this file is now written against.
#
#3. surya-ocr==0.22.x (current PyPI default if you `pip install surya-ocr`
#   with no version pin) -- a further VLM-based rewrite requiring a
#   running model server. Different again from both of the above.
#
#The lesson: don't assume a Surya upgrade/reinstall keeps the same call
#shape. If you ever bump this dependency, re-check `pip show surya-ocr`
#and re-verify the import shape before assuming this file still works --
#it's changed shape at least 3 times.
#
#── Fallback behaviour ──
#If Surya's imports fail, its models fail to load, or the predictor call
#throws for any reason, this module logs a warning once and falls back
#to ocr_reader_pytesseract.py for that call. The fallback is all-or-
#nothing per batch (not per-page) -- if Surya breaks partway through a
#multi-page document, the whole document's scanned pages go to
#tesseract rather than trying to reconcile a half-Surya, half-tesseract
#result.
#
#── Model caching ──
#Your original _load_surya_models() reloaded all 4 models on every call
#to read_with_surya_pages(). Across a batch of 60+ resumes that's 60+
#model loads. Predictors are now cached at module level after first load.
#
#── REVISION — broadened LaTeX/math markup cleanup ──
#Confirmed on a real scanned job description (SMFG_Job_Document_Scanned
#.pdf): the previous _clean_surya_text() only stripped the specific
#<math>\bullet</math> pattern plus a handful of inline formatting tags
#(b/i/u/del/mark/sup/sub/br). That's too narrow. Surya's format-aware
#recognition model also wraps STACKED/logo-style text -- e.g. a company
#logo rendered as "SMFG" on one line with "IndiaCredit" in smaller type
#underneath it -- in full LaTeX subscript/font markup that looks nothing
#like the bullet case:
#
#    \mathop{\hbox{\rm SMFG}}_{\hbox{\scriptsize \rm Indiacredit}}
#
#_strip_latex_markup() below is a general-purpose peel that handles this
#and the old bullet case with one shared implementation. This part WAS
#verified directly (tested against the exact leaked string above and
#the pre-existing bullet-math case).
#
#── REVISION — layout-aware page reconstruction (SUPERSEDED, see below) ──
#An earlier version of this file tried reusing reconstruct_page_text()
#(the word-level sidebar/two-column-aware function) for Surya's output,
#by treating each Surya TextLine as one "word". That was flagged at the
#time as unverified against a real Surya model. It has now BEEN tested
#against real Surya output, and it actively broke things -- see the
#next REVISION note below. Do not reintroduce reconstruct_page_text()
#here without re-reading that note.
#
#── REVISION (this version) — switched to sort_lines_only(), NOT
#   reconstruct_page_text() ──
#Confirmed on a real scanned resume (Anandu Chandran,
#PDF_image2pdf_20260821121805.pdf): reconstruct_page_text() badly
#scrambled a completely ordinary SINGLE-COLUMN resume. Two different,
#unrelated lines (e.g. two lines of a center-aligned address block) got
#merged onto one output line and reordered left-to-right, because
#Surya provides LINE-granularity bounding boxes (one box per whole
#line), not word-granularity -- and reconstruct_page_text()'s merging
#and sidebar/two-column logic is calibrated for word-level data, where
#merging nearby-top fragments into one line is the whole point. Applied
#to already-whole-line data, that same merging step can wrongly fuse
#two DIFFERENT lines together whenever their measured "top" values
#happen to land close to each other, which is easy to hit with center-
#aligned text (every line has a different x0) or a short heading next
#to a wrapped bullet line. Reproduced exactly against this real
#resume's garbled output using plausible coordinates for the two lines
#that got fused -- see ocr_layout_reconstruction.py's module docstring
#for the full trace.
#
#Fixed by switching to ocr_layout_reconstruction.sort_lines_only():
#sorts Surya's lines by (top, x0) and emits one output line per input
#line, with NO merging and NO sidebar/two-column detection at all.
#Trade-off, accepted deliberately: a short heading can occasionally
#land one line off from its ideal position if Surya's box for it is
#very slightly mismeasured (cosmetic). In exchange, two different
#lines' text can never be fused together again, which was the actually
#severe failure mode breaking ordinary single-column resumes in
#production. This gives up any "smart" sidebar/table reordering for a
#scanned sidebar-style document processed via Surya specifically (that
#capability was never verified working in the first place) in favor of
#guaranteed correctness on the much more common single-column case.
#
#reconstruct_page_text() itself is UNCHANGED and still correct --
#ocr_reader_pytesseract.py's tesseract fallback path still uses it,
#because tesseract's image_to_data() provides genuine word-level
#fragments, which is what that function's thresholds and tolerances
#are calibrated for.
#
#── REVISION — pre-OCR image quality gate ──
#Added a preprocessing step (ingestion/image_preprocessing.py) that runs
#on every rendered/loaded image BEFORE it reaches Surya:
#
#  1. Blur rejection -- raises ImageQualityError for pages too blurry to
#     trust. Calibrated via a controlled blur ladder on a real resume:
#     sentences stay grammatically intact but silently wrong fields
#     (emails, dates, proper nouns) start appearing well before the
#     image looks like obvious garbage -- so this is a hard reject, not
#     a warning. The blur score is resolution-normalized (see that
#     module's _BLUR_REFERENCE_WIDTH) so the same threshold is valid
#     whether an image was rendered at this file's 200 DPI or the
#     tesseract fallback's 300 DPI.
#  2. Deskew -- corrects page rotation. Justified by a real finding on
#     this project: a 4.5-degree rotation caused a section heading to be
#     sorted into the middle of an unrelated bullet sentence, and
#     dropped a line entirely, because reading-order reconstruction
#     sorts by y-coordinate and a rotated page's "same visual line" has
#     different y-coordinates on the left vs. right edge of the page.
#  3. Brightness correction -- inverts pages that are moderately dark
#     overall (a real underexposed photo).
#
#ImageQualityError must propagate all the way up to the caller as a
#genuine rejection -- it is explicitly re-raised BEFORE the generic
#Surya-failure except-block below, so a blurry page is rejected outright
#rather than silently retried with tesseract (tesseract cannot recover
#detail a blur destroyed any better than Surya can).
#"""
#
#import logging
#import re
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF
#
#from ingestion.ocr_layout_reconstruction import sort_lines_only
#from ingestion.image_preprocessing import preprocess_scanned_image, ImageQualityError
#
#logger = logging.getLogger(__name__)
#
#_MATH_BULLET_RE = re.compile(r"<math[^>]*>\s*\\bullet\s*</math>", re.IGNORECASE)
#_MATH_TAG_RE = re.compile(r"</?math[^>]*>", re.IGNORECASE)
#_BARE_BULLET_CMD_RE = re.compile(r"\\bullet")
#_FORMAT_TAG_RE = re.compile(r"</?(b|i|u|del|mark|sup|sub|br)\b[^>]*>", re.IGNORECASE)
#_LONE_BULLET_GLYPH_LINE_RE = re.compile(r"^[\u0660\u06F0.]$")
#
## ── LaTeX/math markup cleanup (see REVISION note above) ──
#_MATH_BLOCK_RE = re.compile(r"<math[^>]*>(.*?)</math>", re.IGNORECASE | re.DOTALL)
#_LATEX_BRACE_CMD_RE = re.compile(r"\\[a-zA-Z]+\{([^{}]*)\}")
#_LATEX_SCRIPT_RE = re.compile(r"[_^]\{([^{}]*)\}")
#_LATEX_NOARG_CMD_RE = re.compile(
#    r"\\(rm|bf|it|tt|sf|em|scriptsize|scriptstyle|displaystyle|textstyle|"
#    r"normalsize|tiny|small|footnotesize|large|Large|LARGE|huge|Huge|"
#    r"mathop|mathrm|mathbf|mathit)\b"
#)
#_LATEX_STRAY_CMD_RE = re.compile(r"\\[a-zA-Z]+")
#_LATEX_STRAY_BRACE_RE = re.compile(r"[{}]")
#
#
#def _strip_latex_markup(text: str) -> str:
#    """
#    General-purpose LaTeX/math markup peel. Order matters:
#      1. Unwrap <math>...</math> tags, keeping inner content.
#      2. Repeatedly peel \\command{content} and _{content}/^{content}
#         groups from the INSIDE OUT (handles nesting like
#         \\mathop{\\hbox{\\rm SMFG}}, resolving one layer per loop
#         iteration until stable).
#      3. Turn \\bullet into a real "•" (before step 4's generic sweep
#         would otherwise just delete it).
#      4. Drop bare style commands with no braces left (\\rm,
#         \\scriptsize, \\mathbf, etc.).
#      5. Sweep up anything left over: stray \\commands with no braces,
#         and any orphaned { or }.
#      6. Collapse extra whitespace this stripping tends to leave.
#    """
#    text = _MATH_BLOCK_RE.sub(r"\1", text)
#    prev = None
#    while prev != text:
#        prev = text
#        text = _LATEX_BRACE_CMD_RE.sub(r"\1", text)
#        text = _LATEX_SCRIPT_RE.sub(r"\1", text)
#    text = _BARE_BULLET_CMD_RE.sub("•", text)
#    text = _LATEX_NOARG_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_BRACE_RE.sub("", text)
#    text = re.sub(r"[ \t]{2,}", " ", text)
#    return text
#
#
#def _clean_surya_text(text: str) -> str:
#    """
#    surya-ocr 0.14.7's recognition model is format-aware, not plain OCR:
#    it wraps text it judged bold in <b>...</b>, tags bullet-point dots
#    as <math display="inline">\\bullet</math>, and -- confirmed on a
#    real scanned job description -- also wraps stacked/logo-style text
#    in full LaTeX subscript markup. See _strip_latex_markup() above.
#
#    Separately, the same model sometimes misreads the bullet glyph
#    entirely as some other small-dot-shaped character -- an Arabic-Indic
#    digit zero (٠, U+0660), Extended Arabic-Indic digit zero (۰,
#    U+06F0), or even a plain period (.). These show up as a lone
#    character alone on its own line where a bullet should be.
#
#    This normalizes all of the above down to a plain "•" and strips
#    other inline formatting tags to their plain text content, so
#    section_splitter.py sees ordinary text instead of markup it was
#    never built to parse.
#    """
#    text = _strip_latex_markup(text)
#    text = _FORMAT_TAG_RE.sub("", text)
#    cleaned_lines = [
#        "•" if _LONE_BULLET_GLYPH_LINE_RE.match(line.strip()) else line
#        for line in text.split("\n")
#    ]
#    return "\n".join(cleaned_lines)
#
## ── Surya availability check + predictor cache (populated lazily, once) ──
#_surya_predictors = None   # (det_predictor, rec_predictor) once loaded
#_surya_import_failed = False
#_surya_warned = False
#_bbox_warned = False
#
## Surya renders at 200 DPI (see pdf_to_images below). Used to convert a
## TextLine's bbox (pixel coordinates) into PDF points, same coordinate
## space pdfplumber's extract_words() and the tesseract-fallback path
## both use.
#OCR_DPI = 200
#_PDF_POINTS_PER_INCH = 72
#_PIXEL_TO_POINT = _PDF_POINTS_PER_INCH / OCR_DPI
#
#
#def _warn_once(message: str):
#    global _surya_warned
#    if not _surya_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _surya_warned = True
#
#
#def _warn_bbox_once(message: str):
#    global _bbox_warned
#    if not _bbox_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _bbox_warned = True
#
#
#def _load_surya_predictors():
#    """Loads and caches Surya's detection + recognition predictors.
#    Raises on any failure -- callers must catch and fall back to
#    pytesseract."""
#    global _surya_predictors
#    if _surya_predictors is not None:
#        return _surya_predictors
#
#    # surya-ocr==0.14.7 API: predictor classes, not standalone
#    # load_model()/load_processor() functions -- see module docstring.
#    from surya.detection import DetectionPredictor
#    from surya.recognition import RecognitionPredictor
#
#    print("[INFO] Loading Surya OCR predictors (first call only; cached after)...")
#    det_predictor = DetectionPredictor()
#    rec_predictor = RecognitionPredictor()
#
#    _surya_predictors = (det_predictor, rec_predictor)
#    return _surya_predictors
#
#
#def _text_line_to_line_dict(line) -> dict:
#    """
#    Converts one Surya TextLine into the dict shape sort_lines_only()
#    expects: {"text", "x0", "top"}. Raises AttributeError if `.bbox`
#    isn't present in the shape expected -- caller catches this per
#    page (see PRODUCTION SAFETY note below).
#    """
#    x0, top, x1, bottom = line.bbox
#    return {
#        "text": line.text,
#        "x0": x0 * _PIXEL_TO_POINT,
#        "top": top * _PIXEL_TO_POINT,
#    }
#
#
#def _run_surya_on_images(images: list) -> list:
#    """Raises on any failure -- callers must catch and fall back."""
#    det_predictor, rec_predictor = _load_surya_predictors()
#
#    # 0.14.7 dropped the explicit `langs` parameter entirely -- passing
#    # it raises TypeError. Detection is supplied via det_predictor, and
#    # recognition runs language-agnostic OCR by default.
#    results = rec_predictor(images, det_predictor=det_predictor)
#
#    all_text = []
#    for page_result in results:
#        confident_lines = [line for line in page_result.text_lines if line.confidence > 0.3]
#
#        # Sort by position only -- see module docstring's REVISION note
#        # on why reconstruct_page_text()'s word-level merging/column
#        # detection must NOT be used on Surya's line-granularity data.
#        # PRODUCTION SAFETY: still wrapped in try/except per PAGE (not
#        # per batch) in case a given Surya version's TextLine shape
#        # doesn't expose .bbox as expected -- falls back to the plain
#        # engine-order join for just that page if so.
#        try:
#            line_dicts = [_text_line_to_line_dict(line) for line in confident_lines]
#            page_text = sort_lines_only(line_dicts)
#            if not page_text.strip():
#                raise ValueError("line sort produced empty text")
#        except Exception as e:
#            _warn_bbox_once(
#                f"Surya line-position sort failed ({type(e).__name__}: {e}). "
#                f"Falling back to plain engine-order line join for any page "
#                f"where this happens -- other pages/files are unaffected. "
#                f"This means Surya's TextLine shape in your installed version "
#                f"may not expose .bbox as expected (see ocr_reader.py's "
#                f"module docstring)."
#            )
#            page_text = "\n".join(line.text for line in confident_lines)
#
#        page_text = _clean_surya_text(page_text)
#        all_text.append(page_text)
#    return all_text
#
#
#def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
#    """Converts pages of a PDF into PIL Images at 200 DPI (good balance
#    of Surya accuracy vs. speed). page_indices=None converts every page.
#
#    Each rendered page goes through preprocess_scanned_image() (blur
#    gate + deskew + brightness correction) before being handed back to
#    the caller. A page that fails the blur gate raises
#    ImageQualityError here -- read_with_surya_pages() below is
#    responsible for NOT swallowing that into the generic
#    tesseract-fallback path."""
#    doc = fitz.open(pdf_path)
#    images = []
#    indices = page_indices if page_indices is not None else range(len(doc))
#    for page_num in indices:
#        page = doc[page_num]
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        img = preprocess_scanned_image(img, context=f"page {page_num + 1} of {pdf_path}")
#        images.append(img)
#    doc.close()
#    return images
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    OCRs the given 0-based page indices, Surya first, tesseract fallback
#    on any failure. Returns {page_index: text}. This is the function
#    pdf_reader.py calls for pages classified "scanned".
#
#    ImageQualityError (raised by pdf_to_images()'s preprocessing step)
#    is deliberately re-raised BEFORE the generic except-block below, so
#    a blur rejection is never mistaken for "Surya failed, try tesseract
#    instead" -- tesseract cannot recover detail a blur destroyed any
#    better than Surya can.
#    """
#    if not page_indices:
#        return {}
#
#    global _surya_import_failed
#    if not _surya_import_failed:
#        try:
#            images = pdf_to_images(pdf_path, page_indices=page_indices)
#            texts = _run_surya_on_images(images)
#            return dict(zip(page_indices, texts))
#        except ImageQualityError:
#            raise
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract for this and all subsequent pages. "
#                f"Check `pip show surya-ocr` is exactly 0.6.13, and that model "
#                f"downloads aren't being blocked by network/firewall."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages as tesseract_fallback
#    return tesseract_fallback(pdf_path, page_indices)
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Whole-file OCR for a standalone image or a fully-scanned PDF,
#    joined into one string. Surya first, tesseract fallback on failure.
#
#    Standalone image uploads (.jpg/.png/etc.) also go through
#    preprocess_scanned_image() before OCR, same as PDF pages. See
#    read_with_surya_pages() docstring for why ImageQualityError is
#    re-raised before the generic fallback except-block.
#    """
#    global _surya_import_failed
#    path = Path(file_path)
#
#    if not _surya_import_failed:
#        try:
#            if path.suffix.lower() == ".pdf":
#                images = pdf_to_images(file_path)
#            elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#                img = Image.open(file_path).convert("RGB")
#                img = preprocess_scanned_image(img, context=file_path)
#                images = [img]
#            else:
#                raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#            all_text = _run_surya_on_images(images)
#            return "\n\n".join(all_text)
#        except ImageQualityError:
#            raise
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya as tesseract_fallback
#    return tesseract_fallback(file_path)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_with_surya(sys.argv[1]))
#










#"""
#ocr_reader.py — Layer 0, full-page OCR (scanned pages)
#
#Surya is the PRIMARY OCR engine. pytesseract is an automatic FALLBACK,
#used only if Surya can't be imported, can't load its models, or throws
#during inference. You don't need to choose one manually -- this module
#tries Surya first every time and only drops to tesseract if that fails.
#
#TARGETS: surya-ocr==0.14.7 (confirmed via `pip show surya-ocr` against
#the actual installed environment -- see version history below for why
#this matters more than it sounds like it should).
#
#── Three different Surya APIs found across versions, confirmed by
#   downloading and inspecting the actual PyPI packages, not guessing ──
#
#1. surya-ocr==0.6.13 -- what the ORIGINAL ocr_reader.py was written
#   against: standalone functions, `surya.model.detection.model.load_model`,
#   `surya.model.recognition.model.load_model`, `surya.ocr.run_ocr(images,
#   langs, det_model, det_processor, rec_model, rec_processor)`. Even in
#   this exact version the original code had a wrong import path
#   (`surya.model.detection.PROCESSOR.load_processor` -- that function
#   actually lives in `surya.model.detection.MODEL`, confirmed against
#   the real file contents).
#
#2. surya-ocr==0.14.7 -- what's ACTUALLY INSTALLED in your venv (confirmed
#   via your `pip show surya-ocr` output). No `surya.model` package at
#   all anymore -- detection and recognition are now predictor CLASSES:
#   `surya.detection.DetectionPredictor`, `surya.recognition.RecognitionPredictor`.
#   No `langs` parameter either (dropped in this generation). Call shape:
#
#       det_predictor = DetectionPredictor()
#       rec_predictor = RecognitionPredictor()
#       results = rec_predictor(images, det_predictor=det_predictor)
#
#   This is the version this file is now written against.
#
#3. surya-ocr==0.22.x (current PyPI default if you `pip install surya-ocr`
#   with no version pin) -- a further VLM-based rewrite requiring a
#   running model server. Different again from both of the above.
#
#The lesson: don't assume a Surya upgrade/reinstall keeps the same call
#shape. If you ever bump this dependency, re-check `pip show surya-ocr`
#and re-verify the import shape before assuming this file still works --
#it's changed shape at least 3 times.
#
#── Fallback behaviour ──
#If Surya's imports fail, its models fail to load, or the predictor call
#throws for any reason, this module logs a warning once and falls back
#to ocr_reader_pytesseract.py for that call. The fallback is all-or-
#nothing per batch (not per-page) -- if Surya breaks partway through a
#multi-page document, the whole document's scanned pages go to
#tesseract rather than trying to reconcile a half-Surya, half-tesseract
#result.
#
#── Model caching ──
#Your original _load_surya_models() reloaded all 4 models on every call
#to read_with_surya_pages(). Across a batch of 60+ resumes that's 60+
#model loads. Predictors are now cached at module level after first load.
#
#── REVISION — broadened LaTeX/math markup cleanup ──
#Confirmed on a real scanned job description (SMFG_Job_Document_Scanned
#.pdf): the previous _clean_surya_text() only stripped the specific
#<math>\bullet</math> pattern plus a handful of inline formatting tags
#(b/i/u/del/mark/sup/sub/br). That's too narrow. Surya's format-aware
#recognition model also wraps STACKED/logo-style text -- e.g. a company
#logo rendered as "SMFG" on one line with "IndiaCredit" in smaller type
#underneath it -- in full LaTeX subscript/font markup that looks nothing
#like the bullet case:
#
#    \mathop{\hbox{\rm SMFG}}_{\hbox{\scriptsize \rm Indiacredit}}
#
#_strip_latex_markup() below is a general-purpose peel that handles this
#and the old bullet case with one shared implementation. This part WAS
#verified directly (tested against the exact leaked string above and
#the pre-existing bullet-math case).
#
#── REVISION — layout-aware page reconstruction ──
#Confirmed on the same SMFG document: a 2-column label/value table
#("Job Title" | "Asst. Manager, Cyber Defence") came out with rows
#scrambled, because this module previously just joined Surya's
#`text_lines` top-to-bottom in whatever order Surya itself returned them
#in, discarding position information entirely. pdf_reader.py already has
#layout-reconstruction logic (a separate, OCR-only copy -- see
#ocr_layout_reconstruction.py) that fixes exactly this kind of problem
#for native PDF text; this revision reuses it here too, by reading each
#TextLine's `.bbox` attribute (documented as available on Surya's
#recognition output) and building word dicts from it, then calling
#ocr_layout_reconstruction.reconstruct_page_text() instead of a flat join.
#
#IMPORTANT — THIS PART IS NOT VERIFIED AGAINST A REAL SURYA MODEL.
#The equivalent fix WAS verified end-to-end against the tesseract
#fallback path (ocr_reader_pytesseract.py) using this document's real
#OCR'd coordinates, and Surya's TextLine.bbox attribute is documented in
#surya-ocr's own output schema, but "documented" isn't "confirmed
#against your installed 0.14.7 in practice."
#
#To keep this safe for production: bbox extraction is wrapped in a
#try/except AttributeError per PAGE, not per batch. If bbox isn't
#available or reconstruction fails for any reason, that single page
#falls back to the OLD plain top-to-bottom line join -- it does not
#raise, and does not affect any other page or file. A warning is logged
#once (not once per page) the first time this happens.
#
#ACTION NEEDED FROM YOU: run this against real scanned resumes/job docs
#and check your terminal for "[WARNING] Surya bbox reconstruction
#failed...". If you see it, report back the exact error message so the
#bbox-extraction shape can be adjusted to match your installed version.
#
#── REVISION (this version) — pre-OCR image quality gate ──
#Added a preprocessing step (ingestion/image_preprocessing.py) that runs
#on every rendered/loaded image BEFORE it reaches Surya:
#
#  1. Blur rejection -- raises ImageQualityError for pages too blurry to
#     trust. Calibrated via a controlled blur ladder on a real resume:
#     sentences stay grammatically intact but silently wrong fields
#     (emails, dates, proper nouns) start appearing well before the
#     image looks like obvious garbage -- so this is a hard reject, not
#     a warning. The blur score is resolution-normalized (see that
#     module's _BLUR_REFERENCE_WIDTH) so the same threshold is valid
#     whether an image was rendered at this file's 200 DPI or the
#     tesseract fallback's 300 DPI.
#  2. Deskew -- corrects page rotation. Justified by a real finding on
#     this project: a 4.5-degree rotation caused a section heading to be
#     sorted into the middle of an unrelated bullet sentence, and
#     dropped a line entirely, because reading-order reconstruction
#     sorts by y-coordinate and a rotated page's "same visual line" has
#     different y-coordinates on the left vs. right edge of the page.
#  3. Brightness correction -- inverts pages that are moderately dark
#     overall (a real underexposed photo).
#
#ImageQualityError must propagate all the way up to the caller as a
#genuine rejection -- it is explicitly re-raised BEFORE the generic
#Surya-failure except-block below, so a blurry page is rejected outright
#rather than silently retried with tesseract (tesseract cannot recover
#detail a blur destroyed any better than Surya can).
#"""
#
#import logging
#import re
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF
#
#from ingestion.ocr_layout_reconstruction import reconstruct_page_text
#from ingestion.image_preprocessing import preprocess_scanned_image, ImageQualityError
#
#logger = logging.getLogger(__name__)
#
#_MATH_BULLET_RE = re.compile(r"<math[^>]*>\s*\\bullet\s*</math>", re.IGNORECASE)
#_MATH_TAG_RE = re.compile(r"</?math[^>]*>", re.IGNORECASE)
#_BARE_BULLET_CMD_RE = re.compile(r"\\bullet")
#_FORMAT_TAG_RE = re.compile(r"</?(b|i|u|del|mark|sup|sub|br)\b[^>]*>", re.IGNORECASE)
#_LONE_BULLET_GLYPH_LINE_RE = re.compile(r"^[\u0660\u06F0.]$")
#
## ── LaTeX/math markup cleanup (see REVISION note above) ──
#_MATH_BLOCK_RE = re.compile(r"<math[^>]*>(.*?)</math>", re.IGNORECASE | re.DOTALL)
#_LATEX_BRACE_CMD_RE = re.compile(r"\\[a-zA-Z]+\{([^{}]*)\}")
#_LATEX_SCRIPT_RE = re.compile(r"[_^]\{([^{}]*)\}")
#_LATEX_NOARG_CMD_RE = re.compile(
#    r"\\(rm|bf|it|tt|sf|em|scriptsize|scriptstyle|displaystyle|textstyle|"
#    r"normalsize|tiny|small|footnotesize|large|Large|LARGE|huge|Huge|"
#    r"mathop|mathrm|mathbf|mathit)\b"
#)
#_LATEX_STRAY_CMD_RE = re.compile(r"\\[a-zA-Z]+")
#_LATEX_STRAY_BRACE_RE = re.compile(r"[{}]")
#
#
#def _strip_latex_markup(text: str) -> str:
#    """
#    General-purpose LaTeX/math markup peel. Order matters:
#      1. Unwrap <math>...</math> tags, keeping inner content.
#      2. Repeatedly peel \\command{content} and _{content}/^{content}
#         groups from the INSIDE OUT (handles nesting like
#         \\mathop{\\hbox{\\rm SMFG}}, resolving one layer per loop
#         iteration until stable).
#      3. Turn \\bullet into a real "•" (before step 4's generic sweep
#         would otherwise just delete it).
#      4. Drop bare style commands with no braces left (\\rm,
#         \\scriptsize, \\mathbf, etc.).
#      5. Sweep up anything left over: stray \\commands with no braces,
#         and any orphaned { or }.
#      6. Collapse extra whitespace this stripping tends to leave.
#    """
#    text = _MATH_BLOCK_RE.sub(r"\1", text)
#    prev = None
#    while prev != text:
#        prev = text
#        text = _LATEX_BRACE_CMD_RE.sub(r"\1", text)
#        text = _LATEX_SCRIPT_RE.sub(r"\1", text)
#    text = _BARE_BULLET_CMD_RE.sub("•", text)
#    text = _LATEX_NOARG_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_BRACE_RE.sub("", text)
#    text = re.sub(r"[ \t]{2,}", " ", text)
#    return text
#
#
#def _clean_surya_text(text: str) -> str:
#    """
#    surya-ocr 0.14.7's recognition model is format-aware, not plain OCR:
#    it wraps text it judged bold in <b>...</b>, tags bullet-point dots
#    as <math display="inline">\\bullet</math>, and -- confirmed on a
#    real scanned job description -- also wraps stacked/logo-style text
#    in full LaTeX subscript markup. See _strip_latex_markup() above.
#
#    Separately, the same model sometimes misreads the bullet glyph
#    entirely as some other small-dot-shaped character -- an Arabic-Indic
#    digit zero (٠, U+0660), Extended Arabic-Indic digit zero (۰,
#    U+06F0), or even a plain period (.). These show up as a lone
#    character alone on its own line where a bullet should be.
#
#    This normalizes all of the above down to a plain "•" and strips
#    other inline formatting tags to their plain text content, so
#    section_splitter.py sees ordinary text instead of markup it was
#    never built to parse.
#    """
#    text = _strip_latex_markup(text)
#    text = _FORMAT_TAG_RE.sub("", text)
#    cleaned_lines = [
#        "•" if _LONE_BULLET_GLYPH_LINE_RE.match(line.strip()) else line
#        for line in text.split("\n")
#    ]
#    return "\n".join(cleaned_lines)
#
## ── Surya availability check + predictor cache (populated lazily, once) ──
#_surya_predictors = None   # (det_predictor, rec_predictor) once loaded
#_surya_import_failed = False
#_surya_warned = False
#_bbox_warned = False
#
## Surya renders at 200 DPI (see pdf_to_images below). Used to convert a
## TextLine's bbox (assumed to be in rendered-image PIXEL coordinates --
## see PRODUCTION SAFETY note in the module docstring for why this is
## flagged as unverified) into PDF points, same coordinate space
## pdfplumber's extract_words() and the tesseract-fallback path both use.
#OCR_DPI = 200
#_PDF_POINTS_PER_INCH = 72
#_PIXEL_TO_POINT = _PDF_POINTS_PER_INCH / OCR_DPI
#
#
#def _warn_once(message: str):
#    global _surya_warned
#    if not _surya_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _surya_warned = True
#
#
#def _warn_bbox_once(message: str):
#    global _bbox_warned
#    if not _bbox_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _bbox_warned = True
#
#
#def _load_surya_predictors():
#    """Loads and caches Surya's detection + recognition predictors.
#    Raises on any failure -- callers must catch and fall back to
#    pytesseract."""
#    global _surya_predictors
#    if _surya_predictors is not None:
#        return _surya_predictors
#
#    # surya-ocr==0.14.7 API: predictor classes, not standalone
#    # load_model()/load_processor() functions -- see module docstring.
#    from surya.detection import DetectionPredictor
#    from surya.recognition import RecognitionPredictor
#
#    print("[INFO] Loading Surya OCR predictors (first call only; cached after)...")
#    det_predictor = DetectionPredictor()
#    rec_predictor = RecognitionPredictor()
#
#    _surya_predictors = (det_predictor, rec_predictor)
#    return _surya_predictors
#
#
#def _text_line_to_word_dict(line) -> dict:
#    """
#    Converts one Surya TextLine into the word-dict shape
#    ocr_layout_reconstruction.py expects. Raises AttributeError if `.bbox`
#    isn't present in the shape expected -- caller catches this per page
#    (see PRODUCTION SAFETY note in module docstring).
#
#    NOTE: Surya operates at LINE granularity, not word granularity --
#    each TextLine is typically a whole phrase/line, not a single word.
#    Feeding one dict per LINE (rather than splitting into individual
#    words) into reconstruct_page_text() still works for detecting
#    "does this row belong left-of or right-of that row" (which is what
#    fixed the SMFG table via the tesseract path), but a genuine dense
#    two-column layout that needs word-level x0 statistics to detect
#    reliably (>=30 words per side, see _is_true_two_column in
#    ocr_layout_reconstruction.py) may not have enough "words" at line
#    granularity to trigger that specific detector. Sidebar/table-style
#    reconstruction (line-granularity groups by row) is the case this
#    was actually verified to fix; true multi-paragraph two-column OCR
#    pages are a known gap here, not yet validated either way.
#    """
#    x0, top, x1, bottom = line.bbox
#    return {
#        "text": line.text,
#        "x0": x0 * _PIXEL_TO_POINT,
#        "x1": x1 * _PIXEL_TO_POINT,
#        "top": top * _PIXEL_TO_POINT,
#        "bottom": bottom * _PIXEL_TO_POINT,
#        "height": (bottom - top) * _PIXEL_TO_POINT,
#    }
#
#
#def _run_surya_on_images(images: list) -> list:
#    """Raises on any failure -- callers must catch and fall back."""
#    det_predictor, rec_predictor = _load_surya_predictors()
#
#    # 0.14.7 dropped the explicit `langs` parameter entirely -- passing
#    # it raises TypeError. Detection is supplied via det_predictor, and
#    # recognition runs language-agnostic OCR by default.
#    results = rec_predictor(images, det_predictor=det_predictor)
#
#    all_text = []
#    for page_idx, page_result in enumerate(results):
#        confident_lines = [line for line in page_result.text_lines if line.confidence > 0.3]
#
#        # Try layout-aware reconstruction first (see REVISION note in
#        # module docstring -- unverified against real Surya output).
#        # Falls back to the old plain top-to-bottom join per PAGE, not
#        # per batch, if anything about this goes wrong.
#        try:
#            page_width_px = images[page_idx].width
#            words = [_text_line_to_word_dict(line) for line in confident_lines]
#            page_text = reconstruct_page_text(words, page_width_px * _PIXEL_TO_POINT)
#            if not page_text.strip():
#                raise ValueError("layout reconstruction produced empty text")
#        except Exception as e:
#            _warn_bbox_once(
#                f"Surya bbox reconstruction failed ({type(e).__name__}: {e}). "
#                f"Falling back to plain top-to-bottom line join for any page "
#                f"where this happens -- other pages/files are unaffected. "
#                f"This means Surya's TextLine shape in your installed version "
#                f"may not match what this code expects (see ACTION NEEDED in "
#                f"ocr_reader.py's module docstring)."
#            )
#            page_text = "\n".join(line.text for line in confident_lines)
#
#        page_text = _clean_surya_text(page_text)
#        all_text.append(page_text)
#    return all_text
#
#
#def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
#    """Converts pages of a PDF into PIL Images at 200 DPI (good balance
#    of Surya accuracy vs. speed). page_indices=None converts every page.
#
#    Each rendered page goes through preprocess_scanned_image() (blur
#    gate + deskew + brightness correction) before being handed back to
#    the caller. A page that fails the blur gate raises
#    ImageQualityError here -- read_with_surya_pages() below is
#    responsible for NOT swallowing that into the generic
#    tesseract-fallback path."""
#    doc = fitz.open(pdf_path)
#    images = []
#    indices = page_indices if page_indices is not None else range(len(doc))
#    for page_num in indices:
#        page = doc[page_num]
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        img = preprocess_scanned_image(img, context=f"page {page_num + 1} of {pdf_path}")
#        images.append(img)
#    doc.close()
#    return images
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    OCRs the given 0-based page indices, Surya first, tesseract fallback
#    on any failure. Returns {page_index: text}. This is the function
#    pdf_reader.py calls for pages classified "scanned".
#
#    ImageQualityError (raised by pdf_to_images()'s preprocessing step)
#    is deliberately re-raised BEFORE the generic except-block below, so
#    a blur rejection is never mistaken for "Surya failed, try tesseract
#    instead" -- tesseract cannot recover detail a blur destroyed any
#    better than Surya can.
#    """
#    if not page_indices:
#        return {}
#
#    global _surya_import_failed
#    if not _surya_import_failed:
#        try:
#            images = pdf_to_images(pdf_path, page_indices=page_indices)
#            texts = _run_surya_on_images(images)
#            return dict(zip(page_indices, texts))
#        except ImageQualityError:
#            raise
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract for this and all subsequent pages. "
#                f"Check `pip show surya-ocr` is exactly 0.6.13, and that model "
#                f"downloads aren't being blocked by network/firewall."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages as tesseract_fallback
#    return tesseract_fallback(pdf_path, page_indices)
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Whole-file OCR for a standalone image or a fully-scanned PDF,
#    joined into one string. Surya first, tesseract fallback on failure.
#
#    Standalone image uploads (.jpg/.png/etc.) also go through
#    preprocess_scanned_image() before OCR, same as PDF pages. See
#    read_with_surya_pages() docstring for why ImageQualityError is
#    re-raised before the generic fallback except-block.
#    """
#    global _surya_import_failed
#    path = Path(file_path)
#
#    if not _surya_import_failed:
#        try:
#            if path.suffix.lower() == ".pdf":
#                images = pdf_to_images(file_path)
#            elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#                img = Image.open(file_path).convert("RGB")
#                img = preprocess_scanned_image(img, context=file_path)
#                images = [img]
#            else:
#                raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#            all_text = _run_surya_on_images(images)
#            return "\n\n".join(all_text)
#        except ImageQualityError:
#            raise
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya as tesseract_fallback
#    return tesseract_fallback(file_path)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_with_surya(sys.argv[1]))
#







#commenting just to check the code pasted after coming from home works or not
#"""
#ocr_reader.py — Layer 0, full-page OCR (scanned pages)
#
#Surya is the PRIMARY OCR engine. pytesseract is an automatic FALLBACK,
#used only if Surya can't be imported, can't load its models, or throws
#during inference. You don't need to choose one manually -- this module
#tries Surya first every time and only drops to tesseract if that fails.
#
#TARGETS: surya-ocr==0.14.7 (confirmed via `pip show surya-ocr` against
#the actual installed environment -- see version history below for why
#this matters more than it sounds like it should).
#
#── Three different Surya APIs found across versions, confirmed by
#   downloading and inspecting the actual PyPI packages, not guessing ──
#
#1. surya-ocr==0.6.13 -- what the ORIGINAL ocr_reader.py was written
#   against: standalone functions, `surya.model.detection.model.load_model`,
#   `surya.model.recognition.model.load_model`, `surya.ocr.run_ocr(images,
#   langs, det_model, det_processor, rec_model, rec_processor)`. Even in
#   this exact version the original code had a wrong import path
#   (`surya.model.detection.PROCESSOR.load_processor` -- that function
#   actually lives in `surya.model.detection.MODEL`, confirmed against
#   the real file contents).
#
#2. surya-ocr==0.14.7 -- what's ACTUALLY INSTALLED in your venv (confirmed
#   via your `pip show surya-ocr` output). No `surya.model` package at
#   all anymore -- detection and recognition are now predictor CLASSES:
#   `surya.detection.DetectionPredictor`, `surya.recognition.RecognitionPredictor`.
#   No `langs` parameter either (dropped in this generation). Call shape:
#
#       det_predictor = DetectionPredictor()
#       rec_predictor = RecognitionPredictor()
#       results = rec_predictor(images, det_predictor=det_predictor)
#
#   This is the version this file is now written against.
#
#3. surya-ocr==0.22.x (current PyPI default if you `pip install surya-ocr`
#   with no version pin) -- a further VLM-based rewrite requiring a
#   running model server. Different again from both of the above.
#
#The lesson: don't assume a Surya upgrade/reinstall keeps the same call
#shape. If you ever bump this dependency, re-check `pip show surya-ocr`
#and re-verify the import shape before assuming this file still works --
#it's changed shape at least 3 times.
#
#── Fallback behaviour ──
#If Surya's imports fail, its models fail to load, or the predictor call
#throws for any reason, this module logs a warning once and falls back
#to ocr_reader_pytesseract.py for that call. The fallback is all-or-
#nothing per batch (not per-page) -- if Surya breaks partway through a
#multi-page document, the whole document's scanned pages go to
#tesseract rather than trying to reconcile a half-Surya, half-tesseract
#result.
#
#── Model caching ──
#Your original _load_surya_models() reloaded all 4 models on every call
#to read_with_surya_pages(). Across a batch of 60+ resumes that's 60+
#model loads. Predictors are now cached at module level after first load.
#
#── REVISION (this version) — broadened LaTeX/math markup cleanup ──
#Confirmed on a real scanned job description (SMFG_Job_Document_Scanned
#.pdf): the previous _clean_surya_text() only stripped the specific
#<math>\bullet</math> pattern plus a handful of inline formatting tags
#(b/i/u/del/mark/sup/sub/br). That's too narrow. Surya's format-aware
#recognition model also wraps STACKED/logo-style text -- e.g. a company
#logo rendered as "SMFG" on one line with "IndiaCredit" in smaller type
#underneath it -- in full LaTeX subscript/font markup that looks nothing
#like the bullet case:
#
#    \mathop{\hbox{\rm SMFG}}_{\hbox{\scriptsize \rm Indiacredit}}
#
#None of the old regexes touch \mathop, \hbox, \rm, or \scriptsize, so
#this leaked straight into extracted text unchanged. _strip_latex_markup()
#below is a general-purpose peel: it unwraps any <math>...</math> wrapper
#keeping the inner text, repeatedly peels \command{content} and
#_{content}/^{content} groups from the inside out (handles nesting like
#the case above, where \hbox{...} sits inside \mathop{...}), drops bare
#style commands (\rm, \scriptsize, \mathbf, etc. with no braces), keeps
#\bullet -> "•" as before, and finally sweeps up any stray backslash-
#commands or leftover braces that survive as debris. This is a strict
#superset of the old behavior -- the original <math>\bullet</math> and
#\bullet-only cases still resolve to "•" exactly as before (see
#tests run against both the bullet case and this SMFG logo case before
#this revision was accepted).
#
#NOTE: this was validated by running it directly against the actual
#leaked string above (confirmed output: "SMFG Indiacredit") and against
#the pre-existing bullet-math case, but NOT against a live Surya model
#in every possible shape it might emit -- if you see other un-stripped
#LaTeX-looking debris in future output, it's a sign this peel needs
#another case added, not that the approach is wrong.
#
#── REVISION (this version) — pre-OCR image quality gate ──
#Added a preprocessing step (ingestion/image_preprocessing.py) that runs
#on every rendered/loaded image BEFORE it reaches Surya or tesseract:
#
#  1. Blur rejection -- raises ImageQualityError for pages too blurry to
#     trust (calibrated via a controlled blur ladder: sharp sentences but
#     silently wrong fields like emails/dates/proper-nouns start
#     appearing well before the image looks like obvious garbage, so this
#     is a hard reject, not a warning).
#  2. Deskew -- corrects page rotation. Justified by a real finding on
#     this project: a 4.5-degree rotation caused a section heading to be
#     sorted into the middle of an unrelated bullet sentence, and dropped
#     a line entirely, because reading-order reconstruction sorts by
#     y-coordinate and a rotated page's "same visual line" has different
#     y-coordinates on the left vs. right edge.
#  3. Brightness correction -- inverts pages that are moderately dark
#     overall (a real underexposed photo), narrower than the equivalent
#     invoice-pipeline heuristic since resumes commonly have colored
#     design elements that a naive whole-page check could misfire on.
#
#ImageQualityError must propagate all the way up to the caller as a
#genuine rejection -- it is explicitly NOT caught by the generic
#Surya-failure except-block below, so a blurry page is rejected outright
#rather than silently retried with tesseract (tesseract cannot recover
#detail a blur destroyed any better than Surya can).
#"""
#
#import logging
#import re
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF
#
#from ingestion.image_preprocessing import preprocess_scanned_image, ImageQualityError
#
#logger = logging.getLogger(__name__)
#
## ── LaTeX/math markup cleanup (see REVISION note above) ──
#_MATH_BLOCK_RE = re.compile(r"<math[^>]*>(.*?)</math>", re.IGNORECASE | re.DOTALL)
#_LATEX_BRACE_CMD_RE = re.compile(r"\\[a-zA-Z]+\{([^{}]*)\}")
#_LATEX_SCRIPT_RE = re.compile(r"[_^]\{([^{}]*)\}")
#_LATEX_NOARG_CMD_RE = re.compile(
#    r"\\(rm|bf|it|tt|sf|em|scriptsize|scriptstyle|displaystyle|textstyle|"
#    r"normalsize|tiny|small|footnotesize|large|Large|LARGE|huge|Huge|"
#    r"mathop|mathrm|mathbf|mathit)\b"
#)
#_BARE_BULLET_CMD_RE = re.compile(r"\\bullet")
#_LATEX_STRAY_CMD_RE = re.compile(r"\\[a-zA-Z]+")
#_LATEX_STRAY_BRACE_RE = re.compile(r"[{}]")
#_FORMAT_TAG_RE = re.compile(r"</?(b|i|u|del|mark|sup|sub|br)\b[^>]*>", re.IGNORECASE)
#_LONE_BULLET_GLYPH_LINE_RE = re.compile(r"^[\u0660\u06F0.]$")
#
#
#def _strip_latex_markup(text: str) -> str:
#    """
#    General-purpose LaTeX/math markup peel. Order matters:
#      1. Unwrap <math>...</math> tags, keeping inner content (the tag
#         itself carries no information we need, but the content inside
#         -- e.g. \bullet -- still needs further processing below).
#      2. Repeatedly peel \command{content} and _{content}/^{content}
#         groups from the INSIDE OUT (the regexes only match when the
#         braced content itself has no further braces, so nested cases
#         like \mathop{\hbox{\rm SMFG}} resolve one layer per loop
#         iteration until stable).
#      3. Turn \bullet into a real "•" (must happen before step 4, or
#         the generic stray-command sweep below would just delete it).
#      4. Drop bare style commands with no braces (\rm, \scriptsize,
#         \mathbf, etc.) that steps 1-2 already stripped their braced
#         argument away from.
#      5. Sweep up anything left over: stray \commands with no braces
#         at all, and any orphaned { or } that never had a matching
#         command peeled around them.
#      6. Collapse the extra runs of whitespace all this stripping
#         tends to leave behind.
#    """
#    text = _MATH_BLOCK_RE.sub(r"\1", text)
#    prev = None
#    while prev != text:
#        prev = text
#        text = _LATEX_BRACE_CMD_RE.sub(r"\1", text)
#        text = _LATEX_SCRIPT_RE.sub(r"\1", text)
#    text = _BARE_BULLET_CMD_RE.sub("•", text)
#    text = _LATEX_NOARG_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_BRACE_RE.sub("", text)
#    text = re.sub(r"[ \t]{2,}", " ", text)
#    return text
#
#
#def _clean_surya_text(text: str) -> str:
#    """
#    surya-ocr 0.14.7's recognition model is format-aware, not plain OCR:
#    it wraps text it judged bold in <b>...</b>, tags bullet-point dots
#    as <math display="inline">\\bullet</math>, and -- confirmed on a
#    real scanned job description (SMFG_Job_Document_Scanned.pdf) -- also
#    wraps stacked/logo-style text (a company name over a smaller tagline)
#    in full LaTeX subscript markup like
#    \\mathop{\\hbox{\\rm SMFG}}_{\\hbox{\\scriptsize \\rm Indiacredit}}.
#    See _strip_latex_markup() above for the general-purpose cleanup that
#    now handles all of these cases, not just the bullet one.
#
#    Separately, the same model sometimes misreads the bullet glyph
#    entirely as some other small-dot-shaped character -- an Arabic-Indic
#    digit zero (٠, U+0660), Extended Arabic-Indic digit zero (۰,
#    U+06F0), or even a plain period (.) -- all of which render as a
#    small dot in many fonts, visually close enough to fool it. These
#    show up as a lone character alone on its own line where a bullet
#    should be. Confirmed on the same resume: two bullets came back as a
#    bare "." on their own line.
#
#    This normalizes all of the above down to a plain "•" and strips
#    other inline formatting tags to their plain text content, so
#    section_splitter.py sees ordinary text instead of markup it was
#    never built to parse.
#    """
#    text = _strip_latex_markup(text)
#    text = _FORMAT_TAG_RE.sub("", text)
#    cleaned_lines = [
#        "•" if _LONE_BULLET_GLYPH_LINE_RE.match(line.strip()) else line
#        for line in text.split("\n")
#    ]
#    return "\n".join(cleaned_lines)
#
## ── Surya availability check + predictor cache (populated lazily, once) ──
#_surya_predictors = None   # (det_predictor, rec_predictor) once loaded
#_surya_import_failed = False
#_surya_warned = False
#
#
#def _warn_once(message: str):
#    global _surya_warned
#    if not _surya_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _surya_warned = True
#
#
#def _load_surya_predictors():
#    """Loads and caches Surya's detection + recognition predictors.
#    Raises on any failure -- callers must catch and fall back to
#    pytesseract."""
#    global _surya_predictors
#    if _surya_predictors is not None:
#        return _surya_predictors
#
#    # surya-ocr==0.14.7 API: predictor classes, not standalone
#    # load_model()/load_processor() functions -- see module docstring.
#    from surya.detection import DetectionPredictor
#    from surya.recognition import RecognitionPredictor
#
#    print("[INFO] Loading Surya OCR predictors (first call only; cached after)...")
#    det_predictor = DetectionPredictor()
#    rec_predictor = RecognitionPredictor()
#
#    _surya_predictors = (det_predictor, rec_predictor)
#    return _surya_predictors
#
#
#def _run_surya_on_images(images: list) -> list:
#    """Raises on any failure -- callers must catch and fall back."""
#    det_predictor, rec_predictor = _load_surya_predictors()
#
#    # 0.14.7 dropped the explicit `langs` parameter entirely -- passing
#    # it raises TypeError. Detection is supplied via det_predictor, and
#    # recognition runs language-agnostic OCR by default.
#    results = rec_predictor(images, det_predictor=det_predictor)
#
#    all_text = []
#    for page_result in results:
#        page_lines = [line.text for line in page_result.text_lines if line.confidence > 0.3]
#        page_text = _clean_surya_text("\n".join(page_lines))
#        all_text.append(page_text)
#    return all_text
#
#
#def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
#    """Converts pages of a PDF into PIL Images at 200 DPI (good balance
#    of Surya accuracy vs. speed). page_indices=None converts every page.
#
#    NEW: each rendered page now goes through preprocess_scanned_image()
#    (blur gate + deskew + brightness correction) before being handed
#    back to the caller. A page that fails the blur gate raises
#    ImageQualityError here -- see read_with_surya_pages() below, which
#    is responsible for NOT swallowing that into the generic
#    tesseract-fallback path."""
#    doc = fitz.open(pdf_path)
#    images = []
#    indices = page_indices if page_indices is not None else range(len(doc))
#    for page_num in indices:
#        page = doc[page_num]
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        img = preprocess_scanned_image(img, context=f"page {page_num + 1} of {pdf_path}")
#        images.append(img)
#    doc.close()
#    return images
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    OCRs the given 0-based page indices, Surya first, tesseract fallback
#    on any failure. Returns {page_index: text}. This is the function
#    pdf_reader.py calls for pages classified "scanned".
#
#    NEW: ImageQualityError (raised by pdf_to_images()'s preprocessing
#    step, via preprocess_scanned_image()) is deliberately re-raised
#    BEFORE the generic except-block below, so a blur rejection is never
#    mistaken for "Surya failed, try tesseract instead" -- tesseract
#    cannot recover detail a blur destroyed any better than Surya can,
#    so falling back to it here would just waste time re-processing an
#    image we've already determined is unusable, and risk silently
#    returning low-quality tesseract output instead of a clear rejection.
#    """
#    if not page_indices:
#        return {}
#
#    global _surya_import_failed
#    if not _surya_import_failed:
#        try:
#            images = pdf_to_images(pdf_path, page_indices=page_indices)
#            texts = _run_surya_on_images(images)
#            return dict(zip(page_indices, texts))
#        except ImageQualityError:
#            raise
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract for this and all subsequent pages. "
#                f"Check `pip show surya-ocr` is exactly 0.6.13, and that model "
#                f"downloads aren't being blocked by network/firewall."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages as tesseract_fallback
#    return tesseract_fallback(pdf_path, page_indices)
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Whole-file OCR for a standalone image or a fully-scanned PDF,
#    joined into one string. Surya first, tesseract fallback on failure.
#
#    NEW: standalone image uploads (.jpg/.png/etc.) now also go through
#    preprocess_scanned_image() before OCR, same as PDF pages. See
#    read_with_surya_pages() docstring for why ImageQualityError is
#    re-raised before the generic fallback except-block.
#    """
#    global _surya_import_failed
#    path = Path(file_path)
#
#    if not _surya_import_failed:
#        try:
#            if path.suffix.lower() == ".pdf":
#                images = pdf_to_images(file_path)
#            elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#                img = Image.open(file_path).convert("RGB")
#                img = preprocess_scanned_image(img, context=file_path)
#                images = [img]
#            else:
#                raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#            all_text = _run_surya_on_images(images)
#            return "\n\n".join(all_text)
#        except ImageQualityError:
#            raise
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya as tesseract_fallback
#    return tesseract_fallback(file_path)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_with_surya(sys.argv[1]))
#
#
#







##worked just commenting for adding image preprocessing step and written above
#"""
#ocr_reader.py — Layer 0, full-page OCR (scanned pages)
#
#Surya is the PRIMARY OCR engine. pytesseract is an automatic FALLBACK,
#used only if Surya can't be imported, can't load its models, or throws
#during inference. You don't need to choose one manually -- this module
#tries Surya first every time and only drops to tesseract if that fails.
#
#TARGETS: surya-ocr==0.14.7 (confirmed via `pip show surya-ocr` against
#the actual installed environment -- see version history below for why
#this matters more than it sounds like it should).
#
#── Three different Surya APIs found across versions, confirmed by
#   downloading and inspecting the actual PyPI packages, not guessing ──
#
#1. surya-ocr==0.6.13 -- what the ORIGINAL ocr_reader.py was written
#   against: standalone functions, `surya.model.detection.model.load_model`,
#   `surya.model.recognition.model.load_model`, `surya.ocr.run_ocr(images,
#   langs, det_model, det_processor, rec_model, rec_processor)`. Even in
#   this exact version the original code had a wrong import path
#   (`surya.model.detection.PROCESSOR.load_processor` -- that function
#   actually lives in `surya.model.detection.MODEL`, confirmed against
#   the real file contents).
#
#2. surya-ocr==0.14.7 -- what's ACTUALLY INSTALLED in your venv (confirmed
#   via your `pip show surya-ocr` output). No `surya.model` package at
#   all anymore -- detection and recognition are now predictor CLASSES:
#   `surya.detection.DetectionPredictor`, `surya.recognition.RecognitionPredictor`.
#   No `langs` parameter either (dropped in this generation). Call shape:
#
#       det_predictor = DetectionPredictor()
#       rec_predictor = RecognitionPredictor()
#       results = rec_predictor(images, det_predictor=det_predictor)
#
#   This is the version this file is now written against.
#
#3. surya-ocr==0.22.x (current PyPI default if you `pip install surya-ocr`
#   with no version pin) -- a further VLM-based rewrite requiring a
#   running model server. Different again from both of the above.
#
#The lesson: don't assume a Surya upgrade/reinstall keeps the same call
#shape. If you ever bump this dependency, re-check `pip show surya-ocr`
#and re-verify the import shape before assuming this file still works --
#it's changed shape at least 3 times.
#
#── Fallback behaviour ──
#If Surya's imports fail, its models fail to load, or the predictor call
#throws for any reason, this module logs a warning once and falls back
#to ocr_reader_pytesseract.py for that call. The fallback is all-or-
#nothing per batch (not per-page) -- if Surya breaks partway through a
#multi-page document, the whole document's scanned pages go to
#tesseract rather than trying to reconcile a half-Surya, half-tesseract
#result.
#
#── Model caching ──
#Your original _load_surya_models() reloaded all 4 models on every call
#to read_with_surya_pages(). Across a batch of 60+ resumes that's 60+
#model loads. Predictors are now cached at module level after first load.
#
#── REVISION (this version) — broadened LaTeX/math markup cleanup ──
#Confirmed on a real scanned job description (SMFG_Job_Document_Scanned
#.pdf): the previous _clean_surya_text() only stripped the specific
#<math>\bullet</math> pattern plus a handful of inline formatting tags
#(b/i/u/del/mark/sup/sub/br). That's too narrow. Surya's format-aware
#recognition model also wraps STACKED/logo-style text -- e.g. a company
#logo rendered as "SMFG" on one line with "IndiaCredit" in smaller type
#underneath it -- in full LaTeX subscript/font markup that looks nothing
#like the bullet case:
#
#    \mathop{\hbox{\rm SMFG}}_{\hbox{\scriptsize \rm Indiacredit}}
#
#None of the old regexes touch \mathop, \hbox, \rm, or \scriptsize, so
#this leaked straight into extracted text unchanged. _strip_latex_markup()
#below is a general-purpose peel: it unwraps any <math>...</math> wrapper
#keeping the inner text, repeatedly peels \command{content} and
#_{content}/^{content} groups from the inside out (handles nesting like
#the case above, where \hbox{...} sits inside \mathop{...}), drops bare
#style commands (\rm, \scriptsize, \mathbf, etc. with no braces), keeps
#\bullet -> "•" as before, and finally sweeps up any stray backslash-
#commands or leftover braces that survive as debris. This is a strict
#superset of the old behavior -- the original <math>\bullet</math> and
#\bullet-only cases still resolve to "•" exactly as before (see
#tests run against both the bullet case and this SMFG logo case before
#this revision was accepted).
#
#NOTE: this was validated by running it directly against the actual
#leaked string above (confirmed output: "SMFG Indiacredit") and against
#the pre-existing bullet-math case, but NOT against a live Surya model
#in every possible shape it might emit -- if you see other un-stripped
#LaTeX-looking debris in future output, it's a sign this peel needs
#another case added, not that the approach is wrong.
#"""
#
#import logging
#import re
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF
#
#logger = logging.getLogger(__name__)
#
## ── LaTeX/math markup cleanup (see REVISION note above) ──
#_MATH_BLOCK_RE = re.compile(r"<math[^>]*>(.*?)</math>", re.IGNORECASE | re.DOTALL)
#_LATEX_BRACE_CMD_RE = re.compile(r"\\[a-zA-Z]+\{([^{}]*)\}")
#_LATEX_SCRIPT_RE = re.compile(r"[_^]\{([^{}]*)\}")
#_LATEX_NOARG_CMD_RE = re.compile(
#    r"\\(rm|bf|it|tt|sf|em|scriptsize|scriptstyle|displaystyle|textstyle|"
#    r"normalsize|tiny|small|footnotesize|large|Large|LARGE|huge|Huge|"
#    r"mathop|mathrm|mathbf|mathit)\b"
#)
#_BARE_BULLET_CMD_RE = re.compile(r"\\bullet")
#_LATEX_STRAY_CMD_RE = re.compile(r"\\[a-zA-Z]+")
#_LATEX_STRAY_BRACE_RE = re.compile(r"[{}]")
#_FORMAT_TAG_RE = re.compile(r"</?(b|i|u|del|mark|sup|sub|br)\b[^>]*>", re.IGNORECASE)
#_LONE_BULLET_GLYPH_LINE_RE = re.compile(r"^[\u0660\u06F0.]$")
#
#
#def _strip_latex_markup(text: str) -> str:
#    """
#    General-purpose LaTeX/math markup peel. Order matters:
#      1. Unwrap <math>...</math> tags, keeping inner content (the tag
#         itself carries no information we need, but the content inside
#         -- e.g. \bullet -- still needs further processing below).
#      2. Repeatedly peel \command{content} and _{content}/^{content}
#         groups from the INSIDE OUT (the regexes only match when the
#         braced content itself has no further braces, so nested cases
#         like \mathop{\hbox{\rm SMFG}} resolve one layer per loop
#         iteration until stable).
#      3. Turn \bullet into a real "•" (must happen before step 4, or
#         the generic stray-command sweep below would just delete it).
#      4. Drop bare style commands with no braces (\rm, \scriptsize,
#         \mathbf, etc.) that steps 1-2 already stripped their braced
#         argument away from.
#      5. Sweep up anything left over: stray \commands with no braces
#         at all, and any orphaned { or } that never had a matching
#         command peeled around them.
#      6. Collapse the extra runs of whitespace all this stripping
#         tends to leave behind.
#    """
#    text = _MATH_BLOCK_RE.sub(r"\1", text)
#    prev = None
#    while prev != text:
#        prev = text
#        text = _LATEX_BRACE_CMD_RE.sub(r"\1", text)
#        text = _LATEX_SCRIPT_RE.sub(r"\1", text)
#    text = _BARE_BULLET_CMD_RE.sub("•", text)
#    text = _LATEX_NOARG_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_CMD_RE.sub("", text)
#    text = _LATEX_STRAY_BRACE_RE.sub("", text)
#    text = re.sub(r"[ \t]{2,}", " ", text)
#    return text
#
#
#def _clean_surya_text(text: str) -> str:
#    """
#    surya-ocr 0.14.7's recognition model is format-aware, not plain OCR:
#    it wraps text it judged bold in <b>...</b>, tags bullet-point dots
#    as <math display="inline">\\bullet</math>, and -- confirmed on a
#    real scanned job description (SMFG_Job_Document_Scanned.pdf) -- also
#    wraps stacked/logo-style text (a company name over a smaller tagline)
#    in full LaTeX subscript markup like
#    \\mathop{\\hbox{\\rm SMFG}}_{\\hbox{\\scriptsize \\rm Indiacredit}}.
#    See _strip_latex_markup() above for the general-purpose cleanup that
#    now handles all of these cases, not just the bullet one.
#
#    Separately, the same model sometimes misreads the bullet glyph
#    entirely as some other small-dot-shaped character -- an Arabic-Indic
#    digit zero (٠, U+0660), Extended Arabic-Indic digit zero (۰,
#    U+06F0), or even a plain period (.) -- all of which render as a
#    small dot in many fonts, visually close enough to fool it. These
#    show up as a lone character alone on its own line where a bullet
#    should be. Confirmed on the same resume: two bullets came back as a
#    bare "." on their own line.
#
#    This normalizes all of the above down to a plain "•" and strips
#    other inline formatting tags to their plain text content, so
#    section_splitter.py sees ordinary text instead of markup it was
#    never built to parse.
#    """
#    text = _strip_latex_markup(text)
#    text = _FORMAT_TAG_RE.sub("", text)
#    cleaned_lines = [
#        "•" if _LONE_BULLET_GLYPH_LINE_RE.match(line.strip()) else line
#        for line in text.split("\n")
#    ]
#    return "\n".join(cleaned_lines)
#
## ── Surya availability check + predictor cache (populated lazily, once) ──
#_surya_predictors = None   # (det_predictor, rec_predictor) once loaded
#_surya_import_failed = False
#_surya_warned = False
#
#
#def _warn_once(message: str):
#    global _surya_warned
#    if not _surya_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _surya_warned = True
#
#
#def _load_surya_predictors():
#    """Loads and caches Surya's detection + recognition predictors.
#    Raises on any failure -- callers must catch and fall back to
#    pytesseract."""
#    global _surya_predictors
#    if _surya_predictors is not None:
#        return _surya_predictors
#
#    # surya-ocr==0.14.7 API: predictor classes, not standalone
#    # load_model()/load_processor() functions -- see module docstring.
#    from surya.detection import DetectionPredictor
#    from surya.recognition import RecognitionPredictor
#
#    print("[INFO] Loading Surya OCR predictors (first call only; cached after)...")
#    det_predictor = DetectionPredictor()
#    rec_predictor = RecognitionPredictor()
#
#    _surya_predictors = (det_predictor, rec_predictor)
#    return _surya_predictors
#
#
#def _run_surya_on_images(images: list) -> list:
#    """Raises on any failure -- callers must catch and fall back."""
#    det_predictor, rec_predictor = _load_surya_predictors()
#
#    # 0.14.7 dropped the explicit `langs` parameter entirely -- passing
#    # it raises TypeError. Detection is supplied via det_predictor, and
#    # recognition runs language-agnostic OCR by default.
#    results = rec_predictor(images, det_predictor=det_predictor)
#
#    all_text = []
#    for page_result in results:
#        page_lines = [line.text for line in page_result.text_lines if line.confidence > 0.3]
#        page_text = _clean_surya_text("\n".join(page_lines))
#        all_text.append(page_text)
#    return all_text
#
#
#def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
#    """Converts pages of a PDF into PIL Images at 200 DPI (good balance
#    of Surya accuracy vs. speed). page_indices=None converts every page."""
#    doc = fitz.open(pdf_path)
#    images = []
#    indices = page_indices if page_indices is not None else range(len(doc))
#    for page_num in indices:
#        page = doc[page_num]
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        images.append(img)
#    doc.close()
#    return images
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    OCRs the given 0-based page indices, Surya first, tesseract fallback
#    on any failure. Returns {page_index: text}. This is the function
#    pdf_reader.py calls for pages classified "scanned".
#    """
#    if not page_indices:
#        return {}
#
#    global _surya_import_failed
#    if not _surya_import_failed:
#        try:
#            images = pdf_to_images(pdf_path, page_indices=page_indices)
#            texts = _run_surya_on_images(images)
#            return dict(zip(page_indices, texts))
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract for this and all subsequent pages. "
#                f"Check `pip show surya-ocr` is exactly 0.6.13, and that model "
#                f"downloads aren't being blocked by network/firewall."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages as tesseract_fallback
#    return tesseract_fallback(pdf_path, page_indices)
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Whole-file OCR for a standalone image or a fully-scanned PDF,
#    joined into one string. Surya first, tesseract fallback on failure.
#    """
#    global _surya_import_failed
#    path = Path(file_path)
#
#    if not _surya_import_failed:
#        try:
#            if path.suffix.lower() == ".pdf":
#                images = pdf_to_images(file_path)
#            elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#                images = [Image.open(file_path).convert("RGB")]
#            else:
#                raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#            all_text = _run_surya_on_images(images)
#            return "\n\n".join(all_text)
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya as tesseract_fallback
#    return tesseract_fallback(file_path)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_with_surya(sys.argv[1]))
#
#





##commenting to improve smgp resume
#"""
#ocr_reader.py — Layer 0, full-page OCR (scanned pages)
#
#Surya is the PRIMARY OCR engine. pytesseract is an automatic FALLBACK,
#used only if Surya can't be imported, can't load its models, or throws
#during inference. You don't need to choose one manually -- this module
#tries Surya first every time and only drops to tesseract if that fails.
#
#TARGETS: surya-ocr==0.14.7 (confirmed via `pip show surya-ocr` against
#the actual installed environment -- see version history below for why
#this matters more than it sounds like it should).
#
#── Three different Surya APIs found across versions, confirmed by
#   downloading and inspecting the actual PyPI packages, not guessing ──
#
#1. surya-ocr==0.6.13 -- what the ORIGINAL ocr_reader.py was written
#   against: standalone functions, `surya.model.detection.model.load_model`,
#   `surya.model.recognition.model.load_model`, `surya.ocr.run_ocr(images,
#   langs, det_model, det_processor, rec_model, rec_processor)`. Even in
#   this exact version the original code had a wrong import path
#   (`surya.model.detection.PROCESSOR.load_processor` -- that function
#   actually lives in `surya.model.detection.MODEL`, confirmed against
#   the real file contents).
#
#2. surya-ocr==0.14.7 -- what's ACTUALLY INSTALLED in your venv (confirmed
#   via your `pip show surya-ocr` output). No `surya.model` package at
#   all anymore -- detection and recognition are now predictor CLASSES:
#   `surya.detection.DetectionPredictor`, `surya.recognition.RecognitionPredictor`.
#   No `langs` parameter either (dropped in this generation). Call shape:
#
#       det_predictor = DetectionPredictor()
#       rec_predictor = RecognitionPredictor()
#       results = rec_predictor(images, det_predictor=det_predictor)
#
#   This is the version this file is now written against.
#
#3. surya-ocr==0.22.x (current PyPI default if you `pip install surya-ocr`
#   with no version pin) -- a further VLM-based rewrite requiring a
#   running model server. Different again from both of the above.
#
#The lesson: don't assume a Surya upgrade/reinstall keeps the same call
#shape. If you ever bump this dependency, re-check `pip show surya-ocr`
#and re-verify the import shape before assuming this file still works --
#it's changed shape at least 3 times.
#
#── Fallback behaviour ──
#If Surya's imports fail, its models fail to load, or the predictor call
#throws for any reason, this module logs a warning once and falls back
#to ocr_reader_pytesseract.py for that call. The fallback is all-or-
#nothing per batch (not per-page) -- if Surya breaks partway through a
#multi-page document, the whole document's scanned pages go to
#tesseract rather than trying to reconcile a half-Surya, half-tesseract
#result.
#
#── Model caching ──
#Your original _load_surya_models() reloaded all 4 models on every call
#to read_with_surya_pages(). Across a batch of 60+ resumes that's 60+
#model loads. Predictors are now cached at module level after first load.
#"""
#
#import logging
#import re
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF
#
#logger = logging.getLogger(__name__)
#
#_MATH_BULLET_RE = re.compile(r"<math[^>]*>\s*\\bullet\s*</math>", re.IGNORECASE)
#_MATH_TAG_RE = re.compile(r"</?math[^>]*>", re.IGNORECASE)
#_BARE_BULLET_CMD_RE = re.compile(r"\\bullet")
#_FORMAT_TAG_RE = re.compile(r"</?(b|i|u|del|mark|sup|sub|br)\b[^>]*>", re.IGNORECASE)
#_LONE_BULLET_GLYPH_LINE_RE = re.compile(r"^[\u0660\u06F0.]$")
#
#
#def _clean_surya_text(text: str) -> str:
#    """
#    surya-ocr 0.14.7's recognition model is format-aware, not plain OCR:
#    it wraps text it judged bold in <b>...</b>, and tags bullet-point
#    dots as <math display="inline">\\bullet</math> -- a filled circle
#    glyph looks identical to the LaTeX \\bullet symbol to a math-aware
#    model. Confirmed on a real resume (Amol Jadhav): nearly every
#    bullet point came back wrapped this way instead of as plain "•".
#
#    Separately, the same model sometimes misreads the bullet glyph
#    entirely as some other small-dot-shaped character -- an Arabic-Indic
#    digit zero (٠, U+0660), Extended Arabic-Indic digit zero (۰,
#    U+06F0), or even a plain period (.) -- all of which render as a
#    small dot in many fonts, visually close enough to fool it. These
#    show up as a lone character alone on its own line where a bullet
#    should be. Confirmed on the same resume: two bullets came back as a
#    bare "." on their own line.
#
#    This normalizes all of the above down to a plain "•" and strips
#    other inline formatting tags to their plain text content, so
#    section_splitter.py sees ordinary text instead of markup it was
#    never built to parse.
#    """
#    text = _MATH_BULLET_RE.sub("•", text)
#    text = _MATH_TAG_RE.sub("", text)
#    text = _BARE_BULLET_CMD_RE.sub("•", text)
#    text = _FORMAT_TAG_RE.sub("", text)
#    cleaned_lines = [
#        "•" if _LONE_BULLET_GLYPH_LINE_RE.match(line.strip()) else line
#        for line in text.split("\n")
#    ]
#    return "\n".join(cleaned_lines)
#
## ── Surya availability check + predictor cache (populated lazily, once) ──
#_surya_predictors = None   # (det_predictor, rec_predictor) once loaded
#_surya_import_failed = False
#_surya_warned = False
#
#
#def _warn_once(message: str):
#    global _surya_warned
#    if not _surya_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _surya_warned = True
#
#
#def _load_surya_predictors():
#    """Loads and caches Surya's detection + recognition predictors.
#    Raises on any failure -- callers must catch and fall back to
#    pytesseract."""
#    global _surya_predictors
#    if _surya_predictors is not None:
#        return _surya_predictors
#
#    # surya-ocr==0.14.7 API: predictor classes, not standalone
#    # load_model()/load_processor() functions -- see module docstring.
#    from surya.detection import DetectionPredictor
#    from surya.recognition import RecognitionPredictor
#
#    print("[INFO] Loading Surya OCR predictors (first call only; cached after)...")
#    det_predictor = DetectionPredictor()
#    rec_predictor = RecognitionPredictor()
#
#    _surya_predictors = (det_predictor, rec_predictor)
#    return _surya_predictors
#
#
#def _run_surya_on_images(images: list) -> list:
#    """Raises on any failure -- callers must catch and fall back."""
#    det_predictor, rec_predictor = _load_surya_predictors()
#
#    # 0.14.7 dropped the explicit `langs` parameter entirely -- passing
#    # it raises TypeError. Detection is supplied via det_predictor, and
#    # recognition runs language-agnostic OCR by default.
#    results = rec_predictor(images, det_predictor=det_predictor)
#
#    all_text = []
#    for page_result in results:
#        page_lines = [line.text for line in page_result.text_lines if line.confidence > 0.3]
#        page_text = _clean_surya_text("\n".join(page_lines))
#        all_text.append(page_text)
#    return all_text
#
#
#def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
#    """Converts pages of a PDF into PIL Images at 200 DPI (good balance
#    of Surya accuracy vs. speed). page_indices=None converts every page."""
#    doc = fitz.open(pdf_path)
#    images = []
#    indices = page_indices if page_indices is not None else range(len(doc))
#    for page_num in indices:
#        page = doc[page_num]
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        images.append(img)
#    doc.close()
#    return images
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    OCRs the given 0-based page indices, Surya first, tesseract fallback
#    on any failure. Returns {page_index: text}. This is the function
#    pdf_reader.py calls for pages classified "scanned".
#    """
#    if not page_indices:
#        return {}
#
#    global _surya_import_failed
#    if not _surya_import_failed:
#        try:
#            images = pdf_to_images(pdf_path, page_indices=page_indices)
#            texts = _run_surya_on_images(images)
#            return dict(zip(page_indices, texts))
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract for this and all subsequent pages. "
#                f"Check `pip show surya-ocr` is exactly 0.6.13, and that model "
#                f"downloads aren't being blocked by network/firewall."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages as tesseract_fallback
#    return tesseract_fallback(pdf_path, page_indices)
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Whole-file OCR for a standalone image or a fully-scanned PDF,
#    joined into one string. Surya first, tesseract fallback on failure.
#    """
#    global _surya_import_failed
#    path = Path(file_path)
#
#    if not _surya_import_failed:
#        try:
#            if path.suffix.lower() == ".pdf":
#                images = pdf_to_images(file_path)
#            elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#                images = [Image.open(file_path).convert("RGB")]
#            else:
#                raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#            all_text = _run_surya_on_images(images)
#            return "\n\n".join(all_text)
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya as tesseract_fallback
#    return tesseract_fallback(file_path)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_with_surya(sys.argv[1]))
#
#






#
#"""
#ocr_reader.py — Layer 0, full-page OCR (scanned pages)
#
#Surya is the PRIMARY OCR engine. pytesseract is an automatic FALLBACK,
#used only if Surya can't be imported, can't load its models, or throws
#during inference. You don't need to choose one manually -- this module
#tries Surya first every time and only drops to tesseract if that fails.
#
#TARGETS: surya-ocr==0.14.7 (confirmed via `pip show surya-ocr` against
#the actual installed environment -- see version history below for why
#this matters more than it sounds like it should).
#
#── Three different Surya APIs found across versions, confirmed by
#   downloading and inspecting the actual PyPI packages, not guessing ──
#
#1. surya-ocr==0.6.13 -- what the ORIGINAL ocr_reader.py was written
#   against: standalone functions, `surya.model.detection.model.load_model`,
#   `surya.model.recognition.model.load_model`, `surya.ocr.run_ocr(images,
#   langs, det_model, det_processor, rec_model, rec_processor)`. Even in
#   this exact version the original code had a wrong import path
#   (`surya.model.detection.PROCESSOR.load_processor` -- that function
#   actually lives in `surya.model.detection.MODEL`, confirmed against
#   the real file contents).
#
#2. surya-ocr==0.14.7 -- what's ACTUALLY INSTALLED in your venv (confirmed
#   via your `pip show surya-ocr` output). No `surya.model` package at
#   all anymore -- detection and recognition are now predictor CLASSES:
#   `surya.detection.DetectionPredictor`, `surya.recognition.RecognitionPredictor`.
#   No `langs` parameter either (dropped in this generation). Call shape:
#
#       det_predictor = DetectionPredictor()
#       rec_predictor = RecognitionPredictor()
#       results = rec_predictor(images, det_predictor=det_predictor)
#
#   This is the version this file is now written against.
#
#3. surya-ocr==0.22.x (current PyPI default if you `pip install surya-ocr`
#   with no version pin) -- a further VLM-based rewrite requiring a
#   running model server. Different again from both of the above.
#
#The lesson: don't assume a Surya upgrade/reinstall keeps the same call
#shape. If you ever bump this dependency, re-check `pip show surya-ocr`
#and re-verify the import shape before assuming this file still works --
#it's changed shape at least 3 times.
#
#── Fallback behaviour ──
#If Surya's imports fail, its models fail to load, or the predictor call
#throws for any reason, this module logs a warning once and falls back
#to ocr_reader_pytesseract.py for that call. The fallback is all-or-
#nothing per batch (not per-page) -- if Surya breaks partway through a
#multi-page document, the whole document's scanned pages go to
#tesseract rather than trying to reconcile a half-Surya, half-tesseract
#result.
#
#── Model caching ──
#Your original _load_surya_models() reloaded all 4 models on every call
#to read_with_surya_pages(). Across a batch of 60+ resumes that's 60+
#model loads. Predictors are now cached at module level after first load.
#"""
#
#import logging
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF
#
#logger = logging.getLogger(__name__)
#
## ── Surya availability check + predictor cache (populated lazily, once) ──
#_surya_predictors = None   # (det_predictor, rec_predictor) once loaded
#_surya_import_failed = False
#_surya_warned = False
#
#
#def _warn_once(message: str):
#    global _surya_warned
#    if not _surya_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _surya_warned = True
#
#
#def _load_surya_predictors():
#    """Loads and caches Surya's detection + recognition predictors.
#    Raises on any failure -- callers must catch and fall back to
#    pytesseract."""
#    global _surya_predictors
#    if _surya_predictors is not None:
#        return _surya_predictors
#
#    # surya-ocr==0.14.7 API: predictor classes, not standalone
#    # load_model()/load_processor() functions -- see module docstring.
#    from surya.detection import DetectionPredictor
#    from surya.recognition import RecognitionPredictor
#
#    print("[INFO] Loading Surya OCR predictors (first call only; cached after)...")
#    det_predictor = DetectionPredictor()
#    rec_predictor = RecognitionPredictor()
#
#    _surya_predictors = (det_predictor, rec_predictor)
#    return _surya_predictors
#
#
#def _run_surya_on_images(images: list) -> list:
#    """Raises on any failure -- callers must catch and fall back."""
#    det_predictor, rec_predictor = _load_surya_predictors()
#
#    # 0.14.7 dropped the explicit `langs` parameter entirely -- passing
#    # it raises TypeError. Detection is supplied via det_predictor, and
#    # recognition runs language-agnostic OCR by default.
#    results = rec_predictor(images, det_predictor=det_predictor)
#
#    all_text = []
#    for page_result in results:
#        page_lines = [line.text for line in page_result.text_lines if line.confidence > 0.3]
#        all_text.append("\n".join(page_lines))
#    return all_text
#
#
#def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
#    """Converts pages of a PDF into PIL Images at 200 DPI (good balance
#    of Surya accuracy vs. speed). page_indices=None converts every page."""
#    doc = fitz.open(pdf_path)
#    images = []
#    indices = page_indices if page_indices is not None else range(len(doc))
#    for page_num in indices:
#        page = doc[page_num]
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        images.append(img)
#    doc.close()
#    return images
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    OCRs the given 0-based page indices, Surya first, tesseract fallback
#    on any failure. Returns {page_index: text}. This is the function
#    pdf_reader.py calls for pages classified "scanned".
#    """
#    if not page_indices:
#        return {}
#
#    global _surya_import_failed
#    if not _surya_import_failed:
#        try:
#            images = pdf_to_images(pdf_path, page_indices=page_indices)
#            texts = _run_surya_on_images(images)
#            return dict(zip(page_indices, texts))
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract for this and all subsequent pages. "
#                f"Check `pip show surya-ocr` is exactly 0.6.13, and that model "
#                f"downloads aren't being blocked by network/firewall."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages as tesseract_fallback
#    return tesseract_fallback(pdf_path, page_indices)
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Whole-file OCR for a standalone image or a fully-scanned PDF,
#    joined into one string. Surya first, tesseract fallback on failure.
#    """
#    global _surya_import_failed
#    path = Path(file_path)
#
#    if not _surya_import_failed:
#        try:
#            if path.suffix.lower() == ".pdf":
#                images = pdf_to_images(file_path)
#            elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#                images = [Image.open(file_path).convert("RGB")]
#            else:
#                raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#            all_text = _run_surya_on_images(images)
#            return "\n\n".join(all_text)
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya as tesseract_fallback
#    return tesseract_fallback(file_path)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_with_surya(sys.argv[1]))
#







#Not worked
#"""
#ocr_reader.py — Layer 0, full-page OCR (scanned pages)
#
#Surya is the PRIMARY OCR engine. pytesseract is an automatic FALLBACK,
#used only if Surya can't be imported, can't load its models, or throws
#during inference. You don't need to choose one manually -- this module
#tries Surya first every time and only drops to tesseract if that fails.
#
#REQUIRES: pip install surya-ocr==0.6.13
#(exact pin -- see note below on why this version specifically, and why
#"just pip install surya-ocr" silently gives you a broken import today.)
#
#── Two separate bugs this version fixes vs. your original ocr_reader.py ──
#
#1. WRONG IMPORT PATH (present even in 0.6.13 itself, not a version-drift
#   issue). Your original code had:
#
#       from surya.model.detection.processor import load_processor as load_det_processor
#
#   Verified directly against the actual 0.6.13 package contents:
#   `load_processor` for the DETECTION model lives in
#   `surya.model.detection.model`, not `.processor` --
#   `detection/processor.py` only contains an internal
#   `SegformerImageProcessor` class, no `load_processor` function at
#   all. (Recognition's split is fine as originally written --
#   `recognition/model.py` has `load_model`, `recognition/processor.py`
#   has `load_processor` -- it's specifically the detection side that's
#   organized differently.) Corrected below:
#
#       from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
#
#2. VERSION DRIFT. Current PyPI surya-ocr (0.22.x as of writing) removed
#   `surya.ocr.run_ocr` and the `surya.model.*` module layout entirely,
#   replacing it with a VLM-based RecognitionPredictor/DetectionPredictor
#   setup that expects a running model server. A bare `pip install
#   surya-ocr` gives you that, and none of this file's imports would
#   resolve. Pin the version:
#
#       pip install surya-ocr==0.6.13
#
#── Fallback behaviour ──
#If Surya's imports fail, its models fail to download/load, or run_ocr()
#throws for any reason, this module logs a warning once and falls back
#to ocr_reader_pytesseract.py for that call. The fallback is all-or-
#nothing per batch (not per-page) -- if Surya breaks partway through a
#multi-page document, the whole document's scanned pages go to
#tesseract rather than trying to reconcile a half-Surya, half-tesseract
#result.
#
#── Model caching ──
#Your original _load_surya_models() reloaded all 4 models on every call
#to read_with_surya_pages(). Across a batch of 60+ resumes that's 60+
#model loads. Models are now cached at module level after first load.
#"""
#
#import logging
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF
#
#logger = logging.getLogger(__name__)
#
## ── Surya availability check + model cache (populated lazily, once) ──
#_surya_models = None       # (det_model, det_processor, rec_model, rec_processor) once loaded
#_surya_import_failed = False
#_surya_warned = False
#
#
#def _warn_once(message: str):
#    global _surya_warned
#    if not _surya_warned:
#        logger.warning(message)
#        print(f"[WARNING] {message}")
#        _surya_warned = True
#
#
#def _load_surya_models():
#    """Loads and caches Surya's 4 models. Raises on any failure --
#    callers must catch and fall back to pytesseract."""
#    global _surya_models
#    if _surya_models is not None:
#        return _surya_models
#
#    # Corrected import: detection's load_model AND load_processor both
#    # live in surya.model.detection.model -- see module docstring, fix (1).
#    from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
#    from surya.model.recognition.model import load_model as load_rec_model
#    from surya.model.recognition.processor import load_processor as load_rec_processor
#
#    print("[INFO] Loading Surya OCR models (first call only; cached after)...")
#    det_model = load_det_model()
#    det_processor = load_det_processor()
#    rec_model = load_rec_model()
#    rec_processor = load_rec_processor()
#
#    _surya_models = (det_model, det_processor, rec_model, rec_processor)
#    return _surya_models
#
#
#def _run_surya_on_images(images: list) -> list:
#    """Raises on any failure -- callers must catch and fall back."""
#    from surya.ocr import run_ocr
#
#    det_model, det_processor, rec_model, rec_processor = _load_surya_models()
#
#    results = run_ocr(
#        images=images,
#        langs=[["en"]] * len(images),
#        det_model=det_model,
#        det_processor=det_processor,
#        rec_model=rec_model,
#        rec_processor=rec_processor,
#    )
#
#    all_text = []
#    for page_result in results:
#        page_lines = [line.text for line in page_result.text_lines if line.confidence > 0.3]
#        all_text.append("\n".join(page_lines))
#    return all_text
#
#
#def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
#    """Converts pages of a PDF into PIL Images at 200 DPI (good balance
#    of Surya accuracy vs. speed). page_indices=None converts every page."""
#    doc = fitz.open(pdf_path)
#    images = []
#    indices = page_indices if page_indices is not None else range(len(doc))
#    for page_num in indices:
#        page = doc[page_num]
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        images.append(img)
#    doc.close()
#    return images
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    OCRs the given 0-based page indices, Surya first, tesseract fallback
#    on any failure. Returns {page_index: text}. This is the function
#    pdf_reader.py calls for pages classified "scanned".
#    """
#    if not page_indices:
#        return {}
#
#    global _surya_import_failed
#    if not _surya_import_failed:
#        try:
#            images = pdf_to_images(pdf_path, page_indices=page_indices)
#            texts = _run_surya_on_images(images)
#            return dict(zip(page_indices, texts))
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract for this and all subsequent pages. "
#                f"Check `pip show surya-ocr` is exactly 0.6.13, and that model "
#                f"downloads aren't being blocked by network/firewall."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya_pages as tesseract_fallback
#    return tesseract_fallback(pdf_path, page_indices)
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Whole-file OCR for a standalone image or a fully-scanned PDF,
#    joined into one string. Surya first, tesseract fallback on failure.
#    """
#    global _surya_import_failed
#    path = Path(file_path)
#
#    if not _surya_import_failed:
#        try:
#            if path.suffix.lower() == ".pdf":
#                images = pdf_to_images(file_path)
#            elif path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
#                images = [Image.open(file_path).convert("RGB")]
#            else:
#                raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#            all_text = _run_surya_on_images(images)
#            return "\n\n".join(all_text)
#        except Exception as e:
#            _surya_import_failed = True
#            _warn_once(
#                f"Surya OCR failed ({type(e).__name__}: {e}). "
#                f"Falling back to pytesseract."
#            )
#
#    from ingestion.ocr_reader_pytesseract import read_with_surya as tesseract_fallback
#    return tesseract_fallback(file_path)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_with_surya(sys.argv[1]))
#










# commenting just to work this when adding scanned pdfs code to work.
#"""
#OCR Reader using Surya — Layer 0
#Handles scanned PDFs and image files (JPG, PNG, TIFF).
#Surya is a modern document OCR model that handles complex layouts.
#
#CHANGES (this revision):
#  read_with_surya() used to build a per-page list internally
#  (`all_text`) and then immediately join it into one string before
#  returning -- so callers could never recover per-page boundaries.
#
#  This mattered less when scanned/not-scanned was a whole-document
#  decision (the old is_scanned_pdf() check): if a doc was scanned, you
#  OCR'd the whole thing and joining pages was fine. Now that
#  pdf_reader.py classifies PAGES individually (a doc can have some
#  scanned pages and some real-text pages mixed together), the caller
#  needs to know which OCR'd string belongs to which page, so it can be
#  merged into the right position in reading order alongside the
#  pdfplumber-extracted pages.
#
#  Two additions, both backward compatible -- read_with_surya() keeps
#  its old signature and behaviour:
#    - pdf_to_images() gains an optional `page_indices` param, so only
#      specific pages get rendered (avoids converting every page when
#      only e.g. page 0 of a 2-page doc is scanned).
#    - New read_with_surya_pages(file_path, page_indices) -> dict[int, str]
#      loads the models ONCE and runs OCR ONCE across just the requested
#      pages (batched), returning a {page_index: text} mapping. This is
#      what pdf_reader.py's per-page loop should call for the "scanned"
#      pages it collects, instead of calling read_with_surya() in a loop
#      (which would reload all 4 models on every single page).
#"""
#
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF — used to convert PDF pages to images
#
#
#def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
#    """
#    Converts pages of a PDF into PIL Image objects.
#    Surya works on images, not PDFs directly.
#    Resolution of 200 DPI is a good balance of speed and accuracy.
#
#    page_indices: optional list of specific 0-based page numbers to
#      convert. If None (default, unchanged behaviour), converts every
#      page in the document.
#    """
#    doc = fitz.open(pdf_path)
#    images = []
#
#    indices = page_indices if page_indices is not None else range(len(doc))
#
#    for page_num in indices:
#        page = doc[page_num]
#        # mat = zoom matrix. 200/72 ≈ 2.78x zoom for 200 DPI
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        images.append(img)
#
#    doc.close()
#    return images
#
#
#def _load_surya_models():
#    from surya.model.detection.model import load_model as load_det_model
#    from surya.model.detection.processor import load_processor as load_det_processor
#    from surya.model.recognition.model import load_model as load_rec_model
#    from surya.model.recognition.processor import load_processor as load_rec_processor
#
#    print("[INFO] Loading Surya OCR models (first run may take ~30 seconds to download)...")
#    det_processor = load_det_processor()
#    det_model = load_det_model()
#    rec_model = load_rec_model()
#    rec_processor = load_rec_processor()
#    return det_model, det_processor, rec_model, rec_processor
#
#
#def _run_surya_on_images(images: list) -> list:
#    """
#    Shared OCR core: loads models once, runs OCR once across all given
#    images, returns a list of per-page text strings (same order as the
#    input images list). Both read_with_surya() and
#    read_with_surya_pages() build on this so model-loading only ever
#    happens once per call, regardless of how many pages are involved.
#    """
#    from surya.ocr import run_ocr
#
#    det_model, det_processor, rec_model, rec_processor = _load_surya_models()
#
#    results = run_ocr(
#        images=images,
#        langs=[["en"]] * len(images),
#        det_model=det_model,
#        det_processor=det_processor,
#        rec_model=rec_model,
#        rec_processor=rec_processor,
#    )
#
#    all_text = []
#    for page_result in results:
#        page_lines = []
#        for line in page_result.text_lines:
#            if line.confidence > 0.3:
#                page_lines.append(line.text)
#        all_text.append("\n".join(page_lines))
#
#    return all_text
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Reads a scanned PDF or image file using Surya OCR.
#    Unchanged behaviour: whole-file OCR, pages joined into one string.
#    Kept for callers that still want a single flat string (e.g. a
#    standalone image file, or a PDF that's scanned front-to-back).
#    """
#    path = Path(file_path)
#
#    if path.suffix.lower() == ".pdf":
#        images = pdf_to_images(file_path)
#    elif path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]:
#        images = [Image.open(file_path).convert("RGB")]
#    else:
#        raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#
#    all_text = _run_surya_on_images(images)
#    return "\n\n".join(all_text)
#
#
#def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
#    """
#    NEW: OCRs only the specified pages of a PDF, loading Surya's models
#    just once for the whole batch, and returns a {page_index: text}
#    mapping so the caller (pdf_reader.py) can merge each page's OCR'd
#    text into the correct position among the document's other
#    (non-scanned) pages.
#
#    This is the function to use when a document mixes scanned pages
#    with real-text pages -- calling read_with_surya() in a per-page
#    loop instead would reload all 4 Surya models on every page.
#    """
#    if not page_indices:
#        return {}
#
#    images = pdf_to_images(pdf_path, page_indices=page_indices)
#    texts = _run_surya_on_images(images)
#
#    return dict(zip(page_indices, texts))
#






#worked one - changing for new resumes to work
#"""
#OCR Reader using Surya — Layer 0
#Handles scanned PDFs and image files (JPG, PNG, TIFF).
#Surya is a modern document OCR model that handles complex layouts.
#"""
#
#from pathlib import Path
#from PIL import Image
#import fitz  # PyMuPDF — used to convert PDF pages to images
#
#
#def pdf_to_images(pdf_path: str) -> list:
#    """
#    Converts each page of a PDF into a PIL Image object.
#    Surya works on images, not PDFs directly.
#    Resolution of 200 DPI is a good balance of speed and accuracy.
#    """
#    doc = fitz.open(pdf_path)
#    images = []
#
#    for page_num in range(len(doc)):
#        page = doc[page_num]
#        # mat = zoom matrix. 200/72 ≈ 2.78x zoom for 200 DPI
#        mat = fitz.Matrix(200 / 72, 200 / 72)
#        pix = page.get_pixmap(matrix=mat)
#        # Convert to PIL Image
#        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#        images.append(img)
#
#    doc.close()
#    return images
#
#
#def read_with_surya(file_path: str) -> str:
#    """
#    Reads a scanned PDF or image file using Surya OCR.
#
#    Surya automatically:
#    - Detects text regions (layout analysis)
#    - Handles multi-column layouts
#    - Reads text in correct reading order
#    - Works in multiple languages
#    """
#    from surya.ocr import run_ocr
#    from surya.model.detection.model import load_model as load_det_model
#    from surya.model.detection.processor import load_processor as load_det_processor
#    from surya.model.recognition.model import load_model as load_rec_model
#    from surya.model.recognition.processor import load_processor as load_rec_processor
#
#    path = Path(file_path)
#
#    if path.suffix.lower() == ".pdf":
#        images = pdf_to_images(file_path)
#    elif path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]:
#        images = [Image.open(file_path).convert("RGB")]
#    else:
#        raise ValueError(f"Unsupported file type for OCR: {path.suffix}")
#
#    print(f"[INFO] Loading Surya OCR models (first run may take ~30 seconds to download)...")
#
#    det_processor = load_det_processor()
#    det_model     = load_det_model()
#    rec_model     = load_rec_model()
#    rec_processor = load_rec_processor()
#
#    results = run_ocr(
#        images=images,
#        langs=[["en"]] * len(images),
#        det_model=det_model,
#        det_processor=det_processor,
#        rec_model=rec_model,
#        rec_processor=rec_processor,
#    )
#
#    all_text = []
#    for page_result in results:
#        page_lines = []
#        for line in page_result.text_lines:
#            if line.confidence > 0.3:
#                page_lines.append(line.text)
#        all_text.append("\n".join(page_lines))
#
#    return "\n\n".join(all_text)
#