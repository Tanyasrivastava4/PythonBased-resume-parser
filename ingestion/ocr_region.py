"""
OCR-region helper for Layer 0 (pdf_reader.py).

Handles the "hybrid page" case: a page that has genuine extractable text
AND one or more embedded images that themselves contain text (contact
bars, icon+text rows, header graphics). This is distinct from a fully
scanned page (handled separately via full-page OCR / Surya).

Public functions:
  classify_page(pdfplumber_page, fitz_page) -> "scanned" | "hybrid" | "text"
  get_ocr_worthy_image_blocks(fitz_page) -> list[fitz.Rect]
  ocr_region_to_words(fitz_page, bbox, zoom=4, min_conf=30, psm=6) -> list[word-dict]

Word-dicts returned by ocr_region_to_words() match pdfplumber's
extract_words() shape: {"text", "x0", "x1", "top", "bottom", "height"}
so they can be concatenated directly into a pdfplumber words list and
flow through the existing sidebar/two-column/single-column logic
unchanged.

REVISION (this version) — ocr_region_to_words() now accepts an optional
psm override.
--------------------------------------------------------------------------
This function was originally built for ONE specific region shape: a
single-line contact bar (name/email/phone all on roughly one visual
line), for which PSM 6 ("assume a single uniform block of text") is the
right tesseract setting. pdf_reader.py's header-gap detection later
reused this same function for a DIFFERENT region shape -- a multi-line
header block with a large name, a smaller subtitle, and a small contact
row, three very different text sizes stacked vertically -- which needs
PSM 4 ("assume a single column of text of variable sizes") instead.

Before this revision, OCR_PSM was hardcoded inside this function with no
way to override it, so pdf_reader.py's header-gap call site
(ocr_region_to_words(fpage, header_gap, psm=4)) would raise
"TypeError: unexpected keyword argument 'psm'" the moment it ran on a
real header gap. Fixed by adding a psm parameter that defaults to the
original OCR_PSM (6) -- every existing call site that doesn't pass psm
keeps its exact original behavior; only callers that explicitly need a
different PSM (like the header-gap case) need to pass one.
"""

import fitz
import pytesseract
from PIL import Image, ImageOps


# ── thresholds (tune against your resume corpus) ──
MIN_PAGE_CHARS_FOR_TEXT = 20        # below this -> treat page as "scanned"
MIN_IMAGE_WIDTH_PT = 25             # skip narrower images (bullet icons, single glyphs)
MIN_IMAGE_HEIGHT_PT = 6             # skip thinner images (decorative divider rules)
MAX_IMAGE_PAGE_AREA_RATIO = 0.85    # image covering nearly the whole page -> not a "region", treat page as scanned
OCR_MIN_CONFIDENCE = 30             # discard low-confidence OCR tokens (usually icon glyphs, not text)
OCR_MIN_ALNUM_CHARS = 2             # discard tokens with <2 letters/digits (icon misreads like "(A)", "H_", "©")
OCR_ZOOM = 6                        # render scale for OCR accuracy (6x needed to get email text past conf=0)
OCR_PSM = 6                         # default tesseract page-segmentation mode: "uniform block of
                                     # text" -- right for the original single-line contact-bar case.
                                     # Callers can override per-call (see ocr_region_to_words below)
                                     # for differently-shaped regions, e.g. pdf_reader.py's header-gap
                                     # detection passes psm=4 for a multi-line, variable-size header.


def _page_char_count(fitz_page) -> int:
    return len(fitz_page.get_text().strip())


def classify_page(fitz_page, native_words: list = None) -> str:
    """
    Per-page classification (this is the key fix vs. doc-level is_scanned_pdf):
      "scanned" -> essentially no extractable text; caller should run full-page OCR
      "hybrid"  -> has real text AND has candidate image blocks worth OCR'ing
                   (i.e. image regions with NO native text already inside them --
                   see get_ocr_worthy_image_blocks() for why this check matters)
      "text"    -> normal text page, no OCR needed

    native_words: optional list of pdfplumber word dicts already extracted for
      this page. Passing this lets classify_page correctly return "text"
      instead of "hybrid" for pages whose only images are decorative
      backgrounds (e.g. colored bars behind section headings) that already
      have real text drawn on top of them natively -- OCR'ing those would
      just re-read text that's already been extracted, producing duplicates.
      If omitted, decorative-image pages will be misclassified as "hybrid".
    """
    char_count = _page_char_count(fitz_page)
    if char_count < MIN_PAGE_CHARS_FOR_TEXT:
        return "scanned"

    if get_ocr_worthy_image_blocks(fitz_page, native_words=native_words):
        return "hybrid"

    return "text"


def _block_already_has_text(bbox, native_words: list) -> bool:
    """
    Checks whether native (already-extracted) words exist inside the given
    image bbox. If they do, the image is almost certainly a decorative
    background (a colored bar/shape behind a heading, a shaded panel, etc.)
    with real selectable text drawn on top of it -- NOT a flattened image
    containing text that only OCR can recover.

    Confirmed on a real resume (Abhishek Kumar): thin full-width colored
    bars behind each section heading ("PROFESSIONAL SNAPSHOT", "CORE
    COMPETENCIES", etc.) passed the width/height/area filters below and
    looked exactly like Adarsh Bhagat's genuinely-image-only contact bar.
    But unlike that case, pdfplumber's native extract_words() already had
    "PROFESSIONAL", "SNAPSHOT" etc. sitting right inside that bbox --
    rendering + OCR'ing the region re-read the same visible pixels and
    produced a duplicate ("PROFESSIONAL PROFESSIONAL SNAPSHOT SNAPSHOT")
    alongside the correct native extraction. Skipping OCR whenever native
    words already cover a block avoids this without needing to guess at
    "is this decorative" from the image's visual properties.

    A word counts as "inside" the bbox if its center point falls within
    it -- using the center (not full containment) avoids edge cases where
    a word straddles the block's border by a point or two.
    """
    if not native_words:
        return False
    for w in native_words:
        cx = (w["x0"] + w["x1"]) / 2
        cy = (w["top"] + w["bottom"]) / 2
        if bbox.x0 <= cx <= bbox.x1 and bbox.y0 <= cy <= bbox.y1:
            return True
    return False


def get_ocr_worthy_image_blocks(fitz_page, native_words: list = None) -> list:
    """
    Finds embedded image blocks on the page likely to contain real text
    NOT already captured by native extraction (contact bars, icon+label
    rows) as opposed to:
      - decorative elements (thin divider rules, tiny bullet icons,
        background graphics)
      - a single full-page background image (handled as "scanned" instead)
      - decorative colored bars/panels that have real, already-extracted
        text drawn on top of them (see _block_already_has_text() --
        OCR'ing these would duplicate content that's already correct)

    native_words: optional list of pdfplumber word dicts for this page.
      Strongly recommended -- without it, decorative-bar-behind-heading
      layouts will get incorrectly OCR'd and duplicated (confirmed on a
      real resume, see _block_already_has_text() docstring).

    Returns a list of fitz.Rect bboxes, in PDF point coordinates.
    """
    page_rect = fitz_page.rect
    page_area = page_rect.width * page_rect.height

    blocks = fitz_page.get_text("dict")["blocks"]
    candidates = []

    for block in blocks:
        if block.get("type") != 1:  # 1 = image block in fitz
            continue
        x0, top, x1, bottom = block["bbox"]
        width = x1 - x0
        height = bottom - top
        if width < MIN_IMAGE_WIDTH_PT or height < MIN_IMAGE_HEIGHT_PT:
            continue  # decorative rule line / tiny icon
        area_ratio = (width * height) / page_area
        if area_ratio > MAX_IMAGE_PAGE_AREA_RATIO:
            continue  # near full-page image -> not a "region", page should be classified scanned

        bbox = fitz.Rect(x0, top, x1, bottom)
        if _block_already_has_text(bbox, native_words):
            continue  # decorative background with real text already on top of it -- skip, don't duplicate

        candidates.append(bbox)

    return candidates


def ocr_region_to_words(fitz_page, bbox, zoom: float = OCR_ZOOM,
                         min_conf: int = OCR_MIN_CONFIDENCE,
                         psm: int = OCR_PSM) -> list:
    """
    Renders the given bbox region at high resolution, runs word-level OCR,
    and maps each recognised word's bounding box back into PDF point
    coordinates (same coordinate space pdfplumber's extract_words() uses).

    Filters out low-confidence tokens and single-character tokens, since
    these are almost always icon glyphs (phone/envelope/LinkedIn icons
    etc.) misread as garbage characters, not real text.

    psm: tesseract page-segmentation mode override, defaults to OCR_PSM
      (6 -- "uniform block of text", right for the original single-line
      contact-bar use case). Pass a different value for differently-
      shaped regions -- e.g. pdf_reader.py's header-gap detection passes
      psm=4 ("single column, variable text sizes") for a multi-line
      header block containing a large name + smaller subtitle + small
      contact row, three very different text sizes stacked vertically,
      which PSM 6 does not handle well.
    """
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bbox)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = ImageOps.grayscale(img)  # notably improves confidence on small embedded-image text

    data = pytesseract.image_to_data(img, config=f"--psm {psm}",
                                      output_type=pytesseract.Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        alnum_count = sum(1 for ch in text if ch.isalnum())
        if alnum_count < OCR_MIN_ALNUM_CHARS:
            continue  # icon glyphs misread as e.g. "(A)", "H_", "©" -- almost never real content
        try:
            conf = int(float(data["conf"][i]))
        except (ValueError, TypeError):
            conf = -1
        if conf < min_conf:
            continue

        x, y, w, h = (data["left"][i], data["top"][i],
                       data["width"][i], data["height"][i])

        words.append({
            "text": text,
            "x0": bbox.x0 + x / zoom,
            "x1": bbox.x0 + (x + w) / zoom,
            "top": bbox.y0 + y / zoom,
            "bottom": bbox.y0 + (y + h) / zoom,
            "height": h / zoom,
            "source": "ocr",  # tag so downstream layers can flag low-trust fields if needed
        })

    return words








#pasted the updated code above after coming from home
#"""
#OCR-region helper for Layer 0 (pdf_reader.py).
#
#Handles the "hybrid page" case: a page that has genuine extractable text
#AND one or more embedded images that themselves contain text (contact
#bars, icon+text rows, header graphics). This is distinct from a fully
#scanned page (handled separately via full-page OCR / Surya).
#
#Public functions:
#  classify_page(pdfplumber_page, fitz_page) -> "scanned" | "hybrid" | "text"
#  get_ocr_worthy_image_blocks(fitz_page) -> list[fitz.Rect]
#  ocr_region_to_words(fitz_page, bbox, zoom=4, min_conf=30) -> list[word-dict]
#
#Word-dicts returned by ocr_region_to_words() match pdfplumber's
#extract_words() shape: {"text", "x0", "x1", "top", "bottom", "height"}
#so they can be concatenated directly into a pdfplumber words list and
#flow through the existing sidebar/two-column/single-column logic
#unchanged.
#"""
#
#import fitz
import pytesseract
from PIL import Image, ImageOps
#
#
## ── thresholds (tune against your resume corpus) ──
#MIN_PAGE_CHARS_FOR_TEXT = 20        # below this -> treat page as "scanned"
#MIN_IMAGE_WIDTH_PT = 25             # skip narrower images (bullet icons, single glyphs)
#MIN_IMAGE_HEIGHT_PT = 6             # skip thinner images (decorative divider rules)
#MAX_IMAGE_PAGE_AREA_RATIO = 0.85    # image covering nearly the whole page -> not a "region", treat page as scanned
#OCR_MIN_CONFIDENCE = 30             # discard low-confidence OCR tokens (usually icon glyphs, not text)
#OCR_MIN_ALNUM_CHARS = 2             # discard tokens with <2 letters/digits (icon misreads like "(A)", "H_", "©")
#OCR_ZOOM = 6                        # render scale for OCR accuracy (6x needed to get email text past conf=0)
#OCR_PSM = 6                         # tesseract page-segmentation mode: "uniform block of text" -- best for single-line contact bars
#
#
#def _page_char_count(fitz_page) -> int:
#    return len(fitz_page.get_text().strip())
#
#
#def classify_page(fitz_page, native_words: list = None) -> str:
#    """
#    Per-page classification (this is the key fix vs. doc-level is_scanned_pdf):
#      "scanned" -> essentially no extractable text; caller should run full-page OCR
#      "hybrid"  -> has real text AND has candidate image blocks worth OCR'ing
#                   (i.e. image regions with NO native text already inside them --
#                   see get_ocr_worthy_image_blocks() for why this check matters)
#      "text"    -> normal text page, no OCR needed
#
#    native_words: optional list of pdfplumber word dicts already extracted for
#      this page. Passing this lets classify_page correctly return "text"
#      instead of "hybrid" for pages whose only images are decorative
#      backgrounds (e.g. colored bars behind section headings) that already
#      have real text drawn on top of them natively -- OCR'ing those would
#      just re-read text that's already been extracted, producing duplicates.
#      If omitted, decorative-image pages will be misclassified as "hybrid".
#    """
#    char_count = _page_char_count(fitz_page)
#    if char_count < MIN_PAGE_CHARS_FOR_TEXT:
#        return "scanned"
#
#    if get_ocr_worthy_image_blocks(fitz_page, native_words=native_words):
#        return "hybrid"
#
#    return "text"
#
#
#def _block_already_has_text(bbox, native_words: list) -> bool:
#    """
#    Checks whether native (already-extracted) words exist inside the given
#    image bbox. If they do, the image is almost certainly a decorative
#    background (a colored bar/shape behind a heading, a shaded panel, etc.)
#    with real selectable text drawn on top of it -- NOT a flattened image
#    containing text that only OCR can recover.
#
#    Confirmed on a real resume (Abhishek Kumar): thin full-width colored
#    bars behind each section heading ("PROFESSIONAL SNAPSHOT", "CORE
#    COMPETENCIES", etc.) passed the width/height/area filters below and
#    looked exactly like Adarsh Bhagat's genuinely-image-only contact bar.
#    But unlike that case, pdfplumber's native extract_words() already had
#    "PROFESSIONAL", "SNAPSHOT" etc. sitting right inside that bbox --
#    rendering + OCR'ing the region re-read the same visible pixels and
#    produced a duplicate ("PROFESSIONAL PROFESSIONAL SNAPSHOT SNAPSHOT")
#    alongside the correct native extraction. Skipping OCR whenever native
#    words already cover a block avoids this without needing to guess at
#    "is this decorative" from the image's visual properties.
#
#    A word counts as "inside" the bbox if its center point falls within
#    it -- using the center (not full containment) avoids edge cases where
#    a word straddles the block's border by a point or two.
#    """
#    if not native_words:
#        return False
#    for w in native_words:
#        cx = (w["x0"] + w["x1"]) / 2
#        cy = (w["top"] + w["bottom"]) / 2
#        if bbox.x0 <= cx <= bbox.x1 and bbox.y0 <= cy <= bbox.y1:
#            return True
#    return False
#
#
#def get_ocr_worthy_image_blocks(fitz_page, native_words: list = None) -> list:
#    """
#    Finds embedded image blocks on the page likely to contain real text
#    NOT already captured by native extraction (contact bars, icon+label
#    rows) as opposed to:
#      - decorative elements (thin divider rules, tiny bullet icons,
#        background graphics)
#      - a single full-page background image (handled as "scanned" instead)
#      - decorative colored bars/panels that have real, already-extracted
#        text drawn on top of them (see _block_already_has_text() --
#        OCR'ing these would duplicate content that's already correct)
#
#    native_words: optional list of pdfplumber word dicts for this page.
#      Strongly recommended -- without it, decorative-bar-behind-heading
#      layouts will get incorrectly OCR'd and duplicated (confirmed on a
#      real resume, see _block_already_has_text() docstring).
#
#    Returns a list of fitz.Rect bboxes, in PDF point coordinates.
#    """
#    page_rect = fitz_page.rect
#    page_area = page_rect.width * page_rect.height
#
#    blocks = fitz_page.get_text("dict")["blocks"]
#    candidates = []
#
#    for block in blocks:
#        if block.get("type") != 1:  # 1 = image block in fitz
#            continue
#        x0, top, x1, bottom = block["bbox"]
#        width = x1 - x0
#        height = bottom - top
#        if width < MIN_IMAGE_WIDTH_PT or height < MIN_IMAGE_HEIGHT_PT:
#            continue  # decorative rule line / tiny icon
#        area_ratio = (width * height) / page_area
#        if area_ratio > MAX_IMAGE_PAGE_AREA_RATIO:
#            continue  # near full-page image -> not a "region", page should be classified scanned
#
#        bbox = fitz.Rect(x0, top, x1, bottom)
#        if _block_already_has_text(bbox, native_words):
#            continue  # decorative background with real text already on top of it -- skip, don't duplicate
#
#        candidates.append(bbox)
#
#    return candidates
#
#
#def ocr_region_to_words(fitz_page, bbox, zoom: float = OCR_ZOOM,
#                         min_conf: int = OCR_MIN_CONFIDENCE) -> list:
#    """
#    Renders the given bbox region at high resolution, runs word-level OCR,
#    and maps each recognised word's bounding box back into PDF point
#    coordinates (same coordinate space pdfplumber's extract_words() uses).
#
#    Filters out low-confidence tokens and single-character tokens, since
#    these are almost always icon glyphs (phone/envelope/LinkedIn icons
#    etc.) misread as garbage characters, not real text.
#    """
#    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bbox)
#    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
#    img = ImageOps.grayscale(img)  # notably improves confidence on small embedded-image text
#
#    data = pytesseract.image_to_data(img, config=f"--psm {OCR_PSM}",
#                                      output_type=pytesseract.Output.DICT)
#
#    words = []
#    n = len(data["text"])
#    for i in range(n):
#        text = data["text"][i].strip()
#        if not text:
#            continue
#        alnum_count = sum(1 for ch in text if ch.isalnum())
#        if alnum_count < OCR_MIN_ALNUM_CHARS:
#            continue  # icon glyphs misread as e.g. "(A)", "H_", "©" -- almost never real content
#        try:
#            conf = int(float(data["conf"][i]))
#        except (ValueError, TypeError):
#            conf = -1
#        if conf < min_conf:
#            continue
#
#        x, y, w, h = (data["left"][i], data["top"][i],
#                       data["width"][i], data["height"][i])
#
#        words.append({
#            "text": text,
#            "x0": bbox.x0 + x / zoom,
#            "x1": bbox.x0 + (x + w) / zoom,
#            "top": bbox.y0 + y / zoom,
#            "bottom": bbox.y0 + (y + h) / zoom,
#            "height": h / zoom,
#            "source": "ocr",  # tag so downstream layers can flag low-trust fields if needed
#        })
#
#    return words
#
#









#
#
#"""
#OCR-region helper for Layer 0 (pdf_reader.py).
#
#Handles the "hybrid page" case: a page that has genuine extractable text
#AND one or more embedded images that themselves contain text (contact
#bars, icon+text rows, header graphics). This is distinct from a fully
#scanned page (handled separately via full-page OCR / Surya).
#
#Public functions:
#  classify_page(pdfplumber_page, fitz_page) -> "scanned" | "hybrid" | "text"
#  get_ocr_worthy_image_blocks(fitz_page) -> list[fitz.Rect]
#  ocr_region_to_words(fitz_page, bbox, zoom=4, min_conf=30) -> list[word-dict]
#
#Word-dicts returned by ocr_region_to_words() match pdfplumber's
#extract_words() shape: {"text", "x0", "x1", "top", "bottom", "height"}
#so they can be concatenated directly into a pdfplumber words list and
#flow through the existing sidebar/two-column/single-column logic
#unchanged.
#"""
#
#import fitz
#import pytesseract
#from PIL import Image, ImageOps
#
#
## ── thresholds (tune against your resume corpus) ──
#MIN_PAGE_CHARS_FOR_TEXT = 20        # below this -> treat page as "scanned"
#MIN_IMAGE_WIDTH_PT = 25             # skip narrower images (bullet icons, single glyphs)
#MIN_IMAGE_HEIGHT_PT = 6             # skip thinner images (decorative divider rules)
#MAX_IMAGE_PAGE_AREA_RATIO = 0.85    # image covering nearly the whole page -> not a "region", treat page as scanned
#OCR_MIN_CONFIDENCE = 30             # discard low-confidence OCR tokens (usually icon glyphs, not text)
#OCR_MIN_ALNUM_CHARS = 2             # discard tokens with <2 letters/digits (icon misreads like "(A)", "H_", "©")
#OCR_ZOOM = 6                        # render scale for OCR accuracy (6x needed to get email text past conf=0)
#OCR_PSM = 6                         # tesseract page-segmentation mode: "uniform block of text" -- best for single-line contact bars
#
#
#def _page_char_count(fitz_page) -> int:
#    return len(fitz_page.get_text().strip())
#
#
#def classify_page(fitz_page) -> str:
#    """
#    Per-page classification (this is the key fix vs. doc-level is_scanned_pdf):
#      "scanned" -> essentially no extractable text; caller should run full-page OCR
#      "hybrid"  -> has real text AND has candidate image blocks worth OCR'ing
#      "text"    -> normal text page, no OCR needed
#    """
#    char_count = _page_char_count(fitz_page)
#    if char_count < MIN_PAGE_CHARS_FOR_TEXT:
#        return "scanned"
#
#    if get_ocr_worthy_image_blocks(fitz_page):
#        return "hybrid"
#
#    return "text"
#
#
#def get_ocr_worthy_image_blocks(fitz_page) -> list:
#    """
#    Finds embedded image blocks on the page likely to contain real text
#    (contact bars, icon+label rows) as opposed to decorative elements
#    (thin divider rules, tiny bullet icons, background graphics) or a
#    single full-page background image (which should be handled as a
#    "scanned" page, not region-injected).
#
#    Returns a list of fitz.Rect bboxes, in PDF point coordinates.
#    """
#    page_rect = fitz_page.rect
#    page_area = page_rect.width * page_rect.height
#
#    blocks = fitz_page.get_text("dict")["blocks"]
#    candidates = []
#
#    for block in blocks:
#        if block.get("type") != 1:  # 1 = image block in fitz
#            continue
#        x0, top, x1, bottom = block["bbox"]
#        width = x1 - x0
#        height = bottom - top
#        if width < MIN_IMAGE_WIDTH_PT or height < MIN_IMAGE_HEIGHT_PT:
#            continue  # decorative rule line / tiny icon
#        area_ratio = (width * height) / page_area
#        if area_ratio > MAX_IMAGE_PAGE_AREA_RATIO:
#            continue  # near full-page image -> not a "region", page should be classified scanned
#        candidates.append(fitz.Rect(x0, top, x1, bottom))
#
#    return candidates
#
#
#def ocr_region_to_words(fitz_page, bbox, zoom: float = OCR_ZOOM,
#                         min_conf: int = OCR_MIN_CONFIDENCE) -> list:
#    """
#    Renders the given bbox region at high resolution, runs word-level OCR,
#    and maps each recognised word's bounding box back into PDF point
#    coordinates (same coordinate space pdfplumber's extract_words() uses).
#
#    Filters out low-confidence tokens and single-character tokens, since
#    these are almost always icon glyphs (phone/envelope/LinkedIn icons
#    etc.) misread as garbage characters, not real text.
#    """
#    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bbox)
#    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
#    img = ImageOps.grayscale(img)  # notably improves confidence on small embedded-image text
#
#    data = pytesseract.image_to_data(img, config=f"--psm {OCR_PSM}",
#                                      output_type=pytesseract.Output.DICT)
#
#    words = []
#    n = len(data["text"])
#    for i in range(n):
#        text = data["text"][i].strip()
#        if not text:
#            continue
#        alnum_count = sum(1 for ch in text if ch.isalnum())
#        if alnum_count < OCR_MIN_ALNUM_CHARS:
#            continue  # icon glyphs misread as e.g. "(A)", "H_", "©" -- almost never real content
#        try:
#            conf = int(float(data["conf"][i]))
#        except (ValueError, TypeError):
#            conf = -1
#        if conf < min_conf:
#            continue
#
#        x, y, w, h = (data["left"][i], data["top"][i],
#                       data["width"][i], data["height"][i])
#
#        words.append({
#            "text": text,
#            "x0": bbox.x0 + x / zoom,
#            "x1": bbox.x0 + (x + w) / zoom,
#            "top": bbox.y0 + y / zoom,
#            "bottom": bbox.y0 + (y + h) / zoom,
#            "height": h / zoom,
#            "source": "ocr",  # tag so downstream layers can flag low-trust fields if needed
#        })
#
#    return words