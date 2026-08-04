"""
OCR Reader using Surya — Layer 0
Handles scanned PDFs and image files (JPG, PNG, TIFF).
Surya is a modern document OCR model that handles complex layouts.

CHANGES (this revision):
  read_with_surya() used to build a per-page list internally
  (`all_text`) and then immediately join it into one string before
  returning -- so callers could never recover per-page boundaries.

  This mattered less when scanned/not-scanned was a whole-document
  decision (the old is_scanned_pdf() check): if a doc was scanned, you
  OCR'd the whole thing and joining pages was fine. Now that
  pdf_reader.py classifies PAGES individually (a doc can have some
  scanned pages and some real-text pages mixed together), the caller
  needs to know which OCR'd string belongs to which page, so it can be
  merged into the right position in reading order alongside the
  pdfplumber-extracted pages.

  Two additions, both backward compatible -- read_with_surya() keeps
  its old signature and behaviour:
    - pdf_to_images() gains an optional `page_indices` param, so only
      specific pages get rendered (avoids converting every page when
      only e.g. page 0 of a 2-page doc is scanned).
    - New read_with_surya_pages(file_path, page_indices) -> dict[int, str]
      loads the models ONCE and runs OCR ONCE across just the requested
      pages (batched), returning a {page_index: text} mapping. This is
      what pdf_reader.py's per-page loop should call for the "scanned"
      pages it collects, instead of calling read_with_surya() in a loop
      (which would reload all 4 models on every single page).
"""

from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF — used to convert PDF pages to images


def pdf_to_images(pdf_path: str, page_indices: list = None) -> list:
    """
    Converts pages of a PDF into PIL Image objects.
    Surya works on images, not PDFs directly.
    Resolution of 200 DPI is a good balance of speed and accuracy.

    page_indices: optional list of specific 0-based page numbers to
      convert. If None (default, unchanged behaviour), converts every
      page in the document.
    """
    doc = fitz.open(pdf_path)
    images = []

    indices = page_indices if page_indices is not None else range(len(doc))

    for page_num in indices:
        page = doc[page_num]
        # mat = zoom matrix. 200/72 ≈ 2.78x zoom for 200 DPI
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


def _load_surya_models():
    from surya.model.detection.model import load_model as load_det_model
    from surya.model.detection.processor import load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor

    print("[INFO] Loading Surya OCR models (first run may take ~30 seconds to download)...")
    det_processor = load_det_processor()
    det_model = load_det_model()
    rec_model = load_rec_model()
    rec_processor = load_rec_processor()
    return det_model, det_processor, rec_model, rec_processor


def _run_surya_on_images(images: list) -> list:
    """
    Shared OCR core: loads models once, runs OCR once across all given
    images, returns a list of per-page text strings (same order as the
    input images list). Both read_with_surya() and
    read_with_surya_pages() build on this so model-loading only ever
    happens once per call, regardless of how many pages are involved.
    """
    from surya.ocr import run_ocr

    det_model, det_processor, rec_model, rec_processor = _load_surya_models()

    results = run_ocr(
        images=images,
        langs=[["en"]] * len(images),
        det_model=det_model,
        det_processor=det_processor,
        rec_model=rec_model,
        rec_processor=rec_processor,
    )

    all_text = []
    for page_result in results:
        page_lines = []
        for line in page_result.text_lines:
            if line.confidence > 0.3:
                page_lines.append(line.text)
        all_text.append("\n".join(page_lines))

    return all_text


def read_with_surya(file_path: str) -> str:
    """
    Reads a scanned PDF or image file using Surya OCR.
    Unchanged behaviour: whole-file OCR, pages joined into one string.
    Kept for callers that still want a single flat string (e.g. a
    standalone image file, or a PDF that's scanned front-to-back).
    """
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        images = pdf_to_images(file_path)
    elif path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]:
        images = [Image.open(file_path).convert("RGB")]
    else:
        raise ValueError(f"Unsupported file type for OCR: {path.suffix}")

    all_text = _run_surya_on_images(images)
    return "\n\n".join(all_text)


def read_with_surya_pages(pdf_path: str, page_indices: list) -> dict:
    """
    NEW: OCRs only the specified pages of a PDF, loading Surya's models
    just once for the whole batch, and returns a {page_index: text}
    mapping so the caller (pdf_reader.py) can merge each page's OCR'd
    text into the correct position among the document's other
    (non-scanned) pages.

    This is the function to use when a document mixes scanned pages
    with real-text pages -- calling read_with_surya() in a per-page
    loop instead would reload all 4 Surya models on every page.
    """
    if not page_indices:
        return {}

    images = pdf_to_images(pdf_path, page_indices=page_indices)
    texts = _run_surya_on_images(images)

    return dict(zip(page_indices, texts))







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