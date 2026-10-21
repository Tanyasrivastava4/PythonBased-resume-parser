import fitz, pytesseract
from ingestion.ocr_reader_pytesseract import _render_page
from ingestion.image_preprocessing import preprocess_scanned_image
from ingestion.table_grid_detector import detect_and_ocr_tables

pdf = "/home/intileo/Desktop/python-based-parser/ats_parser/resumes/DOC-20241227-WA0006. (1).pdf"
doc = fitz.open(pdf)

img = _render_page(doc[0])
img.save("ocr_compare/step1_rendered.png")
pre = preprocess_scanned_image(img, context="debug")
pre.save("ocr_compare/step2_preprocessed.png")

print("size before/after preprocessing:", img.size, pre.size)
print("OCR of rendered     :", pytesseract.image_to_string(img, config="--psm 3").splitlines()[:3])
print("OCR of preprocessed :", pytesseract.image_to_string(pre, config="--psm 3").splitlines()[:3])

tables = detect_and_ocr_tables(pre)
print("tables detected:", len(tables))
for t in tables:
    print(str(t)[:300])