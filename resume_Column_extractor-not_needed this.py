import pdfplumber
import numpy as np
import sys


def get_words(page):
    return page.extract_words(use_text_flow=False, keep_blank_chars=False)


def ocr_words_for_page(pil_img, dpi, page_width_pt, page_height_pt):
    """OCR an image-rendered page and return word dicts in PDF-point coordinates."""
    import pytesseract
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    scale = 72.0 / dpi  # convert pixel coords (rendered at `dpi`) back to PDF points
    words = []
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt:
            continue
        try:
            if int(data['conf'][i]) < 0:
                continue
        except (ValueError, TypeError):
            pass
        x0 = data['left'][i] * scale
        y0 = data['top'][i] * scale
        w = data['width'][i] * scale
        h = data['height'][i] * scale
        words.append({'text': txt, 'x0': x0, 'x1': x0 + w, 'top': y0, 'bottom': y0 + h})
    return words


def process_scanned_pdf(path, dpi=200):
    """Fallback pipeline for image-only PDFs (no extractable text layer): OCR then reuse
    the same column-geometry logic on the OCR'd word boxes."""
    from pdf2image import convert_from_path
    imgs = convert_from_path(path, dpi=dpi)
    all_out = []
    for pno, img in enumerate(imgs):
        page_width_pt = img.width * 72.0 / dpi
        page_height_pt = img.height * 72.0 / dpi
        words = ocr_words_for_page(img, dpi, page_width_pt, page_height_pt)
        res = _process_words(words, page_width_pt, page_height_pt)
        all_out.append((pno + 1,) + res)
    return all_out


def find_split_x(words, page_width, min_frac=0.10):
    """Find best vertical split using WORD x0/x1 (not pre-merged lines)."""
    lo = page_width * 0.22
    hi = page_width * 0.78
    n = len(words)
    min_side = max(8, int(n * min_frac))
    best_x, best_score = None, -1e9
    for x in np.linspace(lo, hi, 80):
        left = [w for w in words if w['x1'] <= x + 1]
        right = [w for w in words if w['x0'] >= x - 1]
        crossing = [w for w in words if w['x0'] < x - 1 and w['x1'] > x + 1]
        if len(left) < min_side or len(right) < min_side:
            continue
        score = len(left) + len(right) - 6 * len(crossing)
        if score > best_score:
            best_score, best_x = score, x
    return best_x, best_score


def group_words_into_lines(words, y_tol=3):
    if not words:
        return []
    words = sorted(words, key=lambda w: (w['top'], w['x0']))
    lines, cur, cur_top = [], [], None
    for w in words:
        if cur_top is None or abs(w['top'] - cur_top) <= y_tol:
            cur.append(w)
            cur_top = w['top'] if cur_top is None else min(cur_top, w['top'])
        else:
            lines.append(cur)
            cur, cur_top = [w], w['top']
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        ln = sorted(ln, key=lambda w: w['x0'])
        out.append({
            'text': " ".join(w['text'] for w in ln),
            'top': min(w['top'] for w in ln),
            'x0': min(w['x0'] for w in ln),
            'x1': max(w['x1'] for w in ln),
        })
    out.sort(key=lambda l: l['top'])
    return out


HEADER_BAND_FRAC = 0.14  # top 14% of page treated as single full-width flow (name/contact banners)


def _process_words(words, page_width, page_height, y_tol=2.5):
    if not words:
        return None, None, []
    header_cut = page_height * HEADER_BAND_FRAC
    header_words = [w for w in words if w['top'] < header_cut]
    body_words = [w for w in words if w['top'] >= header_cut]

    result = []
    for l in group_words_into_lines(header_words, y_tol=y_tol):
        result.append(('HEADER', l['text']))

    if not body_words:
        return None, None, result

    split_x, score = find_split_x(body_words, page_width)

    if split_x is None:
        for l in group_words_into_lines(body_words, y_tol=y_tol):
            result.append(('BODY', l['text']))
        return split_x, score, result

    left_words, right_words = [], []
    for w in body_words:
        cx = (w['x0'] + w['x1']) / 2
        if cx <= split_x:
            left_words.append(w)
        else:
            right_words.append(w)

    for l in group_words_into_lines(left_words, y_tol=y_tol):
        result.append(('LEFT', l['text']))
    for l in group_words_into_lines(right_words, y_tol=y_tol):
        result.append(('RIGHT', l['text']))

    return split_x, score, result


def process_page(page, y_tol=2.5):
    words = get_words(page)
    split_x, score, result = _process_words(words, page.width, page.height, y_tol=y_tol)
    return result, split_x, score


def process_pdf(path, ocr_word_threshold=3):
    """Try native text extraction; fall back to OCR per-page if a page has ~no text layer."""
    all_out = []
    needs_ocr_pages = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages):
            words = get_words(page)
            if len(words) < ocr_word_threshold:
                needs_ocr_pages.append(pno)
                all_out.append([pno + 1, None, None, None])  # placeholder
                continue
            split_x, score, res = _process_words(words, page.width, page.height, y_tol=y_tol_default)
            all_out.append([pno + 1, split_x, score, res])

    if needs_ocr_pages:
        from pdf2image import convert_from_path
        imgs = convert_from_path(path, dpi=200)
        for pno in needs_ocr_pages:
            img = imgs[pno]
            page_width_pt = img.width * 72.0 / 200
            page_height_pt = img.height * 72.0 / 200
            words = ocr_words_for_page(img, 200, page_width_pt, page_height_pt)
            split_x, score, res = _process_words(words, page_width_pt, page_height_pt)
            all_out[pno][1:] = [split_x, score, res]
            all_out[pno].append(True)  # mark as OCR'd

    return all_out


y_tol_default = 2.5


if __name__ == '__main__':
    path = sys.argv[1]
    for row in process_pdf(path):
        pno, split_x, score, res = row[0], row[1], row[2], row[3]
        ocr_flag = " [OCR]" if len(row) > 4 and row[4] else ""
        print(f"--- Page {pno}{ocr_flag} | split_x={split_x} score={score} ---")
        for tag, text in res:
            print(f"[{tag:6s}] {text}")