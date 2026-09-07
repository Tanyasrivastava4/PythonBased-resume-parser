"""
ingestion/image_preprocessing.py — Layer 0 pre-OCR image quality gate.

Applies to "scanned" pages only (PDF pages with no extractable text, and
standalone image uploads) -- text/hybrid pages never touch this module,
since they don't need OCR at all.

Three checks, run in this order for every reason below:

  1. BLUR REJECTION (hard gate, raises ImageQualityError)
     Measured via Laplacian variance (same technique used in the invoice
     OCR pipeline this was adapted from). Calibrated against a controlled
     blur ladder run on a real resume through this exact pipeline at the
     same 200 DPI render resolution used in production:

         Gaussian blur radius   Laplacian variance   Surya output quality
         ---------------------  -------------------  -----------------------------------
         1.5  (mild)             53.13                clean, correct
         3.5  (moderate)         10.18                clean, correct
         6.0  (heavy)             3.42                sentences intact but silently wrong
                                                        fields (email, dates, proper nouns
                                                        misread e.g. "NIC"->"NRC",
                                                        "May 2023"->"Many 2012 3")
         9.0  (severe)            1.66                hallucinated fluent-sounding garbage,
                                                        not related to source text at all
         13.0 (extreme)           0.81                same collapse, worse

     The gap between "moderate" (10.18, safe) and "heavy" (3.42, unsafe --
     silently wrong data reaches the parser) sets MIN_BLUR_VARIANCE at the
     midpoint. This deliberately blocks "heavy" and worse: letting heavy
     through was considered and rejected, because the failure mode there
     is NOT visible garbage -- it's plausible-looking wrong data (wrong
     email character, wrong dates, wrong proper nouns) that a downstream
     ATS match would trust without any signal something was off. Rejecting
     up front and asking for a re-upload is safer than silently corrupting
     candidate data.

     NOTE: calibrated on ONE resume's synthetic Gaussian blur ladder. This
     is a well-reasoned starting threshold, not a proven-in-production
     number. Real-world blur (camera shake / motion blur / low-light
     noise / JPEG artifacts) will not have identical statistics to a pure
     Gaussian blur. Log real production blur scores and revisit this
     constant once genuine bad scans have been observed.

  2. DESKEW (corrective, always safe to apply)
     Ported from the invoice OCR pipeline's Hough-line angle detection.
     Justified by direct evidence on this project (not the invoice
     pipeline's use case): a 4.5-degree rotation caused a section heading
     ("Summary") to be sorted into the MIDDLE of an unrelated bullet
     sentence, and a whole line ("station") to disappear from the output
     entirely. Root cause: reading order is reconstructed by sorting
     words on their y-coordinate ("top"), and on a rotated page the same
     visual line has measurably different y-coordinates on the left edge
     of the page vs. the right edge, which is enough to scramble the sort
     for lines that are close together vertically.

     Deskewing does not have the same "is this even a good idea" question
     that blur correction does: unlike blur (irreversible information
     loss), skew is a clean geometric transform. Rotating back by the
     measured angle restores the original layout with no data loss.

  3. BRIGHTNESS / INVERSION (corrective, conservative)
     Ported from the invoice pipeline's brightness heuristic. Applied
     more conservatively here than in the invoice case: resumes commonly
     have colored sidebars, shaded section-heading bars, and colored
     icons (all things invoices don't usually have), so a naive "whole
     page brightness" check could misfire on a perfectly good resume with
     a dark-colored design element. This only inverts when the WHOLE page
     is moderately dark overall (60-85 mean grayscale, same band the
     invoice pipeline used) -- a real photo taken in poor lighting, not a
     resume with some dark decorative elements on an otherwise bright
     background. No "reject if too dark" step is included yet (unlike the
     invoice pipeline's <60 reject) since that threshold hasn't been
     validated against real resume scans -- add it later if evidence
     shows a need, following the same calibrate-before-gating approach
     used for blur above.
"""

import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Calibrated / ported thresholds (see module docstring for reasoning) ──
MIN_BLUR_VARIANCE = 6.0          # below this -> reject (see calibration table above)
MIN_SKEW_ANGLE_TO_CORRECT = 2.0  # degrees; skip correction below this (avoid needless warping)
DARK_INVERT_LOW = 60             # mean grayscale brightness band that triggers inversion
DARK_INVERT_HIGH = 85


class ImageQualityError(Exception):
    """
    Raised when a scanned page/image fails the blur quality gate and
    should be rejected rather than sent to OCR at all. Callers (see
    ocr_reader.py) must NOT treat this as a generic OCR failure -- it
    must propagate all the way up to the user as "please re-upload a
    clearer scan", not silently trigger the Surya->tesseract fallback
    path, and not be swallowed anywhere in between.
    """
    def __init__(self, message: str, blur_score: float, context: str = ""):
        self.blur_score = blur_score
        self.context = context
        super().__init__(message)


def _pil_to_gray_np(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# Laplacian variance is resolution-dependent: the SAME physical blur
# scores noticeably differently depending on the render DPI feeding it.
# Confirmed directly: the "moderate" blur level from our calibration
# ladder scored 10.18 at 200 DPI (Surya's render resolution, in
# ocr_reader.py) but only 6.77 at 300 DPI (the tesseract fallback's
# render resolution, in ocr_reader_pytesseract.py) -- nearly a 2x
# difference for identical blur. Since this module is shared by both
# OCR paths, a single un-normalized threshold would be silently wrong
# for whichever path doesn't match the DPI it was calibrated against
# (a real resume at the "moderate, should pass" level would have
# scored dangerously close to rejection on the 300 DPI path).
#
# Fix: resize to a fixed reference width before scoring. 1700px is
# roughly what a standard 8.5in-wide page renders to at 200 DPI.
# Re-verified after this change: moderate normalizes to ~7.4-7.9 and
# heavy to ~2.2-2.4 regardless of whether the source was rendered at
# 200 or 300 DPI -- MIN_BLUR_VARIANCE=6.0 still cleanly separates them
# either way.
_BLUR_REFERENCE_WIDTH = 1700


def compute_blur_score(img: Image.Image) -> float:
    """Laplacian variance -- higher means sharper. See calibration table
    in the module docstring for what specific values mean in practice.
    Normalized to a fixed reference width first so the same threshold
    is valid regardless of what DPI the caller rendered at (see comment
    on _BLUR_REFERENCE_WIDTH above)."""
    gray_img = img.convert("L")
    if gray_img.width != _BLUR_REFERENCE_WIDTH:
        scale = _BLUR_REFERENCE_WIDTH / gray_img.width
        new_size = (_BLUR_REFERENCE_WIDTH, max(1, round(gray_img.height * scale)))
        gray_img = gray_img.resize(new_size, Image.LANCZOS)
    gray = np.array(gray_img)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_brightness_score(img: Image.Image) -> float:
    """Mean grayscale brightness, 0 (black) - 255 (white)."""
    gray = _pil_to_gray_np(img)
    return float(np.mean(gray))


def deskew_image(img: Image.Image):
    """
    Detects rotation via Hough line transform and corrects it. Returns
    (corrected_image, detected_angle, was_corrected). If no reliable
    lines are found, or the detected angle is below
    MIN_SKEW_ANGLE_TO_CORRECT, returns the original image unchanged.

    Ported from the invoice pipeline's deskew_image(), adapted to take
    and return PIL Images (the type used throughout ingestion/) instead
    of raw cv2/BGR arrays.
    """
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is None:
        return img, 0.0, False

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if 5 < abs(angle) < 85:
            angles.append(angle)

    if not angles:
        return img, 0.0, False

    median_angle = float(np.median(angles))

    if abs(median_angle) < MIN_SKEW_ANGLE_TO_CORRECT:
        return img, median_angle, False

    h, w = bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed_bgr = cv2.warpAffine(
        bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    deskewed_rgb = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(deskewed_rgb), median_angle, True


def adjust_brightness(img: Image.Image, brightness_score: float) -> Image.Image:
    """
    Inverts the image if it falls in the moderately-dark band. See module
    docstring for why this is narrower/more conservative than the
    invoice pipeline's version.
    """
    if DARK_INVERT_LOW <= brightness_score <= DARK_INVERT_HIGH:
        rgb = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        inverted_bgr = cv2.bitwise_not(bgr)
        inverted_rgb = cv2.cvtColor(inverted_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(inverted_rgb)
    return img


def preprocess_scanned_image(img: Image.Image, context: str = "") -> Image.Image:
    """
    Main entry point. Runs blur gate -> deskew -> brightness correction,
    in that order, on a single page/image about to be sent to OCR.

    context: human-readable label for error messages / logging, e.g.
      "page 2 of resume.pdf" or the image filename. Purely cosmetic.

    Raises ImageQualityError if the image fails the blur gate. Callers
    must let this propagate -- do not catch it as a generic OCR failure.
    """
    # Blank / solid-color page check: a blank page (e.g. empty trailing page in a PDF)
    # has zero edges, so its Laplacian variance is 0.00. It is NOT a blurry scan,
    # just an empty page. Bypass blur rejection for blank images (std < 5.0).
    gray_arr = np.array(img.convert("L"))
    if np.std(gray_arr) < 5.0:
        return img

    blur_score = compute_blur_score(img)
    if blur_score < MIN_BLUR_VARIANCE:
        raise ImageQualityError(
            f"Image quality too low to process reliably "
            f"(blur score {blur_score:.2f}, minimum {MIN_BLUR_VARIANCE}). "
            f"{('[' + context + '] ') if context else ''}"
            f"Please re-upload a clearer scan or photo.",
            blur_score=blur_score,
            context=context,
        )

    from PIL import ImageOps
    img = ImageOps.expand(img, border=30, fill="white")
    img, angle, corrected = deskew_image(img)
    if corrected:
        logger.info(f"Deskewed {context or 'image'} by {angle:.2f} degrees")

    brightness_score = compute_brightness_score(img)
    img = adjust_brightness(img, brightness_score)

    return img















#updating after coming from home and seeing it will work or not just pasted above
#"""
#ingestion/image_preprocessing.py — Layer 0 pre-OCR image quality gate.
#
#Applies to "scanned" pages only (PDF pages with no extractable text, and
#standalone image uploads) -- text/hybrid pages never touch this module,
#since they don't need OCR at all.
#
#Three checks, run in this order for every reason below:
#
#  1. BLUR REJECTION (hard gate, raises ImageQualityError)
#     Measured via Laplacian variance (same technique used in the invoice
#     OCR pipeline this was adapted from). Calibrated against a controlled
#     blur ladder run on a real resume through this exact pipeline at the
#     same 200 DPI render resolution used in production:
#
#         Gaussian blur radius   Laplacian variance   Surya output quality
#         ---------------------  -------------------  -----------------------------------
#         1.5  (mild)             53.13                clean, correct
#         3.5  (moderate)         10.18                clean, correct
#         6.0  (heavy)             3.42                sentences intact but silently wrong
#                                                        fields (email, dates, proper nouns
#                                                        misread e.g. "NIC"->"NRC",
#                                                        "May 2023"->"Many 2012 3")
#         9.0  (severe)            1.66                hallucinated fluent-sounding garbage,
#                                                        not related to source text at all
#         13.0 (extreme)           0.81                same collapse, worse
#
#     The gap between "moderate" (10.18, safe) and "heavy" (3.42, unsafe --
#     silently wrong data reaches the parser) sets MIN_BLUR_VARIANCE at the
#     midpoint. This deliberately blocks "heavy" and worse: letting heavy
#     through was considered and rejected, because the failure mode there
#     is NOT visible garbage -- it's plausible-looking wrong data (wrong
#     email character, wrong dates, wrong proper nouns) that a downstream
#     ATS match would trust without any signal something was off. Rejecting
#     up front and asking for a re-upload is safer than silently corrupting
#     candidate data.
#
#     NOTE: calibrated on ONE resume's synthetic Gaussian blur ladder. This
#     is a well-reasoned starting threshold, not a proven-in-production
#     number. Real-world blur (camera shake / motion blur / low-light
#     noise / JPEG artifacts) will not have identical statistics to a pure
#     Gaussian blur. Log real production blur scores and revisit this
#     constant once genuine bad scans have been observed.
#
#  2. DESKEW (corrective, always safe to apply)
#     Ported from the invoice OCR pipeline's Hough-line angle detection.
#     Justified by direct evidence on this project (not the invoice
#     pipeline's use case): a 4.5-degree rotation caused a section heading
#     ("Summary") to be sorted into the MIDDLE of an unrelated bullet
#     sentence, and a whole line ("station") to disappear from the output
#     entirely. Root cause: reading order is reconstructed by sorting
#     words on their y-coordinate ("top"), and on a rotated page the same
#     visual line has measurably different y-coordinates on the left edge
#     of the page vs. the right edge, which is enough to scramble the sort
#     for lines that are close together vertically.
#
#     Deskewing does not have the same "is this even a good idea" question
#     that blur correction does: unlike blur (irreversible information
#     loss), skew is a clean geometric transform. Rotating back by the
#     measured angle restores the original layout with no data loss.
#
#  3. BRIGHTNESS / INVERSION (corrective, conservative)
#     Ported from the invoice pipeline's brightness heuristic. Applied
#     more conservatively here than in the invoice case: resumes commonly
#     have colored sidebars, shaded section-heading bars, and colored
#     icons (all things invoices don't usually have), so a naive "whole
#     page brightness" check could misfire on a perfectly good resume with
#     a dark-colored design element. This only inverts when the WHOLE page
#     is moderately dark overall (60-85 mean grayscale, same band the
#     invoice pipeline used) -- a real photo taken in poor lighting, not a
#     resume with some dark decorative elements on an otherwise bright
#     background. No "reject if too dark" step is included yet (unlike the
#     invoice pipeline's <60 reject) since that threshold hasn't been
#     validated against real resume scans -- add it later if evidence
#     shows a need, following the same calibrate-before-gating approach
#     used for blur above.
#"""
#
#import logging
#import cv2
#import numpy as np
#from PIL import Image
#
#logger = logging.getLogger(__name__)
#
#
## ── Calibrated / ported thresholds (see module docstring for reasoning) ──
#MIN_BLUR_VARIANCE = 6.0          # below this -> reject (see calibration table above)
#MIN_SKEW_ANGLE_TO_CORRECT = 2.0  # degrees; skip correction below this (avoid needless warping)
#DARK_INVERT_LOW = 60             # mean grayscale brightness band that triggers inversion
#DARK_INVERT_HIGH = 85
#
#
#class ImageQualityError(Exception):
#    """
#    Raised when a scanned page/image fails the blur quality gate and
#    should be rejected rather than sent to OCR at all. Callers (see
#    ocr_reader.py) must NOT treat this as a generic OCR failure -- it
#    must propagate all the way up to the user as "please re-upload a
#    clearer scan", not silently trigger the Surya->tesseract fallback
#    path, and not be swallowed anywhere in between.
#    """
#    def __init__(self, message: str, blur_score: float, context: str = ""):
#        self.blur_score = blur_score
#        self.context = context
#        super().__init__(message)
#
#
#def _pil_to_gray_np(img: Image.Image) -> np.ndarray:
#    rgb = np.array(img.convert("RGB"))
#    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#
#
## Laplacian variance is resolution-dependent: the SAME physical blur
## scores noticeably differently depending on the render DPI feeding it.
## Confirmed directly: the "moderate" blur level from our calibration
## ladder scored 10.18 at 200 DPI (Surya's render resolution, in
## ocr_reader.py) but only 6.77 at 300 DPI (the tesseract fallback's
## render resolution, in ocr_reader_pytesseract.py) -- nearly a 2x
## difference for identical blur. Since this module is shared by both
## OCR paths, a single un-normalized threshold would be silently wrong
## for whichever path doesn't match the DPI it was calibrated against
## (a real resume at the "moderate, should pass" level would have
## scored dangerously close to rejection on the 300 DPI path).
##
## Fix: resize to a fixed reference width before scoring. 1700px is
## roughly what a standard 8.5in-wide page renders to at 200 DPI.
## Re-verified after this change: moderate normalizes to ~7.4-7.9 and
## heavy to ~2.2-2.4 regardless of whether the source was rendered at
## 200 or 300 DPI -- MIN_BLUR_VARIANCE=6.0 still cleanly separates them
## either way.
#_BLUR_REFERENCE_WIDTH = 1700
#
#
#def compute_blur_score(img: Image.Image) -> float:
#    """Laplacian variance -- higher means sharper. See calibration table
#    in the module docstring for what specific values mean in practice.
#    Normalized to a fixed reference width first so the same threshold
#    is valid regardless of what DPI the caller rendered at (see comment
#    on _BLUR_REFERENCE_WIDTH above)."""
#    gray_img = img.convert("L")
#    if gray_img.width != _BLUR_REFERENCE_WIDTH:
#        scale = _BLUR_REFERENCE_WIDTH / gray_img.width
#        new_size = (_BLUR_REFERENCE_WIDTH, max(1, round(gray_img.height * scale)))
#        gray_img = gray_img.resize(new_size, Image.LANCZOS)
#    gray = np.array(gray_img)
#    return cv2.Laplacian(gray, cv2.CV_64F).var()
#
#
#def compute_brightness_score(img: Image.Image) -> float:
#    """Mean grayscale brightness, 0 (black) - 255 (white)."""
#    gray = _pil_to_gray_np(img)
#    return float(np.mean(gray))
#
#
#def deskew_image(img: Image.Image):
#    """
#    Detects rotation via Hough line transform and corrects it. Returns
#    (corrected_image, detected_angle, was_corrected). If no reliable
#    lines are found, or the detected angle is below
#    MIN_SKEW_ANGLE_TO_CORRECT, returns the original image unchanged.
#
#    Ported from the invoice pipeline's deskew_image(), adapted to take
#    and return PIL Images (the type used throughout ingestion/) instead
#    of raw cv2/BGR arrays.
#    """
#    rgb = np.array(img.convert("RGB"))
#    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#
#    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#    gray = cv2.medianBlur(gray, 3)
#
#    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
#
#    lines = cv2.HoughLinesP(
#        edges, 1, np.pi / 180,
#        threshold=100, minLineLength=100, maxLineGap=10
#    )
#
#    if lines is None:
#        return img, 0.0, False
#
#    angles = []
#    for line in lines:
#        x1, y1, x2, y2 = line[0]
#        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
#        if 5 < abs(angle) < 85:
#            angles.append(angle)
#
#    if not angles:
#        return img, 0.0, False
#
#    median_angle = float(np.median(angles))
#
#    if abs(median_angle) < MIN_SKEW_ANGLE_TO_CORRECT:
#        return img, median_angle, False
#
#    h, w = bgr.shape[:2]
#    center = (w // 2, h // 2)
#    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
#    deskewed_bgr = cv2.warpAffine(
#        bgr, M, (w, h),
#        flags=cv2.INTER_CUBIC,
#        borderMode=cv2.BORDER_REPLICATE,
#    )
#    deskewed_rgb = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2RGB)
#    return Image.fromarray(deskewed_rgb), median_angle, True
#
#
#def adjust_brightness(img: Image.Image, brightness_score: float) -> Image.Image:
#    """
#    Inverts the image if it falls in the moderately-dark band. See module
#    docstring for why this is narrower/more conservative than the
#    invoice pipeline's version.
#    """
#    if DARK_INVERT_LOW <= brightness_score <= DARK_INVERT_HIGH:
#        rgb = np.array(img.convert("RGB"))
#        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#        inverted_bgr = cv2.bitwise_not(bgr)
#        inverted_rgb = cv2.cvtColor(inverted_bgr, cv2.COLOR_BGR2RGB)
#        return Image.fromarray(inverted_rgb)
#    return img
#
#
#def preprocess_scanned_image(img: Image.Image, context: str = "") -> Image.Image:
#    """
#    Main entry point. Runs blur gate -> deskew -> brightness correction,
#    in that order, on a single page/image about to be sent to OCR.
#
#    context: human-readable label for error messages / logging, e.g.
#      "page 2 of resume.pdf" or the image filename. Purely cosmetic.
#
#    Raises ImageQualityError if the image fails the blur gate. Callers
#    must let this propagate -- do not catch it as a generic OCR failure.
#    """
#    blur_score = compute_blur_score(img)
#    if blur_score < MIN_BLUR_VARIANCE:
#        raise ImageQualityError(
#            f"Image quality too low to process reliably "
#            f"(blur score {blur_score:.2f}, minimum {MIN_BLUR_VARIANCE}). "
#            f"{('[' + context + '] ') if context else ''}"
#            f"Please re-upload a clearer scan or photo.",
#            blur_score=blur_score,
#            context=context,
#        )
#
#    img, angle, corrected = deskew_image(img)
#    if corrected:
#        logger.info(f"Deskewed {context or 'image'} by {angle:.2f} degrees")
#
#    brightness_score = compute_brightness_score(img)
#    img = adjust_brightness(img, brightness_score)
#
#    return img
#













#"""
#ingestion/image_preprocessing.py — Layer 0 pre-OCR image quality gate.
#
#Applies to "scanned" pages only (PDF pages with no extractable text, and
#standalone image uploads) -- text/hybrid pages never touch this module,
#since they don't need OCR at all.
#
#Three checks, run in this order for every reason below:
#
#  1. BLUR REJECTION (hard gate, raises ImageQualityError)
#     Measured via Laplacian variance (same technique used in the invoice
#     OCR pipeline this was adapted from). Calibrated against a controlled
#     blur ladder run on a real resume through this exact pipeline at the
#     same 200 DPI render resolution used in production:
#
#         Gaussian blur radius   Laplacian variance   Surya output quality
#         ---------------------  -------------------  -----------------------------------
#         1.5  (mild)             53.13                clean, correct
#         3.5  (moderate)         10.18                clean, correct
#         6.0  (heavy)             3.42                sentences intact but silently wrong
#                                                        fields (email, dates, proper nouns
#                                                        misread e.g. "NIC"->"NRC",
#                                                        "May 2023"->"Many 2012 3")
#         9.0  (severe)            1.66                hallucinated fluent-sounding garbage,
#                                                        not related to source text at all
#         13.0 (extreme)           0.81                same collapse, worse
#
#     The gap between "moderate" (10.18, safe) and "heavy" (3.42, unsafe --
#     silently wrong data reaches the parser) sets MIN_BLUR_VARIANCE at the
#     midpoint. This deliberately blocks "heavy" and worse: letting heavy
#     through was considered and rejected, because the failure mode there
#     is NOT visible garbage -- it's plausible-looking wrong data (wrong
#     email character, wrong dates, wrong proper nouns) that a downstream
#     ATS match would trust without any signal something was off. Rejecting
#     up front and asking for a re-upload is safer than silently corrupting
#     candidate data.
#
#     NOTE: calibrated on ONE resume's synthetic Gaussian blur ladder. This
#     is a well-reasoned starting threshold, not a proven-in-production
#     number. Real-world blur (camera shake / motion blur / low-light
#     noise / JPEG artifacts) will not have identical statistics to a pure
#     Gaussian blur. Log real production blur scores and revisit this
#     constant once genuine bad scans have been observed.
#
#  2. DESKEW (corrective, always safe to apply)
#     Ported from the invoice OCR pipeline's Hough-line angle detection.
#     Justified by direct evidence on this project (not the invoice
#     pipeline's use case): a 4.5-degree rotation caused a section heading
#     ("Summary") to be sorted into the MIDDLE of an unrelated bullet
#     sentence, and a whole line ("station") to disappear from the output
#     entirely. Root cause: reading order is reconstructed by sorting
#     words on their y-coordinate ("top"), and on a rotated page the same
#     visual line has measurably different y-coordinates on the left edge
#     of the page vs. the right edge, which is enough to scramble the sort
#     for lines that are close together vertically.
#
#     Deskewing does not have the same "is this even a good idea" question
#     that blur correction does: unlike blur (irreversible information
#     loss), skew is a clean geometric transform. Rotating back by the
#     measured angle restores the original layout with no data loss.
#
#  3. BRIGHTNESS / INVERSION (corrective, conservative)
#     Ported from the invoice pipeline's brightness heuristic. Applied
#     more conservatively here than in the invoice case: resumes commonly
#     have colored sidebars, shaded section-heading bars, and colored
#     icons (all things invoices don't usually have), so a naive "whole
#     page brightness" check could misfire on a perfectly good resume with
#     a dark-colored design element. This only inverts when the WHOLE page
#     is moderately dark overall (60-85 mean grayscale, same band the
#     invoice pipeline used) -- a real photo taken in poor lighting, not a
#     resume with some dark decorative elements on an otherwise bright
#     background. No "reject if too dark" step is included yet (unlike the
#     invoice pipeline's <60 reject) since that threshold hasn't been
#     validated against real resume scans -- add it later if evidence
#     shows a need, following the same calibrate-before-gating approach
#     used for blur above.
#"""
#
#import logging
#import cv2
#import numpy as np
#from PIL import Image
#
#logger = logging.getLogger(__name__)
#
#
## ── Calibrated / ported thresholds (see module docstring for reasoning) ──
#MIN_BLUR_VARIANCE = 6.0          # below this -> reject (see calibration table above)
#MIN_SKEW_ANGLE_TO_CORRECT = 2.0  # degrees; skip correction below this (avoid needless warping)
#DARK_INVERT_LOW = 60             # mean grayscale brightness band that triggers inversion
#DARK_INVERT_HIGH = 85
#
#
#class ImageQualityError(Exception):
#    """
#    Raised when a scanned page/image fails the blur quality gate and
#    should be rejected rather than sent to OCR at all. Callers (see
#    ocr_reader.py) must NOT treat this as a generic OCR failure -- it
#    must propagate all the way up to the user as "please re-upload a
#    clearer scan", not silently trigger the Surya->tesseract fallback
#    path, and not be swallowed anywhere in between.
#    """
#    def __init__(self, message: str, blur_score: float, context: str = ""):
#        self.blur_score = blur_score
#        self.context = context
#        super().__init__(message)
#
#
#def _pil_to_gray_np(img: Image.Image) -> np.ndarray:
#    rgb = np.array(img.convert("RGB"))
#    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#
#
#def compute_blur_score(img: Image.Image) -> float:
#    """Laplacian variance -- higher means sharper. See calibration table
#    in the module docstring for what specific values mean in practice."""
#    gray = _pil_to_gray_np(img)
#    return cv2.Laplacian(gray, cv2.CV_64F).var()
#
#
#def compute_brightness_score(img: Image.Image) -> float:
#    """Mean grayscale brightness, 0 (black) - 255 (white)."""
#    gray = _pil_to_gray_np(img)
#    return float(np.mean(gray))
#
#
#def deskew_image(img: Image.Image):
#    """
#    Detects rotation via Hough line transform and corrects it. Returns
#    (corrected_image, detected_angle, was_corrected). If no reliable
#    lines are found, or the detected angle is below
#    MIN_SKEW_ANGLE_TO_CORRECT, returns the original image unchanged.
#
#    Ported from the invoice pipeline's deskew_image(), adapted to take
#    and return PIL Images (the type used throughout ingestion/) instead
#    of raw cv2/BGR arrays.
#    """
#    rgb = np.array(img.convert("RGB"))
#    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#
#    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#    gray = cv2.medianBlur(gray, 3)
#
#    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
#
#    lines = cv2.HoughLinesP(
#        edges, 1, np.pi / 180,
#        threshold=100, minLineLength=100, maxLineGap=10
#    )
#
#    if lines is None:
#        return img, 0.0, False
#
#    angles = []
#    for line in lines:
#        x1, y1, x2, y2 = line[0]
#        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
#        if 5 < abs(angle) < 85:
#            angles.append(angle)
#
#    if not angles:
#        return img, 0.0, False
#
#    median_angle = float(np.median(angles))
#
#    if abs(median_angle) < MIN_SKEW_ANGLE_TO_CORRECT:
#        return img, median_angle, False
#
#    h, w = bgr.shape[:2]
#    center = (w // 2, h // 2)
#    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
#    deskewed_bgr = cv2.warpAffine(
#        bgr, M, (w, h),
#        flags=cv2.INTER_CUBIC,
#        borderMode=cv2.BORDER_REPLICATE,
#    )
#    deskewed_rgb = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2RGB)
#    return Image.fromarray(deskewed_rgb), median_angle, True
#
#
#def adjust_brightness(img: Image.Image, brightness_score: float) -> Image.Image:
#    """
#    Inverts the image if it falls in the moderately-dark band. See module
#    docstring for why this is narrower/more conservative than the
#    invoice pipeline's version.
#    """
#    if DARK_INVERT_LOW <= brightness_score <= DARK_INVERT_HIGH:
#        rgb = np.array(img.convert("RGB"))
#        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#        inverted_bgr = cv2.bitwise_not(bgr)
#        inverted_rgb = cv2.cvtColor(inverted_bgr, cv2.COLOR_BGR2RGB)
#        return Image.fromarray(inverted_rgb)
#    return img
#
#
#def preprocess_scanned_image(img: Image.Image, context: str = "") -> Image.Image:
#    """
#    Main entry point. Runs blur gate -> deskew -> brightness correction,
#    in that order, on a single page/image about to be sent to OCR.
#
#    context: human-readable label for error messages / logging, e.g.
#      "page 2 of resume.pdf" or the image filename. Purely cosmetic.
#
#    Raises ImageQualityError if the image fails the blur gate. Callers
#    must let this propagate -- do not catch it as a generic OCR failure.
#    """
#    blur_score = compute_blur_score(img)
#    if blur_score < MIN_BLUR_VARIANCE:
#        raise ImageQualityError(
#            f"Image quality too low to process reliably "
#            f"(blur score {blur_score:.2f}, minimum {MIN_BLUR_VARIANCE}). "
#            f"{('[' + context + '] ') if context else ''}"
#            f"Please re-upload a clearer scan or photo.",
#            blur_score=blur_score,
#            context=context,
#        )
#
#    img, angle, corrected = deskew_image(img)
#    if corrected:
#        logger.info(f"Deskewed {context or 'image'} by {angle:.2f} degrees")
#
#    brightness_score = compute_brightness_score(img)
#    img = adjust_brightness(img, brightness_score)
#
#    return img
#