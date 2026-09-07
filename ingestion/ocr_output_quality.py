"""
ingestion/ocr_output_quality.py — Layer 0, POST-OCR text quality gate.

# adding code from antigravity

WHY THIS FILE EXISTS (READ THIS BEFORE ADJUSTING THE THRESHOLD)
-------------------------------------------------------------------
image_preprocessing.py's blur gate rejects a page BEFORE OCR runs, based
on how sharp the image looks. That catches a lot of bad scans, but it
is a PROXY for the thing we actually care about (can the OCR engine
read this reliably) -- and proxies can be wrong. Confirmed on a real
resume (test_scan_blurry.pdf, Jitendra Singh): the image scored 7.88 on
the blur-variance scale (comfortably above the MIN_BLUR_VARIANCE=6.0
reject threshold). The image passed the blur gate, but Surya's actual
OCR output on it was badly degenerate -- the model visibly looping and
repeating its own earlier guess back into the output, e.g.:

    "...contribute to a dynamic organization. I aspire to de
     academic knowledge and passion for technology to contribute to a
     dynamic organization. I aspire to..."

That repeated-phrase pattern is a well-known signature of a
sequence-recognition model losing track of what it's reading and
looping -- and it turns out to be a MUCH more reliable signal than the
two alternatives that were considered:

  1. Raw average OCR confidence: tested directly against this exact
     file. Average confidence on the degenerate output was 87.2 --
     barely different from three known-clean real resumes (89.5-91.7).
     Too small a gap to set a reliable threshold on. NOT used.
  2. Image blur score: already shown above to have passed this exact
     file through undetected.
  3. REPEATED-PHRASE RATIO (what this file implements): tested
     directly -- ratio 0.146 on degenerate output vs. 0.0000 on all
     three clean resumes. Large, consistent gap.

HOW THIS INTERACTS WITH THE TWO OCR ENGINES (Surya vs tesseract)
-------------------------------------------------------------------
Confirmed on the same real blurry file: Surya produced degenerate,
looping output, but tesseract's own independent reading of the exact
same image was mostly clean. So a blurry scan does not doom BOTH
engines equally.

  - ocr_reader.py (Surya): a page whose Surya output is flagged
    degenerate gets ONE retry via tesseract, before anything is
    rejected. Only if tesseract is ALSO degenerate is DegenerateOCROutputError
    raised.
  - ocr_reader_pytesseract.py (tesseract, last engine in the chain):
    no further fallback available, so degenerate output here is a
    genuine hard rejection.

CALIBRATION STATUS
-------------------------------------------------------------------
MAX_REPEATED_NGRAM_RATIO=0.04 was chosen from real measurements (one
confirmed-bad file at 0.146, three confirmed-clean real resumes all at
0.0000). Log real scores in production and revisit this threshold once
more genuine bad scans (or false rejections) have been observed.
"""

from collections import Counter

# How many consecutive words make up one "phrase" being checked for
# repetition. 6 is long enough that two different sentences landing on
# the same 6 words by coincidence is very unlikely, but short enough
# to catch real repeated/looping phrases.
MIN_NGRAM_LENGTH = 6

# See CALIBRATION STATUS above. A page whose repeated-6-word-phrase
# ratio exceeds this is flagged as degenerate.
MAX_REPEATED_NGRAM_RATIO = 0.04

# Below this many words there isn't enough text for a ratio to mean
# anything reliable. Pages shorter than this are never flagged.
MIN_WORDS_TO_CHECK = MIN_NGRAM_LENGTH * 2


class DegenerateOCROutputError(Exception):
    """
    Raised when OCR output looks degenerate (repeating/looping text)
    on EVERY engine tried for a page, with no further fallback.
    Callers must let this propagate all the way up to the user as a
    genuine rejection ("please re-upload a clearer scan"), exactly
    like image_preprocessing.ImageQualityError -- never catch this
    as a generic OCR failure and never silently swallow it.

    Deliberately NOT a subclass of ImageQualityError: this is a
    different failure category (the image passed the pre-OCR quality
    gate; the actual OCR OUTPUT came out degenerate).
    """
    def __init__(self, message: str, repeated_ratio: float, context: str = ""):
        self.repeated_ratio = repeated_ratio
        self.context = context
        super().__init__(message)


def repeated_ngram_ratio(text: str, n: int = MIN_NGRAM_LENGTH) -> float:
    """
    What fraction of consecutive n-word windows in `text` are exact
    duplicates of an earlier window in the same text?

    Ordinary prose (even resumes with two similar bullet points, or a
    section heading that appears more than once) scores at or very
    close to 0.0 -- verified directly against three real, full, clean
    resume OCR outputs (all scored exactly 0.0000).

    Degenerate/looping OCR output scores much higher -- verified
    directly against a real garbled resume (scored 0.146).

    Returns 0.0 (never flagged) if there isn't enough text to compute
    a meaningful ratio -- see MIN_WORDS_TO_CHECK.
    """
    words = text.split()
    if len(words) < MIN_WORDS_TO_CHECK:
        return 0.0

    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    # Each ngram that appears k times contributes (k-1) "extra"
    # occurrences beyond its first, legitimate appearance.
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def check_not_degenerate(text: str, context: str = "",
                          max_ratio: float = MAX_REPEATED_NGRAM_RATIO) -> float:
    """
    Raises DegenerateOCROutputError if `text` looks degenerate.

    Use this as a LAST-RESORT check -- only after every OCR engine/
    retry available to the caller has already been tried. Callers that
    still have another engine to try first should call
    repeated_ngram_ratio() directly instead, so they can decide
    whether to retry before deciding whether to reject.

    Returns the computed ratio (for logging) if it passed.
    """
    ratio = repeated_ngram_ratio(text)
    if ratio > max_ratio:
        raise DegenerateOCROutputError(
            f"OCR output looks degenerate: a {MIN_NGRAM_LENGTH}-word phrase "
            f"repeats far more than normal text ever does (repeated-phrase "
            f"ratio {ratio:.3f}, threshold {max_ratio}). This usually means "
            f"the OCR engine lost track of what it was reading and started "
            f"looping -- a known failure mode on scans that are degraded in "
            f"a way the earlier sharpness check didn't catch. "
            f"{('[' + context + '] ') if context else ''}"
            f"Please re-upload a clearer scan or photo.",
            repeated_ratio=ratio,
            context=context,
        )
    return ratio
