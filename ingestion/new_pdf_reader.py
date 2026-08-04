"""
PDF Reader — Layer 0
Reads a PDF file and returns clean text.
Automatically detects if the PDF is scanned (image-based) or text-based.
Handles three layout types:
  1. Single column  — standard reading order
  2. True two-column — two independent content columns side by side
  3. Sidebar layout — narrow heading column on left, wide content column on right
"""

import re
import pdfplumber
import fitz  # PyMuPDF
from pathlib import Path


def is_scanned_pdf(pdf_path: str) -> bool:
    """
    Detects if a PDF is scanned (image-only) or has embedded text.
    """
    doc = fitz.open(pdf_path)
    total_text = ""
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        total_text += page.get_text()
    doc.close()
    return len(total_text.strip()) < 100


def _find_sidebar_split(words: list, page_width: float) -> float:
    """
    Detects if a page has a sidebar layout — a narrow left column
    containing ONLY section headings, and a wide right column with content.

    Returns the x-coordinate of the split point if sidebar detected,
    or None if not a sidebar layout.

    Strategy:
    We look for a consistent "content start" x-position — the x coordinate
    where the main body text always begins. In a sidebar resume, content
    ALWAYS starts at the same x position (e.g. x=166 for Abhay, x=167 for
    Aarti). The sidebar headings always start at a much smaller x (e.g. x=54).

    If we find that most content lines start at a consistent x position,
    and there are heading-like words consistently to the LEFT of that
    position, it is a sidebar layout.
    """
    if not words:
        return None

    # Only consider left 35% of page for sidebar analysis
    left_threshold = page_width * 0.35

    # Find the most common x0 starting position of words
    # In a sidebar resume, content lines all start at the same x
    from collections import Counter
    x0_counter = Counter(round(w["x0"]) for w in words)

    # Find content start x — the x position where main body text begins.
    # In a sidebar resume this is where all content lines start consistently.
    #
    # Strategy: find the most frequent x0 position that:
    # 1. Is in the left half of page (not too far right)
    # 2. Is not too close to the left edge (not a heading position)
    # 3. Has the most words starting there (dominant content column)
    #
    # We look for the most frequent x0 in the range [80, 45% of page width].
    # This skips sidebar heading positions (usually x < 80) and finds
    # the content column start.
    content_start_x = None
    best_count = 0
    for x0, count in x0_counter.items():
        if 80 < x0 < page_width * 0.45 and count > best_count:
            best_count = count
            content_start_x = x0

    # Content column must have at least 3 lines starting there
    if content_start_x is None or best_count < 3:
        return None

    # Find words that start BEFORE content_start_x — these are sidebar words
    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]

    if not sidebar_candidates:
        return None

    # Sidebar candidates must be few (just section headings)
    if len(sidebar_candidates) > 25:
        return None

    # Find the natural heading cluster within sidebar candidates.
    # Sidebar words like "PROFILE", "SUMMARY", "EXPERIENCE" cluster at
    # small x values (e.g. x=54-108). Sometimes wrapped content lines
    # also start at slightly smaller x than content_start_x (e.g. x=154).
    # We find the heading cluster by looking for a natural gap in x0
    # positions within sidebar candidates — and take everything BEFORE
    # that gap as the true heading column.
    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))

    # Find the biggest gap within sidebar candidate x0 positions
    heading_boundary = sidebar_x0s[-1]  # default: all sidebar words
    for i in range(len(sidebar_x0s) - 1):
        inner_gap = sidebar_x0s[i+1] - sidebar_x0s[i]
        if inner_gap > 30:  # significant gap within sidebar itself
            heading_boundary = sidebar_x0s[i]
            break

    # Gap from heading cluster to content start must be meaningful
    gap = content_start_x - heading_boundary
    if gap < 30:
        return None

    # Confirmed sidebar — split at midpoint
    split = (heading_boundary + content_start_x) / 2
    return split


def _find_column_split(words: list, page_width: float):
    """
    Finds the actual column split point for a two-column layout.

    Instead of always using the page midpoint, we look for the gap
    in x0 start positions that best divides words into two groups.

    Returns (split_x, left_words, right_words) or None if no split found.
    """
    from collections import Counter

    if not words:
        return None

    x0_counter = Counter(round(w["x0"]) for w in words)
    all_x0     = sorted(x0_counter.keys())

    if len(all_x0) < 2:
        return None

    # Find the gap that best splits words into two substantial groups
    best_split = None
    best_score = 0

    for i in range(len(all_x0) - 1):
        gap = all_x0[i+1] - all_x0[i]
        if gap < 15:
            continue

        split_x     = (all_x0[i] + all_x0[i+1]) / 2
        left_count  = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i+1])

        # Both sides must be substantial
        if left_count < 20 or right_count < 20:
            continue

        # Split must be in the middle 40-75% of the page
        # (avoids detecting margins as columns)
        if split_x < page_width * 0.35 or split_x > page_width * 0.80:
            continue

        # Score = gap size × balance between columns
        balance = min(left_count, right_count) / max(left_count, right_count)
        score   = gap * balance

        if score > best_score:
            best_score = score
            best_split = split_x

    if best_split is None:
        return None

    left_words  = [w for w in words if w["x1"] <= best_split]
    right_words = [w for w in words if w["x0"] >  best_split]

    return best_split, left_words, right_words


def _is_true_two_column(words: list, mid_x: float) -> bool:
    """
    Determines whether a page genuinely has a two-column layout.
    Uses _find_column_split to find the actual split rather than
    assuming it is always at the page midpoint.
    """
    result = _find_column_split(words, mid_x * 2)  # mid_x*2 = page_width
    if result is None:
        return False

    _, left_words, right_words = result

    if len(left_words) < 20 or len(right_words) < 20:
        return False

    # Right column must span reasonable height
    if right_words and left_words:
        right_span = max(w["bottom"] for w in right_words) - min(w["top"] for w in right_words)
        left_span  = max(w["bottom"] for w in left_words)  - min(w["top"] for w in left_words)
        if left_span > 0 and right_span / left_span < 0.25:
            return False

    return True


def _extract_words_to_lines(words: list) -> str:
    """
    Reconstructs text lines from pdfplumber word dicts.
    Groups words into lines by vertical position and joins with spaces.
    """
    if not words:
        return ""

    lines = []
    current_line = []
    current_top  = None

    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))

    for word in sorted_words:
        if current_top is None:
            current_top  = word["top"]
            current_line = [word]
            continue

        line_tolerance = max(word["height"], 1) * 0.5
        if abs(word["top"] - current_top) <= line_tolerance:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
            current_top  = word["top"]

    if current_line:
        lines.append(current_line)

    text_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["x0"])
        parts = [w["text"] for w in line_sorted]
        text_lines.append(" ".join(parts))

    return "\n".join(text_lines)


def _merge_sidebar_with_content(sidebar_words: list, content_words: list) -> str:
    """
    Merges sidebar heading words with content words by vertical position.

    In a sidebar layout, section headings sit in a narrow left column
    and content sits in a wide right column. We reconstruct the correct
    reading order by inserting each heading before the content line
    that starts at the same vertical position.

    e.g.:
      sidebar:  top=207 "PERSONAL SUMMARY"
      content:  top=208 "Senior Oracle ERP consultant..."
    Result:
      PERSONAL SUMMARY
      Senior Oracle ERP consultant...
    """
    # Group sidebar words into heading lines by vertical position
    heading_lines = {}
    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))

    current_top  = None
    current_words = []
    for word in sorted_sidebar:
        if current_top is None:
            current_top   = word["top"]
            current_words = [word]
        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 1.5:
            # Use 1.5x height tolerance (not 0.5x) for sidebar headings.
            # Sidebar headings like "PROFESSIONAL\nSUMMARY" can span
            # two lines with a gap of ~12pts — larger than body text spacing.
            current_words.append(word)
            current_top = word["top"]  # update to latest top
        else:
            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
            heading_lines[round(current_top)] = heading_text
            current_top   = word["top"]
            current_words = [word]
    if current_words:
        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
        heading_lines[round(current_top)] = heading_text

    # Build content lines with vertical positions
    content_line_list = []
    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))

    current_top   = None
    current_words = []
    for word in sorted_content:
        if current_top is None:
            current_top   = word["top"]
            current_words = [word]
        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
            current_words.append(word)
        else:
            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
            content_line_list.append((round(current_top), line_text))
            current_top   = word["top"]
            current_words = [word]
    if current_words:
        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
        content_line_list.append((round(current_top), line_text))

    # Merge: insert heading before the content line at the same vertical position
    result_lines = []
    used_headings = set()

    for content_top, content_text in content_line_list:
        # Find sidebar heading closest to this content line (within 20 pts)
        for heading_top, heading_text in sorted(heading_lines.items()):
            if heading_top in used_headings:
                continue
            if abs(heading_top - content_top) <= 20:
                result_lines.append(heading_text)
                used_headings.add(heading_top)
                break

        result_lines.append(content_text)

    # Add any headings that didn't match content lines
    for heading_top, heading_text in sorted(heading_lines.items()):
        if heading_top not in used_headings:
            result_lines.append(heading_text)

    return "\n".join(result_lines)


def read_text_pdf(pdf_path: str) -> str:
    """
    Reads a normal (text-based) PDF using pdfplumber.
    Handles single-column, two-column, and sidebar layouts.
    """
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_width = page.width
            mid_x      = page_width / 2

            words = page.extract_words(
                x_tolerance=1.5,
                y_tolerance=3,
                keep_blank_chars=False,
            )

            if not words:
                continue

            # ── Check for sidebar layout first ──
            sidebar_split = _find_sidebar_split(words, page_width)
            if sidebar_split:
                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
                content_words = [w for w in words if w["x0"] > sidebar_split]
                text = _merge_sidebar_with_content(sidebar_words, content_words)
                if text.strip():
                    full_text.append(text.strip())

            # ── Check for true two-column layout ──
            elif _is_true_two_column(words, mid_x):
                # Use _find_column_split to get the actual split point
                # and correctly separated word groups
                result = _find_column_split(words, page_width)
                if result:
                    _, left_words, right_words = result
                else:
                    left_words  = [w for w in words if w["x1"] <= mid_x]
                    right_words = [w for w in words if w["x0"] >  mid_x]
                left_text   = _extract_words_to_lines(left_words)
                right_text  = _extract_words_to_lines(right_words)
                if left_text.strip():
                    full_text.append(left_text.strip())
                if right_text.strip():
                    full_text.append(right_text.strip())

            # ── Single column ──
            else:
                text = _extract_words_to_lines(words)
                if text.strip():
                    full_text.append(text.strip())

    return "\n\n".join(full_text)


# ──────────────────────────────────────────────────────────────
# SIDEBAR HEADING SPLITTER
# Safety net for sidebar-layout pages where _find_sidebar_split
# doesn't trigger (e.g. page 3 of Abhay's resume).
# Splits lines where a known heading is merged with content.
# e.g. "CERTIFICATION Certificate on PHP..."
#   -> "CERTIFICATION\nCertificate on PHP..."
# ──────────────────────────────────────────────────────────────
_SIDEBAR_HEADINGS = [
    # Multi-word headings first
    "PROFESSIONAL SUMMARY", "PERSONAL SUMMARY", "PERSONAL DETAILS",
    "CORE COMPETENCIES", "PROFILE SUMMARY", "LIVE PROJECTS",
    "WORK EXPERIENCE", "WORK HISTORY", "EMPLOYMENT HISTORY",
    "KEY SKILLS", "IT SKILLS", "TECHNICAL SKILLS", "LANGUAGE SKILLS",
    "ROLES & RESPONSIBILITIES", "ROLES RESPONSIBILITIES",
    "CUSTOM SECTION", "ACADEMIC BACKGROUND",
    # Single-word headings
    "CERTIFICATION", "CERTIFICATIONS", "EDUCATION", "EXPERIENCE",
    "DECLARATION", "SUMMARY", "OBJECTIVE", "PROFILE",
    "SKILLS", "PROJECTS", "ACHIEVEMENTS", "AWARDS",
    "LANGUAGES", "INTERESTS", "HOBBIES", "REFERENCES",
    "PERSONAL",
]


def _split_merged_headings(text: str) -> str:
    """
    Splits lines where a sidebar section heading is merged with content.
    Only runs when _find_sidebar_split misses a page (safety net).
    """
    lines = text.split("\n")
    result = []

    for line in lines:
        stripped = line.strip()
        split_done = False

        for heading in _SIDEBAR_HEADINGS:
            pattern = re.compile(
                rf"^({re.escape(heading)})\s+([A-Za-z0-9][^\n]{{3,}})$",
                re.IGNORECASE
            )
            match = pattern.match(stripped)
            if match:
                result.append(match.group(1))
                result.append(match.group(2))
                split_done = True
                break

        if not split_done:
            result.append(line)

    return "\n".join(result)


def _join_split_headings(text: str) -> str:
    """
    Joins consecutive single-word lines that form a known multi-word heading.
    e.g. "PROFILE" on line N and "SUMMARY" on line N+1
      -> "PROFILE SUMMARY"
    """
    multi_word_headings = [h for h in _SIDEBAR_HEADINGS if " " in h]
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        joined = False
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            combined = line + " " + next_line
            for heading in multi_word_headings:
                if combined.upper() == heading.upper():
                    result.append(heading)
                    i += 2
                    joined = True
                    break
        if not joined:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def read_pdf(pdf_path: str) -> str:
    """
    Main entry point. Decides which method to use based on PDF type.
    Applies sidebar heading splitting as a post-processing safety net.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    if is_scanned_pdf(pdf_path):
        from ingestion.ocr_reader import read_with_surya
        print(f"[INFO] Scanned PDF detected. Using Surya OCR for: {pdf_path}")
        text = read_with_surya(pdf_path)
    else:
        print(f"[INFO] Text PDF detected. Using pdfplumber for: {pdf_path}")
        text = read_text_pdf(pdf_path)

    # Post-processing safety net for sidebar layouts
    text = _split_merged_headings(text)
    text = _join_split_headings(text)
    return text


















#"""
#PDF Reader — Layer 0
#Reads a PDF file and returns clean text.
#Automatically detects if the PDF is scanned (image-based) or text-based.
#Handles three layout types:
#  1. Single column  — standard reading order
#  2. True two-column — two independent content columns side by side
#  3. Sidebar layout — narrow heading column on left, wide content column on right
#"""
#
#import re
#import pdfplumber
#import fitz  # PyMuPDF
#from pathlib import Path
#
#
#def is_scanned_pdf(pdf_path: str) -> bool:
#    """
#    Detects if a PDF is scanned (image-only) or has embedded text.
#    """
#    doc = fitz.open(pdf_path)
#    total_text = ""
#    for page_num in range(min(3, len(doc))):
#        page = doc[page_num]
#        total_text += page.get_text()
#    doc.close()
#    return len(total_text.strip()) < 100
#
#
#def _find_sidebar_split(words: list, page_width: float) -> float:
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
#    # Only consider left 35% of page for sidebar analysis
#    left_threshold = page_width * 0.35
#
#    # Find the most common x0 starting position of words
#    # In a sidebar resume, content lines all start at the same x
#    from collections import Counter
#    x0_counter = Counter(round(w["x0"]) for w in words)
#
#    # Find content start x — the x position where main body text begins.
#    # In a sidebar resume this is where all content lines start consistently.
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
#        inner_gap = sidebar_x0s[i+1] - sidebar_x0s[i]
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
#def _is_true_two_column(words: list, mid_x: float) -> bool:
#    """
#    Determines whether a page genuinely has a two-column layout.
#    """
#    left_words   = [w for w in words if w["x1"] < mid_x - 20]
#    right_words  = [w for w in words if w["x0"] > mid_x + 20]
#    gutter_words = [w for w in words if w["x0"] < mid_x + 20 and w["x1"] > mid_x - 20]
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return False
#    if len(gutter_words) > 5:
#        return False
#
#    if right_words:
#        right_top    = min(w["top"]    for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span   = right_bottom - right_top
#        left_top     = min(w["top"]    for w in left_words)
#        left_bottom  = max(w["bottom"] for w in left_words)
#        left_span    = left_bottom - left_top
#        if left_span > 0 and right_span / left_span < 0.6:
#            return False
#
#    return True
#
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from pdfplumber word dicts.
#    Groups words into lines by vertical position and joins with spaces.
#    """
#    if not words:
#        return ""
#
#    lines = []
#    current_line = []
#    current_top  = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top  = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top  = word["top"]
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
#    current_top  = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top   = word["top"]
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
#            current_top   = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    # Build content lines with vertical positions
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top   = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top   = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top   = word["top"]
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
#def read_text_pdf(pdf_path: str) -> str:
#    """
#    Reads a normal (text-based) PDF using pdfplumber.
#    Handles single-column, two-column, and sidebar layouts.
#    """
#    full_text = []
#
#    with pdfplumber.open(pdf_path) as pdf:
#        for page in pdf.pages:
#            page_width = page.width
#            mid_x      = page_width / 2
#
#            words = page.extract_words(
#                x_tolerance=1.5,
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#
#            if not words:
#                continue
#
#            # ── Check for sidebar layout first ──
#            sidebar_split = _find_sidebar_split(words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#                content_words = [w for w in words if w["x0"] > sidebar_split]
#                text = _merge_sidebar_with_content(sidebar_words, content_words)
#                if text.strip():
#                    full_text.append(text.strip())
#
#            # ── Check for true two-column layout ──
#            elif _is_true_two_column(words, mid_x):
#                left_words  = [w for w in words if w["x1"] <= mid_x]
#                right_words = [w for w in words if w["x0"] >  mid_x]
#                left_text   = _extract_words_to_lines(left_words)
#                right_text  = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    full_text.append(left_text.strip())
#                if right_text.strip():
#                    full_text.append(right_text.strip())
#
#            # ── Single column ──
#            else:
#                text = _extract_words_to_lines(words)
#                if text.strip():
#                    full_text.append(text.strip())
#
#    return "\n\n".join(full_text)
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























#"""
#PDF Reader — Layer 0
#Reads a PDF file and returns clean text.
#Automatically detects if the PDF is scanned (image-based) or text-based.
#Handles three layout types:
#  1. Single column  — standard reading order
#  2. True two-column — two independent content columns side by side
#  3. Sidebar layout — narrow heading column on left, wide content column on right
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
#    """
#    doc = fitz.open(pdf_path)
#    total_text = ""
#    for page_num in range(min(3, len(doc))):
#        page = doc[page_num]
#        total_text += page.get_text()
#    doc.close()
#    return len(total_text.strip()) < 100
#
#
#def _find_sidebar_split(words: list, page_width: float) -> float:
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
#    # Only consider left 35% of page for sidebar analysis
#    left_threshold = page_width * 0.35
#
#    # Find the most common x0 starting position of words
#    # In a sidebar resume, content lines all start at the same x
#    from collections import Counter
#    x0_counter = Counter(round(w["x0"]) for w in words)
#
#    # Find content start x — the x position where main body text begins.
#    # In a sidebar resume this is where all content lines start consistently.
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
#        inner_gap = sidebar_x0s[i+1] - sidebar_x0s[i]
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
#def _is_true_two_column(words: list, mid_x: float) -> bool:
#    """
#    Determines whether a page genuinely has a two-column layout.
#    """
#    left_words   = [w for w in words if w["x1"] < mid_x - 20]
#    right_words  = [w for w in words if w["x0"] > mid_x + 20]
#    gutter_words = [w for w in words if w["x0"] < mid_x + 20 and w["x1"] > mid_x - 20]
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return False
#    if len(gutter_words) > 5:
#        return False
#
#    if right_words:
#        right_top    = min(w["top"]    for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span   = right_bottom - right_top
#        left_top     = min(w["top"]    for w in left_words)
#        left_bottom  = max(w["bottom"] for w in left_words)
#        left_span    = left_bottom - left_top
#        if left_span > 0 and right_span / left_span < 0.6:
#            return False
#
#    return True
#
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from pdfplumber word dicts.
#    Groups words into lines by vertical position and joins with spaces.
#    """
#    if not words:
#        return ""
#
#    lines = []
#    current_line = []
#    current_top  = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top  = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top  = word["top"]
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
#    current_top  = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top   = word["top"]
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
#            current_top   = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    # Build content lines with vertical positions
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top   = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top   = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top   = word["top"]
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
#def read_text_pdf(pdf_path: str) -> str:
#    """
#    Reads a normal (text-based) PDF using pdfplumber.
#    Handles single-column, two-column, and sidebar layouts.
#    """
#    full_text = []
#
#    with pdfplumber.open(pdf_path) as pdf:
#        for page in pdf.pages:
#            page_width = page.width
#            mid_x      = page_width / 2
#
#            words = page.extract_words(
#                x_tolerance=1.5,
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#
#            if not words:
#                continue
#
#            # ── Check for sidebar layout first ──
#            sidebar_split = _find_sidebar_split(words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#                content_words = [w for w in words if w["x0"] > sidebar_split]
#                text = _merge_sidebar_with_content(sidebar_words, content_words)
#                if text.strip():
#                    full_text.append(text.strip())
#
#            # ── Check for true two-column layout ──
#            elif _is_true_two_column(words, mid_x):
#                left_words  = [w for w in words if w["x1"] <= mid_x]
#                right_words = [w for w in words if w["x0"] >  mid_x]
#                left_text   = _extract_words_to_lines(left_words)
#                right_text  = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    full_text.append(left_text.strip())
#                if right_text.strip():
#                    full_text.append(right_text.strip())
#
#            # ── Single column ──
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













#"""
#PDF Reader — Layer 0
#Reads a PDF file and returns clean text.
#Automatically detects if the PDF is scanned (image-based) or text-based.
#Handles three layout types:
#  1. Single column  — standard reading order
#  2. True two-column — two independent content columns side by side
#  3. Sidebar layout — narrow heading column on left, wide content column on right
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
#    """
#    doc = fitz.open(pdf_path)
#    total_text = ""
#    for page_num in range(min(3, len(doc))):
#        page = doc[page_num]
#        total_text += page.get_text()
#    doc.close()
#    return len(total_text.strip()) < 100
#
#
#def _find_sidebar_split(words: list, page_width: float) -> float:
#    """
#    Detects if a page has a sidebar layout — a narrow left column
#    containing ONLY section headings, and a wide right column with content.
#
#    Returns the x-coordinate of the split point if sidebar detected,
#    or None if not a sidebar layout.
#
#    A sidebar layout has these properties:
#    1. A large gap (> 50 pts) in x0 positions in the left third of the page
#    2. Very few words on the left side of the gap (< 15 words)
#    3. The gap appears consistently (not just on one line)
#    """
#    if not words:
#        return None
#
#    # Only look for sidebar split in left third of page
#    left_threshold = page_width * 0.35
#
#    # Get x0 positions of words in the left third
#    left_x0s = sorted(set(
#        round(w["x0"])
#        for w in words
#        if w["x0"] < left_threshold
#    ))
#
#    if len(left_x0s) < 2:
#        return None
#
#    # Find largest gap in left third
#    best_gap = 0
#    best_split = None
#    for i in range(len(left_x0s) - 1):
#        gap = left_x0s[i+1] - left_x0s[i]
#        if gap > best_gap:
#            best_gap = gap
#            best_split = (left_x0s[i] + left_x0s[i+1]) / 2
#
#    # Gap must be significant (> 40 pts) to be a sidebar.
#    # Lowered from 50 to 40 to catch tighter sidebar layouts
#    # like resumes where heading column ends at x=108 and
#    # content starts at x=154 — a gap of 46 pts.
#    if best_gap < 40 or best_split is None:
#        return None
#
#    # Left side of split must have very few words (just headings)
#    sidebar_words = [w for w in words if w["x0"] < best_split]
#    if len(sidebar_words) > 15:
#        return None
#
#    return best_split
#
#
#def _is_true_two_column(words: list, mid_x: float) -> bool:
#    """
#    Determines whether a page genuinely has a two-column layout.
#    """
#    left_words   = [w for w in words if w["x1"] < mid_x - 20]
#    right_words  = [w for w in words if w["x0"] > mid_x + 20]
#    gutter_words = [w for w in words if w["x0"] < mid_x + 20 and w["x1"] > mid_x - 20]
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return False
#    if len(gutter_words) > 5:
#        return False
#
#    if right_words:
#        right_top    = min(w["top"]    for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span   = right_bottom - right_top
#        left_top     = min(w["top"]    for w in left_words)
#        left_bottom  = max(w["bottom"] for w in left_words)
#        left_span    = left_bottom - left_top
#        if left_span > 0 and right_span / left_span < 0.6:
#            return False
#
#    return True
#
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from pdfplumber word dicts.
#    Groups words into lines by vertical position and joins with spaces.
#    """
#    if not words:
#        return ""
#
#    lines = []
#    current_line = []
#    current_top  = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top  = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top  = word["top"]
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
#    current_top  = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top   = word["top"]
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
#            current_top   = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    # Build content lines with vertical positions
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top   = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top   = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top   = word["top"]
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
#def read_text_pdf(pdf_path: str) -> str:
#    """
#    Reads a normal (text-based) PDF using pdfplumber.
#    Handles single-column, two-column, and sidebar layouts.
#    """
#    full_text = []
#
#    with pdfplumber.open(pdf_path) as pdf:
#        for page in pdf.pages:
#            page_width = page.width
#            mid_x      = page_width / 2
#
#            words = page.extract_words(
#                x_tolerance=1.5,
#                y_tolerance=3,
#                keep_blank_chars=False,
#            )
#
#            if not words:
#                continue
#
#            # ── Check for sidebar layout first ──
#            sidebar_split = _find_sidebar_split(words, page_width)
#            if sidebar_split:
#                sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#                content_words = [w for w in words if w["x0"] > sidebar_split]
#                text = _merge_sidebar_with_content(sidebar_words, content_words)
#                if text.strip():
#                    full_text.append(text.strip())
#
#            # ── Check for true two-column layout ──
#            elif _is_true_two_column(words, mid_x):
#                left_words  = [w for w in words if w["x1"] <= mid_x]
#                right_words = [w for w in words if w["x0"] >  mid_x]
#                left_text   = _extract_words_to_lines(left_words)
#                right_text  = _extract_words_to_lines(right_words)
#                if left_text.strip():
#                    full_text.append(left_text.strip())
#                if right_text.strip():
#                    full_text.append(right_text.strip())
#
#            # ── Single column ──
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