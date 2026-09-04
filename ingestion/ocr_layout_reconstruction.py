"""
ocr_layout_reconstruction.py — Layer 0, layout logic for SCANNED/OCR pages only

WHY THIS FILE IS SEPARATE FROM pdf_reader.py (READ THIS BEFORE EDITING)
-------------------------------------------------------------------------
This file is a DELIBERATE, INTENTIONAL COPY of the layout-detection
logic that also lives inside pdf_reader.py (sidebar / true-two-column /
single-column reading-order reconstruction from word positions). It is
NOT imported by pdf_reader.py, and pdf_reader.py is NOT imported by
this file or by anything that uses this file. The two are fully
independent on purpose.

Why duplicate instead of share: scanned/OCR pages and native PDF text
pages turned out to need DIFFERENT tuning almost immediately. Confirmed
on a real scanned job document (SMFG_Job_Document_Scanned.pdf): a
label/value table ("Job Title" | "Asst. Manager, Cyber Defence") needed
a tighter heading-merge tolerance (0.6x instead of 1.5x -- see REVISION
note below) than resume sidebars did. If this logic were shared with
pdf_reader.py, every future OCR-specific tuning decision would risk
changing behavior on the already-proven native-PDF pipeline (60+
resumes tested), and vice versa. Keeping them separate means:
  - A bug fix or tuning change made HERE, for a scanned-resume problem,
    can NEVER affect pdf_reader.py's native-PDF behavior. Zero risk,
    not "probably fine."
  - pdf_reader.py never needs to be touched, tested, or even re-read
    when fixing a scanned-resume-only issue.
  - The trade-off: if a genuine bug is found in the shared ALGORITHM
    itself (not a tuning number, an actual logic error), it needs to be
    fixed in both places by hand. That's an accepted cost in exchange
    for isolation between two pipelines with different tuning needs.

This file is used by ocr_reader_pytesseract.py ONLY -- never by
pdf_reader.py, and no longer by ocr_reader.py's Surya path either (see
that module's own REVISION notes: Surya's line-granularity boxes are
handled via sort_lines_only(), not this file's word-level logic, and
table detection there now uses table_grid_detector.py directly).

NOTE ON THE PRE-OCR IMAGE QUALITY GATE (blur rejection / deskew /
brightness correction): this file does NOT need any changes for that.
It operates purely on word-position dicts (text/x0/x1/top/bottom/
height) that the OCR modules hand it AFTER preprocessing and OCR have
already happened -- it never sees a raw image or a fitz page.

REVISION -- tightened the sidebar heading-merge tolerance
-----------------------------------------------------------
_merge_sidebar_with_content() groups words into separate heading LINES
by checking the vertical gap between consecutive words: if the gap is
within `word_height * heading_tolerance_mult`, they're treated as the
same line; otherwise a new line starts.

The old value, 1.5, was tuned against resume sidebars, where distinct
section headings ("PROFESSIONAL SUMMARY", "EDUCATION", ...) normally
have generous vertical space between them. Confirmed on the SMFG
document's Job Title/Department/Grade/IC-PM table: row-to-row spacing
there (35-41px) is SMALLER than 1.5x a single row's own word-height
(52.5px), so the old tolerance merged all 4 separate table rows into
one garbled heading blob instead of keeping them apart -- verified by
manually tracing the merge loop against the real OCR'd coordinates.

Lowered to 0.6 (verified against the same real coordinates). 0.6 was
chosen deliberately just above the 0.5 used for CONTENT line grouping
elsewhere in this file, but well below the old 1.5.

IMPORTANT FOR PRODUCTION ROLLOUT: this tolerance change affects every
sidebar-layout page. It was only tested against the SMFG table's real
coordinates, not re-run against your full 60+ resume regression set.

REVISION -- genuine multi-column table detector (_reconstruct_table)
-----------------------------------------------------------------
[Unchanged from before -- see _reconstruct_table()'s own docstring
below for the full history of this function.]

REVISION (this version) — reconstruct_page_text() can now skip its
own internal table detection entirely
-----------------------------------------------------------------
_reconstruct_table() (used internally by reconstruct_page_text() as
its first-resort check) guesses table structure from GAPS between
word positions. This turned out to have a real, confirmed failure mode
beyond the table case it was built for: on a page with NO genuine
bordered table but a two-column LAYOUT (e.g. a resume with a "Skills"
sidebar next to unrelated "Work Experience" text), the gap between the
left column's text and the right column's text can look exactly like
a table-column gap, especially once several rows happen to align at
similar heights. Confirmed on a real scanned resume (Anmol Srivastava,
Anmol_Srivastava_Mobile_App_Developer_GT_page-0001.pdf): unrelated
left-column and right-column lines (e.g. the candidate's name next to
their phone number, and later, an achievement bullet next to unrelated
"Skills" chip text) were getting tab-joined together as if they were
one table row -- corrupting completely unrelated content into a single
merged line.

ingestion/table_grid_detector.py's detect_and_ocr_tables() now handles
genuine bordered tables far more reliably, by finding the table's
actual printed grid lines in the page image directly, instead of
guessing from word gaps (see that module's docstring for the full
reasoning and verification notes). Once a caller has already run that
detector and merged any genuine table's text in via
merge_tables_into_words() (see ocr_reader_pytesseract.py), there is no
reason for reconstruct_page_text() to ALSO attempt its own, less
reliable, word-gap-based table guess on the same words -- doing so is
not just redundant, it is actively harmful on borderless two-column
layouts, per the confirmed bug above.

reconstruct_page_text() now accepts an optional skip_table_detection
parameter (default False, so any caller that doesn't pass it keeps the
EXACT previous behavior -- this change is 100% backward compatible).
ocr_reader_pytesseract.py passes skip_table_detection=True at both of
its call sites, since it always runs the image-based table detector
first. _reconstruct_table() itself is UNCHANGED and still fully
functional -- it's simply no longer invoked automatically by
reconstruct_page_text() when the caller opts out.
"""

from collections import Counter


# ──────────────────────────────────────────────────────────────
# MULTI-COLUMN TABLE DETECTION (superseded as the primary strategy --
# see module docstring's latest REVISION note. Left in place, fully
# functional, for any caller that still wants word-gap-based table
# guessing -- currently no caller does by default.)
# ──────────────────────────────────────────────────────────────

_TABLE_MIN_COLUMNS = 3            # fewer than this isn't a "table" in the sense this
                                   # detector exists for -- 2 columns is already handled
                                   # by the existing two-column detector below
_TABLE_MIN_GAP = 22               # points; horizontal gap (word.x1 to next word.x0) on
                                   # the SAME row needed to call it a cell boundary,
                                   # rather than normal same-cell word spacing. NOTE:
                                   # this is the single most important knob to re-check
                                   # against REAL OCR coordinates from your pipeline --
                                   # this was only validated against an approximate,
                                   # hand-estimated reconstruction of a real table's
                                   # layout (word positions estimated from the PDF image,
                                   # not actual Surya/tesseract bounding boxes), so treat
                                   # this exact number as a starting point, not a
                                   # calibrated constant like MIN_BLUR_VARIANCE was.
_TABLE_CLUSTER_TOLERANCE = 30     # points; how close two rows' cell-start x0 positions
                                   # need to be to count as "the same column" (absorbs
                                   # OCR jitter between rows)
_TABLE_MIN_ROWS_PER_COLUMN = 2    # a column cluster supported by fewer rows than this is
                                   # more likely coincidence than a real repeated column
_TABLE_MIN_ROWS_WITH_MULTIPLE_COLUMNS = 3  # how many rows must show 2+ cells landing in
                                            # 2+ different column clusters before this is
                                            # trusted as a real table, not a false positive
_TABLE_ROW_TOLERANCE_MULT = 0.6   # slightly more forgiving than the 0.5 used for plain
                                   # content lines elsewhere in this file -- OCR baselines
                                   # across different columns in the SAME row tend to
                                   # jitter a touch more than words within one column


def _group_words_into_rows(words: list, tolerance_mult: float = _TABLE_ROW_TOLERANCE_MULT) -> list:
    """
    Groups words into visual rows by vertical (top) position, same idea
    as _extract_words_to_lines()'s line-grouping but returning the raw
    word groups (not joined text) so _reconstruct_table() can split
    each row into cells afterward.
    """
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    rows = []
    current_top = None
    current_row = []
    for word in sorted_words:
        if current_top is None:
            current_top = word["top"]
            current_row = [word]
            continue
        tolerance = max(word["height"], 1) * tolerance_mult
        if abs(word["top"] - current_top) <= tolerance:
            current_row.append(word)
        else:
            rows.append(current_row)
            current_row = [word]
            current_top = word["top"]
    if current_row:
        rows.append(current_row)
    return rows


def _split_row_into_cells(row_words: list, min_gap: float = _TABLE_MIN_GAP) -> list:
    """
    Splits ONE row's words into cells using LOCAL horizontal gaps within
    that row only -- this is the key fix over the rejected flat-histogram
    approach (see module docstring): a multi-word cell like "Madurai
    kamaraj V.V.M College" has small gaps between ITS OWN words, so it
    stays one cell, regardless of what any other row's words look like.
    A gap >= min_gap between one word's right edge and the next word's
    left edge, on the SAME row, marks a new cell.

    Returns a list of cells, each a list of word dicts (still in
    left-to-right order).
    """
    if not row_words:
        return []
    sorted_words = sorted(row_words, key=lambda w: w["x0"])
    cells = [[sorted_words[0]]]
    for prev, curr in zip(sorted_words, sorted_words[1:]):
        gap = curr["x0"] - prev["x1"]
        if gap >= min_gap:
            cells.append([curr])
        else:
            cells[-1].append(curr)
    return cells


def _cluster_cell_starts(cell_starts: list, tolerance: float = _TABLE_CLUSTER_TOLERANCE) -> list:
    """
    Greedily clusters cell-start x0 positions (collected across ALL
    rows) into column groups: values within `tolerance` points of the
    previous cluster's last value are merged into it. This is what
    lets row 1's "M.COM" cell (starting at, say, x=40) and row 2's
    "B.COM" cell (starting at x=42) be recognized as the same column
    despite small OCR-jitter differences in exact position.

    Returns a sorted list of cluster center x0 positions.
    """
    if not cell_starts:
        return []
    sorted_starts = sorted(cell_starts)
    clusters = [[sorted_starts[0]]]
    for x in sorted_starts[1:]:
        if x - clusters[-1][-1] <= tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [sum(c) / len(c) for c in clusters]


def _nearest_cluster_index(x0: float, cluster_centers: list) -> int:
    return min(range(len(cluster_centers)), key=lambda i: abs(cluster_centers[i] - x0))


def _reconstruct_table(words: list, page_width: float):
    """
    Detects and reconstructs a genuine multi-column (N >= 3) grid table
    purely from word-position gaps. SUPERSEDED as the default/automatic
    strategy by table_grid_detector.py's image-based detection (see
    module docstring's latest REVISION note) -- kept here, fully
    functional and unchanged, in case it's ever useful as an additional
    fallback, but reconstruct_page_text() no longer calls it
    automatically when skip_table_detection=True is passed.

    Returns the reconstructed text (one row per line, cells separated
    by a tab -- same convention docx_reader.py uses for table rows) if
    a genuine table is detected, or None if this page doesn't look like
    one.
    """
    rows = _group_words_into_rows(words)
    if len(rows) < 2:
        return None

    row_cells = [_split_row_into_cells(row) for row in rows]

    all_cell_starts = [
        min(w["x0"] for w in cell)
        for cells in row_cells for cell in cells
    ]
    cluster_centers = _cluster_cell_starts(all_cell_starts)
    if len(cluster_centers) < _TABLE_MIN_COLUMNS:
        return None

    row_cluster_cells = []  # per row: {cluster_index: [word,...]}
    cluster_row_support = [0] * len(cluster_centers)
    for cells in row_cells:
        cluster_map = {}
        seen_clusters_this_row = set()
        for cell in cells:
            start_x0 = min(w["x0"] for w in cell)
            ci = _nearest_cluster_index(start_x0, cluster_centers)
            cluster_map.setdefault(ci, []).extend(cell)
            seen_clusters_this_row.add(ci)
        for ci in seen_clusters_this_row:
            cluster_row_support[ci] += 1
        row_cluster_cells.append(cluster_map)

    valid_cluster_indices = {
        ci for ci, support in enumerate(cluster_row_support)
        if support >= _TABLE_MIN_ROWS_PER_COLUMN
    }
    if len(valid_cluster_indices) < _TABLE_MIN_COLUMNS:
        return None

    multi_cell_rows = sum(
        1 for cluster_map in row_cluster_cells
        if len(set(cluster_map.keys()) & valid_cluster_indices) >= 2
    )
    if multi_cell_rows < _TABLE_MIN_ROWS_WITH_MULTIPLE_COLUMNS:
        return None

    ordered_valid_clusters = sorted(valid_cluster_indices, key=lambda ci: cluster_centers[ci])

    result_lines = []
    for cluster_map in row_cluster_cells:
        cells_text = []
        for ci in ordered_valid_clusters:
            cell_words = cluster_map.get(ci, [])
            cell_words_sorted = sorted(cell_words, key=lambda w: w["x0"])
            cells_text.append(" ".join(w["text"] for w in cell_words_sorted))
        while cells_text and cells_text[-1] == "":
            cells_text.pop()
        if any(c.strip() for c in cells_text):
            result_lines.append("\t".join(cells_text))

    return "\n".join(result_lines) if result_lines else None


# ──────────────────────────────────────────────────────────────
# SIDEBAR LAYOUT DETECTION
# ──────────────────────────────────────────────────────────────

def _find_sidebar_split(words: list, page_width: float):
    """
    Detects if a page has a sidebar layout — a narrow left column
    containing ONLY section headings, and a wide right column with content.
    Returns the x-coordinate of the split point if sidebar detected,
    or None if not a sidebar layout.
    """
    if not words:
        return None

    x0_counter = Counter(round(w["x0"]) for w in words)
    content_start_x = None
    best_count = 0

    for x0, count in x0_counter.items():
        if 80 < x0 < page_width * 0.45 and count > best_count:
            best_count = count
            content_start_x = x0

    if content_start_x is None or best_count < 3:
        return None

    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]

    if not sidebar_candidates:
        return None
    if len(sidebar_candidates) > 25:
        return None

    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
    heading_boundary = sidebar_x0s[-1]
    for i in range(len(sidebar_x0s) - 1):
        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
        if inner_gap > 30:
            heading_boundary = sidebar_x0s[i]
            break

    gap = content_start_x - heading_boundary
    if gap < 30:
        return None

    split = (heading_boundary + content_start_x) / 2
    return split


# Tuned per the REVISION note above. Kept as a named constant (not a
# magic number inline) specifically so a future regression against
# resume sidebars can dial this back without hunting through the merge
# function body.
_SIDEBAR_HEADING_TOLERANCE_MULT = 0.6


def _merge_sidebar_with_content(sidebar_words: list, content_words: list,
                                 heading_tolerance_mult: float = _SIDEBAR_HEADING_TOLERANCE_MULT) -> str:
    """
    Merges sidebar heading words with content words by vertical position.
    """
    heading_lines = {}
    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))

    current_top = None
    current_words = []
    for word in sorted_sidebar:
        if current_top is None:
            current_top = word["top"]
            current_words = [word]
        elif abs(word["top"] - current_top) <= max(word["height"], 1) * heading_tolerance_mult:
            current_words.append(word)
            current_top = word["top"]
        else:
            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
            heading_lines[round(current_top)] = heading_text
            current_top = word["top"]
            current_words = [word]
    if current_words:
        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
        heading_lines[round(current_top)] = heading_text

    content_line_list = []
    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))

    current_top = None
    current_words = []
    for word in sorted_content:
        if current_top is None:
            current_top = word["top"]
            current_words = [word]
        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
            current_words.append(word)
        else:
            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
            content_line_list.append((round(current_top), line_text))
            current_top = word["top"]
            current_words = [word]
    if current_words:
        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
        content_line_list.append((round(current_top), line_text))

    result_lines = []
    used_headings = set()

    for content_top, content_text in content_line_list:
        for heading_top, heading_text in sorted(heading_lines.items()):
            if heading_top in used_headings:
                continue
            if abs(heading_top - content_top) <= 20:
                result_lines.append(heading_text)
                used_headings.add(heading_top)
                break
        result_lines.append(content_text)

    for heading_top, heading_text in sorted(heading_lines.items()):
        if heading_top not in used_headings:
            result_lines.append(heading_text)

    return "\n".join(result_lines)


# ──────────────────────────────────────────────────────────────
# TWO-COLUMN LAYOUT DETECTION
# ──────────────────────────────────────────────────────────────

def _find_column_split(words: list, page_width: float):
    """
    Finds the actual column split point for a two-column layout, using
    the gap in x0 start positions rather than assuming the page midpoint.
    """
    if not words:
        return None

    x0_counter = Counter(round(w["x0"]) for w in words)
    all_x0 = sorted(x0_counter.keys())

    if len(all_x0) < 2:
        return None

    best_split = None
    best_score = 0

    for i in range(len(all_x0) - 1):
        gap = all_x0[i + 1] - all_x0[i]
        if gap < 15:
            continue

        split_x = (all_x0[i] + all_x0[i + 1]) / 2
        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])

        if left_count < 20 or right_count < 20:
            continue

        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
            continue

        balance = min(left_count, right_count) / max(left_count, right_count)
        score = gap * balance

        if score > best_score:
            best_score = score
            best_split = split_x

    if best_split is None:
        return None

    left_words = [w for w in words if w["x1"] <= best_split]
    right_words = [w for w in words if w["x0"] > best_split]
    return best_split, left_words, right_words


def _is_true_two_column(words: list, page_width: float):
    """
    Determines whether a page genuinely has a two-column layout, AND
    returns the correctly split word groups if so.
    Requires: both sides substantial (>=30 words), near-empty gutter
    (<5 straddling words), right column spans >=15% of left column's
    vertical height.
    """
    result = _find_column_split(words, page_width)
    if result is None:
        return None

    split_x, left_words, right_words = result

    if len(left_words) < 30 or len(right_words) < 30:
        return None

    gutter_words = [
        w for w in words
        if w["x0"] < split_x - 2 and w["x1"] > split_x + 2
    ]
    if len(gutter_words) > 5:
        return None

    if right_words and left_words:
        right_top = min(w["top"] for w in right_words)
        right_bottom = max(w["bottom"] for w in right_words)
        right_span = right_bottom - right_top

        left_top = min(w["top"] for w in left_words)
        left_bottom = max(w["bottom"] for w in left_words)
        left_span = left_bottom - left_top

        if left_span > 0 and right_span / left_span < 0.15:
            return None

    return left_words, right_words


# ──────────────────────────────────────────────────────────────
# LINE RECONSTRUCTION
# ──────────────────────────────────────────────────────────────

def _extract_words_to_lines(words: list) -> str:
    """
    Reconstructs text lines from word dicts, grouping by vertical (top)
    position and joining with a single space. Works identically whether
    a word came from pdfplumber's native extraction, from
    ocr_region_to_words() (hybrid-page image regions), or from a
    full-page OCR engine's word/line output -- all use the same dict
    shape.
    """
    if not words:
        return ""

    lines: list = []
    current_line: list = []
    current_top = None

    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))

    for word in sorted_words:
        if current_top is None:
            current_top = word["top"]
            current_line = [word]
            continue

        line_tolerance = max(word["height"], 1) * 0.5
        if abs(word["top"] - current_top) <= line_tolerance:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
            current_top = word["top"]

    if current_line:
        lines.append(current_line)

    text_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["x0"])
        parts = [w["text"] for w in line_sorted]
        text_lines.append(" ".join(parts))

    return "\n".join(text_lines)

# ──────────────────────────────────────────────────────────────
# LINE-GRANULARITY RECONSTRUCTION -- Surya only (see REVISION note above)
# ──────────────────────────────────────────────────────────────
#added this sort_lines function manually because getting error pasted on chatgpt

def sort_lines_only(lines: list) -> str:
    """
    For already-line-granularity OCR data (Surya's TextLine output,
    where each item is a whole line of recognized text, not a single
    word): sorts by vertical position (top), tie-broken by horizontal
    position (x0), and outputs one line per input item -- NO merging,
    NO sidebar/two-column detection.

    Unlike _extract_words_to_lines() / reconstruct_page_text(), this
    never combines two different input items onto one output line.
    That's the point: at this granularity every input item already IS
    a complete line, so there is nothing to assemble, and attempting to
    (as reconstruct_page_text() does) risks merging two genuinely
    different lines together if their measured vertical positions
    happen to land close to each other -- confirmed in production on a
    real resume (see module docstring's REVISION note).

    Trade-off, accepted deliberately: a short heading whose Surya
    bounding box is very slightly mismeasured can land one line off
    from its ideal position (cosmetic). In exchange, this guarantees a
    single-column page's lines never get scrambled together, which is
    the far more common and far more damaging failure mode this was
    built to eliminate.

    lines: list of dicts with keys "text", "x0", "top" (and any others
      -- only these three are used). Returns "" for an empty list.
    """
    if not lines:
        return ""
    sorted_lines = sorted(lines, key=lambda w: (round(w["top"], 1), w["x0"]))
    return "\n".join(l["text"] for l in sorted_lines)


# ──────────────────────────────────────────────────────────────
# SHARED ENTRY POINT — used by ocr_reader_pytesseract.py
# ──────────────────────────────────────────────────────────────

def reconstruct_page_text(words: list, page_width: float, skip_table_detection: bool = False) -> str:
    """
    Given a flat list of word dicts (keys: text, x0, x1, top, bottom,
    height) for ONE page, reconstructs reading-order text using a
    table -> sidebar -> two-column -> single-column fallback chain.

    skip_table_detection: if True, skips this function's own internal
      word-gap-based _reconstruct_table() check entirely and goes
      straight to sidebar -> two-column -> single-column. Default False
      preserves the exact previous behavior for any existing caller.

      Pass True when the caller has ALREADY run
      table_grid_detector.py's detect_and_ocr_tables() (image-based,
      far more reliable -- see that module's docstring) and merged any
      genuine table's text in separately via merge_tables_into_words().
      In that case, letting this function ALSO attempt its own,
      less-reliable word-gap table guess on the same words is not just
      redundant, it's actively harmful: confirmed on a real scanned
      resume with a borderless two-column layout (Anmol Srivastava),
      the word-gap heuristic mistook the gap between the left and
      right columns for table-column boundaries and tab-merged
      unrelated left/right content together (e.g. the candidate's name
      merged with their phone number on one line). See this module's
      own top-of-file REVISION note for the full story.

      ocr_reader_pytesseract.py -- the only current caller of this
      function -- always passes skip_table_detection=True.

    Table detection (when not skipped) runs FIRST because it's the
    most specific, structured shape -- a genuine N-column grid should
    never be mistaken for the looser sidebar or two-column shapes tried
    after it.

    Returns "" for an empty words list so callers can safely skip
    appending it.
    """
    if not words:
        return ""

    if not skip_table_detection:
        table_text = _reconstruct_table(words, page_width)
        if table_text:
            return table_text

    sidebar_split = _find_sidebar_split(words, page_width)
    if sidebar_split:
        sidebar_words = [w for w in words if w["x0"] < sidebar_split]
        content_words = [w for w in words if w["x0"] > sidebar_split]
        text = _merge_sidebar_with_content(sidebar_words, content_words)
        if text.strip():
            return text

    two_col_result = _is_true_two_column(words, page_width)
    if two_col_result:
        left_words, right_words = two_col_result
        left_text = _extract_words_to_lines(left_words)
        right_text = _extract_words_to_lines(right_words)
        parts = [t for t in (left_text.strip(), right_text.strip()) if t]
        if parts:
            return "\n\n".join(parts)

    return _extract_words_to_lines(words)

















#"""
#ocr_layout_reconstruction.py — Layer 0, layout logic for SCANNED/OCR pages only
#
#WHY THIS FILE IS SEPARATE FROM pdf_reader.py (READ THIS BEFORE EDITING)
#-------------------------------------------------------------------------
#This file is a DELIBERATE, INTENTIONAL COPY of the layout-detection
#logic that also lives inside pdf_reader.py (sidebar / true-two-column /
#single-column reading-order reconstruction from word positions). It is
#NOT imported by pdf_reader.py, and pdf_reader.py is NOT imported by
#this file or by anything that uses this file. The two are fully
#independent on purpose.
#
#Why duplicate instead of share: scanned/OCR pages and native PDF text
#pages turned out to need DIFFERENT tuning almost immediately. Confirmed
#on a real scanned job document (SMFG_Job_Document_Scanned.pdf): a
#label/value table ("Job Title" | "Asst. Manager, Cyber Defence") needed
#a tighter heading-merge tolerance (0.6x instead of 1.5x -- see REVISION
#note below) than resume sidebars did. If this logic were shared with
#pdf_reader.py, every future OCR-specific tuning decision would risk
#changing behavior on the already-proven native-PDF pipeline (60+
#resumes tested), and vice versa. Keeping them separate means:
#  - A bug fix or tuning change made HERE, for a scanned-resume problem,
#    can NEVER affect pdf_reader.py's native-PDF behavior. Zero risk,
#    not "probably fine."
#  - pdf_reader.py never needs to be touched, tested, or even re-read
#    when fixing a scanned-resume-only issue.
#  - The trade-off: if a genuine bug is found in the shared ALGORITHM
#    itself (not a tuning number, an actual logic error), it needs to be
#    fixed in both places by hand. That's an accepted cost in exchange
#    for isolation between two pipelines with different tuning needs.
#
#This file is used by ocr_reader.py (Surya) and ocr_reader_pytesseract.py
#(tesseract fallback) ONLY -- never by pdf_reader.py.
#
#NOTE ON THE PRE-OCR IMAGE QUALITY GATE (blur rejection / deskew /
#brightness correction): this file does NOT need any changes for that.
#It operates purely on word-position dicts (text/x0/x1/top/bottom/
#height) that the OCR modules hand it AFTER preprocessing and OCR have
#already happened -- it never sees a raw image or a fitz page.
#
#REVISION -- tightened the sidebar heading-merge tolerance
#-----------------------------------------------------------
#_merge_sidebar_with_content() groups words into separate heading LINES
#by checking the vertical gap between consecutive words: if the gap is
#within `word_height * heading_tolerance_mult`, they're treated as the
#same line; otherwise a new line starts.
#
#The old value, 1.5, was tuned against resume sidebars, where distinct
#section headings ("PROFESSIONAL SUMMARY", "EDUCATION", ...) normally
#have generous vertical space between them. Confirmed on the SMFG
#document's Job Title/Department/Grade/IC-PM table: row-to-row spacing
#there (35-41px) is SMALLER than 1.5x a single row's own word-height
#(52.5px), so the old tolerance merged all 4 separate table rows into
#one garbled heading blob instead of keeping them apart -- verified by
#manually tracing the merge loop against the real OCR'd coordinates.
#
#Lowered to 0.6 (verified against the same real coordinates). 0.6 was
#chosen deliberately just above the 0.5 used for CONTENT line grouping
#elsewhere in this file, but well below the old 1.5.
#
#IMPORTANT FOR PRODUCTION ROLLOUT: this tolerance change affects every
#sidebar-layout page. It was only tested against the SMFG table's real
#coordinates, not re-run against your full 60+ resume regression set.
#
#REVISION (this version) — genuine multi-column table detector
#-----------------------------------------------------------------
#Confirmed on a real scanned resume (Anandu Chandran): an education
#table with FOUR columns (Course | University/Board | College | % of
#mark), including a wrapped two-line column header ("University/" /
#"Board" on separate lines), came out with headers and row data
#completely interleaved -- e.g. "Course" followed immediately by
#"University/ Name of % of mark" followed by "Board college" followed
#by "M.COM" etc.
#
#Root cause: the SMFG fix above only handles a 2-COLUMN label/value
#shape by squeezing it into the existing sidebar detector (one narrow
#column + one wide column). It does not, and structurally cannot,
#handle a genuine N-column grid (N >= 3) with a header row, because
#neither the sidebar detector (needs exactly one narrow heading column)
#nor the two-column detector (needs exactly one gap, i.e. exactly two
#zones) can represent more than 2 zones. Before this revision, a table
#like this fell all the way through to the flat top-to-bottom
#_extract_words_to_lines() fallback, which sorts purely by vertical
#position -- and once one cell's text wraps to a second line while a
#neighboring cell's doesn't, that fallback interleaves cells from
#different rows and columns, exactly matching the scrambled output seen
#on the real resume.
#
#FIRST ATTEMPT (rejected, not shipped): clustering column boundaries
#from a flat, page-wide histogram of every individual word's x0 (the
#same style _find_column_split already uses for the 2-column case).
#Tested against a reconstruction of the real broken layout and it
#failed -- a multi-word cell like "Madurai kamaraj V.V.M College"
#contributes several DIFFERENT x0 values that don't line up with any
#other row's words at all, so the flat histogram sees a large number of
#small, spurious gaps instead of a small number of real, repeated
#column-start positions. The row structure itself carries the signal
#that a flat, order-blind histogram throws away.
#
#ACTUAL APPROACH (implemented below): row-first, not histogram-first.
#  1. Group words into rows (_group_words_into_rows), same as the rest
#     of this file already does.
#  2. Within EACH row independently, split that row's words into cells
#     using LOCAL horizontal gaps between consecutive words (sorted
#     left-to-right): a gap >= _TABLE_MIN_GAP marks a cell boundary,
#     anything smaller is normal same-cell word spacing
#     (_split_row_into_cells). This is the fix for the rejected
#     approach above -- gaps are measured within one row at a time, so
#     a multi-word cell's internal word spacing never gets compared
#     against unrelated words from other rows.
#  3. Collect every row's cell START x0 positions across the WHOLE
#     page, then greedily cluster nearby start positions together
#     (_cluster_cell_starts) -- e.g. row 1's "M.COM" cell starting at
#     x=40 and row 2's "B.COM" cell starting at x=42 land in the same
#     cluster despite the 2pt OCR jitter. Column count is simply the
#     number of resulting clusters.
#  4. Requires N >= _TABLE_MIN_COLUMNS clusters, each supported by
#     enough distinct rows, and requires several rows to have 2+ cells
#     landing in 2+ DIFFERENT clusters simultaneously -- the real
#     signature of a table (independent short cells side by side on one
#     row) as opposed to ordinary wrapped prose (one long flowing line
#     that happens to span a wide x-range).
#  5. If validated, reconstructs one line per row with cells in column
#     order separated by a tab -- matching the convention docx_reader.py
#     already uses for table rows.
#
#VERIFIED: this version (row-first, cluster-based) was tested against a
#reconstruction of the real Anandu Chandran education-table coordinates
#and correctly separates the 4 columns and merges the wrapped 2-line
#header ("University/" / "Board") into its own column, instead of
#interleaving header and row data. NOT yet re-run against a broader
#resume set. As with the SMFG tolerance change above, re-run your test
#batch after adopting this and specifically spot-check any resumes that
#contain tables, to confirm ordinary bulleted/prose pages aren't being
#mistakenly detected as tables. If a false positive shows up, the knobs
#to tighten first are _TABLE_MIN_GAP (raise it) and
#_TABLE_MIN_ROWS_WITH_MULTIPLE_COLUMNS (raise it) -- both make the
#detector stricter about what counts as "genuinely tabular."
#"""
#
#from collections import Counter
#
#
## ──────────────────────────────────────────────────────────────
## MULTI-COLUMN TABLE DETECTION (NEW)
## ──────────────────────────────────────────────────────────────
#
#_TABLE_MIN_COLUMNS = 3            # fewer than this isn't a "table" in the sense this
#                                   # detector exists for -- 2 columns is already handled
#                                   # by the existing two-column detector below
#_TABLE_MIN_GAP = 22               # points; horizontal gap (word.x1 to next word.x0) on
#                                   # the SAME row needed to call it a cell boundary,
#                                   # rather than normal same-cell word spacing. NOTE:
#                                   # this is the single most important knob to re-check
#                                   # against REAL OCR coordinates from your pipeline --
#                                   # this was only validated against an approximate,
#                                   # hand-estimated reconstruction of a real table's
#                                   # layout (word positions estimated from the PDF image,
#                                   # not actual Surya/tesseract bounding boxes), so treat
#                                   # this exact number as a starting point, not a
#                                   # calibrated constant like MIN_BLUR_VARIANCE was.
#_TABLE_CLUSTER_TOLERANCE = 30     # points; how close two rows' cell-start x0 positions
#                                   # need to be to count as "the same column" (absorbs
#                                   # OCR jitter between rows)
#_TABLE_MIN_ROWS_PER_COLUMN = 2    # a column cluster supported by fewer rows than this is
#                                   # more likely coincidence than a real repeated column
#_TABLE_MIN_ROWS_WITH_MULTIPLE_COLUMNS = 3  # how many rows must show 2+ cells landing in
#                                            # 2+ different column clusters before this is
#                                            # trusted as a real table, not a false positive
#_TABLE_ROW_TOLERANCE_MULT = 0.6   # slightly more forgiving than the 0.5 used for plain
#                                   # content lines elsewhere in this file -- OCR baselines
#                                   # across different columns in the SAME row tend to
#                                   # jitter a touch more than words within one column
#
#
#def _group_words_into_rows(words: list, tolerance_mult: float = _TABLE_ROW_TOLERANCE_MULT) -> list:
#    """
#    Groups words into visual rows by vertical (top) position, same idea
#    as _extract_words_to_lines()'s line-grouping but returning the raw
#    word groups (not joined text) so _reconstruct_table() can split
#    each row into cells afterward.
#    """
#    if not words:
#        return []
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#    rows = []
#    current_top = None
#    current_row = []
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_row = [word]
#            continue
#        tolerance = max(word["height"], 1) * tolerance_mult
#        if abs(word["top"] - current_top) <= tolerance:
#            current_row.append(word)
#        else:
#            rows.append(current_row)
#            current_row = [word]
#            current_top = word["top"]
#    if current_row:
#        rows.append(current_row)
#    return rows
#
#
#def _split_row_into_cells(row_words: list, min_gap: float = _TABLE_MIN_GAP) -> list:
#    """
#    Splits ONE row's words into cells using LOCAL horizontal gaps within
#    that row only -- this is the key fix over the rejected flat-histogram
#    approach (see module docstring): a multi-word cell like "Madurai
#    kamaraj V.V.M College" has small gaps between ITS OWN words, so it
#    stays one cell, regardless of what any other row's words look like.
#    A gap >= min_gap between one word's right edge and the next word's
#    left edge, on the SAME row, marks a new cell.
#
#    Returns a list of cells, each a list of word dicts (still in
#    left-to-right order).
#    """
#    if not row_words:
#        return []
#    sorted_words = sorted(row_words, key=lambda w: w["x0"])
#    cells = [[sorted_words[0]]]
#    for prev, curr in zip(sorted_words, sorted_words[1:]):
#        gap = curr["x0"] - prev["x1"]
#        if gap >= min_gap:
#            cells.append([curr])
#        else:
#            cells[-1].append(curr)
#    return cells
#
#
#def _cluster_cell_starts(cell_starts: list, tolerance: float = _TABLE_CLUSTER_TOLERANCE) -> list:
#    """
#    Greedily clusters cell-start x0 positions (collected across ALL
#    rows) into column groups: values within `tolerance` points of the
#    previous cluster's last value are merged into it. This is what
#    lets row 1's "M.COM" cell (starting at, say, x=40) and row 2's
#    "B.COM" cell (starting at x=42) be recognized as the same column
#    despite small OCR-jitter differences in exact position.
#
#    Returns a sorted list of cluster center x0 positions.
#    """
#    if not cell_starts:
#        return []
#    sorted_starts = sorted(cell_starts)
#    clusters = [[sorted_starts[0]]]
#    for x in sorted_starts[1:]:
#        if x - clusters[-1][-1] <= tolerance:
#            clusters[-1].append(x)
#        else:
#            clusters.append([x])
#    return [sum(c) / len(c) for c in clusters]
#
#
#def _nearest_cluster_index(x0: float, cluster_centers: list) -> int:
#    return min(range(len(cluster_centers)), key=lambda i: abs(cluster_centers[i] - x0))
#
#
#def _reconstruct_table(words: list, page_width: float):
#    """
#    Detects and reconstructs a genuine multi-column (N >= 3) grid table,
#    using a row-first, cell-clustering approach (see module docstring's
#    REVISION note for why this replaced an earlier flat-histogram
#    attempt that didn't actually work).
#
#    Returns the reconstructed text (one row per line, cells separated
#    by a tab -- same convention docx_reader.py uses for table rows) if
#    a genuine table is detected, or None if this page doesn't look like
#    one, so the caller falls through to the existing sidebar ->
#    two-column -> single-column chain unchanged.
#    """
#    rows = _group_words_into_rows(words)
#    if len(rows) < 2:
#        return None
#
#    row_cells = [_split_row_into_cells(row) for row in rows]
#
#    all_cell_starts = [
#        min(w["x0"] for w in cell)
#        for cells in row_cells for cell in cells
#    ]
#    cluster_centers = _cluster_cell_starts(all_cell_starts)
#    if len(cluster_centers) < _TABLE_MIN_COLUMNS:
#        return None
#
#    # Assign each cell to its nearest column cluster, and track which
#    # rows use which clusters, so we can reject clusters that only ever
#    # appear once (more likely noise than a real recurring column) and
#    # reject the whole thing if too few rows show genuine multi-column
#    # structure (the real signature of a table vs. ordinary prose).
#    row_cluster_cells = []  # per row: {cluster_index: [word,...]}
#    cluster_row_support = [0] * len(cluster_centers)
#    for cells in row_cells:
#        cluster_map = {}
#        seen_clusters_this_row = set()
#        for cell in cells:
#            start_x0 = min(w["x0"] for w in cell)
#            ci = _nearest_cluster_index(start_x0, cluster_centers)
#            cluster_map.setdefault(ci, []).extend(cell)
#            seen_clusters_this_row.add(ci)
#        for ci in seen_clusters_this_row:
#            cluster_row_support[ci] += 1
#        row_cluster_cells.append(cluster_map)
#
#    valid_cluster_indices = {
#        ci for ci, support in enumerate(cluster_row_support)
#        if support >= _TABLE_MIN_ROWS_PER_COLUMN
#    }
#    if len(valid_cluster_indices) < _TABLE_MIN_COLUMNS:
#        return None
#
#    multi_cell_rows = sum(
#        1 for cluster_map in row_cluster_cells
#        if len(set(cluster_map.keys()) & valid_cluster_indices) >= 2
#    )
#    if multi_cell_rows < _TABLE_MIN_ROWS_WITH_MULTIPLE_COLUMNS:
#        return None
#
#    ordered_valid_clusters = sorted(valid_cluster_indices, key=lambda ci: cluster_centers[ci])
#
#    result_lines = []
#    for cluster_map in row_cluster_cells:
#        cells_text = []
#        for ci in ordered_valid_clusters:
#            cell_words = cluster_map.get(ci, [])
#            cell_words_sorted = sorted(cell_words, key=lambda w: w["x0"])
#            cells_text.append(" ".join(w["text"] for w in cell_words_sorted))
#        while cells_text and cells_text[-1] == "":
#            cells_text.pop()
#        if any(c.strip() for c in cells_text):
#            result_lines.append("\t".join(cells_text))
#
#    return "\n".join(result_lines) if result_lines else None
#
#
## ──────────────────────────────────────────────────────────────
## SIDEBAR LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_sidebar_split(words: list, page_width: float):
#    """
#    Detects if a page has a sidebar layout — a narrow left column
#    containing ONLY section headings, and a wide right column with content.
#    Returns the x-coordinate of the split point if sidebar detected,
#    or None if not a sidebar layout.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    content_start_x = None
#    best_count = 0
#
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    if content_start_x is None or best_count < 3:
#        return None
#
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#    if len(sidebar_candidates) > 25:
#        return None
#
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#    heading_boundary = sidebar_x0s[-1]
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
## Tuned per the REVISION note above. Kept as a named constant (not a
## magic number inline) specifically so a future regression against
## resume sidebars can dial this back without hunting through the merge
## function body.
#_SIDEBAR_HEADING_TOLERANCE_MULT = 0.6
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list,
#                                 heading_tolerance_mult: float = _SIDEBAR_HEADING_TOLERANCE_MULT) -> str:
#    """
#    Merges sidebar heading words with content words by vertical position.
#    """
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * heading_tolerance_mult:
#            current_words.append(word)
#            current_top = word["top"]
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#        result_lines.append(content_text)
#
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
## ──────────────────────────────────────────────────────────────
## TWO-COLUMN LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_column_split(words: list, page_width: float):
#    """
#    Finds the actual column split point for a two-column layout, using
#    the gap in x0 start positions rather than assuming the page midpoint.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    best_split = None
#    best_score = 0
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        if left_count < 20 or right_count < 20:
#            continue
#
#        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
#            continue
#
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = gap * balance
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#    return best_split, left_words, right_words
#
#
#def _is_true_two_column(words: list, page_width: float):
#    """
#    Determines whether a page genuinely has a two-column layout, AND
#    returns the correctly split word groups if so.
#    Requires: both sides substantial (>=30 words), near-empty gutter
#    (<5 straddling words), right column spans >=15% of left column's
#    vertical height.
#    """
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return None
#
#    gutter_words = [
#        w for w in words
#        if w["x0"] < split_x - 2 and w["x1"] > split_x + 2
#    ]
#    if len(gutter_words) > 5:
#        return None
#
#    if right_words and left_words:
#        right_top = min(w["top"] for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span = right_bottom - right_top
#
#        left_top = min(w["top"] for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.15:
#            return None
#
#    return left_words, right_words
#
#
## ──────────────────────────────────────────────────────────────
## LINE RECONSTRUCTION
## ──────────────────────────────────────────────────────────────
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from word dicts, grouping by vertical (top)
#    position and joining with a single space. Works identically whether
#    a word came from pdfplumber's native extraction, from
#    ocr_region_to_words() (hybrid-page image regions), or from a
#    full-page OCR engine's word/line output -- all use the same dict
#    shape.
#    """
#    if not words:
#        return ""
#
#    lines: list = []
#    current_line: list = []
#    current_top = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
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
## ──────────────────────────────────────────────────────────────
## LINE-GRANULARITY RECONSTRUCTION -- Surya only (see REVISION note above)
## ──────────────────────────────────────────────────────────────
##added this sort_lines function manually because getting error pasted on chatgpt
#
#def sort_lines_only(lines: list) -> str:
#    """
#    For already-line-granularity OCR data (Surya's TextLine output,
#    where each item is a whole line of recognized text, not a single
#    word): sorts by vertical position (top), tie-broken by horizontal
#    position (x0), and outputs one line per input item -- NO merging,
#    NO sidebar/two-column detection.
#
#    Unlike _extract_words_to_lines() / reconstruct_page_text(), this
#    never combines two different input items onto one output line.
#    That's the point: at this granularity every input item already IS
#    a complete line, so there is nothing to assemble, and attempting to
#    (as reconstruct_page_text() does) risks merging two genuinely
#    different lines together if their measured vertical positions
#    happen to land close to each other -- confirmed in production on a
#    real resume (see module docstring's REVISION note).
#
#    Trade-off, accepted deliberately: a short heading whose Surya
#    bounding box is very slightly mismeasured can land one line off
#    from its ideal position (cosmetic). In exchange, this guarantees a
#    single-column page's lines never get scrambled together, which is
#    the far more common and far more damaging failure mode this was
#    built to eliminate.
#
#    lines: list of dicts with keys "text", "x0", "top" (and any others
#      -- only these three are used). Returns "" for an empty list.
#    """
#    if not lines:
#        return ""
#    sorted_lines = sorted(lines, key=lambda w: (round(w["top"], 1), w["x0"]))
#    return "\n".join(l["text"] for l in sorted_lines)
#
#
## ──────────────────────────────────────────────────────────────
## SHARED ENTRY POINT — used by OCR modules (see module docstring)
## ──────────────────────────────────────────────────────────────
#
#def reconstruct_page_text(words: list, page_width: float) -> str:
#    """
#    Given a flat list of word dicts (keys: text, x0, x1, top, bottom,
#    height) for ONE page, reconstructs reading-order text using a
#    table -> sidebar -> two-column -> single-column fallback chain.
#
#    Table detection runs FIRST (see _reconstruct_table() and the
#    module docstring's REVISION note) because it's the most specific,
#    structured shape -- a genuine N-column grid should never be
#    mistaken for the looser sidebar or two-column shapes tried after
#    it. This is a convenience wrapper specifically for the OCR modules
#    (ocr_reader.py / ocr_reader_pytesseract.py), which -- unlike
#    pdf_reader.py -- process one page's OCR result in isolation and
#    don't maintain a cross-page primary/secondary two-column stream.
#
#    Returns "" for an empty words list so callers can safely skip
#    appending it.
#    """
#    if not words:
#        return ""
#
#    table_text = _reconstruct_table(words, page_width)
#    if table_text:
#        return table_text
#
#    sidebar_split = _find_sidebar_split(words, page_width)
#    if sidebar_split:
#        sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#        content_words = [w for w in words if w["x0"] > sidebar_split]
#        text = _merge_sidebar_with_content(sidebar_words, content_words)
#        if text.strip():
#            return text
#
#    two_col_result = _is_true_two_column(words, page_width)
#    if two_col_result:
#        left_words, right_words = two_col_result
#        left_text = _extract_words_to_lines(left_words)
#        right_text = _extract_words_to_lines(right_words)
#        parts = [t for t in (left_text.strip(), right_text.strip()) if t]
#        if parts:
#            return "\n\n".join(parts)
#
#    return _extract_words_to_lines(words)
#
#









#"""
#ocr_layout_reconstruction.py — Layer 0, layout logic for SCANNED/OCR pages only
#
#WHY THIS FILE IS SEPARATE FROM pdf_reader.py (READ THIS BEFORE EDITING)
#-------------------------------------------------------------------------
#This file is a DELIBERATE, INTENTIONAL COPY of the layout-detection
#logic that also lives inside pdf_reader.py (sidebar / true-two-column /
#single-column reading-order reconstruction from word positions). It is
#NOT imported by pdf_reader.py, and pdf_reader.py is NOT imported by
#this file or by anything that uses this file. The two are fully
#independent on purpose.
#
#Why duplicate instead of share: scanned/OCR pages and native PDF text
#pages turned out to need DIFFERENT tuning almost immediately. Confirmed
#on a real scanned job document (SMFG_Job_Document_Scanned.pdf): a
#label/value table ("Job Title" | "Asst. Manager, Cyber Defence") needed
#a tighter heading-merge tolerance (0.6x instead of 1.5x -- see REVISION
#note below) than resume sidebars did. If this logic were shared with
#pdf_reader.py, every future OCR-specific tuning decision would risk
#changing behavior on the already-proven native-PDF pipeline (60+
#resumes tested), and vice versa.
#
#This file is used by ocr_reader.py (Surya) and ocr_reader_pytesseract.py
#(tesseract fallback) ONLY -- never by pdf_reader.py.
#
#REVISION -- tightened the sidebar heading-merge tolerance
#-----------------------------------------------------------
#_merge_sidebar_with_content() groups words into separate heading LINES
#by checking the vertical gap between consecutive words: if the gap is
#within `word_height * heading_tolerance_mult`, they're treated as the
#same line; otherwise a new line starts.
#
#The old value, 1.5, was tuned against resume sidebars, where distinct
#section headings ("PROFESSIONAL SUMMARY", "EDUCATION", ...) normally
#have generous vertical space between them. Confirmed on the SMFG
#document's Job Title/Department/Grade/IC-PM table: row-to-row spacing
#there (35-41px) is SMALLER than 1.5x a single row's own word-height
#(52.5px), so the old tolerance merged all 4 separate table rows into
#one garbled heading blob instead of keeping them apart. Lowered to 0.6.
#
#REVISION -- added sort_lines_only() for Surya's LINE-granularity data
#-------------------------------------------------------------------------
#Confirmed on a real scanned resume (Anandu Chandran,
#PDF_image2pdf_20260821121805.pdf): reconstruct_page_text() -- built and
#tuned for WORD-level data (many small fragments per line, as tesseract's
#image_to_data() provides) -- badly scrambled a completely ordinary
#SINGLE-COLUMN resume when fed Surya's LINE-level TextLine data instead.
#
#Root cause, traced against the real garbled output: Surya hands back
#ONE bounding box per whole LINE of recognized text, not per word.
#_extract_words_to_lines() (used by reconstruct_page_text()'s single-
#column fallback) was built to merge individual WORD fragments into a
#line by checking if their vertical ("top") positions are within half a
#line-height of each other -- correct when the input really is word
#fragments (many words on one true line all share nearly identical
#top), but dangerous when the input is already whole lines: two
#DIFFERENT, adjacent lines can easily land within half a line-height of
#each other (especially with CENTER-ALIGNED text, e.g. a resume header,
#where every line has a genuinely different x0), causing them to be
#wrongly merged onto one output line and re-sorted left-to-right by
#x0 -- scrambling two unrelated lines together. Reproduced exactly
#against this real resume's header ("Plavila veedu, Mathra.P.O" /
#"Punalur-691333, Kollam District, Kerala, India" merging into one
#garbled line) using plausible coordinates for those two lines.
#
#The sidebar/two-column detectors have the same underlying problem for
#a different reason: they're calibrated against word-COUNT thresholds
#(e.g. >=30 words per side for two-column, <=25 for a sidebar) that
#assume rich word-level statistics. A resume page has maybe 20-40
#lines total at Surya's granularity -- nowhere near enough for these
#thresholds to mean what they were designed to mean, so they can fire
#(or fail to fire) on essentially coincidental x0 clustering that has
#nothing to do with the page's real visual layout.
#
#sort_lines_only() is the fix: for line-granularity data, skip ALL
#merging and column detection entirely -- just sort by (top, x0) and
#emit one output line per input line, unchanged. This can occasionally
#place a short heading one line off from its ideal position if Surya's
#box for it is slightly mismeasured (verified: acceptable, cosmetic),
#but it can NEVER merge two different lines' text together, which was
#the actually severe failure mode. This deliberately gives up the
#"smart" sidebar/table reordering reconstruct_page_text() could in
#theory provide for a scanned sidebar-style document processed via
#Surya (never verified working in the first place -- see ocr_reader.py's
#own docstring) in exchange for guaranteed correctness on the much more
#common case: an ordinary single-column resume, which is what this bug
#was actively breaking in production.
#
#reconstruct_page_text() (word-level, sidebar/two-column aware) is
#UNCHANGED below and remains correct for ocr_reader_pytesseract.py's
#tesseract path, which provides genuine word-level data via
#pytesseract.image_to_data() -- the word-count thresholds and merge
#tolerances mean what they were designed to mean there. Only Surya's
#line-granularity path needed to stop using it.
#"""
#
#from collections import Counter
#
#
## ──────────────────────────────────────────────────────────────
## SIDEBAR LAYOUT DETECTION (word-level data only -- see module docstring)
## ──────────────────────────────────────────────────────────────
#
#def _find_sidebar_split(words: list, page_width: float):
#    """
#    Detects if a page has a sidebar layout — a narrow left column
#    containing ONLY section headings, and a wide right column with content.
#    Returns the x-coordinate of the split point if sidebar detected,
#    or None if not a sidebar layout.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    content_start_x = None
#    best_count = 0
#
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    if content_start_x is None or best_count < 3:
#        return None
#
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#    if len(sidebar_candidates) > 25:
#        return None
#
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#    heading_boundary = sidebar_x0s[-1]
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
## Tuned per the REVISION note above. Kept as a named constant (not a
## magic number inline) specifically so a future regression against
## resume sidebars can dial this back without hunting through the merge
## function body.
#_SIDEBAR_HEADING_TOLERANCE_MULT = 0.6
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list,
#                                 heading_tolerance_mult: float = _SIDEBAR_HEADING_TOLERANCE_MULT) -> str:
#    """
#    Merges sidebar heading words with content words by vertical position.
#    """
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * heading_tolerance_mult:
#            current_words.append(word)
#            current_top = word["top"]
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#        result_lines.append(content_text)
#
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
## ──────────────────────────────────────────────────────────────
## TWO-COLUMN LAYOUT DETECTION (word-level data only -- see module docstring)
## ──────────────────────────────────────────────────────────────
#
#def _find_column_split(words: list, page_width: float):
#    """
#    Finds the actual column split point for a two-column layout, using
#    the gap in x0 start positions rather than assuming the page midpoint.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    best_split = None
#    best_score = 0
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        if left_count < 20 or right_count < 20:
#            continue
#
#        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
#            continue
#
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = gap * balance
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#    return best_split, left_words, right_words
#
#
#def _is_true_two_column(words: list, page_width: float):
#    """
#    Determines whether a page genuinely has a two-column layout, AND
#    returns the correctly split word groups if so.
#    Requires: both sides substantial (>=30 words), near-empty gutter
#    (<5 straddling words), right column spans >=15% of left column's
#    vertical height.
#    """
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return None
#
#    gutter_words = [
#        w for w in words
#        if w["x0"] < split_x - 2 and w["x1"] > split_x + 2
#    ]
#    if len(gutter_words) > 5:
#        return None
#
#    if right_words and left_words:
#        right_top = min(w["top"] for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span = right_bottom - right_top
#
#        left_top = min(w["top"] for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.15:
#            return None
#
#    return left_words, right_words
#
#
## ──────────────────────────────────────────────────────────────
## LINE RECONSTRUCTION (word-level data only -- see module docstring)
## ──────────────────────────────────────────────────────────────
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from word dicts, grouping by vertical (top)
#    position and joining with a single space. Works identically whether
#    a word came from pdfplumber's native extraction, from
#    ocr_region_to_words() (hybrid-page image regions), or from
#    tesseract's image_to_data() -- all provide genuine WORD-level
#    fragments, which is what the merge tolerance here is calibrated for.
#
#    Do NOT feed this LINE-granularity data (e.g. Surya's TextLine
#    output) -- see sort_lines_only() below and the module docstring's
#    REVISION note for why that scrambles otherwise-correct single-
#    column pages.
#    """
#    if not words:
#        return ""
#
#    lines: list = []
#    current_line: list = []
#    current_top = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
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
## ──────────────────────────────────────────────────────────────
## LINE-GRANULARITY RECONSTRUCTION -- Surya only (see REVISION note above)
## ──────────────────────────────────────────────────────────────
#
#def sort_lines_only(lines: list) -> str:
#    """
#    For already-line-granularity OCR data (Surya's TextLine output,
#    where each item is a whole line of recognized text, not a single
#    word): sorts by vertical position (top), tie-broken by horizontal
#    position (x0), and outputs one line per input item -- NO merging,
#    NO sidebar/two-column detection.
#
#    Unlike _extract_words_to_lines() / reconstruct_page_text(), this
#    never combines two different input items onto one output line.
#    That's the point: at this granularity every input item already IS
#    a complete line, so there is nothing to assemble, and attempting to
#    (as reconstruct_page_text() does) risks merging two genuinely
#    different lines together if their measured vertical positions
#    happen to land close to each other -- confirmed in production on a
#    real resume (see module docstring's REVISION note).
#
#    Trade-off, accepted deliberately: a short heading whose Surya
#    bounding box is very slightly mismeasured can land one line off
#    from its ideal position (cosmetic). In exchange, this guarantees a
#    single-column page's lines never get scrambled together, which is
#    the far more common and far more damaging failure mode this was
#    built to eliminate.
#
#    lines: list of dicts with keys "text", "x0", "top" (and any others
#      -- only these three are used). Returns "" for an empty list.
#    """
#    if not lines:
#        return ""
#    sorted_lines = sorted(lines, key=lambda w: (round(w["top"], 1), w["x0"]))
#    return "\n".join(l["text"] for l in sorted_lines)
#
#
## ──────────────────────────────────────────────────────────────
## SHARED ENTRY POINT — word-level data only (tesseract fallback path)
## ──────────────────────────────────────────────────────────────
#
#def reconstruct_page_text(words: list, page_width: float) -> str:
#    """
#    Given a flat list of WORD-level dicts (keys: text, x0, x1, top,
#    bottom, height) for ONE page, reconstructs reading-order text using
#    the sidebar -> two-column -> single-column fallback chain
#    pdf_reader.py uses for native (non-OCR) text/hybrid pages.
#
#    IMPORTANT: this function assumes genuine word-level input (many
#    small fragments, as pytesseract.image_to_data() provides) -- its
#    sidebar/two-column detection is calibrated against word-COUNT
#    thresholds that only mean what they're supposed to mean at that
#    granularity. Do NOT call this with Surya's LINE-granularity
#    TextLine data -- use sort_lines_only() instead (see that function's
#    docstring and this module's REVISION note for why, backed by a real
#    production bug this caused).
#
#    Used by ocr_reader_pytesseract.py's tesseract-fallback path only.
#
#    Returns "" for an empty words list so callers can safely skip
#    appending it.
#    """
#    if not words:
#        return ""
#
#    sidebar_split = _find_sidebar_split(words, page_width)
#    if sidebar_split:
#        sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#        content_words = [w for w in words if w["x0"] > sidebar_split]
#        text = _merge_sidebar_with_content(sidebar_words, content_words)
#        if text.strip():
#            return text
#
#    two_col_result = _is_true_two_column(words, page_width)
#    if two_col_result:
#        left_words, right_words = two_col_result
#        left_text = _extract_words_to_lines(left_words)
#        right_text = _extract_words_to_lines(right_words)
#        parts = [t for t in (left_text.strip(), right_text.strip()) if t]
#        if parts:
#            return "\n\n".join(parts)
#
#    return _extract_words_to_lines(words)
#


















#"""
#ocr_layout_reconstruction.py — Layer 0, layout logic for SCANNED/OCR pages only
#
#WHY THIS FILE IS SEPARATE FROM pdf_reader.py (READ THIS BEFORE EDITING)
#-------------------------------------------------------------------------
#This file is a DELIBERATE, INTENTIONAL COPY of the layout-detection
#logic that also lives inside pdf_reader.py (sidebar / true-two-column /
#single-column reading-order reconstruction from word positions). It is
#NOT imported by pdf_reader.py, and pdf_reader.py is NOT imported by
#this file or by anything that uses this file. The two are fully
#independent on purpose.
#
#Why duplicate instead of share: scanned/OCR pages and native PDF text
#pages turned out to need DIFFERENT tuning almost immediately. Confirmed
#on a real scanned job document (SMFG_Job_Document_Scanned.pdf): a
#label/value table ("Job Title" | "Asst. Manager, Cyber Defence") needed
#a tighter heading-merge tolerance (0.6x instead of 1.5x -- see REVISION
#note below) than resume sidebars did. If this logic were shared with
#pdf_reader.py, every future OCR-specific tuning decision would risk
#changing behavior on the already-proven native-PDF pipeline (60+
#resumes tested), and vice versa. Keeping them separate means:
#  - A bug fix or tuning change made HERE, for a scanned-resume problem,
#    can NEVER affect pdf_reader.py's native-PDF behavior. Zero risk,
#    not "probably fine."
#  - pdf_reader.py never needs to be touched, tested, or even re-read
#    when fixing a scanned-resume-only issue.
#  - The trade-off: if a genuine bug is found in the shared ALGORITHM
#    itself (not a tuning number, an actual logic error), it needs to be
#    fixed in both places by hand. That's an accepted cost in exchange
#    for isolation between two pipelines with different tuning needs.
#
#This file is used by ocr_reader.py (Surya) and ocr_reader_pytesseract.py
#(tesseract fallback) ONLY -- never by pdf_reader.py.
#
#NOTE ON THE PRE-OCR IMAGE QUALITY GATE (blur rejection / deskew /
#brightness correction): this file does NOT need any changes for that.
#It operates purely on word-position dicts (text/x0/x1/top/bottom/
#height) that the OCR modules hand it AFTER preprocessing and OCR have
#already happened -- it never sees a raw image or a fitz page.
#Preprocessing quality (a sharper, correctly-oriented source image)
#should, if anything, make the geometry this file relies on more
#reliable, not less -- no logic change is implied here.
#
#REVISION -- tightened the sidebar heading-merge tolerance
#-----------------------------------------------------------
#_merge_sidebar_with_content() groups words into separate heading LINES
#by checking the vertical gap between consecutive words: if the gap is
#within `word_height * heading_tolerance_mult`, they're treated as the
#same line; otherwise a new line starts.
#
#The old value, 1.5, was tuned against resume sidebars, where distinct
#section headings ("PROFESSIONAL SUMMARY", "EDUCATION", ...) normally
#have generous vertical space between them. Confirmed on the SMFG
#document's Job Title/Department/Grade/IC-PM table: row-to-row spacing
#there (35-41px) is SMALLER than 1.5x a single row's own word-height
#(52.5px), so the old tolerance merged all 4 separate table rows into
#one garbled heading blob instead of keeping them apart -- verified by
#manually tracing the merge loop against the real OCR'd coordinates.
#
#Lowered to 0.6 (verified against the same real coordinates: it keeps
#each of the 4 table rows separate, and each ends up within the existing
#20-unit heading-to-content vertical matching tolerance in
#_merge_sidebar_with_content, so they pair up with the correct value).
#0.6 was chosen deliberately just above the 0.5 used for CONTENT line
#grouping elsewhere in this file (headings can be very slightly more
#irregular from OCR noise than body text, so a touch more slack than
#content is reasonable) but well below the old 1.5.
#
#IMPORTANT FOR PRODUCTION ROLLOUT: this tolerance change affects every
#sidebar-layout page, including the resume sidebars this was originally
#tuned against. It was only tested against the SMFG table's real
#coordinates, not re-run against your full 60+ resume regression set.
#Re-run that full batch after adopting this file and specifically
#spot-check any resumes that use a sidebar layout, to confirm no
#headings that used to merge correctly (e.g. a heading that legitimately
#wraps onto two lines) now get incorrectly split apart.
#"""
#
#from collections import Counter
#
#
## ──────────────────────────────────────────────────────────────
## SIDEBAR LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_sidebar_split(words: list, page_width: float):
#    """
#    Detects if a page has a sidebar layout — a narrow left column
#    containing ONLY section headings, and a wide right column with content.
#    Returns the x-coordinate of the split point if sidebar detected,
#    or None if not a sidebar layout.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    content_start_x = None
#    best_count = 0
#
#    for x0, count in x0_counter.items():
#        if 80 < x0 < page_width * 0.45 and count > best_count:
#            best_count = count
#            content_start_x = x0
#
#    if content_start_x is None or best_count < 3:
#        return None
#
#    sidebar_candidates = [w for w in words if w["x0"] < content_start_x - 10]
#
#    if not sidebar_candidates:
#        return None
#    if len(sidebar_candidates) > 25:
#        return None
#
#    sidebar_x0s = sorted(set(round(w["x0"]) for w in sidebar_candidates))
#    heading_boundary = sidebar_x0s[-1]
#    for i in range(len(sidebar_x0s) - 1):
#        inner_gap = sidebar_x0s[i + 1] - sidebar_x0s[i]
#        if inner_gap > 30:
#            heading_boundary = sidebar_x0s[i]
#            break
#
#    gap = content_start_x - heading_boundary
#    if gap < 30:
#        return None
#
#    split = (heading_boundary + content_start_x) / 2
#    return split
#
#
## Tuned per the REVISION note above. Kept as a named constant (not a
## magic number inline) specifically so a future regression against
## resume sidebars can dial this back without hunting through the merge
## function body.
#_SIDEBAR_HEADING_TOLERANCE_MULT = 0.6
#
#
#def _merge_sidebar_with_content(sidebar_words: list, content_words: list,
#                                 heading_tolerance_mult: float = _SIDEBAR_HEADING_TOLERANCE_MULT) -> str:
#    """
#    Merges sidebar heading words with content words by vertical position.
#    """
#    heading_lines = {}
#    sorted_sidebar = sorted(sidebar_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_sidebar:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * heading_tolerance_mult:
#            current_words.append(word)
#            current_top = word["top"]
#        else:
#            heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            heading_lines[round(current_top)] = heading_text
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        heading_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        heading_lines[round(current_top)] = heading_text
#
#    content_line_list = []
#    sorted_content = sorted(content_words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    current_top = None
#    current_words = []
#    for word in sorted_content:
#        if current_top is None:
#            current_top = word["top"]
#            current_words = [word]
#        elif abs(word["top"] - current_top) <= max(word["height"], 1) * 0.5:
#            current_words.append(word)
#        else:
#            line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#            content_line_list.append((round(current_top), line_text))
#            current_top = word["top"]
#            current_words = [word]
#    if current_words:
#        line_text = " ".join(w["text"] for w in sorted(current_words, key=lambda w: w["x0"]))
#        content_line_list.append((round(current_top), line_text))
#
#    result_lines = []
#    used_headings = set()
#
#    for content_top, content_text in content_line_list:
#        for heading_top, heading_text in sorted(heading_lines.items()):
#            if heading_top in used_headings:
#                continue
#            if abs(heading_top - content_top) <= 20:
#                result_lines.append(heading_text)
#                used_headings.add(heading_top)
#                break
#        result_lines.append(content_text)
#
#    for heading_top, heading_text in sorted(heading_lines.items()):
#        if heading_top not in used_headings:
#            result_lines.append(heading_text)
#
#    return "\n".join(result_lines)
#
#
## ──────────────────────────────────────────────────────────────
## TWO-COLUMN LAYOUT DETECTION
## ──────────────────────────────────────────────────────────────
#
#def _find_column_split(words: list, page_width: float):
#    """
#    Finds the actual column split point for a two-column layout, using
#    the gap in x0 start positions rather than assuming the page midpoint.
#    """
#    if not words:
#        return None
#
#    x0_counter = Counter(round(w["x0"]) for w in words)
#    all_x0 = sorted(x0_counter.keys())
#
#    if len(all_x0) < 2:
#        return None
#
#    best_split = None
#    best_score = 0
#
#    for i in range(len(all_x0) - 1):
#        gap = all_x0[i + 1] - all_x0[i]
#        if gap < 15:
#            continue
#
#        split_x = (all_x0[i] + all_x0[i + 1]) / 2
#        left_count = sum(c for x, c in x0_counter.items() if x <= all_x0[i])
#        right_count = sum(c for x, c in x0_counter.items() if x >= all_x0[i + 1])
#
#        if left_count < 20 or right_count < 20:
#            continue
#
#        if split_x < page_width * 0.20 or split_x > page_width * 0.80:
#            continue
#
#        balance = min(left_count, right_count) / max(left_count, right_count)
#        score = gap * balance
#
#        if score > best_score:
#            best_score = score
#            best_split = split_x
#
#    if best_split is None:
#        return None
#
#    left_words = [w for w in words if w["x1"] <= best_split]
#    right_words = [w for w in words if w["x0"] > best_split]
#    return best_split, left_words, right_words
#
#
#def _is_true_two_column(words: list, page_width: float):
#    """
#    Determines whether a page genuinely has a two-column layout, AND
#    returns the correctly split word groups if so.
#    Requires: both sides substantial (>=30 words), near-empty gutter
#    (<5 straddling words), right column spans >=15% of left column's
#    vertical height.
#    """
#    result = _find_column_split(words, page_width)
#    if result is None:
#        return None
#
#    split_x, left_words, right_words = result
#
#    if len(left_words) < 30 or len(right_words) < 30:
#        return None
#
#    gutter_words = [
#        w for w in words
#        if w["x0"] < split_x - 2 and w["x1"] > split_x + 2
#    ]
#    if len(gutter_words) > 5:
#        return None
#
#    if right_words and left_words:
#        right_top = min(w["top"] for w in right_words)
#        right_bottom = max(w["bottom"] for w in right_words)
#        right_span = right_bottom - right_top
#
#        left_top = min(w["top"] for w in left_words)
#        left_bottom = max(w["bottom"] for w in left_words)
#        left_span = left_bottom - left_top
#
#        if left_span > 0 and right_span / left_span < 0.15:
#            return None
#
#    return left_words, right_words
#
#
## ──────────────────────────────────────────────────────────────
## LINE RECONSTRUCTION
## ──────────────────────────────────────────────────────────────
#
#def _extract_words_to_lines(words: list) -> str:
#    """
#    Reconstructs text lines from word dicts, grouping by vertical (top)
#    position and joining with a single space. Works identically whether
#    a word came from pdfplumber's native extraction, from
#    ocr_region_to_words() (hybrid-page image regions), or from a
#    full-page OCR engine's word/line output -- all use the same dict
#    shape.
#    """
#    if not words:
#        return ""
#
#    lines: list = []
#    current_line: list = []
#    current_top = None
#
#    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
#
#    for word in sorted_words:
#        if current_top is None:
#            current_top = word["top"]
#            current_line = [word]
#            continue
#
#        line_tolerance = max(word["height"], 1) * 0.5
#        if abs(word["top"] - current_top) <= line_tolerance:
#            current_line.append(word)
#        else:
#            lines.append(current_line)
#            current_line = [word]
#            current_top = word["top"]
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
## ──────────────────────────────────────────────────────────────
## SHARED ENTRY POINT — used by OCR modules (see module docstring)
## ──────────────────────────────────────────────────────────────
#
#def reconstruct_page_text(words: list, page_width: float) -> str:
#    """
#    Given a flat list of word dicts (keys: text, x0, x1, top, bottom,
#    height) for ONE page, reconstructs reading-order text using the
#    same sidebar -> two-column -> single-column fallback chain
#    pdf_reader.py already uses for native (non-OCR) text/hybrid pages.
#
#    This is a convenience wrapper specifically for the OCR modules
#    (ocr_reader.py / ocr_reader_pytesseract.py), which -- unlike
#    pdf_reader.py -- process one page's OCR result in isolation and
#    don't maintain a cross-page primary/secondary two-column stream.
#    For a two-column OCR'd page, this simply returns "left column text,
#    then right column text" for that one page; it does NOT carry a
#    "secondary" stream across pages the way pdf_reader.py's own
#    primary_parts/secondary_parts split does for native two-column
#    pages. If a scanned resume turns out to have a genuine multi-page
#    two-column layout where getting the cross-page ordering exactly
#    right matters, that's a follow-up enhancement, not something this
#    function currently does.
#
#    Returns "" for an empty words list so callers can safely skip
#    appending it.
#    """
#    if not words:
#        return ""
#
#    sidebar_split = _find_sidebar_split(words, page_width)
#    if sidebar_split:
#        sidebar_words = [w for w in words if w["x0"] < sidebar_split]
#        content_words = [w for w in words if w["x0"] > sidebar_split]
#        text = _merge_sidebar_with_content(sidebar_words, content_words)
#        if text.strip():
#            return text
#
#    two_col_result = _is_true_two_column(words, page_width)
#    if two_col_result:
#        left_words, right_words = two_col_result
#        left_text = _extract_words_to_lines(left_words)
#        right_text = _extract_words_to_lines(right_words)
#        parts = [t for t in (left_text.strip(), right_text.strip()) if t]
#        if parts:
#            return "\n\n".join(parts)
#
#    return _extract_words_to_lines(words)
#