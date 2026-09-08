"""
DOCX Reader — Layer 0

Reads three kinds of content out of a .docx file, in document order:
  1. Normal paragraphs                          (<w:p>)
  2. Tables, including merged cells             (<w:tbl>)
  3. Floating text boxes / infographic shapes   (<w:drawing> / <w:pict> containing <w:txbxContent>)

WHY #3 EXISTS
-------------
Many "designed" resume templates (common on Naukri and similar builders)
don't type the name/contact/skills/education/summary as normal
paragraphs at all -- they place them inside absolutely-positioned text
boxes anchored to a picture/shape, purely for visual layout. Confirmed
on a real resume (Praveen Kumar Pedapapa): the ENTIRE header, contact
row, employment details, skills list, education, and profile summary
lived inside 13 text boxes nested in a single <w:drawing> on the very
first body paragraph. python-docx's Paragraph.text only reads direct
<w:r>/<w:t> runs of a paragraph -- it never descends into
<w:drawing>/<w:txbxContent> -- so all of that content was silently
dropped with no error, and only the plain "Projects Overview" section
(which happened to be typed as normal paragraphs) made it through.
This affects any similarly-designed docx template, not just this file.

Two things make text boxes tricky to extract correctly, both handled
below:

  a) DUPLICATION: Word stores each text box twice inside
     <mc:AlternateContent> -- once as modern DrawingML
     (<mc:Choice Requires="wps">/"wpg") and once as a legacy VML
     fallback (<mc:Fallback>) for old Word versions. Reading both
     would double every line. We only read <mc:Choice>, and also
     dedupe by exact text as a backstop for any straggler.

  b) READING ORDER: text boxes are positioned by explicit X/Y offsets
     (EMU units), and their order IN THE XML is not their visual
     top-to-bottom order at all -- confirmed on the same resume, where
     the bottom-most box ("Profile Summary") appears FIRST in the XML
     and the name appears LAST. We recover visual reading order by
     bucketing boxes into "rows" using their Y offset (within a small
     tolerance, since boxes on the same visual row rarely share the
     exact same Y), then sorting left-to-right by X within each row.
     A short/ALL-CAPS heading-shaped line is nudged ahead of other
     text at the same position, since a heading box and its body-text
     box sometimes share identical anchor coordinates and would
     otherwise keep whatever arbitrary order they happened to be
     stored in.

This is purely additive: paragraphs and tables with no drawings in
them are read exactly as before, so documents that don't use this
kind of template are unaffected.
"""

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from pathlib import Path
import re

# --- XML namespaces used for manual (lxml-level) drawing/text-box parsing ---
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_NS = {"w": _W_NS, "mc": _MC_NS, "wp": _WP_NS}

# Same tolerance is used both to decide "same visual row" (EMU units;
# 914400 EMU = 1 inch, so 50000 EMU is roughly 0.05in / a hair's width --
# enough to absorb small alignment jitter between boxes on one row
# without merging genuinely different rows).
_ROW_TOLERANCE_EMU = 50_000

_HEADING_SHAPE_RE = re.compile(r"^[A-Z0-9\s&/]+$")


def _looks_like_heading(text: str) -> bool:
    """Cheap heuristic: short, ALL-CAPS-shaped line looks like a section
    heading rather than body text. Used only to break ties when two text
    boxes share identical anchor coordinates (see module docstring, b)."""
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first_line or len(first_line.split()) > 4:
        return False
    return bool(_HEADING_SHAPE_RE.match(first_line))


def _get_cell_paragraphs(cell) -> list:
   """One entry per non-empty paragraph in a table cell (never joined
   together), so a stray heading inside a multi-paragraph cell can
   still be detected as its own line downstream."""
   return [p.text.strip() for p in cell.paragraphs if p.text.strip()]


def _vmerge_status(tc):
   tcPr = tc.tcPr
   if tcPr is None:
       return None
   el = tcPr.find(qn("w:vMerge"))
   if el is None:
       return None
   return el.get(qn("w:val")) or "continue"


def _extract_table_rows(table) -> list:
   """
   Extracts text lines from a table, preserving paragraph boundaries
   within cells and handling vertically merged cells by inheriting the
   first line of text from the cell above.
   """
   rows_out = []
   last_seen_by_col = {}
   for row in table.rows:
       col_paragraphs = []
       for col_idx, tc in enumerate(row._tr.tc_lst):
           cell = _Cell(tc, table)
           paras = _get_cell_paragraphs(cell)
           vmerge = _vmerge_status(tc)
           if not paras and vmerge is not None and col_idx in last_seen_by_col:
               paras = [last_seen_by_col[col_idx]]
           if paras:
               last_seen_by_col[col_idx] = paras[0]
           col_paragraphs.append(paras)
       if not any(col_paragraphs):
           continue
       max_lines = max((len(p) for p in col_paragraphs), default=0)
       for line_idx in range(max_lines):
           line_cells = []
           for paras in col_paragraphs:
               if line_idx < len(paras):
                   line_cells.append(paras[line_idx])
               elif line_idx == 0:
                   line_cells.append("")
           while line_cells and line_cells[-1] == "":
               line_cells.pop()
           if line_cells:
               rows_out.append("\t".join(line_cells))
   return rows_out


def _group_into_rows(items):
    """items: list of (y, x, text, is_heading), already sorted by y.
    Returns a list of rows, each row a list of items sharing (approx)
    the same Y, sorted left-to-right by X (heading-shaped text wins
    ties at identical coordinates)."""
    rows = []
    for it in items:
        y = it[0]
        if rows and (y - rows[-1][0][0]) < _ROW_TOLERANCE_EMU:
            rows[-1].append(it)
        else:
            rows.append([it])
    for row in rows:
        row.sort(key=lambda it: (it[1], not it[3]))
    return rows


def _split_into_columns(items):
   """Cluster items into 2 side-by-side columns by X position, splitting
   at the single biggest gap between consecutive X values. Returns
   (left_items, right_items), each still carrying (y, x, text, is_heading)
   and NOT yet sorted by Y (caller sorts)."""
   if len(items) < 2:
       return items, []
   xs = sorted(set(it[1] for it in items))
   if len(xs) < 2:
       return items, []
   gaps = [(xs[i + 1] - xs[i], xs[i]) for i in range(len(xs) - 1)]
   _, split_after_x = max(gaps, key=lambda g: g[0])
   left = [it for it in items if it[1] <= split_after_x]
   right = [it for it in items if it[1] > split_after_x]
   return left, right


def _reconstruct_reading_order(items):
    """
    items: list of (y, x, text, is_heading).

    Sweeps the page top-to-bottom one visual row at a time (this handles
    the page header: name / title / contact row, which are one column
    wide or a single aligned row -- see module docstring). The moment a
    row contains 2+ heading-shaped items side by side, that's the sign
    the page has split into parallel columns (e.g. "EMPLOYEMENT DETAILS"
    next to "EDUCATION DETAILS"). From that row onward, instead of
    continuing to sweep row-by-row (which would interleave the two
    columns' unrelated content, splitting each heading from its own
    body -- confirmed on a real resume, Praveen Kumar Pedapapa, where
    "EDUCATION DETAILS" printed immediately after "EMPLOYEMENT DETAILS"
    and before either heading's actual content), we cluster ALL
    remaining items into two columns by X position and emit one column
    completely (top to bottom) before the other, so each heading stays
    directly above its own content.
    """
    if not items:
        return []

    items = sorted(items, key=lambda it: it[0])
    rows = _group_into_rows(items)

    ordered = []
    for row_idx, row in enumerate(rows):
        heading_count = sum(1 for it in row if it[3])
        if len(row) >= 2 and heading_count >= 2:
            # Columns start here: cluster this row + everything after it.
            remaining = [it for r in rows[row_idx:] for it in r]
            left, right = _split_into_columns(remaining)
            for column in (left, right):
                for col_row in _group_into_rows(sorted(column, key=lambda it: it[0])):
                    for _, _, text, _h in col_row:
                        ordered.extend(text.splitlines())
            return ordered
        for _, _, text, _h in row:
            ordered.extend(text.splitlines())

    return ordered


def _extract_textbox_lines(paragraph_xml) -> list:
    """
    Given the raw XML element of a <w:p>, find every text box anchored
    to it and return their contents as a list of lines, in reconstructed
    visual reading order.

    Returns [] if the paragraph has no text boxes -- callers should fall
    back to normal Paragraph.text handling in that case.
    """
    choices = paragraph_xml.findall(".//mc:Choice", _NS)
    if not choices:
        return []

    seen_text = set()
    items = []  # (y, x, text, is_heading)

    for choice in choices:
        for tb in choice.findall(".//w:txbxContent", _NS):
            paras = tb.findall(".//w:p", _NS)
            para_texts = [
                "".join(t.text or "" for t in p.findall(".//w:t", _NS))
                for p in paras
            ]
            text = "\n".join(t for t in para_texts if t.strip())
            if not text.strip() or text in seen_text:
                continue
            seen_text.add(text)

            x = y = 0
            node = tb
            while node is not None:
                pv = node.find("wp:positionV/wp:posOffset", _NS)
                ph = node.find("wp:positionH/wp:posOffset", _NS)
                if pv is not None and pv.text is not None:
                    y = int(pv.text)
                if ph is not None and ph.text is not None:
                    x = int(ph.text)
                node = node.getparent()

            items.append((y, x, text, _looks_like_heading(text)))

    return _reconstruct_reading_order(items)


def read_docx(docx_path: str) -> str:
   path = Path(docx_path)
   if not path.exists():
       raise FileNotFoundError(f"File not found: {docx_path}")
   doc = Document(docx_path)
   text_parts = []
   for element in doc.element.body:
       tag = element.tag
       if tag == qn("w:p"):
           # Text boxes first: a paragraph carrying a floating text box
           # normally has no meaningful text of its own (the box's text
           # lives in a completely separate nested paragraph tree), so
           # check for boxes before falling back to normal paragraph text.
           box_lines = _extract_textbox_lines(element)
           if box_lines:
               text_parts.extend(box_lines)
           # Use python-docx's own Paragraph.text (not a hand-rolled
           # direct-<w:r>-children scan) so text wrapped in
           # <w:hyperlink> (mailto:, LinkedIn/portfolio links, etc.) is
           # still captured -- a manual scan silently drops those runs.
           stripped = Paragraph(element, doc).text.strip()
           if stripped:
               text_parts.append(stripped)
       elif tag == qn("w:tbl"):
           table = Table(element, doc)
           text_parts.extend(_extract_table_rows(table))

   # Extract embedded hyperlink relationship targets
   docx_links = []
   try:
       for rel in doc.part.rels.values():
           if "hyperlink" in rel.reltype and rel.target_ref:
               target = rel.target_ref.strip()
               if target.startswith("mailto:"):
                   clean = target[7:].strip()
                   if clean:
                       docx_links.append(clean)
               elif target.startswith("http://") or target.startswith("https://") or target.startswith("www."):
                   docx_links.append(target)
   except Exception:
       pass

   if docx_links:
       unique_links = list(dict.fromkeys(docx_links))
       text_parts.append("Links:\n" + "\n".join(unique_links))

   return "\n".join(text_parts)


if __name__ == "__main__":
   import sys
   print(read_docx(sys.argv[1]))









#"""
#DOCX Reader — Layer 0
#
#Reads three kinds of content out of a .docx file, in document order:
#  1. Normal paragraphs                          (<w:p>)
#  2. Tables, including merged cells             (<w:tbl>)
#  3. Floating text boxes / infographic shapes   (<w:drawing> / <w:pict> containing <w:txbxContent>)
#
#WHY #3 EXISTS
#-------------
#Many "designed" resume templates (common on Naukri and similar builders)
#don't type the name/contact/skills/education/summary as normal
#paragraphs at all -- they place them inside absolutely-positioned text
#boxes anchored to a picture/shape, purely for visual layout. Confirmed
#on a real resume (Praveen Kumar Pedapapa): the ENTIRE header, contact
#row, employment details, skills list, education, and profile summary
#lived inside 13 text boxes nested in a single <w:drawing> on the very
#first body paragraph. python-docx's Paragraph.text only reads direct
#<w:r>/<w:t> runs of a paragraph -- it never descends into
#<w:drawing>/<w:txbxContent> -- so all of that content was silently
#dropped with no error, and only the plain "Projects Overview" section
#(which happened to be typed as normal paragraphs) made it through.
#This affects any similarly-designed docx template, not just this file.
#
#Two things make text boxes tricky to extract correctly, both handled
#below:
#
#  a) DUPLICATION: Word stores each text box twice inside
#     <mc:AlternateContent> -- once as modern DrawingML
#     (<mc:Choice Requires="wps">/"wpg") and once as a legacy VML
#     fallback (<mc:Fallback>) for old Word versions. Reading both
#     would double every line. We only read <mc:Choice>, and also
#     dedupe by exact text as a backstop for any straggler.
#
#  b) READING ORDER: text boxes are positioned by explicit X/Y offsets
#     (EMU units), and their order IN THE XML is not their visual
#     top-to-bottom order at all -- confirmed on the same resume, where
#     the bottom-most box ("Profile Summary") appears FIRST in the XML
#     and the name appears LAST. We recover visual reading order by
#     bucketing boxes into "rows" using their Y offset (within a small
#     tolerance, since boxes on the same visual row rarely share the
#     exact same Y), then sorting left-to-right by X within each row.
#     A short/ALL-CAPS heading-shaped line is nudged ahead of other
#     text at the same position, since a heading box and its body-text
#     box sometimes share identical anchor coordinates and would
#     otherwise keep whatever arbitrary order they happened to be
#     stored in.
#
#This is purely additive: paragraphs and tables with no drawings in
#them are read exactly as before, so documents that don't use this
#kind of template are unaffected.
#"""
#
#from docx import Document
#from docx.oxml.ns import qn
#from docx.table import Table, _Cell
#from docx.text.paragraph import Paragraph
#from pathlib import Path
#import re
#
## --- XML namespaces used for manual (lxml-level) drawing/text-box parsing ---
#_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
#_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
#_WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
#_NS = {"w": _W_NS, "mc": _MC_NS, "wp": _WP_NS}
#
## Same tolerance is used both to decide "same visual row" (EMU units;
## 914400 EMU = 1 inch, so 50000 EMU is roughly 0.05in / a hair's width --
## enough to absorb small alignment jitter between boxes on one row
## without merging genuinely different rows).
#_ROW_TOLERANCE_EMU = 50_000
#
#_HEADING_SHAPE_RE = re.compile(r"^[A-Z0-9\s&/]+$")
#
#
#def _looks_like_heading(text: str) -> bool:
#    """Cheap heuristic: short, ALL-CAPS-shaped line looks like a section
#    heading rather than body text. Used only to break ties when two text
#    boxes share identical anchor coordinates (see module docstring, b)."""
#    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
#    if not first_line or len(first_line.split()) > 4:
#        return False
#    return bool(_HEADING_SHAPE_RE.match(first_line))
#
#
#def _get_cell_paragraphs(cell) -> list:
#    """One entry per non-empty paragraph in a table cell (never joined
#    together), so a stray heading inside a multi-paragraph cell can
#    still be detected as its own line downstream."""
#    return [p.text.strip() for p in cell.paragraphs if p.text.strip()]
#
#
#def _vmerge_status(tc):
#    tcPr = tc.tcPr
#    if tcPr is None:
#        return None
#    el = tcPr.find(qn("w:vMerge"))
#    if el is None:
#        return None
#    return el.get(qn("w:val")) or "continue"
#
#
#def _extract_table_rows(table) -> list:
#    """
#    Extracts text lines from a table, preserving paragraph boundaries
#    within cells and handling vertically merged cells by inheriting the
#    first line of text from the cell above.
#    """
#    rows_out = []
#    last_seen_by_col = {}
#
#    for row in table.rows:
#        col_paragraphs = []
#        for col_idx, tc in enumerate(row._tr.tc_lst):
#            cell = _Cell(tc, table)
#            paras = _get_cell_paragraphs(cell)
#            vmerge = _vmerge_status(tc)
#
#            if not paras and vmerge is not None and col_idx in last_seen_by_col:
#                paras = [last_seen_by_col[col_idx]]
#
#            if paras:
#                last_seen_by_col[col_idx] = paras[0]
#
#            col_paragraphs.append(paras)
#
#        if not any(col_paragraphs):
#            continue
#
#        max_lines = max((len(p) for p in col_paragraphs), default=0)
#        for line_idx in range(max_lines):
#            line_cells = []
#            for paras in col_paragraphs:
#                if line_idx < len(paras):
#                    line_cells.append(paras[line_idx])
#                elif line_idx == 0:
#                    line_cells.append("")
#            while line_cells and line_cells[-1] == "":
#                line_cells.pop()
#            if line_cells:
#                rows_out.append("\t".join(line_cells))
#
#    return rows_out
#
#
#def _group_into_rows(items):
#    """items: list of (y, x, text, is_heading), already sorted by y.
#    Returns a list of rows, each row a list of items sharing (approx)
#    the same Y, sorted left-to-right by X (heading-shaped text wins
#    ties at identical coordinates)."""
#    rows = []
#    for it in items:
#        y = it[0]
#        if rows and (y - rows[-1][0][0]) < _ROW_TOLERANCE_EMU:
#            rows[-1].append(it)
#        else:
#            rows.append([it])
#    for row in rows:
#        row.sort(key=lambda it: (it[1], not it[3]))
#    return rows
#
#
#def _split_into_columns(items):
#    """Cluster items into 2 side-by-side columns by X position, splitting
#    at the single biggest gap between consecutive X values. Returns
#    (left_items, right_items), each still carrying (y, x, text, is_heading)
#    and NOT yet sorted by Y (caller sorts)."""
#    if len(items) < 2:
#        return items, []
#    xs = sorted(set(it[1] for it in items))
#    if len(xs) < 2:
#        return items, []
#    gaps = [(xs[i + 1] - xs[i], xs[i]) for i in range(len(xs) - 1)]
#    _, split_after_x = max(gaps, key=lambda g: g[0])
#    left = [it for it in items if it[1] <= split_after_x]
#    right = [it for it in items if it[1] > split_after_x]
#    return left, right
#
#
#def _reconstruct_reading_order(items):
#    """
#    items: list of (y, x, text, is_heading).
#
#    Sweeps the page top-to-bottom one visual row at a time (this handles
#    the page header: name / title / contact row, which are one column
#    wide or a single aligned row -- see module docstring). The moment a
#    row contains 2+ heading-shaped items side by side, that's the sign
#    the page has split into parallel columns (e.g. "EMPLOYEMENT DETAILS"
#    next to "EDUCATION DETAILS"). From that row onward, instead of
#    continuing to sweep row-by-row (which would interleave the two
#    columns' unrelated content, splitting each heading from its own
#    body -- confirmed on a real resume, Praveen Kumar Pedapapa, where
#    "EDUCATION DETAILS" printed immediately after "EMPLOYEMENT DETAILS"
#    and before either heading's actual content), we cluster ALL
#    remaining items into two columns by X position and emit one column
#    completely (top to bottom) before the other, so each heading stays
#    directly above its own content.
#    """
#    if not items:
#        return []
#
#    items = sorted(items, key=lambda it: it[0])
#    rows = _group_into_rows(items)
#
#    ordered = []
#    for row_idx, row in enumerate(rows):
#        heading_count = sum(1 for it in row if it[3])
#        if len(row) >= 2 and heading_count >= 2:
#            # Columns start here: cluster this row + everything after it.
#            remaining = [it for r in rows[row_idx:] for it in r]
#            left, right = _split_into_columns(remaining)
#            for column in (left, right):
#                for col_row in _group_into_rows(sorted(column, key=lambda it: it[0])):
#                    for _, _, text, _h in col_row:
#                        ordered.extend(text.splitlines())
#            return ordered
#        for _, _, text, _h in row:
#            ordered.extend(text.splitlines())
#
#    return ordered
#
#
#def _extract_textbox_lines(paragraph_xml) -> list:
#    """
#    Given the raw XML element of a <w:p>, find every text box anchored
#    to it and return their contents as a list of lines, in reconstructed
#    visual reading order.
#
#    Returns [] if the paragraph has no text boxes -- callers should fall
#    back to normal Paragraph.text handling in that case.
#    """
#    choices = paragraph_xml.findall(".//mc:Choice", _NS)
#    if not choices:
#        return []
#
#    seen_text = set()
#    items = []  # (y, x, text, is_heading)
#
#    for choice in choices:
#        for tb in choice.findall(".//w:txbxContent", _NS):
#            paras = tb.findall(".//w:p", _NS)
#            para_texts = [
#                "".join(t.text or "" for t in p.findall(".//w:t", _NS))
#                for p in paras
#            ]
#            text = "\n".join(t for t in para_texts if t.strip())
#            if not text.strip() or text in seen_text:
#                continue
#            seen_text.add(text)
#
#            x = y = 0
#            node = tb
#            while node is not None:
#                pv = node.find("wp:positionV/wp:posOffset", _NS)
#                ph = node.find("wp:positionH/wp:posOffset", _NS)
#                if pv is not None and pv.text is not None:
#                    y = int(pv.text)
#                if ph is not None and ph.text is not None:
#                    x = int(ph.text)
#                node = node.getparent()
#
#            items.append((y, x, text, _looks_like_heading(text)))
#
#    return _reconstruct_reading_order(items)
#
#
#def read_docx(docx_path: str) -> str:
#    path = Path(docx_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {docx_path}")
#
#    doc = Document(docx_path)
#    text_parts = []
#
#    for element in doc.element.body:
#        tag = element.tag
#
#        if tag == qn("w:p"):
#            # Text boxes first: a paragraph carrying a floating text box
#            # normally has no meaningful text of its own (the box's text
#            # lives in a completely separate nested paragraph tree), so
#            # check for boxes before falling back to normal paragraph text.
#            box_lines = _extract_textbox_lines(element)
#            if box_lines:
#                text_parts.extend(box_lines)
#
#            # Use python-docx's own Paragraph.text (not a hand-rolled
#            # direct-<w:r>-children scan) so text wrapped in
#            # <w:hyperlink> (mailto:, LinkedIn/portfolio links, etc.) is
#            # still captured -- a manual scan silently drops those runs.
#            stripped = Paragraph(element, doc).text.strip()
#            if stripped:
#                text_parts.append(stripped)
#
#        elif tag == qn("w:tbl"):
#            table = Table(element, doc)
#            text_parts.extend(_extract_table_rows(table))
#
#    return "\n".join(text_parts)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_docx(sys.argv[1]))
#














#Breaked and doesn't worked .
#"""
#DOCX Reader — Layer 0
#
#Reads three kinds of content out of a .docx file, in document order:
#  1. Normal paragraphs                          (<w:p>)
#  2. Tables, including merged cells             (<w:tbl>)
#  3. Floating text boxes / infographic shapes   (<w:drawing> / <w:pict> containing <w:txbxContent>)
#
#WHY #3 EXISTS
#-------------
#Many "designed" resume templates (common on Naukri and similar builders)
#don't type the name/contact/skills/education/summary as normal
#paragraphs at all -- they place them inside absolutely-positioned text
#boxes anchored to a picture/shape, purely for visual layout. Confirmed
#on a real resume (Praveen Kumar Pedapapa): the ENTIRE header, contact
#row, employment details, skills list, education, and profile summary
#lived inside 13 text boxes nested in a single <w:drawing> on the very
#first body paragraph. python-docx's Paragraph.text only reads direct
#<w:r>/<w:t> runs of a paragraph -- it never descends into
#<w:drawing>/<w:txbxContent> -- so all of that content was silently
#dropped with no error, and only the plain "Projects Overview" section
#(which happened to be typed as normal paragraphs) made it through.
#This affects any similarly-designed docx template, not just this file.
#
#Two things make text boxes tricky to extract correctly, both handled
#below:
#
#  a) DUPLICATION: Word stores each text box twice inside
#     <mc:AlternateContent> -- once as modern DrawingML
#     (<mc:Choice Requires="wps">/"wpg") and once as a legacy VML
#     fallback (<mc:Fallback>) for old Word versions. Reading both
#     would double every line. We only read <mc:Choice>, and also
#     dedupe by exact text as a backstop for any straggler.
#
#  b) READING ORDER: text boxes are positioned by explicit X/Y offsets
#     (EMU units), and their order IN THE XML is not their visual
#     top-to-bottom order at all -- confirmed on the same resume, where
#     the bottom-most box ("Profile Summary") appears FIRST in the XML
#     and the name appears LAST. We recover visual reading order by
#     bucketing boxes into "rows" using their Y offset (within a small
#     tolerance, since boxes on the same visual row rarely share the
#     exact same Y), then sorting left-to-right by X within each row.
#     A short/ALL-CAPS heading-shaped line is nudged ahead of other
#     text at the same position, since a heading box and its body-text
#     box sometimes share identical anchor coordinates and would
#     otherwise keep whatever arbitrary order they happened to be
#     stored in.
#
#This is purely additive: paragraphs and tables with no drawings in
#them are read exactly as before, so documents that don't use this
#kind of template are unaffected.
#"""
#
#from docx import Document
#from docx.oxml.ns import qn
#from docx.table import Table, _Cell
#from docx.text.paragraph import Paragraph
#from pathlib import Path
#import re
#
## --- XML namespaces used for manual (lxml-level) drawing/text-box parsing ---
#_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
#_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
#_WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
#_NS = {"w": _W_NS, "mc": _MC_NS, "wp": _WP_NS}
#
## Same tolerance is used both to decide "same visual row" (EMU units;
## 914400 EMU = 1 inch, so 50000 EMU is roughly 0.05in / a hair's width --
## enough to absorb small alignment jitter between boxes on one row
## without merging genuinely different rows).
#_ROW_TOLERANCE_EMU = 50_000
#
#_HEADING_SHAPE_RE = re.compile(r"^[A-Z0-9\s&/]+$")
#
#
#def _looks_like_heading(text: str) -> bool:
#    """Cheap heuristic: short, ALL-CAPS-shaped line looks like a section
#    heading rather than body text. Used only to break ties when two text
#    boxes share identical anchor coordinates (see module docstring, b)."""
#    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
#    if not first_line or len(first_line.split()) > 4:
#        return False
#    return bool(_HEADING_SHAPE_RE.match(first_line))
#
#
#def _get_cell_paragraphs(cell) -> list:
#    """One entry per non-empty paragraph in a table cell (never joined
#    together), so a stray heading inside a multi-paragraph cell can
#    still be detected as its own line downstream."""
#    return [p.text.strip() for p in cell.paragraphs if p.text.strip()]
#
#
#def _vmerge_status(tc):
#    tcPr = tc.tcPr
#    if tcPr is None:
#        return None
#    el = tcPr.find(qn("w:vMerge"))
#    if el is None:
#        return None
#    return el.get(qn("w:val")) or "continue"
#
#
#def _extract_table_rows(table) -> list:
#    """
#    Extracts text lines from a table, preserving paragraph boundaries
#    within cells and handling vertically merged cells by inheriting the
#    first line of text from the cell above.
#    """
#    rows_out = []
#    last_seen_by_col = {}
#
#    for row in table.rows:
#        col_paragraphs = []
#        for col_idx, tc in enumerate(row._tr.tc_lst):
#            cell = _Cell(tc, table)
#            paras = _get_cell_paragraphs(cell)
#            vmerge = _vmerge_status(tc)
#
#            if not paras and vmerge is not None and col_idx in last_seen_by_col:
#                paras = [last_seen_by_col[col_idx]]
#
#            if paras:
#                last_seen_by_col[col_idx] = paras[0]
#
#            col_paragraphs.append(paras)
#
#        if not any(col_paragraphs):
#            continue
#
#        max_lines = max((len(p) for p in col_paragraphs), default=0)
#        for line_idx in range(max_lines):
#            line_cells = []
#            for paras in col_paragraphs:
#                if line_idx < len(paras):
#                    line_cells.append(paras[line_idx])
#                elif line_idx == 0:
#                    line_cells.append("")
#            while line_cells and line_cells[-1] == "":
#                line_cells.pop()
#            if line_cells:
#                rows_out.append("\t".join(line_cells))
#
#    return rows_out
#
#
#def _extract_textbox_lines(paragraph_xml) -> list:
#    """
#    Given the raw XML element of a <w:p>, find every text box anchored
#    to it and return their contents as a list of lines, in reconstructed
#    visual reading order (top-to-bottom, left-to-right).
#
#    Returns [] if the paragraph has no text boxes -- callers should fall
#    back to normal Paragraph.text handling in that case.
#    """
#    choices = paragraph_xml.findall(".//mc:Choice", _NS)
#    if not choices:
#        return []
#
#    seen_text = set()
#    items = []  # (y, x, text, is_heading)
#
#    for choice in choices:
#        for tb in choice.findall(".//w:txbxContent", _NS):
#            paras = tb.findall(".//w:p", _NS)
#            para_texts = [
#                "".join(t.text or "" for t in p.findall(".//w:t", _NS))
#                for p in paras
#            ]
#            text = "\n".join(t for t in para_texts if t.strip())
#            if not text.strip() or text in seen_text:
#                continue
#            seen_text.add(text)
#
#            x = y = 0
#            node = tb
#            while node is not None:
#                pv = node.find("wp:positionV/wp:posOffset", _NS)
#                ph = node.find("wp:positionH/wp:posOffset", _NS)
#                if pv is not None and pv.text is not None:
#                    y = int(pv.text)
#                if ph is not None and ph.text is not None:
#                    x = int(ph.text)
#                node = node.getparent()
#
#            items.append((y, x, text, _looks_like_heading(text)))
#
#    if not items:
#        return []
#
#    # Group into visual rows by Y (within tolerance), preserving the
#    # order items were first encountered for any true ties.
#    items.sort(key=lambda it: it[0])
#    rows = []
#    for y, x, text, is_heading in items:
#        if rows and (y - rows[-1][0][0]) < _ROW_TOLERANCE_EMU:
#            rows[-1].append((y, x, text, is_heading))
#        else:
#            rows.append([(y, x, text, is_heading)])
#
#    ordered_lines = []
#    for row in rows:
#        # Within a row: left-to-right by X, but a heading-shaped line
#        # jumps ahead of anything sharing its exact X position (handles
#        # a heading box and its body-text box anchored at identical
#        # coordinates).
#        row.sort(key=lambda it: (it[1], not it[3]))
#        for _, _, text, _is_heading in row:
#            ordered_lines.extend(text.splitlines())
#
#    return ordered_lines
#
#
#def read_docx(docx_path: str) -> str:
#    path = Path(docx_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {docx_path}")
#
#    doc = Document(docx_path)
#    text_parts = []
#
#    for element in doc.element.body:
#        tag = element.tag
#
#        if tag == qn("w:p"):
#            # Text boxes first: a paragraph carrying a floating text box
#            # normally has no meaningful text of its own (the box's text
#            # lives in a completely separate nested paragraph tree), so
#            # check for boxes before falling back to normal paragraph text.
#            box_lines = _extract_textbox_lines(element)
#            if box_lines:
#                text_parts.extend(box_lines)
#
#            # Use python-docx's own Paragraph.text (not a hand-rolled
#            # direct-<w:r>-children scan) so text wrapped in
#            # <w:hyperlink> (mailto:, LinkedIn/portfolio links, etc.) is
#            # still captured -- a manual scan silently drops those runs.
#            stripped = Paragraph(element, doc).text.strip()
#            if stripped:
#                text_parts.append(stripped)
#
#        elif tag == qn("w:tbl"):
#            table = Table(element, doc)
#            text_parts.extend(_extract_table_rows(table))
#
#    return "\n".join(text_parts)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_docx(sys.argv[1]))
#
#








##worked just commenting to work the extraction of text from the box.
#"""
#DOCX Reader — Layer 0  (fixed: preserves paragraph boundaries inside table cells)
#"""
#
#from docx import Document
#from docx.oxml.ns import qn
#from docx.table import Table, _Cell
#from pathlib import Path
#
#
#def _get_cell_paragraphs(cell) -> list:
#    """
#    Returns a list of stripped, non-empty paragraph texts for a cell,
#    ONE ENTRY PER PARAGRAPH -- unlike the old _get_cell_text, this never
#    joins paragraphs together. A cell can legitimately contain several
#    unrelated paragraphs (a project title, its bullet points, and -- as
#    seen in a real resume -- even a stray section heading that landed in
#    the wrong cell), and collapsing them with " ".join() destroys the
#    paragraph break that a downstream line-based segmenter relies on to
#    detect headings. Confirmed on a real resume (Priyanshu Mishra):
#    "Education" was authored as the 4th paragraph inside the same table
#    cell as the "Currency Converter" project. The old space-join produced
#    "...seamless user interaction. Education", permanently hiding the
#    heading mid-sentence so segmentation could never find it.
#    """
#    return [p.text.strip() for p in cell.paragraphs if p.text.strip()]
#
#
#def _vmerge_status(tc):
#    tcPr = tc.tcPr
#    if tcPr is None:
#        return None
#    el = tcPr.find(qn("w:vMerge"))
#    if el is None:
#        return None
#    return el.get(qn("w:val")) or "continue"
#
#
#def _extract_table_rows(table) -> list:
#    """
#    Extracts text lines from a table, preserving paragraph boundaries.
#
#    Cells in the same row are paired by PARAGRAPH INDEX rather than
#    collapsed into one string each. For a normal single-paragraph-per-cell
#    row (e.g. "Institution" | "Date"), this behaves exactly like before:
#    one tab-joined line. For a cell with multiple paragraphs (e.g. a
#    project title + description + a stray heading, all sharing a row with
#    a single "Live Link" cell), each paragraph becomes its own line; only
#    the first paragraph of the multi-paragraph cell is tab-paired with the
#    other column's (usually single) paragraph, since that's the one that
#    is actually aligned with it visually. Later paragraphs in that cell
#    (description, or an unrelated heading like "Education") come out as
#    their own standalone lines with nothing spuriously tab-appended to
#    them, so a line-based segmenter can still detect them as headings.
#
#    Vertical merges (vMerge) are still handled: a "continue" cell carries
#    no text of its own in the XML, so we inherit the last non-empty FIRST
#    LINE seen for that column, matching what a reader would visually see.
#    """
#    rows_out = []
#    last_seen_by_col = {}
#
#    for row in table.rows:
#        col_paragraphs = []  # list of paragraph-lists, one per column
#        for col_idx, tc in enumerate(row._tr.tc_lst):
#            cell = _Cell(tc, table)
#            paras = _get_cell_paragraphs(cell)
#            vmerge = _vmerge_status(tc)
#
#            if not paras and vmerge is not None and col_idx in last_seen_by_col:
#                paras = [last_seen_by_col[col_idx]]  # inherit from the restart cell above
#
#            if paras:
#                last_seen_by_col[col_idx] = paras[0]
#
#            col_paragraphs.append(paras)
#
#        if not any(col_paragraphs):
#            continue
#
#        max_lines = max((len(p) for p in col_paragraphs), default=0)
#        for line_idx in range(max_lines):
#            line_cells = []
#            for paras in col_paragraphs:
#                if line_idx < len(paras):
#                    line_cells.append(paras[line_idx])
#                elif line_idx == 0:
#                    line_cells.append("")  # keep column alignment on the primary line only
#            # Drop a wholly-empty line, and drop trailing empty cells so we
#            # don't emit stray tabs for columns that only had one paragraph.
#            while line_cells and line_cells[-1] == "":
#                line_cells.pop()
#            if line_cells:
#                rows_out.append("\t".join(line_cells))
#
#    return rows_out
#
#
#def read_docx(docx_path: str) -> str:
#    path = Path(docx_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {docx_path}")
#
#    doc = Document(docx_path)
#    text_parts = []
#
#    for element in doc.element.body:
#        tag = element.tag
#
#        if tag == qn("w:p"):
#            # Use python-docx's own Paragraph.text rather than hand-rolling a
#            # direct-<w:r>-children scan. The manual scan silently drops any
#            # run wrapped in <w:hyperlink> (mailto: links, LinkedIn/portfolio
#            # links, etc.) -- and because such paragraphs usually still have
#            # OTHER plain-text runs alongside the link, the manual result is
#            # non-empty and never falls back, so the link text just vanishes
#            # with no error. Confirmed on a real resume (R. Lokesh): the
#            # "Mail ID :" line has the address wrapped in <w:hyperlink>, and
#            # the old logic dropped it while keeping "Mail ID :" itself.
#            from docx.text.paragraph import Paragraph
#            stripped = Paragraph(element, doc).text.strip()
#            if stripped:
#                text_parts.append(stripped)
#
#        elif tag == qn("w:tbl"):
#            table = Table(element, doc)
#            text_parts.extend(_extract_table_rows(table))
#
#    return "\n".join(text_parts)
#
#
#if __name__ == "__main__":
#    import sys
#    print(read_docx(sys.argv[1]))
#










#"""
#DOCX Reader — Layer 0
#Reads Microsoft Word documents.
#"""
#
#from docx import Document
#from docx.oxml.ns import qn
#from docx.table import Table, _Cell
#from pathlib import Path
#
#
#def _get_cell_text(cell) -> str:
#    """Returns stripped text from a table cell, collapsing inner newlines."""
#    return " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
#
#
#def _vmerge_status(tc):
#    """
#    Returns the raw vMerge value for a <w:tc> element: "restart", "continue"
#    (the implicit default when <w:vMerge/> has no w:val attribute), or None
#    if the cell isn't part of a vertical merge at all.
#    """
#    tcPr = tc.tcPr
#    if tcPr is None:
#        return None
#    el = tcPr.find(qn("w:vMerge"))
#    if el is None:
#        return None
#    return el.get(qn("w:val")) or "continue"
#
#
#def _extract_table_rows(table) -> list:
#    """
#    Extracts one tab-joined string per table row.
#
#    FIX -- what was wrong before:
#      The previous version used python-docx's `row.cells` (which "unrolls"
#      horizontally-merged cells into repeated entries for convenience) plus
#      an `id(cell._tc)` check to de-duplicate those repeats. This is broken:
#      `cell._tc` creates a fresh lxml element-proxy wrapper on every access
#      rather than returning a stable cached object, so once an earlier
#      wrapper gets garbage-collected, Python is free to reuse its memory
#      address for a completely unrelated cell's wrapper. Confirmed on a
#      real resume (R. Lokesh): "Arts&Science" (row 1) and "BBA" (row 0) --
#      two totally unrelated cells -- ended up with the exact same
#      `id(cell._tc)`, so "Arts&Science" was wrongly treated as an
#      already-seen duplicate and silently dropped. This happened
#      unpredictably throughout every table in the document, since which
#      ids collide depends on CPython's memory allocator/GC timing, not on
#      anything about the actual document structure.
#
#    FIX -- what we do now:
#      Iterate `row._tr.tc_lst` directly -- the row's REAL, un-unrolled
#      <w:tc> XML elements. At the raw-XML level, a horizontal merge (a
#      cell spanning multiple grid columns) is stored as exactly ONE <w:tc>
#      with a gridSpan attribute, never repeated -- so there is nothing to
#      de-duplicate, and no unreliable identity comparison is needed at all.
#
#      Vertical merges are handled separately and correctly: a vMerge
#      "continue" cell is a real, distinct <w:tc> per row, but by OOXML
#      convention it carries NO text of its own -- the actual text lives
#      only in the "restart" cell at the top of the merge, and Word just
#      visually extends that value down through the continuation rows.
#      Confirmed on the same resume: rows 2-4 of the education table
#      turned up completely empty for several columns once the id()-based
#      false-matching was removed, because that's genuinely what the XML
#      contains for those cells. We track the last non-empty value seen
#      per column position and inherit it into any empty vMerge cell below
#      it, which reconstructs exactly what a person reading the rendered
#      Word table would see.
#    """
#    rows_out = []
#    last_seen_by_col = {}
#
#    for row in table.rows:
#        row_texts = []
#        for col_idx, tc in enumerate(row._tr.tc_lst):
#            cell = _Cell(tc, table)
#            cell_text = _get_cell_text(cell)
#            vmerge = _vmerge_status(tc)
#
#            if not cell_text and vmerge is not None and col_idx in last_seen_by_col:
#                cell_text = last_seen_by_col[col_idx]  # inherit from the restart cell above
#
#            if cell_text:
#                last_seen_by_col[col_idx] = cell_text
#                row_texts.append(cell_text)
#
#        if row_texts:
#            rows_out.append("\t".join(row_texts))
#
#    return rows_out
#
#
#def read_docx(docx_path: str) -> str:
#    """
#    Reads a .docx file and returns all text in document order.
#
#    We walk doc.element.body directly and process each child element in
#    the order it appears in the XML, so paragraphs and tables are both
#    handled in one pass and the extracted text mirrors the visual layout
#    (rather than, e.g., dumping every table's content after all paragraph
#    text regardless of where the table actually sits in the document).
#    """
#    path = Path(docx_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {docx_path}")
#
#    doc = Document(docx_path)
#    text_parts = []
#
#    for element in doc.element.body:
#        tag = element.tag
#
#        # ── Paragraph ────────────────────────────────────────────────
#        if tag == qn("w:p"):
#            para_text = "".join(run.text for run in element.iterchildren(qn("w:r"))
#                                if hasattr(run, "text") and run.text)
#
#            # Fallback: use python-docx Paragraph wrapper
#            if not para_text.strip():
#                from docx.text.paragraph import Paragraph
#                para_text = Paragraph(element, doc).text
#
#            stripped = para_text.strip()
#            if stripped:
#                text_parts.append(stripped)
#
#        # ── Table ─────────────────────────────────────────────────────
#        elif tag == qn("w:tbl"):
#            table = Table(element, doc)
#            text_parts.extend(_extract_table_rows(table))
#
#    return "\n".join(text_parts)
#












#"""
#DOCX Reader — Layer 0
#Reads Microsoft Word documents.
#"""
#
#from docx import Document
#from docx.oxml.ns import qn
#from pathlib import Path
#
#
#def _get_cell_text(cell) -> str:
#    """Returns stripped text from a table cell, collapsing inner newlines."""
#    return " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
#
#
#def read_docx(docx_path: str) -> str:
#    """
#    Reads a .docx file and returns all text in document order.
#
#    FIX — What was wrong before:
#      The old code iterated doc.paragraphs first, then doc.tables.
#      This completely destroyed document order: every table (skills,
#      education, experience laid out in a table) was appended AFTER all
#      paragraph text, regardless of where the table actually appeared in
#      the document. On resumes that mix paragraphs and tables, this
#      produced a garbled, out-of-sequence result.
#
#    FIX — What we do now:
#      We walk doc.element.body directly and process each child element
#      in the order it appears in the XML. Paragraphs and tables are both
#      handled in one pass, so the extracted text mirrors the visual layout.
#    """
#    path = Path(docx_path)
#    if not path.exists():
#        raise FileNotFoundError(f"File not found: {docx_path}")
#
#    doc = Document(docx_path)
#    text_parts = []
#
#    for element in doc.element.body:
#        tag = element.tag
#
#        # ── Paragraph ────────────────────────────────────────────────
#        if tag == qn("w:p"):
#            # Reconstruct paragraph from its runs so we don't lose
#            # inline formatting boundaries that create word-run gaps
#            para_text = "".join(run.text for run in element.iterchildren(qn("w:r"))
#                                if hasattr(run, "text") and run.text)
#
#            # Fallback: use python-docx Paragraph wrapper
#            if not para_text.strip():
#                from docx.text.paragraph import Paragraph
#                para_text = Paragraph(element, doc).text
#
#            stripped = para_text.strip()
#            if stripped:
#                text_parts.append(stripped)
#
#        # ── Table ─────────────────────────────────────────────────────
#        elif tag == qn("w:tbl"):
#            from docx.table import Table
#            table = Table(element, doc)
#            seen_cells: set = set()   # de-duplicate merged cells
#
#            for row in table.rows:
#                row_texts = []
#                for cell in row.cells:
#                    # python-docx exposes merged cells multiple times
#                    cell_id = id(cell._tc)
#                    if cell_id in seen_cells:
#                        continue
#                    seen_cells.add(cell_id)
#
#                    cell_text = _get_cell_text(cell)
#                    if cell_text:
#                        row_texts.append(cell_text)
#
#                if row_texts:
#                    text_parts.append("\t".join(row_texts))
#
#    return "\n".join(text_parts)
#