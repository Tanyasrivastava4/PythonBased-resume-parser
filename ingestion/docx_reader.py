"""
DOCX Reader — Layer 0
Reads Microsoft Word documents.
"""

from docx import Document
from docx.oxml.ns import qn
from pathlib import Path


def _get_cell_text(cell) -> str:
    """Returns stripped text from a table cell, collapsing inner newlines."""
    return " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())


def read_docx(docx_path: str) -> str:
    """
    Reads a .docx file and returns all text in document order.

    FIX — What was wrong before:
      The old code iterated doc.paragraphs first, then doc.tables.
      This completely destroyed document order: every table (skills,
      education, experience laid out in a table) was appended AFTER all
      paragraph text, regardless of where the table actually appeared in
      the document. On resumes that mix paragraphs and tables, this
      produced a garbled, out-of-sequence result.

    FIX — What we do now:
      We walk doc.element.body directly and process each child element
      in the order it appears in the XML. Paragraphs and tables are both
      handled in one pass, so the extracted text mirrors the visual layout.
    """
    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {docx_path}")

    doc = Document(docx_path)
    text_parts = []

    for element in doc.element.body:
        tag = element.tag

        # ── Paragraph ────────────────────────────────────────────────
        if tag == qn("w:p"):
            # Reconstruct paragraph from its runs so we don't lose
            # inline formatting boundaries that create word-run gaps
            para_text = "".join(run.text for run in element.iterchildren(qn("w:r"))
                                if hasattr(run, "text") and run.text)

            # Fallback: use python-docx Paragraph wrapper
            if not para_text.strip():
                from docx.text.paragraph import Paragraph
                para_text = Paragraph(element, doc).text

            stripped = para_text.strip()
            if stripped:
                text_parts.append(stripped)

        # ── Table ─────────────────────────────────────────────────────
        elif tag == qn("w:tbl"):
            from docx.table import Table
            table = Table(element, doc)
            seen_cells: set = set()   # de-duplicate merged cells

            for row in table.rows:
                row_texts = []
                for cell in row.cells:
                    # python-docx exposes merged cells multiple times
                    cell_id = id(cell._tc)
                    if cell_id in seen_cells:
                        continue
                    seen_cells.add(cell_id)

                    cell_text = _get_cell_text(cell)
                    if cell_text:
                        row_texts.append(cell_text)

                if row_texts:
                    text_parts.append("\t".join(row_texts))

    return "\n".join(text_parts)
