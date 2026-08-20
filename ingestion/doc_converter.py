"""
DOC Converter — Layer 0 pre-processing step, for .doc only.

.doc (pre-2007 binary OLE/compound-file format) cannot be opened by
python-docx at all -- it only understands .docx (a zip of XML). Rather
than hand-parsing the old binary format, or rasterizing to an image and
OCR'ing it (which would throw away a perfectly good text layer and
reintroduce OCR errors for no reason -- confirmed by testing), we shell
out to LibreOffice in headless mode to convert .doc -> .docx, then hand
the result to the EXISTING, already-tested docx_reader.read_docx().

This keeps ingestion/docx_reader.py completely untouched: .doc support
is just a conversion step bolted on in front of it, not a new parser.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager


class DocConversionError(Exception):
    pass


def _run_soffice_convert(src: Path, out_dir: Path, timeout: int) -> Path:
    try:
        result = subprocess.run(
            [
                "soffice", "--headless", "--norestore",
                "--convert-to", "docx",
                "--outdir", str(out_dir),
                str(src),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise DocConversionError(f"LibreOffice conversion timed out for {src}") from e
    except FileNotFoundError as e:
        raise DocConversionError(
            "LibreOffice ('soffice') is not installed or not on PATH. "
            "Install it (e.g. apt install libreoffice-writer) to enable .doc support."
        ) from e

    converted = out_dir / (src.stem + ".docx")
    if result.returncode != 0 or not converted.exists():
        raise DocConversionError(
            f"LibreOffice failed to convert {src}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return converted


def convert_doc_to_docx(doc_path: str, timeout: int = 60) -> str:
    """
    Converts a .doc file to .docx using LibreOffice headless mode.
    Returns the path to the converted .docx file, written into a fresh
    temp directory that is left in place for the caller to read from
    (and clean up -- see convert_doc_to_docx_tmp() below for a
    context-manager version that cleans up automatically).
    """
    src = Path(doc_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {doc_path}")
    if src.suffix.lower() != ".doc":
        raise ValueError(f"Expected a .doc file, got: {src.suffix}")

    out_dir = Path(tempfile.mkdtemp(prefix="doc_convert_"))
    try:
        converted = _run_soffice_convert(src, out_dir, timeout)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return str(converted)


@contextmanager
def convert_doc_to_docx_tmp(doc_path: str, timeout: int = 60):
    """
    Same as convert_doc_to_docx(), but as a context manager that deletes
    the temp directory automatically when done. Prefer this in a batch
    pipeline processing many resumes, so temp files don't pile up.

    Usage:
        with convert_doc_to_docx_tmp("resume.doc") as docx_path:
            text = read_docx(docx_path)
        # temp file already cleaned up here
    """
    src = Path(doc_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {doc_path}")
    if src.suffix.lower() != ".doc":
        raise ValueError(f"Expected a .doc file, got: {src.suffix}")

    out_dir = Path(tempfile.mkdtemp(prefix="doc_convert_"))
    try:
        converted = _run_soffice_convert(src, out_dir, timeout)
        yield str(converted)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    out = convert_doc_to_docx(sys.argv[1])
    print(f"Converted to: {out}")