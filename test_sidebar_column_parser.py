"""
test_sidebar_column_parser.py
──────────────────────────────
Standalone test script for testing sidebar layout & multi-column PDF resumes
(Left Sidebar, Right Sidebar, 50/50 Two-Column, Single Column).

This file is 100% standalone and does NOT touch or modify any existing core files.

Usage:
  1. Create a folder named 'pdfs' (or place PDFs in 'resumes/' or any folder).
  2. Put your test resume PDFs inside that folder.
  3. Run:
       python test_sidebar_column_parser.py pdfs/Abhinav_Ashish.pdf
     OR run on all PDFs in a folder:
       python test_sidebar_column_parser.py pdfs/
"""

import sys
import os
import re
import glob
from pathlib import Path
from collections import Counter
import fitz  # PyMuPDF

# Import section splitter & extractors from current workspace
try:
    from segmenter import split_into_sections, get_section_text
    from extractors.experience import extract_experience
except ImportError:
    split_into_sections = None
    extract_experience = None


def extract_lines_from_words(words: list) -> str:
    """Groups words into horizontal lines based on top coordinate tolerance."""
    if not words:
        return ""
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    curr_line = []
    curr_top = None

    for w in sorted_words:
        if curr_top is None:
            curr_top = w["top"]
            curr_line = [w]
            continue
        line_tolerance = max(w.get("height", 1) * 0.7, 3.5)
        if abs(w["top"] - curr_top) <= line_tolerance:
            curr_line.append(w)
        else:
            lines.append(curr_line)
            curr_line = [w]
            curr_top = w["top"]
    if curr_line:
        lines.append(curr_line)

    text_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["x0"])
        sub_lines = []
        curr_sub = [line_sorted[0]["text"]]
        for prev, curr in zip(line_sorted, line_sorted[1:]):
            if curr["x0"] - prev["x1"] > 100:
                sub_lines.append(" ".join(curr_sub))
                curr_sub = [curr["text"]]
            else:
                curr_sub.append(curr["text"])
        if curr_sub:
            sub_lines.append(" ".join(curr_sub))
        text_lines.extend(sub_lines)

    return "\n".join(text_lines)


def analyze_page_layout(words: list, page_width: float, page_height: float):
    """
    Detects page layout type:
      - LEFT_SIDEBAR (split_x between 0.20 and 0.40 * width)
      - TWO_COLUMN_BALANCED (split_x between 0.40 and 0.65 * width)
      - RIGHT_SIDEBAR (split_x between 0.65 and 0.80 * width)
      - SINGLE_COLUMN (no clear vertical split)
    """
    if not words or len(words) < 25:
        return "SINGLE_COLUMN", None, 0.0

    # Filter out full-width top header words (e.g. candidate name, horizontal rule)
    # to find true body column splits
    body_words = [w for w in words if w["top"] > page_height * 0.08]
    if len(body_words) < 20:
        body_words = words

    x0_counts = Counter(round(w["x0"]) for w in body_words)
    sorted_x0 = sorted(x0_counts.keys())

    best_split = None
    best_score = 0.0
    layout_type = "SINGLE_COLUMN"
    header_bottom_y = 0.0

    # Detect header bottom Y (where full-width title/name ends)
    sorted_w = sorted(words, key=lambda w: (w["top"], w["x0"]))
    top_lines = []
    curr_line = []
    curr_top = None
    for w in sorted_w:
        if curr_top is None or abs(w["top"] - curr_top) <= 3.0:
            curr_line.append(w)
            curr_top = w["top"] if curr_top is None else curr_top
        else:
            top_lines.append(curr_line)
            curr_line = [w]
            curr_top = w["top"]
    if curr_line:
        top_lines.append(curr_line)

    for i in range(len(sorted_x0) - 1):
        x1 = sorted_x0[i]
        x2 = sorted_x0[i + 1]

        left_cluster = [w for w in body_words if w["x0"] <= x1]
        right_cluster = [w for w in body_words if w["x0"] >= x2]

        if not left_cluster or not right_cluster:
            continue

        max_left_x1 = max(w["x1"] for w in left_cluster)
        min_right_x0 = min(w["x0"] for w in right_cluster)

        gap = min_right_x0 - max_left_x1
        if gap < 6:
            continue

        split_x = (max_left_x1 + min_right_x0) / 2.0
        ratio = split_x / page_width

        # Accept candidate splits between 18% and 82% of page width
        if ratio < 0.18 or ratio > 0.82:
            continue

        # Check straddling words
        straddlers = [w for w in body_words if w["x0"] < split_x - 5 and w["x1"] > split_x + 5]
        if len(straddlers) > 3:
            continue

        left_w_count = len([w for w in body_words if w["x1"] <= split_x])
        right_w_count = len([w for w in body_words if w["x0"] >= split_x])

        if left_w_count < 10 or right_w_count < 10:
            continue

        # Scoring split: reward clean gap and substantial word presence on both sides
        score = gap * (min(left_w_count, right_w_count) + 10)

        if score > best_score:
            best_score = score
            best_split = split_x

    if best_split is not None:
        ratio = best_split / page_width
        if 0.18 <= ratio < 0.38:
            layout_type = "LEFT_SIDEBAR"
        elif 0.38 <= ratio <= 0.65:
            layout_type = "TWO_COLUMN_BALANCED"
        elif 0.65 < ratio <= 0.82:
            layout_type = "RIGHT_SIDEBAR"

        # Calculate header bottom Y above column split (constrained to top 15% of page height)
        for line_w in top_lines:
            min_x0 = min(w["x0"] for w in line_w)
            max_x1 = max(w["x1"] for w in line_w)
            max_y = max(w["bottom"] for w in line_w)
            if min_x0 < best_split - 15 and max_x1 > best_split + 15 and max_y < page_height * 0.15:
                if max_y > header_bottom_y:
                    header_bottom_y = max_y

    return layout_type, best_split, header_bottom_y


def read_pdf_sidebar_aware(pdf_path: str) -> str:
    """
    Reads a PDF file page by page and processes text column-by-column
    to prevent sidebar line interleaving / zippering.
    """
    doc = fitz.open(pdf_path)
    page_texts = []

    for page_num, page in enumerate(doc):
        words = page.get_text("words")
        if not words:
            continue

        page_width = page.rect.width
        page_height = page.rect.height

        w_dicts = [
            {
                "x0": w[0],
                "top": w[1],
                "x1": w[2],
                "bottom": w[3],
                "text": w[4],
                "height": w[3] - w[1],
            }
            for w in words
        ]

        layout_type, split_x, header_bottom_y = analyze_page_layout(w_dicts, page_width, page_height)

        if layout_type == "SINGLE_COLUMN" or split_x is None:
            # Standard single column top-to-bottom reading
            page_text = extract_lines_from_words(w_dicts)
            page_texts.append(page_text)
            continue

        # Separate top header (if any) from vertical columns
        header_words = []
        body_words = w_dicts

        if header_bottom_y > 0:
            header_words = [w for w in w_dicts if w["top"] < header_bottom_y - 2]
            body_words = [w for w in w_dicts if w["top"] >= header_bottom_y - 2]

        left_words = [w for w in body_words if w["x0"] < split_x]
        right_words = [w for w in body_words if w["x0"] >= split_x]

        header_text = extract_lines_from_words(header_words)
        left_text = extract_lines_from_words(left_words)
        right_text = extract_lines_from_words(right_words)

        page_parts = []
        if header_text.strip():
            page_parts.append(header_text.strip())

        if layout_type == "LEFT_SIDEBAR":
            # For LEFT_SIDEBAR: Read Main Body (Right Column) FIRST, Sidebar (Left Column) SECOND
            if right_text.strip():
                page_parts.append(right_text.strip())
            if left_text.strip():
                page_parts.append("\n--- SIDEBAR ---")
                page_parts.append(left_text.strip())

        elif layout_type == "RIGHT_SIDEBAR":
            # For RIGHT_SIDEBAR: Read Main Body (Left Column) FIRST, Sidebar (Right Column) SECOND
            if left_text.strip():
                page_parts.append(left_text.strip())
            if right_text.strip():
                page_parts.append("\n--- SIDEBAR ---")
                page_parts.append(right_text.strip())

        else:
            # TWO_COLUMN_BALANCED: Read Left Column FIRST, Right Column SECOND
            if left_text.strip():
                page_parts.append(left_text.strip())
            if right_text.strip():
                page_parts.append(right_text.strip())

        page_texts.append("\n\n".join(page_parts))

    return "\n\n".join(page_texts)


def process_pdf_file(pdf_path: str, output_dir: Path):
    """Processes a single PDF file, segments sections, extracts experience, and saves text output."""
    print(f"\n{'═' * 70}")
    print(f"PROCESSING PDF: {Path(pdf_path).name}")
    print(f"{'═' * 70}")

    extracted_text = read_pdf_sidebar_aware(pdf_path)
    print(f"[Extracted Text Character Length: {len(extracted_text):,}]")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem

    # Save extracted text
    text_out_path = output_dir / f"{stem}_extracted.txt"
    with open(text_out_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    print(f"[Saved Extracted Text to: {text_out_path}]")

    # Run Section Splitter
    if split_into_sections is not None:
        sections = split_into_sections(extracted_text)
        print(f"\n--- SECTION SEGMENTATION RESULTS ({len(sections)} sections) ---")
        for s in sections:
            print(f"  [{s.label.upper()}] (conf: {round(s.confidence, 2)}, start_line: {s.start_line})")
            snippet = s.raw_text.replace('\n', ' ')[:120]
            print(f"      -> {snippet}...")

        # Save segmented output
        seg_out_path = output_dir / f"{stem}_segmented.txt"
        with open(seg_out_path, "w", encoding="utf-8") as f:
            for s in sections:
                f.write(f"\n{'─' * 60}\n")
                f.write(f"LABEL: {s.label}   CONFIDENCE: {round(s.confidence, 2)}   START LINE: {s.start_line}\n")
                f.write(f"{'─' * 60}\n")
                f.write(s.raw_text + "\n")
        print(f"\n[Saved Segmented Output to: {seg_out_path}]")

        # Run Experience Extractor
        if extract_experience is not None:
            exp_sections = [s for s in sections if s.label == "experience"]
            exp_text = "\n".join(s.raw_text for s in exp_sections) if exp_sections else ""
            if exp_text.strip():
                exp_res = extract_experience(exp_text)
                jobs = exp_res.get("experience", [])
                print(f"\n--- EXPERIENCE EXTRACTION RESULTS ({len(jobs)} entries) ---")
                for idx, j in enumerate(jobs, 1):
                    print(f"  Entry {idx}:")
                    print(f"    Role      : {j.get('role')}")
                    print(f"    Company   : {j.get('company')}")
                    print(f"    Start-End : {j.get('start')} - {j.get('end')}")
                    print(f"    Bullets   : {len(j.get('bullets', []))} bullet points")
            else:
                print("\n[No experience section found to extract entries.]")
    else:
        print("\n[segmenter module not imported; printed extracted text only]")


def main():
    output_dir = Path("output_sidebar_test")

    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target) and target.lower().endswith(".pdf"):
            process_pdf_file(target, output_dir)
        elif os.path.isdir(target):
            pdf_files = sorted(glob.glob(os.path.join(target, "*.pdf"))) + sorted(glob.glob(os.path.join(target, "*.PDF")))
            if not pdf_files:
                print(f"No .pdf files found inside directory '{target}'")
                return
            print(f"Found {len(pdf_files)} PDF resume files in '{target}' directory.\n")
            for pdf_file in pdf_files:
                process_pdf_file(pdf_file, output_dir)
        else:
            print(f"Invalid target file or directory: '{target}'")
    else:
        # Default fallback: check 'pdfs/' or 'resumes/' directory
        default_folders = ["pdfs", "resumes"]
        found_any = False
        for folder in default_folders:
            if os.path.exists(folder):
                pdf_files = sorted(glob.glob(os.path.join(folder, "*.pdf")))
                if pdf_files:
                    found_any = True
                    print(f"Found {len(pdf_files)} PDF resumes in '{folder}/' folder:\n")
                    for pdf_file in pdf_files:
                        process_pdf_file(pdf_file, output_dir)
                    break
        if not found_any:
            print("No input target passed and no 'pdfs/' or 'resumes/' folder found with PDF files.")
            print("Usage: python test_sidebar_column_parser.py <path_to_pdf_or_folder>")


if __name__ == "__main__":
    main()
