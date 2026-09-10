"""
run_pipeline.py
===============
Unified End-to-End ATS Resume Parser Pipeline

Executes all pipeline layers in pure Python with a single function call or CLI command:
  Layer 0: Ingestion (read PDF/DOCX/DOC/Image/TXT)
  Layer 1: Section Segmentation
  Layer 2 & 3: Extraction & Assembly
  Layer 4: Schema Flattening (final display JSON output)

CLI Usage:
    # Process a single resume:
    python run_pipeline.py "resumes/Akash Resume - Developer.pdf"

    # Process all resumes in resumes/ folder:
    python run_pipeline.py --all

Python API Usage:
    from run_pipeline import process_resume

    result_json = process_resume("resumes/candidate.pdf")
"""

import sys
import glob
import json
import argparse
from pathlib import Path

from ingestion import read_resume_file
from segmenter import split_into_sections
from layer3.flatten import flatten_resume_record


EXTRACTED_TEXT_DIR = Path("extracted_text")
SEGMENTED_DIR = Path("segmented_text")
FLAT_JSON_DIR = Path("flat_json")


def _save_raw_text(file_path: str, text: str, output_dir: Path) -> Path:
    """Saves Layer 0 extracted raw text to extracted_text/ folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    out_path = output_dir / f"{stem}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def _save_segmented_text(file_path: str, sections: list, output_dir: Path) -> Path:
    """Saves Layer 1 segmented section output to segmented_text/ folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    out_path = output_dir / f"{stem}_segmented.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for section in sections:
            f.write(f"\n{'─' * 60}\n")
            f.write(
                f"LABEL: {section.label}   "
                f"CONFIDENCE: {round(section.confidence, 2)}   "
                f"START LINE: {section.start_line}\n"
            )
            f.write(f"{'─' * 60}\n")
            f.write(section.raw_text)
            f.write("\n")

        labels_found = [s.label for s in sections]
        f.write(f"\n{'─' * 60}\n")
        f.write(f"[Found {len(sections)} sections total: {labels_found}]\n")

    return out_path


def process_resume(
    file_path: str,
    output_dir: str = "flat_json",
    verbose: bool = True
) -> dict:
    """
    Processes a single resume file through the entire ATS parser pipeline.
    
    Args:
        file_path: Path to resume file (.pdf, .docx, .doc, .png, .jpg, .txt)
        output_dir: Folder to save final flat JSON output
        verbose: Whether to print progress to console
        
    Returns:
        dict: The final flat JSON output record
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    if verbose:
        print(f"\n{'═' * 65}")
        print(f"PROCESSING RESUME : {input_path.name}")
        print(f"{'═' * 65}")

    # Layer 0: Ingestion
    if input_path.suffix.lower() == ".txt":
        raw_text = input_path.read_text(encoding="utf-8")
    else:
        raw_text = read_resume_file(str(input_path))

    extracted_txt_path = _save_raw_text(str(input_path), raw_text, EXTRACTED_TEXT_DIR)
    if verbose:
        print(f"✓ Layer 0 (Ingestion)     : Extracted {len(raw_text):,} characters -> {extracted_txt_path}")

    # Layer 1: Section Segmentation
    sections = split_into_sections(raw_text)
    segmented_txt_path = _save_segmented_text(str(input_path), sections, SEGMENTED_DIR)
    if verbose:
        labels = [s.label for s in sections]
        print(f"✓ Layer 1 (Segmentation)  : Found {len(sections)} sections {labels} -> {segmented_txt_path}")

    # Layer 2, 3 & 4: Extraction, Assembly & Flattening
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    flat_record = flatten_resume_record(str(segmented_txt_path))

    flat_json_path = out_dir_path / f"{input_path.stem}_flat.json"
    with open(flat_json_path, "w", encoding="utf-8") as f:
        json.dump(flat_record, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✓ Layer 2-4 (Extract/Flat): Generated output file -> {flat_json_path}")
        print(f"{'═' * 65}\n")

    return flat_record


def process_all(
    input_dir: str = "resumes",
    output_dir: str = "flat_json"
) -> list[dict]:
    """
    Processes all resume files in a folder through the ATS parser pipeline.
    
    Args:
        input_dir: Directory containing resume files
        output_dir: Directory to save flat JSON outputs
        
    Returns:
        list[dict]: List of generated flat JSON records
    """
    folder = Path(input_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    supported_extensions = ["*.pdf", "*.docx", "*.doc", "*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(str(folder / ext)))
        files.extend(glob.glob(str(folder / ext.upper())))

    files = sorted(list(set(files)))
    if not files:
        print(f"No supported resume files found in {input_dir}/")
        return []

    print(f"\n🚀 Starting batch execution for {len(files)} resume(s) in '{input_dir}/'...")
    results = []
    success_count = 0
    fail_count = 0

    for idx, f in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}]")
        try:
            res = process_resume(f, output_dir=output_dir, verbose=True)
            results.append(res)
            success_count += 1
        except Exception as e:
            print(f"❌ ERROR processing {f}: {e}")
            fail_count += 1

    print(f"\n{'═' * 65}")
    print(f"BATCH COMPLETE: {success_count} succeeded, {fail_count} failed out of {len(files)} files.")
    print(f"Outputs saved to: {output_dir}/")
    print(f"{'═' * 65}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end ATS Resume Parser Pipeline.")
    parser.add_argument("file", nargs="?", help="Path to a single resume file to process.")
    parser.add_argument("--all", action="store_true", help="Process all resumes in the default resumes/ folder.")
    parser.add_argument("--dir", type=str, default=None, help="Process all resumes in a custom directory.")
    parser.add_argument("--output", type=str, default="flat_json", help="Directory to save flat JSON outputs.")

    args = parser.parse_args()

    if args.all or args.dir:
        target_dir = args.dir if args.dir else "resumes"
        process_all(input_dir=target_dir, output_dir=args.output)
    elif args.file:
        process_resume(args.file, output_dir=args.output, verbose=True)
    else:
        # Default behavior if no arguments passed: process resumes/ folder
        process_all(input_dir="resumes", output_dir=args.output)
