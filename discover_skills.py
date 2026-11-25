#!/usr/bin/env python3
"""
discover_skills.py
==================
Automated Skill Discovery Pipeline using Groq Free LLM.
Scans resumes one-by-one, extracts skills, cross-checks against skills_taxonomy.json,
and generates a clean list of missing skills in data/pending_skills.json for human review.

Zero duplicate guarantee:
- Any skill already in taxonomy is automatically ignored.
- Only genuinely novel/missing skills are accumulated.

Usage:
    # Scan a folder of new resumes (e.g. 10 resumes for testing):
    python discover_skills.py --dir resumes --limit 10

    # Scan all resumes in a specific directory:
    python discover_skills.py --dir /path/to/new_resumes

    # Scan using pre-segmented text files (fastest):
    python discover_skills.py --segmented-dir segmented_text --limit 20

    # Scan a single resume file:
    python discover_skills.py --file resumes/candidate.pdf
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from skill_discovery.groq_client import GroqSkillExtractor, DEFAULT_MODEL
from skill_discovery.discovery_engine import SkillDiscoveryEngine
from ingestion import read_resume_file
from segmenter import split_into_sections

DEFAULT_OUTPUT = BASE_DIR / "data" / "pending_skills.json"
DEFAULT_TAXONOMY = BASE_DIR / "data" / "skills_taxonomy.json"
DEFAULT_TRACKER = BASE_DIR / "data" / "scanned_resumes.json"
DEFAULT_LOG = BASE_DIR / "data" / "llm_extraction_log.json"


def extract_skills_text_from_file(file_path: Path) -> str:
    """Extracts strictly the skills section from a raw resume file."""
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    else:
        raw_text = read_resume_file(str(file_path))

    if not raw_text or len(raw_text.strip()) < 20:
        return ""

    try:
        sections = split_into_sections(raw_text)
        skills_text = ""

        for s in sections:
            if s.label == "skills":
                skills_text += "\n" + s.raw_text

        # Strict: If explicit skills section exists, use ONLY that!
        if skills_text.strip() and len(skills_text.strip()) > 10:
            return f"SKILLS SECTION:\n{skills_text.strip()}"

        # Option A (Strict Mode — Remove Fallback):
        # Do NOT fall back to raw_text / experience / summary!
        return ""
    except Exception:
        return ""


def extract_skills_text_from_segmented_file(file_path: Path) -> str:
    """Extracts text from a segmented text artifact strictly under 'LABEL: skills'."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    in_target = False
    target_lines = []

    for line in lines:
        if line.startswith("LABEL:"):
            label = line.split("CONFIDENCE:")[0].replace("LABEL:", "").strip().lower()
            in_target = (label == "skills")
            continue
        if line.startswith("───"):
            continue
        if in_target:
            target_lines.append(line)

    if target_lines:
        return "SKILLS SECTION:\n" + "\n".join(target_lines)[:8000]
    return ""


def load_existing_pending(path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads existing pending skills if already present so results accumulate."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def save_pending_skills(pending_dict: Dict[str, Dict[str, Any]], path: Path):
    """Saves pending skills sorted by frequency descending."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort by frequency descending
    sorted_items = sorted(
        pending_dict.items(),
        key=lambda kv: kv[1].get("frequency", 1),
        reverse=True
    )
    sorted_dict = {k: v for k, v in sorted_items}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted_dict, f, indent=2, ensure_ascii=False)


def load_scanned_tracker(path: Path = DEFAULT_TRACKER) -> set:
    """Loads set of resume filenames that have already been scanned."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return set(data)
                elif isinstance(data, dict):
                    return set(data.keys())
        except Exception:
            pass
    return set()


def save_scanned_tracker(scanned_set: set, path: Path = DEFAULT_TRACKER):
    """Saves updated set of scanned resume filenames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(scanned_set)), f, indent=2)


def load_extraction_log(path: Path = DEFAULT_LOG) -> Dict[str, Any]:
    """Loads audit log of raw LLM extractions per resume."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def save_extraction_log(log_data: Dict[str, Any], path: Path = DEFAULT_LOG):
    """Saves audit log of raw LLM extractions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Automated Skill Discovery Pipeline using Groq Free LLM"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to directory containing resume files (.pdf, .docx, .doc, .txt)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single resume file to scan"
    )
    parser.add_argument(
        "--segmented-dir",
        type=str,
        default=None,
        help="Path to segmented_text directory (faster pre-extracted text)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of resumes to process (for quick testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path for pending skills (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Groq model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset/clear existing pending_skills.json before starting"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Force re-scanning all resumes, ignoring tracking history"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed list of raw skills extracted by LLM on console"
    )

    args = parser.parse_args()

    # Verify Groq API Key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n❌ ERROR: GROQ_API_KEY environment variable is not set!")
        print("Set it via: export GROQ_API_KEY='your-groq-key' or add it to .env\n")
        sys.exit(1)

    output_path = Path(args.output)
    pending_dict = {} if args.reset else load_existing_pending(output_path)
    extraction_log = {} if args.reset else load_extraction_log()
    if pending_dict and not args.reset:
        print(f"ℹ️  Loaded {len(pending_dict)} existing pending skills from: {output_path.name}")

    print(f"\n{'═' * 70}")
    print(f"🚀 ATS SKILL DISCOVERY PIPELINE (Groq Free LLM)")
    print(f"{'═' * 70}")
    print(f"• LLM Model       : {args.model}")
    print(f"• Taxonomy Base   : {DEFAULT_TAXONOMY}")
    print(f"• Staging Output  : {output_path}")

    # Initialize Engine
    try:
        extractor = GroqSkillExtractor(api_key=api_key, model=args.model)
        engine = SkillDiscoveryEngine(taxonomy_path=DEFAULT_TAXONOMY, extractor=extractor)
        print(f"• Active Taxonomy : {len(engine.taxonomy):,} canonical skills loaded")
    except Exception as e:
        print(f"\n❌ Failed to initialize engine: {e}")
        sys.exit(1)

    # Gather files
    files_to_process: List[Path] = []
    is_segmented = False

    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        files_to_process = [p]
    elif args.segmented_dir:
        seg_p = Path(args.segmented_dir)
        if not seg_p.exists():
            print(f"❌ Directory not found: {args.segmented_dir}")
            sys.exit(1)
        files_to_process = sorted(list(seg_p.glob("*_segmented.txt")))
        is_segmented = True
    elif args.dir:
        dir_p = Path(args.dir)
        if not dir_p.exists():
            print(f"❌ Directory not found: {args.dir}")
            sys.exit(1)
        valid_exts = {".pdf", ".docx", ".doc", ".txt"}
        files_to_process = sorted([f for f in dir_p.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    else:
        # Default to resumes directory if it exists
        default_res_dir = BASE_DIR / "resumes"
        if default_res_dir.exists():
            valid_exts = {".pdf", ".docx", ".doc", ".txt"}
            files_to_process = sorted([f for f in default_res_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
            print(f"• Source Directory: {default_res_dir} (default)")
        else:
            print("❌ Please specify --dir, --file, or --segmented-dir.")
            sys.exit(1)

    if not files_to_process:
        print("❌ No matching resume files found to process.")
        sys.exit(1)

    # Filter for new un-scanned resumes unless --all or --file is specified
    scanned_tracker = load_scanned_tracker()
    if not args.all and not args.file and scanned_tracker:
        new_files = [f for f in files_to_process if f.name not in scanned_tracker]
        skipped_count = len(files_to_process) - len(new_files)
        print(f"• Already Scanned : {skipped_count:,} resume(s) previously (skipped)")
        print(f"• NEW Resumes     : {len(new_files):,} new resume(s) detected to scan")
        files_to_process = new_files

    if not files_to_process:
        print("\n✅ All resumes in the directory have already been scanned for skill discovery!")
        print("   No new un-scanned resumes found. (Use --all to force re-scanning everything)\n")
        return

    if args.limit and args.limit > 0:
        files_to_process = files_to_process[:args.limit]

    total_files = len(files_to_process)
    print(f"• Files to Scan   : {total_files} resume(s)")
    print(f"{'═' * 70}\n")

    t_start = time.perf_counter()
    total_extracted = 0
    total_known = 0
    total_new = 0

    for idx, fpath in enumerate(files_to_process, 1):
        resume_name = fpath.name
        # Short display name
        display_name = resume_name[:40] + "..." if len(resume_name) > 40 else resume_name
        print(f"[{idx:>3}/{total_files}] Scanning: {display_name:<43} ", end="", flush=True)

        t0 = time.perf_counter()
        try:
            if is_segmented:
                text_to_scan = extract_skills_text_from_segmented_file(fpath)
            else:
                text_to_scan = extract_skills_text_from_file(fpath)

            if not text_to_scan or len(text_to_scan.strip()) < 15:
                print("✓ [No Skills Section Found - 0 skills]")
                scanned_tracker.add(resume_name)
                extraction_log[resume_name] = {
                    "total_extracted": 0,
                    "all_skills_extracted_by_llm": [],
                    "already_in_taxonomy": [],
                    "new_missing_skills": [],
                    "status": "No dedicated skills section found"
                }
                continue

            stats = engine.process_skills_from_text(resume_name, text_to_scan, pending_dict)
            dt = time.perf_counter() - t0

            total_extracted += stats["extracted"]
            total_known += stats["already_known"]
            total_new += stats["new_found"]
            scanned_tracker.add(resume_name)

            # Record audit log
            extraction_log[resume_name] = {
                "total_extracted": stats["extracted"],
                "all_skills_extracted_by_llm": stats.get("all_extracted_skills", []),
                "already_in_taxonomy": stats.get("known_skills", []),
                "new_missing_skills": stats.get("missing_skills", [])
            }

            if stats["new_found"] > 0:
                print(f"✨ Found {stats['new_found']:>2} new  ({stats['already_known']} known) [{dt:.2f}s]")
            else:
                print(f"✓ All known   ({stats['already_known']} skills) [{dt:.2f}s]")

            if args.verbose:
                print(f"      ├─ LLM Extracted : {', '.join(stats.get('all_extracted_skills', []))}")
                if stats.get('missing_skills'):
                    print(f"      └─ \033[1;33mMissing Skills\033[0m : {', '.join(stats.get('missing_skills', []))}")

            # Incremental save every 5 files to avoid data loss
            if idx % 5 == 0:
                save_pending_skills(pending_dict, output_path)
                save_scanned_tracker(scanned_tracker)
                save_extraction_log(extraction_log)

            # Polite pause to stay smoothly within Groq 8,000 TPM limit
            time.sleep(1.5)

        except KeyboardInterrupt:
            print("\n\n🛑 Process interrupted by user. Saving progress...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    # Final save
    save_pending_skills(pending_dict, output_path)
    save_scanned_tracker(scanned_tracker)
    save_extraction_log(extraction_log)
    total_time = time.perf_counter() - t_start

    print(f"\n{'═' * 70}")
    print(f"🎉 DISCOVERY COMPLETE IN {total_time:.2f}s")
    print(f"{'═' * 70}")
    print(f"• Resumes Scanned        : {total_files}")
    print(f"• Total Skills Extracted : {total_extracted:,}")
    print(f"• Skills Already Known   : {total_known:,} (skipped to prevent duplicates)")
    print(f"• Unique Missing Skills  : {len(pending_dict):,} saved to {output_path.name}")
    print(f"{'═' * 70}\n")
    print("👉 NEXT STEP: Review and approve the newly discovered skills:")
    print("   Run: python approve_skills.py\n")


if __name__ == "__main__":
    main()
