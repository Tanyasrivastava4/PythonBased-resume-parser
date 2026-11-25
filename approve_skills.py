#!/usr/bin/env python3
"""
approve_skills.py
=================
Interactive Human-in-the-Loop Review & Merge Tool for Discovered Skills.
Reads candidate skills from data/pending_skills.json, allows you to review,
edit, approve, or reject them, and safely merges approved skills into
data/skills_taxonomy.json with zero duplicates or collisions.

Usage:
    # Interactive 1-by-1 review:
    python approve_skills.py

    # 1-Click approve all pending skills:
    python approve_skills.py --all

    # Approve only high-frequency skills (seen in >= 2 resumes):
    python approve_skills.py --min-freq 2

    # View pending skills without modifying:
    python approve_skills.py --list
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent
PENDING_FILE = BASE_DIR / "data" / "pending_skills.json"
TAXONOMY_FILE = BASE_DIR / "data" / "skills_taxonomy.json"
TAXONOMY_BACKUP = BASE_DIR / "data" / "skills_taxonomy.json.bak"
LOG_FILE = BASE_DIR / "data" / "llm_extraction_log.json"


def load_pending_skills() -> Dict[str, Dict[str, Any]]:
    if not PENDING_FILE.exists():
        return {}
    try:
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"❌ Error reading {PENDING_FILE}: {e}")
        return {}


def save_pending_skills(data: Dict[str, Dict[str, Any]]):
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_taxonomy() -> Dict[str, Any]:
    if not TAXONOMY_FILE.exists():
        raise FileNotFoundError(f"Taxonomy file not found at: {TAXONOMY_FILE}")
    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_taxonomy(taxonomy: Dict[str, Any]):
    # Create backup first
    if TAXONOMY_FILE.exists():
        shutil.copyfile(TAXONOMY_FILE, TAXONOMY_BACKUP)

    with open(TAXONOMY_FILE, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)


def merge_approved_skills(approved_skills: List[Dict[str, Any]]) -> int:
    """Safely merges approved skills into data/skills_taxonomy.json."""
    if not approved_skills:
        return 0

    taxonomy = load_taxonomy()
    added_count = 0

    # Build existing lookup for collision check
    canonical_lookup = {k.strip().lower(): k for k in taxonomy.keys()}
    alias_lookup = {}
    for canonical, entry in taxonomy.items():
        for a in entry.get("aliases", []):
            norm_a = str(a).strip().lower()
            if norm_a:
                alias_lookup[norm_a] = canonical

    for item in approved_skills:
        cname = item["canonical_name"].strip()
        norm_c = cname.lower()

        # If already exists, just update/append new aliases
        if norm_c in canonical_lookup:
            existing_c = canonical_lookup[norm_c]
            curr_aliases = set(taxonomy[existing_c].get("aliases", []))
            for a in item.get("aliases", []):
                norm_a = a.strip().lower()
                if norm_a and norm_a not in alias_lookup:
                    curr_aliases.add(norm_a)
            taxonomy[existing_c]["aliases"] = sorted(list(curr_aliases))
            continue

        # Clean aliases to avoid collisions
        valid_aliases = []
        for a in item.get("aliases", []):
            norm_a = a.strip().lower()
            if not norm_a or norm_a == norm_c:
                continue
            if norm_a not in alias_lookup and norm_a not in canonical_lookup:
                valid_aliases.append(norm_a)
                alias_lookup[norm_a] = cname

        taxonomy[cname] = {
            "aliases": sorted(list(set(valid_aliases))),
            "category": item.get("category", "other"),
            "source": "groq_discovery"
        }
        canonical_lookup[norm_c] = cname
        added_count += 1

    save_taxonomy(taxonomy)
    return added_count


def list_pending(pending: Dict[str, Dict[str, Any]]):
    print(f"\n📋 Currently Pending Skills ({len(pending)} total):")
    print(f"{'─' * 70}")
    print(f"{'#':<3} {'Skill Name':<28} {'Category':<18} {'Freq':<6} {'Aliases'}")
    print(f"{'─' * 70}")
    for idx, (name, item) in enumerate(pending.items(), 1):
        aliases_str = ", ".join(item.get("aliases", []))
        if len(aliases_str) > 25:
            aliases_str = aliases_str[:22] + "..."
        print(f"{idx:<3} {name:<28} {item.get('category','other'):<18} {item.get('frequency',1):<6} {aliases_str}")
    print(f"{'─' * 70}\n")


def list_history():
    """Lists all skills that have been discovered and merged into skills_taxonomy.json."""
    taxonomy = load_taxonomy()
    discovered = {k: v for k, v in taxonomy.items() if v.get("source") == "groq_discovery"}
    print(f"\n🌟 Skills Discovered & Added to Taxonomy ({len(discovered)} total):")
    print(f"{'─' * 70}")
    print(f"{'#':<3} {'Skill Name':<32} {'Category':<20} {'Aliases'}")
    print(f"{'─' * 70}")
    for idx, (name, item) in enumerate(discovered.items(), 1):
        aliases_str = ", ".join(item.get("aliases", []))
        if len(aliases_str) > 25:
            aliases_str = aliases_str[:22] + "..."
        print(f"{idx:<3} {name:<32} {item.get('category','other'):<20} {aliases_str}")
    print(f"{'─' * 70}\n")


def show_llm_log(filter_name: str = None):
    """Shows the exact skills extracted by the LLM for each resume."""
    if not LOG_FILE.exists():
        print(f"\n⚠️ No extraction log found at {LOG_FILE.name}.")
        print("Run `python discover_skills.py` first to generate logs.\n")
        return

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception as e:
        print(f"Error reading {LOG_FILE.name}: {e}")
        return

    if not logs:
        print("\n⚠️ Extraction log is empty.\n")
        return

    matched = 0
    print(f"\n🤖 EXACT SKILLS EXTRACTED BY LLM ({len(logs)} resumes logged):")
    print(f"{'═' * 75}")

    for resume_name, data in logs.items():
        if filter_name and filter_name.lower() not in resume_name.lower():
            continue

        matched += 1
        all_skills = data.get("all_skills_extracted_by_llm", [])
        known = data.get("already_in_taxonomy", [])
        missing = data.get("new_missing_skills", [])

        print(f"\n📄 Resume: \033[1;36m{resume_name}\033[0m")
        print(f"   • Total Extracted by LLM : {len(all_skills)}")
        print(f"   • All Skills Found       : {', '.join(all_skills) if all_skills else 'None'}")
        print(f"   • Already in Taxonomy    : {', '.join(known) if known else 'None'}")
        print(f"   • Flagged as Missing     : \033[1;33m{', '.join(missing) if missing else 'None'}\033[0m")
        print(f"{'─' * 75}")

    if filter_name and matched == 0:
        print(f"No logged resumes matching '{filter_name}'.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Review and merge discovered skills into skills_taxonomy.json"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Approve and merge ALL pending skills in one shot"
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=None,
        help="Approve only skills appearing in at least N resumes"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all pending skills waiting for approval"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="List all skills that have been approved and added to taxonomy"
    )
    parser.add_argument(
        "--log",
        nargs="?",
        const="",
        help="Show exact skills extracted by LLM (optionally filter by resume name)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all pending skills"
    )

    args = parser.parse_args()

    # Handle history and log view first (available anytime)
    if args.history:
        list_history()
        return

    if args.log is not None:
        show_llm_log(filter_name=args.log.strip() if args.log else None)
        return

    pending = load_pending_skills()
    if args.list:
        if not pending:
            print("\n✅ No pending skills waiting for approval.")
        else:
            list_pending(pending)
        return

    if not pending:
        print("\n✅ No pending skills found in data/pending_skills.json.")
        print("   To discover new skills, run: python discover_skills.py\n")
        return

    if args.clear:
        save_pending_skills({})
        print(f"🧹 Cleared all {len(pending)} pending skills from {PENDING_FILE.name}")
        return

    if args.list:
        list_pending(pending)
        return

    # Fast batch mode: --all or --min-freq
    if args.all or args.min_freq is not None:
        min_f = args.min_freq or 1
        to_approve = []
        remaining = {}

        for name, item in pending.items():
            if item.get("frequency", 1) >= min_f:
                to_approve.append(item)
            else:
                remaining[name] = item

        if not to_approve:
            print(f"⚠️ No skills found with frequency >= {min_f}")
            return

        added = merge_approved_skills(to_approve)
        save_pending_skills(remaining)
        print(f"\n🎉 Successfully approved and merged {added} new skills into {TAXONOMY_FILE.name}!")
        print(f"   Remaining pending skills: {len(remaining)}")
        print(f"   Backup saved to: {TAXONOMY_BACKUP.name}\n")
        return

    # Interactive Review Mode
    print(f"\n{'═' * 70}")
    print(f"🔍 INTERACTIVE SKILL REVIEW ({len(pending)} pending skills)")
    print(f"{'═' * 70}")
    print("Commands:")
    print("  [y] Approve as-is")
    print("  [n] Reject / Skip (delete from queue)")
    print("  [e] Edit canonical name, aliases, or category")
    print("  [a] Approve ALL remaining skills")
    print("  [q] Save decisions and Quit\n")

    approved_list = []
    keys = list(pending.keys())
    remaining_keys = set(keys)

    for idx, key in enumerate(keys, 1):
        item = pending[key]
        cname = item.get("canonical_name", key)
        category = item.get("category", "other")
        freq = item.get("frequency", 1)
        aliases = item.get("aliases", [])
        resumes = item.get("resumes", [])

        resumes_preview = ", ".join(resumes[:3])
        if len(resumes) > 3:
            resumes_preview += f" (+{len(resumes) - 3} more)"

        print(f"{'─' * 70}")
        print(f"[{idx}/{len(keys)}] Skill: \033[1;36m{cname}\033[0m")
        print(f"   • Category  : {category}")
        print(f"   • Frequency : {freq} resume(s) [{resumes_preview}]")
        print(f"   • Aliases   : {aliases if aliases else 'None'}")

        while True:
            try:
                choice = input("   👉 Action [y/n/e/a/q] (default: y): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

            if choice in ["", "y"]:
                approved_list.append(item)
                remaining_keys.discard(key)
                print("   ✓ Approved!")
                break
            elif choice == "n":
                remaining_keys.discard(key)
                print("   ✗ Rejected / Removed.")
                break
            elif choice == "e":
                print("   Editing skill:")
                new_name = input(f"   - Canonical name [{cname}]: ").strip() or cname
                new_cat = input(f"   - Category [{category}]: ").strip() or category
                curr_als_str = ", ".join(aliases)
                new_als_str = input(f"   - Aliases comma-separated [{curr_als_str}]: ").strip()
                if new_als_str:
                    new_aliases = [a.strip().lower() for a in new_als_str.split(",") if a.strip()]
                else:
                    new_aliases = aliases

                item["canonical_name"] = new_name
                item["category"] = new_cat
                item["aliases"] = new_aliases
                approved_list.append(item)
                remaining_keys.discard(key)
                print("   ✓ Saved edits & Approved!")
                break
            elif choice == "a":
                # Approve this and all remaining
                approved_list.append(item)
                remaining_keys.discard(key)
                for rem_key in keys[idx:]:
                    approved_list.append(pending[rem_key])
                    remaining_keys.discard(rem_key)
                print(f"\n   ✓ Approved this and all remaining {len(keys) - idx} skills!")
                break
            elif choice == "q":
                print("\n   Saving decisions and exiting...")
                break
            else:
                print("   Invalid option. Please enter y, n, e, a, or q.")

        if choice in ["a", "q"]:
            break

    # Perform merge of all approved items
    if approved_list:
        added = merge_approved_skills(approved_list)
        # Update pending file
        new_pending = {k: pending[k] for k in remaining_keys if k in pending}
        save_pending_skills(new_pending)
        print(f"\n{'═' * 70}")
        print(f"🎉 MERGE COMPLETE!")
        print(f"• Approved & Merged : {added} new skill(s) into {TAXONOMY_FILE.name}")
        print(f"• Remaining Pending : {len(new_pending)} skill(s)")
        print(f"• Backup Created    : {TAXONOMY_BACKUP.name}")
        print(f"{'═' * 70}\n")
    else:
        print("\nℹ️ No skills were approved in this session.\n")


if __name__ == "__main__":
    main()
