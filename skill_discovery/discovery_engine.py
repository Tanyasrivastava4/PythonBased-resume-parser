"""
discovery_engine.py
===================
Engine to discover new skills from resumes, check against current taxonomy
to guarantee ZERO duplicates, and aggregate missing skills into a review queue.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from .groq_client import GroqSkillExtractor


class SkillDiscoveryEngine:
    """Discovers new skills from resumes and compares against existing taxonomy."""

    def __init__(
        self,
        taxonomy_path: Optional[Path] = None,
        extractor: Optional[GroqSkillExtractor] = None
    ):
        if taxonomy_path is None:
            # Default to ats_parser/data/skills_taxonomy.json
            self.taxonomy_path = Path(__file__).resolve().parent.parent / "data" / "skills_taxonomy.json"
        else:
            self.taxonomy_path = Path(taxonomy_path)

        self.extractor = extractor or GroqSkillExtractor()
        self.taxonomy = self._load_taxonomy()
        self._build_lookup_indices()

    def _load_taxonomy(self) -> Dict[str, Any]:
        """Loads existing skills_taxonomy.json."""
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found at: {self.taxonomy_path}")
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_lookup_indices(self):
        """
        Builds lowercased inverted lookup dictionaries for:
        1. Canonical skill names (e.g. 'python' -> 'Python')
        2. Aliases (e.g. 'py' -> 'Python')
        """
        self.canonical_lookup: Dict[str, str] = {}
        self.alias_lookup: Dict[str, str] = {}

        for canonical, entry in self.taxonomy.items():
            norm_c = canonical.strip().lower()
            self.canonical_lookup[norm_c] = canonical

            for a in entry.get("aliases", []):
                norm_a = str(a).strip().lower()
                if norm_a:
                    self.alias_lookup[norm_a] = canonical

    def is_skill_in_taxonomy(self, canonical_name: str, aliases: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Determines if a candidate skill or any of its aliases already exists in taxonomy.
        Returns: (True, existing_canonical) if found, else (False, None).
        """
        norm_c = canonical_name.strip().lower()
        if norm_c in self.canonical_lookup:
            return True, self.canonical_lookup[norm_c]
        if norm_c in self.alias_lookup:
            return True, self.alias_lookup[norm_c]

        # Check aliases
        for a in aliases:
            norm_a = str(a).strip().lower()
            if norm_a in self.canonical_lookup:
                return True, self.canonical_lookup[norm_a]
            if norm_a in self.alias_lookup:
                return True, self.alias_lookup[norm_a]

        return False, None

    def filter_unclaimed_aliases(self, canonical_name: str, aliases: List[str]) -> List[str]:
        """
        Removes any alias that is already claimed by another existing skill in taxonomy,
        ensuring zero collisions.
        """
        norm_c = canonical_name.strip().lower()
        cleaned = []
        for a in aliases:
            norm_a = str(a).strip().lower()
            if not norm_a or norm_a == norm_c:
                continue
            # If alias is already mapped to an existing taxonomy skill, do not claim it
            if norm_a in self.canonical_lookup or norm_a in self.alias_lookup:
                continue
            cleaned.append(norm_a)
        return list(dict.fromkeys(cleaned))

    def process_skills_from_text(
        self,
        resume_name: str,
        text: str,
        pending_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Extracts skills from text via Groq, compares with taxonomy,
        and adds only MISSING skills into pending_dict.

        Returns statistics: {"extracted": X, "already_known": Y, "new_found": Z}
        """
        extracted = self.extractor.extract_skills_from_text(text)
        known_count = 0
        new_count = 0
        known_skills = []
        new_skills = []

        for skill in extracted:
            cname = skill["canonical_name"]
            aliases = skill.get("aliases", [])
            category = skill.get("category", "other")

            is_known, existing_name = self.is_skill_in_taxonomy(cname, aliases)
            if is_known:
                known_count += 1
                known_skills.append(f"{cname} (in taxonomy as '{existing_name}')")
                continue

            # This is a genuine missing skill!
            new_count += 1
            new_skills.append(cname)
            valid_aliases = self.filter_unclaimed_aliases(cname, aliases)

            # Check if this skill is already in our pending_dict (from another resume)
            key = cname.strip()
            # Case-insensitive key check
            existing_key = None
            for k in pending_dict.keys():
                if k.lower() == key.lower():
                    existing_key = k
                    break

            if existing_key:
                entry = pending_dict[existing_key]
                entry["frequency"] += 1
                if resume_name not in entry["resumes"]:
                    entry["resumes"].append(resume_name)
                # Merge aliases
                for a in valid_aliases:
                    if a not in entry["aliases"]:
                        entry["aliases"].append(a)
            else:
                pending_dict[key] = {
                    "canonical_name": key,
                    "aliases": valid_aliases,
                    "category": category,
                    "frequency": 1,
                    "resumes": [resume_name],
                    "status": "pending"
                }

        return {
            "extracted": len(extracted),
            "already_known": known_count,
            "new_found": new_count,
            "all_extracted_skills": [s["canonical_name"] for s in extracted],
            "known_skills": known_skills,
            "missing_skills": new_skills
        }
