r"""
Layer 3 — Structured JSON Assembly + Confidence Scoring

Takes one Layer 1 *_segmented.txt file, routes each section to its
Layer 2 extractor (same routing every test_*.py script already does
individually), merges every extractor's output into one JSON record,
computes an overall confidence score, and builds a review queue of
every field that came back low-confidence or missing.

DESIGN NOTES:

- Section parsing: reuses the same SECTION_RE pattern every test_*.py
  script duplicates, but fixed to collect EVERY block matching a label
  (not just the first), same fix already applied in
  test_experience_extractor.py's extract_section() -- a resume can
  legitimately have the same label appear twice (confirmed on a real
  resume, Abhinav Ashish), and dropping everything after the first
  occurrence silently discards real content. Centralizing this here
  means all five extractors get the fix, not just experience.

- early_career routing: experience.py's extract_experience() takes an
  optional early_career_section_text param specifically for the
  "early_career" label the section splitter produces (condensed
  one-line-per-job recaps). Wired through here.

- achievements routing: certifications.py's extract_certifications()
  takes an optional achievement_section_text. The section splitter can
  produce either "achievements" or "other" for this content depending
  on the resume's own heading wording -- fall back to "other" if
  "achievements" wasn't found, same as test_certifications.py already
  does.

- Review queue: every extractor already computes its own per-field or
  per-entry confidence/needs_review flags (contact fields, skills,
  each experience job, each education entry, each certification).
  Rather than re-deriving confidence here, Layer 3 just WALKS those
  existing flags and collects anything under REVIEW_THRESHOLD (or
  already flagged needs_review by its own extractor) into one flat
  "_review_queue" list with a dotted field path, so a downstream
  reviewer or the LLM fallback (per the architecture diagram) can find
  exactly which field to look at without re-parsing the whole record.

- Overall confidence: a simple average of every SECTION that actually
  produced content (a section with zero output isn't a confidence
  failure, it's just absent from the resume -- e.g. no certifications
  section shouldn't drag down someone's overall score). Sections that
  found nothing are excluded from the average rather than counted as 0.
"""

import re
from pathlib import Path
from datetime import datetime, timezone

from extractors.contact_extractor import extract_contact
from extractors.skills import extract_skills
from extractors.experience import extract_experience
from extractors.education import extract_education
from extractors.certifications import extract_certifications


REVIEW_THRESHOLD = 0.75

SECTION_RE = re.compile(
    r"LABEL:\s*(?P<label>\w+).*?\n[─\-]+\n(?P<body>.*?)(?=\n[─\-]+\s*\nLABEL:|\n\[Found|\Z)",
    re.DOTALL
)


def parse_segmented_file(full_text: str) -> dict:
    """
    Parses a Layer 1 *_segmented.txt file's raw text into
    {label: combined_text}. Collects EVERY block matching a label and
    joins them with a blank line -- see module docstring for why this
    matters (a resume can have the same label twice).
    """
    sections = {}
    for m in SECTION_RE.finditer(full_text):
        label = m.group("label").strip().lower()
        body = m.group("body").strip()
        if not body:
            continue
        if label in sections:
            sections[label] += "\n\n" + body
        else:
            sections[label] = body
    return sections


# ── Schema validation ──────────────────────────────────────────────

REQUIRED_TOP_LEVEL_KEYS = {
    "meta", "contact", "summary", "skills", "experience",
    "education", "certifications", "achievements",
    "_confidence", "_needs_review", "_review_queue",
}


def validate_schema(record: dict) -> list:
    """
    Lightweight structural check -- confirms every extractor's output
    landed in the shape downstream layers (embeddings, Qdrant storage)
    expect. Returns a list of violation messages; empty means valid.
    This is deliberately shallow (top-level shape, not exhaustive type
    checking of every nested field) since each extractor already
    guarantees its own internal shape -- this just catches the case
    where merging itself went wrong (e.g. a key renamed, a None where
    a dict was expected).
    """
    errors = []

    missing = REQUIRED_TOP_LEVEL_KEYS - record.keys()
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")

    if not isinstance(record.get("contact"), dict):
        errors.append("contact must be a dict")
    if not isinstance(record.get("skills", {}).get("items"), list):
        errors.append("skills.items must be a list")
    if not isinstance(record.get("experience", {}).get("jobs"), list):
        errors.append("experience.jobs must be a list")
    if not isinstance(record.get("education", {}).get("entries"), list):
        errors.append("education.entries must be a list")
    if not isinstance(record.get("certifications", {}).get("items"), list):
        errors.append("certifications.items must be a list")
    if not isinstance(record.get("_review_queue"), list):
        errors.append("_review_queue must be a list")

    return errors


# ── Review-queue helpers ────────────────────────────────────────────

def _flag(review_queue: list, field: str, confidence: float, reason: str):
    review_queue.append({
        "field": field,
        "confidence": confidence,
        "reason": reason,
    })


# ── Main assembly ────────────────────────────────────────────────────

def build_resume_record(segmented_filepath: str) -> dict:
    """
    Reads one *_segmented.txt file, runs every section through its
    Layer 2 extractor, and returns one merged, schema-validated JSON
    record with an overall confidence score and a review queue.
    """
    path = Path(segmented_filepath)
    full_text = path.read_text(encoding="utf-8")
    sections = parse_segmented_file(full_text)

    review_queue = []
    component_confidences = []

    # ── Contact ──────────────────────────────────────────────────
    # extract_contact scans the FULL text (not just a section), since
    # LinkedIn/GitHub links sometimes sit outside the header -- see
    # contact_extractor.py's own docstring.
    contact = extract_contact(full_text)
    contact_found_confs = []
    for field_name, field_data in contact.items():
        if field_data["value"]:
            contact_found_confs.append(field_data["confidence"])
            if field_data["confidence"] < REVIEW_THRESHOLD:
                _flag(review_queue, f"contact.{field_name}", field_data["confidence"], "low_confidence")
        # A missing name/email is worth flagging even at 0 confidence,
        # since those two are the fields a recruiter can't do without.
        elif field_name in ("name", "email"):
            _flag(review_queue, f"contact.{field_name}", 0.0, "missing")
    if contact_found_confs:
        component_confidences.append(sum(contact_found_confs) / len(contact_found_confs))

    # ── Skills ───────────────────────────────────────────────────
    skills_result = extract_skills(sections=sections, full_text=full_text)
    if skills_result["skills"]:
        component_confidences.append(skills_result["_confidence"])
        if skills_result["_confidence"] < REVIEW_THRESHOLD:
            _flag(review_queue, "skills", skills_result["_confidence"], "low_confidence")
    elif "skills" in sections:
        # A skills section existed but nothing was extracted from it --
        # worth a look, unlike a resume that simply has no skills section.
        _flag(review_queue, "skills", 0.0, "section_present_but_empty")

    # ── Experience (+ early_career recap, if the splitter found one) ──
    exp_text = sections.get("experience", "")
    early_career_text = sections.get("early_career", "")
    experience_result = extract_experience(exp_text, early_career_text)
    has_experience_content = bool(experience_result["experience"] or experience_result["early_career"])
    if has_experience_content:
        component_confidences.append(experience_result["_confidence"])
        if experience_result["_confidence"] < REVIEW_THRESHOLD:
            _flag(review_queue, "experience", experience_result["_confidence"], "low_confidence")
    elif exp_text or early_career_text:
        _flag(review_queue, "experience", 0.0, "section_present_but_empty")

    for i, job in enumerate(experience_result["experience"]):
        if job["needs_review"]:
            _flag(review_queue, f"experience.jobs[{i}]", job["confidence"], "incomplete_fields")
    for i, job in enumerate(experience_result["early_career"]):
        if job["needs_review"]:
            _flag(review_queue, f"experience.early_career[{i}]", job["confidence"], "incomplete_fields")

    # ── Education ────────────────────────────────────────────────
    edu_text = sections.get("education", "")
    education_result = extract_education(edu_text)
    if education_result["education"]:
        component_confidences.append(education_result["_confidence"])
        if education_result["_confidence"] < REVIEW_THRESHOLD:
            _flag(review_queue, "education", education_result["_confidence"], "low_confidence")
    elif edu_text:
        _flag(review_queue, "education", 0.0, "section_present_but_empty")

    for i, entry in enumerate(education_result["education"]):
        if entry["_needs_review"]:
            _flag(review_queue, f"education.entries[{i}]", entry["_confidence"], "incomplete_fields")

    # ── Certifications & Achievements ───────────────────────────
    # The section splitter can land extra credential/training content
    # under "achievements" OR "other" depending on the resume's own
    # heading wording -- fall back to "other" the same way
    # test_certifications.py already does.
    cert_text = sections.get("certifications", "")
    achievement_text = sections.get("achievements", "") or sections.get("other", "")
    cert_result = extract_certifications(cert_text, achievement_text)

    cert_conf = cert_result["_confidence"].get("certifications")
    ach_conf = cert_result["_confidence"].get("achievements")

    if cert_result["certifications"]:
        component_confidences.append(cert_conf)
        if cert_conf is not None and cert_conf < REVIEW_THRESHOLD:
            _flag(review_queue, "certifications", cert_conf, "low_confidence")
    elif cert_text:
        _flag(review_queue, "certifications", 0.0, "section_present_but_empty")

    for i, c in enumerate(cert_result["certifications"]):
        if c["needs_review"]:
            _flag(review_queue, f"certifications.items[{i}]", c["confidence"], "incomplete_fields")

    # ── Summary ──────────────────────────────────────────────────
    # No dedicated Layer 2 extractor exists for this yet -- pass the
    # raw section text through as-is so it isn't lost, and so Layer 5
    # (embedding generation) has real prose to embed even before a
    # summary extractor exists.
    summary_text = sections.get("summary", "").strip() or None

    # ── Overall confidence ───────────────────────────────────────
    overall_confidence = (
        round(sum(component_confidences) / len(component_confidences), 2)
        if component_confidences else 0.0
    )

    record = {
        "meta": {
            "source_file": path.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sections_found": sorted(sections.keys()),
        },
        "contact": contact,
        "summary": summary_text,
        "skills": {
            "items": skills_result["skills"],
            "by_category": skills_result["skills_by_category"],
            "_confidence": skills_result["_confidence"],
            "_method": skills_result["_method"],
        },
        "experience": {
            "jobs": experience_result["experience"],
            "early_career": experience_result["early_career"],
            "total_experience": experience_result["_total_experience"],
            "_confidence": experience_result["_confidence"],
        },
        "education": {
            "entries": education_result["education"],
            "_confidence": education_result["_confidence"],
        },
        "certifications": {
            "items": cert_result["certifications"],
            "_confidence": cert_conf if cert_conf is not None else 0.0,
        },
        "achievements": {
            "items": cert_result["achievements"],
            "_confidence": ach_conf if ach_conf is not None else 0.0,
        },
        "_confidence": {
            "overall": overall_confidence,
        },
        "_needs_review": False,   # set below, after schema check
        "_review_queue": review_queue,
    }

    schema_errors = validate_schema(record)
    record["meta"]["schema_errors"] = schema_errors

    record["_needs_review"] = bool(review_queue) or bool(schema_errors)

    return record