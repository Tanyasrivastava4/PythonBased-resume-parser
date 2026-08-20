"""
layer3/flatten.py

Transforms a build_resume_record() output into the flat, display-ready
schema — one dict per resume, with a "<field>_confidence" sibling key
next to every field so confidence is visible without nesting each
value into an object.
"""

from pathlib import Path

from layer3.assemble import build_resume_record, parse_segmented_file
from segmenter.section_splitter import _SPOKEN_LANGUAGE_KEYWORDS


def _fmt_education_line(i: int, entry: dict) -> str:
    degree = entry["degree_name"] or entry["degree_type"] or "Unknown"
    institution = entry["institution"] or "Unknown"
    year = entry["year"] or "None"
    return (f"{i}. {degree} — {institution} ({year}) "
            f"[conf {entry['_confidence']}, review={entry['_needs_review']}]")


def _fmt_experience_line(i: int, job: dict) -> str:
    role = job["role"] or "Unknown role"
    company = job["company"] or "Unknown company"
    start = job["start"] or "?"
    end = job["end"] or "?"
    return (f"{i}. {role} @ {company} ({start} - {end}) "
            f"[conf {job['confidence']}, review={job['needs_review']}]")


def _fmt_cert_line(i: int, cert: dict) -> str:
    name = cert["name"] or "Unknown"
    issuer = f" — {cert['issuer']}" if cert["issuer"] else ""
    year = f" ({cert['year']})" if cert["year"] else ""
    return f"{i}. {name}{issuer}{year} [conf {cert['confidence']}]"


def _extract_languages_simple(text: str) -> str:
    if not text:
        return ""
    found = []
    for word in text.replace(",", " ").split():
        clean = word.strip(".,;:()").lower()
        if clean in _SPOKEN_LANGUAGE_KEYWORDS and clean.capitalize() not in found:
            found.append(clean.capitalize())
    return ", ".join(found)


def flatten_resume_record(segmented_filepath: str) -> dict:
    """
    Builds the full Layer 3 record, then reshapes it into the flat
    display schema. Every field gets a "<field>_confidence" sibling:
      - Contact fields (email, mobile_number, name, linkedin, github,
        website): the extractor's own per-field confidence, unchanged.
      - skills / certification / Achievements: the SECTION-level
        confidence already computed by that extractor (one number for
        the whole list — see skills.py / certifications.py).
      - EDUCATION / company_details: the section-level average
        confidence too, even though each individual line ALSO carries
        its own inline "[conf X, review=Y]" — the sibling key gives a
        single number to sort/filter resumes by without parsing every
        line's inline text.
      - summary / languages: no extractor scores these yet (no
        dedicated Layer 2 extractor exists for either), so their
        confidence is set to None rather than a made-up number.
    """
    record = build_resume_record(segmented_filepath)
    full_text = Path(segmented_filepath).read_text(encoding="utf-8")
    sections = parse_segmented_file(full_text)

    contact = record["contact"]
    exp = record["experience"]
    edu = record["education"]
    cert = record["certifications"]
    ach = record["achievements"]
    te = exp["total_experience"]

    seen_degrees = []
    for e in edu["entries"]:
        label = e["degree_name"] or e["degree_type"]
        if label and label not in seen_degrees:
            seen_degrees.append(label)

    seen_titles = []
    for job in exp["jobs"]:
        if job["role"] and job["role"] not in seen_titles:
            seen_titles.append(job["role"])

    company_details = [
        f"EXPERIENCE ({len(exp['jobs'])} jobs, {len(exp['early_career'])} early_career entries)",
        f"Total experience: {te['total_years']} years ({te['total_months']} months"
        + (f", {te['overlapping_periods_merged']} overlaps merged)" if te['overlapping_periods_merged'] else ")"),
    ]
    company_details += [_fmt_experience_line(i, j) for i, j in enumerate(exp["jobs"], 1)]

    languages_text = sections.get("languages", "")

    flat = {
        "email": contact["email"]["value"],
        "email_confidence": contact["email"]["confidence"],

        "mobile_number": contact["phone"]["value"],
        "mobile_number_confidence": contact["phone"]["confidence"],

        "name": contact["name"]["value"],
        "name_confidence": contact["name"]["confidence"],

        "linkedin": contact["linkedin"]["value"],
        "linkedin_confidence": contact["linkedin"]["confidence"],

        "github": contact["github"]["value"],
        "github_confidence": contact["github"]["confidence"],

        "website": contact["website"]["value"],
        "website_confidence": contact["website"]["confidence"],

        "summary": record["summary"],
        "summary_confidence": None,  # no dedicated summary extractor exists yet

        "EDUCATION": [_fmt_education_line(i, e) for i, e in enumerate(edu["entries"], 1)],
        "EDUCATION_confidence": edu["_confidence"],

        "degree": seen_degrees,

        "company_details": company_details,
        "company_details_confidence": exp["_confidence"],

        "designation": seen_titles,

        "skills": record["skills"]["items"],
        "skills_confidence": record["skills"]["_confidence"],

        "languages": _extract_languages_simple(languages_text),
        "languages_confidence": None,  # no dedicated language extractor yet — see note below

        "total_experience": te["total_years"],
        "total_experience_confidence": exp["_confidence"],

        "certification": [_fmt_cert_line(i, c) for i, c in enumerate(cert["items"], 1)],
        "certification_confidence": cert["_confidence"],

        "Achievements": ach["items"],
        "Achievements_confidence": ach["_confidence"],
    }

    # Drop fields with nothing behind them (value AND its confidence
    # both empty/None) rather than keeping a placeholder — but keep a
    # field's confidence key paired with it, never one without the other.
    cleaned = {}
    keys = list(flat.keys())
    i = 0
    while i < len(keys):
        k = keys[i]
        if k.endswith("_confidence"):
            i += 1
            continue
        conf_key = f"{k}_confidence"
        value = flat[k]
        if value not in (None, "", [], {}):
            cleaned[k] = value
            if conf_key in flat:
                cleaned[conf_key] = flat[conf_key]
        i += 1

    return cleaned