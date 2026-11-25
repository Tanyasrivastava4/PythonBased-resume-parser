"""
layer3/flatten.py

Transforms a build_resume_record() output into the flat, display-ready
schema — one dict per resume, with a "<field>_confidence" sibling key
next to every field so confidence is visible without nesting each
value into an object.
"""

import re
from pathlib import Path

from layer3.assemble import build_resume_record, parse_segmented_file
from segmenter.section_splitter import _SPOKEN_LANGUAGE_KEYWORDS
from extractors.demographics import extract_demographics


def _fmt_education_line(i: int, entry: dict) -> str:
    degree = entry["degree_name"] or entry["degree_type"] or "Unknown"
    institution = entry["institution"] or "Unknown"
    year = entry["year"] or "None"
    return (f"{i}. {degree} — {institution} ({year}) "
            f"[conf {entry['_confidence']}, review={entry['_needs_review']}]")


_NON_TITLE_DISQUALIFIED = {
    "the project", "project", "projects", "unknown role", "unknown - role", "unknown", "role",
    "title", "description", "responsibilities", "key responsibilities", "experience",
    "work experience", "summary", "present", "current", "duration", "employment"
}

_GEO_LOCATION_PATTERN = re.compile(
    r"^(?:(?:[A-Za-z\s]+,\s*(?:india|usa|uk|uae|canada|singapore|australia))|"
    r"(?:delhi|new\s*delhi|gurgaon|gurugram|noida|mumbai|pune|bangalore|bengaluru|hyderabad|chennai|kolkata|lucknow|indore|jaipur|ahmedabad|chandigarh|mohali)(?:,\s*[A-Za-z\s]+)?)$",
    re.I
)


def _sanitize_title(raw_title: str) -> str:
    if not raw_title:
        return ""
    # Strip leading bullet artifacts & symbols: Clean ^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\—\|\s\d\.\)\(]+
    t = re.sub(r"^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\—\|\s\d\.\)\(]+", "", str(raw_title)).strip()
    # Strip trailing empty parens, brackets, colons, dashes
    t = re.sub(r"\s*\(\s*\)$", "", t).strip()
    t = re.sub(r"[\s:\-–—,|]+$", "", t).strip()
    if t.lower() in _NON_TITLE_DISQUALIFIED or len(t) < 3 or t.isdigit() or _GEO_LOCATION_PATTERN.match(t):
        return ""
    words = t.split()
    if len(words) >= 3 and len(set(w.lower() for w in words)) == 1:
        return ""
    return t


def _sanitize_company(raw_comp: str) -> str:
    if not raw_comp:
        return ""
    c = re.sub(r"^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\—\|\s\d\.\)\(]+", "", str(raw_comp)).strip()
    c = re.sub(r"\s*\(\s*\)$", "", c).strip()
    c = re.sub(r"[\s:\-–—,|]+$", "", c).strip()
    if c.count("(") > c.count(")"):
        c += ")"
    if c.lower() in ("unknown company", "company", "unknown", "none", "null") or (len(c) <= 2 and c.isdigit()):
        return ""
    return c


def _fmt_experience_line(i: int, job: dict) -> str:
    role = _sanitize_title(job.get("role")) or job.get("role") or "Unknown role"
    comp = _sanitize_company(job.get("company")) or job.get("company") or "Unknown company"
    start = job.get("start")
    end = job.get("end")
    if not start and end:
        date_str = f"({end})"
    elif start and not end:
        date_str = f"({start})"
    elif start and end:
        date_str = f"({start} - {end})"
    else:
        date_str = "(? - ?)"
    return (f"{i}. {role} @ {comp} {date_str} "
            f"[conf {job.get('confidence', 0.95)}, review={job.get('needs_review', False)}]")



def _fmt_cert_line(i: int, cert: dict) -> str:
    name = cert["name"] or "Unknown"
    issuer = f" — {cert['issuer']}" if cert["issuer"] else ""
    year = f" ({cert['year']})" if cert["year"] else ""
    return f"{i}. {name}{issuer}{year} [conf {cert['confidence']}]"


def _extract_languages_simple(text: str, fallback_text: str = "") -> str:
    text_to_scan = text if (text and text.strip()) else fallback_text
    if not text_to_scan:
        return ""
    found = []
    for word in text_to_scan.replace(",", " ").replace("\n", " ").split():
        clean = word.strip(".,;:()•·-").lower()
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

    _GEO_LOCATION_PATTERN = re.compile(
        r"^(?:(?:[A-Za-z\s]+,\s*(?:india|usa|uk|uae|canada|singapore|australia))|"
        r"(?:delhi|new\s*delhi|gurgaon|gurugram|noida|mumbai|pune|bangalore|bengaluru|hyderabad|chennai|kolkata|lucknow|indore|jaipur|ahmedabad|chandigarh|mohali)(?:,\s*[A-Za-z\s]+)?)$",
        re.I
    )

    seen_titles = []
    for job in exp["jobs"]:
        r = _sanitize_title(job.get("role"))
        if r and not any(r.lower() == st.lower() for st in seen_titles):
            seen_titles.append(r)

    # Recover any prominent titles from the experience text that were separated by multi-column layout
    exp_raw = sections.get("experience", "") + "\n" + sections.get("roles_responsibilities", "")
    _TITLE_RECOVERY = re.compile(
        r"\b((?:Senior\s+|Lead\s+|Associate\s+|Assistant\s+|Regional\s+)?(?:Sales\s+Manager|Growth\s+Manager|Business\s+Development\s+Manager|Product\s+Manager|Account\s+Manager|Marketing\s+Manager|Client\s+Solutions|GTM\s+Intern|UI/UX\s+Designer|UX\s+Designer|Software\s+Engineer|Data\s+Analyst|Consultant|Chartered\s+Accountant|Internal\s+Auditor|Risk\s+Consultant))\b",
        re.I
    )
    for m in _TITLE_RECOVERY.finditer(exp_raw):
        found_title = _sanitize_title(m.group(1))
        if found_title and not any(found_title.lower() == st.lower() for st in seen_titles):
            seen_titles.append(found_title)

    # Structured work experience preserving all bullets & responsibilities
    structured_work_exp = []
    for job in exp["jobs"]:
        clean_r = _sanitize_title(job.get("role")) or job.get("role")
        clean_c = _sanitize_company(job.get("company")) or job.get("company")
        b_list = [
            b.strip() for b in job.get("bullets", [])
            if len(b.strip()) > 5 and not _GEO_LOCATION_PATTERN.match(b)
        ]
        structured_work_exp.append({
            "role": clean_r,
            "company": clean_c,
            "start": job.get("start"),
            "end": job.get("end"),
            "responsibilities": b_list
        })

    company_details = [
        f"EXPERIENCE ({len(exp['jobs'])} jobs, {len(exp['early_career'])} early_career entries)",
        f"Total experience: {te['total_years']} years ({te['total_months']} months"
        + (f", {te['overlapping_periods_merged']} overlaps merged)" if te['overlapping_periods_merged'] else ")"),
    ]
    company_details += [_fmt_experience_line(i, j) for i, j in enumerate(exp["jobs"], 1)]

    languages_text = sections.get("languages", "")
    if not languages_text:
        pd_text = sections.get("personal_details", "")
        m = re.search(r"(?:languages\s*known|spoken\s*languages|languages)\s*[:\-]\s*([^\n]+)", full_text, re.I)
        if m:
            languages_text = m.group(1)
        elif pd_text:
            languages_text = pd_text

    demo = extract_demographics(
        text=full_text,
        name=contact["name"]["value"],
        education_lines=[_fmt_education_line(i, e) for i, e in enumerate(edu["entries"], 1)],
        total_experience=te["total_years"]
    )

    flat = {
        "email": contact["email"]["value"],
        "email_confidence": contact["email"]["confidence"],

        "mobile_number": contact["phone"]["value"],
        "mobile_number_confidence": contact["phone"]["confidence"],

        "name": contact["name"]["value"],
        "name_confidence": contact["name"]["confidence"],

        "gender": demo["gender"],
        "gender_confidence": demo["gender_confidence"],

        "dob": demo["dob"],
        "dob_confidence": demo["age_confidence"],

        "age": demo["age"],
        "age_confidence": demo["age_confidence"],

        "nationality": demo.get("nationality"),
        "nationality_confidence": demo.get("nationality_confidence"),

        "citizenship": demo.get("citizenship"),
        "citizenship_confidence": demo.get("citizenship_confidence"),

        "marital_status": demo.get("marital_status"),
        "marital_status_confidence": demo.get("marital_status_confidence"),

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

        "work_experience": structured_work_exp,
        "work_experience_confidence": exp["_confidence"],

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

        "projects": [p.strip() for p in sections.get("projects", "").splitlines() if len(p.strip()) > 3 and not re.match(r"^(?:projects?|key projects?|academic projects?)[:\-]?$", p.strip(), re.I)][:10],
        "projects_confidence": 0.85 if sections.get("projects") else None,
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