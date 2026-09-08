"""
Certifications & Achievements Extractor — Layer 2
Extracts: certification names, issuing bodies, years, achievement statements.

DESIGN NOTE (matches the architecture diagram's single "Certs & achievements"
box): certs and achievements often live in one segmented section, or in two
separate ones depending on how the resume is written. extract_certifications()
therefore takes both a cert_section_text and an optional achievement_section_text,
same as the 4.6 draft — pass "" for achievement_section_text if your segmenter
doesn't produce a separate "achievements" label (e.g. route a "custom
section" / "training" label there instead, since that's where extra
training/credential-adjacent lines tend to land, as in Aarti's resume).
"""

import re

# Known certification issuers / training platforms — expand as needed.
# Longer, more specific phrases are listed so they win over short generic ones
# when both could match (e.g. "Amazon Web Services" vs "Google").
CERT_ISSUERS = [
    "Amazon Web Services", "AWS", "Google Cloud Platform", "Google Cloud", "GCP",
    "Google", "Microsoft Azure", "Microsoft", "Azure", "Cisco", "Oracle", "Meta",
    "Facebook", "Coursera", "Udemy", "edX", "LinkedIn Learning",
    "Project Management Institute", "PMI", "PMP", "Scrum Alliance", "SAFe",
    "CompTIA", "Red Hat", "IBM", "Salesforce", "MongoDB University",
    "HackerRank", "HackerEarth", "Databricks", "Snowflake", "Tableau",
    "HashiCorp", "Kubernetes", "CNCF", "ITIL", "Six Sigma", "PRINCE2", "SAP",
    "NASSCOM", "upGrad", "Simplilearn", "Great Learning", "DataCamp", "ISTQB",
]
ISSUER_RE = re.compile(
    r'\b(' + '|'.join(re.escape(i) for i in CERT_ISSUERS) + r')\b',
    re.IGNORECASE
)
YEAR_RE = re.compile(r'\b(20\d{2}|19\d{2})\b')

BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*]\s+|^\d+\.\s+')
SEPARATORS = ["–", "—", "-", "|", ":"]

ACHIEVEMENT_KEYWORDS_RE = re.compile(
    r'\b(award(?:ed)?|winner|recogni[sz]ed|ranked|top\s+\d|honou?r|medal|'
    r'scholarship|published|patent|certificate of appreciation)\b',
    re.IGNORECASE
)


def _strip_trailing_punct(s: str, strip_parens: bool = False) -> str:
    chars = " -–—|,:()" if strip_parens else " -–—|,:"
    return s.strip(chars).strip()


def parse_cert_line(line: str) -> dict:
    """Parses a single certification line into name / issuer / year."""
    cert = {"name": None, "issuer": None, "year": None, "confidence": 0.6, "needs_review": True}
    original = line

    line = BULLET_RE.sub("", line).strip()

    year_match = YEAR_RE.search(line)
    if year_match:
        cert["year"] = year_match.group(1)
        # remove the year and any wrapping punctuation around it, e.g. "(2022)"
        line = (line[:year_match.start()] + line[year_match.end():])
        line = re.sub(r'\(\s*\)', '', line)  # leftover empty parens
        line = _strip_trailing_punct(line, strip_parens=True)

    # Try splitting on a separator where the right-hand side is (mostly) a
    # known issuer — that means the resume wrote "Cert Name - Issuing Body".
    # If the issuer is just an inherent part of the cert's own name (e.g.
    # "AWS Certified Solutions Architect"), there's no such clean split, and
    # we correctly fall through to keeping the whole line as the name.
    split_done = False
    for sep in SEPARATORS:
        if sep in line:
            left, _, right = line.rpartition(sep)
            left, right = left.strip(), right.strip()
            if left and right and ISSUER_RE.fullmatch(right):
                cert["name"] = left
                cert["issuer"] = right
                split_done = True
                break

    if not split_done:
        cert["name"] = line if line else original.strip()
        issuer_match = ISSUER_RE.search(line)
        if issuer_match:
            cert["issuer"] = issuer_match.group(1)

    cert["name"] = _strip_trailing_punct(cert["name"], strip_parens=False) or original.strip()

    # confidence scoring
    score = 0.6
    if cert["year"]:
        score += 0.15
    if cert["issuer"]:
        score += 0.2
    cert["confidence"] = round(min(score, 0.95), 2)
    cert["needs_review"] = cert["confidence"] < 0.75

    return cert


def extract_certifications(cert_section_text: str, achievement_section_text: str = "") -> dict:
    """
    Extracts certifications and achievements.
    """
    result = {
        "certifications": [],
        "achievements": [],
        "_confidence": {},
        "_needs_review": False,
    }

    # ── Certifications ──
    if cert_section_text.strip():
        for line in cert_section_text.split("\n"):
            line = line.strip()
            if not line or len(line) < 5:
                continue
            if line.startswith("─") or line.startswith("[") or line.startswith("LABEL:"):
                continue
            # skip section-heading-only lines like "Certifications" / "Licenses"
            if re.match(r'^certif|^licen', line, re.IGNORECASE) and len(line) < 25:
                continue
            cert = parse_cert_line(line)
            if cert["name"]:
                result["certifications"].append(cert)

        if result["certifications"]:
            avg = sum(c["confidence"] for c in result["certifications"]) / len(result["certifications"])
            result["_confidence"]["certifications"] = round(avg, 2)
            if any(c["needs_review"] for c in result["certifications"]):
                result["_needs_review"] = True
        else:
            result["_confidence"]["certifications"] = 0.0

    # ── Achievements ──
    if achievement_section_text.strip():
        for raw in achievement_section_text.split("\n"):
            line = BULLET_RE.sub("", raw.strip()).strip()
            if line and len(line) > 8:
                result["achievements"].append(line)
        result["_confidence"]["achievements"] = 0.85 if result["achievements"] else 0.0

    return result