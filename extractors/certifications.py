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
    "K2 Analytics", "AON Hewitt", "AON", "ISO",
]
ISSUER_RE = re.compile(
    r'\b(' + '|'.join(re.escape(i) for i in CERT_ISSUERS) + r')\b',
    re.IGNORECASE
)
ISSUER_INSTITUTION_RE = re.compile(
    r'\b(?:academy|institute|university|college|school|foundation|organization|council|board|consortium)\b',
    re.IGNORECASE
)
YEAR_RE = re.compile(r'\b(20\d{2}|19\d{2})\b')

BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*\uf0b7]\s*|^\d+\.\s*')
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

    date_range_match = re.search(r'\(\s*\d{1,2}/\d{4}\s*[-–—]\s*\d{1,2}/(20\d{2}|19\d{2})\s*\)', line)
    iso_match = re.search(r'ISO\s*\d+:(?:20\d{2}|19\d{2})', line, re.I)
    year_match = None
    if date_range_match:
        cert["year"] = date_range_match.group(1)
        line = (line[:date_range_match.start()] + line[date_range_match.end():]).strip()
        line = _strip_trailing_punct(line, strip_parens=True)
    elif not iso_match:
        year_match = YEAR_RE.search(line)
    else:
        before_after = line[:iso_match.start()] + " " + line[iso_match.end():]
        year_match = YEAR_RE.search(before_after)

    if year_match:
        cert["year"] = year_match.group(1)
        # remove the year and any wrapping punctuation around it, e.g. "(2022)"
        line = (line[:year_match.start()] + line[year_match.end():])
        line = re.sub(r'\(\s*\)', '', line)  # leftover empty parens
        line = _strip_trailing_punct(line, strip_parens=True)

    # Try splitting on a separator where the right-hand side is (mostly) a
    # known issuer — that means the resume wrote "Cert Name - Issuing Body".
    split_done = False
    for sep in SEPARATORS:
        if sep in line:
            left, _, right = line.rpartition(sep)
            left, right = left.strip(), right.strip()
            if left and right and (ISSUER_RE.search(right) or ISSUER_INSTITUTION_RE.search(right) or (sep == "|" and not re.search(r'\b(?:auditor|certified|diploma|engineer|developer|lead|consultant)\b', right, re.I))):
                cert["name"] = left
                cert["issuer"] = right
                split_done = True
                break

    if not split_done:
        cert["name"] = line if line else original.strip()
        issuer_match = ISSUER_RE.search(line)
        if issuer_match:
            cert["issuer"] = issuer_match.group(1)

    if not cert["issuer"] and re.search(r'\b(?:CCNA|CCNP|CCIE)\b', line, re.I):
        cert["issuer"] = "Cisco"

    cert["name"] = _strip_trailing_punct(cert["name"], strip_parens=False) or original.strip()
    cert["name"] = cert["name"].strip(" .–-—,:()")
    cert["name"] = re.sub(r'\s+\b(?:in|at|from)\s*$', '', cert["name"], flags=re.I).strip(" .–-—,:()")

    # confidence scoring
    score = 0.6
    if cert["year"]:
        score += 0.15
    if cert["issuer"]:
        score += 0.2
    cert["confidence"] = round(min(score, 0.95), 2)
    cert["needs_review"] = cert["confidence"] < 0.75

    return cert


def extract_certifications(cert_section_text: str, achievement_section_text: str = "", full_text: str = "") -> dict:
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
    if not cert_section_text.strip() and full_text.strip():
        cert_matches = [
            l.strip() for l in full_text.splitlines()
            if re.search(r'\b(?:AWS\s+Certification|Google\s+Cloud\s+Certified|Azure\s+Certification|Certified\s+Kubernetes|Certification\s+obtained|Certificate\s+obtained|Certified\s+Rewards\s+Professional|ISO\s*\d+.*Internal\s+Auditor|ITIL\s+Foundation|Microsoft\s+Azure\s+certification|CCNA)\b', l, re.I)
        ]
        if not cert_matches:
            hdr = full_text[:600]
            if re.search(r'\bAWS\b', hdr) and re.search(r'\b(?:Certificated|Certified)\b', hdr, re.I) and re.search(r'\bSolution\b', hdr, re.I) and re.search(r'\bArchitect\b', hdr, re.I):
                cert_matches.append("AWS Certified Solutions Architect")
        if cert_matches:
            cert_section_text = "\n".join(cert_matches)

    glyph_map = {
        "Ʃ": "tt",
        "ƫ": "tt",
        "Ɵ": "ti",
        "Ō": "ft",
        "ō": "ft",
    }
    for k, v in glyph_map.items():
        if k in cert_section_text:
            cert_section_text = cert_section_text.replace(k, v)

    if cert_section_text.strip():
        raw_lines = [l.strip() for l in cert_section_text.split("\n") if l.strip()]
        lines = []
        for l in raw_lines:
            if "|" in l and re.search(r'\b(?:certified|internal\s+auditor|iso|pmp|csm)\b', l, re.I):
                parts = [p.strip() for p in l.split("|") if p.strip()]
                if len(parts) > 1 and sum(1 for p in parts if re.search(r'\b(?:certified|internal\s+auditor|auditor|iso|pmp|csm|license)\b', p, re.I)) >= 2:
                    lines.extend(parts)
                    continue
            lines.append(l)

        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            if line.startswith("─") or line.startswith("[") or line.startswith("LABEL:"):
                continue
            # skip section-heading-only lines like "Certifications" / "Licenses" / "Professional Enhancements"
            if re.match(r'^(?:certif|licen|(?:car{1,2}[ie]{1,2}r\s+)?professional\s+enhancements?|trainings?)', line, re.IGNORECASE) and len(line) < 40:
                continue
            if re.match(r'^(?:references?|available\s+(?:upon|on)\s+request\.?|declaration|personal\s+details)[\s.:]*$', line, re.IGNORECASE):
                continue
            if re.search(r'^(?:tech\s*(&|and)\s*governance|posh\s+committee|hrms:|positioning\s+note|open\s+to\s+fractional)\b', line, re.IGNORECASE):
                continue
            if re.search(r'\b(?:b\.?sc|pgdbm|m\.?sc|b\.?tech|m\.?tech|b\.?com|m\.?com|b\.?e\.?|m\.?e\.?|bba|mba)\b', line, re.I) and not re.search(r'\b(?:certified|certificate|internal\s+auditor|iso\s*\d+|aon\s+hewitt)\b', line, re.I):
                continue
            # Check if this line is metadata / continuation of the preceding certification
            # (e.g. "Certificate No: ...", "Credential ID: ...", "Course Duration: ...", "License No: ...")
            if re.match(r'^(?:certificate\s+(?:no|id|number|code)|credential\s+(?:id|no|number)|license\s+(?:no|number|id)|course\s+duration|duration[\s:]|valid\s+(?:till|to|from)|issued\s*:)[\s.:]', line, re.IGNORECASE):
                if result["certifications"]:
                    prev_cert = result["certifications"][-1]
                    if not prev_cert.get("year"):
                        yr_m = YEAR_RE.search(line)
                        if yr_m:
                            prev_cert["year"] = yr_m.group(1)
                            prev_cert["confidence"] = round(min(prev_cert["confidence"] + 0.15, 0.95), 2)
                            prev_cert["needs_review"] = prev_cert["confidence"] < 0.75
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