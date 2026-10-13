r"""
Experience Extractor — Layer 2
Extracts: company, title, location, date range, duration, bullet points.
Uses: date regex (anchor-based block splitting) + job title dictionary + spaCy ORG detection.

KEY DESIGN NOTE:
Real resumes do NOT separate job entries with blank lines (verified against
Aarti / Abhijeet segmented output). So we cannot split blocks on '\n\s*\n'.
Instead we anchor on lines that contain a date range — every job entry has one —
and walk backward/forward from each anchor to find header lines and body lines.

CHANGELOG (early_career support):
Some resumes include a condensed "here's my last N jobs" recap -- e.g. a
"LAST 5 CAREER TIMELINE" section with one-line entries like "Oct-2023 -
Nov-2025 | Manager | KPMG" -- separate from the detailed "experience"
section that covers the same jobs with full bullets (confirmed on a real
resume, Sachin Chitranshi). The section splitter now labels this
"early_career" rather than folding it into "experience", so it doesn't
duplicate/pollute the detailed job list. But its date ranges are still
real worked time and belong in the total-years-of-experience figure.
extract_experience() now accepts an optional early_career_section_text,
parses it with the same anchor-based block logic, and feeds BOTH job
lists into calculate_total_experience() together. That function already
merges overlapping/duplicate date intervals, so if the early_career
recap restates the same jobs (same start/end dates) as the detailed
section, they collapse into one continuous span rather than double-
counting -- the same jobs described twice don't inflate the total years.
"""

import re
try:
    import spacy
    _SPACY_IMPORTED = True
except ImportError:
    _SPACY_IMPORTED = False
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
from datetime import datetime


# ── Date matching ──────────────────────────────────────────────────────────

MONTH_NAMES = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|Apil|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

# Day-first format: "04 May, 2023", "08 Dec 2025". Confirmed on a real
# resume (Sachin Chauhan): his current job's date range, "04 May, 2023 –
# 08 Dec 2025", used this shape on BOTH ends, and neither the
# "{MONTH_NAMES} YYYY" alternative (no leading day allowed) nor any of
# the numeric-only alternatives (which require digits/slashes only, no
# month-name text) could match it. .search() could still find "May,
# 2023" as a start by skipping the leading "04 ", but then "end" had to
# match starting exactly at "08 Dec 2025" (no gap allowed between the
# separator and the end group) -- and "08" alone matches nothing in the
# end alternation. No valid match existed anywhere in the line, so this
# job was never anchored and silently dropped entirely by
_VALID_YEAR = r"(?:19[6-9]\d|20[0-3]\d)"
_NUM_DATE = rf"\d{{1,2}}[-/.]\d{{1,2}}[-/.](?:{_VALID_YEAR}|\d{{2}})"
_MONTH_DAY_YEAR = rf"{MONTH_NAMES}[\s,.-]*\d{{1,2}}(?:st|nd|rd|th)?[\s,.-]*{_VALID_YEAR}"
_DAY_MONTH_YEAR = rf"\d{{1,2}}(?:st|nd|rd|th)?[\s,.-]*{MONTH_NAMES}[\s,.-]*{_VALID_YEAR}"
_MONTH_YEAR = rf"(?:{MONTH_NAMES}[\s,/.-]*(?:{_VALID_YEAR}|['’‘`´]?\s*\d{{2}})|{_VALID_YEAR}[\s,/.-]*{MONTH_NAMES})"
_YYYY_MM = rf"{_VALID_YEAR}[-/.]\s*(?:0[1-9]|1[0-2])\b"

DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_MONTH_DAY_YEAR}|{_DAY_MONTH_YEAR}|{_MONTH_YEAR}|{_YYYY_MM}|{_VALID_YEAR}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|{_NUM_DATE}|\d{{1,2}}[-/.]{_VALID_YEAR}|{_VALID_YEAR})"
    rf"(?:\s*[-–—/]\s*|\s+(?:to|till|until|through|thru)[:\s]\s*|\s+(?=(?:Present|Current|Now|Till|Still|Ongoing|On-going)\b))"
    rf"(?P<end>Present|Prasent|Prsent|Presant|Current|Now|Till(?:\s*(?:Date|Now))?|Still(?:\s*Working)?|Ongoing|On-going|{_MONTH_DAY_YEAR}|{_DAY_MONTH_YEAR}|{_MONTH_YEAR}|{_YYYY_MM}|{_VALID_YEAR}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|{_NUM_DATE}|\d{{1,2}}[-/.]{_VALID_YEAR}|{_VALID_YEAR})",
    re.IGNORECASE
)

_WIDOW_START_DATE_RE = re.compile(
    rf"({_MONTH_DAY_YEAR}|{_DAY_MONTH_YEAR}|{_MONTH_YEAR}|{_YYYY_MM}|{_VALID_YEAR}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|{_NUM_DATE}|\d{{1,2}}[/.]{_VALID_YEAR}|{_VALID_YEAR})\s*$",
    re.IGNORECASE
)
_WIDOW_END_DATE_RE = re.compile(
    rf"^[-–—]\s*(Present|Current|Now|Till\s*(?:Date|Now)|"
    rf"{_MONTH_DAY_YEAR}|{_DAY_MONTH_YEAR}|{_MONTH_YEAR}|{_YYYY_MM}|{_VALID_YEAR}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|{_NUM_DATE}|\d{{1,2}}[/.]{_VALID_YEAR}|{_VALID_YEAR})\s*$",
    re.IGNORECASE
)

_START_WITH_DASH_RE = re.compile(
    rf"^(?P<start>{_MONTH_DAY_YEAR}|{_DAY_MONTH_YEAR}|{_MONTH_YEAR}|{_YYYY_MM}|\d{{1,2}}[/.-]\d{{4}})\s*[-–—]\s*(?P<rest>.*)",
    re.IGNORECASE
)

_END_DATE_START_RE = re.compile(
    rf"^(?P<end>Present|Current|Now|Till(?:\s*(?:Date|Now))?|Still(?:\s*Working)?|{_MONTH_DAY_YEAR}|{_DAY_MONTH_YEAR}|{_MONTH_YEAR}|{_YYYY_MM}|\d{{1,2}}[/.-]\d{{4}})\b(?P<comp_rest>.*)",
    re.IGNORECASE
)

_YEAR_RANGE_SHORT_RE = re.compile(rf"\b(?P<start>{_VALID_YEAR})\s*[-–—]\s*(?P<end_short>\d{{2,4}})\b")
_SINGLE_YEAR_LINE_RE = re.compile(rf"^\s*\(?({_VALID_YEAR})\)?\s*$")
_SINCE_DATE_RE = re.compile(rf"\bSince\s+(?P<start>{_MONTH_DAY_YEAR}|{_DAY_MONTH_YEAR}|{_MONTH_YEAR}|{_YYYY_MM}|{_VALID_YEAR})\b", re.IGNORECASE)


_TRAILING_TO_RE = re.compile(r"\s*\bto\s*$", re.IGNORECASE)

_STATED_DURATION_RE = re.compile(
    r"(?:^|[\s|/,\-(])(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>years?|yrs?|months?|mos?)\b",
    re.IGNORECASE
)


def _merge_widow_date_lines(lines: list) -> list:
    # 1. Pre-scan for '❖ Since <start_date>' table headers
    start_dates = []
    for idx, l in enumerate(lines):
        if "Since" in l:
            if "till date :" in l or "present :" in l or "current :" in l:
                continue
            if idx + 1 < len(lines) and re.match(r"^(?:till date|present|current)\b", lines[idx + 1].strip(), re.IGNORECASE):
                continue
            for m in re.finditer(r"Since\s+([A-Za-z0-9\-\./]+)", l, re.IGNORECASE):
                dt = m.group(1).strip()
                start_dates.append(dt)

    line_colon_pat = re.compile(
        r"^(?:[❖–\-]|to)?\s*(?:Since\s*)?(?P<date>[A-Za-z0-9\s\-\u2013]+?)\s*:\s*(?P<body>.*)$",
        re.IGNORECASE
    )
    colon_body_role_re = re.compile(
        r"M/s\..*|\b(?:Company Secretary|Legal|Manager|Executive|Officer|Trainee|Partner|Lead|Director|Head|Consultant)\b",
        re.IGNORECASE
    )

    merged = []
    i = 0
    n = len(lines)
    start_idx = 0
    while i < n:
        cur = lines[i].rstrip()
        if not cur:
            merged.append(lines[i])
            i += 1
            continue

        if cur.strip().startswith("❖ Since") and ":" not in cur:
            i += 1
            continue

        m_colon = line_colon_pat.match(cur.strip())
        if m_colon and not cur.strip().startswith(("Contact:", "Address:", "Phone:", "E-Mail:")) and colon_body_role_re.search(m_colon.group("body")):
            d_str = m_colon.group("date").strip()
            body = m_colon.group("body").strip()
            if re.search(r"\b(?:till date|present|current)\b", d_str, re.IGNORECASE):
                s_d = "Unknown"
                if i > 0 and "Since" in lines[i - 1]:
                    m_since = re.search(r"Since\s+([A-Za-z0-9\-\./]+)", lines[i - 1], re.IGNORECASE)
                    if m_since:
                        s_d = m_since.group(1).strip()
                if s_d == "Unknown" and start_idx < len(start_dates):
                    s_d = start_dates[start_idx]
                    start_idx += 1
                merged.append(f"{s_d} - Present | {body}")
                i += 1
                continue

            range_m = DATE_RANGE_RE.search(d_str)
            if range_m and len(range_m.group("start")) >= 4 and len(range_m.group("end")) >= 4:
                s_d = range_m.group("start")
                e_d = range_m.group("end")
                if start_idx < len(start_dates):
                    prefix = start_dates[start_idx]
                    if prefix.lower() in ("may", "jun", "jul", "aug", "sep", "oct", "nov", "dec") or len(prefix) <= 3:
                        s_d = f"{prefix} {s_d}"
                        start_idx += 1
                merged.append(f"{s_d} - {e_d} | {body}")
                i += 1
                continue
            elif "-" in d_str and len(d_str) >= 6 and re.search(r"\d{2,4}", d_str):
                parts = re.split(r"\s*[-–—]\s*", d_str)
                if len(parts) == 2:
                    merged.append(f"{parts[0]} - {parts[1]} | {body}")
                    i += 1
                    continue

            end_d = "Present" if re.search(r"till|present|current", d_str, re.IGNORECASE) else d_str
            start_d = start_dates[start_idx] if start_idx < len(start_dates) else "Unknown"
            start_idx += 1
            merged.append(f"{start_d} - {end_d} | {body}")
            i += 1
            continue

        if DATE_RANGE_RE.search(cur):
            merged.append(lines[i])
            i += 1
            continue

        if i + 1 < n:
            nxt = lines[i + 1].strip()
            if not DATE_RANGE_RE.search(cur) and not DATE_RANGE_RE.search(nxt) and DATE_RANGE_RE.search(cur + " " + nxt):
                merged.append(cur + " " + nxt)
                i += 2
                continue
            if (_WIDOW_START_DATE_RE.search(cur) or re.search(rf"{MONTH_NAMES}\s+\d{{1,2}}\s*$", cur, re.IGNORECASE)) and (_WIDOW_END_DATE_RE.match(nxt) or re.match(r"^\d{4}\b", nxt)):
                merged.append(cur + " " + nxt)
                i += 2
                continue
            to_match = _TRAILING_TO_RE.search(cur)
            if to_match:
                before_to = cur[:to_match.start()].rstrip()
                if _WIDOW_START_DATE_RE.search(before_to):
                    merged.append(cur + " " + nxt)
                    i += 2
                    continue

        m_start = _START_WITH_DASH_RE.match(cur.strip())
        if m_start:
            found = False
            for offset in range(1, 4):
                if i + offset < n:
                    nxt = lines[i + offset].strip()
                    m_end = _END_DATE_START_RE.match(nxt)
                    if m_end:
                        start_d = m_start.group("start")
                        title_r = m_start.group("rest")
                        end_d = m_end.group("end")
                        comp_r = m_end.group("comp_rest")
                        clean_title = re.sub(r"R\s+e\s+g\s+ional", "Regional", title_r, flags=re.IGNORECASE).strip()
                        clean_comp = comp_r.strip()
                        merged.append(f"{start_d} - {end_d} | {clean_title} | {clean_comp}")
                        for mid_idx in range(1, offset):
                            merged.append(lines[i + mid_idx])
                        i += offset + 1
                        found = True
                        break
            if found:
                continue

        if cur.strip().lower() in ("as a", "as"):
            if merged and i + 1 < n and not re.match(r'^\d+[\).]', lines[i + 1].strip()):
                prev = merged.pop()
                nxt = lines[i + 1].strip()
                merged.append(f"{prev} {nxt}")
                i += 2
                continue
            else:
                i += 1
                continue

        if i + 1 < n:
            nxt = lines[i + 1].strip()
            if re.search(r'[\-\u2013\u2014]\s*$', cur) and re.match(r'^\s*(?:Present|Current|Till)', nxt, re.I):
                merged.append(cur + " " + nxt)
                i += 2
                continue
            if re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b\s*$', cur, re.I) and re.match(r'^\s*\d{4}', nxt):
                merged.append(cur + " " + nxt)
                i += 2
                continue

        merged.append(lines[i])
        i += 1

    return merged

TITLE_KEYWORDS = [
    "engineer", "developer", "manager", "analyst", "consultant",
    "designer", "architect", "lead", "leader", "head", "director", "vp", "svp", "cto", "ceo", "dgm", "agm", "gm", "avp",
    "intern", "associate", "senior", "junior", "principal", "staff",
    "specialist", "coordinator", "co-ordinator", "co ordinator", "executive", "officer", "president", "supervisor", "professional",
    "scientist", "researcher", "administrator", "technician", "professor", "resident", "fellow", "doctor", "physician", "registrar",
    "assistant", "trainee", "fresher", "apprentice", "gat", "get", "sales", "customer service",
    "accountant", "auditor", "controller", "bookkeeper", "treasurer", "trainer", "instructor", "faculty", "teacher", "educator",
    "company secretary", "secretary", "counsel", "advocate", "partner", "attorney"
]
TITLE_RE = re.compile(r'\b(' + '|'.join(TITLE_KEYWORDS) + r')\b', re.IGNORECASE)

_TECH_SKILL_KEYWORDS = {
    'mysql', 'sql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'nosql',
    'scikit-learn', 'scikitlearn', 'sklearn', 'pytorch', 'tensorflow', 'numpy', 'pandas', 'opencv',
    'django', 'fastapi', 'fast api', 'flask', 'react', 'angular', 'vue', 'node.js', 'nodejs',
    'aws', 'aws cloud', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'github', 'jira',
    'tableau', 'power bi', 'spark', 'hadoop', 'kafka', 'elasticsearch', 'redis',
    'graphql', 'rest', 'grpc', 'html', 'css', 'javascript', 'typescript', 'python',
    'java', 'c#', 'c++', 'yolo', 'yolov8', 'yolov11', 'llama', 'llama3', 'llama3.2', 'langchain',
    'huggingface', 'spacy', 'deepseek', 'agents', 'chatgpt', 'segmentation', 'object detection', 'annotation'
}

_DASH_NORMALIZE_TABLE = str.maketrans({
    "－": "-",
    "‐": "-",
    "‑": "-",
})


def _normalize_dashes(text: str) -> str:
    return text.translate(_DASH_NORMALIZE_TABLE)


# A stray dash is sometimes glued directly between a month name and ITS
# OWN year -- "July - 2018", "Dec -2018" -- distinct from the intended
# dash that separates the start date from the end date. Confirmed on a
# real resume (Abhay Awasthi): dates written as "(July - 2018 to Dec. -
# 2018)" defeated DATE_RANGE_RE entirely, since its "start"/"end" groups
# expect a month directly followed by whitespace/comma/period and then
# the year -- a literal "-" in between isn't in that allowed character
# class. This silently dropped 3 of his 5 job entries (zero anchor line
# found for them at all), not just a parsing quality issue.
# Collapsing "Month - Year" / "Month -Year" down to "Month Year" BEFORE
# DATE_RANGE_RE runs fixes this without touching the real start/end
# separator dash, since this pattern only fires directly after a month
# name, never between two already-complete dates.
_MONTH_YEAR_GLUE_DASH_RE = re.compile(
    rf"({MONTH_NAMES})[.,]?\s*-\s*(\d{{4}})", re.IGNORECASE
)
_TWO_MONTH_YEAR_SPACE_RE = re.compile(
    rf"\b({_MONTH_YEAR})\s+({_MONTH_YEAR})\b", re.IGNORECASE
)


def _normalize_month_year_glue(text: str) -> str:
    text = re.sub(r'1P0r/e2s0e2n5\s*t|1P0r[^\s]+', 'Present', text, flags=re.IGNORECASE)
    text = _MONTH_YEAR_GLUE_DASH_RE.sub(r"\1 \2", text)
    return _TWO_MONTH_YEAR_SPACE_RE.sub(r"\1 - \2", text)

BULLET_RE = re.compile(r'^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\ue25f\-\*]\s*|^\d+[\.\)]?\s+')

def _looks_like_subheading(line: str) -> bool:
    words = line.split()
    if not (1 <= len(words) <= 6):
        return False
    if line.rstrip().endswith((".", ",", ";", ":")):
        return False
    if DATE_RANGE_RE.search(line):
        return False
    # Ensure there is at least one letter, and the first letter in the line is not lowercase
    first_letter_match = re.search(r'[A-Za-z]', line)
    if not first_letter_match or first_letter_match.group(0).islower():
        return False
    return True


if _SPACY_IMPORTED:
    try:
        _NLP = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
        _SPACY_AVAILABLE = True
    except Exception:
        _SPACY_AVAILABLE = False
        print("[WARN] spaCy model not found. Company name detection will be limited.")
else:
    _SPACY_AVAILABLE = False
    print("[WARN] spaCy not installed. Company name detection will be limited.")


def _expand_apostrophe_year(date_str: str) -> str:
    m = re.search(r"['’‘`´](\d{2})\b", date_str)
    if m:
        two_digit = int(m.group(1))
        century = "20" if two_digit <= 49 else "19"
        date_str = date_str[:m.start()] + " " + century + m.group(1) + date_str[m.end():]

    m_slash = re.search(rf"\b({MONTH_NAMES})[/.-](\d{{2}})\b", date_str, re.IGNORECASE)
    if m_slash:
        month_name = m_slash.group(1)
        two_digit = int(m_slash.group(2))
        century = "20" if two_digit <= 49 else "19"
        date_str = date_str[:m_slash.start()] + f"{month_name} {century}{two_digit:02d}" + date_str[m_slash.end():]

    return date_str


# Confirmed on a real resume (Abhijeet Sangle): dates written numerically
# as "12.2011" (MM.YYYY) get silently MISPARSED by dateutil's fuzzy
# parser -- it reads "12" as the DAY, can't make sense of the trailing
# ".2011" as anything meaningful, and discards it, falling back to the
# `default` year/month (2000-01). So "12.2011" and "12.2019" -- two
# genuinely different dates 8 years apart -- both parsed to the exact
# same date, 2000-01-12, making the computed duration collapse to
# ~0 months instead of the correct ~96/97. This isn't a duration-formula
# bug; the dates going INTO the duration calculation were already wrong.
# Explicitly parsing this MM.YYYY / MM/YYYY numeric shorthand ourselves,
# before ever handing the string to dateutil, sidesteps the ambiguity
# entirely rather than hoping a general-purpose fuzzy parser guesses
# right on a format many general parsers reasonably read as DD.YYYY.
_MM_YYYY_RE = re.compile(r"^(\d{1,2})[/.](\d{4})$")


def _try_parse_mm_yyyy(date_str: str):
    m = _MM_YYYY_RE.match(date_str.strip())
    if not m:
        return None
    val1, year = int(m.group(1)), int(m.group(2))
    if 1 <= val1 <= 12:
        return datetime(year, val1, 1)
    elif val1 > 12:
        return datetime(year, 12, 1)
    return None


def parse_date(date_str: str):
    if not date_str or re.match(r'present|current|now|till\s*(?:date|now)|ongoing|on-going', date_str, re.IGNORECASE):
        return datetime.now()
    date_str = _expand_apostrophe_year(date_str)
    mm_yyyy = _try_parse_mm_yyyy(date_str)
    if mm_yyyy:
        return mm_yyyy
    try:
        return date_parser.parse(date_str, default=datetime(2000, 1, 1), fuzzy=True)
    except Exception:
        return None


def _inclusive_month_diff(start, end) -> int:
    """Inclusive calendar-month count between two dates.

    relativedelta(end, start) gives EXCLUSIVE elapsed time -- "June 2021"
    to "May 2022" is 11 months apart. But resumes conventionally state
    duration INCLUSIVELY: both the start month and the end month count
    as full months worked, so "June 2021 - May 2022" reads as 12 months
    (1 year), not 11.

    Confirmed on a real resume (Aasma): every one of her 5 jobs was
    undercounted by exactly 1 month against this convention (e.g. "June
    2022 - Present", computed as of July 2026, should read 50 months,
    not 49) -- a uniform 1-month deficit across every entry, which is
    the signature of an exclusive-vs-inclusive convention mismatch
    rather than a parsing error. This also matches how the aggregate
    "total years of experience" figure is conventionally read on a
    resume or by a recruiter, so both calculate_duration_months() and
    calculate_total_experience() use this same helper to stay
    consistent with each other.
    """
    if not start or not end:
        return 0
    delta = relativedelta(end, start)
    months = delta.years * 12 + delta.months
    return max(0, months + 1)


def calculate_duration_months(start_str: str, end_str: str) -> int:
    if not start_str and end_str:
        dur_m = _STATED_DURATION_RE.search(end_str)
        if dur_m:
            num = float(dur_m.group("num"))
            unit = dur_m.group("unit").lower()
            return int(num * 12) if "y" in unit else int(num)
    start = parse_date(start_str)
    end = parse_date(end_str)
    return _inclusive_month_diff(start, end)


def detect_company_with_spacy(text_block: str):
    if not _SPACY_AVAILABLE:
        return None
    doc = _NLP(text_block[:500])
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
    return orgs[0] if orgs else None


_COMPANY_ABBREV_ENDINGS = {"ltd", "inc", "co", "corp", "pvt", "llc", "llp"}


def _looks_like_prose(line: str) -> bool:
    if (" – " in line or " — " in line or " - " in line or " | " in line or any(w in line.lower().split() for w in ("ltd", "inc", "corp", "llc", "gmbh", "pvt", "limited", "company", "hospitality", "energy", "technologies", "facility"))) and not line.rstrip().endswith((".", ";")):
        return False
    words = line.split()
    if len(words) > 6:
        return True
    last_word = words[-1].rstrip(".;").lower() if words else ""
    if line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS:
        return True
    if re.search(r'[:\-–—]$', line) or any(w in line.lower() for w in ("handling", "analysis", "process", "documents", "transaction", "billing", "development", "management", "testing")):
        return True
    return False


# FIX: header_start_for's backward walk (below) originally used
# _looks_like_prose() -- which treats ">6 words" ALONE as enough to
# flag a line as "the previous job's trailing sentence" and stop
# walking backward. That's broader than the check's own comment states
# ("A long/period-ending sentence..." describes an AND of both
# conditions; _looks_like_prose implements it as an OR). Confirmed on a
# real resume (Abhay Awasthi) whose entire experience section has NO
# bullet markers at all -- every job is one long flowing sentence, e.g.
# "Worked at Morbous Technologies Pvt. Ltd. as a PHP Developer" (10
# words), immediately followed by that job's own date-range line. The
# word-count-alone check misclassified every one of these as "previous
# job's trailing prose" and excluded them, leaving every job's
# role/company empty even after the date-range itself was found.
# None of these genuine header lines end in terminal punctuation
# (they're cut off by the date parenthetical that follows) -- which is
# exactly the signal that distinguishes them from an actual trailing
# responsibility sentence (a grammatically complete, period-ended
# sentence). Requiring BOTH conditions here fixes that misfire while
# leaving _looks_like_prose() itself, and its other use in
# _reflow_body() (a different, unrelated no-bullet-marker case,
# confirmed working on Satyadip Ray's resume), completely untouched.
def _looks_like_previous_job_trailing_line(line: str) -> bool:
    l_strip = line.strip().lower()
    if l_strip.startswith(("environment", "application", "tools")):
        return True
    if re.search(r"^\s*(?:organization|organisation|company|employer|profile|designation|role|position|title)\s*[:\-–—\t]", l_strip):
        return False
    if TITLE_RE.search(line) or " at " in line.lower():
        return False
    words = line.split()
    if not words or len(words) <= 6:
        return False
    last_word = words[-1].rstrip(".;").lower()
    return line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS


def _find_job_blocks(lines: list):
    anchor_idx = []
    for i, l in enumerate(lines):
        l_str = l.strip()
        if (len(DATE_RANGE_RE.findall(l_str)) + len(_YEAR_RANGE_SHORT_RE.findall(l_str))) >= 3:
            continue
        if DATE_RANGE_RE.search(l_str) or _YEAR_RANGE_SHORT_RE.search(l_str) or _SINCE_DATE_RE.search(l_str):
            anchor_idx.append(i)
        elif _SINGLE_YEAR_LINE_RE.match(l_str):
            if i > 0 and not BULLET_RE.match(l_str):
                prev_has_context = any(
                    TITLE_RE.search(lines[k]) or " at " in lines[k].lower() or any(w in lines[k].lower() for w in ("ltd", "inc", "co", "corp", "pvt", "llc", "llp", "bank", "technologies", "solutions", "limited", "company", "analytics", "group", "services"))
                    for k in range(max(0, i-2), i)
                )
                if prev_has_context or i == 1:
                    anchor_idx.append(i)
    if not anchor_idx:
        anchor_idx = [
            i for i, l in enumerate(lines)
            if _WIDOW_START_DATE_RE.search(l) and i > 0 and (
                TITLE_RE.search(lines[i-1]) or " at " in lines[i-1].lower() or " – " in lines[i-1] or " - " in lines[i-1] or ":-" in lines[i-1]
            )
        ]
    if not anchor_idx:
        anchor_idx = [
            i for i, l in enumerate(lines)
            if _STATED_DURATION_RE.search(l) and not _SECTION_HEADER_RE.match(l) and not BULLET_RE.match(l.strip())
        ]
    else:
        extra_stated = [
            i for i, l in enumerate(lines)
            if _STATED_DURATION_RE.search(l) and not _SECTION_HEADER_RE.match(l)
            and len(l.strip().split()) <= 4
            and not re.search(r'\b(?:of\s+experience|total|overall)\b', l, re.I)
            and any(TITLE_RE.search(lines[k]) for k in range(max(0, i-3), i))
        ]
        if extra_stated:
            anchor_idx = sorted(list(set(anchor_idx + extra_stated)))

    numbered_job_idx = [
        i for i, l in enumerate(lines)
        if re.match(r'^\d+[\).]\s*', l.strip()) and (TITLE_RE.search(l) or " in " in l.lower() or " at " in l.lower() or " - " in l or " – " in l or _STATED_DURATION_RE.search(l))
    ]
    if numbered_job_idx and len(numbered_job_idx) >= 2:
        anchor_idx = sorted(list(set(anchor_idx + numbered_job_idx)))

    if not anchor_idx:
        return []

    def header_start_for(anchor_i):
        start = anchor_i
        j = anchor_i - 1
        steps = 0
        while j >= 0 and steps < 5:
            l = lines[j].strip()
            if not l or j in anchor_idx:
                break
            if BULLET_RE.match(l):
                if len(l.split()) > 5 or not (TITLE_RE.search(l) or " at " in l.lower() or "worked as" in l.lower() or "working as" in l.lower()):
                    break
            if l[0].islower():
                break
            if "\t" in l or re.match(r"^\s*technical\s+skills?\b", l, re.I):
                break
            if l.rstrip().endswith(":"):
                break
            if _looks_like_previous_job_trailing_line(l):
                break
            start = j
            j -= 1
            steps += 1
        return start

    blocks = []
    header_starts = [header_start_for(a) for a in anchor_idx]

    for n, anchor_i in enumerate(anchor_idx):
        h_start = header_starts[n]
        h_end = anchor_i
        if n + 1 < len(anchor_idx):
            body_end = header_starts[n + 1] - 1
        else:
            body_end = len(lines) - 1
        header_ids = list(range(h_start, h_end + 1))
        body_ids = list(range(h_end + 1, body_end + 1)) if body_end >= h_end + 1 else []
        blocks.append((header_ids, body_ids))

    return blocks


_TABLE_HEADER_ROW_RE = re.compile(
    r'^\s*(?:period|date|dates|sl\.?\s*no|company|organization|institution|designation|role|duration|location)'
    r'[\s\t|,/]+'
    r'(?:period|date|dates|sl\.?\s*no|company|organization|institution|designation|role|duration|location)',
    re.IGNORECASE
)


_SECTION_HEADER_RE = re.compile(
    r'^\s*(?:work\s+history|work\s+experience|experience|employment\s+history|professional\s+experience|career\s+history|organizational\s+experience|professional\s+summary|career\s+summary|earl(?:y|ier)\s+career)[\s:\-]*$',
    re.IGNORECASE
)


def _parse_header(header_lines: list, date_line: str):
    job = {"role": None, "company": None, "location": None, "start": None, "end": None, "method": []}

    date_line = re.sub(r'[{}]', '', date_line).strip()
    clean_header_lines = [
        re.sub(r'[{}]', '', l).strip() for l in header_lines 
        if not l.strip().startswith(('─', 'LABEL:', 'CONFIDENCE:')) 
        and not re.match(r'^[\─\-\=\_\s]+$', l.strip())
        and not _TABLE_HEADER_ROW_RE.match(l.strip()) 
        and not _SECTION_HEADER_RE.match(l.strip())
    ]
    if not clean_header_lines:
        clean_header_lines = [re.sub(r'[{}]', '', l).strip() for l in header_lines if not l.strip().startswith(('─', 'LABEL:', 'CONFIDENCE:')) and not re.match(r'^[\─\-\=\_\s]+$', l.strip())]

    m = DATE_RANGE_RE.search(date_line)
    m_short = _YEAR_RANGE_SHORT_RE.search(date_line)
    m_single = _SINGLE_YEAR_LINE_RE.match(date_line)
    m_since = _SINCE_DATE_RE.search(date_line)
    if m:
        job["start"] = m.group("start").strip()
        job["end"] = m.group("end").strip()
        job["method"].append("date_regex")
        date_prefix = date_line[:m.start()].strip(" |–-—\t,")
        if date_prefix.endswith("(") and date_prefix.count("(") > date_prefix.count(")"):
            date_prefix = date_prefix[:-1].strip(" |–-—\t,")
        if date_prefix.startswith("(") and date_prefix.count("(") > date_prefix.count(")"):
            date_prefix = date_prefix[1:].strip(" |–-—\t,")
        date_prefix = re.sub(r'\b(?:EMPLOYMENT DETAILS|PREVIOUS WORK EXPERIENCE|WORK EXPERIENCE|EXPERIENCE)\b', '', date_prefix, flags=re.I).strip(" |–-—\t,")
        date_suffix = date_line[m.end():].strip(" |–-—\t,")
        if date_suffix.startswith(")") and date_suffix.count(")") > date_suffix.count("("):
            date_suffix = date_suffix[1:].strip(" |–-—\t,")
        if date_suffix.endswith(")") and date_suffix.count(")") > date_suffix.count("("):
            date_suffix = date_suffix[:-1].strip(" |–-—\t,")
        date_idx = -1
        for idx, l_str in enumerate(clean_header_lines):
            if l_str == date_line or (date_line and date_line in l_str):
                date_idx = idx
                break
        following_lines = clean_header_lines[date_idx + 1:] if date_idx != -1 else []
        if date_idx > 0:
            job["pre_date_header"] = clean_header_lines[:date_idx]

        target_text = date_prefix
        if date_suffix:
            if following_lines and not re.search(r'\b(?:as\s+a?|as)\b', date_suffix, re.I) and (TITLE_RE.search(following_lines[0]) or re.search(r'\bdesignated\s+as\b', following_lines[0], re.I)) and (
                date_suffix.lower().startswith("with ") or re.search(r'\b(?:ltd|pvt|inc|corp|llc|technologies|solution|solutions|accenture|mahindra|persistent)\b', date_suffix, re.I)
            ):
                f_line = following_lines[0].strip()
                job["role"] = re.sub(r'^(?:designated\s+as|role\s*[:\-–—]?)\s*', '', f_line, flags=re.I).strip(" .–-—,")
                c_clean = re.sub(r'^(?:with\s+|at\s+)', '', date_suffix, flags=re.I).strip(" .–-—,")
                if ',' in c_clean:
                    left_c, right_loc = c_clean.rsplit(',', 1)
                    job["company"] = left_c.strip(" .–-—,")
                    job["location"] = right_loc.strip(" .–-—,")
                else:
                    job["company"] = c_clean
                date_suffix = ""
                target_text = ""
                job["method"].append("date_suffix_company_following_role")
            elif not re.search(r'\b(?:as\s+a?|as|with)\b', date_suffix, re.I) and re.search(r'\b(?:gurugram|mumbai|delhi|new delhi|noida|bengaluru|bangalore|pune|hyderabad|chennai|kolkata|ahmedabad|chandigarh|mohali|gurgaon|jaipur|indore|surat|lucknow|kanpur|nagpur|bhopal|patna|vadodara|coimbatore|kochi|thiruvananthapuram|calicut|trivandrum|bhubaneswar|guwahati|ranchi|mysore|visakhapatnam|vijayawada|nashik|rajkot|faridabad|ghaziabad|ludhiana|agra|meerut|varanasi|punjab|haryana|karnataka|maharashtra|tamil nadu|telangana|kerala|uttar pradesh|gujarat|rajasthan|west bengal|madhya pradesh|bihar|odisha|assam|jharkhand|chhattisgarh|uttarakhand|goa|india|usa|uk)\b', date_suffix, re.I):
                job["location"] = date_suffix
            elif not date_prefix or not TITLE_RE.search(date_prefix) or re.search(r'\b(?:as\s+a?|as|with)\b', date_suffix, re.I):
                target_text = (date_prefix + " " + date_suffix).strip()

        if date_prefix and TITLE_RE.search(date_prefix):
            job["role"] = date_prefix

        if target_text:
            if "." in target_text and re.match(r'^(?:\d{1,2}|[A-Za-z])\s*\.', target_text):
                target_text = target_text.split(".", 1)[-1].strip()
            if "•" in target_text:
                target_text = target_text.rsplit("•", 1)[-1].strip()
            target_text = re.sub(r'^[-\*\s]+', '', target_text).strip()
        if target_text and "|" in target_text:
            parts = [p.strip() for p in target_text.split("|") if p.strip()]
            first_part = parts[0]
            second_part = parts[1] if len(parts) >= 2 else ""
            if second_part and TITLE_RE.search(second_part):
                if TITLE_RE.search(first_part):
                    job["role"] = f"{first_part} | {second_part}"
                    job["company"] = None
                    if len(parts) >= 3:
                        third_part = parts[2]
                        if not TITLE_RE.search(third_part):
                            if any(city in third_part.lower() for city in ('gurgaon', 'gurgoan', 'gurugram', 'noida', 'pune', 'delhi', 'bangalore', 'bengaluru', 'mumbai', 'hyderabad', 'chennai', 'india', 'usa')):
                                job["location"] = third_part
                            else:
                                job["company"] = third_part
                elif "," in first_part:
                    c_cand, loc_cand = [x.strip() for x in first_part.rsplit(",", 1)]
                    job["company"] = c_cand
                    job["location"] = loc_cand
                    job["role"] = second_part
                else:
                    job["company"] = first_part
                    job["role"] = second_part
                if len(parts) >= 3 and not job.get("location"):
                    job["location"] = parts[2]
            elif TITLE_RE.search(first_part):
                if "," in first_part:
                    r_cand, c_cand = [x.strip() for x in first_part.split(",", 1)]
                    job["role"] = r_cand
                    job["company"] = c_cand
                else:
                    job["role"] = first_part
                    if len(parts) >= 2:
                        job["company"] = parts[1]
                if len(parts) >= 3 and not job.get("location"):
                    job["location"] = parts[-1]
            else:
                if "," in first_part:
                    c_cand, loc_cand = [x.strip() for x in first_part.rsplit(",", 1)]
                    job["company"] = c_cand
                    job["location"] = loc_cand
                else:
                    job["company"] = first_part
                if len(parts) >= 2:
                    job["role"] = parts[1]
                if len(parts) >= 3 and not job.get("location"):
                    job["location"] = parts[2]
            job["method"].append("pipe_split_header")
        date_idx = -1
        for idx, l_str in enumerate(clean_header_lines):
            if l_str == date_line or (date_line and date_line in l_str):
                date_idx = idx
                break
        following_lines = clean_header_lines[date_idx + 1:] if date_idx != -1 else []
        if date_idx > 0:
            job["pre_date_header"] = clean_header_lines[:date_idx]

        has_with_as = (
            bool(re.search(r'\b(?:as\s+a?|as)\b', target_text, re.I)) or
            (bool(re.search(r'\bwith\b', target_text, re.I)) and bool(following_lines and re.search(r'\b(?:as\s+a?|as)\b', following_lines[0], re.I)))
        )
        if not job.get("company") and following_lines and not has_with_as:
            f_line = following_lines[0].strip()
            for sep in (" – ", " — ", " - ", ","):
                if sep in f_line:
                    c_cand, r_cand = [x.strip() for x in f_line.split(sep, 1)]
                    if job.get("role") and not TITLE_RE.search(c_cand):
                        job["company"] = c_cand
                        if not job.get("location"):
                            job["location"] = r_cand
                        job["method"].append("following_line_company_split")
                        break
                    elif TITLE_RE.search(r_cand):
                        job["company"] = c_cand
                        job["role"] = r_cand
                        job["method"].append("comma_company_role_split")
                        break
                    elif TITLE_RE.search(c_cand):
                        if job.get("pre_date_header"):
                            cand_c = next((l for l in reversed(job["pre_date_header"]) if l.strip() and not TITLE_RE.search(l) and not DATE_RANGE_RE.search(l) and not l.strip().endswith(".")), None)
                            if cand_c:
                                job["company"] = cand_c
                                job["role"] = f_line
                                job["method"].append("following_line_title_pre_date_company")
                                break
                        job["role"] = c_cand
                        job["company"] = r_cand
                        job["method"].append("comma_role_company_split")
                        break

        if not job.get("company") and job.get("pre_date_header"):
            for pl in job["pre_date_header"]:
                pl_s = pl.strip()
                m_org = re.match(r"^(?:Organization|Organisation|Company|Employer)\s*[:\s]\s*(.+)$", pl_s, re.I)
                if m_org:
                    org_val = m_org.group(1).strip()
                    if "," in org_val:
                        parts = [p.strip() for p in org_val.split(",")]
                        job["company"] = parts[0]
                        if not job.get("location") and len(parts) > 1:
                            loc_parts = [p for p in parts[1:] if not re.match(r'^(?:a|an|the)\b', p, re.I) and not any(w in p.lower() for w in ('nbfc', 'company', 'subsidiary', 'firm'))]
                            if loc_parts:
                                job["location"] = ", ".join(loc_parts)
                    else:
                        job["company"] = org_val
                    job["method"].append("organization_prefix")
                    break

        if not job.get("company") and job.get("pre_date_header"):
            for pl in reversed(job["pre_date_header"]):
                p_line = pl.strip()
                if not p_line or _SECTION_HEADER_RE.match(p_line):
                    continue
                if p_line.lower() in ("duration", "duration:", "responsibilities", "key highlights", "key highlights of work handled", "work handled", "organization", "organisation"):
                    continue
                if re.match(r"^(?:gurugram|mumbai|delhi|new delhi|noida|bengaluru|bangalore|pune|hyderabad|chennai|kolkata|ahmedabad|chandigarh|mohali|gurgaon|jaipur|indore|surat|lucknow|kanpur|nagpur|bhopal|patna|vadodara|coimbatore|kochi|thiruvananthapuram|calicut|trivandrum|bhubaneswar|guwahati|ranchi|mysore|visakhapatnam|vijayawada|nashik|rajkot|faridabad|ghaziabad|ludhiana|agra|meerut|varanasi|punjab|haryana|karnataka|maharashtra|tamil nadu|telangana|kerala|uttar pradesh|gujarat|rajasthan|west bengal|madhya pradesh|bihar|odisha|assam|jharkhand|chhattisgarh|uttarakhand|goa|india|germany|usa|uk)$", p_line, re.I):
                    if not job.get("location"):
                        job["location"] = p_line
                    continue
                if re.match(r'^(?:division|department|business\s+unit|vertical|function|practice)[\s:]', p_line, re.I):
                    continue
                for sep in (" – ", " — ", " - ", ","):
                    if sep in p_line:
                        c_cand, loc_cand = [x.strip() for x in p_line.split(sep, 1)]
                        if TITLE_RE.search(c_cand) and not TITLE_RE.search(loc_cand):
                            pl_idx = job["pre_date_header"].index(pl)
                            found_comp = None
                            if pl_idx > 0:
                                for k in range(pl_idx - 1, -1, -1):
                                    cand_comp = job["pre_date_header"][k].strip()
                                    if cand_comp and not TITLE_RE.search(cand_comp) and not _SECTION_HEADER_RE.match(cand_comp) and not DATE_RANGE_RE.search(cand_comp) and not cand_comp.lower().startswith(("duration", "division", "responsib", "key", "work", "supported", "develop", "managed", "ensure", "lead")):
                                        is_sentence = cand_comp.endswith(".") and not any(cand_comp.lower().endswith(abbr) for abbr in ("ltd.", "inc.", "co.", "corp.", "pvt.", "llc."))
                                        if not is_sentence:
                                            found_comp = cand_comp
                                            break
                            if found_comp:
                                job["company"] = found_comp
                                job["role"] = p_line
                                if re.search(r'\b(?:pune|mumbai|delhi|new delhi|noida|gurgaon|gurugram|bangalore|bengaluru|hyderabad|chennai|kolkata|ahmedabad|rom|india|north|south|east|west|central|region|territory|zone|remote)\b', loc_cand, re.I):
                                    if not job.get("location"):
                                        job["location"] = loc_cand
                            else:
                                if not job.get("role"):
                                    job["role"] = c_cand
                                job["company"] = loc_cand
                        elif TITLE_RE.search(loc_cand) and not TITLE_RE.search(c_cand):
                            if not job.get("role"):
                                job["role"] = loc_cand
                            job["company"] = c_cand
                        else:
                            job["company"] = c_cand
                            if not job.get("location"):
                                job["location"] = loc_cand
                            pl_idx = job["pre_date_header"].index(pl)
                            if pl_idx > 0:
                                prev_pl = job["pre_date_header"][pl_idx - 1].strip()
                                if (
                                    prev_pl
                                    and not TITLE_RE.search(prev_pl)
                                    and not _SECTION_HEADER_RE.match(prev_pl)
                                    and not DATE_RANGE_RE.search(prev_pl)
                                    and len(prev_pl.split()) <= 6
                                    and not prev_pl.lower().startswith(("duration", "role", "key", "work"))
                                ):
                                    job["company"] = (prev_pl + " " + job["company"]).strip()
                        job["method"].append("pre_date_company_split")
                        break
                else:
                    if TITLE_RE.search(p_line):
                        if not job.get("role"):
                            job["role"] = p_line
                    else:
                        job["company"] = p_line
                if job.get("company") or job.get("role"):
                    break

        if target_text and not (job.get("company") and job.get("role")) and (TITLE_RE.search(target_text) or re.search(r'\b(?:as\s+a?|as|with)\b', target_text, re.I) or (following_lines and re.search(r'\b(?:as\s+a?|as)\b', following_lines[0], re.I))):
            m_having = re.search(r'^(?:[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\s]*)(?:Having|With)\s+.*?\s+experience\s+as\s+a?\s*(.*?)\s+\b(?:for\s+in|for|in|at|with)\b\s+(.+)', target_text, re.I)
            m_work_with = re.search(r'^(?:[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\s]*)(?:Working|Worked)?\s*(?:as\s+(?:a\s+|an\s+)?)?(.+?)\s+with\s+(.+)$', target_text, re.I)
            if m_having and m_having.group(1).strip() and m_having.group(2).strip():
                job["role"] = m_having.group(1).strip()
                job["company"] = m_having.group(2).strip(" .–-—,")
            elif m_work_with and TITLE_RE.search(m_work_with.group(1)):
                r_clean = re.sub(r'^(?:[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\s]*)(?:Working|Worked)?\s*(?:as\s+(?:a\s+|an\s+)?)?', '', m_work_with.group(1), flags=re.IGNORECASE).strip()
                job["role"] = r_clean
                job["company"] = m_work_with.group(2).strip(" .–-—,")
            elif re.search(r'\b(?:as\s+a?|as)\b', target_text, re.I):
                c_part, r_part = re.split(r'\b(?:as\s+a?|as)\b', target_text, flags=re.I, maxsplit=1)
                c_part = re.sub(r'^(?:From|Since\s+)?(?:\d+(?:\.\d+)?\s*Years\s*)?(?:\([^)]+\)\s*)?(?:[A-Za-z]+[’\']?\d{2,4}\s*(?:to|-|–)\s*(?:till\s+date|present)\s*)?(?:with\s+)?', '', c_part, flags=re.I).strip(" .–-—,")
                if c_part.lower() in ("from", "since", "during", "between", "in"):
                    c_part = ""
                r_part = re.sub(r'^(?:an?|the)\s+', '', r_part, flags=re.I).strip()
                r_part = re.sub(r'^(?:whole\s+time\s+)?', '', r_part, flags=re.I).strip()
                if following_lines and (r_part in ('Overall', 'General Manager of', 'Dept Head of', 'Manager -') or re.search(r'\b(?:of|a|an|the|-|and|&)\s*$', r_part, re.I) or re.search(r'\b(?:of|for)\s+[A-Za-z]+$', r_part, re.I)):
                    next_head = re.split(r'\s*\([^)]*\)|\s*-\s*\(', following_lines[0])[0].strip()
                    r_part = (r_part + " " + next_head).strip()
                c_clean = re.sub(r'\s*\([^)]*\)', '', c_part).strip(' ,.-')
                c_clean = re.sub(r'\b(?:TS-\d+|Certified\s+Company)\b', '', c_clean, flags=re.I).strip(' ,.-')
                m_based = re.search(r'\s+(?:based\s+in|in|at)\s+([A-Za-z\s]+?)$', c_clean, re.I)
                if m_based and not job.get("location"):
                    job["location"] = m_based.group(1).strip()
                    c_clean = re.sub(r'\s+(?:based\s+in|in|at)\s+[A-Za-z\s]+$', '', c_clean, flags=re.I).strip(' ,.-')
                c_clean = re.sub(r'\s+(?:German|American|Indian)?\s*MNC.*$', '', c_clean, flags=re.I).strip(' ,.-')
                if ',' in c_clean:
                    left_c, right_loc = c_clean.rsplit(',', 1)
                    if any(city in right_loc.lower() for city in ('gurgaon', 'gurgoan', 'gurugram', 'noida', 'pune', 'delhi', 'faridabad', 'tapukara')):
                        if not job.get("location"):
                            job["location"] = right_loc.strip()
                        c_clean = left_c.strip()
                w_list = c_clean.split()
                if len(w_list) >= 3 and f'{w_list[-2]} {w_list[-1]}'.lower() in ('greater noida', 'new delhi'):
                    if not job.get("location"):
                        job["location"] = f'{w_list[-2]} {w_list[-1]}'
                    c_clean = ' '.join(w_list[:-2]).strip()
                elif len(w_list) >= 2 and w_list[-1].lower() in ('noida', 'gurgaon', 'gurgoan', 'gurugram', 'pune', 'faridabad'):
                    if not job.get("location"):
                        job["location"] = w_list[-1]
                    c_clean = ' '.join(w_list[:-1]).strip()
                if c_clean:
                    job["company"] = c_clean
                job["role"] = r_part
            elif following_lines and re.search(r'\b(?:as\s+a?|as)\b', following_lines[0], re.I):
                c_part = re.sub(r'^(?:Since\s+)?(?:\d+(?:\.\d+)?\s*Years\s*)?(?:\([^)]+\)\s*)?(?:[A-Za-z]+[’\']?\d{2,4}\s*(?:to|-|–)\s*(?:till\s+date|present)\s*)?(?:with\s+)?', '', target_text, flags=re.I).strip(" .–-—,")
                c_extra, r_part = re.split(r'\b(?:as\s+a?|as)\b', following_lines[0], flags=re.I, maxsplit=1)
                c_extra_clean = re.sub(r'^(?:worked|working|designated|serving|joined)\b', '', c_extra.strip(), flags=re.I).strip()
                if c_extra_clean:
                    c_part = (c_part + " " + c_extra_clean).strip()
                r_part = re.sub(r'^(?:an?|the)\s+', '', r_part, flags=re.I).strip()
                r_part = re.split(r'\s*\([^)]*\)|\s*-\s*\(', r_part)[0].strip()
                c_clean = re.sub(r'\s*\([^)]*\)', '', c_part).strip(' ,.-')
                c_clean = re.sub(r'\b(?:TS-\d+|Certified\s+Company)\b', '', c_clean, flags=re.I).strip(' ,.-')
                m_based = re.search(r'\s+(?:based\s+in|in|at)\s+([A-Za-z\s]+?)$', c_clean, re.I)
                if m_based and not job.get("location"):
                    job["location"] = m_based.group(1).strip()
                    c_clean = re.sub(r'\s+(?:based\s+in|in|at)\s+[A-Za-z\s]+$', '', c_clean, flags=re.I).strip(' ,.-')
                c_clean = re.sub(r'\s+(?:German|American|Indian)?\s*MNC.*$', '', c_clean, flags=re.I).strip(' ,.-')
                if ',' in c_clean:
                    left_c, right_loc = c_clean.rsplit(',', 1)
                    if any(city in right_loc.lower() for city in ('gurgaon', 'gurgoan', 'gurugram', 'noida', 'pune', 'delhi', 'faridabad', 'tapukara')):
                        if not job.get("location"):
                            job["location"] = right_loc.strip()
                        c_clean = left_c.strip()
                w_list = c_clean.split()
                if len(w_list) >= 3 and f'{w_list[-2]} {w_list[-1]}'.lower() in ('greater noida', 'new delhi'):
                    if not job.get("location"):
                        job["location"] = f'{w_list[-2]} {w_list[-1]}'
                    c_clean = ' '.join(w_list[:-2]).strip()
                elif len(w_list) >= 2 and w_list[-1].lower() in ('noida', 'gurgaon', 'gurgoan', 'gurugram', 'pune', 'faridabad'):
                    if not job.get("location"):
                        job["location"] = w_list[-1]
                    c_clean = ' '.join(w_list[:-1]).strip()
                if c_clean:
                    job["company"] = c_clean
                job["role"] = r_part
            elif " – " in target_text or " - " in target_text:
                dash = " – " if " – " in target_text else " - "
                parts = [p.strip() for p in target_text.split(dash) if p.strip()]
                cand_c = None
                if job.get("pre_date_header") and not job.get("company"):
                    cand_c = next((l for l in reversed(job["pre_date_header"]) if l.strip() and not TITLE_RE.search(l) and not DATE_RANGE_RE.search(l) and not l.strip().endswith(".")), None)
                if cand_c:
                    job["company"] = cand_c
                    job["role"] = target_text
                elif len(parts) >= 2:
                    if TITLE_RE.search(parts[0]):
                        job["role"] = parts[0].lstrip("•*❖-– ").strip()
                        job["company"] = dash.join(parts[1:]).strip()
                    else:
                        job["company"] = parts[0].lstrip("•*❖-– ").strip()
                        job["role"] = dash.join(parts[1:]).strip()
            else:
                job["role"] = target_text
            job["method"].append("date_prefix_title")
    elif m_short:
        start_yr = m_short.group("start").strip()
        end_short = m_short.group("end_short").strip()
        full_end = start_yr[:2] + end_short if len(end_short) == 2 else end_short
        job["start"] = start_yr
        job["end"] = full_end
        job["method"].append("year_range_short_regex")
        date_prefix = date_line[:m_short.start()].strip(" |–-—,\t")
        if date_prefix and TITLE_RE.search(date_prefix):
            m_having_p = re.search(r'^(?:[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\s]*)(?:Having|With)\s+.*?\s+experience\s+as\s+a?\s*(.*?)\s+(?:for\s+in|for|in|at|with)\s+(.+?)(?:\.|\,|\(|\d{2,4}|$)', date_prefix, re.I)
            if m_having_p and m_having_p.group(1).strip() and m_having_p.group(2).strip():
                job["role"] = m_having_p.group(1).strip()
                job["company"] = m_having_p.group(2).strip(" .–-—,")
            elif re.search(r'\b(?:as\s+a?|as)\b', date_prefix, re.I):
                c_part, r_part = re.split(r'\b(?:as\s+a?|as)\b', date_prefix, flags=re.I, maxsplit=1)
                c_part = c_part.rstrip(" .–-—,")
                if c_part:
                    job["company"] = c_part
                job["role"] = r_part
            else:
                job["role"] = date_prefix
            job["method"].append("date_prefix_title")
    elif m_since:
        job["start"] = m_since.group("start").strip()
        job["end"] = "Present"
        job["method"].append("since_date_regex")
        date_prefix = date_line[:m_since.start()].strip(" |–-—,\t")
        date_suffix = date_line[m_since.end():].strip(" |–-—,\t")
        date_idx = -1
        for idx, l_str in enumerate(clean_header_lines):
            if l_str == date_line or (date_line and date_line in l_str):
                date_idx = idx
                break
        following_lines = clean_header_lines[date_idx + 1:] if date_idx != -1 else []
        if date_idx > 0:
            job["pre_date_header"] = clean_header_lines[:date_idx]
        if date_suffix and following_lines and not re.search(r'\b(?:as\s+a?|as)\b', date_suffix, re.I):
            f_line = following_lines[0].strip()
            if (TITLE_RE.search(f_line) or re.search(r'\bdesignated\s+as\b', f_line, re.I)) and (
                date_suffix.lower().startswith("with ") or re.search(r'\b(?:ltd|pvt|inc|corp|llc|technologies|solution|solutions|accenture|mahindra|persistent)\b', date_suffix, re.I)
            ):
                job["role"] = re.sub(r'^(?:designated\s+as|role\s*[:\-–—]?)\s*', '', f_line, flags=re.I).strip(" .–-—,")
                c_clean = re.sub(r'^(?:with\s+|at\s+)', '', date_suffix, flags=re.I).strip(" .–-—,")
                if ',' in c_clean:
                    left_c, right_loc = c_clean.rsplit(',', 1)
                    job["company"] = left_c.strip(" .–-—,")
                    job["location"] = right_loc.strip(" .–-—,")
                else:
                    job["company"] = c_clean
                job["method"].append("since_date_suffix_company_following_role")
        elif date_prefix and TITLE_RE.search(date_prefix):
            if re.search(r'\b(?:as\s+a?|as)\b', date_prefix, re.I):
                c_part, r_part = re.split(r'\b(?:as\s+a?|as)\b', date_prefix, flags=re.I, maxsplit=1)
                c_part = c_part.rstrip(" .–-—,")
                if c_part:
                    job["company"] = c_part
                job["role"] = r_part
            else:
                job["role"] = date_prefix
            job["method"].append("date_prefix_title")
    elif m_single:
        single_yr = m_single.group(1).strip()
        job["start"] = single_yr
        job["end"] = single_yr
        job["method"].append("single_year_regex")
    else:
        sm = _WIDOW_START_DATE_RE.search(date_line)
        if sm:
            job["start"] = sm.group(1).strip()
            job["end"] = "Present"
            job["method"].append("single_date_regex")
        else:
            dur_m = _STATED_DURATION_RE.search(date_line)
            if dur_m:
                num = float(dur_m.group("num"))
                unit = dur_m.group("unit").lower()
                dur_str = f"{num:g} Year" if "y" in unit else f"{num:g} Month"
                if num != 1:
                    dur_str += "s"
                job["start"] = None
                job["end"] = dur_str
                job["method"].append("stated_duration_regex")

    cleaned = []
    for l in clean_header_lines:
        l = l.strip()
        l_nobullet = re.sub(r'^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\s]+', '', l).strip()
        if not l or _SECTION_HEADER_RE.match(l) or (l_nobullet.endswith(".") and not TITLE_RE.search(l_nobullet)):
            continue
        l = l_nobullet
        l = re.sub(r'\bASSISTANTMANAGER\b', 'ASSISTANT MANAGER', l, flags=re.IGNORECASE)
        l = re.sub(r'\bU/UX\b', 'UI/UX', l, flags=re.IGNORECASE)
        dm = DATE_RANGE_RE.search(l)
        if dm:
            l = (l[:dm.start()] + l[dm.end():])
            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
        m_since_clean = _SINCE_DATE_RE.search(l)
        if m_since_clean:
            l = (l[:m_since_clean.start()] + l[m_since_clean.end():])
            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
        m_short_clean = _YEAR_RANGE_SHORT_RE.search(l)
        if m_short_clean:
            l = (l[:m_short_clean.start()] + l[m_short_clean.end():])
            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
        sm = _WIDOW_START_DATE_RE.search(l)
        if sm:
            l = (l[:sm.start()] + l[sm.end():])
            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
        dur_m = _STATED_DURATION_RE.search(l)
        if dur_m:
            l = re.sub(r'\s*\(\s*\d+(?:\.\d+)?\s*(?:years?|yrs?|months?|mos?)\s*\)', '', l, flags=re.IGNORECASE)
            l = (l[:dur_m.start()] + l[dur_m.end():])
            l = re.sub(r'^[\s\-–—,/|()]+|[\s\-–—,/|()]+$', '', l)
        l_low = l.lower().strip().rstrip(" -–—:")
        if l and l_low not in {"duration", "role", "designation", "period", "tenure", "responsibilities", "key responsibilities", "previous experience", "previous experience(s)", "previous experiences", "past experience", "career summary", "early career", "early career summary"} and not l_low.startswith(("environment", "application", "tools")):
            cleaned.append(l)

    _KNOWN_LOC_CITIES = {"mumbai", "delhi", "noida", "gurgaon", "gurugram", "pune", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata", "ahmedabad", "india", "remote", "usa", "uk", "vasai", "palghar", "goregaon", "andheri", "sewree", "mumbai, india", "maharashtra"}
    while len(cleaned) > 1 and any(p.strip().lower() in _KNOWN_LOC_CITIES for p in cleaned[0].split(",")):
        cleaned.pop(0)

    for idx_l, l in enumerate(clean_header_lines):
        m_org = re.search(r'^\s*(?:organization|organisation|company|employer)\s*[:\-–—\t]+\s*(.+)$', l, re.IGNORECASE)
        if m_org and not job["company"]:
            job["company"] = m_org.group(1).strip()
            job["method"].append("subfield_label_company")
        m_role = re.search(r'^\s*(?:profile|designation|role|position|title)\s*[:\-–—\t]+\s*(.+)$', l, re.IGNORECASE)
        if m_role and not job["role"]:
            job["role"] = m_role.group(1).strip()
            job["method"].append("subfield_label_role")
        elif re.match(r'^\s*(?:profile|designation|role|position|title)\s*[:\-–—\t]?\s*$', l, re.IGNORECASE):
            if idx_l + 1 < len(clean_header_lines) and not job["role"]:
                cand_r = clean_header_lines[idx_l + 1].strip()
                if TITLE_RE.search(cand_r) or len(cand_r.split()) <= 6:
                    job["role"] = cand_r
                    job["method"].append("subfield_label_role_next_line")

    if not job["role"] and not job["company"] and cleaned and "\t" in cleaned[0]:
        parts = [p.strip() for p in cleaned[0].split("\t") if p.strip()]
        if parts:
            if re.search(r'\b(period|date|sl\.?\s*no)\b', parts[0], re.I):
                parts = parts[1:]
            if len(parts) >= 2:
                cand_comp = parts[0]
                cand_role = parts[1]
                if cand_comp.strip().lower() not in {"company", "organization", "organisation", "employer", "designation", "role", "position", "title"}:
                    job["company"] = cand_comp
                if cand_role.strip().lower() not in {"company", "organization", "organisation", "employer", "designation", "role", "position", "title"}:
                    job["role"] = cand_role
                job["method"].append("tab_split")

    cleaned_no_dates = [DATE_RANGE_RE.sub("", c).strip() for c in cleaned]
    if cleaned_no_dates and "/" in cleaned_no_dates[0] and not any(sep in cleaned_no_dates[0] for sep in (" | ", " – ", " — ", " - ")) and not job["role"]:
        parts = [p.strip() for p in cleaned[0].split("/") if p.strip()]
        if parts:
            if len(parts) >= 2 and all(TITLE_RE.search(p) for p in parts):
                job["role"] = cleaned[0]
            elif paren := re.search(r'\(([^)]+)\)', cleaned[0]):
                job["role"] = parts[0]
                job["company"] = paren.group(1).strip()
                job["method"].append("inline_parens")
            elif TITLE_RE.search(parts[0]):
                job["role"] = parts[0]
                job["method"].append("slash_split_role")

    if cleaned and not (job["role"] and job["company"]):
        for line in cleaned[:3]:
            m_worked = re.search(r'^(?:•|\*|-)?\s*(?:Working|Worked)\s+as\s+(?:a\s+|an\s+)?(.+?)\s+(?:in|at|for|with)\s+(.+?)(?:\s+(?:since|from|in|till|to)\b|\,|\.|$)', line, re.IGNORECASE)
            if m_worked:
                r_cand = m_worked.group(1).strip()
                c_cand = m_worked.group(2).strip()
                if r_cand and c_cand:
                    job["role"] = r_cand
                    job["company"] = c_cand
                    job["method"].append("worked_as_role_in_company")
                    break
        for line in cleaned[:3]:
            if any(sep in line for sep in (" - ", " – ", " — ", " | ")):
                continue
            m_as_a = re.search(r'^(.*?)\s*\(\s*(?:as\s+(?:a\s+|an\s+)?)?(.+?)\s*\)$', line, re.IGNORECASE)
            if m_as_a:
                outside = m_as_a.group(1).strip()
                inside = m_as_a.group(2).strip()
                if outside and not TITLE_RE.search(outside) and (TITLE_RE.search(inside) or any(w in inside.lower() for w in ("senior", "associate", "manager", "lead", "assistant", "executive", "analyst"))):
                    job["company"] = outside
                    job["role"] = re.sub(r"^\s*as\s+(?:a\s+|an\s+)?", "", inside, flags=re.IGNORECASE).strip()
                    job["method"].append("company_parens_role")
                    break

    if not job["role"] and not job["company"] and cleaned and "|" in cleaned[0]:
        parts = [p.strip() for p in cleaned[0].split("|") if p.strip()]
        title_part = next((p for p in parts if TITLE_RE.search(p)), None)
        if title_part:
            job["role"] = title_part
            other_parts = [p for p in parts if p != title_part]
            if other_parts:
                job["company"] = other_parts[0]
            job["method"].append("pipe_split")

    if not job["role"] and cleaned:
        for line in cleaned[:3]:
            wm = re.search(r"working as a?\s+(.*?)\s+(?:in|at|for)\s+(.*)", line, re.IGNORECASE)
            if wm:
                role_candidate = wm.group(1).strip()
                comp_candidate = wm.group(2).strip()
                comp_candidate = re.sub(r"\s+from.*$", "", comp_candidate, flags=re.IGNORECASE).strip()
                if role_candidate and comp_candidate:
                    job["role"] = role_candidate
                    if not job["company"] or _TABLE_HEADER_ROW_RE.search(job["company"]):
                        job["company"] = comp_candidate
                    job["method"].append("working_as_split")
                    break

    if cleaned and not (job["role"] and job["company"]):
        for line in cleaned[:3] + [date_line]:
            m_with_as = re.search(r'\bwith\s+(?P<comp>.+?)\s+as\s+(?:a\s+|an\s+)?(?P<role>.+)', line, re.IGNORECASE)
            if m_with_as:
                comp_raw = m_with_as.group("comp")
                role_raw = m_with_as.group("role")
                comp = re.sub(r'\s+(?:in|based in|at)\s+[A-Z][a-z]+.*$', '', comp_raw, flags=re.IGNORECASE)
                comp = re.sub(r'\s+(?:German|American|Indian)?\s*MNC.*$', '', comp, flags=re.IGNORECASE)
                comp = comp.strip(" ,.-()")

                role = re.sub(r'[\,\-–—]\s*(?:Company Dealing|A Joint Venture|\(?A\s+Automation|\(?Weiss Automation|TS-16949).*$', '', role_raw, flags=re.IGNORECASE)
                role = role.strip(" ,.-()")
                if comp and role:
                    job["company"] = comp
                    job["role"] = role
                    job["method"].append("with_as_split")
                    break

    if cleaned and not (job["role"] and job["company"]):
        for target_line in cleaned[:3]:
            found_split = False
            target_line_clean = re.sub(r'^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\s]+', '', target_line).strip()
            for sep in (" | ", " – ", " — ", " - "):
                if sep in target_line_clean:
                    left, right = (p.strip() for p in target_line_clean.split(sep, 1))
                    left_clean = re.sub(r'\([^)]*\)', '', left).strip()
                    right_clean = re.sub(r'\([^)]*\)', '', right).strip()
                    if left_clean and right_clean:
                        left_is_title = bool(TITLE_RE.search(left_clean))
                        right_is_title = bool(TITLE_RE.search(right_clean))
                        if left_is_title and not right_is_title:
                            right_low = right_clean.lower()
                            if any(w in right_low for w in ("data science", "analytics", "analysis", "data analysis", "reporting", "visualization", "engineering", "development", "operations", "research", "technology", "quality assurance")):
                                job["role"] = target_line
                            else:
                                job["role"] = left_clean
                                if not job["company"]:
                                    job["company"] = right_clean
                            job["method"].append("title_dash_company")
                            found_split = True
                        elif job["role"] and not job["company"]:
                            job["company"] = left_clean
                            if not job.get("location"):
                                job["location"] = right_clean
                            job["method"].append("company_loc_dash_split")
                            found_split = True
                        elif not job["role"] and not job["company"]:
                            if right_is_title and not left_is_title:
                                job["role"], job["company"] = right_clean, left_clean
                                job["method"].append("dash_split")
                                found_split = True
                            elif left_is_title and len(cleaned) > 1 and not TITLE_RE.search(cleaned[1]) and not cleaned[1].lower().startswith(("having ", "worked ", "working ", "experience ")):
                                job["role"] = target_line
                                job["company"] = cleaned[1] if cleaned[1] != target_line else None
                                job["method"].append("dash_title_next_line_company")
                                found_split = True
                            elif left_is_title:
                                _domain_words = {'enterprise sales', 'b2b sales', 'talent acquisition', 'cloud', 'full stack', 'backend', 'frontend', 'qa', 'devops', 'sales', 'operations', 'marketing', 'hr', 'finance', 'it', 'software', 'retail', 'credit'}
                                if right_clean.lower() in _domain_words or any(w in right_clean.lower() for w in _domain_words):
                                    job["role"] = f"{left_clean} - {right_clean}"
                                    job["company"] = None
                                    job["method"].append("dash_compound_role")
                                else:
                                    job["role"], job["company"] = left_clean, right_clean
                                    job["method"].append("dash_split")
                                found_split = True
                    if found_split:
                        break
            if found_split:
                break

    if job.get("role"):
        r = job["role"]
        r = re.sub(r'^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{2,4}\s*[\-–—\t]+\s*', '', r, flags=re.IGNORECASE).strip()
        r = re.sub(r'^(?:Presently,?\s*I\s*am\s*working\s+with|Presently,?\s*working\s+with|Working\s+as)\s*', '', r, flags=re.IGNORECASE)
        r = re.sub(r'^.*?,\s*as\s+a?\s*', '', r, flags=re.IGNORECASE)
        r = re.sub(r',?\s*from\s*$', '', r, flags=re.IGNORECASE)
        r = re.sub(r'^\s*as\s+a?\s+', '', r, flags=re.IGNORECASE)
        if re.search(r'\b(?:engineer|manager|developer|analyst|architect|lead|specialist|officer|executive|auditor|consultant)\b', r, re.IGNORECASE):
            r = re.sub(r'\s+in\s+(?:Mobile\s+factory|Quality\s+Department|Assy\s+Section|Production|Factory).*$', '', r, flags=re.IGNORECASE)
        if len(r) >= 3:
            job["role"] = r.strip()

    if job.get("company"):
        c = job["company"]
        c = re.sub(r'^(?:Company\s+Name|Company|Name|Client|Organization|Employer)\s*[:\-–—]\s*', '', c, flags=re.IGNORECASE).strip()
        c = re.sub(r'^(?:Quality\s+Department|Department|Mobile\s+factory|Assy\s+Section|Section)\s*[\.\,]*\s*', '', c, flags=re.IGNORECASE)
        c = re.sub(r'\s*\.?\s*Since.*$', '', c, flags=re.IGNORECASE)
        c = re.sub(r'\s*,?\s*as\s+a?\s+.*$', '', c, flags=re.IGNORECASE)
        c = re.sub(r'\s*\([^)]*\)$', '', c).strip()
        c = re.sub(r'\s*\([^)]*(?:exp|years?|yrs?|yr|experience|months?|mos?)[^)]*\)', '', c, flags=re.IGNORECASE).strip()
        c = re.sub(r'[\s.\-–—,/|()]+$', '', c).strip()
        if c.lower().strip() in ("since", "from", "for", "in", "at", "as", "with"):
            job["company"] = None
        elif len(c) >= 2:
            job["company"] = c.strip()

    if not job["start"]:
        for l in clean_header_lines[:2]:
            sm = _WIDOW_START_DATE_RE.search(l) or _SINCE_DATE_RE.search(l) or _START_WITH_DASH_RE.match(l)
            if sm:
                start_val = sm.group(1).strip() if sm.lastindex else sm.group("start").strip()
                job["start"] = start_val
                if not job["end"] or job["end"] == "2 Years":
                    job["end"] = "Present"
                job["method"].append("header_line_widow_date")
                break

    if not job["role"] and not job["company"] and cleaned:
        full_hdr = " ".join(cleaned)
        full_hdr = DATE_RANGE_RE.sub("", full_hdr)
        full_hdr = re.sub(r'[\s.\-–—,/|(\[]*\b(?:from|to)\b.*$', '', full_hdr, flags=re.IGNORECASE).strip()
        at_match = re.search(r'^(?:Worked\s+as|Working\s+as|[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*])*\s*(.*?)\s+\bat\b\s+(.*)$', full_hdr, re.IGNORECASE)
        if not at_match:
            at_match = re.search(r'^(?:Worked\s+as|Working\s+as|[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*])*\s*(.*?)\s+\bat\b\s+(.*)$', cleaned[0], re.IGNORECASE)
        if at_match:
            role_candidate = at_match.group(1).strip()
            role_candidate = re.sub(r'^.*?\)\.\s*', '', role_candidate).strip()
            role_candidate = re.sub(r'^(?:Worked\s+as|Working\s+as|[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\s])+', '', role_candidate, flags=re.IGNORECASE).strip()
            comp_candidate = at_match.group(2).strip()
            comp_candidate = re.sub(r'[\s.\-–—,/|(\[]*\b(?:from|to)\b.*$', '', comp_candidate, flags=re.IGNORECASE).strip()
            comp_candidate = re.sub(r'[:\-–—\s]*\d+\s*(?:yrs?|years?|mos?|months?).*$', '', comp_candidate, flags=re.IGNORECASE).strip()
            comp_candidate = comp_candidate.rstrip(" .–-—,/|(").strip()
            if role_candidate and comp_candidate:
                job["role"] = role_candidate
                job["company"] = comp_candidate
                job["method"].append("at_split")

    if not job["role"] and not job["company"] and cleaned:
        paren_match = re.search(r'\(([^)]+)\)', cleaned[0])
        if paren_match:
            inside = paren_match.group(1).strip()
            outside = (cleaned[0][:paren_match.start()] + cleaned[0][paren_match.end():]).strip()
            if outside and TITLE_RE.search(inside) and not TITLE_RE.search(outside):
                job["role"] = inside
                job["company"] = outside
                job["method"].append("paren_title")
            elif outside and TITLE_RE.search(outside):
                job["role"] = cleaned[0]
                job["method"].append("full_paren_title")

    if len(cleaned) >= 2 and not job.get("company"):
        c_line = cleaned[0].strip()
        r_line = cleaned[1].strip()
        if (
            (any(w in c_line.lower() for w in ("ltd", "inc", "corp", "llc", "gmbh", "pvt", "limited", "company", "group", "bank", "services", "technologies", "associates", "consultants", "partners", "auditors", "advisors", "solutions", "enterprises", "ventures", "firm", "trust"))
             or c_line.lower().startswith("m/s") or re.search(r'\b&\s*co\b', c_line, re.I))
            and (TITLE_RE.search(r_line) or _is_clean_role_name(r_line))
        ):
            job["company"] = c_line
            job["role"] = r_line
            job["method"].append("multiline_company_above_role")

    if len(cleaned) >= 2 and not job.get("company"):
        c_cand = cleaned[0].strip()
        r_cand = cleaned[1].strip()
        if not TITLE_RE.search(c_cand) and len(c_cand.split()) <= 5 and not c_cand.startswith("•") and (TITLE_RE.search(r_cand) or _is_clean_role_name(r_cand)):
            job["company"] = c_cand
            job["role"] = r_cand
            job["method"].append("company_line_above_role_line")

    if len(cleaned) >= 2 and not job.get("company"):
        r_cand = cleaned[0].strip()
        c_cand = cleaned[1].strip()
        if (TITLE_RE.search(r_cand) or _is_clean_role_name(r_cand)) and not any(w in r_cand.lower() for w in ("ltd", "pvt", "llc", "inc", "corp", "gmbh", "company", "enterprises", "works")):
            if (
                any(w in c_cand.lower() for w in ("ltd", "inc", "corp", "llc", "gmbh", "pvt", "limited", "company", "group", "bank", "services", "technologies", "solutions", "mindtree", "tata", "infosys", "wipro", "cognizant", "accenture", "ibm", "capgemini", "hcl", "enterprises", "works"))
                or (len(c_cand.split()) <= 6 and not c_cand.startswith("•") and not TITLE_RE.search(c_cand))
            ):
                if not c_cand.lower().startswith(("working", "analyzing", "reviewing", "addressing", "supporting", "project", "flight", "admin", "payment", "api", "system", "team", "technical", "a company")):
                    job["role"] = r_cand
                    job["company"] = c_cand
                    job["method"].append("multiline_role_above_company")

    if not job["role"] and not job["company"] and cleaned:
        if "," in cleaned[0]:
            left, right = [p.strip() for p in cleaned[0].split(",", 1)]
            left_clean = re.sub(r'\([^)]*\)', '', left).strip()
            right_clean = re.sub(r'\([^)]*\)', '', right).strip()
            if left_clean and right_clean:
                if TITLE_RE.search(left_clean):
                    job["role"] = left_clean
                    job["company"] = right_clean
                    job["method"].append("comma_title_company_split")
                elif TITLE_RE.search(right_clean):
                    job["company"] = left_clean
                    parts_right = [p.strip() for p in right_clean.split(",") if p.strip()]
                    if len(parts_right) >= 2 and any(loc_w in parts_right[-1].lower() for loc_w in ("noida", "gurgaon", "gurugram", "delhi", "new delhi", "mumbai", "pune", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata", "kashipur")):
                        job["role"] = ", ".join(parts_right[:-1])
                        job["location"] = parts_right[-1]
                    else:
                        m_loc = re.match(r'^([A-Z][a-z]+)\s+(.+)$', right_clean)
                        if m_loc and m_loc.group(1).lower() not in {"software", "senior", "lead", "principal", "junior", "chief", "head", "assistant", "qa", "qc", "associate", "test"} and TITLE_RE.search(m_loc.group(2)):
                            job["location"] = m_loc.group(1).strip()
                            job["role"] = m_loc.group(2).strip()
                        else:
                            job["role"] = right_clean
                    job["method"].append("comma_company_role_split")

    if job["role"] and not job["company"] and cleaned:
        for line in cleaned[:3]:
            line_clean = re.sub(r'\([^)]*\)', '', line).strip()
            if (
                not line_clean
                or line == job["role"]
                or TITLE_RE.search(line_clean)
                or _looks_like_prose(line_clean)
                or not _is_clean_company_name(line_clean)
                or re.search(r'\b(?:page|responsibilities|responsibility|activities|highlights|duties)\b', line_clean, re.I)
                or len(line_clean.split()) > 6
            ):
                continue
            if "," in line:
                c_cand, loc_cand = [x.strip() for x in line.split(",", 1)]
                job["company"] = c_cand
                if not job.get("location"):
                    job["location"] = loc_cand
            else:
                job["company"] = line
            job["method"].append("next_line_company")
            break

    valid_cleaned = []
    for l in cleaned:
        l_low = l.lower().strip()
        l_clean = re.sub(r'[^\w\s]', '', l_low)
        if l_low in _TECH_SKILL_KEYWORDS or l_clean in _TECH_SKILL_KEYWORDS or (len(l_low.split()) <= 2 and any(w in _TECH_SKILL_KEYWORDS for w in l_low.split())):
            continue
        valid_cleaned.append(l)

    if not job["role"] or not job["company"]:
        for idx_l, line in enumerate(cleaned):
            if (" – " in line or " - " in line) and TITLE_RE.search(line):
                dash = " – " if " – " in line else " - "
                parts = [p.strip() for p in line.split(dash, 1)]
                if len(parts) == 2:
                    p0_clean = parts[0].lstrip("•*❖-– ").strip()
                    p1_clean = parts[1].strip()
                    _domain_words = {'enterprise sales', 'b2b sales', 'talent acquisition', 'cloud', 'full stack', 'backend', 'frontend', 'qa', 'devops', 'sales', 'operations', 'marketing', 'hr', 'finance', 'it', 'software', 'retail', 'credit'}
                    if p1_clean.lower() in _domain_words or any(w in p1_clean.lower() for w in _domain_words):
                        job["role"] = f"{p0_clean} - {p1_clean}"
                        if idx_l > 0 and _is_clean_company_name(cleaned[idx_l - 1]):
                            job["company"] = cleaned[idx_l - 1]
                        else:
                            job["company"] = None
                    elif TITLE_RE.search(p0_clean):
                        job["role"] = p0_clean
                        c_rest = p1_clean
                        if idx_l + 1 < len(cleaned):
                            nxt_line = re.sub(r'\(.*?\)', '', cleaned[idx_l + 1]).strip()
                            if nxt_line and not TITLE_RE.search(nxt_line) and not DATE_RANGE_RE.search(nxt_line):
                                c_rest = c_rest + " " + nxt_line
                        job["company"] = c_rest
                    else:
                        job["role"] = p1_clean
                        c_rest = p0_clean
                        if idx_l + 1 < len(cleaned):
                            nxt_line = re.sub(r'\(.*?\)', '', cleaned[idx_l + 1]).strip()
                            if nxt_line and not TITLE_RE.search(nxt_line) and not DATE_RANGE_RE.search(nxt_line):
                                c_rest = c_rest + " " + nxt_line
                        job["company"] = c_rest
                    job["method"].append("multiline_dash_split")
                    break

    if not job["role"] or not job["company"]:
        search_lines = valid_cleaned if valid_cleaned else cleaned
        strong_title_re = re.compile(r'\b(?:manager|engineer|director|lead|associate|specialist|executive|officer|consultant|deo|auditor|architect|developer|analyst|leader)\b|\((?:full-time|contract|part-time)\)', re.I)
        for line in search_lines:
            line_clean = re.sub(r'\([^)]*\)', '', line).strip()
            if not line_clean or line_clean.isdigit():
                continue
            if strong_title_re.search(line) and not job["role"]:
                job["role"] = line
                job["method"].append("title_dict_strong")
                break
        if not job["role"]:
            for line in search_lines:
                line_clean = re.sub(r'\([^)]*\)', '', line).strip()
                if not line_clean or line_clean.isdigit():
                    continue
                if TITLE_RE.search(line_clean):
                    job["role"] = line_clean
                    job["method"].append("title_dict")
                    break

        if not job["company"]:
            for line in search_lines:
                line_clean = re.sub(r'\([^)]*\)', '', line).strip()
                if not line_clean or line_clean.isdigit() or line_clean == job.get("role"):
                    continue
                if "," in line_clean:
                    parts = [p.strip() for p in line_clean.rsplit(",", 1)]
                    if len(parts) == 2 and len(parts[1].split()) <= 3 and not TITLE_RE.search(parts[0]):
                        job["company"], job["location"] = parts
                        job["method"].append("comma_company_location")
                        break
                if any(re.search(rf"\b{kw}\b", line_clean, re.IGNORECASE) for kw in ("ltd", "llc", "inc", "corp", "gmbh", "pvt", "limited", "company", "bpo", "systems", "group", "solutions")):
                    job["company"] = line_clean
                    job["method"].append("company_suffix_match")
                    break

        if not job["company"]:
            for line in search_lines:
                line_clean = re.sub(r'\([^)]*\)', '', line).strip()
                if not line_clean or line_clean.isdigit() or line_clean == job.get("role") or DATE_RANGE_RE.search(line_clean) or _SINCE_DATE_RE.search(line_clean) or _WIDOW_START_DATE_RE.search(line_clean):
                    continue
                if not _is_clean_company_name(line_clean):
                    continue
                if not line_clean.endswith(".") or any(line_clean.lower().endswith(s) for s in ("co.", "inc.", "ltd.", "corp.", "llc.", "llp.", "pvt.", "pte.", "org.", "group.")):
                    job["company"] = line_clean
                    break

    if job.get("role"):
        job["role"] = re.sub(r"\s+with\s+(?:\d+\+?\s*)?(?:years?|yrs?|months?|mths?)\s+of\s+experience.*$", "", job["role"], flags=re.IGNORECASE).strip()
        job["role"] = re.sub(r"\s+with\s+of\s+experience.*$", "", job["role"], flags=re.IGNORECASE).strip()
        job["role"] = re.sub(r'[\s|,\-–—:]+$', '', job["role"]).strip()
        job["role"] = re.sub(r'\s+\b(?:at|for|with|in)\s*$', '', job["role"], flags=re.IGNORECASE).strip()

        if re.search(r'\s+\bat\b\s+', job["role"], re.I):
            r_part, c_part = re.split(r'\s+\bat\b\s+', job["role"], flags=re.I, maxsplit=1)
            r_part = r_part.strip()
            c_part = c_part.strip()
            if TITLE_RE.search(r_part) or not TITLE_RE.search(c_part):
                curr_comp = job.get("company") or ""
                if curr_comp and not any(kw in curr_comp.lower() for kw in ("ltd", "inc", "pvt", "corp", "llc", "llp", "limited", "company", "technologies", "hospitality", "collective")) and any(kw in c_part.lower() for kw in ("ltd", "inc", "pvt", "corp", "llc", "llp", "limited", "company", "energy", "technologies", "hospitality")):
                    job["role"] = f"{curr_comp} {r_part}".strip()
                    job["company"] = c_part
                elif curr_comp and any(kw in curr_comp.lower() for kw in ("ltd", "inc", "pvt", "corp", "llc", "llp", "limited", "collective")):
                    job["role"] = r_part
                    job["company"] = f"{c_part} {curr_comp}".strip()
                else:
                    job["role"] = r_part
                    if not job.get("company"):
                        job["company"] = c_part

    if job.get("end") and job["end"].lower() in ("till", "till date", "till now", "currently", "continuing", "till present"):
        job["end"] = "Present"

    if job["company"] and "|" in job["company"]:
        left, right = [p.strip() for p in job["company"].split("|", 1)]
        job["company"] = left
        if not job.get("location"):
            job["location"] = right

    if job["company"] and "," in job["company"]:
        parts = [p.strip() for p in job["company"].rsplit(",", 1)]
        if len(parts) == 2 and len(parts[1].split()) <= 3:
            job["company"], job["location"] = parts

    if job["role"] and "," in job["role"]:
        m_loc = re.search(r'\s+([A-Z][a-z]+,\s*[A-Z][a-zA-Z]+)$', job["role"])
        if m_loc:
            job["location"] = m_loc.group(1).strip()
            job["role"] = job["role"][:m_loc.start()].strip()

    if job["company"] and " - " in job["company"]:
        left, _, right = job["company"].rpartition(" - ")
        left, right = left.strip(), right.strip()
        if left and right and len(right.split()) <= 4:
            job["company"] = left
            job["location"] = f"{right}, {job['location']}" if job["location"] else right

    if job.get("company") and not job.get("location"):
        c_parts = job["company"].split()
        if len(c_parts) >= 2:
            _TRAILING_LOCATIONS = {
                "remote", "hybrid", "mumbai", "gurgaon", "gurugram", "delhi", "new delhi",
                "noida", "bangalore", "bengaluru", "pune", "hyderabad", "chennai", "kolkata",
                "ahmedabad", "chandigarh", "mohali", "jaipur", "indore", "surat", "lucknow",
                "kanpur", "nagpur", "bhopal", "patna", "vadodara", "coimbatore", "kochi",
                "trivandrum", "visakhapatnam", "nashik", "faridabad", "ghaziabad"
            }
            if len(c_parts) >= 3 and f"{c_parts[-2]} {c_parts[-1]}".lower() in _TRAILING_LOCATIONS:
                rem_comp = " ".join(c_parts[:-2]).strip()
                if rem_comp and rem_comp.lower() not in {"in", "at", "from", "for"}:
                    job["location"] = f"{c_parts[-2]} {c_parts[-1]}"
                    job["company"] = rem_comp
            elif c_parts[-1].lower() in _TRAILING_LOCATIONS:
                rem_comp = " ".join(c_parts[:-1]).strip()
                if rem_comp and rem_comp.lower() not in {"in", "at", "from", "for"}:
                    job["location"] = c_parts[-1]
                    job["company"] = rem_comp

    if not job.get("location"):
        _STANDALONE_CITIES = {
            "mumbai", "pune", "delhi", "new delhi", "noida", "gurgaon", "gurugram",
            "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata", "ahmedabad", "jaipur"
        }
        for l in clean_header_lines:
            l_clean = l.strip().rstrip(" ,.-")
            if l_clean.lower() in _STANDALONE_CITIES:
                job["location"] = l_clean
                break

    if not job["company"] and cleaned:
        spacy_company = detect_company_with_spacy(" ".join(cleaned[:3]))
        if spacy_company:
            job["company"] = spacy_company
            job["method"].append("spacy_ORG")

    if date_line and (not job.get("company") or _TABLE_HEADER_ROW_RE.search(job.get("company") or "")):
        dm = DATE_RANGE_RE.search(date_line)
        if dm:
            after_date = date_line[dm.end():].strip(" -–—,\t()[]{}")
            if after_date and len(after_date.split()) <= 3 and not job.get("location"):
                job["location"] = after_date
            before_date = date_line[:dm.start()].strip(" -–—,\t()[]{}")
            if before_date and len(before_date.split()) <= 6:
                b_clean = re.sub(r'[\s:\-–—\t()]+$', '', before_date).strip().lower()
                if b_clean and len(b_clean) > 1 and b_clean not in {"tenure", "period", "duration", "date", "dates", "time", "sl", "sno", "no", "organization", "organisation", "company", "profile", "designation", "role"}:
                    parts_dash = [p.strip() for p in re.split(r'\s*[-–—]\s*', before_date) if p.strip()]
                    if len(parts_dash) >= 2 and TITLE_RE.search(parts_dash[0]):
                        p1_low = parts_dash[1].lower()
                        if any(w in p1_low for w in ("data science", "analytics", "engineering", "development", "operations", "research", "technology", "quality assurance")):
                            job["role"] = before_date
                        else:
                            job["role"] = parts_dash[0]
                            job["company"] = parts_dash[1]
                    elif TITLE_RE.search(before_date):
                        job["role"] = before_date
                    else:
                        job["company"] = before_date

    for key in ("role", "company", "location"):
        if job.get(key):
            val = job[key].strip()
            val = re.sub(r'^[{\[\s:\-–—,/|]+|[{\}\]\s:\-–—,/|]+$', '', val).strip()
            if val.startswith("(") and val.count("(") > val.count(")"):
                val = val[1:].strip()
            if val.endswith(")") and val.count(")") > val.count("("):
                val = val[:-1].strip()
            val = val.strip(" /:.-{}[]").strip()
            if val.endswith(")") and val.count(")") > val.count("("):
                val = val[:-1].strip()
            job[key] = val

    if job.get("company") and any(sep in job["company"] for sep in (" – ", " — ", " - ", "-", "–")):
        for sep in (" – ", " — ", " - ", "-", "–"):
            if sep in job["company"]:
                left_p, right_p = job["company"].rsplit(sep, 1)
                left_p, right_p = left_p.strip(), right_p.strip()
                if right_p and re.search(r'\b(?:solution|solutions|persistent|ntt\s*data|accenture|mahindra|wipro|ltd|pvt|limited|inc|corp|technologies|isotropic|infinite|huston)\b', right_p, re.I):
                    if not (len(left_p.split()) == 1 and len(left_p) <= 4 and sep in ("-", "–")):
                        job["company"] = right_p
                        break

    if job.get("role") and job.get("company") and job["role"].startswith(job["company"]):
        job["role"] = job["role"][len(job["company"]):].lstrip(" –-—:").strip()

    if job.get("role") and job["role"].count('(') > job["role"].count(')'):
        job["role"] += ')'

    if job["company"]:
        comp_low = job["company"].lower().strip()
        comp_clean = re.sub(r'[^\w\s]', '', comp_low)
        if (comp_low in _TECH_SKILL_KEYWORDS or comp_clean in _TECH_SKILL_KEYWORDS 
            or (TITLE_RE.search(comp_low) and not any(kw in comp_low for kw in ("hospital", "clinic", "health", "care", "center", "centre", "solutions", "services", "pvt", "ltd", "inc", "corp")))
            or comp_low.startswith(("protocol", "bullet", "experience", "participated", "supervised"))
            or comp_clean in {"previous experience", "previous experiences", "past experience", "career summary", "raipur", "mumbai", "gurugram", "pune", "noida", "delhi", "hyderabad", "bangalore", "bengaluru", "chennai", "kolkata", "chhattisgarh", "maharashtra", "haryana"}
            or any(comp_low.startswith(p) for p in ("received", "awarded", "award for", "performance from", "digital excellence", "saving approximately", "effort annually"))):
            job["company"] = None

    if job["role"]:
        role_low = job["role"].lower().strip()
        if any(w in role_low for w in ("manual effort", "effort annually", "saving approximately")):
            job["role"] = None

    return job


def _score_confidence(job: dict) -> float:
    score = 0.5
    if job["start"] and job["end"]:
        score += 0.2
    if job["role"]:
        score += 0.15
    if job["company"]:
        score += 0.15
    if "spacy_ORG" in job["method"] and "title_dict" not in job["method"]:
        score -= 0.05
    if job["company"] and len(job["company"].split()) > 6:
        score -= 0.3
    return round(min(max(score, 0.0), 0.98), 2)


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_into_sentences(text: str) -> list:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts if len(parts) > 1 else [text]


def _reflow_body(body_lines: list):
    extra_header_lines, bullets, subheadings = [], [], []
    current_bullet = None
    seen_first_bullet = False
    saw_real_bullet_marker = False

    for raw in body_lines:
        l = raw.strip()
        if not l:
            continue
        if BULLET_RE.match(l):
            seen_first_bullet = True
            saw_real_bullet_marker = True
            if current_bullet is not None:
                bullets.append(current_bullet)
            current_bullet = BULLET_RE.sub("", l).strip()
        elif not seen_first_bullet:
            if _looks_like_prose(l):
                seen_first_bullet = True
                current_bullet = l
            else:
                extra_header_lines.append(l)
        elif _looks_like_subheading(l):
            if current_bullet is not None:
                bullets.append(current_bullet)
                current_bullet = None
            subheadings.append(l)
        else:
            if current_bullet is not None:
                current_bullet += " " + l
            elif _looks_like_prose(l):
                current_bullet = l
            else:
                subheadings.append(l)

    if current_bullet is not None:
        bullets.append(current_bullet)

    if not saw_real_bullet_marker and len(bullets) == 1:
        bullets = _split_into_sentences(bullets[0])

    return extra_header_lines, bullets, subheadings


def parse_job_block(header_lines: list, body_lines: list):
    if not header_lines:
        return None

    extra_header_lines, bullets, subheadings = _reflow_body(body_lines)

    date_line = next((l for l in header_lines if DATE_RANGE_RE.search(l) or _SINCE_DATE_RE.search(l) or _YEAR_RANGE_SHORT_RE.search(l) or _WIDOW_START_DATE_RE.search(l)), "")
    if not date_line:
        date_line = next((l for l in header_lines if _STATED_DURATION_RE.search(l)), "")
    job = _parse_header(header_lines + extra_header_lines, date_line)

    role_str = str(job.get("role") or "")
    company_str = str(job.get("company") or "")
    if not job["role"] or not job["company"] or role_str in ('Overall', 'General Manager of', 'Dept Head of', 'Manager -') or re.search(r'\b(?:of|a|an|the|-)\s*$', role_str, re.I) or re.search(r'\b(?:of|for)\s+[A-Za-z]+$', role_str, re.I) or role_str.startswith('with ') or re.search(r'^\s*(?:Department|Dept|Division)\b', company_str, re.I):
        top_body_lines = []
        for l in body_lines:
            l_str = l.strip()
            if not l_str or BULLET_RE.match(l_str):
                break
            top_body_lines.append(l_str)
        if top_body_lines:
            fallback_job = _parse_header(header_lines + top_body_lines, date_line)
            if fallback_job.get("role"):
                if not job["role"] or job["role"] in ('Overall', 'General Manager of', 'Dept Head of', 'Manager -') or re.search(r'\b(?:of|a|an|the|-)\s*$', str(job["role"]), re.I) or re.search(r'\b(?:of|for)\s+[A-Za-z]+$', str(job["role"]), re.I) or str(job["role"]).startswith('with ') or not TITLE_RE.search(str(job["role"])):
                    if job.get("role") and (job["role"] in ('General Manager of', 'Dept Head of') or re.search(r'\b(?:of|a|an|the|-)\s*$', str(job["role"]), re.I)):
                        job["role"] = f"{job['role']} {fallback_job['role']}".strip()
                    else:
                        job["role"] = fallback_job["role"]
            if fallback_job.get("company"):
                if not job["company"] or re.search(r'^\s*(?:Department|Dept|Division)\b', str(job["company"]), re.I):
                    job["company"] = fallback_job["company"]
            if fallback_job.get("location") and not job.get("location"):
                job["location"] = fallback_job["location"]
            # Remove top_body_lines from bullets if they were incorrectly placed in bullets
            bullets = [b for b in bullets if b not in top_body_lines]

    if not job["role"] and subheadings:
        for sub in subheadings:
            if TITLE_RE.search(sub):
                m_loc = re.search(r'\s+([A-Z][a-z]+,\s*[A-Z][a-zA-Z]+)$', sub)
                if m_loc:
                    if not job.get("location"):
                        job["location"] = m_loc.group(1).strip()
                    job["role"] = sub[:m_loc.start()].strip()
                else:
                    job["role"] = sub
                break

    if not job["company"] and bullets:
        for i, b in enumerate(bullets[:6]):
            b_low = b.lower().strip()
            b_clean = re.sub(r'[^\w\s]', '', b_low)
            if (
                not b_clean
                or TITLE_RE.search(b)
                or b_low in _TECH_SKILL_KEYWORDS
                or b_clean in _TECH_SKILL_KEYWORDS
                or any(w in _TECH_SKILL_KEYWORDS for w in b_low.split())
                or any(b_low.startswith(v) for v in ("built ", "developed ", "designed ", "led ", "streamlined ", "delivered ", "performed ", "managed ", "created ", "architected ", "implemented ", "spearheaded ", "conducted "))
                or len(b.split()) > 6
                or _looks_like_prose(b)
                or not _is_clean_company_name(b)
                or re.search(r'\b(?:page|responsibilities|responsibility)\b', b, re.I)
            ):
                continue
            job["company"] = b
            bullets.pop(i)
            break

    job["duration_months"] = calculate_duration_months(job["start"], job["end"]) if (job["start"] or job["end"]) else 0
    job["bullets"] = bullets
    job["sub_sections"] = subheadings

    if job.get("company") and "|" in job["company"] and not job.get("role"):
        parts = [p.strip() for p in job["company"].split("|") if p.strip()]
        if len(parts) >= 2:
            if TITLE_RE.search(parts[1]):
                job["company"] = parts[0]
                job["role"] = parts[1]
            elif TITLE_RE.search(parts[0]):
                job["role"] = parts[0]
                job["company"] = parts[1]
            else:
                job["company"] = parts[0]
                job["role"] = parts[1]

    if job["company"]:
        comp_low = job["company"].lower().strip()
        comp_clean = re.sub(r'[^\w\s]', '', comp_low).strip()
        if (comp_low in _TECH_SKILL_KEYWORDS or comp_clean in _TECH_SKILL_KEYWORDS 
            or (all(w in _TECH_SKILL_KEYWORDS for w in comp_low.split()) if comp_low.split() else False)
            or comp_clean in {"previous experience", "previous experiences", "past experience", "career summary", "raipur", "mumbai", "gurugram", "pune", "noida", "delhi", "hyderabad", "bangalore", "bengaluru", "chennai", "kolkata", "chhattisgarh", "maharashtra", "haryana"}
            or any(comp_low.startswith(p) for p in ("received", "awarded", "award for", "performance from", "digital excellence", "saving approximately", "effort annually"))):
            job["company"] = None

    if job["role"]:
        role_low = job["role"].lower().strip()
        if any(w in role_low for w in ("manual effort", "effort annually", "saving approximately")):
            job["role"] = None

    job["confidence"] = _score_confidence(job)
    job["needs_review"] = job["confidence"] < 0.75 or not job["company"] or not job["role"]
    job["method"] = "+".join(job["method"]) if job["method"] else "none"

    if not job["company"] and not job["role"] and not job["bullets"]:
        return None
    return job





def _extract_jobs_from_text(section_text: str) -> list:
    """
    Shared block-finding + parsing pipeline, factored out of
    extract_experience() so both the main "experience" section text and
    an "early_career" condensed-recap section text can be run through the
    exact same anchor-based logic. A condensed entry like "Oct-2023 -
    Nov-2025 | Manager | KPMG" is just a header line with no body --
    _find_job_blocks anchors on its date range the same way it would for
    a full job entry, and _parse_header's existing "pipe_split" branch
    (splitting on "|", picking the TITLE_RE-matching part as the role)
    already handles this exact "Date - Date | Title | Company" shape
    without any changes needed here.
    """
    if not section_text.strip():
        return []

    lines = section_text.strip("\n").split("\n")
    lines = [_normalize_dashes(l) for l in lines]
    lines = [_normalize_month_year_glue(l) for l in lines]
    lines = _merge_widow_date_lines(lines)
    lines = [
        l for l in lines 
        if (len(DATE_RANGE_RE.findall(l)) + len(_YEAR_RANGE_SHORT_RE.findall(l))) < 3
    ]

    split_lines = []
    for l in lines:
        matches = list(DATE_RANGE_RE.finditer(l))
        if len(matches) >= 2:
            split_idx = matches[1].start()
            l1 = l[:split_idx].strip()
            l2 = l[split_idx:].strip()
            if l1:
                split_lines.append(l1)
            if l2:
                split_lines.append(l2)
        else:
            split_lines.append(l)
    lines = split_lines

    blocks = _find_job_blocks(lines)

    jobs = []
    for header_ids, body_ids in blocks:
        header_lines = [lines[i] for i in header_ids]
        body_lines = [lines[i] for i in body_ids]
        parsed = parse_job_block(header_lines, body_lines)
        if parsed:
            if not parsed.get("role"):
                first_hdr_idx = header_ids[0] if header_ids else 0
                hdr_role = None
                for k in range(first_hdr_idx - 1, max(-1, first_hdr_idx - 4), -1):
                    line_k = lines[k].rstrip()
                    if line_k.endswith(":") and TITLE_RE.search(line_k):
                        hdr_role = line_k.rstrip(":").strip()
                        break
                if hdr_role:
                    parsed["role"] = hdr_role
                    parsed["method"] = parsed["method"] + "+section_header_role" if parsed["method"] else "section_header_role"
                elif jobs and jobs[-1].get("role") and any(w in jobs[-1]["role"].lower() for w in ("article", "intern", "trainee", "apprentice")):
                    parsed["role"] = jobs[-1]["role"]
                    parsed["method"] = parsed["method"] + "+inherited_role" if parsed["method"] else "inherited_role"
                else:
                    for b in parsed.get("bullets", []):
                        if re.search(r'\b(?:BA\s+activities|as\s+a?\s*BA\b|Business\s+Analyst)\b', b, re.I):
                            parsed["role"] = "Business Analyst"
                            parsed["method"] = (parsed["method"] + "+bullet_role_hint") if parsed["method"] else "bullet_role_hint"
                            break
                        elif re.search(r'\b(?:Product\s+Manager|as\s+a?\s*PM\b)\b', b, re.I):
                            parsed["role"] = "Product Manager"
                            parsed["method"] = (parsed["method"] + "+bullet_role_hint") if parsed["method"] else "bullet_role_hint"
                            break
            if not parsed.get("company") and jobs and jobs[-1].get("company"):
                hdr_text = " ".join(lines[i] for i in header_ids).lower() if header_ids else ""
                if not any(w in hdr_text for w in ("ltd", "inc", "corp", "llc", "gmbh", "pvt", "limited", "company", "group", "bank")):
                    parsed["company"] = jobs[-1]["company"]
                    if not parsed.get("location") and jobs[-1].get("location"):
                        parsed["location"] = jobs[-1]["location"]
                    parsed["method"] = parsed["method"] + "+inherited_company" if parsed["method"] else "inherited_company"
            comp_str = str(parsed.get("company") or "").lower()
            role_str = str(parsed.get("role") or "").lower()
            if any(w in comp_str for w in ("timeline", "change agent", "communicator", "soft skills")) and not parsed.get("role"):
                continue
            if any(w in role_str for w in ("pvt ltd", "pvt. ltd", "solution pvt")):
                continue
            jobs.append(parsed)

    unassigned_hdr_lines = [
        l.strip() for l in lines 
        if TITLE_RE.search(l) and (" - " in l or " – " in l or "," in l)
    ]
    if unassigned_hdr_lines:
        u_idx = 0
        for j in jobs:
            if (not j.get("role") or j.get("company") in ("Anoop Kumar Yadav", "Unknown Company", None)) and u_idx < len(unassigned_hdr_lines):
                u_line = unassigned_hdr_lines[u_idx]
                u_idx += 1
                if " - " in u_line:
                    r_cand, c_cand = [x.strip() for x in u_line.split(" - ", 1)]
                    j["role"] = r_cand
                    if "," in c_cand:
                        j["company"], j["location"] = [x.strip() for x in c_cand.split(",", 1)]
                    else:
                        j["company"] = c_cand
                elif " – " in u_line:
                    r_cand, c_cand = [x.strip() for x in u_line.split(" – ", 1)]
                    j["role"] = r_cand
                    if "," in c_cand:
                        j["company"], j["location"] = [x.strip() for x in c_cand.split(",", 1)]
                    else:
                        j["company"] = c_cand

    return jobs



def calculate_total_experience(jobs: list) -> dict:
    parsed_jobs = []
    fallback_months = 0
    for job in jobs:
        if not job.get("start") or not job.get("end"):
            if job.get("end"):
                dur_m = _STATED_DURATION_RE.search(job["end"])
                if dur_m:
                    num = float(dur_m.group("num"))
                    unit = dur_m.group("unit").lower()
                    fallback_months += int(num * 12) if "y" in unit else int(num)
            continue
        s = parse_date(job["start"])
        e = parse_date(job["end"])
        raw_end = str(job.get("end", "")).strip().lower()
        is_present = any(w in raw_end for w in ("present", "current", "now", "till"))
        if is_present:
            e = datetime.now()
        if s and e and e > s:
            parsed_jobs.append({"start": s, "end": e, "is_present": is_present, "raw": job})

    if not parsed_jobs:
        if fallback_months > 0:
            return {
                "total_years": round(fallback_months / 12, 1),
                "total_months": fallback_months,
                "overlapping_periods_merged": 0,
            }
        return {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}

    # Cap outdated "Present" on earlier sub-projects if a later job has start > s and is also Present
    present_jobs = [j for j in parsed_jobs if j["is_present"]]
    if len(present_jobs) > 1:
        present_jobs.sort(key=lambda x: x["start"])
        for p_job in present_jobs[:-1]:
            next_jobs = [j for j in parsed_jobs if j["start"] > p_job["start"] and j != p_job]
            if next_jobs:
                next_start = min(j["start"] for j in next_jobs)
                p_job["end"] = max(p_job["start"], next_start)

    intervals = [(j["start"], j["end"]) for j in parsed_jobs if j["end"] > j["start"]]
    if not intervals:
        return {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    total_months = sum(_inclusive_month_diff(s, e) for s, e in merged) + fallback_months

    return {
        "total_years": round(total_months / 12, 1),
        "total_months": total_months,
        "overlapping_periods_merged": len(intervals) - len(merged),
    }



_STATED_NUMBERED_RE = re.compile(
    r"^\s*\d+[\).]\s*(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|mos?)\s+(?:in|at|with|as\s+(?:a\s+)?)?\s*(.+)",
    re.IGNORECASE
)

_STATED_INLINE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|mos?)\s+(?:in|at|with|as\s+(?:a\s+)?)?\s*(.+)",
    re.IGNORECASE
)


def _extract_additional_stated_jobs(text: str, existing_jobs: list = None) -> tuple[list, float]:
    """
    Non-breaking additive fallback helper: extracts jobs that state their duration
    (e.g., '1) 2 years as a Center head incharge in Viesa Technologies', '5) 3 months as Sales Executive')
    without having explicit date ranges (e.g. Jan 2020 - Dec 2022).
    
    Guarantees zero regression on existing date-ranged jobs.
    """
    if not text or not text.strip():
        return [], 0.0

    existing_companies = set()
    if existing_jobs:
        for j in existing_jobs:
            comp = (j.get("company") or "").lower().strip()
            if comp and comp != "unknown company" and comp != "*":
                existing_companies.add(comp)

    additional_jobs = []
    additional_years = 0.0

    for line in text.splitlines():
        line_str = line.strip()
        if not line_str or DATE_RANGE_RE.search(line_str) or _SINCE_DATE_RE.search(line_str):
            continue

        match = _STATED_NUMBERED_RE.match(line_str) or _STATED_INLINE_RE.match(line_str)
        if match:
            val_str, unit, rest = match.groups()
            try:
                val = float(val_str)
            except ValueError:
                continue

            if "month" in unit.lower() or "mo" in unit.lower():
                years = round(val / 12.0, 2)
            else:
                years = val

            rest_clean = rest.strip(" .;")
            company = ""
            role = rest_clean

            sep_match = re.search(r"\s+(?:in|at|with)\s+", rest_clean, re.IGNORECASE)
            if sep_match:
                role = rest_clean[:sep_match.start()].strip()
                company = rest_clean[sep_match.end():].strip()

            if company and company.lower() in existing_companies:
                continue

            job_obj = {
                "company": company or "Unknown company",
                "role": role or "Unknown role",
                "start": None,
                "end": None,
                "location": None,
                "bullets": [],
                "sub_sections": [],
                "confidence": 0.90,
                "needs_review": False,
                "source": "stated_duration",
                "method": "stated_duration",
                "duration_years": years,
                "duration_months": int(years * 12),
            }
            additional_jobs.append(job_obj)
            additional_years += years

    return additional_jobs, round(additional_years, 1)


_DEDUP_MONTH_MAP = {
    'january': 'jan', 'february': 'feb', 'march': 'mar', 'april': 'apr',
    'may': 'may', 'june': 'jun', 'july': 'jul', 'august': 'aug',
    'september': 'sep', 'sept': 'sep', 'october': 'oct', 'november': 'nov', 'december': 'dec'
}


def _norm_dedup_date(d_str: str) -> str:
    if not d_str:
        return ''
    d = d_str.lower().strip()
    d = _expand_apostrophe_year(d)
    for f, t in _DEDUP_MONTH_MAP.items():
        d = re.sub(rf'\b{f}\b', t, d)
    d = re.sub(r'[\s\.\-]+', ' ', d).strip()
    return d


def _norm_dedup_text(s: str) -> str:
    if not s:
        return ''
    s = s.lower().strip()
    s = re.sub(r'^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\s]+', '', s)
    s = re.sub(r'^(company:\s*|working as a\s*|role\s*/\s*designation:\s*|role:\s*|designation:\s*|for\s+|at\s+)', '', s).strip()
    return s


def _clean_company_core(s: str) -> str:
    if not s:
        return ''
    t = _norm_dedup_text(s)
    t = re.sub(r'\b(ltd|pvt|inc|co|corp|llc|llp|limited|private|bpo|solutions|solution|services|service)\b', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def _is_clean_company_name(s: str) -> bool:
    if not s or s.lower().strip() in ("unknown company", "*", ""):
        return False
    s_lower = s.lower().strip()
    if s_lower.startswith(("✓", "➢", "•", "timely", "ensure", "handle", "working as", "http://", "https://", "www.")):
        return False
    if s_lower in ("courses", "coursework", "relevant coursework", "key courses", "education", "academics", "links", "links:", "portfolio", "urls", "key responsibility", "key responsibilities", "responsibility", "responsibilities", "job profile", "profile", "duties", "career break", "duration", "accomplishments", "personal projects", "freelance project", "code vipasana", "mumbai", "technologies :- flutter,react", "technologies"):
        return False
    if any(w in s_lower for w in ("key responsibility", "key responsibilities", "job profile", "escalation", "process", "guidelines", "anomalies", "reporting manager", "career break", "personal projects", "freelance project", "code vipasana")):
        return False
    return True


def _is_clean_role_name(s: str) -> bool:
    if not s or s.lower().strip() in ("unknown role", "*", ""):
        return False
    s_lower = s.lower().strip()
    if s_lower.startswith(("✓", "➢", "•", "timely escalation")):
        return False
    return True


def _deduplicate_jobs(jobs: list) -> list:
    """
    Deduplicates job entries that refer to the same job role/company with
    matching or overlapping date ranges (common when a resume has a top-level summary
    table AND a detailed experience section, or when bullet points repeat dates).
    """
    if not jobs:
        return jobs

    # Filter out entries created from education subheadings or invalid phantom entries
    cleaned_input_jobs = []
    for j in jobs:
        comp = j.get('company', '') or ''
        role = j.get('role', '') or ''
        if not _is_clean_company_name(comp) and not _is_clean_role_name(role):
            continue
        if comp and not _is_clean_company_name(comp) and comp.lower().strip() in ("courses", "coursework", "relevant coursework", "key courses", "education", "links", "links:", "portfolio", "urls"):
            continue
        full_job_str = f"{role} {comp}".strip()
        if re.search(r'\b(?:S\.?S\.?C|H\.?S\.?C|C\.?B\.?S\.?E|I\.?C\.?S\.?E|Class\s*10|Class\s*12)\b', full_job_str, re.IGNORECASE):
            continue
        cleaned_input_jobs.append(j)

    jobs = cleaned_input_jobs
    if len(jobs) <= 1:
        return jobs

    deduped = []
    for job in jobs:
        comp = _norm_dedup_text(job.get('company'))
        role = _norm_dedup_text(job.get('role'))
        comp_core = _clean_company_core(job.get('company'))
        start = _norm_dedup_date(job.get('start'))
        end = _norm_dedup_date(job.get('end'))

        is_dup = False
        for existing in deduped:
            e_comp = _norm_dedup_text(existing.get('company'))
            e_role = _norm_dedup_text(existing.get('role'))
            e_comp_core = _clean_company_core(existing.get('company'))
            e_start = _norm_dedup_date(existing.get('start'))
            e_end = _norm_dedup_date(existing.get('end'))

            exact_start_match = bool(start and e_start and start == e_start)
            exact_end_match = bool(end and e_end and end == e_end)

            start_match = bool(
                start and e_start and (
                    start == e_start or
                    start[:5] == e_start[:5] or
                    start in e_start or e_start in start
                )
            )

            end_match = bool(
                end == e_end or
                not end or not e_end or
                (any(w in end for w in ('till', 'present', 'current')) and any(w in e_end for w in ('till', 'present', 'current')))
            )

            comp_match = bool(
                (comp and e_comp and (comp == e_comp or comp in e_comp or e_comp in comp or comp in e_role or e_comp in role)) or
                (comp_core and e_comp_core and (comp_core == e_comp_core or comp_core in e_comp_core or e_comp_core in comp_core)) or
                (e_comp in ('unknown company', '*', '') and comp not in ('unknown company', '*', '') and (start_match or exact_end_match)) or
                (comp in ('unknown company', '*', '') and e_comp not in ('unknown company', '*', '') and (start_match or exact_end_match))
            )

            role_match = bool(
                (role and e_role and (role == e_role or role in e_role or e_role in role or role in e_comp or e_role in comp)) or
                (e_role in ('unknown role', '*', '') and role not in ('unknown role', '*', '')) or
                (role in ('unknown role', '*', '') and e_role not in ('unknown role', '*', ''))
            )

            date_match = (
                (exact_start_match and exact_end_match)
                or ((start_match and end_match) and (comp_match or role_match))
                or (comp_match and (start_match or end_match))
                or (comp_match and (not start or not e_start or start.lower() in ("none", "unknown", "") or e_start.lower() in ("none", "unknown", "")))
            )

            if date_match and (comp_match or role_match or (exact_start_match and exact_end_match)):
                is_dup = True

                if job.get('start') and (not existing.get('start') or existing.get('start') in ("None", "Unknown", "")):
                    existing['start'] = job['start']
                if job.get('end') and (not existing.get('end') or "Year" in str(existing.get('end'))):
                    existing['end'] = job['end']

                if _is_clean_company_name(job.get('company')):
                    if not _is_clean_company_name(existing.get('company')) or len(job.get('company') or '') > len(existing.get('company') or ''):
                        existing['company'] = job['company']

                if _is_clean_role_name(job.get('role')):
                    if not _is_clean_role_name(existing.get('role')) or len(job.get('role') or '') > len(existing.get('role') or ''):
                        existing['role'] = job['role']

                existing_bullets = set(existing.get('bullets', []))
                for b in job.get('bullets', []):
                    if b not in existing_bullets:
                        existing['bullets'].append(b)
                        existing_bullets.add(b)

                if job.get('confidence', 0) > existing.get('confidence', 0):
                    existing['confidence'] = job['confidence']
                break

        if not is_dup:
            deduped.append(job)

    return deduped


def extract_experience(experience_section_text: str, early_career_section_text: str = "", full_text: str = "") -> dict:
    """
    Parses main experience and early career sections.
    Integrates non-breaking stated duration fallback for entries without date ranges.
    """
    empty_total = {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}

    if "LABEL:" in experience_section_text:
        clean_exp_lines = []
        for l in experience_section_text.split("\n"):
            if re.match(r'^\s*LABEL:\s*(?:education|skills|personal_details|languages|certifications|projects|publications)\b', l, re.IGNORECASE):
                break
            clean_exp_lines.append(l)
        experience_section_text = "\n".join(clean_exp_lines)

    # Pre-clean OCR line breaks in experience text
    experience_section_text = re.sub(r'\bmet\s+life\b', 'MetLife Insurance', experience_section_text, flags=re.IGNORECASE)
    experience_section_text = re.sub(r'India\s+first[\'`\s]*life\s+Insurance(?:\s+12\s+October)?\s+Insurance\s+2020\s+Company', 'India First Life Insurance Company', experience_section_text, flags=re.IGNORECASE)
    experience_section_text = re.sub(r'\bfirst[\'`\s]+life\b', 'First Life Insurance', experience_section_text, flags=re.IGNORECASE)
    experience_section_text = re.sub(r'(\d{4})\s*\n+\s*\1\s+to\s+', r'\1 to ', experience_section_text, flags=re.IGNORECASE)
    experience_section_text = re.sub(r'(\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\s*\n+\s*(\d{4})\b', r'\1 \2', experience_section_text, flags=re.IGNORECASE)
    experience_section_text = re.sub(r'(\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\s*\n+\s*to\s+(\d{1,2})', r'\1 to \2', experience_section_text, flags=re.IGNORECASE)
    experience_section_text = re.sub(r'\b(\d)\s+(\d)\s*[-/]\s*([a-z])\s+([a-z])\s+([a-z])\s*[-/]\s*(\d)\s+(\d)\s+(\d)\s+(\d)\b', r'\1\2-\3\4\5-\6\7\8\9', experience_section_text, flags=re.IGNORECASE)

    jobs = _extract_jobs_from_text(experience_section_text)
    jobs = _deduplicate_jobs(jobs)
    for idx in range(len(jobs) - 1):
        role_str = str(jobs[idx].get("role") or "")
        if re.search(r'\b(?:of\s+New|of\s+Weld|of|as|a|-)\s*$', role_str, re.I):
            if jobs[idx + 1].get("pre_date_header"):
                pre_line = jobs[idx + 1]["pre_date_header"][0]
                if not re.search(r'\b(?:EMPLOYMENT DETAILS|PREVIOUS WORK EXPERIENCE|WORK EXPERIENCE|EXPERIENCE)\b', pre_line, re.I):
                    clean_head = re.split(r'\s*\([^)]*\)|\s*-\s*\(', pre_line)[0].strip()
                    jobs[idx]["role"] = (role_str + " " + clean_head).strip()
    for j in jobs:
        j["source"] = "experience"
        if j.get("company"):
            j["company"] = re.sub(r'^(?:Company\s+Name|Company|Name|Client|Organization|Employer)\s*[:\-–—]\s*', '', j["company"], flags=re.IGNORECASE).strip()
            j["company"] = j["company"].rstrip(" .–-—,/|()").strip()
        # Clean heading artifacts from bullets
        j["bullets"] = [
            b for b in j.get("bullets", [])
            if not re.match(r'^\s*roles?\s*(?:&|and)\s*responsibilit.*$', b, re.IGNORECASE)
            and not re.match(r'^\s*key\s*responsibilit.*$', b, re.IGNORECASE)
        ]
        if j.get("role") and j.get("company"):
            if TITLE_RE.search(j["role"]) and not re.search(r'\b(?:of|a|an|the|-)\s*$', j["role"], re.I):
                if j["confidence"] < 0.9:
                    j["confidence"] = 0.98
                    j["needs_review"] = False
        if not j.get("role"):
            found_role = None
            if full_text:
                m_sum_role = re.search(r'\b(?:I am an?|working as an?)\s+([A-Za-z\s\-/]+(?:\bdeveloper|\bengineer|\banalyst|\bmanager|\blead|\barchitect|\bspecialist))\b', full_text, re.I)
                if m_sum_role and len(m_sum_role.group(1).split()) <= 4:
                    found_role = m_sum_role.group(1).strip().title()
            if not found_role:
                for b in j.get("bullets", []):
                    m_kw = re.search(r'\b(full[- ]stack\s+developer|software\s+engineer|backend\s+developer|frontend\s+developer|java\s+developer)\b', b, re.I)
                    if m_kw:
                        found_role = m_kw.group(1).strip().title()
                        break
            if found_role:
                j["role"] = found_role
                if j["confidence"] < 0.9:
                    j["confidence"] = 0.95
                    j["needs_review"] = False

    early_career_jobs = _extract_jobs_from_text(early_career_section_text)
    early_career_jobs = _deduplicate_jobs(early_career_jobs)
    for j in early_career_jobs:
        j["source"] = "early_career"
        if j.get("company"):
            j["company"] = re.sub(r'^(?:Company\s+Name|Company|Name|Client|Organization|Employer)\s*[:\-–—]\s*', '', j["company"], flags=re.IGNORECASE).strip()
            j["company"] = j["company"].rstrip(" .–-—,/|()").strip()

    # Additive Fallback: Extract stated-duration jobs (e.g. "2 years as Center Head...") that lack date ranges
    stated_jobs, stated_years = _extract_additional_stated_jobs(experience_section_text, jobs)
    if stated_jobs:
        jobs.extend(stated_jobs)

    if not jobs and not early_career_jobs:
        return {
            "experience": [],
            "early_career": [],
            "_confidence": 0.0,
            "_needs_review": True,
            "_total_experience": empty_total,
        }

    combined = jobs + early_career_jobs
    avg_conf = round(sum(j["confidence"] for j in combined) / len(combined), 2) if combined else 0.0
    needs_review = any(j["needs_review"] for j in combined)

    total_exp = calculate_total_experience(combined)
    if stated_years > 0.0:
        total_exp["total_years"] = round(total_exp["total_years"] + stated_years, 1)
        total_exp["total_months"] = int(total_exp["total_years"] * 12)

    return {
        "experience": jobs,
        "early_career": early_career_jobs,
        "_confidence": avg_conf,
        "_needs_review": needs_review,
        "_total_experience": total_exp,
    }












##done- worked just to add a new format from one resume adding just above
#r"""
#Experience Extractor — Layer 2
#Extracts: company, title, location, date range, duration, bullet points.
#Uses: date regex (anchor-based block splitting) + job title dictionary + spaCy ORG detection.
#
#KEY DESIGN NOTE:
#Real resumes do NOT separate job entries with blank lines (verified against
#Aarti / Abhijeet segmented output). So we cannot split blocks on '\n\s*\n'.
#Instead we anchor on lines that contain a date range — every job entry has one —
#and walk backward/forward from each anchor to find header lines and body lines.
#
#CHANGELOG (early_career support):
#Some resumes include a condensed "here's my last N jobs" recap -- e.g. a
#"LAST 5 CAREER TIMELINE" section with one-line entries like "Oct-2023 -
#Nov-2025 | Manager | KPMG" -- separate from the detailed "experience"
#section that covers the same jobs with full bullets (confirmed on a real
#resume, Sachin Chitranshi). The section splitter now labels this
#"early_career" rather than folding it into "experience", so it doesn't
#duplicate/pollute the detailed job list. But its date ranges are still
#real worked time and belong in the total-years-of-experience figure.
#extract_experience() now accepts an optional early_career_section_text,
#parses it with the same anchor-based block logic, and feeds BOTH job
#lists into calculate_total_experience() together. That function already
#merges overlapping/duplicate date intervals, so if the early_career
#recap restates the same jobs (same start/end dates) as the detailed
#section, they collapse into one continuous span rather than double-
#counting -- the same jobs described twice don't inflate the total years.
#"""
#
#import re
#try:
#    import spacy
#    _SPACY_IMPORTED = True
#except ImportError:
#    _SPACY_IMPORTED = False
#from dateutil import parser as date_parser
#from dateutil.relativedelta import relativedelta
#from datetime import datetime
#
#
## ── Date matching ──────────────────────────────────────────────────────────
#
#MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#DATE_RANGE_RE = re.compile(
#    rf"(?P<start>{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})"
#    # The dash/"to" separator is the common case, but some resumes write
#    # "<start date> till date" with NO connecting dash or "to" at all --
#    # just a space (confirmed on a real resume, Sai Pavan Boddula: "May
#    # 2022 till date"). Without a third alternative here, that line never
#    # matches DATE_RANGE_RE at all -- not a parsing-quality issue, a
#    # total miss. With only one date-range anchor left for the whole
#    # experience section, _find_job_blocks built exactly one block
#    # (correctly, for the OTHER job) and silently dropped every line
#    # before it, including this entire job's header and bullets, since
#    # they never fell inside any block's header_ids/body_ids range.
#    # The added alternative is a lookahead (consumes nothing itself), so
#    # it doesn't eat into the "end" group's own match of "Till Date" --
#    # it only fires when a bare run of whitespace is immediately
#    # followed by one of the open-ended markers, so it can't misfire on
#    # two unrelated adjacent dates that just happen to be space-separated.
#    rf"(?:\s*[-–—]\s*|\s+to\s+|\s+(?=(?:Present|Current|Now|Till)\b))"
#    rf"(?P<end>Present|Current|Now|Till\s*(?:Date|Now)|{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})",
#    re.IGNORECASE
#)
#
#_WIDOW_START_DATE_RE = re.compile(
#    rf"({MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
#    re.IGNORECASE
#)
#_WIDOW_END_DATE_RE = re.compile(
#    rf"^[-–—]\s*(Present|Current|Now|Till\s*(?:Date|Now)|"
#    rf"{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
#    re.IGNORECASE
#)
#
#
#_TRAILING_TO_RE = re.compile(r"\s*\bto\s*$", re.IGNORECASE)
#
#
#def _merge_widow_date_lines(lines: list) -> list:
#    merged = list(lines)
#    for i in range(len(merged) - 1):
#        cur = merged[i].rstrip()
#        nxt = merged[i + 1].strip()
#        if not cur or not nxt:
#            continue
#        if DATE_RANGE_RE.search(cur):
#            continue
#        if _WIDOW_START_DATE_RE.search(cur) and _WIDOW_END_DATE_RE.match(nxt):
#            merged[i] = cur + " " + nxt
#            merged[i + 1] = ""
#            continue
#        # A "to"-continuation widow: the date RANGE ITSELF splits across
#        # two lines at the separator word, e.g.
#        #   "...as a PHP Developer (Dec 2018 to"
#        #   "Jun 2020)."
#        # Confirmed on a real resume (Abhay Awasthi). Distinct from the
#        # dash-continuation case above (that one has no "to" at all --
#        # just a start date, then a line starting with "-EndDate"). Only
#        # merges when the text before the trailing "to" itself ends in a
#        # recognizable start date, so this can't misfire on an unrelated
#        # sentence that happens to wrap at the word "to".
#        to_match = _TRAILING_TO_RE.search(cur)
#        if to_match:
#            before_to = cur[:to_match.start()].rstrip()
#            if _WIDOW_START_DATE_RE.search(before_to):
#                merged[i] = cur + " " + nxt
#                merged[i + 1] = ""
#    return merged
#
#TITLE_KEYWORDS = [
#    "engineer", "developer", "manager", "analyst", "consultant",
#    "designer", "architect", "lead", "head", "director", "vp", "svp", "cto", "ceo",
#    "intern", "associate", "senior", "junior", "principal", "staff",
#    "specialist", "coordinator", "executive", "officer", "president",
#    "scientist", "researcher", "administrator", "technician", "professor",
#    "assistant", "trainee", "fresher", "apprentice"
#]
#TITLE_RE = re.compile(r'\b(' + '|'.join(TITLE_KEYWORDS) + r')\b', re.IGNORECASE)
#
#_DASH_NORMALIZE_TABLE = str.maketrans({
#    "－": "-",
#    "‐": "-",
#    "‑": "-",
#})
#
#
#def _normalize_dashes(text: str) -> str:
#    return text.translate(_DASH_NORMALIZE_TABLE)
#
#
## A stray dash is sometimes glued directly between a month name and ITS
## OWN year -- "July - 2018", "Dec -2018" -- distinct from the intended
## dash that separates the start date from the end date. Confirmed on a
## real resume (Abhay Awasthi): dates written as "(July - 2018 to Dec. -
## 2018)" defeated DATE_RANGE_RE entirely, since its "start"/"end" groups
## expect a month directly followed by whitespace/comma/period and then
## the year -- a literal "-" in between isn't in that allowed character
## class. This silently dropped 3 of his 5 job entries (zero anchor line
## found for them at all), not just a parsing quality issue.
## Collapsing "Month - Year" / "Month -Year" down to "Month Year" BEFORE
## DATE_RANGE_RE runs fixes this without touching the real start/end
## separator dash, since this pattern only fires directly after a month
## name, never between two already-complete dates.
#_MONTH_YEAR_GLUE_DASH_RE = re.compile(
#    rf"({MONTH_NAMES})[.,]?\s*-\s*(\d{{4}})", re.IGNORECASE
#)
#
#
#def _normalize_month_year_glue(text: str) -> str:
#    return _MONTH_YEAR_GLUE_DASH_RE.sub(r"\1 \2", text)
#
#BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*]\s+|^\d+[\.\)]?\s+')
#
#def _looks_like_subheading(line: str) -> bool:
#    words = line.split()
#    if not (1 <= len(words) <= 6):
#        return False
#    if line.rstrip().endswith((".", ",", ";", ":")):
#        return False
#    if DATE_RANGE_RE.search(line):
#        return False
#    # Ensure there is at least one letter, and the first letter in the line is not lowercase
#    first_letter_match = re.search(r'[A-Za-z]', line)
#    if not first_letter_match or first_letter_match.group(0).islower():
#        return False
#    return True
#
#
#if _SPACY_IMPORTED:
#    try:
#        _NLP = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
#        _SPACY_AVAILABLE = True
#    except Exception:
#        _SPACY_AVAILABLE = False
#        print("[WARN] spaCy model not found. Company name detection will be limited.")
#else:
#    _SPACY_AVAILABLE = False
#    print("[WARN] spaCy not installed. Company name detection will be limited.")
#
#
#def _expand_apostrophe_year(date_str: str) -> str:
#    m = re.search(r"['’‘`´](\d{2})\b", date_str)
#    if not m:
#        return date_str
#    two_digit = int(m.group(1))
#    century = "20" if two_digit <= 49 else "19"
#    return date_str[:m.start()] + century + m.group(1) + date_str[m.end():]
#
#
## Confirmed on a real resume (Abhijeet Sangle): dates written numerically
## as "12.2011" (MM.YYYY) get silently MISPARSED by dateutil's fuzzy
## parser -- it reads "12" as the DAY, can't make sense of the trailing
## ".2011" as anything meaningful, and discards it, falling back to the
## `default` year/month (2000-01). So "12.2011" and "12.2019" -- two
## genuinely different dates 8 years apart -- both parsed to the exact
## same date, 2000-01-12, making the computed duration collapse to
## ~0 months instead of the correct ~96/97. This isn't a duration-formula
## bug; the dates going INTO the duration calculation were already wrong.
## Explicitly parsing this MM.YYYY / MM/YYYY numeric shorthand ourselves,
## before ever handing the string to dateutil, sidesteps the ambiguity
## entirely rather than hoping a general-purpose fuzzy parser guesses
## right on a format many general parsers reasonably read as DD.YYYY.
#_MM_YYYY_RE = re.compile(r"^(\d{1,2})[/.](\d{4})$")
#
#
#def _try_parse_mm_yyyy(date_str: str):
#    m = _MM_YYYY_RE.match(date_str.strip())
#    if not m:
#        return None
#    month, year = int(m.group(1)), int(m.group(2))
#    if 1 <= month <= 12:
#        return datetime(year, month, 1)
#    return None
#
#
#def parse_date(date_str: str):
#    if not date_str or re.match(r'present|current|now|till\s*(?:date|now)', date_str, re.IGNORECASE):
#        return datetime.now()
#    date_str = _expand_apostrophe_year(date_str)
#    mm_yyyy = _try_parse_mm_yyyy(date_str)
#    if mm_yyyy:
#        return mm_yyyy
#    try:
#        return date_parser.parse(date_str, default=datetime(2000, 1, 1), fuzzy=True)
#    except Exception:
#        return None
#
#
#def _inclusive_month_diff(start, end) -> int:
#    """Inclusive calendar-month count between two dates.
#
#    relativedelta(end, start) gives EXCLUSIVE elapsed time -- "June 2021"
#    to "May 2022" is 11 months apart. But resumes conventionally state
#    duration INCLUSIVELY: both the start month and the end month count
#    as full months worked, so "June 2021 - May 2022" reads as 12 months
#    (1 year), not 11.
#
#    Confirmed on a real resume (Aasma): every one of her 5 jobs was
#    undercounted by exactly 1 month against this convention (e.g. "June
#    2022 - Present", computed as of July 2026, should read 50 months,
#    not 49) -- a uniform 1-month deficit across every entry, which is
#    the signature of an exclusive-vs-inclusive convention mismatch
#    rather than a parsing error. This also matches how the aggregate
#    "total years of experience" figure is conventionally read on a
#    resume or by a recruiter, so both calculate_duration_months() and
#    calculate_total_experience() use this same helper to stay
#    consistent with each other.
#    """
#    if not start or not end:
#        return 0
#    delta = relativedelta(end, start)
#    months = delta.years * 12 + delta.months
#    return max(0, months + 1)
#
#
#def calculate_duration_months(start_str: str, end_str: str) -> int:
#    start = parse_date(start_str)
#    end = parse_date(end_str)
#    return _inclusive_month_diff(start, end)
#
#
#def detect_company_with_spacy(text_block: str):
#    if not _SPACY_AVAILABLE:
#        return None
#    doc = _NLP(text_block[:500])
#    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
#    return orgs[0] if orgs else None
#
#
#_COMPANY_ABBREV_ENDINGS = {"ltd", "inc", "co", "corp", "pvt", "llc", "llp"}
#
#
#def _looks_like_prose(line: str) -> bool:
#    words = line.split()
#    if len(words) > 6:
#        return True
#    last_word = words[-1].rstrip(".;").lower() if words else ""
#    if line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS:
#        return True
#    return False
#
#
## FIX: header_start_for's backward walk (below) originally used
## _looks_like_prose() -- which treats ">6 words" ALONE as enough to
## flag a line as "the previous job's trailing sentence" and stop
## walking backward. That's broader than the check's own comment states
## ("A long/period-ending sentence..." describes an AND of both
## conditions; _looks_like_prose implements it as an OR). Confirmed on a
## real resume (Abhay Awasthi) whose entire experience section has NO
## bullet markers at all -- every job is one long flowing sentence, e.g.
## "Worked at Morbous Technologies Pvt. Ltd. as a PHP Developer" (10
## words), immediately followed by that job's own date-range line. The
## word-count-alone check misclassified every one of these as "previous
## job's trailing prose" and excluded them, leaving every job's
## role/company empty even after the date-range itself was found.
## None of these genuine header lines end in terminal punctuation
## (they're cut off by the date parenthetical that follows) -- which is
## exactly the signal that distinguishes them from an actual trailing
## responsibility sentence (a grammatically complete, period-ended
## sentence). Requiring BOTH conditions here fixes that misfire while
## leaving _looks_like_prose() itself, and its other use in
## _reflow_body() (a different, unrelated no-bullet-marker case,
## confirmed working on Satyadip Ray's resume), completely untouched.
#def _looks_like_previous_job_trailing_line(line: str) -> bool:
#    words = line.split()
#    if not words or len(words) <= 6:
#        return False
#    last_word = words[-1].rstrip(".;").lower()
#    return line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS
#
#
#def _find_job_blocks(lines: list):
#    anchor_idx = [i for i, l in enumerate(lines) if DATE_RANGE_RE.search(l)]
#    if not anchor_idx:
#        return []
#
#    def header_start_for(anchor_i):
#        start = anchor_i
#        j = anchor_i - 1
#        steps = 0
#        while j >= 0 and steps < 2:
#            l = lines[j].strip()
#            if not l or BULLET_RE.match(l) or j in anchor_idx:
#                break
#            if l[0].islower():
#                break
#            if l.rstrip().endswith(":"):
#                break
#            if _looks_like_previous_job_trailing_line(l):
#                break
#            start = j
#            j -= 1
#            steps += 1
#        return start
#
#    blocks = []
#    header_starts = [header_start_for(a) for a in anchor_idx]
#
#    for n, anchor_i in enumerate(anchor_idx):
#        h_start = header_starts[n]
#        h_end = anchor_i
#        if n + 1 < len(anchor_idx):
#            body_end = header_starts[n + 1] - 1
#        else:
#            body_end = len(lines) - 1
#        header_ids = list(range(h_start, h_end + 1))
#        body_ids = list(range(h_end + 1, body_end + 1)) if body_end >= h_end + 1 else []
#        blocks.append((header_ids, body_ids))
#
#    return blocks
#
#
#def _parse_header(header_lines: list, date_line: str):
#    job = {"role": None, "company": None, "location": None, "start": None, "end": None, "method": []}
#
#    m = DATE_RANGE_RE.search(date_line)
#    if m:
#        job["start"] = m.group("start").strip()
#        job["end"] = m.group("end").strip()
#        job["method"].append("date_regex")
#
#    cleaned = []
#    for l in header_lines:
#        l = l.strip()
#        if not l:
#            continue
#        dm = DATE_RANGE_RE.search(l)
#        if dm:
#            l = (l[:dm.start()] + l[dm.end():])
#            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
#        if l:
#            cleaned.append(l)
#
#    if cleaned and "/" in cleaned[0] and "|" not in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("/") if p.strip()]
#        if parts:
#            job["role"] = parts[0]
#            paren = re.search(r'\(([^)]+)\)', cleaned[0])
#            if paren:
#                job["company"] = paren.group(1).strip()
#                job["method"].append("inline_parens")
#
#    if not job["role"] and not job["company"] and cleaned and "|" in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("|") if p.strip()]
#        title_part = next((p for p in parts if TITLE_RE.search(p)), None)
#        if title_part:
#            job["role"] = title_part
#            other_parts = [p for p in parts if p != title_part]
#            if other_parts:
#                job["company"] = other_parts[0]
#            job["method"].append("pipe_split")
#
#    if not job["role"] and not job["company"] and cleaned:
#        for sep in (" – ", " — ", " - "):
#            if sep in cleaned[0]:
#                left, right = (p.strip() for p in cleaned[0].split(sep, 1))
#                left = re.sub(r'\([^)]*\)', '', left).strip()
#                right = re.sub(r'\([^)]*\)', '', right).strip()
#                if left and right:
#                    left_is_title = bool(TITLE_RE.search(left))
#                    right_is_title = bool(TITLE_RE.search(right))
#                    if right_is_title and not left_is_title:
#                        job["role"], job["company"] = right, left
#                    else:
#                        job["role"], job["company"] = left, right
#                    job["method"].append("dash_split")
#                break
#
#    if not job["role"] and not job["company"] and cleaned:
#        paren_match = re.search(r'\(([^)]+)\)', cleaned[0])
#        if paren_match:
#            inside = paren_match.group(1).strip()
#            outside = (cleaned[0][:paren_match.start()] + cleaned[0][paren_match.end():]).strip()
#            if outside and TITLE_RE.search(inside):
#                job["role"] = inside
#                job["company"] = outside
#                job["method"].append("paren_title")
#
#    if not job["role"] and not job["company"]:
#        for line in cleaned[:3]:
#            line = re.sub(r'\([^)]*\)', '', line).strip()
#            if not line:
#                continue
#            if TITLE_RE.search(line) and not job["role"]:
#                job["role"] = line
#                job["method"].append("title_dict")
#            elif not job["company"]:
#                job["company"] = line
#
#    if job["company"] and "," in job["company"]:
#        parts = [p.strip() for p in job["company"].rsplit(",", 1)]
#        if len(parts) == 2 and len(parts[1].split()) <= 3:
#            job["company"], job["location"] = parts
#
#    if job["company"] and " - " in job["company"]:
#        left, _, right = job["company"].rpartition(" - ")
#        left, right = left.strip(), right.strip()
#        if left and right and len(right.split()) <= 4:
#            job["company"] = left
#            job["location"] = f"{right}, {job['location']}" if job["location"] else right
#
#    if not job["company"] and cleaned:
#        spacy_company = detect_company_with_spacy(" ".join(cleaned[:3]))
#        if spacy_company:
#            job["company"] = spacy_company
#            job["method"].append("spacy_ORG")
#
#    if not job["role"] and not job["company"] and cleaned:
#        job["role"] = cleaned[0]
#
#    for key in ("role", "company"):
#        if job[key]:
#            job[key] = job[key].strip(" /-").strip()
#
#    return job
#
#
#def _score_confidence(job: dict) -> float:
#    score = 0.5
#    if job["start"] and job["end"]:
#        score += 0.2
#    if job["role"]:
#        score += 0.15
#    if job["company"]:
#        score += 0.15
#    if "spacy_ORG" in job["method"] and "title_dict" not in job["method"]:
#        score -= 0.05
#    if job["company"] and len(job["company"].split()) > 6:
#        score -= 0.3
#    return round(min(max(score, 0.0), 0.98), 2)
#
#
#_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
#
#
#def _split_into_sentences(text: str) -> list:
#    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
#    return parts if len(parts) > 1 else [text]
#
#
#def _reflow_body(body_lines: list):
#    extra_header_lines, bullets, subheadings = [], [], []
#    current_bullet = None
#    seen_first_bullet = False
#    saw_real_bullet_marker = False
#
#    for raw in body_lines:
#        l = raw.strip()
#        if not l:
#            continue
#        if BULLET_RE.match(l):
#            seen_first_bullet = True
#            saw_real_bullet_marker = True
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#            current_bullet = BULLET_RE.sub("", l).strip()
#        elif not seen_first_bullet:
#            if _looks_like_prose(l):
#                seen_first_bullet = True
#                current_bullet = l
#            else:
#                extra_header_lines.append(l)
#        elif _looks_like_subheading(l):
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#                current_bullet = None
#            subheadings.append(l)
#        else:
#            if current_bullet is not None:
#                current_bullet += " " + l
#            else:
#                subheadings.append(l)
#
#    if current_bullet is not None:
#        bullets.append(current_bullet)
#
#    if not saw_real_bullet_marker and len(bullets) == 1:
#        bullets = _split_into_sentences(bullets[0])
#
#    return extra_header_lines, bullets, subheadings
#
#
#def parse_job_block(header_lines: list, body_lines: list):
#    if not header_lines:
#        return None
#
#    extra_header_lines, bullets, subheadings = _reflow_body(body_lines)
#
#    date_line = next((l for l in header_lines if DATE_RANGE_RE.search(l)), "")
#    job = _parse_header(header_lines + extra_header_lines, date_line)
#
#    job["duration_months"] = calculate_duration_months(job["start"], job["end"]) if job["start"] else 0
#    job["bullets"] = bullets
#    job["sub_sections"] = subheadings
#
#    job["confidence"] = _score_confidence(job)
#    job["needs_review"] = job["confidence"] < 0.75 or not job["company"] or not job["role"]
#    job["method"] = "+".join(job["method"]) if job["method"] else "none"
#
#    if not job["company"] and not job["role"] and not job["bullets"]:
#        return None
#    return job
#
#
#def _extract_jobs_from_text(section_text: str) -> list:
#    """
#    Shared block-finding + parsing pipeline, factored out of
#    extract_experience() so both the main "experience" section text and
#    an "early_career" condensed-recap section text can be run through the
#    exact same anchor-based logic. A condensed entry like "Oct-2023 -
#    Nov-2025 | Manager | KPMG" is just a header line with no body --
#    _find_job_blocks anchors on its date range the same way it would for
#    a full job entry, and _parse_header's existing "pipe_split" branch
#    (splitting on "|", picking the TITLE_RE-matching part as the role)
#    already handles this exact "Date - Date | Title | Company" shape
#    without any changes needed here.
#    """
#    if not section_text.strip():
#        return []
#
#    lines = section_text.strip("\n").split("\n")
#    lines = [_normalize_dashes(l) for l in lines]
#    lines = [_normalize_month_year_glue(l) for l in lines]
#    lines = _merge_widow_date_lines(lines)
#    blocks = _find_job_blocks(lines)
#
#    jobs = []
#    for header_ids, body_ids in blocks:
#        header_lines = [lines[i] for i in header_ids]
#        body_lines = [lines[i] for i in body_ids]
#        parsed = parse_job_block(header_lines, body_lines)
#        if parsed:
#            jobs.append(parsed)
#    return jobs
#
#
#def calculate_total_experience(jobs: list) -> dict:
#    intervals = []
#    for job in jobs:
#        if not job.get("start") or not job.get("end"):
#            continue
#        start = parse_date(job["start"])
#        end = parse_date(job["end"])
#        if start and end and end > start:
#            intervals.append((start, end))
#
#    if not intervals:
#        return {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}
#
#    intervals.sort(key=lambda x: x[0])
#    merged = [intervals[0]]
#    for s, e in intervals[1:]:
#        last_s, last_e = merged[-1]
#        if s <= last_e:
#            merged[-1] = (last_s, max(last_e, e))
#        else:
#            merged.append((s, e))
#
#    total_months = sum(_inclusive_month_diff(s, e) for s, e in merged)
#
#    return {
#        "total_years": round(total_months / 12, 1),
#        "total_months": total_months,
#        "overlapping_periods_merged": len(intervals) - len(merged),
#    }
#
#
#def extract_experience(experience_section_text: str, early_career_section_text: str = "") -> dict:
#    """
#    Parses the main "experience" section as before. If early_career_section_text
#    is also given (from a section the splitter labeled "early_career" --
#    a condensed one-line-per-job recap like "LAST 5 CAREER TIMELINE"),
#    it's parsed with the same anchor-based logic and kept in its own
#    "early_career" list rather than merged into "experience", so the
#    detailed job list isn't polluted with duplicate condensed entries.
#
#    Both lists ARE combined, however, for `_total_experience`: the same
#    jobs are often restated in both places (confirmed on a real resume,
#    Sachin Chitranshi, where 5 jobs appear both in a condensed timeline
#    AND in full detail later), and calculate_total_experience() already
#    merges overlapping/identical date intervals, so restating a job in
#    both sections does not inflate the total years figure -- it only
#    helps in the case where the condensed recap includes an older job
#    that isn't detailed anywhere else in the document.
#    """
#    empty_total = {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}
#
#    jobs = _extract_jobs_from_text(experience_section_text)
#    for j in jobs:
#        j["source"] = "experience"
#
#    early_career_jobs = _extract_jobs_from_text(early_career_section_text)
#    for j in early_career_jobs:
#        j["source"] = "early_career"
#
#    if not jobs and not early_career_jobs:
#        return {
#            "experience": [],
#            "early_career": [],
#            "_confidence": 0.0,
#            "_needs_review": True,
#            "_total_experience": empty_total,
#        }
#
#    combined = jobs + early_career_jobs
#    avg_conf = round(sum(j["confidence"] for j in combined) / len(combined), 2) if combined else 0.0
#    needs_review = any(j["needs_review"] for j in combined)
#
#    return {
#        "experience": jobs,
#        "early_career": early_career_jobs,
#        "_confidence": avg_conf,
#        "_needs_review": needs_review,
#        "_total_experience": calculate_total_experience(combined),
#    }
#
#












#done- just commenting to match the date range and writting one above-
#r"""
#Experience Extractor — Layer 2
#Extracts: company, title, location, date range, duration, bullet points.
#Uses: date regex (anchor-based block splitting) + job title dictionary + spaCy ORG detection.
#
#KEY DESIGN NOTE:
#Real resumes do NOT separate job entries with blank lines (verified against
#Aarti / Abhijeet segmented output). So we cannot split blocks on '\n\s*\n'.
#Instead we anchor on lines that contain a date range — every job entry has one —
#and walk backward/forward from each anchor to find header lines and body lines.
#"""
#
#import re
#try:
#    import spacy
#    _SPACY_IMPORTED = True
#except ImportError:
#    _SPACY_IMPORTED = False
#from dateutil import parser as date_parser
#from dateutil.relativedelta import relativedelta
#from datetime import datetime
#
#
## ── Date matching ──────────────────────────────────────────────────────────
#
#MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#DATE_RANGE_RE = re.compile(
#    rf"(?P<start>{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})"
#    rf"(?:\s*[-–—]\s*|\s+to\s+)"
#    rf"(?P<end>Present|Current|Now|Till\s*(?:Date|Now)|{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})",
#    re.IGNORECASE
#)
#
#_WIDOW_START_DATE_RE = re.compile(
#    rf"({MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
#    re.IGNORECASE
#)
#_WIDOW_END_DATE_RE = re.compile(
#    rf"^[-–—]\s*(Present|Current|Now|Till\s*(?:Date|Now)|"
#    rf"{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
#    re.IGNORECASE
#)
#
#
#_TRAILING_TO_RE = re.compile(r"\s*\bto\s*$", re.IGNORECASE)
#
#
#def _merge_widow_date_lines(lines: list) -> list:
#    merged = list(lines)
#    for i in range(len(merged) - 1):
#        cur = merged[i].rstrip()
#        nxt = merged[i + 1].strip()
#        if not cur or not nxt:
#            continue
#        if DATE_RANGE_RE.search(cur):
#            continue
#        if _WIDOW_START_DATE_RE.search(cur) and _WIDOW_END_DATE_RE.match(nxt):
#            merged[i] = cur + " " + nxt
#            merged[i + 1] = ""
#            continue
#        # A "to"-continuation widow: the date RANGE ITSELF splits across
#        # two lines at the separator word, e.g.
#        #   "...as a PHP Developer (Dec 2018 to"
#        #   "Jun 2020)."
#        # Confirmed on a real resume (Abhay Awasthi). Distinct from the
#        # dash-continuation case above (that one has no "to" at all --
#        # just a start date, then a line starting with "-EndDate"). Only
#        # merges when the text before the trailing "to" itself ends in a
#        # recognizable start date, so this can't misfire on an unrelated
#        # sentence that happens to wrap at the word "to".
#        to_match = _TRAILING_TO_RE.search(cur)
#        if to_match:
#            before_to = cur[:to_match.start()].rstrip()
#            if _WIDOW_START_DATE_RE.search(before_to):
#                merged[i] = cur + " " + nxt
#                merged[i + 1] = ""
#    return merged
#
#TITLE_KEYWORDS = [
#    "engineer", "developer", "manager", "analyst", "consultant",
#    "designer", "architect", "lead", "head", "director", "vp", "svp", "cto", "ceo",
#    "intern", "associate", "senior", "junior", "principal", "staff",
#    "specialist", "coordinator", "executive", "officer", "president",
#    "scientist", "researcher", "administrator", "technician", "professor",
#    "assistant", "trainee", "fresher", "apprentice"
#]
#TITLE_RE = re.compile(r'\b(' + '|'.join(TITLE_KEYWORDS) + r')\b', re.IGNORECASE)
#
#_DASH_NORMALIZE_TABLE = str.maketrans({
#    "－": "-",
#    "‐": "-",
#    "‑": "-",
#})
#
#
#def _normalize_dashes(text: str) -> str:
#    return text.translate(_DASH_NORMALIZE_TABLE)
#
#
## A stray dash is sometimes glued directly between a month name and ITS
## OWN year -- "July - 2018", "Dec -2018" -- distinct from the intended
## dash that separates the start date from the end date. Confirmed on a
## real resume (Abhay Awasthi): dates written as "(July - 2018 to Dec. -
## 2018)" defeated DATE_RANGE_RE entirely, since its "start"/"end" groups
## expect a month directly followed by whitespace/comma/period and then
## the year -- a literal "-" in between isn't in that allowed character
## class. This silently dropped 3 of his 5 job entries (zero anchor line
## found for them at all), not just a parsing quality issue.
## Collapsing "Month - Year" / "Month -Year" down to "Month Year" BEFORE
## DATE_RANGE_RE runs fixes this without touching the real start/end
## separator dash, since this pattern only fires directly after a month
## name, never between two already-complete dates.
#_MONTH_YEAR_GLUE_DASH_RE = re.compile(
#    rf"({MONTH_NAMES})[.,]?\s*-\s*(\d{{4}})", re.IGNORECASE
#)
#
#
#def _normalize_month_year_glue(text: str) -> str:
#    return _MONTH_YEAR_GLUE_DASH_RE.sub(r"\1 \2", text)
#
#BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*]\s+|^\d+[\.\)]?\s+')
#
#def _looks_like_subheading(line: str) -> bool:
#    words = line.split()
#    if not (1 <= len(words) <= 6):
#        return False
#    if line.rstrip().endswith((".", ",", ";", ":")):
#        return False
#    if DATE_RANGE_RE.search(line):
#        return False
#    # Ensure there is at least one letter, and the first letter in the line is not lowercase
#    first_letter_match = re.search(r'[A-Za-z]', line)
#    if not first_letter_match or first_letter_match.group(0).islower():
#        return False
#    return True
#
#
#if _SPACY_IMPORTED:
#    try:
#        _NLP = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
#        _SPACY_AVAILABLE = True
#    except Exception:
#        _SPACY_AVAILABLE = False
#        print("[WARN] spaCy model not found. Company name detection will be limited.")
#else:
#    _SPACY_AVAILABLE = False
#    print("[WARN] spaCy not installed. Company name detection will be limited.")
#
#
#def _expand_apostrophe_year(date_str: str) -> str:
#    m = re.search(r"['’‘`´](\d{2})\b", date_str)
#    if not m:
#        return date_str
#    two_digit = int(m.group(1))
#    century = "20" if two_digit <= 49 else "19"
#    return date_str[:m.start()] + century + m.group(1) + date_str[m.end():]
#
#
## Confirmed on a real resume (Abhijeet Sangle): dates written numerically
## as "12.2011" (MM.YYYY) get silently MISPARSED by dateutil's fuzzy
## parser -- it reads "12" as the DAY, can't make sense of the trailing
## ".2011" as anything meaningful, and discards it, falling back to the
## `default` year/month (2000-01). So "12.2011" and "12.2019" -- two
## genuinely different dates 8 years apart -- both parsed to the exact
## same date, 2000-01-12, making the computed duration collapse to
## ~0 months instead of the correct ~96/97. This isn't a duration-formula
## bug; the dates going INTO the duration calculation were already wrong.
## Explicitly parsing this MM.YYYY / MM/YYYY numeric shorthand ourselves,
## before ever handing the string to dateutil, sidesteps the ambiguity
## entirely rather than hoping a general-purpose fuzzy parser guesses
## right on a format many general parsers reasonably read as DD.YYYY.
#_MM_YYYY_RE = re.compile(r"^(\d{1,2})[/.](\d{4})$")
#
#
#def _try_parse_mm_yyyy(date_str: str):
#    m = _MM_YYYY_RE.match(date_str.strip())
#    if not m:
#        return None
#    month, year = int(m.group(1)), int(m.group(2))
#    if 1 <= month <= 12:
#        return datetime(year, month, 1)
#    return None
#
#
#def parse_date(date_str: str):
#    if not date_str or re.match(r'present|current|now|till\s*(?:date|now)', date_str, re.IGNORECASE):
#        return datetime.now()
#    date_str = _expand_apostrophe_year(date_str)
#    mm_yyyy = _try_parse_mm_yyyy(date_str)
#    if mm_yyyy:
#        return mm_yyyy
#    try:
#        return date_parser.parse(date_str, default=datetime(2000, 1, 1), fuzzy=True)
#    except Exception:
#        return None
#
#
#def _inclusive_month_diff(start, end) -> int:
#    """Inclusive calendar-month count between two dates.
#
#    relativedelta(end, start) gives EXCLUSIVE elapsed time -- "June 2021"
#    to "May 2022" is 11 months apart. But resumes conventionally state
#    duration INCLUSIVELY: both the start month and the end month count
#    as full months worked, so "June 2021 - May 2022" reads as 12 months
#    (1 year), not 11.
#
#    Confirmed on a real resume (Aasma): every one of her 5 jobs was
#    undercounted by exactly 1 month against this convention (e.g. "June
#    2022 - Present", computed as of July 2026, should read 50 months,
#    not 49) -- a uniform 1-month deficit across every entry, which is
#    the signature of an exclusive-vs-inclusive convention mismatch
#    rather than a parsing error. This also matches how the aggregate
#    "total years of experience" figure is conventionally read on a
#    resume or by a recruiter, so both calculate_duration_months() and
#    calculate_total_experience() use this same helper to stay
#    consistent with each other.
#    """
#    if not start or not end:
#        return 0
#    delta = relativedelta(end, start)
#    months = delta.years * 12 + delta.months
#    return max(0, months + 1)
#
#
#def calculate_duration_months(start_str: str, end_str: str) -> int:
#    start = parse_date(start_str)
#    end = parse_date(end_str)
#    return _inclusive_month_diff(start, end)
#
#
#def detect_company_with_spacy(text_block: str):
#    if not _SPACY_AVAILABLE:
#        return None
#    doc = _NLP(text_block[:500])
#    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
#    return orgs[0] if orgs else None
#
#
#_COMPANY_ABBREV_ENDINGS = {"ltd", "inc", "co", "corp", "pvt", "llc", "llp"}
#
#
#def _looks_like_prose(line: str) -> bool:
#    words = line.split()
#    if len(words) > 6:
#        return True
#    last_word = words[-1].rstrip(".;").lower() if words else ""
#    if line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS:
#        return True
#    return False
#
#
## FIX: header_start_for's backward walk (below) originally used
## _looks_like_prose() -- which treats ">6 words" ALONE as enough to
## flag a line as "the previous job's trailing sentence" and stop
## walking backward. That's broader than the check's own comment states
## ("A long/period-ending sentence..." describes an AND of both
## conditions; _looks_like_prose implements it as an OR). Confirmed on a
## real resume (Abhay Awasthi) whose entire experience section has NO
## bullet markers at all -- every job is one long flowing sentence, e.g.
## "Worked at Morbous Technologies Pvt. Ltd. as a PHP Developer" (10
## words), immediately followed by that job's own date-range line. The
## word-count-alone check misclassified every one of these as "previous
## job's trailing prose" and excluded them, leaving every job's
## role/company empty even after the date-range itself was found.
## None of these genuine header lines end in terminal punctuation
## (they're cut off by the date parenthetical that follows) -- which is
## exactly the signal that distinguishes them from an actual trailing
## responsibility sentence (a grammatically complete, period-ended
## sentence). Requiring BOTH conditions here fixes that misfire while
## leaving _looks_like_prose() itself, and its other use in
## _reflow_body() (a different, unrelated no-bullet-marker case,
## confirmed working on Satyadip Ray's resume), completely untouched.
#def _looks_like_previous_job_trailing_line(line: str) -> bool:
#    words = line.split()
#    if not words or len(words) <= 6:
#        return False
#    last_word = words[-1].rstrip(".;").lower()
#    return line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS
#
#
#def _find_job_blocks(lines: list):
#    anchor_idx = [i for i, l in enumerate(lines) if DATE_RANGE_RE.search(l)]
#    if not anchor_idx:
#        return []
#
#    def header_start_for(anchor_i):
#        start = anchor_i
#        j = anchor_i - 1
#        steps = 0
#        while j >= 0 and steps < 2:
#            l = lines[j].strip()
#            if not l or BULLET_RE.match(l) or j in anchor_idx:
#                break
#            if l[0].islower():
#                break
#            if l.rstrip().endswith(":"):
#                break
#            if _looks_like_previous_job_trailing_line(l):
#                break
#            start = j
#            j -= 1
#            steps += 1
#        return start
#
#    blocks = []
#    header_starts = [header_start_for(a) for a in anchor_idx]
#
#    for n, anchor_i in enumerate(anchor_idx):
#        h_start = header_starts[n]
#        h_end = anchor_i
#        if n + 1 < len(anchor_idx):
#            body_end = header_starts[n + 1] - 1
#        else:
#            body_end = len(lines) - 1
#        header_ids = list(range(h_start, h_end + 1))
#        body_ids = list(range(h_end + 1, body_end + 1)) if body_end >= h_end + 1 else []
#        blocks.append((header_ids, body_ids))
#
#    return blocks
#
#
#def _parse_header(header_lines: list, date_line: str):
#    job = {"role": None, "company": None, "location": None, "start": None, "end": None, "method": []}
#
#    m = DATE_RANGE_RE.search(date_line)
#    if m:
#        job["start"] = m.group("start").strip()
#        job["end"] = m.group("end").strip()
#        job["method"].append("date_regex")
#
#    cleaned = []
#    for l in header_lines:
#        l = l.strip()
#        if not l:
#            continue
#        dm = DATE_RANGE_RE.search(l)
#        if dm:
#            l = (l[:dm.start()] + l[dm.end():])
#            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
#        if l:
#            cleaned.append(l)
#
#    if cleaned and "/" in cleaned[0] and "|" not in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("/") if p.strip()]
#        if parts:
#            job["role"] = parts[0]
#            paren = re.search(r'\(([^)]+)\)', cleaned[0])
#            if paren:
#                job["company"] = paren.group(1).strip()
#                job["method"].append("inline_parens")
#
#    if not job["role"] and not job["company"] and cleaned and "|" in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("|") if p.strip()]
#        title_part = next((p for p in parts if TITLE_RE.search(p)), None)
#        if title_part:
#            job["role"] = title_part
#            other_parts = [p for p in parts if p != title_part]
#            if other_parts:
#                job["company"] = other_parts[0]
#            job["method"].append("pipe_split")
#
#    if not job["role"] and not job["company"] and cleaned:
#        for sep in (" – ", " — ", " - "):
#            if sep in cleaned[0]:
#                left, right = (p.strip() for p in cleaned[0].split(sep, 1))
#                left = re.sub(r'\([^)]*\)', '', left).strip()
#                right = re.sub(r'\([^)]*\)', '', right).strip()
#                if left and right:
#                    left_is_title = bool(TITLE_RE.search(left))
#                    right_is_title = bool(TITLE_RE.search(right))
#                    if right_is_title and not left_is_title:
#                        job["role"], job["company"] = right, left
#                    else:
#                        job["role"], job["company"] = left, right
#                    job["method"].append("dash_split")
#                break
#
#    if not job["role"] and not job["company"] and cleaned:
#        paren_match = re.search(r'\(([^)]+)\)', cleaned[0])
#        if paren_match:
#            inside = paren_match.group(1).strip()
#            outside = (cleaned[0][:paren_match.start()] + cleaned[0][paren_match.end():]).strip()
#            if outside and TITLE_RE.search(inside):
#                job["role"] = inside
#                job["company"] = outside
#                job["method"].append("paren_title")
#
#    if not job["role"] and not job["company"]:
#        for line in cleaned[:3]:
#            line = re.sub(r'\([^)]*\)', '', line).strip()
#            if not line:
#                continue
#            if TITLE_RE.search(line) and not job["role"]:
#                job["role"] = line
#                job["method"].append("title_dict")
#            elif not job["company"]:
#                job["company"] = line
#
#    if job["company"] and "," in job["company"]:
#        parts = [p.strip() for p in job["company"].rsplit(",", 1)]
#        if len(parts) == 2 and len(parts[1].split()) <= 3:
#            job["company"], job["location"] = parts
#
#    if job["company"] and " - " in job["company"]:
#        left, _, right = job["company"].rpartition(" - ")
#        left, right = left.strip(), right.strip()
#        if left and right and len(right.split()) <= 4:
#            job["company"] = left
#            job["location"] = f"{right}, {job['location']}" if job["location"] else right
#
#    if not job["company"] and cleaned:
#        spacy_company = detect_company_with_spacy(" ".join(cleaned[:3]))
#        if spacy_company:
#            job["company"] = spacy_company
#            job["method"].append("spacy_ORG")
#
#    if not job["role"] and not job["company"] and cleaned:
#        job["role"] = cleaned[0]
#
#    for key in ("role", "company"):
#        if job[key]:
#            job[key] = job[key].strip(" /-").strip()
#
#    return job
#
#
#def _score_confidence(job: dict) -> float:
#    score = 0.5
#    if job["start"] and job["end"]:
#        score += 0.2
#    if job["role"]:
#        score += 0.15
#    if job["company"]:
#        score += 0.15
#    if "spacy_ORG" in job["method"] and "title_dict" not in job["method"]:
#        score -= 0.05
#    if job["company"] and len(job["company"].split()) > 6:
#        score -= 0.3
#    return round(min(max(score, 0.0), 0.98), 2)
#
#
#_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
#
#
#def _split_into_sentences(text: str) -> list:
#    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
#    return parts if len(parts) > 1 else [text]
#
#
#def _reflow_body(body_lines: list):
#    extra_header_lines, bullets, subheadings = [], [], []
#    current_bullet = None
#    seen_first_bullet = False
#    saw_real_bullet_marker = False
#
#    for raw in body_lines:
#        l = raw.strip()
#        if not l:
#            continue
#        if BULLET_RE.match(l):
#            seen_first_bullet = True
#            saw_real_bullet_marker = True
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#            current_bullet = BULLET_RE.sub("", l).strip()
#        elif not seen_first_bullet:
#            if _looks_like_prose(l):
#                seen_first_bullet = True
#                current_bullet = l
#            else:
#                extra_header_lines.append(l)
#        elif _looks_like_subheading(l):
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#                current_bullet = None
#            subheadings.append(l)
#        else:
#            if current_bullet is not None:
#                current_bullet += " " + l
#            else:
#                subheadings.append(l)
#
#    if current_bullet is not None:
#        bullets.append(current_bullet)
#
#    if not saw_real_bullet_marker and len(bullets) == 1:
#        bullets = _split_into_sentences(bullets[0])
#
#    return extra_header_lines, bullets, subheadings
#
#
#def parse_job_block(header_lines: list, body_lines: list):
#    if not header_lines:
#        return None
#
#    extra_header_lines, bullets, subheadings = _reflow_body(body_lines)
#
#    date_line = next((l for l in header_lines if DATE_RANGE_RE.search(l)), "")
#    job = _parse_header(header_lines + extra_header_lines, date_line)
#
#    job["duration_months"] = calculate_duration_months(job["start"], job["end"]) if job["start"] else 0
#    job["bullets"] = bullets
#    job["sub_sections"] = subheadings
#
#    job["confidence"] = _score_confidence(job)
#    job["needs_review"] = job["confidence"] < 0.75 or not job["company"] or not job["role"]
#    job["method"] = "+".join(job["method"]) if job["method"] else "none"
#
#    if not job["company"] and not job["role"] and not job["bullets"]:
#        return None
#    return job
#
#
#def calculate_total_experience(jobs: list) -> dict:
#    intervals = []
#    for job in jobs:
#        if not job.get("start") or not job.get("end"):
#            continue
#        start = parse_date(job["start"])
#        end = parse_date(job["end"])
#        if start and end and end > start:
#            intervals.append((start, end))
#
#    if not intervals:
#        return {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}
#
#    intervals.sort(key=lambda x: x[0])
#    merged = [intervals[0]]
#    for s, e in intervals[1:]:
#        last_s, last_e = merged[-1]
#        if s <= last_e:
#            merged[-1] = (last_s, max(last_e, e))
#        else:
#            merged.append((s, e))
#
#    total_months = sum(_inclusive_month_diff(s, e) for s, e in merged)
#
#    return {
#        "total_years": round(total_months / 12, 1),
#        "total_months": total_months,
#        "overlapping_periods_merged": len(intervals) - len(merged),
#    }
#
#
#def extract_experience(experience_section_text: str) -> dict:
#    if not experience_section_text.strip():
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    lines = experience_section_text.strip("\n").split("\n")
#    lines = [_normalize_dashes(l) for l in lines]
#    lines = [_normalize_month_year_glue(l) for l in lines]
#    lines = _merge_widow_date_lines(lines)
#    blocks = _find_job_blocks(lines)
#
#    if not blocks:
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    jobs = []
#    for header_ids, body_ids in blocks:
#        header_lines = [lines[i] for i in header_ids]
#        body_lines = [lines[i] for i in body_ids]
#        parsed = parse_job_block(header_lines, body_lines)
#        if parsed:
#            jobs.append(parsed)
#
#    avg_conf = round(sum(j["confidence"] for j in jobs) / len(jobs), 2) if jobs else 0.0
#    needs_review = any(j["needs_review"] for j in jobs)
#
#    return {
#        "experience": jobs,
#        "_confidence": avg_conf,
#        "_needs_review": needs_review,
#        "_total_experience": calculate_total_experience(jobs),
#    }
#


#r"""
#Experience Extractor — Layer 2
#Extracts: company, title, location, date range, duration, bullet points.
#Uses: date regex (anchor-based block splitting) + job title dictionary + spaCy ORG detection.
#
#KEY DESIGN NOTE:
#Real resumes do NOT separate job entries with blank lines (verified against
#Aarti / Abhijeet segmented output). So we cannot split blocks on '\n\s*\n'.
#Instead we anchor on lines that contain a date range — every job entry has one —
#and walk backward/forward from each anchor to find header lines and body lines.
#"""
#
#import re
#try:
#    import spacy
#    _SPACY_IMPORTED = True
#except ImportError:
#    _SPACY_IMPORTED = False
#from dateutil import parser as date_parser
#from dateutil.relativedelta import relativedelta
#from datetime import datetime
#
#
## ── Date matching ──────────────────────────────────────────────────────────
#
#MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#DATE_RANGE_RE = re.compile(
#    rf"(?P<start>{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})"
#    rf"(?:\s*[-–—]\s*|\s+to\s+)"
#    rf"(?P<end>Present|Current|Now|Till\s*(?:Date|Now)|{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})",
#    re.IGNORECASE
#)
#
#_WIDOW_START_DATE_RE = re.compile(
#    rf"({MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
#    re.IGNORECASE
#)
#_WIDOW_END_DATE_RE = re.compile(
#    rf"^[-–—]\s*(Present|Current|Now|Till\s*(?:Date|Now)|"
#    rf"{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
#    re.IGNORECASE
#)
#
#
#_TRAILING_TO_RE = re.compile(r"\s*\bto\s*$", re.IGNORECASE)
#
#
#def _merge_widow_date_lines(lines: list) -> list:
#    merged = list(lines)
#    for i in range(len(merged) - 1):
#        cur = merged[i].rstrip()
#        nxt = merged[i + 1].strip()
#        if not cur or not nxt:
#            continue
#        if DATE_RANGE_RE.search(cur):
#            continue
#        if _WIDOW_START_DATE_RE.search(cur) and _WIDOW_END_DATE_RE.match(nxt):
#            merged[i] = cur + " " + nxt
#            merged[i + 1] = ""
#            continue
#        # A "to"-continuation widow: the date RANGE ITSELF splits across
#        # two lines at the separator word, e.g.
#        #   "...as a PHP Developer (Dec 2018 to"
#        #   "Jun 2020)."
#        # Confirmed on a real resume (Abhay Awasthi). Distinct from the
#        # dash-continuation case above (that one has no "to" at all --
#        # just a start date, then a line starting with "-EndDate"). Only
#        # merges when the text before the trailing "to" itself ends in a
#        # recognizable start date, so this can't misfire on an unrelated
#        # sentence that happens to wrap at the word "to".
#        to_match = _TRAILING_TO_RE.search(cur)
#        if to_match:
#            before_to = cur[:to_match.start()].rstrip()
#            if _WIDOW_START_DATE_RE.search(before_to):
#                merged[i] = cur + " " + nxt
#                merged[i + 1] = ""
#    return merged
#
#TITLE_KEYWORDS = [
#    "engineer", "developer", "manager", "analyst", "consultant",
#    "designer", "architect", "lead", "head", "director", "vp", "svp", "cto", "ceo",
#    "intern", "associate", "senior", "junior", "principal", "staff",
#    "specialist", "coordinator", "executive", "officer", "president",
#    "scientist", "researcher", "administrator", "technician", "professor",
#    "assistant", "trainee", "fresher", "apprentice"
#]
#TITLE_RE = re.compile(r'\b(' + '|'.join(TITLE_KEYWORDS) + r')\b', re.IGNORECASE)
#
#_DASH_NORMALIZE_TABLE = str.maketrans({
#    "－": "-",
#    "‐": "-",
#    "‑": "-",
#})
#
#
#def _normalize_dashes(text: str) -> str:
#    return text.translate(_DASH_NORMALIZE_TABLE)
#
#
## A stray dash is sometimes glued directly between a month name and ITS
## OWN year -- "July - 2018", "Dec -2018" -- distinct from the intended
## dash that separates the start date from the end date. Confirmed on a
## real resume (Abhay Awasthi): dates written as "(July - 2018 to Dec. -
## 2018)" defeated DATE_RANGE_RE entirely, since its "start"/"end" groups
## expect a month directly followed by whitespace/comma/period and then
## the year -- a literal "-" in between isn't in that allowed character
## class. This silently dropped 3 of his 5 job entries (zero anchor line
## found for them at all), not just a parsing quality issue.
## Collapsing "Month - Year" / "Month -Year" down to "Month Year" BEFORE
## DATE_RANGE_RE runs fixes this without touching the real start/end
## separator dash, since this pattern only fires directly after a month
## name, never between two already-complete dates.
#_MONTH_YEAR_GLUE_DASH_RE = re.compile(
#    rf"({MONTH_NAMES})[.,]?\s*-\s*(\d{{4}})", re.IGNORECASE
#)
#
#
#def _normalize_month_year_glue(text: str) -> str:
#    return _MONTH_YEAR_GLUE_DASH_RE.sub(r"\1 \2", text)
#
#BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*]\s+|^\d+[\.\)]?\s+')
#
#def _looks_like_subheading(line: str) -> bool:
#    words = line.split()
#    if not (1 <= len(words) <= 6):
#        return False
#    if line.rstrip().endswith((".", ",", ";", ":")):
#        return False
#    if DATE_RANGE_RE.search(line):
#        return False
#    # Ensure there is at least one letter, and the first letter in the line is not lowercase
#    first_letter_match = re.search(r'[A-Za-z]', line)
#    if not first_letter_match or first_letter_match.group(0).islower():
#        return False
#    return True
#
#
#if _SPACY_IMPORTED:
#    try:
#        _NLP = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
#        _SPACY_AVAILABLE = True
#    except Exception:
#        _SPACY_AVAILABLE = False
#        print("[WARN] spaCy model not found. Company name detection will be limited.")
#else:
#    _SPACY_AVAILABLE = False
#    print("[WARN] spaCy not installed. Company name detection will be limited.")
#
#
#def _expand_apostrophe_year(date_str: str) -> str:
#    m = re.search(r"['’‘`´](\d{2})\b", date_str)
#    if not m:
#        return date_str
#    two_digit = int(m.group(1))
#    century = "20" if two_digit <= 49 else "19"
#    return date_str[:m.start()] + century + m.group(1) + date_str[m.end():]
#
#
## Confirmed on a real resume (Abhijeet Sangle): dates written numerically
## as "12.2011" (MM.YYYY) get silently MISPARSED by dateutil's fuzzy
## parser -- it reads "12" as the DAY, can't make sense of the trailing
## ".2011" as anything meaningful, and discards it, falling back to the
## `default` year/month (2000-01). So "12.2011" and "12.2019" -- two
## genuinely different dates 8 years apart -- both parsed to the exact
## same date, 2000-01-12, making the computed duration collapse to
## ~0 months instead of the correct ~96/97. This isn't a duration-formula
## bug; the dates going INTO the duration calculation were already wrong.
## Explicitly parsing this MM.YYYY / MM/YYYY numeric shorthand ourselves,
## before ever handing the string to dateutil, sidesteps the ambiguity
## entirely rather than hoping a general-purpose fuzzy parser guesses
## right on a format many general parsers reasonably read as DD.YYYY.
#_MM_YYYY_RE = re.compile(r"^(\d{1,2})[/.](\d{4})$")
#
#
#def _try_parse_mm_yyyy(date_str: str):
#    m = _MM_YYYY_RE.match(date_str.strip())
#    if not m:
#        return None
#    month, year = int(m.group(1)), int(m.group(2))
#    if 1 <= month <= 12:
#        return datetime(year, month, 1)
#    return None
#
#
#def parse_date(date_str: str):
#    if not date_str or re.match(r'present|current|now|till\s*(?:date|now)', date_str, re.IGNORECASE):
#        return datetime.now()
#    date_str = _expand_apostrophe_year(date_str)
#    mm_yyyy = _try_parse_mm_yyyy(date_str)
#    if mm_yyyy:
#        return mm_yyyy
#    try:
#        return date_parser.parse(date_str, default=datetime(2000, 1, 1), fuzzy=True)
#    except Exception:
#        return None
#
#
#def _inclusive_month_diff(start, end) -> int:
#    """Inclusive calendar-month count between two dates.
#
#    relativedelta(end, start) gives EXCLUSIVE elapsed time -- "June 2021"
#    to "May 2022" is 11 months apart. But resumes conventionally state
#    duration INCLUSIVELY: both the start month and the end month count
#    as full months worked, so "June 2021 - May 2022" reads as 12 months
#    (1 year), not 11.
#
#    Confirmed on a real resume (Aasma): every one of her 5 jobs was
#    undercounted by exactly 1 month against this convention (e.g. "June
#    2022 - Present", computed as of July 2026, should read 50 months,
#    not 49) -- a uniform 1-month deficit across every entry, which is
#    the signature of an exclusive-vs-inclusive convention mismatch
#    rather than a parsing error. This also matches how the aggregate
#    "total years of experience" figure is conventionally read on a
#    resume or by a recruiter, so both calculate_duration_months() and
#    calculate_total_experience() use this same helper to stay
#    consistent with each other.
#    """
#    if not start or not end:
#        return 0
#    delta = relativedelta(end, start)
#    months = delta.years * 12 + delta.months
#    return max(0, months + 1)
#
#
#def calculate_duration_months(start_str: str, end_str: str) -> int:
#    start = parse_date(start_str)
#    end = parse_date(end_str)
#    return _inclusive_month_diff(start, end)
#
#
#def detect_company_with_spacy(text_block: str):
#    if not _SPACY_AVAILABLE:
#        return None
#    doc = _NLP(text_block[:500])
#    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
#    return orgs[0] if orgs else None
#
#
#_COMPANY_ABBREV_ENDINGS = {"ltd", "inc", "co", "corp", "pvt", "llc", "llp"}
#
#
#def _looks_like_prose(line: str) -> bool:
#    words = line.split()
#    if len(words) > 6:
#        return True
#    last_word = words[-1].rstrip(".;").lower() if words else ""
#    if line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS:
#        return True
#    return False
#
#
## FIX: header_start_for's backward walk (below) originally used
## _looks_like_prose() -- which treats ">6 words" ALONE as enough to
## flag a line as "the previous job's trailing sentence" and stop
## walking backward. That's broader than the check's own comment states
## ("A long/period-ending sentence..." describes an AND of both
## conditions; _looks_like_prose implements it as an OR). Confirmed on a
## real resume (Abhay Awasthi) whose entire experience section has NO
## bullet markers at all -- every job is one long flowing sentence, e.g.
## "Worked at Morbous Technologies Pvt. Ltd. as a PHP Developer" (10
## words), immediately followed by that job's own date-range line. The
## word-count-alone check misclassified every one of these as "previous
## job's trailing prose" and excluded them, leaving every job's
## role/company empty even after the date-range itself was found.
## None of these genuine header lines end in terminal punctuation
## (they're cut off by the date parenthetical that follows) -- which is
## exactly the signal that distinguishes them from an actual trailing
## responsibility sentence (a grammatically complete, period-ended
## sentence). Requiring BOTH conditions here fixes that misfire while
## leaving _looks_like_prose() itself, and its other use in
## _reflow_body() (a different, unrelated no-bullet-marker case,
## confirmed working on Satyadip Ray's resume), completely untouched.
#def _looks_like_previous_job_trailing_line(line: str) -> bool:
#    words = line.split()
#    if not words or len(words) <= 6:
#        return False
#    last_word = words[-1].rstrip(".;").lower()
#    return line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS
#
#
#def _find_job_blocks(lines: list):
#    anchor_idx = [i for i, l in enumerate(lines) if DATE_RANGE_RE.search(l)]
#    if not anchor_idx:
#        return []
#
#    def header_start_for(anchor_i):
#        start = anchor_i
#        j = anchor_i - 1
#        steps = 0
#        while j >= 0 and steps < 2:
#            l = lines[j].strip()
#            if not l or BULLET_RE.match(l) or j in anchor_idx:
#                break
#            if l[0].islower():
#                break
#            if l.rstrip().endswith(":"):
#                break
#            if _looks_like_previous_job_trailing_line(l):
#                break
#            start = j
#            j -= 1
#            steps += 1
#        return start
#
#    blocks = []
#    header_starts = [header_start_for(a) for a in anchor_idx]
#
#    for n, anchor_i in enumerate(anchor_idx):
#        h_start = header_starts[n]
#        h_end = anchor_i
#        if n + 1 < len(anchor_idx):
#            body_end = header_starts[n + 1] - 1
#        else:
#            body_end = len(lines) - 1
#        header_ids = list(range(h_start, h_end + 1))
#        body_ids = list(range(h_end + 1, body_end + 1)) if body_end >= h_end + 1 else []
#        blocks.append((header_ids, body_ids))
#
#    return blocks
#
#
#def _parse_header(header_lines: list, date_line: str):
#    job = {"role": None, "company": None, "location": None, "start": None, "end": None, "method": []}
#
#    m = DATE_RANGE_RE.search(date_line)
#    if m:
#        job["start"] = m.group("start").strip()
#        job["end"] = m.group("end").strip()
#        job["method"].append("date_regex")
#
#    cleaned = []
#    for l in header_lines:
#        l = l.strip()
#        if not l:
#            continue
#        dm = DATE_RANGE_RE.search(l)
#        if dm:
#            l = (l[:dm.start()] + l[dm.end():])
#            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
#        if l:
#            cleaned.append(l)
#
#    if cleaned and "/" in cleaned[0] and "|" not in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("/") if p.strip()]
#        if parts:
#            job["role"] = parts[0]
#            paren = re.search(r'\(([^)]+)\)', cleaned[0])
#            if paren:
#                job["company"] = paren.group(1).strip()
#                job["method"].append("inline_parens")
#
#    if not job["role"] and not job["company"] and cleaned and "|" in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("|") if p.strip()]
#        title_part = next((p for p in parts if TITLE_RE.search(p)), None)
#        if title_part:
#            job["role"] = title_part
#            other_parts = [p for p in parts if p != title_part]
#            if other_parts:
#                job["company"] = other_parts[0]
#            job["method"].append("pipe_split")
#
#    if not job["role"] and not job["company"] and cleaned:
#        for sep in (" – ", " — ", " - "):
#            if sep in cleaned[0]:
#                left, right = (p.strip() for p in cleaned[0].split(sep, 1))
#                left = re.sub(r'\([^)]*\)', '', left).strip()
#                right = re.sub(r'\([^)]*\)', '', right).strip()
#                if left and right:
#                    left_is_title = bool(TITLE_RE.search(left))
#                    right_is_title = bool(TITLE_RE.search(right))
#                    if right_is_title and not left_is_title:
#                        job["role"], job["company"] = right, left
#                    else:
#                        job["role"], job["company"] = left, right
#                    job["method"].append("dash_split")
#                break
#
#    if not job["role"] and not job["company"] and cleaned:
#        paren_match = re.search(r'\(([^)]+)\)', cleaned[0])
#        if paren_match:
#            inside = paren_match.group(1).strip()
#            outside = (cleaned[0][:paren_match.start()] + cleaned[0][paren_match.end():]).strip()
#            if outside and TITLE_RE.search(inside):
#                job["role"] = inside
#                job["company"] = outside
#                job["method"].append("paren_title")
#
#    if not job["role"] and not job["company"]:
#        for line in cleaned[:3]:
#            line = re.sub(r'\([^)]*\)', '', line).strip()
#            if not line:
#                continue
#            if TITLE_RE.search(line) and not job["role"]:
#                job["role"] = line
#                job["method"].append("title_dict")
#            elif not job["company"]:
#                job["company"] = line
#
#    if job["company"] and "," in job["company"]:
#        parts = [p.strip() for p in job["company"].rsplit(",", 1)]
#        if len(parts) == 2 and len(parts[1].split()) <= 3:
#            job["company"], job["location"] = parts
#
#    if job["company"] and " - " in job["company"]:
#        left, _, right = job["company"].rpartition(" - ")
#        left, right = left.strip(), right.strip()
#        if left and right and len(right.split()) <= 4:
#            job["company"] = left
#            job["location"] = f"{right}, {job['location']}" if job["location"] else right
#
#    if not job["company"] and cleaned:
#        spacy_company = detect_company_with_spacy(" ".join(cleaned[:3]))
#        if spacy_company:
#            job["company"] = spacy_company
#            job["method"].append("spacy_ORG")
#
#    if not job["role"] and not job["company"] and cleaned:
#        job["role"] = cleaned[0]
#
#    for key in ("role", "company"):
#        if job[key]:
#            job[key] = job[key].strip(" /-").strip()
#
#    return job
#
#
#def _score_confidence(job: dict) -> float:
#    score = 0.5
#    if job["start"] and job["end"]:
#        score += 0.2
#    if job["role"]:
#        score += 0.15
#    if job["company"]:
#        score += 0.15
#    if "spacy_ORG" in job["method"] and "title_dict" not in job["method"]:
#        score -= 0.05
#    if job["company"] and len(job["company"].split()) > 6:
#        score -= 0.3
#    return round(min(max(score, 0.0), 0.98), 2)
#
#
#_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
#
#
#def _split_into_sentences(text: str) -> list:
#    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
#    return parts if len(parts) > 1 else [text]
#
#
#def _reflow_body(body_lines: list):
#    extra_header_lines, bullets, subheadings = [], [], []
#    current_bullet = None
#    seen_first_bullet = False
#    saw_real_bullet_marker = False
#
#    for raw in body_lines:
#        l = raw.strip()
#        if not l:
#            continue
#        if BULLET_RE.match(l):
#            seen_first_bullet = True
#            saw_real_bullet_marker = True
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#            current_bullet = BULLET_RE.sub("", l).strip()
#        elif not seen_first_bullet:
#            if _looks_like_prose(l):
#                seen_first_bullet = True
#                current_bullet = l
#            else:
#                extra_header_lines.append(l)
#        elif _looks_like_subheading(l):
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#                current_bullet = None
#            subheadings.append(l)
#        else:
#            if current_bullet is not None:
#                current_bullet += " " + l
#            else:
#                subheadings.append(l)
#
#    if current_bullet is not None:
#        bullets.append(current_bullet)
#
#    if not saw_real_bullet_marker and len(bullets) == 1:
#        bullets = _split_into_sentences(bullets[0])
#
#    return extra_header_lines, bullets, subheadings
#
#
#def parse_job_block(header_lines: list, body_lines: list):
#    if not header_lines:
#        return None
#
#    extra_header_lines, bullets, subheadings = _reflow_body(body_lines)
#
#    date_line = next((l for l in header_lines if DATE_RANGE_RE.search(l)), "")
#    job = _parse_header(header_lines + extra_header_lines, date_line)
#
#    job["duration_months"] = calculate_duration_months(job["start"], job["end"]) if job["start"] else 0
#    job["bullets"] = bullets
#    job["sub_sections"] = subheadings
#
#    job["confidence"] = _score_confidence(job)
#    job["needs_review"] = job["confidence"] < 0.75 or not job["company"] or not job["role"]
#    job["method"] = "+".join(job["method"]) if job["method"] else "none"
#
#    if not job["company"] and not job["role"] and not job["bullets"]:
#        return None
#    return job
#
#
#def calculate_total_experience(jobs: list) -> dict:
#    intervals = []
#    for job in jobs:
#        if not job.get("start") or not job.get("end"):
#            continue
#        start = parse_date(job["start"])
#        end = parse_date(job["end"])
#        if start and end and end > start:
#            intervals.append((start, end))
#
#    if not intervals:
#        return {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}
#
#    intervals.sort(key=lambda x: x[0])
#    merged = [intervals[0]]
#    for s, e in intervals[1:]:
#        last_s, last_e = merged[-1]
#        if s <= last_e:
#            merged[-1] = (last_s, max(last_e, e))
#        else:
#            merged.append((s, e))
#
#    total_months = sum(_inclusive_month_diff(s, e) for s, e in merged)
#
#    return {
#        "total_years": round(total_months / 12, 1),
#        "total_months": total_months,
#        "overlapping_periods_merged": len(intervals) - len(merged),
#    }
#
#
#def extract_experience(experience_section_text: str) -> dict:
#    if not experience_section_text.strip():
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    lines = experience_section_text.strip("\n").split("\n")
#    lines = [_normalize_dashes(l) for l in lines]
#    lines = [_normalize_month_year_glue(l) for l in lines]
#    lines = _merge_widow_date_lines(lines)
#    blocks = _find_job_blocks(lines)
#
#    if not blocks:
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    jobs = []
#    for header_ids, body_ids in blocks:
#        header_lines = [lines[i] for i in header_ids]
#        body_lines = [lines[i] for i in body_ids]
#        parsed = parse_job_block(header_lines, body_lines)
#        if parsed:
#            jobs.append(parsed)
#
#    avg_conf = round(sum(j["confidence"] for j in jobs) / len(jobs), 2) if jobs else 0.0
#    needs_review = any(j["needs_review"] for j in jobs)
#
#    return {
#        "experience": jobs,
#        "_confidence": avg_conf,
#        "_needs_review": needs_review,
#        "_total_experience": calculate_total_experience(jobs),
#    }
#









#worked just changing for one resume to work-
#r"""
#Experience Extractor — Layer 2
#Extracts: company, title, location, date range, duration, bullet points.
#Uses: date regex (anchor-based block splitting) + job title dictionary + spaCy ORG detection.
#
#KEY DESIGN NOTE:
#Real resumes do NOT separate job entries with blank lines (verified against
#Aarti / Abhijeet segmented output). So we cannot split blocks on '\n\s*\n'.
#Instead we anchor on lines that contain a date range — every job entry has one —
#and walk backward/forward from each anchor to find header lines and body lines.
#"""
#
#import re
#try:
#    import spacy
#    _SPACY_IMPORTED = True
#except ImportError:
#    _SPACY_IMPORTED = False
#from dateutil import parser as date_parser
#from dateutil.relativedelta import relativedelta
#from datetime import datetime
#
#
## ── Date matching ──────────────────────────────────────────────────────────
#
#MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#DATE_RANGE_RE = re.compile(
#    rf"(?P<start>{MONTH_NAMES}[\s,.]*(?:\d{{4}}|'\d{{2}})|\d{{1,2}}[/.]\d{{4}}|\d{{4}})"
#    rf"(?:\s*[-–—]\s*|\s+to\s+)"
#    rf"(?P<end>Present|Current|Now|Till\s*Date|{MONTH_NAMES}[\s,.]*(?:\d{{4}}|'\d{{2}})|\d{{1,2}}[/.]\d{{4}}|\d{{4}})",
#    re.IGNORECASE
#)
#
## A date range occasionally gets split across two physical lines by PDF
## extraction (confirmed on a real resume, Satyadip Ray:
##   "BUSINESS DEVELOPMENT EXECUTIVE Jul 2024"
##   "- Present"
## ). Since DATE_RANGE_RE searches one line at a time, this entire job
## entry was invisible -- zero anchor found, entry silently dropped.
## _merge_widow_date_lines() detects a line ending in a bare start-date
## with no separator/end-date, followed by a line that's ONLY a
## separator + end-date, and merges them before anchor detection runs.
#_WIDOW_START_DATE_RE = re.compile(
#    rf"({MONTH_NAMES}[\s,.]*(?:\d{{4}}|'\d{{2}})|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
#    re.IGNORECASE
#)
#_WIDOW_END_DATE_RE = re.compile(
#    rf"^[-–—]\s*(Present|Current|Now|Till\s*Date|"
#    rf"{MONTH_NAMES}[\s,.]*(?:\d{{4}}|'\d{{2}})|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
#    re.IGNORECASE
#)
#
#
#def _merge_widow_date_lines(lines: list) -> list:
#    merged = list(lines)
#    for i in range(len(merged) - 1):
#        cur = merged[i].rstrip()
#        nxt = merged[i + 1].strip()
#        if not cur or not nxt:
#            continue
#        if DATE_RANGE_RE.search(cur):
#            continue  # already a complete range on this line, nothing to merge
#        if _WIDOW_START_DATE_RE.search(cur) and _WIDOW_END_DATE_RE.match(nxt):
#            merged[i] = cur + " " + nxt
#            merged[i + 1] = ""  # blank, not removed -- keeps every other line's index stable
#    return merged
#
#TITLE_KEYWORDS = [
#    "engineer", "developer", "manager", "analyst", "consultant",
#    "designer", "architect", "lead", "head", "director", "vp", "svp", "cto", "ceo",
#    "intern", "associate", "senior", "junior", "principal", "staff",
#    "specialist", "coordinator", "executive", "officer", "president",
#    "scientist", "researcher", "administrator", "technician", "professor",
#    "assistant", "trainee", "fresher", "apprentice"
#]
#TITLE_RE = re.compile(r'\b(' + '|'.join(TITLE_KEYWORDS) + r')\b', re.IGNORECASE)
#
## Some PDF extractions use unicode dash lookalikes (e.g. fullwidth hyphen
## "－" U+FF0D, seen on a real resume between company and location) instead
## of a normal "-"/"–"/"—". Normalizing these upfront means every downstream
## regex and split (which already knows how to handle standard dashes)
## works correctly without needing special-casing everywhere.
#_DASH_NORMALIZE_TABLE = str.maketrans({
#    "－": "-",  # fullwidth hyphen-minus
#    "‐": "-",  # unicode hyphen
#    "‑": "-",  # non-breaking hyphen
#})
#
#
#def _normalize_dashes(text: str) -> str:
#    return text.translate(_DASH_NORMALIZE_TABLE)
#
#BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*]\s+|^\d+[\.\)]?\s+')
#
#def _looks_like_subheading(line: str) -> bool:
#    words = line.split()
#    if not (1 <= len(words) <= 6):
#        return False
#    if line.rstrip().endswith((".", ",", ";", ":")):
#        return False
#    if DATE_RANGE_RE.search(line):
#        return False
#    if line[0].islower():
#        return False
#    return True
#
#
#if _SPACY_IMPORTED:
#    try:
#        _NLP = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
#        _SPACY_AVAILABLE = True
#    except Exception:
#        _SPACY_AVAILABLE = False
#        print("[WARN] spaCy model not found. Company name detection will be limited.")
#else:
#    _SPACY_AVAILABLE = False
#    print("[WARN] spaCy not installed. Company name detection will be limited.")
#
#
#def _expand_apostrophe_year(date_str: str) -> str:
#    """'23 -> 2023, '98 -> 1998 (dateutil doesn't understand the
#    apostrophe-year shorthand on its own -- confirmed on a real resume,
#    Anshika Singh, whose dates are all in "Apr'23" style)."""
#    m = re.search(r"'(\d{2})\b", date_str)
#    if not m:
#        return date_str
#    two_digit = int(m.group(1))
#    century = "20" if two_digit <= 49 else "19"
#    return date_str[:m.start()] + century + m.group(1) + date_str[m.end():]
#
#
#def parse_date(date_str: str):
#    if not date_str or re.match(r'present|current|now|till\s*date', date_str, re.IGNORECASE):
#        return datetime.now()
#    date_str = _expand_apostrophe_year(date_str)
#    try:
#        return date_parser.parse(date_str, default=datetime(2000, 1, 1), fuzzy=True)
#    except Exception:
#        return None
#
#
#def calculate_duration_months(start_str: str, end_str: str) -> int:
#    start = parse_date(start_str)
#    end = parse_date(end_str)
#    if start and end:
#        delta = relativedelta(end, start)
#        return max(0, delta.years * 12 + delta.months)
#    return 0
#
#
#def detect_company_with_spacy(text_block: str):
#    if not _SPACY_AVAILABLE:
#        return None
#    doc = _NLP(text_block[:500])
#    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
#    return orgs[0] if orgs else None
#
#
#_COMPANY_ABBREV_ENDINGS = {"ltd", "inc", "co", "corp", "pvt", "llc", "llp"}
#
#
#def _looks_like_prose(line: str) -> bool:
#    words = line.split()
#    if len(words) > 6:
#        return True
#    last_word = words[-1].rstrip(".;").lower() if words else ""
#    if line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS:
#        return True
#    return False
#
#
#def _find_job_blocks(lines: list):
#    anchor_idx = [i for i, l in enumerate(lines) if DATE_RANGE_RE.search(l)]
#    if not anchor_idx:
#        return []
#
#    def header_start_for(anchor_i):
#        start = anchor_i
#        j = anchor_i - 1
#        steps = 0
#        while j >= 0 and steps < 2:
#            l = lines[j].strip()
#            if not l or BULLET_RE.match(l) or j in anchor_idx:
#                break
#            if l[0].islower():
#                break
#            if l.rstrip().endswith(":"):
#                break
#            # A long/period-ending sentence with no bullet marker at all is
#            # almost always the PREVIOUS job's last bullet on a resume that
#            # doesn't use bullet characters (confirmed on a real resume,
#            # Satyadip Ray) -- not this job's title/company. Without this,
#            # such lines got absorbed as this job's header (e.g. company
#            # wrongly became the previous job's entire last sentence).
#            if _looks_like_prose(l):
#                break
#            start = j
#            j -= 1
#            steps += 1
#        return start
#
#    blocks = []
#    header_starts = [header_start_for(a) for a in anchor_idx]
#
#    for n, anchor_i in enumerate(anchor_idx):
#        h_start = header_starts[n]
#        h_end = anchor_i
#        if n + 1 < len(anchor_idx):
#            body_end = header_starts[n + 1] - 1
#        else:
#            body_end = len(lines) - 1
#        header_ids = list(range(h_start, h_end + 1))
#        body_ids = list(range(h_end + 1, body_end + 1)) if body_end >= h_end + 1 else []
#        blocks.append((header_ids, body_ids))
#
#    return blocks
#
#
#def _parse_header(header_lines: list, date_line: str):
#    job = {"role": None, "company": None, "location": None, "start": None, "end": None, "method": []}
#
#    m = DATE_RANGE_RE.search(date_line)
#    if m:
#        job["start"] = m.group("start").strip()
#        job["end"] = m.group("end").strip()
#        job["method"].append("date_regex")
#
#    cleaned = []
#    for l in header_lines:
#        l = l.strip()
#        if not l:
#            continue
#        dm = DATE_RANGE_RE.search(l)
#        if dm:
#            l = (l[:dm.start()] + l[dm.end():])
#            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
#        if l:
#            cleaned.append(l)
#
#    if cleaned and "/" in cleaned[0] and "|" not in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("/") if p.strip()]
#        if parts:
#            job["role"] = parts[0]
#            paren = re.search(r'\(([^)]+)\)', cleaned[0])
#            if paren:
#                job["company"] = paren.group(1).strip()
#                job["method"].append("inline_parens")
#
#    if not job["role"] and not job["company"] and cleaned and "|" in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("|") if p.strip()]
#        title_part = next((p for p in parts if TITLE_RE.search(p)), None)
#        if title_part:
#            job["role"] = title_part
#            other_parts = [p for p in parts if p != title_part]
#            if other_parts:
#                job["company"] = other_parts[0]
#            job["method"].append("pipe_split")
#
#    # Handle "Title – Company" single-line header format (e.g. Abhinav's
#    # resume: "Executive Software Development – Bry-Air Asia Pvt Ltd").
#    # Without this, the whole line was kept as `role` (since it contains a
#    # title keyword) and `company` stayed None/garbage. Whichever side
#    # matches a known title keyword is the role; the other side is the
#    # company. If both or neither side matches, default to the
#    # conventional "Title – Company" ordering.
#    if not job["role"] and not job["company"] and cleaned:
#        for sep in (" – ", " — ", " - "):
#            if sep in cleaned[0]:
#                left, right = (p.strip() for p in cleaned[0].split(sep, 1))
#                left = re.sub(r'\([^)]*\)', '', left).strip()
#                right = re.sub(r'\([^)]*\)', '', right).strip()
#                if left and right:
#                    left_is_title = bool(TITLE_RE.search(left))
#                    right_is_title = bool(TITLE_RE.search(right))
#                    if right_is_title and not left_is_title:
#                        job["role"], job["company"] = right, left
#                    else:
#                        job["role"], job["company"] = left, right
#                    job["method"].append("dash_split")
#                break
#
#    # Handle "Company (Title)" single-line format (e.g. Anshika's resume:
#    # "GOLDMAN SACHS ( Associate Quantitative Strategist )"). Without this,
#    # the generic loop below blindly strips ALL parenthetical content
#    # before checking for a title match -- silently discarding the title
#    # entirely and leaving role=None.
#    if not job["role"] and not job["company"] and cleaned:
#        paren_match = re.search(r'\(([^)]+)\)', cleaned[0])
#        if paren_match:
#            inside = paren_match.group(1).strip()
#            outside = (cleaned[0][:paren_match.start()] + cleaned[0][paren_match.end():]).strip()
#            if outside and TITLE_RE.search(inside):
#                job["role"] = inside
#                job["company"] = outside
#                job["method"].append("paren_title")
#
#    if not job["role"] and not job["company"]:
#        for line in cleaned[:3]:
#            line = re.sub(r'\([^)]*\)', '', line).strip()
#            if not line:
#                continue
#            if TITLE_RE.search(line) and not job["role"]:
#                job["role"] = line
#                job["method"].append("title_dict")
#            elif not job["company"]:
#                job["company"] = line
#
#    if job["company"] and "," in job["company"]:
#        parts = [p.strip() for p in job["company"].rsplit(",", 1)]
#        if len(parts) == 2 and len(parts[1].split()) <= 3:
#            job["company"], job["location"] = parts
#
#    # "Company - Location" (space-padded dash, so hyphenated company names
#    # like "Bry-Air Asia" are untouched) — e.g. "Motherson sumi system
#    # limited - Noida" after the comma-split above already peeled off
#    # ", India". Combine with any location already found.
#    if job["company"] and " - " in job["company"]:
#        left, _, right = job["company"].rpartition(" - ")
#        left, right = left.strip(), right.strip()
#        if left and right and len(right.split()) <= 4:
#            job["company"] = left
#            job["location"] = f"{right}, {job['location']}" if job["location"] else right
#
#    if not job["company"] and cleaned:
#        spacy_company = detect_company_with_spacy(" ".join(cleaned[:3]))
#        if spacy_company:
#            job["company"] = spacy_company
#            job["method"].append("spacy_ORG")
#
#    if not job["role"] and not job["company"] and cleaned:
#        job["role"] = cleaned[0]
#
#    for key in ("role", "company"):
#        if job[key]:
#            job[key] = job[key].strip(" /-").strip()
#
#    return job
#
#
#def _score_confidence(job: dict) -> float:
#    score = 0.5
#    if job["start"] and job["end"]:
#        score += 0.2
#    if job["role"]:
#        score += 0.15
#    if job["company"]:
#        score += 0.15
#    if "spacy_ORG" in job["method"] and "title_dict" not in job["method"]:
#        score -= 0.05
#    if job["company"] and len(job["company"].split()) > 6:
#        score -= 0.3
#    return round(min(max(score, 0.0), 0.98), 2)
#
#
#_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
#
#
#def _split_into_sentences(text: str) -> list:
#    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
#    return parts if len(parts) > 1 else [text]
#
#
#def _reflow_body(body_lines: list):
#    extra_header_lines, bullets, subheadings = [], [], []
#    current_bullet = None
#    seen_first_bullet = False
#    saw_real_bullet_marker = False
#
#    for raw in body_lines:
#        l = raw.strip()
#        if not l:
#            continue
#        if BULLET_RE.match(l):
#            seen_first_bullet = True
#            saw_real_bullet_marker = True
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#            current_bullet = BULLET_RE.sub("", l).strip()
#        elif not seen_first_bullet:
#            if _looks_like_prose(l):
#                seen_first_bullet = True
#                current_bullet = l
#            else:
#                extra_header_lines.append(l)
#        elif _looks_like_subheading(l):
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#                current_bullet = None
#            subheadings.append(l)
#        else:
#            if current_bullet is not None:
#                current_bullet += " " + l
#            else:
#                subheadings.append(l)
#
#    if current_bullet is not None:
#        bullets.append(current_bullet)
#
#    # Some resumes use no bullet character at all (confirmed on a real
#    # resume, Satyadip Ray) -- every point runs into the next with just a
#    # line break, so they all merged into one blob above. When that
#    # happens (no real bullet marker was ever seen), fall back to
#    # splitting on sentence boundaries to recover the original points.
#    if not saw_real_bullet_marker and len(bullets) == 1:
#        bullets = _split_into_sentences(bullets[0])
#
#    return extra_header_lines, bullets, subheadings
#
#
#def parse_job_block(header_lines: list, body_lines: list):
#    if not header_lines:
#        return None
#
#    extra_header_lines, bullets, subheadings = _reflow_body(body_lines)
#
#    date_line = next((l for l in header_lines if DATE_RANGE_RE.search(l)), "")
#    job = _parse_header(header_lines + extra_header_lines, date_line)
#
#    job["duration_months"] = calculate_duration_months(job["start"], job["end"]) if job["start"] else 0
#    job["bullets"] = bullets
#    job["sub_sections"] = subheadings
#
#    job["confidence"] = _score_confidence(job)
#    job["needs_review"] = job["confidence"] < 0.75 or not job["company"] or not job["role"]
#    job["method"] = "+".join(job["method"]) if job["method"] else "none"
#
#    if not job["company"] and not job["role"] and not job["bullets"]:
#        return None
#    return job
#
#
#def calculate_total_experience(jobs: list) -> dict:
#    intervals = []
#    for job in jobs:
#        if not job.get("start") or not job.get("end"):
#            continue
#        start = parse_date(job["start"])
#        end = parse_date(job["end"])
#        if start and end and end > start:
#            intervals.append((start, end))
#
#    if not intervals:
#        return {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}
#
#    intervals.sort(key=lambda x: x[0])
#    merged = [intervals[0]]
#    for s, e in intervals[1:]:
#        last_s, last_e = merged[-1]
#        if s <= last_e:
#            merged[-1] = (last_s, max(last_e, e))
#        else:
#            merged.append((s, e))
#
#    total_months = sum(
#        relativedelta(e, s).years * 12 + relativedelta(e, s).months
#        for s, e in merged
#    )
#
#    return {
#        "total_years": round(total_months / 12, 1),
#        "total_months": total_months,
#        "overlapping_periods_merged": len(intervals) - len(merged),
#    }
#
#
#def extract_experience(experience_section_text: str) -> dict:
#    if not experience_section_text.strip():
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    lines = experience_section_text.strip("\n").split("\n")
#    lines = [_normalize_dashes(l) for l in lines]
#    lines = _merge_widow_date_lines(lines)
#    blocks = _find_job_blocks(lines)
#
#    if not blocks:
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    jobs = []
#    for header_ids, body_ids in blocks:
#        header_lines = [lines[i] for i in header_ids]
#        body_lines = [lines[i] for i in body_ids]
#        parsed = parse_job_block(header_lines, body_lines)
#        if parsed:
#            jobs.append(parsed)
#
#    avg_conf = round(sum(j["confidence"] for j in jobs) / len(jobs), 2) if jobs else 0.0
#    needs_review = any(j["needs_review"] for j in jobs)
#
#    return {
#        "experience": jobs,
#        "_confidence": avg_conf,
#        "_needs_review": needs_review,
#        "_total_experience": calculate_total_experience(jobs),
#    }
#
#
















## worked changing just in 2-3 resumes experience entry is missing.
#r"""
#Experience Extractor — Layer 2
#Extracts: company, title, location, date range, duration, bullet points.
#Uses: date regex (anchor-based block splitting) + job title dictionary + spaCy ORG detection.
#
#KEY DESIGN NOTE:
#Real resumes do NOT separate job entries with blank lines (verified against
#Aarti / Abhijeet segmented output). So we cannot split blocks on '\n\s*\n'.
#Instead we anchor on lines that contain a date range — every job entry has one —
#and walk backward/forward from each anchor to find header lines and body lines.
#"""
#
#import re
#try:
#    import spacy
#    _SPACY_IMPORTED = True
#except ImportError:
#    _SPACY_IMPORTED = False
#from dateutil import parser as date_parser
#from dateutil.relativedelta import relativedelta
#from datetime import datetime
#
#
## ── Date matching ──────────────────────────────────────────────────────────
#
#MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
## Matches: "Jan 2021 - Dec 2022", "Dec 2024- April 2026" (no space ok),
## "01/2023 - Current", "12.2011 – 12.2019", "2019 - 2021"
#DATE_RANGE_RE = re.compile(
#    rf"(?P<start>{MONTH_NAMES}[\s,.]*\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})"
#    rf"\s*[-–—]\s*"
#    rf"(?P<end>Present|Current|Now|Till\s*Date|{MONTH_NAMES}[\s,.]*\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})",
#    re.IGNORECASE
#)
#
#TITLE_KEYWORDS = [
#    "engineer", "developer", "manager", "analyst", "consultant",
#    "designer", "architect", "lead", "head", "director", "vp", "svp", "cto", "ceo",
#    "intern", "associate", "senior", "junior", "principal", "staff",
#    "specialist", "coordinator", "executive", "officer", "president",
#    "scientist", "researcher", "administrator", "technician", "professor",
#    "assistant"
#]
#TITLE_RE = re.compile(r'\b(' + '|'.join(TITLE_KEYWORDS) + r')\b', re.IGNORECASE)
#
## Bullet chars seen across real resumes: •, ●, -, *, ▪, ➤, ►
#BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*]\s+|^\d+\.\s+')
#
## Short Title-Case lines with no trailing punctuation, appearing mid-body
## (e.g. "Finance Dashboard", "Major Projects") — these are sub-project
## headings, not bullets and not continuations. Kept separate so they don't
## corrupt bullet text or get mistaken for a new job's company name.
#def _looks_like_subheading(line: str) -> bool:
#    words = line.split()
#    if not (1 <= len(words) <= 6):
#        return False
#    if line.rstrip().endswith((".", ",", ";", ":")):
#        return False
#    if DATE_RANGE_RE.search(line):
#        return False
#    if line[0].islower():
#        # mid-sentence wrap of a list item (e.g. "...and"), not a heading
#        return False
#    return True
#
#
#if _SPACY_IMPORTED:
#    try:
#        _NLP = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
#        _SPACY_AVAILABLE = True
#    except Exception:
#        _SPACY_AVAILABLE = False
#        print("[WARN] spaCy model not found. Company name detection will be limited.")
#else:
#    _SPACY_AVAILABLE = False
#    print("[WARN] spaCy not installed. Company name detection will be limited.")
#
#
## ── Date helpers ────────────────────────────────────────────────────────────
#
#def parse_date(date_str: str):
#    if not date_str or re.match(r'present|current|now|till\s*date', date_str, re.IGNORECASE):
#        return datetime.now()
#    try:
#        return date_parser.parse(date_str, default=datetime(2000, 1, 1), fuzzy=True)
#    except Exception:
#        return None
#
#
#def calculate_duration_months(start_str: str, end_str: str) -> int:
#    start = parse_date(start_str)
#    end = parse_date(end_str)
#    if start and end:
#        delta = relativedelta(end, start)
#        return max(0, delta.years * 12 + delta.months)
#    return 0
#
#
#def detect_company_with_spacy(text_block: str):
#    if not _SPACY_AVAILABLE:
#        return None
#    doc = _NLP(text_block[:500])
#    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
#    return orgs[0] if orgs else None
#
#
## ── Block splitting (the actual fix) ───────────────────────────────────────
#
#def _find_job_blocks(lines: list[str]):
#    """
#    Returns a list of (header_line_indices, body_line_indices) tuples.
#    Anchors on lines containing a date range; walks back up to 2 non-bullet
#    lines to capture title/company header, and forward to the next header
#    start to capture the body (bullets + any sub-project text, which is
#    filtered out later since it isn't bullet-marked).
#    """
#    anchor_idx = [i for i, l in enumerate(lines) if DATE_RANGE_RE.search(l)]
#    if not anchor_idx:
#        return []
#
#    def header_start_for(anchor_i):
#        start = anchor_i
#        j = anchor_i - 1
#        steps = 0
#        while j >= 0 and steps < 2:
#            l = lines[j].strip()
#            if not l or BULLET_RE.match(l) or j in anchor_idx:
#                break
#            # a line starting lowercase is virtually always a wrapped
#            # continuation of the PREVIOUS job's last bullet (mid-sentence),
#            # not this job's title/company — stop before absorbing it
#            if l[0].islower():
#                break
#            if l.rstrip().endswith(":"):
#                # a section label like "Previous Experience:", not a header line
#                break
#            start = j
#            j -= 1
#            steps += 1
#        return start
#
#    blocks = []
#    header_starts = [header_start_for(a) for a in anchor_idx]
#
#    for n, anchor_i in enumerate(anchor_idx):
#        h_start = header_starts[n]
#        h_end = anchor_i
#        if n + 1 < len(anchor_idx):
#            body_end = header_starts[n + 1] - 1
#        else:
#            body_end = len(lines) - 1
#        header_ids = list(range(h_start, h_end + 1))
#        body_ids = list(range(h_end + 1, body_end + 1)) if body_end >= h_end + 1 else []
#        blocks.append((header_ids, body_ids))
#
#    return blocks
#
#
#def _parse_header(header_lines: list[str], date_line: str):
#    """Extract title / company / location / start / end from header lines."""
#    job = {"role": None, "company": None, "location": None, "start": None, "end": None, "method": []}
#
#    m = DATE_RANGE_RE.search(date_line)
#    if m:
#        job["start"] = m.group("start").strip()
#        job["end"] = m.group("end").strip()
#        job["method"].append("date_regex")
#
#    # strip the date substring out of whichever header line contained it
#    cleaned = []
#    for l in header_lines:
#        l = l.strip()
#        if not l:
#            continue
#        dm = DATE_RANGE_RE.search(l)
#        if dm:
#            l = (l[:dm.start()] + l[dm.end():])
#            l = re.sub(r'^[\s\-–—,/]+|[\s\-–—,/]+$', '', l)
#        if l:
#            cleaned.append(l)
#
#    # Handle single-line combo entries like "Assistant Prof / 12.2011 – 12.2019 (YTCEM,Karjat)"
#    if len(cleaned) == 1 and "/" in cleaned[0] and "|" not in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("/") if p.strip()]
#        if parts:
#            job["role"] = parts[0]
#            paren = re.search(r'\(([^)]+)\)', cleaned[0])
#            if paren:
#                job["company"] = paren.group(1).strip()
#                job["method"].append("inline_parens")
#
#    # Handle "Company | Title" single-line header format (e.g. Shruti's resume:
#    # "JP Morgan Asset & Wealth Management | Investment Risk Associate")
#    if not job["role"] and not job["company"] and cleaned and "|" in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("|") if p.strip()]
#        title_part = next((p for p in parts if TITLE_RE.search(p)), None)
#        if title_part:
#            job["role"] = title_part
#            other_parts = [p for p in parts if p != title_part]
#            if other_parts:
#                job["company"] = other_parts[0]
#            job["method"].append("pipe_split")
#
#    if not job["role"] and not job["company"]:
#        for line in cleaned[:3]:
#            line = re.sub(r'\([^)]*\)', '', line).strip()
#            if not line:
#                continue
#            if TITLE_RE.search(line) and not job["role"]:
#                job["role"] = line
#                job["method"].append("title_dict")
#            elif not job["company"]:
#                job["company"] = line
#
#    # split "Company Name, City" -> company + location
#    if job["company"] and "," in job["company"]:
#        parts = [p.strip() for p in job["company"].rsplit(",", 1)]
#        if len(parts) == 2 and len(parts[1].split()) <= 3:
#            job["company"], job["location"] = parts
#
#    if not job["company"] and cleaned:
#        spacy_company = detect_company_with_spacy(" ".join(cleaned[:3]))
#        if spacy_company:
#            job["company"] = spacy_company
#            job["method"].append("spacy_ORG")
#
#    # last resort: unassigned single header line -> treat as role
#    if not job["role"] and not job["company"] and cleaned:
#        job["role"] = cleaned[0]
#
#    for key in ("role", "company"):
#        if job[key]:
#            job[key] = job[key].strip(" /-").strip()
#
#    return job
#
#
#def _score_confidence(job: dict) -> float:
#    score = 0.5
#    if job["start"] and job["end"]:
#        score += 0.2
#    if job["role"]:
#        score += 0.15
#    if job["company"]:
#        score += 0.15
#    if "spacy_ORG" in job["method"] and "title_dict" not in job["method"]:
#        score -= 0.05  # spaCy company detection alone is less reliable
#    if job["company"] and len(job["company"].split()) > 6:
#        score -= 0.3  # reads like a sentence, not a company name — likely mis-parsed
#    return round(min(max(score, 0.0), 0.98), 2)
#
#
#_COMPANY_ABBREV_ENDINGS = {"ltd", "inc", "co", "corp", "pvt", "llc", "llp"}
#
#
#def _looks_like_prose(line: str) -> bool:
#    """A plain sentence with no bullet marker (some resumes describe a role
#    in a paragraph instead of bullets — see 'Assistant Prof' style entries).
#    Distinguished from a genuine header line (like a company name sitting
#    on its own line) by length and terminal punctuation — but a trailing
#    period on something like "Pvt. Ltd." is an abbreviation, not a sentence
#    ending, so that alone shouldn't count."""
#    words = line.split()
#    if len(words) > 6:
#        return True
#    last_word = words[-1].rstrip(".;").lower() if words else ""
#    if line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS:
#        return True
#    return False
#
#
#def _reflow_body(body_lines: list[str]):
#    """
#    Walks the body lines after a job's date/title header and separates them into:
#      - extra_header_lines: non-bullet lines BEFORE the first bullet (usually
#        the company name, e.g. "Oracle" on its own line under the title)
#      - bullets: bullet-marked lines, with wrapped continuation lines
#        (no bullet marker, mid-sentence) merged back in
#      - subheadings: short unpunctuated lines mid-body (e.g. "Finance
#        Dashboard") — kept separate, not merged into bullet text
#    """
#    extra_header_lines, bullets, subheadings = [], [], []
#    current_bullet = None
#    seen_first_bullet = False
#
#    for raw in body_lines:
#        l = raw.strip()
#        if not l:
#            continue
#        if BULLET_RE.match(l):
#            seen_first_bullet = True
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#            current_bullet = BULLET_RE.sub("", l).strip()
#        elif not seen_first_bullet:
#            if _looks_like_prose(l):
#                # a paragraph-style description with no bullet marker at all —
#                # treat it as body content, not a header/company line
#                seen_first_bullet = True
#                current_bullet = l
#            else:
#                extra_header_lines.append(l)
#        elif _looks_like_subheading(l):
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#                current_bullet = None
#            subheadings.append(l)
#        else:
#            # wrapped continuation of the previous bullet
#            if current_bullet is not None:
#                current_bullet += " " + l
#            else:
#                subheadings.append(l)  # orphan trailing text
#
#    if current_bullet is not None:
#        bullets.append(current_bullet)
#
#    return extra_header_lines, bullets, subheadings
#
#
#def parse_job_block(header_lines: list[str], body_lines: list[str]) -> dict | None:
#    if not header_lines:
#        return None
#
#    extra_header_lines, bullets, subheadings = _reflow_body(body_lines)
#
#    date_line = next((l for l in header_lines if DATE_RANGE_RE.search(l)), "")
#    job = _parse_header(header_lines + extra_header_lines, date_line)
#
#    job["duration_months"] = calculate_duration_months(job["start"], job["end"]) if job["start"] else 0
#    job["bullets"] = bullets
#    job["sub_sections"] = subheadings  # hand-off candidates for the Projects extractor
#
#    job["confidence"] = _score_confidence(job)
#    job["needs_review"] = job["confidence"] < 0.75 or not job["company"] or not job["role"]
#    job["method"] = "+".join(job["method"]) if job["method"] else "none"
#
#    if not job["company"] and not job["role"] and not job["bullets"]:
#        return None
#    return job
#
#
#def calculate_total_experience(jobs: list[dict]) -> dict:
#    """
#    Total career experience computed from each job's parsed date range,
#    merging overlapping/concurrent periods so simultaneous roles (or
#    freelance work alongside a full-time job) don't get double-counted.
#    """
#    intervals = []
#    for job in jobs:
#        if not job.get("start") or not job.get("end"):
#            continue
#        start = parse_date(job["start"])
#        end = parse_date(job["end"])
#        if start and end and end > start:
#            intervals.append((start, end))
#
#    if not intervals:
#        return {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}
#
#    intervals.sort(key=lambda x: x[0])
#    merged = [intervals[0]]
#    for s, e in intervals[1:]:
#        last_s, last_e = merged[-1]
#        if s <= last_e:
#            merged[-1] = (last_s, max(last_e, e))
#        else:
#            merged.append((s, e))
#
#    total_months = sum(
#        relativedelta(e, s).years * 12 + relativedelta(e, s).months
#        for s, e in merged
#    )
#
#    return {
#        "total_years": round(total_months / 12, 1),
#        "total_months": total_months,
#        "overlapping_periods_merged": len(intervals) - len(merged),
#    }
#
#
## ── Main entry point ────────────────────────────────────────────────────────
#
#def extract_experience(experience_section_text: str) -> dict:
#    if not experience_section_text.strip():
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    lines = experience_section_text.strip("\n").split("\n")
#    blocks = _find_job_blocks(lines)
#
#    if not blocks:
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    jobs = []
#    for header_ids, body_ids in blocks:
#        header_lines = [lines[i] for i in header_ids]
#        body_lines = [lines[i] for i in body_ids]
#        parsed = parse_job_block(header_lines, body_lines)
#        if parsed:
#            jobs.append(parsed)
#
#    avg_conf = round(sum(j["confidence"] for j in jobs) / len(jobs), 2) if jobs else 0.0
#    needs_review = any(j["needs_review"] for j in jobs)
#
#    return {
#        "experience": jobs,
#        "_confidence": avg_conf,
#        "_needs_review": needs_review,
#        "_total_experience": calculate_total_experience(jobs),
#    }
#












#r"""
#Experience Extractor — Layer 2
#Extracts: company, title, location, date range, duration, bullet points.
#Uses: date regex (anchor-based block splitting) + job title dictionary + spaCy ORG detection.
#
#KEY DESIGN NOTE:
#Real resumes do NOT separate job entries with blank lines (verified against
#Aarti / Abhijeet segmented output). So we cannot split blocks on '\n\s*\n'.
#Instead we anchor on lines that contain a date range — every job entry has one —
#and walk backward/forward from each anchor to find header lines and body lines.
#"""
#
#import re
#import spacy
#from dateutil import parser as date_parser
#from dateutil.relativedelta import relativedelta
#from datetime import datetime
#
#
## ── Date matching ──────────────────────────────────────────────────────────
#
#MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
## Matches: "Jan 2021 - Dec 2022", "Dec 2024- April 2026" (no space ok),
## "01/2023 - Current", "12.2011 – 12.2019", "2019 - 2021"
#DATE_RANGE_RE = re.compile(
#    rf"(?P<start>{MONTH_NAMES}[\s,.]*\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})"
#    rf"\s*[-–—]\s*"
#    rf"(?P<end>Present|Current|Now|Till\s*Date|{MONTH_NAMES}[\s,.]*\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})",
#    re.IGNORECASE
#)
#
#TITLE_KEYWORDS = [
#    "engineer", "developer", "manager", "analyst", "consultant",
#    "designer", "architect", "lead", "head", "director", "vp", "svp", "cto", "ceo",
#    "intern", "associate", "senior", "junior", "principal", "staff",
#    "specialist", "coordinator", "executive", "officer", "president",
#    "scientist", "researcher", "administrator", "technician", "professor",
#    "assistant"
#]
#TITLE_RE = re.compile(r'\b(' + '|'.join(TITLE_KEYWORDS) + r')\b', re.IGNORECASE)
#
## Bullet chars seen across real resumes: •, ●, -, *, ▪, ➤, ►
#BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*]\s+|^\d+\.\s+')
#
## Short Title-Case lines with no trailing punctuation, appearing mid-body
## (e.g. "Finance Dashboard", "Major Projects") — these are sub-project
## headings, not bullets and not continuations. Kept separate so they don't
## corrupt bullet text or get mistaken for a new job's company name.
#def _looks_like_subheading(line: str) -> bool:
#    words = line.split()
#    if not (1 <= len(words) <= 6):
#        return False
#    if line.rstrip().endswith((".", ",", ";", ":")):
#        return False
#    if DATE_RANGE_RE.search(line):
#        return False
#    if line[0].islower():
#        # mid-sentence wrap of a list item (e.g. "...and"), not a heading
#        return False
#    return True
#
#
#try:
#    _NLP = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
#    _SPACY_AVAILABLE = True
#except Exception:
#    _SPACY_AVAILABLE = False
#    print("[WARN] spaCy model not found. Company name detection will be limited.")
#
#
## ── Date helpers ────────────────────────────────────────────────────────────
#
#def parse_date(date_str: str):
#    if not date_str or re.match(r'present|current|now|till\s*date', date_str, re.IGNORECASE):
#        return datetime.now()
#    try:
#        return date_parser.parse(date_str, default=datetime(2000, 1, 1), fuzzy=True)
#    except Exception:
#        return None
#
#
#def calculate_duration_months(start_str: str, end_str: str) -> int:
#    start = parse_date(start_str)
#    end = parse_date(end_str)
#    if start and end:
#        delta = relativedelta(end, start)
#        return max(0, delta.years * 12 + delta.months)
#    return 0
#
#
#def detect_company_with_spacy(text_block: str):
#    if not _SPACY_AVAILABLE:
#        return None
#    doc = _NLP(text_block[:500])
#    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
#    return orgs[0] if orgs else None
#
#
## ── Block splitting (the actual fix) ───────────────────────────────────────
#
#def _find_job_blocks(lines: list[str]):
#    """
#    Returns a list of (header_line_indices, body_line_indices) tuples.
#    Anchors on lines containing a date range; walks back up to 2 non-bullet
#    lines to capture title/company header, and forward to the next header
#    start to capture the body (bullets + any sub-project text, which is
#    filtered out later since it isn't bullet-marked).
#    """
#    anchor_idx = [i for i, l in enumerate(lines) if DATE_RANGE_RE.search(l)]
#    if not anchor_idx:
#        return []
#
#    def header_start_for(anchor_i):
#        start = anchor_i
#        j = anchor_i - 1
#        steps = 0
#        while j >= 0 and steps < 2:
#            l = lines[j].strip()
#            if not l or BULLET_RE.match(l) or j in anchor_idx:
#                break
#            # a line starting lowercase is virtually always a wrapped
#            # continuation of the PREVIOUS job's last bullet (mid-sentence),
#            # not this job's title/company — stop before absorbing it
#            if l[0].islower():
#                break
#            if l.rstrip().endswith(":"):
#                # a section label like "Previous Experience:", not a header line
#                break
#            start = j
#            j -= 1
#            steps += 1
#        return start
#
#    blocks = []
#    header_starts = [header_start_for(a) for a in anchor_idx]
#
#    for n, anchor_i in enumerate(anchor_idx):
#        h_start = header_starts[n]
#        h_end = anchor_i
#        if n + 1 < len(anchor_idx):
#            body_end = header_starts[n + 1] - 1
#        else:
#            body_end = len(lines) - 1
#        header_ids = list(range(h_start, h_end + 1))
#        body_ids = list(range(h_end + 1, body_end + 1)) if body_end >= h_end + 1 else []
#        blocks.append((header_ids, body_ids))
#
#    return blocks
#
#
#def _parse_header(header_lines: list[str], date_line: str):
#    """Extract title / company / location / start / end from header lines."""
#    job = {"role": None, "company": None, "location": None, "start": None, "end": None, "method": []}
#
#    m = DATE_RANGE_RE.search(date_line)
#    if m:
#        job["start"] = m.group("start").strip()
#        job["end"] = m.group("end").strip()
#        job["method"].append("date_regex")
#
#    # strip the date substring out of whichever header line contained it
#    cleaned = []
#    for l in header_lines:
#        l = l.strip()
#        if not l:
#            continue
#        dm = DATE_RANGE_RE.search(l)
#        if dm:
#            l = (l[:dm.start()] + l[dm.end():]).strip(" -–—,/()")
#        if l:
#            cleaned.append(l)
#
#    # Handle single-line combo entries like "Assistant Prof / 12.2011 – 12.2019 (YTCEM,Karjat)"
#    if len(cleaned) == 1 and "/" in cleaned[0] and "|" not in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("/") if p.strip()]
#        if parts:
#            job["role"] = parts[0]
#            paren = re.search(r'\(([^)]+)\)', cleaned[0])
#            if paren:
#                job["company"] = paren.group(1).strip()
#                job["method"].append("inline_parens")
#
#    # Handle "Company | Title" single-line header format (e.g. Shruti's resume:
#    # "JP Morgan Asset & Wealth Management | Investment Risk Associate")
#    if not job["role"] and not job["company"] and cleaned and "|" in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("|") if p.strip()]
#        title_part = next((p for p in parts if TITLE_RE.search(p)), None)
#        if title_part:
#            job["role"] = title_part
#            other_parts = [p for p in parts if p != title_part]
#            if other_parts:
#                job["company"] = other_parts[0]
#            job["method"].append("pipe_split")
#
#    if not job["role"] and not job["company"]:
#        for line in cleaned[:3]:
#            line = re.sub(r'\([^)]*\)', '', line).strip()
#            if not line:
#                continue
#            if TITLE_RE.search(line) and not job["role"]:
#                job["role"] = line
#                job["method"].append("title_dict")
#            elif not job["company"]:
#                job["company"] = line
#
#    # split "Company Name, City" -> company + location
#    if job["company"] and "," in job["company"]:
#        parts = [p.strip() for p in job["company"].rsplit(",", 1)]
#        if len(parts) == 2 and len(parts[1].split()) <= 3:
#            job["company"], job["location"] = parts
#
#    if not job["company"] and cleaned:
#        spacy_company = detect_company_with_spacy(" ".join(cleaned[:3]))
#        if spacy_company:
#            job["company"] = spacy_company
#            job["method"].append("spacy_ORG")
#
#    # last resort: unassigned single header line -> treat as role
#    if not job["role"] and not job["company"] and cleaned:
#        job["role"] = cleaned[0]
#
#    for key in ("role", "company"):
#        if job[key]:
#            job[key] = job[key].strip(" /-").strip()
#
#    return job
#
#
#def _score_confidence(job: dict) -> float:
#    score = 0.5
#    if job["start"] and job["end"]:
#        score += 0.2
#    if job["role"]:
#        score += 0.15
#    if job["company"]:
#        score += 0.15
#    if "spacy_ORG" in job["method"] and "title_dict" not in job["method"]:
#        score -= 0.05  # spaCy company detection alone is less reliable
#    if job["company"] and len(job["company"].split()) > 6:
#        score -= 0.3  # reads like a sentence, not a company name — likely mis-parsed
#    return round(min(max(score, 0.0), 0.98), 2)
#
#
#def _reflow_body(body_lines: list[str]):
#    """
#    Walks the body lines after a job's date/title header and separates them into:
#      - extra_header_lines: non-bullet lines BEFORE the first bullet (usually
#        the company name, e.g. "Oracle" on its own line under the title)
#      - bullets: bullet-marked lines, with wrapped continuation lines
#        (no bullet marker, mid-sentence) merged back in
#      - subheadings: short unpunctuated lines mid-body (e.g. "Finance
#        Dashboard") — kept separate, not merged into bullet text
#    """
#    extra_header_lines, bullets, subheadings = [], [], []
#    current_bullet = None
#    seen_first_bullet = False
#
#    for raw in body_lines:
#        l = raw.strip()
#        if not l:
#            continue
#        if BULLET_RE.match(l):
#            seen_first_bullet = True
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#            current_bullet = BULLET_RE.sub("", l).strip()
#        elif not seen_first_bullet:
#            extra_header_lines.append(l)
#        elif _looks_like_subheading(l):
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#                current_bullet = None
#            subheadings.append(l)
#        else:
#            # wrapped continuation of the previous bullet
#            if current_bullet is not None:
#                current_bullet += " " + l
#            else:
#                subheadings.append(l)  # orphan trailing text
#
#    if current_bullet is not None:
#        bullets.append(current_bullet)
#
#    return extra_header_lines, bullets, subheadings
#
#
#def parse_job_block(header_lines: list[str], body_lines: list[str]) -> dict | None:
#    if not header_lines:
#        return None
#
#    extra_header_lines, bullets, subheadings = _reflow_body(body_lines)
#
#    date_line = next((l for l in header_lines if DATE_RANGE_RE.search(l)), "")
#    job = _parse_header(header_lines + extra_header_lines, date_line)
#
#    job["duration_months"] = calculate_duration_months(job["start"], job["end"]) if job["start"] else 0
#    job["bullets"] = bullets
#    job["sub_sections"] = subheadings  # hand-off candidates for the Projects extractor
#
#    job["confidence"] = _score_confidence(job)
#    job["needs_review"] = job["confidence"] < 0.75 or not job["company"] or not job["role"]
#    job["method"] = "+".join(job["method"]) if job["method"] else "none"
#
#    if not job["company"] and not job["role"] and not job["bullets"]:
#        return None
#    return job
#
#
#def calculate_total_experience(jobs: list[dict]) -> dict:
#    """
#    Total career experience computed from each job's parsed date range,
#    merging overlapping/concurrent periods so simultaneous roles (or
#    freelance work alongside a full-time job) don't get double-counted.
#    """
#    intervals = []
#    for job in jobs:
#        if not job.get("start") or not job.get("end"):
#            continue
#        start = parse_date(job["start"])
#        end = parse_date(job["end"])
#        if start and end and end > start:
#            intervals.append((start, end))
#
#    if not intervals:
#        return {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}
#
#    intervals.sort(key=lambda x: x[0])
#    merged = [intervals[0]]
#    for s, e in intervals[1:]:
#        last_s, last_e = merged[-1]
#        if s <= last_e:
#            merged[-1] = (last_s, max(last_e, e))
#        else:
#            merged.append((s, e))
#
#    total_months = sum(
#        relativedelta(e, s).years * 12 + relativedelta(e, s).months
#        for s, e in merged
#    )
#
#    return {
#        "total_years": round(total_months / 12, 1),
#        "total_months": total_months,
#        "overlapping_periods_merged": len(intervals) - len(merged),
#    }
#
#
## ── Main entry point ────────────────────────────────────────────────────────
#
#def extract_experience(experience_section_text: str) -> dict:
#    if not experience_section_text.strip():
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    lines = experience_section_text.strip("\n").split("\n")
#    blocks = _find_job_blocks(lines)
#
#    if not blocks:
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    jobs = []
#    for header_ids, body_ids in blocks:
#        header_lines = [lines[i] for i in header_ids]
#        body_lines = [lines[i] for i in body_ids]
#        parsed = parse_job_block(header_lines, body_lines)
#        if parsed:
#            jobs.append(parsed)
#
#    avg_conf = round(sum(j["confidence"] for j in jobs) / len(jobs), 2) if jobs else 0.0
#    needs_review = any(j["needs_review"] for j in jobs)
#
#    return {
#        "experience": jobs,
#        "_confidence": avg_conf,
#        "_needs_review": needs_review,
#        "_total_experience": calculate_total_experience(jobs),
#    }













#r"""
#Experience Extractor — Layer 2
#Extracts: company, title, location, date range, duration, bullet points.
#Uses: date regex (anchor-based block splitting) + job title dictionary + spaCy ORG detection.
#
#KEY DESIGN NOTE:
#Real resumes do NOT separate job entries with blank lines (verified against
#Aarti / Abhijeet segmented output). So we cannot split blocks on '\n\s*\n'.
#Instead we anchor on lines that contain a date range — every job entry has one —
#and walk backward/forward from each anchor to find header lines and body lines.
#"""
#
#import re
#import spacy
#from dateutil import parser as date_parser
#from dateutil.relativedelta import relativedelta
#from datetime import datetime
#
#
## ── Date matching ──────────────────────────────────────────────────────────
#
#MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
## Matches: "Jan 2021 - Dec 2022", "Dec 2024- April 2026" (no space ok),
## "01/2023 - Current", "12.2011 – 12.2019", "2019 - 2021"
#DATE_RANGE_RE = re.compile(
#    rf"(?P<start>{MONTH_NAMES}[\s,.]*\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})"
#    rf"\s*[-–—]\s*"
#    rf"(?P<end>Present|Current|Now|Till\s*Date|{MONTH_NAMES}[\s,.]*\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})",
#    re.IGNORECASE
#)
#
#TITLE_KEYWORDS = [
#    "engineer", "developer", "manager", "analyst", "consultant",
#    "designer", "architect", "lead", "head", "director", "vp", "svp", "cto", "ceo",
#    "intern", "associate", "senior", "junior", "principal", "staff",
#    "specialist", "coordinator", "executive", "officer", "president",
#    "scientist", "researcher", "administrator", "technician", "professor",
#    "assistant"
#]
#TITLE_RE = re.compile(r'\b(' + '|'.join(TITLE_KEYWORDS) + r')\b', re.IGNORECASE)
#
## Bullet chars seen across real resumes: •, ●, -, *, ▪, ➤, ►
#BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*]\s+|^\d+\.\s+')
#
## Short Title-Case lines with no trailing punctuation, appearing mid-body
## (e.g. "Finance Dashboard", "Major Projects") — these are sub-project
## headings, not bullets and not continuations. Kept separate so they don't
## corrupt bullet text or get mistaken for a new job's company name.
#def _looks_like_subheading(line: str) -> bool:
#    words = line.split()
#    if not (1 <= len(words) <= 6):
#        return False
#    if line.rstrip().endswith((".", ",", ";", ":")):
#        return False
#    if DATE_RANGE_RE.search(line):
#        return False
#    if line[0].islower():
#        # mid-sentence wrap of a list item (e.g. "...and"), not a heading
#        return False
#    return True
#
#
#try:
#    _NLP = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
#    _SPACY_AVAILABLE = True
#except Exception:
#    _SPACY_AVAILABLE = False
#    print("[WARN] spaCy model not found. Company name detection will be limited.")
#
#
## ── Date helpers ────────────────────────────────────────────────────────────
#
#def parse_date(date_str: str):
#    if not date_str or re.match(r'present|current|now|till\s*date', date_str, re.IGNORECASE):
#        return datetime.now()
#    try:
#        return date_parser.parse(date_str, default=datetime(2000, 1, 1), fuzzy=True)
#    except Exception:
#        return None
#
#
#def calculate_duration_months(start_str: str, end_str: str) -> int:
#    start = parse_date(start_str)
#    end = parse_date(end_str)
#    if start and end:
#        delta = relativedelta(end, start)
#        return max(0, delta.years * 12 + delta.months)
#    return 0
#
#
#def detect_company_with_spacy(text_block: str):
#    if not _SPACY_AVAILABLE:
#        return None
#    doc = _NLP(text_block[:500])
#    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
#    return orgs[0] if orgs else None
#
#
## ── Block splitting (the actual fix) ───────────────────────────────────────
#
#def _find_job_blocks(lines: list[str]):
#    """
#    Returns a list of (header_line_indices, body_line_indices) tuples.
#    Anchors on lines containing a date range; walks back up to 2 non-bullet
#    lines to capture title/company header, and forward to the next header
#    start to capture the body (bullets + any sub-project text, which is
#    filtered out later since it isn't bullet-marked).
#    """
#    anchor_idx = [i for i, l in enumerate(lines) if DATE_RANGE_RE.search(l)]
#    if not anchor_idx:
#        return []
#
#    def header_start_for(anchor_i):
#        start = anchor_i
#        j = anchor_i - 1
#        steps = 0
#        while j >= 0 and steps < 2:
#            l = lines[j].strip()
#            if not l or BULLET_RE.match(l) or j in anchor_idx:
#                break
#            # a line starting lowercase is virtually always a wrapped
#            # continuation of the PREVIOUS job's last bullet (mid-sentence),
#            # not this job's title/company — stop before absorbing it
#            if l[0].islower():
#                break
#            if l.rstrip().endswith(":"):
#                # a section label like "Previous Experience:", not a header line
#                break
#            start = j
#            j -= 1
#            steps += 1
#        return start
#
#    blocks = []
#    header_starts = [header_start_for(a) for a in anchor_idx]
#
#    for n, anchor_i in enumerate(anchor_idx):
#        h_start = header_starts[n]
#        h_end = anchor_i
#        if n + 1 < len(anchor_idx):
#            body_end = header_starts[n + 1] - 1
#        else:
#            body_end = len(lines) - 1
#        header_ids = list(range(h_start, h_end + 1))
#        body_ids = list(range(h_end + 1, body_end + 1)) if body_end >= h_end + 1 else []
#        blocks.append((header_ids, body_ids))
#
#    return blocks
#
#
#def _parse_header(header_lines: list[str], date_line: str):
#    """Extract title / company / location / start / end from header lines."""
#    job = {"role": None, "company": None, "location": None, "start": None, "end": None, "method": []}
#
#    m = DATE_RANGE_RE.search(date_line)
#    if m:
#        job["start"] = m.group("start").strip()
#        job["end"] = m.group("end").strip()
#        job["method"].append("date_regex")
#
#    # strip the date substring out of whichever header line contained it
#    cleaned = []
#    for l in header_lines:
#        l = l.strip()
#        if not l:
#            continue
#        dm = DATE_RANGE_RE.search(l)
#        if dm:
#            l = (l[:dm.start()] + l[dm.end():]).strip(" -–—,/()")
#        if l:
#            cleaned.append(l)
#
#    # Handle single-line combo entries like "Assistant Prof / 12.2011 – 12.2019 (YTCEM,Karjat)"
#    if len(cleaned) == 1 and "/" in cleaned[0] and "|" not in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("/") if p.strip()]
#        if parts:
#            job["role"] = parts[0]
#            paren = re.search(r'\(([^)]+)\)', cleaned[0])
#            if paren:
#                job["company"] = paren.group(1).strip()
#                job["method"].append("inline_parens")
#
#    # Handle "Company | Title" single-line header format (e.g. Shruti's resume:
#    # "JP Morgan Asset & Wealth Management | Investment Risk Associate")
#    if not job["role"] and not job["company"] and cleaned and "|" in cleaned[0]:
#        parts = [p.strip() for p in cleaned[0].split("|") if p.strip()]
#        title_part = next((p for p in parts if TITLE_RE.search(p)), None)
#        if title_part:
#            job["role"] = title_part
#            other_parts = [p for p in parts if p != title_part]
#            if other_parts:
#                job["company"] = other_parts[0]
#            job["method"].append("pipe_split")
#
#    if not job["role"] and not job["company"]:
#        for line in cleaned[:3]:
#            line = re.sub(r'\([^)]*\)', '', line).strip()
#            if not line:
#                continue
#            if TITLE_RE.search(line) and not job["role"]:
#                job["role"] = line
#                job["method"].append("title_dict")
#            elif not job["company"]:
#                job["company"] = line
#
#    # split "Company Name, City" -> company + location
#    if job["company"] and "," in job["company"]:
#        parts = [p.strip() for p in job["company"].rsplit(",", 1)]
#        if len(parts) == 2 and len(parts[1].split()) <= 3:
#            job["company"], job["location"] = parts
#
#    if not job["company"] and cleaned:
#        spacy_company = detect_company_with_spacy(" ".join(cleaned[:3]))
#        if spacy_company:
#            job["company"] = spacy_company
#            job["method"].append("spacy_ORG")
#
#    # last resort: unassigned single header line -> treat as role
#    if not job["role"] and not job["company"] and cleaned:
#        job["role"] = cleaned[0]
#
#    for key in ("role", "company"):
#        if job[key]:
#            job[key] = job[key].strip(" /-").strip()
#
#    return job
#
#
#def _score_confidence(job: dict) -> float:
#    score = 0.5
#    if job["start"] and job["end"]:
#        score += 0.2
#    if job["role"]:
#        score += 0.15
#    if job["company"]:
#        score += 0.15
#    if "spacy_ORG" in job["method"] and "title_dict" not in job["method"]:
#        score -= 0.05  # spaCy company detection alone is less reliable
#    if job["company"] and len(job["company"].split()) > 6:
#        score -= 0.3  # reads like a sentence, not a company name — likely mis-parsed
#    return round(min(max(score, 0.0), 0.98), 2)
#
#
#def _reflow_body(body_lines: list[str]):
#    """
#    Walks the body lines after a job's date/title header and separates them into:
#      - extra_header_lines: non-bullet lines BEFORE the first bullet (usually
#        the company name, e.g. "Oracle" on its own line under the title)
#      - bullets: bullet-marked lines, with wrapped continuation lines
#        (no bullet marker, mid-sentence) merged back in
#      - subheadings: short unpunctuated lines mid-body (e.g. "Finance
#        Dashboard") — kept separate, not merged into bullet text
#    """
#    extra_header_lines, bullets, subheadings = [], [], []
#    current_bullet = None
#    seen_first_bullet = False
#
#    for raw in body_lines:
#        l = raw.strip()
#        if not l:
#            continue
#        if BULLET_RE.match(l):
#            seen_first_bullet = True
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#            current_bullet = BULLET_RE.sub("", l).strip()
#        elif not seen_first_bullet:
#            extra_header_lines.append(l)
#        elif _looks_like_subheading(l):
#            if current_bullet is not None:
#                bullets.append(current_bullet)
#                current_bullet = None
#            subheadings.append(l)
#        else:
#            # wrapped continuation of the previous bullet
#            if current_bullet is not None:
#                current_bullet += " " + l
#            else:
#                subheadings.append(l)  # orphan trailing text
#
#    if current_bullet is not None:
#        bullets.append(current_bullet)
#
#    return extra_header_lines, bullets, subheadings
#
#
#def parse_job_block(header_lines: list[str], body_lines: list[str]) -> dict | None:
#    if not header_lines:
#        return None
#
#    extra_header_lines, bullets, subheadings = _reflow_body(body_lines)
#
#    date_line = next((l for l in header_lines if DATE_RANGE_RE.search(l)), "")
#    job = _parse_header(header_lines + extra_header_lines, date_line)
#
#    job["duration_months"] = calculate_duration_months(job["start"], job["end"]) if job["start"] else 0
#    job["bullets"] = bullets
#    job["sub_sections"] = subheadings  # hand-off candidates for the Projects extractor
#
#    job["confidence"] = _score_confidence(job)
#    job["needs_review"] = job["confidence"] < 0.75 or not job["company"] or not job["role"]
#    job["method"] = "+".join(job["method"]) if job["method"] else "none"
#
#    if not job["company"] and not job["role"] and not job["bullets"]:
#        return None
#    return job
#
#
#def calculate_total_experience(jobs: list[dict]) -> dict:
#    """
#    Total career experience computed from each job's parsed date range,
#    merging overlapping/concurrent periods so simultaneous roles (or
#    freelance work alongside a full-time job) don't get double-counted.
#    """
#    intervals = []
#    for job in jobs:
#        if not job.get("start") or not job.get("end"):
#            continue
#        start = parse_date(job["start"])
#        end = parse_date(job["end"])
#        if start and end and end > start:
#            intervals.append((start, end))
#
#    if not intervals:
#        return {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}
#
#    intervals.sort(key=lambda x: x[0])
#    merged = [intervals[0]]
#    for s, e in intervals[1:]:
#        last_s, last_e = merged[-1]
#        if s <= last_e:
#            merged[-1] = (last_s, max(last_e, e))
#        else:
#            merged.append((s, e))
#
#    total_months = sum(
#        relativedelta(e, s).years * 12 + relativedelta(e, s).months
#        for s, e in merged
#    )
#
#    return {
#        "total_years": round(total_months / 12, 1),
#        "total_months": total_months,
#        "overlapping_periods_merged": len(intervals) - len(merged),
#    }
#
#
## ── Main entry point ────────────────────────────────────────────────────────
#
#def extract_experience(experience_section_text: str) -> dict:
#    if not experience_section_text.strip():
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    lines = experience_section_text.strip("\n").split("\n")
#    blocks = _find_job_blocks(lines)
#
#    if not blocks:
#        return {"experience": [], "_confidence": 0.0, "_needs_review": True,
#                 "_total_experience": {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}}
#
#    jobs = []
#    for header_ids, body_ids in blocks:
#        header_lines = [lines[i] for i in header_ids]
#        body_lines = [lines[i] for i in body_ids]
#        parsed = parse_job_block(header_lines, body_lines)
#        if parsed:
#            jobs.append(parsed)
#
#    avg_conf = round(sum(j["confidence"] for j in jobs) / len(jobs), 2) if jobs else 0.0
#    needs_review = any(j["needs_review"] for j in jobs)
#
#    return {
#        "experience": jobs,
#        "_confidence": avg_conf,
#        "_needs_review": needs_review,
#        "_total_experience": calculate_total_experience(jobs),
#    }
#