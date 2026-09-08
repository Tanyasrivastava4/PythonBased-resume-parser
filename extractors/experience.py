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
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
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
# _find_job_blocks, not just parsed incompletely.
_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.-]*{MONTH_NAMES}[\s,.-]*\d{{4}}"

DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DAY_MONTH_YEAR}|{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})"
    # The dash/"to" separator is the common case, but some resumes write
    # "<start date> till date" with NO connecting dash or "to" at all --
    # just a space (confirmed on a real resume, Sai Pavan Boddula: "May
    # 2022 till date"). Without a third alternative here, that line never
    # matches DATE_RANGE_RE at all -- not a parsing-quality issue, a
    # total miss. With only one date-range anchor left for the whole
    # experience section, _find_job_blocks built exactly one block
    # (correctly, for the OTHER job) and silently dropped every line
    # before it, including this entire job's header and bullets, since
    # they never fell inside any block's header_ids/body_ids range.
    # The added alternative is a lookahead (consumes nothing itself), so
    # it doesn't eat into the "end" group's own match of "Till Date" --
    # it only fires when a bare run of whitespace is immediately
    # followed by one of the open-ended markers, so it can't misfire on
    # two unrelated adjacent dates that just happen to be space-separated.
    rf"(?:\s*[-–—]\s*|\s+to\s+|\s+(?=(?:Present|Current|Now|Till)\b))"
    rf"(?P<end>Present|Current|Now|Till\s*(?:Date|Now)|{_DAY_MONTH_YEAR}|{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})",
    re.IGNORECASE
)

_WIDOW_START_DATE_RE = re.compile(
    rf"({_DAY_MONTH_YEAR}|{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
    re.IGNORECASE
)
_WIDOW_END_DATE_RE = re.compile(
    rf"^[-–—]\s*(Present|Current|Now|Till\s*(?:Date|Now)|"
    rf"{_DAY_MONTH_YEAR}|{MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]\d{{2}})|\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})\s*$",
    re.IGNORECASE
)


_TRAILING_TO_RE = re.compile(r"\s*\bto\s*$", re.IGNORECASE)


def _merge_widow_date_lines(lines: list) -> list:
    merged = list(lines)
    for i in range(len(merged) - 1):
        cur = merged[i].rstrip()
        nxt = merged[i + 1].strip()
        if not cur or not nxt:
            continue
        if DATE_RANGE_RE.search(cur):
            continue
        if _WIDOW_START_DATE_RE.search(cur) and _WIDOW_END_DATE_RE.match(nxt):
            merged[i] = cur + " " + nxt
            merged[i + 1] = ""
            continue
        # A "to"-continuation widow: the date RANGE ITSELF splits across
        # two lines at the separator word, e.g.
        #   "...as a PHP Developer (Dec 2018 to"
        #   "Jun 2020)."
        # Confirmed on a real resume (Abhay Awasthi). Distinct from the
        # dash-continuation case above (that one has no "to" at all --
        # just a start date, then a line starting with "-EndDate"). Only
        # merges when the text before the trailing "to" itself ends in a
        # recognizable start date, so this can't misfire on an unrelated
        # sentence that happens to wrap at the word "to".
        to_match = _TRAILING_TO_RE.search(cur)
        if to_match:
            before_to = cur[:to_match.start()].rstrip()
            if _WIDOW_START_DATE_RE.search(before_to):
                merged[i] = cur + " " + nxt
                merged[i + 1] = ""
    return merged

TITLE_KEYWORDS = [
    "engineer", "developer", "manager", "analyst", "consultant",
    "designer", "architect", "lead", "head", "director", "vp", "svp", "cto", "ceo",
    "intern", "associate", "senior", "junior", "principal", "staff",
    "specialist", "coordinator", "executive", "officer", "president",
    "scientist", "researcher", "administrator", "technician", "professor",
    "assistant", "trainee", "fresher", "apprentice"
]
TITLE_RE = re.compile(r'\b(' + '|'.join(TITLE_KEYWORDS) + r')\b', re.IGNORECASE)

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


def _normalize_month_year_glue(text: str) -> str:
    return _MONTH_YEAR_GLUE_DASH_RE.sub(r"\1 \2", text)

BULLET_RE = re.compile(r'^[•●○◦▪➤►‣\-\*]\s+|^\d+[\.\)]?\s+')

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
    if not m:
        return date_str
    two_digit = int(m.group(1))
    century = "20" if two_digit <= 49 else "19"
    return date_str[:m.start()] + century + m.group(1) + date_str[m.end():]


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
    month, year = int(m.group(1)), int(m.group(2))
    if 1 <= month <= 12:
        return datetime(year, month, 1)
    return None


def parse_date(date_str: str):
    if not date_str or re.match(r'present|current|now|till\s*(?:date|now)', date_str, re.IGNORECASE):
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
    words = line.split()
    if len(words) > 6:
        return True
    last_word = words[-1].rstrip(".;").lower() if words else ""
    if line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS:
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
    if TITLE_RE.search(line) or " at " in line.lower():
        return False
    words = line.split()
    if not words or len(words) <= 6:
        return False
    last_word = words[-1].rstrip(".;").lower()
    return line.rstrip().endswith((".", ";")) and last_word not in _COMPANY_ABBREV_ENDINGS


def _find_job_blocks(lines: list):
    anchor_idx = [i for i, l in enumerate(lines) if DATE_RANGE_RE.search(l)]
    if not anchor_idx:
        anchor_idx = [
            i for i, l in enumerate(lines)
            if _WIDOW_START_DATE_RE.search(l) and i > 0 and (
                TITLE_RE.search(lines[i-1]) or " at " in lines[i-1].lower() or " – " in lines[i-1] or " - " in lines[i-1] or ":-" in lines[i-1]
            )
        ]
    if not anchor_idx:
        return []

    def header_start_for(anchor_i):
        start = anchor_i
        j = anchor_i - 1
        steps = 0
        while j >= 0 and steps < 2:
            l = lines[j].strip()
            if not l or BULLET_RE.match(l) or j in anchor_idx:
                break
            if l[0].islower():
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


def _parse_header(header_lines: list, date_line: str):
    job = {"role": None, "company": None, "location": None, "start": None, "end": None, "method": []}

    m = DATE_RANGE_RE.search(date_line)
    if m:
        job["start"] = m.group("start").strip()
        job["end"] = m.group("end").strip()
        job["method"].append("date_regex")
    else:
        sm = _WIDOW_START_DATE_RE.search(date_line)
        if sm:
            job["start"] = sm.group(1).strip()
            job["end"] = "Present"
            job["method"].append("single_date_regex")

    cleaned = []
    for l in header_lines:
        l = l.strip()
        if not l:
            continue
        dm = DATE_RANGE_RE.search(l)
        if dm:
            l = (l[:dm.start()] + l[dm.end():])
            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
        sm = _WIDOW_START_DATE_RE.search(l)
        if sm:
            l = (l[:sm.start()] + l[sm.end():])
            l = re.sub(r'^[\s\-–—,/|]+|[\s\-–—,/|]+$', '', l)
        if l:
            cleaned.append(l)

    if cleaned and "/" in cleaned[0] and "|" not in cleaned[0]:
        parts = [p.strip() for p in cleaned[0].split("/") if p.strip()]
        if parts:
            job["role"] = parts[0]
            paren = re.search(r'\(([^)]+)\)', cleaned[0])
            if paren:
                job["company"] = paren.group(1).strip()
                job["method"].append("inline_parens")

    if not job["role"] and not job["company"] and cleaned and "|" in cleaned[0]:
        parts = [p.strip() for p in cleaned[0].split("|") if p.strip()]
        title_part = next((p for p in parts if TITLE_RE.search(p)), None)
        if title_part:
            job["role"] = title_part
            other_parts = [p for p in parts if p != title_part]
            if other_parts:
                job["company"] = other_parts[0]
            job["method"].append("pipe_split")

    if not job["role"] and not job["company"] and cleaned:
        for sep in (" – ", " — ", " - "):
            if sep in cleaned[0]:
                left, right = (p.strip() for p in cleaned[0].split(sep, 1))
                left = re.sub(r'\([^)]*\)', '', left).strip()
                right = re.sub(r'\([^)]*\)', '', right).strip()
                if left and right:
                    left_is_title = bool(TITLE_RE.search(left))
                    right_is_title = bool(TITLE_RE.search(right))
                    if right_is_title and not left_is_title:
                        job["role"], job["company"] = right, left
                    else:
                        job["role"], job["company"] = left, right
                    job["method"].append("dash_split")
                break

    if not job["role"] and not job["company"] and cleaned:
        at_match = re.search(r'^(.*?)\s+\bat\b\s+(.*)$', cleaned[0], re.IGNORECASE)
        if at_match:
            role_candidate = at_match.group(1).strip()
            comp_candidate = at_match.group(2).strip()
            comp_candidate = re.sub(r'[:\-–—\s]*\d+\s*(?:yrs?|years?|mos?|months?).*$', '', comp_candidate, flags=re.IGNORECASE).strip()
            if role_candidate and comp_candidate:
                job["role"] = role_candidate
                job["company"] = comp_candidate
                job["method"].append("at_split")

    if not job["role"] and not job["company"] and cleaned:
        paren_match = re.search(r'\(([^)]+)\)', cleaned[0])
        if paren_match:
            inside = paren_match.group(1).strip()
            outside = (cleaned[0][:paren_match.start()] + cleaned[0][paren_match.end():]).strip()
            if outside and TITLE_RE.search(inside):
                job["role"] = inside
                job["company"] = outside
                job["method"].append("paren_title")

    if not job["role"] and not job["company"]:
        for line in cleaned[:3]:
            line = re.sub(r'\([^)]*\)', '', line).strip()
            if not line:
                continue
            if TITLE_RE.search(line) and not job["role"]:
                job["role"] = line
                job["method"].append("title_dict")
            elif not job["company"]:
                job["company"] = line

    if job["company"] and "," in job["company"]:
        parts = [p.strip() for p in job["company"].rsplit(",", 1)]
        if len(parts) == 2 and len(parts[1].split()) <= 3:
            job["company"], job["location"] = parts

    if job["company"] and " - " in job["company"]:
        left, _, right = job["company"].rpartition(" - ")
        left, right = left.strip(), right.strip()
        if left and right and len(right.split()) <= 4:
            job["company"] = left
            job["location"] = f"{right}, {job['location']}" if job["location"] else right

    if not job["company"] and cleaned:
        spacy_company = detect_company_with_spacy(" ".join(cleaned[:3]))
        if spacy_company:
            job["company"] = spacy_company
            job["method"].append("spacy_ORG")

    if not job["role"] and not job["company"] and cleaned:
        job["role"] = cleaned[0]

    for key in ("role", "company"):
        if job[key]:
            job[key] = job[key].strip(" /-").strip()

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

    date_line = next((l for l in header_lines if DATE_RANGE_RE.search(l) or _WIDOW_START_DATE_RE.search(l)), "")
    job = _parse_header(header_lines + extra_header_lines, date_line)

    job["duration_months"] = calculate_duration_months(job["start"], job["end"]) if job["start"] else 0
    job["bullets"] = bullets
    job["sub_sections"] = subheadings

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
    blocks = _find_job_blocks(lines)

    jobs = []
    for header_ids, body_ids in blocks:
        header_lines = [lines[i] for i in header_ids]
        body_lines = [lines[i] for i in body_ids]
        parsed = parse_job_block(header_lines, body_lines)
        if parsed:
            jobs.append(parsed)
    return jobs


def calculate_total_experience(jobs: list) -> dict:
    parsed_jobs = []
    for job in jobs:
        if not job.get("start") or not job.get("end"):
            continue
        s = parse_date(job["start"])
        e = parse_date(job["end"])
        raw_end = str(job.get("end", "")).strip().lower()
        is_present = raw_end in ("present", "current", "now", "till date", "till now")
        if s and e and e > s:
            parsed_jobs.append({"start": s, "end": e, "is_present": is_present, "raw": job})

    if not parsed_jobs:
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

    total_months = sum(_inclusive_month_diff(s, e) for s, e in merged)

    return {
        "total_years": round(total_months / 12, 1),
        "total_months": total_months,
        "overlapping_periods_merged": len(intervals) - len(merged),
    }


def extract_experience(experience_section_text: str, early_career_section_text: str = "") -> dict:
    """
    Parses the main "experience" section as before. If early_career_section_text
    is also given (from a section the splitter labeled "early_career" --
    a condensed one-line-per-job recap like "LAST 5 CAREER TIMELINE"),
    it's parsed with the same anchor-based logic and kept in its own
    "early_career" list rather than merged into "experience", so the
    detailed job list isn't polluted with duplicate condensed entries.

    Both lists ARE combined, however, for `_total_experience`: the same
    jobs are often restated in both places (confirmed on a real resume,
    Sachin Chitranshi, where 5 jobs appear both in a condensed timeline
    AND in full detail later), and calculate_total_experience() already
    merges overlapping/identical date intervals, so restating a job in
    both sections does not inflate the total years figure -- it only
    helps in the case where the condensed recap includes an older job
    that isn't detailed anywhere else in the document.
    """
    empty_total = {"total_years": 0.0, "total_months": 0, "overlapping_periods_merged": 0}

    jobs = _extract_jobs_from_text(experience_section_text)
    for j in jobs:
        j["source"] = "experience"

    early_career_jobs = _extract_jobs_from_text(early_career_section_text)
    for j in early_career_jobs:
        j["source"] = "early_career"

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

    return {
        "experience": jobs,
        "early_career": early_career_jobs,
        "_confidence": avg_conf,
        "_needs_review": needs_review,
        "_total_experience": calculate_total_experience(combined),
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