import re
import difflib
from dataclasses import dataclass


def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
    core = "|".join(core_alternatives)
    return re.compile(
        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
        re.IGNORECASE
    )


_SKILL_PREFIX_WORDS = (
    r"(technical|core|key|professional|functional|domain|business|"
    r"soft|hard|tech|it|general|primary|specialized|relevant)"
)
_SKILLS_PATTERN = re.compile(
    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
    rf"|technical\s+specialization"
    rf"|functional\s+specialization"
    rf"|technology\s+stack"
    rf"|technology\s+summary"
    rf"|knowledge\s+summary"
    rf"|knowledge\s+base"
    rf"|technical\s+snapshot)$",
    re.IGNORECASE
)

_PROJECTS_PATTERN = re.compile(
    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
    r"|portfolio"
    r"|projects?\s*#?\s*\d+)$",
    re.IGNORECASE
)

_EXPERIENCE_DATE_RANGE_TAIL = (
    r"(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*"
    r"(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?"
)
_EXPERIENCE_PATTERN = re.compile(
    rf"^(?:((work|industry|relevant|professional|previous|past|prior)\s*(and\s*)?){{0,2}}experience"
    rf"|employment(\s+(history|details|records?|background))?"
    rf"|professional\s*background"
    rf"|career\s*history|work\s*history|internships?"
    rf"|corporate\s+success|career\s+journey|professional\s+journey"
    rf"|career\s+chronology|employment\s+timeline)"
    rf"{_EXPERIENCE_DATE_RANGE_TAIL}$",
    re.IGNORECASE
)


SECTION_PATTERNS = {
    "summary": _build_section_pattern([
        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
        r"objective", r"profile",
        r"about\s+me", r"career\s+objective", r"personal\s+statement",
    ]),
    "experience": _EXPERIENCE_PATTERN,
    "education": _build_section_pattern([
        r"education(al)?(\s+background)?",
        r"academics?(\s+background)?",
        r"academic\s+qualifications?", r"educational\s+qualifications?",
        r"qualifications?",
        r"degrees?", r"university", r"college",
    ]),
    "skills": _SKILLS_PATTERN,
    "projects": _PROJECTS_PATTERN,
    "certifications": _build_section_pattern([
        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
        r"accreditations?", r"credentials?", r"professional\s+certifications?",
        r"trainings?",
    ]),
    "achievements": _build_section_pattern([
        r"(key|major|notable|special|top)\s+achievements?",
        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
        r"accomplishments?", r"accolades?",
        r"rewards?",
    ]),
    "languages": _build_section_pattern([
        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
        r"language\s+skills",
    ]),
    "interests": _build_section_pattern([
        r"interests?", r"hobbies", r"activities",
    ]),
    "other": _build_section_pattern([
        r"custom\s+section", r"additional\s+information",
        r"miscellaneous", r"additional\s+details",
    ]),
    "strengths": _build_section_pattern([
        r"(key\s+)?strengths?", r"core\s+strengths?",
    ]),
    "personal_details": _build_section_pattern([
        r"personal\s+(details|information|profile|data)",
        r"bio\s*-?\s*data",
    ]),
    "declaration": _build_section_pattern([
        r"declaration", r"self[\s-]?declaration",
    ]),
    "roles_responsibilities": re.compile(
        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
        re.IGNORECASE
    ),
    "early_career": re.compile(
        r"^(?:early\s+career(s)?"
        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
        r"|career\s+synopsis"
        r"|career\s+snapshot"
        r"|career\s+at\s+a\s+glance"
        r"|(prior|past|previous)\s+engagements?)$",
        re.IGNORECASE
    ),
}


_LIST_SECTION_LABELS = {
    "skills", "education", "certifications", "achievements",
    "languages", "projects", "interests", "strengths", "experience",
    "early_career",
}


_SECTION_KEYWORDS = [
    "skill", "experience", "education", "project", "certif", "licen",
    "achievement", "award", "honor", "honour", "summary", "objective",
    "qualification", "employment", "career", "academic", "competenc",
    "expertise", "technolog", "portfolio", "credential", "accreditation",
    "recognition", "accomplishment", "language", "interest", "hobbies",
    "research", "leadership", "internship", "training", "volunteer",
    "publication", "reference", "extracurricular", "strength",
    "personal", "declaration", "responsibilit",
    "corporate", "success", "journey", "chronology", "timeline",
    "reward", "synopsis", "snapshot", "glance", "engagement",
]


_KEYWORD_TO_LABEL = {
    "summary": "summary", "objective": "summary",
    "skill": "skills", "expertise": "skills", "competenc": "skills",
    "career timeline": "early_career", "career synopsis": "early_career",
    "career snapshot": "early_career", "career at a glance": "early_career",
    "early career": "early_career", "timeline": "early_career",
    "synopsis": "early_career", "snapshot": "early_career",
    "experience": "experience", "employment": "experience", "career": "experience",
    "internship": "experience",
    "education": "education", "qualification": "education", "academic": "education",
    "project": "projects", "portfolio": "projects",
    "certif": "certifications", "licen": "certifications", "credential": "certifications",
    "accreditation": "certifications", "training": "certifications",
    "achievement": "achievements", "award": "achievements", "honor": "achievements",
    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
    "reward": "achievements",
    "language": "languages",
    "interest": "interests", "hobbies": "interests",
    "strength": "strengths",
    "declaration": "declaration",
    "personal": "personal_details",
    "responsibilit": "roles_responsibilities",
}


@dataclass
class Section:
    label: str
    raw_text: str
    start_line: int
    confidence: float


def _clean_heading_candidate(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
    cleaned = cleaned.strip(":-—–_ ")
    return cleaned


def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
    if " " in candidate:
        return False
    if candidate != candidate.lower():
        return False
    for prev in reversed(prev_lines):
        prev_stripped = prev.strip()
        if prev_stripped:
            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
    return False


def split_inline_heading(line: str):
    stripped = line.strip()
    for separator in ["•", "●", ":", "-", "–", "—"]:
        if separator not in stripped:
            continue
        heading_candidate = stripped.split(separator, 1)[0].strip()
        if not heading_candidate:
            continue
        for label, pattern in SECTION_PATTERNS.items():
            if pattern.match(heading_candidate):
                remaining = stripped[len(heading_candidate):].strip()
                remaining = remaining.lstrip("•●:-–— ").strip()
                return heading_candidate, remaining
    return None, None


def normalize_inline_bullets(text: str) -> str:
    normalized = []
    for raw_line in text.splitlines():
        bullet_count = raw_line.count("•") + raw_line.count("●")
        if bullet_count <= 1:
            normalized.append(raw_line)
            continue
        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
        for piece in pieces:
            normalized.append(f"• {piece}")
    return "\n".join(normalized)


_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
_BARE_PHONE_LINE = re.compile(
    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
    re.IGNORECASE,
)


def _is_bare_contact_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _JOB_DATE_RANGE.search(stripped):
        return False
    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))


_JOB_MONTH_NAMES = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"

_JOB_DATE_RANGE = re.compile(
    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
    rf"\s*(?:[-–—\u2013\u2014]+\s*|\bto\b\s*|(?=(?:Current|Present|Now|Till)\b))"
    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
    re.IGNORECASE
)


_FUZZY_HEADING_KEYWORDS = {
    "experience": "experience",
    "summary": "summary",
    "objective": "summary",
    "profile": "summary",
    "education": "education",
    "skills": "skills",
    "projects": "projects",
    "certifications": "certifications",
    "achievements": "achievements",
    "languages": "languages",
    "interests": "interests",
    "declaration": "declaration",
    "strengths": "strengths",
}


def _fuzzy_heading_label(candidate: str):
    word = candidate.strip().lower()
    if not word.isalpha() or len(word) < 5:
        return None
    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
        if abs(len(word) - len(keyword)) > 2:
            continue
        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
            return label
    return None


_SUBFIELD_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z \t]{0,40}?)\s*:")


def _experience_subfield_label(stripped: str):
    m = _SUBFIELD_LABEL_RE.match(stripped)
    if not m:
        return None
    return m.group(1).strip().lower()


def _looks_like_job_block_header(stripped: str, idx: int, lines: list) -> bool:
    if not stripped or len(stripped) > 60:
        return False
    found_role = False
    found_org = False
    checked = 0
    j = idx + 1
    while j < len(lines) and checked < 4:
        s = lines[j].strip()
        if s:
            checked += 1
            label = _experience_subfield_label(s)
            if label in ("role", "designation"):
                found_role = True
            elif label in ("organization", "organisation", "company", "employer"):
                found_org = True
            if found_role and found_org:
                return True
        j += 1
    return False


def _looks_like_job_entry(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    match = _JOB_DATE_RANGE.search(stripped)
    if not match:
        return False
    return bool(_MONTH_OR_ONGOING_RE.search(match.group()))


_NON_COMPANY_LINE_PREFIXES = re.compile(
    r"^(duration|project|role|project\s*#|project\s+title|project\s+description)\b",
    re.IGNORECASE
)

_MONTH_OR_ONGOING_RE = re.compile(
    rf"{_JOB_MONTH_NAMES}|present|current|now|till",
    re.IGNORECASE
)


def _looks_like_company_entry(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 150:
        return False
    if _NON_COMPANY_LINE_PREFIXES.match(stripped):
        return False
    match = _JOB_DATE_RANGE.search(stripped)
    if not match:
        return False
    if not _MONTH_OR_ONGOING_RE.search(match.group()):
        return False
    prefix = stripped[:match.start()].strip(" \t:-")
    return len(prefix.split()) >= 2


def _next_nonblank_line(lines: list, start_index: int) -> str:
    i = start_index
    while i < len(lines):
        if lines[i].strip():
            return lines[i].strip()
        i += 1
    return ""


_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")


def _looks_like_caps_heading_shape(stripped: str) -> bool:
    return bool(
        stripped == stripped.upper()
        and _CAPS_HEADING_SHAPE.match(stripped)
        and len(stripped.split()) <= 4
        and not stripped.endswith(".")
    )


def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
    upcoming = _next_nonblank_line(all_lines, line_index + 1)
    return _looks_like_job_entry(upcoming)


def _looks_like_bare_company_line(stripped: str) -> bool:
    if not stripped or len(stripped) > 60:
        return False
    if ":" in stripped:
        return False
    if stripped.endswith((".", ",")):
        return False
    if _JOB_DATE_RANGE.search(stripped):
        return False
    if len(stripped.split()) > 6:
        return False
    return not stripped[0].islower()


def _looks_like_date_prefixed_job_entry(line: str) -> bool:
    """
    Detects the REVERSED job-entry format some resumes use: the date
    range comes FIRST, followed by a colon, then the company/role info
    -- e.g. "Nov'24 - Sep'25: Kalp Digital Infra Pvt. Ltd., Noida as
    Engineering Manager". _looks_like_company_entry() assumes the
    company name comes BEFORE the date (its own prefix-word-count check
    requires >=2 words before the date match starts), so it never
    recognizes this reversed shape. This is a separate, narrower check
    used only by _achievements_heading_is_injob() below.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 150:
        return False
    match = _JOB_DATE_RANGE.match(stripped)  # must start at position 0
    if not match:
        return False
    if not _MONTH_OR_ONGOING_RE.search(match.group()):
        return False
    remainder = stripped[match.end():].lstrip()
    return remainder.startswith(":")


_NON_EXPERIENCE_STOP_LABELS = {
    "education", "certifications", "skills", "languages",
    "personal_details", "declaration", "interests", "strengths",
    "summary", "projects",
}


def _achievements_heading_is_injob(line_index: int, lines: list, lookahead: int = 25) -> bool:
    """
    Confirmed on a real resume (Abhishek Kumar): "Key Achievements" is
    used repeatedly as an IN-JOB subheading -- one appears after EVERY
    job entry's intro paragraph, listing that specific role's
    achievements, before the NEXT job entry starts. Because "Key
    Achievements" (and plain "Achievements") is ALSO a perfectly valid
    TOP-LEVEL section heading on its own (a standalone, career-wide
    achievements list), text pattern alone can't distinguish the two --
    both match the exact same SECTION_PATTERNS regex with full (0.95)
    confidence, so the existing "only override a LOW-confidence match"
    guard elsewhere in this file doesn't catch this case at all.

    The distinguishing signal is what comes right AFTER the bullets
    that follow this heading: if another job entry shows up (checked
    here via _looks_like_date_prefixed_job_entry() for this resume's
    date-first format, and _looks_like_company_entry() for the more
    common company-first format) before the resume clearly moves into
    territory that has nothing to do with work history, the
    "achievements" heading just seen was for that ongoing job listing,
    not a new resume section on its own.

    REVISION: the lookahead originally stopped at the FIRST confidently
    matched heading of ANY kind. Confirmed on the SAME real resume this
    is too eager -- the last job's "Key Achievements" bullets are
    followed by a genuine "EARLY CAREER" heading (itself just a
    continuation of work history: a brief list of older jobs), and
    stopping there caused this function to wrongly conclude the "Key
    Achievements" heading must be a real top-level section, splitting
    that job's achievements away from "experience" even though "EARLY
    CAREER" has nothing to do with that decision. Now only a heading
    whose label is in _NON_EXPERIENCE_STOP_LABELS -- something that
    unambiguously signals the resume has left the work-history block
    entirely (education, skills, certifications, etc.) -- stops the
    lookahead early. A heading like "achievements" or "early_career"
    found along the way is compatible with still being inside an
    ongoing experience block, so scanning continues past it.

    lookahead=25 is generous on purpose: a single job's achievements
    list can run to 6+ bullets before the next job entry appears (seen
    on the real resume this was built against), so a short window would
    miss the very case this function exists to catch.
    """
    limit = min(line_index + 1 + lookahead, len(lines))
    for j in range(line_index + 1, limit):
        candidate = lines[j].strip()
        if not candidate:
            continue
        if _looks_like_date_prefixed_job_entry(candidate) or _looks_like_company_entry(candidate):
            return True
        confident_label, confidence = detect_section_label(candidate, prev_lines=lines[:j])
        if confident_label in _NON_EXPERIENCE_STOP_LABELS and confidence >= 0.9:
            return False
    return False


_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")


def _try_two_line_heading(lines: list, i: int):
    if i + 1 >= len(lines):
        return None, 0.0
    first = lines[i].strip()
    second = lines[i + 1].strip()
    if not first or not second:
        return None, 0.0
    if _LEADING_BULLET_RE.match(second):
        return None, 0.0
    if _is_wrapped_word(first, lines[:i]):
        return None, 0.0

    joined = _clean_heading_candidate(f"{first} {second}")
    for label, pattern in SECTION_PATTERNS.items():
        if pattern.match(joined):
            return label, 0.9
    return None, 0.0


def detect_section_label(line: str, prev_lines: list = None) -> tuple:
    if prev_lines is None:
        prev_lines = []
    candidate = _clean_heading_candidate(line)
    if not candidate:
        return None, 0.0
    for label, pattern in SECTION_PATTERNS.items():
        if pattern.match(candidate):
            if _is_wrapped_word(candidate, prev_lines):
                return None, 0.0
            return label, 0.95
    return None, 0.0


def _contains_section_keyword(line: str) -> bool:
    lower = line.lower()
    return any(keyword in lower for keyword in _SECTION_KEYWORDS)


def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
    stripped = line.strip()
    if len(stripped) == 0:
        return 0.0
    if stripped[:1] in ("•", "●"):
        return 0.0
    if line_index <= 1:
        return 0.0
    word_count = len(stripped.split())
    if word_count > 8 or len(stripped) > 60:
        return 0.0
    if stripped.endswith("."):
        return 0.0
    score = 0.0
    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
    if is_all_caps:
        score += 0.3
    if word_count <= 4:
        score += 0.2
    elif word_count <= 6:
        score += 0.1
    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
    if blank_below:
        score += 0.2
    if blank_above:
        score += 0.15
    if stripped.endswith(":"):
        score += 0.15
    if "," not in stripped and ". " not in stripped:
        score += 0.1
    if _contains_section_keyword(stripped):
        score += 0.35
    else:
        if not blank_below and not blank_above:
            score *= 0.5

    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
        score *= 0.3

    return min(score, 1.0)


_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")


def _is_letter_spaced_heading(line: str) -> bool:
    tokens = line.split()
    if len(tokens) < 4:
        return False
    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
    return (single_char / len(tokens)) >= 0.7


def _resolve_letter_spaced_heading(line: str):
    if not _is_letter_spaced_heading(line):
        return None
    collapsed = "".join(line.split()).lower()
    if len(collapsed) < 4:
        return None
    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
        if keyword in collapsed:
            return mapped_label
    return None


_TECH_STACK_KEYWORDS = {
    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
}

_SPOKEN_LANGUAGE_KEYWORDS = {
    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
}

_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")


def _looks_like_tech_stack_content(text: str) -> bool:
    words = _WORD_TOKEN_RE.findall(text.lower())
    return any(w in _TECH_STACK_KEYWORDS for w in words)


def _looks_like_spoken_language_content(text: str) -> bool:
    words = _WORD_TOKEN_RE.findall(text.lower())
    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)


def _resolve_unknown_heading(stripped: str):
    colon_idx = stripped.find(":")
    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
        return None

    if len(stripped.split()) > 4:
        return None

    lower = stripped.lower()
    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
        if keyword in lower:
            if mapped_label == "roles_responsibilities" and "role" not in lower:
                continue
            return mapped_label
    return None


def _find_embedded_heading_split(line: str):
    stripped = line.strip()
    words = stripped.split()
    if len(words) < 2:
        return None, None, None
    max_n = min(4, len(words) - 1)
    for n in range(max_n, 0, -1):
        tail_words = words[-n:]
        if not all(w[0].isupper() for w in tail_words if w[:1].isalpha()):
            continue
        candidate_clean = _clean_heading_candidate(" ".join(tail_words))
        for label, pattern in SECTION_PATTERNS.items():
            if label == "interests" and n == 1:
                continue
            if pattern.match(candidate_clean):
                prefix = " ".join(words[:-n]).strip()
                if prefix:
                    return prefix, " ".join(tail_words), label
    return None, None, None


def split_into_sections(text: str) -> list:
    lines = text.split("\n")
    sections = []
    current_label = "header"
    current_start = 0
    current_lines = []
    current_confidence = 0.9
    experience_seen = False
    deferred_contact_lines = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        stripped = line.strip()
        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
        inline_content = None

        if label is not None and stripped != "" and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if (nxt and not _LEADING_BULLET_RE.match(nxt)
                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
                combined = _clean_heading_candidate(f"{stripped} {nxt}")
                for combined_label, pattern in SECTION_PATTERNS.items():
                    if pattern.match(combined):
                        label = combined_label
                        confidence = max(confidence, 0.9)
                        skip_next = True
                        break

        if label is None and stripped != "":
            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
            if two_line_label is not None:
                label = two_line_label
                confidence = two_line_confidence
                skip_next = True

        if label is None and stripped != "":
            if current_label not in _LIST_SECTION_LABELS:
                heading_candidate, remaining = split_inline_heading(stripped)
                if heading_candidate is not None:
                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
                    if label is not None:
                        inline_content = remaining

        if label is None and stripped != "":
            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
            if letter_spaced_label is not None:
                label = letter_spaced_label
                confidence = 0.85

        if label is None and stripped != "":
            if not _is_wrapped_word(stripped, lines[:i]):
                fuzzy_label = _fuzzy_heading_label(stripped)
                if fuzzy_label is not None:
                    label = fuzzy_label
                    confidence = 0.8

        if label is None and stripped != "":
            if not _is_wrapped_word(stripped, lines[:i]):
                # GUARD (see REVISION note below): none of these three
                # "no heading word at all, just infer a new job entry
                # from shape" detectors are allowed to fire while we're
                # currently inside an "education" section. A school
                # name followed by a graduation/completion date range
                # ("Ghaziabad, UP — B.Tech" / "Aug 2018 - Aug 2022 |
                # 7.8 SGPA") is text-shape-identical to a company name
                # followed by an employment date range -- both are
                # short, capitalized, no colon, followed by a line with
                # a month name and a year range. Confirmed on a real
                # resume (Abhinav Srivastav): this caused the rest of
                # the EDUCATION section (B.Tech dates, two school
                # entries with their own dates) to get pulled out into
                # a bogus "experience" section, merging education
                # content into experience. Once inside education, a
                # short capitalized line followed by a date-shaped line
                # is overwhelmingly more likely to be "school name,
                # continued" than a genuine unheaded job entry -- so
                # skip promotion here and let a real, confidently-
                # matched EXPERIENCE heading (still handled normally by
                # detect_section_label() above, unaffected by this
                # guard) be what actually starts a new experience
                # section after education.
                if current_label == "education":
                    pass
                elif _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
                    label = "experience"
                    confidence = 0.85
                elif _looks_like_bare_company_line(stripped) and _next_line_is_job_entry(i, lines):
                    label = "experience"
                    confidence = 0.85
                    inline_content = stripped
                elif _looks_like_company_entry(stripped):
                    label = "experience"
                    confidence = 0.85
                    inline_content = stripped

        if label is None and stripped != "":
            if not _is_wrapped_word(stripped, lines[:i]):
                if _looks_like_job_block_header(stripped, i, lines):
                    label = "experience"
                    confidence = 0.9

        if label is None and stripped != "":
            if not _is_wrapped_word(stripped, lines[:i]):
                heading_score = score_heading_line(stripped, i, lines)
                if heading_score >= 0.65:
                    resolved = _resolve_unknown_heading(stripped)
                    if resolved is not None:
                        label = resolved
                        confidence = max(heading_score, 0.8)
                    else:
                        label = "unknown"
                        confidence = heading_score

        if label is None and stripped != "":
            embed_prefix, embed_heading_text, embed_label = _find_embedded_heading_split(stripped)
            if embed_label is not None and embed_label != current_label:
                current_lines.append(embed_prefix)
                label = embed_label
                confidence = 0.85

        if label == "languages" and stripped != "":
            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
                label = None
                confidence = 0.0
                inline_content = None

        if label == "summary" and not experience_seen and stripped != "":
            upcoming = _next_nonblank_line(lines, i + 1)
            if _looks_like_job_entry(upcoming):
                label = "experience"
                confidence = 0.85

        if current_label == "projects" and experience_seen and stripped != "":
            if _looks_like_company_entry(stripped):
                label = "experience"
                confidence = 0.9
                inline_content = stripped

        if label is not None and label == current_label:
            current_lines.append(line)
            continue

        if current_label in _LIST_SECTION_LABELS and label == "unknown":
            promoted_label = _resolve_unknown_heading(stripped)
            if promoted_label is None:
                current_lines.append(line)
                continue
            label = promoted_label
            confidence = 0.8

        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
            # NEW GUARD (see _achievements_heading_is_injob docstring):
            # a confident "achievements"/"Key Achievements" match while
            # already inside experience gets a lookahead check BEFORE
            # the old confidence<0.9 rule below even runs -- this is
            # the one case that rule was letting through, because an
            # exact "Key Achievements" heading scores full (0.95)
            # confidence via detect_section_label(), not the low
            # confidence the old rule assumed an in-job subheading
            # would have.
            if label == "achievements" and _achievements_heading_is_injob(i, lines):
                current_lines.append(line)
                continue
            if confidence < 0.9 and (label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-")):
                current_lines.append(line)
                continue

        if label is not None and stripped != "":
            if label == "experience":
                experience_seen = True
            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
            if section_text:
                sections.append(Section(
                    label=current_label,
                    raw_text=section_text,
                    start_line=current_start,
                    confidence=current_confidence
                ))
            current_label = label
            current_start = i
            current_lines = []
            if inline_content:
                current_lines.append(inline_content)
            elif label == "unknown":
                current_lines.append(stripped)
            current_confidence = confidence
        else:
            if current_label != "header" and _is_bare_contact_line(stripped):
                deferred_contact_lines.append(stripped)
            else:
                current_lines.append(line)

    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
    if section_text:
        sections.append(Section(
            label=current_label,
            raw_text=section_text,
            start_line=current_start,
            confidence=current_confidence
        ))

    if deferred_contact_lines:
        for s in sections:
            if s.label == "header":
                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
                break
        else:
            sections.insert(0, Section(
                label="header",
                raw_text="\n".join(deferred_contact_lines),
                start_line=0,
                confidence=0.9,
            ))

    return sections


def get_section_text(sections: list, label: str) -> str:
    matching = [s for s in sections if s.label == label]
    if not matching:
        return ""
    return "\n\n".join([s.raw_text for s in matching])










##just commentimmng to fix achievemnet section- worked-
#import re
#import difflib
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#_EXPERIENCE_DATE_RANGE_TAIL = (
#    r"(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*"
#    r"(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?"
#)
#_EXPERIENCE_PATTERN = re.compile(
#    rf"^(?:((work|industry|relevant|professional|previous|past|prior)\s*(and\s*)?){{0,2}}experience"
#    rf"|employment(\s+(history|details|records?|background))?"
#    rf"|professional\s*background"
#    rf"|career\s*history|work\s*history|internships?"
#    rf"|corporate\s+success|career\s+journey|professional\s+journey"
#    rf"|career\s+chronology|employment\s+timeline)"
#    rf"{_EXPERIENCE_DATE_RANGE_TAIL}$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _EXPERIENCE_PATTERN,
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills",
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
#_JOB_DATE_RANGE = re.compile(
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|\bto\b\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#_SUBFIELD_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z \t]{0,40}?)\s*:")
#
#
#def _experience_subfield_label(stripped: str):
#    m = _SUBFIELD_LABEL_RE.match(stripped)
#    if not m:
#        return None
#    return m.group(1).strip().lower()
#
#
#def _looks_like_job_block_header(stripped: str, idx: int, lines: list) -> bool:
#    if not stripped or len(stripped) > 60:
#        return False
#    found_role = False
#    found_org = False
#    checked = 0
#    j = idx + 1
#    while j < len(lines) and checked < 4:
#        s = lines[j].strip()
#        if s:
#            checked += 1
#            label = _experience_subfield_label(s)
#            if label in ("role", "designation"):
#                found_role = True
#            elif label in ("organization", "organisation", "company", "employer"):
#                found_org = True
#            if found_role and found_org:
#                return True
#        j += 1
#    return False
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    match = _JOB_DATE_RANGE.search(stripped)
#    if not match:
#        return False
#    return bool(_MONTH_OR_ONGOING_RE.search(match.group()))
#
#
#_NON_COMPANY_LINE_PREFIXES = re.compile(
#    r"^(duration|project|role|project\s*#|project\s+title|project\s+description)\b",
#    re.IGNORECASE
#)
#
#_MONTH_OR_ONGOING_RE = re.compile(
#    rf"{_JOB_MONTH_NAMES}|present|current|now|till",
#    re.IGNORECASE
#)
#
#
#def _looks_like_company_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 150:
#        return False
#    if _NON_COMPANY_LINE_PREFIXES.match(stripped):
#        return False
#    match = _JOB_DATE_RANGE.search(stripped)
#    if not match:
#        return False
#    if not _MONTH_OR_ONGOING_RE.search(match.group()):
#        return False
#    prefix = stripped[:match.start()].strip(" \t:-")
#    return len(prefix.split()) >= 2
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#def _looks_like_bare_company_line(stripped: str) -> bool:
#    if not stripped or len(stripped) > 60:
#        return False
#    if ":" in stripped:
#        return False
#    if stripped.endswith((".", ",")):
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    if len(stripped.split()) > 6:
#        return False
#    return not stripped[0].islower()
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    if len(stripped.split()) > 4:
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            if mapped_label == "roles_responsibilities" and "role" not in lower:
#                continue
#            return mapped_label
#    return None
#
#
#def _find_embedded_heading_split(line: str):
#    stripped = line.strip()
#    words = stripped.split()
#    if len(words) < 2:
#        return None, None, None
#    max_n = min(4, len(words) - 1)
#    for n in range(max_n, 0, -1):
#        tail_words = words[-n:]
#        if not all(w[0].isupper() for w in tail_words if w[:1].isalpha()):
#            continue
#        candidate_clean = _clean_heading_candidate(" ".join(tail_words))
#        for label, pattern in SECTION_PATTERNS.items():
#            if label == "interests" and n == 1:
#                continue
#            if pattern.match(candidate_clean):
#                prefix = " ".join(words[:-n]).strip()
#                if prefix:
#                    return prefix, " ".join(tail_words), label
#    return None, None, None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                # GUARD (see REVISION note below): none of these three
#                # "no heading word at all, just infer a new job entry
#                # from shape" detectors are allowed to fire while we're
#                # currently inside an "education" section. A school
#                # name followed by a graduation/completion date range
#                # ("Ghaziabad, UP — B.Tech" / "Aug 2018 - Aug 2022 |
#                # 7.8 SGPA") is text-shape-identical to a company name
#                # followed by an employment date range -- both are
#                # short, capitalized, no colon, followed by a line with
#                # a month name and a year range. Confirmed on a real
#                # resume (Abhinav Srivastav): this caused the rest of
#                # the EDUCATION section (B.Tech dates, two school
#                # entries with their own dates) to get pulled out into
#                # a bogus "experience" section, merging education
#                # content into experience. Once inside education, a
#                # short capitalized line followed by a date-shaped line
#                # is overwhelmingly more likely to be "school name,
#                # continued" than a genuine unheaded job entry -- so
#                # skip promotion here and let a real, confidently-
#                # matched EXPERIENCE heading (still handled normally by
#                # detect_section_label() above, unaffected by this
#                # guard) be what actually starts a new experience
#                # section after education.
#                if current_label == "education":
#                    pass
#                elif _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#                elif _looks_like_bare_company_line(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#                    inline_content = stripped
#                elif _looks_like_company_entry(stripped):
#                    label = "experience"
#                    confidence = 0.85
#                    inline_content = stripped
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_job_block_header(stripped, i, lines):
#                    label = "experience"
#                    confidence = 0.9
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        if label is None and stripped != "":
#            embed_prefix, embed_heading_text, embed_label = _find_embedded_heading_split(stripped)
#            if embed_label is not None and embed_label != current_label:
#                current_lines.append(embed_prefix)
#                label = embed_label
#                confidence = 0.85
#
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if current_label == "projects" and experience_seen and stripped != "":
#            if _looks_like_company_entry(stripped):
#                label = "experience"
#                confidence = 0.9
#                inline_content = stripped
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if confidence < 0.9 and (label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-")):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            elif label == "unknown":
#                current_lines.append(stripped)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#











#worked - just commenting to fix the bug we see in education and experience section - written code above
#import re
#import difflib
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#_EXPERIENCE_DATE_RANGE_TAIL = (
#    r"(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*"
#    r"(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?"
#)
#_EXPERIENCE_PATTERN = re.compile(
#    rf"^(?:((work|industry|relevant|professional|previous|past|prior)\s*(and\s*)?){{0,2}}experience"
#    rf"|employment(\s+(history|details|records?|background))?"
#    rf"|professional\s*background"
#    rf"|career\s*history|work\s*history|internships?"
#    rf"|corporate\s+success|career\s+journey|professional\s+journey"
#    rf"|career\s+chronology|employment\s+timeline)"
#    rf"{_EXPERIENCE_DATE_RANGE_TAIL}$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _EXPERIENCE_PATTERN,
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills",
#    # NOTE: "technolog" was deliberately removed from this loose,
#    # substring-based fallback table. It's redundant for genuine cases --
#    # detect_section_label() already matches whole-line "Technology
#    # Stack" / "Technology Summary" headings via _SKILLS_PATTERN's own
#    # strict alternatives, and that strict path always runs first. Left
#    # in here, "technolog" matched as a bare substring, so any company
#    # name containing "Technology"/"Technologies" (extremely common,
#    # e.g. "MoveInSync Technology Solutions", "DXC Technology") was
#    # being misread as a new "skills" section heading and silently
#    # discarded as heading text -- real data loss. Confirmed on a real
#    # resume (Praveen Kumar Pedapapa).
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
#_JOB_DATE_RANGE = re.compile(
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|\bto\b\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
## --- Job-block header detection (Naukri-style "CAREER PROFILE: N"
## resumes) ---------------------------------------------------------
##
## Some templates don't head each job with "Company Name (dates)" on one
## line -- they use a bare index heading like "CAREER PROFILE: 6" and then
## spell out "Role:", "Organization:", "Description:" as separate labeled
## fields on the following lines. That heading doesn't match any
## SECTION_PATTERNS alternative, and score_heading_line's "colon followed
## by content" penalty (meant to suppress "Duration: 6 months"-style body
## lines) fires on the trailing digit and drags the score below the
## classification threshold anyway -- so the heading was invisible and
## the whole job block got silently absorbed into whatever section was
## currently open.
##
## Fixed with a content-based (not heading-text-based) rule, so it isn't
## tied to the literal words "career profile" and still works if a
## template calls it "Assignment 3" or similar: look ahead a few lines
## for "Role:"/"Designation:" AND "Organization:"/"Company:" fields -- if
## both appear, the current line is the start of a job entry regardless
## of what it says.
#
#_SUBFIELD_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z \t]{0,40}?)\s*:")
#
#
#def _experience_subfield_label(stripped: str):
#    m = _SUBFIELD_LABEL_RE.match(stripped)
#    if not m:
#        return None
#    return m.group(1).strip().lower()
#
#
#def _looks_like_job_block_header(stripped: str, idx: int, lines: list) -> bool:
#    if not stripped or len(stripped) > 60:
#        return False
#    found_role = False
#    found_org = False
#    checked = 0
#    j = idx + 1
#    while j < len(lines) and checked < 4:
#        s = lines[j].strip()
#        if s:
#            checked += 1
#            label = _experience_subfield_label(s)
#            if label in ("role", "designation"):
#                found_role = True
#            elif label in ("organization", "organisation", "company", "employer"):
#                found_org = True
#            if found_role and found_org:
#                return True
#        j += 1
#    return False
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    match = _JOB_DATE_RANGE.search(stripped)
#    if not match:
#        return False
#    # Same reasoning as _looks_like_company_entry() below: a bare
#    # "2005-07" year-only range is just as common in an education/
#    # qualifications table as a real job date, so require a month name
#    # or an ongoing-role word to actually call this a JOB date range.
#    # Confirmed on a real resume (Ritu Verma): the Qualifications table
#    # header row ("Examination  School/University  Year Of Passing
#    # Board") was being misread as a company-name-shaped line purely
#    # because the row right below it ("...2005-07...") satisfied this
#    # check without the guard, triggering a false switch out of the
#    # Education section.
#    return bool(_MONTH_OR_ONGOING_RE.search(match.group()))
#
#
#_NON_COMPANY_LINE_PREFIXES = re.compile(
#    r"^(duration|project|role|project\s*#|project\s+title|project\s+description)\b",
#    re.IGNORECASE
#)
#
## A bare "(2005-2009)" year-only range is just as common in EDUCATION
## entries ("BE with ECE - Anna University, Dharmapuri, India
## (2005-2009)") as it is in real employment entries, so a date range
## alone isn't a reliable "this is a company/job entry" signal. Real job
## entries in these resumes almost always name a month, or say
## Present/Current/Now/Till for an ongoing role -- require one of those
## to disambiguate.
#_MONTH_OR_ONGOING_RE = re.compile(
#    rf"{_JOB_MONTH_NAMES}|present|current|now|till",
#    re.IGNORECASE
#)
#
#
#def _looks_like_company_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 150:
#        return False
#    if _NON_COMPANY_LINE_PREFIXES.match(stripped):
#        return False
#    match = _JOB_DATE_RANGE.search(stripped)
#    if not match:
#        return False
#    if not _MONTH_OR_ONGOING_RE.search(match.group()):
#        # Confirmed on a real resume (Mohan Kumar K): without this
#        # guard, "BE with ECE - Anna university, Dharmapuri, India
#        # (2005-2009)" -- a plain education line -- was being misread
#        # as a new employer entry, because a bare year range looks
#        # identical in shape to a real job's date range.
#        return False
#    prefix = stripped[:match.start()].strip(" \t:-")
#    return len(prefix.split()) >= 2
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
## _looks_like_caps_heading_shape() above only recognizes a SHOUTY
## heading ("EXPERIENCE") sitting directly above a job-entry (date-range)
## line. Some templates instead put the bare, mixed-case company name
## itself right above the role+dates line -- e.g. "MoveInSync Technology
## Solutions" followed by "Senior Software Engineer _ Sept 2023 to
## Present." -- with no heading word at all. That line isn't a heading;
## it's the company name, i.e. real content, not something to discard.
## Confirmed on a real resume (Praveen Kumar Pedapapa): without this,
## the company line scored just high enough on generic heading heuristics
## to get pulled out as an "unknown" section, discarding real data.
#def _looks_like_bare_company_line(stripped: str) -> bool:
#    if not stripped or len(stripped) > 60:
#        return False
#    # Must have NO colon at all -- a "Label : Value" field like "Client
#    # Name : Baker Hughes" (common inside a Projects block, right above
#    # a "Tenure : <date range>" line) is short and capitalized too, and
#    # would otherwise false-positive here just as easily as a genuine
#    # bare company name. Confirmed on a real resume (Praveen Kumar
#    # Pedapapa): without this guard, every "Client Name : X" line in
#    # each of 4 project entries got wrongly split into its own bogus
#    # "experience" section, fragmenting the whole Projects block.
#    if ":" in stripped:
#        return False
#    if stripped.endswith((".", ",")):
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    if len(stripped.split()) > 6:
#        return False
#    return not stripped[0].islower()
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    if len(stripped.split()) > 4:
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            # roles_responsibilities' own strict SECTION_PATTERNS regex
#            # requires the word "role(s)" to be present alongside
#            # "responsibilit-" -- this loose substring fallback must
#            # honor that same requirement, or a bare in-job field label
#            # like "Responsibility:" / "Additional responsibility:"
#            # (which contains no "role" at all) gets misread as the
#            # start of a brand new top-level section. Confirmed on a
#            # real resume (Mohan Kumar K): each of 6 job blocks under
#            # "CAREER PROFILE: N" has its own "Responsibility:" field,
#            # and every one of them was incorrectly splitting off its
#            # own roles_responsibilities section.
#            if mapped_label == "roles_responsibilities" and "role" not in lower:
#                continue
#            return mapped_label
#    return None
#
#
#def _find_embedded_heading_split(line: str):
#    stripped = line.strip()
#    words = stripped.split()
#    if len(words) < 2:
#        return None, None, None
#    max_n = min(4, len(words) - 1)
#    # A single bare tail word (n=1) is normally fine -- it's what
#    # correctly catches real cases like "...Summary" (the motivating
#    # case, and still needed: "Personal & Key Skills Summary" relies on
#    # exactly this to split off a genuine "summary" section). The
#    # problem is specific to "interests": "activities"/"interest(s)" are
#    # such common, generic nouns that they show up constantly as the
#    # last word of an ordinary business phrase that has nothing to do
#    # with a personal-interests section. Confirmed on a real resume
#    # (Ritu Verma): "Employee Engagement and Operational Activities" (an
#    # ordinary Experience subheading) was having "Activities" sliced off
#    # and misread as a new "interests" heading. So: allow n=1 in
#    # general, just never resolve to "interests" at n=1 -- that label
#    # still gets caught fine at n=2+ ("Technical Interests" etc.) or via
#    # its own dedicated section-heading line elsewhere.
#    for n in range(max_n, 0, -1):
#        tail_words = words[-n:]
#        if not all(w[0].isupper() for w in tail_words if w[:1].isalpha()):
#            continue
#        candidate_clean = _clean_heading_candidate(" ".join(tail_words))
#        for label, pattern in SECTION_PATTERNS.items():
#            if label == "interests" and n == 1:
#                continue
#            if pattern.match(candidate_clean):
#                prefix = " ".join(words[:-n]).strip()
#                if prefix:
#                    return prefix, " ".join(tail_words), label
#    return None, None, None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#                elif _looks_like_bare_company_line(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#                    inline_content = stripped
#                elif _looks_like_company_entry(stripped):
#                    # "Company Name (date range)" all on one line. This
#                    # detector already existed, but was previously wired
#                    # up ONLY for the narrow "coming back out of a
#                    # projects run" case (current_label == "projects").
#                    # The same shape signals a new job entry no matter
#                    # what section happened to be open before it.
#                    # Confirmed on a real resume (Ritu Verma): "Torrent
#                    # Power Ltd.  (16th Sept 2012 to 15th Nov 2016.)"
#                    # appeared right after a Languages section and was
#                    # being absorbed as languages content instead of
#                    # starting a new experience section.
#                    label = "experience"
#                    confidence = 0.85
#                    inline_content = stripped
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_job_block_header(stripped, i, lines):
#                    label = "experience"
#                    confidence = 0.9
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        if label is None and stripped != "":
#            embed_prefix, embed_heading_text, embed_label = _find_embedded_heading_split(stripped)
#            if embed_label is not None and embed_label != current_label:
#                current_lines.append(embed_prefix)
#                label = embed_label
#                confidence = 0.85
#
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if current_label == "projects" and experience_seen and stripped != "":
#            if _looks_like_company_entry(stripped):
#                label = "experience"
#                confidence = 0.9
#                inline_content = stripped
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            # Only swallow this as an in-job sub-bullet (e.g. "Key
#            # achievements:" under one role) when it's a loosely-inferred
#            # match. A clean strict-pattern hit (confidence >= 0.9, e.g.
#            # a standalone "AWARDS AND ACHIEVEMENTS:" heading) is a real
#            # top-level section and must still be allowed to split off,
#            # even though it ends with ":" and even inside an experience
#            # run.
#            if confidence < 0.9 and (label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-")):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            elif label == "unknown":
#                # "unknown" means the scoring heuristic thought this
#                # line LOOKS heading-shaped, but nothing could actually
#                # resolve it to a real section label -- i.e. we are NOT
#                # confident it's really a heading. A confidently
#                # detected real heading is safe to drop (it's redundant
#                # with the section label), but for an unconfirmed guess
#                # the safer default is to keep the line as content
#                # rather than silently discard it. Confirmed on a real
#                # resume (Praveen Kumar Pedapapa): "MoveInSync
#                # Technology Solutions" (a company name, not a heading)
#                # scored 0.65 purely because it contains "Technology"
#                # and is short/comma-free, and was being dropped
#                # entirely instead of kept as the company name it is.
#                current_lines.append(stripped)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#
#








#worked changing just for padepapa resume to work
#import re
#import difflib
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#_EXPERIENCE_DATE_RANGE_TAIL = (
#    r"(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*"
#    r"(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?"
#)
#_EXPERIENCE_PATTERN = re.compile(
#    rf"^(?:((work|industry|relevant|professional|previous|past|prior)\s*(and\s*)?){{0,2}}experience"
#    rf"|employment(\s+(history|details|records?|background))?"
#    rf"|professional\s*background"
#    rf"|career\s*history|work\s*history|internships?"
#    rf"|corporate\s+success|career\s+journey|professional\s+journey"
#    rf"|career\s+chronology|employment\s+timeline)"
#    rf"{_EXPERIENCE_DATE_RANGE_TAIL}$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _EXPERIENCE_PATTERN,
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills",
#    # NOTE: "technolog" was deliberately removed from this loose,
#    # substring-based fallback table. It's redundant for genuine cases --
#    # detect_section_label() already matches whole-line "Technology
#    # Stack" / "Technology Summary" headings via _SKILLS_PATTERN's own
#    # strict alternatives, and that strict path always runs first. Left
#    # in here, "technolog" matched as a bare substring, so any company
#    # name containing "Technology"/"Technologies" (extremely common,
#    # e.g. "MoveInSync Technology Solutions", "DXC Technology") was
#    # being misread as a new "skills" section heading and silently
#    # discarded as heading text -- real data loss. Confirmed on a real
#    # resume (Praveen Kumar Pedapapa).
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
#_JOB_DATE_RANGE = re.compile(
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|\bto\b\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
## --- Job-block header detection (Naukri-style "CAREER PROFILE: N"
## resumes) ---------------------------------------------------------
##
## Some templates don't head each job with "Company Name (dates)" on one
## line -- they use a bare index heading like "CAREER PROFILE: 6" and then
## spell out "Role:", "Organization:", "Description:" as separate labeled
## fields on the following lines. That heading doesn't match any
## SECTION_PATTERNS alternative, and score_heading_line's "colon followed
## by content" penalty (meant to suppress "Duration: 6 months"-style body
## lines) fires on the trailing digit and drags the score below the
## classification threshold anyway -- so the heading was invisible and
## the whole job block got silently absorbed into whatever section was
## currently open.
##
## Fixed with a content-based (not heading-text-based) rule, so it isn't
## tied to the literal words "career profile" and still works if a
## template calls it "Assignment 3" or similar: look ahead a few lines
## for "Role:"/"Designation:" AND "Organization:"/"Company:" fields -- if
## both appear, the current line is the start of a job entry regardless
## of what it says.
#
#_SUBFIELD_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z \t]{0,40}?)\s*:")
#
#
#def _experience_subfield_label(stripped: str):
#    m = _SUBFIELD_LABEL_RE.match(stripped)
#    if not m:
#        return None
#    return m.group(1).strip().lower()
#
#
#def _looks_like_job_block_header(stripped: str, idx: int, lines: list) -> bool:
#    if not stripped or len(stripped) > 60:
#        return False
#    found_role = False
#    found_org = False
#    checked = 0
#    j = idx + 1
#    while j < len(lines) and checked < 4:
#        s = lines[j].strip()
#        if s:
#            checked += 1
#            label = _experience_subfield_label(s)
#            if label in ("role", "designation"):
#                found_role = True
#            elif label in ("organization", "organisation", "company", "employer"):
#                found_org = True
#            if found_role and found_org:
#                return True
#        j += 1
#    return False
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#_NON_COMPANY_LINE_PREFIXES = re.compile(
#    r"^(duration|project|role|project\s*#|project\s+title|project\s+description)\b",
#    re.IGNORECASE
#)
#
#
#def _looks_like_company_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 150:
#        return False
#    if _NON_COMPANY_LINE_PREFIXES.match(stripped):
#        return False
#    match = _JOB_DATE_RANGE.search(stripped)
#    if not match:
#        return False
#    prefix = stripped[:match.start()].strip(" \t:-")
#    return len(prefix.split()) >= 2
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    if len(stripped.split()) > 4:
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            # roles_responsibilities' own strict SECTION_PATTERNS regex
#            # requires the word "role(s)" to be present alongside
#            # "responsibilit-" -- this loose substring fallback must
#            # honor that same requirement, or a bare in-job field label
#            # like "Responsibility:" / "Additional responsibility:"
#            # (which contains no "role" at all) gets misread as the
#            # start of a brand new top-level section. Confirmed on a
#            # real resume (Mohan Kumar K): each of 6 job blocks under
#            # "CAREER PROFILE: N" has its own "Responsibility:" field,
#            # and every one of them was incorrectly splitting off its
#            # own roles_responsibilities section.
#            if mapped_label == "roles_responsibilities" and "role" not in lower:
#                continue
#            return mapped_label
#    return None
#
#
#def _find_embedded_heading_split(line: str):
#    stripped = line.strip()
#    words = stripped.split()
#    if len(words) < 2:
#        return None, None, None
#    max_n = min(4, len(words) - 1)
#    for n in range(max_n, 0, -1):
#        tail_words = words[-n:]
#        if not all(w[0].isupper() for w in tail_words if w[:1].isalpha()):
#            continue
#        candidate_clean = _clean_heading_candidate(" ".join(tail_words))
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(candidate_clean):
#                prefix = " ".join(words[:-n]).strip()
#                if prefix:
#                    return prefix, " ".join(tail_words), label
#    return None, None, None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_job_block_header(stripped, i, lines):
#                    label = "experience"
#                    confidence = 0.9
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        if label is None and stripped != "":
#            embed_prefix, embed_heading_text, embed_label = _find_embedded_heading_split(stripped)
#            if embed_label is not None and embed_label != current_label:
#                current_lines.append(embed_prefix)
#                label = embed_label
#                confidence = 0.85
#
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if current_label == "projects" and experience_seen and stripped != "":
#            if _looks_like_company_entry(stripped):
#                label = "experience"
#                confidence = 0.9
#                inline_content = stripped
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            # Only swallow this as an in-job sub-bullet (e.g. "Key
#            # achievements:" under one role) when it's a loosely-inferred
#            # match. A clean strict-pattern hit (confidence >= 0.9, e.g.
#            # a standalone "AWARDS AND ACHIEVEMENTS:" heading) is a real
#            # top-level section and must still be allowed to split off,
#            # even though it ends with ":" and even inside an experience
#            # run.
#            if confidence < 0.9 and (label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-")):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            elif label == "unknown":
#                # "unknown" means the scoring heuristic thought this
#                # line LOOKS heading-shaped, but nothing could actually
#                # resolve it to a real section label -- i.e. we are NOT
#                # confident it's really a heading. A confidently
#                # detected real heading is safe to drop (it's redundant
#                # with the section label), but for an unconfirmed guess
#                # the safer default is to keep the line as content
#                # rather than silently discard it. Confirmed on a real
#                # resume (Praveen Kumar Pedapapa): "MoveInSync
#                # Technology Solutions" (a company name, not a heading)
#                # scored 0.65 purely because it contains "Technology"
#                # and is short/comma-free, and was being dropped
#                # entirely instead of kept as the company name it is.
#                current_lines.append(stripped)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#
#






##worked commenting just to fix padepapa resume
#import re
#import difflib
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#_EXPERIENCE_DATE_RANGE_TAIL = (
#    r"(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*"
#    r"(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?"
#)
#_EXPERIENCE_PATTERN = re.compile(
#    rf"^(?:((work|industry|relevant|professional|previous|past|prior)\s*(and\s*)?){{0,2}}experience"
#    rf"|employment(\s+(history|details|records?|background))?"
#    rf"|professional\s*background"
#    rf"|career\s*history|work\s*history|internships?"
#    rf"|corporate\s+success|career\s+journey|professional\s+journey"
#    rf"|career\s+chronology|employment\s+timeline)"
#    rf"{_EXPERIENCE_DATE_RANGE_TAIL}$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _EXPERIENCE_PATTERN,
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
#_JOB_DATE_RANGE = re.compile(
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|\bto\b\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
## --- Job-block header detection (Naukri-style "CAREER PROFILE: N"
## resumes) ---------------------------------------------------------
##
## Some templates don't head each job with "Company Name (dates)" on one
## line -- they use a bare index heading like "CAREER PROFILE: 6" and then
## spell out "Role:", "Organization:", "Description:" as separate labeled
## fields on the following lines. That heading doesn't match any
## SECTION_PATTERNS alternative, and score_heading_line's "colon followed
## by content" penalty (meant to suppress "Duration: 6 months"-style body
## lines) fires on the trailing digit and drags the score below the
## classification threshold anyway -- so the heading was invisible and
## the whole job block got silently absorbed into whatever section was
## currently open.
##
## Fixed with a content-based (not heading-text-based) rule, so it isn't
## tied to the literal words "career profile" and still works if a
## template calls it "Assignment 3" or similar: look ahead a few lines
## for "Role:"/"Designation:" AND "Organization:"/"Company:" fields -- if
## both appear, the current line is the start of a job entry regardless
## of what it says.
#
#_SUBFIELD_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z \t]{0,40}?)\s*:")
#
#
#def _experience_subfield_label(stripped: str):
#    m = _SUBFIELD_LABEL_RE.match(stripped)
#    if not m:
#        return None
#    return m.group(1).strip().lower()
#
#
#def _looks_like_job_block_header(stripped: str, idx: int, lines: list) -> bool:
#    if not stripped or len(stripped) > 60:
#        return False
#    found_role = False
#    found_org = False
#    checked = 0
#    j = idx + 1
#    while j < len(lines) and checked < 4:
#        s = lines[j].strip()
#        if s:
#            checked += 1
#            label = _experience_subfield_label(s)
#            if label in ("role", "designation"):
#                found_role = True
#            elif label in ("organization", "organisation", "company", "employer"):
#                found_org = True
#            if found_role and found_org:
#                return True
#        j += 1
#    return False
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#_NON_COMPANY_LINE_PREFIXES = re.compile(
#    r"^(duration|project|role|project\s*#|project\s+title|project\s+description)\b",
#    re.IGNORECASE
#)
#
#
#def _looks_like_company_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 150:
#        return False
#    if _NON_COMPANY_LINE_PREFIXES.match(stripped):
#        return False
#    match = _JOB_DATE_RANGE.search(stripped)
#    if not match:
#        return False
#    prefix = stripped[:match.start()].strip(" \t:-")
#    return len(prefix.split()) >= 2
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    if len(stripped.split()) > 4:
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            # roles_responsibilities' own strict SECTION_PATTERNS regex
#            # requires the word "role(s)" to be present alongside
#            # "responsibilit-" -- this loose substring fallback must
#            # honor that same requirement, or a bare in-job field label
#            # like "Responsibility:" / "Additional responsibility:"
#            # (which contains no "role" at all) gets misread as the
#            # start of a brand new top-level section. Confirmed on a
#            # real resume (Mohan Kumar K): each of 6 job blocks under
#            # "CAREER PROFILE: N" has its own "Responsibility:" field,
#            # and every one of them was incorrectly splitting off its
#            # own roles_responsibilities section.
#            if mapped_label == "roles_responsibilities" and "role" not in lower:
#                continue
#            return mapped_label
#    return None
#
#
#def _find_embedded_heading_split(line: str):
#    stripped = line.strip()
#    words = stripped.split()
#    if len(words) < 2:
#        return None, None, None
#    max_n = min(4, len(words) - 1)
#    for n in range(max_n, 0, -1):
#        tail_words = words[-n:]
#        if not all(w[0].isupper() for w in tail_words if w[:1].isalpha()):
#            continue
#        candidate_clean = _clean_heading_candidate(" ".join(tail_words))
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(candidate_clean):
#                prefix = " ".join(words[:-n]).strip()
#                if prefix:
#                    return prefix, " ".join(tail_words), label
#    return None, None, None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_job_block_header(stripped, i, lines):
#                    label = "experience"
#                    confidence = 0.9
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        if label is None and stripped != "":
#            embed_prefix, embed_heading_text, embed_label = _find_embedded_heading_split(stripped)
#            if embed_label is not None and embed_label != current_label:
#                current_lines.append(embed_prefix)
#                label = embed_label
#                confidence = 0.85
#
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if current_label == "projects" and experience_seen and stripped != "":
#            if _looks_like_company_entry(stripped):
#                label = "experience"
#                confidence = 0.9
#                inline_content = stripped
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            # Only swallow this as an in-job sub-bullet (e.g. "Key
#            # achievements:" under one role) when it's a loosely-inferred
#            # match. A clean strict-pattern hit (confidence >= 0.9, e.g.
#            # a standalone "AWARDS AND ACHIEVEMENTS:" heading) is a real
#            # top-level section and must still be allowed to split off,
#            # even though it ends with ":" and even inside an experience
#            # run.
#            if confidence < 0.9 and (label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-")):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#










#worked just commented for mohankumari- career profile bug fix
#import re
#import difflib
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
## A resume heading can carry a trailing date-range qualifier right on the
## same line, e.g. "PAST EXPERIENCE FROM 2005 TO 2015", "EXPERIENCE
## (2015-2020)", "WORK HISTORY 2010-2015" -- the base experience pattern
## (built like every other section from _build_section_pattern) only
## allowed an optional "& word"/"and word" suffix, so any trailing date
## range broke the match completely, and the heading fell through as
## ordinary body text. Confirmed on a real resume (Mrityunjay Prasad Roy):
## "PAST EXPERIENCE FROM 2005 TO 2015" -- a genuine section heading
## introducing a list of past employers -- stayed absorbed inside "header"
## for exactly this reason. Two fixes bundled here: (1) "past" and "prior"
## added to the allowed prefix words (previously only work/industry/
## relevant/professional/previous), (2) an optional trailing date-range
## tail appended to the whole pattern.
#_EXPERIENCE_DATE_RANGE_TAIL = (
#    r"(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*"
#    r"(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?"
#)
#_EXPERIENCE_PATTERN = re.compile(
#    rf"^(?:((work|industry|relevant|professional|previous|past|prior)\s*(and\s*)?){{0,2}}experience"
#    rf"|employment(\s+(history|details|records?|background))?"
#    rf"|professional\s*background"
#    rf"|career\s*history|work\s*history|internships?"
#    rf"|corporate\s+success|career\s+journey|professional\s+journey"
#    rf"|career\s+chronology|employment\s+timeline)"
#    rf"{_EXPERIENCE_DATE_RANGE_TAIL}$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _EXPERIENCE_PATTERN,
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
#_JOB_DATE_RANGE = re.compile(
#    # NOTE: the separator group below originally only accepted a literal
#    # dash (-/–/—) between the two dates, or a lookahead for "Present/
#    # Current/Now/Till". That silently failed to match the extremely
#    # common "Month YYYY to Month YYYY" format -- confirmed on a real
#    # resume (Milan Mohite) where every single job entry is written as
#    # "(September 2022 to Present)" -- so _looks_like_job_entry and
#    # everything downstream of it (including _looks_like_company_entry
#    # used by the projects->experience switch-back below) never
#    # recognized these as date ranges at all. Added `\bto\b` as an
#    # accepted separator alongside the dash forms; this only WIDENS what
#    # matches, so it can't break any range that already matched before.
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|\bto\b\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
## --- "switch back" out of a projects run into experience --------------
##
## _PROJECTS_PATTERN matches things like "Project#1" / "Project Title...",
## so the moment one of those appears right after a job line inside an
## experience section, current_label flips experience -> projects and
## NOTHING ever flips it back -- every following company block (e.g.
## "Capgemini...", "Price Waterhouse Coopers...") just gets swept in as
## plain body text of that one never-ending "projects" run.
##
## The fix has two parts:
##   1. _looks_like_company_entry() below recognizes a "Company Name
##      (date range)" line -- the shape of a new employer block -- while
##      explicitly excluding lines that start with project-ish label
##      words (Duration/Project/Role/...), so it can't mistake a project
##      metadata line for a company line.
##   2. Where it's used in split_into_sections(), the switch back to
##      "experience" keeps the company+date line itself as inline_content
##      instead of discarding it as a bare heading -- a first attempt at
##      this fix flipped the label correctly but still discarded the
##      line's text (correct for a real heading like "WORK EXPERIENCE:",
##      which has no content of its own -- wrong here, since the
##      company+date line IS the content), which produced a zero-length
##      section that then got silently dropped from the output.
#_NON_COMPANY_LINE_PREFIXES = re.compile(
#    r"^(duration|project|role|project\s*#|project\s+title|project\s+description)\b",
#    re.IGNORECASE
#)
#
#
#def _looks_like_company_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 150:
#        return False
#    if _NON_COMPANY_LINE_PREFIXES.match(stripped):
#        return False
#    match = _JOB_DATE_RANGE.search(stripped)
#    if not match:
#        return False
#    prefix = stripped[:match.start()].strip(" \t:-")
#    return len(prefix.split()) >= 2   # a company name is real text, not a label word
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    """
#    Guard: only promote a SHORT candidate (<=4 words) via this fuzzy,
#    substring-based keyword match. A genuine section heading reached
#    through this fallback is almost always compact ("Certifications",
#    "Last 5 Career Timeline" -- both <=4 words). A longer, descriptive
#    subheading that merely CONTAINS a keyword as one word among several
#    is a different thing -- it's still describing the same ongoing topic
#    as its neighboring, correctly-absorbed subheadings, not introducing
#    a real new section. Confirmed on a real resume (Mrityunjay Prasad
#    Roy): "TRADE MARK & OTHER LICENSES" is one of several all-caps
#    subheadings inside a long "Areas of Expertise" list -- it happened to
#    contain "LICENSES" and got incorrectly promoted into its own
#    "certifications" section without this guard.
#    """
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    if len(stripped.split()) > 4:
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            return mapped_label
#    return None
#
#
#def _find_embedded_heading_split(line: str):
#    """
#    Catches a section heading glued onto the END of a content line with
#    ZERO line break in the source document -- confirmed via raw XML on a
#    real resume (Milan Mohite .docx): "...with 56%" and "Work
#    Experience:" are literally the same <w:p> paragraph, distinguished
#    in Word only by the "Work Experience:" run being bold+underlined --
#    a formatting signal that's gone once we're working with plain text.
#    Every other heuristic in this file is line-based and has nothing to
#    split on here, since there's no line boundary at all to find, not
#    even a lost one.
#
#    This is a LAST-RESORT fallback: it's only ever tried after every
#    other detector above has already returned None for the line. It
#    tries the last 1-4 words of the line as a heading candidate, longest
#    match first (so "Work Experience:" is preferred over just
#    "Experience:"), and only accepts a candidate if every word in it is
#    capitalized -- genuine embedded headings in these templates are
#    always Title-Case/ALL-CAPS, and this guard is what stops it from
#    false-triggering on ordinary lowercase text that happens to contain
#    a section keyword (e.g. "...fluent in English language" must NOT
#    become a "languages" heading split, since "language" is lowercase
#    there). It only returns a result when there's real content left
#    over as a prefix, so a line that's just the heading by itself is
#    left alone (detect_section_label already handles that case).
#    """
#    stripped = line.strip()
#    words = stripped.split()
#    if len(words) < 2:
#        return None, None, None
#    max_n = min(4, len(words) - 1)
#    for n in range(max_n, 0, -1):
#        tail_words = words[-n:]
#        if not all(w[0].isupper() for w in tail_words if w[:1].isalpha()):
#            continue
#        candidate_clean = _clean_heading_candidate(" ".join(tail_words))
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(candidate_clean):
#                prefix = " ".join(words[:-n]).strip()
#                if prefix:
#                    return prefix, " ".join(tail_words), label
#    return None, None, None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        # --- NEW: heading glued to the tail of a content line, no line
#        # break at all in the source (see _find_embedded_heading_split
#        # docstring). Only tried as an absolute last resort, after every
#        # detector above has already failed to classify this line.
#        if label is None and stripped != "":
#            embed_prefix, embed_heading_text, embed_label = _find_embedded_heading_split(stripped)
#            if embed_label is not None and embed_label != current_label:
#                current_lines.append(embed_prefix)
#                label = embed_label
#                confidence = 0.85
#        # --- END NEW ---
#
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if current_label == "projects" and experience_seen and stripped != "":
#            if _looks_like_company_entry(stripped):
#                label = "experience"
#                confidence = 0.9
#                inline_content = stripped   # keep the company/date line itself as content
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#









#not worked
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#_EXPERIENCE_DATE_RANGE_TAIL = (
#    r"(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*"
#    r"(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?"
#)
#_EXPERIENCE_PATTERN = re.compile(
#    rf"^(?:((work|industry|relevant|professional|previous|past|prior)\s*(and\s*)?){{0,2}}experience"
#    rf"|employment(\s+(history|details|records?|background))?"
#    rf"|professional\s*background"
#    rf"|career\s*history|work\s*history|internships?"
#    rf"|corporate\s+success|career\s+journey|professional\s+journey"
#    rf"|career\s+chronology|employment\s+timeline)"
#    rf"{_EXPERIENCE_DATE_RANGE_TAIL}$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _EXPERIENCE_PATTERN,
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
#_JOB_DATE_RANGE = re.compile(
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    """
#    True if `line` contains a known section keyword, either as an exact
#    substring OR as a close typo (e.g. "Employement" -> "employment").
#
#    FIX (typo tolerance): resumes routinely misspell their own section
#    headings -- confirmed on a real resume (Praveen Kumar Pedapapa):
#    "EMPLOYEMENT DETAILS" (extra E) scored just under the heading
#    threshold and was never recognized, because the plain substring
#    check requires "employment" spelled correctly. We fall back to a
#    per-word fuzzy match (cutoff 0.82, 5+ letter words only, to avoid
#    matching short unrelated words) against the same keyword list.
#
#    Callers must only invoke this on lines already gated as heading-
#    shaped (see score_heading_line) -- see _contains_section_keyword's
#    sibling gate there for why that matters.
#    """
#    lower = line.lower()
#    if any(keyword in lower for keyword in _SECTION_KEYWORDS):
#        return True
#    words = re.findall(r"[a-z]+", lower)
#    for word in words:
#        if len(word) < 5:
#            continue
#        if difflib.get_close_matches(word, _SECTION_KEYWORDS, n=1, cutoff=0.82):
#            return True
#    return False
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#
#    # FIX (false-positive guard): only award the keyword bonus for
#    # lines that already LOOK like a heading (ALL CAPS, or ending in a
#    # colon) -- real section headings almost always are. Without this
#    # gate, an ordinary sentence or company name that merely CONTAINS a
#    # keyword substring gets promoted to a fake heading. Confirmed on a
#    # real resume (Praveen Kumar Pedapapa): the plain body line
#    # "MoveInSync  Technology Solutions" (an employer name, Title Case,
#    # no colon) contains "Technology", which matches the "technolog"
#    # keyword stem (meant for headings like "Technology Stack") and was
#    # scoring high enough to get mislabeled as a "skills" section
#    # heading. "Technology" is an extremely common word inside ordinary
#    # company names, so this isn't a one-off -- it would misfire on any
#    # resume whose employer name contains it.
#    heading_shaped = is_all_caps or stripped.endswith(":")
#    if heading_shaped and _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    if len(stripped.split()) > 4:
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            return mapped_label
#
#    # Typo tolerance -- same rationale as _contains_section_keyword.
#    # Only reached for candidates that already crossed the heading-score
#    # threshold (heading-shaped: ALL CAPS or colon-terminated), so this
#    # doesn't loosen matching on ordinary short phrases in general.
#    words = re.findall(r"[a-z]+", lower)
#    for word in words:
#        if len(word) < 5:
#            continue
#        matches = difflib.get_close_matches(word, list(_KEYWORD_TO_LABEL.keys()), n=1, cutoff=0.82)
#        if matches:
#            return _KEYWORD_TO_LABEL[matches[0]]
#    return None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#






##commenting just to fix parveen padepapa resume - and write 2 more above this- 2nd one(just above not worked) we are using
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#_EXPERIENCE_DATE_RANGE_TAIL = (
#    r"(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*"
#    r"(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?"
#)
#
## FIX: Removed the trailing $ to match "EMPLOYEMENT DETAILS" with any trailing spaces
#_EXPERIENCE_PATTERN = re.compile(
#    r"^(?:"
#    r"(?:(?:work|industry|relevant|professional|previous|past|prior)\s+(?:and\s+)?)?experience"
#    r"|employ(?:e)?ment(?:\s+(?:history|details|records?|background))?"
#    r"|professional\s+background"
#    r"|career\s+history|work\s+history|internships?"
#    r"|corporate\s+success|career\s+journey|professional\s+journey"
#    r"|career\s+chronology|employment\s+timeline"
#    r")(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?",
#    re.IGNORECASE
#)


#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _EXPERIENCE_PATTERN,
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#    "technical interests": "skills",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
#_JOB_DATE_RANGE = re.compile(
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_title(line: str) -> bool:
#    """Detects lines like 'Senior Software Engineer _ Sept 2023 to Present.'"""
#    stripped = line.strip()
#    if not stripped:
#        return False
#    # Pattern: Job title + underscore/separator + date range
#    if re.search(r'[_\-\u2013\u2014]\s*(?:Sept|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Oct|Nov|Dec)', stripped, re.IGNORECASE):
#        return True
#    # Pattern: Job title + date range with "to" or "-"
#    if re.search(r'(?:to|\-|\u2013)\s*(?:Present|Current|Now|\d{4})', stripped, re.IGNORECASE):
#        # Make sure it's not a project description
#        if not re.search(r'Project\s+Name|Client\s+Name|Tenure|Description', stripped, re.IGNORECASE):
#            return True
#    return False
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    
#    if _looks_like_job_title(stripped):
#        return 0.0
#    
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    if len(stripped.split()) > 4:
#        return None
#
#    lower = stripped.lower()
#    
#    if "technical interests" in lower:
#        return "skills"
#    
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            return mapped_label
#    return None


##def split_into_sections(text: str) -> list:
##    lines = text.split("\n")
##    sections = []
##    current_label = "header"
##    current_start = 0
##    current_lines = []
##    current_confidence = 0.9
##    experience_seen = False
##    deferred_contact_lines = []
##    skip_next = False
##
##    for i, line in enumerate(lines):
##        if skip_next:
##            skip_next = False
##            continue
##
##        stripped = line.strip()
##        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
##        inline_content = None
##
##        if label is not None and stripped != "" and i + 1 < len(lines):
##            nxt = lines[i + 1].strip()
##            if (nxt and not _LEADING_BULLET_RE.match(nxt)
##                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
##                combined = _clean_heading_candidate(f"{stripped} {nxt}")
##                for combined_label, pattern in SECTION_PATTERNS.items():
##                    if pattern.match(combined):
##                        label = combined_label
##                        confidence = max(confidence, 0.9)
##                        skip_next = True
##                        break
##
##        if label is None and stripped != "":
##            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
##            if two_line_label is not None:
##                label = two_line_label
##                confidence = two_line_confidence
##                skip_next = True
##
##        if label is None and stripped != "":
##            if current_label not in _LIST_SECTION_LABELS:
##                heading_candidate, remaining = split_inline_heading(stripped)
##                if heading_candidate is not None:
##                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
##                    if label is not None:
##                        inline_content = remaining
##
##        if label is None and stripped != "":
##            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
##            if letter_spaced_label is not None:
##                label = letter_spaced_label
##                confidence = 0.85
##
##        if label is None and stripped != "":
##            if not _is_wrapped_word(stripped, lines[:i]):
##                fuzzy_label = _fuzzy_heading_label(stripped)
##                if fuzzy_label is not None:
##                    label = fuzzy_label
##                    confidence = 0.8
##
##        if label is None and stripped != "":
##            if not _is_wrapped_word(stripped, lines[:i]):
##                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
##                    label = "experience"
##                    confidence = 0.85
##
##        if label is None and stripped != "":
##            if not _is_wrapped_word(stripped, lines[:i]):
##                heading_score = score_heading_line(stripped, i, lines)
##                if heading_score >= 0.65:
##                    resolved = _resolve_unknown_heading(stripped)
##                    if resolved is not None:
##                        label = resolved
##                        confidence = max(heading_score, 0.8)
##                    else:
##                        label = "unknown"
##                        confidence = heading_score
##
##        if label == "languages" and stripped != "":
##            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
##            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
##                label = None
##                confidence = 0.0
##                inline_content = None
##
##        if label == "summary" and not experience_seen and stripped != "":
##            upcoming = _next_nonblank_line(lines, i + 1)
##            if _looks_like_job_entry(upcoming):
##                label = "experience"
##                confidence = 0.85
##
##        if label is not None and label == current_label:
##            current_lines.append(line)
##            continue
##
##        if current_label in _LIST_SECTION_LABELS and label == "unknown":
##            promoted_label = _resolve_unknown_heading(stripped)
##            if promoted_label is None:
##                current_lines.append(line)
##                continue
##            label = promoted_label
##            confidence = 0.8
##
##        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
##            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
##                current_lines.append(line)
##                continue
##
##        if label is not None and stripped != "":
##            if label == "experience":
##                experience_seen = True
##            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
##            if section_text:
##                sections.append(Section(
##                    label=current_label,
##                    raw_text=section_text,
##                    start_line=current_start,
##                    confidence=current_confidence
##                ))
##            current_label = label
##            current_start = i
##            current_lines = []
##            if inline_content:
##                current_lines.append(inline_content)
##            current_confidence = confidence
##        else:
##            if current_label != "header" and _is_bare_contact_line(stripped):
##                deferred_contact_lines.append(stripped)
##            else:
##                if current_label == "education" and stripped.upper() in ["DETAILS", "DETAIL"]:
##                    continue
##                current_lines.append(line)
##
##    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
##    if section_text:
##        sections.append(Section(
##            label=current_label,
##            raw_text=section_text,
##            start_line=current_start,
##            confidence=current_confidence
##        ))
##
##    if deferred_contact_lines:
##        for s in sections:
##            if s.label == "header":
##                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
##                break
##        else:
##            sections.insert(0, Section(
##                label="header",
##                raw_text="\n".join(deferred_contact_lines),
##                start_line=0,
##                confidence=0.9,
##            ))
##
##    return sections

#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        # --- FIX: When in experience section, be conservative about starting new sections ---
#        if current_label == "experience":
#            # If we're in experience section, don't start a new section unless:
#            # 1. It's clearly a section heading (all caps, short, blank line above)
#            # 2. OR confidence is very high (>= 0.9)
#            # 3. OR it matches a section pattern strongly
#            if label is not None and label != "experience":
#                # Check if it's a clear section heading
#                is_clear_heading = (
#                    stripped == stripped.upper() and 
#                    len(stripped.split()) <= 4 and 
#                    (i > 0 and lines[i-1].strip() == "")
#                )
#                if not is_clear_heading and confidence < 0.9:
#                    # It's probably a company name or job detail - keep in experience
#                    current_lines.append(line)
#                    continue
#        # --- END FIX ---
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            
#            # Add the heading line(s) to the new section
#            if skip_next and i + 1 < len(lines):
#                # If we skipped a line (two-line heading), include both
#                current_lines.append(stripped)
#                current_lines.append(lines[i + 1].strip())
#            else:
#                # Normal case: add the heading itself
#                current_lines.append(stripped)
#            
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                if current_label == "education" and stripped.upper() in ["DETAILS", "DETAIL"]:
#                    continue
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#
##not worked - fixing parveen resume to work well.
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
## A resume heading can carry a trailing date-range qualifier right on the
## same line, e.g. "PAST EXPERIENCE FROM 2005 TO 2015", "EXPERIENCE
## (2015-2020)", "WORK HISTORY 2010-2015" -- the base experience pattern
## (built from _build_section_pattern, like every other section) only
## allows an optional "& word"/"and word" suffix, so any trailing date
## range broke the match completely, and the heading fell through as
## ordinary body text. Confirmed on a real resume (Mrityunjay Prasad Roy):
## "PAST EXPERIENCE FROM 2005 TO 2015" -- a genuine section heading
## introducing a list of past employers -- stayed absorbed inside "header"
## for exactly this reason. Two separate fixes bundled here: (1) "past"
## and "prior" added to the allowed prefix words (previously only
## work/industry/relevant/professional/previous), (2) an optional trailing
## date-range tail appended to the whole pattern. Once the heading line
## itself matches, split_into_sections()'s existing loop already sweeps
## every following line into the new "experience" section automatically
## until the next real heading -- no separate logic is needed for the
## date range or company names appearing on the SAME line vs the NEXT
## line, since a bare "Past Experience" heading (no date suffix at all)
## matches this pattern just as well and the following lines get absorbed
## exactly the same way either way.
#_EXPERIENCE_DATE_RANGE_TAIL = (
#    r"(?:\s*[:\-\u2013\u2014]?\s*\(?\s*(?:from\s+)?\d{4}\s*"
#    r"(?:to|-|\u2013|\u2014)\s*\d{4}\s*\)?)?"
#)
#_EXPERIENCE_PATTERN = re.compile(
#    rf"^(?:((work|industry|relevant|professional|previous|past|prior)\s*(and\s*)?){{0,2}}experience"
#    rf"|employment(\s+(history|details|records?|background))?"
#    rf"|professional\s*background"
#    rf"|career\s*history|work\s*history|internships?"
#    rf"|corporate\s+success|career\s+journey|professional\s+journey"
#    rf"|career\s+chronology|employment\s+timeline)"
#    rf"{_EXPERIENCE_DATE_RANGE_TAIL}$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _EXPERIENCE_PATTERN,
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}


#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
#_JOB_DATE_RANGE = re.compile(
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)


#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")


#def _is_letter_spaced_heading(line: str) -> bool:
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    """
#    Guard 2 (word-count cap) added: only promote a SHORT candidate
#    (<=4 words) via this fuzzy, substring-based keyword match. A genuine
#    section heading reached through this fallback is almost always
#    compact ("Certifications", "Last 5 Career Timeline" -- both <=4
#    words). A longer, descriptive subheading that merely CONTAINS a
#    keyword as one word among several is a different thing -- it's still
#    describing the same ongoing topic as its neighboring, correctly-
#    absorbed subheadings, not introducing a real new section. Confirmed
#    on a real resume (Mrityunjay Prasad Roy): "TRADE MARK & OTHER
#    LICENSES" is one of several all-caps subheadings inside a long "Areas
#    of Expertise" list (siblings "FEMA / RBI COMPLIANCES" and "SECURITIES
#    LAW & EXCHANGE COMPLIANCES" were correctly absorbed as plain content,
#    since neither contains a recognized keyword). This one subheading
#    alone happened to contain the word "LICENSES", matching the "licen"
#    -> "certifications" mapping, and got incorrectly promoted into its
#    own "certifications" section -- even though it's the same kind of
#    topic subheading as its unpromoted siblings, not an actual list of
#    the candidate's certifications (no certificate names, issuing
#    bodies, or dates anywhere in the content that followed).
#    """
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    if len(stripped.split()) > 4:
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            return mapped_label
#    return None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#
#








##worked - final one, changing just for to work docx resumes
#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 22)
#
#CHANGELOG (rev 22):
#- _is_bare_contact_line() no longer misclassifies numeric date ranges
#  ("2022-04-03 - 2023-07-15") as bare phone-number lines. Confirmed on a
#  real resume (Rohit Shivnath Zimbar): four of five job date ranges
#  (every one using a plain ASCII hyphen) were silently deferred out of
#  the "experience" section entirely and dumped into "header", because
#  _BARE_PHONE_LINE's character class has no way to distinguish a phone
#  number from a same-shaped date range. Checking _JOB_DATE_RANGE first
#  and bailing out on a real match fixes this.
#
#CHANGELOG (rev 21):
#- _JOB_DATE_RANGE now recognizes day-first date ranges ("04 May, 2023 –
#  08 Dec 2025"), mirroring the same fix applied to experience.py's
#  DATE_RANGE_RE. Confirmed on a real resume (Sachin Chauhan).
#
#CHANGELOG (rev 20):
#- _JOB_DATE_RANGE now recognizes month-name date ranges ("Oct 2023 –
#  Nov 2025"), not just numeric formats. This directly fixes the
#  summary->experience promotion check for resumes whose real job
#  history sits under a heading that also matches "summary" (e.g. a
#  second "PROFESSIONAL SUMMARY" heading actually introducing detailed
#  job entries) when those job entries use month names rather than
#  numeric dates. Confirmed on a real resume (Sachin Chitranshi).
#
#CHANGELOG (rev 19):
#- Widened the "skills" pattern to cover a broader alias list: "technical
#  expertise", "technology stack", "technology summary", "knowledge
#  summary", "knowledge base", "technical snapshot", "skill matrix".
#- Added tech-stack vs spoken-language disambiguation for the "languages"
#  label. Confirmed on a real resume (Sachin Chauhan): a skills-matrix
#  table used "Languages" as a row label ("Languages: PHP, JavaScript"),
#  which correctly matched the real "languages" section pattern by its
#  own rules, but wrongly split the skills matrix into a separate
#  "languages" section and dragged every following row into it. Now,
#  before committing a "languages" match, we check what follows the
#  heading -- tech-stack terms (php, react, mysql...) suppress the split
#  so the line falls through and gets absorbed into whatever section
#  (usually "skills") is already open; recognizable spoken-language names
#  (English, Hindi...) still split normally. See
#  _looks_like_tech_stack_content / _looks_like_spoken_language_content.
#
#CHANGELOG (rev 18):
#- Added a new "early_career" section label for condensed one-line-per-job
#  recap tables (e.g. "LAST 5 CAREER TIMELINE" followed by lines like
#  "Oct-2023 - Nov-2025 | Manager | KPMG"), kept distinct from the main
#  "experience" section so detailed bullets and condensed recap entries
#  don't get merged into one messy section.
#- Generalized unknown-heading resolution: previously _resolve_unknown_heading()
#  only ran when deciding whether to absorb a heading-scored line into an
#  already-open list-type section. A heading transitioning FROM a
#  non-list section (e.g. "summary") straight into "unknown" never got a
#  chance to resolve at all -- confirmed on a real resume (Sachin
#  Chitranshi) where this was exactly why "LAST 5 CAREER TIMELINE" landed
#  as a bare "unknown" section instead of a real label. Resolution now
#  runs at the point "unknown" is first assigned, not just later at the
#  absorption-check.
#
#CHANGELOG (rev 17):
#- Added _is_letter_spaced_heading() / _resolve_letter_spaced_heading() to
#  handle resume templates whose headings extract with a literal space
#  between every letter (e.g. "P R O F E S S I O N A L S U M M A R Y"),
#  which defeats both SECTION_PATTERNS (word-boundary regexes) and
#  score_heading_line (word-count cutoff) simultaneously. Confirmed on a
#  real resume (Ritika Jain) where this collapsed the entire document
#  into one "header" section. See _resolve_letter_spaced_heading's
#  docstring for the matching approach.
#
#CHANGELOG (rev 16):
#- certifications/achievements patterns widened to tolerate a trailing
#  parenthetical qualifier (e.g. "Certifications (Technical + Management)")
#  and, for achievements, a leading qualifier word (e.g. "KEY ACHIEVEMENTS"),
#  matching the prefix support "skills" already had.
#- Added _resolve_unknown_heading() + updated the _LIST_SECTION_LABELS
#  absorption check in split_into_sections(). Previously ANY line scoring
#  "unknown" inside a list-type section (skills/education/experience/etc.)
#  was silently absorbed into that section, with no way to tell a harmless
#  sub-label ("Programming Languages:" inside a skills list) apart from a
#  genuine new section heading whose exact phrasing wasn't in
#  SECTION_PATTERNS (e.g. "Certifications (Technical + Management)" landing
#  inside "experience", "KEY ACHIEVEMENTS" landing inside "strengths" --
#  both confirmed on real resumes, Sachin Singh and Saheb Jaggi). Now, before
#  absorbing, we try to resolve the line to a real label via the same
#  keyword set score_heading_line already used to flag it as heading-shaped
#  in the first place. Only promotes when the line looks like a bare heading
#  (nothing meaningful after a trailing colon) so a genuine in-list bullet
#  like "Certifications: AWS, Azure" is NOT mistaken for a new section.
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
## Core keywords widened to also cover "expertise" (e.g. "technical
## expertise"), and the optional suffix widened to also cover "matrix"
## (e.g. "skill matrix") alongside the existing "sets?" ("skill set").
## Standalone phrases that don't have "skill"/"competenc*"/"expertise" as
## their own head noun at all -- "technology stack", "technology summary",
## "knowledge summary", "knowledge base", "technical snapshot" -- are
## added as separate top-level alternatives instead, since they don't fit
## the prefix+core structure.
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"((work|industry|relevant|professional|previous)\s*(and\s*)?){0,2}experience",
#        r"employment(\s+(history|details|records?|background))?", r"professional\s*background",
#        r"career\s*history", r"work\s*history", r"internships?",
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    # Widened to tolerate a trailing parenthetical qualifier, e.g.
#    # "Certifications (Technical + Management)" (confirmed on Sachin
#    # Singh's resume) which the old pattern rejected outright since it
#    # only allowed an "& word" / "and word" suffix, not a parenthetical.
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    # Widened to tolerate a leading qualifier word, e.g. "KEY ACHIEVEMENTS"
#    # (confirmed on Saheb Jaggi's resume) -- "skills" already supported an
#    # analogous prefix group; achievements never got the same treatment.
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    # Condensed "here's my last N jobs in one line each" recap, distinct
#    # from the full "experience" section's detailed per-job bullets.
#    # Confirmed on a real resume (Sachin Chitranshi): "LAST 5 CAREER
#    # TIMELINE" headed a block of one-line entries like "Oct-2023 -
#    # Nov-2025 | Manager | KPMG" that restate the same jobs covered in
#    # full detail later in the document. Kept as its own label (not
#    # folded into "experience") so the detailed section isn't polluted
#    # with duplicate condensed entries -- the two are merged only at the
#    # total-years-of-experience calculation step (see experience.py),
#    # where the interval-merge logic already collapses overlapping/
#    # identical date ranges.
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
## Maps a keyword substring (same vocabulary as _SECTION_KEYWORDS, which is
## what caused score_heading_line to flag a line as heading-shaped in the
## first place) to the actual section label it corresponds to. Used by
## _resolve_unknown_heading() to turn a generic "unknown" classification
## into a real label instead of the line being blindly absorbed into
## whatever list-type section happens to be open.
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    # Checked BEFORE the generic "career" -> "experience" mapping below,
#    # since dict iteration order determines which mapping wins first on
#    # a substring match, and these are more specific than a bare "career".
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    """
#    Confirmed on a real resume (Rohit Shivnath Zimbar): _BARE_PHONE_LINE's
#    character class ([\\d\\-\\s()], bookended by digits, min length 8) has
#    no way to tell a phone number apart from a numeric date range --
#    "2022-04-03 - 2023-07-15" is entirely digits/hyphens/spaces and well
#    over 8 characters, so it matched. Four of his five job date ranges
#    (every one written with a plain ASCII hyphen rather than an en-dash)
#    got misclassified as bare contact info and silently deferred out of
#    the "experience" section entirely, landing at the top of "header"
#    instead -- not a parsing-quality issue, the date lines were deleted
#    from the section that needed them. _JOB_DATE_RANGE already recognizes
#    this exact numeric YYYY-MM-DD shape, so checking it first and bailing
#    out on a real date-range match fixes this without narrowing what
#    _BARE_PHONE_LINE itself is allowed to match (still needed for actual
#    phone numbers that happen to be dash-grouped).
#    """
#    stripped = line.strip()
#    if not stripped:
#        return False
#    if _JOB_DATE_RANGE.search(stripped):
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
## Day-first format: "04 May, 2023", "08 Dec 2025" -- see the matching fix
## in experience.py's DATE_RANGE_RE (Sachin Chauhan's resume: his current
## job's date range used this shape on both ends, and nothing in the old
## alternation set -- "Month YYYY" or numeric-only -- could match it).
## Mirrored here so _looks_like_job_entry() recognizes this shape too.
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
## Widened to also recognize month-name dates ("Oct 2023 – Nov 2025"), not
## just numeric formats. Confirmed on a real resume (Sachin Chitranshi):
## a "PROFESSIONAL SUMMARY" heading sits directly above "KPMG (Oct 2023 –
## Nov 2025)" -- the detailed job history for all 5 roles, not an actual
## summary. The summary->experience promotion check further down (see
## split_into_sections) already exists for exactly this shape of mistake
## (confirmed working on Aarti Suranje's resume), but it depends entirely
## on _looks_like_job_entry() recognizing the date range on the next
## line -- and a month-name-only date range like "Oct 2023 – Nov 2025"
## never matched here, so the promotion silently never fired and this
## entire job history (KPMG, HCL, EY, Sailfin, MobiQuest -- ~100 lines of
## real experience) stayed mislabeled as "summary".
#_JOB_DATE_RANGE = re.compile(
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    # Same bare-space-before-open-ended-marker widening as experience.py's
#    # DATE_RANGE_RE (see the "till date" fix from earlier), applied here
#    # too for consistency.
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    """
#    Some resume templates (Canva-style / heavily tracked heading fonts)
#    render section headings with a literal space inserted between every
#    letter after PDF text extraction, e.g. "P R O F E S S I O N A L
#    S U M M A R Y". Word boundaries are lost in the process -- there's no
#    double-space between "PROFESSIONAL" and "SUMMARY" either, both are
#    single spaces, identical to the inter-letter spacing -- so none of
#    the normal SECTION_PATTERNS regexes (which expect whole words) can
#    ever match. score_heading_line can't rescue it either: word_count
#    balloons past its own >8-word cutoff since every letter counts as
#    its own "word" (e.g. "EDUCATION" alone becomes word_count == 9).
#    Confirmed on a real resume (Ritika Jain): every heading in the
#    document used this style and the entire file collapsed into one
#    "header" section as a result -- not a partial miss, a total one.
#    """
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    """
#    Collapses a letter-spaced heading candidate back into one solid
#    string and matches it against the same keyword vocabulary used
#    elsewhere (_KEYWORD_TO_LABEL), rather than trying to reconstruct
#    word boundaries -- which isn't recoverable from spacing alone, since
#    inter-word and inter-letter spacing are identical in the source.
#    "P R O F E S S I O N A L S U M M A R Y" -> "professionalsummary",
#    which contains "summary" -> resolves to the "summary" label.
#    """
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    """
#    A line inside a list-type section that scored 'unknown' via
#    score_heading_line is ambiguous: it could be a harmless sub-label
#    (e.g. "Programming Languages:" inside a skills list, followed on the
#    same line by the actual list of languages) or a genuine new section
#    whose heading just didn't match any SECTION_PATTERNS regex (e.g.
#    "Certifications (Technical + Management)" landing inside "experience",
#    "KEY ACHIEVEMENTS" landing inside "strengths" -- both confirmed on
#    real resumes). Both look identical to the caller: label == "unknown".
#
#    We re-derive a real label here from the same keyword vocabulary that
#    caused score_heading_line to score this line highly in the first
#    place (_SECTION_KEYWORDS / _contains_section_keyword), instead of
#    silently absorbing it either way.
#
#    Guard: only promote when the line looks like a BARE heading. If there
#    is real content after a trailing colon (e.g. "Certifications: AWS,
#    Azure" as one bullet-style line among others in a skills list), this
#    is a sub-label with inline content, not a new section -- so we leave
#    it alone and let it be absorbed as before.
#    """
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            return mapped_label
#    return None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        # A heading can legitimately match on its own but still be the FIRST
#        # half of a line-wrapped heading (e.g. "Achievements" / "and
#        # Appreciations:"). Before committing, check whether folding in the
#        # very next line also produces a valid match -- if so, treat the
#        # next line as part of the heading, not as body text. Only extends
#        # when the next line is short, has no bullet marker, and isn't
#        # itself a date-range line, so this can't misfire on a heading
#        # immediately followed by real bullet/date content.
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        # Letter-spaced heading check runs before the fuzzy-match fallback:
#        # a letter-spaced candidate like "P R O F E S S I O N A L
#        # S U M M A R Y" still contains real spaces, so it would never
#        # pass _fuzzy_heading_label's word.isalpha() check anyway (spaces
#        # aren't alphabetic), but resolving it here first is more direct
#        # and avoids relying on that accidental non-collision.
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    # Try to resolve straight to a real label here, not
#                    # only later when deciding whether to absorb into an
#                    # already-open list section. Confirmed on a real
#                    # resume (Sachin Chitranshi): "LAST 5 CAREER
#                    # TIMELINE" transitions out of a "summary" section
#                    # (not a list-type section), so the absorption check
#                    # further down never even runs for it -- without
#                    # resolving here, it opens as a bare "unknown"
#                    # section instead of "early_career".
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        # "Languages" is genuinely ambiguous on resumes: it can mean spoken
#        # languages (English, Hindi...) OR it can be a row label inside a
#        # technical skills matrix ("Languages: PHP, JavaScript"). The
#        # SECTION_PATTERNS regex matches the bare word correctly either
#        # way -- it has no way to know which sense is meant. Confirmed on
#        # a real resume (Sachin Chauhan): a skills-matrix table with row
#        # labels "Database" / "Languages" / "Frameworks & Libraries" /
#        # "Web Tools" / "Tools" had its "Languages" row label match the
#        # real "languages" section pattern, splitting the skills matrix in
#        # half and dragging every subsequent row into a bogus "languages"
#        # section. Disambiguate using what actually follows the heading --
#        # tech-stack terms (php, react, mysql...) mean this is a skills
#        # row, not a spoken-languages section -- and suppress the split in
#        # that case so the line falls through and gets absorbed into
#        # whatever section (usually "skills") is already open.
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        # This must run BEFORE the same-label-absorb check directly below:
#        # a second "summary"-labelled heading (e.g. "Professional Summary"
#        # appearing after an earlier "Personal Summary") sometimes isn't
#        # really a second summary at all -- it's actually introducing the
#        # first job entry (confirmed on a real resume, Aarti Suranje: her
#        # "PROFESSIONAL SUMMARY" heading sits directly above "PRINCIPAL
#        # CONSULTANT, 01/2023 - Current"). If this promotion ran AFTER the
#        # same-label-absorb check instead, the absorb-and-continue would
#        # fire first (both headings map to "summary") and this promotion
#        # would never get a chance to run -- her entire job history
#        # (Principal Consultant, Technology Lead, ...) would silently
#        # merge into "summary" instead of becoming its own "experience"
#        # section.
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            # Don't blindly absorb: this line scored heading-shaped and
#            # contains a section keyword -- check whether it actually
#            # resolves to a real, different section label before treating
#            # it as harmless in-list noise. See _resolve_unknown_heading().
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#








#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 21)
#
#CHANGELOG (rev 21):
#- _JOB_DATE_RANGE now recognizes day-first date ranges ("04 May, 2023 –
#  08 Dec 2025"), mirroring the same fix applied to experience.py's
#  DATE_RANGE_RE. Confirmed on a real resume (Sachin Chauhan).
#
#CHANGELOG (rev 20):
#- _JOB_DATE_RANGE now recognizes month-name date ranges ("Oct 2023 –
#  Nov 2025"), not just numeric formats. This directly fixes the
#  summary->experience promotion check for resumes whose real job
#  history sits under a heading that also matches "summary" (e.g. a
#  second "PROFESSIONAL SUMMARY" heading actually introducing detailed
#  job entries) when those job entries use month names rather than
#  numeric dates. Confirmed on a real resume (Sachin Chitranshi).
#
#CHANGELOG (rev 19):
#- Widened the "skills" pattern to cover a broader alias list: "technical
#  expertise", "technology stack", "technology summary", "knowledge
#  summary", "knowledge base", "technical snapshot", "skill matrix".
#- Added tech-stack vs spoken-language disambiguation for the "languages"
#  label. Confirmed on a real resume (Sachin Chauhan): a skills-matrix
#  table used "Languages" as a row label ("Languages: PHP, JavaScript"),
#  which correctly matched the real "languages" section pattern by its
#  own rules, but wrongly split the skills matrix into a separate
#  "languages" section and dragged every following row into it. Now,
#  before committing a "languages" match, we check what follows the
#  heading -- tech-stack terms (php, react, mysql...) suppress the split
#  so the line falls through and gets absorbed into whatever section
#  (usually "skills") is already open; recognizable spoken-language names
#  (English, Hindi...) still split normally. See
#  _looks_like_tech_stack_content / _looks_like_spoken_language_content.
#
#CHANGELOG (rev 18):
#- Added a new "early_career" section label for condensed one-line-per-job
#  recap tables (e.g. "LAST 5 CAREER TIMELINE" followed by lines like
#  "Oct-2023 - Nov-2025 | Manager | KPMG"), kept distinct from the main
#  "experience" section so detailed bullets and condensed recap entries
#  don't get merged into one messy section.
#- Generalized unknown-heading resolution: previously _resolve_unknown_heading()
#  only ran when deciding whether to absorb a heading-scored line into an
#  already-open list-type section. A heading transitioning FROM a
#  non-list section (e.g. "summary") straight into "unknown" never got a
#  chance to resolve at all -- confirmed on a real resume (Sachin
#  Chitranshi) where this was exactly why "LAST 5 CAREER TIMELINE" landed
#  as a bare "unknown" section instead of a real label. Resolution now
#  runs at the point "unknown" is first assigned, not just later at the
#  absorption-check.
#
#CHANGELOG (rev 17):
#- Added _is_letter_spaced_heading() / _resolve_letter_spaced_heading() to
#  handle resume templates whose headings extract with a literal space
#  between every letter (e.g. "P R O F E S S I O N A L S U M M A R Y"),
#  which defeats both SECTION_PATTERNS (word-boundary regexes) and
#  score_heading_line (word-count cutoff) simultaneously. Confirmed on a
#  real resume (Ritika Jain) where this collapsed the entire document
#  into one "header" section. See _resolve_letter_spaced_heading's
#  docstring for the matching approach.
#
#CHANGELOG (rev 16):
#- certifications/achievements patterns widened to tolerate a trailing
#  parenthetical qualifier (e.g. "Certifications (Technical + Management)")
#  and, for achievements, a leading qualifier word (e.g. "KEY ACHIEVEMENTS"),
#  matching the prefix support "skills" already had.
#- Added _resolve_unknown_heading() + updated the _LIST_SECTION_LABELS
#  absorption check in split_into_sections(). Previously ANY line scoring
#  "unknown" inside a list-type section (skills/education/experience/etc.)
#  was silently absorbed into that section, with no way to tell a harmless
#  sub-label ("Programming Languages:" inside a skills list) apart from a
#  genuine new section heading whose exact phrasing wasn't in
#  SECTION_PATTERNS (e.g. "Certifications (Technical + Management)" landing
#  inside "experience", "KEY ACHIEVEMENTS" landing inside "strengths" --
#  both confirmed on real resumes, Sachin Singh and Saheb Jaggi). Now, before
#  absorbing, we try to resolve the line to a real label via the same
#  keyword set score_heading_line already used to flag it as heading-shaped
#  in the first place. Only promotes when the line looks like a bare heading
#  (nothing meaningful after a trailing colon) so a genuine in-list bullet
#  like "Certifications: AWS, Azure" is NOT mistaken for a new section.
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
## Core keywords widened to also cover "expertise" (e.g. "technical
## expertise"), and the optional suffix widened to also cover "matrix"
## (e.g. "skill matrix") alongside the existing "sets?" ("skill set").
## Standalone phrases that don't have "skill"/"competenc*"/"expertise" as
## their own head noun at all -- "technology stack", "technology summary",
## "knowledge summary", "knowledge base", "technical snapshot" -- are
## added as separate top-level alternatives instead, since they don't fit
## the prefix+core structure.
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"((work|industry|relevant|professional|previous)\s*(and\s*)?){0,2}experience",
#        r"employment(\s+(history|details|records?|background))?", r"professional\s*background",
#        r"career\s*history", r"work\s*history", r"internships?",
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    # Widened to tolerate a trailing parenthetical qualifier, e.g.
#    # "Certifications (Technical + Management)" (confirmed on Sachin
#    # Singh's resume) which the old pattern rejected outright since it
#    # only allowed an "& word" / "and word" suffix, not a parenthetical.
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    # Widened to tolerate a leading qualifier word, e.g. "KEY ACHIEVEMENTS"
#    # (confirmed on Saheb Jaggi's resume) -- "skills" already supported an
#    # analogous prefix group; achievements never got the same treatment.
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    # Condensed "here's my last N jobs in one line each" recap, distinct
#    # from the full "experience" section's detailed per-job bullets.
#    # Confirmed on a real resume (Sachin Chitranshi): "LAST 5 CAREER
#    # TIMELINE" headed a block of one-line entries like "Oct-2023 -
#    # Nov-2025 | Manager | KPMG" that restate the same jobs covered in
#    # full detail later in the document. Kept as its own label (not
#    # folded into "experience") so the detailed section isn't polluted
#    # with duplicate condensed entries -- the two are merged only at the
#    # total-years-of-experience calculation step (see experience.py),
#    # where the interval-merge logic already collapses overlapping/
#    # identical date ranges.
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
## Maps a keyword substring (same vocabulary as _SECTION_KEYWORDS, which is
## what caused score_heading_line to flag a line as heading-shaped in the
## first place) to the actual section label it corresponds to. Used by
## _resolve_unknown_heading() to turn a generic "unknown" classification
## into a real label instead of the line being blindly absorbed into
## whatever list-type section happens to be open.
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    # Checked BEFORE the generic "career" -> "experience" mapping below,
#    # since dict iteration order determines which mapping wins first on
#    # a substring match, and these are more specific than a bare "career".
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_MONTH_NAMES = (
#    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
#    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
#    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
#)
#
## Day-first format: "04 May, 2023", "08 Dec 2025" -- see the matching fix
## in experience.py's DATE_RANGE_RE (Sachin Chauhan's resume: his current
## job's date range used this shape on both ends, and nothing in the old
## alternation set -- "Month YYYY" or numeric-only -- could match it).
## Mirrored here so _looks_like_job_entry() recognizes this shape too.
#_JOB_DAY_MONTH_YEAR = rf"\d{{1,2}}[\s,.]*{_JOB_MONTH_NAMES}[\s,.]*\d{{4}}"
#
## Widened to also recognize month-name dates ("Oct 2023 – Nov 2025"), not
## just numeric formats. Confirmed on a real resume (Sachin Chitranshi):
## a "PROFESSIONAL SUMMARY" heading sits directly above "KPMG (Oct 2023 –
## Nov 2025)" -- the detailed job history for all 5 roles, not an actual
## summary. The summary->experience promotion check further down (see
## split_into_sections) already exists for exactly this shape of mistake
## (confirmed working on Aarti Suranje's resume), but it depends entirely
## on _looks_like_job_entry() recognizing the date range on the next
## line -- and a month-name-only date range like "Oct 2023 – Nov 2025"
## never matched here, so the promotion silently never fired and this
## entire job history (KPMG, HCL, EY, Sailfin, MobiQuest -- ~100 lines of
## real experience) stayed mislabeled as "summary".
#_JOB_DATE_RANGE = re.compile(
#    rf"(?:,\s*)?(?:{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))"
#    # Same bare-space-before-open-ended-marker widening as experience.py's
#    # DATE_RANGE_RE (see the "till date" fix from earlier), applied here
#    # too for consistency.
#    rf"\s*(?:[-–—\u2013\u2014]+\s*|(?=(?:Current|Present|Now|Till)\b))"
#    rf"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|"
#    rf"{_JOB_DAY_MONTH_YEAR}|{_JOB_MONTH_NAMES}[\s,.]*(?:\d{{4}}|['’‘`´]?\d{{2}})|"
#    rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}}|"
#    rf"(?:\d{{1,2}}/)?(?:\d{{4}}|['’‘`´]?\d{{2}}))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    """
#    Some resume templates (Canva-style / heavily tracked heading fonts)
#    render section headings with a literal space inserted between every
#    letter after PDF text extraction, e.g. "P R O F E S S I O N A L
#    S U M M A R Y". Word boundaries are lost in the process -- there's no
#    double-space between "PROFESSIONAL" and "SUMMARY" either, both are
#    single spaces, identical to the inter-letter spacing -- so none of
#    the normal SECTION_PATTERNS regexes (which expect whole words) can
#    ever match. score_heading_line can't rescue it either: word_count
#    balloons past its own >8-word cutoff since every letter counts as
#    its own "word" (e.g. "EDUCATION" alone becomes word_count == 9).
#    Confirmed on a real resume (Ritika Jain): every heading in the
#    document used this style and the entire file collapsed into one
#    "header" section as a result -- not a partial miss, a total one.
#    """
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    """
#    Collapses a letter-spaced heading candidate back into one solid
#    string and matches it against the same keyword vocabulary used
#    elsewhere (_KEYWORD_TO_LABEL), rather than trying to reconstruct
#    word boundaries -- which isn't recoverable from spacing alone, since
#    inter-word and inter-letter spacing are identical in the source.
#    "P R O F E S S I O N A L S U M M A R Y" -> "professionalsummary",
#    which contains "summary" -> resolves to the "summary" label.
#    """
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    """
#    A line inside a list-type section that scored 'unknown' via
#    score_heading_line is ambiguous: it could be a harmless sub-label
#    (e.g. "Programming Languages:" inside a skills list, followed on the
#    same line by the actual list of languages) or a genuine new section
#    whose heading just didn't match any SECTION_PATTERNS regex (e.g.
#    "Certifications (Technical + Management)" landing inside "experience",
#    "KEY ACHIEVEMENTS" landing inside "strengths" -- both confirmed on
#    real resumes). Both look identical to the caller: label == "unknown".
#
#    We re-derive a real label here from the same keyword vocabulary that
#    caused score_heading_line to score this line highly in the first
#    place (_SECTION_KEYWORDS / _contains_section_keyword), instead of
#    silently absorbing it either way.
#
#    Guard: only promote when the line looks like a BARE heading. If there
#    is real content after a trailing colon (e.g. "Certifications: AWS,
#    Azure" as one bullet-style line among others in a skills list), this
#    is a sub-label with inline content, not a new section -- so we leave
#    it alone and let it be absorbed as before.
#    """
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            return mapped_label
#    return None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        # A heading can legitimately match on its own but still be the FIRST
#        # half of a line-wrapped heading (e.g. "Achievements" / "and
#        # Appreciations:"). Before committing, check whether folding in the
#        # very next line also produces a valid match -- if so, treat the
#        # next line as part of the heading, not as body text. Only extends
#        # when the next line is short, has no bullet marker, and isn't
#        # itself a date-range line, so this can't misfire on a heading
#        # immediately followed by real bullet/date content.
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        # Letter-spaced heading check runs before the fuzzy-match fallback:
#        # a letter-spaced candidate like "P R O F E S S I O N A L
#        # S U M M A R Y" still contains real spaces, so it would never
#        # pass _fuzzy_heading_label's word.isalpha() check anyway (spaces
#        # aren't alphabetic), but resolving it here first is more direct
#        # and avoids relying on that accidental non-collision.
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    # Try to resolve straight to a real label here, not
#                    # only later when deciding whether to absorb into an
#                    # already-open list section. Confirmed on a real
#                    # resume (Sachin Chitranshi): "LAST 5 CAREER
#                    # TIMELINE" transitions out of a "summary" section
#                    # (not a list-type section), so the absorption check
#                    # further down never even runs for it -- without
#                    # resolving here, it opens as a bare "unknown"
#                    # section instead of "early_career".
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        # "Languages" is genuinely ambiguous on resumes: it can mean spoken
#        # languages (English, Hindi...) OR it can be a row label inside a
#        # technical skills matrix ("Languages: PHP, JavaScript"). The
#        # SECTION_PATTERNS regex matches the bare word correctly either
#        # way -- it has no way to know which sense is meant. Confirmed on
#        # a real resume (Sachin Chauhan): a skills-matrix table with row
#        # labels "Database" / "Languages" / "Frameworks & Libraries" /
#        # "Web Tools" / "Tools" had its "Languages" row label match the
#        # real "languages" section pattern, splitting the skills matrix in
#        # half and dragging every subsequent row into a bogus "languages"
#        # section. Disambiguate using what actually follows the heading --
#        # tech-stack terms (php, react, mysql...) mean this is a skills
#        # row, not a spoken-languages section -- and suppress the split in
#        # that case so the line falls through and gets absorbed into
#        # whatever section (usually "skills") is already open.
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        # This must run BEFORE the same-label-absorb check directly below:
#        # a second "summary"-labelled heading (e.g. "Professional Summary"
#        # appearing after an earlier "Personal Summary") sometimes isn't
#        # really a second summary at all -- it's actually introducing the
#        # first job entry (confirmed on a real resume, Aarti Suranje: her
#        # "PROFESSIONAL SUMMARY" heading sits directly above "PRINCIPAL
#        # CONSULTANT, 01/2023 - Current"). If this promotion ran AFTER the
#        # same-label-absorb check instead, the absorb-and-continue would
#        # fire first (both headings map to "summary") and this promotion
#        # would never get a chance to run -- her entire job history
#        # (Principal Consultant, Technology Lead, ...) would silently
#        # merge into "summary" instead of becoming its own "experience"
#        # section.
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            # Don't blindly absorb: this line scored heading-shaped and
#            # contains a section keyword -- check whether it actually
#            # resolves to a real, different section label before treating
#            # it as harmless in-list noise. See _resolve_unknown_heading().
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#











# worked - just commenting and putting above for one thing
#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 19)
#
#CHANGELOG (rev 19):
#- Widened the "skills" pattern to cover a broader alias list: "technical
#  expertise", "technology stack", "technology summary", "knowledge
#  summary", "knowledge base", "technical snapshot", "skill matrix".
#- Added tech-stack vs spoken-language disambiguation for the "languages"
#  label. Confirmed on a real resume (Sachin Chauhan): a skills-matrix
#  table used "Languages" as a row label ("Languages: PHP, JavaScript"),
#  which correctly matched the real "languages" section pattern by its
#  own rules, but wrongly split the skills matrix into a separate
#  "languages" section and dragged every following row into it. Now,
#  before committing a "languages" match, we check what follows the
#  heading -- tech-stack terms (php, react, mysql...) suppress the split
#  so the line falls through and gets absorbed into whatever section
#  (usually "skills") is already open; recognizable spoken-language names
#  (English, Hindi...) still split normally. See
#  _looks_like_tech_stack_content / _looks_like_spoken_language_content.
#
#CHANGELOG (rev 18):
#- Added a new "early_career" section label for condensed one-line-per-job
#  recap tables (e.g. "LAST 5 CAREER TIMELINE" followed by lines like
#  "Oct-2023 - Nov-2025 | Manager | KPMG"), kept distinct from the main
#  "experience" section so detailed bullets and condensed recap entries
#  don't get merged into one messy section.
#- Generalized unknown-heading resolution: previously _resolve_unknown_heading()
#  only ran when deciding whether to absorb a heading-scored line into an
#  already-open list-type section. A heading transitioning FROM a
#  non-list section (e.g. "summary") straight into "unknown" never got a
#  chance to resolve at all -- confirmed on a real resume (Sachin
#  Chitranshi) where this was exactly why "LAST 5 CAREER TIMELINE" landed
#  as a bare "unknown" section instead of a real label. Resolution now
#  runs at the point "unknown" is first assigned, not just later at the
#  absorption-check.
#
#CHANGELOG (rev 17):
#- Added _is_letter_spaced_heading() / _resolve_letter_spaced_heading() to
#  handle resume templates whose headings extract with a literal space
#  between every letter (e.g. "P R O F E S S I O N A L S U M M A R Y"),
#  which defeats both SECTION_PATTERNS (word-boundary regexes) and
#  score_heading_line (word-count cutoff) simultaneously. Confirmed on a
#  real resume (Ritika Jain) where this collapsed the entire document
#  into one "header" section. See _resolve_letter_spaced_heading's
#  docstring for the matching approach.
#
#CHANGELOG (rev 16):
#- certifications/achievements patterns widened to tolerate a trailing
#  parenthetical qualifier (e.g. "Certifications (Technical + Management)")
#  and, for achievements, a leading qualifier word (e.g. "KEY ACHIEVEMENTS"),
#  matching the prefix support "skills" already had.
#- Added _resolve_unknown_heading() + updated the _LIST_SECTION_LABELS
#  absorption check in split_into_sections(). Previously ANY line scoring
#  "unknown" inside a list-type section (skills/education/experience/etc.)
#  was silently absorbed into that section, with no way to tell a harmless
#  sub-label ("Programming Languages:" inside a skills list) apart from a
#  genuine new section heading whose exact phrasing wasn't in
#  SECTION_PATTERNS (e.g. "Certifications (Technical + Management)" landing
#  inside "experience", "KEY ACHIEVEMENTS" landing inside "strengths" --
#  both confirmed on real resumes, Sachin Singh and Saheb Jaggi). Now, before
#  absorbing, we try to resolve the line to a real label via the same
#  keyword set score_heading_line already used to flag it as heading-shaped
#  in the first place. Only promotes when the line looks like a bare heading
#  (nothing meaningful after a trailing colon) so a genuine in-list bullet
#  like "Certifications: AWS, Azure" is NOT mistaken for a new section.
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
## Core keywords widened to also cover "expertise" (e.g. "technical
## expertise"), and the optional suffix widened to also cover "matrix"
## (e.g. "skill matrix") alongside the existing "sets?" ("skill set").
## Standalone phrases that don't have "skill"/"competenc*"/"expertise" as
## their own head noun at all -- "technology stack", "technology summary",
## "knowledge summary", "knowledge base", "technical snapshot" -- are
## added as separate top-level alternatives instead, since they don't fit
## the prefix+core structure.
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies)|expertise)(?:\s+(sets?|matrix))?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization"
#    rf"|technology\s+stack"
#    rf"|technology\s+summary"
#    rf"|knowledge\s+summary"
#    rf"|knowledge\s+base"
#    rf"|technical\s+snapshot)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"((work|industry|relevant|professional|previous)\s*(and\s*)?){0,2}experience",
#        r"employment(\s+(history|details|records?|background))?", r"professional\s*background",
#        r"career\s*history", r"work\s*history", r"internships?",
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    # Widened to tolerate a trailing parenthetical qualifier, e.g.
#    # "Certifications (Technical + Management)" (confirmed on Sachin
#    # Singh's resume) which the old pattern rejected outright since it
#    # only allowed an "& word" / "and word" suffix, not a parenthetical.
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    # Widened to tolerate a leading qualifier word, e.g. "KEY ACHIEVEMENTS"
#    # (confirmed on Saheb Jaggi's resume) -- "skills" already supported an
#    # analogous prefix group; achievements never got the same treatment.
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#    # Condensed "here's my last N jobs in one line each" recap, distinct
#    # from the full "experience" section's detailed per-job bullets.
#    # Confirmed on a real resume (Sachin Chitranshi): "LAST 5 CAREER
#    # TIMELINE" headed a block of one-line entries like "Oct-2023 -
#    # Nov-2025 | Manager | KPMG" that restate the same jobs covered in
#    # full detail later in the document. Kept as its own label (not
#    # folded into "experience") so the detailed section isn't polluted
#    # with duplicate condensed entries -- the two are merged only at the
#    # total-years-of-experience calculation step (see experience.py),
#    # where the interval-merge logic already collapses overlapping/
#    # identical date ranges.
#    "early_career": re.compile(
#        r"^(?:early\s+career(s)?"
#        r"|last\s+\d+\s+(years?\s+)?career\s+timeline"
#        r"|career\s+timeline"
#        r"|career\s+synopsis"
#        r"|career\s+snapshot"
#        r"|career\s+at\s+a\s+glance"
#        r"|(prior|past|previous)\s+engagements?)$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#    "early_career",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward", "synopsis", "snapshot", "glance", "engagement",
#]
#
#
## Maps a keyword substring (same vocabulary as _SECTION_KEYWORDS, which is
## what caused score_heading_line to flag a line as heading-shaped in the
## first place) to the actual section label it corresponds to. Used by
## _resolve_unknown_heading() to turn a generic "unknown" classification
## into a real label instead of the line being blindly absorbed into
## whatever list-type section happens to be open.
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    # Checked BEFORE the generic "career" -> "experience" mapping below,
#    # since dict iteration order determines which mapping wins first on
#    # a substring match, and these are more specific than a bare "career".
#    "career timeline": "early_career", "career synopsis": "early_career",
#    "career snapshot": "early_career", "career at a glance": "early_career",
#    "early career": "early_career", "timeline": "early_career",
#    "synopsis": "early_career", "snapshot": "early_career",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2}))"
#    # Same widening as experience.py's DATE_RANGE_RE: a dash is the
#    # common case, but "<date> till date" with no connecting word at all
#    # (just whitespace) also occurs. The lookahead alternative consumes
#    # nothing, so the literal "Present"/"Till Date"/etc. text is still
#    # matched by the end alternation right after it, not swallowed here.
#    r"\s*(?:[-–—\u2013\u2014]+\s*|(?=(?:Current|Present|Now|Till)\b))"
#    r"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2})))",
#    re.IGNORECASE
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    """
#    Some resume templates (Canva-style / heavily tracked heading fonts)
#    render section headings with a literal space inserted between every
#    letter after PDF text extraction, e.g. "P R O F E S S I O N A L
#    S U M M A R Y". Word boundaries are lost in the process -- there's no
#    double-space between "PROFESSIONAL" and "SUMMARY" either, both are
#    single spaces, identical to the inter-letter spacing -- so none of
#    the normal SECTION_PATTERNS regexes (which expect whole words) can
#    ever match. score_heading_line can't rescue it either: word_count
#    balloons past its own >8-word cutoff since every letter counts as
#    its own "word" (e.g. "EDUCATION" alone becomes word_count == 9).
#    Confirmed on a real resume (Ritika Jain): every heading in the
#    document used this style and the entire file collapsed into one
#    "header" section as a result -- not a partial miss, a total one.
#    """
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    """
#    Collapses a letter-spaced heading candidate back into one solid
#    string and matches it against the same keyword vocabulary used
#    elsewhere (_KEYWORD_TO_LABEL), rather than trying to reconstruct
#    word boundaries -- which isn't recoverable from spacing alone, since
#    inter-word and inter-letter spacing are identical in the source.
#    "P R O F E S S I O N A L S U M M A R Y" -> "professionalsummary",
#    which contains "summary" -> resolves to the "summary" label.
#    """
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#_TECH_STACK_KEYWORDS = {
#    "php", "html", "html5", "css", "css3", "javascript", "js", "java", "python",
#    "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "react",
#    "reactjs", "angular", "angularjs", "vue", "vuejs", "node", "nodejs",
#    "jquery", "bootstrap", "laravel", "codeigniter", "django", "flask",
#    "spring", "typescript", "ruby", "rails", "golang", "kotlin", "swift",
#    "dotnet", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
#    "ajax", "rest", "graphql", "redux", "express", "webpack", "sass", "less",
#    "xml", "json", "linux", "c", "c++", "c#", "r", "scala", "perl", "bash",
#    "shell", "matlab", "sqlite", "oracle", "firebase", "npm", "yarn",
#}
#
#_SPOKEN_LANGUAGE_KEYWORDS = {
#    "english", "hindi", "spanish", "french", "german", "mandarin", "chinese",
#    "cantonese", "arabic", "portuguese", "russian", "japanese", "korean",
#    "italian", "punjabi", "bengali", "tamil", "telugu", "marathi", "gujarati",
#    "urdu", "kannada", "malayalam", "dutch", "turkish", "vietnamese", "thai",
#    "polish", "swedish", "greek", "hebrew", "indonesian", "farsi", "persian",
#}
#
#_WORD_TOKEN_RE = re.compile(r"[a-zA-Z+#.]+")
#
#
#def _looks_like_tech_stack_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _TECH_STACK_KEYWORDS for w in words)
#
#
#def _looks_like_spoken_language_content(text: str) -> bool:
#    words = _WORD_TOKEN_RE.findall(text.lower())
#    return any(w in _SPOKEN_LANGUAGE_KEYWORDS for w in words)
#
#
#def _resolve_unknown_heading(stripped: str):
#    """
#    A line inside a list-type section that scored 'unknown' via
#    score_heading_line is ambiguous: it could be a harmless sub-label
#    (e.g. "Programming Languages:" inside a skills list, followed on the
#    same line by the actual list of languages) or a genuine new section
#    whose heading just didn't match any SECTION_PATTERNS regex (e.g.
#    "Certifications (Technical + Management)" landing inside "experience",
#    "KEY ACHIEVEMENTS" landing inside "strengths" -- both confirmed on
#    real resumes). Both look identical to the caller: label == "unknown".
#
#    We re-derive a real label here from the same keyword vocabulary that
#    caused score_heading_line to score this line highly in the first
#    place (_SECTION_KEYWORDS / _contains_section_keyword), instead of
#    silently absorbing it either way.
#
#    Guard: only promote when the line looks like a BARE heading. If there
#    is real content after a trailing colon (e.g. "Certifications: AWS,
#    Azure" as one bullet-style line among others in a skills list), this
#    is a sub-label with inline content, not a new section -- so we leave
#    it alone and let it be absorbed as before.
#    """
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            return mapped_label
#    return None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        # A heading can legitimately match on its own but still be the FIRST
#        # half of a line-wrapped heading (e.g. "Achievements" / "and
#        # Appreciations:"). Before committing, check whether folding in the
#        # very next line also produces a valid match -- if so, treat the
#        # next line as part of the heading, not as body text. Only extends
#        # when the next line is short, has no bullet marker, and isn't
#        # itself a date-range line, so this can't misfire on a heading
#        # immediately followed by real bullet/date content.
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        # Letter-spaced heading check runs before the fuzzy-match fallback:
#        # a letter-spaced candidate like "P R O F E S S I O N A L
#        # S U M M A R Y" still contains real spaces, so it would never
#        # pass _fuzzy_heading_label's word.isalpha() check anyway (spaces
#        # aren't alphabetic), but resolving it here first is more direct
#        # and avoids relying on that accidental non-collision.
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    # Try to resolve straight to a real label here, not
#                    # only later when deciding whether to absorb into an
#                    # already-open list section. Confirmed on a real
#                    # resume (Sachin Chitranshi): "LAST 5 CAREER
#                    # TIMELINE" transitions out of a "summary" section
#                    # (not a list-type section), so the absorption check
#                    # further down never even runs for it -- without
#                    # resolving here, it opens as a bare "unknown"
#                    # section instead of "early_career".
#                    resolved = _resolve_unknown_heading(stripped)
#                    if resolved is not None:
#                        label = resolved
#                        confidence = max(heading_score, 0.8)
#                    else:
#                        label = "unknown"
#                        confidence = heading_score
#
#        # "Languages" is genuinely ambiguous on resumes: it can mean spoken
#        # languages (English, Hindi...) OR it can be a row label inside a
#        # technical skills matrix ("Languages: PHP, JavaScript"). The
#        # SECTION_PATTERNS regex matches the bare word correctly either
#        # way -- it has no way to know which sense is meant. Confirmed on
#        # a real resume (Sachin Chauhan): a skills-matrix table with row
#        # labels "Database" / "Languages" / "Frameworks & Libraries" /
#        # "Web Tools" / "Tools" had its "Languages" row label match the
#        # real "languages" section pattern, splitting the skills matrix in
#        # half and dragging every subsequent row into a bogus "languages"
#        # section. Disambiguate using what actually follows the heading --
#        # tech-stack terms (php, react, mysql...) mean this is a skills
#        # row, not a spoken-languages section -- and suppress the split in
#        # that case so the line falls through and gets absorbed into
#        # whatever section (usually "skills") is already open.
#        if label == "languages" and stripped != "":
#            lookahead = inline_content if inline_content else _next_nonblank_line(lines, i + 1)
#            if lookahead and _looks_like_tech_stack_content(lookahead) and not _looks_like_spoken_language_content(lookahead):
#                label = None
#                confidence = 0.0
#                inline_content = None
#
#        # This must run BEFORE the same-label-absorb check directly below:
#        # a second "summary"-labelled heading (e.g. "Professional Summary"
#        # appearing after an earlier "Personal Summary") sometimes isn't
#        # really a second summary at all -- it's actually introducing the
#        # first job entry (confirmed on a real resume, Aarti Suranje: her
#        # "PROFESSIONAL SUMMARY" heading sits directly above "PRINCIPAL
#        # CONSULTANT, 01/2023 - Current"). If this promotion ran AFTER the
#        # same-label-absorb check instead, the absorb-and-continue would
#        # fire first (both headings map to "summary") and this promotion
#        # would never get a chance to run -- her entire job history
#        # (Principal Consultant, Technology Lead, ...) would silently
#        # merge into "summary" instead of becoming its own "experience"
#        # section.
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            # Don't blindly absorb: this line scored heading-shaped and
#            # contains a section keyword -- check whether it actually
#            # resolves to a real, different section label before treating
#            # it as harmless in-list noise. See _resolve_unknown_heading().
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#
#














##Done, just writting above to work some other technical words.
#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 17)
#
#CHANGELOG (rev 17):
#- Added _is_letter_spaced_heading() / _resolve_letter_spaced_heading() to
#  handle resume templates whose headings extract with a literal space
#  between every letter (e.g. "P R O F E S S I O N A L S U M M A R Y"),
#  which defeats both SECTION_PATTERNS (word-boundary regexes) and
#  score_heading_line (word-count cutoff) simultaneously. Confirmed on a
#  real resume (Ritika Jain) where this collapsed the entire document
#  into one "header" section. See _resolve_letter_spaced_heading's
#  docstring for the matching approach.
#
#CHANGELOG (rev 16):
#- certifications/achievements patterns widened to tolerate a trailing
#  parenthetical qualifier (e.g. "Certifications (Technical + Management)")
#  and, for achievements, a leading qualifier word (e.g. "KEY ACHIEVEMENTS"),
#  matching the prefix support "skills" already had.
#- Added _resolve_unknown_heading() + updated the _LIST_SECTION_LABELS
#  absorption check in split_into_sections(). Previously ANY line scoring
#  "unknown" inside a list-type section (skills/education/experience/etc.)
#  was silently absorbed into that section, with no way to tell a harmless
#  sub-label ("Programming Languages:" inside a skills list) apart from a
#  genuine new section heading whose exact phrasing wasn't in
#  SECTION_PATTERNS (e.g. "Certifications (Technical + Management)" landing
#  inside "experience", "KEY ACHIEVEMENTS" landing inside "strengths" --
#  both confirmed on real resumes, Sachin Singh and Saheb Jaggi). Now, before
#  absorbing, we try to resolve the line to a real label via the same
#  keyword set score_heading_line already used to flag it as heading-shaped
#  in the first place. Only promotes when the line looks like a bare heading
#  (nothing meaningful after a trailing colon) so a genuine in-list bullet
#  like "Certifications: AWS, Azure" is NOT mistaken for a new section.
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies))(?:\s+sets?)?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"((work|industry|relevant|professional|previous)\s*(and\s*)?){0,2}experience",
#        r"employment(\s+(history|details|records?|background))?", r"professional\s*background",
#        r"career\s*history", r"work\s*history", r"internships?",
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    # Widened to tolerate a trailing parenthetical qualifier, e.g.
#    # "Certifications (Technical + Management)" (confirmed on Sachin
#    # Singh's resume) which the old pattern rejected outright since it
#    # only allowed an "& word" / "and word" suffix, not a parenthetical.
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    # Widened to tolerate a leading qualifier word, e.g. "KEY ACHIEVEMENTS"
#    # (confirmed on Saheb Jaggi's resume) -- "skills" already supported an
#    # analogous prefix group; achievements never got the same treatment.
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward",
#]
#
#
## Maps a keyword substring (same vocabulary as _SECTION_KEYWORDS, which is
## what caused score_heading_line to flag a line as heading-shaped in the
## first place) to the actual section label it corresponds to. Used by
## _resolve_unknown_heading() to turn a generic "unknown" classification
## into a real label instead of the line being blindly absorbed into
## whatever list-type section happens to be open.
#_KEYWORD_TO_LABEL = {
#    "summary": "summary", "objective": "summary",
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2}))\s*[-–—\u2013\u2014]+\s*"
#    r"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2})))"
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#_LETTER_SPACED_TOKEN_RE = re.compile(r"^[A-Za-z0-9&]$")
#
#
#def _is_letter_spaced_heading(line: str) -> bool:
#    """
#    Some resume templates (Canva-style / heavily tracked heading fonts)
#    render section headings with a literal space inserted between every
#    letter after PDF text extraction, e.g. "P R O F E S S I O N A L
#    S U M M A R Y". Word boundaries are lost in the process -- there's no
#    double-space between "PROFESSIONAL" and "SUMMARY" either, both are
#    single spaces, identical to the inter-letter spacing -- so none of
#    the normal SECTION_PATTERNS regexes (which expect whole words) can
#    ever match. score_heading_line can't rescue it either: word_count
#    balloons past its own >8-word cutoff since every letter counts as
#    its own "word" (e.g. "EDUCATION" alone becomes word_count == 9).
#    Confirmed on a real resume (Ritika Jain): every heading in the
#    document used this style and the entire file collapsed into one
#    "header" section as a result -- not a partial miss, a total one.
#    """
#    tokens = line.split()
#    if len(tokens) < 4:
#        return False
#    single_char = sum(1 for t in tokens if _LETTER_SPACED_TOKEN_RE.match(t))
#    return (single_char / len(tokens)) >= 0.7
#
#
#def _resolve_letter_spaced_heading(line: str):
#    """
#    Collapses a letter-spaced heading candidate back into one solid
#    string and matches it against the same keyword vocabulary used
#    elsewhere (_KEYWORD_TO_LABEL), rather than trying to reconstruct
#    word boundaries -- which isn't recoverable from spacing alone, since
#    inter-word and inter-letter spacing are identical in the source.
#    "P R O F E S S I O N A L S U M M A R Y" -> "professionalsummary",
#    which contains "summary" -> resolves to the "summary" label.
#    """
#    if not _is_letter_spaced_heading(line):
#        return None
#    collapsed = "".join(line.split()).lower()
#    if len(collapsed) < 4:
#        return None
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in collapsed:
#            return mapped_label
#    return None
#
#
#def _resolve_unknown_heading(stripped: str):
#    """
#    A line inside a list-type section that scored 'unknown' via
#    score_heading_line is ambiguous: it could be a harmless sub-label
#    (e.g. "Programming Languages:" inside a skills list, followed on the
#    same line by the actual list of languages) or a genuine new section
#    whose heading just didn't match any SECTION_PATTERNS regex (e.g.
#    "Certifications (Technical + Management)" landing inside "experience",
#    "KEY ACHIEVEMENTS" landing inside "strengths" -- both confirmed on
#    real resumes). Both look identical to the caller: label == "unknown".
#
#    We re-derive a real label here from the same keyword vocabulary that
#    caused score_heading_line to score this line highly in the first
#    place (_SECTION_KEYWORDS / _contains_section_keyword), instead of
#    silently absorbing it either way.
#
#    Guard: only promote when the line looks like a BARE heading. If there
#    is real content after a trailing colon (e.g. "Certifications: AWS,
#    Azure" as one bullet-style line among others in a skills list), this
#    is a sub-label with inline content, not a new section -- so we leave
#    it alone and let it be absorbed as before.
#    """
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            return mapped_label
#    return None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        # A heading can legitimately match on its own but still be the FIRST
#        # half of a line-wrapped heading (e.g. "Achievements" / "and
#        # Appreciations:"). Before committing, check whether folding in the
#        # very next line also produces a valid match -- if so, treat the
#        # next line as part of the heading, not as body text. Only extends
#        # when the next line is short, has no bullet marker, and isn't
#        # itself a date-range line, so this can't misfire on a heading
#        # immediately followed by real bullet/date content.
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        # Letter-spaced heading check runs before the fuzzy-match fallback:
#        # a letter-spaced candidate like "P R O F E S S I O N A L
#        # S U M M A R Y" still contains real spaces, so it would never
#        # pass _fuzzy_heading_label's word.isalpha() check anyway (spaces
#        # aren't alphabetic), but resolving it here first is more direct
#        # and avoids relying on that accidental non-collision.
#        if label is None and stripped != "":
#            letter_spaced_label = _resolve_letter_spaced_heading(stripped)
#            if letter_spaced_label is not None:
#                label = letter_spaced_label
#                confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        # This must run BEFORE the same-label-absorb check directly below:
#        # a second "summary"-labelled heading (e.g. "Professional Summary"
#        # appearing after an earlier "Personal Summary") sometimes isn't
#        # really a second summary at all -- it's actually introducing the
#        # first job entry (confirmed on a real resume, Aarti Suranje: her
#        # "PROFESSIONAL SUMMARY" heading sits directly above "PRINCIPAL
#        # CONSULTANT, 01/2023 - Current"). If this promotion ran AFTER the
#        # same-label-absorb check instead, the absorb-and-continue would
#        # fire first (both headings map to "summary") and this promotion
#        # would never get a chance to run -- her entire job history
#        # (Principal Consultant, Technology Lead, ...) would silently
#        # merge into "summary" instead of becoming its own "experience"
#        # section.
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            # Don't blindly absorb: this line scored heading-shaped and
#            # contains a section keyword -- check whether it actually
#            # resolves to a real, different section label before treating
#            # it as harmless in-list noise. See _resolve_unknown_heading().
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#








#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 16)
#
#CHANGELOG (rev 16):
#- certifications/achievements patterns widened to tolerate a trailing
#  parenthetical qualifier (e.g. "Certifications (Technical + Management)")
#  and, for achievements, a leading qualifier word (e.g. "KEY ACHIEVEMENTS"),
#  matching the prefix support "skills" already had.
#- Added _resolve_unknown_heading() + updated the _LIST_SECTION_LABELS
#  absorption check in split_into_sections(). Previously ANY line scoring
#  "unknown" inside a list-type section (skills/education/experience/etc.)
#  was silently absorbed into that section, with no way to tell a harmless
#  sub-label ("Programming Languages:" inside a skills list) apart from a
#  genuine new section heading whose exact phrasing wasn't in
#  SECTION_PATTERNS (e.g. "Certifications (Technical + Management)" landing
#  inside "experience", "KEY ACHIEVEMENTS" landing inside "strengths" --
#  both confirmed on real resumes, Sachin Singh and Saheb Jaggi). Now, before
#  absorbing, we try to resolve the line to a real label via the same
#  keyword set score_heading_line already used to flag it as heading-shaped
#  in the first place. Only promotes when the line looks like a bare heading
#  (nothing meaningful after a trailing colon) so a genuine in-list bullet
#  like "Certifications: AWS, Azure" is NOT mistaken for a new section.
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies))(?:\s+sets?)?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"((work|industry|relevant|professional|previous)\s*(and\s*)?){0,2}experience",
#        r"employment(\s+(history|details|records?|background))?", r"professional\s*background",
#        r"career\s*history", r"work\s*history", r"internships?",
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    # Widened to tolerate a trailing parenthetical qualifier, e.g.
#    # "Certifications (Technical + Management)" (confirmed on Sachin
#    # Singh's resume) which the old pattern rejected outright since it
#    # only allowed an "& word" / "and word" suffix, not a parenthetical.
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?(\s*\([^)]*\))?", r"licen[sc]es?(\s*\([^)]*\))?",
#        r"accreditations?", r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    # Widened to tolerate a leading qualifier word, e.g. "KEY ACHIEVEMENTS"
#    # (confirmed on Saheb Jaggi's resume) -- "skills" already supported an
#    # analogous prefix group; achievements never got the same treatment.
#    "achievements": _build_section_pattern([
#        r"(key|major|notable|special|top)\s+achievements?",
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward",
#]
#
#
## Maps a keyword substring (same vocabulary as _SECTION_KEYWORDS, which is
## what caused score_heading_line to flag a line as heading-shaped in the
## first place) to the actual section label it corresponds to. Used by
## _resolve_unknown_heading() to turn a generic "unknown" classification
## into a real label instead of the line being blindly absorbed into
## whatever list-type section happens to be open.
#_KEYWORD_TO_LABEL = {
#    "skill": "skills", "expertise": "skills", "competenc": "skills", "technolog": "skills",
#    "experience": "experience", "employment": "experience", "career": "experience",
#    "internship": "experience",
#    "education": "education", "qualification": "education", "academic": "education",
#    "project": "projects", "portfolio": "projects",
#    "certif": "certifications", "licen": "certifications", "credential": "certifications",
#    "accreditation": "certifications", "training": "certifications",
#    "achievement": "achievements", "award": "achievements", "honor": "achievements",
#    "honour": "achievements", "recognition": "achievements", "accomplishment": "achievements",
#    "reward": "achievements",
#    "language": "languages",
#    "interest": "interests", "hobbies": "interests",
#    "strength": "strengths",
#    "declaration": "declaration",
#    "personal": "personal_details",
#    "responsibilit": "roles_responsibilities",
#}
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2}))\s*[-–—\u2013\u2014]+\s*"
#    r"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2})))"
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#def _resolve_unknown_heading(stripped: str):
#    """
#    A line inside a list-type section that scored 'unknown' via
#    score_heading_line is ambiguous: it could be a harmless sub-label
#    (e.g. "Programming Languages:" inside a skills list, followed on the
#    same line by the actual list of languages) or a genuine new section
#    whose heading just didn't match any SECTION_PATTERNS regex (e.g.
#    "Certifications (Technical + Management)" landing inside "experience",
#    "KEY ACHIEVEMENTS" landing inside "strengths" -- both confirmed on
#    real resumes). Both look identical to the caller: label == "unknown".
#
#    We re-derive a real label here from the same keyword vocabulary that
#    caused score_heading_line to score this line highly in the first
#    place (_SECTION_KEYWORDS / _contains_section_keyword), instead of
#    silently absorbing it either way.
#
#    Guard: only promote when the line looks like a BARE heading. If there
#    is real content after a trailing colon (e.g. "Certifications: AWS,
#    Azure" as one bullet-style line among others in a skills list), this
#    is a sub-label with inline content, not a new section -- so we leave
#    it alone and let it be absorbed as before.
#    """
#    colon_idx = stripped.find(":")
#    if colon_idx != -1 and stripped[colon_idx + 1:].strip():
#        return None
#
#    lower = stripped.lower()
#    for keyword, mapped_label in _KEYWORD_TO_LABEL.items():
#        if keyword in lower:
#            return mapped_label
#    return None
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        # A heading can legitimately match on its own but still be the FIRST
#        # half of a line-wrapped heading (e.g. "Achievements" / "and
#        # Appreciations:"). Before committing, check whether folding in the
#        # very next line also produces a valid match -- if so, treat the
#        # next line as part of the heading, not as body text. Only extends
#        # when the next line is short, has no bullet marker, and isn't
#        # itself a date-range line, so this can't misfire on a heading
#        # immediately followed by real bullet/date content.
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        # This must run BEFORE the same-label-absorb check directly below:
#        # a second "summary"-labelled heading (e.g. "Professional Summary"
#        # appearing after an earlier "Personal Summary") sometimes isn't
#        # really a second summary at all -- it's actually introducing the
#        # first job entry (confirmed on a real resume, Aarti Suranje: her
#        # "PROFESSIONAL SUMMARY" heading sits directly above "PRINCIPAL
#        # CONSULTANT, 01/2023 - Current"). If this promotion ran AFTER the
#        # same-label-absorb check instead, the absorb-and-continue would
#        # fire first (both headings map to "summary") and this promotion
#        # would never get a chance to run -- her entire job history
#        # (Principal Consultant, Technology Lead, ...) would silently
#        # merge into "summary" instead of becoming its own "experience"
#        # section.
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            # Don't blindly absorb: this line scored heading-shaped and
#            # contains a section keyword -- check whether it actually
#            # resolves to a real, different section label before treating
#            # it as harmless in-list noise. See _resolve_unknown_heading().
#            promoted_label = _resolve_unknown_heading(stripped)
#            if promoted_label is None:
#                current_lines.append(line)
#                continue
#            label = promoted_label
#            confidence = 0.8
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#











##just changing and writting on above 
#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 15)
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and|/)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies))(?:\s+sets?)?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"((work|industry|relevant|professional|previous)\s*(and\s*)?){0,2}experience",
#        r"employment(\s+(history|details|records?|background))?", r"professional\s*background",
#        r"career\s*history", r"work\s*history", r"internships?",
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward",
#]
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2}))\s*[-–—\u2013\u2014]+\s*"
#    r"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2})))"
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        # A heading can legitimately match on its own but still be the FIRST
#        # half of a line-wrapped heading (e.g. "Achievements" / "and
#        # Appreciations:"). Before committing, check whether folding in the
#        # very next line also produces a valid match -- if so, treat the
#        # next line as part of the heading, not as body text. Only extends
#        # when the next line is short, has no bullet marker, and isn't
#        # itself a date-range line, so this can't misfire on a heading
#        # immediately followed by real bullet/date content.
#        if label is not None and stripped != "" and i + 1 < len(lines):
#            nxt = lines[i + 1].strip()
#            if (nxt and not _LEADING_BULLET_RE.match(nxt)
#                    and len(nxt.split()) <= 5 and not _JOB_DATE_RANGE.search(nxt)):
#                combined = _clean_heading_candidate(f"{stripped} {nxt}")
#                for combined_label, pattern in SECTION_PATTERNS.items():
#                    if pattern.match(combined):
#                        label = combined_label
#                        confidence = max(confidence, 0.9)
#                        skip_next = True
#                        break
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        # This must run BEFORE the same-label-absorb check directly below:
#        # a second "summary"-labelled heading (e.g. "Professional Summary"
#        # appearing after an earlier "Personal Summary") sometimes isn't
#        # really a second summary at all -- it's actually introducing the
#        # first job entry (confirmed on a real resume, Aarti Suranje: her
#        # "PROFESSIONAL SUMMARY" heading sits directly above "PRINCIPAL
#        # CONSULTANT, 01/2023 - Current"). If this promotion ran AFTER the
#        # same-label-absorb check instead, the absorb-and-continue would
#        # fire first (both headings map to "summary") and this promotion
#        # would never get a chance to run -- her entire job history
#        # (Principal Consultant, Technology Lead, ...) would silently
#        # merge into "summary" instead of becoming its own "experience"
#        # section.
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            current_lines.append(line)
#            continue
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#








#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 14)
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies))(?:\s+sets?)?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization)$",
#    re.IGNORECASE
#)
#
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"((work|industry|relevant|professional|previous)\s*(and\s*)?){0,2}experiences?:?",
#        r"employment(\s+(history|details|records?|background))?", r"professional\s*background",
#        r"career\s*history", r"work\s*history", r"internships?",
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward",
#]
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2}))\s*[-–—\u2013\u2014]+\s*"
#    r"(?:Current|Present|Now|Till\s*(?:Date|Now)|current|present|now|(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2})))"
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        # This must run BEFORE the same-label-absorb check directly below:
#        # a second "summary"-labelled heading (e.g. "Professional Summary"
#        # appearing after an earlier "Personal Summary") sometimes isn't
#        # really a second summary at all -- it's actually introducing the
#        # first job entry (confirmed on a real resume, Aarti Suranje: her
#        # "PROFESSIONAL SUMMARY" heading sits directly above "PRINCIPAL
#        # CONSULTANT, 01/2023 - Current"). If this promotion ran AFTER the
#        # same-label-absorb check instead, the absorb-and-continue would
#        # fire first (both headings map to "summary") and this promotion
#        # would never get a chance to run -- her entire job history
#        # (Principal Consultant, Technology Lead, ...) would silently
#        # merge into "summary" instead of becoming its own "experience"
#        # section.
#        if label == "summary" and not experience_seen and stripped != "":
#            upcoming = _next_nonblank_line(lines, i + 1)
#            if _looks_like_job_entry(upcoming):
#                label = "experience"
#                confidence = 0.85
#
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            current_lines.append(line)
#            continue
#
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#
#











#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 14)
#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Fuzzy typo match on single-word headings
#  Strategy C: Structural signal — short all-caps line directly followed by
#              a job-date-range line (handles novel/unseen "Experience"
#              synonyms like "CORPORATE SUCCESS" that no keyword list will
#              ever fully enumerate)
#  Strategy D: Line feature scoring (handles remaining ambiguous headings)
#
#REVISION 14 changes (this revision):
#
#  FIX 25 -- "skills" heading prefix was a fixed enumerated list
#  (technical|core|key|professional), so any other qualifier word in front
#  of "Skills"/"Competencies" fell through to "unknown". Confirmed on two
#  real resumes: Abhay Awasthi's "IT SKILLS" and Abhishek Saraniya's
#  "TECHNICAL SKILL SET" both missed entirely (no "skills" label appeared
#  in either resume's section list at all). Rather than keep enumerating
#  prefixes one at a time forever (soft skills, domain skills, functional
#  skills, ...), the skills pattern is now prefix-AGNOSTIC: it matches any
#  short heading (0-2 leading qualifier words) ending in "skill(s)" or
#  "competenc(y|ies)", optionally followed by "set". This also means: if a
#  resume has BOTH a "Core Competencies" heading and a separate "Technical
#  Skills" heading, both correctly resolve to the same "skills" label and
#  their content merges into one section (existing absorption logic in
#  split_into_sections already does this once both map to the same label --
#  this revision's fix is what makes the *coverage* wide enough for that
#  merging to actually trigger across more real-world heading variants).
#
#  FIX 26 -- inline sub-category labels ("Languages: Python, Java...",
#  "Databases & Tools: MongoDB...") inside an active list-type section
#  (skills, education, etc.) were being mistaken for genuine new section
#  headings by split_inline_heading(), because the text before the colon
#  ("Languages") happens to also be a real section-heading keyword.
#  Confirmed on a real resume (Abhishek Chaudhary): his "Technical Skills"
#  section lists categories like "Languages: Python, Java, C/C++,
#  JavaScript" and "Frameworks & MLOps: PyTorch, ...". The very first such
#  line ("Languages: ...") was detected as a brand-new "languages" section,
#  which silently closed the (now-empty) "Technical Skills" section and
#  swallowed the ENTIRE rest of the skills block -- LLMs & GenAI, ML/NLP,
#  Computer Vision, Frameworks, Databases -- into a section mislabelled
#  "languages". This is a general failure class, not specific to this
#  resume: ANY "Category: items" sub-line inside a list section risks
#  colliding with a real section keyword (Languages, Certifications, and
#  Projects are all common category labels *and* real section names).
#
#  THE FIX: a genuine top-level section heading is virtually always on its
#  own line; an inline "Label: content" line with substantial content
#  packed onto the SAME line is characteristic of a sub-category tag
#  within an already-open list, not a real section transition. So:
#  split_inline_heading()'s result is now only allowed to trigger a
#  section switch when we are NOT already inside a _LIST_SECTION_LABELS
#  section. While inside one, an inline "Label:" line is left for the
#  normal fuzzy/Strategy C/Strategy D fallbacks to evaluate on its own
#  merits (which correctly reject it -- FIX 22's "Label: Value" colon
#  penalty already scores these below the 0.65 threshold), so it's simply
#  appended as ordinary content of the CURRENT section instead of starting
#  a new one. Genuine section transitions (a heading on its own line) are
#  completely unaffected, since those are matched by Strategy A's full-line
#  detect_section_label() before split_inline_heading() is ever consulted.
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
## FIX 25: dedicated (non-_build_section_pattern-based) compiled pattern for
## "skills" -- prefix-AGNOSTIC-ISH, see revision docstring above. Uses a
## curated (not fully open-ended) prefix word list rather than a bare
## `[a-z]+`, because a fully unconstrained leading-word slot was confirmed
## by testing to create a real collision: "LANGUAGE SKILLS" fit the shape
## "<word> skills" just as well as "IT SKILLS" does, so it got claimed by
## this pattern instead of the dedicated "languages" pattern below (which
## already handles "Language Skills" correctly on its own). The prefix
## list here is deliberately broader than the original 4-word list
## (technical/core/key/professional) to close the real gaps found on real
## resumes (IT Skills, Domain Skills, Soft Skills, Functional Skills) --
## but "language(s)" is deliberately excluded since that's its own
## distinct, already-handled category.
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^(?:({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies))(?:\s+sets?)?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?"
#    # FIX 30: real resume (Ajit A Tupale) heads its skills section
#    # "Function and Technical Specialization" -- an ERP/consulting-
#    # domain convention (functional expertise + technical expertise)
#    # that contains neither "skill" nor "competenc" anywhere, so no
#    # amount of widening the prefix list around those two keywords
#    # would ever match it. Added as an explicit literal alternative,
#    # same treatment as "corporate success" for experience earlier.
#    rf"|function(al)?\s+(and\s+)?technical\s+specialization"
#    rf"|technical\s+specialization"
#    rf"|functional\s+specialization)$",
#    re.IGNORECASE
#)
#
## FIX 33: real resume (Ajit Kumar) numbers its project entries as their
## own sub-headings -- "Project #1:-", "Project #2:-", etc. The generic
## "projects?" core (via _build_section_pattern) can't match these: the
## trailing "#1" has nothing in the pattern to account for it, so the
## anchored full-string match fails and the heading falls through to
## Strategy D's generic score_heading_line(). There it scores as generic
## "unknown" rather than the specific "projects" label -- and because
## "unknown"-scored headings get absorbed into whatever list-type section
## is currently open (by design, for spurious sub-bullet mis-detections),
## "Project #1:-" was silently swallowed into the preceding "education"
## section instead of starting a real "projects" section. Giving it the
## SPECIFIC "projects" label here avoids that absorption trap entirely,
## since the absorption rule only catches generic "unknown", not a
## concretely-resolved label that differs from the currently open one.
#_PROJECTS_PATTERN = re.compile(
#    r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?"
#    r"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){0,2})?"
#    r"|portfolio"
#    r"|projects?\s*#?\s*\d+)$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"((work|industry|relevant|professional|previous)\s*(and\s*)?){0,2}experience",
#        r"employment(\s+(history|details|records?|background))?", r"professional\s*background",
#        r"career\s*history", r"work\s*history", r"internships?",
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    # FIX 25: "skills" is now handled by the dedicated _SKILLS_PATTERN
#    # above (prefix-agnostic), not via _build_section_pattern. It's still
#    # registered in this dict under the "skills" key so all the existing
#    # lookup code (detect_section_label, split_inline_heading) works
#    # unchanged.
#    "skills": _SKILLS_PATTERN,
#    "projects": _PROJECTS_PATTERN,
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward",
#]
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    # Strip a leading bullet/checkmark marker before the colon/dash strip
#    # below. Needed once PUA glyph fallbacks are normalized to "•" (see
#    # ingestion/__init__.py's _normalize_pua_glyphs) -- a heading line
#    # that arrives as "• Objective:" would otherwise never match any
#    # SECTION_PATTERNS entry, since those are anchored to start with the
#    # keyword itself. Also covers resumes that natively use checkmark/
#    # arrow bullets in front of headings (confirmed on a real resume,
#    # Abhishek Saraniya, which uses "✓" as its bullet marker throughout).
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2})\s*[-\u2013\u2014]+\s*"
#    r"(?:Current|Present|current|present|(?:\d{1,2}/)?(?:\d{4}|['’‘`´]?\d{2}))"
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
## FIX 28: real resume (Ajit A Tupale) has its skills heading rendered as
## "Function and Technical" on one physical line and "Specialization" on
## the next -- a narrow sidebar column wrapped a heading that would have
## matched _SKILLS_PATTERN cleanly on a single line. Neither half alone
## matches anything, so the whole skills block silently fell into
## "header". This is a distinct failure class from the prefix/keyword
## fixes above: no amount of widening vocabulary helps when the heading
## TEXT ITSELF has been split across two lines by PDF layout.
##
## Only tried when the current line alone doesn't already match (see call
## site), and only succeeds when the two lines concatenated together
## EXACTLY match one of SECTION_PATTERNS' anchored (^...$) regexes -- so
## this can't misfire on two unrelated adjacent body lines; ordinary
## prose joined together essentially never happens to equal one of our
## curated heading phrases. The second line is also required not to
## start with a bullet marker, since a genuine wrapped-heading
## continuation is never itself a bullet item (belt-and-suspenders on
## top of the anchored full-string match).
#_LEADING_BULLET_RE = re.compile(r"^[•●○◦▪➤►‣✓✔☑\-\*]")
#
#
#def _try_two_line_heading(lines: list, i: int):
#    if i + 1 >= len(lines):
#        return None, 0.0
#    first = lines[i].strip()
#    second = lines[i + 1].strip()
#    if not first or not second:
#        return None, 0.0
#    if _LEADING_BULLET_RE.match(second):
#        return None, 0.0
#    if _is_wrapped_word(first, lines[:i]):
#        return None, 0.0
#
#    joined = _clean_heading_candidate(f"{first} {second}")
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(joined):
#            return label, 0.9
#    return None, 0.0
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    # FIX 31: narrowed from r':\s*\S' to r':\s*[A-Za-z0-9]' -- confirmed
#    # on a real resume (Ajit Kumar) that a project heading rendered as
#    # "Project #1:-" (a decorative trailing dash after the colon, not
#    # real content) was wrongly caught by the old \S check, which
#    # matches ANY non-whitespace character including punctuation. That
#    # tanked its score below the 0.65 threshold and left it silently
#    # absorbed into the preceding "education" section instead of
#    # starting a new "projects" section. Requiring an actual alphanumeric
#    # character after the colon still catches every real "Label: Value"
#    # case from FIX 22 (the value is always real text/data) while no
#    # longer misfiring on a heading whose only "value" is punctuation.
#    if re.search(r':\s*[A-Za-z0-9]', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#    skip_next = False
#
#    for i, line in enumerate(lines):
#        if skip_next:
#            skip_next = False
#            continue
#
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#
#        # FIX 28: try the two-line wrapped-heading join right after
#        # single-line Strategy A fails, before the inline-heading (colon)
#        # check -- it's the same precedence tier as Strategy A, since
#        # it's also an exact anchored-regex match, just spanning two
#        # physical lines instead of one.
#        if label is None and stripped != "":
#            two_line_label, two_line_confidence = _try_two_line_heading(lines, i)
#            if two_line_label is not None:
#                label = two_line_label
#                confidence = two_line_confidence
#                skip_next = True
#
#        if label is None and stripped != "":
#            # FIX 26: an inline "Label: content" heading is only trusted
#            # to trigger a section SWITCH when we are not already inside
#            # a list-type section. See revision docstring above -- inside
#            # an active list section, a colon-separated sub-category line
#            # ("Languages: Python, Java...") is far more likely to be a
#            # category tag within that list than a genuine new section.
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        # FIX 32: same-label headings always merge into the currently
#        # open section, regardless of section type -- not just list-type
#        # sections. Two genuinely different heading TEXTS that both
#        # resolve to the same canonical label (e.g. "Objective:" and
#        # "Professional Summary:", both -> "summary") should produce ONE
#        # merged section, not two separate same-labelled Section objects.
#        # The "unknown" absorption below (spurious Strategy-D heading
#        # guesses inside an open list, e.g. a stray sub-bullet scored as
#        # a heading) stays restricted to list-type sections only --
#        # that's a different failure mode and unchanged from before.
#        if label is not None and label == current_label:
#            current_lines.append(line)
#            continue
#        if current_label in _LIST_SECTION_LABELS and label == "unknown":
#            current_lines.append(line)
#            continue
#
#        # Prevent sub-headings under jobs (like "Achievements:", "Key Projects:", "Roles and Responsibilities")
#        # from splitting the active experience section.
#        if current_label == "experience" and label in {"achievements", "projects", "roles_responsibilities"}:
#            if label in {"achievements", "roles_responsibilities"} or stripped.endswith(":") or stripped.endswith(":-"):
#                current_lines.append(line)
#                continue
#
#        if label is not None and stripped != "":
#            if label == "summary" and not experience_seen:
#                upcoming = _next_nonblank_line(lines, i + 1)
#                if _looks_like_job_entry(upcoming):
#                    label = "experience"
#                    confidence = 0.85
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#
#














#Worked - just changing and pasting just above
#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 14)
#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Fuzzy typo match on single-word headings
#  Strategy C: Structural signal — short all-caps line directly followed by
#              a job-date-range line (handles novel/unseen "Experience"
#              synonyms like "CORPORATE SUCCESS" that no keyword list will
#              ever fully enumerate)
#  Strategy D: Line feature scoring (handles remaining ambiguous headings)
#
#REVISION 14 changes (this revision):
#
#  FIX 25 -- "skills" heading prefix was a fixed enumerated list
#  (technical|core|key|professional), so any other qualifier word in front
#  of "Skills"/"Competencies" fell through to "unknown". Confirmed on two
#  real resumes: Abhay Awasthi's "IT SKILLS" and Abhishek Saraniya's
#  "TECHNICAL SKILL SET" both missed entirely (no "skills" label appeared
#  in either resume's section list at all). Rather than keep enumerating
#  prefixes one at a time forever (soft skills, domain skills, functional
#  skills, ...), the skills pattern is now prefix-AGNOSTIC: it matches any
#  short heading (0-2 leading qualifier words) ending in "skill(s)" or
#  "competenc(y|ies)", optionally followed by "set". This also means: if a
#  resume has BOTH a "Core Competencies" heading and a separate "Technical
#  Skills" heading, both correctly resolve to the same "skills" label and
#  their content merges into one section (existing absorption logic in
#  split_into_sections already does this once both map to the same label --
#  this revision's fix is what makes the *coverage* wide enough for that
#  merging to actually trigger across more real-world heading variants).
#
#  FIX 26 -- inline sub-category labels ("Languages: Python, Java...",
#  "Databases & Tools: MongoDB...") inside an active list-type section
#  (skills, education, etc.) were being mistaken for genuine new section
#  headings by split_inline_heading(), because the text before the colon
#  ("Languages") happens to also be a real section-heading keyword.
#  Confirmed on a real resume (Abhishek Chaudhary): his "Technical Skills"
#  section lists categories like "Languages: Python, Java, C/C++,
#  JavaScript" and "Frameworks & MLOps: PyTorch, ...". The very first such
#  line ("Languages: ...") was detected as a brand-new "languages" section,
#  which silently closed the (now-empty) "Technical Skills" section and
#  swallowed the ENTIRE rest of the skills block -- LLMs & GenAI, ML/NLP,
#  Computer Vision, Frameworks, Databases -- into a section mislabelled
#  "languages". This is a general failure class, not specific to this
#  resume: ANY "Category: items" sub-line inside a list section risks
#  colliding with a real section keyword (Languages, Certifications, and
#  Projects are all common category labels *and* real section names).
#
#  THE FIX: a genuine top-level section heading is virtually always on its
#  own line; an inline "Label: content" line with substantial content
#  packed onto the SAME line is characteristic of a sub-category tag
#  within an already-open list, not a real section transition. So:
#  split_inline_heading()'s result is now only allowed to trigger a
#  section switch when we are NOT already inside a _LIST_SECTION_LABELS
#  section. While inside one, an inline "Label:" line is left for the
#  normal fuzzy/Strategy C/Strategy D fallbacks to evaluate on its own
#  merits (which correctly reject it -- FIX 22's "Label: Value" colon
#  penalty already scores these below the 0.65 threshold), so it's simply
#  appended as ordinary content of the CURRENT section instead of starting
#  a new one. Genuine section transitions (a heading on its own line) are
#  completely unaffected, since those are matched by Strategy A's full-line
#  detect_section_label() before split_inline_heading() is ever consulted.
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
## FIX 25: dedicated (non-_build_section_pattern-based) compiled pattern for
## "skills" -- prefix-AGNOSTIC-ISH, see revision docstring above. Uses a
## curated (not fully open-ended) prefix word list rather than a bare
## `[a-z]+`, because a fully unconstrained leading-word slot was confirmed
## by testing to create a real collision: "LANGUAGE SKILLS" fit the shape
## "<word> skills" just as well as "IT SKILLS" does, so it got claimed by
## this pattern instead of the dedicated "languages" pattern below (which
## already handles "Language Skills" correctly on its own). The prefix
## list here is deliberately broader than the original 4-word list
## (technical/core/key/professional) to close the real gaps found on real
## resumes (IT Skills, Domain Skills, Soft Skills, Functional Skills) --
## but "language(s)" is deliberately excluded since that's its own
## distinct, already-handled category.
#_SKILL_PREFIX_WORDS = (
#    r"(technical|core|key|professional|functional|domain|business|"
#    r"soft|hard|tech|it|general|primary|specialized|relevant)"
#)
#_SKILLS_PATTERN = re.compile(
#    rf"^({_SKILL_PREFIX_WORDS}\s+){{0,2}}(skills?|competenc(y|ies))(\s+set)?"
#    rf"(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#    re.IGNORECASE
#)
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+|profile\s+|executive\s+|career\s+|brief\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"(work\s+|industry\s+|relevant\s+|professional\s+|previous\s+)?experience",
#        r"employment(\s+history)?", r"professional\s+background",
#        r"career\s+history", r"work\s+history", r"internships?",
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    # FIX 25: "skills" is now handled by the dedicated _SKILLS_PATTERN
#    # above (prefix-agnostic), not via _build_section_pattern. It's still
#    # registered in this dict under the "skills" key so all the existing
#    # lookup code (detect_section_label, split_inline_heading) works
#    # unchanged.
#    "skills": _SKILLS_PATTERN,
#    "projects": _build_section_pattern([
#        r"(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?",
#        r"portfolio",
#    ]),
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    "corporate", "success", "journey", "chronology", "timeline",
#    "reward",
#]
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    # Strip a leading bullet/checkmark marker before the colon/dash strip
#    # below. Needed once PUA glyph fallbacks are normalized to "•" (see
#    # ingestion/__init__.py's _normalize_pua_glyphs) -- a heading line
#    # that arrives as "• Objective:" would otherwise never match any
#    # SECTION_PATTERNS entry, since those are anchored to start with the
#    # keyword itself. Also covers resumes that natively use checkmark/
#    # arrow bullets in front of headings (confirmed on a real resume,
#    # Abhishek Saraniya, which uses "✓" as its bullet marker throughout).
#    cleaned = re.sub(r"^[•●○◦▪➤►‣✓✔☑\-\*]+\s*", "", cleaned)
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{1,2}/)?\d{4}\s*[-\u2013\u2014]+\s*"
#    r"(?:Current|Present|current|present|(?:\d{1,2}/)?\d{4})"
#)
#
#
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    if re.search(r':\s*\S', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#
#    for i, line in enumerate(lines):
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#        if label is None and stripped != "":
#            # FIX 26: an inline "Label: content" heading is only trusted
#            # to trigger a section SWITCH when we are not already inside
#            # a list-type section. See revision docstring above -- inside
#            # an active list section, a colon-separated sub-category line
#            # ("Languages: Python, Java...") is far more likely to be a
#            # category tag within that list than a genuine new section.
#            if current_label not in _LIST_SECTION_LABELS:
#                heading_candidate, remaining = split_inline_heading(stripped)
#                if heading_candidate is not None:
#                    label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                    if label is not None:
#                        inline_content = remaining
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        if current_label in _LIST_SECTION_LABELS and label in (current_label, "unknown"):
#            current_lines.append(line)
#            continue
#
#        if label is not None and stripped != "":
#            if label == "summary" and not experience_seen:
#                upcoming = _next_nonblank_line(lines, i + 1)
#                if _looks_like_job_entry(upcoming):
#                    label = "experience"
#                    confidence = 0.85
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#
#












#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 13)
#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Fuzzy typo match on single-word headings
#  Strategy C: Structural signal — short all-caps line directly followed by
#              a job-date-range line (handles novel/unseen "Experience"
#              synonyms like "CORPORATE SUCCESS" that no keyword list will
#              ever fully enumerate)
#  Strategy D: Line feature scoring (handles remaining ambiguous headings)
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"(work\s+|industry\s+|relevant\s+|professional\s+|previous\s+)?experience",
#        r"employment(\s+history)?", r"professional\s+background",
#        r"career\s+history", r"work\s+history", r"internships?",
#        # FIX 23: real resume (Abhishek Kumar) uses "CORPORATE SUCCESS" as
#        # its experience heading. No keyword list can ever fully enumerate
#        # every synonym companies/candidates invent for "Experience", so
#        # this is a known-synonym patch on top of the structural Strategy C
#        # fallback below, which catches novel synonyms this list misses.
#        r"corporate\s+success", r"career\s+journey", r"professional\s+journey",
#        r"career\s+chronology", r"employment\s+timeline",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?",
#        # FIX 24: real resume (Abhishek Kumar) heads this section
#        # "ACADEMICS & CREDENTIALS" -- plural "academics" wasn't accepted
#        # before (only singular "academic"), so the whole heading fell
#        # through to the generic Strategy D scorer instead of matching here.
#        r"academics?(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _build_section_pattern([
#        r"(technical\s+|core\s+|key\s+|professional\s+)?(skills?|competenc(y|ies))",
#        r"expertise", r"technologies",
#        r"programming\s+(languages?|skills?)",
#        r"tools?\s*(&|and)\s*technologies",
#        r"core\s+competencies\s*(&|and)\s*technical\s+skills",
#    ]),
#    "projects": _build_section_pattern([
#        # FIX 20: real resume (Abhay Awasthi) has "LIVE PROJECTS" as its
#        # standalone projects heading; "live" wasn't an accepted prefix,
#        # so it fell through to Strategy B as generic "unknown" while
#        # still inside "experience" -- and since "projects" is in
#        # _LIST_SECTION_LABELS, that "unknown" got silently absorbed as
#        # more experience content instead of starting a new section.
#        r"(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?",
#        r"portfolio",
#    ]),
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#        # FIX 24: real resume (Abhishek Kumar) heads this section
#        # "TRAININGS & CERTIFICATIONS" -- the core alternative must match
#        # the FIRST word of the heading (regex is anchored ^...$), and
#        # "trainings" alone wasn't accepted before, even though
#        # "certifications" appears later in the same line.
#        r"trainings?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#        # FIX 24: real resume (Abhishek Kumar) heads this section
#        # "REWARDS & RECOGNITIONS" -- "recognitions" was already covered,
#        # but the core alternative must match the FIRST word of the
#        # heading (regex is anchored), and "rewards" alone wasn't accepted.
#        r"rewards?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    # FIX 21: real resume (Abhay Awasthi) has a "Roles & Responsibilities"
#    # heading that PDF extraction reordered as "ROLES RESPONSIBILITIES &"
#    # (the "&" pushed to the end). Not built via _build_section_pattern
#    # since that helper assumes the qualifier trails the core phrase --
#    # here the "&"/"and" can legitimately sit in either position, so this
#    # is a dedicated pattern that accepts both orderings directly.
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#    # FIX 23: widen the keyword net so Strategy D's keyword-penalty logic
#    # doesn't nuke the score of common "Experience" synonyms that lack a
#    # blank-line/date-range signal to fall back on (e.g. heading is
#    # followed by a one-line intro sentence before the first job entry).
#    "corporate", "success", "journey", "chronology", "timeline",
#    # FIX 24: same idea for "Rewards & Recognitions".
#    "reward",
#]
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{1,2}/)?\d{4}\s*[-\u2013\u2014]+\s*"
#    r"(?:Current|Present|current|present|(?:\d{1,2}/)?\d{4})"
#)
#
#
## FIX 19: catches single-word heading TYPOS that Strategy A's exact regex
## can never match and that Strategy B's shape-scoring also misses whenever
## the typo happens to break the keyword substring check (confirmed on a
## real resume, Aastha Gupta: "EXPERINCE" for "EXPERIENCE" scored only 0.3
## via Strategy D -- well under the 0.65 threshold -- because
## _contains_section_keyword requires the literal substring "experience",
## which a missing letter breaks. Her entire experience section was
## silently absorbed into "summary" as a result).
##
## Deliberately narrow, so it can't misfire on ordinary short content
## words: only tried as a last-resort fallback after Strategy A has
## already failed, and only matches a candidate that is a SINGLE
## alphabetic word within edit-distance ~1 of a known heading keyword
## (via difflib, stdlib -- no new dependency).
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
## FIX 23: structural signal for Strategy C. A short, all-caps line
## directly followed by a line that looks like a job-date-range entry
## (company + "2021 - 2023" / "Aug'22 - Sep'23" / "2024 - Present" etc.)
## is almost certainly a new "experience" heading -- regardless of whether
## its text matches any known keyword. This is far more stable at scale
## than any keyword list, since candidates/companies invent endless
## synonyms for "Experience" (Corporate Success, Career Journey, Career
## Chronology, Professional Milestones, ...) but the structure that
## follows a job-history heading (dated entries) is much more consistent.
#_CAPS_HEADING_SHAPE = re.compile(r"^[A-Z\s&/]+$")
#
#
#def _looks_like_caps_heading_shape(stripped: str) -> bool:
#    return bool(
#        stripped == stripped.upper()
#        and _CAPS_HEADING_SHAPE.match(stripped)
#        and len(stripped.split()) <= 4
#        and not stripped.endswith(".")
#    )
#
#
#def _next_line_is_job_entry(line_index: int, all_lines: list) -> bool:
#    upcoming = _next_nonblank_line(all_lines, line_index + 1)
#    return _looks_like_job_entry(upcoming)
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    # FIX 22: a "Label: Value" line (colon followed by real content, e.g.
#    # "Driving Licence: YES", "Gender: MALE", "Date of Birth: ...") is
#    # almost always a personal-detail data line, not a section heading --
#    # even when it happens to contain a heading-shaped keyword substring.
#    # Confirmed on a real resume (Satyadip Ray): "Driving Licence: YES"
#    # scored exactly 0.65 purely from the "licen" keyword bonus (meant
#    # for "Licenses"/"Certifications" headings) and incorrectly split his
#    # PERSONAL DETAILS section into two pieces. A genuine heading that
#    # ends WITH a bare colon (e.g. "SKILLS:") is unaffected -- this only
#    # penalizes colons that are followed by actual content on the line.
#    if re.search(r':\s*\S', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#
#    for i, line in enumerate(lines):
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#        if label is None and stripped != "":
#            heading_candidate, remaining = split_inline_heading(stripped)
#            if heading_candidate is not None:
#                label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                if label is not None:
#                    inline_content = remaining
#
#        # FIX 19: try fuzzy typo match before falling back to Strategy D's
#        # generic "unknown" scoring -- a specific label beats a generic one
#        # whenever both could in principle apply.
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        # FIX 23 (Strategy C): short all-caps line directly followed by a
#        # job-date-range line -> treat as a new "experience" heading even
#        # with zero keyword overlap and no blank-line spacing. This is
#        # tried BEFORE Strategy D's generic scoring because it's a more
#        # specific/reliable signal than the generic heading-shape score,
#        # and it's what correctly reclassifies novel headings like
#        # "CORPORATE SUCCESS" that no keyword list enumerates.
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                if _looks_like_caps_heading_shape(stripped) and _next_line_is_job_entry(i, lines):
#                    label = "experience"
#                    confidence = 0.85
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        if current_label in _LIST_SECTION_LABELS and label in (current_label, "unknown"):
#            current_lines.append(line)
#            continue
#
#        if label is not None and stripped != "":
#            if label == "summary" and not experience_seen:
#                upcoming = _next_nonblank_line(lines, i + 1)
#                if _looks_like_job_entry(upcoming):
#                    label = "experience"
#                    confidence = 0.85
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#











##worked - just commenting to work one resume
#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 11)
#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Line feature scoring (handles ambiguous/unrecognised headings)
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"(work\s+|industry\s+|relevant\s+|professional\s+|previous\s+)?experience",
#        r"employment(\s+history)?", r"professional\s+background",
#        r"career\s+history", r"work\s+history", r"internships?",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?", r"academic(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _build_section_pattern([
#        r"(technical\s+|core\s+|key\s+|professional\s+)?(skills?|competenc(y|ies))",
#        r"expertise", r"technologies",
#        r"programming\s+(languages?|skills?)",
#        r"tools?\s*(&|and)\s*technologies",
#        r"core\s+competencies\s*(&|and)\s*technical\s+skills",
#    ]),
#    "projects": _build_section_pattern([
#        # FIX 20: real resume (Abhay Awasthi) has "LIVE PROJECTS" as its
#        # standalone projects heading; "live" wasn't an accepted prefix,
#        # so it fell through to Strategy B as generic "unknown" while
#        # still inside "experience" -- and since "projects" is in
#        # _LIST_SECTION_LABELS, that "unknown" got silently absorbed as
#        # more experience content instead of starting a new section.
#        r"(personal\s+|side\s+|key\s+|notable\s+|academic\s+|live\s+)?projects?",
#        r"portfolio",
#    ]),
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#    # FIX 21: real resume (Abhay Awasthi) has a "Roles & Responsibilities"
#    # heading that PDF extraction reordered as "ROLES RESPONSIBILITIES &"
#    # (the "&" pushed to the end). Not built via _build_section_pattern
#    # since that helper assumes the qualifier trails the core phrase --
#    # here the "&"/"and" can legitimately sit in either position, so this
#    # is a dedicated pattern that accepts both orderings directly.
#    "roles_responsibilities": re.compile(
#        r"^roles?\s+(and|&)?\s*responsibilit(y|ies)\s*(and|&)?$",
#        re.IGNORECASE
#    ),
#}
#
#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration", "responsibilit",
#]
#
#
#@dataclass
#class Section:
#    label: str
#    raw_text: str
#    start_line: int
#    confidence: float
#
#
#def _clean_heading_candidate(line: str) -> str:
#    cleaned = line.strip()
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    stripped = line.strip()
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{1,2}/)?\d{4}\s*[-\u2013\u2014]+\s*"
#    r"(?:Current|Present|current|present|(?:\d{1,2}/)?\d{4})"
#)
#
#
## FIX 19: catches single-word heading TYPOS that Strategy A's exact regex
## can never match and that Strategy B's shape-scoring also misses whenever
## the typo happens to break the keyword substring check (confirmed on a
## real resume, Aastha Gupta: "EXPERINCE" for "EXPERIENCE" scored only 0.3
## via Strategy B -- well under the 0.65 threshold -- because
## _contains_section_keyword requires the literal substring "experience",
## which a missing letter breaks. Her entire experience section was
## silently absorbed into "summary" as a result).
##
## Deliberately narrow, so it can't misfire on ordinary short content
## words: only tried as a last-resort fallback after BOTH Strategy A and
## Strategy B have already failed, and only matches a candidate that is a
## SINGLE alphabetic word within edit-distance ~1 of a known heading
## keyword (via difflib, stdlib -- no new dependency).
#_FUZZY_HEADING_KEYWORDS = {
#    "experience": "experience",
#    "summary": "summary",
#    "objective": "summary",
#    "profile": "summary",
#    "education": "education",
#    "skills": "skills",
#    "projects": "projects",
#    "certifications": "certifications",
#    "achievements": "achievements",
#    "languages": "languages",
#    "interests": "interests",
#    "declaration": "declaration",
#    "strengths": "strengths",
#}
#
#import difflib
#
#
#def _fuzzy_heading_label(candidate: str):
#    word = candidate.strip().lower()
#    if not word.isalpha() or len(word) < 5:
#        return None
#    for keyword, label in _FUZZY_HEADING_KEYWORDS.items():
#        if abs(len(word) - len(keyword)) > 2:
#            continue
#        if difflib.get_close_matches(word, [keyword], n=1, cutoff=0.82):
#            return label
#    return None
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    if prev_lines is None:
#        prev_lines = []
#    candidate = _clean_heading_candidate(line)
#    if not candidate:
#        return None, 0.0
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#    if line_index <= 1:
#        return 0.0
#    word_count = len(stripped.split())
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#    if stripped.endswith("."):
#        return 0.0
#    score = 0.0
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#    if stripped.endswith(":"):
#        score += 0.15
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    # FIX 22: a "Label: Value" line (colon followed by real content, e.g.
#    # "Driving Licence: YES", "Gender: MALE", "Date of Birth: ...") is
#    # almost always a personal-detail data line, not a section heading --
#    # even when it happens to contain a heading-shaped keyword substring.
#    # Confirmed on a real resume (Satyadip Ray): "Driving Licence: YES"
#    # scored exactly 0.65 purely from the "licen" keyword bonus (meant
#    # for "Licenses"/"Certifications" headings) and incorrectly split his
#    # PERSONAL DETAILS section into two pieces. A genuine heading that
#    # ends WITH a bare colon (e.g. "SKILLS:") is unaffected -- this only
#    # penalizes colons that are followed by actual content on the line.
#    if re.search(r':\s*\S', stripped) and not stripped.endswith(':'):
#        score *= 0.3
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    lines = text.split("\n")
#    sections = []
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#    experience_seen = False
#    deferred_contact_lines = []
#
#    for i, line in enumerate(lines):
#        stripped = line.strip()
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#        inline_content = None
#        if label is None and stripped != "":
#            heading_candidate, remaining = split_inline_heading(stripped)
#            if heading_candidate is not None:
#                label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                if label is not None:
#                    inline_content = remaining
#
#        # FIX 19: try fuzzy typo match before falling back to Strategy B's
#        # generic "unknown" scoring -- a specific label beats a generic one
#        # whenever both could in principle apply.
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                fuzzy_label = _fuzzy_heading_label(stripped)
#                if fuzzy_label is not None:
#                    label = fuzzy_label
#                    confidence = 0.8
#
#        if label is None and stripped != "":
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        if current_label in _LIST_SECTION_LABELS and label in (current_label, "unknown"):
#            current_lines.append(line)
#            continue
#
#        if label is not None and stripped != "":
#            if label == "summary" and not experience_seen:
#                upcoming = _next_nonblank_line(lines, i + 1)
#                if _looks_like_job_entry(upcoming):
#                    label = "experience"
#                    confidence = 0.85
#            if label == "experience":
#                experience_seen = True
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#
















##worked just chaning for some(2) resume header not seperated
#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 11)
#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Line feature scoring (handles ambiguous/unrecognised headings)
#
#REVISION HISTORY (kept here so we don't repeat fixed mistakes):
#
#v1 bug (fixed): none of the original regex patterns had an end-anchor,
#so they matched the START of a line and ignored everything after it.
#Fixed by anchoring every pattern at both start (^) and end ($).
#
#v2 bug (fixed in revision 2): once patterns were strictly anchored,
#they became too rigid -- compound headings like "CERTIFICATIONS &
#LICENCES" didn't match. Fixed by allowing each core heading phrase to
#optionally take a short "& <1-3 words>" / "and <1-3 words>" qualifier.
#
#v3 fixes (revision 3) — real resume (Aarti Suranje, sidebar layout):
#FIX 1 — "PERSONAL SUMMARY" not recognised; added "personal" as an
#allowed summary prefix alongside "professional".
#FIX 2 — A content word wrapped onto its own line (e.g. "experience"
#alone, continuing a sentence from the line above) was misread as a
#heading. Added _is_wrapped_word() guard: suppress a Strategy A match
#if the matched line is a single all-lowercase word AND the previous
#non-empty line doesn't end with sentence-closing punctuation.
#
#v4 fixes (revision 4) — continued debugging of Aarti Suranje's resume:
#
#FIX 3 (SUPERSEDED by FIX 16, v10 -- see below) — originally: any
#second heading matching "summary" was unconditionally forced to
#"experience". This turned out to be too blunt: a real resume
#(Abhishek Sharma) legitimately has "PROFESSIONAL SUMMARY"-style
#headings appearing more than once for genuinely separate summary-like
#content, and the old rule wrongly merged all of it into "experience".
#Replaced entirely by FIX 16.
#
#FIX 4 — "LANGUAGE SKILLS" not recognised; languages pattern now allows
#an optional "skills"/"known"/"proficiency" suffix.
#FIX 5 — "CUSTOM SECTION" had no matching label at all; added a new
#"other" catch-all label.
#
#v5 fixes (revision 5) — heading merged with content on the same line:
#FIX 6 — "CORE COMPETENCIES • SQL • HDL Data Loader" never matched the
#skills pattern as a whole line, so it stayed glued onto whatever
#section preceded it. We do NOT weaken the anchored regex (that's what
#v1 was built to enforce) -- instead split_inline_heading() runs as a
#FALLBACK after detect_section_label() has already failed on the full
#line: finds a known heading phrase at the start of the line, splits it
#from the remaining content at a separator (bullet/colon/dash), and
#re-runs detect_section_label() on just the heading part. Generic
#across every heading and every common separator.
#FIX 7 — Multiple skills packed onto one bullet line (e.g.
#"• SQL • HDL Data Loader") are split into one bullet per line by
#normalize_inline_bullets(), applied whenever a section's raw_text is
#finalised.
#
#v6 fixes (revision 6) — three missing heading patterns (Aasma resume):
#FIX 8 — "Professional Skills" not recognised; added "professional" as
#an allowed skills prefix.
#FIX 9 — "Strengths" had no pattern at all; added new "strengths" label.
#FIX 10 — "Personal Details" / "Declaration" had no pattern either;
#added new "personal_details" and "declaration" labels.
#
#v7 fixes (revision 7) — two compounding bugs found on a real resume
#(Abhay Tyagi) that together fragmented one TECHNICAL SKILLS block into
#five separate pieces:
#
#FIX 11 — Bullet content lines misread as headings by Strategy B purely
#because they contain a keyword substring. "• Natural Language
#Processing" and "• Large Language Models(LLM)" both scored exactly
#0.65 (just over the 0.65 threshold) because "language" is in
#_SECTION_KEYWORDS. A section heading is never itself a bullet list
#item, so score_heading_line() now returns 0.0 immediately for any line
#that starts with a bullet marker, before any other scoring runs.
#
#FIX 12 (superseded by FIX 13 below) — originally scoped only to
#"skills": a sub-category label like "Core Competencies" sitting
#directly under "TECHNICAL SKILLS" with no blank line between them kept
#starting a brand-new "skills" section instead of continuing the one
#already open.
#
#v8 fixes (this revision) — the same bug as FIX 12, confirmed on the
#SAME resume in a second section type:
#
#FIX 13 — Generalized FIX 12. Abhay's resume also has "Higher Secondary
#Education" as a sub-heading directly under "EDUCATION" (his second
#school entry), with no blank line separating it from his first degree.
#"Higher Secondary Education" doesn't match the "education" pattern as
#a whole line (the pattern only matches "education"/"educational
#background" alone, not an arbitrary phrase containing the word), so it
#falls to Strategy B, scores exactly 0.65 (word count <=4, no comma,
#contains the keyword "education"), and gets tagged "unknown" -- which
#incorrectly opened a second, separate section instead of continuing
#the existing "education" one. Identical failure mode to FIX 12, just
#in a different section type.
#
#Rather than patch this one label at a time as we keep hitting it
#(skills, now education, likely certifications/languages/projects
#next), the fix is generalized: _LIST_SECTION_LABELS names every
#section type where multiple sub-entries with their own short
#sub-heading-like lines are common. Once current_label is one of these,
#any further match of the SAME label or "unknown" is treated as a
#sub-heading/sub-entry line and kept as content in the same section,
#instead of opening a new one. Any other real label (e.g. "languages"
#appearing after "education") still closes the section normally -- this
#only prevents a section from fragmenting into copies of itself.
#
#v9 fixes (this revision) — real resume (Abhijeet Sangle, 3-page,
#two-column layout):
#
#FIX 14 — "Major Projects" and "Previous Experience:" are sub-headings
#INSIDE the professional experience narrative on this resume (each one
#describes dashboards/projects delivered under whichever job precedes
#it -- not a separate standalone "projects" section like another
#resume's "KEY PROJECTS" closing section, which has no job context and
#correctly still gets its own section since it matches the "projects"
#pattern directly and confidently via Strategy A). Neither heading
#matches any SECTION_PATTERNS entry, so both fell to Strategy B and
#landed as "unknown" -- but since "unknown" while inside "experience"
#wasn't covered by _LIST_SECTION_LABELS, each occurrence opened a new
#"unknown" blob instead of staying part of the same experience section.
#Fixed by adding "experience" to _LIST_SECTION_LABELS. Also added
#"previous\s+" as an allowed "experience" prefix (harmless either way,
#and correct if "Previous Experience" ever appears as a genuine
#standalone second experience block elsewhere). We deliberately did NOT
#add "major" as a "projects" prefix -- that would have made "Major
#Projects" match confidently via Strategy A and always split into its
#own section, which is wrong specifically on THIS resume's structure.
#
#FIX 15 (CORRECTED below by FIX 17) — Bare contact-info lines (a phone
#number or email address with nothing else on the line) were silently
#absorbed into whatever content section happened to be open, with no
#heading at all in front of them. The original version of this fix
#DROPPED such lines entirely on the assumption that Layer 2's
#extract_contact() always scans Layer 0's full raw text independently
#of section boundaries -- that assumption was WRONG for how this
#pipeline is actually run: test_contact_extractor.py calls
#extract_contact() on the SAVED segmented_text/*.txt file (Layer 1's
#output), not on Layer 0's raw extracted_text. Dropping the line meant
#it was gone from that file entirely, and extract_contact() had nothing
#left to find -- confirmed on Abhijeet's resume, where every contact
#field came back None after this fix was applied. See FIX 17 (v11)
#below for the corrected version.
#
#v10 fixes — replaced the blunt "second summary always means
#experience" rule (FIX 3) with a much narrower one, after confirming on
#a real resume (Abhishek Sharma) that FIX 3 wrongly merged
#legitimately-repeated "PROFESSIONAL SUMMARY"-style headings into
#"experience" when they weren't actually job history at all:
#
#FIX 16 — A "summary"-matched heading is now ONLY reclassified as
#"experience" when BOTH of these hold:
#  (a) no real "experience" section has been found ANYWHERE in the
#      document yet (tracked via experience_seen), and
#  (b) the very next non-blank line looks like an actual job entry --
#      a title followed by a date range, e.g. "PRINCIPAL CONSULTANT,
#      01/2023 - Current" (checked with _looks_like_job_entry()).
#This is deliberately much narrower than the earlier date-range
#heuristic that was proposed and rejected during the v4 fixes (that
#version would have run on ANY heading and risked misfiring on
#education entries, which also contain date ranges). This version only
#ever runs on a heading that already matched "summary" specifically --
#education never matches that pattern in the first place, so that
#earlier risk doesn't apply here. Confirmed against three real resumes:
#Aarti's second "PROFESSIONAL SUMMARY" (immediately followed by
#"PRINCIPAL CONSULTANT, 01/2023 - Current") correctly overrides to
#"experience"; Abhay's and Abhijeet's single "PROFESSIONAL SUMMARY" /
#"Professional Summary" (immediately followed by ordinary prose, not a
#job-entry line) correctly stays "summary", and their actual experience
#sections are found normally via direct Strategy A matches later in the
#document (which also sets experience_seen = True, so the override
#condition (a) is unavailable from then on for any further summary
#headings that might appear afterward).
#
#v11 fixes (this revision) — corrects the FIX 15 regression:
#
#FIX 17 — Bare contact lines are no longer dropped. Instead, any line
#that would previously have been dropped is collected into
#deferred_contact_lines and appended to the "header" section's raw_text
#once the whole document has been processed (or, in the rare case no
#header section exists at all, inserted as a new one). This keeps
#"experience" (and other content sections) free of stray phone/email
#lines -- the original goal -- while guaranteeing the contact info is
#still present SOMEWHERE in the segmented output file, since that's
#what extract_contact() is actually run against in this pipeline.
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    """
#    Builds a regex pattern for one section type, given a list of "core"
#    phrase alternatives (e.g. ["certifications?", "licen[sc]es?"]).
#
#    The resulting pattern matches a line if it is EXACTLY one of those
#    core phrases, OR one of those phrases followed by a short qualifier
#    like " & Licences" or " and Research" (capped at 3 trailing words).
#    """
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
## ─────────────────────────────────────────────────────────────
## SECTION HEADING PATTERNS
## ─────────────────────────────────────────────────────────────
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"(work\s+|industry\s+|relevant\s+|professional\s+|previous\s+)?experience",
#        r"employment(\s+history)?", r"professional\s+background",
#        r"career\s+history", r"work\s+history", r"internships?",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?", r"academic(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _build_section_pattern([
#        r"(technical\s+|core\s+|key\s+|professional\s+)?(skills?|competenc(y|ies))",
#        r"expertise", r"technologies",
#        r"programming\s+(languages?|skills?)",
#        r"tools?\s*(&|and)\s*technologies",
#        r"core\s+competencies\s*(&|and)\s*technical\s+skills",
#    ]),
#    "projects": _build_section_pattern([
#        r"(personal\s+|side\s+|key\s+|notable\s+|academic\s+)?projects?",
#        r"portfolio",
#    ]),
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#}
#
#
## FIX 13 (v8): section types where multiple distinct sub-entries with
## their own short sub-heading-like lines are common (e.g. a skills
## block with "Core Competencies" / "Frameworks..." sub-labels, or an
## education block with a second "Higher Secondary Education" entry).
## Used by split_into_sections() to stop a section from fragmenting
## into copies of itself whenever a sub-heading re-matches its own
## label or falls through to "unknown".
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
## A list of common section-related keywords used by Strategy B to check
## whether an unrecognised heading-shaped line is at least "about" one of
## the topics we know resumes cover.
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration",
#]
#
#
#@dataclass
#class Section:
#    """Represents one identified section of a resume."""
#    label: str           # e.g. "experience", "skills"
#    raw_text: str        # the full text content of this section
#    start_line: int      # line number where this section starts
#    confidence: float    # 0.0 to 1.0, how sure we are this is the right label
#
#
#def _clean_heading_candidate(line: str) -> str:
#    """
#    Strips common heading decoration (colons, dashes, surrounding
#    whitespace) before checking a line against our patterns, so
#    "EDUCATION:" or "- SKILLS -" still match plain "EDUCATION"/"SKILLS".
#    """
#    cleaned = line.strip()
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    """
#    FIX 2 (v3): Returns True if a line that matched a section pattern
#    is actually a content word wrapped onto its own line due to PDF
#    text extraction, NOT a real section heading.
#
#    A wrapped word has ALL of these properties:
#      1. It is a single word (no spaces).
#      2. It is entirely lowercase.
#      3. The previous non-empty line ends mid-sentence (no . : ; ? !).
#    """
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    """
#    FIX 6 (v5): Some resumes place a heading and its first bullet(s) of
#    content on the SAME physical line, e.g.:
#
#        "CORE COMPETENCIES • SQL • HDL Data Loader"
#
#    detect_section_label() requires an exact, whole-line match by
#    design (see v1 fix), so a line like this never matches "skills" at
#    all and silently stays glued onto whatever section came before it.
#    This only runs as a FALLBACK, after detect_section_label() has
#    already failed to match the full line.
#
#    Tries to find a known heading phrase at the START of the line,
#    followed by a separator (bullet, colon, dash) and remaining
#    content. Returns (heading, remaining_content) if found, or
#    (None, None) if this doesn't look like a merged heading line.
#    """
#    stripped = line.strip()
#
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    """
#    FIX 7 (v5): Some resumes pack multiple skills onto one physical
#    bullet line, e.g. "• SQL • HDL Data Loader" (one bullet visually,
#    two skills). Splits any line containing more than one bullet
#    marker into one bullet per line.
#    """
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
##def _is_bare_contact_line(line: str) -> bool:
##    """
##    FIX 15 (v9): True if a line is JUST a phone number or JUST an email
##    address, with nothing else on it. Confirmed on a real resume
##    (Abhijeet Sangle): Layer 0's primary/secondary column-stream
##    ordering places contact info with no heading in front of it, right
##    where the last primary-stream chunk happens to end -- so it
##    silently gets absorbed as trailing content of whatever content
##    section is open at that point. Skipping these lines doesn't lose
##    the phone/email: Layer 2's extract_contact() already scans the
##    full resume text independently of section boundaries.
##    """
##    stripped = line.strip()
##    if not stripped:
##        return False
##    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#def _is_bare_contact_line(line: str) -> bool:
#    """
#    FIX 15 (v9): True if a line is JUST a phone number or JUST an email
#    address, with nothing else on it. Confirmed on a real resume
#    (Abhijeet Sangle): Layer 0's primary/secondary column-stream
#    ordering places contact info with no heading in front of it, right
#    where the last primary-stream chunk happens to end -- so it
#    silently gets absorbed as trailing content of whatever content
#    section is open at that point.
#
#    FIX 18 (v12): also strips a leading icon/glyph (📍📞✉☎ etc.) before
#    testing. Some resume templates prefix contact lines with an icon
#    instead of a text label (e.g. "📞 8299435521"), which previously
#    failed to match since the emoji broke the pattern from the first
#    character. Generic fix: strip any leading run of non-word,
#    non-"+" characters first. This does NOT touch location-only lines
#    like "📍 Delhi, India" -- after stripping the icon, that still
#    doesn't match a phone or email pattern, so it's correctly left
#    alone as ordinary content.
#    """
#    stripped = line.strip()
#    if not stripped:
#        return False
#    stripped = re.sub(r"^[^\w+]+", "", stripped).strip()
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{1,2}/)?\d{4}\s*[-\u2013\u2014]+\s*"
#    r"(?:Current|Present|current|present|(?:\d{1,2}/)?\d{4})"
#)
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    """
#    FIX 16 (v10): True if a line contains a recognisable date range in
#    a job-history shape (e.g. "PRINCIPAL CONSULTANT, 01/2023 -
#    Current"). Deliberately narrow -- only ever called on the line
#    immediately after a heading that ALREADY matched "summary", never
#    on arbitrary lines, so it cannot misfire on education date ranges
#    the way a general-purpose version would.
#    """
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    """Returns the first non-blank line at or after start_index, or ''."""
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    """
#    Strategy A: Checks a line against all known section heading patterns.
#    Returns (label, confidence) or (None, 0.0) if no match.
#    """
#    if prev_lines is None:
#        prev_lines = []
#
#    candidate = _clean_heading_candidate(line)
#
#    if not candidate:
#        return None, 0.0
#
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    """
#    Checks whether a line contains any word commonly associated with
#    resume section headings.
#    """
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    """
#    Strategy B: Scores a line on how likely it is to be a section heading,
#    for cases where Strategy A didn't recognise the wording at all.
#    Returns a score from 0.0 (not a heading) to 1.0 (definitely a heading).
#    """
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#
#    # FIX 11 (v7): a line starting with a bullet marker is a list item
#    # by definition, never a section heading -- regardless of what
#    # keywords it happens to contain. Confirmed on a real resume:
#    # "• Large Language Models(LLM)" and "• Natural Language Processing"
#    # both scored exactly 0.65 (just over threshold) purely because
#    # "language" is in _SECTION_KEYWORDS, fragmenting one skills block
#    # into multiple pieces at essentially random bullet lines.
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#
#    if line_index <= 1:
#        return 0.0
#
#    word_count = len(stripped.split())
#
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#
#    if stripped.endswith("."):
#        return 0.0
#
#    score = 0.0
#
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#
#    if stripped.endswith(":"):
#        score += 0.15
#
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    """
#    Main entry point for Layer 1.
#
#    Takes the full resume text (Layer 0's output) and returns a list of
#    Section objects -- each with a label, its text content, the line it
#    started on, and a confidence score.
#    """
#    lines = text.split("\n")
#    sections = []
#
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#
#    # FIX 16 (v10): tracks whether a real "experience" section has been
#    # found anywhere in the document yet. Used to scope the
#    # summary-heading job-entry override below.
#    experience_seen = False
#
#    # FIX 17 (v11): bare contact lines (phone/email, nothing else on
#    # the line) that would otherwise pollute a content section are
#    # collected here instead of being dropped, and appended to the
#    # "header" section once the whole document has been processed.
#    deferred_contact_lines = []
#
#    for i, line in enumerate(lines):
#        stripped = line.strip()
#
#        # Pass previous lines so detect_section_label can apply
#        # the wrapped-word guard (FIX 2).
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#
#        # FIX 6 (v5): if the whole line didn't match a heading on its
#        # own, check whether it's a heading MERGED with its first bit
#        # of content on the same physical line. Only tried as a
#        # fallback -- behaviour for lines that already match cleanly
#        # on their own is completely untouched.
#        inline_content = None
#        if label is None and stripped != "":
#            heading_candidate, remaining = split_inline_heading(stripped)
#            if heading_candidate is not None:
#                label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                if label is not None:
#                    inline_content = remaining
#
#        if label is None and stripped != "":
#            # Apply the same wrapped-word guard to Strategy B.
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        # FIX 13 (v8): suppress false sub-section boundaries inside any
#        # "list-like" section (skills, education, certifications, etc.
#        # -- see _LIST_SECTION_LABELS). Sub-entries such as "Core
#        # Competencies" under a skills heading, or "Higher Secondary
#        # Education" under an education heading, often sit with no
#        # blank line separating them from the section they belong to,
#        # and because they're topically related words themselves, they
#        # trip either Strategy A directly (re-matching their own label)
#        # or Strategy B's keyword bonus (landing as "unknown"). Once
#        # we're already inside one of these section types, treat any
#        # further match of the SAME label or "unknown" as a sub-heading
#        # line -- keep it as content in the SAME section instead of
#        # starting a new one. Any other real label (e.g. "languages"
#        # appearing after "education") still closes the section
#        # normally -- this only prevents a section from fragmenting
#        # into copies of itself.
#        if current_label in _LIST_SECTION_LABELS and label in (current_label, "unknown"):
#            current_lines.append(line)
#            continue
#
#        if label is not None and stripped != "":
#            # FIX 16 (v10): a "summary" heading is only reclassified as
#            # "experience" if no real experience section has been found
#            # anywhere yet AND the very next line looks like an actual
#            # job entry (title + date range). Much narrower than the
#            # old FIX 3, which forced EVERY repeated "summary" match to
#            # "experience" unconditionally -- that wrongly merged
#            # legitimately-repeated summary-style headings on other
#            # resumes (confirmed on Abhishek Sharma's resume).
#            if label == "summary" and not experience_seen:
#                upcoming = _next_nonblank_line(lines, i + 1)
#                if _looks_like_job_entry(upcoming):
#                    label = "experience"
#                    confidence = 0.85
#
#            if label == "experience":
#                experience_seen = True
#
#            # FIX 7 (v5): normalize multi-bullet lines before finalising
#            # the section that's about to be closed.
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            # FIX 6 (v5): seed the new section with whatever content
#            # was merged onto the same line as its heading.
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            # FIX 17 (v11, corrects FIX 15): don't let a bare contact
#            # line (just a phone number or just an email, no heading,
#            # no other text on the line) pollute a real content section
#            # -- except in "header", where it genuinely belongs. Unlike
#            # the original FIX 15, we do NOT drop it: this pipeline's
#            # test_contact_extractor.py runs extract_contact() on the
#            # SAVED segmented_text/*.txt file, not on Layer 0's raw
#            # text, so dropping the line removed it from the pipeline
#            # entirely (confirmed on Abhijeet's resume -- every contact
#            # field came back None). Instead we stash it and merge it
#            # into the "header" section once the document is done.
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                deferred_contact_lines.append(stripped)
#            else:
#                current_lines.append(line)
#
#    # FIX 7 (v5): normalize the final section too.
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    # FIX 17 (v11): merge any deferred contact lines into the "header"
#    # section now that the whole document has been processed. If no
#    # header section exists at all (very unlikely -- header always
#    # starts as current_label), insert a new minimal one instead so the
#    # contact info isn't lost.
#    if deferred_contact_lines:
#        for s in sections:
#            if s.label == "header":
#                s.raw_text = (s.raw_text + "\n" + "\n".join(deferred_contact_lines)).strip()
#                break
#        else:
#            sections.insert(0, Section(
#                label="header",
#                raw_text="\n".join(deferred_contact_lines),
#                start_line=0,
#                confidence=0.9,
#            ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    """
#    Helper: returns the combined text of all sections with a given label.
#    """
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#

















##worked-changing for email and phone number to move on header section
#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 10)
#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Line feature scoring (handles ambiguous/unrecognised headings)
#
#REVISION HISTORY (kept here so we don't repeat fixed mistakes):
#
#v1 bug (fixed): none of the original regex patterns had an end-anchor,
#so they matched the START of a line and ignored everything after it.
#Fixed by anchoring every pattern at both start (^) and end ($).
#
#v2 bug (fixed in revision 2): once patterns were strictly anchored,
#they became too rigid -- compound headings like "CERTIFICATIONS &
#LICENCES" didn't match. Fixed by allowing each core heading phrase to
#optionally take a short "& <1-3 words>" / "and <1-3 words>" qualifier.
#
#v3 fixes (revision 3) — real resume (Aarti Suranje, sidebar layout):
#FIX 1 — "PERSONAL SUMMARY" not recognised; added "personal" as an
#allowed summary prefix alongside "professional".
#FIX 2 — A content word wrapped onto its own line (e.g. "experience"
#alone, continuing a sentence from the line above) was misread as a
#heading. Added _is_wrapped_word() guard: suppress a Strategy A match
#if the matched line is a single all-lowercase word AND the previous
#non-empty line doesn't end with sentence-closing punctuation.
#
#v4 fixes (revision 4) — continued debugging of Aarti Suranje's resume:
#
#FIX 3 (SUPERSEDED by FIX 16, v10 -- see below) — originally: any
#second heading matching "summary" was unconditionally forced to
#"experience". This turned out to be too blunt: a real resume
#(Abhishek Sharma) legitimately has "PROFESSIONAL SUMMARY"-style
#headings appearing more than once for genuinely separate summary-like
#content, and the old rule wrongly merged all of it into "experience".
#Replaced entirely by FIX 16.
#
#FIX 4 — "LANGUAGE SKILLS" not recognised; languages pattern now allows
#an optional "skills"/"known"/"proficiency" suffix.
#FIX 5 — "CUSTOM SECTION" had no matching label at all; added a new
#"other" catch-all label.
#
#v5 fixes (revision 5) — heading merged with content on the same line:
#FIX 6 — "CORE COMPETENCIES • SQL • HDL Data Loader" never matched the
#skills pattern as a whole line, so it stayed glued onto whatever
#section preceded it. We do NOT weaken the anchored regex (that's what
#v1 was built to enforce) -- instead split_inline_heading() runs as a
#FALLBACK after detect_section_label() has already failed on the full
#line: finds a known heading phrase at the start of the line, splits it
#from the remaining content at a separator (bullet/colon/dash), and
#re-runs detect_section_label() on just the heading part. Generic
#across every heading and every common separator.
#FIX 7 — Multiple skills packed onto one bullet line (e.g.
#"• SQL • HDL Data Loader") are split into one bullet per line by
#normalize_inline_bullets(), applied whenever a section's raw_text is
#finalised.
#
#v6 fixes (revision 6) — three missing heading patterns (Aasma resume):
#FIX 8 — "Professional Skills" not recognised; added "professional" as
#an allowed skills prefix.
#FIX 9 — "Strengths" had no pattern at all; added new "strengths" label.
#FIX 10 — "Personal Details" / "Declaration" had no pattern either;
#added new "personal_details" and "declaration" labels.
#
#v7 fixes (revision 7) — two compounding bugs found on a real resume
#(Abhay Tyagi) that together fragmented one TECHNICAL SKILLS block into
#five separate pieces:
#
#FIX 11 — Bullet content lines misread as headings by Strategy B purely
#because they contain a keyword substring. "• Natural Language
#Processing" and "• Large Language Models(LLM)" both scored exactly
#0.65 (just over the 0.65 threshold) because "language" is in
#_SECTION_KEYWORDS. A section heading is never itself a bullet list
#item, so score_heading_line() now returns 0.0 immediately for any line
#that starts with a bullet marker, before any other scoring runs.
#
#FIX 12 (superseded by FIX 13 below) — originally scoped only to
#"skills": a sub-category label like "Core Competencies" sitting
#directly under "TECHNICAL SKILLS" with no blank line between them kept
#starting a brand-new "skills" section instead of continuing the one
#already open.
#
#v8 fixes (this revision) — the same bug as FIX 12, confirmed on the
#SAME resume in a second section type:
#
#FIX 13 — Generalized FIX 12. Abhay's resume also has "Higher Secondary
#Education" as a sub-heading directly under "EDUCATION" (his second
#school entry), with no blank line separating it from his first degree.
#"Higher Secondary Education" doesn't match the "education" pattern as
#a whole line (the pattern only matches "education"/"educational
#background" alone, not an arbitrary phrase containing the word), so it
#falls to Strategy B, scores exactly 0.65 (word count <=4, no comma,
#contains the keyword "education"), and gets tagged "unknown" -- which
#incorrectly opened a second, separate section instead of continuing
#the existing "education" one. Identical failure mode to FIX 12, just
#in a different section type.
#
#Rather than patch this one label at a time as we keep hitting it
#(skills, now education, likely certifications/languages/projects
#next), the fix is generalized: _LIST_SECTION_LABELS names every
#section type where multiple sub-entries with their own short
#sub-heading-like lines are common. Once current_label is one of these,
#any further match of the SAME label or "unknown" is treated as a
#sub-heading/sub-entry line and kept as content in the same section,
#instead of opening a new one. Any other real label (e.g. "languages"
#appearing after "education") still closes the section normally -- this
#only prevents a section from fragmenting into copies of itself.
#
#v9 fixes (this revision) — real resume (Abhijeet Sangle, 3-page,
#two-column layout):
#
#FIX 14 — "Major Projects" and "Previous Experience:" are sub-headings
#INSIDE the professional experience narrative on this resume (each one
#describes dashboards/projects delivered under whichever job precedes
#it -- not a separate standalone "projects" section like another
#resume's "KEY PROJECTS" closing section, which has no job context and
#correctly still gets its own section since it matches the "projects"
#pattern directly and confidently via Strategy A). Neither heading
#matches any SECTION_PATTERNS entry, so both fell to Strategy B and
#landed as "unknown" -- but since "unknown" while inside "experience"
#wasn't covered by _LIST_SECTION_LABELS, each occurrence opened a new
#"unknown" blob instead of staying part of the same experience section.
#Fixed by adding "experience" to _LIST_SECTION_LABELS. Also added
#"previous\s+" as an allowed "experience" prefix (harmless either way,
#and correct if "Previous Experience" ever appears as a genuine
#standalone second experience block elsewhere). We deliberately did NOT
#add "major" as a "projects" prefix -- that would have made "Major
#Projects" match confidently via Strategy A and always split into its
#own section, which is wrong specifically on THIS resume's structure.
#
#FIX 15 — Bare contact-info lines (a phone number or email address with
#nothing else on the line) were silently absorbed into whatever content
#section happened to be open, with no heading at all in front of them.
#Confirmed on Abhijeet's resume: Layer 0's primary/secondary column-
#stream ordering places his "Mob No 91- 8976365575" / email lines with
#no heading before them, right where the last primary-stream chunk
#happens to end -- so FIX 14 above (correctly) merged them into
#"experience" along with everything else in that stretch. This does NOT
#affect contact extraction accuracy: Layer 2's extract_contact() already
#scans the full resume text independently of section boundaries, so the
#phone/email are found either way. This fix is purely about section
#content purity, in case "experience" text is used directly by future
#per-section parsing. _is_bare_contact_line() detects a line that is
#JUST a phone number or JUST an email (nothing else), and such a line is
#dropped when it would otherwise land in any content section EXCEPT
#"header" -- in "header" a phone/email genuinely belongs and is kept.
#
#v10 fixes (this revision) — replaced the blunt "second summary always
#means experience" rule (FIX 3) with a much narrower one, after
#confirming on a real resume (Abhishek Sharma) that FIX 3 wrongly
#merged legitimately-repeated "PROFESSIONAL SUMMARY"-style headings
#into "experience" when they weren't actually job history at all:
#
#FIX 16 — A "summary"-matched heading is now ONLY reclassified as
#"experience" when BOTH of these hold:
#  (a) no real "experience" section has been found ANYWHERE in the
#      document yet (tracked via experience_seen), and
#  (b) the very next non-blank line looks like an actual job entry --
#      a title followed by a date range, e.g. "PRINCIPAL CONSULTANT,
#      01/2023 - Current" (checked with _looks_like_job_entry()).
#This is deliberately much narrower than the earlier date-range
#heuristic that was proposed and rejected during the v4 fixes (that
#version would have run on ANY heading and risked misfiring on
#education entries, which also contain date ranges). This version only
#ever runs on a heading that already matched "summary" specifically --
#education never matches that pattern in the first place, so that
#earlier risk doesn't apply here. Confirmed against three real resumes:
#Aarti's second "PROFESSIONAL SUMMARY" (immediately followed by
#"PRINCIPAL CONSULTANT, 01/2023 - Current") correctly overrides to
#"experience"; Abhay's and Abhijeet's single "PROFESSIONAL SUMMARY" /
#"Professional Summary" (immediately followed by ordinary prose, not a
#job-entry line) correctly stays "summary", and their actual experience
#sections are found normally via direct Strategy A matches later in the
#document (which also sets experience_seen = True, so the override
#condition (a) is unavailable from then on for any further summary
#headings that might appear afterward).
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    """
#    Builds a regex pattern for one section type, given a list of "core"
#    phrase alternatives (e.g. ["certifications?", "licen[sc]es?"]).
#
#    The resulting pattern matches a line if it is EXACTLY one of those
#    core phrases, OR one of those phrases followed by a short qualifier
#    like " & Licences" or " and Research" (capped at 3 trailing words).
#    """
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
## ─────────────────────────────────────────────────────────────
## SECTION HEADING PATTERNS
## ─────────────────────────────────────────────────────────────
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"(work\s+|industry\s+|relevant\s+|professional\s+|previous\s+)?experience",
#        r"employment(\s+history)?", r"professional\s+background",
#        r"career\s+history", r"work\s+history", r"internships?",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?", r"academic(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _build_section_pattern([
#        r"(technical\s+|core\s+|key\s+|professional\s+)?(skills?|competenc(y|ies))",
#        r"expertise", r"technologies",
#        r"programming\s+(languages?|skills?)",
#        r"tools?\s*(&|and)\s*technologies",
#        r"core\s+competencies\s*(&|and)\s*technical\s+skills",
#    ]),
#    "projects": _build_section_pattern([
#        r"(personal\s+|side\s+|key\s+|notable\s+|academic\s+)?projects?",
#        r"portfolio",
#    ]),
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#}
#
#
## FIX 13 (v8): section types where multiple distinct sub-entries with
## their own short sub-heading-like lines are common (e.g. a skills
## block with "Core Competencies" / "Frameworks..." sub-labels, or an
## education block with a second "Higher Secondary Education" entry).
## Used by split_into_sections() to stop a section from fragmenting
## into copies of itself whenever a sub-heading re-matches its own
## label or falls through to "unknown".
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}
#
#
## A list of common section-related keywords used by Strategy B to check
## whether an unrecognised heading-shaped line is at least "about" one of
## the topics we know resumes cover.
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration",
#]
#
#
#@dataclass
#class Section:
#    """Represents one identified section of a resume."""
#    label: str           # e.g. "experience", "skills"
#    raw_text: str        # the full text content of this section
#    start_line: int      # line number where this section starts
#    confidence: float    # 0.0 to 1.0, how sure we are this is the right label
#
#
#def _clean_heading_candidate(line: str) -> str:
#    """
#    Strips common heading decoration (colons, dashes, surrounding
#    whitespace) before checking a line against our patterns, so
#    "EDUCATION:" or "- SKILLS -" still match plain "EDUCATION"/"SKILLS".
#    """
#    cleaned = line.strip()
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    """
#    FIX 2 (v3): Returns True if a line that matched a section pattern
#    is actually a content word wrapped onto its own line due to PDF
#    text extraction, NOT a real section heading.
#
#    A wrapped word has ALL of these properties:
#      1. It is a single word (no spaces).
#      2. It is entirely lowercase.
#      3. The previous non-empty line ends mid-sentence (no . : ; ? !).
#    """
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False
#
#
#def split_inline_heading(line: str):
#    """
#    FIX 6 (v5): Some resumes place a heading and its first bullet(s) of
#    content on the SAME physical line, e.g.:
#
#        "CORE COMPETENCIES • SQL • HDL Data Loader"
#
#    detect_section_label() requires an exact, whole-line match by
#    design (see v1 fix), so a line like this never matches "skills" at
#    all and silently stays glued onto whatever section came before it.
#    This only runs as a FALLBACK, after detect_section_label() has
#    already failed to match the full line.
#
#    Tries to find a known heading phrase at the START of the line,
#    followed by a separator (bullet, colon, dash) and remaining
#    content. Returns (heading, remaining_content) if found, or
#    (None, None) if this doesn't look like a merged heading line.
#    """
#    stripped = line.strip()
#
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    """
#    FIX 7 (v5): Some resumes pack multiple skills onto one physical
#    bullet line, e.g. "• SQL • HDL Data Loader" (one bullet visually,
#    two skills). Splits any line containing more than one bullet
#    marker into one bullet per line.
#    """
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#_BARE_EMAIL_LINE = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w+$")
#_BARE_PHONE_LINE = re.compile(
#    r"^(mob(ile)?\.?\s*(no\.?|number)?\s*[:\-]?\s*|contact\s*[:\-]?\s*|"
#    r"phone\s*[:\-]?\s*|tel\s*[:\-]?\s*)?\+?\d[\d\-\s()]{6,}\d$",
#    re.IGNORECASE,
#)
#
#
#def _is_bare_contact_line(line: str) -> bool:
#    """
#    FIX 15 (v9): True if a line is JUST a phone number or JUST an email
#    address, with nothing else on it. Confirmed on a real resume
#    (Abhijeet Sangle): Layer 0's primary/secondary column-stream
#    ordering places contact info with no heading in front of it, right
#    where the last primary-stream chunk happens to end -- so it
#    silently gets absorbed as trailing content of whatever content
#    section is open at that point. Skipping these lines doesn't lose
#    the phone/email: Layer 2's extract_contact() already scans the
#    full resume text independently of section boundaries.
#    """
#    stripped = line.strip()
#    if not stripped:
#        return False
#    return bool(_BARE_EMAIL_LINE.match(stripped) or _BARE_PHONE_LINE.match(stripped))
#
#
#_JOB_DATE_RANGE = re.compile(
#    r"(?:,\s*)?(?:\d{1,2}/)?\d{4}\s*[-\u2013\u2014]+\s*"
#    r"(?:Current|Present|current|present|(?:\d{1,2}/)?\d{4})"
#)
#
#
#def _looks_like_job_entry(line: str) -> bool:
#    """
#    FIX 16 (v10): True if a line contains a recognisable date range in
#    a job-history shape (e.g. "PRINCIPAL CONSULTANT, 01/2023 -
#    Current"). Deliberately narrow -- only ever called on the line
#    immediately after a heading that ALREADY matched "summary", never
#    on arbitrary lines, so it cannot misfire on education date ranges
#    the way a general-purpose version would.
#    """
#    stripped = line.strip()
#    if not stripped or len(stripped) > 120:
#        return False
#    return bool(_JOB_DATE_RANGE.search(stripped))
#
#
#def _next_nonblank_line(lines: list, start_index: int) -> str:
#    """Returns the first non-blank line at or after start_index, or ''."""
#    i = start_index
#    while i < len(lines):
#        if lines[i].strip():
#            return lines[i].strip()
#        i += 1
#    return ""
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    """
#    Strategy A: Checks a line against all known section heading patterns.
#    Returns (label, confidence) or (None, 0.0) if no match.
#    """
#    if prev_lines is None:
#        prev_lines = []
#
#    candidate = _clean_heading_candidate(line)
#
#    if not candidate:
#        return None, 0.0
#
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    """
#    Checks whether a line contains any word commonly associated with
#    resume section headings.
#    """
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    """
#    Strategy B: Scores a line on how likely it is to be a section heading,
#    for cases where Strategy A didn't recognise the wording at all.
#    Returns a score from 0.0 (not a heading) to 1.0 (definitely a heading).
#    """
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#
#    # FIX 11 (v7): a line starting with a bullet marker is a list item
#    # by definition, never a section heading -- regardless of what
#    # keywords it happens to contain. Confirmed on a real resume:
#    # "• Large Language Models(LLM)" and "• Natural Language Processing"
#    # both scored exactly 0.65 (just over threshold) purely because
#    # "language" is in _SECTION_KEYWORDS, fragmenting one skills block
#    # into multiple pieces at essentially random bullet lines.
#    if stripped[:1] in ("•", "●"):
#        return 0.0
#
#    if line_index <= 1:
#        return 0.0
#
#    word_count = len(stripped.split())
#
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#
#    if stripped.endswith("."):
#        return 0.0
#
#    score = 0.0
#
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#
#    if stripped.endswith(":"):
#        score += 0.15
#
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    """
#    Main entry point for Layer 1.
#
#    Takes the full resume text (Layer 0's output) and returns a list of
#    Section objects -- each with a label, its text content, the line it
#    started on, and a confidence score.
#    """
#    lines = text.split("\n")
#    sections = []
#
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#
#    # FIX 16 (v10): tracks whether a real "experience" section has been
#    # found anywhere in the document yet. Used to scope the
#    # summary-heading job-entry override below.
#    experience_seen = False
#
#    for i, line in enumerate(lines):
#        stripped = line.strip()
#
#        # Pass previous lines so detect_section_label can apply
#        # the wrapped-word guard (FIX 2).
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#
#        # FIX 6 (v5): if the whole line didn't match a heading on its
#        # own, check whether it's a heading MERGED with its first bit
#        # of content on the same physical line. Only tried as a
#        # fallback -- behaviour for lines that already match cleanly
#        # on their own is completely untouched.
#        inline_content = None
#        if label is None and stripped != "":
#            heading_candidate, remaining = split_inline_heading(stripped)
#            if heading_candidate is not None:
#                label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                if label is not None:
#                    inline_content = remaining
#
#        if label is None and stripped != "":
#            # Apply the same wrapped-word guard to Strategy B.
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        # FIX 13 (v8): suppress false sub-section boundaries inside any
#        # "list-like" section (skills, education, certifications, etc.
#        # -- see _LIST_SECTION_LABELS). Sub-entries such as "Core
#        # Competencies" under a skills heading, or "Higher Secondary
#        # Education" under an education heading, often sit with no
#        # blank line separating them from the section they belong to,
#        # and because they're topically related words themselves, they
#        # trip either Strategy A directly (re-matching their own label)
#        # or Strategy B's keyword bonus (landing as "unknown"). Once
#        # we're already inside one of these section types, treat any
#        # further match of the SAME label or "unknown" as a sub-heading
#        # line -- keep it as content in the SAME section instead of
#        # starting a new one. Any other real label (e.g. "languages"
#        # appearing after "education") still closes the section
#        # normally -- this only prevents a section from fragmenting
#        # into copies of itself.
#        if current_label in _LIST_SECTION_LABELS and label in (current_label, "unknown"):
#            current_lines.append(line)
#            continue
#
#        if label is not None and stripped != "":
#            # FIX 16 (v10): a "summary" heading is only reclassified as
#            # "experience" if no real experience section has been found
#            # anywhere yet AND the very next line looks like an actual
#            # job entry (title + date range). Much narrower than the
#            # old FIX 3, which forced EVERY repeated "summary" match to
#            # "experience" unconditionally -- that wrongly merged
#            # legitimately-repeated summary-style headings on other
#            # resumes (confirmed on Abhishek Sharma's resume).
#            if label == "summary" and not experience_seen:
#                upcoming = _next_nonblank_line(lines, i + 1)
#                if _looks_like_job_entry(upcoming):
#                    label = "experience"
#                    confidence = 0.85
#
#            if label == "experience":
#                experience_seen = True
#
#            # FIX 7 (v5): normalize multi-bullet lines before finalising
#            # the section that's about to be closed.
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            # FIX 6 (v5): seed the new section with whatever content
#            # was merged onto the same line as its heading.
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            # FIX 15 (v9): don't let a bare contact line (just a phone
#            # number or just an email, no heading, no other text on the
#            # line) pollute a real content section -- except in
#            # "header", where a phone/email genuinely belongs. Dropping
#            # it here doesn't lose it: Layer 2's extract_contact() scans
#            # the full resume text independently of these boundaries.
#            if current_label != "header" and _is_bare_contact_line(stripped):
#                pass
#            else:
#                current_lines.append(line)
#
#    # FIX 7 (v5): normalize the final section too.
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    """
#    Helper: returns the combined text of all sections with a given label.
#    """
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#












#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 8)#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Line feature scoring (handles ambiguous/unrecognised headings)#
#REVISION HISTORY (kept here so we don't repeat fixed mistakes):#
#v1 bug (fixed): none of the original regex patterns had an end-anchor,
#so they matched the START of a line and ignored everything after it.
#Fixed by anchoring every pattern at both start (^) and end ($).#
#v2 bug (fixed in revision 2): once patterns were strictly anchored,
#they became too rigid -- compound headings like "CERTIFICATIONS &
#LICENCES" didn't match. Fixed by allowing each core heading phrase to
#optionally take a short "& <1-3 words>" / "and <1-3 words>" qualifier.#
#v3 fixes (revision 3) — real resume (Aarti Suranje, sidebar layout):
#FIX 1 — "PERSONAL SUMMARY" not recognised; added "personal" as an
#allowed summary prefix alongside "professional".
#FIX 2 — A content word wrapped onto its own line (e.g. "experience"
#alone, continuing a sentence from the line above) was misread as a
#heading. Added _is_wrapped_word() guard: suppress a Strategy A match
#if the matched line is a single all-lowercase word AND the previous
#non-empty line doesn't end with sentence-closing punctuation.#
#v4 fixes (revision 4) — continued debugging of Aarti's resume:
#FIX 3 — A second heading matching "summary" (duplicate/mislabeled)
#swallowed the entire experience section. A date-range-based "does this
#look like a job entry" override was considered and REJECTED (education
#entries can also contain date ranges and would misfire the same way).
#Instead: track whether a "summary" section has already been seen once;
#a second "summary" match is forced to "experience". Scoped narrowly to
#"summary" only.
#FIX 4 — "LANGUAGE SKILLS" not recognised; languages pattern now allows
#an optional "skills"/"known"/"proficiency" suffix.
#FIX 5 — "CUSTOM SECTION" had no matching label at all; added a new
#"other" catch-all label.#
#v5 fixes (revision 5) — heading merged with content on the same line:
#FIX 6 — "CORE COMPETENCIES • SQL • HDL Data Loader" never matched the
#skills pattern as a whole line, so it stayed glued onto whatever
#section preceded it. We do NOT weaken the anchored regex (that's what
#v1 was built to enforce) -- instead split_inline_heading() runs as a
#FALLBACK after detect_section_label() has already failed on the full
#line: finds a known heading phrase at the start of the line, splits it
#from the remaining content at a separator (bullet/colon/dash), and
#re-runs detect_section_label() on just the heading part. Generic
#across every heading and every common separator.
#FIX 7 — Multiple skills packed onto one bullet line (e.g.
#"• SQL • HDL Data Loader") are split into one bullet per line by
#normalize_inline_bullets(), applied whenever a section's raw_text is
#finalised.#
#v6 fixes (revision 6) — three missing heading patterns (Aasma resume):
#FIX 8 — "Professional Skills" not recognised; added "professional" as
#an allowed skills prefix.
#FIX 9 — "Strengths" had no pattern at all; added new "strengths" label.
#FIX 10 — "Personal Details" / "Declaration" had no pattern either;
#added new "personal_details" and "declaration" labels.#
#v7 fixes (revision 7) — two compounding bugs found on a real resume
#(Abhay Tyagi) that together fragmented one TECHNICAL SKILLS block into
#five separate pieces:#
#FIX 11 — Bullet content lines misread as headings by Strategy B purely
#because they contain a keyword substring. "• Natural Language
#Processing" and "• Large Language Models(LLM)" both scored exactly
#0.65 (just over the 0.65 threshold) because "language" is in
#_SECTION_KEYWORDS. A section heading is never itself a bullet list
#item, so score_heading_line() now returns 0.0 immediately for any line
#that starts with a bullet marker, before any other scoring runs.#
#FIX 12 (superseded by FIX 13 below) — originally scoped only to
#"skills": a sub-category label like "Core Competencies" sitting
#directly under "TECHNICAL SKILLS" with no blank line between them kept
#starting a brand-new "skills" section instead of continuing the one
#already open.#
#v8 fixes (this revision) — the same bug as FIX 12, confirmed on the
#SAME resume in a second section type:#
#FIX 13 — Generalized FIX 12. Abhay's resume also has "Higher Secondary
#Education" as a sub-heading directly under "EDUCATION" (his second
#school entry), with no blank line separating it from his first degree.
#"Higher Secondary Education" doesn't match the "education" pattern as
#a whole line (the pattern only matches "education"/"educational
#background" alone, not an arbitrary phrase containing the word), so it
#falls to Strategy B, scores exactly 0.65 (word count <=4, no comma,
#contains the keyword "education"), and gets tagged "unknown" -- which
#incorrectly opened a second, separate section instead of continuing
#the existing "education" one. Identical failure mode to FIX 12, just
#in a different section type.#
#Rather than patch this one label at a time as we keep hitting it
#(skills, now education, likely certifications/languages/projects
#next), the fix is generalized: _LIST_SECTION_LABELS names every
#section type where multiple sub-entries with their own short
#sub-heading-like lines are common. Once current_label is one of these,
#any further match of the SAME label or "unknown" is treated as a
#sub-heading/sub-entry line and kept as content in the same section,
#instead of opening a new one. Any other real label (e.g. "languages"
#appearing after "education") still closes the section normally -- this
#only prevents a section from fragmenting into copies of itself.
#"""#
#import re
#from dataclasses import dataclass##
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    """
#    Builds a regex pattern for one section type, given a list of "core"
#    phrase alternatives (e.g. ["certifications?", "licen[sc]es?"]).#
#    The resulting pattern matches a line if it is EXACTLY one of those
#    core phrases, OR one of those phrases followed by a short qualifier
#    like " & Licences" or " and Research" (capped at 3 trailing words).
#    """
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )##
## ─────────────────────────────────────────────────────────────
## SECTION HEADING PATTERNS
## ─────────────────────────────────────────────────────────────#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+|personal\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"(work\s+|industry\s+|relevant\s+|professional\s+|previous\s+)?experience",
#        r"employment(\s+history)?", r"professional\s+background",
#        r"career\s+history", r"work\s+history", r"internships?",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?", r"academic(\s+background)?",
#        r"academic\s+qualifications?", r"educational\s+qualifications?",
#        r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _build_section_pattern([
#        r"(technical\s+|core\s+|key\s+|professional\s+)?(skills?|competenc(y|ies))",
#        r"expertise", r"technologies",
#        r"programming\s+(languages?|skills?)",
#        r"tools?\s*(&|and)\s*technologies",
#        r"core\s+competencies\s*(&|and)\s*technical\s+skills",
#    ]),
#    "projects": _build_section_pattern([
#        r"(personal\s+|side\s+|key\s+|notable\s+|academic\s+)?projects?",
#        r"portfolio",
#    ]),
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#    "strengths": _build_section_pattern([
#        r"(key\s+)?strengths?", r"core\s+strengths?",
#    ]),
#    "personal_details": _build_section_pattern([
#        r"personal\s+(details|information|profile|data)",
#        r"bio\s*-?\s*data",
#    ]),
#    "declaration": _build_section_pattern([
#        r"declaration", r"self[\s-]?declaration",
#    ]),
#}##
## FIX 13 (v8): section types where multiple distinct sub-entries with
## their own short sub-heading-like lines are common (e.g. a skills
## block with "Core Competencies" / "Frameworks..." sub-labels, or an
## education block with a second "Higher Secondary Education" entry).
## Used by split_into_sections() to stop a section from fragmenting
## into copies of itself whenever a sub-heading re-matches its own
## label or falls through to "unknown".
##_LIST_SECTION_LABELS = {
##    "skills", "education", "certifications", "achievements",
##    "languages", "projects", "interests", "strengths",
##}#
#_LIST_SECTION_LABELS = {
#    "skills", "education", "certifications", "achievements",
#    "languages", "projects", "interests", "strengths", "experience",
#}##
## A list of common section-related keywords used by Strategy B to check
## whether an unrecognised heading-shaped line is at least "about" one of
## the topics we know resumes cover.
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular", "strength",
#    "personal", "declaration",
#]##
#@dataclass
#class Section:
#    """Represents one identified section of a resume."""
#    label: str           # e.g. "experience", "skills"
#    raw_text: str        # the full text content of this section
#    start_line: int      # line number where this section starts
#    confidence: float    # 0.0 to 1.0, how sure we are this is the right label##
#def _clean_heading_candidate(line: str) -> str:
#    """
#    Strips common heading decoration (colons, dashes, surrounding
#    whitespace) before checking a line against our patterns, so
#    "EDUCATION:" or "- SKILLS -" still match plain "EDUCATION"/"SKILLS".
#    """
#    cleaned = line.strip()
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned##
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    """
#    FIX 2 (v3): Returns True if a line that matched a section pattern
#    is actually a content word wrapped onto its own line due to PDF
#    text extraction, NOT a real section heading.#
#    A wrapped word has ALL of these properties:
#      1. It is a single word (no spaces).
#      2. It is entirely lowercase.
#      3. The previous non-empty line ends mid-sentence (no . : ; ? !).
#    """
#    if " " in candidate:
#        return False
#    if candidate != candidate.lower():
#        return False
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#    return False##
#def split_inline_heading(line: str):
#    """
#    FIX 6 (v5): Some resumes place a heading and its first bullet(s) of
#    content on the SAME physical line, e.g.:#
#        "CORE COMPETENCIES • SQL • HDL Data Loader"#
#    detect_section_label() requires an exact, whole-line match by
#    design (see v1 fix), so a line like this never matches "skills" at
#    all and silently stays glued onto whatever section came before it.
#    This only runs as a FALLBACK, after detect_section_label() has
#    already failed to match the full line.#
#    Tries to find a known heading phrase at the START of the line,
#    followed by a separator (bullet, colon, dash) and remaining
#    content. Returns (heading, remaining_content) if found, or
#    (None, None) if this doesn't look like a merged heading line.
#    """
#    stripped = line.strip()#
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue#
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue#
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining#
#    return None, None##
#def normalize_inline_bullets(text: str) -> str:
#    """
#    FIX 7 (v5): Some resumes pack multiple skills onto one physical
#    bullet line, e.g. "• SQL • HDL Data Loader" (one bullet visually,
#    two skills). Splits any line containing more than one bullet
#    marker into one bullet per line.
#    """
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)##
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    """
#    Strategy A: Checks a line against all known section heading patterns.
#    Returns (label, confidence) or (None, 0.0) if no match.
#    """
#    if prev_lines is None:
#        prev_lines = []#
#    candidate = _clean_heading_candidate(line)#
#    if not candidate:
#        return None, 0.0#
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95#
#    return None, 0.0##
#def _contains_section_keyword(line: str) -> bool:
#    """
#    Checks whether a line contains any word commonly associated with
#    resume section headings.
#    """
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)##
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    """
#    Strategy B: Scores a line on how likely it is to be a section heading,
#    for cases where Strategy A didn't recognise the wording at all.
#    Returns a score from 0.0 (not a heading) to 1.0 (definitely a heading).
#    """
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0#
#    # FIX 11 (v7): a line starting with a bullet marker is a list item
#    # by definition, never a section heading -- regardless of what
#    # keywords it happens to contain. Confirmed on a real resume:
#    # "• Large Language Models(LLM)" and "• Natural Language Processing"
#    # both scored exactly 0.65 (just over threshold) purely because
#    # "language" is in _SECTION_KEYWORDS, fragmenting one skills block
#    # into multiple pieces at essentially random bullet lines.
#    if stripped[:1] in ("•", "●"):
#        return 0.0#
#    if line_index <= 1:
#        return 0.0#
#    word_count = len(stripped.split())#
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0#
#    if stripped.endswith("."):
#        return 0.0#
#    score = 0.0#
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3#
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1#
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")#
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15#
#    if stripped.endswith(":"):
#        score += 0.15#
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1#
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5#
#    return min(score, 1.0)##
#def split_into_sections(text: str) -> list:
#    """
#    Main entry point for Layer 1.#
#    Takes the full resume text (Layer 0's output) and returns a list of
#    Section objects -- each with a label, its text content, the line it
#    started on, and a confidence score.
#    """
#    lines = text.split("\n")
#    sections = []#
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9#
#    # FIX 3 (v4): duplicate-summary guard state.
#    summary_already_seen = False#
#    for i, line in enumerate(lines):
#        stripped = line.strip()#
#        # Pass previous lines so detect_section_label can apply
#        # the wrapped-word guard (FIX 2).
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])#
#        # FIX 6 (v5): if the whole line didn't match a heading on its
#        # own, check whether it's a heading MERGED with its first bit
#        # of content on the same physical line. Only tried as a
#        # fallback -- behaviour for lines that already match cleanly
#        # on their own is completely untouched.
#        inline_content = None
#        if label is None and stripped != "":
#            heading_candidate, remaining = split_inline_heading(stripped)
#            if heading_candidate is not None:
#                label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                if label is not None:
#                    inline_content = remaining#
#        if label is None and stripped != "":
#            # Apply the same wrapped-word guard to Strategy B.
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score#
#        # FIX 13 (v8): suppress false sub-section boundaries inside any
#        # "list-like" section (skills, education, certifications, etc.
#        # -- see _LIST_SECTION_LABELS). Sub-entries such as "Core
#        # Competencies" under a skills heading, or "Higher Secondary
#        # Education" under an education heading, often sit with no
#        # blank line separating them from the section they belong to,
#        # and because they're topically related words themselves, they
#        # trip either Strategy A directly (re-matching their own label)
#        # or Strategy B's keyword bonus (landing as "unknown"). Once
#        # we're already inside one of these section types, treat any
#        # further match of the SAME label or "unknown" as a sub-heading
#        # line -- keep it as content in the SAME section instead of
#        # starting a new one. Any other real label (e.g. "languages"
#        # appearing after "education") still closes the section
#        # normally -- this only prevents a section from fragmenting
#        # into copies of itself.
#        if current_label in _LIST_SECTION_LABELS and label in (current_label, "unknown"):
#            current_lines.append(line)
#            continue#
#        if label is not None and stripped != "":
#            # FIX 3 (v4): a second heading matching "summary" is never
#            # really a second summary in practice -- it's almost always
#            # a mislabeled start of the experience section. Scoped
#            # narrowly to "summary" only.
#            if label == "summary":
#                if summary_already_seen:
#                    label = "experience"
#                    confidence = 0.85
#                else:
#                    summary_already_seen = True#
#            # FIX 7 (v5): normalize multi-bullet lines before finalising
#            # the section that's about to be closed.
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            # FIX 6 (v5): seed the new section with whatever content
#            # was merged onto the same line as its heading.
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            current_lines.append(line)#
#    # FIX 7 (v5): normalize the final section too.
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))#
#    return sections##
#def get_section_text(sections: list, label: str) -> str:
#    """
#    Helper: returns the combined text of all sections with a given label.
#    """
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])#














#11r"""
#11Section Splitter — Layer 1 (Fixed Production Version, revision 6)
#11
#11Splits resume text into labelled sections using a cascade of strategies:
#11  Strategy A: Regex heading detection (handles the large majority of resumes)
#11  Strategy B: Line feature scoring (handles ambiguous/unrecognised headings)
#11
#11REVISION HISTORY (kept here so we don't repeat fixed mistakes):
#11
#11v1 bug (fixed): none of the original regex patterns had an end-anchor,
#11so they matched the START of a line and ignored everything after it.
#11Caused real sentences ("Experienced in partnering...") to be wrongly
#11treated as headings, and caused real headings with unexpected wording
#11("INDUSTRY EXPERIENCE") to be missed entirely. Fixed by anchoring every
#11pattern at both start (^) and end ($).
#11
#11v2 bug (fixed in revision 2): once patterns were strictly anchored,
#11they became too rigid -- a real resume used compound headings like
#11"CERTIFICATIONS & LICENCES", "KEY PROJECTS & RESEARCH", and
#11"ACHIEVEMENTS & LEADERSHIP" that don't exactly match any single phrase
#11we anticipated. Strategy B (the structural fallback) was ALSO supposed
#11to catch these, but its blank-line-based penalty wrongly punished them
#11too, since this particular resume has no blank lines between sections
#11at all. Confirmed by testing: all three of these missed headings scored
#110.36, well under our 0.65 threshold, purely due to that penalty.
#11
#11Fix applied in revision 2: SECTION_PATTERNS now allow each core heading
#11phrase to be optionally followed by a SHORT qualifier ("& <1-3 words>"
#11or "and <1-3 words>"), e.g. "Certifications" OR "Certifications &
#11Licences" both match. The qualifier length is deliberately capped at 3
#11words so this stays safe.
#11
#11v3 fixes (revision 3) — two separate bugs found on a real resume
#11(Aarti Suranje, sidebar-style layout):
#11
#11FIX 1 — "PERSONAL SUMMARY" not recognized as a summary section.
#11The summary pattern only allowed "professional" as a prefix
#11(r"(professional\s+)?summary"), so "PERSONAL SUMMARY" matched nothing
#11in Strategy A and fell through to Strategy B as "unknown". Added
#11"personal" as an additional allowed prefix so both "PERSONAL SUMMARY"
#11and "PROFESSIONAL SUMMARY" correctly label as "summary".
#11
#11FIX 2 — Single content word on its own line falsely triggers a section
#11break. In Aarti's resume, Layer 0 produced this (due to PDF text
#11wrapping in a sidebar layout):
#11
#11  Line 5: "Senior Oracle ERP consultant ... with over 15+ years of"
#11  Line 6: "experience"
#11  Line 7: "in enterprise transformations across finance, HCM..."
#11
#11Line 6 is the word "experience" wrapped onto its own line — it's the
#11continuation of the sentence on line 5, NOT a section heading. But
#11Strategy A's experience pattern matches it perfectly (it's just the
#11word "experience" alone) and splits the summary mid-sentence.
#11
#11The fix: in detect_section_label(), after Strategy A matches a line,
#11check whether (a) the matched line is a single lowercase word AND
#11(b) the previous non-empty line ends mid-sentence (no . : ; ? ! at the
#11end). If both are true, suppress the match — treat it as a wrapped
#11content word, not a heading.
#11
#11v4 fixes (revision 4) — three more bugs found while continuing to
#11debug Aarti Suranje's resume:
#11
#11FIX 3 — Duplicate/mislabeled "summary" heading swallowing the entire
#11experience section. A date-range-based "does this look like a job
#11entry" override was considered and REJECTED, because education
#11entries can also contain date ranges (e.g. "VESIT, Mumbai, 2016 -
#112020") and would have misfired the same way. Instead we track whether
#11a "summary" section has already been seen once; a second heading
#11matching "summary" is forced to "experience" instead. Scoped narrowly
#11to "summary" only.
#11
#11FIX 4 — "LANGUAGE SKILLS" heading not recognised. Fixed by allowing
#11"languages" to optionally take a "skills"/"known"/"proficiency"
#11suffix.
#11
#11FIX 5 — "CUSTOM SECTION" had no matching label at all, so it silently
#11merged into whatever section came before it. Added a new "other"
#11catch-all label for this and similar generic template headings.
#11
#11v5 fixes (revision 5) — heading merged with its first line of content
#11on the same physical line:
#11
#11FIX 6 — A section's raw_text ended up with content from the WRONG
#11section whenever a resume placed a heading and its first bullet(s) on
#11the same physical line, e.g. "CORE COMPETENCIES • SQL • HDL Data
#11Loader". We do NOT weaken _build_section_pattern() or the anchored
#11regex to fix this -- that regex is doing exactly what v1 was built to
#11enforce. Instead, split_inline_heading() runs as a FALLBACK only after
#11detect_section_label() has already failed to match the full line: it
#11looks for a known heading phrase at the start of the line followed by
#11a separator (bullet, colon, dash) and remaining content, splits them
#11apart, and re-runs detect_section_label() on just the heading part.
#11Generic across every heading in SECTION_PATTERNS and every common
#11separator, not hardcoded to one heading.
#11
#11FIX 7 — Multiple skills packed onto one bullet line, e.g.
#11"• SQL • HDL Data Loader". normalize_inline_bullets() splits any line
#11containing more than one bullet marker into one bullet per line.
#11
#11v6 fixes (this revision) — three missing heading patterns found on a
#11second real resume (Aasma, a different template):
#11
#11FIX 8 — "Professional Skills" heading not recognised. The skills
#11pattern only allowed technical/core/key as a prefix, not
#11"professional". Added "professional" as an additional allowed prefix.
#11
#11FIX 9 — "Strengths" heading had no pattern at all anywhere in
#11SECTION_PATTERNS, so her four strength bullets ("Quick learner and
#11adaptable", etc.) fell through to Strategy B and got dumped into the
#11same "unknown" block as her education lines above them, with no
#11boundary between the two. Added a new "strengths" label.
#11
#11FIX 10 — "Personal Details" and "Declaration" headings also had no
#11pattern anywhere, so that whole block (DOB, marital status, parents'
#11names, languages known, and the declaration statement) came out as
#11one generic "unknown" section with no internal structure. Added two
#11new labels, "personal_details" and "declaration", so these now split
#11into their own sections the same way "education" or "certifications"
#11already do.
#11"""
#11
#11import re
#11from dataclasses import dataclass
#11
#11
#11def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#11    """
#11    Builds a regex pattern for one section type, given a list of "core"
#11    phrase alternatives (e.g. ["certifications?", "licen[sc]es?"]).
#11
#11    The resulting pattern matches a line if it is EXACTLY one of those
#11    core phrases, OR one of those phrases followed by a short qualifier
#11    like " & Licences" or " and Research" (capped at 3 trailing words).
#11    This lets us recognise compound real-world headings (e.g.
#11    "Certifications & Licences") without having to predict every exact
#11    combination in advance, while still rejecting full sentences that
#11    merely start with a matching word.
#11    """
#11    core = "|".join(core_alternatives)
#11    return re.compile(
#11        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#11        re.IGNORECASE
#11    )
#11
#11
#11# ─────────────────────────────────────────────────────────────
#11# SECTION HEADING PATTERNS
#11# Each entry lists the CORE phrases for that section. _build_section_
#11# pattern() wraps them so the line must match start-to-end, optionally
#11# allowing one short " & qualifier" or " and qualifier" suffix.
#11# ─────────────────────────────────────────────────────────────
#11
#11SECTION_PATTERNS = {
#11    "summary": _build_section_pattern([
#11        r"(professional\s+|personal\s+)?summary",
#11        r"objective", r"profile",
#11        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#11    ]),
#11    "experience": _build_section_pattern([
#11        r"(work\s+|industry\s+|relevant\s+|professional\s+)?experience",
#11        r"employment(\s+history)?", r"professional\s+background",
#11        r"career\s+history", r"work\s+history", r"internships?",
#11    ]),
#11    #"education": _build_section_pattern([
#11    #    r"education(al)?(\s+background)?", r"academic(\s+background)?",
#11    #    r"academic\s+qualifications?", r"qualifications?",
#11    #    r"degrees?", r"university", r"college",
#11    #]),
#11
#11    "education": _build_section_pattern([
#11    r"education(al)?(\s+background)?", r"academic(\s+background)?",
#11    r"academic\s+qualifications?", r"educational\s+qualifications?",
#11    r"qualifications?",
#11    r"degrees?", r"university", r"college",
#11    ]),
#11
#11
#11
#11    "skills": _build_section_pattern([
#11        # FIX 8 (v6): added "professional\s+" as an additional allowed
#11        # prefix, so "Professional Skills" is recognised the same way
#11        # "Technical Skills" / "Core Skills" / "Key Skills" already were.
#11        r"(technical\s+|core\s+|key\s+|professional\s+)?(skills?|competenc(y|ies))",
#11        r"expertise", r"technologies",
#11        r"programming\s+(languages?|skills?)",
#11        r"tools?\s*(&|and)\s*technologies",
#11        r"core\s+competencies\s*(&|and)\s*technical\s+skills",
#11    ]),
#11    "projects": _build_section_pattern([
#11        r"(personal\s+|side\s+|key\s+|notable\s+|academic\s+)?projects?",
#11        r"portfolio",
#11    ]),
#11    "certifications": _build_section_pattern([
#11        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#11        r"credentials?", r"professional\s+certifications?",
#11    ]),
#11    "achievements": _build_section_pattern([
#11        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#11        r"accomplishments?", r"accolades?",
#11    ]),
#11    "languages": _build_section_pattern([
#11        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#11        r"language\s+skills",
#11    ]),
#11    "interests": _build_section_pattern([
#11        r"interests?", r"hobbies", r"activities",
#11    ]),
#11    "other": _build_section_pattern([
#11        r"custom\s+section", r"additional\s+information",
#11        r"miscellaneous", r"additional\s+details",
#11    ]),
#11    # FIX 9 (v6): "Strengths" had no pattern anywhere, so it fell
#11    # through to Strategy B and merged into whatever "unknown" block
#11    # came before it instead of getting its own section.
#11    "strengths": _build_section_pattern([
#11        r"(key\s+)?strengths?", r"core\s+strengths?",
#11    ]),
#11    # FIX 10 (v6): "Personal Details" and "Declaration" had no pattern
#11    # anywhere either, so DOB/marital status/parents' names/declaration
#11    # text all came out as one undifferentiated "unknown" block.
#11    "personal_details": _build_section_pattern([
#11        r"personal\s+(details|information|profile|data)",
#11        r"bio\s*-?\s*data",
#11    ]),
#11    "declaration": _build_section_pattern([
#11        r"declaration", r"self[\s-]?declaration",
#11    ]),
#11}
#11
#11
#11# A list of common section-related keywords used by Strategy B to check
#11# whether an unrecognised heading-shaped line is at least "about" one of
#11# the topics we know resumes cover.
#11_SECTION_KEYWORDS = [
#11    "skill", "experience", "education", "project", "certif", "licen",
#11    "achievement", "award", "honor", "honour", "summary", "objective",
#11    "qualification", "employment", "career", "academic", "competenc",
#11    "expertise", "technolog", "portfolio", "credential", "accreditation",
#11    "recognition", "accomplishment", "language", "interest", "hobbies",
#11    "research", "leadership", "internship", "training", "volunteer",
#11    "publication", "reference", "extracurricular", "strength",
#11    "personal", "declaration",
#11]
#11
#11
#11@dataclass
#11class Section:
#11    """Represents one identified section of a resume."""
#11    label: str           # e.g. "experience", "skills"
#11    raw_text: str        # the full text content of this section
#11    start_line: int      # line number where this section starts
#11    confidence: float    # 0.0 to 1.0, how sure we are this is the right label
#11
#11
#11def _clean_heading_candidate(line: str) -> str:
#11    """
#11    Strips common heading decoration (colons, dashes, surrounding
#11    whitespace) before checking a line against our patterns, so
#11    "EDUCATION:" or "- SKILLS -" still match plain "EDUCATION"/"SKILLS".
#11    """
#11    cleaned = line.strip()
#11    cleaned = cleaned.strip(":-—–_ ")
#11    return cleaned
#11
#11
#11def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#11    """
#11    FIX 2 (v3): Returns True if a line that matched a section pattern
#11    is actually a content word wrapped onto its own line due to PDF
#11    text extraction, NOT a real section heading.
#11
#11    A wrapped word has ALL of these properties:
#11      1. It is a single word (no spaces).
#11      2. It is entirely lowercase — real headings are always at least
#11         title-case or all-caps; a bare lowercase word is a content
#11         word that happened to land alone on a line.
#11      3. The previous non-empty line ends mid-sentence — it doesn't end
#11         with sentence-closing punctuation (. : ; ? !), meaning the
#11         previous line's text was still ongoing and this line is its
#11         continuation, not a new section.
#11    """
#11    # Condition 1: single word only
#11    if " " in candidate:
#11        return False
#11
#11    # Condition 2: entirely lowercase (no capitalisation at all)
#11    if candidate != candidate.lower():
#11        return False
#11
#11    # Condition 3: previous non-empty line ends mid-sentence
#11    for prev in reversed(prev_lines):
#11        prev_stripped = prev.strip()
#11        if prev_stripped:
#11            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#11
#11    return False
#11
#11
#11def split_inline_heading(line: str):
#11    """
#11    FIX 6 (v5): Some resumes place a heading and its first bullet(s) of
#11    content on the SAME physical line, e.g.:
#11
#11        "CORE COMPETENCIES • SQL • HDL Data Loader"
#11
#11    detect_section_label() requires an exact, whole-line match by
#11    design (see v1 fix), so a line like this never matches "skills" at
#11    all and silently stays glued onto whatever section came before it.
#11    This only runs as a FALLBACK, after detect_section_label() has
#11    already failed to match the full line -- so it can never interfere
#11    with a line that already matches cleanly on its own.
#11
#11    Tries to find a known heading phrase at the START of the line,
#11    followed by a separator (bullet, colon, dash) and remaining
#11    content. Returns (heading, remaining_content) if found, or
#11    (None, None) if this doesn't look like a merged heading line.
#11    """
#11    stripped = line.strip()
#11
#11    for separator in ["•", "●", ":", "-", "–", "—"]:
#11        if separator not in stripped:
#11            continue
#11
#11        heading_candidate = stripped.split(separator, 1)[0].strip()
#11        if not heading_candidate:
#11            continue
#11
#11        for label, pattern in SECTION_PATTERNS.items():
#11            if pattern.match(heading_candidate):
#11                remaining = stripped[len(heading_candidate):].strip()
#11                remaining = remaining.lstrip("•●:-–— ").strip()
#11                return heading_candidate, remaining
#11
#11    return None, None
#11
#11
#11def normalize_inline_bullets(text: str) -> str:
#11    """
#11    FIX 7 (v5): Some resumes pack multiple skills onto one physical
#11    bullet line, e.g. "• SQL • HDL Data Loader" (one bullet visually,
#11    two skills). Splits any line containing more than one bullet
#11    marker into one bullet per line, so Layer 2's skills extractor
#11    sees each skill as its own token instead of one run-on line.
#11    """
#11    normalized = []
#11    for raw_line in text.splitlines():
#11        bullet_count = raw_line.count("•") + raw_line.count("●")
#11        if bullet_count <= 1:
#11            normalized.append(raw_line)
#11            continue
#11        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#11        for piece in pieces:
#11            normalized.append(f"• {piece}")
#11    return "\n".join(normalized)
#11
#11
#11def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#11    """
#11    Strategy A: Checks a line against all known section heading patterns.
#11    Returns (label, confidence) or (None, 0.0) if no match.
#11
#11    prev_lines: the lines seen before this one in the document, used
#11    by the wrapped-word guard (FIX 2) to suppress false positives where
#11    a content word wraps onto its own line and happens to match a pattern.
#11    """
#11    if prev_lines is None:
#11        prev_lines = []
#11
#11    candidate = _clean_heading_candidate(line)
#11
#11    if not candidate:
#11        return None, 0.0
#11
#11    for label, pattern in SECTION_PATTERNS.items():
#11        if pattern.match(candidate):
#11            # FIX 2 guard: suppress if this looks like a wrapped content
#11            # word rather than a genuine heading.
#11            if _is_wrapped_word(candidate, prev_lines):
#11                return None, 0.0
#11            return label, 0.95
#11
#11    return None, 0.0
#11
#11
#11def _contains_section_keyword(line: str) -> bool:
#11    """
#11    Checks whether a line contains any word commonly associated with
#11    resume section headings.
#11    """
#11    lower = line.lower()
#11    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#11
#11
#11def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#11    """
#11    Strategy B: Scores a line on how likely it is to be a section heading,
#11    for cases where Strategy A didn't recognise the wording at all.
#11    Returns a score from 0.0 (not a heading) to 1.0 (definitely a heading).
#11    """
#11    stripped = line.strip()
#11    if len(stripped) == 0:
#11        return 0.0
#11
#11    if line_index <= 1:
#11        return 0.0
#11
#11    word_count = len(stripped.split())
#11
#11    if word_count > 8 or len(stripped) > 60:
#11        return 0.0
#11
#11    if stripped.endswith("."):
#11        return 0.0
#11
#11    score = 0.0
#11
#11    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#11    if is_all_caps:
#11        score += 0.3
#11
#11    if word_count <= 4:
#11        score += 0.2
#11    elif word_count <= 6:
#11        score += 0.1
#11
#11    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#11    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#11
#11    if blank_below:
#11        score += 0.2
#11    if blank_above:
#11        score += 0.15
#11
#11    if stripped.endswith(":"):
#11        score += 0.15
#11
#11    if "," not in stripped and ". " not in stripped:
#11        score += 0.1
#11
#11    if _contains_section_keyword(stripped):
#11        score += 0.35
#11    else:
#11        if not blank_below and not blank_above:
#11            score *= 0.5
#11
#11    return min(score, 1.0)
#11
#11
#11def split_into_sections(text: str) -> list:
#11    """
#11    Main entry point for Layer 1.
#11
#11    Takes the full resume text (Layer 0's output) and returns a list of
#11    Section objects -- each with a label, its text content, the line it
#11    started on, and a confidence score.
#11    """
#11    lines = text.split("\n")
#11    sections = []
#11
#11    current_label = "header"
#11    current_start = 0
#11    current_lines = []
#11    current_confidence = 0.9
#11
#11    # FIX 3 (v4): duplicate-summary guard state.
#11    summary_already_seen = False
#11
#11    for i, line in enumerate(lines):
#11        stripped = line.strip()
#11
#11        # Pass previous lines so detect_section_label can apply
#11        # the wrapped-word guard (FIX 2).
#11        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#11
#11        # FIX 6 (v5): if the whole line didn't match a heading on its
#11        # own, check whether it's a heading MERGED with its first bit
#11        # of content on the same physical line. Only tried as a
#11        # fallback -- behaviour for lines that already match cleanly
#11        # on their own is completely untouched.
#11        inline_content = None
#11        if label is None and stripped != "":
#11            heading_candidate, remaining = split_inline_heading(stripped)
#11            if heading_candidate is not None:
#11                label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#11                if label is not None:
#11                    inline_content = remaining
#11
#11        if label is None and stripped != "":
#11            # Apply the same wrapped-word guard to Strategy B: if the
#11            # line is a single lowercase word preceded by a mid-sentence
#11            # line, don't let Strategy B score it as a heading either --
#11            # it's a content word, not a section boundary.
#11            if not _is_wrapped_word(stripped, lines[:i]):
#11                heading_score = score_heading_line(stripped, i, lines)
#11                if heading_score >= 0.65:
#11                    label = "unknown"
#11                    confidence = heading_score
#11
#11        if label is not None and stripped != "":
#11            # FIX 3 (v4): a second heading matching "summary" is never
#11            # really a second summary in practice -- it's almost always
#11            # a mislabeled start of the experience section. Scoped
#11            # narrowly to "summary" only; every other label is left
#11            # exactly as matched.
#11            if label == "summary":
#11                if summary_already_seen:
#11                    label = "experience"
#11                    confidence = 0.85
#11                else:
#11                    summary_already_seen = True
#11
#11            # FIX 7 (v5): normalize multi-bullet lines before finalising
#11            # the section that's about to be closed.
#11            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#11            if section_text:
#11                sections.append(Section(
#11                    label=current_label,
#11                    raw_text=section_text,
#11                    start_line=current_start,
#11                    confidence=current_confidence
#11                ))
#11            current_label = label
#11            current_start = i
#11            current_lines = []
#11            # FIX 6 (v5): seed the new section with whatever content
#11            # was merged onto the same line as its heading.
#11            if inline_content:
#11                current_lines.append(inline_content)
#11            current_confidence = confidence
#11        else:
#11            current_lines.append(line)
#11
#11    # FIX 7 (v5): normalize the final section too.
#11    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#11    if section_text:
#11        sections.append(Section(
#11            label=current_label,
#11            raw_text=section_text,
#11            start_line=current_start,
#11            confidence=current_confidence
#11        ))
#11
#11    return sections
#11
#11
#11def get_section_text(sections: list, label: str) -> str:
#11    """
#11    Helper: returns the combined text of all sections with a given label.
#11    """
#11    matching = [s for s in sections if s.label == label]
#11    if not matching:
#11        return ""
#11    return "\n\n".join([s.raw_text for s in matching])







##worked on aarti resume
#r"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 5)
#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Line feature scoring (handles ambiguous/unrecognised headings)
#
#REVISION HISTORY (kept here so we don't repeat fixed mistakes):
#
#v1 bug (fixed): none of the original regex patterns had an end-anchor,
#so they matched the START of a line and ignored everything after it.
#Caused real sentences ("Experienced in partnering...") to be wrongly
#treated as headings, and caused real headings with unexpected wording
#("INDUSTRY EXPERIENCE") to be missed entirely. Fixed by anchoring every
#pattern at both start (^) and end ($).
#
#v2 bug (fixed in revision 2): once patterns were strictly anchored,
#they became too rigid -- a real resume used compound headings like
#"CERTIFICATIONS & LICENCES", "KEY PROJECTS & RESEARCH", and
#"ACHIEVEMENTS & LEADERSHIP" that don't exactly match any single phrase
#we anticipated. Strategy B (the structural fallback) was ALSO supposed
#to catch these, but its blank-line-based penalty wrongly punished them
#too, since this particular resume has no blank lines between sections
#at all. Confirmed by testing: all three of these missed headings scored
#0.36, well under our 0.65 threshold, purely due to that penalty.
#
#Fix applied in revision 2: SECTION_PATTERNS now allow each core heading
#phrase to be optionally followed by a SHORT qualifier ("& <1-3 words>"
#or "and <1-3 words>"), e.g. "Certifications" OR "Certifications &
#Licences" both match. The qualifier length is deliberately capped at 3
#words so this stays safe.
#
#v3 fixes (revision 3) — two separate bugs found on a real resume
#(Aarti Suranje, sidebar-style layout):
#
#FIX 1 — "PERSONAL SUMMARY" not recognized as a summary section.
#The summary pattern only allowed "professional" as a prefix
#(r"(professional\s+)?summary"), so "PERSONAL SUMMARY" matched nothing
#in Strategy A and fell through to Strategy B as "unknown". Added
#"personal" as an additional allowed prefix so both "PERSONAL SUMMARY"
#and "PROFESSIONAL SUMMARY" correctly label as "summary".
#
#FIX 2 — Single content word on its own line falsely triggers a section
#break. In Aarti's resume, Layer 0 produced this (due to PDF text
#wrapping in a sidebar layout):
#
#  Line 5: "Senior Oracle ERP consultant ... with over 15+ years of"
#  Line 6: "experience"
#  Line 7: "in enterprise transformations across finance, HCM..."
#
#Line 6 is the word "experience" wrapped onto its own line — it's the
#continuation of the sentence on line 5, NOT a section heading. But
#Strategy A's experience pattern matches it perfectly (it's just the
#word "experience" alone) and splits the summary mid-sentence, placing
#lines 7+ into a new "experience" section while "PRINCIPAL CONSULTANT..."
#(the actual job history, which arrives later) ends up in "summary".
#
#The fix: in detect_section_label(), after Strategy A matches a line,
#check whether (a) the matched line is a single lowercase word AND
#(b) the previous non-empty line ends mid-sentence (no . : ; ? ! at the
#end). If both are true, suppress the match — treat it as a wrapped
#content word, not a heading. A real section heading is never a
#single lowercase word with no capitalisation at all.
#
#This guard is tight enough to only suppress genuine false positives:
#a real heading like "EXPERIENCE" is uppercase, "Experience" is
#title-case — neither is all-lowercase. And a wrapped word like
#"experience" at the end of a sentence continuation is always preceded
#by a line that doesn't end with sentence-ending punctuation.
#
#v4 fixes (revision 4) — three more bugs found while continuing to
#debug Aarti Suranje's resume:
#
#FIX 3 — Duplicate/mislabeled "summary" heading swallowing the entire
#experience section. Aarti's resume has TWO headings that match the
#summary pattern: the real "PERSONAL SUMMARY" near the top, and then a
#second "PROFESSIONAL SUMMARY" heading sitting directly above her job
#history. A date-range-based "does this look like a job entry" override
#was considered and REJECTED, because education entries can also
#contain date ranges (e.g. "VESIT, Mumbai, 2016 - 2020") and would have
#misfired the same way. Instead we track whether a "summary" section
#has already been seen once; a second heading matching "summary" is
#forced to "experience" instead. This is scoped narrowly to "summary"
#only and never touches any other label.
#
#FIX 4 — "LANGUAGE SKILLS" heading not recognised (skills pattern only
#allowed technical/core/key prefixes, languages pattern didn't allow a
#"skills" suffix). Fixed by allowing "languages" to optionally take a
#"skills"/"known"/"proficiency" suffix.
#
#FIX 5 — "CUSTOM SECTION" (a common resume-builder-template heading)
#had no matching label at all, so it silently merged into whatever
#section came before it. Added a new "other" catch-all label for this
#and similar generic template headings.
#
#v5 fixes (this revision) — heading merged with its first line of
#content on the same physical line:
#
#FIX 6 — A section's raw_text was still ending up with content from
#the WRONG section whenever a resume placed a heading and its first
#bullet(s) on the same physical line, e.g.:
#
#    "CORE COMPETENCIES • SQL • HDL Data Loader"
#
#detect_section_label() requires an exact, whole-line match by design
#(this is intentional -- see the v1 fix, which exists specifically to
#stop partial-line matches from misfiring on ordinary sentences). So a
#merged line like this never matches "skills" at all, and silently
#stays glued onto whatever section came before it (in Aarti's case, it
#got swallowed into "experience" once FIX 3 correctly relabelled that
#block). Confirmed on her real resume.
#
#We do NOT weaken _build_section_pattern() or the anchored regex to fix
#this -- that regex is doing exactly what v1 was built to enforce.
#Instead, split_inline_heading() runs as a FALLBACK only after
#detect_section_label() has already failed to match the full line: it
#looks for a known heading phrase at the start of the line followed by
#a separator (bullet, colon, dash) and remaining content, splits them
#apart, and re-runs detect_section_label() on just the heading part.
#The leftover content becomes the first line of the new section instead
#of being lost. This is generic across every heading in
#SECTION_PATTERNS and every common separator, not hardcoded to
#"CORE COMPETENCIES" alone -- so it also handles "TECHNICAL SKILLS:
#Python, SQL", "SKILLS - Python, SQL", etc. with the same one rule.
#
#FIX 7 — Multiple skills packed onto one bullet line, e.g.
#"• SQL • HDL Data Loader" (one bullet character visually next to the
#heading, but two distinct skills separated by a second bullet
#mid-line). normalize_inline_bullets() splits any line containing more
#than one bullet marker into one bullet per line, applied once a
#section's raw_text is finalised, so Layer 2's skills extractor sees
#each skill as its own token instead of one run-on line.
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    """
#    Builds a regex pattern for one section type, given a list of "core"
#    phrase alternatives (e.g. ["certifications?", "licen[sc]es?"]).
#
#    The resulting pattern matches a line if it is EXACTLY one of those
#    core phrases, OR one of those phrases followed by a short qualifier
#    like " & Licences" or " and Research" (capped at 3 trailing words).
#    This lets us recognise compound real-world headings (e.g.
#    "Certifications & Licences") without having to predict every exact
#    combination in advance, while still rejecting full sentences that
#    merely start with a matching word.
#    """
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
## ─────────────────────────────────────────────────────────────
## SECTION HEADING PATTERNS
## Each entry lists the CORE phrases for that section. _build_section_
## pattern() wraps them so the line must match start-to-end, optionally
## allowing one short " & qualifier" or " and qualifier" suffix.
## ─────────────────────────────────────────────────────────────
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        # FIX 1 (v3): added "personal\s+" as an additional allowed prefix
#        # so "PERSONAL SUMMARY" is recognised the same way "PROFESSIONAL
#        # SUMMARY" already was.
#        r"(professional\s+|personal\s+)?summary",
#        r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"(work\s+|industry\s+|relevant\s+|professional\s+)?experience",
#        r"employment(\s+history)?", r"professional\s+background",
#        r"career\s+history", r"work\s+history", r"internships?",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?", r"academic(\s+background)?",
#        r"academic\s+qualifications?", r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _build_section_pattern([
#        r"(technical\s+|core\s+|key\s+)?(skills?|competenc(y|ies))",
#        r"expertise", r"technologies",
#        r"programming\s+(languages?|skills?)",
#        r"tools?\s*(&|and)\s*technologies",
#        r"core\s+competencies\s*(&|and)\s*technical\s+skills",
#    ]),
#    "projects": _build_section_pattern([
#        r"(personal\s+|side\s+|key\s+|notable\s+|academic\s+)?projects?",
#        r"portfolio",
#    ]),
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#    ]),
#    "languages": _build_section_pattern([
#        # FIX 4 (v4): allow an optional "skills"/"known"/"proficiency"
#        # suffix so "LANGUAGE SKILLS" / "Languages Known" / "Language
#        # Proficiency" are all recognised, not just the bare word
#        # "Languages".
#        r"languages?(\s+(skills|known|proficiency))?", r"spoken\s+languages?",
#        r"language\s+skills",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#    # FIX 5 (v4): catch-all for resume-builder-template headings that
#    # don't map to a specific known field (e.g. "Custom Section",
#    # "Additional Information"). We don't know what's inside without
#    # reading it, so we give it its own label rather than letting it
#    # silently merge into whichever section happened to come before it.
#    "other": _build_section_pattern([
#        r"custom\s+section", r"additional\s+information",
#        r"miscellaneous", r"additional\s+details",
#    ]),
#}
#
#
## A list of common section-related keywords used by Strategy B to check
## whether an unrecognised heading-shaped line is at least "about" one of
## the topics we know resumes cover.
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular",
#]
#
#
#@dataclass
#class Section:
#    """Represents one identified section of a resume."""
#    label: str           # e.g. "experience", "skills"
#    raw_text: str        # the full text content of this section
#    start_line: int      # line number where this section starts
#    confidence: float    # 0.0 to 1.0, how sure we are this is the right label
#
#
#def _clean_heading_candidate(line: str) -> str:
#    """
#    Strips common heading decoration (colons, dashes, surrounding
#    whitespace) before checking a line against our patterns, so
#    "EDUCATION:" or "- SKILLS -" still match plain "EDUCATION"/"SKILLS".
#    """
#    cleaned = line.strip()
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _is_wrapped_word(candidate: str, prev_lines: list) -> bool:
#    """
#    FIX 2 (v3): Returns True if a line that matched a section pattern
#    is actually a content word wrapped onto its own line due to PDF
#    text extraction, NOT a real section heading.
#
#    A wrapped word has ALL of these properties:
#      1. It is a single word (no spaces).
#      2. It is entirely lowercase — real headings are always at least
#         title-case or all-caps; a bare lowercase word is a content
#         word that happened to land alone on a line.
#      3. The previous non-empty line ends mid-sentence — it doesn't end
#         with sentence-closing punctuation (. : ; ? !), meaning the
#         previous line's text was still ongoing and this line is its
#         continuation, not a new section.
#
#    Confirmed on a real resume: Layer 0 extracted "experience" alone on
#    line 6, preceded by "...with over 15+ years of" on line 5. All three
#    conditions are true → correctly suppressed as a false positive.
#
#    A real section heading ("EXPERIENCE", "Experience", "experience:")
#    fails at least one of these checks:
#    - "EXPERIENCE" → not all-lowercase (condition 2 fails)
#    - "Experience" → not all-lowercase (condition 2 fails)
#    - "experience:" → not a single clean alpha word, has colon stripped
#      already by _clean_heading_candidate but would still not be purely
#      alphabetic before cleaning — and in practice headings almost always
#      have some capitalisation anyway.
#    """
#    # Condition 1: single word only
#    if " " in candidate:
#        return False
#
#    # Condition 2: entirely lowercase (no capitalisation at all)
#    if candidate != candidate.lower():
#        return False
#
#    # Condition 3: previous non-empty line ends mid-sentence
#    for prev in reversed(prev_lines):
#        prev_stripped = prev.strip()
#        if prev_stripped:
#            return prev_stripped[-1] not in {'.', ':', ';', '?', '!'}
#
#    return False
#
#
#def split_inline_heading(line: str):
#    """
#    FIX 6 (v5): Some resumes place a heading and its first bullet(s) of
#    content on the SAME physical line, e.g.:
#
#        "CORE COMPETENCIES • SQL • HDL Data Loader"
#
#    detect_section_label() requires an exact, whole-line match by
#    design (see v1 fix), so a line like this never matches "skills" at
#    all and silently stays glued onto whatever section came before it.
#    This only runs as a FALLBACK, after detect_section_label() has
#    already failed to match the full line -- so it can never interfere
#    with a line that already matches cleanly on its own. All v1-v4
#    behaviour is unchanged.
#
#    Tries to find a known heading phrase at the START of the line,
#    followed by a separator (bullet, colon, dash) and remaining
#    content. Returns (heading, remaining_content) if found, or
#    (None, None) if this doesn't look like a merged heading line.
#    """
#    stripped = line.strip()
#
#    for separator in ["•", "●", ":", "-", "–", "—"]:
#        if separator not in stripped:
#            continue
#
#        heading_candidate = stripped.split(separator, 1)[0].strip()
#        if not heading_candidate:
#            continue
#
#        for label, pattern in SECTION_PATTERNS.items():
#            if pattern.match(heading_candidate):
#                remaining = stripped[len(heading_candidate):].strip()
#                remaining = remaining.lstrip("•●:-–— ").strip()
#                return heading_candidate, remaining
#
#    return None, None
#
#
#def normalize_inline_bullets(text: str) -> str:
#    """
#    FIX 7 (v5): Some resumes pack multiple skills onto one physical
#    bullet line, e.g. "• SQL • HDL Data Loader" (one bullet visually,
#    two skills). Splits any line containing more than one bullet
#    marker into one bullet per line, so Layer 2's skills extractor
#    sees each skill as its own token instead of one run-on line.
#    """
#    normalized = []
#    for raw_line in text.splitlines():
#        bullet_count = raw_line.count("•") + raw_line.count("●")
#        if bullet_count <= 1:
#            normalized.append(raw_line)
#            continue
#        pieces = [p.strip() for p in re.split(r"[•●]", raw_line) if p.strip()]
#        for piece in pieces:
#            normalized.append(f"• {piece}")
#    return "\n".join(normalized)
#
#
#def detect_section_label(line: str, prev_lines: list = None) -> tuple:
#    """
#    Strategy A: Checks a line against all known section heading patterns.
#    Returns (label, confidence) or (None, 0.0) if no match.
#
#    prev_lines: the lines seen before this one in the document, used
#    by the wrapped-word guard (FIX 2) to suppress false positives where
#    a content word wraps onto its own line and happens to match a pattern.
#    """
#    if prev_lines is None:
#        prev_lines = []
#
#    candidate = _clean_heading_candidate(line)
#
#    if not candidate:
#        return None, 0.0
#
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            # FIX 2 guard: suppress if this looks like a wrapped content
#            # word rather than a genuine heading.
#            if _is_wrapped_word(candidate, prev_lines):
#                return None, 0.0
#            return label, 0.95
#
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    """
#    Checks whether a line contains any word commonly associated with
#    resume section headings.
#    """
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    """
#    Strategy B: Scores a line on how likely it is to be a section heading,
#    for cases where Strategy A didn't recognise the wording at all.
#    Returns a score from 0.0 (not a heading) to 1.0 (definitely a heading).
#    """
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#
#    if line_index <= 1:
#        return 0.0
#
#    word_count = len(stripped.split())
#
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#
#    if stripped.endswith("."):
#        return 0.0
#
#    score = 0.0
#
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#
#    if stripped.endswith(":"):
#        score += 0.15
#
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    """
#    Main entry point for Layer 1.
#
#    Takes the full resume text (Layer 0's output) and returns a list of
#    Section objects -- each with a label, its text content, the line it
#    started on, and a confidence score.
#    """
#    lines = text.split("\n")
#    sections = []
#
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#
#    # FIX 3 (v4): duplicate-summary guard state. See module docstring
#    # for why this is scoped narrowly to "summary" only, instead of a
#    # more general (and riskier) date-range-based job-entry detector.
#    summary_already_seen = False
#
#    for i, line in enumerate(lines):
#        stripped = line.strip()
#
#        # Pass previous lines so detect_section_label can apply
#        # the wrapped-word guard (FIX 2).
#        label, confidence = detect_section_label(stripped, prev_lines=lines[:i])
#
#        # FIX 6 (v5): if the whole line didn't match a heading on its
#        # own, check whether it's a heading MERGED with its first bit
#        # of content on the same physical line. Only tried as a
#        # fallback -- v1-v4 behaviour for lines that already match
#        # cleanly on their own is completely untouched.
#        inline_content = None
#        if label is None and stripped != "":
#            heading_candidate, remaining = split_inline_heading(stripped)
#            if heading_candidate is not None:
#                label, confidence = detect_section_label(heading_candidate, prev_lines=lines[:i])
#                if label is not None:
#                    inline_content = remaining
#
#        if label is None and stripped != "":
#            # Apply the same wrapped-word guard to Strategy B: if the
#            # line is a single lowercase word preceded by a mid-sentence
#            # line, don't let Strategy B score it as a heading either --
#            # it's a content word, not a section boundary.
#            if not _is_wrapped_word(stripped, lines[:i]):
#                heading_score = score_heading_line(stripped, i, lines)
#                if heading_score >= 0.65:
#                    label = "unknown"
#                    confidence = heading_score
#
#        if label is not None and stripped != "":
#            # FIX 3 (v4): a second heading matching "summary" is never
#            # really a second summary in practice -- it's almost always
#            # a mislabeled start of the experience section. Scoped
#            # narrowly to "summary" only; every other label is left
#            # exactly as matched.
#            if label == "summary":
#                if summary_already_seen:
#                    label = "experience"
#                    confidence = 0.85
#                else:
#                    summary_already_seen = True
#
#            # FIX 7 (v5): normalize multi-bullet lines before finalising
#            # the section that's about to be closed.
#            section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            # FIX 6 (v5): seed the new section with whatever content
#            # was merged onto the same line as its heading.
#            if inline_content:
#                current_lines.append(inline_content)
#            current_confidence = confidence
#        else:
#            current_lines.append(line)
#
#    # FIX 7 (v5): normalize the final section too.
#    section_text = normalize_inline_bullets("\n".join(current_lines).strip())
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    """
#    Helper: returns the combined text of all sections with a given label.
#    """
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#















#worked changing for new resume o/p's
#"""
#Section Splitter — Layer 1 (Fixed Production Version, revision 2)
#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Line feature scoring (handles ambiguous/unrecognised headings)
#
#REVISION HISTORY (kept here so we don't repeat fixed mistakes):
#
#v1 bug (fixed): none of the original regex patterns had an end-anchor,
#so they matched the START of a line and ignored everything after it.
#Caused real sentences ("Experienced in partnering...") to be wrongly
#treated as headings, and caused real headings with unexpected wording
#("INDUSTRY EXPERIENCE") to be missed entirely. Fixed by anchoring every
#pattern at both start (^) and end ($).
#
#v2 bug (fixed in this revision): once patterns were strictly anchored,
#they became too rigid -- a real resume used compound headings like
#"CERTIFICATIONS & LICENCES", "KEY PROJECTS & RESEARCH", and
#"ACHIEVEMENTS & LEADERSHIP" that don't exactly match any single phrase
#we anticipated. Strategy B (the structural fallback) was ALSO supposed
#to catch these, but its blank-line-based penalty wrongly punished them
#too, since this particular resume has no blank lines between sections
#at all. Confirmed by testing: all three of these missed headings scored
#0.36, well under our 0.65 threshold, purely due to that penalty.
#
#Fix applied here: SECTION_PATTERNS now allow each core heading phrase to
#be optionally followed by a SHORT qualifier ("& <1-3 words>" or
#"and <1-3 words>"), e.g. "Certifications" OR "Certifications & Licences"
#both match. The qualifier length is deliberately capped at 3 words so
#this stays safe -- a full sentence that happens to start with a heading
#word (e.g. "Achievements and awards are listed in detail below for
#review purposes") still does NOT match, since 3+ words follow "and".
#This was verified directly: see the test suite run after this file was
#built, covering both real headings (must match) and risky full
#sentences (must NOT match).
#"""
#
#import re
#from dataclasses import dataclass
#
#
#def _build_section_pattern(core_alternatives: list) -> "re.Pattern":
#    """
#    Builds a regex pattern for one section type, given a list of "core"
#    phrase alternatives (e.g. ["certifications?", "licen[sc]es?"]).
#
#    The resulting pattern matches a line if it is EXACTLY one of those
#    core phrases, OR one of those phrases followed by a short qualifier
#    like " & Licences" or " and Research" (capped at 3 trailing words).
#    This lets us recognise compound real-world headings (e.g.
#    "Certifications & Licences") without having to predict every exact
#    combination in advance, while still rejecting full sentences that
#    merely start with a matching word.
#    """
#    core = "|".join(core_alternatives)
#    return re.compile(
#        rf"^(?:{core})(\s*(&|and)\s*[a-z]+(\s+[a-z]+){{0,2}})?$",
#        re.IGNORECASE
#    )
#
#
## ─────────────────────────────────────────────────────────────
## SECTION HEADING PATTERNS
## Each entry lists the CORE phrases for that section. _build_section_
## pattern() wraps them so the line must match start-to-end, optionally
## allowing one short " & qualifier" or " and qualifier" suffix.
## ─────────────────────────────────────────────────────────────
#
#SECTION_PATTERNS = {
#    "summary": _build_section_pattern([
#        r"(professional\s+)?summary", r"objective", r"profile",
#        r"about\s+me", r"career\s+objective", r"personal\s+statement",
#    ]),
#    "experience": _build_section_pattern([
#        r"(work\s+|industry\s+|relevant\s+|professional\s+)?experience",
#        r"employment(\s+history)?", r"professional\s+background",
#        r"career\s+history", r"work\s+history", r"internships?",
#    ]),
#    "education": _build_section_pattern([
#        r"education(al)?(\s+background)?", r"academic(\s+background)?",
#        r"academic\s+qualifications?", r"qualifications?",
#        r"degrees?", r"university", r"college",
#    ]),
#    "skills": _build_section_pattern([
#        r"(technical\s+|core\s+|key\s+)?(skills?|competenc(y|ies))",
#        r"expertise", r"technologies",
#        r"programming\s+(languages?|skills?)",
#        r"tools?\s*(&|and)\s*technologies",
#        r"core\s+competencies\s*(&|and)\s*technical\s+skills",
#    ]),
#    "projects": _build_section_pattern([
#        r"(personal\s+|side\s+|key\s+|notable\s+|academic\s+)?projects?",
#        r"portfolio",
#    ]),
#    "certifications": _build_section_pattern([
#        r"certif(ication|icate)s?", r"licen[sc]es?", r"accreditations?",
#        r"credentials?", r"professional\s+certifications?",
#    ]),
#    "achievements": _build_section_pattern([
#        r"achievements?", r"awards?", r"honou?rs?", r"recognitions?",
#        r"accomplishments?", r"accolades?",
#    ]),
#    "languages": _build_section_pattern([
#        r"languages?", r"spoken\s+languages?",
#    ]),
#    "interests": _build_section_pattern([
#        r"interests?", r"hobbies", r"activities",
#    ]),
#}
#
#
## A list of common section-related keywords used by Strategy B to check
## whether an unrecognised heading-shaped line is at least "about" one of
## the topics we know resumes cover. This is what fixed the v2 bug: a
## line containing one of these words is far more likely to be a genuine
## (if unusually-worded) section heading than a person's name, even
## without blank lines around it for support.
#_SECTION_KEYWORDS = [
#    "skill", "experience", "education", "project", "certif", "licen",
#    "achievement", "award", "honor", "honour", "summary", "objective",
#    "qualification", "employment", "career", "academic", "competenc",
#    "expertise", "technolog", "portfolio", "credential", "accreditation",
#    "recognition", "accomplishment", "language", "interest", "hobbies",
#    "research", "leadership", "internship", "training", "volunteer",
#    "publication", "reference", "extracurricular",
#]
#
#
#@dataclass
#class Section:
#    """Represents one identified section of a resume."""
#    label: str           # e.g. "experience", "skills"
#    raw_text: str        # the full text content of this section
#    start_line: int       # line number where this section starts
#    confidence: float    # 0.0 to 1.0, how sure we are this is the right label
#
#
#def _clean_heading_candidate(line: str) -> str:
#    """
#    Strips common heading decoration (colons, dashes, surrounding
#    whitespace) before checking a line against our patterns, so
#    "EDUCATION:" or "- SKILLS -" still match plain "EDUCATION"/"SKILLS".
#    """
#    cleaned = line.strip()
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def detect_section_label(line: str) -> tuple:
#    """
#    Strategy A: Checks a line against all known section heading patterns.
#    Returns (label, confidence) or (None, 0.0) if no match.
#    """
#    candidate = _clean_heading_candidate(line)
#
#    if not candidate:
#        return None, 0.0
#
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            return label, 0.95
#
#    return None, 0.0
#
#
#def _contains_section_keyword(line: str) -> bool:
#    """
#    Checks whether a line contains any word commonly associated with
#    resume section headings (even partially, e.g. "certif" matches both
#    "Certifications" and "Certificate"). Used by Strategy B as a strong
#    positive signal that distinguishes real (if unusually-worded)
#    headings from short lines like a candidate's name or a company name.
#    """
#    lower = line.lower()
#    return any(keyword in lower for keyword in _SECTION_KEYWORDS)
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    """
#    Strategy B: Scores a line on how likely it is to be a section heading,
#    for cases where Strategy A didn't recognise the wording at all.
#    Returns a score from 0.0 (not a heading) to 1.0 (definitely a heading).
#    """
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#
#    # A resume's very first two lines are reserved for name / tagline,
#    # never a section heading -- regardless of how heading-like they look.
#    if line_index <= 1:
#        return 0.0
#
#    word_count = len(stripped.split())
#
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#
#    # A line ending in a period is, by construction, the end of a
#    # sentence -- real headings are short labels, never full sentences,
#    # and never end with a period. This guard fixed a real regression:
#    # "customer experience." (the tail end of a wrapped sentence in a
#    # professional summary) was being scored as a heading purely because
#    # it's short and contains the word "experience" -- exactly the kind
#    # of sentence-fragment false positive this check eliminates.
#    if stripped.endswith("."):
#        return 0.0
#
#    score = 0.0
#
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3
#
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#
#    if stripped.endswith(":"):
#        score += 0.15
#
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#
#    # Strong positive signal: does this line actually contain a word
#    # commonly used in resume section headings? This is the fix for the
#    # v2 bug -- it directly checks the thing that actually distinguishes
#    # a real (if unrecognised) heading from a person's name, instead of
#    # relying only on blank-line spacing, which not all resumes use.
#    if _contains_section_keyword(stripped):
#        score += 0.35
#    else:
#        # No blank-line support AND no section-related keyword at all --
#        # this combination is the strongest signal that we're looking at
#        # something like a name or a company name, not a heading.
#        if not blank_below and not blank_above:
#            score *= 0.5
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    """
#    Main entry point for Layer 1.
#
#    Takes the full resume text (Layer 0's output) and returns a list of
#    Section objects -- each with a label, its text content, the line it
#    started on, and a confidence score.
#    """
#    lines = text.split("\n")
#    sections = []
#
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#
#    for i, line in enumerate(lines):
#        stripped = line.strip()
#
#        label, confidence = detect_section_label(stripped)
#
#        if label is None and stripped != "":
#            heading_score = score_heading_line(stripped, i, lines)
#            if heading_score >= 0.65:
#                label = "unknown"
#                confidence = heading_score
#
#        if label is not None and stripped != "":
#            section_text = "\n".join(current_lines).strip()
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            current_confidence = confidence
#        else:
#            current_lines.append(line)
#
#    section_text = "\n".join(current_lines).strip()
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    """
#    Helper: returns the combined text of all sections with a given label.
#    """
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#






















#just changing because shruti resume some headers not recognised .
#"""
#Section Splitter — Layer 1 (Fixed Production Version)
#
#Splits resume text into labelled sections using a cascade of strategies:
#  Strategy A: Regex heading detection (handles the large majority of resumes)
#  Strategy B: Line feature scoring (handles ambiguous/unrecognised headings)
#
#This version keeps the overall architecture from the original draft
#(confidence scores, the Section dataclass, multi-section merging via
#get_section_text) but fixes a verified bug: none of the original regex
#patterns had an end-anchor, so they matched the START of a line and
#ignored everything after it. That caused two kinds of real failures,
#confirmed against real resumes:
#  1. Ordinary sentences that happen to START with a matching word (e.g.
#     "Experienced in partnering with Product, Business...") were wrongly
#     treated as new section headings.
#  2. Real headings using slightly different wording than expected (e.g.
#     "INDUSTRY EXPERIENCE", which starts with "INDUSTRY" not "WORK") were
#     never recognised at all, since the pattern's alternatives didn't
#     anticipate that prefix.
#
#We also tightened the heading-shape scorer (Strategy B), since it was
#flagging candidates' own names (e.g. "MADAN CHAWLA") as headings purely
#because they're short and uppercase.
#"""
#
#import re
#from dataclasses import dataclass
#
#
## ─────────────────────────────────────────────────────────────
## SECTION HEADING PATTERNS
##
## Each pattern is wrapped so it must match the ENTIRE line (after we
## strip trailing colons/dashes), not just the start of it. This is done
## with a leading ^(?: ... )$ wrapper around all the alternatives, instead
## of leaving each alternative open-ended. We also added a few extra
## common prefixes seen in real resumes (e.g. "industry experience",
## "relevant experience") that the original pattern set missed.
## ─────────────────────────────────────────────────────────────
#
#SECTION_PATTERNS = {
#    "summary": re.compile(
#        r"^(?:(professional\s+)?summary|objective|profile|about\s+me|"
#        r"career\s+objective|personal\s+statement)$",
#        re.IGNORECASE
#    ),
#    "experience": re.compile(
#        r"^(?:(work\s+|industry\s+|relevant\s+|professional\s+)?experience|"
#        r"employment(\s+history)?|professional\s+background|"
#        r"career\s+history|work\s+history|internships?)$",
#        re.IGNORECASE
#    ),
#    "education": re.compile(
#        r"^(?:education(al)?(\s+background)?|academic(\s+background)?|"
#        r"academic\s+qualifications?|qualifications?|degrees?|university|college)$",
#        re.IGNORECASE
#    ),
#    "skills": re.compile(
#        r"^(?:(technical\s+|core\s+|key\s+)?(skills?|competenc(y|ies))|"
#        r"expertise|technologies|"
#        r"programming\s+(languages?|skills?)|"
#        r"tools?\s*(&|and)\s*technologies|"
#        r"core\s+competencies\s*(&|and)\s*technical\s+skills)$",
#        re.IGNORECASE
#    ),
#    "projects": re.compile(
#        r"^(?:(personal\s+|side\s+|key\s+|notable\s+|academic\s+)?projects?|"
#        r"portfolio|"
#        r"key\s+projects?\s*(&|and)\s*achievements?)$",
#        re.IGNORECASE
#    ),
#    "certifications": re.compile(
#        r"^(?:certif(ication|icate)s?|licen[sc]es?|accreditations?|"
#        r"credentials?|professional\s+certifications?)$",
#        re.IGNORECASE
#    ),
#    "achievements": re.compile(
#        r"^(?:achievements?|awards?|honou?rs?|recognitions?|"
#        r"accomplishments?|accolades?)$",
#        re.IGNORECASE
#    ),
#    "languages": re.compile(
#        r"^(?:languages?|spoken\s+languages?)$",
#        re.IGNORECASE
#    ),
#    "interests": re.compile(
#        r"^(?:interests?|hobbies|activities)$",
#        re.IGNORECASE
#    ),
#}
#
#
#@dataclass
#class Section:
#    """Represents one identified section of a resume."""
#    label: str           # e.g. "experience", "skills"
#    raw_text: str        # the full text content of this section
#    start_line: int       # line number where this section starts
#    confidence: float    # 0.0 to 1.0, how sure we are this is the right label
#
#
#def _clean_heading_candidate(line: str) -> str:
#    """
#    Strips common heading decoration (colons, dashes, surrounding
#    whitespace) before checking a line against our patterns, so
#    "EDUCATION:" or "- SKILLS -" still match plain "EDUCATION"/"SKILLS".
#    """
#    cleaned = line.strip()
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def detect_section_label(line: str) -> tuple:
#    """
#    Strategy A: Checks a line against all known section heading patterns.
#    Returns (label, confidence) or (None, 0.0) if no match.
#
#    Because every pattern above is now anchored at both ends (^...$),
#    this only matches when the ENTIRE cleaned line is a heading phrase
#    -- not when a heading word merely appears at the start of a longer
#    sentence.
#    """
#    candidate = _clean_heading_candidate(line)
#
#    if not candidate:
#        return None, 0.0
#
#    for label, pattern in SECTION_PATTERNS.items():
#        if pattern.match(candidate):
#            return label, 0.95
#
#    return None, 0.0
#
#
#def score_heading_line(line: str, line_index: int, all_lines: list) -> float:
#    """
#    Strategy B: Scores a line on how likely it is to be a section heading,
#    for cases where Strategy A didn't recognise the wording at all (e.g.
#    a section called "LEADERSHIP" or "VOLUNTEER WORK" that we have no
#    pattern for). Returns a score from 0.0 (not a heading) to 1.0
#    (definitely a heading).
#
#    Fix vs. the original version: the very first two lines of a resume
#    are almost always the candidate's name and/or job title/tagline --
#    never a section heading. The original scorer had no awareness of
#    line position, so short, all-caps names (e.g. "MADAN CHAWLA") were
#    incorrectly scored as headings. We now exclude lines 0 and 1
#    entirely from Strategy B, and slightly raised the uppercase/length
#    bar so common short ALL-CAPS phrases need more supporting signals
#    (blank lines around them) before being trusted.
#    """
#    stripped = line.strip()
#    if len(stripped) == 0:
#        return 0.0
#
#    # A resume's very first two lines are reserved for name / tagline,
#    # never a section heading -- regardless of how heading-like they look.
#    if line_index <= 1:
#        return 0.0
#
#    word_count = len(stripped.split())
#
#    # Very long lines are almost never headings
#    if word_count > 8 or len(stripped) > 60:
#        return 0.0
#
#    score = 0.0
#
#    is_all_caps = stripped == stripped.upper() and re.match(r"^[A-Z\s&/]+$", stripped)
#    if is_all_caps:
#        score += 0.3   # lowered from 0.4 -- uppercase alone is weaker evidence than before
#
#    if word_count <= 4:
#        score += 0.2
#    elif word_count <= 6:
#        score += 0.1
#
#    blank_below = (line_index + 1 < len(all_lines) and all_lines[line_index + 1].strip() == "")
#    blank_above = (line_index > 0 and all_lines[line_index - 1].strip() == "")
#
#    if blank_below:
#        score += 0.2
#    if blank_above:
#        score += 0.15
#
#    if stripped.endswith(":"):
#        score += 0.15
#
#    if "," not in stripped and ". " not in stripped:
#        score += 0.1
#
#    # Require at least one blank-line signal (above or below) in addition
#    # to everything else -- a short all-caps line sitting in the middle of
#    # dense text (no blank lines around it) is much more likely to be a
#    # company name or a tag/label, not a heading. This is the main extra
#    # guard that prevents names and short labels from scoring high.
#    if not blank_below and not blank_above:
#        score *= 0.6
#
#    return min(score, 1.0)
#
#
#def split_into_sections(text: str) -> list:
#    """
#    Main entry point for Layer 1.
#
#    Takes the full resume text (Layer 0's output) and returns a list of
#    Section objects -- each with a label, its text content, the line it
#    started on, and a confidence score.
#
#    Text before the first recognised heading is labelled "header" (name,
#    contact details, headline). A heading that looks heading-shaped
#    (Strategy B) but doesn't match any known pattern is labelled
#    "unknown" rather than silently absorbed into the wrong section, so
#    we can see in testing whether real resumes use section names we
#    haven't accounted for yet.
#    """
#    lines = text.split("\n")
#    sections = []
#
#    current_label = "header"
#    current_start = 0
#    current_lines = []
#    current_confidence = 0.9
#
#    for i, line in enumerate(lines):
#        stripped = line.strip()
#
#        label, confidence = detect_section_label(stripped)
#
#        if label is None and stripped != "":
#            heading_score = score_heading_line(stripped, i, lines)
#            if heading_score >= 0.65:
#                label = "unknown"
#                confidence = heading_score
#
#        if label is not None and stripped != "":
#            section_text = "\n".join(current_lines).strip()
#            if section_text:
#                sections.append(Section(
#                    label=current_label,
#                    raw_text=section_text,
#                    start_line=current_start,
#                    confidence=current_confidence
#                ))
#            current_label = label
#            current_start = i
#            current_lines = []
#            current_confidence = confidence
#        else:
#            current_lines.append(line)
#
#    section_text = "\n".join(current_lines).strip()
#    if section_text:
#        sections.append(Section(
#            label=current_label,
#            raw_text=section_text,
#            start_line=current_start,
#            confidence=current_confidence
#        ))
#
#    return sections
#
#
#def get_section_text(sections: list, label: str) -> str:
#    """
#    Helper: returns the combined text of all sections with a given label.
#    Some resumes have two sections that map to the same label (e.g. two
#    separate "experience" blocks for full-time vs. freelance work) --
#    this merges them on request rather than during the main pass, so the
#    original separation stays available if ever needed.
#    """
#    matching = [s for s in sections if s.label == label]
#    if not matching:
#        return ""
#    return "\n\n".join([s.raw_text for s in matching])
#






















##worked with our version but change for advanced version
#"""
#segmenter/section_splitter.py
#Layer 1 — Section Segmenter
#
#Job: take the full plain-text resume (Layer 0's output) and split it into
#labeled chunks: summary, experience, education, skills, projects,
#certifications, achievements.
#
#This uses "Strategy A" from the architecture: explicit heading detection
#via regex patterns. This alone is expected to handle the large majority
#of real resumes, since most candidates use recognisable (if varied)
#heading words. We are deliberately NOT building the ML classifier
#(Strategy B) or date-based fallback (Strategy C) yet — we'll only add
#those if we run into real resumes that this simpler approach fails on.
#"""
#
#import re
#
#
## ── Section heading patterns ────────────────────────────────────────────
## Each entry says: "if a line matches this pattern, it's the start of
## this section." (?i) means "ignore uppercase/lowercase differences" —
## so "EDUCATION", "education", and "Education" all match the same way.
##
## ^ means "the line must START with this" and $ means "the line must END
## here" (with optional trailing punctuation/whitespace handled separately
## below). Together, ^...$ means "the ENTIRE line, start to finish, must
## look like this" — not just contain these words somewhere in a sentence.
## This is what stops a normal sentence like "7 years of experience in
## fintech" from being mistaken for an "Experience" heading: that sentence
## doesn't match start-to-end, it has extra words around "experience".
#SECTION_PATTERNS = {
#    "summary": r"(?i)^(professional\s+)?summary$|^objective$|^profile$|^about\s+me$",
#
#    "experience": r"(?i)^(work\s+|industry\s+)?experience$|^employment\s+history$|^professional\s+background$|^career\s+history$|^work\s+history$",
#
#    "education": r"(?i)^education$|^academic\s+qualifications?$|^qualifications?$",
#
#    "skills": r"(?i)^(technical\s+|core\s+)?(skills|competencies)\s*(&\s*technical\s*skills)?$|^expertise$|^technologies$",
#
#    "projects": r"(?i)^projects?$|^personal\s+projects?$|^side\s+projects?$|^portfolio$|^key\s+projects?(\s*&\s*achievements?)?$",
#
#    "certifications": r"(?i)^certifications?$|^licenses?$|^accreditations?$|^credentials?$",
#
#    "achievements": r"(?i)^achievements?$|^awards?$|^honou?rs?$|^recognition$|^accomplishments?$",
#}
#
#
#def _clean_heading_line(line: str) -> str:
#    """
#    Before checking a line against our patterns, strip away common
#    decoration that resumes use around headings — colons, dashes, and
#    extra surrounding whitespace — so "EDUCATION:" or "- EDUCATION -"
#    still matches the same as a plain "EDUCATION".
#    """
#    cleaned = line.strip()
#    cleaned = cleaned.strip(":-—–_ ")
#    return cleaned
#
#
#def _match_section_heading(line: str):
#    """
#    Checks a single line against every section pattern.
#    Returns the matching section name (e.g. "skills") if this line looks
#    like that section's heading, or None if it doesn't look like any
#    known heading at all.
#    """
#    candidate = _clean_heading_line(line)
#
#    # Extra safety check: a real heading is short. If a line is unusually
#    # long, it's almost certainly a full sentence that happens to start
#    # with a matching word, not an actual heading. This guards against
#    # edge cases our ^...$ anchors might not catch on their own (e.g. a
#    # heading-like phrase embedded in a longer line due to a Layer 0
#    # quirk). 40 characters comfortably fits every real heading we've
#    # seen so far (e.g. "CORE COMPETENCIES & TECHNICAL SKILLS" is 37).
#    if len(candidate) > 45:
#        return None
#
#    if not candidate:
#        return None
#
#    for section_name, pattern in SECTION_PATTERNS.items():
#        if re.match(pattern, candidate):
#            return section_name
#
#    return None
#
#
#def split_into_sections(resume_text: str) -> dict:
#    """
#    Main entry point for Layer 1.
#
#    Takes the full resume text (a single string, as produced by Layer 0)
#    and returns a dictionary mapping section names to their text content.
#
#    Any text that appears BEFORE the first recognised heading (typically
#    the candidate's name, contact details, and headline) is stored under
#    the special key "header" — this is useful later for Layer 2's contact
#    extractor, which needs to scan this area for email/phone/LinkedIn.
#
#    Any text that doesn't match the 7 known section types but DOES look
#    like some kind of heading we don't have a pattern for yet is stored
#    under "unmatched_<heading text>", so we never silently lose content —
#    we can see in testing if real resumes use heading words we haven't
#    accounted for.
#    """
#    lines = resume_text.split("\n")
#
#    sections = {}
#    current_section = "header"
#    current_lines = []
#
#    for line in lines:
#        heading_match = _match_section_heading(line)
#
#        if heading_match is not None:
#            # We've hit a new recognised heading. Save whatever we were
#            # collecting for the PREVIOUS section before starting fresh.
#            if current_lines:
#                _append_to_section(sections, current_section, current_lines)
#
#            current_section = heading_match
#            current_lines = []
#        else:
#            current_lines.append(line)
#
#    # After the loop ends, there's always one last section still being
#    # collected that never got saved inside the loop — save it now.
#    if current_lines:
#        _append_to_section(sections, current_section, current_lines)
#
#    return sections
#
#
#def _append_to_section(sections: dict, section_name: str, lines: list):
#    """
#    Joins a list of lines into one text block and adds it to the
#    sections dictionary. If this section name already exists (e.g. a
#    resume has two separate "EXPERIENCE" blocks for some reason), we
#    append rather than overwrite, so we never silently lose content.
#    """
#    text_block = "\n".join(lines).strip()
#    if not text_block:
#        return
#
#    if section_name in sections:
#        sections[section_name] += "\n\n" + text_block
#    else:
#        sections[section_name] = text_block