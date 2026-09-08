"""
Education Extractor — Layer 2
Extracts: degree name, degree type, institution, year of completion, GPA/percentage.

Design notes (why this differs from a naive line-by-line parser):
- Real resumes wrap education entries across lines inconsistently. Sometimes the
  institution appears BEFORE the degree line (e.g. "VESIT, Mumbai, 2008" then
  "B.E.: I.T"). Sometimes two entries sit back-to-back with NO blank line between
  them (e.g. an M.E. entry immediately followed by a B.E. entry).
- Because of that, entries are NOT split on blank lines. They're split on
  "trigger lines" — lines that match a degree pattern. Lines before the first
  trigger attach to entry 1; lines after a trigger attach to that entry until the
  next trigger appears. This handles both orderings above correctly.
- Each entry's lines are joined into a single string before running the
  institution/year/GPA regexes, so a name wrapped across two lines
  ("... Mumbai" / "University, Mumbai Jan 2018") is still matched as one phrase.
"""

import json
import re
from pathlib import Path

from flashtext import KeywordProcessor
from rapidfuzz import fuzz, process

# ─────────────────────────────────────────────────────────────
# DEGREE PATTERNS (checked in priority order — first match wins)
# ─────────────────────────────────────────────────────────────

def _load_degree_taxonomy() -> dict:
    """Loads data/degree_taxonomy.json, built once by build_degree_taxonomy.py.
    Unlike institutions, degree names are a closed, well-known vocabulary —
    same curated-taxonomy-plus-fuzzy-match pattern your skills extractor uses,
    rather than a handful of broad regex buckets."""
    path = Path("data/degree_taxonomy.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_DEGREE_TAXONOMY = _load_degree_taxonomy()

# flashtext: alias (lowercase) -> canonical degree name, e.g. "b.tech" -> "B.Tech"
_DEGREE_MATCHER = KeywordProcessor(case_sensitive=False)
_DEGREE_CATEGORY = {}          # canonical name -> category (bachelors/masters/...)
_ALL_ALIASES = []              # flat list, used for the fuzzy fallback

for _canonical, _info in _DEGREE_TAXONOMY.items():
    _DEGREE_CATEGORY[_canonical] = _info["category"]
    for _alias in _info["aliases"]:
        _DEGREE_MATCHER.add_keyword(_alias, _canonical)
        _ALL_ALIASES.append((_alias, _canonical))

_FUZZY_DEGREE_THRESHOLD = 85

# Punctuation/space-stripped index, used as a second-tier exact match
# BETWEEN the strict matcher and the fuzzy fallback. flashtext treats "."
# as a word boundary, so a taxonomy alias like "bca" never matches resume
# text written as "[B.C.A]" or "B. C. A" via the strict matcher above —
# and the fuzzy fallback below often rejects it too, since the extra dots
# drag the similarity ratio under threshold even though the letters match
# exactly once punctuation is removed. Confirmed on a real resume (Ajit
# Kumar): "[B.C.A] Subharti University , Meerut : 68%" matched nothing at
# all, so no new entry ever started for it — it silently got swallowed
# into the adjacent "12th" entry's institution-line merge logic instead.
_DEGREE_MATCHER_STRIPPED = KeywordProcessor(case_sensitive=False)
for _alias, _canonical in _ALL_ALIASES:
    _stripped_alias = re.sub(r'[.\s]', '', _alias)
    if _stripped_alias:
        _DEGREE_MATCHER_STRIPPED.add_keyword(_stripped_alias, _canonical)


def match_degree(line: str) -> tuple[str, str, int, int] | tuple[None, None, None, None]:
    """Returns (canonical_degree_name, category, match_start, match_end) for
    the first degree found in a line, or all-None. The span lets us find
    specialization text right after the degree token, and mask the degree
    token itself out before institution search (so e.g. 'School' inside
    'High School' doesn't get mistaken for the institution's own name)."""
    hits = _DEGREE_MATCHER.extract_keywords(line, span_info=True)
    if hits:
        canonical, start, end = hits[0]
        return canonical, _DEGREE_CATEGORY[canonical], start, end

    # Punctuation-stripped tier — catches aliases that only differ from the
    # line by punctuation/spacing (e.g. "B.C.A" vs taxonomy alias "bca").
    # No reliable span on stripped text, so field extraction/masking is
    # skipped here, same as the fuzzy path below already does.
    stripped_line = re.sub(r'[.\s]', '', line)
    if stripped_line:
        stripped_hits = _DEGREE_MATCHER_STRIPPED.extract_keywords(stripped_line)
        if stripped_hits:
            canonical = stripped_hits[0]
            return canonical, _DEGREE_CATEGORY[canonical], None, None

    # Fuzzy fallback — only worth trying on short tokens (a handful of words),
    # since fuzzy-matching a whole sentence against "B.Tech" is meaningless.
    # No reliable span here, so field extraction/masking is skipped for fuzzy matches.
    for token in re.findall(r"[A-Za-z][A-Za-z.\-]{1,12}", line):
        if len(token) < 2:
            continue
        best = process.extractOne(
            token.lower(), [a for a, _ in _ALL_ALIASES], scorer=fuzz.ratio
        )
        if best and best[1] >= _FUZZY_DEGREE_THRESHOLD:
            matched_alias = best[0]
            canonical = next(c for a, c in _ALL_ALIASES if a == matched_alias)
            return canonical, _DEGREE_CATEGORY[canonical], None, None
    return None, None, None, None


_FIELD_PAREN_RE = re.compile(r'^\(([^)]+)\)')
_FIELD_COLON_RE = re.compile(r'^:\s*(.+)')
_FIELD_CONNECTOR_RE = re.compile(r'^(?:in|of|with|stream\s*:?|specialization\s*:?)\s+(.+)', re.IGNORECASE)
_FIELD_DASH_RE = re.compile(r'^[-\u2013]\s*(.+)')


def _extract_degree_field(trigger_line: str, match_end: int | None) -> str | None:
    """Pulls the specialization/field out of the same line as the degree
    token, e.g. 'B.E.: I.T' -> 'I.T', 'M.E. (Mechanical Engg.)' -> 'Mechanical
    Engg.', 'B.Tech in Computer Science' -> 'Computer Science'. Only looks at
    text immediately after the degree token — if nothing recognizable follows
    (no colon/parens/in/of/dash), we don't guess. A parenthetical that's just
    a bare year ('(2019)') or a restatement of the degree abbreviation itself
    ('Bachelor of Technology (B.tech)') isn't a field — skip it and look past it."""
    if match_end is None:
        return None
    remainder = trigger_line[match_end:].strip()
    if not remainder:
        return None

    m = _FIELD_PAREN_RE.match(remainder)
    if m:
        content = m.group(1).strip()
        is_bare_year = re.fullmatch(r'(19|20)\d{2}', content)
        is_degree_restatement = bool(_DEGREE_MATCHER.extract_keywords(content))
        if not is_bare_year and not is_degree_restatement:
            return content or None
        # Skip past this parenthetical and look for a field marker after it
        remainder = remainder[m.end():].strip()
        if not remainder:
            return None

    field = None
    for pattern in (_FIELD_COLON_RE, _FIELD_CONNECTOR_RE, _FIELD_DASH_RE):
        m = pattern.match(remainder)
        if m:
            field = m.group(1)
            break
    if field is None:
        return None

    # Stop at the first comma/bullet/pipe — usually separates field from
    # institution/city or from ranking blurbs ("Finance • Top 10% of 572")
    field = re.split(r'[,•|]', field)[0].strip(" .")
    return field or None

# Year — plain year, or a fuller date like "23 Sep 2011" / "Jan 2018"
YEAR_RE = re.compile(
    r'\b(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+'
    r'(19[6-9]\d|20[0-3]\d)\b|\b(19[6-9]\d|20[0-3]\d)\b',
    re.IGNORECASE
)

# Full date range, e.g. "Aug 2019 - June 2023", "2019 – 2023", or
# "Aug'20 – May'25". Tried BEFORE the single-year fallback in
# _extract_year(), since resumes usually give the whole span and
# returning only the end year (the old behavior) throws away real
# information. The year token accepts either a full 4-digit year OR an
# apostrophe + 2-digit year — confirmed on a real resume (Akash Verma)
# whose dates were written as "Aug'20 – May'25" with zero whitespace
# between month and year and a 2-digit year, which the old 4-digit-only,
# space-required token couldn't match at all.
_YEAR_TOKEN_RE = r"(?:19[6-9]\d|20[0-3]\d|['’‘`´]\d{2})"
_DATE_TOKEN_RE = (
    r'(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*)?'
    rf'{_YEAR_TOKEN_RE}'
)
DATE_RANGE_RE = re.compile(
    rf'({_DATE_TOKEN_RE})\s*(?:[-–—]|to)\s*({_DATE_TOKEN_RE})',
    re.IGNORECASE
)


def _expand_apostrophe_year(date_str: str) -> str:
    """'20 -> 2020, '78 -> 1978. Same <=49-is-2000s convention used
    elsewhere in this pipeline (see experience.py), applied here so a
    range like "Aug'20 - May'25" renders as "Aug 2020 - May 2025"
    instead of leaving the ambiguous 2-digit form in the output."""
    m = re.search(r"['’‘`´](\d{2})\b", date_str)
    if not m:
        return date_str
    two_digit = int(m.group(1))
    century = "20" if two_digit <= 49 else "19"
    return date_str[:m.start()] + century + m.group(1) + date_str[m.end():]

# GPA — keyword-led first ("CGPA: 8.16 /10", "Percentage:- 89%")
GPA_KEYWORD_RE = re.compile(
    r'(?:gpa|cgpa|sgpa|percentage|grade|score|marks)[:\s\-]+([0-9.]+(?:\s*/\s*[0-9.]+)?%?)',
    re.IGNORECASE
)
# Reversed order — number BEFORE the keyword ("7.8 CGPA", "7.8 SGPA")
GPA_KEYWORD_REVERSED_RE = re.compile(
    r'\b([0-9.]+)\s*(?:gpa|cgpa|sgpa)\b', re.IGNORECASE
)
# Fallback: bare "8.16 /10" style CGPA with no keyword in front
GPA_BARE_SCALE_RE = re.compile(r'\b(\d(?:\.\d{1,2})?)\s*/\s*(10|4)\b')
# Fallback: bare percentage with no keyword, e.g. "62.46%" — but NOT a class-rank
# phrase like "Top 10% of 572" or "Top 2% in Finance", which isn't a GPA at all.
GPA_BARE_PERCENT_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\s*%')
_RANKING_PHRASE_RE = re.compile(r'\btop\b', re.IGNORECASE)

# Institution keyword fallback now uses _INSTITUTION_KEYWORDS_RE defined
# alongside _match_keyword_fallback() below (anchor + expand, not one greedy regex).


def _load_institutions_db() -> dict:
    """Loads data/institutions_db.json, built once by build_institutions_db.py
    from the AISHE colleges dataset + Hipolabs India universities list."""
    db_path = Path("data/institutions_db.json")
    if db_path.exists():
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"full_names": [], "acronym_index": {}}


_INSTITUTIONS_DB = _load_institutions_db()

# flashtext processor for whole-phrase, case-insensitive full-name matching.
# Full names rarely collide across institutions, so this tier is high-confidence.
_FULL_NAME_MATCHER = KeywordProcessor(case_sensitive=False)
for _name in _INSTITUTIONS_DB.get("full_names", []):
    _FULL_NAME_MATCHER.add_keyword(_name)


_LEADING_WORD_RE = re.compile(r"^[A-Z][a-z.&'\-]*$|^[A-Z]{2,20}$")  # Title-Case OR a plausible all-caps word
_CONNECTOR_WORDS = {"of", "the", "and", "for"}

# All-caps tokens that are degree/GPA jargon, not institution-name acronyms —
# these should never be swept into an institution name during expansion.
_NON_INSTITUTION_ACRONYMS = {"CGPA", "SGPA", "GPA", "SSC", "HSC", "PGDM", "PGDCA", "PGDBA"} | {
    a.upper().replace(".", "") for a, _ in _ALL_ALIASES if a.isupper() or "." in a
}


def _is_expandable_word(tok: str) -> bool:
    if not _LEADING_WORD_RE.match(tok):
        return False
    if tok.upper().replace(".", "") in _NON_INSTITUTION_ACRONYMS:
        return False
    return True


def _expand_left(entry_text: str, start: int) -> list:
    """Expands leftward from `start` over Title-Case words/connectors, but
    guards against sweeping in a specialization word — e.g. in 'PGDM in
    Finance IIM Ahmedabad', 'Finance' would otherwise get pulled into the
    institution name just because it's Title-Case. If exactly one word
    would be pulled in and it's directly preceded by 'in', that's a
    specialization clause ('degree in X'), not part of the institution."""
    before = entry_text[:start].rstrip()
    tokens_before = before.split(" ") if before else []
    extended = []
    stop_token = None
    for tok in reversed(tokens_before):
        if _is_expandable_word(tok) or tok.lower() in _CONNECTOR_WORDS:
            extended.insert(0, tok)
        else:
            stop_token = tok
            break
    if len(extended) == 1 and stop_token and stop_token.lower() == "in":
        return []
    return extended


def _expand_right(entry_text: str, end: int) -> list:
    """Expands rightward from `end` over Title-Case words/connectors, stopping
    at a comma/paren/digit. Fixes DB hits like 'Indian Institute of
    Management' truncating the real name 'Indian Institute of Management
    Calcutta' (connector 'of' must be allowed through, not just Title-Case
    words, or the expansion stops one word too early)."""
    after = entry_text[end:].lstrip()
    tokens_after = after.split(" ") if after else []
    extended = []
    for tok in tokens_after:
        clean = tok.rstrip(",")
        if _is_expandable_word(clean) or clean.lower() in _CONNECTOR_WORDS:
            extended.append(clean)
            if clean != tok:  # trailing comma — stop after including this word
                break
        else:
            break
    return extended


def _match_full_name(entry_text: str) -> str | None:
    hits = _FULL_NAME_MATCHER.extract_keywords(entry_text, span_info=True)
    if not hits:
        return None
    # Longest span wins among direct hits (more specific match).
    keyword, start, end = max(hits, key=lambda h: h[2] - h[1])
    left = _expand_left(entry_text, start)
    right = _expand_right(entry_text, end)
    prefix = " ".join(left)
    suffix = " ".join(right)
    parts = [p for p in (prefix, keyword, suffix) if p]
    return " ".join(parts).strip()


_INSTITUTION_KEYWORDS_SAFE_RE = re.compile(
    r'\b(university|institute|institution|college|school|board|iit|nit|bits|iim|ignou|'
    r'nmims|aktu|vtu|rntu|srm|vit|vesit|hvn|polytechnic|vidyalaya|vidya\s*niketan|'
    r'vishwa\s*vidyalaya|shikshan|parishad|kendriya|convent|high\s+school|public\s+school)\b',
    re.IGNORECASE
)
_INSTITUTION_KEYWORDS_RISKY_RE = re.compile(r'\b(academy)\b', re.IGNORECASE)
# Kept for _extract_institution's final-string matching, where both matter equally
_INSTITUTION_KEYWORDS_RE = re.compile(
    r'\b(university|institute|institution|academy|college|school|board|iit|nit|bits|iim|ignou|'
    r'nmims|aktu|vtu|rntu|srm|vit|vesit|hvn|polytechnic|vidyalaya|vidya\s*niketan|'
    r'vishwa\s*vidyalaya|shikshan|parishad|kendriya|convent|high\s+school|public\s+school)\b',
    re.IGNORECASE
)
_BARE_GENERIC_INSTITUTION = {"school", "college", "university", "institute", "board"}

# Vocabulary that shows up in table-style education sections' column-header
# row (e.g. "Qualification University Year Of Passing"). Used only to
# detect and skip that header row entirely — see _looks_like_table_header_row().
_TABLE_HEADER_WORDS = {
    "qualification", "degree", "exam", "board", "institute", "institution",
    "university", "year", "of", "passing", "duration", "percentage",
    "cgpa", "gpa", "score", "specialization", "course", "stream", "marks",
}


def _looks_like_table_header_row(line: str) -> bool:
    """Some resumes render education as a table; the segmenter keeps the
    column-header row as plain text (e.g. 'Qualification University Year
    Of Passing'). Confirmed on a real resume (Akash Kumar) where this row
    became its own bogus entry, since it contains "University" -> counts
    as institution signal -> survives the existing bare-generic-word
    filter (which only catches a SINGLE generic word alone, not a whole
    header phrase).

    A real entry always has at least one digit in it somewhere (a year,
    a percentage, a CGPA figure) — a pure header row doesn't. Requiring
    BOTH "every word is header vocabulary" AND "no digits present" means
    this can't misfire on a real entry that happens to mention
    "University" once among real names/numbers."""
    if re.search(r'\d', line):
        return False
    words = [w.strip(" .,:").lower() for w in line.split()]
    words = [w for w in words if w]
    return bool(words) and all(w in _TABLE_HEADER_WORDS for w in words)


def _match_keyword_fallback(entry_text: str) -> str | None:
    m = _INSTITUTION_KEYWORDS_RE.search(entry_text)
    if not m:
        return None
    start, end = m.start(), m.end()
    left = _expand_left(entry_text, start)
    right = _expand_right(entry_text, end)

    parts = left + [m.group(0)] + right
    candidate = " ".join(parts).strip()
    return candidate or None


def _extract_institution(entry_text: str) -> tuple[str | None, bool]:
    """Returns (institution, needs_review). Priority: full name in DB (safest)
    -> generic keyword regex (not DB-verified, but usually accurate when a
    resume spells the name out in full).

    There used to be a third tier here: matching bare acronyms (SVIT, EIILM,
    IIMT...) against a DB index built from institutions' generated initials.
    Removed after testing showed every real acronym-tier match came back
    flagged needs_review anyway (unique-in-DB never proved correct-in-reality
    — see e.g. "SVIT" resolving to a real but wrong college in Telangana).
    Since a flagged guess can still be misused by anything downstream that
    doesn't check the flag, and None cannot be, this trades a small amount of
    partial coverage (pure-acronym mentions with no institution keyword
    nearby) for eliminating an entire recurring class of false-confidence
    bugs (2-letter state-code collisions, cross-state acronym collisions)."""
    full = _match_full_name(entry_text)
    if full:
        return full, False

    fallback = _match_keyword_fallback(entry_text)
    if fallback:
        # A bare generic word with nothing else ("School" alone, no name
        # attached) usually means the real name got masked out along with
        # the degree phrase on the same line (e.g. "Senior Secondary
        # School" where the school's own name IS "Senior Secondary
        # School") — not enough to trust confidently.
        needs_review = fallback.strip().lower() in _BARE_GENERIC_INSTITUTION
        return fallback, needs_review

    return None, False


def _extract_year(entry_text: str, trigger_line: str | None = None) -> str | None:
    # Try the trigger line first — it's the most tightly-scoped, reliable
    # source for this entry's own date(s), and avoids picking up unrelated
    # dates from other lines that got swept into the block (e.g. a birth
    # date from a mis-segmented trailing "Personal Details" section).
    #
    # Look for a full range first ("Aug 2019 - June 2023", "Aug'20 -
    # May'25") before falling back to a single year — returning only the
    # end year silently threw away the start date even when the resume
    # stated the full range.
    if trigger_line:
        range_match = DATE_RANGE_RE.search(trigger_line)
        if range_match:
            start = _expand_apostrophe_year(range_match.group(1).strip())
            end = _expand_apostrophe_year(range_match.group(2).strip())
            return f"{start} - {end}"
        trigger_matches = list(YEAR_RE.finditer(trigger_line))
        if trigger_matches:
            m = trigger_matches[-1]
            return m.group(1) or m.group(2)

    # Fall back to the full joined entry text (covers cases where the date
    # range sits on the institution line rather than the trigger/degree line).
    range_match = DATE_RANGE_RE.search(entry_text)
    if range_match:
        start = _expand_apostrophe_year(range_match.group(1).strip())
        end = _expand_apostrophe_year(range_match.group(2).strip())
        return f"{start} - {end}"

    # Last resort: single year — prefer the LAST year found, since resumes
    # write "2010 – 2014" and the end of the range is the graduation/
    # completion year, not the start year. This path only fires now when
    # no recognizable range exists at all (e.g. a bare "2023" with no
    # start date given anywhere).
    matches = list(YEAR_RE.finditer(entry_text))
    if not matches:
        return None
    m = matches[-1]
    return m.group(1) or m.group(2)


def _extract_gpa(line_text: str) -> str | None:
    m = GPA_KEYWORD_RE.search(line_text)
    if m:
        return m.group(1).strip()
    m = GPA_KEYWORD_REVERSED_RE.search(line_text)
    if m:
        return m.group(1).strip()
    m = GPA_BARE_SCALE_RE.search(line_text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    for m in GPA_BARE_PERCENT_RE.finditer(line_text):
        window_start = max(0, m.start() - 10)
        if _RANKING_PHRASE_RE.search(line_text[window_start:m.start()]):
            continue  # "Top 10% of 572" — a class rank, not a GPA
        return f"{m.group(1)}%"
    return None


def _extract_degree(trigger_line: str, degree_type: str) -> str:
    """Best-effort clean degree label from the line that triggered this entry.
    Cuts off at the first comma so trailing institution text doesn't leak in
    (e.g. 'B.E. (Mechanical Engg.) SVIT, Pune' -> 'B.E. (Mechanical Engg.) SVIT'
    — imperfect but keeps the label short; institution field carries the real name)."""
    return trigger_line.strip()


def _line_has_institution_signal(line: str) -> bool:
    """Lightweight per-line check used only to decide entry boundaries — does
    this single line look like it's introducing an institution mention?
    Doesn't need to reconstruct the full name; that happens later via
    _extract_institution() on the complete joined block.

    Most institution keywords (university, institute, college, school...)
    essentially never show up in ordinary prose within an education section,
    so a simple presence check is fine. "academy" is the exception — it can
    appear incidentally in a sentence ("...Student association academy") —
    so it additionally requires a capitalized (proper-noun-shaped) neighbor
    before counting as a real signal."""
    if _INSTITUTION_KEYWORDS_SAFE_RE.search(line):
        return True

    m = _INSTITUTION_KEYWORDS_RISKY_RE.search(line)
    if m:
        before_tokens = line[:m.start()].rstrip().split(" ") if m.start() > 0 else []
        after_tokens = line[m.end():].lstrip().split(" ") if m.end() < len(line) else []
        before = before_tokens[-1] if before_tokens else ""
        after = after_tokens[0] if after_tokens else ""
        if after.lower().rstrip(",") in _CONNECTOR_WORDS and len(after_tokens) > 1:
            after = after_tokens[1]
        before_clean = before.strip(" ,.-")
        after_clean = after.strip(" ,.-")
        neighbor_capitalized = bool(
            (before_clean and before_clean[0].isupper()) or
            (after_clean and after_clean[0].isupper())
        )
        if neighbor_capitalized:
            return True

    if _FULL_NAME_MATCHER.extract_keywords(line):
        return True
    return False


_EXPLICIT_LEVEL_RE = re.compile(r'\((10|12)\)|\b(10|12)th\b', re.IGNORECASE)


def _explicit_school_level(text: str) -> str | None:
    """Some resumes write contradictory things like 'Higher Secondary School
    (10)' — the English name says 12th, but the explicit (10) means the
    person means 10th. When an unambiguous (10)/(12)/'10th'/'12th' marker is
    present, it should win over the fuzzy English-name-based classification.
    Deliberately narrow (paren-wrapped or 'th'-suffixed only) so it doesn't
    false-trigger on '8.16 /10' style GPA-out-of-10 scales."""
    m = _EXPLICIT_LEVEL_RE.search(text)
    if not m:
        return None
    digits = m.group(1) or m.group(2)
    return f"{digits}th"


def _flush_entry(current: dict) -> dict | None:
    """Finalizes one accumulated entry into an output record, or returns None
    if it has nothing usable (pure noise lines with no degree or institution
    signal at all)."""
    if not current["lines"]:
        return None

    # Mask out the exact degree-token substring (not the whole line!) before
    # institution search — otherwise a word like "School" inside the degree
    # phrase "High School" gets mistaken for the institution's own name,
    # while real institution text on the same line ("TDS School, Aligarh")
    # would still need to survive the mask.
    entry_text = " ".join(current["lines"])
    if current["trigger_line_index"] is not None and current["degree_start"] is not None:
        offset = len(" ".join(current["lines"][:current["trigger_line_index"]]))
        if offset:
            offset += 1  # the joining space
        abs_start = offset + current["degree_start"]
        abs_end = offset + current["degree_end"]
        entry_text = entry_text[:abs_start] + (" " * (abs_end - abs_start)) + entry_text[abs_end:]

    institution, needs_review = _extract_institution(entry_text)
    year = _extract_year(entry_text, current["trigger_line"])
    gpa = _extract_gpa(entry_text)

    degree_name = current["degree_name"]
    degree_type = current["degree_category"]
    degree = current["trigger_line"].strip() if current["trigger_line"] else None
    degree_field = _extract_degree_field(current["trigger_line"], current["degree_end"]) \
        if current["trigger_line"] else None

    # Some resumes write contradictory things like "Higher Secondary School (10)"
    # — an explicit (10)/(12)/"10th"/"12th" marker overrides the fuzzy
    # English-name-based classification, and the contradiction itself is a
    # signal this entry deserves a second look.
    explicit_level = _explicit_school_level(entry_text)
    if explicit_level and degree_type in ("10th", "12th") and explicit_level != degree_type:
        degree_type = explicit_level
        degree_name = explicit_level
        needs_review = True

    if not degree_name and not institution:
        return None  # nothing usable at all — pure noise lines

    if not degree_name and institution and institution.strip().lower() in _BARE_GENERIC_INSTITUTION:
        return None  # e.g. a table header row ("Degree/Exam Board/Institute Year") — not a real entry

    if not degree_name:
        # Institution-only entry (e.g. an executive program with no
        # recognized degree word) — still worth keeping, just flagged.
        needs_review = True

    fields_found = sum(1 for v in (degree_type, institution, year) if v)
    base_confidence = min(0.5 + 0.15 * fields_found, 0.95)
    if needs_review:
        base_confidence = min(base_confidence, 0.55)

    return {
        "degree": degree,
        "degree_name": degree_name,
        "degree_type": degree_type,
        "degree_field": degree_field,
        "institution": institution,
        "year": year,
        "gpa": gpa,
        "_needs_review": needs_review,
        "_confidence": round(base_confidence, 2),
    }


def _new_current() -> dict:
    return {
        "lines": [],
        "degree_name": None,
        "degree_category": None,
        "trigger_line": None,
        "trigger_line_index": None,
        "degree_start": None,
        "degree_end": None,
        "has_institution_signal": False,
        "last_institution_line_idx": None,
    }


def extract_education(education_section_text: str) -> dict:
    """
    Parses the education section into a list of entries.

    Rather than writing a separate rule for every ordering resumes use
    (degree-then-college, college-then-degree, same-line, across-lines,
    with/without location...), each line is classified for two signals —
    "does this line introduce a degree?" and "does this line introduce an
    institution?" — and entries are flushed whenever a line would introduce
    a SECOND degree or a SECOND institution into the entry currently being
    built. This one mechanism covers every ordering: it doesn't matter
    whether the degree or the institution comes first, only that a repeat
    of either one means we've moved on to the next entry. Lines with neither
    signal (bullet points, job-description-style prose under an education
    entry) are just carried along as context for GPA/year extraction.
    """
    if not education_section_text.strip():
        return {"education": [], "_confidence": 0.0}

    lines = [l.strip() for l in education_section_text.strip().split("\n") if l.strip()]

    education_entries = []
    current = _new_current()

    for line_idx, line in enumerate(lines):
        # Table-rendered education sections leave a column-header row (e.g.
        # "Qualification University Year Of Passing") in the plain text.
        # Skip it outright — it carries institution signal ("University")
        # but describes no real entry.
        if _looks_like_table_header_row(line):
            continue

        canonical, category, degree_start, degree_end = match_degree(line)
        has_inst_signal = _line_has_institution_signal(line)

        institution_conflict = False
        if has_inst_signal and current["has_institution_signal"]:
            last_idx = current["last_institution_line_idx"]
            adjacent = last_idx is not None and (line_idx - last_idx) == 1
            # Adjacent lines both carrying institution signal are usually one
            # institution name wrapped across a line break (e.g. "SVIT, Pune"
            # then "University ,23 Sep" — together "SVIT, Pune University"),
            # UNLESS this line also introduces its own degree word — that's
            # the tell that it's actually a new entry starting, not a
            # continuation (e.g. "...Indian Institute" then "MS in Cyber Law
            # ... National Law Institute" are two different entries, even
            # though both lines happen to carry institution signal back to back).
            if adjacent and not canonical:
                institution_conflict = False
            else:
                institution_conflict = True

        # A degree canonical can legitimately repeat on the SAME entry's own
        # institution line — e.g. taxonomy alias "Polytechnic" -> Diploma,
        # matched once on "Diploma (Computer Engineering)" and again on
        # "Saraswati Polytechnic College" two lines later. Confirmed on a
        # real resume (Sahil Singla): treating that repeat as a brand-new
        # degree forced a split, orphaning the institution from its own
        # degree line into a separate bogus entry. Only treat a REPEATED
        # identical canonical as a new entry if the line that repeats it
        # does NOT also carry institution signal — a bare restatement of
        # the exact same degree word alongside a college/university name is
        # almost always the institution's own name, not a second qualification.
        degree_conflict = bool(canonical and current["degree_name"])
        if degree_conflict and canonical == current["degree_name"] and has_inst_signal:
            degree_conflict = False

        conflict = degree_conflict or institution_conflict
        if conflict:
            finalized = _flush_entry(current)
            if finalized:
                education_entries.append(finalized)
            current = _new_current()

        current["lines"].append(line)
        if canonical and not current["degree_name"]:
            current["degree_name"] = canonical
            current["degree_category"] = category
            current["trigger_line"] = line
            current["trigger_line_index"] = len(current["lines"]) - 1
            current["degree_start"] = degree_start
            current["degree_end"] = degree_end
        if has_inst_signal:
            current["has_institution_signal"] = True
            current["last_institution_line_idx"] = line_idx

    finalized = _flush_entry(current)
    if finalized:
        education_entries.append(finalized)

    overall_confidence = round(
        sum(e["_confidence"] for e in education_entries) / len(education_entries), 2
    ) if education_entries else 0.0

    return {
        "education": education_entries,
        "_confidence": overall_confidence,
    }
















#"""
#Education Extractor — Layer 2
#Extracts: degree name, degree type, institution, year of completion, GPA/percentage.
#
#Design notes (why this differs from a naive line-by-line parser):
#- Real resumes wrap education entries across lines inconsistently. Sometimes the
#  institution appears BEFORE the degree line (e.g. "VESIT, Mumbai, 2008" then
#  "B.E.: I.T"). Sometimes two entries sit back-to-back with NO blank line between
#  them (e.g. an M.E. entry immediately followed by a B.E. entry).
#- Because of that, entries are NOT split on blank lines. They're split on
#  "trigger lines" — lines that match a degree pattern. Lines before the first
#  trigger attach to entry 1; lines after a trigger attach to that entry until the
#  next trigger appears. This handles both orderings above correctly.
#- Each entry's lines are joined into a single string before running the
#  institution/year/GPA regexes, so a name wrapped across two lines
#  ("... Mumbai" / "University, Mumbai Jan 2018") is still matched as one phrase.
#"""
#
#import json
#import re
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import fuzz, process
#
## ─────────────────────────────────────────────────────────────
## DEGREE PATTERNS (checked in priority order — first match wins)
## ─────────────────────────────────────────────────────────────
#
#def _load_degree_taxonomy() -> dict:
#    """Loads data/degree_taxonomy.json, built once by build_degree_taxonomy.py.
#    Unlike institutions, degree names are a closed, well-known vocabulary —
#    same curated-taxonomy-plus-fuzzy-match pattern your skills extractor uses,
#    rather than a handful of broad regex buckets."""
#    path = Path("data/degree_taxonomy.json")
#    if path.exists():
#        with open(path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {}
#
#
#_DEGREE_TAXONOMY = _load_degree_taxonomy()
#
## flashtext: alias (lowercase) -> canonical degree name, e.g. "b.tech" -> "B.Tech"
#_DEGREE_MATCHER = KeywordProcessor(case_sensitive=False)
#_DEGREE_CATEGORY = {}          # canonical name -> category (bachelors/masters/...)
#_ALL_ALIASES = []              # flat list, used for the fuzzy fallback
#
#for _canonical, _info in _DEGREE_TAXONOMY.items():
#    _DEGREE_CATEGORY[_canonical] = _info["category"]
#    for _alias in _info["aliases"]:
#        _DEGREE_MATCHER.add_keyword(_alias, _canonical)
#        _ALL_ALIASES.append((_alias, _canonical))
#
#_FUZZY_DEGREE_THRESHOLD = 85
#
#
#def match_degree(line: str) -> tuple[str, str, int, int] | tuple[None, None, None, None]:
#    """Returns (canonical_degree_name, category, match_start, match_end) for
#    the first degree found in a line, or all-None. The span lets us find
#    specialization text right after the degree token, and mask the degree
#    token itself out before institution search (so e.g. 'School' inside
#    'High School' doesn't get mistaken for the institution's own name)."""
#    hits = _DEGREE_MATCHER.extract_keywords(line, span_info=True)
#    if hits:
#        canonical, start, end = hits[0]
#        return canonical, _DEGREE_CATEGORY[canonical], start, end
#
#    # Fuzzy fallback — only worth trying on short tokens (a handful of words),
#    # since fuzzy-matching a whole sentence against "B.Tech" is meaningless.
#    # No reliable span here, so field extraction/masking is skipped for fuzzy matches.
#    for token in re.findall(r"[A-Za-z][A-Za-z.\-]{1,12}", line):
#        if len(token) < 2:
#            continue
#        best = process.extractOne(
#            token.lower(), [a for a, _ in _ALL_ALIASES], scorer=fuzz.ratio
#        )
#        if best and best[1] >= _FUZZY_DEGREE_THRESHOLD:
#            matched_alias = best[0]
#            canonical = next(c for a, c in _ALL_ALIASES if a == matched_alias)
#            return canonical, _DEGREE_CATEGORY[canonical], None, None
#    return None, None, None, None
#
#
#_FIELD_PAREN_RE = re.compile(r'^\(([^)]+)\)')
#_FIELD_COLON_RE = re.compile(r'^:\s*(.+)')
#_FIELD_CONNECTOR_RE = re.compile(r'^(?:in|of)\s+(.+)', re.IGNORECASE)
#_FIELD_DASH_RE = re.compile(r'^[-\u2013]\s*(.+)')
#
#
#def _extract_degree_field(trigger_line: str, match_end: int | None) -> str | None:
#    """Pulls the specialization/field out of the same line as the degree
#    token, e.g. 'B.E.: I.T' -> 'I.T', 'M.E. (Mechanical Engg.)' -> 'Mechanical
#    Engg.', 'B.Tech in Computer Science' -> 'Computer Science'. Only looks at
#    text immediately after the degree token — if nothing recognizable follows
#    (no colon/parens/in/of/dash), we don't guess. A parenthetical that's just
#    a bare year ('(2019)') or a restatement of the degree abbreviation itself
#    ('Bachelor of Technology (B.tech)') isn't a field — skip it and look past it."""
#    if match_end is None:
#        return None
#    remainder = trigger_line[match_end:].strip()
#    if not remainder:
#        return None
#
#    m = _FIELD_PAREN_RE.match(remainder)
#    if m:
#        content = m.group(1).strip()
#        is_bare_year = re.fullmatch(r'(19|20)\d{2}', content)
#        is_degree_restatement = bool(_DEGREE_MATCHER.extract_keywords(content))
#        if not is_bare_year and not is_degree_restatement:
#            return content or None
#        # Skip past this parenthetical and look for a field marker after it
#        remainder = remainder[m.end():].strip()
#        if not remainder:
#            return None
#
#    field = None
#    for pattern in (_FIELD_COLON_RE, _FIELD_CONNECTOR_RE, _FIELD_DASH_RE):
#        m = pattern.match(remainder)
#        if m:
#            field = m.group(1)
#            break
#    if field is None:
#        return None
#
#    # Stop at the first comma/bullet/pipe — usually separates field from
#    # institution/city or from ranking blurbs ("Finance • Top 10% of 572")
#    field = re.split(r'[,•|]', field)[0].strip(" .")
#    return field or None
#
## Year — plain year, or a fuller date like "23 Sep 2011" / "Jan 2018"
#YEAR_RE = re.compile(
#    r'\b(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+'
#    r'(19[6-9]\d|20[0-3]\d)\b|\b(19[6-9]\d|20[0-3]\d)\b',
#    re.IGNORECASE
#)
#
## Full date range, e.g. "Aug 2019 - June 2023" or "2019 – 2023" or
## "2019 to 2023". Tried BEFORE the single-year fallback in _extract_year(),
## since resumes usually give the whole span and returning only the end
## year (the old behavior) throws away real information the user wants
## preserved. Each side reuses the same "optional month + year" shape as
## YEAR_RE so it matches the same range of date styles.
#_DATE_TOKEN_RE = (
#    r'(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+)?'
#    r'(?:19[6-9]\d|20[0-3]\d)'
#)
#DATE_RANGE_RE = re.compile(
#    rf'({_DATE_TOKEN_RE})\s*(?:[-–—]|to)\s*({_DATE_TOKEN_RE})',
#    re.IGNORECASE
#)
#
## GPA — keyword-led first ("CGPA: 8.16 /10", "Percentage:- 89%")
#GPA_KEYWORD_RE = re.compile(
#    r'(?:gpa|cgpa|sgpa|percentage|grade|score|marks)[:\s\-]+([0-9.]+(?:\s*/\s*[0-9.]+)?%?)',
#    re.IGNORECASE
#)
## Reversed order — number BEFORE the keyword ("7.8 CGPA", "7.8 SGPA")
#GPA_KEYWORD_REVERSED_RE = re.compile(
#    r'\b([0-9.]+)\s*(?:gpa|cgpa|sgpa)\b', re.IGNORECASE
#)
## Fallback: bare "8.16 /10" style CGPA with no keyword in front
#GPA_BARE_SCALE_RE = re.compile(r'\b(\d(?:\.\d{1,2})?)\s*/\s*(10|4)\b')
## Fallback: bare percentage with no keyword, e.g. "62.46%" — but NOT a class-rank
## phrase like "Top 10% of 572" or "Top 2% in Finance", which isn't a GPA at all.
#GPA_BARE_PERCENT_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\s*%')
#_RANKING_PHRASE_RE = re.compile(r'\btop\b', re.IGNORECASE)
#
## Institution keyword fallback now uses _INSTITUTION_KEYWORDS_RE defined
## alongside _match_keyword_fallback() below (anchor + expand, not one greedy regex).
#
#
#def _load_institutions_db() -> dict:
#    """Loads data/institutions_db.json, built once by build_institutions_db.py
#    from the AISHE colleges dataset + Hipolabs India universities list."""
#    db_path = Path("data/institutions_db.json")
#    if db_path.exists():
#        with open(db_path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {"full_names": [], "acronym_index": {}}
#
#
#_INSTITUTIONS_DB = _load_institutions_db()
#
## flashtext processor for whole-phrase, case-insensitive full-name matching.
## Full names rarely collide across institutions, so this tier is high-confidence.
#_FULL_NAME_MATCHER = KeywordProcessor(case_sensitive=False)
#for _name in _INSTITUTIONS_DB.get("full_names", []):
#    _FULL_NAME_MATCHER.add_keyword(_name)
#
#
#_LEADING_WORD_RE = re.compile(r"^[A-Z][a-z.&'\-]*$|^[A-Z]{2,20}$")  # Title-Case OR a plausible all-caps word
#_CONNECTOR_WORDS = {"of", "the", "and", "for"}
#
## All-caps tokens that are degree/GPA jargon, not institution-name acronyms —
## these should never be swept into an institution name during expansion.
#_NON_INSTITUTION_ACRONYMS = {"CGPA", "SGPA", "GPA", "SSC", "HSC", "PGDM", "PGDCA", "PGDBA"} | {
#    a.upper().replace(".", "") for a, _ in _ALL_ALIASES if a.isupper() or "." in a
#}
#
#
#def _is_expandable_word(tok: str) -> bool:
#    if not _LEADING_WORD_RE.match(tok):
#        return False
#    if tok.upper().replace(".", "") in _NON_INSTITUTION_ACRONYMS:
#        return False
#    return True
#
#
#def _expand_left(entry_text: str, start: int) -> list:
#    """Expands leftward from `start` over Title-Case words/connectors, but
#    guards against sweeping in a specialization word — e.g. in 'PGDM in
#    Finance IIM Ahmedabad', 'Finance' would otherwise get pulled into the
#    institution name just because it's Title-Case. If exactly one word
#    would be pulled in and it's directly preceded by 'in', that's a
#    specialization clause ('degree in X'), not part of the institution."""
#    before = entry_text[:start].rstrip()
#    tokens_before = before.split(" ") if before else []
#    extended = []
#    stop_token = None
#    for tok in reversed(tokens_before):
#        if _is_expandable_word(tok) or tok.lower() in _CONNECTOR_WORDS:
#            extended.insert(0, tok)
#        else:
#            stop_token = tok
#            break
#    if len(extended) == 1 and stop_token and stop_token.lower() == "in":
#        return []
#    return extended
#
#
#def _expand_right(entry_text: str, end: int) -> list:
#    """Expands rightward from `end` over Title-Case words/connectors, stopping
#    at a comma/paren/digit. Fixes DB hits like 'Indian Institute of
#    Management' truncating the real name 'Indian Institute of Management
#    Calcutta' (connector 'of' must be allowed through, not just Title-Case
#    words, or the expansion stops one word too early)."""
#    after = entry_text[end:].lstrip()
#    tokens_after = after.split(" ") if after else []
#    extended = []
#    for tok in tokens_after:
#        clean = tok.rstrip(",")
#        if _is_expandable_word(clean) or clean.lower() in _CONNECTOR_WORDS:
#            extended.append(clean)
#            if clean != tok:  # trailing comma — stop after including this word
#                break
#        else:
#            break
#    return extended
#
#
#def _match_full_name(entry_text: str) -> str | None:
#    hits = _FULL_NAME_MATCHER.extract_keywords(entry_text, span_info=True)
#    if not hits:
#        return None
#    # Longest span wins among direct hits (more specific match).
#    keyword, start, end = max(hits, key=lambda h: h[2] - h[1])
#    left = _expand_left(entry_text, start)
#    right = _expand_right(entry_text, end)
#    prefix = " ".join(left)
#    suffix = " ".join(right)
#    parts = [p for p in (prefix, keyword, suffix) if p]
#    return " ".join(parts).strip()
#
#
#_INSTITUTION_KEYWORDS_SAFE_RE = re.compile(
#    r'\b(university|institute|institution|college|school|board|iit|nit|bits|iim|ignou)\b',
#    re.IGNORECASE
#)
#_INSTITUTION_KEYWORDS_RISKY_RE = re.compile(r'\b(academy)\b', re.IGNORECASE)
## Kept for _extract_institution's final-string matching, where both matter equally
#_INSTITUTION_KEYWORDS_RE = re.compile(
#    r'\b(university|institute|institution|academy|college|school|board|iit|nit|bits|iim|ignou)\b',
#    re.IGNORECASE
#)
#_BARE_GENERIC_INSTITUTION = {"school", "college", "university", "institute", "board"}
#
#
#def _match_keyword_fallback(entry_text: str) -> str | None:
#    m = _INSTITUTION_KEYWORDS_RE.search(entry_text)
#    if not m:
#        return None
#    start, end = m.start(), m.end()
#    left = _expand_left(entry_text, start)
#    right = _expand_right(entry_text, end)
#
#    parts = left + [m.group(0)] + right
#    candidate = " ".join(parts).strip()
#    return candidate or None
#
#
#def _extract_institution(entry_text: str) -> tuple[str | None, bool]:
#    """Returns (institution, needs_review). Priority: full name in DB (safest)
#    -> generic keyword regex (not DB-verified, but usually accurate when a
#    resume spells the name out in full).
#
#    There used to be a third tier here: matching bare acronyms (SVIT, EIILM,
#    IIMT...) against a DB index built from institutions' generated initials.
#    Removed after testing showed every real acronym-tier match came back
#    flagged needs_review anyway (unique-in-DB never proved correct-in-reality
#    — see e.g. "SVIT" resolving to a real but wrong college in Telangana).
#    Since a flagged guess can still be misused by anything downstream that
#    doesn't check the flag, and None cannot be, this trades a small amount of
#    partial coverage (pure-acronym mentions with no institution keyword
#    nearby) for eliminating an entire recurring class of false-confidence
#    bugs (2-letter state-code collisions, cross-state acronym collisions)."""
#    full = _match_full_name(entry_text)
#    if full:
#        return full, False
#
#    fallback = _match_keyword_fallback(entry_text)
#    if fallback:
#        # A bare generic word with nothing else ("School" alone, no name
#        # attached) usually means the real name got masked out along with
#        # the degree phrase on the same line (e.g. "Senior Secondary
#        # School" where the school's own name IS "Senior Secondary
#        # School") — not enough to trust confidently.
#        needs_review = fallback.strip().lower() in _BARE_GENERIC_INSTITUTION
#        return fallback, needs_review
#
#    return None, False
#
#
#def _extract_year(entry_text: str, trigger_line: str | None = None) -> str | None:
#    # Try the trigger line first — it's the most tightly-scoped, reliable
#    # source for this entry's own date(s), and avoids picking up unrelated
#    # dates from other lines that got swept into the block (e.g. a birth
#    # date from a mis-segmented trailing "Personal Details" section).
#    #
#    # Look for a full range first ("Aug 2019 - June 2023") before falling
#    # back to a single year — returning only the end year silently threw
#    # away the start date even when the resume stated the full range.
#    if trigger_line:
#        range_match = DATE_RANGE_RE.search(trigger_line)
#        if range_match:
#            return f"{range_match.group(1).strip()} - {range_match.group(2).strip()}"
#        trigger_matches = list(YEAR_RE.finditer(trigger_line))
#        if trigger_matches:
#            m = trigger_matches[-1]
#            return m.group(1) or m.group(2)
#
#    # Fall back to the full joined entry text (covers cases like Adarsh's
#    # resume, where the date range sits on the institution line rather
#    # than the trigger/degree line).
#    range_match = DATE_RANGE_RE.search(entry_text)
#    if range_match:
#        return f"{range_match.group(1).strip()} - {range_match.group(2).strip()}"
#
#    # Last resort: single year — prefer the LAST year found, since resumes
#    # write "2010 – 2014" and the end of the range is the graduation/
#    # completion year, not the start year. This path only fires now when
#    # no recognizable range exists at all (e.g. a bare "2023" with no
#    # start date given anywhere).
#    matches = list(YEAR_RE.finditer(entry_text))
#    if not matches:
#        return None
#    m = matches[-1]
#    return m.group(1) or m.group(2)
#
#
#def _extract_gpa(line_text: str) -> str | None:
#    m = GPA_KEYWORD_RE.search(line_text)
#    if m:
#        return m.group(1).strip()
#    m = GPA_KEYWORD_REVERSED_RE.search(line_text)
#    if m:
#        return m.group(1).strip()
#    m = GPA_BARE_SCALE_RE.search(line_text)
#    if m:
#        return f"{m.group(1)}/{m.group(2)}"
#    for m in GPA_BARE_PERCENT_RE.finditer(line_text):
#        window_start = max(0, m.start() - 10)
#        if _RANKING_PHRASE_RE.search(line_text[window_start:m.start()]):
#            continue  # "Top 10% of 572" — a class rank, not a GPA
#        return f"{m.group(1)}%"
#    return None
#
#
#def _extract_degree(trigger_line: str, degree_type: str) -> str:
#    """Best-effort clean degree label from the line that triggered this entry.
#    Cuts off at the first comma so trailing institution text doesn't leak in
#    (e.g. 'B.E. (Mechanical Engg.) SVIT, Pune' -> 'B.E. (Mechanical Engg.) SVIT'
#    — imperfect but keeps the label short; institution field carries the real name)."""
#    return trigger_line.strip()
#
#
#def _line_has_institution_signal(line: str) -> bool:
#    """Lightweight per-line check used only to decide entry boundaries — does
#    this single line look like it's introducing an institution mention?
#    Doesn't need to reconstruct the full name; that happens later via
#    _extract_institution() on the complete joined block.
#
#    Most institution keywords (university, institute, college, school...)
#    essentially never show up in ordinary prose within an education section,
#    so a simple presence check is fine. "academy" is the exception — it can
#    appear incidentally in a sentence ("...Student association academy") —
#    so it additionally requires a capitalized (proper-noun-shaped) neighbor
#    before counting as a real signal."""
#    if _INSTITUTION_KEYWORDS_SAFE_RE.search(line):
#        return True
#
#    m = _INSTITUTION_KEYWORDS_RISKY_RE.search(line)
#    if m:
#        before_tokens = line[:m.start()].rstrip().split(" ") if m.start() > 0 else []
#        after_tokens = line[m.end():].lstrip().split(" ") if m.end() < len(line) else []
#        before = before_tokens[-1] if before_tokens else ""
#        after = after_tokens[0] if after_tokens else ""
#        if after.lower().rstrip(",") in _CONNECTOR_WORDS and len(after_tokens) > 1:
#            after = after_tokens[1]
#        before_clean = before.strip(" ,.-")
#        after_clean = after.strip(" ,.-")
#        neighbor_capitalized = bool(
#            (before_clean and before_clean[0].isupper()) or
#            (after_clean and after_clean[0].isupper())
#        )
#        if neighbor_capitalized:
#            return True
#
#    if _FULL_NAME_MATCHER.extract_keywords(line):
#        return True
#    return False
#
#
#_EXPLICIT_LEVEL_RE = re.compile(r'\((10|12)\)|\b(10|12)th\b', re.IGNORECASE)
#
#
#def _explicit_school_level(text: str) -> str | None:
#    """Some resumes write contradictory things like 'Higher Secondary School
#    (10)' — the English name says 12th, but the explicit (10) means the
#    person means 10th. When an unambiguous (10)/(12)/'10th'/'12th' marker is
#    present, it should win over the fuzzy English-name-based classification.
#    Deliberately narrow (paren-wrapped or 'th'-suffixed only) so it doesn't
#    false-trigger on '8.16 /10' style GPA-out-of-10 scales."""
#    m = _EXPLICIT_LEVEL_RE.search(text)
#    if not m:
#        return None
#    digits = m.group(1) or m.group(2)
#    return f"{digits}th"
#
#
#def _flush_entry(current: dict) -> dict | None:
#    """Finalizes one accumulated entry into an output record, or returns None
#    if it has nothing usable (pure noise lines with no degree or institution
#    signal at all)."""
#    if not current["lines"]:
#        return None
#
#    # Mask out the exact degree-token substring (not the whole line!) before
#    # institution search — otherwise a word like "School" inside the degree
#    # phrase "High School" gets mistaken for the institution's own name,
#    # while real institution text on the same line ("TDS School, Aligarh")
#    # would still need to survive the mask.
#    entry_text = " ".join(current["lines"])
#    if current["trigger_line_index"] is not None and current["degree_start"] is not None:
#        offset = len(" ".join(current["lines"][:current["trigger_line_index"]]))
#        if offset:
#            offset += 1  # the joining space
#        abs_start = offset + current["degree_start"]
#        abs_end = offset + current["degree_end"]
#        entry_text = entry_text[:abs_start] + (" " * (abs_end - abs_start)) + entry_text[abs_end:]
#
#    institution, needs_review = _extract_institution(entry_text)
#    year = _extract_year(entry_text, current["trigger_line"])
#    gpa = _extract_gpa(entry_text)
#
#    degree_name = current["degree_name"]
#    degree_type = current["degree_category"]
#    degree = current["trigger_line"].strip() if current["trigger_line"] else None
#    degree_field = _extract_degree_field(current["trigger_line"], current["degree_end"]) \
#        if current["trigger_line"] else None
#
#    # Some resumes write contradictory things like "Higher Secondary School (10)"
#    # — an explicit (10)/(12)/"10th"/"12th" marker overrides the fuzzy
#    # English-name-based classification, and the contradiction itself is a
#    # signal this entry deserves a second look.
#    explicit_level = _explicit_school_level(entry_text)
#    if explicit_level and degree_type in ("10th", "12th") and explicit_level != degree_type:
#        degree_type = explicit_level
#        degree_name = explicit_level
#        needs_review = True
#
#    if not degree_name and not institution:
#        return None  # nothing usable at all — pure noise lines
#
#    if not degree_name and institution and institution.strip().lower() in _BARE_GENERIC_INSTITUTION:
#        return None  # e.g. a table header row ("Degree/Exam Board/Institute Year") — not a real entry
#
#    if not degree_name:
#        # Institution-only entry (e.g. an executive program with no
#        # recognized degree word) — still worth keeping, just flagged.
#        needs_review = True
#
#    fields_found = sum(1 for v in (degree_type, institution, year) if v)
#    base_confidence = min(0.5 + 0.15 * fields_found, 0.95)
#    if needs_review:
#        base_confidence = min(base_confidence, 0.55)
#
#    return {
#        "degree": degree,
#        "degree_name": degree_name,
#        "degree_type": degree_type,
#        "degree_field": degree_field,
#        "institution": institution,
#        "year": year,
#        "gpa": gpa,
#        "_needs_review": needs_review,
#        "_confidence": round(base_confidence, 2),
#    }
#
#
#def _new_current() -> dict:
#    return {
#        "lines": [],
#        "degree_name": None,
#        "degree_category": None,
#        "trigger_line": None,
#        "trigger_line_index": None,
#        "degree_start": None,
#        "degree_end": None,
#        "has_institution_signal": False,
#        "last_institution_line_idx": None,
#    }
#
#
#def extract_education(education_section_text: str) -> dict:
#    """
#    Parses the education section into a list of entries.
#
#    Rather than writing a separate rule for every ordering resumes use
#    (degree-then-college, college-then-degree, same-line, across-lines,
#    with/without location...), each line is classified for two signals —
#    "does this line introduce a degree?" and "does this line introduce an
#    institution?" — and entries are flushed whenever a line would introduce
#    a SECOND degree or a SECOND institution into the entry currently being
#    built. This one mechanism covers every ordering: it doesn't matter
#    whether the degree or the institution comes first, only that a repeat
#    of either one means we've moved on to the next entry. Lines with neither
#    signal (bullet points, job-description-style prose under an education
#    entry) are just carried along as context for GPA/year extraction.
#    """
#    if not education_section_text.strip():
#        return {"education": [], "_confidence": 0.0}
#
#    lines = [l.strip() for l in education_section_text.strip().split("\n") if l.strip()]
#
#    education_entries = []
#    current = _new_current()
#
#    for line_idx, line in enumerate(lines):
#        canonical, category, degree_start, degree_end = match_degree(line)
#        has_inst_signal = _line_has_institution_signal(line)
#
#        institution_conflict = False
#        if has_inst_signal and current["has_institution_signal"]:
#            last_idx = current["last_institution_line_idx"]
#            adjacent = last_idx is not None and (line_idx - last_idx) == 1
#            # Adjacent lines both carrying institution signal are usually one
#            # institution name wrapped across a line break (e.g. "SVIT, Pune"
#            # then "University ,23 Sep" — together "SVIT, Pune University"),
#            # UNLESS this line also introduces its own degree word — that's
#            # the tell that it's actually a new entry starting, not a
#            # continuation (e.g. "...Indian Institute" then "MS in Cyber Law
#            # ... National Law Institute" are two different entries, even
#            # though both lines happen to carry institution signal back to back).
#            if adjacent and not canonical:
#                institution_conflict = False
#            else:
#                institution_conflict = True
#
#        conflict = (canonical and current["degree_name"]) or institution_conflict
#        if conflict:
#            finalized = _flush_entry(current)
#            if finalized:
#                education_entries.append(finalized)
#            current = _new_current()
#
#        current["lines"].append(line)
#        if canonical and not current["degree_name"]:
#            current["degree_name"] = canonical
#            current["degree_category"] = category
#            current["trigger_line"] = line
#            current["trigger_line_index"] = len(current["lines"]) - 1
#            current["degree_start"] = degree_start
#            current["degree_end"] = degree_end
#        if has_inst_signal:
#            current["has_institution_signal"] = True
#            current["last_institution_line_idx"] = line_idx
#
#    finalized = _flush_entry(current)
#    if finalized:
#        education_entries.append(finalized)
#
#    overall_confidence = round(
#        sum(e["_confidence"] for e in education_entries) / len(education_entries), 2
#    ) if education_entries else 0.0
#
#    return {
#        "education": education_entries,
#        "_confidence": overall_confidence,
#    }
#
#







## Worked- final one changing just for a date range - pasting above and more improved one - 1st one above(pasted 2 improved code above)
#"""
#Education Extractor — Layer 2
#Extracts: degree name, degree type, institution, year of completion, GPA/percentage.
#
#Design notes (why this differs from a naive line-by-line parser):
#- Real resumes wrap education entries across lines inconsistently. Sometimes the
#  institution appears BEFORE the degree line (e.g. "VESIT, Mumbai, 2008" then
#  "B.E.: I.T"). Sometimes two entries sit back-to-back with NO blank line between
#  them (e.g. an M.E. entry immediately followed by a B.E. entry).
#- Because of that, entries are NOT split on blank lines. They're split on
#  "trigger lines" — lines that match a degree pattern. Lines before the first
#  trigger attach to entry 1; lines after a trigger attach to that entry until the
#  next trigger appears. This handles both orderings above correctly.
#- Each entry's lines are joined into a single string before running the
#  institution/year/GPA regexes, so a name wrapped across two lines
#  ("... Mumbai" / "University, Mumbai Jan 2018") is still matched as one phrase.
#"""
#
#import json
#import re
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import fuzz, process
#
## ─────────────────────────────────────────────────────────────
## DEGREE PATTERNS (checked in priority order — first match wins)
## ─────────────────────────────────────────────────────────────
#
#def _load_degree_taxonomy() -> dict:
#    """Loads data/degree_taxonomy.json, built once by build_degree_taxonomy.py.
#    Unlike institutions, degree names are a closed, well-known vocabulary —
#    same curated-taxonomy-plus-fuzzy-match pattern your skills extractor uses,
#    rather than a handful of broad regex buckets."""
#    path = Path("data/degree_taxonomy.json")
#    if path.exists():
#        with open(path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {}
#
#
#_DEGREE_TAXONOMY = _load_degree_taxonomy()
#
## flashtext: alias (lowercase) -> canonical degree name, e.g. "b.tech" -> "B.Tech"
#_DEGREE_MATCHER = KeywordProcessor(case_sensitive=False)
#_DEGREE_CATEGORY = {}          # canonical name -> category (bachelors/masters/...)
#_ALL_ALIASES = []              # flat list, used for the fuzzy fallback
#
#for _canonical, _info in _DEGREE_TAXONOMY.items():
#    _DEGREE_CATEGORY[_canonical] = _info["category"]
#    for _alias in _info["aliases"]:
#        _DEGREE_MATCHER.add_keyword(_alias, _canonical)
#        _ALL_ALIASES.append((_alias, _canonical))
#
#_FUZZY_DEGREE_THRESHOLD = 85
#
#
#def match_degree(line: str) -> tuple[str, str, int, int] | tuple[None, None, None, None]:
#    """Returns (canonical_degree_name, category, match_start, match_end) for
#    the first degree found in a line, or all-None. The span lets us find
#    specialization text right after the degree token, and mask the degree
#    token itself out before institution search (so e.g. 'School' inside
#    'High School' doesn't get mistaken for the institution's own name)."""
#    hits = _DEGREE_MATCHER.extract_keywords(line, span_info=True)
#    if hits:
#        canonical, start, end = hits[0]
#        return canonical, _DEGREE_CATEGORY[canonical], start, end
#
#    # Fuzzy fallback — only worth trying on short tokens (a handful of words),
#    # since fuzzy-matching a whole sentence against "B.Tech" is meaningless.
#    # No reliable span here, so field extraction/masking is skipped for fuzzy matches.
#    for token in re.findall(r"[A-Za-z][A-Za-z.\-]{1,12}", line):
#        if len(token) < 2:
#            continue
#        best = process.extractOne(
#            token.lower(), [a for a, _ in _ALL_ALIASES], scorer=fuzz.ratio
#        )
#        if best and best[1] >= _FUZZY_DEGREE_THRESHOLD:
#            matched_alias = best[0]
#            canonical = next(c for a, c in _ALL_ALIASES if a == matched_alias)
#            return canonical, _DEGREE_CATEGORY[canonical], None, None
#    return None, None, None, None
#
#
#_FIELD_PAREN_RE = re.compile(r'^\(([^)]+)\)')
#_FIELD_COLON_RE = re.compile(r'^:\s*(.+)')
#_FIELD_CONNECTOR_RE = re.compile(r'^(?:in|of)\s+(.+)', re.IGNORECASE)
#_FIELD_DASH_RE = re.compile(r'^[-\u2013]\s*(.+)')
#
#
#def _extract_degree_field(trigger_line: str, match_end: int | None) -> str | None:
#    """Pulls the specialization/field out of the same line as the degree
#    token, e.g. 'B.E.: I.T' -> 'I.T', 'M.E. (Mechanical Engg.)' -> 'Mechanical
#    Engg.', 'B.Tech in Computer Science' -> 'Computer Science'. Only looks at
#    text immediately after the degree token — if nothing recognizable follows
#    (no colon/parens/in/of/dash), we don't guess. A parenthetical that's just
#    a bare year ('(2019)') or a restatement of the degree abbreviation itself
#    ('Bachelor of Technology (B.tech)') isn't a field — skip it and look past it."""
#    if match_end is None:
#        return None
#    remainder = trigger_line[match_end:].strip()
#    if not remainder:
#        return None
#
#    m = _FIELD_PAREN_RE.match(remainder)
#    if m:
#        content = m.group(1).strip()
#        is_bare_year = re.fullmatch(r'(19|20)\d{2}', content)
#        is_degree_restatement = bool(_DEGREE_MATCHER.extract_keywords(content))
#        if not is_bare_year and not is_degree_restatement:
#            return content or None
#        # Skip past this parenthetical and look for a field marker after it
#        remainder = remainder[m.end():].strip()
#        if not remainder:
#            return None
#
#    field = None
#    for pattern in (_FIELD_COLON_RE, _FIELD_CONNECTOR_RE, _FIELD_DASH_RE):
#        m = pattern.match(remainder)
#        if m:
#            field = m.group(1)
#            break
#    if field is None:
#        return None
#
#    # Stop at the first comma/bullet/pipe — usually separates field from
#    # institution/city or from ranking blurbs ("Finance • Top 10% of 572")
#    field = re.split(r'[,•|]', field)[0].strip(" .")
#    return field or None
#
## Year — plain year, or a fuller date like "23 Sep 2011" / "Jan 2018"
#YEAR_RE = re.compile(
#    r'\b(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+'
#    r'(19[6-9]\d|20[0-3]\d)\b|\b(19[6-9]\d|20[0-3]\d)\b',
#    re.IGNORECASE
#)
#
## GPA — keyword-led first ("CGPA: 8.16 /10", "Percentage:- 89%")
#GPA_KEYWORD_RE = re.compile(
#    r'(?:gpa|cgpa|sgpa|percentage|grade|score|marks)[:\s\-]+([0-9.]+(?:\s*/\s*[0-9.]+)?%?)',
#    re.IGNORECASE
#)
## Reversed order — number BEFORE the keyword ("7.8 CGPA", "7.8 SGPA")
#GPA_KEYWORD_REVERSED_RE = re.compile(
#    r'\b([0-9.]+)\s*(?:gpa|cgpa|sgpa)\b', re.IGNORECASE
#)
## Fallback: bare "8.16 /10" style CGPA with no keyword in front
#GPA_BARE_SCALE_RE = re.compile(r'\b(\d(?:\.\d{1,2})?)\s*/\s*(10|4)\b')
## Fallback: bare percentage with no keyword, e.g. "62.46%" — but NOT a class-rank
## phrase like "Top 10% of 572" or "Top 2% in Finance", which isn't a GPA at all.
#GPA_BARE_PERCENT_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\s*%')
#_RANKING_PHRASE_RE = re.compile(r'\btop\b', re.IGNORECASE)
#
## Institution keyword fallback now uses _INSTITUTION_KEYWORDS_RE defined
## alongside _match_keyword_fallback() below (anchor + expand, not one greedy regex).
#
#
#def _load_institutions_db() -> dict:
#    """Loads data/institutions_db.json, built once by build_institutions_db.py
#    from the AISHE colleges dataset + Hipolabs India universities list."""
#    db_path = Path("data/institutions_db.json")
#    if db_path.exists():
#        with open(db_path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {"full_names": [], "acronym_index": {}}
#
#
#_INSTITUTIONS_DB = _load_institutions_db()
#
## flashtext processor for whole-phrase, case-insensitive full-name matching.
## Full names rarely collide across institutions, so this tier is high-confidence.
#_FULL_NAME_MATCHER = KeywordProcessor(case_sensitive=False)
#for _name in _INSTITUTIONS_DB.get("full_names", []):
#    _FULL_NAME_MATCHER.add_keyword(_name)
#
#
#_LEADING_WORD_RE = re.compile(r"^[A-Z][a-z.&'\-]*$|^[A-Z]{2,20}$")  # Title-Case OR a plausible all-caps word
#_CONNECTOR_WORDS = {"of", "the", "and", "for"}
#
## All-caps tokens that are degree/GPA jargon, not institution-name acronyms —
## these should never be swept into an institution name during expansion.
#_NON_INSTITUTION_ACRONYMS = {"CGPA", "SGPA", "GPA", "SSC", "HSC", "PGDM", "PGDCA", "PGDBA"} | {
#    a.upper().replace(".", "") for a, _ in _ALL_ALIASES if a.isupper() or "." in a
#}
#
#
#def _is_expandable_word(tok: str) -> bool:
#    if not _LEADING_WORD_RE.match(tok):
#        return False
#    if tok.upper().replace(".", "") in _NON_INSTITUTION_ACRONYMS:
#        return False
#    return True
#
#
#def _expand_left(entry_text: str, start: int) -> list:
#    """Expands leftward from `start` over Title-Case words/connectors, but
#    guards against sweeping in a specialization word — e.g. in 'PGDM in
#    Finance IIM Ahmedabad', 'Finance' would otherwise get pulled into the
#    institution name just because it's Title-Case. If exactly one word
#    would be pulled in and it's directly preceded by 'in', that's a
#    specialization clause ('degree in X'), not part of the institution."""
#    before = entry_text[:start].rstrip()
#    tokens_before = before.split(" ") if before else []
#    extended = []
#    stop_token = None
#    for tok in reversed(tokens_before):
#        if _is_expandable_word(tok) or tok.lower() in _CONNECTOR_WORDS:
#            extended.insert(0, tok)
#        else:
#            stop_token = tok
#            break
#    if len(extended) == 1 and stop_token and stop_token.lower() == "in":
#        return []
#    return extended
#
#
#def _expand_right(entry_text: str, end: int) -> list:
#    """Expands rightward from `end` over Title-Case words/connectors, stopping
#    at a comma/paren/digit. Fixes DB hits like 'Indian Institute of
#    Management' truncating the real name 'Indian Institute of Management
#    Calcutta' (connector 'of' must be allowed through, not just Title-Case
#    words, or the expansion stops one word too early)."""
#    after = entry_text[end:].lstrip()
#    tokens_after = after.split(" ") if after else []
#    extended = []
#    for tok in tokens_after:
#        clean = tok.rstrip(",")
#        if _is_expandable_word(clean) or clean.lower() in _CONNECTOR_WORDS:
#            extended.append(clean)
#            if clean != tok:  # trailing comma — stop after including this word
#                break
#        else:
#            break
#    return extended
#
#
#def _match_full_name(entry_text: str) -> str | None:
#    hits = _FULL_NAME_MATCHER.extract_keywords(entry_text, span_info=True)
#    if not hits:
#        return None
#    # Longest span wins among direct hits (more specific match).
#    keyword, start, end = max(hits, key=lambda h: h[2] - h[1])
#    left = _expand_left(entry_text, start)
#    right = _expand_right(entry_text, end)
#    prefix = " ".join(left)
#    suffix = " ".join(right)
#    parts = [p for p in (prefix, keyword, suffix) if p]
#    return " ".join(parts).strip()
#
#
#_INSTITUTION_KEYWORDS_SAFE_RE = re.compile(
#    r'\b(university|institute|institution|college|school|board|iit|nit|bits|iim|ignou)\b',
#    re.IGNORECASE
#)
#_INSTITUTION_KEYWORDS_RISKY_RE = re.compile(r'\b(academy)\b', re.IGNORECASE)
## Kept for _extract_institution's final-string matching, where both matter equally
#_INSTITUTION_KEYWORDS_RE = re.compile(
#    r'\b(university|institute|institution|academy|college|school|board|iit|nit|bits|iim|ignou)\b',
#    re.IGNORECASE
#)
#_BARE_GENERIC_INSTITUTION = {"school", "college", "university", "institute", "board"}
#
#
#def _match_keyword_fallback(entry_text: str) -> str | None:
#    m = _INSTITUTION_KEYWORDS_RE.search(entry_text)
#    if not m:
#        return None
#    start, end = m.start(), m.end()
#    left = _expand_left(entry_text, start)
#    right = _expand_right(entry_text, end)
#
#    parts = left + [m.group(0)] + right
#    candidate = " ".join(parts).strip()
#    return candidate or None
#
#
#def _extract_institution(entry_text: str) -> tuple[str | None, bool]:
#    """Returns (institution, needs_review). Priority: full name in DB (safest)
#    -> generic keyword regex (not DB-verified, but usually accurate when a
#    resume spells the name out in full).
#
#    There used to be a third tier here: matching bare acronyms (SVIT, EIILM,
#    IIMT...) against a DB index built from institutions' generated initials.
#    Removed after testing showed every real acronym-tier match came back
#    flagged needs_review anyway (unique-in-DB never proved correct-in-reality
#    — see e.g. "SVIT" resolving to a real but wrong college in Telangana).
#    Since a flagged guess can still be misused by anything downstream that
#    doesn't check the flag, and None cannot be, this trades a small amount of
#    partial coverage (pure-acronym mentions with no institution keyword
#    nearby) for eliminating an entire recurring class of false-confidence
#    bugs (2-letter state-code collisions, cross-state acronym collisions)."""
#    full = _match_full_name(entry_text)
#    if full:
#        return full, False
#
#    fallback = _match_keyword_fallback(entry_text)
#    if fallback:
#        # A bare generic word with nothing else ("School" alone, no name
#        # attached) usually means the real name got masked out along with
#        # the degree phrase on the same line (e.g. "Senior Secondary
#        # School" where the school's own name IS "Senior Secondary
#        # School") — not enough to trust confidently.
#        needs_review = fallback.strip().lower() in _BARE_GENERIC_INSTITUTION
#        return fallback, needs_review
#
#    return None, False
#
#
#def _extract_year(entry_text: str, trigger_line: str | None = None) -> str | None:
#    # Try the trigger line first — it's the most tightly-scoped, reliable
#    # source for this entry's own year, and avoids picking up unrelated
#    # years from other lines that got swept into the block (e.g. a birth
#    # date from a mis-segmented trailing "Personal Details" section).
#    if trigger_line:
#        trigger_matches = list(YEAR_RE.finditer(trigger_line))
#        if trigger_matches:
#            m = trigger_matches[-1]
#            return m.group(1) or m.group(2)
#
#    # Prefer the LAST year found — resumes write "2010 – 2014", and the end
#    # of the range is the graduation/completion year, not the start year.
#    matches = list(YEAR_RE.finditer(entry_text))
#    if not matches:
#        return None
#    m = matches[-1]
#    return m.group(1) or m.group(2)
#
#
#def _extract_gpa(line_text: str) -> str | None:
#    m = GPA_KEYWORD_RE.search(line_text)
#    if m:
#        return m.group(1).strip()
#    m = GPA_KEYWORD_REVERSED_RE.search(line_text)
#    if m:
#        return m.group(1).strip()
#    m = GPA_BARE_SCALE_RE.search(line_text)
#    if m:
#        return f"{m.group(1)}/{m.group(2)}"
#    for m in GPA_BARE_PERCENT_RE.finditer(line_text):
#        window_start = max(0, m.start() - 10)
#        if _RANKING_PHRASE_RE.search(line_text[window_start:m.start()]):
#            continue  # "Top 10% of 572" — a class rank, not a GPA
#        return f"{m.group(1)}%"
#    return None
#
#
#def _extract_degree(trigger_line: str, degree_type: str) -> str:
#    """Best-effort clean degree label from the line that triggered this entry.
#    Cuts off at the first comma so trailing institution text doesn't leak in
#    (e.g. 'B.E. (Mechanical Engg.) SVIT, Pune' -> 'B.E. (Mechanical Engg.) SVIT'
#    — imperfect but keeps the label short; institution field carries the real name)."""
#    return trigger_line.strip()
#
#
#def _line_has_institution_signal(line: str) -> bool:
#    """Lightweight per-line check used only to decide entry boundaries — does
#    this single line look like it's introducing an institution mention?
#    Doesn't need to reconstruct the full name; that happens later via
#    _extract_institution() on the complete joined block.
#
#    Most institution keywords (university, institute, college, school...)
#    essentially never show up in ordinary prose within an education section,
#    so a simple presence check is fine. "academy" is the exception — it can
#    appear incidentally in a sentence ("...Student association academy") —
#    so it additionally requires a capitalized (proper-noun-shaped) neighbor
#    before counting as a real signal."""
#    if _INSTITUTION_KEYWORDS_SAFE_RE.search(line):
#        return True
#
#    m = _INSTITUTION_KEYWORDS_RISKY_RE.search(line)
#    if m:
#        before_tokens = line[:m.start()].rstrip().split(" ") if m.start() > 0 else []
#        after_tokens = line[m.end():].lstrip().split(" ") if m.end() < len(line) else []
#        before = before_tokens[-1] if before_tokens else ""
#        after = after_tokens[0] if after_tokens else ""
#        if after.lower().rstrip(",") in _CONNECTOR_WORDS and len(after_tokens) > 1:
#            after = after_tokens[1]
#        before_clean = before.strip(" ,.-")
#        after_clean = after.strip(" ,.-")
#        neighbor_capitalized = bool(
#            (before_clean and before_clean[0].isupper()) or
#            (after_clean and after_clean[0].isupper())
#        )
#        if neighbor_capitalized:
#            return True
#
#    if _FULL_NAME_MATCHER.extract_keywords(line):
#        return True
#    return False
#
#
#_EXPLICIT_LEVEL_RE = re.compile(r'\((10|12)\)|\b(10|12)th\b', re.IGNORECASE)
#
#
#def _explicit_school_level(text: str) -> str | None:
#    """Some resumes write contradictory things like 'Higher Secondary School
#    (10)' — the English name says 12th, but the explicit (10) means the
#    person means 10th. When an unambiguous (10)/(12)/'10th'/'12th' marker is
#    present, it should win over the fuzzy English-name-based classification.
#    Deliberately narrow (paren-wrapped or 'th'-suffixed only) so it doesn't
#    false-trigger on '8.16 /10' style GPA-out-of-10 scales."""
#    m = _EXPLICIT_LEVEL_RE.search(text)
#    if not m:
#        return None
#    digits = m.group(1) or m.group(2)
#    return f"{digits}th"
#
#
#def _flush_entry(current: dict) -> dict | None:
#    """Finalizes one accumulated entry into an output record, or returns None
#    if it has nothing usable (pure noise lines with no degree or institution
#    signal at all)."""
#    if not current["lines"]:
#        return None
#
#    # Mask out the exact degree-token substring (not the whole line!) before
#    # institution search — otherwise a word like "School" inside the degree
#    # phrase "High School" gets mistaken for the institution's own name,
#    # while real institution text on the same line ("TDS School, Aligarh")
#    # would still need to survive the mask.
#    entry_text = " ".join(current["lines"])
#    if current["trigger_line_index"] is not None and current["degree_start"] is not None:
#        offset = len(" ".join(current["lines"][:current["trigger_line_index"]]))
#        if offset:
#            offset += 1  # the joining space
#        abs_start = offset + current["degree_start"]
#        abs_end = offset + current["degree_end"]
#        entry_text = entry_text[:abs_start] + (" " * (abs_end - abs_start)) + entry_text[abs_end:]
#
#    institution, needs_review = _extract_institution(entry_text)
#    year = _extract_year(entry_text, current["trigger_line"])
#    gpa = _extract_gpa(entry_text)
#
#    degree_name = current["degree_name"]
#    degree_type = current["degree_category"]
#    degree = current["trigger_line"].strip() if current["trigger_line"] else None
#    degree_field = _extract_degree_field(current["trigger_line"], current["degree_end"]) \
#        if current["trigger_line"] else None
#
#    # Some resumes write contradictory things like "Higher Secondary School (10)"
#    # — an explicit (10)/(12)/"10th"/"12th" marker overrides the fuzzy
#    # English-name-based classification, and the contradiction itself is a
#    # signal this entry deserves a second look.
#    explicit_level = _explicit_school_level(entry_text)
#    if explicit_level and degree_type in ("10th", "12th") and explicit_level != degree_type:
#        degree_type = explicit_level
#        degree_name = explicit_level
#        needs_review = True
#
#    if not degree_name and not institution:
#        return None  # nothing usable at all — pure noise lines
#
#    if not degree_name and institution and institution.strip().lower() in _BARE_GENERIC_INSTITUTION:
#        return None  # e.g. a table header row ("Degree/Exam Board/Institute Year") — not a real entry
#
#    if not degree_name:
#        # Institution-only entry (e.g. an executive program with no
#        # recognized degree word) — still worth keeping, just flagged.
#        needs_review = True
#
#    fields_found = sum(1 for v in (degree_type, institution, year) if v)
#    base_confidence = min(0.5 + 0.15 * fields_found, 0.95)
#    if needs_review:
#        base_confidence = min(base_confidence, 0.55)
#
#    return {
#        "degree": degree,
#        "degree_name": degree_name,
#        "degree_type": degree_type,
#        "degree_field": degree_field,
#        "institution": institution,
#        "year": year,
#        "gpa": gpa,
#        "_needs_review": needs_review,
#        "_confidence": round(base_confidence, 2),
#    }
#
#
#def _new_current() -> dict:
#    return {
#        "lines": [],
#        "degree_name": None,
#        "degree_category": None,
#        "trigger_line": None,
#        "trigger_line_index": None,
#        "degree_start": None,
#        "degree_end": None,
#        "has_institution_signal": False,
#        "last_institution_line_idx": None,
#    }
#
#
#def extract_education(education_section_text: str) -> dict:
#    """
#    Parses the education section into a list of entries.
#
#    Rather than writing a separate rule for every ordering resumes use
#    (degree-then-college, college-then-degree, same-line, across-lines,
#    with/without location...), each line is classified for two signals —
#    "does this line introduce a degree?" and "does this line introduce an
#    institution?" — and entries are flushed whenever a line would introduce
#    a SECOND degree or a SECOND institution into the entry currently being
#    built. This one mechanism covers every ordering: it doesn't matter
#    whether the degree or the institution comes first, only that a repeat
#    of either one means we've moved on to the next entry. Lines with neither
#    signal (bullet points, job-description-style prose under an education
#    entry) are just carried along as context for GPA/year extraction.
#    """
#    if not education_section_text.strip():
#        return {"education": [], "_confidence": 0.0}
#
#    lines = [l.strip() for l in education_section_text.strip().split("\n") if l.strip()]
#
#    education_entries = []
#    current = _new_current()
#
#    for line_idx, line in enumerate(lines):
#        canonical, category, degree_start, degree_end = match_degree(line)
#        has_inst_signal = _line_has_institution_signal(line)
#
#        institution_conflict = False
#        if has_inst_signal and current["has_institution_signal"]:
#            last_idx = current["last_institution_line_idx"]
#            adjacent = last_idx is not None and (line_idx - last_idx) == 1
#            # Adjacent lines both carrying institution signal are usually one
#            # institution name wrapped across a line break (e.g. "SVIT, Pune"
#            # then "University ,23 Sep" — together "SVIT, Pune University"),
#            # UNLESS this line also introduces its own degree word — that's
#            # the tell that it's actually a new entry starting, not a
#            # continuation (e.g. "...Indian Institute" then "MS in Cyber Law
#            # ... National Law Institute" are two different entries, even
#            # though both lines happen to carry institution signal back to back).
#            if adjacent and not canonical:
#                institution_conflict = False
#            else:
#                institution_conflict = True
#
#        conflict = (canonical and current["degree_name"]) or institution_conflict
#        if conflict:
#            finalized = _flush_entry(current)
#            if finalized:
#                education_entries.append(finalized)
#            current = _new_current()
#
#        current["lines"].append(line)
#        if canonical and not current["degree_name"]:
#            current["degree_name"] = canonical
#            current["degree_category"] = category
#            current["trigger_line"] = line
#            current["trigger_line_index"] = len(current["lines"]) - 1
#            current["degree_start"] = degree_start
#            current["degree_end"] = degree_end
#        if has_inst_signal:
#            current["has_institution_signal"] = True
#            current["last_institution_line_idx"] = line_idx
#
#    finalized = _flush_entry(current)
#    if finalized:
#        education_entries.append(finalized)
#
#    overall_confidence = round(
#        sum(e["_confidence"] for e in education_entries) / len(education_entries), 2
#    ) if education_entries else 0.0
#
#    return {
#        "education": education_entries,
#        "_confidence": overall_confidence,
#    }
#
#










#just changing for one
#"""
#Education Extractor — Layer 2
#Extracts: degree name, degree type, institution, year of completion, GPA/percentage.
#
#Design notes (why this differs from a naive line-by-line parser):
#- Real resumes wrap education entries across lines inconsistently. Sometimes the
#  institution appears BEFORE the degree line (e.g. "VESIT, Mumbai, 2008" then
#  "B.E.: I.T"). Sometimes two entries sit back-to-back with NO blank line between
#  them (e.g. an M.E. entry immediately followed by a B.E. entry).
#- Because of that, entries are NOT split on blank lines. They're split on
#  "trigger lines" — lines that match a degree pattern. Lines before the first
#  trigger attach to entry 1; lines after a trigger attach to that entry until the
#  next trigger appears. This handles both orderings above correctly.
#- Each entry's lines are joined into a single string before running the
#  institution/year/GPA regexes, so a name wrapped across two lines
#  ("... Mumbai" / "University, Mumbai Jan 2018") is still matched as one phrase.
#"""
#
#import json
#import re
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import fuzz, process
#
## ─────────────────────────────────────────────────────────────
## DEGREE PATTERNS (checked in priority order — first match wins)
## ─────────────────────────────────────────────────────────────
#
#def _load_degree_taxonomy() -> dict:
#    """Loads data/degree_taxonomy.json, built once by build_degree_taxonomy.py.
#    Unlike institutions, degree names are a closed, well-known vocabulary —
#    same curated-taxonomy-plus-fuzzy-match pattern your skills extractor uses,
#    rather than a handful of broad regex buckets."""
#    path = Path("data/degree_taxonomy.json")
#    if path.exists():
#        with open(path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {}
#
#
#_DEGREE_TAXONOMY = _load_degree_taxonomy()
#
## flashtext: alias (lowercase) -> canonical degree name, e.g. "b.tech" -> "B.Tech"
#_DEGREE_MATCHER = KeywordProcessor(case_sensitive=False)
#_DEGREE_CATEGORY = {}          # canonical name -> category (bachelors/masters/...)
#_ALL_ALIASES = []              # flat list, used for the fuzzy fallback
#
#for _canonical, _info in _DEGREE_TAXONOMY.items():
#    _DEGREE_CATEGORY[_canonical] = _info["category"]
#    for _alias in _info["aliases"]:
#        _DEGREE_MATCHER.add_keyword(_alias, _canonical)
#        _ALL_ALIASES.append((_alias, _canonical))
#
#_FUZZY_DEGREE_THRESHOLD = 85
#
#
#def match_degree(line: str) -> tuple[str, str, int, int] | tuple[None, None, None, None]:
#    """Returns (canonical_degree_name, category, match_start, match_end) for
#    the first degree found in a line, or all-None. The span lets us find
#    specialization text right after the degree token, and mask the degree
#    token itself out before institution search (so e.g. 'School' inside
#    'High School' doesn't get mistaken for the institution's own name)."""
#    hits = _DEGREE_MATCHER.extract_keywords(line, span_info=True)
#    if hits:
#        canonical, start, end = hits[0]
#        return canonical, _DEGREE_CATEGORY[canonical], start, end
#
#    # Fuzzy fallback — only worth trying on short tokens (a handful of words),
#    # since fuzzy-matching a whole sentence against "B.Tech" is meaningless.
#    # No reliable span here, so field extraction/masking is skipped for fuzzy matches.
#    for token in re.findall(r"[A-Za-z][A-Za-z.\-]{1,12}", line):
#        if len(token) < 2:
#            continue
#        best = process.extractOne(
#            token.lower(), [a for a, _ in _ALL_ALIASES], scorer=fuzz.ratio
#        )
#        if best and best[1] >= _FUZZY_DEGREE_THRESHOLD:
#            matched_alias = best[0]
#            canonical = next(c for a, c in _ALL_ALIASES if a == matched_alias)
#            return canonical, _DEGREE_CATEGORY[canonical], None, None
#    return None, None, None, None
#
#
#_FIELD_PAREN_RE = re.compile(r'^\(([^)]+)\)')
#_FIELD_COLON_RE = re.compile(r'^:\s*(.+)')
#_FIELD_CONNECTOR_RE = re.compile(r'^(?:in|of)\s+(.+)', re.IGNORECASE)
#_FIELD_DASH_RE = re.compile(r'^[-\u2013]\s*(.+)')
#
#
#def _extract_degree_field(trigger_line: str, match_end: int | None) -> str | None:
#    """Pulls the specialization/field out of the same line as the degree
#    token, e.g. 'B.E.: I.T' -> 'I.T', 'M.E. (Mechanical Engg.)' -> 'Mechanical
#    Engg.', 'B.Tech in Computer Science' -> 'Computer Science'. Only looks at
#    text immediately after the degree token — if nothing recognizable follows
#    (no colon/parens/in/of/dash), we don't guess. A parenthetical that's just
#    a bare year ('(2019)') or a restatement of the degree abbreviation itself
#    ('Bachelor of Technology (B.tech)') isn't a field — skip it and look past it."""
#    if match_end is None:
#        return None
#    remainder = trigger_line[match_end:].strip()
#    if not remainder:
#        return None
#
#    m = _FIELD_PAREN_RE.match(remainder)
#    if m:
#        content = m.group(1).strip()
#        is_bare_year = re.fullmatch(r'(19|20)\d{2}', content)
#        is_degree_restatement = bool(_DEGREE_MATCHER.extract_keywords(content))
#        if not is_bare_year and not is_degree_restatement:
#            return content or None
#        # Skip past this parenthetical and look for a field marker after it
#        remainder = remainder[m.end():].strip()
#        if not remainder:
#            return None
#
#    field = None
#    for pattern in (_FIELD_COLON_RE, _FIELD_CONNECTOR_RE, _FIELD_DASH_RE):
#        m = pattern.match(remainder)
#        if m:
#            field = m.group(1)
#            break
#    if field is None:
#        return None
#
#    # Stop at the first comma/bullet/pipe — usually separates field from
#    # institution/city or from ranking blurbs ("Finance • Top 10% of 572")
#    field = re.split(r'[,•|]', field)[0].strip(" .")
#    return field or None
#
## Year — plain year, or a fuller date like "23 Sep 2011" / "Jan 2018"
#YEAR_RE = re.compile(
#    r'\b(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+'
#    r'(19[6-9]\d|20[0-3]\d)\b|\b(19[6-9]\d|20[0-3]\d)\b',
#    re.IGNORECASE
#)
#
## GPA — keyword-led first ("CGPA: 8.16 /10", "Percentage:- 89%")
#GPA_KEYWORD_RE = re.compile(
#    r'(?:gpa|cgpa|sgpa|percentage|grade|score|marks)[:\s\-]+([0-9.]+(?:\s*/\s*[0-9.]+)?%?)',
#    re.IGNORECASE
#)
## Reversed order — number BEFORE the keyword ("7.8 CGPA", "7.8 SGPA")
#GPA_KEYWORD_REVERSED_RE = re.compile(
#    r'\b([0-9.]+)\s*(?:gpa|cgpa|sgpa)\b', re.IGNORECASE
#)
## Fallback: bare "8.16 /10" style CGPA with no keyword in front
#GPA_BARE_SCALE_RE = re.compile(r'\b(\d(?:\.\d{1,2})?)\s*/\s*(10|4)\b')
## Fallback: bare percentage with no keyword, e.g. "62.46%" — but NOT a class-rank
## phrase like "Top 10% of 572" or "Top 2% in Finance", which isn't a GPA at all.
#GPA_BARE_PERCENT_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\s*%')
#_RANKING_PHRASE_RE = re.compile(r'\btop\b', re.IGNORECASE)
#
## Institution keyword fallback now uses _INSTITUTION_KEYWORDS_RE defined
## alongside _match_keyword_fallback() below (anchor + expand, not one greedy regex).
#
#
#def _load_institutions_db() -> dict:
#    """Loads data/institutions_db.json, built once by build_institutions_db.py
#    from the AISHE colleges dataset + Hipolabs India universities list."""
#    db_path = Path("data/institutions_db.json")
#    if db_path.exists():
#        with open(db_path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {"full_names": [], "acronym_index": {}}
#
#
#_INSTITUTIONS_DB = _load_institutions_db()
#
## flashtext processor for whole-phrase, case-insensitive full-name matching.
## Full names rarely collide across institutions, so this tier is high-confidence.
#_FULL_NAME_MATCHER = KeywordProcessor(case_sensitive=False)
#for _name in _INSTITUTIONS_DB.get("full_names", []):
#    _FULL_NAME_MATCHER.add_keyword(_name)
#
#
#_LEADING_WORD_RE = re.compile(r"^[A-Z][a-z.&'\-]*$|^[A-Z]{2,20}$")  # Title-Case OR a plausible all-caps word
#_CONNECTOR_WORDS = {"of", "the", "and", "for"}
#
## All-caps tokens that are degree/GPA jargon, not institution-name acronyms —
## these should never be swept into an institution name during expansion.
#_NON_INSTITUTION_ACRONYMS = {"CGPA", "SGPA", "GPA", "SSC", "HSC", "PGDM", "PGDCA", "PGDBA"} | {
#    a.upper().replace(".", "") for a, _ in _ALL_ALIASES if a.isupper() or "." in a
#}
#
#
#def _is_expandable_word(tok: str) -> bool:
#    if not _LEADING_WORD_RE.match(tok):
#        return False
#    if tok.upper().replace(".", "") in _NON_INSTITUTION_ACRONYMS:
#        return False
#    return True
#
#
#def _expand_left(entry_text: str, start: int) -> list:
#    """Expands leftward from `start` over Title-Case words/connectors, but
#    guards against sweeping in a specialization word — e.g. in 'PGDM in
#    Finance IIM Ahmedabad', 'Finance' would otherwise get pulled into the
#    institution name just because it's Title-Case. If exactly one word
#    would be pulled in and it's directly preceded by 'in', that's a
#    specialization clause ('degree in X'), not part of the institution."""
#    before = entry_text[:start].rstrip()
#    tokens_before = before.split(" ") if before else []
#    extended = []
#    stop_token = None
#    for tok in reversed(tokens_before):
#        if _is_expandable_word(tok) or tok.lower() in _CONNECTOR_WORDS:
#            extended.insert(0, tok)
#        else:
#            stop_token = tok
#            break
#    if len(extended) == 1 and stop_token and stop_token.lower() == "in":
#        return []
#    return extended
#
#
#def _expand_right(entry_text: str, end: int) -> list:
#    """Expands rightward from `end` over Title-Case words/connectors, stopping
#    at a comma/paren/digit. Fixes DB hits like 'Indian Institute of
#    Management' truncating the real name 'Indian Institute of Management
#    Calcutta' (connector 'of' must be allowed through, not just Title-Case
#    words, or the expansion stops one word too early)."""
#    after = entry_text[end:].lstrip()
#    tokens_after = after.split(" ") if after else []
#    extended = []
#    for tok in tokens_after:
#        clean = tok.rstrip(",")
#        if _is_expandable_word(clean) or clean.lower() in _CONNECTOR_WORDS:
#            extended.append(clean)
#            if clean != tok:  # trailing comma — stop after including this word
#                break
#        else:
#            break
#    return extended
#
#
#def _match_full_name(entry_text: str) -> str | None:
#    hits = _FULL_NAME_MATCHER.extract_keywords(entry_text, span_info=True)
#    if not hits:
#        return None
#    # Longest span wins among direct hits (more specific match).
#    keyword, start, end = max(hits, key=lambda h: h[2] - h[1])
#    left = _expand_left(entry_text, start)
#    right = _expand_right(entry_text, end)
#    prefix = " ".join(left)
#    suffix = " ".join(right)
#    parts = [p for p in (prefix, keyword, suffix) if p]
#    return " ".join(parts).strip()
#
#
#_INSTITUTION_KEYWORDS_SAFE_RE = re.compile(
#    r'\b(university|institute|institution|college|school|board|iit|nit|bits|iim|ignou)\b',
#    re.IGNORECASE
#)
#_INSTITUTION_KEYWORDS_RISKY_RE = re.compile(r'\b(academy)\b', re.IGNORECASE)
## Kept for _extract_institution's final-string matching, where both matter equally
#_INSTITUTION_KEYWORDS_RE = re.compile(
#    r'\b(university|institute|institution|academy|college|school|board|iit|nit|bits|iim|ignou)\b',
#    re.IGNORECASE
#)
#_BARE_GENERIC_INSTITUTION = {"school", "college", "university", "institute", "board"}
#
#
#def _match_keyword_fallback(entry_text: str) -> str | None:
#    m = _INSTITUTION_KEYWORDS_RE.search(entry_text)
#    if not m:
#        return None
#    start, end = m.start(), m.end()
#    left = _expand_left(entry_text, start)
#    right = _expand_right(entry_text, end)
#
#    parts = left + [m.group(0)] + right
#    candidate = " ".join(parts).strip()
#    return candidate or None
#
#
#def _extract_institution(entry_text: str) -> tuple[str | None, bool]:
#    """Returns (institution, needs_review). Priority: full name in DB (safest)
#    -> generic keyword regex (not DB-verified, but usually accurate when a
#    resume spells the name out in full).
#
#    There used to be a third tier here: matching bare acronyms (SVIT, EIILM,
#    IIMT...) against a DB index built from institutions' generated initials.
#    Removed after testing showed every real acronym-tier match came back
#    flagged needs_review anyway (unique-in-DB never proved correct-in-reality
#    — see e.g. "SVIT" resolving to a real but wrong college in Telangana).
#    Since a flagged guess can still be misused by anything downstream that
#    doesn't check the flag, and None cannot be, this trades a small amount of
#    partial coverage (pure-acronym mentions with no institution keyword
#    nearby) for eliminating an entire recurring class of false-confidence
#    bugs (2-letter state-code collisions, cross-state acronym collisions)."""
#    full = _match_full_name(entry_text)
#    if full:
#        return full, False
#
#    fallback = _match_keyword_fallback(entry_text)
#    if fallback:
#        # A bare generic word with nothing else ("School" alone, no name
#        # attached) usually means the real name got masked out along with
#        # the degree phrase on the same line (e.g. "Senior Secondary
#        # School" where the school's own name IS "Senior Secondary
#        # School") — not enough to trust confidently.
#        needs_review = fallback.strip().lower() in _BARE_GENERIC_INSTITUTION
#        return fallback, needs_review
#
#    return None, False
#
#
#def _extract_year(entry_text: str) -> str | None:
#    # Prefer the LAST year found — resumes write "2010 – 2014", and the end
#    # of the range is the graduation/completion year, not the start year.
#    matches = list(YEAR_RE.finditer(entry_text))
#    if not matches:
#        return None
#    m = matches[-1]
#    return m.group(1) or m.group(2)
#
#
#def _extract_gpa(line_text: str) -> str | None:
#    m = GPA_KEYWORD_RE.search(line_text)
#    if m:
#        return m.group(1).strip()
#    m = GPA_KEYWORD_REVERSED_RE.search(line_text)
#    if m:
#        return m.group(1).strip()
#    m = GPA_BARE_SCALE_RE.search(line_text)
#    if m:
#        return f"{m.group(1)}/{m.group(2)}"
#    for m in GPA_BARE_PERCENT_RE.finditer(line_text):
#        window_start = max(0, m.start() - 10)
#        if _RANKING_PHRASE_RE.search(line_text[window_start:m.start()]):
#            continue  # "Top 10% of 572" — a class rank, not a GPA
#        return f"{m.group(1)}%"
#    return None
#
#
#def _extract_degree(trigger_line: str, degree_type: str) -> str:
#    """Best-effort clean degree label from the line that triggered this entry.
#    Cuts off at the first comma so trailing institution text doesn't leak in
#    (e.g. 'B.E. (Mechanical Engg.) SVIT, Pune' -> 'B.E. (Mechanical Engg.) SVIT'
#    — imperfect but keeps the label short; institution field carries the real name)."""
#    return trigger_line.strip()
#
#
#def _line_has_institution_signal(line: str) -> bool:
#    """Lightweight per-line check used only to decide entry boundaries — does
#    this single line look like it's introducing an institution mention?
#    Doesn't need to reconstruct the full name; that happens later via
#    _extract_institution() on the complete joined block.
#
#    Most institution keywords (university, institute, college, school...)
#    essentially never show up in ordinary prose within an education section,
#    so a simple presence check is fine. "academy" is the exception — it can
#    appear incidentally in a sentence ("...Student association academy") —
#    so it additionally requires a capitalized (proper-noun-shaped) neighbor
#    before counting as a real signal."""
#    if _INSTITUTION_KEYWORDS_SAFE_RE.search(line):
#        return True
#
#    m = _INSTITUTION_KEYWORDS_RISKY_RE.search(line)
#    if m:
#        before_tokens = line[:m.start()].rstrip().split(" ") if m.start() > 0 else []
#        after_tokens = line[m.end():].lstrip().split(" ") if m.end() < len(line) else []
#        before = before_tokens[-1] if before_tokens else ""
#        after = after_tokens[0] if after_tokens else ""
#        if after.lower().rstrip(",") in _CONNECTOR_WORDS and len(after_tokens) > 1:
#            after = after_tokens[1]
#        before_clean = before.strip(" ,.-")
#        after_clean = after.strip(" ,.-")
#        neighbor_capitalized = bool(
#            (before_clean and before_clean[0].isupper()) or
#            (after_clean and after_clean[0].isupper())
#        )
#        if neighbor_capitalized:
#            return True
#
#    if _FULL_NAME_MATCHER.extract_keywords(line):
#        return True
#    return False
#
#
#_EXPLICIT_LEVEL_RE = re.compile(r'\((10|12)\)|\b(10|12)th\b', re.IGNORECASE)
#
#
#def _explicit_school_level(text: str) -> str | None:
#    """Some resumes write contradictory things like 'Higher Secondary School
#    (10)' — the English name says 12th, but the explicit (10) means the
#    person means 10th. When an unambiguous (10)/(12)/'10th'/'12th' marker is
#    present, it should win over the fuzzy English-name-based classification.
#    Deliberately narrow (paren-wrapped or 'th'-suffixed only) so it doesn't
#    false-trigger on '8.16 /10' style GPA-out-of-10 scales."""
#    m = _EXPLICIT_LEVEL_RE.search(text)
#    if not m:
#        return None
#    digits = m.group(1) or m.group(2)
#    return f"{digits}th"
#
#
#def _flush_entry(current: dict) -> dict | None:
#    """Finalizes one accumulated entry into an output record, or returns None
#    if it has nothing usable (pure noise lines with no degree or institution
#    signal at all)."""
#    if not current["lines"]:
#        return None
#
#    # Mask out the exact degree-token substring (not the whole line!) before
#    # institution search — otherwise a word like "School" inside the degree
#    # phrase "High School" gets mistaken for the institution's own name,
#    # while real institution text on the same line ("TDS School, Aligarh")
#    # would still need to survive the mask.
#    entry_text = " ".join(current["lines"])
#    if current["trigger_line_index"] is not None and current["degree_start"] is not None:
#        offset = len(" ".join(current["lines"][:current["trigger_line_index"]]))
#        if offset:
#            offset += 1  # the joining space
#        abs_start = offset + current["degree_start"]
#        abs_end = offset + current["degree_end"]
#        entry_text = entry_text[:abs_start] + (" " * (abs_end - abs_start)) + entry_text[abs_end:]
#
#    institution, needs_review = _extract_institution(entry_text)
#    year = _extract_year(entry_text)
#    gpa = _extract_gpa(entry_text)
#
#    degree_name = current["degree_name"]
#    degree_type = current["degree_category"]
#    degree = current["trigger_line"].strip() if current["trigger_line"] else None
#    degree_field = _extract_degree_field(current["trigger_line"], current["degree_end"]) \
#        if current["trigger_line"] else None
#
#    # Some resumes write contradictory things like "Higher Secondary School (10)"
#    # — an explicit (10)/(12)/"10th"/"12th" marker overrides the fuzzy
#    # English-name-based classification, and the contradiction itself is a
#    # signal this entry deserves a second look.
#    explicit_level = _explicit_school_level(entry_text)
#    if explicit_level and degree_type in ("10th", "12th") and explicit_level != degree_type:
#        degree_type = explicit_level
#        degree_name = explicit_level
#        needs_review = True
#
#    if not degree_name and not institution:
#        return None  # nothing usable at all — pure noise lines
#
#    if not degree_name and institution and institution.strip().lower() in _BARE_GENERIC_INSTITUTION:
#        return None  # e.g. a table header row ("Degree/Exam Board/Institute Year") — not a real entry
#
#    if not degree_name:
#        # Institution-only entry (e.g. an executive program with no
#        # recognized degree word) — still worth keeping, just flagged.
#        needs_review = True
#
#    fields_found = sum(1 for v in (degree_type, institution, year) if v)
#    base_confidence = min(0.5 + 0.15 * fields_found, 0.95)
#    if needs_review:
#        base_confidence = min(base_confidence, 0.55)
#
#    return {
#        "degree": degree,
#        "degree_name": degree_name,
#        "degree_type": degree_type,
#        "degree_field": degree_field,
#        "institution": institution,
#        "year": year,
#        "gpa": gpa,
#        "_needs_review": needs_review,
#        "_confidence": round(base_confidence, 2),
#    }
#
#
#def _new_current() -> dict:
#    return {
#        "lines": [],
#        "degree_name": None,
#        "degree_category": None,
#        "trigger_line": None,
#        "trigger_line_index": None,
#        "degree_start": None,
#        "degree_end": None,
#        "has_institution_signal": False,
#        "last_institution_line_idx": None,
#    }
#
#
#def extract_education(education_section_text: str) -> dict:
#    """
#    Parses the education section into a list of entries.
#
#    Rather than writing a separate rule for every ordering resumes use
#    (degree-then-college, college-then-degree, same-line, across-lines,
#    with/without location...), each line is classified for two signals —
#    "does this line introduce a degree?" and "does this line introduce an
#    institution?" — and entries are flushed whenever a line would introduce
#    a SECOND degree or a SECOND institution into the entry currently being
#    built. This one mechanism covers every ordering: it doesn't matter
#    whether the degree or the institution comes first, only that a repeat
#    of either one means we've moved on to the next entry. Lines with neither
#    signal (bullet points, job-description-style prose under an education
#    entry) are just carried along as context for GPA/year extraction.
#    """
#    if not education_section_text.strip():
#        return {"education": [], "_confidence": 0.0}
#
#    lines = [l.strip() for l in education_section_text.strip().split("\n") if l.strip()]
#
#    education_entries = []
#    current = _new_current()
#
#    for line_idx, line in enumerate(lines):
#        canonical, category, degree_start, degree_end = match_degree(line)
#        has_inst_signal = _line_has_institution_signal(line)
#
#        institution_conflict = False
#        if has_inst_signal and current["has_institution_signal"]:
#            last_idx = current["last_institution_line_idx"]
#            adjacent = last_idx is not None and (line_idx - last_idx) == 1
#            # Adjacent lines both carrying institution signal are usually one
#            # institution name wrapped across a line break (e.g. "SVIT, Pune"
#            # then "University ,23 Sep" — together "SVIT, Pune University"),
#            # UNLESS this line also introduces its own degree word — that's
#            # the tell that it's actually a new entry starting, not a
#            # continuation (e.g. "...Indian Institute" then "MS in Cyber Law
#            # ... National Law Institute" are two different entries, even
#            # though both lines happen to carry institution signal back to back).
#            if adjacent and not canonical:
#                institution_conflict = False
#            else:
#                institution_conflict = True
#
#        conflict = (canonical and current["degree_name"]) or institution_conflict
#        if conflict:
#            finalized = _flush_entry(current)
#            if finalized:
#                education_entries.append(finalized)
#            current = _new_current()
#
#        current["lines"].append(line)
#        if canonical and not current["degree_name"]:
#            current["degree_name"] = canonical
#            current["degree_category"] = category
#            current["trigger_line"] = line
#            current["trigger_line_index"] = len(current["lines"]) - 1
#            current["degree_start"] = degree_start
#            current["degree_end"] = degree_end
#        if has_inst_signal:
#            current["has_institution_signal"] = True
#            current["last_institution_line_idx"] = line_idx
#
#    finalized = _flush_entry(current)
#    if finalized:
#        education_entries.append(finalized)
#
#    overall_confidence = round(
#        sum(e["_confidence"] for e in education_entries) / len(education_entries), 2
#    ) if education_entries else 0.0
#
#    return {
#        "education": education_entries,
#        "_confidence": overall_confidence,
#    }
#

















#
##worked just changing minimal improvement in o/p
#"""
#Education Extractor — Layer 2
#Extracts: degree name, degree type, institution, year of completion, GPA/percentage.
#
#Design notes (why this differs from a naive line-by-line parser):
#- Real resumes wrap education entries across lines inconsistently. Sometimes the
#  institution appears BEFORE the degree line (e.g. "VESIT, Mumbai, 2008" then
#  "B.E.: I.T"). Sometimes two entries sit back-to-back with NO blank line between
#  them (e.g. an M.E. entry immediately followed by a B.E. entry).
#- Because of that, entries are NOT split on blank lines. They're split on
#  "trigger lines" — lines that match a degree pattern. Lines before the first
#  trigger attach to entry 1; lines after a trigger attach to that entry until the
#  next trigger appears. This handles both orderings above correctly.
#- Each entry's lines are joined into a single string before running the
#  institution/year/GPA regexes, so a name wrapped across two lines
#  ("... Mumbai" / "University, Mumbai Jan 2018") is still matched as one phrase.
#"""
#
#import json
#import re
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import fuzz, process
#
## ─────────────────────────────────────────────────────────────
## DEGREE PATTERNS (checked in priority order — first match wins)
## ─────────────────────────────────────────────────────────────
#
#def _load_degree_taxonomy() -> dict:
#    """Loads data/degree_taxonomy.json, built once by build_degree_taxonomy.py.
#    Unlike institutions, degree names are a closed, well-known vocabulary —
#    same curated-taxonomy-plus-fuzzy-match pattern your skills extractor uses,
#    rather than a handful of broad regex buckets."""
#    path = Path("data/degree_taxonomy.json")
#    if path.exists():
#        with open(path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {}
#
#
#_DEGREE_TAXONOMY = _load_degree_taxonomy()
#
## flashtext: alias (lowercase) -> canonical degree name, e.g. "b.tech" -> "B.Tech"
#_DEGREE_MATCHER = KeywordProcessor(case_sensitive=False)
#_DEGREE_CATEGORY = {}          # canonical name -> category (bachelors/masters/...)
#_ALL_ALIASES = []              # flat list, used for the fuzzy fallback
#
#for _canonical, _info in _DEGREE_TAXONOMY.items():
#    _DEGREE_CATEGORY[_canonical] = _info["category"]
#    for _alias in _info["aliases"]:
#        _DEGREE_MATCHER.add_keyword(_alias, _canonical)
#        _ALL_ALIASES.append((_alias, _canonical))
#
#_FUZZY_DEGREE_THRESHOLD = 85
#
#
#def match_degree(line: str) -> tuple[str, str, int, int] | tuple[None, None, None, None]:
#    """Returns (canonical_degree_name, category, match_start, match_end) for
#    the first degree found in a line, or all-None. The span lets us find
#    specialization text right after the degree token, and mask the degree
#    token itself out before institution search (so e.g. 'School' inside
#    'High School' doesn't get mistaken for the institution's own name)."""
#    hits = _DEGREE_MATCHER.extract_keywords(line, span_info=True)
#    if hits:
#        canonical, start, end = hits[0]
#        return canonical, _DEGREE_CATEGORY[canonical], start, end
#
#    # Fuzzy fallback — only worth trying on short tokens (a handful of words),
#    # since fuzzy-matching a whole sentence against "B.Tech" is meaningless.
#    # No reliable span here, so field extraction/masking is skipped for fuzzy matches.
#    for token in re.findall(r"[A-Za-z][A-Za-z.\-]{1,12}", line):
#        if len(token) < 2:
#            continue
#        best = process.extractOne(
#            token.lower(), [a for a, _ in _ALL_ALIASES], scorer=fuzz.ratio
#        )
#        if best and best[1] >= _FUZZY_DEGREE_THRESHOLD:
#            matched_alias = best[0]
#            canonical = next(c for a, c in _ALL_ALIASES if a == matched_alias)
#            return canonical, _DEGREE_CATEGORY[canonical], None, None
#    return None, None, None, None
#
#
#_FIELD_PAREN_RE = re.compile(r'^\(([^)]+)\)')
#_FIELD_COLON_RE = re.compile(r'^:\s*(.+)')
#_FIELD_CONNECTOR_RE = re.compile(r'^(?:in|of)\s+(.+)', re.IGNORECASE)
#_FIELD_DASH_RE = re.compile(r'^[-\u2013]\s*(.+)')
#
#
#def _extract_degree_field(trigger_line: str, match_end: int | None) -> str | None:
#    """Pulls the specialization/field out of the same line as the degree
#    token, e.g. 'B.E.: I.T' -> 'I.T', 'M.E. (Mechanical Engg.)' -> 'Mechanical
#    Engg.', 'B.Tech in Computer Science' -> 'Computer Science'. Only looks at
#    text immediately after the degree token — if nothing recognizable follows
#    (no colon/parens/in/of/dash), we don't guess. A parenthetical that's just
#    a bare year ('(2019)') or a restatement of the degree abbreviation itself
#    ('Bachelor of Technology (B.tech)') isn't a field — skip it and look past it."""
#    if match_end is None:
#        return None
#    remainder = trigger_line[match_end:].strip()
#    if not remainder:
#        return None
#
#    m = _FIELD_PAREN_RE.match(remainder)
#    if m:
#        content = m.group(1).strip()
#        is_bare_year = re.fullmatch(r'(19|20)\d{2}', content)
#        is_degree_restatement = bool(_DEGREE_MATCHER.extract_keywords(content))
#        if not is_bare_year and not is_degree_restatement:
#            return content or None
#        # Skip past this parenthetical and look for a field marker after it
#        remainder = remainder[m.end():].strip()
#        if not remainder:
#            return None
#
#    field = None
#    for pattern in (_FIELD_COLON_RE, _FIELD_CONNECTOR_RE, _FIELD_DASH_RE):
#        m = pattern.match(remainder)
#        if m:
#            field = m.group(1)
#            break
#    if field is None:
#        return None
#
#    # Stop at the first comma/bullet/pipe — usually separates field from
#    # institution/city or from ranking blurbs ("Finance • Top 10% of 572")
#    field = re.split(r'[,•|]', field)[0].strip(" .")
#    return field or None
#
## Year — plain year, or a fuller date like "23 Sep 2011" / "Jan 2018"
#YEAR_RE = re.compile(
#    r'\b(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+'
#    r'(19[6-9]\d|20[0-3]\d)\b|\b(19[6-9]\d|20[0-3]\d)\b',
#    re.IGNORECASE
#)
#
## GPA — keyword-led first ("CGPA: 8.16 /10", "Percentage:- 89%")
#GPA_KEYWORD_RE = re.compile(
#    r'(?:gpa|cgpa|sgpa|percentage|grade|score|marks)[:\s\-]+([0-9.]+(?:\s*/\s*[0-9.]+)?%?)',
#    re.IGNORECASE
#)
## Fallback: bare "8.16 /10" style CGPA with no keyword in front
#GPA_BARE_SCALE_RE = re.compile(r'\b(\d(?:\.\d{1,2})?)\s*/\s*(10|4)\b')
## Fallback: bare percentage with no keyword, e.g. "62.46%" — but NOT a class-rank
## phrase like "Top 10% of 572" or "Top 2% in Finance", which isn't a GPA at all.
#GPA_BARE_PERCENT_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\s*%')
#_RANKING_PHRASE_RE = re.compile(r'\btop\b', re.IGNORECASE)
#
## Institution keyword fallback now uses _INSTITUTION_KEYWORDS_RE defined
## alongside _match_keyword_fallback() below (anchor + expand, not one greedy regex).
#
#
#def _load_institutions_db() -> dict:
#    """Loads data/institutions_db.json, built once by build_institutions_db.py
#    from the AISHE colleges dataset + Hipolabs India universities list."""
#    db_path = Path("data/institutions_db.json")
#    if db_path.exists():
#        with open(db_path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {"full_names": [], "acronym_index": {}}
#
#
#_INSTITUTIONS_DB = _load_institutions_db()
#
## flashtext processor for whole-phrase, case-insensitive full-name matching.
## Full names rarely collide across institutions, so this tier is high-confidence.
#_FULL_NAME_MATCHER = KeywordProcessor(case_sensitive=False)
#for _name in _INSTITUTIONS_DB.get("full_names", []):
#    _FULL_NAME_MATCHER.add_keyword(_name)
#
#
#_LEADING_WORD_RE = re.compile(r"^[A-Z][a-z.&'\-]*$|^[A-Z]{2,8}$")  # Title-Case OR a plausible all-caps acronym prefix
#_CONNECTOR_WORDS = {"of", "the", "and", "for"}
#
## All-caps tokens that are degree/GPA jargon, not institution-name acronyms —
## these should never be swept into an institution name during expansion.
#_NON_INSTITUTION_ACRONYMS = {"CGPA", "SGPA", "GPA", "SSC", "HSC", "PGDM", "PGDCA", "PGDBA"} | {
#    a.upper().replace(".", "") for a, _ in _ALL_ALIASES if a.isupper() or "." in a
#}
#
#
#def _is_expandable_word(tok: str) -> bool:
#    if not _LEADING_WORD_RE.match(tok):
#        return False
#    if tok.upper().replace(".", "") in _NON_INSTITUTION_ACRONYMS:
#        return False
#    return True
#
#
#def _expand_left(entry_text: str, start: int) -> list:
#    """Expands leftward from `start` over Title-Case words/connectors, but
#    guards against sweeping in a specialization word — e.g. in 'PGDM in
#    Finance IIM Ahmedabad', 'Finance' would otherwise get pulled into the
#    institution name just because it's Title-Case. If exactly one word
#    would be pulled in and it's directly preceded by 'in', that's a
#    specialization clause ('degree in X'), not part of the institution."""
#    before = entry_text[:start].rstrip()
#    tokens_before = before.split(" ") if before else []
#    extended = []
#    stop_token = None
#    for tok in reversed(tokens_before):
#        if _is_expandable_word(tok) or tok.lower() in _CONNECTOR_WORDS:
#            extended.insert(0, tok)
#        else:
#            stop_token = tok
#            break
#    if len(extended) == 1 and stop_token and stop_token.lower() == "in":
#        return []
#    return extended
#
#
#def _expand_right(entry_text: str, end: int) -> list:
#    """Expands rightward from `end` over Title-Case words/connectors, stopping
#    at a comma/paren/digit. Fixes DB hits like 'Indian Institute of
#    Management' truncating the real name 'Indian Institute of Management
#    Calcutta' (connector 'of' must be allowed through, not just Title-Case
#    words, or the expansion stops one word too early)."""
#    after = entry_text[end:].lstrip()
#    tokens_after = after.split(" ") if after else []
#    extended = []
#    for tok in tokens_after:
#        clean = tok.rstrip(",")
#        if _is_expandable_word(clean) or clean.lower() in _CONNECTOR_WORDS:
#            extended.append(clean)
#            if clean != tok:  # trailing comma — stop after including this word
#                break
#        else:
#            break
#    return extended
#
#
#def _match_full_name(entry_text: str) -> str | None:
#    hits = _FULL_NAME_MATCHER.extract_keywords(entry_text, span_info=True)
#    if not hits:
#        return None
#    # Longest span wins among direct hits (more specific match).
#    keyword, start, end = max(hits, key=lambda h: h[2] - h[1])
#    left = _expand_left(entry_text, start)
#    right = _expand_right(entry_text, end)
#    prefix = " ".join(left)
#    suffix = " ".join(right)
#    parts = [p for p in (prefix, keyword, suffix) if p]
#    return " ".join(parts).strip()
#
#
#_INSTITUTION_KEYWORDS_RE = re.compile(
#    r'\b(university|institute|college|school|board|iit|nit|bits|iim|ignou)\b', re.IGNORECASE
#)
#_BARE_GENERIC_INSTITUTION = {"school", "college", "university", "institute", "board"}
#
#
#def _match_keyword_fallback(entry_text: str) -> str | None:
#    m = _INSTITUTION_KEYWORDS_RE.search(entry_text)
#    if not m:
#        return None
#    start, end = m.start(), m.end()
#    left = _expand_left(entry_text, start)
#    right = _expand_right(entry_text, end)
#
#    parts = left + [m.group(0)] + right
#    candidate = " ".join(parts).strip()
#    return candidate or None
#
#
#def _extract_institution(entry_text: str) -> tuple[str | None, bool]:
#    """Returns (institution, needs_review). Priority: full name in DB (safest)
#    -> generic keyword regex (not DB-verified, but usually accurate when a
#    resume spells the name out in full).
#
#    There used to be a third tier here: matching bare acronyms (SVIT, EIILM,
#    IIMT...) against a DB index built from institutions' generated initials.
#    Removed after testing showed every real acronym-tier match came back
#    flagged needs_review anyway (unique-in-DB never proved correct-in-reality
#    — see e.g. "SVIT" resolving to a real but wrong college in Telangana).
#    Since a flagged guess can still be misused by anything downstream that
#    doesn't check the flag, and None cannot be, this trades a small amount of
#    partial coverage (pure-acronym mentions with no institution keyword
#    nearby) for eliminating an entire recurring class of false-confidence
#    bugs (2-letter state-code collisions, cross-state acronym collisions)."""
#    full = _match_full_name(entry_text)
#    if full:
#        return full, False
#
#    fallback = _match_keyword_fallback(entry_text)
#    if fallback:
#        # A bare generic word with nothing else ("School" alone, no name
#        # attached) usually means the real name got masked out along with
#        # the degree phrase on the same line (e.g. "Senior Secondary
#        # School" where the school's own name IS "Senior Secondary
#        # School") — not enough to trust confidently.
#        needs_review = fallback.strip().lower() in _BARE_GENERIC_INSTITUTION
#        return fallback, needs_review
#
#    return None, False
#
#
#def _extract_year(entry_text: str) -> str | None:
#    m = YEAR_RE.search(entry_text)
#    if not m:
#        return None
#    return m.group(1) or m.group(2)
#
#
#def _extract_gpa(line_text: str) -> str | None:
#    m = GPA_KEYWORD_RE.search(line_text)
#    if m:
#        return m.group(1).strip()
#    m = GPA_BARE_SCALE_RE.search(line_text)
#    if m:
#        return f"{m.group(1)}/{m.group(2)}"
#    for m in GPA_BARE_PERCENT_RE.finditer(line_text):
#        window_start = max(0, m.start() - 10)
#        if _RANKING_PHRASE_RE.search(line_text[window_start:m.start()]):
#            continue  # "Top 10% of 572" — a class rank, not a GPA
#        return f"{m.group(1)}%"
#    return None
#
#
#def _extract_degree(trigger_line: str, degree_type: str) -> str:
#    """Best-effort clean degree label from the line that triggered this entry.
#    Cuts off at the first comma so trailing institution text doesn't leak in
#    (e.g. 'B.E. (Mechanical Engg.) SVIT, Pune' -> 'B.E. (Mechanical Engg.) SVIT'
#    — imperfect but keeps the label short; institution field carries the real name)."""
#    return trigger_line.strip()
#
#
#def _line_has_institution_signal(line: str) -> bool:
#    """Lightweight per-line check used only to decide entry boundaries — does
#    this single line look like it's introducing an institution mention?
#    Doesn't need to reconstruct the full name; that happens later via
#    _extract_institution() on the complete joined block."""
#    if _INSTITUTION_KEYWORDS_RE.search(line):
#        return True
#    if _FULL_NAME_MATCHER.extract_keywords(line):
#        return True
#    return False
#
#
#_EXPLICIT_LEVEL_RE = re.compile(r'\((10|12)\)|\b(10|12)th\b', re.IGNORECASE)
#
#
#def _explicit_school_level(text: str) -> str | None:
#    """Some resumes write contradictory things like 'Higher Secondary School
#    (10)' — the English name says 12th, but the explicit (10) means the
#    person means 10th. When an unambiguous (10)/(12)/'10th'/'12th' marker is
#    present, it should win over the fuzzy English-name-based classification.
#    Deliberately narrow (paren-wrapped or 'th'-suffixed only) so it doesn't
#    false-trigger on '8.16 /10' style GPA-out-of-10 scales."""
#    m = _EXPLICIT_LEVEL_RE.search(text)
#    if not m:
#        return None
#    digits = m.group(1) or m.group(2)
#    return f"{digits}th"
#
#
#def _flush_entry(current: dict) -> dict | None:
#    """Finalizes one accumulated entry into an output record, or returns None
#    if it has nothing usable (pure noise lines with no degree or institution
#    signal at all)."""
#    if not current["lines"]:
#        return None
#
#    # Mask out the exact degree-token substring (not the whole line!) before
#    # institution search — otherwise a word like "School" inside the degree
#    # phrase "High School" gets mistaken for the institution's own name,
#    # while real institution text on the same line ("TDS School, Aligarh")
#    # would still need to survive the mask.
#    entry_text = " ".join(current["lines"])
#    if current["trigger_line_index"] is not None and current["degree_start"] is not None:
#        offset = len(" ".join(current["lines"][:current["trigger_line_index"]]))
#        if offset:
#            offset += 1  # the joining space
#        abs_start = offset + current["degree_start"]
#        abs_end = offset + current["degree_end"]
#        entry_text = entry_text[:abs_start] + (" " * (abs_end - abs_start)) + entry_text[abs_end:]
#
#    institution, needs_review = _extract_institution(entry_text)
#    year = _extract_year(entry_text)
#    gpa = _extract_gpa(entry_text)
#
#    degree_name = current["degree_name"]
#    degree_type = current["degree_category"]
#    degree = current["trigger_line"].strip() if current["trigger_line"] else None
#    degree_field = _extract_degree_field(current["trigger_line"], current["degree_end"]) \
#        if current["trigger_line"] else None
#
#    # Some resumes write contradictory things like "Higher Secondary School (10)"
#    # — an explicit (10)/(12)/"10th"/"12th" marker overrides the fuzzy
#    # English-name-based classification, and the contradiction itself is a
#    # signal this entry deserves a second look.
#    explicit_level = _explicit_school_level(entry_text)
#    if explicit_level and degree_type in ("10th", "12th") and explicit_level != degree_type:
#        degree_type = explicit_level
#        degree_name = explicit_level
#        needs_review = True
#
#    if not degree_name and not institution:
#        return None  # nothing usable at all — pure noise lines
#
#    if not degree_name:
#        # Institution-only entry (e.g. an executive program with no
#        # recognized degree word) — still worth keeping, just flagged.
#        needs_review = True
#
#    fields_found = sum(1 for v in (degree_type, institution, year) if v)
#    base_confidence = min(0.5 + 0.15 * fields_found, 0.95)
#    if needs_review:
#        base_confidence = min(base_confidence, 0.55)
#
#    return {
#        "degree": degree,
#        "degree_name": degree_name,
#        "degree_type": degree_type,
#        "degree_field": degree_field,
#        "institution": institution,
#        "year": year,
#        "gpa": gpa,
#        "_needs_review": needs_review,
#        "_confidence": round(base_confidence, 2),
#    }
#
#
#def _new_current() -> dict:
#    return {
#        "lines": [],
#        "degree_name": None,
#        "degree_category": None,
#        "trigger_line": None,
#        "trigger_line_index": None,
#        "degree_start": None,
#        "degree_end": None,
#        "has_institution_signal": False,
#        "last_institution_line_idx": None,
#    }
#
#
#def extract_education(education_section_text: str) -> dict:
#    """
#    Parses the education section into a list of entries.
#
#    Rather than writing a separate rule for every ordering resumes use
#    (degree-then-college, college-then-degree, same-line, across-lines,
#    with/without location...), each line is classified for two signals —
#    "does this line introduce a degree?" and "does this line introduce an
#    institution?" — and entries are flushed whenever a line would introduce
#    a SECOND degree or a SECOND institution into the entry currently being
#    built. This one mechanism covers every ordering: it doesn't matter
#    whether the degree or the institution comes first, only that a repeat
#    of either one means we've moved on to the next entry. Lines with neither
#    signal (bullet points, job-description-style prose under an education
#    entry) are just carried along as context for GPA/year extraction.
#    """
#    if not education_section_text.strip():
#        return {"education": [], "_confidence": 0.0}
#
#    lines = [l.strip() for l in education_section_text.strip().split("\n") if l.strip()]
#
#    education_entries = []
#    current = _new_current()
#
#    for line_idx, line in enumerate(lines):
#        canonical, category, degree_start, degree_end = match_degree(line)
#        has_inst_signal = _line_has_institution_signal(line)
#
#        institution_conflict = False
#        if has_inst_signal and current["has_institution_signal"]:
#            last_idx = current["last_institution_line_idx"]
#            adjacent = last_idx is not None and (line_idx - last_idx) == 1
#            # Adjacent lines both carrying institution signal are usually one
#            # institution name wrapped across a line break (e.g. "SVIT, Pune"
#            # then "University ,23 Sep" — together "SVIT, Pune University"),
#            # UNLESS this line also introduces its own degree word — that's
#            # the tell that it's actually a new entry starting, not a
#            # continuation (e.g. "...Indian Institute" then "MS in Cyber Law
#            # ... National Law Institute" are two different entries, even
#            # though both lines happen to carry institution signal back to back).
#            if adjacent and not canonical:
#                institution_conflict = False
#            else:
#                institution_conflict = True
#
#        conflict = (canonical and current["degree_name"]) or institution_conflict
#        if conflict:
#            finalized = _flush_entry(current)
#            if finalized:
#                education_entries.append(finalized)
#            current = _new_current()
#
#        current["lines"].append(line)
#        if canonical and not current["degree_name"]:
#            current["degree_name"] = canonical
#            current["degree_category"] = category
#            current["trigger_line"] = line
#            current["trigger_line_index"] = len(current["lines"]) - 1
#            current["degree_start"] = degree_start
#            current["degree_end"] = degree_end
#        if has_inst_signal:
#            current["has_institution_signal"] = True
#            current["last_institution_line_idx"] = line_idx
#
#    finalized = _flush_entry(current)
#    if finalized:
#        education_entries.append(finalized)
#
#    overall_confidence = round(
#        sum(e["_confidence"] for e in education_entries) / len(education_entries), 2
#    ) if education_entries else 0.0
#
#    return {
#        "education": education_entries,
#        "_confidence": overall_confidence,
#    }
#
#
















#"""
#Education Extractor — Layer 2
#Extracts: degree name, degree type, institution, year of completion, GPA/percentage.
#
#Design notes (why this differs from a naive line-by-line parser):
#- Real resumes wrap education entries across lines inconsistently. Sometimes the
#  institution appears BEFORE the degree line (e.g. "VESIT, Mumbai, 2008" then
#  "B.E.: I.T"). Sometimes two entries sit back-to-back with NO blank line between
#  them (e.g. an M.E. entry immediately followed by a B.E. entry).
#- Because of that, entries are NOT split on blank lines. They're split on
#  "trigger lines" — lines that match a degree pattern. Lines before the first
#  trigger attach to entry 1; lines after a trigger attach to that entry until the
#  next trigger appears. This handles both orderings above correctly.
#- Each entry's lines are joined into a single string before running the
#  institution/year/GPA regexes, so a name wrapped across two lines
#  ("... Mumbai" / "University, Mumbai Jan 2018") is still matched as one phrase.
#"""
#
#import json
#import re
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import fuzz, process
#
## ─────────────────────────────────────────────────────────────
## DEGREE PATTERNS (checked in priority order — first match wins)
## ─────────────────────────────────────────────────────────────
#
#def _load_degree_taxonomy() -> dict:
#    """Loads data/degree_taxonomy.json, built once by build_degree_taxonomy.py.
#    Unlike institutions, degree names are a closed, well-known vocabulary —
#    same curated-taxonomy-plus-fuzzy-match pattern your skills extractor uses,
#    rather than a handful of broad regex buckets."""
#    path = Path("data/degree_taxonomy.json")
#    if path.exists():
#        with open(path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {}
#
#
#_DEGREE_TAXONOMY = _load_degree_taxonomy()
#
## flashtext: alias (lowercase) -> canonical degree name, e.g. "b.tech" -> "B.Tech"
#_DEGREE_MATCHER = KeywordProcessor(case_sensitive=False)
#_DEGREE_CATEGORY = {}          # canonical name -> category (bachelors/masters/...)
#_ALL_ALIASES = []              # flat list, used for the fuzzy fallback
#
#for _canonical, _info in _DEGREE_TAXONOMY.items():
#    _DEGREE_CATEGORY[_canonical] = _info["category"]
#    for _alias in _info["aliases"]:
#        _DEGREE_MATCHER.add_keyword(_alias, _canonical)
#        _ALL_ALIASES.append((_alias, _canonical))
#
#_FUZZY_DEGREE_THRESHOLD = 85
#
#
#def match_degree(line: str) -> tuple[str, str] | tuple[None, None]:
#    """Returns (canonical_degree_name, category) for the first degree found
#    in a line, or (None, None). Tries an exact taxonomy/alias match first
#    (flashtext); falls back to fuzzy matching only on short, degree-shaped
#    tokens, so we don't accidentally fuzzy-match unrelated words in a long
#    institution name."""
#    hits = _DEGREE_MATCHER.extract_keywords(line)
#    if hits:
#        canonical = hits[0]
#        return canonical, _DEGREE_CATEGORY[canonical]
#
#    # Fuzzy fallback — only worth trying on short tokens (a handful of words),
#    # since fuzzy-matching a whole sentence against "B.Tech" is meaningless.
#    for token in re.findall(r"[A-Za-z][A-Za-z.\-]{1,12}", line):
#        if len(token) < 2:
#            continue
#        best = process.extractOne(
#            token.lower(), [a for a, _ in _ALL_ALIASES], scorer=fuzz.ratio
#        )
#        if best and best[1] >= _FUZZY_DEGREE_THRESHOLD:
#            matched_alias = best[0]
#            canonical = next(c for a, c in _ALL_ALIASES if a == matched_alias)
#            return canonical, _DEGREE_CATEGORY[canonical]
#    return None, None
#
## Year — plain year, or a fuller date like "23 Sep 2011" / "Jan 2018"
#YEAR_RE = re.compile(
#    r'\b(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+'
#    r'(19[6-9]\d|20[0-3]\d)\b|\b(19[6-9]\d|20[0-3]\d)\b',
#    re.IGNORECASE
#)
#
## GPA — keyword-led first ("CGPA: 8.16 /10", "Percentage: 89%")
#GPA_KEYWORD_RE = re.compile(
#    r'(?:gpa|cgpa|percentage|grade|score)[:\s]+([0-9.]+(?:\s*/\s*[0-9.]+)?%?)',
#    re.IGNORECASE
#)
## Fallback: bare "8.16 /10" style CGPA with no keyword in front
#GPA_BARE_SCALE_RE = re.compile(r'\b(\d(?:\.\d{1,2})?)\s*/\s*(10|4)\b')
## Fallback: bare percentage with no keyword, e.g. "62.46%"
#GPA_BARE_PERCENT_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\s*%')
#
## Institution keyword fallback now uses _INSTITUTION_KEYWORDS_RE defined
## alongside _match_keyword_fallback() below (anchor + expand, not one greedy regex).
#
#
#def _load_institutions_db() -> dict:
#    """Loads data/institutions_db.json, built once by build_institutions_db.py
#    from the AISHE colleges dataset + Hipolabs India universities list."""
#    db_path = Path("data/institutions_db.json")
#    if db_path.exists():
#        with open(db_path, "r", encoding="utf-8") as f:
#            return json.load(f)
#    return {"full_names": [], "acronym_index": {}}
#
#
#_INSTITUTIONS_DB = _load_institutions_db()
#
## flashtext processor for whole-phrase, case-insensitive full-name matching.
## Full names rarely collide across institutions, so this tier is high-confidence.
#_FULL_NAME_MATCHER = KeywordProcessor(case_sensitive=False)
#for _name in _INSTITUTIONS_DB.get("full_names", []):
#    _FULL_NAME_MATCHER.add_keyword(_name)
#
#_ACRONYM_INDEX = _INSTITUTIONS_DB.get("acronym_index", {})
#
## Short tokens that look like acronyms but aren't institution names —
## exclude these from acronym lookup so we don't try to resolve "CGPA" as a college.
#_ACRONYM_EXCLUDE = {
#    "GPA", "CGPA", "SSC", "HSC", "SGPA", "CBSE", "ICSE", "IT", "IB",
#    "AI", "ML", "OK", "II", "III",
#}
#
#_ACRONYM_TOKEN_RE = re.compile(r'\b[A-Z]{2,6}\b')
#
#
#_LEADING_WORD_RE = re.compile(r"^[A-Z][a-z.&'\-]*$")  # Title-Case only — excludes all-caps tokens like "MBBS"
#_CONNECTOR_WORDS = {"of", "the", "and", "for"}
#
#
#def _expand_left(entry_text: str, start: int) -> list:
#    """Expands leftward from `start` over Title-Case words/connectors, but
#    guards against sweeping in a specialization word — e.g. in 'PGDM in
#    Finance IIM Ahmedabad', 'Finance' would otherwise get pulled into the
#    institution name just because it's Title-Case. If exactly one word
#    would be pulled in and it's directly preceded by 'in', that's a
#    specialization clause ('degree in X'), not part of the institution."""
#    before = entry_text[:start].rstrip()
#    tokens_before = before.split(" ") if before else []
#    extended = []
#    stop_token = None
#    for tok in reversed(tokens_before):
#        if _LEADING_WORD_RE.match(tok) or tok.lower() in _CONNECTOR_WORDS:
#            extended.insert(0, tok)
#        else:
#            stop_token = tok
#            break
#    if len(extended) == 1 and stop_token and stop_token.lower() == "in":
#        return []
#    return extended
#
#
#def _match_full_name(entry_text: str) -> str | None:
#    hits = _FULL_NAME_MATCHER.extract_keywords(entry_text, span_info=True)
#    if not hits:
#        return None
#    # Longest span wins among direct hits (more specific match).
#    keyword, start, end = max(hits, key=lambda h: h[2] - h[1])
#    extended = _expand_left(entry_text, start)
#    prefix = " ".join(extended)
#    return f"{prefix} {keyword}".strip() if prefix else keyword
#
#
#def _match_acronym(entry_text: str) -> tuple[str | None, bool]:
#    """Returns (institution_name_or_raw_acronym, needs_review).
#    needs_review=True whenever the match came from the acronym tier at all —
#    a unique hit in the DB is not proof it's the RIGHT institution (see SVIT:
#    Gujarat's 'Sardar Vallabhbhai Institute of Technology' and whatever SVIT
#    the resume actually means can share the same acronym by coincidence)."""
#    for token in _ACRONYM_TOKEN_RE.findall(entry_text):
#        if token in _ACRONYM_EXCLUDE:
#            continue
#        candidates = _ACRONYM_INDEX.get(token)
#        if not candidates:
#            continue
#        if len(candidates) == 1:
#            return candidates[0]["name"], True
#        # Ambiguous — multiple real institutions share this acronym.
#        # Don't guess among them; surface the raw acronym and flag it.
#        return token, True
#    return None, False
#
#
#_TRAILING_WORD_RE = re.compile(r"^[A-Z][a-z.&'\-]*$")
#_INSTITUTION_KEYWORDS_RE = re.compile(
#    r'\b(university|institute|college|school|iit|nit|bits|iim|ignou)\b', re.IGNORECASE
#)
#
#
#def _match_keyword_fallback(entry_text: str) -> str | None:
#    m = _INSTITUTION_KEYWORDS_RE.search(entry_text)
#    if not m:
#        return None
#    start, end = m.start(), m.end()
#    left = _expand_left(entry_text, start)
#
#    # Expand right over Title-Case words only, stopping at a comma/digit/month
#    after = entry_text[end:].lstrip()
#    tokens_after = after.split(" ") if after else []
#    right = []
#    for tok in tokens_after:
#        clean = tok.rstrip(",")
#        if _TRAILING_WORD_RE.match(clean):
#            right.append(clean)
#            if clean != tok:  # a comma followed this word — stop after including it
#                break
#        else:
#            break
#
#    parts = left + [m.group(0)] + right
#    candidate = " ".join(parts).strip()
#    return candidate or None
#
#
#def _extract_institution(entry_text: str) -> tuple[str | None, bool]:
#    """Returns (institution, needs_review). Priority: full name in DB (safest)
#    -> acronym in DB (flagged, since unique-in-DB != correct-in-reality)
#    -> generic keyword regex (not DB-verified, but usually accurate when a
#    resume spells the name out in full)."""
#    full = _match_full_name(entry_text)
#    if full:
#        return full, False
#
#    acronym_match, needs_review = _match_acronym(entry_text)
#    if acronym_match:
#        return acronym_match, needs_review
#
#    fallback = _match_keyword_fallback(entry_text)
#    if fallback:
#        return fallback, False
#
#    return None, False
#
#
#def _extract_year(entry_text: str) -> str | None:
#    m = YEAR_RE.search(entry_text)
#    if not m:
#        return None
#    return m.group(1) or m.group(2)
#
#
#def _extract_gpa(entry_text: str) -> str | None:
#    m = GPA_KEYWORD_RE.search(entry_text)
#    if m:
#        return m.group(1).strip()
#    m = GPA_BARE_SCALE_RE.search(entry_text)
#    if m:
#        return f"{m.group(1)}/{m.group(2)}"
#    m = GPA_BARE_PERCENT_RE.search(entry_text)
#    if m:
#        return f"{m.group(1)}%"
#    return None
#
#
#def _extract_degree(trigger_line: str, degree_type: str) -> str:
#    """Best-effort clean degree label from the line that triggered this entry.
#    Cuts off at the first comma so trailing institution text doesn't leak in
#    (e.g. 'B.E. (Mechanical Engg.) SVIT, Pune' -> 'B.E. (Mechanical Engg.) SVIT'
#    — imperfect but keeps the label short; institution field carries the real name)."""
#    return trigger_line.strip()
#
#
#def extract_education(education_section_text: str) -> dict:
#    """
#    Parses the education section into a list of entries.
#
#    Entry boundaries are anchored on lines that match a degree pattern
#    ("trigger lines"), not on blank lines — see module docstring for why.
#    """
#    if not education_section_text.strip():
#        return {"education": [], "_confidence": 0.0}
#
#    lines = [l.strip() for l in education_section_text.strip().split("\n") if l.strip()]
#
#    # Find trigger line indices + which degree (canonical name + category) each matched
#    triggers = []  # list of (line_index, canonical_degree, category)
#    for i, line in enumerate(lines):
#        canonical, category = match_degree(line)
#        if canonical:
#            triggers.append((i, canonical, category))
#
#    if not triggers:
#        return {"education": [], "_confidence": 0.0}
#
#    # Build blocks: lines before the first trigger attach to entry 0;
#    # lines after a trigger attach to that entry until the next trigger.
#    blocks = []  # list of (canonical_degree, category, trigger_line, [lines])
#    for idx, (line_i, canonical, category) in enumerate(triggers):
#        start = 0 if idx == 0 else line_i
#        end = triggers[idx + 1][0] if idx + 1 < len(triggers) else len(lines)
#        block_lines = lines[start:end]
#        blocks.append((canonical, category, lines[line_i], block_lines))
#
#    education_entries = []
#    for canonical_degree, degree_type, trigger_line, block_lines in blocks:
#        entry_text = " ".join(block_lines)
#
#        institution, needs_review = _extract_institution(entry_text)
#        year = _extract_year(entry_text)
#        gpa = _extract_gpa(entry_text)
#        degree = _extract_degree(trigger_line, degree_type)
#
#        entry = {
#            "degree": degree,
#            "degree_name": canonical_degree,
#            "degree_type": degree_type,
#            "institution": institution,
#            "year": year,
#            "gpa": gpa,
#            "_needs_review": needs_review,
#        }
#
#        # Per-entry confidence: base + a bump for each field actually found,
#        # but capped lower when the institution came from an unverified
#        # acronym match (unique-in-DB is not the same as correct-in-reality).
#        fields_found = sum(1 for v in (degree_type, institution, year) if v)
#        base_confidence = min(0.5 + 0.15 * fields_found, 0.95)
#        if needs_review:
#            base_confidence = min(base_confidence, 0.55)
#        entry["_confidence"] = round(base_confidence, 2)
#
#        education_entries.append(entry)
#
#    overall_confidence = round(
#        sum(e["_confidence"] for e in education_entries) / len(education_entries), 2
#    ) if education_entries else 0.0
#
#    return {
#        "education": education_entries,
#        "_confidence": overall_confidence,
#    }