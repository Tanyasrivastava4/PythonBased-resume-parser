"""
extractors/skills.py
━━━━━━━━━━━━━━━━━━━
Layer 2 — Skills Extractor

Pipeline:
  Step 1 → FlashText exact/alias match on skills section ONLY
  Step 2 → RapidFuzz fuzzy match on skills section tokens (catches misspellings)

Primary source: skills section text only.
We do NOT scan experience/projects/full_text to avoid false positives.
"""

import json
import re
import time
from pathlib import Path

from flashtext import KeywordProcessor
from rapidfuzz import process, fuzz


_TAXONOMY_PATHS = [
    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",
    Path("data/skills_taxonomy.json"),
]

FUZZY_THRESHOLD    = 88
FUZZY_MIN_TOKEN    = 4
FUZZY_MAX_TOKEN    = 50

FUZZY_MAX_CANDIDATE_WORDS = 2
FUZZY_MIN_CANDIDATE_LEN   = 4
FUZZY_MIN_LEN_RATIO = 0.75
FUZZY_MAX_LEN_RATIO = 1.35

FUZZY_SKIP_TOKENS = {
    "education", "training", "audit", "compliance", "discovery",
    "management", "performance", "foundation", "integration",
    "architecture", "orchestration", "observability", "governance",
    "documentation", "communication", "certification", "automation",
    "monitoring", "planning", "analysis", "research", "reporting",
    "presentation", "presentations", "negotiation", "development",
    "relationship", "coordination", "administration", "execution",
    "implementation", "optimization", "optimisation", "engagement",
    "acquisition", "retention", "leadership", "collaboration",
    "networking", "forecasting", "budgeting", "scheduling",
}

TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")

_WORD_SPLIT_RE = re.compile(r"[\s\-]+")

FUZZY_MIN_FIRST_WORD_RATIO = 75   # for multi-word matches, the leading word of
                                    # the input token and the leading word of the
                                    # matched candidate must independently be this
                                    # similar. Confirmed necessary by direct
                                    # testing: "Log Analysis" scored 91.7% against
                                    # "lbo analysis" (an alias of "LBO") purely
                                    # because both share the exact trailing word
                                    # "analysis" -- WRatio's blended scoring lets a
                                    # single matching word carry the whole phrase
                                    # even when the OTHER word is unrelated ("log"
                                    # vs "lbo", 66.7% on its own). A real typo
                                    # phrase agrees on every word, not just one;
                                    # this check isolates and independently
                                    # verifies the word that ISN'T just coincidence.
                                    # Confirmed NOT to break legitimate multi-word
                                    # matches: "fine tuning" vs "fine-tuning" scores
                                    # 100% on this same check (first words "fine"/
                                    # "fine" are identical after hyphen-splitting).


def _load_taxonomy() -> dict:
    for path in _TAXONOMY_PATHS:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        "skills_taxonomy.json not found. "
        "Run build_skills_taxonomy.py first.\n"
        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
    )


def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
    kp = KeywordProcessor(case_sensitive=False)
    for canonical, entry in taxonomy.items():
        kp.add_keyword(canonical, canonical)
        for alias in entry.get("aliases", []):
            if alias.strip():
                kp.add_keyword(alias, canonical)
    return kp


def _build_fuzzy_pool(taxonomy: dict) -> dict:
    pool = {}
    for canonical, entry in taxonomy.items():
        canonical_lower = canonical.lower()
        if (len(canonical_lower.split()) <= FUZZY_MAX_CANDIDATE_WORDS
                and len(canonical_lower) >= FUZZY_MIN_CANDIDATE_LEN):
            pool[canonical_lower] = canonical

        for alias in entry.get("aliases", []):
            alias = alias.strip()
            if (len(alias.split()) <= FUZZY_MAX_CANDIDATE_WORDS
                    and len(alias) >= FUZZY_MIN_CANDIDATE_LEN):
                pool[alias] = canonical
    return pool


_TAXONOMY    = _load_taxonomy()
_KP          = _build_keyword_processor(_TAXONOMY)
_FUZZY_POOL  = _build_fuzzy_pool(_TAXONOMY)
_FUZZY_SEARCH_STRINGS = list(_FUZZY_POOL.keys())

print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills "
      f"({len(_FUZZY_SEARCH_STRINGS):,} fuzzy-searchable names+aliases).")


def _category_of(skill_name: str) -> str:
    entry = _TAXONOMY.get(skill_name)
    if entry:
        return entry.get("category", "other")
    return "other"


def _should_skip_fuzzy(token: str) -> bool:
    lower = token.strip().lower()
    if lower in FUZZY_SKIP_TOKENS:
        return True
    words = lower.split()
    if len(words) > 1 and all(w in FUZZY_SKIP_TOKENS for w in words):
        return True
    return False


def _best_fuzzy_match(token: str):
    """
    Finds the best fuzzy match for a token against the restricted
    fuzzy pool, applying two safeguards on top of the score threshold:

      1. Length-ratio check -- rejects matches where the input token's
         length is wildly different from the matched candidate's length.

      2. First-word similarity check (NEW) -- for multi-word matches
         only, independently verifies that the leading word of the
         token and the leading word of the matched candidate are
         actually similar to each other, not just the phrase as a
         whole. See FUZZY_MIN_FIRST_WORD_RATIO docstring above for why:
         a shared trailing word (e.g. "analysis") can make WRatio score
         a phrase as a near-total match even when the leading,
         meaning-carrying word is a completely different skill.
    """
    token_lower = token.lower()
    threshold = 83 if len(token_lower) <= 9 else FUZZY_THRESHOLD
    result = process.extractOne(
        token_lower, _FUZZY_SEARCH_STRINGS, scorer=fuzz.WRatio, score_cutoff=threshold
    )
    if not result:
        return None

    matched_string, score, _ = result
    ratio = len(token_lower) / len(matched_string)
    if not (FUZZY_MIN_LEN_RATIO <= ratio <= FUZZY_MAX_LEN_RATIO):
        return None

    token_words = _WORD_SPLIT_RE.split(token_lower)
    match_words = _WORD_SPLIT_RE.split(matched_string)
    if len(token_words) > 1 or len(match_words) > 1:
        first_word_score = fuzz.ratio(token_words[0], match_words[0])
        if first_word_score < FUZZY_MIN_FIRST_WORD_RATIO:
            return None

    return _FUZZY_POOL[matched_string], round(score, 1)


def _tokenize_for_fuzzy(text: str) -> list:
    raw_chunks = TOKEN_SPLIT_RE.split(text)
    tokens = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if FUZZY_MIN_TOKEN <= len(chunk) <= FUZZY_MAX_TOKEN:
            tokens.append(chunk)
            continue

        if len(chunk) > FUZZY_MAX_TOKEN and not TOKEN_SPLIT_RE.search(chunk):
            for word in chunk.split():
                word = word.strip()
                if FUZZY_MIN_TOKEN <= len(word) <= FUZZY_MAX_TOKEN:
                    tokens.append(word)

    return tokens


def extract_skills(sections: dict, full_text: str = "") -> dict:
    t0 = time.perf_counter()

    found_skills  = set()
    methods_used  = []

    skills_text = sections.get("skills", "") or ""
    skills_text = skills_text.replace("\n", " ")
    skills_text = " ".join(skills_text.split())

    if skills_text.strip():
        matches = _KP.extract_keywords(skills_text)
        if matches:
            found_skills.update(matches)
            methods_used.append("flashtext_skills_section")

    if skills_text.strip():
        tokens     = _tokenize_for_fuzzy(skills_text)
        fuzzy_hits = []

        for token in tokens:
            if token in found_skills:
                continue
            if _should_skip_fuzzy(token):
                continue

            result = _best_fuzzy_match(token)
            if result:
                canonical_match, score = result
                found_skills.add(canonical_match)
                fuzzy_hits.append(f"{token}→{canonical_match}({score:.0f}%)")

        if fuzzy_hits:
            preview = ", ".join(fuzzy_hits[:5])
            suffix  = "..." if len(fuzzy_hits) > 5 else ""
            methods_used.append(f"fuzzy({preview}{suffix})")

    sorted_skills = sorted(found_skills)

    skills_by_category = {}
    for skill in sorted_skills:
        cat = _category_of(skill)
        skills_by_category.setdefault(cat, []).append(skill)

    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
        confidence = 0.92
    elif len(sorted_skills) >= 1:
        confidence = 0.75
    else:
        confidence = 0.0

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "skills":              sorted_skills,
        "skills_by_category":  skills_by_category,
        "_confidence":         confidence,
        "_method":             " | ".join(methods_used) if methods_used else "none",
        "_timing_ms":          elapsed_ms,
    }
















##worked chaing for new resumes to workk
#"""
#extractors/skills.py
#━━━━━━━━━━━━━━━━━━━
#Layer 2 — Skills Extractor
#
#Pipeline:
#  Step 1 → FlashText exact/alias match on skills section ONLY
#  Step 2 → RapidFuzz fuzzy match on skills section tokens (catches misspellings)
#
#Primary source: skills section text only.
#We do NOT scan experience/projects/full_text to avoid false positives.
#
#Input:  sections dict from segmenter  (only "skills" key is used)
#        full_text string (kept for API compatibility, not used)
#
#Output: {
#    "skills": ["Python", "AWS", "React", ...],
#    "skills_by_category": {
#        "programming_language": ["Python"],
#        "cloud_infra": ["AWS"],
#        ...
#    },
#    "_confidence": 0.92,
#    "_method": "flashtext_skills_section | fuzzy(...)",
#    "_timing_ms": 12.4
#}
#"""
#
#import json
#import re
#import time
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import process, fuzz
#
#
## ──────────────────────────────────────────────────────────────
## TAXONOMY PATH
## ──────────────────────────────────────────────────────────────
#_TAXONOMY_PATHS = [
#    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",
#    Path("data/skills_taxonomy.json"),
#]
#
## ──────────────────────────────────────────────────────────────
## FUZZY MATCH SETTINGS
## ──────────────────────────────────────────────────────────────
#FUZZY_THRESHOLD    = 88   # minimum similarity % to accept a match
#FUZZY_MIN_TOKEN    = 4    # ignore input tokens shorter than this
#FUZZY_MAX_TOKEN    = 50   # ignore tokens longer than this (probably a sentence)
#
## These three settings control which CANDIDATE strings (canonical names
## and aliases) are even allowed to be fuzzy-matched against. This is a
## different, more precisely-targeted approach than restricting which
## INPUT tokens attempt fuzzy matching (e.g. a flat "skip tokens under
## 12 characters" rule) -- that approach was tried and confirmed broken:
## it blocked virtually every realistic short typo ("Pyhton", "Dockerr",
## etc, all under 12 chars) while not actually addressing the real cause.
##
## The real cause, confirmed by direct testing: short common words like
## "Education" or "Audit" score ~90% against long, unrelated O*NET
## product names (e.g. "Pearson Education PHStat2", "ACL Audit
## Exchange") because WRatio's partial-match logic rewards a short
## string being substring-like inside a much longer one. This has
## nothing to do with how long the TYPED word is -- it's entirely about
## what kind of CANDIDATE it's being compared against. So we fix it by
## restricting the candidate pool instead:
#FUZZY_MAX_CANDIDATE_WORDS = 2     # exclude 3+ word canonical names entirely
#FUZZY_MIN_CANDIDATE_LEN   = 4     # exclude very short candidates (e.g. "R") --
#                                   # these cause false positives the opposite
#                                   # way (any word containing that letter
#                                   # scores deceptively high)
#FUZZY_MIN_LEN_RATIO = 0.75   # the input token must be at least this fraction
#FUZZY_MAX_LEN_RATIO = 1.35   # of the matched candidate's length, and at most
#                              # this multiple of it. A real typo is always
#                              # close in length to the correct word -- this
#                              # is what rejects e.g. "Discovery" (9 chars)
#                              # matching "BioDiscoveryImaGene" (20 chars,
#                              # ratio 0.47) even after the word-count and
#                              # min-length filters, since that's a single
#                              # compound word, not a multi-word phrase.
#
## Common single words that should never be fuzzy matched alone, as a
## final belt-and-suspenders layer on top of the candidate-side
## restrictions above.
#FUZZY_SKIP_TOKENS = {
#    "education", "training", "audit", "compliance", "discovery",
#    "management", "performance", "foundation", "integration",
#    "architecture", "orchestration", "observability", "governance",
#    "documentation", "communication", "certification", "automation",
#    "monitoring", "planning", "analysis", "research", "reporting",
#    "presentation", "presentations", "negotiation", "development",
#    "relationship", "coordination", "administration", "execution",
#    "implementation", "optimization", "optimisation", "engagement",
#    "acquisition", "retention", "leadership", "collaboration",
#    "networking", "forecasting", "budgeting", "scheduling",
#}
#
## ──────────────────────────────────────────────────────────────
## TOKEN SPLIT PATTERN
## Split on: comma, pipe, bullet, newline, tab, slash, ampersand etc.
## ──────────────────────────────────────────────────────────────
#TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")
#
#
## ══════════════════════════════════════════════════════════════
## MODULE-LEVEL INITIALISATION  (runs once at import time)
## ══════════════════════════════════════════════════════════════
#
#def _load_taxonomy() -> dict:
#    for path in _TAXONOMY_PATHS:
#        if path.exists():
#            with open(path, "r", encoding="utf-8") as f:
#                return json.load(f)
#    raise FileNotFoundError(
#        "skills_taxonomy.json not found. "
#        "Run build_skills_taxonomy.py first.\n"
#        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
#    )
#
#
#def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
#    """
#    Build FlashText processor from taxonomy.
#    Each alias maps back to canonical name.
#    e.g. "reactjs" → "React", "k8s" → "Kubernetes"
#    """
#    kp = KeywordProcessor(case_sensitive=False)
#    for canonical, entry in taxonomy.items():
#        kp.add_keyword(canonical, canonical)
#        for alias in entry.get("aliases", []):
#            if alias.strip():
#                kp.add_keyword(alias, canonical)
#    return kp
#
#
#def _build_fuzzy_pool(taxonomy: dict) -> dict:
#    """
#    Builds the restricted lookup used for fuzzy matching: every alias
#    AND every canonical name (lowercased) mapped back to its canonical
#    name -- but excluding entries that are too risky for fuzzy matching.
#
#    BUG FIX: the word-count check must be applied to EACH STRING
#    INDIVIDUALLY (the canonical name, and separately, every alias) --
#    not just once for the canonical name, with aliases let through
#    automatically afterward. Confirmed by direct testing: "AWS" itself
#    is 1 word (passes), but its alias "amazon web services" is 3 words
#    and should be excluded on its own merits. The original version of
#    this fix checked only the canonical name's word count and then
#    added ALL of its aliases unconditionally -- so "amazon web
#    services" slipped through anyway, and the ordinary word "services"
#    scored 90% against it (the exact same risk pattern as "Education"
#    matching "Pearson Education PHStat2"). Each string -- canonical
#    name or alias -- now passes through the same word-count and
#    length checks independently.
#    """
#    pool = {}
#    for canonical, entry in taxonomy.items():
#        canonical_lower = canonical.lower()
#        if (len(canonical_lower.split()) <= FUZZY_MAX_CANDIDATE_WORDS
#                and len(canonical_lower) >= FUZZY_MIN_CANDIDATE_LEN):
#            pool[canonical_lower] = canonical
#
#        for alias in entry.get("aliases", []):
#            alias = alias.strip()
#            if (len(alias.split()) <= FUZZY_MAX_CANDIDATE_WORDS
#                    and len(alias) >= FUZZY_MIN_CANDIDATE_LEN):
#                pool[alias] = canonical
#    return pool
#
#
## ── Load once at import ──
#_TAXONOMY    = _load_taxonomy()
#_KP          = _build_keyword_processor(_TAXONOMY)
#_FUZZY_POOL  = _build_fuzzy_pool(_TAXONOMY)
#_FUZZY_SEARCH_STRINGS = list(_FUZZY_POOL.keys())
#
#print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills "
#      f"({len(_FUZZY_SEARCH_STRINGS):,} fuzzy-searchable names+aliases).")
#
#
## ══════════════════════════════════════════════════════════════
## HELPER FUNCTIONS
## ══════════════════════════════════════════════════════════════
#
#def _category_of(skill_name: str) -> str:
#    entry = _TAXONOMY.get(skill_name)
#    if entry:
#        return entry.get("category", "other")
#    return "other"
#
#
#def _should_skip_fuzzy(token: str) -> bool:
#    """
#    Returns True if this token should be skipped for fuzzy matching
#    entirely, before even attempting it.
#
#    This is now a much smaller safety net than before -- the primary
#    defense against false positives is the restricted candidate pool
#    (FUZZY_MAX_CANDIDATE_WORDS / FUZZY_MIN_CANDIDATE_LEN, applied in
#    _build_fuzzy_pool) plus the length-ratio check applied after
#    matching in _best_fuzzy_match. This function only catches the case
#    where the token itself is an exact known generic word, as a final
#    explicit belt-and-suspenders layer.
#    """
#    lower = token.strip().lower()
#    if lower in FUZZY_SKIP_TOKENS:
#        return True
#    words = lower.split()
#    if len(words) > 1 and all(w in FUZZY_SKIP_TOKENS for w in words):
#        return True
#    return False
#
#
#def _best_fuzzy_match(token: str):
#    """
#    Finds the best fuzzy match for a token against the restricted
#    fuzzy pool, applying a length-ratio safeguard on top of the score
#    threshold. Returns (canonical_name, score) or None.
#
#    The length-ratio check rejects matches where the input token's
#    length is wildly different from the matched candidate's length --
#    confirmed necessary by direct testing: "Discovery" (9 chars) scored
#    90%+ against "BioDiscoveryImaGene" (a single compound word, so the
#    word-count filter alone couldn't exclude it) purely because
#    "Discovery" is a near-complete substring of the much longer name.
#    A real typo is always close in length to the correct word it's a
#    typo of (ratio near 1.0); a short common word accidentally
#    matching as a substring of an unrelated long name is not.
#    """
#    token_lower = token.lower()
#    # Short tokens (typos of short words like "Python", "Docker") need
#    # a slightly lower threshold — transposition errors score ~83-85%.
#    # Longer tokens are kept at the standard threshold.
#    threshold = 83 if len(token_lower) <= 9 else FUZZY_THRESHOLD
#    result = process.extractOne(
#        token_lower, _FUZZY_SEARCH_STRINGS, scorer=fuzz.WRatio, score_cutoff=threshold
#    )
#    if not result:
#        return None
#
#    matched_string, score, _ = result
#    ratio = len(token_lower) / len(matched_string)
#    if not (FUZZY_MIN_LEN_RATIO <= ratio <= FUZZY_MAX_LEN_RATIO):
#        return None
#
#    return _FUZZY_POOL[matched_string], round(score, 1)
#
#
#def _tokenize_for_fuzzy(text: str) -> list:
#    """
#    Split skills section into individual tokens for fuzzy matching.
#    Each token = one candidate skill name FlashText didn't match exactly.
#
#    Primary split: comma/pipe/bullet/newline/tab/slash/&/;/()/[] --
#    covers the vast majority of resumes (comma or bullet separated
#    skills lists). Chunks produced this way are NOT further split on
#    whitespace, so multi-word skill names like "Power BI" or "Machine
#    Learning" stay intact as single tokens for the comma/bullet case.
#
#    Fallback (fixes the "space-only skills list" gap): if a resulting
#    chunk is still oversized (> FUZZY_MAX_TOKEN) AND contains none of
#    the delimiter characters above, it's very likely a resume that
#    listed its skills separated ONLY by spaces, e.g.
#    "Python Java AWS Docker" with no commas/bullets anywhere. Without
#    this fallback that entire line becomes one unmatchable token and
#    every skill in it silently skips the fuzzy/typo-catching step
#    (FlashText's exact-match step still works fine on it, since it
#    scans continuous text -- only fuzzy matching was affected). In
#    that specific case, and only that case, we additionally split on
#    whitespace so each individual word still gets a fuzzy check.
#    """
#    raw_chunks = TOKEN_SPLIT_RE.split(text)
#    tokens = []
#
#    for chunk in raw_chunks:
#        chunk = chunk.strip()
#        if not chunk:
#            continue
#
#        if FUZZY_MIN_TOKEN <= len(chunk) <= FUZZY_MAX_TOKEN:
#            tokens.append(chunk)
#            continue
#
#        # Oversized chunk with no leftover delimiter punctuation →
#        # treat as a space-only separated skills list and split further.
#        # (An oversized chunk that STILL contains a delimiter character
#        # would mean TOKEN_SPLIT_RE already tried to split it and this
#        # piece is a genuine leftover long fragment, not a skills list
#        # -- so we deliberately do NOT fall back to whitespace-splitting
#        # in that case.)
#        if len(chunk) > FUZZY_MAX_TOKEN and not TOKEN_SPLIT_RE.search(chunk):
#            for word in chunk.split():
#                word = word.strip()
#                if FUZZY_MIN_TOKEN <= len(word) <= FUZZY_MAX_TOKEN:
#                    tokens.append(word)
#
#    return tokens
#
#
## ══════════════════════════════════════════════════════════════
## MAIN EXTRACTOR FUNCTION
## ══════════════════════════════════════════════════════════════
#
#def extract_skills(sections: dict, full_text: str = "") -> dict:
#    """
#    Extract skills from the skills section of a resume only.
#
#    Args:
#        sections  : dict from segmenter. Only "skills" key is used.
#        full_text : kept for API compatibility, not used.
#
#    Returns:
#        {
#            "skills":             sorted list of canonical skill names,
#            "skills_by_category": dict of category -> [skills],
#            "_confidence":        float 0.0-1.0,
#            "_method":            which steps fired,
#            "_timing_ms":         time taken
#        }
#    """
#    t0 = time.perf_counter()
#
#    found_skills  = set()
#    methods_used  = []
#
#    # ── PRIMARY SOURCE: skills section only ───────────────────
#    skills_text = sections.get("skills", "") or ""
#
#    # Normalise line breaks that split multi-word skill names.
#    # e.g. PDF sometimes extracts "Power\nBI" across two lines.
#    # Replace newlines with a space so "Power BI" stays intact
#    # for FlashText to match correctly.
#    skills_text = skills_text.replace("\n", " ")
#    skills_text = " ".join(skills_text.split())   # collapse multiple spaces
#
#    # ── STEP 1: FlashText exact + alias match ─────────────────
#    # Scans the entire skills section in one pass.
#    # Catches: "reactjs"→"React", "k8s"→"Kubernetes", "postgres"→"PostgreSQL"
#    if skills_text.strip():
#        matches = _KP.extract_keywords(skills_text)
#        if matches:
#            found_skills.update(matches)
#            methods_used.append("flashtext_skills_section")
#
#    # ── STEP 2: RapidFuzz fuzzy match ─────────────────────────
#    # Tokenize skills section and fuzzy-match each token.
#    # Catches misspellings: "Postgress"→"PostgreSQL", "Kubernets"→"Kubernetes"
#    # Skips short/generic tokens to prevent false positives.
#    if skills_text.strip():
#        tokens     = _tokenize_for_fuzzy(skills_text)
#        fuzzy_hits = []
#
#        for token in tokens:
#            # Already found exactly by FlashText — skip
#            if token in found_skills:
#                continue
#
#            # Too short or too generic — skip fuzzy for this token
#            if _should_skip_fuzzy(token):
#                continue
#
#            result = _best_fuzzy_match(token)
#            if result:
#                canonical_match, score = result
#                found_skills.add(canonical_match)
#                fuzzy_hits.append(f"{token}→{canonical_match}({score:.0f}%)")
#
#        if fuzzy_hits:
#            preview = ", ".join(fuzzy_hits[:5])
#            suffix  = "..." if len(fuzzy_hits) > 5 else ""
#            methods_used.append(f"fuzzy({preview}{suffix})")
#
#    # ── BUILD OUTPUT ──────────────────────────────────────────
#    sorted_skills = sorted(found_skills)
#
#    skills_by_category = {}
#    for skill in sorted_skills:
#        cat = _category_of(skill)
#        skills_by_category.setdefault(cat, []).append(skill)
#
#    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
#        confidence = 0.92
#    elif len(sorted_skills) >= 1:
#        confidence = 0.75
#    else:
#        confidence = 0.0
#
#    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
#
#    return {
#        "skills":              sorted_skills,
#        "skills_by_category":  skills_by_category,
#        "_confidence":         confidence,
#        "_method":             " | ".join(methods_used) if methods_used else "none",
#        "_timing_ms":          elapsed_ms,
#    }
#








##commenting for adding space logic here
#"""
#extractors/skills.py
#━━━━━━━━━━━━━━━━━━━
#Layer 2 — Skills Extractor
#
#Pipeline:
#  Step 1 → FlashText exact/alias match on skills section ONLY
#  Step 2 → RapidFuzz fuzzy match on skills section tokens (catches misspellings)
#
#Primary source: skills section text only.
#We do NOT scan experience/projects/full_text to avoid false positives.
#
#Input:  sections dict from segmenter  (only "skills" key is used)
#        full_text string (kept for API compatibility, not used)
#
#Output: {
#    "skills": ["Python", "AWS", "React", ...],
#    "skills_by_category": {
#        "programming_language": ["Python"],
#        "cloud_infra": ["AWS"],
#        ...
#    },
#    "_confidence": 0.92,
#    "_method": "flashtext_skills_section | fuzzy(...)",
#    "_timing_ms": 12.4
#}
#"""
#
#import json
#import re
#import time
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import process, fuzz
#
#
## ──────────────────────────────────────────────────────────────
## TAXONOMY PATH
## ──────────────────────────────────────────────────────────────
#_TAXONOMY_PATHS = [
#    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",
#    Path("data/skills_taxonomy.json"),
#]
#
## ──────────────────────────────────────────────────────────────
## FUZZY MATCH SETTINGS
## ──────────────────────────────────────────────────────────────
#FUZZY_THRESHOLD    = 88   # minimum similarity % to accept a match
#FUZZY_MIN_TOKEN    = 4    # ignore input tokens shorter than this
#FUZZY_MAX_TOKEN    = 50   # ignore tokens longer than this (probably a sentence)
#
## These three settings control which CANDIDATE strings (canonical names
## and aliases) are even allowed to be fuzzy-matched against. This is a
## different, more precisely-targeted approach than restricting which
## INPUT tokens attempt fuzzy matching (e.g. a flat "skip tokens under
## 12 characters" rule) -- that approach was tried and confirmed broken:
## it blocked virtually every realistic short typo ("Pyhton", "Dockerr",
## etc, all under 12 chars) while not actually addressing the real cause.
##
## The real cause, confirmed by direct testing: short common words like
## "Education" or "Audit" score ~90% against long, unrelated O*NET
## product names (e.g. "Pearson Education PHStat2", "ACL Audit
## Exchange") because WRatio's partial-match logic rewards a short
## string being substring-like inside a much longer one. This has
## nothing to do with how long the TYPED word is -- it's entirely about
## what kind of CANDIDATE it's being compared against. So we fix it by
## restricting the candidate pool instead:
#FUZZY_MAX_CANDIDATE_WORDS = 2     # exclude 3+ word canonical names entirely
#FUZZY_MIN_CANDIDATE_LEN   = 4     # exclude very short candidates (e.g. "R") --
#                                   # these cause false positives the opposite
#                                   # way (any word containing that letter
#                                   # scores deceptively high)
#FUZZY_MIN_LEN_RATIO = 0.75   # the input token must be at least this fraction
#FUZZY_MAX_LEN_RATIO = 1.35   # of the matched candidate's length, and at most
#                              # this multiple of it. A real typo is always
#                              # close in length to the correct word -- this
#                              # is what rejects e.g. "Discovery" (9 chars)
#                              # matching "BioDiscoveryImaGene" (20 chars,
#                              # ratio 0.47) even after the word-count and
#                              # min-length filters, since that's a single
#                              # compound word, not a multi-word phrase.
#
## Common single words that should never be fuzzy matched alone, as a
## final belt-and-suspenders layer on top of the candidate-side
## restrictions above.
#FUZZY_SKIP_TOKENS = {
#    "education", "training", "audit", "compliance", "discovery",
#    "management", "performance", "foundation", "integration",
#    "architecture", "orchestration", "observability", "governance",
#    "documentation", "communication", "certification", "automation",
#    "monitoring", "planning", "analysis", "research", "reporting",
#    "presentation", "presentations", "negotiation", "development",
#    "relationship", "coordination", "administration", "execution",
#    "implementation", "optimization", "optimisation", "engagement",
#    "acquisition", "retention", "leadership", "collaboration",
#    "networking", "forecasting", "budgeting", "scheduling",
#}
#
## ──────────────────────────────────────────────────────────────
## TOKEN SPLIT PATTERN
## Split on: comma, pipe, bullet, newline, tab, slash, ampersand etc.
## ──────────────────────────────────────────────────────────────
#TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")
#
#
## ══════════════════════════════════════════════════════════════
## MODULE-LEVEL INITIALISATION  (runs once at import time)
## ══════════════════════════════════════════════════════════════
#
#def _load_taxonomy() -> dict:
#    for path in _TAXONOMY_PATHS:
#        if path.exists():
#            with open(path, "r", encoding="utf-8") as f:
#                return json.load(f)
#    raise FileNotFoundError(
#        "skills_taxonomy.json not found. "
#        "Run build_skills_taxonomy.py first.\n"
#        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
#    )
#
#
#def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
#    """
#    Build FlashText processor from taxonomy.
#    Each alias maps back to canonical name.
#    e.g. "reactjs" → "React", "k8s" → "Kubernetes"
#    """
#    kp = KeywordProcessor(case_sensitive=False)
#    for canonical, entry in taxonomy.items():
#        kp.add_keyword(canonical, canonical)
#        for alias in entry.get("aliases", []):
#            if alias.strip():
#                kp.add_keyword(alias, canonical)
#    return kp
#
#
#def _build_fuzzy_pool(taxonomy: dict) -> dict:
#    """
#    Builds the restricted lookup used for fuzzy matching: every alias
#    AND every canonical name (lowercased) mapped back to its canonical
#    name -- but excluding entries that are too risky for fuzzy matching.
#
#    BUG FIX: the word-count check must be applied to EACH STRING
#    INDIVIDUALLY (the canonical name, and separately, every alias) --
#    not just once for the canonical name, with aliases let through
#    automatically afterward. Confirmed by direct testing: "AWS" itself
#    is 1 word (passes), but its alias "amazon web services" is 3 words
#    and should be excluded on its own merits. The original version of
#    this fix checked only the canonical name's word count and then
#    added ALL of its aliases unconditionally -- so "amazon web
#    services" slipped through anyway, and the ordinary word "services"
#    scored 90% against it (the exact same risk pattern as "Education"
#    matching "Pearson Education PHStat2"). Each string -- canonical
#    name or alias -- now passes through the same word-count and
#    length checks independently.
#    """
#    pool = {}
#    for canonical, entry in taxonomy.items():
#        canonical_lower = canonical.lower()
#        if (len(canonical_lower.split()) <= FUZZY_MAX_CANDIDATE_WORDS
#                and len(canonical_lower) >= FUZZY_MIN_CANDIDATE_LEN):
#            pool[canonical_lower] = canonical
#
#        for alias in entry.get("aliases", []):
#            alias = alias.strip()
#            if (len(alias.split()) <= FUZZY_MAX_CANDIDATE_WORDS
#                    and len(alias) >= FUZZY_MIN_CANDIDATE_LEN):
#                pool[alias] = canonical
#    return pool
#
#
## ── Load once at import ──
#_TAXONOMY    = _load_taxonomy()
#_KP          = _build_keyword_processor(_TAXONOMY)
#_FUZZY_POOL  = _build_fuzzy_pool(_TAXONOMY)
#_FUZZY_SEARCH_STRINGS = list(_FUZZY_POOL.keys())
#
#print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills "
#      f"({len(_FUZZY_SEARCH_STRINGS):,} fuzzy-searchable names+aliases).")
#
#
## ══════════════════════════════════════════════════════════════
## HELPER FUNCTIONS
## ══════════════════════════════════════════════════════════════
#
#def _category_of(skill_name: str) -> str:
#    entry = _TAXONOMY.get(skill_name)
#    if entry:
#        return entry.get("category", "other")
#    return "other"
#
#
#def _should_skip_fuzzy(token: str) -> bool:
#    """
#    Returns True if this token should be skipped for fuzzy matching
#    entirely, before even attempting it.
#
#    This is now a much smaller safety net than before -- the primary
#    defense against false positives is the restricted candidate pool
#    (FUZZY_MAX_CANDIDATE_WORDS / FUZZY_MIN_CANDIDATE_LEN, applied in
#    _build_fuzzy_pool) plus the length-ratio check applied after
#    matching in _best_fuzzy_match. This function only catches the case
#    where the token itself is an exact known generic word, as a final
#    explicit belt-and-suspenders layer.
#    """
#    lower = token.strip().lower()
#    if lower in FUZZY_SKIP_TOKENS:
#        return True
#    words = lower.split()
#    if len(words) > 1 and all(w in FUZZY_SKIP_TOKENS for w in words):
#        return True
#    return False
#
#
#def _best_fuzzy_match(token: str):
#    """
#    Finds the best fuzzy match for a token against the restricted
#    fuzzy pool, applying a length-ratio safeguard on top of the score
#    threshold. Returns (canonical_name, score) or None.
#
#    The length-ratio check rejects matches where the input token's
#    length is wildly different from the matched candidate's length --
#    confirmed necessary by direct testing: "Discovery" (9 chars) scored
#    90%+ against "BioDiscoveryImaGene" (a single compound word, so the
#    word-count filter alone couldn't exclude it) purely because
#    "Discovery" is a near-complete substring of the much longer name.
#    A real typo is always close in length to the correct word it's a
#    typo of (ratio near 1.0); a short common word accidentally
#    matching as a substring of an unrelated long name is not.
#    """
#    token_lower = token.lower()
#    # Short tokens (typos of short words like "Python", "Docker") need
#    # a slightly lower threshold — transposition errors score ~83-85%.
#    # Longer tokens are kept at the standard threshold.
#    threshold = 83 if len(token_lower) <= 9 else FUZZY_THRESHOLD
#    result = process.extractOne(
#        token_lower, _FUZZY_SEARCH_STRINGS, scorer=fuzz.WRatio, score_cutoff=threshold
#    )
#    if not result:
#        return None
#
#    matched_string, score, _ = result
#    ratio = len(token_lower) / len(matched_string)
#    if not (FUZZY_MIN_LEN_RATIO <= ratio <= FUZZY_MAX_LEN_RATIO):
#        return None
#
#    return _FUZZY_POOL[matched_string], round(score, 1)
#
#
#def _tokenize_for_fuzzy(text: str) -> list:
#    """
#    Split skills section into individual tokens for fuzzy matching.
#    Each token = one candidate skill name FlashText didn't match exactly.
#    """
#    raw_tokens = TOKEN_SPLIT_RE.split(text)
#    tokens = []
#    for t in raw_tokens:
#        t = t.strip()
#        if FUZZY_MIN_TOKEN <= len(t) <= FUZZY_MAX_TOKEN:
#            tokens.append(t)
#    return tokens
#
#
## ══════════════════════════════════════════════════════════════
## MAIN EXTRACTOR FUNCTION
## ══════════════════════════════════════════════════════════════
#
#def extract_skills(sections: dict, full_text: str = "") -> dict:
#    """
#    Extract skills from the skills section of a resume only.
#
#    Args:
#        sections  : dict from segmenter. Only "skills" key is used.
#        full_text : kept for API compatibility, not used.
#
#    Returns:
#        {
#            "skills":             sorted list of canonical skill names,
#            "skills_by_category": dict of category -> [skills],
#            "_confidence":        float 0.0-1.0,
#            "_method":            which steps fired,
#            "_timing_ms":         time taken
#        }
#    """
#    t0 = time.perf_counter()
#
#    found_skills  = set()
#    methods_used  = []
#
#    # ── PRIMARY SOURCE: skills section only ───────────────────
#    skills_text = sections.get("skills", "") or ""
#
#    # Normalise line breaks that split multi-word skill names.
#    # e.g. PDF sometimes extracts "Power\nBI" across two lines.
#    # Replace newlines with a space so "Power BI" stays intact
#    # for FlashText to match correctly.
#    skills_text = skills_text.replace("\n", " ")
#    skills_text = " ".join(skills_text.split())   # collapse multiple spaces
#
#    # ── STEP 1: FlashText exact + alias match ─────────────────
#    # Scans the entire skills section in one pass.
#    # Catches: "reactjs"→"React", "k8s"→"Kubernetes", "postgres"→"PostgreSQL"
#    if skills_text.strip():
#        matches = _KP.extract_keywords(skills_text)
#        if matches:
#            found_skills.update(matches)
#            methods_used.append("flashtext_skills_section")
#
#    # ── STEP 2: RapidFuzz fuzzy match ─────────────────────────
#    # Tokenize skills section and fuzzy-match each token.
#    # Catches misspellings: "Postgress"→"PostgreSQL", "Kubernets"→"Kubernetes"
#    # Skips short/generic tokens to prevent false positives.
#    if skills_text.strip():
#        tokens     = _tokenize_for_fuzzy(skills_text)
#        fuzzy_hits = []
#
#        for token in tokens:
#            # Already found exactly by FlashText — skip
#            if token in found_skills:
#                continue
#
#            # Too short or too generic — skip fuzzy for this token
#            if _should_skip_fuzzy(token):
#                continue
#
#            result = _best_fuzzy_match(token)
#            if result:
#                canonical_match, score = result
#                found_skills.add(canonical_match)
#                fuzzy_hits.append(f"{token}→{canonical_match}({score:.0f}%)")
#
#        if fuzzy_hits:
#            preview = ", ".join(fuzzy_hits[:5])
#            suffix  = "..." if len(fuzzy_hits) > 5 else ""
#            methods_used.append(f"fuzzy({preview}{suffix})")
#
#    # ── BUILD OUTPUT ──────────────────────────────────────────
#    sorted_skills = sorted(found_skills)
#
#    skills_by_category = {}
#    for skill in sorted_skills:
#        cat = _category_of(skill)
#        skills_by_category.setdefault(cat, []).append(skill)
#
#    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
#        confidence = 0.92
#    elif len(sorted_skills) >= 1:
#        confidence = 0.75
#    else:
#        confidence = 0.0
#
#    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
#
#    return {
#        "skills":              sorted_skills,
#        "skills_by_category":  skills_by_category,
#        "_confidence":         confidence,
#        "_method":             " | ".join(methods_used) if methods_used else "none",
#        "_timing_ms":          elapsed_ms,
#    }
#

















#"""
#extractors/skills.py
#━━━━━━━━━━━━━━━━━━━
#Layer 2 — Skills Extractor
#
#Pipeline:
#  Step 1 → FlashText exact/alias match on skills section ONLY
#  Step 2 → RapidFuzz fuzzy match on skills section tokens (catches misspellings)
#
#Primary source: skills section text only.
#We do NOT scan experience/projects/full_text to avoid false positives.
#
#Input:  sections dict from segmenter  (only "skills" key is used)
#        full_text string (kept for API compatibility, not used)
#
#Output: {
#    "skills": ["Python", "AWS", "React", ...],
#    "skills_by_category": {
#        "programming_language": ["Python"],
#        "cloud_infra": ["AWS"],
#        ...
#    },
#    "_confidence": 0.92,
#    "_method": "flashtext_skills_section | fuzzy(...)",
#    "_timing_ms": 12.4
#}
#"""
#
#import json
#import re
#import time
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import process, fuzz
#
#
## ──────────────────────────────────────────────────────────────
## TAXONOMY PATH
## ──────────────────────────────────────────────────────────────
#_TAXONOMY_PATHS = [
#    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",
#    Path("data/skills_taxonomy.json"),
#]
#
## ──────────────────────────────────────────────────────────────
## FUZZY MATCH SETTINGS
## ──────────────────────────────────────────────────────────────
#FUZZY_THRESHOLD    = 88   # minimum similarity % to accept a match
#FUZZY_MIN_TOKEN    = 4    # ignore input tokens shorter than this
#FUZZY_MAX_TOKEN    = 50   # ignore tokens longer than this (probably a sentence)
#
## These three settings control which CANDIDATE strings (canonical names
## and aliases) are even allowed to be fuzzy-matched against. This is a
## different, more precisely-targeted approach than restricting which
## INPUT tokens attempt fuzzy matching (e.g. a flat "skip tokens under
## 12 characters" rule) -- that approach was tried and confirmed broken:
## it blocked virtually every realistic short typo ("Pyhton", "Dockerr",
## etc, all under 12 chars) while not actually addressing the real cause.
##
## The real cause, confirmed by direct testing: short common words like
## "Education" or "Audit" score ~90% against long, unrelated O*NET
## product names (e.g. "Pearson Education PHStat2", "ACL Audit
## Exchange") because WRatio's partial-match logic rewards a short
## string being substring-like inside a much longer one. This has
## nothing to do with how long the TYPED word is -- it's entirely about
## what kind of CANDIDATE it's being compared against. So we fix it by
## restricting the candidate pool instead:
#FUZZY_MAX_CANDIDATE_WORDS = 2     # exclude 3+ word canonical names entirely
#FUZZY_MIN_CANDIDATE_LEN   = 4     # exclude very short candidates (e.g. "R") --
#                                   # these cause false positives the opposite
#                                   # way (any word containing that letter
#                                   # scores deceptively high)
#FUZZY_MIN_LEN_RATIO = 0.75   # the input token must be at least this fraction
#FUZZY_MAX_LEN_RATIO = 1.35   # of the matched candidate's length, and at most
#                              # this multiple of it. A real typo is always
#                              # close in length to the correct word -- this
#                              # is what rejects e.g. "Discovery" (9 chars)
#                              # matching "BioDiscoveryImaGene" (20 chars,
#                              # ratio 0.47) even after the word-count and
#                              # min-length filters, since that's a single
#                              # compound word, not a multi-word phrase.
#
## Common single words that should never be fuzzy matched alone, as a
## final belt-and-suspenders layer on top of the candidate-side
## restrictions above.
#FUZZY_SKIP_TOKENS = {
#    "education", "training", "audit", "compliance", "discovery",
#    "management", "performance", "foundation", "integration",
#    "architecture", "orchestration", "observability", "governance",
#    "documentation", "communication", "certification", "automation",
#    "monitoring", "planning", "analysis", "research", "reporting",
#    "presentation", "presentations", "negotiation", "development",
#    "relationship", "coordination", "administration", "execution",
#    "implementation", "optimization", "optimisation", "engagement",
#    "acquisition", "retention", "leadership", "collaboration",
#    "networking", "forecasting", "budgeting", "scheduling",
#}
#
## ──────────────────────────────────────────────────────────────
## TOKEN SPLIT PATTERN
## Split on: comma, pipe, bullet, newline, tab, slash, ampersand etc.
## ──────────────────────────────────────────────────────────────
#TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")
#
#
## ══════════════════════════════════════════════════════════════
## MODULE-LEVEL INITIALISATION  (runs once at import time)
## ══════════════════════════════════════════════════════════════
#
#def _load_taxonomy() -> dict:
#    for path in _TAXONOMY_PATHS:
#        if path.exists():
#            with open(path, "r", encoding="utf-8") as f:
#                return json.load(f)
#    raise FileNotFoundError(
#        "skills_taxonomy.json not found. "
#        "Run build_skills_taxonomy.py first.\n"
#        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
#    )
#
#
#def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
#    """
#    Build FlashText processor from taxonomy.
#    Each alias maps back to canonical name.
#    e.g. "reactjs" → "React", "k8s" → "Kubernetes"
#    """
#    kp = KeywordProcessor(case_sensitive=False)
#    for canonical, entry in taxonomy.items():
#        kp.add_keyword(canonical, canonical)
#        for alias in entry.get("aliases", []):
#            if alias.strip():
#                kp.add_keyword(alias, canonical)
#    return kp
#
#
#def _build_fuzzy_pool(taxonomy: dict) -> dict:
#    """
#    Builds the restricted lookup used for fuzzy matching: every alias
#    AND every canonical name (lowercased) mapped back to its canonical
#    name -- but excluding entries that are too risky for fuzzy matching.
#
#    BUG FIX: the word-count check must be applied to EACH STRING
#    INDIVIDUALLY (the canonical name, and separately, every alias) --
#    not just once for the canonical name, with aliases let through
#    automatically afterward. Confirmed by direct testing: "AWS" itself
#    is 1 word (passes), but its alias "amazon web services" is 3 words
#    and should be excluded on its own merits. The original version of
#    this fix checked only the canonical name's word count and then
#    added ALL of its aliases unconditionally -- so "amazon web
#    services" slipped through anyway, and the ordinary word "services"
#    scored 90% against it (the exact same risk pattern as "Education"
#    matching "Pearson Education PHStat2"). Each string -- canonical
#    name or alias -- now passes through the same word-count and
#    length checks independently.
#    """
#    pool = {}
#    for canonical, entry in taxonomy.items():
#        canonical_lower = canonical.lower()
#        if (len(canonical_lower.split()) <= FUZZY_MAX_CANDIDATE_WORDS
#                and len(canonical_lower) >= FUZZY_MIN_CANDIDATE_LEN):
#            pool[canonical_lower] = canonical
#
#        for alias in entry.get("aliases", []):
#            alias = alias.strip()
#            if (len(alias.split()) <= FUZZY_MAX_CANDIDATE_WORDS
#                    and len(alias) >= FUZZY_MIN_CANDIDATE_LEN):
#                pool[alias] = canonical
#    return pool
#
#
## ── Load once at import ──
#_TAXONOMY    = _load_taxonomy()
#_KP          = _build_keyword_processor(_TAXONOMY)
#_FUZZY_POOL  = _build_fuzzy_pool(_TAXONOMY)
#_FUZZY_SEARCH_STRINGS = list(_FUZZY_POOL.keys())
#
#print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills "
#      f"({len(_FUZZY_SEARCH_STRINGS):,} fuzzy-searchable names+aliases).")
#
#
## ══════════════════════════════════════════════════════════════
## HELPER FUNCTIONS
## ══════════════════════════════════════════════════════════════
#
#def _category_of(skill_name: str) -> str:
#    entry = _TAXONOMY.get(skill_name)
#    if entry:
#        return entry.get("category", "other")
#    return "other"
#
#
#def _should_skip_fuzzy(token: str) -> bool:
#    """
#    Returns True if this token should be skipped for fuzzy matching
#    entirely, before even attempting it.
#
#    This is now a much smaller safety net than before -- the primary
#    defense against false positives is the restricted candidate pool
#    (FUZZY_MAX_CANDIDATE_WORDS / FUZZY_MIN_CANDIDATE_LEN, applied in
#    _build_fuzzy_pool) plus the length-ratio check applied after
#    matching in _best_fuzzy_match. This function only catches the case
#    where the token itself is an exact known generic word, as a final
#    explicit belt-and-suspenders layer.
#    """
#    lower = token.strip().lower()
#    if lower in FUZZY_SKIP_TOKENS:
#        return True
#    words = lower.split()
#    if len(words) > 1 and all(w in FUZZY_SKIP_TOKENS for w in words):
#        return True
#    return False
#
#
#def _best_fuzzy_match(token: str):
#    """
#    Finds the best fuzzy match for a token against the restricted
#    fuzzy pool, applying a length-ratio safeguard on top of the score
#    threshold. Returns (canonical_name, score) or None.
#
#    The length-ratio check rejects matches where the input token's
#    length is wildly different from the matched candidate's length --
#    confirmed necessary by direct testing: "Discovery" (9 chars) scored
#    90%+ against "BioDiscoveryImaGene" (a single compound word, so the
#    word-count filter alone couldn't exclude it) purely because
#    "Discovery" is a near-complete substring of the much longer name.
#    A real typo is always close in length to the correct word it's a
#    typo of (ratio near 1.0); a short common word accidentally
#    matching as a substring of an unrelated long name is not.
#    """
#    token_lower = token.lower()
#    # Short tokens (typos of short words like "Python", "Docker") need
#    # a slightly lower threshold — transposition errors score ~83-85%.
#    # Longer tokens are kept at the standard threshold.
#    threshold = 83 if len(token_lower) <= 9 else FUZZY_THRESHOLD
#    result = process.extractOne(
#        token_lower, _FUZZY_SEARCH_STRINGS, scorer=fuzz.WRatio, score_cutoff=threshold
#    )
#    if not result:
#        return None
#
#    matched_string, score, _ = result
#    ratio = len(token_lower) / len(matched_string)
#    if not (FUZZY_MIN_LEN_RATIO <= ratio <= FUZZY_MAX_LEN_RATIO):
#        return None
#
#    return _FUZZY_POOL[matched_string], round(score, 1)
#
#
#def _tokenize_for_fuzzy(text: str) -> list:
#    """
#    Split skills section into individual tokens for fuzzy matching.
#    Each token = one candidate skill name FlashText didn't match exactly.
#    """
#    raw_tokens = TOKEN_SPLIT_RE.split(text)
#    tokens = []
#    for t in raw_tokens:
#        t = t.strip()
#        if FUZZY_MIN_TOKEN <= len(t) <= FUZZY_MAX_TOKEN:
#            tokens.append(t)
#    return tokens
#
#
## ══════════════════════════════════════════════════════════════
## MAIN EXTRACTOR FUNCTION
## ══════════════════════════════════════════════════════════════
#
#def extract_skills(sections: dict, full_text: str = "") -> dict:
#    """
#    Extract skills from the skills section of a resume only.
#
#    Args:
#        sections  : dict from segmenter. Only "skills" key is used.
#        full_text : kept for API compatibility, not used.
#
#    Returns:
#        {
#            "skills":             sorted list of canonical skill names,
#            "skills_by_category": dict of category -> [skills],
#            "_confidence":        float 0.0-1.0,
#            "_method":            which steps fired,
#            "_timing_ms":         time taken
#        }
#    """
#    t0 = time.perf_counter()
#
#    found_skills  = set()
#    methods_used  = []
#
#    # ── PRIMARY SOURCE: skills section only ───────────────────
#    skills_text = sections.get("skills", "") or ""
#
#    # ── STEP 1: FlashText exact + alias match ─────────────────
#    # Scans the entire skills section in one pass.
#    # Catches: "reactjs"→"React", "k8s"→"Kubernetes", "postgres"→"PostgreSQL"
#    if skills_text.strip():
#        matches = _KP.extract_keywords(skills_text)
#        if matches:
#            found_skills.update(matches)
#            methods_used.append("flashtext_skills_section")
#
#    # ── STEP 2: RapidFuzz fuzzy match ─────────────────────────
#    # Tokenize skills section and fuzzy-match each token.
#    # Catches misspellings: "Postgress"→"PostgreSQL", "Kubernets"→"Kubernetes"
#    # Skips short/generic tokens to prevent false positives.
#    if skills_text.strip():
#        tokens     = _tokenize_for_fuzzy(skills_text)
#        fuzzy_hits = []
#
#        for token in tokens:
#            # Already found exactly by FlashText — skip
#            if token in found_skills:
#                continue
#
#            # Too short or too generic — skip fuzzy for this token
#            if _should_skip_fuzzy(token):
#                continue
#
#            result = _best_fuzzy_match(token)
#            if result:
#                canonical_match, score = result
#                found_skills.add(canonical_match)
#                fuzzy_hits.append(f"{token}→{canonical_match}({score:.0f}%)")
#
#        if fuzzy_hits:
#            preview = ", ".join(fuzzy_hits[:5])
#            suffix  = "..." if len(fuzzy_hits) > 5 else ""
#            methods_used.append(f"fuzzy({preview}{suffix})")
#
#    # ── BUILD OUTPUT ──────────────────────────────────────────
#    sorted_skills = sorted(found_skills)
#
#    skills_by_category = {}
#    for skill in sorted_skills:
#        cat = _category_of(skill)
#        skills_by_category.setdefault(cat, []).append(skill)
#
#    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
#        confidence = 0.92
#    elif len(sorted_skills) >= 1:
#        confidence = 0.75
#    else:
#        confidence = 0.0
#
#    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
#
#    return {
#        "skills":              sorted_skills,
#        "skills_by_category":  skills_by_category,
#        "_confidence":         confidence,
#        "_method":             " | ".join(methods_used) if methods_used else "none",
#        "_timing_ms":          elapsed_ms,
#    }


















#"""
#extractors/skills.py
#━━━━━━━━━━━━━━━━━━━
#Layer 2 — Skills Extractor
#
#Pipeline:
#  Step 1 → FlashText exact/alias match on skills section ONLY
#  Step 2 → RapidFuzz fuzzy match on skills section tokens (catches misspellings)
#
#Primary source: skills section text only.
#We do NOT scan experience/projects/full_text to avoid false positives.
#
#Input:  sections dict from segmenter  (only "skills" key is used)
#        full_text string (kept for API compatibility, not used)
#
#Output: {
#    "skills": ["Python", "AWS", "React", ...],
#    "skills_by_category": {
#        "programming_language": ["Python"],
#        "cloud_infra": ["AWS"],
#        ...
#    },
#    "_confidence": 0.92,
#    "_method": "flashtext_skills_section | fuzzy(...)",
#    "_timing_ms": 12.4
#}
#"""
#
#import json
#import re
#import time
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import process, fuzz
#
#
## ──────────────────────────────────────────────────────────────
## TAXONOMY PATH
## ──────────────────────────────────────────────────────────────
#_TAXONOMY_PATHS = [
#    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",
#    Path("data/skills_taxonomy.json"),
#]
#
## ──────────────────────────────────────────────────────────────
## FUZZY MATCH SETTINGS
## ──────────────────────────────────────────────────────────────
#FUZZY_THRESHOLD    = 88   # minimum similarity % to accept a match
#FUZZY_MIN_TOKEN    = 3    # ignore tokens shorter than this
#FUZZY_MAX_TOKEN    = 50   # ignore tokens longer than this (probably a sentence)
#
## Key fix: tokens shorter than this are too generic for fuzzy matching.
## e.g. "Education", "Audit", "Training", "Discovery" are all < 12 chars
## and cause false positives by fuzzy-matching long O*NET tool names.
## FlashText handles short exact tokens fine — fuzzy is only for
## multi-word or longer tokens like "Kubernets", "Postgress" etc.
#FUZZY_MIN_TOKEN_LENGTH_FOR_FUZZY = 12
#
## Common single words that should never be fuzzy matched alone
## even if they are >= 12 chars. Belt-and-suspenders protection.
#FUZZY_SKIP_TOKENS = {
#    # Tech domain generic words
#    "education", "training", "audit", "compliance", "discovery",
#    "management", "performance", "foundation", "integration",
#    "architecture", "orchestration", "observability", "governance",
#    "documentation", "communication", "certification", "automation",
#    # Business domain generic words
#    "monitoring", "planning", "analysis", "research", "reporting",
#    "presentation", "presentations", "negotiation", "development",
#    "relationship", "coordination", "administration", "execution",
#    "implementation", "optimization", "optimisation", "engagement",
#    "acquisition", "retention", "leadership", "collaboration",
#    "networking", "forecasting", "budgeting", "scheduling",
#}
#
## ──────────────────────────────────────────────────────────────
## TOKEN SPLIT PATTERN
## Split on: comma, pipe, bullet, newline, tab, slash, ampersand etc.
## ──────────────────────────────────────────────────────────────
#TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")
#
#
## ══════════════════════════════════════════════════════════════
## MODULE-LEVEL INITIALISATION  (runs once at import time)
## ══════════════════════════════════════════════════════════════
#
#def _load_taxonomy() -> dict:
#    for path in _TAXONOMY_PATHS:
#        if path.exists():
#            with open(path, "r", encoding="utf-8") as f:
#                return json.load(f)
#    raise FileNotFoundError(
#        "skills_taxonomy.json not found. "
#        "Run build_skills_taxonomy.py first.\n"
#        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
#    )
#
#
#def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
#    """
#    Build FlashText processor from taxonomy.
#    Each alias maps back to canonical name.
#    e.g. "reactjs" → "React", "k8s" → "Kubernetes"
#    """
#    kp = KeywordProcessor(case_sensitive=False)
#    for canonical, entry in taxonomy.items():
#        kp.add_keyword(canonical, canonical)
#        for alias in entry.get("aliases", []):
#            if alias.strip():
#                kp.add_keyword(alias, canonical)
#    return kp
#
#
#def _build_canonical_list(taxonomy: dict) -> list:
#    return list(taxonomy.keys())
#
#
## ── Load once at import ──
#_TAXONOMY       = _load_taxonomy()
#_KP             = _build_keyword_processor(_TAXONOMY)
#_CANONICAL_LIST = _build_canonical_list(_TAXONOMY)
#
#print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills from taxonomy.")
#
#
## ══════════════════════════════════════════════════════════════
## HELPER FUNCTIONS
## ══════════════════════════════════════════════════════════════
#
#def _category_of(skill_name: str) -> str:
#    entry = _TAXONOMY.get(skill_name)
#    if entry:
#        return entry.get("category", "other")
#    return "other"
#
#
#def _should_skip_fuzzy(token: str) -> bool:
#    """
#    Returns True if this token should be skipped for fuzzy matching.
#
#    Three rules:
#    1. Token is too short (< 12 chars) — short tokens like "Audit",
#       "Training" fuzzy-match long O*NET tool names because
#       RapidFuzz WRatio rewards substring matches.
#
#    2. Token exactly matches a known generic single word.
#
#    3. Token is a multi-word phrase made entirely of generic words.
#       e.g. "Performance Monitoring" → both words are in skip list
#            → skip it, otherwise it matches "Omnitracs Performance Monitoring"
#    """
#    stripped = token.strip()
#    lower    = stripped.lower()
#
#    # Rule 1 — too short
#    if len(stripped) < FUZZY_MIN_TOKEN_LENGTH_FOR_FUZZY:
#        return True
#
#    # Rule 2 — single generic word
#    if lower in FUZZY_SKIP_TOKENS:
#        return True
#
#    # Rule 3 — multi-word phrase where ALL words are generic
#    words = lower.split()
#    if len(words) > 1 and all(w in FUZZY_SKIP_TOKENS for w in words):
#        return True
#
#    return False
#
#
#def _tokenize_for_fuzzy(text: str) -> list:
#    """
#    Split skills section into individual tokens for fuzzy matching.
#    Each token = one candidate skill name FlashText didn't match exactly.
#    """
#    raw_tokens = TOKEN_SPLIT_RE.split(text)
#    tokens = []
#    for t in raw_tokens:
#        t = t.strip()
#        if FUZZY_MIN_TOKEN <= len(t) <= FUZZY_MAX_TOKEN:
#            tokens.append(t)
#    return tokens
#
#
## ══════════════════════════════════════════════════════════════
## MAIN EXTRACTOR FUNCTION
## ══════════════════════════════════════════════════════════════
#
#def extract_skills(sections: dict, full_text: str = "") -> dict:
#    """
#    Extract skills from the skills section of a resume only.
#
#    Args:
#        sections  : dict from segmenter. Only "skills" key is used.
#        full_text : kept for API compatibility, not used.
#
#    Returns:
#        {
#            "skills":             sorted list of canonical skill names,
#            "skills_by_category": dict of category -> [skills],
#            "_confidence":        float 0.0-1.0,
#            "_method":            which steps fired,
#            "_timing_ms":         time taken
#        }
#    """
#    t0 = time.perf_counter()
#
#    found_skills  = set()
#    methods_used  = []
#
#    # ── PRIMARY SOURCE: skills section only ───────────────────
#    skills_text = sections.get("skills", "") or ""
#
#    # ── STEP 1: FlashText exact + alias match ─────────────────
#    # Scans the entire skills section in one pass.
#    # Catches: "reactjs"→"React", "k8s"→"Kubernetes", "postgres"→"PostgreSQL"
#    if skills_text.strip():
#        matches = _KP.extract_keywords(skills_text)
#        if matches:
#            found_skills.update(matches)
#            methods_used.append("flashtext_skills_section")
#
#    # ── STEP 2: RapidFuzz fuzzy match ─────────────────────────
#    # Tokenize skills section and fuzzy-match each token.
#    # Catches misspellings: "Postgress"→"PostgreSQL", "Kubernets"→"Kubernetes"
#    # Skips short/generic tokens to prevent false positives.
#    if skills_text.strip():
#        tokens     = _tokenize_for_fuzzy(skills_text)
#        fuzzy_hits = []
#
#        for token in tokens:
#            # Already found exactly by FlashText — skip
#            if token in found_skills:
#                continue
#
#            # Too short or too generic — skip fuzzy for this token
#            if _should_skip_fuzzy(token):
#                continue
#
#            result = process.extractOne(
#                token,
#                _CANONICAL_LIST,
#                scorer=fuzz.WRatio,
#                score_cutoff=FUZZY_THRESHOLD,
#            )
#            if result:
#                canonical_match, score, _ = result
#                found_skills.add(canonical_match)
#                fuzzy_hits.append(f"{token}→{canonical_match}({score:.0f}%)")
#
#        if fuzzy_hits:
#            preview = ", ".join(fuzzy_hits[:5])
#            suffix  = "..." if len(fuzzy_hits) > 5 else ""
#            methods_used.append(f"fuzzy({preview}{suffix})")
#
#    # ── BUILD OUTPUT ──────────────────────────────────────────
#    sorted_skills = sorted(found_skills)
#
#    skills_by_category = {}
#    for skill in sorted_skills:
#        cat = _category_of(skill)
#        skills_by_category.setdefault(cat, []).append(skill)
#
#    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
#        confidence = 0.92
#    elif len(sorted_skills) >= 1:
#        confidence = 0.75
#    else:
#        confidence = 0.0
#
#    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
#
#    return {
#        "skills":              sorted_skills,
#        "skills_by_category":  skills_by_category,
#        "_confidence":         confidence,
#        "_method":             " | ".join(methods_used) if methods_used else "none",
#        "_timing_ms":          elapsed_ms,
#    }


















#"""
#extractors/skills.py
#━━━━━━━━━━━━━━━━━━━
#Layer 2 — Skills Extractor
#
#Pipeline:
#  Step 1 → FlashText exact/alias match on skills section ONLY
#  Step 2 → RapidFuzz fuzzy match on skills section tokens (catches misspellings)
#
#Primary source: skills section text only.
#We do NOT scan experience/projects/full_text to avoid false positives.
#
#Input:  sections dict from segmenter  (only "skills" key is used)
#        full_text string (kept for API compatibility, not used)
#
#Output: {
#    "skills": ["Python", "AWS", "React", ...],
#    "skills_by_category": {
#        "programming_language": ["Python"],
#        "cloud_infra": ["AWS"],
#        ...
#    },
#    "_confidence": 0.92,
#    "_method": "flashtext_skills_section | fuzzy(...)",
#    "_timing_ms": 12.4
#}
#"""
#
#import json
#import re
#import time
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import process, fuzz
#
#
## ──────────────────────────────────────────────────────────────
## TAXONOMY PATH
## ──────────────────────────────────────────────────────────────
#_TAXONOMY_PATHS = [
#    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",
#    Path("data/skills_taxonomy.json"),
#]
#
## ──────────────────────────────────────────────────────────────
## FUZZY MATCH SETTINGS
## ──────────────────────────────────────────────────────────────
#FUZZY_THRESHOLD    = 88   # minimum similarity % to accept a match
#FUZZY_MIN_TOKEN    = 3    # ignore tokens shorter than this
#FUZZY_MAX_TOKEN    = 50   # ignore tokens longer than this (probably a sentence)
#
## Key fix: tokens shorter than this are too generic for fuzzy matching.
## e.g. "Education", "Audit", "Training", "Discovery" are all < 12 chars
## and cause false positives by fuzzy-matching long O*NET tool names.
## FlashText handles short exact tokens fine — fuzzy is only for
## multi-word or longer tokens like "Kubernets", "Postgress" etc.
#FUZZY_MIN_TOKEN_LENGTH_FOR_FUZZY = 12
#
## Common single words that should never be fuzzy matched alone
## even if they are >= 12 chars. Belt-and-suspenders protection.
#FUZZY_SKIP_TOKENS = {
#    # Tech domain generic words
#    "education", "training", "audit", "compliance", "discovery",
#    "management", "performance", "foundation", "integration",
#    "architecture", "orchestration", "observability", "governance",
#    "documentation", "communication", "certification", "automation",
#    # Business domain generic words
#    "monitoring", "planning", "analysis", "research", "reporting",
#    "presentation", "presentations", "negotiation", "development",
#    "relationship", "coordination", "administration", "execution",
#    "implementation", "optimization", "optimisation", "engagement",
#    "acquisition", "retention", "leadership", "collaboration",
#    "networking", "forecasting", "budgeting", "scheduling",
#}
#
## ──────────────────────────────────────────────────────────────
## TOKEN SPLIT PATTERN
## Split on: comma, pipe, bullet, newline, tab, slash, ampersand etc.
## ──────────────────────────────────────────────────────────────
#TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")
#
#
## ══════════════════════════════════════════════════════════════
## MODULE-LEVEL INITIALISATION  (runs once at import time)
## ══════════════════════════════════════════════════════════════
#
#def _load_taxonomy() -> dict:
#    for path in _TAXONOMY_PATHS:
#        if path.exists():
#            with open(path, "r", encoding="utf-8") as f:
#                return json.load(f)
#    raise FileNotFoundError(
#        "skills_taxonomy.json not found. "
#        "Run build_skills_taxonomy.py first.\n"
#        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
#    )
#
#
#def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
#    """
#    Build FlashText processor from taxonomy.
#    Each alias maps back to canonical name.
#    e.g. "reactjs" → "React", "k8s" → "Kubernetes"
#    """
#    kp = KeywordProcessor(case_sensitive=False)
#    for canonical, entry in taxonomy.items():
#        kp.add_keyword(canonical, canonical)
#        for alias in entry.get("aliases", []):
#            if alias.strip():
#                kp.add_keyword(alias, canonical)
#    return kp
#
#
#def _build_canonical_list(taxonomy: dict) -> list:
#    return list(taxonomy.keys())
#
#
## ── Load once at import ──
#_TAXONOMY       = _load_taxonomy()
#_KP             = _build_keyword_processor(_TAXONOMY)
#_CANONICAL_LIST = _build_canonical_list(_TAXONOMY)
#
#print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills from taxonomy.")
#
#
## ══════════════════════════════════════════════════════════════
## HELPER FUNCTIONS
## ══════════════════════════════════════════════════════════════
#
#def _category_of(skill_name: str) -> str:
#    entry = _TAXONOMY.get(skill_name)
#    if entry:
#        return entry.get("category", "other")
#    return "other"
#
#
#def _should_skip_fuzzy(token: str) -> bool:
#    """
#    Returns True if this token should be skipped for fuzzy matching.
#
#    Three rules:
#    1. Token is too short (< 12 chars) — short tokens like "Audit",
#       "Training" fuzzy-match long O*NET tool names because
#       RapidFuzz WRatio rewards substring matches.
#
#    2. Token exactly matches a known generic single word.
#
#    3. Token is a multi-word phrase made entirely of generic words.
#       e.g. "Performance Monitoring" → both words are in skip list
#            → skip it, otherwise it matches "Omnitracs Performance Monitoring"
#    """
#    stripped = token.strip()
#    lower    = stripped.lower()
#
#    # Rule 1 — too short
#    if len(stripped) < FUZZY_MIN_TOKEN_LENGTH_FOR_FUZZY:
#        return True
#
#    # Rule 2 — single generic word
#    if lower in FUZZY_SKIP_TOKENS:
#        return True
#
#    # Rule 3 — multi-word phrase where ALL words are generic
#    words = lower.split()
#    if len(words) > 1 and all(w in FUZZY_SKIP_TOKENS for w in words):
#        return True
#
#    return False
#
#
#def _tokenize_for_fuzzy(text: str) -> list:
#    """
#    Split skills section into individual tokens for fuzzy matching.
#    Each token = one candidate skill name FlashText didn't match exactly.
#    """
#    raw_tokens = TOKEN_SPLIT_RE.split(text)
#    tokens = []
#    for t in raw_tokens:
#        t = t.strip()
#        if FUZZY_MIN_TOKEN <= len(t) <= FUZZY_MAX_TOKEN:
#            tokens.append(t)
#    return tokens
#
#
## ══════════════════════════════════════════════════════════════
## MAIN EXTRACTOR FUNCTION
## ══════════════════════════════════════════════════════════════
#
#def extract_skills(sections: dict, full_text: str = "") -> dict:
#    """
#    Extract skills from the skills section of a resume only.
#
#    Args:
#        sections  : dict from segmenter. Only "skills" key is used.
#        full_text : kept for API compatibility, not used.
#
#    Returns:
#        {
#            "skills":             sorted list of canonical skill names,
#            "skills_by_category": dict of category -> [skills],
#            "_confidence":        float 0.0-1.0,
#            "_method":            which steps fired,
#            "_timing_ms":         time taken
#        }
#    """
#    t0 = time.perf_counter()
#
#    found_skills  = set()
#    methods_used  = []
#
#    # ── PRIMARY SOURCE: skills section only ───────────────────
#    skills_text = sections.get("skills", "") or ""
#
#    # ── STEP 1: FlashText exact + alias match ─────────────────
#    # Scans the entire skills section in one pass.
#    # Catches: "reactjs"→"React", "k8s"→"Kubernetes", "postgres"→"PostgreSQL"
#    if skills_text.strip():
#        matches = _KP.extract_keywords(skills_text)
#        if matches:
#            found_skills.update(matches)
#            methods_used.append("flashtext_skills_section")
#
#    # ── STEP 2: RapidFuzz fuzzy match ─────────────────────────
#    # Tokenize skills section and fuzzy-match each token.
#    # Catches misspellings: "Postgress"→"PostgreSQL", "Kubernets"→"Kubernetes"
#    # Skips short/generic tokens to prevent false positives.
#    if skills_text.strip():
#        tokens     = _tokenize_for_fuzzy(skills_text)
#        fuzzy_hits = []
#
#        for token in tokens:
#            # Already found exactly by FlashText — skip
#            if token in found_skills:
#                continue
#
#            # Too short or too generic — skip fuzzy for this token
#            if _should_skip_fuzzy(token):
#                continue
#
#            result = process.extractOne(
#                token,
#                _CANONICAL_LIST,
#                scorer=fuzz.WRatio,
#                score_cutoff=FUZZY_THRESHOLD,
#            )
#            if result:
#                canonical_match, score, _ = result
#                found_skills.add(canonical_match)
#                fuzzy_hits.append(f"{token}→{canonical_match}({score:.0f}%)")
#
#        if fuzzy_hits:
#            preview = ", ".join(fuzzy_hits[:5])
#            suffix  = "..." if len(fuzzy_hits) > 5 else ""
#            methods_used.append(f"fuzzy({preview}{suffix})")
#
#    # ── BUILD OUTPUT ──────────────────────────────────────────
#    sorted_skills = sorted(found_skills)
#
#    skills_by_category = {}
#    for skill in sorted_skills:
#        cat = _category_of(skill)
#        skills_by_category.setdefault(cat, []).append(skill)
#
#    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
#        confidence = 0.92
#    elif len(sorted_skills) >= 1:
#        confidence = 0.75
#    else:
#        confidence = 0.0
#
#    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
#
#    return {
#        "skills":              sorted_skills,
#        "skills_by_category":  skills_by_category,
#        "_confidence":         confidence,
#        "_method":             " | ".join(methods_used) if methods_used else "none",
#        "_timing_ms":          elapsed_ms,
#    }























#"""
#extractors/skills.py
#━━━━━━━━━━━━━━━━━━━
#Layer 2 — Skills Extractor
#
#Pipeline:
#  Step 1 → FlashText exact/alias match on skills section ONLY
#  Step 2 → RapidFuzz fuzzy match on skills section tokens (catches misspellings)
#
#Primary source: skills section text only.
#We do NOT scan experience/projects/full_text to avoid false positives.
#
#Input:  sections dict from segmenter  (only "skills" key is used)
#        full_text string (kept for API compatibility, not used)
#
#Output: {
#    "skills": ["Python", "AWS", "React", ...],
#    "skills_by_category": {
#        "programming_language": ["Python"],
#        "cloud_infra": ["AWS"],
#        ...
#    },
#    "_confidence": 0.92,
#    "_method": "flashtext_skills_section | fuzzy(...)",
#    "_timing_ms": 12.4
#}
#"""
#
#import json
#import re
#import time
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import process, fuzz
#
#
## ──────────────────────────────────────────────────────────────
## TAXONOMY PATH
## ──────────────────────────────────────────────────────────────
#_TAXONOMY_PATHS = [
#    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",
#    Path("data/skills_taxonomy.json"),
#]
#
## ──────────────────────────────────────────────────────────────
## FUZZY MATCH SETTINGS
## ──────────────────────────────────────────────────────────────
#FUZZY_THRESHOLD    = 88   # minimum similarity % to accept a match
#FUZZY_MIN_TOKEN    = 3    # ignore tokens shorter than this
#FUZZY_MAX_TOKEN    = 50   # ignore tokens longer than this (probably a sentence)
#
## Key fix: tokens shorter than this are too generic for fuzzy matching.
## e.g. "Education", "Audit", "Training", "Discovery" are all < 12 chars
## and cause false positives by fuzzy-matching long O*NET tool names.
## FlashText handles short exact tokens fine — fuzzy is only for
## multi-word or longer tokens like "Kubernets", "Postgress" etc.
#FUZZY_MIN_TOKEN_LENGTH_FOR_FUZZY = 12
#
## Common single words that should never be fuzzy matched alone
## even if they are >= 12 chars. Belt-and-suspenders protection.
#FUZZY_SKIP_TOKENS = {
#    "education", "training", "audit", "compliance", "discovery",
#    "management", "performance", "foundation", "integration",
#    "architecture", "orchestration", "observability", "governance",
#    "documentation", "communication", "certification", "automation",
#}
#
## ──────────────────────────────────────────────────────────────
## TOKEN SPLIT PATTERN
## Split on: comma, pipe, bullet, newline, tab, slash, ampersand etc.
## ──────────────────────────────────────────────────────────────
#TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")
#
#
## ══════════════════════════════════════════════════════════════
## MODULE-LEVEL INITIALISATION  (runs once at import time)
## ══════════════════════════════════════════════════════════════
#
#def _load_taxonomy() -> dict:
#    for path in _TAXONOMY_PATHS:
#        if path.exists():
#            with open(path, "r", encoding="utf-8") as f:
#                return json.load(f)
#    raise FileNotFoundError(
#        "skills_taxonomy.json not found. "
#        "Run build_skills_taxonomy.py first.\n"
#        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
#    )
#
#
#def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
#    """
#    Build FlashText processor from taxonomy.
#    Each alias maps back to canonical name.
#    e.g. "reactjs" → "React", "k8s" → "Kubernetes"
#    """
#    kp = KeywordProcessor(case_sensitive=False)
#    for canonical, entry in taxonomy.items():
#        kp.add_keyword(canonical, canonical)
#        for alias in entry.get("aliases", []):
#            if alias.strip():
#                kp.add_keyword(alias, canonical)
#    return kp
#
#
#def _build_canonical_list(taxonomy: dict) -> list:
#    return list(taxonomy.keys())
#
#
## ── Load once at import ──
#_TAXONOMY       = _load_taxonomy()
#_KP             = _build_keyword_processor(_TAXONOMY)
#_CANONICAL_LIST = _build_canonical_list(_TAXONOMY)
#
#print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills from taxonomy.")
#
#
## ══════════════════════════════════════════════════════════════
## HELPER FUNCTIONS
## ══════════════════════════════════════════════════════════════
#
#def _category_of(skill_name: str) -> str:
#    entry = _TAXONOMY.get(skill_name)
#    if entry:
#        return entry.get("category", "other")
#    return "other"
#
#
#def _should_skip_fuzzy(token: str) -> bool:
#    """
#    Returns True if this token should be skipped for fuzzy matching.
#
#    Two rules:
#    1. Token is too short — short tokens like "Audit", "Training",
#       "Discovery" fuzzy-match long O*NET tool names because
#       RapidFuzz WRatio rewards substring matches.
#       e.g. "Audit" (5 chars) → "Elite Software Energy Audit" (90%)
#
#    2. Token is a known generic word — even if >= 12 chars,
#       words like "orchestration", "observability" are domain words
#       that appear in O*NET tool names and cause false positives.
#    """
#    lower = token.strip().lower()
#    if len(token.strip()) < FUZZY_MIN_TOKEN_LENGTH_FOR_FUZZY:
#        return True
#    if lower in FUZZY_SKIP_TOKENS:
#        return True
#    return False
#
#
#def _tokenize_for_fuzzy(text: str) -> list:
#    """
#    Split skills section into individual tokens for fuzzy matching.
#    Each token = one candidate skill name FlashText didn't match exactly.
#    """
#    raw_tokens = TOKEN_SPLIT_RE.split(text)
#    tokens = []
#    for t in raw_tokens:
#        t = t.strip()
#        if FUZZY_MIN_TOKEN <= len(t) <= FUZZY_MAX_TOKEN:
#            tokens.append(t)
#    return tokens
#
#
## ══════════════════════════════════════════════════════════════
## MAIN EXTRACTOR FUNCTION
## ══════════════════════════════════════════════════════════════
#
#def extract_skills(sections: dict, full_text: str = "") -> dict:
#    """
#    Extract skills from the skills section of a resume only.
#
#    Args:
#        sections  : dict from segmenter. Only "skills" key is used.
#        full_text : kept for API compatibility, not used.
#
#    Returns:
#        {
#            "skills":             sorted list of canonical skill names,
#            "skills_by_category": dict of category -> [skills],
#            "_confidence":        float 0.0-1.0,
#            "_method":            which steps fired,
#            "_timing_ms":         time taken
#        }
#    """
#    t0 = time.perf_counter()
#
#    found_skills  = set()
#    methods_used  = []
#
#    # ── PRIMARY SOURCE: skills section only ───────────────────
#    skills_text = sections.get("skills", "") or ""
#
#    # ── STEP 1: FlashText exact + alias match ─────────────────
#    # Scans the entire skills section in one pass.
#    # Catches: "reactjs"→"React", "k8s"→"Kubernetes", "postgres"→"PostgreSQL"
#    if skills_text.strip():
#        matches = _KP.extract_keywords(skills_text)
#        if matches:
#            found_skills.update(matches)
#            methods_used.append("flashtext_skills_section")
#
#    # ── STEP 2: RapidFuzz fuzzy match ─────────────────────────
#    # Tokenize skills section and fuzzy-match each token.
#    # Catches misspellings: "Postgress"→"PostgreSQL", "Kubernets"→"Kubernetes"
#    # Skips short/generic tokens to prevent false positives.
#    if skills_text.strip():
#        tokens     = _tokenize_for_fuzzy(skills_text)
#        fuzzy_hits = []
#
#        for token in tokens:
#            # Already found exactly by FlashText — skip
#            if token in found_skills:
#                continue
#
#            # Too short or too generic — skip fuzzy for this token
#            if _should_skip_fuzzy(token):
#                continue
#
#            result = process.extractOne(
#                token,
#                _CANONICAL_LIST,
#                scorer=fuzz.WRatio,
#                score_cutoff=FUZZY_THRESHOLD,
#            )
#            if result:
#                canonical_match, score, _ = result
#                found_skills.add(canonical_match)
#                fuzzy_hits.append(f"{token}→{canonical_match}({score:.0f}%)")
#
#        if fuzzy_hits:
#            preview = ", ".join(fuzzy_hits[:5])
#            suffix  = "..." if len(fuzzy_hits) > 5 else ""
#            methods_used.append(f"fuzzy({preview}{suffix})")
#
#    # ── BUILD OUTPUT ──────────────────────────────────────────
#    sorted_skills = sorted(found_skills)
#
#    skills_by_category = {}
#    for skill in sorted_skills:
#        cat = _category_of(skill)
#        skills_by_category.setdefault(cat, []).append(skill)
#
#    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
#        confidence = 0.92
#    elif len(sorted_skills) >= 1:
#        confidence = 0.75
#    else:
#        confidence = 0.0
#
#    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
#
#    return {
#        "skills":              sorted_skills,
#        "skills_by_category":  skills_by_category,
#        "_confidence":         confidence,
#        "_method":             " | ".join(methods_used) if methods_used else "none",
#        "_timing_ms":          elapsed_ms,
#    }























#"""
#extractors/skills.py
#━━━━━━━━━━━━━━━━━━━
#Layer 2 — Skills Extractor
#
#Pipeline:
#  Step 1 → FlashText exact/alias match on skills section ONLY
#  Step 2 → RapidFuzz fuzzy match on skills section tokens (catches misspellings)
#
#Primary source: skills section text only.
#We do NOT scan experience/projects/full_text to avoid false positives.
#
#Input:  sections dict from segmenter  (only "skills" key is used)
#        full_text string (kept for API compatibility, not used)
#
#Output: {
#    "skills": ["Python", "AWS", "React", ...],
#    "skills_by_category": {
#        "programming_language": ["Python"],
#        "cloud_infra": ["AWS"],
#        ...
#    },
#    "_confidence": 0.92,
#    "_method": "flashtext_skills_section | fuzzy(...)",
#    "_timing_ms": 12.4
#}
#"""
#
#import json
#import re
#import time
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import process, fuzz
#
#
## ──────────────────────────────────────────────────────────────
## TAXONOMY PATH
## ──────────────────────────────────────────────────────────────
#_TAXONOMY_PATHS = [
#    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",
#    Path("data/skills_taxonomy.json"),
#]
#
## ──────────────────────────────────────────────────────────────
## FUZZY MATCH SETTINGS
## ──────────────────────────────────────────────────────────────
#FUZZY_THRESHOLD    = 88   # minimum similarity % to accept a match
#FUZZY_MIN_TOKEN    = 3    # ignore tokens shorter than this
#FUZZY_MAX_TOKEN    = 50   # ignore tokens longer than this (probably a sentence)
#
## Key fix: tokens shorter than this are too generic for fuzzy matching.
## e.g. "Education", "Audit", "Training", "Discovery" are all < 12 chars
## and cause false positives by fuzzy-matching long O*NET tool names.
## FlashText handles short exact tokens fine — fuzzy is only for
## multi-word or longer tokens like "Kubernets", "Postgress" etc.
#FUZZY_MIN_TOKEN_LENGTH_FOR_FUZZY = 12
#
## Common single words that should never be fuzzy matched alone
## even if they are >= 12 chars. Belt-and-suspenders protection.
#FUZZY_SKIP_TOKENS = {
#    "education", "training", "audit", "compliance", "discovery",
#    "management", "performance", "foundation", "integration",
#    "architecture", "orchestration", "observability", "governance",
#    "documentation", "communication", "certification", "automation",
#}
#
## ──────────────────────────────────────────────────────────────
## TOKEN SPLIT PATTERN
## Split on: comma, pipe, bullet, newline, tab, slash, ampersand etc.
## ──────────────────────────────────────────────────────────────
#TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")
#
#
## ══════════════════════════════════════════════════════════════
## MODULE-LEVEL INITIALISATION  (runs once at import time)
## ══════════════════════════════════════════════════════════════
#
#def _load_taxonomy() -> dict:
#    for path in _TAXONOMY_PATHS:
#        if path.exists():
#            with open(path, "r", encoding="utf-8") as f:
#                return json.load(f)
#    raise FileNotFoundError(
#        "skills_taxonomy.json not found. "
#        "Run build_skills_taxonomy.py first.\n"
#        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
#    )
#
#
#def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
#    """
#    Build FlashText processor from taxonomy.
#    Each alias maps back to canonical name.
#    e.g. "reactjs" → "React", "k8s" → "Kubernetes"
#    """
#    kp = KeywordProcessor(case_sensitive=False)
#    for canonical, entry in taxonomy.items():
#        kp.add_keyword(canonical, canonical)
#        for alias in entry.get("aliases", []):
#            if alias.strip():
#                kp.add_keyword(alias, canonical)
#    return kp
#
#
#def _build_canonical_list(taxonomy: dict) -> list:
#    return list(taxonomy.keys())
#
#
## ── Load once at import ──
#_TAXONOMY       = _load_taxonomy()
#_KP             = _build_keyword_processor(_TAXONOMY)
#_CANONICAL_LIST = _build_canonical_list(_TAXONOMY)
#
#print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills from taxonomy.")
#
#
## ══════════════════════════════════════════════════════════════
## HELPER FUNCTIONS
## ══════════════════════════════════════════════════════════════
#
#def _category_of(skill_name: str) -> str:
#    entry = _TAXONOMY.get(skill_name)
#    if entry:
#        return entry.get("category", "other")
#    return "other"
#
#
#def _should_skip_fuzzy(token: str) -> bool:
#    """
#    Returns True if this token should be skipped for fuzzy matching.
#
#    Two rules:
#    1. Token is too short — short tokens like "Audit", "Training",
#       "Discovery" fuzzy-match long O*NET tool names because
#       RapidFuzz WRatio rewards substring matches.
#       e.g. "Audit" (5 chars) → "Elite Software Energy Audit" (90%)
#
#    2. Token is a known generic word — even if >= 12 chars,
#       words like "orchestration", "observability" are domain words
#       that appear in O*NET tool names and cause false positives.
#    """
#    lower = token.strip().lower()
#    if len(token.strip()) < FUZZY_MIN_TOKEN_LENGTH_FOR_FUZZY:
#        return True
#    if lower in FUZZY_SKIP_TOKENS:
#        return True
#    return False
#
#
#def _tokenize_for_fuzzy(text: str) -> list:
#    """
#    Split skills section into individual tokens for fuzzy matching.
#    Each token = one candidate skill name FlashText didn't match exactly.
#    """
#    raw_tokens = TOKEN_SPLIT_RE.split(text)
#    tokens = []
#    for t in raw_tokens:
#        t = t.strip()
#        if FUZZY_MIN_TOKEN <= len(t) <= FUZZY_MAX_TOKEN:
#            tokens.append(t)
#    return tokens
#
#
## ══════════════════════════════════════════════════════════════
## MAIN EXTRACTOR FUNCTION
## ══════════════════════════════════════════════════════════════
#
#def extract_skills(sections: dict, full_text: str = "") -> dict:
#    """
#    Extract skills from the skills section of a resume only.
#
#    Args:
#        sections  : dict from segmenter. Only "skills" key is used.
#        full_text : kept for API compatibility, not used.
#
#    Returns:
#        {
#            "skills":             sorted list of canonical skill names,
#            "skills_by_category": dict of category -> [skills],
#            "_confidence":        float 0.0-1.0,
#            "_method":            which steps fired,
#            "_timing_ms":         time taken
#        }
#    """
#    t0 = time.perf_counter()
#
#    found_skills  = set()
#    methods_used  = []
#
#    # ── PRIMARY SOURCE: skills section only ───────────────────
#    skills_text = sections.get("skills", "") or ""
#
#    # ── STEP 1: FlashText exact + alias match ─────────────────
#    # Scans the entire skills section in one pass.
#    # Catches: "reactjs"→"React", "k8s"→"Kubernetes", "postgres"→"PostgreSQL"
#    if skills_text.strip():
#        matches = _KP.extract_keywords(skills_text)
#        if matches:
#            found_skills.update(matches)
#            methods_used.append("flashtext_skills_section")
#
#    # ── STEP 2: RapidFuzz fuzzy match ─────────────────────────
#    # Tokenize skills section and fuzzy-match each token.
#    # Catches misspellings: "Postgress"→"PostgreSQL", "Kubernets"→"Kubernetes"
#    # Skips short/generic tokens to prevent false positives.
#    if skills_text.strip():
#        tokens     = _tokenize_for_fuzzy(skills_text)
#        fuzzy_hits = []
#
#        for token in tokens:
#            # Already found exactly by FlashText — skip
#            if token in found_skills:
#                continue
#
#            # Too short or too generic — skip fuzzy for this token
#            if _should_skip_fuzzy(token):
#                continue
#
#            result = process.extractOne(
#                token,
#                _CANONICAL_LIST,
#                scorer=fuzz.WRatio,
#                score_cutoff=FUZZY_THRESHOLD,
#            )
#            if result:
#                canonical_match, score, _ = result
#                found_skills.add(canonical_match)
#                fuzzy_hits.append(f"{token}→{canonical_match}({score:.0f}%)")
#
#        if fuzzy_hits:
#            preview = ", ".join(fuzzy_hits[:5])
#            suffix  = "..." if len(fuzzy_hits) > 5 else ""
#            methods_used.append(f"fuzzy({preview}{suffix})")
#
#    # ── BUILD OUTPUT ──────────────────────────────────────────
#    sorted_skills = sorted(found_skills)
#
#    skills_by_category = {}
#    for skill in sorted_skills:
#        cat = _category_of(skill)
#        skills_by_category.setdefault(cat, []).append(skill)
#
#    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
#        confidence = 0.92
#    elif len(sorted_skills) >= 1:
#        confidence = 0.75
#    else:
#        confidence = 0.0
#
#    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
#
#    return {
#        "skills":              sorted_skills,
#        "skills_by_category":  skills_by_category,
#        "_confidence":         confidence,
#        "_method":             " | ".join(methods_used) if methods_used else "none",
#        "_timing_ms":          elapsed_ms,
#    }























#"""
#extractors/skills.py
#━━━━━━━━━━━━━━━━━━━
#Layer 2 — Skills Extractor
#
#Pipeline:
#  Step 1 → FlashText exact/alias match on skills section ONLY
#  Step 2 → RapidFuzz fuzzy match on skills section tokens (catches misspellings)
#
#Primary source: skills section text only.
#We do NOT scan experience/projects/full_text — broad scanning causes
#false positives from O*NET entries accidentally matching resume sentences.
#
#Input:  sections dict from segmenter  (only "skills" key is used)
#        full_text string (kept for API compatibility, not used)
#
#Output: {
#    "skills": ["Python", "AWS", "React", ...],
#    "skills_by_category": {
#        "programming_language": ["Python"],
#        "cloud_infra": ["AWS"],
#        ...
#    },
#    "_confidence": 0.92,
#    "_method": "flashtext_skills_section | fuzzy(...)",
#    "_timing_ms": 12.4
#}
#"""
#
#import json
#import re
#import time
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import process, fuzz
#
#
## ──────────────────────────────────────────────────────────────
## TAXONOMY PATH
## ──────────────────────────────────────────────────────────────
#_TAXONOMY_PATHS = [
#    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",
#    Path("data/skills_taxonomy.json"),
#]
#
## ──────────────────────────────────────────────────────────────
## FUZZY MATCH SETTINGS
## ──────────────────────────────────────────────────────────────
#FUZZY_THRESHOLD = 88   # minimum similarity % to accept a match
#FUZZY_MIN_TOKEN = 3    # ignore tokens shorter than this
#FUZZY_MAX_TOKEN = 50   # ignore tokens longer than this (probably a sentence)
#
## ──────────────────────────────────────────────────────────────
## TOKEN SPLIT PATTERN
## Split on: comma, pipe, bullet, newline, tab, slash, ampersand etc.
## ──────────────────────────────────────────────────────────────
#TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")
#
#
## ══════════════════════════════════════════════════════════════
## MODULE-LEVEL INITIALISATION  (runs once at import time)
## ══════════════════════════════════════════════════════════════
#
#def _load_taxonomy() -> dict:
#    for path in _TAXONOMY_PATHS:
#        if path.exists():
#            with open(path, "r", encoding="utf-8") as f:
#                return json.load(f)
#    raise FileNotFoundError(
#        "skills_taxonomy.json not found. "
#        "Run build_skills_taxonomy.py first.\n"
#        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
#    )
#
#
#def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
#    """
#    Build FlashText processor from taxonomy.
#    Each alias maps back to canonical name.
#    e.g. "reactjs" → "React", "k8s" → "Kubernetes"
#    """
#    kp = KeywordProcessor(case_sensitive=False)
#    for canonical, entry in taxonomy.items():
#        kp.add_keyword(canonical, canonical)
#        for alias in entry.get("aliases", []):
#            if alias.strip():
#                kp.add_keyword(alias, canonical)
#    return kp
#
#
#def _build_canonical_list(taxonomy: dict) -> list:
#    return list(taxonomy.keys())
#
#
## ── Load once at import ──
#_TAXONOMY       = _load_taxonomy()
#_KP             = _build_keyword_processor(_TAXONOMY)
#_CANONICAL_LIST = _build_canonical_list(_TAXONOMY)
#
#print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills from taxonomy.")
#
#
## ══════════════════════════════════════════════════════════════
## HELPER FUNCTIONS
## ══════════════════════════════════════════════════════════════
#
#def _category_of(skill_name: str) -> str:
#    entry = _TAXONOMY.get(skill_name)
#    if entry:
#        return entry.get("category", "other")
#    return "other"
#
#
#def _tokenize_for_fuzzy(text: str) -> list:
#    """
#    Split skills section into individual tokens for fuzzy matching.
#    Each token = one candidate skill name FlashText didn't match exactly.
#    """
#    raw_tokens = TOKEN_SPLIT_RE.split(text)
#    tokens = []
#    for t in raw_tokens:
#        t = t.strip()
#        if FUZZY_MIN_TOKEN <= len(t) <= FUZZY_MAX_TOKEN:
#            tokens.append(t)
#    return tokens
#
#
## ══════════════════════════════════════════════════════════════
## MAIN EXTRACTOR FUNCTION
## ══════════════════════════════════════════════════════════════
#
#def extract_skills(sections: dict, full_text: str = "") -> dict:
#    """
#    Extract skills from the skills section of a resume only.
#
#    Args:
#        sections  : dict from segmenter. Only "skills" key is used.
#        full_text : kept for API compatibility, not used.
#
#    Returns:
#        {
#            "skills":             sorted list of canonical skill names,
#            "skills_by_category": dict of category -> [skills],
#            "_confidence":        float 0.0-1.0,
#            "_method":            which steps fired,
#            "_timing_ms":         time taken
#        }
#    """
#    t0 = time.perf_counter()
#
#    found_skills  = set()
#    methods_used  = []
#
#    # ── PRIMARY SOURCE: skills section only ───────────────────
#    skills_text = sections.get("skills", "") or ""
#
#    # ── STEP 1: FlashText exact + alias match ─────────────────
#    # Scans the entire skills section in one pass.
#    # Catches: "reactjs"→"React", "k8s"→"Kubernetes", "postgres"→"PostgreSQL"
#    if skills_text.strip():
#        matches = _KP.extract_keywords(skills_text)
#        if matches:
#            found_skills.update(matches)
#            methods_used.append("flashtext_skills_section")
#
#    # ── STEP 2: RapidFuzz fuzzy match ─────────────────────────
#    # Tokenize skills section and fuzzy-match each token.
#    # Catches misspellings: "Postgress"→"PostgreSQL", "Kubernets"→"Kubernetes"
#    # Only runs on skills section — never on full text (avoids false positives).
#    if skills_text.strip():
#        tokens     = _tokenize_for_fuzzy(skills_text)
#        fuzzy_hits = []
#
#        for token in tokens:
#            if token in found_skills:
#                continue   # already found by exact match
#
#            result = process.extractOne(
#                token,
#                _CANONICAL_LIST,
#                scorer=fuzz.WRatio,
#                score_cutoff=FUZZY_THRESHOLD,
#            )
#            if result:
#                canonical_match, score, _ = result
#                found_skills.add(canonical_match)
#                fuzzy_hits.append(f"{token}→{canonical_match}({score:.0f}%)")
#
#        if fuzzy_hits:
#            preview = ", ".join(fuzzy_hits[:5])
#            suffix  = "..." if len(fuzzy_hits) > 5 else ""
#            methods_used.append(f"fuzzy({preview}{suffix})")
#
#    # ── BUILD OUTPUT ──────────────────────────────────────────
#    sorted_skills = sorted(found_skills)
#
#    skills_by_category = {}
#    for skill in sorted_skills:
#        cat = _category_of(skill)
#        skills_by_category.setdefault(cat, []).append(skill)
#
#    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
#        confidence = 0.92
#    elif len(sorted_skills) >= 1:
#        confidence = 0.75
#    else:
#        confidence = 0.0
#
#    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
#
#    return {
#        "skills":              sorted_skills,
#        "skills_by_category":  skills_by_category,
#        "_confidence":         confidence,
#        "_method":             " | ".join(methods_used) if methods_used else "none",
#        "_timing_ms":          elapsed_ms,
#    }























#"""
#extractors/skills.py
#━━━━━━━━━━━━━━━━━━━
#Layer 2 — Skills Extractor
#
#Pipeline:
#  Step 1 → FlashText exact/alias match on skills section text
#  Step 2 → FlashText scan on full resume text (catches skills in bullet points)
#  Step 3 → RapidFuzz fuzzy match for misspellings on skills section tokens
#
#Input:  sections dict from segmenter  (keys: "skills", "experience", etc.)
#        full_text string (entire cleaned resume)
#
#Output: {
#    "skills": ["Python", "AWS", "React", ...],   ← canonical names, sorted
#    "skills_by_category": {
#        "programming_language": ["Python", "Go"],
#        "cloud_infra": ["AWS", "GCP"],
#        ...
#    },
#    "_confidence": 0.90,
#    "_method": "flashtext+fuzzy"
#}
#"""
#
#import json
#import re
#import time
#from pathlib import Path
#
#from flashtext import KeywordProcessor
#from rapidfuzz import process, fuzz
#
#
## ──────────────────────────────────────────────────────────────
## TAXONOMY PATH  (relative to project root OR absolute)
## Adjust this if your folder layout differs.
## ──────────────────────────────────────────────────────────────
#_TAXONOMY_PATHS = [
#    Path(__file__).parent.parent / "data" / "skills_taxonomy.json",   # ats_parser/data/
#    Path("data/skills_taxonomy.json"),                                  # cwd/data/
#]
#
## ──────────────────────────────────────────────────────────────
## FUZZY MATCH SETTINGS
## ──────────────────────────────────────────────────────────────
#FUZZY_THRESHOLD   = 88   # minimum similarity % (0-100) to accept a match
#FUZZY_MIN_TOKEN   = 3    # ignore tokens shorter than this (avoids "C" → "C#")
#FUZZY_MAX_TOKEN   = 50   # ignore tokens longer than this (probably a sentence)
#
## ──────────────────────────────────────────────────────────────
## TOKEN SPLIT PATTERN  (how to break the skills section into chunks)
## Split on: comma, pipe, bullet, middle-dot, newline, tab, slash, ampersand
## ──────────────────────────────────────────────────────────────
#TOKEN_SPLIT_RE = re.compile(r"[,|•·\n\t/&;()\[\]]+")
#
#
## ══════════════════════════════════════════════════════════════
## MODULE-LEVEL INITIALISATION  (runs once at import time)
## ══════════════════════════════════════════════════════════════
#
#def _load_taxonomy() -> dict:
#    """Load skills_taxonomy.json from the first path that exists."""
#    for path in _TAXONOMY_PATHS:
#        if path.exists():
#            with open(path, "r", encoding="utf-8") as f:
#                return json.load(f)
#    raise FileNotFoundError(
#        "skills_taxonomy.json not found. "
#        "Run build_skills_taxonomy.py first.\n"
#        f"Looked in: {[str(p) for p in _TAXONOMY_PATHS]}"
#    )
#
#
#def _build_keyword_processor(taxonomy: dict) -> KeywordProcessor:
#    """
#    Build a FlashText KeywordProcessor from the taxonomy.
#
#    FlashText uses the Aho-Corasick algorithm internally.
#    It scans text ONCE and finds ALL keywords simultaneously — 
#    50× faster than running thousands of regex patterns one by one.
#
#    Each alias maps back to the canonical name so we always store
#    the standard form (e.g. "reactjs" → "React").
#    """
#    kp = KeywordProcessor(case_sensitive=False)
#    for canonical, entry in taxonomy.items():
#        # Add the canonical name itself
#        kp.add_keyword(canonical, canonical)
#        # Add every alias, mapping it to the canonical name
#        for alias in entry.get("aliases", []):
#            if alias.strip():
#                kp.add_keyword(alias, canonical)
#    return kp
#
#
#def _build_canonical_list(taxonomy: dict) -> list[str]:
#    """Return just the canonical names for fuzzy matching."""
#    return list(taxonomy.keys())
#
#
## ── Load everything once at module import ──
#_TAXONOMY      = _load_taxonomy()
#_KP            = _build_keyword_processor(_TAXONOMY)
#_CANONICAL_LIST = _build_canonical_list(_TAXONOMY)
#
#print(f"[skills extractor] Loaded {len(_TAXONOMY):,} skills from taxonomy.")
#
#
## ══════════════════════════════════════════════════════════════
## HELPER FUNCTIONS
## ══════════════════════════════════════════════════════════════
#
#def _category_of(skill_name: str) -> str:
#    """Return the category for a canonical skill name."""
#    entry = _TAXONOMY.get(skill_name)
#    if entry:
#        return entry.get("category", "other")
#    return "other"
#
#
#def _tokenize_for_fuzzy(text: str) -> list[str]:
#    """
#    Split the skills section text into individual tokens for fuzzy matching.
#    Each token is a candidate skill name that FlashText didn't recognise exactly.
#
#    We split on delimiters and strip whitespace.
#    Tokens that are too short or too long are discarded.
#    """
#    raw_tokens = TOKEN_SPLIT_RE.split(text)
#    tokens = []
#    for t in raw_tokens:
#        t = t.strip()
#        if FUZZY_MIN_TOKEN <= len(t) <= FUZZY_MAX_TOKEN:
#            tokens.append(t)
#    return tokens
#
#
## ══════════════════════════════════════════════════════════════
## MAIN EXTRACTOR FUNCTION
## ══════════════════════════════════════════════════════════════
#
#def extract_skills(sections: dict, full_text: str = "") -> dict:
#    """
#    Extract skills from a resume.
#
#    Args:
#        sections : dict produced by the section segmenter.
#                   Expected keys (all optional):
#                     "skills"     → text of the skills section
#                     "experience" → text of work experience section
#                     "projects"   → text of projects section
#                     "summary"    → text of summary/objective section
#        full_text : entire cleaned resume text as a single string.
#                    Used for a broad FlashText scan to catch skills
#                    mentioned in bullet points outside the skills section.
#
#    Returns:
#        dict with keys:
#          "skills"            → sorted list of canonical skill names
#          "skills_by_category"→ dict mapping category → list of skills
#          "_confidence"       → float 0.0–1.0
#          "_method"           → string describing which methods fired
#          "_timing_ms"        → extraction time in milliseconds
#    """
#    t0 = time.perf_counter()
#
#    found_skills: set[str] = set()
#    methods_used: list[str] = []
#
#    # ── Pull relevant section texts ──────────────────────────
#    skills_text     = sections.get("skills", "") or ""
#    experience_text = sections.get("experience", "") or ""
#    projects_text   = sections.get("projects", "") or ""
#    summary_text    = sections.get("summary", "") or ""
#
#    # ── STEP 1: FlashText on the dedicated skills section ────
#    # This is the highest-quality signal: the candidate explicitly
#    # listed these as their skills.
#    if skills_text.strip():
#        matches = _KP.extract_keywords(skills_text)
#        if matches:
#            found_skills.update(matches)
#            methods_used.append("flashtext_skills_section")
#
#    # ── STEP 2: FlashText on experience + projects + summary ─
#    # Developers often write "built a RAG pipeline using LangChain"
#    # in their experience bullets — these are real skills too.
#    secondary_texts = " ".join(filter(None, [
#        experience_text, projects_text, summary_text
#    ]))
#    if secondary_texts.strip():
#        sec_matches = _KP.extract_keywords(secondary_texts)
#        if sec_matches:
#            new_from_secondary = set(sec_matches) - found_skills
#            found_skills.update(sec_matches)
#            if new_from_secondary:
#                methods_used.append("flashtext_experience_projects")
#
#    # ── STEP 3: Broad scan of full_text as safety net ────────
#    # Catches skills in headers, certifications, or anywhere else.
#    if full_text.strip():
#        full_matches = _KP.extract_keywords(full_text)
#        if full_matches:
#            new_from_full = set(full_matches) - found_skills
#            found_skills.update(full_matches)
#            if new_from_full:
#                methods_used.append("flashtext_full_text")
#
#    # ── STEP 4: RapidFuzz fuzzy match on skills section ──────
#    # Catches misspellings like "Postgress", "Pyhton", "Kubernets".
#    # We ONLY fuzz the skills section (not full text) to avoid
#    # false positives from normal English words.
#    if skills_text.strip():
#        tokens = _tokenize_for_fuzzy(skills_text)
#        fuzzy_found: list[str] = []
#
#        for token in tokens:
#            # Skip if already found exactly by FlashText
#            if token in found_skills:
#                continue
#
#            # RapidFuzz: find the single best matching canonical skill
#            # WRatio handles token order differences well (e.g. "JS React" vs "React JS")
#            result = process.extractOne(
#                token,
#                _CANONICAL_LIST,
#                scorer=fuzz.WRatio,
#                score_cutoff=FUZZY_THRESHOLD,
#            )
#            if result:
#                canonical_match, score, _ = result
#                found_skills.add(canonical_match)
#                fuzzy_found.append(f"{token}→{canonical_match}({score:.0f}%)")
#
#        if fuzzy_found:
#            methods_used.append(f"fuzzy({', '.join(fuzzy_found[:5])}{'...' if len(fuzzy_found)>5 else ''})")
#
#    # ── BUILD OUTPUT ─────────────────────────────────────────
#    sorted_skills = sorted(found_skills)
#
#    # Group by category
#    skills_by_category: dict[str, list[str]] = {}
#    for skill in sorted_skills:
#        cat = _category_of(skill)
#        skills_by_category.setdefault(cat, []).append(skill)
#
#    # Confidence: higher if we found skills in the dedicated section
#    if "flashtext_skills_section" in methods_used and len(sorted_skills) >= 3:
#        confidence = 0.92
#    elif len(sorted_skills) >= 1:
#        confidence = 0.75
#    else:
#        confidence = 0.0
#
#    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
#
#    return {
#        "skills":              sorted_skills,
#        "skills_by_category":  skills_by_category,
#        "_confidence":         confidence,
#        "_method":             " | ".join(methods_used) if methods_used else "none",
#        "_timing_ms":          elapsed_ms,
#    }