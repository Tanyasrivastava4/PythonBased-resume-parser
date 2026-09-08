"""
extractors/contact_extractor.py
Layer 2 — Contact Extractor

Extracts: name, email, phone, LinkedIn URL, GitHub URL, personal website.
Pure regex, no ML — these formats are standardised enough that regex
alone gets very high accuracy, as long as the regex is actually correct.

This file combines and fixes ideas from two earlier drafts. Every fix
below was independently verified with a test before being included --
see test_contact_extractor.py for the actual test cases.

BUGS FOUND AND FIXED:

1. LinkedIn/GitHub URL digits being misread as a phone number.
   e.g. "linkedin.com/in/anshika-singh-279926144" -> a naive phone
   regex extracts "279926144" as if it were a phone number. This is
   NOT prevented by only checking the immediate left/right characters
   around a digit sequence (a "lookaround" boundary check) -- a
   hyphen-and-letters boundary still looks like a valid phone
   boundary to that kind of check. It only showed up as wrong in our
   real test data when a LinkedIn URL happened to appear BEFORE the
   real phone number in the text; with the opposite ordering, the
   bug stayed hidden because the code took the first valid-looking
   match and stopped. Fixed by removing emails and LinkedIn/GitHub
   URLs from the text entirely before searching for phone numbers --
   so their digits never get a chance to be evaluated at all,
   regardless of ordering.

2. A "website" field that was fully designed (a complete, working
   regex for personal websites/portfolios) but never actually wired
   into the extraction function -- it would always return None.
   Fixed by actually calling it. Doing so directly exposed a second,
   smaller issue: the website pattern alone also matches email
   domains (e.g. "gmail.com" inside "jane@gmail.com") and LinkedIn/
   GitHub URLs (since those are also "word.tld" shaped). Fixed the
   same way as the phone bug -- strip emails and social URLs from the
   text first, then search what's left for a website.

3. A placeholder-email filter that rejected any email containing the
   substring "example" ANYWHERE in it -- which would incorrectly
   reject a real email like "sarah.example.recruiter@gmail.com" just
   because "example" happens to appear in the username. Fixed by only
   checking the DOMAIN portion (the part after @) against a list of
   known placeholder domains, leaving the username untouched.

4. NAME extraction added. A name has no unique syntax (unlike email/
   phone/URLs), so it is NOT searched across the full resume text --
   that invites false positives from job titles and company names
   that are also Title-Case/ALL-CAPS. Instead we require the caller to
   pass in Layer 1's "header" section specifically (see
   extract_header_section() below), and walk it top-down line by line,
   rejecting anything that's a contact line, an address line (has a
   digit), or resume/job-title boilerplate, before accepting the first
   remaining name-shaped line. This is regex-first; if nothing matches,
   spaCy PERSON NER is tried as a lower-confidence fallback.
"""

import re
from typing import Optional

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None


EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)"                  # not preceded by a letter/digit/underscore
    r"(\+?[\d\s\-().]{7,18})"   # the phone number itself
    r"(?!\w)",                  # not followed by a letter/digit/underscore
    re.IGNORECASE
)

LINKEDIN_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com/(?:in|pub|profile)/|in/|linkedin/)([\w\-]{3,100})/?",
    re.IGNORECASE
)

GITHUB_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:github\.com/|github/)([\w\-]{3,50})/?",
    re.IGNORECASE
)

WEBSITE_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?([\w\-]+\.(?:com|io|dev|me|co\.in|in|net|org)(?:/[\w\-/]*)?)",
    re.IGNORECASE
)

# Digit sequences that LOOK like a phone number but are something else --
# a bare 4-digit year, a 3-digit code, a short date-range fragment, etc.
PHONE_BLACKLIST = re.compile(r"^(19|20)\d{2}$|^\d{3,4}$")

# Domains used as generic placeholders in templates/examples -- if an
# email's DOMAIN (not the whole address) matches one of these, treat it
# as not a real contact email.
PLACEHOLDER_DOMAINS = {"example.com", "example.org", "domain.com", "test.com", "email.com"}

# Keywords that indicate an email belongs to a reference, boss, manager, or supervisor
SKIP_EMAIL_KEYWORDS = [
    "reference", "referred by", "manager email", "supervisor email", "boss email", "ref email", "reference email"
]

# GitHub reserves these path names for its own site navigation -- they
# can never be real usernames, so a match here is a false positive from
# a GitHub link that isn't actually a profile (e.g. "github.com/pricing").
GITHUB_RESERVED_PATHS = {"features", "about", "contact", "pricing", "login", "settings", "explore", "trending", "topics"}

# ── NAME EXTRACTION CONSTANTS ─────────────────────────────
PHONE_HINT_PATTERN = re.compile(r"\d{3}.{0,3}\d{3}.{0,4}\d{3,4}")
URL_HINT_PATTERN = re.compile(r"(linkedin\.com|github\.com|https?://|www\.)", re.IGNORECASE)
DIGIT_PATTERN = re.compile(r"\d")
NAME_PREFIX_CLEANER = re.compile(r"^(?:name|candidate name|cv|curriculum vitae)\s*[:\-]\s*", re.IGNORECASE)

NON_NAME_BLACKLIST = {
    "resume", "curriculum vitae", "cv", "biodata", "profile",
    "personal summary", "professional summary", "career summary",
    "core competencies", "professional experience", "work experience",
    "technical skills", "key skills", "education", "certifications",
    "projects", "objective", "summary", "contact", "contact information",
}

TITLE_WORDS = {
    "consultant", "engineer", "developer", "manager", "analyst", "lead",
    "specialist", "architect", "director", "executive", "administrator",
    "coordinator", "designer", "officer", "intern", "resume", "profile",
}

NAME_LINE_PATTERN = re.compile(r"^[A-Z][a-zA-Z.\-']*(?:\s+[A-Z][a-zA-Z.\-']*){0,3}$")


def _strip_known_matches(text: str) -> str:
    """
    Removes every email, LinkedIn URL, and GitHub URL from a COPY of the
    text. Used before searching for phone numbers or websites, so that
    digits or domain-shaped fragments INSIDE those URLs never get a
    chance to be mistaken for something else.
    """
    cleaned = EMAIL_PATTERN.sub(" ", text)
    cleaned = LINKEDIN_PATTERN.sub(" ", cleaned)
    cleaned = GITHUB_PATTERN.sub(" ", cleaned)
    return cleaned


def _is_placeholder_email(email: str) -> bool:
    """
    Checks only the DOMAIN portion of an email against known placeholder
    domains.
    """
    domain = email.rsplit("@", 1)[-1].lower()
    return domain in PLACEHOLDER_DOMAINS


def _is_reference_email(email: str, text: str) -> bool:
    """
    Checks if an email is listed under a reference/manager context line.
    """
    email_low = email.lower()
    for line in text.splitlines():
        if email_low in line.lower():
            line_low = line.lower()
            if any(kw in line_low for kw in SKIP_EMAIL_KEYWORDS):
                return True
    return False


def clean_phone(raw_phone: str) -> str:
    """
    Cleans and validates a raw phone-number-shaped string.
    """
    digits_only = re.sub(r"\D", "", raw_phone)

    if len(digits_only) < 7 or len(digits_only) > 15:
        return None

    if PHONE_BLACKLIST.match(digits_only):
        return None

    cleaned = re.sub(r"[^\d+\-\s()]", "", raw_phone).strip()
    return cleaned


# ── NAME HELPERS ───────────────────────────────────────────

def extract_header_section(full_text: str) -> str:
    """
    Pulls just the "header" block out of a Layer 1 *_segmented.txt file.
    """
    blocks = re.split(r"─{5,}", full_text)
    for i, block in enumerate(blocks):
        if re.match(r"\s*LABEL:\s*header", block):
            return blocks[i + 1].strip() if i + 1 < len(blocks) else ""

    # Fallback: no Layer 1 labels present -- use first 5 non-empty lines.
    lines = [l for l in full_text.splitlines() if l.strip()][:5]
    return "\n".join(lines)


def _is_contact_line(line: str) -> bool:
    return bool(EMAIL_PATTERN.search(line) or URL_HINT_PATTERN.search(line) or PHONE_HINT_PATTERN.search(line))


def _is_address_line(line: str) -> bool:
    return bool(DIGIT_PATTERN.search(line))


def _is_blacklisted_line(line: str) -> bool:
    lowered = line.strip().lower()
    if lowered in NON_NAME_BLACKLIST:
        return True
    return any(w in TITLE_WORDS for w in lowered.split())


def _regex_name_candidate(header_text: str) -> Optional[str]:
    for raw_line in header_text.splitlines():
        line = raw_line.strip()
        if not line or _is_blacklisted_line(line):
            continue

        parts = [p.strip() for p in re.split(r"[|•]", line) if p.strip()]
        for part in parts:
            part = NAME_PREFIX_CLEANER.sub("", part).strip()
            if not part or _is_contact_line(part) or _is_address_line(part) or _is_blacklisted_line(part):
                continue
            if NAME_LINE_PATTERN.match(part) or (part.isupper() and 1 <= len(part.split()) <= 4):
                words = part.split()
                if 1 <= len(words) <= 4:
                    return " ".join(w.capitalize() for w in words)
    return None


def _ner_name_candidate(header_text: str) -> Optional[str]:
    if _NLP is None:
        return None
    doc = _NLP(header_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return None


def extract_name(full_text: str) -> dict:
    """
    Extracts candidate name.
    """
    result = {"value": None, "confidence": 0.0, "method": "regex"}

    header_text = extract_header_section(full_text)
    if not header_text.strip():
        return result

    name = _regex_name_candidate(header_text)
    if name:
        result.update(value=name, confidence=0.9, method="regex")
        return result

    name = _ner_name_candidate(header_text)
    if name:
        result.update(value=name, confidence=0.6, method="ner")
        return result

    return result


def extract_contact(full_text: str) -> dict:
    """
    Main entry point for the contact extractor.
    """
    result = {
        "name": {"value": None, "confidence": 0.0, "method": "regex"},
        "email": {"value": None, "confidence": 0.0, "method": "regex"},
        "phone": {"value": None, "confidence": 0.0, "method": "regex"},
        "linkedin": {"value": None, "confidence": 0.0, "method": "regex"},
        "github": {"value": None, "confidence": 0.0, "method": "regex"},
        "website": {"value": None, "confidence": 0.0, "method": "regex"},
    }

    # ── NAME ───────────────────────────────────────────────
    result["name"] = extract_name(full_text)

    # ── EMAIL ──────────────────────────────────────────────
    raw_emails = EMAIL_PATTERN.findall(full_text)
    real_emails = [e for e in raw_emails if not _is_placeholder_email(e)]
    primary_email = None
    for e in real_emails:
        if not _is_reference_email(e, full_text):
            primary_email = e
            break
    if not primary_email and real_emails:
        primary_email = real_emails[0]

    if primary_email:
        result["email"] = {
            "value": primary_email.lower(),
            "confidence": 0.99,
            "method": "regex",
        }

    # ── LINKEDIN ───────────────────────────────────────────
    linkedin_match = LINKEDIN_PATTERN.search(full_text)
    if linkedin_match:
        username = linkedin_match.group(1)
        result["linkedin"] = {
            "value": f"linkedin.com/in/{username}",
            "confidence": 0.99,
            "method": "regex",
        }

    # ── GITHUB ─────────────────────────────────────────────
    github_match = GITHUB_PATTERN.search(full_text)
    if github_match:
        username = github_match.group(1)
        if username.lower() not in GITHUB_RESERVED_PATHS:
            result["github"] = {
                "value": f"github.com/{username}",
                "confidence": 0.99,
                "method": "regex",
            }

    # ── PHONE ──────────────────────────────────────────────
    phone_search_text = _strip_known_matches(full_text)
    for raw_match in PHONE_PATTERN.findall(phone_search_text):
        cleaned = clean_phone(raw_match)
        if cleaned:
            result["phone"] = {
                "value": cleaned,
                "confidence": 0.95,
                "method": "regex",
            }
            break

    # ── WEBSITE ────────────────────────────────────────────
    website_search_text = _strip_known_matches(full_text)
    website_match = WEBSITE_PATTERN.search(website_search_text)
    if website_match:
        result["website"] = {
            "value": website_match.group(1),
            "confidence": 0.85,
            "method": "regex",
        }

    return result














## working commenting just for adding name pattern also
#"""
#extractors/contact_extractor.py
#Layer 2 — Contact Extractor
#
#Extracts: email, phone, LinkedIn URL, GitHub URL, personal website.
#Pure regex, no ML — these formats are standardised enough that regex
#alone gets very high accuracy, as long as the regex is actually correct.
#
#This file combines and fixes ideas from two earlier drafts. Every fix
#below was independently verified with a test before being included --
#see test_contact_extractor.py for the actual test cases.
#
#BUGS FOUND AND FIXED:
#
#1. LinkedIn/GitHub URL digits being misread as a phone number.
#   e.g. "linkedin.com/in/anshika-singh-279926144" -> a naive phone
#   regex extracts "279926144" as if it were a phone number. This is
#   NOT prevented by only checking the immediate left/right characters
#   around a digit sequence (a "lookaround" boundary check) -- a
#   hyphen-and-letters boundary still looks like a valid phone
#   boundary to that kind of check. It only showed up as wrong in our
#   real test data when a LinkedIn URL happened to appear BEFORE the
#   real phone number in the text; with the opposite ordering, the
#   bug stayed hidden because the code took the first valid-looking
#   match and stopped. Fixed by removing emails and LinkedIn/GitHub
#   URLs from the text entirely before searching for phone numbers --
#   so their digits never get a chance to be evaluated at all,
#   regardless of ordering.
#
#2. A "website" field that was fully designed (a complete, working
#   regex for personal websites/portfolios) but never actually wired
#   into the extraction function -- it would always return None.
#   Fixed by actually calling it. Doing so directly exposed a second,
#   smaller issue: the website pattern alone also matches email
#   domains (e.g. "gmail.com" inside "jane@gmail.com") and LinkedIn/
#   GitHub URLs (since those are also "word.tld" shaped). Fixed the
#   same way as the phone bug -- strip emails and social URLs from the
#   text first, then search what's left for a website.
#
#3. A placeholder-email filter that rejected any email containing the
#   substring "example" ANYWHERE in it -- which would incorrectly
#   reject a real email like "sarah.example.recruiter@gmail.com" just
#   because "example" happens to appear in the username. Fixed by only
#   checking the DOMAIN portion (the part after @) against a list of
#   known placeholder domains, leaving the username untouched.
#"""
#
#import re
#
#
#EMAIL_PATTERN = re.compile(
#    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
#    re.IGNORECASE
#)
#
#PHONE_PATTERN = re.compile(
#    r"(?<!\w)"                  # not preceded by a letter/digit/underscore
#    r"(\+?[\d\s\-().]{7,18})"   # the phone number itself
#    r"(?!\w)",                  # not followed by a letter/digit/underscore
#    re.IGNORECASE
#)
#
#LINKEDIN_PATTERN = re.compile(
#    r"(?:https?://)?(?:www\.)?linkedin\.com/in/([\w\-]+)/?",
#    re.IGNORECASE
#)
#
#GITHUB_PATTERN = re.compile(
#    r"(?:https?://)?(?:www\.)?github\.com/([\w\-]+)/?",
#    re.IGNORECASE
#)
#
#WEBSITE_PATTERN = re.compile(
#    r"(?:https?://)?(?:www\.)?([\w\-]+\.(?:com|io|dev|me|co\.in|in|net|org)(?:/[\w\-/]*)?)",
#    re.IGNORECASE
#)
#
## Digit sequences that LOOK like a phone number but are something else --
## a bare 4-digit year, a 3-digit code, a short date-range fragment, etc.
#PHONE_BLACKLIST = re.compile(r"^(19|20)\d{2}$|^\d{3,4}$")
#
## Domains used as generic placeholders in templates/examples -- if an
## email's DOMAIN (not the whole address) matches one of these, treat it
## as not a real contact email.
#PLACEHOLDER_DOMAINS = {"example.com", "example.org", "domain.com", "test.com", "email.com"}
#
## GitHub reserves these path names for its own site navigation -- they
## can never be real usernames, so a match here is a false positive from
## a GitHub link that isn't actually a profile (e.g. "github.com/pricing").
#GITHUB_RESERVED_PATHS = {"features", "about", "contact", "pricing", "login", "settings", "explore"}
#
#
#def _strip_known_matches(text: str) -> str:
#    """
#    Removes every email, LinkedIn URL, and GitHub URL from a COPY of the
#    text. Used before searching for phone numbers or websites, so that
#    digits or domain-shaped fragments INSIDE those URLs never get a
#    chance to be mistaken for something else. This is what fixes bug 1
#    and the second half of bug 2 -- and crucially, it works regardless
#    of which order these things appear in the text, unlike an approach
#    that just takes "whichever valid match comes first."
#    """
#    cleaned = EMAIL_PATTERN.sub(" ", text)
#    cleaned = LINKEDIN_PATTERN.sub(" ", cleaned)
#    cleaned = GITHUB_PATTERN.sub(" ", cleaned)
#    return cleaned
#
#
#def _is_placeholder_email(email: str) -> bool:
#    """
#    Checks only the DOMAIN portion of an email against known placeholder
#    domains. Fixes bug 3: checking the whole email string for the
#    substring "example" would wrongly reject a real email like
#    "sarah.example.recruiter@gmail.com".
#    """
#    domain = email.rsplit("@", 1)[-1].lower()
#    return domain in PLACEHOLDER_DOMAINS
#
#
#def clean_phone(raw_phone: str) -> str:
#    """
#    Cleans and validates a raw phone-number-shaped string.
#    Returns None if it doesn't actually look like a real phone number
#    (too short, too long, or matches a known non-phone pattern like a
#    bare year).
#    """
#    digits_only = re.sub(r"\D", "", raw_phone)
#
#    if len(digits_only) < 7 or len(digits_only) > 15:
#        return None
#
#    if PHONE_BLACKLIST.match(digits_only):
#        return None
#
#    cleaned = re.sub(r"[^\d+\-\s()]", "", raw_phone).strip()
#    return cleaned
#
#
#def extract_contact(full_text: str) -> dict:
#    """
#    Main entry point for the contact extractor.
#
#    Takes the FULL resume text (not just the header section) and
#    returns every contact field found, each with a confidence score --
#    matching the architecture's Layer 3 design where every extracted
#    field carries a confidence value so low-confidence fields can be
#    flagged for review later.
#
#    We deliberately search the entire resume, not just the header,
#    since candidates sometimes place a LinkedIn/GitHub link in a
#    project description or footer rather than only at the top.
#    """
#    result = {
#        "email": {"value": None, "confidence": 0.0, "method": "regex"},
#        "phone": {"value": None, "confidence": 0.0, "method": "regex"},
#        "linkedin": {"value": None, "confidence": 0.0, "method": "regex"},
#        "github": {"value": None, "confidence": 0.0, "method": "regex"},
#        "website": {"value": None, "confidence": 0.0, "method": "regex"},
#    }
#
#    # ── EMAIL ──────────────────────────────────────────────
#    raw_emails = EMAIL_PATTERN.findall(full_text)
#    real_emails = [e for e in raw_emails if not _is_placeholder_email(e)]
#    if real_emails:
#        result["email"] = {
#            "value": real_emails[0].lower(),
#            "confidence": 0.99,
#            "method": "regex",
#        }
#
#    # ── LINKEDIN ───────────────────────────────────────────
#    linkedin_match = LINKEDIN_PATTERN.search(full_text)
#    if linkedin_match:
#        username = linkedin_match.group(1)
#        result["linkedin"] = {
#            "value": f"linkedin.com/in/{username}",
#            "confidence": 0.99,
#            "method": "regex",
#        }
#
#    # ── GITHUB ─────────────────────────────────────────────
#    github_match = GITHUB_PATTERN.search(full_text)
#    if github_match:
#        username = github_match.group(1)
#        if username.lower() not in GITHUB_RESERVED_PATHS:
#            result["github"] = {
#                "value": f"github.com/{username}",
#                "confidence": 0.99,
#                "method": "regex",
#            }
#
#    # ── PHONE ──────────────────────────────────────────────
#    # Search only the text with emails/LinkedIn/GitHub already removed,
#    # so a URL's digits can never be mistaken for a phone number.
#    phone_search_text = _strip_known_matches(full_text)
#    for raw_match in PHONE_PATTERN.findall(phone_search_text):
#        cleaned = clean_phone(raw_match)
#        if cleaned:
#            result["phone"] = {
#                "value": cleaned,
#                "confidence": 0.95,
#                "method": "regex",
#            }
#            break
#
#    # ── WEBSITE ────────────────────────────────────────────
#    # Same reasoning as phone: search only the text with emails/
#    # LinkedIn/GitHub already removed, so e.g. "gmail.com" inside an
#    # email address is never reported as someone's personal website.
#    website_search_text = _strip_known_matches(full_text)
#    website_match = WEBSITE_PATTERN.search(website_search_text)
#    if website_match:
#        result["website"] = {
#            "value": website_match.group(1),
#            "confidence": 0.85,
#            "method": "regex",
#        }
#
#    return result
#






















## editing for advance version
#"""
#extractors/contact_extractor.py
#Layer 2 — Contact Extractor
#
#Job: pull out email, phone number, LinkedIn URL, and GitHub URL from the
#resume's full text using regex. This is the most reliable extractor in
#the whole pipeline, since these formats are highly standardised and
#don't vary by resume template the way section headings do.
#
#Per the architecture: we scan the FULL resume text, not just the header
#section, since candidates sometimes place a LinkedIn/GitHub link in a
#footer or even inside a project description rather than only at the top.
#
#BUGS FOUND AND FIXED DURING TESTING (against real resume data):
#
#1. The original email pattern only allowed ONE dot in the domain
#   ending (e.g. "gmail.com"), so multi-part domains like
#   "iimidr.ac.in" got truncated to "iimidr.ac" -- silently dropping
#   the final ".in". Fixed by allowing any number of dot-separated
#   domain segments before the final one.
#
#2. The original phone pattern, run directly on raw text, wrongly
#   matched the numeric suffix of LinkedIn URLs as a phone number
#   (e.g. "linkedin.com/in/anshika-singh-279926144" produced a false
#   phone match of "279926144"). Fixed by extracting emails and
#   LinkedIn/GitHub URLs FIRST, removing them from the text, and only
#   then searching what's left for phone numbers -- so a URL's digits
#   never get a chance to be misread as a phone number.
#"""
#
#import re
#
#
#EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}")
#PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-().]{7,14}\d")
#LINKEDIN_PATTERN = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
#GITHUB_PATTERN = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)
#
#
#def extract_emails(text: str) -> list:
#    """Returns every email address found in the text, in order of appearance."""
#    return EMAIL_PATTERN.findall(text)
#
#
#def extract_linkedin_urls(text: str) -> list:
#    """Returns every LinkedIn profile URL found in the text."""
#    return LINKEDIN_PATTERN.findall(text)
#
#
#def extract_github_urls(text: str) -> list:
#    """Returns every GitHub profile URL found in the text."""
#    return GITHUB_PATTERN.findall(text)
#
#
#def extract_phones(text: str) -> list:
#    """
#    Returns every phone number found in the text.
#
#    Before searching for phone numbers, we strip out emails and
#    LinkedIn/GitHub URLs from a COPY of the text. This prevents the
#    numeric portion of a URL (e.g. a LinkedIn profile ID) from being
#    mistaken for a phone number -- a real false positive we found
#    during testing.
#    """
#    cleaned = EMAIL_PATTERN.sub(" ", text)
#    cleaned = LINKEDIN_PATTERN.sub(" ", cleaned)
#    cleaned = GITHUB_PATTERN.sub(" ", cleaned)
#    return PHONE_PATTERN.findall(cleaned)
#
#
#def extract_contact_info(full_resume_text: str) -> dict:
#    """
#    Main entry point for the contact extractor.
#
#    Takes the FULL resume text (not just the header section) and
#    returns a dictionary with the contact fields found, each carrying
#    a confidence score -- matching the architecture's Layer 3 design,
#    where every extracted field needs a confidence value so low-
#    confidence fields can later be flagged for review.
#
#    If a field has multiple matches (e.g. two phone numbers), we keep
#    the first one as the primary value but also return the full list,
#    so nothing is silently discarded.
#    """
#    emails = extract_emails(full_resume_text)
#    phones = extract_phones(full_resume_text)
#    linkedin_urls = extract_linkedin_urls(full_resume_text)
#    github_urls = extract_github_urls(full_resume_text)
#
#    result = {}
#
#    result["email"] = _build_field(emails, confidence=0.99)
#    result["phone"] = _build_field(phones, confidence=0.95)
#    result["linkedin"] = _build_field(linkedin_urls, confidence=0.99)
#    result["github"] = _build_field(github_urls, confidence=0.99)
#
#    return result
#
#
#def _build_field(matches: list, confidence: float) -> dict:
#    """
#    Builds the standard field-result shape used across the pipeline:
#    a primary value, a confidence score, the method used, and the full
#    list of all matches found (in case there's more than one, e.g. a
#    candidate listing two phone numbers).
#
#    If nothing was found at all, value is None and confidence is 0.0 --
#    this naturally flags as a low-confidence field for the review queue
#    described in the architecture's Layer 3.
#    """
#    if not matches:
#        return {
#            "value": None,
#            "confidence": 0.0,
#            "method": "regex",
#            "all_matches": [],
#        }
#
#    return {
#        "value": matches[0],
#        "confidence": confidence,
#        "method": "regex",
#        "all_matches": matches,
#    }