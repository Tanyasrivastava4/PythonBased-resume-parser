"""
ingestion/__init__.py
Text normalisation utilities — cleans raw text before parsing.
"""

import re
import unicodedata


_SIDEBAR_HEADINGS = [
    "PROFESSIONAL SUMMARY", "PERSONAL SUMMARY", "PERSONAL DETAILS",
    "CORE COMPETENCIES", "PROFILE SUMMARY", "LIVE PROJECTS",
    "WORK EXPERIENCE", "WORK HISTORY", "EMPLOYMENT HISTORY",
    "KEY SKILLS", "IT SKILLS", "TECHNICAL SKILLS", "LANGUAGE SKILLS",
    "ROLES & RESPONSIBILITIES", "ROLES RESPONSIBILITIES",
    "CUSTOM SECTION", "ACADEMIC BACKGROUND",
    "CERTIFICATION", "CERTIFICATIONS", "EDUCATION", "EXPERIENCE",
    "DECLARATION", "SUMMARY", "OBJECTIVE", "PROFILE",
    "SKILLS", "PROJECTS", "ACHIEVEMENTS", "AWARDS",
    "LANGUAGES", "INTERESTS", "HOBBIES", "REFERENCES",
    "PERSONAL",
]

_CID_GLYPH_PATTERN = re.compile(r"\(cid:\d+\)")


def _replace_cid_glyphs_with_bullets(text: str) -> str:
    return _CID_GLYPH_PATTERN.sub("•", text)


# Matches a PDF text-extraction fallback for an unmapped glyph from a
# legacy Wingdings/Symbol-style font, e.g. U+F076, U+F0B7, U+F0A7.
# Confirmed on a real resume (Ajit Kumar): five distinct PUA codepoints
# appeared, all bullet markers of one kind or another (section-heading
# bullets used a different codepoint than body-bullet-list markers) --
# same underlying bug family as _CID_GLYPH_PATTERN above (PDF extraction
# hit a glyph with no real Unicode mapping and fell back to something
# else), just a different fallback shape: an actual Unicode codepoint
# in the Private Use Area rather than a literal "(cid:N)" string.
#
# U+F020 is handled separately (see _normalize_pua_glyphs below) because
# it is reliably NOT a bullet: legacy symbol fonts map their internal
# byte codes into the PUA as 0xF000 + original_byte, and byte 0x20 is
# space in essentially every such encoding -- confirmed by testing,
# U+F020 always appeared glued to the end of the previous word (e.g.
# "years.\uf020") immediately before the next real bullet glyph, exactly
# where a plain space belongs.
_PUA_STANDALONE_TOKEN_RE = re.compile(r"(?<!\S)[\uE000-\uF8FF](?!\S)")


def _normalize_pua_glyphs(text: str) -> str:
    """
    Cleans up Private Use Area glyph fallbacks from legacy Wingdings/
    Symbol-style fonts. Two passes, order matters:

      1. U+F020 -> a real space (it's a space in the original font's
         encoding, not a bullet -- see docstring on the regex above).
         Must run FIRST, since it's often glued directly onto the
         previous word with no space in between (e.g. "years.\uf020"),
         and doing this first means the bullet glyph that follows it
         becomes properly whitespace-delimited for step 2 to catch.

      2. Any remaining PUA character appearing as its own
         whitespace-delimited token -> a real bullet "•". This is safe
         specifically because it only touches STANDALONE PUA tokens
         (surrounded by whitespace on both sides) -- a PUA character
         glued into the middle of a real word is left alone, since that
         pattern doesn't match a "this glyph IS the whole token" bullet
         marker.
    """
    text = text.replace("\uf020", " ")
    text = _PUA_STANDALONE_TOKEN_RE.sub("•", text)
    return text


def _split_merged_headings(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        split_done = False
        for heading in _SIDEBAR_HEADINGS:
            pattern = re.compile(
                rf"^({re.escape(heading)})\s+([•●\-\*]?\s*[A-Za-z0-9][^\n]{{3,}})$",
                re.IGNORECASE
            )
            match = pattern.match(stripped)
            if match:
                result.append(match.group(1))
                result.append(match.group(2))
                split_done = True
                break
        if not split_done:
            result.append(line)
    return "\n".join(result)


def _join_split_headings(text: str) -> str:
    multi_word_headings = [h for h in _SIDEBAR_HEADINGS if " " in h]
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        joined = False
        if i + 1 < len(lines):
            next_line = lines[i+1].strip()
            combined = line + " " + next_line
            for heading in multi_word_headings:
                if combined.upper() == heading.upper():
                    result.append(heading)
                    i += 2
                    joined = True
                    break
        if not joined:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def normalise_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = unicodedata.normalize("NFC", raw_text)
    for char in ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"]:
        text = text.replace(char, "")
    text = _replace_cid_glyphs_with_bullets(text)
    text = _normalize_pua_glyphs(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = _split_merged_headings(text)
    text = _join_split_headings(text)
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def read_resume_file(file_path: str) -> str:
    from pathlib import Path
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        from ingestion.pdf_reader import read_pdf
        raw_text = read_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        from ingestion.docx_reader import read_docx
        raw_text = read_docx(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
        from ingestion.ocr_reader import read_with_surya
        raw_text = read_with_surya(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: PDF, DOCX, JPG, PNG, TIFF")

    return normalise_text(raw_text)















##worked chaing so that one can work
#"""
#ingestion/__init__.py
#Text normalisation utilities — cleans raw text before parsing.
#
#FIX (this revision) — "(cid:127)" literal strings appearing instead of
#bullet characters:
#
#Some PDFs use a custom/embedded font for their bullet glyph that has
#no ToUnicode mapping in the font's character map. When a text
#extractor (pdfplumber/PyMuPDF) can't resolve a glyph to a real Unicode
#character, it falls back to printing the raw internal glyph ID as
#literal text, e.g. "(cid:127)". Confirmed on a real resume (Aasma):
#every bullet point in her skills, strengths, and personal-details
#sections came through as the literal string "(cid:127)" instead of
#"•". This isn't a section-heading problem, so it doesn't get fixed by
#anything in Layer 1 -- it needs to be cleaned up here in Layer 0
#normalisation, before the text ever reaches the segmenter, so that
#downstream bullet-aware logic (Layer 1's normalize_inline_bullets(),
#and eventually Layer 2's bullet-based skill tokenising) sees a real
#bullet character to work with.
#
#Fixed by replacing any "(cid:<digits>)" pattern with a real bullet
#character "•". This is a safe, general fix rather than one hardcoded
#to "(cid:127)" specifically -- different fonts/PDFs can fall back to
#different cid numbers for the same visual bullet glyph, and in every
#case we've seen, an unresolved (cid:N) in resume body text is a bullet
#marker, never meaningful content on its own.
#"""
#
#import re
#import unicodedata
#
#
#_SIDEBAR_HEADINGS = [
#    "PROFESSIONAL SUMMARY", "PERSONAL SUMMARY", "PERSONAL DETAILS",
#    "CORE COMPETENCIES", "PROFILE SUMMARY", "LIVE PROJECTS",
#    "WORK EXPERIENCE", "WORK HISTORY", "EMPLOYMENT HISTORY",
#    "KEY SKILLS", "IT SKILLS", "TECHNICAL SKILLS", "LANGUAGE SKILLS",
#    "ROLES & RESPONSIBILITIES", "ROLES RESPONSIBILITIES",
#    "CUSTOM SECTION", "ACADEMIC BACKGROUND",
#    "CERTIFICATION", "CERTIFICATIONS", "EDUCATION", "EXPERIENCE",
#    "DECLARATION", "SUMMARY", "OBJECTIVE", "PROFILE",
#    "SKILLS", "PROJECTS", "ACHIEVEMENTS", "AWARDS",
#    "LANGUAGES", "INTERESTS", "HOBBIES", "REFERENCES",
#    "PERSONAL",
#]
#
## Matches PDF text-extraction fallback output for an unmapped glyph,
## e.g. "(cid:127)". Confirmed on real resumes to always be a bullet
## marker in this context, never meaningful content.
#_CID_GLYPH_PATTERN = re.compile(r"\(cid:\d+\)")
#
#
#def _replace_cid_glyphs_with_bullets(text: str) -> str:
#    """
#    Replaces literal "(cid:N)" fallback text (produced when a PDF's
#    font has no Unicode mapping for a glyph -- almost always a custom
#    bullet character) with a real bullet character "•".
#    """
#    return _CID_GLYPH_PATTERN.sub("•", text)
#
#
#def _split_merged_headings(text: str) -> str:
#    """
#    Splits lines where a sidebar section heading got merged with content.
#    e.g. "EDUCATION 2018 Master..." -> "EDUCATION\n2018 Master..."
#    Safety net for sidebar-layout PDFs where pdf_reader misses some pages.
#    """
#    lines = text.split("\n")
#    result = []
#
#    for line in lines:
#        stripped = line.strip()
#        split_done = False
#
#        for heading in _SIDEBAR_HEADINGS:
#            pattern = re.compile(
#                rf"^({re.escape(heading)})\s+([•●\-\*]?\s*[A-Za-z0-9][^\n]{{3,}})$",
#                re.IGNORECASE
#            )
#            match = pattern.match(stripped)
#            if match:
#                result.append(match.group(1))
#                result.append(match.group(2))
#                split_done = True
#                break
#
#        if not split_done:
#            result.append(line)
#
#    return "\n".join(result)
#
#
#
#def _join_split_headings(text: str) -> str:
#    """
#    Joins consecutive single-word lines that together form a known
#    multi-word sidebar heading.
#    e.g. line N:   "PROFILE"
#         line N+1: "SUMMARY"
#      -> "PROFILE SUMMARY"
#    """
#    multi_word_headings = [h for h in _SIDEBAR_HEADINGS if " " in h]
#    lines = text.split("\n")
#    result = []
#    i = 0
#    while i < len(lines):
#        line = lines[i].strip()
#        joined = False
#        if i + 1 < len(lines):
#            next_line = lines[i+1].strip()
#            combined = line + " " + next_line
#            for heading in multi_word_headings:
#                if combined.upper() == heading.upper():
#                    result.append(heading)
#                    i += 2
#                    joined = True
#                    break
#        if not joined:
#            result.append(lines[i])
#            i += 1
#    return "\n".join(result)
#
#def normalise_text(raw_text: str) -> str:
#    """
#    Cleans raw extracted text.
#    Steps: Unicode normalise → remove invisibles → replace unmapped
#         cid glyphs with bullets → standardise line endings → fix
#         punctuation spaces → split merged headings → strip lines
#         → collapse blank lines
#    """
#    if not raw_text:
#        return ""
#
#    text = unicodedata.normalize("NFC", raw_text)
#
#    invisible_chars = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"]
#    for char in invisible_chars:
#        text = text.replace(char, "")
#
#    # Replace "(cid:N)" fallback text (unresolved PDF glyphs, almost
#    # always bullet markers) with a real bullet character. Done early,
#    # before the parenthesis-spacing fix below, so there's no
#    # interaction between the two.
#    text = _replace_cid_glyphs_with_bullets(text)
#
#    text = text.replace("\r\n", "\n").replace("\r", "\n")
#
#    text = re.sub(r"\(\s+", "(", text)
#    text = re.sub(r"\s+\)", ")", text)
#
#    # Safety net: split any remaining merged sidebar headings
#    text = _split_merged_headings(text)
#
#    # Join consecutive lines that together form a known multi-word
#    # heading.
#    # e.g. "PROFILE" on line N and "SUMMARY" on line N+1
#    #   -> "PROFILE SUMMARY" on one line
#    # This happens when sidebar detection splits a multi-word heading
#    # across two lines (each word on its own line).
#    text = _join_split_headings(text)
#
#    lines = [line.strip() for line in text.split("\n")]
#
#    cleaned_lines = []
#    blank_count   = 0
#    for line in lines:
#        if line == "":
#            blank_count += 1
#            if blank_count <= 2:
#                cleaned_lines.append(line)
#        else:
#            blank_count = 0
#            cleaned_lines.append(line)
#
#    return "\n".join(cleaned_lines).strip()
#
#
#def read_resume_file(file_path: str) -> str:
#    """
#    Master entry point for Layer 0.
#    Detects file type and routes to the right reader.
#    Returns clean normalised text.
#    """
#    from pathlib import Path
#    path = Path(file_path)
#    ext  = path.suffix.lower()
#
#    if ext == ".pdf":
#        from ingestion.pdf_reader import read_pdf
#        raw_text = read_pdf(file_path)
#    elif ext in [".docx", ".doc"]:
#        from ingestion.docx_reader import read_docx
#        raw_text = read_docx(file_path)
#    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
#        from ingestion.ocr_reader import read_with_surya
#        raw_text = read_with_surya(file_path)
#    else:
#        raise ValueError(
#            f"Unsupported file type: {ext}. Supported: PDF, DOCX, JPG, PNG, TIFF"
#        )
#
#    return normalise_text(raw_text)
#

















##just changing for segementer o/p
#"""
#ingestion/__init__.py
#Text normalisation utilities — cleans raw text before parsing.
#"""
#
#import re
#import unicodedata
#
#
#_SIDEBAR_HEADINGS = [
#    "PROFESSIONAL SUMMARY", "PERSONAL SUMMARY", "PERSONAL DETAILS",
#    "CORE COMPETENCIES", "PROFILE SUMMARY", "LIVE PROJECTS",
#    "WORK EXPERIENCE", "WORK HISTORY", "EMPLOYMENT HISTORY",
#    "KEY SKILLS", "IT SKILLS", "TECHNICAL SKILLS", "LANGUAGE SKILLS",
#    "ROLES & RESPONSIBILITIES", "ROLES RESPONSIBILITIES",
#    "CUSTOM SECTION", "ACADEMIC BACKGROUND",
#    "CERTIFICATION", "CERTIFICATIONS", "EDUCATION", "EXPERIENCE",
#    "DECLARATION", "SUMMARY", "OBJECTIVE", "PROFILE",
#    "SKILLS", "PROJECTS", "ACHIEVEMENTS", "AWARDS",
#    "LANGUAGES", "INTERESTS", "HOBBIES", "REFERENCES",
#    "PERSONAL",
#]
#
#
#def _split_merged_headings(text: str) -> str:
#    """
#    Splits lines where a sidebar section heading got merged with content.
#    e.g. "EDUCATION 2018 Master..." -> "EDUCATION\n2018 Master..."
#    Safety net for sidebar-layout PDFs where pdf_reader misses some pages.
#    """
#    lines = text.split("\n")
#    result = []
#
#    for line in lines:
#        stripped = line.strip()
#        split_done = False
#
#        for heading in _SIDEBAR_HEADINGS:
#            pattern = re.compile(
#                rf"^({re.escape(heading)})\s+([A-Za-z0-9][^\n]{{3,}})$",
#                re.IGNORECASE
#            )
#            match = pattern.match(stripped)
#            if match:
#                result.append(match.group(1))
#                result.append(match.group(2))
#                split_done = True
#                break
#
#        if not split_done:
#            result.append(line)
#
#    return "\n".join(result)
#
#
#
#def _join_split_headings(text: str) -> str:
#    """
#    Joins consecutive single-word lines that together form a known
#    multi-word sidebar heading.
#    e.g. line N:   "PROFILE"
#         line N+1: "SUMMARY"
#      -> "PROFILE SUMMARY"
#    """
#    multi_word_headings = [h for h in _SIDEBAR_HEADINGS if " " in h]
#    lines = text.split("\n")
#    result = []
#    i = 0
#    while i < len(lines):
#        line = lines[i].strip()
#        joined = False
#        if i + 1 < len(lines):
#            next_line = lines[i+1].strip()
#            combined = line + " " + next_line
#            for heading in multi_word_headings:
#                if combined.upper() == heading.upper():
#                    result.append(heading)
#                    i += 2
#                    joined = True
#                    break
#        if not joined:
#            result.append(lines[i])
#            i += 1
#    return "\n".join(result)
#
#def normalise_text(raw_text: str) -> str:
#    """
#    Cleans raw extracted text.
#    Steps: Unicode normalise → remove invisibles → standardise line endings
#         → fix punctuation spaces → split merged headings → strip lines
#         → collapse blank lines
#    """
#    if not raw_text:
#        return ""
#
#    text = unicodedata.normalize("NFC", raw_text)
#
#    invisible_chars = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"]
#    for char in invisible_chars:
#        text = text.replace(char, "")
#
#    text = text.replace("\r\n", "\n").replace("\r", "\n")
#
#    text = re.sub(r"\(\s+", "(", text)
#    text = re.sub(r"\s+\)", ")", text)
#
#    # Safety net: split any remaining merged sidebar headings
#    text = _split_merged_headings(text)
#
#    # Join consecutive lines that together form a known multi-word heading.
#    # e.g. "PROFILE" on line N and "SUMMARY" on line N+1
#    #   -> "PROFILE SUMMARY" on one line
#    # This happens when sidebar detection splits a multi-word heading
#    # across two lines (each word on its own line).
#    text = _join_split_headings(text)
#
#    lines = [line.strip() for line in text.split("\n")]
#
#    cleaned_lines = []
#    blank_count   = 0
#    for line in lines:
#        if line == "":
#            blank_count += 1
#            if blank_count <= 2:
#                cleaned_lines.append(line)
#        else:
#            blank_count = 0
#            cleaned_lines.append(line)
#
#    return "\n".join(cleaned_lines).strip()
#
#
#def read_resume_file(file_path: str) -> str:
#    """
#    Master entry point for Layer 0.
#    Detects file type and routes to the right reader.
#    Returns clean normalised text.
#    """
#    from pathlib import Path
#    path = Path(file_path)
#    ext  = path.suffix.lower()
#
#    if ext == ".pdf":
#        from ingestion.pdf_reader import read_pdf
#        raw_text = read_pdf(file_path)
#    elif ext in [".docx", ".doc"]:
#        from ingestion.docx_reader import read_docx
#        raw_text = read_docx(file_path)
#    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
#        from ingestion.ocr_reader import read_with_surya
#        raw_text = read_with_surya(file_path)
#    else:
#        raise ValueError(
#            f"Unsupported file type: {ext}. Supported: PDF, DOCX, JPG, PNG, TIFF"
#        )
#
#    return normalise_text(raw_text)
#






















##work changing because of fixing arti resume
#"""
#ingestion/__init__.py
#Text normalisation utilities — cleans raw text before parsing.
#"""
#
#import re
#import unicodedata
#
#
#def normalise_text(raw_text: str) -> str:
#    """
#    Cleans raw extracted text. Does the following:
#    1. Normalises Unicode (converts special characters to standard form)
#    2. Removes zero-width and invisible characters
#    3. Standardises line endings
#    4. Collapses excessive blank lines to max 2
#    5. Strips leading/trailing whitespace from each line
#    """
#    if not raw_text:
#        return ""
#
#    # Step 1: Normalise Unicode to NFC form
#    text = unicodedata.normalize("NFC", raw_text)
#
#    # Step 2: Remove zero-width characters and other invisibles
#    invisible_chars = [
#        "\u200b",  # zero width space
#        "\u200c",  # zero width non-joiner
#        "\u200d",  # zero width joiner
#        "\ufeff",  # byte order mark
#        "\u00ad",  # soft hyphen
#    ]
#    for char in invisible_chars:
#        text = text.replace(char, "")
#
#    # Step 3: Standardise line endings
#    text = text.replace("\r\n", "\n").replace("\r", "\n")
#
#    # Step 4: Collapse stray spaces around punctuation introduced by PDF
#    # extraction (e.g. "( Associate Strategist )" -> "(Associate Strategist)",
#    # "man-hours ( 2 weeks" -> "man-hours (2 weeks"). This happens when the
#    # source PDF has slightly wider letter-spacing around brackets/hyphens
#    # than around regular words, which our word-level extractor faithfully
#    # preserves. This is purely cosmetic cleanup, not a parsing concern, so
#    # it lives here rather than in pdf_reader.py.
#    text = re.sub(r"\(\s+", "(", text)
#    text = re.sub(r"\s+\)", ")", text)
#
#    # Step 5: Strip each line individually
#    lines = [line.strip() for line in text.split("\n")]
#
#    # Step 6: Collapse more than 2 consecutive blank lines into 2
#    cleaned_lines = []
#    blank_count   = 0
#    for line in lines:
#        if line == "":
#            blank_count += 1
#            if blank_count <= 2:
#                cleaned_lines.append(line)
#        else:
#            blank_count = 0
#            cleaned_lines.append(line)
#
#    return "\n".join(cleaned_lines).strip()
#
#
#def read_resume_file(file_path: str) -> str:
#    """
#    Master entry point for Layer 0.
#    Detects file type and routes to the right reader.
#    Returns clean normalised text.
#    """
#    from pathlib import Path
#    path = Path(file_path)
#    ext  = path.suffix.lower()
#
#    if ext == ".pdf":
#        from ingestion.pdf_reader import read_pdf
#        raw_text = read_pdf(file_path)
#    elif ext in [".docx", ".doc"]:
#        from ingestion.docx_reader import read_docx
#        raw_text = read_docx(file_path)
#    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
#        from ingestion.ocr_reader import read_with_surya
#        raw_text = read_with_surya(file_path)
#    else:
#        raise ValueError(
#            f"Unsupported file type: {ext}. Supported: PDF, DOCX, JPG, PNG, TIFF"
#        )
#
#    return normalise_text(raw_text)
#