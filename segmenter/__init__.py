"""
segmenter/__init__.py
Makes the segmenter folder importable as a package, and exposes the
main entry points so other code can do:
    from segmenter import split_into_sections, get_section_text
"""

from segmenter.section_splitter import split_into_sections, get_section_text, Section
