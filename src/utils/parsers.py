# src/utils/parsers.py

"""
Parsing utilities specific to 3GPP specification documents.

These functions were originally in nodes/reader.py. They are moved here so
that parsing logic is reusable and testable independently from the node.

Functions:
    parse_sections(md_text) — splits a Markdown document into filtered section dicts
"""

import re
from config.settings import FILTER_SYMBOLIC_TITLES

# Keywords used to filter sections relevant to OpenAPI rule extraction.
# Sections whose text (title + content) does not contain any of these keywords
# are discarded — they are unlikely to contain mappable OpenAPI rules.
RELEVANT_KEYWORDS = [
    'template', 'nrm', 'mapping', 'stage',
    'json', 'yang', 'openapi', 'attribute', 'class'
]


def _has_real_words(title: str) -> bool:
    """
    Returns True if the title contains at least one word with 3+ letters.
    Used to discard symbolic titles like '+---+---+' or '====' that appear
    in 3GPP document cover-page tables.
    Controlled by FILTER_SYMBOLIC_TITLES in settings.py.
    """
    return bool(re.search(r'[a-zA-Z]{3,}', title))


def parse_sections(md_text: str) -> list[dict]:
    """
    Splits a Markdown document into sections and returns only those relevant
    to OpenAPI rule extraction.

    HOW IT WORKS:
        1. Splits the full document text on '## ' (level-2 headers), which is
           the standard header level used for sections in 3GPP Markdown specs.
        2. For each section, separates the first line (title) from the rest (content).
        3. If FILTER_SYMBOLIC_TITLES is True, discards sections whose title contains
           no real words (e.g. cover-page table borders like '+------+---+').
        4. Filters out sections that contain none of the RELEVANT_KEYWORDS.
        5. Returns the surviving sections as a list of dicts.

    Returns:
        list[dict] — each dict has:
            section_id (str) — sequential index of the section in the document
            title      (str) — the section header text (first line after '## ')
            content    (str) — the body text of the section
    """
    raw_sections = [s.strip() for s in md_text.split('## ') if s.strip()]

    sections = []
    for i, section in enumerate(raw_sections):
        lines = section.splitlines()

        # First line is the section title (the text that followed '## ')
        title = lines[0].strip()

        # Everything after the title is the section body
        content = "\n".join(lines[1:]).strip()

        # Discard sections with symbolic/non-word titles (e.g. cover-page tables)
        if FILTER_SYMBOLIC_TITLES and not _has_real_words(title):
            continue

        # Keep only sections relevant to OpenAPI rule extraction
        if any(kw in section.lower() for kw in RELEVANT_KEYWORDS):
            sections.append({
                "section_id": str(i),
                "title": title,
                "content": content,
            })

    return sections
