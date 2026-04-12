# src/utils/parsers.py

"""
Parsing utilities specific to 3GPP specification documents.

These functions were originally in nodes/reader.py. They are moved here so
that parsing logic is reusable and testable independently from the node.

Functions:
    parse_sections(md_text) — splits a Markdown document into filtered section dicts
"""

import re
from config.settings import FILTER_SYMBOLIC_TITLES, FILTER_BY_KEYWORDS

# Keywords used to filter sections relevant to OpenAPI rule extraction.
# Used only when FILTER_BY_KEYWORDS = True in settings.py.
# Organized by the rule_type each group targets. Add terms after reviewing
# the target document to avoid discarding valid sections.
RELEVANT_KEYWORDS = [
    # path_operation — HTTP method mappings to IS operations
    'operation', 'http', 'method', 'put', 'get', 'post', 'patch', 'delete',
    'resource', 'uri', 'url', 'path',
    # schema_property — NRM attribute/data model definitions
    'attribute', 'property', 'schema', 'type', 'integer', 'string', 'boolean',
    'nrm', 'class', 'yang', 'template',
    # path_parameter / query_parameter — URI variables and query string params
    'parameter', 'query', 'filter', 'scope', 'version',
    # response — HTTP response codes and payloads
    'response', 'status',
    # request_body — payload definitions
    'request', 'body', 'payload', 'content',
    # security_scheme — authentication and authorization
    'security', 'oauth', 'authentication', 'authorization', 'bearer', 'token',
    # General OpenAPI / 3GPP terms
    'openapi', 'mapping', 'stage', 'json', 'api',
]


def _has_real_words(title: str) -> bool:
    """
    Returns True if the title contains at least one word with 3+ letters.
    Used to discard symbolic titles like '+---+---+' or '====' that appear
    in 3GPP document cover-page tables.
    Controlled by FILTER_SYMBOLIC_TITLES in settings.py.
    """
    return bool(re.search(r'[a-zA-Z]{3,}', title))


def parse_sections(md_text: str) -> tuple[list[dict], list[dict]]:
    """
    Splits a Markdown document into sections and separates relevant from excluded.

    HOW IT WORKS:
        1. Splits the full document text on '## ' (level-2 headers), which is
           the standard header level used for sections in 3GPP Markdown specs.
        2. For each section, separates the first line (title) from the rest (content).
        3. If FILTER_SYMBOLIC_TITLES is True, discards sections whose title contains
           no real words (e.g. cover-page table borders like '+------+---+').
        4. If FILTER_BY_KEYWORDS is True, discards sections that contain none of the
           RELEVANT_KEYWORDS. Disabled by default — enable only after validating the
           keyword list covers all relevant terminology in the target document.
        5. Returns both the surviving sections and the excluded ones (with reasons).

    Returns:
        tuple(kept, excluded) where each element is a list[dict]:
          kept:
            section_id (str) — sequential index of the section in the document
            title      (str) — the section header text (first line after '## ')
            content    (str) — the body text of the section
          excluded:
            section_id (str) — sequential index
            title      (str) — section header text
            reason     (str) — "symbolic_title" | "keyword_filter"
    """
    raw_sections = [s.strip() for s in md_text.split('## ') if s.strip()]

    sections: list[dict] = []
    excluded: list[dict] = []

    for i, section in enumerate(raw_sections):
        lines = section.splitlines()

        # First line is the section title (the text that followed '## ')
        title = lines[0].strip()

        # Everything after the title is the section body
        content = "\n".join(lines[1:]).strip()

        # Discard sections with symbolic/non-word titles (e.g. cover-page tables)
        if FILTER_SYMBOLIC_TITLES and not _has_real_words(title):
            excluded.append({"section_id": str(i), "title": title, "reason": "symbolic_title"})
            continue

        # Optionally filter by keywords (disabled by default — see settings.py)
        if FILTER_BY_KEYWORDS and not any(kw in section.lower() for kw in RELEVANT_KEYWORDS):
            excluded.append({"section_id": str(i), "title": title, "reason": "keyword_filter"})
            continue

        sections.append({
            "section_id": str(i),
            "title": title,
            "content": content,
        })

    return sections, excluded
