# src/utils/parsers.py

"""
Parsing utilities specific to 3GPP specification documents.

These functions were originally in nodes/reader.py. They are moved here so
that parsing logic is reusable and testable independently from the node.

Functions:
    parse_sections(md_text) — splits a Markdown document into filtered section dicts
"""

import re
from openapi_rulesbank.config.settings import FILTER_SYMBOLIC_TITLES, FILTER_BY_KEYWORDS

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


def is_content_empty(content: str) -> bool:
    """True when a section body carries no prose of its own.

    An umbrella clause is split from its children, so its body is left holding
    only their markdown markers ("#", "###") or nothing. Such a section states
    nothing to extract. This is a FACT about the text (no alphanumeric content),
    not a relevance judgement — used to keep the Planner from selecting these
    sections and the Extractor from mining the RAG context for them. It does NOT
    remove the section from the document (the ContractBuilder's locator still sees
    every heading); it only guards selection and extraction downstream.
    """
    stripped = content.strip()
    return not stripped or not re.search(r'[a-zA-Z0-9]', stripped.replace("#", ""))


def _is_embedded_openapi_annex(title: str, content: str) -> bool:
    """
    Returns True for an annex section that embeds a finished OpenAPI document.

    3GPP specs append the resulting YAML in an annex ("OpenAPI document
    <name>.yaml", body starting with `openapi: <version>`). Rules must be derived
    from the specification text, not copied from that answer — and for a release
    whose YAML does not exist yet the annex is absent anyway.
    """
    body = content.lstrip()
    return "openapi document" in title.lower() and bool(
        re.match(r'openapi:\s*\d', body)
    )


# Setext level-2 heading: a non-empty text line immediately followed by a line
# of dashes (----). Older pandoc versions (used to build the original TSpec .md
# corpus) emit setext headings, while newer pandoc emits ATX (## ). We normalize
# setext level-2 headings to ATX so parse_sections() works for both, without
# altering the .md on disk.
_SETEXT_H2_RE = re.compile(r'^(?P<title>[^\n]+)\n-{2,}[ \t]*$', re.MULTILINE)


def _normalize_setext_headings(md_text: str) -> str:
    """Rewrites setext level-2 headings ('Title\\n----') as ATX ('## Title')."""
    return _SETEXT_H2_RE.sub(lambda m: f"## {m.group('title').strip()}", md_text)


def parse_sections(md_text: str) -> tuple[list[dict], list[dict]]:
    """
    Splits a Markdown document into sections and separates relevant from excluded.

    HOW IT WORKS:
        0. Normalizes setext level-2 headings ('Title\n----') to ATX ('## Title')
           so documents produced by older pandoc (setext) parse the same as those
           produced by newer pandoc (ATX). See _normalize_setext_headings().
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
    md_text = _normalize_setext_headings(md_text)
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

        # Heading-only sections (a parent clause whose body is just its children's
        # markdown markers) are deliberately KEPT: the umbrella clause that names
        # each Management Service is one of them, and the ContractBuilder's locator
        # needs it to find the target service's block. Dropping them made the
        # locator fall back to another service's clause entirely.

        # Discard the annex embedding the finished OpenAPI document. Extracting
        # from it would copy the answer instead of deriving it from the spec — and
        # a new release has no such annex until the YAML exists.
        if _is_embedded_openapi_annex(title, content):
            excluded.append({"section_id": str(i), "title": title, "reason": "openapi_annex"})
            continue

        # Optionally filter by keywords (disabled by default — see settings.py)
        if FILTER_BY_KEYWORDS and not any(kw in section.lower() for kw in RELEVANT_KEYWORDS):
            excluded.append({"section_id": str(i), "title": title, "reason": "keyword_filter"})
            continue

        # Number the kept sections consecutively. Using the pre-filter index would
        # leave gaps wherever a section was dropped (1, 2, 4, 7, ...); the LLM stops
        # reading such a column as an identifier and answers with the clause number
        # from the title instead, which matches nothing downstream.
        sections.append({
            "section_id": str(len(sections) + 1),
            "title": title,
            "content": content,
        })

    return sections, excluded
