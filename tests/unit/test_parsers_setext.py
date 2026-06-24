# tests/unit/test_parsers_setext.py

"""
Unit tests for parse_sections() heading handling — no external deps.

Verifies that both ATX ('## ') and setext ('Title\\n----') level-2 headings are
recognized, so .md produced by either old (setext) or new (ATX) pandoc parses
the same way.
"""

from openapi_rulesbank.utils.parsers import parse_sections


def test_atx_headings_no_regression():
    md = "## Intro\nfoo bar baz\n\n## Scope\nhello world\n"
    kept, _ = parse_sections(md)
    assert [s["title"] for s in kept] == ["Intro", "Scope"]


def test_setext_level2_headings_recognized():
    md = "Intro\n-----\nfoo bar baz\n\nDefinitions\n------\nmore text here\n"
    kept, _ = parse_sections(md)
    assert [s["title"] for s in kept] == ["Intro", "Definitions"]


def test_setext_level1_not_normalized_to_section_split():
    # '====' is a level-1 heading; it must NOT be normalized to a '## ' split.
    # Its body stays attached to its own segment (the level-1 line is kept verbatim,
    # not rewritten to ATX), while the level-2 'Sub' below it IS split out.
    md = "Title\n=====\nbig title body here\n\nSub\n---\nsection body here\n"
    kept, _ = parse_sections(md)
    sub = next(s for s in kept if s["title"] == "Sub")
    assert sub["content"] == "section body here"
    # The level-1 heading line is preserved (not turned into a '## ' split point).
    first = kept[0]
    assert first["content"].startswith("=====")
