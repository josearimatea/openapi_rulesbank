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


def format_section_verdict(verdict: dict | None) -> str:
    """Render one section's pre-analysis verdict for a Reflector/Validator prompt.

    verdict is section_analysis_by_id[section_id] = {section_kind,
    emits_schema_properties, base_ref, inherited_fields}, or None when the
    pre-analyzer produced nothing for the section. Kept to a couple of lines so the
    prompt context stays flat — only the current section's entry is ever passed.
    """
    if not verdict:
        return "No pre-analysis verdict for this section."
    home = verdict.get("emits_schema_properties", False)
    lines = [
        f"section_kind: {verdict.get('section_kind', 'unknown')}",
        f"is_schema_home: {home}",
    ]
    if verdict.get("base_ref"):
        lines.append(f"embedded base: {verdict['base_ref']}")
        lines.append(f"inherited fields (not own): {verdict.get('inherited_fields', [])}")

    # What this clause maps, and where its callbacks hang. Both were DECIDED in
    # the pre-analysis, from the sibling clauses this section cannot see: the
    # section's own text names the delivery method and never the owning
    # operation. A judge given only the section content therefore reads a correct
    # anchor as wrong and "corrects" it to the method the text does mention.
    op, path, method = (verdict.get("maps_operation", ""),
                        verdict.get("maps_path", ""),
                        verdict.get("maps_method", ""))
    if op and path and method:
        lines.append(f"this clause maps: {op} -> {method} {path}")
    if verdict.get("callback_anchor"):
        lines.append(
            f"callback anchor (decided in pre-analysis, source: "
            f"{verdict.get('callback_anchor_source', '?')}): "
            f"{verdict['callback_anchor']}"
        )
        lines.append(
            "  Callback rules of this section MUST hang off that anchor: "
            "\"<anchor>.callbacks.<notificationName>\". It was resolved from "
            "clauses this section does not contain, so do NOT re-derive it from "
            "this section's text and do NOT rewrite its method to the one the "
            "section mentions — the section describes how the notification is "
            "DELIVERED (POST), while the anchor is the operation through which "
            "the consumer SUPPLIED its address."
        )
    elif verdict.get("callback_anchor_source") == "unresolved":
        lines.append(
            "callback anchor: UNRESOLVED — the pre-analysis could not determine "
            "it. A callback rule here has no verified anchor; treat a missing one "
            "as a reported gap, not as a rule to invent an anchor for."
        )

    # Types whose $ref this section could not write yet (origin not stated, or
    # external with no filename). Their rules were HELD ON PURPOSE — kept, and
    # emitted by the section that settles the type. Not missing, not wrong.
    deferred = verdict.get("defer_external_refs", []) or []
    if deferred:
        lines.append(
            f"held $refs (withheld on purpose, NOT missing): {deferred}. A rule "
            "whose $ref points at one of these types was held, not emitted here: "
            "where the type is defined is not settled yet. It is kept and emitted by "
            "the section that settles it. Do NOT report such a rule as missing and "
            "do NOT correct this section for its absence. Conversely, a rule that DOES "
            "appear among this section's rules with a $ref to one of these types "
            "(unless listed below as a held rule completed here) was emitted when it "
            "should have been held: its $ref form asserts an origin nothing states."
        )

    # The final review of the rules this section held and no section settled. They
    # are judged once here; holding them again would only send them back to wait.
    review = verdict.get("held_review")
    if review:
        lines.append(
            f"FINAL REVIEW OF HELD RULES (types never settled: {review}). The rules "
            "judged now were written by this section and held because where their "
            "$ref type is defined was not stated — and no section of the document "
            "stated it. This is their review, and they are not held again: never "
            "instruct to move one to deferred_rules. Judge each as any rule of this "
            "section. If it must not exist — an inherited field the allOf base "
            "already provides, a rule another section owns, a duplicate of a rule "
            "already validated — discard it and say why. Otherwise correct what is "
            "wrong in it. Where its $ref type is defined is still unstated: do not "
            "invent an origin, and do not fail the rule for that alone — the $ref "
            "stands as written unless a source you were given states the type's origin."
        )

    # Held rules emitted HERE: written in an earlier section, emitted now because
    # this section settled where their type is defined. They are not in this
    # section's text, and that is expected.
    carried = verdict.get("completed_held_rules", []) or []
    if carried:
        items = "; ".join(
            f"{c.get('rule_type')} {c.get('source_name')!r} "
            f"(object={c.get('openapi_object', '?')} field={c.get('openapi_field', '?')}) "
            f"written in section {c.get('from_section')} -> {c.get('type')} settled "
            f"as {c.get('settled_as')}"
            for c in carried
        )
        lines.append(
            f"held rules completed and emitted in this section: {items}. Each was "
            "written by the section named, held there because the origin of its $ref "
            "type was not stated, and emitted ONCE here, by the code, because this "
            "section states it; it now carries this section's id. Its type is "
            "SETTLED here, so it is never held again. Judge it by the earlier "
            "section's statement and this one's settling of the type — do NOT reject "
            "it for not appearing in this section's text."
        )

    # Filenames built by the naming convention (no source gave them).
    fallback = verdict.get("fallback_filenames", {}) or {}
    if fallback:
        lines.append(
            f"filenames built by convention (last resort): {fallback}. No source — "
            "user auxiliary context, the document's annex, the corpus — gave the YAML "
            "file of these types, so it was built as TS<spec>_ComDefs.yaml. A rule "
            "whose $ref uses one of them is scored at the acceptance floor (0.5), "
            "flagged, and its reasoning says the lookup found nothing and the "
            "convention was used."
        )
    return "\n".join(lines)


def build_sibling_summary(
    section_title: str,
    parsed_sections: list[dict],
    analysis_by_id: dict,
    exclude_id: str,
    levels: int = 3,
) -> str:
    """A light summary of the sibling clauses in the same service block.

    Siblings share the section's clause prefix at `levels` (e.g. "12.1.1" groups a
    solution set's clauses). Each line is id + title + the kind already decided for
    it (from section_analysis_by_id), plus which operation it maps when it maps
    one. Titles only — no content — so a prompt can see the neighbourhood's shape
    without ballooning. Empty when there are no siblings.

    The grouping is the whole solution set, not the sub-block a clause sits in:
    an operation is mapped in one sub-block (".1.x") and the resource it acts on
    is defined in another (".3.x"), so a narrower prefix puts a clause and the
    operation it needs in different groups. That is 59 title-only lines here
    against 19, for ~3.8k characters — cheap beside the retrieved spec chunks the
    same prompt already carries.

    Shared by the Extractor (whose triage resolves an anchor from it), the
    Reflector and the Validator. The titles are the ONLY place a canonical name
    appears in full — "12.1.1.2.4 Notification notifyMOIAttributeValueChanges" —
    while a section's own data-type table may spell the same notification
    differently. A judge without this list can check that a name has the right
    SHAPE but not that it is the right NAME, and will accept whatever spelling the
    section's table happens to carry.
    """
    def prefix(title: str) -> str:
        head = title.strip().split(" ", 1)[0]
        return ".".join(head.split(".")[:levels])

    my_prefix = prefix(section_title)
    if not my_prefix:
        return ""
    lines = []
    for s in parsed_sections:
        if s["section_id"] == exclude_id:
            continue
        if prefix(s["title"]) != my_prefix:
            continue
        verdict = analysis_by_id.get(s["section_id"], {})
        kind = verdict.get("section_kind", "not yet analyzed")
        op, path, method = (verdict.get("maps_operation", ""),
                            verdict.get("maps_path", ""),
                            verdict.get("maps_method", ""))
        maps = f" — maps {op}: {method} {path}" if op and path and method else ""
        lines.append(f"  [{s['section_id']}] {s['title'][:60]} — {kind}{maps}")
    return "\n".join(lines)


def discards_for_section(validation_errors: list[dict], section_id: str) -> list[dict]:
    """The Validator's discard errors for one section, from state['validation_errors'].

    A single source of truth for "which rules were discarded in this section" — the
    Extractor (to not re-emit them), the Validator (to skip re-validating them), and
    the Reflector (to not re-list them as missing) all need the same set, so they
    all filter through here instead of repeating the predicate.
    """
    out = []
    for e in (validation_errors or []):
        if e.get("error_type") != "discard":
            continue
        rule = e.get("rule", {})
        sid = e.get("section_id", "") or rule.get("section_id", "")
        if sid == section_id:
            out.append(e)
    return out


def format_rule(rule: dict) -> str:
    """One rule as a judge must see it whenever it compares against it: whole.

    Every list of rules a judge reads — the section's other rules, the ones already
    validated, the discarded and the corrected — goes through here. A summary cut
    to a few words, or one without the object, hides where the rule points: a judge
    shown "POST" without its path took an existing path operation for a missing one.
    """
    m = rule.get("openapi_mapping", {}) or {}
    return (
        f"  [{rule.get('rule_type', '?')}] source_name={rule.get('source_name', '')!r}\n"
        f"    rule_text : {rule.get('rule_text', '')}\n"
        f"    object    : {m.get('openapi_object', '')}\n"
        f"    field     : {m.get('openapi_field', '')}\n"
        f"    value     : {m.get('openapi_value', '')}\n"
        f"    references: {format_references(m)}"
    )


def format_rules(rules: list[dict], empty: str = "None") -> str:
    """Several rules, each whole (format_rule); `empty` when there are none."""
    return "\n".join(format_rule(r) for r in (rules or [])) or empty


def format_discards(discards: list[dict]) -> str:
    """Render discard errors for a prompt: the whole rule and the whole reason."""
    if not discards:
        return "None"
    return "\n".join(
        f"{format_rule(e.get('rule', {}))}\n"
        f"    DISCARDED : {e.get('instruction', 'discarded')}"
        for e in discards
    )


def corrections_for_section(validation_errors: list[dict], section_id: str) -> list[dict]:
    """The Validator's correction errors for one section, from validation_errors.

    A correction means the rule EXISTS but needs a fix — so its target must not be
    listed as missing. The missing-side phases (Reflector Fase 2, Validator Stage 3)
    use this to avoid duplicating a target as both correction and missing.
    """
    out = []
    for e in (validation_errors or []):
        if e.get("error_type") != "correction":
            continue
        rule = e.get("rule", {})
        sid = e.get("section_id", "") or rule.get("section_id", "")
        if sid == section_id:
            out.append(e)
    return out


def format_references(mapping: dict) -> str:
    """Render a rule's `references` for a prompt: "file#schema" per entry, or none."""
    refs = (mapping or {}).get("references") or []
    return ", ".join(
        f"{r.get('file') or '(this document)'}#{r.get('schema_name', '')}" for r in refs
    ) or "none"


def format_corrections(corrections: list[dict]) -> str:
    """Render correction errors for a prompt: the whole rule and the whole fix asked."""
    if not corrections:
        return "None"
    return "\n".join(
        f"{format_rule(e.get('rule', {}))}\n"
        f"    BEING CORRECTED: {e.get('instruction', '')}"
        for e in corrections
    )


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
        # The content IS kept on this excluded entry (only for the annex), so the
        # reader can consult it for ONE narrow purpose: resolving the YAML filename
        # of an external $ref whose type is already known to be imported. It is
        # never fed back into extraction — excluded sections do not reach the
        # Planner — so the "do not derive rules from the annex" rule stands.
        if _is_embedded_openapi_annex(title, content):
            excluded.append({"section_id": str(i), "title": title,
                             "reason": "openapi_annex", "content": content})
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
