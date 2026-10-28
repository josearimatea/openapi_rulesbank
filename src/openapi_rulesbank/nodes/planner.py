# src/nodes/planner.py

"""
Planner Node: analyzes the parsed 3GPP sections from the Reader and produces
an extraction plan that guides the Extractor Node.

LLM call: yes (1 call). No RAG.

HOW IT WORKS:
    Receives the stage 3 sections (the service's RESTful solution set) and builds a
    compact row per section (section_id | title | path | first 500 chars | tables
    | signals) to avoid sending the full document text. The LLM reasons over these
    rows, the rule type definitions and the start of the OpenAPI overview, and
    decides EVERY section: include it (with priority and extraction focus) or not
    (with the reason).

    The answer schema has one required decision field per section offered
    (planner_decisions_schema), so no section can be left undecided; the
    decisions are turned into the ExtractionPlan the state carries.

State reads:
    parsed_sections           (list[dict]) — sections from reader_node
    helper_context            (str)        — auxiliary 3GPP documents context
    openapi_reference_context (str)        — first 3000 chars used as OpenAPI overview
    service_contract          (dict)       — scope contract from contract_builder_node;
                                             strongly informs which sections are in scope

State writes:
    extraction_plan (dict) — serialized ExtractionPlan containing:
        document_summary       (str)        — brief summary of the document's purpose
        sections_to_extract    (list[dict]) — each entry has:
            section_id         (str)  — matches a section_id from parsed_sections
            title              (str)  — section title
            priority           (str)  — "high" | "medium" | "low"
            extraction_focus   (str)  — what the Extractor should focus on
"""

import re

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.llm_config import llm
from openapi_rulesbank.config.settings import PLANNER_SECTION_SIGNALS, PLANNER_TABLE_CAPTIONS
from openapi_rulesbank.graph.state import RuleBankState
from openapi_rulesbank.prompts.planner_prompts import (
    ExtractionPlan, SectionPlan, planner_decisions_schema, planner_prompt,
)
from openapi_rulesbank.prompts.rule_type_definitions import format_section_signals
from openapi_rulesbank.schemas.contract import format_contract_summary
from openapi_rulesbank.utils.parsers import is_content_empty

logger = get_logger(__name__)

# Number of characters from each section's content sent to the Planner.
# The Planner only needs a preview to decide relevance — full text goes to Extractor.
_SECTION_PREVIEW_CHARS = 500

# Number of characters from the OpenAPI spec sent as overview context.
# The full spec (296k chars) would exceed token limits — the intro is enough here.
_OPENAPI_OVERVIEW_CHARS = 3000


def _stage3_sections(sections: list[dict], service_contract: dict) -> list[dict]:
    """
    Keeps only the sections in the target service's stage 3 block.

    Stage 3 (the RESTful HTTP solution set) is the sole source of OpenAPI rules;
    stage 2 and administrative sections are not. The ContractBuilder's locator
    already identified the block as `stage3_clause` (the RESTful HTTP solution-set
    clause itself) and used the
    same prefix test to compute the contract's out-of-scope signals — we reuse it
    here so the Planner reasons over a few dozen relevant sections instead of the
    whole document. A section is in stage 3 when its clause number equals
    stage3_clause or starts with it followed by a dot.

    Returns all sections unchanged when the contract has no stage3_clause (the
    ContractBuilder was skipped or found no stage 3), so the Planner is never
    starved.
    """
    stage3_clause = service_contract.get("stage3_clause", "")
    if not stage3_clause:
        return sections

    prefix = stage3_clause + "."
    in_scope = [
        s for s in sections
        if (clause := s["title"].strip().split(" ", 1)[0]) == stage3_clause
        or clause.startswith(prefix)
    ]
    # Drop umbrella clauses whose body is only markdown markers ("###") with no
    # prose: they define nothing, and offering them to the Planner lets it pick one
    # (non-deterministically) as an extraction target, after which the Extractor
    # mines the RAG context and invents rules for an empty section. This is a fact
    # about the text, not a relevance call. The sections stay in the document (the
    # locator already ran); they are only withheld from the Planner's choice.
    kept = [s for s in in_scope if not is_content_empty(s["content"])]
    dropped_empty = len(in_scope) - len(kept)
    logger.info(
        f"Planner scope: {len(kept)} of {len(sections)} section(s) are extractable "
        f"stage 3 content (clause {stage3_clause}.x); dropped {dropped_empty} "
        f"empty umbrella clause(s) and the out-of-scope / stage 2 rest."
    )
    return kept


# A table caption, "Table <number>: <caption>", possibly in bold and wrapped over
# several lines up to the blank line that ends it.
_TABLE_CAPTION = re.compile(r"^\**(Table\s+[0-9A-Z][\w.\-]*:.*?)\**\s*(?:\n\s*\n|\Z)", re.M | re.S)


_BORDER = re.compile(r"\s*[\-=+:|*]{3,}[\s\-=+:|*]*")
_GRID_HEAD_END = re.compile(r"\s*\+[=:+]+\+?\s*")      # "+=====+===+" closes a grid header
_SIMPLE_COLUMNS = re.compile(r"\s*-+(?:\s+-+)+\s*")    # "------ -----" marks simple-table columns


def _table_lines(block: str) -> list[str]:
    """The lines of the table that opens `block`; the prose after it is not the table."""
    started, table = False, []
    for l in block.splitlines():
        if _BORDER.fullmatch(l) and started and table and not re.search(r"\w", table[-1]):
            break
        if l.strip():
            started = True
            table.append(l)
        elif started and table and _BORDER.fullmatch(table[-1]):
            break  # blank line right after a border: the table is over
    return table


def _is_empty_table(table: list[str]) -> bool:
    """A table whose rows after the header hold only "n/a" — the way a 3GPP clause
    states that nothing is supported (e.g. "URI query parameters ... n/a")."""
    rows = [l.strip() for l in table if not re.fullmatch(r"[\s\-=+:|*]+", l)]
    first_na = next((i for i, l in enumerate(rows) if "n/a" in l.lower()), None)
    if first_na is None:
        return False
    return all(re.fullmatch(r"(?:\s|\||n/a)+", l, re.I) for l in rows[first_na:])


def _table_header(table: list[str]) -> list[str]:
    """The column names of a table, copied from its header row(s).

    Two layouts come out of the .docx conversion. A grid table ('+---+' borders,
    '|' cells) has its header above the '+===+' line. A simple table has it above
    the line of dash groups ('---- -----'), whose groups give each column's
    extent — so a name wrapped over two lines ("HTTP" / "Method") is put back
    together. No header found: [] (the caption is still given).
    """
    clean = lambda t: " ".join(t.replace("*", "").split())
    if table and table[0].lstrip().startswith("+"):
        end = next((i for i, l in enumerate(table) if _GRID_HEAD_END.fullmatch(l)), None)
        if end is None:
            return []
        cols: list[str] = []
        for l in table[1:end]:
            for i, cell in enumerate(l.strip().strip("|").split("|")):
                cols[i:i + 1] = [((cols[i] + " ") if i < len(cols) else "") + cell]
        return [c for c in map(clean, cols) if c]
    sep = next((i for i, l in enumerate(table) if _SIMPLE_COLUMNS.fullmatch(l)), None)
    if not sep:
        return []
    spans = [m.span() for m in re.finditer(r"-+", table[sep])]
    head = [l for l in table[:sep] if not _BORDER.fullmatch(l)]
    cols = []
    for j, (start, _end) in enumerate(spans):
        stop = spans[j + 1][0] if j + 1 < len(spans) else None
        cols.append(clean(" ".join(l[start:stop] for l in head)))
    return [c for c in cols if c]


def _table_captions(content: str) -> list[str]:
    """The tables a section holds, one line each: caption, columns, empty mark.

    The preview is only the first N characters, and a section's decisive table is
    often further down: a method clause opening with an empty query-parameter
    table looks like it holds nothing, while its request- and response-body tables
    come after the cut. Each table is named with its caption and its column names,
    wherever it is, and a table that lists nothing ("n/a" rows only) is marked.
    All of it copied from the text, nothing interpreted.
    """
    out = []
    matches = list(_TABLE_CAPTION.finditer(content))
    for i, m in enumerate(matches):
        block = content[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(content)]
        table = _table_lines(block)
        caption = " ".join(m.group(1).replace("*", "").split())
        header = _table_header(table)
        columns = f" [columns: {' · '.join(header)}]" if header else ""
        out.append(caption + columns + (" (empty: n/a)" if _is_empty_table(table) else ""))
    return out


def _heading_path(section: dict, by_clause: dict[str, str], depth: int = 3) -> str:
    """The titles above a section, nearest `depth` levels, then its own — so a
    clause titled only "POST" or "URI" carries the resource it belongs to."""
    def clean(title: str) -> str:
        t = re.sub(r"\s*\{#[^}]*\}", "", title)          # pandoc heading attributes
        t = t.replace("\\", "").strip()
        return t.split(" ", 1)[1] if " " in t else t     # drop the clause number
    clause = section["title"].strip().split(" ", 1)[0]
    parts = clause.split(".")
    above = [by_clause[p] for p in (".".join(parts[:k]) for k in range(len(parts) - 1, 0, -1))
             if p in by_clause][:depth]
    return " › ".join([clean(t) for t in reversed(above)] + [clean(section["title"])])


def _build_sections_summary(sections: list[dict], all_sections: list[dict] | None = None) -> str:
    """
    Formats parsed_sections into a compact text table for the prompt.
    Each row: section_id | title | heading path | first N chars of content | the
    tables (caption, columns, empty mark) | the signals of each rule type the
    whole text shows (read by code with the vocabulary of the rule type
    definitions). The heading path is built from `all_sections` (the whole
    document), so parents the Planner is not offered still name the section.
    """
    by_clause = {s["title"].strip().split(" ", 1)[0]: s["title"] for s in (all_sections or sections)}
    rows = []
    for s in sections:
        preview = s["content"][:_SECTION_PREVIEW_CHARS].replace("\n", " ")
        path = _heading_path(s, by_clause)
        captions = _table_captions(s["content"]) if PLANNER_TABLE_CAPTIONS else []
        tables = f" | tables: {'; '.join(captions)}" if captions else ""
        # DELICATE — pattern-matched hints that can be false positives; switchable
        # in settings (PLANNER_SECTION_SIGNALS), see the note there.
        signals = (f" | signals: {format_section_signals(s['content'])}"
                   if PLANNER_SECTION_SIGNALS else "")
        rows.append(f"[{s['section_id']}] {s['title']} | path: {path} | {preview}{tables}{signals}")
    return "\n".join(rows)


def planner_node(state: RuleBankState) -> dict:
    """Reasons over parsed sections and produces a structured extraction plan."""
    logger.info("Planner Node started.")

    service_contract    = state.get("service_contract", {})
    stage3_sections     = _stage3_sections(state["parsed_sections"], service_contract)

    sections_summary    = _build_sections_summary(stage3_sections, state["parsed_sections"])
    openapi_overview    = state["openapi_reference_context"][:_OPENAPI_OVERVIEW_CHARS]
    helper_context      = state.get("helper_context", "") or "No auxiliary context provided."
    contract_summary    = (
        format_contract_summary(service_contract)
        or "No service contract provided."
    )

    logger.debug(f"Sending {len(stage3_sections)} section(s) to Planner LLM.")

    # The answer schema has one required decision per section offered, so no
    # section can be left without one (see planner_decisions_schema). The ids come
    # from the schema's own field names, so the answer cannot name a section that
    # was not offered either.
    structured_llm = llm.with_structured_output(planner_decisions_schema(stage3_sections))
    chain = planner_prompt | structured_llm
    inputs = {
        "sections_summary":    sections_summary,
        "openapi_reference_overview": openapi_overview,
        "helper_context":      helper_context,
        "contract_summary":    contract_summary,
    }
    try:
        answer = chain.invoke(inputs)
    except Exception as e:
        # An answer missing a required decision fails validation; ask once more
        # rather than plan with a section silently dropped.
        logger.warning(f"Planner answer invalid ({str(e)[:200]}) — asking once more.")
        answer = chain.invoke(inputs)

    # Decisions back into the plan the rest of the pipeline reads, in document
    # order: section_id is the sequential document index, so the Extractor handles
    # a clause that declares something (e.g. a 'General' clause listing imported
    # types) before the clauses that use it.
    plan_sections, reasons = [], {}
    for s in stage3_sections:
        d = getattr(answer, f"s{s['section_id']}")
        if d.include:
            plan_sections.append(SectionPlan(
                section_id=s["section_id"], title=s["title"], priority=d.priority or "low",
                extraction_focus=d.extraction_focus, notes=d.notes))
        else:
            reasons[s["section_id"]] = d.reason
            logger.info(f"[planner] excluded [{s['section_id']}] {s['title'][:60]} — {d.reason}")
    plan_sections.sort(key=lambda p: int(p.section_id) if p.section_id.isdigit() else 1_000_000)
    plan = ExtractionPlan(document_summary=answer.document_summary,
                          sections_to_extract=plan_sections)

    selected_ids = {s.section_id for s in plan_sections}

    # Sections the Planner did not select — computed here so the accounting
    # is always accurate regardless of test limitations applied outside the node.
    excluded_sections_planner = [
        {"section_id": s["section_id"], "title": s["title"],
         **({"reason": reasons[s["section_id"]]} if s["section_id"] in reasons else {})}
        for s in state["parsed_sections"]
        if s["section_id"] not in selected_ids
    ]

    selected = len(plan_sections)
    excluded = len(excluded_sections_planner)
    logger.info(
        f"Planner Node complete — {selected} section(s) selected, {excluded} excluded."
    )
    logger.debug(f"Document summary: {plan.document_summary}")

    return {
        "extraction_plan":          plan.model_dump(),
        "excluded_sections_planner": excluded_sections_planner,
    }
