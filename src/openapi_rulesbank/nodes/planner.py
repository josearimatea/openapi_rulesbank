# src/nodes/planner.py

"""
Planner Node: analyzes the parsed 3GPP sections from the Reader and produces
an extraction plan that guides the Extractor Node.

LLM call: yes (1 call). No RAG.

HOW IT WORKS:
    Receives all sections parsed by the Reader and builds a compact summary
    table (section_id | title | first 200 chars of content) to avoid sending
    the full document text. The LLM reasons over this table and the first
    3000 chars of the OpenAPI spec overview to decide which sections are
    relevant, their extraction priority (high/medium/low), and the extraction
    focus for each section.

    Uses with_structured_output(ExtractionPlan) to return a validated Pydantic
    object serialized to dict for the state.

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

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.llm_config import llm
from openapi_rulesbank.graph.state import RuleBankState
from openapi_rulesbank.prompts.planner_prompts import ExtractionPlan, planner_prompt
from openapi_rulesbank.schemas.contract import format_contract_summary

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
    already identified the block as `stage3_clause` (e.g. "12.3") and used the
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
    kept = [
        s for s in sections
        if (clause := s["title"].strip().split(" ", 1)[0]) == stage3_clause
        or clause.startswith(prefix)
    ]
    logger.info(
        f"Planner scope: {len(kept)} of {len(sections)} section(s) are in the "
        f"stage 3 block (clause {stage3_clause}.x); the rest are stage 2 or "
        f"other-service sections and are not sent to the LLM."
    )
    return kept


def _build_sections_summary(sections: list[dict]) -> str:
    """
    Formats parsed_sections into a compact text table for the prompt.
    Each row: section_id | title | first N chars of content.
    """
    rows = []
    for s in sections:
        preview = s["content"][:_SECTION_PREVIEW_CHARS].replace("\n", " ")
        rows.append(f"[{s['section_id']}] {s['title']} | {preview}")
    return "\n".join(rows)


def planner_node(state: RuleBankState) -> dict:
    """Reasons over parsed sections and produces a structured extraction plan."""
    logger.info("Planner Node started.")

    service_contract    = state.get("service_contract", {})
    stage3_sections     = _stage3_sections(state["parsed_sections"], service_contract)

    sections_summary    = _build_sections_summary(stage3_sections)
    openapi_overview    = state["openapi_reference_context"][:_OPENAPI_OVERVIEW_CHARS]
    helper_context      = state.get("helper_context", "") or "No auxiliary context provided."
    contract_summary    = (
        format_contract_summary(service_contract)
        or "No service contract provided."
    )

    logger.debug(f"Sending {len(stage3_sections)} section(s) to Planner LLM.")

    # Bind structured output so the LLM returns a validated ExtractionPlan object
    structured_llm = llm.with_structured_output(ExtractionPlan)
    chain = planner_prompt | structured_llm

    plan: ExtractionPlan = chain.invoke({
        "sections_summary":    sections_summary,
        "openapi_reference_overview": openapi_overview,
        "helper_context":      helper_context,
        "contract_summary":    contract_summary,
    })

    # Keep only plan entries whose section_id was actually sent to the LLM (the
    # stage 3 subset). The LLM occasionally returns an id it never saw — a
    # hallucination, a duplicate, or a stage 2 id — and counting those would break
    # the Builder's section accounting (planned + excluded_planner must equal
    # parsed_sections) and point the Extractor outside the intended scope.
    input_ids = {s["section_id"] for s in stage3_sections}
    valid_plan_sections = [
        s for s in plan.sections_to_extract if s.section_id in input_ids
    ]
    dropped = len(plan.sections_to_extract) - len(valid_plan_sections)
    if dropped:
        logger.warning(
            f"Planner returned {dropped} section id(s) not in the stage 3 subset — dropped."
        )
    plan.sections_to_extract = valid_plan_sections

    selected_ids = {s.section_id for s in valid_plan_sections}

    # Sections the Planner did not select — computed here so the accounting
    # is always accurate regardless of test limitations applied outside the node.
    excluded_sections_planner = [
        {"section_id": s["section_id"], "title": s["title"]}
        for s in state["parsed_sections"]
        if s["section_id"] not in selected_ids
    ]

    selected = len(valid_plan_sections)
    excluded = len(excluded_sections_planner)
    logger.info(
        f"Planner Node complete — {selected} section(s) selected, {excluded} excluded."
    )
    logger.debug(f"Document summary: {plan.document_summary}")

    return {
        "extraction_plan":          plan.model_dump(),
        "excluded_sections_planner": excluded_sections_planner,
    }
