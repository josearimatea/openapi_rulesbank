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

State writes:
    extraction_plan (dict) — serialized ExtractionPlan containing:
        document_summary       (str)        — brief summary of the document's purpose
        sections_to_extract    (list[dict]) — each entry has:
            section_id         (str)  — matches a section_id from parsed_sections
            title              (str)  — section title
            priority           (str)  — "high" | "medium" | "low"
            extraction_focus   (str)  — what the Extractor should focus on
"""

from config import get_logger
from config.llm_config import llm
from graph.state import RuleBankState
from prompts.planner_prompts import ExtractionPlan, planner_prompt

logger = get_logger(__name__)

# Number of characters from each section's content sent to the Planner.
# The Planner only needs a preview to decide relevance — full text goes to Extractor.
_SECTION_PREVIEW_CHARS = 200

# Number of characters from the OpenAPI spec sent as overview context.
# The full spec (296k chars) would exceed token limits — the intro is enough here.
_OPENAPI_OVERVIEW_CHARS = 3000


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

    sections_summary    = _build_sections_summary(state["parsed_sections"])
    openapi_overview    = state["openapi_reference_context"][:_OPENAPI_OVERVIEW_CHARS]
    helper_context      = state.get("helper_context", "") or "No auxiliary context provided."

    logger.debug(f"Sending {len(state['parsed_sections'])} section(s) to Planner LLM.")

    # Bind structured output so the LLM returns a validated ExtractionPlan object
    structured_llm = llm.with_structured_output(ExtractionPlan)
    chain = planner_prompt | structured_llm

    plan: ExtractionPlan = chain.invoke({
        "sections_summary":    sections_summary,
        "openapi_reference_overview": openapi_overview,
        "helper_context":      helper_context,
    })

    selected_ids = {s.section_id for s in plan.sections_to_extract}

    # Sections the Planner did not select — computed here so the accounting
    # is always accurate regardless of test limitations applied outside the node.
    excluded_sections_planner = [
        {"section_id": s["section_id"], "title": s["title"]}
        for s in state["parsed_sections"]
        if s["section_id"] not in selected_ids
    ]

    selected = len(plan.sections_to_extract)
    excluded = len(excluded_sections_planner)
    logger.info(
        f"Planner Node complete — {selected} section(s) selected, {excluded} excluded."
    )
    logger.debug(f"Document summary: {plan.document_summary}")

    return {
        "extraction_plan":          plan.model_dump(),
        "excluded_sections_planner": excluded_sections_planner,
    }
