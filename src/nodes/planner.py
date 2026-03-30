# src/nodes/planner.py

"""
Planner Node: analyzes the parsed 3GPP sections from the Reader and produces
an extraction plan that guides the Extractor Node.

This is the first LLM call in the pipeline. It performs structured reasoning
over section titles and content previews — not full section text — to keep
the token budget manageable. Full section text is passed to the Extractor.

State reads:
    parsed_sections      (list[dict]) — sections from reader_node
    helper_context       (str)        — auxiliary 3GPP documents context
    openapi_spec_context (str)        — local OpenAPI spec snapshot

State writes:
    extraction_plan      (dict)       — serialized ExtractionPlan with
                                        document_summary and sections_to_extract
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
    openapi_overview    = state["openapi_spec_context"][:_OPENAPI_OVERVIEW_CHARS]
    helper_context      = state.get("helper_context", "") or "No auxiliary context provided."

    logger.debug(f"Sending {len(state['parsed_sections'])} section(s) to Planner LLM.")

    # Bind structured output so the LLM returns a validated ExtractionPlan object
    structured_llm = llm.with_structured_output(ExtractionPlan)
    chain = planner_prompt | structured_llm

    plan: ExtractionPlan = chain.invoke({
        "sections_summary":    sections_summary,
        "openapi_spec_overview": openapi_overview,
        "helper_context":      helper_context,
    })

    selected = len(plan.sections_to_extract)
    logger.info(f"Planner Node complete — {selected} section(s) selected for extraction.")
    logger.debug(f"Document summary: {plan.document_summary}")

    # Serialize to dict so LangGraph can merge it into the state cleanly
    return {
        "extraction_plan": plan.model_dump(),
    }
