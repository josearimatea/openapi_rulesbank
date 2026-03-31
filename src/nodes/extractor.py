# src/nodes/extractor.py

"""
Extractor Node: extracts raw OpenAPI rules from each 3GPP section guided
by the Planner's extraction plan.

This is the second LLM call in the pipeline. It processes one section at a
time, retrieving relevant OpenAPI reference chunks from Qdrant via RAG before
each call to keep the context focused and token-efficient.

On loop-back (Validator → Extractor), previously failed rules are injected
as feedback so the LLM can correct or replace them.

State reads:
    parsed_sections    (list[dict]) — full section content from reader_node
    extraction_plan    (dict)       — sections_to_extract from planner_node
    helper_context     (str)        — auxiliary 3GPP documents context
    validation_errors  (list[dict]) — failed rules from previous iteration (may be empty)
    iteration_count    (int)        — current loop iteration number

State writes:
    raw_rules          (list[dict]) — RawRule objects serialized as dicts
    iteration_count    (int)        — incremented by 1
"""

from config import get_logger
from config.llm_config import llm
from graph.state import RuleBankState
from prompts.extractor_prompts import SectionRules, extractor_prompt
from tools.rag_tools import search_openapi_reference

logger = get_logger(__name__)


def _build_sections_index(parsed_sections: list[dict]) -> dict[str, dict]:
    """
    Builds a lookup dict from section_id to section dict for O(1) access.
    """
    return {s["section_id"]: s for s in parsed_sections}


def _build_error_feedback(validation_errors: list[dict]) -> str:
    """
    Formats validation errors from the previous iteration into a feedback
    string to inject into the prompt on loop-back.

    Returns "" on first iteration (no errors yet).
    """
    if not validation_errors:
        return ""

    lines = ["The following rules from the previous extraction failed validation:"]
    for i, err in enumerate(validation_errors, start=1):
        rule = err.get("rule", {})
        lines.append(
            f"  {i}. [{rule.get('section_id', '?')}] {rule.get('rule_text', '?')!r} "
            f"— Reason: {err.get('reason', '?')}"
        )
    lines.append("Avoid repeating these errors in your new extraction.")
    return "\n".join(lines)


def extractor_node(state: RuleBankState) -> dict:
    """
    Iterates over sections in the extraction plan and extracts raw OpenAPI rules.

    For each section:
      1. Retrieves RAG chunks from the OpenAPI reference collection.
      2. Calls the LLM with the section content + RAG context + error feedback.
      3. Collects RawRule objects into raw_rules.
    """
    logger.info("Extractor Node started.")

    sections_index    = _build_sections_index(state["parsed_sections"])
    plan_sections     = state["extraction_plan"]["sections_to_extract"]
    helper_context    = state.get("helper_context", "") or "No auxiliary context provided."
    validation_errors = state.get("validation_errors", []) or []
    iteration_count   = state.get("iteration_count", 0) or 0
    error_feedback    = _build_error_feedback(validation_errors)

    if error_feedback:
        logger.info(
            f"Loop-back iteration {iteration_count + 1} — "
            f"{len(validation_errors)} error(s) fed back to Extractor."
        )

    structured_llm = llm.with_structured_output(SectionRules)
    chain = extractor_prompt | structured_llm

    raw_rules: list[dict] = []

    for plan_section in plan_sections:
        section_id    = plan_section["section_id"]
        section_title = plan_section["title"]
        focus         = plan_section["extraction_focus"]

        section = sections_index.get(section_id)
        if section is None:
            logger.warning(
                f"Section '{section_id}' in plan not found in parsed_sections — skipping."
            )
            continue

        # Retrieve relevant OpenAPI spec chunks for this section via RAG
        rag_query = f"{section_title} — {focus}"
        openapi_reference_overview = search_openapi_reference(rag_query)

        logger.debug(
            f"Extracting [{section_id}] '{section_title}' (priority={plan_section['priority']})."
        )

        # On loop-back, append error feedback to helper context
        section_helper = helper_context
        if error_feedback:
            section_helper = f"{helper_context}\n\n{error_feedback}"

        result: SectionRules = chain.invoke({
            "section_id":                 section_id,
            "section_title":              section_title,
            "extraction_focus":           focus,
            "section_content":            section["content"],
            "openapi_reference_overview": openapi_reference_overview,
            "helper_context":             section_helper,
        })

        section_rules = result.rules
        logger.debug(f"  → {len(section_rules)} rule(s) extracted from [{section_id}].")
        raw_rules.extend([rule.model_dump() for rule in section_rules])

    logger.info(
        f"Extractor Node complete — {len(raw_rules)} raw rule(s) from "
        f"{len(plan_sections)} section(s)."
    )

    return {
        "raw_rules":       raw_rules,
        "iteration_count": iteration_count + 1,
    }
