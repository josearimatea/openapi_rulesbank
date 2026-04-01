# src/nodes/extractor.py

"""
Extractor Node: extracts raw OpenAPI rules from each 3GPP section guided
by the Planner's extraction plan.

LLM call: yes (1 call per section). RAG: yes (1 query per section via Qdrant).

HOW IT WORKS:
    Iterates over sections in extraction_plan["sections_to_extract"]. For each:
      1. Retrieves relevant OpenAPI reference chunks from Qdrant via RAG
         (query: section_title — extraction_focus).
      2. Invokes the LLM with the section content, RAG context, and helper context.
      3. The LLM extracts RawRule objects, each with a rule_type that categorizes
         the OpenAPI construct (path_operation, schema_property, path_parameter,
         query_parameter, response, request_body, security_scheme).

    The prompt enforces: one rule per HTTP method (never combined), extract ONLY
    from section content (not from the OpenAPI reference), and use the OpenAPI
    reference chunks ONLY to understand valid construct names and field formats.

ON LOOP-BACK (validation_errors not empty):
    Only sections whose section_id appears in validation_errors are re-processed
    (not all sections). The LLM receives a CORRECTION TASK listing the specific
    failed rules and their reasons, instructing it to return exactly one corrected
    rule per failed entry. New rules must not be added.

State reads:
    parsed_sections   (list[dict]) — full section content from reader_node
    extraction_plan   (dict)       — sections_to_extract from planner_node
    helper_context    (str)        — auxiliary 3GPP documents context
    validation_errors (list[dict]) — failed rules from previous iteration (empty on iter 1)
    iteration_count   (int)        — current loop iteration number

State writes:
    raw_rules       (list[dict]) — RawRule dicts (section_id, section_title, rule_type,
                                   rule_text, openapi_mapping{object, field, value})
    iteration_count (int)        — incremented by 1
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


def _get_sections_to_reprocess(
    plan_sections: list[dict],
    validation_errors: list[dict],
) -> list[dict]:
    """
    On loop-back, returns only the plan sections that had validation errors.
    Preserves original plan order.
    """
    failed_ids = {err.get("rule", {}).get("section_id") for err in validation_errors}
    return [s for s in plan_sections if s["section_id"] in failed_ids]


def _build_correction_task(validation_errors: list[dict]) -> str:
    """
    Returns a correction instruction for loop-back iterations.
    Lists only the failed rules so the LLM corrects them specifically.
    Returns "" on first iteration.
    """
    if not validation_errors:
        return ""

    lines = [
        "CORRECTION TASK — do NOT extract all rules from this section.",
        "ONLY correct and re-extract the specific rules listed below that failed validation.",
        "Return exactly one corrected rule per entry. Do not add new rules.",
        "",
        "Failed rules to correct:",
    ]
    for i, err in enumerate(validation_errors, start=1):
        rule = err.get("rule", {})
        lines.append(
            f"  {i}. rule_text: {rule.get('rule_text', '?')!r}\n"
            f"     Reason it failed: {err.get('reason', '?')}"
        )
    return "\n".join(lines)


def extractor_node(state: RuleBankState) -> dict:
    """
    Iterates over sections in the extraction plan and extracts raw OpenAPI rules.

    For each section:
      1. Retrieves RAG chunks from the OpenAPI reference collection.
      2. Calls the LLM with the section content + RAG context.
      3. Collects RawRule objects into raw_rules.

    On loop-back (validation_errors not empty):
      - Only re-processes sections that had failing rules.
      - Instructs the LLM to correct only the specific failed rules.
    """
    logger.info("Extractor Node started.")

    sections_index    = _build_sections_index(state["parsed_sections"])
    all_plan_sections = state["extraction_plan"]["sections_to_extract"]
    helper_context    = state.get("helper_context", "") or "No auxiliary context provided."
    validation_errors = state.get("validation_errors", []) or []
    iteration_count   = state.get("iteration_count", 0) or 0

    # On loop-back: restrict to sections with errors and build correction instruction
    if validation_errors:
        plan_sections   = _get_sections_to_reprocess(all_plan_sections, validation_errors)
        correction_task = _build_correction_task(validation_errors)
        logger.info(
            f"Loop-back iteration {iteration_count + 1} — "
            f"{len(validation_errors)} error(s), re-processing "
            f"{len(plan_sections)} section(s)."
        )
    else:
        plan_sections   = all_plan_sections
        correction_task = ""

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

        result: SectionRules = chain.invoke({
            "section_id":                 section_id,
            "section_title":              section_title,
            "extraction_focus":           focus,
            "section_content":            section["content"],
            "openapi_reference_overview": openapi_reference_overview,
            "helper_context":             helper_context,
            "correction_task":            correction_task,
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
