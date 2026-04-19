# src/nodes/extractor.py

"""
Extractor Node: extracts raw OpenAPI rules from the current section guided
by the Planner's extraction plan.

LLM call: yes (1 call per section attempt). RAG: yes (1 query per section via Qdrant).

HOW IT WORKS:
    Processes ONE section per call — the one at current_section_idx in the plan.

    First attempt (section_iteration_count == 0):
      1. Retrieves relevant OpenAPI reference chunks from Qdrant via RAG
         (query: section_title — extraction_focus).
      2. Invokes the LLM with the section content, RAG context, and helper context.
      3. The LLM extracts RawRule objects, each with a rule_type that categorizes
         the OpenAPI construct (path_operation, schema_property, path_parameter,
         query_parameter, response, request_body, security_scheme).

    Retry (section_iteration_count > 0, validation_errors or section_reflection not empty):
      - LLM receives a CORRECTION TASK listing failed rules and missing rules.
      - Only acts on what is listed — does not re-extract already-validated rules.

    The prompt enforces: one rule per HTTP method (never combined), extract ONLY
    from section content (not from the OpenAPI reference).

State reads:
    parsed_sections         (list[dict]) — full section content from reader_node
    extraction_plan         (dict)       — sections_to_extract from planner_node
    helper_context          (str)        — auxiliary 3GPP documents context
    validation_errors             (list[dict]) — correction errors from Validator (empty on first)
    validator_section_reflection  (list[dict]) — missing-rules decided by Validator (empty on first)
    current_section_idx     (int)        — index of the section to process
    section_iteration_count (int)        — how many times this section has been attempted

State writes:
    raw_rules               (list[dict]) — RawRule dicts (section_id, section_title, rule_type,
                                          source_name, rule_text, openapi_mapping{object, field, value})
    section_iteration_count (int)        — incremented by 1
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


def _build_correction_task(
    validation_errors: list[dict],
    section_reflection: list[dict],
) -> str:
    """
    Returns a correction instruction for retry attempts on a section.
    Lists failed rules and missing rules so the LLM can act on each specifically.
    Returns "" on the first attempt (both inputs empty).
    """
    if not validation_errors and not section_reflection:
        return ""

    lines = [
        "CORRECTION TASK — this is a retry attempt for this section.",
        "Re-extract ONLY what is listed below. Follow each instruction exactly.",
        "You may produce more or fewer rules than before — quality over quantity.",
        "Do NOT re-extract rules that were already validated successfully.",
        "",
    ]

    if validation_errors:
        lines.append("RULES TO CORRECT (existing rules that failed):")
        for i, err in enumerate(validation_errors, start=1):
            rule = err.get("rule", {})
            lines.append(
                f"  {i}. [{err.get('section_id', rule.get('section_id', '?'))}] "
                f"{err.get('error_type', 'correction').upper()}\n"
                f"     original rule_text : {rule.get('rule_text', '?')!r}\n"
                f"     rule_type          : {rule.get('rule_type', '?')}\n"
                f"     source_name        : {rule.get('source_name', '?')!r}\n"
                f"     instruction        : {err.get('instruction', '?')}"
            )

    if section_reflection:
        lines.append("")
        lines.append("RULES TO EXTRACT FROM SCRATCH (missing rules identified by Reflector):")
        for sr in section_reflection:
            for j, missing in enumerate(sr.get("missing_rules", []), start=1):
                lines.append(f"  [{sr['section_id']}] {j}. {missing}")

    return "\n".join(lines)


def extractor_node(state: RuleBankState) -> dict:
    """
    Processes the current section (current_section_idx) from the extraction plan.

    First attempt (section_iteration_count == 0): fresh extraction.
    Retry (section_iteration_count > 0): correction extraction with CORRECTION TASK prompt.

    After the section's retry loop ends, the Builder advances current_section_idx
    and resets section_iteration_count so the next call here starts a fresh section.
    """
    logger.info("Extractor Node started.")

    sections_index          = _build_sections_index(state["parsed_sections"])
    all_plan_sections       = state["extraction_plan"]["sections_to_extract"]
    helper_context          = state.get("helper_context", "") or "No auxiliary context provided."
    # correction + split only — discard errors need no Extractor action
    validation_errors  = [
        e for e in (state.get("validation_errors", []) or [])
        if e.get("error_type") != "discard"
    ]
    # Final missing-rules decided by the Validator (written to validator_section_reflection)
    section_reflection = state.get("validator_section_reflection", []) or []
    section_iteration_count = state.get("section_iteration_count", 0) or 0
    current_section_idx     = state.get("current_section_idx", 0) or 0

    if current_section_idx >= len(all_plan_sections):
        logger.error(
            f"current_section_idx={current_section_idx} out of bounds "
            f"(plan has {len(all_plan_sections)} sections). Returning empty raw_rules."
        )
        return {
            "raw_rules":               [],
            "section_iteration_count": section_iteration_count + 1,
        }

    plan_section = all_plan_sections[current_section_idx]
    section_id   = plan_section["section_id"]

    if validation_errors or section_reflection:
        correction_task = _build_correction_task(validation_errors, section_reflection)
        logger.info(
            f"Retry {section_iteration_count + 1} for [{section_id}] — "
            f"{len(validation_errors)} error(s), "
            f"{sum(len(sr.get('missing_rules', [])) for sr in section_reflection)} missing rule(s)."
        )
    else:
        correction_task = ""

    structured_llm = llm.with_structured_output(SectionRules)
    chain = extractor_prompt | structured_llm

    section = sections_index.get(section_id)
    if section is None:
        logger.warning(
            f"Section '{section_id}' not found in parsed_sections — returning empty raw_rules."
        )
        return {
            "raw_rules":               [],
            "section_iteration_count": section_iteration_count + 1,
        }

    section_title = plan_section["title"]
    focus         = plan_section["extraction_focus"]

    # Retrieve relevant OpenAPI spec chunks for this section via RAG
    rag_query = f"{section_title} — {focus}"
    openapi_reference_overview = search_openapi_reference(rag_query)

    logger.debug(
        f"Extracting [{section_id}] '{section_title}' "
        f"(priority={plan_section['priority']}, attempt={section_iteration_count + 1})."
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

    raw_rules = [rule.model_dump() for rule in result.rules]
    logger.info(
        f"Extractor Node complete — {len(raw_rules)} raw rule(s) from [{section_id}] "
        f"(section attempt {section_iteration_count + 1})."
    )

    return {
        "raw_rules":               raw_rules,
        "section_iteration_count": section_iteration_count + 1,
    }
