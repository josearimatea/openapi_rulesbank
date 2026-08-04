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
    service_contract        (dict)       — scope contract from contract_builder_node;
                                           the LLM extracts only rules that belong to
                                           the target service's OpenAPI YAML
    validation_errors             (list[dict]) — correction errors from Validator (empty on first)
    validator_section_reflection  (list[dict]) — missing-rules decided by Validator (empty on first)
    current_section_idx     (int)        — index of the section to process
    section_iteration_count (int)        — how many times this section has been attempted

State writes:
    raw_rules               (list[dict]) — RawRule dicts (section_id, section_title, rule_type,
                                          source_name, rule_text, openapi_mapping{object, field, value})
    section_iteration_count (int)        — incremented by 1
"""

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.llm_config import llm
from openapi_rulesbank.config.settings import (
    SPEC_RETRIEVE_CHUNKS,
    SPEC_SIBLING_RETRIEVE_CHUNKS,
    RULES_EXTRACTOR_RETRIEVE_CHUNKS,
)
from openapi_rulesbank.graph.state import RuleBankState
from openapi_rulesbank.prompts.extractor_prompts import SectionRules, extractor_prompt
from openapi_rulesbank.schemas.contract import format_contract_summary
from openapi_rulesbank.tools.rag_tools import (
    search_openapi_reference,
    search_target_spec,
    search_rules_by_text,
    umbrella_mns_for as _umbrella_mns,
)

logger = get_logger(__name__)


def _build_sections_index(parsed_sections: list[dict]) -> dict[str, dict]:
    """
    Builds a lookup dict from section_id to section dict for O(1) access.
    """
    return {s["section_id"]: s for s in parsed_sections}


def _format_rules_compact(rules: list[dict]) -> list[str]:
    """One line per rule: type, name, mapping — enough to recognise it, no prose."""
    lines = []
    for r in rules:
        m = r.get("openapi_mapping", {}) or {}
        lines.append(
            f"  [{r.get('rule_type', '?')}] {r.get('source_name', '?')} -> "
            f"{m.get('openapi_object', '')} {m.get('openapi_field', '')} "
            f"{m.get('openapi_value', '')}".rstrip()
        )
    return lines


def _build_correction_task(
    validation_errors: list[dict],
    section_reflection: list[dict],
    already_validated: list[dict] | None = None,
    discarded_errors: list[dict] | None = None,
) -> str:
    """
    Returns a correction instruction for retry attempts on a section.

    Lists, in order: the rules already validated and the ones the Validator
    discarded (neither must be produced again), the rules that failed and how to
    fix them, and the missing rules to add. Without those first two lists the LLM
    cannot honour "do not duplicate", so it re-extracts the whole section on every
    retry — including rules just thrown away.
    Returns "" on the first attempt (no errors and no missing rules).
    """
    if not validation_errors and not section_reflection:
        return ""

    lines = [
        "CORRECTION TASK — this section has ALREADY been extracted.",
        "Return ONLY the rules listed under 'RULES TO CORRECT' and 'RULES TO EXTRACT",
        "FROM SCRATCH' below. Do NOT re-extract the section from the beginning.",
        "You may produce more or fewer rules than before — quality over quantity.",
        "",
    ]

    if already_validated:
        lines.append(
            "ALREADY VALIDATED — these rules exist and are correct. "
            "Do NOT produce them again, in any form:"
        )
        lines += _format_rules_compact(already_validated)
        lines.append("")

    if discarded_errors:
        lines.append(
            "DISCARDED — the Validator rejected these (duplicates of other sections, "
            "or not mappable to OpenAPI). Do NOT extract them again:"
        )
        for err in discarded_errors:
            rule = err.get("rule", {})
            lines.append(
                f"  [{rule.get('rule_type', '?')}] {rule.get('source_name', '?')}: "
                f"{err.get('instruction', 'discarded')}"
            )
        lines.append("")

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

    n_fix     = len(validation_errors)
    n_missing = sum(len(sr.get("missing_rules", [])) for sr in section_reflection)
    lines += [
        "",
        f"EXPECTED OUTPUT: {n_fix} corrected rule(s) and {n_missing} new rule(s). "
        "Anything already listed as validated above must not appear in your output.",
    ]
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
    contract_summary        = (
        format_contract_summary(state.get("service_contract", {}))
        or "No service contract provided."
    )
    all_validation_errors = state.get("validation_errors", []) or []
    # correction + split are the ones the Extractor must act on
    validation_errors  = [
        e for e in all_validation_errors if e.get("error_type") != "discard"
    ]
    # discards need no fix, but the Extractor must know about them — otherwise it
    # re-extracts on the next retry exactly what the Validator just threw away
    # (duplicates of other sections, unmappable statements, ...).
    discarded_errors   = [
        e for e in all_validation_errors if e.get("error_type") == "discard"
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
        # Rules the Validator already accepted for this section — the Extractor must
        # see them to honour "do not duplicate"; without them it re-extracts the
        # whole section on every retry.
        already_validated = (
            state.get("validated_rules_by_section", {}) or {}
        ).get(section_id, [])
        correction_task = _build_correction_task(
            validation_errors, section_reflection, already_validated, discarded_errors,
        )
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

    # Retrieve related chunks from the 3GPP document itself (temporary spec RAG).
    # A section is extracted in isolation, so two things it needs live in OTHER
    # sections of the same service: the canonical IS operation name (stage 2), and
    # the resource path, schema name/casing and property list (the sibling stage 3
    # sections). We fetch each with its own scoped query so both signals are
    # present and neither crowds out the other. Both are scoped to the same MnS to
    # stay off other services' look-alike sections.
    spec_collection = state.get("spec_collection_name", "")
    target_mns      = _umbrella_mns(section_title, state["parsed_sections"])
    # The section names the resources, types and operations it refers to, and the
    # sibling sections are titled after those names — so seeding the query with the
    # section's own text (title, the Planner's focus, and the opening of the body)
    # is what pulls the right siblings. A short per-stage intent hint biases
    # retrieval toward the abstract definition (stage 2) or the concrete
    # resource/schema sections (stage 3) without parsing any names out.
    seed = f"{section_title} {focus} {section['content'][:300]}"

    def _scoped(stage: str, hint: str, k: int) -> str:
        filters = {"stage": stage}
        if target_mns:
            filters["mns"] = target_mns
        return search_target_spec(f"{seed} {hint}", spec_collection, k=k, filters=filters)

    stage2_context = _scoped(
        "2", "abstract IS operation and notification definition",
        SPEC_RETRIEVE_CHUNKS,
    ) or "No stage 2 definition retrieved."
    stage3_context = _scoped(
        "3", "resource URI path, HTTP method, data type and schema definition",
        SPEC_SIBLING_RETRIEVE_CHUNKS,
    ) or "No related stage 3 section retrieved."

    # Rules already validated in OTHER sections that resemble this one, so the
    # Extractor does not re-emit a rule that another section already owns — even
    # when it would phrase the value differently (the deterministic dedup misses
    # those, since the value differs). Empty until earlier sections are validated.
    existing_rules = search_rules_by_text(
        seed,
        state.get("rules_collection_name", ""),
        k=RULES_EXTRACTOR_RETRIEVE_CHUNKS,
        exclude_section_id=section_id,
    ) or "None yet."

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
        "stage2_context":             stage2_context,
        "stage3_context":             stage3_context,
        "existing_rules":             existing_rules,
        "helper_context":             helper_context,
        "contract_summary":           contract_summary,
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
