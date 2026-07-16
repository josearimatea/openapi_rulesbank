# src/nodes/reflector.py

"""
Reflector Node: applies Chain-of-Thought self-reflection over each raw rule
produced by the Extractor before passing them to the Validator.

LLM call: yes (1 call per rule + 1 per section for Fase 2). RAG: yes (1 query per rule).

HOW IT WORKS:
    Fase 1 — Per-rule reflection:
      For each RawRule:
        1. Builds a RAG query from rule_text + openapi_object + openapi_field.
        2. Retrieves relevant OpenAPI reference chunks from Qdrant via
           search_openapi_reference() to ground the assessment.
        3. Invokes the LLM with the rule fields (including rule_type) and the
           retrieved context. The LLM answers 6 CoT questions and returns
           confidence, reasoning, flagged, split_suggestion, discard_suggestion.
        4. Merges the ReflectionResult into the rule dict → ReflectedRule.

    Fase 2 — Section-level completeness:
      After all rules are reflected, one LLM call per section checks whether
      any rules are genuinely missing from the section content. Produces a
      section_reflection list that the Extractor uses on loop-back.

    Rules with confidence < 0.7 or ambiguous mappings should be flagged by
    the LLM for priority scrutiny in the Validator (enforced via prompt).

State reads:
    raw_rules                  (list[dict]) — RawRule dicts from extractor_node
    parsed_sections            (list[dict]) — used to retrieve section content
    validated_rules_by_section (dict)       — already-validated rules per section
    section_reflection         (list[dict]) — previous iteration's missing rules

State writes:
    reflected_rules   (list[dict]) — ReflectedRule dicts extending each RawRule with:
                                     reflection_confidence, reflection_reasoning,
                                     reflection_flagged, reflection_rag_context,
                                     split_suggestion, discard_suggestion
    section_reflection (list[dict]) — SectionFeedback dicts for sections with
                                      genuinely missing rules (Fase 2 output)
"""

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.llm_config import llm
from openapi_rulesbank.graph.state import RuleBankState
from openapi_rulesbank.prompts.reflector_prompts import (
    ReflectionResult, SectionReflection,
    reflector_prompt, reflector_section_prompt,
)
from openapi_rulesbank.tools.rag_tools import (
    search_openapi_reference,
    search_spec,
    umbrella_mns_for as _umbrella_mns,
)

logger = get_logger(__name__)


def _build_rag_query(rule: dict) -> str:
    """
    Builds a focused RAG query from the rule to retrieve the most relevant
    OpenAPI reference chunks for the self-reflection assessment.
    """
    mapping = rule.get("openapi_mapping", {})
    return (
        f"{rule.get('rule_text', '')} "
        f"{mapping.get('openapi_object', '')} "
        f"{mapping.get('openapi_field', '')}"
    ).strip()


def reflector_node(state: RuleBankState) -> dict:
    """
    Fase 1: CoT self-reflection for each raw rule → reflected_rules.
    Fase 2: section-level completeness check → section_reflection.

    One LLM call is made per rule (Fase 1), then one call per section (Fase 2).
    Rules with confidence < 0.7 or ambiguous mappings are flagged for priority validation.
    """
    logger.info("Reflector Node started.")

    raw_rules = state.get("raw_rules", []) or []

    if not raw_rules:
        logger.warning("Reflector received no raw_rules — returning empty reflected_rules.")
        return {"reflected_rules": [], "section_reflection": []}

    sections_index = {
        s["section_id"]: s
        for s in (state.get("parsed_sections", []) or [])
    }

    # -------------------------------------------------------------------------
    # Fase 1 — Per-rule reflection
    # -------------------------------------------------------------------------
    structured_llm = llm.with_structured_output(ReflectionResult)
    chain = reflector_prompt | structured_llm

    reflected_rules: list[dict] = []

    for i, rule in enumerate(raw_rules):
        mapping = rule.get("openapi_mapping", {})
        section_id = rule.get("section_id", "")
        section_content = sections_index.get(section_id, {}).get(
            "content", "Section content not available."
        )

        # Retrieve relevant OpenAPI reference chunks for this rule
        rag_query   = _build_rag_query(rule)
        rag_context = search_openapi_reference(rag_query)

        # Retrieve related chunks from the source 3GPP document (spec RAG), scoped
        # to the rule's service stage 2 definitions, to check the rule is grounded
        # in the spec — not just consistent with OpenAPI. Empty when unavailable.
        section_title = rule.get("section_title", "")
        rule_mns = _umbrella_mns(section_title, state.get("parsed_sections", []) or [])
        spec_filters = {"stage": "2"}
        if rule_mns:
            spec_filters["mns"] = rule_mns
        spec_context = search_spec(
            rag_query, state.get("spec_collection_name", ""), filters=spec_filters,
        ) or "No related spec context retrieved."

        logger.debug(
            f"Reflecting rule {i + 1}/{len(raw_rules)} "
            f"[{section_id}]: {rule.get('rule_text', '')[:60]}"
        )

        result: ReflectionResult = chain.invoke({
            "openapi_reference_context": rag_context or "No relevant context retrieved.",
            "spec_context":  spec_context,
            "section_id":    section_id,
            "section_title": rule.get("section_title", ""),
            "rule_type":     rule.get("rule_type", ""),
            "rule_text":     rule.get("rule_text", ""),
            "openapi_object": mapping.get("openapi_object", ""),
            "openapi_field":  mapping.get("openapi_field", ""),
            "openapi_value":  mapping.get("openapi_value", ""),
            "section_content": section_content,
        })

        # Merge reflection fields into the rule dict → ReflectedRule
        reflected_rule = {
            **rule,
            "reflection_confidence":  result.reflection_confidence,
            "reflection_reasoning":   result.reflection_reasoning,
            "reflection_flagged":     result.reflection_flagged,
            "reflection_rag_context": rag_context,
            "split_suggestion":       result.split_suggestion,
            "discard_suggestion":     result.discard_suggestion,
        }
        reflected_rules.append(reflected_rule)

        logger.debug(
            f"  → confidence={result.reflection_confidence:.2f}  "
            f"flagged={result.reflection_flagged}  "
            f"split={bool(result.split_suggestion)}  "
            f"discard={result.discard_suggestion}"
        )

    flagged_count = sum(1 for r in reflected_rules if r["reflection_flagged"])
    split_count   = sum(1 for r in reflected_rules if r["split_suggestion"])
    discard_count = sum(1 for r in reflected_rules if r["discard_suggestion"])
    logger.info(
        f"Reflector Fase 1 complete — {len(reflected_rules)} rule(s) reflected, "
        f"{flagged_count} flagged, {split_count} split suggestions, "
        f"{discard_count} discard suggestions."
    )

    # -------------------------------------------------------------------------
    # Fase 2 — Section-level completeness check
    # -------------------------------------------------------------------------
    already_validated_by_section: dict[str, list[dict]] = (
        state.get("validated_rules_by_section", {}) or {}
    )
    previous_section_reflection: list[dict] = (
        state.get("section_reflection", []) or []
    )
    previous_missing_by_section: dict[str, list[str]] = {
        sr["section_id"]: sr.get("missing_rules", [])
        for sr in previous_section_reflection
    }

    # Build compact summaries of rules reflected in this iteration, per section
    sections_in_this_iteration: dict[str, list[str]] = {}
    for r in reflected_rules:
        sid   = r.get("section_id", "")
        entry = f"[{r['rule_type']}] {r.get('source_name', '')}: {r.get('rule_text', '')[:80]}"
        sections_in_this_iteration.setdefault(sid, []).append(entry)

    section_chain = reflector_section_prompt | llm.with_structured_output(SectionReflection)

    section_reflection: list[dict] = []

    for section_id, current_rule_lines in sections_in_this_iteration.items():
        section = sections_index.get(section_id, {})
        section_content = section.get("content", "Section content not available.")

        already_validated = already_validated_by_section.get(section_id, [])
        already_validated_str = "\n".join(
            f"  [{r.get('rule_type', '?')}] {r.get('source_name', '')}: {r.get('rule_text', '')[:80]}"
            for r in already_validated
        ) or "None"

        previous_missing = previous_missing_by_section.get(section_id, [])
        previous_missing_str = "\n".join(
            f"  - {m}" for m in previous_missing
        ) or "None"

        logger.debug(f"Reflector Fase 2 — section [{section_id}]")

        result: SectionReflection = section_chain.invoke({
            "section_id":            section_id,
            "section_content":       section_content,
            "current_rules":         "\n".join(current_rule_lines),
            "already_validated_rules": already_validated_str,
            "previous_missing_rules":  previous_missing_str,
        })

        if result.missing_rules:
            logger.debug(
                f"  → {len(result.missing_rules)} missing rule(s) identified in [{section_id}]"
            )
            section_reflection.append({
                "section_id":    section_id,
                "missing_rules": result.missing_rules,
                "reasoning":     result.reasoning,
            })

    logger.info(
        f"Reflector Fase 2 complete — "
        f"{len(section_reflection)} section(s) with missing rules identified."
    )

    return {"reflected_rules": reflected_rules, "section_reflection": section_reflection}
