# src/nodes/reflector.py

"""
Reflector Node: applies Chain-of-Thought self-reflection over each raw rule
produced by the Extractor before passing them to the Validator.

LLM call: yes (1 call per rule). RAG: yes (1 query per rule via Qdrant).

HOW IT WORKS:
    For each RawRule:
      1. Builds a RAG query from rule_text + openapi_object + openapi_field.
      2. Retrieves relevant OpenAPI reference chunks from Qdrant via
         search_openapi_reference() to ground the assessment.
      3. Invokes the LLM with the rule fields (including rule_type) and the
         retrieved context. The LLM answers 4 CoT questions:
           - Is the rule grounded in the section content?
           - Is the openapi_mapping (object, field, value) correct?
           - Does the OpenAPI reference context confirm or contradict it?
           - How confident is the model in this rule?
      4. Merges the ReflectionResult (confidence, reasoning, flagged) into
         the rule dict to produce a ReflectedRule.

    Rules with confidence < 0.7 or ambiguous mappings should be flagged by
    the LLM for priority scrutiny in the Validator (enforced via prompt).

State reads:
    raw_rules (list[dict]) — RawRule dicts from extractor_node

State writes:
    reflected_rules (list[dict]) — ReflectedRule dicts, extending each RawRule with:
                                   reflection_confidence  (float 0.0–1.0)
                                   reflection_reasoning   (str — CoT explanation)
                                   reflection_flagged     (bool — True = priority validation)
                                   reflection_rag_context (str — retrieved OpenAPI chunks)
"""

from config import get_logger
from config.llm_config import llm
from graph.state import RuleBankState
from prompts.reflector_prompts import ReflectionResult, reflector_prompt
from tools.rag_tools import search_openapi_reference

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
    Applies CoT self-reflection to each raw rule and produces reflected_rules.

    One LLM call is made per rule. Rules with confidence < 0.7 or ambiguous
    mappings are automatically flagged for priority validation.
    """
    logger.info("Reflector Node started.")

    raw_rules = state.get("raw_rules", []) or []

    if not raw_rules:
        logger.warning("Reflector received no raw_rules — returning empty reflected_rules.")
        return {"reflected_rules": []}

    sections_index = {
        s["section_id"]: s
        for s in (state.get("parsed_sections", []) or [])
    }

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

        logger.debug(
            f"Reflecting rule {i + 1}/{len(raw_rules)} "
            f"[{section_id}]: {rule.get('rule_text', '')[:60]}"
        )

        result: ReflectionResult = chain.invoke({
            "openapi_reference_context": rag_context or "No relevant context retrieved.",
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
            "missing_rules":          result.missing_rules,
        }
        reflected_rules.append(reflected_rule)

        logger.debug(
            f"  → confidence={result.reflection_confidence:.2f}  "
            f"flagged={result.reflection_flagged}  "
            f"split={bool(result.split_suggestion)}  "
            f"discard={result.discard_suggestion}  "
            f"missing={len(result.missing_rules)}"
        )

    flagged_count  = sum(1 for r in reflected_rules if r["reflection_flagged"])
    split_count    = sum(1 for r in reflected_rules if r["split_suggestion"])
    discard_count  = sum(1 for r in reflected_rules if r["discard_suggestion"])
    missing_count  = sum(len(r["missing_rules"]) for r in reflected_rules)
    logger.info(
        f"Reflector Node complete — {len(reflected_rules)} rule(s) reflected, "
        f"{flagged_count} flagged, {split_count} split suggestions, "
        f"{discard_count} discard suggestions, {missing_count} missing rule(s) identified."
    )

    return {"reflected_rules": reflected_rules}
