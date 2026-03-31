# src/nodes/reflector.py

"""
Reflector Node: applies Chain-of-Thought self-reflection over each raw rule
produced by the Extractor before passing them to the Validator.

For each RawRule, the Reflector:
  1. Builds a RAG query from the rule text and openapi_object.
  2. Retrieves relevant OpenAPI reference chunks via search_openapi_reference().
  3. Invokes the LLM to assess confidence, produce reasoning, and flag uncertain rules.
  4. Merges the reflection result into the rule → ReflectedRule.

State reads:
    raw_rules   (list[dict]) — RawRule objects from extractor_node

State writes:
    reflected_rules (list[dict]) — ReflectedRule objects with confidence,
                                   reasoning, flagged, and rag_context fields
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

    structured_llm = llm.with_structured_output(ReflectionResult)
    chain = reflector_prompt | structured_llm

    reflected_rules: list[dict] = []

    for i, rule in enumerate(raw_rules):
        mapping = rule.get("openapi_mapping", {})

        # Retrieve relevant OpenAPI reference chunks for this rule
        rag_query   = _build_rag_query(rule)
        rag_context = search_openapi_reference(rag_query)

        logger.debug(
            f"Reflecting rule {i + 1}/{len(raw_rules)} "
            f"[{rule.get('section_id', '?')}]: {rule.get('rule_text', '')[:60]}"
        )

        result: ReflectionResult = chain.invoke({
            "openapi_reference_context": rag_context or "No relevant context retrieved.",
            "section_id":    rule.get("section_id", ""),
            "section_title": rule.get("section_title", ""),
            "rule_text":     rule.get("rule_text", ""),
            "openapi_object": mapping.get("openapi_object", ""),
            "openapi_field":  mapping.get("openapi_field", ""),
            "openapi_value":  mapping.get("openapi_value", ""),
        })

        # Merge reflection fields into the rule dict → ReflectedRule
        reflected_rule = {
            **rule,
            "confidence": result.confidence,
            "reasoning":  result.reasoning,
            "flagged":    result.flagged,
            "rag_context": rag_context,
        }
        reflected_rules.append(reflected_rule)

        logger.debug(
            f"  → confidence={result.confidence:.2f}  flagged={result.flagged}"
        )

    flagged_count = sum(1 for r in reflected_rules if r["flagged"])
    logger.info(
        f"Reflector Node complete — {len(reflected_rules)} rule(s) reflected, "
        f"{flagged_count} flagged for priority validation."
    )

    return {"reflected_rules": reflected_rules}
