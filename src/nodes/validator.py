# src/nodes/validator.py

"""
Validator Node: applies two-stage validation to each ReflectedRule.

Stage 1 — Structural (Pydantic, no LLM call):
    Attempts to construct a ValidatedRule from the rule dict.
    Any missing field or wrong type produces a ValidationError immediately.

Stage 2 — Semantic (LLM):
    For each structurally valid rule, the LLM checks whether the rule_text is
    grounded in the section content and whether the openapi_mapping is correct.
    Flagged rules (from the Reflector) are explicitly noted in the prompt so the
    LLM applies extra scrutiny.

Rules that pass both stages are added to validated_rules.
Rules that fail either stage are added to validation_errors.
conditions.should_loop_or_build() then decides whether to loop or proceed.

State reads:
    reflected_rules  (list[dict]) — ReflectedRule objects from reflector_node
    parsed_sections  (list[dict]) — used to retrieve section content for context

State writes:
    validated_rules   (list[dict]) — ValidatedRule objects that passed validation
    validation_errors (list[dict]) — ValidationError objects for failed rules
"""

from pydantic import ValidationError as PydanticValidationError

from config import get_logger
from config.llm_config import llm
from graph.state import RuleBankState
from prompts.validator_prompts import ValidationVerdict, validator_prompt
from schemas.rules import ValidatedRule, ValidationError

logger = get_logger(__name__)


def _build_sections_index(parsed_sections: list[dict]) -> dict[str, dict]:
    """Builds a lookup dict from section_id to section dict for O(1) access."""
    return {s["section_id"]: s for s in parsed_sections}


def _validate_structurally(rule: dict) -> tuple[dict | None, dict | None]:
    """
    Attempts to construct a ValidatedRule from the rule dict.

    Returns:
        (validated_rule_dict, None)  if the rule passes structural validation.
        (None, validation_error_dict) if it fails Pydantic validation.
    """
    try:
        validated = ValidatedRule(**rule)
        return validated.model_dump(), None
    except PydanticValidationError as e:
        error = ValidationError(
            rule=rule,
            reason=str(e),
            stage="structural",
        )
        return None, error.model_dump()


def _validate_semantically(
    rule: dict,
    section_content: str,
    chain,
) -> tuple[dict | None, dict | None]:
    """
    Calls the LLM to verify that the rule is semantically correct.

    Returns:
        (validated_rule_dict, None)  if the LLM deems the rule valid.
        (None, validation_error_dict) if the LLM deems it invalid.
    """
    mapping = rule.get("openapi_mapping", {})

    verdict: ValidationVerdict = chain.invoke({
        "section_id":     rule.get("section_id", ""),
        "section_title":  rule.get("section_title", ""),
        "rule_text":      rule.get("rule_text", ""),
        "openapi_object": mapping.get("openapi_object", ""),
        "openapi_field":  mapping.get("openapi_field", ""),
        "openapi_value":  mapping.get("openapi_value", ""),
        "confidence":     f"{rule.get('confidence', 0.0):.2f}",
        "flagged":        str(rule.get("flagged", False)),
        "reasoning":      rule.get("reasoning", ""),
        "section_content": section_content,
    })

    if verdict.valid:
        validated = ValidatedRule(
            **{k: v for k, v in rule.items() if k != "validation_notes"},
            validation_notes=verdict.reason,
        )
        return validated.model_dump(), None
    else:
        error = ValidationError(
            rule=rule,
            reason=verdict.reason,
            stage="semantic",
        )
        return None, error.model_dump()


def validator_node(state: RuleBankState) -> dict:
    """
    Validates all reflected rules through structural then semantic checks.

    Structural failures are logged and collected without an LLM call.
    Semantic validation is performed for all structurally valid rules.
    """
    logger.info("Validator Node started.")

    reflected_rules  = state.get("reflected_rules", []) or []
    sections_index   = _build_sections_index(state.get("parsed_sections", []))

    if not reflected_rules:
        logger.warning("Validator received no reflected_rules — returning empty results.")
        return {"validated_rules": [], "validation_errors": []}

    structured_llm = llm.with_structured_output(ValidationVerdict)
    chain = validator_prompt | structured_llm

    validated_rules:   list[dict] = []
    validation_errors: list[dict] = []

    for i, rule in enumerate(reflected_rules):
        section_id = rule.get("section_id", "?")
        logger.debug(
            f"Validating rule {i + 1}/{len(reflected_rules)} [{section_id}]: "
            f"{rule.get('rule_text', '')[:60]}"
        )

        # Stage 1 — Structural validation (Pydantic, free)
        valid_dict, error_dict = _validate_structurally(rule)
        if error_dict:
            logger.debug(f"  → FAIL (structural): {error_dict['reason'][:80]}")
            validation_errors.append(error_dict)
            continue

        # Stage 2 — Semantic validation (LLM)
        section = sections_index.get(section_id, {})
        section_content = section.get("content", "Section content not available.")

        valid_dict, error_dict = _validate_semantically(rule, section_content, chain)
        if error_dict:
            flag_note = " [flagged]" if rule.get("flagged") else ""
            logger.debug(f"  → FAIL (semantic){flag_note}: {error_dict['reason'][:80]}")
            validation_errors.append(error_dict)
        else:
            logger.debug(f"  → PASS (confidence={rule.get('confidence', 0):.2f})")
            validated_rules.append(valid_dict)

    logger.info(
        f"Validator Node complete — "
        f"{len(validated_rules)} validated, {len(validation_errors)} failed "
        f"out of {len(reflected_rules)} rule(s)."
    )

    return {
        "validated_rules":   validated_rules,
        "validation_errors": validation_errors,
    }
