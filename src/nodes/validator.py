# src/nodes/validator.py

"""
Validator Node: applies three-stage validation to each ReflectedRule.

Stage 1a — Structural / Pydantic (no LLM call):
    Attempts to construct a ValidatedRule from the rule dict.
    Any missing field or wrong type produces a ValidationError immediately.

Stage 1b — Mapping consistency / rules_check (no LLM call):
    Calls utils.rules_check.check_mapping_for_type() to verify that
    openapi_object, openapi_field, and openapi_value are consistent with
    the rule's rule_type (e.g. a path_operation must have a single lowercase
    HTTP method as openapi_field, never "put, patch").
    This is a deterministic check that runs only if Stage 1a passes.

Stage 2 — Semantic (LLM):
    For each structurally valid rule, the LLM checks whether the rule_text is
    grounded in the section content and whether the openapi_mapping is correct.
    The LLM also evaluates the Reflector's split/discard/missing suggestions
    and decides whether to agree. Flagged rules receive extra scrutiny.

Rules that pass all three stages are added to validated_rules.
Rules that fail any stage are added to validation_errors (as ValidationError dicts).
When the Validator agrees with missing_rules suggestions, SectionFeedback entries
are produced in section_feedback for the Extractor to act on.
conditions.should_loop_or_build() then decides whether to loop or proceed.

State reads:
    reflected_rules  (list[dict]) — ReflectedRule objects from reflector_node
    parsed_sections  (list[dict]) — used to retrieve section content for context

State writes:
    validated_rules   (list[dict]) — ValidatedRule objects that passed validation
    validation_errors (list[dict]) — ValidationError objects for failed rules
    section_feedback  (list[dict]) — SectionFeedback objects for missing rules
"""

from pydantic import ValidationError as PydanticValidationError

from config import get_logger
from config.llm_config import llm
from config.settings import DIAGNOSTIC_MODE
from graph.state import RuleBankState
from prompts.validator_prompts import ValidationVerdict, validator_prompt
from schemas.rules import ValidatedRule, ValidationError, SectionFeedback
from utils.diagnostic import _diagnostic
from utils.rules_check import check_mapping_for_type

logger = get_logger(__name__)


def _build_sections_index(parsed_sections: list[dict]) -> dict[str, dict]:
    """Builds a lookup dict from section_id to section dict for O(1) access."""
    return {s["section_id"]: s for s in parsed_sections}


def _validate_structurally(rule: dict) -> tuple[dict | None, dict | None]:
    """
    Validates a rule structurally in two steps:

    Stage 1a — Pydantic:
        Verifies all required fields exist with correct Python types.
        Failure means the rule is malformed and cannot be processed further.

    Stage 1b — Mapping consistency (rules_check):
        Verifies that openapi_object/field/value match the rule's rule_type.
        E.g. a path_operation must have a single lowercase HTTP method as
        openapi_field. Runs only if Stage 1a passes.

    Returns:
        (validated_rule_dict, None)   if both stages pass.
        (None, validation_error_dict) if either stage fails.
    """
    section_id = rule.get("section_id", "")

    # Stage 1a — Pydantic structural check
    try:
        validated = ValidatedRule(**rule)
    except PydanticValidationError as e:
        return None, ValidationError(
            error_type="correction",
            stage="structural",
            section_id=section_id,
            rule=rule,
            instruction=str(e),
        ).model_dump()

    # Stage 1b — rule_type / mapping consistency check
    mapping_errors = check_mapping_for_type(
        rule.get("rule_type", ""),
        rule.get("openapi_mapping", {}),
    )
    if mapping_errors:
        return None, ValidationError(
            error_type="correction",
            stage="structural",
            section_id=section_id,
            rule=rule,
            instruction="; ".join(mapping_errors),
        ).model_dump()

    return validated.model_dump(), None


def _validate_semantically(
    rule: dict,
    section_content: str,
    chain,
) -> tuple[dict | None, dict | None, bool]:
    """
    Calls the LLM to verify that the rule is semantically correct and evaluates
    the Reflector's split/discard/missing suggestions.

    Returns:
        (validated_rule_dict, None, agreed_with_missing)
            if the LLM deems the rule valid.
        (None, validation_error_dict, agreed_with_missing)
            if the LLM deems it invalid.
    """
    mapping = rule.get("openapi_mapping", {})

    verdict: ValidationVerdict = chain.invoke({
        "section_id":     rule.get("section_id", ""),
        "section_title":  rule.get("section_title", ""),
        "rule_type":      rule.get("rule_type", ""),
        "rule_text":      rule.get("rule_text", ""),
        "openapi_object": mapping.get("openapi_object", ""),
        "openapi_field":  mapping.get("openapi_field", ""),
        "openapi_value":  mapping.get("openapi_value", ""),
        "reflection_confidence": f"{rule.get('reflection_confidence', 0.0):.2f}",
        "reflection_flagged":    str(rule.get("reflection_flagged", False)),
        "reflection_reasoning":  rule.get("reflection_reasoning", ""),
        "split_suggestion":      rule.get("split_suggestion", ""),
        "discard_suggestion":    str(rule.get("discard_suggestion", False)),
        "missing_rules":         str(rule.get("missing_rules", [])),
        "section_content":       section_content,
    })

    if verdict.valid:
        validated = ValidatedRule(
            **{k: v for k, v in rule.items() if k != "validation_notes"},
            validation_notes=verdict.instruction or "",
            validation_passed=True,
        )
        return validated.model_dump(), None, verdict.agreed_with_missing
    else:
        error = ValidationError(
            error_type=verdict.error_type,
            stage="semantic",
            section_id=rule.get("section_id", ""),
            rule=rule,
            instruction=verdict.instruction,
        )
        return None, error.model_dump(), verdict.agreed_with_missing


def validator_node(state: RuleBankState) -> dict:
    """
    Validates all reflected rules through structural then semantic checks.

    Structural failures are logged and collected without an LLM call.
    Semantic validation is performed for all structurally valid rules.
    Missing rules feedback (SectionFeedback) is produced when the Validator
    agrees with the Reflector's missing_rules suggestions.
    """
    logger.info("Validator Node started.")

    reflected_rules = state.get("reflected_rules", []) or []
    sections_index  = _build_sections_index(state.get("parsed_sections", []))

    if not reflected_rules:
        logger.warning("Validator received no reflected_rules — returning empty results.")
        return {"validated_rules": [], "validation_errors": [], "section_feedback": []}

    # Snapshot the input for diagnostic reporting (reflected_rules may be shadowed below)
    reflected_rules_input = reflected_rules

    structured_llm = llm.with_structured_output(ValidationVerdict)
    chain = validator_prompt | structured_llm

    validated_rules:      list[dict] = []
    validation_errors:    list[dict] = []
    section_feedback_map: dict[str, list[str]] = {}

    for i, rule in enumerate(reflected_rules):
        section_id = rule.get("section_id", "?")
        logger.debug(
            f"Validating rule {i + 1}/{len(reflected_rules)} [{section_id}]: "
            f"{rule.get('rule_text', '')[:60]}"
        )

        # Stage 1a + 1b — Structural validation (Pydantic + mapping consistency, no LLM)
        valid_dict, error_dict = _validate_structurally(rule)
        if error_dict:
            logger.debug(f"  → FAIL (structural): {error_dict['instruction'][:80]}")
            validation_errors.append(error_dict)
            continue

        # Stage 2 — Semantic validation (LLM)
        section = sections_index.get(section_id, {})
        section_content = section.get("content", "Section content not available.")

        valid_dict, error_dict, agreed_with_missing = _validate_semantically(
            rule, section_content, chain
        )

        # Collect missing-rules feedback regardless of pass/fail
        if agreed_with_missing:
            missing = rule.get("missing_rules", [])
            if missing:
                section_feedback_map.setdefault(section_id, []).extend(missing)

        if error_dict:
            flag_note = " [flagged]" if rule.get("reflection_flagged") else ""
            logger.debug(
                f"  → FAIL ({error_dict['error_type']}){flag_note}: "
                f"{error_dict['instruction'][:80]}"
            )
            validation_errors.append(error_dict)
        else:
            logger.debug(f"  → PASS (confidence={rule.get('reflection_confidence', 0):.2f})")
            validated_rules.append(valid_dict)

    section_feedback = [
        SectionFeedback(section_id=sid, missing_rules=rules).model_dump()
        for sid, rules in section_feedback_map.items()
    ]

    logger.info(
        f"Validator Node complete — "
        f"{len(validated_rules)} validated, {len(validation_errors)} failed "
        f"out of {len(reflected_rules_input)} rule(s). "
        f"{len(section_feedback)} section(s) with missing-rules feedback."
    )

    if DIAGNOSTIC_MODE and _diagnostic is not None:
        _diagnostic.record_iteration(
            iteration_num=state.get("iteration_count", 1),
            reflected_rules=reflected_rules_input,
            validation_errors=validation_errors,
            validated_rules=validated_rules,
            section_feedback=section_feedback,
            error_rate=(
                len(validation_errors) / len(reflected_rules_input)
                if reflected_rules_input else 0.0
            ),
        )

    return {
        "validated_rules":   validated_rules,
        "validation_errors": validation_errors,
        "section_feedback":  section_feedback,
    }
