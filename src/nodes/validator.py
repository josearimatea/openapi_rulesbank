# src/nodes/validator.py

"""
Validator Node: applies structural then semantic validation to each ReflectedRule,
and produces the final missing-rules instruction for the Extractor.

STRUCTURAL VALIDATION (no LLM — runs first for every rule):
  Stage 1a — Pydantic: verifies all required fields exist with correct types.
  Stage 1b — Mapping consistency (rules_check): verifies openapi_object/field/value
             match the rule's rule_type.
  Structural failures go to validation_errors immediately; the rule is skipped
  for semantic validation.

SEMANTIC VALIDATION (LLM — three ordered stages per section):
  Stage 1 — Initial section analysis (1 call per section, before per-rule):
    Filters previous_missing_rules against current_rules and already_validated_rules.
    Produces the starting missing_rules list for this iteration.

  Stage 2 — Per-rule verdict (1 call per structurally valid rule):
    Assesses whether the rule is correct. Produces valid/error_type/instruction.
    Also emits new_missing_rules when assessing a rule reveals absent siblings.
    missing_rules grows as each rule is processed (accumulated, deduplicated).

  Stage 3 — Reflector review (1 call per section, after all per-rule verdicts):
    Incorporates the Reflector's section_reflection into the final missing_rules.
    Confirms, adds, or excludes based on full context.
    Produces the definitive section_reflection written to state.

State reads:
    reflected_rules    (list[dict]) — ReflectedRule dicts from reflector_node
    validated_rules    (list[dict]) — accumulation of all prior iterations (operator.add)
    parsed_sections    (list[dict]) — used to retrieve section content for context
    section_reflection (list[dict]) — Reflector Fase 2 missing-rules analysis (input)

State writes:
    validated_rules            (list[dict])       — ValidatedRule objects that passed this iteration
    validation_errors          (list[dict])       — ValidationError objects for failed rules
    section_reflection         (list[dict])       — Validator's final missing-rules decision
    validated_rules_by_section (dict[str, list])  — compact summaries of ALL validated rules
                                                    keyed by section_id; read by Reflector Fase 2
"""

from pydantic import ValidationError as PydanticValidationError

from config import get_logger
from config.llm_config import llm
from config.settings import DIAGNOSTIC_MODE
from graph.state import RuleBankState
from prompts.validator_prompts import (
    SectionAnalysisResult, ValidationVerdict, ReflectorReviewResult,
    validator_section_init_prompt, validator_prompt, validator_reflector_review_prompt,
)
from schemas.rules import ValidatedRule, ValidationError
from utils.diagnostic import _diagnostic
from utils.rules_check import check_mapping_for_type

logger = get_logger(__name__)

# Fields present in a ReflectedRule dict that must NOT be carried into ValidatedRule.
_REFLECTOR_ONLY_FIELDS = {"reflection_rag_context"}


def _build_sections_index(parsed_sections: list[dict]) -> dict[str, dict]:
    return {s["section_id"]: s for s in parsed_sections}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_rules_compact(rules: list[dict]) -> str:
    if not rules:
        return "(none)"
    lines = []
    for r in rules:
        mapping = r.get("openapi_mapping", {})
        lines.append(
            f"  [{r.get('rule_type', '')}] source={r.get('source_name', '')} "
            f"field={mapping.get('openapi_field', '')} "
            f"value={mapping.get('openapi_value', '')}"
        )
    return "\n".join(lines)


def _format_already_validated(
    section_id: str,
    already_validated_previous: list[dict],
    already_validated_this_iteration: list[dict],
) -> str:
    same_section = [
        r for r in (already_validated_previous + already_validated_this_iteration)
        if r.get("section_id") == section_id
    ]
    return _format_rules_compact(same_section)


def _format_missing_rules(missing: list[str]) -> str:
    if not missing:
        return "(none)"
    return "\n".join(f"  - {m}" for m in missing)


def _format_reflector_missing(section_id: str, section_reflection: list[dict]) -> str:
    for sr in section_reflection:
        if sr.get("section_id") == section_id:
            missing = sr.get("missing_rules", [])
            return _format_missing_rules(missing)
    return "(none)"


def _format_failed_rules(errors: list[dict]) -> str:
    if not errors:
        return "(none)"
    lines = []
    for e in errors:
        rule = e.get("rule", {})
        mapping = rule.get("openapi_mapping", {})
        lines.append(
            f"  [{e.get('error_type', '')}] [{rule.get('rule_type', '')}] "
            f"field={mapping.get('openapi_field', '')} — {e.get('instruction', '')[:80]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structural validation (no LLM)
# ---------------------------------------------------------------------------

def _validate_structurally(rule: dict) -> tuple[dict | None, dict | None]:
    """
    Stage 1a (Pydantic) + Stage 1b (mapping consistency).
    Returns (validated_dict, None) on success or (None, error_dict) on failure.
    """
    section_id = rule.get("section_id", "")

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


# ---------------------------------------------------------------------------
# Semantic Stage 1 — Initial section analysis
# ---------------------------------------------------------------------------

def _run_section_init(
    section_id: str,
    current_rules: list[dict],
    already_validated: list[dict],
    previous_missing: list[str],
    chain_init,
) -> list[str]:
    """
    Filters previous_missing_rules against current coverage.
    Returns the updated missing_rules list (may be empty).
    """
    if not previous_missing:
        return []

    result: SectionAnalysisResult = chain_init.invoke({
        "section_id":            section_id,
        "current_rules":         _format_rules_compact(current_rules),
        "already_validated_rules": _format_rules_compact(already_validated),
        "previous_missing_rules":  _format_missing_rules(previous_missing),
    })
    return result.missing_rules or []


# ---------------------------------------------------------------------------
# Semantic Stage 2 — Per-rule verdict
# ---------------------------------------------------------------------------

def _validate_semantically(
    rule: dict,
    section_content: str,
    chain_rule,
    already_validated_previous: list[dict],
    already_validated_this_iteration: list[dict],
    current_missing_rules: list[str],
) -> tuple[dict | None, dict | None, list[str]]:
    """
    Semantic verdict for one rule.
    Returns (validated_dict, None, new_missing) or (None, error_dict, new_missing).
    """
    mapping = rule.get("openapi_mapping", {})
    section_id = rule.get("section_id", "")

    already_validated_str = _format_already_validated(
        section_id,
        already_validated_previous,
        already_validated_this_iteration,
    )

    verdict: ValidationVerdict = chain_rule.invoke({
        "section_id":     section_id,
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
        "already_validated_rules": already_validated_str,
        "current_missing_rules":   _format_missing_rules(current_missing_rules),
        "section_content":         section_content,
    })

    new_missing = verdict.new_missing_rules or []

    if verdict.valid:
        validated = ValidatedRule(
            **{k: v for k, v in rule.items()
               if k not in _REFLECTOR_ONLY_FIELDS | {"validation_notes"}},
            validation_notes=verdict.instruction or "",
            validation_passed=True,
        )
        return validated.model_dump(), None, new_missing
    else:
        error = ValidationError(
            error_type=verdict.error_type,
            stage="semantic",
            section_id=section_id,
            rule=rule,
            instruction=verdict.instruction,
        )
        return None, error.model_dump(), new_missing


# ---------------------------------------------------------------------------
# Semantic Stage 3 — Reflector review
# ---------------------------------------------------------------------------

def _run_reflector_review(
    section_id: str,
    validated_this_iteration: list[dict],
    already_validated: list[dict],
    current_missing_rules: list[str],
    reflector_section_reflection: list[dict],
    chain_review,
) -> list[str]:
    """
    Incorporates the Reflector's section analysis into the final missing_rules.
    Returns the definitive missing_rules list.
    """
    reflector_missing_str = _format_reflector_missing(
        section_id, reflector_section_reflection
    )
    if reflector_missing_str == "(none)" and not current_missing_rules:
        return []

    result: ReflectorReviewResult = chain_review.invoke({
        "section_id":                   section_id,
        "validated_rules_this_iteration": _format_rules_compact(validated_this_iteration),
        "already_validated_rules":        _format_rules_compact(already_validated),
        "current_missing_rules":          _format_missing_rules(current_missing_rules),
        "reflector_missing_rules":        reflector_missing_str,
    })
    return result.missing_rules or []


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def validator_node(state: RuleBankState) -> dict:
    """
    Validates all reflected rules through structural then semantic checks,
    and produces the final section_reflection for the Extractor.
    """
    logger.info("Validator Node started.")

    reflected_rules              = state.get("reflected_rules", []) or []
    sections_index               = _build_sections_index(state.get("parsed_sections", []))
    reflector_section_reflection = state.get("section_reflection", []) or []
    prior_validated_rules        = state.get("validated_rules", []) or []

    if not reflected_rules:
        logger.warning("Validator received no reflected_rules — returning empty results.")
        return {
            "validated_rules":              [],
            "validation_errors":            [],
            "validator_section_reflection": [],
            "validated_rules_by_section":   {},
        }

    reflected_rules_input = reflected_rules

    # Build LLM chains
    chain_init   = validator_section_init_prompt   | llm.with_structured_output(SectionAnalysisResult)
    chain_rule   = validator_prompt                | llm.with_structured_output(ValidationVerdict)
    chain_review = validator_reflector_review_prompt | llm.with_structured_output(ReflectorReviewResult)

    # Keys of rules already validated in prior iterations (deduplication guard)
    existing_keys: set[tuple] = {
        (r.get("section_id",  ""),
         r.get("rule_type",   ""),
         r.get("source_name", ""),
         r.get("openapi_mapping", {}).get("openapi_field", ""))
        for r in prior_validated_rules
    }

    # Keys of rules marked for discard — skip re-validation on loop-back
    discard_keys: set[tuple] = {
        (e.get("section_id", ""),
         e.get("rule", {}).get("rule_type",   ""),
         e.get("rule", {}).get("source_name", ""),
         e.get("rule", {}).get("openapi_mapping", {}).get("openapi_field", ""))
        for e in (state.get("validation_errors", []) or [])
        if e.get("error_type") == "discard"
    }

    # -------------------------------------------------------------------------
    # Structural validation — pass 1 (no LLM)
    # -------------------------------------------------------------------------
    structurally_valid: list[dict] = []
    validation_errors:  list[dict] = []

    for rule in reflected_rules:
        section_id = rule.get("section_id", "?")
        rule_key = (
            section_id,
            rule.get("rule_type",   ""),
            rule.get("source_name", ""),
            rule.get("openapi_mapping", {}).get("openapi_field", ""),
        )

        if rule_key in discard_keys:
            logger.debug(f"  → SKIP (previously discarded): {rule_key}")
            continue

        valid_dict, error_dict = _validate_structurally(rule)
        if error_dict:
            logger.debug(f"  → FAIL (structural) [{section_id}]: {error_dict['instruction'][:80]}")
            validation_errors.append(error_dict)
        else:
            structurally_valid.append(rule)

    logger.info(
        f"Structural validation: {len(structurally_valid)} valid, "
        f"{len(validation_errors)} failed out of {len(reflected_rules)} rule(s)."
    )

    if not structurally_valid:
        # Nothing to semantically validate — skip LLM stages
        return {
            "validated_rules":              [],
            "validation_errors":            validation_errors,
            "validator_section_reflection": [],
            "validated_rules_by_section":   _build_validated_by_section(prior_validated_rules, []),
        }

    # Group structurally valid rules by section for staged processing
    sections_with_rules: dict[str, list[dict]] = {}
    for rule in structurally_valid:
        sid = rule.get("section_id", "")
        sections_with_rules.setdefault(sid, []).append(rule)

    # previous_missing_rules per section (from prior iteration's section_reflection)
    previous_missing_by_section: dict[str, list[str]] = {
        sr["section_id"]: sr.get("missing_rules", [])
        for sr in reflector_section_reflection
    }

    validated_rules:  list[dict] = []
    new_section_reflection: list[dict] = []

    # -------------------------------------------------------------------------
    # Semantic validation — per section
    # -------------------------------------------------------------------------
    for section_id, section_rules in sections_with_rules.items():
        section_content = sections_index.get(section_id, {}).get(
            "content", "Section content not available."
        )
        already_validated_for_section = [
            r for r in prior_validated_rules if r.get("section_id") == section_id
        ]
        previous_missing = previous_missing_by_section.get(section_id, [])

        logger.debug(
            f"Semantic validation — section [{section_id}]: "
            f"{len(section_rules)} rule(s), "
            f"{len(previous_missing)} previous missing rule(s)."
        )

        # Stage 1 — Filter previous missing rules
        current_missing: list[str] = _run_section_init(
            section_id=        section_id,
            current_rules=     section_rules,
            already_validated= already_validated_for_section,
            previous_missing=  previous_missing,
            chain_init=        chain_init,
        )
        logger.debug(
            f"  Stage 1: {len(current_missing)} missing rule(s) carried forward "
            f"from previous {len(previous_missing)}."
        )

        # Stage 2 — Per-rule verdict
        validated_this_section: list[dict] = []
        section_errors: list[dict] = []

        for i, rule in enumerate(section_rules):
            rule_key = (
                section_id,
                rule.get("rule_type",   ""),
                rule.get("source_name", ""),
                rule.get("openapi_mapping", {}).get("openapi_field", ""),
            )

            logger.debug(
                f"  Validating rule {i + 1}/{len(section_rules)} [{section_id}]: "
                f"{rule.get('rule_text', '')[:60]}"
            )

            valid_dict, error_dict, new_missing = _validate_semantically(
                rule=                           rule,
                section_content=                section_content,
                chain_rule=                     chain_rule,
                already_validated_previous=     already_validated_for_section,
                already_validated_this_iteration= validated_this_section,
                current_missing_rules=          current_missing,
            )

            # Accumulate new missing rules (deduplicated)
            for m in new_missing:
                if m not in current_missing:
                    current_missing.append(m)

            if error_dict:
                flag_note = " [flagged]" if rule.get("reflection_flagged") else ""
                logger.debug(
                    f"    → FAIL ({error_dict['error_type']}){flag_note}: "
                    f"{error_dict['instruction'][:80]}"
                )
                section_errors.append(error_dict)
                validation_errors.append(error_dict)
            else:
                if rule_key in existing_keys:
                    logger.debug(f"    → SKIP (already validated in prior iteration): {rule_key}")
                    continue
                logger.debug(f"    → PASS (confidence={rule.get('reflection_confidence', 0):.2f})")
                validated_this_section.append(valid_dict)
                validated_rules.append(valid_dict)
                existing_keys.add(rule_key)

        logger.debug(
            f"  Stage 2: {len(validated_this_section)} validated, "
            f"{len(section_errors)} failed, "
            f"{len(current_missing)} missing rule(s) so far."
        )

        # Stage 3 — Reflector review
        final_missing = _run_reflector_review(
            section_id=                   section_id,
            validated_this_iteration=     validated_this_section,
            already_validated=            already_validated_for_section,
            current_missing_rules=        current_missing,
            reflector_section_reflection= reflector_section_reflection,
            chain_review=                 chain_review,
        )
        logger.debug(
            f"  Stage 3: final missing_rules = {len(final_missing)} rule(s)."
        )

        if final_missing:
            new_section_reflection.append({
                "section_id":    section_id,
                "missing_rules": final_missing,
                "reasoning":     "",
            })

    # Build validated_rules_by_section (all iterations)
    all_validated = list(prior_validated_rules) + validated_rules
    validated_rules_by_section = _build_validated_by_section(prior_validated_rules, validated_rules)

    logger.info(
        f"Validator Node complete — "
        f"{len(validated_rules)} validated, {len(validation_errors)} failed "
        f"out of {len(reflected_rules_input)} rule(s). "
        f"validator_section_reflection: {len(new_section_reflection)} section(s) with missing rules."
    )

    if DIAGNOSTIC_MODE and _diagnostic is not None:
        _diagnostic.record_iteration(
            iteration_num=                  state.get("section_iteration_count", 1),
            reflected_rules=                reflected_rules_input,
            validation_errors=              validation_errors,
            validated_rules=                validated_rules,
            section_feedback=               [],
            reflector_section_reflection=   reflector_section_reflection,
            validator_section_reflection=   new_section_reflection,
            error_rate=(
                len(validation_errors) / len(reflected_rules_input)
                if reflected_rules_input else 0.0
            ),
        )

    return {
        "validated_rules":              validated_rules,
        "validation_errors":            validation_errors,
        "validator_section_reflection": new_section_reflection,
        "validated_rules_by_section":   validated_rules_by_section,
    }


def _build_validated_by_section(
    prior: list[dict],
    current: list[dict],
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for vr in (prior + current):
        sid = vr.get("section_id", "")
        compact = {
            "rule_type":       vr.get("rule_type", ""),
            "source_name":     vr.get("source_name", ""),
            "rule_text":       vr.get("rule_text", "")[:120],
            "openapi_mapping": vr.get("openapi_mapping", {}),
        }
        result.setdefault(sid, []).append(compact)
    return result
