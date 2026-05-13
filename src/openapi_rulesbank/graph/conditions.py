# src/graph/conditions.py
# Conditional edge routing functions for the LangGraph pipeline.
#
# HOW IT WORKS:
#   LangGraph calls these functions after a node finishes running.
#   Each function receives the current state and returns a string — the name
#   of the next node to execute. That string must match a key in the routing
#   map defined in rule_bank_flow.py when add_conditional_edges() is called.
#
# ROUTING LOGIC (per-section loop):
#
#   After Validator → should_retry_section_or_build:
#       If the current section still has errors (above threshold) or missing rules,
#       AND the section has not yet exhausted its retry budget → "extractor" (retry).
#       Otherwise → "builder" (write section results and advance).
#
#   After Builder → should_next_section_or_end:
#       If there are more sections to process → "extractor" (next section).
#       Otherwise → "end" (pipeline complete).

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.settings import VALIDATION_ERROR_THRESHOLD, MAX_ITERATIONS
from openapi_rulesbank.graph.state import RuleBankState

logger = get_logger(__name__)


def should_retry_section_or_build(state: RuleBankState) -> str:
    """
    Called after the Validator Node.

    Decides whether to retry the current section or pass to the Builder:
      - "extractor" → retry if error rate exceeds threshold OR missing rules exist,
                      AND the section has not yet used all its retry attempts.
      - "builder"   → proceed to write this section's results and advance.

    State fields read:
        validation_errors       (list[dict]) — rules that failed validation this attempt
        reflected_rules         (list[dict]) — rules that entered the validator
        section_reflection      (list[dict]) — sections with missing rules (Reflector Fase 2)
        section_iteration_count (int)        — how many times this section has been attempted
        current_section_idx     (int)        — for logging the current section id
    """
    # Only correction errors count toward the retry threshold.
    # Split/discard outcomes are handled via validator_section_reflection.
    errors             = [
        e for e in (state.get("validation_errors", []) or [])
        if e.get("error_type") == "correction"
    ]
    section_reflection = state.get("validator_section_reflection", [])
    total              = len(state.get("reflected_rules", []))
    section_iter       = state.get("section_iteration_count", 0)
    current_idx        = state.get("current_section_idx", 0)

    plan_sections = state.get("extraction_plan", {}).get("sections_to_extract", [])
    section_id = (
        plan_sections[current_idx]["section_id"]
        if current_idx < len(plan_sections) else "?"
    )

    error_rate        = len(errors) / total if total > 0 else 0.0
    has_missing       = bool(section_reflection)
    exceeds_threshold = error_rate > VALIDATION_ERROR_THRESHOLD

    logger.info(
        f"[{section_id}] Attempt {section_iter}/{MAX_ITERATIONS} — "
        f"error rate: {error_rate:.1%} ({len(errors)}/{total} rules), "
        f"missing-rule sections: {len(section_reflection)}"
    )

    if (exceeds_threshold or has_missing) and section_iter < MAX_ITERATIONS:
        reasons = []
        if exceeds_threshold:
            reasons.append(
                f"error rate {error_rate:.1%} exceeds threshold {VALIDATION_ERROR_THRESHOLD:.0%}"
            )
        if has_missing:
            missing_count = sum(len(sr.get("missing_rules", [])) for sr in section_reflection)
            reasons.append(f"{missing_count} missing rule(s) identified by Validator")
        logger.warning(
            f"[{section_id}] Retrying ({'; '.join(reasons)}) — attempt {section_iter + 1}."
        )
        return "extractor"

    if section_iter >= MAX_ITERATIONS:
        logger.warning(
            f"[{section_id}] Max attempts ({MAX_ITERATIONS}) reached. "
            "Proceeding to Builder with current results."
        )

    return "builder"


def should_next_section_or_end(state: RuleBankState) -> str:
    """
    Called after the Builder Node.

    Decides whether to process the next section or end the pipeline:
      - "extractor" → there are more sections to process (current_section_idx was
                      advanced by Builder and is still within bounds).
      - "end"       → all sections have been processed.

    State fields read:
        current_section_idx (int)  — already advanced by Builder
        extraction_plan     (dict) — sections_to_extract list
    """
    current_idx   = state.get("current_section_idx", 0)
    plan_sections = state.get("extraction_plan", {}).get("sections_to_extract", [])
    total         = len(plan_sections)

    if current_idx < total:
        next_section = plan_sections[current_idx]
        logger.info(
            f"Advancing to section [{next_section['section_id']}] "
            f"({current_idx + 1}/{total})."
        )
        return "extractor"

    logger.info(f"All {total} section(s) processed. Pipeline complete.")
    return "end"
