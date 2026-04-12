# src/graph/conditions.py
# Conditional edge routing functions for the LangGraph pipeline.
#
# HOW IT WORKS:
#   LangGraph calls these functions after a node finishes running.
#   Each function receives the current state and returns a string — the name
#   of the next node to execute. That string must match a key in the routing
#   map defined in rule_bank_flow.py when add_conditional_edges() is called.
#
#   Example in rule_bank_flow.py:
#       graph.add_conditional_edges(
#           "validator",
#           should_loop_or_build,               # ← function defined here
#           {"extractor": "extractor",           # ← routing map
#            "builder": "builder"}
#       )
#
# ADDING NEW CONDITIONS:
#   If the pipeline grows (e.g., Planner deciding to skip sections, Reflector
#   requesting re-extraction), add new routing functions here and wire them
#   in rule_bank_flow.py.

from config import get_logger
from config.settings import VALIDATION_ERROR_THRESHOLD, MAX_ITERATIONS
from graph.state import RuleBankState

logger = get_logger(__name__)


def should_loop_or_build(state: RuleBankState) -> str:
    """
    Called after the Validator Node.

    Computes the validation error rate and decides:
      - "extractor" → loop back if too many rules failed validation and
                       the iteration limit has not been reached yet.
      - "builder"   → proceed to build the final output.

    State fields read:
        validation_errors (list[dict]) — rules that failed validation
        reflected_rules   (list[dict]) — total rules entering the validator
        iteration_count   (int)        — how many loops have run so far
    """
    errors    = state.get("validation_errors", [])
    total     = len(state.get("reflected_rules", []))
    iteration = state.get("iteration_count", 0)

    # Error rate measures: of the rules attempted this iteration, how many failed?
    # Denominator is reflected_rules only (not cumulative validated_rules) so the
    # rate reflects current-iteration quality, not accumulated history.
    error_rate = len(errors) / total if total > 0 else 0.0

    logger.info(
        f"Validator result — error rate: {error_rate:.1%} "
        f"({len(errors)}/{total} rules this iteration), iteration: {iteration}/{MAX_ITERATIONS}"
    )

    if error_rate > VALIDATION_ERROR_THRESHOLD and iteration < MAX_ITERATIONS:
        logger.warning(
            f"Error rate {error_rate:.1%} exceeds threshold "
            f"{VALIDATION_ERROR_THRESHOLD:.0%}. "
            f"Looping back to Extractor (iteration {iteration + 1})."
        )
        return "extractor"

    if iteration >= MAX_ITERATIONS:
        logger.warning(
            f"Max iterations ({MAX_ITERATIONS}) reached. "
            "Proceeding to Builder with current validated rules."
        )

    return "builder"
