# src/nodes/builder.py

"""
Builder Node: assembles the final rules bank JSON from validated rules and
saves it to data/outputs/rules_bank/.

No LLM call. No RAG. Last node in the pipeline.

HOW IT WORKS:
    1. Reads validated_rules accumulated across all iterations from the state.
       Because validated_rules uses operator.add in the state definition,
       rules that passed in iteration 1 are preserved even after loop-back.
    2. Computes a summary: rule counts by rule_type and by section_id.
    3. Validates the complete output structure against schemas.output.RulesBank
       (Pydantic) before writing — ensures the file is always well-formed.
    4. Saves the result as JSON to data/outputs/rules_bank/ with filename:
       rules_bank_<doc_stem>_<full|partial>_<YYYYMMDD_HHMMSS>.json
       "full" if all relevant sections were planned; "partial" otherwise.

OUTPUT JSON STRUCTURE:
    metadata — source_document, generated_at, model, total_rules,
               sections_planned, sections_with_rules, iterations_run
    summary  — total_validated, final_error_count,
               rules_by_type (count per rule_type, desc),
               rules_by_section (count per section_id, desc)
    rules    — list of ValidatedRule dicts (full fields including confidence,
               reasoning, flagged, rule_type, openapi_mapping, validation_notes)

State reads:
    main_doc_path     (str)        — path to the main 3GPP document
    validated_rules   (list[dict]) — all rules that passed validation (all iterations)
    validation_errors (list[dict]) — rules still failing after the last iteration
    iteration_count   (int)        — total loop iterations that ran
    extraction_plan   (dict)       — used to count sections that were planned

State writes:
    final_output_path (str) — absolute path to the saved rules bank JSON file
"""

import json
import os
from collections import Counter
from datetime import datetime

from config import get_logger
from config.paths import OUTPUTS_RULES_DIR
from config.settings import MODEL, MAX_ITERATIONS
from graph.state import RuleBankState
from schemas.output import RulesBank, RulesBankMetadata, RulesBankSummary
from schemas.rules import ValidatedRule

logger = get_logger(__name__)


def builder_node(state: RuleBankState) -> dict:
    """
    Assembles and validates the rules bank, then saves it to disk as JSON.

    Filename format: rules_bank_<doc_stem>_<full|partial>_<YYYYMMDD_HHMMSS>.json
    """
    logger.info("Builder Node started.")

    validated_rules   = list(state.get("validated_rules",   []) or [])
    validation_errors = state.get("validation_errors", []) or []
    main_doc_path     = state.get("main_doc_path", "unknown")
    iteration_count   = state.get("iteration_count", 1) or 1
    plan_sections     = state.get("extraction_plan", {}).get("sections_to_extract", [])
    parsed_sections   = state.get("parsed_sections", []) or []

    # At max iterations, force-include rules still in validation_errors rather than
    # discarding them. Each rule is included as-is with a note explaining why.
    if iteration_count >= MAX_ITERATIONS and validation_errors:
        logger.warning(
            f"Max iterations reached — force-including {len(validation_errors)} "
            "rule(s) that failed validation."
        )
        for error_dict in validation_errors:
            rule_dict = error_dict.get("rule", {})
            reason    = error_dict.get("reason", "")
            try:
                forced = ValidatedRule(
                    **{k: v for k, v in rule_dict.items() if k != "validation_notes"},
                    validation_notes=f"Force-included at max iterations. Failed: {reason}",
                )
                validated_rules.append(forced.model_dump())
                logger.debug(
                    f"  Force-included [{rule_dict.get('section_id', '?')}]: "
                    f"{rule_dict.get('rule_text', '')[:60]}"
                )
            except Exception as e:
                logger.debug(f"  Skipped malformed rule (could not construct ValidatedRule): {e}")
        validation_errors = []

    sections_with_rules = sorted({r.get("section_id", "") for r in validated_rules})

    # Summary — rule counts by rule_type and section_id, sorted by count descending
    rules_by_type    = dict(sorted(
        Counter(r.get("rule_type",  "unknown") for r in validated_rules).items(),
        key=lambda x: x[1], reverse=True,
    ))
    rules_by_section = dict(sorted(
        Counter(r.get("section_id", "unknown") for r in validated_rules).items(),
        key=lambda x: x[1], reverse=True,
    ))

    now        = datetime.now().astimezone()
    run_scope  = "full" if len(plan_sections) >= len(parsed_sections) else "partial"

    # Validate the complete output against the RulesBank schema before saving
    rules_bank = RulesBank(
        metadata=RulesBankMetadata(
            source_document=     main_doc_path,
            generated_at=        now.isoformat(),
            model=               MODEL,
            total_rules=         len(validated_rules),
            sections_planned=    len(plan_sections),
            sections_with_rules= len(sections_with_rules),
            iterations_run=      iteration_count,
        ),
        summary=RulesBankSummary(
            total_validated=   len(validated_rules),
            final_error_count= len(validation_errors),
            rules_by_type=     rules_by_type,
            rules_by_section=  rules_by_section,
        ),
        rules=validated_rules,
    )

    doc_stem  = os.path.splitext(os.path.basename(main_doc_path))[0]
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename  = f"rules_bank_{doc_stem}_{run_scope}_{timestamp}.json"

    os.makedirs(OUTPUTS_RULES_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_RULES_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rules_bank.model_dump(), f, ensure_ascii=False, indent=2)

    logger.info(
        f"Builder Node complete — {len(validated_rules)} rule(s) saved to '{output_path}'."
    )
    for rule_type, count in rules_by_type.items():
        logger.debug(f"  {rule_type}: {count} rule(s)")

    return {"final_output_path": output_path}
