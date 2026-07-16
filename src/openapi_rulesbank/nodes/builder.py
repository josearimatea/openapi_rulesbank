# src/nodes/builder.py

"""
Builder Node: writes the rules bank JSON file after each section completes.

No LLM call. No RAG. Called once per section.

HOW IT WORKS:
    Called after every section finishes its retry loop (Extractor → Reflector →
    Validator). On each call it:
      1. Force-includes any remaining semantic errors if the section exhausted its
         retry budget (section_iteration_count >= MAX_ITERATIONS).
      2. Writes (or overwrites) the rules bank JSON with ALL validated rules
         accumulated so far — rules from previous sections plus the current one.
         The file is always a valid, complete snapshot of everything processed
         up to this point.
      3. Advances current_section_idx by 1 and resets section_iteration_count to 0
         so the next section starts fresh.
      4. Clears validation_errors and section_reflection so they don't bleed into
         the next section's first Extractor call.

    First call: generates the output filename (with timestamp) and creates the file.
    Subsequent calls: reuses the same path (final_output_path already set in state)
    and overwrites the file with the updated content.

OUTPUT JSON STRUCTURE:
    metadata — source_document, generated_at, model, total_rules,
               sections_total, sections_for_planner, sections_planned,
               sections_with_rules, excluded_by_reader (list), excluded_by_planner (list)
    summary  — rules_validated_count, rules_error_count,
               rules_by_type (count per rule_type, desc),
               rules_by_section (count per section_id, desc)
    rules    — list of ValidatedRule dicts (full fields including confidence,
               reasoning, flagged, rule_type, openapi_mapping, validation_notes)

State reads:
    main_doc_path              (str)        — path to the main 3GPP document
    validated_rules            (list[dict]) — all rules that passed validation (all sections so far)
    validation_errors          (list[dict]) — rules still failing after the last attempt
    section_iteration_count    (int)        — how many attempts this section used
    current_section_idx        (int)        — index of the section just completed
    extraction_plan            (dict)       — used to count sections that were planned
    parsed_sections            (list[dict]) — used for sections_for_planner count
    excluded_sections_reader   (list[dict]) — sections removed by reader-level filters
    excluded_sections_planner  (list[dict]) — sections not selected by Planner
    final_output_path          (str)        — existing output path (empty on first call)

State writes:
    final_output_path       (str)       — absolute path to the rules bank JSON file
    current_section_idx     (int)       — advanced by 1 (next section to process)
    section_iteration_count (int)       — reset to 0
    validation_errors       (list[dict])— cleared to [] (fresh start for next section)
    section_reflection      (list[dict])— cleared to [] (fresh start for next section)
"""

import json
import os
from collections import Counter
from datetime import datetime

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.paths import OUTPUTS_RULES_DIR
from openapi_rulesbank.config.settings import DIAGNOSTIC_MODE, MODEL, MAX_ITERATIONS
from openapi_rulesbank.graph.state import RuleBankState
from openapi_rulesbank.schemas.output import RulesBank, RulesBankMetadata, RulesBankSummary, RuleOutput, RULE_OUTPUT_FIELDS
from openapi_rulesbank.schemas.rules import ValidatedRule
from openapi_rulesbank.tools.rag_tools import drop_target_spec_collection

logger = get_logger(__name__)

# Fields stripped from a failed rule before constructing a forced ValidatedRule.
_FORCE_EXCLUDE_FIELDS = {
    "missing_rules",
    "split_suggestion",
    "discard_suggestion",
    "reflection_rag_context",
    "validation_notes",
}


def builder_node(state: RuleBankState) -> dict:
    """
    Writes or updates the rules bank JSON after the current section completes.
    """
    logger.info("Builder Node started.")

    validated_rules         = list(state.get("validated_rules",   []) or [])
    validation_errors       = state.get("validation_errors", []) or []
    main_doc_path           = state.get("main_doc_path", "unknown")
    section_iteration_count = state.get("section_iteration_count", 0) or 0
    current_section_idx     = state.get("current_section_idx", 0) or 0
    plan_sections           = state.get("extraction_plan", {}).get("sections_to_extract", [])
    parsed_sections         = state.get("parsed_sections", []) or []

    current_section_id = (
        plan_sections[current_section_idx]["section_id"]
        if current_section_idx < len(plan_sections) else "?"
    )

    # Force-include semantic errors when the section exhausted its retry budget.
    # Structural errors are not force-included — they indicate a malformed rule.
    if section_iteration_count >= MAX_ITERATIONS and validation_errors:
        semantic_errors = [e for e in validation_errors if e.get("stage") == "semantic"]
        if semantic_errors:
            logger.warning(
                f"[{current_section_id}] Max attempts reached — checking "
                f"{len(semantic_errors)} semantic error(s) for force-inclusion."
            )
            validated_identity = {
                (r.get("section_id", ""), r.get("rule_type", ""), r.get("source_name", ""),
                 r.get("openapi_mapping", {}).get("openapi_field", ""))
                for r in validated_rules
            }
            force_count = 0
            for err in semantic_errors:
                rule_dict = err.get("rule", {})
                key = (
                    rule_dict.get("section_id",  ""),
                    rule_dict.get("rule_type",   ""),
                    rule_dict.get("source_name", ""),
                    rule_dict.get("openapi_mapping", {}).get("openapi_field", ""),
                )
                if key in validated_identity:
                    continue
                try:
                    forced = ValidatedRule(
                        **{k: v for k, v in rule_dict.items()
                           if k not in _FORCE_EXCLUDE_FIELDS},
                        validation_notes=(
                            f"Force-included at max attempts. Failed: {err.get('instruction', '')}"
                        ),
                        validation_passed=False,
                    )
                    validated_rules.append(forced.model_dump())
                    validated_identity.add(key)
                    force_count += 1
                    logger.debug(
                        f"  Force-included [{rule_dict.get('section_id', '?')}]: "
                        f"{rule_dict.get('rule_text', '')[:60]}"
                    )
                except Exception as e:
                    logger.warning(f"  Skipped malformed rule (could not construct ValidatedRule): {e}")
            if force_count:
                logger.info(f"Force-included {force_count} semantic rule(s).")

    sections_with_rules = sorted({r.get("section_id", "") for r in validated_rules})

    excluded_by_reader  = state.get("excluded_sections_reader",  []) or []
    excluded_by_planner = state.get("excluded_sections_planner", []) or []

    sections_total       = len(parsed_sections) + len(excluded_by_reader)
    sections_for_planner = len(parsed_sections)
    planner_accounted    = len(plan_sections) + len(excluded_by_planner)

    if planner_accounted == sections_for_planner:
        run_scope = "full"
    else:
        run_scope = "partial"

    rules_by_type = dict(sorted(
        Counter(r.get("rule_type",  "unknown") for r in validated_rules).items(),
        key=lambda x: x[1], reverse=True,
    ))
    rules_by_section = dict(sorted(
        Counter(r.get("section_id", "unknown") for r in validated_rules).items(),
        key=lambda x: x[1], reverse=True,
    ))

    now = datetime.now().astimezone()

    rule_outputs = [
        RuleOutput(**{k: v for k, v in r.items() if k in RULE_OUTPUT_FIELDS})
        for r in validated_rules
    ]

    rules_bank = RulesBank(
        metadata=RulesBankMetadata(
            source_document=           main_doc_path,
            generated_at=              now.isoformat(),
            model=                     MODEL,
            total_rules=               len(rule_outputs),
            sections_total=            sections_total,
            sections_for_planner=      sections_for_planner,
            sections_planned=          len(plan_sections),
            sections_with_rules=       len(sections_with_rules),
            excluded_by_reader_count=  len(excluded_by_reader),
            excluded_by_planner_count= len(excluded_by_planner),
            excluded_by_reader=        excluded_by_reader,
            excluded_by_planner=       excluded_by_planner,
        ),
        summary=RulesBankSummary(
            rules_validated_count= len([r for r in rule_outputs if r.validation_passed]),
            rules_error_count=     len([r for r in rule_outputs if not r.validation_passed]),
            rules_by_type=         rules_by_type,
            rules_by_section=      rules_by_section,
        ),
        rules=rule_outputs,
    )

    # Reuse the output file from the first call; generate a new name only once.
    existing_path = state.get("final_output_path", "") or ""
    if existing_path:
        output_path = existing_path
    else:
        doc_stem  = os.path.splitext(os.path.basename(main_doc_path))[0]
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename  = f"rules_bank_{doc_stem}_{run_scope}_{timestamp}.json"
        os.makedirs(OUTPUTS_RULES_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUTS_RULES_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rules_bank.model_dump(), f, ensure_ascii=False, indent=2)
        f.flush()

    sections_done = current_section_idx + 1
    logger.info(
        f"[{current_section_id}] Builder wrote {len(validated_rules)} rule(s) to '{output_path}' "
        f"({sections_done}/{len(plan_sections)} section(s) done)."
    )

    # Save the diagnostic report only on the last section to produce a single file
    # covering the full run. Saving after each section would create one file per section.
    is_last_section = (current_section_idx + 1) >= len(plan_sections)
    if DIAGNOSTIC_MODE and is_last_section:
        from utils.diagnostic import _diagnostic
        if _diagnostic is not None:
            doc_stem = os.path.splitext(os.path.basename(main_doc_path))[0]
            diag_path = _diagnostic.save(doc_stem)
            logger.info(f"Diagnostic report saved to '{diag_path}'.")

    # Drop the temporary spec RAG collection once the run is done, so it does not
    # accumulate in Qdrant. Non-fatal — a leftover collection is harmless.
    if is_last_section:
        spec_collection_name = state.get("spec_collection_name", "")
        if spec_collection_name:
            try:
                drop_target_spec_collection(spec_collection_name)
            except Exception as e:
                logger.warning(f"Could not drop spec collection '{spec_collection_name}': {e}")

    # Rules that were force-included exist only in the local validated_rules list —
    # they were never returned by the Validator so operator.add never saw them.
    # Return them here so they are accumulated into the state for subsequent sections.
    state_rules   = state.get("validated_rules") or []
    state_rule_set = {
        (r.get("section_id",""), r.get("rule_type",""), r.get("source_name",""),
         r.get("openapi_mapping",{}).get("openapi_field",""))
        for r in state_rules
    }
    force_included = [
        r for r in validated_rules
        if (r.get("section_id",""), r.get("rule_type",""), r.get("source_name",""),
            r.get("openapi_mapping",{}).get("openapi_field","")) not in state_rule_set
    ]

    result = {
        "final_output_path":            output_path,
        "current_section_idx":          current_section_idx + 1,
        "section_iteration_count":      0,
        "validation_errors":            [],
        "section_reflection":           [],
        "validator_section_reflection": [],
    }
    if force_included:
        result["validated_rules"] = force_included
    return result
