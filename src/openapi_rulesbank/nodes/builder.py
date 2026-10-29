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
      4. Clears validation_errors, section_reflection, and the section's declared
         gaps with the context fetched for them, so none of it bleeds into the
         next section's first Extractor call.

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
import re
from collections import Counter
from datetime import datetime

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.llm_config import usage_tracker
from openapi_rulesbank.config.paths import final_rules_dir_for, rules_dir_for
from openapi_rulesbank.config.settings import DIAGNOSTIC_MODE, MODEL, MAX_ITERATIONS
from openapi_rulesbank.graph.state import RuleBankState
from openapi_rulesbank.schemas.output import RulesBank, RulesBankMetadata, RulesBankSummary, RuleOutput, RULE_OUTPUT_FIELDS
from openapi_rulesbank.schemas.rules import ValidatedRule
from openapi_rulesbank.tools.rag_tools import (
    drop_rules_collection,
    drop_target_spec_collection,
)

logger = get_logger(__name__)


def _clause_number(section: dict) -> str:
    """Document clause number of an excluded section, read from its title.

    3GPP titles start with the clause number (e.g. "12.3.1.2.4.1 General"), so we
    take that leading dotted-number token. Falls back to the internal section_id
    when the title has no clause number.
    """
    title = (section.get("title") or "").strip()
    head = title.split(" ", 1)[0]
    return head if re.match(r'^\d+(\.\d+)*$', head) else section.get("section_id", "")


def _dump_json_inlining(obj: dict, inline_keys: set[str]) -> str:
    """Serialize obj with indent=2, but render the named list keys on one line.

    Python's json indents every list across many lines. For the excluded-section
    clause lists that is hundreds of near-empty lines, so we keep those specific
    keys compact. Each inline key's value is dumped to a placeholder token first,
    then the token line is replaced by the compact one-line array — the array is
    still real JSON, just not spread out. Other lists are unaffected.
    """
    placeholders: dict[str, str] = {}
    staged = dict(obj)
    metadata = dict(staged.get("metadata", {}))
    for key in inline_keys:
        if key in metadata:
            token = f"__INLINE_{key}__"
            placeholders[token] = json.dumps(metadata[key], ensure_ascii=False)
            metadata[key] = token
    staged["metadata"] = metadata

    text = json.dumps(staged, ensure_ascii=False, indent=2)
    for token, inline in placeholders.items():
        text = text.replace(f'"{token}"', inline)
    return text


# Fields stripped from a failed rule before constructing a forced ValidatedRule.
_FORCE_EXCLUDE_FIELDS = {
    "missing_rules",
    "split_suggestion",
    "discard_suggestion",
    "reflection_rag_context",
    "validation_notes",
}


# Rule types whose openapi_value is part of the rule's identity: the same
# object+field may legitimately recur with a different value, so the value must
# stay in the dedup key. OpenAPI 'content' is a MAP of media types, so one
# operation may carry several request bodies / response bodies that differ only
# in media type (the ProvMnS PATCH has 4 request-body media types, its GET 200
# has 3 response media types). For every OTHER type the address is unique by
# construction — a schema keyword (enum, required, oneOf…) occurs once per
# schema, a callback once per name — so a differing value is only wording of the
# same rule, and keeping it in the key would let a true duplicate survive
# (the [UP]/[DOWN] enum split, the notifyMOIChanges callback repeat).
_VALUE_IS_IDENTITY = {"request_body", "response"}


def _canonical_rule_key(rule: dict) -> tuple:
    """Build a comparison key that ignores harmless wording differences.

    Sections often describe one operation several ways, so the same rule arrives
    with the path written differently ("/{className}={id}" vs the full
    "/{MnSRoot}/ProvMnS/{MnSVersion}/.../{className}={id}") or with the schema name
    in another case. The key therefore uses:
      - the last path segment only, lowercased (path prefixes vary, the tail does not)
      - lowercased field, and the value ONLY for the types where it is part of the
        identity (see _VALUE_IS_IDENTITY) — elsewhere a differing value is the same
        rule written another way and must not keep two copies apart.
    Used ONLY to detect duplicates — the stored rule is never rewritten.
    """
    m = rule.get("openapi_mapping", {}) or {}
    obj = (m.get("openapi_object", "") or "").rstrip("/")
    # Keep the meaningful tail: "paths./a/b/{className}={id}" -> "{classname}={id}"
    tail = obj.split("/")[-1].lower() if "/" in obj else obj.lower()
    rtype = (rule.get("rule_type", "") or "").lower()
    key = [rtype, tail, (m.get("openapi_field", "") or "").lower()]
    if rtype in _VALUE_IS_IDENTITY:
        key.append(str(m.get("openapi_value", "") or "").lower())
    return tuple(key)


def _dedupe_rules(rules: list[dict]) -> list[dict]:
    """Keep the first rule for each canonical key, dropping later repeats."""
    seen: set[tuple] = set()
    unique: list[dict] = []
    for r in rules:
        key = _canonical_rule_key(r)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    dropped = len(rules) - len(unique)
    if dropped:
        logger.info(f"Deduplicated {dropped} rule(s) — {len(unique)} unique rule(s) kept.")
    return unique


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

    # Final review of held rules: when the plan is done and some held rules were
    # never settled, one entry per section that held them is appended to the plan,
    # flagged "held_review". The graph goes on to the Extractor as for any section;
    # the rules are judged, corrected or discarded there (see in_held_review).
    plan_update: dict = {}
    if (current_section_idx + 1) >= len(plan_sections) \
            and not any(s.get("held_review") for s in plan_sections):
        holding: dict[str, str] = {}
        for entry in (state.get("other_spec_ref_memory", {}) or {}).values():
            for held in (entry or {}).get("pending_rules", []) or []:
                if not held.get("completed_in"):
                    holding.setdefault(held.get("section_id", ""), held.get("section_title", ""))
        by_id = {s["section_id"]: s for s in plan_sections}
        review = [
            {**by_id.get(sid, {"section_id": sid, "title": title, "priority": "low",
                               "extraction_focus": ""}),
             "held_review": True}
            for sid, title in holding.items() if sid
        ]
        if review:
            plan_sections = plan_sections + review
            plan_update = {**state.get("extraction_plan", {}), "sections_to_extract": plan_sections}
            logger.info(f"Held rules never settled in section(s) {sorted(holding)} — "
                        "queued for a final review.")

    # Held rules never settled (piece 5): a rule held because the origin of its
    # $ref type was not stated, and that no later section settled, is emitted at the
    # end of the run — marked as not validated — so it does not vanish. The final
    # review takes them first, so this only catches one it never reached.
    if (current_section_idx + 1) >= len(plan_sections):
        orphan_count = 0
        for type_name, entry in (state.get("other_spec_ref_memory", {}) or {}).items():
            for held in (entry or {}).get("pending_rules", []) or []:
                if held.get("completed_in"):
                    continue
                rule_dict = dict(held.get("rule") or {})
                try:
                    orphan = ValidatedRule(
                        **{k: v for k, v in rule_dict.items()
                           if k not in _FORCE_EXCLUDE_FIELDS},
                        reflector_score=0.0,
                        reflection_reasoning=(
                            "Not reviewed: held because no section stated where "
                            f"{type_name!r} is defined."
                        ),
                        reflection_flagged=True,
                        validation_notes=(
                            f"Held rule never settled: no section stated whether "
                            f"{type_name!r} is local or imported; its $ref target "
                            "was left as written."
                        ),
                        validation_passed=False,
                    )
                    validated_rules.append(orphan.model_dump())
                    orphan_count += 1
                except Exception as e:
                    logger.warning(f"  Skipped held rule (could not construct ValidatedRule): {e}")
        if orphan_count:
            logger.info(f"Emitted {orphan_count} held rule(s) never settled, marked not validated.")

    # Final safety net: drop rules that say the same thing. The Validator already
    # discards cross-section duplicates it recognises, but variants can still slip
    # through, so compare on a canonical key. Normalisation is for COMPARISON ONLY —
    # the rule kept keeps its original text, names and path exactly as extracted.
    validated_rules = _dedupe_rules(validated_rules)

    sections_with_rules = sorted({r.get("section_id", "") for r in validated_rules})

    excluded_by_reader  = state.get("excluded_sections_reader",  []) or []
    excluded_by_planner = state.get("excluded_sections_planner", []) or []

    # In the output, list the excluded sections by their document clause number
    # (e.g. "12.3.1.2.4.1"), read from the title, not the internal section_id — the
    # clause number is what a reader recognises. Full dicts (title + reason) bloated
    # the report; the counts plus this compact list are enough to see what dropped.
    excluded_by_reader_clauses  = [_clause_number(s) for s in excluded_by_reader]
    excluded_by_planner_clauses = [_clause_number(s) for s in excluded_by_planner]

    sections_total       = len(parsed_sections) + len(excluded_by_reader)
    sections_for_planner = len(parsed_sections)
    # Final-review entries re-visit planned sections; they are not planned ones.
    planned_sections     = [s for s in plan_sections if not s.get("held_review")]
    planner_accounted    = len(planned_sections) + len(excluded_by_planner)

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
            # Tokens and money spent so far. The Builder runs once per section and
            # rewrites the whole file, so the figure in the finished file is the
            # run's total rather than the last section's.
            usage=                     usage_tracker.as_dict(),
            sections_total=            sections_total,
            sections_for_planner=      sections_for_planner,
            sections_planned=          len(planned_sections),
            sections_with_rules=       len(sections_with_rules),
            excluded_by_reader_count=  len(excluded_by_reader),
            excluded_by_planner_count= len(excluded_by_planner),
            excluded_by_reader=        excluded_by_reader_clauses,
            excluded_by_planner=       excluded_by_planner_clauses,
            held_rules_discarded=      state.get("held_review_discards", []) or [],
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
        # One directory per target service, so runs for different MnS do not mix.
        # Within it, a run that covered the whole document goes to Final_rules/ —
        # that is the result worth keeping — while a partial run (a scoped test,
        # an interrupted run) stays in the service's root folder.
        service = state.get("target_yaml", "") or ""
        out_dir = (
            final_rules_dir_for(service) if run_scope == "full"
            else rules_dir_for(service)
        )
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_dump_json_inlining(
            rules_bank.model_dump(),
            inline_keys={"excluded_by_reader", "excluded_by_planner"},
        ))
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
            diag_path = _diagnostic.save(doc_stem, state.get("target_yaml", "") or "")
            logger.info(f"Diagnostic report saved to '{diag_path}'.")

    # Drop the temporary spec RAG collection once the run is done, so it does not
    # accumulate in Qdrant. Non-fatal — a leftover collection is harmless.
    if is_last_section:
        for name, drop in (
            (state.get("spec_collection_name", ""),  drop_target_spec_collection),
            (state.get("rules_collection_name", ""), drop_rules_collection),
        ):
            if not name:
                continue
            try:
                drop(name)
            except Exception as e:
                logger.warning(f"Could not drop temporary collection '{name}': {e}")

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
        # Both belong to the section just finished: the gaps it declared and the
        # text fetched for them. Cleared with the rest so the next section does
        # not start out answering another section's questions, and so the fetched
        # blocks do not pile up in the state across all sections.
        "missing_context":              [],
        "retrieved_context":            {},
    }
    if force_included:
        result["validated_rules"] = force_included
    if plan_update:
        result["extraction_plan"] = plan_update
    return result
