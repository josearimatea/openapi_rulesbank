# src/utils/diagnostic.py

"""
Diagnostic collector for pipeline inspection.

Activated by DIAGNOSTIC_MODE=True in config/settings.py.
Saves a detailed JSON report of every iteration alongside the rules bank output.

The report shows, for each iteration:
  - Counts: total reflected, validated, errors, error_rate
  - Per-rule breakdown: reflector signals + validator outcome
  - Section-level missing-rules feedback

Usage:
  - DiagnosticCollector is instantiated as a module-level singleton in
    validator_node (once per process, only when DIAGNOSTIC_MODE=True).
  - record_iteration() is called at the end of each validator_node run.
  - save() is called by builder_node to write the final report to disk.
"""

import json
import os
from datetime import datetime

from config.paths import OUTPUTS_DIAGNOSTIC_DIR
from config.settings import DIAGNOSTIC_MODE


class DiagnosticCollector:
    """Collects per-iteration validation telemetry for offline inspection."""

    def __init__(self):
        self.iterations: list[dict] = []

    def record_iteration(
        self,
        iteration_num: int,
        reflected_rules: list[dict],
        validation_errors: list[dict],
        validated_rules: list[dict],
        section_feedback: list[dict],
        error_rate: float,
    ) -> None:
        """
        Records a snapshot of one Validator run.

        Args:
            iteration_num    : current iteration number (state["iteration_count"])
            reflected_rules  : rules entering the Validator this iteration
            validation_errors: rules that failed (ValidationError dicts)
            validated_rules  : rules that passed (ValidatedRule dicts)
            section_feedback : SectionFeedback dicts produced this iteration
            error_rate       : fraction of reflected_rules that failed
        """
        # Build an index so each rule can be matched to its error (if any)
        errors_by_key: dict[tuple, dict] = {}
        for err in validation_errors:
            rule = err.get("rule", {})
            key = (
                err.get("section_id") or rule.get("section_id", ""),
                rule.get("rule_type", ""),
                rule.get("source_name", ""),
            )
            errors_by_key[key] = err

        rules_detail = []
        for rule in reflected_rules:
            key = (rule.get("section_id", ""), rule.get("rule_type", ""), rule.get("source_name", ""))
            err = errors_by_key.get(key)
            rules_detail.append({
                "section_id":  rule.get("section_id"),
                "rule_type":   rule.get("rule_type"),
                "source_name": rule.get("source_name"),
                "rule_text":   rule.get("rule_text", "")[:120],
                "reflector": {
                    "confidence":         rule.get("reflection_confidence"),
                    "flagged":            rule.get("reflection_flagged"),
                    "split_suggestion":   rule.get("split_suggestion", ""),
                    "discard_suggestion": rule.get("discard_suggestion", False),
                    "missing_rules":      rule.get("missing_rules", []),
                },
                "validator": {
                    "passed":      err is None,
                    "error_type":  err.get("error_type")  if err else None,
                    "stage":       err.get("stage")        if err else None,
                    "instruction": err.get("instruction")  if err else None,
                },
            })

        self.iterations.append({
            "iteration":        iteration_num,
            "total_reflected":  len(reflected_rules),
            "total_validated":  len(validated_rules),
            "total_errors":     len(validation_errors),
            "error_rate":       round(error_rate, 4),
            "section_feedback": section_feedback,
            "rules":            rules_detail,
        })

    def save(self, doc_stem: str) -> str:
        """
        Writes the collected diagnostic report to disk.

        Args:
            doc_stem : base name of the source document (without extension),
                       used to name the output file.

        Returns:
            Absolute path to the written JSON file.
        """
        os.makedirs(OUTPUTS_DIAGNOSTIC_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            OUTPUTS_DIAGNOSTIC_DIR,
            f"diagnostic_{doc_stem}_{timestamp}.json",
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"iterations": self.iterations}, f, ensure_ascii=False, indent=2)
        return path


# ---------------------------------------------------------------------------
# Module-level singleton — created once per process when DIAGNOSTIC_MODE=True.
# Both validator_node (record_iteration) and builder_node (save) import this.
# ---------------------------------------------------------------------------
_diagnostic: "DiagnosticCollector | None" = DiagnosticCollector() if DIAGNOSTIC_MODE else None
