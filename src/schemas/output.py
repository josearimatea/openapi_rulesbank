# src/schemas/output.py

"""
Pydantic models for the final rules bank output file produced by the Builder Node.

Validates the complete output structure before it is written to disk,
ensuring metadata, summary, and rules are well-formed.

Classes:
    RulesBankMetadata — provenance and run configuration
    RulesBankSummary  — statistics computed from the validated rules
    RulesBank         — top-level model wrapping metadata, summary and rules
"""

from pydantic import BaseModel, Field
from schemas.rules import ValidatedRule


class RulesBankMetadata(BaseModel):
    """
    Provenance and run configuration for a rules bank generation run.

    Section accounting (verified by Builder at runtime — must hold):
        sections_total == sections_planned + excluded_by_planner_count + excluded_by_reader_count

    Pipeline stages:
        Document raw splits  (sections_total)
          └─ Reader filters  → excluded_by_reader_count removed
               └─ sections_for_planner  sent to Planner
                    └─ Planner filters  → excluded_by_planner_count removed
                         └─ sections_planned  sent to Extractor
    """

    source_document:  str = Field(description="Absolute path to the main 3GPP document.")
    generated_at:     str = Field(description="ISO-8601 UTC timestamp of generation.")
    model:            str = Field(description="LLM model identifier used for extraction.")
    total_rules:      int = Field(description="Number of validated rules in this file.")
    iterations_run:   int = Field(description="Number of Extractor→Validator loop iterations.")

    # --- Section accounting (integers for easy verification) ---
    sections_total:            int = Field(description="Total sections split from the document before any filtering.")
    sections_for_planner:      int = Field(description="Sections that passed reader filters and were sent to the Planner.")
    sections_planned:          int = Field(description="Sections selected by the Planner for extraction.")
    sections_with_rules:       int = Field(description="Sections that produced at least one validated rule.")
    excluded_by_reader_count:  int = Field(description="Count of sections removed by reader-level filters.")
    excluded_by_planner_count: int = Field(description="Count of sections the Planner did not select.")

    # --- Excluded section details (full list for traceability) ---
    excluded_by_reader:  list[dict] = Field(
        description=(
            "Sections excluded by reader-level filters (symbolic titles, keyword filter). "
            "Each entry: {section_id, title, reason}."
        ),
        default_factory=list,
    )
    excluded_by_planner: list[dict] = Field(
        description=(
            "Sections the Planner did not select for extraction. "
            "Each entry: {section_id, title}."
        ),
        default_factory=list,
    )


class RulesBankSummary(BaseModel):
    """
    Statistics computed from the validated rules and final validation errors.
    Embedded in the output file to give a quick overview of the extraction run.
    """

    total_validated:   int = Field(description="Total number of rules that passed all validation stages.")
    final_error_count: int = Field(description="Number of rules still failing after the last iteration.")
    rules_by_type:     dict[str, int] = Field(
        description="Count of validated rules per rule_type, sorted by count descending."
    )
    rules_by_section:  dict[str, int] = Field(
        description="Count of validated rules per section_id, sorted by count descending."
    )


class RulesBank(BaseModel):
    """
    Final output of the rules bank pipeline.

    Produced by: Builder Node
    Saved as   : data/outputs/rules_bank/rules_bank_<doc>_<timestamp>.json

    Structure:
        metadata — source document, model, timestamps, section counts
        summary  — rule counts broken down by type and section
        rules    — full list of ValidatedRule objects
    """

    metadata: RulesBankMetadata
    summary:  RulesBankSummary
    rules:    list[ValidatedRule]
