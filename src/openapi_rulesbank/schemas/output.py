# src/schemas/output.py

"""
Pydantic models for the final rules bank output file produced by the Builder Node.

Validates the complete output structure before it is written to disk,
ensuring metadata, summary, and rules are well-formed.

Classes:
    RulesBankMetadata — provenance and run configuration
    RulesBankSummary  — statistics computed from the validated rules
    RuleOutput        — clean representation of one rule for the output file
    RulesBank         — top-level model wrapping metadata, summary and rules

Constants:
    RULE_OUTPUT_FIELDS — set of ValidatedRule fields written to the final JSON;
                         internal pipeline fields (rag_context, etc.) are excluded
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Fields of ValidatedRule that appear in the rules bank output.
# Internal pipeline fields (reflection_rag_context, etc.) are intentionally
# omitted — they are only used within the pipeline, not in the final artifact.
# ---------------------------------------------------------------------------

RULE_OUTPUT_FIELDS = {
    "section_id",
    "section_title",
    "rule_type",
    "source_name",
    "rule_text",
    "openapi_mapping",
    "reflection_confidence",
    "reflection_reasoning",
    "reflection_flagged",
    "split_suggestion",
    "discard_suggestion",
    "validation_notes",
    "validation_passed",
}


class RuleOutput(BaseModel):
    """
    Representation of a validated rule in the final rules bank JSON.
    Contains only fields relevant to the user — internal pipeline fields are excluded.
    """

    section_id:            str   = Field(...)
    section_title:         str   = Field(...)
    rule_type:             str   = Field(...)
    source_name:           str   = Field(...)
    rule_text:             str   = Field(...)
    openapi_mapping:       dict  = Field(...)
    reflection_confidence: float = Field(...)
    reflection_reasoning:  str   = Field(default="")
    reflection_flagged:    bool  = Field(default=False)
    split_suggestion:      str   = Field(default="")
    discard_suggestion:    bool  = Field(default=False)
    validation_notes:      str   = Field(default="")
    validation_passed:     bool  = Field(default=True)


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

    rules_validated_count: int = Field(description="Rules that passed all validation stages (validation_passed=True).")
    rules_error_count:     int = Field(description="Rules force-included at max iterations with validation_passed=False.")
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
        rules    — list of RuleOutput objects (user-facing fields only)
    """

    metadata: RulesBankMetadata
    summary:  RulesBankSummary
    rules:    list[RuleOutput]
