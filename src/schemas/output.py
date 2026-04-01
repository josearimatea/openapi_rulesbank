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
    """Provenance and run configuration for a rules bank generation run."""

    source_document:     str = Field(description="Absolute path to the main 3GPP document.")
    generated_at:        str = Field(description="ISO-8601 UTC timestamp of generation.")
    model:               str = Field(description="LLM model identifier used for extraction.")
    total_rules:         int = Field(description="Number of validated rules in this file.")
    sections_planned:    int = Field(description="Number of sections selected by the Planner.")
    sections_with_rules: int = Field(description="Number of sections that produced at least one rule.")
    iterations_run:      int = Field(description="Number of Extractor→Validator loop iterations.")


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
