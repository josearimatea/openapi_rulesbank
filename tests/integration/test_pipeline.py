# tests/integration/test_pipeline.py

"""
Integration test for the full Rules Bank pipeline.

Runs the complete LangGraph flow (Reader → Planner → Extractor → Reflector →
Validator → Builder) against the standard test document (28532-i00.md) and
verifies the output JSON is well-formed and internally consistent.

Makes real LLM calls — run explicitly with:
    pytest tests/integration/test_pipeline.py -v

Do NOT include in regular test runs to avoid unintended token usage.
"""

import json
import os

import pytest

from config.paths import TSPEC_DATA_TEST_FILE
from graph.rule_bank_flow import get_compiled_graph


@pytest.fixture(scope="module")
def pipeline_output():
    """
    Runs the full pipeline once and returns the parsed output JSON.
    scope="module" ensures the pipeline runs only once for all tests in this file.
    """
    initial_state = {
        "main_doc_path":       os.path.abspath(TSPEC_DATA_TEST_FILE),
        "auxiliary_doc_paths": [],
        "validated_rules":     [],
        "validation_errors":   [],
        "iteration_count":     0,
        "messages":            [],
    }

    pipeline = get_compiled_graph()
    result = pipeline.invoke(initial_state)

    output_path = result.get("final_output_path", "")
    assert output_path and os.path.isfile(output_path), (
        f"Pipeline did not produce a valid output file: {output_path}"
    )

    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.integration
def test_output_top_level_structure(pipeline_output):
    """Output JSON has the expected top-level keys."""
    assert set(pipeline_output.keys()) == {"metadata", "summary", "rules"}


@pytest.mark.integration
def test_metadata_fields(pipeline_output):
    """All required metadata fields are present."""
    meta = pipeline_output["metadata"]
    required = (
        "source_document", "generated_at", "model", "total_rules", "iterations_run",
        "sections_total", "sections_for_planner", "sections_planned", "sections_with_rules",
        "excluded_by_reader_count", "excluded_by_planner_count",
        "excluded_by_reader", "excluded_by_planner",
    )
    for field in required:
        assert field in meta, f"Missing metadata field: {field}"


@pytest.mark.integration
def test_section_accounting(pipeline_output):
    """
    sections_total == sections_planned + excluded_by_planner_count + excluded_by_reader_count

    Also verifies that the detail lists match their counts.
    """
    meta       = pipeline_output["metadata"]
    total      = meta["sections_total"]
    planned    = meta["sections_planned"]
    ex_reader  = meta["excluded_by_reader_count"]
    ex_planner = meta["excluded_by_planner_count"]

    assert planned + ex_planner + ex_reader == total, (
        f"Section accounting mismatch: "
        f"{planned} planned + {ex_planner} excluded by planner + "
        f"{ex_reader} excluded by reader = {planned + ex_planner + ex_reader} ≠ {total} total"
    )
    assert len(meta["excluded_by_reader"])  == ex_reader
    assert len(meta["excluded_by_planner"]) == ex_planner


@pytest.mark.integration
def test_produces_rules(pipeline_output):
    """Pipeline extracted at least one validated rule."""
    rules = pipeline_output["rules"]
    meta  = pipeline_output["metadata"]
    summary = pipeline_output["summary"]

    assert len(rules) > 0, "Pipeline produced no rules."
    assert meta["total_rules"]          == len(rules)
    assert summary["total_validated"]   == len(rules)


@pytest.mark.integration
def test_rule_fields(pipeline_output):
    """Every rule contains the required fields."""
    required = {
        "section_id", "section_title", "rule_type", "source_name",
        "rule_text", "openapi_mapping", "validation_passed",
    }
    for i, rule in enumerate(pipeline_output["rules"]):
        missing = required - set(rule.keys())
        assert not missing, f"Rule {i} missing fields: {missing}"


@pytest.mark.integration
def test_summary_counts(pipeline_output):
    """rules_by_type and rules_by_section counts sum to total_rules."""
    rules   = pipeline_output["rules"]
    summary = pipeline_output["summary"]

    assert sum(summary["rules_by_type"].values())    == len(rules)
    assert sum(summary["rules_by_section"].values()) == len(rules)
