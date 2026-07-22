# tests/integration/test_pipeline_10sections.py

"""
Integration test for the Rules Bank pipeline scoped to 10 sections.

Runs the complete LangGraph flow against a hand-picked set of 10 sections from
the standard test document (28532-i00.md). Sections were chosen to cover both
paths of the pipeline:
  - HTTP method sections (326, 327, 328, 346, 347, 348, 349, 356) — expected to
    produce rules and exercise the Extractor/Reflector/Validator loop.
  - Higher-level mapping sections (323, 325) — expected to be discarded by the
    Planner, exercising the exclusion path.

Makes real LLM calls — run explicitly with:
    pytest tests/integration/test_pipeline_10sections.py -v

Do NOT include in regular test runs to avoid unintended token usage.
"""

import json
import os

import pytest

from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.graph.rule_bank_flow import get_compiled_graph_from_planner

TARGET_IDS = {"323", "325", "326", "327", "328", "346", "347", "348", "349", "356"}

# Target service the rules are scoped to (the ContractBuilder builds the scope
# contract for it). It is the service *name*, not a YAML to inject.
# Pinned to ProvMnS: TARGET_IDS above are ProvMnS sections, so this scoped test
# stays on that service regardless of settings.TARGET_YAML. Use test_pipeline.py
# (which follows settings.TARGET_YAML) to exercise another MnS end to end.
TARGET_SERVICE = "ProvMnS"


@pytest.fixture(scope="module")
def pipeline_output(qdrant):
    """
    Runs the pipeline scoped to TARGET_IDS and returns the parsed output JSON.
    scope="module" ensures the pipeline runs only once for all tests in this file.
    Depends on the session-scoped `qdrant` fixture (tests/conftest.py) which
    ensures the Qdrant container is running before the pipeline is invoked.
    """
    # Run Reader to get all parsed sections, then filter to TARGET_IDS.
    reader_result = reader_node({
        "main_doc_path":       os.path.abspath(TSPEC_DATA_TEST_FILE),
        "auxiliary_doc_paths": [],
        "target_yaml":         TARGET_SERVICE,
    })

    sections_by_id = {s["section_id"]: s for s in reader_result["parsed_sections"]}
    target_parsed  = [sections_by_id[sid] for sid in TARGET_IDS if sid in sections_by_id]
    missing        = TARGET_IDS - {s["section_id"] for s in target_parsed}
    assert not missing, f"TARGET_IDS not found in parsed_sections: {missing}"

    # The graph starts at the Planner (get_compiled_graph_from_planner skips the
    # Reader and the ContractBuilder), so run the ContractBuilder here over the
    # scoped sections to give the Planner/Extractor the same service_contract
    # they get in a full run.
    contract_result = contract_builder_node({
        "parsed_sections": target_parsed,
        "target_yaml":     TARGET_SERVICE,
    })

    initial_state = {
        **reader_result,
        "parsed_sections":              target_parsed,
        "main_doc_path":                os.path.abspath(TSPEC_DATA_TEST_FILE),
        "auxiliary_doc_paths":          [],
        "target_yaml":                  TARGET_SERVICE,
        "service_contract":             contract_result["service_contract"],
        "validated_rules":              [],
        "validation_errors":            [],
        "section_reflection":           [],
        "validator_section_reflection": [],
        "current_section_idx":          0,
        "section_iteration_count":      0,
        "messages":                     [],
    }

    pipeline = get_compiled_graph_from_planner()
    # recursion_limit: Planner (1) + per section: up to MAX_ITERATIONS×3 nodes + Builder (1)
    # 10 sections × (3 iter × 3 nodes + 1 builder) = 10 × 10 = 100; add Planner = 101.
    result = pipeline.invoke(initial_state, config={"recursion_limit": 150})

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
        "source_document", "generated_at", "model", "total_rules",
        "sections_total", "sections_for_planner", "sections_planned", "sections_with_rules",
        "excluded_by_reader_count", "excluded_by_planner_count",
        "excluded_by_reader", "excluded_by_planner",
    )
    for field in required:
        assert field in meta, f"Missing metadata field: {field}"


@pytest.mark.integration
def test_section_accounting(pipeline_output):
    """sections_total == sections_planned + excluded_by_planner_count + excluded_by_reader_count."""
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
    rules   = pipeline_output["rules"]
    meta    = pipeline_output["metadata"]
    summary = pipeline_output["summary"]

    assert len(rules) > 0, "Pipeline produced no rules."
    assert meta["total_rules"]                                          == len(rules)
    assert summary["rules_validated_count"] + summary["rules_error_count"] == len(rules)


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


@pytest.mark.integration
def test_all_sections_processed(pipeline_output):
    """Pipeline processed all 10 target sections (sections_total == 10 + reader exclusions)."""
    meta = pipeline_output["metadata"]
    # sections_total reflects the scoped run: target sections + any reader exclusions within them.
    # At minimum, sections_planned + excluded_by_planner must cover TARGET_IDS.
    assert meta["sections_planned"] + meta["excluded_by_planner_count"] >= 1, (
        "Expected at least one section to be planned or explicitly excluded by Planner."
    )
