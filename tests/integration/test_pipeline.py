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
from datetime import datetime

import pytest

from openapi_rulesbank.config.paths import OUTPUTS_RULES_DIR, TSPEC_DATA_TEST_FILE
from openapi_rulesbank.graph.rule_bank_flow import build_initial_state, get_compiled_graph

# Full-document runs are saved under rules_bank/Final_rules/ to keep them
# separated from scoped/exploratory runs that land directly in rules_bank/.
FINAL_RULES_DIR = os.path.join(OUTPUTS_RULES_DIR, "Final_rules")

# Target service the rules are scoped to. It is the service *name* the rules are
# built for (the ContractBuilder builds the scope contract for it) — not a YAML
# to inject. Run with only the spec (--doc) and this target; no aux/legacy.
TARGET_SERVICE = "ProvMnS"


@pytest.fixture(scope="module")
def pipeline_output(qdrant):
    """
    Runs the full pipeline once and returns the parsed output JSON.
    scope="module" ensures the pipeline runs only once for all tests in this file.
    Depends on the session-scoped `qdrant` fixture (tests/conftest.py) which
    ensures the Qdrant container is running before the pipeline is invoked.
    """
    # Pre-compute the output path so the Builder writes directly into Final_rules/
    # instead of the default rules_bank/ root. Builder reuses final_output_path
    # if it is already set in state — see src/nodes/builder.py.
    os.makedirs(FINAL_RULES_DIR, exist_ok=True)
    doc_stem  = os.path.splitext(os.path.basename(TSPEC_DATA_TEST_FILE))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        FINAL_RULES_DIR, f"rules_bank_{doc_stem}_full_{timestamp}.json"
    )

    # Same starting state the CLI produces (build_initial_state), scoped to
    # TARGET_SERVICE. Only the spec and target are provided — no aux/legacy.
    initial_state = build_initial_state(
        main_doc_path=os.path.abspath(TSPEC_DATA_TEST_FILE),
        target_yaml=TARGET_SERVICE,
    )
    # Pre-set the output path so the Builder writes into Final_rules/ (it reuses
    # final_output_path when already present — see src/nodes/builder.py).
    initial_state["final_output_path"] = output_file

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
        "source_document", "generated_at", "model", "total_rules",
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
