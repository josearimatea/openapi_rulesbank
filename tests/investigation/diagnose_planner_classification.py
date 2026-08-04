# tests/investigation/diagnose_planner_classification.py

"""
Diagnostic (not a pytest test): inspect how the Planner classifies sections
for the service in settings.TARGET_YAML.

Runs Reader -> ContractBuilder -> Planner and prints:
  1. the service contract (stage scope the Planner is told to respect);
  2. selected sections grouped by top-level clause;
  3. EXCLUDED sections under a stage-3 clause (12.x) with a body preview, so we
     can tell apart the two failure modes:
       - dropped a real subsection that carries rules  -> under-inclusion (BAD)
       - dropped an empty umbrella/heading clause       -> correct
  4. any SELECTED section whose body is empty/heading-only (the risk reopened by
     reverting the deterministic heading filter).

Makes real LLM calls and uses RAG. Requires a running Qdrant on localhost:6333
and OPENAI_API_KEY in .env. Run directly:

    uv run python tests/investigation/diagnose_planner_classification.py
"""

from collections import Counter

from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.config.settings import TARGET_YAML
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.planner import planner_node
from openapi_rulesbank.schemas.contract import format_contract_summary
from openapi_rulesbank.tools.rag_tools import drop_target_spec_collection


def clause2(title: str) -> str:
    """Top-level clause of a section title, e.g. '12.3.1.2 Foo' -> '12.3'."""
    parts = title.strip().split(" ", 1)[0].split(".")
    return ".".join(parts[:2])


def main() -> None:
    target = TARGET_YAML
    print(f"=== TARGET_YAML = {target} ===\n")

    reader_result = reader_node({
        "main_doc_path":       TSPEC_DATA_TEST_FILE,
        "auxiliary_doc_paths": [],
        "target_yaml":         target,
    })
    sections = reader_result["parsed_sections"]
    sections_by_id = {s["section_id"]: s for s in sections}
    print(f"Reader: {len(sections)} sections parsed.\n")

    contract_result = contract_builder_node({
        "parsed_sections": sections,
        "target_yaml":     target,
    })
    contract = contract_result["service_contract"]
    spec_collection = contract_result.get("spec_collection_name", "")
    print("----- SERVICE CONTRACT -----")
    print(format_contract_summary(contract))
    print()

    planner_result = planner_node({**reader_result, "service_contract": contract})
    plan = planner_result["extraction_plan"]
    selected = plan["sections_to_extract"]
    excluded = planner_result.get("excluded_sections_planner", [])

    def body_preview(section_id: str, n: int = 80) -> str:
        body = sections_by_id.get(section_id, {}).get("content", "").strip()
        return body[:n].replace("\n", " ") if body else "(EMPTY / heading-only)"

    print(f"=== SELECTED {len(selected)} | EXCLUDED {len(excluded)} ===\n")

    print("Selected sections by top-level clause:")
    for clause, n in sorted(Counter(clause2(s["title"]) for s in selected).items()):
        print(f"  {clause}: {n}")

    print(f"\n----- SELECTED sections (id | priority | title) -----")
    for s in sorted(selected, key=lambda x: x["section_id"]):
        print(f"[{s['section_id']:>4}] ({s.get('priority','?'):>6}) {s['title'][:60]}")

    print(f"\n----- EXCLUDED under a 12.x clause (candidates for under-inclusion) -----")
    shown = 0
    for e in sorted(excluded, key=lambda x: x["section_id"]):
        if clause2(e["title"]).startswith("12."):
            print(f"[{e['section_id']:>4}] {e['title'][:56]:56} | {body_preview(e['section_id'])}")
            shown += 1
    if not shown:
        print("  (none — nothing under 12.x was excluded)")

    print(f"\n----- SELECTED but EMPTY/heading-only (should be NONE) -----")
    empties = [s for s in selected if body_preview(s["section_id"]).startswith("(EMPTY")]
    for s in empties:
        print(f"  [{s['section_id']}] {s['title'][:70]}")
    print(f"  -> {len(empties)} empty section(s) selected.")

    if spec_collection:
        drop_target_spec_collection(spec_collection)
        print(f"\nDropped temporary spec collection: {spec_collection}")


if __name__ == "__main__":
    main()
