"""Is the Planner's choice of sections stable? (28/out/2026)

Runs Reader + ContractBuilder once, then the Planner N times on the same state,
and prints the contract's stage 3 clause and each plan — selected sections with
their focus and notes; the excluded ones and the reason the Planner gave are in
the log lines "[planner] excluded ...". No rule is extracted: this isolates which
sections enter the plan, which a full run mixes with the variation of every
later stage. Cheap (one contract + N Planner calls).

    uv run python tests/investigation/test_planner_plan.py <N> <TARGET> \
        > data/outputs/diagnostic/planner_<TARGET>_<label>_<date>.log 2>&1
"""

import sys

from openapi_rulesbank.config.llm_config import usage_tracker
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.planner import planner_node
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.tools.rag_tools import drop_rules_collection, drop_target_spec_collection

DOC = "data/inputs/3gpp/md/28532-i20.md"


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    target = sys.argv[2] if len(sys.argv) > 2 else "PerfMnS"

    r = reader_node({"main_doc_path": DOC, "auxiliary_doc_paths": [], "target_yaml": target})
    cb = contract_builder_node({**r, "target_yaml": target, "main_doc_path": DOC})
    c = cb["service_contract"]
    print(f"=== CONTRACT stage3_clause={c.get('stage3_clause')!r}")
    print(f"    out_of_scope={c.get('out_of_scope_signals')}")

    state = {**r, **cb, "target_yaml": target, "helper_context": ""}
    for i in range(runs):
        plan = planner_node(state)["extraction_plan"]["sections_to_extract"]
        print(f"\n=== PLAN {i}: {len(plan)} sections")
        for p in plan:
            print(f"  {p['section_id']:>4} {p['title'][:55]:55} [{p['priority']}]")
            print(f"       focus: {p['extraction_focus'][:220]}")
            if p.get("notes"):
                print(f"       notes: {p['notes'][:220]}")

    u = usage_tracker.as_dict()
    print(f"\nUSAGE {u['llm_calls']} calls US$ {u['cost_usd']:.4f}")
    drop_target_spec_collection(cb.get("spec_collection_name", ""))
    if cb.get("rules_collection_name"):
        drop_rules_collection(cb["rules_collection_name"])


if __name__ == "__main__":
    main()
