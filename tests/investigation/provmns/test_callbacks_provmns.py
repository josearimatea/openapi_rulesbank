"""Do the ProvMnS notifications still come out as callbacks? (27/out/2026)

The PerfMnS spurious callback (R.6) was fixed by giving every stage one definition
of each rule type (prompts/rule_type_definitions.py): a callback needs an operation
of this API carrying the delivery address, and the address is a runtime value; a
concrete path is a path_operation wherever the resource is hosted. The ProvMnS is
the opposite case and must not regress: its notifications go to
{notificationTarget}, the address carried by the subscription the consumer creates
with PUT, and the official YAML has them as
paths./{className}={id}.put.callbacks.<notification> keyed
'{request.body#/notificationRecipientAddress}'.

Runs the notification clauses through the REAL graph (Extractor -> Reflector ->
Validator -> Builder, retries included), Planner bypassed:

  12.1.1.2.1      Mapping of notifications — {notificationTarget} for each one
  12.1.1.2.2      Notification notifyMOICreation
  12.1.1.3.1.2    Resource structure on the MnS consumer — {notificationTarget} POST
  12.1.1.3.2.4.x  Resource "{notificationTarget}": Description, URI, POST

Expected: callback rules anchored on the PUT, no path_operation on
/{notificationTarget}. Write the log somewhere durable:

    uv run python tests/investigation/provmns/test_callbacks_provmns.py \
        > data/outputs/diagnostic/callbacks_provmns_<date>.log 2>&1
"""

from openapi_rulesbank.config.llm_config import usage_tracker
from openapi_rulesbank.graph.rule_bank_flow import _build_graph, build_initial_state
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.reader import reader_node

DOC = "data/inputs/3gpp/md/28532-i20.md"
TARGET = "ProvMnS"
PREFIXES = [
    "12.1.1.2.1 Introduction",
    "12.1.1.2.2 Notification notifyMOICreation",
    "12.1.1.3.1.2 Resource structure on the MnS consumer",
    "12.1.1.3.2.4.1 Description",
    "12.1.1.3.2.4.2 URI",
    "12.1.1.3.2.4.3.1 POST",
]


def main():
    state = build_initial_state(main_doc_path=DOC, target_yaml=TARGET)
    state.update(reader_node(state))
    state.update(contract_builder_node(state))
    secs = state["parsed_sections"]
    chosen = [next(s for s in secs if s["title"].startswith(p)) for p in PREFIXES]
    state["extraction_plan"] = {"sections_to_extract": [
        {"section_id": s["section_id"], "title": s["title"], "priority": "high",
         "extraction_focus": "Extract all concrete rules this section defines.",
         "notes": ""} for s in chosen]}
    state["current_section_idx"] = 0

    graph = _build_graph("extractor")
    result = graph.invoke(state, config={"recursion_limit": 300})

    print("\n" + "=" * 78)
    print("CALLBACK ANCHOR PER SECTION")
    for s in chosen:
        v = (result.get("section_analysis_by_id") or {}).get(s["section_id"], {})
        print(f"[{s['section_id']}] {s['title'][:55]}")
        print(f"    kind={v.get('section_kind')} anchor={v.get('callback_anchor')!r} "
              f"source={v.get('callback_anchor_source')!r}")

    print("\nFINAL RULES")
    for r in result.get("validated_rules", []):
        m = r["openapi_mapping"]
        ok = "OK" if r.get("validation_passed", True) else "XX"
        print(f"  {ok} [{r.get('section_id')}] {r['rule_type']:15} "
              f"{m['openapi_object'][:62]:62} {m['openapi_field'][:12]:12} "
              f"{str(m['openapi_value'])[:50]}")
        if not r.get("validation_passed", True):
            print(f"       note: {(r.get('validation_notes') or '')[:160]}")

    u = usage_tracker.as_dict()
    print(f"\nUSAGE — {u['llm_calls']} call(s), {u['total_tokens']} tokens, "
          f"US$ {u['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
