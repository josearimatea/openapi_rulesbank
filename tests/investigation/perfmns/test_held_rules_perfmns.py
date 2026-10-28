"""Why the PerfMnS bank regressed after the held-rule mechanism (20/out/2026).

The full PerfMnS run of 20/out lost three things the September runs had: the allOf
NotificationHeader base of NotifyThresholdCrossing (its five header fields came out
as own properties), the `hysteresis` property, and correct PerfMetricValue /
PerfMetricDirection properties (force-included as failed, while a completed copy
was dropped by the Builder's dedup). Its log was lost, so this reruns the five
sections involved through the REAL graph — Extractor -> Reflector -> Validator ->
Builder, retries included — and prints what each one concluded:

  242  Mapping of notifications       — POST /notificationSink: path, not callback
  248  Resource Description           — "a resource on a MnS consumer" (no rule)
  249  Resource URI                   — "provided by the notification subscription"
  251  POST /notificationSink         — 4XX/5XX written as "Error-Response"
  253  General (data types tables)    — should record NotificationHeader + fields
  255  Type NotifyThresholdCrossing   — should find the allOf base; holds PerfMetric*
  261  Simple data types              — defines PerfMetricValue (settles it local)
  263  Enumeration PerfMetricDirection

The Planner is bypassed (plan fixed to these sections) so the run is cheap.
Write the log somewhere durable:

    uv run python tests/investigation/perfmns/test_held_rules_perfmns.py \
        > data/outputs/diagnostic/held_rules_perfmns_<date>.log 2>&1
"""

from openapi_rulesbank.config.llm_config import usage_tracker
from openapi_rulesbank.graph.rule_bank_flow import _build_graph, build_initial_state
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.reader import reader_node

DOC = "data/inputs/3gpp/md/28532-i20.md"
TARGET = "PerfMnS"
PREFIXES = [
    # The notification mapping and the consumer-side resource description: where
    # the spurious callback came from (R.6), and what the Planner picked.
    "12.3.1.2.2.1 Introduction",
    "12.3.1.2.3.2.1.1 Description",
    "12.3.1.2.3.2.1.2 URI",
    "12.3.1.2.3.2.1.3.1 POST",
    "12.3.1.2.4.1 General",
    "12.3.1.2.4.2.1 Type NotifyThresholdCrossing",
    "12.3.1.2.4.6.2 Simple data types",
    "12.3.1.2.4.6.4 Enumeration PerfMetricDirection",
]
WATCH = ["NotificationHeader", "ErrorResponse", "Error-Response",
         "PerfMetricValue", "PerfMetricDirection", "NotificationType"]


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
    print("PRE-ANALYSIS VERDICT PER SECTION")
    for s in chosen:
        v = (result.get("section_analysis_by_id") or {}).get(s["section_id"], {})
        print(f"[{s['section_id']}] {s['title'][:50]}")
        print(f"    kind={v.get('section_kind')} base_ref={v.get('base_ref')!r}")
        print(f"    inherited={v.get('inherited_fields')}")
        print(f"    held={v.get('defer_external_refs')}")
        for c in v.get("completed_held_rules", []) or []:
            print(f"    emitted here: {c}")

    print("\nREFERENCE MEMORY AT THE END (watched types)")
    mem = result.get("other_spec_ref_memory") or {}
    for t in WATCH:
        e = mem.get(t)
        if e is None:
            print(f"  {t:22} — not in memory")
            continue
        pend = [(p.get("section_id"), p["rule"].get("source_name"), p.get("completed_in") or "OPEN")
                for p in e.get("pending_rules", [])]
        print(f"  {t:22} origin={e.get('origin')} spec={e.get('spec')!r} "
              f"file={e.get('file')!r} fields={e.get('fields')}")
        if pend:
            print(f"  {'':22} held rules (from, name, completed_in): {pend}")

    print("\nFINAL RULES")
    for r in result.get("validated_rules", []):
        m = r["openapi_mapping"]
        ok = "OK" if r.get("validation_passed", True) else "XX"
        print(f"  {ok} [{r.get('section_id')}] {r['rule_type']:15} "
              f"{m['openapi_field'][:32]:32} {str(m['openapi_value'])[:60]}")
        if not r.get("validation_passed", True):
            print(f"       note: {(r.get('validation_notes') or '')[:160]}")

    u = usage_tracker.as_dict()
    print(f"\nUSAGE — {u['llm_calls']} call(s), {u['total_tokens']} tokens, "
          f"US$ {u['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
