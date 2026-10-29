"""What the judges reported missing in the PerfMnS notification mapping (28/out/2026).

In the full PerfMnS run of 28/out 23:15, section 12.3.1.2.2.1 (the table
"notifyThresholdCrossing | POST | /notificationSink | M") used all three attempts:
the Reflector and the Validator kept reporting one missing rule, and the retries
produced nothing new. The run logs only the count, never the text. This reruns
that section alone through the REAL graph, with the focus the Planner gave it in
that run, and prints every missing rule the Reflector (Fase 2) and the Validator
(final list) report, attempt by attempt, plus what the Extractor returned.

    uv run python tests/investigation/perfmns/test_missing_rules_perfmns.py \
        > data/outputs/diagnostic/missing_rules_perfmns_<date>.log 2>&1
"""

import sys

from openapi_rulesbank.config.llm_config import usage_tracker
from openapi_rulesbank.graph import rule_bank_flow
from openapi_rulesbank.graph.rule_bank_flow import build_initial_state
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.reader import reader_node

DOC = "data/inputs/3gpp/md/28532-i20.md"
TARGET = "PerfMnS"
PREFIX = "12.3.1.2.2.1 Introduction"
# The focus the Planner wrote for this section in the 23:15 run.
FOCUS = ("Extract the notifyThresholdCrossing notification-to-HTTP mapping, including "
         "the POST operation, concrete resource URI, and any response/status information.")
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def _wrap(name, node):
    def wrapped(state):
        out = node(state)
        attempt = state.get("section_iteration_count", 0)
        if name == "extractor":
            for r in out.get("raw_rules", []):
                m = r.get("openapi_mapping") or {}
                print(f"  [extractor try {attempt}] {r.get('rule_type')} "
                      f"{m.get('openapi_object')} {m.get('openapi_field')} = "
                      f"{str(m.get('openapi_value'))[:60]}")
        elif name == "reflector":
            for sr in out.get("section_reflection", []):
                for m in sr.get("missing_rules", []):
                    print(f"  [reflector try {attempt}] MISSING: {m}")
                if sr.get("reasoning"):
                    print(f"  [reflector try {attempt}] reasoning: {sr['reasoning']}")
        elif name == "validator":
            for sr in out.get("validator_section_reflection", []):
                for m in sr.get("missing_rules", []):
                    print(f"  [validator try {attempt}] FINAL MISSING: {m}")
            for e in out.get("validation_errors", []):
                print(f"  [validator try {attempt}] {e.get('error_type')}: {e.get('instruction')}")
        return out
    return wrapped


def main():
    for name in ("extractor", "reflector", "validator"):
        attr = f"{name}_node"
        setattr(rule_bank_flow, attr, _wrap(name, getattr(rule_bank_flow, attr)))

    base = build_initial_state(main_doc_path=DOC, target_yaml=TARGET)
    base.update(reader_node(base))
    base.update(contract_builder_node(base))
    sec = next(s for s in base["parsed_sections"] if s["title"].startswith(PREFIX))

    for run in range(1, RUNS + 1):
        print("\n" + "=" * 78 + f"\nRUN {run} — [{sec['section_id']}] {sec['title']}")
        state = dict(base)
        state["extraction_plan"] = {"sections_to_extract": [
            {"section_id": sec["section_id"], "title": sec["title"], "priority": "high",
             "extraction_focus": FOCUS, "notes": ""}]}
        state["current_section_idx"] = 0
        graph = rule_bank_flow._build_graph("extractor")
        result = graph.invoke(state, config={"recursion_limit": 100})
        for r in result.get("validated_rules", []):
            m = r["openapi_mapping"]
            print(f"  FINAL {'OK' if r.get('validation_passed', True) else 'XX'} "
                  f"{r['rule_type']} {m['openapi_object']} {m['openapi_field']}")

    u = usage_tracker.as_dict()
    print(f"\nUSAGE — {u['llm_calls']} call(s), {u['total_tokens']} tokens, "
          f"US$ {u['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
