"""Directed test of the Phase 3 scope guard, without the full pipeline.

Reproduces the case where section 562 (the POST, sole owner of responses 204/4xx/
5xx) was suppressed: sections 553 and 554 run first and create path/callback/
request_body; then 562 runs. With Phase 3, 562 should still emit its responses —
they are not in the skip list — instead of being self-censored to zero.

Runs Extractor -> Reflector -> Validator for 553, 554, 562 in document order,
carrying validated_rules and the rules collection between them exactly as the
graph does, and reports whether the responses survive.

    uv run python tests/investigation/perfmns/test_scope_responses.py
"""

from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.config.settings import TARGET_YAML
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.extractor import extractor_node
from openapi_rulesbank.nodes.reflector import reflector_node
from openapi_rulesbank.nodes.validator import validator_node
from openapi_rulesbank.tools.rag_tools import (
    drop_target_spec_collection, drop_rules_collection,
)

# 553 = Mapping of notifications (intro), 554 = Notification mapping table,
# 562 = the POST section that owns the responses. Document order.
SEQUENCE = ["12.3.1.2.2.1", "12.3.1.2.2.2", "12.3.1.2.3.2.1.3.1"]


def main():
    r = reader_node({"main_doc_path": TSPEC_DATA_TEST_FILE,
                     "auxiliary_doc_paths": [], "target_yaml": TARGET_YAML})
    secs = r["parsed_sections"]
    cb = contract_builder_node({"parsed_sections": secs, "target_yaml": TARGET_YAML})
    spec_coll = cb["spec_collection_name"]
    rules_coll = cb.get("rules_collection_name", "")

    chosen = [next(s for s in secs if s["title"].startswith(p)) for p in SEQUENCE]
    plan = {"sections_to_extract": [
        {"section_id": s["section_id"], "title": s["title"], "priority": "high",
         "extraction_focus": "Extract all concrete rules this section defines.", "notes": ""}
        for s in chosen
    ]}

    base = {
        **r, "service_contract": cb["service_contract"], "spec_collection_name": spec_coll,
        "rules_collection_name": rules_coll, "extraction_plan": plan, "helper_context": "",
        "other_spec_ref_memory": {}, "section_analysis_by_id": {},
    }

    validated_rules = []           # accumulates across sections (operator.add in the graph)
    for idx, s in enumerate(chosen):
        sid = s["section_id"]
        st = {**base, "current_section_idx": idx, "section_iteration_count": 0,
              "validated_rules": validated_rules, "validation_errors": [],
              "section_reflection": [], "validator_section_reflection": []}
        ext = extractor_node(st)
        st = {**st, **ext}
        rfl = reflector_node(st)
        st = {**st, **rfl}
        val = validator_node(st)
        new = val.get("validated_rules", [])
        validated_rules = validated_rules + new
        kinds = [f"{x['rule_type']}:{x['openapi_mapping'].get('openapi_field','')}" for x in new]
        print(f"[{sid}] {s['title'][:40]:40} -> {len(new)} validated: {kinds}")
        # --- instrumentation: errors (correction/discard) and missing ---
        for e in val.get("validation_errors", []):
            m = e.get("rule", {}).get("openapi_mapping", {})
            print(f"    ERROR [{e.get('error_type')}] "
                  f"{e.get('rule', {}).get('rule_type')}:{m.get('openapi_field','')}"
                  f" -> {e.get('instruction','')[:90]}")
        for sr in val.get("validator_section_reflection", []):
            for mr in sr.get("missing_rules", []):
                print(f"    MISSING [{sr.get('section_id')}] {mr[:100]}")

    resp = [x for x in validated_rules if x["rule_type"] == "response"]
    print()
    print("response rules total:", len(resp),
          "| codes:", sorted(x["openapi_mapping"].get("openapi_field", "") for x in resp))
    print("EXPECT: 204, 4XX, 5XX present")

    drop_target_spec_collection(spec_coll)
    if rules_coll:
        drop_rules_collection(rules_coll)


if __name__ == "__main__":
    main()
