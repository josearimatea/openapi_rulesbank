"""Directed test of the inheritance guard on section 566, without the full pipeline.

Runs Reader -> ContractBuilder, then Extractor -> Reflector -> Validator on ONLY
the 566 "Type NotifyThresholdCrossing" section, with the NotificationHeader
reference memory pre-populated (as the 564 'General' clause would leave it). Reports
at each stage whether the 5 inherited header fields (href, notificationId,
notificationType, eventTime, systemDN) survive as own properties — they should be
gone by the end (allOf $ref + only the 7 own properties).

    uv run python tests/investigation/test_inheritance_566.py
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

INHERITED = {"href", "notificationId", "notificationType", "eventTime", "systemDN"}


def inherited_props(rules):
    return sorted({
        r["source_name"] for r in rules
        if r.get("rule_type") == "schema_property"
        and r.get("openapi_mapping", {}).get("openapi_field", "").startswith("properties.")
        and r["source_name"] in INHERITED
    })


def main():
    r = reader_node({"main_doc_path": TSPEC_DATA_TEST_FILE,
                     "auxiliary_doc_paths": [], "target_yaml": TARGET_YAML})
    secs = r["parsed_sections"]
    cb = contract_builder_node({"parsed_sections": secs, "target_yaml": TARGET_YAML})
    contract = cb["service_contract"]
    spec_coll = cb["spec_collection_name"]
    rules_coll = cb.get("rules_collection_name", "")

    s566 = next(s for s in secs if s["title"].startswith("12.3.1.2.4.2.1"))
    sid = s566["section_id"]

    # Memory as the 564 'General' clause leaves it: NotificationHeader with fields.
    ref_memory = {"NotificationHeader": {
        "spec": "TS 28.623",
        "fields": ["href", "notificationId", "notificationType", "eventTime", "systemDN"],
        "note": "common notification header",
    }}

    plan = {"sections_to_extract": [{
        "section_id": sid, "title": s566["title"], "priority": "high",
        "extraction_focus": "Extract all concrete rules this section defines.",
        "notes": "",
    }]}
    base = {
        **r, "service_contract": contract, "spec_collection_name": spec_coll,
        "rules_collection_name": rules_coll, "extraction_plan": plan,
        "helper_context": "", "validation_errors": [], "validator_section_reflection": [],
        "other_spec_ref_memory": ref_memory, "section_analysis_by_id": {},
        "current_section_idx": 0, "section_iteration_count": 0,
    }

    # --- Extractor ---
    ext = extractor_node(base)
    raw = ext["raw_rules"]
    print(f"EXTRACTOR: {len(raw)} rules | inherited-as-property: {inherited_props(raw)}")
    print("  allOf present?", any(x["openapi_mapping"].get("openapi_field") == "allOf" for x in raw))

    # --- Reflector ---
    ref_state = {**base, **ext, "section_analysis_by_id": ext.get("section_analysis_by_id", {})}
    rfl = reflector_node(ref_state)
    reflected = rfl["reflected_rules"]
    flagged_inh = sorted({
        x["source_name"] for x in reflected
        if x["source_name"] in INHERITED and x.get("discard_suggestion", False)
    })
    print(f"REFLECTOR: discard-suggested among inherited: {flagged_inh or 'none'}")

    # --- Validator ---
    val_state = {**ref_state, **rfl}
    val = validator_node(val_state)
    validated = val["validated_rules"]
    print(f"VALIDATOR: {len(validated)} validated | inherited surviving as property: {inherited_props(validated)}")
    print("  allOf survived?", any(x["openapi_mapping"].get("openapi_field") == "allOf" for x in validated))

    if spec_coll:
        drop_target_spec_collection(spec_coll)
    if rules_coll:
        drop_rules_collection(rules_coll)


if __name__ == "__main__":
    main()
