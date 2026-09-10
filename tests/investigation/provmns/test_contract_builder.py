"""What the ContractBuilder puts in the scope contract, and what it leaves out.

The contract is the only place the pipeline states what the target service is, so
a wrong entry in it is not a local error: every node that reads it goes looking
for something the target YAML does not contain. A ProvMnS contract was found
listing notifyEvent among the service's notifications — that name belongs to the
ONAP VES solution set, a sibling clause the target YAML does not cover, so the
Extractor emitting it was following the contract rather than inventing.

Wrong is the failure that matters here; incomplete is not. A capability the
builder could not identify only leaves the later steps with less, and the
contract now says so through capabilities_complete instead of presenting the list
as closed.

So this runs the ContractBuilder alone and prints the contract, checking each
capability against the notifications and operations the service's own solution
set defines. It asserts nothing about how many were found — only about whether
what was found belongs.

    uv run python tests/investigation/provmns/test_contract_builder.py
"""

from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.config.settings import TARGET_YAML
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.tools.rag_tools import (
    drop_target_spec_collection, drop_rules_collection,
)

# What the target YAML actually covers, read off TS28532_ProvMnS.yaml: the four
# notifications it declares as callbacks, and the four IS operations its clauses
# map to HTTP methods. Anything else the contract claims is worth reading twice.
EXPECTED_NOTIFICATIONS = {
    "notifyMOICreation", "notifyMOIDeletion",
    "notifyMOIAttributeValueChanges", "notifyMOIChanges",
}
EXPECTED_OPERATIONS = {
    "createMOI", "getMOIAttributes", "modifyMOIAttributes", "deleteMOI",
}
# Declared by the abstract stage 2 clauses of THIS service, but not mapped by the
# stage 3 mapping tables — so real capabilities that the target YAML does not
# contain. They belong in the contract with bound_in_stage3=False, not dropped
# and not passed off as extractable.
UNBOUND = {
    "notifyEvent": "declared in 11.1.1.10, absent from the 12.1.1.2.1-1 mapping table",
}
# Names belonging to a different Management Service. These do not belong at all.
FOREIGN = {
    "notifyFileReady": "FileDataReportingMnS",
    "notifyThresholdCrossing": "PerfMnS",
    "notifyNewAlarm": "FaultMnS",
    "subscribe": "another service's subscription operation",
    "unsubscribe": "another service's subscription operation",
}


def main():
    print(f"=== ContractBuilder for {TARGET_YAML} ===\n")

    r = reader_node({"main_doc_path": TSPEC_DATA_TEST_FILE,
                     "auxiliary_doc_paths": [], "target_yaml": TARGET_YAML})
    cb = contract_builder_node({"parsed_sections": r["parsed_sections"],
                                "target_yaml": TARGET_YAML})
    contract = cb["service_contract"]

    print(f"service_name : {contract.get('service_name', '')}")
    print(f"purpose      : {(contract.get('purpose', '') or '')[:150]}")
    print(f"complete?    : {contract.get('capabilities_complete', False)}"
          "   (False is the expected answer — the list is what it identified)")
    print()

    caps = contract.get("capabilities", []) or []
    print(f"capabilities ({len(caps)}):")
    foreign_found, mismarked, expected_found = [], [], set()
    for c in caps:
        name, kind = c.get("name", ""), c.get("type", "")
        bound = c.get("bound_in_stage3", True)
        note = ""
        if name in FOREIGN:
            note = f"   <-- ANOTHER SERVICE: {FOREIGN[name]}"
            foreign_found.append(name)
        elif name in UNBOUND:
            # The point of the flag: kept, but marked as outside the YAML.
            if bound:
                note = f"   <-- MARKED AS BOUND, BUT: {UNBOUND[name]}"
                mismarked.append(name)
            else:
                note = "   (correctly flagged: in the service, not in this YAML)"
        elif name in EXPECTED_NOTIFICATIONS or name in EXPECTED_OPERATIONS:
            expected_found.add(name)
            if not bound:
                note = "   <-- FLAGGED UNBOUND, BUT THE YAML DOES CONTAIN IT"
                mismarked.append(name)
        print(f"  {kind:14} {'bound' if bound else 'unbound':8} {name}{note}")

    missing = (EXPECTED_NOTIFICATIONS | EXPECTED_OPERATIONS) - expected_found
    print()
    print(f"of the {len(EXPECTED_NOTIFICATIONS | EXPECTED_OPERATIONS)} the YAML "
          f"covers, {len(expected_found)} are listed"
          + (f"; not listed: {sorted(missing)}" if missing else ""))
    print("  (a gap here is acceptable — the contract says the list is partial)")

    print()
    print(f"stage_scope  : {(contract.get('stage_scope', '') or '')[:200]}")
    print(f"out_of_scope : {contract.get('out_of_scope_signals', [])}")

    print()
    print("=" * 74)
    if foreign_found:
        print(f"DEFECT: {len(foreign_found)} capability(ies) of another service: "
              f"{foreign_found}")
    if mismarked:
        print(f"DEFECT: {len(mismarked)} capability(ies) whose bound_in_stage3 does "
              f"not match the stage 3 mapping tables: {mismarked}")
        print("A capability marked bound is read as being in the target YAML.")
    if not foreign_found and not mismarked:
        print("Every capability belongs to this service, and each is marked "
              "according to whether stage 3 binds it.")

    drop_target_spec_collection(cb.get("spec_collection_name", ""))
    drop_rules_collection(cb.get("rules_collection_name", ""))


if __name__ == "__main__":
    main()
