"""Directed test of the notification-vs-callback decision, without the full pipeline.

The target YAML nests the four notifications as `callbacks` under the PUT of
/{className}={id}, keyed by the runtime expression
'{request.body#/notificationRecipientAddress}'. The full run instead emitted a
path_operation plus response rules on 'paths./{notificationTarget}' — the exact
string the extractor prompt names as wrong.

WHAT IS BEING TESTED. The extractor prompt already carries the rule that decides
this (CALLBACK vs PATH — DECIDE BY WHO OWNS THE URI), naming
'paths./{notificationTarget}' as wrong, and the reference memory already carries
notificationRecipientAddress from section 363. So the definition of a callback is
not what is missing. The one thing no source holds is the ANCHOR the prompt
demands for openapi_object: "paths.<path>.<method>.callbacks.<name>". Section 365
never mentions the PUT that owns it — that lives in section 355 — and a PerfMnS
run once emitted the literal '<owning-path>' from the prompt, which is what a
model does when it accepts the callback but cannot fill the anchor in.

Hypothesis: the anchor, not the concept, is what fails. So section 365 is run
twice over the same inputs, and only the anchor differs:

  A (baseline)  exactly what the pipeline does today.
  B (anchored)  the same, with the owning operation stated in the stage 3 context
                the Extractor already reads.

If a callback rule forms in B and not in A, the anchor is the cause and the fix
belongs where the anchor is resolved. If B also yields a path, the anchor is not
the blocker and the cause is upstream of it.

    uv run python tests/investigation/provmns/test_notification_as_path.py
"""

from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.config.settings import TARGET_YAML
from openapi_rulesbank.nodes import extractor as extractor_mod
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.extractor import extractor_node
from openapi_rulesbank.nodes.reflector import reflector_node
from openapi_rulesbank.nodes.validator import validator_node
from openapi_rulesbank.tools.rag_tools import (
    drop_target_spec_collection, drop_rules_collection,
)

# 363 defines {notificationTarget} as the consumer's address; 365 carries the
# payloads and responses the producer sends there. Document order.
URI_SECTION  = "12.1.1.3.2.4.2"
POST_SECTION = "12.1.1.3.2.4.3.1"

FOCUS = "Extract all concrete rules this section defines."

WRONG_PATH = "paths./{notificationTarget}"

# The fact section 365 lacks: which operation owns the callback. Worded the way a
# retrieved stage 3 sibling would read, so arm B differs from A only in that the
# fact is present — not in how it is phrased.
ANCHOR_CONTEXT = """\
### 12.1.1.3.2.1.3.1 HTTP PUT (resource "/{className}={id}")

The MnS consumer creates or replaces a resource with HTTP PUT on the resource
URI "/{className}={id}". The request body carries the resource representation,
including the notificationRecipientAddress field by which the consumer supplies
the address where it receives notifications. This is the operation in which the
consumer supplies its notification address.
"""


def _run_section(section, base_state, idx, *, anchored: bool):
    """Extract -> Reflect -> Validate one section; optionally inject the anchor.

    The anchor is added to the stage 3 sibling context — the channel the Extractor
    already reads for facts that live in neighbouring sections — by wrapping the
    module's search_target_spec. Only the stage 3 call is touched: it is the one
    whose hint asks for "resource URI path, HTTP method", so a sibling PUT section
    is exactly what it would have returned had retrieval reached it.
    """
    original = extractor_mod.search_target_spec

    def patched(query, collection, k=None, filters=None):
        got = original(query, collection, k=k, filters=filters) or ""
        if anchored and (filters or {}).get("stage") == "3":
            return f"{ANCHOR_CONTEXT}\n\n{got}"
        return got

    extractor_mod.search_target_spec = patched
    try:
        st = {**base_state, "current_section_idx": idx, "section_iteration_count": 0,
              "validation_errors": [], "section_reflection": [],
              "validator_section_reflection": []}
        ext = extractor_node(st)
        st = {**st, **ext}
        rfl = reflector_node(st)
        st = {**st, **rfl}
        val = validator_node(st)
        return ext, val
    finally:
        extractor_mod.search_target_spec = original


def _report(label, ext, val):
    rules = val.get("validated_rules", [])
    print(f"--- {label} — extracted {len(ext.get('raw_rules', []))}, "
          f"validated {len(rules)}")
    for x in rules:
        m = x["openapi_mapping"]
        print(f"      {x['rule_type']:16} {x['source_name'][:26]:26} "
              f"{str(m.get('openapi_object', ''))[:60]}")
    callbacks = [x for x in rules if x["rule_type"] == "callback"]
    on_wrong  = [x for x in rules
                 if str(x["openapi_mapping"].get("openapi_object", "")).startswith(WRONG_PATH)]
    placeholder = [x for x in rules
                   if "<" in str(x["openapi_mapping"].get("openapi_object", ""))]
    print(f"    callback rules: {len(callbacks)} | on '{WRONG_PATH}': {len(on_wrong)}"
          f" | unresolved-placeholder anchors: {len(placeholder)}")
    for c in callbacks:
        m = c["openapi_mapping"]
        print(f"      -> {c['source_name']}: {m.get('openapi_object','')}")
        print(f"         field={m.get('openapi_field','')} value={m.get('openapi_value','')}")
    print()
    return len(callbacks)


def main():
    print(f"=== TARGET_YAML = {TARGET_YAML} ===")
    print("A = pipeline as it runs today | B = same, with the owning PUT stated\n")

    r = reader_node({"main_doc_path": TSPEC_DATA_TEST_FILE,
                     "auxiliary_doc_paths": [], "target_yaml": TARGET_YAML})
    secs = r["parsed_sections"]
    cb = contract_builder_node({"parsed_sections": secs, "target_yaml": TARGET_YAML})
    spec_coll  = cb["spec_collection_name"]
    rules_coll = cb.get("rules_collection_name", "")

    s_uri  = next(s for s in secs if s["title"].startswith(URI_SECTION))
    s_post = next(s for s in secs if s["title"].startswith(POST_SECTION))

    plan = {"sections_to_extract": [
        {"section_id": s["section_id"], "title": s["title"], "priority": "high",
         "extraction_focus": FOCUS, "notes": ""}
        for s in (s_uri, s_post)
    ]}
    base = {
        **r, "service_contract": cb["service_contract"], "spec_collection_name": spec_coll,
        "rules_collection_name": rules_coll, "extraction_plan": plan, "helper_context": "",
        "other_spec_ref_memory": {}, "section_analysis_by_id": {},
        "validated_rules": [],
    }

    # Section 363 first, so 365 starts from the same memory the graph would give it
    # (notificationRecipientAddress, learned from the URI table).
    print(f"[{s_uri['section_id']}] {s_uri['title']}")
    ext_uri, val_uri = _run_section(s_uri, base, 0, anchored=False)
    _report("URI section", ext_uri, val_uri)

    carried = {
        **base,
        "other_spec_ref_memory":  ext_uri.get("other_spec_ref_memory", {}),
        "section_analysis_by_id": ext_uri.get("section_analysis_by_id", {}),
        "validated_rules":        val_uri.get("validated_rules", []),
    }
    print(f"[{s_post['section_id']}] {s_post['title']}")
    print(f"    memory carried in: {sorted(carried['other_spec_ref_memory'])}\n")

    n_a = _report("A (baseline)", *_run_section(s_post, carried, 1, anchored=False))
    n_b = _report("B (anchored)", *_run_section(s_post, carried, 1, anchored=True))

    print("=" * 70)
    if n_b and not n_a:
        print("=> The anchor is the blocker: naming the owning PUT is what lets the "
              "callback form. Fix belongs where the anchor is resolved.")
    elif n_a and n_b:
        print("=> A callback formed either way — the anchor is not the blocker; "
              "this run disagrees with the full-pipeline result, so look at what "
              "else differs there.")
    elif not n_a and not n_b:
        print("=> Neither arm produced a callback: supplying the anchor is not "
              "enough, so the cause sits upstream of it (the consumer-address "
              "judgement itself, or the shape the prompt asks for).")
    else:
        print("=> Only the baseline produced a callback — unexpected; the injected "
              "context changed something other than the anchor.")

    drop_target_spec_collection(spec_coll)
    if rules_coll:
        drop_rules_collection(rules_coll)


if __name__ == "__main__":
    main()
