"""What the Extractor reports as missing context, before anything acts on it.

First step of the retrieval loop: the Extractor now returns `missing_context`
alongside its rules — the facts it needed to map a rule correctly and found in
neither the section nor any context it was given. Nothing consumes the field yet;
this checks whether what it reports is worth building the rest on.

The bar is not "does it report something" but:
  * SILENT where the section is self-contained (a Type clause defining its own
    fields needs nothing from elsewhere — a gap reported there is noise);
  * SPECIFIC where a mapping genuinely failed — naming the one fact, not a topic.

So it runs a mix of both kinds, Extractor only (no Reflector/Validator — nothing
downstream reads the field yet), in document order so the reference memory builds
up the way the graph would leave it:

  363  URI                — establishes {notificationTarget} as the consumer address
  365  POST               — the callback case: known to fall back to a path
  355  HTTP PUT           — the operation the callbacks should anchor on
  372  Type MoiChange     — self-contained schema definition (expect silence)
  384  Enumeration        — self-contained enum (expect silence)

    uv run python tests/investigation/provmns/test_missing_context.py
"""

from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.config.settings import TARGET_YAML
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.extractor import extractor_node
from openapi_rulesbank.tools.rag_tools import (
    drop_target_spec_collection, drop_rules_collection,
)

# (clause prefix, whether a gap is expected) — document order.
# The expectation is a prior, not ground truth: where the run disagrees, read the
# wording before assuming the Extractor is wrong. It already showed that the PUT
# clause does lack a fact (the request media type) that the YAML has.
# Five sections, one per behaviour worth watching — a wider sweep of eleven
# showed the same three shapes repeating, so it only cost tokens.
SECTIONS = [
    ("12.1.1.3.2.4.2",     True),   # URI: consumer-owned address, anchor lives elsewhere
    ("12.1.1.3.2.4.3.1",   True),   # POST: the callback case — anchor and media type
    ("12.1.1.3.2.1.3.1",   True),   # HTTP PUT: request media type is not in the clause
    ("12.1.1.4.1a.4",      False),  # Type MoiChange: defines its own fields — expect silence
    ("12.1.1.4.4.4",       False),  # Enumeration SourceIndicator: likewise
]

FOCUS = "Extract all concrete rules this section defines."


def main():
    print(f"=== TARGET_YAML = {TARGET_YAML} ===")
    print("expected: silence on self-contained sections, one specific fact on 365\n")

    r = reader_node({"main_doc_path": TSPEC_DATA_TEST_FILE,
                     "auxiliary_doc_paths": [], "target_yaml": TARGET_YAML})
    secs = r["parsed_sections"]
    cb = contract_builder_node({"parsed_sections": secs, "target_yaml": TARGET_YAML})
    spec_coll  = cb["spec_collection_name"]
    rules_coll = cb.get("rules_collection_name", "")

    chosen = [(next(s for s in secs if s["title"].startswith(p)), expect)
              for p, expect in SECTIONS]

    plan = {"sections_to_extract": [
        {"section_id": s["section_id"], "title": s["title"], "priority": "high",
         "extraction_focus": FOCUS, "notes": ""}
        for s, _ in chosen
    ]}
    base = {
        **r, "service_contract": cb["service_contract"], "spec_collection_name": spec_coll,
        "rules_collection_name": rules_coll, "extraction_plan": plan, "helper_context": "",
        "validated_rules": [], "validation_errors": [],
        "section_reflection": [], "validator_section_reflection": [],
    }

    ref_memory, analysis_by_id = {}, {}
    agreed = 0

    for idx, (s, expected_gap) in enumerate(chosen):
        st = {**base, "current_section_idx": idx, "section_iteration_count": 0,
              "other_spec_ref_memory": ref_memory,
              "section_analysis_by_id": analysis_by_id}
        ext = extractor_node(st)
        ref_memory     = {**ref_memory, **ext.get("other_spec_ref_memory", {})}
        analysis_by_id = {**analysis_by_id, **ext.get("section_analysis_by_id", {})}

        gaps  = ext.get("missing_context", []) or []
        rules = ext.get("raw_rules", [])
        match = bool(gaps) == expected_gap
        agreed += match

        print(f"[{s['section_id']}] {s['title'][:52]}")
        print(f"    {len(rules)} rule(s) | gaps reported: {len(gaps)} "
              f"| expected a gap: {expected_gap} {'OK' if match else '<-- DISAGREES'}")
        # The rules are printed next to the gaps on purpose: a gap is only worth
        # acting on if it explains something in what was actually emitted.
        for x in rules:
            m = x["openapi_mapping"]
            print(f"      · {x['rule_type']:16} {x['source_name'][:22]:22} "
                  f"{str(m.get('openapi_object', ''))[:44]} {str(m.get('openapi_field', ''))[:18]}")
        for g in gaps:
            print(f"      - GAP: {g}")
        # A section that fell back to a path while claiming nothing was missing is
        # the case this whole mechanism exists to catch — call it out either way.
        fell_back = [x for x in rules
                     if x["rule_type"] == "path_operation"
                     and "notificationTarget" in str(x["openapi_mapping"].get("openapi_object", ""))]
        if fell_back:
            print(f"      (emitted {len(fell_back)} path_operation on the consumer "
                  f"address — the fallback shape)")
        print()

    print("=" * 70)
    print(f"sections matching expectation: {agreed}/{len(chosen)}")
    print("Read the wording, not just the count: a gap that names a topic rather "
          "than one fact is not usable as a retrieval query.")

    drop_target_spec_collection(spec_coll)
    if rules_coll:
        drop_rules_collection(rules_coll)


if __name__ == "__main__":
    main()
