"""Does a declared gap turn into a query, and does the query answer it?

Extractor -> Reflector -> retrieval only. The Validator is deliberately left out:
wiring it in made the loop worse (a section fell from 6 rules to 0, because it
rejected rules citing a "retrieved fact" that never arrived), and that regression
hides whether the retrieval itself works. First make the fetch good, then reopen
the rest of the loop.

Three things are being watched, in order — each only matters if the one before it
worked:

  1. does the Extractor declare a gap where one exists, and stay quiet otherwise;
  2. does the Reflector turn it into a query — the previous run produced ZERO,
     traced to a filter in the node that has since been removed;
  3. does the query bring back the fact — the point of the whole mechanism, and
     the one thing no assertion can decide. The retrieved text is printed in full
     so it can be read: a block that looks plausible and answers nothing is the
     failure mode here.

    uv run python tests/investigation/provmns/test_gap_retrieval.py
"""

from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.config.settings import TARGET_YAML
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.extractor import extractor_node
from openapi_rulesbank.nodes.reflector import reflector_node
from openapi_rulesbank.tools.rag_tools import (
    drop_target_spec_collection, drop_rules_collection,
)

# Three that declared a gap, two self-contained ones that should stay silent, and
# the Introduction clause — the one that spells the URI out in full, so it is
# where a path written as the document writes it (base URL and all) shows up, and
# where the base-URL variables get emitted as path parameters they are not.
SECTIONS = [
    "12.1.1.1.1",         # Introduction — the full URI and its variables table
    "12.1.1.3.2.4.2",     # URI — consumer-owned address, anchor lives elsewhere
    "12.1.1.3.2.4.3.1",   # POST — the callback case
    "12.1.1.3.2.1.3.1",   # HTTP PUT — request media type is not in the clause
    "12.1.1.4.1a.4",      # Type MoiChange — self-contained
    "12.1.1.4.4.4",       # Enumeration SourceIndicator — self-contained
]

# The path key the target YAML uses for the resource these clauses describe. A
# rule anchored anywhere else is the fragmentation this run is checking for.
CANONICAL_PATH = "paths./{className}={id}"

# Variables of the base URL: the URI-variables table defines them by pointing at
# TS 32.158 rather than describing them, and OpenAPI puts them under `servers`.
# They are neither part of a path key nor path parameters of their own.
BASE_URL_VARS = ("MnSRoot", "MnSVersion", "URI-LDN-first-part")

FOCUS = "Extract all concrete rules this section defines."


def main():
    print(f"=== TARGET_YAML = {TARGET_YAML} — Extractor -> Reflector -> retrieval ===\n")

    r = reader_node({"main_doc_path": TSPEC_DATA_TEST_FILE,
                     "auxiliary_doc_paths": [], "target_yaml": TARGET_YAML})
    secs = r["parsed_sections"]
    cb = contract_builder_node({"parsed_sections": secs, "target_yaml": TARGET_YAML})
    spec_coll  = cb["spec_collection_name"]
    rules_coll = cb.get("rules_collection_name", "")

    chosen = [next(s for s in secs if s["title"].startswith(p)) for p in SECTIONS]
    plan = {"sections_to_extract": [
        {"section_id": s["section_id"], "title": s["title"], "priority": "high",
         "extraction_focus": FOCUS, "notes": ""}
        for s in chosen
    ]}
    base = {
        **r, "service_contract": cb["service_contract"], "spec_collection_name": spec_coll,
        "rules_collection_name": rules_coll, "extraction_plan": plan, "helper_context": "",
        "validated_rules": [], "validation_errors": [],
        "section_reflection": [], "validator_section_reflection": [],
        "validated_rules_by_section": {},
    }

    ref_memory, analysis_by_id = {}, {}
    declared = fetched = 0
    off_canonical: list[str] = []    # path keys that are not CANONICAL_PATH
    base_var_params: list[str] = []  # path_parameter rules for base-URL variables

    for idx, s in enumerate(chosen):
        section_id = s["section_id"]
        state = {**base, "current_section_idx": idx, "section_iteration_count": 0,
                 "other_spec_ref_memory": ref_memory,
                 "section_analysis_by_id": analysis_by_id,
                 "missing_context": [], "retrieved_context": {}}

        ext = extractor_node(state)
        state = {**state, **ext,
                 "other_spec_ref_memory":  {**ref_memory, **ext.get("other_spec_ref_memory", {})},
                 "section_analysis_by_id": {**analysis_by_id, **ext.get("section_analysis_by_id", {})}}
        ref_memory     = state["other_spec_ref_memory"]
        analysis_by_id = state["section_analysis_by_id"]

        gaps  = ext.get("missing_context", []) or []
        rules = ext.get("raw_rules", [])
        declared += len(gaps)

        print(f"[{section_id}] {s['title'][:56]}")
        print(f"  extracted {len(rules)} rule(s), declared {len(gaps)} gap(s)")
        for x in rules:
            m = x.get("openapi_mapping", {})
            obj = str(m.get("openapi_object", ""))
            # Flag the two path defects as they are printed, so the run reads as
            # a verdict and not as a list to eyeball.
            note = ""
            if obj.startswith("paths") and not obj.startswith(CANONICAL_PATH):
                if any(v in obj for v in BASE_URL_VARS):
                    note = "  <-- base URL in the path key"
                    off_canonical.append(obj)
                elif "notificationTarget" not in obj:
                    note = "  <-- not the canonical path key"
                    off_canonical.append(obj)
            if (x.get("rule_type") == "path_parameter"
                    and str(x.get("source_name", "")) in BASE_URL_VARS):
                note = "  <-- base URL variable is not a path parameter"
                base_var_params.append(str(x.get("source_name", "")))
            print(f"    · {x.get('rule_type',''):16} {str(x.get('source_name',''))[:22]:22} "
                  f"{obj[:46]} {str(m.get('openapi_field',''))[:14]}{note}")
        for g in gaps:
            print(f"    - GAP: {g}")

        if not gaps:
            print("    (no gap declared — nothing to retrieve)\n")
            continue

        rfl = reflector_node(state)
        retrieved = (rfl.get("retrieved_context", {}) or {}).get(section_id, "")

        if not retrieved:
            print("    !! the Reflector produced no query for these gaps\n")
            continue

        fetched += 1
        print(f"    RETRIEVED {len(retrieved)} chars — read this, it is the point:")
        for block in retrieved.split("\n\n### "):
            head, _, body = block.partition("\n")
            print(f"      > {head.removeprefix('### ')[:100]}")
            print(f"        {' '.join(body.split())[:700]}")
        print()

    print("=" * 74)
    print(f"gaps declared: {declared} | sections that got an answer back: {fetched}")
    print("A query that runs is not a query that answers. Judge each block against "
          "the gap above it: does the text actually state the missing fact?")
    print()
    print(f"path keys off '{CANONICAL_PATH}': {len(off_canonical)}")
    for o in dict.fromkeys(off_canonical):
        print(f"    {o}")
    print(f"path_parameter rules for base-URL variables: {len(base_var_params)}"
          f" {sorted(set(base_var_params)) if base_var_params else ''}")

    drop_target_spec_collection(spec_coll)
    if rules_coll:
        drop_rules_collection(rules_coll)


if __name__ == "__main__":
    main()
