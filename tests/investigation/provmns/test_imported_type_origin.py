"""Why a type the document declares as imported still gets an internal $ref.

Thirty references in the last full run point inside the document where the YAML
points at another file (ErrorResponse 17, Filter 3, Uri 2, and so on). The
separation of type references from context references made Phase 1 strict — it
records an origin only where one row or one sentence names the type and the spec
together — and the origins it used to guess right went out with the ones it used
to invent.

But the link IS declared. Table 12.1.1.4.1-2 "Data types imported" lists eleven
types against "TS 28.623 [44]". So the question is not whether the document
states it; it is which link in the chain drops it. Three candidates, and this is
built to tell them apart:

  (a) THE TABLE IS NOT REACHABLE WHEN IT IS NEEDED. The table lives in clause
      12.1.1.4.1. The methods that consume the types — PUT, GET, PATCH, DELETE,
      clauses 12.1.1.3.2.1.3.x — come EARLIER in the document, so when they are
      triaged the memory is still empty and no clause of their own names the
      spec. Document order alone would explain most of the thirty.

  (b) THE TABLE IS THERE AND PHASE 1 DOES NOT PARSE IT. The pandoc conversion
      splits each cell across two lines: "TS 28.623" on the first, "\\[44\\]" on
      the second, with the description wrapping between them. A rule asking for
      the type and the spec on one row may not see that as one row.

  (c) PHASE 1 READS IT AND PHASE 2 DOES NOT STORE IT. memory_updates carries the
      YAML filename, and it came back empty in 48 of 57 references in the last
      run. An external $ref is impossible without it — the spec number alone
      never supplies the filename.

A fourth thing, separate from all three: ErrorResponseDefault, ErrorResponseGet
and ErrorResponsePatch are the names the method clauses actually use, and NONE is
in the imported table — only the unsuffixed "ErrorResponse" is. Those are 21 of
the 30. No amount of parsing finds a link the table never states.

Sections run in DOCUMENT ORDER through the real extractor_node, carrying the
reference memory forward the way the graph's reducers do, so each section is
triaged with exactly the memory it would have. Reflector and Validator are not
run: this probe is about what the triage records, and the rules it prints are the
raw ones, before any judging.

    uv run python tests/investigation/provmns/test_imported_type_origin.py
"""

import re

from openapi_rulesbank.config.llm_config import usage_tracker
from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.config.settings import TARGET_YAML
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.extractor import extractor_node
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.tools.rag_tools import (
    drop_target_spec_collection, drop_rules_collection,
)

# The eleven types table 12.1.1.4.1-2 attributes to TS 28.623, and the filename
# that spec publishes them in — what an external $ref needs before the '#'.
IMPORTED = {
    "DateTime", "Dn", "SystemDN", "Uri", "AttributeNameValuePairSet",
    "AttributeValueChangeSet", "Filter", "NotificationId", "NotificationType",
    "NotificationHeader", "ErrorResponse",
}
EXPECTED_FILE = "TS28623_ComDefs.yaml"

# The key the extractor stores the filename under in other_spec_ref_memory. The
# schema field is `source_file`, but _format_ref_memory reads it back as `file`,
# so reading the wrong one here would report "no filename" for every entry —
# exactly the kind of instrument error this investigation has already hit three
# times.
FILE_KEY = "file"

# The suffixed names the method clauses use, which the table does NOT list.
UNLISTED = {"ErrorResponseDefault", "ErrorResponseGet", "ErrorResponsePatch"}

# Document order matters — it is candidate (a). The consumers come first in the
# file, the declaring table later, and the memory is carried forward.
SECTIONS = [
    ("12.1.1.3.2.1.3.1", "PUT — consumes ErrorResponseDefault",           "consumer"),
    ("12.1.1.3.2.1.3.2", "GET — consumes ErrorResponseGet, Filter",       "consumer"),
    ("12.1.1.3.2.1.3.3", "PATCH — consumes ErrorResponseDefault/Patch",   "consumer"),
    ("12.1.1.3.2.1.3.4", "DELETE — consumes ErrorResponseDefault",        "consumer"),
    ("12.1.1.4.1",       "THE TABLE: 'Data types imported' (11 types)",   "declarer"),
    ("12.1.1.4.1a.1",    "Type Resource — consumes Dn",                   "consumer"),
    ("12.1.1.4.1a.4",    "Type MoiChange — consumes Uri",                 "consumer"),
    ("12.1.1.4.1a.5",    "Type NotifyMoiCreation — NotificationHeader",   "consumer"),
]

FOCUS = "Extract all concrete rules this section defines."

_REF = re.compile(r"([A-Za-z0-9_]+\.yaml)?#/components/schemas/([A-Za-z0-9_]+)")


def _classify(name, spec, src_file):
    """What this memory entry would produce downstream."""
    if name in UNLISTED:
        return "NAME NOT IN THE TABLE — no row pairs it with any spec"
    if name not in IMPORTED:
        return "local type — correctly not an import"
    if not spec:
        return "NO SPEC — will emit an internal $ref"
    if not src_file:
        return "spec but NO FILENAME — still cannot emit an external $ref"
    if EXPECTED_FILE.lower() not in src_file.lower():
        return f"filename {src_file!r} is not {EXPECTED_FILE}"
    return "COMPLETE — external $ref possible"


def _refs_in(rules):
    """The $ref each rule wrote, split by whether it names a file."""
    ext, internal = [], []
    for r in rules:
        v = str(r.get("openapi_mapping", {}).get("openapi_value", ""))
        for m in _REF.finditer(v):
            (ext if m.group(1) else internal).append(m.group(2))
    return ext, internal


def main():
    print("=== Where the imported-type origin is lost ===")
    print(f"    table 12.1.1.4.1-2 attributes {len(IMPORTED)} types to TS 28.623")
    print(f"    an external $ref also needs the filename: {EXPECTED_FILE}\n")

    r = reader_node({"main_doc_path": TSPEC_DATA_TEST_FILE,
                     "auxiliary_doc_paths": [], "target_yaml": TARGET_YAML})
    secs = r["parsed_sections"]
    cb = contract_builder_node({"parsed_sections": secs, "target_yaml": TARGET_YAML,
                                "main_doc_path": TSPEC_DATA_TEST_FILE,
                                "target_spec_number": r.get("target_spec_number", "")})
    spec_coll  = cb["spec_collection_name"]
    rules_coll = cb.get("rules_collection_name", "")

    chosen = [(next(s for s in secs if s["title"].startswith(p)), what, role)
              for p, what, role in SECTIONS]
    plan = {"sections_to_extract": [
        {"section_id": s["section_id"], "title": s["title"], "priority": "high",
         "extraction_focus": FOCUS, "notes": ""}
        for s, _, _ in chosen
    ]}
    base = {
        **r, "service_contract": cb["service_contract"],
        "spec_collection_name": spec_coll, "rules_collection_name": rules_coll,
        "extraction_plan": plan,
        # Deliberately empty: the auxiliary document outranks the document, the
        # convention and the OpenAPI reference, so handing a filename through it
        # would answer the very question this probe asks.
        "helper_context": "",
        "other_spec_ref_memory": {}, "cross_ref_facts": {},
        "section_analysis_by_id": {},
        "validated_rules": [], "validated_rules_by_section": {},
    }

    # What the graph's reducers accumulate across sections.
    carried: dict = {
        "other_spec_ref_memory": {}, "cross_ref_facts": {},
        "section_analysis_by_id": {},
    }
    all_ext, all_int = [], []

    for idx, (s, what, role) in enumerate(chosen):
        sid = s["section_id"]
        print("-" * 78)
        print(f"[{sid}] {what}")
        print(f"  role       : {role}")
        mem_in = carried["other_spec_ref_memory"]
        print(f"  memory in  : {sorted(mem_in) or '(EMPTY)'}")
        # Which of those entries already carry a filename — the only ones that
        # can produce an external $ref in this section.
        with_file = [k for k, v in mem_in.items()
                     if isinstance(v, dict) and v.get(FILE_KEY)]
        print(f"  of those, with a filename: {sorted(with_file) or 'NONE'}")
        # Candidate (a) hinges on this: a consumer clause that never names the
        # spec can only get the origin from the memory it was handed.
        body = s["content"]
        print(f"  this clause names TS 28.623: {'28.623' in body}")

        state = {**base, **carried,
                 "current_section_idx": idx, "section_iteration_count": 0,
                 "validation_errors": [], "section_reflection": [],
                 "validator_section_reflection": [], "missing_context": [],
                 "retrieved_context": {}}

        # The real node: it builds every prompt argument itself, so this probe
        # does not break when a prompt gains a field (which is exactly how the
        # first version of this file broke).
        ext = extractor_node(state)

        mem_out = ext.get("other_spec_ref_memory", {}) or {}
        new = {k: v for k, v in mem_out.items() if k not in mem_in}
        if not new:
            print("  memory NEW entries: NONE")
        for name, v in sorted(new.items()):
            spec = (v or {}).get("spec", "")
            sf   = (v or {}).get(FILE_KEY, "")
            print(f"  memory NEW : {name!r} spec={spec!r} file={sf!r}")
            print(f"          -> {_classify(name, spec, sf)}")

        rules = ext.get("raw_rules", []) or []
        e, i = _refs_in(rules)
        all_ext += e
        all_int += i
        print(f"  rules: {len(rules)} | $ref external: {e or '—'} | internal: {i or '—'}")
        for g in (ext.get("missing_context", []) or []):
            print(f"      - GAP: {g}")

        carried = {
            "other_spec_ref_memory": {**mem_in, **mem_out},
            "cross_ref_facts": {**carried["cross_ref_facts"],
                                **(ext.get("cross_ref_facts", {}) or {})},
            "section_analysis_by_id": {**carried["section_analysis_by_id"],
                                       **(ext.get("section_analysis_by_id", {}) or {})},
        }
        print()

    # --- What the run decided ------------------------------------------------
    mem = carried["other_spec_ref_memory"]
    print("=" * 78)
    print("MEMORY AT THE END — what a later section would be handed:\n")
    for name in sorted(mem):
        m = mem[name] if isinstance(mem[name], dict) else {}
        print(f"  {name:28} spec={str(m.get('spec') or '—'):12} "
              f"file={str(m.get(FILE_KEY) or '—')}")

    got_file = sorted(n for n, m in mem.items()
                      if n in IMPORTED and isinstance(m, dict) and m.get(FILE_KEY))
    no_file  = sorted(n for n, m in mem.items()
                      if n in IMPORTED and isinstance(m, dict) and not m.get(FILE_KEY))
    never    = sorted(IMPORTED - set(mem))

    print(f"\n  imported types WITH filename (external $ref possible): {len(got_file)} {got_file}")
    print(f"  imported types WITHOUT filename                      : {len(no_file)} {no_file}")
    print(f"  imported types never recorded at all                 : {len(never)} {never}")
    print(f"\n  $ref written across all sections — external {len(all_ext)} {all_ext}")
    print(f"                                     internal {len(all_int)} {all_int}")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("  If the four method clauses recorded nothing and the table's own")
    print("  clause recorded the eleven, the loss is DOCUMENT ORDER (a): the")
    print("  consumers are triaged before the declaring table is ever read.")
    print("  If the table's clause ALSO recorded nothing, Phase 1 is not parsing")
    print("  the split cells (b).")
    print("  If specs came back but source_file stayed empty, the origin reaches")
    print("  Phase 2 and dies there (c).")
    print("  Names in UNLISTED (ErrorResponse{Default,Get,Patch}) cannot be linked")
    print("  by any parsing — no row of the table pairs them with a spec. That is")
    print("  a separate defect, and it is 21 of the 30.")

    u = usage_tracker.as_dict()
    print(f"\nUSAGE — {u['llm_calls']} call(s), {u['total_tokens']} tokens, "
          f"US$ {u['cost_usd']:.6f}")

    drop_target_spec_collection(spec_coll)
    if rules_coll:
        drop_rules_collection(rules_coll)


if __name__ == "__main__":
    main()
