"""What the pre-analyzer does with a reference to another specification.

Phase 1 is the only place in the pipeline that follows a cross-spec reference: it
names the referenced thing plus a few key_terms, the code builds a query around
them, and one lookup runs per reference. Phase 2 then reads those results and
writes the memory and the guidance the Extractor works from. Neither phase
searches again after that.

Both phases run here, in document order and carrying the reference memory
forward, so each section is triaged exactly as the pipeline triages it. For each
section the output shows the references Phase 1 reported, the query the code
assembled, whether the answer came back — and then what Phase 2 made of it.

The callback anchor is what this is aimed at. Clause 12.1.1.3.2.4.2 points at
"notificationRecipientAddress defined in clause 4.3.22.2 in TS 28.622", and that
clause states a consumer receives notifications by CREATING an
NtfSubscriptionControl instance — the operation the callbacks hang off. If Phase 1
retrieves that and the Extractor still anchors the callback on the notification's
own address, the loss is in Phase 2: its outputs (section_kind, schema_name,
base_ref, inherited_fields, guidance) are all about schema shape, and a fact about
WHICH OPERATION CREATES A THING may have nowhere to go.

    uv run python tests/investigation/provmns/test_spec_references.py
"""

from openapi_rulesbank.config.llm_config import llm
from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.config.settings import IMPORTED_TYPE_RETRIEVE_CHUNKS
from openapi_rulesbank.nodes.extractor import _build_sibling_summary, _format_ref_memory
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.prompts.section_analyzer_prompts import (
    SectionAnalysis, SectionImports,
    section_analyzer_prompt, section_import_ref_prompt,
)
from openapi_rulesbank.tools.rag_tools import search_3gpp_corpus

# Document order, so the memory each section receives is the one it would have.
# (clause, what kind of reference it makes, words an answer would contain)
CASES = [
    ("12.1.1.1.3",
     "types the operation mapping borrows from another spec",
     ("attributes", "fields")),
    ("12.1.1.1.4.2",
     "patch media types defined in another spec",
     ("merge-patch", "json-patch")),
    ("12.1.1.3.2.1.2",
     "URI variables defined by the common URI structure",
     ("MnSRoot", "URI structure")),
    ("12.1.1.3.2.4.2",
     "a URI variable that is an attribute of another spec's object",
     ("NtfSubscriptionControl", "has to create")),
]


def main():
    print("=== Phase 1 and Phase 2 over four kinds of cross-spec reference ===\n")

    r = reader_node({"main_doc_path": TSPEC_DATA_TEST_FILE,
                     "auxiliary_doc_paths": [], "target_yaml": "ProvMnS"})
    secs = r["parsed_sections"]
    p1 = section_import_ref_prompt | llm.with_structured_output(SectionImports)
    p2 = section_analyzer_prompt | llm.with_structured_output(SectionAnalysis)

    ref_memory: dict = {}
    cross_ref_facts: dict = {}
    analysis_by_id: dict = {}

    for prefix, kind, wanted in CASES:
        s = next(x for x in secs if x["title"].startswith(prefix))
        sid = s["section_id"]
        print(f"[{sid}] {s['title'][:58]}")
        print(f"  reference kind : {kind}")
        print(f"  memory in      : {sorted(ref_memory) or '(empty)'}")
        print(f"  cross-refs in  : {sorted(cross_ref_facts) or '(empty)'}")

        # --- Phase 1: find the references, one lookup each ---------------------
        imports: SectionImports = p1.invoke({
            "ref_memory":      _format_ref_memory(ref_memory),
            "section_title":   s["title"],
            "section_content": s["content"],
        })

        rag_blocks = []
        if not imports.references:
            print("  PHASE 1        : no reference found — nothing is looked up")
        for ref in imports.references:
            # The query the code builds, reproduced exactly (rag_tools.py).
            query = f"{ref.type_name} type: object properties required {ref.key_terms}".strip()
            got = search_3gpp_corpus(query, k=IMPORTED_TYPE_RETRIEVE_CHUNKS) or ""
            hits = [w for w in wanted if w in got]
            if got:
                rag_blocks.append(f"### {ref.type_name} (from {ref.source_spec}):\n{got}")
            print(f"  PHASE 1 ref    : {ref.type_name} -> {ref.source_spec}")
            print(f"          query  : {query[:88]!r}")
            print(f"          got    : {len(got)} chars | landmarks {hits or 'NONE'}")

        # --- Phase 2: what it makes of those results ---------------------------
        analysis: SectionAnalysis = p2.invoke({
            "section_id":         sid,
            "section_title":      s["title"],
            "section_content":    s["content"],
            "ref_memory":         _format_ref_memory(ref_memory),
            "phase1_explanation": imports.explanation or "None.",
            "rag_results":        "\n\n".join(rag_blocks) or "No definitions retrieved.",
            "sibling_summary":    _build_sibling_summary(
                                      s["title"], secs, analysis_by_id, exclude_id=sid
                                  ) or "No sibling sections summarized yet.",
        })

        print(f"  PHASE 2 kind   : {analysis.section_kind} | "
              f"emits_schema={analysis.emits_schema_properties} | "
              f"schema={analysis.schema_name!r}")
        print(f"          memory : "
              f"{ {e.type_name: e.fields for e in analysis.memory_updates} or '{}'}")
        # The channel added for facts a list of fields cannot hold. It has to work
        # for every kind of reference above, not only the callback one: a URI
        # variable that belongs to the base URL, a patch format defined elsewhere,
        # a type an operation borrows — each states something a later section
        # needs and none of them is a schema shape.
        if analysis.cross_refs:
            for c in analysis.cross_refs:
                print(f"          CROSS-REF: {c.subject}: {c.fact[:150]}")
                if c.source:
                    print(f"          {'':<10} source: {c.source}")
        else:
            print("          CROSS-REF: none")
        print(f"          guidance: {(analysis.guidance or '(none)')[:260]}")

        ref_memory = {**ref_memory, **{
            e.type_name: {"spec": e.source_spec, "fields": e.fields, "note": e.note}
            for e in analysis.memory_updates if e.type_name
        }}
        cross_ref_facts = {**cross_ref_facts, **{
            c.subject: {"fact": c.fact, "source": c.source}
            for c in (analysis.cross_refs or []) if c.subject and c.fact
        }}
        analysis_by_id[sid] = {"section_kind": analysis.section_kind}
        print()

    print("=" * 78)
    print("Read the last section's guidance against what Phase 1 retrieved for it:")
    print("if the answer came back and the guidance says nothing about which")
    print("operation creates the subscription, the triage had the fact and dropped")
    print("it — and no later stage looks for it again.")


if __name__ == "__main__":
    main()
