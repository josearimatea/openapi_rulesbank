"""End-to-end check of the missing-context loop, over a handful of sections.

The loop closes like this:

  Extractor  declares the facts it lacked (missing_context)
  Reflector  judges each gap, writes one query, retrieves the answer
  Validator  sees the gaps and rejects the rules built on them (correction)
  conditions the error rate rises past the threshold -> retry
  Extractor  gets the fetched text inside its correction task, and remaps

Every piece has been checked on its own; what has not is whether they produce
the right result together. Four things can each fail quietly:

  1. the Reflector writes a query in the vocabulary of the complaint, which
     retrieves clauses that also fail to state the fact;
  2. the retrieval returns something irrelevant, and the next attempt is no
     better informed than the first;
  3. the Validator rejects nothing (the rules look well-formed) — or rejects
     everything, including rules the gap never touched;
  4. the second attempt receives the fact and maps the same way regardless.

So this runs the real graph — Extractor -> Reflector -> Validator -> retry —
over the same five sections the earlier probe used, and prints what each stage
did per attempt. It asserts nothing: the retrieved text has to be read to judge
whether it answers the gap.

    uv run python tests/investigation/provmns/test_gap_retrieval_loop.py
"""

from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE
from openapi_rulesbank.config.settings import MAX_ITERATIONS, TARGET_YAML
from openapi_rulesbank.graph.conditions import should_retry_section_or_build
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.extractor import extractor_node
from openapi_rulesbank.nodes.reflector import reflector_node
from openapi_rulesbank.nodes.validator import validator_node
from openapi_rulesbank.tools.rag_tools import (
    drop_target_spec_collection, drop_rules_collection,
)

# Same five as the missing-context probe: three that declared a gap there, two
# self-contained ones that stayed silent and should still go through untouched.
# The two self-contained sections are kept as controls: the first attempt at
# this loop made the Validator reject far too much (a section fell from 6 rules
# to 0), and a section with no gap at all is where that shows up.
SECTIONS = [
    # Document order, and every clause the anchor actually depends on. The method
    # clause (HTTP PUT) never names an operation — it opens with "This method
    # shall support..." and its tables carry parameters and response codes. What
    # ties createMOI to that method is the operation-mapping clause, whose title
    # names the operation and whose tables say HTTP PUT. Run without it, the
    # triage has nothing to record and the anchor cannot be resolved from the
    # spec — which is what earlier runs were measuring.
    "12.1.1.1.2",         # Operation createMOI — names the operation and its method
    "12.1.1.3.2.1.3.1",   # HTTP PUT — the resource clause it is mapped onto
    "12.1.1.3.2.4.2",     # URI — where notificationRecipientAddress enters the memory
    "12.1.1.3.2.4.3.1",   # POST on {notificationTarget} — the callback case
    # Put these back to check the Validator is not over-rejecting (an earlier run
    # took a section from 6 rules to 0, and a section with no gap at all is where
    # that shows):
    # "12.1.1.4.1a.4",      # Type MoiChange — self-contained
    # "12.1.1.4.4.4",       # Enumeration SourceIndicator — self-contained
]

FOCUS = "Extract all concrete rules this section defines."

# Anything anchored here is the fallback shape the loop is supposed to correct.
WRONG_PATH_MARK = "notificationTarget"


def _fmt_rules(rules):
    for x in rules:
        m = x.get("openapi_mapping", {})
        yield (f"{x.get('rule_type',''):16} {str(x.get('source_name',''))[:24]:24} "
               f"{str(m.get('openapi_object',''))[:52]} {str(m.get('openapi_field',''))[:14]}")


def _run_section(section, base, idx, carried):
    """Drive one section through the retry loop the way the graph does.

    `carried` holds what the state reducers accumulate ACROSS sections in the
    real graph — the reference memory and the per-section verdicts. Rebuilding
    the state from `base` alone would reset them at every section, which is what
    strands a fact one clause discovers and the next one needs (the URI clause
    learns notificationRecipientAddress; the POST clause is where it matters).
    """
    state = {**base, **carried,
             "current_section_idx": idx, "section_iteration_count": 0,
             "validation_errors": [], "section_reflection": [],
             "validator_section_reflection": [], "missing_context": [],
             "retrieved_context": {}}

    section_id = section["section_id"]
    print(f"[{section_id}] {section['title'][:58]}")
    print(f"  memory in: {sorted(state.get('other_spec_ref_memory', {})) or '(empty)'}")
    for subj, info in (state.get("cross_ref_facts", {}) or {}).items():
        print(f"  cross-ref: {subj}: {info.get('fact', '')[:110]}")

    for attempt in range(1, MAX_ITERATIONS + 1):
        ext = extractor_node(state)
        state = {**state, **ext,
                 # the reducers merge these in the real graph
                 "other_spec_ref_memory": {**state.get("other_spec_ref_memory", {}),
                                           **ext.get("other_spec_ref_memory", {})},
                 "cross_ref_facts": {**state.get("cross_ref_facts", {}),
                                     **ext.get("cross_ref_facts", {})},
                 "section_analysis_by_id": {**state.get("section_analysis_by_id", {}),
                                            **ext.get("section_analysis_by_id", {})}}

        gaps  = ext.get("missing_context", []) or []
        rules = ext.get("raw_rules", [])
        print(f"  attempt {attempt}: extracted {len(rules)} rule(s), "
              f"{len(gaps)} gap(s) declared")
        for line in _fmt_rules(rules):
            print(f"      · {line}")
        for g in gaps:
            print(f"      - GAP: {g}")

        rfl = reflector_node(state)
        state = {**state, **rfl}
        retrieved = (rfl.get("retrieved_context", {}) or {}).get(section_id, "")
        if retrieved:
            # The block headers carry the gap each answer belongs to; the body is
            # what has to be read to tell a real answer from a plausible-looking one.
            print(f"      RETRIEVED {len(retrieved)} chars:")
            for block in retrieved.split("\n\n### "):
                head, _, body = block.partition("\n")
                print(f"        > {head.removeprefix('### ')[:96]}")
                print(f"          {' '.join(body.split())[:300]}")
        elif gaps:
            print("      RETRIEVED nothing — the Reflector judged no gap worth a query")

        val = validator_node(state)
        state = {**state, **val,
                 # validated_rules carries operator.add in the real state, so a
                 # retry ADDS to what earlier attempts validated. Overwriting it
                 # with this attempt's rules alone would throw away everything
                 # already accepted — and a retry only re-emits what the
                 # correction task asked for, so the section would end up with
                 # just the last attempt's handful.
                 "validated_rules": (state.get("validated_rules", [])
                                     + val.get("validated_rules", []))}
        errors      = val.get("validation_errors", []) or []
        corrections = [e for e in errors if e.get("error_type") == "correction"]
        validated   = val.get("validated_rules", []) or []
        print(f"      validator: {len(validated)} passed, {len(corrections)} correction(s)")
        for e in corrections:
            m = (e.get("rule") or {}).get("openapi_mapping", {})
            print(f"        ! {str((e.get('rule') or {}).get('rule_type','')):16} "
                  f"{str(m.get('openapi_field',''))[:14]} -> {str(e.get('instruction',''))[:88]}")

        decision = should_retry_section_or_build(state)
        print(f"      -> {decision}")
        if decision != "extractor":
            break
        print()

    final = state.get("validated_rules", []) or []
    on_wrong = [x for x in final
                if WRONG_PATH_MARK in str(x.get("openapi_mapping", {}).get("openapi_object", ""))]
    callbacks = [x for x in final
                 if "callbacks." in str(x.get("openapi_mapping", {}).get("openapi_object", ""))]
    print(f"  FINAL: {len(final)} rule(s) | on the fallback path: {len(on_wrong)} "
          f"| anchored as callbacks: {len(callbacks)}")
    print()
    # Hand back what the reducers would carry into the next section.
    return final, {
        "other_spec_ref_memory":  state.get("other_spec_ref_memory", {}),
        "cross_ref_facts":        state.get("cross_ref_facts", {}),
        "section_analysis_by_id": state.get("section_analysis_by_id", {}),
    }


def main():
    print(f"=== TARGET_YAML = {TARGET_YAML} | MAX_ITERATIONS = {MAX_ITERATIONS} ===\n")

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
        "other_spec_ref_memory": {}, "cross_ref_facts": {},
        "section_analysis_by_id": {},
        "validated_rules": [], "validated_rules_by_section": {},
    }

    carried: dict = {}
    for idx, s in enumerate(chosen):
        _, carried = _run_section(s, base, idx, carried)

    print("=" * 74)
    print("Judge by the retrieved blocks: a query that answers its gap is the "
          "whole point, and a plausible-looking block that does not is the "
          "failure mode to watch for.")

    drop_target_spec_collection(spec_coll)
    if rules_coll:
        drop_rules_collection(rules_coll)


if __name__ == "__main__":
    main()
