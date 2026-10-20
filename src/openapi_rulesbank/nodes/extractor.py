# src/nodes/extractor.py

"""
Extractor Node: extracts raw OpenAPI rules from the current section guided
by the Planner's extraction plan.

LLM call: yes (1 call per section attempt). RAG: yes (1 query per section via Qdrant).

HOW IT WORKS:
    Processes ONE section per call — the one at current_section_idx in the plan.

    First attempt (section_iteration_count == 0):
      1. Retrieves relevant OpenAPI reference chunks from Qdrant via RAG
         (query: section_title — extraction_focus).
      2. Invokes the LLM with the section content, RAG context, and helper context.
      3. The LLM extracts RawRule objects, each with a rule_type that categorizes
         the OpenAPI construct (path_operation, schema_property, path_parameter,
         query_parameter, response, request_body, security_scheme).

    Retry (section_iteration_count > 0, validation_errors or section_reflection not empty):
      - LLM receives a CORRECTION TASK listing failed rules and missing rules.
      - Only acts on what is listed — does not re-extract already-validated rules.

    The prompt enforces: one rule per HTTP method (never combined), extract ONLY
    from section content (not from the OpenAPI reference).

State reads:
    parsed_sections         (list[dict]) — full section content from reader_node
    extraction_plan         (dict)       — sections_to_extract from planner_node
    helper_context          (str)        — auxiliary 3GPP documents context
    service_contract        (dict)       — scope contract from contract_builder_node;
                                           the LLM extracts only rules that belong to
                                           the target service's OpenAPI YAML
    validation_errors             (list[dict]) — correction errors from Validator (empty on first)
    validator_section_reflection  (list[dict]) — missing-rules decided by Validator (empty on first)
    current_section_idx     (int)        — index of the section to process
    section_iteration_count (int)        — how many times this section has been attempted

State writes:
    raw_rules               (list[dict]) — RawRule dicts (section_id, section_title, rule_type,
                                          source_name, rule_text, openapi_mapping{object, field, value})
    section_iteration_count (int)        — incremented by 1
"""

import copy
import re

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.llm_config import llm
from openapi_rulesbank.config.settings import (
    SPEC_RETRIEVE_CHUNKS,
    SPEC_SIBLING_RETRIEVE_CHUNKS,
    RULES_EXTRACTOR_RETRIEVE_CHUNKS,
)
from openapi_rulesbank.graph.state import RuleBankState
from openapi_rulesbank.prompts.extractor_prompts import SectionRules, extractor_prompt
from openapi_rulesbank.prompts.section_analyzer_prompts import (
    SectionImports,
    SectionAnalysis,
    SectionScope,
    section_import_ref_prompt,
    section_analyzer_prompt,
    section_scope_prompt,
)
from openapi_rulesbank.schemas.contract import format_contract_summary
# The sibling summary is shared: the triage resolves an anchor from it, and the
# Reflector and Validator need the same clause titles to check a name against its
# defining heading rather than against its shape.
from openapi_rulesbank.utils.parsers import build_sibling_summary
from openapi_rulesbank.tools.rag_tools import (
    search_openapi_reference,
    search_target_spec,
    search_rules_by_text,
    search_imported_type_ref,
    # A context reference names no type, so the type-shaped query that
    # search_imported_type_ref builds ("<name> type: object properties required")
    # does not fit it: what it points at is a procedure or a design rule, not a
    # schema block. It goes to the corpus with the clause's own words instead.
    search_3gpp_corpus,
    resolve_external_filename,
    umbrella_mns_for as _umbrella_mns,
)

logger = get_logger(__name__)


def _build_sections_index(parsed_sections: list[dict]) -> dict[str, dict]:
    """
    Builds a lookup dict from section_id to section dict for O(1) access.
    """
    return {s["section_id"]: s for s in parsed_sections}


def _format_ref_memory(ref_memory: dict) -> str:
    """Render the reference memory ({type: {spec, fields}}) for a prompt."""
    if not ref_memory:
        return "None yet."
    lines = []
    for t, info in ref_memory.items():
        spec = info.get("spec", "")
        fields = info.get("fields", [])
        f = f" fields={fields}" if fields else ""
        # Origin settled as this document: an internal $ref.
        if info.get("origin") == "local":
            lines.append(f"{t} -> LOCAL (defined in this document)")
            continue
        # Used earlier, but nothing has stated where it is defined yet — neither
        # local nor imported may be assumed; rules that $ref it are held.
        if not spec:
            lines.append(f"{t} -> [ORIGIN NOT STATED yet — rules that $ref it are "
                         f"held until a section states where it is defined]{f}")
            continue
        # The YAML filename a retrieval revealed. The spec number alone does not
        # supply it — one specification publishes several files. But the clause
        # that named the spec already settled that the type is EXTERNAL, and that
        # has to survive a missing filename: the internal form is what a LOCAL
        # type takes, so falling back to it discards the fact the document stated
        # and emits a rule asserting the opposite.
        file = info.get("file", "")
        # Several files possible (the lookup found the type in more than one):
        # show them, so the model can pick one by context (Phase 2, Q2b).
        options = info.get("file_options") or []
        if file:
            where = f"{spec} [{file}]"
        elif options:
            where = (f"{spec} [EXTERNAL — filename AMBIGUOUS, choose by context "
                     f"among: {', '.join(options)}]")
        else:
            where = f"{spec} [EXTERNAL — filename not known yet]"
        lines.append(f"{t} -> {where}{f}")
    return "\n".join(lines)


def _analyze_section(
    section_id: str,
    section_title: str,
    section_content: str,
    ref_memory: dict,
    sibling_summary: str = "",
    helper_context: str = "",
    contract_summary: str = "",
    annex_text: str = "",
) -> tuple[SectionAnalysis | None, dict]:
    """Triage a section before extraction: build reference memory + extractor guidance.

    Phase 1 (find & fetch): read the section, with the running reference memory in
    view, and list every type it references from another spec — each with a query
    to fetch that type's own definition — plus a note on why the other spec is
    mentioned. Any reference is a trigger to look it up, so a 'General' clause's
    imports and a 'Type' clause's embedded base both get investigated. The node
    runs the queries (RAG).

    Phase 2 (write memory & guide): given the section, the memory, Phase 1's
    explanation and the fetched definitions, it (A) returns memory updates — each
    referenced type with its spec, its own fields (from the fetch) and a note, so
    later sections recognise a type they embed by field overlap; and (B) classifies
    the section and drafts the extractor guidance (schema home or not, embedded
    base -> one allOf $ref + inherited fields).

    Returns (analysis, new_refs, new_cross_refs); new_cross_refs are the
    relationships the fetched text stated, {subject: {fact, source}}, merged into
    cross_ref_facts so a later section reads what this one learned.
    new_refs are memory entries {type: {spec, fields,
    note}} to merge into other_spec_ref_memory. analysis is None if a call fails
    (extraction then proceeds without the extra guidance).
    """
    ref_memory = ref_memory or {}
    try:
        logger.info(
            f"[analyzer {section_id}] memory in: "
            f"{ {t: v.get('fields', []) for t, v in ref_memory.items()} }"
        )
        # Phase 1 — find references to other specs and how to look each one up.
        p1_chain = section_import_ref_prompt | llm.with_structured_output(SectionImports)
        imports: SectionImports = p1_chain.invoke({
            "contract_summary": contract_summary,
            "ref_memory":      _format_ref_memory(ref_memory),
            "section_title":   section_title,
            "section_content": section_content,
        })
        # Both lists are logged: the split between them is what has to be verified,
        # and a type wrongly recorded as imported is invisible when only the type
        # list is printed.
        logger.info(
            f"[analyzer {section_id}] phase1 refs: "
            f"{[(r.type_name, r.source_spec) for r in imports.references]} "
            f"| context: {[(c.source_spec, c.subject[:40]) for c in imports.context_references]}"
        )

        # Run the RAG for every reference the section makes. Both kinds are looked
        # up; they differ in what the result means downstream.
        rag_blocks = []
        # Imported types: fetch the type's own definition block.
        for ref in imports.references:
            # The definition query is built from the type name and key_terms
            # alone, so it runs even for a CHECK 4 pending import whose spec is
            # not known yet (source_spec == "").
            fetched = search_imported_type_ref(ref.type_name, ref.key_terms)
            if fetched:
                origin = ref.source_spec or "spec not yet known"
                rag_blocks.append(
                    f"### TYPE {ref.type_name} (defined in {origin}):\n{fetched}"
                )
        # Context references: a spec the clause points at for procedure or design
        # rules. Worth fetching — that text often settles how a rule must be
        # written — but it defines no type, so its block is labelled so phase 2
        # does not turn it into a type-memory entry.
        for cref in imports.context_references:
            fetched = search_3gpp_corpus(f"{cref.subject} {cref.source_spec}")
            if fetched:
                rag_blocks.append(
                    f"### CONTEXT from {cref.source_spec} — {cref.subject} "
                    f"(no type defined here):\n{fetched}"
                )
        rag_results = "\n\n".join(rag_blocks) or "No definitions retrieved."

        # Resolve each referenced type's YAML filename BEFORE Phase 2 runs. Phase 2
        # decides what to defer by whether the filename is known; resolving only
        # afterwards made it defer a type the annex had already answered, and the
        # rule was lost. Cascade: helper_context > document annex > corpus RAG —
        # the filename is only ever READ, never derived from a spec number.
        resolved: dict[str, tuple[str, list[str], str]] = {}
        for ref in imports.references:
            if not ref.type_name or ref.type_name in resolved:
                continue
            # Origin not stated (CHECK 4): local or imported is a question for the
            # spec's text, not for the annex — a filename is sought only once the
            # type is known to be imported.
            if not ref.source_spec:
                continue
            if ((ref_memory or {}).get(ref.type_name) or {}).get("file"):
                continue  # already known from an earlier section
            resolved[ref.type_name] = resolve_external_filename(
                ref.type_name, ref.source_spec, helper_context, annex_text,
            )
            f, opts, src = resolved[ref.type_name]
            if f:
                logger.info(f"[analyzer {section_id}] filename for "
                            f"{ref.type_name!r}: {f!r} (via {src})")
            elif opts:
                logger.info(f"[analyzer {section_id}] filename for "
                            f"{ref.type_name!r} ambiguous ({src}): {opts}")

        # The memory Phase 2 sees: what earlier sections recorded, plus the
        # filename (or the ambiguous options) just resolved for this section's
        # references, so its Q2/Q3 read the filename as KNOWN where it is.
        spec_of = {r.type_name: r.source_spec for r in imports.references}
        memory_view = {k: dict(v or {}) for k, v in (ref_memory or {}).items()}
        for t, (f, opts, src) in resolved.items():
            entry = memory_view.setdefault(t, {"spec": spec_of.get(t, "")})
            if f:
                entry["file"] = f
            elif opts:
                entry["file_options"] = opts

        # Serialize Phase 1's per-type filename presence (its own output) for the
        # Phase 2 prompt — same transport as phase1_explanation just above. Phase 2
        # uses it to tell a type it can reference completely now from one whose rule
        # must be deferred. No decision here: filename_in_section is Phase 1's call.
        filename_presence = "\n".join(
            f"  - {p.type_name}: "
            + (f"filename in THIS section = {p.filename}" if p.filename_in_section
               else "filename NOT in this section")
            for p in imports.filename_presence
        ) or "None."

        # Phase 2 — write the memory updates and the extractor guidance.
        p2_chain = section_analyzer_prompt | llm.with_structured_output(SectionAnalysis)
        analysis: SectionAnalysis = p2_chain.invoke({
            "contract_summary": contract_summary,
            "section_id":         section_id,
            "section_title":      section_title,
            "section_content":    section_content,
            "ref_memory":         _format_ref_memory(memory_view),
            "phase1_explanation": imports.explanation or "None.",
            "filename_presence":  filename_presence,
            "rag_results":        rag_results,
            "sibling_summary":    sibling_summary or "No sibling sections summarized yet.",
            # First step of the callback-anchor cascade: a decision stated here
            # outranks the document, the convention and the OpenAPI reference.
            "helper_context":     helper_context or "None provided.",
        })

        # A later section that reports a type it merely USES knows its spec from
        # the memory but may retrieve nothing carrying the filename — and writing
        # the entry wholesale would then erase a filename an earlier section had
        # already established. Keep whichever side actually has each field: a
        # known filename is a fact, and a section that did not find one has not
        # refuted it. (Measured: DateTime gained its filename in the clause that
        # declares the imports and reached the end of the run without it.)
        prev = ref_memory or {}
        new_refs = {}
        for e in analysis.memory_updates:
            if not e.type_name:
                continue
            was = prev.get(e.type_name, {}) or {}
            spec = e.source_spec or was.get("spec", "")
            note = e.note or was.get("note", "")
            # Filename as the section/memory already has it. Phase 2 writes one only
            # when THIS section's text spells it out; a using section usually has none.
            file = e.source_file or was.get("file", "")
            # Still missing: use the cascade result computed before Phase 2 (or
            # run it now for a type Phase 2 added that Phase 1 did not reference).
            # Its source goes into the note, so the Reflector and the bank show
            # where the filename came from. Ambiguous: Phase 2 was shown the
            # options and should have written one (Q2b); if it did not, the
            # options are recorded and the file stays empty.
            if not file and e.type_name:
                found, options, src = resolved.get(e.type_name) or \
                    resolve_external_filename(
                        e.type_name, spec, helper_context, annex_text)
                if found:
                    file = found
                    note = (note + " | " if note else "") + \
                        f"filename '{file}' resolved via {src}"
                elif options:
                    note = (note + " | " if note else "") + \
                        f"filename AMBIGUOUS via {src}; choose by context among: " \
                        + ", ".join(options)
            # Start from what the entry already holds (origin, held rules), so the
            # right-wins reducer does not drop it when this section rewrites it.
            new_refs[e.type_name] = {
                **was,
                "spec":   spec,
                "file":   file,
                "fields": e.fields or was.get("fields", []),
                "note":   note,
            }
        # A filename the cascade resolved must not be lost because Phase 2 left the
        # type out of memory_updates: record it, keeping what memory already has.
        for t, (found, _opts, src) in resolved.items():
            if not found or (new_refs.get(t) or {}).get("file"):
                continue
            was = dict(new_refs.get(t) or prev.get(t) or {})
            was.setdefault("spec", spec_of.get(t, ""))
            was["file"] = found
            was["note"] = (was.get("note", "") + " | " if was.get("note") else "") + \
                f"filename '{found}' resolved via {src}"
            new_refs[t] = was

        # Origin of each type, as this section's text settles it — the fact a held
        # rule waits for, and the one later sections read. External: paired with a
        # spec. Local: the text shows it is defined in this document (always
        # recorded — a later section that uses the type needs to know). Not stated:
        # tracked so its rules are held, never presumed either way.
        def _entry(t: str) -> dict:
            return dict(new_refs.get(t) or prev.get(t) or {})

        for r in imports.references:
            if not r.type_name:
                continue
            entry = _entry(r.type_name)
            if r.source_spec:
                entry.setdefault("spec", r.source_spec)
                entry["spec"] = entry.get("spec") or r.source_spec
                entry["origin"] = "external"
            elif entry.get("origin") not in ("local", "external") and not entry.get("spec"):
                entry.update({"spec": "", "file": "", "origin": "unknown"})
            new_refs[r.type_name] = entry
        for t in (imports.local_types or []):
            if not t:
                continue
            entry = _entry(t)
            entry.update({"origin": "local", "spec": "", "file": ""})
            new_refs[t] = entry

        # Every type whose origin is still not stated is deferred, whatever Phase 2
        # wrote — the decision is the text's, and here the text said nothing.
        unknown = [t for t, v in new_refs.items() if (v or {}).get("origin") == "unknown"]
        if unknown:
            analysis.defer_external_refs = sorted(
                set(analysis.defer_external_refs or []) | set(unknown)
            )
        new_cross_refs = {
            c.subject: {"fact": c.fact, "source": c.source}
            for c in (analysis.cross_refs or []) if c.subject and c.fact
        }
        if new_cross_refs:
            logger.info(f"[analyzer {section_id}] cross-refs: {new_cross_refs}")
        # Which step of the cascade answered, so the decision is visible in a run.
        if analysis.callback_anchor or analysis.callback_anchor_source:
            logger.info(
                f"[analyzer {section_id}] callback anchor: "
                f"{analysis.callback_anchor!r} (from {analysis.callback_anchor_source})"
            )
        logger.info(
            f"[analyzer {section_id}] phase2: kind={analysis.section_kind} "
            f"emits_schema={analysis.emits_schema_properties} "
            f"base_ref={analysis.base_ref!r} inherited={analysis.inherited_fields} "
            # What this clause maps, which is what a later section reads off the
            # sibling list. Logged because an empty one is invisible otherwise:
            # the later section simply has nothing to anchor on and no trace says why.
            f"maps={analysis.maps_operation!r}:{analysis.maps_method!r} "
            f"{analysis.maps_path!r} "
            f"| memory_updates: { {t: (v or {}).get('fields', []) for t, v in new_refs.items()} }"
        )
        return analysis, new_refs, new_cross_refs
    except Exception as e:
        logger.warning(f"Section analyzer failed for [{section_id}] ({e}); extracting without it.")
        return None, {}, {}


def _scope_section(
    section_id: str,
    section_title: str,
    section_content: str,
    section_verdict: str,
    existing_rules: str,
    sibling_summary: str,
    contract_summary: str = "",
) -> tuple[str, str]:
    """Phase 3 — the section's candidate rules (from its text) and the skip list.

    Always runs. Anchored in the section's own content, it lists the rules the
    section supports (candidate_rules) and marks the subset another section already
    owns (skip). The candidates are a HINT, not a ceiling: they help the Extractor
    stay in the section's content (and not mine the RAG context for rules the text
    does not state), while the Extractor still discerns and may emit a legitimate
    rule the analysis did not list. Returns (candidate_str, skip_str) for the
    Extractor prompt; on failure, neutral text so extraction still proceeds.
    """
    try:
        chain = section_scope_prompt | llm.with_structured_output(SectionScope)
        scope: SectionScope = chain.invoke({
            "contract_summary": contract_summary,
            "section_id":      section_id,
            "section_title":   section_title,
            "section_content": section_content,
            "section_verdict": section_verdict,
            "sibling_summary": sibling_summary or "No sibling sections listed.",
            "existing_rules":  existing_rules or "None yet.",
        })
        logger.info(f"[scope {section_id}] candidates: {scope.candidate_rules} | skip: {scope.skip}")
        if scope.candidate_rules:
            candidates = "\n".join(f"  - {c}" for c in scope.candidate_rules)
        else:
            candidates = ("The prior analysis found no rule anchored in this section's "
                          "text. Extract only if the content clearly supports a rule; "
                          "do NOT pull rules from the retrieved context.")
        skip = "\n".join(f"  - {s}" for s in scope.skip) or "None — no candidate already exists elsewhere."
        return candidates, skip
    except Exception as e:
        logger.warning(f"Section scope failed for [{section_id}] ({e}); extracting without pre-scope.")
        return ("No pre-scoped candidates available — extract every rule THIS section's "
                "content states.", "None.")


def _format_cross_refs(facts: dict) -> str:
    """Render the accumulated cross-reference facts for a prompt."""
    if not facts:
        return "None."
    lines = []
    for subject, info in facts.items():
        src = f" ({info['source']})" if info.get("source") else ""
        lines.append(f"  - {subject}: {info.get('fact', '')}{src}")
    return "\n".join(lines)


def _format_analysis_guidance(analysis: SectionAnalysis | None) -> str:
    """Render the analyzer's verdict as directives injected into the main prompt."""
    if analysis is None:
        return "No pre-analysis available; apply the general rules."
    lines = [f"Section kind: {analysis.section_kind}."]

    if analysis.emits_schema_properties:
        lines.append("This section IS a schema home: emit its schema_property rules.")
        if analysis.schema_name:
            lines.append(f"Canonical schema name (exact casing): {analysis.schema_name}.")
        if analysis.base_ref:
            inh = ', '.join(analysis.inherited_fields)
            lines.append(
                f"EMBEDDED BASE — this type is allOf of an imported base plus its own "
                f"properties. Emit EXACTLY ONE schema_property with openapi_field "
                f"'allOf' and value \"{analysis.base_ref}\". These fields are INHERITED "
                f"from that base: [{inh}] — do NOT emit any of them as "
                f"properties.<name>, and do NOT put them in 'required'. Emit "
                f"properties only for the REMAINING (own) attributes."
            )
        if analysis.guidance:
            lines.append(analysis.guidance)
    else:
        # The structured flag wins over the prose. A non-home section must emit no
        # schema_property whatever the free-text guidance says, so we state the hard
        # rule and omit the analyzer's prose (which has slipped into "emit ... schema
        # with properties ..." and misled the Extractor).
        lines.append(
            "This section is NOT a schema home — do NOT emit ANY schema_property "
            "rule (no properties.<name>, no allOf, no enum, no required). Emit only "
            "its HTTP-level rules (path_operation, callback, request_body, response). "
            "Any attributes it lists are defined in their own 'Type <Name>' section."
        )

    # The anchor was decided in the triage, which sees the sibling clauses this
    # section cannot. Hand it over resolved: the Extractor writes callbacks under
    # it and does not work one out for itself.
    if analysis.callback_anchor:
        lines.append(
            f"CALLBACK ANCHOR (decided in pre-analysis, source: "
            f"{analysis.callback_anchor_source}): this section's notification is "
            f"delivered to a consumer-supplied address, so it is a callback. Use "
            f"\"{analysis.callback_anchor}\" as the anchor — emit the callback rule "
            f"as \"{analysis.callback_anchor}.callbacks.<notificationName>\", and "
            f"its request_body and response rules under that same anchor. Do NOT "
            f"emit a path_operation for the delivery address, and do NOT resolve "
            f"an anchor of your own."
        )
    elif analysis.callback_anchor_source == "unresolved":
        lines.append(
            "CALLBACK ANCHOR: this section's notification is delivered to a "
            "consumer-supplied address, but the pre-analysis could not resolve the "
            "operation to anchor it on. Emit what the section does support and "
            "report the missing anchor in missing_context — never a placeholder."
        )

    # Types whose $ref cannot be written yet: origin not stated in the text, or
    # external with no filename. A rule pointing at one would come out in the
    # internal form, asserting the type is local. The Extractor HOLDS those rules
    # (deferred_rules); the code keeps them and the section that settles the type
    # emits them, with the $ref completed.
    if analysis.defer_external_refs:
        names = ", ".join(analysis.defer_external_refs)
        lines.append(
            f"HOLD THESE $refs — where these types are defined is not settled yet: "
            f"[{names}]. A rule whose $ref points at one of them goes in "
            f"`deferred_rules`, NOT in `rules`. Write it completely, exactly as you "
            f"would otherwise, with the $ref written as "
            f"\"$ref: '#/components/schemas/<Type>'\" and the type in `references` "
            f"(file empty): the code completes the target (internal, or the external "
            f"file) once a section states where the type is defined, and emits the "
            f"rule then. Emit every OTHER rule in `rules` as normal."
        )
    return "\n".join(lines)


def _format_rules_compact(rules: list[dict]) -> list[str]:
    """One line per rule: type, name, mapping — enough to recognise it, no prose."""
    lines = []
    for r in rules:
        m = r.get("openapi_mapping", {}) or {}
        lines.append(
            f"  [{r.get('rule_type', '?')}] {r.get('source_name', '?')} -> "
            f"{m.get('openapi_object', '')} {m.get('openapi_field', '')} "
            f"{m.get('openapi_value', '')}".rstrip()
        )
    return lines


def _build_correction_task(
    validation_errors: list[dict],
    section_reflection: list[dict],
    already_validated: list[dict] | None = None,
    discarded_errors: list[dict] | None = None,
    retrieved_context: str = "",
) -> str:
    """
    Returns a correction instruction for retry attempts on a section.

    Lists, in order: the facts fetched for the gaps this section declared last
    time, the rules already validated and the ones the Validator discarded
    (neither must be produced again), the rules that failed and how to fix them,
    and the missing rules to add. Without those lists the LLM cannot honour "do
    not duplicate", so it re-extracts the whole section on every retry —
    including rules just thrown away.

    retrieved_context belongs here rather than among the standing reference
    contexts: it answers what THIS section reported missing on its previous
    attempt, so it exists only on a retry and is read as part of the correction,
    next to the rules it is meant to fix.

    Returns "" on the first attempt (no errors, no missing rules, no fetched
    context).
    """
    if not validation_errors and not section_reflection and not retrieved_context:
        return ""

    lines = [
        "CORRECTION TASK — this section has ALREADY been extracted.",
        "Return ONLY the rules listed under 'RULES TO CORRECT' and 'RULES TO EXTRACT",
        "FROM SCRATCH' below. Do NOT re-extract the section from the beginning.",
        "You may produce more or fewer rules than before — quality over quantity.",
        "",
        "WHERE AN INSTRUCTION NAMES A VALUE, WRITE THAT VALUE. An instruction that",
        "states it (\"Set openapi_value to application/merge-patch+json\", \"Fix",
        "openapi_field from 200 to 201\") is the DECISION for that field: copy it",
        "literally. It outranks the modelling conventions and the OpenAPI reference,",
        "which are the default for a field nobody has decided — this one has now",
        "been decided. Re-deriving the field from the convention instead reproduces",
        "the exact value that was just rejected, and the section spends its",
        "remaining attempts re-emitting it.",
        "Where an instruction names NO value — it asks you to apply a fact, resolve",
        "a gap or consult a source — there is nothing new to write: keep the rule as",
        "it stands and report in missing_context that the correction named no value.",
        "Where an instruction says to MOVE a rule to deferred_rules (its $ref points",
        "at a held type), put that rule in deferred_rules, written the same way, and",
        "leave it out of rules — that is the correction.",
        "",
    ]

    if retrieved_context:
        lines += [
            "ANSWERS TO WHAT YOU REPORTED MISSING — each block names the gap it",
            "answers and was fetched for it. Read the block for a CONCRETE VALUE:",
            "the media type, the status code, the path key, the operation name it",
            "supplies. Where it gives one, write that value into the rule whose",
            "mapping you had fallen back to for want of it. Where a block states no",
            "value — it discusses the subject without settling the field — the rule",
            "you already emitted stands: do not reword it to look answered, and say",
            "again in missing_context that the fact is still missing. A value you",
            "took from a convention because the clause is silent is NOT wrong and is",
            "not what these blocks are here to overturn; only a block naming a",
            "different concrete value changes it.",
            "Reference only — never extract a rule of the section this text was",
            "pulled from.",
            "",
            retrieved_context,
            "",
        ]

    if already_validated:
        lines.append(
            "ALREADY VALIDATED — these rules exist and are correct. "
            "Do NOT produce them again, in any form:"
        )
        lines += _format_rules_compact(already_validated)
        lines.append("")

    if discarded_errors:
        lines.append(
            ">>>>> FORBIDDEN RULES — DO NOT EMIT ANY OF THESE, IN ANY FORM <<<<<")
        lines.append(
            "The Validator already REJECTED each rule below and it MUST NOT exist. "
            "These are not rules to fix — they are BANNED. Producing any of them "
            "again (same target, same field, reworded, split, or renamed) is a hard "
            "error: the rule will be rejected again and the whole extraction fails "
            "its purpose. Each was rejected because it does not belong to THIS "
            "section (e.g. an inherited field that comes from a base type via allOf, "
            "or a rule another section owns). Leave every one of them OUT:")
        for i, err in enumerate(discarded_errors, start=1):
            rule = err.get("rule", {})
            mapping = rule.get("openapi_mapping", {})
            lines.append(
                f"  {i}. BANNED [{rule.get('rule_type', '?')}] "
                f"object={mapping.get('openapi_object', '?')} "
                f"field={mapping.get('openapi_field', '?')} "
                f"(source={rule.get('source_name', '?')})\n"
                f"     reason: {err.get('instruction', 'discarded')}"
            )
        lines.append(
            ">>>>> END FORBIDDEN RULES — none of the above may appear in your output <<<<<")
        lines.append("")

    if validation_errors:
        lines.append("RULES TO CORRECT (existing rules that failed):")
        for i, err in enumerate(validation_errors, start=1):
            rule = err.get("rule", {})
            lines.append(
                f"  {i}. [{err.get('section_id', rule.get('section_id', '?'))}] "
                f"{err.get('error_type', 'correction').upper()}\n"
                f"     original rule_text : {rule.get('rule_text', '?')!r}\n"
                f"     rule_type          : {rule.get('rule_type', '?')}\n"
                f"     source_name        : {rule.get('source_name', '?')!r}\n"
                f"     instruction        : {err.get('instruction', '?')}"
            )

    if section_reflection:
        lines.append("")
        lines.append("RULES TO EXTRACT FROM SCRATCH (missing rules identified by Reflector):")
        for sr in section_reflection:
            for j, missing in enumerate(sr.get("missing_rules", []), start=1):
                lines.append(f"  [{sr['section_id']}] {j}. {missing}")

    n_fix     = len(validation_errors)
    n_missing = sum(len(sr.get("missing_rules", [])) for sr in section_reflection)
    lines += [
        "",
        f"EXPECTED OUTPUT: {n_fix} corrected rule(s) and {n_missing} new rule(s) — "
        "a corrected or new rule whose $ref points at a held type goes in "
        "deferred_rules, the rest in rules. "
        "Anything already listed as validated above must not appear in your output.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Held rules — a rule whose $ref points at a type whose origin is not settled yet
# is kept under that type in other_spec_ref_memory ("pending_rules") and emitted
# by the section that settles it, with the $ref target completed by code.
# ---------------------------------------------------------------------------

_SCHEMA_TARGET = re.compile(r"#/components/schemas/([A-Za-z0-9_\-]+)")


def _ref_targets(rule: dict) -> list[str]:
    """The schema names a rule's $ref points at: `references` first, else the value."""
    m = rule.get("openapi_mapping") or {}
    names = [r.get("schema_name") for r in (m.get("references") or []) if r.get("schema_name")]
    return names or _SCHEMA_TARGET.findall(str(m.get("openapi_value", "")))


def _settled_file(entry: dict) -> str | None:
    """Where a held type's $ref now points: "" for local, the YAML file for an
    import whose file is known, None while its origin (or file) is still open."""
    if (entry or {}).get("origin") == "local":
        return ""
    if (entry or {}).get("spec") and (entry or {}).get("file"):
        return entry["file"]
    return None


def _complete_held_rule(rule: dict, type_name: str, file: str) -> dict:
    """The held rule with its $ref to `type_name` pointed at the settled target."""
    r = copy.deepcopy(rule)
    m = r.setdefault("openapi_mapping", {})
    target = f"{file}#/components/schemas/{type_name}"
    m["openapi_value"] = re.sub(
        rf"[A-Za-z0-9_.\-]*#/components/schemas/{re.escape(type_name)}(?![A-Za-z0-9_])",
        target, str(m.get("openapi_value", "")),
    )
    refs = [dict(x) for x in (m.get("references") or [])]
    for x in refs:
        if x.get("schema_name") == type_name:
            x["file"] = file
    if not any(x.get("schema_name") == type_name for x in refs):
        refs.append({"file": file, "schema_name": type_name})
    m["references"] = refs
    return r


def extractor_node(state: RuleBankState) -> dict:
    """
    Processes the current section (current_section_idx) from the extraction plan.

    First attempt (section_iteration_count == 0): fresh extraction.
    Retry (section_iteration_count > 0): correction extraction with CORRECTION TASK prompt.

    After the section's retry loop ends, the Builder advances current_section_idx
    and resets section_iteration_count so the next call here starts a fresh section.
    """
    logger.info("Extractor Node started.")

    sections_index          = _build_sections_index(state["parsed_sections"])
    all_plan_sections       = state["extraction_plan"]["sections_to_extract"]
    helper_context          = state.get("helper_context", "") or "No auxiliary context provided."
    contract_summary        = (
        format_contract_summary(state.get("service_contract", {}))
        or "No service contract provided."
    )
    all_validation_errors = state.get("validation_errors", []) or []
    # correction + split are the ones the Extractor must act on
    validation_errors  = [
        e for e in all_validation_errors if e.get("error_type") != "discard"
    ]
    # discards need no fix, but the Extractor must know about them — otherwise it
    # re-extracts on the next retry exactly what the Validator just threw away
    # (duplicates of other sections, unmappable statements, ...).
    discarded_errors   = [
        e for e in all_validation_errors if e.get("error_type") == "discard"
    ]
    # Final missing-rules decided by the Validator (written to validator_section_reflection)
    section_reflection = state.get("validator_section_reflection", []) or []
    section_iteration_count = state.get("section_iteration_count", 0) or 0
    current_section_idx     = state.get("current_section_idx", 0) or 0

    if current_section_idx >= len(all_plan_sections):
        logger.error(
            f"current_section_idx={current_section_idx} out of bounds "
            f"(plan has {len(all_plan_sections)} sections). Returning empty raw_rules."
        )
        return {
            "raw_rules":               [],
            "section_iteration_count": section_iteration_count + 1,
        }

    plan_section = all_plan_sections[current_section_idx]
    section_id   = plan_section["section_id"]

    # Facts fetched for the gaps this section declared on its previous attempt.
    # Empty on the first one, so it only ever reaches a retry.
    retrieved_context = (state.get("retrieved_context", {}) or {}).get(section_id, "")

    if validation_errors or section_reflection or retrieved_context:
        # Rules the Validator already accepted for this section — the Extractor must
        # see them to honour "do not duplicate"; without them it re-extracts the
        # whole section on every retry.
        already_validated = (
            state.get("validated_rules_by_section", {}) or {}
        ).get(section_id, [])
        correction_task = _build_correction_task(
            validation_errors, section_reflection, already_validated, discarded_errors,
            retrieved_context,
        )
        logger.info(
            f"Retry {section_iteration_count + 1} for [{section_id}] — "
            f"{len(validation_errors)} error(s), "
            f"{sum(len(sr.get('missing_rules', [])) for sr in section_reflection)} missing rule(s)"
            f"{', with fetched context' if retrieved_context else ''}."
        )
    else:
        correction_task = ""

    structured_llm = llm.with_structured_output(SectionRules)
    chain = extractor_prompt | structured_llm

    section = sections_index.get(section_id)
    if section is None:
        logger.warning(
            f"Section '{section_id}' not found in parsed_sections — returning empty raw_rules."
        )
        return {
            "raw_rules":               [],
            "section_iteration_count": section_iteration_count + 1,
        }

    section_title = plan_section["title"]
    focus         = plan_section["extraction_focus"]

    # Retrieve relevant OpenAPI spec chunks for this section via RAG
    rag_query = f"{section_title} — {focus}"
    openapi_reference_overview = search_openapi_reference(rag_query)

    # Retrieve related chunks from the 3GPP document itself (temporary spec RAG).
    # A section is extracted in isolation, so two things it needs live in OTHER
    # sections of the same service: the canonical IS operation name (stage 2), and
    # the resource path, schema name/casing and property list (the sibling stage 3
    # sections). We fetch each with its own scoped query so both signals are
    # present and neither crowds out the other. Both are scoped to the same MnS to
    # stay off other services' look-alike sections.
    spec_collection = state.get("spec_collection_name", "")
    target_mns      = _umbrella_mns(section_title, state["parsed_sections"])
    # The section names the resources, types and operations it refers to, and the
    # sibling sections are titled after those names — so seeding the query with the
    # section's own text (title, the Planner's focus, and the opening of the body)
    # is what pulls the right siblings. A short per-stage intent hint biases
    # retrieval toward the abstract definition (stage 2) or the concrete
    # resource/schema sections (stage 3) without parsing any names out.
    seed = f"{section_title} {focus} {section['content'][:300]}"

    def _scoped(stage: str, hint: str, k: int) -> str:
        filters = {"stage": stage}
        if target_mns:
            filters["mns"] = target_mns
        return search_target_spec(f"{seed} {hint}", spec_collection, k=k, filters=filters)

    stage2_context = _scoped(
        "2", "abstract IS operation and notification definition",
        SPEC_RETRIEVE_CHUNKS,
    ) or "No stage 2 definition retrieved."
    stage3_context = _scoped(
        "3", "resource URI path, HTTP method, data type and schema definition",
        SPEC_SIBLING_RETRIEVE_CHUNKS,
    ) or "No related stage 3 section retrieved."

    # Triage the section up front (two focused steps): classify it (definition vs
    # mapping) and resolve any imported base type into an allOf $ref + inherited
    # fields, so the main extraction receives those subtle judgements ready-made.
    # It reads this section's own declared imports and reuses the running memory of
    # references seen in earlier sections (a 'General' clause feeds the later
    # 'Type' clauses); new_refs are merged back into that memory below.
    # It also gets a light summary of the SIBLING sections in the same service (id +
    # title + the kind of those already classified), so it can judge this section
    # relative to its neighbours (e.g. a Type section owns a schema a Mapping
    # section only references) without their content bloating the prompt.
    sibling_summary = build_sibling_summary(
        section_title, state["parsed_sections"], state.get("section_analysis_by_id", {}),
        exclude_id=section_id,
    )
    analysis, new_refs, new_cross_refs = _analyze_section(
        section_id, section_title, section["content"],
        state.get("other_spec_ref_memory", {}),
        sibling_summary,
        helper_context,
        contract_summary,
        state.get("annex_text", "") or "",
    )
    section_analysis = _format_analysis_guidance(analysis)
    # What earlier sections learned about how things relate, plus whatever this
    # section's own triage just added. A section maps its rules from its own text;
    # these are the facts that text depends on and does not contain.
    cross_ref_context = _format_cross_refs({
        **(state.get("cross_ref_facts", {}) or {}), **new_cross_refs,
    })

    # Rules already validated in OTHER sections that resemble this one. Fed to the
    # scope phase (below), NOT straight to the Extractor: handing the raw list to
    # the big prompt made it self-censor and drop the whole section. Empty until
    # earlier sections are validated.
    existing_rules = search_rules_by_text(
        seed,
        state.get("rules_collection_name", ""),
        k=RULES_EXTRACTOR_RETRIEVE_CHUNKS,
        exclude_section_id=section_id,
    ) or "None yet."

    # Phase 3 — scope: anchored in the section's text, list the candidate rules the
    # section supports (a hint that keeps the Extractor in the content, not the RAG
    # context) and the subset another section already owns (skip). The Extractor
    # still discerns and may emit a legitimate rule not listed.
    candidate_rules, skip_rules = _scope_section(
        section_id, section_title, section["content"],
        section_analysis, existing_rules, sibling_summary,
        contract_summary,
    )

    logger.debug(
        f"Extracting [{section_id}] '{section_title}' "
        f"(priority={plan_section['priority']}, attempt={section_iteration_count + 1})."
    )

    result: SectionRules = chain.invoke({
        "section_id":                 section_id,
        "section_title":              section_title,
        "extraction_focus":           focus,
        "section_analysis":           section_analysis,
        "section_content":            section["content"],
        "openapi_reference_overview": openapi_reference_overview,
        "cross_ref_context":          cross_ref_context,
        # Which spec each imported type comes from, and the YAML filename a
        # retrieval revealed for it. The rule that writes an external $ref needs
        # that filename, and until now only base_ref carried it — so a type used
        # outside an allOf fell back to an internal reference. This section's own
        # findings are merged in, so a type first seen here is usable here.
        "ref_memory":                 _format_ref_memory({
            **(state.get("other_spec_ref_memory", {}) or {}), **new_refs,
        }),
        # The types this section must hold, named where the $ref form is decided
        # (REFERENCE CASES, case 3) — filled per section from its own analysis.
        "held_types":                 ", ".join(
            (analysis.defer_external_refs or []) if analysis else []) or "none",
        "stage2_context":             stage2_context,
        "stage3_context":             stage3_context,
        "candidate_rules":            candidate_rules,
        "skip_rules":                 skip_rules,
        "helper_context":             helper_context,
        "contract_summary":           contract_summary,
        "correction_task":            correction_task,
    })

    raw_rules = [rule.model_dump() for rule in result.rules]

    # --- Held rules: keep them (piece 1) and emit the settled ones (piece 4) ---
    # The memory as it stands after this section's own findings.
    mem_now = {k: dict(v or {}) for k, v in
               {**(state.get("other_spec_ref_memory", {}) or {}), **new_refs}.items()}
    annex_text = state.get("annex_text", "") or ""

    # Piece 1 — store each rule the Extractor held under the type it waits for.
    for held in (result.deferred_rules or []):
        r = held.model_dump()
        m = r.get("openapi_mapping") or {}
        key = (section_id, r.get("rule_type", ""),
               m.get("openapi_object", ""), m.get("openapi_field", ""))
        targets = [t for t in _ref_targets(r) if _settled_file(mem_now.get(t, {})) is None]
        for t in targets or _ref_targets(r)[:1]:
            entry = mem_now.setdefault(t, {"spec": "", "file": "", "origin": "unknown"})
            pend = [dict(p) for p in entry.get("pending_rules", [])]
            if not any(p.get("key") == list(key) for p in pend):
                pend.append({"key": list(key), "rule": r, "section_id": section_id,
                             "section_title": section_title, "completed_in": ""})
            entry["pending_rules"] = pend
            new_refs[t] = entry
            logger.info(f"[held {section_id}] {r.get('rule_type')} "
                        f"{r.get('source_name')!r} waits for {t!r}")

    # Piece 4 — a held rule whose type is now settled is completed and emitted
    # here. A held type with a known spec but no filename yet gets the cascade
    # (helper > annex > corpus > convention fallback) first.
    completed_held = []
    for t, entry in mem_now.items():
        pend = [dict(p) for p in entry.get("pending_rules", [])]
        if not pend:
            continue
        file = _settled_file(entry)
        if file is None and entry.get("spec"):
            found, _opts, src = resolve_external_filename(
                t, entry["spec"], helper_context, annex_text)
            if found:
                entry["file"], file = found, found
                entry["note"] = (entry.get("note", "") + " | " if entry.get("note") else "") \
                    + f"filename '{found}' resolved via {src}"
        if file is None:
            continue
        for p in pend:
            if p.get("completed_in") and p["completed_in"] != section_id:
                continue  # already emitted by another section
            raw_rules.append(_complete_held_rule(p["rule"], t, file))
            p["completed_in"] = section_id
            completed_held.append({
                "type": t, "rule_type": p["rule"].get("rule_type", ""),
                "source_name": p["rule"].get("source_name", ""),
                "from_section": p.get("section_id", ""),
                "settled_as": "local" if file == "" else f"external ({file})",
            })
        entry["pending_rules"] = pend
        new_refs[t] = entry
    if completed_held:
        logger.info(f"[held {section_id}] emitted here: {completed_held}")

    # Types whose filename was built by the naming convention, not read anywhere —
    # the Reflector scores their rules at the acceptance floor and says why.
    fallback_files = {
        t: v.get("file") for t, v in mem_now.items()
        if "via convention-fallback" in (v.get("note") or "") and v.get("file")
    }
    # Facts the section depended on and no context supplied. Reported here only;
    # nothing acts on them yet — the Reflector is where they will be weighed and
    # turned into a retrieval query for the next attempt.
    missing_context = list(result.missing_context or [])
    logger.info(
        f"Extractor Node complete — {len(raw_rules)} raw rule(s) from [{section_id}] "
        f"(section attempt {section_iteration_count + 1})."
    )
    if missing_context:
        logger.info(
            f"[missing-context {section_id}] {missing_context}"
        )

    # Record the pre-analysis verdict for this section so the Reflector and Validator
    # can enforce it (e.g. reject schema_property from a section that is not a schema
    # home) instead of re-judging the raw table. Keyed by section_id; the reducer
    # merges it into the running map.
    verdict_update = {}
    if analysis is not None:
        verdict_update = {section_id: {
            "section_kind":            analysis.section_kind,
            "emits_schema_properties": analysis.emits_schema_properties,
            "base_ref":                analysis.base_ref,
            "inherited_fields":        analysis.inherited_fields,
            # Which operation this clause maps, and where. A later section that
            # has to refer to an operation — a callback naming the one it anchors
            # on — reads it off the sibling summary instead of re-deriving it from
            # a clause it was never given.
            "maps_operation":          analysis.maps_operation,
            "maps_path":               analysis.maps_path,
            "maps_method":             analysis.maps_method,
            # The anchor this section's callbacks hang off, as the cascade
            # resolved it. The Reflector and the Validator judge those callback
            # rules later, and without this they see only the section's own text —
            # which names the delivery method (POST) and never the owning
            # operation. That is how a correct "…put.callbacks.X" gets "corrected"
            # to ".post": not a wrong judgement, a missing premise.
            "callback_anchor":         analysis.callback_anchor,
            "callback_anchor_source":  analysis.callback_anchor_source,
            # Phase-3 positive scope for this section, so the Reflector's
            # completeness check measures "missing" against what the section is
            # actually supposed to emit — not the raw attribute table.
            "candidate_rules":         candidate_rules,
            # Phase-3 skip list — rules another section already owns. Handed to the
            # Reflector so it never lists one of these as "missing" here.
            "skip_rules":              skip_rules,
            # External types Phase 2 says to DEFER (external, filename unknown here).
            # Read by the Extractor (do not emit their $ref, park the rule) and by the
            # Reflector/Validator (a deferred $ref is withheld on purpose, not missing).
            "defer_external_refs":     list(analysis.defer_external_refs or []),
        }}
    # Held rules emitted HERE (written in an earlier section, settled by this one)
    # and filenames built by convention. The Reflector and Validator read them so a
    # carried rule is judged by why it is emitted here, not by this section's text.
    if completed_held or fallback_files:
        verdict_update.setdefault(section_id, {}).update({
            "completed_held_rules": completed_held,
            "fallback_filenames":   fallback_files,
        })

    return {
        "raw_rules":               raw_rules,
        "section_iteration_count": section_iteration_count + 1,
        # Merge this section's newly declared other-spec references into the running
        # memory (see _merge_ref_memory), so later sections can resolve them.
        "other_spec_ref_memory":   new_refs,
        "cross_ref_facts":         new_cross_refs,
        "section_analysis_by_id":  verdict_update,
        "missing_context":         missing_context,
    }
