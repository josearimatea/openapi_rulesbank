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
from openapi_rulesbank.tools.rag_tools import (
    search_openapi_reference,
    search_target_spec,
    search_rules_by_text,
    search_imported_type_ref,
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
        lines.append(f"{t} -> {spec}{f}")
    return "\n".join(lines)


def _build_sibling_summary(
    section_title: str,
    parsed_sections: list[dict],
    analysis_by_id: dict,
    exclude_id: str,
    levels: int = 4,
) -> str:
    """A light summary of the sibling sections in the same service block.

    Siblings share the section's clause prefix at `levels` (e.g. "12.3.1.2" groups
    the performance-threshold service's clauses). Each line is id + title + the kind
    already decided for it (from section_analysis_by_id), or "not yet analyzed" for
    ones not processed yet. Titles only — no content — so the Phase 2 prompt can see
    the neighbourhood's shape without ballooning. Empty when there are no siblings.
    """
    def prefix(title: str) -> str:
        head = title.strip().split(" ", 1)[0]
        return ".".join(head.split(".")[:levels])

    my_prefix = prefix(section_title)
    if not my_prefix:
        return ""
    lines = []
    for s in parsed_sections:
        if s["section_id"] == exclude_id:
            continue
        if prefix(s["title"]) != my_prefix:
            continue
        kind = analysis_by_id.get(s["section_id"], {}).get("section_kind", "not yet analyzed")
        lines.append(f"  [{s['section_id']}] {s['title'][:60]} — {kind}")
    return "\n".join(lines)


def _analyze_section(
    section_id: str,
    section_title: str,
    section_content: str,
    ref_memory: dict,
    sibling_summary: str = "",
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

    Returns (analysis, new_refs); new_refs are memory entries {type: {spec, fields,
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
            "ref_memory":      _format_ref_memory(ref_memory),
            "section_title":   section_title,
            "section_content": section_content,
        })
        logger.info(
            f"[analyzer {section_id}] phase1 refs: "
            f"{[(r.type_name, r.source_spec) for r in imports.references]}"
        )

        # Run the RAG for each reference the section makes (a reference triggers a
        # lookup of what that type/spec is), collecting the fetched definitions.
        rag_blocks = []
        for ref in imports.references:
            fetched = search_imported_type_ref(ref.type_name, ref.key_terms)
            if fetched:
                rag_blocks.append(f"### {ref.type_name} (from {ref.source_spec}):\n{fetched}")
        rag_results = "\n\n".join(rag_blocks) or "No definitions retrieved."

        # Phase 2 — write the memory updates and the extractor guidance.
        p2_chain = section_analyzer_prompt | llm.with_structured_output(SectionAnalysis)
        analysis: SectionAnalysis = p2_chain.invoke({
            "section_id":         section_id,
            "section_title":      section_title,
            "section_content":    section_content,
            "ref_memory":         _format_ref_memory(ref_memory),
            "phase1_explanation": imports.explanation or "None.",
            "rag_results":        rag_results,
            "sibling_summary":    sibling_summary or "No sibling sections summarized yet.",
        })

        new_refs = {
            e.type_name: {"spec": e.source_spec, "fields": e.fields, "note": e.note}
            for e in analysis.memory_updates if e.type_name
        }
        logger.info(
            f"[analyzer {section_id}] phase2: kind={analysis.section_kind} "
            f"emits_schema={analysis.emits_schema_properties} "
            f"base_ref={analysis.base_ref!r} inherited={analysis.inherited_fields} "
            f"| memory_updates: { {t: v['fields'] for t, v in new_refs.items()} }"
        )
        return analysis, new_refs
    except Exception as e:
        logger.warning(f"Section analyzer failed for [{section_id}] ({e}); extracting without it.")
        return None, {}


def _scope_section(
    section_id: str,
    section_title: str,
    section_content: str,
    section_verdict: str,
    existing_rules: str,
) -> str:
    """Phase 3 — the precise per-rule SKIP list for this section.

    Given the section, its verdict, and the rules already created elsewhere, decide
    which of this section's rules exactly duplicate an existing one (skip) and leave
    everything else for the Extractor to emit. Returns the skip list rendered for the
    Extractor prompt, or "None — emit everything this section defines." when nothing
    overlaps or the call fails (so the Extractor never wrongly suppresses itself).
    """
    if not existing_rules or existing_rules.strip() in ("", "None yet."):
        return "None — emit everything this section defines."
    try:
        chain = section_scope_prompt | llm.with_structured_output(SectionScope)
        scope: SectionScope = chain.invoke({
            "section_id":      section_id,
            "section_title":   section_title,
            "section_content": section_content,
            "section_verdict": section_verdict,
            "existing_rules":  existing_rules,
        })
        logger.info(f"[scope {section_id}] skip: {scope.skip}")
        if not scope.skip:
            return "None — emit everything this section defines."
        return "\n".join(f"  - {s}" for s in scope.skip)
    except Exception as e:
        logger.warning(f"Section scope failed for [{section_id}] ({e}); emitting without a skip list.")
        return "None — emit everything this section defines."


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
) -> str:
    """
    Returns a correction instruction for retry attempts on a section.

    Lists, in order: the rules already validated and the ones the Validator
    discarded (neither must be produced again), the rules that failed and how to
    fix them, and the missing rules to add. Without those first two lists the LLM
    cannot honour "do not duplicate", so it re-extracts the whole section on every
    retry — including rules just thrown away.
    Returns "" on the first attempt (no errors and no missing rules).
    """
    if not validation_errors and not section_reflection:
        return ""

    lines = [
        "CORRECTION TASK — this section has ALREADY been extracted.",
        "Return ONLY the rules listed under 'RULES TO CORRECT' and 'RULES TO EXTRACT",
        "FROM SCRATCH' below. Do NOT re-extract the section from the beginning.",
        "You may produce more or fewer rules than before — quality over quantity.",
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
            "DISCARDED — the Validator rejected these (duplicates of other sections, "
            "or not mappable to OpenAPI). Do NOT extract them again:"
        )
        for err in discarded_errors:
            rule = err.get("rule", {})
            lines.append(
                f"  [{rule.get('rule_type', '?')}] {rule.get('source_name', '?')}: "
                f"{err.get('instruction', 'discarded')}"
            )
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
        f"EXPECTED OUTPUT: {n_fix} corrected rule(s) and {n_missing} new rule(s). "
        "Anything already listed as validated above must not appear in your output.",
    ]
    return "\n".join(lines)


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

    if validation_errors or section_reflection:
        # Rules the Validator already accepted for this section — the Extractor must
        # see them to honour "do not duplicate"; without them it re-extracts the
        # whole section on every retry.
        already_validated = (
            state.get("validated_rules_by_section", {}) or {}
        ).get(section_id, [])
        correction_task = _build_correction_task(
            validation_errors, section_reflection, already_validated, discarded_errors,
        )
        logger.info(
            f"Retry {section_iteration_count + 1} for [{section_id}] — "
            f"{len(validation_errors)} error(s), "
            f"{sum(len(sr.get('missing_rules', [])) for sr in section_reflection)} missing rule(s)."
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
    sibling_summary = _build_sibling_summary(
        section_title, state["parsed_sections"], state.get("section_analysis_by_id", {}),
        exclude_id=section_id,
    )
    analysis, new_refs = _analyze_section(
        section_id, section_title, section["content"],
        state.get("other_spec_ref_memory", {}),
        sibling_summary,
    )
    section_analysis = _format_analysis_guidance(analysis)

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

    # Phase 3 — scope: from the verdict + the already-created rules, decide the
    # precise per-rule SKIP list (rules another section already owns), so the
    # Extractor drops exactly those and still emits everything else this section
    # defines (e.g. a POST section's responses even if its path/request_body exist).
    skip_rules = _scope_section(
        section_id, section_title, section["content"],
        _format_analysis_guidance(analysis), existing_rules,
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
        "stage2_context":             stage2_context,
        "stage3_context":             stage3_context,
        "skip_rules":                 skip_rules,
        "helper_context":             helper_context,
        "contract_summary":           contract_summary,
        "correction_task":            correction_task,
    })

    raw_rules = [rule.model_dump() for rule in result.rules]
    logger.info(
        f"Extractor Node complete — {len(raw_rules)} raw rule(s) from [{section_id}] "
        f"(section attempt {section_iteration_count + 1})."
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
        }}

    return {
        "raw_rules":               raw_rules,
        "section_iteration_count": section_iteration_count + 1,
        # Merge this section's newly declared other-spec references into the running
        # memory (see _merge_ref_memory), so later sections can resolve them.
        "other_spec_ref_memory":   new_refs,
        "section_analysis_by_id":  verdict_update,
    }
