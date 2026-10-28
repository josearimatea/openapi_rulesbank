# src/nodes/contract_builder.py

"""
ContractBuilder Node: builds the semantic scope contract for the target service.

Runs once per pipeline run, between Reader and Planner.
No RAG. Two LLM calls — one to locate relevant sections, one to extract the contract.

HOW IT WORKS:
    1. Builds a summary table (section_id | title | first N chars) from
       parsed_sections. Lightweight — no full content is sent in this call.
       N = settings.CONTRACT_LOCATOR_PREVIEW_CHARS.
    2. LLM call 1 (locator): reads the summary table and returns which section IDs
       hold the service's purpose, capabilities, and key concepts (SectionLocator).
       Empty lists are valid.
    3. Fetches the full content of the located sections only.
       Fallback: if the locator returned no IDs at all, uses the first
       settings.CONTRACT_FALLBACK_SECTIONS sections (in 3GPP specs these are
       usually the overview and definitions).
    4. LLM call 2 (extractor): reads the full content of the located sections and
       builds the ServiceContract — purpose, capabilities (with type), key_concepts,
       inclusion_keywords, exclusion_keywords, out_of_scope_signals.

State reads:
    parsed_sections  (list[dict]) — sections from Reader (section_id, title, content)
    target_yaml      (str)        — name of the target service (e.g. "ProvMnS")

State writes:
    service_contract (dict)       — ServiceContract as a dict, injected into the
                                    Planner and Extractor. Empty contract (name only)
                                    when there are no sections to read.
"""

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.llm_config import llm
from openapi_rulesbank.config.settings import (
    CONTRACT_FALLBACK_SECTIONS,
    CONTRACT_LOCATOR_PREVIEW_CHARS,
    TARGET_YAML,
)
from openapi_rulesbank.graph.state import RuleBankState
from openapi_rulesbank.prompts.contract_builder_prompts import (
    SectionLocator,
    contract_locator_prompt,
    contract_extract_prompt,
)
from openapi_rulesbank.schemas.contract import ServiceContract
from openapi_rulesbank.tools.document_tools import load_markdown
from openapi_rulesbank.tools.rag_tools import (
    create_rules_collection,
    index_target_spec_sections,
    search_3gpp_corpus,
)

logger = get_logger(__name__)


def _build_sections_summary(sections: list[dict]) -> str:
    """
    Formats parsed_sections into a compact table for the locator prompt.
    Each row: section_id | title | first N chars of content.

    section_id (the first column, e.g. "5") is the canonical unique key the
    locator should return and that contract_builder_node matches against; the
    title is shown alongside for context. section_id is used as the key because
    titles are not guaranteed unique across specs (e.g. repeated "Description").
    """
    rows = []
    for s in sections:
        preview = s["content"][:CONTRACT_LOCATOR_PREVIEW_CHARS].replace("\n", " ")
        rows.append(f"{s['section_id']} | {s['title']} | {preview}")
    return "\n".join(rows)


def contract_builder_node(state: RuleBankState) -> dict:
    """Builds the ServiceContract from spec text. Two LLM calls, no RAG."""
    logger.info("ContractBuilder Node started.")

    parsed_sections = state.get("parsed_sections", []) or []
    service_name = state.get("target_yaml") or TARGET_YAML

    if not parsed_sections:
        logger.warning(
            "ContractBuilder: no parsed_sections available — returning empty contract."
        )
        return {
            "service_contract": ServiceContract(service_name=service_name).model_dump()
        }

    # --- LLM call 1: locate relevant sections via lightweight summaries ---
    locator_chain = contract_locator_prompt | llm.with_structured_output(SectionLocator)
    locator: SectionLocator = locator_chain.invoke({
        "service_name":      service_name,
        "section_summaries": _build_sections_summary(parsed_sections),
    })

    logger.info(
        f"ContractBuilder locator: purpose={locator.purpose_section_ids}, "
        f"capabilities={locator.capabilities_section_ids}, "
        f"concepts={locator.concepts_section_ids}"
    )

    # --- Fetch full content of located sections ---
    located_ids = set(
        locator.purpose_section_ids
        + locator.capabilities_section_ids
        + locator.concepts_section_ids
    )

    # section_id is the canonical key (unique per document). The locator is asked
    # to return it, but sometimes returns the title instead, so we also accept an
    # exact title match as a safety net. Both are exact dict lookups — no string
    # parsing. by_id takes precedence; by_title only fills unresolved values.
    by_id    = {s["section_id"]: s for s in parsed_sections}
    by_title = {s["title"]: s for s in parsed_sections}

    def _resolve(refs) -> list[dict]:
        """Maps locator refs (ids, or titles as a fallback) to sections, deduped."""
        out, seen_ids = [], set()
        for ref in refs:
            section = by_id.get(ref) or by_title.get(ref)
            if section is not None and section["section_id"] not in seen_ids:
                out.append(section)
                seen_ids.add(section["section_id"])
        return out

    selected_sections = _resolve(located_ids)

    if not selected_sections:
        logger.warning(
            "ContractBuilder: locator returned no matching section IDs "
            f"(got {sorted(located_ids)[:5]}...) — falling back to first "
            f"{CONTRACT_FALLBACK_SECTIONS} sections."
        )
        selected_sections = parsed_sections[:CONTRACT_FALLBACK_SECTIONS]

    relevant_content = "\n\n".join(
        f"### {s['title']}\n{s['content']}" for s in selected_sections
    )

    # --- LLM call 2: extract the contract from full section content ---
    # Ground the contract in the wider 3GPP corpus: retrieve chunks about the
    # target service from the persistent corpus collection (referenced specs like
    # TS 28.533/32.160 that define the MnS). Empty when the corpus is not indexed.
    corpus_context = search_3gpp_corpus(
        f"{service_name} management service purpose capabilities operations notifications key concepts",
        exclude_spec=state.get("target_spec_number", "") or "",
    ) or "No external 3GPP corpus context available."

    contract_chain = contract_extract_prompt | llm.with_structured_output(ServiceContract)
    contract: ServiceContract = contract_chain.invoke({
        "service_name":     service_name,
        "relevant_content": relevant_content,
        "corpus_context":   corpus_context,
    })

    # Deterministic dedup: the prompt asks for unique entries, but the LLM can
    # still repeat values (observed: the same concept emitted hundreds of times).
    # Enforce uniqueness here while preserving first-seen order.
    def _dedup(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # Record the document's other Management Services (found by the locator) as
    # out of scope. This gives the Planner/Extractor explicit negative signals
    # without the extractor having to read those services' sections — the
    # extractor only saw the target's sections, so it cannot name them itself.
    other_services = [f"{name} (out of scope)" for name in locator.other_services]

    # Also flag the located target sections that fall OUTSIDE the stage 3 block as
    # out of scope. These are sections about the target service but not part of the
    # solution set, so no rules should come from them. Generic: the stage 3 clause
    # comes from the LLM; membership is a clause-number prefix test, no fixed
    # numbering. We do not label WHAT they are (could be stage 1 or 2) — only that
    # they are out of scope. To keep the contract small, collapse the sections to
    # a block as deep as the stage 3 clause itself (e.g. with a stage 3 clause of
    # three parts, "12.3.2.4" -> "12.3.2") and emit one signal per distinct block
    # instead of one per subsection. A fixed depth would name the stage 3 clause's
    # own parent — and so the stage 3 block — as out of scope whenever it has
    # sibling solution sets.
    out_of_scope_sections: list[str] = []
    if locator.stage3_clause:
        prefix = locator.stage3_clause + "."
        depth = len(locator.stage3_clause.split("."))
        blocks: list[str] = []
        for s in selected_sections:
            clause = s["title"].strip().split(" ", 1)[0]
            in_stage3 = clause == locator.stage3_clause or clause.startswith(prefix)
            if not in_stage3:
                blocks.append(".".join(clause.split(".")[:depth]))
        out_of_scope_sections = [
            f"{b}.x (out of scope)" for b in dict.fromkeys(blocks)  # unique, ordered
        ]

    contract.key_concepts        = _dedup(contract.key_concepts)
    contract.inclusion_keywords  = _dedup(contract.inclusion_keywords)
    contract.exclusion_keywords  = _dedup(contract.exclusion_keywords)
    contract.out_of_scope_signals = _dedup(
        contract.out_of_scope_signals + other_services + out_of_scope_sections
    )

    # Record where the OpenAPI rules must come from: the target's stage 3 block
    # (the RESTful HTTP solution set) identified by the locator. Stage 2 sections
    # (abstract definitions) are not a rule source. This guides the Planner to
    # extract only from stage 3.
    if locator.stage3_clause:
        contract.stage3_clause = locator.stage3_clause
        contract.stage_scope = (
            f"OpenAPI rules come ONLY from the stage 3 block (the RESTful HTTP "
            f"solution set) under clause {locator.stage3_clause}.x. Stage 2 "
            f"sections (abstract operation/parameter definitions, e.g. 'Input "
            f"parameters', 'Output parameters', 'Operations and notifications') "
            f"have no protocol binding and are NOT a rule source — exclude them."
        )

    logger.info(
        f"ContractBuilder complete: {len(contract.capabilities)} capabilities, "
        f"{len(contract.inclusion_keywords)} inclusion keyword(s), "
        f"{len(contract.exclusion_keywords)} exclusion keyword(s)."
    )

    # --- Spec RAG (temporary) ---
    # Index the document's sections into a per-run Qdrant collection, tagging each
    # with its MnS and stage (using the stage3_clause just identified) so the
    # Extractor can retrieve the target service's stage 2 definitions — where the
    # canonical IS operation names live. The Builder drops this collection when the
    # run ends. Non-fatal: the pipeline continues without spec RAG on failure.
    spec_collection_name = ""
    try:
        # The whole document, cut the way the corpus is (qdrant_3gpp_rag), without
        # its annex when settings.EXCLUDE_TARGET_ANNEX is on. A state without the
        # document path (a caller building it by hand) falls back to indexing the
        # parsed sections.
        doc_path = state.get("main_doc_path", "") or ""
        spec_collection_name = index_target_spec_sections(
            parsed_sections, locator.stage3_clause,
            doc_text=load_markdown(doc_path) if doc_path else "",
        )
    except Exception as e:
        logger.warning(f"Spec RAG indexing failed ({e}); continuing without spec RAG.")

    # Empty collection the Validator fills as rules pass, so it can detect a rule
    # duplicating one already accepted in another section. Also dropped by the Builder.
    rules_collection_name = ""
    try:
        rules_collection_name = create_rules_collection()
    except Exception as e:
        logger.warning(f"Rules RAG setup failed ({e}); continuing without rules RAG.")

    return {
        "service_contract":      contract.model_dump(),
        "spec_collection_name":  spec_collection_name,
        "rules_collection_name": rules_collection_name,
    }
