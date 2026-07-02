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

logger = get_logger(__name__)


def _build_sections_summary(sections: list[dict]) -> str:
    """
    Formats parsed_sections into a compact table for the locator prompt.
    Each row: section_id | title | first N chars of content.
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

    if located_ids:
        selected_sections = [s for s in parsed_sections if s["section_id"] in located_ids]
    else:
        logger.warning(
            "ContractBuilder: locator returned no section IDs — "
            f"falling back to first {CONTRACT_FALLBACK_SECTIONS} sections."
        )
        selected_sections = parsed_sections[:CONTRACT_FALLBACK_SECTIONS]

    relevant_content = "\n\n".join(
        f"### {s['title']}\n{s['content']}" for s in selected_sections
    )

    # --- LLM call 2: extract the contract from full section content ---
    contract_chain = contract_extract_prompt | llm.with_structured_output(ServiceContract)
    contract: ServiceContract = contract_chain.invoke({
        "service_name":     service_name,
        "relevant_content": relevant_content,
    })

    logger.info(
        f"ContractBuilder complete: {len(contract.capabilities)} capabilities, "
        f"{len(contract.inclusion_keywords)} inclusion keyword(s), "
        f"{len(contract.exclusion_keywords)} exclusion keyword(s)."
    )

    return {"service_contract": contract.model_dump()}
