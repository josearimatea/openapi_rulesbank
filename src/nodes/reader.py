# src/nodes/reader.py

"""
Reader Node: loads all source documents and populates the shared state
with filtered sections ready for the Planner and Extractor nodes.

No LLM call. No RAG. Purely deterministic text loading and filtering.

HOW IT WORKS:
    1. Loads the main 3GPP document from a local file path.
    2. Splits the document into sections using utils.parsers.parse_sections(),
       which splits on '## ' headers and optionally filters sections by keywords
       (see FILTER_SYMBOLIC_TITLES and FILTER_BY_KEYWORDS in settings.py).
    3. Loads each auxiliary 3GPP document and truncates to 2000 chars each,
       joining them into a single helper_context string.
    4. Reads the local OpenAPI specification snapshot from
       data/references/openapi_reference/ via tools.document_tools.load_openapi_reference().

State reads:
    main_doc_path        (str)       — local file path to the main 3GPP spec
    auxiliary_doc_paths  (list[str]) — local file paths to auxiliary 3GPP specs
                                       (empty list if none provided)

State writes:
    parsed_sections           (list[dict]) — relevant sections from the main doc.
                                             Each dict: {section_id, title, content}.
                                             Only sections with OpenAPI keywords are kept.
    helper_context            (str)        — auxiliary docs concatenated into one string
                                             (2000 chars per doc, max). Used by Planner
                                             and Extractor for background context.
    openapi_reference_context (str)        — full text of the local OpenAPI spec snapshot.
                                             Used by Planner as overview reference.
                                             Empty string if data/references/openapi_reference/
                                             is missing or empty.
"""

from config import get_logger
from graph.state import RuleBankState
from tools.document_tools import load_markdown, load_openapi_reference
from utils.parsers import parse_sections

logger = get_logger(__name__)


def reader_node(state: RuleBankState) -> dict:
    """Loads, splits, and filters all source documents into the state."""
    logger.info("Reader Node started.")

    # --- Main 3GPP document ---
    # Reads from state["main_doc_path"], set by main.py before the graph runs.
    main_doc = state["main_doc_path"]
    raw_main = load_markdown(main_doc)

    # parse_sections() splits by ## headers and filters sections (see parsers.py).
    # Returns both the kept sections and those excluded by the reader-level filters.
    parsed_sections, excluded_sections_reader = parse_sections(raw_main)

    # Deterministic accounting: total = kept + excluded (verified here, carried to Builder)
    sections_total = len(parsed_sections) + len(excluded_sections_reader)
    assert sections_total > 0, "Document produced no sections after splitting."
    logger.info(
        f"Reader sections: {sections_total} total = "
        f"{len(parsed_sections)} for planner + "
        f"{len(excluded_sections_reader)} excluded by reader"
    )

    # --- Auxiliary 3GPP documents ---
    # Each auxiliary doc is loaded and truncated to 2000 characters to avoid
    # flooding downstream nodes with too much context.
    # All docs are joined into a single string separated by a divider.
    helper_parts = []
    for path in state.get("auxiliary_doc_paths", []):
        helper_parts.append(load_markdown(path)[:2000])
    helper_context = "\n\n---\n\n".join(helper_parts)
    logger.debug(f"Loaded {len(helper_parts)} auxiliary document(s).")

    # --- Local OpenAPI spec snapshot ---
    # load_openapi_reference_context() reads from OPENAPI_REFERENCE_DIR using discover_specs().
    # Returns "" with a warning if the directory is missing or empty — in that case
    # the pipeline continues without OpenAPI context (guard is inside the function).
    openapi_reference_context = load_openapi_reference()

    logger.info("Reader Node complete.")

    # Return only the fields this node is responsible for writing.
    # LangGraph merges these into the full shared state automatically.
    return {
        "parsed_sections":          parsed_sections,
        "excluded_sections_reader": excluded_sections_reader,
        "helper_context":           helper_context,
        "openapi_reference_context": openapi_reference_context,
    }
