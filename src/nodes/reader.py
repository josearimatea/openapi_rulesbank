# src/nodes/reader.py

"""
Reader Node: loads all source documents and populates the shared state
with filtered sections ready for the Planner and Extractor nodes.

This node is purely deterministic — it does not call any LLM.
Its only job is to load, split, and filter text. All reasoning about
what to extract happens in the Planner Node.

State reads:
    main_doc_path        (str)       — path or URL to the main 3GPP spec
    auxiliary_doc_paths  (list[str]) — paths or URLs to auxiliary 3GPP specs

State writes:
    parsed_sections      (list[dict]) — relevant sections from the main doc,
                                        each with section_id, title, content
    helper_context       (str)        — auxiliary docs concatenated into one
                                        context string for downstream nodes
    openapi_spec_context (str)        — content of the local OpenAPI spec
                                        snapshot from data/references/openapi_spec/
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
    # Supports both local file paths and HTTP URLs.
    main_doc = state["main_doc_path"]
    raw_main = fetch_url(main_doc) if main_doc.startswith("http") else load_markdown(main_doc)

    # parse_sections() splits by ## headers and keeps only sections that
    # contain keywords relevant to OpenAPI (mapping, yang, template, etc.)
    parsed_sections = parse_sections(raw_main)
    logger.debug(f"Parsed {len(parsed_sections)} relevant sections from main doc.")

    # --- Auxiliary 3GPP documents ---
    # Each auxiliary doc is loaded and truncated to 2000 characters to avoid
    # flooding downstream nodes with too much context.
    # All docs are joined into a single string separated by a divider.
    helper_parts = []
    for path in state.get("auxiliary_doc_paths", []):
        content = fetch_url(path) if path.startswith("http") else load_markdown(path)
        helper_parts.append(content[:2000])
    helper_context = "\n\n---\n\n".join(helper_parts)
    logger.debug(f"Loaded {len(helper_parts)} auxiliary document(s).")

    # --- Local OpenAPI spec snapshot ---
    # load_openapi_spec_context() reads from OPENAPI_SPEC_DIR using discover_specs().
    # Returns "" with a warning if the directory is missing or empty — in that case
    # the pipeline continues without OpenAPI context (guard is inside the function).
    openapi_spec_context = load_openapi_reference()

    logger.info("Reader Node complete.")

    # Return only the fields this node is responsible for writing.
    # LangGraph merges these into the full shared state automatically.
    return {
        "parsed_sections": parsed_sections,
        "helper_context": helper_context,
        "openapi_spec_context": openapi_spec_context,
    }
