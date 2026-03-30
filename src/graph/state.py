# src/graph/state.py
# Defines the shared state that flows through the entire LangGraph pipeline.
#
# HOW IT WORKS:
#   The state is a single dictionary-like object created once at the start of
#   the pipeline and passed through every node. Each node receives the full
#   state, reads the fields it needs, and returns only the fields it updates.
#   LangGraph merges those updates back into the shared state before calling
#   the next node.
#
#   rule_bank_flow.py  → defines HOW nodes connect (graph structure)
#   state.py           → defines WHAT travels between them (shared memory)
#
# FIELD OWNERSHIP (which node writes each field):
#   Reader    → parsed_sections, helper_context, openapi_reference_context
#   Planner   → extraction_plan
#   Extractor → raw_rules
#   Reflector → reflected_rules
#   Validator → validated_rules, validation_errors
#   Builder   → final_output_path
#   (flow control fields are managed by the graph itself)

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class RuleBankState(TypedDict):

    # -------------------------------------------------------------------------
    # INPUT — set by main.py before the graph starts running
    # -------------------------------------------------------------------------

    # Absolute path to the main 3GPP specification file (e.g., TS 28.312)
    main_doc_path: str

    # Absolute paths to auxiliary 3GPP specification files (e.g., TS 32.160).
    # Empty list if no auxiliary documents are provided.
    auxiliary_doc_paths: list[str]

    # -------------------------------------------------------------------------
    # READER OUTPUT — produced by the Reader Node
    # -------------------------------------------------------------------------

    # List of relevant sections extracted from the main 3GPP document.
    # Each entry is a dict with keys: "section_id", "title", "content".
    # Only sections that match OpenAPI-relevant keywords are kept.
    parsed_sections: list[dict]

    # Concatenated summary of auxiliary 3GPP documents.
    # Used as background context by the Planner and Extractor nodes.
    helper_context: str

    # Content retrieved from the local snapshot of the OpenAPI/Swagger
    # specification (data/references/openapi_reference/).
    # Used by the Reflector node for RAG-grounded self-reflection.
    openapi_reference_context: str

    # -------------------------------------------------------------------------
    # PLANNER OUTPUT — produced by the Planner Node
    # -------------------------------------------------------------------------

    # Strategy produced by the LLM after reasoning about the document structure.
    # Contains: section priority list, extraction granularity, focus areas.
    # Example: {"priority_sections": ["6.2", "7.1"], "granularity": "attribute"}
    extraction_plan: dict

    # -------------------------------------------------------------------------
    # EXTRACTOR OUTPUT — produced by the Extractor Node
    # -------------------------------------------------------------------------

    # Raw rules extracted from parsed_sections, guided by extraction_plan.
    # Each entry is a dict with keys: "section_id", "rule_text", "openapi_mapping".
    # Not yet validated — may contain errors or incomplete mappings.
    raw_rules: list[dict]

    # -------------------------------------------------------------------------
    # REFLECTOR OUTPUT — produced by the Reflector Node
    # -------------------------------------------------------------------------

    # Rules after Chain-of-Thought self-reflection.
    # Each entry extends raw_rules with: "confidence" (float), "reasoning" (str),
    # "flagged" (bool — True means priority validation is needed).
    reflected_rules: list[dict]

    # -------------------------------------------------------------------------
    # VALIDATOR OUTPUT — produced by the Validator Node
    # -------------------------------------------------------------------------

    # Rules that passed both structural (Pydantic) and semantic (LLM) validation.
    validated_rules: list[dict]

    # Rules that failed validation, with failure reasons attached.
    # Used to compute the error rate and decide whether to loop back to Extractor.
    # Each entry is a dict with keys: "rule" (original rule dict), "reason" (str).
    validation_errors: list[dict]

    # -------------------------------------------------------------------------
    # BUILDER OUTPUT — produced by the Builder Node
    # -------------------------------------------------------------------------

    # Absolute path to the final rules bank JSON file saved in data/outputs/rules_bank/.
    final_output_path: str

    # -------------------------------------------------------------------------
    # FLOW CONTROL — managed by the graph (conditions.py) and main.py
    # -------------------------------------------------------------------------

    # Counts how many times the Extractor → Reflector → Validator loop has run.
    # Prevents infinite loops if the error rate never drops below the threshold.
    iteration_count: int

    # Maximum number of allowed loop iterations before forcing forward to Builder.
    max_iterations: int

    # -------------------------------------------------------------------------
    # TRACEABILITY — reasoning and message history
    # -------------------------------------------------------------------------

    # Full message history from all LLM calls across all nodes.
    # The add_messages annotation tells LangGraph to append new messages
    # instead of overwriting the list on each node update.
    messages: Annotated[list, add_messages]
