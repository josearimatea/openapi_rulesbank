# src/graph/rule_bank_flow.py

"""
Constructs and compiles the LangGraph StateGraph for the Rule Bank pipeline.

INPUT PRE-PROCESSING (outside the graph, run once):

  data/inputs/3gpp/*.docx ──[scripts/convert_docx_inputs.py]──▶ data/inputs/3gpp/md/*.md

  The 3GPP sources ship as .docx. A separate pre-processing step converts them
  to Markdown with pandoc (via the bundled pypandoc-binary), mirroring the
  dataset's process-3GPP.ipynb. The graph below consumes the generated .md only.

FLOW DIAGRAM:

  [Reader] → [ContractBuilder] → [Planner] → [Extractor] → [Reflector] → [Validator] ──→ [Builder] → END
                                                  ↑                            │               │
                                                  │  error rate > threshold    │    next section available
                                                  │  AND section retries < max │               │
                                                  └──────── retry section ─────┘               │
                                                  └──────────────────── next section ──────────┘

═══════════════════════════════════════════════════════════════════════════════════
NODE DESCRIPTIONS
═══════════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ READER                                                                      │
  │ LLM: no  │  RAG: no  │  Deterministic                                      │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Loads all source documents and prepares them for downstream nodes.          │
  │                                                                             │
  │ Steps:                                                                      │
  │   1. Load main 3GPP document (.md, pre-converted from .docx).               │
  │   2. Split into sections on '## ' headers via utils.parsers.parse_sections. │
  │      Setext headings (----/====) are normalized to ATX first, so .md from   │
  │      any pandoc version parses. Optional symbolic-title/keyword filters.    │
  │   3. Load auxiliary 3GPP documents (truncated to 2000 chars each).          │
  │   4. If legacy_yaml_path is set, append that previous-release YAML as extra   │
  │      context (e.g. a Rel-17 YAML when targeting Rel-18).                      │
  │   5. Read local OpenAPI spec snapshot from data/references/openapi_reference│
  │                                                                             │
  │ The document's Annex A (the embedded target YAML) is deliberately NOT used   │
  │ as context — the rules are built from the spec sections only.               │
  │                                                                             │
  │ Input  (state):  main_doc_path, auxiliary_doc_paths, target_yaml,            │
  │                  legacy_yaml_path                                            │
  │ Output (state):  parsed_sections   — list[{section_id, title, content}]     │
  │                  helper_context    — aux docs (+ optional legacy YAML),       │
  │                                      joined                                   │
  │                  openapi_reference_context — full OpenAPI spec text         │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ CONTRACT BUILDER                                                            │
  │ LLM: yes (2 calls)  │  RAG: no  │  Runs once per pipeline execution         │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Builds the semantic scope contract for the target Management Service.       │
  │ Defines what the target OpenAPI YAML covers, so the Planner keeps only in-  │
  │ scope sections and the Extractor extracts only rules that belong to it.     │
  │                                                                             │
  │ Steps:                                                                      │
  │   1. Build summary table (section_id | title | first N chars).             │
  │   2. LLM call 1 (locator): identifies which section IDs contain purpose,   │
  │      capabilities, and key concepts. Returns SectionLocator.               │
  │      Fallback: if no IDs found, uses first CONTRACT_FALLBACK_SECTIONS.      │
  │   3. Fetches full content of located sections only.                        │
  │   4. LLM call 2 (extractor): builds ServiceContract from full content —    │
  │      purpose, capabilities (with type from spec text), key_concepts,       │
  │      inclusion_keywords, exclusion_keywords, out_of_scope_signals.         │
  │                                                                             │
  │ Input  (state):  parsed_sections, target_yaml                              │
  │ Output (state):  service_contract — ServiceContract as dict                │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ PLANNER                                                                     │
  │ LLM: yes (1 call)  │  RAG: no                                               │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Reasons over section titles and content previews to define what to extract. │
  │                                                                             │
  │ Steps:                                                                      │
  │   1. Build compact summary table (section_id | title | first 200 chars).   │
  │   2. Send table + first 3000 chars of OpenAPI overview to LLM.              │
  │   3. LLM returns ExtractionPlan with priority and extraction_focus per      │
  │      section (high/medium/low). Uses with_structured_output(ExtractionPlan).│
  │                                                                             │
  │ Input  (state):  parsed_sections, helper_context, openapi_reference_context,│
  │                  service_contract                                           │
  │ Output (state):  extraction_plan — {document_summary,                       │
  │                    sections_to_extract: [{section_id, title,                │
  │                                          priority, extraction_focus}]}      │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ EXTRACTOR                                                                   │
  │ LLM: yes (1 call/section)  │  RAG: yes (1 query/section, Qdrant)            │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Extracts raw OpenAPI rules from each planned section.                       │
  │                                                                             │
  │ Steps (per section):                                                        │
  │   1. RAG query: "{section_title} — {extraction_focus}" → 5 OpenAPI chunks. │
  │   2. LLM extracts RawRule objects from section content ONLY. OpenAPI chunks │
  │      are reference context, not source. Enforces one rule per HTTP method.  │
  │   3. Each rule has: section_id, section_title, rule_type, rule_text,        │
  │      openapi_mapping {openapi_object, openapi_field, openapi_value}.        │
  │                                                                             │
  │ rule_type values: path_operation | schema_property | path_parameter |       │
  │                   query_parameter | response | request_body | security_scheme│
  │                                                                             │
  │ On loop-back: processes ONLY sections with failing rules. Sends CORRECTION  │
  │ TASK prompt listing the exact failed rules and reasons, instructing the LLM │
  │ to return one corrected rule per entry without adding new rules.            │
  │                                                                             │
  │ Input  (state):  parsed_sections, extraction_plan, helper_context,          │
  │                  validation_errors, section_reflection, current_section_idx │
  │ Output (state):  raw_rules — list[RawRule dicts]                            │
  │                  section_iteration_count — incremented by 1                 │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ REFLECTOR                                                                   │
  │ LLM: yes (1 call/rule)  │  RAG: yes (1 query/rule, Qdrant)                  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Applies CoT self-reflection to each raw rule before validation.             │
  │                                                                             │
  │ Steps (per rule):                                                           │
  │   1. RAG query: rule_text + openapi_object + openapi_field → 5 chunks.     │
  │   2. LLM answers 4 CoT questions: Is the rule grounded? Is the mapping     │
  │      correct? Does the OpenAPI context confirm it? Confidence score?        │
  │   3. Returns ReflectionResult: confidence (0.0–1.0), reasoning, flagged.   │
  │   4. Merges into rule dict → ReflectedRule (adds confidence, reasoning,    │
  │      flagged, rag_context).                                                 │
  │                                                                             │
  │ Input  (state):  raw_rules                                                  │
  │ Output (state):  reflected_rules — list[ReflectedRule dicts]                │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ VALIDATOR                                                                   │
  │ LLM: yes (1 call/rule for Stage 2)  │  RAG: no                             │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Validates each reflected rule through three stages:                         │
  │                                                                             │
  │ Stage 1a — Pydantic structural (no LLM):                                   │
  │   ValidatedRule(**rule) — checks all required fields and types.             │
  │                                                                             │
  │ Stage 1b — Mapping consistency / utils.rules_check (no LLM):               │
  │   check_mapping_for_type(rule_type, openapi_mapping) — verifies that        │
  │   openapi_field and openapi_value match the constraints for rule_type.      │
  │   E.g. path_operation: field must be a single lowercase HTTP method.        │
  │   Runs only if Stage 1a passes.                                             │
  │                                                                             │
  │ Stage 2 — LLM semantic:                                                     │
  │   Checks rule_text is grounded in section content and openapi_mapping is   │
  │   accurate. Flagged rules receive extra scrutiny. Prompt includes rule_type │
  │   with type-specific validation rules (e.g. response field = HTTP code).   │
  │   Runs only if Stages 1a and 1b pass.                                       │
  │                                                                             │
  │ Rules passing all stages → validated_rules (accumulated via operator.add). │
  │ Rules failing any stage → validation_errors (with reason and stage).       │
  │                                                                             │
  │ Input  (state):  reflected_rules, parsed_sections                           │
  │ Output (state):  validated_rules   — list[ValidatedRule dicts] (accumulated)│
  │                  validation_errors — list[ValidationError dicts]            │
  │                                                                             │
  │ Conditional routing (conditions.should_retry_section_or_build):            │
  │   error_rate = len(validation_errors) / len(reflected_rules)               │
  │   if error_rate > VALIDATION_ERROR_THRESHOLD (10%)                          │
  │      AND section_iteration_count < MAX_ITERATIONS (3) → retry section       │
  │   otherwise → proceed to Builder                                            │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ BUILDER                                                                     │
  │ LLM: no  │  RAG: no  │  Deterministic                                      │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Assembles and persists the final rules bank JSON file.                      │
  │                                                                             │
  │ Steps:                                                                      │
  │   1. Read validated_rules (all iterations accumulated via operator.add).   │
  │   2. Compute summary: rules_by_type and rules_by_section (Counter).        │
  │   3. Validate complete output against schemas.output.RulesBank (Pydantic). │
  │   4. Save to data/outputs/rules_bank/rules_bank_<doc>_<timestamp>.json.    │
  │                                                                             │
  │ Output JSON:  metadata (provenance, model, counts, iterations)              │
  │               summary  (rules_by_type, rules_by_section, error counts)     │
  │               rules    (list of full ValidatedRule dicts)                   │
  │                                                                             │
  │ Input  (state):  validated_rules, validation_errors, main_doc_path,         │
  │                  section_iteration_count, current_section_idx,              │
  │                  extraction_plan, parsed_sections                            │
  │ Output (state):  final_output_path — absolute path to the saved JSON file  │
  └─────────────────────────────────────────────────────────────────────────────┘
"""

from langgraph.graph import StateGraph, END
from openapi_rulesbank.config import get_logger
from openapi_rulesbank.graph.state import RuleBankState
from openapi_rulesbank.graph.conditions import should_retry_section_or_build, should_next_section_or_end
from openapi_rulesbank.nodes.reader import reader_node
from openapi_rulesbank.nodes.contract_builder import contract_builder_node
from openapi_rulesbank.nodes.planner import planner_node
from openapi_rulesbank.nodes.extractor import extractor_node
from openapi_rulesbank.nodes.reflector import reflector_node
from openapi_rulesbank.nodes.validator import validator_node
from openapi_rulesbank.nodes.builder import builder_node

logger = get_logger(__name__)


def _build_graph(entry_point: str, checkpointer=None):
    """
    Shared graph builder used by both get_compiled_graph variants.

    checkpointer: optional LangGraph checkpointer (e.g. an AsyncSqliteSaver) for
    persistence/resumability when the graph is driven by an external host such as
    the chatbot UI. None (default) compiles without persistence — the behaviour
    used by the CLI.
    """
    graph = StateGraph(RuleBankState)

    # Add nodes
    graph.add_node("reader",           reader_node)
    graph.add_node("contract_builder", contract_builder_node)
    graph.add_node("planner",          planner_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("validator", validator_node)
    graph.add_node("builder",   builder_node)

    # Fixed edges
    graph.add_edge("reader",           "contract_builder")
    graph.add_edge("contract_builder", "planner")
    graph.add_edge("planner",          "extractor")
    graph.add_edge("extractor", "reflector")
    graph.add_edge("reflector", "validator")

    # After Validator: retry the current section or proceed to Builder
    graph.add_conditional_edges(
        "validator",
        should_retry_section_or_build,
        {"extractor": "extractor", "builder": "builder"},
    )

    # After Builder: process next section or end the pipeline
    graph.add_conditional_edges(
        "builder",
        should_next_section_or_end,
        {"extractor": "extractor", "end": END},
    )

    # Entry point
    graph.set_entry_point(entry_point)

    # compile(checkpointer=None) is equivalent to no persistence, so the CLI
    # path is unaffected.
    return graph.compile(checkpointer=checkpointer)


def get_compiled_graph(checkpointer=None):
    """Full pipeline starting at the Reader. See _build_graph for checkpointer."""
    return _build_graph("reader", checkpointer=checkpointer)


def get_compiled_graph_from_planner(checkpointer=None):
    """
    Pipeline starting at the Planner — Reader is skipped.

    Use when parsed_sections is already populated (e.g. scoped tests that
    pre-filter sections before invoking the graph).
    """
    return _build_graph("planner", checkpointer=checkpointer)


def build_initial_state(
    main_doc_path: str,
    target_yaml: str,
    auxiliary_doc_paths: list[str] | None = None,
    legacy_yaml_path: str = "",
) -> dict:
    """
    Builds the initial RuleBankState with all flow-control defaults set.

    Centralizes the defaults that the pipeline expects before the graph runs
    (accumulators, counters, message history) so every caller — the CLI and any
    external host such as the chatbot UI — produces an identical starting state.

    Args:
        main_doc_path:       absolute path to the main 3GPP spec (.md).
        target_yaml:         OpenAPI document to build rules for (e.g. "ProvMnS").
        auxiliary_doc_paths: optional auxiliary 3GPP .md paths.
        legacy_yaml_path:    optional previous-release YAML path ("" if none).
    """
    return {
        "main_doc_path":                main_doc_path,
        "auxiliary_doc_paths":          auxiliary_doc_paths or [],
        "target_yaml":                  target_yaml,
        "legacy_yaml_path":             legacy_yaml_path,
        "service_contract":             {},  # populated by ContractBuilder before Planner runs
        "validated_rules":              [],  # accumulated via operator.add across sections
        "validation_errors":            [],  # cleared by Builder after each section
        "section_reflection":           [],  # Reflector Fase 2 output; reset by Builder
        "validator_section_reflection": [],  # Validator final missing-rules; reset by Builder
        "current_section_idx":          0,   # advanced by Builder after each section
        "section_iteration_count":      0,   # reset by Builder after each section
        "other_spec_ref_memory":        {},  # Extractor pre-analyzer: type->spec refs, merged across sections
        "section_analysis_by_id":       {},  # Extractor pre-analyzer: per-section verdict, read by Reflector/Validator
        "messages":                     [],  # LLM message history across all nodes
    }