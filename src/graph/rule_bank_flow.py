# src/graph/rule_bank_flow.py

"""
Constructs and compiles the LangGraph StateGraph for the Rule Bank pipeline.

FLOW DIAGRAM:

  [Reader] → [Planner] → [Extractor] → [Reflector] → [Validator] ──→ [Builder] → END
                              ↑                            │
                              │       error rate > threshold
                              │       AND iterations < max
                              └──────────── loop back ─────┘

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
  │   1. Load main 3GPP document (local file or HTTP URL).                      │
  │   2. Split into sections on '## ' headers via utils.parsers.parse_sections. │
  │      Keep only sections with OpenAPI-relevant keywords (mapping, yang,      │
  │      template, attribute, openapi, nrm, etc.).                              │
  │   3. Load auxiliary 3GPP documents (truncated to 2000 chars each).          │
  │   4. Read local OpenAPI spec snapshot from data/references/openapi_reference│
  │                                                                             │
  │ Input  (state):  main_doc_path, auxiliary_doc_paths                         │
  │ Output (state):  parsed_sections   — list[{section_id, title, content}]     │
  │                  helper_context    — auxiliary docs joined as one string     │
  │                  openapi_reference_context — full OpenAPI spec text         │
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
  │ Input  (state):  parsed_sections, helper_context, openapi_reference_context │
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
  │                  validation_errors (empty on iter 1), iteration_count       │
  │ Output (state):  raw_rules — list[RawRule dicts]                            │
  │                  iteration_count — incremented by 1                         │
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
  │ Conditional routing (conditions.should_loop_or_build):                     │
  │   error_rate = len(validation_errors) / len(reflected_rules)               │
  │   if error_rate > VALIDATION_ERROR_THRESHOLD (10%)                          │
  │      AND iteration_count < MAX_ITERATIONS (3) → loop to Extractor           │
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
  │                  iteration_count, extraction_plan                            │
  │ Output (state):  final_output_path — absolute path to the saved JSON file  │
  └─────────────────────────────────────────────────────────────────────────────┘
"""

from langgraph.graph import StateGraph, END
from config import get_logger
from graph.state import RuleBankState
from graph.conditions import should_loop_or_build
from nodes.reader import reader_node
from nodes.planner import planner_node
from nodes.extractor import extractor_node
from nodes.reflector import reflector_node
from nodes.validator import validator_node
from nodes.builder import builder_node

logger = get_logger(__name__)


def get_compiled_graph():
    """Builds and returns the compiled LangGraph StateGraph."""
    graph = StateGraph(RuleBankState)

    # Add nodes
    graph.add_node("reader", reader_node)
    graph.add_node("planner", planner_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("validator", validator_node)
    graph.add_node("builder", builder_node)

    # Fixed edges
    graph.add_edge("reader", "planner")
    graph.add_edge("planner", "extractor")
    graph.add_edge("extractor", "reflector")
    graph.add_edge("reflector", "validator")
    graph.add_edge("builder", END)

    # Conditional edge: after Validator, loop back or proceed to Builder
    graph.add_conditional_edges(
        "validator",
        should_loop_or_build,
        {"extractor": "extractor", "builder": "builder"}
    )

    # Entry point
    graph.set_entry_point("reader")

    return graph.compile()