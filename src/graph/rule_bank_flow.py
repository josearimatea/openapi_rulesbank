# src/graph/rule_bank_flow.py

"""
Constructs and compiles the LangGraph StateGraph for the Rule Bank pipeline.

FLOW DIAGRAM:

  [Reader] → [Planner] → [Extractor] → [Reflector] → [Validator] → [Builder]
                                                           ↑             |
                                            error > threshold            |
                                            └──── loop back ─────────────┘

NODE DESCRIPTIONS:

  Reader   : Loads the main 3GPP document and auxiliary references.
             Filters sections relevant to OpenAPI (templates, NRM, mappings).
             State output: parsed_sections, helper_context, openapi_reference_context.

  Planner  : Reasons about the filtered sections using LLM.
             Defines extraction strategy: section priority, granularity, focus areas.
             State output: extraction_plan.

  Extractor: Iterates over parsed_sections guided by extraction_plan.
             Uses LLM to extract raw OpenAPI rules section by section.
             State output: raw_rules.

  Reflector: Applies Chain-of-Thought self-reflection over raw_rules.
             Retrieves relevant context from Qdrant (swagger.io + 3GPP) via RAG.
             Flags uncertain rules for priority validation.
             State output: reflected_rules (with confidence scores and reasoning).

  Validator: Validates reflected_rules in two steps:
             1. Structural — Pydantic schema check.
             2. Semantic   — LLM alignment check against helper context.
             State output: validated_rules, validation_errors.
             Conditional: if error rate > threshold and iterations < max → loop to Extractor.

  Builder  : Assembles the final rules bank JSON from validated_rules.
             Adds metadata (source document, generation date, model used, statistics).
             Saves result to data/outputs/rules_bank/.
             State output: final_output_path.
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