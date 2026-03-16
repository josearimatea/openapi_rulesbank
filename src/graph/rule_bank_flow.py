# src/flows/rule_bank_flow.py
# Defines the LangGraph flow for building the Rule Bank.
# Nodes: reader -> extractor -> validator (loop if fails) -> builder.

# Educational Flow Diagram (as comment):
# This section explains the flow step-by-step with arrows.
# It shows how data moves through the agents to build the Rule Bank JSON.
#
# Initial State: Empty dict {} (or with input_file/output_file from main.py)
#                ↓ (Entry Point)
# Reader Agent: Loads and parses DOC_TEST (main 3GPP doc, e.g., 28.312-i11.md) 
#               and DOC_HELPER (OpenAPI specs + TS 32.160).
#               - Filters relevant sections (templates, NRM, mappings to JSON/YANG/OpenAPI).
#               - Handles chunking to avoid token limits.
#               - Updates state: Adds 'parsed_main' (filtered text/summaries) Use LLM for optional summarization if prompt exists and debug mode is on.
#                 and 'helper_context' (helper summaries).
#                ↓ (Fixed Edge)
# Extractor Agent: Extracts raw rules from 'parsed_main' using LLM.
#                  - Processes in chunks: For each chunk, prompt LLM to find rules 
#                    (e.g., "Extract mappings for OpenAPI from {chunk} using {helper_context}").
#                  - Outputs list of dicts: [{'section': '6.2', 'rule': 'Attribute maps to YANG leaf', ...}].
#                  - Updates state: Adds 'extracted_rules'.
#                ↓ (Fixed Edge)
# Validator Agent: Validates 'extracted_rules' against helpers.
#                  - Structural: Uses Pydantic to check format.
#                  - Semantic: LLM checks alignment (e.g., "Validate {rule} vs. OpenAPI schemas").
#                  - Processes in batches to manage tokens.
#                  - Updates state: Adds 'validated_rules' and 'validation_errors' (error rate).
#                ↓ (Conditional Edge: Checks should_loop)
#                  - If validation_errors > 10% threshold: ← Loop back to Extractor 
#                    (with feedback in state, e.g., 'feedback': 'Refine YANG mappings').
#                  - Else: Proceed forward.
#                ↓ (Fixed Edge if no loop)
# Builder Agent: Builds final JSON from 'validated_rules'.
#                - Structures: Categorizes rules (e.g., by stage/category), adds metadata 
#                  (doc source, generation date).
#                - Saves to output_file as JSON (e.g., {'rules': [...], 'metadata': {...}}).
#                - Updates state: Adds 'final_output' (path to JSON).
#                ↓ (Finish Point)
# End: Flow complete, result in state.

from langgraph.graph import Graph, END
from config import get_logger
from agents.reader_agent import reader_agent
from agents.extractor_agent import extractor_agent
from agents.validator_agent import validator_agent
from agents.builder_agent import builder_agent

logger = get_logger(__name__)

def should_loop(state: dict) -> str:
    """Conditional function: Loop back to extractor if validation fails > threshold."""
    if state.get("validation_errors", 0) > 0.1 * len(state.get("extracted_rules", [])):  # Example: 10% error threshold
        logger.warning("Validation failed threshold - looping back to extractor.")
        return "extractor"
    return "builder"

def get_compiled_graph(llm):
    """Builds and returns the compiled LangGraph graph."""
    graph = Graph()

    # Add nodes (agents)
    graph.add_node("reader", lambda state: reader_agent(state, llm))
    graph.add_node("extractor", lambda state: extractor_agent(state, llm))
    graph.add_node("validator", lambda state: validator_agent(state, llm))
    graph.add_node("builder", lambda state: builder_agent(state, llm))

    # Add edges
    graph.add_edge("reader", "extractor")
    graph.add_conditional_edges("validator", should_loop, {"extractor": "extractor", "builder": "builder"})
    graph.add_edge("builder", END)

    # Set entry point
    graph.set_entry_point("reader")

    # Compile and return
    return graph.compile()