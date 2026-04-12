# src/main.py

"""
Entry point for the OpenAPI Rules Bank extraction pipeline.

Runs the full LangGraph flow end-to-end:
    Reader → Planner → Extractor → Reflector → Validator → Builder

The extracted rules are saved as a JSON file to data/outputs/rules_bank/.

Usage (from src/):
    python main.py --doc /path/to/spec.md
    python main.py --doc /path/to/spec.md --aux /path/to/aux1.md /path/to/aux2.md

Arguments:
    --doc  PATH    Path to the main 3GPP specification file (.md). Required.
    --aux  PATH…   Paths to auxiliary 3GPP documents (optional, space-separated).
"""

import argparse
import os
import sys

from config import get_logger
from graph.rule_bank_flow import get_compiled_graph

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    # Defines --doc (required) and --aux (optional, multiple) CLI arguments
    parser = argparse.ArgumentParser(
        description="Run the OpenAPI Rules Bank extraction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --doc ../data/inputs/3gpp/28532-i00.md\n"
            "  python main.py --doc spec.md --aux aux1.md aux2.md\n"
        ),
    )
    parser.add_argument(
        "--doc",
        required=True,
        metavar="PATH",
        help="Path to the main 3GPP specification file (.md).",
    )
    parser.add_argument(
        "--aux",
        nargs="*",
        default=[],
        metavar="PATH",
        help="Paths to auxiliary 3GPP specification files (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve all paths to absolute so downstream nodes work regardless of cwd
    main_doc_path = os.path.abspath(args.doc)
    aux_doc_paths = [os.path.abspath(p) for p in args.aux]

    # Fail early if any file is missing — avoids wasting LLM calls
    if not os.path.isfile(main_doc_path):
        logger.error(f"Document not found: {main_doc_path}")
        sys.exit(1)

    for p in aux_doc_paths:
        if not os.path.isfile(p):
            logger.error(f"Auxiliary document not found: {p}")
            sys.exit(1)

    logger.info("Starting Rules Bank pipeline.")
    logger.info(f"  Document : {main_doc_path}")
    for p in aux_doc_paths:
        logger.info(f"  Auxiliary: {p}")

    # Initial state passed into the LangGraph pipeline.
    # Fields written by nodes (parsed_sections, extraction_plan, etc.) are
    # populated automatically as the graph runs — only inputs are set here.
    initial_state = {
        "main_doc_path":       main_doc_path,
        "auxiliary_doc_paths": aux_doc_paths,
        "validated_rules":     [],   # accumulated via operator.add across iterations
        "validation_errors":   [],   # reset each iteration by the Validator
        "iteration_count":     0,    # incremented by the Extractor each loop
        "messages":            [],   # LLM message history across all nodes
    }

    # Compile and run the full Reader→Planner→Extractor→Reflector→Validator→Builder graph
    pipeline = get_compiled_graph()

    try:
        result = pipeline.invoke(initial_state)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    # Builder writes final_output_path into the state when it saves the JSON file
    output_path = result.get("final_output_path", "")
    if output_path:
        logger.info("Pipeline complete.")
        logger.info(f"  Rules bank saved to: {output_path}")
    else:
        logger.error("Pipeline finished but no output file was produced.")
        sys.exit(1)


if __name__ == "__main__":
    main()
