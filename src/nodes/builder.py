# src/nodes/builder.py

"""
Builder Node: assembles the final rules bank JSON from validated rules and
saves it to data/outputs/rules_bank/.

This is the last node in the pipeline. It does not call the LLM.

The output JSON has two top-level keys:
  - metadata : source document, generation timestamp, model, run statistics
  - rules     : list of validated rule dicts (ValidatedRule schema)

State reads:
    main_doc_path   (str)        — path to the main 3GPP document
    validated_rules (list[dict]) — rules that passed all validation stages
    iteration_count (int)        — total loop iterations that ran
    extraction_plan (dict)       — used to count sections that were planned

State writes:
    final_output_path (str) — absolute path to the saved rules bank JSON file
"""

import json
import os
from datetime import datetime, timezone

from config import get_logger
from config.paths import OUTPUTS_RULES_DIR
from config.settings import MODEL
from graph.state import RuleBankState

logger = get_logger(__name__)


def builder_node(state: RuleBankState) -> dict:
    """
    Assembles the rules bank and saves it to disk as a JSON file.

    Filename format: rules_bank_<doc_stem>_<YYYYMMDD_HHMMSS>.json
    """
    logger.info("Builder Node started.")

    validated_rules = state.get("validated_rules", []) or []
    main_doc_path   = state.get("main_doc_path", "unknown")
    iteration_count = state.get("iteration_count", 1) or 1
    plan_sections   = state.get("extraction_plan", {}).get("sections_to_extract", [])

    # Collect unique section_ids that contributed at least one rule
    sections_with_rules = sorted({r.get("section_id", "") for r in validated_rules})

    doc_stem = os.path.splitext(os.path.basename(main_doc_path))[0]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"rules_bank_{doc_stem}_{timestamp}.json"

    os.makedirs(OUTPUTS_RULES_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_RULES_DIR, filename)

    rules_bank = {
        "metadata": {
            "source_document": main_doc_path,
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "model":           MODEL,
            "total_rules":     len(validated_rules),
            "sections_planned":    len(plan_sections),
            "sections_with_rules": len(sections_with_rules),
            "iterations_run":      iteration_count,
        },
        "rules": validated_rules,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rules_bank, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Builder Node complete — {len(validated_rules)} rule(s) saved to '{output_path}'."
    )

    return {"final_output_path": output_path}
