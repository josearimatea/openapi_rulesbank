"""Run the full PerfMnS pipeline and surface the section-analyzer trace.

Not a pytest test — a diagnostic. Runs the same graph as test_pipeline.py but
directly (so the analyzer's INFO logs show), to see how the other-spec reference
memory flows across sections: whether the 'General' clause populates it and
whether the 'Type' clause then resolves the embedded base (allOf).

    uv run python tests/investigation/run_pipeline_traced.py
"""

import os
from datetime import datetime

from openapi_rulesbank.config.paths import TSPEC_DATA_TEST_FILE, rules_dir_for
from openapi_rulesbank.config.settings import TARGET_YAML
from openapi_rulesbank.graph.rule_bank_flow import build_initial_state, get_compiled_graph

FINAL_RULES_DIR = os.path.join(rules_dir_for(TARGET_YAML), "Final_rules")
os.makedirs(FINAL_RULES_DIR, exist_ok=True)
stem = os.path.splitext(os.path.basename(TSPEC_DATA_TEST_FILE))[0]
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = os.path.join(FINAL_RULES_DIR, f"rules_bank_{stem}_full_{ts}.json")

state = build_initial_state(
    main_doc_path=os.path.abspath(TSPEC_DATA_TEST_FILE),
    target_yaml=TARGET_YAML,
)
state["final_output_path"] = out

result = get_compiled_graph().invoke(state)
print("OUTPUT:", result.get("final_output_path", ""))
