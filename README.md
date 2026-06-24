# OpenAPI Rules Bank

Multi-agent LangGraph pipeline that extracts a structured rules bank from 3GPP technical specifications to support the creation of 3GPP-compliant OpenAPIs.

## Overview

The pipeline reads 3GPP normative documents (e.g., TS 28.532) and auxiliary references (e.g., TS 32.160, the OpenAPI/Swagger specification), reasons about their content, and produces a structured JSON rules bank that can guide OpenAPI generation.

### Inputs
- **Main 3GPP specification** — the normative document from which rules are extracted (e.g., TS 28.532). Provided as `.docx` under `data/inputs/3gpp/` and converted to Markdown by a pre-processing step (see [Input pre-processing](#input-pre-processing)); the pipeline consumes the generated `.md`.
- **Target OpenAPI YAML** (`--target`, **required**) — the OpenAPI document the rules are built for (e.g. `ProvMnS`, `PerfMnS`). The Reader extracts **only that document's subsection** from the spec's Annex A (e.g. `TS28532_ProvMnS.yaml`) and injects it into `helper_context`, so the Planner and Extractor see exactly the target YAML format the rules must support.
- **Auxiliary 3GPP specification** (`--aux`) — optional supporting document(s) that may provide additional context.
- **Legacy OpenAPI YAML** (`--legacy-yaml`) — optional previous-release YAML (e.g. a Rel-17 `ProvMnS.yaml`) added as extra context when building rules for a newer release (e.g. Rel-18).
- **OpenAPI Specification** — fetched from [swagger.io/specification](https://swagger.io/specification/) and indexed locally via RAG (Qdrant).

> **Per-YAML selection.** TS 28.532's Annex A bundles several OpenAPI documents (`TS28532_ProvMnS.yaml`, `…PerfMnS.yaml`, `…HeartbeatNtf.yaml`, etc.), each under its own `## A.x.y OpenAPI document "<name>.yaml"` heading. The Reader extracts **only** the `--target` document's subsection (its heading up to the next heading) — not the whole annex — so only the chosen YAML's format reaches the prompts.

### Input pre-processing

The 3GPP sources are `.docx`. Convert them to Markdown once, before running the pipeline:

```bash
uv run python scripts/convert_docx_inputs.py
```

This writes `data/inputs/3gpp/md/<spec>.md` using pandoc (bundled via the `pypandoc-binary` dependency — no system pandoc needed), mirroring the dataset's `process-3GPP.ipynb`. It is idempotent: existing `.md` files are skipped unless `--overwrite` is passed.

### Output
- **Rules bank** — a structured JSON file containing validated OpenAPI rules derived from the 3GPP specification
- **Diagnostic report** — optional per-iteration telemetry written to `data/outputs/diagnostic/` when `DIAGNOSTIC_MODE=True`

---

## Agent Flow

The pipeline runs **per section** with two conditional edges:
- After the **Validator**: retry the current section (back to Extractor) or proceed to Builder
- After the **Builder**: process the next section (back to Extractor) or end

```
[Reader] → [Planner] → [Extractor] → [Reflector] → [Validator] ──→ [Builder] → END
                            ↑                            │              │
                            │  error rate > threshold    │   next section available
                            │  AND retries < max         │              │
                            └────── retry section ───────┘              │
                            └─────────────── next section ──────────────┘
```

| Node | Responsibility |
|---|---|
| **Reader** | Loads/filters the main 3GPP doc and auxiliary docs, injects the target YAML's Annex A subsection (+ optional legacy YAML), and loads the OpenAPI reference |
| **Planner** | Reasons about document structure and defines the extraction strategy |
| **Extractor** | Extracts raw OpenAPI rules from each section guided by the plan and Validator feedback |
| **Reflector** | Per-rule CoT self-reflection (Fase 1) + per-section completeness analysis (Fase 2), grounded by RAG |
| **Validator** | Structural validation (Pydantic + mapping consistency) + 3-stage semantic validation (LLM) |
| **Builder** | Assembles and saves the rules bank JSON after each section; force-includes unresolved rules at max retries |

### Node Descriptions

#### Reader
**LLM:** no | **RAG:** no | Deterministic

Loads all source documents and prepares them for downstream nodes.

1. Load main 3GPP document (`.md`, pre-converted from `.docx`).
2. Split into sections via `utils.parsers.parse_sections`. Setext headings (`----`/`====`) are normalized to ATX first, so `.md` produced by any pandoc version splits on `## ` consistently. Optional symbolic-title and keyword filters apply.
3. Load auxiliary 3GPP documents (truncated to 2000 chars each).
4. **Extract the target YAML's Annex A subsection** — `tools.docx_converter.extract_annex_a_yaml(raw, target_yaml)` pulls **only** the `target_yaml` document's subsection from Annex A (e.g. `ProvMnS` → the `TS28532_ProvMnS.yaml` block, heading up to the next heading — not the whole annex) and appends it to `helper_context` as the target format. Controlled by `ANNEX_A_AS_HELPER` / `ANNEX_A_MAX_CHARS` in `settings.py`. **This is where the target YAML enters the flow** — it then reaches the Planner and Extractor through `helper_context`.
5. If `legacy_yaml_path` is set, append that previous-release YAML as extra context (truncated by `LEGACY_YAML_MAX_CHARS`).
6. Read local OpenAPI spec snapshot from `data/references/openapi_reference`.

| | |
|---|---|
| **Input** | `main_doc_path`, `auxiliary_doc_paths`, `target_yaml`, `legacy_yaml_path` |
| **Output** | `parsed_sections` — list of `{section_id, title, content}`; `helper_context` — auxiliary docs **+ the target YAML subsection** (+ optional legacy YAML) joined as one string; `openapi_reference_context` — full OpenAPI spec text; `excluded_sections_reader` — sections filtered out by symbolic-title heuristics |

---

#### Planner
**LLM:** yes (1 call) | **RAG:** no

Reasons over section titles and content previews to define what to extract.

1. Build compact summary table (`section_id | title | first 200 chars`).
2. Send table + first 3000 chars of OpenAPI overview to LLM.
3. LLM returns `ExtractionPlan` with `priority` and `extraction_focus` per section (`high`/`medium`/`low`), using `with_structured_output(ExtractionPlan)`. Sections not selected go to `excluded_sections_planner`.

| | |
|---|---|
| **Input** | `parsed_sections`, `helper_context`, `openapi_reference_context` |
| **Output** | `extraction_plan` — `{document_summary, sections_to_extract: [{section_id, title, priority, extraction_focus}]}`; `excluded_sections_planner` |

---

#### Extractor
**LLM:** yes (1 call/section) | **RAG:** yes (1 query/section, Qdrant)

Extracts raw OpenAPI rules from the **current** section guided by Validator feedback.

Steps per section:
1. RAG query: `"{section_title} — {extraction_focus}"` → 5 OpenAPI chunks.
2. LLM extracts `RawRule` objects from section content only. OpenAPI chunks are reference context, not source. Enforces one rule per HTTP method.
3. Each rule has: `section_id`, `section_title`, `rule_type`, `rule_text`, `openapi_mapping {openapi_object, openapi_field, openapi_value}`.

`rule_type` values: `path_operation` | `schema_property` | `path_parameter` | `query_parameter` | `response` | `request_body` | `security_scheme`

On **section retry** (loop-back): the Extractor receives two distinct instructions from the Validator:
- `validation_errors` filtered to **`correction`** type only — list of rules to re-extract with the Validator's instruction. `discard` errors require no Extractor action.
- `validator_section_reflection` — final list of missing rules to add to the section.

| | |
|---|---|
| **Input** | `parsed_sections`, `extraction_plan`, `helper_context`, `validation_errors` (filtered), `validator_section_reflection`, `current_section_idx`, `section_iteration_count` |
| **Output** | `raw_rules`; `section_iteration_count` incremented |

---

#### Reflector
**LLM:** yes (1 call/rule for Fase 1, 1 call/section for Fase 2) | **RAG:** yes (1 query/rule, Qdrant)

Two phases of analysis — Fase 1 per rule, Fase 2 per section.

**Fase 1 — Per-rule CoT reflection:**
1. RAG query: `rule_text + openapi_object + openapi_field` → 5 chunks.
2. LLM answers CoT questions: Is the rule grounded? Is the mapping correct? Does the OpenAPI context confirm it? Confidence score? Should it be split or discarded?
3. Returns `ReflectionResult`: `confidence` (0.0–1.0), `reasoning`, `flagged`, `split_suggestion`, `discard_suggestion`.
4. Merges into rule dict → `ReflectedRule` (adds `reflection_*` fields).

**Fase 2 — Per-section completeness analysis:**
1. Looks at all rules from this section + previous-iteration `validator_section_reflection` + already-validated rules from earlier iterations (`validated_rules_by_section`).
2. LLM identifies which OpenAPI rules are still missing from the section.
3. Writes `section_reflection` — analysis input consumed by Validator Stage 3 only. **Not** consumed by the Extractor.

| | |
|---|---|
| **Input** | `raw_rules`, `parsed_sections`, `validated_rules_by_section`, `validator_section_reflection` (previous iteration) |
| **Output** | `reflected_rules`; `section_reflection` |

---

#### Validator
**LLM:** yes (multiple calls per section) | **RAG:** no

Validates each reflected rule via structural checks (no LLM) followed by a 3-stage semantic flow (LLM). Produces two distinct outputs for the Extractor: per-rule verdicts and a per-section missing-rules instruction.

**Structural validation (no LLM, runs first for all rules):**
- **Stage 1a — Pydantic structural:** `ValidatedRule(**rule)` — checks all required fields and types.
- **Stage 1b — Mapping consistency:** `utils.rules_check.check_mapping_for_type(rule_type, openapi_mapping)` — verifies that `openapi_field` and `openapi_value` match the constraints for `rule_type`. E.g. `path_operation`: field must be a single lowercase HTTP method.

**Semantic validation (LLM, 3 ordered stages per section):**
- **Stage 1 — Section init** *(`SectionAnalysisResult`)*: filter the previous iteration's `validator_section_reflection.missing_rules`. Remove items now covered by current rules; keep only what is still absent. No new rules added here.
- **Stage 2 — Per-rule verdict** *(`ValidationVerdict`)*: for each rule, decide `valid` / `correction` / `split` / `discard` and emit `new_missing_rules` — sibling gaps discovered while assessing this rule.
- **Stage 3 — Reflector review** *(`ReflectorReviewResult`)*: incorporate the Reflector's Fase 2 `section_reflection` into the final `validator_section_reflection.missing_rules`.

Rules passing all stages → `validated_rules` (accumulated via `operator.add`). Rules failing any stage → `validation_errors` (with `error_type`, `stage`, `instruction`).

**Conditional routing** (`conditions.should_retry_section_or_build`):
- Counts only **`correction`** errors against `MAX_ITERATIONS` (`split` and `discard` route through `validator_section_reflection`, not retry).
- `error_rate = len(correction_errors) / len(reflected_rules)`
- If `error_rate > VALIDATION_ERROR_THRESHOLD` (10%) AND `section_iteration_count < MAX_ITERATIONS` (3) → retry section
- Otherwise → proceed to Builder

| | |
|---|---|
| **Input** | `reflected_rules`, `section_reflection` (Reflector Fase 2), `validator_section_reflection` (previous iteration), `parsed_sections` |
| **Output** | `validated_rules` (accumulated); `validation_errors`; `validator_section_reflection` (final per-section instruction); `validated_rules_by_section` (compact summary for Reflector context) |

---

#### Builder
**LLM:** no | **RAG:** no | Deterministic

Called once per section. Writes (or overwrites) the rules bank JSON file with **all rules accumulated so far** (current section + all previous sections).

1. **Force-include** at max retries: if `section_iteration_count >= MAX_ITERATIONS` and unresolved semantic errors remain, append them to `validated_rules` with `validation_passed=False` and a `validation_notes` explaining the failure. Force-included rules are returned via `validated_rules` so `operator.add` accumulates them into state across sections.
2. Compute summary (`rules_by_type`, `rules_by_section`).
3. Validate output against `schemas.output.RulesBank` (Pydantic) and write to `data/outputs/rules_bank/rules_bank_<doc>_<scope>_<timestamp>.json` (`scope` = `full` or `partial`).
4. Advance `current_section_idx` by 1, reset `section_iteration_count`, clear `validation_errors`, `section_reflection`, and `validator_section_reflection` for the next section.
5. On the **last** section, save the diagnostic report to `data/outputs/diagnostic/` if `DIAGNOSTIC_MODE=True`.

Output JSON: `metadata` (provenance, model, counts, section accounting), `summary` (rules_by_type, rules_by_section, error counts), `rules` (list of full `RuleOutput` dicts including `validation_passed`).

| | |
|---|---|
| **Input** | `validated_rules`, `validation_errors`, `main_doc_path`, `section_iteration_count`, `current_section_idx`, `extraction_plan`, `parsed_sections`, `excluded_sections_reader`, `excluded_sections_planner`, `final_output_path` |
| **Output** | `final_output_path`; `current_section_idx` (+1); `section_iteration_count` (=0); cleared per-section fields; `validated_rules` (force-included additions, if any) |

---

## Feedback Loop Architecture

Two distinct per-section feedback channels separate **analysis** (Reflector) from **decision** (Validator):

| State field | Written by | Read by | Purpose |
|---|---|---|---|
| `section_reflection` | Reflector (Fase 2) | Validator (Stage 3) | Reflector's analysis of which rules are missing — input only |
| `validator_section_reflection` | Validator (Stage 3) | Extractor, `conditions.py` | Validator's **final** instruction to the Extractor on missing rules |
| `validation_errors` (`correction`) | Validator (Stage 2) | Extractor | Per-rule corrections to apply on retry |
| `validation_errors` (`discard` / `split`) | Validator (Stage 2) | (handled via missing_rules) | Routed through `validator_section_reflection` |

The Reflector reasons; the Validator decides. The Extractor only acts on what the Validator says.

---

## Project Structure

```
openapi_rulesbank/
│
├── pyproject.toml                 # Project metadata and dependencies
├── pytest.ini                     # Pytest config (pythonpath, integration marker)
├── .env / .env.example            # Environment variables
│
├── data/
│   ├── inputs/
│   │   ├── 3gpp/                  # Source 3GPP specs (.docx) + md/ (converted .md)
│   │   └── auxiliary/             # Optional auxiliary documents
│   ├── outputs/
│   │   ├── rules_bank/            # Generated rules bank JSON files
│   │   └── diagnostic/            # Per-iteration diagnostic reports (DIAGNOSTIC_MODE=True)
│   └── references/
│       └── openapi_reference/     # Local snapshot of the OAI OpenAPI specification
│
├── scripts/
│   ├── convert_docx_inputs.py     # Pre-processing: .docx → .md (pandoc via pypandoc-binary)
│   ├── fetch_openapi_reference.py # Fetches the OpenAPI spec snapshot
│   └── index_openapi_reference.py # Indexes the OpenAPI reference into Qdrant
│
├── src/openapi_rulesbank/
│   ├── main.py                    # Pipeline entry point (consumes .md only)
│   │
│   ├── config/                    # Centralized configuration
│   │   ├── __init__.py            # Exports get_logger() and all config symbols
│   │   ├── settings.py            # Env vars, MAX_ITERATIONS, VALIDATION_ERROR_THRESHOLD, DIAGNOSTIC_MODE
│   │   ├── paths.py               # File and directory path definitions
│   │   ├── llm_config.py          # LLM instance initialization
│   │   ├── hardware.py            # GPU/CPU device detection
│   │   └── logging_config.py      # Logging setup
│   │
│   ├── graph/                     # LangGraph graph definition
│   │   ├── state.py               # RuleBankState TypedDict (shared state schema)
│   │   ├── conditions.py          # should_retry_section_or_build, should_next_section_or_end
│   │   └── rule_bank_flow.py      # get_compiled_graph, get_compiled_graph_from_planner
│   │
│   ├── nodes/                     # Graph node implementations
│   │   ├── reader.py              # Loads and filters 3GPP and helper documents
│   │   ├── planner.py             # Plans the extraction strategy
│   │   ├── extractor.py           # Extracts raw rules per section, acts on Validator feedback
│   │   ├── reflector.py           # Fase 1 (per-rule CoT) + Fase 2 (per-section analysis)
│   │   ├── validator.py           # Structural + 3-stage semantic validation
│   │   └── builder.py             # Builds and saves the rules bank JSON; force-include at max retries
│   │
│   ├── tools/
│   │   ├── docx_converter.py      # .docx → .md conversion + per-target Annex A YAML extraction
│   │   ├── document_tools.py      # Document loading utilities
│   │   ├── web_tools.py           # Fetching content from swagger.io
│   │   └── rag_tools.py           # RAG retrieval helpers
│   │
│   ├── rag/                       # RAG pipeline (Qdrant vector store)
│   │   ├── qdrant_factory.py      # Qdrant client + embeddings factory
│   │   ├── retriever.py           # Semantic search over indexed documents
│   │   └── self_query.py          # Self-querying retriever
│   │
│   ├── schemas/                   # Pydantic data models
│   │   ├── rules.py               # RawRule, ReflectedRule, ValidatedRule, ValidationVerdict, etc.
│   │   └── output.py              # RulesBank, RulesBankMetadata, RulesBankSummary, RuleOutput
│   │
│   ├── prompts/                   # Prompt templates (one file per node)
│   │   ├── planner_prompts.py
│   │   ├── extractor_prompts.py
│   │   ├── reflector_prompts.py   # Fase 1 + Fase 2 prompts
│   │   └── validator_prompts.py   # Stage 1 + Stage 2 + Stage 3 prompts
│   │
│   └── utils/                     # Shared utility functions
│       ├── parsers.py             # 3GPP Markdown/table parsing
│       ├── rules_check.py         # Structural mapping consistency checks
│       └── diagnostic.py          # DiagnosticCollector — per-iteration telemetry
│
└── tests/
    ├── conftest.py                # Session-scoped qdrant fixture (auto-starts Docker)
    ├── unit/                      # Fast unit tests (no LLM/RAG)
    │   ├── test_docx_converter.py # convert_docx_to_md guards + extract_annex_a
    │   └── test_parsers_setext.py # parse_sections ATX + setext heading handling
    └── integration/
        ├── test_pipeline.py                 # Full-document integration test
        └── test_pipeline_10sections.py      # Scoped 10-section integration test
```

---

## Setup

```bash
# Copy and fill in environment variables
cp .env.example .env

# Install dependencies (includes pypandoc-binary, which bundles pandoc)
uv sync

# Convert the 3GPP .docx inputs to Markdown (one-time pre-processing)
uv run python scripts/convert_docx_inputs.py

# Index the OpenAPI reference into Qdrant (one-time)
docker run -d --name qdrant-local -p 6333:6333 qdrant/qdrant
uv run python scripts/index_openapi_reference.py

# Run the pipeline on a converted .md (--target is required)
uv run python -m openapi_rulesbank.main --doc data/inputs/3gpp/md/28532-i20.md --target ProvMnS
# (equivalently, via the installed entry point:)
uv run openapi-rulesbank --doc data/inputs/3gpp/md/28532-i20.md --target ProvMnS

# Pick any OpenAPI document from the spec's Annex A, and/or pass a legacy YAML
uv run openapi-rulesbank --doc data/inputs/3gpp/md/28532-i20.md --target PerfMnS
uv run openapi-rulesbank --doc data/inputs/3gpp/md/28532-i20.md --target ProvMnS --legacy-yaml rel17_ProvMnS.yaml
```

### Running tests

```bash
# Fast unit tests (no LLM/RAG)
uv run pytest tests/unit -q

# Full-document integration test (real LLM calls — slow, costs tokens)
pytest tests/integration/test_pipeline.py -v

# Scoped 10-section integration test (faster, still real LLM calls)
pytest tests/integration/test_pipeline_10sections.py -v
```

The session-scoped `qdrant` fixture in `tests/conftest.py` starts the `qdrant-local` Docker container automatically before integration tests and stops it on teardown.

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4.1-mini |
| Graph orchestration | LangGraph |
| LLM framework | LangChain |
| Data validation | Pydantic |
| Vector store | Qdrant |
| Document conversion | pandoc (bundled via `pypandoc-binary`) |
| Document parsing | Markdown, jsonschema |
| Package manager | uv |
