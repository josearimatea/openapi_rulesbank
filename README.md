# OpenAPI Rules Bank

Multi-agent LangGraph pipeline that extracts a structured rules bank from 3GPP technical specifications to support the creation of 3GPP-compliant OpenAPIs.

## Overview

The pipeline reads 3GPP normative documents (e.g., TS 28.312) and auxiliary references (e.g., TS 32.160, the OpenAPI/Swagger specification), reasons about their content, and produces a structured JSON rules bank that can guide OpenAPI generation.

### Inputs
- **Main 3GPP specification** — the normative document from which rules are extracted (e.g., TS 28.312)
- **Auxiliary 3GPP specification** — optional supporting document that may provide additional context
- **OpenAPI Specification** — fetched from [swagger.io/specification](https://swagger.io/specification/) and indexed locally via RAG

### Output
- **Rules bank** — a structured JSON file containing validated OpenAPI rules derived from the 3GPP specification

---

## Agent Flow

```
[Reader] → [Planner] → [Extractor] → [Reflector] → [Validator] → [Builder]
                                                          ↑             |
                                          error > 10% threshold        |
                                          └─────── loop back ──────────┘
```

| Node | Responsibility |
|---|---|
| **Reader** | Loads and filters the main 3GPP doc and auxiliary references |
| **Planner** | Reasons about document structure and defines the extraction strategy |
| **Extractor** | Extracts raw OpenAPI rules from each section guided by the plan |
| **Reflector** | Applies Chain-of-Thought self-reflection, grounded by RAG retrieval |
| **Validator** | Validates rules structurally (Pydantic) and semantically (LLM) |
| **Builder** | Assembles and saves the final structured rules bank JSON |

### Node Descriptions

#### Reader
**LLM:** no | **RAG:** no | Deterministic

Loads all source documents and prepares them for downstream nodes.

1. Load main 3GPP document (local file or HTTP URL).
2. Split into sections on `## ` headers via `utils.parsers.parse_sections`. Keep only sections with OpenAPI-relevant keywords (`mapping`, `yang`, `template`, `attribute`, `openapi`, `nrm`, etc.).
3. Load auxiliary 3GPP documents (truncated to 2000 chars each).
4. Read local OpenAPI spec snapshot from `data/references/openapi_reference`.

| | |
|---|---|
| **Input** | `main_doc_path`, `auxiliary_doc_paths` |
| **Output** | `parsed_sections` — list of `{section_id, title, content}`; `helper_context` — auxiliary docs joined as one string; `openapi_reference_context` — full OpenAPI spec text |

---

#### Planner
**LLM:** yes (1 call) | **RAG:** no

Reasons over section titles and content previews to define what to extract.

1. Build compact summary table (`section_id | title | first 200 chars`).
2. Send table + first 3000 chars of OpenAPI overview to LLM.
3. LLM returns `ExtractionPlan` with `priority` and `extraction_focus` per section (`high`/`medium`/`low`), using `with_structured_output(ExtractionPlan)`.

| | |
|---|---|
| **Input** | `parsed_sections`, `helper_context`, `openapi_reference_context` |
| **Output** | `extraction_plan` — `{document_summary, sections_to_extract: [{section_id, title, priority, extraction_focus}]}` |

---

#### Extractor
**LLM:** yes (1 call/section) | **RAG:** yes (1 query/section, Qdrant)

Extracts raw OpenAPI rules from each planned section.

Steps per section:
1. RAG query: `"{section_title} — {extraction_focus}"` → 5 OpenAPI chunks.
2. LLM extracts `RawRule` objects from section content only. OpenAPI chunks are reference context, not source. Enforces one rule per HTTP method.
3. Each rule has: `section_id`, `section_title`, `rule_type`, `rule_text`, `openapi_mapping {openapi_object, openapi_field, openapi_value}`.

`rule_type` values: `path_operation` | `schema_property` | `path_parameter` | `query_parameter` | `response` | `request_body` | `security_scheme`

On **loop-back**: processes only sections with failing rules. Sends a CORRECTION TASK prompt listing the exact failed rules and reasons, instructing the LLM to return one corrected rule per entry without adding new rules.

| | |
|---|---|
| **Input** | `parsed_sections`, `extraction_plan`, `helper_context`, `validation_errors` (empty on iter 1), `iteration_count` |
| **Output** | `raw_rules` — list of `RawRule` dicts; `iteration_count` — incremented by 1 |

---

#### Reflector
**LLM:** yes (1 call/rule) | **RAG:** yes (1 query/rule, Qdrant)

Applies CoT self-reflection to each raw rule before validation.

Steps per rule:
1. RAG query: `rule_text + openapi_object + openapi_field` → 5 chunks.
2. LLM answers 4 CoT questions: Is the rule grounded? Is the mapping correct? Does the OpenAPI context confirm it? Confidence score?
3. Returns `ReflectionResult`: `confidence` (0.0–1.0), `reasoning`, `flagged`.
4. Merges into rule dict → `ReflectedRule` (adds `confidence`, `reasoning`, `flagged`, `rag_context`).

| | |
|---|---|
| **Input** | `raw_rules` |
| **Output** | `reflected_rules` — list of `ReflectedRule` dicts |

---

#### Validator
**LLM:** yes (1 call/rule for Stage 2) | **RAG:** no

Validates each reflected rule through three stages:

- **Stage 1a — Pydantic structural (no LLM):** `ValidatedRule(**rule)` — checks all required fields and types.
- **Stage 1b — Mapping consistency / `utils.rules_check` (no LLM):** `check_mapping_for_type(rule_type, openapi_mapping)` — verifies that `openapi_field` and `openapi_value` match the constraints for `rule_type`. E.g. `path_operation`: field must be a single lowercase HTTP method. Runs only if Stage 1a passes.
- **Stage 2 — LLM semantic:** Checks `rule_text` is grounded in section content and `openapi_mapping` is accurate. Flagged rules receive extra scrutiny. Prompt includes `rule_type` with type-specific validation rules (e.g. `response` field = HTTP code). Runs only if Stages 1a and 1b pass.

Rules passing all stages → `validated_rules` (accumulated via `operator.add`). Rules failing any stage → `validation_errors` (with reason and stage).

**Conditional routing** (`conditions.should_loop_or_build`):
- `error_rate = len(validation_errors) / len(reflected_rules)`
- If `error_rate > 10%` AND `iteration_count < 3` → loop to Extractor
- Otherwise → proceed to Builder

| | |
|---|---|
| **Input** | `reflected_rules`, `parsed_sections` |
| **Output** | `validated_rules` — list of `ValidatedRule` dicts (accumulated); `validation_errors` — list of `ValidationError` dicts |

---

#### Builder
**LLM:** no | **RAG:** no | Deterministic

Assembles and persists the final rules bank JSON file.

1. Read `validated_rules` (all iterations accumulated via `operator.add`).
2. Compute summary: `rules_by_type` and `rules_by_section` (Counter).
3. Validate complete output against `schemas.output.RulesBank` (Pydantic).
4. Save to `data/outputs/rules_bank/rules_bank_<doc>_<timestamp>.json`.

Output JSON structure: `metadata` (provenance, model, counts, iterations), `summary` (rules_by_type, rules_by_section, error counts), `rules` (list of full `ValidatedRule` dicts).

| | |
|---|---|
| **Input** | `validated_rules`, `validation_errors`, `main_doc_path`, `iteration_count`, `extraction_plan` |
| **Output** | `final_output_path` — absolute path to the saved JSON file |

---

## Project Structure

```
openapi_rulesbank/
│
├── main.py                        # Pipeline entry point
├── pyproject.toml                 # Project metadata and dependencies
├── .env                           # Environment variables (never commit)
├── .env.example                   # Template for environment variables
│
├── data/
│   ├── inputs/
│   │   ├── 3gpp/                  # Main 3GPP specification files (e.g., 28312)
│   │   └── auxiliary/             # Auxiliary 3GPP specification files (e.g., 32.160)
│   ├── outputs/
│   │   ├── rules_bank/            # Generated rules bank JSON files
│   │   └── reports/               # Validation reports
│   └── references/
│       └── openapi_reference/     # Local snapshot of the OAI OpenAPI specification
│
├── src/
│   │
│   ├── config/                    # Centralized configuration
│   │   ├── __init__.py            # Exports get_logger() and all config symbols
│   │   ├── settings.py            # Environment variables and app settings
│   │   ├── paths.py               # File and directory path definitions
│   │   ├── llm_config.py          # LLM instance initialization
│   │   ├── hardware.py            # GPU/CPU device detection
│   │   └── logging_config.py      # Logging setup (level, format, suppression)
│   │
│   ├── graph/                     # LangGraph graph definition
│   │   ├── __init__.py
│   │   ├── state.py               # RuleBankState TypedDict (shared state schema)
│   │   ├── conditions.py          # Conditional edge routing functions
│   │   └── rule_bank_flow.py      # Graph construction and compilation
│   │
│   ├── nodes/                     # Graph node implementations
│   │   ├── __init__.py
│   │   ├── reader.py              # Loads and filters 3GPP and helper documents
│   │   ├── planner.py             # Reasons about structure and plans extraction
│   │   ├── extractor.py           # Extracts raw rules section by section
│   │   ├── reflector.py           # Self-reflection and CoT reasoning over rules
│   │   ├── validator.py           # Structural + semantic validation of rules
│   │   └── builder.py             # Builds and saves the final rules bank JSON
│   │
│   ├── tools/                     # LangGraph tool definitions
│   │   ├── __init__.py
│   │   ├── document_tools.py      # PDF/Markdown loading and chunking
│   │   ├── web_tools.py           # Fetching and caching content from swagger.io
│   │   └── schema_tools.py        # JSON Schema and OpenAPI schema validation
│   │
│   ├── rag/                       # RAG pipeline (Qdrant vector store)
│   │   ├── __init__.py
│   │   ├── ingestion.py           # Document chunking and embedding
│   │   ├── retriever.py           # Semantic search over indexed documents
│   │   └── indexer.py             # Qdrant collection management and upsert
│   │
│   ├── schemas/                   # Pydantic data models
│   │   ├── __init__.py
│   │   ├── state.py               # RuleBankState Pydantic model (for test validation)
│   │   ├── rule.py                # Single rule data model
│   │   └── output.py              # Final rules bank output model
│   │
│   ├── prompts/                   # Prompt templates (one file per node)
│   │   ├── __init__.py
│   │   ├── reader_prompts.py
│   │   ├── planner_prompts.py
│   │   ├── extractor_prompts.py
│   │   ├── reflector_prompts.py
│   │   └── validator_prompts.py
│   │
│   └── utils/                     # Shared utility functions
│       ├── __init__.py
│       ├── parsers.py             # 3GPP Markdown/table parsing utilities
│       └── formatters.py          # Output formatting helpers
│
└── tests/
    ├── unit/                      # Per-node interactive notebooks
    │   ├── test_reader.ipynb
    │   ├── test_extractor.ipynb
    │   └── test_validator.ipynb
    └── integration/               # End-to-end pipeline notebook
        └── test_flow.ipynb
```

---

## Setup

```bash
# Copy and fill in environment variables
cp .env.example .env

# Install dependencies
uv sync

# Run the pipeline
uv run run-flow
```

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4.1-mini |
| Graph orchestration | LangGraph |
| LLM framework | LangChain |
| Data validation | Pydantic |
| Vector store | Qdrant |
| Document parsing | Markdown, jsonschema |
| Package manager | uv |
