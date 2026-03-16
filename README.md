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
│       └── openapi_spec/          # Local snapshot of the swagger.io OpenAPI spec
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
