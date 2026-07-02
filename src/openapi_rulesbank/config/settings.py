# src/app/config/settings.py
"""
Settings values and environment variables.
Centralizes paths, hosts, ports, and API keys.
No side effects on import.
"""

import os
from .paths import *  # Import all paths and settings from paths.py for centralization
from dotenv import load_dotenv  # Load .env file

load_dotenv()  # Ensure .env is loaded before os.getenv

# APP_ENV is set in .env and determines which dataset and collection to use
# APP_ENV Options: "prod", "test" or "test_file"
APP_ENV = os.getenv("APP_ENV", "test_file")  # Default to 'test_file' for safety

# TSPEC_DATA settings
TSPEC_DATA = TSPEC_DATA_TEST_FILE if APP_ENV == 'test_file' else TSPEC_DATA_TEST if APP_ENV == 'test' else TSPEC_DATA_PROD

# Chunk settings
CHUNKS_FILE = CHUNKS_FILE_TEST_FILE if APP_ENV == 'test_file' else CHUNKS_FILE_TEST if APP_ENV == 'test' else CHUNKS_FILE_PROD

# Qdrant settings
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
COLLECTION_NAME = COLLECTION_NAME_TEST_FILE if APP_ENV == 'test_file' else COLLECTION_NAME_TEST if APP_ENV == 'test' else COLLECTION_NAME_PROD

# Retrieval settings
NUMBER_RETRIEVE_CHUNKS = 10

# Load OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# LLM Settings
MODEL = "gpt-4.1-mini"
# MODEL = "gpt-5-mini"
TEMPERATURE = 0

# OpenAPI reference version to fetch.
# The 3GPP ProvMnS specs under data/inputs/3gpp declare `openapi: 3.0.1`, so the
# reference used to generate rules must match that legacy version rather than the
# latest published spec (e.g. 3.2.0).
OPENAPI_REFERENCE_VERSION = "3.0.1"

# RAG — OpenAPI reference collection
# Same embedding model used in openapi_chatbotUI (free, local, no API key needed).
OPENAPI_COLLECTION_NAME  = "openapi_reference"
CHUNK_SIZE               = 1000  # characters per chunk
CHUNK_OVERLAP            = 200   # overlap between consecutive chunks
EMBEDDING_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM            = 384   # output dimension for all-MiniLM-L6-v2
OPENAPI_REFERENCE_RETRIEVE_CHUNKS = 5  # number of chunks returned by search_openapi_reference()

# Pipeline flow control
VALIDATION_ERROR_THRESHOLD = 0.10  # Loop back to Extractor if error rate exceeds 10%
MAX_ITERATIONS = 3                  # Maximum Extractor → Validator loops before forcing forward

# ContractBuilder — semantic scope contract for the target service
# CONTRACT_LOCATOR_PREVIEW_CHARS: chars of each section's content sent in the
#   locator table (LLM call 1). The locator only needs a preview to decide
#   relevance — the full content of located sections goes to LLM call 2.
# CONTRACT_FALLBACK_SECTIONS: sections used when the locator finds nothing — the
#   opening sections of a 3GPP spec are usually the overview and definitions.
CONTRACT_LOCATOR_PREVIEW_CHARS = 200
CONTRACT_FALLBACK_SECTIONS     = 5

# Section parsing — title validity filter
# When True, parse_sections() discards sections whose title contains no real words
# (i.e., titles made up entirely of symbols like table borders: "+---+---+").
# This safely removes 3GPP document cover-page tables that pass the keyword filter
# because their cell content contains relevant keywords (e.g. "NRM", "mapping").
# Set to False to disable and keep all sections regardless of title format.
FILTER_SYMBOLIC_TITLES = True

# Section parsing — keyword relevance filter
# When True, parse_sections() keeps only sections whose text contains at least one
# keyword from RELEVANT_KEYWORDS in parsers.py. Disabled by default because the
# keyword list may not cover all relevant terminology in a given spec.
# Enable only after reviewing and validating the keyword list for the target document.
FILTER_BY_KEYWORDS = False

# Target OpenAPI YAML — which OpenAPI document the rules are built for. Used as a
# label for the legacy YAML and downstream metadata; the rules are built from the
# spec sections only (the document's Annex A is not used as context).
#
# In normal use the target is REQUIRED from the user via main.py --target. This
# value is the fallback used by local tests and programmatic reader_node() calls
# that don't set state["target_yaml"]; it is "ProvMnS" for the local test setup.
TARGET_YAML = "ProvMnS"

# Optional legacy OpenAPI YAML — a previous release's YAML (e.g. Rel-17) given as
# extra context for building Rel-18 rules. Provided via main.py --legacy-yaml; no
# default (the feature is off unless a path is passed). Truncated by the value
# below when injected into helper_context.
LEGACY_YAML_MAX_CHARS = 8000

# Diagnostic mode — when True, the Validator records per-iteration details
# (reflected rules, validation errors, section feedback, error rate) and the
# Builder writes a full JSON report to data/outputs/diagnostic/.
# Set to False in production to avoid the overhead.
DIAGNOSTIC_MODE = False