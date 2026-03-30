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
TEMPERATURE = 0

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

# Section parsing — title validity filter
# When True, parse_sections() discards sections whose title contains no real words
# (i.e., titles made up entirely of symbols like table borders: "+---+---+").
# This safely removes 3GPP document cover-page tables that pass the keyword filter
# because their cell content contains relevant keywords (e.g. "NRM", "mapping").
# Set to False to disable and keep all sections regardless of title format.
FILTER_SYMBOLIC_TITLES = True