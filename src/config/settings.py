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