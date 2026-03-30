# src/app/config/paths.py
"""
Path configurations for data and files.
These are base paths used internally by settings.py for environment-based switching.
No side effects on import.
"""

import os

# Anchor: resolve ROOT from this file's location so all paths work
# regardless of the working directory (notebooks, scripts, tests).
# paths.py is at src/config/paths.py -> dirname twice = project root.
_THIS_FILE = os.path.abspath(__file__)
ROOT       = f"{os.path.dirname(_THIS_FILE)}/../.."

#--------------------------------------------------------
# Parent repository (Dataset lives outside the project root)
PARENT_REPO    = f"{ROOT}/../.."
DATA_DIRECTORY = f"{PARENT_REPO}/Dataset"

# Specific dataset paths
TSPEC_DATA_TEST_FILE = f"{DATA_DIRECTORY}/TSpec-LLM/3GPP-clean/Rel-18/28_series/28532-i00.md"
TSPEC_DATA_TEST = f"{DATA_DIRECTORY}/TSpec-LLM/3GPP-clean/Rel-18/28_series/"
TSPEC_DATA_PROD = f"{DATA_DIRECTORY}/TSpec-LLM/3GPP-clean"

#--------------------------------------------------------
# Main Paths repository
FILES_DIRECTORY=f"{ROOT}/files"

# Chunk file paths
CHUNKS_DIRECTORY=f"{FILES_DIRECTORY}/chunks"
CHUNKS_FILE_TEST_FILE = f"{CHUNKS_DIRECTORY}/tspec_chunks_test_rel_18_28_28532.pkl"
CHUNKS_FILE_TEST      = f"{CHUNKS_DIRECTORY}/tspec_chunks_test_rel_18_28.pkl"
CHUNKS_FILE_PROD      = f"{CHUNKS_DIRECTORY}/tspec_chunks_prod.pkl"

# Qdrant collection names
COLLECTION_NAME_TEST_FILE = '3gpp_rel18_28_28532'
COLLECTION_NAME_TEST      = '3gpp_rel18_28'
COLLECTION_NAME_PROD      = '3gpp'
SRC_DIRECTORY   = f"{ROOT}/src"
FILES_DIRECTORY = f"{ROOT}/files"
SCRIPTS_DIR     = f"{ROOT}/scripts"

#--------------------------------------------------------
# Project data paths
DATA_DIR = f"{ROOT}/data"
INPUTS_3GPP_DIR      = f"{DATA_DIR}/inputs/3gpp"
INPUTS_AUXILIARY_DIR = f"{DATA_DIR}/inputs/auxiliary"
OUTPUTS_RULES_DIR    = f"{DATA_DIR}/outputs/rules_bank"
OUTPUTS_REPORTS_DIR  = f"{DATA_DIR}/outputs/reports"
OPENAPI_REFERENCE_DIR = f"{DATA_DIR}/references/openapi_reference"
