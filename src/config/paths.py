# src/app/config/paths.py
"""
Path configurations for data and files.
These are base paths used internally by settings.py for environment-based switching.
No side effects on import.
"""

import os

#--------------------------------------------------------
# Parent repository
PARENT_REPO = '../../../../..'
DATA_DIRECTORY = f"{PARENT_REPO}/Dataset"

# Specific dataset paths
TSPEC_DATA_TEST_FILE = f"{DATA_DIRECTORY}/TSpec-LLM/3GPP-clean/Rel-18/28_series/28532-i00.md"
TSPEC_DATA_TEST = f"{DATA_DIRECTORY}/TSpec-LLM/3GPP-clean/Rel-18/28_series/"
TSPEC_DATA_PROD = f"{DATA_DIRECTORY}/TSpec-LLM/3GPP-clean"

#--------------------------------------------------------
# Main Paths repository
ROOT='../..'
SRC_DIRECTORY=f"{ROOT}/src"
FILES_DIRECTORY=f"{ROOT}/files"
