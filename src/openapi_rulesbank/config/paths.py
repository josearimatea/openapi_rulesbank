# src/app/config/paths.py
"""
Path configurations for data and files.
These are base paths used internally by settings.py for environment-based switching.
No side effects on import.
"""

import os

# Anchor: resolve ROOT from this file's location so all paths work
# regardless of the working directory (notebooks, scripts, tests).
# paths.py is at src/openapi_rulesbank/config/paths.py -> three levels up = project root.
_THIS_FILE = os.path.abspath(__file__)
ROOT       = f"{os.path.dirname(_THIS_FILE)}/../../.."

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
# Markdown files converted from the .docx sources are written here, separating
# derived artifacts from the original .docx inputs in INPUTS_3GPP_DIR.
INPUTS_3GPP_MD_DIR   = f"{INPUTS_3GPP_DIR}/md"
INPUTS_AUXILIARY_DIR = f"{DATA_DIR}/inputs/auxiliary"
OUTPUTS_DIR             = f"{DATA_DIR}/outputs"
OUTPUTS_RULES_DIR       = f"{OUTPUTS_DIR}/rules_bank"
OUTPUTS_REPORTS_DIR     = f"{OUTPUTS_DIR}/reports"
OUTPUTS_DIAGNOSTIC_DIR  = f"{OUTPUTS_DIR}/diagnostic"
OUTPUTS_CONTRACTS_DIR   = f"{OUTPUTS_DIR}/contracts"
OPENAPI_REFERENCE_DIR = f"{DATA_DIR}/references/openapi_reference"


# --------------------------------------------------------
# Per-service output directories
#
# Runs are scoped to one Management Service (--target), so each service's
# artifacts live under its own subdirectory — data/outputs/<kind>/<Service>/ —
# keeping e.g. ProvMnS and PerfMnS results side by side instead of mixed in one
# folder. Passing no service falls back to the shared directory.
# --------------------------------------------------------

def service_output_dir(base_dir: str, service_name: str = "") -> str:
    """Return `base_dir/<service_name>`, or `base_dir` when no service is given."""
    return os.path.join(base_dir, service_name) if service_name else base_dir


def rules_dir_for(service_name: str = "") -> str:
    """Directory for a service's rules bank JSON files."""
    return service_output_dir(OUTPUTS_RULES_DIR, service_name)


# Name of the subdirectory holding the full-document runs of a service.
FINAL_RULES_SUBDIR = "Final_rules"


def final_rules_dir_for(service_name: str = "") -> str:
    """Directory for a service's full-document rule banks.

    A full run over the whole spec is the result worth keeping, so it lands in
    <service>/Final_rules/ while the scoped and exploratory runs stay in the
    service's root folder. Both the CLI and the full-pipeline test resolve the
    directory here, so they cannot drift apart.
    """
    return os.path.join(rules_dir_for(service_name), FINAL_RULES_SUBDIR)


def diagnostic_dir_for(service_name: str = "") -> str:
    """Directory for a service's diagnostic reports."""
    return service_output_dir(OUTPUTS_DIAGNOSTIC_DIR, service_name)


def contracts_dir_for(service_name: str = "") -> str:
    """Directory for a service's generated service contracts."""
    return service_output_dir(OUTPUTS_CONTRACTS_DIR, service_name)
