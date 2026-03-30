# src/tools/document_tools.py

"""
Tools for loading source documents from the local filesystem.

These functions were originally in nodes/reader.py. They are moved here so
that any node in the pipeline can load documents without duplicating code.

Functions:
    load_markdown(file_path)       — reads a local Markdown file and returns its text
    discover_specs(path)           — discovers all .md spec files under a path and
                                     returns their file paths with extracted metadata
                                     (release, series, spec). Supports single file or
                                     directory mode. Adapted from openapi_chatbotUI.
    load_openapi_reference()       — loads the OpenAPI reference document from
                                     OPENAPI_REFERENCE_DIR saved by fetch_openapi_reference()
                                     in web_tools.py. Returns "" if missing.
"""

import os
from typing import List, Dict
from config import get_logger
from config.paths import OPENAPI_REFERENCE_DIR

logger = get_logger(__name__)


def load_markdown(file_path: str) -> str:
    """
    Reads a local Markdown file and returns its full text content.

    Raises FileNotFoundError (with a logged error) if the path does not exist.
    Used for loading 3GPP specs and auxiliary documents stored under data/inputs/.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise



def discover_specs(path: str) -> List[Dict[str, str]]:
    """
    Discovers all .md spec files under a path and returns their metadata.

    Supports two modes:
      - Single file: path points directly to a .md file
      - Directory:   recursively walks all subdirectories for .md files

    Metadata is extracted from the folder hierarchy:
      - release: folder starting with 'Rel-'  (e.g., 'Rel-18')
      - series:  folder ending with '_series' (e.g., '28_series')
      - spec:    filename without .md          (e.g., '28532-i00')

    Returns a list of dicts with keys: file_path, release, series, spec.
    Adapted from openapi_chatbotUI extract_3gpp.ipynb.
    """
    if os.path.isfile(path):
        files_to_process = [(os.path.dirname(path), os.path.basename(path))]
        logger.info(f"discover_specs: single file mode — {os.path.basename(path)}")
    else:
        files_to_process = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.md'):
                    files_to_process.append((root, file))
        logger.info(f"discover_specs: directory mode — found {len(files_to_process)} .md files")

    results = []
    for root, file in files_to_process:
        spec_name = file[:-3]
        path_parts = os.path.normpath(root).split(os.sep)

        release = "unknown"
        series = "unknown"

        for i in range(len(path_parts) - 1, -1, -1):
            part = path_parts[i].strip()
            if part.startswith('Rel-'):
                release = part
                if i + 1 < len(path_parts):
                    next_part = path_parts[i + 1].strip()
                    if next_part.endswith('_series'):
                        series = next_part
                break
            elif part.endswith('_series'):
                series = part
                if i - 1 >= 0 and path_parts[i - 1].strip().startswith('Rel-'):
                    release = path_parts[i - 1].strip()

        if series == "unknown":
            series = os.path.basename(root).strip() or "unknown"

        results.append({
            "file_path": os.path.join(root, file),
            "release": release,
            "series": series,
            "spec": spec_name,
        })

    logger.debug(f"discover_specs: returning {len(results)} entries")
    return results


def load_openapi_reference() -> str:
    """
    Loads the local OpenAPI reference document from OPENAPI_REFERENCE_DIR
    (data/references/openapi_reference/) and returns its content as a single string.

    The file is saved by fetch_openapi_reference() in web_tools.py.
    Run it once to populate the directory before starting the pipeline.

    Guard behaviour:
      - Directory does not exist or has no .md files → logs a warning, returns "".
      - Files found → loads each file, joins with blank line separator.
    """
    entries = discover_specs(OPENAPI_REFERENCE_DIR)

    if not entries:
        logger.warning(
            f"No OpenAPI reference file found in '{OPENAPI_REFERENCE_DIR}'. "
            "Run fetch_openapi_reference() from tools/web_tools.py to populate it. "
            "Continuing with empty OpenAPI context."
        )
        return ""

    parts = [load_markdown(entry["file_path"]) for entry in entries]
    logger.debug(f"Loaded {len(parts)} OpenAPI reference file(s) from '{OPENAPI_REFERENCE_DIR}'.")
    return "\n\n".join(parts)
