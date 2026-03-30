# src/tools/web_tools.py

"""
Tools for fetching external web content.

Functions:
    fetch_openapi_reference() — fetches the latest OpenAPI Specification from the
                                OAI GitHub repository and saves it locally as
                                data/references/openapi_reference/openapi_reference.md
                                for offline use by the pipeline.
"""

import os
import re
import requests

from config import get_logger
from config.paths import OPENAPI_REFERENCE_DIR

logger = get_logger(__name__)

GITHUB_API_VERSIONS = "https://api.github.com/repos/OAI/OpenAPI-Specification/contents/versions"
RAW_BASE_URL        = "https://raw.githubusercontent.com/OAI/OpenAPI-Specification/main/versions"
OUTPUT_FILE         = os.path.join(OPENAPI_REFERENCE_DIR, "openapi_reference.md")


def _get_latest_version_url() -> str:
    """
    Queries the GitHub API to list all files in the OAI versions/ directory,
    filters versioned .md files (e.g. 3.1.1.md), and returns the raw URL
    for the latest version based on semantic versioning.
    """
    response = requests.get(GITHUB_API_VERSIONS, timeout=15)
    response.raise_for_status()

    files = response.json()
    version_files = [
        f["name"] for f in files
        if re.match(r"^\d+\.\d+\.\d+\.md$", f["name"])
    ]

    if not version_files:
        raise ValueError("No versioned spec files found in OAI GitHub repository.")

    latest = sorted(version_files, key=lambda v: [int(x) for x in v[:-3].split(".")])[-1]
    logger.info(f"Latest OpenAPI reference version found: {latest[:-3]}")

    return f"{RAW_BASE_URL}/{latest}"


def fetch_openapi_reference() -> None:
    """
    Fetches the latest OpenAPI Specification from the OAI GitHub repository
    and saves it to OPENAPI_REFERENCE_DIR/openapi_reference.md for offline use.

    Raises requests.RequestException if any HTTP request fails.
    Raises ValueError if no versioned files are found or content is empty.
    """
    url = _get_latest_version_url()

    logger.info(f"Fetching OpenAPI reference from {url} ...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        raise

    content = response.text
    logger.info(f"  {response.status_code} — {len(content)} chars received.")

    if not content.strip():
        raise ValueError("Fetched content is empty.")

    os.makedirs(OPENAPI_REFERENCE_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    logger.info(f"  Saved -> {OUTPUT_FILE}  ({size_kb:.1f} KB)")
