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

from openapi_rulesbank.config import get_logger
from openapi_rulesbank.config.paths import OPENAPI_REFERENCE_DIR
from openapi_rulesbank.config.settings import OPENAPI_REFERENCE_VERSION

logger = get_logger(__name__)

RAW_BASE_URL = "https://raw.githubusercontent.com/OAI/OpenAPI-Specification/main/versions"


def _version_url(version: str) -> str:
    """
    Returns the raw GitHub URL for a specific OpenAPI Specification version
    (e.g. '3.0.1' -> .../versions/3.0.1.md).

    Raises ValueError if the version string is not a valid X.Y.Z semantic version.
    """
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        raise ValueError(
            f"Invalid OpenAPI reference version '{version}'. Expected format X.Y.Z."
        )
    return f"{RAW_BASE_URL}/{version}.md"


def fetch_openapi_reference(version: str = OPENAPI_REFERENCE_VERSION) -> None:
    """
    Fetches a specific OpenAPI Specification version from the OAI GitHub repository
    and saves it to OPENAPI_REFERENCE_DIR/openapi_spec_<version>.md for offline use.

    Defaults to settings.OPENAPI_REFERENCE_VERSION (the legacy version the 3GPP
    ProvMnS input specs declare, e.g. 3.0.1) rather than the latest published spec,
    so the generated rules match the OpenAPI version actually used by the inputs.

    Raises requests.RequestException if the HTTP request fails.
    Raises ValueError if the version is invalid or the content is empty.
    """
    url = _version_url(version)
    output_file = os.path.join(OPENAPI_REFERENCE_DIR, f"openapi_spec_{version}.md")

    logger.info(f"Fetching OpenAPI reference v{version} from {url} ...")
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
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    size_kb = os.path.getsize(output_file) / 1024
    logger.info(f"  Saved -> {output_file}  ({size_kb:.1f} KB)")
