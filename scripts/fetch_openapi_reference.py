#!/usr/bin/env python3
# scripts/fetch_openapi_reference.py

"""
CLI wrapper for fetch_openapi_reference() from tools/web_tools.py.

Fetches the configured OpenAPI Specification version (settings.OPENAPI_REFERENCE_VERSION,
default 3.0.1 — the version declared by the 3GPP ProvMnS inputs) from the OAI GitHub
repository and saves it locally as
data/references/openapi_reference/openapi_spec_<version>.md.

Run once before starting the pipeline, or re-run to refresh the snapshot.

Usage:
    python scripts/fetch_openapi_reference.py            # uses settings.OPENAPI_REFERENCE_VERSION
    python scripts/fetch_openapi_reference.py 3.0.1      # explicit version
"""

import sys
from openapi_rulesbank.tools.web_tools import fetch_openapi_reference

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            fetch_openapi_reference(sys.argv[1])
        else:
            fetch_openapi_reference()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
