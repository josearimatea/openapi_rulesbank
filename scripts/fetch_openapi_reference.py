#!/usr/bin/env python3
# scripts/fetch_openapi_reference.py

"""
CLI wrapper for fetch_openapi_reference() from tools/web_tools.py.

Fetches the latest OpenAPI Specification from the OAI GitHub repository
and saves it locally as data/references/openapi_reference/openapi_reference.md.

Run once before starting the pipeline, or re-run to refresh the snapshot.

Usage:
    python scripts/fetch_openapi_reference.py
"""

import sys
from tools.web_tools import fetch_openapi_reference

if __name__ == "__main__":
    try:
        fetch_openapi_reference()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
