#!/usr/bin/env python3
# scripts/index_openapi_reference.py

"""
CLI wrapper for index_openapi_reference() from tools/rag_tools.py.

Indexes the local OpenAPI reference document into Qdrant for use by the
Extractor and Reflector nodes. Run once after fetch_openapi_reference.py.
The Qdrant collection persists across pipeline runs — no need to reindex.

Usage:
    python scripts/index_openapi_reference.py           # skip if already indexed
    python scripts/index_openapi_reference.py --force   # force full reindex
"""

import sys
import argparse
from tools.rag_tools import index_openapi_reference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index the OpenAPI reference document into Qdrant."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    args = parser.parse_args()

    try:
        index_openapi_reference(force=args.force)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
