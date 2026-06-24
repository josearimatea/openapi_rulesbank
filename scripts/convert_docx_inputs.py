#!/usr/bin/env python3
# scripts/convert_docx_inputs.py

"""
CLI wrapper for convert_docx_to_md() from tools/docx_converter.py.

Pre-processing step (run once, before the pipeline): converts the 3GPP .docx
sources in data/inputs/3gpp/ to Markdown via pandoc, writing the results to
data/inputs/3gpp/md/. This mirrors the conversion done by the dataset notebook
process-3GPP.ipynb. The pipeline (main.py) then consumes the generated .md only.

The pandoc binary ships with the `pypandoc-binary` dependency (installed by
`uv sync`), so no system-wide pandoc install is needed.

Usage:
    # Convert every .docx in data/inputs/3gpp/ -> data/inputs/3gpp/md/
    python scripts/convert_docx_inputs.py

    # Convert specific files
    python scripts/convert_docx_inputs.py data/inputs/3gpp/28532-i20.docx

    # Re-convert even if the .md already exists
    python scripts/convert_docx_inputs.py --overwrite
"""

import argparse
import os
import sys

from openapi_rulesbank.config.paths import INPUTS_3GPP_DIR, INPUTS_3GPP_MD_DIR
from openapi_rulesbank.tools.docx_converter import convert_docx_to_md, discover_docx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 3GPP .docx inputs to Markdown for the pipeline.",
    )
    parser.add_argument(
        "docx",
        nargs="*",
        metavar="PATH",
        help="Specific .docx files to convert. Defaults to all .docx in "
             "data/inputs/3gpp/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run pandoc even if the target .md already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    docx_paths = args.docx or discover_docx(INPUTS_3GPP_DIR)
    if not docx_paths:
        print(f"No .docx files found in {INPUTS_3GPP_DIR}.", file=sys.stderr)
        sys.exit(1)

    for docx in docx_paths:
        out_path = os.path.join(
            INPUTS_3GPP_MD_DIR,
            os.path.splitext(os.path.basename(docx))[0] + ".md",
        )
        try:
            md = convert_docx_to_md(docx, out_path, overwrite=args.overwrite)
            print(f"OK  {os.path.basename(docx)} -> {md}")
        except Exception as e:
            print(f"ERROR converting {docx}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
