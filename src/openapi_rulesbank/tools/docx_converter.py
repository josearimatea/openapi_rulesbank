# src/tools/docx_converter.py

"""
Tools for converting 3GPP source documents (.docx) into Markdown.

The conversion replicates the step performed by the dataset notebook
`process-3GPP.ipynb` (the source of the .md TSpec corpus): pandoc in standalone
mode (`pandoc -s`), with no extra flags, so the generated Markdown matches the
existing TSpec pattern (media images, anchors, section headings). The pandoc
binary is provided by the `pypandoc-binary` package (installed via uv), so no
system-wide pandoc install is required; pypandoc locates the bundled binary.

Functions:
    convert_docx_to_md(docx_path, out_path=None, *, overwrite=False) -> str
        Converts a single .docx file to Markdown via pandoc. Idempotent.
    discover_docx(directory) -> list[str]
        Lists all .docx files under a directory (non-recursive), sorted.
"""

import os
from typing import List, Optional

import pypandoc

from openapi_rulesbank.config import get_logger

logger = get_logger(__name__)


def convert_docx_to_md(
    docx_path: str,
    out_path: Optional[str] = None,
    *,
    overwrite: bool = False,
) -> str:
    """
    Converts a .docx file to Markdown with pandoc in standalone mode (`-s`).

    Uses pypandoc (binary bundled by pypandoc-binary), matching the conversion
    in the dataset's process-3GPP.ipynb. No extra flags, so the output follows
    the TSpec .md pattern.

    Args:
        docx_path: path to the source .docx file.
        out_path:  destination .md path. Defaults to the same directory and base
                   name as the .docx (e.g. 28532-i20.docx -> 28532-i20.md).
        overwrite: if False (default) and the .md already exists, the existing
                   file path is returned without re-running pandoc (idempotent,
                   matching the notebook's "skip if exists" behaviour).

    Returns:
        The absolute path to the generated (or existing) Markdown file.

    Raises:
        FileNotFoundError: if docx_path does not exist.
        OSError / RuntimeError: if pandoc conversion fails.
    """
    docx_path = os.path.abspath(docx_path)
    if not os.path.isfile(docx_path):
        logger.error(f"docx not found: {docx_path}")
        raise FileNotFoundError(docx_path)

    if out_path is None:
        out_path = os.path.splitext(docx_path)[0] + ".md"
    out_path = os.path.abspath(out_path)

    if os.path.isfile(out_path) and not overwrite:
        logger.info(f"convert_docx_to_md: '{out_path}' already exists, skipping.")
        return out_path

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    logger.info(f"convert_docx_to_md: pandoc -s '{docx_path}' -> '{out_path}'")
    # Standalone (-s) Markdown, as in process-3GPP.ipynb (cell 5).
    pypandoc.convert_file(
        docx_path, "markdown", outputfile=out_path, extra_args=["-s"]
    )

    return out_path


def discover_docx(directory: str) -> List[str]:
    """
    Returns the sorted list of .docx file paths directly under `directory`.

    Non-recursive: matches the flat layout of data/inputs/3gpp/. Mirrors the
    discovery style of document_tools.discover_specs().
    """
    if not os.path.isdir(directory):
        logger.warning(f"discover_docx: '{directory}' is not a directory.")
        return []

    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".docx")
    )
    logger.info(f"discover_docx: found {len(files)} .docx file(s) in '{directory}'.")
    return files
