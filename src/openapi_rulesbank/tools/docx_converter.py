# src/tools/docx_converter.py

"""
Tools for converting 3GPP source documents (.docx) into Markdown and for
extracting the OpenAPI specification from their normative Annex A.

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
    extract_annex_a_yaml(md_text, yaml_type) -> str
        Returns ONLY the Annex A subsection for a single target OpenAPI YAML
        (e.g. yaml_type="ProvMnS" -> the "TS28532_ProvMnS.yaml" block), as the
        target-format context for downstream nodes — not the whole annex.
"""

import os
import re
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


# Any ATX heading (one or more leading '#'), used to bound a subsection: a
# subsection runs from its own heading to the next heading of any level.
_ANY_HEADING_RE = re.compile(r"^#+\s+\S", re.MULTILINE)


def _openapi_doc_heading_re(yaml_type: str) -> re.Pattern:
    """
    Heading that opens the Annex A subsection for a single OpenAPI document.

    Pandoc renders it like:  '## A.1.1 OpenAPI document "TS28532_ProvMnS.yaml"'
    The quotes may be escaped (\\") and the TSxxxxx_ prefix varies, so the match
    keys on the heading text plus the target type before '.yaml'. yaml_type is
    matched case-insensitively (e.g. "ProvMnS").
    """
    return re.compile(
        r'^#+\s+A\.[0-9.]+\s+OpenAPI\s+document\b[^\n]*?'
        + re.escape(yaml_type)
        + r'\.yaml',
        re.IGNORECASE | re.MULTILINE,
    )


def extract_annex_a_yaml(md_text: str, yaml_type: str) -> str:
    """
    Extracts ONLY the Annex A subsection for a single target OpenAPI document.

    Annex A bundles several OpenAPI YAML documents, each under its own heading
    (e.g. '## A.1.1 OpenAPI document "TS28532_ProvMnS.yaml"'). This returns just
    the subsection for `yaml_type` — from that heading up to the next heading of
    any level — so downstream nodes only see the YAML the rules must build.

    The embedded YAML (between '<CODE BEGINS>' / '<CODE ENDS>') loses its
    indentation and gains escapes after the pandoc conversion, so it is NOT valid
    YAML; it is returned verbatim as a target-format context string.

    Args:
        md_text:   the full converted Markdown of the 3GPP spec.
        yaml_type: the OpenAPI document type to extract, e.g. "ProvMnS",
                   "FaultMnS", "PerfMnS" (case-insensitive, the part before
                   '.yaml' in the heading).

    Returns:
        The target subsection text, or "" if no matching heading is found.
    """
    heading_re = _openapi_doc_heading_re(yaml_type)
    start = heading_re.search(md_text)
    if start is None:
        logger.warning(
            f"extract_annex_a_yaml: no Annex A subsection for '{yaml_type}'."
        )
        return ""

    rest = md_text[start.start():]
    # Bound by the next heading of any level, searched AFTER this heading line.
    nxt = _ANY_HEADING_RE.search(rest, pos=1)
    subsection = rest[: nxt.start()] if nxt else rest

    logger.info(
        f"extract_annex_a_yaml: extracted {len(subsection)} chars "
        f"for '{yaml_type}'."
    )
    return subsection.strip()
