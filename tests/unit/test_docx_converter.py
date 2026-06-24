# tests/unit/test_docx_converter.py

"""
Unit tests for tools.docx_converter — no pandoc, network, or LLM required.

convert_docx_to_md is exercised only for its guard behaviour (missing file,
idempotent skip), since running pandoc needs the binary and a real .docx.
extract_annex_a_yaml is tested against the real converted spec
(data/inputs/3gpp/md/28532-i20.md), whose Annex A bundles several OpenAPI
documents (ProvMnS, PerfMnS, ...). The test is skipped if that file has not been
generated yet (run scripts/convert_docx_inputs.py first).
"""

import os

import pytest

from openapi_rulesbank.config.paths import INPUTS_3GPP_MD_DIR
from openapi_rulesbank.tools.docx_converter import (
    convert_docx_to_md,
    discover_docx,
    extract_annex_a_yaml,
)

# Real converted spec used by the extract_annex_a_yaml tests below.
_REAL_MD = os.path.join(INPUTS_3GPP_MD_DIR, "28532-i20.md")
_real_md_missing = pytest.mark.skipif(
    not os.path.isfile(_REAL_MD),
    reason=f"{_REAL_MD} not found — run scripts/convert_docx_inputs.py first.",
)


def test_convert_docx_to_md_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_docx_to_md(str(tmp_path / "nope.docx"))


def test_convert_docx_to_md_idempotent_skip(tmp_path):
    # A pre-existing .md with overwrite=False must be returned without pandoc.
    docx = tmp_path / "spec.docx"
    docx.write_bytes(b"not a real docx")  # existence is enough for the skip path
    md = tmp_path / "spec.md"
    md.write_text("already converted")

    out = convert_docx_to_md(str(docx), str(md), overwrite=False)
    assert out == str(md)
    assert md.read_text() == "already converted"  # untouched


def test_discover_docx(tmp_path):
    (tmp_path / "a.docx").write_bytes(b"x")
    (tmp_path / "b.DOCX").write_bytes(b"x")
    (tmp_path / "ignore.md").write_text("x")
    found = discover_docx(str(tmp_path))
    assert [os.path.basename(p) for p in found] == ["a.docx", "b.DOCX"]


def test_discover_docx_non_directory(tmp_path):
    assert discover_docx(str(tmp_path / "missing")) == []


@_real_md_missing
def test_extract_annex_a_yaml_selects_only_target():
    raw = open(_REAL_MD, encoding="utf-8").read()
    prov = extract_annex_a_yaml(raw, "ProvMnS")
    # The ProvMnS subsection: its heading, the embedded YAML, and CODE markers.
    assert "ProvMnS.yaml" in prov
    assert "openapi: 3.0.1" in prov
    assert "CODE BEGINS" in prov and "CODE ENDS" in prov
    # Only the ProvMnS subsection — other documents' headings must be excluded.
    assert "PerfMnS.yaml" not in prov
    assert "HeartbeatNtf.yaml" not in prov


@_real_md_missing
def test_extract_annex_a_yaml_other_type():
    raw = open(_REAL_MD, encoding="utf-8").read()
    perf = extract_annex_a_yaml(raw, "PerfMnS")
    assert "PerfMnS.yaml" in perf
    assert "ProvMnS.yaml" not in perf


@_real_md_missing
def test_extract_annex_a_yaml_case_insensitive():
    raw = open(_REAL_MD, encoding="utf-8").read()
    assert "ProvMnS.yaml" in extract_annex_a_yaml(raw, "provmns")


@_real_md_missing
def test_extract_annex_a_yaml_absent_returns_empty():
    raw = open(_REAL_MD, encoding="utf-8").read()
    assert extract_annex_a_yaml(raw, "DoesNotExist") == ""
