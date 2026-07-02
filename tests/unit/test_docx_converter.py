# tests/unit/test_docx_converter.py

"""
Unit tests for tools.docx_converter — no pandoc, network, or LLM required.

convert_docx_to_md is exercised only for its guard behaviour (missing file,
idempotent skip), since running pandoc needs the binary and a real .docx.
"""

import os

import pytest

from openapi_rulesbank.tools.docx_converter import (
    convert_docx_to_md,
    discover_docx,
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
