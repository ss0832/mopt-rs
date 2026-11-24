"""The documentation is shipped with the package and agrees with its sources.

``python/mopt_rs/docs/*.md`` are the copies of ``docs/source/*.md`` that a wheel installs
(``tools/docs/sync_package_docs.py`` makes them); ``mopt_rs.docs`` finds and prints them.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sync_package_docs", ROOT / "tools/docs/sync_package_docs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_copies_in_the_package_are_the_sources() -> None:
    found = _tool().differences(ROOT)
    assert not found, (
        "the documentation in the package differs from docs/source; run "
        "`python -I tools/docs/sync_package_docs.py .`: " + "; ".join(found)
    )


def test_the_documents_are_found_from_the_package() -> None:
    import mopt_rs.docs

    names = mopt_rs.docs.names()
    assert "running" in names and "migration" in names
    assert mopt_rs.docs.path().is_dir()
    text = mopt_rs.docs.read("running.md")
    assert "ONIOM" in text
    assert text == (ROOT / "docs/source/running.md").read_text(encoding="utf-8")
    try:
        mopt_rs.docs.read("no_such_document")
    except FileNotFoundError as error:
        assert "running" in str(error)
    else:
        raise AssertionError("a missing document must be an error")
