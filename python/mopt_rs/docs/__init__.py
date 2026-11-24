"""The documentation of mopt-rs, shipped with the package.

The Markdown sources of the documentation (``docs/source`` of the repository) are installed
with the package, so that they can be read where the package is, without the repository and
without a network::

    python -m mopt_rs.docs              # the documents there are
    python -m mopt_rs.docs running      # print one (here: running.md)
    python -m mopt_rs.docs --path       # the directory that holds them

    import mopt_rs.docs
    mopt_rs.docs.names()                # ["architecture", "benchmarks", ...]
    mopt_rs.docs.read("running")        # the text
    mopt_rs.docs.path()                 # pathlib.Path of the directory

The copies are made from ``docs/source`` by ``tools/docs/sync_package_docs.py``;
``tests/test_package_docs.py`` fails when a copy differs from its source.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from pathlib import Path


def path() -> Path:
    """The directory of the documents."""
    return Path(__file__).resolve().parent


def names() -> list[str]:
    """The names of the documents (file names without ``.md``), sorted."""
    return sorted(p.stem for p in path().glob("*.md"))


def read(name: str) -> str:
    """The text of the document ``name`` (with or without ``.md``)."""
    stem = name[:-3] if name.endswith(".md") else name
    file = path() / f"{stem}.md"
    if not file.is_file():
        known = ", ".join(names())
        raise FileNotFoundError(f"no document '{name}' (there are: {known})")
    return file.read_text(encoding="utf-8")
