"""Sphinx configuration of the documentation of mopt-rs.

The documents are the Markdown files of this directory (read with MyST); ``index.rst`` holds
the table of contents. Build with::

    python -m pip install -r docs/requirements.txt
    python -m sphinx -b html docs/source docs/build/html

Read the Docs builds the same from ``.readthedocs.yaml``. The package is not imported, so a
build needs neither the Rust toolchain nor the compiled extension.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
with (_ROOT / "pyproject.toml").open("rb") as _file:
    _PROJECT = tomllib.load(_file)["project"]

project = "mopt-rs"
author = "ss0832"
copyright = "2023-2026, ss0832"  # noqa: A001 (the name Sphinx reads)
release = str(_PROJECT["version"])
version = release

extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns: list[str] = []

# Headings down to the third level get anchors, so that links between the documents
# (``running.md#...``) resolve.
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = f"mopt-rs {release}"
