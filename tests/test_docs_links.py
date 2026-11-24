"""The links of the documentation resolve (``tools/docs/check_links.py``): pages and headings
that links name exist, the pages that the code mentions exist, and every page is in the table
of contents and among the pages that ``mopt mcp`` serves.

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
        "check_links", ROOT / "tools/docs/check_links.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_links_of_the_documentation_resolve() -> None:
    found = _tool().problems(ROOT)
    assert not found, "; ".join(found)


def test_the_anchors_are_those_of_myst() -> None:
    tool = _tool()
    assert tool.slug("Variable cell (`optmain -vc`)") == "variable-cell-optmain--vc"
    assert tool.slug("RS-I-RFO") == "rs-i-rfo"
    assert tool.slug("**Fixed** atoms and [links](x.md)") == "fixed-atoms-and-links"
    text = "# Page\n\n## Same\n\n## Same\n\n#### Deep\n\n```\n## Not a heading\n```\n"
    assert tool.anchors(text) == {"page", "same", "same-1"}


def test_broken_links_are_found(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "source"
    source.mkdir(parents=True)
    (source / "a.md").write_text(
        "# A\n\n[b](b.md#there) [gone](b.md#gone) [none](c.md) [self](#a) [no](#no)\n"
        "`[code](c.md)` [web](https://example.org/c.md)\n",
        encoding="utf-8",
    )
    (source / "b.md").write_text("# B\n\n## There\n", encoding="utf-8")
    (source / "index.rst").write_text(".. toctree::\n\n   a\n", encoding="utf-8")
    bin_dir = tmp_path / "crates" / "mopt-rs" / "src" / "bin"
    bin_dir.mkdir(parents=True)
    # Built from parts, so that the checker does not read these lines as mentions of pages.
    pages = "docs" + "/source/"
    (bin_dir / "mopt.rs").write_text(
        f'include_str!("../../../../{pages}a.md"),\n'
        f'include_str!("../../../../{pages}b.md"),\n'
        f"// see {pages}missing.md\n",
        encoding="utf-8",
    )
    found = _tool().problems(tmp_path)
    assert "a.md:3: no heading #gone in b.md" in found
    assert "a.md:3: no page c.md" in found
    assert "a.md:3: no heading #no in a.md" in found
    assert any(f"no page {pages}missing.md" in f for f in found)
    assert "b.md: not in the table of contents (docs/source/index.rst)" in found
    assert len(found) == 5, found
