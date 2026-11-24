"""The documentation builds as a web site (Sphinx with MyST; Read the Docs and GitHub Pages).

``docs/source/conf.py`` and ``docs/source/index.rst`` make the Markdown documents a Sphinx
project; ``.readthedocs.yaml`` builds it on Read the Docs and
``.github/workflows/docs.yml`` on GitHub Pages. The checks that need no Sphinx (every
document is in the table of contents, the settings agree with each other) always run; the
build itself runs where Sphinx is installed.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "source"


def _listed() -> list[str]:
    """The documents named in the tables of contents of the index, in order."""
    out: list[str] = []
    in_toctree = False
    for line in (SOURCE / "index.rst").read_text(encoding="utf-8").splitlines():
        if line.startswith(".. toctree::"):
            in_toctree = True
        elif in_toctree and line and not line.startswith(" "):
            in_toctree = False
        elif in_toctree and re.fullmatch(r"\s+[A-Za-z0-9_]+", line):
            out.append(line.strip())
    return out


def test_every_document_is_in_the_table_of_contents_once() -> None:
    listed = _listed()
    documents = sorted(p.stem for p in SOURCE.glob("*.md"))
    assert sorted(listed) == documents, (
        "docs/source/index.rst must name every docs/source/*.md once: "
        f"missing {sorted(set(documents) - set(listed))}, "
        f"without a file {sorted(set(listed) - set(documents))}, "
        f"repeated {sorted({n for n in listed if listed.count(n) > 1})}"
    )


def test_the_settings_of_the_two_sites_agree() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wanted = sorted(project["optional-dependencies"]["docs"])
    lines = (ROOT / "docs" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert sorted(x for x in lines if x and not x.startswith("#")) == wanted

    rtd = (ROOT / ".readthedocs.yaml").read_text(encoding="utf-8")
    assert "configuration: docs/source/conf.py" in rtd
    assert "requirements: docs/requirements.txt" in rtd

    pages = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    assert "docs/requirements.txt" in pages
    assert "docs/source" in pages and "deploy-pages" in pages

    conf = (SOURCE / "conf.py").read_text(encoding="utf-8")
    assert "myst_parser" in conf and "sphinx_rtd_theme" in conf


def test_the_documents_keep_their_placeholders_as_text() -> None:
    # Outside code, "<stem>" is a tag to a Markdown reader and vanishes from the page.
    found = []
    for file in sorted(SOURCE.glob("*.md")):
        text = file.read_text(encoding="utf-8")
        text = re.sub(r"(?ms)^```.*?^```", "", text)
        text = re.sub(r"(?s)`[^`]*`", "", text)
        for tag in re.findall(r"(?<!\\)<[A-Za-z_][^<>\n]*>", text):
            found.append(f"{file.name}: {tag}")
    assert not found, "placeholders outside code spans: " + ", ".join(sorted(set(found)))


@pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("sphinx", "myst_parser", "sphinx_rtd_theme")),
    reason="Sphinx, MyST and the Read the Docs theme are not installed",
)
def test_the_site_builds_without_warnings(tmp_path: Path) -> None:
    run = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-W", "--keep-going", "-q"]
        + [str(SOURCE), str(tmp_path / "html")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-4000:]
    pages = {p.stem for p in (tmp_path / "html").glob("*.html")}
    assert {p.stem for p in SOURCE.glob("*.md")} <= pages
    reference = (tmp_path / "html" / "cli_reference.html").read_text(encoding="utf-8")
    assert "&lt;stem&gt;_Sample_&lt;k&gt;_OPT" in reference
