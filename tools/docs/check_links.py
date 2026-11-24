"""Check the links of the documentation: every link of ``docs/source/*.md`` to a page of the
documentation names a page that exists and, with ``#anchor``, a heading of that page; every
mention of ``docs/source/<page>.md`` in the code, the tests, and the tools names a page that
exists; and every page is in the table of contents (``index.rst``) and in the pages that
``mopt mcp`` serves (``crates/mopt-rs/src/bin/mopt.rs``).

Anchors follow MyST (``myst_heading_anchors = 3`` in ``conf.py``): headings of levels 1 to 3,
the slug of markdown-it's anchors plugin (lower case, spaces to hyphens, characters other than
word characters, hyphens, and spaces dropped; a repeated slug gets ``-1``, ``-2``, ...).

    python -I tools/docs/check_links.py <repository root>

Exits with 1 and lists what is wrong (``tests/test_docs_links.py`` does the same).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
from pathlib import Path

HEADING_LEVELS = 3
_CLEAN = re.compile(r"[^\w一-鿿\- ]")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LINK = re.compile(r"(?<!!)\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)\)")
_PAGE_MENTION = re.compile(r"docs/source/([A-Za-z0-9_]+)\.md(#[A-Za-z0-9_\-]+)?")
_CODE_FILES = ("*.rs", "*.py", "*.pyi", "*.json", "*.toml", "*.md", "*.rst", "*.sh")


def _plain(title: str) -> str:
    """The text of a heading as MyST renders it: inline code, links, and emphasis unwrapped."""
    text = re.sub(r"`([^`]*)`", r"\1", title)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    return text


def slug(title: str) -> str:
    """The anchor of a heading (markdown-it's anchors plugin, as MyST uses it)."""
    return _CLEAN.sub("", _plain(title).strip().lower().replace(" ", "-"))


def _outside_code(text: str) -> list[tuple[int, str]]:
    """The lines of a page outside fenced code blocks, with their numbers (1-based)."""
    out = []
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
            continue
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        out.append((number, line))
    return out


def anchors(text: str) -> set[str]:
    """The anchors of the headings of a page (levels 1 to ``HEADING_LEVELS``)."""
    seen: dict[str, int] = {}
    out = set()
    for _, line in _outside_code(text):
        match = _HEADING.match(line)
        if not match or len(match.group(1)) > HEADING_LEVELS:
            continue
        base = slug(match.group(2))
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.add(base if count == 0 else f"{base}-{count}")
    return out


def _links(text: str) -> list[tuple[int, str]]:
    """The targets of the links of a page outside code (fenced and inline), with lines."""
    out = []
    for number, line in _outside_code(text):
        line = re.sub(r"`[^`]*`", "", line)
        for match in _LINK.finditer(line):
            out.append((number, match.group(1)))
    return out


def _code_files(root: Path) -> list[str]:
    """The files of the repository that may mention a page (relative, sorted): the sources,
    tests, tools, and documents, not build output, worktrees, or the copies in the package."""
    skip_dirs = {"target", ".git", "node_modules", "build", "__pycache__", ".claude", "dist"}
    copies = "python/mopt_rs/docs"
    out = []
    for directory, dirs, files in os.walk(root):
        rel_dir = Path(directory).relative_to(root).as_posix()
        dirs[:] = [
            d
            for d in dirs
            if d not in skip_dirs
            and ":" not in d
            and (f"{rel_dir}/{d}" if rel_dir != "." else d) != copies
        ]
        for name in files:
            if any(fnmatch.fnmatch(name, p) for p in _CODE_FILES):
                out.append(name if rel_dir == "." else f"{rel_dir}/{name}")
    return sorted(out)


def problems(root: Path) -> list[str]:
    """What is wrong with the links of the documentation (empty when nothing is)."""
    source = root / "docs" / "source"
    pages = {p.stem: p.read_text(encoding="utf-8") for p in sorted(source.glob("*.md"))}
    known = {name: anchors(text) for name, text in pages.items()}
    out = []
    for name, text in pages.items():
        for number, target in _links(text):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                continue
            path, _, anchor = target.partition("#")
            if path and not path.endswith(".md"):
                continue
            page = Path(path).stem if path else name
            if path and (Path(path).parent != Path(".") or page not in known):
                out.append(f"{name}.md:{number}: no page {path}")
                continue
            if anchor and anchor not in known[page]:
                out.append(f"{name}.md:{number}: no heading #{anchor} in {page}.md")
    # Mentions of pages in the code, the tests, the tools, and the documents.
    for rel in _code_files(root):
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for match in _PAGE_MENTION.finditer(line):
                page, anchor = match.group(1), match.group(2)
                if page not in known:
                    out.append(f"{rel}:{number}: no page docs/source/{page}.md")
                elif anchor and anchor[1:] not in known[page]:
                    out.append(f"{rel}:{number}: no heading {anchor} in {page}.md")
    # Every page in the table of contents and among the pages the MCP server serves.
    index = (source / "index.rst").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\s{3}([A-Za-z0-9_]+)\s*$", index, flags=re.MULTILINE))
    served = set(
        re.findall(
            r'include_str!\("\.\./\.\./\.\./\.\./docs/source/([A-Za-z0-9_]+)\.md"\)',
            (root / "crates/mopt-rs/src/bin/mopt.rs").read_text(encoding="utf-8"),
        )
    )
    for name in pages:
        if name not in listed:
            out.append(f"{name}.md: not in the table of contents (docs/source/index.rst)")
        if name not in served:
            out.append(f"{name}.md: not among the pages of crates/mopt-rs/src/bin/mopt.rs")
    for name in sorted(listed - set(pages)):
        out.append(f"index.rst: no page {name}.md")
    for name in sorted(served - set(pages)):
        out.append(f"crates/mopt-rs/src/bin/mopt.rs: no page {name}.md")
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path.cwd()
    found = problems(root)
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
