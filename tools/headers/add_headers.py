"""Add the line "This file is part of mopt-rs." (or mopt-py) to the header of every source file.

Files of the Rust library, its tests, and the development tools belong to mopt-rs; the Python
package, the binding crate, and the Python tests belong to mopt-py. The line goes directly
above the copyright line, or at the end of the leading doc comment / module docstring when a
file has no copyright line. Files that already have it are left alone, so the script can be
run again after new files were added. Generated files get the line from their generators
(``tools/cli/gen_cli.py``, ``tools/params/gen_*.py``), which this script also updates.

Usage:
    python -I tools/headers/add_headers.py <repository root> [--check]

``--check`` changes nothing and lists the files without the line (exit code 1 if any).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import sys
from pathlib import Path

MARK = "This file is part of mopt-"
COPYRIGHT = "Copyright (C) 2023-2026 ss0832"
RS = "This file is part of mopt-rs."
PY = "This file is part of mopt-py (the Python interface of mopt-rs)."
SKIP_DIRS = {"target", "reference", "__pycache__", ".git", ".venv", ".venv-ref", "build", "dist"}
ROOTS = ["crates", "python", "tests", "tools"]
SUFFIXES = {".rs", ".py", ".pyi", ".sh"}


def sentence(rel: str) -> str:
    """The line for the file at the repository-relative path ``rel``."""
    if rel.startswith(("crates/mopt-py/", "python/", "tests/")):
        return PY
    return RS


def source_files(root: Path) -> list[Path]:
    """Every source file of the project under ``root``."""
    out = []
    for top in ROOTS:
        for path in sorted((root / top).rglob("*")):
            if path.suffix in SUFFIXES and not SKIP_DIRS.intersection(path.parts):
                out.append(path)
    return out


def _into_rust_header(lines: list[str], text: str) -> list[str]:
    last = -1
    for k, line in enumerate(lines):
        if line.startswith("//!"):
            last = k
        elif line.strip() and last >= 0 or line.strip():
            break
    if last < 0:
        return [f"//! {text}", "", *lines]
    return [*lines[: last + 1], "//!", f"//! {text}", *lines[last + 1 :]]


def _into_python_docstring(source: str, text: str) -> str:
    stripped = source.lstrip()
    lead = source[: len(source) - len(stripped)]
    if not stripped.startswith('"""'):
        return f'"""{text}"""\n\n{source}' if source.strip() else f'"""{text}"""\n'
    end = stripped.index('"""', 3)
    body = stripped[3:end].rstrip()
    return f'{lead}"""{body}\n\n{text}\n"""{stripped[end + 3 :]}'


def with_header(source: str, suffix: str, text: str) -> str:
    """``source`` with the line added."""
    lines = source.split("\n")
    for k, line in enumerate(lines):
        if COPYRIGHT in line and '"' not in line.split(COPYRIGHT)[0]:
            prefix = line.split(COPYRIGHT)[0]
            return "\n".join([*lines[:k], f"{prefix}{text}", *lines[k:]])
    if suffix == ".rs":
        return "\n".join(_into_rust_header(lines, text))
    if suffix == ".py":
        return _into_python_docstring(source, text)
    # Stubs and shell scripts: a comment after a shebang, else on the first line.
    at = 1 if lines and lines[0].startswith("#!") else 0
    return "\n".join([*lines[:at], f"# {text}", *lines[at:]])


def has_line(source: str, text: str) -> bool:
    """Whether ``source`` carries ``text`` itself (not only in a header it generates)."""
    return any(
        text in line and not line.lstrip().startswith('"//!') for line in source.splitlines()
    )


def update_generator(source: str) -> str:
    """A generator whose emitted header gets the line above its copyright line."""
    key = f'"//! {COPYRIGHT}",'
    if key not in source or f'"//! {RS}",' in source:
        return source
    at = source.index(key)
    indent = source[source.rfind("\n", 0, at) + 1 : at]
    return f'{source[:at]}"//! {RS}",\n{indent}{source[at:]}'


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    check = "--check" in sys.argv[2:]
    missing = []
    for path in source_files(root):
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        updated = source
        if rel.startswith("tools/") and path.suffix == ".py":
            updated = update_generator(updated)
        if not has_line(updated, sentence(rel)):
            updated = with_header(updated, path.suffix, sentence(rel))
        if updated != source:
            missing.append(rel)
            if not check:
                path.write_text(updated, encoding="utf-8", newline="\n")
    for rel in missing:
        print(("missing: " if check else "updated: ") + rel)
    print(f"{len(missing)} file(s) {'without the line' if check else 'updated'}")
    return 1 if check and missing else 0


if __name__ == "__main__":
    sys.exit(main())
