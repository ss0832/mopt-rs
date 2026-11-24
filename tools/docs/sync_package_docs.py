"""Copy the documentation into the Python package, so that pip installs it.

The Markdown sources live in ``docs/source``; a wheel holds only what is inside the package
directory, so every ``*.md`` there is copied to ``python/mopt_rs/docs/``, where
``mopt_rs.docs`` finds it (``python -m mopt_rs.docs``). Run after a change of the
documentation::

    python -I tools/docs/sync_package_docs.py <repository root> [--check]

``--check`` writes nothing and exits with 1 when a copy is missing, differs from its source,
or has no source any more (``tests/test_package_docs.py`` does the same).

This file is part of mopt-py (the Python interface of mopt-rs).
This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def differences(root: Path) -> list[str]:
    """What differs between the sources and the copies (empty when they agree)."""
    source = root / "docs" / "source"
    package = root / "python" / "mopt_rs" / "docs"
    out = []
    names = {p.name for p in source.glob("*.md")}
    for name in sorted(names):
        copy = package / name
        if not copy.is_file():
            out.append(f"{name}: not in the package")
        elif copy.read_bytes() != (source / name).read_bytes():
            out.append(f"{name}: differs from docs/source")
    for copy in sorted(package.glob("*.md")):
        if copy.name not in names:
            out.append(f"{copy.name}: no longer in docs/source")
    return out


def sync(root: Path) -> list[str]:
    """Make the copies agree with the sources; returns the names written or removed."""
    source = root / "docs" / "source"
    package = root / "python" / "mopt_rs" / "docs"
    package.mkdir(parents=True, exist_ok=True)
    changed = []
    names = {p.name for p in source.glob("*.md")}
    for name in sorted(names):
        data = (source / name).read_bytes()
        copy = package / name
        if not copy.is_file() or copy.read_bytes() != data:
            copy.write_bytes(data)
            changed.append(name)
    for copy in sorted(package.glob("*.md")):
        if copy.name not in names:
            copy.unlink()
            changed.append(copy.name)
    return changed


def is_repository_root(root: Path) -> bool:
    """Whether ``root`` is the top of a mopt-rs checkout (sources and package both there)."""
    return (
        (root / "Cargo.toml").is_file()
        and (root / "docs" / "source").is_dir()
        and (root / "python" / "mopt_rs" / "__init__.py").is_file()
    )


def main(argv: list[str]) -> int:
    """Exit code of the tool."""
    parser = argparse.ArgumentParser(
        prog="sync_package_docs.py",
        description="Copy docs/source/*.md into python/mopt_rs/docs/ so that pip installs them.",
    )
    parser.add_argument(
        "root", type=Path, help="the repository root (holds Cargo.toml and docs/source)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit with 1 when a copy is missing, differs, or has no source",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not is_repository_root(root):
        print(
            f"sync_package_docs.py: {root} is not a mopt-rs repository root "
            "(expected Cargo.toml, docs/source/ and python/mopt_rs/ there)",
            file=sys.stderr,
        )
        return 2
    if args.check:
        found = differences(root)
        for line in found:
            print(line)
        return 1 if found else 0
    for name in sync(root):
        print(f"updated {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))