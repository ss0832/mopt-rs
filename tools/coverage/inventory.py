"""Inventory of MultiOptPy's features, for checking that mopt-rs leaves none out.

Usage:

    python -I tools/coverage/inventory.py <MultiOptPy root> <inventory.json>

Every module, class, function, and method of ``multioptpy/`` is listed (read with ``ast``,
never executed), together with every command-line option of ``interface.py``. The mapping of
these items to mopt-rs lives in ``tools/coverage/mapping.json``; ``tests/test_coverage.py``
fails when an item has no mapping.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import ast
import hashlib
import json
import sys
from pathlib import Path

# Modules left out of the inventory on purpose (wrappers of engines that mopt-rs does not
# name), identified by the SHA-256 of their path relative to the MultiOptPy root; such engines
# remain usable through the generic `ase:module:Class` calculator.
EXCLUDED_MODULES = {
    "d67bc62125b74802094827ed5e57edd200b72c2022e9045d5d19ba9510f133fc",
}


def items_of(path: Path, rel: str) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    out = [{"id": rel, "kind": "module"}]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.append({"id": f"{rel}:{node.name}", "kind": "class"})
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append({"id": f"{rel}:{node.name}.{sub.name}", "kind": "method"})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append({"id": f"{rel}:{node.name}", "kind": "function"})
    return out


def cli_options(interface: Path) -> list[dict]:
    tree = ast.parse(interface.read_text(encoding="utf-8", errors="replace"))
    out: list[dict] = []
    for func in tree.body:
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
                out.append({"id": f"cli:{func.name}:{flags[-1]}", "kind": "option"})
    return out


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    pkg = root / "multioptpy"
    items: list[dict] = []
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in rel:
            continue
        if hashlib.sha256(rel.encode()).hexdigest() in EXCLUDED_MODULES:
            continue
        items += items_of(path, rel)
    items += cli_options(pkg / "interface.py")
    seen = set()
    unique = []
    for it in items:
        if it["id"] not in seen:
            seen.add(it["id"])
            unique.append(it)
    out.write_text(
        json.dumps({"source": "MultiOptPy 1.21.2", "items": unique}, indent=0), encoding="utf-8"
    )
    kinds: dict[str, int] = {}
    for it in unique:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print(f"wrote {out}: {len(unique)} items {kinds}")


if __name__ == "__main__":
    main()
