"""The type stubs ``python/mopt_rs/_core.pyi`` name exactly what the compiled module has: every
public function, class, and constant of ``mopt_rs._core`` has a stub and every stub is in the
module (type aliases of the stubs aside), and the public methods and attributes of each class
agree.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import ast
from pathlib import Path

from mopt_rs import _core

STUB = Path(__file__).resolve().parents[1] / "python/mopt_rs/_core.pyi"


def _stub() -> tuple[set[str], set[str], dict[str, set[str]]]:
    """Names of the stubs, names of their type aliases, and the members of each class."""
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    names: set[str] = set()
    aliases: set[str] = set()
    members: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            found: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    found.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    found.add(item.target.id)
            members[node.name] = {m for m in found if not m.startswith("_")}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names, aliases, members


def test_every_name_of_the_module_has_a_stub_and_back() -> None:
    names, aliases, _ = _stub()
    module = {n for n in dir(_core) if not n.startswith("_")} | {"__version__"}
    stubbed = {n for n in names if not n.startswith("_")} | ({"__version__"} & names)
    assert not module - stubbed, f"add stubs to _core.pyi for {sorted(module - stubbed)}"
    assert not stubbed - module, f"_core.pyi has stubs of what is not there: {sorted(stubbed - module)}"
    # Type aliases exist only in the stubs.
    assert not (aliases - {n for n in aliases if n.startswith("_")}) & module


def test_the_members_of_the_classes_agree() -> None:
    _, _, members = _stub()
    wrong = {}
    for name, stubbed in members.items():
        cls = getattr(_core, name)
        public = {m for m in dir(cls) if not m.startswith("_")}
        if public != stubbed:
            wrong[name] = (sorted(public - stubbed), sorted(stubbed - public))
    assert not wrong, f"(in the module only, in the stubs only) per class: {wrong}"
