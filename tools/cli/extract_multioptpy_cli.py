"""Extract the command-line interface of MultiOptPy into a JSON specification.

Reads the argparse definitions of the ten MultiOptPy commands (``interface.py``, the
``Entrypoints`` modules, and ``Utils/rcmc.py``) with the ``ast`` module, without importing or
executing MultiOptPy, and writes one JSON document with every option of every command.

Usage:
    python -I tools/cli/extract_multioptpy_cli.py <multioptpy package dir> <output.json>

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

# Parser-building functions per command, in the order MultiOptPy applies them.
COMMANDS: dict[str, list[tuple[str, str]]] = {
    "optmain": [("interface.py", "call_optimizeparser"), ("interface.py", "parser_for_biasforce")],
    "ieipmain": [("interface.py", "call_ieipparser"), ("interface.py", "parser_for_biasforce")],
    "nebmain": [("interface.py", "call_nebparser"), ("interface.py", "parser_for_biasforce")],
    "mdmain": [("interface.py", "call_mdparser"), ("interface.py", "parser_for_biasforce")],
    "confsearch": [("Entrypoints/conformation_search.py", "run_confsearch")],
    "relaxedscan": [("Entrypoints/relaxed_scan.py", "run_relaxedscan")],
    "orientsearch": [("Entrypoints/orientation_search.py", "run_orientsearch")],
    "run_autots": [("Entrypoints/autots.py", "run_autots")],
    "run_mapper": [("Entrypoints/mapper.py", "run_mapper")],
    "run_rcmc": [
        ("Utils/rcmc.py", "main"),
        ("Utils/rcmc.py", "build_parser"),
        ("Utils/rcmc.py", "parse_args"),
    ],
}


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return {"expr": ast.unparse(node)}


def add_argument_calls(fn: ast.FunctionDef) -> list[dict[str, Any]]:
    """All ``*.add_argument(...)`` calls in ``fn`` (including nested functions and groups)."""
    out: list[dict[str, Any]] = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        names = [literal(a) for a in node.args]
        kwargs = {k.arg: literal(k.value) for k in node.keywords if k.arg is not None}
        if "type" in kwargs and isinstance(kwargs["type"], dict):
            kwargs["type"] = kwargs["type"]["expr"]
        out.append({"names": names, "line": node.lineno, **kwargs})
    out.sort(key=lambda a: a["line"])
    return out


def main() -> None:
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    spec: dict[str, Any] = {"source": "MultiOptPy 1.21.2", "commands": {}}
    for command, sources in COMMANDS.items():
        options: list[dict[str, Any]] = []
        used: list[str] = []
        for rel, fname in sources:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
            fn = find_function(tree, fname)
            if fn is None:
                continue
            calls = add_argument_calls(fn)
            if calls:
                used.append(f"{rel}:{fname}")
                for c in calls:
                    c["origin"] = f"{rel}:{fname}"
                options.extend(calls)
        spec["commands"][command] = {"sources": used, "options": options}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for command, c in spec["commands"].items():
        print(f"{command:13s} {len(c['options']):4d} options from {', '.join(c['sources'])}")


if __name__ == "__main__":
    main()
