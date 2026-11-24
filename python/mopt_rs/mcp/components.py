"""The synchronous tools for components: the commands, the help and the schemas of a command,
the calculators, a single point of one structure, and the tools that compute from structures
(model Hessians, normal modes, coordinate systems, first images, superposition, bond graphs,
dispersion, Hessian updates, units, elements).

The help text comes from the command itself (``COMMAND -h`` in a child process: the compiled
core prints it to the standard output, which belongs to the protocol in the server). Single
points use the built-in potentials through ``mopt_rs._core.evaluate_native`` and the other
engines through ``mopt_rs.calculators.create``. The tools that compute from structures are
those of the Rust server, called through ``mopt_rs._core.mcp_component`` with the texts of the
structures, which are read here through the sandbox.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from mopt_rs._core import ANGSTROM_TO_BOHR as BOHR_PER_ANGSTROM
from mopt_rs.mcp.paths import Sandbox, ToolError, path_like_parts
from mopt_rs.mcp.policy import Policy
from mopt_rs.mcp.tables import (
    Command,
    command_tool,
    compute_tools,
    matching_options,
    option_schema,
    positional,
)

_STRUCTURE_LIMIT = 16 << 20
#: The most atoms of a single point, which the server computes while it waits.
SINGLE_POINT_ATOMS = 2000
#: The most atoms of a single point with the Hessian.
SINGLE_POINT_HESSIAN_ATOMS = 300


def list_commands(commands: list[Command]) -> dict[str, Any]:
    """The commands, whether they are implemented, and their tools."""
    out = []
    for c in commands:
        given = positional(c)
        out.append(
            {
                "command": c["name"],
                "implemented": c["implemented"],
                "feature": c["feature"],
                "options": len(c["options"]),
                "implemented_options": sum(1 for o in c["options"] if o["implemented"]),
                "input": f"{given['name']} ({given['help']})",
                "tool": c["name"] if c["implemented"] else None,
            }
        )
    return {"commands": out}


def command_help(
    commands: list[Command], args: dict[str, Any], program: list[str], cwd: Path
) -> dict[str, Any]:
    """The help of a command (the arguments of ``command_help``): its ``--help`` text, or the
    options that ``options`` names with their schemas, and with ``schema`` the full input
    schema of its tool."""
    name = args.get("command")
    if not isinstance(name, str):
        raise ToolError("give the command")
    command = next((c for c in commands if c["name"] == name), None)
    if command is None:
        raise ToolError(f"no command '{name}'")
    out: dict[str, Any] = {"command": name}
    queries = args.get("options")
    if queries is None:
        run = subprocess.run(
            [*program, name, "-h"],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        out["help"] = run.stdout.strip() or run.stderr.strip()
    else:
        if (
            not isinstance(queries, list)
            or len(queries) > 100
            or not all(isinstance(q, str) for q in queries)
        ):
            raise ToolError("options is a list of at most 100 option names")
        matches = []
        for q in queries:
            entries = []
            for o in matching_options(command, q):
                entry: dict[str, Any] = {
                    "key": o["key"],
                    "flags": o["flags"],
                    "help": o["help"].strip(),
                    "default": o["default"],
                    "implemented": o["implemented"],
                }
                if o["implemented"]:
                    entry["schema"] = option_schema(o)
                else:
                    entry["feature"] = o["feature"]
                entries.append(entry)
            matches.append({"query": q, "options": entries})
        out["matches"] = matches
    if args.get("schema") is True:
        out["inputSchema"] = command_tool(command, True)["inputSchema"]
    return out


def read_structure(sandbox: Sandbox, path: str) -> str:
    """The text of the structure file ``path`` under the root, read through the sandbox."""
    opened, meta = sandbox.open_regular(sandbox.existing(path))
    with opened:
        if meta.st_size > _STRUCTURE_LIMIT:
            raise ToolError(f"{path} has more than {_STRUCTURE_LIMIT} bytes")
        return opened.read(_STRUCTURE_LIMIT).decode("utf-8", errors="replace")


def structure_texts(sandbox: Sandbox, name: str, args: dict[str, Any]) -> dict[str, str]:
    """The structure texts of the component tool ``name``: each structure argument as a file
    under the root or as its text, never both."""
    _, structures = compute_tools()
    out: dict[str, str] = {}
    for file_key, text_key in structures.get(name, []):
        for key in (file_key, text_key):
            if args.get(key) is not None and not isinstance(args.get(key), str):
                raise ToolError(f"{key} is a string")
        path, text = args.get(file_key), args.get(text_key)
        if path is not None and text is not None:
            raise ToolError(f"give {file_key} or {text_key}, not both")
        if path is not None:
            out[file_key] = read_structure(sandbox, path)
        elif text is not None:
            out[file_key] = text
    return out


def computed(sandbox: Sandbox, policy: Policy, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """A tool that computes from structures (``mopt_rs._core.mcp_component``); for
    ``vibrations`` with a calculator and no Hessian, the Hessian of the calculator first."""
    from mopt_rs import _core

    texts = structure_texts(sandbox, name, args)
    given = dict(args)
    if name == "vibrations" and "hessian" not in args:
        if "calculator" not in args:
            raise ToolError("give the hessian, or a calculator to compute it")
        point = {k: v for k, v in args.items() if k not in ("energy", "temperature", "pressure")}
        point["hessian"] = True
        evaluated = single_point(sandbox, policy, point)
        given["hessian"] = evaluated["hessian_hartree_per_bohr2"]
        given["energy"] = evaluated["energy_hartree"]
    try:
        out: dict[str, Any] = json.loads(
            _core.mcp_component(name, json.dumps(given), json.dumps(texts))
        )
    except ValueError as e:
        raise ToolError(str(e)) from None
    return out


def list_calculators() -> dict[str, Any]:
    """The calculators of the package."""
    from mopt_rs import _core
    from mopt_rs.calculators.registry import available

    return {
        "calculators": available(),
        "built_in": list(_core.native_calculators()),
        "objectives": list(_core.objective_names()),
        "note": "every engine of mopt_rs.calculators; programs outside Python (ORCA, Gaussian, "
        "...) need their paths in software_path.conf, as on the command line",
    }


def _xyz(text: str) -> tuple[list[str], list[list[float]]]:
    lines = [x for x in text.splitlines()]
    symbols: list[str] = []
    rows: list[list[float]] = []
    start = 2 if lines and lines[0].strip().isdigit() else 0
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            xyz = [float(v) for v in parts[1:4]]
        except ValueError:
            continue
        symbols.append(parts[0])
        rows.append(xyz)
    if not symbols:
        raise ToolError("the xyz text holds no atoms")
    return symbols, rows


def single_point(sandbox: Sandbox, policy: Policy, args: dict[str, Any]) -> dict[str, Any]:
    """Energy and gradient of one structure."""
    from mopt_rs import _core

    calculator = args.get("calculator")
    if not isinstance(calculator, str):
        raise ToolError("give the calculator (see list_calculators)")
    policy.check_calculator(calculator)
    structure, xyz = args.get("structure"), args.get("xyz")
    if isinstance(structure, str) and xyz is None:
        text = read_structure(sandbox, structure)
    elif isinstance(xyz, str) and structure is None:
        text = xyz
    else:
        raise ToolError("give the structure as one of structure (a file) or xyz (its text)")
    charge = args.get("charge", 0)
    multiplicity = args.get("multiplicity", 1)
    if not isinstance(charge, int) or not isinstance(multiplicity, int) or multiplicity < 1:
        raise ToolError("charge is an integer and multiplicity an integer of at least 1")
    settings_arg = args.get("settings") or {}
    if not isinstance(settings_arg, dict):
        raise ToolError("settings is an object of KEY: VALUE")
    settings = [
        (str(k), str(v).lower() if isinstance(v, bool) else str(v)) for k, v in settings_arg.items()
    ]
    for key, value in settings:
        token = f"{key}={value}"
        policy.check_option("calc_option", [token])
        for part in path_like_parts(token):
            sandbox.check_value_part(sandbox.root, part)
    need_hessian = bool(args.get("hessian", False))
    symbols, rows = _xyz(text)
    limit = SINGLE_POINT_HESSIAN_ATOMS if need_hessian else SINGLE_POINT_ATOMS
    if len(symbols) > limit:
        with_hessian = " with the Hessian" if need_hessian else ""
        raise ToolError(
            f"a single point takes at most {limit} atoms{with_hessian}; run larger structures "
            "as jobs (freqmain for Hessians)"
        )
    flat = [v * BOHR_PER_ANGSTROM for row in rows for v in row]
    warnings: list[str] = []
    try:
        if calculator.lower() in _core.native_calculators():
            energy, gradient, hessian = _core.evaluate_native(
                calculator.lower(),
                symbols,
                flat,
                charge,
                multiplicity,
                need_hessian,
                settings=settings or None,
            )
            gradient_list = list(gradient)
            hessian_list = None if hessian is None else list(hessian)
        else:
            import numpy as np

            from mopt_rs.calculators.registry import (
                CalculatorOptions,
                create,
                parse_option_value,
            )
            from mopt_rs.optimize import Task

            extra = {k: parse_option_value(v) for k, v in settings}
            engine = create(calculator, CalculatorOptions(extra=extra))
            task = Task(
                label="single_point",
                calculator=calculator,
                symbols=symbols,
                positions_bohr=np.asarray(flat).reshape(-1, 3),
                charge=charge,
                multiplicity=multiplicity,
                need_hessian=need_hessian,
            )
            result = engine.evaluate(task)
            energy = float(result.energy)
            gradient_list = [float(g) for g in np.asarray(result.gradient).ravel()]
            hessian_list = (
                None
                if result.hessian is None
                else [float(h) for h in np.asarray(result.hessian).ravel()]
            )
            warnings = list(result.notes)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"{calculator}: {e}") from None
    n = len(symbols)
    out: dict[str, Any] = {
        "calculator": calculator,
        "elements": symbols,
        "energy_hartree": energy,
        "gradient_hartree_per_bohr": [gradient_list[3 * i : 3 * i + 3] for i in range(n)],
        "max_abs_gradient": max((abs(g) for g in gradient_list), default=0.0),
        "rms_gradient": math.sqrt(sum(g * g for g in gradient_list) / len(gradient_list))
        if gradient_list
        else 0.0,
        "warnings": warnings,
    }
    if hessian_list is not None:
        out["hessian_hartree_per_bohr2"] = hessian_list
    return out
