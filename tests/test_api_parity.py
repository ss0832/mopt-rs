"""What Rust offers, Python offers too, and the other way round.

``tools/api/parity.json`` pairs the modules and items of the Rust library with the Python names
that reach them, lists what exists on one side by nature (ASE and the other Python programs;
the infrastructure of the Rust side) with the reason, and lists what Python cannot reach yet
(``pending``). This test checks from the Python side that

- every module of the Rust library is accounted for (so a new module cannot be forgotten),
- every Python name of the table exists,
- every public Python name is in the table (so a new function cannot be forgotten),
- everything the registries of the Rust side list works through the Python functions.

``crates/mopt-rs/tests/api_parity.rs`` checks the Rust items of the same table. With
``MOPT_API_PARITY_RELEASE=1`` (for 2.0.0) nothing may be pending.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import mopt_rs
from mopt_rs import _core, optimize, variables

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/mopt-rs/src"
ANG = _core.ANGSTROM_TO_BOHR
NOT_YET_MODEL_HESSIANS: set[str] = set()
# Hessians from extra gradients rather than models: asked of the calculator, see mopt_rs.O1NumHess.
FROM_GRADIENTS = {"o1numhess"}
WATER = (["O", "H", "H"], [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])


def _table() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((ROOT / "tools/api/parity.json").read_text(encoding="utf-8"))
    return data


def _rust_modules() -> list[str]:
    """Modules of the library down to the second level (``opt``, ``opt::step``, ...)."""
    out = []
    for top in sorted(SOURCE.iterdir()):
        if top.is_file():
            if top.suffix == ".rs" and top.stem not in {"lib", "mod"}:
                out.append(top.stem)
            continue
        out.append(top.name)
        for child in sorted(top.iterdir()):
            if child.is_dir() or (child.suffix == ".rs" and child.stem != "mod"):
                out.append(f"{top.name}::{child.stem}")
    return out


def _accounts_for(key: str, module: str) -> bool:
    """``key`` is the module, something inside it, or a module that contains it."""
    return key == module or key.startswith(module + "::") or module.startswith(key + "::")


def _resolve(name: str) -> Any:
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(name)


def test_every_rust_module_is_accounted_for() -> None:
    table = _table()
    keys = [k for c in table["components"].values() for k in c["rust"]]
    keys += list(table["pending"]) + list(table["rust_only"])
    missing = [m for m in _rust_modules() if not any(_accounts_for(k, m) for k in keys)]
    assert not missing, f"add these modules to tools/api/parity.json: {missing}"
    # A module is in one list only.
    both = set(table["pending"]) & set(table["rust_only"])
    assert not both, both


def test_python_names_of_the_table_exist() -> None:
    table = _table()
    for feature, entry in table["components"].items():
        assert entry["python"], f"'{feature}' names nothing in Python"
        for name in entry["python"]:
            _resolve(name)
    for name in table["python_only"]:
        _resolve(name)


def test_every_public_python_name_is_accounted_for() -> None:
    table = _table()
    listed = {n for c in table["components"].values() for n in c["python"]}
    listed_objects = [_resolve(n) for n in listed]
    public = [f"mopt_rs.{n}" for n in mopt_rs.__all__]
    public += [f"mopt_rs._core.{n}" for n in dir(_core) if not n.startswith("_")]
    public += [f"mopt_rs.optimize.{n}" for n in optimize.__all__]
    public += [f"mopt_rs.variables.{n}" for n in variables.__all__]
    missing = []
    for name in public:
        if name in listed or any(name.startswith(p) for p in table["python_only"]):
            continue
        # The same object under another name (a re-export) counts.
        if any(_resolve(name) is obj for obj in listed_objects):
            continue
        missing.append(name)
    assert not missing, f"add these names to tools/api/parity.json: {missing}"


def _flat(rows: list[list[float]]) -> list[float]:
    return (np.asarray(rows) * ANG).ravel().tolist()


def test_what_the_registries_list_works_from_python() -> None:
    # Built-in calculators: a structure each of them accepts.
    structures = {
        "lj": (["Ar", "Ar"], [[0.0, 0.0, 0.0], [3.9, 0.0, 0.0]]),
        "emt": (["Cu", "Cu"], [[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]]),
        "eam": (["Cu", "Ag"], [[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]]),
        "meam": (["Fe", "Cr"], [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
        "tersoff": (["Si", "Si"], [[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]]),
        "muller_brown": (["H"], [[-0.5, 1.4, 0.0]]),
        "gfn0xtb": WATER,
        "gfn1xtb": WATER,
        "gfnff": WATER,
    }
    names = _core.native_calculators()
    assert set(names) == set(structures), "give every built-in calculator a structure here"
    for name in names:
        symbols, rows = structures[name]
        energy, gradient, _ = _core.evaluate_native(name, symbols, _flat(rows))
        assert np.isfinite(energy) and np.all(np.isfinite(gradient)), name
        assert np.isfinite(mopt_rs.evaluate_native(name, symbols, _flat(rows))[0])
    # Model Hessians.
    not_yet = set()
    for name in _core.model_hessian_names():
        if name in FROM_GRADIENTS:
            with pytest.raises(ValueError, match="O1NumHessCalculator"):
                _core.model_hessian(name, WATER[0], _flat(WATER[1]), [0.0] * 9)
            continue
        try:
            h = np.asarray(_core.model_hessian(name, WATER[0], _flat(WATER[1]), [0.0] * 9))
        except ValueError as e:
            assert "not available yet" in str(e), name
            not_yet.add(name)
            continue
        assert h.shape == (81,) and np.all(np.isfinite(h)), name
    # Listed, but answered with "not available yet" on both sides; remove a name here when it
    # is implemented.
    assert not_yet == NOT_YET_MODEL_HESSIANS
    # Objectives of plain variables.
    arguments = {"mb2d": ("mb2d", [-0.5, 1.4]), "expr": ("expr:(q1-1)^2 + q2^2", [0.0, 0.0])}
    for name in _core.objective_names():
        full, x = arguments.get(name, (name, [-0.5, 1.4]))
        value, gradient, _ = _core.evaluate_objective(full, x)
        assert np.isfinite(value) and len(gradient) == len(x), name
    # Commands.
    assert _core.run_cli("optmain", ["-h"]) == 0
    assert _core.run_cli("freqmain", ["-h"]) == 0


def test_nothing_is_pending_at_release() -> None:
    pending = _table()["pending"]
    for reason in pending.values():
        assert reason.strip()
    if os.environ.get("MOPT_API_PARITY_RELEASE") == "1":
        assert not pending, f"not reachable from Python yet: {sorted(pending)}"


def test_settings_of_built_in_calculators_from_python() -> None:
    """The settings of ``-calc_opt`` reach the Rust potentials from Python as well."""
    from mopt_rs.calculators.bridges import NativeEngine
    from mopt_rs.optimize import NativeCalculator, Task

    symbols = ["Ar", "Ar"]

    def energy(distance: float, settings: list[tuple[str, str]] | None) -> float:
        return _core.evaluate_native(
            "lj", symbols, [0.0, 0.0, 0.0, distance, 0.0, 0.0], settings=settings
        )[0]

    # Inside the cutoff of 10 Bohr the potential is switched off; beyond it there is none.
    cut = [("cutoff", str(10.0 * _core.BOHR_TO_ANGSTROM))]
    assert energy(7.0, cut) == pytest.approx(energy(7.0, None), rel=1e-12)
    assert abs(energy(9.5, cut)) < abs(energy(9.5, None))
    assert energy(10.5, cut) == 0.0 != energy(10.5, None)
    with pytest.raises(Exception, match="unknown setting"):
        energy(7.0, [("rc", "5")])

    task = Task(
        label="pair",
        calculator="lj",
        symbols=symbols,
        positions_bohr=np.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]]),
        charge=0,
        multiplicity=1,
        need_hessian=False,
    )
    kept = NativeCalculator("lj", settings={"cutoff": 10.0 * _core.BOHR_TO_ANGSTROM})
    assert kept.evaluate(task).energy == pytest.approx(energy(9.5, cut), abs=1e-15)
    assert "lj" in NativeEngine("lj", settings=cut).describe()

    # The reference of a pair of the modified embedded-atom method.
    pair = [("pair", "Fe-Cr,b2,4.2,2.5,5.1")]
    args = ("meam", ["Fe", "Cr"], [0.0, 0.0, 0.0, 4.6, 0.0, 0.0])
    assert _core.evaluate_native(*args, settings=pair)[0] != _core.evaluate_native(*args)[0]


def test_stress_of_built_in_calculators_from_python() -> None:
    """Stress in the convention of ASE, and its six components."""
    a = 3.5 * _core.ANGSTROM_TO_BOHR
    h = a / 2.0
    cell = [a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a]
    positions = [0.0, 0.0, 0.0, h, h, 0.0, h, 0.0, h, 0.0, h, h]
    for name in ("emt", "eam", "meam"):
        stress = np.array(_core.native_stress(name, ["Cu"] * 4, positions, cell))
        assert stress.shape == (3, 3) and np.allclose(stress, stress.T, atol=1e-12)
        # Copper squeezed to a lattice constant of 3.5 Angstrom pushes outwards.
        assert stress[0, 0] < 0.0 and abs(stress[0, 1]) < 1e-10
        voigt = _core.stress_to_ase_voigt(stress.tolist())
        assert voigt[0] == pytest.approx(stress[0, 0] * 183.631531, rel=1e-6)
        assert np.allclose(_core.stress_from_ase_voigt(voigt), stress, atol=1e-15)
    # The cutoff of Lennard-Jones is a setting; without it the stress comes from differences.
    argon = [x * 1.5 for x in positions]
    wide = [x * 1.5 for x in cell]
    cut = np.array(_core.native_stress("lj", ["Ar"] * 4, argon, wide, [("cutoff", "8.0")]))
    assert np.isfinite(cut).all() and abs(cut[0, 0]) > 0.0
