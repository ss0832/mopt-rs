"""Entry points the Rust commands call to use Python engines (``optmain -os pyscf`` run from
the pip-installed commands or ``mopt-py``).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mopt_rs.calculators.base import EngineCalculator, Structure
from mopt_rs.calculators.registry import (
    CalculatorOptions,
    create,
    parse_option_value,
)
from mopt_rs.calculators.workers import WorkerCalculator

Calculator = EngineCalculator | WorkerCalculator
"""What the commands evaluate with: one calculator, or copies of it in worker processes."""


def _workers(extra: dict[str, Any], parallel: Any) -> int:
    """The copies in worker processes: ``-parallel N`` (``parallel``, which the command
    hands over as the larger of ``-parallel`` and ``-calc_opt workers=N``), with ``workers``
    taken out of the program keywords and checked (1: no worker processes)."""
    found = extra.pop("workers", 1)
    if isinstance(found, bool) or not isinstance(found, int) or found < 1:
        raise ValueError(f"-calc_opt workers={found}: expected a number of processes (1, 2, ...)")
    return max(found, int(parallel or 1))


def make(name: str, options: Mapping[str, Any]) -> tuple[Calculator, dict[str, Any]]:
    """The calculator ``name`` and what the Rust side needs to know about it."""
    opts = dict(options)
    extra = {k: parse_option_value(str(v)) for k, v in opts.pop("extra", [])}
    workers = _workers(extra, opts.pop("parallel", 1))
    pairs = {k: dict(opts.pop(k, [])) for k in ("sub_basis", "ecp", "software_paths")}
    solvation = opts.pop("solvation", None)
    o = CalculatorOptions.from_mapping(
        {**opts, **pairs, "extra": extra, "solvation": tuple(solvation) if solvation else None}
    )
    calc: Calculator = WorkerCalculator(name, o, workers) if workers > 1 else create(name, o)
    return calc, calculator_info(calc)


def calculator_info(calc: Calculator) -> dict[str, Any]:
    """Capabilities used by the Rust calculator adapter."""
    caps = calc.engine.capabilities
    return {
        "name": calc.name,
        "description": calc.describe(),
        "citations": list(calc.engine.citations),
        "pbc": caps.pbc,
        "excited_states": caps.excited_states,
        "electrons": caps.electrons,
        "rigid_body_invariant": caps.rigid_body_invariant,
        "stability": calc.checks_stability,
        "point_charges": caps.point_charges,
        "analytic_hessian": analytic_hessian(calc),
    }


def analytic_hessian(calc: Calculator) -> bool:
    """Whether the engine gives analytic Hessians now (an ASE engine whose Hessian method
    failed has gone over to differences)."""
    return bool(calc.engine.capabilities.analytic_hessian)


def orbitals(
    calc: Calculator,
    symbols: list[str],
    positions_bohr: list[float],
    charge: int,
    multiplicity: int,
    cell_bohr: list[float] | None,
    pbc: list[bool],
) -> (
    tuple[
        list[float],
        int,
        list[tuple[list[float], int, list[float], list[float]]],
        list[int] | None,
    ]
    | None
):
    """The orbitals of a structure as flat lists: the overlap (row by row) and its size, for
    each set the coefficients (row by row), the number of orbitals, their energies, and their
    occupations, and the atom of every atomic orbital (``None`` if the engine does not say);
    ``None`` if the engine has no orbitals."""
    a, b, c = (bool(v) for v in pbc)
    s = Structure.build(
        symbols,
        positions_bohr,
        charge=charge,
        multiplicity=multiplicity,
        cell_bohr=cell_bohr,
        pbc=(a, b, c) if cell_bohr is not None else None,
    )
    found = calc.orbitals(s)
    if found is None:
        return None
    overlap = found["overlap"]
    sets = [
        (
            one["coefficients"].ravel().tolist(),
            int(one["coefficients"].shape[1]),
            one["energies"].tolist(),
            one["occupations"].tolist(),
        )
        for one in found["sets"]
    ]
    return overlap.ravel().tolist(), int(overlap.shape[0]), sets, found.get("atom_of_ao")


def state_characters(
    calc: Calculator,
    symbols: list[str],
    positions_bohr: list[float],
    charge: int,
    multiplicity: int,
    cell_bohr: list[float] | None,
    pbc: list[bool],
    count: int,
) -> tuple[list[float], int, int, list[float], list[float]] | None:
    """The lowest ``count`` excited states of a structure as flat lists: the orbitals (row by
    row) and their number, the number of occupied ones, the excitation energies, and the
    amplitudes (row by row, a column per state); ``None`` if the engine has none."""
    a, b, c = (bool(v) for v in pbc)
    s = Structure.build(
        symbols,
        positions_bohr,
        charge=charge,
        multiplicity=multiplicity,
        cell_bohr=cell_bohr,
        pbc=(a, b, c) if cell_bohr is not None else None,
    )
    found = calc.state_characters(s, count)
    if found is None:
        return None
    orbitals = found["orbitals"]
    return (
        orbitals.ravel().tolist(),
        int(orbitals.shape[0]),
        int(found["occupied"]),
        found["energies"].tolist(),
        found["amplitudes"].ravel().tolist(),
    )


def transitions(
    calc: Calculator,
    symbols: list[str],
    positions_bohr: list[float],
    charge: int,
    multiplicity: int,
    cell_bohr: list[float] | None,
    pbc: list[bool],
    count: int,
) -> list[tuple[float, list[float], list[float], list[float] | None, list[float] | None]] | None:
    """The lowest ``count`` excited states of a structure as ``(excitation, transition dipole,
    population changes, dipole change, transition populations)`` (the last two ``None`` if the
    engine does not give them); ``None`` if the engine has none."""
    a, b, c = (bool(v) for v in pbc)
    s = Structure.build(
        symbols,
        positions_bohr,
        charge=charge,
        multiplicity=multiplicity,
        cell_bohr=cell_bohr,
        pbc=(a, b, c) if cell_bohr is not None else None,
    )
    found = calc.transitions(s, count)
    if found is None:
        return None
    return [
        (
            float(t["excitation"]),
            t["transition_dipole"].tolist(),
            t["population_change"].tolist(),
            None if t.get("dipole_change") is None else t["dipole_change"].tolist(),
            None if t.get("transition_population") is None else t["transition_population"].tolist(),
        )
        for t in found
    ]


def evaluate_embedded(
    calc: Calculator,
    symbols: list[str],
    positions: list[float],
    charge: int,
    multiplicity: int,
    root: int,
    hessian: bool,
    sites: list[list[float]],
    charges: list[float],
    elements: list[str | None],
) -> tuple[float, list[float], list[float] | None, list[list[float]]]:
    """Evaluate an ONIOM model in the external field (atomic units)."""
    model = Structure.build(
        symbols, positions, charge=charge, multiplicity=multiplicity, excited_state=root
    )
    value, gradient = calc.evaluate_embedded(model, sites, charges, elements, hessian)
    e, g, h = value.as_tuple()
    return e, g, h, gradient.tolist()


def evaluate(
    calc: Calculator,
    symbols: list[str],
    positions_bohr: list[float],
    charge: int,
    multiplicity: int,
    cell_bohr: list[float] | None,
    pbc: list[bool],
    need_hessian: bool,
    excited_state: int,
    need_stress: bool = False,
    need_gradient: bool = True,
) -> tuple[
    float,
    list[float],
    list[float] | None,
    float | None,
    str | None,
    list[float] | None,
    list[str],
]:
    """One evaluation in atomic units as flat lists, ``<S^2>`` where the engine tells it,
    what the program printed, the stress of the cell (row by row) when it was asked for
    and the engine has one, and what the calculator has to say about it. Without
    ``need_gradient`` an engine that can leave it out gives the energy alone (the gradient
    is then zero)."""
    a, b, c = (bool(v) for v in pbc)
    s = Structure.build(
        symbols,
        positions_bohr,
        charge=charge,
        multiplicity=multiplicity,
        cell_bohr=cell_bohr,
        pbc=(a, b, c) if cell_bohr is not None else None,
        excited_state=excited_state,
    )
    ev = calc.evaluate_structure(s, need_hessian, need_stress, need_gradient)
    h = None if ev.hessian is None else np.asarray(ev.hessian, dtype=np.float64).ravel().tolist()
    gradient = np.asarray(ev.gradient, dtype=np.float64).ravel().tolist()
    stress = None if ev.stress is None else np.asarray(ev.stress, dtype=np.float64).ravel().tolist()
    return float(ev.energy), gradient, h, ev.spin_square, ev.output, stress, list(ev.notes)
