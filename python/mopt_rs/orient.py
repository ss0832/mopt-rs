"""Orientation search of a complex from Python with any calculator.

:func:`orientation_search` runs the ask/tell search of the Rust side
(:class:`OrientationSearch`): the parts of the complex are placed at random (MultiOptPy's
scheme, uniform rotations, or van der Waals contact), every start is optimized by
:class:`mopt_rs.Optimizer` with the given calculator, and the results that are the same minimum
(same bonds, RMSD after matching equivalent atoms and identical molecules, energy) are merged.
The command ``orientsearch`` (``mopt_rs.cli.orientsearch``, ``_core.run_cli("orientsearch",
...)``) does the same with every option of ``optmain`` and writes its files.

MultiOptPy: `Entrypoints/orientation_search.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from mopt_rs._core import OrientationSearch, compare_structures, complex_parts
from mopt_rs.optimize import Calculator, Optimizer, run

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "OrientMinimum",
    "OrientResult",
    "OrientationSearch",
    "compare_structures",
    "complex_parts",
    "orientation_search",
]


@dataclass
class OrientMinimum:
    """One distinct minimum."""

    positions_bohr: FloatArray
    """Shape (N, 3)."""
    energy: float
    weight: float
    """Boltzmann weight at the temperature of the search."""
    hits: int
    """How many samples ended in it."""
    first_sample: int
    sample: int
    """The sample whose structure it is."""


@dataclass
class OrientResult:
    """Outcome of :func:`orientation_search`."""

    minima: list[OrientMinimum]
    """Lowest first."""
    samples: list[dict[str, Any]] = field(default_factory=list)
    """Per sample: ``index``, ``origin``, ``converged``, ``energy``, ``iterations``, ``minimum``
    (number of the distinct minimum in the order found, from 1), ``new``."""
    evaluations: int = 0
    """Evaluations made by the optimizations."""


def orientation_search(
    symbols: Sequence[str],
    positions_bohr: npt.ArrayLike,
    calculator: Calculator | Mapping[str, Calculator],
    parts: Sequence[Sequence[int]] | None = None,
    *,
    samples: int = 5,
    temperature: float = 298.15,
    optimizer: Mapping[str, Any] | None = None,
    **search: Any,
) -> OrientResult:
    """Search the relative orientations of the ``parts`` of a complex (lists of atoms from 1,
    the rest one more part; ``None``: its molecules).

    ``optimizer`` holds keywords of :class:`mopt_rs.Optimizer` (``methods``, ``criteria``,
    ``coordinates``, ...); ``search`` those of :class:`OrientationSearch` (``scheme``,
    ``seed``, ``contact``, ``refine``, ``rmsd_angstrom``, ...). A sample whose optimization
    raises counts as failed and the search goes on.
    """
    symbols = list(symbols)
    flat = np.asarray(positions_bohr, dtype=np.float64).ravel().tolist()
    parts_list = None if parts is None else [list(p) for p in parts]
    s = OrientationSearch(symbols, flat, parts_list, samples=samples, **search)
    settings = dict(optimizer or {})
    records: list[dict[str, Any]] = []
    evaluations = 0
    while True:
        request = s.ask()
        if request is None:
            break
        entry: dict[str, Any] = {"index": request["index"], "origin": request["origin"]}
        try:
            opt = Optimizer(symbols, list(request["positions_bohr"]), **settings)
            result = run(opt, calculator)
        except (RuntimeError, ValueError, ArithmeticError) as e:
            told = s.tell(None, 0.0, False)
            entry.update(converged=False, energy=None, iterations=0, error=str(e))
        else:
            evaluations += len(result.energies)
            energy = result.energies[-1] if result.energies else float("nan")
            told = s.tell(result.positions_bohr.ravel().tolist(), energy, result.converged)
            entry.update(converged=result.converged, energy=energy, iterations=result.iterations)
        entry.update(minimum=told["minimum"], new=told["new"])
        records.append(entry)
    minima = [
        OrientMinimum(
            positions_bohr=np.asarray(m["positions_bohr"], dtype=np.float64).reshape(-1, 3),
            energy=float(m["energy"]),
            weight=float(m["weight"]),
            hits=int(m["hits"]),
            first_sample=int(m["first_sample"]),
            sample=int(m["sample"]),
        )
        for m in s.minima(temperature)
    ]
    return OrientResult(minima=minima, samples=records, evaluations=evaluations)
