"""The direct MaxFlux method: a reaction path as a smooth curve between two points.

The path is a B-spline between two fixed ends. It is moved until the flux along it,
``integral exp(beta E) ds``, is smallest at a low temperature ``1/beta``; then its highest
point is an estimate of the transition state. Energies and gradients are needed at a few
points of the path only, which move towards its top (Koda, Saito, J. Chem. Theory
Comput. 2024, 20, 2798; the definitions of the authors' program PyDMF).

:func:`dmf_search` runs the method on a function ``x in R^n -> value, gradient`` (a Python
function, a built-in objective such as ``"mb2d"``, or an expression). :class:`DirectMaxFlux`
is the ask/tell state machine underneath, for any other way of evaluating the points (the
Cartesian coordinates of structures with a calculator, for example). :class:`DmfPath` is
the path alone (the spline, its arc length and the objective at given energies) and
:func:`dmf_profile` the energy profile through the points. For structure files,
``nebmain INPUT -dmf`` does the same with every calculator, with the ends minimized first.

The defaults are those of the method for energies in Hartree and coordinates in Bohr
(``beta`` = 10 per eV, the tolerance 0.04 eV/Angstrom, the largest step 0.3 Bohr, the gates
of the moves of the points); objectives on other scales need them scaled.

MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's ``nebmain -dmf`` is not ported,
see the migration notes, "The direct MaxFlux method").

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from mopt_rs._core import DirectMaxFlux, DmfPath, dmf_profile
from mopt_rs.variables import FloatArray, NativeObjective, ObjectiveFunction, _unpack

__all__ = [
    "DirectMaxFlux",
    "DmfPath",
    "DmfResult",
    "dmf_profile",
    "dmf_search",
]


@dataclass
class DmfResult:
    """Outcome of :func:`dmf_search`."""

    #: Why the run ended: ``converged``, ``settled`` (the top of the path no longer came
    #: down; the run left the path with the lowest top), ``iteration_limit``; ``None`` if
    #: the callback stopped it.
    end: str | None
    #: Whether it ended on one of its criteria (``converged`` or ``settled``).
    converged: bool
    #: The highest point of the path: the estimate of the transition state.
    transition_state: FloatArray
    #: The energy there, from the profile through the points.
    value: float
    #: Where it is, as a share of the length of the path.
    share: float
    #: Points of the path at equal shares of its length, the ends among them.
    path: FloatArray
    #: The energy profile along the path: rows of share of the length and energy.
    profile: FloatArray
    #: The points the energies were known at: shares of the length and energies.
    shares: FloatArray
    energies: FloatArray
    #: The length of the path.
    length: float
    iterations: int
    #: Evaluations (each with a gradient).
    gradient_evaluations: int
    reports: list[dict[str, Any]] = field(default_factory=list)


def dmf_search(
    objective: str | ObjectiveFunction,
    chain: npt.ArrayLike,
    *,
    points: int = 3,
    guards: int = 1,
    segments: int = 4,
    degree: int = 3,
    beta: float | None = None,
    moves: bool = True,
    move_settings: Mapping[str, float] | None = None,
    tolerance: float | None = None,
    max_iterations: int = 200,
    max_step: float = 0.3,
    first_curvature: float | None = None,
    memory: int = 6,
    rise: float = 0.01,
    settle: float = 1e-3,
    patience: int = 8,
    refit_unevenness: float = 3.0,
    fixed: Sequence[int] | None = None,
    samples: int = 41,
    options: Mapping[str, Any] | None = None,
    callback: Callable[[dict[str, Any]], bool | None] | None = None,
) -> DmfResult:
    """Relax the path through the points ``chain`` (rows; its first and last row are the
    ends, which do not move) by the direct MaxFlux method.

    ``objective`` is a function ``f(x) -> (value, gradient)`` (a Hessian, if it returns one,
    is not used), or the name of a built-in one (``"mb2d"``, ``"expr:..."``; ``options``
    are its settings). ``points`` points between the ends move towards the top of the
    energy profile (``moves``; ``move_settings`` changes the keys ``largest_share``,
    ``spacing``, ``start``, ``start_sharpness``, ``settle``, ``settle_sharpness``,
    ``settle_rate``), with ``guards`` more between each end and them at equal shares of
    the length. ``beta`` is in 1/(unit of the values) and ``tolerance`` is on the largest
    component of the gradient of the objective that changes the path. A step that raises
    the top of the path by more than ``rise`` of the barrier is taken back; the run ends
    when the top has not come down by ``settle`` of the barrier in ``patience``
    iterations. ``fixed`` lists variables that do not move. ``callback`` receives every
    report and stops the run by returning ``True``.
    """
    rows = np.atleast_2d(np.asarray(chain, dtype=np.float64))
    if rows.shape[0] < 2:
        raise ValueError("the path needs two points at least (its ends)")
    n = rows.shape[1]
    native = (
        NativeObjective(objective, **dict(options or {})) if isinstance(objective, str) else None
    )
    if native is None and options:
        raise ValueError("options are the settings of a built-in objective")
    fun: ObjectiveFunction | None = None if isinstance(objective, str) else objective

    def evaluate(x: FloatArray) -> tuple[float, FloatArray]:
        if native is not None:
            value, g, _ = native.evaluate(x, False)
            return value, g
        assert fun is not None
        value, g, _ = _unpack(fun(x), n)
        return value, g

    machine = DirectMaxFlux(
        rows.tolist(),
        points=points,
        guards=guards,
        segments=segments,
        degree=degree,
        beta=beta,
        moves=moves,
        move_settings=None if move_settings is None else dict(move_settings),
        tolerance=tolerance,
        max_iterations=max_iterations,
        max_step=max_step,
        first_curvature=first_curvature,
        memory=memory,
        rise=rise,
        settle=settle,
        patience=patience,
        refit_unevenness=refit_unevenness,
        fixed=None if fixed is None else [int(k) for k in fixed],
    )
    reports: list[dict[str, Any]] = []
    evaluations = 0
    while True:
        asked = machine.ask()
        if asked is None:
            break
        values: list[tuple[float, list[float]]] = []
        for point in asked:
            value, g = evaluate(np.asarray(point["x"], dtype=np.float64))
            evaluations += 1
            values.append((value, g.tolist()))
        report = machine.tell(values)
        if report is not None:
            reports.append(report)
            if callback is not None and callback(report):
                break
    result = machine.result
    if result is None:
        raise RuntimeError("the direct MaxFlux method ran no iteration")
    profile = _profile_rows(result)
    return DmfResult(
        end=machine.end,
        converged=machine.converged,
        transition_state=np.asarray(result["highest"], dtype=np.float64),
        value=float(result["top_energy"]),
        share=float(result["top_share"]),
        path=np.asarray(machine.path_points(samples), dtype=np.float64),
        profile=profile,
        shares=np.asarray(result["shares"], dtype=np.float64),
        energies=np.asarray(result["energies"], dtype=np.float64),
        length=float(result["length"]),
        iterations=machine.iterations,
        gradient_evaluations=evaluations,
        reports=reports,
    )


def _profile_rows(report: Mapping[str, Any]) -> FloatArray:
    """The energy profile of a report of :class:`DirectMaxFlux` as rows of share of the
    length and energy."""
    return np.asarray(report["profile"], dtype=np.float64).reshape(-1, 2)
