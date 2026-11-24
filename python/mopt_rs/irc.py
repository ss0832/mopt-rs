"""Intrinsic reaction coordinate (IRC): the steepest-descent path in mass-weighted
coordinates from a transition state to the minima it connects.

:func:`irc_path` follows the path on a function ``x -> value, gradient[, Hessian]`` (a Python
function, a built-in objective such as ``"mb2d"``, or an expression) from a transition state
(both sides), from any other point (one path down the gradient, a meta-IRC), or removes
imaginary modes (``method="modekill"``). With ``symbols`` the variables are the Cartesian
coordinates of a structure in Bohr, weighted by the atomic masses, and rigid-body motion is
projected out; otherwise they are plain variables with the masses given (one each by
default). After every point of the path ``callback`` is called with the report of the step
(side, structure, energy, length of the path); returning ``False`` ends the run cleanly.

:class:`Irc` is the ask/tell state machine underneath, for any other way of evaluating the
points. For structure files with any calculator, ``ircmain TS.xyz ...`` and
``optmain ... -irc STEP_SIZE MAX_STEP METHOD`` (:func:`mopt_rs.run_optmain`,
``mopt_rs._core.run_cli``) do the same and write MultiOptPy's files.

Methods: ``lqa``, ``hpc``, ``dvv``, ``euler``, ``rk4``, ``modekill`` (MultiOptPy's) and
``gs2`` (Gonzalez-Schlegel). ``options`` are the ``KEY=VALUE`` settings of ``-irc_opt``
(``first_step=check`` tests the first point of each side and corrects it on a hypersphere
around the transition state).

MultiOptPy: `irc.py:IRC` and `IRC/`; see the migration notes for what differs.

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

from mopt_rs._core import Irc, irc_first_step_test, irc_lqa_step, irc_methods
from mopt_rs.variables import (
    FloatArray,
    HessianFunction,
    NativeObjective,
    ObjectiveFunction,
    _unpack,
    finite_difference_hessian,
)

__all__ = [
    "Irc",
    "IrcResult",
    "IrcSide",
    "irc_first_step_test",
    "irc_lqa_step",
    "irc_methods",
    "irc_path",
]


@dataclass
class IrcSide:
    """One side of the path."""

    #: ``forward`` (along the transition vector), ``backward``, or ``single``.
    branch: str
    #: Why it ended: ``converged``, ``oscillation``, ``max_steps``, ``energy_rise``,
    #: ``modes_removed``, ``nothing_to_remove``; ``None`` if the callback stopped it.
    end: str | None = None
    #: The points of the path, one row each, from the first point after the start.
    points: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    #: Energies at the points (with bias potentials when the surface has them).
    energies: FloatArray = field(default_factory=lambda: np.empty(0))
    #: Signed mass-weighted length of the path from the start to each point (negative on
    #: the backward side).
    arc_lengths: FloatArray = field(default_factory=lambda: np.empty(0))
    #: The reports of the steps as the state machine gave them.
    steps: list[dict[str, Any]] = field(default_factory=list)
    #: Points that were dropped because their energy was above that of the point before
    #: (their steps were taken again, shorter).
    dropped_points: int = 0
    #: Points that were dropped because the gradient there pointed back along the step
    #: (the step went past the floor of the valley).
    overshot_points: int = 0
    #: Times the step length was halved after a point that was kept (the path turned back).
    shortened_steps: int = 0
    #: Steps of the final approach to the minimum.
    approach_steps: int = 0


@dataclass
class IrcResult:
    """Outcome of :func:`irc_path`."""

    #: ``transition_state``, ``meta_irc``, or ``modekill``.
    kind: str | None
    #: The sides in the order they were run.
    sides: list[IrcSide]
    #: Whether the callback stopped the run.
    interrupted: bool
    #: The report of the starting structure (imaginary modes, transition vector, ...).
    start: dict[str, Any] | None
    #: The tests of the first points (``first_step=check``).
    first_steps: list[dict[str, Any]]
    #: Evaluations of the function.
    evaluations: int
    #: The whole path from the end of the forward side over the start to the end of the
    #: backward side, one row per point (the order of ``irc_combined_path.xyz``).
    path: FloatArray
    #: Energies along ``path``.
    path_energies: FloatArray
    #: Signed lengths along ``path`` (0 at the start).
    path_arc_lengths: FloatArray

    def side(self, branch: str) -> IrcSide | None:
        """The side ``branch``, if it was run."""
        return next((s for s in self.sides if s.branch == branch), None)


def irc_path(
    objective: ObjectiveFunction | str,
    x: npt.ArrayLike,
    *,
    symbols: Sequence[str] | None = None,
    masses: npt.ArrayLike | None = None,
    project_rigid: bool = True,
    step_size: float = 0.1,
    max_steps: int = 300,
    method: str = "lqa",
    exact_hessian_interval: int | None = None,
    options: Sequence[str] | Mapping[str, Any] | None = None,
    objective_options: Mapping[str, Any] | None = None,
    hessian: HessianFunction | None = None,
    callback: Callable[[dict[str, Any]], bool | None] | None = None,
    cell_bohr: npt.ArrayLike | None = None,
    pbc: Sequence[bool] | None = None,
    fix_atoms: Sequence[int] | None = None,
    constraints: Sequence[str] | None = None,
    user_constraints: Sequence[str] | None = None,
    constraint_functions: Sequence[Any] | None = None,
) -> IrcResult:
    """Follow the intrinsic reaction coordinate from ``x``.

    ``objective`` is a function ``f(x) -> (value, gradient)`` or ``(value, gradient,
    hessian)``, or the name of a built-in one (``"mb2d"``, ``"expr:..."``;
    ``objective_options`` are its settings). The Hessian is needed at the start (and every
    ``exact_hessian_interval`` points): the one the objective returns, else ``hessian(x)``,
    else central differences of the gradient. ``step_size`` is in mass-weighted coordinates
    (amu^0.5 Bohr for a structure), ``max_steps`` is MultiOptPy's ``MAX_STEP``. ``options``
    are the settings of ``-irc_opt`` as ``KEY=VALUE`` strings or a mapping.

    ``callback(report)`` is called after every point of the path with ``branch``, ``step``,
    ``x`` (array), ``energy``, ``biased_energy``, ``arc_length`` (from the first point of
    the side), ``s`` (signed length from the start), and the other entries of the step
    report; when it returns ``False`` the run ends there and the result has
    ``interrupted=True``.

    A structure may be periodic: ``cell_bohr`` (the lattice vectors as rows, 3 x 3 or 9
    numbers) and ``pbc`` (default: periodic along all three), as for :class:`Optimizer`.
    Constraints are those of :class:`Optimizer` with the same spelling: ``fix_atoms``
    (atoms from 1, ``-fix``), ``constraints`` (the tokens of ``-pc``), ``user_constraints``
    (``-uc``), ``constraint_functions``. They are resolved once at ``x`` (a target that is not
    given is its value at ``x``), and every point of the path holds them to 1e-10 (frozen
    atoms exactly): the path of steepest descent within them, as ``ircmain`` follows it.
    """
    x0 = np.asarray(x, dtype=np.float64).ravel()
    n = x0.size
    native = (
        NativeObjective(objective, **dict(objective_options or {}))
        if isinstance(objective, str)
        else None
    )
    if native is None and objective_options:
        raise ValueError("objective_options are the settings of a built-in objective")
    fun: ObjectiveFunction | None = None if isinstance(objective, str) else objective

    def evaluate(p: FloatArray, need_hessian: bool) -> tuple[float, FloatArray, FloatArray | None]:
        if native is not None:
            return native.evaluate(p, need_hessian)
        assert fun is not None
        value, g, h = _unpack(fun(p), n)
        if need_hessian and h is None:
            if hessian is not None:
                h = np.asarray(hessian(p), dtype=np.float64)
                if h.shape != (n, n):
                    raise ValueError(f"the Hessian has shape {h.shape}, expected ({n}, {n})")
            else:
                h = finite_difference_hessian(lambda y: _unpack(fun(y), n)[1], p)
        return value, g, h if need_hessian else None

    if isinstance(options, Mapping):
        option_list = [f"{key}={value}" for key, value in options.items()]
    else:
        option_list = list(options or [])
    irc = Irc(
        x0.tolist(),
        symbols=None if symbols is None else list(symbols),
        masses=None if masses is None else np.asarray(masses, dtype=np.float64).ravel().tolist(),
        project_rigid=project_rigid,
        step_size=step_size,
        max_steps=max_steps,
        method=method,
        exact_hessian_interval=exact_hessian_interval,
        options=option_list,
        cell_bohr=None if cell_bohr is None else np.ravel(np.asarray(cell_bohr, float)).tolist(),
        pbc=None if pbc is None else tuple(bool(v) for v in pbc),
        fix_atoms=None if fix_atoms is None else [int(a) for a in fix_atoms],
        constraints=None if constraints is None else list(constraints),
        user_constraints=None if user_constraints is None else list(user_constraints),
        constraint_functions=None if constraint_functions is None else list(constraint_functions),
    )
    # A structure is moved to its centre of mass: the start of the path as it is followed.
    x0 = np.asarray(irc.start, dtype=np.float64)
    q0 = np.asarray(irc.mass_weighted(x0.tolist()), dtype=np.float64)
    sides: dict[str, IrcSide] = {}
    offsets: dict[str, float] = {}
    order: list[str] = []
    start: dict[str, Any] | None = None
    first_steps: list[dict[str, Any]] = []
    start_energy = float("nan")
    interrupted = False

    def take(events: list[dict[str, Any]]) -> bool:
        """Record the events; False when the callback asked to stop."""
        nonlocal start, start_energy
        go_on = True
        for event in events:
            kind = event["event"]
            if kind == "start":
                start = event
                start_energy = float(event["biased_energy"])
            elif kind == "first_step":
                first_steps.append(event)
            elif kind == "side_end":
                side = sides.setdefault(event["branch"], IrcSide(event["branch"]))
                if event["branch"] not in order:
                    order.append(event["branch"])
                side.end = event["end"]
                side.dropped_points = int(event["dropped_points"])
                side.overshot_points = int(event["overshot_points"])
                side.shortened_steps = int(event["shortened_steps"])
                side.approach_steps = int(event["approach_steps"])
            elif kind == "step":
                branch = event["branch"]
                side = sides.setdefault(branch, IrcSide(branch))
                if branch not in order:
                    order.append(branch)
                point = np.asarray(event["x"], dtype=np.float64)
                if branch not in offsets:
                    q = np.asarray(irc.mass_weighted(point.tolist()), dtype=np.float64)
                    offsets[branch] = float(np.linalg.norm(q - q0)) - float(event["arc_length"])
                sign = -1.0 if branch == "backward" else 1.0
                report = dict(event)
                report["x"] = point
                report["s"] = sign * (offsets[branch] + float(event["arc_length"]))
                side.steps.append(report)
                if callback is not None and callback(report) is False:
                    go_on = False
        return go_on

    while True:
        request = irc.ask()
        if not take(irc.events()):
            interrupted = True
            break
        if request is None:
            break
        value, g, h = evaluate(np.asarray(request["x"], dtype=np.float64), bool(request["hessian"]))
        irc.tell(value, g.tolist(), None if h is None else h.ravel().tolist())
        if not take(irc.events()):
            interrupted = True
            break

    for side in sides.values():
        side.points = np.array([s["x"] for s in side.steps], dtype=np.float64).reshape(-1, n)
        side.energies = np.array([s["biased_energy"] for s in side.steps], dtype=np.float64)
        side.arc_lengths = np.array([s["s"] for s in side.steps], dtype=np.float64)
    forward = sides.get("forward") or sides.get("single")
    backward = sides.get("backward")
    rows = [x0[None, :]]
    energies = [np.array([start_energy])]
    lengths = [np.array([0.0])]
    if forward is not None and len(forward.steps) > 0:
        rows.insert(0, forward.points[::-1])
        energies.insert(0, forward.energies[::-1])
        lengths.insert(0, forward.arc_lengths[::-1])
    if backward is not None and len(backward.steps) > 0:
        rows.append(backward.points)
        energies.append(backward.energies)
        lengths.append(backward.arc_lengths)
    return IrcResult(
        kind=irc.kind,
        sides=[sides[b] for b in order],
        interrupted=interrupted,
        start=start,
        first_steps=first_steps,
        evaluations=irc.evaluations,
        path=np.vstack(rows),
        path_energies=np.concatenate(energies),
        path_arc_lengths=np.concatenate(lengths),
    )
