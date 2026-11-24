"""Molecular dynamics from Python with any surface.

:func:`run_md` integrates the equations of motion on a function ``x -> value, gradient[,
Hessian]`` (a Python function, a built-in objective such as ``"mb2d"``, or an expression) of
plain variables with unit masses (or ``masses``), or, with ``symbols``, on any
:class:`mopt_rs.Calculator` for the Cartesian coordinates of a structure (Bohr; atomic
weights). The ensembles are those of ``mdmain -mt`` (``nve``, ``nosehoover``,
``nosehooverchain``, ``langevin``, ``berendsen``, ``bussi``, ``andersen``, ``npt``,
``npt_aniso``, ``npt_berendsen``, ``npt_berendsen_aniso``, ``npt_crescale``,
``npt_crescale_aniso``); ``options`` are the
``KEY=VALUE`` settings of ``-md_opt`` (``thermostat``, ``tau`` in fs, ``seed``,
``freeze=negative``, ...). After every step ``callback`` is called with the report of the
step; returning ``False`` ends the run cleanly.

Surface hopping by the Zhu-Nakamura theory (``options={"hopping_states": "0,1,2"}``, see
``mdmain``) takes the electronic states from two functions of one's own, so that any program
plugs in without a change of the library: ``ground(x) -> (energy, gradient)`` for state 0 and
``excited(x, state, need_gradient) -> (energy, gradient or None)`` for the excited states
(state 1 is the lowest). The run asks for the energies of all the states at every step, the
gradient of the state the trajectory is on, and the gradients of its neighbours only while
they are closer than ``hopping_gap``; ``electronic_state`` is the state it starts on.

:class:`MolecularDynamics` is the ask/tell state machine underneath, for any other way of
evaluating the surface. For structure files with any calculator, ``mdmain INPUT ...``
(``mopt_rs._core.run_cli("mdmain", [...])``) runs the same dynamics and writes MultiOptPy's
files.

:class:`TrajectoryAnalysis` analyzes frames in memory or the files of a run (radial
distribution functions, mean squared displacements and diffusion coefficients, velocity
autocorrelation and vibrational density of states, RMSD and RMSF, distances and angles,
principal components; Angstrom, fs, atoms from 0); :func:`block_average` gives the error of
the mean of a correlated series; :func:`analyze_md_run` writes the files of ``mdmain
-analyze_only`` for a run directory or a trajectory file.

Units: atomic units throughout (Bohr, Hartree, atomic time; ``time_fs`` in the reports).

MultiOptPy: `moleculardynamics.py:MD` and `Wrapper/md_wrapper.py:MDJob`; see the migration
notes for what differs.

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

from mopt_rs._core import (
    MolecularDynamics,
    TrajectoryAnalysis,
    analyze_md_run,
    autocorrelation,
    block_average,
    green_kubo,
    infrared_spectrum,
    md_parts,
    power_spectrum,
)
from mopt_rs.optimize import Calculator, Task
from mopt_rs.variables import (
    FloatArray,
    NativeObjective,
    ObjectiveFunction,
    _unpack,
    finite_difference_hessian,
)

__all__ = [
    "MdResult",
    "MolecularDynamics",
    "TrajectoryAnalysis",
    "analyze_md_run",
    "autocorrelation",
    "block_average",
    "green_kubo",
    "infrared_spectrum",
    "md_parts",
    "power_spectrum",
    "run_md",
]


@dataclass
class MdResult:
    """Outcome of :func:`run_md`."""

    #: The reports of the steps kept (every ``keep_every``-th and the last).
    reports: list[dict[str, Any]]
    #: Positions of the steps kept, one row each.
    positions: FloatArray
    #: Velocities after the last step.
    velocities: FloatArray
    #: Potential energy (with bias potentials), temperature (K), and conserved quantity of
    #: the steps kept.
    energies: FloatArray
    temperatures: FloatArray
    conserved: FloatArray
    #: Degrees of freedom of the temperature.
    dof: float
    #: Whether the callback stopped the run.
    interrupted: bool
    #: Evaluations of the surface.
    evaluations: int
    #: Parts of the run: integrator, thermostat, barostat.
    parts: tuple[str, str, str | None] = field(default=("", "", None))


def _options(options: Sequence[str] | Mapping[str, Any] | None) -> list[str]:
    if isinstance(options, Mapping):
        return [f"{key}={value}" for key, value in options.items()]
    return list(options or [])


def _evaluator(
    surface: ObjectiveFunction | Calculator | str,
    n: int,
    symbols: Sequence[str] | None,
    charge: int,
    multiplicity: int,
    objective_options: Mapping[str, Any] | None,
    hessian: Callable[[FloatArray], npt.ArrayLike] | None,
) -> Callable[[Mapping[str, Any]], tuple[float, FloatArray, FloatArray | None, FloatArray | None]]:
    """``request -> (energy, gradient, Hessian, stress)`` for the surface."""
    if isinstance(surface, str):
        native = NativeObjective(surface, **dict(objective_options or {}))

        def objective(r: Mapping[str, Any]) -> tuple[float, FloatArray, FloatArray | None, None]:
            value, g, h = native.evaluate(np.asarray(r["x"], dtype=np.float64), bool(r["hessian"]))
            return value, g, h, None

        return objective
    if objective_options:
        raise ValueError("objective_options are the settings of a built-in objective")
    if symbols is not None and hasattr(surface, "evaluate"):
        calc: Calculator = surface

        def structure(r: Mapping[str, Any]) -> tuple[float, FloatArray, Any, Any]:
            cell = r["cell_bohr"]
            task = Task(
                label="pes",
                calculator="default",
                symbols=list(symbols),
                positions_bohr=np.asarray(r["x"], dtype=np.float64).reshape(-1, 3),
                charge=charge,
                multiplicity=multiplicity,
                need_hessian=bool(r["hessian"]),
                cell_bohr=None
                if cell is None
                else np.asarray(cell, dtype=np.float64).reshape(3, 3),
                pbc=(cell is not None,) * 3,
            )
            ev = calc.evaluate(task)
            if r["stress"] and ev.stress is None:
                raise ValueError("the barostat needs the stress, and the calculator gave none")
            g = np.asarray(ev.gradient, dtype=np.float64).ravel()
            h = None if ev.hessian is None else np.asarray(ev.hessian, dtype=np.float64)
            return float(ev.energy), g, h, ev.stress

        return structure
    fun: ObjectiveFunction = surface  # type: ignore[assignment]

    def function(r: Mapping[str, Any]) -> tuple[float, FloatArray, FloatArray | None, None]:
        x = np.asarray(r["x"], dtype=np.float64)
        value, g, h = _unpack(fun(x), n)
        if r["hessian"] and h is None:
            if hessian is not None:
                h = np.asarray(hessian(x), dtype=np.float64)
            else:
                h = finite_difference_hessian(lambda y: _unpack(fun(y), n)[1], x)
        return value, g, h, None

    return function


def _state_functions(
    ground: Callable[[FloatArray], tuple[float, npt.ArrayLike]] | None,
    excited: Callable[[FloatArray, int, bool], tuple[float, npt.ArrayLike | None]] | None,
) -> Callable[[Mapping[str, Any]], tuple[float, FloatArray, list[float], list[list[float] | None]]]:
    """``request -> (energy, gradient, state energies, state gradients)`` for surface hopping
    from the functions of the ground and the excited states."""

    def call(x: FloatArray, state: int, need_gradient: bool) -> tuple[float, FloatArray | None]:
        g: npt.ArrayLike | None
        if state == 0 and ground is not None:
            energy, g = ground(x)
        elif excited is not None:
            energy, g = excited(x, state, need_gradient)
        else:
            raise ValueError(f"no function gives state {state} (excited is missing)")
        if not need_gradient:
            return float(energy), None
        if g is None:
            raise ValueError(f"the function of state {state} gave no gradient")
        grad = np.asarray(g, dtype=np.float64).ravel()
        if grad.size != x.size:
            raise ValueError(f"state {state}: {grad.size} gradient values for {x.size} variables")
        return float(energy), grad

    def evaluate(
        r: Mapping[str, Any],
    ) -> tuple[float, FloatArray, list[float], list[list[float] | None]]:
        x = np.asarray(r["x"], dtype=np.float64)
        req = r["states"]
        active = int(req["active"])
        energies: dict[int, float] = {}
        gradients: dict[int, FloatArray | None] = {}
        energy, g = call(x, active, True)
        assert g is not None
        energies[active], gradients[active] = energy, g
        expected = {int(s) for s in req.get("expected", [])}
        for state in req["states"]:
            if state != active:
                # The gradients of the neighbours that were close at the step before at once.
                energies[state], gradients[state] = call(x, state, state in expected)
        gap = float(req["gradient_gap"])
        for state in req["neighbours"]:
            if gradients[state] is None and abs(energies[state] - energy) < gap:
                energies[state], gradients[state] = call(x, state, True)
        order = [int(s) for s in req["states"]]
        return (
            energy,
            g,
            [energies[s] for s in order],
            [None if grad is None else grad.tolist() for grad in (gradients[s] for s in order)],
        )

    return evaluate


def run_md(
    surface: ObjectiveFunction | Calculator | str | None,
    x: npt.ArrayLike,
    *,
    symbols: Sequence[str] | None = None,
    masses: npt.ArrayLike | None = None,
    cell_bohr: npt.ArrayLike | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    ensemble: str = "nve",
    time_step: float | None = None,
    steps: int = 100000,
    temperature: float = 298.15,
    pressure: float = 101.3,
    options: Sequence[str] | Mapping[str, Any] | None = None,
    velocities: npt.ArrayLike | None = None,
    fix_atoms: Sequence[int] | None = None,
    constraints: Sequence[str] | None = None,
    user_constraints: Sequence[str] | None = None,
    constraint_functions: Sequence[Any] | None = None,
    objective_options: Mapping[str, Any] | None = None,
    hessian: Callable[[FloatArray], npt.ArrayLike] | None = None,
    callback: Callable[[dict[str, Any]], bool | None] | None = None,
    keep_every: int = 1,
    coordinates: str | None = None,
    ground: Callable[[FloatArray], tuple[float, npt.ArrayLike]] | None = None,
    excited: Callable[[FloatArray, int, bool], tuple[float, npt.ArrayLike | None]] | None = None,
    electronic_state: int | None = None,
) -> MdResult:
    """Run molecular dynamics from ``x`` on ``surface``.

    ``surface`` is a function ``f(x) -> (value, gradient)`` or ``(value, gradient, hessian)``
    of plain variables, the name of a built-in objective (``"mb2d"``, ``"expr:..."``;
    ``objective_options`` are its settings), or, with ``symbols``, a calculator that
    evaluates a :class:`mopt_rs.Task`. A Hessian is needed only for frozen modes
    (``options={"freeze": "negative"}``): the one the surface returns, else ``hessian(x)``,
    else central differences of the gradient. ``time_step`` is in atomic units of time
    (default 0.5 fs for structures, 0.1 for plain variables); ``pressure`` in kPa; ``steps``
    defaults to 100000, as ``mdmain -time``.
    ``coordinates`` (a name of ``-coord``, such as ``"dic"``) runs the dynamics of a structure
    in those non-redundant generalized coordinates.

    With surface hopping (``options`` with ``hopping_states``), the states come from
    ``ground(x) -> (energy, gradient)`` and ``excited(x, state, need_gradient) -> (energy,
    gradient or None)`` (``surface`` is then not used and may be ``None``); without
    ``ground``, state 0 is asked of ``excited`` too. ``electronic_state`` is the starting
    state (the lowest of ``hopping_states`` without it).
    """
    x0 = np.asarray(x, dtype=np.float64).ravel()
    md = MolecularDynamics(
        x0.tolist(),
        symbols=None if symbols is None else list(symbols),
        masses=None if masses is None else np.asarray(masses, dtype=np.float64).ravel().tolist(),
        cell_bohr=None
        if cell_bohr is None
        else np.asarray(cell_bohr, dtype=np.float64).ravel().tolist(),
        charge=charge,
        multiplicity=multiplicity,
        ensemble=ensemble,
        time_step=time_step,
        steps=steps,
        temperature=temperature,
        pressure=pressure,
        options=_options(options),
        velocities=None
        if velocities is None
        else np.asarray(velocities, dtype=np.float64).ravel().tolist(),
        fix_atoms=None if fix_atoms is None else list(fix_atoms),
        constraints=None if constraints is None else list(constraints),
        user_constraints=None if user_constraints is None else list(user_constraints),
        constraint_functions=None if constraint_functions is None else list(constraint_functions),
        coordinates=coordinates,
        electronic_state=electronic_state,
    )
    states = None
    if ground is not None or excited is not None:
        states = _state_functions(ground, excited)
        evaluate = None
    elif surface is None:
        raise ValueError("run_md needs a surface, or ground and excited for surface hopping")
    else:
        evaluate = _evaluator(
            surface, x0.size, symbols, charge, multiplicity, objective_options, hessian
        )
    kept: list[dict[str, Any]] = []
    last: dict[str, Any] | None = None
    interrupted = False
    while True:
        request = md.ask()
        if request is None:
            break
        if states is not None:
            if request.get("states") is None:
                raise ValueError("ground and excited are for surface hopping: give hopping_states")
            energy, g, energies, gradients = states(request)
            report = md.tell(
                float(energy),
                g.tolist(),
                state_energies=energies,
                state_gradients=gradients,
            )
        else:
            assert evaluate is not None
            energy, g, h, stress = evaluate(request)
            report = md.tell(
                float(energy),
                np.asarray(g, dtype=np.float64).ravel().tolist(),
                None if h is None else np.asarray(h, dtype=np.float64).ravel().tolist(),
                stress=None
                if stress is None
                else np.asarray(stress, dtype=np.float64).ravel().tolist(),
            )
        report["x"] = np.asarray(report["x"], dtype=np.float64)
        report["v"] = np.asarray(report["v"], dtype=np.float64)
        last = report
        if report["step"] % max(keep_every, 1) == 0:
            kept.append(report)
        if callback is not None and callback(report) is False:
            interrupted = not md.finished
            break
    if last is not None and (not kept or kept[-1]["step"] != last["step"]):
        kept.append(last)
    n = x0.size
    return MdResult(
        reports=kept,
        positions=np.array([r["x"] for r in kept], dtype=np.float64).reshape(-1, n),
        velocities=np.zeros(n) if last is None else np.asarray(last["v"], dtype=np.float64),
        energies=np.array([r["energy"] for r in kept], dtype=np.float64),
        temperatures=np.array([r["temperature"] for r in kept], dtype=np.float64),
        conserved=np.array([r["conserved"] for r in kept], dtype=np.float64),
        dof=float(md.dof),
        interrupted=interrupted,
        evaluations=int(md.evaluations),
        parts=tuple(md.parts),  # type: ignore[arg-type]
    )
