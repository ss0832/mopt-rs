"""The searches of ``ieipmain`` on functions of plain variables (or any other evaluation).

- :func:`ieip_search`: the improved elastic image pair method between two points
  (Liu, Qi, Lei, J. Chem. Theory Comput. 2023, 19, 2410), MultiOptPy's procedure;
- :func:`dimer_search`: the dimer method from one point (Henkelman, Jonsson 1999; Kastner,
  Sherwood 2008);
- :func:`spring_pair_search`: the spring pair method from one point (Cui, Jiang,
  Phys. Rev. E 2024, 110, 064123);
- :func:`two_point_shs`: the two-point scaled hypersphere search between two minima
  (Maeda, Ohno, Chem. Phys. Lett. 2005, 404, 95);
- :func:`newton_trajectory`: the growing Newton trajectory from a minimum (Quapp 2005);
- :func:`addf_search`: the anharmonic downward distortion following from a minimum
  (Ohno, Maeda 2004);
- :func:`gsm_search`: the growing string, double-ended or along driving coordinates
  (Peters et al. 2004; Zimmerman 2013, 2015); :func:`fsm_search`: the freezing string
  (Behn et al. 2011); :func:`dsafir_search`: the double-sphere AFIR (Maeda et al. 2018).

:func:`pair_methods` lists the registry of two-point methods (the names autots and the
mapper choose from) and :func:`run_pair_method` runs any of them by name on a built-in
objective.

Each takes a function ``x in R^n -> value, gradient[, Hessian]`` (a Python function, or the
name of a built-in objective such as ``"mb2d"``) and returns the estimate of the transition
state and how the search went. The ask/tell classes underneath (:class:`ElasticImagePair`,
:class:`Dimer`, :class:`SpringPair`, :class:`TwoPointShs`, :class:`NewtonTrajectory`,
:class:`Addf`) can be driven with any other evaluation (structures with a calculator, for
example). For structure files, the command ``ieipmain`` does the same with every calculator
(``mopt_rs._core.run_cli("ieipmain", [...])``).

MultiOptPy: `ieip.py`, `OtherMethod/` (see the migration notes for what differs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mopt_rs._core import (
    Addf,
    Dimer,
    DsAfir,
    ElasticImagePair,
    FreezingString,
    GrowingString,
    NewtonTrajectory,
    SpringPair,
    TwoPointShs,
    pair_methods,
    run_cli,
    run_pair_method,
)
from mopt_rs.variables import (
    FloatArray,
    NativeObjective,
    ObjectiveFunction,
    _unpack,
    finite_difference_hessian,
)

__all__ = [
    "Addf",
    "Dimer",
    "DsAfir",
    "ElasticImagePair",
    "FreezingString",
    "GrowingString",
    "IeipJob",
    "NewtonTrajectory",
    "SearchResult",
    "SpringPair",
    "TwoPointShs",
    "addf_search",
    "dimer_search",
    "dsafir_search",
    "fsm_search",
    "gsm_search",
    "ieip_search",
    "newton_trajectory",
    "pair_methods",
    "run_pair_method",
    "spring_pair_search",
    "two_point_shs",
]


@dataclass
class SearchResult:
    """Outcome of one of the searches."""

    #: Whether the search ended on its criteria.
    converged: bool
    #: The estimate of the transition state (``None`` if there is none).
    transition_state: FloatArray | None
    #: Its value, where it was evaluated.
    value: float | None
    #: Evaluations of the function (with a gradient or without).
    evaluations: int
    #: Further estimates (``addf_search``: one per path that passed a transition state).
    others: list[FloatArray] = field(default_factory=list)


Evaluate = Callable[[FloatArray, bool], tuple[float, FloatArray, "FloatArray | None"]]


def _evaluator(
    objective: str | ObjectiveFunction, n: int, options: Mapping[str, Any] | None
) -> Evaluate:
    if isinstance(objective, str):
        native = NativeObjective(objective, **dict(options or {}))
        return lambda x, h: native.evaluate(x, h)
    if options:
        raise ValueError("options are the settings of a built-in objective")
    fun = objective

    def evaluate(x: FloatArray, need_hessian: bool) -> tuple[float, FloatArray, FloatArray | None]:
        value, g, h = _unpack(fun(x), n)
        if need_hessian and h is None:
            h = finite_difference_hessian(lambda y: _unpack(fun(y), n)[1], x)
        return value, g, h if need_hessian else None

    return evaluate


def _drive(search: Any, evaluate: Evaluate) -> int:
    """Run an ask/tell search of points to its end; returns the number of evaluations."""
    count = 0
    while True:
        points = search.ask()
        if points is None:
            return count
        values: list[tuple[float, list[float] | None, list[float] | None]] = []
        for p in points:
            x = np.asarray(p["x"], dtype=np.float64)
            value, g, h = evaluate(x, bool(p["hessian"]))
            count += 1
            values.append(
                (
                    value,
                    g.tolist() if p["gradient"] else None,
                    None if h is None or not p["hessian"] else h.ravel().tolist(),
                )
            )
        search.tell(values)


def _array(v: npt.ArrayLike) -> FloatArray:
    return np.asarray(v, dtype=np.float64).ravel()


def ieip_search(
    objective: str | ObjectiveFunction,
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    max_iterations: int = 999,
    beta: float = 1.0,
    options: Mapping[str, Any] | None = None,
) -> SearchResult:
    """The improved elastic image pair method between ``first`` and ``second``; the estimate
    is the midpoint of the two images at the end."""
    a, b = _array(first), _array(second)
    evaluate = _evaluator(objective, a.size, options)
    pair = ElasticImagePair(a.tolist(), b.tolist(), max_iterations=max_iterations, beta=beta)
    count = 0
    while True:
        images = pair.ask()
        if images is None:
            break
        told = []
        for x in images:
            value, g, _ = evaluate(np.asarray(x, dtype=np.float64), False)
            told.append((value, g.tolist(), None, None))
            count += 1
        pair.tell((told[0], told[1]))
    ts = np.asarray(pair.transition_state_estimate, dtype=np.float64)
    value = evaluate(ts, False)[0]
    return SearchResult(pair.end == "converged", ts, value, count + 1)


def dimer_search(
    objective: str | ObjectiveFunction,
    start: npt.ArrayLike,
    axis: npt.ArrayLike,
    *,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SearchResult:
    """The dimer method from ``start`` with the axis ``axis`` (settings: ``separation``,
    ``trial_angle``, ``max_iterations``, ``max_rotations``, ``rotation_tolerance``,
    ``max_step``, ``criteria``)."""
    x0 = _array(start)
    evaluate = _evaluator(objective, x0.size, options)
    d = Dimer(x0.tolist(), _array(axis).tolist(), **settings)
    count = _drive(d, evaluate)
    center = np.asarray(d.center, dtype=np.float64)
    return SearchResult(bool(d.converged), center, evaluate(center, False)[0], count + 1)


def spring_pair_search(
    objective: str | ObjectiveFunction,
    start: npt.ArrayLike,
    direction: npt.ArrayLike,
    *,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SearchResult:
    """The spring pair method from the minimum ``start``, the second particle along
    ``direction`` (settings: ``spring_length``, ``offset``, ``drift_rate``, ``spring_rate``,
    ``climb_rate``, ``drift_tolerance``, ``drift_steps``, ``max_cycles``, ``max_step``,
    ``criteria``)."""
    x0 = _array(start)
    evaluate = _evaluator(objective, x0.size, options)
    s = SpringPair(x0.tolist(), _array(direction).tolist(), **settings)
    count = _drive(s, evaluate)
    estimate = s.saddle_estimate
    ts = None if estimate is None else np.asarray(estimate, dtype=np.float64)
    value = None if ts is None else evaluate(ts, False)[0]
    return SearchResult(bool(s.converged), ts, value, count + (ts is not None))


def two_point_shs(
    objective: str | ObjectiveFunction,
    center: npt.ArrayLike,
    other: npt.ArrayLike,
    *,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SearchResult:
    """The two-point scaled hypersphere search from ``other`` to the minimum ``center``
    (settings: ``step``, ``max_spheres``, ``sphere_iterations``, ``sphere_gradient``); the
    estimate is the highest point of the path."""
    c = _array(center)
    evaluate = _evaluator(objective, c.size, options)
    s = TwoPointShs(c.tolist(), _array(other).tolist(), **settings)
    count = _drive(s, evaluate)
    found = s.transition_state_estimate
    if found is None:
        return SearchResult(False, None, None, count)
    return SearchResult(True, np.asarray(found[0], dtype=np.float64), found[1], count)


def newton_trajectory(
    objective: str | ObjectiveFunction,
    start: npt.ArrayLike,
    final: npt.ArrayLike,
    *,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SearchResult:
    """The growing Newton trajectory from the minimum ``start`` towards ``final`` (settings:
    ``step_length``, ``rms_tolerance``, ``corrector_iterations``, ``max_iterations``)."""
    x0 = _array(start)
    evaluate = _evaluator(objective, x0.size, options)
    nt = NewtonTrajectory(x0.tolist(), final_point=_array(final).tolist(), **settings)
    count = _drive(nt, evaluate)
    found = nt.transition_state_estimate
    if found is None:
        return SearchResult(False, None, None, count)
    return SearchResult(True, np.asarray(found[0], dtype=np.float64), found[1], count)


class IeipJob:
    """An ``ieipmain`` run defined from a script (MultiOptPy's ``IEIPJob``): options by their
    long names (``"othersoft"``, ``"NSTEP"``, ``"use_dimer"``, ...), the run by the command
    with a settings file. ``run()`` returns the exit code; ``get_results()`` the run
    directory."""

    def __init__(self, input_path: str | Path) -> None:
        # Absolute, since the run is made from its own directory.
        self.input = str(Path(input_path).resolve())
        self.options: dict[str, Any] = {}
        self._directory: Path | None = None

    def set_option(self, key: str, value: Any) -> None:
        """One option (`True` switches a flag; a list gives several values)."""
        self.options[key] = value

    def set_options(self, **options: Any) -> None:
        """Several options."""
        self.options.update(options)

    def run(self, directory: str | Path = ".") -> int:
        """Run the command in ``directory`` (the run directory is made under it)."""
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        settings = {"INPUT": [self.input], **self.options}
        path = base / "ieipmain_job.json"
        path.write_text(json.dumps(settings, indent=1))
        before = set(base.rglob("*_iEIP_*"))
        cwd = Path.cwd()
        try:
            os.chdir(base)
            code = int(run_cli("ieipmain", ["--config", path.name]))
        finally:
            os.chdir(cwd)
        made = [p for p in base.rglob("*_iEIP_*") if p.is_dir() and p not in before]
        self._directory = max(made, key=lambda p: p.stat().st_mtime) if made else None
        return code

    def get_results(self) -> Path | None:
        """The run directory of the last ``run()``."""
        return self._directory


def addf_search(
    objective: str | ObjectiveFunction,
    start: npt.ArrayLike,
    *,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SearchResult:
    """The anharmonic downward distortion following from the minimum ``start`` (settings:
    ``initial_length``, ``step_length``, ``max_spheres``, ``paths``, ``sphere_iterations``,
    ``sphere_gradient``, ``duplicate_angle``, ``gradient_tolerance``); the estimate is that of
    the lowest transition state found, the others follow in ``others``."""
    x0 = _array(start)
    evaluate = _evaluator(objective, x0.size, options)
    a = Addf(x0.tolist(), **settings)
    count = _drive(a, evaluate)
    found = sorted(a.transition_states, key=lambda t: t[2])
    if not found:
        return SearchResult(False, None, None, count)
    points = [np.asarray(x, dtype=np.float64) for _, x, _ in found]
    return SearchResult(True, points[0], found[0][2], count, points[1:])


def _string_search(machine: Any, evaluate: Evaluate) -> SearchResult:
    count = _drive(machine, evaluate)
    found = machine.transition_state_estimate
    if found is None:
        return SearchResult(False, None, None, count)
    return SearchResult(
        bool(machine.converged), np.asarray(found[0], dtype=np.float64), found[1], count
    )


def _settings(settings: Mapping[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in settings.items()}


def gsm_search(
    objective: str | ObjectiveFunction,
    first: npt.ArrayLike,
    second: npt.ArrayLike | None = None,
    *,
    driving: list[str] | None = None,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SearchResult:
    """The growing string between ``first`` and ``second`` (DE-GSM), or from ``first`` along
    ``driving`` (SE-GSM; ``"VAR k value"``, counted from 1); the settings are the keys of
    ``gsm`` in the registry (``nodes``, ``max_step``, ``climb_tolerance``, ...)."""
    a = _array(first)
    evaluate = _evaluator(objective, a.size, options)
    b = None if second is None else _array(second).tolist()
    m = GrowingString(a.tolist(), b, driving=driving, settings=_settings(settings))
    return _string_search(m, evaluate)


def fsm_search(
    objective: str | ObjectiveFunction,
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SearchResult:
    """The freezing string between ``first`` and ``second`` (the keys of ``fsm``); the
    estimate is the highest frozen node."""
    a = _array(first)
    evaluate = _evaluator(objective, a.size, options)
    m = FreezingString(a.tolist(), _array(second).tolist(), settings=_settings(settings))
    count = _drive(m, evaluate)
    found = m.transition_state_estimate
    if found is None:
        return SearchResult(False, None, None, count)
    return SearchResult(True, np.asarray(found[0], dtype=np.float64), found[1], count)


def dsafir_search(
    objective: str | ObjectiveFunction,
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SearchResult:
    """DS-AFIR between ``first`` and ``second`` (the keys of ``dsafir``, among them
    ``variant="paper"``); the estimate is the highest point of the path."""
    a = _array(first)
    evaluate = _evaluator(objective, a.size, options)
    m = DsAfir(a.tolist(), _array(second).tolist(), settings=_settings(settings))
    return _string_search(m, evaluate)
