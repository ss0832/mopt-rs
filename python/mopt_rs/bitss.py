"""Binary-image transition state search (BITSS) between two points.

Two images, one in each of two basins, are minimized together while their separation is
reduced step by step; an energy term keeps either from being pulled over the ridge, so
they meet at the lowest point of the ridge between the basins, a transition state
(S. J. Avis, J. R. Panter, H. Kusumaatmaja, J. Chem. Phys. 2022, 157, 124107).

:func:`bitss_search` runs the search on a function ``x in R^n -> value, gradient[,
Hessian]`` (a Python function, a built-in objective such as ``"mb2d"``, or an expression);
:class:`Bitss` is the ask/tell state machine underneath, for any other way of evaluating
the points (the Cartesian coordinates of two structures with a calculator, for example).
For structure files, ``optmain INPUT -mf bitss REFERENCE`` (:func:`mopt_rs.run_optmain`)
does the same with every calculator.

MultiOptPy: `ModelFunction/binary_image_ts_search_model_function.py:BITSSModelFunction`
(``-mf bitss``); see the migration notes for what differs.

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

from mopt_rs._core import Bitss, bitss_coefficients, bitss_function
from mopt_rs.variables import (
    FloatArray,
    HessianFunction,
    NativeObjective,
    ObjectiveFunction,
    _unpack,
    finite_difference_hessian,
)

__all__ = [
    "Bitss",
    "BitssResult",
    "bitss_coefficients",
    "bitss_function",
    "bitss_search",
]


@dataclass
class BitssResult:
    """Outcome of :func:`bitss_search`."""

    #: Why the search ended: ``midpoint_gradient``, ``separation``, ``midpoint_shift``
    #: (converged), ``no_barrier``, ``stage_limit``, ``iteration_limit``; ``None`` if the
    #: callback stopped it.
    end: str | None
    #: Whether it ended on one of its criteria.
    converged: bool
    #: The point halfway between the images: the estimate of the transition state.
    transition_state: FloatArray
    #: Its value, as the last survey of the line between the images found it.
    value: float | None
    #: The two images at the end.
    first: FloatArray
    second: FloatArray
    stages: int
    iterations: int
    #: Evaluations with a gradient and of the value alone.
    gradient_evaluations: int
    value_evaluations: int
    reports: list[dict[str, Any]] = field(default_factory=list)


def bitss_search(
    objective: str | ObjectiveFunction,
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    methods: str | Sequence[str] = "lbfgs",
    max_iterations: int = 1000,
    criteria: str = "normal",
    exact_hessian_interval: int | None = None,
    hessian: HessianFunction | None = None,
    max_trust_radius: float | None = None,
    min_trust_radius: float = 0.01,
    reduction: float = 0.5,
    alpha: float = 10.0,
    beta: float = 0.1,
    coefficient_interval: int = 100,
    interpolation_points: int = 3,
    min_separation: float = 1e-2,
    max_stages: int = 30,
    step_share: float | None = 0.5,
    midpoint_shift: float | None = None,
    stop_on_midpoint_gradient: bool = True,
    frozen: Sequence[int] | None = None,
    options: Mapping[str, Any] | None = None,
    callback: Callable[[dict[str, Any]], bool | None] | None = None,
) -> BitssResult:
    """Search the transition state between the basins of ``first`` and ``second``.

    ``objective`` is a function ``f(x) -> (value, gradient)`` or ``(value, gradient,
    hessian)``, or the name of a built-in one (``"mb2d"``, ``"expr:..."``; ``options`` are its
    settings). ``methods`` is the minimizer of the stages (one method or two blended ones,
    as ``-opt``). ``reduction`` is the share by which the separation asked for shrinks from
    stage to stage, ``alpha`` and ``beta`` set the strengths of the energy and the distance
    terms, and the coefficients are set again every ``coefficient_interval`` iterations of
    a minimization, from the energies of ``interpolation_points`` points on the line
    between the images. No step is longer than ``step_share`` times the separation asked
    for. The search ends when the gradient at the midpoint of the images meets the force
    thresholds of ``criteria`` (``stop_on_midpoint_gradient``), when the images are closer
    than ``min_separation``, or when the midpoint moved by less than ``midpoint_shift``
    times the reduction of the separation of a stage. ``frozen`` lists variables that stay
    (the two points must agree in them). With ``exact_hessian_interval=k`` exact Hessians of
    the two images are used every ``k`` iterations: those the objective returns, else
    ``hessian(x)``, else central differences of the gradient. ``callback`` receives every
    report and stops the search by returning ``True``.
    """
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    n = a.size
    if b.size != n:
        raise ValueError(f"the two points have {n} and {b.size} variables")
    native = (
        NativeObjective(objective, **dict(options or {})) if isinstance(objective, str) else None
    )
    if native is None and options:
        raise ValueError("options are the settings of a built-in objective")
    fun: ObjectiveFunction | None = None if isinstance(objective, str) else objective

    def evaluate(x: FloatArray, need_hessian: bool) -> tuple[float, FloatArray, FloatArray | None]:
        if native is not None:
            return native.evaluate(x, need_hessian)
        assert fun is not None
        value, g, h = _unpack(fun(x), n)
        if need_hessian and h is None:
            if hessian is not None:
                h = np.asarray(hessian(x), dtype=np.float64)
                if h.shape != (n, n):
                    raise ValueError(f"the Hessian has shape {h.shape}, expected ({n}, {n})")
            else:
                h = finite_difference_hessian(lambda y: _unpack(fun(y), n)[1], x)
        return value, g, h if need_hessian else None

    search = Bitss(
        a.tolist(),
        b.tolist(),
        methods=[methods] if isinstance(methods, str) else list(methods),
        max_iterations=max_iterations,
        criteria=criteria,
        exact_hessian_interval=exact_hessian_interval,
        max_trust_radius=max_trust_radius,
        min_trust_radius=min_trust_radius,
        reduction=reduction,
        alpha=alpha,
        beta=beta,
        coefficient_interval=coefficient_interval,
        interpolation_points=interpolation_points,
        min_separation=min_separation,
        max_stages=max_stages,
        step_share=step_share,
        midpoint_shift=midpoint_shift,
        stop_on_midpoint_gradient=stop_on_midpoint_gradient,
        frozen=None if frozen is None else [int(k) for k in frozen],
    )
    reports: list[dict[str, Any]] = []
    with_gradient = 0
    value_only = 0
    while True:
        points = search.ask()
        if points is None:
            break
        values: list[tuple[float, list[float] | None, list[float] | None]] = []
        for point in points:
            x = np.asarray(point["x"], dtype=np.float64)
            value, g, h = evaluate(x, bool(point["hessian"]))
            if point["gradient"]:
                with_gradient += 1
            else:
                value_only += 1
            values.append(
                (
                    value,
                    g.tolist() if point["gradient"] else None,
                    None if h is None else h.ravel().tolist(),
                )
            )
        report = search.tell(values)
        reports.append(report)
        if callback is not None and callback(report):
            break
    image_1, image_2 = search.images
    return BitssResult(
        end=search.end,
        converged=search.converged,
        transition_state=np.asarray(search.transition_state_estimate, dtype=np.float64),
        value=search.midpoint_energy,
        first=np.asarray(image_1, dtype=np.float64),
        second=np.asarray(image_2, dtype=np.float64),
        stages=search.stage,
        iterations=search.iterations,
        gradient_evaluations=with_gradient,
        value_evaluations=value_only,
        reports=reports,
    )
