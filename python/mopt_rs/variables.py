"""Optimization of plain variables (no atoms; any number of variables).

The same optimizers as for molecules (minima and saddle points; ``rsirfo_*``, ``FIRE``,
``lbfgs``, ``ccqn``, ...) applied to a function ``x in R^n -> value, gradient[, Hessian]``:
a Python function, a built-in objective (``"mb2d"``, the two-variable Muller-Brown surface),
or an expression of ``q1, q2, ...`` (``"expr:(q1-1)^2 + q2^2"``). :func:`optimize_variables`
runs the loop; :class:`VectorOptimizer` is the ask/tell optimizer underneath. Thresholds and
trust radii are plain numbers in the units of the variables and of the objective.

MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy optimizes atomic coordinates only).

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

from mopt_rs import _core
from mopt_rs._core import VectorOptimizer, objective_names

FloatArray = npt.NDArray[np.float64]
ObjectiveFunction = Callable[[FloatArray], Sequence[Any]]
HessianFunction = Callable[[FloatArray], npt.ArrayLike]

__all__ = [
    "NativeObjective",
    "VariableResult",
    "VectorOptimizer",
    "finite_difference_hessian",
    "objective_names",
    "optimize_variables",
]


class NativeObjective:
    """A built-in objective by name (``mb2d``, ``expr:...``), evaluated in Rust.

    ``options`` are its settings (``energy_scale`` of ``mb2d``). Calling it returns
    ``(value, gradient, hessian)``, the analytic Hessian included.
    """

    def __init__(self, name: str, **options: Any) -> None:
        self.name = name
        self.options = [(k, str(v)) for k, v in options.items()]
        # Fail now on unknown names and settings rather than at the first evaluation.
        head = name.split(":", 1)[0].strip().lower()
        if head not in objective_names():
            known = ", ".join(objective_names())
            raise ValueError(f"unknown objective '{head}' (known: {known})")

    def evaluate(
        self, x: npt.ArrayLike, need_hessian: bool = False
    ) -> tuple[float, FloatArray, FloatArray | None]:
        """Value, gradient, and (when asked for) the Hessian at ``x``."""
        xs = np.asarray(x, dtype=np.float64).ravel()
        value, g, h = _core.evaluate_objective(self.name, xs.tolist(), need_hessian, self.options)
        n = xs.size
        hess = None if h is None else np.asarray(h, dtype=np.float64).reshape(n, n)
        return value, np.asarray(g, dtype=np.float64), hess

    def __call__(self, x: npt.ArrayLike) -> tuple[float, FloatArray, FloatArray]:
        value, g, h = self.evaluate(x, need_hessian=True)
        assert h is not None
        return value, g, h


@dataclass
class VariableResult:
    """Outcome of :func:`optimize_variables`."""

    termination: str | None
    iterations: int
    x: FloatArray
    value: float
    gradient: FloatArray
    values: list[float] = field(default_factory=list)
    reports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.termination == "converged"


def finite_difference_hessian(
    gradient: Callable[[FloatArray], npt.ArrayLike], x: npt.ArrayLike, step: float = 1e-4
) -> FloatArray:
    """Symmetrized central-difference Hessian from ``gradient`` (``2 n`` gradient calls)."""
    x0 = np.asarray(x, dtype=np.float64).ravel()
    n = x0.size
    h = np.empty((n, n))
    for i in range(n):
        d = np.zeros(n)
        d[i] = step
        gp = np.asarray(gradient(x0 + d), dtype=np.float64).ravel()
        gm = np.asarray(gradient(x0 - d), dtype=np.float64).ravel()
        h[i] = (gp - gm) / (2.0 * step)
    return 0.5 * (h + h.T)


def _unpack(out: Sequence[Any], n: int) -> tuple[float, FloatArray, FloatArray | None]:
    if len(out) not in (2, 3):
        raise ValueError(
            "the objective must return (value, gradient) or (value, gradient, hessian)"
        )
    g = np.asarray(out[1], dtype=np.float64).ravel()
    if g.size != n:
        raise ValueError(f"the gradient has {g.size} components, expected {n}")
    h = None
    if len(out) == 3 and out[2] is not None:
        h = np.asarray(out[2], dtype=np.float64)
        if h.shape != (n, n):
            raise ValueError(f"the Hessian has shape {h.shape}, expected ({n}, {n})")
    return float(out[0]), g, h


def optimize_variables(
    objective: str | ObjectiveFunction,
    x0: npt.ArrayLike,
    *,
    methods: str | Sequence[str] = "rsirfo_bfgs",
    saddle_order: int = 0,
    max_iterations: int = 1000,
    criteria: str = "normal",
    exact_hessian_interval: int | None = None,
    hessian: HessianFunction | None = None,
    max_trust_radius: float | None = None,
    min_trust_radius: float = 0.01,
    ccqn_direction: npt.ArrayLike | None = None,
    ccqn_cos_phi: float | None = None,
    mode_kill: bool = False,
    options: Mapping[str, Any] | None = None,
    callback: Callable[[dict[str, Any]], bool | None] | None = None,
) -> VariableResult:
    """Optimize the variables ``x`` of ``objective`` from ``x0``.

    ``objective`` is a function ``f(x) -> (value, gradient)`` or ``(value, gradient,
    hessian)``, or the name of a built-in one (``"mb2d"``, ``"expr:..."``; ``options`` are its
    settings). ``methods`` is one method or two blended ones, as ``-opt``; ``saddle_order=1``
    searches a first-order saddle point. With ``exact_hessian_interval=k`` the exact Hessian is
    used every ``k`` iterations: the one the objective returns, else ``hessian(x)``, else
    central differences of the gradient. ``ccqn_direction`` is the axis of the ``ccqn``
    methods. With ``mode_kill`` the Hessian is checked at convergence, and a stationary point
    with more negative curvatures than ``saddle_order`` is left along the surplus modes.
    ``callback`` receives every iteration report and stops the run by returning ``True``.
    """
    start = np.asarray(x0, dtype=np.float64).ravel()
    n = start.size
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

    opt = VectorOptimizer(
        start.tolist(),
        methods=[methods] if isinstance(methods, str) else list(methods),
        saddle_order=saddle_order,
        max_iterations=max_iterations,
        criteria=criteria,
        exact_hessian_interval=exact_hessian_interval,
        max_trust_radius=max_trust_radius,
        min_trust_radius=min_trust_radius,
        ccqn_direction=(
            None
            if ccqn_direction is None
            else np.asarray(ccqn_direction, dtype=np.float64).ravel().tolist()
        ),
        ccqn_cos_phi=ccqn_cos_phi,
        mode_kill=mode_kill,
    )
    values: list[float] = []
    reports: list[dict[str, Any]] = []
    last: tuple[float, FloatArray] | None = None
    while True:
        request = opt.ask()
        if request is None:
            break
        x = np.asarray(request[0], dtype=np.float64)
        value, g, h = evaluate(x, request[1])
        report = opt.tell(value, g.tolist(), None if h is None else h.ravel().tolist())
        values.append(value)
        reports.append(report)
        last = (value, g)
        if callback is not None and callback(report):
            break
    if last is None:
        raise RuntimeError("no iteration was performed")
    return VariableResult(
        termination=opt.termination,
        iterations=opt.iterations,
        x=np.asarray(reports[-1]["x"], dtype=np.float64),
        value=last[0],
        gradient=last[1],
        values=values,
        reports=reports,
    )
