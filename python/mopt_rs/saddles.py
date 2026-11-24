"""Saddle points of index ``k``: high-index saddle dynamics and the bands on a ridge.

- :func:`saddle_search`: a saddle point of index ``order`` from one point and ``order``
  directions (HiOSD of Yin, Zhang, Zhang, SIAM J. Sci. Comput. 2019; GGAD of Quapp, Bofill,
  Theor. Chem. Acc. 2014; iHiSD of Su et al. 2025);
- :func:`lowest_modes`: the lowest curvatures and their directions at a point from
  gradients (LOBPCG on differences of gradients);
- :func:`hisd_upward_starts`, :func:`hisd_downward_starts`: the starts of the upward and
  downward searches of the solution landscape (Yin, Yu, Zhang, Sci. China Math. 2021);
- :func:`ridge_band_search`: a band between two transition states whose highest image climbs
  to the index-2 saddle between them (the moving frame of Maronsson, Jonsson, Vegge, Phys.
  Chem. Chem. Phys. 2012, or the fixed direction on the mode-maximized surface).

Each takes a function ``x in R^n -> value, gradient[, Hessian]`` (a Python function, or the
name of a built-in objective) like the searches of :mod:`mopt_rs.ieip`. The ask/tell
classes underneath (:class:`HighIndexSaddleSearch`, :class:`LowestModes`,
:class:`RidgeBand`, :class:`FixedRidgeBand`) take structures too (``symbols=``). The index of
what is found is decided by its Hessian, not by the search.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mopt_rs._core import (
    FixedRidgeBand,
    HighIndexSaddleSearch,
    LowestModes,
    RidgeBand,
    hisd_downward_starts,
    hisd_upward_starts,
)
from mopt_rs.ieip import _array, _drive, _evaluator
from mopt_rs.variables import ObjectiveFunction

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "FixedRidgeBand",
    "HighIndexSaddleSearch",
    "LowestModes",
    "RidgeBand",
    "SaddleResult",
    "hisd_downward_starts",
    "hisd_upward_starts",
    "lowest_modes",
    "ridge_band_search",
    "saddle_search",
]


@dataclass
class SaddleResult:
    """Outcome of a search for a saddle point of index ``k``."""

    #: Why the search ended (``converged``, ``handoff``, ``wrongindex``, ...).
    end: str | None
    #: The point found (the highest image of a band).
    point: FloatArray | None
    #: Its value.
    value: float | None
    #: The directions at the point (unit length each).
    directions: list[FloatArray]
    #: The curvatures along them (Ritz values; empty for a band).
    curvatures: list[float]
    #: Evaluations of the function.
    evaluations: int


def saddle_search(
    objective: str | ObjectiveFunction,
    start: npt.ArrayLike,
    directions: Sequence[npt.ArrayLike],
    order: int,
    *,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SaddleResult:
    """A saddle point of index ``order`` from ``start`` with ``order`` directions (the
    settings of :class:`HighIndexSaddleSearch`: ``method``, ``update``, ``curvature``,
    ``guard``, ``flow``, ``dimer_length``, ``max_step``, ``max_iterations``, ``criteria``,
    ``hand_off``, ...)."""
    x0 = _array(start)
    evaluate = _evaluator(objective, x0.size, options)
    search = HighIndexSaddleSearch(
        x0.tolist(), [_array(d).tolist() for d in directions], order=order, **settings
    )
    count = _drive(search, evaluate)
    point = np.asarray(search.point, dtype=np.float64)
    return SaddleResult(
        search.end,
        point,
        evaluate(point, False)[0],
        [np.asarray(d, dtype=np.float64) for d in search.directions],
        list(search.curvatures),
        count + 1,
    )


def lowest_modes(
    objective: str | ObjectiveFunction,
    point: npt.ArrayLike,
    count: int = 2,
    *,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> tuple[list[float], list[FloatArray]]:
    """The ``count`` lowest curvatures at ``point`` and their directions (settings of
    :class:`LowestModes`)."""
    x0 = _array(point)
    evaluate = _evaluator(objective, x0.size, options)
    modes = LowestModes(x0.tolist(), count=count, **settings)
    _drive(modes, evaluate)
    return list(modes.values), [np.asarray(d, dtype=np.float64) for d in modes.directions]


def ridge_band_search(
    objective: str | ObjectiveFunction,
    first: npt.ArrayLike,
    last: npt.ArrayLike,
    first_mode: npt.ArrayLike,
    last_mode: npt.ArrayLike,
    *,
    images: int = 7,
    fixed: bool = False,
    options: Mapping[str, Any] | None = None,
    **settings: Any,
) -> SaddleResult:
    """A band between the transition states ``first`` and ``last`` (with their imaginary
    modes); its highest image is the estimate of the index-2 saddle between them. With
    ``fixed`` the band on the mode-maximized surface of one direction (modes alike),
    otherwise the moving-frame band from a straight start."""
    a, b = _array(first), _array(last)
    evaluate = _evaluator(objective, a.size, options)
    ma, mb = _array(first_mode), _array(last_mode)
    band: Any
    if fixed:
        band = FixedRidgeBand(a.tolist(), b.tolist(), ma.tolist(), mb.tolist(), images=images, **settings)
        count = _drive(band, evaluate)
        top = band.saddle_point
        point = None if top is None else np.asarray(top, dtype=np.float64)
        end = band.end
    else:
        chain = [(a + (b - a) * i / (images + 1)).tolist() for i in range(images + 2)]
        band = RidgeBand(chain, (ma.tolist(), mb.tolist()), **settings)
        count = _drive(band, evaluate)
        point = np.asarray(band.chain[band.highest], dtype=np.float64)
        end = None
    value = None if point is None else evaluate(point, False)[0]
    return SaddleResult(
        end,
        point,
        value,
        [np.asarray(d, dtype=np.float64) for d in band.saddle_directions],
        [],
        count + (point is not None),
    )
