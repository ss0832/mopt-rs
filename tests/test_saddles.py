"""Saddle points of index k from Python: high-index saddle dynamics, the lowest modes at a
point, the starts of the upward and downward searches, and the bands on a ridge.

The surface is two independent double wells and a stiff third variable,
``E = (x^2 - 1)^2 + 1.7 (y^2 - 1)^2 + 2 z^2``: its index-2 saddle is the origin, its
transition states lie at ``(+-1, 0, 0)`` and ``(0, +-1, 0)``.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import numpy as np
import pytest

from mopt_rs import (
    HighIndexSaddleSearch,
    hisd_downward_starts,
    hisd_upward_starts,
    lowest_modes,
    ridge_band_search,
    saddle_search,
)


def surface(p: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    x, y, z = p
    e = (x * x - 1) ** 2 + 1.7 * (y * y - 1) ** 2 + 2 * z * z
    g = np.array([4 * x * (x * x - 1), 6.8 * y * (y * y - 1), 4 * z])
    h = np.diag([12 * x * x - 4, 1.7 * (12 * y * y - 4), 4.0])
    return e, g, h


def test_an_index_two_saddle_from_gradients_and_from_hessians() -> None:
    start = [0.15, -0.1, 0.05]
    directions = [[1.0, 0.3, 0.0], [0.2, 1.0, 0.1]]
    for settings in ({}, {"update": "lobpcg"}, {"method": "ggad"}, {"curvature": "hessian", "update": "eigenvectors"}):
        r = saddle_search(surface, start, directions, 2, criteria="tight", **settings)
        assert r.end == "converged", settings
        assert r.point is not None and np.linalg.norm(r.point) < 1e-4
        assert len(r.curvatures) == 2 and max(r.curvatures) < 0
    with pytest.raises(ValueError, match="independent directions"):
        HighIndexSaddleSearch(start, [[1.0, 0.0, 0.0]], order=2)


def test_lowest_modes_and_the_starts_of_the_landscape() -> None:
    values, directions = lowest_modes(surface, [1.0, 0.0, 0.0], 3, tolerance=1e-6, curvature="central")
    assert values == pytest.approx([-6.8, 4.0, 8.0], abs=1e-3)
    assert abs(directions[0][1]) == pytest.approx(1.0, abs=1e-6)
    # Upward along x (z is convex everywhere: no index-2 saddle that way).
    up = hisd_upward_starts([1.0, 0.0, 0.0], [list(directions[0])], [list(directions[2])], 0.1)
    assert len(up) == 2 and up[0]["order"] == 2
    down = hisd_downward_starts([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0.1, 4)
    assert len(down) == 8 and all(s["order"] == 1 for s in down)
    # From x = 0.9 the climb along x leads to the saddle at the origin; from x = 1.1 it leads
    # away (the energy rises without end beyond the minimum at x = 1).
    found = []
    for s in up:
        r = saddle_search(surface, s["point"], s["directions"], s["order"], criteria="tight")
        if r.end == "converged" and r.point is not None:
            found.append(float(np.linalg.norm(r.point)))
    assert len(found) == 1 and found[0] < 1e-4


def test_the_bands_on_the_ridge_climb_to_the_index_two_saddle() -> None:
    # Side by side along x: both imaginary modes along y.
    first, last = [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]
    mode = [0.0, 1.0, 0.0]
    for fixed in (False, True):
        r = ridge_band_search(surface, first, last, mode, mode, fixed=fixed, tolerance=1e-5, spring=1.0) if not fixed else ridge_band_search(
            surface, first, last, mode, mode, fixed=True, tolerance=1e-5
        )
        assert r.point is not None and np.linalg.norm(r.point) < 1e-3, fixed
        assert len(r.directions) == 2
