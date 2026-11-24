"""The first images between two structures from Python (``_core.images_between``).

MultiOptPy: no counterpart (new in mopt-py).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import numpy as np
import pytest

from mopt_rs import _core

# HCN and HNC on one line (Bohr): on the straight line the hydrogen passes through the
# other two atoms.
HCN = [0.0, 0.0, 0.0, 2.18, 0.0, 0.0, -2.01, 0.0, 0.0]
HNC = [0.0, 0.0, 0.0, 2.18, 0.0, 0.0, 4.06, 0.0, 0.0]


def nearest_to_hydrogen(images: list[list[float]]) -> float:
    x = np.asarray(images).reshape(len(images), 3, 3)
    return float(np.min(np.linalg.norm(x[:, :2, :] - x[:, 2:3, :], axis=2)))


def test_the_hydrogen_goes_round_with_every_method_but_the_straight_line() -> None:
    images, iterations, converged = _core.images_between("linear", HCN, HNC, 7)
    assert len(images) == 7 and iterations == 0 and converged
    assert nearest_to_hydrogen(images) < 0.6
    for method in ("lst", "idpp", "sidpp", "geodesic", "fbenm", "cfbenm"):
        images, iterations, converged = _core.images_between(
            method, HCN, HNC, 7, symbols=["C", "N", "H"]
        )
        assert len(images) == 7 and iterations > 0 and converged, method
        assert nearest_to_hydrogen(images) > 1.5, method


def test_fixed_atoms_stay_and_wrong_input_is_refused() -> None:
    images, _, _ = _core.images_between("geodesic", HCN, HNC, 3, fixed_atoms=[0, 1])
    for image in images:
        assert image[:6] == pytest.approx(HCN[:6], abs=1e-12)
    with pytest.raises(ValueError):
        _core.images_between("spline", HCN, HNC, 3)
    with pytest.raises(ValueError):
        _core.images_between("idpp", HCN, HNC[:6], 3)
    # The flat-bottom elastic network is made of bonds: it needs the elements.
    with pytest.raises(ValueError):
        _core.images_between("cfbenm", HCN, HNC, 3)
