"""Engines for the tests of crossing searches with an engine of the user's own: a model of
three atoms with two surfaces whose crossing points are known.

``E_s = f_s(r12) + K (r23 - a_s)^2 + K (r13 - C)^2`` with ``f_0(r) = A (r - R1)^2 (r - R2)^2``
(two minima and a transition state between them) and ``f_1(r) = KC (r - RC)^2 + D`` (one
minimum between them).

- ``-os engine:crossing_engine:Surfaces``: surface 1 is the triplet, or the first excited
  state of the singlet, as a program gives states that are named by what they are.
- ``-os engine:crossing_engine:Ordered``: the two surfaces are the excited states 1 and 2 in
  the order of their energies at every structure, as a program for electronic structure
  counts them, and are handed out with their characters so that they can be followed.

Nothing of mopt-rs knows these engines: they are found by where they are imported from.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from mopt_rs.calculators import Capabilities, Engine, EngineResult, Structure
from mopt_rs.calculators.base import ATOMIC_UNITS

R1, R2, A = 2.0, 5.5, 0.004
RC, KC, D = 3.5, 0.02, 0.004
K, C = 0.05, 4.0
AS = (2.5, 3.0)

FloatArray = npt.NDArray[np.float64]


def curve(s: int, r: float) -> tuple[float, float]:
    """``f_s(r)`` and its slope."""
    if s == 0:
        p, q = r - R1, r - R2
        return A * p * p * q * q, 2.0 * A * p * q * (p + q)
    return KC * (r - RC) ** 2 + D, 2.0 * KC * (r - RC)


def surface(x: FloatArray, s: int) -> tuple[float, FloatArray]:
    """The energy and the gradient of the surface ``s`` at the positions ``x`` (Bohr)."""
    energy = 0.0
    gradient = np.zeros((3, 3))
    for (i, j), term in (
        ((0, 1), lambda r: curve(s, r)),
        ((1, 2), lambda r: (K * (r - AS[s]) ** 2, 2.0 * K * (r - AS[s]))),
        ((0, 2), lambda r: (K * (r - C) ** 2, 2.0 * K * (r - C))),
    ):
        between = x[j] - x[i]
        r = float(np.linalg.norm(between))
        value, slope = term(r)
        energy += value
        gradient[i] -= slope * between / r
        gradient[j] += slope * between / r
    return energy, gradient


def on_seam(r12: float) -> float:
    """The energy on the seam of the two surfaces at ``r12`` (``r13 = C``, the ``r23`` at
    which they are equal)."""
    r23 = (curve(1, r12)[0] - curve(0, r12)[0] + K * (AS[1] ** 2 - AS[0] ** 2)) / (
        2.0 * K * (AS[1] - AS[0])
    )
    return curve(0, r12)[0] + K * (r23 - AS[0]) ** 2


def seam_minimum(lo: float, hi: float) -> float:
    """The lowest energy of the seam for ``r12`` between ``lo`` and ``hi`` (golden section)."""
    ratio = 0.5 * (5.0**0.5 - 1.0)
    a, b = lo, hi
    for _ in range(200):
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        if on_seam(c) < on_seam(d):
            b = d
        else:
            a = c
    return on_seam(0.5 * (a + b))


class Surfaces(Engine):
    """The model with its surfaces named: surface 1 for the triplet, and for the first
    excited state of the singlet."""

    name = "crossing_surfaces"
    units = ATOMIC_UNITS
    capabilities = Capabilities(excited_states=True)

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        s = min(int(structure.state.multiplicity == 3) + structure.excited_state, 1)
        energy, gradient = surface(structure.positions_bohr, s)
        return EngineResult(energy, gradient=gradient)


class Ordered(Engine):
    """The model with its two surfaces as the excited states 1 and 2 in the order of their
    energies, handed out with their characters; the ground state lies below and is not
    used."""

    name = "crossing_ordered"
    units = ATOMIC_UNITS
    capabilities = Capabilities(excited_states=True)

    @staticmethod
    def _order(x: FloatArray) -> list[int]:
        return sorted((0, 1), key=lambda s: surface(x, s)[0])

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        x = structure.positions_bohr
        if structure.excited_state == 0:
            energy, gradient = surface(x, 0)
            return EngineResult(energy - 1.0, gradient=gradient)
        s = self._order(x)[min(structure.excited_state, 2) - 1]
        energy, gradient = surface(x, s)
        return EngineResult(energy, gradient=gradient)

    def state_characters(self, structure: Structure, count: int) -> dict[str, Any] | None:
        x = structure.positions_bohr
        order = self._order(x)[:count]
        amplitudes = np.zeros((2, len(order)))
        for k, s in enumerate(order):
            amplitudes[s, k] = 1.0
        return {
            "orbitals": np.eye(3),
            "occupied": 1,
            "energies": [surface(x, s)[0] + 1.0 for s in order],
            "amplitudes": amplitudes,
        }
