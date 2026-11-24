"""The vibrations transverse to a reaction path and where the path bifurcates.

:func:`path_profile` takes the points of a path (an IRC, a scan, the images of a band) with
their Hessians, or with a calculator that gives them, and returns what ``freqmain -path``
writes to ``path_profile.csv`` and ``bifurcations.txt``: the arc length, the tangents, how
far the gradient runs along the path, the curvature of the path, the generalized normal modes
orthogonal to it at every point (wavenumbers, curvatures, curvature couplings, zero-point
energy), the lowest of them followed along the path, and the valley-ridge transitions: where
the lowest transverse curvature turns from positive to negative, the valley becomes a ridge,
and the path bifurcates (the two branches leave along the mode that turns unstable).

Units: positions Bohr, gradients Hartree/Bohr, Hessians Hartree/Bohr^2, arc lengths
sqrt(amu) Bohr, curvatures of the modes Hartree / (Bohr^2 amu), wavenumbers cm^-1 (negative:
imaginary), energies Hartree.

MultiOptPy: `PESAnalyzer/calc_irc_curvature.py:calc_irc_curvature_properties` (the tangent,
the curvature, and the curvature couplings); the transverse modes followed along the path and
the valley-ridge transitions are new (``mopt_rs::vib::transverse_profile``).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mopt_rs import _core
from mopt_rs.optimize import Calculator, Task

__all__ = ["PathBifurcation", "PathProfile", "path_profile"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PathBifurcation:
    """A valley-ridge transition between the points ``after`` and ``after + 1``."""

    after: int
    """Index (from 0) of the last point with a positive lowest transverse curvature."""
    arc_length: float
    """Where that curvature is zero by linear interpolation (sqrt(amu) Bohr)."""
    curvature_before: float
    """The lowest transverse curvature at ``after``."""
    curvature_after: float
    """The lowest transverse curvature at ``after + 1`` (negative)."""
    mode: FloatArray
    """Cartesian displacement of the mode that turns unstable, shape (N, 3): the two branches
    leave along ``+mode`` and ``-mode``."""


@dataclass(frozen=True)
class PathProfile:
    """The path and its transverse vibrations, point by point (see the module)."""

    arc_length: FloatArray
    """Shape (P,), from the first point."""
    energies: FloatArray | None
    """Energies of the points where a calculator gave them or they were given."""
    tangent: FloatArray
    """Unit tangents in mass-weighted coordinates from the first point towards the last,
    shape (P, 3N)."""
    tangent_source: tuple[str, ...]
    """Where each tangent came from: ``gradient``, ``points``, or ``given``."""
    tangents_from_points: FloatArray | None
    """The Cartesian tangents ``dx/ds`` from the points alone, shape (P, 3N) (two points or
    more)."""
    gradient_alignment: FloatArray | None
    """Overlap (0 to 1) of each gradient with the tangent from the points: 1 on a
    steepest-descent path; NaN where the gradient vanishes (``None`` without gradients)."""
    curvature: FloatArray
    """Scalar curvature of the path, shape (P,)."""
    curvature_vector: FloatArray
    """Its vector (mass-weighted), shape (P, 3N)."""
    wavenumbers: tuple[FloatArray, ...]
    """Transverse wavenumbers at every point, ascending."""
    curvatures: tuple[FloatArray, ...]
    """Their curvatures (eigenvalues of the projected mass-weighted Hessian)."""
    curvature_couplings: tuple[FloatArray, ...]
    """``B_kF`` of the transverse modes, in the order of ``wavenumbers``."""
    zero_point_energy: FloatArray
    """Of the transverse modes with real frequencies."""
    lowest_wavenumber: FloatArray
    lowest_curvature: FloatArray
    lowest_mode: FloatArray
    """Cartesian displacement of the lowest mode at every point, shape (P, N, 3)."""
    overlap: FloatArray
    """Of the lowest mode with that of the point before (1 for the first)."""
    tracks: tuple[tuple[int, ...], ...]
    """For every point, the index into ``wavenumbers`` of each mode followed from the first."""
    mode_couplings: tuple[FloatArray, ...]
    """``B_kl`` between the followed modes, a square matrix per point."""
    bifurcations: tuple[PathBifurcation, ...]
    """The valley-ridge transitions found."""

    @staticmethod
    def from_dict(
        found: dict[str, Any], n_atoms: int, energies: FloatArray | None = None
    ) -> PathProfile:
        """From the dict of ``mopt_rs._core.transverse_profile``."""

        def rows(key: str) -> FloatArray:
            return np.asarray(found[key], dtype=np.float64)

        def ragged(key: str) -> tuple[FloatArray, ...]:
            return tuple(np.asarray(v, dtype=np.float64) for v in found[key])

        couplings = []
        for flat in found["mode_couplings"]:
            m = np.asarray(flat, dtype=np.float64)
            size = int(round(np.sqrt(m.size)))
            couplings.append(m.reshape(size, size))
        from_points = found["tangents_from_points"]
        alignment = found["gradient_alignment"]
        return PathProfile(
            arc_length=rows("arc_length"),
            energies=energies,
            tangent=rows("tangent"),
            tangent_source=tuple(found["tangent_source"]),
            tangents_from_points=None
            if from_points is None
            else np.asarray(from_points, dtype=np.float64),
            gradient_alignment=None if alignment is None else np.asarray(alignment, np.float64),
            curvature=rows("curvature"),
            curvature_vector=rows("curvature_vector"),
            wavenumbers=ragged("wavenumbers"),
            curvatures=ragged("curvatures"),
            curvature_couplings=ragged("curvature_couplings"),
            zero_point_energy=rows("zero_point_energy"),
            lowest_wavenumber=rows("lowest_wavenumber"),
            lowest_curvature=rows("lowest_curvature"),
            lowest_mode=rows("lowest_mode").reshape(-1, n_atoms, 3),
            overlap=rows("overlap"),
            tracks=tuple(tuple(int(i) for i in t) for t in found["tracks"]),
            mode_couplings=tuple(couplings),
            bifurcations=tuple(
                PathBifurcation(
                    after=int(b["after"]),
                    arc_length=float(b["arc_length"]),
                    curvature_before=float(b["curvature_before"]),
                    curvature_after=float(b["curvature_after"]),
                    mode=np.asarray(b["mode"], dtype=np.float64).reshape(n_atoms, 3),
                )
                for b in found["bifurcations"]
            ),
        )


def _flat_rows(values: Sequence[Any] | None, count: int, n3: int, what: str) -> list[list[float]]:
    if values is None:
        return []
    out = [np.asarray(v, dtype=np.float64).ravel() for v in values]
    if len(out) != count or any(v.size != n3 for v in out):
        raise ValueError(f"{what}: one array of {n3} numbers for each of the {count} points")
    return [v.tolist() for v in out]


def path_profile(
    symbols: Sequence[str],
    positions_bohr: Sequence[Any],
    *,
    hessians: Sequence[Any] | None = None,
    gradients: Sequence[Any] | None = None,
    calculator: Calculator | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    tangent: str = "gradient",
    tangents: Sequence[Any] | None = None,
    project: bool = True,
    stationary_rms: float = 3e-4,
) -> PathProfile:
    """The transverse vibrations along the path through ``positions_bohr`` (one structure,
    shape (N, 3) or (3N,), per point, in a common orientation as an IRC gives them).

    The Hessians are ``hessians`` (one (3N, 3N) array per point) or, without them, those of
    ``calculator`` (anything with ``evaluate(Task)``: :class:`mopt_rs.NativeCalculator`, an
    engine of :func:`mopt_rs.calculators.create`), which gives the gradients and energies as
    well. The tangent at a point is, as ``freqmain -path_tangent`` chooses it: with
    ``tangent="gradient"`` (a steepest-descent path) the gradient where its RMS is above
    ``stationary_rms`` Hartree/Bohr and the line through the neighbouring points elsewhere;
    with ``"points"`` that line everywhere (a scan, a band); with ``"given"`` the displacements
    ``tangents``. ``project`` removes translations and rotations (not for a surface fixed in
    space)."""
    points = [np.asarray(x, dtype=np.float64).ravel() for x in positions_bohr]
    count = len(points)
    n3 = 3 * len(symbols)
    if count == 0 or any(x.size != n3 for x in points):
        raise ValueError(f"positions_bohr: one array of {n3} numbers per point, at least one")
    energies: FloatArray | None = None
    if hessians is None:
        if calculator is None:
            raise ValueError("path_profile needs the Hessians of the points or a calculator")
        found_h, found_g, found_e = [], [], []
        for k, x in enumerate(points):
            ev = calculator.evaluate(
                Task(
                    label=f"path_point_{k}",
                    calculator="python",
                    symbols=list(symbols),
                    positions_bohr=x.reshape(-1, 3),
                    charge=charge,
                    multiplicity=multiplicity,
                    need_hessian=True,
                )
            )
            if ev.hessian is None:
                raise ValueError(f"the calculator gave no Hessian at path point {k}")
            found_h.append(np.asarray(ev.hessian, dtype=np.float64))
            found_g.append(np.asarray(ev.gradient, dtype=np.float64))
            found_e.append(float(ev.energy))
        hessians = found_h
        if gradients is None:
            gradients = found_g
        energies = np.asarray(found_e, dtype=np.float64)
    elif calculator is not None:
        raise ValueError("give the Hessians or a calculator, not both")
    h = _flat_rows(hessians, count, n3 * n3, "hessians")
    g = _flat_rows(gradients, count, n3, "gradients") or None
    t = _flat_rows(tangents, count, n3, "tangents") or None
    found = _core.transverse_profile(
        list(symbols),
        [x.tolist() for x in points],
        h,
        gradients=g,
        tangent=tangent,
        tangents=t,
        project=project,
        stationary_rms=stationary_rms,
    )
    return PathProfile.from_dict(found, len(symbols), energies)
