"""The vibrations transverse to a reaction path from Python (:func:`mopt_rs.path_profile`):
on the model surfaces of the Rust tests of ``vib::path`` (a valley that turns into a ridge, a
circular valley) the numbers those tests check, and along the inversion of ammonia with the
built-in GFN-FF the numbers ``freqmain -path`` writes.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import mopt_rs
from mopt_rs import _core
from mopt_rs.optimize import NativeCalculator


def _ridge(x: float) -> tuple[list[float], np.ndarray, list[float]]:
    """One atom on ``E = -x^3 / 3 + (1 - x) y^2 / 2 + z^2 / 2`` at ``(x, 0, 0)``: position,
    Hessian, gradient (``vib::path`` tests, a valley for ``x < 1`` and a ridge beyond)."""
    return [x, 0.0, 0.0], np.diag([-2.0 * x, 1.0 - x, 1.0]), [-x * x, 0.0, 0.0]


def test_the_valley_ridge_transition_of_the_model_surface() -> None:
    xs = [0.2 * k for k in range(1, 10)]
    points = [_ridge(x) for x in xs]
    profile = mopt_rs.path_profile(
        ["H"],
        [p for p, _, _ in points],
        hessians=[h for _, h, _ in points],
        gradients=[g for _, _, g in points],
        project=False,
    )
    # Arc lengths are mass-weighted distances: sqrt(m) times 0.2 Bohr per step.
    mass = (profile.arc_length[1] / 0.2) ** 2
    assert mass == pytest.approx(1.008, abs=1e-3)
    assert profile.arc_length[8] == pytest.approx(np.sqrt(mass) * 1.6, abs=1e-10)
    assert profile.tangent_source == ("gradient",) * 9
    assert all(len(w) == 2 for w in profile.wavenumbers)
    # The lowest transverse curvature is (1 - x) / m, along y.
    assert np.allclose(profile.lowest_curvature, (1.0 - np.asarray(xs)) / mass, atol=1e-10)
    assert np.all(np.abs(profile.lowest_mode[:, 0, 1]) > 0.9 * np.abs(profile.lowest_mode[:, 0, 0]))
    assert np.allclose(profile.overlap, 1.0, atol=1e-10)
    # One transition, at x = 1, unstable along y.
    assert len(profile.bifurcations) == 1
    found = profile.bifurcations[0]
    # The point at x = 1 (index 4) has the curvature 0, the last that is not negative.
    assert found.after == 4
    assert found.arc_length == pytest.approx(np.sqrt(mass) * 0.8, abs=1e-6 * np.sqrt(mass))
    assert found.curvature_before >= 0.0 > found.curvature_after
    assert found.curvature_before == profile.lowest_curvature[4]
    assert found.curvature_after == profile.lowest_curvature[5]
    assert abs(found.mode[0, 1]) > 1e-3 and abs(found.mode[0, 0]) < 1e-10
    # A straight path: no curvature and no couplings; the tangent points along +x although
    # the gradient points along -x, and the gradient is along the path.
    assert np.allclose(profile.curvature, 0.0, atol=1e-10)
    assert np.allclose(profile.tangent[:, 0], 1.0, atol=1e-12)
    assert all(np.allclose(b, 0.0, atol=1e-10) for b in profile.curvature_couplings)
    assert profile.gradient_alignment is not None
    assert np.allclose(profile.gradient_alignment, 1.0, atol=1e-12)
    # A path that stays in the valley has none.
    valley = mopt_rs.path_profile(
        ["H"],
        [p for p, _, _ in points[:4]],
        hessians=[h for _, h, _ in points[:4]],
        gradients=[g for _, _, g in points[:4]],
        project=False,
    )
    assert valley.bifurcations == ()


def test_the_circular_valley_and_the_tangents_from_the_points() -> None:
    # The floor of E = (r - R)^2 / 2 + z^2 in the xy plane: the gradient vanishes there, so
    # the tangents come from the points (or are given).
    radius = 2.0
    angles = np.arange(9) * 0.1
    positions = [[radius * np.cos(a), radius * np.sin(a), 0.0] for a in angles]
    hessians = []
    for a in angles:
        e = np.array([np.cos(a), np.sin(a), 0.0])
        hessians.append(np.outer(e, e) + np.diag([0.0, 0.0, 2.0]))
    from_points = mopt_rs.path_profile(
        ["H"], positions, hessians=hessians, gradients=[[0.0] * 3] * 9, project=False
    )
    given = mopt_rs.path_profile(
        ["H"],
        positions,
        hessians=hessians,
        tangent="given",
        tangents=[[-np.sin(a), np.cos(a), 0.0] for a in angles],
        project=False,
    )
    assert from_points.tangent_source == ("points",) * 9
    assert given.tangent_source == ("given",) * 9
    mass = (given.arc_length[1] / (radius * 2.0 * np.sin(0.05))) ** 2
    expected = 1.0 / (np.sqrt(mass) * radius)
    assert np.allclose(given.curvature, expected, rtol=0.01)
    for b, c in zip(given.curvature_couplings, given.curvature, strict=True):
        assert abs(abs(b[0]) - c) < 1e-8 and abs(b[1]) < 1e-10
    assert np.allclose(given.zero_point_energy, given.zero_point_energy[0], atol=1e-12)
    assert all(t == (0, 1) for t in given.tracks)
    # The gradient vanishes: no alignment.
    assert from_points.gradient_alignment is not None
    assert np.all(np.isnan(from_points.gradient_alignment))
    assert from_points.tangents_from_points is not None
    assert np.allclose(from_points.curvature, expected, rtol=0.02)


def test_wrong_input_is_refused() -> None:
    x, h, g = _ridge(0.2)
    with pytest.raises(ValueError, match="Hessians of the points or a calculator"):
        mopt_rs.path_profile(["H"], [x])
    with pytest.raises(ValueError, match="one array of 9 numbers"):
        mopt_rs.path_profile(["H"], [x, x], hessians=[h])
    with pytest.raises(ValueError, match="coincide"):
        mopt_rs.path_profile(["H"], [x, x], hessians=[h, h], gradients=[g, g])
    with pytest.raises(ValueError, match="tangent must be"):
        mopt_rs.path_profile(["H"], [x], hessians=[h], gradients=[g], tangent="sideways")
    with pytest.raises(ValueError, match="needs the tangents"):
        mopt_rs.path_profile(["H"], [x], hessians=[h], tangent="given")


def _inversion(path: Path, heights: list[float]) -> list[np.ndarray]:
    """Ammonia with its nitrogen at the heights (Angstrom) above the plane of the hydrogens:
    the frames written to ``path`` and their positions in Bohr."""
    text, frames = "", []
    for z in heights:
        rows = [("N", 0.0, 0.0, z), ("H", 0.94, 0.0, 0.0), ("H", -0.47, 0.814, 0.0)]
        rows.append(("H", -0.47, -0.814, 0.0))
        text += "4\n\n" + "".join(f"{s} {x} {y} {w}\n" for s, x, y, w in rows)
        frames.append(np.array([r[1:] for r in rows]) * _core.ANGSTROM_TO_BOHR)
    path.write_text(text)
    return frames


def test_the_numbers_of_freqmain_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    frames = _inversion(tmp_path / "nh3.xyz", [0.45, 0.3, 0.15, 0.0, -0.15, -0.3])
    assert _core.run_cli("freqmain", ["nh3.xyz", "-os", "gfnff", "-path"]) == 0
    (log,) = sorted(tmp_path.rglob("mopt.log"))
    with (log.parent / "path_profile.csv").open() as f:
        rows = list(csv.DictReader(f))
    profile = mopt_rs.path_profile(
        ["N", "H", "H", "H"], frames, calculator=NativeCalculator("gfnff")
    )
    assert profile.energies is not None and len(rows) == 6
    for k, row in enumerate(rows):
        assert float(row["arc_length_sqrt_amu_bohr"]) == pytest.approx(
            profile.arc_length[k], abs=1e-8
        )
        assert float(row["energy_hartree"]) == pytest.approx(profile.energies[k], abs=1e-11)
        assert float(row["zero_point_energy_hartree"]) == pytest.approx(
            profile.zero_point_energy[k], abs=1e-11
        )
        assert float(row["path_curvature"]) == pytest.approx(profile.curvature[k], rel=1e-7)
        assert float(row["lowest_transverse_cm-1"]) == pytest.approx(
            profile.lowest_wavenumber[k], abs=1e-4
        )
        assert float(row["overlap_with_previous"]) == pytest.approx(profile.overlap[k], abs=1e-6)
        listed = [float(w) for w in row["transverse_cm-1"].split()]
        assert np.allclose(listed, profile.wavenumbers[k], atol=1e-4)
        if row["gradient_alignment"]:
            assert profile.gradient_alignment is not None
            assert float(row["gradient_alignment"]) == pytest.approx(
                profile.gradient_alignment[k], abs=1e-6
            )
    report = (log.parent / "bifurcations.txt").read_text()
    if profile.bifurcations:
        for b in profile.bifurcations:
            assert f"between points {b.after} and {b.after + 1} at arc length {b.arc_length:.6f}" in (
                report
            )
    else:
        assert report.startswith("No valley-ridge transition")
