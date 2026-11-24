"""Rust components one by one from Python: coordinate systems, superposition, the bond graph,
sampling priorities, and electron-transfer rates.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import numpy as np
import pytest

from mopt_rs import _core

ANG = _core.ANGSTROM_TO_BOHR
WATER = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]) * ANG
SYMBOLS = ["O", "H", "H"]


def dimer() -> tuple[list[str], np.ndarray]:
    second = WATER + np.array([3.0, 0.2, 0.3]) * ANG
    return SYMBOLS * 2, np.vstack([WATER, second])


def test_coordinate_systems_one_by_one() -> None:
    names = _core.coordinate_names()
    assert {"cart", "ric", "dic", "tric", "tric-p", "auto"} <= set(names)
    symbols, x = dimer()
    flat = x.ravel().tolist()
    for name in names:
        c = _core.Coordinates(name, symbols, flat)
        n = c.dimension
        q = np.asarray(c.values(flat))
        b = np.asarray(c.wilson_b(flat)).reshape(n, 18)
        assert q.shape == (n,) and np.all(np.isfinite(b)), name
        # The Wilson matrix is the derivative of the displacement.
        rng = np.random.default_rng(1)
        dx = 1e-5 * rng.standard_normal(18)
        dq = np.asarray(c.displacement((x.ravel() + dx).tolist(), (x.ravel() - dx).tolist()))
        assert np.allclose(dq, 2.0 * b @ dx, atol=1e-9), name
        # A step in the coordinates and back.
        step = 0.02 * rng.standard_normal(n)
        if name in {"ric", "tric-p", "spherical", "hyperspherical"}:
            # Redundant sets (spherical coordinates about the centroid: by the translations):
            # only steps in the range of the Wilson matrix can be realized, and a first-order
            # one is realized up to its second-order remainder.
            step = b @ (2e-3 * rng.standard_normal(18))
        moved = x.ravel() + np.asarray(c.cartesian_step(flat, step.tolist()))
        got = np.asarray(c.displacement(moved.tolist(), flat))
        assert np.allclose(got, step, atol=1e-5), name
        assert not c.needs_reinitialization(flat)
    assert {"spherical", "hyperspherical"} <= set(names)
    assert _core.Coordinates("polar", symbols, flat).name == "hyperspherical"
    assert _core.Coordinates("auto", symbols, flat).name == "auto (tric-cart)"
    assert _core.Coordinates("auto", SYMBOLS, WATER.ravel().tolist()).name.startswith("auto (")
    with pytest.raises(ValueError, match="unknown coordinate system"):
        _core.Coordinates("no_such_system", symbols, flat)


def test_tric_asks_for_new_coordinates_after_a_quarter_turn() -> None:
    symbols, x = dimer()
    c = _core.Coordinates("tric", symbols, x.ravel().tolist())
    center = x[3:].mean(axis=0)
    angle = 1.8
    rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1.0]]
    )
    turned = x.copy()
    turned[3:] = (x[3:] - center) @ rot.T + center
    flat = turned.ravel().tolist()
    assert c.needs_reinitialization(flat)
    c.reinitialize(flat)
    assert not c.needs_reinitialization(flat)


def test_superposition_and_bond_graph() -> None:
    symbols, x = dimer()
    angle = 0.7
    rot = np.array(
        [[1.0, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )
    moved = x @ rot.T + np.array([1.0, -2.0, 0.5])
    aligned, rmsd = _core.align(symbols, moved.ravel().tolist(), x.ravel().tolist())
    assert rmsd < 1e-8
    assert np.allclose(np.asarray(aligned).reshape(-1, 3), x, atol=1e-8)
    graph = _core.connectivity(symbols, x.ravel().tolist())
    assert sorted(map(tuple, graph["bonds"])) == [(1, 2), (1, 3), (4, 5), (4, 6)]
    assert sorted(map(tuple, graph["fragments"])) == [(1, 2, 3), (4, 5, 6)]
    assert len(graph["angles"]) == 2 and graph["dihedrals"] == []


def test_sampling_priorities_and_electron_transfer() -> None:
    names = _core.sampling_priority_names()
    assert {"boltzmann", "tabu", "water_filling"} <= set(names)
    energies = [0.0, 0.002, 0.02]
    for name in names:
        rule = _core.SamplingPriority(name, seed=7)
        picks = [rule.select(energies, [0, 0, 0]) for _ in range(20)]
        assert all(p is None or 0 <= p < 3 for p in picks), name
        # The same seed gives the same choices.
        again = _core.SamplingPriority(name, seed=7)
        assert picks == [again.select(energies, [0, 0, 0]) for _ in range(20)], name
    # At room temperature a minimum 0.02 Hartree up is practically never chosen.
    rule = _core.SamplingPriority("boltzmann", seed=1)
    assert 2 not in {rule.select(energies, [0, 0, 0]) for _ in range(200)}
    with pytest.raises(ValueError):
        _core.SamplingPriority("boltzmann").select([], [])

    r = _core.electron_transfer_rate(0.02, 0.0, -0.01, 0.002)
    assert r["reorganization_energy"] > 0.0
    assert 0.0 < r["landau_zener_probability"] <= 1.0
    assert 0.0 < r["transmission_coefficient"] <= 1.0
    assert r["barrier"] == pytest.approx(r["nonadiabatic_barrier"] - 0.002)
    slower = _core.electron_transfer_rate(0.03, 0.0, -0.01, 0.002)
    assert slower["rate_constant"] < r["rate_constant"]


def test_maps_of_a_series_of_structures() -> None:
    # Water with one O-H bond stretched step by step.
    frames = []
    for k in range(6):
        x = WATER.copy()
        x[1] += 0.15 * k * (x[1] - x[0]) / np.linalg.norm(x[1] - x[0])
        frames.append(x.ravel().tolist())
    d = np.asarray(_core.structure_distances(SYMBOLS, frames))
    assert d.shape == (6, 6) and np.allclose(d, d.T) and np.all(np.diag(d) == 0.0)
    # The structures lie nearly on a line (the superposition moves the heavy atom a
    # little with every step): nearly equal steps, and one coordinate holds nearly all.
    assert np.allclose(np.diff(d[0]), d[0, 1], rtol=1e-3)
    cmds = _core.classical_mds(d.tolist(), 2)
    points = np.asarray(cmds["coordinates"])
    assert points.shape == (6, 2)
    assert cmds["contributions"][0] > 0.999
    assert cmds["reproducibility"] == pytest.approx(1.0, abs=1e-3)
    assert points[0, 0] < 0.0 and np.all(np.diff(points[:, 0]) > 0.0)
    assert np.allclose(np.abs(points[:, 0] - points[0, 0]), d[0], rtol=1e-3, atol=1e-4)
    # The map has the distances it was made from.
    on_map = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    assert np.allclose(on_map, d, atol=1e-3)

    values, names = _core.path_variables("distances", SYMBOLS, frames)
    assert names == ["1 O - 2 H", "1 O - 3 H", "2 H - 3 H"]
    pca = _core.principal_components(values)
    assert pca["contributions"][0] > 0.999
    # The stretched bond and, with it, the H-H distance; the other bond stays.
    vector = pca["vectors"][0]
    assert vector[0] > vector[2] > 0.0 and abs(vector[1]) < 1e-9
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert np.asarray(pca["scores"]).shape == (6, len(pca["variances"]))
    z, z_names = _core.path_variables("z_matrix", SYMBOLS, frames)
    assert z_names == ["2 H - 1 O", "3 H - 2 H", "3 H - 2 H - 1 O"]
    cartesian, c_names = _core.path_variables("cartesian", SYMBOLS, frames)
    assert len(c_names) == 9 and np.asarray(cartesian).shape == (6, 9)
    with pytest.raises(ValueError, match="kind"):
        _core.path_variables("angles", SYMBOLS, frames)
    with pytest.raises(ValueError):
        _core.classical_mds([[0.0, 1.0], [2.0, 0.0]])


def test_dynamic_modes_of_iterations() -> None:
    # A decaying, alternating sequence: x(k + 1) = -0.6 x(k) in one direction.
    direction = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
    snapshots = [((-0.6) ** k * direction).tolist() for k in range(8)]
    modes = _core.dynamic_modes(snapshots, 1e-8)
    assert modes is not None and modes["rank"] == 1
    assert modes["eigenvalues"][0] == pytest.approx((-0.6, 0.0), abs=1e-10)
    assert modes["behavior"][0].startswith("damping, alternating sign")
    mode = np.asarray(modes["modes"][0])[:, 0]
    assert abs(abs(mode @ direction) / np.linalg.norm(mode) - 1.0) < 1e-10
    assert _core.dynamic_modes([[1.0, 1.0]] * 4) is not None
    assert _core.dynamic_modes([[0.0, 0.0]] * 4) is None
    psi = _core.koopman_observables([0.0, 0.0, 0.0, 0.0, 0.0, 2.0], [0.1] * 6)
    assert len(psi) == 6 + 6 + 1 + 6 and psi[12] == pytest.approx(2.0)


def test_hyperspherical_coordinates_of_a_point() -> None:
    x = [0.4, -1.1, 0.7, 0.2, -0.3]
    q = _core.to_hyperspherical(x)
    assert np.allclose(_core.from_hyperspherical(q), x, atol=1e-13)
    assert abs(q[0] - np.linalg.norm(x)) < 1e-14
    n = len(x)
    jacobian = np.asarray(_core.hyperspherical_jacobian(q)).reshape(n, n)
    wilson = np.asarray(_core.hyperspherical_wilson(x)).reshape(n, n)
    assert np.allclose(wilson @ jacobian, np.eye(n), atol=1e-10)
    # f = |x|^2 / 2: all of the gradient is along the radius.
    g = _core.gradient_to_hyperspherical(x, x)
    assert abs(g[0] - q[0]) < 1e-13 and np.allclose(g[1:], 0.0, atol=1e-13)
    with pytest.raises((ValueError, RuntimeError)):
        _core.to_hyperspherical([1.0])
