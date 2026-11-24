"""Scans from Python: ReSPer coordinates and their pre-images, constrained optimizations of
plain variables, the points and stationary points of a scan, rigid structures, and the
relaxedscan command.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import mopt_rs
from mopt_rs import ResperMap, VectorOptimizer, _core

MB_MINIMA = [(-0.558224, 1.441726), (0.623499, 0.028038), (-0.050011, 0.466694)]
ANGSTROM = 1.0 / 0.529177210903


def bent_curve() -> list[list[float]]:
    return [[2.0 * s - 1.0, 0.8 * (2.0 * s - 1.0) ** 2] for s in np.linspace(0.0, 1.0, 9)]


def test_resper_coordinates_of_points_and_their_preimages() -> None:
    curve = bent_curve()
    m = ResperMap.from_points(curve)
    assert m.components == 2
    assert abs(sum(m.contributions) - 1.0) < 1e-12
    # Gower's formula gives every reference its own coordinates back.
    for p, c in zip(curve, m.coordinates, strict=True):
        values, gradients = m.project(p)
        assert np.allclose(values, c, atol=1e-10)
        assert len(gradients) == 2 and len(gradients[0]) == 2
    # The gradient is that of the coordinates.
    x, h = [0.2, 0.5], 1e-6
    _, g = m.project(x)
    for j in range(2):
        a, b = list(x), list(x)
        a[j] += h
        b[j] -= h
        numeric = (m.project(a)[0][0] - m.project(b)[0][0]) / (2 * h)
        assert abs(numeric - g[0][j]) < 1e-6
    # A structure for a point between two references lands on it after the correction.
    t = 0.5 * (m.coordinates[3][0] + m.coordinates[4][0])
    raw = m.preimage([0], [t], correct=False)
    fixed = m.preimage([0], [t])
    assert not raw["extrapolated"] and fixed["remaining"] < 1e-8
    assert m.preimage([0, 1], [10.0, 10.0], correct=False)["extrapolated"]
    with pytest.raises(ValueError):
        m.preimage([2], [0.0])


def test_resper_map_of_structures() -> None:
    frames = []
    for k in range(7):
        phi = 0.25 * k
        frames.append(
            [0.0, 0.0, 0.0, 1.8, 0.0, 0.0, -0.5, 1.7, 0.0]
            + [2.4, 1.6 * math.cos(phi), 1.6 * math.sin(phi) + 0.1 * k]
        )
    m = ResperMap(["O", "H", "H", "C"], frames)
    assert m.symbols == ["O", "H", "H", "C"]
    values, _ = m.project(frames[2])
    assert np.allclose(values, m.coordinates[2], atol=1e-8)
    p = m.preimage([0], [m.coordinates[3][0]])
    assert len(p["x"]) == 12 and p["remaining"] < 1e-8


def test_a_constrained_optimization_of_plain_variables() -> None:
    # The lowest point of the Muller-Brown surface at x = -0.5.
    opt = VectorOptimizer([-0.4, 1.3])
    opt.hold(variables=[(0, -0.5)])
    while (task := opt.ask()) is not None:
        x, need_hessian = task
        value, gradient, _ = _core.evaluate_objective("mb2d", x)
        opt.tell(value, gradient)
    assert opt.termination == "converged"
    assert abs(opt.x[0] + 0.5) < 1e-12
    assert abs(opt.x[1] - MB_MINIMA[0][1]) < 0.05
    # On a ReSPer coordinate of a path.
    m = ResperMap.from_points(bent_curve())
    target = m.coordinates[4][0]
    opt = VectorOptimizer([0.1, 0.0])
    opt.hold(resper=[(m, 0, target)])
    while (task := opt.ask()) is not None:
        x, _ = task
        value = (x[0] - 0.3) ** 2 + (x[1] - 0.2) ** 2
        opt.tell(value, [2 * (x[0] - 0.3), 2 * (x[1] - 0.2)])
    assert abs(m.project(opt.x)[0][0] - target) < 1e-8


def test_scan_points_and_stationary_points() -> None:
    s = _core.scan_points([(0.0, 1.0), (0.0, 2.0)], [4, 3], grid=True)
    assert s["shape"] == [4, 3]
    order = s["order"]
    assert sorted(order) == list(range(12))
    # Snake order: every point starts from the one before it, a neighbour on the grid.
    assert s["start"][0] is None and s["start"][1:] == order[:-1]
    joint = _core.scan_points([(1.3, 2.6)], [8])
    assert joint["values"][7] == [pytest.approx(2.6)]
    # A saddle of x^2 - y^2 off the grid points.
    nx, ny = 9, 7
    energies = [((p % nx) - 4.3) ** 2 - ((p // nx) - 2.8) ** 2 for p in range(nx * ny)]
    found = _core.scan_stationary_points([nx, ny], energies)
    assert [c["kind"] for c in found] == ["saddle"]
    assert found[0]["offset"] == pytest.approx([0.3, -0.2])
    # A threefold torsion over one whole turn: the minimum at the ends counts once.
    torsion = [1.0 + math.cos(3.0 * math.radians(-180.0 + 30.0 * k)) for k in range(13)]
    kinds = [c["kind"] for c in _core.scan_stationary_points([13], torsion, periodic=[True])]
    assert kinds.count("minimum") == 3 and kinds.count("saddle") == 3


def test_a_rigid_structure_turns_a_methyl_group() -> None:
    symbols = ["C", "C", "H", "H", "H", "H", "H", "H"]
    xyz = [
        (0.0, 0.0, 0.0),
        (1.53, 0.0, 0.0),
        (-0.39, 1.03, 0.0),
        (-0.39, -0.51, 0.89),
        (-0.39, -0.51, -0.89),
        (1.92, -1.03, 0.0),
        (1.92, 0.51, 0.89),
        (1.92, 0.51, -0.89),
    ]
    flat = [c * ANGSTROM for p in xyz for c in p]
    moved = np.array(
        _core.rigid_scan_structure(symbols, flat, [("dihedral", "3,1,2,6", 60.0)])
    ).reshape(-1, 3)
    start = np.array(flat).reshape(-1, 3)
    for a, b in [(0, 1), (1, 5), (5, 6), (0, 2)]:
        assert abs(np.linalg.norm(moved[a] - moved[b]) - np.linalg.norm(start[a] - start[b])) < 1e-9


def test_the_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mb.txt").write_text("x -0.5\ny 1.4\n")
    out = mopt_rs.run_relaxedscan(
        ["mb.txt", "-os", "mb2d", "-scan", "var", "x", "-0.9,0.7", "-nsample", "9"],
        console=False,
    )
    assert not out["interrupted"]
    assert len(out["points"]) == 9 and all(p["converged"] for p in out["points"])
    assert any(c["kind"] == "minimum" for c in out["candidates"])
    assert (Path(out["directory"]) / "scan_points.csv").is_file()
