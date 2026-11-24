"""The orientation search from Python: the parts of a complex, the comparison of structures
with exchangeable atoms and molecules, the ask/tell search with a Python-driven optimizer, and
the ``orientsearch`` command through the Python registry.

MultiOptPy: `Entrypoints/orientation_search.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import mopt_rs
from mopt_rs import _core
from mopt_rs.optimize import NativeCalculator

ANG = _core.ANGSTROM_TO_BOHR
WATER = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
SYMBOLS = ["O", "H", "H"] * 2


def dimer(offset: float = 4.0) -> np.ndarray:
    return np.vstack([WATER, WATER + np.array([offset, 0.0, 0.0])]) * ANG


def _superposed_rmsd(a: np.ndarray, b: np.ndarray, mirrored: bool) -> float:
    """RMSD of ``b`` superposed on ``a`` by a rotation (by a rotation and a reflection when
    ``mirrored``), atom by atom."""
    p, q = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(q.T @ p)
    d = np.sign(np.linalg.det(u @ vt))
    if mirrored:
        d = -d
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((q @ r - p) ** 2, axis=1))))


def test_parts_and_the_comparison_of_structures() -> None:
    x = dimer()
    parts = _core.complex_parts(SYMBOLS, x.ravel().tolist())
    assert parts["atoms"] == [[1, 2, 3], [4, 5, 6]] and parts["fixed"] == [False, False]
    given = _core.complex_parts(SYMBOLS, x.ravel().tolist(), [[1, 2, 3]], fixed=[2])
    assert given["atoms"] == [[1, 2, 3], [4, 5, 6]] and given["fixed"] == [False, True]
    with pytest.raises(ValueError, match="two parts"):
        _core.complex_parts(SYMBOLS[:3], (WATER * ANG).ravel().tolist())
    # The molecules swapped and the whole turned: the same structure.
    angle = 0.8
    rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    swapped = np.vstack([x[3:], x[:3]]) @ rot.T + np.array([1.0, 2.0, -0.5])
    found = mopt_rs.orient.compare_structures(SYMBOLS, x.ravel().tolist(), swapped.ravel().tolist())
    assert found["same_bonds"] and found["rmsd"] < 1e-8
    # The dimer of two planar molecules has several exact matches ([4, 5, 6, 1, 2, 3] by the
    # rotation, [1, 3, 2, 4, 6, 5] by a half turn about the axis between the oxygens, the
    # identity with a reflection), and which one is found depends on rounding: the one found
    # must superpose the structures (with a reflection where it says so) and keep the molecules.
    p = found["permutation"]
    assert sorted(p) == [1, 2, 3, 4, 5, 6]
    assert {frozenset(p[:3]), frozenset(p[3:])} == {frozenset({1, 2, 3}), frozenset({4, 5, 6})}
    assert _superposed_rmsd(x, swapped[np.asarray(p) - 1], found["mirrored"]) < 1e-8
    proper = mopt_rs.orient.compare_structures(
        SYMBOLS, x.ravel().tolist(), swapped.ravel().tolist(), mirror=False
    )
    assert not proper["mirrored"] and proper["rmsd"] < 1e-8
    assert _superposed_rmsd(x, swapped[np.asarray(proper["permutation"]) - 1], False) < 1e-8
    # A proton on the other molecule: other bonds.
    moved = x.copy()
    moved[1] = x[3] + np.array([-0.95, 0.0, 0.2]) * ANG
    other = _core.compare_structures(SYMBOLS, x.ravel().tolist(), moved.ravel().tolist())
    assert not other["same_bonds"] and other["rmsd"] is None


def test_an_ask_tell_search_with_a_python_optimizer() -> None:
    calc = NativeCalculator("gfnff")
    result = mopt_rs.orientation_search(
        SYMBOLS,
        dimer(),
        calc,
        samples=4,
        seed=3,
        optimizer={"criteria": "tight"},
    )
    assert len(result.samples) == 4 and result.evaluations > 0
    assert all(s["converged"] for s in result.samples)
    assert result.minima, result.samples
    energies = [m.energy for m in result.minima]
    assert energies == sorted(energies)
    assert sum(m.weight for m in result.minima) == pytest.approx(1.0)
    assert sum(m.hits for m in result.minima) == 4
    # The lowest minimum is the hydrogen-bonded dimer: one O-H...O contact near 1.9 Angstrom.
    x = result.minima[0].positions_bohr / ANG
    contacts = [np.linalg.norm(x[h] - x[o]) for h, o in [(1, 3), (2, 3), (4, 0), (5, 0)]]
    assert 1.7 < min(contacts) < 2.2, contacts
    # The search itself: the same seed gives the same starts.
    first = _core.OrientationSearch(SYMBOLS, dimer().ravel().tolist(), seed=5).ask()
    again = _core.OrientationSearch(SYMBOLS, dimer().ravel().tolist(), seed=5).ask()
    assert first is not None and again is not None
    assert first["positions_bohr"] == again["positions_bohr"]
    with pytest.raises(ValueError, match="scheme"):
        _core.OrientationSearch(SYMBOLS, dimer().ravel().tolist(), scheme="nope")


def test_the_command_through_the_python_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lines = ["6", "water dimer"] + [
        f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for s, p in zip(SYMBOLS, dimer() / ANG, strict=True)
    ]
    (tmp_path / "w2.xyz").write_text("\n".join(lines) + "\n", encoding="utf-8")
    code = _core.run_cli(
        "orientsearch", ["w2.xyz", "-os", "gfnff", "-nsample", "3", "-orient_opt", "seed=2"]
    )
    assert code == 0
    runs = list(tmp_path.glob("*/w2_ORIENT_*"))
    assert len(runs) == 1
    run = runs[0]
    assert (run / "w2_Sample_0.xyz").exists() and (run / "w2_Sample_2_OPT").is_dir()
    with (run / "orientsearch_minima.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["rank"] == "1" and float(rows[0]["interaction_kcal_mol"]) < -3.0
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    assert "seed=2" in settings["orient_options"]
    assert (run / "w2_minimum_1.xyz").exists() and (run / "w2_minima.xyz").exists()
