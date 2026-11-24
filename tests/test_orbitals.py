"""Orbital analyses from Python: following orbitals, natural reaction orbitals, reactive
orbitals, and the orbitals the engines hand out.

MultiOptPy: `WFAnalyzer/MO_analysis.py:NROAnalysis`.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mopt_rs import _core
from mopt_rs.calculators import CalculatorOptions, create
from mopt_rs.calculators.base import Structure

ANG = _core.ANGSTROM_TO_BOHR


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def model(x: float, electrons: int = 2) -> tuple[list[list[float]], list[float], list[float]]:
    """Orbitals of ``H(x) = H0 + x H1`` in five orthonormal atomic orbitals."""
    h0 = np.diag([-2.0, -1.1, -0.4, 0.5, 1.4])
    i, j = np.indices((5, 5))
    h1 = 0.3 * np.sin((i + 1.0) * (j + 2.0) + (j + 1.0) * (i + 2.0))
    energies, vectors = np.linalg.eigh(h0 + 0.05 * (h1 - np.diag(np.diag(h1))) + x * h1)
    occupations = [1.0 if k < electrons else 0.0 for k in range(5)]
    return vectors.tolist(), energies.tolist(), occupations


def test_following_and_natural_reaction_orbitals_of_a_model() -> None:
    unit = np.eye(5).tolist()
    reference = model(0.0)
    step = 1e-4
    plus, minus = model(step), model(-step)
    # The displaced orbitals in another order and with other signs.
    order = [2, 0, 4, 1, 3]
    signs = np.array([1.0, -1.0, -1.0, 1.0, -1.0])
    shuffled = (
        (np.asarray(plus[0])[:, order] * signs).tolist(),
        [plus[1][k] for k in order],
        [plus[2][k] for k in order],
    )
    followed = _core.follow_orbitals(unit, reference, shuffled)
    assert followed["source"] == [1, 3, 0, 4, 2]
    assert np.allclose(followed["coefficients"], plus[0], atol=1e-12) or np.allclose(
        np.abs(followed["coefficients"]), np.abs(plus[0]), atol=1e-12
    )
    assert np.allclose(followed["energies"], plus[1])
    assert min(followed["overlaps"]) > 0.999999

    found = _core.natural_reaction_orbitals(unit, reference, shuffled, minus, step)
    # First-order perturbation theory: U_ai = (C^T H1 C)_ai / (e_i - e_a).
    c = np.asarray(reference[0])
    e = np.asarray(reference[1])
    h1 = (
        np.asarray(model(1.0)[0]) @ np.diag(model(1.0)[1]) @ np.asarray(model(1.0)[0]).T
        - c @ np.diag(e) @ c.T
    )
    coupled = c.T @ h1 @ c
    exact = coupled[2:, :2] / (e[None, :2] - e[2:, None])
    singular = np.linalg.svd(exact, compute_uv=False)
    assert np.allclose(found["singular_values"], singular, rtol=1e-5)
    assert found["sum_of_squares"] == pytest.approx(float((exact**2).sum()), rel=1e-5)
    assert sum(found["shares"]) == pytest.approx(1.0)
    assert found["occupied_indices"] == [0, 1] and found["virtual_indices"] == [2, 3, 4]
    assert found["weakest_overlap"] > 0.999999
    assert np.asarray(found["occupied"]).shape == (5, 2)

    changes, occupied, unoccupied = _core.reactive_orbitals(
        [-0.5, -0.4, 0.1, 0.3], [-0.6, -0.38, 0.25, 0.29], [2.0, 2.0, 0.0, 0.0]
    )
    assert (occupied, unoccupied) == (0, 2)
    assert changes[0] == pytest.approx(-0.2 / 1.1)
    dx = _core.mass_weighted_displacement([1.0, 16.0], [1.0, 0, 0, -1.0, 0, 0], 0.02)
    assert np.sqrt(sum(m * v * v for m, v in zip([1, 1, 1, 16, 16, 16], dx, strict=True))) == (
        pytest.approx(0.02)
    )
    with pytest.raises(ValueError):
        _core.natural_reaction_orbitals(unit, reference, plus, minus, 0.0)


def water() -> Structure:
    return Structure.build(
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]]) * ANG,
    )


def check_orbitals(found: dict[str, Any] | None, sets: int, electrons: float) -> None:
    assert found is not None
    overlap = found["overlap"]
    assert len(found["sets"]) == sets
    total = 0.0
    for one in found["sets"]:
        c = one["coefficients"]
        assert np.allclose(c.T @ overlap @ c, np.eye(c.shape[1]), atol=1e-8)
        assert np.all(np.diff(one["energies"]) >= -1e-10)
        total += float(one["occupations"].sum())
    assert total == pytest.approx(electrons, abs=1e-6)


def test_built_in_engines_and_force_fields() -> None:
    # A force field has no orbitals to hand out; the built-in GFN0-xTB has.
    calc = create("lj")
    argon = Structure.build(["Ar", "Ar"], [[0.0, 0.0, 0.0], [7.2, 0.0, 0.0]])
    assert calc.orbitals(argon) is None
    gfn0 = create("gfn0xtb")
    check_orbitals(gfn0.orbitals(water()), 1, 8.0)
    # The atom of every atomic orbital, the atoms in order (the projection method needs it).
    found = gfn0.orbitals(water())
    assert found is not None
    atoms = found["atom_of_ao"]
    assert len(atoms) == found["overlap"].shape[0]
    assert atoms == sorted(atoms) and set(atoms) == {0, 1, 2}
    radical = Structure.build(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]], multiplicity=2)
    check_orbitals(gfn0.orbitals(radical), 2, 7.0)
    # The pairs of a bond stretch from these orbitals, step by step in Python.
    x = water().positions_bohr.ravel()
    direction = np.zeros(9)
    direction[3:6] = x[3:6] - x[0:3]
    masses = [15.999, 15.999, 15.999, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008]
    step = 0.01
    dx = np.asarray(_core.mass_weighted_displacement(masses[::3], direction.tolist(), step))

    def sets(shift: float) -> tuple[Any, Any]:
        moved = Structure.build(["O", "H", "H"], (x + shift * dx).reshape(3, 3))
        found = gfn0.orbitals(moved)
        assert found is not None
        one = found["sets"][0]
        triple = (
            one["coefficients"].tolist(),
            one["energies"].tolist(),
            one["occupations"].tolist(),
        )
        return found["overlap"].tolist(), triple

    overlap, reference = sets(0.0)
    nro = _core.natural_reaction_orbitals(overlap, reference, sets(1.0)[1], sets(-1.0)[1], step)
    assert len(nro["singular_values"]) == 4 and nro["singular_values"][0] > 0.1
    assert nro["weakest_overlap"] > 0.9999


@pytest.mark.skipif(not has("pyscf"), reason="PySCF not installed")
def test_pyscf_hands_out_orbitals() -> None:
    calc = create("pyscf", CalculatorOptions(method="hf", basis="sto-3g"))
    check_orbitals(calc.orbitals(water()), 1, 10.0)
    # STO-3G: five atomic orbitals on oxygen, one on each hydrogen.
    found = calc.orbitals(water())
    assert found is not None
    assert found["atom_of_ao"] == [0, 0, 0, 0, 0, 1, 2]
    radical = Structure.build(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]], multiplicity=2)
    check_orbitals(calc.orbitals(radical), 2, 9.0)


@pytest.mark.skipif(not has("tblite"), reason="tblite not installed")
def test_tblite_hands_out_orbitals() -> None:
    calc = create("tblite:GFN2-xTB")
    check_orbitals(calc.orbitals(water()), 1, 8.0)


def inversion(path: Path, heights: list[float]) -> str:
    text = ""
    for z in heights:
        text += f"4\n\nN 0.0 0.0 {z}\nH 0.94 0.0 0.0\nH -0.47 0.814 0.0\nH -0.47 -0.814 0.0\n"
    path.write_text(text)
    return str(path)


def sums(directory: Path) -> list[float]:
    rows = (directory / "nro.csv").read_text().splitlines()[1:]
    return [float(row.split(",")[3]) for row in rows]


def run_directory(tmp_path: Path) -> Path:
    logs = sorted(tmp_path.rglob("mopt.log"))
    assert len(logs) == 1
    return logs[0].parent


@pytest.mark.parametrize(
    ("engine", "arguments"),
    [
        ("gfn0xtb", ["-os", "gfn0xtb"]),
        pytest.param(
            "pyscf",
            ["-os", "pyscf", "-func", "hf", "-bs", "sto-3g"],
            marks=pytest.mark.skipif(not has("pyscf"), reason="PySCF not installed"),
        ),
        pytest.param(
            "tblite",
            ["-xtb", "GFN2-xTB"],
            marks=pytest.mark.skipif(not has("tblite"), reason="tblite not installed"),
        ),
    ],
)
def test_command_along_a_path(
    engine: str, arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = inversion(tmp_path / "nh3.xyz", [0.4, 0.2, 0.0, -0.2, -0.4])
    code = _core.run_cli("freqmain", [path, *arguments, "-nro", "-roet"])
    assert code == 0, engine
    directory = run_directory(tmp_path)
    values = sums(directory)
    assert len(values) == 5 and all(v > 0.0 for v in values)
    # Mirror frames respond alike.
    assert values[0] == pytest.approx(values[4], rel=1e-3)
    assert values[1] == pytest.approx(values[3], rel=1e-3)
    report = (directory / "reactive_orbitals.txt").read_text()
    assert "occupied reactive orbital" in report
    assert (directory / "orbital_energies.csv").exists()
    assert "natural reaction orbitals" in (directory / "mopt.log").read_text()
