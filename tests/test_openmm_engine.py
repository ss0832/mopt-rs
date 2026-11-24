"""OpenMM as an engine (``-os openmm``) and as a lower level of ONIOM; skipped where OpenMM
is not installed.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mopt_rs import _core
from mopt_rs.calculators import CalculatorOptions, create
from mopt_rs.calculators.base import HARTREE_TO_KJMOL, CalculatorError, Structure

openmm = pytest.importorskip("openmm")

from mopt_rs.calculators.openmm import OpenMMEngine  # noqa: E402

BOHR_PER_ANGSTROM = _core.ANGSTROM_TO_BOHR
KJ_PER_HARTREE = HARTREE_TO_KJMOL

Atom = tuple[str, tuple[float, float, float]]

# Two water molecules (Angstrom): the residues, atom names, and order of the PDB files.
DIMER: list[Atom] = [
    ("O", (0.000, 0.000, 0.000)),
    ("H", (0.957, 0.000, 0.000)),
    ("H", (-0.240, 0.927, 0.000)),
    ("O", (2.900, 0.000, 0.000)),
    ("H", (3.500, 0.750, 0.000)),
    ("H", (3.500, -0.750, 0.000)),
]


def _pdb(atoms: list[Atom]) -> str:
    lines = []
    for k, (symbol, (x, y, z)) in enumerate(atoms):
        name = "O" if symbol == "O" else f"H{k % 3}"
        lines.append(
            f"HETATM{k + 1:5d}  {name:<3s} HOH A{k // 3 + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00           {symbol}"
        )
    return "\n".join([*lines, "END", ""])


def _structure(atoms: list[Atom]) -> Structure:
    symbols = [s for s, _ in atoms]
    positions = np.array([p for _, p in atoms]) * BOHR_PER_ANGSTROM
    return Structure.build(symbols, positions)


def _moved(atoms: list[Atom], atom: int, axis: int, by: float) -> list[Atom]:
    """``atoms`` with one coordinate of one atom changed by ``by`` (Angstrom)."""
    out = []
    for k, (symbol, position) in enumerate(atoms):
        q = list(position)
        if k == atom:
            q[axis] += by
        out.append((symbol, (q[0], q[1], q[2])))
    return out


def _serialized_water(directory: Path) -> Path:
    """A water molecule with two harmonic bonds and a harmonic angle, as a serialized system."""
    system = openmm.System()
    for mass in (15.999, 1.008, 1.008):
        system.addParticle(mass)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.1, 400000.0)
    bonds.addBond(0, 2, 0.1, 400000.0)
    angles = openmm.HarmonicAngleForce()
    angles.addAngle(1, 0, 2, math.radians(104.5), 500.0)
    system.addForce(bonds)
    system.addForce(angles)
    path = directory / "water.xml"
    path.write_text(openmm.XmlSerializer.serialize(system))
    return path


def test_a_serialized_system(tmp_path: Path) -> None:
    path = _serialized_water(tmp_path)
    water: list[Atom] = [("O", (0.0, 0.0, 0.0)), ("H", (1.1, 0.0, 0.0)), ("H", (-0.2, 0.9, 0.0))]
    engine = OpenMMEngine(system=str(path), platform="Reference")
    found = engine.compute(_structure(water), False)
    # The energy of the two bonds and the angle (kJ/mol; OpenMM's lengths are nanometres).
    x = np.array([p for _, p in water]) * 0.1
    r1, r2 = np.linalg.norm(x[1] - x[0]), np.linalg.norm(x[2] - x[0])
    theta = math.acos(float(np.dot(x[1] - x[0], x[2] - x[0]) / (r1 * r2)))
    by_hand = (
        0.5 * 400000.0 * ((r1 - 0.1) ** 2 + (r2 - 0.1) ** 2)
        + 0.5 * 500.0 * (theta - math.radians(104.5)) ** 2
    )
    assert found.energy == pytest.approx(by_hand, rel=1e-10)
    # The forces (kJ/mol per Angstrom) are those of the energy.
    forces = np.asarray(found.forces)
    step = 1.0e-4
    for atom in range(3):
        for axis in range(3):
            plus = engine.compute(_structure(_moved(water, atom, axis, step)), False).energy
            minus = engine.compute(_structure(_moved(water, atom, axis, -step)), False).energy
            assert forces[atom, axis] == pytest.approx(-(plus - minus) / (2.0 * step), abs=1e-4)
    # By its name, in atomic units, with a Hessian from differences of its forces.
    calculator = create(
        "openmm", CalculatorOptions(extra={"system": str(path), "platform": "Reference"})
    )
    evaluation = calculator.evaluate_structure(_structure(water), need_hessian=True)
    assert evaluation.energy == pytest.approx(by_hand / KJ_PER_HARTREE, rel=1e-9)
    hessian = np.asarray(evaluation.hessian)
    assert hessian.shape == (9, 9) and np.allclose(hessian, hessian.T, atol=1e-6)
    assert "platform Reference" in calculator.describe()


def test_a_force_field_on_topologies(tmp_path: Path) -> None:
    dimer, single = tmp_path / "dimer.pdb", tmp_path / "water.pdb"
    dimer.write_text(_pdb(DIMER))
    single.write_text(_pdb(DIMER[:3]))
    engine = OpenMMEngine(
        forcefield="tip3p.xml", topology=f"{dimer},{single}", platform="Reference"
    )
    # Each structure takes the system of its atoms.
    two = engine.compute(_structure(DIMER), False)
    one = engine.compute(_structure(DIMER[:3]), False)
    other = engine.compute(_structure(DIMER[3:]), False)
    assert np.asarray(two.forces).shape == (6, 3) and np.asarray(one.forces).shape == (3, 3)
    # The interaction of the two molecules: attractive at this hydrogen-bond distance.
    assert two.energy < one.energy + other.energy
    # Bonds are flexible: a stretched bond costs energy.
    stretched: list[Atom] = [*DIMER[:1], ("H", (1.2, 0.0, 0.0)), *DIMER[2:3]]
    assert engine.compute(_structure(stretched), False).energy > one.energy + 10.0
    # Atoms without a system, and the atoms of a system in another order.
    ammonia: list[Atom] = [("N", (0.0, 0.0, 0.0))]
    ammonia += [("H", (1.0, float(k), 0.0)) for k in range(3)]
    with pytest.raises(CalculatorError, match="model system"):
        engine.compute(_structure(ammonia), False)
    with pytest.raises(CalculatorError, match="no system"):
        engine.compute(_structure([DIMER[1], DIMER[0], DIMER[2]]), False)
    # What is missing is said.
    with pytest.raises(CalculatorError, match="forcefield needs topology"):
        OpenMMEngine(forcefield="tip3p.xml")
    with pytest.raises(CalculatorError, match="system=FILE.xml"):
        OpenMMEngine()


def test_openmm_below_a_built_in_calculator_in_oniom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dimer, single = tmp_path / "dimer.pdb", tmp_path / "water.pdb"
    dimer.write_text(_pdb(DIMER))
    single.write_text(_pdb(DIMER[:3]))
    text = "6\n\n" + "".join(f"{s} {x} {y} {z}\n" for s, (x, y, z) in DIMER)
    low = [
        "low.forcefield=tip3p.xml",
        f"low.topology={dimer},{single}",
        "low.platform=Reference",
    ]

    def run(name: str, extra: list[str]) -> Path:
        work = tmp_path / name
        work.mkdir()
        (work / "dimer.xyz").write_text(text)
        arguments = [str(work / "dimer.xyz"), "-os", "gfnff", "-oniom", "1-3", "none", "openmm"]
        assert _core.run_cli("optmain", [*arguments, "-oniom_opt", *low, *extra]) == 0
        return next(p for p in work.parent.rglob("dimer_OPT_ASE_*") if name in p.parts)

    # Every atom in the optimizer: the first energy is that of the starting structure,
    # E_low(whole) + E_high(model) - E_low(model), with the two calculators by themselves.
    plain = run("plain", ["micro=off", "-ns", "1"])
    first = float((plain / "energy_profile.csv").read_text().splitlines()[1].split(",")[3])
    engine = OpenMMEngine(
        forcefield="tip3p.xml", topology=f"{dimer},{single}", platform="Reference"
    )
    whole = engine.compute(_structure(DIMER), False).energy / KJ_PER_HARTREE
    model_low = engine.compute(_structure(DIMER[:3]), False).energy / KJ_PER_HARTREE
    model_high = create("gfnff").evaluate_structure(_structure(DIMER[:3])).energy
    assert first == pytest.approx(whole + model_high - model_low, abs=1e-8)

    # With microiterations (the default) the second molecule is relaxed with OpenMM before
    # every evaluation, and the energy goes down.
    relaxed = run("relaxed", ["-ns", "8"])
    log = (relaxed / "mopt.log").read_text()
    assert "ONIOM microiterations: 3 core atom(s), 3 environment atom(s)" in log
    rows = (relaxed / "energy_profile.csv").read_text().splitlines()[1:]
    energies = [float(row.split(",")[3]) for row in rows]
    assert energies[-1] < energies[0] < first
