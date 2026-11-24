"""The system of a model system of ONIOM cut out of an OpenMM system (link atoms in the
place of cut bonds); skipped where OpenMM is not installed.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from mopt_rs.calculators.base import BOHR_TO_ANGSTROM, CalculatorError, Structure

openmm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

from mopt_rs.calculators.openmm import OpenMMEngine  # noqa: E402
from mopt_rs.calculators.openmm_model import (  # noqa: E402
    LINK_EPSILON_KJ,
    LINK_SIGMA_NM,
    model_system,
)

NM_PER_BOHR = BOHR_TO_ANGSTROM / 10.0
COULOMB = 138.935456  # kJ nm / (mol e^2), the constant of OpenMM

# A chain of four carbons (nm), not in a plane.
CHAIN = np.array([[0.0, 0.0, 0.0], [0.153, 0.0, 0.0], [0.21, 0.14, 0.0], [0.36, 0.15, 0.05]])
BOND = (0.15, 2.0e5)
ANGLE = (1.9, 400.0)
TORSION = (3, 0.2, 5.0)
CHARGES = (-0.3, 0.2, 0.25, -0.15)
SIGMA, EPSILON = 0.34, 0.45
SCALE_14 = (0.8333, 0.5)


def chain_system() -> Any:
    """Bonds, angles, one torsion, charges and Lennard-Jones wells, with the exceptions of a
    force field: nothing between atoms one or two bonds apart, scaled between the ends."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.011)
    bonds = openmm.HarmonicBondForce()
    for a in range(3):
        bonds.addBond(a, a + 1, *BOND)
    angles = openmm.HarmonicAngleForce()
    for a in range(2):
        angles.addAngle(a, a + 1, a + 2, *ANGLE)
    torsions = openmm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2, 3, *TORSION)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for q in CHARGES:
        nonbonded.addParticle(q, SIGMA, EPSILON)
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]:
        nonbonded.addException(a, b, 0.0, SIGMA, 0.0)
    nonbonded.addException(
        0, 3, SCALE_14[0] * CHARGES[0] * CHARGES[3], SIGMA, SCALE_14[1] * EPSILON
    )
    for force in (bonds, angles, torsions, nonbonded):
        system.addForce(force)
    system.addForce(openmm.CMMotionRemover())
    return system


def energy(system: Any, positions_nm: np.ndarray) -> float:
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    return math.acos(float(u @ v) / (distance(a, b) * distance(c, b)))


def twist(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    return math.atan2(float(np.cross(n1, n2) @ b2) / float(np.linalg.norm(b2)), float(n1 @ n2))


def harmonic(value: float, rest: float, constant: float) -> float:
    return 0.5 * constant * (value - rest) ** 2


def test_two_atoms_and_a_link_atom() -> None:
    # The first two carbons, with a link atom at 0.7 of the bond that is cut.
    g = 0.7
    model = model_system(chain_system(), [0, 1], [(1, 2, g)])
    assert model.getNumParticles() == 3
    assert model.getParticleMass(2).value_in_unit(unit.dalton) == pytest.approx(1.008)
    # The motion remover of the whole system is left out.
    assert {type(f).__name__ for f in model.getForces()} == {
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
    }
    link = CHAIN[1] + g * (CHAIN[2] - CHAIN[1])
    positions = np.array([CHAIN[0], CHAIN[1], link])
    # The bond of the region, the cut bond on the link atom with its length scaled, and
    # the angle of the outer atom on the link atom. The two ends are two bonds apart:
    # nothing between them.
    expected = (
        harmonic(distance(CHAIN[0], CHAIN[1]), *BOND)
        + harmonic(distance(CHAIN[1], link), g * BOND[0], BOND[1])
        + harmonic(bend(CHAIN[0], CHAIN[1], link), *ANGLE)
    )
    assert energy(model, positions) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    # The link bond follows the cut bond: g^2 times its energy, whatever its length.
    cut = harmonic(distance(CHAIN[1], CHAIN[2]), *BOND)
    assert harmonic(distance(CHAIN[1], link), g * BOND[0], BOND[1]) == pytest.approx(g * g * cut)


def test_three_atoms_with_the_torsion_and_the_scaled_pair() -> None:
    d = 2.06  # Bohr: a link atom at a fixed distance
    model = model_system(chain_system(), [0, 1, 2], [(2, 3, d)], "distance")
    direction = (CHAIN[3] - CHAIN[2]) / distance(CHAIN[3], CHAIN[2])
    link = CHAIN[2] + d * NM_PER_BOHR * direction
    positions = np.array([CHAIN[0], CHAIN[1], CHAIN[2], link])
    r = distance(CHAIN[0], link)
    sigma = 0.5 * (SIGMA + LINK_SIGMA_NM)
    well = SCALE_14[1] * math.sqrt(EPSILON * LINK_EPSILON_KJ)
    periodicity, phase, constant = TORSION
    expected = (
        harmonic(distance(CHAIN[0], CHAIN[1]), *BOND)
        + harmonic(distance(CHAIN[1], CHAIN[2]), *BOND)
        # The link atom is where its bond wants it.
        + 0.0
        + harmonic(bend(CHAIN[0], CHAIN[1], CHAIN[2]), *ANGLE)
        + harmonic(bend(CHAIN[1], CHAIN[2], link), *ANGLE)
        + constant
        * (1.0 + math.cos(periodicity * twist(CHAIN[0], CHAIN[1], CHAIN[2], link) - phase))
        # The pair three bonds apart: the well of the two atoms scaled as that of the
        # whole system, and no charge on the link atom.
        + 4.0 * well * ((sigma / r) ** 12 - (sigma / r) ** 6)
    )
    assert energy(model, positions) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    # All atoms and no link: the system itself.
    whole = model_system(chain_system(), [0, 1, 2, 3])
    assert energy(whole, CHAIN) == pytest.approx(energy(chain_system(), CHAIN), rel=1e-12)
    # In another order of the atoms: the same energy at the same places.
    turned = model_system(chain_system(), [3, 2, 1, 0])
    assert energy(turned, CHAIN[::-1]) == pytest.approx(energy(chain_system(), CHAIN), rel=1e-12)
    q = COULOMB * SCALE_14[0] * CHARGES[0] * CHARGES[3] / distance(CHAIN[0], CHAIN[3])
    assert abs(q) > 1.0  # the pair of the whole system has its charges


def test_two_cut_bonds_at_one_atom() -> None:
    # The second carbon alone: two link atoms, the angle between them, nothing else.
    g = 0.72
    model = model_system(chain_system(), [1], [(1, 0, g), (1, 2, g)])
    first = CHAIN[1] + g * (CHAIN[0] - CHAIN[1])
    second = CHAIN[1] + g * (CHAIN[2] - CHAIN[1])
    expected = (
        harmonic(distance(CHAIN[1], first), g * BOND[0], BOND[1])
        + harmonic(distance(CHAIN[1], second), g * BOND[0], BOND[1])
        + harmonic(bend(first, CHAIN[1], second), *ANGLE)
    )
    assert energy(model, np.array([CHAIN[1], first, second])) == pytest.approx(expected, rel=1e-9)


def test_what_cannot_be_cut_out_is_said() -> None:
    with pytest.raises(CalculatorError, match="does not lead out of the model"):
        model_system(chain_system(), [0, 1], [(2, 3, 0.7)])
    with pytest.raises(CalculatorError, match="beyond the 4"):
        model_system(chain_system(), [0, 7])
    with pytest.raises(CalculatorError, match="twice"):
        model_system(chain_system(), [0, 0])
    other = chain_system()
    other.addForce(openmm.CustomExternalForce("x"))
    with pytest.raises(CalculatorError, match="CustomExternalForce"):
        model_system(other, [0, 1], [(1, 2, 0.7)])


def test_the_engine_cuts_the_model_out_of_the_whole_system() -> None:
    g = 0.7
    engine = OpenMMEngine(
        system=chain_system(), model_atoms=[0, 1], model_links=[(1, 2, g)], platform="Reference"
    )
    link = CHAIN[1] + g * (CHAIN[2] - CHAIN[1])
    positions = np.array([CHAIN[0], CHAIN[1], link])
    structure = Structure.build(["C", "C", "H"], (positions / NM_PER_BOHR).ravel().tolist())
    found = engine.compute(structure, False)
    expected = energy(model_system(chain_system(), [0, 1], [(1, 2, g)]), positions)
    assert found.energy == pytest.approx(expected, rel=1e-10)
    assert any("cut out of" in note for note in engine.notes)
    # The whole structure is still the whole structure.
    whole = Structure.build(["C"] * 4, (CHAIN / NM_PER_BOHR).ravel().tolist())
    assert engine.compute(whole, False).energy == pytest.approx(
        energy(chain_system(), CHAIN), rel=1e-10
    )
    # A structure that is neither has no system.
    with pytest.raises(CalculatorError, match="no system for the 2 atoms"):
        engine.compute(Structure.build(["C", "C"], [0.0, 0.0, 0.0, 2.9, 0.0, 0.0]), False)


def test_oniom_with_a_link_atom_and_openmm_on_both_levels(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same force field as the high and as the low level: the two model terms cancel,
    # whatever the link atom is, when both calculators are told what the model system is
    # made of and cut it out of the whole system. The energy is that of the whole chain.
    from mopt_rs import _core
    from mopt_rs.calculators.base import HARTREE_TO_KJMOL

    monkeypatch.chdir(tmp_path)
    whole = tmp_path / "whole.xml"
    whole.write_text(openmm.XmlSerializer.serialize(chain_system()))
    xyz = tmp_path / "chain.xyz"
    rows = "".join(f"C {10 * x:.8f} {10 * y:.8f} {10 * z:.8f}\n" for x, y, z in CHAIN)
    xyz.write_text(f"4\n\n{rows}")
    arguments = [str(xyz), "-os", "openmm", "-calc_opt", f"system={whole}", "platform=Reference"]
    arguments += ["-oniom", "1-2", "auto", "openmm", "-ns", "1"]
    arguments += ["-oniom_opt", f"low.system={whole}", "low.platform=Reference", "micro=off"]
    assert _core.run_cli("optmain", arguments) == 0
    run = next(tmp_path.rglob("chain_OPT_ASE_*"))
    first = float((run / "energy_profile.csv").read_text().splitlines()[1].split(",")[3])
    assert first == pytest.approx(energy(chain_system(), CHAIN) / HARTREE_TO_KJMOL, abs=1e-9)
    log = (run / "mopt.log").read_text()
    assert "1 link atom" in log or "link" in log.lower()
