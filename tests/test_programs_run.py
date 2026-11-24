"""MOPAC, GPAW, and Quantum ESPRESSO run for real through their engines. Each test is
skipped where its program is not installed.

MultiOptPy: `Calculator/ase_tools/mopac.py`, `Calculator/gpaw_calculation_tools.py`,
`Calculator/pwscf_calculation_tools.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mopt_rs import _core
from mopt_rs.calculators import (
    CalculatorOptions,
    EngineCalculator,
    FileProgramEngine,
    Structure,
    create,
)

HAS_ASE = importlib.util.find_spec("ase") is not None
needs_mopac = pytest.mark.skipif(
    not (HAS_ASE and shutil.which("mopac")), reason="MOPAC (and ASE) not installed"
)
needs_gpaw = pytest.mark.skipif(
    not (HAS_ASE and importlib.util.find_spec("gpaw")), reason="GPAW not installed"
)
needs_espresso = pytest.mark.skipif(
    not (HAS_ASE and shutil.which("pw.x") and shutil.which("ld1.x")),
    reason="Quantum ESPRESSO (pw.x, ld1.x) not installed",
)

ANG = 1.0 / 0.529177210903
WATER = np.array([[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]]) * ANG


@pytest.fixture(autouse=True)
def in_an_empty_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The programs write their files where they run."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("MOPT_CALCULATOR_OPTIONS", raising=False)
    monkeypatch.chdir(tmp_path)


def water(x: np.ndarray, charge: int = 0, multiplicity: int = 1) -> Structure:
    return Structure.build(["O", "H", "H"], x, charge=charge, multiplicity=multiplicity)


def slope(
    calc: EngineCalculator,
    build: Callable[[np.ndarray], Structure],
    x: np.ndarray,
    index: int,
    step: float,
) -> float:
    """dE/dx of coordinate ``index`` from two energies."""
    plus, minus = x.copy(), x.copy()
    plus.flat[index] += step
    minus.flat[index] -= step
    high = calc.evaluate_structure(build(plus)).energy
    low = calc.evaluate_structure(build(minus)).energy
    return float(high - low) / (2.0 * step)


def optimized(tmp_path: Path, stem: str) -> tuple[np.ndarray, str]:
    """The structure (Angstrom) a run of ``optmain`` ended with, and its log."""
    newest = max(tmp_path.rglob(f"{stem}_optimized.xyz"), key=lambda p: p.stat().st_mtime)
    rows = [line.split() for line in newest.read_text().splitlines()[2:] if line.strip()]
    xyz = np.array([[float(v) for v in row[1:4]] for row in rows])
    return xyz, (newest.parent / "mopt.log").read_text()


def engine_atoms(calc: EngineCalculator, structure: Structure) -> Any:
    """The ASE structure with the engine's calculator, to ask the program more than the
    engine does."""
    engine = calc.engine
    assert isinstance(engine, FileProgramEngine)
    atoms = engine.atoms(structure)
    atoms.calc = engine.calculator(structure.state)
    return atoms


@needs_mopac
def test_mopac_energy_gradient_spin_and_keywords(tmp_path: Path) -> None:
    calc = create("mopac")
    ev = calc.evaluate_structure(water(WATER))
    # PM7 heat of formation of water near -58 kcal/mol at this structure.
    assert -0.11 < ev.energy < -0.08
    assert np.abs(ev.gradient.sum(axis=0)).max() < 1e-9
    # MOPAC's own gradients are good to a few 1e-5 Hartree/Bohr with these settings.
    for index in (2, 4, 5):
        numeric = slope(calc, water, WATER, index, 0.01)
        assert ev.gradient.flat[index] == pytest.approx(numeric, abs=5e-5)

    # The doublet cation: an unrestricted calculation, some 12 eV above the molecule.
    def cation(x: np.ndarray) -> Structure:
        return water(x, charge=1, multiplicity=2)

    ion = calc.evaluate_structure(cation(WATER))
    first = (tmp_path / "mopac.mop").read_text().splitlines()[0]
    assert "CHARGE=1" in first and "DOUBLET UHF" in first
    assert 10.0 < (ion.energy - ev.energy) * 27.2114 < 14.0
    assert ion.gradient.flat[2] == pytest.approx(slope(calc, cation, WATER, 2, 0.01), abs=5e-5)

    # Keywords ASE has no parameter for reach the program: water in water is more stable
    # by a few kcal/mol.
    wet = create("mopac", CalculatorOptions(threads=1, extra={"keywords": "EPS=78.4"}))
    solvation = (wet.evaluate_structure(water(WATER)).energy - ev.energy) * _core.HARTREE_TO_KCAL_MOL
    first = (tmp_path / "mopac.mop").read_text().splitlines()[0]
    assert "EPS=78.4" in first and "THREADS=1" in first
    assert -12.0 < solvation < -4.0

    # Another method by -func, and the Hessian from differences of the forces.
    am1 = create("mopac", CalculatorOptions(method="AM1"))
    assert abs(am1.evaluate_structure(water(WATER)).energy - ev.energy) > 1e-4
    hessian = calc.evaluate_structure(water(WATER), True).hessian
    assert hessian is not None and hessian.shape == (9, 9)
    assert np.abs(hessian - hessian.T).max() < 1e-12
    # A program that is not there is an error that names it.
    missing = create("mopac", CalculatorOptions(extra={"command": "no_such_mopac PREFIX.mop"}))
    with pytest.raises(Exception, match="mopac"):
        missing.evaluate_structure(water(WATER))


@needs_mopac
def test_mopac_optimization_from_the_command_line(tmp_path: Path) -> None:
    (tmp_path / "water.xyz").write_text(
        "3\n\nO 0.0 0.0 0.119\nH 0.0 0.763 -0.477\nH 0.0 -0.763 -0.477\n"
    )
    # Defaults kept in a file: tighter SCF and gradients for the run.
    (tmp_path / "mopt_calculator_options.json").write_text(
        json.dumps({"calculators": {"mopac": {"keywords": "PRECISE", "relscf": 0.01}}})
    )
    code = _core.run_cli("optmain", ["water.xyz", "-os", "mopac", "-opt", "rsirfo_bfgs"])
    assert code == 0
    xyz, log = optimized(tmp_path, "water")
    assert "mopt_calculator_options.json: keywords, relscf" in log
    assert "MOPAC" in log  # the reference to cite
    first = next(tmp_path.rglob("mopac.mop")).read_text().splitlines()[0]
    assert "PRECISE" in first and "RELSCF=0.01" in first
    oh = [float(np.linalg.norm(xyz[k] - xyz[0])) for k in (1, 2)]
    assert all(0.93 < d < 0.99 for d in oh), oh
    cosine = float(np.dot(xyz[1] - xyz[0], xyz[2] - xyz[0])) / (oh[0] * oh[1])
    assert 100.0 < np.degrees(np.arccos(cosine)) < 110.0
    # At the minimum the Hessian (differences of MOPAC's forces) has three vibrations, the
    # ones MOPAC finds itself with its FORCE calculation at the same structure.
    calc = create("mopac", CalculatorOptions(extra={"keywords": "PRECISE"}))
    x = xyz * ANG
    hessian = calc.evaluate_structure(water(x), True).hessian
    assert hessian is not None
    modes = _core.normal_modes(["O", "H", "H"], x.ravel().tolist(), hessian.ravel().tolist())
    wavenumbers = np.sort(np.asarray(modes["wavenumbers"]))
    assert np.count_nonzero(np.abs(wavenumbers) < 1.0) == 6
    rows = [
        f"{s} {r[0]:.10f} 1 {r[1]:.10f} 1 {r[2]:.10f} 1" for s, r in zip("OHH", xyz, strict=True)
    ]
    (tmp_path / "force.mop").write_text("\n".join(["PM7 FORCE PRECISE", "water", "", *rows]) + "\n")
    subprocess.run(["mopac", "force.mop"], capture_output=True, check=True)
    report = (tmp_path / "force.out").read_text().splitlines()
    own = sorted(float(line.split()[1]) for line in report if line.strip().startswith("FREQUENCY"))
    assert len(own) == 3
    assert np.allclose(wavenumbers[-3:], own, atol=2.0), (wavenumbers[-3:], own)


@needs_gpaw
def test_gpaw_molecule_in_a_box() -> None:
    options = CalculatorOptions(method="PBE", extra={"h": 0.25, "basis": "dzp", "vacuum": 4.0})
    calc = create("gpaw", options)
    engine = calc.engine
    assert isinstance(engine, FileProgramEngine)
    ev = calc.evaluate_structure(water(WATER))
    assert np.isfinite(ev.energy)
    assert engine._box is not None
    cell = engine._box[1].copy()
    # A structure that leaves the symmetry of the first one, in the same box (GPAW would
    # stop at it with its point-group symmetry on), and the slope of the energy.
    numeric = slope(calc, water, WATER, 5, 0.02)
    assert ev.gradient.flat[5] == pytest.approx(numeric, abs=2e-4)
    assert np.array_equal(engine._box[1], cell)
    # The forces of the two hydrogens mirror each other.
    assert ev.gradient[1, 1] == pytest.approx(-ev.gradient[2, 1], abs=1e-6)
    assert ev.gradient[1, 2] == pytest.approx(ev.gradient[2, 2], abs=1e-6)

    # Atoms that come close to a wall get a new box, with a warning.
    stretched = WATER.copy()
    stretched[1, 1] += 2.5 * ANG
    with pytest.warns(UserWarning, match="new box"):
        engine.atoms(water(stretched))
    assert not np.array_equal(engine._box[1], cell)

    # The multiplicity makes the calculation spin-polarized with that many unpaired
    # electrons.
    oh = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]) * ANG
    radical = engine_atoms(calc, Structure.build(["O", "H"], oh, multiplicity=2))
    radical.get_potential_energy()
    assert radical.calc.get_number_of_spins() == 2
    assert radical.calc.get_magnetic_moment() == pytest.approx(1.0, abs=0.01)
    oxygen = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]]) * ANG
    triplet = engine_atoms(calc, Structure.build(["O", "O"], oxygen, multiplicity=3))
    triplet.get_potential_energy()
    assert triplet.calc.get_magnetic_moment() == pytest.approx(2.0, abs=0.01)


@needs_gpaw
def test_gpaw_optimization_from_the_command_line(tmp_path: Path) -> None:
    (tmp_path / "h2.xyz").write_text("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.90\n")
    arguments = ["h2.xyz", "-os", "gpaw", "-func", "PBE", "-opt", "rsirfo_bfgs"]
    settings = ["-calc_opt", "h=0.2", "basis=dzp", "vacuum=4.0"]
    code = _core.run_cli("optmain", [*arguments, *settings])
    assert code == 0
    xyz, log = optimized(tmp_path, "h2")
    assert "GPAW" in log
    # PBE with a double-zeta basis: near 0.76 Angstrom.
    assert 0.72 < float(np.linalg.norm(xyz[1] - xyz[0])) < 0.80


@needs_gpaw
def test_gpaw_crystal_with_plane_waves() -> None:
    # Silicon, two atoms in the primitive cell, one of them off its site: plane waves and
    # k-points through the settings of the command line (mode.name=pw mode.ecut=...).
    a = 5.43 * ANG
    cell = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    x = np.array([[0.0, 0.0, 0.0], [0.27 * a, 0.25 * a, 0.25 * a]])

    def silicon(p: np.ndarray) -> Structure:
        return Structure.build(["Si", "Si"], p, cell_bohr=cell, pbc=(True, True, True))

    options = CalculatorOptions(
        method="PBE",
        extra={"mode.name": "pw", "mode.ecut": 250.0, "kpts": (2, 2, 2)},
    )
    calc = create("gpaw", options)
    ev = calc.evaluate_structure(silicon(x))
    assert np.isfinite(ev.energy)
    # The displaced atom is pushed back along x; the two forces cancel.
    assert ev.gradient[1, 0] > 1e-3
    assert np.abs(ev.gradient.sum(axis=0)).max() < 1e-4
    assert ev.gradient[1, 0] == pytest.approx(slope(calc, silicon, x, 3, 0.02), abs=3e-4)

    # The stress, when it is asked for: against the energies of the cell scaled up and
    # down. With 250 eV the plane waves change with the cell, which the stress of GPAW
    # does not follow: 2 % here (0.7 % with 400 eV).
    assert ev.stress is None
    stressed = calc.evaluate_structure(silicon(x), need_stress=True)
    assert stressed.stress is not None
    assert np.array_equal(stressed.stress, stressed.stress.T)

    def scaled(s: float) -> Structure:
        return Structure.build(["Si", "Si"], x * s, cell_bohr=cell * s, pbc=(True, True, True))

    step = 0.004
    high = calc.evaluate_structure(scaled(1.0 + step)).energy
    low = calc.evaluate_structure(scaled(1.0 - step)).energy
    volume = abs(float(np.linalg.det(cell)))
    assert np.trace(stressed.stress) * volume == pytest.approx(
        (high - low) / (2.0 * step), rel=0.05
    )


@needs_gpaw
def test_gpaw_variable_cell_optimization_from_the_command_line(tmp_path: Path) -> None:
    # Silicon in a cell that is too large: the cell and the atoms are relaxed with the
    # stress of GPAW.
    a = 5.75
    rows = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    cell_text = " ".join(f"{v:.8f}" for v in rows.ravel())
    quarter = 0.25 * a
    (tmp_path / "si.xyz").write_text(
        f'2\nLattice="{cell_text}" pbc="T T T"\n'
        f"Si 0.0 0.0 0.0\nSi {quarter:.8f} {quarter:.8f} {quarter:.8f}\n"
    )
    arguments = ["si.xyz", "-os", "gpaw", "-func", "PBE", "-vc", "-opt", "rsirfo_bfgs"]
    settings = ["-calc_opt", "mode.name=pw", "mode.ecut=300", "kpts=(2,2,2)"]
    code = _core.run_cli("optmain", [*arguments, *settings])
    assert code == 0
    newest = max(tmp_path.rglob("si_optimized.xyz"), key=lambda p: p.stat().st_mtime)
    values = newest.read_text().split('Lattice="')[1].split('"')[0].split()
    lattice = np.array([float(v) for v in values]).reshape(3, 3)
    volume = abs(float(np.linalg.det(lattice)))
    # Smaller than the start (47.5 Angstrom^3) and of the size of silicon (40.0 in
    # experiment; this coarse k-point mesh is not converged).
    assert 36.0 < volume < 45.0, volume
    log = (newest.parent / "mopt.log").read_text()
    assert "STRESS (unit a.u.; xx yy zz yz xz xy):" in log and "converged" in log


LD1_INPUT = """ &input
   title='H', zed=1.0, rel=0, config='1s1', iswitch=3, dft='PBE'
 /
 &inputp
   pseudotype=1, file_pseudopw='H.pbe-tm.UPF', lloc=0, tm=.true.
 /
1
1S  1  0  1.00  0.00  0.80  0.80  0.0
"""
"""A local norm-conserving pseudopotential of hydrogen (Troullier-Martins, PBE) made by
Quantum ESPRESSO's atomic program, so that the test needs no file from elsewhere."""


@needs_espresso
def test_quantum_espresso_molecule_charge_and_spin(tmp_path: Path) -> None:
    (tmp_path / "h.in").write_text(LD1_INPUT)
    with (tmp_path / "h.in").open() as source, (tmp_path / "h.out").open("w") as sink:
        subprocess.run(["ld1.x"], stdin=source, stdout=sink, stderr=subprocess.STDOUT, check=True)
    assert (tmp_path / "H.pbe-tm.UPF").is_file()

    options = CalculatorOptions(
        extra={
            "pseudopotentials.H": "H.pbe-tm.UPF",
            "pseudo_dir": str(tmp_path),
            "input_data.system.ecutwfc": 30,
            "input_data.electrons.conv_thr": 1e-9,
            "vacuum": 3.0,
        }
    )
    calc = create("espresso", options)
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]]) * ANG

    def h2(p: np.ndarray) -> Structure:
        return Structure.build(["H", "H"], p)

    ev = calc.evaluate_structure(h2(x))
    # PBE: -1.166 Hartree at the minimum with the full potential of the nuclei.
    assert -1.20 < ev.energy < -1.10
    assert ev.gradient[1, 2] > 0.0  # stretched: the atoms are pulled together
    assert ev.gradient[1, 2] == pytest.approx(-ev.gradient[0, 2], abs=1e-6)
    assert ev.gradient[1, 2] == pytest.approx(slope(calc, h2, x, 5, 0.02), abs=2e-4)
    text = (tmp_path / "espresso.pwi").read_text()
    assert "tot_charge" not in text and "nspin" not in text

    # The cation: charge and spin from the state of the structure.
    ion = calc.evaluate_structure(Structure.build(["H", "H"], x, charge=1, multiplicity=2))
    assert ion.energy > ev.energy + 0.3
    assert ion.gradient[1, 2] < 0.0  # one electron holds the nuclei less: they repel here
    text = (tmp_path / "espresso.pwi").read_text()
    for needle in ("tot_charge", "nspin", "tot_magnetization", "starting_magnetization"):
        assert needle in text, needle
    report = (tmp_path / "espresso.pwo").read_text()
    moments = [
        line.split("=")[1].split()[0] for line in report.splitlines() if "total magn" in line
    ]
    assert float(moments[-1]) == pytest.approx(1.0, abs=0.01)


@needs_espresso
def test_quantum_espresso_stress(tmp_path: Path) -> None:
    (tmp_path / "h.in").write_text(LD1_INPUT)
    with (tmp_path / "h.in").open() as source, (tmp_path / "h.out").open("w") as sink:
        subprocess.run(["ld1.x"], stdin=source, stdout=sink, stderr=subprocess.STDOUT, check=True)
    # Hydrogen molecules in a small cell. The stress of pw.x leaves out the change of the
    # plane waves with the cell, which is large for this pseudopotential below 100 Ry
    # (30 Ry: four times the slope of the energy); with 120 Ry the two agree to 0.2 %.
    options = CalculatorOptions(
        extra={
            "pseudopotentials.H": "H.pbe-tm.UPF",
            "pseudo_dir": str(tmp_path),
            "input_data.system.ecutwfc": 120,
            "input_data.electrons.conv_thr": 1e-10,
            "kpts": (2, 2, 2),
        }
    )
    calc = create("espresso", options)
    cell = np.diag([3.0, 3.2, 3.6]) * ANG
    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.78]]) * ANG

    def crystal(s: float) -> Structure:
        return Structure.build(["H", "H"], x * s, cell_bohr=cell * s)

    # The stress is computed in the run that gives the energy and the forces, and only
    # when it is asked for.
    ev = calc.evaluate_structure(crystal(1.0), need_stress=True)
    assert calc.evaluations == 1
    assert "tstress" in (tmp_path / "espresso.pwi").read_text()
    assert ev.stress is not None and np.array_equal(ev.stress, ev.stress.T)
    step = 0.004
    high = calc.evaluate_structure(crystal(1.0 + step))
    assert high.stress is None
    assert "tstress" not in (tmp_path / "espresso.pwi").read_text()
    low = calc.evaluate_structure(crystal(1.0 - step))
    volume = abs(float(np.linalg.det(cell)))
    assert np.trace(ev.stress) * volume == pytest.approx(
        (high.energy - low.energy) / (2.0 * step), rel=0.03
    )
