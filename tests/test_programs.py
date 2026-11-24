"""Programs run through input and output files (Gaussian, ORCA, GAMESS-US, NWChem, MOPAC):
readers of their Hessians and the input the engines write. No program is run.

MultiOptPy: `Calculator/ase_tools/gaussian.py`, `orca.py`, `gamess.py`, `nwchem.py`,
`mopac.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mopt_rs.calculators import CalculatorOptions, available, create_engine
from mopt_rs.calculators.base import ElectronicState
from mopt_rs.calculators.programs import (
    PROGRAMS,
    FileProgramEngine,
    gamess_basis,
    gaussian_basis,
    nwchem_memory,
    orca_basis_block,
    program_engine,
    read_gaussian_hessian,
    read_orca_hessian,
)

HAS_ASE = importlib.util.find_spec("ase") is not None
needs_ase = pytest.mark.skipif(not HAS_ASE, reason="ASE not installed")


def symmetric(n: int, seed: int = 7) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, n))
    return 0.5 * (a + a.T)


def gaussian_section(h: np.ndarray) -> str:
    """The force constants as a Gaussian output prints them: the lower triangle in blocks
    of five columns, numbers with a D exponent."""
    n = h.shape[0]
    lines = [" Force constants in Cartesian coordinates: "]
    for first in range(0, n, 5):
        columns = range(first, min(first + 5, n))
        lines.append("        " + "".join(f"{c + 1:14d}" for c in columns))
        for row in range(first, n):
            values = "".join(f"{h[row, c]:14.6E}".replace("E", "D") for c in columns if c <= row)
            lines.append(f"{row + 1:7d}{values}")
    return "\n".join(lines)


def orca_block(h: np.ndarray) -> str:
    """The ``$hessian`` block of an ORCA ``.hess`` file: full rows, five columns a block."""
    n = h.shape[0]
    lines = ["$hessian", str(n)]
    for first in range(0, n, 5):
        columns = range(first, min(first + 5, n))
        lines.append("          " + "".join(f"{c:19d}" for c in columns))
        for row in range(n):
            lines.append(f"{row:5d}   " + "".join(f"{h[row, c]:19.10E}" for c in columns))
    return "\n".join(lines)


def test_gaussian_force_constants_are_read() -> None:
    h = symmetric(12)
    text = "\n".join(
        [
            " NAtoms=    4 NActive=    4",
            # An earlier section (of another step) must not be the one that is read.
            gaussian_section(symmetric(12, seed=1)),
            " Leave Link  716",
            gaussian_section(h),
            " Force constants in internal coordinates: ",
            "                1             2",
            "      1  0.100000D+01",
        ]
    )
    read = read_gaussian_hessian(text)
    assert read.shape == (12, 12)
    assert np.allclose(read, h, atol=5e-7 * np.abs(h).max())
    assert np.array_equal(read, read.T)
    with pytest.raises(ValueError, match="Force constants"):
        read_gaussian_hessian("SCF Done")
    # A section cut short.
    cut = "\n".join(gaussian_section(h).splitlines()[:9])
    with pytest.raises(ValueError, match="incomplete"):
        read_gaussian_hessian(cut)


def test_orca_hessian_file_is_read() -> None:
    h = symmetric(9)
    text = "\n".join(
        [
            "$orca_hessian_file",
            "",
            "$act_atom",
            "  0",
            "",
            orca_block(h),
            "",
            "$vibrational_frequencies",
            "9",
            "    0        0.000000",
        ]
    )
    read = read_orca_hessian(text)
    assert read.shape == (9, 9)
    assert np.allclose(read, h, atol=1e-9)
    with pytest.raises(ValueError, match="hessian"):
        read_orca_hessian("$vibrational_frequencies\n3\n")
    with pytest.raises(ValueError, match="announced|incomplete"):
        read_orca_hessian(orca_block(h).replace("$hessian\n9", "$hessian\n12"))


def test_names_of_basis_sets_and_memory() -> None:
    assert gaussian_basis("def2-SVP") == "def2SVP"
    assert gaussian_basis("Def2-TZVPP") == "def2TZVPP"
    assert gaussian_basis("6-31G(d)") == "6-31G(d)"
    assert gamess_basis("STO-3G") == {"gbasis": "STO", "ngauss": 3}
    assert gamess_basis("3-21G") == {"gbasis": "N21", "ngauss": 3}
    assert gamess_basis("6-31G(d)") == {"gbasis": "N31", "ngauss": 6, "ndfunc": 1}
    assert gamess_basis("6-31G**") == {"gbasis": "N31", "ngauss": 6, "ndfunc": 1, "npfunc": 1}
    assert gamess_basis("6-311++G(2d,p)") == {
        "gbasis": "N311",
        "ngauss": 6,
        "ndfunc": 2,
        "npfunc": 1,
        "diffsp": True,
        "diffs": True,
    }
    assert gamess_basis("cc-pVTZ") == {"gbasis": "CCT"}
    assert gamess_basis("aug-cc-pVDZ") == {"gbasis": "ACCD"}
    with pytest.raises(ValueError, match="gbasis"):
        gamess_basis("def2-SVP")
    assert nwchem_memory("2GB") == "2 gb"
    assert nwchem_memory(" 500 MB ") == "500 mb"
    with pytest.raises(ValueError, match="memory"):
        nwchem_memory("plenty")


def test_the_registry_knows_the_programs() -> None:
    names = set(available())
    for name in PROGRAMS:
        assert name in names
    options = CalculatorOptions(method="PBE0", basis="6-31G(d)", memory="4GB", threads=8)
    engine = create_engine("gaussian", options)
    assert isinstance(engine, FileProgramEngine)
    assert engine.name == "gaussian"
    assert engine.capabilities.analytic_hessian
    assert engine.kwargs["xc"] == "PBE0" and engine.kwargs["mem"] == "4GB"
    assert engine.kwargs["nprocshared"] == 8
    assert any("Gaussian 16" in c for c in engine.citations)
    assert any("ASE" in c for c in engine.citations)
    # Other names of the same programs, and options that address input groups.
    assert create_engine("pwscf").name == "espresso"
    gamess = program_engine(
        "gamess", CalculatorOptions(extra={"contrl.scftyp": "rohf", "system.mwords": 200})
    )
    assert gamess.kwargs["contrl"] == {"scftyp": "rohf"}
    assert gamess.kwargs["system"] == {"mwords": 200}
    assert not gamess.capabilities.analytic_hessian
    with pytest.raises(KeyError, match="no program"):
        program_engine("turbomole", CalculatorOptions())


@needs_ase
def test_orca_basis_for_heavy_elements() -> None:
    assert orca_basis_block(["C", "H", "Zn"], "6-31G(d)", "def2-SVP") == ""
    block = orca_basis_block(["C", "I", "H", "Br", "I"], "6-31G(d)", "def2-TZVP")
    assert block.splitlines() == [
        "%basis",
        '  NewGTO Br "def2-TZVP"',
        '  NewGTO I "def2-TZVP"',
        "end",
    ]
    # Other basis sets have these elements.
    assert orca_basis_block(["I"], "def2-SVP", "def2-SVP") == ""


def water() -> Any:
    from ase import Atoms

    return Atoms("OH2", positions=[(0.0, 0.0, 0.119), (0.0, 0.763, -0.477), (0.0, -0.763, -0.477)])


def calculator(
    name: str, directory: Path, state: ElectronicState, method: str = "B3LYP", **extra: Any
) -> Any:
    options = CalculatorOptions(
        method=method,
        basis="6-31G(d)",
        memory="2GB",
        extra={"directory": str(directory), **extra},
    )
    return program_engine(name, options).calculator(state)


@needs_ase
def test_gaussian_input_and_hessian(tmp_path: Path) -> None:
    calc = calculator("gaussian", tmp_path, ElectronicState(1, 2), empiricaldispersion="gd3bj")
    atoms = water()
    calc.write_input(atoms, properties=["energy", "forces"])
    text = (tmp_path / "Gaussian.com").read_text().lower()
    assert "%mem=2gb" in text
    assert "b3lyp/6-31g(d)" in text
    assert "scf(xqc,maxcon=128,maxcyc=32,conver=8)" in text
    assert "nosymm" in text and "force" in text
    assert "empiricaldispersion(gd3bj)" in text
    assert "\n1 2\n" in text

    # The Hessian: a frequency job is run and its force constants are read (the run itself
    # is replaced by writing the output).
    h = symmetric(9)
    seen: dict[str, Any] = {}

    def fake(atoms: Any = None, properties: Any = None, system_changes: Any = None) -> None:
        seen["extra"] = calc.parameters.get("extra")
        seen["properties"] = properties
        (tmp_path / "Gaussian.log").write_text(gaussian_section(h) + "\n Leave Link 716\n")

    calc.calculate = fake
    from ase import units

    got = calc.get_hessian(atoms)
    assert seen["extra"] == "freq=noraman" and seen["properties"] == ["energy"]
    assert np.allclose(got, h * units.Hartree / units.Bohr**2, rtol=1e-5, atol=1e-5)
    # The settings of the gradient runs are back.
    assert "extra" not in calc.parameters and calc.results == {}


@needs_ase
def test_orca_input_and_hessian(tmp_path: Path) -> None:
    from ase import Atoms

    program = tmp_path / "bin" / "orca"
    program.parent.mkdir()
    program.write_text("")
    calc = calculator(
        "orca",
        tmp_path,
        ElectronicState(-1, 1),
        path=str(program.parent),
        simple="D3BJ",
        blocks="%scf maxiter 300 end",
        nprocs=4,
        maxcore=1500,
    )
    atoms = Atoms(
        "CH3I",
        positions=[
            (0, 0, 0),
            (1.03, 0, -0.36),
            (-0.51, 0.89, -0.36),
            (-0.51, -0.89, -0.36),
            (0, 0, 2.14),
        ],
    )
    calc.write_inputfiles(atoms, ["energy", "forces"])
    text = (tmp_path / "orca.inp").read_text()
    assert "! B3LYP 6-31G(d) D3BJ EnGrad" in text
    assert "%pal nprocs 4 end" in text and "%maxcore 1500" in text
    assert "%scf maxiter 300 end" in text
    assert 'NewGTO I "def2-SVP"' in text
    assert "*xyz -1 1" in text

    h = symmetric(15)
    seen: dict[str, Any] = {}

    def fake(atoms: Any, properties: Any, system_changes: Any) -> None:
        seen["task"] = calc.task
        (tmp_path / "orca.hess").write_text("$orca_hessian_file\n\n" + orca_block(h) + "\n\n")

    calc.calculate = fake
    from ase import units

    got = calc.get_hessian(atoms)
    assert seen["task"] == "Freq" and calc.task == "EnGrad"
    assert np.allclose(got, h * units.Hartree / units.Bohr**2, rtol=1e-9, atol=1e-9)

    strict = calculator(
        "orca", tmp_path, ElectronicState(0, 1), path=str(program), auto_fix_basis=False
    )
    with pytest.raises(ValueError, match="zinc"):
        strict.write_inputfiles(atoms, ["energy"])


@needs_ase
def test_charge_and_multiplicity_reach_every_program(tmp_path: Path) -> None:
    atoms = water()
    state = ElectronicState(1, 2)

    mopac = calculator("mopac", tmp_path, state, method="PM6-D3H4")
    mopac.write_input(atoms)
    text = (tmp_path / "mopac.mop").read_text()
    assert "CHARGE=1" in text and "DOUBLET UHF" in text
    assert text.startswith("PM6-D3H4 1SCF GRADIENTS DISP")
    # The structure that was passed keeps its own magnetic moments.
    assert not atoms.get_initial_magnetic_moments().any()

    gamess = calculator("gamessus", tmp_path, state)
    gamess.write_input(atoms, properties=["energy", "forces"])
    text = (tmp_path / "gamess_us.inp").read_text().upper()
    for needle in (
        "ICHARG=1",
        "MULT=2",
        "SCFTYP=UHF",
        "DFTTYP=B3LYP",
        "RUNTYP=GRADIENT",
        "GBASIS=N31",
        "NGAUSS=6",
        "NDFUNC=1",
    ):
        assert needle in text, needle

    nwchem = calculator("nwchem", tmp_path, state)
    nwchem.write_input(atoms, properties=["energy", "forces"])
    text = (tmp_path / "nwchem.nwi").read_text().lower()
    for needle in ("charge 1", "mult 2", "odft", "memory 2 gb", "xc b3lyp", "6-31g(d)"):
        assert needle in text, needle


@needs_ase
def test_a_structure_without_a_cell_is_put_in_a_box(monkeypatch: pytest.MonkeyPatch) -> None:
    from ase.calculators.lj import LennardJones

    from mopt_rs.calculators.base import EngineCalculator, Structure

    seen: dict[str, Any] = {}

    class Probe(LennardJones):  # type: ignore[misc]
        def calculate(
            self, atoms: Any = None, properties: Any = None, system_changes: Any = None
        ) -> None:
            seen["lengths"] = atoms.cell.lengths().tolist()
            seen["charge"] = self.parameters.get("charge")
            seen["moments"] = atoms.get_initial_magnetic_moments().tolist()
            super().calculate(atoms, properties, system_changes)

    def build(*, charge: int = 0, vacuum: float = 6.0, **keywords: Any) -> Any:
        del vacuum
        return Probe(charge=charge, **keywords)

    # A program that needs a cell, stood in for by a pair potential.
    monkeypatch.setitem(PROGRAMS, "gpaw", PROGRAMS["gpaw"]._replace(build=build))
    engine = program_engine("gpaw", CalculatorOptions(extra={"vacuum": 4.0, "sigma": 3.4}))
    structure = Structure.build(
        ["Ar", "Ar"], [[0.0, 0.0, 0.0], [7.5, 0.0, 0.0]], charge=1, multiplicity=2
    )
    result = EngineCalculator(engine).evaluate_structure(structure)
    assert np.isfinite(result.energy)
    # 4 Angstrom on every side; the pair lies along x.
    separation = 7.5 * 0.529177210903
    assert np.allclose(seen["lengths"], [8.0 + separation, 8.0, 8.0], atol=1e-6)
    assert seen["charge"] == 1
    # The doublet: magnetic moments that add up to one unpaired electron.
    assert seen["moments"] == [0.5, 0.5]
    # The gradient is that of the pair, whatever the box.
    assert abs(result.gradient[0, 0] + result.gradient[1, 0]) < 1e-12

    # The box of the first structure is kept for the structures that follow (the grid of a
    # program stays the same), until atoms come closer to a wall than half the vacuum.
    first = list(seen["lengths"])
    longer = Structure.build(
        ["Ar", "Ar"], [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], charge=1, multiplicity=2
    )
    EngineCalculator(engine).evaluate_structure(longer)
    assert seen["lengths"] == first
    far = Structure.build(
        ["Ar", "Ar"], [[0.0, 0.0, 0.0], [12.5, 0.0, 0.0]], charge=1, multiplicity=2
    )
    with pytest.warns(UserWarning, match="new box"):
        EngineCalculator(engine).evaluate_structure(far)
    assert seen["lengths"][0] == pytest.approx(8.0 + 12.5 * 0.529177210903, abs=1e-6)


@needs_ase
def test_orca_tests_its_solution_when_asked(tmp_path: Path) -> None:
    from ase import Atoms

    from mopt_rs.calculators.base import Structure

    program = tmp_path / "bin" / "orca"
    program.parent.mkdir()
    program.write_text("")
    options = CalculatorOptions(
        method="B3LYP",
        basis="def2-SVP",
        extra={"directory": str(tmp_path), "path": str(program.parent)},
    )
    engine = program_engine("orca", options)
    structure = Structure.build(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 4.7]])
    calc = engine.calculator(structure.state)
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 2.5)])
    for mode, restart in (("switch", True), ("report", False)):
        report = engine.check_stability(structure, mode)
        assert report is not None and report.changed
        calc.write_inputfiles(atoms, ["energy", "forces"])
        text = (tmp_path / "orca.inp").read_text()
        assert "%scf STABPerform true" in text
        assert ("STABRestartUHFifUnstable true" in text) == restart
        # The next input is a plain one again.
        calc.write_inputfiles(atoms, ["energy", "forces"])
        assert "STABPerform" not in (tmp_path / "orca.inp").read_text()
    # Gaussian has no such test in one job: it says so (None).
    gaussian = program_engine("gaussian", CalculatorOptions(extra={"directory": str(tmp_path)}))
    assert gaussian.check_stability(structure, "report") is None
