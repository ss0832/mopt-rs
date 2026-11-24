"""Settings of the programs run through files: every documented option reaches the input
that is written, defaults kept in files, values read from files, and the listing of what a
program takes. No program is run.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from mopt_rs.calculators import (
    AseEngine,
    CalculatorOptions,
    FileProgramEngine,
    create,
    create_engine,
)
from mopt_rs.calculators.base import ElectronicState, Structure
from mopt_rs.calculators.option_files import (
    default_options,
    file_value,
    option_file_paths,
    read_option_file,
)
from mopt_rs.calculators.programs import (
    OTHER,
    PROGRAMS,
    describe_program,
    main,
    program_engine,
)
from mopt_rs.calculators.registry import parse_option_value

HAS_ASE = importlib.util.find_spec("ase") is not None
needs_ase = pytest.mark.skipif(not HAS_ASE, reason="ASE not installed")

ANGSTROM = 1.0 / 0.529177210903


@pytest.fixture(autouse=True)
def no_files_of_the_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts in an empty directory, without settings files of the user."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("MOPT_CALCULATOR_OPTIONS", raising=False)
    monkeypatch.chdir(tmp_path)


def water() -> Any:
    from ase import Atoms

    return Atoms("OH2", positions=[(0.0, 0.0, 0.119), (0.0, 0.763, -0.477), (0.0, -0.763, -0.477)])


def calculator(name: str, state: ElectronicState, /, **extra: Any) -> Any:
    options = CalculatorOptions(method=None, basis="6-31G(d)", threads=4, extra=extra)
    return program_engine(name, options).calculator(state)


def test_values_read_from_files(tmp_path: Path) -> None:
    (tmp_path / "blocks.inp").write_text("%scf\n  maxiter 300\nend\n\n")
    assert file_value("@blocks.inp") == "%scf\n  maxiter 300\nend"
    assert file_value("@@not_a_file") == "@not_a_file"
    assert file_value("plain") == "plain" and file_value(3) == 3 and file_value("@") == "@"
    # Inside mappings and lists, and relative to a directory that is given.
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.txt").write_text("text\n")
    assert file_value({"a": ["@x.txt", 1]}, other) == {"a": ["text", 1]}
    with pytest.raises(ValueError, match="missing.inp"):
        file_value("@missing.inp")
    # The command line: the text is not read as a Python literal.
    (tmp_path / "number.txt").write_text("12\n")
    assert parse_option_value("@number.txt") == "12"
    assert parse_option_value("12") == 12
    assert parse_option_value("@@x") == "@x"


def test_defaults_kept_in_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert option_file_paths() == []
    assert default_options(["orca"]) == ({}, ())
    (tmp_path / "blocks.inp").write_text("%scf maxiter 300 end\n")
    local = tmp_path / "mopt_calculator_options.json"
    local.write_text(
        json.dumps(
            {
                "calculators": {
                    "ORCA": {"simple": "RIJCOSX D3BJ", "blocks": "@blocks.inp", "maxcore": 1500},
                    "espresso": {"input_data": {"system": {"ecutwfc": 30}}},
                    "ase": {"sigma": 3.0},
                    "ase:ase.calculators.lj:LennardJones": {"epsilon": 0.5},
                }
            }
        )
    )
    named = tmp_path / "elsewhere" / "more.json"
    named.parent.mkdir()
    named.write_text(json.dumps({"calculators": {"orca": {"maxcore": 3000}}}))
    monkeypatch.setenv("MOPT_CALCULATOR_OPTIONS", str(named))
    assert option_file_paths() == [Path("mopt_calculator_options.json"), named]
    assert read_option_file(local)["orca"]["blocks"] == "%scf maxiter 300 end"

    # The file named in the environment ranks above the one of the directory; what is
    # given on the command line ranks above both and is not listed as coming from a file.
    options, notes = default_options(["orca"], given={"simple"})
    assert options == {"blocks": "%scf maxiter 300 end", "maxcore": 3000}
    assert len(notes) == 2 and "blocks, maxcore" in notes[0] and "more.json: maxcore" in notes[1]

    engine = create_engine("orca", CalculatorOptions(extra={"simple": "TightSCF"}))
    assert isinstance(engine, FileProgramEngine)
    assert engine.kwargs["simple"] == "TightSCF"
    assert engine.kwargs["blocks"] == "%scf maxiter 300 end"
    assert engine.kwargs["maxcore"] == 3000
    text = engine.describe()
    assert "mopt_calculator_options.json: blocks, maxcore" in text and "more.json" in text
    assert "simple" not in text.split("options from")[1]

    # Another name of the program finds the settings of the program; a key of the command
    # line refines a group of the file.
    qe = create_engine(
        "qe",
        CalculatorOptions(
            extra={"input_data.system.ecutrho": 240, "pseudopotentials.Si": "Si.UPF"}
        ),
    )
    assert isinstance(qe, FileProgramEngine)
    assert qe.kwargs["input_data"] == {"system": {"ecutwfc": 30}, "system.ecutrho": 240}
    assert qe.kwargs["pseudopotentials"] == {"Si": "Si.UPF"}

    # Any calculator: the part of the name before the colon, then the whole name.
    if HAS_ASE:
        lj = create_engine("ase:ase.calculators.lj:LennardJones")
        assert isinstance(lj, AseEngine)
        assert lj.kwargs == {"sigma": 3.0, "epsilon": 0.5}
        assert "options from" in create("ase:ase.calculators.lj:LennardJones").describe()

    local.write_text("{}")
    with pytest.raises(ValueError, match="calculators"):
        create_engine("orca")
    local.write_text('{"calculators": {"orca": 3}}')
    with pytest.raises(ValueError, match="orca"):
        create_engine("orca")
    local.write_text("{")
    with pytest.raises(ValueError, match="mopt_calculator_options.json"):
        create_engine("orca")


def test_the_settings_of_a_program_are_listed(capsys: pytest.CaptureFixture[str]) -> None:
    for name, program in PROGRAMS.items():
        text = describe_program(name)
        assert text.startswith(f"-os {name}\n")
        for key in program.options:
            assert key in text
        assert "KEY=@FILE" in text and "mopt_calculator_options.json" in text
    assert describe_program("QE") == describe_program("espresso")
    assert OTHER in PROGRAMS["gaussian"].options and OTHER not in PROGRAMS["orca"].options
    assert main(["mopac"]) == 0
    assert "THREADS=N" in capsys.readouterr().out
    assert main([]) == 0
    assert capsys.readouterr().out.count("-os ") == len(PROGRAMS)
    assert main(["turbomole"]) == 2
    assert "no program" in capsys.readouterr().err


def test_a_setting_a_program_does_not_have_is_an_error() -> None:
    # ORCA and MOPAC write only what their functions take: a misspelt key cannot vanish.
    with pytest.raises(ValueError, match=r"orca: unknown option simpel\n-os orca"):
        program_engine("orca", CalculatorOptions(extra={"simpel": "D3BJ"}))
    with pytest.raises(ValueError, match="unknown options eps, precise"):
        program_engine("mopac", CalculatorOptions(extra={"eps": 78.4, "precise": True}))
    with pytest.raises(ValueError, match="GROUP.KEY"):
        program_engine("gamessus", CalculatorOptions(extra={"contrl.": 1}))
    with pytest.raises(ValueError, match="not to a group"):
        program_engine("nwchem", CalculatorOptions(extra={"theory": "mp2", "theory.x": 1}))


@needs_ase
def test_settings_reach_the_input_files(tmp_path: Path) -> None:
    atoms = water()

    # MOPAC: keywords that ASE has no parameter for, and the threads of -core.
    mopac = calculator("mopac", ElectronicState(-1, 1), keywords="EPS=78.4 PRECISE", relscf=0.01)
    mopac.write_input(atoms)
    first = (tmp_path / "mopac.mop").read_text().splitlines()[0]
    assert first.startswith("PM7 1SCF GRADIENTS DISP EPS=78.4 PRECISE THREADS=4 RELSCF=0.01")
    assert "CHARGE=-1" in first

    # GAMESS-US: any input group.
    neutral = ElectronicState(0, 1)
    gamess = calculator(
        "gamessus",
        neutral,
        **{"pcm.solvnt": "WATER", "contrl.maxit": 150, "scf": {"dirscf": True}},
    )
    gamess.write_input(atoms, properties=["energy", "forces"])
    text = (tmp_path / "gamess_us.inp").read_text().upper()
    for needle in ("$PCM", "SOLVNT=WATER", "MAXIT=150", "$SCF", "DIRSCF=.T.", "ICHARG=0"):
        assert needle in text, needle

    # NWChem: any block, and a theory other than DFT.
    nwchem = calculator(
        "nwchem", neutral, **{"scf.maxiter": 200, "theory": "mp2", "mp2.freeze": "atomic"}
    )
    nwchem.write_input(atoms, properties=["energy", "forces"])
    text = (tmp_path / "nwchem.nwi").read_text().lower()
    for needle in ("maxiter 200", "freeze atomic", "task mp2 gradient"):
        assert needle in text, needle
    assert "xc " not in text

    # Gaussian: a section after the structure read from a file, and the Hessian job.
    (tmp_path / "gen.txt").write_text("O H 0\n6-31G(d)\n****\n")
    gaussian = calculator(
        "gaussian",
        neutral,
        addsec=parse_option_value("@gen.txt"),
        scrf="pcm,solvent=water",
        hessian_route="freq=(noraman,cubic)",
    )
    gaussian.write_input(atoms, properties=["energy", "forces"])
    text = (tmp_path / "Gaussian.com").read_text()
    assert "scrf(pcm,solvent=water)" in text.lower()
    assert "O H 0\n6-31G(d)\n****" in text
    seen: dict[str, Any] = {}

    def fake(atoms: Any = None, properties: Any = None, system_changes: Any = None) -> None:
        seen["extra"] = gaussian.parameters.get("extra")
        raise RuntimeError("no program")

    gaussian.calculate = fake
    with pytest.raises(RuntimeError):
        gaussian.get_hessian(atoms)
    assert seen["extra"] == "freq=(noraman,cubic)"
    with pytest.raises(ValueError, match="no groups"):
        calculator("gaussian", neutral, **{"scrf.solvent": "water"})

    # ORCA: the task of the Hessian job.
    program = tmp_path / "bin" / "orca"
    program.parent.mkdir()
    program.write_text("")
    orca = calculator(
        "orca", neutral, path=str(program), hessian_task="NumFreq", blocks="%cpcm\nend"
    )

    def numfreq(atoms: Any, properties: Any, system_changes: Any) -> None:
        orca.write_inputfiles(atoms, properties)
        raise RuntimeError("no program")

    orca.calculate = numfreq
    with pytest.raises(RuntimeError):
        orca.get_hessian(atoms)
    text = (tmp_path / "orca.inp").read_text()
    assert "! B3LYP 6-31G(d) NumFreq" in text and "%cpcm\nend" in text
    assert "%pal nprocs 4 end" in text
    assert orca.task == "EnGrad"


@needs_ase
def test_quantum_espresso_input(tmp_path: Path) -> None:
    engine = program_engine(
        "espresso",
        CalculatorOptions(
            method="pbe",
            extra={
                "pseudopotentials.O": "O.UPF",
                "pseudopotentials.H": "H.UPF",
                "pseudo_dir": str(tmp_path),
                "input_data.system.ecutwfc": 40,
                "input_data.electrons.conv_thr": 1e-8,
                "kpts": (2, 2, 2),
                "vacuum": 5.0,
            },
        ),
    )
    structure = Structure.build(
        ["O", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97 * ANGSTROM]],
        charge=-1,
        multiplicity=3,
    )
    atoms = engine.atoms(structure)
    # A box around the molecule, and moments that add up to the unpaired electrons.
    assert atoms.cell.lengths().min() >= 10.0 - 1e-9
    assert atoms.get_initial_magnetic_moments().sum() == pytest.approx(2.0)
    calc = engine.calculator(structure.state)
    calc.template.write_input(calc.profile, tmp_path, atoms, dict(calc.parameters), ["energy"])
    text = (tmp_path / "espresso.pwi").read_text().lower()
    for needle in (
        "tprnfor",
        "ecutwfc",
        "conv_thr",
        "input_dft",
        "tot_charge",
        "nspin",
        "tot_magnetization",
        "starting_magnetization",
        "o.upf",
        "h.upf",
        "2 2 2",
    ):
        assert needle in text, needle
    values = {
        line.split("=")[0].strip(): line.split("=")[1].strip().rstrip(",")
        for line in text.splitlines()
        if "=" in line
    }
    assert float(values["tot_charge"]) == -1.0
    assert int(values["nspin"]) == 2 and float(values["tot_magnetization"]) == 2.0

    # What the input data say about the charge and the spin wins.
    own = program_engine(
        "espresso",
        CalculatorOptions(
            extra={
                "pseudopotentials": {"O": "O.UPF", "H": "H.UPF"},
                "pseudo_dir": str(tmp_path),
                "input_data.system.tot_magnetization": 0,
                "input_data.tot_charge": -2,
            }
        ),
    ).calculator(structure.state)
    data = own.parameters["input_data"]
    assert data["system"]["tot_magnetization"] == 0 and data["tot_charge"] == -2
    assert "tot_charge" not in data["system"]
    with pytest.raises(ValueError, match="pseudopotentials"):
        program_engine("espresso", CalculatorOptions()).calculator(structure.state)
