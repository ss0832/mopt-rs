"""Tests of SCF solutions for instabilities (``-stability``): stretched H2 with PySCF, whose
symmetric solution is unstable within UHF and, with RHF, towards UHF.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from mopt_rs import _core
from mopt_rs.calculators import CalculatorOptions, create
from mopt_rs.calculators.base import EngineCalculator, Structure
from mopt_rs.calculators.stability import StabilityCheck

ANG = _core.ANGSTROM_TO_BOHR
needs_pyscf = pytest.mark.skipif(
    importlib.util.find_spec("pyscf") is None, reason="PySCF not installed"
)


def h2(distance: float = 2.5) -> Structure:
    return Structure.build(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, distance * ANG]])


def hf(**options: Any) -> EngineCalculator:
    extra = {"verbose": 0, **{k: v for k, v in options.items() if k == "reference"}}
    kept = {k: v for k, v in options.items() if k != "reference"}
    return create(
        "pyscf",
        CalculatorOptions(method="hf", basis="sto-3g", extra=extra, **kept),
    )


def test_every_nth_evaluation_and_the_settings() -> None:
    check = StabilityCheck(3)
    assert [check.due() for _ in range(7)] == [False, False, True, False, False, True, False]
    assert check.tests == 2
    assert not any(StabilityCheck(0).due() for _ in range(5))
    with pytest.raises(ValueError):
        StabilityCheck(1, "sometimes")
    with pytest.raises(ValueError):
        StabilityCheck(-1)


def test_an_engine_without_a_test_says_so_once() -> None:
    lj = create("lj", CalculatorOptions(stability=1))
    argon = Structure.build(["Ar", "Ar"], [[0.0, 0.0, 0.0], [0.0, 0.0, 7.2]])
    first = lj.evaluate_structure(argon)
    assert any("has no test of its SCF solution" in n for n in first.notes)
    moved = Structure.build(["Ar", "Ar"], [[0.0, 0.0, 0.0], [0.0, 0.0, 7.3]])
    assert not lj.evaluate_structure(moved).notes


@needs_pyscf
def test_an_unstable_unrestricted_solution_is_left_for_a_stabler_one() -> None:
    plain = hf(reference="u")
    tested = hf(reference="u", stability=1)
    symmetric = plain.evaluate_structure(h2()).energy
    ev = tested.evaluate_structure(h2())
    assert ev.energy < symmetric - 1e-3
    assert any("unstable within its kind" in n for n in ev.notes)
    # A broken-symmetry singlet: <S^2> near 1.
    assert ev.spin_square is not None and ev.spin_square > 0.5
    # The next structure starts from the stable solution: tested again, nothing to say.
    near = tested.evaluate_structure(h2(2.52))
    assert near.energy < plain.evaluate_structure(h2(2.52)).energy - 1e-3
    assert not near.notes


@needs_pyscf
def test_a_restricted_solution_is_reported_or_switched_to_unrestricted() -> None:
    restricted = hf().evaluate_structure(h2()).energy
    reported = hf(stability=1).evaluate_structure(h2())
    assert reported.energy == pytest.approx(restricted, abs=1e-8)
    assert any(
        "towards an unrestricted wavefunction; the kind is kept" in n for n in reported.notes
    )
    within = hf(stability=1, stability_external="off").evaluate_structure(h2())
    assert not within.notes
    switching = hf(stability=1, stability_external="switch")
    ev = switching.evaluate_structure(h2())
    assert ev.energy < restricted - 1e-3
    assert any("the run goes on with UHF/UKS" in n for n in ev.notes)
    assert "switched to UHF/UKS" in switching.describe()


@needs_pyscf
def test_the_commands_take_the_option(tmp_path: Path) -> None:
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nstretched\nH 0.0 0.0 0.0\nH 0.0 0.0 2.5\n")
    args = [
        str(xyz),
        "-os",
        "pyscf",
        "-func",
        "hf",
        "-bs",
        "sto-3g",
        "-stability",
        "1",
        "-stability_external",
        "switch",
        "-ns",
        "3",
    ]
    _core.run_cli("optmain", args)
    logs = list(tmp_path.rglob("mopt.log"))
    assert logs
    text = logs[0].read_text()
    assert "SCF stability" in text and "the run goes on with UHF/UKS" in text
