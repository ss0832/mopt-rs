"""Excited states of a program outside the package (``-calc_opt excited_setup=``,
``state_reader=``): the frame through which such a program gives ``-es N`` and what
``-es_follow`` compares, with a model program in the place of a real one.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import fake_ase
from mopt_rs import _core
from mopt_rs.calculators import AseEngine, EngineCalculator, Structure, describe_program
from mopt_rs.calculators.base import BOHR_TO_ANGSTROM
from mopt_rs.calculators.program_states import (
    STATES_FILE,
    RawStates,
    file_reader,
    read_states_file,
    state_characters,
    write_states_file,
)


def raw() -> RawStates:
    overlap = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    # Orbitals that are orthonormal in this basis: C^T S C = 1.
    lower = np.linalg.cholesky(overlap)
    coefficients = np.linalg.inv(lower).T
    amplitudes = np.array([[[3.0, 4.0]], [[2.0, -1.5]]])
    return RawStates(coefficients, 1, np.array([0.2, 0.3]), amplitudes, overlap)


def test_what_a_program_writes_becomes_what_is_compared() -> None:
    found = state_characters(raw(), 6)
    orbitals = found["orbitals"]
    assert np.allclose(orbitals.T @ orbitals, np.eye(3), atol=1e-12)
    assert found["occupied"] == 1 and found["energies"].tolist() == [0.2, 0.3]
    assert found["amplitudes"].shape == (2, 2)
    assert np.allclose(found["amplitudes"][:, 0], [0.6, 0.8])
    assert np.allclose(np.linalg.norm(found["amplitudes"], axis=0), 1.0)
    # No more states than were asked for; orthonormal functions need no overlap.
    one = state_characters(RawStates(np.eye(3), 1, raw().energies, raw().amplitudes), 1)
    assert one["amplitudes"].shape == (2, 1) and np.array_equal(one["orbitals"], np.eye(3))

    with pytest.raises(ValueError, match="every orbital"):
        state_characters(RawStates(np.eye(3)[:, :2], 1, raw().energies, raw().amplitudes), 2)
    with pytest.raises(ValueError, match=r"expected \[state, occupied, empty\]"):
        state_characters(RawStates(np.eye(3), 1, raw().energies, np.zeros((2, 2))), 2)
    with pytest.raises(ValueError, match="ascending"):
        state_characters(RawStates(np.eye(3), 1, np.array([0.3, 0.2]), raw().amplitudes), 2)
    with pytest.raises(ValueError, match="without amplitudes"):
        state_characters(RawStates(np.eye(3), 1, raw().energies, np.zeros((2, 1, 2))), 2)


def test_the_file_of_the_states(tmp_path: Path) -> None:
    write_states_file(tmp_path / STATES_FILE, raw())
    back = read_states_file(tmp_path / STATES_FILE)
    assert np.array_equal(back.coefficients, raw().coefficients)
    assert back.overlap is not None and np.array_equal(back.overlap, raw().overlap)  # type: ignore[arg-type]
    assert np.array_equal(back.amplitudes, raw().amplitudes) and back.occupied == 1

    class Program:
        directory = str(tmp_path)

    found = file_reader(Program(), None, 2)
    assert found is not None and np.array_equal(found.energies, [0.2, 0.3])
    Program.directory = str(tmp_path / "nothing_here")
    assert file_reader(Program(), None, 2) is None
    (tmp_path / "broken.json").write_text('{"occupied": 1}')
    with pytest.raises(ValueError, match="no key"):
        read_states_file(tmp_path / "broken.json")


def pair(distance_bohr: float) -> Structure:
    return Structure.build(["H", "H"], [0.0, 0.0, 0.0, 0.0, 0.0, distance_bohr])


def test_an_engine_with_the_two_functions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ase", fake_ase.module())
    engine = AseEngine(
        fake_ase.TwoStateProgram,
        kwargs={"directory": str(tmp_path)},
        electrons=False,
        excited_setup=fake_ase.choose_root,
        state_reader="file",
    )
    assert engine.capabilities.excited_states
    calc = EngineCalculator(engine, check_electrons=False)
    # At 2.6 Bohr the second state by energy is the one with its minimum at 1.6.
    at = pair(2.6)
    second = calc.evaluate_structure(
        Structure.build(["H", "H"], at.positions_bohr.ravel().tolist(), excited_state=2)
    )
    assert abs(second.energy - 0.5) < 1e-7
    assert abs(second.gradient[1][2] - 1.0) < 1e-7
    ground = calc.evaluate_structure(at)
    assert abs(ground.energy + 1.0) < 1e-7
    program = engine.calculator(at.state)
    assert program.runs[0] == (2, 2) and program.runs[-1][0] == 0

    found = calc.state_characters(at, 6)
    assert found is not None
    assert program.runs[-1] == (1, 6)
    assert np.allclose(found["orbitals"], np.eye(3), atol=1e-12)
    # The lower state here is the excitation into the second empty orbital.
    assert np.allclose(found["amplitudes"], [[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(found["energies"], [1.13, 1.5])

    # Without a reader the states are not handed out, and that is said once.
    plain = AseEngine(fake_ase.TwoStateProgram, kwargs={"directory": str(tmp_path)})
    assert not plain.capabilities.excited_states
    with pytest.warns(RuntimeWarning, match="state_reader=module:function"):
        assert plain.state_characters(at, 2) is None


def test_a_command_follows_a_state_of_a_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ase", fake_ase.module())
    monkeypatch.chdir(tmp_path)
    xyz = tmp_path / "pair.xyz"
    xyz.write_text(f"2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 {2.6 * BOHR_TO_ANGSTROM:.10f}\n")
    arguments = [str(xyz), "-os", "ase:fake_ase:TwoStateProgram", "-es", "2", "-es_follow"]
    arguments += ["-ns", "60", "-calc_opt", f"directory={tmp_path}"]
    arguments += ["excited_setup=fake_ase:choose_root", "state_reader=file"]
    assert _core.run_cli("optmain", arguments) == 0
    found = sorted(tmp_path.rglob("*_optimized.xyz"))
    assert len(found) == 1
    rows = [line.split() for line in found[0].read_text().splitlines()[2:4]]
    a, b = (np.array([float(v) for v in row[1:4]]) for row in rows)
    # Through the crossing at 2.336 Bohr down to the minimum of the state at 1.6.
    assert abs(float(np.linalg.norm(a - b)) / BOHR_TO_ANGSTROM - 1.6) < 1e-3
    log = next(tmp_path.rglob("mopt.log")).read_text()
    assert "is state 1 in the order of the energies here" in log


def test_the_programs_name_the_two_settings() -> None:
    text = describe_program("orca")
    assert "excited_setup" in text and "state_reader" in text
