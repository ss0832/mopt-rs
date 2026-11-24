"""Excited states followed from structure to structure (``-es_follow``) with engines of
Python: a model engine whose two states change places (a command follows the state through
the crossing), the overlaps as a function, and what PySCF hands out for its
Tamm-Dancoff states.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mopt_rs import _core
from mopt_rs.calculators import (
    CalculatorOptions,
    Capabilities,
    Engine,
    EngineCalculator,
    EngineResult,
    Structure,
    create,
    register,
)
from mopt_rs.calculators.base import ATOMIC_UNITS, BOHR_TO_ANGSTROM

needs_pyscf = pytest.mark.skipif(
    importlib.util.find_spec("pyscf") is None, reason="PySCF not installed"
)


class TwoStates(Engine):
    """Two atoms at the distance ``d`` (Bohr) with two excited states, ``A = (d - 1.6)^2 /
    2`` (the excitation into the first empty orbital) and ``B = (d - 3)^2 / 2 + 0.05``
    (into the second), which cross at ``d = 2.336``; the ground state is flat."""

    name = "two_states"
    units = ATOMIC_UNITS
    capabilities = Capabilities(excited_states=True, electrons=False)

    @staticmethod
    def _states(d: float) -> list[tuple[float, float, int]]:
        """Energy, slope, and excitation of the states, the lower one first."""
        a = (0.5 * (d - 1.6) ** 2, d - 1.6, 0)
        b = (0.5 * (d - 3.0) ** 2 + 0.05, d - 3.0, 1)
        return sorted([a, b])

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        between = structure.positions_bohr[1] - structure.positions_bohr[0]
        d = float(np.linalg.norm(between))
        if structure.excited_state == 0:
            return EngineResult(-1.0, gradient=np.zeros((2, 3)))
        energy, slope, _ = self._states(d)[structure.excited_state - 1]
        pull = slope * between / d
        return EngineResult(energy, gradient=np.array([-pull, pull]))

    def state_characters(self, structure: Structure, count: int) -> dict[str, Any] | None:
        between = structure.positions_bohr[1] - structure.positions_bohr[0]
        states = self._states(float(np.linalg.norm(between)))[:count]
        amplitudes = np.zeros((2, len(states)))
        for k, (_, _, excitation) in enumerate(states):
            amplitudes[excitation, k] = 1.0
        return {
            "orbitals": np.eye(3),
            "occupied": 1,
            "energies": [energy for energy, _, _ in states],
            "amplitudes": amplitudes,
        }


def _pair(path: Path, distance_bohr: float) -> str:
    f = path / "pair.xyz"
    f.write_text(f"2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 {distance_bohr * BOHR_TO_ANGSTROM:.10f}\n")
    return str(f)


def _distance_bohr(xyz: Path) -> float:
    rows = [line.split() for line in xyz.read_text().splitlines()[2:4]]
    a, b = (np.array([float(v) for v in row[1:4]]) for row in rows)
    return float(np.linalg.norm(a - b)) / BOHR_TO_ANGSTROM


def test_a_command_follows_a_state_of_a_python_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register("two_states_test", lambda arg, o: TwoStates(), replace_existing=True)
    monkeypatch.chdir(tmp_path)
    # At 2.6 Bohr the state with its minimum at 1.6 is the second one by energy. Followed,
    # the minimization of state 2 goes through the crossing down to that minimum.
    xyz = _pair(tmp_path, 2.6)
    arguments = [xyz, "-os", "two_states_test", "-es", "2", "-es_follow", "-ns", "60"]
    assert _core.run_cli("optmain", arguments) == 0
    found = sorted(tmp_path.rglob("*_optimized.xyz"))
    assert len(found) == 1
    assert abs(_distance_bohr(found[0]) - 1.6) < 1e-3
    log = next(tmp_path.rglob("mopt.log")).read_text()
    assert "excited states followed between structures" in log
    assert "electronic state 2" in log
    assert "is state 1 in the order of the energies here" in log


def test_state_following_from_python_tasks() -> None:
    from mopt_rs import FollowedCalculator, Task

    calc = FollowedCalculator(EngineCalculator(TwoStates(), check_electrons=False))
    for d in (2.6, 2.1, 1.6):
        task = Task(
            "pes",
            "two_states",
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, d]]),
            0,
            1,
            False,
            2,
        )
        value = calc.evaluate(task)
        assert value.energy == pytest.approx(0.5 * (d - 1.6) ** 2)


def test_a_calculator_checks_what_the_engine_hands_out() -> None:
    calc = EngineCalculator(TwoStates(), check_electrons=False)
    s = Structure.build(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    found = calc.state_characters(s, 6)
    assert found is not None
    assert found["amplitudes"].shape == (2, 2) and found["occupied"] == 1
    assert calc.state_characters(s, 1)["amplitudes"].shape == (2, 1)  # type: ignore[index]

    class Wrong(TwoStates):
        def state_characters(self, structure: Structure, count: int) -> dict[str, Any] | None:
            found = super().state_characters(structure, count)
            assert found is not None
            return {**found, "amplitudes": np.zeros((3, 2))}

    with pytest.raises(Exception, match="amplitudes of shape"):
        EngineCalculator(Wrong(), check_electrons=False).state_characters(s, 2)
    # An engine without them says so by None.
    assert Engine.state_characters(TwoStates(), s, 2) is None


def test_the_overlaps_of_the_states_of_two_structures() -> None:
    def states(angle: float, order: tuple[int, int]) -> Any:
        c, s = np.cos(angle), np.sin(angle)
        orbitals = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
        amplitudes = np.zeros((2, 2))
        for k, excitation in enumerate(order):
            amplitudes[excitation, k] = 1.0
        return (orbitals, 1, amplitudes.tolist())

    here = states(0.0, (0, 1))
    same, kept = _core.state_overlaps(here, here)
    assert np.allclose(same, np.eye(2)) and abs(kept - 1.0) < 1e-14
    # The empty orbitals turned a little and the states in the other order.
    found, kept = _core.state_overlaps(here, states(0.1, (1, 0)))
    assert found[0][1] > 0.98 and found[1][0] > 0.98
    assert found[0][0] < 0.02 and abs(kept - 1.0) < 1e-14
    with pytest.raises(ValueError):
        _core.state_overlaps(here, ([[1.0, 0.0], [0.0, 1.0]], 1, [[1.0]]))


WATER_BOHR = [[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]]


def _hf() -> EngineCalculator:
    return create(
        "pyscf",
        CalculatorOptions(method="hf", basis="sto-3g", extra={"verbose": 0}),
    )


@needs_pyscf
def test_pyscf_hands_out_its_tamm_dancoff_states() -> None:
    calc = _hf()
    water = Structure.build(["O", "H", "H"], WATER_BOHR)
    found = calc.state_characters(water, 4)
    assert found is not None
    orbitals, amplitudes = found["orbitals"], found["amplitudes"]
    # Seven orbitals, five of them occupied, orthonormal; four states of length one.
    assert orbitals.shape == (7, 7) and found["occupied"] == 5
    assert np.allclose(orbitals.T @ orbitals, np.eye(7), atol=1e-8)
    assert amplitudes.shape == (10, 4)
    assert np.allclose(np.sum(amplitudes**2, axis=0), 1.0)
    energies = found["energies"]
    assert np.all(np.diff(energies) >= 0.0) and energies[0] > 0.1
    # The excitation energies are those of the evaluations of the states.
    ground = calc.evaluate_structure(water).energy
    for k in (1, 2):
        excited = Structure.build(["O", "H", "H"], WATER_BOHR, excited_state=k)
        assert abs(calc.evaluate_structure(excited).energy - ground - energies[k - 1]) < 1e-6
    # A slightly different structure has the same states in the same order.
    moved = np.array(WATER_BOHR)
    moved[1, 1] += 0.02
    other = calc.state_characters(Structure.build(["O", "H", "H"], moved), 4)
    assert other is not None
    overlaps, kept = _core.state_overlaps(
        (orbitals.tolist(), 5, amplitudes.tolist()),
        (other["orbitals"].tolist(), 5, other["amplitudes"].tolist()),
    )
    assert kept > 0.999
    assert all(overlaps[k][k] > 0.95 for k in range(4)), overlaps
    # An open shell has no such states here.
    cation = Structure.build(["O", "H", "H"], WATER_BOHR, charge=1, multiplicity=2)
    assert calc.state_characters(cation, 2) is None


@needs_pyscf
def test_a_command_follows_a_state_of_pyscf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = "\n".join(
        f"{s} {x * BOHR_TO_ANGSTROM:.8f} {y * BOHR_TO_ANGSTROM:.8f} {z * BOHR_TO_ANGSTROM:.8f}"
        for s, (x, y, z) in zip(["O", "H", "H"], WATER_BOHR, strict=True)
    )
    options = ["-os", "pyscf", "-func", "hf", "-bs", "sto-3g", "-es", "1", "-ns", "3"]
    # The run directory is made next to the input.
    for name, more in (("followed", ["-es_follow"]), ("counted", [])):
        directory = tmp_path / name
        directory.mkdir()
        xyz = directory / "water.xyz"
        xyz.write_text(f"3\n\n{rows}\n")
        monkeypatch.chdir(directory)
        assert _core.run_cli("optmain", [str(xyz), *options, *more]) == 0

    def frames(directory: Path) -> Any:
        rows = next(directory.rglob("*_traj.xyz")).read_text().split()
        return np.array([float(v) for v in rows if v[-1].isdigit() and "." in v])

    # No two states change places in these steps: the same structures with and without (to
    # the noise of the SCF, which starts from another density when the states are looked at).
    followed, counted = frames(tmp_path / "followed"), frames(tmp_path / "counted")
    assert followed.shape == counted.shape and followed.size >= 27
    assert np.allclose(followed, counted, atol=1e-5)
    log = next((tmp_path / "followed").rglob("mopt.log")).read_text()
    assert "excited states followed between structures" in log


def test_states_as_diabatic_states() -> None:
    # Two straight states with a constant coupling: E = -+ sqrt((a s)^2 + v^2), and the
    # adiabatic states turn by half the angle of (a s, v).
    a, v = 0.02, 0.004
    arc = [-20.0 + 0.1 * k for k in range(401)]

    def vectors(s: float) -> list[list[float]]:
        t = 0.5 * np.arctan2(v, a * s)
        return [[-np.sin(t), np.cos(t)], [np.cos(t), np.sin(t)]]

    gap = [float(np.hypot(a * s, v)) for s in arc]
    energies = [[-g, g] for g in gap]
    overlaps = [
        (np.array(vectors(s)) @ np.array(vectors(t)).T).tolist()
        for s, t in zip(arc, arc[1:], strict=False)
    ]
    found = _core.diabatic_states(arc, energies, overlaps)
    assert len(found["crossings"]) == 1
    crossing = found["crossings"][0]
    assert crossing["states"] == (0, 1)
    assert abs(crossing["position"]) < 0.05
    assert abs(crossing["coupling"] - v) < 1e-3 * v
    assert abs(crossing["slope_difference"] - 2 * a) < 1e-3 * 2 * a
    assert abs(found["from_gaps"][0]["coupling"] - v) < 1e-9
    weights = np.array(found["weights"])
    assert weights[0, 0, 0] == 1.0 and weights[-1, 0, 1] > 0.99
    assert min(found["kept"]) > 1.0 - 1e-9
    with pytest.raises(ValueError, match="overlap matrices"):
        _core.diabatic_states(arc, energies, overlaps[:-1])

    # Three states that are all there is: the diabatic matrix is the Hamiltonian in the
    # states of the first point, whatever the distance of the points.
    def hamiltonian(s: float) -> np.ndarray:
        h = np.diag([0.02 * s, 0.003 - 0.004 * s, 0.001 - 0.015 * s])
        h[0, 1] = h[1, 0] = 0.002
        h[0, 2] = h[2, 0] = 0.0015
        h[1, 2] = h[2, 1] = 0.001
        return h

    points = [-4.0 + 2.0 * k for k in range(5)]
    solved = [np.linalg.eigh(hamiltonian(s)) for s in points]
    three = _core.diabatic_states(
        points,
        [e.tolist() for e, _ in solved],
        [(u.T @ w).tolist() for (_, u), (_, w) in zip(solved, solved[1:], strict=False)],
    )
    first = solved[0][1]
    for s, potential in zip(points, three["potentials"], strict=True):
        assert np.allclose(potential, first.T @ hamiltonian(s) @ first, atol=1e-12)
    assert {c["states"] for c in three["crossings"]} == {(0, 1), (0, 2), (1, 2)}
    assert len(three["from_gaps"]) == 2

    # exp(-2 pi H12^2 / (v dF)): no coupling stays, a strong one follows the adiabatic state.
    assert _core.landau_zener_stay(0.0, 0.01, 1e-3) == 1.0
    assert _core.landau_zener_stay(0.05, 0.01, 1e-3) < 1e-10
    slow, fast = (_core.landau_zener_stay(1e-4, 2e-3, t) for t in (5e-4, 2e-3))
    assert 0.9 < slow < fast < 1.0

    # The overlaps with their signs: their squares are the squared overlaps.
    orbitals = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    here = (orbitals, 1, [[1.0, 0.0], [0.0, -1.0]])
    there = (orbitals, 1, [[0.6, 0.8], [0.8, -0.6]])
    signed, _ = _core.state_overlaps(here, there, signed=True)
    squared, _ = _core.state_overlaps(here, there)
    assert np.allclose(signed, [[0.6, 0.8], [-0.8, 0.6]])
    assert np.allclose(np.square(signed), squared)
