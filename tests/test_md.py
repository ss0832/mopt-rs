"""Molecular dynamics from Python: the ask/tell state machine, the Python driver, and
``mdmain`` with an ASE calculator (ASE, or the stand-in of ``fake_ase`` without it).

MultiOptPy: `moleculardynamics.py:MD`, `Wrapper/md_wrapper.py:MDJob`.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fake_ase
import mopt_rs
from mopt_rs import MolecularDynamics, NativeCalculator, _core, run_md
from mopt_rs._core import md_parts

HAS_ASE = importlib.util.find_spec("ase") is not None
ANG = 1.0 / 0.529177210903


def oscillator(x: np.ndarray) -> tuple[float, np.ndarray]:
    return 0.5 * float(x @ x), x.copy()


def test_ask_and_tell_conserve_the_energy_of_an_oscillator() -> None:
    md = MolecularDynamics([0.0], time_step=0.05, steps=400, velocities=[1.0])
    reports = []
    while (request := md.ask()) is not None:
        value, g = oscillator(np.asarray(request["x"]))
        reports.append(md.tell(value, g.tolist()))
    assert md.finished and md.evaluations == 401
    energies = np.array([r["conserved"] for r in reports])
    assert np.max(np.abs(energies - 0.5)) < 1e-3
    assert reports[0]["kinetic"] == pytest.approx(0.5)
    integrators, thermostats, barostats, ensembles = md_parts()
    assert "baoab" in integrators and "bussi" in thermostats and "mtk" in barostats
    assert "npt" in ensembles


def test_the_saddle_direction_of_muller_brown_stays_frozen() -> None:
    seen: list[int] = []

    def watch(report: dict[str, Any]) -> bool:
        seen.append(int(report["step"]))
        return int(report["step"]) < 300

    result = run_md(
        "mb2d",
        [-0.822002, 0.624313],
        ensemble="nve",
        time_step=0.01,
        steps=2000,
        temperature=200.0,
        options={"freeze": "negative", "seed": 8},
        callback=watch,
    )
    assert result.interrupted and seen[-1] == 300
    assert result.dof == 1.0
    assert result.reports[0]["freeze"]["eigenvalues"][0] < 0.0
    distance = np.linalg.norm(result.positions - np.array([-0.822002, 0.624313]), axis=1)
    assert np.max(distance) < 0.15
    assert np.ptp(result.conserved) < 1e-6


def test_a_python_function_with_a_thermostat_and_the_same_seed() -> None:
    a = np.diag([1.0, 2.0, 3.0, 4.0])

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.5 * float(x @ a @ x), a @ x

    runs = [
        run_md(
            quadratic,
            np.zeros(4),
            ensemble="langevin",
            time_step=0.1,
            steps=300,
            temperature=400.0,
            options={"tau": 0.5, "seed": 31},
        )
        for _ in range(2)
    ]
    assert np.array_equal(runs[0].positions, runs[1].positions)
    assert runs[0].parts == ("baoab", "langevin", None)


def test_structures_with_a_built_in_calculator() -> None:
    symbols = ["Ar"] * 3
    x = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 3.3, 0.0]]) * ANG
    result = run_md(
        NativeCalculator("lj"),
        x,
        symbols=symbols,
        ensemble="nve",
        time_step=2.0 / 0.024188843265857,
        steps=200,
        temperature=20.0,
        options={"seed": 2},
    )
    assert result.dof == 3.0
    assert np.ptp(result.conserved) < 1e-7
    # A frozen atom does not move.
    held = run_md(
        NativeCalculator("lj"),
        x,
        symbols=symbols,
        ensemble="langevin",
        steps=50,
        temperature=20.0,
        fix_atoms=[1],
        options={"seed": 2},
    )
    assert np.allclose(held.positions[:, :3], x[0])


def test_dynamics_in_delocalized_internal_coordinates() -> None:
    # The Ar3 triangle: three internal coordinates; the same start as Cartesian dynamics
    # without rigid motion, the same trajectory of the distances.
    symbols = ["Ar"] * 3
    x = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 3.3, 0.0]]) * ANG
    common: dict[str, Any] = {
        "symbols": symbols,
        "ensemble": "nve",
        "time_step": 2.0 / 0.024188843265857,
        "steps": 200,
        "temperature": 20.0,
        "options": {"seed": 2},
    }
    cart = run_md(NativeCalculator("lj"), x, **common)
    internal = run_md(NativeCalculator("lj"), x, coordinates="dic", **common)
    assert internal.dof == 3.0
    assert np.ptp(internal.conserved) < 1e-7
    assert abs(cart.reports[0]["kinetic"] - internal.reports[0]["kinetic"]) < 1e-14

    def r12(positions: np.ndarray) -> np.ndarray:
        d: np.ndarray = np.linalg.norm(positions[:, 0:3] - positions[:, 3:6], axis=1)
        return d

    assert np.max(np.abs(r12(cart.positions) - r12(internal.positions))) < 1e-4
    with pytest.raises(ValueError, match="redundant"):
        MolecularDynamics(x.ravel().tolist(), symbols=symbols, coordinates="ric")


def test_a_python_constraint_function_holds_its_value() -> None:
    def distance(
        positions: list[float], cell: list[float] | None, iteration: int, need_hessian: bool
    ) -> tuple[float, list[float], None]:
        x = np.asarray(positions).reshape(-1, 3)
        d = x[0] - x[1]
        r = float(np.linalg.norm(d))
        g = np.zeros_like(x)
        g[0], g[1] = d / r, -d / r
        return r, g.ravel().tolist(), None

    x = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 3.3, 0.0]]) * ANG
    result = run_md(
        NativeCalculator("lj"),
        x,
        symbols=["Ar"] * 3,
        ensemble="nve",
        steps=100,
        temperature=30.0,
        constraint_functions=[distance],
        options={"seed": 6},
    )
    frames = result.positions.reshape(len(result.positions), -1, 3)
    lengths = np.linalg.norm(frames[:, 0] - frames[:, 1], axis=1)
    assert np.ptp(lengths) < 1e-8
    assert result.dof == 2.0
    assert max(r["constraint_residual"] for r in result.reports) < 1e-9


def test_wham_and_mbar_of_a_flat_surface() -> None:
    rng = np.random.default_rng(3)
    kt, k = 1.0e-3, 0.05
    centers = np.linspace(-1.0, 1.0, 9)
    # On a flat surface every window samples a Gaussian around its centre.
    samples = [rng.normal(c, np.sqrt(kt / k), 4000).tolist() for c in centers]
    for method in (_core.wham, _core.mbar):
        profile = method(centers.tolist(), [k], samples, kt, bins=18, range=(-0.9, 0.9))
        assert len(profile["centers"]) == 18
        assert np.ptp(profile["free_energy"]) < 0.3 * kt
        assert profile["window_free_energies"][0] == 0.0


@pytest.fixture
def an_ase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not HAS_ASE:
        monkeypatch.setitem(sys.modules, "ase", fake_ase.module())
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_mdmain_with_an_ase_calculator(an_ase: Path) -> None:
    (an_ase / "pair.xyz").write_text("2\n\nAr 0.0 0.0 0.0\nAr 0.0 0.0 1.2\n")
    arguments = ["pair.xyz", "-os", "ase:fake_ase:Harmonic", "-mt", "bussi", "-time", "40"]
    options = ["-calc_opt", "x0=[[0.0,0.0,0.0],[0.0,0.0,1.0]]", "k=2.0", "-md_opt", "seed=5"]
    assert mopt_rs._core.run_cli("mdmain", arguments + options) == 0
    runs = [p for p in (an_ase / ".").rglob("pair_MD_ASE_*") if p.is_dir()]
    assert len(runs) == 1
    rows = (runs[0] / "md_properties.csv").read_text().splitlines()
    assert len(rows) == 42
    assert (runs[0] / "pair_traj.xyz").is_file()
    log = (runs[0] / "mopt.log").read_text()
    assert "Evaluations (this input): 41 gradient calculation(s)" in log


def test_constant_pressure_with_the_stress_of_an_ase_calculator(an_ase: Path) -> None:
    (an_ase / "ar.xyz").write_text(
        '1\nLattice="4.3 0.0 0.0 0.0 4.3 0.0 0.0 0.0 4.3" pbc="T T T"\nAr 0.0 0.0 0.0\n'
    )
    arguments = ["ar.xyz", "-os", "ase:fake_ase:VolumeSpring", "-mt", "npt_berendsen"]
    options = [
        "-time",
        "200",
        "-ts",
        "2fs",
        "-calc_opt",
        "v0=64.0",
        "k=0.002",
        "-md_opt",
        "rigid=none",
        "seed=1",
        "tau_p=100",
        "compressibility=1.0",
    ]
    assert mopt_rs._core.run_cli("mdmain", arguments + options) == 0
    run = next(p for p in an_ase.rglob("ar_MD_ASE_*") if p.is_dir())
    rows = (run / "md_properties.csv").read_text().splitlines()
    volumes = [float(r.split(",")[10]) for r in rows[1:]]
    assert abs(volumes[-1] - 64.0) < abs(volumes[0] - 64.0)


def test_surface_hopping_with_functions_of_ones_own_for_the_states() -> None:
    """Zhu-Nakamura surface hopping on two states given only as functions (the plugin form):
    straight diabatic states -F_k x with the coupling V along one coordinate of mass mu,
    passed once from the lower state; the share that ends on the upper state is the
    probability of eq. 2.11 and the energy is kept across the hops."""
    f1, f2, v, mu = 0.02, 0.005, 0.002, 918.6
    above = 0.0055
    calls = {"energy only": 0}

    def adiabats(x: float) -> tuple[list[float], list[float]]:
        mean, half = -0.5 * (f1 + f2) * x, -0.5 * (f1 - f2) * x
        root = (half * half + v * v) ** 0.5
        slope = half * (-0.5 * (f1 - f2)) / root
        return [mean - root, mean + root], [-0.5 * (f1 + f2) - slope, -0.5 * (f1 + f2) + slope]

    def ground(x: np.ndarray) -> tuple[float, np.ndarray]:
        e, g = adiabats(float(x[0]))
        return e[0], np.array([g[0]])

    def excited(x: np.ndarray, state: int, need_gradient: bool) -> tuple[float, Any]:
        e, g = adiabats(float(x[0]))
        if not need_gradient:
            calls["energy only"] += 1
            return e[state], None
        return e[state], np.array([g[state]])

    a2 = (f1 * f2) ** 0.5 * (f1 - f2) / (2.0 * mu * (2.0 * v) ** 3)
    b2 = above * (f1 - f2) / (2.0 * (f1 * f2) ** 0.5 * v)
    root = (b2 * b2 + 0.4 * a2 + 0.7) ** 0.5
    exact = float(np.exp(-np.pi / (4.0 * a2**0.5) * (2.0 / (b2 + root)) ** 0.5))
    n, upper, estimates = 150, 0, []
    for k in range(n):
        x0 = -0.8 - 0.1 * k / n
        speed = (2.0 * (above - adiabats(x0)[0][0]) / mu) ** 0.5
        result = run_md(
            None,
            [x0],
            masses=[mu],
            time_step=10.0,
            steps=60,
            velocities=[speed],
            options={"seed": 100 + k, "hopping_states": "0,1"},
            ground=ground,
            excited=excited,
        )
        last = result.reports[-1]
        upper += int(last["electronic_state"] == 1)
        drift = np.abs(result.conserved - result.conserved[0]).max()
        assert drift < 2e-5, drift
        estimates += [h["probability"] for r in result.reports for h in r["hops"]]
    assert calls["energy only"] > 0
    assert abs(np.mean(estimates) / exact - 1.0) < 0.05, (np.mean(estimates), exact)
    sigma = (exact * (1.0 - exact) / n) ** 0.5
    assert abs(upper / n - exact) < 4.0 * sigma, (upper / n, exact)
