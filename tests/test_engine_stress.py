"""The stress of a periodic cell from Python engines: asked for when it is needed, handed
to the Rust side, and used by the variable-cell optimization.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import fake_ase
from mopt_rs import _core
from mopt_rs.calculators import AseEngine, EngineCalculator, Structure, create

HAS_ASE = importlib.util.find_spec("ase") is not None
needs_ase = pytest.mark.skipif(not HAS_ASE, reason="ASE not installed")

ANG = 1.0 / 0.529177210903
EV = 1.0 / 27.211386245988


@pytest.fixture(autouse=True)
def an_ase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ASE, or its stand-in where it is not installed; runs write into an empty directory."""
    if not HAS_ASE:
        monkeypatch.setitem(sys.modules, "ase", fake_ase.module())
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    fake_ase.VolumeSpring.calls.update(energy=0, stress=0)


def cubic(edge_angstrom: float, shift: float = 0.0) -> Structure:
    """One argon atom in a cube."""
    cell = np.eye(3) * edge_angstrom * ANG
    return Structure.build(["Ar"], [[shift, 0.0, 0.0]], cell_bohr=cell)


def test_the_stress_is_asked_for_when_it_is_needed() -> None:
    engine = AseEngine(fake_ase.VolumeSpring, kwargs={"v0": 64.0, "k": 0.002}, electrons=False)
    calc = EngineCalculator(engine)
    calls = fake_ase.VolumeSpring.calls
    # Not asked for: none is computed.
    plain = calc.evaluate_structure(cubic(4.3))
    assert plain.stress is None and calls == {"energy": 1, "stress": 0}
    # Asked for at the same structure: the engine runs again, with the stress.
    ev = calc.evaluate_structure(cubic(4.3), need_stress=True)
    assert calls["stress"] == 1
    assert ev.stress is not None and ev.stress.shape == (3, 3)
    expected = 0.002 * (4.3**3 - 64.0) * EV / ANG**3
    assert np.allclose(ev.stress, expected * np.eye(3), rtol=1e-9, atol=1e-18)
    assert ev.energy == pytest.approx(0.5 * 0.002 * (4.3**3 - 64.0) ** 2 * EV)
    # ... and is remembered with it.
    again = calc.evaluate_structure(cubic(4.3), need_stress=True)
    assert calls["stress"] == 1 and again.stress is not None
    assert np.array_equal(again.stress, ev.stress)
    assert calc.evaluate_structure(cubic(4.3)).stress is None
    # The stress is the derivative of the energy with respect to a strain, per volume.
    step = 1e-4
    volume = (4.3 * ANG) ** 3
    high = calc.evaluate_structure(cubic(4.3 * (1.0 + step))).energy
    low = calc.evaluate_structure(cubic(4.3 * (1.0 - step))).energy
    assert np.trace(ev.stress) == pytest.approx((high - low) / (2.0 * step) / volume, rel=1e-6)

    # A calculator without a stress is asked once and gives none; a structure without a
    # cell has none.
    springs = AseEngine(fake_ase.Harmonic, kwargs={"x0": [[0.0, 0.0, 0.0]]}, electrons=False)
    without = EngineCalculator(springs)
    assert without.evaluate_structure(cubic(4.0, 0.1), need_stress=True).stress is None
    assert without.evaluate_structure(cubic(4.0, 0.1), need_stress=True).stress is None
    assert without.evaluations == 1
    free = Structure.build(["Ar"], [[0.1, 0.0, 0.0]])
    assert without.evaluate_structure(free, need_stress=True).stress is None

    # An engine may hand out the tensor itself.
    class Tensor(AseEngine):
        def compute(self, structure, hessian, stress=False):  # type: ignore[no-untyped-def]
            result = super().compute(structure, hessian, False)
            if stress:
                result.extra["stress_hartree_per_bohr3"] = [[1e-4, 2e-5, 0.0]] * 3
            return result

    own = EngineCalculator(
        Tensor(fake_ase.Harmonic, kwargs={"x0": [[0.0, 0.0, 0.0]]}, electrons=False)
    ).evaluate_structure(cubic(4.0, 0.1), need_stress=True)
    assert own.stress is not None and np.array_equal(own.stress, own.stress.T)
    assert own.stress[0, 1] == pytest.approx(0.5 * (2e-5 + 1e-4))


def lattice(text: str) -> np.ndarray:
    """The lattice vectors of the comment line of an extended xyz file."""
    values = text.split('Lattice="')[1].split('"')[0].split()
    return np.array([float(v) for v in values]).reshape(3, 3)


def test_variable_cell_optimization_with_the_stress_of_the_engine(tmp_path: Path) -> None:
    # tests/ is where the calculator of ``-os ase:fake_ase:VolumeSpring`` is found.
    (tmp_path / "ar.xyz").write_text(
        '1\nLattice="4.3 0.0 0.0 0.0 4.3 0.0 0.0 0.0 4.3" pbc="T T T"\nAr 0.0 0.0 0.0\n'
    )
    arguments = ["ar.xyz", "-os", "ase:fake_ase:VolumeSpring", "-vc", "-opt", "rsirfo_bfgs"]
    code = _core.run_cli("optmain", [*arguments, "-calc_opt", "v0=64.0", "k=0.002"])
    assert code == 0
    newest = max(tmp_path.rglob("ar_optimized.xyz"), key=lambda p: p.stat().st_mtime)
    volume = abs(np.linalg.det(lattice(newest.read_text())))
    assert volume == pytest.approx(64.0, rel=1e-3)
    # One calculation per structure: the stress comes from the calculator, not from
    # energies of strained cells (twelve more per structure).
    calls = fake_ase.VolumeSpring.calls
    assert calls["stress"] > 1 and calls["energy"] == calls["stress"]


@needs_ase
def test_stress_of_ase_emt_against_strains_and_the_built_in_emt(tmp_path: Path) -> None:
    # Copper, four atoms off their sites in a sheared cell.
    edge = 3.55 * ANG
    cell = edge * np.array([[1.0, 0.0, 0.0], [0.06, 0.97, 0.0], [-0.04, 0.05, 1.03]])
    half = 0.5 * edge
    x = np.array(
        [
            [0.05, 0.02, -0.03],
            [half + 0.04, half - 0.06, 0.02],
            [half - 0.03, 0.05, half + 0.07],
            [-0.02, half + 0.03, half - 0.05],
        ]
    )

    def copper(strain: np.ndarray | None = None) -> Structure:
        deform = np.eye(3) if strain is None else np.eye(3) + strain
        return Structure.build(["Cu"] * 4, x @ deform, cell_bohr=cell @ deform)

    calc = create("ase:ase.calculators.emt:EMT")
    ev = calc.evaluate_structure(copper(), need_stress=True)
    assert ev.stress is not None
    volume = abs(np.linalg.det(cell))
    step = 1e-4
    numeric = np.zeros((3, 3))
    for a in range(3):
        for b in range(a, 3):
            strain = np.zeros((3, 3))
            strain[a, b] += 0.5 * step
            strain[b, a] += 0.5 * step
            high = calc.evaluate_structure(copper(strain)).energy
            low = calc.evaluate_structure(copper(-strain)).energy
            numeric[a, b] = numeric[b, a] = (high - low) / (2.0 * step * volume)
    scale = np.abs(numeric).max()
    assert np.allclose(ev.stress, numeric, atol=1e-6 * scale)
    # The EMT of mopt-rs is the same potential: the same energy and the same stress.
    energy, _, _ = _core.evaluate_native(
        "emt", ["Cu"] * 4, x.ravel().tolist(), 0, 1, False, cell.ravel().tolist()
    )
    assert energy == pytest.approx(ev.energy, abs=1e-9)
    native = np.array(
        _core.native_stress("emt", ["Cu"] * 4, x.ravel().tolist(), cell.ravel().tolist())
    )
    assert np.allclose(ev.stress, native, atol=1e-8 * scale)

    # The variable-cell optimization ends at the same crystal with either.
    rows = " ".join(f"{v / ANG:.10f}" for v in cell.ravel())
    atoms = "".join(f"Cu {p[0] / ANG:.10f} {p[1] / ANG:.10f} {p[2] / ANG:.10f}\n" for p in x)
    (tmp_path / "cu.xyz").write_text(f'4\nLattice="{rows}" pbc="T T T"\n{atoms}')
    volumes = {}
    for name in ("ase:ase.calculators.emt:EMT", "emt"):
        code = _core.run_cli("optmain", ["cu.xyz", "-os", name, "-vc", "-opt", "rsirfo_bfgs"])
        assert code == 0, name
        newest = max(tmp_path.rglob("cu_optimized.xyz"), key=lambda p: p.stat().st_mtime)
        volumes[name] = abs(np.linalg.det(lattice(newest.read_text())))
    assert volumes["emt"] == pytest.approx(volumes["ase:ase.calculators.emt:EMT"], rel=1e-4)
