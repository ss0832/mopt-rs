"""Python access to chain, environment, and electronic-state workflows.

MultiOptPy: no counterpart (component API).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mopt_rs import NativeCalculator, Optimizer, Task, relax_band, relax_environment
from mopt_rs.calculators import (
    Capabilities,
    Engine,
    EngineCalculator,
    EngineResult,
    Structure,
    create,
)
from mopt_rs.calculators._bridge import evaluate_embedded
from mopt_rs.optimize import Checkpoint


def test_native_periodic_axes_and_checkpoint_identity(tmp_path: Path) -> None:
    lj = NativeCalculator("lj")
    ar = Task(
        "pes",
        "lj",
        ["Ar", "Ar"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 24.0]]),
        0,
        1,
        False,
        cell_bohr=np.diag([48.0, 48.0, 32.0]),
        pbc=(True, True, True),
    )
    slab = replace(ar, pbc=(True, True, False))
    slab_energy = lj.evaluate(slab).energy
    assert slab_energy == pytest.approx(-5.564895734444022e-7)
    assert lj.evaluate(ar).energy == pytest.approx(-0.00029681177710069517)
    path = tmp_path / "checkpoint"
    cp = Checkpoint(path)
    for _ in range(3):
        cp.evaluate(lj, ar)
    cp.close()
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    path.write_bytes(data + b'{"unfinished":\r\n')
    for _ in range(2):
        cp = Checkpoint(path)
        assert len(cp.pending) == 3
        cp.close()
        assert path.read_bytes() == data
    cp = Checkpoint(path)
    for changed in [
        slab,
        replace(ar, cell_bohr=np.diag([48.0, 48.0, 60.0])),
        replace(ar, symbols=["Ne", "Ne"]),
    ]:
        assert cp.evaluate(lj, changed)[0] == pytest.approx(lj.evaluate(changed).energy)
        assert cp.replayed == 0
    cp.evaluate(lj, ar)
    assert cp.replayed == 1
    cp.close()


def test_native_state_and_three_dimensional_gfnff() -> None:
    water = Task(
        "pes",
        "gfn1xtb",
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]]),
        0,
        1,
        False,
    )
    calc = NativeCalculator("gfn1xtb")
    assert calc.evaluate(water).energy == pytest.approx(-5.768629100383)
    assert calc.evaluate(replace(water, excited_state=1)).energy == pytest.approx(-5.425941069517)
    cell = Structure.build(water.symbols, water.positions_bohr, cell_bohr=np.eye(3) * 25)
    assert np.isfinite(create("gfnff").evaluate_structure(cell).energy)
    opt = Optimizer(
        water.symbols,
        water.positions_bohr.ravel().tolist(),
        model_function=["seam"],
        excited_state=0,
    )
    assert [t["excited_state"] for t in opt.ask()] == [0, 1]


def test_md_high_layer_selects_matching_positions_and_velocities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mopt_rs import _core

    frames = {}
    for mode in ("off", "on"):
        work = tmp_path / mode
        work.mkdir()
        monkeypatch.chdir(work)
        Path("argon.xyz").write_text("4\ncluster\nAr 0 0 0\nAr 3.8 0 0\nAr 0 3.8 0\nAr 0 0 3.8\n")
        args = [
            "argon.xyz",
            "-os",
            "lj",
            "-oniom",
            "2,4",
            "none",
            "lj",
            "-oniom_middle",
            "1,2,4",
            "none",
            "lj",
            "-oniom_opt",
            f"traj_high_only={mode}",
            "-time",
            "2",
            "-md_opt",
            "seed=5",
            "write_every=1",
        ]
        assert _core.run_cli("mdmain", args) == 0
        for kind in ("traj", "velocities"):
            lines = next(work.rglob(f"argon_{kind}.xyz")).read_text().splitlines()
            n = 4 if mode == "off" else 2
            assert all(int(line) == n for line in lines[:: n + 2])
            frames[mode, kind] = [lines[i + 2 : i + 2 + n] for i in range(0, len(lines), n + 2)]
        assert int(next(work.rglob("argon_final.xyz")).read_text().splitlines()[0]) == 4
    for kind in ("traj", "velocities"):
        assert frames["on", kind] == [[frame[1], frame[3]] for frame in frames["off", kind]]


def test_microiterations_with_registry_method() -> None:
    def harmonic(
        x: list[float], need_hessian: bool
    ) -> tuple[float, list[float], list[float] | None]:
        return x[0] ** 2 / 2, x, [1.0] if need_hessian else None

    for method in ("lbfgs", "fire"):
        result = relax_environment(harmonic, [0.25], method=method, has_hessian=True)
        assert result["converged"] and abs(result["positions"][0]) < 1e-5


def test_chain_driver_with_observer_and_files(tmp_path: Path) -> None:
    seen = []

    def surface(x: list[float], image: int, hessian: bool) -> tuple[float, list[float], None]:
        return x[1] ** 2, [0.0, 2 * x[1]], None

    def observe(
        iteration: int, images: list[list[float]], energies: list[float], largest: float
    ) -> bool:
        seen.append((iteration, len(images)))
        return True

    result = relax_band(
        surface,
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
        inserted=1,
        surrogate=True,
        output_dir=str(tmp_path),
        observer=observe,
    )
    assert seen == [(0, 5)]
    assert len(result["images"]) == 5 and "predicted_images" in result
    assert (tmp_path / "energy_plot.csv").is_file()
    assert list(tmp_path.glob("*.png")) or list(tmp_path.rglob("*.svg"))


def test_internal_chain_calls_surface_in_cartesian_coordinates() -> None:
    atoms = ["O", "H", "H"]
    x = np.array([0.0, 0.0, 0.0, 1.8, 0.0, 0.0, -0.45, 1.75, 0.0])

    def harmonic(p: list[float], i: int, hessian: bool) -> tuple[float, list[float], None]:
        assert len(p) == 9
        d = np.asarray(p) - x
        return float(d @ d / 2), d.tolist(), None

    result = relax_band(
        harmonic,
        [x.tolist(), (x * 1.01).tolist(), (x * 1.02).tolist()],
        symbols=atoms,
        coordinates="dic",
        max_iterations=1,
        # The images as they are (-nd 0): nebmain spaces them 0.5 Angstrom apart by default.
        options=["-nd", "0"],
    )
    assert np.shape(result["images"]) == (3, 9)


def test_embedded_python_surface_returns_both_gradients() -> None:
    class Field(Engine):
        capabilities = Capabilities(point_charges=True, electrons=False)

        def compute(self, structure: Structure, hessian: bool) -> EngineResult:
            raise AssertionError("embedded evaluation called the vacuum surface")

        def compute_embedded(
            self,
            structure: Structure,
            positions_bohr: Any,
            charges: Any,
            elements: Any,
            hessian: bool,
        ) -> tuple[EngineResult, Any]:
            delta = structure.positions_bohr[0] - positions_bohr[0]
            return EngineResult(float(delta @ delta / 2), gradient=delta), -delta.reshape(1, 3)

    e, g, _, q = evaluate_embedded(
        EngineCalculator(Field()),
        ["He"],
        [2.0, 0.0, 0.0],
        0,
        1,
        0,
        False,
        [[1.0, 0.0, 0.0]],
        [1.0],
        ["H"],
    )
    assert e == 0.5 and g == [1.0, 0.0, 0.0] and q == [[-1.0, 0.0, 0.0]]
