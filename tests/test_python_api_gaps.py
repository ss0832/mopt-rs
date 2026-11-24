"""The parts of the Python API that a workflow built from the building blocks needs: structure
files, constraints and cells of chains and IRCs, the defaults and methods of ``nebmain`` in
``relax_band``, an initial Hessian of ``Optimizer``, the ONIOM calculator, and the name of the
distribution.

MultiOptPy: no counterpart (Python access to the Rust workflows).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from mopt_rs import OniomCalculator, _core
from mopt_rs.irc import irc_path
from mopt_rs.optimize import Evaluation, NativeCalculator, Optimizer, Task, run
from mopt_rs.workflows import relax_band
from mopt_rs.xyz import XyzFrame, append_xyz, read_xyz, read_xyz_frames, write_xyz

ANG = _core.ANGSTROM_TO_BOHR
WATER = ["O", "H", "H"]
WATER_ANGSTROM = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])


# --- structure files ---
def test_xyz_frames_are_written_and_read_by_the_library(tmp_path: Path) -> None:
    path = tmp_path / "frames.xyz"
    shifted = WATER_ANGSTROM + 0.123456789012
    write_xyz(path, WATER, WATER_ANGSTROM, comment="first")
    append_xyz(path, [(WATER, shifted, "second"), (WATER, shifted.ravel())])
    frames = read_xyz_frames(path)
    assert [f.comment for f in frames] == ["first", "second", ""]
    assert all(f.symbols == WATER and f.lattice_bohr is None for f in frames)
    # Twelve decimals of Angstrom: the values come back to within 5e-13 Angstrom.
    assert np.max(np.abs(frames[1].positions_angstrom - shifted)) < 1e-12
    assert np.max(np.abs(read_xyz(path).positions_bohr - WATER_ANGSTROM * ANG)) < 1e-11
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 4:
            assert all(re.fullmatch(r"-?\d+\.\d{12}", v) for v in fields[1:]), line
    # Writing replaces the file.
    write_xyz(path, frames[:1])
    assert len(read_xyz_frames(path)) == 1


def test_extended_xyz_keeps_the_lattice_and_the_state(tmp_path: Path) -> None:
    path = tmp_path / "cell.xyz"
    lattice = np.diag([10.0, 11.0, 12.5]) + np.array([[0, 0, 0], [0.3, 0, 0], [0, 0, 0]])
    write_xyz(path, WATER, WATER_ANGSTROM, comment="cell", lattice_angstrom=lattice)
    frame = read_xyz(path)
    assert frame.lattice_angstrom is not None
    assert np.max(np.abs(frame.lattice_angstrom - lattice)) < 1e-14
    assert frame.pbc == (True, True, True)
    header = path.read_text(encoding="utf-8").splitlines()[1]
    numbers = re.search(r'Lattice="([^"]*)"', header)
    assert numbers is not None
    assert all(re.fullmatch(r"-?\d+\.\d{15}", v) for v in numbers.group(1).split())
    # The frame object goes back out unchanged.
    write_xyz(tmp_path / "again.xyz", [frame])
    again = read_xyz(tmp_path / "again.xyz")
    assert again.lattice_bohr is not None and np.array_equal(again.lattice_bohr, frame.lattice_bohr)
    assert again.comment.count("Lattice=") == 1 and again.comment.endswith("cell")
    # MultiOptPy's line of the state.
    state = tmp_path / "state.xyz"
    state.write_text("3\n-1 2\nO 0 0 0\nH 0 0 1\nH 0 1 0\n", encoding="utf-8")
    assert read_xyz(state).state == (-1, 2)
    assert isinstance(read_xyz(state), XyzFrame)


# --- the distribution ---
def test_the_installed_distribution_is_mopt_py() -> None:
    from importlib import metadata

    found = metadata.distribution("mopt-py")
    assert found.metadata["Name"] == "mopt-py"
    # Imported as mopt_rs, with the version of the library.
    assert "mopt-py" in metadata.packages_distributions()["mopt_rs"]
    assert found.version == _core.__version__

# --- the initial Hessian of Optimizer ---
AR4 = ["Ar"] * 4
AR4_ANGSTROM = np.array(
    [[0.0, 0.0, 0.0], [3.9, 0.1, 0.0], [1.8, 3.5, 0.2], [1.9, 1.2, 3.3]], dtype=np.float64
)


class Counted:
    """A built-in calculator that counts the Hessians it is asked for."""

    def __init__(self) -> None:
        self.calculator = NativeCalculator("lj")
        self.hessians = 0

    def evaluate(self, task: Task) -> Evaluation:
        self.hessians += int(task.need_hessian)
        return self.calculator.evaluate(task)


def test_an_initial_hessian_gives_the_first_step_of_optmain_ih(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = (AR4_ANGSTROM * ANG).ravel()
    n = x.size
    _, _, flat = _core.evaluate_native("lj", AR4, x.tolist(), need_hessian=True)
    assert flat is not None
    exact = np.reshape(flat, (n, n))
    write_xyz(tmp_path / "ar4.xyz", AR4, AR4_ANGSTROM)
    np.save(tmp_path / "exact.npy", exact)
    monkeypatch.chdir(tmp_path)
    # A run stopped by its iteration limit returns the last structure it evaluated: after two
    # iterations that is the structure the first step led to.
    method = ["-opt", "rsirfo_block_fsb", "-ns", "2"]
    jobs = _core.run_optmain(["ar4.xyz", "-os", "lj", "-ih", "exact.npy", *method], False)
    cli = np.ravel(jobs[0]["final_geometry"])

    def first_step(**given: object) -> np.ndarray:
        opt = Optimizer(AR4, x.tolist(), methods=["rsirfo_block_fsb"], max_iterations=1, **given)
        engine = Counted()
        result = run(opt, engine)
        assert result.iterations == 1
        assert engine.hessians == 0, "the Hessian given is not asked of the calculator"
        return result.positions_bohr.ravel()

    for given in (tmp_path / "exact.npy", str(tmp_path / "exact.npy"), exact, exact.ravel()):
        assert np.max(np.abs(first_step(initial_hessian=given) - cli)) < 1e-10
    # Without it the first step is another one.
    assert np.max(np.abs(first_step() - cli)) > 1e-6
    # A name is the model Hessian of -modelhess.
    named = first_step(initial_hessian="fischerd3")
    assert np.array_equal(named, first_step(model_hessian="fischerd3"))
    with pytest.raises(ValueError, match="neither a file nor a model Hessian"):
        Optimizer(AR4, x.tolist(), initial_hessian="no_such_thing")
    with pytest.raises(ValueError, match="matrix"):
        Optimizer(AR4, x.tolist(), initial_hessian=np.eye(3))
    with pytest.raises(ValueError, match="not symmetric"):
        Optimizer(AR4, x.tolist(), initial_hessian=exact + np.triu(np.ones((n, n)), 1))

# --- relax_band follows nebmain ---
AR4_PLANAR = np.array(
    [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 3.3, 0.0], [5.7, 3.3, 0.0]], dtype=np.float64
)
CU4 = ["Cu"] * 4
CU4_START = np.array(
    [[5.0, 5.0, 5.0], [7.5, 5.0, 5.0], [6.25, 7.2, 5.0], [6.25, 5.7, 7.1]], dtype=np.float64
)
CU4_END = np.array(
    [[5.0, 5.0, 5.0], [7.5, 5.0, 5.0], [6.25, 7.2, 5.0], [6.25, 6.4, 2.9]], dtype=np.float64
)
CELL = np.diag([14.0, 14.0, 14.0])


def last_chain_of_nebmain(tmp_path: Path, arguments: list[str]) -> list[float]:
    """The energies of the last chain of a run of nebmain in ``tmp_path``."""
    assert _core.run_cli("nebmain", [*arguments, "-neb_opt", "figures=off"]) == 0
    runs = [p for p in tmp_path.rglob("*_NEB_*") if p.is_dir()]
    assert len(runs) == 1
    table = (runs[0] / "energy_plot.csv").read_text(encoding="utf-8").splitlines()
    return [float(v) for v in table[-1].split(",")]


def test_relax_band_without_options_is_nebmain_without_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_xyz("ar4.xyz", [(AR4, AR4_ANGSTROM, "start"), (AR4, AR4_PLANAR, "end")])
    command = last_chain_of_nebmain(tmp_path, ["ar4.xyz", "-os", "lj"])
    images = [(x * ANG).ravel().tolist() for x in (AR4_ANGSTROM, AR4_PLANAR)]
    band = relax_band("lj", images, symbols=AR4)
    assert len(band["energies"]) == len(command)
    assert np.max(np.abs(np.asarray(band["energies"]) - command)) < 1e-10
    # A Python function of the same surface relaxes the same chain.
    engine = _core.NativeEvaluator("lj")

    def surface(x: list[float], image: int, need_hessian: bool) -> tuple[float, list[float], None]:
        energy, gradient, _ = engine.evaluate(AR4, x)
        return energy, gradient, None

    by_function = relax_band(surface, images, symbols=AR4)
    assert np.max(np.abs(np.asarray(by_function["energies"]) - command)) < 1e-10


def test_relax_band_in_a_cell_is_nebmain_in_the_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_xyz(
        "cu4.xyz",
        [(CU4, CU4_START, "start"), (CU4, CU4_END, "end")],
        lattice_angstrom=CELL,
    )
    command = last_chain_of_nebmain(tmp_path, ["cu4.xyz", "-os", "emt", "-ns", "8"])
    images = [(x * ANG).ravel().tolist() for x in (CU4_START, CU4_END)]
    cell = (CELL * ANG).ravel().tolist()
    band = relax_band("emt", images, symbols=CU4, cell_bohr=cell, max_iterations=8)
    assert np.max(np.abs(np.asarray(band["energies"]) - command)) < 1e-10
    assert any("Periodic cell" in note for note in band["notes"])


def test_relax_band_takes_the_methods_of_nebmain_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    start, end = tmp_path / "start.txt", tmp_path / "end.txt"
    start.write_text("x -0.558\ny 1.442\n")
    end.write_text("x 0.623\ny 0.028\n")
    points = [[-0.558, 1.442], [0.623, 0.028]]
    for extra in ([], ["-lup"], ["-ewbneb"], ["-qsmv2"], ["-modelhess", "none"]):
        method = ["-p", "8", *extra]
        command = last_chain_of_nebmain(tmp_path, [str(start), str(end), "-os", "mb2d", *method])
        band = relax_band("mb2d", points, options=method)
        assert np.max(np.abs(np.asarray(band["energies"]) - command)) < 1e-10, method
        for run in list(tmp_path.rglob("*_NEB_*")):
            if run.is_dir():
                shutil.rmtree(run)
    # The keyword arguments are the options they name.
    named = relax_band("mb2d", points, options=["-p", "8", "-ns", "5"])
    by_keyword = relax_band("mb2d", points, inserted=8, max_iterations=5)
    assert named["energies"] == by_keyword["energies"]
    # What the commands refuse is refused with their reason.
    with pytest.raises(ValueError, match="one of them"):
        relax_band("mb2d", points, options=["-qsm", "-qsmv2"])
    # -vc: the chain of the atoms and the cell of nebmain -vc, the cell of every image with it.
    write_xyz(
        "cu4.xyz",
        [(CU4, CU4_START, "start"), (CU4, CU4_END, "end")],
        lattice_angstrom=CELL,
    )
    vc = ["-vc", "-ns", "4", "-modelhess", "none"]
    command = last_chain_of_nebmain(tmp_path, ["cu4.xyz", "-os", "emt", *vc])
    cu = [(x * ANG).ravel().tolist() for x in (CU4_START, CU4_END)]
    cell = (CELL * ANG).ravel().tolist()
    band = relax_band("emt", cu, symbols=CU4, cell_bohr=cell, options=vc)
    assert np.max(np.abs(np.asarray(band["energies"]) - command)) < 1e-10
    assert len(band["cells_bohr"]) == len(command)
    assert all(len(c) == 9 for c in band["cells_bohr"])
    # Without a cell there is nothing to vary.
    images = [(x * ANG).ravel().tolist() for x in (AR4_ANGSTROM, AR4_PLANAR)]
    with pytest.raises(ValueError, match="periodic cell"):
        relax_band("lj", images, symbols=AR4, options=["-vc"])


def distance(x: np.ndarray, a: int, b: int) -> float:
    r = np.reshape(x, (-1, 3))
    return float(np.linalg.norm(r[a - 1] - r[b - 1]))


def test_relax_band_holds_hard_constraints_in_every_image() -> None:
    images = [(x * ANG).ravel() for x in (AR4_ANGSTROM, AR4_PLANAR)]
    target = 3.85
    band = relax_band(
        "lj",
        [x.tolist() for x in images],
        symbols=AR4,
        fix_atoms=[1],
        constraints=["manual", "bond", "2,3", str(target)],
        max_iterations=6,
    )
    chain = [np.asarray(x) for x in band["images"]]
    assert len(chain) > 2
    for x in chain:
        assert abs(distance(x, 2, 3) - target * ANG) <= 1e-10
        # Frozen atoms are where the input has them, bit for bit.
        assert np.array_equal(np.reshape(x, (-1, 3))[0], np.reshape(images[0], (-1, 3))[0])
    # The same constraints given as nebmain options.
    as_options = relax_band(
        "lj",
        [x.tolist() for x in images],
        symbols=AR4,
        options=["-fix", "1", "-pc", "manual", "bond", "2,3", str(target), "-ns", "6"],
    )
    assert as_options["energies"] == band["energies"]


# --- irc_path with constraints and a cell ---
def lj_function(cell: list[float] | None = None):  # type: ignore[no-untyped-def]
    engine = _core.NativeEvaluator("emt" if cell else "lj")
    symbols = CU4 if cell else AR4

    def energy_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        e, g, _ = engine.evaluate(symbols, np.ravel(x).tolist(), 0, None, False, cell)
        return e, np.asarray(g)

    def hessian(x: np.ndarray) -> np.ndarray:
        _, g, h = engine.evaluate(symbols, np.ravel(x).tolist(), 0, None, True, cell)
        return np.reshape(h, (len(g), len(g)))

    return energy_and_gradient, hessian


def test_irc_path_holds_hard_constraints_at_every_point() -> None:
    x0 = (AR4_ANGSTROM * ANG).ravel()
    f, h = lj_function()
    common = {"symbols": AR4, "hessian": h, "step_size": 0.05, "max_steps": 15}
    held = ["bond", "1,3", "bond", "2,4"]
    bonds = irc_path(f, x0, constraints=held, **common)  # type: ignore[arg-type]
    points = [p for side in bonds.sides for p in side.points]
    assert len(points) > 3
    for p in points:
        assert abs(distance(p, 1, 3) - distance(x0, 1, 3)) <= 1e-10
        assert abs(distance(p, 2, 4) - distance(x0, 2, 4)) <= 1e-10
    frozen = irc_path(f, x0, fix_atoms=[1, 2], **common)  # type: ignore[arg-type]
    points = [p for side in frozen.sides for p in side.points]
    assert len(points) > 3
    for p in points:
        # Frozen atoms are where the start has them, bit for bit.
        assert np.array_equal(np.reshape(p, (-1, 3))[:2], np.reshape(x0, (-1, 3))[:2])
    with pytest.raises(ValueError, match="need atoms"):
        irc_path(f, [0.0, 0.0], fix_atoms=[1])


def test_irc_path_in_a_cell() -> None:
    cell = (CELL * ANG).ravel().tolist()
    f, h = lj_function(cell)
    x0 = (CU4_START * ANG).ravel()
    path = irc_path(f, x0, symbols=CU4, hessian=h, cell_bohr=cell, step_size=0.05, max_steps=5)
    points = [p for side in path.sides for p in side.points]
    assert points and all(np.all(np.isfinite(p)) for p in points)
    # In a cell the start is not moved to the centre of mass; without one it is.
    assert _core.Irc(x0.tolist(), symbols=CU4, cell_bohr=cell).start == x0.tolist()
    assert _core.Irc(x0.tolist(), symbols=CU4).start != x0.tolist()

# --- the ONIOM calculator ---
AR8 = ["Ar"] * 8
AR8_ANGSTROM = np.array(
    [
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [1.9, 3.3, 0.0],
        [1.9, 1.1, 3.1],
        [5.7, 3.3, 0.0],
        [-1.9, 3.3, 0.2],
        [3.8, 2.2, 3.1],
        [0.1, 2.2, 3.3],
    ],
    dtype=np.float64,
)


@pytest.mark.parametrize(
    "settings",
    [
        {"low.cutoff": 6.0},
        {
            "low.cutoff": 6.0,
            "micro_hessian_solve": "dense",
            "micro_correct": True,
            "micro_tolerance_factor": 0.5,
            "micro_check_tolerance": 1e-5,
            "micro_replicas": 2,
            "micro_replica_temperature": 300,
        },
    ],
)
def test_optimizer_on_the_oniom_calculator_is_optmain_oniom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: dict[str, object]
) -> None:
    monkeypatch.chdir(tmp_path)
    write_xyz("ar8.xyz", AR8, AR8_ANGSTROM)
    oniom = OniomCalculator("1,2", "lj", high_calculator="lj", links="none", settings=settings)
    assert oniom.microiterations
    assert oniom.options[:6] == ["-os", "lj", "-oniom", "1,2", "none", "lj"]
    method = ["-opt", "rsirfo_block_fsb", "-ns", "40"]
    jobs = _core.run_optmain(["ar8.xyz", *oniom.options, *method], False)
    x = (AR8_ANGSTROM * ANG).ravel().tolist()
    opt = Optimizer(AR8, x, methods=["rsirfo_block_fsb"], max_iterations=40, calculator=oniom)
    assert opt.has_own_calculator
    result = run(opt, oniom)
    assert result.iterations == jobs[0]["iterations"]
    assert abs(result.reports[-1]["energy"] - jobs[0]["final_energy"]) < 1e-10
    # The optimizer moves the core (the two atoms of the high layer and nothing else here).
    assert result.positions_bohr.shape == (2, 3)
    assert len(opt.whole_positions_bohr) == 3 * len(AR8)


def test_the_oniom_calculator_is_refused_where_the_commands_refuse() -> None:
    with pytest.raises(ValueError, match="micro"):
        OniomCalculator("1,2", "lj", high_calculator="lj", settings={"micro_replicas": 0})
    with pytest.raises(ValueError, match="middle_atoms and middle_calculator"):
        OniomCalculator("1,2", "lj", high_calculator="lj", middle_atoms=[1, 2, 3])
    oniom = OniomCalculator("1,2", "lj", high_calculator="lj", links="none")
    x = (AR8_ANGSTROM * ANG).ravel().tolist()
    with pytest.raises(ValueError, match="Optimizer"):
        oniom.evaluate(AR8, x)
    # Without microiterations every atom moves, and the whole surface is a calculator.
    whole = OniomCalculator(
        "1,2", "lj", high_calculator="lj", links="none", settings={"micro": False}
    )
    energy, gradient, _ = whole.evaluate(AR8, x)
    assert np.isfinite(energy) and len(gradient) == len(x)