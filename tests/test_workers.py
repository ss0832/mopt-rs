"""Copies of a calculator in worker processes (``-calc_opt workers=N``): several structures
of the Python engines calculated at once.

MultiOptPy: no counterpart (its images are calculated one after the other).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest

import mopt_rs
from mopt_rs.calculators import (
    CalculatorError,
    CalculatorOptions,
    Structure,
    WorkerCalculator,
    create,
)

SLOW = "engine:slow_engine:Slow"


def bent(angle: float) -> Structure:
    """Three atoms 1.2 and 1.3 Angstrom apart with ``angle`` (radians) between the bonds."""
    a = 1.2 / 0.529177210903
    b = 1.3 / 0.529177210903
    positions = [0.0, 0.0, 0.0, a, 0.0, 0.0, a + b * np.cos(angle), b * np.sin(angle), 0.0]
    return Structure.build(["Ar", "Ar", "Ar"], positions)


def calls(log: Path) -> list[tuple[int, float, float]]:
    """The calls of the engine: process, start, end."""
    rows = [line.split() for line in log.read_text().splitlines()]
    return [(int(p), float(s), float(e)) for p, s, e in rows]


def test_an_engine_by_where_it_is_imported_from() -> None:
    calc = create(SLOW)
    assert calc.name == "slow"
    found = calc.evaluate_structure(bent(1.0))
    assert abs(found.energy - 0.5 * 0.1**2 / 27.211386245988) < 1e-12
    with pytest.raises(TypeError, match="not an Engine"):
        create("engine:slow_engine:time")
    with pytest.raises(ImportError):
        create("engine:slow_engine:NoSuchClass")


def test_copies_in_worker_processes_calculate_at_once(tmp_path: Path) -> None:
    log = tmp_path / "calls.txt"
    options = CalculatorOptions(extra={"delay": 0.5, "log": str(log)})
    pool = WorkerCalculator(SLOW, options, workers=3)
    assert pool.name == "slow" and "3 copies" in pool.describe()
    assert not pool.engine.capabilities.electrons

    def at_once(group: list[Structure]) -> list[object]:
        found: list[object] = [None] * len(group)

        def work(k: int) -> None:
            found[k] = pool.evaluate_structure(group[k])

        threads = [threading.Thread(target=work, args=(k,)) for k in range(len(group))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return found

    # Three copies start with three structures asked for at once (how long a copy takes to
    # start depends on the load of the machine, so these calls are not timed).
    at_once([bent(2.0 + 0.3 * k) for k in range(3)])
    assert len({p for p, _, _ in calls(log)}) == 3
    structures = [bent(0.8 + 0.3 * k) for k in range(3)]
    found = at_once(structures)
    # The same numbers as one calculator in this process.
    here = create(SLOW)
    for structure, got in zip(structures, found, strict=True):
        want = here.evaluate_structure(structure)
        assert got is not None
        assert got.energy == want.energy  # type: ignore[attr-defined]
        assert np.array_equal(got.gradient, want.gradient)  # type: ignore[attr-defined]
    everything = calls(log)
    rows = everything[3:]
    processes = {p for p, _, _ in everything}
    assert len(rows) == 3 and len(processes) == 3 and os.getpid() not in processes
    # The three copies were inside the engine at the same time.
    assert max(s for _, s, _ in rows) < min(e for _, _, e in rows)
    assert pool.evaluations == 6

    # A structure goes back to the copy that had it (which remembers it: no new call) ...
    for structure in structures:
        pool.evaluate_structure(structure)
    assert len(calls(log)) == 6
    # ... and a structure near it to the same copy.
    for structure in structures:
        had = _owner(pool, structure)
        moved = Structure.build(
            list(structure.symbols), (structure.positions_bohr + 1e-3).ravel().tolist()
        )
        count = len(calls(log))
        pool.evaluate_structure(moved)
        rows = calls(log)
        assert len(rows) == count + 1 and rows[-1][0] == had
    pool.close()


def _owner(pool: WorkerCalculator, structure: Structure) -> int:
    """The process of the copy whose last structure is ``structure``."""
    for worker in pool._all:
        if worker.last is not None and worker.last.key() == structure.key():
            return int(worker.process.pid)
    raise AssertionError("no copy has this structure")


def test_what_a_new_process_cannot_create_is_said() -> None:
    with pytest.raises(CalculatorError, match="new process can create"):
        WorkerCalculator("no_such_calculator_of_the_tests", workers=2)
    with pytest.raises(CalculatorError, match="by its name"):
        WorkerCalculator(create(SLOW).engine, workers=2)  # type: ignore[arg-type]


def test_a_copy_that_ends_is_replaced(tmp_path: Path) -> None:
    log = tmp_path / "calls.txt"
    pool = WorkerCalculator(
        SLOW, CalculatorOptions(extra={"log": str(log), "die_at": 2}), workers=2
    )
    pool.evaluate_structure(bent(1.0))
    with pytest.raises(CalculatorError, match=r"ended while it was calculating \(exit code 3\)"):
        pool.evaluate_structure(bent(1.1))
    # The next structure is calculated by a new copy.
    found = pool.evaluate_structure(bent(1.2))
    assert np.isfinite(found.energy)
    assert len({p for p, _, _ in calls(log)}) == 2
    pool.close()


def chain_file(directory: Path) -> Path:
    frames = "".join(
        f"3\n\nAr 0.0 0.0 0.0\nAr 1.2 0.0 0.0\nAr {1.2 + 1.3 * np.cos(a):.6f} "
        f"{1.3 * np.sin(a):.6f} 0.0\n"
        for a in np.linspace(0.6, 2.4, 6)
    )
    directory.mkdir()
    path = directory / "chain.xyz"
    path.write_text(frames)
    return path


def last_energies(directory: Path) -> list[float]:
    runs = [p for p in directory.iterdir() if p.is_dir() and "_NEB_" in p.name]
    assert len(runs) == 1
    line = (runs[0] / "energy_plot.csv").read_text().splitlines()[-1]
    return [float(v) for v in line.split(",")]


def test_the_images_of_a_chain_in_worker_processes(tmp_path: Path) -> None:
    log = tmp_path / "calls.txt"
    # -core 1: the threads of one copy; the command still gives every copy a structure.
    common = ["-ns", "3", "-nd", "0", "-adpred", "0", "-fe", "3", "-core", "1", "-os", SLOW]
    with_workers = chain_file(tmp_path / "workers")
    code = mopt_rs._core.run_cli(
        "nebmain",
        [str(with_workers), *common, "-calc_opt", "workers=3", "delay=0.1", f"log={log}"],
    )
    assert code == 0
    rows = calls(log)
    processes = {p for p, _, _ in rows}
    assert len(processes) == 3 and os.getpid() not in processes
    # At some moment three images were being calculated.
    assert max(sum(1 for _, s, e in rows if s <= start < e) for _, start, _ in rows) == 3

    alone = chain_file(tmp_path / "alone")
    assert mopt_rs._core.run_cli("nebmain", [str(alone), *common]) == 0
    assert last_energies(tmp_path / "workers") == pytest.approx(
        last_energies(tmp_path / "alone"), abs=1e-12
    )


def test_parallel_is_the_number_of_copies(tmp_path: Path) -> None:
    # -core is the threads of one calculation, -parallel the calculations at a time.
    log = tmp_path / "calls.txt"
    path = chain_file(tmp_path / "parallel")
    arguments = [str(path), "-ns", "2", "-nd", "0", "-adpred", "0", "-fe", "3", "-os", SLOW]
    arguments += ["-core", "1", "-parallel", "3", "-calc_opt", "delay=0.1", f"log={log}"]
    assert mopt_rs._core.run_cli("nebmain", arguments) == 0
    rows = calls(log)
    assert len({p for p, _, _ in rows}) == 3
    assert max(sum(1 for _, s, e in rows if s <= start < e) for _, start, _ in rows) == 3
    text = next((tmp_path / "parallel").rglob("mopt.log")).read_text()
    assert "up to 3 copies in worker processes" in text


def test_the_number_of_workers_is_checked(tmp_path: Path) -> None:
    path = chain_file(tmp_path / "wrong")
    arguments = [str(path), "-os", SLOW, "-ns", "1", "-calc_opt", "workers=0"]
    assert mopt_rs._core.run_cli("nebmain", arguments) != 0
