"""Geometry optimization from Python with any calculator.

The optimizer itself runs in Rust and never calls a calculator: :meth:`Optimizer.ask` returns
the evaluations needed at the current structure, the Python side computes them, and
:meth:`Optimizer.tell` advances the optimizer. :func:`run` performs this loop with calculators
that implement :class:`Calculator`. All quantities are in atomic units (Bohr, Hartree);
calculators always receive Cartesian structures, whatever coordinate system the optimizer uses.

MultiOptPy: `Wrapper/optimize_wrapper.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from mopt_rs import _core
from mopt_rs._core import Optimizer

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "Calculator",
    "Checkpoint",
    "Evaluation",
    "NativeCalculator",
    "Optimizer",
    "Result",
    "Task",
    "run",
]


@dataclass(frozen=True)
class Task:
    """One evaluation requested by the optimizer."""

    label: str
    calculator: str
    symbols: list[str]
    positions_bohr: FloatArray
    """Cartesian positions, shape (N, 3)."""
    charge: int
    multiplicity: int
    need_hessian: bool
    excited_state: int = 0
    cell_bohr: FloatArray | None = None
    """Lattice vectors as rows, shape (3, 3), for periodic systems."""
    pbc: tuple[bool, bool, bool] = (False, False, False)
    """Periodic lattice directions (a, b, c)."""

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Task:
        cell = d.get("cell_bohr")
        return Task(
            label=str(d["label"]),
            calculator=str(d["calculator"]),
            symbols=list(d["symbols"]),
            positions_bohr=np.asarray(d["positions_bohr"], dtype=np.float64).reshape(-1, 3),
            charge=int(d["charge"]),
            multiplicity=int(d["multiplicity"]),
            need_hessian=bool(d["need_hessian"]),
            excited_state=int(d.get("excited_state", 0)),
            cell_bohr=None if cell is None else np.asarray(cell, dtype=np.float64).reshape(3, 3),
            pbc=_pbc(d.get("pbc"), cell is not None),
        )


def _pbc(value: Any, has_cell: bool) -> tuple[bool, bool, bool]:
    if value is None:
        return (has_cell, has_cell, has_cell)
    a, b, c = (bool(v) for v in value)
    return (a, b, c)


@dataclass
class Evaluation:
    """Energy (Hartree), gradient (Hartree/Bohr, shape (N, 3) or (3N,)) and optional Hessian
    (Hartree/Bohr^2, shape (3N, 3N))."""

    energy: float
    gradient: FloatArray
    hessian: FloatArray | None = None
    spin_square: float | None = None
    """Expectation value ``<S^2>`` of the wavefunction, where the calculator tells it."""
    output: str | None = None
    """What the program printed during the evaluation (its own output), if it was kept."""
    stress: FloatArray | None = None
    """Stress of a periodic cell (Hartree/Bohr^3, shape (3, 3), ``(1/V) dE/d(strain)`` as in
    ASE), when it was asked for and the calculator has one."""
    notes: tuple[str, ...] = ()
    """What the calculator has to say about this evaluation (a test of the stability of its
    SCF solution, for example), for the log of the run."""

    def as_tuple(self) -> tuple[float, list[float], list[float] | None]:
        g = np.asarray(self.gradient, dtype=np.float64).ravel()
        h = None if self.hessian is None else np.asarray(self.hessian, dtype=np.float64).ravel()
        return float(self.energy), g.tolist(), None if h is None else h.tolist()


class Calculator(Protocol):
    """Anything that evaluates a :class:`Task`."""

    def evaluate(self, task: Task) -> Evaluation: ...


def native_settings(
    settings: Mapping[str, object] | Iterable[tuple[str, object]] | None,
) -> list[tuple[str, str]]:
    """Settings of a built-in calculator as the pairs of strings the Rust side takes."""
    if settings is None:
        return []
    pairs = settings.items() if isinstance(settings, Mapping) else settings
    return [(str(key), str(value)) for key, value in pairs]


class NativeCalculator:
    """A built-in mopt-rs calculator (``lj``, ``emt``, ``eam``, ``meam``, ``tersoff``,
    ``muller_brown``, ``gfn0xtb``, ``gfn1xtb``, ``gfnff``), kept between evaluations.

    ``settings`` are those of ``-calc_opt`` (``{"cutoff": 8.0}`` for ``lj``; a list of pairs
    where a key is given more than once, such as ``pair`` of ``meam``)."""

    def __init__(
        self,
        name: str,
        settings: Mapping[str, object] | Iterable[tuple[str, object]] | None = None,
        *,
        follow_states: bool = False,
    ) -> None:
        self.name = name
        # One Rust calculator for all evaluations (GFN-FF keeps the topology of the first).
        self._evaluator = _core.NativeEvaluator(
            name, native_settings(settings), follow_states=follow_states
        )

    def evaluate(self, task: Task) -> Evaluation:
        cell = None if task.cell_bohr is None else task.cell_bohr.ravel().tolist()
        e, g, h = self._evaluator.evaluate(
            task.symbols,
            task.positions_bohr.ravel().tolist(),
            task.charge,
            task.multiplicity,
            task.need_hessian,
            cell,
            pbc=task.pbc,
            excited_state=task.excited_state,
        )
        for note in self._evaluator.take_warnings():
            warnings.warn(note, RuntimeWarning, stacklevel=2)
        n = len(g)
        hess = None if h is None else np.asarray(h, dtype=np.float64).reshape(n, n)
        return Evaluation(e, np.asarray(g, dtype=np.float64), hess)


@dataclass
class Result:
    """Outcome of :func:`run`."""

    termination: str | None
    iterations: int
    positions_bohr: FloatArray
    energies: list[float] = field(default_factory=list)
    reports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.termination == "converged"


class Checkpoint:
    """A journal of evaluations for restarting :func:`run` (one JSON line per evaluation).

    Every evaluation is appended and flushed to disk before the run continues. Running the
    same optimizer setup again with the same checkpoint answers its evaluations from the file
    while the requested structures match (within ``tolerance_bohr``), then continues with the
    calculator, so an interrupted run (Ctrl-C, a killed job) resumes where it stopped. A
    truncated last line (killed while writing) is ignored.
    """

    def __init__(self, path: str | os.PathLike[str], tolerance_bohr: float = 1e-9) -> None:
        self.path = Path(path)
        self.tolerance_bohr = tolerance_bohr
        self.pending: list[dict[str, Any]] = []
        self.replayed = 0
        valid = 0
        if self.path.exists():
            with self.path.open("r+b") as f:
                for line in f:
                    if not line.endswith(b"\n"):
                        break
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        break
                    self.pending.append(rec)
                    valid += len(line)
                f.truncate(valid)
        self._file = self.path.open("a", encoding="utf-8")

    def _matches(self, rec: Mapping[str, Any], task: Task) -> bool:
        x = task.positions_bohr.ravel()
        y = np.asarray(rec["positions_bohr"], dtype=np.float64)
        return bool(
            rec.get("symbols") == task.symbols
            and rec.get("pbc") == list(task.pbc)
            and rec.get("cell_bohr")
            == (None if task.cell_bohr is None else task.cell_bohr.tolist())
            and rec["calculator"] == task.calculator
            and rec["charge"] == task.charge
            and rec["multiplicity"] == task.multiplicity
            and rec["excited_state"] == task.excited_state
            and (rec["hessian"] is not None or not task.need_hessian)
            and y.shape == x.shape
            and np.max(np.abs(y - x), initial=0.0) <= self.tolerance_bohr
        )

    def evaluate(
        self, calc: Calculator, task: Task
    ) -> tuple[float, list[float], list[float] | None]:
        """The result for ``task``: from the file when recorded, else from ``calc`` (recorded)."""
        for k, rec in enumerate(self.pending):
            if self._matches(rec, task):
                self.pending.pop(k)
                self.replayed += 1
                h = rec["hessian"] if task.need_hessian else None
                return float(rec["energy"]), list(rec["gradient"]), h
        e, g, h = calc.evaluate(task).as_tuple()
        rec = {
            "calculator": task.calculator,
            "symbols": task.symbols,
            "pbc": list(task.pbc),
            "cell_bohr": None if task.cell_bohr is None else task.cell_bohr.tolist(),
            "charge": task.charge,
            "multiplicity": task.multiplicity,
            "excited_state": task.excited_state,
            "positions_bohr": task.positions_bohr.ravel().tolist(),
            "energy": e,
            "gradient": g,
            "hessian": h,
        }
        self._file.write(json.dumps(rec) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        return e, g, h

    def close(self) -> None:
        self._file.close()


def run(
    optimizer: Optimizer,
    calculators: Calculator | Mapping[str, Calculator],
    callback: Callable[[dict[str, Any]], bool | None] | None = None,
    checkpoint: str | os.PathLike[str] | Checkpoint | None = None,
) -> Result:
    """Drive ``optimizer`` to termination.

    ``calculators`` is one calculator, used for every evaluation, or a mapping from calculator
    labels to calculators (combined surfaces). An optimizer made with a calculator of its own
    (``Optimizer(..., calculator=OniomCalculator(...))``) is evaluated with it, on the
    surface ``optmain -oniom`` relaxes (the core with the environment relaxed); pass that
    calculator here as well. ``callback`` receives every iteration report and
    stops the run by returning ``True``. ``checkpoint`` (a file or :class:`Checkpoint`) records
    the evaluations, and running again with it continues an interrupted run.
    """
    by_label: Mapping[str, Calculator] | None = (
        calculators if isinstance(calculators, Mapping) else None
    )
    single: Calculator | None = None if by_label is not None else calculators  # type: ignore[assignment]
    journal = (
        checkpoint
        if isinstance(checkpoint, Checkpoint) or checkpoint is None
        else Checkpoint(checkpoint)
    )
    try:
        return _drive(optimizer, single, by_label, callback, journal)
    finally:
        if journal is not None and not isinstance(checkpoint, Checkpoint):
            journal.close()


def _drive(
    optimizer: Optimizer,
    single: Calculator | None,
    by_label: Mapping[str, Calculator] | None,
    callback: Callable[[dict[str, Any]], bool | None] | None,
    journal: Checkpoint | None,
) -> Result:
    energies: list[float] = []
    reports: list[dict[str, Any]] = []
    own = bool(getattr(optimizer, "has_own_calculator", False))
    if getattr(optimizer, "wants_o1numhess", False):
        # model_hessian="o1numhess": calculators without Hessians of their own use O1NumHess.
        for calc in [single] if by_label is None else list(by_label.values()):
            if getattr(calc, "hessian_method", None) == "central":
                calc.hessian_method = "o1numhess"  # type: ignore[union-attr]
    while True:
        requests: Sequence[Mapping[str, Any]] | None = optimizer.ask()
        if requests is None:
            break
        results = []
        spins: list[float | None] = []
        for d in requests:
            task = Task.from_dict(d)
            if own:
                e, g, h = optimizer.evaluate_own(
                    task.positions_bohr.ravel().tolist(), task.need_hessian, task.excited_state
                )
                results.append((e, g, h))
                spins.append(None)
                continue
            calc = single if by_label is None else by_label[task.calculator]
            if calc is None:
                raise KeyError(f"no calculator for '{task.calculator}'")
            if journal is None:
                evaluation = calc.evaluate(task)
                results.append(evaluation.as_tuple())
                spins.append(evaluation.spin_square)
            else:
                results.append(journal.evaluate(calc, task))
                spins.append(None)
        report = optimizer.tell(results, spins)
        energies.append(float(report["energy"]))
        reports.append(report)
        if callback is not None and callback(report):
            break
    return Result(
        termination=optimizer.termination,
        iterations=optimizer.iterations,
        positions_bohr=np.asarray(optimizer.positions_bohr, dtype=np.float64).reshape(-1, 3),
        energies=energies,
        reports=reports,
    )
