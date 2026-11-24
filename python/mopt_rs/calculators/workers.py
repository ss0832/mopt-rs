"""Copies of a calculator in worker processes: several structures evaluated at once.

A command evaluates the structures of one iteration together (the images of a chain, the two
structures of a pair, the displacements of a finite-difference Hessian). The calculators
written in Rust do that in parallel threads. An engine of the Python side is one object
with one program behind it and takes one call at a time
(:class:`~mopt_rs.calculators.base.EngineCalculator`), so with it the structures are
calculated one after the other.

:class:`WorkerCalculator` runs up to ``workers`` copies of such a calculator, each in a
process of its own, and gives every structure to a copy that is free::

    nebmain path.xyz -os pyscf -parallel 4 -core 2
    calc = mopt_rs.calculators.WorkerCalculator("pyscf", options, workers=4)

What to know:

- every copy is created from the name of the calculator and its options in a new process,
  so the calculator must be one that a new process can create: a built-in name, a profile,
  ``ase:module:Class`` of a module on the search path, a registered entry point. An engine
  object made in a script cannot be copied;
- the threads and the memory of the options are those of one copy: ``workers`` times the
  threads of ``-core`` should not be more than the processors there are;
- a structure goes to the copy that last had the same structure, else to the free copy
  whose last structure is nearest, else to a new copy. In a chain every image so keeps
  its copy from iteration to iteration, and with it the orbitals its calculation starts
  from;
- programs that work in a directory with fixed file names
  (:mod:`mopt_rs.calculators.programs`) work in ``mopt_worker_<k>/`` of their directory,
  one per copy;
- ``-stability N`` counts the evaluations of each copy;
- a run that follows excited states (``-es_follow``) evaluates its structures in order,
  with or without workers.

The copies talk to this process through their standard input and output (pickled messages
with a length in front); what a program prints goes to the standard error stream.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import contextlib
import os
import pickle
import struct
import subprocess
import sys
import threading
import weakref
from dataclasses import dataclass
from typing import IO, Any

import numpy as np

from mopt_rs.calculators.base import WORKER_ENV, CalculatorError, Capabilities, Structure
from mopt_rs.calculators.registry import CalculatorOptions
from mopt_rs.optimize import Evaluation

_START = "from mopt_rs.calculators.workers import serve; raise SystemExit(serve())"


def _send(stream: IO[bytes], message: Any) -> None:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def _receive(stream: IO[bytes]) -> Any:
    """The next message; ``EOFError`` when the other side has closed."""
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError
    (size,) = struct.unpack("<Q", head)
    data = stream.read(size)
    if len(data) < size:
        raise EOFError
    return pickle.loads(data)


@dataclass(frozen=True)
class EngineFacts:
    """What a calculator in a worker says about its engine."""

    name: str
    description: str
    capabilities: Capabilities
    citations: tuple[str, ...]
    checks_stability: bool


def serve() -> int:
    """The loop of a worker process: create the calculator named in the first message,
    then answer one request after the other until the input ends."""
    from mopt_rs.calculators.registry import create

    out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    # Programs and libraries print to the standard output; the answers must be alone on it.
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    inp = sys.stdin.buffer
    try:
        spec, options, path = _receive(inp)
    except EOFError:
        return 0
    for entry in path:
        if entry not in sys.path:
            sys.path.append(entry)
    try:
        calc = create(spec, options)
        facts = EngineFacts(
            calc.name,
            calc.describe(),
            calc.engine.capabilities,
            tuple(calc.engine.citations),
            calc.checks_stability,
        )
    except BaseException as error:  # noqa: BLE001 (reported to the parent, whatever it is)
        _send(out, ("failed", f"{type(error).__name__}: {error}", 0))
        return 1
    _send(out, ("ok", facts, 0))
    while True:
        try:
            kind, structure, arguments = _receive(inp)
        except EOFError:
            return 0
        try:
            if kind == "evaluate":
                value: Any = calc.evaluate_structure(structure, *arguments)
            elif kind == "embedded":
                value = calc.evaluate_embedded(structure, *arguments)
            elif kind == "orbitals":
                value = calc.orbitals(structure)
            elif kind == "transitions":
                value = calc.transitions(structure, *arguments)
            elif kind == "state_characters":
                value = calc.state_characters(structure, *arguments)
            else:
                raise ValueError(f"unknown request '{kind}'")
            reply = ("ok", value, calc.evaluations)
        except CalculatorError as error:
            reply = ("failed", str(error), calc.evaluations)
        except Exception as error:  # noqa: BLE001 (reported to the parent)
            reply = ("failed", f"{type(error).__name__}: {error}", calc.evaluations)
        _send(out, reply)


class _Worker:
    """One copy of the calculator in a process of its own."""

    def __init__(self, index: int, spec: str, options: CalculatorOptions) -> None:
        self.index = index
        self.evaluations = 0
        self.last: Structure | None = None
        env = dict(os.environ)
        env[WORKER_ENV] = str(index)
        self.process = subprocess.Popen(
            [sys.executable, "-c", _START],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )
        try:
            self.facts: EngineFacts = self.ask((spec, options, list(sys.path)))
        except CalculatorError as error:
            self.close()
            raise CalculatorError(
                f"a worker process could not create the calculator '{spec}': {error} "
                "(a calculator with workers must be one that a new process can create "
                "from its name: a built-in name, a profile, ase:module:Class, an entry point)"
            ) from None

    def ask(self, message: Any) -> Any:
        """The answer of the worker to ``message``."""
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            _send(self.process.stdin, message)
            status, value, evaluations = _receive(self.process.stdout)
        except (EOFError, OSError, pickle.PickleError) as error:
            # EOF can arrive before the OS publishes the child's exit status.
            # Reap it here so a broken connection cannot return to the pool.
            self.close()
            detail = f": {error}" if str(error) else ""
            raise CalculatorError(
                f"worker {self.index} ended while it was calculating "
                f"(exit code {self.process.returncode}){detail}"
            ) from None
        self.evaluations = int(evaluations)
        if status != "ok":
            raise CalculatorError(str(value))
        return value

    def distance(self, structure: Structure) -> float:
        """How far ``structure`` is from the last one of this worker (0: the same structure;
        infinite: none yet, or other atoms)."""
        last = self.last
        if last is None or last.symbols != structure.symbols:
            return float("inf")
        if last.key() == structure.key():
            return 0.0
        moved = float(np.abs(last.positions_bohr - structure.positions_bohr).max())
        # Another state or cell at the same place is not the same structure.
        return max(moved, 1e-12)

    def alive(self) -> bool:
        return self.process.poll() is None

    def close(self) -> None:
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def _close_all(workers: list[_Worker]) -> None:
    for worker in workers:
        worker.close()
    workers.clear()


class WorkerCalculator:
    """Up to ``workers`` copies of the calculator ``spec`` with ``options``, each in a
    process of its own (see the module). It is used like an
    :class:`~mopt_rs.calculators.base.EngineCalculator` and may be called from several
    threads at once; a call waits while every copy is busy."""

    def __init__(self, spec: str, options: CalculatorOptions | None = None, workers: int = 2):
        if not isinstance(spec, str):
            raise CalculatorError(
                "a calculator with workers is given by its name (a new process creates it)"
            )
        if workers < 1:
            raise ValueError(f"workers must be at least 1 (found {workers})")
        self.spec = spec
        self.options = options or CalculatorOptions()
        self.workers = workers
        self._all: list[_Worker] = []
        self._free: list[_Worker] = []
        self._starting = 0
        self._started = 1
        self._finished = 0
        self._state = threading.Condition()
        self._closer = weakref.finalize(self, _close_all, self._all)
        first = _Worker(0, self.spec, self.options)
        self._all.append(first)
        self._free.append(first)
        self.facts = first.facts

    @property
    def name(self) -> str:
        return self.facts.name

    @property
    def engine(self) -> EngineFacts:
        """What the engine of the copies says about itself (name, capabilities, citations)."""
        return self.facts

    @property
    def checks_stability(self) -> bool:
        return self.facts.checks_stability

    @property
    def evaluations(self) -> int:
        """Engine calls of all copies so far."""
        with self._state:
            return self._finished + sum(w.evaluations for w in self._all)

    def describe(self) -> str:
        return f"{self.facts.description}; up to {self.workers} copies in worker processes"

    def close(self) -> None:
        """End the worker processes (also done when the calculator is dropped)."""
        with self._state:
            self._finished += sum(w.evaluations for w in self._all)
            _close_all(self._all)
            self._free.clear()

    def _take(self, structure: Structure) -> _Worker:
        """A copy for ``structure``: the free one nearest to it, a new one while fewer than
        ``workers`` run, else the next that gets free."""
        with self._state:
            while True:
                if self._free:
                    worker = min(self._free, key=lambda w: (w.distance(structure), w.index))
                    self._free.remove(worker)
                    return worker
                if len(self._all) + self._starting < self.workers:
                    self._starting += 1
                    index = self._started
                    self._started += 1
                    break
                self._state.wait()
        try:
            worker = _Worker(index, self.spec, self.options)
        except BaseException:
            with self._state:
                self._starting -= 1
                self._state.notify()
            raise
        with self._state:
            self._starting -= 1
            self._all.append(worker)
        return worker

    def _give_back(self, worker: _Worker) -> None:
        with self._state:
            if worker.alive():
                self._free.append(worker)
            else:
                # Its place is free for a new copy.
                self._finished += worker.evaluations
                if worker in self._all:
                    self._all.remove(worker)
                worker.close()
            self._state.notify()

    def _ask(self, kind: str, structure: Structure, arguments: tuple[Any, ...]) -> Any:
        worker = self._take(structure)
        try:
            value = worker.ask((kind, structure, arguments))
            worker.last = structure
            return value
        finally:
            self._give_back(worker)

    def evaluate_structure(
        self,
        structure: Structure,
        need_hessian: bool = False,
        need_stress: bool = False,
        need_gradient: bool = True,
    ) -> Evaluation:
        """As :meth:`EngineCalculator.evaluate_structure`, in a copy that is free."""
        found = self._ask("evaluate", structure, (need_hessian, need_stress, need_gradient))
        assert isinstance(found, Evaluation)
        return found

    def evaluate_embedded(
        self,
        structure: Structure,
        positions_bohr: Any,
        charges: Any,
        elements: list[str | None],
        hessian: bool = False,
    ) -> tuple[Evaluation, Any]:
        """Evaluate the embedded structure in an available worker."""
        result: tuple[Evaluation, Any] = self._ask(
            "embedded", structure, (positions_bohr, charges, elements, hessian)
        )
        return result

    def orbitals(self, structure: Structure) -> dict[str, Any] | None:
        """As :meth:`EngineCalculator.orbitals`."""
        found: dict[str, Any] | None = self._ask("orbitals", structure, ())
        return found

    def transitions(self, structure: Structure, count: int) -> list[dict[str, Any]] | None:
        """As :meth:`EngineCalculator.transitions`."""
        found: list[dict[str, Any]] | None = self._ask("transitions", structure, (count,))
        return found

    def state_characters(self, structure: Structure, count: int) -> dict[str, Any] | None:
        """As :meth:`EngineCalculator.state_characters`."""
        found: dict[str, Any] | None = self._ask("state_characters", structure, (count,))
        return found
