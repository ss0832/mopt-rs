"""An engine for the tests of the worker processes: a harmonic well whose calls take a
moment and say in a file when and where they ran.

``-os engine:slow_engine:Slow -calc_opt delay=0.2 log=FILE``: every call appends
``<process id> <start> <end>`` (seconds) to ``FILE``. ``die_at=N`` ends the process in its
N-th call, as a program that crashes.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import os
import time

import numpy as np

from mopt_rs.calculators import ASE_UNITS, Capabilities, Engine, EngineResult, Structure


class Slow(Engine):
    """``E = 1/2 sum |x - x_first_atom_pair_rest|^2`` in eV and Angstrom: a harmonic bond of
    length 1.2 between neighbours in the list of atoms, slow to compute."""

    name = "slow"
    units = ASE_UNITS
    capabilities = Capabilities(electrons=False)

    def __init__(self, delay: float = 0.0, log: str | None = None, die_at: int = 0) -> None:
        self.delay = float(delay)
        self.log = log
        self.die_at = int(die_at)
        self.calls = 0

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        self.calls += 1
        if self.die_at and self.calls >= self.die_at:
            os._exit(3)
        start = time.time()
        time.sleep(self.delay)
        x = structure.positions_angstrom
        energy = 0.0
        forces = np.zeros_like(x)
        for i in range(len(x) - 1):
            d = x[i + 1] - x[i]
            r = float(np.linalg.norm(d))
            energy += 0.5 * (r - 1.2) ** 2
            pull = (r - 1.2) * d / r
            forces[i] += pull
            forces[i + 1] -= pull
        if self.log:
            with open(self.log, "a", encoding="utf-8") as file:
                file.write(f"{os.getpid()} {start:.6f} {time.time():.6f}\n")
        return EngineResult(energy, forces=forces)
