"""Tests of SCF solutions for instabilities, for every command (``-stability N``).

An SCF can converge to a saddle point of the energy in the space of the orbitals (a
solution that is not the ground state of the method): a stretched bond with restricted
orbitals, a symmetric solution where the symmetry breaks. Every ``N``-th evaluation of a
calculator (``-stability N``; ``0``, the default, never) asks its engine to test the solution
it found (:meth:`mopt_rs.calculators.base.Engine.check_stability`). An engine that finds the
solution unstable moves to a stabler one (the SCF goes on from the orbitals along the
unstable direction, until the solution is stable), and the evaluation is repeated there, so
that what the run gets is the energy and the gradient of the stable solution; the journal of
the run records that one. An instability towards another kind of wavefunction (restricted
to unrestricted: "external") is tested as well with ``-stability_external report`` (the
default: it is said, the kind is kept) or ``switch`` (the engine changes to the unrestricted
kind for the rest of the run); ``off`` tests within the kind only. What was found is said
in the log of the run with the evaluation it belongs to.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["EXTERNAL_MODES", "StabilityCheck", "StabilityReport"]

EXTERNAL_MODES = ("off", "report", "switch")
"""What is done about an instability towards another kind of wavefunction."""


@dataclass
class StabilityReport:
    """What an engine found when it tested its solution."""

    changed: bool
    """The engine moved to another (stabler) solution: the evaluation is to be repeated."""
    lines: list[str] = field(default_factory=list)
    """What to say in the log of the run (nothing when the solution was stable)."""


class StabilityCheck:
    """When the solutions of a calculator are tested: every ``every``-th evaluation."""

    def __init__(self, every: int = 0, external: str = "report") -> None:
        if every < 0:
            raise ValueError(f"-stability takes a number of evaluations (0: never), not {every}")
        mode = (external or "report").lower()
        if mode not in EXTERNAL_MODES:
            raise ValueError(
                f"-stability_external is one of {', '.join(EXTERNAL_MODES)}, not '{external}'"
            )
        self.every = int(every)
        self.external = mode
        self._count = 0
        self.unsupported_said = False
        self.tests = 0
        """Tests made so far."""

    def due(self) -> bool:
        """Count one evaluation; whether its solution is to be tested."""
        if self.every <= 0:
            return False
        self._count += 1
        if self._count >= self.every:
            self._count = 0
            self.tests += 1
            return True
        return False
