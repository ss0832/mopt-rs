"""Console entry points: the ten MultiOptPy commands, ``freqmain``, ``ircmain``, and
``mopt-py <command>``.

MultiOptPy: `Entrypoints/core.py` and the scripts of `pyproject.toml`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

from mopt_rs import _core


@dataclass(frozen=True)
class EvaluationCounts:
    """What the calculators of a run were asked for."""

    #: Evaluations of the energy alone.
    energies: int
    #: Evaluations with the gradient (those with the Hessian are among them).
    gradients: int
    #: Evaluations with the Hessian.
    hessians: int


def run_counted(command: str, args: Sequence[str]) -> tuple[int, EvaluationCounts]:
    """Run ``command`` (``optmain``, ``nebmain``, ``ircmain``, ...) with ``args`` as on the
    command line; returns its exit code and the evaluations its calculators made (the
    numbers of the line ``Evaluations (in total): ...`` at the end of the output).
    Evaluations that a restarted run reads from its journal are not made and not counted.
    """
    code, energies, gradients, hessians = _core.run_cli_counted(command, list(args))
    return code, EvaluationCounts(energies, gradients, hessians)


def _run(command: str) -> None:
    sys.exit(_core.run_cli(command, sys.argv[1:]))


def main() -> None:
    """``mopt-py <command> [options]``."""
    sys.exit(_core.main_from_argv(["mopt-py", *sys.argv[1:]]))


def optmain() -> None:
    """Geometry optimization."""
    _run("optmain")


def nebmain() -> None:
    """Nudged elastic band and other path methods."""
    _run("nebmain")


def ieipmain() -> None:
    """iEIP and other two-structure methods."""
    _run("ieipmain")


def mdmain() -> None:
    """Molecular dynamics."""
    _run("mdmain")


def confsearch() -> None:
    """Conformation search."""
    _run("confsearch")


def relaxedscan() -> None:
    """Relaxed scan."""
    _run("relaxedscan")


def orientsearch() -> None:
    """Orientation search."""
    _run("orientsearch")


def run_autots() -> None:
    """Automated transition-state search."""
    _run("run_autots")


def run_mapper() -> None:
    """Reaction network mapper."""
    _run("run_mapper")


def run_rcmc() -> None:
    """Rate constant matrix contraction with MultiOptPy's options and numbers."""
    _run("run_rcmc")


def run_kinetics() -> None:
    """Kinetics of a reaction network (rate constants, populations, superstates)."""
    _run("run_kinetics")


def run_automecp() -> None:
    """Automated search for minimum-energy crossing points of two states (MECP, MECI)."""
    _run("run_automecp")


def freqmain() -> None:
    """Vibrational analysis and thermochemistry of given structures."""
    _run("freqmain")


def etmain() -> None:
    """Rate constant of an electron transfer: EDEEL within a molecule, or the hop of a charge."""
    _run("etmain")


def ircmain() -> None:
    """Intrinsic reaction coordinate from a given structure."""
    _run("ircmain")
