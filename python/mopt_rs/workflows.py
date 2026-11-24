"""Python-driven chain relaxations, environment optimizations, and state following.

MultiOptPy: no counterpart (component API).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from mopt_rs._core import FollowedEvaluator, MapperNetwork, relax_band, relax_environment
from mopt_rs.optimize import Evaluation, Task

if TYPE_CHECKING:
    from mopt_rs.calculators.base import EngineCalculator
    from mopt_rs.calculators.workers import WorkerCalculator

__all__ = ["FollowedCalculator", "MapperNetwork", "relax_band", "relax_environment"]


class FollowedCalculator:
    """Track the states of a Python engine with state characters across ask/tell requests."""

    def __init__(self, calculator: EngineCalculator | WorkerCalculator) -> None:
        self._evaluator = FollowedEvaluator(calculator)

    def evaluate(self, task: Task) -> Evaluation:
        cell = None if task.cell_bohr is None else task.cell_bohr.ravel().tolist()
        energy, grad, matrix = self._evaluator.evaluate(
            task.symbols,
            task.positions_bohr.ravel().tolist(),
            task.charge,
            task.multiplicity,
            task.excited_state,
            task.need_hessian,
            cell,
            task.pbc,
        )
        for message in self._evaluator.take_warnings():
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return Evaluation(
            energy,
            np.asarray(grad).reshape(-1, 3),
            None if matrix is None else np.asarray(matrix).reshape(len(grad), len(grad)),
        )
