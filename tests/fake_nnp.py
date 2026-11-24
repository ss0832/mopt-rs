"""A stand-in for a machine-learned potential with an ASE calculator, built the way such
packages are: a model is loaded first (here: counted) and handed to the calculator.

``load_model`` and ``Potential`` follow the two-step pattern (``model = load(...)``,
``calc = Calculator(model, device=...)``); ``Zoo.ase()`` follows the pattern of a model object
that returns its own calculator. The surface is harmonic, ``E = k/2 |x - x0|^2`` in eV and
Angstrom, so results are known.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Any

import numpy as np

LOADS: list[dict[str, Any]] = []
"""One entry per loaded model (to check that a run loads the model once)."""


class Model:
    """A loaded model: a force constant and the device it lives on."""

    def __init__(self, k: float, device: str, path: str | None) -> None:
        self.k = k
        self.device = device
        self.path = path


def load_model(name: str = "small", device: str = "cpu", path: str | None = None) -> Model:
    """Load the model ``name`` (``small``: k = 1, ``large``: k = 2), optionally from ``path``."""
    k = {"small": 1.0, "large": 2.0}[name]
    LOADS.append({"name": name, "device": device, "path": path})
    return Model(k, device, path)


class Potential:
    """ASE-style calculator around a loaded model."""

    def __init__(self, model: Model, device: str = "cpu", shift: float = 0.0) -> None:
        if device != model.device:
            raise ValueError(f"model on {model.device}, calculator on {device}")
        self.model = model
        self.device = device
        self.shift = shift
        self.calls = 0

    def get_potential_energy(self, atoms: Any = None) -> float:
        self.calls += 1
        x = np.asarray(atoms.positions, dtype=np.float64)
        return float(0.5 * self.model.k * np.sum((x - self.shift) ** 2))

    def get_forces(self, atoms: Any = None) -> Any:
        x = np.asarray(atoms.positions, dtype=np.float64)
        return -self.model.k * (x - self.shift)


class Zoo:
    """A model object that returns its own calculator (``Zoo("large").ase()``)."""

    def __init__(self, name: str = "small", device: str = "cpu") -> None:
        self.model = load_model(name, device)

    def ase(self) -> Potential:
        return Potential(self.model, self.model.device)
