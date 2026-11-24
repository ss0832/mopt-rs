"""A minimal stand-in for ASE, so the ASE route of the calculators is tested without ASE.

:func:`module` is a module ``ase`` with an ``Atoms`` class following ASE's call conventions;
tests register it only when ASE is not installed (otherwise the real one is used). The calculators below follow
ASE's calculator conventions (eV, Angstrom, ``get_potential_energy(atoms)``,
``get_forces(atoms)``) for a harmonic surface ``E = k/2 |x - x0|^2``, with and without a
Hessian method.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np


class Atoms:
    """The part of ``ase.Atoms`` that mopt-rs uses."""

    def __init__(
        self,
        symbols: Any = (),
        positions: Any = None,
        cell: Any = None,
        pbc: Any = False,
    ) -> None:
        self.symbols = list(symbols)
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.cell = cell
        self.pbc = pbc
        self.info: dict[str, Any] = {}
        self.calc: Any = None

    def __len__(self) -> int:
        return len(self.symbols)

    def get_positions(self) -> np.ndarray:
        return self.positions.copy()

    def get_potential_energy(self, apply_constraint: bool = True) -> float:
        return float(self.calc.get_potential_energy(self))

    def get_forces(self, apply_constraint: bool = True) -> np.ndarray:
        return np.asarray(self.calc.get_forces(self))

    def get_stress(self, voigt: bool = True, apply_constraint: bool = True) -> np.ndarray:
        if not hasattr(self.calc, "get_stress"):
            raise NotImplementedError("no stress")
        return np.asarray(self.calc.get_stress(self))


def module() -> types.ModuleType:
    """The stand-in as a module named ``ase`` (register it with ``monkeypatch.setitem(
    sys.modules, "ase", module())`` when ASE is not installed)."""
    stand_in = types.ModuleType("ase")
    stand_in.Atoms = Atoms  # type: ignore[attr-defined]
    return stand_in


class Harmonic:
    """``E = k/2 |x - x0|^2`` (eV, Angstrom) without a Hessian, like most ASE calculators."""

    implemented_properties = ("energy", "forces")

    def __init__(self, x0: Any, k: float = 2.0) -> None:
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.k = k
        self.hessian_calls = 0

    def _d(self, atoms: Any) -> np.ndarray:
        return np.asarray(atoms.get_positions()) - self.x0

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        d = self._d(atoms)
        return 0.5 * self.k * float(np.sum(d * d))

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        return -self.k * self._d(atoms)


class VolumeSpring:
    """A crystal whose energy depends on the volume of its cell only, ``E = k/2 (V - V0)^2``
    (eV, Angstrom^3): no forces, and the stress ``k (V - V0)`` on the diagonal. ``calls``
    counts what was asked of all instances."""

    implemented_properties = ("energy", "forces", "stress")
    calls: dict[str, int] = {"energy": 0, "stress": 0}  # noqa: RUF012 - shared on purpose

    def __init__(self, v0: float = 64.0, k: float = 0.002) -> None:
        self.v0 = v0
        self.k = k

    def _excess(self, atoms: Any) -> float:
        cell = np.asarray(atoms.cell, dtype=np.float64).reshape(3, 3)
        return abs(float(np.linalg.det(cell))) - self.v0

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        self.calls["energy"] += 1
        return 0.5 * self.k * self._excess(atoms) ** 2

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        return np.zeros((len(atoms), 3))

    def get_stress(self, atoms: Any = None) -> np.ndarray:
        self.calls["stress"] += 1
        s = self.k * self._excess(atoms)
        return np.array([s, s, s, 0.0, 0.0, 0.0])


class DoubleWell:
    """Two atoms whose energy depends on their distance ``r`` (Angstrom) alone,
    ``E = a ((r - c)^2 - w^2)^2`` (eV): minima at ``c - w`` and ``c + w``, the barrier
    ``a w^4`` at ``c``."""

    implemented_properties = ("energy", "forces")

    def __init__(self, a: float = 100.0, c: float = 1.5, w: float = 0.3) -> None:
        self.a, self.c, self.w = a, c, w

    def _r(self, atoms: Any) -> tuple[float, np.ndarray]:
        p = np.asarray(atoms.get_positions(), dtype=np.float64)
        d = p[0] - p[1]
        return float(np.linalg.norm(d)), d

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        r, _ = self._r(atoms)
        return self.a * ((r - self.c) ** 2 - self.w**2) ** 2

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        r, d = self._r(atoms)
        de_dr = 4.0 * self.a * ((r - self.c) ** 2 - self.w**2) * (r - self.c)
        f = np.zeros((len(atoms), 3))
        f[0] = -de_dr * d / r
        f[1] = de_dr * d / r
        return f


class PairLennardJones:
    """``E = sum 4 eps ((s / r)^12 - (s / r)^6)`` over all pairs (eV, Angstrom): a cluster
    with several minima (the argon values of the UFF without arguments)."""

    implemented_properties = ("energy", "forces")

    def __init__(self, epsilon: float = 0.010277, sigma: float = 3.4006) -> None:
        self.epsilon, self.sigma = epsilon, sigma

    def _pairs(self, atoms: Any) -> tuple[float, np.ndarray]:
        p = np.asarray(atoms.get_positions(), dtype=np.float64)
        n = len(p)
        e, f = 0.0, np.zeros((n, 3))
        for i in range(n):
            for j in range(i + 1, n):
                d = p[i] - p[j]
                r = float(np.linalg.norm(d))
                s6 = (self.sigma / r) ** 6
                e += 4.0 * self.epsilon * (s6 * s6 - s6)
                de_dr = 4.0 * self.epsilon * (-12.0 * s6 * s6 + 6.0 * s6) / r
                f[i] -= de_dr * d / r
                f[j] += de_dr * d / r
        return e, f

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        return self._pairs(atoms)[0]

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        return self._pairs(atoms)[1]


class HarmonicWithHessian(Harmonic):
    """With an analytic ``get_hessian`` (eV/Angstrom^2)."""

    def get_hessian(self, atoms: Any) -> np.ndarray:
        self.hessian_calls += 1
        return self.k * np.eye(3 * len(atoms))


class FailingHessian(Harmonic):
    """A ``get_hessian`` that is declared but not implemented."""

    def get_hessian(self, atoms: Any) -> np.ndarray:
        raise NotImplementedError("Hessian not implemented for this method")


class MisshapenHessian(Harmonic):
    """A ``get_hessian`` returning a matrix of the wrong size."""

    def get_hessian(self, atoms: Any) -> np.ndarray:
        return np.eye(2)


HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903


class TwoStateProgram:
    """A program with excited states, as one run through files gives them: two atoms at the
    distance ``d`` (Bohr) with the states ``A = (d - 1.6)^2 / 2`` (the excitation into the
    first empty orbital) and ``B = (d - 3)^2 / 2 + 0.05`` (into the second), in Hartree,
    over a flat ground state. ``root`` chooses the state of the energy and the forces
    (``0``: the ground state; the states are numbered by their energy at the structure),
    and every calculation writes the states to ``mopt_states.json`` in ``directory``, in a
    basis that is not orthonormal and with amplitudes that are not normalized."""

    implemented_properties = ("energy", "forces")
    OVERLAP = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]])

    def __init__(self, directory: str = ".", root: int = 0, nroots: int = 2) -> None:
        self.directory = directory
        self.root = root
        self.nroots = nroots
        self.runs: list[tuple[int, int]] = []

    @staticmethod
    def states(d: float) -> list[tuple[float, float, int]]:
        """Energy, slope, and excitation of the states at the distance ``d``, the lower one
        first (atomic units)."""
        a = (0.5 * (d - 1.6) ** 2, d - 1.6, 0)
        b = (0.5 * (d - 3.0) ** 2 + 0.05, d - 3.0, 1)
        return sorted([a, b])

    def _between(self, atoms: Any) -> tuple[np.ndarray, float]:
        positions = np.asarray(atoms.get_positions()) / BOHR_ANGSTROM
        between = positions[1] - positions[0]
        return between, float(np.linalg.norm(between))

    def _run(self, atoms: Any) -> tuple[float, np.ndarray]:
        import json
        from pathlib import Path

        between, d = self._between(atoms)
        states = self.states(d)[: max(self.nroots, 1)]
        self.runs.append((self.root, self.nroots))
        values, vectors = np.linalg.eigh(self.OVERLAP)
        inverse_root = (vectors / np.sqrt(values)) @ vectors.T
        amplitudes = np.zeros((len(states), 1, 2))
        for k, (_, _, excitation) in enumerate(states):
            amplitudes[k, 0, excitation] = 0.7
        Path(self.directory, "mopt_states.json").write_text(
            json.dumps(
                {
                    "coefficients": inverse_root.tolist(),
                    "overlap": self.OVERLAP.tolist(),
                    "occupied": 1,
                    "energies": [energy + 1.0 for energy, _, _ in states],
                    "amplitudes": amplitudes.tolist(),
                }
            )
        )
        if self.root == 0:
            return -1.0 * HARTREE_EV, np.zeros((2, 3))
        energy, slope, _ = self.states(d)[self.root - 1]
        pull = slope * between / d * (HARTREE_EV / BOHR_ANGSTROM)
        return energy * HARTREE_EV, np.array([pull, -pull])

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        return self._run(atoms)[0]

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        return self._run(atoms)[1]


def choose_root(calculator: Any, state: int, count: int) -> None:
    """``excited_setup`` of :class:`TwoStateProgram`."""
    calculator.root = state
    calculator.nroots = count


class NativeEmt:
    """The EMT of mopt-rs behind the conventions of an ASE calculator (eV, Angstrom; the stress
    in eV/Angstrom^3 as ASE's six components, xx yy zz yz xz xy): a periodic run through the ASE
    route with a potential whose results the built-in route gives too."""

    implemented_properties = ("energy", "forces", "stress")
    calls: dict[str, int] = {"energy": 0, "stress": 0}  # noqa: RUF012 - shared on purpose

    HARTREE_EV = 27.211386245988
    BOHR_ANGSTROM = 0.529177210903

    def _args(self, atoms: Any) -> tuple[list[str], list[float], list[float] | None]:
        symbols = list(atoms.symbols)
        x = (np.asarray(atoms.get_positions(), dtype=np.float64) / self.BOHR_ANGSTROM).ravel()
        cell = None
        if atoms.cell is not None and np.any(atoms.pbc):
            lattice = np.asarray(atoms.cell, dtype=np.float64).reshape(3, 3)
            cell = (lattice / self.BOHR_ANGSTROM).ravel().tolist()
        return symbols, x.tolist(), cell

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        from mopt_rs import _core

        self.calls["energy"] += 1
        symbols, x, cell = self._args(atoms)
        energy, _, _ = _core.evaluate_native("emt", symbols, x, 0, 1, False, cell)
        return float(energy) * self.HARTREE_EV

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        from mopt_rs import _core

        symbols, x, cell = self._args(atoms)
        _, gradient, _ = _core.evaluate_native("emt", symbols, x, 0, 1, False, cell)
        g = np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
        return -g * self.HARTREE_EV / self.BOHR_ANGSTROM

    def get_stress(self, atoms: Any = None) -> np.ndarray:
        from mopt_rs import _core

        self.calls["stress"] += 1
        symbols, x, cell = self._args(atoms)
        if cell is None:
            raise NotImplementedError("no cell")
        s = np.asarray(_core.native_stress("emt", symbols, x, cell), dtype=np.float64)
        s *= self.HARTREE_EV / self.BOHR_ANGSTROM**3
        return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def arranged(h: np.ndarray, layout: str) -> np.ndarray:
    """The (3N, 3N) matrix ``h`` in ``layout``: ``(3N, 3N)``, ``(N, 3, N, 3)``,
    ``(3N, N, 3)``, or ``(N, N, 3, 3)`` (atoms first, then their coordinates)."""
    n = h.shape[0] // 3
    if layout == "(3N, 3N)":
        return h.copy()
    if layout == "(N, 3, N, 3)":
        return h.reshape(n, 3, n, 3).copy()
    if layout == "(3N, N, 3)":
        return h.reshape(3 * n, n, 3).copy()
    if layout == "(N, N, 3, 3)":
        return h.reshape(n, 3, n, 3).transpose(0, 2, 1, 3).copy()
    raise ValueError(layout)


class Quadratic:
    """``E = 1/2 d^T K d`` with ``d = x - x0`` (eV, Angstrom) and a general symmetric ``K``
    (eV/Angstrom^2), whose Hessian the subclasses hand out in the ways ASE calculators do: in
    ``layout``, times ``scale`` (a Hessian in another unit), and with ``skew`` times an
    antisymmetric matrix added. ``hessian_calls`` counts the Hessians asked for."""

    implemented_properties: tuple[str, ...] = ("energy", "forces")

    def __init__(
        self,
        x0: Any,
        k: Any,
        layout: str = "(3N, 3N)",
        scale: float = 1.0,
        skew: float = 0.0,
    ) -> None:
        self.x0 = np.asarray(x0, dtype=np.float64).ravel()
        self.k = np.asarray(k, dtype=np.float64)
        self.layout, self.scale, self.skew = layout, scale, skew
        self.hessian_calls = 0
        self.force_calls = 0
        self.last: np.ndarray | None = None

    def _d(self, atoms: Any) -> np.ndarray:
        self.last = np.asarray(atoms.get_positions(), dtype=np.float64).ravel()
        return self.last - self.x0

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        d = self._d(atoms)
        return 0.5 * float(d @ self.k @ d)

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        self.force_calls += 1
        return -(self.k @ self._d(atoms)).reshape(-1, 3)

    def handed_out(self) -> np.ndarray:
        self.hessian_calls += 1
        n3 = self.k.shape[0]
        a = np.arange(n3 * n3, dtype=np.float64).reshape(n3, n3) / (n3 * n3)
        h = self.k + self.skew * float(np.max(np.abs(self.k))) * (a - a.T)
        return arranged(h * self.scale, self.layout)


class QuadraticGetHessian(Quadratic):
    """``get_hessian(atoms)``."""

    def get_hessian(self, atoms: Any = None) -> np.ndarray:
        return self.handed_out()


class QuadraticCalculateHessian(Quadratic):
    """``calculate_hessian()`` without an argument: of the structure of the last calculation."""

    def calculate_hessian(self) -> np.ndarray:
        if self.last is None:
            raise RuntimeError("nothing calculated yet")
        return self.handed_out()


class QuadraticHessianMethod(Quadratic):
    """A method ``hessian(atoms)``."""

    def hessian(self, atoms: Any) -> np.ndarray:
        return self.handed_out()


class QuadraticHessianProperty(Quadratic):
    """The property ``hessian`` through ASE's ``get_property``."""

    implemented_properties = ("energy", "forces", "hessian")

    def get_property(self, name: str, atoms: Any = None, allow_calculation: bool = True) -> Any:
        if name != "hessian":
            raise KeyError(name)
        return self.handed_out()


class QuadraticResults(Quadratic):
    """The Hessian among the results of every calculation, as ASE's ``ForceField`` keeps it,
    and no method for it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.results: dict[str, Any] = {}

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        forces = super().get_forces(atoms)
        self.results = {"forces": forces, "hessian": self.handed_out()}
        return forces


class QuadraticBrokenGetHessian(QuadraticResults):
    """A ``get_hessian`` that fails while the results hold the Hessian (ASE 3.26's
    ``ForceField``, whose ``get_hessian`` asks for a property it does not list)."""

    def get_hessian(self, atoms: Any = None) -> np.ndarray:
        raise NotImplementedError("hessian property not implemented")


class QuadraticNumerical(Quadratic):
    """Only a ``get_numerical_hessian``, which the automatic route does not call."""

    def get_numerical_hessian(self, atoms: Any = None) -> np.ndarray:
        return self.handed_out()
