"""The calculator contract: engines compute, :class:`EngineCalculator` does everything else.

MultiOptPy repeats the same services in every calculator module: reading geometry files from
run directories, converting units, parsing charge and multiplicity (each with its own spin
convention), scheduling exact Hessians, projecting out rigid-body motion, dumping ``.npy``
files, and turning exceptions into a "finished" flag. Here an :class:`Engine` only maps a
:class:`Structure` to energy and derivatives in its own units; :class:`EngineCalculator`
adds, once for every engine:

- conversion to atomic units (Hartree, Bohr) from the engine's declared :class:`EngineUnits`;
- the electronic state in one convention (:class:`ElectronicState`, multiplicity ``2S + 1``)
  translated by :class:`SpinConvention` for each program;
- Hessians by central differences of the engine's gradient when it has none of its own;
- a small cache, so that a Hessian requested at the structure of the last gradient reuses it;
- shape and finiteness checks with the engine's name in every error.

When to compute Hessians, rigid-body projection, and output files belong to the optimizer
and the drivers, not to calculators.

MultiOptPy: the `Calculation` and `CalculationEngine` classes repeated in every
 `Calculator/*_calculation_tools.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import enum
import inspect
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from mopt_rs import _core
from mopt_rs.calculators.stability import StabilityCheck, StabilityReport
from mopt_rs.optimize import Evaluation, Task

WORKER_ENV = "MOPT_WORKER"
"""Set in a worker process of :mod:`mopt_rs.calculators.workers` to its number (0, 1, ...)."""


def worker_index() -> int | None:
    """The number of this process among the workers of a calculator; ``None`` in a process
    that is not a worker."""
    text = os.environ.get(WORKER_ENV)
    return int(text) if text and text.isdigit() else None


FloatArray = npt.NDArray[np.float64]

_UNITS = _core.units()
BOHR_TO_ANGSTROM: float = _UNITS["bohr_to_angstrom"]
HARTREE_TO_EV: float = _UNITS["hartree_to_ev"]
HARTREE_TO_KCALMOL: float = _UNITS["hartree_to_kcalmol"]
HARTREE_TO_KJMOL: float = _UNITS["hartree_to_kjmol"]

__all__ = [
    "ATOMIC_UNITS",
    "ASE_UNITS",
    "BOHR_TO_ANGSTROM",
    "HARTREE_TO_EV",
    "Capabilities",
    "CalculatorError",
    "missing_package",
    "ElectronicState",
    "Engine",
    "EngineCalculator",
    "EngineResult",
    "EngineUnits",
    "SpinConvention",
    "Structure",
]


class CalculatorError(RuntimeError):
    """An engine failed or returned something unusable."""


def missing_package(package: str, extra: str) -> CalculatorError:
    """The error of an engine whose package is not installed: the engines of other packages
    are optional, installed as they are needed (``pip install "mopt-py[EXTRA]"``)."""
    return CalculatorError(
        f"the package '{package}' is not installed (the engines of other packages are "
        f'optional): pip install "mopt-py[{extra}]", or pip install {package} (all the '
        "engines and what they need: python -m mopt_rs.calculators)"
    )


class SpinConvention(enum.Enum):
    """How a program expects the spin state."""

    MULTIPLICITY = "multiplicity"
    """``2S + 1`` (ASE semiempirical engines, Psi4, ORCA, Gaussian, FAIRChem)."""
    UNPAIRED = "unpaired"
    """``2S = N_alpha - N_beta`` (PySCF ``mol.spin``, xTB/tblite ``uhf``, dxtb ``spin``)."""
    SPIN = "spin"
    """``S`` itself (half-integers)."""


@dataclass(frozen=True)
class ElectronicState:
    """Total charge and spin multiplicity ``2S + 1``: the only convention used in mopt-rs."""

    charge: int = 0
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if self.multiplicity < 1:
            raise ValueError(
                f"multiplicity must be 2S + 1 >= 1, got {self.multiplicity} "
                "(PySCF's mol.spin is 2S: give multiplicity = spin + 1)"
            )

    @property
    def unpaired_electrons(self) -> int:
        """``2S``."""
        return self.multiplicity - 1

    def spin_as(self, convention: SpinConvention) -> int | float:
        """The spin in ``convention``."""
        if convention is SpinConvention.MULTIPLICITY:
            return self.multiplicity
        if convention is SpinConvention.UNPAIRED:
            return self.unpaired_electrons
        return self.unpaired_electrons / 2.0

    @staticmethod
    def from_spin(charge: int, value: float, convention: SpinConvention) -> ElectronicState:
        """The state from a spin value in ``convention``."""
        if convention is SpinConvention.MULTIPLICITY:
            mult = value
        elif convention is SpinConvention.UNPAIRED:
            mult = value + 1
        else:
            mult = 2 * value + 1
        if abs(mult - round(mult)) > 1e-9:
            raise ValueError(f"spin {value} ({convention.value}) is not a whole multiplicity")
        return ElectronicState(int(charge), int(round(mult)))

    def check_electrons(self, numbers: npt.ArrayLike) -> None:
        """Raise when the multiplicity is impossible for the electron count."""
        electrons = int(np.sum(np.asarray(numbers))) - self.charge
        if electrons < 0:
            raise ValueError(f"charge {self.charge} leaves {electrons} electrons")
        if electrons < self.unpaired_electrons or (electrons - self.unpaired_electrons) % 2:
            raise ValueError(
                f"multiplicity {self.multiplicity} is impossible with {electrons} electrons"
            )


@dataclass(frozen=True)
class Structure:
    """What an engine receives: Cartesian positions and the electronic state."""

    symbols: tuple[str, ...]
    numbers: npt.NDArray[np.int64]
    positions_bohr: FloatArray
    """Shape (N, 3)."""
    state: ElectronicState = field(default_factory=ElectronicState)
    cell_bohr: FloatArray | None = None
    """Lattice vectors as rows, shape (3, 3)."""
    pbc: tuple[bool, bool, bool] = (False, False, False)
    excited_state: int = 0
    label: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def positions_angstrom(self) -> FloatArray:
        return self.positions_bohr * BOHR_TO_ANGSTROM

    @property
    def cell_angstrom(self) -> FloatArray | None:
        return None if self.cell_bohr is None else self.cell_bohr * BOHR_TO_ANGSTROM

    @property
    def is_periodic(self) -> bool:
        return any(self.pbc)

    def engine_copy(self) -> Structure:
        """Give a backend private arrays so its coordinate conventions cannot move the caller."""
        return replace(
            self,
            numbers=self.numbers.copy(),
            positions_bohr=self.positions_bohr.copy(),
            cell_bohr=None if self.cell_bohr is None else self.cell_bohr.copy(),
        )

    def displaced(self, dof: int, step_bohr: float) -> Structure:
        """A copy with one Cartesian component moved."""
        x = self.positions_bohr.copy()
        x.flat[dof] += step_bohr
        return Structure(
            self.symbols,
            self.numbers,
            x,
            self.state,
            self.cell_bohr,
            self.pbc,
            self.excited_state,
            self.label,
        )

    def key(self) -> tuple[Any, ...]:
        """Identity of the structure for caching."""
        cell = None if self.cell_bohr is None else self.cell_bohr.tobytes()
        return (
            self.symbols,
            self.positions_bohr.tobytes(),
            self.state,
            cell,
            self.pbc,
            self.excited_state,
        )

    @staticmethod
    def from_task(task: Task) -> Structure:
        numbers = np.asarray(_core.atomic_numbers(task.symbols), dtype=np.int64)
        cell = task.cell_bohr
        return Structure(
            tuple(task.symbols),
            numbers,
            np.asarray(task.positions_bohr, dtype=np.float64).reshape(-1, 3),
            ElectronicState(task.charge, task.multiplicity),
            cell,
            task.pbc,
            task.excited_state,
            task.label,
        )

    @staticmethod
    def build(
        symbols: list[str] | tuple[str, ...],
        positions_bohr: npt.ArrayLike,
        *,
        charge: int = 0,
        multiplicity: int | None = None,
        cell_bohr: npt.ArrayLike | None = None,
        pbc: tuple[bool, bool, bool] | None = None,
        excited_state: int = 0,
        unpaired_electrons: int | None = None,
    ) -> Structure:
        """A structure for standalone use of an engine. The spin is the multiplicity ``2S + 1``
        or, instead, ``unpaired_electrons`` ``2S`` (the "spin" of PySCF, xTB, and several Rust
        engines); both must agree when given; the default is a singlet."""
        if unpaired_electrons is not None:
            if unpaired_electrons < 0:
                raise ValueError(f"unpaired_electrons cannot be negative ({unpaired_electrons})")
            if multiplicity is not None and multiplicity != unpaired_electrons + 1:
                raise ValueError(
                    f"multiplicity {multiplicity} and unpaired_electrons {unpaired_electrons} "
                    "disagree (multiplicity = 2S + 1)"
                )
            multiplicity = unpaired_electrons + 1
        if multiplicity is None:
            multiplicity = 1
        cell = None if cell_bohr is None else np.asarray(cell_bohr, np.float64).reshape(3, 3)
        if pbc is None:
            pbc = (True, True, True) if cell is not None else (False, False, False)
        return Structure(
            tuple(symbols),
            np.asarray(_core.atomic_numbers(list(symbols)), dtype=np.int64),
            np.asarray(positions_bohr, dtype=np.float64).reshape(-1, 3),
            ElectronicState(charge, multiplicity),
            cell,
            pbc,
            excited_state,
        )


@dataclass(frozen=True)
class EngineUnits:
    """Units of an engine's results."""

    energy: str = "hartree"
    """``hartree``, ``ev``, ``kcal/mol``, or ``kj/mol``."""
    length: str = "bohr"
    """``bohr`` or ``angstrom``."""

    @property
    def energy_to_hartree(self) -> float:
        factors = {
            "hartree": 1.0,
            "ev": 1.0 / HARTREE_TO_EV,
            "kcal/mol": 1.0 / HARTREE_TO_KCALMOL,
            "kj/mol": 1.0 / HARTREE_TO_KJMOL,
        }
        try:
            return factors[self.energy.lower()]
        except KeyError:
            raise ValueError(f"unknown energy unit '{self.energy}'") from None

    @property
    def length_to_bohr(self) -> float:
        factors = {"bohr": 1.0, "angstrom": 1.0 / BOHR_TO_ANGSTROM}
        try:
            return factors[self.length.lower()]
        except KeyError:
            raise ValueError(f"unknown length unit '{self.length}'") from None


ATOMIC_UNITS = EngineUnits("hartree", "bohr")
ASE_UNITS = EngineUnits("ev", "angstrom")


@dataclass(frozen=True)
class Capabilities:
    """What an engine can do itself."""

    analytic_hessian: bool = False
    """The engine returns analytic Hessians (otherwise they come from finite differences).
    They are those of the positions: an engine has no place for second derivatives along the
    strain of a cell, which a variable-cell run takes from differences."""
    numerical_hessian: bool = False
    """The engine returns Hessians that its program takes from differences itself (a method
    named as numerical): asked of the engine like analytic ones, but not analytic."""
    pbc: bool = False
    excited_states: bool = False
    electrons: bool = True
    """Charge and multiplicity matter (quantum chemistry); False for force fields."""
    rigid_body_invariant: bool = True
    """The energy does not change under rigid translations and rotations (False for surfaces
    fixed in space, such as model potentials or external fields)."""
    point_charges: bool = False
    """The engine implements compute_embedded in an external field of point charges."""


@dataclass
class EngineResult:
    """An engine's result in the engine's units.

    Give either ``gradient`` (dE/dx) or ``forces`` (-dE/dx), shape (N, 3) or (3N,); the
    Hessian, when computed, has shape (3N, 3N) or (N, 3, N, 3).
    """

    energy: float
    gradient: npt.ArrayLike | None = None
    forces: npt.ArrayLike | None = None
    hessian: npt.ArrayLike | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Anything else worth keeping (charges, dipole, stress, the SCF object). ``spin_square``
    (the expectation value ``<S^2>``) is shown at every iteration of a run."""


class Engine(ABC):
    """A program that computes energies and derivatives."""

    name: str = "engine"
    units: EngineUnits = ATOMIC_UNITS
    capabilities: Capabilities = Capabilities()
    program: str | None = None
    """Top-level module of the program the engine runs (``"pyscf"``), for :meth:`describe`."""
    notes: tuple[str, ...] = ()
    """Remarks for :meth:`describe` (for example a path taken from ``software_path.conf``)."""
    citations: tuple[str, ...] = ()
    """The papers to cite for the program and the method, shown in the run log."""

    def describe(self) -> str:
        """Where the program comes from (package, version, file), for run logs."""
        from mopt_rs.calculators.provenance import module_origin

        text = self.name if self.program is None else f"{self.name}: {module_origin(self.program)}"
        return "; ".join((text, *self.notes))

    def compute_embedded(
        self,
        structure: Structure,
        positions_bohr: FloatArray,
        charges: FloatArray,
        elements: list[str | None],
        hessian: bool,
    ) -> tuple[EngineResult, FloatArray]:
        """Evaluate in fixed point charges and return their position gradients in engine units."""
        raise CalculatorError(f"{self.name} does not implement electronic embedding")

    @abstractmethod
    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        """Energy and gradient (and the Hessian when ``hessian``; only requested when
        ``capabilities.analytic_hessian`` or ``capabilities.numerical_hessian``). Lines for
        the run log go in ``EngineResult.extra`` under :data:`NOTES`.

        An engine that can give the stress of a periodic cell declares a third parameter,
        ``compute(self, structure, hessian, stress=False)``: it is called with
        ``stress=True`` when the stress is needed (a variable-cell optimization) and puts it
        in ``EngineResult.extra`` under one of :data:`STRESS_KEYS`. An engine that can give
        the energy alone more cheaply declares ``gradient=True`` among its parameters: it is
        called with ``gradient=False`` when only the energy is needed (the other electronic
        states of surface hopping) and may then return no gradient."""

    def orbitals(self, structure: Structure) -> dict[str, Any] | None:
        """The molecular orbitals of ``structure``, for analyses of the wavefunction
        (natural reaction orbitals, orbital energies along a path), or ``None`` if the engine
        does not hand them out: ``{"overlap": S, "sets": [{"coefficients": C, "energies": e,
        "occupations": n}, ...]}`` with one set (the orbitals of a closed shell, occupations
        up to 2) or two (alpha, then beta). ``S`` is the overlap of the atomic orbitals and
        ``C`` has a column per orbital. An engine may add ``"atom_of_ao"``, the atom (0-based)
        of every atomic orbital, which the transfer integrals of the projection method need
        (the orbitals of a molecule alone written in the atomic orbitals of a pair)."""
        return None

    def transitions(self, structure: Structure, count: int) -> list[dict[str, Any]] | None:
        """The lowest ``count`` excited states of ``structure`` with what the coupling of an
        electron transfer needs (Mulliken-Hush in ``etmain``), or ``None`` if the engine does not
        hand them out: a list of ``{"excitation": E, "transition_dipole": mu,
        "population_change": d}``, ``E`` in Hartree, ``mu`` the transition dipole from the ground
        state (three numbers, e Bohr), and ``d`` the electrons each atom gains from the ground to
        the excited state (one number per atom, negative for a loss; the populations of the
        unrelaxed difference density, as Gaussian's ``Pop=dCT``). An engine may add
        ``"dipole_change"`` (the dipole of the state less that of the ground state, three
        numbers, e Bohr; the generalized Mulliken-Hush coupling) and ``"transition_population"``
        (the Mulliken populations of the transition density on each atom; the fragment charge
        difference)."""
        return None

    def state_characters(self, structure: Structure, count: int) -> dict[str, Any] | None:
        """The lowest ``count`` excited states of ``structure`` in the form that is compared
        between structures, so that a state can be followed (``-es_follow``), or ``None``
        if the engine does not hand them out: ``{"orbitals": C, "occupied": n, "energies":
        e, "amplitudes": X}``. ``C`` has a column per orbital in an orthonormal basis that
        belongs to the atoms (Loewdin orthogonalized atomic orbitals, ``S**0.5`` times the
        coefficients), the doubly occupied ones first; ``e`` are the excitation energies
        (Hartree, ascending); ``X`` has a column per state with the amplitudes of the
        single excitations ``(i, a)`` at ``i * n_virtual + a``, of length one."""
        return None

    def check_stability(self, structure: Structure, external: str) -> StabilityReport | None:
        """Test the SCF solution of ``structure`` for instabilities (``-stability``; see
        :mod:`mopt_rs.calculators.stability`): if it is unstable, move to a stabler one so
        that the next :meth:`compute` of ``structure`` gives it, and say so. ``external``
        (``off``, ``report``, ``switch``) is what to do about an instability towards another
        kind of wavefunction. ``None``: the engine has no such test (the default)."""
        return None

    def use_hessian(
        self, hessian: str | None = "auto", *, units: str = "eV/Angstrom^2", check: bool = False
    ) -> None:
        """Choose where Hessians come from (``-calc_opt hessian=...``): ``"auto"`` (the
        engine's own where it has one, else differences), ``"analytic"`` (the engine's own;
        an error for an engine without), or ``"numerical"`` (differences of the gradient even
        where the engine has its own). The ASE engines also take the name of a method and
        the units and the check of their arrays
        (:meth:`mopt_rs.calculators.ase.AseEngine.use_hessian`)."""
        choice = "numerical" if hessian is None else str(hessian).strip().lower()
        if units != "eV/Angstrom^2" or check:
            raise ValueError(
                f"{self.name}: hessian_units and hessian_check are settings of ASE calculators"
            )
        own = self.capabilities.analytic_hessian or self.capabilities.numerical_hessian
        if choice == "auto":
            return
        if choice == "analytic":
            if not own:
                raise ValueError(
                    f"{self.name}: hessian=analytic, but the engine has no Hessian of its own "
                    "(auto or numerical: differences of the gradient)"
                )
            return
        if choice in ("numerical", "numeric"):
            if own:
                self.capabilities = replace(
                    self.capabilities, analytic_hessian=False, numerical_hessian=False
                )
                self.notes = (*self.notes, "Hessian: differences of the gradient (hessian=numerical)")
            return
        raise ValueError(
            f"{self.name}: hessian={hessian} is not auto, analytic, or numerical (the name of a "
            "method is taken by ASE calculators only)"
        )

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (files, GPU memory, processes)."""


@dataclass
class _Cached:
    energy: float
    gradient: FloatArray
    hessian: FloatArray | None
    extra: dict[str, Any]


STRESS_ASKED = "stress_asked"
"""Key of the results of an evaluation in which the engine was asked for the stress."""
NO_GRADIENT = "no_gradient"
"""Marks a result of the energy alone (the gradient was not computed and is zero)."""
NOTES = "notes"
"""Key of ``EngineResult.extra`` under which an engine hands out lines for the run log (a
list of strings, shown once with the evaluation)."""

STRESS_KEYS = ("stress_hartree_per_bohr3", "stress_voigt_ev_per_a3")
"""Keys of ``EngineResult.extra`` under which an engine hands out the stress of a periodic
cell: the 3 x 3 tensor in Hartree/Bohr^3, or the six components of ASE (``xx, yy, zz, yz,
xz, xy`` in eV/Angstrom^3); both ``(1/V) dE/d(strain)``."""


def _stress(extra: Mapping[str, Any]) -> FloatArray | None:
    """The stress among the results of an engine as a 3 x 3 tensor in Hartree/Bohr^3."""
    tensor = extra.get(STRESS_KEYS[0])
    if tensor is not None:
        found = np.asarray(tensor, dtype=np.float64)
        if found.shape != (3, 3):
            raise CalculatorError(f"a stress of shape {found.shape} (3 x 3 expected)")
    else:
        voigt = extra.get(STRESS_KEYS[1])
        if voigt is None:
            return None
        six = np.asarray(voigt, dtype=np.float64).ravel()
        if six.size != 6:
            raise CalculatorError(f"a stress of {six.size} components (6 expected)")
        found = np.asarray(_core.stress_from_ase_voigt(six.tolist()), dtype=np.float64)
    if not np.all(np.isfinite(found)):
        raise CalculatorError("non-finite stress")
    return 0.5 * (found + found.T)


HESSIAN_METHODS = ("central", "o1numhess")
"""Where a Hessian comes from when the engine has none (see :class:`EngineCalculator`)."""


class EngineCalculator:
    """An :class:`Engine` as a mopt-rs calculator (``evaluate(Task) -> Evaluation``).

    When the engine has no Hessian of its own, ``hessian_differences`` says where one comes
    from: ``"central"`` (central differences along every coordinate, 2 x 3N gradient
    evaluations with the displacement ``hessian_step_bohr``) or ``"o1numhess"`` (O1NumHess: a
    number of gradients that does not grow with the size of the system). ``hessian_method``
    is its older name and still accepted (``hessian_differences`` wins when both are given).

    With ``stability=N`` every ``N``-th evaluation that runs the engine asks it to test its
    SCF solution (:meth:`Engine.check_stability`, ``stability_external`` for instabilities
    towards another kind of wavefunction) and is repeated where the engine moved to a
    stabler solution; what was found comes with the evaluation (``Evaluation.notes``).
    """

    def __init__(
        self,
        engine: Engine,
        *,
        hessian_step_bohr: float = 1e-3,
        hessian_method: str = "central",
        cache_size: int = 4,
        check_electrons: bool = True,
        stability: int = 0,
        stability_external: str = "report",
        hessian_differences: str | None = None,
    ) -> None:
        if hessian_step_bohr <= 0.0:
            raise ValueError("hessian_step_bohr must be positive")
        if hessian_differences is not None:
            hessian_method = hessian_differences
        if hessian_method not in HESSIAN_METHODS:
            raise ValueError(
                f"hessian_differences (hessian_method) must be one of {', '.join(HESSIAN_METHODS)}"
            )
        self.engine = engine
        self.hessian_step_bohr = hessian_step_bohr
        self.hessian_method = hessian_method
        self.check_electrons = check_electrons
        self._cache: OrderedDict[tuple[Any, ...], _Cached] = OrderedDict()
        self._cache_size = max(cache_size, 1)
        self._takes_stress = "stress" in inspect.signature(engine.compute).parameters
        self._takes_gradient = "gradient" in inspect.signature(engine.compute).parameters
        self.stability = StabilityCheck(stability, stability_external)
        self.evaluations = 0
        """Engine calls so far (including finite-difference displacements)."""
        # The commands evaluate several structures (the images of a chain, the two
        # structures of a pair) in parallel threads. An engine is one object with one
        # program behind it, which is not made for two calls at once, so the calls into it
        # wait for each other.
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self.engine.name

    def describe(self) -> str:
        """Where the engine's program comes from (see :meth:`Engine.describe`)."""
        return self.engine.describe()

    def evaluate(self, task: Task) -> Evaluation:
        """Evaluate an optimizer task."""
        return self.evaluate_structure(Structure.from_task(task), task.need_hessian)

    def evaluate_structure(
        self,
        structure: Structure,
        need_hessian: bool = False,
        need_stress: bool = False,
        need_gradient: bool = True,
    ) -> Evaluation:
        """Energy (Hartree), gradient (Hartree/Bohr, shape (N, 3)) and, when requested, the
        Hessian (Hartree/Bohr^2, shape (3N, 3N)) and the stress of the cell (Hartree/Bohr^3,
        shape (3, 3); ``None`` where the engine has none) at ``structure``. Calls from
        several threads are made one after the other. Without ``need_gradient`` an engine
        that can leave it out gives the energy alone, and the gradient is zero."""
        with self._lock:
            return self._evaluate_structure(structure, need_hessian, need_stress, need_gradient)

    def _evaluate_structure(
        self,
        structure: Structure,
        need_hessian: bool,
        need_stress: bool,
        need_gradient: bool = True,
    ) -> Evaluation:
        caps = self.engine.capabilities
        if structure.is_periodic and not caps.pbc:
            raise CalculatorError(f"{self.name}: periodic structures are not supported")
        if structure.excited_state > 0 and not caps.excited_states:
            raise CalculatorError(f"{self.name}: excited states are not supported")
        if self.check_electrons and caps.electrons:
            try:
                structure.state.check_electrons(structure.numbers)
            except ValueError as e:
                raise CalculatorError(f"{self.name}: {e}") from None
        key = structure.key()
        hit = self._cache.get(key)
        analytic = need_hessian and (caps.analytic_hessian or caps.numerical_hessian)
        stress = need_stress and structure.is_periodic and self._takes_stress
        energy_only = not (need_gradient or need_hessian or stress) and self._takes_gradient
        notes: list[str] = []
        if (
            hit is None
            or (analytic and hit.hessian is None)
            or (stress and not hit.extra.get(STRESS_ASKED))
            or (not energy_only and hit.extra.get(NO_GRADIENT))
        ):
            hit = self._run(structure, analytic, stress, gradient=not energy_only)
            if self.stability.due():
                report = self._check_stability(structure)
                if report is not None:
                    notes.extend(report.lines)
                    if report.changed:
                        hit = self._run(structure, analytic, stress)
        if need_hessian and hit.hessian is None:
            # No Hessian of the engine's own (or none for this state): differences of the
            # gradient; the gradient at the structure itself is reused.
            if self.hessian_method == "o1numhess":
                hit.hessian = self._o1numhess_hessian(structure, hit.gradient)
            else:
                hit.hessian = self._finite_difference_hessian(structure)
        self._remember(key, hit)
        # What the engine has to say for the run log, once.
        notes.extend(str(line) for line in hit.extra.pop(NOTES, ()))
        spin_square = hit.extra.get("spin_square")
        # What the program printed is handed on once, with the evaluation that produced it.
        output = hit.extra.pop("output", None)
        return Evaluation(
            hit.energy,
            hit.gradient.copy(),
            None if not need_hessian else hit.hessian,
            None if spin_square is None else float(spin_square),
            None if output is None else str(output),
            _stress(hit.extra) if need_stress else None,
            tuple(notes),
        )

    def evaluate_embedded(
        self,
        structure: Structure,
        positions_bohr: npt.ArrayLike,
        charges: npt.ArrayLike,
        elements: list[str | None],
        hessian: bool = False,
    ) -> tuple[Evaluation, FloatArray]:
        """Compute the model and charge gradients in atomic units, without sharing a vacuum cache."""
        p = np.asarray(positions_bohr, dtype=float).reshape(-1, 3)
        q = np.asarray(charges, dtype=float).reshape(-1)
        if (
            len(p) != len(q)
            or len(q) != len(elements)
            or not (np.isfinite(p).all() and np.isfinite(q).all())
        ):
            raise CalculatorError("invalid point charges")
        caps = self.engine.capabilities
        if not caps.point_charges or (structure.excited_state and not caps.excited_states):
            raise CalculatorError(f"{self.name}: unsupported embedded state")
        if structure.is_periodic and not caps.pbc:
            raise CalculatorError(f"{self.name}: periodic structures are not supported")
        if self.check_electrons and caps.electrons:
            structure.state.check_electrons(structure.numbers)
        with self._lock:
            model = structure.engine_copy()
            field = p.copy()
            result, field_gradient = self.engine.compute_embedded(
                model, field, q.copy(), elements, hessian
            )
            shift = self._frame_translation(structure.positions_bohr, model.positions_bohr)
            if not np.allclose(field - p, shift, rtol=0.0, atol=1e-9):
                raise CalculatorError(f"{self.name}: model and point charges changed frames separately")
            self.evaluations += 1
            value = self._convert(structure, result)
            g = np.asarray(field_gradient, dtype=float)
            if g.shape != p.shape or not np.isfinite(g).all():
                raise CalculatorError("invalid point charge gradients")
            factor = self.engine.units.energy_to_hartree / self.engine.units.length_to_bohr
            return Evaluation(value.energy, value.gradient, value.hessian), g * factor

    @property
    def checks_stability(self) -> bool:
        """Whether the engine tests its SCF solutions (``-stability``)."""
        return type(self.engine).check_stability is not Engine.check_stability

    def _check_stability(self, structure: Structure) -> StabilityReport | None:
        try:
            report = self.engine.check_stability(structure.engine_copy(), self.stability.external)
        except Exception as e:
            raise CalculatorError(f"{self.name}: the test of the SCF solution failed: {e}") from e
        if report is None and not self.stability.unsupported_said:
            self.stability.unsupported_said = True
            return StabilityReport(
                False,
                [f"-stability: {self.name} has no test of its SCF solution; nothing is tested"],
            )
        return report

    def orbitals(self, structure: Structure) -> dict[str, Any] | None:
        """The molecular orbitals of ``structure`` as the engine hands them out (see
        :meth:`Engine.orbitals`), as float arrays; ``None`` if it does not."""
        caps = self.engine.capabilities
        if structure.is_periodic and not caps.pbc:
            raise CalculatorError(f"{self.name}: periodic structures are not supported")
        if self.check_electrons and caps.electrons:
            try:
                structure.state.check_electrons(structure.numbers)
            except ValueError as e:
                raise CalculatorError(f"{self.name}: {e}") from None
        try:
            with self._lock:
                found = self.engine.orbitals(structure.engine_copy())
        except Exception as e:
            raise CalculatorError(f"{self.name}: {e}") from e
        if found is None:
            return None
        overlap = np.asarray(found["overlap"], dtype=np.float64)
        n = overlap.shape[0]
        sets = []
        for s in found["sets"]:
            c = np.asarray(s["coefficients"], dtype=np.float64)
            energies = np.asarray(s["energies"], dtype=np.float64).ravel()
            occupations = np.asarray(s["occupations"], dtype=np.float64).ravel()
            if overlap.shape != (n, n) or c.ndim != 2 or c.shape[0] != n:
                raise CalculatorError(
                    f"{self.name}: orbitals of shape {c.shape} for an overlap of {overlap.shape}"
                )
            if energies.size != c.shape[1] or occupations.size != c.shape[1]:
                raise CalculatorError(
                    f"{self.name}: {c.shape[1]} orbitals with {energies.size} energies and "
                    f"{occupations.size} occupations"
                )
            sets.append({"coefficients": c, "energies": energies, "occupations": occupations})
        if len(sets) not in (1, 2):
            raise CalculatorError(f"{self.name}: {len(sets)} sets of orbitals (one or two)")
        out: dict[str, Any] = {"overlap": overlap, "sets": sets}
        if found.get("atom_of_ao") is not None:
            atoms = [int(a) for a in np.asarray(found["atom_of_ao"]).ravel()]
            if len(atoms) != n or any(a < 0 or a >= len(structure.numbers) for a in atoms):
                raise CalculatorError(
                    f"{self.name}: atoms of the atomic orbitals for {len(atoms)} atomic orbitals "
                    f"of {len(structure.numbers)} atoms ({n} atomic orbitals)"
                )
            out["atom_of_ao"] = atoms
        return out

    def transitions(self, structure: Structure, count: int) -> list[dict[str, Any]] | None:
        """The transitions the engine hands out (see :meth:`Engine.transitions`), with their
        sizes checked; ``None`` if it does not."""
        try:
            with self._lock:
                found = self.engine.transitions(structure.engine_copy(), count)
        except Exception as e:
            raise CalculatorError(f"{self.name}: {e}") from e
        if found is None:
            return None
        n = len(structure.symbols)
        out = []
        for k, t in enumerate(found):
            mu = np.asarray(t["transition_dipole"], dtype=np.float64).ravel()
            d = np.asarray(t["population_change"], dtype=np.float64).ravel()
            if mu.size != 3 or d.size != n:
                raise CalculatorError(
                    f"{self.name}: transition {k + 1} has {mu.size} dipole components and "
                    f"{d.size} population changes for {n} atoms"
                )
            one: dict[str, Any] = {
                "excitation": float(t["excitation"]),
                "transition_dipole": mu,
                "population_change": d,
            }
            if t.get("dipole_change") is not None:
                dm = np.asarray(t["dipole_change"], dtype=np.float64).ravel()
                if dm.size != 3:
                    raise CalculatorError(
                        f"{self.name}: transition {k + 1} has a dipole change of {dm.size} "
                        "components"
                    )
                one["dipole_change"] = dm
            if t.get("transition_population") is not None:
                tp = np.asarray(t["transition_population"], dtype=np.float64).ravel()
                if tp.size != n:
                    raise CalculatorError(
                        f"{self.name}: transition {k + 1} has {tp.size} transition populations "
                        f"for {n} atoms"
                    )
                one["transition_population"] = tp
            out.append(one)
        return out

    def state_characters(self, structure: Structure, count: int) -> dict[str, Any] | None:
        """The lowest ``count`` excited states of ``structure`` as the engine hands them out
        (see :meth:`Engine.state_characters`), as float arrays with their sizes checked;
        ``None`` if it does not."""
        try:
            with self._lock:
                found = self.engine.state_characters(structure.engine_copy(), count)
        except Exception as e:
            raise CalculatorError(f"{self.name}: {e}") from e
        if found is None:
            return None
        orbitals = np.asarray(found["orbitals"], dtype=np.float64)
        occupied = int(found["occupied"])
        energies = np.asarray(found["energies"], dtype=np.float64).ravel()
        amplitudes = np.asarray(found["amplitudes"], dtype=np.float64)
        n = orbitals.shape[0]
        if orbitals.ndim != 2 or orbitals.shape != (n, n) or not 0 < occupied < n:
            raise CalculatorError(
                f"{self.name}: orbitals of shape {orbitals.shape} with {occupied} occupied ones"
            )
        if amplitudes.shape != (occupied * (n - occupied), energies.size):
            raise CalculatorError(
                f"{self.name}: amplitudes of shape {amplitudes.shape} for {occupied} occupied "
                f"and {n - occupied} empty orbitals and {energies.size} states"
            )
        return {
            "orbitals": orbitals,
            "occupied": occupied,
            "energies": energies,
            "amplitudes": amplitudes,
        }

    def _remember(self, key: tuple[Any, ...], value: _Cached) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _frame_translation(self, before: FloatArray, after: FloatArray) -> FloatArray:
        """Cartesian derivatives survive translation; a rotated backend must return input axes."""
        if np.array_equal(before, after):
            return np.zeros(3)
        delta = after - before
        if not np.allclose(delta, delta[0], rtol=0.0, atol=1e-9):
            raise CalculatorError(f"{self.name}: backend changed the input axes or internal geometry")
        return delta[0]

    def _run(
        self, structure: Structure, hessian: bool, stress: bool = False, gradient: bool = True
    ) -> _Cached:
        n = structure.n_atoms
        model = structure.engine_copy()
        try:
            if not gradient:
                # Engines whose compute() takes ``gradient``: the energy alone.
                res = cast(Any, self.engine).compute(model, hessian, gradient=False)
                if res.gradient is None and res.forces is None:
                    res.gradient = np.zeros(3 * n)
                    res.extra[NO_GRADIENT] = True
            elif stress:
                # Engines whose compute() takes ``stress`` (see :class:`Engine`).
                res = cast(Any, self.engine).compute(model, hessian, stress=True)
                res.extra[STRESS_ASKED] = True
            else:
                res = self.engine.compute(model, hessian)
            self._frame_translation(structure.positions_bohr, model.positions_bohr)
        except CalculatorError:
            raise
        except Exception as e:
            raise CalculatorError(f"{self.name} failed for '{structure.label}': {e}") from e
        self.evaluations += 1
        return self._convert(structure, res)

    def _convert(self, structure: Structure, res: EngineResult) -> _Cached:
        n = structure.n_atoms
        units = self.engine.units
        e_factor = units.energy_to_hartree
        g_factor = e_factor / units.length_to_bohr
        if res.gradient is not None:
            g = np.asarray(res.gradient, dtype=np.float64)
        elif res.forces is not None:
            g = -np.asarray(res.forces, dtype=np.float64)
        else:
            raise CalculatorError(f"{self.name}: no gradient or forces returned")
        if g.size != 3 * n:
            raise CalculatorError(f"{self.name}: gradient has {g.size} values for {n} atoms")
        g = g.reshape(n, 3) * g_factor
        h = None
        if res.hessian is not None:
            h = np.asarray(res.hessian, dtype=np.float64)
            if h.size != 9 * n * n:
                raise CalculatorError(f"{self.name}: Hessian has {h.size} values for {n} atoms")
            h = h.reshape(3 * n, 3 * n) * (g_factor / units.length_to_bohr)
            h = 0.5 * (h + h.T)
        energy = float(res.energy) * e_factor
        if not (np.isfinite(energy) and np.all(np.isfinite(g))):
            raise CalculatorError(f"{self.name}: non-finite energy or gradient")
        if h is not None and not np.all(np.isfinite(h)):
            raise CalculatorError(f"{self.name}: non-finite Hessian")
        return _Cached(energy, g, h, dict(res.extra))

    def _finite_difference_hessian(self, structure: Structure) -> FloatArray:
        n3 = 3 * structure.n_atoms
        h = np.zeros((n3, n3))
        step = self.hessian_step_bohr
        for k in range(n3):
            gp = self._run(structure.displaced(k, step), False).gradient.ravel()
            gm = self._run(structure.displaced(k, -step), False).gradient.ravel()
            h[:, k] = (gp - gm) / (2.0 * step)
        return 0.5 * (h + h.T)

    def _o1numhess_hessian(self, structure: Structure, gradient: FloatArray) -> FloatArray:
        n3 = 3 * structure.n_atoms
        cell = None if structure.cell_bohr is None else structure.cell_bohr.ravel().tolist()
        rigid = self.engine.capabilities.rigid_body_invariant
        job = _core.O1NumHess(
            list(structure.symbols),
            structure.positions_bohr.ravel().tolist(),
            gradient.ravel().tolist(),
            cell_bohr=cell,
            translation=rigid,
            rotation=rigid and not structure.is_periodic,
        )
        while (points := job.ask()) is not None:
            gradients = []
            for x in points:
                moved = replace(
                    structure, positions_bohr=np.asarray(x, dtype=np.float64).reshape(-1, 3)
                )
                gradients.append(self._run(moved, False).gradient.ravel().tolist())
            job.tell(gradients)
        return np.asarray(job.hessian(), dtype=np.float64).reshape(n3, n3)

    def last_extra(self) -> Mapping[str, Any]:
        """Extra results of the most recent evaluation."""
        if not self._cache:
            return {}
        return next(reversed(self._cache.values())).extra

    def close(self) -> None:
        self.engine.close()
