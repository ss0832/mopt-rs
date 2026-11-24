"""Engines that call other code: mopt-rs native potentials (Rust) and an installed MultiOptPy.

``multioptpy:<software_type>`` imports the MultiOptPy installed in the current environment at
run time and uses its ASE setup for that program (useful for programs without a mopt-rs
profile, such as the file-based ORCA and Gaussian wrappers). No MultiOptPy code is part of
mopt-rs.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterable, Mapping

import numpy as np

from mopt_rs import _core
from mopt_rs.calculators.base import (
    ASE_UNITS,
    ATOMIC_UNITS,
    Capabilities,
    Engine,
    EngineResult,
    Structure,
)

__all__ = ["MultiOptPyEngine", "NativeEngine"]


class NativeEngine(Engine):
    """A potential implemented in Rust (``lj``, ``emt``, ``eam``, ``meam``, ``tersoff``, ...)."""

    units = ATOMIC_UNITS

    def __init__(
        self,
        name: str,
        settings: Mapping[str, object] | Iterable[tuple[str, object]] | None = None,
    ) -> None:
        if name not in _core.native_calculators():
            raise KeyError(
                f"no native calculator '{name}' (known: {', '.join(_core.native_calculators())})"
            )
        self.name = name
        # One Rust calculator for all evaluations (GFN-FF keeps the topology of the first).
        if settings is None:
            pairs: list[tuple[str, str]] = []
        else:
            items = settings.items() if isinstance(settings, Mapping) else settings
            pairs = [(str(key), str(value)) for key, value in items]
        self._settings = pairs
        self._evaluator = _core.NativeEvaluator(name, pairs)
        self.capabilities = Capabilities(
            analytic_hessian=True,
            pbc=name not in ("gfn0xtb", "gfn1xtb"),
            excited_states=name in ("gfn0xtb", "gfn1xtb"),
            electrons=name in ("gfn0xtb", "gfn1xtb"),
            rigid_body_invariant=name != "muller_brown",
        )

    def describe(self) -> str:
        from mopt_rs import __version__

        return f"{self.name}: built into mopt-rs {__version__} (Rust)"

    def orbitals(self, structure: Structure) -> dict[str, object] | None:
        if structure.is_periodic:
            return None
        found = _core.native_orbitals(
            self.name,
            list(structure.symbols),
            structure.positions_bohr.ravel().tolist(),
            structure.state.charge,
            structure.state.multiplicity,
            None,
            self._settings,
        )
        if found is None:
            return None
        overlap, n, sets, atom_of_ao = found
        out: dict[str, object] = {
            "overlap": np.asarray(overlap).reshape(n, n),
            "sets": [
                {
                    "coefficients": np.asarray(c).reshape(n, m),
                    "energies": np.asarray(e),
                    "occupations": np.asarray(occ),
                }
                for c, m, e, occ in sets
            ],
        }
        if atom_of_ao is not None:
            out["atom_of_ao"] = list(atom_of_ao)
        return out

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        cell = None if structure.cell_bohr is None else structure.cell_bohr.ravel().tolist()
        e, g, h = self._evaluator.evaluate(
            list(structure.symbols),
            structure.positions_bohr.ravel().tolist(),
            structure.state.charge,
            structure.state.multiplicity,
            hessian,
            cell,
            pbc=structure.pbc,
            excited_state=structure.excited_state,
        )
        notes = self._evaluator.take_warnings()
        for note in notes:
            warnings.warn(note, RuntimeWarning, stacklevel=2)
        return EngineResult(
            e,
            gradient=np.asarray(g),
            hessian=None if h is None else np.asarray(h),
            extra={"warnings": notes} if notes else {},
        )


class MultiOptPyEngine(Engine):
    """A program through the installed MultiOptPy's ASE setup (``setup_calculator``)."""

    units = ASE_UNITS
    program = "multioptpy"
    citations = (
        "ss0832, MultiOptPy, https://github.com/ss0832/MultiOptPy; cite the program it runs "
        "as its authors ask",
    )

    def __init__(
        self,
        software_type: str,
        *,
        software_paths: Mapping[str, str] | None = None,
        functional: str | None = None,
        basis_set: str | None = None,
        memory: str | None = None,
    ) -> None:
        try:
            module = importlib.import_module("multioptpy.Calculator.ase_calculation_tools")
        except ImportError as e:
            raise ImportError(
                f"multioptpy:{software_type} needs MultiOptPy installed in this environment ({e})"
            ) from e
        self._setup = module.setup_calculator
        self.software_type = software_type
        self.paths = dict(software_paths or {})
        self.functional = functional
        self.basis_set = basis_set
        self.memory = memory
        self.name = f"multioptpy:{software_type}"
        if self.paths:
            self.notes = (f"software_path.conf entries: {', '.join(sorted(self.paths))}",)
        # Hessian units differ between MultiOptPy's wrappers: finite differences.
        self.capabilities = Capabilities(analytic_hessian=False, pbc=True)

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        from ase import Atoms

        atoms = Atoms(
            list(structure.symbols),
            positions=structure.positions_angstrom,
            cell=structure.cell_angstrom if structure.cell_bohr is not None else np.zeros((3, 3)),
            pbc=structure.pbc,
        )
        wrapper = self._setup(
            atoms,
            self.software_type,
            [structure.state.charge, structure.state.multiplicity],
            software_path_dict=self.paths,
            functional=self.functional,
            basis_set=self.basis_set,
            set_memory=self.memory,
        )
        result = wrapper.run()
        atoms = result if isinstance(result, Atoms) else getattr(wrapper, "atom_obj", atoms)
        energy = float(atoms.get_potential_energy(apply_constraint=False))
        forces = np.asarray(atoms.get_forces(apply_constraint=False))
        return EngineResult(energy, forces=forces)
