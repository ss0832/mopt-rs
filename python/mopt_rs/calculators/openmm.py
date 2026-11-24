"""OpenMM force fields as an engine (``-os openmm``; optional: OpenMM is not a dependency).

An OpenMM ``System`` belongs to one set of atoms, so the engine holds one system per set it is
asked to compute and picks the one that fits a structure:

- ``system=FILE.xml[,FILE2.xml...]``: serialized systems (``XmlSerializer``); a structure
  takes the one with its number of particles and, where masses are set, its elements;
- ``forcefield=amber14-all.xml[,...]`` with ``topology=FILE.pdb[,FILE2.pdb...]``: the force
  field of OpenMM's ``app.ForceField`` on the topologies of PDB files, whose atoms are in the
  order of the structures (their elements are compared).

As a lower level of ONIOM the engine is asked for the whole structure and for every model
system (a region with its link hydrogens). The system or the topology of the whole
structure is enough (``-oniom_opt low.topology=whole.pdb low.forcefield=...``): the
command tells the engine which atoms a model system is made of and which bonds it cuts,
and the system of the model is cut out of that of the whole structure
(:mod:`mopt_rs.calculators.openmm_model`). A system or a topology given for the model
system (``low.topology=whole.pdb,model.pdb``) is used in its place.

Bonds are flexible (no constraints, no rigid water): the optimizer moves every atom.
Energies are in kJ/mol and positions in nanometres inside OpenMM; the engine hands out
kJ/mol and Angstrom. In a periodic cell the lattice vectors have to be in OpenMM's reduced
form (``a`` along x, ``b`` in the xy plane); other cells are refused with what is wrong.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mopt_rs.calculators.base import (
    BOHR_TO_ANGSTROM,
    CalculatorError,
    Capabilities,
    Engine,
    EngineResult,
    EngineUnits,
    Structure,
)
from mopt_rs.calculators.program_options import ProgramSetup

__all__ = ["OpenMMEngine"]

_NM_PER_ANGSTROM = 0.1
#: The methods of ``NonbondedForce`` by the names of ``openmm.app``.
_NONBONDED = ("NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "Ewald", "PME", "LJPME")
#: Masses (amu) agree with an element within this.
_MASS_TOLERANCE = 0.2


def _openmm() -> Any:
    try:
        return importlib.import_module("openmm")
    except ImportError as e:
        raise CalculatorError(
            "the calculator 'openmm' needs the OpenMM package (conda install -c conda-forge openmm)"
        ) from e


def _paths(value: Any) -> list[Any]:
    """A setting that names one file or several (comma-separated), or objects as given."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [p.strip() for p in str(value).split(",") if p.strip()]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


@dataclass
class _Entry:
    """One system with what identifies the structures it belongs to."""

    system: Any
    symbols: tuple[str, ...] | None
    source: str
    context: Any = None
    masses: list[float] = field(default_factory=list)


class OpenMMEngine(Engine):
    """Energies and forces of OpenMM systems.

    ``system``: serialized systems (file names) or ``openmm.System`` objects;
    ``forcefield`` and ``topology``: the files of a force field and PDB files (or
    ``openmm.app.Topology`` objects) to build systems from; ``nonbonded``: the method of the
    non-bonded forces of such systems (``NoCutoff`` without a cell, ``PME`` in one, unless
    given); ``cutoff``: its cutoff in nanometres; ``platform`` and ``properties``: the OpenMM
    platform (``CUDA``, ``OpenCL``, ``Reference``, ``CPU``) and its properties; ``create``:
    further keywords of ``ForceField.createSystem``.

    Without ``platform`` the engine takes one that computes in double precision: CUDA or
    OpenCL with ``Precision=double`` where there is one, the Reference platform otherwise.
    The CPU platform works in single precision (energies to about seven digits), which is
    not enough for the energy differences of a converging optimization of a large system,
    nor for differences of its forces (on an enzyme of 5696 atoms a Hessian-vector product
    by forward differences along 1e-4 Bohr is off by 1.6 percent, against 0.15 percent on
    Reference, where it takes ten times as long: 54 against 5.6 ms per evaluation); it is
    used when asked for, and the run log says so.
    """

    name = "openmm"
    units = EngineUnits("kj/mol", "angstrom")
    program = "openmm"
    citations = (
        "P. Eastman, R. Galvelis, R. P. Pelaez, C. R. A. Abreu, S. E. Farr, E. Gallicchio, "
        "A. Gorenko, M. M. Henry, F. Hu, J. Huang, A. Kramer, J. Michel, J. A. Mitchell, "
        "V. S. Pande, J. P. Rodrigues, J. Rodriguez-Guerra, A. C. Simmonett, S. Singh, "
        "J. Swails, P. Turner, Y. Wang, I. Zhang, J. D. Chodera, G. De Fabritiis, "
        "T. E. Markland, J. Phys. Chem. B 2024, 128, 109 (OpenMM 8)",
    )
    capabilities = Capabilities(analytic_hessian=False, pbc=True, electrons=False)

    def __init__(
        self,
        *,
        system: Any = None,
        forcefield: Any = None,
        topology: Any = None,
        nonbonded: str | None = None,
        cutoff: float = 1.0,
        platform: str | None = None,
        properties: dict[str, str] | None = None,
        create: dict[str, Any] | None = None,
        setup: ProgramSetup | None = None,
        model_atoms: Sequence[int] | None = None,
        model_links: Sequence[tuple[int, int, float]] = (),
        model_link_placement: str = "scaled",
    ) -> None:
        self.model_atoms = None if model_atoms is None else [int(a) for a in model_atoms]
        """The atoms of the whole structure (from 0) that the structures of this engine are
        made of, when it computes a model system of ONIOM."""
        self.model_links = [(int(i), int(o), float(f)) for i, o, f in model_links]
        self.model_link_placement = model_link_placement
        if nonbonded is not None and nonbonded not in _NONBONDED:
            raise CalculatorError(
                f"openmm: nonbonded is one of {', '.join(_NONBONDED)}, got '{nonbonded}'"
            )
        self.nonbonded = nonbonded
        self.cutoff_nm = float(cutoff)
        self.platform = platform
        self.properties = dict(properties or {})
        if platform == "CPU" or self.properties.get("Precision", "double").lower() != "double":
            # Measured on an enzyme of 5696 atoms: a Hessian-vector product by forward
            # differences of the forces along a step of 1e-4 Bohr is off by 1.6 percent on the
            # CPU platform and by 0.15 percent (the error of the differences) on Reference.
            self.notes = (
                *self.notes,
                "single or mixed precision as asked: differences of its gradients (numerical "
                "Hessians, the Hessian of the core with ONIOM microiterations) carry the "
                "rounding of its forces; without a platform the engine computes in double "
                "precision",
            )
        self.create = dict(create or {})
        self.setup = setup or ProgramSetup()
        self._forcefield_files = [str(f) for f in _paths(forcefield)]
        self._topologies = _paths(topology)
        self._entries: list[_Entry] = []
        for item in _paths(system):
            self._entries.append(self._serialized(item))
        if self._topologies and not self._forcefield_files:
            raise CalculatorError("openmm: topology needs forcefield (the files of a force field)")
        if self._forcefield_files and not self._topologies:
            raise CalculatorError(
                "openmm: forcefield needs topology (PDB files with the atoms in the order of "
                "the structures)"
            )
        if not self._entries and not self._topologies:
            raise CalculatorError(
                "openmm: give system=FILE.xml (serialized systems) or forcefield=... with "
                "topology=FILE.pdb"
            )

    # -- systems ---------------------------------------------------------------------------

    def _serialized(self, item: Any) -> _Entry:
        """A system from a file of ``XmlSerializer``, or one given as an object."""
        mm = _openmm()
        if isinstance(item, (str, Path)):
            text = Path(item).read_text(encoding="utf-8")
            system = mm.XmlSerializer.deserialize(text)
            source = str(item)
        else:
            system, source = item, "a System object"
        if not hasattr(system, "getNumParticles"):
            raise CalculatorError(f"openmm: {source} is not an OpenMM System")
        return _Entry(system=system, symbols=None, source=source, masses=self._masses(system))

    @staticmethod
    def _masses(system: Any) -> list[float]:
        unit = importlib.import_module("openmm.unit")
        return [
            float(system.getParticleMass(i).value_in_unit(unit.dalton))
            for i in range(system.getNumParticles())
        ]

    def _built(self, periodic: bool) -> list[_Entry]:
        """The systems of the force field on the topologies (built once, for molecules or
        for a cell as the first structure asks)."""
        app = importlib.import_module("openmm.app")
        unit = importlib.import_module("openmm.unit")
        field_ = app.ForceField(*self._forcefield_files)
        method = self.nonbonded or ("PME" if periodic else "NoCutoff")
        keywords: dict[str, Any] = {
            "nonbondedMethod": getattr(app, method),
            "constraints": None,
            "rigidWater": False,
            "removeCMMotion": False,
            **self.create,
        }
        if method != "NoCutoff":
            keywords["nonbondedCutoff"] = self.cutoff_nm * unit.nanometer
        entries = []
        for item in self._topologies:
            if isinstance(item, (str, Path)):
                topology, source = app.PDBFile(str(item)).topology, str(item)
            else:
                topology, source = item, "a Topology object"
            symbols = tuple(
                atom.element.symbol if atom.element is not None else "X"
                for atom in topology.atoms()
            )
            system = field_.createSystem(topology, **keywords)
            entries.append(
                _Entry(system=system, symbols=symbols, source=source, masses=self._masses(system))
            )
        return entries

    def _entry(self, structure: Structure) -> _Entry:
        """The system that belongs to ``structure``."""
        if self._topologies:
            self._entries.extend(self._built(structure.is_periodic))
            self._topologies = []
        elements = importlib.import_module("openmm.app").element
        dalton = importlib.import_module("openmm.unit").dalton
        for entry in self._entries:
            if len(entry.masses) != structure.n_atoms:
                continue
            if entry.symbols is not None:
                if tuple(s.capitalize() for s in entry.symbols) == tuple(
                    s.capitalize() for s in structure.symbols
                ):
                    return entry
                continue
            # A serialized system knows masses only: every atom with a mass has to have that
            # of its element (massless particles are fixed atoms or virtual sites).
            fits = True
            for symbol, mass in zip(structure.symbols, entry.masses, strict=True):
                if mass <= 0.0:
                    continue
                try:
                    element = elements.Element.getBySymbol(symbol.capitalize())
                except KeyError:
                    fits = False
                    break
                expected = float(element.mass.value_in_unit(dalton))
                if abs(expected - mass) > _MASS_TOLERANCE:
                    fits = False
                    break
            if fits:
                return entry
        cut = self._cut_out(structure)
        if cut is not None:
            return cut
        known = "; ".join(
            f"{e.source} ({len(e.masses)} atoms"
            + (
                f": {''.join(e.symbols[:12])}{'...' if len(e.symbols) > 12 else ''})"
                if e.symbols
                else ")"
            )
            for e in self._entries
        )
        raise CalculatorError(
            f"openmm: no system for the {structure.n_atoms} atoms "
            f"{''.join(structure.symbols[:12])}{'...' if structure.n_atoms > 12 else ''} "
            f"(systems: {known}). Every set of atoms the calculator is asked to compute needs "
            "its own system or topology, in the order of the atoms of the structure; as a "
            "lower level of ONIOM that is the whole structure and every model system with its "
            "link hydrogens."
        )

    def _cut_out(self, structure: Structure) -> _Entry | None:
        """The system of a model system of ONIOM, cut out of the system of the whole
        structure (made once); ``None`` when the engine was not told of a model system or
        ``structure`` is not one."""
        if self.model_atoms is None:
            return None
        if structure.n_atoms != len(self.model_atoms) + len(self.model_links):
            return None
        from mopt_rs.calculators.openmm_model import model_system

        highest = max(self.model_atoms, default=-1)
        whole = [e for e in self._entries if len(e.masses) > highest]
        if not whole:
            raise CalculatorError(
                f"openmm: the model system has atom {highest + 1} of the whole structure, "
                "and no system that was given has that many atoms"
            )
        # The largest one is the whole structure.
        source = max(whole, key=lambda e: len(e.masses))
        system = model_system(
            source.system, self.model_atoms, self.model_links, self.model_link_placement
        )
        entry = _Entry(
            system=system,
            symbols=tuple(structure.symbols),
            source=f"the model system of {len(self.model_atoms)} atoms and "
            f"{len(self.model_links)} link atom(s) cut out of {source.source}",
            masses=self._masses(system),
        )
        self._entries.append(entry)
        self.notes = (*self.notes, entry.source)
        return entry

    def _platform(self) -> tuple[Any, dict[str, str]]:
        """The platform and its properties: the one asked for, or one in double precision."""
        mm = _openmm()
        names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
        if self.platform is not None:
            if self.platform not in names:
                raise CalculatorError(
                    f"openmm: no platform '{self.platform}' (there are: {', '.join(names)})"
                )
            return mm.Platform.getPlatformByName(self.platform), dict(self.properties)
        for name in ("CUDA", "OpenCL"):
            if name in names:
                properties = {"Precision": "double", **self.properties}
                return mm.Platform.getPlatformByName(name), properties
        return mm.Platform.getPlatformByName("Reference"), dict(self.properties)

    def _context(self, entry: _Entry) -> Any:
        if entry.context is None:
            mm = _openmm()
            unit = importlib.import_module("openmm.unit")
            integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
            platform, properties = self._platform()
            try:
                context = mm.Context(entry.system, integrator, platform, properties)
            except Exception as e:
                if self.platform is not None:
                    raise CalculatorError(
                        f"openmm: the platform {platform.getName()} cannot be used: {e}"
                    ) from e
                # A platform that is there but does not start (no device): the reference one.
                integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
                context = mm.Context(
                    entry.system, integrator, mm.Platform.getPlatformByName("Reference")
                )
            entry.context = self.setup.run(context)
            self.notes = (
                *[n for n in self.notes if not n.startswith("platform ")],
                f"platform {entry.context.getPlatform().getName()}",
            )
        return entry.context

    # -- evaluation ------------------------------------------------------------------------

    @staticmethod
    def _box(structure: Structure) -> Any:
        """The lattice vectors in nanometres, in OpenMM's reduced form or refused."""
        cell = np.asarray(structure.cell_angstrom, dtype=np.float64) * _NM_PER_ANGSTROM
        a, b, c = cell
        upper = max(abs(a[1]), abs(a[2]), abs(b[2]))
        if upper > 1.0e-8 * float(np.abs(cell).max()):
            raise CalculatorError(
                "openmm: the first lattice vector has to lie along x and the second in the xy "
                "plane (a lower triangular cell); turn the structure and its cell accordingly"
            )
        if not all(structure.pbc):
            raise CalculatorError("openmm: a cell is periodic in all three directions")
        return cell

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        unit = importlib.import_module("openmm.unit")
        entry = self._entry(structure)
        context = self._context(entry)
        if structure.is_periodic:
            box = self._box(structure)
            if not entry.system.usesPeriodicBoundaryConditions():
                raise CalculatorError(
                    f"openmm: the structure has a cell, the system of {entry.source} is not "
                    "periodic"
                )
            context.setPeriodicBoxVectors(*(row * unit.nanometer for row in box))
        positions = np.asarray(structure.positions_bohr, dtype=np.float64) * (
            BOHR_TO_ANGSTROM * _NM_PER_ANGSTROM
        )
        context.setPositions(positions * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        forces = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
            dtype=np.float64,
        )
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise CalculatorError("openmm: the energy or the forces are not finite")
        # kJ/mol/nm -> kJ/mol/Angstrom.
        return EngineResult(energy=energy, forces=forces * _NM_PER_ANGSTROM)

    def close(self) -> None:
        for entry in self._entries:
            entry.context = None
