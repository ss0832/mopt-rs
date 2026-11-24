"""The OpenMM system of a model system of ONIOM, cut out of the system of the whole
structure.

As a lower level of ONIOM a force field is asked for the whole structure and for every
model system: a region of atoms with a hydrogen (a link atom) in the place of every bond
that the region cuts. The force field of the whole structure has all it takes to describe
the region, so :func:`model_system` makes the system of the model from it, and nothing has
to be written by hand:

- the atoms of the region keep their masses, charges, and Lennard-Jones parameters;
- a term of the force field (a bond, an angle, a torsion, an exception of the non-bonded
  forces) is kept when all its atoms are atoms of the region, or when its one atom outside
  is the outer atom of a cut bond and stands for the link atom of that bond (the term then
  acts on the link atom);
- a link atom has no charge and the Lennard-Jones parameters of a hydrogen on carbon
  (:data:`LINK_SIGMA_NM`, :data:`LINK_EPSILON_KJ`). The bond to it has the force constant of
  the cut bond and its length scaled as the link atom is placed (``g`` times the length for
  link atoms at ``g`` times the cut bond, the fixed distance otherwise), so that the term
  follows the cut bond. Its 1-4 interactions keep the scale of those of the outer atom.

The non-bonded forces of the model are those of a molecule (no cutoff, no cell), whatever
those of the whole structure are. What cancels in the ONIOM energy cancels however the
link atoms are described: ``E = E_low(whole) + E_high(model) - E_low(model)``; the
description above is a choice for the terms at the border, not a parametrization of a
hydrogen.

Force types that are handled: ``HarmonicBondForce``, ``HarmonicAngleForce``,
``PeriodicTorsionForce``, ``RBTorsionForce``, ``CMAPTorsionForce``, ``NonbondedForce``;
``CMMotionRemover`` and barostats are left out; any other force is refused by its name.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import Any

from mopt_rs.calculators.base import BOHR_TO_ANGSTROM, CalculatorError

__all__ = ["LINK_EPSILON_KJ", "LINK_SIGMA_NM", "model_system"]

LINK_SIGMA_NM = 0.2649533
"""Lennard-Jones diameter of a link atom (nm): the hydrogen on an aliphatic carbon of the
Amber force fields (type HC)."""
LINK_EPSILON_KJ = 0.0656888
"""Lennard-Jones well depth of a link atom (kJ/mol), of the same type."""
LINK_MASS = 1.008

_LEFT_OUT = ("CMMotionRemover", "MonteCarloBarostat", "MonteCarloAnisotropicBarostat")


class _Map:
    """The atoms of the whole structure as atoms of the model system."""

    def __init__(self, atoms: Sequence[int], links: Sequence[tuple[int, int, float]]) -> None:
        self.index = {atom: k for k, atom in enumerate(atoms)}
        if len(self.index) != len(atoms):
            raise CalculatorError("openmm: an atom is twice in a model system")
        # Outer atom -> its link atoms as (number in the model, inner atom, factor).
        self.links: dict[int, list[tuple[int, int, float]]] = {}
        for k, (inner, outer, factor) in enumerate(links):
            if inner not in self.index or outer in self.index:
                raise CalculatorError(
                    f"openmm: the link {inner + 1}-{outer + 1} does not lead out of the model "
                    "system"
                )
            self.links.setdefault(outer, []).append((len(atoms) + k, inner, float(factor)))

    def link(self, outer: int, others: Sequence[int]) -> tuple[int, int, float] | None:
        """The link atom that `outer` stands for in a term with the atoms `others`: the
        one whose inner atom is in the term; in a pair (an exception of the non-bonded
        forces between atoms two or three bonds apart) also the only one there is."""
        found = self.links.get(outer, [])
        with_inner = [entry for entry in found if entry[1] in others]
        if len(with_inner) == 1:
            return with_inner[0]
        return found[0] if len(found) == 1 and len(others) == 2 else None

    def term(self, particles: Sequence[int]) -> list[int] | None:
        """The atoms of a term in the model system, or ``None`` when the term is not one
        of the model."""
        out = []
        for p in particles:
            if p in self.index:
                out.append(self.index[p])
                continue
            entry = self.link(p, particles)
            if entry is None:
                return None
            out.append(entry[0])
        return out if len(set(out)) == len(out) else None


def _is_link(index: int, atoms: int) -> bool:
    return index >= atoms


def model_system(
    system: Any,
    atoms: Sequence[int],
    links: Sequence[tuple[int, int, float]] = (),
    placement: str = "scaled",
) -> Any:
    """The system of the model made of the atoms ``atoms`` of ``system`` (numbers from 0, in
    the order of the model) and one link atom for each of ``links`` after them: ``(inner
    atom, outer atom, factor)``, with ``factor`` the ``g`` of link atoms at ``g`` times the
    cut bond (``placement="scaled"``) or their distance from the inner atom in Bohr
    (``placement="distance"``). See the module."""
    mm = importlib.import_module("openmm")
    unit = importlib.import_module("openmm.unit")
    if placement not in ("scaled", "distance"):
        raise CalculatorError(
            f"openmm: link atoms are 'scaled' or at a 'distance', not '{placement}'"
        )
    count = system.getNumParticles()
    if any(a < 0 or a >= count for a in atoms):
        raise CalculatorError(
            f"openmm: a model system with atoms beyond the {count} of the system it is cut from"
        )
    n = len(atoms)
    mapping = _Map(atoms, links)
    model = mm.System()
    for atom in atoms:
        model.addParticle(system.getParticleMass(atom))
    for _ in links:
        model.addParticle(LINK_MASS * unit.dalton)
    for force in system.getForces():
        kind = type(force).__name__
        if kind in _LEFT_OUT:
            continue
        if kind == "HarmonicBondForce":
            new = mm.HarmonicBondForce()
            for k in range(force.getNumBonds()):
                p1, p2, length, constant = force.getBondParameters(k)
                found = mapping.term((p1, p2))
                if found is None:
                    continue
                for particle in (p1, p2):
                    entry = (
                        mapping.link(particle, (p1, p2)) if particle not in mapping.index else None
                    )
                    if entry is None:
                        continue
                    if placement == "scaled":
                        length = length * entry[2]
                    else:
                        length = entry[2] * BOHR_TO_ANGSTROM * 0.1 * unit.nanometer
                new.addBond(found[0], found[1], length, constant)
        elif kind == "HarmonicAngleForce":
            new = mm.HarmonicAngleForce()
            for k in range(force.getNumAngles()):
                *particles, angle, constant = force.getAngleParameters(k)
                found = mapping.term(particles)
                if found is not None:
                    new.addAngle(*found, angle, constant)
        elif kind == "PeriodicTorsionForce":
            new = mm.PeriodicTorsionForce()
            for k in range(force.getNumTorsions()):
                *particles, periodicity, phase, constant = force.getTorsionParameters(k)
                found = mapping.term(particles)
                if found is not None:
                    new.addTorsion(*found, periodicity, phase, constant)
        elif kind == "RBTorsionForce":
            new = mm.RBTorsionForce()
            for k in range(force.getNumTorsions()):
                parameters = force.getTorsionParameters(k)
                found = mapping.term(parameters[:4])
                if found is not None:
                    new.addTorsion(*found, *parameters[4:])
        elif kind == "CMAPTorsionForce":
            new = mm.CMAPTorsionForce()
            for k in range(force.getNumMaps()):
                size, energy = force.getMapParameters(k)
                new.addMap(size, energy)
            for k in range(force.getNumTorsions()):
                which, *particles = force.getTorsionParameters(k)
                first, second = mapping.term(particles[:4]), mapping.term(particles[4:])
                if first is not None and second is not None:
                    new.addTorsion(which, *first, *second)
        elif kind == "NonbondedForce":
            new = _nonbonded(mm, unit, force, atoms, len(links), mapping)
        else:
            raise CalculatorError(
                f"openmm: a model system cannot be cut out of a system with a {kind} "
                "(handled: HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, "
                "RBTorsionForce, CMAPTorsionForce, NonbondedForce); give the system of the "
                "model system as a file (system=WHOLE.xml,MODEL.xml)"
            )
        new.setForceGroup(force.getForceGroup())
        model.addForce(new)
    assert model.getNumParticles() == n + len(links)
    return model


def _nonbonded(
    mm: Any, unit: Any, force: Any, atoms: Sequence[int], links: int, mapping: _Map
) -> Any:
    """The non-bonded forces of the model: those of a molecule, with the exceptions of the
    whole structure among its atoms."""
    n = len(atoms)
    new = mm.NonbondedForce()
    new.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    wells = []
    for atom in atoms:
        charge, sigma, epsilon = force.getParticleParameters(atom)
        new.addParticle(charge, sigma, epsilon)
        wells.append(float(epsilon.value_in_unit(unit.kilojoule_per_mole)))
    for _ in range(links):
        new.addParticle(
            0.0, LINK_SIGMA_NM * unit.nanometer, LINK_EPSILON_KJ * unit.kilojoule_per_mole
        )
        wells.append(LINK_EPSILON_KJ)
    for k in range(force.getNumExceptions()):
        p1, p2, product, sigma, epsilon = force.getExceptionParameters(k)
        found = mapping.term((p1, p2))
        if found is None:
            continue
        i, j = found
        if not (_is_link(i, n) or _is_link(j, n)):
            new.addException(i, j, product, sigma, epsilon)
            continue
        # A pair with a link atom: no charge, and the Lennard-Jones well of the two atoms
        # scaled as that of the pair of the whole structure was (nothing for an exclusion).
        depth = float(epsilon.value_in_unit(unit.kilojoule_per_mole))
        full = math.sqrt(
            float(force.getParticleParameters(p1)[2].value_in_unit(unit.kilojoule_per_mole))
            * float(force.getParticleParameters(p2)[2].value_in_unit(unit.kilojoule_per_mole))
        )
        scale = depth / full if full > 0.0 else (0.0 if depth == 0.0 else 0.5)
        sigma_i = float(new.getParticleParameters(i)[1].value_in_unit(unit.nanometer))
        sigma_j = float(new.getParticleParameters(j)[1].value_in_unit(unit.nanometer))
        new.addException(
            i,
            j,
            0.0,
            0.5 * (sigma_i + sigma_j) * unit.nanometer,
            scale * math.sqrt(wells[i] * wells[j]) * unit.kilojoule_per_mole,
        )
    return new
