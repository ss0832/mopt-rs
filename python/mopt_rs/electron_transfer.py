"""Electron-transfer rate constants and their parts.

``run_etmain`` runs the procedure of ``etmain`` (EDEEL's diabatic states on the fragments of
one structure, the crossings along the solvent coordinate of SAES, the barriers, and the
Marcus and Zhu-Nakamura rate constants; ``docs/electron_transfer.md``). The other functions
are its parts one by one, in atomic units (Hartree, Bohr, e Bohr; mass-weighted gradients in
Hartree / (Bohr amu^(1/2))):

* ``inner_sphere``: the harmonic inner sphere at a crossing from the gradients of the two
  states (thesis of A. Mutsuji, Hokkaido University 2025, eqn (4.S4)-(4.S9));
* ``scher_holstein``: the nuclear tunnelling factor, eqn (4.12)-(4.16);
* ``marcus_rate`` and ``zhu_nakamura_rate``: the rate constants at a crossing;
* ``marcus_levich_jortner``: the rate constant of one quantum mode in a classical solvent;
* ``mulliken_hush``, ``generalized_mulliken_hush``, ``fragment_charge_difference``,
  ``charge_transfer``: couplings from the excited states and the charge-transfer index;
* ``solvent_shifts``: the solvent activation of SAES;
* ``apparent_barrier``: the free energy of activation of a rate constant by the Eyring
  equation;
* ``projection_coupling``, ``transfer_integral``, ``fock_from_orbitals``: the transfer
  integral of the projection method (Valeev et al. 2006) between the frontier orbitals of two
  molecules, from a calculator or from matrices;
* ``four_point_reorganization`` and ``hopping_rate``: the reorganization energy of the
  four-point scheme (Nelsen et al. 1987) and the Marcus rate of a hop;
* ``proton_states`` and ``pcet_rate``: the parts of a proton-coupled electron transfer, the
  vibrational states of the proton on a grid and the nonadiabatic vibronic rate.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _core

__all__ = [
    "run_etmain",
    "inner_sphere",
    "scher_holstein",
    "marcus_rate",
    "zhu_nakamura_rate",
    "marcus_levich_jortner",
    "mulliken_hush",
    "generalized_mulliken_hush",
    "fragment_charge_difference",
    "charge_transfer",
    "solvent_shifts",
    "apparent_barrier",
    "transfer_integral",
    "fock_from_orbitals",
    "projection_coupling",
    "four_point_reorganization",
    "hopping_rate",
    "proton_states",
    "pcet_rate",
    "PROTON_MASS_AMU",
    "DEUTERON_MASS_AMU",
]

PROTON_MASS_AMU = 1.007276466621
"""The mass of the proton (atomic mass units)."""

DEUTERON_MASS_AMU = 2.013553212745
"""The mass of the deuteron (atomic mass units)."""


def run_etmain(args: Sequence[str], console: bool = True) -> list[dict[str, Any]]:
    """Run ``etmain`` with ``args`` (as on the command line) and return, for every input, its
    directory, its crossings (barriers, couplings, rate constants), and the largest rate
    constants along the scan; with ``-et_method hopping`` the transfer integral, the
    four-point reorganization energies, the reaction energy, and the rates of the hop both
    ways (``method`` says which)."""
    return _core.run_etmain([str(a) for a in args], console)


def inner_sphere(
    g_initial: Sequence[float], g_final: Sequence[float], barrier: float
) -> dict[str, Any]:
    """The inner sphere at a crossing: ``quantum``, ``wavenumber_cm1``, ``reaction_energy``,
    ``reorganization``, ``q_crossing``, ``q_final``, ``slopes``."""
    return _core.et_inner_sphere(
        [float(v) for v in g_initial], [float(v) for v in g_final], float(barrier)
    )


def scher_holstein(
    quantum: float, reaction_energy: float, reorganization: float, temperature: float = 298.15
) -> tuple[float, float, float]:
    """``(ln Gamma, dG(T), dG(inf))`` of the tunnelling factor of Scher and Holstein."""
    return _core.et_scher_holstein(
        float(quantum), float(reaction_energy), float(reorganization), float(temperature)
    )


def marcus_rate(
    ln_gamma: float,
    coupling: float,
    reorganization: float,
    barrier: float,
    temperature: float = 298.15,
) -> float:
    """The Marcus rate constant (1/s) with the tunnelling factor ``exp(ln_gamma)``."""
    return _core.et_marcus_rate(
        float(ln_gamma), float(coupling), float(reorganization), float(barrier), float(temperature)
    )


def zhu_nakamura_rate(
    g_initial: Sequence[float],
    g_final: Sequence[float],
    inner_barrier: float,
    coupling: float,
    barrier: float,
    temperature: float = 298.15,
) -> tuple[str, float, float, float]:
    """``(kind, ln kappa, share below the crossing, rate)`` of the Zhu-Nakamura rate constant
    at the crossing of the two gradients; ``kind`` is ``peaked`` or ``sloped``."""
    return _core.et_zhu_nakamura_rate(
        [float(v) for v in g_initial],
        [float(v) for v in g_final],
        float(inner_barrier),
        float(coupling),
        float(barrier),
        float(temperature),
    )


def marcus_levich_jortner(
    coupling: float,
    quantum: float,
    inner_reorganization: float,
    solvent_reorganization: float,
    reaction_energy: float,
    temperature: float = 298.15,
) -> float:
    """The Marcus-Levich-Jortner rate constant (1/s): one quantum mode ``quantum`` with the
    inner reorganization energy, a classical solvent, and the reaction free energy."""
    return _core.et_marcus_levich_jortner(
        float(coupling),
        float(quantum),
        float(inner_reorganization),
        float(solvent_reorganization),
        float(reaction_energy),
        float(temperature),
    )


def mulliken_hush(
    excitation: float, transition_dipole: Sequence[float], distance: float
) -> float:
    """The Mulliken-Hush coupling ``E_ex |mu| / r_ab``."""
    return _core.et_mulliken_hush(
        float(excitation), [float(v) for v in transition_dipole], float(distance)
    )


def generalized_mulliken_hush(
    excitation: float, transition_dipole: Sequence[float], dipole_difference: Sequence[float]
) -> float:
    """The generalized Mulliken-Hush coupling of Cave and Newton."""
    return _core.et_generalized_mulliken_hush(
        float(excitation),
        [float(v) for v in transition_dipole],
        [float(v) for v in dipole_difference],
    )


def fragment_charge_difference(
    excitation: float,
    transition_population: Sequence[float],
    population_change: Sequence[float],
    donor: Sequence[int],
    acceptor: Sequence[int],
) -> float:
    """The fragment-charge-difference coupling of Voityuk and Rosch from the populations of the
    transition density and the change of the populations on each atom, with the atoms
    (0-based) of the donor and of the acceptor."""
    return _core.et_fragment_charge_difference(
        float(excitation),
        [float(v) for v in transition_population],
        [float(v) for v in population_change],
        [int(a) for a in donor],
        [int(a) for a in acceptor],
    )


def charge_transfer(
    positions: Sequence[Sequence[float]], population_change: Sequence[float]
) -> tuple[float, float, list[float], list[float]]:
    """``(q_CT, d_CT, gained_at, lost_at)`` of the changes of the atomic populations."""
    return _core.et_charge_transfer(
        [[float(c) for c in p] for p in positions], [float(v) for v in population_change]
    )


def solvent_shifts(
    lambda_initial: float, lambda_final: float, reaction_energy: float, alpha: float
) -> tuple[float, float, float]:
    """``(G_SA^i, G_SA^f, x_s)``: the shifts of SAES at ``alpha`` and the solvent coordinate."""
    return _core.et_solvent_shifts(
        float(lambda_initial), float(lambda_final), float(reaction_energy), float(alpha)
    )


def apparent_barrier(rate: float, temperature: float = 298.15) -> float:
    """The free energy of activation (Hartree) of a unimolecular rate constant (1/s)."""
    return _core.et_apparent_barrier(float(rate), float(temperature))


def _columns(orbitals: Any) -> list[list[float]]:
    """A list of orbitals from one orbital (a vector) or several (the columns of a matrix)."""
    a = np.asarray(orbitals, dtype=np.float64)
    if a.ndim == 1:
        return [a.tolist()]
    return [a[:, k].tolist() for k in range(a.shape[1])]


def transfer_integral(fock: Any, overlap: Any, a: Any, b: Any) -> dict[str, Any]:
    """The transfer integral (Hartree) of the orbitals ``a`` and ``b`` in the atomic orbitals
    of a pair with its Fock matrix and overlap. ``a`` and ``b`` are one orbital each, or the
    columns of a degenerate set, whose best-matched combinations are taken. A dict of
    ``coupling``, ``site_energies``, ``overlap``, ``effective`` (``J_eff``),
    ``effective_site_energies``, and ``degeneracy``."""
    return _core.et_transfer_integral(
        np.asarray(fock, dtype=np.float64).tolist(),
        np.asarray(overlap, dtype=np.float64).tolist(),
        _columns(a),
        _columns(b),
    )


def fock_from_orbitals(overlap: Any, coefficients: Any, energies: Sequence[float]) -> Any:
    """The Fock matrix ``S C diag(e) C^T S`` of the orbitals (a column each)."""
    return np.asarray(
        _core.et_fock_from_orbitals(
            np.asarray(overlap, dtype=np.float64).tolist(),
            np.asarray(coefficients, dtype=np.float64).tolist(),
            [float(e) for e in energies],
        )
    )


def _closed_shell(found: dict[str, Any] | None, what: str) -> tuple[Any, ...]:
    if found is None:
        raise ValueError(f"projection coupling: the calculator gives no orbitals of {what}")
    if found.get("atom_of_ao") is None:
        raise ValueError(
            f"projection coupling: the orbitals of {what} do not say which atom each atomic "
            "orbital is on (an engine adds 'atom_of_ao')"
        )
    if len(found["sets"]) != 1:
        raise ValueError(f"projection coupling: {what} is not a closed shell")
    one = found["sets"][0]
    return (
        np.asarray(found["overlap"], dtype=np.float64).tolist(),
        np.asarray(one["coefficients"], dtype=np.float64).tolist(),
        [float(e) for e in one["energies"]],
        [float(o) for o in one["occupations"]],
        [int(a) for a in found["atom_of_ao"]],
    )


def projection_coupling(
    calculator: Any,
    structure: Any,
    atoms_a: Sequence[int],
    atoms_b: Sequence[int],
    carrier: str = "hole",
    charges: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """The transfer integral of the projection method (Hartree) of a hole or an electron
    between the molecules ``atoms_a`` and ``atoms_b`` (0-based) of ``structure`` (a
    :class:`mopt_rs.calculators.base.Structure` of the neutral pair) with ``calculator`` (a
    calculator of :mod:`mopt_rs.calculators` whose orbitals say which atom each atomic orbital
    is on: the built-in GFN0-xTB and GFN1-xTB, PySCF). Each molecule is computed alone at its
    place in the pair, in the charge of ``charges`` (closed shells). A dict as
    :func:`transfer_integral`'s."""
    from .calculators.base import Structure

    def part(atoms: Sequence[int], charge: int) -> Any:
        idx = [int(a) for a in atoms]
        return Structure.build(
            [structure.symbols[k] for k in idx],
            structure.positions_bohr[idx],
            charge=charge,
        )

    return _core.et_projection_coupling(
        _closed_shell(calculator.orbitals(structure), "the pair"),
        _closed_shell(calculator.orbitals(part(atoms_a, charges[0])), "the first molecule"),
        _closed_shell(calculator.orbitals(part(atoms_b, charges[1])), "the second molecule"),
        [int(a) for a in atoms_a],
        [int(a) for a in atoms_b],
        carrier,
    )


def four_point_reorganization(a: Sequence[float], b: Sequence[float]) -> float:
    """The four-point reorganization energy (Hartree) of a hop from ``a`` to ``b``: for each
    molecule its energy in the charge it ends with, at the equilibrium of the charge it starts
    with and at its own, ``(at_start, at_own)``."""
    return _core.et_four_point_reorganization(
        (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
    )


def hopping_rate(
    coupling: float, reorganization: float, delta_g: float, temperature: float = 298.15
) -> float:
    """The Marcus rate (1/s) of a hop with the transfer integral, the reorganization energy,
    and the free energy of the hop (Hartree) at ``temperature`` (K)."""
    return _core.et_hopping_rate(
        float(coupling), float(reorganization), float(delta_g), float(temperature)
    )


def proton_states(
    first: float,
    spacing: float,
    potential: Sequence[float],
    mass_amu: float = PROTON_MASS_AMU,
    count: int = 10,
) -> dict[str, Any]:
    """The lowest ``count`` vibrational states of a proton (or of a particle of ``mass_amu``) in
    ``potential`` (Hartree) at the points ``first + i spacing`` (Bohr), by the discrete variable
    representation of Colbert and Miller (J. Chem. Phys. 1992): ``energies``, ``vectors`` (a
    column per state, of unit length), ``mean_positions``, ``grid``."""
    return _core.et_proton_states(
        float(first), float(spacing), [float(v) for v in potential], float(mass_amu), int(count)
    )


def pcet_rate(
    first: float,
    spacing: float,
    reactant: Sequence[float],
    product: Sequence[float],
    coupling: float,
    reorganization: float,
    delta_g0: float = 0.0,
    temperature: float = 298.15,
    mass_amu: float = PROTON_MASS_AMU,
    count: int = 10,
) -> dict[str, Any]:
    """The nonadiabatic PCET rate (1/s) between the proton potentials of the reactant and of the
    product on one grid (Hartree, with the electronic energies of their diabatic states), with
    the electronic coupling, the solvent reorganization energy, and the free energy the
    potentials do not hold (Hartree): ``rate``, ``channels`` (``(mu, nu, share)``, largest
    first), ``overlaps``, and the proton states of both sides. The proton donor-acceptor
    distance is fixed."""
    return _core.et_pcet_rate(
        float(first),
        float(spacing),
        [float(v) for v in reactant],
        [float(v) for v in product],
        float(coupling),
        float(reorganization),
        float(delta_g0),
        float(temperature),
        float(mass_amu),
        int(count),
    )
