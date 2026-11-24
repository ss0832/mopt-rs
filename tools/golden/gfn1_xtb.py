"""Reference GFN1-xTB energies, gradients, and charges from tblite, for mopt-rs's GFN1-xTB.

Usage (in an environment with tblite installed):

    python -I tools/golden/gfn1_xtb.py tests/golden/gfn1_xtb.json [--keep OLD.json]

With ``--keep``, the records of ``OLD.json`` are kept as they are (with the tblite version
that made them) and only the systems it lacks are computed. Every record names the tblite
version that made it (``program``). tblite 0.8.0 gives the electronic energies of systems
with Br or I 1e-6 to 2e-5 Hartree away from tblite 0.4.0 (the light elements agree to
1e-14), so a record with Br or I from a version other than 0.4.0 is marked
``"compare": "classical"``: only its repulsion, halogen-bond and dispersion energies are
compared.

A record made since the fourth period was added also stores ``electron_count_error``, the sum
of tblite's orbital occupations less the nearest integer: tblite ends its search of the Fermi
level within about sqrt(machine epsilon) of the electron count, which moves its energy by that
times the Fermi level (1e-8 Hartree for high-spin iron(II) with its large smearing).

The systems of the fifth and sixth periods (``fifth_and_sixth_periods``, every element from Rb
to Rn other than Ru and I) come last and were made with tblite 0.4.0.

Geometries are built here from internal coordinates (Angstrom, degrees), then displaced by a
fixed pseudo-random vector (seeded) so that the gradients have no symmetry zeros, and
converted to Bohr. The JSON stores the Bohr coordinates actually used (flat, 3N), so the
Angstrom constant used here does not matter for a consumer of the file. tblite runs at
accuracy 1e-4 (SCC thresholds 1e-10 Hartree and 2e-9 e).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
from tblite.interface import Calculator

# CODATA 2018 Bohr radius in Angstrom (used only to build the geometries).
BOHR_IN_ANGSTROM = 0.529177210903
SYMBOLS = {
    z + 1: s
    for z, s in enumerate(
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga "
        "Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
        "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
    )
}
NUMBERS = {v: k for k, v in SYMBOLS.items()}


def zmat_to_xyz(rows):
    """rows: list of (symbol, (i, r), (j, angle), (k, dihedral)); 0-based refs."""
    xyz = []
    syms = []
    for n, row in enumerate(rows):
        sym = row[0]
        syms.append(sym)
        if n == 0:
            xyz.append(np.zeros(3))
            continue
        i, r = row[1]
        if n == 1:
            xyz.append(xyz[i] + np.array([0.0, 0.0, r]))
            continue
        j, ang = row[2]
        ang = math.radians(ang)
        if n == 2:
            a = xyz[i]
            b = xyz[j]
            u = (b - a) / np.linalg.norm(b - a)
            # perpendicular reference axis
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(perp, u)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            perp = perp - np.dot(perp, u) * u
            perp /= np.linalg.norm(perp)
            pos = a + r * (math.cos(ang) * u + math.sin(ang) * perp)
            xyz.append(pos)
            continue
        k, dih = row[3]
        dih = math.radians(dih)
        a = xyz[i]
        b = xyz[j]
        c = xyz[k]
        bc = (a - b) / np.linalg.norm(a - b)
        nrm = np.cross(b - c, bc)
        nrm /= np.linalg.norm(nrm)
        m = np.cross(nrm, bc)
        d2 = np.array(
            [
                -r * math.cos(ang),
                r * math.sin(ang) * math.cos(dih),
                r * math.sin(ang) * math.sin(dih),
            ]
        )
        pos = a + d2[0] * bc + d2[1] * m + d2[2] * nrm
        xyz.append(pos)
    return syms, np.array(xyz)


def systems():
    out = []

    out.append(("H2", 0, 0, zmat_to_xyz([("H",), ("H", (0, 0.74))])))

    out.append(
        ("H2O", 0, 0, zmat_to_xyz([("O",), ("H", (0, 0.958)), ("H", (0, 0.958), (1, 104.5))]))
    )

    out.append(
        ("HCN", 0, 0, zmat_to_xyz([("C",), ("N", (0, 1.156)), ("H", (0, 1.066), (1, 179.0))]))
    )

    out.append(
        (
            "CH3NC",
            0,
            0,
            zmat_to_xyz(
                [
                    ("C",),
                    ("N", (0, 1.424)),
                    ("C", (1, 1.166), (0, 179.0)),
                    ("H", (0, 1.090), (1, 109.5), (2, 0.0)),
                    ("H", (0, 1.090), (1, 109.5), (3, 120.0)),
                    ("H", (0, 1.090), (1, 109.5), (3, 240.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "formamide",
            0,
            0,
            zmat_to_xyz(
                [
                    ("C",),
                    ("O", (0, 1.212)),
                    ("N", (0, 1.360), (1, 124.7)),
                    ("H", (0, 1.100), (1, 122.0), (2, 180.0)),
                    ("H", (2, 1.010), (0, 121.0), (1, 0.0)),
                    ("H", (2, 1.010), (0, 120.0), (1, 180.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "allyl_bromide",
            0,
            0,
            zmat_to_xyz(
                [
                    ("C",),
                    ("C", (0, 1.335)),
                    ("C", (1, 1.495), (0, 124.0)),
                    ("Br", (2, 1.965), (1, 111.0), (0, 110.0)),
                    ("H", (0, 1.085), (1, 121.5), (2, 0.0)),
                    ("H", (0, 1.085), (1, 121.5), (2, 180.0)),
                    ("H", (1, 1.088), (0, 119.0), (2, 180.0)),
                    ("H", (2, 1.092), (1, 111.5), (3, 120.0)),
                    ("H", (2, 1.092), (1, 111.5), (3, -120.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "CH2Cl2",
            0,
            0,
            zmat_to_xyz(
                [
                    ("C",),
                    ("Cl", (0, 1.772)),
                    ("Cl", (0, 1.772), (1, 112.0)),
                    ("H", (0, 1.087), (1, 108.0), (2, 121.0)),
                    ("H", (0, 1.087), (1, 108.0), (2, -121.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "CF3I",
            0,
            0,
            zmat_to_xyz(
                [
                    ("C",),
                    ("I", (0, 2.138)),
                    ("F", (0, 1.330), (1, 110.5)),
                    ("F", (0, 1.330), (1, 110.5), (2, 120.0)),
                    ("F", (0, 1.330), (1, 110.5), (2, -120.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "thiophene",
            0,
            0,
            zmat_to_xyz(
                [
                    ("S",),
                    ("C", (0, 1.714)),
                    ("C", (1, 1.370), (0, 111.5)),
                    ("C", (2, 1.423), (1, 112.4), (0, 0.0)),
                    ("C", (3, 1.370), (2, 112.4), (1, 0.0)),
                    ("H", (1, 1.078), (2, 128.0), (3, 180.0)),
                    ("H", (2, 1.081), (1, 123.5), (0, 180.0)),
                    ("H", (3, 1.081), (4, 123.5), (0, 180.0)),
                    ("H", (4, 1.078), (3, 128.0), (2, 180.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "water_dimer",
            0,
            0,
            zmat_to_xyz(
                [
                    ("O",),
                    ("H", (0, 0.966)),
                    ("O", (1, 1.950), (0, 172.0)),
                    ("H", (0, 0.958), (1, 105.0), (2, 120.0)),
                    ("H", (2, 0.960), (1, 112.0), (0, 60.0)),
                    ("H", (2, 0.960), (1, 112.0), (0, -60.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "allyl_vinyl_ether",
            0,
            0,
            zmat_to_xyz(
                [
                    ("C",),
                    ("C", (0, 1.335)),
                    ("O", (1, 1.360), (0, 127.0)),
                    ("C", (2, 1.420), (1, 118.0), (0, 180.0)),
                    ("C", (3, 1.500), (2, 108.0), (1, 180.0)),
                    ("C", (4, 1.335), (3, 124.0), (2, 120.0)),
                    ("H", (0, 1.083), (1, 121.0), (2, 0.0)),
                    ("H", (0, 1.083), (1, 121.0), (2, 180.0)),
                    ("H", (1, 1.087), (0, 122.0), (2, 180.0)),
                    ("H", (3, 1.095), (2, 110.0), (4, 120.0)),
                    ("H", (3, 1.095), (2, 110.0), (4, -120.0)),
                    ("H", (4, 1.088), (3, 116.0), (5, 180.0)),
                    ("H", (5, 1.085), (4, 121.5), (3, 0.0)),
                    ("H", (5, 1.085), (4, 121.5), (3, 180.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "NH2_radical",
            0,
            1,
            zmat_to_xyz([("N",), ("H", (0, 1.024)), ("H", (0, 1.024), (1, 103.4))]),
        )
    )

    out.append(("hydroxide", -1, 0, zmat_to_xyz([("O",), ("H", (0, 0.970))])))

    out.append(
        (
            "NH3_Br2",
            0,
            0,
            zmat_to_xyz(
                [
                    ("N",),
                    ("Br", (0, 2.750)),
                    ("Br", (1, 2.310), (0, 178.0)),
                    ("H", (0, 1.012), (1, 110.5), (2, 0.0)),
                    ("H", (0, 1.012), (1, 110.5), (3, 120.0)),
                    ("H", (0, 1.012), (1, 110.5), (3, 240.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "NH3_ClF",
            0,
            0,
            zmat_to_xyz(
                [
                    ("N",),
                    ("Cl", (0, 2.380)),
                    ("F", (1, 1.680), (0, 178.0)),
                    ("H", (0, 1.012), (1, 110.5), (2, 0.0)),
                    ("H", (0, 1.012), (1, 110.5), (3, 120.0)),
                    ("H", (0, 1.012), (1, 110.5), (3, 240.0)),
                ]
            ),
        )
    )

    out.append(
        (
            "NH3_ICF3",
            0,
            0,
            zmat_to_xyz(
                [
                    ("N",),
                    ("I", (0, 2.950)),
                    ("C", (1, 2.150), (0, 178.0)),
                    ("F", (2, 1.335), (1, 110.5), (0, 0.0)),
                    ("F", (2, 1.335), (1, 110.5), (3, 120.0)),
                    ("F", (2, 1.335), (1, 110.5), (3, 240.0)),
                    ("H", (0, 1.012), (1, 110.5), (2, 60.0)),
                    ("H", (0, 1.012), (1, 110.5), (6, 120.0)),
                    ("H", (0, 1.012), (1, 110.5), (6, 240.0)),
                ]
            ),
        )
    )

    # extra systems: open shell with uhf=2 and a small-gap case with fractional
    # occupations (tests the Fermi smearing and the electronic entropy term)
    out.append(("O2_triplet", 0, 2, zmat_to_xyz([("O",), ("O", (0, 1.210))])))
    out.append(
        (
            "H4_rectangle",
            0,
            0,
            (
                ["H", "H", "H", "H"],
                np.array(
                    [[0.0, 0.0, 0.0], [1.20, 0.0, 0.0], [1.20, 1.25, 0.03], [0.0, 1.20, -0.02]]
                ),
            ),
        )
    )

    # The elements to Ar (appended last, so that the displacements of the systems above stay
    # those of the earlier file).
    def tetra(center, r, ligand="H"):
        return [
            (center,),
            (ligand, (0, r)),
            (ligand, (0, r), (1, 109.47)),
            (ligand, (0, r), (1, 109.47), (2, 120.0)),
            (ligand, (0, r), (1, 109.47), (2, -120.0)),
        ]

    out.append(("He_dimer", 0, 0, zmat_to_xyz([("He",), ("He", (0, 3.00))])))
    out.append(("LiH", 0, 0, zmat_to_xyz([("Li",), ("H", (0, 1.595))])))
    out.append(("LiF", 0, 0, zmat_to_xyz([("Li",), ("F", (0, 1.564))])))
    out.append(
        ("BeH2", 0, 0, zmat_to_xyz([("Be",), ("H", (0, 1.334)), ("H", (0, 1.334), (1, 179.0))]))
    )
    out.append(
        (
            "ammonia_borane",
            0,
            0,
            zmat_to_xyz(
                [
                    ("B",),
                    ("N", (0, 1.658)),
                    ("H", (0, 1.216), (1, 104.7)),
                    ("H", (0, 1.216), (1, 104.7), (2, 120.0)),
                    ("H", (0, 1.216), (1, 104.7), (2, -120.0)),
                    ("H", (1, 1.014), (0, 109.3), (2, 60.0)),
                    ("H", (1, 1.014), (0, 109.3), (5, 120.0)),
                    ("H", (1, 1.014), (0, 109.3), (5, -120.0)),
                ]
            ),
        )
    )
    out.append(
        (
            "BF3",
            0,
            0,
            zmat_to_xyz(
                [
                    ("B",),
                    ("F", (0, 1.307)),
                    ("F", (0, 1.307), (1, 120.0)),
                    ("F", (0, 1.307), (1, 120.0), (2, 180.0)),
                ]
            ),
        )
    )
    # Not BF4-: tblite 0.8.0 converges it to B -5.8 e, F +1.2 e, 1.66 Hartree below the
    # physical solution (the third-order term q^3 is unbounded below); mopt-rs's own test
    # checks that its SCC stays on the physical one.
    out.append(("BH4_anion", -1, 0, zmat_to_xyz(tetra("B", 1.239))))
    out.append(
        (
            "Ne_water",
            0,
            0,
            zmat_to_xyz(
                [
                    ("O",),
                    ("H", (0, 0.958)),
                    ("H", (0, 0.958), (1, 104.5)),
                    ("Ne", (0, 3.20), (1, 110.0), (2, 120.0)),
                ]
            ),
        )
    )
    out.append(("NaCl", 0, 0, zmat_to_xyz([("Na",), ("Cl", (0, 2.361))])))
    out.append(
        (
            "Na_cation_water",
            1,
            0,
            zmat_to_xyz(
                [
                    ("O",),
                    ("H", (0, 0.958)),
                    ("H", (0, 0.958), (1, 104.5)),
                    ("Na", (0, 2.240), (1, 127.0), (2, 180.0)),
                ]
            ),
        )
    )
    out.append(
        (
            "MgCl2",
            0,
            0,
            zmat_to_xyz([("Mg",), ("Cl", (0, 2.180)), ("Cl", (0, 2.180), (1, 179.0))]),
        )
    )
    out.append(
        (
            "AlCl3",
            0,
            0,
            zmat_to_xyz(
                [
                    ("Al",),
                    ("Cl", (0, 2.060)),
                    ("Cl", (0, 2.060), (1, 120.0)),
                    ("Cl", (0, 2.060), (1, 120.0), (2, 180.0)),
                ]
            ),
        )
    )
    out.append(("SiH4", 0, 0, zmat_to_xyz(tetra("Si", 1.480))))
    out.append(
        (
            "silylamine",
            0,
            0,
            zmat_to_xyz(
                [
                    ("Si",),
                    ("N", (0, 1.725)),
                    ("H", (0, 1.480), (1, 110.0)),
                    ("H", (0, 1.480), (1, 110.0), (2, 120.0)),
                    ("H", (0, 1.480), (1, 110.0), (2, -120.0)),
                    ("H", (1, 1.010), (0, 120.0), (2, 0.0)),
                    ("H", (1, 1.010), (0, 120.0), (2, 180.0)),
                ]
            ),
        )
    )
    out.append(
        (
            "PH3",
            0,
            0,
            zmat_to_xyz(
                [
                    ("P",),
                    ("H", (0, 1.420)),
                    ("H", (0, 1.420), (1, 93.3)),
                    ("H", (0, 1.420), (1, 93.3), (2, 95.0)),
                ]
            ),
        )
    )
    out.append(
        (
            "POCl3",
            0,
            0,
            zmat_to_xyz(
                [
                    ("P",),
                    ("O", (0, 1.450)),
                    ("Cl", (0, 1.990), (1, 115.0)),
                    ("Cl", (0, 1.990), (1, 115.0), (2, 120.0)),
                    ("Cl", (0, 1.990), (1, 115.0), (2, -120.0)),
                ]
            ),
        )
    )
    out.append(
        (
            "borane_phosphine",
            0,
            0,
            zmat_to_xyz(
                [
                    ("B",),
                    ("P", (0, 1.930)),
                    ("H", (0, 1.210), (1, 103.0)),
                    ("H", (0, 1.210), (1, 103.0), (2, 120.0)),
                    ("H", (0, 1.210), (1, 103.0), (2, -120.0)),
                    ("H", (1, 1.400), (0, 117.0), (2, 60.0)),
                    ("H", (1, 1.400), (0, 117.0), (5, 120.0)),
                    ("H", (1, 1.400), (0, 117.0), (5, -120.0)),
                ]
            ),
        )
    )
    out.append(
        (
            "Ar_HF",
            0,
            0,
            zmat_to_xyz([("Ar",), ("F", (0, 3.340)), ("H", (1, 0.920), (0, 175.0))]),
        )
    )
    out.append(
        (
            "PH3_BrCF3",
            0,
            0,
            zmat_to_xyz(
                [
                    ("P",),
                    ("Br", (0, 3.300)),
                    ("C", (1, 1.930), (0, 178.0)),
                    ("F", (2, 1.335), (1, 110.5), (0, 0.0)),
                    ("F", (2, 1.335), (1, 110.5), (3, 120.0)),
                    ("F", (2, 1.335), (1, 110.5), (3, 240.0)),
                    ("H", (0, 1.420), (1, 120.0), (2, 60.0)),
                    ("H", (0, 1.420), (1, 120.0), (6, 120.0)),
                    ("H", (0, 1.420), (1, 120.0), (6, 240.0)),
                ]
            ),
        )
    )

    out.extend(fourth_period())
    # Appended last, so that the displacements of the systems above stay those of the earlier
    # file.
    out.extend(fifth_and_sixth_periods())
    return out


def octahedral(metal, distance, ligand):
    """A metal with six ligands on the axes, each ligand ("H2O" or "NH3") with its donor atom at
    `distance` (Angstrom) and its hydrogens turned away from the metal."""
    axes = [
        np.array(v, dtype=float)
        for v in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])
    ]
    syms = [metal]
    xyz = [np.zeros(3)]
    for k, u in enumerate(axes):
        # a direction perpendicular to the axis
        w = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.5 else np.array([1.0, 0.0, 0.0])
        w = w - np.dot(w, u) * u
        w /= np.linalg.norm(w)
        v = np.cross(u, w)
        donor = distance * u
        if ligand == "H2O":
            syms += ["O", "H", "H"]
            xyz.append(donor)
            for sign in (1.0, -1.0):
                d = math.cos(math.radians(52.0)) * u + sign * math.sin(math.radians(52.0)) * w
                xyz.append(donor + 0.97 * d)
        else:
            syms += ["N", "H", "H", "H"]
            xyz.append(donor)
            for j in range(3):
                phi = 2.0 * math.pi * j / 3.0 + 0.3 * k
                d = math.cos(math.radians(70.0)) * u + math.sin(math.radians(70.0)) * (
                    math.cos(phi) * w + math.sin(phi) * v
                )
                xyz.append(donor + 1.02 * d)
    return syms, np.array(xyz)


def star(center, distance, ligands, directions):
    """`center` with the atoms `ligands` at `distance` along `directions`."""
    syms = [center] + list(ligands)
    xyz = [np.zeros(3)] + [
        distance[k] * np.asarray(d, dtype=float) / np.linalg.norm(d)
        for k, d in enumerate(directions)
    ]
    return syms, np.array(xyz)


TETRAHEDRON = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
TRIGONAL = [[1, 0, 0], [-0.5, 0.8660254, 0], [-0.5, -0.8660254, 0]]


def carbonyl(metal, bonds, directions):
    """`metal` with CO ligands along `directions` at the metal-carbon distances `bonds`."""
    syms = [metal]
    xyz = [np.zeros(3)]
    for d, u in zip(bonds, directions):
        u = np.asarray(u, dtype=float) / np.linalg.norm(u)
        syms += ["C", "O"]
        xyz += [d * u, (d + 1.15) * u]
    return syms, np.array(xyz)


def ferrocene():
    syms = ["Fe"]
    xyz = [np.zeros(3)]
    for z, offset in ((1.66, 0.0), (-1.66, 36.0)):
        for k in range(5):
            phi = math.radians(72.0 * k + offset)
            syms.append("C")
            xyz.append(np.array([1.22 * math.cos(phi), 1.22 * math.sin(phi), z]))
        for k in range(5):
            phi = math.radians(72.0 * k + offset)
            syms.append("H")
            xyz.append(np.array([2.30 * math.cos(phi), 2.30 * math.sin(phi), 1.04 * z]))
    return syms, np.array(xyz)


def fourth_period():
    """Systems with the elements of the fourth period and ruthenium (name, charge, unpaired
    electrons, (symbols, Angstrom))."""
    return [
        ("KF", 0, 0, zmat_to_xyz([("K",), ("F", (0, 2.17))])),
        ("CaCl2", 0, 0, star("Ca", [2.48, 2.48], ["Cl", "Cl"], [[1, 0, 0], [-1, 0.05, 0]])),
        ("ScCl3", 0, 0, star("Sc", [2.29] * 3, ["Cl"] * 3, TRIGONAL)),
        ("TiCl4", 0, 0, star("Ti", [2.17] * 4, ["Cl"] * 4, TETRAHEDRON)),
        (
            "VOCl3",
            0,
            0,
            star("V", [1.57, 2.14, 2.14, 2.14], ["O", "Cl", "Cl", "Cl"], TETRAHEDRON),
        ),
        (
            "CrO2Cl2",
            0,
            0,
            star("Cr", [1.58, 1.58, 2.13, 2.13], ["O", "O", "Cl", "Cl"], TETRAHEDRON),
        ),
        ("MnO4-", -1, 0, star("Mn", [1.63] * 4, ["O"] * 4, TETRAHEDRON)),
        (
            "Fe(CO)5",
            0,
            0,
            carbonyl(
                "Fe",
                [1.81, 1.81, 1.83, 1.83, 1.83],
                [[0, 0, 1], [0, 0, -1]] + TRIGONAL,
            ),
        ),
        ("Fe(H2O)6_2+", 2, 4, octahedral("Fe", 2.12, "H2O")),
        ("Fe(H2O)6_3+", 3, 5, octahedral("Fe", 2.00, "H2O")),
        ("V(H2O)6_3+", 3, 2, octahedral("V", 2.00, "H2O")),
        ("Co(NH3)6_3+", 3, 0, octahedral("Co", 1.97, "NH3")),
        ("Ni(CO)4", 0, 0, carbonyl("Ni", [1.82] * 4, TETRAHEDRON)),
        (
            "CuCl4_2-",
            -2,
            1,
            star("Cu", [2.25] * 4, ["Cl"] * 4, [[1, 0, 0.2], [-1, 0, 0.2], [0, 1, -0.2], [0, -1, -0.2]]),
        ),
        ("ZnCl2", 0, 0, star("Zn", [2.07, 2.07], ["Cl", "Cl"], [[1, 0, 0], [-1, 0.03, 0]])),
        ("GaCl3", 0, 0, star("Ga", [2.10] * 3, ["Cl"] * 3, TRIGONAL)),
        ("GeH4", 0, 0, star("Ge", [1.53] * 4, ["H"] * 4, TETRAHEDRON)),
        (
            "AsH3",
            0,
            0,
            star("As", [1.52] * 3, ["H"] * 3, [[1, 0, -0.35], [-0.5, 0.866, -0.35], [-0.5, -0.866, -0.35]]),
        ),
        ("H2Se", 0, 0, zmat_to_xyz([("Se",), ("H", (0, 1.46)), ("H", (0, 1.46), (1, 91.0))])),
        ("KrF2", 0, 0, star("Kr", [1.89, 1.89], ["F", "F"], [[1, 0, 0], [-1, 0.02, 0]])),
        ("Ferrocene", 0, 0, ferrocene()),
        ("RuO4", 0, 0, star("Ru", [1.70] * 4, ["O"] * 4, TETRAHEDRON)),
        ("Ru(NH3)6_2+", 2, 0, octahedral("Ru", 2.14, "NH3")),
        ("Ru(H2O)6_3+", 3, 1, octahedral("Ru", 2.03, "H2O")),
        # Two metals: the pair scalings of d-block metals (3d-4d; and two coinage metals in
        # GFN0), and an SCC whose extrapolated Fock matrices settle elsewhere.
        ("TiRuH4", 0, 0, ti_ru_hydride()),
        (
            "Cu2Cl2",
            0,
            0,
            (["Cu", "Cu", "Cl", "Cl"], np.array([[1.25, 0, 0], [-1.25, 0, 0], [0, 1.75, 0.1], [0, -1.75, -0.1]])),
        ),
    ]


OCTAHEDRON = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
SQUARE = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
PYRAMID = [[1, 0, -0.35], [-0.5, 0.866, -0.35], [-0.5, -0.866, -0.35]]
# Trigonal, slightly pyramidal (lanthanide trihalides are not planar).
TRIGONAL_PYRAMID = [[1, 0, -0.2], [-0.5, 0.866, -0.2], [-0.5, -0.866, -0.2]]


def coordination(metal, ligands):
    """`metal` with `ligands`, each (kind, distance, direction): an atom symbol, "CO", or
    "NH3" (with its hydrogens turned away from the metal)."""
    syms = [metal]
    xyz = [np.zeros(3)]
    for k, (kind, distance, direction) in enumerate(ligands):
        u = np.asarray(direction, dtype=float)
        u /= np.linalg.norm(u)
        donor = distance * u
        if kind == "CO":
            syms += ["C", "O"]
            xyz += [donor, (distance + 1.15) * u]
        elif kind == "NH3":
            w = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.5 else np.array([1.0, 0.0, 0.0])
            w = w - np.dot(w, u) * u
            w /= np.linalg.norm(w)
            v = np.cross(u, w)
            syms += ["N", "H", "H", "H"]
            xyz.append(donor)
            for j in range(3):
                phi = 2.0 * math.pi * j / 3.0 + 0.3 * k
                d = math.cos(math.radians(70.0)) * u + math.sin(math.radians(70.0)) * (
                    math.cos(phi) * w + math.sin(phi) * v
                )
                xyz.append(donor + 1.02 * d)
        else:
            syms.append(kind)
            xyz.append(donor)
    return syms, np.array(xyz)


def dimethyl(metal, bond):
    """A linear `metal(CH3)2` (Angstrom), staggered methyl groups."""
    return zmat_to_xyz(
        [
            (metal,),
            ("C", (0, bond)),
            ("C", (0, bond), (1, 179.0)),
            ("H", (1, 1.09), (0, 110.0), (2, 0.0)),
            ("H", (1, 1.09), (0, 110.0), (3, 120.0)),
            ("H", (1, 1.09), (0, 110.0), (3, 240.0)),
            ("H", (2, 1.09), (0, 110.0), (1, 60.0)),
            ("H", (2, 1.09), (0, 110.0), (6, 120.0)),
            ("H", (2, 1.09), (0, 110.0), (6, 240.0)),
        ]
    )


# Lanthanide trichlorides (Ln-Cl in Angstrom); La is with the d-block metals below.
LANTHANIDE_CHLORIDES = [
    ("Ce", 2.57), ("Pr", 2.55), ("Nd", 2.54), ("Pm", 2.53), ("Sm", 2.52), ("Eu", 2.51),
    ("Gd", 2.49), ("Tb", 2.48), ("Dy", 2.47), ("Ho", 2.46), ("Er", 2.45), ("Tm", 2.44),
    ("Yb", 2.43), ("Lu", 2.42),
]  # fmt: skip


def fifth_and_sixth_periods():
    """Systems with every element from Rb to Rn other than Ru and I (name, charge, unpaired
    electrons, (symbols, Angstrom)): hydrides, halides, oxides, carbonyls, ammine complexes,
    organometallics, noble gases; closed and open shells, cations and anions."""
    out = [
        ("RbF", 0, 0, zmat_to_xyz([("Rb",), ("F", (0, 2.27))])),
        ("SrCl2", 0, 0, star("Sr", [2.67, 2.67], ["Cl", "Cl"], [[1, 0, 0], [-1, 0.3, 0]])),
        ("YCl3", 0, 0, star("Y", [2.43] * 3, ["Cl"] * 3, TRIGONAL)),
        ("ZrCl4", 0, 0, star("Zr", [2.32] * 4, ["Cl"] * 4, TETRAHEDRON)),
        ("NbCl4_doublet", 0, 1, star("Nb", [2.29] * 4, ["Cl"] * 4, TETRAHEDRON)),
        ("Mo(CO)6", 0, 0, coordination("Mo", [("CO", 2.06, d) for d in OCTAHEDRON])),
        ("TcO4-", -1, 0, star("Tc", [1.71] * 4, ["O"] * 4, TETRAHEDRON)),
        ("Rh(NH3)6_3+", 3, 0, coordination("Rh", [("NH3", 2.08, d) for d in OCTAHEDRON])),
        ("PdCl4_2-", -2, 0, star("Pd", [2.31] * 4, ["Cl"] * 4, SQUARE)),
        ("AgCl", 0, 0, zmat_to_xyz([("Ag",), ("Cl", (0, 2.28))])),
        ("CdCl2", 0, 0, star("Cd", [2.24, 2.24], ["Cl", "Cl"], [[1, 0, 0], [-1, 0.03, 0]])),
        ("InCl3", 0, 0, star("In", [2.29] * 3, ["Cl"] * 3, TRIGONAL)),
        ("SnH4", 0, 0, star("Sn", [1.71] * 4, ["H"] * 4, TETRAHEDRON)),
        ("SnH3_radical", 0, 1, star("Sn", [1.72] * 3, ["H"] * 3, PYRAMID)),
        ("SbH3", 0, 0, star("Sb", [1.71] * 3, ["H"] * 3, PYRAMID)),
        ("H2Te", 0, 0, zmat_to_xyz([("Te",), ("H", (0, 1.66)), ("H", (0, 1.66), (1, 90.5))])),
        ("XeF2", 0, 0, star("Xe", [1.98, 1.98], ["F", "F"], [[1, 0, 0], [-1, 0.02, 0]])),
        ("CsF", 0, 0, zmat_to_xyz([("Cs",), ("F", (0, 2.35))])),
        (
            "Cs_cation_water",
            1,
            0,
            zmat_to_xyz(
                [
                    ("O",),
                    ("H", (0, 0.958)),
                    ("H", (0, 0.958), (1, 104.5)),
                    ("Cs", (0, 3.050), (1, 127.0), (2, 180.0)),
                ]
            ),
        ),
        ("BaO", 0, 0, zmat_to_xyz([("Ba",), ("O", (0, 1.94))])),
        ("LaCl3", 0, 0, star("La", [2.59] * 3, ["Cl"] * 3, TRIGONAL_PYRAMID)),
    ]
    for ln, bond in LANTHANIDE_CHLORIDES:
        out.append((f"{ln}Cl3", 0, 0, star(ln, [bond] * 3, ["Cl"] * 3, TRIGONAL_PYRAMID)))
    out += [
        ("GdH2_doublet", 0, 1, zmat_to_xyz([("Gd",), ("H", (0, 2.05)), ("H", (0, 2.05), (1, 120.0))])),
        ("HfCl4", 0, 0, star("Hf", [2.33] * 4, ["Cl"] * 4, TETRAHEDRON)),
        (
            "TaCl5",
            0,
            0,
            star(
                "Ta",
                [2.37, 2.37, 2.27, 2.27, 2.27],
                ["Cl"] * 5,
                [[0, 0, 1], [0, 0.02, -1]] + TRIGONAL,
            ),
        ),
        ("W(CO)6", 0, 0, coordination("W", [("CO", 2.06, d) for d in OCTAHEDRON])),
        (
            "HRe(CO)5",
            0,
            0,
            coordination("Re", [("H", 1.80, [0, 0, 1])] + [("CO", 2.01, d) for d in OCTAHEDRON[:4]] + [("CO", 1.98, [0, 0, -1])]),
        ),
        ("ReO4-", -1, 0, star("Re", [1.72] * 4, ["O"] * 4, TETRAHEDRON)),
        ("OsO4", 0, 0, star("Os", [1.71] * 4, ["O"] * 4, TETRAHEDRON)),
        ("IrCl6_2-", -2, 1, star("Ir", [2.33] * 6, ["Cl"] * 6, OCTAHEDRON)),
        (
            "cisplatin",
            0,
            0,
            coordination(
                "Pt",
                [("Cl", 2.32, [1, 0, 0]), ("Cl", 2.32, [0, 1, 0]), ("NH3", 2.05, [-1, 0, 0]), ("NH3", 2.05, [0, -1, 0])],
            ),
        ),
        (
            "PtH2(NH3)2",
            0,
            0,
            coordination(
                "Pt",
                [("H", 1.60, [1, 0, 0]), ("H", 1.60, [0, 1, 0]), ("NH3", 2.15, [-1, 0, 0]), ("NH3", 2.15, [0, -1, 0])],
            ),
        ),
        ("AuCl4-", -1, 0, star("Au", [2.28] * 4, ["Cl"] * 4, SQUARE)),
        (
            "ClAgAuCl",
            0,
            0,
            (["Ag", "Au", "Cl", "Cl"], np.array([[0, 0, 0], [2.55, 0, 0], [-2.30, 0.1, 0], [4.83, -0.1, 0.05]])),
        ),
        ("HgCl2", 0, 0, star("Hg", [2.25, 2.25], ["Cl", "Cl"], [[1, 0, 0], [-1, 0.03, 0]])),
        ("Hg(CH3)2", 0, 0, dimethyl("Hg", 2.09)),
        ("TlCl", 0, 0, zmat_to_xyz([("Tl",), ("Cl", (0, 2.48))])),
        ("PbH4", 0, 0, star("Pb", [1.75] * 4, ["H"] * 4, TETRAHEDRON)),
        ("BiCl3", 0, 0, star("Bi", [2.42] * 3, ["Cl"] * 3, PYRAMID)),
        ("H2Po", 0, 0, zmat_to_xyz([("Po",), ("H", (0, 1.73)), ("H", (0, 1.73), (1, 90.5))])),
        ("HAt", 0, 0, zmat_to_xyz([("At",), ("H", (0, 1.72))])),
        (
            "NH3_AtCF3",
            0,
            0,
            zmat_to_xyz(
                [
                    ("N",),
                    ("At", (0, 3.050)),
                    ("C", (1, 2.250), (0, 178.0)),
                    ("F", (2, 1.335), (1, 110.5), (0, 0.0)),
                    ("F", (2, 1.335), (1, 110.5), (3, 120.0)),
                    ("F", (2, 1.335), (1, 110.5), (3, 240.0)),
                    ("H", (0, 1.012), (1, 110.5), (2, 60.0)),
                    ("H", (0, 1.012), (1, 110.5), (6, 120.0)),
                    ("H", (0, 1.012), (1, 110.5), (6, 240.0)),
                ]
            ),
        ),
        (
            "Rn_water",
            0,
            0,
            zmat_to_xyz(
                [
                    ("O",),
                    ("H", (0, 0.958)),
                    ("H", (0, 0.958), (1, 104.5)),
                    ("Rn", (0, 3.60), (1, 110.0), (2, 120.0)),
                ]
            ),
        ),
    ]
    return out


def ti_ru_hydride():
    """A titanium-ruthenium hydride (Bohr coordinates of the probe of the SCC, in Angstrom)."""
    bohr = [
        [0.0, 0.0, 0.0], [0.1, 0.2, 4.8], [3.1, 0.0, -0.8], [-3.0, 0.4, -0.9],
        [0.2, 3.05, 5.7], [-0.3, -3.1, 5.6],
    ]
    return ["Ti", "Ru", "H", "H", "H", "H"], np.array(bohr) * BOHR_IN_ANGSTROM


def parse_log(log):
    """Energy decomposition printed by tblite at verbosity 2."""
    import re

    keys = {
        "repulsion": "repulsion energy",
        "halogen": "halogen bonding energy",
        "dispersion": "dispersion energy",
        "electronic": "electronic energy",
        "total": "total energy",
    }
    out = {}
    for k, label in keys.items():
        m = re.search(label + r"\s+([-+]?\d\.\d+E[-+]\d+)", log)
        out[k] = float(m.group(1)) if m else 0.0
    return out


class CaptureStdout:
    """Capture output written by the Fortran library to file descriptor 1."""

    def __enter__(self):
        sys.stdout.flush()
        self.tmp = tempfile.TemporaryFile(mode="w+b")
        self.saved = os.dup(1)
        os.dup2(self.tmp.fileno(), 1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self.saved, 1)
        os.close(self.saved)
        self.tmp.seek(0)
        self.text = self.tmp.read().decode("utf-8", errors="replace")
        self.tmp.close()
        return False


def run(numbers, pos_bohr, charge, uhf, accuracy, verbosity):
    calc = Calculator("GFN1-xTB", numbers, pos_bohr, charge=float(charge), uhf=int(uhf))
    calc.set("accuracy", accuracy)
    calc.set("max-iter", 500)
    calc.set("verbosity", verbosity)
    with CaptureStdout() as cap:
        res = calc.singlepoint()
    return res, cap.text


def tblite_version():
    from tblite.library import get_version

    return "tblite " + ".".join(str(v) for v in get_version())


def main():
    out_path = sys.argv[1]
    kept = {}
    if len(sys.argv) == 4 and sys.argv[2] == "--keep":
        with open(sys.argv[3]) as fh:
            old = json.load(fh)
        for rec in old["systems"]:
            rec.setdefault("program", old.get("program", "unknown"))
            kept[rec["name"]] = rec
    version = tblite_version()
    rng = np.random.RandomState(20261114)
    records = []
    for name, charge, uhf, (syms, xyz_ang) in systems():
        numbers = np.array([NUMBERS[s] for s in syms], dtype=np.int32)
        # fixed displacement, amplitude 0.03 Angstrom per Cartesian component
        disp = rng.uniform(-0.03, 0.03, size=xyz_ang.shape)
        xyz_ang = xyz_ang - xyz_ang.mean(axis=0) + disp
        pos = xyz_ang / BOHR_IN_ANGSTROM
        if name in kept:
            if kept[name]["positions_bohr"] != pos.ravel().tolist():
                raise SystemExit(f"{name}: the kept record has other positions")
            records.append(kept[name])
            print(f"{name:20s} kept ({kept[name]['program']})", flush=True)
            continue

        tight, log_tight = run(numbers, pos, charge, uhf, 1.0e-4, 2)

        rec = {
            "name": name,
            "symbols": syms,
            "positions_bohr": pos.ravel().tolist(),
            "charge": charge,
            "uhf": uhf,
            "energy": float(tight.get("energy")),
            "gradient": np.asarray(tight.get("gradient")).ravel().tolist(),
            "charges": np.asarray(tight.get("charges")).tolist(),
            "energy_decomposition": parse_log(log_tight),
            "program": version,
        }
        # tblite ends its search of the Fermi level at an electron count within about
        # sqrt(machine epsilon) of the right one; with large smearing (high-spin d shells) the
        # count is then off by 1e-8, and the energy by that times the Fermi level.
        occupations = np.asarray(tight.get("orbital-occupations")).ravel()
        total = float(occupations.sum())
        rec["electron_count_error"] = total - round(total)
        if version != "tblite 0.4.0" and ({"Br", "I"} & set(syms)):
            rec["compare"] = "classical"
        records.append(rec)
        print(f"{name:20s} E = {rec['energy']:.12f}", flush=True)

    doc = {
        "description": "GFN1-xTB reference data from tblite (Python API, Calculator('GFN1-xTB')).",
        "program": "tblite (PyPI wheel); the version of each record is its 'program'",
        "units": {
            "positions": "Bohr",
            "energy": "Hartree",
            "gradient": "Hartree/Bohr",
            "charges": "e (Mulliken, positive = electron deficient)",
            "dipole": "e*Bohr",
            "orbital_energies": "Hartree",
        },
        "settings": {
            "accuracy": 1.0e-4,
            "accuracy_note": "accuracy 1e-4 gives SCC thresholds 1e-10 Eh (energy) and 2e-9 e "
            "(charges)",
            "electronic_temperature": "tblite default, kT = 300 K * 3.166808578545117e-6 Eh/K "
            "= 9.500425735635e-4 Eh",
            "energy_decomposition_note": "parsed from the tblite printout (verbosity 2); "
            "'electronic' includes H0, Coulomb, third order and the "
            "electronic free-energy term -TS",
            "guess": "tblite default (SAD)",
            "max_iter": 500,
        },
        "geometry_note": "Built from internal coordinates in Angstrom (0.529177210903 A/Bohr), "
        "centred, then displaced by a seeded uniform random vector of amplitude "
        "0.03 A per component (numpy RandomState(20261114)).",
        "generator": "tools/golden/gfn1_xtb.py",
        "systems": records,
    }
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
