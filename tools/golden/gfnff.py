"""Reference GFN-FF energies (term by term) and gradients from xtb, for mopt-rs's GFN-FF.

Usage (with the ``xtb`` executable of xtb 6.7.1 on ``PATH``):

    python -I tools/golden/gfnff.py <xtb source tree> <work dir> <output json>

Molecules: those of xtb's ``test/unit/molstock.f90`` (read as text, never executed) made of
the elements H to Rn (among them the transition-metal complexes ``grubbs``, ``manganese``,
``vcpco4``, ``feco5``, ``co_cnx6``, ``fe_cnx6`` and the cerium complex ``Ce_0a7745``), plus
small molecules and complexes built here for the hydrogen-bond cases (bound H, carbonyl, nitro,
two-coordinated N acceptor), halogen bonds, ions, the noble gases, main-group compounds of the
fifth and sixth periods, 4d and 5d complexes (sandwich, piano-stool and eta complexes,
hydrides), lanthanide complexes, and hydrogen and halogen bonds with heavy donors and
acceptors, and the systems with the elements beyond neon of ``tests/golden/gfn1_xtb.json``
(its geometries, named ``elements_<name>``). Each molecule runs ``xtb coord --gfnff --grad`` in
an empty directory of its own (xtb reads files such as ``charges`` from the working directory).
Positions are in Bohr, energies in Hartree, gradients in Hartree/Bohr.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

ANGSTROM = 1.0 / 0.52917721067
SYMBOLS = {
    z + 1: s
    for z, s in enumerate(
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
        "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm "
        "Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
    )
}
ALLOWED = set(SYMBOLS.values())
NEW_ELEMENT_SYSTEMS = [
    "He_dimer", "LiH", "LiF", "BeH2", "ammonia_borane", "BF3", "BH4_anion", "Ne_water",
    "NaCl", "Na_cation_water", "MgCl2", "AlCl3", "SiH4", "silylamine", "PH3", "POCl3",
    "borane_phosphine", "Ar_HF",
    # The fourth period and Ru.
    "KF", "CaCl2", "ScCl3", "TiCl4", "VOCl3", "CrO2Cl2", "MnO4-", "Fe(CO)5", "Fe(H2O)6_2+",
    "Fe(H2O)6_3+", "V(H2O)6_3+", "Co(NH3)6_3+", "Ni(CO)4", "CuCl4_2-", "ZnCl2", "GaCl3", "GeH4",
    "AsH3", "H2Se", "KrF2", "Ferrocene", "RuO4", "Ru(NH3)6_2+", "Ru(H2O)6_3+", "TiRuH4", "Cu2Cl2",
    # The fifth and sixth periods.
    "PH3_BrCF3", "RbF", "SrCl2", "YCl3", "ZrCl4", "NbCl4_doublet", "Mo(CO)6", "TcO4-",
    "Rh(NH3)6_3+", "PdCl4_2-", "AgCl", "CdCl2", "InCl3", "SnH4", "SnH3_radical", "SbH3", "H2Te",
    "XeF2", "CsF", "Cs_cation_water", "BaO", "LaCl3", "CeCl3", "PrCl3", "NdCl3", "PmCl3",
    "SmCl3", "EuCl3", "GdCl3", "TbCl3", "DyCl3", "HoCl3", "ErCl3", "TmCl3", "YbCl3", "LuCl3",
    "GdH2_doublet", "HfCl4", "TaCl5", "W(CO)6", "HRe(CO)5", "ReO4-", "OsO4", "IrCl6_2-",
    "cisplatin", "PtH2(NH3)2", "AuCl4-", "ClAgAuCl", "HgCl2", "Hg(CH3)2", "TlCl", "PbH4",
    "BiCl3", "H2Po", "HAt", "NH3_AtCF3", "Rn_water",
]  # fmt: skip
MOLSTOCK = [
    "caffeine", "rivaroxaban", "taxol", "pdb_4qxx", "x06_benzene", "h2o", "bug332",
    "grubbs", "manganese", "vcpco4", "feco5", "co_cnx6", "fe_cnx6", "Ce_0a7745",
]  # fmt: skip
TERMS = {
    "bond": "bond energy",
    "angle": "angle energy",
    "torsion": "torsion energy",
    "repulsion": "repulsion energy",
    "electrostatic": "electrostat energy",
    "dispersion": "dispersion energy",
    "hbond": "HB energy",
    "xbond": "XB energy",
    "batm": "bonded atm energy",
    "total": "total energy",
}


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def substituents(center, toward, length, count=3):
    """Positions of ``count`` tetrahedral substituents of ``center`` opposite to ``toward``."""
    u = unit([toward[k] - center[k] for k in range(3)])
    a = [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0]
    e1 = unit([a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2], a[0] * u[1] - a[1] * u[0]])
    e2 = [u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2], u[0] * e1[1] - u[1] * e1[0]]
    c, s = math.cos(math.radians(109.47)), math.sin(math.radians(109.47))
    out = []
    for m in range(count):
        p = 2.0 * math.pi * m / count
        d = [c * u[k] + s * (math.cos(p) * e1[k] + math.sin(p) * e2[k]) for k in range(3)]
        out.append([center[k] + length * d[k] for k in range(3)])
    return out


def built():
    """Small molecules and complexes (symbols, Angstrom, charge)."""
    mols = {}
    c, br, n = [0.0, 0.0, 0.0], [0.0, 0.0, 1.94], [0.0, 0.0, 4.94]
    mols["bromomethane_ammonia"] = (
        ["C", "Br", "H", "H", "H", "N", "H", "H", "H"],
        [c, br, *substituents(c, br, 1.09), n, *substituents(n, [0.0, 0.0, 9.0], 1.01)],
        0,
    )
    i, n = [0.0, 0.0, 2.14], [0.0, 0.0, 5.04]
    mols["trifluoroiodomethane_ammonia"] = (
        ["C", "I", "F", "F", "F", "N", "H", "H", "H"],
        [c, i, *substituents(c, i, 1.35), n, *substituents(n, [0.0, 0.0, 9.0], 1.01)],
        0,
    )
    mols["hf_dimer"] = (
        ["F", "H", "F", "H"],
        [[0.0, 0.0, 0.0], [0.92, 0.0, 0.0], [2.75, 0.0, 0.0], [3.05, 0.88, 0.0]],
        0,
    )
    nn = [0.0, 0.0, 1.49]
    mols["methylammonium"] = (
        ["C", "N", "H", "H", "H", "H", "H", "H"],
        [c, nn, *substituents(c, nn, 1.09), *substituents(nn, c, 1.02)],
        1,
    )
    c2 = [1.52, 0.0, 0.0]
    mols["acetate_water"] = (
        ["C", "C", "O", "O", "H", "H", "H", "O", "H", "H"],
        [
            c,
            c2,
            [2.15, 1.10, 0.0],
            [2.15, -1.10, 0.0],
            *substituents(c, c2, 1.09),
            [4.80, 2.05, 0.0],
            [3.86, 1.78, 0.0],
            [5.04, 3.00, 0.0],
        ],
        -1,
    )
    nn = [1.49, 0.0, 0.0]
    mols["nitromethane_water"] = (
        ["C", "N", "O", "O", "H", "H", "H", "O", "H", "H"],
        [
            c,
            nn,
            [2.09, 1.07, 0.0],
            [2.09, -1.07, 0.0],
            *substituents(c, nn, 1.09),
            [2.90, 3.78, 0.0],
            [2.60, 2.85, 0.0],
            [3.85, 3.54, 0.0],
        ],
        0,
    )
    ring = [
        [1.39 * math.cos(math.radians(60 * m)), 1.39 * math.sin(math.radians(60 * m)), 0.0]
        for m in range(6)
    ]
    hs = [
        [2.47 * math.cos(math.radians(60 * m)), 2.47 * math.sin(math.radians(60 * m)), 0.0]
        for m in range(1, 6)
    ]
    mols["pyridine_water"] = (
        ["N", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "O", "H"],
        [*ring, *hs, [3.29, 0.0, 0.0], [4.27, 0.0, 0.0], [4.51, 0.95, 0.0]],
        0,
    )
    s, o, ca, cb = [0.0, 0.0, 0.0], [0.0, 0.0, 1.50], [1.40, 0.0, -1.13], [-0.70, 1.21, -1.13]
    mols["dimethyl_sulfoxide"] = (
        ["S", "O", "C", "C", "H", "H", "H", "H", "H", "H"],
        [s, o, ca, cb, *substituents(ca, s, 1.09), *substituents(cb, s, 1.09)],
        0,
    )
    sx = [1.82, 0.0, 0.0]
    mols["methanethiol"] = (
        ["C", "S", "H", "H", "H", "H"],
        [c, sx, [2.15, 1.30, 0.0], *substituents(c, sx, 1.09)],
        0,
    )
    h = [0.0, 0.0, 1.09]
    mols["chloroform"] = (["C", "H", "Cl", "Cl", "Cl"], [c, h, *substituents(c, h, 1.77)], 0)
    mols["hcn_hf"] = (
        ["H", "C", "N", "H", "F"],
        [[-1.07, 0.0, 0.0], c, [1.16, 0.0, 0.0], [3.10, 0.02, 0.0], [4.02, 0.05, 0.0]],
        0,
    )
    mols["fluoride_water"] = (
        ["F", "H", "O", "H"],
        [c, [1.55, 0.0, 0.0], [2.52, 0.0, 0.0], [2.76, 0.93, 0.0]],
        -1,
    )
    mols["formamide"] = (
        ["C", "O", "N", "H", "H", "H"],
        [
            [0.0, 0.42, 0.0],
            [1.2, 0.6, 0.02],
            [-0.6, -0.8, 0.0],
            [-0.64, 1.31, 0.03],
            [-0.08, -1.62, -0.02],
            [-1.6, -0.88, 0.05],
        ],
        0,
    )
    mols["bromoiodoethyne"] = (
        ["Br", "C", "C", "I"],
        [[-1.80, 0.0, 0.0], [0.0, 0.0, 0.0], [1.21, 0.0, 0.0], [3.20, 0.0, 0.05]],
        0,
    )
    return {
        k: (sym, [[v * ANGSTROM for v in p] for p in pos], chrg)
        for k, (sym, pos, chrg) in mols.items()
    }


def add(a, b, s=1.0):
    return [a[k] + s * b[k] for k in range(3)]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def frame(axis):
    """Two unit vectors perpendicular to ``axis`` and to each other."""
    u = unit(axis)
    a = [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0]
    e1 = unit(cross(a, u))
    return e1, cross(u, e1)


def methyl(center, toward, length=1.09):
    """The three H of a methyl group on ``center`` opposite to ``toward``."""
    return substituents(center, toward, length)


def ring(center, axis, n, r_c, r_h, phase=0.0):
    """A planar ``n``-ring of C (radius ``r_c``) with radial H (radius ``r_h``) around ``axis``."""
    e1, e2 = frame(axis)
    cs, hs = [], []
    for m in range(n):
        p = 2.0 * math.pi * m / n + phase
        d = [math.cos(p) * e1[k] + math.sin(p) * e2[k] for k in range(3)]
        cs.append(add(center, d, r_c))
        hs.append(add(center, d, r_h))
    return cs, hs


def octahedral(center, length):
    return [add(center, d, length) for d in OCTA]


OCTA = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
TETRA = [unit(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]


def ligand_water(metal, d, dist):
    """A water O at ``dist`` from ``metal`` along ``d`` with its H atoms pointing away."""
    u = unit(d)
    o = add(metal, u, dist)
    e1, _ = frame(u)
    c, s = math.cos(math.radians(52.25)), math.sin(math.radians(52.25))
    h1 = add(o, [c * u[k] + s * e1[k] for k in range(3)], 0.96)
    h2 = add(o, [c * u[k] - s * e1[k] for k in range(3)], 0.96)
    return [o, h1, h2]


def ligand_ammine(metal, d, dist):
    u = unit(d)
    n = add(metal, u, dist)
    return [n, *substituents(n, metal, 1.01)]


def heavy_built():
    """Compounds of the fifth and sixth periods, 4d/5d and lanthanide complexes, noble gases,
    and hydrogen and halogen bonds with heavy atoms (symbols, Angstrom, charge)."""
    o = [0.0, 0.0, 0.0]
    z = [0.0, 0.0, 1.0]
    mols = {}
    # Noble gases.
    mols["xenon_atom"] = (["Xe"], [o], 0)
    mols["radon_atom"] = (["Rn"], [o], 0)
    mols["xenon_dimer"] = (["Xe", "Xe"], [o, [4.36, 0.0, 0.0]], 0)
    mols["krypton_xenon"] = (["Kr", "Xe"], [o, [4.10, 0.0, 0.0]], 0)
    mols["xenon_water"] = (["Xe", *["O", "H", "H"]], [o, *ligand_water(o, [1, 0, 0], 3.80)], 0)
    mols["xenon_tetrafluoride"] = (
        ["Xe", "F", "F", "F", "F"],
        [o, *[add(o, d, 1.95) for d in OCTA[:4]]],
        0,
    )
    # Main-group compounds of the fifth and sixth periods.
    for name, m, d in [("tetramethyltin", "Sn", 2.14), ("tetramethyllead", "Pb", 2.24)]:
        cs = [add(o, t, d) for t in TETRA]
        mols[name] = (
            [m, "C", "C", "C", "C"] + ["H"] * 12,
            [o, *cs, *[h for c in cs for h in methyl(c, o)]],
            0,
        )
    a = math.radians(95.0 / 2)
    mols["tin_dichloride"] = (
        ["Sn", "Cl", "Cl"],
        [o, [2.35 * math.sin(a), 0.0, 2.35 * math.cos(a)], [-2.35 * math.sin(a), 0.0, 2.35 * math.cos(a)]],
        0,
    )
    ax, eq = math.radians(85.0), math.radians(51.0)
    mols["tellurium_tetrachloride"] = (
        ["Te", "Cl", "Cl", "Cl", "Cl"],
        [
            o,
            [2.43 * math.sin(ax), 0.0, 2.43 * math.cos(ax)],
            [-2.43 * math.sin(ax), 0.0, 2.43 * math.cos(ax)],
            [0.0, 2.29 * math.sin(eq), -2.29 * math.cos(eq)],
            [0.0, -2.29 * math.sin(eq), -2.29 * math.cos(eq)],
        ],
        0,
    )
    b = math.radians(81.9)
    mols["iodine_pentafluoride"] = (
        ["I", "F", "F", "F", "F", "F"],
        [o, [0.0, 0.0, 1.84]]
        + [
            [1.87 * math.sin(b) * math.cos(p), 1.87 * math.sin(b) * math.sin(p), 1.87 * math.cos(b)]
            for p in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
        ],
        0,
    )
    mols["antimony_pentachloride"] = (
        ["Sb"] + ["Cl"] * 5,
        [o, [0.0, 0.0, 2.34], [0.0, 0.0, -2.34]]
        + [[2.27 * math.cos(p), 2.27 * math.sin(p), 0.0] for p in (0.0, 2.0944, 4.1888)],
        0,
    )
    mols["triiodide"] = (["I", "I", "I"], [[-2.92, 0.0, 0.0], o, [2.92, 0.0, 0.0]], -1)
    mols["hexafluoroantimonate"] = (["Sb"] + ["F"] * 6, [o, *octahedral(o, 1.88)], -1)
    cs = [[2.16 * math.cos(p), 2.16 * math.sin(p), 0.0] for p in (0.0, 2.0944, 4.1888)]
    mols["trimethylindium"] = (
        ["In", "C", "C", "C"] + ["H"] * 9,
        [o, *cs, *[h for c in cs for h in methyl(c, o)]],
        0,
    )
    cs = substituents(o, [0.0, 0.0, 1.0], 2.26)
    mols["trimethylbismuth"] = (
        ["Bi", "C", "C", "C"] + ["H"] * 9,
        [o, *cs, *[h for c in cs for h in methyl(c, o)]],
        0,
    )
    mols["thallium_cation_water"] = (
        ["Tl", "O", "H", "H"],
        [o, *ligand_water(o, [1, 0, 0], 2.75)],
        1,
    )
    # 4d and 5d complexes.
    for name, m in [("ruthenocene", "Ru"), ("osmocene", "Os")]:
        c1, h1 = ring([0.0, 0.0, 1.82], z, 5, 1.21, 2.28)
        c2, h2 = ring([0.0, 0.0, -1.82], z, 5, 1.21, 2.28)
        mols[name] = ([m] + ["C"] * 10 + ["H"] * 10, [o, *c1, *c2, *h1, *h2], 0)
    half = math.radians(130.0 / 2)
    cents = [[2.21 * math.sin(half), 0.0, 2.21 * math.cos(half)], [-2.21 * math.sin(half), 0.0, 2.21 * math.cos(half)]]
    cl = math.radians(97.0 / 2)
    sym, pos = ["Zr", "Cl", "Cl"], [o, [0.0, 2.44 * math.sin(cl), -2.44 * math.cos(cl)], [0.0, -2.44 * math.sin(cl), -2.44 * math.cos(cl)]]
    for cen in cents:
        cr, hr = ring(cen, cen, 5, 1.20, 2.27)
        sym += ["C"] * 5 + ["H"] * 5
        pos += cr + hr
    mols["zirconocene_dichloride"] = (sym, pos, 0)
    # (eta6-benzene)Mo(CO)3 and (eta5-Cp)Re(CO)3.
    for name, m, n, rc, rh, dc in [
        ("benzene_molybdenum_tricarbonyl", "Mo", 6, 1.41, 2.49, 1.72),
        ("cyclopentadienyl_rhenium_tricarbonyl", "Re", 5, 1.21, 2.28, 1.96),
    ]:
        cr, hr = ring([0.0, 0.0, dc], z, n, rc, rh)
        sym, pos = [m] + ["C"] * n + ["H"] * n, [o, *cr, *hr]
        t = math.radians(125.0)
        for p in (0.0, 2.0944, 4.1888):
            d = [math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)]
            # M-C-O bent by 3 degrees: with an exactly linear M-C-O, xtb's analytic gradient
            # (and mopt-rs's) reached 1e6 Hartree/Bohr, which differences of the energy do not
            # show.
            side = unit(cross(d, z))
            sym += ["C", "O"]
            pos += [add(o, d, 1.96), add(add(o, d, 1.96 + 1.16), side, 1.16 * math.sin(math.radians(3.0)))]
        mols[name] = (sym, pos, 0)
    ps = [add(o, t, 2.33) for t in TETRA]
    mols["tetrakis_phosphine_palladium"] = (
        ["Pd", "P", "P", "P", "P"] + ["H"] * 12,
        [o, *ps, *[h for p in ps for h in substituents(p, o, 1.42)]],
        0,
    )
    mols["tungsten_hexachloride"] = (["W"] + ["Cl"] * 6, [o, *octahedral(o, 2.26)], 0)
    mols["niobium_pentafluoride"] = (
        ["Nb"] + ["F"] * 5,
        [o, [0.0, 0.0, 1.88], [0.0, 0.0, -1.88]]
        + [[1.86 * math.cos(p), 1.86 * math.sin(p), 0.0] for p in (0.0, 2.0944, 4.1888)],
        0,
    )
    mols["diammine_silver"] = (
        ["Ag", "N", "H", "H", "H", "N", "H", "H", "H"],
        [o, *ligand_ammine(o, [1, 0, 0], 2.12), *ligand_ammine(o, [-1, 0, 0], 2.12)],
        1,
    )
    p = [0.0, 0.0, 2.23]
    mols["chloro_phosphine_gold"] = (
        ["Au", "Cl", "P", "H", "H", "H"],
        [o, [0.0, 0.0, -2.28], p, *substituents(p, o, 1.42)],
        0,
    )
    c1, c2 = [0.0, 0.0, 2.11], [0.0, 0.0, -2.11]
    mols["dimethylcadmium"] = (
        ["Cd", "C", "C"] + ["H"] * 6,
        [o, c1, c2, *methyl(c1, o), *methyl(c2, o)],
        0,
    )
    mols["hexahydridoiridate"] = (["Ir"] + ["H"] * 6, [o, *octahedral(o, 1.66)], -3)
    # Zeise's anion: PtCl3 in the xy plane, ethylene perpendicular to it along -x.
    cc = [[-2.02, 0.0, 0.69], [-2.02, 0.0, -0.69]]
    hh = []
    for c in cc:
        sgn = 1.0 if c[2] > 0 else -1.0
        hh += [[c[0] - 0.35, 0.92, c[2] + sgn * 0.50], [c[0] - 0.35, -0.92, c[2] + sgn * 0.50]]
    mols["zeise_anion"] = (
        ["Pt", "Cl", "Cl", "Cl", "C", "C", "H", "H", "H", "H"],
        [o, [2.32, 0.0, 0.0], [0.0, 2.30, 0.0], [0.0, -2.30, 0.0], *cc, *hh],
        -1,
    )
    # Lanthanide complexes.
    sym, pos = ["La"], [o]
    for k in range(8):
        p = math.pi / 4 * k
        t = math.radians(57.0 if k % 2 == 0 else 123.0)
        sym += ["O", "H", "H"]
        pos += ligand_water(o, [math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)], 2.55)
    mols["lanthanum_octaaqua"] = (sym, pos, 3)
    sym, pos = ["Gd"], [o]
    for d in OCTA:
        sym += ["N", "H", "H", "H"]
        pos += ligand_ammine(o, d, 2.55)
    mols["gadolinium_hexaammine"] = (sym, pos, 3)
    mols["lanthanum_trihydride"] = (["La", "H", "H", "H"], [o, *substituents(o, z, 2.13)], 0)
    cs = [[2.33 * math.cos(p), 2.33 * math.sin(p), 0.0] for p in (0.0, 2.0944, 4.1888)]
    mols["trimethyllutetium"] = (
        ["Lu", "C", "C", "C"] + ["H"] * 9,
        [o, *cs, *[h for c in cs for h in methyl(c, o)]],
        0,
    )
    sym, pos = ["Ce"], [o]
    for p in (0.0, 2.0944, 4.1888):
        cen = [2.52 * math.cos(p), 2.52 * math.sin(p), 0.0]
        cr, hr = ring(cen, cen, 5, 1.20, 2.27)
        sym += ["C"] * 5 + ["H"] * 5
        pos += cr + hr
    mols["tris_cyclopentadienyl_cerium"] = (sym, pos, 0)
    mols["europium_hydroxide_water"] = (
        ["Eu", "O", "H", "O", "H", "H"],
        [o, [2.15, 0.0, 0.0], [3.10, 0.0, 0.0], *ligand_water(o, [-1, 0.2, 0], 2.50)],
        2,
    )
    # Hydrogen and halogen bonds with heavy atoms.
    sb = o
    # The H atoms toward +z, the lone pair and the water toward -z.
    hs = substituents(sb, [0.0, 0.0, -1.0], 1.70)
    ow = [0.0, 0.0, -3.60]
    mols["water_stibine"] = (
        ["Sb", "H", "H", "H", "O", "H", "H"],
        [sb, *hs, ow, [0.0, 0.0, -2.64], [0.92, 0.0, -3.86]],
        0,
    )
    ow = [3.46, 0.0, 0.0]
    mols["hydrogen_iodide_water"] = (
        ["I", "H", "O", "H", "H"],
        [o, [1.61, 0.0, 0.0], ow, [3.78, 0.78, 0.45], [3.78, -0.78, 0.45]],
        0,
    )
    mols["iodide_water"] = (
        ["I", "H", "O", "H"],
        [o, [2.62, 0.0, 0.0], [3.58, 0.0, 0.0], [3.82, 0.93, 0.0]],
        -1,
    )
    half = math.radians(94.0 / 2)
    c1 = [-2.14 * math.cos(half), 2.14 * math.sin(half), 0.0]
    c2 = [-2.14 * math.cos(half), -2.14 * math.sin(half), 0.0]
    n = add(o, unit([-c1[k] for k in range(3)]), 3.10)
    mols["dimethyltelluride_ammonia"] = (
        ["Te", "C", "C"] + ["H"] * 6 + ["N", "H", "H", "H"],
        [o, c1, c2, *methyl(c1, o), *methyl(c2, o), n, *substituents(n, o, 1.01)],
        0,
    )
    n = [0.0, 0.0, 2.70]
    mols["iodine_chloride_ammonia"] = (
        ["I", "Cl", "N", "H", "H", "H"],
        [o, [0.0, 0.0, -2.32], n, *substituents(n, o, 1.01)],
        0,
    )
    return {
        k: (sym, [[v * ANGSTROM for v in p] for p in pos], chrg)
        for k, (sym, pos, chrg) in mols.items()
    }


def new_elements():
    """The systems of ``NEW_ELEMENT_SYSTEMS`` from ``tests/golden/gfn1_xtb.json`` (Bohr)."""
    path = Path(__file__).resolve().parents[2] / "tests/golden/gfn1_xtb.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for rec in data["systems"]:
        if rec["name"] in NEW_ELEMENT_SYSTEMS:
            x = rec["positions_bohr"]
            n = len(rec["symbols"])
            out["elements_" + rec["name"]] = (
                rec["symbols"],
                [x[3 * k : 3 * k + 3] for k in range(n)],
                rec["charge"],
            )
    return out


def molstock(path):
    """Molecules of ``molstock.f90`` with the allowed elements (Bohr)."""
    text = path.read_text(encoding="utf-8")
    out = {}
    for name in MOLSTOCK:
        m = re.search(rf"subroutine\s+{name}\(mol\)(.*?)end subroutine\s+{name}", text, re.S)
        if m is None:
            raise SystemExit(f"{name} not found in molstock.f90")
        body = m.group(1)
        if "lattice" in body:
            continue
        sym_m = re.search(r"sym\(nat\)\s*=\s*\[character\(len=\d+\)\s*::(.*?)\]", body, re.S)
        at_m = re.search(r"at\(nat\)\s*=\s*&?\s*\[(.*?)\]", body, re.S)
        xyz_m = re.search(r"xyz\(3,\s*nat\)\s*=\s*reshape\(\[(.*?)\]", body, re.S)
        if xyz_m is None or (sym_m is None and at_m is None):
            raise SystemExit(f"{name}: unexpected layout")
        if sym_m is not None:
            sym = [s.strip().capitalize() for s in re.findall(r'"([^"]+)"', sym_m.group(1))]
        else:
            sym = [SYMBOLS.get(int(z), "X") for z in re.findall(r"\d+", at_m.group(1))]
        vals = [float(v) for v in re.findall(r"[-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?", xyz_m.group(1))]
        if not set(sym) <= ALLOWED:
            continue
        if len(vals) != 3 * len(sym):
            raise SystemExit(f"{name}: {len(vals)} coordinates for {len(sym)} atoms")
        chrg_m = re.search(r"chrg\s*=\s*([-+]?\d+)", body)
        chrg = int(chrg_m.group(1)) if chrg_m else 0
        xyz = [vals[3 * a : 3 * a + 3] for a in range(len(sym))]
        out[name] = (sym, xyz, chrg)
        # A charge given as ``charge = <q>_wp`` (manganese, bug332, co_cnx6, fe_cnx6,
        # Ce_0a7745) is not read for the record above, which is then of the neutral species;
        # the charged species is a record of its own.
        q_m = re.search(r"::\s*charge\s*=\s*([-+]?\d+)\.\d*_wp", body)
        if q_m is not None and int(q_m.group(1)) != 0:
            out[name + "_charged"] = (sym, xyz, int(q_m.group(1)))
    return out


def run_xtb(work, name, sym, pos, chrg):
    d = work / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    lines = ["$coord"]
    lines += [
        f"{p[0]:22.14f} {p[1]:22.14f} {p[2]:22.14f} {s.lower()}"
        for s, p in zip(sym, pos, strict=True)
    ]
    lines.append("$end")
    (d / "coord").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = ["xtb", "coord", "--gfnff", "--grad", "--chrg", str(chrg)]
    res = subprocess.run(cmd, cwd=d, capture_output=True, text=True, check=False)
    (d / "out.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        raise SystemExit(f"{name}: xtb failed ({res.returncode})")
    terms = {}
    for key, label in TERMS.items():
        m = re.findall(rf"::\s*{label}\s+([-+]?\d+\.\d+)\s+Eh", res.stdout)
        if not m:
            raise SystemExit(f"{name}: no '{label}' in the xtb output")
        terms[key] = float(m[-1])
    grad_lines = (d / "gradient").read_text(encoding="utf-8").splitlines()
    n = len(sym)
    g = []
    for line in grad_lines[2 + n : 2 + 2 * n]:
        g += [float(v.replace("D", "E")) for v in line.split()]
    if len(g) != 3 * n:
        raise SystemExit(f"{name}: bad gradient file")
    return terms, g


def main():
    root, work, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    version = subprocess.run(["xtb", "--version"], capture_output=True, text=True, check=False)
    vline = next((ln.strip() for ln in version.stdout.splitlines() if "version" in ln), "")
    mols = molstock(root / "test/unit/molstock.f90")
    mols.update(built())
    mols.update(new_elements())
    mols.update(heavy_built())
    cases = []
    for name, (sym, pos, chrg) in mols.items():
        # The positions exactly as written to xtb's coord file.
        pos = [[float(f"{v:.14f}") for v in p] for p in pos]
        try:
            terms, g = run_xtb(work, name, sym, pos, chrg)
        except SystemExit as err:
            # xtb has no energy for some of the charged molstock species (co_cnx6 with -3:
            # NaN); those are left out.
            if not name.endswith("_charged"):
                raise
            print(f"{name}: left out ({err})")
            continue
        cases.append(
            {
                "molecule": name,
                "symbols": sym,
                "positions_bohr": [v for p in pos for v in p],
                "charge": chrg,
                "terms": terms,
                "gradient": g,
            }
        )
        print(f"{name}: {len(sym)} atoms, E = {terms['total']:.12f}")
    doc = {
        "generator": "tools/golden/gfnff.py",
        "reference": f"xtb --gfnff ({vline})",
        "cases": cases,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
