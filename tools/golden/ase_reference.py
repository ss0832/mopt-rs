"""Reference energies and forces for EMT and Tersoff from ASE.

MultiOptPy's own EMT double-counts pair terms and its Tersoff energy/gradient are inconsistent
(see docs/source/migration_surfaces.md), so ASE's implementations serve as the reference for the native
mopt-rs calculators.

Usage: python tools/golden/ase_reference.py <output_dir>

Writes ``single_point_emt.json`` and ``single_point_tersoff.json`` (Bohr, Hartree; the gradient
is ``-forces``). Units are converted with mopt-rs' constants (``Units`` in Rust) so that the
comparison is not affected by different CODATA revisions. ASE's EMT energies use the ASAP
convention (relative to the bulk cohesive energy).

This file is part of mopt-rs.
"""

import json
import os
import platform
import sys
from itertools import product
from math import sqrt

import ase
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.tersoff import Tersoff, TersoffParameters

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_reference import STRUCTURES  # noqa: E402

# mopt-rs constants (crates/mopt-rs/src/domain/units.rs)
BOHR_IN_ANG = 0.52917721067
EV_PER_HARTREE = 27.21138602

# Tersoff, Phys. Rev. B 39, 5566 (1989), Table I: A, B (eV), lambda, mu (1/A), beta, n, c, d,
# h, R, S (A); chi for mixed pairs.
TERSOFF_1989 = {
    "C": dict(
        A=1393.6,
        B=346.74,
        lam=3.4879,
        mu=2.2119,
        beta=1.5724e-7,
        n=0.72751,
        c=3.8049e4,
        d=4.3484,
        h=-0.57058,
        R=1.8,
        S=2.1,
    ),
    "Si": dict(
        A=1830.8,
        B=471.18,
        lam=2.4799,
        mu=1.7322,
        beta=1.1e-6,
        n=0.78734,
        c=1.0039e5,
        d=16.217,
        h=-0.59825,
        R=2.7,
        S=3.0,
    ),
    "Ge": dict(
        A=1769.0,
        B=419.23,
        lam=2.4451,
        mu=1.7047,
        beta=9.0166e-7,
        n=0.75627,
        c=1.0643e5,
        d=15.652,
        h=-0.43884,
        R=2.8,
        S=3.1,
    ),
}
CHI = {frozenset(["Si", "C"]): 0.9776, frozenset(["Si", "Ge"]): 1.00061}


def chi(a, b):
    return 1.0 if a == b else CHI[frozenset([a, b])]


def tersoff_1989_table(elements):
    """LAMMPS-style (i, j, k) parameters from the 1989 single-element sets and mixing rules."""
    table = {}
    for i, j, k in product(elements, repeat=3):
        pi, pj, pk = TERSOFF_1989[i], TERSOFF_1989[j], TERSOFF_1989[k]
        r_ik, s_ik = sqrt(pi["R"] * pk["R"]), sqrt(pi["S"] * pk["S"])
        table[(i, j, k)] = TersoffParameters(
            m=3.0,
            gamma=1.0,
            lambda3=0.0,
            c=pi["c"],
            d=pi["d"],
            h=pi["h"],
            n=pi["n"],
            beta=pi["beta"],
            lambda2=0.5 * (pi["mu"] + pj["mu"]),
            B=chi(i, j) * sqrt(pi["B"] * pj["B"]),
            R=0.5 * (r_ik + s_ik),
            D=0.5 * (s_ik - r_ik),
            lambda1=0.5 * (pi["lam"] + pj["lam"]),
            A=sqrt(pi["A"] * pj["A"]),
        )
    return table


def record(name, atoms):
    e = atoms.get_potential_energy() / EV_PER_HARTREE
    g = -atoms.get_forces() / EV_PER_HARTREE * BOHR_IN_ANG
    rec = {
        "structure": name,
        "symbols": atoms.get_chemical_symbols(),
        "coords_bohr": (atoms.get_positions() / BOHR_IN_ANG).tolist(),
        "energy_hartree": float(e),
        "gradient": g.reshape(-1).tolist(),
    }
    if atoms.pbc.any():
        rec["cell_bohr"] = (np.asarray(atoms.cell) / BOHR_IN_ANG).tolist()
        rec["pbc"] = [bool(p) for p in atoms.pbc]
    return rec


def rattled(atoms, seed, amplitude=0.08):
    rng = np.random.default_rng(seed)
    atoms = atoms.copy()
    atoms.positions += rng.uniform(-amplitude, amplitude, atoms.positions.shape)
    return atoms


def meta():
    return {"python": platform.python_version(), "ase": ase.__version__, "numpy": np.__version__}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)

    emt_systems = []
    for name in ["cu7", "cu_al4"]:
        symbols, coords = STRUCTURES[name]
        emt_systems.append((name, Atoms(symbols, positions=coords)))
    emt_systems.append(("cu_fcc_bulk", rattled(bulk("Cu", "fcc", a=3.61, cubic=True), 1)))
    emt_systems.append(("cu3au_bulk", rattled(bulk("Cu", "fcc", a=3.75, cubic=True), 2)))
    emt_systems[-1][1].symbols[0] = "Au"
    emt = []
    for name, atoms in emt_systems:
        atoms.calc = EMT()
        emt.append(record(name, atoms))
    with open(os.path.join(out, "single_point_emt.json"), "w") as f:
        json.dump({"meta": meta(), "source": "ase.calculators.emt.EMT", "points": emt}, f, indent=1)

    tersoff_systems = []
    symbols, coords = STRUCTURES["si5"]
    tersoff_systems.append(("si5", Atoms(symbols, positions=coords)))
    tersoff_systems.append(
        ("si_diamond_bulk", rattled(bulk("Si", "diamond", a=5.43, cubic=True), 3))
    )
    sic = Atoms(
        "SiCSiCSi",
        positions=[[0, 0, 0], [1.85, 0.1, 0], [3.1, 1.2, 0.3], [1.0, 1.7, 1.1], [2.3, -1.2, 1.0]],
    )
    tersoff_systems.append(("sic5", sic))
    tersoff = []
    for name, atoms in tersoff_systems:
        elements = sorted(set(atoms.get_chemical_symbols()))
        atoms.calc = Tersoff(tersoff_1989_table(elements))
        tersoff.append(record(name, atoms))
    with open(os.path.join(out, "single_point_tersoff.json"), "w") as f:
        json.dump(
            {
                "meta": meta(),
                "source": "ase.calculators.tersoff.Tersoff with Tersoff (1989) parameters",
                "points": tersoff,
            },
            f,
            indent=1,
        )
    print("wrote EMT and Tersoff references", file=sys.stderr)


if __name__ == "__main__":
    main()
