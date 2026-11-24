"""Compare MultiOptPy's Tersoff (Si) with ASE's Tersoff using the same parameters.

Usage: python tools/golden/check_tersoff.py <multioptpy_root>

This file is part of mopt-rs.
"""

import contextlib
import io
import sys

import numpy as np
from ase import Atoms
from ase.calculators.tersoff import Tersoff, TersoffParameters

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from multioptpy.Calculator.tersoff_calculation_tools import TersoffCore  # noqa: E402
from run_reference import ANG2BOHR, STRUCTURES  # noqa: E402

EV = 27.211386245988
p = TersoffCore.TERSOFF_PARAMETERS["Si"]


def ase_energy_forces(symbols, coords, R, D):
    par = TersoffParameters(
        m=3.0,
        gamma=1.0,
        lambda3=0.0,
        c=p["c"],
        d=p["d"],
        h=p["h"],
        n=p["n"],
        beta=p["beta"],
        lambda2=p["mu"],
        B=p["B"],
        R=R,
        D=D,
        lambda1=p["lambda"],
        A=p["A"],
    )
    atoms = Atoms(symbols, positions=coords, cell=[30, 30, 30], pbc=False)
    atoms.calc = Tersoff({("Si", "Si", "Si"): par})
    return atoms.get_potential_energy(), atoms.get_forces()


symbols, coords = STRUCTURES["si5"]
with contextlib.redirect_stdout(io.StringIO()):
    out = TersoffCore().calculate_energy_and_gradient(coords * ANG2BOHR, symbols)
e_ref = out["energy"] * EV
f_ref = -np.asarray(out["gradient"]).reshape(-1, 3) * EV / 0.52917721067
d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
print("Si-Si distances (A):", np.round(d[np.triu_indices(5, 1)], 3))
print(f"MultiOptPy: E = {e_ref:.6f} eV")
for R, D in [(2.7, 0.3), (2.85, 0.15)]:
    e, f = ase_energy_forces(symbols, coords, R, D)
    print(
        f"ASE (R={R}, D={D}): E = {e:.6f} eV, max|F_ref - F_ase| = {np.abs(f_ref - f).max():.3e} eV/A"
    )

# Finite-difference consistency of the reference energy and gradient (Hartree, Bohr).
core = TersoffCore()
x0 = (coords * ANG2BOHR).copy()
g = np.asarray(out["gradient"]).reshape(-1)
h = 1e-5
worst = 0.0
for k in range(x0.size):
    xp = x0.copy().reshape(-1)
    xm = x0.copy().reshape(-1)
    xp[k] += h
    xm[k] -= h
    with contextlib.redirect_stdout(io.StringIO()):
        ep = core.calculate_energy_and_gradient(xp.reshape(-1, 3), symbols)["energy"]
        em = core.calculate_energy_and_gradient(xm.reshape(-1, 3), symbols)["energy"]
    worst = max(worst, abs((ep - em) / (2 * h) - g[k]))
print(
    f"MultiOptPy gradient vs finite differences of its energy: max error {worst:.3e} Hartree/Bohr"
)
