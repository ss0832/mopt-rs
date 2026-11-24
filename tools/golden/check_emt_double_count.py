"""Test the hypothesis that MultiOptPy's EMT double-counts pair terms.

MultiOptPy loops over a full neighbor list (every pair seen from both atoms) but applies ASE's
pair update, which assumes a half list (each pair once). Restricting the loop to a2 > a1 should
reproduce ASE exactly if that is the only difference.

Usage: python tools/golden/check_emt_double_count.py <multioptpy_root>

This file is part of mopt-rs.
"""

import contextlib
import io
import sys

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT as AseEMT

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from multioptpy.Calculator import emt_calculation_tools as ref  # noqa: E402
from run_reference import STRUCTURES  # noqa: E402


def half_list_energy_forces(calc, positions):
    """Reference code with the neighbor list restricted to a2 > a1 for the pair loop."""
    calc.positions = positions
    n = len(positions)
    calc.energies = np.zeros(n)
    calc.forces_ev_per_ang = np.zeros((n, 3))
    calc.deds = np.zeros(n)
    sigma1 = np.zeros(n)
    ps = {}
    for a1 in range(n):
        diffs = positions - positions[a1]
        dists = np.linalg.norm(diffs, axis=1)
        idx = np.where((np.arange(n) > a1) & (dists < calc.rc_list))[0]
        if len(idx) == 0:
            continue
        a2, d, r = idx, diffs[idx], dists[idx]
        w, dwdroverw = calc._calc_theta(r)
        s1s, s1o = calc._calc_dsigma1(a1, a2, r, w)
        s2s, s2o = calc._calc_dsigma2(a1, a2, r, w)
        sigma1[a1] += s1s.sum()
        np.add.at(sigma1, a2, s1o)
        ps[a1] = dict(
            a2=a2,
            d=d,
            r=r,
            invr=1.0 / r,
            w=w,
            dwdroverw=dwdroverw,
            dsigma1s=s1s,
            dsigma1o=s1o,
            dsigma2s=s2s,
            dsigma2o=s2o,
        )
    for a1 in range(n):
        # _calc_e_c_a2 sums its argument; pass the accumulated sigma1 as a one-element array.
        calc._calc_e_c_a2(a1, np.array([sigma1[a1]]))
    for a1, p in ps.items():
        calc._calc_pairwise_forces(a1, **p)
    calc.energies -= calc.par["E0"][calc.ia2iz]
    return calc.energies.sum(), calc.forces_ev_per_ang


for name in ["cu7", "cu_al4"]:
    symbols, coords = STRUCTURES[name]
    atoms = Atoms(symbols, positions=coords)
    atoms.calc = AseEMT()
    e_ase, f_ase = atoms.get_potential_energy(), atoms.get_forces()
    calc = ref.EMT(tuple(symbols))
    with contextlib.redirect_stdout(io.StringIO()):
        e_full, f_full = calc._get_energy_and_forces_internal(coords.copy())
        e_half, f_half = half_list_energy_forces(calc, coords.copy())
    print(
        f"{name}: E_ase={e_ase:.8f} eV  E_ref(full list)={e_full:.8f}  E_ref(half list)={e_half:.8f}  "
        f"max|F_half - F_ase|={np.abs(f_half - f_ase).max():.2e}  max|F_full - F_ase|={np.abs(f_full - f_ase).max():.2e}"
    )
