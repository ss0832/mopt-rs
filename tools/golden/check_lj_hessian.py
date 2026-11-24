"""Check MultiOptPy's analytic LJ Hessian against finite differences of its own gradient.

Usage: python tools/golden/check_lj_hessian.py <multioptpy_root>

This file is part of mopt-rs.
"""

import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from multioptpy.Calculator.lj_calculation_tools import LennardJonesCore  # noqa: E402
from run_reference import ANG2BOHR, STRUCTURES  # noqa: E402

symbols, coords = STRUCTURES["ar7"]
core = LennardJonesCore()
x0 = coords * ANG2BOHR
H = core.calculate_hessian(x0.copy(), symbols)["hessian"]
h = 1e-4
fd = np.zeros_like(H)
for k in range(x0.size):
    xp = x0.reshape(-1).copy()
    xm = x0.reshape(-1).copy()
    xp[k] += h
    xm[k] -= h
    gp = core.calculate_energy_and_gradient(xp.reshape(-1, 3), symbols)["gradient"].reshape(-1)
    gm = core.calculate_energy_and_gradient(xm.reshape(-1, 3), symbols)["gradient"].reshape(-1)
    fd[:, k] = (gp - gm) / (2 * h)
print(
    f"max |H_analytic - H_fd| = {np.abs(H - fd).max():.3e}   (max |H_fd| = {np.abs(fd).max():.3e})"
)
