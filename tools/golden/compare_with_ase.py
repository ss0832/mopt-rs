"""Cross-check MultiOptPy's native EMT and Tersoff calculators against ASE.

Usage: python tools/golden/compare_with_ase.py <multioptpy_root>

This file is part of mopt-rs.
"""

import contextlib
import io
import sys

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from multioptpy.Calculator.emt_calculation_tools import EMTCore  # noqa: E402
from multioptpy.Calculator.tersoff_calculation_tools import TersoffCore  # noqa: E402
from run_reference import ANG2BOHR, STRUCTURES  # noqa: E402

EV = 27.211386245988  # eV per Hartree used by ASE


def report(label, core, calc, names):
    for name in names:
        symbols, coords = STRUCTURES[name]
        atoms = Atoms(symbols, positions=coords)
        atoms.calc = calc
        e_ase = atoms.get_potential_energy() / EV
        f_ase = atoms.get_forces() / EV * 0.52917721067
        with contextlib.redirect_stdout(io.StringIO()):
            out = core.calculate_energy_and_gradient(coords * ANG2BOHR, symbols)
        g = np.asarray(out["gradient"]).reshape(-1, 3)
        print(
            f"{label} {name}: E_mopt={out['energy']:.10f}  E_ase={e_ase:.10f}  "
            f"max|g + F_ase|={np.abs(g + f_ase).max():.3e}"
        )


report("EMT", EMTCore(), EMT(), ["cu7", "cu_al4"])
try:
    from ase.calculators.tersoff import Tersoff

    report(
        "Tersoff", TersoffCore(), Tersoff.from_lammps("Si.tersoff") if False else Tersoff(), ["si5"]
    )
except Exception as exc:  # noqa: BLE001
    print("Tersoff comparison unavailable:", repr(exc))
