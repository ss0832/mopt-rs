"""Reference data of MultiOptPy's IRC parts that are not affected by its bugs.

Usage (inside the reference environment created by setup_ref_env.sh):

    python tools/golden/irc_reference.py <multioptpy_root> <output_dir>

Writes ``irc_parts.json`` (units: Bohr, Hartree, amu) with

* ``lqa_steps``: ``IRC/lqa.py:LQA.step`` for given mass-weighted Hessians and gradients (no
  earlier points, so no Hessian update: the step formula alone);
* ``dwi``: energies and gradients of ``IRC/hpc.py:DWISurface`` and the end points of
  ``IRC/hpc.py:corrector_step``;
* ``rk4``: a run of ``IRC/rk4.py:RK4`` on a Morse potential of three atoms (the calculator
  and the bias potential are replaced by stand-ins; the trajectory of RK4 depends on the
  gradients only).

The other parts of MultiOptPy's IRC are not used as references: the Hessian handling of LQA,
the mass weighting of HPC, the velocity Verlet step of DVV, and the Euler integrator have
errors that mopt-rs does not repeat (docs/source/migration_paths.md, "IRC").

MultiOptPy: `IRC/lqa.py`, `IRC/hpc.py`, `IRC/rk4.py`.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import contextlib
import csv
import io
import json
import os
import sys
import tempfile

import numpy as np

ELEMENTS = ["C", "N", "H"]
# Morse pairs (i, j, D, a, r0) in atomic units.
MORSE = [(0, 1, 0.30, 1.2, 2.2), (0, 2, 0.17, 1.0, 2.0), (1, 2, 0.12, 0.9, 2.4)]
START = [[0.05, -0.10, 0.02], [2.45, 0.08, -0.03], [0.60, 2.30, 0.10]]


def morse(x):
    x = np.asarray(x, dtype=float).reshape(-1, 3)
    energy = 0.0
    gradient = np.zeros_like(x)
    for i, j, d, a, r0 in MORSE:
        v = x[i] - x[j]
        r = float(np.linalg.norm(v))
        e = np.exp(-a * (r - r0))
        energy += d * (1.0 - e) ** 2
        slope = 2.0 * d * a * (1.0 - e) * e
        gradient[i] += slope * v / r
        gradient[j] -= slope * v / r
    return energy, gradient


def lqa_steps(lqa_module, parameter):
    rng = np.random.default_rng(20261112)
    masses = np.array([parameter.atomic_mass(e) for e in ELEMENTS])
    sqrt_mass = np.sqrt(masses)
    cases = []
    for step_size, scale in [(0.05, 0.02), (0.1, 0.05), (0.3, 0.2), (0.5, 0.004)]:
        a = rng.normal(size=(9, 9))
        hessian = 0.3 * (a + a.T)
        gradient = scale * rng.normal(size=(3, 3))
        geometry = np.array(START) + 0.1 * rng.normal(size=(3, 3))
        with tempfile.TemporaryDirectory() as work:
            lqa = lqa_module.LQA(
                ELEMENTS,
                [0, 1],
                -1,
                work,
                work,
                {},
                max_step=10,
                step_size=step_size,
                init_coord=geometry.copy(),
                init_hess=hessian.copy(),
                calc_engine=None,
            )
            new = lqa.step(gradient.copy(), geometry.copy(), np.zeros((9, 9)), sqrt_mass)
        cases.append(
            {
                "step_size": step_size,
                "mw_hessian": hessian.ravel().tolist(),
                "mw_gradient": gradient.ravel().tolist(),
                "geometry": geometry.ravel().tolist(),
                "new_geometry": np.asarray(new).ravel().tolist(),
            }
        )
    return cases


def dwi(hpc_module):
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(3):
        x1 = rng.normal(size=(2, 3))
        x2 = x1 + 0.2 * rng.normal(size=(2, 3))
        points = []
        for x in (x1, x2):
            a = rng.normal(size=(6, 6))
            points.append((x, float(rng.normal()), 0.5 * rng.normal(size=(2, 3)), a + a.T))
        (x1, e1, g1, h1), (x2, e2, g2, h2) = points
        surface = hpc_module.DWISurface(x1, e1, g1, h1, x2, e2, g2, h2)
        probes = []
        for _ in range(4):
            x = x1 + rng.uniform(-0.5, 1.5) * (x2 - x1) + 0.05 * rng.normal(size=(2, 3))
            probes.append(
                {
                    "x": x.ravel().tolist(),
                    "energy": float(surface.get_energy(x.flatten())),
                    "gradient": surface.get_gradient(x.flatten()).ravel().tolist(),
                }
            )
        end = hpc_module.corrector_step(surface, x1.copy(), 0.05, 100)
        cases.append(
            {
                "first": {
                    "x": x1.ravel().tolist(),
                    "energy": e1,
                    "gradient": g1.ravel().tolist(),
                    "hessian": h1.ravel().tolist(),
                },
                "second": {
                    "x": x2.ravel().tolist(),
                    "energy": e2,
                    "gradient": g2.ravel().tolist(),
                    "hessian": h2.ravel().tolist(),
                },
                "probes": probes,
                "descent_length": 0.05,
                "descent_steps": 100,
                "descent_end": np.asarray(end).ravel().tolist(),
            }
        )
    return cases


class Engine:
    """Stand-in of MultiOptPy's calculator object: the Morse potential."""

    def __init__(self, bohr2angstroms):
        self.b2a = bohr2angstroms
        self.Model_hess = np.eye(9)
        self.FC_COUNT = -1
        self.hessian_flag = False
        self.gradient = None

    def single_point(self, _dir, _elements, _iter, _state, _method, geometry_angstrom):
        geometry = np.asarray(geometry_angstrom, dtype=float) / self.b2a
        energy, gradient = morse(geometry)
        self.gradient = gradient
        return energy, gradient, geometry, False


class NoBias:
    """Stand-in of BiasPotentialCalculation: no bias potential."""

    def __init__(self, _directory):
        pass

    def main(self, e, g, geometry, _elements, _force_data, _g, _iter, _initial):
        return None, e, g, np.zeros((geometry.size, geometry.size))


def rk4_run(rk4_module, parameter):
    b2a = parameter.UnitValueLib().bohr2angstroms
    rk4_module.BiasPotentialCalculation = NoBias
    step_size, max_step = 0.1, 14
    with tempfile.TemporaryDirectory() as work:
        run = rk4_module.RK4(
            ELEMENTS,
            [0, 1],
            -1,
            work,
            work,
            {},
            max_step=max_step,
            step_size=step_size,
            init_coord=np.array(START),
            init_hess=np.eye(9),
            calc_engine=Engine(b2a),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            run.run()
        with open(os.path.join(work, "irc_energies_gradients.csv"), newline="") as f:
            rows = list(csv.reader(f))
        with open(os.path.join(work, "irc_structures.xyz")) as f:
            lines = f.read().splitlines()
    frames = []
    for k in range(0, len(lines), 5):
        frames.append([[float(v) / b2a for v in lines[k + 2 + i].split()[1:4]] for i in range(3)])
    return {
        "elements": ELEMENTS,
        "morse": [list(m) for m in MORSE],
        "start": START,
        "step_size": step_size,
        "max_step": max_step,
        "header": rows[0],
        "steps": [int(r[0]) for r in rows[1:]],
        "energies": [float(r[1]) for r in rows[1:]],
        "rms_gradients": [float(r[3]) for r in rows[1:]],
        "structures": [np.asarray(f).ravel().tolist() for f in frames],
    }


def main():
    root, out_dir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, root)
    with contextlib.redirect_stdout(io.StringIO()):
        from multioptpy.IRC import hpc, lqa, rk4
        from multioptpy.Parameters import parameter

        data = {
            "about": "MultiOptPy 1.21.2: LQA.step, DWISurface, corrector_step, and an RK4 run "
            "(tools/golden/irc_reference.py)",
            "masses": [parameter.atomic_mass(e) for e in ELEMENTS],
            "lqa_steps": lqa_steps(lqa, parameter),
            "dwi": dwi(hpc),
            "rk4": rk4_run(rk4, parameter),
        }
    path = os.path.join(out_dir, "irc_parts.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
