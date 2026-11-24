"""Benchmark the CCQN saddle-region step (RS-P-RFO vs RS-I-RFO) on GFN2-xTB reactions.

Protocol of Wu and Wang (J. Chem. Theory Comput. 2025): for each reaction, locate a reference
transition state (TS), optimize the reactant minimum, and start CCQN from
``TS + alpha (min - TS)``. A run succeeds when it converges to a first-order saddle point (exactly one negative Hessian
eigenvalue) that is the reference TS: within 0.05 kcal/mol of its energy or within 0.05 Angstrom
RMSD (energies identify TSs that differ only by a permutation of equivalent atoms).

Reactions (structures and reactive atoms from MultiOptPy's test set): SN2 (OH- + CH3F),
Diels-Alder (butadiene + methyl acrylate), Claisen rearrangement (allyl vinyl ether).

Usage (after installing mopt-rs and tblite):
    OMP_NUM_THREADS=1 python tools/bench/ccqn_xtb.py <MultiOptPy test dir> <output.json>

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from tblite.interface import Calculator as Xtb

from mopt_rs import _core
from mopt_rs.optimize import Evaluation, Optimizer, Task, run

BOHR = _core.BOHR_TO_ANGSTROM
NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17}

# name: (file, AFIR pushes [(gamma, a, b)], reactive bonds for the CCQN axis), atoms 1-based.
REACTIONS: dict[str, tuple[str, list[tuple[float, int, int]], list[tuple[int, int]]]] = {
    "sn2": ("SN2.xyz", [(150.0, 6, 1)], [(1, 6), (1, 2)]),
    "diels_alder": ("diels_alder_rxn.xyz", [(120.0, 1, 11), (120.0, 13, 8)], [(1, 11), (13, 8)]),
    "claisen": ("claisen_rearrengment.xyz", [(200.0, 6, 12)], [(6, 12), (1, 9)]),
}


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, int, int]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    charge, mult = (int(x) for x in lines[1].split()[:2])
    symbols, xyz = [], []
    for line in lines[2 : 2 + n]:
        p = line.split()
        symbols.append(p[0])
        xyz.append([float(v) for v in p[1:4]])
    return symbols, np.array(xyz) / BOHR, charge, mult


class XtbCalculator:
    """GFN2-xTB through tblite; Hessians by central differences of analytic gradients."""

    def __init__(self, step: float = 5e-3) -> None:
        self.step = step
        self._calc: Any = None
        self._key: tuple[Any, ...] | None = None
        self.gradients = 0

    def _gradient(self, task: Task, x: np.ndarray) -> tuple[float, np.ndarray]:
        key = (tuple(task.symbols), task.charge, task.multiplicity)
        if self._calc is None or key != self._key:
            nums = np.array([NUMBERS[s] for s in task.symbols])
            self._calc = Xtb("GFN2-xTB", nums, x, charge=task.charge, uhf=task.multiplicity - 1)
            self._calc.set("verbosity", 0)
            self._key = key
        else:
            self._calc.update(x)
        r = self._calc.singlepoint()
        self.gradients += 1
        return float(r.get("energy")), np.asarray(r.get("gradient"))

    def evaluate(self, task: Task) -> Evaluation:
        x = task.positions_bohr
        e, g = self._gradient(task, x)
        h = None
        if task.need_hessian:
            n = x.size
            h = np.zeros((n, n))
            for k in range(n):
                xp, xm = x.copy().ravel(), x.copy().ravel()
                xp[k] += self.step
                xm[k] -= self.step
                gp = self._gradient(task, xp.reshape(-1, 3))[1].ravel()
                gm = self._gradient(task, xm.reshape(-1, 3))[1].ravel()
                h[:, k] = (gp - gm) / (2 * self.step)
            h = 0.5 * (h + h.T)
            self._gradient(task, x)  # restore the state at x
        return Evaluation(e, g, h)


def kabsch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate/translate ``p`` onto ``q`` (both (N, 3))."""
    pc, qc = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return pc @ r + q.mean(0)


def rmsd_angstrom(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((kabsch(p, q) - (q - q.mean(0))) ** 2, axis=1)))) * BOHR


def negative_modes(symbols: list[str], x: np.ndarray, charge: int, mult: int) -> int:
    task = Task("h", "xtb", symbols, x, charge, mult, True)
    h = XtbCalculator().evaluate(task).hessian
    assert h is not None
    # Remove translations and rotations.
    n = len(symbols)
    c = x - x.mean(0)
    basis = []
    for a in range(3):
        t = np.zeros((n, 3))
        t[:, a] = 1
        basis.append(t.ravel())
        rot = np.zeros((n, 3))
        rot[:, (a + 1) % 3] = -c[:, (a + 2) % 3]
        rot[:, (a + 2) % 3] = c[:, (a + 1) % 3]
        basis.append(rot.ravel())
    q, _ = np.linalg.qr(np.array(basis).T)
    proj = np.eye(3 * n) - q @ q.T
    ev = np.linalg.eigvalsh(proj @ h @ proj)
    return int(np.sum(ev < -1e-5))


def optimize(symbols: list[str], x: np.ndarray, charge: int, mult: int, **kw: Any) -> Any:
    afir = kw.pop("afir", [])
    opt = Optimizer(symbols, x.ravel().tolist(), charge=charge, multiplicity=mult, **kw)
    for gamma, a, b in afir:
        opt.add_afir(gamma, [a], [b])
    return run(opt, XtbCalculator())


def references(test_dir: Path, name: str) -> dict[str, Any]:
    fname, pushes, bonds = REACTIONS[name]
    symbols, x0, charge, mult = read_xyz(test_dir / fname)
    reactant = optimize(symbols, x0, charge, mult, methods=["rsirfo_bfgs"], max_iterations=500)
    # AFIR path from the reactant; the highest point on the real surface guesses the TS.
    path = optimize(
        symbols,
        reactant.positions_bohr,
        charge,
        mult,
        methods=["rsirfo_bfgs"],
        max_iterations=300,
        afir=pushes,
    )
    k = int(np.argmax(path.energies))
    guess = np.asarray(path.reports[k]["positions_bohr"]).reshape(-1, 3)
    ts = optimize(
        symbols,
        guess,
        charge,
        mult,
        methods=["rsirfo_bofill"],
        saddle_order=1,
        exact_hessian_interval=1,
        max_iterations=200,
        criteria="normal",
    )
    nimag = negative_modes(symbols, ts.positions_bohr, charge, mult)
    return {
        "symbols": symbols,
        "charge": charge,
        "mult": mult,
        "bonds": bonds,
        "reactant": reactant.positions_bohr.tolist(),
        "ts": ts.positions_bohr.tolist(),
        "ts_converged": ts.converged,
        "ts_negative_modes": nimag,
        "ts_energy": ts.energies[-1] if ts.energies else None,
        "barrier_kcal": (ts.energies[-1] - reactant.energies[-1]) * _core.HARTREE_TO_KCAL_MOL,
    }


def one_run(job: tuple[str, dict[str, Any], float, str, int | None]) -> dict[str, Any]:
    name, ref, alpha, method, fc = job
    symbols, charge, mult = ref["symbols"], ref["charge"], ref["mult"]
    ts = np.array(ref["ts"])
    rmin = kabsch(np.array(ref["reactant"]), ts)
    ts_c = ts - ts.mean(0)
    start = ts_c + alpha * (rmin - ts_c)
    t0 = time.time()
    try:
        res = optimize(
            symbols,
            start,
            charge,
            mult,
            methods=[method],
            saddle_order=1,
            exact_hessian_interval=fc,
            max_iterations=300,
            ccqn_bonds=[tuple(b) for b in ref["bonds"]],
            dissociation_distance_angstrom=None,
        )
        rmsd = rmsd_angstrom(res.positions_bohr, ts)
        nimag = negative_modes(symbols, res.positions_bohr, charge, mult) if res.converged else -1
        de = (res.energies[-1] - ref["ts_energy"]) * _core.HARTREE_TO_KCAL_MOL if res.energies else float("nan")
        first_order = res.converged and nimag == 1
        ok = first_order and (rmsd < 0.05 or abs(de) < 0.05)
        out = {
            "converged": res.converged,
            "iterations": res.iterations,
            "rmsd": rmsd,
            "negative_modes": nimag,
            "de_kcal": de,
            "first_order": first_order,
            "success": ok,
        }
    except Exception as e:  # noqa: BLE001 - record any failure of a run
        out = {"converged": False, "error": str(e), "success": False, "iterations": -1}
    out.update(reaction=name, alpha=alpha, method=method, fc=fc, seconds=time.time() - t0)
    return out


def main() -> None:
    test_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    refs_path = out_path.with_suffix(".refs.json")
    if refs_path.exists():
        refs = json.loads(refs_path.read_text())
    else:
        with Pool(len(REACTIONS)) as pool:
            refs = dict(
                zip(
                    REACTIONS,
                    pool.starmap(references, [(test_dir, n) for n in REACTIONS]),
                    strict=True,
                )
            )
        refs_path.write_text(json.dumps(refs))
    for name, r in refs.items():
        print(
            f"{name}: TS converged {r['ts_converged']}, negative modes {r['ts_negative_modes']}, "
            f"barrier {r['barrier_kcal']:.1f} kcal/mol",
            flush=True,
        )
    methods = [
        "ccqn",
        "ccqn_rsirfo",
        "ccqn_block_bofill",
        "ccqn_rsirfo_block_bofill",
        "ccqn_block_fsb",
        "ccqn_rsirfo_block_fsb",
    ]
    jobs = [
        (name, ref, alpha, m, fc)
        for name, ref in refs.items()
        if ref["ts_negative_modes"] == 1
        for alpha in (0.1, 0.3, 0.5)
        for m in methods
        for fc in (None, 5)
    ]
    with Pool(int(os.environ.get("BENCH_WORKERS", "14"))) as pool:
        results = pool.map(one_run, jobs, chunksize=1)
    out_path.write_text(json.dumps(results, indent=1))
    print(f"{'method':28s} {'fc':>5s} {'success':>9s} {'mean iters':>11s}")
    for m in methods:
        for fc in (None, 5):
            rs = [r for r in results if r["method"] == m and r["fc"] == fc]
            ok = [r for r in rs if r["success"]]
            mean = np.mean([r["iterations"] for r in ok]) if ok else float("nan")
            fo = sum(bool(r.get("first_order")) for r in rs)
            print(
                f"{m:28s} {str(fc):>5s} {len(ok):4d}/{len(rs):<4d} {mean:11.1f}  first-order {fo}"
            )


if __name__ == "__main__":
    main()
