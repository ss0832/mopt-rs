"""Measure the sampling schemes and the duplicate tolerances of ``orientsearch``.

Schemes: distinct minima found per gradient, whether the lowest minimum of all runs was found,
and how many samples converged, for complexes of water, ammonia, and benzene with the built-in
GFN-FF and GFN0-xTB. Tolerances: the RMSD (after matching equivalent atoms and identical
molecules) and the energy difference between every pair of converged samples of one run with
duplicate removal switched off, for the default and the tight convergence criteria.

Usage:
    OMP_NUM_THREADS=1 python tools/bench/orientsearch_schemes.py schemes MOPT OUT [options]
    OMP_NUM_THREADS=1 python tools/bench/orientsearch_schemes.py tolerances MOPT OUT [options]
    python tools/bench/orientsearch_schemes.py report MOPT OUT     # the table of OUT again

Options: ``--systems``, ``--engines``, ``--schemes`` (names below), ``--samples``, ``--seeds``,
``--workers``, ``--criteria`` (of the samples: tight, the default of ``orientsearch``, normal,
loose), ``--optmain="-ns 500"`` (further optmain options of every sample, after ``=``).

Minima count as distinct when they keep the bonds of the input and the parts touch (contact
ratio up to 1.5); those with other bonds (a proton moved) and those whose parts are apart are
counted in their own columns. "Found lowest" compares with the lowest such minimum of all runs
of the system and engine.

``MOPT`` is the ``mopt`` binary; ``OUT`` an empty directory for the runs and the results
(``schemes.json``, ``tolerances.json``). ``tolerances`` needs the Python package ``mopt_rs``.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

# CODATA 2018, as mopt_rs::domain::Units (HARTREE_TO_KCAL_MOL, BOHR_TO_ANGSTROM).
HARTREE_KCAL = 627.5094740631
BOHR_ANGSTROM = 0.529177210903
WATER = [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)]
AMMONIA = [
    ("N", 0.0, 0.0, 0.1162),
    ("H", 0.0, 0.9377, -0.2711),
    ("H", 0.8121, -0.4689, -0.2711),
    ("H", -0.8121, -0.4689, -0.2711),
]


def benzene() -> list[tuple[str, float, float, float]]:
    atoms = []
    for k in range(6):
        t = 2.0 * np.pi * k / 6.0
        atoms.append(("C", 1.397 * np.cos(t), 1.397 * np.sin(t), 0.0))
    for k in range(6):
        t = 2.0 * np.pi * k / 6.0
        atoms.append(("H", 2.481 * np.cos(t), 2.481 * np.sin(t), 0.0))
    return atoms


def apart(
    *molecules: list[tuple[str, float, float, float]],
) -> list[tuple[str, float, float, float]]:
    """The molecules 8 Angstrom apart along x."""
    out = []
    for k, m in enumerate(molecules):
        out += [(s, x + 8.0 * k, y, z) for s, x, y, z in m]
    return out


SYSTEMS: dict[str, list[tuple[str, float, float, float]]] = {
    "water_dimer": apart(WATER, WATER),
    "water_trimer": apart(WATER, WATER, WATER),
    "water_tetramer": apart(WATER, WATER, WATER, WATER),
    "ammonia_water": apart(AMMONIA, WATER),
    "benzene_dimer": apart(benzene(), benzene()),
}

SCHEMES: dict[str, list[str]] = {
    "multioptpy": ["scheme=multioptpy", "clash=0"],
    "uniform": ["scheme=uniform", "clash=0.5"],
    "contact": ["scheme=contact", "contact=1.0,1.2"],
    "contact_close": ["scheme=contact", "contact=0.8,1.0"],
    "contact_far": ["scheme=contact", "contact=1.2,1.5"],
    "contact_refine": ["scheme=contact", "contact=1.0,1.2", "refine=0.5"],
    "ring": ["scheme=ring", "contact=1.0,1.2"],
    "ring_close": ["scheme=ring", "contact=0.8,1.0"],
}


def write_xyz(path: Path, atoms: list[tuple[str, float, float, float]]) -> None:
    lines = [str(len(atoms)), "orientsearch benchmark"]
    lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, x, y, z in atoms]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def one_run(job: dict[str, Any]) -> dict[str, Any]:
    """Run one search and summarize it."""
    where = Path(job["dir"])
    where.mkdir(parents=True, exist_ok=True)
    write_xyz(where / "input.xyz", SYSTEMS[job["system"]])
    command = [
        job["mopt"],
        "orientsearch",
        "input.xyz",
        "-os",
        job["engine"],
        "-nsample",
        str(job["samples"]),
        "-core",
        "1",
        *job["optmain"],
        "-orient_opt",
        f"seed={job['seed']}",
        "interaction=off",
        *job["options"],
    ]
    done = subprocess.run(command, cwd=where, capture_output=True, text=True, check=False)
    runs = sorted(where.glob("*/input_ORIENT_*"))
    if done.returncode != 0 or not runs:
        return {**job, "error": done.stdout[-2000:] + done.stderr[-2000:]}
    run = runs[-1]
    return {**job, "run": str(run), **summary(run)}


def summary(run: Path) -> dict[str, Any]:
    """The minima of a run (with the bonds of the input, with other bonds, with the parts
    apart) and its samples."""
    minima = read_csv(run / "orientsearch_minima.csv")
    samples = read_csv(run / "orientsearch_samples.csv")
    same = [m for m in minima if m["bonds"] == "same" and float(m["contact_ratio"]) <= 1.5]
    return {
        "distinct": len(same),
        "energies": [float(m["energy_hartree"]) for m in same],
        "hits": [int(m["hits"]) for m in same],
        "other_bonds": sum(m["bonds"] != "same" for m in minima),
        "apart": sum(m["bonds"] == "same" and float(m["contact_ratio"]) > 1.5 for m in minima),
        "samples": len(samples),
        "converged": sum(s["converged"] == "true" for s in samples),
        "gradients": sum(int(s["gradients"]) for s in samples),
        "hessians": sum(int(s["hessians"]) for s in samples),
    }


def schemes(args: argparse.Namespace) -> None:
    out = Path(args.out)
    jobs = []
    for system in args.systems:
        for engine in args.engines:
            for scheme in args.schemes:
                for seed in range(1, args.seeds + 1):
                    jobs.append(
                        {
                            "mopt": args.mopt,
                            "system": system,
                            "engine": engine,
                            "scheme": scheme,
                            "seed": seed,
                            "samples": args.samples,
                            "options": SCHEMES[scheme] + [f"criteria={args.criteria}"],
                            "optmain": args.optmain,
                            "dir": str(out / system / engine / scheme / f"seed{seed}"),
                        }
                    )
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(one_run, jobs))
    (out / "schemes.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    report(results, args.window)


def report(results: list[dict[str, Any]], window: float) -> None:
    """A table per system and engine: mean distinct minima, gradients, minima per 1000
    gradients, runs that found the lowest minimum of all runs, converged samples."""
    keys = sorted({(r["system"], r["engine"]) for r in results})
    print(
        "| system | engine | scheme | distinct | within window | other bonds | apart | gradients "
        "| per 1000 gradients | found lowest | samples in lowest | converged | failed runs |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for system, engine in keys:
        group = [r for r in results if (r["system"], r["engine"]) == (system, engine)]
        ok = [r for r in group if "error" not in r and r["energies"]]
        failed_runs = [r for r in group if "error" in r]
        for r in failed_runs:
            print(f"<!-- {system} {engine} {r['scheme']} seed {r['seed']}: {r['error'][-300:]} -->")
        if not ok:
            continue
        lowest = min(min(r["energies"]) for r in ok)
        for scheme in dict.fromkeys(r["scheme"] for r in group):
            runs = [r for r in ok if r["scheme"] == scheme]
            failed = sum(1 for r in group if r["scheme"] == scheme and "error" in r)
            if not runs:
                print(
                    f"| {system} | {engine} | {scheme} | - | - | - | - | - | - | - | - | - | {failed} |"
                )
                continue
            distinct = np.mean([r["distinct"] for r in runs])
            near = np.mean(
                [sum((e - lowest) * HARTREE_KCAL <= window for e in r["energies"]) for r in runs]
            )
            gradients = np.mean([r["gradients"] for r in runs])
            found = sum((min(r["energies"]) - lowest) * HARTREE_KCAL < 0.01 for r in runs)
            converged = np.mean([r["converged"] / r["samples"] for r in runs])
            other = np.mean([r["other_bonds"] for r in runs])
            in_lowest = sum(
                h
                for r in runs
                for e, h in zip(r["energies"], r["hits"], strict=True)
                if (e - lowest) * HARTREE_KCAL < 0.01
            ) / sum(r["samples"] for r in runs)
            apart = np.mean([r["apart"] for r in runs])
            print(
                f"| {system} | {engine} | {scheme} | {distinct:.1f} | {near:.1f} | {other:.1f} | "
                f"{apart:.1f} | {gradients:.0f} | {1000 * distinct / gradients:.2f} | "
                f"{found}/{len(runs)} | {100 * in_lowest:.0f}% | {100 * converged:.0f}% | {failed} |"
            )


def tolerances(args: argparse.Namespace) -> None:
    """Pairs of converged samples of one run each without duplicate removal."""
    from mopt_rs import _core

    out = Path(args.out)
    summary = []
    for system in args.systems:
        for engine in args.engines:
            for label, criteria in (("normal", "criteria=normal"), ("tight", "criteria=tight")):
                job = {
                    "mopt": args.mopt,
                    "system": system,
                    "engine": engine,
                    "scheme": "contact",
                    "seed": 1,
                    "samples": args.samples,
                    "options": ["rmsd=0", "energy=0", criteria],
                    "optmain": [],
                    "dir": str(out / "tolerances" / system / engine / label),
                }
                r = one_run(job)
                if "error" in r:
                    summary.append({**r})
                    continue
                run = Path(r["run"])
                structures = []
                for s in read_csv(run / "orientsearch_samples.csv"):
                    if s["converged"] != "true":
                        continue
                    path = run / s["directory"] / f"input_Sample_{s['sample']}_optimized.xyz"
                    lines = path.read_text(encoding="utf-8").splitlines()
                    n = int(lines[0])
                    rows = [line.split() for line in lines[2 : 2 + n]]
                    symbols = [row[0] for row in rows]
                    xyz = np.array([[float(v) for v in row[1:4]] for row in rows]) / BOHR_ANGSTROM
                    structures.append((float(s["energy_hartree"]), symbols, xyz.ravel().tolist()))
                pairs = []
                for i in range(len(structures)):
                    for j in range(i):
                        ei, sym, xi = structures[i]
                        ej, _, xj = structures[j]
                        c = _core.compare_structures(sym, xi, xj)
                        rmsd = None if c["rmsd"] is None else c["rmsd"] * BOHR_ANGSTROM
                        pairs.append({"de": abs(ei - ej), "rmsd": rmsd})
                summary.append({**r, "criteria": label, "pairs": pairs})
                close = [p for p in pairs if p["rmsd"] is not None and p["rmsd"] < 0.5]
                print(
                    f"{system} {engine} {label}: {len(structures)} structures, "
                    f"{r['gradients']} gradients; pairs with RMSD < 0.5 A: "
                    + ", ".join(
                        f"{p['rmsd']:.3f} A/{p['de']:.1e} Eh"
                        for p in sorted(close, key=lambda p: p["rmsd"])
                    )
                )
    (out / "tolerances.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=["schemes", "tolerances", "report"])
    p.add_argument("mopt")
    p.add_argument("out")
    p.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    p.add_argument("--engines", nargs="*", default=["gfnff", "gfn0xtb"])
    p.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--window", type=float, default=1.0, help="kcal/mol above the lowest")
    p.add_argument(
        "--criteria", default="tight", choices=["tight", "normal", "loose"], help="of the samples"
    )
    p.add_argument(
        "--optmain", default="", help="optmain options of every sample, e.g. '-tcc -ns 500'"
    )
    args = p.parse_args()
    args.optmain = args.optmain.split()
    if args.mode == "schemes":
        schemes(args)
    elif args.mode == "report":
        # The table again from the runs of an earlier `schemes` in OUT.
        results = json.loads((Path(args.out) / "schemes.json").read_text(encoding="utf-8"))
        results = [r if "error" in r else {**r, **summary(Path(r["run"]))} for r in results]
        report(results, args.window)
    else:
        tolerances(args)


if __name__ == "__main__":
    main()
