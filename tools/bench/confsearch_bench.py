"""Measure the conformation search: conformers found per gradient for each strategy and
sampling priority (the numbers of docs/source/searches.md, "Conformation search").

The `mopt` binary runs `confsearch` with the built-in GFN-FF on the structures in a
directory (n-butane, n-pentane, n-hexane, Ace-Ala-NMe, ...) under several settings and seeds;
each run's conformers (relative energies) and gradient evaluations are read from its
`conformers.csv` and `mopt.log`. The reference set of a molecule is the union of the
conformers of all its runs (energies closer than 0.05 kcal/mol are one), so that "found" is
the share of the conformers any setting found.

Usage:

    python -I tools/bench/confsearch_bench.py MOPT_BINARY STRUCTURES_DIR WORK_DIR         [CASES [MOLECULES [SEEDS]]]

CASES (optional) selects settings by name (comma-separated, see SETTINGS below), MOLECULES
the structures (`<name>.xyz` in STRUCTURES_DIR), SEEDS the seeds (comma-separated; 1,2,3).
Runs already in WORK_DIR (from separate invocations, run side by side) are read again, so
that the last invocation prints the table of all of them.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

# name -> confsearch options (besides the structure and -os gfnff).
# The AFIR settings name their pair rule, so that the table does not change with the default.
SETTINGS: dict[str, list[str]] = {
    "afir_boltzmann": ["-ms", "50", "-cs_opt", "pairs=multioptpy"],
    "afir_tabu": ["-ms", "50", "-tabu", "-cs_opt", "pairs=multioptpy"],
    "afir_water": ["-ms", "50", "-cs_opt", "pairs=multioptpy", "priority=water_filling"],
    "afir_water_adaptive": [
        "-ms",
        "50",
        "-cs_opt",
        "pairs=multioptpy",
        "priority=water_filling_adaptive",
    ],
    "afir_bonds3": ["-ms", "50", "-cs_opt", "pairs=bonds:3"],
    "imtd_quick": [
        "-cs_opt",
        "method=imtd",
        "mtd_runs=3:1.3,1.5:0.78,0.75:0.468,3:0.281",
        "mtd_time=2",
    ],
    "imtd_crest": ["-cs_opt", "method=imtd"],
    # The searches by moves, with the trials and the termination of the AFIR sampling.
    "move_torsion": ["-ms", "50", "-cs_opt", "method=torsion"],
    "move_kick": ["-ms", "50", "-cs_opt", "method=kick"],
    "move_lowmode": ["-ms", "50", "-cs_opt", "method=lowmode"],
}
MOLECULES = ["butane", "pentane", "hexane", "dipeptide"]
SEEDS = [1, 2, 3]


def run(binary: Path, xyz: Path, work: Path, options: list[str], seed: int) -> dict[str, object]:
    runs = sorted(p for p in work.rglob("*_CS_*") if p.is_dir()) if work.is_dir() else []
    if runs and (runs[-1] / "conformers.csv").is_file():
        return read(runs[-1], 0.0)
    work.mkdir(parents=True, exist_ok=True)
    args = [str(binary), "confsearch", str(xyz), "-os", "gfnff", *options]
    if "-cs_opt" in args:
        args += [f"seed={seed}"]
    else:
        args += ["-cs_opt", f"seed={seed}"]
    start = time.time()
    done = subprocess.run(args, cwd=work, capture_output=True, text=True, check=False)
    elapsed = time.time() - start
    runs = sorted(p for p in work.rglob("*_CS_*") if p.is_dir())
    if done.returncode != 0 or not runs:
        return {"error": done.stdout[-500:] + done.stderr[-500:]}
    return read(runs[-1], elapsed)


def read(run_dir: Path, elapsed: float) -> dict[str, object]:
    log = (run_dir / "mopt.log").read_text()
    m = re.search(r"Evaluations \(this input\): (\d+) gradient calculation\(s\), (\d+) exact", log)
    rows = list(csv.DictReader((run_dir / "conformers.csv").open()))
    return {
        "energies": sorted(float(r["energy_hartree"]) for r in rows),
        "gradients": int(m.group(1)) if m else 0,
        "hessians": int(m.group(2)) if m else 0,
        "seconds": elapsed,
    }


def main() -> None:
    binary, structures, work = (Path(a).resolve() for a in sys.argv[1:4])
    cases = sys.argv[4].split(",") if len(sys.argv) > 4 else list(SETTINGS)
    if len(sys.argv) > 5:
        MOLECULES[:] = sys.argv[5].split(",")
    if len(sys.argv) > 6:
        SEEDS[:] = [int(s) for s in sys.argv[6].split(",")]
    results: dict[tuple[str, str, int], dict[str, object]] = {}
    for mol in MOLECULES:
        xyz = structures / f"{mol}.xyz"
        if not xyz.is_file():
            continue
        for case in cases:
            for seed in SEEDS:
                out = run(binary, xyz, work / mol / case / str(seed), SETTINGS[case], seed)
                results[(mol, case, seed)] = out
                print(
                    mol, case, seed, out.get("gradients"), len(out.get("energies", [])), flush=True
                )
    kcal = 627.5094740631  # CODATA 2018, Units::HARTREE_TO_KCAL_MOL
    print(
        "\n| molecule | setting | conformers found (mean of seeds) | share of the union | "
        "gradients (mean) | Hessians (mean) | conformers per 1000 gradients |"
    )
    print("|---|---|---|---|---|---|---|")
    for mol in MOLECULES:
        union: list[float] = []
        for (m, _, _), out in results.items():
            if m != mol:
                continue
            for e in out.get("energies", []):  # type: ignore[union-attr]
                if all(abs(e - u) * kcal > 0.05 for u in union):
                    union.append(e)
        if not union:
            continue
        for case in cases:
            outs = [results.get((mol, case, s), {}) for s in SEEDS]
            outs = [o for o in outs if "energies" in o]
            if not outs:
                continue
            found = [len(o["energies"]) for o in outs]  # type: ignore[arg-type]
            grads = [int(o["gradients"]) for o in outs]  # type: ignore[call-overload]
            mean_found = sum(found) / len(found)
            mean_grad = sum(grads) / len(grads)
            hess = sum(int(o.get("hessians", 0)) for o in outs) / len(outs)  # type: ignore[call-overload]
            per = 1000.0 * sum(found) / max(1, sum(grads))
            print(
                f"| {mol} | {case} | {mean_found:.1f} | {mean_found / len(union):.2f} "
                f"({len(union)}) | {mean_grad:.0f} | {hess:.1f} | {per:.2f} |"
            )


if __name__ == "__main__":
    main()
