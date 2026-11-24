"""Reference GFN0-xTB energies and gradients from the xtb program (``xtb --gfn 0 --grad``) for
the structures with elements beyond argon (other than Br and I alone), appended to
``tests/golden/gfn0_xtb.json``.

Usage (with an xtb binary whose ``share/xtb`` holds ``param_gfn0-xtb.txt``):

    python -I tools/golden/gfn0_xtb_binary.py tests/golden/gfn0_xtb.json \\
        tests/golden/gfn1_xtb.json XTB_BINARY WORK_DIR [--all]

The structures (Bohr), charges, and unpaired electrons are those of the records of
``gfn1_xtb.json`` with elements beyond argon other than Br and I (made by
``tools/golden/gfn1_xtb.py``), so the two methods are checked on the same molecules. Records
of ``gfn0_xtb.json`` that an xtb program made before are kept and only the structures they
lack are computed; with ``--all`` every such structure is computed again and replaces the
record of its name. Each new record names its program (``source``). xtb runs at ``--acc 0.0001`` with one thread; its GFN0 is
not self-consistent, so the accuracy only sets the integral cutoffs. On the records that
gfn0-rs made (H to Ar, Br, I), xtb 6.7.1 agrees to 6e-11 Hartree in the energy and 9e-10
Hartree/Bohr in the gradient.

The geometries are written with 12 decimals in Angstrom (0.52917721067 Angstrom/Bohr) and the
Bohr positions stored are those read back, so a consumer of the file sees the structure xtb
computed.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import json
import os
import subprocess
import sys
from pathlib import Path

BOHR_IN_ANGSTROM = 0.52917721067
LIGHT = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar Br I".split())


def run_xtb(xtb: str, work: Path, symbols, pos_bohr, charge: int, uhf: int):
    work.mkdir(parents=True, exist_ok=True)
    lines = [str(len(symbols)), "reference"]
    written = []
    for k, s in enumerate(symbols):
        xyz = [round(pos_bohr[3 * k + c] * BOHR_IN_ANGSTROM, 12) for c in range(3)]
        written += [v / BOHR_IN_ANGSTROM for v in xyz]
        lines.append(f"{s} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}")
    (work / "in.xyz").write_text("\n".join(lines) + "\n")
    for name in ("gradient", "energy"):
        (work / name).unlink(missing_ok=True)
    share = Path(xtb).resolve().parent.parent / "share" / "xtb"
    env = dict(os.environ, OMP_NUM_THREADS="1", XTBPATH=str(share))
    run = subprocess.run(
        [xtb, "in.xyz", "--gfn", "0", "--grad", "--chrg", str(charge), "--uhf", str(uhf),
         "--acc", "0.0001"],
        cwd=work, capture_output=True, text=True, env=env, check=False,
    )
    gradient_file = work / "gradient"
    if not gradient_file.exists():
        raise SystemExit("xtb failed:\n" + run.stdout[-2000:] + run.stderr[-2000:])
    text = gradient_file.read_text().split("\n")
    energy = float(text[1].split()[6])
    n = len(symbols)
    grad = []
    for line in text[2 + n : 2 + 2 * n]:
        grad += [float(v.replace("D", "E")) for v in line.split()]
    version = next((ln.strip() for ln in run.stdout.splitlines() if "* xtb version" in ln), "xtb")
    return energy, grad, written, version


def main() -> None:
    out_path, source_path, xtb, work = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    doc = json.load(open(out_path))
    source = json.load(open(source_path))
    new = [r for r in source["systems"] if set(r["symbols"]) - LIGHT]
    if "--all" not in sys.argv[5:]:
        # Keep the records made before; compute only the structures they lack.
        have = {c["molecule"] for c in doc["cases"] if "source" in c}
        new = [r for r in new if r["name"] not in have]
    names = {r["name"] for r in new}
    cases = [c for c in doc["cases"] if c["molecule"] not in names]
    for r in new:
        energy, grad, pos, version = run_xtb(
            xtb, work, r["symbols"], r["positions_bohr"], r["charge"], r["uhf"]
        )
        cases.append(
            {
                "molecule": r["name"],
                "symbols": r["symbols"],
                "positions_bohr": pos,
                "charge": r["charge"],
                "multiplicity": r["uhf"] + 1,
                "energy": energy,
                "gradient": grad,
                "source": version,
            }
        )
        print(f"{r['name']:16s} E = {energy:.12f}", flush=True)
    doc["cases"] = cases
    doc["source_note"] = (
        "records without 'source' come from gfn0-rs 0.4.1 (Gfn0NativeCalculator); the others "
        "name the xtb program that made them (tools/golden/gfn0_xtb_binary.py)"
    )
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
