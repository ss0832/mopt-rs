"""Write tests/golden/hab11.json: the dimers of the HAB11 set and their reference couplings, for
the transfer integrals of the projection method (``kinetics::rate::transfer_integral``).

The structures are those of the supplementary material of A. Kubas, F. Hoffmann, A. Heck,
H. Oberhofer, M. Elstner, J. Blumberger, J. Chem. Phys. 2014, 140, 104105 (``hab11.txt``:
cofacial homodimers at 3.5, 4.0, 4.5, and 5.0 Angstrom); the reference couplings |H_ab| (meV,
hole transfer in the dimer cations) are those of the paper as listed in Table I of S. Manna,
J. Blumberger, J. M. L. Martin, L. Kronik, arXiv:1806.01808 (ten of the eleven dimers; benzene
is not listed there and is left out). The two molecules of each dimer are told apart by their
bonds.

Usage:
    python -I tools/golden/hab11.py <hab11.txt> <repository root>

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DISTANCES = [3.5, 4.0, 4.5, 5.0]
REFERENCE_MEV = {
    "ethylene": [519.20, 270.80, 137.60, 68.50],
    "acetylene": [460.70, 231.80, 114.80, 56.60],
    "cyclopropene": [536.60, 254.00, 118.40, 54.00],
    "cyclobutadiene": [462.70, 239.10, 121.70, 62.20],
    "cyclopentadiene": [465.80, 234.40, 114.30, 53.40],
    "furane": [440.30, 214.90, 101.80, 46.00],
    "pyrrole": [456.30, 228.60, 111.30, 52.20],
    "thiophene": [449.00, 218.90, 106.50, 54.40],
    "imidazole": [411.60, 202.80, 99.10, 49.70],
    "phenol": [375.00, 179.60, 85.20, 41.30],
}
COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}


def read_blocks(path: Path) -> dict[str, list[list[tuple[str, list[float]]]]]:
    """The structures of every molecule of the file: a name line ``<name>.xyz``, then XYZ
    blocks."""
    lines = [line.rstrip() for line in path.read_text().splitlines()]
    out: dict[str, list[list[tuple[str, list[float]]]]] = {}
    name = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith(".xyz"):
            name = line[: -len(".xyz")]
            out[name] = []
            i += 1
            continue
        if line.isdigit() and name is not None:
            n = int(line)
            atoms = []
            for row in lines[i + 2 : i + 2 + n]:
                parts = row.split()
                atoms.append((parts[0], [float(x) for x in parts[1:4]]))
            out[name].append(atoms)
            i += 2 + n
            continue
        i += 1
    return out


def molecules(atoms: list[tuple[str, list[float]]]) -> list[list[int]]:
    """The atoms of each molecule (bonded within 1.25 times the covalent radii)."""
    n = len(atoms)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in range(n):
        for q in range(p + 1, n):
            r = math.dist(atoms[p][1], atoms[q][1])
            if r < 1.25 * (COVALENT[atoms[p][0]] + COVALENT[atoms[q][0]]):
                parent[find(p)] = find(q)
    groups: dict[int, list[int]] = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values())


def centroid(atoms, indices):
    return [sum(atoms[i][1][c] for i in indices) / len(indices) for c in range(3)]


def main() -> None:
    source, root = Path(sys.argv[1]), Path(sys.argv[2])
    blocks = read_blocks(source)
    dimers = []
    for name, reference in REFERENCE_MEV.items():
        for atoms in blocks[name]:
            parts = molecules(atoms)
            if len(parts) != 2 or len(parts[0]) != len(parts[1]):
                raise SystemExit(f"{name}: not two equal molecules")
            d = math.dist(centroid(atoms, parts[0]), centroid(atoms, parts[1]))
            k = min(range(4), key=lambda j: abs(DISTANCES[j] - d))
            if abs(DISTANCES[k] - d) > 0.05:
                raise SystemExit(f"{name}: distance {d:.3f} is not one of the set")
            dimers.append(
                {
                    "name": name,
                    "distance_angstrom": DISTANCES[k],
                    "reference_mev": reference[k],
                    "symbols": [a[0] for a in atoms],
                    "positions_angstrom": [a[1] for a in atoms],
                    "molecules": [[i for i in part] for part in parts],
                }
            )
    out = {
        "description": __doc__.split("\n\n")[0],
        "structures": "A. Kubas et al., J. Chem. Phys. 2014, 140, 104105, supplementary material",
        "reference": "Table I of S. Manna et al., arXiv:1806.01808 (the HAB11 references)",
        "units": {"positions": "Angstrom", "coupling": "meV"},
        "carrier": "hole",
        "dimers": dimers,
    }
    path = root / "tests" / "golden" / "hab11.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}: {len(dimers)} dimers")


if __name__ == "__main__":
    main()
