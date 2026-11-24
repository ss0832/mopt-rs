"""``run_mapper`` on an ONIOM surface with OpenMM as the lower level: vinyl alcohol (GFN1-xTB)
with two water molecules described by an OpenMM system (harmonic bonds and angles, charges and
Lennard-Jones parameters of TIP3P for the water and of a simple set for the alcohol), relaxed by
the microiterations; the AFIR push between the CH2 carbon and the hydrogen of the hydroxy group
meets the rotation of the hydroxy group, and its transition state and IRC converge with the
water relaxed by OpenMM. Skipped where OpenMM is not installed (an optional dependency).

The same system with GFN-FF as the lower level is `crates/mopt-rs/tests/cli_mapper_oniom.rs`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

openmm = pytest.importorskip("openmm")

from mopt_rs import _core  # noqa: E402

# Vinyl alcohol (atoms 1 to 7, syn) and two waters (Angstrom).
ATOMS = [
    ("C", (0.0000, 0.0000, 0.0000)),
    ("C", (1.3300, 0.0000, 0.0000)),
    ("O", (2.0800, 1.1400, 0.0000)),
    ("H", (1.5500, 1.9500, 0.0000)),
    ("H", (-0.5400, 0.9300, 0.0000)),
    ("H", (-0.5400, -0.9300, 0.0000)),
    ("H", (1.8800, -0.9300, 0.0000)),
    ("O", (4.9500, 1.3000, 0.0000)),
    ("H", (3.9935, 1.2467, 0.0000)),
    ("H", (5.1379, 2.2394, 0.0000)),
    ("O", (-3.0000, -1.7000, 1.3000)),
    ("H", (-2.2201, -1.2581, 0.9620)),
    ("H", (-2.9106, -1.6493, 2.2525)),
]
MASSES = {"H": 1.008, "C": 12.011, "O": 15.999}
# Charges (e) and Lennard-Jones sigma (nm) and epsilon (kJ/mol): TIP3P for the waters, a simple
# set for the alcohol (its own terms cancel in the ONIOM energy; its charges and contacts are
# what the water feels).
ENOL = [
    (-0.40, 0.340, 0.36),
    (0.15, 0.340, 0.36),
    (-0.60, 0.307, 0.71),
    (0.42, 0.0, 0.0),
    (0.14, 0.250, 0.13),
    (0.14, 0.250, 0.13),
    (0.15, 0.250, 0.13),
]
WATER = [(-0.834, 0.315061, 0.6364), (0.417, 0.0, 0.0), (0.417, 0.0, 0.0)]
BONDS = [
    (0, 1, 0.133, 459000.0),
    (1, 2, 0.136, 400000.0),
    (2, 3, 0.097, 460000.0),
    (0, 4, 0.108, 307000.0),
    (0, 5, 0.108, 307000.0),
    (1, 6, 0.108, 307000.0),
]


def _system(path: Path) -> None:
    """The OpenMM system of the whole structure, serialized to ``path``."""
    system = openmm.System()
    for symbol, _ in ATOMS:
        system.addParticle(MASSES[symbol])
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    nonbonded = openmm.NonbondedForce()
    pairs = [(i, j) for i, j, _, _ in BONDS]
    for i, j, r0, k in BONDS:
        bonds.addBond(i, j, r0, k)
    for first in (7, 10):
        o, h1, h2 = first, first + 1, first + 2
        bonds.addBond(o, h1, 0.09572, 462750.0)
        bonds.addBond(o, h2, 0.09572, 462750.0)
        angles.addAngle(h1, o, h2, math.radians(104.52), 836.8)
        pairs += [(o, h1), (o, h2), (h1, h2)]
    for q, sigma, epsilon in ENOL + WATER + WATER:
        nonbonded.addParticle(q, sigma if sigma > 0 else 0.1, epsilon)
    # Every pair within a molecule is excluded (bonded and 1-3 pairs of the alcohol too).
    enol_pairs = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    for i, j in sorted(set(pairs + enol_pairs)):
        nonbonded.addException(i, j, 0.0, 0.1, 0.0)
    for force in (bonds, angles, nonbonded):
        system.addForce(force)
    path.write_text(openmm.XmlSerializer.serialize(system))


def _dihedral(x: list[float]) -> float:
    """The dihedral angle C1-C2-O3-H4 of a core (Bohr, flat), in degrees."""
    p = [x[3 * k : 3 * k + 3] for k in range(4)]

    def sub(a: list[float], b: list[float]) -> list[float]:
        return [a[i] - b[i] for i in range(3)]

    def dot(a: list[float], b: list[float]) -> float:
        return sum(a[i] * b[i] for i in range(3))

    def cross(a: list[float], b: list[float]) -> list[float]:
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    b0, b1, b2 = sub(p[0], p[1]), sub(p[2], p[1]), sub(p[3], p[2])
    n = math.sqrt(dot(b1, b1))
    u = [c / n for c in b1]
    v = sub(b0, [dot(b0, u) * c for c in u])
    w = sub(b2, [dot(b2, u) * c for c in u])
    return math.degrees(math.atan2(dot(cross(u, v), w), dot(v, w)))


def test_the_mapper_on_oniom_with_openmm_as_the_lower_level(tmp_path: Path) -> None:
    _system(tmp_path / "whole.xml")
    lines = "".join(f"{s} {x:.4f} {y:.4f} {z:.4f}\n" for s, (x, y, z) in ATOMS)
    (tmp_path / "enol_water2.xyz").write_text(f"{len(ATOMS)}\n0 1\n{lines}")
    oniom = {
        "oniom_flag": ["1-7", "auto", "openmm"],
        "oniom_options": [f"low.system={tmp_path / 'whole.xml'}", "low.platform=Reference"],
    }
    settings = {
        "mapper_settings": {
            "afir_gamma_kJmol": 400.0,
            "dist_upper_ang": 4.0,
            "max_iterations": 2,
            "rng_seed": 1,
            "active_atoms": [4, 1],
        },
        "top_n_candidates": 1,
        "step1_settings": {"othersoft": "gfn1xtb", "NSTEP": 300, **oniom},
        "step3_settings": {"othersoft": "gfn1xtb", "NSTEP": 200, "calc_exact_hess": 5, **oniom},
        "step4_settings": {
            "othersoft": "gfn1xtb",
            "NSTEP": 400,
            "intrinsic_reaction_coordinates": ["0.1", "300", "lqa"],
            **oniom,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(settings))
    out = tmp_path / "run"
    code = _core.run_cli(
        "run_mapper",
        [
            str(tmp_path / "enol_water2.xyz"),
            "-cfg",
            str(tmp_path / "config.json"),
            "--output_dir",
            str(out),
        ],
    )
    assert code == 0
    network = json.loads((out / "reaction_network.json").read_text())
    nodes = network["nodes"]
    # The cores (7 atoms) in the network, the whole system (13) beside them.
    assert len(network["elements"]) == 7
    assert len(network["whole_elements"]) == 13
    edges = network["ts_edges"]
    assert edges, network.get("warnings")
    edge = next(e for e in edges if 0 in e["ends"])
    turn = abs(_dihedral(nodes[edge["ts"]]["coords_bohr"]))
    assert 60.0 < turn < 120.0, turn
    ends = sorted(abs(_dihedral(nodes[k]["coords_bohr"])) for k in edge["ends"])
    assert ends[0] < 30.0 and ends[1] > 150.0, ends
