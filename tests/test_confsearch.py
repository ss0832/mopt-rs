"""The conformation search from Python: its parts one by one (bond graphs, the invariant RMSD,
the conformer ensemble, Z-matrices and crossing, the RMSD metadynamics bias) and the command
with a built-in calculator and with an ASE calculator (ASE, or the stand-in of ``fake_ase``).

MultiOptPy: ``Entrypoints/conformation_search.py``.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import fake_ase
from mopt_rs import _core

HAS_ASE = importlib.util.find_spec("ase") is not None
BOHR = _core.ANGSTROM_TO_BOHR

ETHANE = [
    ("C", (0.000, 0.000, 0.765)),
    ("C", (0.000, 0.000, -0.765)),
    ("H", (1.018, 0.000, 1.162)),
    ("H", (-0.509, 0.882, 1.162)),
    ("H", (-0.509, -0.882, 1.162)),
    ("H", (-1.018, 0.000, -1.162)),
    ("H", (0.509, -0.882, -1.162)),
    ("H", (0.509, 0.882, -1.162)),
]
SYMBOLS = [s for s, _ in ETHANE]
POSITIONS = [c * BOHR for _, p in ETHANE for c in p]


def permuted(order: list[int]) -> list[float]:
    """Positions with atom ``i`` taken from atom ``order[i]``."""
    xyz = np.reshape(POSITIONS, (-1, 3))
    return [float(v) for v in xyz[order].ravel()]


def test_bond_graph_and_isomorphism() -> None:
    g = _core.bond_graph(SYMBOLS, POSITIONS)
    assert len(g["bonds"]) == 7
    assert g["classes"][0] == g["classes"][1]
    assert len(set(g["classes"][2:])) == 1
    assert g["fragments"] == [list(range(1, 9))]
    swapped = permuted([1, 0, 5, 6, 7, 2, 3, 4])
    mapping = _core.graph_isomorphism(SYMBOLS, POSITIONS, swapped)
    assert mapping is not None and sorted(mapping) == list(range(1, 9))


def test_invariant_rmsd_finds_the_numbering() -> None:
    turned = permuted([0, 1, 2, 3, 4, 6, 7, 5])
    rmsd, perm, mirrored = _core.invariant_rmsd(SYMBOLS, POSITIONS, turned)
    assert rmsd < 1e-8 and not mirrored
    assert perm == [1, 2, 3, 4, 5, 8, 6, 7]
    fixed, _, _ = _core.invariant_rmsd(SYMBOLS, POSITIONS, turned, permutations=False)
    assert fixed > 0.5


def test_the_ensemble_tells_duplicates_rotamers_and_conformers_apart() -> None:
    e = _core.ConformerEnsemble()
    assert e.add(SYMBOLS, POSITIONS, -1.0, "input") == ("new", 0)
    assert e.add(SYMBOLS, POSITIONS, -1.0) == ("duplicate", 0)
    assert e.add(SYMBOLS, permuted([0, 1, 2, 3, 4, 6, 7, 5]), -1.0) == ("rotamer", 0)
    assert e.add(SYMBOLS, POSITIONS, -0.999) == ("new", 1)
    assert len(e) == 2
    t = e.thermo(298.15)
    kt = _core.BOLTZMANN_HARTREE_PER_K * 298.15
    w = math.exp(-0.001 / kt)
    assert abs(t["populations"][1] - w / (1 + w)) < 1e-12
    assert t["entropy"] > 0.0


def test_zmatrix_crossing_and_rmsd_metadynamics() -> None:
    rows, values = _core.zmatrix(SYMBOLS, POSITIONS)
    assert len(rows) == 8 and len(values) == 8
    assert rows[0][1:] == [0, 0, 0]
    # Crossing a structure with itself gives nothing new.
    assert _core.crossover_conformers(SYMBOLS, POSITIONS, [POSITIONS, POSITIONS]) == []
    bias = _core.RmsdMetadynamics(0.01, 1.3)
    bias.add_reference(SYMBOLS, POSITIONS)
    assert bias.references == 1
    energy, gradient = bias.evaluate(SYMBOLS, POSITIONS)
    assert abs(energy - 0.01) < 1e-12
    assert max(abs(g) for g in gradient) < 1e-12


BUTANE = """14
n-butane all-anti
C 0.000000 0.000000 0.000000
C 1.268427 0.855565 0.000000
C 2.536855 0.000000 0.000000
C 3.805282 0.855565 0.000000
H -0.875864 0.648816 0.000000
H -0.013866 -0.629150 0.889988
H -0.013866 -0.629150 -0.889988
H 1.268427 1.484886 0.889975
H 1.268427 1.484886 -0.889975
H 2.536855 -0.629321 -0.889975
H 2.536855 -0.629321 0.889975
H 4.681147 0.206750 0.000000
H 3.819149 1.484715 0.889988
H 3.819149 1.484715 -0.889988
"""


def test_confsearch_with_the_built_in_gfn_ff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "butane.xyz").write_text(BUTANE)
    args = ["butane.xyz", "-os", "gfnff", "-tgta", "1,13", "-ms", "2", "-cs_opt", "seed=3"]
    assert _core.run_cli("confsearch", args) == 0
    run = next(p for p in tmp_path.rglob("butane_100KJ_CS_*") if p.is_dir())
    rows = list(csv.DictReader((run / "conformers.csv").open()))
    assert len(rows) == 2
    assert float(rows[0]["boltzmann_weight"]) > float(rows[1]["boltzmann_weight"]) / 2.0


@pytest.fixture
def an_ase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not HAS_ASE:
        monkeypatch.setitem(sys.modules, "ase", fake_ase.module())
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_confsearch_with_an_ase_calculator(an_ase: Path) -> None:
    ar7 = [
        (1.9500, 0.0000, 0.0000),
        (0.6026, 1.8546, 0.0300),
        (-1.5776, 1.1462, 0.0700),
        (-1.5776, -1.1462, 0.1100),
        (0.6026, -1.8546, 0.1500),
        (0.0000, 0.0000, 2.1000),
        (0.2000, 0.0000, -2.0000),
    ]
    text = "7\n\n" + "".join(f"Ar {x} {y} {z}\n" for x, y, z in ar7)
    (an_ase / "ar7.xyz").write_text(text)
    args = ["ar7.xyz", "-os", "ase:fake_ase:PairLennardJones", "-ms", "3", "-cs_opt", "seed=2"]
    assert _core.run_cli("confsearch", args) == 0
    run = next(p for p in an_ase.rglob("ar7_100KJ_CS_*") if p.is_dir())
    rows = list(csv.DictReader((run / "conformers.csv").open()))
    assert rows
    # The pentagonal bipyramid of LJ7: -16.505 epsilon.
    lowest = float(rows[0]["energy_hartree"]) * _core.HARTREE_TO_EV
    assert abs(lowest / 0.010277 + 16.505) < 0.02
    assert "gradient calculation(s)" in (run / "mopt.log").read_text()


def test_a_search_with_a_move_of_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Python callable makes the start of every trial; the search does the rest."""
    import mopt_rs
    from mopt_rs.confsearch import conformer_search

    monkeypatch.chdir(tmp_path)
    (tmp_path / "butane.xyz").write_text(BUTANE)
    assert mopt_rs.conformer_move_names() == ["kick", "lowmode", "torsion"]
    seen: list[tuple[int, int, int]] = []

    def turn(
        symbols: list[str], positions_bohr: list[float], trial: int, conformer: int, seed: int
    ) -> tuple[list[float], str] | None:
        """The half of butane behind the central bond turned by 120 degrees times the trial."""
        seen.append((trial, conformer, seed))
        assert symbols[:4] == ["C", "C", "C", "C"] and len(positions_bohr) == 42
        if trial >= 2:
            return None
        x = np.reshape(positions_bohr, (-1, 3)).copy()
        axis = x[2] - x[1]
        axis /= np.linalg.norm(axis)
        angle = 2.0 * math.pi / 3.0 * (trial + 1)
        for a in (3, 9, 10, 11, 12, 13):
            v = x[a] - x[2]
            x[a] = (
                x[2]
                + v * math.cos(angle)
                + np.cross(axis, v) * math.sin(angle)
                + axis * float(np.dot(axis, v)) * (1.0 - math.cos(angle))
            )
        return [float(c) for c in x.ravel()], f"the end turned by {math.degrees(angle):.0f} degrees"

    args = ["butane.xyz", "-os", "gfnff", "-ms", "9", "-nost", "-cs_opt", "method=turn"]
    result = conformer_search(args, moves={"turn": turn})
    # Anti and gauche (the mirror image of gauche is the same conformer); the move ended it.
    assert result.count == 2
    assert result.energies[0] < result.energies[1]
    assert [t for t, _, _ in seen] == [0, 1, 2]
    assert Path(result.directory, "conformers.xyz").is_file()
    log = Path(result.directory, "mopt.log").read_text()
    assert "trials by the move turn" in log
    assert "nothing more to try" in log
    # A name of the command's own strategies cannot be taken, and what is not callable is
    # refused.
    with pytest.raises(ValueError, match="cannot be registered as 'afir'"):
        conformer_search(args, moves={"afir": turn})
    with pytest.raises(RuntimeError):
        _core.run_confsearch(args, {"turn": 3})  # type: ignore[dict-item]
    # The built-in moves through the same entry.
    built_in = conformer_search(
        ["butane.xyz", "-os", "gfnff", "-ms", "8", "-cs_opt", "method=torsion", "seed=2"]
    )
    assert built_in.count >= 2
