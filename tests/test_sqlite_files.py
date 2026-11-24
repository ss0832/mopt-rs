"""The SQLite database of a mapper run, checked with SQLite itself.

A short run of ``run_mapper`` on seven argon atoms (built-in Lennard-Jones) with
``-map_opt store=sqlite`` writes ``reaction_network.sqlite``; Python's ``sqlite3`` finds it
sound (``PRAGMA integrity_check``), its rows are those of ``reaction_network.json`` (labels,
energies to the last bit, coordinates from the binary column), and ``run_mapper --report`` writes
the page from the database.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

import pytest

from mopt_rs import _core

BOHR = 0.529177210903
AR7 = [
    [6.023853759546511, 0.0, -0.15549746403767273],
    [1.962006769746685, 5.809518090796886, -0.061011157764781325],
    [-4.854674003698997, 3.590479638369873, 0.03347514850811009],
    [-4.999838497421851, -3.5904796383698727, 0.12796145478100152],
    [1.6599768546465556, -5.809518090796886, 0.2224477610538929],
    [-0.08463505395523435, 0.0, 3.794030138169188],
    [0.2933101711363313, 0.0, -3.961405880709738],
]
CONFIG = {
    "mapper_settings": {"afir_gamma_kJmol": 2.0, "dist_upper_ang": 7.0, "max_iterations": 2},
    "top_n_candidates": 1,
    "step1_settings": {"othersoft": "lj", "NSTEP": 200},
    "step3_settings": {"tight_convergence_criteria": True, "calc_exact_hess": 1, "NSTEP": 100},
    "step4_settings": {
        "tight_convergence_criteria": True,
        "calc_exact_hess": 1,
        "NSTEP": 200,
        "intrinsic_reaction_coordinates": ["0.5", "100", "lqa"],
    },
}


def test_the_database_of_a_run_opens_in_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lines = "".join(f"Ar {x * BOHR:.10f} {y * BOHR:.10f} {z * BOHR:.10f}\n" for x, y, z in AR7)
    (tmp_path / "ar7.xyz").write_text(f"7\n0 1\n{lines}")
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    args = ["ar7.xyz", "-cfg", "config.json", "-map_opt", "store=sqlite", "--output_dir", "run"]
    assert _core.run_cli("run_mapper", args) == 0
    run = tmp_path / "run"
    doc = json.loads((run / "reaction_network.json").read_text())
    con = sqlite3.connect(run / "reaction_network.sqlite")
    assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
    assert {"format", "nodes", "ts_edges", "irc_points", "jobs", "warnings"} <= tables
    version = con.execute("SELECT value FROM format WHERE key = 'version'").fetchone()
    assert version == (1,)
    rows = con.execute(
        "SELECT id, label, energy_hartree, coords_bohr FROM nodes ORDER BY id"
    ).fetchall()
    assert len(rows) == len(doc["nodes"]) >= 1
    for (node_id, label, energy, blob), node in zip(rows, doc["nodes"], strict=True):
        assert (node_id, label, energy) == (node["id"], node["label"], node["energy_hartree"])
        coords = list(struct.unpack(f"<{len(blob) // 8}d", blob))
        assert coords == node["coords_bohr"]
    jobs = con.execute("SELECT count(*) FROM jobs").fetchone()[0]
    assert jobs == len(doc["jobs"])
    edges = con.execute("SELECT ts, end_1, end_2 FROM ts_edges ORDER BY id").fetchall()
    assert edges == [(e["ts"], e["ends"][0], e["ends"][1]) for e in doc["ts_edges"]]
    con.close()
    # The report from the database, next to it.
    report = ["--report", str(run / "reaction_network.sqlite"), "-report_opt", "file=db.html"]
    assert _core.run_cli("run_mapper", report) == 0
    page = (run / "db.html").read_text(encoding="utf-8")
    assert 'id="network-data"' in page and "EQ0" in page
