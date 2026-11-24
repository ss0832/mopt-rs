"""Crossing points with an engine of the user's own, with nothing of mopt-rs changed.

The searches for crossing points of ``run_mapper`` (``states=...``) and ``run_automecp`` take
their energies and gradients from the calculator of the run like every other command. With
``-os engine:module:Class`` that is a Python class found by where it is imported from, and what
it has to hand over is

- ``compute(structure, hessian)`` for the charge and multiplicity of ``structure.state`` (a
  crossing of states of different spin, MECP) and for the root ``structure.excited_state`` (a
  conical intersection of two roots, MECI);
- ``state_characters(structure, count)`` when the roots are to be followed from structure to
  structure and from node to node (``follow_states``).

The engines of :mod:`crossing_engine` are a model of three atoms with two surfaces whose
crossing points are known.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from crossing_engine import AS, R1, R2, RC, C, D, seam_minimum
from mopt_rs import _core

BOHR = 0.529177210903


def _steps(engine: str, **step1: Any) -> dict[str, Any]:
    return {
        "step1_settings": {"othersoft": engine, "NSTEP": 300, **step1},
        "step3_settings": {"tight_convergence_criteria": True, "calc_exact_hess": 1, "NSTEP": 300},
        "step4_settings": {
            "tight_convergence_criteria": True,
            "NSTEP": 300,
            "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"],
        },
    }


def _start(path: Path) -> None:
    """The minimum of the first surface at ``r12 = R1`` (helium atoms: no pair is bonded
    for the mapper, so every pair is an AFIR push)."""
    x3 = (R1 * R1 + C * C - AS[0] * AS[0]) / (2.0 * R1)
    y3 = math.sqrt(C * C - x3 * x3)
    atoms = [(0.0, 0.0, 0.0), (R1, 0.0, 0.0), (x3, y3, 0.0)]
    lines = "".join(f"He {x * BOHR:.10f} {y * BOHR:.10f} {z * BOHR:.10f}\n" for x, y, z in atoms)
    (path / "start.xyz").write_text(f"3\n\n{lines}")


def _distance(node: dict[str, Any], i: int, j: int) -> float:
    x = node["coords_bohr"]
    return math.dist(x[3 * i : 3 * i + 3], x[3 * j : 3 * j + 3])


def _network(run: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((run / "reaction_network.json").read_text())
    return doc


def test_the_mapper_finds_the_crossing_points_of_two_spin_states_of_an_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _start(tmp_path)
    config = {
        "mapper_settings": {"max_iterations": 3, "rng_seed": 1, "channels": False},
        **_steps("engine:crossing_engine:Surfaces"),
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = ["start.xyz", "-cfg", "config.json", "--output_dir", "run"]
    args += ["-map_opt", "states=0,1;0,3", "ts_states=none"]
    assert _core.run_cli("run_mapper", args) == 0
    net = _network(tmp_path / "run")
    crossings = [n for n in net["nodes"] if n["kind"] == "MECP"]
    assert crossings, [n["label"] for n in net["nodes"]]
    # The crossing point between R1 and RC lies at the minimum of the seam there.
    lowest = seam_minimum(R1 + 1e-6, RC - 1e-6)
    assert any(abs(n["energy_hartree"] - lowest) < 1e-6 for n in crossings)
    # Each is joined to a minimum of the singlet and one of the triplet (at RC, D above).
    for edge in net["crossing_edges"]:
        ends = [net["nodes"][k] for k in edge["ends"]]
        assert [e["state"]["multiplicity"] for e in ends] == [1, 3]
        assert abs(_distance(ends[1], 0, 1) - RC) < 1e-3
        assert abs(ends[1]["energy_hartree"] - D) < 1e-7
        # The way between the two minima through the crossing point, in one file.
        frames = (tmp_path / "run" / edge["irc"]["file"]).read_text().splitlines()
        comments = [line for line in frames if "energy=" in line]
        assert comments[0].startswith("PES1 minimization")
        assert comments[-1].startswith("PES2 minimization")
        assert any(line.startswith("MECP") for line in comments)


def test_run_automecp_finds_a_conical_intersection_of_two_roots_of_an_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _start(tmp_path)
    # The steps of run_automecp: the path, the crossing point, the meta-IRC and its ends.
    steps = _steps("engine:crossing_engine:Surfaces")
    config = {
        "step1_settings": steps["step1_settings"],
        "step2_settings": steps["step3_settings"],
        "step3_settings": steps["step4_settings"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = ["start.xyz", "-cfg", "config.json", "--output_dir", "out"]
    # A push of the first two atoms towards the first excited state of the singlet.
    args += ["-ma", "100", "1", "2", "-state2", "0", "1", "1"]
    assert _core.run_cli("run_automecp", args) == 0
    out = tmp_path / "out"
    rows = (out / "automecp_summary.csv").read_text().splitlines()
    assert len(rows) >= 2 and rows[1].startswith("MECI0,MECI,PES1,PES2,"), rows
    energy = float(rows[1].split(",")[4])
    assert abs(energy - seam_minimum(R1 + 1e-6, RC - 1e-6)) < 1e-6
    # Its page holds its structures and links to no file of the run.
    page = (out / "automecp_report.html").read_text(encoding="utf-8")
    assert "Step 3 · Meta-IRC and minima" in page and "MECI0_path" in page
    assert 'href="mapper/' not in page and 'href="jobs/' not in page


def test_the_roots_of_an_engine_are_followed_from_node_to_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _start(tmp_path)
    # The two excited states of this engine are counted by their energies and change places
    # between the minima; it hands out their characters, and the run follows them.
    config = {
        "mapper_settings": {"max_iterations": 3, "rng_seed": 1, "channels": False},
        **_steps("engine:crossing_engine:Ordered", follow_states=True),
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = ["start.xyz", "-cfg", "config.json", "--output_dir", "run"]
    args += ["-map_opt", "states=0,1,1;0,1,2", "ts_states=none"]
    assert _core.run_cli("run_mapper", args) == 0
    net = _network(tmp_path / "run")
    minima = [n for n in net["nodes"] if n["kind"] == "EQ"]
    # The input, on the first state, where that state is the lowest excited state.
    assert minima[0]["state"]["excited_state"] == 1
    assert [1, 0] in minima[0]["followed_states"]
    # The minimum of the single well at RC lies on the second state of the run, which is the
    # lowest excited state there: the state was followed through the crossing.
    at_rc = [n for n in minima if abs(_distance(n, 0, 1) - RC) < 1e-3]
    assert at_rc, [(n["label"], _distance(n, 0, 1)) for n in minima]
    assert at_rc[0]["state"]["excited_state"] == 2
    assert [2, 0] in at_rc[0]["followed_states"]
    assert abs(at_rc[0]["energy_hartree"] - D) < 1e-7
    assert any(n["kind"] == "MECI" for n in net["nodes"])
    # No minimum near R2 on the second state: counted by their energies, the second state
    # there would be the single well again.
    assert not any(
        n["state"]["excited_state"] == 2 and abs(_distance(n, 0, 1) - R2) < 0.5 for n in minima
    )
