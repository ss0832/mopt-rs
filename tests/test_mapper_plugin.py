"""An exploration method of the mapper written in Python, with nothing of mopt-rs changed
(``run_mapper -map_opt method=python:MODULE:CLASS``, :mod:`mopt_rs.mapper_plugins`).

On the Muller-Brown surface (``-os mb2d``) the class proposes, from the first minimum, a guess
beside the saddle point next to it and a straight first path towards the middle minimum; the
mapper optimizes the guess to the saddle point, relaxes the path by NEB, and joins the minima
by the IRCs. The settings of the method reach the class, and what it does wrong is said.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mopt_rs import _core

# The known stationary points of the Muller-Brown surface.
MINIMA = [(-0.558224, 1.441726), (0.623499, 0.028038), (-0.050011, 0.466694)]
SADDLES = [(-0.822002, 0.624313), (0.212487, 0.292988)]

METHOD = '''
class NearSaddle:
    """One task per minimum; from the first minimum a guess and a path."""

    def __init__(self, settings):
        self.offset = float(settings.get("offset", "0.03"))
        self.calls = 0

    def tasks(self, node):
        assert node["variables"] is not None and node["names"] == ["x", "y"]
        return [{"key": f"near:{node['id']}", "description": "a guess and a path",
                 "parameters": {"node": node["id"]}}]

    def propose(self, job):
        assert job["task"]["parameters"] == {"node": job["node"]}
        x, y = job["start"]["variables"]
        if abs(y - 1.441726) > 0.1:
            return {"notes": ["not the first minimum"]}
        guess = [-0.822002 + self.offset, 0.624313 - 0.02]
        end = (-0.050011 - 0.05, 0.466694 + 0.05)
        path = [[x + t / 8 * (end[0] - x), y + t / 8 * (end[1] - y)] for t in range(9)]
        return {"guesses": [guess], "paths": [path], "notes": ["a guess and a path"]}


class Broken:
    def __init__(self, settings):
        pass

    def tasks(self, node):
        return [{"key": "same"}, {"key": "same"}]

    def propose(self, job):
        return {}
'''


def _run(tmp_path: Path, method: str, name: str) -> tuple[int, Path]:
    (tmp_path / "start.txt").write_text("x -0.55\ny 1.44\n")
    settings = {
        "mapper_settings": {"max_iterations": 3, "rng_seed": 1},
        "step1_settings": {"othersoft": "mb2d", "NSTEP": 300},
        "step3_settings": {"tight_convergence_criteria": True, "calc_exact_hess": 1, "NSTEP": 300},
        "step4_settings": {
            "tight_convergence_criteria": True,
            "NSTEP": 300,
            "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(settings))
    out = tmp_path / name
    code = _core.run_cli(
        "run_mapper",
        [
            str(tmp_path / "start.txt"),
            "-cfg",
            str(tmp_path / "config.json"),
            "--output_dir",
            str(out),
            "-map_opt",
            f"method={method}",
            "method_offset=0.03",
        ],
    )
    return code, out


def _near(values: list[float], point: tuple[float, float]) -> bool:
    return math.dist(values, point) < 1e-3


def test_a_python_class_proposes_and_the_mapper_follows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "my_methods.py").write_text(METHOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    code, out = _run(tmp_path, "python:my_methods:NearSaddle", "run")
    assert code == 0
    network = json.loads((out / "reaction_network.json").read_text())
    from mopt_rs import MapperNetwork

    loaded = MapperNetwork((out / "reaction_network.json").read_text())
    assert loaded.tasks(0, method="python:my_methods:NearSaddle")[0]["key"] == "near:0"
    assert loaded.select("boltzmann", [(0, 0, 1)]) == 0
    assert loaded.select("boltzmann", [(0, 0, 0)]) is None
    loaded.save(tmp_path / "copy")
    assert (
        json.loads((tmp_path / "copy" / "reaction_network.json").read_text())["nodes"]
        == network["nodes"]
    )
    nodes = network["nodes"]
    saddles = [n["variables"] for n in nodes if n["kind"] == "TS"]
    minima = [n["variables"] for n in nodes if n["kind"] == "EQ"]
    # The saddle point the guess points to and the minimum on its other side.
    assert any(_near(s, SADDLES[0]) for s in saddles), saddles
    assert any(_near(m, MINIMA[2]) for m in minima), minima
    log = (out / "mopt.log").read_text()
    assert "python:my_methods:NearSaddle: a guess and a path" in log


def test_what_a_python_method_does_wrong_is_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "my_methods.py").write_text(METHOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    # Two tasks with one key, a class that is not there, a spec without a class.
    code, _ = _run(tmp_path, "python:my_methods:Broken", "broken")
    assert code != 0
    code, _ = _run(tmp_path, "python:my_methods:Nowhere", "nowhere")
    assert code != 0
    code, _ = _run(tmp_path, "python:my_methods", "no_class")
    assert code != 0
