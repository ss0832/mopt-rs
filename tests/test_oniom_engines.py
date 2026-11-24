"""ONIOM with a calculator of a Python engine as a lower level: an ASE calculator (ASE, or the
stand-in of ``fake_ase``) below a built-in one, with microiterations, and the settings of the
lower level (``-oniom_opt low.KEY=VALUE``).

MultiOptPy: no counterpart (its ``-oniom`` is read and not carried out).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import fake_ase
from mopt_rs import _core

HAS_ASE = importlib.util.find_spec("ase") is not None
HARTREE_EV = _core.HARTREE_TO_EV
EPSILON_EV = 0.010277

AR7 = [
    (1.9500, 0.0000, 0.0000),
    (0.6026, 1.8546, 0.0300),
    (-1.5776, 1.1462, 0.0700),
    (-1.5776, -1.1462, 0.1100),
    (0.6026, -1.8546, 0.1500),
    (0.0000, 0.0000, 2.1000),
    (0.2000, 0.0000, -2.0000),
]


@pytest.fixture
def an_ase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not HAS_ASE:
        monkeypatch.setitem(sys.modules, "ase", fake_ase.module())
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(directory: Path, name: str, extra: list[str]) -> tuple[int, Path | None]:
    """``optmain`` on Ar7 with the Lennard-Jones potential of mopt-rs as the high level of
    the first three atoms and the ASE one below it; the exit code and the run directory."""
    work = directory / name
    work.mkdir()
    text = "7\n\n" + "".join(f"Ar {x} {y} {z}\n" for x, y, z in AR7)
    (work / "ar7.xyz").write_text(text)
    arguments = [
        str(work / "ar7.xyz"),
        "-os",
        "lj",
        "-oniom",
        "1-3",
        "none",
        "ase:fake_ase:PairLennardJones",
        *extra,
    ]
    code = _core.run_cli("optmain", arguments)
    runs = [p for p in directory.rglob("ar7_OPT_ASE_*") if p.is_dir() and name in p.parts]
    return code, (runs[0] if runs else None)


def _last_energy(run: Path) -> float:
    rows = (run / "energy_profile.csv").read_text().strip().splitlines()
    return float(rows[-1].split(",")[3])


def test_an_ase_calculator_as_the_low_level(an_ase: Path) -> None:
    # The forces of argon are small against the usual thresholds: the tight ones.
    code, run = _run(an_ase, "same", ["-tcc"])
    assert code == 0 and run is not None
    log = (run / "mopt.log").read_text()
    assert "converged!!!" in log
    # The environment (four atoms) is relaxed with the ASE calculator in every iteration.
    assert "ONIOM microiterations: 3 core atom(s), 4 environment atom(s)" in log
    # Both levels are the same potential, so the ONIOM energy is that of the whole cluster:
    # the optimization ends at one of the four minima of seven Lennard-Jones atoms (which
    # one depends on where the first relaxation of the environment takes it).
    energy = _last_energy(run) * HARTREE_EV / EPSILON_EV
    minima = (-16.505, -15.935, -15.593, -15.533)
    assert min(abs(energy - m) for m in minima) < 0.01, energy


def test_settings_of_the_low_level_reach_it_alone(an_ase: Path) -> None:
    _, same = _run(an_ase, "same", [])
    assert same is not None
    # A deeper well for the low level only (a keyword of the ASE calculator).
    code, deeper = _run(an_ase, "deeper", ["-oniom_opt", "low.epsilon=0.02"])
    assert code == 0 and deeper is not None
    assert _last_energy(deeper) < 1.3 * _last_energy(same)
    # A setting of the high level is not handed down: the ASE calculator has no `cutoff`.
    code, cut = _run(an_ase, "cut", ["-calc_opt", "cutoff=30"])
    assert code == 0 and cut is not None
    assert abs(_last_energy(cut) - _last_energy(same)) < 1.0e-4
    # A layer that does not exist, and the highest one, are refused.
    assert _run(an_ase, "bottom", ["-oniom_opt", "bottom.epsilon=0.02"])[0] != 0
    assert _run(an_ase, "high", ["-oniom_opt", "high.epsilon=0.02"])[0] != 0
