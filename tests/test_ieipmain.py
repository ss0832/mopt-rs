"""`ieipmain` from Python: the searches on the Muller-Brown surface (functions and the ask/tell
classes), the command with an ASE calculator (the stand-in of ``fake_ase``), and what it
refuses.

MultiOptPy: `ieip.py:iEIP`, `OtherMethod/`.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fake_ase
from mopt_rs import _core
from mopt_rs.ieip import (
    Dimer,
    ElasticImagePair,
    IeipJob,
    addf_search,
    dimer_search,
    dsafir_search,
    fsm_search,
    gsm_search,
    ieip_search,
    newton_trajectory,
    pair_methods,
    run_pair_method,
    spring_pair_search,
    two_point_shs,
)
from mopt_rs.variables import NativeObjective

MB_MINIMA = [(-0.558224, 1.441726), (0.623499, 0.028038), (-0.050011, 0.466694)]
MB_SADDLES = [(-0.822002, 0.624313), (0.212487, 0.292988)]
HAS_ASE = importlib.util.find_spec("ase") is not None


@pytest.fixture(autouse=True)
def an_ase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ASE, or its stand-in where it is not available; runs write into an empty directory."""
    if not HAS_ASE:
        monkeypatch.setitem(sys.modules, "ase", fake_ase.module())
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)


def near(x: object, point: tuple[float, float], tolerance: float) -> bool:
    return bool(np.linalg.norm(np.asarray(x, dtype=float) - np.asarray(point)) < tolerance)


def test_the_searches_find_the_saddles_of_muller_brown() -> None:
    # The saddle between the two lower basins.
    # iEIP ends when the images are closer than 0.15: its midpoint is that close.
    r = ieip_search("mb2d", MB_MINIMA[1], MB_MINIMA[2])
    assert r.converged and near(r.transition_state, MB_SADDLES[1], 0.15)
    r = two_point_shs("mb2d", MB_MINIMA[2], MB_MINIMA[1], step=0.01)
    assert r.converged and near(r.transition_state, MB_SADDLES[1], 0.05)
    r = newton_trajectory("mb2d", MB_MINIMA[1], MB_MINIMA[2], step_length=0.02)
    assert r.converged and near(r.transition_state, MB_SADDLES[1], 0.05)
    r = dimer_search("mb2d", MB_MINIMA[1], np.subtract(MB_MINIMA[2], MB_MINIMA[1]), max_step=0.02)
    assert r.converged and near(r.transition_state, MB_SADDLES[1], 1e-3)
    r = addf_search("mb2d", MB_MINIMA[0], initial_length=0.01, step_length=0.02, paths=0)
    found = [r.transition_state, *r.others]
    assert r.converged and any(near(x, MB_SADDLES[0], 0.06) for x in found)


def test_the_spring_pair_climbs_to_a_saddle() -> None:
    # The rates are those of a surface whose curvatures are some hundreds (the paper's
    # example sets its own). The pair reaches the saddle within 0.004; the force
    # thresholds (atomic units) are not met on the scale of this surface.
    r = spring_pair_search(
        "mb2d",
        MB_MINIMA[1],
        np.subtract(MB_MINIMA[2], MB_MINIMA[1]),
        offset=0.02,
        spring_length=0.005,
        drift_rate=5e-4,
        spring_rate=2e-2,
        climb_rate=5e-3,
        drift_tolerance=1.0,
        drift_steps=50,
        max_step=0.01,
        max_cycles=3000,
    )
    assert near(r.transition_state, MB_SADDLES[1], 1e-2)


def test_the_string_methods_and_the_registry() -> None:
    # Settings for the scale of the Muller-Brown surface (curvatures of some hundreds).
    gsm: dict[str, Any] = {
        "max_step": 0.02,
        "first_scale": 0.001,
        "add_tolerance": 1.0,
        "climb_tolerance": 0.5,
        "criteria": "loose",
    }
    r = gsm_search("mb2d", MB_MINIMA[1], MB_MINIMA[2], **gsm)
    assert r.converged and near(r.transition_state, MB_SADDLES[1], 0.02)
    r = gsm_search(
        "mb2d",
        MB_MINIMA[1],
        driving=[f"VAR 1 {MB_MINIMA[2][0]}", f"VAR 2 {MB_MINIMA[2][1]}"],
        drive_step=0.08,
        drop=0.5,
        **gsm,
    )
    assert r.converged and near(r.transition_state, MB_SADDLES[1], 0.02)
    r = fsm_search("mb2d", MB_MINIMA[1], MB_MINIMA[2], first_scale=0.001, max_component=0.03)
    assert r.converged and near(r.transition_state, MB_SADDLES[1], 0.05)
    # The paper variant (the default adds the relaxation of the path, whose spacing is that of
    # structures).
    r = dsafir_search(
        "mb2d", MB_MINIMA[1], MB_MINIMA[2], variant="paper", force=100, step=0.01, separation=0.02
    )
    assert r.converged and near(r.transition_state, MB_SADDLES[1], 0.02)
    names = [m[0] for m in pair_methods()]
    for name in ["ieip", "gsm", "se_gsm", "fsm", "dsafir", "addf", "neb", "dmf", "bitss"]:
        assert name in names
    out = run_pair_method(
        "dimer",
        "mb2d",
        [list(MB_MINIMA[1]), list(MB_MINIMA[2])],
        settings={"step": "0.02"},
        directory="registry",
    )
    assert out["converged"] and near(out["transition_states"][0][0], MB_SADDLES[1], 1e-3)
    assert out["gradients"] > 0
    with pytest.raises(Exception, match="unknown setting"):
        run_pair_method(
            "gsm", "mb2d", [list(MB_MINIMA[1]), list(MB_MINIMA[2])], settings={"nodez": "3"}
        )


def test_ask_and_tell_by_hand() -> None:
    surface = NativeObjective("mb2d")
    pair = ElasticImagePair(list(MB_MINIMA[1]), list(MB_MINIMA[2]))
    images = pair.ask()
    assert images is not None
    told = []
    for x in images:
        value, g, _ = surface.evaluate(np.asarray(x))
        told.append((value, g.tolist(), None, None))
    report = pair.tell((told[0], told[1]))
    assert report["iteration"] == 0 and report["beta"] == pytest.approx(1.0)
    assert report["separation"] < np.linalg.norm(np.subtract(MB_MINIMA[1], MB_MINIMA[2]))
    assert pair.end is None

    d = Dimer([0.6, 0.05], [-1.0, 0.6])
    points = d.ask()
    assert points is not None and len(points) == 2 and all(p["gradient"] for p in points)
    with pytest.raises(Exception, match="values for"):
        d.tell([])
    with pytest.raises(Exception, match="zero"):
        Dimer([0.0, 0.0], [0.0, 0.0])


def write_pair(directory: Path) -> None:
    directory.mkdir()
    for letter, r in (("A", 1.2), ("B", 1.8)):
        (directory / f"h2_{letter}.xyz").write_text(f"2\n\nH 0.0 0.0 0.0\nH {r} 0.0 0.0\n")


def test_the_command_with_an_ase_calculator() -> None:
    write_pair(Path("pair"))
    code = _core.run_cli(
        "ieipmain",
        ["pair", "-os", "ase:fake_ase:DoubleWell", "-ieip_opt", "figures=off"],
    )
    assert code == 0
    runs = [p for p in Path(".").rglob("pair_iEIP_*") if p.is_dir()]
    assert len(runs) == 1
    run = runs[0]
    guess = (run / "h2_A_ts_guess.xyz").read_text().splitlines()
    distance = abs(float(guess[3].split()[1]) - float(guess[2].split()[1]))
    # The barrier lies at 1.5 Angstrom.
    assert abs(distance - 1.5) < 0.1, distance
    for name in ["approx_TS.txt", "h2_A_traj.xyz", "plot_energy_energy.png", "mopt.log"]:
        assert (run / name).is_file(), name
    # No directory per iteration, and the trajectories of the two images are the one file.
    assert not [p for p in run.iterdir() if p.is_dir()]
    assert not list(run.glob("*_image_*_traj.xyz"))
    assert "Evaluations (this input)" in (run / "mopt.log").read_text()


def test_a_job_in_the_manner_of_multioptpy() -> None:
    write_pair(Path("pair"))
    job = IeipJob("pair")
    job.set_option("othersoft", "ase:fake_ase:DoubleWell")
    job.set_options(ieip_options=["figures=off"], NSTEP=200)
    assert job.run("work") == 0
    run = job.get_results()
    assert run is not None and (run / "h2_A_ts_guess.xyz").is_file()
    settings = (run / "settings.json").read_text()
    assert "DoubleWell" in settings


def test_what_the_command_refuses() -> None:
    write_pair(Path("pair"))
    # SQM1 is not part of mopt-rs.
    assert _core.run_cli("ieipmain", ["pair", "-os", "lj", "-sqm1"]) == 2
    # The relaxation of the images needs structures.
    Path("a.txt").write_text("x 0.623499\ny 0.028038\n")
    Path("b.txt").write_text("x -0.050011\ny 0.466694\n")
    assert _core.run_cli("ieipmain", ["a.txt", "b.txt", "-os", "mb2d", "-mi", "3"]) == 2
    # Bias potentials need atoms.
    assert (
        _core.run_cli("ieipmain", ["a.txt", "b.txt", "-os", "mb2d", "-kp", "0.1", "1", "1,2"]) == 2
    )
