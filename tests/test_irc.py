"""Intrinsic reaction coordinate from Python.

MultiOptPy: `irc.py:IRC` and `IRC/`.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import mopt_rs
from mopt_rs import Irc, NativeObjective, irc_path
from mopt_rs.irc import irc_first_step_test, irc_lqa_step, irc_methods

MB_MINIMA = [(-0.558224, 1.441726), (0.623499, 0.028038), (-0.050011, 0.466694)]
MB_SADDLES = [(-0.822002, 0.624313), (0.212487, 0.292988)]
CONNECTED = [{0, 2}, {1, 2}]


def nearest_minimum(x: Any) -> tuple[int, float]:
    distances = [float(np.linalg.norm(np.asarray(x) - np.asarray(m))) for m in MB_MINIMA]
    k = int(np.argmin(distances))
    return k, distances[k]


def test_methods_are_those_of_the_rust_registry() -> None:
    assert irc_methods() == ["dvv", "euler", "gs2", "hpc", "lqa", "modekill", "rk4"]


@pytest.mark.parametrize(
    ("method", "step", "tolerance"),
    [
        ("lqa", 0.05, 2e-3),
        ("hpc", 0.05, 5e-3),
        ("gs2", 0.05, 0.08),
        ("euler", 0.01, 0.03),
        ("rk4", 0.03, 0.06),
        ("dvv", 0.05, 0.15),
    ],
)
def test_built_in_surface_both_saddles(method: str, step: float, tolerance: float) -> None:
    for saddle, connected in zip(MB_SADDLES, CONNECTED, strict=True):
        r = irc_path("mb2d", saddle, step_size=step, max_steps=600, method=method)
        assert r.kind == "transition_state" and not r.interrupted
        assert [s.branch for s in r.sides] == ["forward", "backward"]
        reached = set()
        for side in r.sides:
            minimum, distance = nearest_minimum(side.points[-1])
            assert distance < tolerance, (method, side.branch, distance, side.end)
            reached.add(minimum)
        assert reached == connected
        # The combined path: forward end, start, backward end; lengths fall along it.
        n = sum(len(s.steps) for s in r.sides) + 1
        assert r.path.shape == (n, 2)
        k = len(r.sides[0].steps)
        assert np.allclose(r.path[k], saddle)
        assert r.path_arc_lengths[k] == 0.0
        assert np.all(np.diff(r.path_arc_lengths) < 0.0)
        assert int(np.argmax(r.path_energies)) == k


def test_python_function_with_masses_and_finite_difference_hessian() -> None:
    # A double well along x with a harmonic valley across; masses stretch the coordinates.
    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        value = 0.05 * (x[0] ** 2 - 1.0) ** 2 + 0.3 * (x[1] - 0.2 * x[0] ** 2) ** 2
        dy = x[1] - 0.2 * x[0] ** 2
        gradient = np.array(
            [0.2 * x[0] * (x[0] ** 2 - 1.0) - 0.24 * x[0] * dy, 0.6 * dy],
        )
        return float(value), gradient

    calls = 0

    def counted(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls
        calls += 1
        return surface(x)

    r = irc_path(counted, [0.0, 0.0], masses=[4.0, 1.0], step_size=0.1, max_steps=300)
    assert r.kind == "transition_state"
    assert r.start is not None and r.start["imaginary"] == 1
    # The Hessian of the start came from central differences of the gradient.
    assert calls == r.evaluations + 4
    ends = sorted(float(s.points[-1][0]) for s in r.sides)
    assert ends == pytest.approx([-1.0, 1.0], abs=2e-3)
    for side in r.sides:
        assert side.end == "converged"
        assert side.points[-1][1] == pytest.approx(0.2, abs=2e-3)
        # Lengths are mass-weighted: the first displacement is 0.01 along x, 0.005 in x.
        assert abs(side.arc_lengths[0]) == pytest.approx(0.01, abs=1e-9)
        assert abs(side.points[0][0]) == pytest.approx(0.005, abs=1e-9)


def test_flat_bent_valley_never_rises_and_ends_in_its_minima() -> None:
    # A valley that is flat along the path, stiff across it, and bent: steps of the full
    # length leave its floor, and the energy would rise.
    a, k, c = 1e-4, 1.0, 0.5

    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        off = x[1] - c * x[0] ** 2
        value = a * (x[0] ** 2 - 1.0) ** 2 + 0.5 * k * off * off
        gradient = np.array(
            [4.0 * a * x[0] * (x[0] ** 2 - 1.0) - 2.0 * c * k * x[0] * off, k * off],
        )
        return float(value), gradient

    r = irc_path(surface, [0.0, 0.0], step_size=0.1, max_steps=600)
    assert sorted(float(np.sign(s.points[-1][0])) for s in r.sides) == [-1.0, 1.0]
    for side in r.sides:
        assert side.end == "converged"
        assert np.all(np.diff(side.energies) <= 1e-10)
        assert abs(side.energies[-1]) < 2e-7
        _, gradient = surface(side.points[-1])
        assert np.max(np.abs(gradient)) < 1.5e-5
        # What the step control did is reported.
        assert side.dropped_points > 0 and side.approach_steps > 0
    # MultiOptPy's way keeps the points and ends before the minima.
    old = irc_path(
        surface,
        [0.0, 0.0],
        step_size=0.1,
        max_steps=600,
        options={"first_step": "off", "energy_rise": "off", "approach": "off"},
    )
    assert all(
        s.dropped_points == 0 and s.overshot_points == 0 and s.approach_steps == 0
        for s in old.sides
    )
    assert max(float(s.energies[-1]) for s in old.sides) > 1e-5


def test_callback_sees_every_step_and_can_stop() -> None:
    seen: list[dict[str, Any]] = []

    def watch(report: dict[str, Any]) -> bool | None:
        seen.append(report)
        return None

    r = irc_path("mb2d", MB_SADDLES[0], step_size=0.05, callback=watch)
    assert len(seen) == sum(len(s.steps) for s in r.sides)
    first = seen[0]
    assert first["branch"] == "forward" and first["step"] == 1
    assert isinstance(first["x"], np.ndarray) and first["x"].shape == (2,)
    assert {"energy", "biased_energy", "arc_length", "s"} <= set(first)
    assert all(a["s"] > 0.0 for a in seen if a["branch"] == "forward")
    assert all(a["s"] < 0.0 for a in seen if a["branch"] == "backward")

    def stop_at_third(report: dict[str, Any]) -> bool:
        return bool(report["step"] < 3)

    r = irc_path("mb2d", MB_SADDLES[0], step_size=0.05, callback=stop_at_third)
    assert r.interrupted
    assert [len(s.steps) for s in r.sides] == [3]
    assert r.sides[0].end is None
    assert r.path.shape == (4, 2)


def test_first_step_check_and_meta_irc() -> None:
    options = {"first_step": "check", "initial": 0.3}
    r = irc_path("mb2d", MB_SADDLES[1], step_size=0.05, options=options)
    assert len(r.first_steps) == 2
    assert any(t["corrected"] for t in r.first_steps)
    for test in r.first_steps:
        assert test["length"] == pytest.approx(0.3)
        if test["corrected"]:
            assert test["converged"] and test["iterations"] >= 2 and test["energy_gain"] < 0.0
    assert {nearest_minimum(s.points[-1])[0] for s in r.sides} == {1, 2}
    # Off the saddle: one path down the gradient.
    r = irc_path("mb2d", [-0.7, 0.9], step_size=0.05)
    assert r.kind == "meta_irc" and [s.branch for s in r.sides] == ["single"]
    minimum, distance = nearest_minimum(r.sides[0].points[-1])
    assert minimum == 0 and distance < 2e-3
    with pytest.raises(ValueError):
        irc_path("mb2d", MB_SADDLES[0], method="lqaa")
    with pytest.raises(ValueError):
        irc_path("mb2d", MB_SADDLES[0], options=["nothing=1"])


def test_ask_tell_by_hand_and_the_parts() -> None:
    mb = NativeObjective("mb2d")
    irc = Irc(list(MB_SADDLES[0]), step_size=0.05, options=["sides=forward"])
    assert irc.method == "lqa" and irc.kind is None
    request = irc.ask()
    assert request is not None
    assert request["role"] == "start" and request["hessian"] and request["branch"] is None
    value, gradient, hessian = mb.evaluate(np.asarray(request["x"]), True)
    assert hessian is not None
    irc.tell(value, gradient.tolist(), hessian.ravel().tolist())
    events = irc.events()
    assert [e["event"] for e in events] == ["start"]
    assert events[0]["kind"] == "transition_state" and irc.kind == "transition_state"
    vector = np.asarray(events[0]["transition_vector"])
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    steps = 0
    taken_again = 0
    while (request := irc.ask()) is not None:
        assert request["branch"] == "forward" and not request["hessian"]
        value, gradient, _ = mb.evaluate(np.asarray(request["x"]), False)
        irc.tell(value, gradient.tolist())
        for event in irc.events():
            steps += event["event"] == "step"
            if event["event"] == "side_end":
                taken_again = event["dropped_points"] + event["overshot_points"]
    # One evaluation per point of the path, the start, and the steps that were taken again
    # (the test of the first point uses the evaluation of that point).
    assert steps > 10 and irc.evaluations == steps + 1 + taken_again
    with pytest.raises(ValueError):
        irc.tell(0.0, [0.0, 0.0])

    # The LQA step of a quadratic valley, and the test of a first step.
    step = irc_lqa_step([0.2, 0.0, 0.0, 3.0], [0.3, 0.4], 0.1)
    assert 0.08 < float(np.linalg.norm(step)) < 0.1 and step[0] < 0.0 and step[1] < 0.0
    angle, rho, predicted, actual = irc_first_step_test(
        [0.2, 0.0], -0.3, 1.0, [0.0, 0.0], 1.0 - 0.006, [-0.06, 0.0]
    )
    assert angle == pytest.approx(0.0, abs=1e-6) and rho == pytest.approx(1.0)
    assert predicted == pytest.approx(-0.006) and actual == pytest.approx(-0.006)


def test_structure_with_symbols_and_the_command(tmp_path: Path) -> None:
    # One atom on the Muller-Brown surface through the command, as the pip commands run it.
    ang = 0.529177210903
    xyz = tmp_path / "h.xyz"
    xyz.write_text(f"1\n\nH {MB_SADDLES[0][0] * ang} {MB_SADDLES[0][1] * ang} 0.0\n")
    code = mopt_rs._core.run_cli(
        "ircmain", [str(xyz), "-os", "muller_brown", "-irc_step", "0.05", "-irc_method", "gs2"]
    )
    assert code == 0
    runs = [p for p in tmp_path.iterdir() if p.is_dir() and "_IRC_" in p.name]
    assert len(runs) == 1
    for name in [
        "irc_forward/irc_structures.xyz",
        "irc_backward/irc_energies_gradients.csv",
        "irc_combined_path.xyz",
        "irc_combined_data.csv",
        "irc_profile.csv",
        "h_irc_endpoint_1.xyz",
        "h_irc_endpoint_2.xyz",
    ]:
        assert (runs[0] / name).exists(), name

    # The same surface as a structure in the state machine: symbols give the masses; the
    # surface depends on the position, so nothing is projected.
    x0 = [MB_SADDLES[0][0], MB_SADDLES[0][1], 0.0]
    mb = NativeObjective("mb2d")

    def atom(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        value, gradient, hessian = mb.evaluate(x[:2], True)
        assert hessian is not None
        g = np.array([gradient[0], gradient[1], x[2]])
        h = np.zeros((3, 3))
        h[:2, :2] = hessian
        h[2, 2] = 1.0
        return value + 0.5 * x[2] ** 2, g, h

    r = irc_path(atom, x0, symbols=["H"], project_rigid=False, step_size=0.05)
    assert r.kind == "transition_state"
    assert {nearest_minimum(s.points[-1][:2])[0] for s in r.sides} == {0, 2}
