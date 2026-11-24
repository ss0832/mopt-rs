"""The direct MaxFlux method from Python: the search, the state machine and its parts.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mopt_rs import DirectMaxFlux, DmfPath, NativeObjective, dmf_search
from mopt_rs.dmf import dmf_profile

MB_MINIMA = [(-0.558224, 1.441726), (0.623499, 0.028038), (-0.050011, 0.466694)]
MB_SADDLES = [(-0.822002, 0.624313), (0.212487, 0.292988)]


def test_the_path_on_the_muller_brown_surface_goes_over_its_saddles() -> None:
    for first, second, saddle in [
        (MB_MINIMA[0], MB_MINIMA[2], MB_SADDLES[0]),
        (MB_MINIMA[2], MB_MINIMA[1], MB_SADDLES[1]),
    ]:
        r = dmf_search("mb2d", [first, second])
        assert r.converged and r.end in ("converged", "settled"), r.end
        assert np.linalg.norm(r.transition_state - np.asarray(saddle)) < 0.05, r.transition_state
        # The estimate is on the path, at the top of its energy profile.
        assert r.path.shape == (41, 2)
        assert np.allclose(r.path[0], first) and np.allclose(r.path[-1], second)
        assert r.profile.shape == (41, 2)
        assert r.value >= r.profile[:, 1].max() - 1e-12
        assert 0.0 < r.share < 1.0
        # Five points between the ends in every iteration, the ends once.
        assert r.gradient_evaluations == 2 + 5 * r.iterations
        assert len(r.reports) == r.iterations
        assert r.length > np.linalg.norm(np.subtract(second, first)) - 1e-9


def test_a_python_function_and_the_state_machine_give_the_same_run() -> None:
    surface = NativeObjective("mb2d")

    def function(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _ = surface.evaluate(x, False)
        return value, gradient

    chain = [MB_MINIMA[0], (-0.7, 1.0), MB_MINIMA[2]]
    seen: list[dict[str, Any]] = []

    def keep(report: dict[str, Any]) -> bool:
        seen.append(report)
        return False

    by_function = dmf_search(function, chain, callback=keep)
    by_name = dmf_search("mb2d", chain)
    assert by_function.iterations == by_name.iterations == len(seen)
    assert np.allclose(by_function.transition_state, by_name.transition_state, atol=1e-12)

    # The same run by hand.
    machine = DirectMaxFlux([list(p) for p in chain])
    assert machine.end is None and not machine.converged and machine.dimension == 2
    first = machine.ask()
    assert first is not None and len(first) == 7
    assert [p["index"] for p in first] == list(range(7))
    asked: list[dict[str, Any]] | None = first
    while asked is not None:
        values = []
        for point in asked:
            value, gradient = function(np.asarray(point["x"]))
            values.append((value, gradient.tolist()))
        report = machine.tell(values)
        assert report is not None
        asked = machine.ask()
        if asked is not None:
            # After the first iteration the ends are known and only the inner points come.
            assert [p["index"] for p in asked] == list(range(1, 6))
    assert machine.end == by_name.end and machine.converged
    assert machine.iterations == by_name.iterations
    estimate = machine.transition_state_estimate
    assert estimate is not None and np.allclose(estimate, by_name.transition_state, atol=1e-12)
    assert len(machine.control_points) == 7
    assert machine.result is not None and machine.report is not None


def test_the_path_its_length_and_the_objective() -> None:
    line = DmfPath([[0.0, 0.0], [3.0, 4.0]])
    assert line.dimension == 2 and len(line.control) == 7
    assert abs(line.length - 5.0) < 1e-12
    assert abs(line.parameter(0.3) - 0.3) < 1e-12 and abs(line.unevenness - 1.0) < 1e-9
    assert np.allclose(line.points([0.0, 0.5, 1.0]), [[0.0, 0.0], [1.5, 2.0], [3.0, 4.0]])
    assert np.allclose(line.velocity(0.2), [3.0, 4.0])

    # The gradient of the objective with respect to the inner control points against
    # differences, with the energies and gradients of a quadratic surface at the points.
    control = np.array(
        [[0.0, 0.0], [0.3, 0.4], [0.8, 0.9], [1.2, 1.1], [1.7, 0.8], [2.2, 0.4], [2.5, 0.0]]
    )
    shares = np.linspace(0.0, 1.0, 6)
    beta = 3.0

    def objective(c: np.ndarray) -> tuple[float, list[list[float]]]:
        path = DmfPath.from_control(c.tolist())
        points = np.asarray(path.points(shares.tolist()))
        energies = 0.5 * (points[:, 0] - 1.0) ** 2 - 0.4 * points[:, 1] ** 2
        gradients = np.column_stack([points[:, 0] - 1.0, -0.8 * points[:, 1]])
        found = path.action(shares.tolist(), energies.tolist(), gradients.tolist(), beta)
        assert abs(sum(found["weights"]) - 1.0) < 1e-12
        assert abs(found["length"] - path.length) < 1e-12
        return float(found["value"]), found["gradient"]

    _, analytic = objective(control)
    assert len(analytic) == 5
    for b in range(5):
        for k in range(2):
            up, down = control.copy(), control.copy()
            up[b + 1, k] += 1e-6
            down[b + 1, k] -= 1e-6
            numeric = (objective(up)[0] - objective(down)[0]) / 2e-6
            # The analytic gradient is the one at fixed energies plus the slopes along the
            # path: the points move with the control points and see other energies.
            assert abs(numeric - analytic[b][k]) < 1e-5, (b, k, numeric, analytic[b][k])


def test_the_energy_profile_through_the_points() -> None:
    shares = [0.0, 0.25, 0.5, 0.75, 1.0]
    energies = [4.0 * s * (1.0 - s) for s in shares]
    slopes = [4.0 - 8.0 * s for s in shares]
    found = dmf_profile(shares, energies, slopes, samples=11)
    assert abs(found["top_share"] - 0.5) < 1e-12 and abs(found["top_energy"] - 1.0) < 1e-12
    # A parabola is a cubic: the pieces are exact.
    for share, energy in found["profile"]:
        assert abs(energy - 4.0 * share * (1.0 - share)) < 1e-12
    # A top between the points is found inside its piece.
    shifted = dmf_profile([0.0, 0.5, 1.0], [0.0, 0.9, 0.0], [3.0, 0.4, -3.0])
    assert 0.5 < shifted["top_share"] < 1.0 and shifted["top_energy"] > 0.9
