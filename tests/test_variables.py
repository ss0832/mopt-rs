"""Optimization of plain variables from Python: built-in and Python objectives.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import mopt_rs
from mopt_rs import NativeObjective, StepMethod, VectorOptimizer, _core, optimize_variables
from mopt_rs.variables import finite_difference_hessian

MB_MINIMA = [(-0.558224, 1.441726), (0.623499, 0.028038), (-0.050011, 0.466694)]
MB_SADDLES = [(-0.822002, 0.624313), (0.212487, 0.292988)]


def rosenbrock(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Five-variable Rosenbrock function (minimum 0 at (1, ..., 1)) scaled by 1e-2."""
    a, b = x[:-1], x[1:]
    value = np.sum(100.0 * (b - a**2) ** 2 + (1.0 - a) ** 2)
    g = np.zeros_like(x)
    g[:-1] += -400.0 * a * (b - a**2) - 2.0 * (1.0 - a)
    g[1:] += 200.0 * (b - a**2)
    return 1e-2 * float(value), 1e-2 * g


def test_muller_brown_minima_and_saddles() -> None:
    for start, target in zip([(-0.4, 1.3), (0.5, 0.1), (0.0, 0.5)], MB_MINIMA, strict=True):
        for method in ["rsirfo_bfgs", "FIRE", "lbfgs"]:
            r = optimize_variables("mb2d", start, methods=method, max_iterations=300)
            assert r.converged, method
            assert np.linalg.norm(r.x - target) < 2e-2, (method, r.x)
            assert len(r.values) == r.iterations == len(r.reports)
    for start, target in zip([(-0.8, 0.6), (0.2, 0.3)], MB_SADDLES, strict=True):
        r = optimize_variables(
            "mb2d", start, methods="rsirfo_bofill", saddle_order=1, exact_hessian_interval=1
        )
        assert r.converged
        assert np.linalg.norm(r.x - target) < 2e-2, r.x
        # One negative curvature at the saddle point.
        _, _, h = NativeObjective("mb2d").evaluate(r.x, need_hessian=True)
        assert h is not None
        assert np.sum(np.linalg.eigvalsh(h) < 0.0) == 1
    # CCQN along a fixed direction, from the side of minimum C toward minimum B.
    c, b, ts = np.array(MB_MINIMA[2]), np.array(MB_MINIMA[1]), np.array(MB_SADDLES[1])
    r = optimize_variables(
        "mb2d",
        ts + 0.3 * (c - ts),
        methods="ccqn",
        saddle_order=1,
        exact_hessian_interval=5,
        ccqn_direction=b - c,
        ccqn_cos_phi=0.9,
    )
    assert r.converged
    assert np.linalg.norm(r.x - ts) < 2e-2


def test_python_objective_with_five_variables() -> None:
    calls = 0

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls
        calls += 1
        return rosenbrock(x)

    r = optimize_variables(fun, np.full(5, 0.5), methods="rsirfo_bfgs", max_iterations=2000)
    assert r.converged
    assert np.max(np.abs(r.x - 1.0)) < 5e-2, r.x
    assert calls == r.iterations
    assert r.value == pytest.approx(rosenbrock(r.x)[0])
    assert np.allclose(r.gradient, rosenbrock(r.x)[1])
    # A callback stops the run.
    seen: list[int] = []

    def stop(report: dict[str, Any]) -> bool:
        seen.append(report["iteration"])
        return len(seen) == 3

    r = optimize_variables(fun, np.full(5, 0.5), callback=stop)
    assert seen == [0, 1, 2]
    assert r.termination is None and not r.converged


def test_exact_hessians_from_the_objective_a_function_or_differences() -> None:
    a = np.diag([-1.0, 2.0, 3.0]) + 0.05
    center = np.array([0.1, 0.2, 0.3])

    def plain(x: np.ndarray) -> tuple[float, np.ndarray]:
        d = x - center
        return 0.5 * float(d @ a @ d), a @ d

    def with_hessian(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        return (*plain(x), a)

    assert np.allclose(finite_difference_hessian(lambda x: plain(x)[1], np.zeros(3)), a)
    asked = 0

    def hessian(x: np.ndarray) -> np.ndarray:
        nonlocal asked
        asked += 1
        return a

    for kwargs in [{}, {"hessian": hessian}]:
        for fun in [plain, with_hessian]:
            r = optimize_variables(
                fun,
                np.zeros(3),
                methods="rsirfo_bofill",
                saddle_order=1,
                exact_hessian_interval=1,
                **kwargs,  # type: ignore[arg-type]
            )
            assert r.converged
            assert np.allclose(r.x, center, atol=1e-4), r.x
    assert asked > 0


def test_expressions_and_the_ask_tell_optimizer() -> None:
    r = optimize_variables("expr:(q1 - 1)^2 + 2*(q2 + 0.5)^2 + (q3 - q4)^2 + q4^2", np.zeros(4))
    assert r.converged
    assert np.allclose(r.x, [1.0, -0.5, 0.0, 0.0], atol=1e-3)
    # By hand: seven variables (not a multiple of three).
    target = np.arange(7.0)
    opt = VectorOptimizer([0.0] * 7, methods=["lbfgs"])
    assert opt.dimension == 7
    while (request := opt.ask()) is not None:
        x = np.asarray(request[0])
        assert request[1] is False
        report = opt.tell(0.5 * float(np.sum((x - target) ** 2)), (x - target).tolist())
        assert set(report) >= {"iteration", "x", "value", "max_gradient", "converged"}
    assert opt.termination == "converged"
    assert np.allclose(opt.x, target, atol=1e-3)


def test_errors_are_reported() -> None:
    assert {"mb2d", "expr"} <= set(mopt_rs.objective_names())
    with pytest.raises(ValueError, match="unknown objective"):
        optimize_variables("no_such_objective", [0.0, 0.0])
    with pytest.raises(ValueError, match="energy_scale"):
        optimize_variables("mb2d", [0.0, 0.0], options={"scale": 2})
    with pytest.raises(ValueError):
        optimize_variables("mb2d", [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="gradient has 1"):
        optimize_variables(lambda x: (0.0, [0.0]), [0.0, 0.0])
    with pytest.raises(ValueError, match="must return"):
        optimize_variables(lambda x: (0.0,), [0.0, 0.0])
    with pytest.raises(ValueError, match="needs atoms"):
        optimize_variables("expr:dist(1,2)", [0.0, 0.0])
    scaled = optimize_variables("mb2d", [-0.4, 1.3], options={"energy_scale": 2e-3})
    plain = optimize_variables("mb2d", [-0.4, 1.3])
    assert scaled.value == pytest.approx(2.0 * plain.value, rel=1e-3)


def test_ported_optimizers_by_their_multioptpy_names() -> None:
    minimizers = [
        "cg",
        "adam",
        "fire2",
        "tr_lbfgs",
        "rsprfo_bfgs",
        "lbfgs_gdiis",
        "rsirfo_bfgs_linesearch",
        "gpmin",
    ]
    for method in minimizers:
        r = optimize_variables("mb2d", (-0.4, 1.3), methods=method, max_iterations=300)
        assert r.converged, method
        assert np.linalg.norm(r.x - MB_MINIMA[0]) < 2e-2, (method, r.x)
    for method in ["rsprfo_bofill", "mf_rsirfo_bofill", "rsirfo_bofill_trim"]:
        r = optimize_variables(
            "mb2d", (-0.8, 0.6), methods=method, saddle_order=1, exact_hessian_interval=1
        )
        assert r.converged, method
        assert np.linalg.norm(r.x - MB_SADDLES[0]) < 2e-2, (method, r.x)


def test_one_method_used_on_its_own() -> None:
    method = StepMethod("lbfgs_gdiis", 2, max_step=0.5)
    assert method.name == "lbfgs"
    assert method.enhancers == ["gdiis"]
    curvature = np.array([2.0, 8.0])
    x = np.array([1.0, -2.0])
    for _ in range(60):
        g = curvature * x
        x = x + np.array(method.propose(x.tolist(), float(0.5 * g @ x), g.tolist()))
    assert np.abs(x).max() < 1e-6
    assert method.iterations == 60
    assert len(method.hessian) == 4
    method.reset()
    assert method.iterations == 0
    with pytest.raises(ValueError):
        method.propose([0.0], 0.0, [0.0])
    with pytest.raises(ValueError):
        StepMethod("lbfgs", 2, max_step=-1.0)


def test_command_line_on_a_variables_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "start.txt").write_text("x -0.4\ny 1.3\n")
    assert _core.run_cli("optmain", ["start.txt", "-os", "mb2d", "-opt", "rsirfo_bfgs"]) == 0
    out = list(tmp_path.glob("*/start_OPT_ASE_*/start_optimized.txt"))
    assert len(out) == 1
    values = [float(line.split()[1]) for line in out[0].read_text().splitlines()[1:]]
    assert np.linalg.norm(np.array(values) - MB_MINIMA[0]) < 2e-2
    assert (out[0].parent / "start_traj.csv").exists()


def test_mode_kill_leaves_a_saddle_point_of_too_high_an_order() -> None:
    objective = "expr:-0.5*q1^2 - 0.3*q2^2 + 0.25*q2^4 + q3^2"
    kept = optimize_variables(
        objective,
        [0.3, 0.0, 0.4],
        methods="rsirfo_bofill",
        saddle_order=1,
        exact_hessian_interval=1000,
    )
    assert kept.converged and abs(kept.x[1]) < 1e-8
    left = optimize_variables(
        objective,
        [0.3, 0.0, 0.4],
        methods="rsirfo_bofill",
        saddle_order=1,
        exact_hessian_interval=1000,
        mode_kill=True,
    )
    assert left.converged
    assert abs(abs(left.x[1]) - np.sqrt(0.6)) < 1e-3

    # A function of one's own without a Hessian: central differences answer the check.
    def well(x: np.ndarray) -> tuple[float, np.ndarray]:
        value = (x[0] ** 2 - 1.0) ** 2 + x[1] ** 2
        return value, np.array([4.0 * x[0] * (x[0] ** 2 - 1.0), 2.0 * x[1]])

    assert abs(optimize_variables(well, [0.0, 0.0]).x[0]) < 1e-12
    assert abs(abs(optimize_variables(well, [0.0, 0.0], mode_kill=True).x[0]) - 1.0) < 1e-3
