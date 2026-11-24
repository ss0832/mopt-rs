"""Python ask/tell API: native and pure-Python calculators.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import mopt_rs
from mopt_rs import _core
from mopt_rs.optimize import Evaluation, NativeCalculator, Optimizer, Task, run

if TYPE_CHECKING:
    from mopt_rs._core import TermCallback

ANG = _core.ANGSTROM_TO_BOHR
AR7 = np.array(
    [
        [1.9500, 0.0000, 0.0000],
        [0.6026, 1.8546, 0.0300],
        [-1.5776, 1.1462, 0.0700],
        [-1.5776, -1.1462, 0.1100],
        [0.6026, -1.8546, 0.1500],
        [0.0000, 0.0000, 2.1000],
        [0.2000, 0.0000, -2.0000],
    ]
)


class Harmonic:
    """E = k/2 |x - x0|^2 per atom, a pure-Python calculator with a known minimum."""

    def __init__(self, x0: np.ndarray, k: float = 0.5) -> None:
        self.x0 = x0
        self.k = k
        self.calls = 0

    def evaluate(self, task: Task) -> Evaluation:
        self.calls += 1
        d = task.positions_bohr - self.x0
        n = d.size
        hess = self.k * np.eye(n) if task.need_hessian else None
        return Evaluation(0.5 * self.k * float(np.sum(d * d)), self.k * d, hess)


def test_python_calculator_reaches_the_known_minimum() -> None:
    x0 = np.array([[0.1, -0.2, 0.3], [1.5, 0.2, -0.1]])
    start = x0 + np.array([[0.3, 0.1, -0.2], [-0.2, 0.25, 0.1]])
    calc = Harmonic(x0)
    opt = Optimizer(
        ["H", "H"],
        start.ravel().tolist(),
        methods=["rsirfo_bfgs"],
        criteria="tight",
        rigid_body_invariant=False,
    )
    result = run(opt, calc)
    assert result.converged
    assert np.allclose(result.positions_bohr, x0, atol=1e-4)
    assert calc.calls == result.iterations


def test_python_loop_matches_the_native_calculator_bitwise() -> None:
    def optimize() -> list[float]:
        opt = Optimizer(["Ar"] * 7, (AR7 * ANG).ravel().tolist(), methods=["rsirfo_bfgs"])
        return run(opt, NativeCalculator("lj")).energies

    a, b = optimize(), optimize()
    assert a == b
    eps = 0.237 / _core.HARTREE_TO_KCAL_MOL
    assert abs(a[-1] / eps + 16.505) < 5e-2


def test_evaluate_native_returns_consistent_derivatives() -> None:
    x = (AR7 * ANG).ravel().tolist()
    e, g, h = _core.evaluate_native("lj", ["Ar"] * 7, x, need_hessian=True)
    assert h is not None and len(h) == 21 * 21
    step = 1e-5
    xp, xm = list(x), list(x)
    xp[4] += step
    xm[4] -= step
    ep, _, _ = _core.evaluate_native("lj", ["Ar"] * 7, xp)
    em, _, _ = _core.evaluate_native("lj", ["Ar"] * 7, xm)
    assert abs((ep - em) / (2 * step) - g[4]) < 1e-9


def test_afir_and_ccqn_through_the_bindings() -> None:
    opt = Optimizer(["Ar"] * 7, (AR7 * ANG).ravel().tolist(), methods=["rsirfo_bfgs"])
    opt.add_afir(50.0, [1], [2])
    result = run(opt, NativeCalculator("lj"))
    assert result.converged
    comps = dict(result.reports[-1]["components"])
    assert "AFIR_pot_0" in comps
    ts = Optimizer(
        ["Ar"] * 7,
        result.positions_bohr.ravel().tolist(),
        methods=["ccqn"],
        saddle_order=1,
        exact_hessian_interval=5,
        ccqn_bonds=[(1, 2)],
        ccqn_cos_phi=0.5,
        coordinates="dic",
        max_iterations=300,
        dissociation_distance_angstrom=None,
    )
    assert run(ts, NativeCalculator("lj")).iterations > 0


def test_bias_options_from_python() -> None:
    pos = [0.0, 0.0, 0.0, 2.6, 0.0, 0.0]
    # -kp 0.5 1.0 1,2: E = k/2 (r - r0)^2 with r0 = 1 Angstrom.
    e, g, h = mopt_rs.evaluate_bias(
        "-kp", ["0.5", "1.0", "1,2"], ["Ar", "Ar"], pos, need_hessian=True
    )
    r0 = _core.ANGSTROM_TO_BOHR
    assert abs(e - 0.25 * (2.6 - r0) ** 2) < 1e-12
    assert abs(g[3] - 0.5 * (2.6 - r0)) < 1e-12 and h is not None
    opt = Optimizer(["Ar", "Ar"], pos, methods=["fire"], max_iterations=500)
    assert opt.add_bias("-kp", ["0.5", "1.0", "1,2", "0.5", "1.0", "1,2"]) == 2
    opt.add_afir(50.0, [1], [2])
    opt.add_afir(60.0, [1], [2])
    result = run(opt, NativeCalculator("lj"))
    labels = {name for name, _ in result.reports[-1]["components"]}
    assert {"keep_pot_0", "keep_pot_1", "AFIR_pot_0", "AFIR_pot_1"} <= labels


def _restraint(k: float, r0: float, with_hessian: bool) -> TermCallback:
    """Python term k/2 (|r1 - r2| - r0)^2 on atoms 1 and 2."""

    def f(
        positions: list[float], cell: list[float] | None, iteration: int, need_hessian: bool
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        x = np.asarray(positions).reshape(-1, 3)
        d = x[0] - x[1]
        r = float(np.linalg.norm(d))
        g = np.zeros_like(x)
        g[0] = k * (r - r0) * d / r
        g[1] = -g[0]
        h = None
        if need_hessian and with_hessian:
            u = d / r
            block = k * (np.outer(u, u) + (r - r0) / r * (np.eye(3) - np.outer(u, u)))
            h = np.zeros((x.size, x.size))
            h[0:3, 0:3] = block
            h[3:6, 3:6] = block
            h[0:3, 3:6] = -block
            h[3:6, 0:3] = -block
        return 0.5 * k * (r - r0) ** 2, g.ravel(), h

    # A function of a distance: the optimizer may remove rigid-body motions with it.
    f.frame = "invariant"  # type: ignore[attr-defined]
    f.history = "static"  # type: ignore[attr-defined]
    return f


def _declared(frame: str | None, history: str | None) -> TermCallback:
    """The restraint of :func:`_restraint` with ``frame`` and ``history`` set (``None``: the
    attribute is left out, as a plain function has it)."""
    f = _restraint(0.5, 9.0, True)
    for name, value in (("frame", frame), ("history", history)):
        if value is None:
            delattr(f, name)
        else:
            setattr(f, name, value)
    return f


@pytest.mark.parametrize("history", [None, "static", "iteration_ramp", "time_dependent", "stateful"])
@pytest.mark.parametrize("frame", [None, "invariant", "translation_only", "absolute"])
def test_the_frame_and_history_of_python_terms(frame: str | None, history: str | None) -> None:
    pos = [0.0, 0.0, 0.0, 7.0, 0.2, 0.0, 3.5, 6.0, 0.1]

    def optimize(rigid: bool) -> mopt_rs.Result:
        opt = Optimizer(
            ["Ar"] * 3,
            pos,
            methods=["rsirfo_bfgs"],
            max_iterations=500,
            rigid_body_invariant=rigid,
        )
        opt.add_term("restraint", _declared(frame, history))
        return run(opt, NativeCalculator("lj"))

    # A term that may depend on where the structure is (the default, "absolute", and
    # "translation_only") cannot go with an optimizer that removes rigid-body motions.
    if frame == "invariant":
        assert optimize(True).converged
    else:
        kind = "TranslationOnly" if frame == "translation_only" else "Absolute"
        with pytest.raises(ValueError, match=rf"'restraint' \({kind}\) depend on the position"):
            optimize(True)
    # With the rigid-body motions kept, every declaration works; "history" (default
    # "stateful") changes nothing within one optimization.
    result = optimize(False)
    assert result.converged
    x = result.positions_bohr
    assert abs(float(np.linalg.norm(x[0] - x[1])) - 9.0) < 0.3


@pytest.mark.parametrize(
    ("name", "value", "allowed"),
    [
        ("frame", "sideways", "'invariant', 'translation_only', 'absolute'"),
        ("history", "never", "'static', 'iteration_ramp', 'time_dependent', 'stateful'"),
        ("frame", "Invariant ", "'invariant', 'translation_only', 'absolute'"),
    ],
)
def test_a_wrong_frame_or_history_is_refused(name: str, value: str, allowed: str) -> None:
    f = _declared(None, None)
    setattr(f, name, value)
    opt = Optimizer(["Ar"] * 3, [0.0, 0.0, 0.0, 7.0, 0.2, 0.0, 3.5, 6.0, 0.1])
    with pytest.raises(ValueError, match=rf"Python term 'restraint': {name} = '.*' is not one of"):
        opt.add_term("restraint", f)
    with pytest.raises(ValueError, match=allowed):
        mopt_rs.check_term(f, ["Ar"] * 3, [0.0, 0.0, 0.0, 7.0, 0.2, 0.0, 3.5, 6.0, 0.1])
    # Upper case is read as lower case.
    f = _declared("INVARIANT", "Static")
    assert mopt_rs.check_term(f, ["Ar"] * 3, [0.0, 0.0, 0.0, 7.0, 0.2, 0.0, 3.5, 6.0, 0.1])


def test_python_terms_are_checked_and_restrain_an_optimization() -> None:
    pos = [0.0, 0.0, 0.0, 7.0, 0.2, 0.0, 3.5, 6.0, 0.1]
    for with_hessian in [True, False]:
        c = mopt_rs.check_term(_restraint(0.5, 9.0, with_hessian), ["Ar"] * 3, pos)
        assert c["gradient_error"] < 1e-8 and c["hessian_error"] < 1e-5, c
        opt = Optimizer(["Ar"] * 3, pos, methods=["rsirfo_bfgs"], max_iterations=500)
        opt.add_term("restraint", _restraint(0.5, 9.0, with_hessian))
        result = run(opt, NativeCalculator("lj"))
        assert result.converged
        x = result.positions_bohr
        assert abs(float(np.linalg.norm(x[0] - x[1])) - 9.0) < 0.3
    c = mopt_rs.check_bias("-ubias", ["0.1*(dist(1,2)-4*ang)^2 + 0.01*x(3)^2"], ["Ar"] * 3, pos)
    assert max(c.values()) < 1e-6, c


def test_constraints_from_python() -> None:
    pos = [0.0, 0.0, 0.0, 7.0, 0.2, 0.0, 3.5, 6.0, 0.1, 3.4, 2.0, 5.5]
    opt = Optimizer(
        ["Ar"] * 4,
        pos,
        methods=["rsirfo_bfgs"],
        fix_atoms=[4],
        constraints=["manual", "bond", "1,2", "4.0"],
    )
    result = run(opt, NativeCalculator("lj"))
    assert result.converged
    x = result.positions_bohr
    assert abs(float(np.linalg.norm(x[0] - x[1])) - 4.0 * ANG) < 1e-6
    assert np.allclose(x[3], np.asarray(pos[9:12]), atol=1e-9)


def _distance_difference(
    positions: list[float], cell: list[float] | None, iteration: int, need_hessian: bool
) -> tuple[float, list[float], None]:
    """c(x) = |r1 - r2| - |r1 - r3| with its gradient (the Hessian comes from differences)."""
    x = np.asarray(positions).reshape(-1, 3)
    g = np.zeros_like(x)
    d12, d13 = x[0] - x[1], x[0] - x[2]
    n12, n13 = float(np.linalg.norm(d12)), float(np.linalg.norm(d13))
    g[0] = d12 / n12 - d13 / n13
    g[1] = -d12 / n12
    g[2] = d13 / n13
    return n12 - n13, g.ravel().tolist(), None


def test_user_constraints_from_python() -> None:
    pos = [0.0, 0.0, 0.0, 7.0, 0.2, 0.0, 3.5, 6.0, 0.1, 3.4, 2.0, 5.5]
    for kwargs in [
        {"user_constraints": ["dist(1,2) = dist(1,3)"]},
        {"constraint_functions": [{"function": _distance_difference, "target": 0.0}]},
    ]:
        opt = Optimizer(["Ar"] * 4, pos, methods=["rsirfo_bfgs"], **kwargs)  # type: ignore[arg-type]
        result = run(opt, NativeCalculator("lj"))
        assert result.converged, kwargs
        x = result.positions_bohr
        d12 = float(np.linalg.norm(x[0] - x[1]))
        d13 = float(np.linalg.norm(x[0] - x[2]))
        assert abs(d12 - d13) < 1e-6, kwargs
    # A bare callable holds the starting value.
    opt = Optimizer(["Ar"] * 4, pos, methods=["fire"], constraint_functions=[_distance_difference])
    result = run(opt, NativeCalculator("lj"))
    x = result.positions_bohr
    start = _distance_difference(pos, None, 0, False)[0]
    end = _distance_difference(x.ravel().tolist(), None, 0, False)[0]
    assert abs(end - start) < 1e-6


def test_checkpoints_resume_interrupted_runs(tmp_path: Any) -> None:
    from mopt_rs.optimize import Checkpoint

    def fresh() -> Optimizer:
        return Optimizer(["Ar"] * 7, (AR7 * ANG).ravel().tolist(), methods=["rsirfo_bfgs"])

    class Counting(NativeCalculator):
        calls = 0

        def evaluate(self, task: Task) -> Evaluation:
            Counting.calls += 1
            return super().evaluate(task)

    reference = run(fresh(), Counting("lj"))
    total = Counting.calls
    path = tmp_path / "run.ckpt"

    class Interrupt(Exception):
        pass

    def stop_after(n: int) -> Any:
        def cb(report: dict[str, Any]) -> bool:
            if report["iteration"] + 1 >= n:
                raise Interrupt
            return False

        return cb

    Counting.calls = 0
    with contextlib.suppress(Interrupt):
        run(fresh(), Counting("lj"), callback=stop_after(total // 2), checkpoint=path)
    first = Counting.calls
    with path.open("a") as f:
        f.write('{"truncated": ')  # killed while writing
    Counting.calls = 0
    ckpt = Checkpoint(path)
    resumed = run(fresh(), Counting("lj"), checkpoint=ckpt)
    ckpt.close()
    assert ckpt.replayed == first
    assert Counting.calls == total - first
    assert resumed.converged and resumed.iterations == reference.iterations
    assert np.array_equal(resumed.positions_bohr, reference.positions_bohr)


def test_spin_square_reaches_the_report() -> None:
    """``<S^2>`` told by a calculator is in the report of every iteration."""

    class Radical:
        def evaluate(self, task: Task) -> Evaluation:
            inner = NativeCalculator("lj").evaluate(task)
            return Evaluation(inner.energy, inner.gradient, inner.hessian, spin_square=0.7812)

    opt = Optimizer(
        ["Ar"] * 7,
        (AR7 * ANG).ravel().tolist(),
        methods=["fire"],
        multiplicity=2,
        max_iterations=3,
    )
    result = run(opt, Radical())
    assert len(result.reports) == 3
    for report in result.reports:
        ((label, value, exact),) = report["spin_squares"]
        assert isinstance(label, str)
        assert (value, exact) == (0.7812, 0.75)
    # A calculator that does not tell it: nothing is shown.
    quiet = Optimizer(["Ar"] * 7, (AR7 * ANG).ravel().tolist(), methods=["fire"], max_iterations=2)
    assert run(quiet, NativeCalculator("lj")).reports[0]["spin_squares"] == []


def test_run_optmain_returns_what_every_job_ended_with(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """MultiOptPy's ``Optimize.all_results``: one summary per input of a pattern."""
    monkeypatch.chdir(tmp_path)
    for name, scale in (("a1.xyz", 1.0), ("a2.xyz", 1.02), ("b.xyz", 1.0)):
        rows = "".join(f"Ar {x * scale} {y * scale} {z * scale}\n" for x, y, z in AR7)
        (tmp_path / name).write_text(f"7\n\n{rows}", encoding="utf-8")
    results = mopt_rs.run_optmain(["a*.xyz", "-os", "lj", "-opt", "rsirfo_bfgs"], console=False)
    assert [r["input_file"] for r in results] == ["a1.xyz", "a2.xyz"]
    for r in results:
        assert r["optimized_flag"] is True and r["interrupted"] is False
        assert r["final_energy"] < 0.0 and r["final_energy"] == r["final_bias_energy"]
        assert r["element_list"] == ["Ar"] * 7 and len(r["final_geometry"]) == 7
        assert r["iterations"] > 0 and isinstance(r["symmetry"], str)
        for key in ("optimized_struct_file", "traj_file", "folder_directory"):
            assert (tmp_path / r[key]).exists(), key
    with pytest.raises(Exception, match="no file matches"):
        mopt_rs.run_optmain(["z*.xyz", "-os", "lj"], console=False)


def test_run_counted_reports_the_evaluations_of_a_command(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The counts at the end of a run: gradients and exact Hessians, per input and in total."""
    from mopt_rs.cli import EvaluationCounts, run_counted

    monkeypatch.chdir(tmp_path)
    rows = "".join(f"Ar {x * 1.03} {y * 1.03} {z * 1.03}\n" for x, y, z in AR7)
    for name in ("a.xyz", "b.xyz"):
        (tmp_path / name).write_text(f"7\n\n{rows}", encoding="utf-8")
    code, counts = run_counted("optmain", ["a.xyz", "b.xyz", "-os", "lj", "-opt", "rsirfo_bfgs"])
    assert code == 0 and isinstance(counts, EvaluationCounts)
    assert counts.gradients > 2 and counts.hessians == 0 and counts.energies == 0
    logs = sorted(tmp_path.glob("**/mopt.log"))
    assert len(logs) == 2
    per_input = counts.gradients // 2
    for log in logs:
        text = log.read_text(encoding="utf-8")
        line = f"Evaluations (this input): {per_input} gradient calculation(s), 0 exact Hessian"
        assert line in text, text[-400:]
    # With the exact Hessian at every iteration every evaluation has one.
    code, exact = run_counted("optmain", ["a.xyz", "-os", "lj", "-opt", "rsirfo_bfgs", "-fc", "1"])
    assert code == 0 and exact.hessians == exact.gradients > 0
