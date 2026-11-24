"""O1NumHess from Python: the state machine by itself, as the Hessian of an engine that has
none, and through ``model_hessian="o1numhess"``.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import numpy as np
import pytest

import mopt_rs
from mopt_rs import _core
from mopt_rs.calculators import (
    Capabilities,
    Engine,
    EngineCalculator,
    EngineResult,
    Structure,
)

ANG = _core.ANGSTROM_TO_BOHR


def alkane(n: int) -> tuple[list[str], np.ndarray]:
    """All-trans alkane with ``n`` carbons, slightly displaced (Bohr)."""
    symbols: list[str] = []
    rows: list[list[float]] = []
    for k in range(n):
        up = k % 2 == 1
        c = [1.277 * k, 0.861 if up else 0.0, 0.0]
        out = 0.63 if up else -0.63
        end = -0.609 if up else 0.609
        symbols += ["C", "H", "H"]
        rows += [c, [c[0], c[1] + out, 0.89], [c[0], c[1] + out, -0.89]]
        if k == 0:
            symbols.append("H")
            rows.append([c[0] - 0.904, c[1] + end, 0.0])
        if k == n - 1:
            symbols.append("H")
            rows.append([c[0] + 0.904, c[1] + end, 0.0])
    x = np.asarray(rows)
    i, a = np.indices(x.shape)
    return symbols, (x + 0.03 * np.sin((7 * i + 3 * a) * 1.37)) * ANG


class Chain(Engine):
    """Morse bonds between atoms closer than 2 Angstrom at the start, and nothing else."""

    name = "chain"
    capabilities = Capabilities(electrons=False)

    def __init__(self, start: np.ndarray) -> None:
        d = np.linalg.norm(start[:, None, :] - start[None, :, :], axis=-1)
        self.pairs = [(i, j, d[i, j]) for i in range(len(d)) for j in range(i) if d[i, j] < 2 * ANG]

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        x = structure.positions_bohr
        energy, gradient = 0.0, np.zeros_like(x)
        for i, j, r0 in self.pairs:
            v = x[i] - x[j]
            r = float(np.linalg.norm(v))
            e = np.exp(-(r - r0))
            energy += 0.1 * (1.0 - e) ** 2
            g = 0.2 * (1.0 - e) * e * v / r
            gradient[i] += g
            gradient[j] -= g
        return EngineResult(energy, gradient=gradient)


def test_the_state_machine_by_itself() -> None:
    symbols, x = alkane(8)
    flat = x.ravel().tolist()
    _, g0, reference = _core.evaluate_native("gfnff", symbols, flat, need_hessian=True)
    assert reference is not None
    job = _core.O1NumHess(symbols, flat, g0)
    with pytest.raises(RuntimeError, match="not finished"):
        job.hessian()
    rounds = 0
    evaluator = _core.NativeEvaluator("gfnff")
    while (points := job.ask()) is not None:
        job.tell([evaluator.evaluate(symbols, p, 0, 1, False, None)[1] for p in points])
        rounds += 1
    n = len(flat)
    h = np.asarray(job.hessian()).reshape(n, n)
    ref = np.asarray(reference).reshape(n, n)
    assert 0 < job.gradient_count < n
    assert np.allclose(h, h.T, atol=1e-12)
    assert np.abs(h - ref).max() < 0.01 * np.abs(ref).max()
    d = np.asarray(job.directions()).reshape(n, -1)
    assert np.allclose(d.T @ d, np.eye(d.shape[1]), atol=1e-10)
    assert job.residual < 1e-2 and rounds >= 1
    with pytest.raises(ValueError, match="differ in length"):
        _core.O1NumHess(symbols, flat, g0[:-1])


def test_engines_without_hessians_use_it_on_request() -> None:
    symbols, x = alkane(12)
    structure = Structure.build(symbols, x)
    central = EngineCalculator(Chain(x))
    reference = central.evaluate_structure(structure, need_hessian=True).hessian
    assert central.evaluations == 1 + 2 * x.size
    calc = EngineCalculator(Chain(x), hessian_method="o1numhess")
    h = calc.evaluate_structure(structure, need_hessian=True).hessian
    assert reference is not None and h is not None
    # Bonds alone leave many directions without curvature; those that come out slightly
    # negative are differentiated in the second round, which is why this is not fewer still.
    assert calc.evaluations < central.evaluations // 2
    assert np.abs(h - reference).max() < 0.01 * np.abs(reference).max()
    with pytest.raises(ValueError, match="hessian_method"):
        EngineCalculator(Chain(x), hessian_method="guess")


def test_model_hessian_o1numhess_asks_the_calculator() -> None:
    symbols, x = alkane(6)
    stretched = x * 1.03

    def run(model: str | None) -> tuple[mopt_rs.Result, EngineCalculator]:
        calc = EngineCalculator(Chain(x))
        opt = mopt_rs.Optimizer(
            symbols,
            stretched.ravel().tolist(),
            methods=["rsirfo_bfgs"],
            model_hessian=model,
            model_hessian_interval=5,
        )
        assert opt.wants_o1numhess == (model is not None)
        return mopt_rs.run(opt, calc), calc

    plain, _ = run(None)
    with_hessian, calc = run("o1numhess")
    assert plain.converged and with_hessian.converged
    assert calc.hessian_method == "o1numhess"
    assert with_hessian.iterations < plain.iterations
    # Exact Hessians take precedence, as with -fc.
    opt = mopt_rs.Optimizer(
        symbols,
        stretched.ravel().tolist(),
        methods=["rsirfo_bfgs"],
        model_hessian="o1numhess",
        exact_hessian_interval=5,
    )
    assert not opt.wants_o1numhess
