"""The kinetics beyond the canonical model from Python: the single corrections, the direct count,
the statistics of trajectory pairs and a pure-Python run of them on a model surface, Markov
state models, the kernel contraction, kmc, and the network models and refinement tasks of a
network file.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from mopt_rs import kinetics


def test_the_registries_list_the_new_names() -> None:
    names = kinetics.part_names()
    for theory in ("eyring", "tst", "rrkm"):
        assert theory in names["theories"]
    for corr in ("kramers", "grote_hynes", "reactive_flux", "collins_kimball", "weak_coupling",
                 "small_curvature", "zero_curvature"):
        assert corr in names["corrections"]
    for model in ("canonical", "mass_action", "egme", "two_grain", "semi_markov"):
        assert model in names["models"]
    assert "kmc" in names["solvers"]


def test_single_corrections_have_their_limits() -> None:
    omega = 3.0e13
    assert math.isclose(kinetics.kramers(0.0, omega), 1.0)
    assert math.isclose(kinetics.kramers(2.0 * omega, omega), math.sqrt(2.0) - 1.0, rel_tol=1e-12)
    # Grote-Hynes without memory is Kramers.
    assert math.isclose(
        kinetics.grote_hynes(omega, gamma=2.0 * omega), math.sqrt(2.0) - 1.0, rel_tol=1e-10
    )
    # A memory far longer than the barrier time (omega tau = 3000): kappa close to 1.
    assert kinetics.grote_hynes(omega, gamma=2.0 * omega, memory=1e-10) > 0.99
    w = kinetics.weak_coupling(1e-6, 0.05, 2.5e-5, 100.0, 300.0, 0.1)
    assert w["p"] < 0.1
    assert math.isclose(w["kappa"], w["classical"] * math.exp(w["p"] ** 3 / 12.0), rel_tol=1e-8)
    s = np.linspace(-2.0, 2.0, 81)
    sct, zct = kinetics.curvature_tunnelling(s, -0.005 * s**2, np.full(s.size, 0.6), 300.0)
    assert sct > zct > 1.0


def test_the_direct_count_enumerates_the_levels() -> None:
    counts = kinetics.beyer_swinehart([20.0, 30.0], 10.0, 13)
    # 2 a + 3 b = j: 1 0 1 1 1 1 2 1 2 2 2 2 3.
    assert counts.tolist() == [1, 0, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3]


def test_branching_statistics_and_their_annotations() -> None:
    pairs = [(1.0, 1, 0)] * 60 + [(1.0, 2, 0)] * 30 + [(1.0, 0, 0)] * 10
    b = kinetics.branching(pairs, "count")
    assert (b.pairs, b.recrossing, b.undecided) == (100, 10, 0)
    assert math.isclose(b.kappa, 0.9)
    first = b.outcomes[0]
    assert first[:2] == (0, 1) and math.isclose(first[2], 2.0 / 3.0)
    doc = b.annotations("TS1", ["EQ0", "EQ1", "EQ2"])
    assert doc["edges"]["TS1"]["outcomes"]["from"] == "EQ0"


def test_saddle_trajectories_split_a_symmetric_bifurcation() -> None:
    b_, c, k0, k1, d = 0.02, 0.02, 0.01, 0.03, 0.02

    def surface(q: np.ndarray) -> tuple[float, np.ndarray]:
        x, y = q
        k = k0 - k1 * x
        v = -b_ * x * x / 2 + c * x**4 / 4 + k * y * y / 2 + d * y**4 / 4
        g = np.array([-b_ * x + c * x**3 - k1 * y * y / 2, k * y + d * y**3])
        return float(v), g

    def classify(q: np.ndarray) -> int | None:
        if q[0] < -0.3:
            return 0
        if q[0] > 0.6 and q[1] > 0.3:
            return 1
        if q[0] > 0.6 and q[1] < -0.3:
            return 2
        return None

    hessian = np.diag([-b_, k0])
    pairs = kinetics.saddle_trajectories(
        surface, np.zeros(2), hessian, classify, pairs=120, temperature=300.0,
        time_step=0.83, max_steps=20000, check_every=5, commit=3, seed=5,
    )
    b = kinetics.branching(pairs, "count")
    shares = {o[1]: o[2] for o in b.outcomes if o[0] == 0}
    total = shares.get(1, 0.0) + shares.get(2, 0.0)
    n = b.pairs - b.undecided - b.recrossing
    assert abs(shares.get(1, 0.0) / total - 0.5) < 3.0 * math.sqrt(0.25 / n)


def test_markov_state_models_and_dwell_times() -> None:
    rng = np.random.default_rng(1)
    t = np.array([[0.9, 0.2], [0.1, 0.8]])  # [to][from]
    s, d = 0, [0]
    for _ in range(50000):
        s = 0 if rng.random() < t[0, s] else 1
        d.append(s)
    m = kinetics.markov_state_model([d], 2, 1, 1.0)
    assert np.allclose(m["transition"], t, atol=0.01)
    pi = np.asarray(m["stationary"])
    assert np.allclose(pi, [2.0 / 3.0, 1.0 / 3.0], atol=0.01)
    dw = kinetics.dwell_times([d], 2, 1.0)
    # The residence of a state of a chain with stay probability 0.9: 1 / 0.1 frames.
    assert abs(dw["mean"][0] - 10.0) < 0.5


def test_the_kernel_contraction_of_a_markov_generator_is_closure_b() -> None:
    k = np.array([[0.0, 50.0, 0.0], [1.0, 0.0, 30.0], [0.0, 40.0, 0.0]])
    k0 = k - np.diag(k.sum(axis=0))
    r = kinetics.kernel_contraction(k0, np.zeros((3, 3)), [1])
    c = kinetics.contract(k, [1])
    gb = np.asarray(r["g"])
    assert np.allclose(gb.sum(axis=0), 0.0, atol=1e-10)
    assert np.allclose(np.asarray(r["d0"]), np.asarray(c.d), rtol=1e-10, atol=1e-10)


def test_kmc_samples_the_master_equation() -> None:
    k = np.array([[0.0, 0.5, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    exact = kinetics.solve(k, [1, 0, 0], [1.0], solver="expm").populations[0]
    sampled = kinetics.solve(k, [1, 0, 0], [1.0], solver="kmc", trajectories=20000, seed=4)
    se = np.sqrt(exact * (1 - exact) / 20000) + 1e-12
    assert np.all(np.abs(sampled.populations[0] - exact) < 3 * se)


def _water_network(dir_: Path) -> tuple[Path, Path]:
    water = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n"
    net = dir_ / "network.json"
    net.write_text(
        json.dumps(
            {
                "nodes": [
                    {"node_id": 0, "energy_hartree": 0.0},
                    {"node_id": 1, "energy_hartree": 0.002},
                ],
                "edges": [
                    {"edge_id": 0, "node_id_1": 0, "node_id_2": 1, "ts_energy_hartree": 0.04}
                ],
            }
        )
    )
    ann = dir_ / "ann.json"
    ann.write_text(
        json.dumps(
            {
                "nodes": {
                    "0": {"xyz": water, "frequencies_cm1": [1600, 3600, 3700]},
                    "1": {"xyz": water, "frequencies_cm1": [1500, 3500, 3650]},
                },
                "edges": {"TS0": {"xyz": water, "frequencies_cm1": [1200, 2800, -1500]}},
            }
        )
    )
    return net, ann


def test_egme_of_a_network_file(tmp_path: Path) -> None:
    net, ann = _water_network(tmp_path)
    m = kinetics.network_model(
        str(net), [1e-9, 1e-6], model="egme", annotations=str(ann), temperature=1500.0,
        grain=200.0, collision="strong", collision_frequency=1e22,
    )
    ka = np.asarray(m["contraction_a"])
    k_inf = np.asarray(m["high_pressure"])
    assert math.isclose(ka[1, 0], k_inf[1, 0], rel_tol=1e-6)
    p = np.asarray(m["populations"])
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_refinement_tasks_of_a_network_file(tmp_path: Path) -> None:
    net = tmp_path / "network.json"
    net.write_text(
        json.dumps(
            {
                "nodes": [
                    {"node_id": k, "energy_hartree": e}
                    for k, e in enumerate([0.0, -0.01, -0.05, -0.05])
                ],
                "edges": [
                    {"edge_id": 0, "node_id_1": 0, "node_id_2": 1, "ts_energy_hartree": 0.025},
                    {"edge_id": 1, "node_id_1": 1, "node_id_2": 2, "ts_energy_hartree": 0.0},
                ],
            }
        )
    )
    tasks = kinetics.refinement_tasks(str(net), [1e-3], [2], sigma=1.0)
    assert tasks[0][1] == "refine_energy"
    assert any(t[2] == "TS0" for t in tasks[:2])
