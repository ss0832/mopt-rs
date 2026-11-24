//! The kinetics of networks (`run_kinetics`, `run_rcmc`, the priority `kinetics` of the
//! mapper) against known answers (`docs/source/kinetics_beyond_tst.md`, section 6): the Eyring
//! rates and the clamp, the analytic solutions of two and three states, the models of section
//! 1.4 with the published RCMC and with contract-and-propagate, the error bound of
//! contract-and-propagate, the snapshot `q = V p` of the published method, MultiOptPy's numbers
//! for `legacy_rcmc` (golden data made by running MultiOptPy's `rcmc.py`), degrees of rate
//! control against finite differences and their exact identities, `tst` against the free
//! energies of the thermochemistry, and the command lines.
//!
//! MultiOptPy: `Utils/rcmc.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::Mat;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::core_infra::rng::Rng;
use mopt_rs::domain::Units;
use mopt_rs::kinetics::contract::{contract, greedy_order, TimeRule};
use mopt_rs::kinetics::expm::expm;
use mopt_rs::kinetics::network::{Bottleneck, KineticNetwork, KineticState};
use mopt_rs::kinetics::observe::{scores, Score};
use mopt_rs::kinetics::rate::{Eyring, RateCorrection, RateSet, Tst};
use mopt_rs::kinetics::sensitivity::rate_control;
use mopt_rs::kinetics::solve::{
    Closure, Evolution, Expm, KineticSolver, LegacyRcmc, Propagate, Rcmc, SolverRegistry,
    SolverSettings,
};
use mopt_rs::kinetics::{Clamp, Conditions, EnergyKind, Generator, KineticsSettings};
use mopt_rs::workflow::kinetics::multioptpy_network;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_kinetics_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn conditions(t: f64, clamp: Clamp) -> Conditions {
    Conditions {
        temperature: t,
        clamp,
        ..Conditions::default()
    }
}

fn rt(t: f64) -> f64 {
    Units::BOLTZMANN_HARTREE_PER_K * t
}

fn kt_h(t: f64) -> f64 {
    Units::BOLTZMANN_J_PER_K * t / Units::PLANCK_J_S
}

/// A network of states with energies `e` (Hartree) and saddle points `(a, b, E_ts)`.
fn network(e: &[f64], edges: &[(usize, usize, f64)]) -> KineticNetwork {
    KineticNetwork {
        states: e
            .iter()
            .enumerate()
            .map(|(k, &x)| KineticState::with_energy(k, &format!("EQ{k}"), x))
            .collect(),
        bottlenecks: edges
            .iter()
            .enumerate()
            .map(|(k, &(a, b, x))| Bottleneck::saddle(k, &format!("TS{k}"), [a, b], x))
            .collect(),
        ..KineticNetwork::default()
    }
}

fn rates_of(net: &KineticNetwork, c: &Conditions) -> RateSet {
    RateSet::assemble(net, &Eyring, &[], c).unwrap()
}

fn evolution(rates: &RateSet) -> Evolution {
    Evolution::Linear(rates.generator().unwrap())
}

fn delta(n: usize, k: usize) -> Vec<f64> {
    let mut p = vec![0.0; n];
    p[k] = 1.0;
    p
}

fn solve(solver: &mut dyn KineticSolver, ev: &Evolution, p0: &[f64], t: f64) -> Vec<f64> {
    let mut rng = Rng::from_seed(1);
    solver.solve(ev, p0, &[t], &mut rng).unwrap().populations[0].clone()
}

/// The energy (Hartree) above a minimum of an Eyring rate constant `k` (1/s) at `t`.
fn barrier(k: f64, t: f64) -> f64 {
    rt(t) * (kt_h(t) / k).ln()
}

#[test]
fn eyring_rates_and_the_two_clamps() {
    let t = 300.0;
    // Two minima and one saddle point: k = (k_B T / h) exp(-dE / RT); K pi = 0.
    let net = network(&[0.0, 0.003], &[(0, 1, 0.02)]);
    let r = rates_of(&net, &conditions(t, Clamp::Consistent));
    let g = r.generator().unwrap();
    let k01 = kt_h(t) * (-0.02 / rt(t)).exp();
    let k10 = kt_h(t) * (-(0.02 - 0.003) / rt(t)).exp();
    assert!((g.ln_rate(1, 0).exp() / k01 - 1.0).abs() < 1e-13);
    assert!((g.ln_rate(0, 1).exp() / k10 - 1.0).abs() < 1e-13);
    assert!(g.detailed_balance_residual().unwrap() < 1e-12);
    // A saddle point below the upper minimum: consistent keeps k_ab / k_ba = exp(-(E_b -
    // E_a) / RT); MultiOptPy's clamp gives (k_B T / h) exp(-(max(E_ts, E_a) - E_a) / RT).
    let net = network(&[0.0, 0.010], &[(0, 1, 0.006)]);
    let r = rates_of(&net, &conditions(t, Clamp::Consistent));
    let g = r.generator().unwrap();
    let ratio = (g.ln_rate(1, 0) - g.ln_rate(0, 1)).exp();
    assert!((ratio / (-0.010 / rt(t)).exp() - 1.0).abs() < 1e-12);
    assert!(r.steps[0].clamped && r.detailed_balance);
    let r = rates_of(&net, &conditions(t, Clamp::Legacy));
    assert!(!r.detailed_balance);
    let g = r.generator().unwrap();
    assert!((g.ln_rate(1, 0).exp() / (kt_h(t) * (-0.006 / rt(t)).exp()) - 1.0).abs() < 1e-13);
    assert!((g.ln_rate(0, 1).exp() / kt_h(t) - 1.0).abs() < 1e-13);
    // Parallel saddle points add.
    let net = network(&[0.0, 0.0], &[(0, 1, 0.02), (1, 0, 0.021)]);
    let g = rates_of(&net, &conditions(t, Clamp::Consistent))
        .generator()
        .unwrap();
    let want = kt_h(t) * ((-0.02 / rt(t)).exp() + (-0.021 / rt(t)).exp());
    assert!((g.ln_rate(1, 0).exp() / want - 1.0).abs() < 1e-13);
}

#[test]
fn two_and_three_states_have_their_analytic_solutions() {
    let t = 300.0;
    let c = conditions(t, Clamp::Consistent);
    // A <-> B: p_A(t) = pi_A + (1 - pi_A) exp(-(kf + kr) t).
    let (kf, kr) = (3.0_f64, 1.2_f64);
    let e_b = -rt(t) * (kf / kr).ln();
    let net = network(&[0.0, e_b], &[(0, 1, barrier(kf, t))]);
    let r = rates_of(&net, &c);
    let ev = evolution(&r);
    let p0 = delta(2, 0);
    for time in [0.01, 0.3, 2.0, 50.0] {
        let pi_a = kr / (kf + kr);
        let want = pi_a + (1.0 - pi_a) * (-(kf + kr) * time).exp();
        let mut expm_solver = Expm;
        let p = solve(&mut expm_solver, &ev, &p0, time);
        assert!((p[0] - want).abs() < 1e-12, "expm {time}: {} {want}", p[0]);
        let mut spectral = mopt_rs::kinetics::solve::Spectral;
        let p = solve(&mut spectral, &ev, &p0, time);
        assert!((p[0] - want).abs() < 1e-10, "spectral {time}");
        let mut prop = Propagate::new(1e-8, Closure::A).unwrap();
        let p = solve(&mut prop, &ev, &p0, time);
        assert!((p[0] - want).abs() < 1e-10, "propagate {time}");
    }
    // A -> B -> C with the back reactions negligible (products 60 RT lower): Bateman.
    let (k1, k2) = (2.0, 0.5);
    let net = network(
        &[0.0, -60.0 * rt(t), -120.0 * rt(t)],
        &[
            (0, 1, barrier(k1, t)),
            (1, 2, -60.0 * rt(t) + barrier(k2, t)),
        ],
    );
    let ev = evolution(&rates_of(&net, &c));
    for time in [0.1, 1.0, 4.0] {
        let a = (-k1 * time).exp();
        let b = k1 / (k2 - k1) * ((-k1 * time).exp() - (-k2 * time).exp());
        let p = solve(&mut Expm, &ev, &delta(3, 0), time);
        assert!(
            (p[0] - a).abs() < 1e-10 && (p[1] - b).abs() < 1e-10,
            "{time}: {p:?}"
        );
    }
    // A <-> B fast, B -> C slow: k_eff = k2 K / (1 + K) with K = [B] / [A].
    let (kab, kba, k2) = (1e6_f64, 2e6_f64, 1.0_f64);
    let e_b = -rt(t) * (kab / kba).ln();
    let net = network(
        &[0.0, e_b, -80.0 * rt(t)],
        &[(0, 1, barrier(kab, t)), (1, 2, e_b + barrier(k2, t))],
    );
    let ev = evolution(&rates_of(&net, &c));
    let k_eq = kab / kba;
    let k_eff = k2 * k_eq / (1.0 + k_eq);
    for time in [0.2, 1.0, 3.0] {
        let p = solve(&mut Expm, &ev, &delta(3, 0), time);
        let left = p[0] + p[1];
        assert!(
            (left - (-k_eff * time).exp()).abs() < 1e-5,
            "{time}: {left}"
        );
        let mut prop = Propagate::new(1e-8, Closure::A).unwrap();
        let q = solve(&mut prop, &ev, &delta(3, 0), time);
        assert!((q[0] + q[1] - left).abs() < 1e-6, "propagate {time}");
    }
}

/// The models of section 1.4: (network, time, the start state).
fn section_models(t: f64) -> Vec<(&'static str, KineticNetwork, f64)> {
    let r = rt(t);
    let mut out = Vec::new();
    // A start state with a lifetime of 100 s next to a fast pair.
    out.push((
        "slow_start_fast_pair",
        network(
            &[0.0, -20.0 * r, -20.5 * r],
            &[
                (0, 1, barrier(0.01, t)),
                (1, 2, -20.0 * r + barrier(1e6, t)),
            ],
        ),
        1.0,
    ));
    // A start state decaying at 1 1/s into ten degenerate conformers.
    let mut e = vec![0.0];
    e.extend([-15.0 * r; 10]);
    let mut edges = vec![(0, 1, barrier(1.0, t))];
    edges.extend((1..10).map(|k| (k, k + 1, -15.0 * r + barrier(1e5, t))));
    out.push(("ten_conformers", network(&e, &edges), 0.4));
    // Two states, 1e-9 1/s.
    out.push((
        "two_states",
        network(&[0.0, -2.0 * r], &[(0, 1, barrier(1e-9, t))]),
        1.0,
    ));
    out
}

#[test]
fn the_models_of_section_1_4_with_rcmc_and_propagate() {
    let t = 300.0;
    for (name, net, time) in section_models(t) {
        let r = rates_of(&net, &conditions(t, Clamp::Consistent));
        let ev = evolution(&r);
        let n = r.states.len();
        let p0 = delta(n, 0);
        let exact = solve(&mut Expm, &ev, &p0, time);
        let mut prop = Propagate::new(1e-8, Closure::A).unwrap();
        let p = solve(&mut prop, &ev, &p0, time);
        let mut rcmc = Rcmc::new(TimeRule::Gershgorin);
        let q = solve(&mut rcmc, &ev, &p0, time);
        eprintln!(
            "{name}: start state left exact {:.4}, propagate {:.4}, rcmc {:.4}",
            exact[0], p[0], q[0]
        );
        for i in 0..n {
            assert!((p[i] - exact[i]).abs() < 1e-3, "{name} propagate {i}");
        }
        // The published snapshot: the slow start state has not reacted.
        assert!((q[0] - 1.0).abs() < 1e-3, "{name} rcmc {}", q[0]);
    }
}

#[test]
fn conformers_in_fast_exchange_follow_curtin_hammett() {
    let t = 300.0;
    let r = rt(t);
    let c = conditions(t, Clamp::Consistent);
    // n degenerate conformers in fast exchange, one of which reacts with k: k / n.
    let (n, k) = (5_usize, 2.0);
    let mut e = vec![0.0; n];
    e.push(-60.0 * r);
    let mut edges: Vec<(usize, usize, f64)> =
        (0..n - 1).map(|i| (i, i + 1, barrier(1e8, t))).collect();
    edges.push((0, n, barrier(k, t)));
    let ev = evolution(&rates_of(&network(&e, &edges), &c));
    for time in [0.5, 2.0, 5.0] {
        let mut prop = Propagate::new(1e-8, Closure::A).unwrap();
        let p = solve(&mut prop, &ev, &delta(n + 1, 2), time);
        let left: f64 = p[..n].iter().sum();
        let want = (-k * time / n as f64).exp();
        assert!((left - want).abs() < 1e-7, "{time}: {left} {want}");
    }
    // Two conformers A, B (B 0.5 RT higher) in fast exchange, each with its own product:
    // the products form in the ratio of the saddle-point energies (Curtin-Hammett).
    let (ts_a, ts_b) = (barrier(1.0, t), barrier(1.0, t) - 1.2 * r);
    let net = network(
        &[0.0, 0.5 * r, -80.0 * r, -80.0 * r],
        &[(0, 1, barrier(1e9, t)), (0, 2, ts_a), (1, 3, ts_b)],
    );
    let ev = evolution(&rates_of(&net, &c));
    let p = solve(&mut Expm, &ev, &delta(4, 0), 100.0);
    let ratio = p[3] / p[2];
    let want = (-(ts_b - ts_a) / r).exp();
    assert!((ratio / want - 1.0).abs() < 1e-6, "{ratio} {want}");
}

#[test]
fn propagate_stays_within_its_error_bound() {
    let t = 300.0;
    // Four states: a fast pair (0, 1) and a slow pair (2, 3), joined by a step whose rate sets
    // the gap.
    for gap in [1e-1, 1e-3, 1e-5, 1e-8] {
        let fast = 1e6;
        let slow = fast * gap;
        let net = network(
            &[0.0, 0.3 * rt(t), -rt(t), -0.5 * rt(t)],
            &[
                (0, 1, barrier(fast, t)),
                (1, 2, 0.3 * rt(t) + barrier(slow, t)),
                (2, 3, -rt(t) + barrier(fast, t)),
            ],
        );
        let r = rates_of(&net, &conditions(t, Clamp::Consistent));
        let ev = evolution(&r);
        for time in [3.0 / fast, 10.0 / slow.max(1e-300), 1e3 / fast] {
            let exact = solve(&mut Expm, &ev, &delta(4, 0), time);
            let mut rng = Rng::from_seed(1);
            let mut prop = Propagate::new(1e-8, Closure::A).unwrap();
            let sol = prop.solve(&ev, &delta(4, 0), &[time], &mut rng).unwrap();
            let d = &sol.diagnostics[0];
            let err = (0..4)
                .map(|i| (sol.populations[0][i] - exact[i]).abs())
                .fold(0.0, f64::max);
            let bound = d
                .eps_hat
                .filter(|_| d.eliminated > 0)
                .unwrap_or(0.0)
                .max(1e-10);
            assert!(err <= bound, "gap {gap} t {time}: {err} > {bound}");
            if d.eliminated > 0 && d.eps_hat.unwrap_or(1.0) <= 1e-9 {
                assert!(err <= 1e-8);
            }
        }
    }
    // Random networks with and without separated time scales.
    for seed in 0..6 {
        let mut rng = Rng::from_seed(100 + seed);
        let n = 10;
        let e: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, 0.02)).collect();
        let mut edges = Vec::new();
        for i in 1..n {
            let j = rng.gen_range_usize(i);
            edges.push((i, j, e[i].max(e[j]) + rng.uniform(0.005, 0.04)));
        }
        let r = rates_of(&network(&e, &edges), &conditions(t, Clamp::Consistent));
        let ev = evolution(&r);
        for time in [1e-9, 1e-4, 1.0, 1e4] {
            let exact = solve(&mut Expm, &ev, &delta(n, 0), time);
            let mut prop = Propagate::new(1e-8, Closure::A).unwrap();
            let p = solve(&mut prop, &ev, &delta(n, 0), time);
            let err = (0..n).map(|i| (p[i] - exact[i]).abs()).fold(0.0, f64::max);
            assert!(err < 1e-8, "seed {seed} t {time}: {err}");
        }
    }
}

#[test]
fn the_snapshot_is_v_p_of_the_published_method() {
    // q = V p with V_TT = diag(1^T M)^-1, M = I + K_TS K_SS^-2 K_ST (Iwata et al.).
    let t = 300.0;
    let mut rng = Rng::from_seed(7);
    let n = 8;
    let e: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, 0.015)).collect();
    let mut edges = Vec::new();
    for i in 1..n {
        let j = rng.gen_range_usize(i);
        edges.push((i, j, e[i].max(e[j]) + rng.uniform(0.004, 0.03)));
    }
    let r = rates_of(&network(&e, &edges), &conditions(t, Clamp::Consistent));
    let rates = r.generator().unwrap().rates();
    let k = rates.matrix();
    let order = greedy_order(&rates, TimeRule::Gershgorin, true).unwrap();
    let p0: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    for m in 1..n - 1 {
        let c = contract(&rates, &order.prefix(m)).unwrap();
        let (tt, ss) = (&c.kept, &c.eliminated);
        let (nt, ns) = (tt.len(), ss.len());
        let kss = Mat::from_fn(ns, ns, |i, j| k[(ss[i], ss[j])]);
        let kst = Mat::from_fn(ns, nt, |i, j| k[(ss[i], tt[j])]);
        let kts = Mat::from_fn(nt, ns, |i, j| k[(tt[i], ss[j])]);
        let lu = kss.partial_piv_lu();
        let x = faer::linalg::solvers::Solve::solve(&lu, kst.as_ref()); // K_SS^-1 K_ST
        let y = faer::linalg::solvers::Solve::solve(&lu, x.as_ref()); // K_SS^-2 K_ST
        let mm = Mat::from_fn(nt, nt, |i, j| {
            let v: f64 = (0..ns).map(|q| kts[(i, q)] * y[(q, j)]).sum();
            v + if i == j { 1.0 } else { 0.0 }
        });
        let w: Vec<f64> = (0..nt).map(|j| (0..nt).map(|i| mm[(i, j)]).sum()).collect();
        let ps = Mat::from_fn(ns, 1, |i, _| p0[ss[i]]);
        let z = faer::linalg::solvers::Solve::solve(&lu, ps.as_ref()); // K_SS^-1 p_S
        let q_t: Vec<f64> = (0..nt)
            .map(|i| (p0[tt[i]] - (0..ns).map(|q| kts[(i, q)] * z[(q, 0)]).sum::<f64>()) / w[i])
            .collect();
        let ours = c.unfold_a(&c.lump(&p0), n);
        for i in 0..nt {
            assert!((ours[tt[i]] / q_t[i] - 1.0).abs() < 1e-10, "{m} {i}");
        }
        for (q, &s) in ss.iter().enumerate() {
            let want: f64 = -(0..nt).map(|i| x[(q, i)] * q_t[i]).sum::<f64>();
            assert!((ours[s] / want - 1.0).abs() < 1e-9, "{m} S {q}");
        }
    }
}

/// MultiOptPy's Boltzmann constant in Hartree/K (`Utils/rcmc.py:K_B_HARTREE`), with which the
/// reference populations of `rcmc_legacy.json` were computed (its prefactor `k_B T / h` uses the
/// exact SI values, as mopt-rs does).
const MOP_K_B_HARTREE: f64 = 3.166811563e-6;

/// The temperature at which mopt-rs' `k_B T` (CODATA 2018, 1.4e-10 above MultiOptPy's `k_B`)
/// equals MultiOptPy's `k_B T` at `t`: the Boltzmann factors then agree exactly and the
/// prefactor `k_B T / h` differs by 1.4e-10.
fn multioptpy_temperature(t: f64) -> f64 {
    t * MOP_K_B_HARTREE / Units::BOLTZMANN_HARTREE_PER_K
}

fn golden() -> JsonValue {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/rcmc_legacy.json");
    JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn close(a: f64, b: f64, rel: f64, floor: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(floor)
}

#[test]
fn legacy_rcmc_reproduces_multioptpy() {
    let doc = golden();
    let cases = doc.get("cases").and_then(JsonValue::as_array).unwrap();
    assert!(cases.len() >= 10);
    for case in cases {
        let name = case.get("name").and_then(JsonValue::as_str).unwrap();
        let temperature = case.get("temperature").and_then(JsonValue::as_f64).unwrap();
        let time = case.get("time").and_then(JsonValue::as_f64).unwrap();
        let energy =
            EnergyKind::parse(case.get("energy_type").and_then(JsonValue::as_str).unwrap())
                .unwrap();
        let start = case.get("start").and_then(JsonValue::as_usize);
        let net = multioptpy_network(case.get("network").unwrap(), energy, name).unwrap();
        let c = Conditions {
            temperature: multioptpy_temperature(temperature),
            energy,
            clamp: Clamp::Legacy,
            ..Conditions::default()
        };
        let r = rates_of(&net, &c);
        let ev = evolution(&r);
        let n = r.states.len();
        // The start state: the one given, or (MultiOptPy's mapper without one) the lowest.
        let lowest = (0..n).fold(0, |b, i| {
            if r.state_energies[i] < r.state_energies[b] {
                i
            } else {
                b
            }
        });
        let index = |id: usize| r.index_of(net.index_of(id).unwrap()).unwrap();
        let first = start.map_or(lowest, index);
        let mut solver = LegacyRcmc::new(start.is_some());
        let mut rng = Rng::from_seed(1);
        let sol = solver
            .solve(&ev, &delta(n, first), &[time], &mut rng)
            .unwrap();
        let pops = case
            .get("populations")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(pops.len(), n, "{name}");
        for (id, v) in pops {
            let want = v.as_f64().unwrap();
            let got = sol.populations[0][index(id.parse().unwrap())];
            assert!(close(got, want, 1e-9, 1e-12), "{name} EQ{id}: {got} {want}");
        }
        let Some(d) = case.get("D").and_then(JsonValue::as_array) else {
            continue;
        };
        let sup = sol.superstates[0].as_ref().unwrap();
        let kept: Vec<usize> = case
            .get("kept")
            .and_then(JsonValue::as_array)
            .unwrap()
            .iter()
            .map(|x| x.as_usize().unwrap())
            .collect();
        let ours: Vec<usize> = sup
            .kept
            .iter()
            .map(|&i| net.states[r.states[i]].id)
            .collect();
        assert_eq!(ours, kept, "{name}");
        let members: Vec<Vec<usize>> = case
            .get("members")
            .and_then(JsonValue::as_array)
            .unwrap()
            .iter()
            .map(|m| {
                m.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_usize().unwrap())
                    .collect()
            })
            .collect();
        let our_members: Vec<Vec<usize>> = sup
            .members
            .iter()
            .map(|m| m.iter().map(|&i| net.states[r.states[i]].id).collect())
            .collect();
        assert_eq!(our_members, members, "{name}");
        let largest = d
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap().abs())
            })
            .fold(0.0, f64::max);
        for (i, row) in d.iter().enumerate() {
            for (j, x) in row.as_array().unwrap().iter().enumerate() {
                let want = x.as_f64().unwrap();
                let got = sup.matrix[(i, j)];
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs().max(1e-12 * largest),
                    "{name} D[{i}][{j}]: {got} {want}"
                );
            }
        }
    }
}

/// Populations of the golden network `random7` from its node 0 at 1 s and 100 s by the matrix
/// exponential in 90-digit decimal arithmetic (Taylor series with scaling and squaring).
const RANDOM7_EXACT: [(f64, [f64; 7]); 2] = [
    (
        1.0,
        [
            1.569063997615562e-08,
            6.724384247090545e-01,
            4.613469549297054e-10,
            1.216_329_506_612_17e-2,
            2.000860483880107e-03,
            7.930397623123616e-04,
            3.126043638266444e-01,
        ],
    ),
    (
        100.0,
        [
            1.569063983504298e-08,
            6.724384186615276e-01,
            9.454775292875888e-09,
            1.216329495673198e-02,
            2.000860465885512e-03,
            7.930397551802156e-04,
            3.126043610152595e-01,
        ],
    ),
];

#[test]
fn propagate_and_expm_against_high_precision() {
    let doc = golden();
    let case = doc
        .get("cases")
        .and_then(JsonValue::as_array)
        .unwrap()
        .iter()
        .find(|c| c.get("name").and_then(JsonValue::as_str) == Some("random7_t1"))
        .unwrap()
        .clone();
    let net =
        multioptpy_network(case.get("network").unwrap(), EnergyKind::Electronic, "r7").unwrap();
    let r = rates_of(
        &net,
        &conditions(multioptpy_temperature(300.0), Clamp::Consistent),
    );
    let ev = evolution(&r);
    for (t, exact) in RANDOM7_EXACT {
        let p0 = delta(7, 0);
        let mut prop = Propagate::new(1e-8, Closure::A).unwrap();
        let a = solve(&mut prop, &ev, &p0, t);
        let b = solve(&mut Expm, &ev, &p0, t);
        let err = |p: &[f64]| (0..7).map(|i| (p[i] - exact[i]).abs()).fold(0.0, f64::max);
        eprintln!("t = {t}: propagate {:.2e}, expm {:.2e}", err(&a), err(&b));
        assert!(err(&a) < 1e-11 && err(&b) < 1e-11);
        // The relative error of the smallest population.
        assert!(
            (b[2] / exact[2] - 1.0).abs() < 1e-9,
            "{} {}",
            b[2],
            exact[2]
        );
        // The Pade form loses the small rates of the stiff generator.
        let k = r.generator().unwrap().rates().matrix();
        let pade = expm(&Mat::from_fn(7, 7, |i, j| k[(i, j)] * t)).unwrap();
        let c: Vec<f64> = (0..7).map(|i| pade[(i, 0)]).collect();
        eprintln!("t = {t}: Pade {:.2e}", err(&c));
    }
}

#[test]
fn run_rcmc_writes_multioptpy_s_file() {
    let doc = golden();
    let cases = doc.get("cases").and_then(JsonValue::as_array).unwrap();
    let dir = scratch("rcmc_cli");
    for case in cases.iter().filter(|c| c.get("csv").is_some()) {
        let name = case.get("name").and_then(JsonValue::as_str).unwrap();
        let file = dir.join(format!("{name}.json"));
        std::fs::write(&file, case.get("network").unwrap().to_string_compact()).unwrap();
        let out = dir.join(format!("{name}_out"));
        let args: Vec<String> = [
            file.display().to_string(),
            "-T".into(),
            format!(
                "{:?}",
                case.get("temperature").and_then(JsonValue::as_f64).unwrap()
            ),
            "-t".into(),
            format!(
                "{:e}",
                case.get("time").and_then(JsonValue::as_f64).unwrap()
            ),
            "--start-node".into(),
            "0".into(),
            "--output-dir".into(),
            out.display().to_string(),
            "--energy-type".into(),
            case.get("energy_type")
                .and_then(JsonValue::as_str)
                .unwrap()
                .to_string(),
        ]
        .into();
        assert_eq!(mopt_rs::cli::run_command("run_rcmc", &args), 0, "{name}");
        let ours = std::fs::read_to_string(out.join("rcmc_K_contracted.csv")).unwrap();
        let want = case.get("csv").and_then(JsonValue::as_str).unwrap();
        let (a, b): (Vec<&str>, Vec<&str>) = (ours.lines().collect(), want.lines().collect());
        assert_eq!(a.len(), b.len(), "{name}:\n{ours}\n{want}");
        for (x, y) in a.iter().zip(&b) {
            if x.starts_with('#') || x.starts_with("node") || x.is_empty() {
                assert_eq!(x, y, "{name}");
                continue;
            }
            let (xs, ys): (Vec<&str>, Vec<&str>) = (x.split(',').collect(), y.split(',').collect());
            assert_eq!(xs[0], ys[0], "{name}");
            for (u, v) in xs[1..].iter().zip(&ys[1..]) {
                let (u, v): (f64, f64) = (u.parse().unwrap(), v.parse().unwrap());
                assert!(close(u, v, 1e-5, 1e-300), "{name}: {x} against {y}");
            }
        }
    }
}

#[test]
fn degrees_of_rate_control_match_finite_differences_and_the_identities() {
    let t = 300.0;
    let mut rng = Rng::from_seed(3);
    let n = 6;
    let e: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, 0.01)).collect();
    let mut edges = Vec::new();
    for i in 1..n {
        let j = rng.gen_range_usize(i);
        edges.push((i, j, e[i].max(e[j]) + rng.uniform(0.01, 0.025)));
    }
    edges.push((0, n - 1, e[0].max(e[n - 1]) + 0.02));
    let net = network(&e, &edges);
    let c = conditions(t, Clamp::Consistent);
    let r = rates_of(&net, &c);
    let p0 = delta(n, 0);
    let time = 1e-3;
    let target = n - 1;
    let x = rate_control(&r, &p0, time, target).unwrap();
    // Exact identities: scaling time, and shifting every free energy.
    let sum_e: f64 = x.steps.iter().sum();
    let sum_j: f64 = x.states.iter().sum();
    assert!((sum_e - x.time_scaling).abs() < 1e-6 * (1.0 + x.time_scaling.abs()));
    assert!((sum_e + sum_j).abs() < 1e-6 * (1.0 + sum_e.abs()));
    // Finite differences of the saddle-point energies: X_e = -RT d ln y / dE_ts.
    let h = 1e-6 * rt(t);
    let y_of = |net: &KineticNetwork| -> f64 {
        let r = rates_of(net, &c);
        solve(&mut Expm, &evolution(&r), &p0, time)[target]
    };
    for (k, xe) in x.steps.iter().enumerate() {
        let mut up = net.clone();
        up.bottlenecks[k].energy += h;
        let mut down = net.clone();
        down.bottlenecks[k].energy -= h;
        let fd = -(y_of(&up).ln() - y_of(&down).ln()) / (2.0 * h) * rt(t);
        assert!(
            (fd - xe).abs() < 1e-6 * (1.0 + xe.abs()),
            "step {k}: {fd} {xe}"
        );
    }
    for (j, xj) in x.states.iter().enumerate() {
        let mut up = net.clone();
        up.states[j].energy += h;
        let mut down = net.clone();
        down.states[j].energy -= h;
        let fd = -(y_of(&up).ln() - y_of(&down).ln()) / (2.0 * h) * rt(t);
        assert!(
            (fd - xj).abs() < 1e-6 * (1.0 + xj.abs()),
            "state {j}: {fd} {xj}"
        );
    }
}

fn settings_with(solver: &str, t: f64) -> KineticsSettings {
    KineticsSettings {
        conditions: conditions(t, Clamp::Consistent),
        solver: solver.to_string(),
        ..KineticsSettings::default()
    }
}

/// Populations at `times` (and the final ones last) by `expm` / `steady` for the reference.
fn exact_populations(net: &KineticNetwork, t: f64, times: &[f64]) -> Vec<Vec<f64>> {
    let r = RateSet::assemble(net, &Eyring, &[], &conditions(t, Clamp::Consistent)).unwrap();
    let ev = evolution(&r);
    let mut rng = Rng::from_seed(1);
    let p0 = delta(r.states.len(), 0);
    let mut out = Expm.solve(&ev, &p0, times, &mut rng).unwrap().populations;
    let fin = mopt_rs::kinetics::solve::Steady
        .solve(&ev, &p0, &[1.0], &mut rng)
        .unwrap()
        .populations;
    out.push(fin[0].clone());
    out
}

/// The 4-state networks of the derivative checks: one without and one with stiffness.
fn derivative_networks(t: f64) -> Vec<(&'static str, KineticNetwork, Vec<f64>)> {
    let r = rt(t);
    let plain = network(
        &[0.0, 0.8 * r, -0.5 * r, 1.5 * r],
        &[
            (0, 1, barrier(2.0, t)),
            (1, 2, 0.8 * r + barrier(1.5, t)),
            (2, 3, 1.5 * r + barrier(0.7, t)),
            (0, 3, 1.5 * r + barrier(0.3, t)),
        ],
    );
    // Rate constants from 1e-3 to 1e9 1/s.
    let stiff = network(
        &[0.0, 2.0 * r, -r, 0.5 * r],
        &[
            (0, 1, 2.0 * r + barrier(1e9, t)),
            (1, 2, 2.0 * r + barrier(1e-3, t)),
            (2, 3, 0.5 * r + barrier(1e5, t)),
            (0, 3, 0.5 * r + barrier(1.0, t)),
        ],
    );
    vec![
        ("plain", plain, vec![0.05, 0.4, 2.0]),
        ("stiff", stiff, vec![1e-8, 1e-3, 10.0, 1e4]),
    ]
}

#[test]
fn yield_derivatives_match_central_differences() {
    use mopt_rs::kinetics::derivative::{derivatives, Parameter};
    use mopt_rs::kinetics::KineticsParts;
    let t = 300.0;
    let parts = KineticsParts::default();
    for (name, net, times) in derivative_networks(t) {
        for solver in ["expm", "propagate"] {
            let start = std::time::Instant::now();
            let d =
                derivatives(&net, &settings_with(solver, t), &parts, &[(0, 1.0)], &times).unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            let mut worst: f64 = 0.0;
            for (m, info) in d.parameters.iter().enumerate() {
                // Central differences of the exact populations, with Richardson's
                // extrapolation of the steps h and h / 2 (error of the order of h^4).
                let shifted = |x: f64| -> Vec<Vec<f64>> {
                    match info.parameter {
                        Parameter::StateEnergy(k) => {
                            let mut a = net.clone();
                            a.states[k].energy += x;
                            exact_populations(&a, t, &times)
                        }
                        Parameter::BottleneckEnergy(k) => {
                            let mut a = net.clone();
                            a.bottlenecks[k].energy += x;
                            exact_populations(&a, t, &times)
                        }
                        // ln k up by x: the saddle point lower by k_B T x.
                        Parameter::LnK(k) => {
                            let mut a = net.clone();
                            a.bottlenecks[k].energy -= x * rt(t);
                            exact_populations(&a, t, &times)
                        }
                        Parameter::Temperature => exact_populations(&net, t + x, &times),
                        Parameter::LnConcentration(_) | Parameter::SpeciesEnergy(_) => {
                            unreachable!("no species")
                        }
                    }
                };
                let h = match info.parameter {
                    Parameter::StateEnergy(_) | Parameter::BottleneckEnergy(_) => 1e-3 * rt(t),
                    Parameter::LnK(_) | Parameter::LnConcentration(_) => 1e-3,
                    Parameter::SpeciesEnergy(_) => 1e-3 * rt(t),
                    Parameter::Temperature => 0.3,
                };
                let (u1, d1) = (shifted(h), shifted(-h));
                let (u2, d2) = (shifted(h / 2.0), shifted(-h / 2.0));
                for k in 0..=times.len() {
                    let ad = if k < times.len() {
                        &d.d_populations[m][k]
                    } else {
                        &d.d_final[m]
                    };
                    let fd: Vec<f64> = (0..4)
                        .map(|i| {
                            let wide = (u1[k][i] - d1[k][i]) / (2.0 * h);
                            let narrow = (u2[k][i] - d2[k][i]) / h;
                            (4.0 * narrow - wide) / 3.0
                        })
                        .collect();
                    // Derivatives that vanish are compared on the scale of the parameter (the
                    // rounding noise of the differences is about 1e-11 there).
                    let natural = match info.parameter {
                        Parameter::StateEnergy(_) | Parameter::BottleneckEnergy(_) => 1.0 / rt(t),
                        Parameter::LnK(_) | Parameter::LnConcentration(_) => 1.0,
                        Parameter::SpeciesEnergy(_) => 1.0 / rt(t),
                        Parameter::Temperature => 1.0 / t,
                    };
                    let scale = fd.iter().fold(1e-3 * natural, |s, x| s.max(x.abs()));
                    for i in 0..4 {
                        let err = (ad[i] - fd[i]).abs() / scale;
                        worst = worst.max(err);
                        assert!(
                            err < 1e-6,
                            "{name} {solver} {} k={k} i={i}: {} {}",
                            info.label,
                            ad[i],
                            fd[i]
                        );
                    }
                }
            }
            eprintln!(
                "{name} {solver}: {} parameters, {} passes, {elapsed:.3} s, largest relative \
                 difference to central differences {worst:.1e}",
                d.parameters.len(),
                d.passes
            );
        }
    }
}

#[test]
fn final_yields_and_the_identities_of_the_degrees_of_rate_control() {
    use mopt_rs::kinetics::derivative::{derivatives, Parameter};
    use mopt_rs::kinetics::sensitivity::rate_control;
    use mopt_rs::kinetics::KineticsParts;
    let t = 300.0;
    let (_, net, times) = derivative_networks(t).remove(0);
    let d = derivatives(
        &net,
        &settings_with("expm", t),
        &KineticsParts::default(),
        &[(0, 1.0)],
        &times,
    )
    .unwrap();
    let kt = rt(t);
    let pf = &d.final_populations;
    for (m, info) in d.parameters.iter().enumerate() {
        for i in 0..4 {
            let want = match info.parameter {
                // The Boltzmann distribution: dp_i / dE_j = -(1/kT) p_i (delta_ij - p_j).
                Parameter::StateEnergy(j) => -pf[i] * (if i == j { 1.0 } else { 0.0 } - pf[j]) / kt,
                Parameter::BottleneckEnergy(_) | Parameter::LnK(_) => 0.0,
                Parameter::Temperature
                | Parameter::LnConcentration(_)
                | Parameter::SpeciesEnergy(_) => d.d_final[m][i],
            };
            assert!(
                (d.d_final[m][i] - want).abs() < 1e-9 / kt,
                "{} {i}",
                info.label
            );
        }
    }
    // Sum over the steps of d ln p / d ln k = t (dp/dt) / p; with the energies of the states
    // it is 0; the adjoint of the exact exponential gives the same numbers.
    let r = rates_of(&net, &conditions(t, Clamp::Consistent));
    let km = r.generator().unwrap().rates().matrix();
    for (k, &time) in times.iter().enumerate() {
        let p = &d.populations[k];
        for i in 0..4 {
            let lnk: Vec<f64> = d
                .parameters
                .iter()
                .enumerate()
                .filter(|(_, x)| matches!(x.parameter, Parameter::LnK(_)))
                .map(|(m, _)| d.d_populations[m][k][i] / p[i])
                .collect();
            let energies: f64 = d
                .parameters
                .iter()
                .enumerate()
                .filter(|(_, x)| matches!(x.parameter, Parameter::StateEnergy(_)))
                .map(|(m, _)| -kt * d.d_populations[m][k][i] / p[i])
                .sum();
            let rate: f64 = (0..4).map(|j| km[(i, j)] * p[j]).sum();
            let sum: f64 = lnk.iter().sum();
            assert!((sum - time * rate / p[i]).abs() < 1e-9, "{time} {i}: {sum}");
            assert!((sum + energies).abs() < 1e-9, "{time} {i}");
            let adjoint = rate_control(&r, &delta(4, 0), time, i).unwrap();
            for (a, b) in lnk.iter().zip(&adjoint.steps) {
                assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{a} {b}");
            }
        }
    }
}

#[test]
fn effective_rate_constants_have_degrees_of_rate_control_summing_to_one() {
    use mopt_rs::kinetics::derivative::{derivatives, Parameter};
    use mopt_rs::kinetics::KineticsParts;
    let t = 300.0;
    let r = rt(t);
    // A <-> I <-> P with the short-lived I: k_eff(A -> P) = k1 k2 / (k-1 + k2).
    let (k1, km1, k2) = (2.0_f64, 3e8_f64, 1e8_f64);
    let e_i = -(r * (k1 / km1).ln());
    let net = network(
        &[0.0, e_i, -3.0 * r],
        &[(0, 1, barrier(k1, t)), (1, 2, e_i + barrier(k2, t))],
    );
    for solver in ["propagate", "rcmc"] {
        let d = derivatives(
            &net,
            &settings_with(solver, t),
            &KineticsParts::default(),
            &[(0, 1.0)],
            &[1e-3],
        )
        .unwrap();
        let (kept, g) = d.superstates[0].clone().unwrap();
        assert_eq!(kept, vec![0, 2], "{solver}: I is eliminated");
        let nt = kept.len();
        let g_pa = g[nt]; // from A (column 0) to P (row 1)
        let lnk: Vec<(String, f64)> = d
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, x)| matches!(x.parameter, Parameter::LnK(_)))
            .map(|(m, x)| {
                (
                    x.label.clone(),
                    d.d_superstates[m][0].as_ref().unwrap()[nt] / g_pa,
                )
            })
            .collect();
        let sum: f64 = lnk.iter().map(|(_, x)| x).sum();
        assert!((sum - 1.0).abs() < 1e-12, "{solver}: {sum}");
        let want = [k2 / (km1 + k2), km1 / (km1 + k2)];
        for ((label, x), w) in lnk.iter().zip(want) {
            assert!((x - w).abs() < 1e-7, "{solver} {label}: {x} {w}");
        }
        assert!((g_pa / (k1 * k2 / (km1 + k2)) - 1.0).abs() < 1e-7);
    }
}

#[test]
fn derivatives_of_a_network_of_thirty_states() {
    use mopt_rs::kinetics::derivative::derivatives;
    use mopt_rs::kinetics::KineticsParts;
    let t = 300.0;
    let mut rng = Rng::from_seed(9);
    let n = 30;
    let e: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, 0.02)).collect();
    let mut edges = Vec::new();
    for i in 1..n {
        let j = rng.gen_range_usize(i);
        edges.push((i, j, e[i].max(e[j]) + rng.uniform(0.005, 0.04)));
        if i > 3 && rng.next_f64() < 0.4 {
            let k = rng.gen_range_usize(i - 1);
            edges.push((i, k, e[i].max(e[k]) + rng.uniform(0.005, 0.04)));
        }
    }
    let net = network(&e, &edges);
    for solver in ["propagate", "expm"] {
        let start = std::time::Instant::now();
        let d = derivatives(
            &net,
            &settings_with(solver, t),
            &KineticsParts::default(),
            &[(0, 1.0)],
            &[1e-9, 1e-3, 1.0],
        )
        .unwrap();
        eprintln!(
            "30 states, {} steps, {solver}: {} parameters in {} passes, {:.2} s",
            edges.len(),
            d.parameters.len(),
            d.passes,
            start.elapsed().as_secs_f64()
        );
        // Every population sums to 1, so every derivative of the sum is 0.
        for dm in &d.d_populations {
            for dp in dm {
                assert!(
                    dp.iter().sum::<f64>().abs()
                        < 1e-8 * (1.0 + dp.iter().map(|x| x.abs()).sum::<f64>())
                );
            }
        }
    }
}

#[test]
fn tst_uses_the_free_energies_of_the_thermochemistry() {
    use mopt_rs::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge};
    use mopt_rs::kinetics::rate::QrrhoSettings;
    let water = |x: f64| {
        MolecularSystem::new(
            vec![
                Element::parse_token("O").unwrap(),
                Element::parse_token("H").unwrap(),
                Element::parse_token("H").unwrap(),
            ],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [x, 0.93, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    };
    let t = 298.15;
    let mut net = network(&[0.0, 0.002], &[(0, 1, 0.03)]);
    net.states[0].system = Some(water(-0.24));
    net.states[1].system = Some(water(-0.20));
    net.states[0].frequencies = Some(vec![1600.0, 3700.0, 3800.0]);
    net.states[1].frequencies = Some(vec![1550.0, 3690.0, 3790.0]);
    net.bottlenecks[0].system = Some(water(-0.22));
    net.bottlenecks[0].frequencies = Some(vec![-900.0, 3650.0, 3750.0]);
    let c = conditions(t, Clamp::Consistent);
    let tst = Tst::new(QrrhoSettings::default());
    let r = RateSet::assemble(&net, &tst, &[], &c).unwrap();
    let g = r.generator().unwrap();
    let gibbs =
        |sys: &MolecularSystem, f: &[f64], e: f64| tst.gibbs(sys, f, e, 1, None, &c).unwrap();
    let g0 = gibbs(
        net.states[0].system.as_ref().unwrap(),
        net.states[0].frequencies.as_ref().unwrap(),
        0.0,
    );
    let gts = gibbs(
        net.bottlenecks[0].system.as_ref().unwrap(),
        net.bottlenecks[0].frequencies.as_ref().unwrap(),
        0.03,
    );
    let want = kt_h(t) * (-(gts - g0) / rt(t)).exp();
    assert!((g.ln_rate(1, 0).exp() / want - 1.0).abs() < 1e-10);
    // The vibrational part alone, in closed form (the same rotor and translation): without
    // the low-frequency treatment, G_vib = sum (h nu / 2 + k T ln(1 - exp(-h nu / k T))).
    let plain = Tst::new(QrrhoSettings {
        method: mopt_rs::vib::QuasiHarmonic::None,
        ..QrrhoSettings::default()
    });
    let same = water(-0.24);
    let hc = Units::PLANCK_J_S * Units::SPEED_OF_LIGHT_CM_S / Units::HARTREE_TO_JOULE;
    let kt = mopt_rs::vib::boltzmann_hartree() * t;
    let vib = |f: &[f64]| -> f64 {
        f.iter()
            .map(|w| {
                let x = hc * w;
                x / 2.0 + kt * (-(-x / kt).exp()).ln_1p()
            })
            .sum()
    };
    let a = plain
        .gibbs(&same, &[1000.0, 2000.0, 3000.0], 0.0, 1, Some(2), &c)
        .unwrap();
    let b = plain
        .gibbs(&same, &[1200.0, 2100.0, 2900.0], 0.0, 1, Some(2), &c)
        .unwrap();
    let want = vib(&[1200.0, 2100.0, 2900.0]) - vib(&[1000.0, 2000.0, 3000.0]);
    assert!(
        ((b - a) - want).abs() < 1e-7 * want.abs(),
        "{} {want}",
        b - a
    );
    // The symmetry number enters as RT ln sigma.
    let s1 = plain.gibbs(&same, &[1000.0], 0.0, 1, Some(1), &c).unwrap();
    let s2 = plain.gibbs(&same, &[1000.0], 0.0, 1, Some(2), &c).unwrap();
    assert!(((s2 - s1) - kt * 2.0_f64.ln()).abs() < 1e-12);
}

#[test]
fn the_priorities_and_the_registries() {
    let t = 300.0;
    let net = network(
        &[0.0, 0.002, -0.001, 0.004],
        &[(0, 1, 0.012), (1, 2, 0.011), (2, 3, 0.015)],
    );
    let r = rates_of(&net, &conditions(t, Clamp::Consistent));
    let ev = evolution(&r);
    let p0 = delta(4, 0);
    let mut rng = Rng::from_seed(1);
    let mut prop = Propagate::new(1e-8, Closure::A).unwrap();
    let pop = scores(&mut prop, &ev, &p0, 1e-6, Score::Population, None, &mut rng).unwrap();
    let exact = solve(&mut Expm, &ev, &p0, 1e-6);
    for i in 0..4 {
        assert!((pop[i] - exact[i]).abs() < 1e-9);
    }
    // The residence time of A <-> B, int_0^t p_A = pi_A t + (1 - pi_A)(1 - e^-kt) / k.
    let mut expm_solver = Expm;
    let two = network(&[0.0, 0.001], &[(0, 1, 0.01)]);
    let r2 = rates_of(&two, &conditions(t, Clamp::Consistent));
    let g2 = r2.generator().unwrap();
    let (kf, kr) = (g2.ln_rate(1, 0).exp(), g2.ln_rate(0, 1).exp());
    let time = 3.0 / (kf + kr);
    let tau = scores(
        &mut expm_solver,
        &evolution(&r2),
        &delta(2, 0),
        time,
        Score::Residence,
        None,
        &mut rng,
    )
    .unwrap();
    let pi = kr / (kf + kr);
    let want = pi * time + (1.0 - pi) * (1.0 - (-(kf + kr) * time).exp()) / (kf + kr);
    assert!((tau[0] / want - 1.0).abs() < 1e-10, "{} {want}", tau[0]);
    let mut rcmc = Rcmc::new(TimeRule::Gershgorin);
    let tau_q = scores(
        &mut rcmc,
        &evolution(&r2),
        &delta(2, 0),
        time,
        Score::Residence,
        None,
        &mut rng,
    )
    .unwrap();
    assert!(tau_q[0] > 0.0 && tau_q[0] <= time * 1.0001);
    // max_population of an intermediate passed early.
    let mp = scores(
        &mut prop,
        &ev,
        &p0,
        1.0,
        Score::MaxPopulation,
        None,
        &mut rng,
    )
    .unwrap();
    let late = solve(&mut Expm, &ev, &p0, 1.0);
    assert!(mp[1] >= late[1] && mp[0] == 1.0);
    // discovery_flux: tau k_guess / (1 + n).
    let df = scores(
        &mut prop,
        &ev,
        &p0,
        1e-6,
        Score::DiscoveryFlux,
        Some(&[0, 3, 0, 1]),
        &mut rng,
    )
    .unwrap();
    assert!(df.iter().all(|x| x.is_finite() && *x >= 0.0));
    // Relabelling the states does not change the scores.
    let perm = [2, 0, 3, 1];
    let mut e2 = vec![0.0; 4];
    for (i, &k) in perm.iter().enumerate() {
        e2[k] = net.states[i].energy;
    }
    let edges2: Vec<(usize, usize, f64)> = net
        .bottlenecks
        .iter()
        .map(|b| (perm[b.states[0]], perm[b.states[1]], b.energy))
        .collect();
    let r3 = rates_of(&network(&e2, &edges2), &conditions(t, Clamp::Consistent));
    let pop2 = scores(
        &mut prop,
        &evolution(&r3),
        &delta(4, perm[0]),
        1e-6,
        Score::Population,
        None,
        &mut rng,
    )
    .unwrap();
    for i in 0..4 {
        assert!((pop2[perm[i]] - pop[i]).abs() < 1e-12);
    }
    // Names that are not implemented are refused by name.
    let reg = SolverRegistry::default();
    for name in ["rcmc_b", "ode"] {
        let e = reg.create(name, &SolverSettings::default()).err().unwrap();
        assert!(e.to_string().contains(name), "{e}");
    }
    assert!(reg.create("nope", &SolverSettings::default()).is_err());
    let models = mopt_rs::kinetics::ModelRegistry::default();
    let ms = mopt_rs::kinetics::ModelSettings::default();
    for name in [
        "canonical",
        "mass_action",
        "egme",
        "two_grain",
        "semi_markov",
    ] {
        assert!(models.create(name, &ms).is_ok(), "{name}");
    }
    assert!(models.create("nope", &ms).is_err());
    for name in ["kmc", "contract", "windows"] {
        assert!(
            reg.create(name, &SolverSettings::default()).is_ok(),
            "{name}"
        );
    }
    let theories = mopt_rs::kinetics::RateTheoryRegistry::default();
    assert!(theories.create("rrkm", &Default::default()).is_ok());
    let corrections = mopt_rs::kinetics::RateCorrectionRegistry::default();
    for name in [
        "wigner",
        "eckart",
        "skodje_truhlar",
        "kappa",
        "landau_zener",
        "zhu_nakamura",
        "weak_coupling",
        "kramers",
        "grote_hynes",
        "reactive_flux",
        "collins_kimball",
        "small_curvature",
        "zero_curvature",
    ] {
        let c: Arc<dyn RateCorrection> = corrections.create(name, &Default::default()).unwrap();
        assert_eq!(c.name(), name);
    }
    for name in ["marcus", "symmetry"] {
        let e = corrections.create(name, &Default::default()).err().unwrap();
        assert!(e.to_string().contains("not implemented"), "{e}");
    }
}

#[test]
fn a_generator_of_a_matrix_has_the_solutions_of_its_matrix() {
    // The Python API passes matrices: the solutions are those of exp(t K).
    let k = vec![
        vec![0.0, 2.0, 0.0],
        vec![5.0, 0.0, 1e-3],
        vec![0.0, 7.0, 0.0],
    ];
    let g = Generator::from_matrix(&k).unwrap();
    let ev = Evolution::Linear(g.clone());
    let km = g.rates().matrix();
    let e = expm(&Mat::from_fn(3, 3, |i, j| km[(i, j)] * 0.7)).unwrap();
    let p = solve(&mut Expm, &ev, &[1.0, 0.0, 0.0], 0.7);
    for i in 0..3 {
        assert!((p[i] - e[(i, 0)]).abs() < 1e-15);
    }
    // Without detailed balance the eigen rule and the spectral solver are refused.
    let rates = g.rates();
    assert!(greedy_order(&rates, TimeRule::Eigen, false).is_err());
    assert!(mopt_rs::kinetics::solve::Spectral
        .solve(&ev, &[1.0, 0.0, 0.0], &[1.0], &mut Rng::from_seed(1))
        .is_err());
}

/// LJ7 near its lowest minimum (Bohr), as in `cli_mapper.rs`.
const AR7: [[f64; 3]; 7] = [
    [6.023853759546511, 0.0, -0.15549746403767273],
    [1.962006769746685, 5.809518090796886, -0.061011157764781325],
    [-4.854674003698997, 3.590479638369873, 0.03347514850811009],
    [-4.999838497421851, -3.5904796383698727, 0.12796145478100152],
    [1.6599768546465556, -5.809518090796886, 0.2224477610538929],
    [-0.08463505395523435, 0.0, 3.794030138169188],
    [0.2933101711363313, 0.0, -3.961405880709738],
];

const MAPPER_CONFIG: &str = r#"{
  "mapper_settings": {"afir_gamma_kJmol": 2.0, "dist_upper_ang": 7.0, "rng_seed": 42},
  "top_n_candidates": 2,
  "step1_settings": {"othersoft": "lj", "NSTEP": 300},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 200},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "calc_exact_hess": 1,
    "NSTEP": 400,
    "intrinsic_reaction_coordinates": ["0.5", "200", "lqa"]
  }
}"#;

#[test]
fn multioptpy_s_rcmc_settings_of_the_mapper_become_the_kinetic_priority() {
    use mopt_rs::cli::mapper::MapperConfig;
    use mopt_rs::cli::option::parser;
    use mopt_rs::cli::spec::RUN_MAPPER;
    let dir = scratch("mapper_settings");
    let input = dir.join("ar7.xyz");
    std::fs::write(&input, "2\n0 1\nAr 0 0 0\nAr 3.8 0 0\n").unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        r#"{"mapper_settings": {"use_rcmc": true, "rcmc_temperature_K": 350,
            "rcmc_reaction_time_s": 1e-6, "rcmc_start_node_id": 0, "kinetics_solver": "rcmc"},
            "step1_settings": {"othersoft": "lj"}}"#,
    )
    .unwrap();
    let parse = |extra: &[&str]| {
        let mut args = vec![
            input.display().to_string(),
            "-cfg".to_string(),
            cfg.display().to_string(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
        MapperConfig::from_args(&parsed, None).unwrap().settings
    };
    let s = parse(&[]);
    assert_eq!(s.priority, "kinetics");
    let o = &s.priority_options;
    assert_eq!(o.get("solver").map(String::as_str), Some("rcmc"));
    assert_eq!(o.get("temperature").map(String::as_str), Some("350"));
    assert_eq!(o.get("start").map(String::as_str), Some("0"));
    assert_eq!(o.get("time").unwrap().parse::<f64>().unwrap(), 1e-6);
    // The command line goes over the file.
    let s = parse(&[
        "--rcmc_time",
        "2e-3",
        "-map_opt",
        "kinetics_score=residence",
    ]);
    assert_eq!(
        s.priority_options.get("time").map(String::as_str),
        Some("2e-3")
    );
    assert_eq!(
        s.priority_options.get("score").map(String::as_str),
        Some("residence")
    );
}

#[test]
fn the_mapper_explores_with_the_kinetic_priority() {
    let dir = scratch("mapper");
    let input = dir.join("ar7.xyz");
    let mut text = String::from("7\n0 1\n");
    for p in AR7 {
        text.push_str(&format!(
            "Ar {:.10} {:.10} {:.10}\n",
            p[0] * Units::BOHR_TO_ANGSTROM,
            p[1] * Units::BOHR_TO_ANGSTROM,
            p[2] * Units::BOHR_TO_ANGSTROM
        ));
    }
    std::fs::write(&input, text).unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, MAPPER_CONFIG).unwrap();
    let run = dir.join("run");
    let args: Vec<String> = [
        input.display().to_string(),
        "-cfg".into(),
        cfg.display().to_string(),
        "--max_iter".into(),
        "3".into(),
        "--use_rcmc".into(),
        "--rcmc_time".into(),
        "1e-9".into(),
        "--rcmc_temperature".into(),
        "40".into(),
        "--output_dir".into(),
        run.display().to_string(),
    ]
    .into();
    assert_eq!(mopt_rs::cli::run_command("run_mapper", &args), 0);
    let log = std::fs::read_to_string(run.join("mopt.log")).unwrap();
    assert!(log.contains("priority: kinetics"), "{log}");
    // The network it wrote, by run_kinetics from the run directory.
    let out = dir.join("kinetics");
    let args: Vec<String> = [
        run.display().to_string(),
        "-T".into(),
        "40".into(),
        "--time-grid".into(),
        "1e-13".into(),
        "1e-6".into(),
        "8".into(),
        "--output-dir".into(),
        out.display().to_string(),
        "--validate".into(),
        "--lump-conformers".into(),
    ]
    .into();
    assert_eq!(mopt_rs::cli::run_command("run_kinetics", &args), 0);
    let csv = std::fs::read_to_string(out.join("populations.csv")).unwrap();
    assert!(
        csv.starts_with("time_s,EQ") && csv.lines().count() == 9,
        "{csv}"
    );
    assert!(out.join("species.csv").is_file());
    let diag = std::fs::read_to_string(out.join("diagnostics.json")).unwrap();
    assert!(!diag.contains("\"passed\": false"), "{diag}");
}

#[test]
fn run_kinetics_writes_its_files_and_refuses_what_it_does_not_have() {
    let dir = scratch("cli");
    let file = dir.join("network.json");
    let doc = golden();
    let case = doc
        .get("cases")
        .and_then(JsonValue::as_array)
        .unwrap()
        .iter()
        .find(|c| c.get("name").and_then(JsonValue::as_str) == Some("random7_t1"))
        .unwrap()
        .clone();
    std::fs::write(&file, case.get("network").unwrap().to_string_compact()).unwrap();
    let out = dir.join("out");
    let args = |extra: &[&str]| -> Vec<String> {
        let mut a = vec![
            file.display().to_string(),
            "--output-dir".to_string(),
            out.display().to_string(),
        ];
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    };
    let code = mopt_rs::cli::run_command(
        "run_kinetics",
        &args(&[
            "--time-grid",
            "1e-12",
            "1e2",
            "8",
            "--validate",
            "--sensitivity",
            "--priority",
            "residence",
            "--spectrum",
            "--uncertainty",
            "0.5",
            "--samples",
            "8",
            "--corrections",
            "wigner",
        ]),
    );
    assert_eq!(code, 0);
    for f in [
        "populations.csv",
        "rates.csv",
        "rcmc_K_contracted.csv",
        "rcmc_superstates.json",
        "diagnostics.json",
        "populations.svg",
        "sensitivity.csv",
        "uncertainty.csv",
        "priority.csv",
        "spectrum.csv",
        "settings.json",
        "mopt.log",
    ] {
        assert!(out.join(f).is_file(), "{f}");
    }
    let diag = std::fs::read_to_string(out.join("diagnostics.json")).unwrap();
    assert!(diag.contains("\"passed\": true") && !diag.contains("\"passed\": false"));
    // Every solver runs from the command line.
    for solver in [
        "rcmc",
        "legacy_rcmc",
        "expm",
        "spectral",
        "steady",
        "propagate",
    ] {
        assert_eq!(
            mopt_rs::cli::run_command("run_kinetics", &args(&["--solver", solver, "-t", "1e-3"])),
            0,
            "{solver}"
        );
    }
    // Bell's correction with the truncated parabola, and with the rosenbrock solver.
    assert_eq!(
        mopt_rs::cli::run_command(
            "run_kinetics",
            &args(&[
                "--corrections",
                "bell",
                "--bell-truncation",
                "on",
                "-t",
                "1e-3"
            ])
        ),
        0
    );
    assert_eq!(
        mopt_rs::cli::run_command(
            "run_kinetics",
            &args(&["--solver", "rosenbrock", "-t", "1e-3"])
        ),
        0
    );
    // Refused by name (exit code 2).
    for extra in [
        &["--bell-truncation", "nope"][..],
        &["--solver", "rcmc_b"],
        &["--corrections", "marcus"],
        &["--time-rule", "nope"],
    ] {
        assert_eq!(
            mopt_rs::cli::run_command("run_kinetics", &args(extra)),
            2,
            "{extra:?}"
        );
    }
    // A missing network file is an error.
    let missing = [
        dir.join("none.json").display().to_string(),
        "--output-dir".to_string(),
        dir.join("none_out").display().to_string(),
    ];
    assert_ne!(mopt_rs::cli::run_command("run_kinetics", &missing), 0);
}

// ---------------------------------------------------------------------------------------------
// Species and bimolecular steps: pseudo-first order, the standard state, mass action.

/// `A + B <-> C` (A free of B on side 0) with the energies of A, C, the saddle point, and B.
fn binding_network(e_a: f64, e_c: f64, e_ts: f64, e_b: f64) -> KineticNetwork {
    use mopt_rs::kinetics::{Partner, Species};
    let mut net = network(&[e_a, e_c], &[(0, 1, e_ts)]);
    net.species.push(Species::new("B", Some(e_b), None));
    net.bottlenecks[0].partner = Some(Partner {
        species: 0,
        free_side: 0,
    });
    net
}

/// `A + B <-> C` with `k2` (L/mol/s at c° = 1 mol/L) and `k_r` (1/s) at `t`.
fn binding_with(k2: f64, kr: f64, t: f64) -> KineticNetwork {
    let e_ts = barrier(k2, t);
    binding_network(0.0, e_ts - barrier(kr, t), e_ts, 0.0)
}

fn species_settings(model: &str, t: f64, conc: &[(&str, f64)]) -> KineticsSettings {
    use mopt_rs::kinetics::SpeciesSettings;
    KineticsSettings {
        conditions: conditions(t, Clamp::Consistent),
        model: model.to_string(),
        solver: if model == "mass_action" {
            "rosenbrock".to_string()
        } else {
            "expm".to_string()
        },
        species: SpeciesSettings {
            concentrations: conc.iter().map(|(n, c)| (n.to_string(), *c)).collect(),
            ..SpeciesSettings::default()
        },
        solver_settings: SolverSettings {
            ode_tolerance: 1e-9,
            ..SolverSettings::default()
        },
        ..KineticsSettings::default()
    }
}

/// The solution of `net` under `s` from the first state.
fn run_species(net: &KineticNetwork, s: &KineticsSettings, times: &[f64]) -> Vec<Vec<f64>> {
    use mopt_rs::kinetics::{prepare, KineticsParts};
    let parts = KineticsParts::default();
    let p = prepare(net, s, &parts).unwrap();
    let mut solver = parts.solvers.create(&s.solver, &s.solver_settings).unwrap();
    let mut rng = Rng::from_seed(1);
    let p0 = delta(p.rates.states.len(), 0);
    solver
        .solve(&p.evolution, &p0, times, &mut rng)
        .unwrap()
        .populations
}

/// The analytic `[C](t)` of `A + B <-> C` from `[C](0) = 0`.
fn binding_analytic(kf: f64, kr: f64, a0: f64, b0: f64, t: f64) -> f64 {
    let s = a0 + b0 + kr / kf;
    let disc = (s * s - 4.0 * a0 * b0).sqrt();
    let (r1, r2) = ((s - disc) / 2.0, (s + disc) / 2.0);
    let e = (-kf * (r2 - r1) * t).exp();
    r1 * (1.0 - e) / (1.0 - (r1 / r2) * e)
}

#[test]
fn pseudo_first_order_steps_scale_with_the_concentration_and_keep_detailed_balance() {
    use mopt_rs::kinetics::{prepare, KineticsParts};
    let t = 300.0;
    let (k2, kr, cb) = (1e3, 0.1, 2e-3);
    let net = binding_with(k2, kr, t);
    let s = species_settings("canonical", t, &[("B", cb)]);
    let p = prepare(&net, &s, &KineticsParts::default()).unwrap();
    let g = p.rates.generator().unwrap();
    // k' = k2 [B] / c° forward, k_r back; detailed balance with mu_B = G°_B + RT ln([B] / c°).
    assert!((g.ln_rate(1, 0).exp() / (k2 * cb) - 1.0).abs() < 1e-12);
    assert!((g.ln_rate(0, 1).exp() / kr - 1.0).abs() < 1e-12);
    assert!(p.rates.detailed_balance && g.detailed_balance_residual().unwrap() < 1e-12);
    // The equilibrium: [C] / [A] = K [B] with K = k2 / k_r.
    let fin = run_species(
        &net,
        &species_settings("canonical", t, &[("B", cb)]),
        &[1e4],
    );
    assert!((fin[0][1] / fin[0][0] / (k2 * cb / kr) - 1.0).abs() < 1e-9);
    // A step without a concentration of its species, or a concentration of an unknown
    // species, is refused.
    assert!(prepare(
        &net,
        &species_settings("canonical", t, &[]),
        &KineticsParts::default()
    )
    .is_err());
    assert!(prepare(
        &net,
        &species_settings("canonical", t, &[("B", cb), ("Q", 1.0)]),
        &KineticsParts::default()
    )
    .is_err());
}

#[test]
fn mass_action_follows_the_analytic_binding_curve_and_conserves_mass() {
    use mopt_rs::kinetics::{prepare, KineticsParts, ModelSettings};
    let t = 300.0;
    let (k2, kr) = (2e3, 0.5);
    let (a0, b0) = (1e-3, 3e-3);
    let net = binding_with(k2, kr, t);
    let mut s = species_settings("mass_action", t, &[("B", b0)]);
    s.model_settings = ModelSettings {
        total_concentration: Some(a0),
        ..ModelSettings::default()
    };
    let times = [1e-3, 0.01, 0.1, 1.0, 10.0, 100.0];
    let parts = KineticsParts::default();
    let p = prepare(&net, &s, &parts).unwrap();
    let mut solver = parts
        .solvers
        .create("rosenbrock", &s.solver_settings)
        .unwrap();
    let mut rng = Rng::from_seed(1);
    let sol = solver
        .solve(&p.evolution, &[1.0, 0.0], &times, &mut rng)
        .unwrap();
    let mut worst: f64 = 0.0;
    for (k, &time) in times.iter().enumerate() {
        let y = &sol.populations[k];
        let c = binding_analytic(k2, kr, a0, b0, time);
        worst = worst.max((y[1] - c).abs() / a0);
        assert!((y[1] - c).abs() < 1e-6 * a0, "{time}: {} {c}", y[1]);
        // A + C and B + C are conserved.
        assert!(
            sol.diagnostics[k].conservation < 1e-12,
            "{}",
            sol.diagnostics[k].conservation
        );
        assert!((y[0] + y[1] - a0).abs() < 1e-15 && (y[2] + y[1] - b0).abs() < 1e-15);
    }
    eprintln!("mass action against the analytic curve: largest error {worst:.1e} of [A]0");
    // The other solvers refuse the non-linear model; the derivatives too.
    assert!(parts
        .solvers
        .create("expm", &s.solver_settings)
        .unwrap()
        .solve(&p.evolution, &[1.0, 0.0], &[1.0], &mut rng)
        .is_err());
    assert!(
        mopt_rs::kinetics::derivative::derivatives(&net, &s, &parts, &[(0, 1.0)], &[1.0]).is_err()
    );
    // Without the total concentration it is refused.
    let s0 = species_settings("mass_action", t, &[("B", b0)]);
    assert!(prepare(&net, &s0, &parts).is_err());
}

#[test]
fn pseudo_first_order_agrees_with_mass_action_in_excess() {
    use mopt_rs::kinetics::ModelSettings;
    let t = 300.0;
    let (k2, kr) = (1e4, 2.0);
    let (a0, b0) = (1e-8, 1e-2);
    let net = binding_with(k2, kr, t);
    let times = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0];
    let pseudo = run_species(
        &net,
        &species_settings("canonical", t, &[("B", b0)]),
        &times,
    );
    let mut s = species_settings("mass_action", t, &[("B", b0)]);
    s.model_settings = ModelSettings {
        total_concentration: Some(a0),
        ..ModelSettings::default()
    };
    let mass = run_species(&net, &s, &times);
    let mut worst: f64 = 0.0;
    for k in 0..times.len() {
        // The analytic pseudo-first-order curve as well.
        let kp = k2 * b0;
        let want = kp / (kp + kr) * (1.0 - (-(kp + kr) * times[k]).exp());
        assert!(
            (pseudo[k][1] - want).abs() < 1e-10,
            "{} {want}",
            pseudo[k][1]
        );
        let d = (mass[k][1] / a0 - pseudo[k][1]).abs();
        worst = worst.max(d);
        assert!(
            d < 1e-5,
            "{}: {} {}",
            times[k],
            mass[k][1] / a0,
            pseudo[k][1]
        );
    }
    eprintln!("pseudo-first order against mass action at [B]/[A] = 1e6: {worst:.1e}");
}

#[test]
fn apparent_reaction_orders_match_finite_differences() {
    use mopt_rs::kinetics::derivative::{derivatives, Parameter};
    use mopt_rs::kinetics::KineticsParts;
    let t = 300.0;
    let (k2, kr, cb) = (1e3, 0.1, 2e-3);
    let net = binding_with(k2, kr, t);
    let times = [1e-6, 0.1, 2.0, 1e4];
    let parts = KineticsParts::default();
    for solver in ["expm", "propagate"] {
        let mut s = species_settings("canonical", t, &[("B", cb)]);
        s.solver = solver.to_string();
        let d = derivatives(&net, &s, &parts, &[(0, 1.0)], &times).unwrap();
        let m = d.index_of(Parameter::LnConcentration(0)).unwrap();
        let h = 1e-4;
        let at = |x: f64| {
            run_species(
                &net,
                &species_settings("canonical", t, &[("B", cb * x.exp())]),
                &times,
            )
        };
        let (up, down) = (at(h), at(-h));
        for k in 0..times.len() {
            for i in 0..2 {
                let fd = (up[k][i] - down[k][i]) / (2.0 * h);
                let ad = d.d_populations[m][k][i];
                assert!((ad - fd).abs() < 1e-7, "{solver} {k} {i}: {ad} {fd}");
            }
        }
        // The order of the yield of C: 1 at the start, 1 - p_C = p_A at equilibrium.
        let order = |k: usize| d.d_populations[m][k][1] / d.populations[k][1];
        assert!((order(0) - 1.0).abs() < 1e-3, "{}", order(0));
        let last = times.len() - 1;
        assert!((order(last) - d.populations[last][0]).abs() < 1e-8);
        // dp/dG°_B = (1 / kT) dp / d ln[B].
        let g = d.index_of(Parameter::SpeciesEnergy(0)).unwrap();
        assert!((d.d_populations[g][1][1] - d.d_populations[m][1][1] / rt(t)).abs() < 1e-6);
    }
}

#[test]
fn the_standard_state_and_the_association_treatments() {
    use mopt_rs::kinetics::species::{trans_rot_entropy, Association};
    use mopt_rs::kinetics::{prepare, KineticsParts};
    let t = 298.15;
    let parts = KineticsParts::default();
    // Free energies of the gas at 1 atm: the association is faster by c° R T / p = 24.47 at
    // 1 mol/L than with the same numbers taken as electronic energies.
    let mut net = binding_with(1e3, 0.1, t);
    for s in &mut net.states {
        s.free_energy = Some(s.energy);
    }
    net.bottlenecks[0].free_energy = Some(net.bottlenecks[0].energy);
    net.species[0].data.free_energy = Some(0.0);
    let mut s = species_settings("canonical", t, &[("B", 1.0)]);
    let electronic = prepare(&net, &s, &parts).unwrap().rates;
    s.conditions.energy = EnergyKind::Free;
    let free = prepare(&net, &s, &parts).unwrap().rates;
    let ratio = (free.steps[0].ln_k[0] - electronic.steps[0].ln_k[0]).exp();
    let want = 1000.0 * Units::GAS_CONSTANT_J_MOL_K * t / 101_325.0;
    assert!((ratio / want - 1.0).abs() < 1e-10, "{ratio} {want}");
    assert!((want - 24.46).abs() < 0.01);
    assert!((free.steps[0].ln_k[1] - electronic.steps[0].ln_k[1]).abs() < 1e-12);
    // -conc: the free energies are at 1 mol/L already; no conversion.
    s.conditions.concentration = Some(1.0);
    let solution = prepare(&net, &s, &parts).unwrap().rates;
    assert!((solution.steps[0].ln_k[0] - electronic.steps[0].ln_k[0]).abs() < 1e-12);
    s.conditions.concentration = None;
    // The treatments multiply the association (and its equilibrium constant); the step keeps
    // balancing its reverse.
    for (text, factor) in [
        ("factor:3", 3.0),
        ("effective:5", 5.0),
        (
            "free_volume:h2o",
            1.0 / Association::parse("free_volume:h2o")
                .unwrap()
                .free_fraction()
                .unwrap(),
        ),
    ] {
        let mut x = s.clone();
        x.species.association = Association::parse(text).unwrap();
        let r = prepare(&net, &x, &parts).unwrap().rates;
        let f = (r.steps[0].ln_k[0] - free.steps[0].ln_k[0]).exp();
        assert!((f / factor - 1.0).abs() < 1e-10, "{text}: {f} {factor}");
        assert!((r.steps[0].ln_k[1] - free.steps[0].ln_k[1]).abs() < 1e-12);
        assert!(r.generator().unwrap().detailed_balance_residual().unwrap() < 1e-12);
        assert_eq!(
            r.species.treatments[0].name(),
            text.replace("h2o", "55.6:27.944")
        );
    }
    // Wertz-type: every structure loses the fraction F of S_trans + S_rot; the association
    // changes by -F (S_ts - S_A - S_B) / k_B in ln k.
    use mopt_rs::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge};
    let argon = |xs: &[[f64; 3]]| {
        MolecularSystem::new(
            vec![Element::from_atomic_number(18).unwrap(); xs.len()],
            &positions::<Angstrom>(xs),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    };
    let mut w = net.clone();
    w.states[0].system = Some(argon(&[[0.0, 0.0, 0.0]]));
    w.states[1].system = Some(argon(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]));
    w.bottlenecks[0].system = Some(argon(&[[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]]));
    w.species[0].data.system = Some(argon(&[[0.0, 0.0, 0.0]]));
    let mut x = s.clone();
    x.species.association = Association::Wertz { fraction: 0.5 };
    let r = prepare(&w, &x, &parts).unwrap().rates;
    let c = x.conditions;
    let sym = mopt_rs::kinetics::rate::tst::SymmetryChoice::Auto;
    let entropy = |sys: &MolecularSystem| trans_rot_entropy(sys, None, sym, &c).unwrap();
    let ds = entropy(w.bottlenecks[0].system.as_ref().unwrap())
        - 2.0 * entropy(w.states[0].system.as_ref().unwrap());
    let kb = Units::BOLTZMANN_HARTREE_PER_K;
    let got = r.steps[0].ln_k[0] - free.steps[0].ln_k[0];
    assert!(
        (got - (-0.5 * ds / kb)).abs() < 1e-9,
        "{got} {}",
        -0.5 * ds / kb
    );
    assert!(got > 0.0);
    assert!(r.generator().unwrap().detailed_balance_residual().unwrap() < 1e-12);
    // Wertz needs every structure.
    assert!(prepare(&net, &x, &parts).is_err());
}

#[test]
fn a_catalytic_cycle_has_directed_rates_with_its_affinity() {
    use mopt_rs::kinetics::derivative::{derivatives, Parameter};
    use mopt_rs::kinetics::{prepare, KineticsParts, Partner, Species};
    let t = 300.0;
    let r = rt(t);
    // E + S -> ES -> EP -> E + P.
    let mut net = network(
        &[0.0, -r, -2.0 * r],
        &[
            (0, 1, barrier(1e6, t)),
            (1, 2, -r + barrier(1e3, t)),
            (2, 0, -2.0 * r + barrier(10.0, t)),
        ],
    );
    net.species.push(Species::new("S", Some(0.0), None));
    net.species.push(Species::new("P", Some(-3.0 * r), None));
    net.bottlenecks[0].partner = Some(Partner {
        species: 0,
        free_side: 0,
    });
    net.bottlenecks[2].partner = Some(Partner {
        species: 1,
        free_side: 1,
    });
    let (cs, cp) = (1e-2, 1e-4);
    let s = species_settings("canonical", t, &[("S", cs), ("P", cp)]);
    let parts = KineticsParts::default();
    let rates = prepare(&net, &s, &parts).unwrap().rates;
    assert!(!rates.detailed_balance);
    assert!(rates.notes.iter().any(|n| n.contains("catalytic cycle")));
    // The product of the rate constants around the cycle over that against it is
    // exp((mu_S - mu_P) / kT) = ([S] / [P]) exp((G_S - G_P) / kT).
    let st = &rates.steps;
    let forward = st[0].ln_k[0] + st[1].ln_k[0] + st[2].ln_k[0];
    let backward = st[0].ln_k[1] + st[1].ln_k[1] + st[2].ln_k[1];
    let want = (cs / cp).ln() + 3.0;
    assert!(
        (forward - backward - want).abs() < 1e-10,
        "{} {want}",
        forward - backward
    );
    // The derivatives work without a potential (against central differences).
    let times = [1e-3, 1.0, 100.0];
    let d = derivatives(&net, &s, &parts, &[(0, 1.0)], &times).unwrap();
    let m = d.index_of(Parameter::LnConcentration(0)).unwrap();
    let h = 1e-4;
    let at = |x: f64| {
        run_species(
            &net,
            &species_settings("canonical", t, &[("S", cs * x.exp()), ("P", cp)]),
            &times,
        )
    };
    let (up, down) = (at(h), at(-h));
    for k in 0..times.len() {
        for i in 0..3 {
            let fd = (up[k][i] - down[k][i]) / (2.0 * h);
            assert!((d.d_populations[m][k][i] - fd).abs() < 1e-7, "{k} {i}");
        }
    }
}

#[test]
fn run_kinetics_with_species_from_the_command_line() {
    let dir = scratch("species_cli");
    let t = 300.0;
    let net = binding_with(1e3, 0.1, t);
    let doc = format!(
        "{{\"nodes\": [{{\"node_id\": 0, \"energy_hartree\": {}}}, {{\"node_id\": 1, \
         \"energy_hartree\": {}}}], \"edges\": [{{\"edge_id\": 0, \"node_id_1\": 0, \
         \"node_id_2\": 1, \"ts_energy_hartree\": {}}}]}}",
        net.states[0].energy, net.states[1].energy, net.bottlenecks[0].energy
    );
    let file = dir.join("network.json");
    std::fs::write(&file, doc).unwrap();
    let out = dir.join("out");
    let base = |extra: &[&str]| -> Vec<String> {
        let mut a = vec![
            file.display().to_string(),
            "--output-dir".to_string(),
            out.display().to_string(),
            "--species".to_string(),
            "B:0.0".to_string(),
            "--partner".to_string(),
            "TS0:B:EQ0".to_string(),
        ];
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    };
    let code = mopt_rs::cli::run_command(
        "run_kinetics",
        &base(&[
            "--concentration",
            "B:2mM",
            "--time-grid",
            "1e-3",
            "1e3",
            "5",
            "--sensitivity",
        ]),
    );
    assert_eq!(code, 0);
    let rates = std::fs::read_to_string(out.join("rates.csv")).unwrap();
    assert!(rates.contains(",B,EQ0,standard,"), "{rates}");
    let sens = std::fs::read_to_string(out.join("sensitivity.csv")).unwrap();
    assert!(sens.contains("ln_concentration,B,") && sens.contains("species_energy,B,"));
    let diag = std::fs::read_to_string(out.join("diagnostics.json")).unwrap();
    assert!(diag.contains("bimolecular_steps") && diag.contains("k2_L_mol-1_s-1"));
    // Mass action from the command line: concentrations.csv and yields.csv.
    let code = mopt_rs::cli::run_command(
        "run_kinetics",
        &base(&[
            "--concentration",
            "B:3e-3",
            "--model",
            "mass_action",
            "--total-concentration",
            "1e-3",
            "--time-grid",
            "1e-3",
            "1e3",
            "5",
            "--association",
            "factor:2",
        ]),
    );
    assert_eq!(code, 0);
    let conc = std::fs::read_to_string(out.join("concentrations.csv")).unwrap();
    assert!(conc.starts_with("time_s,EQ0,EQ1,B"), "{conc}");
    let yields = std::fs::read_to_string(out.join("yields.csv")).unwrap();
    assert!(yields.contains("EQ1,state,") && yields.contains("B,species,"));
    // Refused: a linear solver with mass action, a concentration of an unknown species, a
    // bimolecular edge without a concentration, a bad unit.
    for extra in [
        &[
            "--model",
            "mass_action",
            "--total-concentration",
            "1e-3",
            "--solver",
            "expm",
        ][..],
        &["--concentration", "B:1", "--concentration", "X:1"],
        &[],
        &["--concentration", "B:1furlong"],
    ] {
        assert_ne!(
            mopt_rs::cli::run_command("run_kinetics", &base(extra)),
            0,
            "{extra:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Rate theories of single edges: variational TST, electron transfer, crossings, in one network.

/// Five states joined by a saddle point, a saddle point with a reaction path, a non-adiabatic
/// electron transfer, a crossing point, and an adiabatic electron transfer.
fn mixed_network(t: f64) -> KineticNetwork {
    use mopt_rs::kinetics::network::{BottleneckKind, CvtPath};
    use mopt_rs::kinetics::rate::electron_transfer::{TransferForm, TransferStep};
    let r = rt(t);
    let mut net = network(
        &[0.0, r, -0.5 * r, 0.3 * r, 0.8 * r],
        &[
            (0, 1, barrier(1e6, t)),
            (1, 2, r + barrier(1e4, t)),
            (2, 3, 5.0 * r),
            (3, 4, 0.8 * r + barrier(1e2, t)),
            (0, 4, 0.0),
        ],
    );
    net.bottlenecks[0].imaginary_wavenumber = Some(-1200.0);
    net.bottlenecks[1].imaginary_wavenumber = Some(-800.0);
    let s: Vec<f64> = (0..11).map(|k| -0.5 + 0.1 * k as f64).collect();
    net.bottlenecks[1].path = Some(CvtPath {
        arc_length: s.clone(),
        energies: s.iter().map(|x| -0.002 * x * x).collect(),
        frequencies: Some(s.iter().map(|x| vec![1500.0 + 400.0 * x, 900.0]).collect()),
        ..CvtPath::default()
    });
    net.bottlenecks[2].transfer = Some(TransferStep {
        reorganization: 20.0 * r,
        form: TransferForm::Nonadiabatic,
        delta_g0: None,
        wavenumber: None,
    });
    net.bottlenecks[2].coupling = Some(1e-4);
    net.bottlenecks[3].kind = BottleneckKind::Crossing;
    net.bottlenecks[3].coupling = Some(2e-5);
    net.bottlenecks[3].gradient_difference = Some(0.02);
    net.bottlenecks[3].reduced_mass = Some(10.0);
    net.bottlenecks[4].transfer = Some(TransferStep {
        reorganization: 30.0 * r,
        form: TransferForm::Adiabatic,
        delta_g0: None,
        wavenumber: None,
    });
    net.bottlenecks[4].coupling = Some(1e-3);
    net
}

#[test]
fn a_crossing_from_the_files_of_diabatize_takes_zhu_nakamura() {
    use mopt_rs::kinetics::rate::{
        landau_zener_ln_average, landau_zener_ln_average_sloped, PeakedCrossing,
        ZhuNakamuraCrossing,
    };
    use mopt_rs::kinetics::{prepare, KineticsParts};
    use mopt_rs::workflow::diabatic_crossings::{self, slopes_of, CrossingRow};
    use mopt_rs::workflow::kinetics::apply_annotations_in;
    let t = 300.0;
    let dir = scratch("zhu_nakamura");
    let row = |kind: &str, slopes: [f64; 2]| CrossingRow {
        kind: kind.into(),
        states: [1, 2],
        position: 0.4,
        energy: -1.0,
        coupling: 3e-4,
        slopes,
    };
    diabatic_crossings::write(
        &dir.join("diabatic_crossings.csv"),
        &[
            row("zhu_nakamura", slopes_of(0.01, 0.02)),
            row("crossing", [0.012, -0.008]),
        ],
    )
    .unwrap();
    let annotated = |extra: &str| {
        let mut net = mixed_network(t);
        let doc = JsonValue::parse(&format!(
            r#"{{"edges": {{"TS3": {{"diabatic_crossings": "diabatic_crossings.csv"{extra}}}}}}}"#
        ))
        .unwrap();
        apply_annotations_in(&mut net, &doc, &dir).map(|_| net)
    };
    // Two rows: one must be chosen; a slope of the edge itself and the file together are
    // refused.
    assert!(annotated("").is_err());
    assert!(annotated(r#", "diabatic_crossing": 3"#).is_err());
    assert!(annotated(r#", "diabatic_crossing": 1, "slope_product": 1e-4"#).is_err());
    let net = annotated(r#", "diabatic_crossing": 1, "coupling_hartree": 5e-4"#).unwrap();
    let b = &net.bottlenecks[3];
    // The coupling of the edge is kept; the slopes and the mass come from the file.
    assert_eq!(b.coupling, Some(5e-4));
    assert!((b.slope_product.unwrap() / 1e-4 - 1.0).abs() < 1e-12);
    assert!((b.gradient_difference.unwrap() / 0.02 - 1.0).abs() < 1e-12);
    assert_eq!(b.reduced_mass, Some(1.0));

    let kt = rt(t);
    let factor = |net: &KineticNetwork, name: &str| {
        let s = KineticsSettings {
            conditions: conditions(t, Clamp::Consistent),
            corrections: vec![name.into()],
            solver: "expm".into(),
            ..KineticsSettings::default()
        };
        prepare(net, &s, &KineticsParts::default()).map(|p| {
            p.rates.steps[3]
                .flux
                .factors
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, f)| *f)
                .unwrap()
        })
    };
    let zn = ZhuNakamuraCrossing::from_diabatic(5e-4, 0.02, 1e-4, 1.0).unwrap();
    let got = factor(&net, "zhu_nakamura").unwrap();
    assert!((got - zn.ln_average(kt)).abs() < 1e-9, "{got}");
    assert!(got.exp() > 0.0 && got.exp() <= 0.5);
    // Landau-Zener takes the double passage of a sloped crossing.
    let lz = factor(&net, "landau_zener").unwrap();
    assert!((lz - landau_zener_ln_average_sloped(5e-4, 0.02, 1.0, kt)).abs() < 1e-9);
    assert!((lz - landau_zener_ln_average(5e-4, 0.02, 1.0, kt)).abs() > 1e-3);
    // A peaked crossing: Landau-Zener as before; Zhu-Nakamura of the nonadiabatic-tunnelling
    // type, averaged from the lower of the two barriers (to the higher minimum, 4) up. The
    // edge keeps the coupling it had (2e-5, set on the network), not the one of the file.
    let peaked = annotated(r#", "diabatic_crossing": 2"#).unwrap();
    assert!(peaked.bottlenecks[3].slope_product.unwrap() < 0.0);
    assert_eq!(peaked.bottlenecks[3].coupling, Some(2e-5));
    let lz = factor(&peaked, "landau_zener").unwrap();
    assert!((lz - landau_zener_ln_average(2e-5, 0.02, 1.0, kt)).abs() < 1e-9);
    let zn = PeakedCrossing::from_diabatic(2e-5, 0.02, 0.012 * -0.008, 1.0).unwrap();
    let got = factor(&peaked, "zhu_nakamura").unwrap();
    let want = zn.average(kt, barrier(1e2, t)).ln_kappa;
    assert!((got - want).abs() < 1e-9, "{got} {want}");
    assert!((got - lz).abs() > 1e-3, "{got} {lz}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_complex_and_its_channel_balance_through_the_association() {
    use mopt_rs::kinetics::network::BottleneckKind;
    use mopt_rs::kinetics::rate::encounter::{association_rate, AssociationStep, EncounterPair};
    use mopt_rs::kinetics::{prepare, KineticsParts};
    let t = 298.15;
    let r = rt(t);
    let kcal = 1.0 / Units::HARTREE_TO_KCAL_MOL;
    // A complex 3 kcal/mol below its two fragments.
    let mut net = network(&[0.0, 3.0 * kcal], &[(0, 1, 3.0 * kcal)]);
    net.states[1].fragments = vec![
        KineticState::with_energy(0, "A", 1.0 * kcal),
        KineticState::with_energy(1, "B", 2.0 * kcal),
    ];
    let pair = EncounterPair {
        radii: [5.0, 6.0],
        masses: [18.0, 30.0],
    };
    net.bottlenecks[0].kind = BottleneckKind::Barrierless;
    net.bottlenecks[0].association = Some(AssociationStep {
        channel_side: 1,
        pair,
    });
    let rates = |viscosity: Option<f64>| {
        let s = KineticsSettings {
            conditions: Conditions {
                viscosity,
                ..conditions(t, Clamp::Consistent)
            },
            solver: "expm".into(),
            ..KineticsSettings::default()
        };
        prepare(&net, &s, &KineticsParts::default()).unwrap().rates
    };
    for viscosity in [None, Some(8.9e-4)] {
        let set = rates(viscosity);
        let step = &set.steps[0];
        assert_eq!(
            step.rate,
            if viscosity.is_some() {
                "collins_kimball"
            } else {
                "capture"
            }
        );
        // From the channel: the association at 1 mol/L; from the complex: detailed balance.
        let k_on = association_rate(&pair, t, viscosity);
        assert!((step.ln_k[1] - k_on.ln()).abs() < 1e-9, "{:?}", step.ln_k);
        assert!((step.ln_k[0] - (k_on.ln() - 3.0 * kcal / r)).abs() < 1e-9);
        assert!(set.detailed_balance);
        assert!(!step.clamped);
    }
    // Diffusion slows the association in water.
    assert!(rates(Some(8.9e-4)).steps[0].ln_k[1] < rates(None).steps[0].ln_k[1]);
}

#[test]
fn saddle_points_paths_electron_transfers_and_crossings_mix_in_one_network() {
    use mopt_rs::kinetics::derivative::{derivatives, Parameter};
    use mopt_rs::kinetics::{prepare, KineticsParts};
    let t = 300.0;
    let r = rt(t);
    let net = mixed_network(t);
    let s = KineticsSettings {
        conditions: conditions(t, Clamp::Consistent),
        corrections: vec!["bell".into(), "landau_zener".into(), "cvt".into()],
        solver: "expm".into(),
        ..KineticsSettings::default()
    };
    let parts = KineticsParts::default();
    let p = prepare(&net, &s, &parts).unwrap();
    let rates = &p.rates;
    assert!(rates.detailed_balance);
    assert!(
        rates
            .generator()
            .unwrap()
            .detailed_balance_residual()
            .unwrap()
            < 1e-12
    );
    let names: Vec<&str> = rates.steps.iter().map(|x| x.rate.as_str()).collect();
    assert_eq!(
        names,
        [
            "eyring",
            "eyring",
            "marcus_nonadiabatic",
            "eyring",
            "marcus_adiabatic"
        ]
    );
    let factor = |k: usize, name: &str| {
        rates.steps[k]
            .flux
            .factors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, f)| *f)
    };
    assert!(factor(0, "bell").unwrap() > 0.0 && factor(1, "bell").unwrap() > 0.0);
    assert!(factor(3, "landau_zener").is_some() && factor(2, "bell").is_none());
    // cvt lowers the step with the path and leaves the others at TST, which it says.
    assert!(factor(1, "cvt").unwrap() < 0.0 && factor(0, "cvt") == Some(0.0));
    assert!(rates.steps[0]
        .flux
        .diagnostics
        .iter()
        .any(|d| d.name == "cvt_path" && d.value == 0.0));
    // Every step balances its reverse.
    for st in &rates.steps {
        let [a, b] = st.states;
        let want = -(rates.state_energies[b] - rates.state_energies[a]) / rates.kt;
        assert!(
            ((st.ln_k[0] - st.ln_k[1]) - want).abs() < 1e-9,
            "{}",
            st.rate
        );
    }
    // One solver for all: propagate agrees with the matrix exponential.
    let times = [1e-6, 1e-3, 1.0];
    let mut rng = Rng::from_seed(1);
    let exact = Expm
        .solve(&p.evolution, &delta(5, 0), &times, &mut rng)
        .unwrap();
    let mut prop = Propagate::new(1e-8, Closure::A).unwrap();
    let approx = prop
        .solve(&p.evolution, &delta(5, 0), &times, &mut rng)
        .unwrap();
    for k in 0..times.len() {
        for i in 0..5 {
            assert!((exact.populations[k][i] - approx.populations[k][i]).abs() < 1e-8);
        }
    }
    // The derivatives through the parabolas of the electron transfers and the path of cvt,
    // against central differences.
    let d = derivatives(&net, &s, &parts, &[(0, 1.0)], &times).unwrap();
    let pops = |n: &KineticNetwork, temperature: f64| -> Vec<Vec<f64>> {
        let mut x = s.clone();
        x.conditions.temperature = temperature;
        let p = prepare(n, &x, &parts).unwrap();
        let mut rng = Rng::from_seed(1);
        Expm.solve(&p.evolution, &delta(5, 0), &times, &mut rng)
            .unwrap()
            .populations
    };
    let mut worst: f64 = 0.0;
    for j in 0..5 {
        let m = d.index_of(Parameter::StateEnergy(j)).unwrap();
        let h = 1e-4 * r;
        let mut up = net.clone();
        up.states[j].energy += h;
        let mut down = net.clone();
        down.states[j].energy -= h;
        let (pu, pd) = (pops(&up, t), pops(&down, t));
        for k in 0..times.len() {
            for i in 0..5 {
                let fd = (pu[k][i] - pd[k][i]) / (2.0 * h);
                let err = (d.d_populations[m][k][i] - fd).abs() * r;
                worst = worst.max(err);
                assert!(
                    err < 1e-6,
                    "E{j} t{k} p{i}: {} {fd}",
                    d.d_populations[m][k][i]
                );
            }
        }
    }
    // The bottlenecks with corrections (Bell's barrier and the crossing move with them).
    for b in [0, 1, 3] {
        let m = d.index_of(Parameter::BottleneckEnergy(b)).unwrap();
        let h = 1e-4 * r;
        let mut up = net.clone();
        up.bottlenecks[b].energy += h;
        let mut down = net.clone();
        down.bottlenecks[b].energy -= h;
        let (pu, pd) = (pops(&up, t), pops(&down, t));
        for k in 0..times.len() {
            for i in 0..5 {
                let fd = (pu[k][i] - pd[k][i]) / (2.0 * h);
                let err = (d.d_populations[m][k][i] - fd).abs() * r;
                worst = worst.max(err);
                assert!(
                    err < 1e-6,
                    "TS{b} t{k} p{i}: {} {fd}",
                    d.d_populations[m][k][i]
                );
            }
        }
    }
    // The reference: central differences of the populations with steps of 0.1 K and 0.05 K,
    // combined by Richardson extrapolation (truncation error O(h^4)). A single central
    // difference with a step of 0.01 K is not a reference to 1e-4: the round-off of the
    // computed populations is amplified by its difference quotient (the deviation grows as the
    // step shrinks), so its relative deviation from the derivative scattered between 1e-5 and
    // 1.4e-4 at 299.9, 300 and 300.1 K (about 1e-3 with a step of 1e-3 K), and as much with
    // every correction left out (the path of cvt is not the cause); with 0.1 K alone the
    // truncation error is 1.8e-4 at all three temperatures.
    let m = d.index_of(Parameter::Temperature).unwrap();
    let central = |h: f64| {
        let (pu, pd) = (pops(&net, t + h), pops(&net, t - h));
        (0..times.len())
            .map(|k| {
                (0..5)
                    .map(|i| (pu[k][i] - pd[k][i]) / (2.0 * h))
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<_>>()
    };
    let (coarse, fine) = (central(0.1), central(0.05));
    let mut worst_t: f64 = 0.0;
    for k in 0..times.len() {
        for i in 0..5 {
            let fd = (4.0 * fine[k][i] - coarse[k][i]) / 3.0;
            let deviation = (d.d_populations[m][k][i] - fd).abs() / fd.abs().max(1e-3 / t);
            worst_t = worst_t.max(deviation);
            assert!(
                deviation < 1e-4,
                "T {k} {i}: {} {fd}",
                d.d_populations[m][k][i]
            );
        }
    }
    eprintln!(
        "mixed network: derivatives against central differences, worst {worst:.1e} (x kT), \
         temperature {worst_t:.1e} (relative)"
    );
}

#[test]
fn electron_transfers_and_paths_from_the_annotations() {
    let dir = scratch("edge_theories");
    let t = 298.15;
    let r = rt(t);
    let doc = format!(
        "{{\"nodes\": [{{\"node_id\": 0, \"energy_hartree\": 0.0}}, {{\"node_id\": 1, \
         \"energy_hartree\": {}}}, {{\"node_id\": 2, \"energy_hartree\": {}}}], \"edges\": \
         [{{\"edge_id\": 0, \"node_id_1\": 0, \"node_id_2\": 1, \"ts_energy_hartree\": 0.0}}, \
         {{\"edge_id\": 1, \"node_id_1\": 1, \"node_id_2\": 2, \"ts_energy_hartree\": {}}}]}}",
        -2.0 * r,
        -r,
        barrier(1e5, t)
    );
    let file = dir.join("network.json");
    std::fs::write(&file, doc).unwrap();
    // The profile of freqmain -path for TS1 (its own directory, with its temperature).
    let path_dir = dir.join("path_ts1");
    std::fs::create_dir_all(&path_dir).unwrap();
    let mut csv = String::from(
        "point,arc_length_sqrt_amu_bohr,energy_hartree,adiabatic_energy_hartree,\
         free_energy_hartree\n",
    );
    for k in 0..9 {
        let s = -0.4 + 0.1 * k as f64;
        let e = -0.003 * s * s;
        let g = e + 0.004 + 0.001 * s;
        csv.push_str(&format!("{k},{s},{e},{e},{g}\n"));
    }
    std::fs::write(path_dir.join("free_energy_profile.csv"), csv).unwrap();
    std::fs::write(
        path_dir.join("variational_tst.txt"),
        "Temperature: 298.15 K\n",
    )
    .unwrap();
    let annotations = dir.join("edges.json");
    std::fs::write(
        &annotations,
        "{\"edges\": {\"TS0\": {\"transfer\": \"nonadiabatic\", \"reorganization_kcal_mol\": 20, \
         \"coupling_cm1\": 30}, \"TS1\": {\"cvt_profile_file\": \"path_ts1/free_energy_profile.csv\"}}}",
    )
    .unwrap();
    let out = dir.join("out");
    let code = mopt_rs::cli::run_command(
        "run_kinetics",
        &[
            file.display().to_string(),
            "--annotations".to_string(),
            annotations.display().to_string(),
            "--corrections".to_string(),
            "cvt".to_string(),
            "-T".to_string(),
            "298.15".to_string(),
            "--output-dir".to_string(),
            out.display().to_string(),
        ],
    );
    assert_eq!(code, 0);
    let rates = std::fs::read_to_string(out.join("rates.csv")).unwrap();
    assert!(rates.contains("marcus_nonadiabatic"), "{rates}");
    assert!(rates.contains("cvt_path=1"), "{rates}");
    // The variational shift of that profile: the maximum of g = e + 0.004 + 0.001 s.
    let line = rates.lines().find(|l| l.starts_with("TS1,")).unwrap();
    assert!(line.contains("cvt="), "{line}");
}
