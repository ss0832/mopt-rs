//! Transition states and crossing points together in `run_mapper` (`states=...`), on a model of
//! three atoms with two surfaces whose crossing points are known:
//!
//! `E_s = f_s(r12) + K (r23 - a_s)^2 + K (r13 - C)^2`, with
//! `f_0(r) = A (r - R1)^2 (r - R2)^2` (PES1: minima at R1 and R2, a TS between) and
//! `f_1(r) = KC (r - RC)^2 + D` (PES2: one minimum at RC between them).
//!
//! The surface is PES2 for the triplet (`states=0,1;0,3`, crossing points MECP) or for the first
//! excited state of the singlet (`states=0,1;0,1,1`, MECI). Since both states have the same `K`,
//! the seam `E_0 = E_1` is linear in `r23` at a given `r12`, and its two minima (one between R1
//! and RC, one between RC and R2) follow from a minimization along `r12` here. The network must
//! have the two minima of PES1 and their TS, the minimum of PES2, and the two crossing points,
//! each joined to a minimum of PES1 and that of PES2.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::{Col, Mat};
use mopt_rs::cli::automecp::AutomecpConfig;
use mopt_rs::cli::mapper::{run_with_parts, MapperConfig, MapperParts, MapperRun};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::settings::read_settings;
use mopt_rs::cli::spec::{RUN_AUTOMECP, RUN_MAPPER};
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::Units;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::states::StateCharacters;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::mapper::configuration::Configuration;
use mopt_rs::workflow::mapper::{
    follow_guesses, CrossingKind, ExplorationMethod, Guess, Job, JobContext, JobOutcome,
    MapperObserver, MethodSettings, Network, Node, NodeKind, Source, Task, Told,
};

const R1: f64 = 2.0;
const R2: f64 = 5.5;
const A: f64 = 0.004;
const RC: f64 = 3.5;
const KC: f64 = 0.02;
const D: f64 = 0.004;
const K: f64 = 0.05;
const AS: [f64; 3] = [2.5, 3.0, 2.0];
/// The third surface (`states=0,1;0,3;0,1,2`): one minimum at RC3, D3 above PES1.
const RC3: f64 = 4.6;
const D3: f64 = 0.012;
const C: f64 = 4.0;
const BOHR_TO_ANGSTROM: f64 = Units::BOHR_TO_ANGSTROM;

/// `f_s(r)` and its first two derivatives.
fn curve(s: usize, r: f64) -> (f64, f64, f64) {
    if s == 0 {
        let (p, q) = (r - R1, r - R2);
        (
            A * p * p * q * q,
            2.0 * A * p * q * (p + q),
            2.0 * A * (q * q + 4.0 * p * q + p * p),
        )
    } else if s == 1 {
        let p = r - RC;
        (KC * p * p + D, 2.0 * KC * p, 2.0 * KC)
    } else {
        let p = r - RC3;
        (KC * p * p + D3, 2.0 * KC * p, 2.0 * KC)
    }
}

/// The surface of a multiplicity and a root: the singlet root 0 is PES1; the triplet and the
/// singlet root 1 are the second surface; the singlet root 2 and the triplet root 1 the third.
fn surface(multiplicity: u32, root: usize) -> usize {
    (usize::from(multiplicity == 3) + root).min(2)
}

/// The three-atom model: the surface by the multiplicity (3: PES2) and the root.
struct TwoSurfaces;

impl TwoSurfaces {
    /// A pair term `(e, e', e'')` of the distance between atoms `i` and `j` added to the energy,
    /// the gradient and the Hessian.
    #[allow(clippy::too_many_arguments)]
    fn pair(
        x: &Col<f64>,
        i: usize,
        j: usize,
        term: &dyn Fn(f64) -> (f64, f64, f64),
        e: &mut f64,
        g: &mut Col<f64>,
        h: &mut Mat<f64>,
    ) {
        let d: Vec<f64> = (0..3).map(|k| x[3 * j + k] - x[3 * i + k]).collect();
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let u: Vec<f64> = d.iter().map(|v| v / r).collect();
        let (v, dv, d2v) = term(r);
        *e += v;
        for k in 0..3 {
            g[3 * i + k] -= dv * u[k];
            g[3 * j + k] += dv * u[k];
        }
        for a in 0..3 {
            for b in 0..3 {
                let delta = f64::from(u8::from(a == b));
                let block = d2v * u[a] * u[b] + dv / r * (delta - u[a] * u[b]);
                h[(3 * i + a, 3 * i + b)] += block;
                h[(3 * j + a, 3 * j + b)] += block;
                h[(3 * i + a, 3 * j + b)] -= block;
                h[(3 * j + a, 3 * i + b)] -= block;
            }
        }
    }
}

impl Calculator for TwoSurfaces {
    fn name(&self) -> &str {
        "two_surfaces"
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let s = surface(sys.spin_charge.multiplicity, req.excited_state);
        let x = sys.coords_flat_bohr();
        let mut e = 0.0;
        let mut g = Col::<f64>::zeros(9);
        let mut h = Mat::<f64>::zeros(9, 9);
        let harmonic = |at: f64| move |r: f64| (K * (r - at).powi(2), 2.0 * K * (r - at), 2.0 * K);
        Self::pair(&x, 0, 1, &|r| curve(s, r), &mut e, &mut g, &mut h);
        Self::pair(&x, 1, 2, &harmonic(AS[s]), &mut e, &mut g, &mut h);
        Self::pair(&x, 0, 2, &harmonic(C), &mut e, &mut g, &mut h);
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        if req.need_hessian {
            out.hessian_bohr = Some(h);
        }
        Ok(out)
    }
}

/// The energy of PES1 along the seam at `r12` (the `r23` of the seam, `r13 = C`).
fn on_seam(r12: f64) -> f64 {
    on_seam_of(0, 1, r12)
}

/// The energy along the seam of the surfaces `i` and `j` at `r12`: with the same `K` on both
/// the gap is linear in `r23`, so the seam has one `r23` at every `r12`.
fn on_seam_of(i: usize, j: usize, r12: f64) -> f64 {
    let (fi, fj) = (curve(i, r12).0, curve(j, r12).0);
    let r23 = 0.5 * (AS[i] + AS[j]) - (fi - fj) / (2.0 * K * (AS[j] - AS[i]));
    fi + K * (r23 - AS[i]).powi(2)
}

/// The local minima of the seam of `i` and `j` along `r12` (energies), from a grid refined by
/// golden sections.
fn seam_minima_of(i: usize, j: usize) -> Vec<f64> {
    let grid: Vec<f64> = (0..=600).map(|k| 1.2 + 0.01 * f64::from(k)).collect();
    let mut out = Vec::new();
    for w in grid.windows(3) {
        let (a, b, c) = (
            on_seam_of(i, j, w[0]),
            on_seam_of(i, j, w[1]),
            on_seam_of(i, j, w[2]),
        );
        if b <= a && b <= c {
            let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
            let (mut lo, mut hi) = (w[0], w[2]);
            for _ in 0..100 {
                let x = hi - ratio * (hi - lo);
                let y = lo + ratio * (hi - lo);
                if on_seam_of(i, j, x) < on_seam_of(i, j, y) {
                    hi = y;
                } else {
                    lo = x;
                }
            }
            out.push(on_seam_of(i, j, 0.5 * (lo + hi)));
        }
    }
    out
}

/// The minimum of the seam between `lo` and `hi`: `(r12, energy)` (golden-section search).
fn seam_minimum(lo: f64, hi: f64) -> (f64, f64) {
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    let (mut a, mut b) = (lo, hi);
    for _ in 0..200 {
        let c = b - ratio * (b - a);
        let d = a + ratio * (b - a);
        if on_seam(c) < on_seam(d) {
            b = d;
        } else {
            a = c;
        }
    }
    let r = 0.5 * (a + b);
    (r, on_seam(r))
}

/// The structure with the distances `r12`, `r23`, `r13` (Bohr).
fn triangle(r12: f64, r23: f64, r13: f64) -> [f64; 9] {
    let x3 = (r12 * r12 + r13 * r13 - r23 * r23) / (2.0 * r12);
    let y3 = (r13 * r13 - x3 * x3).max(0.0).sqrt();
    [0.0, 0.0, 0.0, r12, 0.0, 0.0, x3, y3, 0.0]
}

/// The distance of atoms `i` and `j` of a node.
fn distance(n: &Node, i: usize, j: usize) -> f64 {
    let x = n.system.flat();
    (0..3)
        .map(|k| (x[3 * j + k] - x[3 * i + k]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// A method with one task per minimum of PES1: the transition-state guess at the top of the
/// double well.
struct BarrierGuess;

impl ExplorationMethod for BarrierGuess {
    fn name(&self) -> &str {
        "barrier_guess"
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        let on_pes1 = node.state.multiplicity == 1 && node.state.excited_state == 0;
        Ok(if node.kind == NodeKind::Minimum && on_pes1 {
            vec![Task {
                key: "barrier".to_string(),
                description: "a transition-state guess at the top of the double well".to_string(),
                parameters: JsonValue::Null,
            }]
        } else {
            Vec::new()
        })
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let mut outcome = JobOutcome::default();
        let at = triangle(0.5 * (R1 + R2), AS[0], C);
        let guess = job.start.with_flat(&Col::from_fn(9, |i| at[i]))?;
        let guesses = [Guess {
            system: guess,
            origin: "the top of the double well".to_string(),
        }];
        if !follow_guesses(job, ctx, &guesses, &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

fn parts() -> MapperParts {
    let mut parts = MapperParts::default();
    parts.methods.register(
        "barrier_guess",
        Arc::new(|_: &MethodSettings| Ok(Box::new(BarrierGuess) as Box<dyn ExplorationMethod>)),
    );
    parts
}

fn registry() -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    r.register("two_surfaces", |_| Ok(Box::new(TwoSurfaces)));
    r
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("mopt_mapper_states_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn config(dir: &Path, extra: &[&str]) -> MapperConfig {
    config_of(dir, "Ar", extra)
}

/// [`config`] with the atoms `element` (the model does not depend on them; helium atoms are
/// never bonded at these distances, so that every pair is an AFIR push).
fn config_of(dir: &Path, element: &str, extra: &[&str]) -> MapperConfig {
    let start = start_file(dir, element);
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        r#"{
  "mapper_settings": {"max_iterations": 20, "rng_seed": 1, "method": "barrier_guess", "channels": false},
  "step1_settings": {"othersoft": "two_surfaces", "NSTEP": 300},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]
  }
}"#,
    )
    .unwrap();
    let mut args = vec![
        start.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

/// The minimum of PES1 at R1 as `start.xyz` in `dir`, of atoms `element`.
fn start_file(dir: &Path, element: &str) -> PathBuf {
    let start = dir.join("start.xyz");
    let x = triangle(R1, AS[0], C);
    let mut text = String::from("3\nthe minimum of PES1 at R1\n");
    for a in 0..3 {
        text.push_str(&format!(
            "{element} {:.10} {:.10} {:.10}\n",
            x[3 * a] * BOHR_TO_ANGSTROM,
            x[3 * a + 1] * BOHR_TO_ANGSTROM,
            x[3 * a + 2] * BOHR_TO_ANGSTROM
        ));
    }
    std::fs::write(&start, text).unwrap();
    start
}

fn summary(net: &Network) -> String {
    net.nodes
        .iter()
        .map(|n| {
            format!(
                "{} state {:?} r12 {:.4} r23 {:.4} r13 {:.4} E {:.8}",
                n.label(),
                (n.state.multiplicity, n.state.excited_state),
                distance(n, 0, 1),
                distance(n, 1, 2),
                distance(n, 0, 2),
                n.energy
            )
        })
        .chain(
            net.crossing_edges
                .iter()
                .map(|e| format!("crossing edge {}: {} -> {:?}", e.id, e.crossing, e.ends)),
        )
        .chain(
            net.warnings
                .iter()
                .map(|w| format!("warning: {}", w.message)),
        )
        .chain(
            net.jobs
                .iter()
                .map(|j| format!("job {}: {} {}", j.id, j.node, j.task)),
        )
        .collect::<Vec<_>>()
        .join("\n")
}

/// The settings of the search from the minimum itself alone (the model of argon atoms has
/// no pair of atoms for an AFIR push).
const DIRECT: [&str; 2] = ["crossing_afir=off", "crossing_direct=on"];

/// The network both kinds of crossing must give.
fn check(run: &MapperRun, kind: CrossingKind, excited: bool) {
    let net = &run.network;
    let s = summary(net);
    assert_eq!(net.states.len(), 2, "{s}");
    let pes1 = net.states[0];
    let pes2 = net.states[1];
    assert_eq!(pes2.multiplicity, if excited { 1 } else { 3 }, "{s}");
    // The minima: R1 and R2 on PES1, RC on PES2.
    let minimum = |state, r12: f64| {
        net.minima()
            .find(|n| n.state == state && (distance(n, 0, 1) - r12).abs() < 1e-3)
            .map(|n| n.id)
            .unwrap_or_else(|| panic!("the minimum at r12 = {r12}\n{s}"))
    };
    let (m1, m2, mc) = (minimum(pes1, R1), minimum(pes1, R2), minimum(pes2, RC));
    assert_eq!(net.minima().count(), 3, "{s}");
    assert!(net.nodes[m1].energy.abs() < 1e-8, "{s}");
    assert!((net.nodes[mc].energy - D).abs() < 1e-8, "{s}");
    // The transition state of PES1 at the top of the double well.
    let ts: Vec<&Node> = net.of_kind(&NodeKind::TransitionState).collect();
    assert_eq!(ts.len(), 1, "{s}");
    assert!(
        (distance(ts[0], 0, 1) - 0.5 * (R1 + R2)).abs() < 1e-3,
        "{s}"
    );
    assert_eq!(ts[0].state, pes1, "{s}");
    // The two crossing points at the minima of the seam.
    let crossing = NodeKind::Crossing {
        kind,
        states: [pes1, pes2],
    };
    let found: Vec<&Node> = net.of_kind(&crossing).collect();
    assert_eq!(found.len(), 2, "{s}");
    for (lo, hi, side) in [(R1, RC, m1), (RC, R2, m2)] {
        let (r, e) = seam_minimum(lo + 1e-6, hi - 1e-6);
        let x = found
            .iter()
            .find(|n| (distance(n, 0, 1) - r).abs() < 1e-3)
            .unwrap_or_else(|| panic!("the crossing point at r12 = {r:.4}, E = {e:.6}\n{s}"));
        assert!((x.energy - e).abs() < 1e-6, "{} against {e}\n{s}", x.energy);
        assert!(x.label().starts_with(kind.tag()), "{s}");
        // Joined to the minimum of PES1 on its side and to that of PES2.
        let edge = net
            .crossing_edges
            .iter()
            .find(|c| c.crossing == x.id)
            .unwrap_or_else(|| panic!("the crossing edge of {}\n{s}", x.label()));
        assert_eq!(edge.ends, [Some(side), Some(mc)], "{s}");
        let barrier = edge.barriers[0].unwrap();
        assert!((barrier - (e - net.nodes[side].energy)).abs() < 1e-6, "{s}");
        assert!(edge.gradient_difference.is_some_and(|g| g > 0.0), "{s}");
        // The path of the edge in its file: from the minimum of PES1 (the minimization of the
        // end of the meta-IRC backwards, then the meta-IRC backwards) through the crossing
        // point to the minimum of PES2.
        let file = edge.irc.file.as_deref().expect("the path file of the edge");
        assert!(file.ends_with("_path.xyz"), "{file}");
        let frames = path_frames(&run.directory.join(file));
        let labels: Vec<&str> = frames.iter().map(|f| f.0.as_str()).collect();
        let first = labels.first().unwrap();
        let last = labels.last().unwrap();
        assert!(first.starts_with("PES1 minimization"), "{labels:?}");
        assert!(last.starts_with("PES2 minimization"), "{labels:?}");
        let at = labels
            .iter()
            .position(|l| l.starts_with(kind.tag()))
            .unwrap_or_else(|| {
                panic!(
                    "the crossing point of the path
{labels:?}"
                )
            });
        assert!(
            labels[..at].iter().all(|l| l.starts_with("PES1"))
                && labels[at + 1..].iter().all(|l| l.starts_with("PES2")),
            "{labels:?}"
        );
        // Its ends are the two minima, and the crossing point lies where the node does.
        let r12 = |k: usize| frames[k].1;
        assert!(
            (r12(0) - distance(&net.nodes[side], 0, 1)).abs() < 1e-3,
            "{labels:?}"
        );
        assert!((r12(frames.len() - 1) - RC).abs() < 1e-3, "{labels:?}");
        assert!((r12(at) - r).abs() < 1e-3, "{labels:?}");
    }
}

/// The frames of an xyz path file of the model: the comment and the distance of the first two
/// atoms (Bohr) of each.
fn path_frames(file: &Path) -> Vec<(String, f64)> {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        if lines[at].trim().is_empty() {
            at += 1;
            continue;
        }
        let n: usize = lines[at].trim().parse().unwrap();
        let xyz = |k: usize| -> Vec<f64> {
            lines[at + 2 + k]
                .split_whitespace()
                .skip(1)
                .map(|v| v.parse().unwrap())
                .collect()
        };
        let (a, b) = (xyz(0), xyz(1));
        let d = (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        out.push((lines[at + 1].to_string(), d / BOHR_TO_ANGSTROM));
        at += 2 + n;
    }
    out
}

#[test]
fn transition_states_and_crossing_points_of_two_spin_states() {
    let dir = scratch("mecp");
    let c = config(&dir, &["-map_opt", "states=0,1;0,3", DIRECT[0], DIRECT[1]]);
    let run = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    check(&run, CrossingKind::Mecp, false);
    // Every crossing edge has the meta-IRC of both states: each goes down from the crossing
    // point on its own state, and their structures are in the file of the edge.
    for e in &run.network.crossing_edges {
        assert_eq!(e.irc.sides.len(), 2, "{}", summary(&run.network));
        assert_eq!(
            [e.irc.sides[0].name.as_str(), e.irc.sides[1].name.as_str()],
            ["PES1", "PES2"]
        );
        for side in &e.irc.sides {
            let (first, last) = (side.energies[0], side.energies[side.energies.len() - 1]);
            assert!(side.energies.len() >= 3 && last < first, "{side:?}");
        }
        let file = e.irc.file.as_ref().expect("the file of the paths");
        assert!(dir.join("mapper_output").join(file).is_file(), "{file}");
    }
    // The network file and the database keep the states.
    let file = dir.join("mapper_output").join("reaction_network.json");
    let doc = JsonValue::parse(&std::fs::read_to_string(&file).unwrap()).unwrap();
    let (read, _) = mopt_rs::workflow::mapper::network_json::from_json(&doc).unwrap();
    assert_eq!(read.states, run.network.states);
}

#[test]
fn transition_states_and_conical_intersections_of_two_roots() {
    let dir = scratch("meci");
    let c = config(
        &dir,
        &["-map_opt", "states=0,1;0,1,1", DIRECT[0], DIRECT[1]],
    );
    let run = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    check(&run, CrossingKind::Meci, true);
}

#[test]
fn three_states_give_crossing_points_of_every_pair() {
    // PES1 (singlet), PES2 (triplet), PES3 (the second excited singlet): MECP between PES1
    // and PES2 and between PES2 and PES3, MECI between PES1 and PES3.
    let dir = scratch("three");
    let c = config(
        &dir,
        &[
            "-map_opt",
            "states=0,1;0,3;0,1,2",
            "kinetics=on",
            DIRECT[0],
            DIRECT[1],
        ],
    );
    let run = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    let net = &run.network;
    let s = summary(net);
    assert_eq!(net.states.len(), 3, "{s}");
    // The minima of every surface.
    for (state, r12) in [(0, R1), (0, R2), (1, RC), (2, RC3)] {
        assert!(
            net.minima()
                .any(|n| n.state == net.states[state] && (distance(n, 0, 1) - r12).abs() < 1e-3),
            "the minimum of PES{} at r12 = {r12}\n{s}",
            state + 1
        );
    }
    // Every crossing point lies at a minimum of the seam of its pair, and every pair has one.
    let mut pairs = Vec::new();
    for n in net.nodes.iter() {
        let NodeKind::Crossing { kind, states } = &n.kind else {
            continue;
        };
        let i = net.states.iter().position(|x| x == &states[0]).unwrap();
        let j = net.states.iter().position(|x| x == &states[1]).unwrap();
        assert!(i < j, "{s}");
        let expected = if (i, j) == (0, 2) {
            CrossingKind::Meci
        } else {
            CrossingKind::Mecp
        };
        assert_eq!(*kind, expected, "{s}");
        let minima = seam_minima_of(i, j);
        assert!(
            minima.iter().any(|e| (e - n.energy).abs() < 1e-6),
            "{} at {} is at none of the seam minima {minima:?}\n{s}",
            n.label(),
            n.energy
        );
        pairs.push((i, j));
    }
    for pair in [(0, 1), (0, 2), (1, 2)] {
        assert!(pairs.contains(&pair), "no crossing point of {pair:?}\n{s}");
    }
    // Every crossing edge joins minima of the two states of its crossing point.
    for e in &net.crossing_edges {
        let NodeKind::Crossing { states, .. } = net.nodes[e.crossing].kind else {
            panic!("{s}");
        };
        for (side, end) in e.ends.iter().enumerate() {
            let end = end.unwrap_or_else(|| panic!("an end of {}\n{s}", e.crossing));
            assert_eq!(net.nodes[end].state, states[side], "{s}");
            assert_eq!(net.nodes[end].kind, NodeKind::Minimum, "{s}");
        }
    }
    // The kinetics of the network ran with the crossing points as bottlenecks.
    assert!(dir.join("mapper_output").join("kinetics").exists(), "{s}");
}

#[test]
fn adjacent_pairs_leave_out_the_crossing_points_of_states_apart() {
    // With crossing_pairs=adjacent the PES1/PES3 pair is not searched: crossing points of
    // PES1/PES2 and PES2/PES3 only.
    let dir = scratch("adjacent");
    let c = config(
        &dir,
        &[
            "-map_opt",
            "states=0,1;0,3;0,1,2",
            "crossing_pairs=adjacent",
            DIRECT[0],
            DIRECT[1],
        ],
    );
    let run = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    let net = &run.network;
    let s = summary(net);
    let mut pairs = Vec::new();
    for n in net.nodes.iter() {
        if let NodeKind::Crossing { states, .. } = &n.kind {
            let i = net.states.iter().position(|x| x == &states[0]).unwrap();
            let j = net.states.iter().position(|x| x == &states[1]).unwrap();
            pairs.push((i, j));
        }
    }
    assert!(pairs.contains(&(0, 1)) && pairs.contains(&(1, 2)), "{s}");
    assert!(!pairs.contains(&(0, 2)), "{s}");
    assert!(
        net.jobs.iter().all(|j| {
            let here = net.state_of(j.node).unwrap();
            j.task
                .strip_prefix("crossing:")
                .is_none_or(|t| t.parse::<usize>().unwrap().abs_diff(here) == 1)
        }),
        "{s}"
    );
}

/// The crossing points the AFIR jobs towards crossing points found: (states, energy).
#[derive(Default)]
struct ModelAfirFinds(Vec<([usize; 2], f64)>);

impl MapperObserver for ModelAfirFinds {
    fn after_job(
        &mut self,
        _network: &Network,
        job: &Job,
        outcome: &JobOutcome,
        _told: &Told,
    ) -> MoptResult<()> {
        if job.task.key.starts_with("xafir:") || job.task.key.starts_with("gpafir:") {
            for c in &outcome.crossings {
                self.0.push((c.states, c.point.energy));
            }
        }
        Ok(())
    }
}

/// The AFIR pushes towards crossing points (`afir`: `crossing_afir=gp`, the gradient
/// projection, or `crossing_afir=model`, the seam model function for an MECP and the Avoiding
/// model function for an MECI) reach crossing points at the minima of the seam.
fn model_afir(name: &str, states: &str, kind: CrossingKind, afir: &str, more: &[&str]) {
    let dir = scratch(name);
    let mut args = vec!["-map_opt", states, afir];
    args.extend_from_slice(more);
    args.extend_from_slice(&["--max_iter", "30"]);
    let c = config_of(&dir, "He", &args);
    let mut finds = ModelAfirFinds::default();
    let run = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut [&mut finds]).unwrap();
    let net = &run.network;
    let s = summary(net);
    let prefix = if afir.ends_with("model") {
        "xafir:"
    } else {
        "gpafir:"
    };
    assert!(net.jobs.iter().any(|j| j.task.starts_with(prefix)), "{s}");
    assert!(
        !net.jobs.iter().any(|j| j.task.starts_with("crossing:")),
        "{s}"
    );
    // The tasks of a minimum alternate: a crossing task, then a task of the method.
    let first: Vec<&str> = net
        .jobs
        .iter()
        .filter(|j| j.node == 0)
        .map(|j| j.task.as_str())
        .take(2)
        .collect();
    assert!(
        first[0].starts_with(prefix) && first[1] == "barrier",
        "{first:?}\n{s}"
    );
    assert!(
        !finds.0.is_empty(),
        "no AFIR path reached a crossing point\n{s}"
    );
    let minima = seam_minima_of(0, 1);
    for (pair, e) in &finds.0 {
        assert_eq!(*pair, [0, 1], "{s}");
        assert!(
            minima.iter().any(|m| (m - e).abs() < 1e-6),
            "{e} is at none of the seam minima {minima:?}\n{s}"
        );
    }
    for n in net.nodes.iter() {
        if let NodeKind::Crossing { kind: k, .. } = &n.kind {
            assert_eq!(*k, kind, "{s}");
        }
    }
}

#[test]
fn afir_on_the_seam_model_function_reaches_the_mecp() {
    model_afir(
        "smf_afir",
        "states=0,1;0,3",
        CrossingKind::Mecp,
        "crossing_afir=model",
        &[],
    );
}

#[test]
fn afir_on_the_avoiding_model_function_reaches_the_meci() {
    model_afir(
        "amf_afir",
        "states=0,1;0,1,1",
        CrossingKind::Meci,
        "crossing_afir=model",
        &[],
    );
}

#[test]
fn afir_on_the_lower_avoiding_model_function_reaches_the_meci_at_its_ridge() {
    // On the lower adiabatic state the seam is a ridge: the candidates are the maxima of the
    // path.
    model_afir(
        "amf_lower_afir",
        "states=0,1;0,1,1",
        CrossingKind::Meci,
        "crossing_afir=model",
        &["crossing_amf_branch=lower"],
    );
}

#[test]
fn afir_on_the_gradient_projection_reaches_the_mecp_by_default() {
    // crossing_afir=gp is the default with states: nothing is said here.
    model_afir(
        "gp_afir",
        "states=0,1;0,3",
        CrossingKind::Mecp,
        "crossing_pairs=all",
        &[],
    );
}

#[test]
fn afir_on_the_gradient_projection_reaches_the_meci() {
    model_afir(
        "gp_afir_meci",
        "states=0,1;0,1,1",
        CrossingKind::Meci,
        "crossing_afir=gp",
        &[],
    );
}

#[test]
fn transition_states_are_searched_on_the_states_asked_for() {
    // ts_states=2: the method (the guess of the barrier of PES1) runs on no minimum of PES1;
    // the crossing points are still found.
    let dir = scratch("ts_states");
    let c = config(
        &dir,
        &[
            "-map_opt",
            "states=0,1;0,3",
            "ts_states=2",
            DIRECT[0],
            DIRECT[1],
        ],
    );
    let run = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    let net = &run.network;
    let s = summary(net);
    assert_eq!(net.of_kind(&NodeKind::TransitionState).count(), 0, "{s}");
    assert!(!net.jobs.iter().any(|j| j.task == "barrier"), "{s}");
    assert!(!net.crossing_edges.is_empty(), "{s}");
}

#[test]
fn an_input_is_minimized_on_its_own_state() {
    // The first state of the run is the first excited singlet (the second surface of the
    // model, its minimum at RC, D above PES1): the input, the minimum of the ground state at
    // R1, relaxes on that surface and not on the ground state of its spin.
    let dir = scratch("input_state");
    let c = config(&dir, &["-map_opt", "states=0,1,1;0,1,2", "crossings=off"]);
    let run = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    let net = &run.network;
    let s = summary(net);
    let node = &net.nodes[0];
    assert_eq!(
        (node.state.multiplicity, node.state.excited_state),
        (1, 1),
        "{s}"
    );
    assert!((distance(node, 0, 1) - RC).abs() < 1e-4, "{s}");
    assert!((distance(node, 1, 2) - AS[1]).abs() < 1e-4, "{s}");
    assert!((node.energy - D).abs() < 1e-8, "{s}");
}

/// The settings of `run_automecp` on the model in `dir`: the pushes `ma`, the second state
/// `state2`, and `more` arguments.
fn automecp_config(dir: &Path, ma: &[&str], state2: &[&str], more: &[&str]) -> AutomecpConfig {
    let start = start_file(dir, "He");
    let cfg = dir.join("automecp.json");
    std::fs::write(
        &cfg,
        r#"{
  "step1_settings": {"othersoft": "two_surfaces", "NSTEP": 300},
  "step2_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step3_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]
  }
}"#,
    )
    .unwrap();
    let mut args = vec![
        start.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
    ];
    if !ma.is_empty() {
        args.push("-ma".to_string());
        args.extend(ma.iter().map(|s| s.to_string()));
    }
    if !state2.is_empty() {
        args.push("-state2".to_string());
        args.extend(state2.iter().map(|s| s.to_string()));
    }
    args.extend(more.iter().map(|s| s.to_string()));
    let parsed = parser("run_automecp", RUN_AUTOMECP).parse(args).unwrap();
    AutomecpConfig::from_args(&parsed).unwrap()
}

#[test]
fn run_automecp_finds_the_crossing_points_of_the_pushes_it_is_given() {
    let dir = scratch("automecp");
    // A push and a pull of the first two atoms, towards the triplet (MECP).
    let c = automecp_config(&dir, &["100", "1", "2", "-100", "1", "2"], &["0", "3"], &[]);
    let run = mopt_rs::cli::automecp::run_with(&c, &dir, None, &registry()).unwrap();
    let net = &run.mapper.network;
    let s = summary(net);
    // The crossing points lie at minima of the seam, each joined to minima of both states by
    // its meta-IRC paths.
    assert!(!net.crossing_edges.is_empty(), "{s}");
    let minima = seam_minima_of(0, 1);
    for e in &net.crossing_edges {
        let x = &net.nodes[e.crossing];
        assert!(
            matches!(
                x.kind,
                NodeKind::Crossing {
                    kind: CrossingKind::Mecp,
                    ..
                }
            ),
            "{s}"
        );
        assert!(minima.iter().any(|m| (m - x.energy).abs() < 1e-6), "{s}");
        assert!(e.ends.iter().all(Option::is_some), "{s}");
        assert_eq!(e.irc.sides.len(), 2, "{s}");
    }
    // Only the pushes given, from the input alone, and no search for transition states.
    assert_eq!(net.jobs.len(), 2, "{s}");
    assert!(
        net.jobs
            .iter()
            .all(|j| j.node == 0 && j.task.starts_with("gpafir:1:afir:")),
        "{s}"
    );
    assert_eq!(net.of_kind(&NodeKind::TransitionState).count(), 0, "{s}");
    // The files: the summary (one line per crossing edge) and the run of the mapper.
    let text = std::fs::read_to_string(&run.summary).unwrap();
    assert_eq!(text.lines().count(), 1 + net.crossing_edges.len(), "{text}");
    assert!(
        text.lines()
            .nth(1)
            .unwrap()
            .starts_with("MECP0,MECP,PES1,PES2,"),
        "{text}"
    );
    let out = dir.join("automecp_output");
    assert_eq!(run.directory, out);
    for f in [
        "settings.json",
        "mapper_config.json",
        "mapper/reaction_network.json",
        "automecp_report.html",
    ] {
        assert!(out.join(f).is_file(), "{f}");
    }
    // The page of the search: its steps, every crossing point with the way between its two
    // minima, and the energies of the two states along the paths of the pushes.
    assert_eq!(run.report, out.join("automecp_report.html"));
    let page = std::fs::read_to_string(&run.report).unwrap();
    for part in [
        "<title>AutoMECP start</title>",
        "Step 1 · AFIR paths towards the crossing",
        "Step 2 · Crossing points",
        "Step 3 · Meta-IRC and minima",
        "Energies of the two states along the AFIR path",
        "Push 2, candidate 1",
        "PES1: charge 0, multiplicity 1; PES2: charge 0, multiplicity 3",
    ] {
        assert!(page.contains(part), "{part}");
    }
    // A title alone is insufficient: compare the two plotted series with the
    // number of state-energy pairs in the actual per-push files.
    let section = page
        .split_once("id=\"step1\"")
        .unwrap()
        .1
        .split_once("id=\"step2\"")
        .unwrap()
        .0;
    let plots: Vec<_> = section
        .split("<svg")
        .skip(1)
        .map(|svg| svg.split_once("</svg>").unwrap().0)
        .filter(|svg| svg.contains("Energies of the two states along the AFIR path"))
        .collect();
    assert_eq!(plots.len(), net.jobs.len());
    for (job, svg) in net.jobs.iter().zip(plots) {
        let manifest = out.join("mapper").join(job.trajectory.as_ref().unwrap());
        let index = std::fs::read_to_string(&manifest).unwrap();
        let (_, afir) = index
            .lines()
            .find(|row| row.starts_with("\"AFIR path on"))
            .unwrap()
            .rsplit_once(',')
            .unwrap();
        let data = std::fs::read_to_string(manifest.parent().unwrap().join(afir)).unwrap();
        let points = data.lines().filter(|row| row.contains(" states=")).count();
        assert!(points >= 2);
        let series: Vec<_> = svg
            .split("<polyline ")
            .skip(1)
            .map(|line| {
                line.split_once("points=\"")
                    .unwrap()
                    .1
                    .split('"')
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .count()
            })
            .collect();
        assert_eq!(series, [points, points], "AFIR energies of job {}", job.id);
    }
    for e in &net.crossing_edges {
        let x = net.nodes[e.crossing].label();
        assert!(page.contains(&format!("{x}_path")), "{x}");
        assert!(
            page.contains(&format!("Energy along the meta-IRC of each state from {x}")),
            "{x}"
        );
    }
    assert!(page.contains("PES1 minimization") && page.contains("PES2 minimization"));
    assert!(
        !page.contains("did not converge to a crossing point"),
        "a candidate failed"
    );
    // The run directory is not written over, and --restart goes on to the same network.
    assert!(mopt_rs::cli::automecp::run_with(&c, &dir, None, &registry()).is_err());
    let mut again = c.clone();
    again.restart_dir = Some(out);
    let rerun = mopt_rs::cli::automecp::run_with(&again, &dir, None, &registry()).unwrap();
    assert_eq!(rerun.mapper.network.nodes.len(), net.nodes.len());
    assert_eq!(rerun.mapper.network.jobs.len(), 2);
}

#[test]
fn run_automecp_on_the_model_function_and_its_settings() {
    // The path on the Avoiding model function towards the first excited singlet (MECI), by a
    // push of the atoms 1 and 3.
    let dir = scratch("automecp_model");
    let c = automecp_config(
        &dir,
        &["100", "1", "3"],
        &["0", "1", "1"],
        &["-afir", "model"],
    );
    let run = mopt_rs::cli::automecp::run_with(&c, &dir, None, &registry()).unwrap();
    let net = &run.mapper.network;
    let s = summary(net);
    assert!(
        net.jobs.iter().all(|j| j.task.starts_with("xafir:1:afir:")),
        "{s}"
    );
    let minima = seam_minima_of(0, 1);
    let crossings: Vec<&Node> = net
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n.kind,
                NodeKind::Crossing {
                    kind: CrossingKind::Meci,
                    ..
                }
            )
        })
        .collect();
    assert!(!crossings.is_empty(), "{s}");
    for x in &crossings {
        assert!(minima.iter().any(|m| (m - x.energy).abs() < 1e-6), "{s}");
    }
    let text = std::fs::read_to_string(&run.summary).unwrap();
    assert!(
        text.lines()
            .nth(1)
            .is_some_and(|l| l.starts_with("MECI0,MECI,PES1,PES2,")),
        "{text}"
    );
    // The second state and the pushes are needed, and the pushes are triples.
    let other = scratch("automecp_settings");
    for (ma, state2) in [
        (&["100", "1", "2"][..], &[][..]),
        (&[][..], &["0", "3"][..]),
        (&["100", "1"][..], &["0", "3"][..]),
        (&["100", "1", "2"][..], &["0"][..]),
    ] {
        let c = automecp_config(&other, ma, state2, &[]);
        let err = mopt_rs::cli::automecp::run_with(&c, &other, None, &registry());
        assert!(err.is_err(), "{ma:?} {state2:?}");
        assert!(!other.join("automecp_output").exists());
    }
}

#[test]
fn the_settings_of_the_states_are_checked() {
    let dir = scratch("settings");
    let bad = |opt: &str| {
        let start = dir.join("start.xyz");
        let x = triangle(R1, AS[0], C);
        let mut text = String::from("3\n\n");
        for a in 0..3 {
            text.push_str(&format!(
                "Ar {} {} {}\n",
                x[3 * a] * BOHR_TO_ANGSTROM,
                x[3 * a + 1] * BOHR_TO_ANGSTROM,
                x[3 * a + 2] * BOHR_TO_ANGSTROM
            ));
        }
        std::fs::write(&start, text).unwrap();
        let args = vec![
            start.display().to_string(),
            "-map_opt".to_string(),
            opt.to_string(),
        ];
        let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
        MapperConfig::from_args(&parsed, None)
            .err()
            .map(|e| e.to_string())
    };
    // One state, a state twice, a bad state, crossings without states.
    for opt in [
        "states=0,1",
        "states=0,1;0,1",
        "states=0,1;0,x",
        "crossings=on",
        "crossing_pairs=adjacent",
        "crossing_afir=gp",
        "crossing_amf_alpha=0.01",
        "ts_states=1",
        "states=0,1;0,3 ts_states=3",
        "states=0,1;0,3 crossing_afir=seam",
        "states=0,1;0,3 crossing_afir=off",
        "states=0,1;0,3 crossing_gap=0",
        "crossing_gap=0.001",
        "states=0,1;0,3 crossing_candidate_prominence=-1",
        "states=0,1;0,3 crossing_candidate_distance=-1",
        "states=0,1;0,3 crossing_parallel_gradient=0",
        "states=0,1;0,3 crossing_parallel_steps=0",
        "crossing_parallel_stop=on",
    ] {
        assert!(bad(opt).is_some(), "{opt}");
    }
    assert!(bad("states=0,1;0,3").is_none());
    assert!(mopt_rs::workflow::mapper::CrossingPairs::parse("some").is_err());
    // An input on none of the states is refused when the run starts.
    let other = scratch("settings_input");
    let c = config(&other, &["-map_opt", "states=0,3;0,5"]);
    let err = run_with_parts(&c, &other, None, &registry(), &parts(), &mut [])
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("none of the states"), "{err}");
}

#[test]
fn a_search_that_ends_with_an_open_gap_is_no_crossing_point() {
    // With a limit below the gap that any of these searches ends with (1e-8 to 1e-11
    // Hartree), every search that converges is refused: a warning says how far apart the two
    // states are, and nothing becomes a crossing point or runs down from it.
    let dir = scratch("gap");
    let c = config_of(
        &dir,
        "He",
        &[
            "-map_opt",
            "states=0,1;0,3",
            "crossing_gap=1e-14",
            "ts_states=none",
            "--max_iter",
            "4",
        ],
    );
    let run = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    let net = &run.network;
    let s = summary(net);
    assert!(net.crossing_edges.is_empty(), "{s}");
    assert!(
        !net.nodes
            .iter()
            .any(|n| matches!(n.kind, NodeKind::Crossing { .. })),
        "{s}"
    );
    let refused: Vec<&str> = net
        .warnings
        .iter()
        .map(|w| w.message.as_str())
        .filter(|m| m.contains("more than crossing_gap (1.000e-14 Hartree)"))
        .collect();
    assert!(!refused.is_empty(), "{:?}", net.warnings);
    assert!(
        refused
            .iter()
            .all(|m| m.contains("no crossing point, and no meta-IRC from it")),
        "{refused:?}"
    );
}

#[test]
fn a_search_is_ended_where_its_two_states_run_parallel() {
    // The rule is off by default. Asked for with a threshold above the difference of the two
    // gradients of this model at its seams (0.11 Hartree/Bohr), it takes every search that
    // has closed the gap for a search between parallel states and ends it after three such
    // iterations: a warning says so, and nothing becomes a crossing point.
    let dir = scratch("parallel");
    let c = config_of(
        &dir,
        "He",
        &[
            "-map_opt",
            "states=0,1;0,1,1",
            "crossing_parallel_stop=on",
            "crossing_parallel_gradient=1.0",
            "crossing_parallel_steps=3",
            "ts_states=none",
            "--max_iter",
            "4",
        ],
    );
    let run = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    let net = &run.network;
    let s = summary(net);
    assert!(net.crossing_edges.is_empty(), "{s}");
    let ended: Vec<&str> = net
        .warnings
        .iter()
        .map(|w| w.message.as_str())
        .filter(|m| m.contains("was ended where the two states run parallel"))
        .collect();
    assert!(!ended.is_empty(), "{:?}", net.warnings);
    assert!(
        ended
            .iter()
            .all(|m| m.contains("for 3 iterations") && m.contains("crossing_parallel_stop")),
        "{ended:?}"
    );
    // With the threshold of the rule (0.01 Hartree/Bohr) nothing of this model is parallel:
    // the run is that without the rule.
    let dir = scratch("parallel_default");
    let more = ["ts_states=none", "--max_iter", "4"];
    let on = config_of(
        &dir,
        "He",
        &[
            &["-map_opt", "states=0,1;0,1,1", "crossing_parallel_stop=on"][..],
            &more[..],
        ]
        .concat(),
    );
    let with_rule = run_with_parts(&on, &dir, None, &registry(), &parts(), &mut []).unwrap();
    let dir = scratch("parallel_off");
    let off = config_of(
        &dir,
        "He",
        &[&["-map_opt", "states=0,1;0,1,1"][..], &more[..]].concat(),
    );
    let without = run_with_parts(&off, &dir, None, &registry(), &parts(), &mut []).unwrap();
    assert_eq!(summary(&with_rule.network), summary(&without.network));
    assert!(!with_rule.network.crossing_edges.is_empty());
}

struct StopAtFrame {
    count: usize,
    at: usize,
    stop: PathBuf,
}

impl MapperObserver for StopAtFrame {
    fn on_frame(
        &mut self,
        _source: &Source<'_>,
        _system: &Configuration,
        _energy: f64,
        _label: &str,
    ) -> MoptResult<()> {
        if self.count == self.at {
            std::fs::write(&self.stop, "").unwrap();
        }
        self.count += 1;
        Ok(())
    }
}

#[test]
fn a_run_stopped_during_the_crossing_jobs_continues_to_the_same_network() {
    let dir = scratch("restart");
    let whole_dir = dir.join("whole");
    std::fs::create_dir_all(&whole_dir).unwrap();
    let opts = ["-map_opt", "states=0,1;0,3", DIRECT[0], DIRECT[1]];
    let c = config(&whole_dir, &opts);
    let whole = run_with_parts(&c, &whole_dir, None, &registry(), &parts(), &mut []).unwrap();
    let mut interrupted = 0;
    for at in [3, 20, 60, 150] {
        let base = dir.join(format!("at_{at}"));
        std::fs::create_dir_all(&base).unwrap();
        let c = config(&base, &opts);
        let mut stopper = StopAtFrame {
            count: 0,
            at,
            stop: base.join("mapper_output").join("STOP"),
        };
        let stopped =
            run_with_parts(&c, &base, None, &registry(), &parts(), &mut [&mut stopper]).unwrap();
        if !stopped.interrupted {
            continue;
        }
        interrupted += 1;
        let run_dir = base.join("mapper_output");
        let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
        let again = MapperConfig::from_args(&settings, Some(&run_dir)).unwrap();
        let done = run_with_parts(&again, &base, None, &registry(), &parts(), &mut []).unwrap();
        assert!(!done.interrupted, "{at}");
        let (a, b) = (&done.network, &whole.network);
        assert_eq!(a.nodes.len(), b.nodes.len(), "{at}\n{}", summary(a));
        for (x, y) in a.nodes.iter().zip(&b.nodes) {
            assert_eq!((x.kind, x.state), (y.kind, y.state), "{at}");
            assert!((x.energy - y.energy).abs() < 1e-9, "{at}: {}", x.label());
        }
        assert_eq!(a.crossing_edges, b.crossing_edges, "{at}");
        assert_eq!(a.ts_edges.len(), b.ts_edges.len(), "{at}");
    }
    assert!(
        interrupted >= 3,
        "only {interrupted} of the stops came during the run"
    );
}

/// The model with its two excited states counted by their energies, as a program for
/// electronic structure counts them: excited state 1 is the lower of the two surfaces of
/// [`TwoSurfaces`] at every structure and excited state 2 the upper, and the states are handed
/// out with their characters (which of the two surfaces each is), so that they can be followed
/// from structure to structure. The ground state lies far below and is not used.
struct OrderedSurfaces;

impl OrderedSurfaces {
    /// The energy, gradient and Hessian of the surface `s` (0: the double well, 1: the single
    /// well) at `x`.
    fn surface(x: &Col<f64>, s: usize) -> (f64, Col<f64>, Mat<f64>) {
        let mut e = 0.0;
        let mut g = Col::<f64>::zeros(9);
        let mut h = Mat::<f64>::zeros(9, 9);
        let harmonic = |at: f64| move |r: f64| (K * (r - at).powi(2), 2.0 * K * (r - at), 2.0 * K);
        TwoSurfaces::pair(x, 0, 1, &|r| curve(s, r), &mut e, &mut g, &mut h);
        TwoSurfaces::pair(x, 1, 2, &harmonic(AS[s]), &mut e, &mut g, &mut h);
        TwoSurfaces::pair(x, 0, 2, &harmonic(C), &mut e, &mut g, &mut h);
        (e, g, h)
    }

    /// The surfaces in the order of their energies at `x`.
    fn order(x: &Col<f64>) -> [usize; 2] {
        if Self::surface(x, 0).0 <= Self::surface(x, 1).0 {
            [0, 1]
        } else {
            [1, 0]
        }
    }
}

impl Calculator for OrderedSurfaces {
    fn name(&self) -> &str {
        "ordered_surfaces"
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let x = sys.coords_flat_bohr();
        let (e, g, h) = match req.excited_state {
            0 => {
                let (e, g, h) = Self::surface(&x, 0);
                (e - 1.0, g, h)
            }
            k => Self::surface(&x, Self::order(&x)[(k - 1).min(1)]),
        };
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        if req.need_hessian {
            out.hessian_bohr = Some(h);
        }
        Ok(out)
    }

    fn state_characters(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<StateCharacters>> {
        let x = sys.coords_flat_bohr();
        let order = Self::order(&x);
        // Excitation energies from the ground state, ascending; state `k` is the surface
        // `order[k]`, its amplitude on the excitation of that number.
        let values: Vec<f64> = order
            .iter()
            .map(|&s| Self::surface(&x, s).0 + 1.0)
            .collect();
        let amplitudes = Mat::from_fn(2, 2, |p, k| if p == order[k] { 1.0 } else { 0.0 });
        Ok(Some(
            StateCharacters::new(Mat::<f64>::identity(3, 3), 1, values, amplitudes)?.lowest(count),
        ))
    }
}

/// [`BarrierGuess`] from the minima of the first excited state of the run.
struct ExcitedBarrierGuess;

impl ExplorationMethod for ExcitedBarrierGuess {
    fn name(&self) -> &str {
        "excited_barrier_guess"
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        Ok(
            if node.kind == NodeKind::Minimum && node.state.excited_state == 1 {
                vec![Task {
                    key: "barrier".to_string(),
                    description: "a transition-state guess at the top of the double well"
                        .to_string(),
                    parameters: JsonValue::Null,
                }]
            } else {
                Vec::new()
            },
        )
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        BarrierGuess.run(job, ctx)
    }
}

/// A run on [`OrderedSurfaces`] with the states `0,1,1;0,1,2`, the excited states followed
/// from structure to structure when `follow`.
fn ordered_run(name: &str, follow: bool) -> MapperRun {
    let dir = scratch(name);
    let start = start_file(&dir, "He");
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        format!(
            r#"{{
  "mapper_settings": {{"max_iterations": 12, "rng_seed": 1, "method": "excited_barrier_guess", "channels": false}},
  "step1_settings": {{"othersoft": "ordered_surfaces", "NSTEP": 300, "follow_states": {follow}}},
  "step3_settings": {{"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300}},
  "step4_settings": {{
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]
  }}
}}"#
        ),
    )
    .unwrap();
    let args = vec![
        start.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "-map_opt".to_string(),
        "states=0,1,1;0,1,2".to_string(),
    ];
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    let c = MapperConfig::from_args(&parsed, None).unwrap();
    let mut registry = CalculatorRegistry::native();
    registry.register("ordered_surfaces", |_| Ok(Box::new(OrderedSurfaces)));
    let mut parts = MapperParts::default();
    parts.methods.register(
        "excited_barrier_guess",
        Arc::new(|_: &MethodSettings| {
            Ok(Box::new(ExcitedBarrierGuess) as Box<dyn ExplorationMethod>)
        }),
    );
    run_with_parts(&c, &dir, None, &registry, &parts, &mut []).unwrap()
}

#[test]
fn followed_states_are_the_same_states_from_node_to_node() {
    // The two excited states of the model change places between the minima: counted by
    // their energies, "excited state 1" is the double well near R1 and R2 and the single well
    // near RC. Followed from structure to structure and from node to node, the states of
    // the run are the two surfaces themselves, and the network is that of the model with
    // its surfaces named: the minima at R1 and R2 and their transition state on PES1, the
    // minimum at RC on PES2, and the two conical intersections between them.
    let run = ordered_run("followed", true);
    check(&run, CrossingKind::Meci, true);
    let net = &run.network;
    let s = summary(net);
    // Every node knows the place of the states of the run in the order of the energies at
    // its structure: at R1 and R2 the first state is the lowest excited state, at RC the
    // second is.
    let roots_at = |r12: f64| {
        net.minima()
            .find(|n| (distance(n, 0, 1) - r12).abs() < 1e-3)
            .map(|n| n.roots.clone())
            .unwrap_or_else(|| panic!("the minimum at r12 = {r12}\n{s}"))
    };
    for r in [R1, R2] {
        assert!(roots_at(r).contains(&(1, 0)), "{:?}\n{s}", roots_at(r));
    }
    assert!(roots_at(RC).contains(&(2, 0)), "{:?}\n{s}", roots_at(RC));
    // The network file keeps them, and a network read back has them.
    let text = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
    assert!(text.contains("\"followed_states\""), "{s}");
    let again =
        mopt_rs::workflow::mapper::network_json::from_json(&JsonValue::parse(&text).unwrap())
            .unwrap();
    for (a, b) in again.0.nodes.iter().zip(&net.nodes) {
        assert_eq!(a.roots, b.roots, "{}", b.label());
    }
}

#[test]
fn the_states_of_the_ordered_model_change_places_between_the_minima() {
    // What the test above rests on: counted by their energies, excited state 1 of the model
    // is the double well at R1 (energy 0) and the single well at RC (energy D), and the
    // characters that it hands out say which is which.
    let helium = mopt_rs::domain::Element::parse_token("He").unwrap();
    let at = |r12: f64, r23: f64| {
        let x = triangle(r12, r23, C);
        MolecularSystem::from_flat_bohr(
            vec![helium; 3],
            &Col::from_fn(9, |i| x[i]),
            mopt_rs::domain::SpinCharge::default(),
            None,
        )
        .unwrap()
    };
    let first = EvalRequest {
        excited_state: 1,
        ..EvalRequest::ENERGY_GRAD
    };
    let calc = OrderedSurfaces;
    let energy = |sys: &MolecularSystem| calc.evaluate(sys, &first).unwrap().energy.value();
    let (near_r1, near_rc) = (at(R1, AS[0]), at(RC, AS[1]));
    assert!(energy(&near_r1).abs() < 1e-12);
    assert!((energy(&near_rc) - D).abs() < 1e-12);
    let lowest = |sys: &MolecularSystem| {
        let states = calc.state_characters(sys, 2).unwrap().unwrap();
        (0..2).find(|&p| states.amplitudes[(p, 0)] == 1.0).unwrap()
    };
    assert_eq!((lowest(&near_r1), lowest(&near_rc)), (0, 1));
}

/// [`TwoSurfaces`] with one more term in `r12`, the same on every surface: the seams of the
/// model stay where they are, and every minimum, transition state and minimum of a seam is at
/// another place and energy.
struct Tilted;

/// The term of [`Tilted`]: `W (r - RW)^2`.
const W: f64 = 0.0015;
const RW: f64 = 3.0;

impl Calculator for Tilted {
    fn name(&self) -> &str {
        "two_surfaces_tilted"
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let s = surface(sys.spin_charge.multiplicity, req.excited_state);
        let x = sys.coords_flat_bohr();
        let mut e = 0.0;
        let mut g = Col::<f64>::zeros(9);
        let mut h = Mat::<f64>::zeros(9, 9);
        let harmonic = |at: f64| move |r: f64| (K * (r - at).powi(2), 2.0 * K * (r - at), 2.0 * K);
        let tilt = |r: f64| (W * (r - RW).powi(2), 2.0 * W * (r - RW), 2.0 * W);
        TwoSurfaces::pair(&x, 0, 1, &|r| curve(s, r), &mut e, &mut g, &mut h);
        TwoSurfaces::pair(&x, 0, 1, &tilt, &mut e, &mut g, &mut h);
        TwoSurfaces::pair(&x, 1, 2, &harmonic(AS[s]), &mut e, &mut g, &mut h);
        TwoSurfaces::pair(&x, 0, 2, &harmonic(C), &mut e, &mut g, &mut h);
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        if req.need_hessian {
            out.hessian_bohr = Some(h);
        }
        Ok(out)
    }
}

#[test]
fn the_network_of_two_states_is_drawn_again_on_another_surface() {
    // The first network: the two minima of PES1 and their transition state, the minimum of
    // PES2, and the two crossing points.
    let dir = scratch("remap");
    let c = config(&dir, &["-map_opt", "states=0,1;0,3", DIRECT[0], DIRECT[1]]);
    let first = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    check(&first, CrossingKind::Mecp, false);
    // The second run on the tilted surfaces: the minima of the first network as inputs, each
    // on its own state, and its transition state and crossing points as the jobs.
    let cfg = dir.join("second.json");
    std::fs::write(
        &cfg,
        r#"{
  "mapper_settings": {"rng_seed": 1, "channels": false},
  "step1_settings": {"othersoft": "two_surfaces_tilted", "NSTEP": 300},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]
  }
}"#,
    )
    .unwrap();
    let mut both = registry();
    both.register("two_surfaces_tilted", |_| Ok(Box::new(Tilted)));
    let remap = |output: &str, states: &str| {
        let args: Vec<String> = [
            "-remap",
            &first.directory.display().to_string(),
            "-cfg",
            &cfg.display().to_string(),
            "--output_dir",
            output,
            "-map_opt",
            states,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
        MapperConfig::from_args(&parsed, None)
            .and_then(|c| run_with_parts(&c, &dir, None, &both, &MapperParts::default(), &mut []))
    };
    let second = remap("second", "states=0,1;0,3").unwrap();
    let net = &second.network;
    let s = summary(net);
    assert_eq!(net.states, first.network.states, "{s}");
    // The same kinds of nodes, each on the state it was on.
    let count = |n: &Network, f: &dyn Fn(&Node) -> bool| n.nodes.iter().filter(|x| f(x)).count();
    let pes = |k: usize| move |n: &Node| n.kind == NodeKind::Minimum && n.state == net.states[k];
    assert_eq!(count(net, &pes(0)), 2, "{s}");
    assert_eq!(count(net, &pes(1)), 1, "{s}");
    assert_eq!(net.of_kind(&NodeKind::TransitionState).count(), 1, "{s}");
    let crossings: Vec<&Node> = net
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Crossing { .. }))
        .collect();
    assert_eq!(crossings.len(), 2, "{s}");
    assert_eq!(net.crossing_edges.len(), 2, "{s}");
    for e in &net.crossing_edges {
        assert!(e.ends.iter().all(Option::is_some), "{s}");
    }
    // Three jobs: the transition state and the two crossing points of the first network.
    assert_eq!(net.jobs.len(), 3, "{s}");
    assert!(net.jobs.iter().all(|j| j.task.starts_with("remap:")), "{s}");
    // The seam is that of the first surfaces (the term is the same on both), and its minima
    // have moved along it: each crossing point of the second network is on the seam, near
    // one of the first, at another r12 and a higher energy.
    for x in &crossings {
        let r12 = distance(x, 0, 1);
        let r23 = distance(x, 1, 2);
        // (With the same K on both surfaces the seam has one r23 at every r12.)
        let seam = 0.5 * (AS[0] + AS[1])
            - (curve(0, r12).0 - curve(1, r12).0) / (2.0 * K * (AS[1] - AS[0]));
        assert!(
            (r23 - seam).abs() < 1e-3,
            "{}: {r12} {r23}
{s}",
            x.label()
        );
        let old = first
            .network
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Crossing { .. }))
            .min_by(|a, b| {
                (distance(a, 0, 1) - r12)
                    .abs()
                    .total_cmp(&(distance(b, 0, 1) - r12).abs())
            })
            .unwrap();
        let moved = (distance(old, 0, 1) - r12).abs();
        assert!(
            moved > 1e-3 && moved < 0.3,
            "{}: moved {moved}\n{s}",
            x.label()
        );
        assert!(x.energy > old.energy, "{s}");
    }
    // The table of the run.
    let text = std::fs::read_to_string(second.directory.join("remap.csv")).unwrap();
    let rows: Vec<Vec<&str>> = text
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    assert_eq!(rows.len(), 3 + 3, "{text}");
    assert_eq!(
        rows.iter()
            .filter(|r| r[1] == "EQ" && r[6] == "minimized")
            .count(),
        3
    );
    assert_eq!(
        rows.iter()
            .filter(|r| r[1] == "TS" && r[6] == "added")
            .count(),
        1,
        "{text}"
    );
    assert_eq!(
        rows.iter()
            .filter(|r| r[1] == "crossing" && r[6] == "added")
            .count(),
        2,
        "{text}"
    );
    // Every row has the node of its kind nearest to it, a few tenths of an Angstrom away at
    // most.
    for r in &rows {
        let d: f64 = r[9].parse().unwrap_or_else(|_| panic!("{r:?}"));
        assert!(d < 0.3, "{r:?}");
    }
    // A network of two states is not drawn again on one.
    let e = remap("one_state", "channels=false")
        .unwrap_err()
        .to_string();
    assert!(e.contains("2 state(s) and this run 0"), "{e}");
}

/// [`Tilted`] that writes the file `stop` at its evaluation `at` (counted from 0).
struct TiltedStopAt {
    count: Arc<std::sync::atomic::AtomicUsize>,
    at: usize,
    stop: PathBuf,
}

impl Calculator for TiltedStopAt {
    fn name(&self) -> &str {
        "two_surfaces_tilted"
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == self.at {
            std::fs::write(&self.stop, "").unwrap();
        }
        Tilted.evaluate(sys, req)
    }
}

#[test]
fn the_energies_of_the_network_of_two_states_are_computed_on_another_surface() {
    let dir = scratch("points");
    let c = config(&dir, &["-map_opt", "states=0,1;0,3", DIRECT[0], DIRECT[1]]);
    let first = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    check(&first, CrossingKind::Mecp, false);
    let cfg = dir.join("second.json");
    std::fs::write(
        &cfg,
        r#"{
  "mapper_settings": {"rng_seed": 1, "channels": false},
  "step1_settings": {"othersoft": "two_surfaces_tilted", "NSTEP": 300},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]
  }
}"#,
    )
    .unwrap();
    let mut both = registry();
    both.register("two_surfaces_tilted", |_| Ok(Box::new(Tilted)));
    let settings = |output: &str| {
        let args: Vec<String> = [
            "-single_points",
            &first.directory.display().to_string(),
            "-cfg",
            &cfg.display().to_string(),
            "--output_dir",
            output,
            "-map_opt",
            "states=0,1;0,3",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
        MapperConfig::from_args(&parsed, None).unwrap()
    };
    let second = run_with_parts(
        &settings("points"),
        &dir,
        None,
        &both,
        &MapperParts::default(),
        &mut [],
    )
    .unwrap();
    assert!(!second.interrupted);
    let (old, new) = (&first.network, &second.network);
    let s = summary(new);
    assert_eq!(new.nodes.len(), old.nodes.len(), "{s}");
    assert_eq!(new.states, old.states, "{s}");
    // Every node at its structure, with the energy of the tilted surfaces there: the term is
    // the same on both states, so a minimum and a transition state have the first energy plus
    // the term, and a crossing point the mean of its two states plus the term, with the gap it
    // had.
    let tilt = |n: &Node| W * (distance(n, 0, 1) - RW).powi(2);
    for (a, b) in old.nodes.iter().zip(&new.nodes) {
        assert_eq!((a.kind, a.state, a.label()), (b.kind, b.state, b.label()));
        let (p, q) = (a.system.flat(), b.system.flat());
        assert!(p.iter().zip(q.iter()).all(|(x, y)| x == y), "{}", a.label());
        assert!(
            (b.energy - a.energy - tilt(a)).abs() < 1e-9,
            "{}: {} against {} + {}\n{s}",
            a.label(),
            b.energy,
            a.energy,
            tilt(a)
        );
    }
    // The crossing edges: the same ends, the barriers of the new energies.
    assert_eq!(new.crossing_edges.len(), 2, "{s}");
    for (a, b) in old.crossing_edges.iter().zip(&new.crossing_edges) {
        assert_eq!((a.crossing, a.ends), (b.crossing, b.ends));
        for k in 0..2 {
            let end = b.ends[k].unwrap();
            let barrier = new.nodes[b.crossing].energy - new.nodes[end].energy;
            assert!((b.barriers[k].unwrap() - barrier).abs() < 1e-12, "{s}");
        }
    }
    // Three minima and a transition state: one energy each; two crossing points: two each.
    assert_eq!(new.evaluations.energies, 4 + 2 * 2, "{s}");
    assert_eq!(new.evaluations.gradients, 0, "{s}");
    // The table has the gap of the two states at every crossing point (that of the first
    // surfaces, since the term is the same on both: within the limit of a crossing point).
    let table = std::fs::read_to_string(second.directory.join("single_points.csv")).unwrap();
    let gaps: Vec<f64> = table
        .lines()
        .skip(1)
        .filter(|l| l.contains(",MECP,"))
        .map(|l| l.rsplit(',').next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(gaps.len(), 2, "{table}");
    assert!(gaps.iter().all(|g| g.abs() < 1e-3), "{gaps:?}");
    assert_eq!(table.lines().count(), 1 + old.nodes.len(), "{table}");

    // A run stopped on the way writes no network; continued, it gives the same one, with the
    // energies it had computed taken from the journal.
    let mut interrupted = 0;
    for at in [1, 4, 6] {
        let output = format!("stopped_{at}");
        let run_dir = dir.join(&output);
        let mut stopping = registry();
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (c, stop) = (Arc::clone(&count), run_dir.join("STOP"));
        stopping.register("two_surfaces_tilted", move |_| {
            Ok(Box::new(TiltedStopAt {
                count: Arc::clone(&c),
                at,
                stop: stop.clone(),
            }) as Box<dyn Calculator>)
        });
        let stopped = run_with_parts(
            &settings(&output),
            &dir,
            None,
            &stopping,
            &MapperParts::default(),
            &mut [],
        )
        .unwrap();
        if !stopped.interrupted {
            continue;
        }
        interrupted += 1;
        assert!(!run_dir.join("reaction_network.json").exists(), "{at}");
        assert!(!run_dir.join("single_points.csv").exists(), "{at}");
        let read = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
        let again = MapperConfig::from_args(&read, Some(&run_dir)).unwrap();
        let done =
            run_with_parts(&again, &dir, None, &both, &MapperParts::default(), &mut []).unwrap();
        assert!(!done.interrupted, "{at}");
        for (a, b) in done.network.nodes.iter().zip(&new.nodes) {
            assert!((a.energy - b.energy).abs() < 1e-12, "{at}: {}", a.label());
        }
        let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
        assert!(
            log.contains("evaluations taken from evaluations.journal"),
            "{at}: {log}"
        );
    }
    assert!(
        interrupted >= 2,
        "only {interrupted} of the stops came during the run"
    );
}

/// [`Tilted`] that fails on PES1 beyond `r12 = 5`: the second minimum of PES1 and everything
/// that leads to it cannot be computed on this surface.
struct Walled;

impl Calculator for Walled {
    fn name(&self) -> &str {
        "two_surfaces_walled"
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let x = sys.coords_flat_bohr();
        let r12 = (0..3)
            .map(|k| (x[3 + k] - x[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        if surface(sys.spin_charge.multiplicity, req.excited_state) == 0 && r12 > 5.0 {
            return Err(mopt_rs::error::MoptError::calc(
                "the model has no PES1 beyond r12 = 5".to_string(),
            ));
        }
        Tilted.evaluate(sys, req)
    }
}

#[test]
fn what_does_not_come_through_a_remap_is_left_out() {
    // The first network: the minima of PES1 at R1 and R2 with their transition state, the
    // minimum of PES2, and the two crossing points.
    let dir = scratch("remap_thinned");
    let c = config(&dir, &["-map_opt", "states=0,1;0,3", DIRECT[0], DIRECT[1]]);
    let first = run_with_parts(&c, &dir, None, &registry(), &parts(), &mut []).unwrap();
    check(&first, CrossingKind::Mecp, false);
    // The second surface fails on PES1 beyond r12 = 5: the minimum at R2 cannot be minimized,
    // the chain of the transition state runs into the wall, and the crossing point between
    // PES2 and R2 cannot be followed down on PES1. The run goes on without them.
    let cfg = dir.join("walled.json");
    std::fs::write(
        &cfg,
        r#"{
  "mapper_settings": {"rng_seed": 1, "channels": false},
  "step1_settings": {"othersoft": "two_surfaces_walled", "NSTEP": 300},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]
  }
}"#,
    )
    .unwrap();
    let mut walled = registry();
    walled.register("two_surfaces_walled", |_| Ok(Box::new(Walled)));
    let args: Vec<String> = [
        "-remap",
        &first.directory.display().to_string(),
        "-cfg",
        &cfg.display().to_string(),
        "--output_dir",
        "walled",
        "-map_opt",
        "states=0,1;0,3",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    let c = MapperConfig::from_args(&parsed, None).unwrap();
    let second = run_with_parts(&c, &dir, None, &walled, &MapperParts::default(), &mut []).unwrap();
    assert!(!second.interrupted, "{:?}", second.reason);
    let net = &second.network;
    let s = summary(net);
    // What came through: the minimum of PES1 at R1, the minimum of PES2, and the crossing
    // point between them. No node lies beyond the wall.
    assert_eq!(net.minima().count(), 2, "{s}");
    for n in &net.nodes {
        let beyond = n.state == net.states[0] && distance(n, 0, 1) > 5.0;
        assert!(!beyond, "{} beyond the wall\n{s}", n.label());
    }
    assert_eq!(net.of_kind(&NodeKind::TransitionState).count(), 0, "{s}");
    let crossings = net
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Crossing { .. }))
        .count();
    assert!(crossings >= 1, "{s}");
    assert!(
        net.crossing_edges
            .iter()
            .any(|e| e.ends.iter().all(Option::is_some)),
        "{s}"
    );
    // Every item had its job (the items of the minimum that was left out run from another
    // minimum), and the table says what was left out.
    assert_eq!(net.jobs.len(), 3, "{s}");
    let text = std::fs::read_to_string(second.directory.join("remap.csv")).unwrap();
    let rows: Vec<Vec<&str>> = text
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    let count = |kind: &str, status: &str| {
        rows.iter()
            .filter(|r| r[1] == kind && r[6] == status)
            .count()
    };
    assert_eq!(count("EQ", "minimized"), 2, "{text}");
    assert_eq!(count("EQ", "left out"), 1, "{text}");
    assert_eq!(count("TS", "failed") + count("TS", "left out"), 1, "{text}");
    assert_eq!(count("crossing", "added"), 1, "{text}");
    assert_eq!(
        count("crossing", "failed") + count("crossing", "left out"),
        1,
        "{text}"
    );
    assert!(
        net.warnings
            .iter()
            .any(|w| w.job.is_none() && w.message.contains("is left out")),
        "{s}"
    );
    let log = std::fs::read_to_string(second.directory.join("mopt.log")).unwrap();
    assert!(log.contains("is left out"), "{log}");
}
