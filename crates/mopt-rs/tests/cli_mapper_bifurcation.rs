//! `run_mapper` on a surface whose reaction bifurcates: a TS–TS link at the end of an IRC side,
//! and the possible bifurcation the exact Hessians along the IRC show.
//!
//! The model: three hydrogen atoms, the first and the third held by stiff springs (so that the
//! two products are not mirror images of each other), the second on
//! `V = a (X^2 - 1)^2 + b (1/2 - X) Y^2 + c Y^4 + k Z^2` (Bohr, Hartree). The reactant lies at
//! `X = -1`, TS1 at `X = 0`; past `X = 1/2` the valley along `Y = 0` turns into a ridge (a
//! valley-ridge inflection), which runs down into TS2 at `X = 1`, the saddle between the two
//! products at `Y = +-1.6`. By symmetry the IRC of TS1 runs along the ridge into TS2.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::{Col, Mat};
use mopt_rs::cli::mapper::{run_with_parts, MapperConfig, MapperParts, MapperRun};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity, Units};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::mapper::{
    follow_guesses, ExplorationMethod, Guess, Job, JobContext, JobOutcome, Network, Node, NodeKind,
    Task,
};

const A: f64 = 0.02;
const B: f64 = 0.04;
const C: f64 = 0.005;
const KZ: f64 = 0.05;
/// The springs of the first and the third atom (Hartree / Bohr^2) and where they hold them.
const SPRING: f64 = 0.5;
const ANCHORS: [[f64; 3]; 2] = [[-5.0, 0.0, 0.0], [-5.0, 3.0, 0.0]];

struct Bifurcating;

impl Calculator for Bifurcating {
    fn name(&self) -> &str {
        "bifurcating"
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let x = sys.coords_flat_bohr();
        let mut g = Col::<f64>::zeros(9);
        let mut h = Mat::<f64>::zeros(9, 9);
        let mut e = 0.0;
        for (atom, anchor) in [0, 2].into_iter().zip(ANCHORS) {
            for k in 0..3 {
                let d = x[3 * atom + k] - anchor[k];
                e += SPRING * d * d;
                g[3 * atom + k] = 2.0 * SPRING * d;
                h[(3 * atom + k, 3 * atom + k)] = 2.0 * SPRING;
            }
        }
        let (px, py, pz) = (x[3], x[4], x[5]);
        e += A * (px * px - 1.0).powi(2) + B * (0.5 - px) * py * py + C * py.powi(4) + KZ * pz * pz;
        g[3] = 4.0 * A * px * (px * px - 1.0) - B * py * py;
        g[4] = 2.0 * B * (0.5 - px) * py + 4.0 * C * py.powi(3);
        g[5] = 2.0 * KZ * pz;
        h[(3, 3)] = 4.0 * A * (3.0 * px * px - 1.0);
        h[(3, 4)] = -2.0 * B * py;
        h[(4, 3)] = -2.0 * B * py;
        h[(4, 4)] = 2.0 * B * (0.5 - px) + 12.0 * C * py * py;
        h[(5, 5)] = 2.0 * KZ;
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        if req.need_hessian {
            out.hessian_bohr = Some(h);
        }
        Ok(out)
    }
}

/// One job from the input: the guess of TS1 near `X = 0`.
struct SeedTs;

impl ExplorationMethod for SeedTs {
    fn name(&self) -> &str {
        "seed_ts"
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        if node.id != 0 {
            return Ok(Vec::new());
        }
        Ok(vec![Task {
            key: "seed".to_string(),
            description: "the guess of TS1".to_string(),
            parameters: JsonValue::Null,
        }])
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let mut x = job.start.flat();
        x[3] = 0.05;
        x[4] = 0.0;
        x[5] = 0.0;
        let guess = Guess {
            system: job.start.with_flat(&x)?,
            origin: "the model".to_string(),
        };
        let mut outcome = JobOutcome::default();
        if !follow_guesses(job, ctx, &[guess], &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

const CONFIG: &str = r#"{
  "mapper_settings": {"max_iterations": 1, "rng_seed": 1},
  "step1_settings": {"othersoft": "bifurcating", "NSTEP": 100},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 100},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"]
  }
}"#;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_bifurcation_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(dir: &Path, extra: &[&str]) -> MapperRun {
    let input = dir.join("reactant.xyz");
    let b = Units::BOHR_TO_ANGSTROM;
    std::fs::write(
        &input,
        format!(
            "3\n0 2\nH {:.10} 0 0\nH {:.10} 0 0\nH {:.10} {:.10} 0\n",
            -5.0 * b,
            -b,
            -5.0 * b,
            3.0 * b
        ),
    )
    .unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, CONFIG).unwrap();
    let mut args = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "-map_opt".to_string(),
        "method=seed_ts".to_string(),
        "channels=off".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    let config = MapperConfig::from_args(&parsed, None).unwrap();
    let mut registry = CalculatorRegistry::native();
    registry.register("bifurcating", |_| Ok(Box::new(Bifurcating)));
    let mut parts = MapperParts::default();
    parts.methods.register(
        "seed_ts",
        Arc::new(|_| Ok(Box::new(SeedTs) as Box<dyn ExplorationMethod>)),
    );
    run_with_parts(&config, dir, None, &registry, &parts, &mut []).unwrap()
}

#[test]
fn an_endpoint_below_its_configured_threshold_is_saved_as_an_equilibrium() {
    let dir = scratch("weak_endpoint");
    let network = run(&dir, &["irc_end_threshold_cm1=5000"]).network;
    assert_eq!(network.ts_edges.len(), 1);
    assert!(network
        .warnings
        .iter()
        .any(|w| w.message.contains("threshold 5000 cm^-1")));
    assert!(network.minima().any(|node| node
        .frequencies
        .as_ref()
        .is_some_and(|modes| modes.iter().any(|&w| w < -50.0))));
}

#[test]
fn the_irc_of_ts1_ends_at_ts2_and_the_network_has_the_link() {
    let dir = scratch("link");
    let run = run(&dir, &[]);
    let net = &run.network;
    let messages: Vec<&str> = net.warnings.iter().map(|w| w.message.as_str()).collect();
    let ts: Vec<&Node> = net.of_kind(&NodeKind::TransitionState).collect();
    let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap_or_default();
    let excerpt: Vec<&str> = log
        .lines()
        .filter(|l| {
            l.contains("IRC") || l.contains("TS") || l.contains("side") || l.contains("Job")
        })
        .take(80)
        .collect();
    let path =
        std::fs::read_to_string(run.directory.join("paths/TS000000_irc.xyz")).unwrap_or_default();
    let lines: Vec<&str> = path.lines().collect();
    let trace: Vec<String> = lines
        .chunks(5)
        .filter(|c| c.len() == 5)
        .map(|c| format!("{} | {}", c[1], c[3]))
        .collect();
    assert_eq!(
        ts.len(),
        2,
        "{messages:?}\n{}\nPATH\n{}",
        excerpt.join("\n"),
        trace.join("\n")
    );
    assert_eq!(net.ts_edges.len(), 2, "{messages:?}");
    // The edge of TS1 joins the reactant and TS2, below it; the edge of TS2 its two products.
    let link = net
        .ts_edges
        .iter()
        .find(|e| {
            e.ends
                .iter()
                .any(|&n| net.nodes[n].kind == NodeKind::TransitionState)
        })
        .expect("an edge that ends at a transition state");
    let ts1 = &net.nodes[link.ts];
    let ts2 = link
        .ends
        .iter()
        .map(|&n| &net.nodes[n])
        .find(|n| n.kind == NodeKind::TransitionState)
        .unwrap();
    assert!(
        (ts1.energy - A).abs() < 1e-6 && ts2.energy.abs() < 1e-6,
        "{} {}",
        ts1.energy,
        ts2.energy
    );
    let reactant = link
        .ends
        .iter()
        .map(|&n| &net.nodes[n])
        .find(|n| n.kind == NodeKind::Minimum)
        .unwrap();
    assert_eq!(Some(reactant.id), net.inputs[0].node);
    let fork = net.ts_edges.iter().find(|e| e.ts == ts2.id).unwrap();
    let products: Vec<&Node> = fork.ends.iter().map(|&n| &net.nodes[n]).collect();
    assert!(products.iter().all(|n| n.kind == NodeKind::Minimum));
    assert_ne!(products[0].id, products[1].id);
    let y: Vec<f64> = products.iter().map(|n| n.system.flat()[4]).collect();
    assert!(y[0] * y[1] < 0.0 && (y[0] + y[1]).abs() < 1e-3, "{y:?}");
    assert!(
        messages.iter().any(|m| m.contains("TS–TS link")),
        "{messages:?}"
    );
    // Every transition state has its vibrations: 9 (the frame is fixed), one imaginary.
    for n in &ts {
        let w = n.frequencies.as_ref().expect("the start check of the IRC");
        assert_eq!(w.len(), 9, "{}: {w:?}", n.label());
        assert_eq!(
            w.iter().filter(|w| **w < 0.0).count(),
            1,
            "{}: {w:?}",
            n.label()
        );
    }
    // The Hessian is analytic, so the IRC computed it at every tenth point: the forward side of
    // TS1 shows the ridge before TS2 as a possible bifurcation.
    let possible = &link.irc.possible_bifurcations;
    assert_eq!(possible.len(), 1, "{possible:?}");
    assert!(possible[0].wavenumber < -100.0, "{possible:?}");
    assert!(
        messages.iter().any(|m| m.contains("may bifurcate")),
        "{messages:?}"
    );
    let doc = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
    assert!(doc.contains("possible_bifurcations"));
    // The curvature across the path is kept at every exact Hessian of the side (every tenth
    // point), and the arc length at which it changes sign: the valley of this surface turns
    // into a ridge at X = 1/2, half a Bohr of the hydrogen from TS1 (0.50 amu^(1/2) Bohr).
    let side = link
        .irc
        .sides
        .iter()
        .find(|s| s.name == possible[0].side)
        .unwrap();
    assert!(
        !side.transverse.is_empty() && side.transverse.len() < side.arc_lengths.len(),
        "{} of {}",
        side.transverse.len(),
        side.arc_lengths.len()
    );
    // (On the scale of the arc lengths of the side, which count from its first point, 0.05
    // from the transition state: 0.452.)
    let at = possible[0].inflection.unwrap();
    assert!((at - 0.452).abs() < 0.15, "{at}");
    eprintln!("the inflection of the model at 0.452: {at:.4} with a Hessian at every tenth point");
    assert!(doc.contains("transverse_cm1") && doc.contains("inflection_arc_length"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_shorter_interval_of_the_hessians_keeps_more_curvatures() {
    let dir = scratch("transverse");
    let run = run(&dir, &["irc_hessian_interval=2"]);
    let net = &run.network;
    // An exact Hessian at every second point of every IRC: every side has the curvature
    // across the path at half of its points.
    for e in &net.ts_edges {
        for s in &e.irc.sides {
            assert!(
                2 * s.transverse.len() + 3 >= s.arc_lengths.len() && !s.transverse.is_empty(),
                "{}: {} of {}",
                s.name,
                s.transverse.len(),
                s.arc_lengths.len()
            );
        }
    }
    let edge = net
        .ts_edges
        .iter()
        .find(|e| !e.irc.possible_bifurcations.is_empty())
        .unwrap();
    let b = &edge.irc.possible_bifurcations[0];
    let at = b.inflection.unwrap();
    eprintln!(
        "the inflection of the model at 0.452: {at:.4} with a Hessian at every second point \
         (the first negative curvature beyond the threshold at {:.3})",
        b.at
    );
    // (The arc lengths of a side count from its first point, 0.05 from the transition state:
    // the valley turns into a ridge 0.502 from TS1, at 0.452 on that scale.) Measured: 0.4025,
    // the same estimate as with a Hessian at every tenth point.
    assert!((at - 0.452).abs() < 0.06, "{at}");
    assert!(b.after.is_some_and(|a| a < at + 1e-9) && at < b.at + 1e-9);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn without_hessians_along_the_irc_the_link_is_found_but_no_ridge_is_seen() {
    let dir = scratch("no_hessians");
    let run = run(&dir, &["irc_hessians=off"]);
    let net = &run.network;
    assert!(net.ts_edges.iter().any(|e| e
        .ends
        .iter()
        .any(|&n| net.nodes[n].kind == NodeKind::TransitionState)));
    assert!(net
        .ts_edges
        .iter()
        .all(|e| e.irc.possible_bifurcations.is_empty()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_possible_bifurcation_is_confirmed_by_both_products() {
    let dir = scratch("confirm");
    let run = run(&dir, &["confirm_bifurcations=on"]);
    let net = &run.network;
    let messages: Vec<&str> = net.warnings.iter().map(|w| w.message.as_str()).collect();
    let link = net
        .ts_edges
        .iter()
        .find(|e| {
            e.ends
                .iter()
                .any(|&n| net.nodes[n].kind == NodeKind::TransitionState)
        })
        .expect("an edge that ends at a transition state");
    let ts2 = link
        .ends
        .into_iter()
        .find(|&n| net.nodes[n].kind == NodeKind::TransitionState)
        .unwrap();
    let mut products: Vec<usize> = net
        .ts_edges
        .iter()
        .find(|e| e.ts == ts2)
        .unwrap()
        .ends
        .to_vec();
    products.sort_unstable();
    // Both ways across the path at the ridge lead to the two products of TS2.
    let possible = &link.irc.possible_bifurcations;
    assert_eq!(possible.len(), 1, "{possible:?}");
    let mut confirmed = possible[0].confirmed_ends.clone();
    confirmed.sort_unstable();
    assert_eq!(confirmed, products, "{messages:?}");
    assert!(
        messages.iter().any(|m| m.contains("is confirmed")),
        "{messages:?}"
    );
    let doc = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
    assert!(doc.contains("confirmed_ends"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_confirmed_bifurcation_divides_the_flux_of_its_side() {
    use mopt_rs::workflow::kinetics::kinetic_network;
    use mopt_rs::workflow::mapper::configuration::Configuration;
    use mopt_rs::workflow::mapper::{
        ElectronicState, IrcPath, Level, Origin, PossibleBifurcation, TsEdge,
    };
    // Three minima and a transition state on one variable: the forward side of its IRC ends
    // at EQ1 and, across the path, also leads to EQ2.
    let names = Arc::new(vec!["q".to_string()]);
    let mut net = Network::of_variables(names.to_vec());
    let mut add = |kind: NodeKind, x: f64, e: f64| {
        let point = Configuration::variables(Col::from_fn(1, |_| x), Arc::clone(&names)).unwrap();
        let state = ElectronicState::of_point(&point, 0);
        net.add_node(Node::new(
            kind,
            Level::Exploration,
            point,
            e,
            state,
            Origin::default(),
        ))
        .unwrap()
    };
    let (eq0, eq1, eq2) = (
        add(NodeKind::Minimum, 0.0, 0.0),
        add(NodeKind::Minimum, 2.0, 0.0),
        add(NodeKind::Minimum, 3.0, 0.0),
    );
    let ts = add(NodeKind::TransitionState, 1.0, 0.01);
    net.add_ts_edge(TsEdge {
        id: 0,
        level: Level::Exploration,
        ts,
        ends: [eq0, eq1],
        barriers: [0.01, 0.01],
        imaginary_wavenumber: Some(-500.0),
        irc: IrcPath {
            sides: Vec::new(),
            file: None,
            possible_bifurcations: vec![PossibleBifurcation {
                side: "forward".to_string(),
                after: None,
                at: 0.5,
                wavenumber: -200.0,
                confirmed_ends: vec![eq1, eq2],
                inflection: None,
            }],
        },
        job: None,
    })
    .unwrap();
    let k = kinetic_network(&net, Level::Exploration);
    assert_eq!(k.bottlenecks.len(), 2, "{:?}", k.notes);
    for b in &k.bottlenecks {
        assert_eq!(b.share, Some(0.5));
        let ids: Vec<usize> = b.states.iter().map(|&i| k.states[i].id).collect();
        assert!(ids.contains(&eq0), "{ids:?}");
    }
    assert!(
        k.notes.iter().any(|n| n.contains("bifurcates")),
        "{:?}",
        k.notes
    );
}
