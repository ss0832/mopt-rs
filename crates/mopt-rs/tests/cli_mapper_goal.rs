//! `run_mapper` towards a goal structure (`goal=FILE`): the goal-directed search
//! (`priority=rrt`, after Nakao, Harabuchi, Maeda, Tsuda 2022) and the missing connections
//! (`method=connect`, after Carr, Trygubenko, Wales 2005), on surfaces where the answer is
//! known:
//! - the Muller-Brown surface (`mb2d`): its lowest minimum A (-0.558, 1.442) and the minimum
//!   B (0.623, 0.028) are joined only through the minimum C (-0.050, 0.467) and its two saddle
//!   points (-0.822, 0.624) and (0.212, 0.293);
//! - the Lennard-Jones cluster of seven argon atoms (`lj`): its lowest minimum, the pentagonal
//!   bipyramid (-16.505 epsilon), and the capped octahedron (-15.935 epsilon).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mopt_rs::cli::mapper::{run_with, MapperConfig, MapperRun};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::settings::read_settings;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Units};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::mapper::{Network, NodeKind};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mapper_goal_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn mb_config(jobs: usize) -> String {
    format!(
        r#"{{
  "mapper_settings": {{"max_iterations": {jobs}, "rng_seed": 1}},
  "step1_settings": {{"othersoft": "mb2d", "NSTEP": 300}},
  "step3_settings": {{"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300}},
  "step4_settings": {{
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"]
  }}
}}"#
    )
}

const LJ_CONFIG: &str = r#"{
  "mapper_settings": {
    "afir_gamma_kJmol": 2.0,
    "dist_upper_ang": 7.0,
    "max_iterations": 30,
    "rng_seed": 42,
    "channels": false
  },
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

fn configure(
    dir: &Path,
    inputs: &[PathBuf],
    settings: &str,
    extra: &[&str],
) -> MoptResult<MapperConfig> {
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, settings).unwrap();
    let mut args: Vec<String> = inputs.iter().map(|p| p.display().to_string()).collect();
    args.extend(["-cfg".to_string(), cfg.display().to_string()]);
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args)?;
    MapperConfig::from_args(&parsed, None)
}

/// The Muller-Brown inputs: the start at A and the goal at B.
fn mb_files(dir: &Path) -> (PathBuf, PathBuf) {
    let (start, goal) = (dir.join("a.txt"), dir.join("b.txt"));
    std::fs::write(&start, "x -0.55\ny 1.44\n").unwrap();
    std::fs::write(&goal, "x 0.62\ny 0.03\n").unwrap();
    (start, goal)
}

fn near(net: &Network, kind: &NodeKind, p: &[f64; 2]) -> Option<usize> {
    net.of_kind(kind)
        .find(|n| {
            let x = n.system.flat();
            ((x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2)).sqrt() < 1e-3
        })
        .map(|n| n.id)
}

fn summary(net: &Network) -> String {
    net.nodes
        .iter()
        .map(|n| {
            format!(
                "{} {:?} {:.6} {}",
                n.label(),
                n.system.flat().iter().copied().collect::<Vec<_>>(),
                n.energy,
                n.excluded.as_deref().unwrap_or("")
            )
        })
        .chain(
            net.jobs
                .iter()
                .map(|j| format!("job {}: {} {}", j.id, j.task, j.description)),
        )
        .chain(
            net.warnings
                .iter()
                .map(|w| format!("warning: {}", w.message)),
        )
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn missing_connections_join_the_far_minima_of_muller_brown() {
    let dir = scratch("mb_connect");
    let (start, goal) = mb_files(&dir);
    let goal_arg = format!("goal={}", goal.display());
    let c = configure(
        &dir,
        &[start],
        &mb_config(10),
        &["-map_opt", &goal_arg, "method=connect"],
    )
    .unwrap();
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    let net = &run.network;
    assert!(
        run.reason.as_deref().is_some_and(|r| r.contains("goal")),
        "{:?}\n{}",
        run.reason,
        summary(net)
    );
    assert_eq!(net.goal_joined(), Some(true));
    // The goal is B, kept and never explored; the way to it runs through C and both saddles.
    let b = near(net, &NodeKind::Minimum, &MB_MINIMA[1]).unwrap();
    assert_eq!(net.goal_node(), Some(b));
    assert!(net.nodes[b].excluded.is_some());
    assert!(net.jobs.iter().all(|j| j.node != b));
    assert!(
        near(net, &NodeKind::Minimum, &MB_MINIMA[2]).is_some(),
        "{}",
        summary(net)
    );
    for s in &MB_SADDLES {
        assert!(
            near(net, &NodeKind::TransitionState, s).is_some(),
            "{s:?}\n{}",
            summary(net)
        );
    }
    assert!(net.jobs.iter().all(|j| j.task.starts_with("connect:")));
    let doc = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
    assert!(doc.contains("\"goal\": true"), "the goal input is marked");
}

#[test]
fn the_goal_directed_search_reaches_b_on_muller_brown() {
    let dir = scratch("mb_rrt");
    let (start, goal) = mb_files(&dir);
    let goal_arg = format!("goal={}", goal.display());
    // The energies of mb2d are its own numbers (minima at -146.7, -108.2, -80.8): the
    // acceptance filter runs at a temperature where its barriers of about 40 pass.
    let c = configure(
        &dir,
        &[start],
        &mb_config(16),
        &["-map_opt", &goal_arg, "accept_temperature=1e9"],
    )
    .unwrap();
    assert_eq!(c.settings.priority, "rrt");
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    let net = &run.network;
    assert_eq!(
        net.goal_joined(),
        Some(true),
        "{:?}\n{}",
        run.reason,
        summary(net)
    );
    assert!(run.reason.as_deref().is_some_and(|r| r.contains("goal")));
}

#[test]
fn the_settings_of_the_goal_are_checked() {
    let dir = scratch("refused");
    let (start, goal) = mb_files(&dir);
    let goal_arg = format!("goal={}", goal.display());
    let config = mb_config(2);
    for args in [
        vec!["-map_opt", "priority=rrt"],
        vec!["-map_opt", "method=connect"],
        vec!["-map_opt", "rrt_goal_weight=2"],
        vec!["-map_opt", &goal_arg, "rrt_nope=1"],
        vec!["-map_opt", &goal_arg, "accept_temperature=-1"],
        vec!["-map_opt", &goal_arg, "method=connect", "method_path=addf"],
    ] {
        let refused = configure(&dir, std::slice::from_ref(&start), &config, &args)
            .and_then(|c| run_with(&c, &dir, None, &CalculatorRegistry::native()).map(|_| ()));
        assert!(refused.is_err(), "{args:?}");
    }
}

/// LJ7 near its lowest minimum (Bohr).
const AR7: [[f64; 3]; 7] = [
    [6.023853759546511, 0.0, -0.15549746403767273],
    [1.962006769746685, 5.809518090796886, -0.061011157764781325],
    [-4.854674003698997, 3.590479638369873, 0.03347514850811009],
    [-4.999838497421851, -3.5904796383698727, 0.12796145478100152],
    [1.6599768546465556, -5.809518090796886, 0.2224477610538929],
    [-0.08463505395523435, 0.0, 3.794030138169188],
    [0.2933101711363313, 0.0, -3.961405880709738],
];

/// A capped octahedron of seven argon atoms (Bohr), near the second minimum.
fn capped_octahedron() -> Vec<[f64; 3]> {
    let r = 2.70 * Units::ANGSTROM_TO_BOHR;
    vec![
        [r, 0.0, 0.0],
        [-r, 0.0, 0.0],
        [0.0, r, 0.0],
        [0.0, -r, 0.0],
        [0.0, 0.0, r],
        [0.0, 0.0, -r],
        [r, r, r],
    ]
}

fn write_xyz(path: &Path, atoms: &[[f64; 3]]) {
    let mut text = format!("{}\n0 1\n", atoms.len());
    for p in atoms {
        text.push_str(&format!(
            "Ar {:.10} {:.10} {:.10}\n",
            p[0] * Units::BOHR_TO_ANGSTROM,
            p[1] * Units::BOHR_TO_ANGSTROM,
            p[2] * Units::BOHR_TO_ANGSTROM
        ));
    }
    std::fs::write(path, text).unwrap();
}

fn epsilon() -> f64 {
    0.237 / Units::HARTREE_TO_KCAL_MOL
}

fn lj_files(dir: &Path) -> (PathBuf, PathBuf) {
    let (start, goal) = (dir.join("ar7.xyz"), dir.join("goal.xyz"));
    write_xyz(&start, &AR7);
    write_xyz(&goal, &capped_octahedron());
    (start, goal)
}

/// `lj` that writes the file `stop` at its evaluation `at` (counted from 0).
struct StopAt {
    count: Arc<AtomicUsize>,
    at: usize,
    stop: PathBuf,
}

impl Calculator for StopAt {
    fn name(&self) -> &str {
        "lj"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if self.count.fetch_add(1, Ordering::SeqCst) == self.at {
            std::fs::write(&self.stop, "").unwrap();
        }
        LennardJones.evaluate(sys, req)
    }
}

/// The same network: nodes, edges, and jobs.
fn same_network(a: &Network, b: &Network, what: &str) {
    assert_eq!(a.jobs, b.jobs, "{what}");
    assert_eq!(a.nodes.len(), b.nodes.len(), "{what}");
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert_eq!(
            (x.kind, x.origin),
            (y.kind, y.origin),
            "{what}: {}",
            x.label()
        );
        assert_eq!(x.excluded, y.excluded, "{what}: {}", x.label());
        assert!((x.energy - y.energy).abs() < 1e-10, "{what}: {}", x.label());
    }
    assert_eq!(a.ts_edges.len(), b.ts_edges.len(), "{what}");
    for (x, y) in a.ts_edges.iter().zip(&b.ts_edges) {
        assert_eq!((x.ts, x.ends, x.job), (y.ts, y.ends, y.job), "{what}");
    }
}

/// A run with the settings `extra` (`-map_opt` items) on LJ7 towards the capped octahedron, and the same run stopped at the
/// evaluations `stops` and restarted, which must give the same network.
fn lj_run_and_restarts(name: &str, extra: &[&str], stops: &[usize]) -> MapperRun {
    let dir = scratch(name);
    let (start, goal) = lj_files(&dir);
    let goal_arg = format!("goal={}", goal.display());
    let mut args = vec!["--output_dir", "whole", "-map_opt", goal_arg.as_str()];
    args.extend_from_slice(extra);
    let native = CalculatorRegistry::native();
    let whole = run_with(
        &configure(&dir, std::slice::from_ref(&start), LJ_CONFIG, &args).unwrap(),
        &dir,
        None,
        &native,
    )
    .unwrap();
    for &at in stops {
        let out = format!("stopped_{at}");
        let run_dir = dir.join(&out);
        let mut stopping = CalculatorRegistry::native();
        let count = Arc::new(AtomicUsize::new(0));
        let (c, stop) = (Arc::clone(&count), run_dir.join("STOP"));
        stopping.register("lj", move |_| {
            Ok(Box::new(StopAt {
                count: Arc::clone(&c),
                at,
                stop: stop.clone(),
            }) as Box<dyn Calculator>)
        });
        let mut args = vec!["--output_dir", out.as_str(), "-map_opt", goal_arg.as_str()];
        args.extend_from_slice(extra);
        let stopped = run_with(
            &configure(&dir, std::slice::from_ref(&start), LJ_CONFIG, &args).unwrap(),
            &dir,
            None,
            &stopping,
        )
        .unwrap();
        if !stopped.interrupted {
            // The whole run needed fewer evaluations than this stop.
            continue;
        }
        let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
        let again = MapperConfig::from_args(&settings, Some(&run_dir)).unwrap();
        let done = run_with(&again, &dir, None, &native).unwrap();
        assert!(!done.interrupted);
        same_network(
            &done.network,
            &whole.network,
            &format!("{name}, stopped at {at}"),
        );
    }
    whole
}

#[test]
fn missing_connections_join_the_capped_octahedron_of_lj7_and_restart() {
    let run = lj_run_and_restarts("lj_connect", &["method=connect"], &[30, 400]);
    let net = &run.network;
    assert_eq!(
        net.goal_joined(),
        Some(true),
        "{:?}\n{}",
        run.reason,
        summary(net)
    );
    let goal = net.goal_node().unwrap();
    let e = net.nodes[goal].energy / epsilon();
    assert!((e - -15.935).abs() < 1e-3, "{e}");
    println!(
        "lj7 connect: {} job(s), {} gradients, {} Hessians",
        net.jobs.len(),
        net.evaluations.gradients,
        net.evaluations.hessians
    );
}

#[test]
fn the_goal_directed_search_reaches_the_capped_octahedron_of_lj7_and_restarts() {
    let run = lj_run_and_restarts("lj_rrt", &[], &[60, 900, 2500]);
    let net = &run.network;
    assert_eq!(
        net.goal_joined(),
        Some(true),
        "{:?}\n{}",
        run.reason,
        summary(net)
    );
    assert!(run.reason.as_deref().is_some_and(|r| r.contains("goal")));
    println!(
        "lj7 rrt: {} job(s), {} gradients, {} Hessians",
        net.jobs.len(),
        net.evaluations.gradients,
        net.evaluations.hessians
    );
}

#[test]
fn the_acceptance_filter_keeps_but_does_not_explore() {
    // At 50 K (kT = 0.10 kcal/mol) the barriers of LJ7 (0.1 to 0.5 kcal/mol) let few pass.
    let dir = scratch("lj_accept");
    let (start, _) = lj_files(&dir);
    let c = configure(
        &dir,
        &[start],
        LJ_CONFIG,
        &["-map_opt", "accept_temperature=50", "--max_iter", "12"],
    )
    .unwrap();
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    let net = &run.network;
    let refused: Vec<usize> = net
        .minima()
        .filter(|n| {
            n.excluded
                .as_deref()
                .is_some_and(|r| r.contains("acceptance filter"))
        })
        .map(|n| n.id)
        .collect();
    assert!(!refused.is_empty(), "{}", summary(net));
    for id in &refused {
        assert!(
            net.jobs.iter().all(|j| j.node != *id),
            "EQ{id} was explored"
        );
    }
}

/// The chain surface (`expr`): seven Gaussian wells along y = 0 (x = 0 to 6, depth 0.05
/// Hartree, barriers of about 0.02 Hartree between them) and a deeper dead-end valley of four
/// wells (depth 0.08) branching off the second, with a weak confinement. Its stationary points
/// (Newton from a grid): 11 minima, 11 transition states, one index-2 saddle point.
fn chain_expression() -> String {
    let mut e = "0.002*((q1-3)^2+(q2-1)^2)".to_string();
    let wells: Vec<(f64, f64, f64)> = (0..7)
        .map(|k| (f64::from(k), 0.0, 0.05))
        .chain([(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0)].map(|(x, y)| (x, y, 0.08)))
        .collect();
    for (x, y, d) in wells {
        e.push_str(&format!("-{d}*exp(-((q1-{x:.1})^2+(q2-{y:.1})^2)/0.1800)"));
    }
    e
}

/// The minima of the chain at y = 0 (x, from the grid search) and its transition states.
const CHAIN_MINIMA: [f64; 7] = [
    0.026589, 1.015423, 2.007730, 3.0, 3.992245, 4.984470, 5.973446,
];
const CHAIN_SADDLES: [f64; 6] = [0.478447, 1.488805, 2.495902, 3.504098, 4.512304, 5.520551];

fn chain_run(name: &str, extra: &[&str]) -> MapperRun {
    let dir = scratch(name);
    let (start, goal) = (dir.join("start.txt"), dir.join("goal.txt"));
    std::fs::write(&start, "0.0\n0.0\n").unwrap();
    std::fs::write(&goal, "6.0\n0.0\n").unwrap();
    let config = format!(
        r#"{{
  "mapper_settings": {{"max_iterations": 60, "rng_seed": 1}},
  "step1_settings": {{"othersoft": "expr:{}", "NSTEP": 300}},
  "step3_settings": {{"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300}},
  "step4_settings": {{
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"]
  }}
}}"#,
        chain_expression()
    );
    let goal_arg = format!("goal={}", goal.display());
    let mut args = vec!["-map_opt", goal_arg.as_str()];
    args.extend_from_slice(extra);
    let c = configure(&dir, &[start], &config, &args).unwrap();
    run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap()
}

/// Every minimum and transition state of the chain is a node, and the goal is joined.
fn crossed(run: &MapperRun) {
    let net = &run.network;
    assert_eq!(
        net.goal_joined(),
        Some(true),
        "{:?}\n{}",
        run.reason,
        summary(net)
    );
    let on_chain = |kind: &NodeKind, x: f64| {
        net.of_kind(kind).any(|n| {
            let p = n.system.flat();
            (p[0] - x).abs() < 1e-3 && p[1].abs() < 0.02
        })
    };
    for x in CHAIN_MINIMA {
        assert!(
            on_chain(&NodeKind::Minimum, x),
            "EQ at {x}\n{}",
            summary(net)
        );
    }
    for x in CHAIN_SADDLES {
        assert!(
            on_chain(&NodeKind::TransitionState, x),
            "TS at {x}\n{}",
            summary(net)
        );
    }
}

#[test]
fn the_goal_directed_search_crosses_the_chain() {
    let run = chain_run("chain_rrt", &[]);
    crossed(&run);
    println!(
        "chain rrt: {} job(s), {} gradients",
        run.network.jobs.len(),
        run.network.evaluations.gradients
    );
}

#[test]
fn missing_connections_cross_the_chain() {
    let run = chain_run("chain_connect", &["method=connect"]);
    crossed(&run);
    println!(
        "chain connect: {} job(s), {} gradients",
        run.network.jobs.len(),
        run.network.evaluations.gradients
    );
}
