//! The exploration methods `addf` and `nanoreactor` of `run_mapper` on the Lennard-Jones
//! cluster of seven argon atoms (built-in `lj`; minima -16.505, -15.935, -15.593, -15.533
//! epsilon): each finds transition states out of the lowest minimum, their IRCs end at minima
//! of the cluster, the files of the searches are in the directories of the jobs, and a stopped
//! run continues to the same network.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mopt_rs::cli::mapper::{run_with, MapperConfig};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::settings::read_settings;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Units};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::mapper::{Network, NodeKind};

fn epsilon() -> f64 {
    0.237 / Units::HARTREE_TO_KCAL_MOL
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_methods_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
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

fn write_input(dir: &Path) -> PathBuf {
    let path = dir.join("ar7.xyz");
    let mut text = format!("{}\n0 1\n", AR7.len());
    for p in AR7 {
        text.push_str(&format!(
            "Ar {:.10} {:.10} {:.10}\n",
            p[0] * Units::BOHR_TO_ANGSTROM,
            p[1] * Units::BOHR_TO_ANGSTROM,
            p[2] * Units::BOHR_TO_ANGSTROM
        ));
    }
    std::fs::write(&path, text).unwrap();
    path
}

const CONFIG: &str = r#"{
  "mapper_settings": {"max_iterations": 12, "rng_seed": 42},
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

fn config(dir: &Path, extra: &[&str]) -> MapperConfig {
    let cfg = dir.join("config.json");
    if !cfg.exists() {
        std::fs::write(&cfg, CONFIG).unwrap();
    }
    let input = write_input(dir);
    let mut args = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

/// Every transition state is a first-order saddle point above the two ends of its IRC (minima,
/// another transition state where a side of the IRC ends at one: a TS-TS link, or a
/// dissociation channel), and the minima are those of LJ7.
fn check(net: &Network, what: &str) {
    let known = [-16.505384, -15.935060, -15.593010, -15.532582];
    for n in net.minima() {
        let e = n.energy / epsilon();
        assert!(
            known.iter().any(|k| (e - k).abs() < 1e-3),
            "{what}: a minimum at {e} epsilon is not one of LJ7"
        );
    }
    assert!(!net.ts_edges.is_empty(), "{what}: no transition state");
    for e in &net.ts_edges {
        let ts = &net.nodes[e.ts];
        assert_eq!(ts.kind, NodeKind::TransitionState);
        assert!(e.imaginary_wavenumber.is_some_and(|w| w < 0.0));
        for &end in &e.ends {
            let kind = &net.nodes[end].kind;
            assert!(
                matches!(
                    kind,
                    NodeKind::Minimum | NodeKind::TransitionState | NodeKind::Dissociated
                ),
                "{what}: an end of an IRC is a {kind:?}"
            );
            assert!(
                ts.energy > net.nodes[end].energy,
                "{what}: {} at {} is not above its end {} at {} ({})",
                ts.label(),
                ts.energy,
                net.nodes[end].label(),
                net.nodes[end].energy,
                summary(net)
            );
        }
    }
}

fn summary(net: &Network) -> String {
    let mut minima: Vec<f64> = net.minima().map(|n| n.energy / epsilon()).collect();
    minima.sort_by(f64::total_cmp);
    format!(
        "{} jobs, minima {minima:.4?}, {} transition states, warnings {:?}",
        net.jobs.len(),
        net.ts_edges.len(),
        net.warnings.iter().map(|w| &w.message).collect::<Vec<_>>()
    )
}

#[test]
fn minima_are_not_verified_with_verify_minima_off() {
    let dir = scratch("unverified");
    let c = config(&dir, &["-map_opt", "verify_minima=off", "--max_iter", "2"]);
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(run.network.minima().all(|n| n.frequencies.is_none()));
    let doc = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
    assert!(doc.contains("\"minima_verified\": false"));
}

#[test]
fn a_complex_of_two_molecules_has_its_dissociation_channel() {
    let dir = scratch("channel");
    let input = dir.join("water_dimer.xyz");
    std::fs::write(
        &input,
        "6\n0 1\nO 0.000 0.000 0.000\nH 0.957 0.000 0.000\nH -0.240 0.927 0.000\n\
         O 2.900 0.000 0.000\nH 3.240 0.000 0.900\nH 3.240 0.000 -0.900\n",
    )
    .unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        r#"{"mapper_settings": {"max_iterations": 1},
            "step1_settings": {"othersoft": "gfnff"},
            "step4_settings": {"tight_convergence_criteria": true, "NSTEP": 400}}"#,
    )
    .unwrap();
    let args: Vec<String> = [
        input.display().to_string(),
        "-cfg".into(),
        cfg.display().to_string(),
        "-map_opt".into(),
        "store=sqlite".into(),
    ]
    .into();
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    let c = MapperConfig::from_args(&parsed, None).unwrap();
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    let net = &run.network;
    // Every dimer the job finds is a complex of the same two waters: one channel.
    let channels: Vec<_> = net.of_kind(&NodeKind::Dissociated).collect();
    assert_eq!(channels.len(), 1, "{:?}", net.warnings);
    let dc = channels[0];
    // Two waters, one record: each optimized on its own once.
    assert_eq!(dc.parts.len(), 2);
    assert_eq!(net.fragments.len(), 1);
    assert_eq!(net.fragments[0].formula, "H2O");
    assert!(net.fragments[0]
        .frequencies
        .as_ref()
        .is_some_and(|w| w.len() == 3));
    assert!((dc.energy - 2.0 * net.fragments[0].energy).abs() < 1e-12);
    // The input complex is bound: its channel lies above it.
    assert!(net
        .dissociation_edges
        .iter()
        .all(|e| net.nodes[e.minimum].kind == NodeKind::Minimum && e.channel == dc.id));
    let e = &net.dissociation_edges[0];
    assert_eq!(Some(e.minimum), net.inputs[0].node);
    let binding = e.energy_difference * Units::HARTREE_TO_KCAL_MOL;
    eprintln!("water dimer: bound by {binding:.3} kcal/mol");
    assert!((1.0..15.0).contains(&binding), "{binding}");
    let table = std::fs::read_to_string(run.directory.join("dissociation_edges.csv")).unwrap();
    assert!(
        table.contains("EQ0,DC0,1,") && table.contains("H2O + H2O"),
        "{table}"
    );
    assert!(run
        .directory
        .join("structures")
        .join("DC000000.xyz")
        .is_file());
    // The kinetics of the network: the association of the two waters.
    let out = dir.join("kinetics");
    let args: Vec<String> = [
        run.directory
            .join("reaction_network.json")
            .display()
            .to_string(),
        "--output-dir".into(),
        out.display().to_string(),
    ]
    .into();
    assert_eq!(mopt_rs::cli::run_command("run_kinetics", &args), 0);
    let rates = std::fs::read_to_string(out.join("rates.csv")).unwrap();
    assert!(
        rates.contains("EQ0:DC0") && rates.contains("capture"),
        "{rates}"
    );
    // The same kinetics from the database of the run (store=sqlite).
    let from_db = dir.join("kinetics_sqlite");
    let args: Vec<String> = [
        run.directory
            .join("reaction_network.sqlite")
            .display()
            .to_string(),
        "--output-dir".into(),
        from_db.display().to_string(),
    ]
    .into();
    assert_eq!(mopt_rs::cli::run_command("run_kinetics", &args), 0);
    assert_eq!(
        std::fs::read_to_string(from_db.join("rates.csv")).unwrap(),
        rates
    );
}

#[test]
fn addf_finds_transition_states_out_of_the_minima() {
    let dir = scratch("addf");
    let c = config(
        &dir,
        &[
            "-map_opt",
            "method=addf",
            "method_paths=4",
            "method_spheres=60",
        ],
    );
    let start = std::time::Instant::now();
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    eprintln!(
        "addf: {:.1} s, {}",
        start.elapsed().as_secs_f64(),
        summary(&run.network)
    );
    check(&run.network, "addf");
    // The files of the following of the first job.
    let files = run.directory.join("jobs").join("job_000000");
    assert!(files.join("addf_paths.csv").is_file());
    assert!(files.join("addf.log").is_file());
    // A minimum has one task.
    let first: Vec<_> = run.network.jobs.iter().filter(|j| j.node == 0).collect();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].task, "addf");
}

#[test]
fn the_nanoreactor_finds_reactions_and_transition_states() {
    let dir = scratch("nanoreactor");
    let c = config(
        &dir,
        &[
            "-map_opt",
            "method=nanoreactor",
            "method_runs=1",
            "method_steps=1500",
            "method_time_step=4",
            "method_temperature=40",
            "method_quench_every=5",
            "method_events=2",
        ],
    );
    let start = std::time::Instant::now();
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    eprintln!(
        "nanoreactor: {:.1} s, {}",
        start.elapsed().as_secs_f64(),
        summary(&run.network)
    );
    check(&run.network, "nanoreactor");
    assert!(run.network.minima().count() >= 2);
    let files = run.directory.join("jobs").join("job_000000");
    assert!(files.is_dir());
}

#[test]
fn replicas_of_the_nanoreactor_exchange_and_find_reactions() {
    // The top of the ladder: the highest temperature (K), or the factor of k2. A time step of
    // 1 fs (the single trajectory above takes 4 for fewer steps): with 2 fs the walls of these
    // runs throw an atom more than a Bohr in one step near step 360 also without replicas.
    for (exchange, top) in [("temperature", "90"), ("walls", "2")] {
        let dir = scratch(&format!("replicas_{exchange}"));
        let c = config(
            &dir,
            &[
                "-map_opt",
                "method=nanoreactor",
                "method_runs=1",
                "method_steps=3000",
                "method_time_step=1",
                "method_temperature=30",
                "method_quench_every=5",
                "method_events=2",
                "method_replicas=3",
                &format!("method_exchange={exchange}"),
                &format!("method_replica_top={top}"),
                "method_exchange_every=200",
            ],
        );
        let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
        eprintln!("replicas ({exchange}): {}", summary(&run.network));
        check(&run.network, exchange);
        assert!(run.network.minima().count() >= 2, "{exchange}");
        // The log of the job names the ladder and the exchanges that were accepted.
        let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap();
        let line = log
            .lines()
            .find(|l| l.contains("replicas of trajectory 1 on"))
            .unwrap_or_else(|| panic!("{exchange}: no line of the replicas"));
        assert!(
            line.contains("exchanges accepted between neighbouring rungs"),
            "{line}"
        );
        eprintln!("{line}");
    }
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

/// A run of `extra` stopped at evaluation `at` and restarted ends with the network of the
/// run that was not stopped.
fn stopped_run_continues(name: &str, extra: &[&str], at: usize) {
    let dir = scratch(name);
    let with_dir = |out: &str| {
        let mut all = extra.to_vec();
        all.extend(["--output_dir", out]);
        config(&dir, &all)
    };
    let whole = run_with(
        &with_dir("whole"),
        &dir,
        None,
        &CalculatorRegistry::native(),
    )
    .unwrap();
    assert!(!whole.interrupted);
    let run_dir = dir.join("stopped");
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
    let stopped = run_with(&with_dir("stopped"), &dir, None, &stopping).unwrap();
    assert!(
        stopped.interrupted,
        "{name}: the STOP file did not stop the run"
    );
    let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
    let again = MapperConfig::from_args(&settings, Some(&run_dir)).unwrap();
    let done = run_with(&again, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(!done.interrupted);
    let (a, b) = (&done.network, &whole.network);
    assert_eq!(a.jobs, b.jobs, "{name}");
    assert_eq!(a.nodes.len(), b.nodes.len(), "{name}");
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert_eq!(x.kind, y.kind, "{name}: {}", x.label());
        assert!((x.energy - y.energy).abs() < 1e-10, "{name}: {}", x.label());
    }
    assert_eq!(a.ts_edges.len(), b.ts_edges.len(), "{name}");
    for (x, y) in a.ts_edges.iter().zip(&b.ts_edges) {
        assert_eq!((x.ts, x.ends, x.job), (y.ts, y.ends, y.job), "{name}");
    }
}

#[test]
fn stopped_addf_and_nanoreactor_runs_continue_to_the_same_network() {
    stopped_run_continues(
        "addf_stop",
        &[
            "-map_opt",
            "method=addf",
            "method_paths=4",
            "method_spheres=60",
        ],
        400,
    );
    stopped_run_continues(
        "nano_stop",
        &[
            "-map_opt",
            "method=nanoreactor",
            "method_runs=1",
            "method_steps=1500",
            "method_time_step=4",
            "method_temperature=40",
            "method_quench_every=5",
            "method_events=2",
        ],
        900,
    );
}

#[test]
fn settings_of_the_methods_are_checked_before_the_run() {
    let dir = scratch("refused");
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, CONFIG).unwrap();
    let input = write_input(&dir);
    for extra in [
        &["-map_opt", "method=addf", "method_nonsense=1"][..],
        &["-map_opt", "method=nanoreactor", "method_nonsense=1"],
        &["-map_opt", "method=nanoreactor", "method_path=addf"],
        &["-map_opt", "method=nanoreactor", "method_path_nonsense=1"],
        &["-map_opt", "method=nanoreactor", "method_wide=1.5"],
        &["-map_opt", "method=nanoreactor", "method_detect=maybe"],
    ] {
        let mut args = vec![
            input.display().to_string(),
            "-cfg".to_string(),
            cfg.display().to_string(),
            "--output_dir".to_string(),
            dir.join("run").display().to_string(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        assert_eq!(
            mopt_rs::cli::run_command("run_mapper", &args),
            2,
            "{extra:?}"
        );
    }
}
