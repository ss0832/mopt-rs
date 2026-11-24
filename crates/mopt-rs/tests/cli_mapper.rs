//! `run_mapper` from the command line and the library: the network of the Lennard-Jones
//! cluster of seven argon atoms with the built-in `lj` calculator, which has four minima
//! (-16.505, -15.935, -15.593, -15.533 epsilon; Hoare and Pal, Adv. Phys. 1971, 20, 161;
//! Wales and Doye, J. Phys. Chem. A 1997, 101, 5111) and transition states between them.
//!
//! MultiOptPy: `Entrypoints/mapper.py:run_mapper`.
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
use mopt_rs::workflow::mapper::NodeKind;

/// Epsilon of the argon pair of the `lj` calculator (UFF: 0.237 kcal/mol), in Hartree.
fn epsilon() -> f64 {
    0.237 / Units::HARTREE_TO_KCAL_MOL
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mapper_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// LJ7 near its lowest minimum (Bohr), as in the golden data of the optimizers.
const AR7: [[f64; 3]; 7] = [
    [6.023853759546511, 0.0, -0.15549746403767273],
    [1.962006769746685, 5.809518090796886, -0.061011157764781325],
    [-4.854674003698997, 3.590479638369873, 0.03347514850811009],
    [-4.999838497421851, -3.5904796383698727, 0.12796145478100152],
    [1.6599768546465556, -5.809518090796886, 0.2224477610538929],
    [-0.08463505395523435, 0.0, 3.794030138169188],
    [0.2933101711363313, 0.0, -3.961405880709738],
];

fn write_xyz(path: &Path, frames: &[Vec<[f64; 3]>]) {
    let mut text = String::new();
    for f in frames {
        text.push_str(&format!("{}\n0 1\n", f.len()));
        for p in f {
            text.push_str(&format!(
                "Ar {:.10} {:.10} {:.10}\n",
                p[0] * Units::BOHR_TO_ANGSTROM,
                p[1] * Units::BOHR_TO_ANGSTROM,
                p[2] * Units::BOHR_TO_ANGSTROM
            ));
        }
    }
    std::fs::write(path, text).unwrap();
}

const CONFIG: &str = r#"{
  "mapper_settings": {
    "afir_gamma_kJmol": 2.0,
    "dist_upper_ang": 7.0,
    "max_iterations": 30,
    "rng_seed": 42
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

fn config(dir: &Path, inputs: &[&Path], extra: &[&str]) -> MapperConfig {
    let cfg = dir.join("config.json");
    if !cfg.exists() {
        std::fs::write(&cfg, CONFIG).unwrap();
    }
    let mut args: Vec<String> = inputs.iter().map(|p| p.display().to_string()).collect();
    args.extend(["-cfg".to_string(), cfg.display().to_string()]);
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

#[test]
fn mapper_defaults_refresh_the_ts_and_irc_hessians_independently() {
    let dir = scratch("hessian_defaults");
    let input = dir.join("cluster.xyz");
    write_xyz(&input, &[AR7.to_vec()]);
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, r#"{"step1_settings":{"othersoft":"lj"}}"#).unwrap();
    let steps = config(&dir, &[&input], &[]).steps;
    assert_eq!(steps.saddle.settings.exact_hessian_interval, Some(5));
    assert_eq!(steps.minimum.settings.exact_hessian_interval, None);
    assert!(steps.minimum.settings.mode_kill.is_none());
    assert!(steps.verify_minima && steps.irc_hessians && steps.save_job_steps);
    assert!(
        !config(&dir, &[&input], &["-map_opt", "save_job_steps=off"])
            .steps
            .save_job_steps
    );
    assert_eq!(steps.irc_hessian_interval, 10);
    assert_eq!(steps.irc_end_threshold_cm1, 50.0);

    std::fs::write(&cfg, r#"{"mapper_settings":{"irc_end_threshold_cm1":80},"step1_settings":{"othersoft":"lj"},"step3_settings":{"calc_exact_hess":-1},"step4_settings":{"calc_exact_hess":7}}"#).unwrap();
    let steps = config(&dir, &[&input], &[]).steps;
    assert_eq!(steps.saddle.settings.exact_hessian_interval, None);
    assert_eq!(steps.irc.exact_hessian_interval, Some(7));
    assert_eq!(steps.irc_end_threshold_cm1, 80.0);
    assert_eq!(
        config(&dir, &[&input], &["-map_opt", "irc_end_threshold_cm1=65"])
            .steps
            .irc_end_threshold_cm1,
        65.0
    );
}

/// A capped octahedron of seven argon atoms (Bohr), near the second minimum.
fn capped_octahedron() -> Vec<[f64; 3]> {
    let r = 2.70 * Units::ANGSTROM_TO_BOHR;
    let c = 2.70 * Units::ANGSTROM_TO_BOHR;
    vec![
        [r, 0.0, 0.0],
        [-r, 0.0, 0.0],
        [0.0, r, 0.0],
        [0.0, -r, 0.0],
        [0.0, 0.0, r],
        [0.0, 0.0, -r],
        [c, c, c],
    ]
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

#[test]
fn a_stopped_run_continues_to_the_same_network() {
    let dir = scratch("restart");
    let input = dir.join("ar7.xyz");
    write_xyz(&input, &[AR7.to_vec()]);
    let native = CalculatorRegistry::native();
    let whole = run_with(
        &config(
            &dir,
            &[&input],
            &["--max_iter", "3", "--output_dir", "whole"],
        ),
        &dir,
        None,
        &native,
    )
    .unwrap();
    assert!(!whole.interrupted);
    assert_eq!(whole.network.jobs.len(), 3);
    // The same run, stopped by a STOP file within its second job.
    let run_dir = dir.join("stopped");
    let mut stopping = CalculatorRegistry::native();
    let count = Arc::new(AtomicUsize::new(0));
    let (c, stop) = (Arc::clone(&count), run_dir.join("STOP"));
    stopping.register("lj", move |_| {
        Ok(Box::new(StopAt {
            count: Arc::clone(&c),
            at: 260,
            stop: stop.clone(),
        }) as Box<dyn Calculator>)
    });
    let stopped = run_with(
        &config(
            &dir,
            &[&input],
            &["--max_iter", "3", "--output_dir", "stopped"],
        ),
        &dir,
        None,
        &stopping,
    )
    .unwrap();
    assert!(stopped.interrupted, "{:?}", stopped.reason);
    assert!(stopped.network.jobs.len() < 3);
    assert!(!run_dir.join("STOP").exists(), "the STOP file is removed");
    let page = std::fs::read_to_string(run_dir.join("mapper_report.html")).unwrap();
    assert!(
        page.contains("interrupted"),
        "the report of a stopped run says so"
    );
    // Continued from its directory with its own settings: the finished jobs are kept, the
    // stopped one is run again from the journal, and the network is that of the whole run.
    let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
    let again = MapperConfig::from_args(&settings, Some(&run_dir)).unwrap();
    let done = run_with(&again, &dir, None, &native).unwrap();
    assert!(!done.interrupted);
    // The journal answers a request with a record of a structure within 1e-9 Bohr of it, so
    // the numbers agree to rounding; the totals of evaluations may be lower (a structure that
    // the whole run evaluated twice, such as the last point of a search and the start of the
    // IRC from it, is answered by the journal the second time).
    let (a, b) = (&done.network, &whole.network);
    assert!(a.evaluations.gradients <= b.evaluations.gradients);
    assert!(a.evaluations.gradients + 5 >= b.evaluations.gradients);
    assert_eq!(a.jobs, b.jobs);
    assert_eq!(a.warnings, b.warnings);
    assert_eq!(a.nodes.len(), b.nodes.len());
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert_eq!((x.kind, x.origin), (y.kind, y.origin), "{}", x.label());
        assert!((x.energy - y.energy).abs() < 1e-12, "{}", x.label());
        let (cx, cy) = (x.system.flat(), y.system.flat());
        assert!(
            cx.iter().zip(cy.iter()).all(|(p, q)| (p - q).abs() < 1e-7),
            "{}",
            x.label()
        );
    }
    assert_eq!(a.ts_edges.len(), b.ts_edges.len());
    for (x, y) in a.ts_edges.iter().zip(&b.ts_edges) {
        assert_eq!((x.ts, x.ends, x.job), (y.ts, y.ends, y.job));
        assert!((x.barriers[0] - y.barriers[0]).abs() < 1e-12);
        assert_eq!(x.irc.file, y.irc.file);
    }
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(
        log.contains("evaluations taken from evaluations.journal"),
        "{log}"
    );
    assert!(log.contains("Evaluations (the whole run)"));
    // A run directory that exists is not run over.
    assert!(run_with(
        &config(
            &dir,
            &[&input],
            &["--max_iter", "3", "--output_dir", "whole"]
        ),
        &dir,
        None,
        &native
    )
    .is_err());
}

#[test]
fn two_inputs_start_one_network() {
    let dir = scratch("inputs");
    write_xyz(&dir.join("start_a.xyz"), &[AR7.to_vec()]);
    write_xyz(&dir.join("start_b.xyz"), &[capped_octahedron()]);
    let pattern = dir.join("start_*.xyz");
    let c = config(&dir, &[&pattern], &["--max_iter", "1"]);
    assert_eq!(c.inputs.len(), 2);
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    let net = &run.network;
    let nodes: Vec<Option<usize>> = net.inputs.iter().map(|i| i.node).collect();
    assert_eq!(nodes, vec![Some(0), Some(1)]);
    for (node, want) in [(0, -16.505), (1, -15.935)] {
        let e = net.nodes[node].energy / epsilon();
        assert!((e - want).abs() < 1e-3, "EQ{node}: {e}");
        assert_eq!(net.nodes[node].origin.input, Some(node));
    }
    assert_eq!(net.jobs.len(), 1);
    // The files of the run, under their names.
    let out = run.directory;
    for f in [
        "reaction_network.json",
        "nodes.csv",
        "ts_edges.csv",
        "crossing_edges.csv",
        "structures/EQ000000.xyz",
        "structures/EQ000001.xyz",
        "jobs/input_000_traj.xyz",
        "jobs/input_001_traj.xyz",
        "jobs/job_000000/trajectories.csv",
        "jobs/index.html",
        "jobs/job_000000/job_report.html",
        "jobs/job_000000/steps.csv",
        "queue_priority.log",
        "mapper_report.html",
        "settings.json",
        "config_snapshot.json",
        "mopt.log",
    ] {
        assert!(out.join(f).is_file(), "{f}");
    }
    let history = std::fs::read_to_string(out.join("exploration_history.log")).unwrap();
    assert_eq!(history.lines().count(), 1);
    assert!(history.contains("Method: afir_ts") && history.contains("Status: DONE"));
    assert!(history.contains(&net.jobs[0].task));
    let queue = std::fs::read_to_string(out.join("queue_priority.log")).unwrap();
    assert!(queue.contains("Node: EQ000000 | Priority: 1.000000000000e0"));
    assert!(queue.contains("Node: EQ000001") && queue.matches("selected=true").count() == 1);
    let steps = std::fs::read_to_string(out.join("jobs/job_000000/steps.csv")).unwrap();
    let job = out.join("jobs/job_000000");
    assert!(!job.join("queue_priority.log").exists());
    assert!(!out.join("rcmc_K_contracted.csv").exists());
    let page = std::fs::read_to_string(job.join("job_report.html")).unwrap();
    assert!(page.contains("data-model=") && page.contains("Relative energy (kcal/mol)"));
    std::fs::remove_file(job.join("job_report.html")).unwrap();
    mopt_rs::cli::mapper_report::rewrite(&out, &[]).unwrap();
    assert!(job.join("job_report.html").is_file());
    let manifest = std::fs::read_to_string(job.join("trajectories.csv")).unwrap();
    let mut frames = 0;
    for row in manifest.lines().skip(1) {
        let (_, file) = row.rsplit_once(',').unwrap();
        let xyz = std::fs::read_to_string(job.join(file)).unwrap();
        frames += xyz.lines().count() / (AR7.len() + 2);
    }
    assert!(manifest.contains("AFIR path") && manifest.lines().count() > 2);
    assert_eq!(steps.lines().count() - 1, frames);
    assert!(!out.join("jobs/job_000000_traj.xyz").exists());
    let doc = std::fs::read_to_string(out.join("reaction_network.json")).unwrap();
    assert!(
        doc.contains("\"version\": 1") && doc.contains("\"format\": \"mopt-rs reaction network\"")
    );
    let page = std::fs::read_to_string(out.join("mapper_report.html")).unwrap();
    for part in [
        "jobs/index.html",
        "queue_priority.log",
        "data-view=\"map\"",
        "data-view=\"graph\"",
        "data-view=\"dg\"",
        "<svg class=\"netfig\"",
        "id=\"network-data\"",
    ] {
        assert!(page.contains(part), "the figures of the network: {part}");
    }
    assert!(
        !page.contains("<script src=") && !page.contains("<link "),
        "one file"
    );
    // Inputs of other atoms are refused.
    let mut six = AR7.to_vec();
    six.pop();
    write_xyz(&dir.join("six.xyz"), &[AR7.to_vec(), six]);
    let c = config(&dir, &[&dir.join("six.xyz")], &["--output_dir", "six"]);
    let e = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(e.to_string().contains("same atoms"), "{e}");
}

/// The exit code of `run_mapper` with `args` and the settings file `settings`.
fn exit_code(dir: &Path, settings: &str, args: &[&str]) -> i32 {
    let cfg = dir.join("refused.json");
    std::fs::write(&cfg, settings).unwrap();
    let input = dir.join("ar7.xyz");
    write_xyz(&input, &[AR7.to_vec()]);
    let mut all = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        dir.join("refused_run").display().to_string(),
    ];
    all.extend(args.iter().map(|s| s.to_string()));
    mopt_rs::cli::run_command("run_mapper", &all)
}

#[test]
fn what_is_not_implemented_is_refused_by_name() {
    let dir = scratch("refused");
    for args in [
        // A constraint of every step with dissociation channels asked for (their fragments
        // are optimized alone; `docs/source/mapper_design.md`).
        &["-fix", "1", "-map_opt", "channels=on"][..],
        &["-ma", "100", "1", "2", "-map_opt", "method=nanoreactor"],
        &["--use_rcmc", "-map_opt", "priority=tabu"],
        &["--use_rcmc", "-map_opt", "kinetics_solver=nope"],
        &["-map_opt", "kinetics_time=1e-6"],
        &["--resume"],
        &["--max_workers", "0"],
        &["--log_file", "x.log"],
        &["-map_opt", "refine=on"],
        &["-map_opt", "crossings=on"],
        &["-map_opt", "kinetics=rcmc"],
        &["-map_opt", "priority=rcmc"],
        &["-map_opt", "nonsense=1"],
    ] {
        assert_eq!(exit_code(&dir, CONFIG, args), 2, "{args:?}");
    }
    for settings in [
        CONFIG.replace(
            "\"rng_seed\": 42",
            "\"rng_seed\": 42, \"worker_timeout_s\": 0",
        ),
        CONFIG.replace("\"rng_seed\": 42", "\"rng_seed\": 42, \"n_parallel\": 0"),
        CONFIG.replace(
            "\"rng_seed\": 42",
            "\"rng_seed\": 42, \"use_rcmc\": true, \"kinetics_score\": \"nope\"",
        ),
        CONFIG.replace("\"rng_seed\": 42", "\"rng_seed\": 42, \"no_such_key\": 1"),
        CONFIG.replace(
            "\"step3_settings\": {",
            "\"step3_settings\": {\"othersoft\": \"emt\", ",
        ),
    ] {
        assert_eq!(exit_code(&dir, &settings, &[]), 2, "{settings}");
    }
    assert!(!dir.join("refused_run").exists(), "nothing ran");
}

#[test]
fn disabling_job_steps_keeps_the_decision_and_result_without_trajectories() {
    let dir = scratch("quiet_job");
    let input = dir.join("ar7.xyz");
    write_xyz(&input, &[AR7.to_vec()]);
    let config = config(
        &dir,
        &[&input],
        &["--max_iter", "1", "-map_opt", "save_job_steps=false"],
    );
    let run = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    let root = run.directory;
    assert_eq!(run.network.jobs.len(), 1);
    assert!(root.join("exploration_history.log").is_file());
    assert!(root.join("queue_priority.log").is_file());
    assert!(root.join("reaction_network.json").is_file());
    let jobs = root.join("jobs");
    assert!(!jobs.exists() || std::fs::read_dir(jobs).unwrap().next().is_none());
}

#[test]
fn use_rcmc_chooses_the_minima_by_the_populations_of_rcmc() {
    let dir = scratch("rcmc");
    let input = dir.join("ar7.xyz");
    write_xyz(&input, &[AR7.to_vec()]);
    // MultiOptPy's RCMC queue: the published RCMC at 20 K after a nanosecond.
    let c = config(
        &dir,
        &[&input],
        &[
            "--use_rcmc",
            "--rcmc_temperature",
            "20",
            "--rcmc_time",
            "1e-9",
            "--max_iter",
            "8",
        ],
    );
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap();
    // Every job after the first transition state was chosen by the populations of RCMC, and
    // the minimum it started from is among the most populated minima it was chosen from.
    let from_of = |id: usize| -> Option<String> {
        let head = format!("Job {id} (from ");
        log.lines()
            .find_map(|l| l.strip_prefix(&head))
            .and_then(|r| r.split(',').next())
            .map(str::to_string)
    };
    let mut by_rcmc = 0;
    for line in log.lines() {
        let Some((id, reason)) = line
            .strip_prefix("Job ")
            .and_then(|r| r.split_once(": chosen by "))
        else {
            continue;
        };
        if reason.starts_with("kinetics: no transition state yet") {
            continue;
        }
        assert!(
            reason.starts_with("kinetics by rcmc (Population after 1e-9 s): "),
            "{reason}"
        );
        by_rcmc += 1;
        let from = from_of(id.parse().unwrap()).unwrap();
        let named: Vec<&str> = reason
            .split("): ")
            .nth(1)
            .unwrap()
            .split(", ")
            .map(|p| p.split(' ').next().unwrap())
            .collect();
        assert!(
            named.contains(&from.as_str()),
            "job {id} from {from}: {reason}"
        );
    }
    eprintln!(
        "LJ7 with use_rcmc: {} jobs, {} chosen by RCMC, {} minima",
        run.network.jobs.len(),
        by_rcmc,
        run.network.minima().count()
    );
    assert!(by_rcmc >= 3, "{log}");
    let csv = std::fs::read_to_string(run.directory.join("rcmc_K_contracted.csv")).unwrap();
    let last = run.network.jobs.last().unwrap().id;
    assert!(csv.contains(&format!("pop_step={last} ")));
    assert!(csv.contains("# superstate_members:") && csv.contains("node,population"));
    let populations: Vec<f64> = csv
        .split("node,population\n")
        .nth(1)
        .unwrap()
        .lines()
        .map(|row| row.rsplit_once(',').unwrap().1.parse::<f64>().unwrap())
        .collect();
    assert!((populations.iter().sum::<f64>() - 1.0).abs() < 1e-7);
    assert!(run.network.minima().count() >= 2);
}

#[test]
fn lj7_network_has_its_four_minima() {
    let dir = scratch("lj7");
    let input = dir.join("ar7.xyz");
    write_xyz(&input, &[AR7.to_vec()]);
    // The visit penalty of `tabu` spreads the jobs over the minima: the fourth minimum came
    // after 21 jobs with it and after 41 with the Boltzmann priority (seed 42).
    let c = config(&dir, &[&input], &["-map_opt", "priority=tabu"]);
    let start = std::time::Instant::now();
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    let net = &run.network;
    let mut energies: Vec<f64> = net.minima().map(|n| n.energy / epsilon()).collect();
    energies.sort_by(f64::total_cmp);
    eprintln!(
        "LJ7: {:.1} s, {} jobs, minima {energies:.4?}, {} transition states, {} with an \
         unknown end, {} warnings",
        start.elapsed().as_secs_f64(),
        net.jobs.len(),
        net.ts_edges.len(),
        net.open_ts_edges.len(),
        net.warnings.len()
    );
    // Every transition state has one edge: both ends known, or an end that is not known (its
    // minimization did not converge; docs/source/mapper_design.md), with its path file.
    for n in net
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::TransitionState)
    {
        let full = net.ts_edges.iter().filter(|e| e.ts == n.id).count();
        let open = net.open_ts_edges.iter().filter(|e| e.ts == n.id).count();
        assert_eq!(full + open, 1, "{}", n.label());
    }
    for e in &net.open_ts_edges {
        assert!(e.ends.iter().any(Option::is_none));
        assert!(run.directory.join(e.irc.file.as_deref().unwrap()).is_file());
    }
    // The four structures (the fourth is chiral: both mirror images are minima of the network).
    let mut distinct: Vec<f64> = Vec::new();
    for e in &energies {
        if distinct.last().is_none_or(|d| (e - d).abs() > 1e-3) {
            distinct.push(*e);
        }
    }
    assert_eq!(distinct.len(), 4, "{energies:?}");
    for (e, want) in distinct
        .iter()
        .zip([-16.505384, -15.935060, -15.593010, -15.532582])
    {
        assert!((e - want).abs() < 1e-3, "{e} against {want}");
    }
    // Every transition state is a first-order saddle point above the two minima of its IRC,
    // every minimum is the end of one, and the network is connected.
    assert!(net.ts_edges.len() >= 4);
    for e in &net.ts_edges {
        let ts = &net.nodes[e.ts];
        assert_eq!(ts.kind, NodeKind::TransitionState);
        assert!(e.imaginary_wavenumber.is_some_and(|w| w < 0.0));
        for (k, &end) in e.ends.iter().enumerate() {
            assert_eq!(net.nodes[end].kind, NodeKind::Minimum);
            assert!(e.barriers[k] > 0.0 && ts.energy > net.nodes[end].energy);
        }
        assert_eq!(e.irc.sides.len(), 2);
        assert!(run.directory.join(e.irc.file.as_deref().unwrap()).is_file());
        // The path file runs from minimum to minimum: the minimization of the backward end
        // (reversed), the IRC through the transition state, the minimization of the forward
        // end, whose last structures have the energies of the two ends.
        let text =
            std::fs::read_to_string(run.directory.join(e.irc.file.as_deref().unwrap())).unwrap();
        let titles: Vec<&str> = text.lines().skip(1).step_by(9).collect();
        let energy = |title: &str| -> f64 {
            title
                .rsplit("energy=")
                .next()
                .and_then(|v| v.trim().parse().ok())
                .unwrap()
        };
        assert!(titles[0].starts_with("backward minimization"), "{titles:?}");
        assert!(
            titles[titles.len() - 1].starts_with("forward minimization"),
            "{titles:?}"
        );
        assert_eq!(titles.iter().filter(|t| t.starts_with("TS ")).count(), 1);
        for (title, &end) in [titles[0], titles[titles.len() - 1]].iter().zip(&e.ends) {
            assert!(
                (energy(title) - net.nodes[end].energy).abs() < 1e-5,
                "{title} against {}",
                net.nodes[end].energy
            );
        }
        let order: Vec<&str> = titles
            .iter()
            .map(|t| t.split(" energy=").next().unwrap())
            .collect();
        let mut kinds = order.clone();
        kinds.dedup();
        assert_eq!(
            kinds,
            [
                "backward minimization",
                "backward",
                "TS",
                "forward",
                "forward minimization"
            ]
        );
    }
    let mut reached = vec![0_usize];
    let mut grew = true;
    while grew {
        grew = false;
        for e in &net.ts_edges {
            for (a, b) in [(e.ends[0], e.ends[1]), (e.ends[1], e.ends[0])] {
                if reached.contains(&a) && !reached.contains(&b) {
                    reached.push(b);
                    grew = true;
                }
            }
        }
    }
    assert_eq!(reached.len(), net.minima().count(), "connected");
    assert!(net.jobs.len() <= 30);
    // Every minimum was verified by its Hessian (verify_minima, on by default): its
    // wavenumbers are on the node, none imaginary.
    for n in net.minima() {
        let w = n.frequencies.as_ref().expect("the minimum was verified");
        // The 3N - 6 = 15 vibrations of seven atoms, none imaginary.
        assert_eq!(w.len(), 15, "{}: {w:?}", n.label());
        assert!(w.iter().all(|w| *w > 0.0), "{}: {w:?}", n.label());
    }
    // Every transition state has the wavenumbers of the Hessian of the start check of its
    // IRC: 15, one imaginary, that of its edge.
    for e in &net.ts_edges {
        let ts = &net.nodes[e.ts];
        let w = ts.frequencies.as_ref().expect("the start check of the IRC");
        assert_eq!(w.len(), 15, "{}: {w:?}", ts.label());
        assert_eq!(
            w.iter().filter(|w| **w < 0.0).count(),
            1,
            "{}: {w:?}",
            ts.label()
        );
        let lowest = w.iter().copied().fold(f64::INFINITY, f64::min);
        let edge = e.imaginary_wavenumber.unwrap();
        assert!(
            (lowest - edge).abs() < 1e-3 * edge.abs(),
            "{lowest} against {edge}"
        );
    }
    let doc = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
    assert!(
        doc.contains("\"minima_verified\": true"),
        "the network says so"
    );
}
