//! `run_mapper`: what a run keeps or takes beside its network of minima and transition states.
//!
//! - Path peaks (`path_peaks`): the highest point of a path that gave no transition state is
//!   saved with the minima on its two sides, in the files and the database, and is no node or
//!   edge.
//! - Steering (`steer.txt` in the run directory): lines of the user read between the jobs.
//! - `-merge`: the networks of two runs as one.
//! - Step 1 by another two-point method (`method_step1`).
//! - Entrance channels (`entrance_structures`) and the scan that confirms a barrierless
//!   dissociation (`confirm_dissociations`).
//!
//! The surface of the first three is `V = (x^2 - 1)^2 + cos(pi x) (y^2 + z^2) + y^4 + z^4 +
//! 0.5 y^2 z^2` of plain variables (minima at `x = -1` and `x = 1`); the others run on the
//! cluster of seven argon atoms and on two water molecules with GFN-FF.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::Col;
use mopt_rs::cli::mapper::{run_with, run_with_parts, MapperConfig, MapperParts, MapperRun};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::settings::read_settings;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::Units;
use mopt_rs::error::MoptResult;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::mapper::method::PathPoint;
use mopt_rs::workflow::mapper::sqlite_store::{read_sqlite, SQLITE_FILE};
use mopt_rs::workflow::mapper::store::read_network;
use mopt_rs::workflow::mapper::{
    keep_path_peak, ExplorationMethod, Job, JobContext, JobOutcome, MapperObserver, MethodSettings,
    Network, Node, NodeKind, PeakBasis, Task, Told,
};

const SURFACE: &str = "(q1^2-1)^2+cos(3.141592653589793*q1)*(q2^2+q3^2)+q2^4+q3^4+0.5*q2^2*q3^2";

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("mopt_mapper_extras_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A method with one task for the minimum of the input: the straight line from it to the
/// point `(1, 0.3, 0.3)` as a path of 21 points, and its peak kept as a path peak (`found`
/// says whether the job is to count as one that got a transition state from the path).
struct LineOnly {
    found: bool,
}

impl ExplorationMethod for LineOnly {
    fn name(&self) -> &str {
        "line_only"
    }

    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        let input = network.inputs.iter().any(|i| i.node == Some(node.id));
        Ok(if input && node.kind == NodeKind::Minimum {
            vec![Task {
                key: "line".to_string(),
                description: "a line across the barrier".to_string(),
                parameters: JsonValue::Null,
            }]
        } else {
            Vec::new()
        })
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let from = job.start.flat();
        let to = [1.0, 0.3, 0.3];
        let mut path = Vec::new();
        for k in 0..=20 {
            let t = f64::from(k) / 20.0;
            let x = Col::from_fn(3, |i| from[i] + t * (to[i] - from[i]));
            let system = job.start.with_flat(&x)?;
            let Some(energy) = ctx
                .jobs
                .energy(&system, None, &format!("point {k} of the line"))?
            else {
                return Ok(JobOutcome::stopped());
            };
            path.push(PathPoint {
                system,
                energy,
                biased_energy: energy,
            });
        }
        let mut outcome = JobOutcome::default();
        let basis = PeakBasis {
            method: "line of the test".to_string(),
            points: path.len(),
            converged: true,
            searches: 0,
        };
        if !keep_path_peak(job, ctx, &path, self.found, basis, &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

fn parts() -> MapperParts {
    let mut parts = MapperParts::default();
    for (name, found) in [("line_only", false), ("line_found", true)] {
        parts.methods.register(
            name,
            Arc::new(move |_: &MethodSettings| {
                Ok(Box::new(LineOnly { found }) as Box<dyn ExplorationMethod>)
            }),
        );
    }
    parts
}

/// A run on the surface of plain variables from the minimum at `x = -1`.
fn variables_config(dir: &Path, out: &str, extra: &[&str]) -> MapperConfig {
    let start = dir.join("start.txt");
    std::fs::write(&start, "-1.0\n0.05\n0.05\n").unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        format!(
            r#"{{
  "mapper_settings": {{"rng_seed": 1}},
  "step1_settings": {{"othersoft": "expr:{SURFACE}", "NSTEP": 300}},
  "step3_settings": {{"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300}},
  "step4_settings": {{
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"]
  }}
}}"#
        ),
    )
    .unwrap();
    let mut args = vec![
        start.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        out.to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

fn run_variables(dir: &Path, out: &str, extra: &[&str]) -> MapperRun {
    run_with_parts(
        &variables_config(dir, out, extra),
        dir,
        None,
        &CalculatorRegistry::native(),
        &parts(),
        &mut [],
    )
    .unwrap()
}

#[test]
fn the_peak_of_a_path_without_a_transition_state_is_saved_beside_the_network() {
    let dir = scratch("peak");
    let run = run_variables(
        &dir,
        "failed",
        &[
            "-map_opt",
            "method=line_only",
            "store=sqlite",
            "--max_iter",
            "1",
        ],
    );
    let net = &run.network;
    // One record: the peak of the line near x = 0, between the two minima at x = -1 and 1.
    assert_eq!(net.path_peaks.len(), 1, "{:?}", net.warnings);
    let p = &net.path_peaks[0];
    let x = p.system.flat();
    assert!(x[0].abs() < 0.15, "{x:?}");
    let ends: Vec<usize> = p.ends.iter().flatten().copied().collect();
    assert_eq!(ends.len(), 2);
    assert_ne!(ends[0], ends[1]);
    let at = |id: usize| net.nodes[id].system.flat()[0];
    assert!((at(ends[0]) + 1.0).abs() < 1e-4 && (at(ends[1]) - 1.0).abs() < 1e-4);
    // The minima lie at y = z = 0.632 with V = -0.4, and the middle of the line at (0, 0.47,
    // 0.47) with V = 1.55: 1.95 above them.
    let height = p.barriers[0].unwrap();
    assert!((1.8..2.1).contains(&height), "{height}");
    assert_eq!(p.basis.method, "line of the test");
    // No part of the network: two minima, no transition state, no edge; nothing joined.
    assert_eq!(net.nodes.len(), 2);
    assert!(net.nodes.iter().all(|n| n.kind == NodeKind::Minimum));
    assert!(net.ts_edges.is_empty());
    // The files: the table, the structure, the path from minimum to minimum.
    let table = std::fs::read_to_string(run.directory.join("path_peaks.csv")).unwrap();
    assert_eq!(table.lines().count(), 2, "{table}");
    assert!(
        table.lines().nth(1).unwrap().starts_with("0,PP0,1,"),
        "{table}"
    );
    assert!(
        table.contains("\"line of the test\",21,true,0,0,"),
        "{table}"
    );
    assert!(run
        .directory
        .join("structures")
        .join("PP000000.txt")
        .is_file());
    let path =
        std::fs::read_to_string(run.directory.join("paths").join("PP000000_path.txt")).unwrap();
    assert!(
        path.contains("backward minimization")
            && path.contains("\npath energy=")
            && path.contains("forward minimization"),
        "{}",
        &path[..path.len().min(400)]
    );
    // The network file and the database have the record; the page counts it and draws none.
    let (back, _) = read_network(&run.directory).unwrap().unwrap();
    assert_eq!(back.path_peaks.len(), 1);
    assert_eq!(back.path_peaks[0].ends, p.ends);
    let (db, _, notes) = read_sqlite(&run.directory.join(SQLITE_FILE)).unwrap();
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(db.path_peaks.len(), 1);
    assert!((db.path_peaks[0].energy - p.energy).abs() < 1e-12);
    assert_eq!(db.path_peaks[0].basis, p.basis);
    let page = std::fs::read_to_string(run.directory.join("mapper_report.html")).unwrap();
    assert!(
        page.contains("1 path peak(s)"),
        "the page says how many were saved"
    );
    assert!(!page.contains("PP0"), "and draws none");
    let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap();
    assert!(log.contains("is saved as a path peak"), "{log}");
    // A job that got its transition state saves no peak by default; `all` saves it, `off`
    // none at all.
    for (method, mode, saved) in [
        ("line_found", "failed", 0),
        ("line_found", "all", 1),
        ("line_only", "off", 0),
    ] {
        let out = format!("{method}_{mode}");
        let run = run_variables(
            &dir,
            &out,
            &[
                "-map_opt",
                &format!("method={method}"),
                &format!("path_peaks={mode}"),
                "--max_iter",
                "1",
            ],
        );
        assert_eq!(run.network.path_peaks.len(), saved, "{out}");
        assert_eq!(
            run.directory.join("path_peaks.csv").is_file(),
            saved > 0,
            "{out}"
        );
    }
    // The single points of the network leave the path peaks alone: they stay in the first run.
    let points = dir.join("points");
    let args: Vec<String> = [
        "-single_points".to_string(),
        run.directory.display().to_string(),
        "-cfg".to_string(),
        dir.join("config.json").display().to_string(),
        "--output_dir".to_string(),
        points.display().to_string(),
    ]
    .into();
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    let c = MapperConfig::from_args(&parsed, None).unwrap();
    let second = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(second.network.path_peaks.is_empty());
    assert_eq!(second.network.nodes.len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Writes `steer.txt` into the run directory after the job `after`.
struct Steers {
    dir: PathBuf,
    after: usize,
    lines: &'static str,
}

impl MapperObserver for Steers {
    fn after_job(
        &mut self,
        _network: &Network,
        job: &Job,
        _outcome: &JobOutcome,
        _told: &Told,
    ) -> MoptResult<()> {
        if job.id == self.after {
            std::fs::write(self.dir.join("steer.txt"), self.lines)?;
        }
        Ok(())
    }
}

#[test]
fn lines_in_steer_txt_change_a_run_that_goes_on() {
    let dir = scratch("steer");
    // ADDF from the minimum: without the lines, the run goes on beyond two jobs.
    let free = run_variables(
        &dir,
        "free",
        &["-map_opt", "method=addf", "--max_iter", "8"],
    );
    assert!(free.network.jobs.len() >= 3, "{}", free.network.jobs.len());
    // After the first job: a limit of 2 jobs, a line that cannot be read, a comment.
    let run_dir = dir.join("steered");
    let mut steers = Steers {
        dir: run_dir.clone(),
        after: 0,
        lines: "max_jobs 2\nfaster EQ0\n# a note of the user\n",
    };
    let run = run_with_parts(
        &variables_config(
            &dir,
            "steered",
            &["-map_opt", "method=addf", "--max_iter", "8"],
        ),
        &dir,
        None,
        &CalculatorRegistry::native(),
        &parts(),
        &mut [&mut steers],
    )
    .unwrap();
    let net = &run.network;
    assert_eq!(net.jobs.len(), 2);
    assert!(run
        .reason
        .as_deref()
        .is_some_and(|r| r.contains("limit of 2")));
    assert_eq!(net.steering.max_jobs, Some(2));
    assert_eq!(net.steering.applied.len(), 1);
    assert_eq!(net.steering.applied[0].after_jobs, 1);
    // The file is taken, and what became of every line is written down.
    assert!(!run_dir.join("steer.txt").exists());
    let log = std::fs::read_to_string(run_dir.join("steer.log")).unwrap();
    assert_eq!(log.lines().count(), 3, "{log}");
    assert!(log.contains("after 1 job(s): max_jobs 2: the limit of jobs is 2 from now on"));
    assert!(log.contains("faster EQ0: not applied"), "{log}");
    let main = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(main.contains("steer.txt: 'max_jobs 2'"), "{main}");
    // The first two jobs are those of the run that was not steered.
    assert_eq!(net.jobs[..], free.network.jobs[..2]);
    // The network file keeps the limit: a restart does not go beyond it.
    let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
    let again = run_with_parts(
        &MapperConfig::from_args(&settings, Some(&run_dir)).unwrap(),
        &dir,
        None,
        &CalculatorRegistry::native(),
        &parts(),
        &mut [],
    )
    .unwrap();
    assert_eq!(again.network.jobs.len(), 2);
    assert_eq!(again.network.steering, net.steering);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_networks_of_two_runs_are_merged_into_one() {
    let dir = scratch("merge");
    // Two runs of the same surface: a short one and a longer one from the other minimum.
    let a = run_variables(&dir, "a", &["-map_opt", "method=addf", "--max_iter", "2"]);
    std::fs::rename(dir.join("start.txt"), dir.join("first.txt")).unwrap();
    let other = dir.join("b_inputs");
    std::fs::create_dir_all(&other).unwrap();
    let b = {
        let mut c = variables_config(&other, "b", &["-map_opt", "method=addf", "--max_iter", "6"]);
        // (From the minimum at x = 1.)
        std::fs::write(other.join("start.txt"), "1.0\n0.05\n-0.05\n").unwrap();
        c.inputs = vec![other.join("start.txt")];
        run_with_parts(
            &c,
            &other,
            None,
            &CalculatorRegistry::native(),
            &parts(),
            &mut [],
        )
        .unwrap()
    };
    let merge = |runs: &[&Path], out: &str| -> MapperRun {
        let mut args = vec!["-merge".to_string()];
        args.extend(runs.iter().map(|r| r.display().to_string()));
        args.extend([
            "-cfg".to_string(),
            dir.join("config.json").display().to_string(),
            "--output_dir".to_string(),
            out.to_string(),
        ]);
        let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
        let c = MapperConfig::from_args(&parsed, None).unwrap();
        run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap()
    };
    // A run merged with itself: nothing is added but its jobs and inputs.
    let twice = merge(&[&a.directory, &a.directory], "twice");
    assert_eq!(twice.network.nodes.len(), a.network.nodes.len());
    assert_eq!(twice.network.ts_edges.len(), a.network.ts_edges.len());
    assert_eq!(twice.network.jobs.len(), 2 * a.network.jobs.len());
    assert_eq!(twice.network.inputs.len(), 2 * a.network.inputs.len());
    // The two runs: every node of each is a node of the merged network, once.
    let both = merge(&[&a.directory, &b.directory], "both");
    let net = &both.network;
    let matcher = mopt_rs::workflow::explore::identity::StructureMatcher {
        rmsd_tolerance: mopt_rs::domain::Quantity::new(0.01),
        ..mopt_rs::workflow::mapper::mapper_matcher(0.125, 0.05, false)
    };
    for source in [&a.network, &b.network] {
        for n in &source.nodes {
            let same = net
                .find_same(&n.kind, n.level, &n.system, n.energy, &matcher)
                .unwrap();
            assert!(same.is_some(), "{} is in the merged network", n.label());
        }
    }
    assert!(net.nodes.len() >= a.network.nodes.len().max(b.network.nodes.len()));
    assert!(net.nodes.len() <= a.network.nodes.len() + b.network.nodes.len());
    assert_eq!(net.jobs.len(), a.network.jobs.len() + b.network.jobs.len());
    // The numbers of the jobs go on, and every job starts from a minimum of the merged
    // network.
    let ids: Vec<usize> = net.jobs.iter().map(|j| j.id).collect();
    assert_eq!(ids, (0..net.jobs.len()).collect::<Vec<_>>());
    assert!(net
        .jobs
        .iter()
        .all(|j| net.nodes[j.node].kind == NodeKind::Minimum));
    // The edges: every transition state once, with its path file under its number.
    let mut ts: Vec<usize> = net.ts_edges.iter().map(|e| e.ts).collect();
    ts.sort_unstable();
    ts.dedup();
    assert_eq!(ts.len(), net.ts_edges.len());
    for e in &net.ts_edges {
        let file = e.irc.file.as_deref().unwrap();
        assert!(both.directory.join(file).is_file(), "{file}");
        assert!(
            file.contains(&format!("TS{:06}", net.nodes[e.ts].number)),
            "{file}"
        );
    }
    // The table of the merge, and the files of any run.
    let table = std::fs::read_to_string(both.directory.join("merge.csv")).unwrap();
    assert_eq!(
        table.lines().count(),
        1 + a.network.nodes.len() + b.network.nodes.len(),
        "{table}"
    );
    // The surface is even in x: the run from x = 1 finds the mirror images of the minima of
    // the run from x = -1, other points, so all of its nodes are added.
    assert!(
        table.contains(",first") && table.contains(",added") && !table.contains(",known"),
        "{table}"
    );
    // Merged with itself, every node of the second copy is known.
    let table = std::fs::read_to_string(twice.directory.join("merge.csv")).unwrap();
    assert!(
        table.contains(",known") && !table.contains(",added"),
        "{table}"
    );
    assert!(both.directory.join("mapper_report.html").is_file());
    let (back, run_entry) = read_network(&both.directory).unwrap().unwrap();
    assert_eq!(back.nodes.len(), net.nodes.len());
    assert_eq!(
        run_entry.get("method").and_then(JsonValue::as_str),
        Some("merge")
    );
    // An input file with -merge, and a restart of the merged directory, are refused.
    let bad: Vec<String> = [
        dir.join("first.txt").display().to_string(),
        "-merge".to_string(),
        a.directory.display().to_string(),
    ]
    .into();
    let parsed = parser("run_mapper", RUN_MAPPER).parse(bad).unwrap();
    let e = MapperConfig::from_args(&parsed, None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("no input file"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
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

fn argon_config(dir: &Path, out: &str, extra: &[&str]) -> MapperConfig {
    let input = dir.join("ar7.xyz");
    let mut text = format!("{}\n0 1\n", AR7.len());
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
    std::fs::write(
        &cfg,
        r#"{
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
}"#,
    )
    .unwrap();
    let mut args = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        out.to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

/// HCN with GFN0-xTB: its one pair of atoms that are not bonded is H and N.
fn hcn_config(dir: &Path, out: &str, extra: &[&str]) -> MapperConfig {
    let input = dir.join("hcn.xyz");
    std::fs::write(
        &input,
        "3\n0 1\nH -1.066 0.0 0.0\nC 0.0 0.0 0.0\nN 1.156 0.0 0.0\n",
    )
    .unwrap();
    let cfg = dir.join("hcn.json");
    std::fs::write(
        &cfg,
        r#"{
  "mapper_settings": {"rng_seed": 1},
  "step1_settings": {"othersoft": "gfn0xtb", "NSTEP": 300},
  "step3_settings": {"calc_exact_hess": 1, "NSTEP": 150},
  "step4_settings": {"NSTEP": 300, "intrinsic_reaction_coordinates": ["0.1", "300", "lqa"]}
}"#,
    )
    .unwrap();
    let mut args = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        out.to_string(),
        "--max_iter".to_string(),
        "1".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

#[test]
fn step_1_grows_a_path_from_the_minimum_along_the_pair() {
    let dir = scratch("step1");
    let registry = CalculatorRegistry::native();
    let count = |run: &MapperRun| {
        (
            run.network.minima().count(),
            run.network.of_kind(&NodeKind::TransitionState).count(),
            run.network.path_peaks.len(),
            run.network.evaluations.gradients,
            run.network.evaluations.hessians,
        )
    };
    // The one job of HCN (the pair H, N) with the AFIR push and its relaxed chain (the
    // default), and with the single-ended growing string driven by the pair.
    let default = run_with(&hcn_config(&dir, "afir", &[]), &dir, None, &registry).unwrap();
    let string = run_with(
        &hcn_config(&dir, "se_gsm", &["-map_opt", "method_step1=se_gsm"]),
        &dir,
        None,
        &registry,
    )
    .unwrap();
    eprintln!(
        "HCN, the pair H-N (minima, transition states, path peaks, gradients, Hessians): the \
         AFIR push and its chain {:?}, the single-ended growing string {:?}",
        count(&default),
        count(&string)
    );
    // The same task: step 1 changes how the path of a pair is made, not the pairs.
    assert_eq!(string.network.jobs.len(), 1);
    assert_eq!(string.network.jobs[0].task, default.network.jobs[0].task);
    let log = std::fs::read_to_string(string.directory.join("mopt.log")).unwrap();
    assert!(log.contains("se_gsm along the atoms 1 and 3"), "{log}");
    assert!(
        !log.contains("AFIR path") && !log.contains("neb of the AFIR path"),
        "no push and no step 2 with it: {log}"
    );
    // The isomerization to HNC comes of it: a transition state, and HNC at the other end.
    let net = &string.network;
    assert_eq!(count(&string).1, 1, "{:?}", net.warnings);
    let edge = &net.ts_edges[0];
    let barrier = edge.barriers[0].max(edge.barriers[1]) * Units::HARTREE_TO_KCAL_MOL;
    assert!((30.0..80.0).contains(&barrier), "{barrier}");
    assert_eq!(net.minima().count(), 2);
    // The Newton trajectory on the cluster of seven argon atoms: the pair is the distance it
    // shortens.
    let newton = run_with(
        &argon_config(
            &dir,
            "gnt",
            &["--max_iter", "2", "-map_opt", "method_step1=gnt"],
        ),
        &dir,
        None,
        &registry,
    )
    .unwrap();
    eprintln!(
        "Ar7, two pairs with the Newton trajectory: {:?}",
        count(&newton)
    );
    assert_eq!(newton.network.jobs.len(), 2);
    let log = std::fs::read_to_string(newton.directory.join("mopt.log")).unwrap();
    assert!(log.contains("gnt along the atoms"), "{log}");
    // The methods of step 2 and the methods that need a second structure are refused, with
    // the reason.
    for (value, why) in [
        ("neb", "method_relax"),
        ("dmf", "method_relax"),
        ("fsm", "method=connect"),
        ("gsm", "method=connect"),
        ("addf", "as a whole"),
    ] {
        let c = hcn_config(
            &dir,
            "refused",
            &["-map_opt", &format!("method_step1={value}")],
        );
        let e = match run_with(&c, &dir, None, &registry) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("method_step1={value} ran"),
        };
        assert!(e.contains(why), "{value}: {e}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn water_config(dir: &Path, out: &str, options: &[&str]) -> MapperConfig {
    let input = dir.join("waters.xyz");
    // Two water molecules far apart: the channel of the two.
    std::fs::write(
        &input,
        "6\n0 1\nO 0.000 0.000 0.000\nH 0.957 0.000 0.000\nH -0.240 0.927 0.000\n\
         O 7.900 0.000 0.000\nH 8.240 0.000 0.900\nH 8.240 0.000 -0.900\n",
    )
    .unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        r#"{"mapper_settings": {"max_iterations": 1, "rng_seed": 3},
            "step1_settings": {"othersoft": "gfnff", "NSTEP": 400},
            "step4_settings": {"tight_convergence_criteria": true, "NSTEP": 400}}"#,
    )
    .unwrap();
    let mut args = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        out.to_string(),
        "-map_opt".to_string(),
        "store=sqlite".to_string(),
    ];
    args.extend(options.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

#[test]
fn the_structures_of_an_entrance_channel_are_inputs_and_its_dissociation_is_scanned() {
    let dir = scratch("entrance");
    let registry = CalculatorRegistry::native();
    let c = water_config(
        &dir,
        "run",
        &["entrance_structures=3", "confirm_dissociations=on"],
    );
    let run = run_with(&c, &dir, None, &registry).unwrap();
    let net = &run.network;
    // The input and three structures of its channel: four inputs, each a minimum.
    assert_eq!(net.inputs.len(), 4);
    assert!(net.inputs[1..]
        .iter()
        .enumerate()
        .all(|(k, i)| i.file.ends_with(&format!("(entrance structure {})", k + 1))));
    assert!(net.inputs.iter().all(|i| i.node.is_some()));
    // One channel of two waters, which is the entrance of the network.
    let channels: Vec<&Node> = net.of_kind(&NodeKind::Dissociated).collect();
    assert_eq!(channels.len(), 1, "{:?}", net.warnings);
    let entrance = net.entrance.as_ref().expect("the network has its entrance");
    assert_eq!(entrance.channel, channels[0].id);
    assert_eq!((entrance.structures, entrance.inputs.len()), (3, 3));
    assert_eq!(entrance.inputs, vec![1, 2, 3]);
    // The structures in contact fell to complexes, bound below the channel.
    let complexes: std::collections::BTreeSet<usize> = entrance
        .inputs
        .iter()
        .filter_map(|&k| net.inputs[k].node)
        .collect();
    assert!(!complexes.is_empty());
    for &id in &complexes {
        let edge = net
            .dissociation_edges
            .iter()
            .find(|e| e.minimum == id)
            .expect("a complex has its edge to the channel");
        assert_eq!(edge.channel, channels[0].id);
        let binding = edge.energy_difference * Units::HARTREE_TO_KCAL_MOL;
        assert!((1.0..15.0).contains(&binding), "{binding}");
        // The scan of its dissociation: the energy rises to separated molecules without a
        // maximum above the channel.
        let scan = edge.scan.as_ref().expect("the edge was scanned");
        assert!(scan.separated && !scan.barrier, "{scan:?}");
        assert!(scan.distances.len() >= 4, "{scan:?}");
        assert!(scan.distances.windows(2).all(|w| w[1] > w[0]));
        let (first, last) = (scan.energies[0], scan.energies[scan.energies.len() - 1]);
        assert!(
            last > first && last <= channels[0].energy + 1e-4,
            "{scan:?}"
        );
        eprintln!(
            "water dimer: bound by {binding:.2} kcal/mol; scan of {} points to {:.2} Angstrom, \
             the last {:.2} kcal/mol below the channel",
            scan.distances.len(),
            scan.distances[scan.distances.len() - 1] * Units::BOHR_TO_ANGSTROM,
            (channels[0].energy - last) * Units::HARTREE_TO_KCAL_MOL
        );
    }
    // The network file and the database keep the entrance and the scans.
    let (back, _) = read_network(&run.directory).unwrap().unwrap();
    assert_eq!(back.entrance, net.entrance);
    let (db, _, notes) = read_sqlite(&run.directory.join(SQLITE_FILE)).unwrap();
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(db.entrance, net.entrance);
    for (a, b) in db.dissociation_edges.iter().zip(&net.dissociation_edges) {
        assert_eq!(a.scan, b.scan);
    }
    // The kinetics of the network starts on the entrance channel.
    let read = mopt_rs::workflow::kinetics::read_network(
        &run.directory,
        mopt_rs::workflow::mapper::network::Level::Exploration,
        mopt_rs::kinetics::conditions::EnergyKind::Electronic,
    )
    .unwrap();
    assert_eq!(read.default_start, Some(entrance.channel));
    let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Entrance channel: DC0"), "{log}");
    // The same structures for the same seed, others for another.
    let again = run_with(
        &water_config(&dir, "again", &["entrance_structures=3"]),
        &dir,
        None,
        &registry,
    )
    .unwrap();
    for (a, b) in again.network.nodes.iter().zip(&net.nodes) {
        assert!((a.energy - b.energy).abs() < 1e-9, "{}", a.label());
    }
    // One molecule has no entrance channel.
    std::fs::write(
        dir.join("water.xyz"),
        "3\n0 1\nO 0.000 0.000 0.000\nH 0.957 0.000 0.000\nH -0.240 0.927 0.000\n",
    )
    .unwrap();
    let mut one = water_config(&dir, "one", &["entrance_structures=2"]);
    one.inputs = vec![dir.join("water.xyz")];
    let e = match run_with(&one, &dir, None, &registry) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("one molecule had an entrance channel"),
    };
    assert!(e.contains("one molecule"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}
