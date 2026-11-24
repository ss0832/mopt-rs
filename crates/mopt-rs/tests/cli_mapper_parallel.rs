//! `run_mapper` with several jobs at the same time, on the Lennard-Jones cluster of seven argon
//! atoms with the built-in `lj` calculator: the network of a batch does not depend on the
//! number of workers; a `STOP` with jobs in flight, a `STOP` during the commit of a batch, an
//! error during the commit (what a killed run leaves), and a second `STOP` during the restart
//! all continue with `--restart` to the network of a run that was never stopped; a calculator
//! that evaluates in order is never used by two jobs at once.
//!
//! MultiOptPy: `Wrapper/mapper.py:ReactionNetworkMapper._run_parallel_rolling` (what
//! `--max_workers` runs there; the tests are new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use mopt_rs::cli::mapper::{run_with, run_with_parts, MapperConfig, MapperParts, MapperRun};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::settings::read_settings;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::mapper::{Job, JobOutcome, MapperObserver, Network, Told};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mapper_par_{name}_{}", std::process::id()));
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

const CONFIG: &str = r#"{
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

/// The jobs of the runs of these tests: three batches of three.
const JOBS: &str = "9";

fn setup(dir: &Path) -> PathBuf {
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
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    input
}

fn config(dir: &Path, out: &str, extra: &[&str]) -> MapperConfig {
    let input = setup(dir);
    let mut args = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        dir.join("config.json").display().to_string(),
        "--max_iter".to_string(),
        JOBS.to_string(),
        "--output_dir".to_string(),
        out.to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

/// The settings of the run directory `run_dir`, for `--restart`.
fn restart_config(run_dir: &Path) -> MapperConfig {
    let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
    MapperConfig::from_args(&settings, Some(run_dir)).unwrap()
}

/// The two networks are the same: the same jobs, nodes, and edges, energies to 1e-10 Hartree.
fn assert_same(a: &Network, b: &Network, what: &str) {
    assert_eq!(a.jobs, b.jobs, "{what}: jobs");
    assert_eq!(a.warnings, b.warnings, "{what}: warnings");
    assert_eq!(a.nodes.len(), b.nodes.len(), "{what}: nodes");
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert_eq!(
            (x.kind, x.origin),
            (y.kind, y.origin),
            "{what}: {}",
            x.label()
        );
        assert!(
            (x.energy - y.energy).abs() < 1e-10,
            "{what}: {} {} {}",
            x.label(),
            x.energy,
            y.energy
        );
        let (cx, cy) = (x.system.flat(), y.system.flat());
        let apart = cx
            .iter()
            .zip(cy.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(
            apart < 1e-7,
            "{what}: {} is {apart:e} Bohr apart ({} against {})",
            x.label(),
            x.energy,
            y.energy
        );
    }
    assert_eq!(a.ts_edges.len(), b.ts_edges.len(), "{what}: edges");
    for (x, y) in a.ts_edges.iter().zip(&b.ts_edges) {
        assert_eq!((x.ts, x.ends, x.job), (y.ts, y.ends, y.job), "{what}");
        assert!((x.barriers[0] - y.barriers[0]).abs() < 1e-10, "{what}");
        assert!((x.barriers[1] - y.barriers[1]).abs() < 1e-10, "{what}");
    }
}

/// `lj` whose evaluations are counted over all its copies; it writes the file `stop` at the
/// evaluation `at` (counted from 0) and, with `in_order`, says that it evaluates in order and
/// records how many calls each copy has at the same time.
struct Standin {
    count: Arc<AtomicUsize>,
    at: Option<usize>,
    stop: PathBuf,
    in_order: bool,
    /// Calls in this copy now.
    inside: AtomicUsize,
    /// The most calls this copy had at the same time, over all copies.
    most_in_one: Arc<AtomicUsize>,
    /// Calls in every copy now, and the most there were.
    inside_all: Arc<AtomicUsize>,
    most_in_all: Arc<AtomicUsize>,
    pause: std::time::Duration,
}

impl Calculator for Standin {
    fn name(&self) -> &str {
        "lj"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let here = self.inside.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_in_one.fetch_max(here, Ordering::SeqCst);
        let all = self.inside_all.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_in_all.fetch_max(all, Ordering::SeqCst);
        let k = self.count.fetch_add(1, Ordering::SeqCst);
        if k == self.at.unwrap_or(usize::MAX) {
            if self.stop.as_os_str() == "panic" {
                self.inside.fetch_sub(1, Ordering::SeqCst);
                self.inside_all.fetch_sub(1, Ordering::SeqCst);
                panic!("a stand-in calculation that panics");
            }
            std::fs::write(&self.stop, "").unwrap();
        }
        if !self.pause.is_zero() {
            std::thread::sleep(self.pause);
        }
        let out = LennardJones.evaluate(sys, req);
        self.inside.fetch_sub(1, Ordering::SeqCst);
        self.inside_all.fetch_sub(1, Ordering::SeqCst);
        out
    }

    fn evaluates_in_order(&self) -> bool {
        self.in_order
    }
}

/// What the copies of a [`Standin`] saw.
#[derive(Clone, Default)]
struct Seen {
    count: Arc<AtomicUsize>,
    copies: Arc<AtomicUsize>,
    most_in_one: Arc<AtomicUsize>,
    inside_all: Arc<AtomicUsize>,
    most_in_all: Arc<AtomicUsize>,
}

/// A registry whose `lj` is a [`Standin`].
fn standin(
    seen: &Seen,
    at: Option<usize>,
    stop: PathBuf,
    in_order: bool,
    pause_us: u64,
) -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    let seen = seen.clone();
    r.register("lj", move |_| {
        seen.copies.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Standin {
            count: Arc::clone(&seen.count),
            at,
            stop: stop.clone(),
            in_order,
            inside: AtomicUsize::new(0),
            most_in_one: Arc::clone(&seen.most_in_one),
            inside_all: Arc::clone(&seen.inside_all),
            most_in_all: Arc::clone(&seen.most_in_all),
            pause: std::time::Duration::from_micros(pause_us),
        }) as Box<dyn Calculator>)
    });
    r
}

fn run(dir: &Path, c: &MapperConfig, registry: &CalculatorRegistry) -> MapperRun {
    run_with(c, dir, None, registry).unwrap()
}

/// The evaluations recorded in the journal of `run_dir` (every one that was computed, in
/// every session, once), and those of them with a Hessian.
fn recorded(run_dir: &Path) -> (usize, usize) {
    let (records, _) =
        mopt_rs::workflow::journal::read_journal(&run_dir.join("evaluations.journal")).unwrap();
    (
        records.len(),
        records.iter().filter(|r| r.has_hessian).count(),
    )
}

/// Every evaluation is counted once: the totals of the network are those the journal
/// recorded (a STOP saves them; after a crash or a kill the restart counts what the session
/// made after it last wrote the network from the journal).
fn assert_counted_once(run: &MapperRun, exact: bool, what: &str) {
    let (all, hessians) = recorded(&run.directory);
    let e = run.network.evaluations;
    if exact {
        assert_eq!((e.gradients, e.hessians), (all, hessians), "{what}");
    } else {
        assert!(e.gradients <= all && e.hessians <= hessians, "{what}");
    }
}

#[test]
fn the_network_of_a_batch_does_not_depend_on_the_number_of_workers() {
    let dir = scratch("workers");
    let native = CalculatorRegistry::native();
    let start = std::time::Instant::now();
    let one = run(
        &dir,
        &config(&dir, "one", &["-map_opt", "workers=1", "batch_size=3"]),
        &native,
    );
    let t1 = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let three = run(
        &dir,
        &config(&dir, "three", &["--max_workers", "3"]),
        &native,
    );
    let t3 = start.elapsed().as_secs_f64();
    eprintln!(
        "LJ7, {JOBS} jobs in batches of 3: {t1:.1} s with one worker, {t3:.1} s with three; {} \
         gradients",
        one.network.evaluations.gradients
    );
    assert!(!one.interrupted && !three.interrupted);
    assert_eq!(one.network.jobs.len(), 9);
    assert_same(&three.network, &one.network, "workers 3 against 1");
    assert_eq!(three.network.evaluations, one.network.evaluations);
    // The log of the run has the block of every job, whole and in the order of the jobs, as
    // one worker writes them; the log of a job has what it said, and only that.
    let main = std::fs::read_to_string(three.directory.join("mopt.log")).unwrap();
    assert!(
        main.contains("chosen together") && main.contains("kept ("),
        "{main}"
    );
    assert_eq!(blocks(&main), (0..9).collect::<Vec<_>>());
    let one_log = std::fs::read_to_string(one.directory.join("mopt.log")).unwrap();
    assert_eq!(
        block_texts(&main),
        block_texts(&one_log),
        "the blocks of three workers are those of one, apart from the times"
    );
    for job in &three.network.jobs {
        let id = job.id;
        let own =
            std::fs::read_to_string(three.directory.join(format!("jobs/job_{id:06}.log"))).unwrap();
        assert!(own.contains("AFIR path"), "{own}");
        for line in own.lines().filter(|l| l.starts_with("Job ")) {
            assert!(line.starts_with(&format!("Job {id}")), "{id}: {line}");
        }
        assert!(three
            .directory
            .join(format!("jobs/job_{id:06}/trajectories.csv"))
            .is_file());
    }
    // The same with the rolling schedule: every job ends, the network is one of the jobs as
    // they came (not compared), and the blocks are in the order of the jobs as well.
    let rolling = run(
        &dir,
        &config(
            &dir,
            "rolling",
            &["-map_opt", "workers=3", "schedule=rolling"],
        ),
        &native,
    );
    assert!(!rolling.interrupted);
    assert_eq!(rolling.network.jobs.len(), 9);
    let mut ids: Vec<usize> = rolling.network.jobs.iter().map(|j| j.id).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..9).collect::<Vec<_>>());
    let log = std::fs::read_to_string(rolling.directory.join("mopt.log")).unwrap();
    assert_eq!(blocks(&log), ids);
}

/// The number of the job that a line of a log names (`Job 12 ...`, not `Jobs ...`).
fn job_of(line: &str) -> Option<usize> {
    let rest = line.strip_prefix("Job ")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The blocks of the jobs in a log of the run, checked whole: a block starts with
/// `Job N (from ...` and ends with its line `Job N done after ...; the network has ...`, and
/// every line between that names a job names job N. Returns the jobs of the blocks in order.
fn blocks(log: &str) -> Vec<usize> {
    let lines: Vec<&str> = log.lines().collect();
    let mut order = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let Some(n) = job_of(lines[i]).filter(|_| lines[i].contains(" (from ")) else {
            i += 1;
            continue;
        };
        let mut j = i + 1;
        loop {
            assert!(j < lines.len(), "the block of job {n} is not closed");
            let line = lines[j];
            assert!(
                !line.starts_with("Jobs ") && !line.contains(" (from "),
                "the block of job {n} is broken by line {}: {line}",
                j + 1
            );
            if let Some(m) = job_of(line) {
                assert_eq!(m, n, "line {} in the block of job {n}: {line}", j + 1);
                if line.contains("done after") && line.contains("the network has") {
                    break;
                }
            }
            j += 1;
        }
        order.push(n);
        i = j + 1;
    }
    order
}

/// The blocks of a log without the times (seconds) in them, for a comparison of two runs.
fn block_texts(log: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in log.lines() {
        if job_of(line).is_some() && line.contains(" (from ") {
            inside = true;
            out.push(String::new());
        }
        if inside {
            if let Some(b) = out.last_mut() {
                let words: Vec<&str> = line
                    .split_whitespace()
                    .filter(|w| w.parse::<f64>().is_err() || !line.contains(" after "))
                    .collect();
                b.push_str(&words.join(" "));
                b.push('\n');
            }
            if job_of(line).is_some()
                && line.contains("done after")
                && line.contains("the network has")
            {
                inside = false;
            }
        }
    }
    out
}

/// Writes the `STOP` file of the run, or fails, when job `at` is told (in the middle of the
/// commit of its batch).
struct AtCommit {
    at: usize,
    stop: Option<PathBuf>,
    told: Arc<Mutex<Vec<usize>>>,
}

impl MapperObserver for AtCommit {
    fn after_job(
        &mut self,
        _network: &Network,
        job: &Job,
        _outcome: &JobOutcome,
        _told: &Told,
    ) -> MoptResult<()> {
        self.told.lock().unwrap().push(job.id);
        if job.id != self.at {
            return Ok(());
        }
        match &self.stop {
            Some(path) => {
                std::fs::write(path, "")?;
                Ok(())
            }
            None => Err(MoptError::config("the run ends here, as if it were killed")),
        }
    }
}

#[test]
fn a_stop_or_a_crash_at_any_phase_continues_to_the_network_of_a_run_without_stop() {
    let dir = scratch("stop");
    let native = CalculatorRegistry::native();
    let args = ["--max_workers", "3"];
    let seen = Seen::default();
    let whole = run(
        &dir,
        &config(&dir, "whole", &args),
        &standin(&seen, None, dir.join("unused"), false, 0),
    );
    let total = seen.count.load(Ordering::SeqCst);
    assert_eq!(
        seen.copies.load(Ordering::SeqCst),
        1,
        "one calculator, shared"
    );
    eprintln!("LJ7 with three workers: {total} evaluations");
    // A STOP file written at an evaluation within a batch, with the jobs of the batch in
    // flight (some may have ended), then a second STOP during the restart, then the end.
    for (k, share) in [0.15, 0.45, 0.8].into_iter().enumerate() {
        let name = format!("stopped{k}");
        let run_dir = dir.join(&name);
        let at = (share * total as f64) as usize;
        let first = run(
            &dir,
            &config(&dir, &name, &args),
            &standin(&Seen::default(), Some(at), run_dir.join("STOP"), false, 0),
        );
        assert!(first.interrupted, "{name}");
        assert!(!run_dir.join("STOP").exists());
        let kept = first.network.jobs.len();
        assert_eq!(kept % 3, 0, "{name}: whole batches only ({kept} jobs)");
        let again = run(
            &dir,
            &restart_config(&run_dir),
            &standin(&Seen::default(), Some(200), run_dir.join("STOP"), false, 0),
        );
        let again_stopped = again.interrupted;
        let done = if again.interrupted {
            assert_eq!(again.network.jobs.len() % 3, 0);
            run(&dir, &restart_config(&run_dir), &native)
        } else {
            again
        };
        assert!(!done.interrupted);
        assert_same(&done.network, &whole.network, &name);
        let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
        eprintln!(
            "{name}: STOP at evaluation {at}, {kept} jobs kept; the restart {} ; evaluations \
             {:?} against {:?}\n{}",
            if again_stopped {
                "stopped again"
            } else {
                "ran to the end"
            },
            done.network.evaluations,
            whole.network.evaluations,
            log.lines()
                .filter(|l| l.contains("Evaluations") || l.contains("journal"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        // Every evaluation is counted once, and the restart makes those of the run that was
        // not stopped: the records of a job answer that job alone, so no search ends a step
        // earlier or later on the record of another job's structure.
        assert_counted_once(&done, true, &name);
        assert_eq!(
            done.network.evaluations, whole.network.evaluations,
            "{name}"
        );
        assert!(
            log.contains("evaluations taken from evaluations.journal"),
            "{name}"
        );
        // The log of the sessions together has the block of every job once, in the order of
        // the jobs, as the run that was not stopped.
        assert_eq!(blocks(&log), (0..9).collect::<Vec<_>>(), "{name}");
    }
    // A STOP during the commit of the second batch (after the report of job 4): the batch is
    // kept whole and the run stops before the third.
    let run_dir = dir.join("at_commit");
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut stopper = AtCommit {
        at: 4,
        stop: Some(run_dir.join("STOP")),
        told: Arc::clone(&told),
    };
    let first = run_with_parts(
        &config(&dir, "at_commit", &args),
        &dir,
        None,
        &native,
        &MapperParts::default(),
        &mut [&mut stopper],
    )
    .unwrap();
    assert!(first.interrupted);
    assert_eq!(first.network.jobs.len(), 6, "the second batch whole");
    assert_eq!(*told.lock().unwrap(), (0..6).collect::<Vec<_>>());
    let done = run(&dir, &restart_config(&run_dir), &native);
    assert_same(&done.network, &whole.network, "a STOP during the commit");
    assert_counted_once(&done, true, "a STOP during the commit");
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert_eq!(blocks(&log), (0..9).collect::<Vec<_>>());
    // An error during the commit of the second batch ends the run where a kill would: the
    // network file has the first batch, and the restart runs the second again.
    let run_dir = dir.join("crash");
    let mut crash = AtCommit {
        at: 4,
        stop: None,
        told: Arc::default(),
    };
    let e = run_with_parts(
        &config(&dir, "crash", &args),
        &dir,
        None,
        &native,
        &MapperParts::default(),
        &mut [&mut crash],
    )
    .unwrap_err();
    assert!(e.to_string().contains("as if it were killed"), "{e}");
    let (left, _) = mopt_rs::workflow::mapper::store::read_network(&run_dir)
        .unwrap()
        .unwrap();
    assert_eq!(left.jobs.len(), 3, "the network of the first batch");
    let done = run(&dir, &restart_config(&run_dir), &native);
    assert_same(&done.network, &whole.network, "a crash during the commit");
    assert_counted_once(&done, true, "a crash during the commit");
    // Nothing of the batch that was being kept was written before the crash: every block once.
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert_eq!(blocks(&log), (0..9).collect::<Vec<_>>());
}

#[test]
fn a_calculation_that_panics_fails_its_job_alone() {
    let dir = scratch("panic");
    // The 300th evaluation (within the first batch) panics; the job that made it fails, the
    // others of its batch and every later job run, and nothing is left locked.
    let r = run(
        &dir,
        &config(&dir, "panic", &["--max_workers", "3"]),
        &standin(
            &Seen::default(),
            Some(300),
            PathBuf::from("panic"),
            false,
            0,
        ),
    );
    assert!(!r.interrupted);
    assert_eq!(r.network.jobs.len(), 9);
    let failed: Vec<_> = r
        .network
        .jobs
        .iter()
        .filter(|j| {
            matches!(&j.status, mopt_rs::workflow::mapper::JobStatus::Failed(e)
                if e.contains("panicked"))
        })
        .collect();
    assert_eq!(failed.len(), 1, "{:?}", r.network.jobs);
    let history = std::fs::read_to_string(r.directory.join("exploration_history.log")).unwrap();
    assert_eq!(history.lines().count(), 9);
    assert_eq!(history.matches("Status: FAILED").count(), 1);
    assert_eq!(history.matches("Status: DONE").count(), 8);
    assert!(failed[0].id < 3, "a job of the first batch");
    let done = r
        .network
        .jobs
        .iter()
        .filter(|j| j.status == mopt_rs::workflow::mapper::JobStatus::Done)
        .count();
    assert_eq!(done, 8);
    assert_counted_once(&r, true, "a panic");
}

#[test]
fn a_calculator_that_evaluates_in_order_is_never_shared_between_jobs() {
    let dir = scratch("in_order");
    let seen = Seen::default();
    // 50 microseconds per evaluation, so that the jobs overlap.
    let three = run(
        &dir,
        &config(&dir, "three", &["--max_workers", "3"]),
        &standin(&seen, None, dir.join("unused"), true, 50),
    );
    assert!(!three.interrupted);
    assert_eq!(
        seen.most_in_one.load(Ordering::SeqCst),
        1,
        "no copy had two calls at once"
    );
    assert!(
        seen.most_in_all.load(Ordering::SeqCst) >= 2,
        "the jobs ran at the same time"
    );
    // The calculator of the run (the inputs) and one for every job.
    assert_eq!(
        seen.copies.load(Ordering::SeqCst),
        1 + three.network.jobs.len()
    );
    let log = std::fs::read_to_string(three.directory.join("mopt.log")).unwrap();
    assert!(log.contains("calculator of its own"), "{log}");
    // One worker with the same batches, every job with its own calculator as well: the same
    // network.
    let one = run(
        &dir,
        &config(&dir, "one", &["-map_opt", "workers=1", "batch_size=3"]),
        &standin(&Seen::default(), None, dir.join("unused"), true, 0),
    );
    assert_same(
        &three.network,
        &one.network,
        "in order, workers 3 against 1",
    );
}
