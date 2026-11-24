//! `run_mapper` killed (SIGKILL on Unix, TerminateProcess elsewhere) and continued with
//! `--restart RUN_DIR`: the program `mopt` runs the network of the Lennard-Jones cluster of
//! seven argon atoms with one job at a time and with three, is killed at times spread over
//! the run and when a batch is being kept (on its line of the log, while the files of the
//! batch are written), the restart itself is killed at some of the points, and the network
//! the restarts end with must be that of a run that was never killed: the same jobs, nodes,
//! and edges, the energies to 1e-10 Hartree, and every evaluation counted once (the totals
//! equal to the records of the journal, which has every evaluation that was computed once).
//!
//! `MOPT_MAPPER_KILL_POINTS` sets the number of kills at times per number of workers and adds
//! the kills while the first batch is kept (by default one kill at a time and those while the
//! second batch is kept; the results in `docs/source/mapper.md` are of 20 with a release
//! build).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's mapper cannot continue a run
//! whose process was killed during a job).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use mopt_rs::domain::Units;
use mopt_rs::workflow::mapper::store::read_network;
use mopt_rs::workflow::mapper::Network;

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

/// When the process is killed.
#[derive(Debug, Clone, Copy)]
enum Kill {
    Never,
    After(Duration),
    /// On the `n`-th line of the log that has the text (a batch that is being kept).
    AtLine(&'static str, usize),
}

/// Run `mopt run_mapper ARGS` in `cwd`, killed as `kill` says; whether it was killed, and
/// the lines it printed.
fn launch(cwd: &Path, args: &[String], kill: Kill) -> (bool, Vec<String>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_mopt"))
        .arg("run_mapper")
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("the program runs");
    let out = child.stdout.take().expect("its output");
    let (to, from) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            if to.send(line).is_err() {
                break;
            }
        }
    });
    let start = Instant::now();
    let mut lines = Vec::new();
    let mut seen = 0;
    let mut killed = false;
    loop {
        let wait = match kill {
            Kill::After(t) => t.saturating_sub(start.elapsed()),
            _ => Duration::from_millis(200),
        };
        match from.recv_timeout(wait) {
            Ok(line) => {
                if let Kill::AtLine(marker, n) = kill {
                    if line.contains(marker) {
                        seen += 1;
                        if seen == n {
                            killed = child.kill().is_ok();
                        }
                    }
                }
                lines.push(line);
                if killed {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if matches!(kill, Kill::After(_)) {
                    killed = child.kill().is_ok();
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let status = child.wait().expect("the program ends");
    let _ = reader.join();
    (killed && !status.success(), lines)
}

/// A whole number from the generator `state` (a linear congruential generator, so that the
/// kill points are the same in every run of the test).
fn next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// Why `a` is not the network `b`, if it is not.
fn differs(a: &Network, b: &Network) -> Option<String> {
    if a.jobs != b.jobs {
        return Some(format!("jobs ({} against {})", a.jobs.len(), b.jobs.len()));
    }
    if a.warnings != b.warnings {
        return Some("warnings".to_string());
    }
    if a.nodes.len() != b.nodes.len() || a.ts_edges.len() != b.ts_edges.len() {
        return Some(format!(
            "{} nodes and {} edges against {} and {}",
            a.nodes.len(),
            a.ts_edges.len(),
            b.nodes.len(),
            b.ts_edges.len()
        ));
    }
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        if (x.kind, x.origin) != (y.kind, y.origin) || (x.energy - y.energy).abs() >= 1e-10 {
            return Some(format!("node {} ({} against {})", x.id, x.energy, y.energy));
        }
    }
    for (x, y) in a.ts_edges.iter().zip(&b.ts_edges) {
        if (x.ts, x.ends, x.job) != (y.ts, y.ends, y.job)
            || (x.barriers[0] - y.barriers[0]).abs() >= 1e-10
            || (x.barriers[1] - y.barriers[1]).abs() >= 1e-10
        {
            return Some(format!("edge {}", x.id));
        }
    }
    None
}

/// Why the totals of `net` do not count every evaluation once, if they do not: every
/// evaluation that was computed, in every session, is recorded once in the journal of `dir`,
/// and the restart counts those a killed session made after it last wrote the network.
fn miscounted(net: &Network, dir: &Path) -> Option<String> {
    let (records, _) =
        mopt_rs::workflow::journal::read_journal(&dir.join("evaluations.journal")).ok()?;
    let hessians = records.iter().filter(|r| r.has_hessian).count();
    let e = net.evaluations;
    (e.gradients + e.energies != records.len() || e.hessians != hessians).then(|| {
        format!(
            "evaluations not counted once: {} gradients and {} Hessians counted, {} and {} \
             recorded",
            e.gradients,
            e.hessians,
            records.len(),
            hessians
        )
    })
}

/// Whether the log of the run in `dir` has the lines of the optimizations of the jobs.
fn log_has_details(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("mopt.log")).is_ok_and(|l| l.contains("iterations, E ="))
}

/// The number of the job that a line of a log names (`Job 12 ...`, not `Jobs ...`).
fn job_of(line: &str) -> Option<usize> {
    let rest = line.strip_prefix("Job ")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The jobs of the blocks of the log of the run, in order, or why the blocks are not whole: a
/// block starts with `Job N (from ...` and ends with `Job N done after ... the network has`,
/// and every line between that names a job names job N.
fn block_order(log: &str) -> Result<Vec<usize>, String> {
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
            let Some(line) = lines.get(j) else {
                return Err(format!("the block of job {n} is not closed"));
            };
            if line.starts_with("Jobs ") || line.contains(" (from ") {
                return Err(format!("the block of job {n} is broken by: {line}"));
            }
            if let Some(m) = job_of(line) {
                if m != n {
                    return Err(format!("a line of job {m} in the block of job {n}"));
                }
                if line.contains("done after") && line.contains("the network has") {
                    break;
                }
            }
            j += 1;
        }
        order.push(n);
        i = j + 1;
    }
    Ok(order)
}

/// A file of the run directory `dir` that a kill left torn, if there is one: the network file
/// and the settings must read, every trajectory must be whole frames (or empty).
fn torn(dir: &Path) -> Option<String> {
    if dir.join("reaction_network.json").is_file() {
        if let Err(e) = read_network(dir) {
            return Some(format!("the network file does not read: {e}"));
        }
    }
    for name in ["settings.json", "config_snapshot.json"] {
        let path = dir.join(name);
        if path.is_file() {
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            if mopt_rs::core_infra::json::JsonValue::parse(&text).is_err() {
                return Some(format!("{name} does not read"));
            }
        }
    }
    let Ok(entries) = std::fs::read_dir(dir.join("jobs")) else {
        return None;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.display().to_string();
        if !name.ends_with("_traj.xyz") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        if text.trim().is_empty() {
            continue;
        }
        if let Err(e) = mopt_rs::io::xyz::parse_xyz_trajectory_str(
            &text,
            mopt_rs::domain::SpinCharge::default(),
            None,
        ) {
            return Some(format!("{name} is torn: {e}"));
        }
    }
    None
}

fn args(input: &Path, cfg: &Path, out: &Path, workers: usize) -> Vec<String> {
    vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--max_iter".to_string(),
        "9".to_string(),
        "--output_dir".to_string(),
        out.display().to_string(),
        "-map_opt".to_string(),
        format!("workers={workers}"),
    ]
}

fn restart(dir: &Path) -> Vec<String> {
    vec!["--restart".to_string(), dir.display().to_string()]
}

#[test]
fn a_killed_run_continues_to_the_network_of_a_run_that_was_not_killed() {
    let base = std::env::temp_dir().join(format!("mopt_mapper_kill_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let input = base.join("ar7.xyz");
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
    let cfg = base.join("config.json");
    std::fs::write(&cfg, CONFIG).unwrap();
    let given: Option<usize> = std::env::var("MOPT_MAPPER_KILL_POINTS")
        .ok()
        .and_then(|v| v.parse().ok());
    let points = given.unwrap_or(1);
    // The kills at lines of the log: every one with the variable, the second only without.
    let phases: &[usize] = if given.is_some() { &[1, 2] } else { &[2] };
    let mut state = 20_261_117_u64;
    let mut failures = Vec::new();
    let mut tried = 0;
    for workers in [1_usize, 3] {
        // The network of a run that is not killed: one job at a time, and batches of three
        // run three at a time.
        let reference = base.join(format!("reference_{workers}"));
        let start = Instant::now();
        let (killed, screen) = launch(&base, &args(&input, &cfg, &reference, workers), Kill::Never);
        let total = start.elapsed().as_secs_f64();
        assert!(!killed);
        let (goal, _) = read_network(&reference).unwrap().unwrap();
        assert_eq!(goal.jobs.len(), 9);
        // The screen has a short line when a job starts and when it ends, and not what the
        // jobs computed; the log of the run has the block of every job in order.
        for id in 0..9 {
            assert!(
                screen
                    .iter()
                    .any(|l| l.starts_with(&format!("Job {id} started:"))),
                "{screen:#?}"
            );
            assert!(screen
                .iter()
                .any(|l| l.starts_with(&format!("Job {id} done after"))));
        }
        // (The lines of the optimizations, `AFIR path: 61 iterations, E = ...`, stay in the
        // logs; the warnings of the jobs are on the screen.)
        assert!(
            !screen.iter().any(|l| l.contains("iterations, E =")),
            "{screen:#?}"
        );
        assert!(
            log_has_details(&reference),
            "the details are in the log of the run"
        );
        let log = std::fs::read_to_string(reference.join("mopt.log")).unwrap();
        assert_eq!(block_order(&log), Ok((0..9).collect::<Vec<_>>()));
        eprintln!(
            "workers {workers}: the run that is not killed takes {total:.2} s, {} gradients",
            goal.evaluations.gradients
        );
        let marker = if workers == 1 { "done after" } else { "kept (" };
        let mut kills: Vec<(String, Kill)> = (0..points)
            .map(|_| {
                let low = (0.1_f64).min(0.5 * total);
                let t = low + (total - low) * next(&mut state);
                (
                    format!("at {t:.2} s"),
                    Kill::After(Duration::from_secs_f64(t)),
                )
            })
            .collect();
        for &n in phases {
            kills.push((format!("keeping batch {n}"), Kill::AtLine(marker, n)));
        }
        if workers > 1 {
            // A job of the batch has ended, the others run.
            for n in phases.iter().map(|n| 2 * n) {
                kills.push((
                    format!("when job {n} of the run had ended"),
                    Kill::AtLine("done after", n),
                ));
            }
        }
        for (k, (when, kill)) in kills.into_iter().enumerate() {
            tried += 1;
            let dir: PathBuf = base.join(format!("killed_{workers}_{k}"));
            let (killed, lines) = launch(&base, &args(&input, &cfg, &dir, workers), kill);
            let kept_then = lines.iter().filter(|l| l.contains(marker)).count();
            let mut story = format!(
                "workers {workers}, kill {k} {when}: {} ({kept_then} batch(es) kept by then)",
                if killed { "killed" } else { "ended first" }
            );
            // What the kill left: every file of the network whole, every trajectory a list of
            // whole frames.
            if killed {
                if let Some(why) = torn(&dir) {
                    failures.push(format!("{story}; {why}"));
                }
            }
            if killed && !dir.join("settings.json").is_file() {
                story.push_str("; killed before the run had its settings, started again");
                let _ = std::fs::remove_dir_all(&dir);
                launch(&base, &args(&input, &cfg, &dir, workers), Kill::Never);
            }
            // Every second point the restart is killed as well, half way.
            if killed && k % 2 == 1 {
                let half = Duration::from_secs_f64(0.5 * total);
                let (again, _) = launch(&base, &restart(&dir), Kill::After(half));
                story.push_str(if again {
                    "; the restart killed too"
                } else {
                    "; the restart ended before its kill"
                });
            }
            if killed {
                let (again, lines) = launch(&base, &restart(&dir), Kill::Never);
                assert!(!again);
                if let Some(line) = lines.iter().find(|l| l.contains("evaluations taken from")) {
                    story.push_str(&format!("; restart: {}", line.trim()));
                }
            }
            let Some((net, _)) = read_network(&dir).unwrap() else {
                failures.push(format!("{story}; no network"));
                continue;
            };
            // The log of the sessions together: the blocks of the jobs whole and in order, each
            // once.
            let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap_or_default();
            let in_order = match block_order(&log) {
                Ok(order) if order == (0..9).collect::<Vec<_>>() => None,
                Ok(order) => Some(format!("the blocks of the log are of jobs {order:?}")),
                Err(why) => Some(why),
            };
            match differs(&net, &goal)
                .or_else(|| miscounted(&net, &dir))
                .or(in_order)
            {
                None => story.push_str(&format!(
                    "; the same network ({} jobs, {} nodes, {} edges; {} gradients counted, \
                     {} without a kill)",
                    net.jobs.len(),
                    net.nodes.len(),
                    net.ts_edges.len(),
                    net.evaluations.gradients,
                    goal.evaluations.gradients
                )),
                Some(why) => {
                    story.push_str(&format!("; DIFFERENT: {why}"));
                    failures.push(story.clone());
                }
            }
            eprintln!("{story}");
        }
    }
    eprintln!("{tried} kill points, {} failed", failures.len());
    assert!(failures.is_empty(), "{failures:#?}");
    let _ = std::fs::remove_dir_all(&base);
}
