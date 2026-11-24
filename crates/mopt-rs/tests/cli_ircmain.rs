//! `ircmain` and `optmain -irc` through the command-line layer: plain variables and one atom
//! on the Muller-Brown surface, the files of MultiOptPy, interruption and restart, and
//! observers that see every step.
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

use mopt_rs::cli::ircmain::{config_from_cli, run_with, run_with_observers};
use mopt_rs::cli::{optmain, run_command, EXIT_OK, EXIT_USAGE};

use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, SpinCharge, Units};
use mopt_rs::error::MoptResult;
use mopt_rs::io::xyz::read_xyz_trajectory_file;
use mopt_rs::irc::{BranchEnd, IrcEvent, StartKind};
use mopt_rs::pes::native::muller_brown::{MullerBrown, MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::irc::FnObserver;
use mopt_rs::workflow::journal::JOURNAL_FILE;
use mopt_rs::workflow::observers::STOP_FILE;

fn scratch(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("mopt_ircmain_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn variables_file(base: &Path, name: &str, point: [f64; 2]) -> String {
    let path = base.join(name);
    std::fs::write(&path, format!("x {}\ny {}\n", point[0], point[1])).unwrap();
    path.to_string_lossy().into_owned()
}

fn atom_file(base: &Path, name: &str, point: [f64; 2]) -> String {
    let path = base.join(name);
    let a = Units::BOHR_TO_ANGSTROM;
    std::fs::write(
        &path,
        format!("1\n\nH {} {} 0.0\n", point[0] * a, point[1] * a),
    )
    .unwrap();
    path.to_string_lossy().into_owned()
}

/// The values of a variables file written by the command.
fn variables(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
        .collect()
}

fn nearest_minimum(p: &[f64]) -> (usize, f64) {
    let mut best = (0, f64::INFINITY);
    for (k, m) in MB_MINIMA.iter().enumerate() {
        let d = ((p[0] - m[0]).powi(2) + (p[1] - m[1]).powi(2)).sqrt();
        if d < best.1 {
            best = (k, d);
        }
    }
    best
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn plain_variables_with_the_files_of_multioptpy() {
    let base = scratch("variables");
    let ts = variables_file(&base, "ts.txt", MB_SADDLES[0]);
    let registry = CalculatorRegistry::native();
    let config = config_from_cli(&[&ts, "-os", "mb2d", "-irc_step", "0.05"]).unwrap();
    let out = run_with(&config, &base, None, &registry).unwrap();
    let run = &out[0].irc;
    assert_eq!(run.kind, Some(StartKind::TransitionState));
    assert!(!run.interrupted);
    let end = run.end.as_ref().unwrap();
    assert!(end.branches.iter().all(|b| b.end == BranchEnd::Converged));
    let dir = &run.directory;
    assert!(dir.to_string_lossy().contains("ts_IRC_"));
    // The two sides, their tables, and the combined files.
    let header = "Step,Energy (Hartree),Bias Energy (Hartree),RMS Gradient,RMS Bias Gradient";
    for (side, summary) in ["irc_forward", "irc_backward"].iter().zip(&end.branches) {
        let table = read(&dir.join(side).join("irc_energies_gradients.csv"));
        assert_eq!(table.lines().next().unwrap(), header);
        assert_eq!(table.lines().count(), summary.points + 1);
        let points = read(&dir.join(side).join("irc_points.csv"));
        assert_eq!(points.lines().next().unwrap(), "Step,x,y");
        assert_eq!(points.lines().count(), summary.points + 1);
        assert!(dir.join(side).join("plot_IRC_bending_0.png").exists());
    }
    let total: usize = end.branches.iter().map(|b| b.points).sum();
    let data = read(&dir.join("irc_combined_data.csv"));
    assert_eq!(data.lines().next().unwrap(), header);
    assert_eq!(data.lines().count(), total + 2);
    // Backward from its end with negative steps, the start as step 0, forward.
    let steps: Vec<i64> = data
        .lines()
        .skip(1)
        .map(|l| l.split(',').next().unwrap().parse().unwrap())
        .collect();
    assert!(steps.windows(2).all(|w| w[1] == w[0] + 1), "{steps:?}");
    assert_eq!(steps[end.branches[1].points], 0);
    let path = read(&dir.join("irc_combined_path.csv"));
    assert_eq!(path.lines().count(), total + 2);
    // The profile: the length runs from the forward end (positive) to the backward end.
    let profile: Vec<Vec<f64>> = read(&dir.join("irc_profile.csv"))
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(profile.len(), total + 1);
    assert!(profile.windows(2).all(|w| w[1][1] < w[0][1]));
    let top = profile.iter().max_by(|a, b| a[2].total_cmp(&b[2])).unwrap();
    assert_eq!(top[1], 0.0);
    // The end points are the two minima the saddle connects.
    assert_eq!(run.endpoints.len(), 2);
    let mut reached = Vec::new();
    for file in &run.endpoints {
        let (minimum, distance) = nearest_minimum(&variables(file));
        assert!(distance < 2e-3, "{distance}");
        reached.push(minimum);
    }
    reached.sort_unstable();
    assert_eq!(reached, vec![0, 2]);
    assert!(dir.join("settings.json").exists() && dir.join("mopt.log").exists());

    // `-irc STEP_SIZE MAX_STEP METHOD` is the same calculation.
    let config = config_from_cli(&[&ts, "-os", "mb2d", "-irc", "0.05", "300", "LQA"]).unwrap();
    let again = run_with(&config, &base, None, &registry).unwrap();
    assert_eq!(
        read(&again[0].irc.directory.join("irc_combined_data.csv")),
        data
    );

    // Other methods with their options, and their columns.
    let config = config_from_cli(&[
        &ts,
        "-os",
        "mb2d",
        "-irc_method",
        "dvv:v0=0.03:dt0=0.4",
        "-irc_max",
        "40",
    ])
    .unwrap();
    let dvv = run_with(&config, &base, None, &registry).unwrap();
    let table = read(
        &dvv[0]
            .irc
            .directory
            .join("irc_forward/irc_energies_gradients.csv"),
    );
    assert!(table
        .lines()
        .next()
        .unwrap()
        .ends_with("Time Step (fs),Damping Factor"));
    assert!(dvv[0]
        .irc
        .directory
        .join("irc_forward/plot_DVV_timesteps_0.png")
        .exists());
    let config = config_from_cli(&[&ts, "-os", "mb2d", "-irc", "0.03", "30", "rk4"]).unwrap();
    let rk4 = run_with(&config, &base, None, &registry).unwrap();
    let curvature = read(
        &rk4[0]
            .irc
            .directory
            .join("irc_forward/irc_curvature_properties.csv"),
    );
    assert!(curvature.starts_with("Scalar_Curvature,Curvature_Coupling_1"));
    assert_eq!(curvature.lines().count(), 29);
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn first_step_test_and_usage_errors() {
    let base = scratch("first");
    let ts = variables_file(&base, "ts.txt", MB_SADDLES[1]);
    let registry = CalculatorRegistry::native();
    let config = config_from_cli(&[
        &ts,
        "-os",
        "mb2d",
        "-irc_step",
        "0.05",
        "-irc_opt",
        "first_step=check",
        "initial=0.15",
    ])
    .unwrap();
    let out = run_with(&config, &base, None, &registry).unwrap();
    let table = read(&out[0].irc.directory.join("irc_first_step.csv"));
    assert!(table.starts_with("Side,Length,Angle (deg),Rho,"));
    assert_eq!(table.lines().count(), 3);
    assert!(table.lines().nth(1).unwrap().starts_with("forward,0.15"));
    let log = read(&out[0].irc.directory.join("mopt.log"));
    assert!(log.contains("First step (forward): length 0.150000, angle"));
    assert!(log.contains("rho = dE_actual / dE_predicted"));

    assert!(
        config_from_cli(&[&ts, "-os", "mb2d", "-irc_method", "lqaa"])
            .and_then(|c| run_with(&c, &base, None, &registry))
            .is_err()
    );
    assert!(config_from_cli(&[&ts, "-os", "mb2d", "-irc", "0.1", "300"]).is_err());
    assert!(config_from_cli(&[&ts, "-os", "mb2d", "-irc_opt", "nothing=1"]).is_err());
    assert!(config_from_cli(&[&ts, "-os", "mb2d", "-mf", "opt_meci"]).is_err());
    // Constraints need atoms: refused for plain variables when the run starts.
    assert!(config_from_cli(&[&ts, "-os", "mb2d", "-fix", "1"])
        .and_then(|c| run_with(&c, &base, None, &registry))
        .is_err());
    assert!(config_from_cli(&[&ts, "-os", "mb2d", "-vc"])
        .and_then(|c| run_with(&c, &base, None, &registry))
        .is_err());
    assert_eq!(run_command("ircmain", &["-h".to_string()]), EXIT_OK);
    assert_ne!(run_command("ircmain", &[]), EXIT_OK);
    assert_eq!(
        run_command(
            "ircmain",
            &[
                ts.clone(),
                "-os".into(),
                "mb2d".into(),
                "-irc_opt".into(),
                "x".into()
            ]
        ),
        EXIT_USAGE
    );
    let _ = std::fs::remove_dir_all(&base);
}

struct Counting(Arc<AtomicUsize>);

impl Calculator for Counting {
    fn name(&self) -> &str {
        "counting_mb"
    }
    fn is_rigid_body_invariant(&self) -> bool {
        false
    }
    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        self.0.fetch_add(1, Ordering::SeqCst);
        MullerBrown::default().evaluate(sys, req)
    }
}

fn counting(count: &Arc<AtomicUsize>) -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    let c = Arc::clone(count);
    r.register("counting", move |_| Ok(Box::new(Counting(Arc::clone(&c)))));
    r
}

#[test]
fn one_atom_with_structure_files_and_a_restart_in_the_middle() {
    let base = scratch("atom");
    let ts = atom_file(&base, "h.xyz", MB_SADDLES[0]);
    let args = [
        ts.as_str(),
        "-os",
        "counting",
        "-irc_step",
        "0.05",
        "-irc_method",
        "lqa",
    ];

    // An uninterrupted run.
    let count = Arc::new(AtomicUsize::new(0));
    let full = run_with(
        &config_from_cli(&args).unwrap(),
        &base,
        None,
        &counting(&count),
    )
    .unwrap();
    let total = count.load(Ordering::SeqCst);
    let full = &full[0].irc;
    assert_eq!(full.evaluations, total);
    assert_eq!(full.kind, Some(StartKind::TransitionState));
    let dir = &full.directory;
    let frames = read_xyz_trajectory_file(
        full.combined_path.as_ref().unwrap(),
        SpinCharge::default(),
        None,
    )
    .unwrap()
    .frames;
    let end = full.end.as_ref().unwrap();
    let points: usize = end.branches.iter().map(|b| b.points).sum();
    assert_eq!(frames.len(), points + 1);
    // The path runs from the forward end over the saddle to the backward end.
    let at = |k: usize| {
        let x = frames[k].system.coords_flat_bohr();
        [x[0], x[1]]
    };
    assert!(nearest_minimum(&at(0)).1 < 2e-3 && nearest_minimum(&at(frames.len() - 1)).1 < 2e-3);
    assert_ne!(
        nearest_minimum(&at(0)).0,
        nearest_minimum(&at(frames.len() - 1)).0
    );
    let saddle = at(end.branches[0].points);
    assert!((saddle[0] - MB_SADDLES[0][0]).abs() < 1e-9);
    let structures = read(&dir.join("irc_forward/irc_structures.xyz"));
    assert!(structures.starts_with("1\nIRC Step 1\nH  "));
    for k in 1..=2 {
        let endpoint = read(&dir.join(format!("h_irc_endpoint_{k}.xyz")));
        assert!(endpoint.starts_with("1\nTerminal structure\nH  "));
    }

    // MultiOptPy's masses on request (the most abundant isotopes), one side only.
    let isotope_base = base.join("isotope");
    std::fs::create_dir_all(&isotope_base).unwrap();
    let copy = atom_file(&isotope_base, "h.xyz", MB_SADDLES[0]);
    let config = config_from_cli(&[
        copy.as_str(),
        "-os",
        "muller_brown",
        "-irc_step",
        "0.05",
        "-irc_opt",
        "masses=isotope",
        "sides=forward",
    ])
    .unwrap();
    let isotope = run_with(&config, &isotope_base, None, &CalculatorRegistry::native()).unwrap();
    let isotope = &isotope[0].irc;
    assert_eq!(isotope.end.as_ref().unwrap().branches.len(), 1);
    assert!(read(&isotope.directory.join("mopt.log"))
        .contains("IRC: masses of the most abundant isotopes (as MultiOptPy)"));

    // The same run, stopped by a STOP file after the fifth point of the path.
    let count = Arc::new(AtomicUsize::new(0));
    let mut seen = 0;
    let stop_dir = Arc::new(std::sync::Mutex::new(None::<PathBuf>));
    // (Run directories are made next to the input: the same input in its own folder.)
    let stopped_base = base.join("stopped");
    std::fs::create_dir_all(&stopped_base).unwrap();
    let ts = atom_file(&stopped_base, "h.xyz", MB_SADDLES[0]);
    let args = [
        ts.as_str(),
        "-os",
        "counting",
        "-irc_step",
        "0.05",
        "-irc_method",
        "lqa",
    ];
    let config = config_from_cli(&args).unwrap();
    let first = {
        let target = Arc::clone(&stop_dir);
        let root = stopped_base.clone();
        let mut observer = FnObserver::new(move |event: &IrcEvent| {
            if matches!(event, IrcEvent::Step { .. }) {
                seen += 1;
                if seen == 5 {
                    // The run directory is the newest one below the base.
                    let newest = newest_run(&root);
                    std::fs::write(newest.join(STOP_FILE), "").unwrap();
                    *target.lock().unwrap() = Some(newest);
                }
            }
            Ok(true)
        });
        run_with_observers(
            &config,
            &stopped_base,
            None,
            &counting(&count),
            &mut [&mut observer],
        )
        .unwrap()
    };
    let first = &first[0].irc;
    assert!(first.interrupted && first.end.is_none());
    let before = count.load(Ordering::SeqCst);
    assert_eq!(before, 6, "the starting structure and five points");
    assert_eq!(stop_dir.lock().unwrap().as_ref(), Some(&first.directory));
    assert!(first.directory.join(JOURNAL_FILE).exists());
    assert!(!first.directory.join("irc_combined_data.csv").exists());

    // The restart computes what is left and writes the same files.
    let mut config = config_from_cli(&args).unwrap();
    config.restart_dir = Some(first.directory.clone());
    let count = Arc::new(AtomicUsize::new(0));
    let second = run_with(&config, &base, None, &counting(&count)).unwrap();
    let second = &second[0].irc;
    assert!(!second.interrupted);
    assert_eq!(second.directory, first.directory);
    assert_eq!(count.load(Ordering::SeqCst), total - before);
    for file in [
        "irc_combined_data.csv",
        "irc_combined_path.xyz",
        "irc_profile.csv",
        "irc_forward/irc_energies_gradients.csv",
        "irc_forward/irc_structures.xyz",
        "irc_backward/irc_structures.xyz",
        "h_irc_endpoint_1.xyz",
        "h_irc_endpoint_2.xyz",
    ] {
        assert_eq!(
            read(&second.directory.join(file)),
            read(&dir.join(file)),
            "{file}"
        );
    }
    assert!(read(&second.directory.join("mopt.log")).contains("Restart: 6 evaluations taken from"));
    let _ = std::fs::remove_dir_all(&base);
}

/// The one run directory (`..._IRC_<time>`) below `base`, at any depth.
fn newest_run(base: &Path) -> PathBuf {
    let mut found = Vec::new();
    let mut pending = vec![base.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("_IRC_")
            {
                found.push(path);
            } else {
                pending.push(path);
            }
        }
    }
    assert_eq!(found.len(), 1, "{found:?}");
    found.remove(0)
}

#[test]
fn optmain_irc_follows_the_optimized_saddle() {
    let base = scratch("optmain");
    let registry = CalculatorRegistry::native();
    // Plain variables: a saddle search from a point near it, then the path.
    let start = variables_file(&base, "near.txt", [0.25, 0.25]);
    let config = optmain::config_from_cli(&[
        &start,
        "-os",
        "mb2d",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1",
        "-irc",
        "0.05",
        "300",
        "lqa",
    ])
    .unwrap();
    let out = optmain::run_with(&config, &base, None, &registry).unwrap();
    assert!(out[0].converged && !out[0].interrupted);
    let dir = &out[0].directory;
    let log = read(&dir.join("mopt.log"));
    assert!(log.contains("Execute IRC"), "{log}");
    let mut reached = Vec::new();
    for k in 1..=2 {
        let endpoint = variables(&dir.join(format!("near_irc_endpoint_{k}.txt")));
        let (minimum, distance) = nearest_minimum(&endpoint);
        assert!(distance < 2e-3);
        reached.push(minimum);
    }
    reached.sort_unstable();
    assert_eq!(reached, vec![1, 2]);
    assert!(log.contains("Number of imaginary eigenvalues:  1"));
    assert!(log.contains("Execute IRC") && log.contains("IRC calculation is finished."));

    // One atom: the end points are in the summary of the job.
    let start = atom_file(&base, "h.xyz", [0.25, 0.25]);
    let config = optmain::config_from_cli(&[
        &start,
        "-os",
        "muller_brown",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1",
        "-irc",
        "0.05",
        "300",
        "hpc",
    ])
    .unwrap();
    let out = optmain::run_with(&config, &base, None, &registry).unwrap();
    let summary = out[0].summary.as_ref().unwrap();
    assert_eq!(summary.irc_endpoints.len(), 2);
    assert!(summary.irc_endpoints.iter().all(|p| p.exists()));
    assert!(out[0].directory.join("irc_combined_path.xyz").exists());
    // Without -irc nothing of it is written.
    let config =
        optmain::config_from_cli(&[&start, "-os", "muller_brown", "-opt", "rsirfo_bofill"])
            .unwrap();
    let out = optmain::run_with(&config, &base, None, &registry).unwrap();
    assert!(out[0].summary.as_ref().unwrap().irc_endpoints.is_empty());
    assert!(!out[0].directory.join("irc_forward").exists());
    assert!(optmain::config_from_cli(&[&start, "-os", "muller_brown", "-irc", "0.05"]).is_err());
    let _ = std::fs::remove_dir_all(&base);
}

/// The Muller-Brown calculator that writes a STOP file into the run directory below `base`
/// at its `stop_at`-th evaluation.
struct Stopping {
    count: Arc<AtomicUsize>,
    stop_at: usize,
    base: PathBuf,
}

impl Calculator for Stopping {
    fn name(&self) -> &str {
        "stopping_mb"
    }
    fn is_rigid_body_invariant(&self) -> bool {
        false
    }
    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if self.count.fetch_add(1, Ordering::SeqCst) + 1 == self.stop_at {
            for entry in std::fs::read_dir(&self.base).unwrap().flatten() {
                if entry.path().is_dir() {
                    std::fs::write(entry.path().join(STOP_FILE), "").unwrap();
                }
            }
        }
        MullerBrown::default().evaluate(sys, req)
    }
}

fn stopping(count: &Arc<AtomicUsize>, stop_at: usize, base: &Path) -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    let (c, b) = (Arc::clone(count), base.to_path_buf());
    r.register("stopping", move |_| {
        Ok(Box::new(Stopping {
            count: Arc::clone(&c),
            stop_at,
            base: b.clone(),
        }))
    });
    r
}

#[test]
fn optmain_irc_is_interrupted_and_restarted_in_the_path() {
    let base = scratch("optmain_restart");
    let arguments = |xyz: &str| {
        optmain::config_from_cli(&[
            xyz,
            "-os",
            "stopping",
            "-opt",
            "rsirfo_bofill",
            "-order",
            "1",
            "-fc",
            "1",
            "-irc",
            "0.05",
            "300",
            "lqa",
        ])
        .unwrap()
    };
    // An uninterrupted run (the STOP file is never written).
    let full_base = base.join("full");
    std::fs::create_dir_all(&full_base).unwrap();
    let xyz = atom_file(&full_base, "h.xyz", [0.25, 0.25]);
    let count = Arc::new(AtomicUsize::new(0));
    let full = optmain::run_with(
        &arguments(&xyz),
        &full_base,
        None,
        &stopping(&count, usize::MAX, &full_base),
    )
    .unwrap();
    let total = count.load(Ordering::SeqCst);
    assert!(full[0].converged && !full[0].interrupted);
    let iterations = full[0].summary.as_ref().unwrap().iterations;
    // Stopped twelve evaluations before the end: in the path, after the optimization.
    let stop_at = total - 12;
    assert!(
        stop_at > iterations + 5,
        "{stop_at} of {total}, {iterations} iterations"
    );
    let stop_base = base.join("stopped");
    std::fs::create_dir_all(&stop_base).unwrap();
    let xyz = atom_file(&stop_base, "h.xyz", [0.25, 0.25]);
    let count = Arc::new(AtomicUsize::new(0));
    let first = optmain::run_with(
        &arguments(&xyz),
        &stop_base,
        None,
        &stopping(&count, stop_at, &stop_base),
    )
    .unwrap();
    assert!(first[0].converged && first[0].interrupted);
    assert_eq!(count.load(Ordering::SeqCst), stop_at);
    assert!(!first[0].directory.join("irc_combined_data.csv").exists());
    // The restart takes the optimization and the first part of the path from the journal.
    let mut config = arguments(&xyz);
    config.restart_dir = Some(first[0].directory.clone());
    let count = Arc::new(AtomicUsize::new(0));
    let second = optmain::run_with(
        &config,
        &stop_base,
        None,
        &stopping(&count, usize::MAX, &stop_base),
    )
    .unwrap();
    assert!(second[0].converged && !second[0].interrupted);
    assert_eq!(count.load(Ordering::SeqCst), total - stop_at);
    for file in [
        "irc_combined_data.csv",
        "irc_combined_path.xyz",
        "h_irc_endpoint_1.xyz",
        "h_irc_endpoint_2.xyz",
        "h_optimized.xyz",
    ] {
        assert_eq!(
            read(&second[0].directory.join(file)),
            read(&full[0].directory.join(file)),
            "{file}"
        );
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn the_path_starts_from_the_hessian_of_a_file() {
    use mopt_rs::io::npy::write_npy_matrix;
    use mopt_rs::io::xyz::read_xyz_file;
    use mopt_rs::pes::registry::CalculatorOptions;

    let base = scratch("first_hessian");
    let ts = atom_file(&base, "h.xyz", MB_SADDLES[0]);
    let count = Arc::new(AtomicUsize::new(0));
    let registry = counting(&count);
    let args = [
        ts.as_str(),
        "-os",
        "counting",
        "-irc_step",
        "0.05",
        "-irc_method",
        "lqa",
    ];
    let own = run_with(&config_from_cli(&args).unwrap(), &base, None, &registry).unwrap();

    // The Hessian of the calculator at the saddle, in a file: the same path.
    let saddle = read_xyz_file(&ts, SpinCharge::default(), None).unwrap();
    let calculator = registry
        .create("counting", &CalculatorOptions::default(), &saddle)
        .unwrap();
    let hessian = calculator
        .evaluate(&saddle, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap();
    let file = base.join("saddle.npy");
    write_npy_matrix(&file, &hessian).unwrap();
    let mut with_file = args.to_vec();
    let name = file.to_string_lossy().into_owned();
    with_file.extend_from_slice(&["-ih", &name]);
    let read_in = run_with(
        &config_from_cli(&with_file).unwrap(),
        &base,
        None,
        &registry,
    )
    .unwrap();
    let points = |run: &mopt_rs::cli::ircmain::IrcmainResult| -> Vec<usize> {
        let end = run.irc.end.as_ref().unwrap();
        end.branches.iter().map(|b| b.points).collect()
    };
    assert_eq!(points(&read_in[0]), points(&own[0]));
    let last = |run: &mopt_rs::cli::ircmain::IrcmainResult| {
        let frames = read_xyz_trajectory_file(
            run.irc.combined_path.as_ref().unwrap(),
            SpinCharge::default(),
            None,
        )
        .unwrap()
        .frames;
        frames[frames.len() - 1].system.coords_flat_bohr()
    };
    let (a, b) = (last(&read_in[0]), last(&own[0]));
    assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
    let log = read(&read_in[0].irc.directory.join("mopt.log"));
    assert!(log.contains("the first Hessian is given"), "{log}");

    // A Hessian of another size is refused.
    let small = base.join("small.txt");
    std::fs::write(&small, "1.0 0.0\n0.0 1.0\n").unwrap();
    let mut wrong = args.to_vec();
    let name = small.to_string_lossy().into_owned();
    wrong.extend_from_slice(&["-ih", &name]);
    assert!(run_with(&config_from_cli(&wrong).unwrap(), &base, None, &registry).is_err());
    let _ = std::fs::remove_dir_all(&base);
}
