//! HCN <-> HNC with the built-in GFN0-xTB calculator: a transition-state optimization, then
//! the IRC to both minima with every integrator, the test and correction of the first
//! step, and the analysis of the path.
//!
//! `first_step_numbers_and_step_sizes` prints the tables of `docs/source/irc.md`
//! (`MOPT_FULL_MEASUREMENTS=1 cargo test -p mopt-rs --test irc_hcn first_step_numbers --
//! --nocapture`; without the variable it runs a part of them).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use mopt_rs::cli::ircmain::{self, IrcmainResult};
use mopt_rs::cli::optmain;
use mopt_rs::domain::{SpinCharge, Units};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::irc::{BranchEnd, StartKind};
use mopt_rs::pes::registry::CalculatorRegistry;

/// A guess of the transition state: H bridging C and N.
const GUESS: &str = "3\n\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n";

fn scratch(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("mopt_irc_hcn_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Distances H-C and H-N (Angstrom) and the angle H-C-N (degrees) of a structure file.
fn shape(path: &Path) -> (f64, f64, f64) {
    let sys = read_xyz_file(path, SpinCharge::default(), None).unwrap();
    let x = sys.coords_flat_bohr();
    let at = |i: usize| [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
    let (c, n, h) = (at(0), at(1), at(2));
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let (ch, cn) = (sub(h, c), sub(n, c));
    let cos = (ch[0] * cn[0] + ch[1] * cn[1] + ch[2] * cn[2]) / (len(ch) * len(cn));
    (
        len(ch) * Units::BOHR_TO_ANGSTROM,
        len(sub(h, n)) * Units::BOHR_TO_ANGSTROM,
        cos.clamp(-1.0, 1.0).acos().to_degrees(),
    )
}

/// `hcn` when H sits on C opposite to N, `hnc` when it sits on N opposite to C.
fn isomer(path: &Path) -> &'static str {
    let (hc, hn, angle) = shape(path);
    if hc < 1.25 && hn > 2.0 && angle > 150.0 {
        "hcn"
    } else if hn < 1.2 && hc > 2.0 && angle < 30.0 {
        "hnc"
    } else {
        panic!("neither HCN nor HNC: H-C {hc:.3}, H-N {hn:.3}, H-C-N {angle:.1}")
    }
}

/// The optimized transition state (one optimization for all tests), with its energy.
fn transition_state() -> &'static (PathBuf, f64) {
    static TS: OnceLock<(PathBuf, f64)> = OnceLock::new();
    TS.get_or_init(|| {
        let base = scratch("ts");
        let guess = base.join("hcn.xyz");
        std::fs::write(&guess, GUESS).unwrap();
        let config = optmain::config_from_cli(&[
            guess.to_str().unwrap(),
            "-os",
            "gfn0xtb",
            "-opt",
            "rsirfo_bofill",
            "-order",
            "1",
            "-fc",
            "1",
            "-tcc",
        ])
        .unwrap();
        let out = optmain::run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
        assert!(out[0].converged);
        let summary = out[0].summary.as_ref().unwrap();
        (out[0].optimized.clone(), summary.energy.value())
    })
}

fn irc(base: &Path, arguments: &[&str]) -> IrcmainResult {
    // Run directories are made next to the input: a copy of the transition state in `base`.
    let ts = base.join("ts.xyz");
    std::fs::copy(&transition_state().0, &ts).unwrap();
    let mut args = vec![ts.to_str().unwrap(), "-os", "gfn0xtb"];
    args.extend_from_slice(arguments);
    let config = ircmain::config_from_cli(&args).unwrap();
    ircmain::run_with(&config, base, None, &CalculatorRegistry::native())
        .unwrap()
        .remove(0)
}

fn isomers(run: &IrcmainResult) -> Vec<&'static str> {
    let mut found: Vec<&str> = run.irc.endpoints.iter().map(|p| isomer(p)).collect();
    found.sort_unstable();
    found
}

#[test]
fn optmain_irc_from_a_guess_reaches_hcn_and_hnc() {
    let base = scratch("optmain");
    let guess = base.join("hcn.xyz");
    std::fs::write(&guess, GUESS).unwrap();
    let config = optmain::config_from_cli(&[
        guess.to_str().unwrap(),
        "-os",
        "gfn0xtb",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1",
        "-irc",
        "0.1",
        "300",
        "lqa",
    ])
    .unwrap();
    let out = optmain::run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    assert!(out[0].converged);
    let summary = out[0].summary.as_ref().unwrap();
    let mut found: Vec<&str> = summary.irc_endpoints.iter().map(|p| isomer(p)).collect();
    found.sort_unstable();
    assert_eq!(found, vec!["hcn", "hnc"]);
    let dir = &out[0].directory;
    let log = read(&dir.join("mopt.log"));
    assert!(log.contains("Number of imaginary eigenvalues:  1") && log.contains("Execute IRC"));
    assert_eq!(log.matches("Convergence reached. (IRC)").count(), 2);
    // The energies along the combined table: the start is the highest point, and HCN lies
    // below HNC.
    let rows: Vec<Vec<f64>> = read(&dir.join("irc_combined_data.csv"))
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    let top = rows.iter().max_by(|a, b| a[1].total_cmp(&b[1])).unwrap();
    assert_eq!(top[0], 0.0);
    let ends = [rows[0][1], rows[rows.len() - 1][1]];
    let barrier = top[1] - ends[0].min(ends[1]);
    assert!(barrier > 0.03 && barrier < 0.15, "{barrier}");
    assert!((ends[0] - ends[1]).abs() > 0.005);
    // The combined path is a path for freqmain -path: frames in one orientation.
    let frames = read_xyz_trajectory_file(
        dir.join("irc_combined_path.xyz"),
        SpinCharge::default(),
        None,
    )
    .unwrap()
    .frames;
    assert_eq!(frames.len(), rows.len());
    // All frames, the starting structure among them, have their centre of mass at the
    // origin (MultiOptPy copies the start as the input has it).
    for frame in &frames {
        let com = frame.system.center_of_mass().value();
        assert!(com.iter().all(|c| c.abs() < 1e-8), "{com:?}");
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn every_integrator_connects_the_two_isomers() {
    let base = scratch("methods");
    // (arguments, ends that are accepted)
    let cases: [(&[&str], &[BranchEnd]); 6] = [
        (&["-irc", "0.1", "300", "lqa"], &[BranchEnd::Converged]),
        (&["-irc", "0.1", "300", "hpc"], &[BranchEnd::Converged]),
        (
            &["-irc", "0.1", "300", "gs2"],
            &[BranchEnd::Converged, BranchEnd::EnergyRise],
        ),
        (
            &["-irc", "0.05", "400", "rk4", "-irc_opt", "energy_rise=stop"],
            &[BranchEnd::EnergyRise, BranchEnd::Converged],
        ),
        (
            &["-irc", "0.02", "600", "euler"],
            &[BranchEnd::Oscillation, BranchEnd::Converged],
        ),
        (
            &["-irc", "0.1", "600", "dvv", "-irc_opt", "energy_rise=stop"],
            &[BranchEnd::EnergyRise, BranchEnd::Converged],
        ),
    ];
    for (arguments, accepted) in cases {
        let run = irc(&base, arguments);
        assert_eq!(
            run.irc.kind,
            Some(StartKind::TransitionState),
            "{arguments:?}"
        );
        let end = run.irc.end.as_ref().unwrap();
        for side in &end.branches {
            assert!(
                accepted.contains(&side.end),
                "{arguments:?}: {:?}",
                side.end
            );
        }
        assert_eq!(isomers(&run), vec!["hcn", "hnc"], "{arguments:?}");
        let _ = std::fs::remove_dir_all(&run.irc.directory);
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_long_first_step_is_corrected_and_the_path_is_analysed() {
    let base = scratch("first");
    // 0.6 amu^0.5 Bohr along the transition vector in one step.
    let run = irc(
        &base,
        &[
            "-irc",
            "0.1",
            "300",
            "lqa",
            "-irc_opt",
            "first_step=check",
            "initial=0.6",
            "-path_analysis",
        ],
    );
    assert_eq!(isomers(&run), vec!["hcn", "hnc"]);
    let dir = &run.irc.directory;
    let table: Vec<Vec<String>> = read(&dir.join("irc_first_step.csv"))
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect();
    assert_eq!(table.len(), 2);
    let corrected: Vec<&Vec<String>> = table.iter().filter(|r| r[6] == "true").collect();
    assert!(!corrected.is_empty(), "{table:?}");
    for row in corrected {
        // Converged in a few evaluations, with a lower energy than the straight point.
        assert_eq!(row[8], "true");
        assert!(row[7].parse::<usize>().unwrap() <= 10);
        assert!(row[10].parse::<f64>().unwrap() < 0.0);
    }
    let log = read(&dir.join("mopt.log"));
    assert!(log.contains("replaced by the energy minimum on the hypersphere"));
    // The analysis of freqmain -path of the combined path.
    let analysis = run.path_analysis.as_ref().unwrap();
    assert_eq!(analysis, &dir.join("path_analysis"));
    for name in [
        "path_profile.csv",
        "curvature_couplings.csv",
        "variational_tst.txt",
    ] {
        assert!(analysis.join(name).exists(), "{name}");
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_bias_potential_is_followed_and_reported() {
    let base = scratch("bias");
    // A harmonic potential on the C-N distance with its minimum at the distance of the
    // transition state: the start stays a stationary point of the surface with it, and the
    // path feels it as the distance changes.
    let ts = read_xyz_file(&transition_state().0, SpinCharge::default(), None).unwrap();
    let x = ts.coords_flat_bohr();
    let cn =
        ((0..3).map(|c| (x[c] - x[3 + c]).powi(2)).sum::<f64>()).sqrt() * Units::BOHR_TO_ANGSTROM;
    let distance = format!("{cn:.8}");
    let run = irc(
        &base,
        &["-irc", "0.1", "300", "lqa", "-kp", "0.05", &distance, "1,2"],
    );
    assert_eq!(run.irc.kind, Some(StartKind::TransitionState));
    assert_eq!(isomers(&run), vec!["hcn", "hnc"]);
    let rows: Vec<Vec<f64>> = read(&run.irc.directory.join("irc_combined_data.csv"))
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    // "Bias Energy" is the energy with the bias potential: equal at the start, above the
    // energy of the calculator elsewhere, and the two gradients differ.
    let start = rows.iter().find(|r| r[0] == 0.0).unwrap();
    assert!((start[2] - start[1]).abs() < 1e-9);
    assert!(rows.iter().all(|r| r[2] >= r[1] - 1e-12));
    let largest = rows.iter().map(|r| r[2] - r[1]).fold(0.0, f64::max);
    assert!(largest > 1e-5, "{largest}");
    assert!(rows.iter().any(|r| (r[4] - r[3]).abs() > 1e-6));
    // The path with the bias potential is another one than without it.
    let plain = irc(&base, &["-irc", "0.1", "300", "lqa"]);
    assert_ne!(
        read(&plain.irc.directory.join("irc_forward/irc_structures.xyz")),
        read(&run.irc.directory.join("irc_forward/irc_structures.xyz"))
    );
    let _ = std::fs::remove_dir_all(&base);
}

/// The numbers of the test of the first step over its length, and the energy profile over
/// the step size of the integrators: two lengths on every run, where every side has its
/// row; the tables of the documentation with `MOPT_FULL_MEASUREMENTS` set (`cargo test -p
/// mopt-rs --test irc_hcn first_step_numbers -- --nocapture`).
#[test]
fn first_step_numbers_and_step_sizes() {
    let full = std::env::var_os("MOPT_FULL_MEASUREMENTS").is_some();
    let base = scratch("measure");
    let energy_ts = transition_state().1;
    println!(
        "| length | side | angle (deg) | rho | corrected | evaluations | moved | energy change |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    let lengths: &[&str] = if full {
        &[
            "0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.4", "0.6", "0.8",
        ]
    } else {
        &["0.1", "0.4"]
    };
    for length in lengths {
        let initial = format!("initial={length}");
        let run = irc(
            &base,
            &[
                "-irc",
                "0.1",
                "3",
                "lqa",
                "-irc_opt",
                "first_step=check",
                &initial,
            ],
        );
        let table = read(&run.irc.directory.join("irc_first_step.csv"));
        // A row for each of the two sides.
        assert_eq!(table.lines().count(), 3, "{table}");
        for line in table.lines().skip(1) {
            let r: Vec<&str> = line.split(',').collect();
            println!(
                "| {length} | {} | {:.2} | {:.3} | {} | {} | {:.4} | {:.2e} |",
                r[0],
                r[2].parse::<f64>().unwrap(),
                r[3].parse::<f64>().unwrap(),
                r[6],
                r[7],
                r[9].parse::<f64>().unwrap(),
                r[10].parse::<f64>().unwrap()
            );
        }
        let _ = std::fs::remove_dir_all(&run.irc.directory);
    }
    if !full {
        let _ = std::fs::remove_dir_all(&base);
        return;
    }
    // The energy profile over the step size: the energies at the same lengths of the path
    // are compared with those of the path in steps of 0.02 (linear interpolation).
    let profile = |arguments: &[&str]| -> (Vec<(f64, f64)>, usize, Vec<String>) {
        let run = irc(&base, arguments);
        let rows = read(&run.irc.directory.join("irc_profile.csv"))
            .lines()
            .skip(1)
            .map(|l| {
                let v: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
                (v[1], v[3])
            })
            .collect();
        let ends = run
            .irc
            .end
            .as_ref()
            .unwrap()
            .branches
            .iter()
            .map(|b| format!("{:?}", b.end))
            .collect();
        let _ = std::fs::remove_dir_all(&run.irc.directory);
        (rows, run.irc.evaluations, ends)
    };
    let (fine, _, _) = profile(&["-irc", "0.02", "2000", "lqa", "-fc", "1"]);
    let at = |s: f64| -> Option<f64> {
        fine.windows(2).find_map(|w| {
            let ((s0, e0), (s1, e1)) = (w[0], w[1]);
            (s <= s0 && s >= s1 && s0 > s1).then(|| e0 + (e1 - e0) * (s0 - s) / (s0 - s1))
        })
    };
    println!("\n| method | step | evaluations | largest energy difference from the reference (Hartree) | path length | ends |");
    println!("|---|---|---|---|---|---|");
    for (method, step) in [
        ("lqa", "0.05"),
        ("lqa", "0.1"),
        ("lqa", "0.2"),
        ("lqa", "0.5"),
        ("hpc", "0.1"),
        ("hpc", "0.2"),
        ("gs2", "0.1"),
        ("gs2", "0.2"),
        ("rk4", "0.1"),
        ("euler", "0.02"),
        ("dvv", "0.1"),
    ] {
        let mut arguments = vec!["-irc", step, "600", method];
        if method == "rk4" || method == "dvv" {
            arguments.extend(["-irc_opt", "energy_rise=stop"]);
        }
        let (rows, evaluations, ends) = profile(&arguments);
        let worst = rows
            .iter()
            .filter_map(|(s, e)| at(*s).map(|r| (e - r).abs()))
            .fold(0.0, f64::max);
        let length = rows.first().unwrap().0 - rows.last().unwrap().0;
        println!(
            "| {method} | {step} | {evaluations} | {worst:.2e} | {length:.2} | {} |",
            ends.join(", ")
        );
    }
    println!("\nenergy of the transition state: {energy_ts}");
    let _ = std::fs::remove_dir_all(&base);
}
