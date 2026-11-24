//! `relaxedscan` from the command line: MultiOptPy's bond scan with its files and a restart,
//! the differences from MultiOptPy, a bias potential, a rigid scan, an ONIOM scan that keeps
//! its microiterations, a scan in a periodic cell, and the settings files (maps and ReSPer
//! coordinates: `cli_relaxedscan_maps.rs`).
//!
//! MultiOptPy: `Entrypoints/relaxed_scan.py:run_relaxedscan`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mopt_rs::cli::relaxedscan::{config_from_cli, run_with};
use mopt_rs::domain::{AtomId, EvalOutput, EvalRequest, MolecularSystem, SpinCharge, Units};
use mopt_rs::error::MoptResult;
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::pes::Calculator;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_rscan_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

const WATER: &str = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n";

const ETHANE: &str = "8\nethane\nC 0.000 0.000 0.000\nC 1.530 0.000 0.000\n\
H -0.390 1.030 0.000\nH -0.390 -0.510 0.890\nH -0.390 -0.510 -0.890\n\
H 1.920 -1.030 0.000\nH 1.920 0.510 0.890\nH 1.920 0.510 -0.890\n";

const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

fn bond(s: &MolecularSystem, a: usize, b: usize) -> f64 {
    s.distance(AtomId::new(a), AtomId::new(b)).value() * Units::BOHR_TO_ANGSTROM
}

/// GFN-FF that writes a `STOP` file into `stop_dir` after `limit` evaluations.
struct Stopping {
    inner: Box<dyn Calculator>,
    count: Arc<AtomicUsize>,
    limit: usize,
    stop_dir: PathBuf,
}

impl Calculator for Stopping {
    fn name(&self) -> &str {
        "stopping_gfnff"
    }
    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if self.count.fetch_add(1, Ordering::SeqCst) + 1 == self.limit {
            std::fs::write(self.stop_dir.join("STOP"), "").unwrap();
        }
        self.inner.evaluate(sys, req)
    }
}

fn stopping(count: &Arc<AtomicUsize>, limit: usize, stop_dir: &Path) -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    let c = Arc::clone(count);
    let dir = stop_dir.to_path_buf();
    r.register("stopping", move |q| {
        Ok(Box::new(Stopping {
            inner: CalculatorRegistry::native().create(
                "gfnff",
                &CalculatorOptions::default(),
                q.system,
            )?,
            count: Arc::clone(&c),
            limit,
            stop_dir: dir.clone(),
        }))
    });
    r
}

#[test]
fn a_bond_scan_writes_multioptpy_files_and_restarts_where_it_stopped() {
    let dir = scratch("bond");
    let input = file(&dir, "water.xyz", WATER);
    let args = [
        input.as_str(),
        "-os",
        "stopping",
        "-scan",
        "bond",
        "1,2",
        "0.90,1.30",
        "-nsample",
        "5",
    ];
    // The whole scan.
    let count = Arc::new(AtomicUsize::new(0));
    let mut whole = config_from_cli(&args).unwrap();
    whole.restart_dir = Some(dir.join("whole"));
    let full = run_with(&whole, &dir, None, &stopping(&count, usize::MAX, &dir)).unwrap();
    assert!(!full.interrupted);
    assert_eq!(full.points.len(), 5);
    let evaluations = count.load(Ordering::SeqCst);
    let scan = dir.join("whole");
    for k in 1..=5 {
        let s = read_xyz_file(
            scan.join(format!("water_RelaxedScan_{k}.xyz")),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let target = 0.9 + 0.1 * (k - 1) as f64;
        assert!(
            (bond(&s, 0, 1) - target).abs() < 1e-6,
            "{k}: {}",
            bond(&s, 0, 1)
        );
    }
    // MultiOptPy's energy profile: two header lines, then one row per point.
    let profile = std::fs::read_dir(&scan)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.to_string_lossy().contains("_scan_energy_profile_"))
        .unwrap();
    let text = std::fs::read_to_string(profile).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "bond,energy,bias_energy");
    assert_eq!(lines[1], "1_2,(hartree),(hartree)");
    assert_eq!(lines.len(), 7);
    // The energy rises away from the equilibrium bond (about 0.97 Angstrom).
    let e: Vec<f64> = full.points.iter().map(|p| p.energy).collect();
    assert!(e[4] > e[3] && e[3] > e[2] && e[0] > e[1], "{e:?}");
    for name in [
        "scan_points.csv",
        "scan_state.json",
        "settings.json",
        "water_scan_trajectory.xyz",
        "relaxedscan_energy_profile.png",
        "relaxedscan_energy_profile.svg",
        "relaxedscan_report.html",
        "point_003/mopt.log",
        "point_003/evaluations.journal",
    ] {
        assert!(scan.join(name).is_file(), "{name}");
    }
    let page = std::fs::read_to_string(scan.join("relaxedscan_report.html")).unwrap();
    // One file: nothing is loaded from elsewhere.
    for outside in ["src=\"http", "href=\"http", "@import", "url(http"] {
        assert!(!page.contains(outside), "{outside}");
    }
    assert!(page.contains("id=\"models\"") && page.contains("<svg"));

    // Stopped in the middle of a point, then continued: the same energies, and the points
    // already done are not calculated again.
    let count = Arc::new(AtomicUsize::new(0));
    let mut first = config_from_cli(&args).unwrap();
    first.restart_dir = Some(dir.join("stopped"));
    let limit = evaluations / 2;
    let stopped = run_with(
        &first,
        &dir,
        None,
        &stopping(&count, limit, &dir.join("stopped")),
    )
    .unwrap();
    assert!(stopped.interrupted);
    assert!(stopped.points.len() < 5);
    let before = count.load(Ordering::SeqCst);
    let again = run_with(
        &first,
        &dir,
        None,
        &stopping(&count, usize::MAX, &dir.join("stopped")),
    )
    .unwrap();
    assert!(!again.interrupted);
    assert_eq!(again.points.len(), 5);
    for (a, b) in again.points.iter().zip(&full.points) {
        assert_eq!(a.point, b.point);
        assert!(
            (a.energy - b.energy).abs() < 1e-9,
            "{} {}",
            a.energy,
            b.energy
        );
    }
    let total = count.load(Ordering::SeqCst);
    assert!(
        total < evaluations + before / 2,
        "{total} evaluations in two parts, {evaluations} in one"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The differences from MultiOptPy's `relaxedscan` (see migration.md): the constraints of
/// `-pc` hold during the scan (MultiOptPy replaces them by the scan), a point that does not
/// converge is recorded with the structure its energy belongs to, and nothing is written
/// outside the directory of the scan.
#[test]
fn the_constraints_of_the_run_hold_and_every_file_stays_in_the_scan() {
    let dir = scratch("differences");
    let input = file(&dir, "water.xyz", WATER);
    let mut config = config_from_cli(&[
        input.as_str(),
        "-os",
        "gfnff",
        "-scan",
        "bond",
        "1,2",
        "0.90,1.10",
        "-nsample",
        "3",
        "-pc",
        "bond",
        "1,3",
        "-ns",
        "2",
    ])
    .unwrap();
    config.restart_dir = Some(dir.join("scan"));
    let registry = CalculatorRegistry::native();
    let out = run_with(&config, &dir, None, &registry).unwrap();
    let start = read_xyz_file(&input, SpinCharge::default(), None).unwrap();
    let calc = registry
        .create("gfnff", &CalculatorOptions::default(), &start)
        .unwrap();
    // Two iterations are not enough: points are recorded as not converged.
    assert!(out.points.iter().any(|p| !p.converged));
    for p in &out.points {
        let s = read_xyz_file(dir.join("scan").join(&p.file), SpinCharge::default(), None).unwrap();
        assert!((bond(&s, 0, 2) - bond(&start, 0, 2)).abs() < 1e-6);
        let e = calc
            .evaluate(&s, &EvalRequest::ENERGY_GRAD)
            .unwrap()
            .energy
            .value();
        assert!((e - p.energy).abs() < 1e-8, "{e} {}", p.energy);
    }
    let outside: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(outside.len(), 2, "{outside:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_scan_on_a_surface_with_a_bias_potential() {
    let dir = scratch("bias");
    let input = file(&dir, "water.xyz", WATER);
    // A harmonic restraint on the other O-H bond (-kp, keep potential).
    let mut config = config_from_cli(&[
        input.as_str(),
        "-os",
        "gfnff",
        "-scan",
        "bond",
        "1,2",
        "0.90,1.10",
        "-nsample",
        "3",
        "-kp",
        "0.5",
        "1.2",
        "1,3",
    ])
    .unwrap();
    config.restart_dir = Some(dir.join("scan"));
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out.points.iter().all(|p| p.converged));
    // The restraint pulls the free bond out: the energy with it lies above the energy alone.
    for p in &out.points {
        assert!(p.biased > p.energy + 1e-6, "{} {}", p.biased, p.energy);
        let s = read_xyz_file(dir.join("scan").join(&p.file), SpinCharge::default(), None).unwrap();
        assert!(bond(&s, 0, 2) > 1.0, "{}", bond(&s, 0, 2));
    }
    let svg = std::fs::read_to_string(dir.join("scan/relaxedscan_energy_profile.svg")).unwrap();
    assert!(svg.contains("energy with the bias potentials"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_rigid_torsion_turns_the_methyl_group_without_relaxing_it() {
    let dir = scratch("rigid");
    let input = file(&dir, "ethane.xyz", ETHANE);
    let mut config = config_from_cli(&[
        input.as_str(),
        "-os",
        "gfnff",
        "-scan",
        "dihedral",
        "3,1,2,6",
        "-180,0",
        "-nsample",
        "7",
        "-rigid",
    ])
    .unwrap();
    config.restart_dir = Some(dir.join("scan"));
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(out.points.len(), 7);
    let start = read_xyz_file(&input, SpinCharge::default(), None).unwrap();
    let mut energies = Vec::new();
    for k in 1..=7 {
        let s = read_xyz_file(
            dir.join(format!("scan/ethane_RelaxedScan_{k}.xyz")),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        for (a, b) in [(0, 1), (1, 5), (1, 6), (5, 6), (0, 2)] {
            assert!((bond(&s, a, b) - bond(&start, a, b)).abs() < 1e-6);
        }
        energies.push(out.points[k - 1].energy);
    }
    // Eclipsed (0 and 120 degrees from the staggered start) above staggered.
    assert!(
        energies[2] > energies[0] + 1e-3 && energies[6] > energies[4] + 1e-3,
        "{energies:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_oniom_scan_keeps_its_microiterations() {
    let dir = scratch("oniom");
    let input = file(&dir, "ethanol.xyz", ETHANOL);
    let mut config = config_from_cli(&[
        input.as_str(),
        "-os",
        "gfn0xtb",
        "-oniom",
        "2-4,8,9",
        "auto",
        "gfnff",
        "-scan",
        "bond",
        "2,3",
        "1.40,1.50",
        "-nsample",
        "3",
    ])
    .unwrap();
    config.restart_dir = Some(dir.join("scan"));
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out.points.iter().all(|p| p.converged));
    let log = std::fs::read_to_string(dir.join("scan/point_002/mopt.log")).unwrap();
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    for k in 1..=3 {
        let s = read_xyz_file(
            dir.join(format!("scan/ethanol_RelaxedScan_{k}.xyz")),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert_eq!(s.n_atoms(), 9);
        assert!((bond(&s, 1, 2) - (1.40 + 0.05 * (k - 1) as f64)).abs() < 1e-6);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Copper, a 2 x 2 x 2 supercell of the face-centred cubic cell.
fn copper() -> String {
    let a = 3.61;
    let mut sites = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    sites.push([
                        a * (f64::from(i) + f[0]),
                        a * (f64::from(j) + f[1]),
                        a * (f64::from(k) + f[2]),
                    ]);
                }
            }
        }
    }
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
        sites.len(),
        l = 2.0 * a
    );
    for s in &sites {
        text.push_str(&format!("Cu {:.8} {:.8} {:.8}\n", s[0], s[1], s[2]));
    }
    text
}

#[test]
fn a_scan_in_a_periodic_cell_measures_the_nearest_image() {
    let dir = scratch("periodic");
    let input = file(&dir, "cu.xyz", &copper());
    // Atom 1 at the origin and the atom of the site (1.5 a, 1.5 a, 0), whose image at
    // (-0.5 a, -0.5 a, 0) is a nearest neighbour across the boundary of the cell.
    let start = read_xyz_file(&input, SpinCharge::default(), None).unwrap();
    let far = (0..start.n_atoms())
        .find(|&k| {
            let p = start.position(AtomId::new(k)).value();
            let a = 3.61 * Units::ANGSTROM_TO_BOHR;
            (p[0] - 1.5 * a).abs() < 1e-6 && (p[1] - 1.5 * a).abs() < 1e-6 && p[2].abs() < 1e-6
        })
        .unwrap();
    let pair = format!("1,{}", far + 1);
    let mut config = config_from_cli(&[
        input.as_str(),
        "-os",
        "emt",
        "-scan",
        "bond",
        pair.as_str(),
        "2.45,2.75",
        "-nsample",
        "3",
    ])
    .unwrap();
    config.restart_dir = Some(dir.join("scan"));
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(out.points.len(), 3);
    for k in 1..=3 {
        let path = dir.join(format!("scan/cu_RelaxedScan_{k}.xyz"));
        assert!(std::fs::read_to_string(&path).unwrap().contains("Lattice="));
        let s = read_xyz_file(&path, SpinCharge::default(), None).unwrap();
        assert!(s.has_pbc());
        let target = 2.45 + 0.15 * (k - 1) as f64;
        assert!(
            (bond(&s, 0, far) - target).abs() < 1e-6,
            "{}",
            bond(&s, 0, far)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn usage_errors_and_settings_files() {
    let dir = scratch("usage");
    let water = file(&dir, "water.xyz", WATER);
    let mb = file(&dir, "mb.txt", "x -0.5\ny 1.4\n");
    let registry = CalculatorRegistry::native();
    for (input, args) in [
        (&water, vec!["-os", "gfnff", "-scan", "var", "1", "0,1"]),
        (&mb, vec!["-os", "mb2d", "-scan", "bond", "1,2", "1,2"]),
        (&mb, vec!["-os", "mb2d", "-scan", "var", "z", "0,1"]),
    ] {
        let mut full = vec![input.as_str()];
        full.extend(args);
        let mut config = config_from_cli(&full).unwrap();
        config.restart_dir = Some(dir.join("refused"));
        assert!(
            run_with(&config, &dir, None, &registry).is_err(),
            "{full:?}"
        );
    }
    // --dump-config writes the settings, which --config reads back.
    let dumped = dir.join("scan.json");
    let code = mopt_rs::cli::run_command(
        "relaxedscan",
        &[
            water.clone(),
            "-scan".into(),
            "angle".into(),
            "2,1,3".into(),
            "100,110".into(),
            "-grid".into(),
            "--dump-config".into(),
            dumped.to_string_lossy().into_owned(),
        ],
    );
    assert_eq!(code, 0);
    let text = std::fs::read_to_string(&dumped).unwrap();
    assert!(
        text.contains("\"scan_tgt\"") && text.contains("\"scan_grid\": true"),
        "{text}"
    );
    assert_eq!(
        mopt_rs::cli::run_command("relaxedscan", &[water, "-scan".into(), "bond".into()]),
        2
    );
    let _ = std::fs::remove_dir_all(&dir);
}
