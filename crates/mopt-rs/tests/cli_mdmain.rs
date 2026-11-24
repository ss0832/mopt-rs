//! `mdmain`: runs, files, restart, and what is refused.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mopt_rs::cli::mdmain::{config_from_cli, run_with, run_with_observers, MdmainConfig};
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem};
use mopt_rs::error::MoptResult;
use mopt_rs::md::MdStepReport;
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::journal::{read_journal, JOURNAL_FILE};
use mopt_rs::workflow::md::FnMdObserver;
use mopt_rs::workflow::observers::STOP_FILE;

const AR7: [[f64; 3]; 7] = [
    [1.9500, 0.0000, 0.0000],
    [0.6026, 1.8546, 0.0300],
    [-1.5776, 1.1462, 0.0700],
    [-1.5776, -1.1462, 0.1100],
    [0.6026, -1.8546, 0.1500],
    [0.0000, 0.0000, 2.1000],
    [0.2000, 0.0000, -2.0000],
];

struct Counting(Arc<AtomicUsize>);

impl Calculator for Counting {
    fn name(&self) -> &str {
        "counting_lj"
    }
    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        self.0.fetch_add(1, Ordering::SeqCst);
        LennardJones.evaluate(sys, req)
    }
}

fn registry(count: &Arc<AtomicUsize>) -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    let c = Arc::clone(count);
    r.register("counting", move |_| Ok(Box::new(Counting(Arc::clone(&c)))));
    r
}

fn base(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("mopt_md_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// The Ar7 cluster (Angstrom; 3.8 Angstrom apart is the minimum of `lj`).
fn ar7(base: &Path) -> String {
    let path = base.join("ar7.xyz");
    let mut text = String::from("7\n\n");
    for p in AR7 {
        text.push_str(&format!(
            "Ar {} {} {}\n",
            1.95 * p[0],
            1.95 * p[1],
            1.95 * p[2]
        ));
    }
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn an_nve_run_of_a_cluster_writes_its_files_and_keeps_its_energy() {
    let base = base("nve");
    let xyz = ar7(&base);
    let config = config_from_cli(&[
        &xyz,
        "-os",
        "lj",
        "-mt",
        "nve",
        "-time",
        "400",
        "-ts",
        "2fs",
        "-temp",
        "40",
        "-md_opt",
        "seed=3",
        "write_every=20",
        "structure_every=100",
    ])
    .unwrap();
    let runs = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    let dir = &runs[0].directory;
    let rows = lines(&dir.join("md_properties.csv"));
    assert_eq!(rows.len(), 402, "header and 401 steps");
    let conserved: Vec<f64> = rows[1..]
        .iter()
        .map(|r| r.split(',').nth(6).unwrap().parse().unwrap())
        .collect();
    let drift = conserved
        .iter()
        .map(|c| (c - conserved[0]).abs())
        .fold(0.0_f64, f64::max);
    // 40 K over 15 degrees of freedom is 1.4e-3 Hartree of kinetic energy.
    assert!(drift < 2e-5, "{drift}");
    let traj = lines(&runs[0].trajectory);
    assert_eq!(traj.len(), 21 * 9, "a frame every 20 steps");
    for name in [
        "energy_profile.csv",
        "gradient_profile.csv",
        "tot_energy_profile.csv",
        "energy_profile_kcalmol.csv",
        "energy_plot_.png",
        "plot_temperature_.png",
        "final_velocities.txt",
        "settings.json",
        "mopt.log",
    ] {
        assert!(dir.join(name).is_file(), "{name}");
    }
    let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap();
    assert!(log.contains("hamiltonian :") && log.contains("Instantaneous_temperature:"));
    assert!(log.contains("Degrees of freedom of the temperature: 15"));
    let settings = std::fs::read_to_string(dir.join("settings.json")).unwrap();
    assert!(settings.contains("seed=3") && settings.contains("structure_every=100"));
    // The state every 100 steps: structure, velocities, and energies.
    assert_eq!(log.matches("CARTESIAN COORDINATES (unit ang.):").count(), 5);
    let at_100 = &log[log.find("# STEP. 100 (").unwrap()..log.find("# STEP. 101 (").unwrap()];
    for label in [
        "VELOCITIES (unit a.u.: Bohr per atomic unit of time):",
        "ENERGIES (unit hartree):",
        "KINETIC ENERGY         :",
        "ENERGY                 :",
        "BIAS  ENERGY           :",
        "HAMILTONIAN            :",
        "CONSERVED QUANTITY     :",
    ] {
        assert!(at_100.contains(label), "{label}: {at_100}");
    }
    assert_eq!(
        at_100.lines().filter(|l| l.starts_with("Ar   ")).count(),
        14
    );
    assert!(
        !log[log.find("# STEP. 101 (").unwrap()..log.find("# STEP. 102 (").unwrap()]
            .contains("VELOCITIES")
    );
    let _ = std::fs::remove_dir_all(&base);
}

fn langevin(xyz: &str) -> MdmainConfig {
    config_from_cli(&[
        xyz,
        "-os",
        "counting",
        "-mt",
        "langevin",
        "-time",
        "60",
        "-ts",
        "4fs",
        "-temp",
        "60",
        "-md_opt",
        "seed=11",
        "write_every=1",
    ])
    .unwrap()
}

#[test]
fn a_stochastic_run_restarts_from_its_journal_bit_for_bit() {
    let base = base("restart");
    let xyz = ar7(&base);
    let count = Arc::new(AtomicUsize::new(0));
    let full = run_with(&langevin(&xyz), &base, None, &registry(&count)).unwrap();
    let total = count.load(Ordering::SeqCst);
    assert_eq!(total, 61);
    let dir = full[0].directory.clone();
    let reference = std::fs::read_to_string(&full[0].trajectory).unwrap();
    let journal = dir.join(JOURNAL_FILE);
    assert_eq!(read_journal(&journal).unwrap().0.len(), total);

    // Stopped after 25 steps by a STOP file; the restart continues to the same end.
    std::fs::remove_file(&journal).unwrap();
    let mut cfg = langevin(&xyz);
    cfg.restart_dir = Some(dir.clone());
    let stop_at = dir.join(STOP_FILE);
    let mut writer = FnMdObserver::new(|r: &MdStepReport| {
        if r.step == 25 {
            std::fs::write(&stop_at, "").unwrap();
        }
        Ok(true)
    });
    count.store(0, Ordering::SeqCst);
    let stopped =
        run_with_observers(&cfg, &base, None, &registry(&count), &mut [&mut writer]).unwrap();
    assert!(stopped[0].interrupted);
    assert_eq!(count.load(Ordering::SeqCst), 26);
    count.store(0, Ordering::SeqCst);
    let done = run_with(&cfg, &base, None, &registry(&count)).unwrap();
    assert!(!done[0].interrupted);
    assert_eq!(
        count.load(Ordering::SeqCst),
        total - 26,
        "only the rest is computed"
    );
    assert_eq!(
        std::fs::read_to_string(&done[0].trajectory).unwrap(),
        reference,
        "the same trajectory"
    );
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn plain_variables_freeze_the_saddle_direction_of_muller_brown() {
    let base = base("mb");
    let start = base.join("saddle.txt");
    std::fs::write(&start, "x -0.822002\ny 0.624313\n").unwrap();
    let s = start.to_string_lossy().into_owned();
    let config = config_from_cli(&[
        &s,
        "-os",
        "mb2d",
        "-mt",
        "nve",
        "-time",
        "800",
        "-ts",
        "0.01",
        "-temp",
        "200",
        "-md_opt",
        "freeze=negative",
        "seed=4",
    ])
    .unwrap();
    let runs = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    let traj = lines(&runs[0].trajectory);
    assert_eq!(traj[0], "step,time,x,y");
    let log = std::fs::read_to_string(runs[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("Frozen modes at step 0: 1 mode(s)"), "{log}");
    assert!(log.contains("Degrees of freedom of the temperature: 1"));
    // The state of the variables every 10 steps without structure_every.
    assert_eq!(log.matches("VARIABLES                VALUE").count(), 81);
    let settings = std::fs::read_to_string(runs[0].directory.join("settings.json")).unwrap();
    assert!(settings.contains("structure_every=10"), "{settings}");
    for row in &traj[1..] {
        let v: Vec<f64> = row.split(',').skip(2).map(|x| x.parse().unwrap()).collect();
        let d = ((v[0] + 0.822002).powi(2) + (v[1] - 0.624313).powi(2)).sqrt();
        assert!(d < 0.15, "{row}");
    }
    // What needs atoms is refused for plain variables.
    let bias = config_from_cli(&[&s, "-os", "mb2d", "-kp", "1.0", "1.0", "1,2"]).unwrap();
    assert!(run_with(&bias, &base, None, &CalculatorRegistry::native()).is_err());
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_metal_in_a_periodic_cell_follows_the_pressure() {
    let base = base("npt");
    // Copper, fcc, 2 x 2 x 2 conventional cells stretched by 4 percent.
    let a = 3.61 * 1.04;
    let mut atoms = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    atoms.push([
                        (i as f64 + f[0]) * a,
                        (j as f64 + f[1]) * a,
                        (k as f64 + f[2]) * a,
                    ]);
                }
            }
        }
    }
    let l = 2.0 * a;
    let mut text = format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" pbc=\"T T T\"\n",
        atoms.len()
    );
    for p in &atoms {
        text.push_str(&format!("Cu {} {} {}\n", p[0], p[1], p[2]));
    }
    let path = base.join("cu32.xyz");
    std::fs::write(&path, text).unwrap();
    let p = path.to_string_lossy().into_owned();
    let config = config_from_cli(&[
        &p,
        "-os",
        "emt",
        "-mt",
        "npt_berendsen",
        "-time",
        "300",
        "-ts",
        "4fs",
        "-temp",
        "300",
        "-press",
        "101.3",
        "-md_opt",
        "seed=2",
        "tau_p=200",
        "compressibility=0.007",
    ])
    .unwrap();
    let runs = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    let rows = lines(&runs[0].directory.join("md_properties.csv"));
    let volume = |r: &str| r.split(',').nth(10).unwrap().parse::<f64>().unwrap();
    let (v0, v1) = (volume(&rows[1]), volume(rows.last().unwrap()));
    // The stretched crystal pulls itself together.
    assert!(v1 < 0.95 * v0, "{v0} -> {v1}");
    let traj = std::fs::read_to_string(&runs[0].trajectory).unwrap();
    assert!(traj.contains("Lattice="));
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn collective_variables_are_written_restrained_and_analyzed() {
    let base = base("cv");
    let xyz = ar7(&base);
    let config = config_from_cli(&[
        &xyz,
        "-os",
        "lj",
        "-mt",
        "bussi",
        "-time",
        "400",
        "-ts",
        "4fs",
        "-temp",
        "20",
        "-md_opt",
        "seed=2",
        "-cv",
        "distance:1,2",
        "angle:2,1,3",
        "coordination:1/2,3,4,5,6,7:r0=4.5",
        "-cv_bias",
        "distance:1,2;center=7.6;k=0.05",
        "-analysis",
        "statistics",
        "acf",
    ])
    .unwrap();
    let runs = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    let dir = &runs[0].directory;
    let rows = lines(&dir.join("ar7_cv.csv"));
    assert_eq!(rows.len(), 402);
    assert!(rows[0].starts_with("step,time_fs,distance:1 2,angle:2 1 3,"));
    let d: Vec<f64> = rows[1..]
        .iter()
        .map(|r| r.split(',').nth(2).unwrap().parse().unwrap())
        .collect();
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    // The restraint holds the distance near 7.6 Bohr.
    assert!((mean - 7.6).abs() < 0.3, "{mean}");
    let stats =
        std::fs::read_to_string(dir.join("trajectory_analysis").join("statistics.csv")).unwrap();
    assert!(stats.contains("cv distance:1 2,a.u.,"));
    let acf = lines(&dir.join("trajectory_analysis").join("acf_properties.csv"));
    assert!(acf[0].contains("cv coordination:1/2 3 4 5 6 7:r0=4.5"));
    assert!(config_from_cli(&[&xyz, "-cv", "angle:1,2"]).is_err());
    assert!(config_from_cli(&[&xyz, "-cv_bias", "distance:1,2;k=0.1"]).is_err());
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn umbrella_sampling_of_a_distance_gives_a_profile() {
    let base = base("umbrella");
    let path = base.join("ar2.xyz");
    std::fs::write(&path, "2\n\nAr 0.0 0.0 0.0\nAr 3.8 0.0 0.0\n").unwrap();
    let p = path.to_string_lossy().into_owned();
    let config = config_from_cli(&[
        &p,
        "-os",
        "lj",
        "-mt",
        "langevin",
        "-time",
        "300",
        "-ts",
        "10fs",
        "-temp",
        "40",
        "-md_opt",
        "seed=9",
        "-umbrella",
        "cv=dist(1,2)",
        "from=3.6*ang",
        "to=4.4*ang",
        "windows=5",
        "k=0.02",
        "bins=10",
    ])
    .unwrap();
    let runs = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    let dir = &runs[0].directory;
    let profile = lines(&dir.join("umbrella_profile.csv"));
    assert_eq!(profile.len(), 11);
    assert_eq!(lines(&dir.join("umbrella_windows.csv")).len(), 6);
    assert!(dir.join("umbrella_window_4/md_properties.csv").is_file());
    assert!(dir.join("umbrella_profile.png").is_file());
    let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap();
    assert!(log.contains("Free-energy profile"), "{log}");
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn dynamics_on_a_surface_that_combines_two_electronic_states() {
    let base = base("mf");
    let path = base.join("hcn.xyz");
    std::fs::write(
        &path,
        "3\n\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n",
    )
    .unwrap();
    let p = path.to_string_lossy().into_owned();
    let config = config_from_cli(&[
        &p, "-os", "gfn0xtb", "-mf", "seam", "-mt", "nve", "-time", "4", "-ts", "0.25fs", "-temp",
        "100", "-md_opt", "seed=1",
    ])
    .unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let mut registry = CalculatorRegistry::native();
    let c = Arc::clone(&count);
    registry.register("gfn0xtb_counted", move |q| {
        struct Counted(Box<dyn Calculator>, Arc<AtomicUsize>);
        impl Calculator for Counted {
            fn name(&self) -> &str {
                "gfn0xtb"
            }
            fn supports_excited_states(&self) -> bool {
                true
            }
            fn uses_electrons(&self) -> bool {
                true
            }
            fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
                self.1.fetch_add(1, Ordering::SeqCst);
                self.0.evaluate(sys, req)
            }
        }
        let inner = CalculatorRegistry::native().create("gfn0xtb", q.options, q.system)?;
        Ok(Box::new(Counted(inner, Arc::clone(&c))))
    });
    let mut config = config;
    config.calculator = "gfn0xtb_counted".to_string();
    let runs = run_with(&config, &base, None, &registry).unwrap();
    // Two states at every step.
    assert_eq!(count.load(Ordering::SeqCst), 10);
    let log = std::fs::read_to_string(runs[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("Mode: Model Function (seam)"), "{log}");
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn what_cannot_work_together_is_refused_with_a_reason() {
    let base = base("refuse");
    let xyz = ar7(&base);
    let npt_mf = config_from_cli(&[&xyz, "-os", "lj", "-mt", "npt", "-mf", "seam", "0", "3"]);
    assert!(npt_mf.unwrap_err().to_string().contains("stress"));
    let npt_isolated = config_from_cli(&[&xyz, "-os", "lj", "-mt", "npt", "-time", "2"]).unwrap();
    let e = run_with(&npt_isolated, &base, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(e.to_string().contains("periodic"), "{e}");
    assert!(config_from_cli(&[&xyz, "-mt", "gauss"]).is_err());
    let _ = std::fs::remove_dir_all(&base);
}
