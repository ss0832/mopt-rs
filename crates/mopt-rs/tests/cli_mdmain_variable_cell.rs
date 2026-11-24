//! `mdmain -vc`: the cell as a dynamical variable at constant pressure, the common spelling of
//! the constant-pressure ensembles of `-mt` (copper, 32 atoms in a stretched cell, the
//! built-in EMT potential): `-vc` alone is `-mt npt_aniso` at 0 GPa, `-vc` with an ensemble of
//! `-mt` its barostat, `-vc_pressure` the pressure in GPa; the cell follows the pressure,
//! every frame has its lattice, what the dynamics of the cell cannot take is refused with its
//! reason, and the results do not depend on the number of threads.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::mdmain::{config_from_cli, run_with, MdmainResult};
use mopt_rs::domain::SpinCharge;
use mopt_rs::io::xyz::read_xyz_trajectory_file;
use mopt_rs::pes::registry::CalculatorRegistry;

fn base(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_md_vc_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Copper, fcc, 2 x 2 x 2 conventional cells stretched by 4 percent.
fn copper(dir: &Path) -> String {
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
                        (f64::from(i) + f[0]) * a,
                        (f64::from(j) + f[1]) * a,
                        (f64::from(k) + f[2]) * a,
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
    let path = dir.join("cu32.xyz");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn run(dir: &Path, file: &str, extra: &[&str]) -> Result<MdmainResult, String> {
    let mut all = vec![
        file,
        "-os",
        "emt",
        "-time",
        "120",
        "-ts",
        "4fs",
        "-temp",
        "300",
        "-md_opt",
        "seed=2",
        "tau_p=200",
    ];
    all.extend(extra.iter().copied());
    let config = config_from_cli(&all).map_err(|e| e.to_string())?;
    let mut runs =
        run_with(&config, dir, None, &CalculatorRegistry::native()).map_err(|e| e.to_string())?;
    Ok(runs.remove(0))
}

fn properties(out: &MdmainResult) -> String {
    std::fs::read_to_string(out.directory.join("md_properties.csv")).unwrap()
}

fn volumes(out: &MdmainResult) -> Vec<f64> {
    properties(out)
        .lines()
        .skip(1)
        .map(|r| r.split(',').nth(10).unwrap().parse::<f64>().unwrap())
        .collect()
}

#[test]
fn vc_is_the_constant_pressure_of_the_ensembles() {
    let dir = base("same");
    let file = copper(&dir);
    // -vc alone: the fully flexible barostat at 0 GPa.
    let vc = run(&dir, &file, &["-vc"]).unwrap();
    let npt = run(&dir, &file, &["-mt", "npt_aniso", "-press", "0"]).unwrap();
    assert_eq!(properties(&vc), properties(&npt));
    let v = volumes(&vc);
    eprintln!("volume {} -> {} Angstrom^3", v[0], v[v.len() - 1]);
    // The stretched crystal pulls itself together.
    assert!(v[v.len() - 1] < 0.97 * v[0], "{v:?}");
    // Every frame with its lattice, the cell of the last one that of the last volume.
    let frames = read_xyz_trajectory_file(&vc.trajectory, SpinCharge::default(), None).unwrap();
    assert!(frames.frames.len() > 2);
    assert!(frames.frames.iter().all(|f| f.system.has_pbc()));
    let first = frames.frames[0]
        .system
        .active_cell()
        .unwrap()
        .volume_bohr3();
    let last = frames.frames[frames.frames.len() - 1]
        .system
        .active_cell()
        .unwrap()
        .volume_bohr3();
    assert!(last < 0.97 * first, "{first} {last}");
    // With an ensemble of -mt: its barostat.
    let both = run(&dir, &file, &["-mt", "npt", "-vc"]).unwrap();
    let mtk = run(&dir, &file, &["-mt", "npt", "-press", "0"]).unwrap();
    assert_eq!(properties(&both), properties(&mtk));
    // -vc_pressure in GPa is -press in kPa.
    let gpa = run(&dir, &file, &["-vc", "-vc_pressure", "2.0"]).unwrap();
    let kpa = run(&dir, &file, &["-mt", "npt_aniso", "-press", "2000000"]).unwrap();
    let (a, b) = (volumes(&gpa), volumes(&kpa));
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() <= 1e-8 * x.abs(), "{x} {y}");
    }
    // A higher pressure, a smaller cell.
    assert!(a[a.len() - 1] < v[v.len() - 1], "{a:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_the_dynamics_of_the_cell_cannot_take_is_refused_with_its_reason() {
    let dir = base("refused");
    let file = copper(&dir);
    for (extra, expected) in [
        (
            vec!["-mt", "langevin", "-vc"],
            "an ensemble without a barostat",
        ),
        (vec!["-vc", "-vc_fix", "volume"], "-vc_fix"),
        (vec!["-vc", "-uc", "volume()"], "-vc_fix"),
        (vec!["-vc", "-vc_weight", "1.0"], "-vc_weight with mdmain"),
        (vec!["-vc", "-vc_afir", "100", "a"], "-vc_afir with mdmain"),
        (
            vec!["-vc", "-vc_pressure", "1.0", "-press", "100"],
            "given twice",
        ),
        (vec!["-vc_pressure", "1.0"], "needs -vc"),
    ] {
        let message = run(&dir, &file, &extra).unwrap_err();
        assert!(message.contains(expected), "{extra:?}: {message}");
    }
    let vars = dir.join("x.txt");
    std::fs::write(&vars, "x -0.558\ny 1.442\n").unwrap();
    let message = config_from_cli(&[vars.to_str().unwrap(), "-os", "mb2d", "-vc"])
        .unwrap_err()
        .to_string();
    assert!(message.contains("plain variables"), "{message}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_dynamics_of_the_cell_does_not_depend_on_the_threads() {
    let dir = base("threads");
    let file = copper(&dir);
    let one = run(&dir, &file, &["-vc", "-core", "1"]).unwrap();
    let four = run(&dir, &file, &["-vc", "-core", "4"]).unwrap();
    assert_eq!(properties(&one), properties(&four));
    let _ = std::fs::remove_dir_all(&dir);
}
