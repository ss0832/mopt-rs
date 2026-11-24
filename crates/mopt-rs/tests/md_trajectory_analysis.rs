//! The trajectory analysis of `mdmain` on real systems: copper (EAM) as a crystal and as a
//! liquid in a periodic cell, diamond silicon (Tersoff), and water (GFN-FF) against the
//! frequencies of `freqmain`; and `-analyze_only` on a finished run.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::analysis::trajectory::peaks;
use mopt_rs::cli::mdmain::{config_from_cli, run_with};
use mopt_rs::cli::{freqmain, optmain};
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mdana_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A crystal of `k^3` conventional cubic cells of side `a` with the fractional `basis`.
fn crystal(dir: &Path, name: &str, element: &str, a: f64, k: usize, basis: &[[f64; 3]]) -> String {
    let mut atoms = Vec::new();
    for x in 0..k {
        for y in 0..k {
            for z in 0..k {
                for b in basis {
                    atoms.push([
                        (x as f64 + b[0]) * a,
                        (y as f64 + b[1]) * a,
                        (z as f64 + b[2]) * a,
                    ]);
                }
            }
        }
    }
    let l = k as f64 * a;
    let mut text = format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" pbc=\"T T T\"\n",
        atoms.len()
    );
    for p in &atoms {
        text.push_str(&format!("{element} {} {} {}\n", p[0], p[1], p[2]));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

const FCC: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
];

const DIAMOND: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.25],
    [0.75, 0.25, 0.75],
    [0.25, 0.75, 0.75],
];

/// The columns of a CSV file by their names.
fn column(path: &Path, name: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let k = header
        .iter()
        .position(|h| *h == name)
        .unwrap_or_else(|| panic!("no column {name} in {}", path.display()));
    lines
        .map(|l| l.split(',').nth(k).unwrap().parse::<f64>().unwrap())
        .collect()
}

/// The value in `column` of the row whose first field is `row`.
fn cell(path: &Path, row: &str, name: &str) -> f64 {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let k = header.iter().position(|h| *h == name).unwrap();
    lines
        .map(|l| l.split(',').collect::<Vec<_>>())
        .find(|f| f[0] == row)
        .map(|f| f[k].parse::<f64>().unwrap())
        .unwrap_or_else(|| panic!("no row {row} in {}", path.display()))
}

/// The largest value of `ys` with `xs` in `(lo, hi)` and where it is.
fn peak_in(xs: &[f64], ys: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    xs.iter()
        .zip(ys)
        .filter(|(x, _)| **x > lo && **x < hi)
        .fold((0.0, f64::NEG_INFINITY), |best, (x, y)| {
            if *y > best.1 {
                (*x, *y)
            } else {
                best
            }
        })
}

fn at(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let k = xs.iter().position(|v| *v >= x).unwrap();
    ys[k]
}

fn run(args: &[&str], base: &Path) -> PathBuf {
    let config = config_from_cli(args).unwrap();
    let runs = run_with(&config, base, None, &CalculatorRegistry::native()).unwrap();
    runs[0].directory.clone()
}

#[test]
fn copper_as_a_crystal_and_as_a_liquid() {
    let base = scratch("cu");
    // EAM copper (Zhou et al. 2004), 108 atoms; the crystal at its lattice constant.
    let a = 3.615;
    let solid = crystal(&base, "cu_solid.xyz", "Cu", a, 3, &FCC);
    let dir = run(
        &[
            &solid,
            "-os",
            "eam",
            "-mt",
            "bussi",
            "-time",
            "600",
            "-ts",
            "2fs",
            "-temp",
            "300",
            "-md_opt",
            "seed=3",
            "write_every=2",
            "-analysis",
            "statistics",
            "rdf",
            "msd",
            "skip=200",
        ],
        &base,
    );
    let out = dir.join("trajectory_analysis");
    let r = column(&out.join("rdf.csv"), "r_angstrom");
    let g = column(&out.join("rdf.csv"), "g_Cu-Cu");
    let n = column(&out.join("rdf.csv"), "n_Cu-Cu");
    let (first, g1) = peak_in(&r, &g, 2.0, 3.0);
    let (second, _) = peak_in(&r, &g, 3.2, 4.0);
    let twelve = at(&r, &n, 3.1);
    eprintln!("Cu solid: peaks {first} ({g1}), {second}; neighbours within 3.1: {twelve}");
    // The shells of fcc at a / sqrt 2 and a, twelve neighbours in the first.
    assert!((first - a / 2.0_f64.sqrt()).abs() < 0.1, "{first}");
    assert!((second - a).abs() < 0.12, "{second}");
    assert!((twelve - 12.0).abs() < 0.3, "{twelve}");
    let d_solid = cell(&out.join("diffusion.csv"), "all", "D_angstrom2_per_ps");
    let t = cell(&out.join("statistics.csv"), "temperature_K", "mean");
    let t_err = cell(
        &out.join("statistics.csv"),
        "temperature_K",
        "error_of_mean",
    );
    eprintln!("Cu solid: D {d_solid} ang^2/ps, T {t} +- {t_err} K");
    assert!(d_solid.abs() < 0.01, "{d_solid}");
    assert!((t - 300.0).abs() < 30.0 && t_err > 0.0, "{t} +- {t_err}");
    for name in ["rdf.png", "msd.png", "msd.csv", "blocking.csv"] {
        assert!(out.join(name).exists(), "{name}");
    }

    // The same atoms in a cell 4 % wider at 2500 K melt; the first picosecond is left out.
    let liquid = crystal(&base, "cu_liquid.xyz", "Cu", a * 1.04, 3, &FCC);
    let dir = run(
        &[
            &liquid,
            "-os",
            "eam",
            "-mt",
            "bussi",
            "-time",
            "2500",
            "-ts",
            "1fs",
            "-temp",
            "2500",
            "-md_opt",
            "seed=4",
            "write_every=5",
            "-analysis",
            "rdf",
            "msd",
            "skip=1000",
        ],
        &base,
    );
    let out = dir.join("trajectory_analysis");
    let d_liquid = cell(&out.join("diffusion.csv"), "all", "D_angstrom2_per_ps");
    let r = column(&out.join("rdf.csv"), "r_angstrom");
    let gl = column(&out.join("rdf.csv"), "g_Cu-Cu");
    let (first_l, g1_l) = peak_in(&r, &gl, 2.0, 3.0);
    let (_, g2_l) = peak_in(&r, &gl, 3.4, 3.8);
    let (_, g2_s) = peak_in(&r, &g, 3.4, 3.8);
    eprintln!(
        "Cu liquid: D {d_liquid} ang^2/ps; first peak {first_l} ({g1_l}); second shell {g2_l} \
         against {g2_s} of the crystal"
    );
    assert!(
        d_liquid > 0.1 && d_liquid > 20.0 * d_solid.abs(),
        "{d_liquid}"
    );
    assert!(g1_l < 0.7 * g1 && g2_l < 0.6 * g2_s);
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn diamond_silicon_and_the_analysis_of_a_finished_run() {
    let base = scratch("si");
    let a = 5.431;
    let si = crystal(&base, "si64.xyz", "Si", a, 2, &DIAMOND);
    let dir = run(
        &[
            &si,
            "-os",
            "tersoff",
            "-mt",
            "bussi",
            "-time",
            "400",
            "-ts",
            "1fs",
            "-temp",
            "300",
            "-md_opt",
            "seed=7",
            "write_every=2",
            "-analysis",
            "rdf",
            "rmsd",
            "pca",
            "geometry",
            "atoms=1,5;1,5,2",
            "figures=svg",
        ],
        &base,
    );
    let out = dir.join("trajectory_analysis");
    let rdf = out.join("rdf.csv");
    let r = column(&rdf, "r_angstrom");
    let g = column(&rdf, "g_Si-Si");
    let n = column(&rdf, "n_Si-Si");
    let (first, _) = peak_in(&r, &g, 2.0, 2.8);
    let (second, _) = peak_in(&r, &g, 3.2, 4.4);
    let four = at(&r, &n, 2.9);
    eprintln!("Si: peaks {first}, {second}; neighbours within 2.9: {four}");
    // Diamond: four neighbours at a sqrt(3) / 4, twelve at a / sqrt 2.
    assert!((first - a * 3.0_f64.sqrt() / 4.0).abs() < 0.08, "{first}");
    assert!((second - a / 2.0_f64.sqrt()).abs() < 0.12, "{second}");
    assert!((four - 4.0).abs() < 0.1, "{four}");
    // Atoms 1 and 2 are neighbours of atom 5; the bond angle is the tetrahedral one.
    let bond = column(&out.join("geometry.csv"), "distance 1-5 (ang.)");
    let angle = column(&out.join("geometry.csv"), "angle 1-5-2 (deg.)");
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!("Si: bond 1-5 {}, angle 1-5-2 {}", mean(&bond), mean(&angle));
    assert!((mean(&bond) - 2.352).abs() < 0.05);
    assert!((mean(&angle) - 109.47).abs() < 2.0);
    let rmsd = column(&out.join("rmsd.csv"), "rmsd_angstrom");
    let rmsf = column(&out.join("rmsf.csv"), "rmsf_angstrom");
    assert!(rmsd[0] < 1e-12 && rmsd.iter().all(|v| *v < 0.5));
    assert_eq!(rmsf.len(), 64);
    for name in [
        "rdf.svg",
        "rmsd.svg",
        "rmsf.svg",
        "pca.svg",
        "pca.csv",
        "geometry_2.svg",
    ] {
        assert!(out.join(name).exists(), "{name}");
    }
    // The same analysis of the files of the finished run, from the directory or the file.
    let before = std::fs::read_to_string(&rdf).unwrap();
    let d = dir.to_string_lossy().into_owned();
    let again = config_from_cli(&[&d, "-analyze_only", "-analysis", "rdf"]).unwrap();
    let results = run_with(&again, &base, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(results[0].directory, out);
    assert_eq!(std::fs::read_to_string(&rdf).unwrap(), before);
    let traj = dir.join("si64_traj.xyz").to_string_lossy().into_owned();
    let from_file = config_from_cli(&[&traj, "-analyze_only", "-analysis", "rdf"]).unwrap();
    run_with(&from_file, &base, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(std::fs::read_to_string(&rdf).unwrap(), before);
    assert!(out.join("mopt.log").exists());
    let _ = std::fs::remove_dir_all(&base);
}

const WATER: &str = "3\n\nO 0.0 0.0 0.12\nH 0.0 0.80 -0.47\nH 0.0 -0.70 -0.50\n";

#[test]
fn the_vibrations_of_water_in_its_density_of_states_with_gfnff() {
    water_vibrations("gfnff");
}

#[test]
fn the_vibrations_of_water_in_its_density_of_states_with_gfn0_xtb() {
    water_vibrations("gfn0xtb");
}

/// The peaks of the density of states of water near the harmonic frequencies of the
/// optimized structure (`freqmain`) with calculator `method`.
fn water_vibrations(method: &str) {
    let base = scratch(&format!("water_{method}"));
    let start = base.join("water.xyz");
    std::fs::write(&start, WATER).unwrap();
    let s = start.to_string_lossy().into_owned();
    let cfg =
        optmain::config_from_cli(&[&s, "-os", method, "-opt", "rsirfo_bfgs", "-tcc"]).unwrap();
    let optimized = optmain::run(&cfg, &base, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let cfg = freqmain::config_from_cli(&[&xyz, "-os", method]).unwrap();
    let freq = freqmain::run_with(&cfg, &base, None, &CalculatorRegistry::native()).unwrap();
    let harmonic: Vec<f64> = std::fs::read_to_string(freq[0].directory.join("normal_modes.txt"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("Freq [cm^-1]"))
        .flat_map(|l| {
            l.split_whitespace()
                .skip(2)
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(harmonic.len(), 3);
    // NVE for 2 ps from the starting structure (off the minimum along every mode) with the
    // velocities of 300 K and no rotation (which would put side bands at the rotational
    // frequencies around every vibration).
    let dir = run(
        &[
            &s,
            "-os",
            method,
            "-mt",
            "nve",
            "-time",
            "8000",
            "-ts",
            "0.25fs",
            "-temp",
            "300",
            "-md_opt",
            "seed=5",
            "write_every=2",
            "-analysis",
            "vacf",
            "statistics",
            "rmsd",
            "pca",
            "geometry",
            "atoms=1,2;2,1,3",
            "vdos_max=4500",
        ],
        &base,
    );
    let out = dir.join("trajectory_analysis");
    let nu = column(&out.join("vdos.csv"), "wavenumber_cm-1");
    let g = column(&out.join("vdos.csv"), "all");
    let found = peaks(&nu, &g, 0.02);
    eprintln!(
        "water ({method}): harmonic {harmonic:?}; peaks {:?}",
        &found[..found.len().min(8)]
    );
    for f in &harmonic {
        let nearest = found
            .iter()
            .map(|(x, _)| *x)
            .min_by(|a, b| (a - f).abs().total_cmp(&(b - f).abs()))
            .unwrap();
        // Classical anharmonic shifts at 300 K and the resolution of 1 ps of correlation.
        assert!(
            (nearest - f).abs() < 0.02 * f,
            "{f} cm^-1: nearest peak {nearest}"
        );
    }
    // The density of states integrates to 3 N.
    let integral: f64 = g.iter().sum::<f64>() * (nu[1] - nu[0]);
    assert!((integral - 9.0).abs() < 0.5, "{integral}");
    let oh = column(&out.join("geometry.csv"), "distance 1-2 (ang.)");
    let mean = oh.iter().sum::<f64>() / oh.len() as f64;
    assert!(mean > 0.9 && mean < 1.05, "{mean}");
    for name in [
        "vacf.csv",
        "vacf.png",
        "vdos.png",
        "statistics.csv",
        "pca.png",
        "rmsf.csv",
    ] {
        assert!(out.join(name).exists(), "{name}");
    }
    let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap();
    assert!(log.contains("Trajectory analysis (vacf): wrote"));
    assert!(log.contains("vibrational density of states: peaks at"));
    let _ = std::fs::remove_dir_all(&base);
}
