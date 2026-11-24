//! Correlation functions, spectra, and transport coefficients of `mdmain` runs: the
//! vibrational density of states (Blackman window) and the infrared spectrum of water
//! (GFN-FF) against the harmonic frequencies of `freqmain`; the orientational correlation
//! of its bonds; and for the Lennard-Jones fluid the self-diffusion coefficient by Einstein
//! and by Green-Kubo, and the thermal conductivity by Green-Kubo and by the reverse
//! non-equilibrium method of Muller-Plathe.
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
use mopt_rs::domain::Units;
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mdcorr_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

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

/// The row `quantity,method,...` of `transport.csv`: value and error.
fn coefficient(path: &Path, quantity: &str, method: &str) -> (f64, f64) {
    let text = std::fs::read_to_string(path).unwrap();
    let row: Vec<&str> = text
        .lines()
        .map(|l| l.split(',').collect::<Vec<_>>())
        .find(|f| f[0] == quantity && f[1] == method)
        .unwrap_or_else(|| panic!("no {quantity} by {method} in {}", path.display()))
        .into_iter()
        .collect();
    (row[2].parse().unwrap(), row[3].parse().unwrap())
}

fn run(args: &[&str], base: &Path) -> PathBuf {
    let config = config_from_cli(args).unwrap();
    run_with(&config, base, None, &CalculatorRegistry::native()).unwrap()[0]
        .directory
        .clone()
}

const WATER: &str = "3\n\nO 0.0 0.0 0.12\nH 0.0 0.80 -0.47\nH 0.0 -0.70 -0.50\n";

#[test]
fn the_vibrations_of_water_in_its_infrared_spectrum() {
    let base = scratch("water");
    let start = base.join("water.xyz");
    std::fs::write(&start, WATER).unwrap();
    let s = start.to_string_lossy().into_owned();
    let cfg =
        optmain::config_from_cli(&[&s, "-os", "gfnff", "-opt", "rsirfo_bfgs", "-tcc"]).unwrap();
    let optimized = optmain::run(&cfg, &base, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let cfg = freqmain::config_from_cli(&[&xyz, "-os", "gfnff"]).unwrap();
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
    let dir = run(
        &[
            &s,
            "-os",
            "gfnff",
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
            "ir",
            "acf",
            "bonds=1,2;1,3",
            "window=blackman",
            "vdos_max=4500",
        ],
        &base,
    );
    let out = dir.join("trajectory_analysis");
    let check = |file: &str, column_name: &str| {
        let nu = column(&out.join(file), "wavenumber_cm-1");
        let g = column(&out.join(file), column_name);
        let found = peaks(&nu, &g, 0.02);
        eprintln!(
            "{file}: harmonic {harmonic:?}; peaks {:?}",
            &found[..found.len().min(6)]
        );
        for f in &harmonic {
            let nearest = found
                .iter()
                .map(|(x, _)| *x)
                .min_by(|a, b| (a - f).abs().total_cmp(&(b - f).abs()))
                .unwrap();
            assert!(
                (nearest - f).abs() < 0.02 * f,
                "{file}: {f} cm^-1, nearest {nearest}"
            );
        }
    };
    check("vdos.csv", "all");
    check("ir.csv", "intensity");
    let p2 = column(&out.join("orientation.csv"), "P2");
    assert!((p2[0] - 1.0).abs() < 1e-12);
    assert!(dir.join("water_dipoles.csv").exists());
    assert!(out.join("acf_properties.csv").exists() && out.join("vacf_atoms.csv").exists());
    let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap();
    assert!(log.contains("infrared spectrum (Harmonic correction"));
    let _ = std::fs::remove_dir_all(&base);
}

/// UFF argon of the built-in `lj`.
const SIGMA: f64 = 3.817 / 1.122_462_048_309_373;
const EPSILON_K: f64 = 0.237 / Units::BOLTZMANN_KCAL_MOL_K;

/// The Lennard-Jones fluid at `rho* = 0.6` on a simple cubic lattice of `edge^3` atoms.
fn fluid(dir: &Path, edge: usize) -> String {
    let n = edge * edge * edge;
    let side = (n as f64 / 0.6).cbrt() * SIGMA;
    let a = side / edge as f64;
    let mut text = format!("{n}\nLattice=\"{side} 0 0 0 {side} 0 0 0 {side}\" pbc=\"T T T\"\n");
    for i in 0..edge {
        for j in 0..edge {
            for k in 0..edge {
                text.push_str(&format!(
                    "Ar {} {} {}\n",
                    (i as f64 + 0.25) * a,
                    (j as f64 + 0.25) * a,
                    (k as f64 + 0.25) * a
                ));
            }
        }
    }
    let path = dir.join("lj.xyz");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn the_lennard_jones_fluid_conducts_heat_alike_in_equilibrium_and_out_of_it() {
    let base = scratch("lj");
    let xyz = fluid(&base, 6);
    let t = format!("{}", 1.35 * EPSILON_K);
    let cutoff = format!("cutoff={}", 2.5 * SIGMA);
    let common = [
        "-os",
        "lj",
        "-calc_opt",
        &cutoff,
        "switch=0.3",
        "-time",
        "12000",
        "-ts",
        "10fs",
        "-temp",
        &t,
    ];
    let with = |extra: &[&str]| -> Vec<String> {
        let mut a: Vec<String> = vec![xyz.clone()];
        a.extend(common.iter().map(|s| s.to_string()));
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    };
    let go = |a: Vec<String>| {
        let refs: Vec<&str> = a.iter().map(String::as_str).collect();
        run(&refs, &base)
    };
    // Equilibrium: Bussi's thermostat with a long coupling time (5 ps) keeps the temperature
    // and leaves the dynamics; the first 20 ps are left out.
    let eq = go(with(&[
        "-mt",
        "bussi",
        "-md_opt",
        "seed=2",
        "tau=5000",
        "write_every=5",
        "heat_flux=on",
        "stress=on",
        "-analysis",
        "transport",
        "statistics",
        "skip=20000",
    ]));
    let summary = eq.join("trajectory_analysis").join("transport.csv");
    let (d_gk, d_gk_err) = coefficient(&summary, "self-diffusion", "Green-Kubo");
    let (d_e, d_e_err) = coefficient(&summary, "self-diffusion", "Einstein");
    let (k_gk, k_gk_err) = coefficient(&summary, "thermal conductivity", "Green-Kubo");
    let (eta, eta_err) = coefficient(&summary, "shear viscosity", "Green-Kubo");
    // Reverse non-equilibrium: NVE with exchanges every 20 steps, from the fluid of the run
    // before (velocities of the temperature drawn anew), with four seeds. One trajectory is
    // chaotic: a change of 5e-9 Angstrom in the cell of the start moved a single run from
    // 4.20 to 2.11. Measured over eight seeds (3 to 10) of this system: with exchanges every
    // 20 steps kappa* 2.85 +- 0.36 (mean, standard deviation of the runs), the error each run
    // reports 0.12 to 0.21 (it is the scatter of the gradient over the blocks of one run and
    // lies about a factor 2 below the spread of the runs, see molecular_dynamics.md); with
    // exchanges every 100 steps 2.86 +- 1.1. The mean of the seeds is compared, with its
    // standard error.
    let ne_start = eq.join("lj_final.xyz").to_string_lossy().into_owned();
    let mut runs = Vec::new();
    for seed in 3..7 {
        let seed = format!("seed={seed}");
        let mut args = with(&[
            "-mt",
            "nve",
            "-md_opt",
            &seed,
            "write_every=5",
            "exchange=20",
            "-analysis",
            "transport",
            "skip=20000",
        ]);
        args[0] = ne_start.clone();
        let ne = go(args);
        runs.push(coefficient(
            &ne.join("trajectory_analysis").join("transport.csv"),
            "thermal conductivity",
            "reverse non-equilibrium",
        ));
    }
    let n = runs.len() as f64;
    let k_mp = runs.iter().map(|r| r.0).sum::<f64>() / n;
    let spread = (runs.iter().map(|r| (r.0 - k_mp).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let k_mp_err = spread / n.sqrt();
    let reported = runs.iter().map(|r| r.1).sum::<f64>() / n;
    // Reduced units.
    let kb = Units::BOLTZMANN_J_PER_K;
    let sigma = SIGMA * 1e-10;
    let eps = EPSILON_K * kb;
    let mass = 39.948 * Units::AMU_TO_KG;
    let kappa_star = sigma * sigma / kb * (mass / eps).sqrt();
    let eta_star = sigma * sigma / (mass * eps).sqrt();
    eprintln!(
        "LJ rho* 0.6 T* 1.35, 216 atoms: D {d_e:.4} +- {d_e_err:.4} (Einstein), {d_gk:.4} +- \
         {d_gk_err:.4} (Green-Kubo) ang^2/ps; kappa* {:.3} +- {:.3} (Green-Kubo), {:.3} +- {:.3} \
         (reverse, mean of {n} seeds; spread {:.3}, error reported by a run {:.3}); eta* {:.3} \
         +- {:.3}",
        k_gk * kappa_star,
        k_gk_err * kappa_star,
        k_mp * kappa_star,
        k_mp_err * kappa_star,
        spread * kappa_star,
        reported * kappa_star,
        eta * eta_star,
        eta_err * eta_star
    );
    assert!((d_gk - d_e).abs() < 3.0 * (d_gk_err.powi(2) + d_e_err.powi(2)).sqrt() + 0.05 * d_e);
    // The two methods agree within three standard errors (of the mean of the seeds, and of
    // the one Green-Kubo run).
    assert!((k_gk - k_mp).abs() < 3.0 * (k_gk_err.powi(2) + k_mp_err.powi(2)).sqrt());
    // Evans (Phys. Rev. A 1986, 34, 1449): about 3.4 in reduced units; the reverse method
    // in this box of 7.1 sigma gives about 2.85 (above), the Green-Kubo run 3.39 +- 0.54.
    for k in [k_gk, k_mp] {
        assert!((k * kappa_star - 3.4).abs() < 1.2, "{}", k * kappa_star);
    }
    // The error a run reports is the scatter within the run: on this system it lies within a
    // factor 4 of the spread of the runs (measured: 1.7 on these seeds, 2.0 on eight).
    assert!(
        spread < 4.0 * reported,
        "spread {spread}, reported {reported}"
    );
    assert!(
        reported < spread * 4.0,
        "spread {spread}, reported {reported}"
    );
    assert!(eta > 0.0);
    let _ = std::fs::remove_dir_all(&base);
}
