//! `freqmain`: vibrational analysis of given structures through the command-line layer, the
//! same reports as `optmain -freq`, the gradient projected out, a periodic cell, and the
//! transverse vibrations along a path.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::freqmain::{config_from_cli, run_with};
use mopt_rs::cli::{optmain, run_command, EXIT_OK};
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_freq_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

const WATER: &str = "3\n\nO 0.0 0.0 0.12\nH 0.0 0.80 -0.47\nH 0.0 -0.70 -0.50\n";

fn frequencies(dir: &Path) -> Vec<f64> {
    std::fs::read_to_string(dir.join("normal_modes.txt"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("Freq [cm^-1]"))
        .flat_map(|l| {
            l.split_whitespace()
                .skip(2)
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn the_reports_of_optmain_freq_for_a_given_structure() {
    let dir = scratch("same");
    let start = file(&dir, "water.xyz", WATER);
    // Optimize with the analysis at the end, then analyse the optimized structure alone.
    let cfg = optmain::config_from_cli(&[
        &start,
        "-os",
        "gfnff",
        "-opt",
        "rsirfo_bfgs",
        "-tcc",
        "-freq",
    ])
    .unwrap();
    let optimized = optmain::run(&cfg, &dir, None).unwrap();
    let from_optmain = frequencies(&optimized[0].directory);
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[&xyz, "-os", "gfnff"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let run_dir = &out[0].directory;
    assert!(run_dir.to_string_lossy().contains("_FREQ_"));
    let alone = frequencies(run_dir);
    assert_eq!(alone.len(), 3);
    for (a, b) in alone.iter().zip(&from_optmain) {
        // The optimized structure was written with six decimals.
        assert!((a - b).abs() < 0.5, "{a} against {b}");
    }
    let analysis = out[0].analysis.as_ref().unwrap();
    assert_eq!(analysis.point_group.label, "C2v");
    assert_eq!(analysis.thermo.imaginary, 0);
    for name in [
        "thermochemistry.txt",
        "normal_modes.txt",
        "symmetry.txt",
        "mopt.log",
        "settings.json",
    ] {
        assert!(run_dir.join(name).exists(), "{name}");
    }
    // Conditions and cutoffs from the command line.
    let cfg = config_from_cli(&[
        &xyz,
        "-os",
        "gfnff",
        "-temp",
        "500",
        "-press",
        "1e5",
        "-freq_corr",
        "none",
        "-freq_lowest",
        "2000",
        "-symm_tol",
        "1e-9",
    ])
    .unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let analysis = out[0].analysis.as_ref().unwrap();
    assert_eq!(analysis.thermo.temperature, 500.0);
    // Only the two stretches are above 2000 cm^-1, and no symmetry within 1e-9 Bohr.
    assert_eq!(analysis.thermo.vibrations, 2);
    assert_eq!(analysis.thermo.symmetry_number, 1);
    // From the command table.
    assert_eq!(run_command("freqmain", &["-h".to_string()]), EXIT_OK);
}

#[test]
fn the_gradient_is_projected_out_on_request() {
    let dir = scratch("project");
    let xyz = file(&dir, "water.xyz", WATER);
    let run = |extra: &[&str]| {
        let mut args = vec![xyz.as_str(), "-os", "gfnff"];
        args.extend_from_slice(extra);
        let cfg = config_from_cli(&args).unwrap();
        let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
        let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
        (frequencies(&out[0].directory), log)
    };
    // The start is not a stationary point: three modes and a remark.
    let (all, log) = run(&[]);
    assert_eq!(all.len(), 3);
    assert!(log.contains("not a stationary point"), "{log}");
    // Without the direction of the gradient: two.
    let (transverse, log) = run(&["-project_gradient"]);
    assert_eq!(transverse.len(), 2);
    assert!(log.contains("projected out"), "{log}");
}

#[test]
fn a_periodic_cell_gives_the_modes_at_the_zone_centre() {
    let dir = scratch("cell");
    let a = 3.61;
    let xyz = file(
        &dir,
        "cu.xyz",
        &format!(
            "4\nLattice=\"{a} 0.0 0.0 0.0 {a} 0.0 0.0 0.0 {a}\" pbc=\"T T T\"\n\
             Cu 0.02 0.0 0.0\nCu {h} {h} 0.0\nCu {h} 0.0 {h}\nCu 0.0 {h} {h}\n",
            h = a / 2.0
        ),
    );
    let cfg = config_from_cli(&[&xyz, "-os", "emt"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    // Twelve coordinates less three translations.
    assert_eq!(frequencies(&out[0].directory).len(), 9);
    assert!(out[0].analysis.is_none());
    assert!(!out[0].directory.join("thermochemistry.txt").exists());
}

#[test]
fn transverse_vibrations_along_a_path() {
    let dir = scratch("path");
    // Water with one O-H bond stretched in three steps.
    let mut frames = String::new();
    for k in 0..3 {
        let y = 0.80 + 0.05 * k as f64;
        frames.push_str(&format!(
            "3\npoint {k}\nO 0.0 0.0 0.12\nH 0.0 {y} -0.47\nH 0.0 -0.76 -0.47\n"
        ));
    }
    let xyz = file(&dir, "path.xyz", &frames);
    let cfg = config_from_cli(&[&xyz, "-os", "gfnff", "-path"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let profile = out[0].profile.as_ref().unwrap();
    assert_eq!(profile.arc_length.len(), 3);
    assert!(profile.arc_length[2] > profile.arc_length[1] && profile.arc_length[0] == 0.0);
    // Two vibrations orthogonal to the path at every point.
    assert!(profile.wavenumbers.iter().all(|w| w.len() == 2));
    let csv = std::fs::read_to_string(out[0].directory.join("path_profile.csv")).unwrap();
    assert_eq!(csv.lines().count(), 4);
    assert!(csv.starts_with("point,arc_length_sqrt_amu_bohr,energy_hartree,"));
    let report = std::fs::read_to_string(out[0].directory.join("bifurcations.txt")).unwrap();
    assert!(report.contains("No valley-ridge transition"), "{report}");
    // The elements of the reaction-path Hamiltonian and the valley analysis.
    for name in [
        "curvature_couplings.csv",
        "mode_couplings.csv",
        "urva_direction.csv",
        "urva_curvature.csv",
        "reaction_phases.txt",
        "free_energy_profile.csv",
        "variational_tst.txt",
    ] {
        assert!(out[0].directory.join(name).exists(), "{name}");
    }
    assert!(csv.contains("adiabatic_energy_hartree,path_curvature,sct_mass_ratio"));
    // The couplings of the two modes to the curvature add up to the curvature.
    for i in 0..3 {
        let sum: f64 = profile.curvature_couplings[i]
            .iter()
            .map(|b| b * b)
            .sum::<f64>()
            .sqrt();
        assert!((sum - profile.curvature[i]).abs() < 1e-8 * (1.0 + profile.curvature[i]));
    }
    // The path is the stretch of the bond O-H(2): that bond carries its direction.
    let path = out[0].path.as_ref().unwrap();
    assert_eq!(path.components.labels, vec!["R(1-2)", "R(1-3)", "A(2-1-3)"]);
    let direction = &path.components.direction[1];
    assert!(direction[0].abs() > 0.9, "{direction:?}");
    assert!(direction[0].abs() > direction[1].abs() && direction[0].abs() > direction[2].abs());
    let table = std::fs::read_to_string(out[0].directory.join("urva_curvature.csv")).unwrap();
    assert!(table.starts_with("point,arc_length_sqrt_amu_bohr,R(1-2),R(1-3),A(2-1-3)\n"));
    assert_eq!(table.lines().count(), 4);
    // One frame is not a path.
    let single = file(&dir, "one.xyz", WATER);
    let cfg = config_from_cli(&[&single, "-os", "gfnff", "-path"]).unwrap();
    assert!(run_with(&cfg, &dir, None, &CalculatorRegistry::native()).is_err());
}

/// Frames of the umbrella inversion of ammonia with the N-H distance kept: the nitrogen at
/// the height `h` (Angstrom) above the plane of the hydrogens, the centre of mass at rest.
fn ammonia_inversion(heights: &[f64]) -> String {
    let (bond, m_n, m_h) = (1.012_f64, 14.007, 1.008);
    let total = m_n + 3.0 * m_h;
    let mut frames = String::new();
    for h in heights {
        let radius = (bond * bond - h * h).sqrt();
        let (z_n, z_h) = (h * 3.0 * m_h / total, -h * m_n / total);
        frames.push_str(&format!("4\nh = {h}\nN 0.0 0.0 {z_n:.10}\n"));
        for k in 0..3 {
            let phi = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
            frames.push_str(&format!(
                "H {:.10} {:.10} {z_h:.10}\n",
                radius * phi.cos(),
                radius * phi.sin()
            ));
        }
    }
    frames
}

#[test]
fn a_barrier_on_the_path_gives_the_variational_transition_state_and_tunneling() {
    let dir = scratch("barrier");
    let heights: Vec<f64> = (0..9).map(|k| -0.4 + 0.1 * k as f64).collect();
    let xyz = file(&dir, "inversion.xyz", &ammonia_inversion(&heights));
    let cfg = config_from_cli(&[
        &xyz,
        "-os",
        "gfnff",
        "-path",
        "-path_tangent",
        "points",
        "-temp",
        "200",
        "-phase_prominence",
        "0.05",
    ])
    .unwrap();
    assert_eq!(cfg.phase_prominence, 0.05);
    assert!(config_from_cli(&[&xyz, "-path", "-path_tangent", "normal"]).is_err());
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let profile = out[0].profile.as_ref().unwrap();
    let path = out[0].path.as_ref().unwrap();
    // The planar structure in the middle is the top of the barrier, and the path is
    // symmetric about it.
    assert_eq!(path.free_energy.saddle, 4);
    assert!((path.energies[0] - path.energies[8]).abs() < 1e-8);
    assert!(path.energies[4] > path.energies[0]);
    // The path is a scan, not a steepest-descent path: the gradient (which also stretches
    // the bonds) is not along it, and the tangent comes from the points. The inversion is
    // then the direction of the path at the planar structure and not among its modes.
    assert!(path.gradient_alignment.iter().any(|a| *a < 0.9));
    assert_eq!(profile.wavenumbers[4].len(), 5);
    assert!(profile.wavenumbers[4].iter().all(|w| *w > 0.0));
    assert_eq!(path.tunneling.top, 4);
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("-path_tangent points"), "{log}");
    assert_eq!(path.free_energy.temperature, 200.0);
    assert_eq!(path.free_energy.free_energy.len(), 9);
    assert!(path.free_energy.recrossing <= 1.0 + 1e-12 && path.free_energy.recrossing > 0.0);
    // Tunneling raises the rate, cutting the corner raises it further (or leaves it).
    assert!(path.tunneling.zct > 1.0, "{}", path.tunneling.zct);
    assert!(path.tunneling.sct >= path.tunneling.zct - 1e-9);
    assert!(path
        .tunneling
        .mass_ratio
        .iter()
        .all(|r| *r > 0.0 && *r <= 1.0));
    assert!(!path.phases.is_empty());
    assert_eq!(path.phases[0].start, 0);
    assert_eq!(path.phases[path.phases.len() - 1].end, 8);
    let report = std::fs::read_to_string(out[0].directory.join("variational_tst.txt")).unwrap();
    assert!(report.contains("Temperature: 200.00 K"), "{report}");
    assert!(
        report.contains("(ZCT)") && report.contains("(SCT)"),
        "{report}"
    );
    let phases = std::fs::read_to_string(out[0].directory.join("reaction_phases.txt")).unwrap();
    assert!(phases.contains("Phase 1: points 0-"), "{phases}");
    let free = std::fs::read_to_string(out[0].directory.join("free_energy_profile.csv")).unwrap();
    assert_eq!(free.lines().count(), 10);
}

#[test]
fn the_vibrations_of_chosen_atoms_from_the_partial_and_the_effective_hessian() {
    let dir = scratch("partial");
    let start = file(&dir, "water.xyz", WATER);
    let cfg =
        optmain::config_from_cli(&[&start, "-os", "gfnff", "-opt", "rsirfo_bfgs", "-tcc"]).unwrap();
    let optimized = optmain::run(&cfg, &dir, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let run = |extra: &[&str]| {
        let mut args = vec![xyz.as_str(), "-os", "gfnff"];
        args.extend_from_slice(extra);
        let cfg = config_from_cli(&args).unwrap();
        run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap()
    };
    // The full analysis is not changed by the request.
    let out = run(&["-partial_atoms", "1,2", "-partial_hessian", "effective"]);
    let full = frequencies(&out[0].directory);
    assert_eq!(full.len(), 3);
    let (partial, modes) = out[0].partial.as_ref().unwrap();
    assert_eq!(partial.atoms, vec![0, 1]);
    // With O and one H held, the other H can still turn about their axis.
    assert_eq!(partial.free_directions, 1);
    // One vibration of the pair: the local O-H stretch, between the symmetric and the
    // antisymmetric stretch of the molecule.
    let local: Vec<f64> = modes
        .vibrations(1.0, true)
        .iter()
        .map(|k| modes.wavenumbers[*k])
        .collect();
    assert_eq!(local.len(), 1);
    assert!(
        local[0] > full[1] && local[0] < full[2],
        "{local:?} in {full:?}"
    );
    let folder = out[0].directory.join("partial_hessian");
    let summary = std::fs::read_to_string(folder.join("summary.txt")).unwrap();
    assert!(
        summary.contains("Relaxed force constant of the distance 1-2"),
        "{summary}"
    );
    let matrix = std::fs::read_to_string(folder.join("hessian.txt")).unwrap();
    assert_eq!(matrix.lines().count(), 6);
    assert_eq!(frequencies(&folder).len(), 1);
    // The relaxed force constant gives that wavenumber with the reduced mass of O-H.
    let (constant, _) = mopt_rs::vib::relaxed_stretch(partial).unwrap();
    let m = partial.system.masses_amu();
    let reduced = m[0] * m[1] / (m[0] + m[1]);
    let curvature = modes.curvatures[modes.vibrations(1.0, true)[0]];
    assert!((constant / reduced - curvature).abs() < 1e-3 * curvature);
    // The block of all atoms is the molecule itself; the frozen block of one hydrogen has
    // its three motions against the rest.
    let out = run(&["-partial_atoms", "1-3"]);
    let (_, modes) = out[0].partial.as_ref().unwrap();
    let all: Vec<f64> = modes
        .vibrations(1.0, true)
        .iter()
        .map(|k| modes.wavenumbers[*k])
        .collect();
    for (a, b) in all.iter().zip(&full) {
        // The table has four decimals.
        assert!((a - b).abs() < 1e-3, "{a} against {b}");
    }
    let out = run(&["-partial_atoms", "3", "-partial_hessian", "frozen"]);
    let (_, modes) = out[0].partial.as_ref().unwrap();
    assert_eq!(modes.wavenumbers.len(), 3);
    assert!(modes.vibrations(1.0, true).len() >= 2);
    // What is refused.
    assert!(config_from_cli(&[&xyz, "-partial_atoms", "1,2", "-partial_hessian", "half"]).is_err());
    let cfg = config_from_cli(&[&xyz, "-os", "gfnff", "-partial_atoms", "1,7"]).unwrap();
    assert!(run_with(&cfg, &dir, None, &CalculatorRegistry::native()).is_err());
}

#[test]
fn anharmonic_fundamentals_from_a_quartic_force_field() {
    let dir = scratch("anharmonic");
    let start = file(&dir, "water.xyz", WATER);
    let cfg = optmain::config_from_cli(&[&start, "-os", "gfn0xtb", "-opt", "rsirfo_bfgs", "-tcc"])
        .unwrap();
    let optimized = optmain::run(&cfg, &dir, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let run = |extra: &[&str]| {
        let mut args = vec![xyz.as_str(), "-os", "gfn0xtb", "-anharmonic"];
        args.extend_from_slice(extra);
        let cfg = config_from_cli(&args).unwrap();
        run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap()
    };
    let out = run(&[]);
    let found = out[0].anharmonic.as_ref().unwrap();
    assert_eq!(found.field.modes(), 3);
    let harmonic = frequencies(&out[0].directory);
    for (i, reference) in harmonic.iter().enumerate() {
        assert!((found.field.harmonic[i] - reference).abs() < 1e-3);
        // Anharmonicity lowers every fundamental of water, the stretches by more than the
        // bend, in both methods.
        for value in [found.vpt2.fundamentals[i], found.vscf.fundamentals[i]] {
            assert!(
                value < *reference && value > 0.8 * reference,
                "{i}: {value}"
            );
        }
    }
    let shift = |i: usize| harmonic[i] - found.vpt2.fundamentals[i];
    assert!(shift(1) > shift(0) && shift(2) > shift(0));
    assert!(found.vpt2.zero_point < found.vpt2.zero_point_harmonic);
    assert!(found.vscf.zero_point < found.vpt2.zero_point_harmonic);
    assert!(found.vscf.converged && !found.vscf.unbounded);
    // The stretches are bound by the bonds: negative diagonal anharmonic constants.
    assert!(found.vpt2.constants[(1, 1)] < 0.0 && found.vpt2.constants[(2, 2)] < 0.0);
    let report = std::fs::read_to_string(out[0].directory.join("anharmonic.txt")).unwrap();
    assert!(
        report.contains("VPT2") && report.contains("VSCF"),
        "{report}"
    );
    assert!(report.contains("Anharmonic constants x_ij"), "{report}");
    let constants = std::fs::read_to_string(out[0].directory.join("force_field.txt")).unwrap();
    assert!(constants.contains("cubic: i j k phi_ijk"), "{constants}");
    // Twice the displacement changes the fundamentals by less than a wavenumber.
    let wider = run(&["-qff_step", "0.1"]);
    let other = wider[0].anharmonic.as_ref().unwrap();
    for i in 0..3 {
        assert!((other.vpt2.fundamentals[i] - found.vpt2.fundamentals[i]).abs() < 1.0);
    }
    // A smaller basis is enough for water; too small a one is refused.
    let small = run(&["-vscf_basis", "10"]);
    let other = small[0].anharmonic.as_ref().unwrap();
    for i in 0..3 {
        assert!((other.vscf.fundamentals[i] - found.vscf.fundamentals[i]).abs() < 1.0);
    }
    let cfg =
        config_from_cli(&[&xyz, "-os", "gfn0xtb", "-anharmonic", "-vscf_basis", "2"]).unwrap();
    assert!(run_with(&cfg, &dir, None, &CalculatorRegistry::native()).is_err());
}

#[test]
fn thermochemistry_choices_scans_and_populations() {
    let dir = scratch("thermo");
    let start = file(&dir, "water.xyz", WATER);
    let cfg =
        optmain::config_from_cli(&[&start, "-os", "gfnff", "-opt", "rsirfo_bfgs", "-tcc"]).unwrap();
    let optimized = optmain::run(&cfg, &dir, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let run = |inputs: &[&str], extra: &[&str]| {
        let mut args: Vec<&str> = inputs.to_vec();
        args.extend_from_slice(&["-os", "gfnff"]);
        args.extend_from_slice(extra);
        let cfg = config_from_cli(&args).unwrap();
        run_with(&cfg, &dir, None, &CalculatorRegistry::native())
    };
    let plain = run(&[&xyz], &[]).unwrap();
    let reference = &plain[0].analysis.as_ref().unwrap().thermo;
    // A solute at 1 mol/L, scaled frequencies, an electronic energy from elsewhere, and a
    // range of temperatures.
    let out = run(
        &[&xyz],
        &[
            "-conc",
            "1.0",
            "-freq_scale",
            "0.97",
            "-spc_energy",
            "-76.4",
            "-temp_scan",
            "250,350,50",
        ],
    )
    .unwrap();
    let thermo = &out[0].analysis.as_ref().unwrap().thermo;
    assert_eq!(thermo.concentration, Some(1.0));
    assert_eq!(thermo.electronic_energy, -76.4);
    assert!(thermo.entropy.translational < reference.entropy.translational);
    assert!((thermo.zero_point_energy - 0.97 * reference.zero_point_energy).abs() < 1e-12);
    let text = std::fs::read_to_string(out[0].directory.join("thermochemistry.txt")).unwrap();
    assert!(text.contains("Concentration 1 [mol/L]"), "{text}");
    assert!(text.contains("Frequency scale factors: 0.97"), "{text}");
    let scan = std::fs::read_to_string(out[0].directory.join("thermochemistry_scan.csv")).unwrap();
    let rows: Vec<&str> = scan.lines().collect();
    assert_eq!(rows.len(), 4);
    assert!(rows[1].starts_with("250.0000,") && rows[3].starts_with("350.0000,"));
    // The free energy falls with the temperature.
    let gibbs = |row: &str| row.rsplit(',').next().unwrap().parse::<f64>().unwrap();
    assert!(gibbs(rows[3]) < gibbs(rows[1]));
    // The thermal energy treated apart from the entropy.
    let apart = run(&[&xyz], &["-freq_corr_h", "none"]).unwrap();
    let other = &apart[0].analysis.as_ref().unwrap().thermo;
    assert_eq!(other.entropy.vibrational, reference.entropy.vibrational);
    // The moment of the molecule bounds the free rotor more than the average one.
    let bounded = run(&[&xyz], &["-freq_bav", "molecule"]).unwrap();
    let other = &bounded[0].analysis.as_ref().unwrap().thermo;
    assert!(other.entropy.vibrational < reference.entropy.vibrational);
    // Two inputs 1 mHartree apart: populations in the ratio exp(dE / kT).
    let both = run(
        &[&xyz, &xyz],
        &["-spc_energy", "-76.400", "-76.399", "-boltz"],
    )
    .unwrap();
    let (first, second) = (both[0].population.unwrap(), both[1].population.unwrap());
    assert!((first + second - 1.0).abs() < 1e-12);
    let kt = mopt_rs::vib::boltzmann_hartree() * 298.15;
    assert!((first / second - (0.001 / kt).exp()).abs() < 1e-6 * first / second);
    let table = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .flat_map(|entry| std::fs::read_dir(entry.path()).unwrap().flatten())
        .find(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("boltzmann_")
        })
        .unwrap();
    let csv = std::fs::read_to_string(table.path()).unwrap();
    assert!(csv.starts_with("input,gibbs_hartree,relative_kcal_mol,population"));
    assert_eq!(csv.lines().count(), 3);
    // What is refused.
    assert!(run(&[&xyz, &xyz], &["-spc_energy", "-76.4"]).is_err());
    assert!(config_from_cli(&[&xyz, "-temp_scan", "300,200,10"]).is_err());
    assert!(config_from_cli(&[&xyz, "-freq_corr_h", "x"]).is_err());
    assert!(config_from_cli(&[&xyz, "-freq_bav", "x"]).is_err());
}

#[test]
fn local_modes_of_internal_coordinates() {
    let dir = scratch("local");
    let start = file(&dir, "water.xyz", WATER);
    let cfg =
        optmain::config_from_cli(&[&start, "-os", "gfnff", "-opt", "rsirfo_bfgs", "-tcc"]).unwrap();
    let optimized = optmain::run(&cfg, &dir, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let run = |extra: &[&str]| {
        let mut args = vec![xyz.as_str(), "-os", "gfnff"];
        args.extend_from_slice(extra);
        let cfg = config_from_cli(&args)?;
        run_with(&cfg, &dir, None, &CalculatorRegistry::native())
    };
    // One bond, with the effective Hessian of its two atoms for comparison.
    let out = run(&[
        "-local_modes",
        "1-2",
        "2-1-3",
        "-partial_atoms",
        "1,2",
        "-partial_hessian",
        "effective",
    ])
    .unwrap();
    let full = frequencies(&out[0].directory);
    let local = out[0].local_modes.as_ref().unwrap();
    assert_eq!(local.modes.len(), 2);
    assert_eq!(local.modes[0].label, "R(1-2)");
    assert_eq!(local.modes[1].label, "A(2-1-3)");
    let (partial, _) = out[0].partial.as_ref().unwrap();
    let (relaxed, _) = mopt_rs::vib::relaxed_stretch(partial).unwrap();
    let stretch = &local.modes[0];
    assert!(
        (stretch.force_constant - relaxed).abs() < 1e-3 * relaxed,
        "{} against {relaxed}",
        stretch.force_constant
    );
    // The local stretch lies between the two stretches of the molecule, the local bend
    // near its bend.
    assert!(stretch.wavenumber > full[1] && stretch.wavenumber < full[2]);
    assert!((local.modes[1].wavenumber - full[0]).abs() < 0.1 * full[0]);
    for mode in &local.modes {
        assert!((mode.composition.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
    let report = std::fs::read_to_string(out[0].directory.join("local_modes.txt")).unwrap();
    assert!(
        report.contains("R(1-2)") && report.contains("A(2-1-3)"),
        "{report}"
    );
    let table =
        std::fs::read_to_string(out[0].directory.join("local_mode_composition.csv")).unwrap();
    assert_eq!(table.lines().count(), 3);
    // All bonds are the two equivalent ones; everything is the two bonds and the angle.
    let out = run(&["-local_modes", "bonds"]).unwrap();
    let bonds = out[0].local_modes.as_ref().unwrap();
    assert_eq!(bonds.modes.len(), 2);
    let difference = bonds.modes[0].wavenumber - bonds.modes[1].wavenumber;
    assert!(difference.abs() < 1.0, "{difference}");
    let out = run(&["-local_modes", "auto"]).unwrap();
    assert_eq!(out[0].local_modes.as_ref().unwrap().modes.len(), 3);
    // What is refused.
    assert!(run(&["-local_modes", "1-1"]).is_err());
    assert!(run(&["-local_modes", "1-7"]).is_err());
    assert!(run(&["-local_modes", "1-2-3-1-2"]).is_err());
    assert!(run(&["-local_modes", "x"]).is_err());
}

#[test]
fn phonons_of_copper_from_a_supercell() {
    let dir = scratch("phonon");
    let a = 3.61;
    let xyz = file(
        &dir,
        "cu.xyz",
        &format!(
            "4\nLattice=\"{a} 0.0 0.0 0.0 {a} 0.0 0.0 0.0 {a}\" pbc=\"T T T\"\n\
             Cu 0.0 0.0 0.0\nCu {h} {h} 0.0\nCu {h} 0.0 {h}\nCu 0.0 {h} {h}\n",
            h = a / 2.0
        ),
    );
    let cfg = config_from_cli(&[
        &xyz,
        "-os",
        "emt",
        "-phonon_supercell",
        "2",
        "2",
        "2",
        "-phonon_path",
        "0,0,0",
        "0.5,0,0",
        "-phonon_points",
        "4",
        "-phonon_mesh",
        "2",
        "2",
        "2",
        "-temp_scan",
        "100,300,100",
    ])
    .unwrap();
    assert_eq!(cfg.phonon_supercell, Some([2, 2, 2]));
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let phonons = out[0].phonons.as_ref().unwrap();
    assert_eq!(phonons.constants.branches(), 12);
    assert!(phonons.constants.sum_rule_residual < 1e-4);
    // At the zone centre: three acoustic modes at zero and the nine modes of the cell.
    let centre = &phonons.dispersion[0].2;
    assert!(centre[..3].iter().all(|w| w.abs() < 1.0), "{centre:?}");
    let cell = frequencies(&out[0].directory);
    assert_eq!(cell.len(), 9);
    for (found, reference) in centre[3..].iter().zip(&cell) {
        assert!(
            (found - reference).abs() < 1.0,
            "{found} against {reference}"
        );
    }
    // Along the path to the zone boundary the lattice is stable, and the highest
    // frequency is that of copper (about 240 cm^-1 in experiment).
    assert_eq!(phonons.dispersion.len(), 5);
    let edge = &phonons.dispersion[4].2;
    assert!(edge.iter().all(|w| *w > 10.0), "{edge:?}");
    let highest = edge[11];
    assert!(highest > 150.0 && highest < 350.0, "{highest}");
    // The density of states holds the twelve branches.
    let dos = &phonons.density_of_states;
    let step = dos[1].0 - dos[0].0;
    let integral: f64 = dos.iter().map(|(_, d)| d * step).sum();
    assert!((integral - 12.0).abs() < 0.05, "{integral}");
    // Eight wave vectors: all modes but the three acoustic ones at the zone centre, and a
    // heat capacity near that of Dulong and Petit, 3 k per atom, at 298 K.
    let thermo = phonons.thermo.as_ref().unwrap();
    assert_eq!(
        (thermo.modes, thermo.left_out, thermo.imaginary),
        (93, 3, 0)
    );
    let classical = 12.0 * mopt_rs::vib::boltzmann_hartree() * 93.0 / 96.0;
    assert!(thermo.heat_capacity > 0.85 * classical && thermo.heat_capacity < classical);
    assert!(thermo.free_energy < thermo.internal_energy);
    for name in [
        "phonon_dispersion.csv",
        "phonon_dos.csv",
        "phonon_thermochemistry.txt",
        "phonon_thermochemistry_scan.csv",
    ] {
        assert!(out[0].directory.join(name).exists(), "{name}");
    }
    let scan =
        std::fs::read_to_string(out[0].directory.join("phonon_thermochemistry_scan.csv")).unwrap();
    assert_eq!(scan.lines().count(), 4);
    // What is refused: a molecule, and a supercell or a path that makes no sense.
    let water = file(&dir, "water.xyz", WATER);
    let cfg =
        config_from_cli(&[&water, "-os", "gfnff", "-phonon_supercell", "2", "2", "2"]).unwrap();
    assert!(run_with(&cfg, &dir, None, &CalculatorRegistry::native()).is_err());
    assert!(config_from_cli(&[&xyz, "-phonon_supercell", "2", "2"]).is_err());
    assert!(config_from_cli(&[&xyz, "-phonon_supercell", "2", "0", "2"]).is_err());
    assert!(config_from_cli(&[&xyz, "-phonon_path", "0,0"]).is_err());
}

#[test]
fn a_conical_intersection_of_two_excited_states_of_ozone() {
    let dir = scratch("meci");
    // A bent, unsymmetric start; the search goes to the crossing of the first and the second
    // excited singlet state of the simplified Tamm-Dancoff approximation on GFN0-xTB
    // orbitals.
    let start = file(
        &dir,
        "o3.xyz",
        "3\n\nO 0.0 0.0 0.0\nO 1.1 0.3 0.0\nO -1.0 0.5 0.1\n",
    );
    let cfg = optmain::config_from_cli(&[
        &start,
        "-os",
        "gfn0xtb",
        "-es",
        "1",
        "-mf",
        "opt_meci",
        "-opt",
        "rsirfo_bfgs",
        "-freq",
    ])
    .unwrap();
    assert_eq!(cfg.excited_state, 1);
    let optimized = optmain::run(&cfg, &dir, None).unwrap();
    let run_dir = &optimized[0].directory;
    // The excited states at the start, and the seam analysis at the end of the search.
    let listed = std::fs::read_to_string(run_dir.join("excitations.txt")).unwrap();
    assert!(listed.contains("cm^-1"), "{listed}");
    let report = std::fs::read_to_string(run_dir.join("seam.txt")).unwrap();
    assert!(report.contains("conical intersection"), "{report}");
    assert!(report.contains("a minimum of the seam"), "{report}");
    // The same for the optimized structure alone.
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[&xyz, "-os", "gfn0xtb", "-es", "1", "-mf", "opt_meci"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let seam = out[0].seam.as_ref().unwrap();
    assert!(seam.gap.abs() < 5e-4, "{}", seam.gap);
    assert!(seam.seam_gradient < 1e-3, "{}", seam.seam_gradient);
    // Three atoms: three vibrations, two of them the branching plane.
    assert_eq!(seam.modes.curvatures.len(), 1);
    assert!(seam.modes.wavenumbers[0] > 500.0 && seam.modes.wavenumbers[0] < 1500.0);
    // Next to the cusp the Hessians of the two states differ by a large term along the
    // second direction of the branching plane, which the search had found as well.
    let (largest, next, overlap) = seam.coupling_curvature.unwrap();
    assert!(largest.abs() > 5.0 * next.abs(), "{largest} {next}");
    assert!(overlap > 0.9, "{overlap}");
    // A single excited state from the command line: the crossing point is not a minimum of
    // the first excited state.
    let cfg = config_from_cli(&[&xyz, "-os", "gfn0xtb", "-es", "1"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out[0].seam.is_none() && out[0].analysis.is_some());
    // A calculator without excited states refuses.
    let cfg = config_from_cli(&[&xyz, "-os", "gfnff", "-es", "1"]).unwrap();
    assert!(run_with(&cfg, &dir, None, &CalculatorRegistry::native()).is_err());
}

#[test]
fn a_symmetric_start_reaches_a_stationary_point_of_the_seam() {
    let dir = scratch("symmetric");
    // Water in C2v: the crossing of the second and third excited state. The coupling
    // direction breaks the symmetry and does not appear in the gradients of the search.
    let start = file(
        &dir,
        "water.xyz",
        "3\n\nO 0.0 0.0 0.12\nH 0.0 0.78 -0.47\nH 0.0 -0.78 -0.47\n",
    );
    let run = |extra: &[&str]| {
        let mut args = vec![
            start.as_str(),
            "-os",
            "gfn0xtb",
            "-es",
            "2",
            "-mf",
            "opt_meci",
            "-opt",
            "rsirfo_bfgs",
            "-freq",
        ];
        args.extend_from_slice(extra);
        let cfg = optmain::config_from_cli(&args).unwrap();
        let out = optmain::run(&cfg, &dir, None).unwrap();
        (
            std::fs::read_to_string(out[0].directory.join("seam.txt")).unwrap(),
            std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap(),
        )
    };
    // Without the check the search stops as soon as the gap is closed.
    let (report, log) = run(&["-no_crossing_check"]);
    assert!(
        report.contains("not a stationary point of the seam"),
        "{report}"
    );
    assert!(!log.contains("Check of the branching plane"));
    // With it (the default) the search is sent on and ends at a minimum of the seam.
    let (report, log) = run(&[]);
    assert!(
        log.contains("the search goes on with this branching plane"),
        "{log}"
    );
    assert!(
        !report.contains("not a stationary point of the seam"),
        "{report}"
    );
    assert!(report.contains("a minimum of the seam"), "{report}");
    assert!(
        report.contains("from the gradients at displaced structures"),
        "{report}"
    );
}

#[test]
fn an_alloy_cluster_and_a_crystal_with_the_embedded_atom_method() {
    let dir = scratch("eam");
    // A cluster of copper and silver: optimized, all six vibrations are real.
    let start = file(
        &dir,
        "cluster.xyz",
        "4\n\nCu 0.0 0.0 0.0\nCu 2.5 0.1 0.0\nAg 1.2 2.3 0.1\nAg 1.3 0.8 2.2\n",
    );
    let cfg =
        optmain::config_from_cli(&[&start, "-os", "eam", "-opt", "rsirfo_bfgs", "-tcc"]).unwrap();
    let optimized = optmain::run(&cfg, &dir, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[&xyz, "-os", "eam"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let found = frequencies(&out[0].directory);
    assert_eq!(found.len(), 6);
    assert!(found.iter().all(|w| *w > 10.0), "{found:?}");
    // Copper in its cell: the nine vibrations of the cell at the zone centre.
    let a = 2.556162 * 2.0_f64.sqrt();
    let crystal = file(
        &dir,
        "cu.xyz",
        &format!(
            "4\nLattice=\"{a} 0.0 0.0 0.0 {a} 0.0 0.0 0.0 {a}\" pbc=\"T T T\"\n\
             Cu 0.0 0.0 0.0\nCu {h} {h} 0.0\nCu {h} 0.0 {h}\nCu 0.0 {h} {h}\n",
            h = a / 2.0
        ),
    );
    let cfg = config_from_cli(&[&crystal, "-os", "eam"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let cell = frequencies(&out[0].directory);
    assert_eq!(cell.len(), 9);
    assert!(cell.iter().all(|w| *w > 50.0 && *w < 400.0), "{cell:?}");
    // An element without parameters is refused.
    let carbon = file(&dir, "c.xyz", "2\n\nCu 0.0 0.0 0.0\nC 2.0 0.0 0.0\n");
    let cfg = config_from_cli(&[&carbon, "-os", "eam"]).unwrap();
    assert!(run_with(&cfg, &dir, None, &CalculatorRegistry::native()).is_err());
}

#[test]
fn a_cluster_and_a_crystal_with_the_modified_embedded_atom_method() {
    let dir = scratch("meam");
    // A cluster of copper and nickel: optimized, all six vibrations are real.
    let start = file(
        &dir,
        "cluster.xyz",
        "4\n\nCu 0.0 0.0 0.0\nCu 2.5 0.1 0.0\nNi 1.2 2.2 0.1\nNi 1.3 0.8 2.1\n",
    );
    let pair = "pair=Cu-Ni,fcc,4.0,2.52,5.05";
    let cfg = optmain::config_from_cli(&[
        &start,
        "-os",
        "meam",
        "-opt",
        "rsirfo_bfgs",
        "-tcc",
        "-calc_opt",
        pair,
    ])
    .unwrap();
    let optimized = optmain::run(&cfg, &dir, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[&xyz, "-os", "meam", "-calc_opt", pair]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let found = frequencies(&out[0].directory);
    assert_eq!(found.len(), 6);
    assert!(found.iter().all(|w| *w > 10.0), "{found:?}");
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("modified embedded-atom method"), "{log}");
    assert!(!log.contains("not fitted"), "{log}");
    // Without a reference for the pair the log says what is used instead.
    let cfg = config_from_cli(&[&xyz, "-os", "meam"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("no reference given for Ni-Cu"), "{log}");
    // Iron in its cell of two atoms: three vibrations at the zone centre.
    let a = 2.866;
    let crystal = file(
        &dir,
        "fe.xyz",
        &format!(
            "2\nLattice=\"{a} 0.0 0.0 0.0 {a} 0.0 0.0 0.0 {a}\" pbc=\"T T T\"\n\
             Fe 0.0 0.0 0.0\nFe {h} {h} {h}\n",
            h = a / 2.0
        ),
    );
    let cfg = config_from_cli(&[&crystal, "-os", "meam"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let cell = frequencies(&out[0].directory);
    assert_eq!(cell.len(), 3);
    assert!(cell.iter().all(|w| *w > 100.0 && *w < 500.0), "{cell:?}");
    // Settings that are not known and elements without parameters are refused.
    let cfg = config_from_cli(&[&crystal, "-os", "meam", "-calc_opt", "rc=5"]).unwrap();
    assert!(run_with(&cfg, &dir, None, &CalculatorRegistry::native()).is_err());
    let zinc = file(&dir, "zn.xyz", "2\n\nCu 0.0 0.0 0.0\nZn 2.5 0.0 0.0\n");
    let cfg = config_from_cli(&[&zinc, "-os", "meam"]).unwrap();
    assert!(run_with(&cfg, &dir, None, &CalculatorRegistry::native()).is_err());
}

#[test]
fn cutoffs_with_switching_from_the_command_line() {
    let dir = scratch("switching");
    // Argon in its cell of four atoms, the cutoff longer than the cell: all images within
    // it, switched off over the last Angstrom.
    let a = 5.25;
    let crystal = file(
        &dir,
        "ar.xyz",
        &format!(
            "4\nLattice=\"{a} 0.0 0.0 0.0 {a} 0.0 0.0 0.0 {a}\" pbc=\"T T T\"\n\
             Ar 0.0 0.0 0.0\nAr {h} {h} 0.0\nAr {h} 0.0 {h}\nAr 0.0 {h} {h}\n",
            h = a / 2.0
        ),
    );
    let run = |options: &[&str]| {
        let mut args = vec![crystal.as_str(), "-os", "lj"];
        if !options.is_empty() {
            args.push("-calc_opt");
            args.extend_from_slice(options);
        }
        let cfg = config_from_cli(&args)?;
        run_with(&cfg, &dir, None, &CalculatorRegistry::native())
    };
    let out = run(&["cutoff=8.0", "switch=1.0"]).unwrap();
    let found = frequencies(&out[0].directory);
    assert_eq!(found.len(), 9);
    assert!(found.iter().all(|w| *w > 10.0 && *w < 150.0), "{found:?}");
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("pairs within 8.0000 Angstrom, switched off from 7.0000"),
        "{log}"
    );
    // Without a cutoff: all pairs, as in MultiOptPy.
    let out = run(&[]).unwrap();
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(!log.contains("pairs within"), "{log}");
    assert!(run(&["switch=1.0"]).is_err());
    assert!(run(&["cutoff=8.0", "switch=9.0"]).is_err());
    assert!(run(&["rc=8.0"]).is_err());
    // The embedded-atom potentials take the width of their switching range.
    let copper = file(&dir, "cu.xyz", "2\n\nCu 0.0 0.0 0.0\nCu 2.5 0.0 0.0\n");
    for (name, options, fine) in [
        ("eam", "switch=0", true),
        ("eam", "switch=0.8", true),
        ("eam", "width=0.8", false),
        ("emt", "switch=0.5", true),
        ("emt", "switch=-1", false),
    ] {
        let cfg = config_from_cli(&[&copper, "-os", name, "-calc_opt", options]).unwrap();
        let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native());
        assert_eq!(out.is_ok(), fine, "{name} {options}");
    }
}

#[test]
fn the_vibrations_of_a_hessian_that_is_read() {
    use mopt_rs::domain::{EvalRequest, SpinCharge};
    use mopt_rs::io::npy::write_npy_matrix;
    use mopt_rs::io::xyz::read_xyz_file;
    use mopt_rs::pes::registry::CalculatorOptions;

    let dir = scratch("read_hessian");
    let start = file(&dir, "water.xyz", WATER);
    let cfg =
        optmain::config_from_cli(&[&start, "-os", "gfnff", "-opt", "rsirfo_bfgs", "-tcc"]).unwrap();
    let optimized = optmain::run(&cfg, &dir, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let registry = CalculatorRegistry::native();
    let own = run_with(
        &config_from_cli(&[&xyz, "-os", "gfnff"]).unwrap(),
        &dir,
        None,
        &registry,
    )
    .unwrap();
    let own = frequencies(&own[0].directory);

    // Four times the Hessian of the calculator: twice the wavenumbers.
    let water = read_xyz_file(&xyz, SpinCharge::default(), None).unwrap();
    let calculator = registry
        .create("gfnff", &CalculatorOptions::default(), &water)
        .unwrap();
    let hessian = calculator
        .evaluate(&water, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap();
    let stiff = faer::Mat::from_fn(9, 9, |r, c| 4.0 * hessian[(r, c)]);
    let npy = dir.join("stiff.npy");
    write_npy_matrix(&npy, &stiff).unwrap();
    let read = run_with(
        &config_from_cli(&[&xyz, "-os", "gfnff", "-ih", &npy.to_string_lossy()]).unwrap(),
        &dir,
        None,
        &registry,
    )
    .unwrap();
    let read = frequencies(&read[0].directory);
    assert_eq!(read.len(), 3);
    for (a, b) in read.iter().zip(&own) {
        assert!((a - 2.0 * b).abs() < 1e-3 * b, "{a} against twice {b}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
