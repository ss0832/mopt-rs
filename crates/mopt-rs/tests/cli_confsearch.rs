//! `confsearch` from the command line: the conformers of butane with the built-in GFN-FF by
//! MultiOptPy's AFIR sampling and by iMTD-GC, MultiOptPy's files, the restart from the
//! journal, the mode that collects every minimum, ONIOM (GFN0-xTB in GFN-FF), a cluster in a
//! periodic box, and the combinations that are refused.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mopt_rs::cli::confsearch::{config_from_cli, run_with, ConfsearchResult};
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, SpinCharge, Units};
use mopt_rs::error::MoptResult;
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::journal::{read_journal, JOURNAL_FILE};

/// n-butane, all anti.
const BUTANE: &str = "14\nn-butane all-anti\nC 0.000000 0.000000 0.000000\n\
C 1.268427 0.855565 0.000000\nC 2.536855 0.000000 0.000000\nC 3.805282 0.855565 0.000000\n\
H -0.875864 0.648816 0.000000\nH -0.013866 -0.629150 0.889988\nH -0.013866 -0.629150 -0.889988\n\
H 1.268427 1.484886 0.889975\nH 1.268427 1.484886 -0.889975\nH 2.536855 -0.629321 -0.889975\n\
H 2.536855 -0.629321 0.889975\nH 4.681147 0.206750 0.000000\nH 3.819149 1.484715 0.889988\n\
H 3.819149 1.484715 -0.889988\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_confsearch_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn input(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

/// GFN-FF counting its evaluations.
struct Counting(Box<dyn Calculator>, Arc<AtomicUsize>);

impl Calculator for Counting {
    fn name(&self) -> &str {
        "counting_gfnff"
    }
    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        self.1.fetch_add(1, Ordering::SeqCst);
        self.0.evaluate(sys, req)
    }
}

fn counting(count: &Arc<AtomicUsize>) -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    let c = Arc::clone(count);
    r.register("counting", move |request| {
        let inner =
            CalculatorRegistry::native().create("gfnff", &Default::default(), request.system)?;
        Ok(Box::new(Counting(inner, Arc::clone(&c))))
    });
    r
}

fn kcal(a: f64, b: f64) -> f64 {
    (a - b) * Units::HARTREE_TO_KCAL_MOL
}

/// The AFIR sampling of butane with one pair: a terminal carbon and a hydrogen atom out of the
/// plane of the carbons at the other end (the pull of the two terminal carbons of the planar
/// anti structure keeps its symmetry and turns nothing).
fn afir_butane(dir: &Path, file: &str, extra: &[&str]) -> (ConfsearchResult, Arc<AtomicUsize>) {
    let mut args = vec![
        file, "-os", "counting", "-tgta", "1,13", "-ms", "2", "-cs_opt", "seed=3",
    ];
    args.extend_from_slice(extra);
    let config = config_from_cli(&args).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let out = run_with(&config, dir, None, &counting(&count)).unwrap();
    (out, count)
}

#[test]
fn afir_sampling_finds_the_gauche_conformer_and_writes_multioptpy_files() {
    let dir = scratch("afir");
    let file = input(&dir, "butane.xyz", BUTANE);
    let (out, _) = afir_butane(&dir, &file, &[]);
    assert!(!out.interrupted);
    assert_eq!(out.count, 2, "anti and gauche: {:?}", out.energies);
    // GFN-FF puts gauche butane 0.5 to 1 kcal/mol above anti.
    let gap = kcal(out.energies[1], out.energies[0]);
    assert!(gap > 0.2 && gap < 1.5, "{gap}");
    let run = &out.directory;
    assert!(run
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("butane_100KJ_CS_"));
    for name in [
        "butane_EQ0.xyz",
        "butane_EQ1.xyz",
        "EQ_energy.dat",
        "EQ_summary.log",
        "switch_conformer.log",
        "search_atom_pairs.log",
        "reference_bond_table.dat",
        "conformers.xyz",
        "rotamers.xyz",
        "conformers.csv",
        "ensemble.txt",
        "settings.json",
        "mopt.log",
    ] {
        assert!(run.join(name).is_file(), "{name}");
    }
    let energies = std::fs::read_to_string(run.join("EQ_energy.dat")).unwrap();
    assert_eq!(energies.lines().count(), 2);
    let ensemble =
        read_xyz_trajectory_file(run.join("conformers.xyz"), SpinCharge::default(), None).unwrap();
    assert_eq!(ensemble.frames.len(), 2);
    let csv = std::fs::read_to_string(run.join("conformers.csv")).unwrap();
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(rows.len(), 3);
    assert!(rows[0].starts_with("rank,eq,energy_hartree,relative_kcal_mol,boltzmann_weight"));
    // The gauche conformer is chiral: it stands for both of its mirror images.
    assert!(rows[2].contains(",2,"), "{}", rows[2]);
    let pairs = std::fs::read_to_string(run.join("search_atom_pairs.log")).unwrap();
    assert_eq!(pairs.lines().count(), 2, "{pairs}");
    let text = std::fs::read_to_string(run.join("ensemble.txt")).unwrap();
    assert!(text.contains("ensemble entropy"), "{text}");
    let log = std::fs::read_to_string(run.join("mopt.log")).unwrap();
    assert!(log.contains("Sampling conformation:  0"), "{log}");
    // Both entries of the pair list are used (MultiOptPy stops one entry early).
    assert!(log.contains("Sampling conformation:  1"), "{log}");
    assert!(log.contains("Find new conformer."), "{log}");
    assert!(log.contains("gradient calculation(s)"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn coordinates_constraints_and_bias_potentials_pass_through() {
    let dir = scratch("surfaces");
    let file = input(&dir, "butane.xyz", BUTANE);
    // Optimizations in delocalized internal coordinates.
    let (out, _) = afir_butane(&dir, &file, &["-coord", "dic"]);
    assert_eq!(out.count, 2, "{:?}", out.energies);
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(log.contains("-coord dic"), "{log}");
    // A restraint that holds the central dihedral angle near anti: only anti is left.
    let (held, _) = afir_butane(&dir, &file, &["-kda", "0.1", "180", "1,2,3,4"]);
    assert_eq!(held.count, 1, "{:?}", held.energies);
    // A fixed atom: the conformers are found all the same.
    let (fixed, _) = afir_butane(&dir, &file, &["-fix", "1"]);
    assert_eq!(fixed.count, 2, "{:?}", fixed.energies);
    // ... and every conformer written has the fixed atom where it is in the input (10
    // decimals of Angstrom in the file).
    let start = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    let written = read_xyz_trajectory_file(&fixed.conformers, SpinCharge::default(), None).unwrap();
    assert_eq!(written.frames.len(), 2);
    for f in &written.frames {
        let (p, q) = (
            f.system.position(mopt_rs::domain::AtomId::new(0)).value(),
            start.position(mopt_rs::domain::AtomId::new(0)).value(),
        );
        for c in 0..3 {
            assert!((p[c] - q[c]).abs() < 2e-10, "{p:?} {q:?}");
        }
    }
    // The restraint keeps the dihedral near anti in every conformer.
    let written = read_xyz_trajectory_file(&held.conformers, SpinCharge::default(), None).unwrap();
    for f in &written.frames {
        let set = mopt_rs::pes::constraint::ConstraintSet::new(
            &[mopt_rs::pes::constraint::Constraint::with_target(
                mopt_rs::pes::constraint::ConstraintKind::Dihedral(
                    [0, 1, 2, 3].map(mopt_rs::domain::AtomId::new),
                ),
                std::f64::consts::PI,
            )],
            &f.system,
            None,
        )
        .unwrap();
        let r = set.residuals(&f.system).unwrap()[0];
        assert!(r.abs() < 0.2, "{r}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// n-Pentane, planar, with the last methyl group eclipsed: a saddle point by symmetry.
const ECLIPSED_PENTANE: &str = "17\nn-pentane\nC 0.000000 0.000000 0.000000\n\
C 1.268427 0.855565 0.000000\nC 2.536855 0.000000 0.000000\nC 3.805282 0.855565 0.000000\n\
C 5.073710 0.000000 0.000000\nH -0.875864 0.648816 0.000000\nH -0.013866 -0.629150 0.889988\n\
H -0.013866 -0.629150 -0.889988\nH 1.268427 1.484886 0.889975\nH 1.268427 1.484886 -0.889975\n\
H 2.536855 -0.629321 -0.889975\nH 2.536855 -0.629321 0.889975\nH 3.805282 1.484886 0.889975\n\
H 3.805282 1.484886 -0.889975\nH 4.800243 -1.055138 0.000000\nH 5.662242 0.222827 0.889988\n\
H 5.662242 0.222827 -0.889988\n";

#[test]
fn the_optimized_input_is_checked_by_its_hessian() {
    // The optimization keeps the symmetry of the input and ends at the eclipsed saddle point;
    // the check of its Hessian goes on to the staggered minimum, 2.8 kcal/mol lower.
    let dir = scratch("verify");
    let file = input(&dir, "pentane.xyz", ECLIPSED_PENTANE);
    let run = |extra: &str| {
        let config = config_from_cli(&[
            &file,
            "-os",
            "gfnff",
            "-ms",
            "0",
            "-cs_opt",
            "seed=1",
            "nudge=off",
            extra,
        ])
        .unwrap();
        run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap()
    };
    let checked = run("verify=on");
    let unchecked = run("verify=off");
    let gain = kcal(unchecked.energies[0], checked.energies[0]);
    assert!(gain > 2.0 && gain < 3.5, "{gain}");
    let log = std::fs::read_to_string(checked.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("The optimized input is a saddle point"),
        "{log}"
    );
    assert!(!log.contains(" 0 exact Hessian calculation(s)"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_restart_replays_the_journal_and_gives_the_same_ensemble() {
    let dir = scratch("restart");
    let file = input(&dir, "butane.xyz", BUTANE);
    let (full, count) = afir_butane(&dir, &file, &[]);
    let total = count.load(Ordering::SeqCst);
    let journal = full.directory.join(JOURNAL_FILE);
    assert_eq!(read_journal(&journal).unwrap().0.len(), total);
    let reference = std::fs::read_to_string(full.directory.join("conformers.xyz")).unwrap();
    // Killed after half of the evaluations.
    let k = total / 2;
    let bytes = std::fs::read(&journal).unwrap();
    let mut cut = 8usize;
    for _ in 0..k {
        let len = u32::from_le_bytes(bytes[cut..cut + 4].try_into().unwrap()) as usize;
        cut += 8 + len;
    }
    std::fs::write(&journal, &bytes[..cut]).unwrap();
    let mut config = config_from_cli(&[
        &file, "-os", "counting", "-tgta", "1,13", "-ms", "2", "-cs_opt", "seed=3",
    ])
    .unwrap();
    config.restart_dir = Some(full.directory.clone());
    count.store(0, Ordering::SeqCst);
    let again = run_with(&config, &dir, None, &counting(&count)).unwrap();
    assert_eq!(again.directory, full.directory);
    // Every kept record is replayed: only the evaluations after them are made again.
    assert_eq!(count.load(Ordering::SeqCst), total - k);
    let log = std::fs::read_to_string(again.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains(&format!("Restart: {k} evaluations taken")),
        "{log}"
    );
    assert_eq!(
        std::fs::read_to_string(again.directory.join("conformers.xyz")).unwrap(),
        reference
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn imtd_gc_finds_the_conformers_of_butane() {
    let dir = scratch("imtd");
    let file = input(&dir, "butane.xyz", BUTANE);
    let config = config_from_cli(&[
        &file,
        "-os",
        "gfnff",
        "-cs_opt",
        "method=imtd",
        "seed=2",
        "mtd_runs=3:1.3",
        "mtd_time=0.4",
        "dump=100",
        "rounds=1",
        "finish=off",
    ])
    .unwrap();
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(!out.interrupted);
    assert_eq!(out.count, 2, "{:?}", out.energies);
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Metadynamics round 1"), "{log}");
    assert!(log.contains("Genetic crossing"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Ar7, a distorted pentagonal bipyramid.
const AR7: &str = "7\n\nAr 1.9500 0.0000 0.0000\nAr 0.6026 1.8546 0.0300\n\
Ar -1.5776 1.1462 0.0700\nAr -1.5776 -1.1462 0.1100\nAr 0.6026 -1.8546 0.1500\n\
Ar 0.0000 0.0000 2.1000\nAr 0.2000 0.0000 -2.0000\n";

#[test]
fn the_minima_mode_keeps_what_the_conformer_mode_rejects() {
    // With bonds between neighbouring argon atoms (2.2 times the covalent radii), the
    // isomers of LJ7 have other contact graphs: the conformer mode rejects them, the minima
    // mode keeps them.
    let dir = scratch("minima");
    let file = input(&dir, "ar7.xyz", AR7);
    let run = |extra: &[&str]| {
        let mut args = vec![
            &*file,
            "-os",
            "lj",
            "-ms",
            "8",
            "-cs_opt",
            "seed=1",
            "bond_scale=2.2",
        ];
        args.extend_from_slice(extra);
        run_with(
            &config_from_cli(&args).unwrap(),
            &dir,
            None,
            &CalculatorRegistry::native(),
        )
        .unwrap()
    };
    let conformers = run(&[]);
    let minima = run(&["mode=minima"]);
    assert_eq!(conformers.count, 1);
    assert!(minima.count >= 2, "{:?}", minima.energies);
    let rejected = conformers.directory.join("REJECTED_CONFORMERS");
    assert!(rejected.join("rejected_count.dat").is_file());
    assert!(rejected.join("ar7_REJECTED0_info.txt").is_file());
    assert!(!minima.directory.join("REJECTED_CONFORMERS").exists());
    let csv = std::fs::read_to_string(minima.directory.join("conformers.csv")).unwrap();
    assert!(csv.contains(",false,"), "{csv}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn oniom_with_gfn0_xtb_in_gfn_ff() {
    // Ethanol (GFN0-xTB) with one water molecule (GFN-FF for the whole).
    let dir = scratch("oniom");
    let text = "12\nethanol and water\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n\
O 4.800 1.400 0.000\nH 5.200 2.270 0.000\nH 5.300 0.800 0.500\n";
    let file = input(&dir, "solvated.xyz", text);
    let config = config_from_cli(&[
        &file, "-os", "gfn0xtb", "-oniom", "1-9", "none", "gfnff", "-tgta", "1,3,4", "-ms", "2",
        "-cs_opt", "seed=4",
    ])
    .unwrap();
    let registry = CalculatorRegistry::native();
    let out = run_with(&config, &dir, None, &registry).unwrap();
    assert!(out.count >= 1);
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(log.contains("ONIOM with 2 levels"), "{log}");
    // The conformers of the core, with the environment relaxed for every structure.
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    assert!(!log.contains("without microiterations"), "{log}");
    // Every conformer is a minimum of the ONIOM surface.
    let calc = registry
        .create_oniom(
            &config.optmain.calculator,
            &config.optmain.calculator_options,
            &read_xyz_file(&file, SpinCharge::default(), None).unwrap(),
        )
        .unwrap()
        .at_given_structures();
    let ensemble = read_xyz_trajectory_file(
        out.directory.join("conformers.xyz"),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    for frame in &ensemble.frames {
        let g = calc
            .evaluate(&frame.system, &EvalRequest::ENERGY_GRAD)
            .unwrap();
        let largest = g.gradient_bohr.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(largest < 1e-3, "{largest}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_cluster_in_a_periodic_box_and_a_network_that_is_refused() {
    let dir = scratch("periodic");
    // Ar7 in a cubic box of 30 Angstrom.
    let ar7 = "7\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0\" pbc=\"T T T\"\n\
Ar 1.9500 0.0000 0.0000\nAr 0.6026 1.8546 0.0300\nAr -1.5776 1.1462 0.0700\n\
Ar -1.5776 -1.1462 0.1100\nAr 0.6026 -1.8546 0.1500\nAr 0.0000 0.0000 2.1000\n\
Ar 0.2000 0.0000 -2.0000\n";
    let file = input(&dir, "ar7.xyz", ar7);
    let config = config_from_cli(&[&file, "-os", "lj", "-ms", "6", "-cs_opt", "seed=2"]).unwrap();
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out.count >= 1);
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Periodic cell"), "{log}");
    // The lowest is the pentagonal bipyramid of LJ7 (-16.505 epsilon).
    let eps = 0.237 / Units::HARTREE_TO_KCAL_MOL;
    assert!(
        (out.energies[0] / eps + 16.505).abs() < 0.01,
        "{}",
        out.energies[0] / eps
    );
    // A carbon chain through a short cell is a network without conformers.
    let chain = "3\nLattice=\"4.35 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0\" pbc=\"T T T\"\n\
C 0.0 0.0 0.0\nC 1.45 0.0 0.0\nC 2.9 0.0 0.0\n";
    let file = input(&dir, "chain.xyz", chain);
    let config =
        config_from_cli(&[&file, "-os", "tersoff", "-ms", "1", "-cs_opt", "seed=1"]).unwrap();
    let e = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(e.to_string().contains("network"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 2-ethyl-1,5-hexadiene at a chair transition state of its Cope rearrangement, the ethyl
/// group in its higher position (GFN0-xTB).
const ETHYL_COPE_TS: &str = "22\n2-ethyl-1,5-hexadiene, Cope chair transition state\n\
C -0.670891 -1.140282 0.707891\nC -0.018031 0.069167 1.021164\nC -0.742897 1.255670 0.789091\n\
C 1.480696 0.123450 1.119992\nH -1.695899 -1.228954 1.051561\nH -0.102712 -2.052829 0.855736\n\
H -0.236818 2.186905 1.022052\nH -1.780232 1.255681 1.107114\nC -0.966215 -1.090412 -1.055793\n\
C -1.649737 0.119372 -1.297169\nC -0.963978 1.310132 -0.976963\nH -2.728728 0.118383 -1.180740\n\
H -1.511408 -2.009559 -1.240122\nH 0.048555 -1.134783 -1.433086\nH 0.065364 1.362096 -1.316915\n\
H -1.494970 2.242987 -1.128999\nC 2.212228 -0.421121 -0.105863\nH 1.790507 -0.465669 1.986976\n\
H 1.788829 1.158068 1.281889\nH 3.287975 -0.340669 0.042135\nH 1.933329 0.150209 -0.989993\n\
H 1.955035 -1.467843 -0.259956\n";

#[test]
fn conformers_of_a_transition_state() {
    // From the transition state whose ethyl group points at the other allyl fragment, the
    // search with the saddle direction held turns the group and finds the transition state
    // 0.78 kcal/mol lower (GFN0-xTB), with one imaginary frequency and the same reaction.
    let dir = scratch("ts");
    let file = input(&dir, "ethylcope.xyz", ETHYL_COPE_TS);
    // One AFIR sample between the end of the ethyl group and a terminal carbon; an IRC of
    // larger steps (its ends are optimized afterwards) keeps the test short.
    let config = config_from_cli(&[
        &file,
        "-os",
        "gfn0xtb",
        "-tsconf",
        "-tgta",
        "1,17",
        "-ms",
        "1",
        "-irc",
        "0.3",
        "100",
        "lqa",
        "-cs_opt",
        "seed=1",
        "pairs=ratio",
    ])
    .unwrap();
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(!out.interrupted);
    assert_eq!(out.count, 2, "{:?}", out.energies);
    let gap = kcal(out.energies[1], out.energies[0]);
    assert!(gap > 0.5 && gap < 1.2, "{gap}");
    let csv = std::fs::read_to_string(out.directory.join("ts_conformers.csv")).unwrap();
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(rows.len(), 3, "{csv}");
    // The lowest is the new one: one imaginary frequency, the same reaction.
    assert!(rows[1].contains(",true,"), "{csv}");
    assert!(!rows[1].ends_with("\"input\""), "{csv}");
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Reactive bonds held"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refused_combinations_say_why() {
    let dir = scratch("refused");
    let file = input(&dir, "butane.xyz", BUTANE);
    for (args, why) in [
        (vec![file.as_str(), "-vc"], "cell"),
        (vec![file.as_str(), "-order", "1"], "-tsconf"),
        (vec![file.as_str(), "-km"], "one optimization"),
        (vec![file.as_str(), "-ih", "h.npy"], "Hessian"),
    ] {
        let e = config_from_cli(&args).unwrap_err().to_string();
        assert!(e.contains(why), "{args:?}: {e}");
    }
    let config = config_from_cli(&[&file, "-os", "mb2d"]).unwrap();
    let e = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(e.to_string().contains("plain variables"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A move of a user: the first carbon atom and its hydrogen atoms turned about the bond to
/// the second by a third of a turn, then by two thirds, then nothing more.
struct TurnTheEnd;

impl mopt_rs::workflow::confsearch::moves::ConformerMove for TurnTheEnd {
    fn name(&self) -> &str {
        "turn_the_end"
    }

    fn propose(
        &mut self,
        current: &MolecularSystem,
        context: &mopt_rs::workflow::confsearch::moves::MoveContext<'_>,
        _rng: &mut mopt_rs::core_infra::rng::Rng,
    ) -> MoptResult<Option<mopt_rs::workflow::confsearch::moves::Proposal>> {
        if context.trial >= 2 {
            return Ok(None);
        }
        // About the central bond (atoms 2 and 3): the half with atoms 3 and 4 is turned.
        let angle = 2.0 * std::f64::consts::PI / 3.0 * (context.trial + 1) as f64;
        let mut x = current.coords_flat_bohr();
        let at = |x: &faer::Col<f64>, a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
        let (p1, p2) = (at(&x, 1), at(&x, 2));
        let mut axis = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        for a in &mut axis {
            *a /= n;
        }
        let (c, s) = (angle.cos(), angle.sin());
        for a in [3, 9, 10, 11, 12, 13] {
            let p = at(&x, a);
            let v = [p[0] - p2[0], p[1] - p2[1], p[2] - p2[2]];
            let dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
            let cross = [
                axis[1] * v[2] - axis[2] * v[1],
                axis[2] * v[0] - axis[0] * v[2],
                axis[0] * v[1] - axis[1] * v[0],
            ];
            for k in 0..3 {
                x[3 * a + k] = p2[k] + v[k] * c + cross[k] * s + axis[k] * dot * (1.0 - c);
            }
        }
        Ok(Some(mopt_rs::workflow::confsearch::moves::Proposal {
            start: current.with_flat_coords_bohr(&x)?,
            what: format!("the end turned by {:.0} degrees", angle.to_degrees()),
        }))
    }
}

#[test]
fn searches_by_moves_built_in_and_of_a_user() {
    use mopt_rs::cli::confsearch::run_with_moves;
    use mopt_rs::workflow::confsearch::moves::{ConformerMove, MoveRegistry};

    let dir = scratch("moves");
    let file = input(&dir, "butane.xyz", BUTANE);
    let registry = CalculatorRegistry::native();
    // Turns about the rotatable bonds find the gauche conformer of butane.
    let config = config_from_cli(&[
        &file,
        "-os",
        "gfnff",
        "-ms",
        "12",
        "-cs_opt",
        "method=torsion",
        "seed=2",
    ])
    .unwrap();
    let out = run_with(&config, &dir, None, &registry).unwrap();
    assert!(out.count >= 2, "{}", out.count);
    let gauche = kcal(out.energies[1], out.energies[0]);
    assert!(gauche > 0.2 && gauche < 1.5, "{gauche}");
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("Strategy: trials by the move torsion"),
        "{log}"
    );
    assert!(log.contains("Rotatable bonds: 3"), "{log}");
    // The other built-in moves run, with their settings.
    for extra in [
        vec!["method=kick", "move_radius=0.5"],
        vec!["method=lowmode", "move_modes=3", "move_step=1.2"],
    ] {
        let mut args = vec![
            file.as_str(),
            "-os",
            "gfnff",
            "-ms",
            "4",
            "-cs_opt",
            "seed=5",
        ];
        args.extend(extra.iter().copied());
        let config = config_from_cli(&args).unwrap();
        let out = run_with(&config, &dir, None, &registry).unwrap();
        assert!(out.count >= 1);
    }
    // A move of a user, registered by name: it proposes the two gauche conformers and then
    // has nothing more to try.
    let mut moves = MoveRegistry::builtin();
    moves
        .register(
            "turn_the_end",
            Arc::new(|options| {
                options.only("turn_the_end", &[])?;
                Ok(Box::new(TurnTheEnd) as Box<dyn ConformerMove>)
            }),
        )
        .unwrap();
    let config = config_from_cli(&[
        &file,
        "-os",
        "gfnff",
        "-ms",
        "9",
        "-nost",
        "-cs_opt",
        "method=turn_the_end",
    ])
    .unwrap();
    let out = run_with_moves(&config, &dir, None, &registry, &moves).unwrap();
    assert_eq!(
        out.count, 2,
        "anti and gauche (its mirror image is the same conformer)"
    );
    assert!(out.reason.contains("nothing more to try"), "{}", out.reason);
    // Refused before anything is computed: a move that is not there, a setting the move
    // does not have, and a setting of a move with another strategy.
    for bad in [
        vec!["method=turn_the_end"],
        vec!["method=kick", "move_bonds=2"],
        vec!["method=afir", "move_radius=0.5"],
    ] {
        let mut args = vec![file.as_str(), "-os", "gfnff", "-cs_opt"];
        args.extend(bad.iter().copied());
        let refused = config_from_cli(&args).and_then(|c| run_with(&c, &dir, None, &registry));
        assert!(refused.is_err(), "{bad:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
