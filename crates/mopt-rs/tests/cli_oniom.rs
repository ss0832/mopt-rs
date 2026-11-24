//! ONIOM from the command line: ethanol with GFN0-xTB on the hydroxymethyl group and GFN-FF
//! on the whole molecule (`-oniom`), three layers (`-oniom_middle`), the vibrations of the
//! optimized structure on the ONIOM surface (`freqmain -oniom`), and the other commands that
//! take the same options.
//!
//! MultiOptPy: `-oniom` (read, not run: its handler raises an exception).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::{freqmain, ircmain, nebmain, optmain};
use mopt_rs::domain::{EvalRequest, SpinCharge};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::registry::CalculatorRegistry;

const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_oniom_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn input(dir: &Path) -> String {
    let p = dir.join("ethanol.xyz");
    std::fs::write(&p, ETHANOL).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn an_optimization_on_the_oniom_surface_and_its_vibrations() {
    let dir = scratch("opt");
    let file = input(&dir);
    let registry = CalculatorRegistry::native();
    let args = [
        file.as_str(),
        "-os",
        "gfn0xtb",
        "-oniom",
        "2-4,8,9",
        "auto",
        "GFN-FF",
        "-tcc",
    ];
    let config = optmain::config_from_cli(&args).unwrap();
    let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(out[0].converged);
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("ONIOM with 2 levels"), "{log}");
    assert!(log.contains("1 link atom(s)"), "{log}");
    // The optimized structure is a stationary point of the ONIOM surface.
    let optimized = read_xyz_file(&out[0].optimized, SpinCharge::default(), None).unwrap();
    let calc = registry
        .create_oniom(&config.calculator, &config.calculator_options, &optimized)
        .unwrap()
        .at_given_structures();
    let at = calc
        .evaluate(&optimized, &EvalRequest::ENERGY_GRAD)
        .unwrap();
    let largest = at.gradient_bohr.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(largest < 1e-4, "{largest}");
    // Microiterations by default: the core moved, the rest was relaxed to a minimum before
    // every evaluation, and the files hold the whole system.
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    assert_eq!(optimized.n_atoms(), 9);
    // micro=off: every atom moves in the optimizer, to the same stationary point.
    let mut off = args.to_vec();
    off.extend(["-oniom_opt", "micro=off"]);
    let config_off = optmain::config_from_cli(&off).unwrap();
    let out_off = optmain::run_with(&config_off, &dir, None, &registry).unwrap();
    assert!(out_off[0].converged);
    let log_off = std::fs::read_to_string(out_off[0].directory.join("mopt.log")).unwrap();
    assert!(!log_off.contains("ONIOM microiterations:"), "{log_off}");
    let other = read_xyz_file(&out_off[0].optimized, SpinCharge::default(), None).unwrap();
    let e = |s: &mopt_rs::domain::MolecularSystem| {
        calc.evaluate(s, &EvalRequest::ENERGY_GRAD)
            .unwrap()
            .energy
            .value()
    };
    assert!(
        (e(&optimized) - e(&other)).abs() < 1e-6,
        "{} {}",
        e(&optimized),
        e(&other)
    );
    // Vibrations on the ONIOM surface: 3N - 6 real frequencies at the minimum.
    let freq = freqmain::config_from_cli(&[
        out[0].optimized.to_str().unwrap(),
        "-os",
        "gfn0xtb",
        "-oniom",
        "2-4,8,9",
        "auto",
        "gfnff",
    ])
    .unwrap();
    let analysed = freqmain::run_with(&freq, &dir, None, &registry).unwrap();
    let analysis = analysed[0].analysis.as_ref().unwrap();
    assert_eq!(analysis.thermo.imaginary, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_on_the_core_keep_the_microiterations() {
    let dir = scratch("constrained");
    let file = input(&dir);
    let registry = CalculatorRegistry::native();
    // The C-O bond (atoms 2 and 3, both in the model region) held at 1.50 Angstrom.
    let args = [
        file.as_str(),
        "-os",
        "gfn0xtb",
        "-oniom",
        "2-4,8,9",
        "auto",
        "gfnff",
        "-pc",
        "manual",
        "bond",
        "2,3",
        "1.50",
    ];
    let config = optmain::config_from_cli(&args).unwrap();
    let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(out[0].converged);
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    let optimized = read_xyz_file(&out[0].optimized, SpinCharge::default(), None).unwrap();
    assert_eq!(optimized.n_atoms(), 9);
    let bond = optimized
        .distance(
            mopt_rs::domain::AtomId::new(1),
            mopt_rs::domain::AtomId::new(2),
        )
        .value()
        * mopt_rs::domain::Units::BOHR_TO_ANGSTROM;
    assert!((bond - 1.50).abs() < 1e-6, "{bond}");
    // A constraint on an atom of the environment (H 5 of the methyl group): the atom is
    // moved by the optimizer with the core, and the run keeps its microiterations.
    let outside = [
        file.as_str(),
        "-os",
        "gfn0xtb",
        "-oniom",
        "2-4,8,9",
        "auto",
        "gfnff",
        "-pc",
        "bond",
        "1,5",
        "-ns",
        "3",
    ];
    let config = optmain::config_from_cli(&outside).unwrap();
    let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("moved by the optimizer with the core (5)"),
        "{log}"
    );
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    assert!(!log.contains("without microiterations"), "{log}");
    // What cannot be taken to the relaxed surface is refused, never run without them: the
    // nanoreactor acts on every atom, so nothing would be left to relax.
    let mut all = vec![file.as_str()];
    all.extend(["-os", "gfn0xtb", "-oniom", "2-4,8,9", "auto", "gfnff"]);
    all.extend(["-nrp", "8", "14", "1.5", "0.5", "1.0", "0.5", "-ns", "2"]);
    let config = optmain::config_from_cli(&all).unwrap();
    let refused = optmain::run_with(&config, &dir, None, &registry)
        .err()
        .unwrap();
    assert!(
        refused.to_string().contains("no environment to relax"),
        "{refused}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn three_layers_of_one_level_are_the_real_system_and_every_command_takes_the_options() {
    let dir = scratch("layers");
    let file = input(&dir);
    let registry = CalculatorRegistry::native();
    let start = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    let three = optmain::config_from_cli(&[
        file.as_str(),
        "-os",
        "gfnff",
        "-oniom",
        "2-4,8,9",
        "auto",
        "gfnff",
        "-oniom_middle",
        "1-4,8,9",
        "auto",
        "gfnff",
        "-oniom_opt",
        "link=distance",
    ])
    .unwrap();
    // The registry does not hand the ONIOM surface out as a calculator: a run says what
    // it is for.
    assert!(registry
        .create(&three.calculator, &three.calculator_options, &start)
        .is_err());
    let oniom = registry
        .create_oniom(&three.calculator, &three.calculator_options, &start)
        .unwrap()
        .at_given_structures();
    let plain = registry
        .create("gfnff", &Default::default(), &start)
        .unwrap();
    let (a, b) = (
        oniom.evaluate(&start, &EvalRequest::ENERGY_GRAD).unwrap(),
        plain.evaluate(&start, &EvalRequest::ENERGY_GRAD).unwrap(),
    );
    assert!((a.energy.value() - b.energy.value()).abs() < 1e-10);
    assert!(oniom.description().contains("ONIOM with 3 levels"));
    // The other commands read the same options.
    let common = ["-os", "gfn0xtb", "-oniom", "2-4,8,9", "auto", "gfnff"];
    let mut with = vec![file.as_str()];
    with.extend_from_slice(&common);
    assert!(ircmain::config_from_cli(&with).is_ok());
    assert!(nebmain::config_from_cli(&with).is_ok());
    // Usage errors.
    assert!(optmain::config_from_cli(&[file.as_str(), "-oniom", "2-4", "auto"]).is_err());
    assert!(optmain::config_from_cli(&[
        file.as_str(),
        "-oniom",
        "2-4",
        "auto",
        "gfnff",
        "-oniom_opt",
        "high_charge=1"
    ])
    .is_err());
    assert!(
        optmain::config_from_cli(&[file.as_str(), "-oniom_middle", "1-4", "auto", "gfnff"])
            .is_err()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn high_layer_trajectory_keeps_the_real_model_atoms_in_input_order() {
    for micro in ["on", "off"] {
        let dir = scratch(&format!("high_trajectory_{micro}"));
        let file = input(&dir);
        let micro_option = format!("micro={micro}");
        let args = [
            file.as_str(),
            "-os",
            "gfn0xtb",
            "-oniom",
            "2-4,8,9",
            "auto",
            "gfnff",
            "-oniom_opt",
            &micro_option,
            "traj_high_only=on",
            "-ns",
            "2",
        ];
        let config = optmain::config_from_cli(&args).unwrap();
        let runs = optmain::run(&config, &dir, None).unwrap();
        let log = std::fs::read_to_string(runs[0].directory.join("mopt.log")).unwrap();
        let first = log.split("# ITR. 0").nth(1).unwrap();
        let count = first
            .lines()
            .skip(1)
            .take_while(|line| !line.trim().is_empty())
            .count();
        assert_eq!(count, 5);
        let summary = runs[0].summary.as_ref().unwrap();
        let frames = mopt_rs::io::xyz::read_xyz_trajectory_file(
            &summary.trajectory,
            SpinCharge::default(),
            None,
        )
        .unwrap()
        .frames;
        assert!(!frames.is_empty());
        for frame in frames {
            let symbols: Vec<_> = frame.system.elements.iter().map(|e| e.symbol()).collect();
            assert_eq!(symbols, ["C", "O", "H", "H", "H"]);
        }
        assert_eq!(
            read_xyz_file(&runs[0].optimized, SpinCharge::default(), None)
                .unwrap()
                .n_atoms(),
            9
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn afir_on_the_core_and_a_chain_go_with_the_microiterations() {
    let dir = scratch("afir_chain");
    let file = input(&dir);
    let registry = CalculatorRegistry::native();
    let oniom = ["-os", "gfn0xtb", "-oniom", "2-4,8,9", "auto", "gfnff"];
    // AFIR between O 3 and H 8, both in the model region: the bias goes on the core.
    let mut afir = vec![file.as_str()];
    afir.extend_from_slice(&oniom);
    afir.extend(["-ma", "150", "3", "8", "-ns", "12"]);
    let config = optmain::config_from_cli(&afir).unwrap();
    let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    assert!(!log.contains("without microiterations"), "{log}");
    let summary = out[0].summary.as_ref().unwrap();
    let path = mopt_rs::io::xyz::read_xyz_trajectory_file(
        &summary.trajectory,
        SpinCharge::default(),
        None,
    )
    .unwrap();
    assert!(path.frames.len() > 1);
    assert!(path.frames.iter().all(|f| f.system.n_atoms() == 9));
    // On an atom of the environment (H 5 of the methyl group): the atom is moved with the
    // core, and the run keeps its microiterations.
    let mut outside = vec![file.as_str()];
    outside.extend_from_slice(&oniom);
    outside.extend(["-ma", "150", "3", "5", "-ns", "2"]);
    let config = optmain::config_from_cli(&outside).unwrap();
    let out_outside = optmain::run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(out_outside[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("moved by the optimizer with the core (5)"),
        "{log}"
    );
    assert!(!log.contains("without microiterations"), "{log}");

    // A chain from the start to the end of the AFIR path: a chain of the core on the
    // relaxed surface, its files with the whole system.
    let ends = dir.join("ends.xyz");
    let first = std::fs::read_to_string(&file).unwrap();
    let last = std::fs::read_to_string(&out[0].optimized).unwrap();
    std::fs::write(
        &ends,
        format!("{}\n{}\n", first.trim_end(), last.trim_end()),
    )
    .unwrap();
    let ends = ends.to_string_lossy().into_owned();
    let mut chain = vec![ends.as_str()];
    chain.extend_from_slice(&oniom);
    chain.extend(["-ns", "3", "-p", "3", "-nd", "0"]);
    let config = nebmain::config_from_cli(&chain).unwrap();
    let run = nebmain::run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap();
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    assert!(log.contains("with microiterations"), "{log}");
    let candidates = run.candidates.as_ref().unwrap();
    let frames =
        mopt_rs::io::xyz::read_xyz_trajectory_file(&candidates.path, SpinCharge::default(), None)
            .unwrap();
    assert_eq!(frames.frames.len(), run.energies.len());
    assert!(frames.frames.iter().all(|f| f.system.n_atoms() == 9));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_held_environment_keeps_the_neb_core_in_the_enzyme_frame() {
    let dir = scratch("fixed_environment_neb");
    let path = dir.join("chain.xyz");
    let image = |offset: f64| {
        let mut xyz = String::from("9\nethanol in the laboratory frame\n");
        for (atom, line) in ETHANOL.lines().skip(2).enumerate() {
            let fields: Vec<_> = line.split_whitespace().collect();
            let x = fields[1].parse::<f64>().unwrap() + 30.0;
            let y = fields[2].parse::<f64>().unwrap() + if atom == 3 { offset } else { 0.0 };
            xyz.push_str(&format!("{} {x} {y} {}\n", fields[0], fields[3]));
        }
        xyz
    };
    std::fs::write(&path, [image(0.0), image(0.05), image(0.1)].concat()).unwrap();
    let config = nebmain::config_from_cli(&[
        path.to_str().unwrap(),
        "-os",
        "gfn0xtb",
        "-oniom",
        "2-4,8,9",
        "auto",
        "gfnff",
        "-fix",
        "5",
        "-ns",
        "2",
        "-nd",
        "0",
        "-fe",
        "3",
    ])
    .unwrap();
    let run = nebmain::run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    let at = |iteration| {
        read_xyz_file(
            run.directory
                .join(format!("path_ITR_{iteration}_chain/chain_0.xyz")),
            SpinCharge::default(),
            None,
        )
        .unwrap()
        .coords_flat_bohr()
    };
    let first = at(0);
    assert!(first[0] > 40.0);
    let next = at(1);
    // Atom 1, the outer atom of the link, is in the optimized core.
    for k in 0..3 {
        assert!((next[k] - first[k]).abs() < 1e-8);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn electronic_embedding_goes_with_the_microiterations() {
    let dir = scratch("electronic");
    let file = input(&dir);
    let registry = CalculatorRegistry::native();
    let args = [
        file.as_str(),
        "-os",
        "gfn1xtb",
        "-oniom",
        "2-4,8,9",
        "auto",
        "gfnff",
        "-oniom_opt",
        "embedding=electronic",
        "-tcc",
    ];
    let config = optmain::config_from_cli(&args).unwrap();
    let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(out[0].converged);
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("electronic embedding"), "{log}");
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    assert!(log.contains("relaxed on the whole ONIOM surface"), "{log}");
    // A stationary point of the embedded surface, every atom.
    let optimized = read_xyz_file(&out[0].optimized, SpinCharge::default(), None).unwrap();
    assert_eq!(optimized.n_atoms(), 9);
    let calc = registry
        .create_oniom(&config.calculator, &config.calculator_options, &optimized)
        .unwrap()
        .at_given_structures();
    let at = calc
        .evaluate(&optimized, &EvalRequest::ENERGY_GRAD)
        .unwrap();
    let largest = at.gradient_bohr.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(largest < 1e-4, "{largest}");
    // Settings of the embedding without it are refused.
    assert!(optmain::config_from_cli(&[
        file.as_str(),
        "-os",
        "gfn1xtb",
        "-oniom",
        "2-4,8,9",
        "auto",
        "gfnff",
        "-oniom_opt",
        "boundary=zero",
    ])
    .and_then(|c| optmain::run_with(&c, &dir, None, &registry))
    .is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_first_hessian_and_an_excited_state_go_with_the_microiterations() {
    let dir = scratch("ih_es");
    let file = input(&dir);
    let registry = CalculatorRegistry::native();
    let oniom = ["-os", "gfn0xtb", "-oniom", "2-4,8,9", "auto", "gfnff"];
    // An optimization that leaves the Hessian of its core, and one that starts from it.
    let mut first = vec![file.as_str()];
    first.extend_from_slice(&oniom);
    first.extend(["-fc", "1000"]);
    let mut config = optmain::config_from_cli(&first).unwrap();
    let kept = dir.join("core_hessian.npy");
    config.final_hessian = Some(kept.clone());
    let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(out[0].converged);
    let core = mopt_rs::io::npy::read_npy_matrix(&kept).unwrap();
    // The core: the model region (5 atoms) and the outer atom of its link.
    assert_eq!(core.nrows(), 18);
    let optimized = out[0].optimized.to_string_lossy().into_owned();
    let mut again = vec![optimized.as_str()];
    again.extend_from_slice(&oniom);
    again.extend(["-ih", kept.to_str().unwrap()]);
    let config = optmain::config_from_cli(&again).unwrap();
    let out_again = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(out_again[0].converged);
    let log = std::fs::read_to_string(out_again[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    assert!(!log.contains("without microiterations"), "{log}");
    // A Hessian of the whole system is reduced to the core.
    let whole = registry
        .create_oniom(
            &config.calculator,
            &config.calculator_options,
            &read_xyz_file(&out[0].optimized, SpinCharge::default(), None).unwrap(),
        )
        .unwrap()
        .at_given_structures();
    let at = whole
        .evaluate(
            &read_xyz_file(&out[0].optimized, SpinCharge::default(), None).unwrap(),
            &EvalRequest::ENERGY_GRAD_HESS,
        )
        .unwrap();
    let whole_file = dir.join("whole_hessian.npy");
    mopt_rs::io::npy::write_npy_matrix(&whole_file, &at.hessian_bohr.unwrap()).unwrap();
    let mut from_whole = vec![optimized.as_str()];
    from_whole.extend_from_slice(&oniom);
    from_whole.extend(["-ih", whole_file.to_str().unwrap()]);
    let config = optmain::config_from_cli(&from_whole).unwrap();
    let out_whole = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(out_whole[0].converged);
    // An excited state of the model system (the TDA of GFN0-xTB) on the relaxed surface.
    let mut excited = vec![file.as_str()];
    excited.extend_from_slice(&oniom);
    excited.extend(["-es", "1", "-ns", "3"]);
    let config = optmain::config_from_cli(&excited).unwrap();
    let out_es = optmain::run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(out_es[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    assert!(!log.contains("without microiterations"), "{log}");
    // A variable cell cannot go with them: refused, not run without them.
    let mut cell = vec![file.as_str()];
    cell.extend_from_slice(&oniom);
    cell.push("-vc");
    if let Ok(config) = optmain::config_from_cli(&cell) {
        assert!(optmain::run_with(&config, &dir, None, &registry).is_err());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_other_commands_have_the_microiterations_too() {
    let dir = scratch("others");
    let file = input(&dir);
    let registry = CalculatorRegistry::native();
    let oniom = ["-os", "gfn0xtb", "-oniom", "2-4,8,9", "auto", "gfnff"];
    // freqmain relaxes the environment of the structure before it analyses it.
    let mut freq = vec![file.as_str()];
    freq.extend_from_slice(&oniom);
    let config = freqmain::config_from_cli(&freq).unwrap();
    let analysed = freqmain::run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(analysed[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("the environment (3 atoms) relaxed to a minimum"),
        "{log}"
    );
    // ieipmain: a pair of structures of the core on the relaxed surface; the estimate of
    // the transition state is written with the whole system.
    let other = dir.join("rotated.xyz");
    // Ethanol with the hydroxyl hydrogen turned to the other side.
    let turned = ETHANOL.replace("H 2.990 1.330 0.000", "H 1.620 1.900 0.700");
    std::fs::write(&other, turned).unwrap();
    let other = other.to_string_lossy().into_owned();
    let mut pair = vec![file.as_str(), other.as_str()];
    pair.extend_from_slice(&oniom);
    pair.extend(["-ns", "3"]);
    let config = mopt_rs::cli::ieipmain::config_from_cli(&pair).unwrap();
    let run = mopt_rs::cli::ieipmain::run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap();
    assert!(log.contains("ONIOM microiterations:"), "{log}");
    for guess in &run.transition_states {
        let s = read_xyz_file(guess, SpinCharge::default(), None).unwrap();
        assert_eq!(s.n_atoms(), 9);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn each_level_has_its_own_calculator_settings() {
    let dir = scratch("levels");
    let file = input(&dir);
    let registry = CalculatorRegistry::native();
    let base = [
        file.as_str(),
        "-os",
        "gfn0xtb",
        "-oniom",
        "2-4,8,9",
        "auto",
        "gfnff",
        "-ns",
        "2",
    ];
    let run = |extra: &[&str]| {
        let mut args = base.to_vec();
        args.extend_from_slice(extra);
        optmain::config_from_cli(&args)
            .and_then(|config| optmain::run_with(&config, &dir, None, &registry))
    };
    // A setting of the low level reaches its calculator (GFN-FF) and no other.
    assert!(run(&["-oniom_opt", "low.hb_switch=off"]).is_ok());
    let e = run(&["-oniom_opt", "low.no_such_setting=1"])
        .err()
        .unwrap()
        .to_string();
    assert!(
        e.contains("gfnff: unknown setting 'no_such_setting'"),
        "{e}"
    );
    // A setting of the high level is not handed down: GFN-FF, which would refuse a key it
    // does not know (`hessian` is one of GFN0-xTB), does not see it; the high level refuses
    // a key it does not know itself.
    assert!(run(&["-calc_opt", "hessian=analytic"]).is_ok());
    let e = run(&["-calc_opt", "no_such_setting=1"])
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("gfn0xtb: unknown setting"), "{e}");
    // Layers that are not there.
    for (setting, what) in [
        ("middle.hb_switch=off", "no middle layer"),
        (
            "high.hb_switch=off",
            "the highest level takes its settings from -calc_opt",
        ),
        ("bottom.hb_switch=off", "unknown layer 'bottom'"),
    ] {
        let e = run(&["-oniom_opt", setting]).err().unwrap().to_string();
        assert!(e.contains(what), "{setting}: {e}");
    }
    // The options of a lower level: its method and basis by name, the rest as its settings,
    // the threads and the memory of the run.
    let of_the_run = mopt_rs::pes::registry::CalculatorOptions {
        method: Some("b3lyp".into()),
        basis: Some("def2-svp".into()),
        threads: Some(4),
        memory: Some("2GB".into()),
        extra: vec![("grid".into(), "fine".into())],
        ..Default::default()
    };
    let own = of_the_run.for_lower_level(&[
        ("method".into(), "GFN2-xTB".into()),
        ("accuracy".into(), "0.1".into()),
    ]);
    assert_eq!(own.method.as_deref(), Some("GFN2-xTB"));
    assert_eq!(own.basis, None);
    assert_eq!(own.extra, vec![("accuracy".to_string(), "0.1".to_string())]);
    assert_eq!((own.threads, own.memory.as_deref()), (Some(4), Some("2GB")));
}
