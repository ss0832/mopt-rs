//! Excited states followed from structure to structure (`-es_follow`) with the built-in
//! GFN0-xTB: along a line through a crossing of the first and the second excited singlet
//! state of ozone the followed state keeps its character where the states counted by
//! their energies change places; an optimization and its restart from the journal.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use faer::Col;
use mopt_rs::cli::optmain;
use mopt_rs::domain::{EvalRequest, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::calculator::Calculator;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

fn scratch(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("mopt_followed_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn optmain_run(base: &Path, args: &[&str]) -> Vec<optmain::OptmainResult> {
    let config = optmain::OptmainConfig::from_args(
        &mopt_rs::cli::option::parser("optmain", mopt_rs::cli::spec::OPTMAIN)
            .parse(args.iter().copied())
            .unwrap(),
    )
    .unwrap();
    optmain::run_with(&config, base, None, &CalculatorRegistry::native()).unwrap()
}

/// Ozone: the minimum of the first excited singlet state and a structure from a few steps
/// of a minimization in the second, which lies behind the crossing of the two.
fn ozone(base: &Path) -> (MolecularSystem, MolecularSystem) {
    let start = base.join("o3.xyz");
    std::fs::write(
        &start,
        "3\n\nO 0.0 0.0 0.0\nO 1.1 0.3 0.0\nO -1.0 0.5 0.1\n",
    )
    .unwrap();
    let s = start.to_string_lossy().into_owned();
    let read = |path: &Path| read_xyz_file(path, SpinCharge::default(), None).unwrap();
    let first = optmain_run(base, &[&s, "-os", "gfn0xtb", "-es", "1", "-ns", "100"]);
    let second = optmain_run(base, &[&s, "-os", "gfn0xtb", "-es", "2", "-ns", "30"]);
    (read(&first[0].optimized), read(&second[0].optimized))
}

fn energy(calc: &dyn Calculator, sys: &MolecularSystem, state: usize) -> f64 {
    let request = EvalRequest {
        excited_state: state,
        ..EvalRequest::ENERGY_ONLY
    };
    calc.evaluate(sys, &request).unwrap().energy.value()
}

/// The structure at `t` on the line from `a` (0) through `b` (1).
fn on_the_line(a: &MolecularSystem, b: &MolecularSystem, t: f64) -> MolecularSystem {
    let (xa, xb) = (a.coords_flat_bohr(), b.coords_flat_bohr());
    let x = Col::from_fn(xa.nrows(), |i| xa[i] + t * (xb[i] - xa[i]));
    a.with_flat_coords_bohr(&x).unwrap()
}

fn following() -> CalculatorOptions {
    CalculatorOptions {
        follow_states: true,
        ..CalculatorOptions::default()
    }
}

/// The largest second difference of the values.
fn kink(values: &[f64]) -> f64 {
    values
        .windows(3)
        .map(|w| (w[0] - 2.0 * w[1] + w[2]).abs())
        .fold(0.0, f64::max)
}

#[test]
fn a_followed_state_keeps_its_character_through_a_crossing() {
    let base = scratch("line");
    let (a, b) = ozone(&base);
    let registry = CalculatorRegistry::native();
    let plain = registry
        .create("gfn0xtb", &CalculatorOptions::default(), &a)
        .unwrap();
    let followed = registry.create("gfn0xtb", &following(), &a).unwrap();
    assert!(followed
        .description()
        .contains("followed between structures"));
    let (mut by_energy, mut kept, mut said) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..=16 {
        let at = on_the_line(&a, &b, 0.1 * k as f64);
        by_energy.push([energy(&*plain, &at, 1), energy(&*plain, &at, 2)]);
        kept.push([energy(&*followed, &at, 1), energy(&*followed, &at, 2)]);
        said.push(followed.take_warnings());
        // The ground state is the ground state with and without.
        assert_eq!(energy(&*followed, &at, 0), energy(&*plain, &at, 0));
    }
    // The two states change places once on the line: before it the followed states are
    // the states by energy, behind it the other way round, and the place is reported.
    let places: Vec<usize> = (0..said.len()).filter(|k| !said[*k].is_empty()).collect();
    assert_eq!(places.len(), 1, "{said:?}");
    let crossing = places[0];
    assert!(crossing > 5 && crossing < 15, "{crossing}");
    assert_eq!(said[crossing].len(), 2, "{said:?}");
    assert!(
        said[crossing][0].contains("electronic state 1")
            && said[crossing][0].contains("is state 2 in the order of the energies here")
    );
    for k in 0..by_energy.len() {
        let expected = if k < crossing {
            by_energy[k]
        } else {
            [by_energy[k][1], by_energy[k][0]]
        };
        assert_eq!(kept[k], expected, "point {k}");
    }
    // The followed states are smooth along the line; the states by energy have a kink.
    let column = |values: &[[f64; 2]], state: usize| -> Vec<f64> {
        values.iter().map(|v| v[state]).collect()
    };
    for state in 0..2 {
        let smooth = kink(&column(&kept, state));
        let ordered = kink(&column(&by_energy, state));
        assert!(smooth < 1.0e-3, "state {}: {smooth}", state + 1);
        assert!(ordered > 3.0e-3, "state {}: {ordered}", state + 1);
    }
}

#[test]
fn an_optimization_stays_with_its_state_and_so_does_its_restart() {
    use mopt_rs::io::xyz::write_xyz_file;
    use mopt_rs::workflow::journal::{read_journal, JOURNAL_FILE};

    let base = scratch("optmain");
    let (a, b) = ozone(&base);
    // Behind the crossing the state that has its minimum at the first structure is the
    // second one by energy.
    let start = base.join("behind.xyz");
    write_xyz_file(&start, &on_the_line(&a, &b, 1.2), None).unwrap();
    let s = start.to_string_lossy().into_owned();
    let lowest = energy(
        &*CalculatorRegistry::native()
            .create("gfn0xtb", &CalculatorOptions::default(), &a)
            .unwrap(),
        &a,
        1,
    );

    // Followed, the minimization goes through the crossing down to that minimum, where
    // the state is the first one by energy.
    let args = [
        s.as_str(),
        "-os",
        "gfn0xtb",
        "-es",
        "2",
        "-es_follow",
        "-ns",
        "150",
    ];
    let followed = optmain_run(&base, &args);
    assert!(followed[0].converged);
    let reached = followed[0].summary.as_ref().unwrap().energy.value();
    assert!((reached - lowest).abs() < 1.0e-6, "{reached} {lowest}");
    let log = std::fs::read_to_string(followed[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("electronic state 2")
            && log.contains("is state 1 in the order of the energies here"),
        "{log}"
    );
    // Counted by its energy at every structure, the state stays the upper one and the run
    // ends elsewhere.
    let counted = optmain_run(
        &base,
        &[s.as_str(), "-os", "gfn0xtb", "-es", "2", "-ns", "150"],
    );
    let elsewhere = counted[0].summary.as_ref().unwrap().energy.value();
    assert!(elsewhere > lowest + 0.01, "{elsewhere} {lowest}");

    // The journal has which state every evaluation was for. A run that is cut behind the
    // crossing and restarted goes on with the same state and ends at the same structure.
    let dir = followed[0].directory.clone();
    let journal = dir.join(JOURNAL_FILE);
    let (records, _) = read_journal(&journal).unwrap();
    let changed = records
        .iter()
        .position(|r| r.excited_state == 2 && r.state_root == Some(0))
        .expect("the followed state becomes the lowest one");
    assert!(records[..changed].iter().all(|r| r.state_root == Some(1)));
    let keep = changed + 1;
    assert!(keep < records.len(), "{keep} of {}", records.len());
    let reference = std::fs::read_to_string(&followed[0].optimized).unwrap();
    let bytes = std::fs::read(&journal).unwrap();
    let mut cut = 8usize;
    for _ in 0..keep {
        let len = u32::from_le_bytes(bytes[cut..cut + 4].try_into().unwrap()) as usize;
        cut += 8 + len;
    }
    std::fs::write(&journal, &bytes[..cut]).unwrap();
    std::fs::remove_file(&followed[0].optimized).unwrap();
    let mut config = optmain::OptmainConfig::from_args(
        &mopt_rs::cli::option::parser("optmain", mopt_rs::cli::spec::OPTMAIN)
            .parse(args.iter().copied())
            .unwrap(),
    )
    .unwrap();
    config.restart_dir = Some(dir.clone());
    let again = optmain::run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(again[0].directory, dir);
    assert!(again[0].converged);
    assert_eq!(
        std::fs::read_to_string(&again[0].optimized).unwrap(),
        reference
    );
}

#[test]
fn every_command_with_a_calculator_takes_the_option() {
    use mopt_rs::cli::calculator::calculator_options;
    use mopt_rs::cli::spec;

    let commands = [
        ("optmain", spec::OPTMAIN),
        ("nebmain", spec::NEBMAIN),
        ("ieipmain", spec::IEIPMAIN),
        ("mdmain", spec::MDMAIN),
        ("confsearch", spec::CONFSEARCH),
        ("relaxedscan", spec::RELAXEDSCAN),
        ("orientsearch", spec::ORIENTSEARCH),
        ("freqmain", spec::FREQMAIN),
        ("ircmain", spec::IRCMAIN),
    ];
    for (name, options) in commands {
        let parser = mopt_rs::cli::option::parser(name, options);
        let asked = parser
            .parse(["input.xyz", "-es_follow", "-stability", "3"])
            .unwrap();
        let read = calculator_options(&asked).unwrap();
        assert!(read.follow_states && read.stability == 3, "{name}");
        let plain = calculator_options(&parser.parse(["input.xyz"]).unwrap()).unwrap();
        assert!(!plain.follow_states, "{name}");
    }
    // A calculator without excited states says once that nothing is followed.
    let registry = CalculatorRegistry::native();
    let argon = read_xyz_text("2\n\nAr 0.0 0.0 0.0\nAr 3.8 0.0 0.0\n");
    let lj = registry.create("lj", &following(), &argon).unwrap();
    lj.evaluate(&argon, &EvalRequest::ENERGY_ONLY).unwrap();
    let said = lj.take_warnings();
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(said[0].contains("-es_follow: lj has no excited states"));
    assert!(lj.take_warnings().is_empty());
    assert!(!lj.evaluates_in_order());
}

fn read_xyz_text(text: &str) -> MolecularSystem {
    let file = scratch("xyz").join("structure.xyz");
    std::fs::write(&file, text).unwrap();
    read_xyz_file(&file, SpinCharge::default(), None).unwrap()
}

#[test]
fn a_chain_of_images_stays_with_the_state_of_its_first_image() {
    use mopt_rs::cli::nebmain;
    use mopt_rs::io::xyz::write_xyz_file;

    let base = scratch("chain");
    let (a, b) = ozone(&base);
    // From the minimum of the first state to a structure behind the crossing, both held.
    let behind = on_the_line(&a, &b, 1.4);
    let ends = base.join("ends.xyz");
    let frames: String = [&a, &behind]
        .iter()
        .map(|sys| {
            let one = base.join("frame.xyz");
            write_xyz_file(&one, sys, None).unwrap();
            std::fs::read_to_string(&one).unwrap()
        })
        .collect();
    std::fs::write(&ends, frames).unwrap();
    let input = ends.to_string_lossy().into_owned();
    let chain = |more: &[&str]| {
        let mut arguments = vec![
            input.as_str(),
            "-os",
            "gfn0xtb",
            "-es",
            "1",
            "-fe",
            "3",
            "-p",
            "5",
            "-nd",
            "0",
            "-adpred",
            "0",
            "-ns",
            "2",
        ];
        arguments.extend_from_slice(more);
        let config = nebmain::config_from_cli(&arguments).unwrap();
        nebmain::run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap()
    };
    let plain = CalculatorRegistry::native()
        .create("gfn0xtb", &CalculatorOptions::default(), &a)
        .unwrap();
    let at_the_start = energy(&*plain, &a, 1);

    // Followed, every image is in the state that is the first one at the first image: the
    // last image, behind the crossing, in the second state by energy there.
    let followed = chain(&["-es_follow"]);
    let rise = followed.energies[followed.energies.len() - 1] - followed.energies[0];
    assert!(followed.energies.len() > 3, "{:?}", followed.energies);
    assert!((followed.energies[0] - at_the_start).abs() < 1.0e-9);
    assert!(
        (rise - (energy(&*plain, &behind, 2) - at_the_start)).abs() < 1.0e-7,
        "{rise}"
    );
    let log = std::fs::read_to_string(followed.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("is state 2 in the order of the energies here"),
        "{log}"
    );
    // Counted by its energy at every image, the last image is in the lower state there.
    let counted = chain(&[]);
    let rise = counted.energies[counted.energies.len() - 1] - counted.energies[0];
    assert!(
        (rise - (energy(&*plain, &behind, 1) - at_the_start)).abs() < 1.0e-7,
        "{rise}"
    );
}
