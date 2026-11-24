//! `freqmain PATH.xyz -diabatize N M ...`: excited states along the frames of a path as
//! diabatic states, with the states of the built-in GFN0-xTB: the first and the second
//! excited singlet state of ozone along a line through their crossing.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use faer::Col;
use mopt_rs::cli::freqmain::{config_from_cli, run_with};
use mopt_rs::cli::optmain;
use mopt_rs::domain::{MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::{format_xyz_system, read_xyz_file};
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("mopt_diabatic_{name}_{}", std::process::id()));
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
/// of a minimization in the second, which lies behind the crossing of the two (the line
/// of `tests/followed_states.rs`).
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

/// The frames at `count` equal steps of `step` on the line from `a` (0) through `b` (1),
/// every atom moved by `off` times a fixed pattern that has no symmetry.
fn line(
    dir: &Path,
    name: &str,
    ends: &(MolecularSystem, MolecularSystem),
    count: usize,
    step: f64,
    off: f64,
) -> String {
    let (xa, xb) = (ends.0.coords_flat_bohr(), ends.1.coords_flat_bohr());
    let pattern = [0.3, -0.5, 0.8, -0.7, 0.2, 0.4, 0.6, 0.9, -0.1];
    let mut text = String::new();
    for k in 0..count {
        let t = step * k as f64;
        let x = Col::from_fn(xa.nrows(), |i| {
            xa[i] + t * (xb[i] - xa[i]) + off * pattern[i]
        });
        text.push_str(&format_xyz_system(
            &ends.0.with_flat_coords_bohr(&x).unwrap(),
            None,
        ));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// The largest second difference of the values.
fn kink(values: &[f64]) -> f64 {
    values
        .windows(3)
        .map(|w| (w[0] - 2.0 * w[1] + w[2]).abs())
        .fold(0.0, f64::max)
}

#[test]
fn two_states_of_ozone_as_diabatic_states() {
    let dir = scratch("ozone");
    let ends = ozone(&dir);
    let input = line(&dir, "line.xyz", &ends, 33, 0.05, 0.0);
    let cfg = config_from_cli(&[&input, "-os", "gfn0xtb", "-diabatize", "1", "2"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let found = out[0].orbitals.as_ref().unwrap();
    let diabatic = found.diabatic.as_ref().unwrap();
    assert!(out[0].analysis.is_none());
    let (set, states) = (&diabatic.set, &diabatic.states);
    assert_eq!(set.energies.len(), 33);
    let lower: Vec<f64> = set.energies.iter().map(|e| e[0]).collect();
    let upper: Vec<f64> = set.energies.iter().map(|e| e[1]).collect();
    let (h11, h22) = (states.element(0, 0), states.element(1, 1));
    for k in 0..33 {
        assert!(lower[k] <= upper[k]);
        // The two diabatic energies are the two adiabatic ones rotated: the same sum.
        let trace = h11[k] + h22[k] - lower[k] - upper[k];
        assert!(trace.abs() < 1e-12, "frame {k}: {trace}");
    }
    // The states change places once on the line.
    assert_eq!(diabatic.crossings.len(), 1, "{:?}", diabatic.crossings);
    // On this line the two states cross (nothing couples them): the coupling is far below
    // the gaps of the frames next to the crossing.
    assert!(
        diabatic.crossings[0].coupling < 1e-4,
        "{:?}",
        diabatic.crossings
    );
    assert!(states.kept.iter().all(|kept| *kept > 0.99));
    assert!(states.weights[0][(0, 0)] == 1.0 && states.weights[32][(0, 1)] > 0.9);
    // The diabatic energies go through the crossing smoothly; the adiabatic ones do not.
    assert!(kink(&h11) < 0.5 * kink(&lower));
    assert!(kink(&h22) < 0.5 * kink(&upper));

    let run_dir = &out[0].directory;
    let read = |name: &str| std::fs::read_to_string(run_dir.join(name)).unwrap();
    let table = read("diabatic_states.csv");
    assert!(table.starts_with("frame,arc_length,state_1,state_2,e_1,e_2,w_1_1,w_1_2,w_2_2,"));
    assert_eq!(table.lines().count(), 34);
    let report = read("diabatic_crossings.txt");
    assert!(
        report.contains("Crossing 1 of the states 1 and 2: at ") && report.contains("Landau-Zener")
    );
    for name in ["diabatic_states.png", "diabatic_states.svg"] {
        assert!(run_dir.join(name).is_file(), "{name}");
    }

    // Off the line, where nothing keeps the two states apart, they avoid each other: the
    // coupling at the crossing is half the gap that is left.
    let input = line(&dir, "beside.xyz", &ends, 33, 0.05, 0.05);
    let cfg = config_from_cli(&[&input, "-os", "gfn0xtb", "-diabatize", "1", "2"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let beside = out[0].orbitals.as_ref().unwrap().diabatic.as_ref().unwrap();
    let gap = beside
        .set
        .energies
        .iter()
        .map(|e| e[1] - e[0])
        .fold(f64::MAX, f64::min);
    assert_eq!(beside.crossings.len(), 1, "{:?}", beside.crossings);
    let crossing = beside.crossings[0];
    assert!(gap > 1e-3, "{gap}");
    assert!(
        (crossing.coupling - 0.5 * gap).abs() < 0.05 * gap,
        "{crossing:?} {gap}"
    );
    // The estimate from the gap alone agrees with the one from the overlaps.
    let fitted = beside.from_gaps[0].unwrap();
    assert!((fitted.coupling - crossing.coupling).abs() < 0.05 * crossing.coupling);
    assert!(
        (fitted.slope_difference - crossing.slope_difference).abs()
            < 0.05 * crossing.slope_difference
    );
    assert!((fitted.position - crossing.position).abs() < 0.05);
    assert!(beside.states.kept.iter().all(|kept| *kept > 0.99));
    // The same crossings in the file for the kinetics: the one of the overlaps, the one of
    // the gap, and the Zhu-Nakamura model where the crossing is sloped.
    let rows = mopt_rs::workflow::diabatic_crossings::read(
        &out[0].directory.join("diabatic_crossings.csv"),
    )
    .unwrap();
    let models = beside.zhu_nakamura.iter().flatten().count()
        + beside.zhu_nakamura_peaked.iter().flatten().count();
    assert_eq!(rows.len(), 2 + models, "{rows:?}");
    assert_eq!(
        (rows[0].kind.as_str(), rows[0].states),
        ("crossing", [1, 2])
    );
    assert!((rows[0].coupling - crossing.coupling).abs() < 1e-12 * crossing.coupling.max(1.0));
    assert!((rows[0].slope_difference() - crossing.slope_difference).abs() < 1e-9);
    assert_eq!(rows[1].kind, "gap");
    assert!((rows[1].slope_product() - fitted.slopes[0] * fitted.slopes[1]).abs() < 1e-12);
    if let Some(model) = beside.zhu_nakamura[0] {
        assert!(rows[2].slope_product() > 0.0 && rows[2].kind == "zhu_nakamura");
        assert!((model.coupling - fitted.coupling).abs() < 0.05 * fitted.coupling);
    } else {
        // Not sloped: the two slopes of the overlaps have opposite signs, and the model of
        // the peaked type agrees with the overlaps where the top of the lower energy and the
        // bottom of the upper one are inside the path.
        assert!(
            crossing.slopes[0] * crossing.slopes[1] < 0.0,
            "{crossing:?}"
        );
        if let Some(model) = beside.zhu_nakamura_peaked[0] {
            assert!(rows[2].slope_product() < 0.0 && rows[2].kind == "zhu_nakamura");
            assert!(
                (model.coupling - crossing.coupling).abs() < 0.1 * crossing.coupling,
                "{model:?} {crossing:?}"
            );
            assert!(
                (model.position - crossing.position).abs() < 0.1,
                "{model:?} {crossing:?}"
            );
            let difference = model.slopes[0] - model.slopes[1];
            assert!(
                (difference - crossing.slope_difference).abs() < 0.2 * crossing.slope_difference,
                "{model:?} {crossing:?}"
            );
        }
    }

    // Three states: the matrix of the two is its corner as long as the third stays away,
    // and the sum of the diabatic energies is that of the adiabatic ones.
    let cfg = config_from_cli(&[&input, "-os", "gfn0xtb", "-diabatize", "1", "2", "3"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let three = out[0].orbitals.as_ref().unwrap().diabatic.as_ref().unwrap();
    assert_eq!(three.states.states(), 3);
    assert_eq!(three.from_gaps.len(), 2);
    for (k, energies) in three.set.energies.iter().enumerate() {
        let w = &three.states.potentials[k];
        let trace = w[(0, 0)] + w[(1, 1)] + w[(2, 2)] - energies.iter().sum::<f64>();
        assert!(trace.abs() < 1e-11, "frame {k}: {trace}");
    }
    let header = std::fs::read_to_string(out[0].directory.join("diabatic_states.csv")).unwrap();
    assert!(header.starts_with(
        "frame,arc_length,state_1,state_2,state_3,e_1,e_2,e_3,w_1_1,w_1_2,w_1_3,w_2_2,w_2_3,w_3_3,"
    ));
}

#[test]
fn what_cannot_be_made_diabatic_is_refused() {
    let dir = scratch("refused");
    let one = dir.join("one.xyz");
    std::fs::write(&one, "3\n\nO 0.0 0.0 0.0\nO 1.2 0.3 0.0\nO -1.1 0.5 0.0\n").unwrap();
    let one = one.to_string_lossy().into_owned();
    // One frame is no path.
    let cfg = config_from_cli(&[&one, "-os", "gfn0xtb", "-diabatize", "1", "2"]).unwrap();
    let error = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(error.to_string().contains("frames of a path"), "{error}");
    // Two different states are needed.
    for wrong in [vec!["1"], vec!["1", "1"], vec!["0", "2"], vec!["1", "x"]] {
        let mut arguments = vec![one.as_str(), "-os", "gfn0xtb", "-diabatize"];
        arguments.extend(wrong.iter().copied());
        let error = config_from_cli(&arguments).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("two or more different excited states"),
            "{error}"
        );
    }
    // A calculator without excited states to hand out.
    let two = dir.join("two.xyz");
    std::fs::write(
        &two,
        "2\n\nAr 0.0 0.0 0.0\nAr 3.8 0.0 0.0\n2\n\nAr 0.0 0.0 0.0\nAr 3.9 0.0 0.0\n",
    )
    .unwrap();
    let two = two.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[&two, "-os", "lj", "-diabatize", "1", "2"]).unwrap();
    let error = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not hand out its excited states"),
        "{error}"
    );
}
