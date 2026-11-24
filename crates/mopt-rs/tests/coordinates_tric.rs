//! Optimizations in translation-rotation internal coordinates (`-coord tric`, `tric-p`): a
//! water dimer with the built-in GFN-FF, compared with Cartesian coordinates, and a cluster of
//! single atoms.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::coord::system::{CoordinateRegistry, CoordinateSystem, Tric};
use mopt_rs::domain::{positions, Angstrom, Element, EvalRequest, MolecularSystem, SpinCharge};
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::pes::native::Gfnff;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::run_optimization;

fn system(rows: &[(&str, [f64; 3])]) -> MolecularSystem {
    MolecularSystem::new(
        rows.iter()
            .map(|r| Element::parse_token(r.0).unwrap())
            .collect(),
        &positions::<Angstrom>(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

/// Two waters far from their hydrogen-bonded arrangement.
fn water_dimer() -> MolecularSystem {
    system(&[
        ("O", [0.00, 0.00, 0.12]),
        ("H", [0.00, 0.76, -0.47]),
        ("H", [0.00, -0.76, -0.47]),
        ("O", [3.40, 0.50, 0.90]),
        ("H", [3.10, 1.30, 1.35]),
        ("H", [4.20, 0.30, 1.40]),
    ])
}

fn optimize(start: &MolecularSystem, coordinates: &str) -> (Option<Termination>, usize, f64) {
    let calc = Gfnff::starting_from(start);
    let settings = OptimizerSettings {
        methods: vec!["rsirfo_bfgs".to_string()],
        coordinates: coordinates.to_string(),
        max_iterations: 500,
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, start.clone()).unwrap();
    let out = run_optimization(&mut opt, &calc, &mut []).unwrap();
    let e = calc
        .evaluate(opt.current(), &EvalRequest::ENERGY_GRAD)
        .unwrap()
        .energy
        .value();
    (out.termination, out.iterations, e)
}

#[test]
fn registry_knows_both_forms() {
    let reg = CoordinateRegistry::default();
    for (name, canonical) in [
        ("tric", "tric"),
        ("TRIC", "tric"),
        ("tric-p", "tric-p"),
        ("tricp", "tric-p"),
        ("tric_p", "tric-p"),
    ] {
        assert_eq!(reg.create(name).unwrap().name(), canonical);
    }
    let dimer = water_dimer();
    let mut tric = Tric::delocalized();
    tric.initialize(&dimer).unwrap();
    assert_eq!(tric.fragments().len(), 2);
    assert_eq!(tric.dim(&dimer), 18);
    // Three internal coordinates per water; no bond between the molecules.
    assert_eq!(tric.primitives().len(), 6);
}

/// From two waters far from their hydrogen-bonded arrangement, the same minimum as in Cartesian
/// coordinates in about half the iterations (60 against 106 to 119 when the start is scaled by
/// 1 +- 1e-10 or 1e-8): the molecules approach and turn along the translation and rotation
/// coordinates. (Delocalized internal coordinates join the molecules by a bond and end in
/// another, higher minimum from this start. They join them O...O, and when the hydrogen bond
/// becomes linear the angle H-O...O folds to zero; before such sets were generated anew there
/// (`coord::internal::FOLDED_ANGLE_RAD`) the run stalled with steps of 1e-13 Bohr for 2000
/// iterations or converged in 68 to 133, depending on the last digits of the start; now 68 to
/// 120.)
#[test]
fn water_dimer_reaches_the_cartesian_minimum_in_fewer_iterations() {
    let start = water_dimer();
    let (t_cart, n_cart, e_cart) = optimize(&start, "cart");
    assert_eq!(t_cart, Some(Termination::Converged));
    for coordinates in ["tric", "tric-p"] {
        let (t, n, e) = optimize(&start, coordinates);
        assert_eq!(t, Some(Termination::Converged), "{coordinates}");
        assert!((e - e_cart).abs() < 1e-4, "{coordinates}: {e} vs {e_cart}");
        assert!(e <= e_cart + 1e-6, "{coordinates}: {e} vs {e_cart}");
        assert!(
            3 * n < 2 * n_cart,
            "{coordinates}: {n} iterations, Cartesian {n_cart}"
        );
    }
    let (t_dic, _, _) = optimize(&start, "dic");
    assert_eq!(t_dic, Some(Termination::Converged));
}

#[test]
fn command_line_with_single_atoms_and_molecules() {
    let dir = std::env::temp_dir().join(format!("mopt_tric_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Seven argon atoms: every fragment is one atom, so only translations exist.
    let ar7 = dir.join("ar7.xyz");
    std::fs::write(
        &ar7,
        "7\n\nAr 1.95 0.0 0.0\nAr 0.6026 1.8546 0.03\nAr -1.5776 1.1462 0.07\n\
         Ar -1.5776 -1.1462 0.11\nAr 0.6026 -1.8546 0.15\nAr 0.0 0.0 2.1\nAr 0.2 0.0 -2.0\n",
    )
    .unwrap();
    let dimer = dir.join("dimer.xyz");
    std::fs::write(
        &dimer,
        "6\n\nO 0.0 0.0 0.12\nH 0.0 0.76 -0.47\nH 0.0 -0.76 -0.47\n\
         O 3.4 0.5 0.9\nH 3.1 1.3 1.35\nH 4.2 0.3 1.4\n",
    )
    .unwrap();
    for (input, calculator) in [(&ar7, "lj"), (&dimer, "gfnff")] {
        for coordinates in ["tric", "tric-p"] {
            let input = input.to_string_lossy();
            let cfg = config_from_cli(&[
                &input,
                "-os",
                calculator,
                "-opt",
                "rsirfo_bfgs",
                "-coord",
                coordinates,
                "-ns",
                "500",
            ])
            .unwrap();
            let res = run(&cfg, &dir, None).unwrap();
            assert!(res[0].converged, "{calculator} {coordinates}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
