//! `etmain` on a molecule: the radical anion of 1,4-pentadiene, a donor (one vinyl group), a
//! bridge (the CH2), and an acceptor (the other vinyl group), with the built-in GFN1-xTB. The
//! molecule is symmetric, so the final state at its equilibrium has the energy of the initial
//! one, and the barrier of a self-exchange is a quarter of the reorganization energy. And
//! `-et_method hopping` on the ethylene dimer of HAB11: a hole between two equal molecules.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use mopt_rs::cli::etmain::{config_from_cli, run_with};
use mopt_rs::domain::Units;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::electron_transfer::apparent_barrier;

fn scratch(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("mopt_etmain_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// 1,4-Pentadiene (Angstrom), optimized with GFN1-xTB as the neutral molecule: C1=C2-C3-C4=C5,
/// then the hydrogens of C1, C2, C3, C4, C5.
const PENTADIENE: &str = "13
1,4-pentadiene radical anion
C  -2.475780   0.086490   0.007191
C  -1.260558  -0.437376  -0.005142
C   0.000000   0.388339   0.000000
C   1.260558  -0.437376   0.005142
C   2.475780   0.086490  -0.007191
H  -2.646139   1.155744   0.020778
H  -3.363258  -0.531520   0.002753
H  -1.130623  -1.516274  -0.018504
H  -0.005094   1.048767   0.876439
H   0.005094   1.048767  -0.876439
H   1.130623  -1.516274   0.018504
H   2.646139   1.155744  -0.020778
H   3.363258  -0.531520  -0.002753
";

fn input(base: &std::path::Path) -> String {
    let path = base.join("pentadiene.xyz");
    std::fs::write(&path, PENTADIENE).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn a_bridged_self_exchange_in_the_gas_phase() {
    let base = scratch("gas");
    let file = input(&base);
    let config = config_from_cli(&[
        &file,
        "-os",
        "gfn1xtb",
        "-elec",
        "-1",
        "-spin",
        "2",
        "-et_donor",
        "1,2,6-8",
        "-et_acceptor",
        "4,5,11-13",
    ])
    .unwrap();
    let registry = CalculatorRegistry::native();
    let out = run_with(&config, &base, None, &registry).unwrap();
    assert_eq!(out.len(), 1);
    let r = out[0].transfer.as_ref().expect("an EDEEL run");
    // Symmetric: the two states at their equilibria have the same energy.
    let de0 = r.final_state.1 - r.initial.1;
    assert!(de0.abs() < 2e-4, "dE0 = {de0} Hartree");
    // One crossing without a solvent: the inner sphere of a self-exchange.
    assert_eq!(r.crossings.len(), 1);
    let c = &r.crossings[0];
    assert!(c.converged);
    // Converged: the gap force 2 (V11 - V22) is below the force threshold (3e-4).
    assert!((c.v11 - c.v22).abs() < 1.5e-4, "{} {}", c.v11, c.v22);
    assert!(c.barrier > 1e-3 && c.barrier < 0.1, "barrier {}", c.barrier);
    let inner = c.inner.expect("inner sphere");
    assert!(
        (inner.reorganization / (4.0 * c.barrier) - 1.0).abs() < 0.2,
        "lambda {} against 4 dE {}",
        inner.reorganization,
        4.0 * c.barrier
    );
    let v = c.coupling.expect("coupling");
    assert!(v > 0.0 && v < 0.1, "coupling {v}");
    let marcus = c.marcus.expect("Marcus rate");
    let zn = c.zhu_nakamura.expect("Zhu-Nakamura rate");
    assert!(marcus.is_finite() && zn.rate.is_finite() && marcus > 0.0 && zn.rate > 0.0);
    // The free energy of activation is near the electronic one.
    let g = c.free_energy_barrier.expect("free energy barrier");
    assert!(
        (g - c.barrier).abs() < 0.02,
        "dG {g} against dE {}",
        c.barrier
    );
    assert!(c.marcus_free.is_some() && c.zhu_nakamura_free.is_some());
    // The apparent barrier of a rate is the Eyring inverse.
    let back = apparent_barrier(marcus, 298.15);
    let eyring = Units::BOLTZMANN_J_PER_K * 298.15 / Units::PLANCK_J_S
        * (-back / (Units::BOLTZMANN_HARTREE_PER_K * 298.15)).exp();
    assert!((eyring / marcus - 1.0).abs() < 1e-10);
    // The files.
    let dir = &out[0].directory;
    for f in [
        "et_scan.csv",
        "et_summary.json",
        "eq_initial.xyz",
        "eq_final.xyz",
        "crossings.xyz",
    ] {
        assert!(dir.join(f).is_file(), "{f}");
    }
    let summary = std::fs::read_to_string(dir.join("et_summary.json")).unwrap();
    assert!(summary.contains("apparent_barrier_kcal_mol"), "{summary}");
    assert!(summary.contains("initial_free_energy_hartree"));
    let csv = std::fs::read_to_string(dir.join("et_scan.csv")).unwrap();
    assert!(csv.lines().next().unwrap().contains("free_barrier_eV"));
    assert_eq!(csv.lines().count(), 2);
}

#[test]
fn the_solvent_scan_shares_the_activation() {
    let base = scratch("solvent");
    let file = input(&base);
    let config = config_from_cli(&[
        &file,
        "-os",
        "gfn1xtb",
        "-elec",
        "-1",
        "-spin",
        "2",
        "-et_donor",
        "1,2,6-8",
        "-et_acceptor",
        "4,5,11-13",
        "-et_lambda_s",
        "0.6",
        "0.6",
        "-et_alpha",
        "1,0,0.5",
        "-et_coupling",
        "edeel",
        "-et_free_energy",
        "off",
    ])
    .unwrap();
    let registry = CalculatorRegistry::native();
    let out = run_with(&config, &base, None, &registry).unwrap();
    let r = out[0].transfer.as_ref().expect("an EDEEL run");
    let alphas: Vec<f64> = r.crossings.iter().map(|c| c.alpha.unwrap()).collect();
    assert_eq!(alphas, vec![1.0, 0.5, 0.0]);
    // At alpha_SA = 1 the solvent carries the whole activation: the crossing is the initial
    // equilibrium and the barrier is the solvent shift there, above the lambda_s / 4 of a
    // solvent alone (the vertical gap of the inner sphere has to be closed as well).
    let first = &r.crossings[0];
    let lambda_s = 0.6 * Units::EV_TO_HARTREE;
    assert!(first.inner_barrier.abs() < 1e-4, "{}", first.inner_barrier);
    assert!(
        (first.barrier - first.shifts[0]).abs() < 1e-4,
        "{}",
        first.barrier
    );
    assert!(first.barrier > lambda_s / 4.0);
    assert!(first.inner.is_none() && first.free_energy_barrier.is_none());
    // Every crossing is one of the shifted states (to the force threshold, as above).
    for c in &r.crossings {
        let gap = (c.v11 + c.shifts[0]) - (c.v22 + c.shifts[1]);
        assert!(c.converged, "alpha {:?}", c.alpha);
        assert!(gap.abs() < 1.5e-4, "alpha {:?}: gap {gap}", c.alpha);
    }
    // Less solvent activation, more of the inner sphere.
    for w in r.crossings.windows(2) {
        assert!(w[1].inner_barrier > w[0].inner_barrier - 1e-6);
        assert!(w[1].shifts[0] < w[0].shifts[0] + 1e-12);
    }
    assert!(r.crossings.iter().all(|c| c.marcus.is_some()));
    assert!(r.best_marcus.is_some());
}

/// The ethylene dimer of HAB11 at 4.0 Angstrom as an xyz file, with the atoms (1-based) of its
/// two molecules.
fn ethylene_dimer(base: &std::path::Path) -> (String, [String; 2]) {
    use mopt_rs::core_infra::json::JsonValue;
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/hab11.json");
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let dimer = doc
        .get("dimers")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .find(|d| {
            d.get("name").unwrap().as_str() == Some("ethylene")
                && d.get("distance_angstrom").unwrap().as_f64() == Some(4.0)
        })
        .unwrap();
    let symbols = dimer.get("symbols").unwrap().as_string_vec().unwrap();
    let mut xyz = format!("{}\nethylene dimer, HAB11, 4.0 Angstrom\n", symbols.len());
    for (s, p) in symbols
        .iter()
        .zip(dimer.get("positions_angstrom").unwrap().as_array().unwrap())
    {
        let r: Vec<f64> = p
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        xyz.push_str(&format!("{s} {:.8} {:.8} {:.8}\n", r[0], r[1], r[2]));
    }
    let file = base.join("ethylene_dimer.xyz");
    std::fs::write(&file, xyz).unwrap();
    let molecule = |k: usize| -> String {
        dimer.get("molecules").unwrap().as_array().unwrap()[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| (x.as_f64().unwrap() as usize + 1).to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    (
        file.to_string_lossy().into_owned(),
        [molecule(0), molecule(1)],
    )
}

#[test]
fn a_hole_hops_between_two_ethylenes() {
    let base = scratch("hopping");
    let (file, [a, b]) = ethylene_dimer(&base);
    let config = config_from_cli(&[
        &file,
        "-os",
        "gfn1xtb",
        "-et_method",
        "hopping",
        "-et_carrier",
        "hole",
        "-et_donor",
        &a,
        "-et_acceptor",
        &b,
    ])
    .unwrap();
    let registry = CalculatorRegistry::native();
    let out = run_with(&config, &base, None, &registry).unwrap();
    assert!(out[0].transfer.is_none());
    let h = out[0].hopping.as_ref().expect("a hop");
    // The coupling of tests/hab11_couplings.rs at this structure (GFN1-xTB, 35.2 meV).
    let j = h.coupling.effective.abs() * Units::HARTREE_TO_EV * 1000.0;
    assert!((j - 35.2).abs() < 0.1, "J_eff {j} meV");
    // Two equal molecules: the hop is a self-exchange, the same both ways.
    assert!(h.molecules.iter().all(|m| m.converged == [true, true]));
    assert!(h.reaction_energy.abs() < 1e-5, "dE {}", h.reaction_energy);
    let lambda = h.mean_reorganization();
    assert!(
        lambda > 0.05 * Units::EV_TO_HARTREE && lambda < 2.0 * Units::EV_TO_HARTREE,
        "lambda {} eV",
        lambda * Units::HARTREE_TO_EV
    );
    assert!((h.reorganization[0] - h.reorganization[1]).abs() < 1e-5);
    assert!(
        (h.rates[0] / h.rates[1] - 1.0).abs() < 1e-2,
        "{:?}",
        h.rates
    );
    // The Marcus rate of a self-exchange: the barrier is a quarter of lambda.
    let kt = Units::BOLTZMANN_HARTREE_PER_K * 298.15;
    let direct = mopt_rs::kinetics::rate::inner_sphere::marcus_with_tunnelling(
        0.0,
        h.coupling.effective,
        lambda,
        (lambda + h.reaction_energy).powi(2) / (4.0 * lambda),
        kt,
    )
    .unwrap();
    assert!((h.rates[0] / direct - 1.0).abs() < 1e-12);
    assert!((apparent_barrier(h.rates[0], 298.15) - h.apparent_barriers[0]).abs() < 1e-12);
    let dir = &out[0].directory;
    for name in [
        "hop_summary.json",
        "molecule_a_without.xyz",
        "molecule_a_with.xyz",
        "molecule_b_without.xyz",
        "molecule_b_with.xyz",
        "mopt.log",
    ] {
        assert!(dir.join(name).is_file(), "{name}");
    }
    let summary = std::fs::read_to_string(dir.join("hop_summary.json")).unwrap();
    let parsed = mopt_rs::core_infra::json::JsonValue::parse(&summary).unwrap();
    assert_eq!(parsed.get("carrier").unwrap().as_str(), Some("hole"));
    // Four multiplicities, not five, and a method by name.
    let five = ["-et_mult", "1", "2", "1", "2", "1"];
    let mut args = vec![file.as_str(), "-et_method", "hopping", "-et_donor", &a];
    args.extend(["-et_acceptor", &b]);
    args.extend(five);
    assert!(config_from_cli(&args).is_err());
    assert!(config_from_cli(&[
        &file,
        "-et_method",
        "tunnel",
        "-et_donor",
        &a,
        "-et_acceptor",
        &b
    ])
    .is_err());
}
