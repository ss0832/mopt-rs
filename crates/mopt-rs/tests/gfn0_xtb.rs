//! GFN0-xTB of mopt-rs against gfn0-rs and xtb (`tests/golden/gfn0_xtb.json`): energies and
//! gradients for closed shells, a cation, a doublet, and a triplet with every element from H
//! to Ar and Br and I from gfn0-rs 0.4.1 (`tools/golden/gfn0_xtb.py`), and 26 structures with
//! every element from K to Kr and Ru (oxides, halides, carbonyls, aqua and ammine complexes of
//! several charges and spins, ferrocene, a Ti-Ru hydride, Cu2Cl2) from xtb 6.7.1
//! (`tools/golden/gfn0_xtb_binary.py`, the structures of `gfn1_xtb.json`), and 56 structures
//! with every element from Rb to Rn other than Ru and I (hydrides, halides, oxides, carbonyls,
//! ammine complexes, organometallics, noble gases, all fifteen lanthanides; closed and open
//! shells, cations and anions) from the same xtb. The records of
//! gfn0-rs are compared to 1e-10; xtb and gfn0-rs differ by up to 6e-11 Hartree in the energy
//! and 9e-10 Hartree/Bohr in the gradient on the light elements, so the records of xtb are
//! compared to 1e-9 and 2e-9.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Element, MolecularSystem, SpinCharge};
use mopt_rs::pes::native::Gfn0Xtb;

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn gfn0_xtb_reproduces_gfn0_rs() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/gfn0_xtb.json"
    );
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let calc = Gfn0Xtb::default();
    let mut failures = Vec::new();
    let mut worst = (0.0_f64, 0.0_f64);
    for case in doc.get("cases").unwrap().as_array().unwrap() {
        let name = case.get("molecule").unwrap().as_str().unwrap();
        let symbols = case.get("symbols").unwrap().as_string_vec().unwrap();
        let elements: Vec<Element> = symbols
            .iter()
            .map(|s| Element::parse_token(s).unwrap())
            .collect();
        let x = floats(case.get("positions_bohr").unwrap());
        let spin = SpinCharge {
            charge: case.get("charge").unwrap().as_f64().unwrap() as i32,
            multiplicity: case.get("multiplicity").unwrap().as_f64().unwrap() as u32,
        };
        let sys =
            MolecularSystem::from_flat_bohr(elements, &Col::from_fn(x.len(), |i| x[i]), spin, None)
                .unwrap();
        let r = calc.compute(&sys).unwrap();
        let e_ref = case.get("energy").unwrap().as_f64().unwrap();
        let g_ref = floats(case.get("gradient").unwrap());
        let de = (r.energy.total - e_ref).abs();
        let (e_tol, g_tol) = if case.get("source").is_some() {
            (1.0e-9, 2.0e-9)
        } else {
            (1.0e-10, 1.0e-10)
        };
        let dg = (0..g_ref.len())
            .map(|k| (r.gradient[k] - g_ref[k]).abs())
            .fold(0.0, f64::max);
        worst = (worst.0.max(de), worst.1.max(dg));
        println!("{name:20} |dE| {de:.2e} |dg| {dg:.2e}");
        if de > e_tol || dg > g_tol {
            failures.push(format!(
                "{name}: |dE| = {de:.3e} (E = {:.10}, reference {e_ref:.10}), max |dg| = {dg:.3e} \
                 ({:?})",
                r.energy.total, r.energy
            ));
        }
    }
    println!(
        "worst |dE| {:.2e} Hartree, |dg| {:.2e} Hartree/Bohr",
        worst.0, worst.1
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn optmain_with_gfn0_xtb_and_its_model_hessian() {
    use mopt_rs::cli::optmain::{config_from_cli, run};
    use mopt_rs::model_hessian::resolve;

    let dir = std::env::temp_dir().join(format!("mopt_gfn0_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let xyz = dir.join("water.xyz");
    std::fs::write(
        &xyz,
        "3\n\nO 0.0 0.0 0.15\nH 0.0 0.80 -0.45\nH 0.0 -0.70 -0.50\n",
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[
        &xyz,
        "-os",
        "gfn0xtb",
        "-opt",
        "rsirfo_bfgs",
        "-modelhess",
        "gfn0xtb",
    ])
    .unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    assert!(res[0].converged);
    let log = std::fs::read_to_string(res[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("Calculator: gfn0xtb: GFN0-xTB built into mopt-rs"),
        "{log}"
    );
    let _ = std::fs::remove_dir_all(&dir);

    // The model Hessian is the GFN0-xTB Hessian: symmetric, with three stretch/bend modes.
    let sys = MolecularSystem::from_flat_bohr(
        vec![Element::O, Element::H, Element::H],
        &Col::from_fn(9, |i| {
            [0.0, 0.0, 0.22, 0.0, 1.43, -0.89, 0.0, -1.43, -0.89][i]
        }),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    let h = resolve("gfn0xtb")
        .unwrap()
        .hessian(&sys, &Col::zeros(9))
        .unwrap();
    for i in 0..9 {
        for j in 0..9 {
            assert!((h[(i, j)] - h[(j, i)]).abs() < 1e-8);
        }
    }
    let (vals, _) = mopt_rs::math::linalg::symmetric_eigen(&h).unwrap();
    let large = vals.iter().filter(|v| **v > 1e-2).count();
    assert_eq!(large, 3, "{vals:?}");
}
