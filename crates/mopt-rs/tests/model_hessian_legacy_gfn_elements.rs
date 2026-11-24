//! MultiOptPy's GFN-FF-like and GFN0-like model Hessians (`legacy_gfnff`, `legacy_gfn0xtb`)
//! for the elements of MultiOptPy's tables beyond H, C, N, O, F, S, Cl, Br, I, against
//! MultiOptPy 1.21.2 itself (`tests/data/legacy_gfn_heavy.json`, made by
//! `tools/golden/legacy_gfn_heavy.py`): an alkali metal, main-group elements of the third to
//! sixth periods, transition metals of the fourth to sixth periods, a lanthanide, and a noble
//! gas; and the elements outside the tables refused with the range in the message.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;

use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Element, MolecularSystem, SpinCharge};
use mopt_rs::model_hessian::legacy::{gfn_like_has_element, GFN0_LIKE_RANGE, GFNFF_LIKE_RANGE};
use mopt_rs::model_hessian::resolve;

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn system(symbols: &[&str], x: &[f64]) -> MolecularSystem {
    let elements: Vec<Element> = symbols
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect();
    MolecularSystem::from_flat_bohr(
        elements,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

#[test]
fn heavier_elements_reproduce_multioptpy() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/legacy_gfn_heavy.json"
    );
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    let mut checked = 0;
    for case in doc.get("cases").unwrap().as_array().unwrap() {
        let model = case.get("model").unwrap().as_str().unwrap();
        let molecule = case.get("molecule").unwrap().as_str().unwrap();
        let symbols = case.get("symbols").unwrap().as_string_vec().unwrap();
        let sym: Vec<&str> = symbols.iter().map(String::as_str).collect();
        let x = floats(case.get("positions_bohr").unwrap());
        let g = floats(case.get("gradient").unwrap());
        let reference = floats(case.get("hessian").unwrap());
        let sys = system(&sym, &x);
        let h = resolve(&format!("legacy_{model}"))
            .unwrap()
            .hessian(&sys, &Col::from_fn(g.len(), |i| g[i]))
            .unwrap();
        let n = x.len();
        let scale = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
        let diff = (0..n * n)
            .map(|k| (h[(k / n, k % n)] - reference[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        eprintln!("legacy_{model} {molecule}: |dH| = {diff:.3e}, |H| = {scale:.3e}");
        if diff > 1e-10 * scale.max(1.0) {
            failures.push(format!(
                "legacy_{model} on {molecule}: |dH| = {diff:.3e}, |H| = {scale:.3e}"
            ));
        }
        checked += 1;
    }
    assert_eq!(checked, 12);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_ranges_are_those_of_multioptpys_tables() {
    for z in 1..=118u8 {
        assert_eq!(gfn_like_has_element("gfnff", z), z <= 86, "gfnff Z = {z}");
        assert_eq!(
            gfn_like_has_element("gfn0xtb", z),
            z <= 54,
            "gfn0xtb Z = {z}"
        );
    }
    assert!(!gfn_like_has_element("gfnff", 0));
    let g = Col::zeros(6);
    let fr = system(&["Fr", "F"], &[0.0, 0.0, 0.0, 0.0, 0.0, 4.0]);
    let err = resolve("legacy_gfnff")
        .unwrap()
        .hessian(&fr, &g)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(GFNFF_LIKE_RANGE) && err.contains("Fr (Z = 87)"),
        "{err}"
    );
    let cs = system(&["Cs", "F"], &[0.0, 0.0, 0.0, 0.0, 0.0, 4.0]);
    let err = resolve("legacy_gfn0xtb")
        .unwrap()
        .hessian(&cs, &g)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(GFN0_LIKE_RANGE) && err.contains("Cs (Z = 55)") && !err.contains("F (Z"),
        "{err}"
    );
}
