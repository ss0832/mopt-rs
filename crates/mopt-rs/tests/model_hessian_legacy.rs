//! MultiOptPy's model Hessians (`legacy_` names) against MultiOptPy 1.21.2 itself
//! (`tests/golden/model_hessians_legacy.json`, from `tools/golden/model_hessians.py`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;

use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Element, MolecularSystem, SpinCharge};
use mopt_rs::model_hessian::legacy::LEGACY_MODELS;
use mopt_rs::model_hessian::{resolve, ModelSpec};

/// Models whose MultiOptPy version solves `(B B^T) g_int = B g` over all atom pairs: the system
/// is singular from five atoms on, so MultiOptPy's numbers depend on rounding there and only
/// smaller molecules are compared exactly.
const PAIR_GRADIENT_MODELS: &[&str] = &["lindh"];

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn legacy_models_reproduce_multioptpy() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/model_hessians_legacy.json"
    );
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut checked = 0;
    let mut failures = Vec::new();
    for case in doc.get("cases").unwrap().as_array().unwrap() {
        let model = case.get("model").unwrap().as_str().unwrap();
        let spec = ModelSpec::parse(model).unwrap();
        if !LEGACY_MODELS.contains(&spec.base) {
            continue;
        }
        // MultiOptPy itself fails on some inputs (Lindh's singular pair-gradient system for
        // linear HCN); there is nothing to compare.
        if case.get("error").is_some() {
            continue;
        }
        let symbols = case.get("symbols").unwrap().as_string_vec().unwrap();
        let elements: Vec<Element> = symbols
            .iter()
            .map(|s| Element::parse_token(s).unwrap())
            .collect();
        let x = floats(case.get("positions_bohr").unwrap());
        let g = floats(case.get("gradient").unwrap());
        let reference = floats(case.get("hessian").unwrap());
        let sys = MolecularSystem::from_flat_bohr(
            elements,
            &Col::from_fn(x.len(), |i| x[i]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let result = resolve(&format!("legacy_{model}"))
            .unwrap()
            .hessian(&sys, &Col::from_fn(g.len(), |i| g[i]));
        if PAIR_GRADIENT_MODELS.contains(&spec.base) && symbols.len() > 4 {
            if let Ok(h) = result {
                let finite = (0..h.nrows()).all(|i| (0..h.ncols()).all(|j| h[(i, j)].is_finite()));
                assert!(finite, "legacy_{model}");
            }
            continue;
        }
        let h = result.unwrap();
        let n = x.len();
        let scale = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
        let diff = (0..n * n)
            .map(|k| (h[(k / n, k % n)] - reference[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let molecule = case.get("molecule").unwrap().as_str().unwrap();
        // The `ts` modifier reflects eigenvectors; for linear molecules the zero eigenvalues
        // are degenerate and the eigenvectors differ between LAPACK and faer at ~1e-8.
        let linear = molecule == "hcn";
        let tol = if spec.ts && linear { 1e-7 } else { 1e-10 };
        if diff > tol * scale.max(1.0) {
            failures.push(format!(
                "legacy_{model} on {molecule}: |dH| = {diff:.3e}, |H| = {scale:.3e}"
            ));
        }
        checked += 1;
    }
    assert!(checked > 0);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
