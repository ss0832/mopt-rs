//! How much the converged GFN1-xTB energy and gradient of a structure depend on where its SCC
//! starts: from the superposition of atomic densities, or from the converged charges of
//! another structure (as `SccGuess::Previous` does after a structure evaluated by another
//! thread).
//!
//! ```text
//! cargo run --release -p mopt-rs --example bench_scc_start
//! ```
//!
//! For caffeine and taxol of `tests/golden/gfnff.json`, the structure is evaluated from the
//! superposition and then from the charges of structures displaced by 0.02, 0.05 and 0.1 Bohr
//! (every coordinate, a fixed pattern), with the default thresholds of the SCC and with
//! tighter ones; printed are the largest differences of the energy (Hartree) and of the
//! gradient (Hartree/Bohr) from the start of the superposition, and the SCC iterations.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Element, MolecularSystem, SpinCharge};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::native::gfn1::SccGuess;
use mopt_rs::pes::native::Gfn1Xtb;

fn molecule(name: &str) -> MoptResult<MolecularSystem> {
    let path = format!(
        "{}/../../tests/golden/gfnff.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text =
        std::fs::read_to_string(&path).map_err(|e| MoptError::config(format!("{path}: {e}")))?;
    let doc = JsonValue::parse(&text)?;
    let case = doc
        .get("cases")
        .and_then(JsonValue::as_array)
        .and_then(|c| {
            c.iter()
                .find(|c| c.get("molecule").and_then(JsonValue::as_str) == Some(name))
        })
        .ok_or_else(|| MoptError::config(format!("no case {name}")))?;
    let el: Vec<Element> = case
        .get("symbols")
        .and_then(JsonValue::as_string_vec)
        .ok_or_else(|| MoptError::config("symbols"))?
        .iter()
        .map(|s| Element::parse_token(s))
        .collect::<MoptResult<_>>()?;
    let x: Vec<f64> = case
        .get("positions_bohr")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| MoptError::config("positions"))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| MoptError::config("number")))
        .collect::<MoptResult<_>>()?;
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge::new(0, 1),
        None,
    )
}

fn calculator(guess: SccGuess, tight: bool) -> Gfn1Xtb {
    let mut c = Gfn1Xtb::default();
    c.guess = guess;
    if tight {
        c.charge_tolerance = 1e-12;
    }
    c
}

fn main() -> MoptResult<()> {
    println!(
        "| molecule | thresholds | start | max dE (Hartree) | max dg (Hartree/Bohr) | \
         iterations |"
    );
    println!("|---|---|---|---|---|---|");
    for name in ["caffeine", "taxol"] {
        let sys = molecule(name)?;
        let x0 = sys.coords_flat_bohr();
        for tight in [false, true] {
            let label = if tight { "charges 1e-12" } else { "default" };
            let sad = calculator(SccGuess::Sad, tight).compute(&sys)?;
            println!(
                "| {name} | {label} | superposition | 0 | 0 | {} |",
                sad.iterations
            );
            for shift in [0.02, 0.05, 0.1] {
                let calc = calculator(SccGuess::Previous, tight);
                let moved =
                    Col::from_fn(x0.nrows(), |k| x0[k] + shift * ((k as f64 * 0.731).sin()));
                calc.compute(&sys.with_flat_coords_bohr(&moved)?)?;
                let r = calc.compute(&sys)?;
                let de = (r.energy.total - sad.energy.total).abs();
                let dg = (0..x0.nrows())
                    .map(|k| (r.gradient[k] - sad.gradient[k]).abs())
                    .fold(0.0, f64::max);
                println!(
                    "| {name} | {label} | charges at {shift} Bohr | {de:.1e} | {dg:.1e} | {} |",
                    r.iterations
                );
            }
        }
    }
    Ok(())
}
