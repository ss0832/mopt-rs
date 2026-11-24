//! The cost of the analytic Hessian of GFN-FF against the Hessian of central differences of
//! its analytic gradient: molecules of `tests/golden/gfnff.json`, cubic grids of water
//! molecules, and cells of `tests/golden/gfnff_pbc_reference.json`.
//!
//! ```text
//! cargo run --release -p mopt-rs --example bench_gfnff_hessian [-- MAX_NUMERICAL_ATOMS]
//! ```
//!
//! Each structure prints the time of one gradient, of the analytic Hessian (the fastest of
//! three), and of the Hessian of differences (`6N` gradients: measured up to
//! `MAX_NUMERICAL_ATOMS` atoms, 400 by default, and above that the time of one gradient times
//! `6N`), with the largest difference between the two Hessians. The cells also print the
//! time of the analytic Hessian in the positions and the lattice vectors against the analytic
//! Hessian of the positions with the nine lattice columns from central differences (18
//! gradients with the stress).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::time::Instant;

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Bohr, Element, MolecularSystem, PeriodicCell, SpinCharge, Vector3};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::native::Gfnff;

fn floats(v: &JsonValue) -> MoptResult<Vec<f64>> {
    v.as_array()
        .ok_or_else(|| MoptError::config("expected an array"))?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| MoptError::config("expected a number"))
        })
        .collect()
}

fn field<'a>(case: &'a JsonValue, key: &str) -> MoptResult<&'a JsonValue> {
    case.get(key)
        .ok_or_else(|| MoptError::config(format!("missing '{key}'")))
}

fn cases(name: &str) -> MoptResult<Vec<JsonValue>> {
    let path = format!("{}/../../tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    let text =
        std::fs::read_to_string(&path).map_err(|e| MoptError::config(format!("{path}: {e}")))?;
    let doc = JsonValue::parse(&text)?;
    Ok(field(&doc, "cases")?
        .as_array()
        .ok_or_else(|| MoptError::config("cases"))?
        .to_vec())
}

fn system(case: &JsonValue, periodic: bool) -> MoptResult<MolecularSystem> {
    let el: Vec<Element> = field(case, "symbols")?
        .as_string_vec()
        .ok_or_else(|| MoptError::config("symbols"))?
        .iter()
        .map(|s| Element::parse_token(s))
        .collect::<MoptResult<_>>()?;
    let x = floats(field(case, "positions_bohr")?)?;
    let charge = field(case, "charge")?
        .as_f64()
        .ok_or_else(|| MoptError::config("charge"))? as i32;
    let electrons = el.iter().map(|e| e.atomic_number() as i32).sum::<i32>() - charge;
    let cell = if periodic {
        let rows: Vec<Vec<f64>> = field(case, "lattice_bohr")?
            .as_array()
            .ok_or_else(|| MoptError::config("lattice_bohr"))?
            .iter()
            .map(floats)
            .collect::<MoptResult<_>>()?;
        let v = |k: usize| Vector3::<Bohr>::new([rows[k][0], rows[k][1], rows[k][2]]);
        Some(PeriodicCell::from_lattice_vectors(
            v(0),
            v(1),
            v(2),
            [true; 3],
        )?)
    } else {
        None
    };
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge {
            charge,
            multiplicity: if electrons % 2 == 0 { 1 } else { 2 },
        },
        cell,
    )
}

/// `m^3` water molecules on a cubic grid of 5.8 Bohr, each turned a little differently.
fn water_grid(m: usize) -> MoptResult<MolecularSystem> {
    let spacing = 5.8;
    let (r, half) = (1.81, 0.912_f64);
    let mut el = Vec::new();
    let mut x = Vec::new();
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                let o = [a, b, c].map(|v| v as f64 * spacing);
                let turn = 0.7 * (a + 2 * b + 3 * c) as f64;
                let (s, co) = turn.sin_cos();
                let h1 = [r * half.sin() * co, r * half.sin() * s, r * half.cos()];
                let h2 = [-r * half.sin() * co, -r * half.sin() * s, r * half.cos()];
                el.push(Element::parse_token("O")?);
                x.extend_from_slice(&o);
                for h in [h1, h2] {
                    el.push(Element::parse_token("H")?);
                    x.extend((0..3).map(|k| o[k] + h[k]));
                }
            }
        }
    }
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge {
            charge: 0,
            multiplicity: 1,
        },
        None,
    )
}

fn seconds<T>(f: impl FnOnce() -> MoptResult<T>) -> MoptResult<(f64, T)> {
    let start = Instant::now();
    let value = f()?;
    Ok((start.elapsed().as_secs_f64(), value))
}

fn measure(name: &str, sys: &MolecularSystem, max_numerical: usize) -> MoptResult<()> {
    let n = sys.n_atoms();
    let analytic = Gfnff::starting_from(sys);
    let mut numerical = Gfnff::starting_from(sys);
    numerical.analytic_hessian = false;
    let topo = analytic.topology(sys)?;
    let (t_gradient, _) = seconds(|| topo.compute(sys))?;
    let mut t_analytic = f64::INFINITY;
    let mut ha = None;
    for _ in 0..3 {
        let (t, h) = seconds(|| analytic.hessian(sys))?;
        t_analytic = t_analytic.min(t);
        ha = Some(h);
    }
    let ha = ha.ok_or_else(|| MoptError::calc("no Hessian"))?;
    let (t_numerical, deviation, measured) = if n <= max_numerical {
        let (t, hn) = seconds(|| numerical.hessian(sys))?;
        let dim = 3 * n;
        let worst = (0..dim)
            .flat_map(|i| (0..dim).map(move |j| (i, j)))
            .map(|(i, j)| (ha[(i, j)] - hn[(i, j)]).abs())
            .fold(0.0, f64::max);
        (t, format!("{worst:.1e}"), "")
    } else {
        (
            t_gradient * 6.0 * n as f64,
            "-".to_string(),
            " (6N x gradient)",
        )
    };
    println!(
        "| {name} | {n} | {:.2} | {:.3} | {:.3}{measured} | {:.0} | {deviation} |",
        t_gradient * 1e3,
        t_analytic,
        t_numerical,
        t_numerical / t_analytic
    );
    Ok(())
}

/// The Hessian in the positions and the lattice vectors of a cell, analytic and with the
/// lattice columns from central differences.
fn measure_cell(name: &str, sys: &MolecularSystem) -> MoptResult<()> {
    let calc = Gfnff::starting_from(sys);
    let mut t_cell = f64::INFINITY;
    for _ in 0..3 {
        t_cell = t_cell.min(seconds(|| calc.cell_hessian(sys))?.0);
    }
    let (t_atoms, _) = seconds(|| calc.hessian(sys))?;
    let topo = calc.topology(sys)?;
    let (t_stress, _) = seconds(|| topo.compute(sys))?;
    let mixed = t_atoms + 18.0 * t_stress;
    println!(
        "| {name} | {} | {t_cell:.3} | {mixed:.3} | {:.1} |",
        sys.n_atoms(),
        mixed / t_cell
    );
    Ok(())
}

fn main() -> MoptResult<()> {
    let max_numerical = std::env::args()
        .nth(1)
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|e| MoptError::config(format!("MAX_NUMERICAL_ATOMS: {e}")))?
        .unwrap_or(400);
    println!(
        "| structure | atoms | gradient (ms) | analytic Hessian (s) | differences (s) | ratio | \
         max difference (Hartree/Bohr^2) |"
    );
    println!("|---|---|---|---|---|---|---|");
    let molecules = cases("gfnff.json")?;
    for wanted in ["caffeine", "rivaroxaban", "grubbs", "taxol"] {
        if let Some(case) = molecules
            .iter()
            .find(|c| c.get("molecule").and_then(JsonValue::as_str) == Some(wanted))
        {
            measure(wanted, &system(case, false)?, max_numerical)?;
        }
    }
    for m in [4, 6, 8] {
        measure(
            &format!("{} waters", m * m * m),
            &water_grid(m)?,
            max_numerical,
        )?;
    }
    let cells = cases("gfnff_pbc_reference.json")?;
    for wanted in [
        "water_ice_ic",
        "benzene_crystal",
        "diamond_2x2x2",
        "benzene_crystal_2x1x1",
    ] {
        if let Some(case) = cells
            .iter()
            .find(|c| c.get("name").and_then(JsonValue::as_str) == Some(wanted))
        {
            measure(wanted, &system(case, true)?, max_numerical)?;
        }
    }
    println!();
    println!("| cell | atoms | analytic with lattice (s) | positions + 18 gradients (s) | ratio |");
    println!("|---|---|---|---|---|");
    for wanted in [
        "water_ice_ic",
        "benzene_crystal",
        "diamond_2x2x2",
        "benzene_crystal_2x1x1",
    ] {
        if let Some(case) = cells
            .iter()
            .find(|c| c.get("name").and_then(JsonValue::as_str) == Some(wanted))
        {
            measure_cell(wanted, &system(case, true)?)?;
        }
    }
    Ok(())
}
