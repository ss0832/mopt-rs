//! The cost of the built-in engines at fixed numbers of threads, and whether their results
//! depend on the number of threads.
//!
//! ```text
//! cargo run --release -p mopt-rs --example bench_engines [-- FILTER [REPEATS]]
//! ```
//!
//! Every case (an engine, a structure and what is asked of it) is evaluated inside rayon pools
//! of 1 and 8 threads: one evaluation to warm up, then `REPEATS` (5 by default, fewer for the
//! large cases) timed ones, of which the median is printed in milliseconds. The last column
//! says whether the energy, gradient, Hessian and stress were the same bits in both pools.
//! `FILTER` keeps the cases whose name contains it. The structures: caffeine (24 atoms) and
//! taxol (113 atoms) of `tests/golden/gfnff.json`, a cubic grid of 1331 water molecules (3993
//! atoms), the benzene crystal 2x1x1 of `tests/golden/gfnff_pbc_reference.json`, an fcc copper
//! cluster of 864 atoms, and, when the environment variable `MOPT_BENCH_XYZ` names an `.xyz`
//! file, that structure (GFN-FF only; `MOPT_BENCH_CHARGE` gives its charge).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::time::Instant;

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{
    Bohr, Element, EvalOutput, EvalRequest, MolecularSystem, PeriodicCell, SpinCharge, Vector3,
};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::calculator::Calculator;
use mopt_rs::pes::dispersion::{D3Bj, D4};
use mopt_rs::pes::native::gfn1::SccGuess;
use mopt_rs::pes::native::{Eam, Gfn0Xtb, Gfn1Xtb, Gfnff};
use mopt_rs::pes::{PesTerm, TermContext};

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

fn case_of(file: &str, key: &str, wanted: &str) -> MoptResult<JsonValue> {
    let path = format!("{}/../../tests/golden/{file}", env!("CARGO_MANIFEST_DIR"));
    let text =
        std::fs::read_to_string(&path).map_err(|e| MoptError::config(format!("{path}: {e}")))?;
    let doc = JsonValue::parse(&text)?;
    field(&doc, "cases")?
        .as_array()
        .ok_or_else(|| MoptError::config("cases"))?
        .iter()
        .find(|c| c.get(key).and_then(JsonValue::as_str) == Some(wanted))
        .cloned()
        .ok_or_else(|| MoptError::config(format!("{file}: no case {wanted}")))
}

fn spin(el: &[Element], charge: i32) -> SpinCharge {
    let electrons = el.iter().map(|e| e.atomic_number() as i32).sum::<i32>() - charge;
    SpinCharge {
        charge,
        multiplicity: if electrons % 2 == 0 { 1 } else { 2 },
    }
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
    let sc = spin(&el, charge);
    MolecularSystem::from_flat_bohr(el, &Col::from_fn(x.len(), |i| x[i]), sc, cell)
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
                el.push(Element::O);
                x.extend_from_slice(&o);
                for h in [h1, h2] {
                    el.push(Element::H);
                    x.extend((0..3).map(|k| o[k] + h[k]));
                }
            }
        }
    }
    let sc = spin(&el, 0);
    MolecularSystem::from_flat_bohr(el, &Col::from_fn(x.len(), |i| x[i]), sc, None)
}

/// An fcc copper cluster of `m^3` cubic cells (`4 m^3` atoms), slightly distorted.
fn copper_cluster(m: usize) -> MoptResult<MolecularSystem> {
    let a = 6.831;
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut x = Vec::new();
    let mut k = 0usize;
    for i in 0..m {
        for j in 0..m {
            for l in 0..m {
                for b in basis {
                    k += 1;
                    let jitter = 0.05 * ((k as f64) * 0.37).sin();
                    x.push((i as f64 + b[0]) * a + jitter);
                    x.push((j as f64 + b[1]) * a - jitter);
                    x.push((l as f64 + b[2]) * a + 0.5 * jitter);
                }
            }
        }
    }
    let el = vec![Element::from_atomic_number(29)?; x.len() / 3];
    let sc = SpinCharge {
        charge: 0,
        multiplicity: 1,
    };
    MolecularSystem::from_flat_bohr(el, &Col::from_fn(x.len(), |i| x[i]), sc, None)
}

/// What a case evaluates.
enum Engine {
    Calc(Box<dyn Calculator>),
    Term(Box<dyn PesTerm>),
}

struct Case {
    name: String,
    sys: MolecularSystem,
    engine: Engine,
    req: EvalRequest,
    repeats: usize,
}

fn run(case: &Case) -> MoptResult<EvalOutput> {
    match &case.engine {
        Engine::Calc(c) => c.evaluate(&case.sys, &case.req),
        Engine::Term(t) => t.evaluate(
            &case.sys,
            &TermContext::at_iteration(0, &case.sys),
            case.req.need_hessian,
        ),
    }
}

/// The bits of everything an evaluation returned.
fn fingerprint(out: &EvalOutput) -> Vec<u64> {
    let mut bits = vec![out.energy.value().to_bits()];
    bits.extend(out.gradient_bohr.iter().map(|v| v.to_bits()));
    if let Some(h) = &out.hessian_bohr {
        for j in 0..h.ncols() {
            bits.extend((0..h.nrows()).map(|i| h[(i, j)].to_bits()));
        }
    }
    if let Some(s) = &out.stress {
        bits.extend(s.iter().flatten().map(|v| v.to_bits()));
    }
    bits
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        f64::NAN
    } else {
        v[v.len() / 2]
    }
}

fn measure(case: &Case, threads: usize) -> MoptResult<(f64, Vec<u64>)> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| MoptError::config(format!("thread pool: {e}")))?;
    pool.install(|| {
        let first = run(case)?;
        let mut times = Vec::new();
        for _ in 0..case.repeats {
            let start = Instant::now();
            let out = run(case)?;
            times.push(start.elapsed().as_secs_f64() * 1e3);
            if fingerprint(&out) != fingerprint(&first) {
                return Err(MoptError::calc(format!(
                    "{}: a repeated evaluation gave other bits",
                    case.name
                )));
            }
        }
        Ok((median(times), fingerprint(&first)))
    })
}

fn gfn1_sad() -> Gfn1Xtb {
    let mut c = Gfn1Xtb::default();
    c.guess = SccGuess::Sad;
    c
}

/// The names of the cases, in the order they run.
const CASES: [&str; 22] = [
    "gfnff gradient caffeine",
    "gfnff gradient taxol",
    "gfnff gradient water 3993",
    "gfnff gradient crystal",
    "gfnff gradient xyz",
    "gfnff hessian caffeine",
    "gfnff hessian taxol",
    "gfnff numerical hessian caffeine",
    "gfn0 gradient caffeine",
    "gfn0 gradient taxol",
    "gfn0 hessian caffeine",
    "gfn0 hessian taxol",
    "gfn0 numerical hessian caffeine",
    "gfn1 gradient caffeine",
    "gfn1 gradient taxol",
    "gfn1 hessian caffeine",
    "gfn1 numerical hessian caffeine",
    "d3 gradient taxol",
    "d3 hessian caffeine",
    "d4 gradient taxol",
    "d4 hessian caffeine",
    "eam gradient cu864",
];

/// The structure a case is named after (`None`: the case cannot run here).
fn structure(name: &str) -> MoptResult<Option<MolecularSystem>> {
    let molecule = |m: &str| system(&case_of("gfnff.json", "molecule", m)?, false);
    Ok(Some(if name.ends_with("caffeine") {
        molecule("caffeine")?
    } else if name.ends_with("taxol") {
        molecule("taxol")?
    } else if name.ends_with("crystal") {
        system(
            &case_of("gfnff_pbc_reference.json", "name", "benzene_crystal_2x1x1")?,
            true,
        )?
    } else if name.ends_with("water 3993") {
        water_grid(11)?
    } else if name.ends_with("cu864") {
        copper_cluster(6)?
    } else if name.ends_with("xyz") {
        let Some(path) = std::env::var("MOPT_BENCH_XYZ")
            .ok()
            .filter(|p| !p.is_empty())
        else {
            return Ok(None);
        };
        let charge = std::env::var("MOPT_BENCH_CHARGE")
            .ok()
            .and_then(|c| c.parse::<i32>().ok())
            .unwrap_or(0);
        let mut sys = mopt_rs::io::xyz::read_xyz_file(&path, SpinCharge::new(charge, 1), None)?;
        sys.spin_charge = spin(&sys.elements, charge);
        sys
    } else {
        return Ok(None);
    }))
}

/// The case `name` with `repeats` timed evaluations (fewer for the costly ones).
fn build(name: &str, repeats: usize) -> MoptResult<Option<Case>> {
    let Some(sys) = structure(name)? else {
        return Ok(None);
    };
    let few = repeats.clamp(1, 3);
    let hessian = name.contains("hessian");
    let req = if hessian {
        EvalRequest::ENERGY_GRAD_HESS
    } else if sys.has_pbc() {
        EvalRequest::ENERGY_GRAD_STRESS
    } else {
        EvalRequest::ENERGY_GRAD
    };
    let numerical = name.contains("numerical");
    let small = sys.n_atoms() < 50;
    let repeats = match (hessian, small) {
        (false, true) => repeats * 4,
        (false, false) if sys.n_atoms() < 1000 => repeats,
        (true, true) if !numerical => repeats,
        _ => few,
    };
    let engine = if name.starts_with("gfnff") {
        let mut c = Gfnff::starting_from(&sys);
        c.analytic_hessian = !numerical;
        Engine::Calc(Box::new(c))
    } else if name.starts_with("gfn0") {
        Engine::Calc(Box::new(Gfn0Xtb {
            analytic_hessian: !numerical,
            ..Gfn0Xtb::default()
        }))
    } else if name.starts_with("gfn1") {
        let mut c = gfn1_sad();
        c.analytic_hessian = !numerical;
        Engine::Calc(Box::new(c))
    } else if name.starts_with("d3") {
        Engine::Term(Box::new(D3Bj::PBE))
    } else if name.starts_with("d4") {
        Engine::Term(Box::new(D4::PBE))
    } else if name.starts_with("eam") {
        Engine::Calc(Box::new(Eam::zhou2004(&sys.elements)?))
    } else {
        return Ok(None);
    };
    Ok(Some(Case {
        name: name.to_string(),
        sys,
        engine,
        req,
        repeats,
    }))
}

fn main() -> MoptResult<()> {
    let mut args = std::env::args().skip(1);
    let filter = args.next().unwrap_or_default();
    let repeats = args
        .next()
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|e| MoptError::config(format!("REPEATS: {e}")))?
        .unwrap_or(5);
    println!("| case | atoms | setup (ms) | 1 thread (ms) | 8 threads (ms) | same bits |");
    println!("|---|---|---|---|---|---|");
    for name in CASES {
        if !name.contains(&filter) {
            continue;
        }
        let start = Instant::now();
        let Some(case) = build(name, repeats)? else {
            continue;
        };
        let setup = start.elapsed().as_secs_f64() * 1e3;
        let (t1, f1) = measure(&case, 1)?;
        let (t8, f8) = measure(&case, 8)?;
        println!(
            "| {} | {} | {setup:.0} | {t1:.2} | {t8:.2} | {} |",
            case.name,
            case.sys.n_atoms(),
            if f1 == f8 { "yes" } else { "NO" }
        );
    }
    Ok(())
}
