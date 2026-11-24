//! The built-in engines give the same bits for any number of threads: every case is evaluated
//! inside rayon pools of 1, 2, 4 and 8 threads, and the energy, gradient, Hessian and stress
//! are compared bit by bit. The structures are large enough to reach the parallel paths of
//! the engines (their thresholds are in `docs/source/benchmarks.md`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

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

const THREADS: [usize; 4] = [1, 2, 4, 8];

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

fn spin(el: &[Element], charge: i32) -> SpinCharge {
    let electrons = el.iter().map(|e| e.atomic_number() as i32).sum::<i32>() - charge;
    SpinCharge {
        charge,
        multiplicity: if electrons % 2 == 0 { 1 } else { 2 },
    }
}

/// A case of a golden file by the value of `key`.
fn golden(file: &str, key: &str, wanted: &str, periodic: bool) -> MoptResult<MolecularSystem> {
    let path = format!("{}/../../tests/golden/{file}", env!("CARGO_MANIFEST_DIR"));
    let text =
        std::fs::read_to_string(&path).map_err(|e| MoptError::config(format!("{path}: {e}")))?;
    let doc = JsonValue::parse(&text)?;
    let case = field(&doc, "cases")?
        .as_array()
        .ok_or_else(|| MoptError::config("cases"))?
        .iter()
        .find(|c| c.get(key).and_then(JsonValue::as_str) == Some(wanted))
        .cloned()
        .ok_or_else(|| MoptError::config(format!("{file}: no case {wanted}")))?;
    let el: Vec<Element> = field(&case, "symbols")?
        .as_string_vec()
        .ok_or_else(|| MoptError::config("symbols"))?
        .iter()
        .map(|s| Element::parse_token(s))
        .collect::<MoptResult<_>>()?;
    let x = floats(field(&case, "positions_bohr")?)?;
    let charge = field(&case, "charge")?
        .as_f64()
        .ok_or_else(|| MoptError::config("charge"))? as i32;
    let cell = if periodic {
        let rows: Vec<Vec<f64>> = field(&case, "lattice_bohr")?
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
    let (r, half) = (1.81, 0.912_f64);
    let mut el = Vec::new();
    let mut x = Vec::new();
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                let o = [a, b, c].map(|v| v as f64 * 5.8);
                let (s, co) = (0.7 * (a + 2 * b + 3 * c) as f64).sin_cos();
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

/// An fcc copper cluster of `m^3` cubic cells, slightly distorted.
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
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge::new(0, 1),
        None,
    )
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
    if let Some(sh) = &out.strain_hessian {
        bits.extend(format!("{sh:?}").bytes().map(u64::from));
    }
    bits
}

/// Evaluate `f` in pools of 1, 2, 4 and 8 threads and require the same bits.
fn same_bits(name: &str, f: impl Fn() -> MoptResult<EvalOutput> + Sync) -> MoptResult<()> {
    let mut first: Option<Vec<u64>> = None;
    for threads in THREADS {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| MoptError::config(format!("thread pool: {e}")))?;
        let bits = fingerprint(&pool.install(&f)?);
        match &first {
            None => first = Some(bits),
            Some(reference) => {
                if *reference != bits {
                    let differing = reference.iter().zip(&bits).filter(|(a, b)| a != b).count();
                    return Err(MoptError::calc(format!(
                        "{name}: {threads} threads give other bits than 1 thread \
                         ({differing} of {} values differ)",
                        bits.len()
                    )));
                }
            }
        }
    }
    Ok(())
}

/// A fresh calculator for every pool, so that no state carries over between the pools.
fn calc_bits<C: Calculator>(
    name: &str,
    sys: &MolecularSystem,
    req: EvalRequest,
    make: impl Fn() -> MoptResult<C> + Sync,
) -> MoptResult<()> {
    same_bits(name, || make()?.evaluate(sys, &req))
}

fn term_bits(
    name: &str,
    sys: &MolecularSystem,
    hessian: bool,
    term: &(dyn PesTerm + Sync),
) -> MoptResult<()> {
    same_bits(name, || {
        term.evaluate(sys, &TermContext::at_iteration(0, sys), hessian)
    })
}

fn caffeine() -> MoptResult<MolecularSystem> {
    golden("gfnff.json", "molecule", "caffeine", false)
}

#[test]
fn gfnff_is_thread_independent() -> MoptResult<()> {
    let small = caffeine()?;
    let g = EvalRequest::ENERGY_GRAD;
    let h = EvalRequest::ENERGY_GRAD_HESS;
    calc_bits("gfnff gradient caffeine", &small, g, || {
        Ok(Gfnff::starting_from(&small))
    })?;
    calc_bits("gfnff hessian caffeine", &small, h, || {
        Ok(Gfnff::starting_from(&small))
    })?;
    calc_bits("gfnff numerical hessian caffeine", &small, h, || {
        let mut c = Gfnff::starting_from(&small);
        c.analytic_hessian = false;
        Ok(c)
    })?;
    let water = water_grid(6)?;
    calc_bits("gfnff gradient water 648", &water, g, || {
        Ok(Gfnff::starting_from(&water))
    })?;
    let crystal = golden(
        "gfnff_pbc_reference.json",
        "name",
        "benzene_crystal_2x1x1",
        true,
    )?;
    calc_bits(
        "gfnff gradient and stress crystal",
        &crystal,
        EvalRequest::ENERGY_GRAD_STRESS,
        || Ok(Gfnff::starting_from(&crystal)),
    )?;
    let hs = EvalRequest {
        need_stress: true,
        ..h
    };
    calc_bits("gfnff cell hessian crystal", &crystal, hs, || {
        Ok(Gfnff::starting_from(&crystal))
    })
}

#[test]
fn gfn0_is_thread_independent() -> MoptResult<()> {
    let small = caffeine()?;
    calc_bits(
        "gfn0 gradient caffeine",
        &small,
        EvalRequest::ENERGY_GRAD,
        || Ok(Gfn0Xtb::default()),
    )?;
    calc_bits(
        "gfn0 hessian caffeine",
        &small,
        EvalRequest::ENERGY_GRAD_HESS,
        || Ok(Gfn0Xtb::default()),
    )?;
    // Large enough for the products of matrices to be split into tasks.
    let medium = golden("gfnff.json", "molecule", "taxol", false)?;
    calc_bits(
        "gfn0 gradient taxol",
        &medium,
        EvalRequest::ENERGY_GRAD,
        || Ok(Gfn0Xtb::default()),
    )?;
    let tiny = golden("gfnff.json", "molecule", "formamide", false)?;
    calc_bits(
        "gfn0 numerical hessian formamide",
        &tiny,
        EvalRequest::ENERGY_GRAD_HESS,
        || {
            Ok(Gfn0Xtb {
                analytic_hessian: false,
                ..Gfn0Xtb::default()
            })
        },
    )
}

fn gfn1(guess: SccGuess, analytic: bool) -> Gfn1Xtb {
    let mut c = Gfn1Xtb::default();
    c.guess = guess;
    c.analytic_hessian = analytic;
    c
}

#[test]
fn gfn1_is_thread_independent() -> MoptResult<()> {
    let small = caffeine()?;
    for guess in [SccGuess::Sad, SccGuess::Previous] {
        calc_bits(
            &format!("gfn1 gradient caffeine {guess:?}"),
            &small,
            EvalRequest::ENERGY_GRAD,
            || Ok(gfn1(guess, true)),
        )?;
    }
    calc_bits(
        "gfn1 hessian caffeine",
        &small,
        EvalRequest::ENERGY_GRAD_HESS,
        || Ok(gfn1(SccGuess::Previous, true)),
    )?;
    // A d-block metal: the SCC mixes by Broyden's method as well.
    let ferrocene = golden("gfnff.json", "molecule", "elements_Ferrocene", false)?;
    calc_bits(
        "gfn1 gradient ferrocene",
        &ferrocene,
        EvalRequest::ENERGY_GRAD,
        || Ok(gfn1(SccGuess::Sad, true)),
    )?;
    let tiny = golden("gfnff.json", "molecule", "formamide", false)?;
    calc_bits(
        "gfn1 numerical hessian formamide",
        &tiny,
        EvalRequest::ENERGY_GRAD_HESS,
        || Ok(gfn1(SccGuess::Previous, false)),
    )
}

#[test]
fn dispersion_is_thread_independent() -> MoptResult<()> {
    let small = caffeine()?;
    let medium = golden("gfnff.json", "molecule", "taxol", false)?;
    for (name, sys) in [("caffeine", &small), ("taxol", &medium)] {
        term_bits(&format!("d3 gradient {name}"), sys, false, &D3Bj::PBE)?;
        term_bits(&format!("d4 gradient {name}"), sys, false, &D4::PBE)?;
    }
    term_bits("d3 hessian caffeine", &small, true, &D3Bj::PBE)?;
    term_bits("d4 hessian caffeine", &small, true, &D4::PBE)
}

#[test]
fn eam_is_thread_independent() -> MoptResult<()> {
    let copper = copper_cluster(4)?;
    calc_bits(
        "eam gradient cu256",
        &copper,
        EvalRequest::ENERGY_GRAD,
        || Eam::zhou2004(&copper.elements),
    )
}
