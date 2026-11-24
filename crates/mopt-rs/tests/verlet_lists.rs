//! The Verlet lists the EAM, MEAM and Tersoff calculators keep between evaluations give the
//! bits of a fresh neighbour search: along a velocity-Verlet trajectory of copper (atoms leave
//! the cell through its faces and are wrapped) and along a relaxation of its cell (the cell
//! changes at every step), a calculator that keeps its lists gives the same energy, gradient
//! and stress as a new calculator at every structure.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;
use mopt_rs::domain::{
    Bohr, Element, EvalOutput, EvalRequest, MolecularSystem, PeriodicCell, SpinCharge, Vector3,
};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::calculator::Calculator;
use mopt_rs::pes::native::{Eam, Meam, Tersoff};

fn bits(out: &EvalOutput) -> Vec<u64> {
    let mut b = vec![out.energy.value().to_bits()];
    b.extend(out.gradient_bohr.iter().map(|v| v.to_bits()));
    if let Some(s) = &out.stress {
        b.extend(s.iter().flatten().map(|v| v.to_bits()));
    }
    b
}

/// An fcc cell of `m^3` conventional cells of lattice constant `a` (Bohr), slightly
/// distorted, of `element`.
fn fcc(element: Element, m: usize, a: f64) -> MoptResult<MolecularSystem> {
    let mut x = Vec::new();
    let mut k = 0usize;
    for i in 0..m {
        for j in 0..m {
            for l in 0..m {
                for b in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    k += 1;
                    let jitter = 0.08 * ((k as f64) * 0.37).sin();
                    x.push((i as f64 + b[0]) * a + jitter);
                    x.push((j as f64 + b[1]) * a - jitter);
                    x.push((l as f64 + b[2]) * a + 0.5 * jitter);
                }
            }
        }
    }
    let l = a * m as f64;
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Bohr>::new([l, 0.0, 0.0]),
        Vector3::<Bohr>::new([0.0, l, 0.0]),
        Vector3::<Bohr>::new([0.0, 0.0, l]),
        [true; 3],
    )?;
    MolecularSystem::from_flat_bohr(
        vec![element; x.len() / 3],
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge::new(0, 1),
        Some(cell),
    )
}

/// `kept` (one calculator for the whole trajectory) against `fresh()` (a new one at every
/// structure) along `steps` steps of velocity Verlet from `start` with hot initial
/// velocities, and then along a relaxation of the cell by its stress.
fn same_along_trajectories<C: Calculator>(
    name: &str,
    start: &MolecularSystem,
    steps: usize,
    fresh: impl Fn() -> MoptResult<C>,
) -> MoptResult<()> {
    let kept = fresh()?;
    let request = EvalRequest::ENERGY_GRAD_STRESS;
    let mut sys = start.clone();
    let n = sys.dof();
    // Velocities of a few hundred kelvin (Bohr per atomic time unit, mass of copper).
    let mass = 63.5 * 1822.888;
    let dt = 80.0;
    let mut v = Col::from_fn(n, |i| 3e-4 * (((i * 7919) % 101) as f64 / 50.0 - 1.0));
    let mut g = kept.evaluate(&sys, &request)?.gradient_bohr;
    for step in 0..steps {
        let mut x = sys.coords_flat_bohr();
        for i in 0..n {
            v[i] -= 0.5 * dt * g[i] / mass;
            x[i] += dt * v[i];
        }
        sys = sys.with_flat_coords_bohr(&x)?;
        let a = kept.evaluate(&sys, &request)?;
        let b = fresh()?.evaluate(&sys, &request)?;
        if bits(&a) != bits(&b) {
            return Err(MoptError::calc(format!(
                "{name}: step {step} of the dynamics differs from a fresh search"
            )));
        }
        g = a.gradient_bohr;
        for i in 0..n {
            v[i] -= 0.5 * dt * g[i] / mass;
        }
    }
    // The cell relaxed by its stress: a scaling of the cell (and the atoms) at every step.
    for step in 0..12 {
        let out = kept.evaluate(&sys, &request)?;
        let fresh_out = fresh()?.evaluate(&sys, &request)?;
        if bits(&out) != bits(&fresh_out) {
            return Err(MoptError::calc(format!(
                "{name}: step {step} of the cell relaxation differs from a fresh search"
            )));
        }
        let stress = out.stress.ok_or_else(|| MoptError::calc("no stress"))?;
        let pressure = (stress[0][0] + stress[1][1] + stress[2][2]) / 3.0;
        let scale = 1.0 + (2e3 * pressure).clamp(-0.01, 0.01);
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::calc("no cell"))?;
        let h = cell.h_bohr;
        let column =
            |k: usize| Vector3::<Bohr>::new([h[0][k] * scale, h[1][k] * scale, h[2][k] * scale]);
        let cell = PeriodicCell::from_lattice_vectors(column(0), column(1), column(2), [true; 3])?;
        let x = sys.coords_flat_bohr();
        sys = MolecularSystem::from_flat_bohr(
            sys.elements.clone(),
            &Col::from_fn(x.nrows(), |i| x[i] * scale),
            sys.spin_charge,
            Some(cell),
        )?;
    }
    Ok(())
}

#[test]
fn eam_lists_are_those_of_a_fresh_search() -> MoptResult<()> {
    let cu = Element::from_atomic_number(29)?;
    let sys = fcc(cu, 2, 6.831)?;
    same_along_trajectories("eam", &sys, 40, || Eam::zhou2004(&sys.elements))
}

#[test]
fn meam_and_tersoff_lists_are_those_of_a_fresh_search() -> MoptResult<()> {
    let cu = Element::from_atomic_number(29)?;
    let sys = fcc(cu, 2, 6.831)?;
    same_along_trajectories("meam", &sys, 15, || Meam::library(&sys.elements))?;
    let si = Element::from_atomic_number(14)?;
    // Silicon on the fcc sites of its diamond lattice (a crude structure; what matters is that
    // the atoms move).
    let sys = fcc(si, 2, 10.26)?;
    same_along_trajectories("tersoff", &sys, 15, || Tersoff::tersoff_1989(&sys.elements))
}
