//! MultiOptPy's units for the comparisons with its reference data.
//!
//! mopt-rs converts with CODATA 2018 (`Units`); MultiOptPy with its own Bohr radius
//! (0.52917721067 Angstrom) and Hartree in kcal/mol (627.509). Its golden files therefore live in
//! a unit system whose Bohr and Hartree differ from mopt-rs' by 4.4e-10 and 7.6e-7. The
//! Lennard-Jones surface below is mopt-rs' own, evaluated in MultiOptPy's units (coordinates
//! and energies converted with the `MOP_*` constants), so that the optimizers are compared with
//! MultiOptPy like for like.
//!
//! MultiOptPy: `Parameters/unit_values.py:UnitValueLib` (no counterpart for the helper itself).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(dead_code)]

use faer::Col;
use mopt_rs::domain::{EvalOutput, EvalRequest, Hartree, MolecularSystem, Quantity, Units};
use mopt_rs::error::MoptResult;
use mopt_rs::parameters::multioptpy_tables::{MOP_BOHR2ANGSTROMS, MOP_HARTREE2KCALMOL};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::Calculator;

/// mopt-rs Bohr per MultiOptPy Bohr.
pub fn length_scale() -> f64 {
    MOP_BOHR2ANGSTROMS / Units::BOHR_TO_ANGSTROM
}

/// MultiOptPy Hartree per mopt-rs Hartree.
pub fn energy_scale() -> f64 {
    Units::HARTREE_TO_KCAL_MOL / MOP_HARTREE2KCALMOL
}

/// Angstrom to MultiOptPy's Bohr.
pub fn mop_bohr(angstrom: f64) -> f64 {
    angstrom / MOP_BOHR2ANGSTROMS
}

/// mopt-rs' Lennard-Jones surface at `sys`, whose coordinates are in MultiOptPy's Bohr, with
/// energy, gradient and Hessian in MultiOptPy's Hartree and Bohr.
pub fn lennard_jones(sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
    let (s, k) = (length_scale(), energy_scale());
    let x = sys.coords_flat_bohr();
    let ours = sys.with_flat_coords_bohr(&Col::from_fn(x.nrows(), |i| x[i] * s))?;
    let mut out = LennardJones.evaluate(&ours, req)?;
    out.energy = Quantity::<Hartree>::new(out.energy.value() * k);
    out.gradient_bohr = Col::from_fn(out.gradient_bohr.nrows(), |i| out.gradient_bohr[i] * k * s);
    if let Some(h) = out.hessian_bohr.as_mut() {
        let f = k * s * s;
        for j in 0..h.ncols() {
            for i in 0..h.nrows() {
                h[(i, j)] *= f;
            }
        }
    }
    Ok(out)
}
