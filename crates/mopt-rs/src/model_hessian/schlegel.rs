//! Schlegel's model Hessian (H. B. Schlegel, Theor. Chim. Acta 1984, 66, 333) with the stretch
//! parameters of J. M. Wittbrodt and H. B. Schlegel, J. Mol. Struct. (Theochem) 1997, 398-399,
//! 55, on the bonded internal coordinates:
//!
//! - stretch `1.734 / (r - B_ij)^3` with `B` by period pair;
//! - bend `0.160` when a terminal atom is hydrogen, otherwise `0.250`;
//! - torsion `0.0023 - 0.07 (r_bc - r_cov,bc)` about the central bond (not below zero);
//! - wag (out-of-plane at atoms with three bonds) `0.045`.
//!
//! MultiOptPy: `ModelHessian/schlegel.py`, `schlegeld3.py`, `schlegeld4.py` (published model;
//!  MultiOptPy's in `legacy::schlegel*`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::assemble::HessianAccumulator;
use super::{add_dispersion, Dispersion, ModelHessian};
use crate::coord::connectivity::BondConnectivity;
use crate::domain::{Element, MolecularSystem};
use crate::error::MoptResult;
use crate::parameters::covalent_radius_bohr;
use crate::parameters::model_hessian::{period, SCHLEGEL_B};

/// Bond detection factor on the sum of covalent radii.
pub(crate) const BOND_FACTOR: f64 = 1.3;

/// Schlegel's model.
#[derive(Debug, Clone, Copy)]
pub struct Schlegel {
    /// Dispersion term.
    pub dispersion: Dispersion,
}

impl Schlegel {
    /// Model with `dispersion`.
    pub fn new(dispersion: Dispersion) -> Self {
        Self { dispersion }
    }
}

fn b_param(a: Element, b: Element) -> f64 {
    let (i, j) = (period(a).min(6) - 1, period(b).min(6) - 1);
    SCHLEGEL_B[i][j]
}

impl ModelHessian for Schlegel {
    fn name(&self) -> &str {
        match self.dispersion {
            Dispersion::D3 => "schlegeld3",
            Dispersion::D4 => "schlegeld4",
            _ => "schlegel",
        }
    }

    fn hessian(&self, sys: &MolecularSystem, _gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        let conn = BondConnectivity::from_system(sys, BOND_FACTOR);
        let el = |a| sys.element(a);
        let mut acc = HessianAccumulator::new(sys);
        for &[a, b] in &conn.bonds {
            let r = acc.r2(a, b).sqrt();
            let d = r - b_param(el(a), el(b));
            if d > 1e-3 {
                acc.stretch(a, b, 1.734 / (d * d * d));
            }
        }
        for &[a, m, c] in &conn.angles {
            let hydrogen = el(a).atomic_number() == 1 || el(c).atomic_number() == 1;
            acc.bend(a, m, c, if hydrogen { 0.160 } else { 0.250 });
        }
        for &t in &conn.dihedrals {
            let (b, c) = (t[1], t[2]);
            let r = acc.r2(b, c).sqrt();
            let k = 0.0023 - 0.07 * (r - covalent_radius_bohr(el(b)) - covalent_radius_bohr(el(c)));
            acc.torsion(t, k.max(0.0));
        }
        for &[center, a, b, c] in &conn.out_of_planes {
            acc.out_of_plane(center, a, b, c, 0.045);
        }
        let mut h = acc.finish();
        add_dispersion(&mut h, sys, self.dispersion)?;
        Ok(h)
    }
}
