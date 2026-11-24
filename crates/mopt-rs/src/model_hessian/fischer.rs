//! Fischer-Almlof model Hessian (T. H. Fischer, J. Almlof, J. Phys. Chem. 1992, 96, 9768), in
//! the form of pysisyphus' implementation, on the bonded internal coordinates:
//!
//! - stretch `0.3601 exp(-1.944 (r - r_cov))`;
//! - bend `0.089 + 0.11 (r_cov,ab r_cov,bc)^0.42 exp(-0.44 (r_ab + r_bc - r_cov,ab - r_cov,bc))`;
//! - torsion `0.0015 + 14.0 L^0.57 / (r r_cov)^4 exp(-2.85 (r - r_cov))` about the central bond,
//!   `L` the number of bonds of its two atoms besides the central one; torsions with an angle
//!   within about 1.8 degrees of linear are left out.
//!
//! MultiOptPy: `ModelHessian/fischer.py`, `fischerd3.py`, `fischerd4.py` (published model;
//!  MultiOptPy's in `legacy::fischer*`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::assemble::HessianAccumulator;
use super::schlegel::BOND_FACTOR;
use super::{add_dispersion, Dispersion, ModelHessian};
use crate::coord::connectivity::BondConnectivity;
use crate::domain::{AtomId, MolecularSystem};
use crate::error::MoptResult;
use crate::parameters::covalent_radius_bohr;

/// Fischer-Almlof model.
#[derive(Debug, Clone, Copy)]
pub struct Fischer {
    /// Dispersion term.
    pub dispersion: Dispersion,
}

impl Fischer {
    /// Model with `dispersion`.
    pub fn new(dispersion: Dispersion) -> Self {
        Self { dispersion }
    }
}

fn sin2(acc: &HessianAccumulator<'_>, a: AtomId, m: AtomId, c: AtomId) -> f64 {
    let (u, v) = (acc.delta(a, m), acc.delta(c, m));
    let cr = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let (uu, vv) = (
        u.iter().map(|x| x * x).sum::<f64>(),
        v.iter().map(|x| x * x).sum::<f64>(),
    );
    if uu * vv < 1e-12 {
        0.0
    } else {
        cr.iter().map(|x| x * x).sum::<f64>() / (uu * vv)
    }
}

impl ModelHessian for Fischer {
    fn name(&self) -> &str {
        match self.dispersion {
            Dispersion::D3 => "fischerd3",
            Dispersion::D4 => "fischerd4",
            _ => "fischer",
        }
    }

    fn hessian(&self, sys: &MolecularSystem, _gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        let conn = BondConnectivity::from_system(sys, BOND_FACTOR);
        let rcov = |a: AtomId| covalent_radius_bohr(sys.element(a));
        let mut acc = HessianAccumulator::new(sys);
        for &[a, b] in &conn.bonds {
            let r = acc.r2(a, b).sqrt();
            acc.stretch(a, b, 0.3601 * (-1.944 * (r - rcov(a) - rcov(b))).exp());
        }
        for &[a, m, c] in &conn.angles {
            let (rab, rbc) = (acc.r2(a, m).sqrt(), acc.r2(m, c).sqrt());
            let (cab, cbc) = (rcov(a) + rcov(m), rcov(m) + rcov(c));
            let k = 0.089 + 0.11 * (cab * cbc).powf(0.42) * (-0.44 * (rab + rbc - cab - cbc)).exp();
            acc.bend(a, m, c, k);
        }
        for &t in &conn.dihedrals {
            let [a, b, c, d] = t;
            if sin2(&acc, a, b, c) < 1e-3 || sin2(&acc, b, c, d) < 1e-3 {
                continue;
            }
            let r = acc.r2(b, c).sqrt();
            let rc = rcov(b) + rcov(c);
            let bonds = conn.adjacency[b.index()].len() + conn.adjacency[c.index()].len();
            let l = bonds.saturating_sub(2) as f64;
            let k = 0.0015 + 14.0 * l.powf(0.57) / (r * rc).powi(4) * (-2.85 * (r - rc)).exp();
            acc.torsion(t, k);
        }
        let mut h = acc.finish();
        add_dispersion(&mut h, sys, self.dispersion)?;
        Ok(h)
    }
}
