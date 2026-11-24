//! Morse pair model Hessian (MultiOptPy `morse`): every atom pair contributes the second
//! derivative of `De (1 - exp(-a (r - r_eq)))^2` along its bond, `r_eq` the sum of the GNB
//! radii.
//!
//! MultiOptPy: `ModelHessian/morse.py:MorseApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::assemble::HessianAccumulator;
use super::ModelHessian;
use crate::domain::{AtomId, MolecularSystem};
use crate::error::MoptResult;
use crate::parameters::gnb_radius_bohr;

/// Morse pair model.
#[derive(Debug, Clone, Copy)]
pub struct MorseModel {
    /// Well depth `De` (Hartree).
    pub depth: f64,
    /// Steepness `a` (1/Bohr).
    pub a: f64,
}

impl Default for MorseModel {
    /// MultiOptPy's `De = 0.1`, `a = 0.2`.
    fn default() -> Self {
        Self { depth: 0.1, a: 0.2 }
    }
}

impl ModelHessian for MorseModel {
    fn name(&self) -> &str {
        "morse"
    }

    fn hessian(&self, sys: &MolecularSystem, _gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        let n = sys.n_atoms();
        let mut acc = HessianAccumulator::new(sys);
        for i in 0..n {
            for j in i + 1..n {
                let (a, b) = (AtomId::new(i), AtomId::new(j));
                let req = gnb_radius_bohr(sys.element(a)) + gnb_radius_bohr(sys.element(b));
                let d = acc.delta(b, a);
                let r = d.iter().map(|x| x * x).sum::<f64>().sqrt();
                if r <= 1e-12 {
                    continue;
                }
                let x = self.a * (r - req);
                let d2v =
                    self.depth * self.a * self.a * (-2.0 * (-x).exp() + 4.0 * (-2.0 * x).exp());
                let u = d.map(|v| v / r);
                let block: [[f64; 3]; 3] =
                    std::array::from_fn(|p| std::array::from_fn(|q| d2v * u[p] * u[q]));
                acc.pair(a, b, &block);
            }
        }
        Ok(acc.finish())
    }
}
