//! Muller-Brown model surface (Muller and Brown, Theor. Chim. Acta 1979, 53, 75).
//!
//! The surface acts on the `x`, `y` coordinates of a single atom (Bohr are used as the model's
//! length unit and Hartree as its energy unit, scaled by `energy_scale`); `z` is a spectator
//! with a harmonic restraint. Used to verify optimizers, saddle searches, path methods, and IRC
//! against known stationary points.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (test surface).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::AtomId;
use crate::domain::Quantity;
use faer::{Col, Mat};

use crate::domain::{EvalOutput, EvalRequest, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_grad_hessian, Scalar};
use crate::pes::calculator::Calculator;

const A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const BB: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const CC: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

/// Known stationary points `(x, y)`: minima A, B, C and saddle points (first-order).
pub const MB_MINIMA: [[f64; 2]; 3] = [
    [-0.558_224, 1.441_726],
    [0.623_499, 0.028_038],
    [-0.050_011, 0.466_694],
];
/// Known first-order saddle points.
pub const MB_SADDLES: [[f64; 2]; 2] = [[-0.822_002, 0.624_313], [0.212_487, 0.292_988]];

/// Muller-Brown surface for a single atom.
#[derive(Debug, Clone, Copy)]
pub struct MullerBrown {
    /// Factor applied to the original potential (default `1e-3`, keeping energies in a
    /// chemically sized range).
    pub energy_scale: f64,
    /// Harmonic force constant on `z`.
    pub z_force_constant: f64,
}

impl Default for MullerBrown {
    fn default() -> Self {
        Self {
            energy_scale: 1e-3,
            z_force_constant: 1.0,
        }
    }
}

/// Generic Muller-Brown value.
pub fn muller_brown<T: Scalar>(x: T, y: T) -> T {
    let mut v = T::zero();
    for k in 0..4 {
        let dx = x - T::from_f64(X0[k]);
        let dy = y - T::from_f64(Y0[k]);
        let arg = T::from_f64(AA[k]) * dx * dx
            + T::from_f64(BB[k]) * dx * dy
            + T::from_f64(CC[k]) * dy * dy;
        v += T::from_f64(A[k]) * arg.exp();
    }
    v
}

impl Calculator for MullerBrown {
    fn analytic_hessian(&self) -> bool {
        true
    }

    fn citations(&self) -> Vec<String> {
        vec![
            "K. Muller, L. D. Brown, Theor. Chim. Acta 1979, 53, 75 (Muller-Brown surface)"
                .to_string(),
        ]
    }

    fn name(&self) -> &str {
        "muller_brown"
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if sys.n_atoms() != 1 {
            return Err(MoptError::config(
                "the Muller-Brown surface needs exactly one atom",
            ));
        }
        let p = sys.position(AtomId::new(0)).value();
        let (s, kz) = (self.energy_scale, self.z_force_constant);
        let (e, g, h) = eval_value_grad_hessian(&p[..2], None, |v| muller_brown(v[0], v[1]));
        let energy = s * e + 0.5 * kz * p[2] * p[2];
        let gradient = Col::from_fn(3, |i| if i < 2 { s * g[i] } else { kz * p[2] });
        let hessian = req.need_hessian.then(|| {
            Mat::from_fn(3, 3, |i, j| match (i, j) {
                (2, 2) => kz,
                (2, _) | (_, 2) => 0.0,
                _ => s * h[(i, j)],
            })
        });
        Ok(EvalOutput {
            energy: Quantity::new(energy),
            gradient_bohr: gradient,
            hessian_bohr: hessian,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ad::eval_value_and_gradient;

    #[test]
    fn tabulated_points_are_stationary() {
        for p in MB_MINIMA.iter().chain(MB_SADDLES.iter()) {
            let (_, g) = eval_value_and_gradient(p, |v| muller_brown(v[0], v[1]));
            assert!(g[0].abs() < 2e-3 && g[1].abs() < 2e-3, "{p:?}: {g:?}");
        }
    }
}
