//! Combiners defined by a scalar function of the state energies, `F(E_1, ..., E_n)`.
//!
//! The function is written once over [`Scalar`] and differentiated with hyper-dual numbers, so
//! the effective gradient and Hessian are exact:
//! `grad F = sum_k F_k grad E_k`,
//! `Hess F = sum_kl F_kl grad E_k grad E_l^T + sum_k F_k Hess E_k`.
//! New model functions only need [`EnergyFunction::eval`].
//!
//! MultiOptPy: `ModelFunction/` (`opt_meci.py`, `opt_mesx.py`, `opt_mesx_2.py`,
//!  `seam_model_function.py`, `avoiding_model_function.py`, `conical_model_function.py`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{Combined, StateCombiner, StateValues};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_grad_hessian, Scalar};

/// A scalar function of the state energies (Hartree).
pub trait EnergyFunction: Send + Sync {
    /// Canonical name.
    fn name(&self) -> &str;
    /// Number of states.
    fn n_states(&self) -> usize;
    /// `F(E_1, ..., E_n)`.
    fn eval<T: Scalar>(&self, e: &[T]) -> T;
}

/// Combiner from an [`EnergyFunction`].
#[derive(Debug, Clone)]
pub struct FunctionCombiner<F: EnergyFunction> {
    /// The function.
    pub function: F,
}

impl<F: EnergyFunction> FunctionCombiner<F> {
    /// Wrap `function`.
    pub fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F: EnergyFunction> StateCombiner for FunctionCombiner<F> {
    fn name(&self) -> &str {
        self.function.name()
    }

    fn n_states(&self) -> usize {
        self.function.n_states()
    }

    fn combine(&self, _sys: &MolecularSystem, s: &StateValues) -> MoptResult<Combined> {
        let n = s.energies.len();
        if n != self.function.n_states() {
            return Err(MoptError::config(format!(
                "{} needs {} states, got {n}",
                self.function.name(),
                self.function.n_states()
            )));
        }
        let (f, df, d2f) = eval_value_grad_hessian(&s.energies, None, |e| self.function.eval(e));
        let dim = s.gradients[0].nrows();
        let mut grad = Col::zeros(dim);
        for k in 0..n {
            grad += &s.gradients[k] * df[k];
        }
        let hessian = s.hessians.as_ref().map(|hs| {
            let mut h = Mat::zeros(dim, dim);
            for k in 0..n {
                h += &hs[k] * df[k];
                for l in 0..n {
                    let w = d2f[(k, l)];
                    if w != 0.0 {
                        for i in 0..dim {
                            for j in 0..dim {
                                h[(i, j)] += w * s.gradients[k][i] * s.gradients[l][j];
                            }
                        }
                    }
                }
            }
            h
        });
        Ok(Combined {
            energy: f,
            gradient: grad,
            hessian,
        })
    }
}

fn mean<T: Scalar>(e: &[T]) -> T {
    (e[0] + e[1]) * T::from_f64(0.5)
}

/// `(E_1 + E_2) / 2 + w (E_1 - E_2)^2`: the seam model function of Sato et al. with
/// `w = 1 / alpha` (J. Phys. Chem. A 2009, 113, 1704; MultiOptPy `seam`, default
/// `alpha = 0.05`) and EDEEL's crossing penalty with `w = sigma`.
#[derive(Debug, Clone, Copy)]
pub struct QuadraticPenalty {
    /// Weight `w` (1/Hartree).
    pub weight: f64,
}

impl QuadraticPenalty {
    /// Seam model function with parameter `alpha` (Hartree).
    pub fn seam(alpha: f64) -> Self {
        Self {
            weight: 1.0 / alpha,
        }
    }
}

impl EnergyFunction for QuadraticPenalty {
    fn name(&self) -> &str {
        "seam"
    }
    fn n_states(&self) -> usize {
        2
    }
    fn eval<T: Scalar>(&self, e: &[T]) -> T {
        let d = e[0] - e[1];
        mean(e) + d * d * T::from_f64(self.weight)
    }
}

/// Penalty function for conical intersections (Levine, Coe, Martinez, J. Phys. Chem. B 2008,
/// 112, 405): `(E_1 + E_2) / 2 + sigma dE^2 / (dE + alpha)` with `dE = |E_1 - E_2|`
/// (MultiOptPy `conical`: `alpha = 0.025`, `sigma = 3.5`).
///
/// The gap is taken as an absolute value as in the paper (upper minus lower state); MultiOptPy
/// uses the signed difference, which becomes singular at `E_1 - E_2 = -alpha` when the states
/// are given in the opposite order.
#[derive(Debug, Clone, Copy)]
pub struct ConicalPenalty {
    /// Smoothing `alpha` (Hartree).
    pub alpha: f64,
    /// Penalty weight `sigma`.
    pub sigma: f64,
}

impl Default for ConicalPenalty {
    fn default() -> Self {
        Self {
            alpha: 0.025,
            sigma: 3.5,
        }
    }
}

impl EnergyFunction for ConicalPenalty {
    fn name(&self) -> &str {
        "conical"
    }
    fn n_states(&self) -> usize {
        2
    }
    fn eval<T: Scalar>(&self, e: &[T]) -> T {
        let d = (e[0] - e[1]).abs();
        mean(e) + T::from_f64(self.sigma) * d * d / (d + T::from_f64(self.alpha))
    }
}

/// The default `alpha` of [`AvoidingModel`]: 30 kJ/mol in Hartree.
pub const AVOIDING_ALPHA: f64 = 30.0 * crate::domain::Units::KJ_MOL_TO_HARTREE;

/// Avoiding model function (Maeda, Ohno, Morokuma, Adv. Phys. Chem. 2012, 268124): an adiabatic
/// state of two states `E_1`, `E_2` taken as diabatic and coupled by
/// `U = (alpha / 2) exp(-((E_1 - E_2) / alpha)^2)`,
/// `(E_1 + E_2) / 2 +- sqrt((E_1 - E_2)^2 + 4 U^2) / 2` (`-mf avoiding` the upper state,
/// `-mf avoiding_lower` the lower). The coupling fades away from the seam, where the function
/// is the higher (lower) of the two states, and makes it smooth through the seam: on the upper
/// state the seam is a valley, on the lower a ridge.
#[derive(Debug, Clone, Copy)]
pub struct AvoidingModel {
    /// Coupling width `alpha` (Hartree).
    pub alpha: f64,
    /// The lower adiabatic state instead of the upper.
    pub lower: bool,
}

impl Default for AvoidingModel {
    fn default() -> Self {
        Self {
            alpha: AVOIDING_ALPHA,
            lower: false,
        }
    }
}

impl EnergyFunction for AvoidingModel {
    fn name(&self) -> &str {
        if self.lower {
            "avoiding_lower"
        } else {
            "avoiding"
        }
    }
    fn n_states(&self) -> usize {
        2
    }
    fn eval<T: Scalar>(&self, e: &[T]) -> T {
        let d = e[0] - e[1];
        let x = d / T::from_f64(self.alpha);
        let u = T::from_f64(0.5 * self.alpha) * (-(x * x)).exp();
        let root = (d * d + T::from_f64(4.0) * u * u).sqrt() * T::from_f64(0.5);
        if self.lower {
            mean(e) - root
        } else {
            mean(e) + root
        }
    }
}

/// MultiOptPy's `avoiding` (`-mf legacy_avoiding`, `alpha = 0.01`): the upper adiabatic state
/// with the coupling `U = (alpha/2) exp(-dE^2 / alpha)`, whose exponent is an energy and not a
/// number, `(E_1 + E_2) / 2 + sqrt(dE^2 + 4 U^2) / 2`.
///
/// MultiOptPy evaluates the energy as `sqrt(dE^2 + 4 U) / 2` but its gradient is that of
/// `sqrt(dE^2 + 4 U^2)`; the consistent two-state form is used here (see
/// `docs/source/migration_surfaces.md`).
#[derive(Debug, Clone, Copy)]
pub struct LegacyAvoidingModel {
    /// Coupling width `alpha` (Hartree).
    pub alpha: f64,
}

impl Default for LegacyAvoidingModel {
    fn default() -> Self {
        Self { alpha: 0.01 }
    }
}

impl EnergyFunction for LegacyAvoidingModel {
    fn name(&self) -> &str {
        "legacy_avoiding"
    }
    fn n_states(&self) -> usize {
        2
    }
    fn eval<T: Scalar>(&self, e: &[T]) -> T {
        let d = e[0] - e[1];
        let u = T::from_f64(0.5 * self.alpha) * (-(d * d) / T::from_f64(self.alpha)).exp();
        mean(e) + (d * d + T::from_f64(4.0) * u * u).sqrt() * T::from_f64(0.5)
    }
}

/// Weighted sum `sum_k w_k E_k` (e.g. the mean energy, or a state-averaged surface).
#[derive(Debug, Clone)]
pub struct WeightedSum {
    /// Weights.
    pub weights: Vec<f64>,
}

impl EnergyFunction for WeightedSum {
    fn name(&self) -> &str {
        "weighted_sum"
    }
    fn n_states(&self) -> usize {
        self.weights.len()
    }
    fn eval<T: Scalar>(&self, e: &[T]) -> T {
        e.iter()
            .zip(&self.weights)
            .fold(T::zero(), |acc, (x, w)| acc + *x * T::from_f64(*w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_gradient<F: EnergyFunction>(f: &F, e: [f64; 2]) {
        let (_, g, _) = eval_value_grad_hessian(&e, None, |x| f.eval(x));
        let h = 1e-7;
        for k in 0..2 {
            let mut ep = e;
            let mut em = e;
            ep[k] += h;
            em[k] -= h;
            let fd = (f.eval(&ep) - f.eval(&em)) / (2.0 * h);
            assert!(
                (fd - g[k]).abs() < 1e-6,
                "{} d/dE{k}: {fd} vs {}",
                f.name(),
                g[k]
            );
        }
    }

    #[test]
    fn model_functions_are_consistent() {
        for e in [[-1.0, -0.98], [-0.97, -1.0], [-1.0, -1.0 + 1e-4]] {
            check_gradient(&QuadraticPenalty::seam(0.05), e);
            check_gradient(&ConicalPenalty::default(), e);
            check_gradient(&AvoidingModel::default(), e);
            check_gradient(
                &AvoidingModel {
                    lower: true,
                    ..AvoidingModel::default()
                },
                e,
            );
            check_gradient(&LegacyAvoidingModel::default(), e);
        }
        // The Avoiding model function: the mean plus or minus alpha / 2 at the seam, the higher
        // and the lower state far from it.
        let (upper, lower) = (
            AvoidingModel::default(),
            AvoidingModel {
                lower: true,
                ..AvoidingModel::default()
            },
        );
        let half = 0.5 * AVOIDING_ALPHA;
        assert!((upper.eval(&[-1.0, -1.0]) - (-1.0 + half)).abs() < 1e-15);
        assert!((lower.eval(&[-1.0, -1.0]) - (-1.0 - half)).abs() < 1e-15);
        assert!((upper.eval(&[-1.0, -0.8]) + 0.8).abs() < 1e-12);
        assert!((lower.eval(&[-1.0, -0.8]) + 1.0).abs() < 1e-12);
        // Seam penalty vanishes at the crossing; conical is symmetric in the state order.
        let c = ConicalPenalty::default();
        assert!((c.eval(&[-1.0, -0.9]) - c.eval(&[-0.9, -1.0])).abs() < 1e-15);
        assert!((QuadraticPenalty::seam(0.05).eval(&[-1.0, -1.0]) + 1.0).abs() < 1e-15);
    }
}
