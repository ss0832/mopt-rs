//! Per-term ("local") differentiation of functions of a few atoms, and scattering of the local
//! derivatives into global `3N` gradients and `3N x 3N` Hessians.
//!
//! Potentials, bias terms, and internal coordinates are sums of terms that each depend on
//! `K` atoms (`K` = 2 for pairs, 3 for angles, 4 for dihedrals). Differentiating each term over
//! its `3K` local variables keeps the cost of an exact Hessian at `O(K^2)` per term.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (replaces the `torch_*` functions of
//!  `Utils/calc_tools.py` and `Coordinate/redundant_coordinate.py`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::drivers::{eval_value_and_gradient, eval_value_grad_hessian};
use super::dual::Dual;
use super::hyperdual::HyperDual;
use super::scalar::Scalar;
use crate::domain::AtomId;

/// Local derivatives of a `K`-atom term.
#[derive(Debug, Clone)]
pub struct LocalDerivatives<const K: usize> {
    /// Term value.
    pub value: f64,
    /// Gradient with respect to each atom's Cartesian position.
    pub gradient: [[f64; 3]; K],
    /// `3K x 3K` Hessian (row/column `3 * slot + component`), present when requested.
    pub hessian: Option<Mat<f64>>,
}

fn flatten<const K: usize>(positions: &[[f64; 3]; K]) -> Vec<f64> {
    positions.iter().flat_map(|p| p.iter().copied()).collect()
}

fn unflatten<T: Scalar, const K: usize>(vars: &[T]) -> [[T; 3]; K] {
    std::array::from_fn(|s| [vars[3 * s], vars[3 * s + 1], vars[3 * s + 2]])
}

/// Value and gradient of `f` at `positions` (forward-mode `Dual`).
pub fn local_value_grad<const K: usize, F>(positions: &[[f64; 3]; K], f: F) -> LocalDerivatives<K>
where
    F: Fn(&[[Dual; 3]; K]) -> Dual,
{
    let x = flatten(positions);
    let (value, g) = eval_value_and_gradient(&x, |vars| f(&unflatten::<Dual, K>(vars)));
    LocalDerivatives {
        value,
        gradient: std::array::from_fn(|s| [g[3 * s], g[3 * s + 1], g[3 * s + 2]]),
        hessian: None,
    }
}

/// Value, gradient, and Hessian of `f` at `positions` (hyper-dual numbers).
pub fn local_value_grad_hess<const K: usize, F>(
    positions: &[[f64; 3]; K],
    f: F,
) -> LocalDerivatives<K>
where
    F: Fn(&[[HyperDual; 3]; K]) -> HyperDual,
{
    let x = flatten(positions);
    let (value, g, h) =
        eval_value_grad_hessian(&x, None, |vars| f(&unflatten::<HyperDual, K>(vars)));
    LocalDerivatives {
        value,
        gradient: std::array::from_fn(|s| [g[3 * s], g[3 * s + 1], g[3 * s + 2]]),
        hessian: Some(h),
    }
}

/// Add `weight * local_gradient` into the global `3N` gradient.
pub fn scatter_gradient<const K: usize>(
    global: &mut Col<f64>,
    atoms: &[AtomId; K],
    local: &[[f64; 3]; K],
    weight: f64,
) {
    for (slot, &atom) in atoms.iter().enumerate() {
        for c in 0..3 {
            global[atom.dof(c)] += weight * local[slot][c];
        }
    }
}

/// Add `weight * local_hessian` (`3K x 3K`) into the global `3N x 3N` Hessian.
pub fn scatter_hessian<const K: usize>(
    global: &mut Mat<f64>,
    atoms: &[AtomId; K],
    local: &Mat<f64>,
    weight: f64,
) {
    for (sa, &a) in atoms.iter().enumerate() {
        for ca in 0..3 {
            for (sb, &b) in atoms.iter().enumerate() {
                for cb in 0..3 {
                    global[(a.dof(ca), b.dof(cb))] += weight * local[(3 * sa + ca, 3 * sb + cb)];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist<T: Scalar>(p: &[[T; 3]; 2]) -> T {
        let d = [p[0][0] - p[1][0], p[0][1] - p[1][1], p[0][2] - p[1][2]];
        T::norm3(d)
    }

    #[test]
    fn distance_derivatives_and_scatter() {
        let pos = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]];
        let d1 = local_value_grad(&pos, dist);
        assert!((d1.value - 5.0).abs() < 1e-15);
        assert!((d1.gradient[1][0] - 0.6).abs() < 1e-15);
        assert!((d1.gradient[0][1] + 0.8).abs() < 1e-15);

        let d2 = local_value_grad_hess(&pos, dist);
        let h = d2.hessian.unwrap();
        // d2r/dx1^2 = (1 - x^2/r^2)/r = (1 - 0.36)/5
        assert!((h[(3, 3)] - 0.128).abs() < 1e-14);

        let mut g = Col::zeros(9);
        scatter_gradient(&mut g, &[AtomId::new(2), AtomId::new(0)], &d1.gradient, 2.0);
        assert!((g[0] - 1.2).abs() < 1e-15 && (g[6] + 1.2).abs() < 1e-15);
        let mut hg = Mat::zeros(9, 9);
        scatter_hessian(&mut hg, &[AtomId::new(2), AtomId::new(0)], &h, 1.0);
        assert!((hg[(0, 0)] - 0.128).abs() < 1e-14);
        assert!((hg[(6, 0)] + 0.128).abs() < 1e-14);
    }
}
