//! Drivers evaluating values, gradients, Hessians, and Jacobians of closures over AD scalars.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy differentiates with torch).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::dual::Dual;
use super::hyperdual::HyperDual;
use super::scalar::Scalar;

/// Evaluate exact scalar value and gradient `grad` (`n`) of `f(x)` via forward-mode `Dual` numbers.
pub fn eval_value_and_gradient<F>(x: &[f64], mut f: F) -> (f64, Col<f64>)
where
    F: FnMut(&[Dual]) -> Dual,
{
    let n = x.len();
    if n == 0 {
        return (f(&[]).re, Col::zeros(0));
    }
    let mut vars: Vec<Dual> = x.iter().map(|&v| Dual::constant(v)).collect();
    let mut grad = Col::zeros(n);
    let mut f0 = 0.0;
    for i in 0..n {
        vars[i].eps = 1.0;
        let out = f(&vars);
        if i == 0 {
            f0 = out.re;
        }
        grad[i] = out.eps;
        vars[i].eps = 0.0;
    }
    (f0, grad)
}

/// Evaluate exact scalar value, gradient (`n`), and symmetric Hessian (`n x n`) of `f(x)`
/// via forward-mode `HyperDual` numbers (`active_indices` optionally restricts evaluation to
/// the non-zero subset of participating coordinates for `O(|active|^2)` scaling on large systems!).
pub fn eval_value_grad_hessian<F>(
    x: &[f64],
    active_indices: Option<&[usize]>,
    mut f: F,
) -> (f64, Col<f64>, Mat<f64>)
where
    F: FnMut(&[HyperDual]) -> HyperDual,
{
    let n = x.len();
    let mut grad = Col::zeros(n);
    let mut hess = Mat::zeros(n, n);
    if n == 0 {
        return (f(&[]).f0, grad, hess);
    }

    let mut vars: Vec<HyperDual> = x.iter().map(|&v| HyperDual::constant(v)).collect();
    let default_active: Vec<usize>;
    let active: &[usize] = match active_indices {
        Some(idx) if !idx.is_empty() => idx,
        Some(_) => return (f(&vars).f0, grad, hess),
        None => {
            default_active = (0..n).collect();
            &default_active
        }
    };

    let mut f0 = 0.0;
    for (pos_a, &i) in active.iter().enumerate() {
        // Diagonal element (i, i)
        vars[i].f1 = 1.0;
        vars[i].f2 = 1.0;
        let diag_out = f(&vars);
        if pos_a == 0 {
            f0 = diag_out.f0;
        }
        grad[i] = diag_out.f1;
        hess[(i, i)] = diag_out.f12;
        vars[i].f2 = 0.0;

        // Off-diagonal elements (i, j) for j > i in active set
        for &j in &active[pos_a + 1..] {
            vars[j].f2 = 1.0;
            let off_out = f(&vars);
            hess[(i, j)] = off_out.f12;
            hess[(j, i)] = off_out.f12;
            vars[j].f2 = 0.0;
        }
        vars[i].f1 = 0.0;
    }

    (f0, grad, hess)
}

/// Evaluate exact vector value `m` and Jacobian `J` (`m x n`, where `J[(r, c)] = dy_r / dx_c`).
pub fn eval_vector_and_jacobian<F>(x: &[f64], m: usize, mut f: F) -> (Col<f64>, Mat<f64>)
where
    F: FnMut(&[Dual], &mut [Dual]),
{
    let n = x.len();
    let mut val = Col::zeros(m);
    let mut jac = Mat::zeros(m, n);
    if n == 0 || m == 0 {
        return (val, jac);
    }
    let mut vars: Vec<Dual> = x.iter().map(|&v| Dual::constant(v)).collect();
    let mut out = vec![Dual::zero(); m];
    for c in 0..n {
        vars[c].eps = 1.0;
        f(&vars, &mut out);
        if c == 0 {
            for r in 0..m {
                val[r] = out[r].re;
            }
        }
        for r in 0..m {
            jac[(r, c)] = out[r].eps;
        }
        vars[c].eps = 0.0;
    }
    (val, jac)
}
