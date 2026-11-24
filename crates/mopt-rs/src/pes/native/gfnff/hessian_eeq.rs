//! Analytic Hessian of the EEQ electrostatics of GFN-FF, with the response of the charges to
//! the positions, for a molecule (`goed_gfnff`) and for a periodic cell (the Ewald matrix of
//! `goed_pbc_gfnff`).
//!
//! The charges solve `A q + C^T lambda = X`, `C q = Q` (`C` the constraint rows of the total
//! and fragment charges, `X_i = chi_i + cnf_i sqrt(CN_i)`), and the gradient is that of the
//! matrix and of `X` at fixed charges, `g_a = q^T (dA/dx_a) q / 2 - q^T dX/dx_a`. Its derivative
//! is
//!
//! `H_ab = q^T (d2A/dx_a dx_b) q / 2 - q^T d2X/dx_a dx_b - r_a^T P r_b`,
//!
//! with `r_a = (dA/dx_a) q - dX/dx_a` and `P` the atom block of the inverse of the bordered
//! matrix (the derivative of the charges is `dq/dx_b = -P r_b`, which keeps the constrained
//! charges fixed). The part of `X` through the coordination numbers goes to the CN chain rule
//! ([`super::hessian_core::Chain`]); `-r^T P r` is formed here, with the CN Jacobian in `r`.
//!
//! With the lattice vectors as variables (a cell), `r` and the explicit part have their
//! columns too: the real-space sum through the cells of its images, the reciprocal sum
//! through the structure factors ([`ewald::reciprocal_lattice`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::{Accum, Mat};

use super::eg::eeq_system;
use super::ewald;
use super::hessian_core::{radial_block, Acc, Chain, CnSecond, PairSum};
use super::lattice::Lattice;
use super::neighbors::V3;
use super::periodic_eg;
use super::GfnffTopology;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{kernel_par, mul, DenseLu};

/// Regularization of `1 / (2 sqrt(CN))` of the gradient.
const SQRT_GUARD: f64 = 1.0e-16;

/// The charges and the atom block of the inverse of the bordered matrix `a` with the
/// right-hand side `rhs`; `None` when the system is singular.
fn charges_and_response(
    a: &Mat<f64>,
    rhs: &faer::Col<f64>,
    n: usize,
) -> Option<(Vec<f64>, Mat<f64>)> {
    let m = a.nrows();
    let lu = DenseLu::new(a.as_ref()).ok()?;
    let sol = lu.solve(rhs.as_mat());
    let inverse = lu.solve(Mat::<f64>::identity(m, m).as_ref());
    let q: Vec<f64> = (0..n).map(|i| sol[(i, 0)]).collect();
    let p = Mat::from_fn(n, n, |i, j| inverse[(i, j)]);
    let finite =
        q.iter().all(|v| v.is_finite()) && (0..n).all(|j| p.col(j).iter().all(|v| v.is_finite()));
    finite.then_some((q, p))
}

/// `r = (dA/dx) q`: add the pair `(i, j)` with `d = dA_ij / dx_i` to the rows of `i` and `j`.
fn add_pair_response(r: &mut Mat<f64>, q: &[f64], i: usize, j: usize, d: V3) {
    for c in 0..3 {
        r[(i, 3 * i + c)] += q[j] * d[c];
        r[(i, 3 * j + c)] -= q[j] * d[c];
        r[(j, 3 * i + c)] += q[i] * d[c];
        r[(j, 3 * j + c)] -= q[i] * d[c];
    }
}

/// The CN part of `X` (into the chain rule and into `r`), then `-r^T P r`.
fn finish(
    t: &GfnffTopology,
    q: &[f64],
    cn: &CnSecond,
    mut r: Mat<f64>,
    p: &Mat<f64>,
    acc: &mut Acc,
    chain: &mut Chain,
) {
    let s = &t.setup;
    for (i, &qi) in q.iter().enumerate() {
        let c = cn.cn[i];
        let cnf = s.el[i].raw.cnf;
        let root = c.sqrt();
        // dX_i / dCN_i and d2X_i / dCN_i^2 of the regularized gradient.
        let beta = cnf / (2.0 * root + SQRT_GUARD);
        let beta1 = if c > 0.0 {
            -cnf / (root * (2.0 * root + SQRT_GUARD).powi(2))
        } else {
            0.0
        };
        chain.w[i] -= qi * beta;
        chain.m[(i, i)] -= qi * beta1;
        for (a, d) in cn.jacobian_row(i) {
            for cc in 0..3 {
                r[(i, 3 * a + cc)] -= beta * d[cc];
            }
        }
    }
    let pr = mul(p.as_ref(), r.as_ref());
    faer::linalg::matmul::matmul(
        acc.h.as_mut(),
        Accum::Add,
        r.transpose(),
        pr.as_ref(),
        -1.0,
        kernel_par(),
    );
}

/// The EEQ Hessian of a molecule.
pub fn molecule(
    t: &GfnffTopology,
    x: &[V3],
    cn: &CnSecond,
    acc: &mut Acc,
    chain: &mut Chain,
) -> MoptResult<()> {
    let s = &t.setup;
    let n = x.len();
    let (a, rhs, pair) = eeq_system(t, x, &cn.cn);
    let (mut q, p) = charges_and_response(&a, &rhs, n)
        .ok_or_else(|| MoptError::calc("GFN-FF: singular EEQ system"))?;
    if n == 1 {
        q[0] = s.qfrag[0];
    }
    let mut r = Mat::<f64>::zeros(n, acc.vars.dim());
    let sqrtpi = PI.sqrt();
    for i in 0..n {
        for j in 0..i {
            let rv = [0, 1, 2].map(|c| x[i][c] - x[j][c]);
            let r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
            let rr = r2.sqrt();
            let (gammij, erff) = pair[super::eg::pair_index(i, j)];
            let gauss = 2.0 * gammij * (-gammij * gammij * r2).exp() / sqrtpi;
            // erf(g r) / r: its first and second derivatives.
            let d1 = gauss / rr - erff / r2;
            let d2 = -gauss * (2.0 * gammij * gammij + 2.0 / r2) + 2.0 * erff / (r2 * rr);
            let qq = q[i] * q[j];
            acc.pair_block(i, j, &radial_block(rv, qq * d1, qq * d2));
            add_pair_response(&mut r, &q, i, j, rv.map(|c| d1 / rr * c));
        }
    }
    finish(t, &q, cn, r, &p, acc, chain);
    Ok(())
}

/// The EEQ Hessian of a periodic cell (the Ewald matrix at fixed splitting parameter and
/// images, as the gradient).
pub fn periodic(
    t: &GfnffTopology,
    x: &[V3],
    lattice: &Lattice,
    cn: &CnSecond,
    acc: &mut Acc,
    chain: &mut Chain,
) -> MoptResult<()> {
    let n = x.len();
    let system = periodic_eg::eeq_system(t, x, lattice, &cn.cn);
    let (q, p) = charges_and_response(&system.matrix, &system.rhs, n)
        .ok_or_else(|| MoptError::calc("GFN-FF: singular periodic EEQ system"))?;
    let mut r = Mat::<f64>::zeros(n, acc.vars.dim());
    ewald::pair_derivatives(
        x,
        &system.setup,
        &system.images,
        &t.setup.alpeeq,
        |i, j, d, k| {
            let qq = q[i] * q[j];
            acc.pair_block(i, j, &k.map(|row| row.map(|v| qq * v)));
            add_pair_response(&mut r, &q, i, j, d);
        },
    );
    if acc.vars.lattice {
        lattice_parts(x, &q, &system, &t.setup.alpeeq, acc, &mut r);
    }
    finish(t, &q, cn, r, &p, acc, chain);
    Ok(())
}

/// The parts of the explicit Hessian and of `r` that involve the lattice vectors.
fn lattice_parts(
    x: &[V3],
    q: &[f64],
    system: &periodic_eg::PeriodicEeqSystem,
    alpeeq: &[f64],
    acc: &mut Acc,
    r: &mut Mat<f64>,
) {
    let n = x.len();
    let base = 3 * n;
    // The real-space sum: the lattice moments of each pair and of each atom with its images
    // (the diagonal element, whose energy is q_i^2 A_ii / 2).
    ewald::real_lattice_moments(x, &system.setup, &system.images, alpeeq, |i, j, m| {
        let c = if i == j {
            0.5 * q[i] * q[i]
        } else {
            q[i] * q[j]
        };
        let sum = PairSum {
            k0: [[0.0; 3]; 3],
            k1: m.k1,
            k2: m.k2,
        };
        acc.pair_sum(i, j, &sum, c);
        for kk in 0..3 {
            for cc in 0..3 {
                let v = m.d1[kk][cc];
                r[(i, base + 3 * kk + cc)] += q[j] * v;
                if i != j {
                    r[(j, base + 3 * kk + cc)] += q[i] * v;
                }
            }
        }
    });
    // The reciprocal sum.
    let rec = ewald::reciprocal_lattice(x, q, &system.setup);
    for i in 0..n {
        for cc in 0..3 {
            for v in 0..9 {
                let value = rec.xh[i][cc][v];
                acc.h[(3 * i + cc, base + v)] += value;
                acc.h[(base + v, 3 * i + cc)] += value;
            }
        }
        for v in 0..9 {
            r[(i, base + v)] += rec.response[i][v];
        }
    }
    for v in 0..9 {
        for w in 0..9 {
            acc.h[(base + v, base + w)] += rec.hh[v][w];
        }
    }
}
