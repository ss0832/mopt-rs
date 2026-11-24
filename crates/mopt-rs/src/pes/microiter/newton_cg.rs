//! Truncated Newton minimization of a relaxation.
//!
//! Each step solves `H p = -g` by conjugate gradients on Hessian-vector products from forward
//! differences of the gradient (one gradient a product), preconditioned by the inverse of the
//! limited-memory BFGS pairs of the accepted steps. The solve stops at the relative residual
//! `eta = min(0.5, |g|^1/2)` (a forcing term that shrinks with the gradient, so that the
//! steps become Newton steps near the minimum), or at a direction of negative or vanishing
//! curvature (Steihaug's rule: the first direction is then the preconditioned steepest
//! descent). A backtracking line search on the energy follows, as in the limited-memory BFGS.
//!
//! References: R. S. Dembo, S. C. Eisenstat, T. Steihaug, SIAM J. Numer. Anal. 1982, 19, 400
//! (inexact Newton); S. G. Nash, J. Comput. Appl. Math. 2000, 124, 45 (truncated Newton).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::memory::MicroMemory;
use super::solver::CURVATURE_STEP;
use super::{MicroProblem, MicroResult, MicroSettings};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_max_abs, col_norm};

/// Conjugate-gradient iterations (Hessian-vector products) of one Newton step at most.
const PRODUCTS_PER_STEP: usize = 50;

/// `H v` at `y` from forward differences of the gradient (`g` is the gradient at `y`).
fn product(
    problem: &dyn MicroProblem,
    y: &[f64],
    g: &Col<f64>,
    v: &Col<f64>,
) -> MoptResult<Col<f64>> {
    let norm = col_norm(v);
    if norm < 1e-300 {
        return Ok(Col::zeros(v.nrows()));
    }
    let moved: Vec<f64> = y
        .iter()
        .zip(v.iter())
        .map(|(a, b)| a + CURVATURE_STEP * b / norm)
        .collect();
    let gp = problem.evaluate(&moved, false)?.gradient;
    Ok(Col::from_fn(v.nrows(), |i| {
        (gp[i] - g[i]) * norm / CURVATURE_STEP
    }))
}

/// The truncated Newton step at `y` (gradient `g`): `(step, products used)`.
fn newton_step(
    problem: &dyn MicroProblem,
    y: &[f64],
    g: &Col<f64>,
    memory: &MicroMemory,
) -> MoptResult<(Col<f64>, usize)> {
    let n = g.nrows();
    let g_norm = col_norm(g);
    let eta = g_norm.sqrt().min(0.5);
    let mut p = Col::<f64>::zeros(n);
    let mut r = -g.clone();
    let mut z = memory.apply_inverse(&r);
    let mut d = z.clone();
    let mut rz = col_dot(&r, &z);
    let mut products = 0;
    for j in 0..PRODUCTS_PER_STEP {
        let hd = product(problem, y, g, &d)?;
        products += 1;
        let curvature = col_dot(&d, &hd);
        if curvature <= 1e-12 * col_dot(&d, &d) {
            if j == 0 {
                p = d;
            }
            break;
        }
        let alpha = rz / curvature;
        p = Col::from_fn(n, |i| p[i] + alpha * d[i]);
        r = Col::from_fn(n, |i| r[i] - alpha * hd[i]);
        if col_norm(&r) <= eta * g_norm {
            break;
        }
        z = memory.apply_inverse(&r);
        let rz_next = col_dot(&r, &z);
        let beta = rz_next / rz;
        rz = rz_next;
        d = Col::from_fn(n, |i| z[i] + beta * d[i]);
    }
    Ok((p, products))
}

/// One truncated Newton relaxation from `y` (no mode kill). The iterations counted are the
/// Newton steps and the Hessian-vector products together (each costs about one gradient).
pub(super) fn descend_tn(
    problem: &dyn MicroProblem,
    mut y: Vec<f64>,
    s: &MicroSettings,
    memory: &mut MicroMemory,
) -> MoptResult<MicroResult> {
    let n = y.len();
    let mut cur = problem.evaluate(&y, false)?;
    let mut iterations = 0;
    let mut converged = n == 0 || super::is_converged(&cur.gradient, s);
    let mut stall = super::Stall::new(cur.energy, col_max_abs(&cur.gradient));
    let mut stalled = false;
    let mut line_search_failed = false;
    while !converged && iterations < s.max_iterations {
        iterations += 1;
        let g = cur.gradient.clone();
        let (mut p, products) = newton_step(problem, &y, &g, memory)?;
        iterations += products;
        if col_dot(&g, &p) >= 0.0 {
            p = -g.clone();
        }
        let norm = col_norm(&p);
        if norm > s.max_step {
            p *= faer::Scale(s.max_step / norm);
        }
        let slope = col_dot(&g, &p);
        let roundoff = 1e-14 * cur.energy.abs().max(1.0);
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let yn: Vec<f64> = y.iter().zip(p.iter()).map(|(a, b)| a + alpha * b).collect();
            let en = problem.evaluate(&yn, false)?;
            if en.energy.is_finite() && en.energy <= cur.energy + 1e-4 * alpha * slope + roundoff {
                accepted = Some((yn, en));
                break;
            }
            alpha *= 0.5;
        }
        // No step along the direction lowers the energy (or only one at the round-off):
        // the surface does not follow its gradient here, and the relaxation goes on with a
        // method without line searches (`relax_in`).
        let Some((yn, en)) = accepted.filter(|_| alpha >= super::SMALLEST_STEP) else {
            line_search_failed = true;
            break;
        };
        let sv = Col::from_fn(n, |i| yn[i] - y[i]);
        memory.push(sv, &en.gradient - &g, s.memory);
        y = yn;
        cur = en;
        super::tell_step(problem, iterations, cur.energy, &cur.gradient);
        converged = super::is_converged(&cur.gradient, s);
        if !converged && stall.stalled_after(cur.energy, col_max_abs(&cur.gradient)) {
            stalled = true;
            break;
        }
    }
    Ok(MicroResult {
        max_gradient: col_max_abs(&cur.gradient),
        energy: cur.energy,
        y,
        iterations,
        killed_modes: 0,
        converged,
        stalled,
        line_search_failed,
        fallbacks: 0,
    })
}
