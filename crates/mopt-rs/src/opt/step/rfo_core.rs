//! Rational-function-optimization kernels shared by the RS-RFO family.
//!
//! All functions work in the eigenbasis of the (image) Hessian: `eigvals` ascending and `g` the
//! gradient components in that basis. The algorithms (secular-equation solver, restricted-step
//! alpha search with Brent bracketing and safeguarded Newton refinement) follow MultiOptPy's
//! `Optimizer/rsirfo.py`.
//!
//! MultiOptPy: the secular equation and step-length search of `Optimizer/rsirfo.py:RSIRFO`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::math::linalg::col_norm;
use crate::math::roots::brentq;

/// Parameters of the restricted-step search.
#[derive(Debug, Clone, Copy)]
pub struct RfoParams {
    /// Initial scaling `alpha0`.
    pub alpha0: f64,
    /// Upper bound of `alpha`.
    pub alpha_max: f64,
    /// Maximum Newton change of `alpha` per micro-cycle.
    pub alpha_step_max: f64,
    /// Maximum Newton micro-cycles.
    pub max_micro_cycles: usize,
    /// Accepted `| ||step|| - radius |`.
    pub step_norm_tolerance: f64,
}

impl Default for RfoParams {
    fn default() -> Self {
        Self {
            alpha0: 1.0,
            alpha_max: 1000.0,
            alpha_step_max: 10.0,
            max_micro_cycles: 40,
            step_norm_tolerance: 1e-3,
        }
    }
}

fn safe(d: f64, eps: f64) -> f64 {
    if d.abs() < eps {
        if d == 0.0 {
            eps
        } else {
            d.signum() * eps
        }
    } else {
        d
    }
}

/// Safeguarded Newton/bisection for the smallest root of
/// `f(l) = l + sum_i g_i^2 / (lambda_i - l)` below the first pole.
fn solve_secular_safeguarded(lam: &[f64], g2: &[f64], pole: f64, initial: f64) -> f64 {
    let f = |l: f64| {
        l + lam
            .iter()
            .zip(g2)
            .map(|(li, gi)| gi / safe(li - l, 1e-30))
            .sum::<f64>()
    };
    let fp = |l: f64| {
        1.0 + lam
            .iter()
            .zip(g2)
            .map(|(li, gi)| gi / safe(li - l, 1e-30).powi(2))
            .sum::<f64>()
    };
    let mut b = pole;
    let mut a = initial;
    let mut f_a = f(a);
    let g_norm = libm::sqrt(g2.iter().sum());
    let mut limit = 10;
    while f_a > 0.0 && limit > 0 {
        a -= g_norm.max(a.abs() * 0.1).max(1e-8);
        f_a = f(a);
        limit -= 1;
    }
    if f_a > 0.0 {
        return initial;
    }
    let mut l = initial;
    if l <= a || l >= b {
        l = 0.5 * (a + b);
    }
    let tol = 1e-10 * pole.abs() + 1e-12;
    for _ in 0..250 {
        let fl = f(l);
        if fl.abs() < tol {
            return l;
        }
        let fpl = fp(l);
        let delta = if fpl.abs() > 1e-20 { -fl / fpl } else { 0.0 };
        let newton = l + delta;
        let next = if delta != 0.0 && newton > a && newton < b {
            newton
        } else {
            0.5 * (a + b)
        };
        if fl > 0.0 {
            b = l;
        } else {
            a = l;
        }
        l = next;
        if (b - a).abs() < tol {
            return 0.5 * (a + b);
        }
    }
    0.5 * (a + b)
}

/// Smallest root `nu` of `nu + sum_i g_i^2 / (lambda_i - alpha nu) = 0`, i.e. the lowest
/// eigenvalue of the augmented Hessian `[[H, g], [g^T, 0]]` with metric `diag(alpha I, 1)`.
///
/// In scaled variables this is `nu + sum_i (g_i^2 / alpha) / (lambda_i / alpha - nu)`.
/// MultiOptPy scales the squared gradient by `1 / alpha^2`, which makes the step independent of
/// `alpha` and disables the restricted-step search (see `docs/source/migration_optimizers.md`).
pub fn secular_lambda(eigvals: &[f64], g: &[f64], alpha: f64) -> f64 {
    let lam: Vec<f64> = eigvals.iter().map(|v| v / alpha).collect();
    let g2: Vec<f64> = g.iter().map(|v| v * v / alpha).collect();
    let pole = lam
        .iter()
        .zip(&g2)
        .find(|(_, gi)| **gi > NEGLIGIBLE_GRADIENT_SQ)
        .map(|(l, _)| *l);
    let Some(pole) = pole else {
        return lam.first().copied().unwrap_or(0.0);
    };
    let g_norm_sq: f64 = g2.iter().sum();
    let initial = 0.5 * (pole - libm::sqrt((pole * pole + 4.0 * g_norm_sq).max(0.0)));
    solve_secular_safeguarded(&lam, &g2, pole, initial)
}

/// Squared component of the (scaled) gradient below which a mode does not count in the
/// secular equation.
const NEGLIGIBLE_GRADIENT_SQ: f64 = 1e-20;

/// RFO step for a given `alpha`; returns `(step, lambda_aug)`.
///
/// With a gradient that is zero within rounding in every mode the secular equation has no
/// pole to search below. The lowest eigenvalue of the augmented Hessian is then zero (or
/// the lowest curvature, if that is negative), and the step is the Newton step along the
/// modes of positive curvature and nothing along the others. (Taking the lowest curvature
/// itself for the eigenvalue, as was done here before, divides the gradient along that
/// mode by the smallest number the solver allows, `1e-20`: a point at rest was sent away
/// by `1e10`. An optimization stops long before its gradient is that small; an image of
/// a chain that has converged while others still move does not.)
pub fn solve_rfo(eigvals: &[f64], g: &[f64], alpha: f64) -> (Col<f64>, f64) {
    if g.iter().all(|v| v * v / alpha <= NEGLIGIBLE_GRADIENT_SQ) {
        let step = Col::from_fn(eigvals.len(), |i| {
            if eigvals[i] > 0.0 {
                -g[i] / eigvals[i]
            } else {
                0.0
            }
        });
        let lowest = eigvals.iter().copied().fold(f64::INFINITY, f64::min);
        let nu = if lowest.is_finite() {
            (lowest / alpha).min(0.0)
        } else {
            0.0
        };
        return (step, nu);
    }
    let nu = secular_lambda(eigvals, g, alpha);
    let step = Col::from_fn(eigvals.len(), |i| {
        -(g[i] / alpha) / safe(eigvals[i] / alpha - nu, 1e-20)
    });
    (step, nu)
}

/// `d ||step||^2 / d alpha` treating `lambda_aug` as constant (MultiOptPy's approximation).
fn step_norm_sq_derivative(eigvals: &[f64], g: &[f64], alpha: f64, nu: f64) -> f64 {
    const MAX: f64 = 1e20;
    let sum: f64 = eigvals
        .iter()
        .zip(g)
        .map(|(l, gi)| {
            let mut d = l - nu * alpha;
            if d.abs() < 1e-8 {
                // numpy sign(0) = 0 leaves exact zeros untouched in the reference.
                d = d.signum() * if d == 0.0 { 0.0 } else { 1e-8 };
            }
            let d3 = d * d * d;
            if d3.abs() > 1e-10 {
                (gi * gi / d3).clamp(-MAX, MAX)
            } else {
                0.0
            }
        })
        .sum();
    let der = 2.0 * nu * sum;
    if !der.is_finite() || der.abs() > MAX {
        if der == 0.0 {
            1e-8
        } else {
            der.signum() * MAX
        }
    } else {
        der
    }
}

/// The step of the trust-region problem `min g.s + s.H.s / 2` with `||s|| <= radius`, in the
/// eigenbasis of `H` (`eigvals` ascending, `g` the gradient in that basis): the Newton step
/// where `H` is positive definite and the step fits, otherwise the shifted Newton step
/// `s_i = -g_i / (lambda_i + mu)` with the shift `mu > max(0, -lambda_min)` that puts it on the
/// radius, and in the hard case (too little gradient along the lowest mode to reach the radius
/// at any such shift) the shifted step at `mu = -lambda_min` plus the rest of the radius along
/// the lowest mode, downhill (J. J. Moré, D. C. Sorensen, SIAM J. Sci. Stat. Comput. 1983, 4,
/// 553). Its model never rises: `g.s + s.H.s / 2 < 0` unless the gradient is zero and the
/// curvature nowhere negative.
///
/// The RFO steps of every `alpha` lie on the same curve of shifted Newton steps (`mu = -alpha
/// nu`), but the search in `alpha` is bounded by `alpha_max`: with negative curvature and a
/// gradient that has almost nothing along that mode (a structure that converged at a saddle
/// point and was left along its mode by the mode kill, or a quasi-Newton Hessian that turned a
/// curvature negative), every RFO step up to `alpha_max` is far longer than the radius.
/// MultiOptPy then takes the steepest-descent step on the radius, which ignores the curvature
/// and raises the model wherever the gradient points along a stiff mode; on an enzyme on the
/// ONIOM surface it raised the energy for a hundred iterations after the mode kill. This step
/// is taken there instead.
///
/// MultiOptPy: the steepest-descent fallback of `Optimizer/rsirfo.py` (corrected).
pub fn trust_region_step(eigvals: &[f64], g: &[f64], radius: f64) -> Col<f64> {
    let n = eigvals.len().min(g.len());
    if n == 0 || radius.is_nan() || radius <= 0.0 {
        return Col::zeros(eigvals.len());
    }
    let lowest = eigvals[..n].iter().copied().fold(f64::INFINITY, f64::min);
    let g_norm = libm::sqrt(g[..n].iter().map(|v| v * v).sum());
    // The size of the shifted Newton step; infinite at a shift that leaves a mode with a
    // gradient without positive curvature.
    let norm_at = |mu: f64| {
        let mut sum = 0.0;
        for i in 0..n {
            if g[i] == 0.0 {
                continue;
            }
            let d = eigvals[i] + mu;
            if d <= 0.0 {
                return f64::INFINITY;
            }
            sum += (g[i] / d) * (g[i] / d);
        }
        libm::sqrt(sum)
    };
    let shifted = |mu: f64| {
        Col::from_fn(eigvals.len(), |i| {
            let d = if i < n { eigvals[i] + mu } else { 0.0 };
            if i < n && g[i] != 0.0 && d > 0.0 {
                -g[i] / d
            } else {
                0.0
            }
        })
    };
    if lowest > 0.0 && norm_at(0.0) <= radius {
        return shifted(0.0);
    }
    let floor = (-lowest).max(0.0);
    // The modes of the lowest curvature (within rounding) and the gradient along them.
    let span = eigvals[..n]
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    let degenerate = |i: usize| eigvals[i] - lowest <= 1e-12 * span;
    let along_lowest = libm::sqrt((0..n).filter(|&i| degenerate(i)).map(|i| g[i] * g[i]).sum());
    if lowest <= 0.0 && along_lowest <= 1e-10 * g_norm.max(f64::MIN_POSITIVE) {
        // The hard case where the rest of the step falls short of the radius at the floor.
        let mut rest = Col::from_fn(eigvals.len(), |i| {
            if i < n && !degenerate(i) && g[i] != 0.0 {
                -g[i] / (eigvals[i] + floor)
            } else {
                0.0
            }
        });
        let short = col_norm(&rest);
        if short <= radius && lowest < 0.0 {
            // The rest of the radius along the lowest mode, against its (tiny) gradient.
            let k = (0..n).find(|&i| degenerate(i)).unwrap_or(0);
            let sign = if g[k] > 0.0 { -1.0 } else { 1.0 };
            rest[k] = sign * libm::sqrt((radius * radius - short * short).max(0.0));
            return rest;
        }
        if short <= radius {
            // No negative curvature and the gradient is zero along the flat modes: the
            // shifted step at the floor (here zero) already fits.
            return rest;
        }
    }
    // Bisection of the shift between the floor (step too long, or infinite) and a shift at
    // which the step is surely within the radius (`||s|| <= ||g|| / (lambda_min + mu)`).
    let mut lo = floor;
    let mut hi = floor + g_norm / radius;
    if norm_at(hi) > radius {
        hi = floor + 2.0 * g_norm / radius + f64::MIN_POSITIVE;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if norm_at(mid) > radius {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-15 * hi.abs().max(1e-300) {
            break;
        }
    }
    // `hi` keeps the step within the radius.
    shifted(hi)
}

/// The value of the quadratic model `g.s + s.H.s / 2` of a step in the eigenbasis.
fn model_change(eigvals: &[f64], g: &[f64], s: &Col<f64>) -> f64 {
    (0..eigvals.len().min(g.len()).min(s.nrows()))
        .map(|i| g[i] * s[i] + 0.5 * eigvals[i] * s[i] * s[i])
        .sum()
}

/// Step restricted to the trust radius (`||step|| = radius`) by searching `alpha`; `None`
/// where the search finds no `alpha` that makes the step that short.
pub fn restricted_step(eigvals: &[f64], g: &[f64], radius: f64, p: &RfoParams) -> Option<Col<f64>> {
    let norm_at = |alpha: f64| col_norm(&solve_rfo(eigvals, g, alpha).0);
    let objective = |alpha: f64| norm_at(alpha).powi(2) - radius * radius;
    let (lo, hi) = (1e-6, p.alpha_max);
    let mut alpha = p.alpha0;
    if objective(lo) * objective(hi) < 0.0 {
        if let Ok(ab) = brentq(objective, lo, hi, 1e-6, 1e-6, 50) {
            let (step, _) = solve_rfo(eigvals, g, ab);
            if (col_norm(&step) - radius).abs() < p.step_norm_tolerance {
                return Some(step);
            }
            alpha = ab;
        }
    }
    newton_refinement(eigvals, g, radius, alpha, p)
}

fn newton_refinement(
    eigvals: &[f64],
    g: &[f64],
    radius: f64,
    mut alpha: f64,
    p: &RfoParams,
) -> Option<Col<f64>> {
    let mut best: Option<(f64, Col<f64>)> = None;
    let mut history: Vec<f64> = Vec::new();
    let (mut left, mut right): (Option<f64>, Option<f64>) = (None, None);
    for _ in 0..p.max_micro_cycles {
        let (step, nu) = solve_rfo(eigvals, g, alpha);
        let norm = col_norm(&step);
        let diff = (norm - radius).abs();
        if best.as_ref().is_none_or(|(d, _)| diff < *d) {
            best = Some((diff, step.clone()));
        }
        let obj = norm * norm - radius * radius;
        if obj < 0.0 && left.is_none_or(|l| alpha > l) {
            left = Some(alpha);
        } else if obj > 0.0 && right.is_none_or(|r| alpha < r) {
            right = Some(alpha);
        }
        if obj.abs() < 1e-8 || diff < p.step_norm_tolerance {
            return Some(step);
        }
        history.push(norm);
        let der = step_norm_sq_derivative(eigvals, g, alpha, nu);
        let new_alpha = if der.abs() < 1e-10 {
            match (left, right) {
                (Some(l), Some(r)) => 0.5 * (l + r),
                _ if obj > 0.0 => (alpha / 2.0).max(1e-6),
                _ => (alpha * 2.0).min(p.alpha_max),
            }
        } else {
            let raw = -obj / der;
            let mut na = alpha + raw.clamp(-p.alpha_step_max, p.alpha_step_max);
            if let (Some(l), Some(r)) = (left, right) {
                na = na.min(r * 0.99).max(l * 1.01);
            }
            na
        };
        alpha = new_alpha.max(1e-6).min(p.alpha_max);
        // Like the reference, an early exit returns the step of the current micro-cycle.
        if alpha == p.alpha_max || alpha == 1e-6 {
            return Some(step);
        }
        let k = history.len();
        if k >= 3
            && (history[k - 1] - history[k - 2]).abs() < 1e-6
            && (history[k - 2] - history[k - 3]).abs() < 1e-6
        {
            return Some(step);
        }
    }
    match best {
        Some((d, s)) if d < p.step_norm_tolerance * 1.1 => Some(s),
        _ => None,
    }
}

/// RS-RFO step: the unrestricted `alpha0` step if it fits the radius, otherwise the
/// restricted step; where the search in `alpha` ends with no finite step within the radius
/// (an early exit of the Newton refinement returns the step of its last micro-cycle, which
/// may be far longer) or one that would raise the model, the step of the trust-region
/// problem ([`trust_region_step`]).
pub fn rs_step(eigvals: &[f64], g: &[f64], radius: f64, p: &RfoParams) -> Col<f64> {
    let (step, _) = solve_rfo(eigvals, g, p.alpha0);
    let n = col_norm(&step);
    if n.is_finite() && n <= radius {
        return step;
    }
    match restricted_step(eigvals, g, radius, p) {
        Some(s)
            if s.iter().all(|v| v.is_finite())
                && col_norm(&s) <= radius + p.step_norm_tolerance
                && model_change(eigvals, g, &s) <= 0.0 =>
        {
            s
        }
        _ => trust_region_step(eigvals, g, radius),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfo_step_on_quadratic_approaches_newton_step_for_small_gradient() {
        let eig = [1.0, 2.0, 5.0];
        let g = [1e-4, -2e-4, 3e-4];
        let (s, nu) = solve_rfo(&eig, &g, 1.0);
        for i in 0..3 {
            assert!((s[i] + g[i] / eig[i]).abs() < 1e-9);
        }
        assert!(nu < 0.0 && nu > -1e-7);
    }

    #[test]
    fn a_gradient_that_is_zero_within_rounding_gives_no_long_step() {
        // The Newton step, also below the size at which a mode counts in the secular
        // equation (the step along the first mode used to be -g / 1e-20).
        let eig = [0.4, 1.0];
        for size in [1e-9, 1e-10, 1e-11, 1e-13, 0.0] {
            let g = [size, -0.5 * size];
            let (s, nu) = solve_rfo(&eig, &g, 1.0);
            assert!((s[0] + size / 0.4).abs() <= 1e-6 * size, "{size}: {s:?}");
            assert!((s[1] - 0.5 * size).abs() <= 1e-6 * size, "{size}: {s:?}");
            assert!(nu <= 0.0 && nu > -1e-15);
        }
        // Nothing along a mode without positive curvature: there is no gradient to say
        // which way.
        let (s, nu) = solve_rfo(&[-0.3, 0.0, 2.0], &[1e-12, 1e-12, 1e-12], 1.0);
        assert_eq!((s[0], s[1]), (0.0, 0.0));
        assert!((s[2] + 0.5e-12).abs() < 1e-24);
        assert_eq!(nu, -0.3);
        assert_eq!(solve_rfo(&[], &[], 1.0).0.nrows(), 0);
    }

    #[test]
    fn step_norm_decreases_with_alpha() {
        let eig = [0.1, 0.4, 2.0];
        let g = [0.3, -0.2, 0.5];
        let norms: Vec<f64> = [1e-3, 0.1, 1.0, 10.0, 1000.0]
            .iter()
            .map(|&a| col_norm(&solve_rfo(&eig, &g, a).0))
            .collect();
        assert!(norms.windows(2).all(|w| w[1] < w[0]), "{norms:?}");
        // alpha -> 0 recovers the Newton step -g / lambda.
        let (s, _) = solve_rfo(&eig, &g, 1e-9);
        assert!((s[0] + 3.0).abs() < 1e-5);
    }

    #[test]
    fn secular_root_satisfies_equation() {
        let eig = [-0.5, 0.3, 1.2];
        let g = [0.2, 0.1, -0.4];
        let nu = secular_lambda(&eig, &g, 1.0);
        let f: f64 = nu
            + eig
                .iter()
                .zip(&g)
                .map(|(l, gi)| gi * gi / (l - nu))
                .sum::<f64>();
        assert!(f.abs() < 1e-9);
        assert!(nu < -0.5);
    }

    #[test]
    fn negative_curvature_with_almost_no_gradient_along_it_stays_on_the_radius() {
        // The RFO steps up to alpha_max are thousands of Bohr long here (the gradient along
        // the negative mode is 4e-8); the step is the trust-region step: on the radius,
        // downhill along that mode, and the model falls.
        let eig = [-6.024e-4, 0.1225, 0.4302];
        let g = [-4.216e-8, 3.272e-8, -2.283e-7];
        let p = RfoParams::default();
        let s = rs_step(&eig, &g, 0.3125, &p);
        assert!((col_norm(&s) - 0.3125).abs() < 1e-9, "{s:?}");
        assert!(s[0] > 0.3, "{s:?}");
        assert!(model_change(&eig, &g, &s) < 0.0);
        // A stiff mode along the gradient: the steepest-descent step on the radius would
        // raise the model (by 0.005 here); the trust-region step lowers it.
        let eig = [-0.42, -0.092, 1.25];
        let g = [1.1e-5, 4.9e-4, 0.018];
        let s = rs_step(&eig, &g, 0.5, &p);
        let steepest = Col::from_fn(3, |i| {
            -g[i] * 0.5 / libm::sqrt(g.iter().map(|v| v * v).sum())
        });
        assert!(model_change(&eig, &g, &steepest) > 0.0);
        assert!(model_change(&eig, &g, &s) < 0.0);
        assert!(col_norm(&s) <= 0.5 + 1e-9);
    }

    #[test]
    fn the_trust_region_step_handles_the_hard_case_and_the_newton_step() {
        // No gradient along the negative mode: the rest of the radius goes along it.
        let s = trust_region_step(&[-0.2, 0.5, 1.0], &[0.0, 0.01, -0.02], 0.4);
        assert!((col_norm(&s) - 0.4).abs() < 1e-12, "{s:?}");
        assert!((s[1] + 0.01 / 0.7).abs() < 1e-12 && (s[2] - 0.02 / 1.2).abs() < 1e-12);
        assert!(s[0].abs() > 0.39);
        // Positive definite and short: the Newton step.
        let s = trust_region_step(&[0.5, 2.0], &[0.01, -0.02], 0.4);
        assert!((s[0] + 0.02).abs() < 1e-15 && (s[1] - 0.01).abs() < 1e-15);
        // Positive definite and long: on the radius, below the model of the steepest step.
        let s = trust_region_step(&[0.01, 2.0], &[0.01, -0.02], 0.1);
        assert!((col_norm(&s) - 0.1).abs() < 1e-12);
        assert!(model_change(&[0.01, 2.0], &[0.01, -0.02], &s) < 0.0);
        // No gradient, no negative curvature: no step.
        assert_eq!(
            col_norm(&trust_region_step(&[0.1, 1.0], &[0.0, 0.0], 0.3)),
            0.0
        );
        // No gradient at a saddle point: the radius along the negative mode.
        let s = trust_region_step(&[-0.1, 1.0], &[0.0, 0.0], 0.3);
        assert!((s[0].abs() - 0.3).abs() < 1e-15 && s[1] == 0.0);
    }

    #[test]
    fn restricted_step_has_trust_radius_norm() {
        let eig = [0.1, 0.4, 2.0];
        let g = [0.3, -0.2, 0.5];
        let p = RfoParams::default();
        let s = rs_step(&eig, &g, 0.2, &p);
        assert!((col_norm(&s) - 0.2).abs() < 1e-3);
        // Downhill.
        let dot: f64 = s.iter().zip(&g).map(|(a, b)| a * b).sum();
        assert!(dot < 0.0);
    }
}
