//! Exact solution of the cone-constrained step problem of CCQN (eqn (9) of Wu and Wang 2025):
//!
//! `min g^T s + s^T B s / 2` subject to `||s|| = Delta` and `e^T s >= ||s|| cos(phi)`.
//!
//! The feasible set is a spherical cap. Its minimizer is either a local minimizer of the
//! quadratic on the whole sphere that lies inside the cap, or the minimizer on the cap's
//! boundary circle `{s = Delta cos(phi) e + w : w perpendicular to e, ||w|| = Delta sin(phi)}`,
//! which is the same sphere problem in the complement of `e`. On a sphere the quadratic has its
//! global minimizer at `s = -(B - l I)^-1 g` with `l <= lambda_1` (Moré and Sorensen 1983, with
//! the "hard case" when `g` has no component along the lowest eigenvectors) and at most one other
//! local minimizer, with `lambda_1 < l < lambda_2` (Martínez 1994). Both are found from the
//! secular equation `||s(l)|| = Delta` in the eigenbasis of `B`, so the cap problem is solved
//! globally with two eigendecompositions. The paper solves eqn (9) with an augmented Lagrangian
//! method (a local solution); see [`super::ccqn::AlmSettings`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (part of CCQN).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen, symmetrize};

/// Eigenvalues closer than this (relative) are treated as one.
const DEGENERATE: f64 = 1e-10;

/// `||s(l)||` with `s_i = -g_i / (lam_i - l)`.
fn norm_at(lam: &[f64], g: &[f64], l: f64) -> f64 {
    lam.iter()
        .zip(g)
        .map(|(li, gi)| (gi / (li - l)).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn step_at(lam: &[f64], g: &[f64], l: f64) -> Vec<f64> {
    lam.iter().zip(g).map(|(li, gi)| -gi / (li - l)).collect()
}

/// Root of `||s(l)|| = r` in `(lo, hi)` where `||s||` is monotone, by bisection.
fn bisect(lam: &[f64], g: &[f64], r: f64, mut lo: f64, mut hi: f64, increasing: bool) -> f64 {
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        let above = norm_at(lam, g, mid) > r;
        if above == increasing {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Local minimizers of `g^T s + s^T diag(lam) s / 2` on the sphere `||s|| = r` (components in
/// the eigenbasis, `lam` ascending): the global one and, when it exists, the local non-global
/// one (both roots of the secular equation between the two lowest eigenvalues are returned; the
/// caller compares objective values).
pub fn sphere_minimizers(lam: &[f64], g: &[f64], r: f64) -> Vec<Vec<f64>> {
    let n = lam.len();
    if n == 0 {
        return Vec::new();
    }
    let scale = lam.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let tol = DEGENERATE * scale;
    let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    let l1 = lam[0];
    let low: Vec<usize> = (0..n).filter(|&i| lam[i] - l1 <= tol).collect();
    let g_low = low.iter().map(|&i| g[i] * g[i]).sum::<f64>().sqrt();
    let mut out = Vec::new();

    // Global minimizer: l < lambda_1, ||s|| increasing in l.
    if g_low > 1e-14 * g_norm.max(1e-300) {
        let lo = l1 - g_norm / r - tol;
        out.push(step_at(lam, g, bisect(lam, g, r, lo, l1, true)));
    } else {
        // Hard case: no gradient along the lowest eigenspace.
        let s0: Vec<f64> = (0..n)
            .map(|i| {
                if low.contains(&i) {
                    0.0
                } else {
                    -g[i] / (lam[i] - l1)
                }
            })
            .collect();
        let n0 = s0.iter().map(|v| v * v).sum::<f64>().sqrt();
        if n0 <= r {
            let tau = (r * r - n0 * n0).sqrt();
            for sign in [1.0, -1.0] {
                let mut s = s0.clone();
                s[low[0]] = sign * tau;
                out.push(s);
            }
        } else {
            let lo = l1 - g_norm / r - tol;
            out.push(step_at(lam, g, bisect(lam, g, r, lo, l1 - tol, true)));
        }
    }

    // Local non-global minimizer between lambda_1 and the next distinct eigenvalue.
    if let Some(&i2) = (0..n).find(|&i| lam[i] - l1 > tol).as_ref() {
        let l2 = lam[i2];
        let next: Vec<usize> = (0..n).filter(|&i| (lam[i] - l2).abs() <= tol).collect();
        let g_next = next.iter().map(|&i| g[i] * g[i]).sum::<f64>();
        if g_low > 0.0 && g_next > 0.0 {
            // ||s||^2 is convex on (l1, l2): golden-section search for its minimum.
            let (mut a, mut b) = (l1, l2);
            let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
            for _ in 0..200 {
                let c = b - phi * (b - a);
                let d = a + phi * (b - a);
                if norm_at(lam, g, c) < norm_at(lam, g, d) {
                    b = d;
                } else {
                    a = c;
                }
            }
            let m = 0.5 * (a + b);
            if m > l1 && m < l2 && norm_at(lam, g, m) < r {
                out.push(step_at(lam, g, bisect(lam, g, r, l1, m, false)));
                out.push(step_at(lam, g, bisect(lam, g, r, m, l2, true)));
            }
        }
    }
    out.retain(|s| s.iter().all(|v| v.is_finite()));
    out
}

/// Local minimizers on the sphere `||s|| = r` of `g^T s + s^T B s / 2` in the full space.
fn minimizers_full(g: &Col<f64>, b: &Mat<f64>, r: f64) -> MoptResult<Vec<Col<f64>>> {
    let (vals, vecs) = symmetric_eigen(b)?;
    let n = g.nrows();
    let lam: Vec<f64> = (0..n).map(|k| vals[k]).collect();
    let gt: Vec<f64> = (0..n)
        .map(|k| (0..n).map(|i| vecs[(i, k)] * g[i]).sum())
        .collect();
    Ok(sphere_minimizers(&lam, &gt, r)
        .into_iter()
        .map(|st| Col::from_fn(n, |i| (0..n).map(|k| vecs[(i, k)] * st[k]).sum()))
        .collect())
}

/// `M` such that adding `M v v^T` moves `v` above the spectrum of `b`.
fn penalty(b: &Mat<f64>) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..b.nrows() {
        let row: f64 = (0..b.ncols()).map(|j| b[(i, j)].abs()).sum();
        m = m.max(row);
    }
    10.0 * (m + 1.0)
}

fn add_outer(b: &mut Mat<f64>, v: &Col<f64>, w: f64) {
    for i in 0..v.nrows() {
        for j in 0..v.nrows() {
            b[(i, j)] += w * v[i] * v[j];
        }
    }
}

/// Global solution of eqn (9). `excluded` holds orthonormal directions the step must not use
/// (rigid-body modes); `g` and `e` must be orthogonal to them.
pub fn exact_cone_step(
    g: &Col<f64>,
    b: &Mat<f64>,
    e: &Col<f64>,
    radius: f64,
    cos_phi: f64,
    excluded: Option<&Mat<f64>>,
) -> MoptResult<Col<f64>> {
    if !(radius > 0.0 && cos_phi > 0.0 && cos_phi < 1.0) {
        return Err(MoptError::config(format!(
            "CCQN needs a positive radius and 0 < cos(phi) < 1, got {radius} and {cos_phi}"
        )));
    }
    let n = g.nrows();
    let mut bb = b.clone();
    symmetrize(&mut bb);
    let m = penalty(&bb);
    if let Some(d) = excluded {
        for k in 0..d.ncols() {
            add_outer(&mut bb, &d.col(k).to_owned(), m);
        }
    }
    let f = |s: &Col<f64>| col_dot(g, s) + 0.5 * col_dot(s, &(b * s));
    let mut best: Option<(f64, Col<f64>)> = None;
    let mut consider = |s: Col<f64>| {
        let v = f(&s);
        if v.is_finite() && best.as_ref().is_none_or(|(bv, _)| v < *bv) {
            best = Some((v, s));
        }
    };

    // Interior of the cap: sphere minimizers that satisfy the cone constraint.
    for s in minimizers_full(g, &bb, radius)? {
        if col_dot(e, &s) >= radius * cos_phi * (1.0 - 1e-12) {
            consider(s);
        }
    }

    // Boundary circle: s = c e + w, w orthogonal to e, ||w|| = Delta sin(phi).
    let c = radius * cos_phi;
    let rw = radius * (1.0 - cos_phi * cos_phi).sqrt();
    let project = |v: &Col<f64>| v - e * col_dot(e, v);
    let gw = project(&(g + &bb * e * c));
    let mut bw = bb.clone();
    add_outer(&mut bw, e, m);
    // Restrict to the complement of e: P B P + M e e^T.
    let pbp = {
        let mut out = Mat::zeros(n, n);
        for j in 0..n {
            let col = project(&bw.col(j).to_owned());
            out.col_mut(j).copy_from(&col);
        }
        let mut t = Mat::zeros(n, n);
        for i in 0..n {
            let row = project(&out.row(i).transpose().to_owned());
            t.row_mut(i).copy_from(row.transpose());
        }
        add_outer(&mut t, e, m);
        t
    };
    for w in minimizers_full(&gw, &pbp, rw)? {
        let w = project(&w);
        let wn = col_norm(&w);
        if wn > 0.0 {
            consider(e * c + w * (rw / wn));
        }
    }
    best.map(|(_, s)| s)
        .ok_or_else(|| MoptError::linalg("the cone-constrained step problem has no solution"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::core_infra::rng::Rng;

    #[test]
    fn sphere_global_minimizer_and_hard_case() {
        let lam = [-1.0, 0.5, 2.0];
        let g = [0.3, -0.2, 0.1];
        let r = 0.7;
        let all = sphere_minimizers(&lam, &g, r);
        let f = |s: &Vec<f64>| {
            (0..3)
                .map(|i| g[i] * s[i] + 0.5 * lam[i] * s[i] * s[i])
                .sum::<f64>()
        };
        let best = all.iter().map(f).fold(f64::INFINITY, f64::min);
        // Dense scan of the sphere (spherical coordinates).
        let mut scan = f64::INFINITY;
        for i in 0..=400 {
            for j in 0..=800 {
                let th = std::f64::consts::PI * i as f64 / 400.0;
                let ph = 2.0 * std::f64::consts::PI * j as f64 / 800.0;
                let s = vec![
                    r * th.sin() * ph.cos(),
                    r * th.sin() * ph.sin(),
                    r * th.cos(),
                ];
                scan = scan.min(f(&s));
            }
        }
        assert!(
            best <= scan + 1e-12 && scan - best < 1e-5,
            "{best} vs {scan}"
        );
        // Hard case: no gradient along the lowest mode.
        let h = sphere_minimizers(&[-1.0, 1.0], &[0.0, 0.1], 1.0);
        assert!(h
            .iter()
            .all(|s| ((s[0] * s[0] + s[1] * s[1]).sqrt() - 1.0).abs() < 1e-12));
    }

    #[test]
    fn exact_cone_step_matches_a_dense_scan_in_3d() {
        let mut rng = Rng::from_seed(7);
        for _ in 0..20 {
            let a = Mat::from_fn(3, 3, |_, _| rng.next_f64() - 0.5);
            let b = &a + a.transpose();
            let g = Col::from_fn(3, |_| 0.2 * (rng.next_f64() - 0.5));
            let e = {
                let v = Col::from_fn(3, |_| rng.next_f64() - 0.5);
                &v * (1.0 / col_norm(&v))
            };
            let (r, cphi) = (0.3, 0.4 + 0.5 * rng.next_f64());
            let s = exact_cone_step(&g, &b, &e, r, cphi, None).unwrap();
            assert!((col_norm(&s) - r).abs() < 1e-10);
            assert!(col_dot(&e, &s) >= r * cphi - 1e-10);
            let f = |v: &Col<f64>| col_dot(&g, v) + 0.5 * col_dot(v, &(&b * v));
            // Scan the cap s = r (cos t e + sin t (cos p u + sin p w)), t <= phi, then refine
            // around the best grid point.
            let u = {
                let t = if e[0].abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let t = Col::from_fn(3, |k| t[k]);
                let v = &t - &e * col_dot(&e, &t);
                &v * (1.0 / col_norm(&v))
            };
            let w = Col::from_fn(3, |k| {
                let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                e[i] * u[j] - e[j] * u[i]
            });
            let phi = cphi.acos();
            let at = |t: f64, p: f64| (&e * t.cos() + (&u * p.cos() + &w * p.sin()) * t.sin()) * r;
            let (nt, np) = (300, 600);
            let (dt, dp) = (phi / nt as f64, 2.0 * std::f64::consts::PI / np as f64);
            let mut best = (f64::INFINITY, 0.0, 0.0);
            for i in 0..=nt {
                for j in 0..np {
                    let (t, p) = (i as f64 * dt, j as f64 * dp);
                    let v = f(&at(t, p));
                    if v < best.0 {
                        best = (v, t, p);
                    }
                }
            }
            let (_, t0, p0) = best;
            for i in -100..=100 {
                for j in -100..=100 {
                    let t = (t0 + i as f64 * dt / 100.0).clamp(0.0, phi);
                    let p = p0 + j as f64 * dp / 100.0;
                    best.0 = best.0.min(f(&at(t, p)));
                }
            }
            let scan = best.0;
            assert!(
                f(&s) <= scan + 1e-12 && scan - f(&s) < 1e-7,
                "{} vs {scan}",
                f(&s)
            );
        }
    }
}
