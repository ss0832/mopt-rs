//! The classical terms of GFN0-xTB, as in xtb and gfn0-rs (`src/repulsion.rs`, `src/srb.rs`,
//! `src/dispersion.rs`), with analytic gradients:
//!
//! - repulsion `Y_A Y_B exp(-alpha_AB r^1.5) / r` with `alpha_AB = sqrt(a_A a_B) (1 + (0.01 dEN^2
//!   + 0.01 dEN^4) renscale)` and a smooth cutoff between 35 and 40 Bohr;
//! - the short-range basis correction between different elements of B..F,
//!   `srbpre exp(-srbexp (1 + srbken dEN_P^2) (r - r0)^2)`, `r0` from CN-dependent radii;
//! - D4 dispersion with rational damping (GFN0's `a1`, `a2`, `s6`, `s8`), the covalent CN, and
//!   EEQ-2019 charges, smoothly cut off between 55 and 60 Bohr (`s9 = 0`: no three-body term).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::cn::smooth_cutoff;
use super::params::ElementParams;
use crate::parameters::gfn0_xtb as g;
use crate::pes::dispersion::d4::D4Model;

fn pair(x: &[[f64; 3]], i: usize, j: usize) -> ([f64; 3], f64) {
    let d = [x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]];
    (d, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
}

fn add_radial(grad: &mut [f64], i: usize, j: usize, d: [f64; 3], r: f64, de_dr: f64) {
    for k in 0..3 {
        let v = de_dr * d[k] / r;
        grad[3 * i + k] += v;
        grad[3 * j + k] -= v;
    }
}

/// Repulsion energy (gradient added to `grad`).
pub fn repulsion(el: &[ElementParams], x: &[[f64; 3]], grad: &mut [f64]) -> f64 {
    let mut e = 0.0;
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            let (d, r) = pair(x, i, j);
            if r >= 40.0 || r <= 1.0e-12 {
                continue;
            }
            let den = el[i].en - el[j].en;
            let den2 = den * den;
            let alpha = (el[i].repa * el[j].repa).max(1.0e-18).sqrt()
                * (1.0 + (0.01 * den2 + 0.01 * den2 * den2) * g::GFN0_RENSCALE);
            let y = el[i].repb * el[j].repb;
            let r15 = r.powf(1.5);
            let ex = (-alpha * r15).exp();
            let v = ex * y / r;
            let dv = v * (-1.5 * alpha * r.sqrt() - 1.0 / r);
            let (cut, dcut) = smooth_cutoff(r, 35.0, 40.0);
            e += v * cut;
            add_radial(grad, i, j, d, r, dv * cut + v * dcut);
        }
    }
    e
}

const SRB_CUTOFF: f64 = 14.142_135_623_730_95;

/// The row of the SRB polynomial `p(4, 2)`: xtb's `itr` of `approxrab.f90`, which takes the
/// fourth row for every element beyond argon (the fifth and sixth periods included).
fn period_row(z: u8) -> usize {
    match z {
        1..=2 => 0,
        3..=10 => 1,
        11..=18 => 2,
        _ => 3,
    }
}

/// Short-range basis correction (gradient added to `grad`, `dE/dCN` to `w_cn`).
pub fn srb(
    el: &[ElementParams],
    x: &[[f64; 3]],
    cn: &[f64],
    grad: &mut [f64],
    w_cn: &mut [f64],
) -> f64 {
    let mut e = 0.0;
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            let (zi, zj) = (el[i].z, el[j].z);
            if !(5..=9).contains(&zi) || !(5..=9).contains(&zj) || zi == zj {
                continue;
            }
            let (d, r) = pair(x, i, j);
            if r >= SRB_CUTOFF {
                continue;
            }
            let (si, sj) = (el[i].srb, el[j].srb);
            let (ri, rj) = (period_row(zi), period_row(zj));
            let den = (si[2] - sj[2]).abs();
            let k1 = 0.005 * (g::GFN0_SRB_P[ri][0] + g::GFN0_SRB_P[rj][0]);
            let k2 = 0.005 * (g::GFN0_SRB_P[ri][1] + g::GFN0_SRB_P[rj][1]);
            let ff = 1.0 - k1 * den - k2 * den * den;
            let ra = si[0] + si[1] * cn[i] + g::GFN0_SRBSHIFT;
            let rb = sj[0] + sj[1] * cn[j] + g::GFN0_SRBSHIFT;
            let r0 = (ra + rb) * ff;
            let pen = si[3] - sj[3];
            let expo = g::GFN0_SRBEXP * (1.0 + g::GFN0_SRBKEN * pen * pen);
            let v = g::GFN0_SRBPRE * (-expo * (r - r0) * (r - r0)).exp();
            e += v;
            let dv_dr = -2.0 * expo * (r - r0) * v;
            add_radial(grad, i, j, d, r, dv_dr);
            // d r0 / d CN_i = ff cnfak_i.
            w_cn[i] -= dv_dr * ff * si[1];
            w_cn[j] -= dv_dr * ff * sj[1];
        }
    }
    e
}

/// D4 two-body energy; the gradient is added to `grad`, `dE/dCN^cov` to `w_cov`, `dE/dq` to
/// `w_q`.
pub fn dispersion(
    model: &D4Model,
    x: &[[f64; 3]],
    cn_cov: &[f64],
    q: &[f64],
    grad: &mut [f64],
    w_cov: &mut [f64],
    w_q: &mut [f64],
) -> f64 {
    let n = x.len();
    let c6 = model.c6(cn_cov, q);
    let mut e = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            let (d, r) = pair(x, i, j);
            if r >= 60.0 || r <= 1.0e-12 {
                continue;
            }
            let c8_scale = 3.0 * model.q4(i) * model.q4(j);
            let r0 = g::GFN0_A1 * c8_scale.max(0.0).sqrt() + g::GFN0_A2;
            let (r2, r02) = (r * r, r0 * r0);
            let (r6, r8) = (r2 * r2 * r2, r2 * r2 * r2 * r2);
            let (r06, r08) = (r02 * r02 * r02, r02 * r02 * r02 * r02);
            let (t6, t8) = (1.0 / (r6 + r06), 1.0 / (r8 + r08));
            // E_ij = -C6 f(r) cut(r) with f = s6 t6 + s8 c8_scale t8.
            let f = g::GFN0_S6 * t6 + g::GFN0_S8 * c8_scale * t8;
            let df = -6.0 * g::GFN0_S6 * r.powi(5) * t6 * t6
                - 8.0 * g::GFN0_S8 * c8_scale * r.powi(7) * t8 * t8;
            let (cut, dcut) = smooth_cutoff(r, 55.0, 60.0);
            let c = c6[i * n + j];
            let fc = f * cut;
            e -= c[0] * fc;
            w_cov[i] -= c[1] * fc;
            w_cov[j] -= c[2] * fc;
            w_q[i] -= c[3] * fc;
            w_q[j] -= c[4] * fc;
            add_radial(grad, i, j, d, r, -c[0] * (df * cut + f * dcut));
        }
    }
    e
}
