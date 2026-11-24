//! MultiOptPy's Fischer model Hessian with its D3 term (`ModelHessian/fischerd3.py`),
//! reproduced as it is.
//!
//! Bonds, angles (skipped within 0.8 degrees of linear), and dihedrals (scaled by
//! `sin^2 theta_1 sin^2 theta_2`, with the bond count from the 1.1 x covalent-radius
//! connectivity) as in Fischer & Almlof, plus a pair dispersion curvature for non-bonded pairs:
//! D2 `C6` coefficients scaled by `clip(1 - 0.05 (CN - CN_ref), 0.75, 1.25)` (the coordination
//! numbers include the atom itself, a side effect of MultiOptPy's infinite diagonal),
//! `C8 = 3 C6 sqrt(r4r2_i r4r2_j)`, Becke-Johnson-like damping with `a1 R0 + a2` (`a2 + 2` for
//! `C8`) on the D2 radii, and the exact radial and tangential curvatures of the damped pair
//! energy. The result is symmetrized, projected, and replaced by the unprojected or the unit
//! matrix when not finite.
//!
//! MultiOptPy: `ModelHessian/fischerd3.py:FischerD3ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{
    bend2, coords, cov_radius, dot, norm, project_tr_rot, sub, torsion2, Connectivity, V3,
};
use super::swartd2::{mop_d2_c6, mop_d2_vdw};
use crate::domain::{Element, MolecularSystem};
use crate::parameters::multioptpy_tables::{MOP_D3_DEFAULT_R4R2, MOP_D3_R4R2};

pub(crate) const S6: f64 = 1.0;
pub(crate) const S8: f64 = 0.7875;
pub(crate) const A1: f64 = 0.4289;
pub(crate) const A2: f64 = 4.4407;

/// `ref_cn_map` (H..Xe; others 4).
pub(crate) fn ref_cn(el: Element) -> f64 {
    const MAP: [f64; 55] = [
        4.0, 1.0, 0.0, 4.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 4.0, 5.0, 6.0, 1.0,
        0.0, 8.0, 6.0, 12.0, 12.0, 12.0, 6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 2.0,
        1.0, 0.0, 8.0, 6.0, 12.0, 12.0, 12.0, 6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 4.0, 6.0, 4.0, 3.0,
        2.0, 1.0, 0.0,
    ];
    let z = el.atomic_number() as usize;
    if (1..MAP.len()).contains(&z) {
        MAP[z]
    } else {
        4.0
    }
}

pub(crate) fn d3_r4r2(el: Element) -> f64 {
    match MOP_D3_R4R2.get(el.atomic_number() as usize) {
        Some(v) if !v.is_nan() => *v,
        _ => MOP_D3_DEFAULT_R4R2,
    }
}

/// `calc_coordination_numbers` (each atom counts itself once).
pub(crate) fn coordination_numbers(elements: &[Element], x: &[V3]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let term = if i == j {
                        -100.0
                    } else {
                        let rc = cov_radius(elements[i]) + cov_radius(elements[j]);
                        (-16.0 * (4.0 / 3.0 * (norm(sub(x[i], x[j])) / rc) - 1.0))
                            .clamp(-100.0, 100.0)
                    };
                    1.0 / (1.0 + term.exp())
                })
                .sum()
        })
        .collect()
}

fn add_block(h: &mut Mat<f64>, i: usize, j: usize, b: &[[f64; 3]; 3], sign: f64) {
    for p in 0..3 {
        for q in 0..3 {
            h[(3 * i + p, 3 * j + q)] += sign * b[p][q];
        }
    }
}

fn add_outer_blocks(h: &mut Mat<f64>, atoms: &[usize], b: &[V3], fc: f64) {
    for (m, &a) in atoms.iter().enumerate() {
        for (nn, &c) in atoms.iter().enumerate() {
            let mut blk = [[0.0; 3]; 3];
            for p in 0..3 {
                for q in 0..3 {
                    blk[p][q] = fc * b[m][p] * b[nn][q];
                }
            }
            add_block(h, a, c, &blk, 1.0);
        }
    }
}

/// Bonds, angles, and dihedrals of `fischerd3.py` (shared with `fischerd4.py`).
pub(crate) fn fischer_terms(h: &mut Mat<f64>, el: &[Element], x: &[V3], bc: &Connectivity) {
    let rc = |i: usize, j: usize| cov_radius(el[i]) + cov_radius(el[j]);
    for [i, j] in bc.bonds() {
        let rv = sub(x[i], x[j]);
        let r = norm(rv);
        if r < 0.1 {
            continue;
        }
        let fc = 0.3601 * (-1.944 * (r - rc(i, j))).exp();
        let u = [rv[0] / r, rv[1] / r, rv[2] / r];
        let mut blk = [[0.0; 3]; 3];
        for p in 0..3 {
            for q in 0..3 {
                blk[p][q] = fc * u[p] * u[q];
            }
        }
        add_block(h, i, i, &blk, 1.0);
        add_block(h, j, j, &blk, 1.0);
        add_block(h, i, j, &blk, -1.0);
        add_block(h, j, i, &blk, -1.0);
    }
    for [i, j, k] in bc.angles() {
        let (v1, v2) = (sub(x[i], x[j]), sub(x[k], x[j]));
        let (r1, r2) = (norm(v1), norm(v2));
        if r1 < 0.1 || r2 < 0.1 {
            continue;
        }
        if (dot(v1, v2) / (r1 * r2)).abs() > 0.9999 {
            continue;
        }
        let (c1, c2) = (rc(i, j), rc(j, k));
        let val = c1 * c2;
        let fc = if val.abs() < 1e-10 {
            0.0
        } else {
            0.089 + 0.11 / val.powf(-0.42) * (-0.44 * (r1 + r2 - c1 - c2)).exp()
        };
        let (_, b) = bend2(x[i], x[j], x[k]);
        add_outer_blocks(h, &[i, j, k], &b, fc);
    }
    let neighbors: Vec<f64> = bc
        .bonded
        .iter()
        .map(|row| row.iter().filter(|&&v| v).count() as f64)
        .collect();
    for [i, j, k, l] in bc.dihedrals() {
        let (vji, vjk, vkl) = (sub(x[i], x[j]), sub(x[k], x[j]), sub(x[l], x[k]));
        let (nji, njk, nkl) = (norm(vji), norm(vjk), norm(vkl));
        if nji.min(njk).min(nkl) < 0.1 {
            continue;
        }
        let cos1 = dot(vji, vjk) / (nji * njk);
        let cos2 = dot([-vjk[0], -vjk[1], -vjk[2]], vkl) / (njk * nkl);
        let s1 = 1.0 - (cos1 * cos1).min(1.0);
        let s2 = 1.0 - (cos2 * cos2).min(1.0);
        if s1 < 1e-3 || s2 < 1e-3 {
            continue;
        }
        let bond_sum = neighbors[j] + neighbors[k] - 2.0;
        let c = rc(j, k);
        let val = njk * c;
        let mut fc = if val.abs() < 1e-10 {
            0.0
        } else {
            0.0015 + 14.0 * bond_sum.max(0.0).powf(0.57) / val.powf(4.0) * (-2.85 * (njk - c)).exp()
        };
        fc *= s1 * s2;
        let (_, b) = torsion2([x[i], x[j], x[k], x[l]]);
        if !b.iter().flatten().all(|v| v.is_finite()) {
            continue;
        }
        add_outer_blocks(h, &[i, j, k, l], &b, fc);
    }
}

/// The damped pair curvature block `h_par u u^T + h_perp (1 - u u^T)` of `fischerd3.py`.
pub(crate) fn damped_pair_block(
    rv: V3,
    c6: f64,
    c8: f64,
    r0: f64,
    s6: f64,
    s8: f64,
    a1: f64,
    a2: f64,
) -> [[f64; 3]; 3] {
    let r = norm(rv);
    let (a26, a28) = (a2, a2 + 2.0);
    let den6 = r.powi(6) + (a1 * r0 + a26).powi(6);
    let den8 = r.powi(8) + (a1 * r0 + a28).powi(8);
    let f6 = r.powi(6) / den6;
    let f8 = r.powi(8) / den8;
    let df6 = 6.0 * r.powi(5) / den6 - 6.0 * r.powi(12) / (den6 * den6);
    let df8 = 8.0 * r.powi(7) / den8 - 8.0 * r.powi(16) / (den8 * den8);
    let g6 = -s6 * c6 * ((-6.0 / r.powi(7)) * f6 + (1.0 / r.powi(6)) * df6);
    let g8 = -s8 * c8 * ((-8.0 / r.powi(9)) * f8 + (1.0 / r.powi(8)) * df8);
    let h6 = s6 * c6 / r.powi(8) * (42.0 * f6 - r * df6);
    let h8 = s8 * c8 / r.powi(10) * (72.0 * f8 - r * df8);
    let (hp, ht) = (h6 + h8, (g6 + g8) / r);
    let u = [rv[0] / r, rv[1] / r, rv[2] / r];
    let mut b = [[0.0; 3]; 3];
    for p in 0..3 {
        for q in 0..3 {
            let pr = u[p] * u[q];
            let id = if p == q { 1.0 } else { 0.0 };
            b[p][q] = hp * pr + ht * (id - pr);
        }
    }
    b
}

/// Non-bonded pairs `i > j` (row-major, as `np.where` of the lower triangle) farther than 0.1.
pub(crate) fn nonbonded_pairs(x: &[V3], bc: &Connectivity) -> Vec<(usize, usize)> {
    let n = x.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..i {
            if !bc.bonded[i][j] && norm(sub(x[i], x[j])) > 0.1 {
                out.push((i, j));
            }
        }
    }
    out
}

pub(crate) fn add_pair_block(h: &mut Mat<f64>, i: usize, j: usize, b: &[[f64; 3]; 3]) {
    add_block(h, i, i, b, 1.0);
    add_block(h, j, j, b, 1.0);
    add_block(h, i, j, b, -1.0);
    add_block(h, j, i, b, -1.0);
}

/// Symmetrize, project, and fall back as `fischerd3.main` does.
pub(crate) fn finish(mut h: Mat<f64>, x: &[V3]) -> Mat<f64> {
    let n = h.nrows();
    let t = h.transpose().to_owned();
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] = (h[(i, j)] + t[(i, j)]) / 2.0;
        }
    }
    let finite = |m: &Mat<f64>| (0..n).all(|i| (0..n).all(|j| m[(i, j)].is_finite()));
    let p = project_tr_rot(&h, x);
    if finite(&p) {
        p
    } else if finite(&h) {
        h
    } else {
        Mat::identity(n, n)
    }
}

/// MultiOptPy's `FischerD3ApproxHessian.main`.
pub(crate) fn fischerd3(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let el = &sys.elements;
    let n = x.len();
    let bc = Connectivity::new(el, &x);
    let cn = coordination_numbers(el, &x);
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    fischer_terms(&mut h, el, &x, &bc);
    let vdw = |a: usize| {
        let v = mop_d2_vdw(el[a]);
        if v.is_nan() {
            cov_radius(el[a]) * 1.5
        } else {
            v
        }
    };
    for (i, j) in nonbonded_pairs(&x, &bc) {
        let si = (1.0 - 0.05 * (cn[i] - ref_cn(el[i]))).clamp(0.75, 1.25);
        let sj = (1.0 - 0.05 * (cn[j] - ref_cn(el[j]))).clamp(0.75, 1.25);
        let c6 = ((mop_d2_c6(el[i]) * si) * (mop_d2_c6(el[j]) * sj)).sqrt();
        let c8 = 3.0 * c6 * (d3_r4r2(el[i]) * d3_r4r2(el[j])).sqrt();
        let b = damped_pair_block(sub(x[i], x[j]), c6, c8, vdw(i) + vdw(j), S6, S8, A1, A2);
        add_pair_block(&mut h, i, j, &b);
    }
    finish(h, &x)
}
