//! MultiOptPy's older Fischer + D3 model Hessian (`ModelHessian/fischerd3old.py`, the default
//! of `-modelhess` without a name), reproduced as it is: pysisyphus's form of the Fischer model
//! (dihedrals scaled by `sin^2 theta_1 sin^2 theta_2`, skipped below `1e-4`, bond counts from the
//! 1.3 x covalent-radius connectivity) and the damped pair curvature of `fischerd3` with plain D2
//! `C6` coefficients (no coordination-number scaling) for non-bonded pairs.
//!
//! MultiOptPy: `ModelHessian/fischerd3old.py:FischerD3ApproxHessianOld`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{
    add_outer, bend2, coords, cov_radius, dot, norm, project_tr_rot, stretch2, sub, torsion2,
    Connectivity, V3,
};
use super::fischerd3::{add_pair_block, d3_r4r2, damped_pair_block, A1, A2, S6, S8};
use super::swartd2::{mop_d2_c6, mop_d2_vdw};
use crate::domain::{Element, MolecularSystem};

fn bonded_13(el: &[Element], x: &[V3]) -> Vec<Vec<bool>> {
    let n = x.len();
    let mut b = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                b[i][j] = norm(sub(x[i], x[j])) <= (cov_radius(el[i]) + cov_radius(el[j])) * 1.3;
            }
        }
    }
    b
}

/// MultiOptPy's `FischerD3ApproxHessianOld.main`.
pub(crate) fn fischerd3old(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let el = &sys.elements;
    let n = x.len();
    let rc = |i: usize, j: usize| cov_radius(el[i]) + cov_radius(el[j]);
    let bc = Connectivity::new(el, &x);
    let b13 = bonded_13(el, &x);
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    for [i, j] in bc.bonds() {
        let r = norm(sub(x[i], x[j]));
        let fc = 0.3601 * (-1.944 * (r - rc(i, j))).exp();
        let (_, b) = stretch2(x[i], x[j]);
        add_outer(&mut h, &[i, j], &b, fc);
    }
    for [i, j, k] in bc.angles() {
        let (v1, v2) = (sub(x[i], x[j]), sub(x[k], x[j]));
        let (r1, r2) = (norm(v1), norm(v2));
        if r1 < 0.1 || r2 < 0.1 || (dot(v1, v2) / (r1 * r2)).abs() > 0.9999 {
            continue;
        }
        let (c1, c2) = (rc(i, j), rc(j, k));
        let val = c1 * c2;
        let fc = if val.abs() < 1.0e-10 {
            0.0
        } else {
            0.089 + 0.11 / val.powf(-0.42) * (-0.44 * (r1 + r2 - c1 - c2)).exp()
        };
        let (_, b) = bend2(x[i], x[j], x[k]);
        add_outer(&mut h, &[i, j, k], &b, fc);
    }
    let count = |a: usize| b13[a].iter().filter(|&&v| v).count() as f64;
    for [i, j, k, l] in bc.dihedrals() {
        let (vji, vjk, vkl) = (sub(x[i], x[j]), sub(x[k], x[j]), sub(x[l], x[k]));
        let r_jk = norm(vjk);
        let c = rc(j, k);
        let bond_sum = count(j) + count(k) - 2.0;
        let val = r_jk * c;
        let mut fc = if val.abs() < 1.0e-10 {
            0.0
        } else {
            0.0015
                + 14.0 * bond_sum.max(0.0).powf(0.57) / val.powf(4.0) * (-2.85 * (r_jk - c)).exp()
        };
        let n_ji = norm(vji);
        if n_ji < 1e-8 || r_jk < 1e-8 {
            continue;
        }
        let cos1 = dot(vji, vjk) / (n_ji * r_jk);
        let n_kl = norm(vkl);
        if n_kl < 1e-8 {
            continue;
        }
        let cos2 = dot([-vjk[0], -vjk[1], -vjk[2]], vkl) / (r_jk * n_kl);
        let s1 = 1.0 - (cos1 * cos1).min(1.0);
        let s2 = 1.0 - (cos2 * cos2).min(1.0);
        if s1 < 1e-4 || s2 < 1e-4 {
            continue;
        }
        fc *= s1 * s2;
        let (_, b) = torsion2([x[i], x[j], x[k], x[l]]);
        add_outer(&mut h, &[i, j, k, l], &b, fc);
    }
    let vdw = |a: usize| {
        let v = mop_d2_vdw(el[a]);
        if v.is_nan() {
            cov_radius(el[a]) * 1.5
        } else {
            v
        }
    };
    for i in 0..n {
        for j in 0..i {
            if b13[i][j] {
                continue;
            }
            let rv = sub(x[i], x[j]);
            if norm(rv) < 0.1 {
                continue;
            }
            let c6 = (mop_d2_c6(el[i]) * mop_d2_c6(el[j])).sqrt();
            let c8 = 3.0 * c6 * (d3_r4r2(el[i]) * d3_r4r2(el[j])).sqrt();
            let b = damped_pair_block(rv, c6, c8, vdw(i) + vdw(j), S6, S8, A1, A2);
            add_pair_block(&mut h, i, j, &b);
        }
    }
    let t = h.transpose().to_owned();
    for a in 0..3 * n {
        for b in 0..3 * n {
            h[(a, b)] = (h[(a, b)] + t[(a, b)]) / 2.0;
        }
    }
    project_tr_rot(&h, &x)
}
