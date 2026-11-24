//! MultiOptPy's Schlegel model Hessian with its D4 term (`ModelHessian/schlegeld4.py`),
//! reproduced as it is: `schlegeld3`'s pair-distance model with the "D4" curvature of
//! `swartd4` (charges from bond-wise electronegativity transfer, `exp(-3|q|)` scaling,
//! harmonic-mean `C6`, damped `C6` and `C8` terms, scaled by 0.01) and a three-body term of
//! constant 3x3 blocks `0.002 s9 (1 + 3 cos cos cos) C9 f_damp / r^6` (cosines between the
//! cyclic bond vectors `r_j - r_i`, `r_k - r_j`, `r_i - r_k`; skipped below a damping of
//! 1e-8). The Cartesian Hessian is symmetrized by averaging before the projection.
//!
//! MultiOptPy: `ModelHessian/schlegeld4.py:SchlegelD4ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{coords, cov_radius, dot, norm, project_tr_rot, sub, Connectivity, V3};
use super::schlegel::{pair_distance_hessian, pair_index};
use super::swartd3::cyano_groups;
use super::swartd4::{d4_parameters, damp6, damp8, transfer_charges};
use crate::domain::{Element, MolecularSystem};

const B: [[f64; 7]; 7] = [
    [0.2573, 0.3401, 0.6937, 0.7126, 0.8335, 0.9491, 0.9491],
    [0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190, 1.7190],
    [0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185, 2.3185],
    [0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206, 2.5206],
    [0.8335, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110, 2.5110],
    [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110],
    [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110],
];

const S6: f64 = 1.0;
const S8: f64 = 1.03683;
const S9: f64 = 1.0;

fn row(el: Element) -> usize {
    match el.atomic_number() {
        1..=2 => 0,
        3..=10 => 1,
        11..=18 => 2,
        19..=36 => 3,
        37..=54 => 4,
        55..=86 => 5,
        _ => 6,
    }
}

fn d4_correction(r: f64, a: Element, b: Element, qa: f64, qb: f64) -> f64 {
    let (c6, c8, r0) = d4_parameters(a, b, qa, qb);
    (S6 * c6 * (42.0 / r.powi(8)) * damp6(r, r0) + S8 * c8 * (72.0 / r.powi(10)) * damp8(r, r0))
        * 0.01
}

fn constants(el: &[Element], x: &[V3], q: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut f = vec![0.0; n * n.saturating_sub(1) / 2];
    let cyano = cyano_groups(el, x);
    let in_cn = |a: usize| cyano.iter().any(|&(c, nn)| a == c || a == nn);
    let dist = |a: usize, b: usize| norm(sub(x[a], x[b]));
    let corr = |a: usize, b: usize| d4_correction(dist(a, b), el[a], el[b], q[a], q[b]);
    let bc = Connectivity::new(el, x);
    for [i, j] in bc.bonds() {
        let fb = 1.734 / (dist(i, j) - B[row(el[i])][row(el[j])]).powi(3);
        let is_cn = cyano
            .iter()
            .any(|&(c, nn)| (i == c && j == nn) || (i == nn && j == c));
        f[pair_index(n, i, j)] += if is_cn { 2.0 * fb } else { fb } + corr(i, j);
    }
    for [i, j, k] in bc.angles() {
        let fa = if el[i] == Element::H || el[k] == Element::H {
            0.160
        } else {
            0.250
        };
        let fa = if in_cn(i) || in_cn(j) || in_cn(k) {
            1.5 * fa
        } else {
            fa
        };
        f[pair_index(n, i, j)] += fa + corr(i, j) * 0.2;
        f[pair_index(n, j, k)] += fa + corr(j, k) * 0.2;
    }
    for [i, j, k, l] in bc.dihedrals() {
        let ft = 0.0023 - 0.07 * (dist(j, k) - (cov_radius(el[j]) + cov_radius(el[k])));
        let ft = if [i, j, k, l].iter().any(|&a| in_cn(a)) {
            0.5 * ft
        } else {
            ft
        };
        f[pair_index(n, i, j)] += ft + corr(i, j) * 0.05;
        f[pair_index(n, j, k)] += ft + corr(j, k) * 0.05;
        f[pair_index(n, k, l)] += ft + corr(k, l) * 0.05;
    }
    f
}

fn three_body(el: &[Element], x: &[V3], q: &[f64]) -> Mat<f64> {
    let n = x.len();
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    let p = |a: usize, b: usize| d4_parameters(el[a], el[b], q[a], q[b]);
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let (rij, rjk, rki) = (
                    norm(sub(x[i], x[j])),
                    norm(sub(x[j], x[k])),
                    norm(sub(x[k], x[i])),
                );
                let (c6ij, _, r0ij) = p(i, j);
                let (c6jk, _, r0jk) = p(j, k);
                let (c6ki, _, r0ki) = p(k, i);
                let c9 = (c6ij * c6jk * c6ki).cbrt();
                let damp = damp6(rij, r0ij) * damp6(rjk, r0jk) * damp6(rki, r0ki);
                if damp < 1e-8 {
                    continue;
                }
                let (vij, vjk, vki) = (sub(x[j], x[i]), sub(x[k], x[j]), sub(x[i], x[k]));
                let neg = |v: V3| [-v[0], -v[1], -v[2]];
                let cos_ijk = dot(vij, vjk) / (rij * rjk);
                let cos_jki = dot(vjk, neg(vki)) / (rjk * rki);
                let cos_kij = dot(neg(vki), vij) / (rki * rij);
                let angle = 1.0 + 3.0 * cos_ijk * cos_jki * cos_kij;
                let fc = 0.002 * S9 * angle * c9 * damp;
                let (fij, fjk, fki) = (fc / rij.powi(6), fc / rjk.powi(6), fc / rki.powi(6));
                for a in 0..3 {
                    for b in 0..3 {
                        if a == b {
                            h[(3 * i + a, 3 * i + a)] += fij + fki;
                            h[(3 * j + a, 3 * j + a)] += fij + fjk;
                            h[(3 * k + a, 3 * k + a)] += fjk + fki;
                        }
                        h[(3 * i + a, 3 * j + b)] -= fij;
                        h[(3 * j + b, 3 * i + a)] -= fij;
                        h[(3 * j + a, 3 * k + b)] -= fjk;
                        h[(3 * k + b, 3 * j + a)] -= fjk;
                        h[(3 * k + a, 3 * i + b)] -= fki;
                        h[(3 * i + b, 3 * k + a)] -= fki;
                    }
                }
            }
        }
    }
    h
}

/// MultiOptPy's `SchlegelD4ApproxHessian.main`.
pub(crate) fn schlegeld4(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let el = &sys.elements;
    let q = transfer_charges(el, &x);
    let f = constants(el, &x, &q);
    let mut h = pair_distance_hessian(&x, &f);
    h += three_body(el, &x, &q);
    let n = h.nrows();
    for i in 0..n {
        for j in 0..n {
            if h[(i, j)].is_nan() {
                h[(i, j)] = 0.0;
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            let avg = (h[(i, j)] + h[(j, i)]) / 2.0;
            h[(i, j)] = avg;
            h[(j, i)] = avg;
        }
    }
    project_tr_rot(&h, &x)
}
