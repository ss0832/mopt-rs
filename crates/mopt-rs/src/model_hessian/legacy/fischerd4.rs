//! MultiOptPy's Fischer model Hessian with its D4 term (`ModelHessian/fischerd4.py`),
//! reproduced as it is.
//!
//! The Fischer terms of `fischer.py` plus MultiOptPy's "D4" dispersion for non-bonded pairs
//! (1.3 x covalent-radius bonds): charges `0.2 (EN_j - EN_i) / (EN_i + EN_j)` per bond, charge
//! scaling `exp(-3 q^2)`, `C6 = 0.75 x 2 alpha_i alpha_j / (alpha_i / s_i + alpha_j / s_j)` with
//! MultiOptPy's polarizabilities, `C8 = 3 C6 sqrt(r4r2_i r4r2_j)`, `R0 = 4/3 (r_cov,i +
//! r_cov,j)`, Becke-Johnson damping (`a1 = 0.4171`, `a2 = 4.5337`, `a2 + 2` for `C8`), and a
//! three-body term whose Hessian is the forward difference (step 1e-5 Bohr) of an approximate
//! three-body gradient over triplets within 20 Bohr. The sum is symmetrized and projected.
//!
//! MultiOptPy: `ModelHessian/fischerd4.py:FischerD4ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{coords, cov_radius, dot, norm, project_tr_rot, sub, V3};
use super::fischer::fischer_terms;
use super::fischerd3::{add_pair_block, damped_pair_block};
use crate::domain::{Element, MolecularSystem};
use crate::parameters::multioptpy_tables::{
    MOP_D4_DEFAULT_ELECTRONEGATIVITY, MOP_D4_DEFAULT_POLARIZABILITY, MOP_D4_DEFAULT_R4R2,
    MOP_D4_ELECTRONEGATIVITY, MOP_D4_GA, MOP_D4_POLARIZABILITY, MOP_D4_R4R2,
};

const S6: f64 = 1.0;
const S8: f64 = 1.03683;
const S9: f64 = 1.0;
const A1: f64 = 0.4171;
const A2: f64 = 4.5337;

fn value(t: &[f64], el: Element, default: f64) -> f64 {
    match t.get(el.atomic_number() as usize) {
        Some(v) if !v.is_nan() => *v,
        _ => default,
    }
}

fn polarizability(el: Element) -> f64 {
    value(&MOP_D4_POLARIZABILITY, el, MOP_D4_DEFAULT_POLARIZABILITY)
}

fn r4r2(el: Element) -> f64 {
    value(&MOP_D4_R4R2, el, MOP_D4_DEFAULT_R4R2)
}

fn electronegativity(el: Element) -> f64 {
    value(
        &MOP_D4_ELECTRONEGATIVITY,
        el,
        MOP_D4_DEFAULT_ELECTRONEGATIVITY,
    )
}

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

fn charges(el: &[Element], bonded: &[Vec<bool>]) -> Vec<f64> {
    let n = el.len();
    let mut q = vec![0.0; n];
    for i in 0..n {
        for j in i + 1..n {
            if bonded[i][j] {
                let (ei, ej) = (electronegativity(el[i]), electronegativity(el[j]));
                let t = 0.2 * (ej - ei) / (ei + ej);
                q[i] += t;
                q[j] -= t;
            }
        }
    }
    q
}

fn c6(a: Element, b: Element, qa: f64, qb: f64) -> f64 {
    let (aa, ab) = (polarizability(a), polarizability(b));
    let (sa, sb) = ((-MOP_D4_GA * qa * qa).exp(), (-MOP_D4_GA * qb * qb).exp());
    2.0 * aa * ab / ((aa / sa) + (ab / sb)) * 0.75
}

fn r0(a: Element, b: Element) -> f64 {
    (cov_radius(a) + cov_radius(b)) * 4.0 / 3.0
}

fn damp6(r: f64, r0: f64) -> f64 {
    let p = r.powi(6);
    p / (p + (A1 * r0 + A2).powi(6))
}

fn triplets(x: &[V3]) -> Vec<(usize, usize, usize)> {
    let n = x.len();
    let cut2 = 20.0f64 * 20.0;
    let mut nb: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in i + 1..n {
            let d = sub(x[i], x[j]);
            if dot(d, d) < cut2 {
                nb[i].push(j);
                nb[j].push(i);
            }
        }
    }
    let mut out = Vec::new();
    for i in 0..n {
        for &j in &nb[i] {
            if j <= i {
                continue;
            }
            for &k in &nb[j] {
                if k <= j {
                    continue;
                }
                if nb[i].contains(&k) {
                    out.push((i, j, k));
                }
            }
        }
    }
    out
}

fn three_body_gradient(
    el: &[Element],
    x: &[V3],
    q: &[f64],
    trip: &[(usize, usize, usize)],
) -> Vec<V3> {
    let n = x.len();
    let mut g = vec![[0.0; 3]; n];
    let unit = |v: V3, r: f64| [v[0] / r, v[1] / r, v[2] / r];
    for &(i, j, k) in trip {
        let (vij, vjk, vki) = (sub(x[j], x[i]), sub(x[k], x[j]), sub(x[i], x[k]));
        let (rij, rjk, rki) = (
            dot(vij, vij).sqrt(),
            dot(vjk, vjk).sqrt(),
            dot(vki, vki).sqrt(),
        );
        if rij.min(rjk).min(rki) < 0.1 {
            continue;
        }
        let (uij, ujk, uki) = (unit(vij, rij), unit(vjk, rjk), unit(vki, rki));
        let c9 = (c6(el[i], el[j], q[i], q[j])
            * c6(el[j], el[k], q[j], q[k])
            * c6(el[k], el[i], q[k], q[i]))
        .sqrt();
        let neg = |v: V3| [-v[0], -v[1], -v[2]];
        let ci = dot(neg(uki), uij);
        let cj = dot(neg(uij), ujk);
        let ck = dot(neg(ujk), uki);
        let angle = 1.0 + 3.0 * ci * cj * ck;
        let damp = damp6(rij, r0(el[i], el[j]))
            * damp6(rjk, r0(el[j], el[k]))
            * damp6(rki, r0(el[k], el[i]));
        let pre = S9 * c9 * damp * angle;
        let den = (rij * rjk * rki).powi(3);
        let gb = -3.0 * pre / den;
        let ga = 3.0 * pre / den;
        for t in 0..3 {
            g[i][t] += gb * (uij[t] / rij - uki[t] / rki);
            g[j][t] += gb * (ujk[t] / rjk - uij[t] / rij);
            g[k][t] += gb * (uki[t] / rki - ujk[t] / rjk);
        }
        for t in 0..3 {
            g[i][t] += ga * cj * ck * (-uij[t] - uki[t]);
            g[j][t] += ga * ci * ck * (-ujk[t] - uij[t]);
            g[k][t] += ga * ci * cj * (-uki[t] - ujk[t]);
        }
    }
    g
}

fn three_body_hessian(el: &[Element], x: &[V3], q: &[f64]) -> Mat<f64> {
    let n = x.len();
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    let trip = triplets(x);
    if trip.is_empty() {
        return h;
    }
    let base = three_body_gradient(el, x, q, &trip);
    let delta = 1e-5;
    for i in 0..n {
        for c in 0..3 {
            let mut xp = x.to_vec();
            xp[i][c] += delta;
            let gp = three_body_gradient(el, &xp, q, &trip);
            for a in 0..n {
                for t in 0..3 {
                    h[(3 * i + c, 3 * a + t)] = (gp[a][t] - base[a][t]) / delta;
                }
            }
        }
    }
    let t = h.transpose().to_owned();
    for a in 0..3 * n {
        for b in 0..3 * n {
            h[(a, b)] = (h[(a, b)] + t[(a, b)]) / 2.0;
        }
    }
    h
}

/// MultiOptPy's `FischerD4ApproxHessian.main`.
pub(crate) fn fischerd4(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let el = &sys.elements;
    let n = x.len();
    let mut h = fischer_terms(el, &x);
    let bonded = bonded_13(el, &x);
    let q = charges(el, &bonded);
    for i in 0..n {
        for j in 0..i {
            if bonded[i][j] || norm(sub(x[i], x[j])) < 0.1 {
                continue;
            }
            let c = c6(el[i], el[j], q[i], q[j]);
            let c8 = 3.0 * c * (r4r2(el[i]) * r4r2(el[j])).sqrt();
            let b = damped_pair_block(sub(x[i], x[j]), c, c8, r0(el[i], el[j]), S6, S8, A1, A2);
            add_pair_block(&mut h, i, j, &b);
        }
    }
    h += three_body_hessian(el, &x, &q);
    let t = h.transpose().to_owned();
    for a in 0..3 * n {
        for b in 0..3 * n {
            h[(a, b)] = (h[(a, b)] + t[(a, b)]) / 2.0;
        }
    }
    project_tr_rot(&h, &x)
}
