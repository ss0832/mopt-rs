//! MultiOptPy's Swart-type model Hessian with its D4 term (`ModelHessian/swartd4.py`),
//! reproduced as it is.
//!
//! As `swartd3`, with MultiOptPy's "D4" curvature: charges from a bond-wise electronegativity
//! transfer (`0.1 (EN_j - EN_i)` for every pair within 1.3 x covalent radii), charge scaling
//! `exp(-3 |q|)`, `C6` the harmonic mean of `alpha r4r2 exp(-3|q|)` with MultiOptPy's
//! polarizability and `r4r2` tables, `C8 = 3 C6 sqrt(r4r2_i r4r2_j)`, rational damping with
//! `a1 = 0.4171`, `a2 = 4.5337`, `s8 = 1.03683`. A three-body term adds constant 3x3 blocks
//! `0.01 s9 (1 + 3 cos cos cos) C9 f_damp / r^6`. The result is symmetrized by averaging
//! differing mirror entries. (MultiOptPy also computes coordination numbers here, but does not
//! use them.)
//!
//! MultiOptPy: `ModelHessian/swartd4.py:SwartD4ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{
    coords, cov_radius, dot, mop_vdw_distance_bohr, norm, project_tr_rot, sub, V3,
};
use super::swartd3::{average_differing, swart_with_dispersion};
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

fn r4r2(el: Element) -> f64 {
    value(&MOP_D4_R4R2, el, MOP_D4_DEFAULT_R4R2)
}

fn polarizability(el: Element) -> f64 {
    value(&MOP_D4_POLARIZABILITY, el, MOP_D4_DEFAULT_POLARIZABILITY)
}

fn electronegativity(el: Element) -> f64 {
    value(
        &MOP_D4_ELECTRONEGATIVITY,
        el,
        MOP_D4_DEFAULT_ELECTRONEGATIVITY,
    )
}

/// `estimate_atomic_charges`.
pub(crate) fn transfer_charges(elements: &[Element], x: &[V3]) -> Vec<f64> {
    let n = x.len();
    let mut q = vec![0.0; n];
    for i in 0..n {
        for j in i + 1..n {
            let rc = cov_radius(elements[i]) + cov_radius(elements[j]);
            if norm(sub(x[i], x[j])) < 1.3 * rc {
                let t = 0.1 * (electronegativity(elements[j]) - electronegativity(elements[i]));
                q[i] += t;
                q[j] -= t;
            }
        }
    }
    q
}

fn charge_scale(q: f64) -> f64 {
    let arg = -MOP_D4_GA * q.abs();
    if arg < -50.0 {
        0.0
    } else {
        arg.exp()
    }
}

/// `get_d4_parameters`: (C6, C8, R0).
pub(crate) fn d4_parameters(a: Element, b: Element, qa: f64, qb: f64) -> (f64, f64, f64) {
    let (ra, rb) = (r4r2(a), r4r2(b));
    let c1 = polarizability(a) * ra * charge_scale(qa);
    let c2 = polarizability(b) * rb * charge_scale(qb);
    let c6 = 2.0 * c1 * c2 / (c1 + c2);
    let c8 = 3.0 * c6 * (ra * rb).sqrt();
    let r0 = (mop_vdw_distance_bohr(a) * mop_vdw_distance_bohr(b)).sqrt();
    (c6, c8, r0)
}

pub(crate) fn damp6(r: f64, r0: f64) -> f64 {
    1.0 / (1.0 + 6.0 * (r / (A1 * r0)).powf(A2))
}

pub(crate) fn damp8(r: f64, r0: f64) -> f64 {
    1.0 / (1.0 + 6.0 * (r / (A2 * r0)).powf(A1))
}

fn d4_force_const(r: f64, c6: f64, c8: f64, r0: f64) -> f64 {
    S6 * c6 * (42.0 / r.powi(8)) * damp6(r, r0) + S8 * c8 * (72.0 / r.powi(10)) * damp8(r, r0)
}

fn three_body(h: &mut Mat<f64>, el: &[Element], x: &[V3], q: &[f64]) {
    let n = x.len();
    let p = |a: usize, b: usize| d4_parameters(el[a], el[b], q[a], q[b]);
    for i in 0..n {
        for j in 0..i {
            for k in 0..j {
                let (rij, rjk, rki) = (
                    norm(sub(x[i], x[j])),
                    norm(sub(x[j], x[k])),
                    norm(sub(x[k], x[i])),
                );
                let (c6ij, _, r0ij) = p(i, j);
                let (c6jk, _, r0jk) = p(j, k);
                let (c6ki, _, r0ki) = p(k, i);
                let c9 = (c6ij * c6jk * c6ki).powf(1.0 / 3.0);
                let damp = damp6(rij, r0ij) * damp6(rjk, r0jk) * damp6(rki, r0ki);
                let cos_ijk = dot(sub(x[i], x[j]), sub(x[k], x[j])) / (rij * rjk);
                let cos_jki = dot(sub(x[j], x[k]), sub(x[i], x[k])) / (rjk * rki);
                let cos_kij = dot(sub(x[k], x[i]), sub(x[j], x[i])) / (rki * rij);
                let angle = 1.0 + 3.0 * cos_ijk * cos_jki * cos_kij;
                let fc = 0.01 * S9 * angle * c9 * damp;
                let (fij, fjk, fki) = (fc / rij.powi(6), fc / rjk.powi(6), fc / rki.powi(6));
                for a in 0..3 {
                    for b in 0..3 {
                        if a == b {
                            h[(3 * i + a, 3 * i + b)] += fij + fki;
                            h[(3 * j + a, 3 * j + b)] += fij + fjk;
                            h[(3 * k + a, 3 * k + b)] += fjk + fki;
                        }
                        h[(3 * i + a, 3 * j + b)] -= fij;
                        h[(3 * j + a, 3 * i + b)] -= fij;
                        h[(3 * j + a, 3 * k + b)] -= fjk;
                        h[(3 * k + a, 3 * j + b)] -= fjk;
                        h[(3 * k + a, 3 * i + b)] -= fki;
                        h[(3 * i + a, 3 * k + b)] -= fki;
                    }
                }
            }
        }
    }
}

/// MultiOptPy's `SwartD4ApproxHessian.main`.
pub(crate) fn swartd4(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let el = &sys.elements;
    let q = transfer_charges(el, &x);
    let disp = |i: usize, j: usize, r: f64| {
        let (c6, c8, r0) = d4_parameters(el[i], el[j], q[i], q[j]);
        d4_force_const(r, c6, c8, r0)
    };
    let mut h = swart_with_dispersion(sys, &disp);
    three_body(&mut h, el, &x, &q);
    average_differing(&mut h);
    project_tr_rot(&h, &x)
}
