//! MultiOptPy's Swart model Hessian (`ModelHessian/swart.py`), reproduced as it is: the
//! O1NumHess form of Swart & Bickelhaupt's model with screened bonds over all pairs and screened
//! angles (with the near-linear treatment), but without the dihedral and out-of-plane terms,
//! which MultiOptPy leaves empty. When anything is not finite, only the bond part is kept.
//!
//! MultiOptPy: `ModelHessian/swart.py:SwartApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{coords, cross, dot, norm, project_tr_rot, sub, V3};
use crate::domain::{Element, MolecularSystem};

/// MultiOptPy's radii for the screening (Bohr), H..Rn; other elements use 1.0.
const RADII: [f64; 87] = [
    1.0, 0.59, 0.54, 2.43, 1.72, 1.53, 1.40, 1.34, 1.25, 1.18, 1.14, 2.89, 2.53, 2.19, 2.10, 2.04,
    1.97, 1.87, 1.82, 3.42, 3.06, 2.85, 2.70, 2.55, 2.49, 2.49, 2.44, 2.38, 2.32, 2.42, 2.40, 2.27,
    2.19, 2.17, 2.10, 2.04, 2.06, 3.70, 3.40, 3.21, 2.98, 2.85, 2.72, 2.61, 2.55, 2.51, 2.55, 2.68,
    2.72, 2.61, 2.55, 2.51, 2.48, 2.44, 2.48, 4.03, 3.59, 3.34, 3.25, 3.23, 3.21, 3.19, 3.17, 3.17,
    3.15, 3.13, 3.13, 3.11, 3.11, 3.09, 3.09, 3.06, 2.89, 2.76, 2.61, 2.49, 2.46, 2.42, 2.42, 2.55,
    2.72, 2.68, 2.68, 2.68, 2.61, 2.57, 2.63,
];

const WTHR: f64 = 0.3;
const F: f64 = 0.12;
const TOLTH: f64 = 0.2;
const MIN_DIST: f64 = 1e-8;
const MIN_SIN_SQ: f64 = 1e-12;
const MIN_NORM: f64 = 1e-12;

fn radius(el: Element) -> f64 {
    let z = el.atomic_number() as usize;
    if (1..=86).contains(&z) {
        RADII[z]
    } else {
        1.0
    }
}

fn scale(v: V3, s: f64) -> V3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn add_block(h: &mut Mat<f64>, atoms: &[usize], b: &[f64], k: f64) {
    let m = b.len();
    for p in 0..m {
        for q in 0..m {
            h[(3 * atoms[p / 3] + p % 3, 3 * atoms[q / 3] + q % 3)] += k * b[p] * b[q];
        }
    }
}

struct Geometry {
    diff: Vec<Vec<V3>>,
    dist: Vec<Vec<f64>>,
    screen: Vec<Vec<f64>>,
}

fn geometry(elements: &[Element], x: &[V3]) -> Geometry {
    let n = x.len();
    let mut diff = vec![vec![[0.0; 3]; n]; n];
    let mut dist = vec![vec![1.0; n]; n];
    let mut screen = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            diff[i][j] = sub(x[i], x[j]);
            if i != j {
                dist[i][j] = norm(diff[i][j]).max(MIN_DIST);
                let c = (radius(elements[i]) + radius(elements[j])).max(MIN_DIST);
                screen[i][j] = (1.0 - dist[i][j] / c).exp();
            }
        }
    }
    Geometry { diff, dist, screen }
}

fn bonds(h: &mut Mat<f64>, g: &Geometry) {
    let n = g.dist.len();
    for i in 0..n {
        for j in i + 1..n {
            let k = 0.35 * g.screen[i][j].powi(3);
            let u = scale(g.diff[i][j], 1.0 / g.dist[i][j]);
            let b = [u[0], u[1], u[2], -u[0], -u[1], -u[2]];
            add_block(h, &[i, j], &b, k);
        }
    }
}

fn angle_b(v1: V3, v2: V3, l1: f64, l2: f64) -> ([f64; 9], f64, f64) {
    let (l1s, l2s) = (l1.max(MIN_DIST), l2.max(MIN_DIST));
    let (n1, n2) = (scale(v1, 1.0 / l1s), scale(v2, 1.0 / l2s));
    let c = dot(n1, n2).clamp(-1.0, 1.0);
    let s2 = (1.0 - c * c).max(MIN_SIN_SQ);
    let den = s2.sqrt().max(1e-6);
    let mut b = [0.0; 9];
    for t in 0..3 {
        b[t] = (c * n1[t] - n2[t]) / (l1s * den);
        b[6 + t] = (c * n2[t] - n1[t]) / (l2s * den);
        b[3 + t] = -(b[t] + b[6 + t]);
    }
    (b, c, s2)
}

fn linear_b(v1: V3, v2: V3, l1: f64, l2: f64) -> [[f64; 9]; 2] {
    let (l1s, l2s) = (l1.max(MIN_DIST), l2.max(MIN_DIST));
    let mut vn = cross(v1, v2);
    let mut nvn = norm(vn);
    if nvn < MIN_NORM {
        let try_ref = |r: V3| {
            let s = dot(r, v1) / (l1s * l1s);
            let c = sub(r, scale(v1, s));
            (c, norm(c))
        };
        let (c1, n1) = try_ref([1.0, 0.0, 0.0]);
        if n1 >= MIN_NORM {
            vn = c1;
            nvn = n1;
        } else {
            let (c2, n2) = try_ref([0.0, 1.0, 0.0]);
            vn = c2;
            nvn = n2.max(MIN_NORM);
        }
    }
    let vnn = scale(vn, 1.0 / nvn.max(MIN_NORM));
    let vn2 = cross(sub(v1, v2), vnn);
    let vn2n = scale(vn2, 1.0 / norm(vn2).max(MIN_NORM));
    let mut b = [[0.0; 9]; 2];
    for t in 0..3 {
        b[1][t] = vnn[t] / l1s;
        b[1][6 + t] = vnn[t] / l2s;
        b[1][3 + t] = -b[1][t] - b[1][6 + t];
        b[0][t] = vn2n[t] / l1s;
        b[0][6 + t] = vn2n[t] / l2s;
        b[0][3 + t] = -b[0][t] - b[0][6 + t];
    }
    b
}

fn angles(h: &mut Mat<f64>, g: &Geometry) {
    let n = g.dist.len();
    let eps1 = WTHR * WTHR;
    let eps2 = WTHR * WTHR / std::f64::consts::E;
    for j in 0..n {
        let nb: Vec<usize> = (0..n).filter(|&k| g.screen[j][k] >= eps2).collect();
        for (a, &i) in nb.iter().enumerate() {
            for &k in &nb[a + 1..] {
                let s = g.screen[i][j] * g.screen[j][k];
                if s < eps1 {
                    continue;
                }
                let (l1, l2) = (g.dist[i][j], g.dist[k][j]);
                if !(l1 > MIN_DIST && l2 > MIN_DIST) {
                    continue;
                }
                let (v1, v2) = (g.diff[i][j], g.diff[k][j]);
                let (b, c, s2) = angle_b(v1, v2, l1, l2);
                let sin = s2.sqrt();
                let hb = 0.075 * s * s * (F + (1.0 - F) * sin).powi(2);
                let th1 = if c > 1.0 - TOLTH { 1.0 - c } else { 1.0 + c };
                let atoms = [i, j, k];
                if th1 >= TOLTH {
                    add_block(h, &atoms, &b, hb);
                    continue;
                }
                let sl = (1.0 - (th1 / TOLTH.max(1e-9)).powi(2)).powi(2);
                let bl = linear_b(v1, v2, l1, l2);
                if c > 1.0 - TOLTH {
                    let comb: Vec<f64> =
                        (0..9).map(|t| sl * bl[0][t] + (1.0 - sl) * b[t]).collect();
                    add_block(h, &atoms, &bl[1], hb);
                    add_block(h, &atoms, &comb, hb);
                } else {
                    let sc: Vec<f64> = b.iter().map(|v| (1.0 - sl) * v).collect();
                    add_block(h, &atoms, &sc, hb);
                }
            }
        }
    }
}

/// The unprojected Hessian (bonds and angles; bonds only when not finite).
pub(crate) fn swart_raw(elements: &[Element], x: &[V3]) -> Mat<f64> {
    let n = x.len();
    let g = geometry(elements, x);
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    bonds(&mut h, &g);
    angles(&mut h, &g);
    let finite = (0..3 * n).all(|i| (0..3 * n).all(|j| h[(i, j)].is_finite()));
    if !finite {
        h = Mat::zeros(3 * n, 3 * n);
        bonds(&mut h, &g);
    }
    h
}

/// MultiOptPy's `SwartApproxHessian.main`.
pub(crate) fn swart(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    project_tr_rot(&swart_raw(&sys.elements, &x), &x)
}
