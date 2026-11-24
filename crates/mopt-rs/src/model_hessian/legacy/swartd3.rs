//! MultiOptPy's Swart-type model Hessian with its D3 term (`ModelHessian/swartd3.py`),
//! reproduced as it is.
//!
//! The "D3" curvature is MultiOptPy's own: `C6 = r4r2_i r4r2_j`, `C8 = 3 C6 sqrt(r4r2_i r4r2_j)`
//! with the `r4r2` table of `Parameters/d3.py` (not reference C6 coefficients), rational damping
//! with `a1 = 0.4289`, `a2 = 4.4407` on the geometric mean of UFF radii, and the second
//! derivative `42 C6 / r^8 + 72 C8 / r^10` without the damping derivatives. C-N triple bonds
//! ("cyano groups": a C-N distance within 0.3 Bohr of the triple-bond radii and at most one other
//! neighbor of the carbon) use larger bond and angle and smaller torsion and out-of-plane
//! constants. Unlike `swartd2`, the bond blocks are written to both triangles, the angle part
//! ends with averaging of differing mirror entries, and the final fill-in averages.
//!
//! MultiOptPy: `ModelHessian/swartd3.py:SwartD3ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{coords, cov_radius, mop_vdw_distance_bohr, norm, project_tr_rot, sub, V3};
use super::swartd2::{angle_part, bond_fc, out_of_plane_part, torsion_part, Variant};
use crate::domain::{Element, MolecularSystem};
use crate::parameters::multioptpy_tables::{
    MOP_BOHR2ANGSTROMS, MOP_D3_DEFAULT_R4R2, MOP_D3_R4R2, MOP_TRIPLE_COV_ANG,
};

const KD: f64 = 2.00;
const KR: f64 = 0.35;
const KF: f64 = 0.15;
const KT: f64 = 0.005;
const CN_KR: f64 = 0.70;
const CN_KF: f64 = 0.20;
const CN_KT: f64 = 0.002;
const S6: f64 = 1.0;
const S8: f64 = 0.7875;
const A1: f64 = 0.4289;
const A2: f64 = 4.4407;

fn r4r2(el: Element) -> f64 {
    let v = MOP_D3_R4R2
        .get(el.atomic_number() as usize)
        .copied()
        .unwrap_or(f64::NAN);
    if v.is_nan() {
        MOP_D3_DEFAULT_R4R2
    } else {
        v
    }
}

fn triple_radius(el: Element) -> f64 {
    MOP_TRIPLE_COV_ANG
        .get(el.atomic_number() as usize)
        .copied()
        .unwrap_or(f64::NAN)
        / MOP_BOHR2ANGSTROMS
}

/// `get_d3_parameters`: (C6, C8, R0).
fn d3_parameters(a: Element, b: Element) -> (f64, f64, f64) {
    let (ra, rb) = (r4r2(a), r4r2(b));
    let c6 = (ra * ra * (rb * rb)).sqrt();
    let c8 = 3.0 * c6 * (ra * rb).sqrt();
    let r0 = (mop_vdw_distance_bohr(a) * mop_vdw_distance_bohr(b)).sqrt();
    (c6, c8, r0)
}

/// `calc_d3_force_const`.
fn d3_force_const(r: f64, c6: f64, c8: f64, r0: f64) -> f64 {
    let d6 = 1.0 / (1.0 + 6.0 * (r / (A1 * r0)).powf(A2));
    let d8 = 1.0 / (1.0 + 6.0 * (r / (A2 * r0)).powf(A1));
    S6 * c6 * (42.0 / r.powi(8)) * d6 + S8 * c8 * (72.0 / r.powi(10)) * d8
}

fn d3_pair(elements: &[Element], i: usize, j: usize, r: f64) -> f64 {
    let (c6, c8, r0) = d3_parameters(elements[i], elements[j]);
    d3_force_const(r, c6, c8, r0)
}

/// `detect_cyano_groups`: `(C, N)` pairs.
pub(crate) fn cyano_groups(elements: &[Element], x: &[V3]) -> Vec<(usize, usize)> {
    let n = x.len();
    let mut out = Vec::new();
    let ideal = triple_radius(Element::C) + triple_radius(Element::N);
    for i in 0..n {
        if elements[i] != Element::C {
            continue;
        }
        for j in 0..n {
            if i == j || elements[j] != Element::N {
                continue;
            }
            if (norm(sub(x[i], x[j])) - ideal).abs() < 0.3 {
                let neighbors = (0..n)
                    .filter(|&k| k != i && k != j)
                    .filter(|&k| {
                        norm(sub(x[i], x[k]))
                            < 1.3 * (cov_radius(elements[i]) + cov_radius(elements[k]))
                    })
                    .count();
                if neighbors <= 1 {
                    out.push((i, j));
                }
            }
        }
    }
    out
}

fn bond_part(
    h: &mut Mat<f64>,
    elements: &[Element],
    x: &[V3],
    cyano: &[(usize, usize)],
    disp: &dyn Fn(usize, usize, f64) -> f64,
) {
    let n = x.len();
    for i in 0..n {
        for j in 0..i {
            let (xx, yy, zz) = (x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]);
            let r2 = xx * xx + yy * yy + zz * zz;
            let r = r2.sqrt();
            let cov = cov_radius(elements[i]) + cov_radius(elements[j]);
            let d3 = disp(i, j, r);
            let is_cn = cyano
                .iter()
                .any(|&(c, nn)| (i == c && j == nn) || (i == nn && j == c));
            let kr = if is_cn { CN_KR } else { KR };
            let g = kr * bond_fc(1.0, cov, r) + KD * d3;
            let (hxx, hxy, hxz) = (g * xx * xx / r2, g * xx * yy / r2, g * xx * zz / r2);
            let (hyy, hyz, hzz) = (g * yy * yy / r2, g * yy * zz / r2, g * zz * zz / r2);
            for a in [i, j] {
                h[(3 * a, 3 * a)] += hxx;
                h[(3 * a + 1, 3 * a)] += hxy;
                h[(3 * a + 1, 3 * a + 1)] += hyy;
                h[(3 * a + 2, 3 * a)] += hxz;
                h[(3 * a + 2, 3 * a + 1)] += hyz;
                h[(3 * a + 2, 3 * a + 2)] += hzz;
            }
            let block = [[hxx, hxy, hxz], [hxy, hyy, hyz], [hxz, hyz, hzz]];
            for p in 0..3 {
                for q in 0..3 {
                    h[(3 * i + p, 3 * j + q)] -= block[p][q];
                    h[(3 * j + p, 3 * i + q)] -= block[p][q];
                }
            }
        }
    }
}

pub(crate) fn average_differing(h: &mut Mat<f64>) {
    let n = h.nrows();
    for i in 0..n {
        for j in 0..i {
            if (h[(i, j)] - h[(j, i)]).abs() > 1.0e-10 {
                let avg = (h[(i, j)] + h[(j, i)]) / 2.0;
                h[(i, j)] = avg;
                h[(j, i)] = avg;
            }
        }
    }
}

/// The final fill-in of `swartd3.main` (and `swartd4.main`).
pub(crate) fn fill_or_average(h: &mut Mat<f64>) {
    let n = h.nrows();
    for i in 0..n {
        for j in 0..i {
            if h[(i, j)].abs() < 1.0e-10 {
                h[(i, j)] = h[(j, i)];
            } else if h[(j, i)].abs() < 1.0e-10 {
                h[(j, i)] = h[(i, j)];
            } else {
                let avg = (h[(i, j)] + h[(j, i)]) / 2.0;
                h[(i, j)] = avg;
                h[(j, i)] = avg;
            }
        }
    }
}

/// MultiOptPy's `SwartD3ApproxHessian.main`.
pub(crate) fn swartd3(sys: &MolecularSystem) -> Mat<f64> {
    let el = &sys.elements;
    let x = coords(sys);
    let mut h = swart_with_dispersion(sys, &|i, j, r| d3_pair(el, i, j, r));
    fill_or_average(&mut h);
    project_tr_rot(&h, &x)
}

/// The common part of `swartd3.py` and `swartd4.py` (bonds, angles with their averaging,
/// torsions, out-of-plane terms), which differ in the pair dispersion curvature
/// `disp(i, j, r)` and in what follows.
pub(crate) fn swart_with_dispersion(
    sys: &MolecularSystem,
    disp: &dyn Fn(usize, usize, f64) -> f64,
) -> Mat<f64> {
    let x = coords(sys);
    let el = &sys.elements;
    let n = x.len();
    let cyano = cyano_groups(el, &x);
    let in_cn = |a: usize| cyano.iter().any(|&(c, nn)| a == c || a == nn);
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    bond_part(&mut h, el, &x, &cyano, disp);
    let g = |i: usize, j: usize, r: f64| {
        bond_fc(1.0, cov_radius(el[i]) + cov_radius(el[j]), r) + 0.5 * KD * disp(i, j, r)
    };
    let kf = |i: usize, j: usize, k: usize| {
        if in_cn(i) || in_cn(j) || in_cn(k) {
            CN_KF
        } else {
            KF
        }
    };
    let kt = |q: [usize; 4]| {
        if q.iter().any(|&a| in_cn(a)) {
            CN_KT
        } else {
            KT
        }
    };
    let ko = |q: [usize; 4]| {
        if q.iter().any(|&a| in_cn(a)) {
            0.5 * KT
        } else {
            KT
        }
    };
    let angle = |i: usize, j: usize, k: usize, r_ij: f64, r_ik: f64| {
        kf(i, j, k) * g(i, j, r_ij) * g(i, k, r_ik)
    };
    let torsion = |q: [usize; 4], r: [f64; 3]| {
        kt(q) * g(q[0], q[1], r[0]) * g(q[1], q[2], r[1]) * g(q[2], q[3], r[2])
    };
    let oop = |q: [usize; 4], r: [f64; 3]| {
        ko(q) * g(q[0], q[1], r[0]) * g(q[0], q[2], r[1]) * g(q[0], q[3], r[2])
    };
    let v = Variant {
        angle: &angle,
        torsion: &torsion,
        oop: &oop,
    };
    angle_part(&mut h, &x, &v);
    average_differing(&mut h);
    torsion_part(&mut h, &x, &v);
    out_of_plane_part(&mut h, &x, &v);
    h
}
