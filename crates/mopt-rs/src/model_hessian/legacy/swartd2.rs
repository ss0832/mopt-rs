//! MultiOptPy's Swart-type model Hessian with its D2 term (`ModelHessian/swartd2.py`),
//! reproduced as it is: xtb's Swart traversal with bond, angle, torsion, and out-of-plane terms
//! written into (mostly) the lower triangle, MultiOptPy's D2 pair curvature
//! (`calc_params.calc_vdw_isotopic` / `calc_vdw_anisotropic`, which use `0.1` where the D2
//! formula has `1`), and its fill-in of small entries from their mirror image. Index slips are
//! kept: the off-diagonal pair block gets `-xy` and `-yy` both on `(3i+1, 3j+1)` and nothing on
//! `(3i+1, 3j)`, `(3i+2, 3j)`; the self blocks of angles, torsions, and out-of-plane terms only
//! receive their strictly lower parts.
//!
//! MultiOptPy: `ModelHessian/swartd2.py:SwartD2ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use super::common::{
    coords, cov_radius, dot, mop_vdw_distance_bohr, outofplane2, project_tr_rot, sub, torsion2, V3,
};
use crate::domain::{Element, MolecularSystem};
use crate::parameters::multioptpy_tables::{
    MOP_BOHR2ANGSTROMS, MOP_D2_C6_J_NM6_MOL, MOP_D2_VDW_ANG, MOP_HARTREE2J, MOP_MOL2AU,
};

const KD: f64 = 2.00;
const KR: f64 = 0.35;
const KF: f64 = 0.15;
const KT: f64 = 0.005;
const EPS: f64 = 1.0e-12;
/// `Parameters.d2.D2_S6_parameter`.
const D2_S6: f64 = 1.00;

fn table(t: &[f64], el: Element) -> f64 {
    t.get(el.atomic_number() as usize)
        .copied()
        .unwrap_or(f64::NAN)
}

/// `D2_C6_coeff_lib` (Hartree Bohr^6).
pub(crate) fn mop_d2_c6(el: Element) -> f64 {
    table(&MOP_D2_C6_J_NM6_MOL, el) * 1e6
        / MOP_BOHR2ANGSTROMS.powf(6.0)
        / MOP_HARTREE2J
        / MOP_MOL2AU
}

/// `D2_VDW_radii_lib` (Bohr).
pub(crate) fn mop_d2_vdw(el: Element) -> f64 {
    table(&MOP_D2_VDW_ANG, el) / MOP_BOHR2ANGSTROMS
}

/// `calc_params.calc_vdw_isotopic` (diagonal element of the D2 pair curvature).
pub(crate) fn vdw_isotropic(x: f64, y: f64, z: f64, c6: f64, r0: f64) -> f64 {
    let (x, y, z) = (x + 1.0e-12, y + 1.0e-12, z + 1.0e-12);
    let a = 20.0;
    let t1 = D2_S6 * c6;
    let t2 = x * x;
    let t5 = t2 + y * y + z * z;
    let t6 = t5 * t5;
    let t7 = t6 * t6;
    let t10 = t5.powf(0.5);
    let t11 = 0.1 / r0;
    let t15 = (-a * (t10 * t11 - 0.1)).exp() + 1.0e-12;
    let t16 = 0.1 + t15;
    let t17 = 0.1 / t16;
    let t24 = t16 * t16;
    let t25 = 0.1 / t24;
    let t29 = t11 * t15;
    let t33 = 0.1 / t7;
    let t44 = a * a / (r0 * r0);
    let t45 = t15 * t15;
    -0.48 * t1 / t7 / t5 * t17 * t2
        + 0.13 * t1 / t10 / t7 * t25 * t2 * a * t29
        + 0.6 * t1 * t33 * t17
        - 0.2 * t1 * t33 / t24 / t16 * t44 * t2 * t45
        - t1 / t10 / t6 / t5 * t25 * a * t29
        + t1 * t33 * t25 * t44 * t2 * t15
}

/// `calc_params.calc_vdw_anisotropic` (off-diagonal element for the first two arguments).
pub(crate) fn vdw_anisotropic(x: f64, y: f64, z: f64, c6: f64, r0: f64) -> f64 {
    let a = 20.0;
    let (x, y, z) = (x + 1.0e-12, y + 1.0e-12, z + 1.0e-12);
    let t1 = D2_S6 * c6;
    let t5 = x * x + y * y + z * z;
    let t6 = t5 * t5;
    let t7 = t6 * t6;
    let t11 = t5.powf(0.5);
    let t12 = 0.1 / r0;
    let t16 = (-a * (t11 * t12 - 0.1)).exp() + 1.0e-12;
    let t17 = 0.1 + t16;
    let t25 = t17 * t17;
    let t26 = 0.1 / t25;
    let t35 = 0.1 / t7;
    let t43 = a * a / (r0 * r0);
    let t44 = t16 * t16;
    -0.48 * t1 / t7 / t5 / t17 * x * y + 0.13 * t1 / t11 / t7 * t26 * x * a * t12 * y * t16
        - 0.2 * t1 * t35 / t25 / t17 * t43 * x * t44 * y
        + t1 * t35 * t26 * t43 * x * y * t16
}

pub(crate) fn bond_fc(alpha: f64, rcov: f64, r: f64) -> f64 {
    (-alpha * (r / rcov - 1.0)).exp()
}

fn vdw_fc(alpha: f64, rvdw: f64, r: f64) -> f64 {
    (-alpha * (rvdw - r).powi(2)).exp()
}

pub(crate) struct Lengths {
    pub cov: Box<dyn Fn(usize, usize) -> f64>,
    pub vdw: Box<dyn Fn(usize, usize) -> f64>,
}

fn lengths(elements: &[Element]) -> Lengths {
    let e1 = elements.to_vec();
    let e2 = elements.to_vec();
    Lengths {
        cov: Box::new(move |i, j| cov_radius(e1[i]) + cov_radius(e1[j])),
        vdw: Box::new(move |i, j| mop_vdw_distance_bohr(e2[i]) + mop_vdw_distance_bohr(e2[j])),
    }
}

/// `g = kf(1) + kd/2 vdw(5)` of the angle, torsion, and out-of-plane terms.
fn g_pair(l: &Lengths, i: usize, j: usize, r: f64) -> f64 {
    bond_fc(1.0, (l.cov)(i, j), r) + 0.5 * KD * vdw_fc(5.0, (l.vdw)(i, j), r)
}

/// The pair (bond + D2) part; `pair_curvature(dx, dy, dz, i, j)` returns the six
/// dispersion entries (xx, xy, xz, yy, yz, zz) subtracted from the bond curvature.
pub(crate) fn bond_part(
    h: &mut Mat<f64>,
    x: &[V3],
    gmm: &dyn Fn(usize, usize, f64, f64) -> f64,
    pair_curvature: &dyn Fn(f64, f64, f64, usize, usize) -> [f64; 6],
) {
    let n = x.len();
    for i in 0..n {
        for j in 0..i {
            let (xx, yy, zz) = (x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]);
            let r2 = xx * xx + yy * yy + zz * zz;
            let r = r2.sqrt();
            let v = pair_curvature(xx, yy, zz, i, j);
            let g = gmm(i, j, r, r2);
            let hxx = g * xx * xx / r2 - v[0];
            let hxy = g * xx * yy / r2 - v[1];
            let hxz = g * xx * zz / r2 - v[2];
            let hyy = g * yy * yy / r2 - v[3];
            let hyz = g * yy * zz / r2 - v[4];
            let hzz = g * zz * zz / r2 - v[5];
            for a in [i, j] {
                h[(3 * a, 3 * a)] += hxx;
                h[(3 * a + 1, 3 * a)] += hxy;
                h[(3 * a + 1, 3 * a + 1)] += hyy;
                h[(3 * a + 2, 3 * a)] += hxz;
                h[(3 * a + 2, 3 * a + 1)] += hyz;
                h[(3 * a + 2, 3 * a + 2)] += hzz;
            }
            h[(3 * i, 3 * j)] -= hxx;
            h[(3 * i, 3 * j + 1)] -= hxy;
            h[(3 * i, 3 * j + 2)] -= hxz;
            h[(3 * i + 1, 3 * j + 1)] -= hxy;
            h[(3 * i + 1, 3 * j + 1)] -= hyy;
            h[(3 * i + 1, 3 * j + 2)] -= hyz;
            h[(3 * i + 2, 3 * j + 1)] -= hyz;
            h[(3 * i + 2, 3 * j + 2)] -= hzz;
        }
    }
}

/// MultiOptPy's D2 pair curvature (`VDW_xx`, `VDW_xy`, ...).
pub(crate) fn d2_curvature(
    elements: &[Element],
) -> impl Fn(f64, f64, f64, usize, usize) -> [f64; 6] + '_ {
    move |xx, yy, zz, i, j| {
        let c6 = (mop_d2_c6(elements[i]) * mop_d2_c6(elements[j])).sqrt();
        let r0 = mop_d2_vdw(elements[i]) + mop_d2_vdw(elements[j]);
        [
            vdw_isotropic(xx, yy, zz, c6, r0),
            vdw_anisotropic(xx, yy, zz, c6, r0),
            vdw_anisotropic(xx, zz, yy, c6, r0),
            vdw_isotropic(yy, xx, zz, c6, r0),
            vdw_anisotropic(yy, zz, xx, c6, r0),
            vdw_isotropic(zz, xx, yy, c6, r0),
        ]
    }
}

/// The force constants that differ between MultiOptPy's Swart- and Lindh-type models.
pub(crate) struct Variant<'a> {
    /// Angle constant of `(i, j, k)` (center `i`) from `r_ij`, `r_ik`.
    pub angle: &'a dyn Fn(usize, usize, usize, f64, f64) -> f64,
    /// Torsion constant of `(i, j, k, l)` from `r_ij`, `r_jk`, `r_kl`.
    pub torsion: &'a dyn Fn([usize; 4], [f64; 3]) -> f64,
    /// Out-of-plane constant of `(i, j, k, l)` (center `i`) from `r_ij`, `r_ik`, `r_il`.
    pub oop: &'a dyn Fn([usize; 4], [f64; 3]) -> f64,
}

fn add_cross(h: &mut Mat<f64>, a: usize, b: usize, sa: V3, sb: V3, k: f64) {
    // MultiOptPy writes into the block of the larger index first.
    for l in 0..3 {
        for m in 0..3 {
            if a > b {
                h[(3 * a + l, 3 * b + m)] += k * sa[l] * sb[m];
            } else {
                h[(3 * b + l, 3 * a + m)] += k * sb[l] * sa[m];
            }
        }
    }
}

fn add_self_lower(h: &mut Mat<f64>, a: usize, s: V3, k: f64) {
    for l in 0..3 {
        for m in 0..l {
            h[(3 * a + l, 3 * a + m)] += k * s[l] * s[m];
        }
    }
}

pub(crate) fn angle_part(h: &mut Mat<f64>, x: &[V3], v: &Variant<'_>) {
    let n = x.len();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let rij_v = sub(x[i], x[j]);
            let r_ij2 = dot(rij_v, rij_v);
            let r_ij = r_ij2.sqrt();
            for k in 0..j {
                if i == k {
                    continue;
                }
                let rik_v = sub(x[i], x[k]);
                let r_ik2 = dot(rik_v, rik_v);
                let r_ik = r_ik2.sqrt();
                let check = dot(rij_v, rik_v) / (r_ij * r_ik);
                if (check - 1.0).abs() < EPS {
                    continue;
                }
                let rjk_v = sub(x[j], x[k]);
                let r_jk2 = dot(rjk_v, rjk_v);
                let r_jk = r_jk2.sqrt();
                let g_jk = (v.angle)(i, j, k, r_ij, r_ik);
                let (xij, yij, zij) = (rij_v[0], rij_v[1], rij_v[2]);
                let (xik, yik, zik) = (rik_v[0], rik_v[1], rik_v[2]);
                let rc2 = (yij * zik - zij * yik).powi(2)
                    + (zij * xik - xij * zik).powi(2)
                    + (xij * yik - yij * xik).powi(2);
                let r_cross = if rc2 < 1.0e-12 { 0.0 } else { rc2.sqrt() };
                if !(r_ik > EPS && r_ij > EPS && r_jk > EPS) {
                    continue;
                }
                let cos = (r_ij2 + r_ik2 - r_jk2) / (2.0 * r_ij * r_ik);
                let sin = r_cross / (r_ij * r_ik);
                let mut apply = |sj: V3, sk: V3| {
                    let si = [-sj[0] - sk[0], -sj[1] - sk[1], -sj[2] - sk[2]];
                    add_cross(h, i, j, si, sj, g_jk);
                    add_cross(h, i, k, si, sk, g_jk);
                    add_cross(h, j, k, sj, sk, g_jk);
                    add_self_lower(h, j, sj, g_jk);
                    add_self_lower(h, i, si, g_jk);
                    add_self_lower(h, k, sk, g_jk);
                };
                if sin > EPS {
                    let sj = [
                        (xij / r_ij * cos - xik / r_ik) / (r_ij * sin),
                        (yij / r_ij * cos - yik / r_ik) / (r_ij * sin),
                        (zij / r_ij * cos - zik / r_ik) / (r_ij * sin),
                    ];
                    let sk = [
                        (xik / r_ik * cos - xij / r_ij) / (r_ik * sin),
                        (yik / r_ik * cos - yij / r_ij) / (r_ik * sin),
                        (zik / r_ik * cos - zij / r_ij) / (r_ik * sin),
                    ];
                    apply(sj, sk);
                } else {
                    let (v1, v2) = if yij.abs() < EPS && zij.abs() < EPS {
                        (
                            [-yij, xij, 0.0],
                            [-xij * zij, -yij * zij, xij * xij + yij * yij],
                        )
                    } else {
                        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
                    };
                    for w in [v1, v2] {
                        let r1 = dot(w, w).sqrt();
                        let c = [w[0] / r1, w[1] / r1, w[2] / r1];
                        let sj = [-c[0] / r_ij, -c[1] / r_ij, -c[2] / r_ij];
                        let sk = [-c[0] / r_ik, -c[1] / r_ik, -c[2] / r_ik];
                        apply(sj, sk);
                    }
                }
            }
        }
    }
}

pub(crate) fn add_quad(h: &mut Mat<f64>, at: [usize; 4], s: [V3; 4], k: f64) {
    let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for n in 0..3 {
        for m in 0..3 {
            for &(p, q) in &pairs {
                h[(3 * at[p] + n, 3 * at[q] + m)] += k * s[p][n] * s[q][m];
            }
        }
    }
    for n in 0..3 {
        for m in 0..n {
            for p in 0..4 {
                h[(3 * at[p] + n, 3 * at[p] + m)] += k * s[p][n] * s[p][m];
            }
        }
    }
}

pub(crate) fn torsion_part(h: &mut Mat<f64>, x: &[V3], v: &Variant<'_>) {
    let n = x.len();
    let cos_max = (35.0 / 180.0 * PI).cos();
    for j in 0..n {
        for k in 0..n {
            if j >= k {
                continue;
            }
            for i in 0..n {
                let ij = n * j + (i + 1);
                if i >= j || i >= k {
                    continue;
                }
                for ll in 0..n {
                    let kl = n * k + (ll + 1);
                    if ij <= kl || ll >= i || ll >= j || ll >= k {
                        continue;
                    }
                    let (rij, rjk, rkl) = (sub(x[i], x[j]), sub(x[j], x[k]), sub(x[k], x[ll]));
                    let (a2, b2, c2) = (dot(rij, rij), dot(rjk, rjk), dot(rkl, rkl));
                    if (dot(rij, rjk) / (a2 * b2).sqrt()).abs() > cos_max {
                        continue;
                    }
                    if (dot(rkl, rjk) / (c2 * b2).sqrt()).abs() > cos_max {
                        continue;
                    }
                    let t = (v.torsion)([i, j, k, ll], [a2.sqrt(), b2.sqrt(), c2.sqrt()]);
                    let (_, c) = torsion2([x[i], x[j], x[k], x[ll]]);
                    add_quad(h, [i, j, k, ll], c, t);
                }
            }
        }
    }
}

pub(crate) fn out_of_plane_part(h: &mut Mat<f64>, x: &[V3], v: &Variant<'_>) {
    let n = x.len();
    for i in 0..n {
        for j in 0..n {
            if i >= j {
                continue;
            }
            for k in 0..n {
                let ij = n * j + (i + 1);
                if i >= k || j >= k {
                    continue;
                }
                for ll in 0..n {
                    let kl = n * k + (ll + 1);
                    if i >= ll || j >= ll || k >= ll || ij <= kl {
                        continue;
                    }
                    let (rij, rik, ril) = (sub(x[i], x[j]), sub(x[i], x[k]), sub(x[i], x[ll]));
                    let (a2, b2, c2) = (dot(rij, rij), dot(rik, rik), dot(ril, ril));
                    let near = |c: f64| (c.abs() - 1.0).abs() < 1.0e-1;
                    if near(dot(rij, rik) / (a2 * b2).sqrt())
                        || near(dot(rij, ril) / (a2 * c2).sqrt())
                        || near(dot(rik, ril) / (b2 * c2).sqrt())
                    {
                        continue;
                    }
                    let t = (v.oop)([i, j, k, ll], [a2.sqrt(), b2.sqrt(), c2.sqrt()]);
                    let (_, c) = outofplane2([x[j], x[k], x[ll], x[i]]);
                    add_quad(h, [i, j, k, ll], c, t);
                }
            }
        }
    }
}

/// `if |H_ij| < 1e-10: H_ij = H_ji`, in place, row by row.
pub(crate) fn fill_small_from_mirror(h: &mut Mat<f64>) {
    let n = h.nrows();
    for i in 0..n {
        for j in 0..n {
            if h[(i, j)].abs() < 1.0e-10 {
                h[(i, j)] = h[(j, i)];
            }
        }
    }
}

/// The unprojected Hessian of a Swart-type model with the given pair dispersion.
pub(crate) fn swart_family_raw(
    elements: &[Element],
    x: &[V3],
    pair_curvature: &dyn Fn(f64, f64, f64, usize, usize) -> [f64; 6],
) -> Mat<f64> {
    let n = x.len();
    let l = lengths(elements);
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    let gmm = |i: usize, j: usize, r: f64, _r2: f64| {
        KR * bond_fc(1.0, (l.cov)(i, j), r) + KD * vdw_fc(5.0, (l.vdw)(i, j), r)
    };
    bond_part(&mut h, x, &gmm, pair_curvature);
    let g = |i: usize, j: usize, r: f64| g_pair(&l, i, j, r);
    let angle =
        |i: usize, j: usize, k: usize, r_ij: f64, r_ik: f64| KF * g(i, j, r_ij) * g(i, k, r_ik);
    let torsion = |q: [usize; 4], r: [f64; 3]| {
        KT * g(q[0], q[1], r[0]) * g(q[1], q[2], r[1]) * g(q[2], q[3], r[2])
    };
    let oop = |q: [usize; 4], r: [f64; 3]| {
        KT * g(q[0], q[1], r[0]) * g(q[0], q[2], r[1]) * g(q[0], q[3], r[2])
    };
    let v = Variant {
        angle: &angle,
        torsion: &torsion,
        oop: &oop,
    };
    angle_part(&mut h, x, &v);
    torsion_part(&mut h, x, &v);
    out_of_plane_part(&mut h, x, &v);
    fill_small_from_mirror(&mut h);
    h
}

/// MultiOptPy's `SwartD2ApproxHessian.main`.
pub(crate) fn swartd2(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let curvature = d2_curvature(&sys.elements);
    project_tr_rot(&swart_family_raw(&sys.elements, &x, &curvature), &x)
}
