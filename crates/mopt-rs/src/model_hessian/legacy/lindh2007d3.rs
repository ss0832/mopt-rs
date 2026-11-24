//! MultiOptPy's Lindh (2007) model Hessians with its D3 and D4 terms
//! (`ModelHessian/lindh2007d3.py`, `lindh2007d4.py`), reproduced as they are.
//!
//! A rewrite of `lindh2007d2.py`: the bond blocks are complete (no index slips), the angle
//! constant is `kf (g_ij + c d_ij)(g_ik + c d_ik)`, torsions run over loops of their own with
//! per-pair covalent lengths, and out-of-plane terms over `i < j < k < l`. The pair term uses
//! `C6`, `C8`, and `R0` from [`Dispersion`] (`R0` from UFF radii divided by the Bohr radius once
//! more although they are already in Bohr), a damping `r^n / (r^n + ((R0 + a1) a2)^n)`, the
//! energy-like constant `s6 C6 f6 / r^6 + s8 C8 f8 / r^8` for pairs beyond twice their covalent
//! length, and radial "gradient" components subtracted from each pair block. The lower triangle
//! wins in the final symmetrization; the result is `0.1 exp(-|g|^2) |H|` after projection.
//!
//! The D4 version uses the harmonic mean of D2 `C6`, D4 `r4r2`, charge scaling
//! `exp(-3 (q_i^2 + q_j^2))` with charges `0.1 tanh(0.2 dEN)` per bond (mean removed), scales
//! torsions by 1.2 (both central atoms with CN near 4) or 0.8 (both near or below 3) and
//! out-of-plane terms by 1.2 (center with CN near 3), and adds a diagonal three-body term.
//!
//! MultiOptPy: `ModelHessian/lindh2007d3.py:Lindh2007D3ApproxHessian` and
//!  `ModelHessian/lindh2007d4.py:Lindh2007D4ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::{Col, Mat};

use super::common::{
    coords, cov_radius, dot, mop_vdw_distance_bohr, norm, outofplane2, project_tr_rot, sub,
    torsion2, V3,
};
use super::fischerd3::d3_r4r2;
use super::lindh2007::{absolute_scaled, Pairs, KD, KF, KO, KR, KT};
use super::swartd2::{add_quad, angle_part, mop_d2_c6, Variant};
use crate::domain::{Element, MolecularSystem};
use crate::error::MoptResult;
use crate::parameters::multioptpy_tables::{
    MOP_BOHR2ANGSTROMS, MOP_D4_DEFAULT_ELECTRONEGATIVITY, MOP_D4_DEFAULT_R4R2,
    MOP_D4_ELECTRONEGATIVITY, MOP_D4_GA, MOP_D4_GC, MOP_D4_R4R2,
};

/// The dispersion of one variant.
pub(crate) struct Dispersion<'a> {
    /// `(C6, C8, R0, charge scaling)` of a pair.
    pub params: &'a dyn Fn(usize, usize) -> (f64, f64, f64, f64),
    pub a1: f64,
    pub a2: f64,
    pub s6: f64,
    pub s8: f64,
    /// Torsion factor from the central atoms `(j, k)`.
    pub torsion_scale: &'a dyn Fn(usize, usize) -> f64,
    /// Out-of-plane factor from the center `i`.
    pub oop_scale: &'a dyn Fn(usize) -> f64,
}

impl Dispersion<'_> {
    /// `calc_d3_force_const` / `calc_d4_force_const`.
    fn force_const(&self, r: f64, i: usize, j: usize) -> f64 {
        let (c6, c8, r0, q) = (self.params)(i, j);
        let ra = r0 + self.a1;
        let f6 = r.powi(6) / (r.powi(6) + (ra * self.a2).powi(6));
        let f8 = r.powi(8) / (r.powi(8) + (ra * self.a2).powi(8));
        let e6 = -self.s6 * (c6 * q) * f6 / r.powi(6);
        let e8 = -self.s8 * (c8 * q) * f8 / r.powi(8);
        -(e6 + e8)
    }

    /// `calc_d3_gradient_components` / `calc_d4_gradient_components`.
    fn components(&self, x: f64, y: f64, z: f64, i: usize, j: usize) -> [f64; 6] {
        let (c6, c8, r0, q) = (self.params)(i, j);
        let r2 = x * x + y * y + z * z;
        let r = r2.sqrt();
        if r < 0.1 {
            return [0.0; 6];
        }
        let ra = r0 + self.a1;
        let (b6, b8) = ((ra * self.a2).powi(6), (ra * self.a2).powi(8));
        let (r6, r8) = (r.powi(6), r.powi(8));
        let d6 = r6 / (r6 + b6);
        let dd6 = 6.0 * r6 * b6 / ((r6 + b6).powi(2) * r);
        let d8 = r8 / (r8 + b8);
        let dd8 = 8.0 * r8 * b8 / ((r8 + b8).powi(2) * r);
        let f6 = self.s6 * (c6 * q) * (6.0 * d6 / r.powi(7) + dd6 / r.powi(6));
        let f8 = self.s8 * (c8 * q) * (8.0 * d8 / r.powi(9) + dd8 / r.powi(8));
        let s = (f6 + f8) / r;
        [
            s * x * x / r2,
            s * x * y / r2,
            s * x * z / r2,
            s * y * y / r2,
            s * y * z / r2,
            s * z * z / r2,
        ]
    }

    /// `g` of a pair: the Lindh exponential plus `kd/2` times the dispersion constant beyond
    /// twice the covalent length.
    fn g(&self, p: &Pairs<'_>, i: usize, j: usize, r: f64, kd: f64) -> f64 {
        let rc = p.rcov(i, j);
        let mut v = Pairs::fc(p.alpha(i, j), rc, r * r);
        if r > 2.0 * rc {
            v += 0.5 * kd * self.force_const(r, i, j);
        }
        v
    }
}

fn r0_double_converted(a: Element, b: Element) -> f64 {
    mop_vdw_distance_bohr(a) / MOP_BOHR2ANGSTROMS + mop_vdw_distance_bohr(b) / MOP_BOHR2ANGSTROMS
}

fn bonds(h: &mut Mat<f64>, p: &Pairs<'_>, d: &Dispersion<'_>, x: &[V3]) {
    let n = x.len();
    for i in 0..n {
        for j in 0..i {
            let (xx, yy, zz) = (x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]);
            let r2 = xx * xx + yy * yy + zz * zz;
            let r = r2.sqrt();
            let rc = p.rcov(i, j);
            let mut gmm = KR * Pairs::fc(p.alpha(i, j), rc, r2);
            if r > 2.0 * rc {
                gmm += KD * d.force_const(r, i, j);
            }
            let c = d.components(xx, yy, zz, i, j);
            let hxx = gmm * xx * xx / r2 - c[0];
            let hxy = gmm * xx * yy / r2 - c[1];
            let hxz = gmm * xx * zz / r2 - c[2];
            let hyy = gmm * yy * yy / r2 - c[3];
            let hyz = gmm * yy * zz / r2 - c[4];
            let hzz = gmm * zz * zz / r2 - c[5];
            for a in [i, j] {
                h[(3 * a, 3 * a)] += hxx;
                h[(3 * a + 1, 3 * a)] += hxy;
                h[(3 * a + 1, 3 * a + 1)] += hyy;
                h[(3 * a + 2, 3 * a)] += hxz;
                h[(3 * a + 2, 3 * a + 1)] += hyz;
                h[(3 * a + 2, 3 * a + 2)] += hzz;
            }
            let blk = [[hxx, hxy, hxz], [hxy, hyy, hyz], [hxz, hyz, hzz]];
            for a in 0..3 {
                for b in 0..3 {
                    h[(3 * i + a, 3 * j + b)] -= blk[a][b];
                }
            }
        }
    }
}

fn torsions(h: &mut Mat<f64>, p: &Pairs<'_>, d: &Dispersion<'_>, x: &[V3]) {
    let n = x.len();
    let c = 0.5 * KD / KR;
    let cos_max = (35.0 / 180.0 * PI).cos();
    for j in 0..n {
        for k in j + 1..n {
            for i in 0..j {
                if i == k {
                    continue;
                }
                for l in k + 1..n {
                    if l == i || l == j {
                        continue;
                    }
                    let (rij, rjk, rkl) = (sub(x[i], x[j]), sub(x[j], x[k]), sub(x[k], x[l]));
                    let (a2, b2, c2) = (dot(rij, rij), dot(rjk, rjk), dot(rkl, rkl));
                    if (dot(rij, rjk) / (a2 * b2).sqrt()).abs() > cos_max
                        || (dot(rkl, rjk) / (c2 * b2).sqrt()).abs() > cos_max
                    {
                        continue;
                    }
                    let t = KT
                        * (d.g(p, i, j, a2.sqrt(), KD) * c * p.d0(i, j))
                        * (d.g(p, j, k, b2.sqrt(), KD) * c * p.d0(j, k))
                        * (d.g(p, k, l, c2.sqrt(), KD) * c * p.d0(k, l))
                        * (d.torsion_scale)(j, k);
                    let (_, b) = torsion2([x[i], x[j], x[k], x[l]]);
                    add_quad(h, [i, j, k, l], b, t);
                }
            }
        }
    }
}

fn out_of_plane(h: &mut Mat<f64>, p: &Pairs<'_>, d: &Dispersion<'_>, x: &[V3]) {
    let n = x.len();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                for l in k + 1..n {
                    let (rij, rik, ril) = (sub(x[i], x[j]), sub(x[i], x[k]), sub(x[i], x[l]));
                    let (a, b, cc) = (
                        dot(rij, rij).sqrt(),
                        dot(rik, rik).sqrt(),
                        dot(ril, ril).sqrt(),
                    );
                    let near = |v: f64| (v.abs() - 1.0).abs() < 0.1;
                    if near(dot(rij, rik) / (a * b))
                        || near(dot(rij, ril) / (a * cc))
                        || near(dot(rik, ril) / (b * cc))
                    {
                        continue;
                    }
                    let t = KO
                        * d.g(p, i, j, a, 0.0)
                        * d.g(p, i, k, b, 0.0)
                        * d.g(p, i, l, cc, 0.0)
                        * (d.oop_scale)(i);
                    let (_, s) = outofplane2([x[j], x[k], x[l], x[i]]);
                    add_quad(h, [i, j, k, l], s, t);
                }
            }
        }
    }
}

/// The unprojected Hessian of `lindh2007d3.py` / `lindh2007d4.py` (without the D4 three-body
/// term and the final symmetrization).
fn raw(el: &[Element], x: &[V3], d: &Dispersion<'_>) -> Mat<f64> {
    let n = x.len();
    let p = Pairs { el };
    let c = 0.5 * KD / KR;
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    bonds(&mut h, &p, d, x);
    let angle = |i: usize, j: usize, k: usize, r_ij: f64, r_ik: f64| {
        KF * (d.g(&p, i, j, r_ij, KD) + c * p.d0(i, j)) * (d.g(&p, i, k, r_ik, KD) + c * p.d0(i, k))
    };
    let unused = |_: [usize; 4], _: [f64; 3]| 0.0;
    let v = Variant {
        angle: &angle,
        torsion: &unused,
        oop: &unused,
    };
    angle_part(&mut h, x, &v);
    torsions(&mut h, &p, d, x);
    out_of_plane(&mut h, &p, d, x);
    h
}

/// `if |H_ij| < 1e-10: H_ij = H_ji else: H_ji = H_ij` over the lower triangle.
pub(crate) fn lower_wins(h: &mut Mat<f64>) {
    let n = h.nrows();
    for i in 0..n {
        for j in 0..i {
            if h[(i, j)].abs() < 1e-10 {
                h[(i, j)] = h[(j, i)];
            } else {
                h[(j, i)] = h[(i, j)];
            }
        }
    }
}

/// MultiOptPy's `Lindh2007D3ApproxHessian.main`.
pub(crate) fn lindh2007d3(sys: &MolecularSystem, gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
    let x = coords(sys);
    let el = &sys.elements;
    let params = |i: usize, j: usize| {
        let c6 = (mop_d2_c6(el[i]) * mop_d2_c6(el[j])).sqrt();
        let c8 = 3.0 * c6 * (d3_r4r2(el[i]) * d3_r4r2(el[j])).sqrt();
        (c6, c8, r0_double_converted(el[i], el[j]), 1.0)
    };
    let d = Dispersion {
        params: &params,
        a1: 0.4289,
        a2: 4.4407,
        s6: 1.0,
        s8: 0.7875,
        torsion_scale: &|_, _| 1.0,
        oop_scale: &|_| 1.0,
    };
    let mut h = raw(el, &x, &d);
    lower_wins(&mut h);
    absolute_scaled(&project_tr_rot(&h, &x), gradient)
}

fn d4_value(t: &[f64], el: Element, default: f64) -> f64 {
    match t.get(el.atomic_number() as usize) {
        Some(v) if !v.is_nan() => *v,
        _ => default,
    }
}

/// `estimate_atomic_charges` of `lindh2007d4.py`.
fn tanh_charges(el: &[Element], x: &[V3]) -> Vec<f64> {
    let n = x.len();
    let mut q = vec![0.0; n];
    let en = |e: Element| {
        d4_value(
            &MOP_D4_ELECTRONEGATIVITY,
            e,
            MOP_D4_DEFAULT_ELECTRONEGATIVITY,
        )
    };
    for i in 0..n {
        for j in i + 1..n {
            let rs = (cov_radius(el[i]) + cov_radius(el[j])) * 1.0;
            if norm(sub(x[i], x[j])) < rs * 1.3 {
                let t = 0.1 * (0.2 * (en(el[j]) - en(el[i]))).tanh();
                q[i] += t;
                q[j] -= t;
            }
        }
    }
    let mean = q.iter().sum::<f64>() / n.max(1) as f64;
    q.iter().map(|v| v - mean).collect()
}

/// `calculate_coordination_numbers` of `lindh2007d4.py`.
fn coordination(el: &[Element], x: &[V3]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let rc = cov_radius(el[i]) + cov_radius(el[j]);
                    1.0 / (1.0 + (-16.0 * (rc / norm(sub(x[i], x[j])) - 1.0)).exp())
                })
                .sum()
        })
        .collect()
}

/// MultiOptPy's `Lindh2007D4ApproxHessian.main`.
pub(crate) fn lindh2007d4(sys: &MolecularSystem, gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
    let x = coords(sys);
    let el = &sys.elements;
    let n = x.len();
    let q = tanh_charges(el, &x);
    let cn = coordination(el, &x);
    let r4r2 = |e: Element| d4_value(&MOP_D4_R4R2, e, MOP_D4_DEFAULT_R4R2);
    let params = |i: usize, j: usize| {
        let (c1, c2) = (mop_d2_c6(el[i]), mop_d2_c6(el[j]));
        let c6 = 2.0 * c1 * c2 / (c1 + c2);
        let c8 = 3.0 * c6 * (r4r2(el[i]) * r4r2(el[j])).sqrt();
        let qs = (-MOP_D4_GA * (q[i] * q[i] + q[j] * q[j])).exp();
        (c6, c8, r0_double_converted(el[i], el[j]), qs)
    };
    let torsion_scale = |j: usize, k: usize| {
        if (3.9..=4.1).contains(&cn[j]) && (3.9..=4.1).contains(&cn[k]) {
            1.2
        } else if cn[j] <= 3.1 && cn[k] <= 3.1 {
            0.8
        } else {
            1.0
        }
    };
    let oop_scale = |i: usize| {
        if (2.9..=3.1).contains(&cn[i]) {
            1.2
        } else {
            1.0
        }
    };
    let d = Dispersion {
        params: &params,
        a1: 0.4171,
        a2: 4.5337,
        s6: 1.0,
        s8: 1.03683,
        torsion_scale: &torsion_scale,
        oop_scale: &oop_scale,
    };
    let mut h = raw(el, &x, &d);
    if n >= 3 {
        let near = |a: usize, b: usize| {
            norm(sub(x[a], x[b])) < (cov_radius(el[a]) + cov_radius(el[b])) * 1.5
        };
        for i in 0..n {
            for j in i + 1..n {
                if near(i, j) {
                    continue;
                }
                for k in j + 1..n {
                    if near(i, k) || near(j, k) {
                        continue;
                    }
                    let (rij, rik, rjk) = (
                        norm(sub(x[i], x[j])),
                        norm(sub(x[i], x[k])),
                        norm(sub(x[j], x[k])),
                    );
                    let c9 = (params(i, j).0 * params(i, k).0 * params(j, k).0).sqrt();
                    let qs = (-MOP_D4_GC * (q[i] * q[i] + q[j] * q[j] + q[k] * q[k])).exp();
                    if rij > 15.0 || rik > 15.0 || rjk > 15.0 {
                        continue;
                    }
                    let s = 0.002 * 1.0 * qs * c9 / (rij * rik * rjk).powi(3);
                    for a in [i, j, k] {
                        for t in 0..3 {
                            h[(3 * a + t, 3 * a + t)] += s;
                        }
                    }
                }
            }
        }
    }
    lower_wins(&mut h);
    absolute_scaled(&project_tr_rot(&h, &x), gradient)
}
