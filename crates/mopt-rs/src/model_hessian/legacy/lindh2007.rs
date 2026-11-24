//! MultiOptPy's Lindh (2007) model Hessian (`ModelHessian/lindh2007d2.py`, the names
//! `lindh2007` and `lindh2007d2`), reproduced as it is.
//!
//! xtb-style traversal of all pairs, angles, torsions, and out-of-plane quadruples (as in
//! `swartd2`, including its index slips) with Lindh's 2007 constants (`kr = 0.45`, `kf = 0.10`,
//! `kt = 0.0025`, `ko = 0.16`, `kd = 0.05`), exponents `exp(alpha (r_cov^2 - r^2))` on covalent
//! radii, a Gaussian van der Waals term `exp(-16 (R_vdW - r)^2)`, and MultiOptPy's D2 pair
//! curvature. Kept MultiOptPy behavior: the angle constant multiplies `g_ij + c d_ij` by
//! `g_ik c d_ik` (a product where Lindh adds), torsions use the covalent length of the first
//! pair for all three bonds and out-of-plane terms that of the second pair for the third, the
//! element rows are H / He-F / Ne and heavier, and the final Hessian is
//! `0.1 exp(-|g|^2) |H|` (absolute eigenvalues) after projection.
//!
//! MultiOptPy: `ModelHessian/lindh2007d2.py:Lindh2007D2ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::common::{coords, cov_radius, mop_vdw_distance_bohr, project_tr_rot, V3};
use super::swartd2::{
    angle_part, bond_part, d2_curvature, fill_small_from_mirror, out_of_plane_part, torsion_part,
    Variant,
};
use crate::domain::{Element, MolecularSystem};
use crate::error::MoptResult;
use crate::math::linalg::symmetric_eigen;

pub(crate) const KR: f64 = 0.45;
pub(crate) const KF: f64 = 0.10;
pub(crate) const KT: f64 = 0.0025;
pub(crate) const KO: f64 = 0.16;
pub(crate) const KD: f64 = 0.05;

pub(crate) const A_AV: [[f64; 3]; 3] = [
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.1200],
    [0.3949, 0.1200, 0.0600],
];
pub(crate) const D_AV: [[f64; 3]; 3] = [
    [0.0000, 3.6000, 3.6000],
    [3.6000, 5.3000, 5.3000],
    [3.6000, 5.3000, 5.3000],
];

/// `select_idx`: H 0, He-F 1, Ne and heavier 2.
pub(crate) fn row(el: Element) -> usize {
    match el.atomic_number() {
        1 => 0,
        2..=9 => 1,
        _ => 2,
    }
}

/// The pair functions shared by the Lindh 2007 variants.
pub(crate) struct Pairs<'a> {
    pub el: &'a [Element],
}

impl Pairs<'_> {
    pub(crate) fn alpha(&self, i: usize, j: usize) -> f64 {
        A_AV[row(self.el[i])][row(self.el[j])]
    }
    pub(crate) fn d0(&self, i: usize, j: usize) -> f64 {
        D_AV[row(self.el[i])][row(self.el[j])]
    }
    pub(crate) fn rcov(&self, i: usize, j: usize) -> f64 {
        cov_radius(self.el[i]) + cov_radius(self.el[j])
    }
    pub(crate) fn vdw(&self, i: usize, j: usize) -> f64 {
        mop_vdw_distance_bohr(self.el[i]) + mop_vdw_distance_bohr(self.el[j])
    }
    /// `calc_force_const(alpha, r0, r^2)`.
    pub(crate) fn fc(alpha: f64, r0: f64, r2: f64) -> f64 {
        (alpha * (r0 * r0 - r2)).exp()
    }
    /// `calc_vdw_force_const(4.0, R, r)`.
    pub(crate) fn vdw_fc(rvdw: f64, r: f64) -> f64 {
        (-4.0 * 4.0 * (rvdw - r).powi(2)).exp()
    }
    /// `g` of a pair with an explicit covalent length (`kd` scales the van der Waals term).
    pub(crate) fn g(&self, i: usize, j: usize, rcov: f64, r: f64, kd: f64) -> f64 {
        Self::fc(self.alpha(i, j), rcov, r * r) + 0.5 * kd * Self::vdw_fc(self.vdw(i, j), r)
    }
}

/// The Lindh 2007 force constants; `kd` is MultiOptPy's 0.05 for every variant.
pub(crate) fn lindh2007_raw(
    el: &[Element],
    x: &[V3],
    pair_curvature: &dyn Fn(f64, f64, f64, usize, usize) -> [f64; 6],
    bond_dispersion: &dyn Fn(usize, usize, f64) -> f64,
) -> Mat<f64> {
    let n = x.len();
    let p = Pairs { el };
    let c = 0.5 * KD / KR;
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    let gmm = |i: usize, j: usize, r: f64, r2: f64| {
        KR * Pairs::fc(p.alpha(i, j), p.rcov(i, j), r2) + bond_dispersion(i, j, r)
    };
    bond_part(&mut h, x, &gmm, pair_curvature);
    let angle = |i: usize, j: usize, k: usize, r_ij: f64, r_ik: f64| {
        let g_ij = p.g(i, j, p.rcov(i, j), r_ij, KD);
        let g_ik = p.g(i, k, p.rcov(i, k), r_ik, KD);
        KF * (g_ij + c * p.d0(i, j)) * (g_ik * c * p.d0(i, k))
    };
    let torsion = |q: [usize; 4], r: [f64; 3]| {
        let [i, j, k, l] = q;
        let rc = p.rcov(i, j);
        let g_ij = p.g(i, j, rc, r[0], KD);
        let g_jk = p.g(j, k, rc, r[1], KD);
        let g_kl = p.g(k, l, rc, r[2], KD);
        KT * (g_ij * c * p.d0(i, j)) * (g_jk * c * p.d0(j, k)) * (g_kl * c * p.d0(k, l))
    };
    let oop = |q: [usize; 4], r: [f64; 3]| {
        let [i, j, k, l] = q;
        let g_ij = p.g(i, j, p.rcov(i, j), r[0], 0.0);
        let g_ik = p.g(i, k, p.rcov(i, k), r[1], 0.0);
        let g_il = p.g(i, l, p.rcov(i, k), r[2], 0.0);
        KO * g_ij * g_ik * g_il
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

/// `V diag(|w| s) V^T` with `s = 0.1 exp(-|g|^2)`.
pub(crate) fn absolute_scaled(h: &Mat<f64>, gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
    let g2: f64 = gradient.iter().map(|v| v * v).sum();
    let s = 0.1 * (-g2).exp();
    let (w, v) = symmetric_eigen(h)?;
    let n = h.nrows();
    let mut out = Mat::<f64>::zeros(n, n);
    for k in 0..n {
        let f = w[k].abs() * s;
        for i in 0..n {
            for j in 0..n {
                out[(i, j)] += v[(i, k)] * f * v[(j, k)];
            }
        }
    }
    Ok(out)
}

/// MultiOptPy's `Lindh2007D2ApproxHessian.main`.
pub(crate) fn lindh2007(sys: &MolecularSystem, gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
    let x = coords(sys);
    let el = &sys.elements;
    let p = Pairs { el };
    let curvature = d2_curvature(el);
    let disp = |i: usize, j: usize, r: f64| KD * Pairs::vdw_fc(p.vdw(i, j), r);
    let h = lindh2007_raw(el, &x, &curvature, &disp);
    absolute_scaled(&project_tr_rot(&h, &x), gradient)
}
