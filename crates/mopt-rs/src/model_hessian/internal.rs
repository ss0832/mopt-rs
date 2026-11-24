//! Lindh-type model Hessians over all atom pairs, triples, and quadruples within a cutoff
//! (Lindh et al., Chem. Phys. Lett. 1995, 241, 423), following xtb's implementation
//! (`src/model_hessian/internal.f90`, LGPL-3.0): force constants `k rho_ij rho_jk ...` with pair
//! factors that decay with distance, and an optional D2 pair term.
//!
//! MultiOptPy: `ModelHessian/lindh.py`, `lindh2007d2.py`, `lindh2007d3.py`, `lindh2007d4.py`,
//!  `swartd2.py`, `swartd3.py`, `swartd4.py` (published models; MultiOptPy's in `legacy::`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::{Col, Mat};

use super::assemble::HessianAccumulator;
use super::{add_dispersion, Dispersion, ModelHessian};
use crate::domain::{AtomId, Element, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_grad_hessian, HyperDual, Scalar};
use crate::math::stable::logistic;
use crate::parameters::model_hessian::{
    covalent_radius_2009_bohr, lindh_row, mh_d2_c6, mh_d2_vdw_radius_bohr, LINDH_1995_AAV,
    LINDH_1995_RAV, LINDH_2007_AAV, LINDH_2007_DAV, LINDH_2007_RAV, MH_D2_DAMPING,
};

/// Force constants and cutoffs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InternalParams {
    /// Stretch force constant.
    pub kr: f64,
    /// Bend force constant.
    pub kf: f64,
    /// Torsion force constant.
    pub kt: f64,
    /// Out-of-plane force constant.
    pub ko: f64,
    /// Weight of the van der Waals decay term (used as `kd / kr`).
    pub kd: f64,
    /// Squared distance cutoff (Bohr^2) of pairs entering bends, torsions, out-of-plane terms.
    pub rcut2: f64,
    /// Scale of the D2 pair term.
    pub s6: f64,
}

impl InternalParams {
    /// xtb's defaults (`setparam.f90`).
    pub const XTB: Self = Self {
        kr: 0.40,
        kf: 0.13,
        kt: 0.0075,
        ko: 0.0,
        kd: 0.0,
        rcut2: 70.0,
        s6: 20.0,
    };
}

/// Distance decay of the pair factors.
#[derive(Debug, Clone, Copy, PartialEq)]
// The tables are small and copied rarely; boxing them would only add indirection.
#[allow(clippy::large_enum_variant)]
pub enum PairModel {
    /// `exp(alpha (r_ref^2 - r^2)) + s exp(-4 (d_ref - r)^2)` by row pair.
    Lindh {
        /// `r_ref`.
        rav: [[f64; 3]; 3],
        /// `alpha`.
        aav: [[f64; 3]; 3],
        /// `d_ref`.
        dav: [[f64; 3]; 3],
        /// Weight of the van der Waals term in out-of-plane factors.
        oop_dispersion: f64,
    },
    /// `exp(-(r / r_cov - 1)) + s exp(-5 (r_cov - r)^2)` with the 2009 covalent radii.
    Swart,
}

impl PairModel {
    fn factor(&self, a: Element, b: Element, r2: f64, scale: f64, oop: bool) -> f64 {
        let vdw = |alpha: f64, r0: f64| (-alpha * (r0 - r2.sqrt()).powi(2)).exp();
        match self {
            Self::Lindh {
                rav,
                aav,
                dav,
                oop_dispersion,
            } => {
                let (i, j) = (lindh_row(a), lindh_row(b));
                let base = (aav[i][j] * (rav[i][j] * rav[i][j] - r2)).exp();
                let s = if oop { scale * oop_dispersion } else { scale };
                base + s * vdw(4.0, dav[i][j])
            }
            Self::Swart => {
                let r0 = covalent_radius_2009_bohr(a) + covalent_radius_2009_bohr(b);
                (-(r2.sqrt() / r0 - 1.0)).exp() + scale * vdw(5.0, r0)
            }
        }
    }
}

/// A Lindh-type model.
#[derive(Debug, Clone)]
pub struct InternalModel {
    name: &'static str,
    /// Force constants.
    pub params: InternalParams,
    /// Pair factors.
    pub pair: PairModel,
    /// Dispersion pair term.
    pub dispersion: Dispersion,
}

const ZERO3: [[f64; 3]; 3] = [[0.0; 3]; 3];

impl InternalModel {
    /// Lindh 1995 as published: `k_r = 0.45`, `k_phi = 0.15`, `k_tau = 0.005`, no dispersion.
    pub fn lindh_1995() -> Self {
        Self {
            name: "lindh",
            params: InternalParams {
                kr: 0.45,
                kf: 0.15,
                kt: 0.005,
                ko: 0.0,
                kd: 0.0,
                ..InternalParams::XTB
            },
            pair: PairModel::Lindh {
                rav: LINDH_1995_RAV,
                aav: LINDH_1995_AAV,
                dav: ZERO3,
                oop_dispersion: 0.0,
            },
            dispersion: Dispersion::None,
        }
    }

    /// Lindh 2007 tables with the 2007 force constants (`0.45`, `0.10`, `0.0025`, out-of-plane
    /// `0.16`, van der Waals `0.05`) and the given dispersion term.
    pub fn lindh_2007(dispersion: Dispersion) -> Self {
        Self {
            name: "lindh2007",
            params: InternalParams {
                kr: 0.45,
                kf: 0.10,
                kt: 0.0025,
                ko: 0.16,
                kd: 0.05,
                ..InternalParams::XTB
            },
            pair: PairModel::Lindh {
                rav: LINDH_2007_RAV,
                aav: LINDH_2007_AAV,
                dav: LINDH_2007_DAV,
                oop_dispersion: 0.0,
            },
            dispersion: Dispersion::D2,
        }
        .with_dispersion(dispersion)
    }

    /// xtb's `lindh` model (2007 tables) with xtb's defaults.
    pub fn xtb_lindh() -> Self {
        Self {
            name: "xtb_lindh",
            params: InternalParams::XTB,
            pair: PairModel::Lindh {
                rav: LINDH_2007_RAV,
                aav: LINDH_2007_AAV,
                dav: LINDH_2007_DAV,
                oop_dispersion: 0.0,
            },
            dispersion: Dispersion::D2,
        }
    }

    /// xtb's `lindh-d2` model (1995 tables, its default) with xtb's defaults.
    pub fn xtb_lindh_d2() -> Self {
        Self {
            name: "xtb_lindh_d2",
            params: InternalParams::XTB,
            pair: PairModel::Lindh {
                rav: LINDH_1995_RAV,
                aav: LINDH_1995_AAV,
                dav: ZERO3,
                oop_dispersion: 1.0,
            },
            dispersion: Dispersion::D2,
        }
    }

    /// xtb's `swart` model with xtb's defaults.
    pub fn xtb_swart() -> Self {
        Self {
            name: "xtb_swart",
            params: InternalParams::XTB,
            pair: PairModel::Swart,
            dispersion: Dispersion::D2,
        }
    }

    /// The same model with another dispersion term.
    #[must_use]
    pub fn with_dispersion(mut self, dispersion: Dispersion) -> Self {
        self.dispersion = dispersion;
        self.name = match (self.name, dispersion) {
            ("lindh2007", Dispersion::D3) => "lindh2007d3",
            ("lindh2007", Dispersion::D4) => "lindh2007d4",
            ("xtb_swart", Dispersion::D3) => "swartd3",
            ("xtb_swart", Dispersion::D4) => "swartd4",
            (name, _) => name,
        };
        self
    }
}

/// `d2E/dd2` of the model D2 pair term `s6 C6 f(r) / r^6`, `f = 1 / (1 + exp(-20 (r/R0 - 1)))`
/// (xtb adds the negative of the attractive D2 Hessian).
fn d2_block(d: [f64; 3], c6: f64, s6: f64, r0: f64) -> [[f64; 3]; 3] {
    let (_, _, h) = eval_value_grad_hessian(&d, None, |x: &[HyperDual]| {
        let one = HyperDual::one();
        let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        let r = r2.sqrt();
        let f = logistic(HyperDual::from_f64(MH_D2_DAMPING) * (r / HyperDual::from_f64(r0) - one));
        HyperDual::from_f64(s6 * c6) * f / (r2 * r2 * r2)
    });
    std::array::from_fn(|p| std::array::from_fn(|q| h[(p, q)]))
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// xtb's frame of two unit vectors perpendicular to the unit axis `u`.
fn linear_frame(u: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let r = if u[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let p = dot(r, u);
    let e1 = [r[0] - p * u[0], r[1] - p * u[1], r[2] - p * u[2]];
    let n = norm(e1);
    let e1 = e1.map(|x| x / n);
    (e1, cross(u, e1))
}

impl ModelHessian for InternalModel {
    fn name(&self) -> &str {
        self.name
    }

    fn hessian(&self, sys: &MolecularSystem, _gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        let n = sys.n_atoms();
        let p = self.params;
        let el: Vec<Element> = sys.atoms().map(|a| sys.element(a)).collect();
        let id = AtomId::new;
        let mut acc = HessianAccumulator::new(sys);
        let kd = if p.kr != 0.0 { p.kd / p.kr } else { 0.0 };
        let pf = |i: usize, j: usize, r2: f64, s: f64, oop: bool| {
            self.pair.factor(el[i], el[j], r2, s, oop)
        };
        // Pair geometry and cutoff mask.
        let mut d = vec![[0.0; 3]; n * n];
        let mut r2 = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    d[i * n + j] = acc.delta(id(i), id(j));
                    r2[i * n + j] = dot(d[i * n + j], d[i * n + j]);
                }
            }
        }
        let near = |i: usize, j: usize| i != j && r2[i * n + j] <= p.rcut2;
        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| near(i, j)).collect())
            .collect();
        // Stretches (all pairs) and the D2 pair term.
        for i in 0..n {
            for j in 0..i {
                let rr = r2[i * n + j];
                acc.stretch(id(i), id(j), p.kr * pf(i, j, rr, kd, false));
                if self.dispersion == Dispersion::D2 && p.s6 != 0.0 {
                    let c6 = (mh_d2_c6(el[i]) * mh_d2_c6(el[j])).sqrt();
                    let rv = mh_d2_vdw_radius_bohr(el[i]) + mh_d2_vdw_radius_bohr(el[j]);
                    acc.pair(id(i), id(j), &d2_block(d[i * n + j], c6, p.s6, rv));
                }
            }
        }
        // Bends.
        if p.kf != 0.0 {
            for m in 0..n {
                for &i in &neighbors[m] {
                    let vmi = d[i * n + m];
                    let rmi = r2[i * n + m].sqrt();
                    if rmi <= 1e-10 {
                        continue;
                    }
                    for &j in neighbors[m].iter().filter(|&&j| j < i && near(j, i)) {
                        let vmj = d[j * n + m];
                        let rmj = r2[j * n + m].sqrt();
                        if rmj <= 1e-10 || r2[j * n + i].sqrt() <= 1e-10 {
                            continue;
                        }
                        let cosphi = dot(vmi, vmj) / (rmi * rmj);
                        if (cosphi - 1.0).abs() < 1e-12 {
                            continue;
                        }
                        let g = p.kf
                            * pf(m, i, r2[i * n + m], 0.5 * kd, false)
                            * pf(m, j, r2[j * n + m], 0.5 * kd, false);
                        let ui = vmi.map(|x| x / rmi);
                        let sinphi = norm(cross(ui, vmj.map(|x| x / rmj)));
                        if sinphi > 1e-10 {
                            acc.bend(id(i), id(m), id(j), g);
                        } else {
                            let (e1, e2) = linear_frame(ui);
                            acc.linear_bend(id(i), id(m), id(j), e1, g);
                            acc.linear_bend(id(i), id(m), id(j), e2, g);
                        }
                    }
                }
            }
        }
        // Torsions.
        let cos_max = (35.0 * PI / 180.0).cos();
        if p.kt != 0.0 {
            for j in 0..n {
                for &k in &neighbors[j] {
                    for &i in neighbors[j].iter().filter(|&&i| i != k && near(i, k)) {
                        let ij = n * j + i;
                        for &l in neighbors[k]
                            .iter()
                            .filter(|&&l| l != i && l != j && near(l, i) && near(l, j))
                        {
                            if ij <= n * k + l {
                                continue;
                            }
                            let rij = d[i * n + j];
                            let rjk = d[j * n + k];
                            let rkl = d[k * n + l];
                            let (rij2, rjk2, rkl2) = (r2[i * n + j], r2[j * n + k], r2[k * n + l]);
                            if (dot(rij, rjk) / (rij2 * rjk2).sqrt()).abs() > cos_max
                                || (dot(rkl, rjk) / (rkl2 * rjk2).sqrt()).abs() > cos_max
                            {
                                continue;
                            }
                            let t = p.kt
                                * pf(i, j, rij2, 0.5 * kd, false)
                                * pf(j, k, rjk2, 0.5 * kd, false)
                                * pf(k, l, rkl2, 0.5 * kd, false);
                            acc.torsion([id(i), id(j), id(k), id(l)], t);
                        }
                    }
                }
            }
        }
        // Out-of-plane terms (center i, out-of-plane atom j, plane atoms k, l).
        if p.ko != 0.0 {
            for i in 0..n {
                for &j in &neighbors[i] {
                    for &k in neighbors[i].iter().filter(|&&k| k != j && near(k, j)) {
                        for &l in neighbors[i]
                            .iter()
                            .filter(|&&l| l != j && l != k && near(l, k) && near(l, j))
                        {
                            let rij = d[i * n + j];
                            let rik = d[i * n + k];
                            let ril = d[i * n + l];
                            let (rij2, rik2, ril2) = (r2[i * n + j], r2[i * n + k], r2[i * n + l]);
                            let near_linear = |a: [f64; 3], b: [f64; 3], a2: f64, b2: f64| {
                                ((dot(a, b) / (a2 * b2).sqrt()).abs() - 1.0).abs() < 0.1
                            };
                            if near_linear(rij, rik, rij2, rik2)
                                || near_linear(rij, ril, rij2, ril2)
                                || near_linear(rik, ril, rik2, ril2)
                            {
                                continue;
                            }
                            let tau = oop_angle(rij, rik, ril);
                            if tau.abs() > PI / 4.0 {
                                continue;
                            }
                            let t = p.ko
                                * pf(i, j, rij2, 0.5 * kd, true)
                                * pf(i, k, rik2, 0.5 * kd, true)
                                * pf(i, l, ril2, 0.5 * kd, true);
                            acc.out_of_plane(id(i), id(j), id(k), id(l), t);
                        }
                    }
                }
            }
        }
        let mut h = acc.finish();
        add_dispersion(&mut h, sys, self.dispersion)?;
        if (0..h.nrows()).any(|i| (0..h.ncols()).any(|j| !h[(i, j)].is_finite())) {
            return Err(MoptError::calc(format!(
                "model Hessian '{}' is not finite",
                self.name
            )));
        }
        Ok(h)
    }
}

/// Wilson angle of the bond to `-rij` above the plane of `-rik`, `-ril` (vectors from the
/// neighbours to the center).
fn oop_angle(rij: [f64; 3], rik: [f64; 3], ril: [f64; 3]) -> f64 {
    let (a, b, c) = (rij.map(|x| -x), rik.map(|x| -x), ril.map(|x| -x));
    let nrm = cross(b, c);
    let nn = norm(nrm);
    if nn < 1e-12 {
        return 0.0;
    }
    (dot(nrm, a) / (nn * norm(a))).clamp(-1.0, 1.0).asin()
}
