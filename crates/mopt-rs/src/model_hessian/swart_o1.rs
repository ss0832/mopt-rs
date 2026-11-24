//! Swart model Hessian as modified for O1NumHess (B. Wang, S. Luo, Z. Wang, W. Liu, J. Chem.
//! Theory Comput. 2025, 21, 10893, arXiv:2508.07544, Appendix B; reference implementation
//! `O1NumHess_QC/Swart.py`, MIT; original model: M. Swart, F. M. Bickelhaupt, Int. J. Quantum
//! Chem. 2006, 106, 2536).
//!
//! With `rho_AB = exp(1 - r_AB / (r_A + r_B))` (Pyykko covalent radii):
//!
//! - every atom pair is a bond with `k = 0.35 rho^3`;
//! - triples with `rho_AB rho_BC >= 0.09` are angles with
//!   `k = 0.075 (rho_AB rho_BC (0.12 + 0.88 sin theta))^2`;
//! - close to 0 degrees (`cos theta > 0.8`) the angle is scaled by `(1 - s)^2`,
//!   `s = (1 - ((1 - |cos theta|) / 0.2)^2)^2`;
//! - close to 180 degrees (`cos theta < -0.8`) two linear-angle coordinates take over: the
//!   in-plane one with `k` (its B row blended from the ordinary angle by `s`, which keeps the
//!   Hessian continuous) and the out-of-plane one with `s^2 k` (Eqs. 50-52).
//!
//! The reference code applies the near-0 and near-180 branches the other way round (it tests
//! `cos theta > 0.8` for linear angles); MultiOptPy copies that, and its version is available
//! as `legacy_swart`.
//!
//! MultiOptPy: the published model behind `ModelHessian/swart.py` (MultiOptPy's own version is
//!  `legacy/swart.rs`); the O1NumHess modification itself has no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::assemble::HessianAccumulator;
use super::ModelHessian;
use crate::domain::{AtomId, MolecularSystem};
use crate::error::MoptResult;

/// Pyykko covalent radii (Angstrom) of the reference implementation, `Z = 1..=103`.
const COVRAD_ANG: [f64; 103] = [
    0.32, 0.46, 1.33, 1.02, 0.85, 0.75, 0.71, 0.63, 0.64, 0.67, 1.55, 1.39, 1.26, 1.16, 1.11, 1.03,
    0.99, 0.96, 1.96, 1.71, 1.48, 1.36, 1.34, 1.22, 1.19, 1.16, 1.1, 1.11, 1.12, 1.18, 1.24, 1.21,
    1.21, 1.16, 1.14, 1.17, 2.1, 1.85, 1.63, 1.54, 1.47, 1.38, 1.28, 1.25, 1.25, 1.2, 1.28, 1.36,
    1.42, 1.4, 1.4, 1.36, 1.33, 1.31, 2.32, 1.96, 1.8, 1.63, 1.76, 1.74, 1.73, 1.72, 1.68, 1.69,
    1.68, 1.67, 1.66, 1.65, 1.64, 1.7, 1.62, 1.52, 1.46, 1.37, 1.31, 1.29, 1.22, 1.23, 1.24, 1.33,
    1.44, 1.44, 1.51, 1.45, 1.47, 1.42, 2.23, 2.01, 1.86, 1.75, 1.69, 1.7, 1.71, 1.72, 1.66, 1.66,
    1.68, 1.68, 1.65, 1.67, 1.73, 1.76, 1.61,
];

/// The reference implementation's Bohr per Angstrom (`1 / 0.529177249`, an older Bohr radius). Kept
/// on purpose instead of `Units::ANGSTROM_TO_BOHR`: it is part of the published model (the radii
/// are converted with it in `O1NumHess_QC/Swart.py`).
const ANG2BOHR: f64 = 1.0 / 0.529177249;

fn covrad_bohr(z: u8) -> f64 {
    COVRAD_ANG[(z as usize).clamp(1, 103) - 1] * ANG2BOHR
}

/// Which near-0 / near-180 degree treatment to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwartAngles {
    /// As in the paper.
    Paper,
    /// As in the reference code (and MultiOptPy).
    ReferenceCode,
}

/// The O1NumHess Swart model.
#[derive(Debug, Clone, Copy)]
pub struct SwartO1 {
    /// Angle treatment.
    pub angles: SwartAngles,
}

impl Default for SwartO1 {
    fn default() -> Self {
        Self {
            angles: SwartAngles::Paper,
        }
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|x| x * s)
}

/// Ordinary angle B row (`[i, j, k]` blocks) as in the reference code, regularized at 0 and
/// 180 degrees.
fn angle_row(v1: [f64; 3], v2: [f64; 3]) -> [[f64; 3]; 3] {
    let (l1, l2) = (norm(v1), norm(v2));
    let (n1, n2) = (scale(v1, 1.0 / l1), scale(v2, 1.0 / l2));
    let c = dot(n1, n2);
    let s = (1.0 - c * c).max(1e-15).sqrt();
    // d cos / d r_i = (n2 - c n1) / l1; d theta = -d cos / sin.
    let bi = scale(sub(n2, scale(n1, c)), -1.0 / (l1 * s));
    let bk = scale(sub(n1, scale(n2, c)), -1.0 / (l2 * s));
    let bj = scale([bi[0] + bk[0], bi[1] + bk[1], bi[2] + bk[2]], -1.0);
    [bi, bj, bk]
}

/// The two linear-angle B rows of the reference code: in-plane (`[0]`) and out-of-plane
/// (`[1]`).
fn linear_rows(v1: [f64; 3], v2: [f64; 3]) -> [[[f64; 3]; 3]; 2] {
    let (l1, l2) = (norm(v1), norm(v2));
    let mut vn = cross(v1, v2);
    if norm(vn) < 1e-15 {
        vn = sub([1.0, 0.0, 0.0], scale(v1, v1[0] / (l1 * l1)));
        if norm(vn) < 1e-15 {
            vn = sub([0.0, 1.0, 0.0], scale(v1, v1[1] / (l1 * l1)));
        }
    }
    let vn = scale(vn, 1.0 / norm(vn));
    let vn2 = cross(sub(v1, v2), vn);
    let vn2 = scale(vn2, 1.0 / norm(vn2));
    let row = |e: [f64; 3]| {
        let (a, c) = (scale(e, 1.0 / l1), scale(e, 1.0 / l2));
        [a, [-a[0] - c[0], -a[1] - c[1], -a[2] - c[2]], c]
    };
    [row(vn2), row(vn)]
}

fn blend(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], s: f64) -> [[f64; 3]; 3] {
    std::array::from_fn(|p| std::array::from_fn(|q| s * a[p][q] + (1.0 - s) * b[p][q]))
}

impl ModelHessian for SwartO1 {
    fn name(&self) -> &str {
        match self.angles {
            SwartAngles::Paper => "swart",
            SwartAngles::ReferenceCode => "legacy_swart",
        }
    }

    fn hessian(&self, sys: &MolecularSystem, _gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        let n = sys.n_atoms();
        let id = AtomId::new;
        let mut acc = HessianAccumulator::new(sys);
        let rad: Vec<f64> = sys
            .atoms()
            .map(|a| covrad_bohr(sys.element(a).atomic_number()))
            .collect();
        let mut rho = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    rho[i * n + j] = (1.0 - acc.r2(id(i), id(j)).sqrt() / (rad[i] + rad[j])).exp();
                }
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                let d = acc.delta(id(i), id(j));
                let b = scale(d, 1.0 / norm(d));
                acc.add_rows(
                    [id(i), id(j)],
                    &[b, scale(b, -1.0)],
                    0.35 * rho[i * n + j].powi(3),
                );
            }
        }
        let (wthr, f, tolth) = (0.3_f64, 0.12, 0.2);
        let (eps1, eps2) = (wthr * wthr, wthr * wthr / std::f64::consts::E);
        for i in 0..n {
            for j in (0..n).filter(|&j| j != i && rho[i * n + j] >= eps2) {
                for k in (i + 1..n).filter(|&k| k != j) {
                    let s_ijk = rho[i * n + j] * rho[j * n + k];
                    if s_ijk < eps1 {
                        continue;
                    }
                    let (v1, v2) = (acc.delta(id(i), id(j)), acc.delta(id(k), id(j)));
                    let c = dot(v1, v2) / (norm(v1) * norm(v2));
                    let sin = (1.0 - c * c).max(0.0).sqrt();
                    let h = 0.075 * s_ijk * s_ijk * (f + (1.0 - f) * sin).powi(2);
                    let b = angle_row(v1, v2);
                    let atoms = [id(i), id(j), id(k)];
                    let th1 = if c > 1.0 - tolth { 1.0 - c } else { 1.0 + c };
                    if th1 >= tolth {
                        acc.add_rows(atoms, &b, h);
                        continue;
                    }
                    let s = (1.0 - (th1 / tolth).powi(2)).powi(2);
                    // Which side counts as linear.
                    let linear = match self.angles {
                        SwartAngles::Paper => c < 0.0,
                        SwartAngles::ReferenceCode => c > 1.0 - tolth,
                    };
                    if linear {
                        let lin = linear_rows(v1, v2);
                        acc.add_rows(atoms, &blend(&lin[0], &b, s), h);
                        let w = match self.angles {
                            SwartAngles::Paper => s * s,
                            SwartAngles::ReferenceCode => 1.0,
                        };
                        acc.add_rows(atoms, &lin[1], w * h);
                    } else {
                        acc.add_rows(atoms, &b, (1.0 - s).powi(2) * h);
                    }
                }
            }
        }
        Ok(acc.finish())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};

    /// The H2O2 test of the reference implementation (`Swart.py`). The H-H-O angle at the
    /// second hydrogen has `cos theta = 0.874`, where the reference code's linear-angle branch
    /// applies, so the printed matrix is the reference-code variant.
    #[test]
    fn reproduces_the_reference_h2o2_hessian() {
        let sys = MolecularSystem::new(
            vec![Element::H, Element::O, Element::O, Element::H],
            &positions::<Bohr>(&[[0., 2., 0.], [0., 0., 0.], [0., 0., 3.], [2., 0., 3.]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let reference = SwartO1 {
            angles: SwartAngles::ReferenceCode,
        };
        let h = reference.hessian(&sys, &Col::zeros(12)).unwrap();
        // Selected elements of the printed reference.
        let want = [
            ((0, 0), 1.45100659e-03),
            ((1, 1), 2.54271977e-01),
            ((1, 4), -2.48719845e-01),
            ((2, 2), 2.28988945e-02),
            ((5, 5), 1.84144835e-01),
            ((5, 8), -1.61380534e-01),
            ((3, 5), 8.59418131e-03),
            ((2, 7), 1.42536281e-02),
        ];
        for ((r, c), v) in want {
            assert!(
                (h[(r, c)] - v).abs() < 1e-9,
                "H({r},{c}) = {} vs {v}",
                h[(r, c)]
            );
        }
        // The paper's treatment of that angle (close to zero degrees) differs.
        let paper = SwartO1::default().hessian(&sys, &Col::zeros(12)).unwrap();
        assert!((paper[(0, 0)] - h[(0, 0)]).abs() > 1e-5);
    }

    fn bent(theta: f64) -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::C; 3],
            &positions::<Bohr>(&[
                [2.6, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.6 * theta.cos(), 2.6 * theta.sin(), 0.0],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    /// Paper version: continuous where the near-linear and near-zero treatments start
    /// (`|cos theta| = 0.8`), and two equal linear-angle constants at 180 degrees.
    #[test]
    fn paper_angles_are_continuous() {
        let m = SwartO1::default();
        for c in [0.8_f64, -0.8] {
            let (a, b) = (c.acos() - 1e-7, c.acos() + 1e-7);
            let ha = m.hessian(&bent(a), &Col::zeros(9)).unwrap();
            let hb = m.hessian(&bent(b), &Col::zeros(9)).unwrap();
            let diff = (0..9)
                .flat_map(|i| (0..9).map(move |j| (i, j)))
                .fold(0.0_f64, |d, (i, j)| d.max((ha[(i, j)] - hb[(i, j)]).abs()));
            assert!(diff < 1e-6, "cos = {c}: jump {diff}");
        }
        let lin = m
            .hessian(&bent(std::f64::consts::PI), &Col::zeros(9))
            .unwrap();
        // Bends of the end atoms in y and z are equally stiff.
        assert!((lin[(1, 1)] - lin[(2, 2)]).abs() < 1e-12 && lin[(1, 1)] > 0.0);
    }
}
