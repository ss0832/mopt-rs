//! The first-order response of a tight-binding density with Fermi smearing to a perturbation
//! of its Fock and overlap matrices, for analytic Hessians.
//!
//! With the orbitals `C` (`C^T S C = 1`, `F C = S C eps`), the occupations `f_i^s` of each
//! spin `s` at the electronic temperature `kT`, `n_i = sum_s f_i^s`, and a perturbation `F1`,
//! `S1` (atomic orbitals), in the orbital basis `F'_ij = C_i^T F1 C_j`, `S'_ij = C_i^T S1 C_j`:
//!
//! - orbital energies `eps1_i = F'_ii - eps_i S'_ii`;
//! - the chemical potential of each spin keeps its electron count:
//!   `mu1_s = sum_i f'_i^s eps1_i / sum_i f'_i^s`, with `f'^s = -f^s (1 - f^s) / kT`;
//! - the density `P1 = C M C^T` with
//!   `M_ij = F'_ij (n_i - n_j) / (eps_i - eps_j) - S'_ij (n_i eps_i - n_j eps_j) / (eps_i - eps_j)`
//!   (`i != j`; the limits `F'_ij n' - S'_ij (n + eps n')` at equal energies) and
//!   `M_ii = F'_ii n'_i - S'_ii (n_i + eps_i n'_i) - sum_s f'_i^s mu1_s`;
//! - the energy-weighted density `W = sum_i n_i eps_i C_i C_i^T` likewise with `N_i = n_i eps_i`
//!   in place of `n_i` and the diagonal shift `- eps_i sum_s f'_i^s mu1_s`.
//!
//! These follow from `C1 = C U`, `U_ij = (F'_ij - eps_j S'_ij) / (eps_j - eps_i)` (`i != j`),
//! `U_ii = -S'_ii / 2`, and `n1_i = sum_s f'_i^s (eps1_i - mu1_s)`; they are derived here and
//! checked against finite differences in the tests of the Hessians that use them.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::math::linalg::{mul, mul3};

/// Energies closer than this count as degenerate (Hartree).
const DEGENERATE: f64 = 1e-8;

/// The orbitals and occupations of a converged density.
#[derive(Debug, Clone)]
pub struct OrbitalResponse {
    /// `C` (atomic orbitals by orbitals).
    pub c: Mat<f64>,
    /// `eps`.
    pub eps: Vec<f64>,
    /// `n_i`.
    pub n: Vec<f64>,
    /// `d n_i / d eps_i` at fixed chemical potentials.
    pub dn: Vec<f64>,
    /// `f'^s` of each spin.
    pub dfs: [Vec<f64>; 2],
}

/// The response of the density and of the energy-weighted density.
#[derive(Debug, Clone)]
pub struct DensityResponse {
    /// `P1` (atomic orbitals).
    pub p: Mat<f64>,
    /// `W1` (atomic orbitals).
    pub w: Mat<f64>,
}

impl OrbitalResponse {
    /// From the orbitals `c`, their energies `eps`, the occupations of the two spins, and
    /// `kt` (Hartree; 0 for integer occupations, which then do not respond).
    pub fn new(c: Mat<f64>, eps: Vec<f64>, spin: [&[f64]; 2], kt: f64) -> Self {
        let m = eps.len();
        let df = |f: &[f64]| -> Vec<f64> {
            (0..m)
                .map(|i| {
                    if kt > 0.0 {
                        -f[i] * (1.0 - f[i]) / kt
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let dfs = [df(spin[0]), df(spin[1])];
        let n: Vec<f64> = (0..m).map(|i| spin[0][i] + spin[1][i]).collect();
        let dn: Vec<f64> = (0..m).map(|i| dfs[0][i] + dfs[1][i]).collect();
        Self { c, eps, n, dn, dfs }
    }

    /// `C^T A C`.
    pub fn to_orbitals(&self, a: &Mat<f64>) -> Mat<f64> {
        mul3(self.c.transpose(), a.as_ref(), self.c.as_ref())
    }

    /// The response to `F1` and `S1` given in the orbital basis (`f1`, `s1`; `s1` `None`
    /// for an unchanged overlap).
    pub fn respond(&self, f1: &Mat<f64>, s1: Option<&Mat<f64>>) -> DensityResponse {
        self.respond_with(f1, s1, true)
    }

    /// The response of the density alone (`w` is `0 x 0`), for perturbations whose
    /// energy-weighted density is not needed.
    pub fn respond_density(&self, f1: &Mat<f64>, s1: Option<&Mat<f64>>) -> DensityResponse {
        self.respond_with(f1, s1, false)
    }

    fn respond_with(
        &self,
        f1: &Mat<f64>,
        s1: Option<&Mat<f64>>,
        weighted: bool,
    ) -> DensityResponse {
        let m = self.eps.len();
        let s = |i: usize, j: usize| s1.map_or(0.0, |s| s[(i, j)]);
        let eps1: Vec<f64> = (0..m).map(|i| f1[(i, i)] - self.eps[i] * s(i, i)).collect();
        let mu1: Vec<f64> = (0..2)
            .map(|sp| {
                let den: f64 = self.dfs[sp].iter().sum();
                if den.abs() > 0.0 {
                    self.dfs[sp]
                        .iter()
                        .zip(&eps1)
                        .map(|(d, e)| d * e)
                        .sum::<f64>()
                        / den
                } else {
                    0.0
                }
            })
            .collect();
        let shift: Vec<f64> = (0..m)
            .map(|i| self.dfs[0][i] * mu1[0] + self.dfs[1][i] * mu1[1])
            .collect();
        let mut mp = Mat::<f64>::zeros(m, m);
        let mut mw = Mat::<f64>::zeros(m, m);
        for i in 0..m {
            let (ei, ni, di) = (self.eps[i], self.n[i], self.dn[i]);
            let big_ni = ni * ei;
            for j in 0..m {
                let (ej, nj, dj) = (self.eps[j], self.n[j], self.dn[j]);
                let big_nj = nj * ej;
                let (fij, sij) = (f1[(i, j)], s(i, j));
                let de = ei - ej;
                let (p, w) = if de.abs() > DEGENERATE {
                    (
                        fij * (ni - nj) / de - sij * (ni * ei - nj * ej) / de,
                        fij * (big_ni - big_nj) / de - sij * (big_ni * ei - big_nj * ej) / de,
                    )
                } else {
                    let (e, nn, d) = (0.5 * (ei + ej), 0.5 * (ni + nj), 0.5 * (di + dj));
                    // d(n eps)/d eps = n + eps n', d(n eps^2)/d eps = 2 n eps + eps^2 n'.
                    (
                        fij * d - sij * (nn + e * d),
                        fij * (nn + e * d) - sij * (2.0 * nn * e + e * e * d),
                    )
                };
                mp[(i, j)] = p;
                mw[(i, j)] = w;
            }
            mp[(i, i)] -= shift[i];
            mw[(i, i)] -= shift[i] * ei;
        }
        DensityResponse {
            p: self.to_atomic(&mp),
            w: if weighted {
                self.to_atomic(&mw)
            } else {
                Mat::zeros(0, 0)
            },
        }
    }

    /// `C M C^T` of an orbital matrix `M` whose block between the orbitals beyond the last
    /// occupied or responding one (`n_i = 0`, `dn_i = 0`) is zero, as it is for the response:
    /// with the first `k` orbitals active, `C_k (M_k: C^T) + C_v (M_vk C_k^T)` (rectangular
    /// products, the same sum up to rounding; new).
    fn to_atomic(&self, mm: &Mat<f64>) -> Mat<f64> {
        let m = self.eps.len();
        let k = (0..m)
            .rposition(|i| self.n[i] != 0.0 || self.dn[i] != 0.0)
            .map_or(0, |i| i + 1);
        let c = self.c.as_ref();
        if k == m {
            return mul3(c, mm.as_ref(), c.transpose());
        }
        let ck = c.subcols(0, k);
        let cv = c.subcols(k, m - k);
        let top = mul(mm.as_ref().subrows(0, k), c.transpose());
        let mut out = mul(ck, top.as_ref());
        let bottom = mul(mm.as_ref().submatrix(k, 0, m - k, k), ck.transpose());
        out += mul(cv, bottom.as_ref());
        out
    }
}
