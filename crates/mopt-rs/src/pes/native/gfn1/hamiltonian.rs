//! The GFN1-xTB basis, coordination numbers, zeroth-order Hamiltonian, and Coulomb kernel.
//!
//! - Basis: STO-nG expansions of the Slater shells (GFN0-xTB's tables and integrals,
//!   [`crate::pes::native::gfn0::basis`]; Stewart's STO-6G 4s and 4p as tblite), the hydrogen
//!   2s shell orthogonalized to its 1s shell (xtb `src/xtb/basisset.f90`), real spherical d
//!   functions, one-centre overlaps equal to the identity.
//! - Coordination numbers: `CN_A = sum_B 1 / (1 + exp(-16 (R_AB^cov / R_AB - 1)))` with
//!   `R^cov` 4/3 of the D3 covalent radii, pairs within 25 Bohr.
//! - Self energies `h_l^A = E_l^A (1 + 0.01 k_CN,l CN_A)`.
//! - Off-diagonal elements of atoms A != B:
//!   `H_mu nu = 1/2 K_ll' (h_l^A + h_l'^B) Pi(R_AB) S_mu nu`, with
//!   `Pi = (1 + 0.01 k_poly,l^A x)(1 + 0.01 k_poly,l'^B x)`, `x = sqrt(R_AB / (R_A + R_B))`
//!   (atomic radii), and `K_ll' = k_ll' (1 - 0.007 (EN_A - EN_B)^2) k_pair` for two valence
//!   shells (`k_sp = 2.08` for s-p, the mean of `k_s = 1.85`, `k_p = 2.25`, `k_d = 2.0`
//!   otherwise), `(k_ll + k_diff) / 2` with one non-valence shell and `k_diff = 2.85` with
//!   two (no EN factor and no pair parameter then).
//! - Coulomb kernel `gamma_ab = (R_AB^2 + eta_ab^-2)^(-1/2)` with the harmonic mean
//!   `eta_ab = 2 / (1/eta_a + 1/eta_b)` of the shell hardnesses `eta_a = eta_A (1 + kappa_l)`.
//!
//! xtb: `src/xtb/hamiltonian.F90` (`build_SDQH0`, `build_dSDQH0`), `src/xtb/gfn1.f90`,
//! `src/coulomb/klopmanohno.F90`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::params::{
    pair_scaling, Gfn1Element, BOHR_ANGSTROM, CN_CUTOFF, HARTREE_EV, K_DIFF, K_EN, K_SHELL, K_SP,
    STO6G_4P, STO6G_4S,
};
use crate::error::{MoptError, MoptResult};
use crate::pes::native::gfn0::basis::{components, orthogonalize_diffuse_s, Basis, BasisShell};
use crate::pes::native::gfn0::sto::{slater_to_gauss, Primitive};

/// Angstrom to Bohr with tblite's constant.
pub fn bohr(angstrom: f64) -> f64 {
    angstrom / BOHR_ANGSTROM
}

const DFACTORIAL: [f64; 3] = [1.0, 1.0, 3.0];

/// Stewart's STO-6G expansion of a 4s or 4p Slater function with exponent `zeta`, normalized
/// as [`slater_to_gauss`] does.
fn stewart_4sp(l: usize, zeta: f64) -> Vec<Primitive> {
    let (alpha, coeff) = if l == 0 { STO6G_4S } else { STO6G_4P };
    let zeta2 = zeta * zeta;
    alpha
        .iter()
        .zip(coeff)
        .map(|(&a, c)| {
            let exponent = a * zeta2;
            let norm = (2.0 / std::f64::consts::PI * exponent).powf(0.75)
                * (4.0 * exponent).sqrt().powi(l as i32)
                / DFACTORIAL[l].sqrt();
            Primitive {
                exponent,
                coefficient: c * norm,
            }
        })
        .collect()
}

fn primitives(n: u8, l: usize, nprim: usize, zeta: f64) -> MoptResult<Vec<Primitive>> {
    if n == 4 && l <= 1 && nprim == 6 {
        Ok(stewart_4sp(l, zeta))
    } else {
        slater_to_gauss(nprim, n, l, zeta)
    }
}

/// The basis of atoms with parameters `el`.
pub fn basis(el: &[&Gfn1Element]) -> MoptResult<Basis> {
    let mut shells = Vec::new();
    let mut comps = Vec::new();
    for (atom, e) in el.iter().enumerate() {
        let mut valence_s: Option<Vec<Primitive>> = None;
        for sh in e.shells {
            let mut prims = primitives(sh.n, sh.l, sh.nprim, sh.zeta)?;
            if sh.l == 0 {
                if sh.valence {
                    valence_s = Some(prims.clone());
                } else {
                    let reference = valence_s.as_ref().ok_or_else(|| {
                        MoptError::config("GFN1-xTB: a diffuse s shell without a valence s shell")
                    })?;
                    orthogonalize_diffuse_s(reference, &mut prims);
                }
            }
            let ao = components(sh.l)?;
            let first_ao = comps.len();
            let nao = ao.len();
            comps.extend(ao);
            let min_exponent = prims
                .iter()
                .map(|p| p.exponent)
                .fold(f64::INFINITY, f64::min);
            shells.push(BasisShell {
                atom,
                z: e.z,
                l: sh.l,
                valence: sh.valence,
                first_ao,
                nao,
                level: sh.self_energy_ev / HARTREE_EV,
                exponent: sh.zeta,
                kcn: 0.01 * e.cn_factor(sh.l),
                kq: 0.0,
                poly: 0.01 * sh.kpoly,
                en: e.en,
                kqat2: 0.0,
                atomic_radius: bohr(e.atomic_rad),
                primitives: prims,
                min_exponent,
            });
        }
    }
    Ok(Basis {
        shells,
        components: comps,
    })
}

/// Coordination numbers and the derivatives of every pair's count.
pub struct Coordination {
    /// `CN_A`.
    pub cn: Vec<f64>,
    /// `(A, B, d f_AB / d R_A)` for the pairs that count (`d/dR_B` is the negative).
    pub pairs: Vec<(usize, usize, [f64; 3])>,
}

impl Coordination {
    /// The coordination numbers of the structure.
    pub fn new(el: &[&Gfn1Element], x: &[[f64; 3]]) -> Self {
        let n = x.len();
        let rcov: Vec<f64> = el.iter().map(|e| 4.0 / 3.0 * bohr(e.cov_rad_d3)).collect();
        let mut cn = vec![0.0; n];
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in 0..i {
                let d = diff(x, i, j);
                let r2 = dot(d, d);
                if r2 > CN_CUTOFF * CN_CUTOFF {
                    continue;
                }
                let r = r2.sqrt();
                let rc = rcov[i] + rcov[j];
                // The exponent is at most 16: no overflow.
                let e = (-16.0 * (rc / r - 1.0)).exp();
                let f = 1.0 / (1.0 + e);
                let df = -16.0 * rc * e / (r2 * (1.0 + e) * (1.0 + e));
                cn[i] += f;
                cn[j] += f;
                pairs.push((i, j, d.map(|v| df * v / r)));
            }
        }
        Self { cn, pairs }
    }

    /// Add `sum_A dE/dCN_A dCN_A/dR` to `grad` (flat, 3N).
    pub fn contract(&self, de_dcn: &[f64], grad: &mut [f64]) {
        for &(i, j, g) in &self.pairs {
            let w = de_dcn[i] + de_dcn[j];
            for k in 0..3 {
                grad[3 * i + k] += w * g[k];
                grad[3 * j + k] -= w * g[k];
            }
        }
    }
}

/// `x_a - x_b`.
pub fn diff(x: &[[f64; 3]], a: usize, b: usize) -> [f64; 3] {
    [x[a][0] - x[b][0], x[a][1] - x[b][1], x[a][2] - x[b][2]]
}

/// Dot product.
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn k_scale(la: usize, lb: usize) -> f64 {
    if (la, lb) == (0, 1) || (la, lb) == (1, 0) {
        K_SP
    } else {
        0.5 * (K_SHELL[la] + K_SHELL[lb])
    }
}

/// What scales the overlap of a shell pair on different atoms.
pub struct PairScale {
    /// `K_ll'` with the EN factor and the pair parameter.
    pub km: f64,
    /// `Pi(R_AB)`.
    pub poly: f64,
    /// `d Pi / d R_A`.
    pub dpoly: [f64; 3],
}

/// The scaling of shells `sa` (atom A) and `sb` (atom B != A) at `d = R_A - R_B`.
pub fn pair_scale(sa: &BasisShell, sb: &BasisShell, d: [f64; 3]) -> PairScale {
    let km = match (sa.valence, sb.valence) {
        (true, true) => {
            let den = (sa.en - sb.en) * (sa.en - sb.en);
            k_scale(sa.l, sb.l) * (1.0 + K_EN * den) * pair_scaling(sa.z, sb.z)
        }
        (true, false) => 0.5 * (k_scale(sa.l, sa.l) + K_DIFF),
        (false, true) => 0.5 * (k_scale(sb.l, sb.l) + K_DIFF),
        (false, false) => K_DIFF,
    };
    let r2 = dot(d, d);
    let r = r2.sqrt();
    let x = (r / (sa.atomic_radius + sb.atomic_radius)).sqrt();
    let (ka, kb) = (sa.poly, sb.poly);
    let poly = (1.0 + ka * x) * (1.0 + kb * x);
    // dx/dR_A = x / (2 r^2) d.
    let slope = (ka * (1.0 + kb * x) + kb * (1.0 + ka * x)) * x / (2.0 * r2);
    PairScale {
        km,
        poly,
        dpoly: d.map(|v| slope * v),
    }
}

/// The self energies of the shells at the coordination numbers `cn` (Hartree).
pub fn levels(basis: &Basis, cn: &[f64]) -> Vec<f64> {
    basis
        .shells
        .iter()
        .map(|s| s.level * (1.0 + s.kcn * cn[s.atom]))
        .collect()
}

/// The zeroth-order Hamiltonian from the overlap `s` and the shell levels `h`.
pub fn hamiltonian(basis: &Basis, x: &[[f64; 3]], s: &Mat<f64>, h: &[f64]) -> Mat<f64> {
    let n = basis.len();
    let mut out = Mat::<f64>::zeros(n, n);
    for (a, sa) in basis.shells.iter().enumerate() {
        for mu in sa.first_ao..sa.first_ao + sa.nao {
            out[(mu, mu)] = h[a];
        }
    }
    let ns = basis.shells.len();
    for a in 0..ns {
        for b in a + 1..ns {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            if sa.atom == sb.atom {
                continue;
            }
            let d = diff(x, sa.atom, sb.atom);
            if basis.negligible(a, b, dot(d, d)) {
                continue;
            }
            let ps = pair_scale(sa, sb, d);
            let scale = 0.5 * ps.km * (h[a] + h[b]) * ps.poly;
            for mu in sa.first_ao..sa.first_ao + sa.nao {
                for nu in sb.first_ao..sb.first_ao + sb.nao {
                    let v = scale * s[(mu, nu)];
                    out[(mu, nu)] = v;
                    out[(nu, mu)] = v;
                }
            }
        }
    }
    out
}

/// The shell hardnesses `eta_A (1 + kappa_l)`.
pub fn shell_hardness(el: &[&Gfn1Element], basis: &Basis) -> MoptResult<Vec<f64>> {
    basis
        .shells
        .iter()
        .map(|s| {
            let e = el[s.atom];
            let shell = e
                .shells
                .iter()
                .find(|p| p.l == s.l && p.valence == s.valence)
                .ok_or_else(|| MoptError::config("GFN1-xTB: a shell without parameters"))?;
            Ok(e.eta * (1.0 + shell.kappa))
        })
        .collect()
}

/// The Coulomb kernel of the shells.
pub fn gamma(basis: &Basis, x: &[[f64; 3]], eta: &[f64]) -> Mat<f64> {
    let ns = basis.shells.len();
    Mat::from_fn(ns, ns, |a, b| {
        let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
        let g = 2.0 / (1.0 / eta[a] + 1.0 / eta[b]);
        if sa.atom == sb.atom {
            if a == b {
                eta[a]
            } else {
                g
            }
        } else {
            let d = diff(x, sa.atom, sb.atom);
            1.0 / (dot(d, d) + 1.0 / (g * g)).sqrt()
        }
    })
}

/// `d gamma_ab / d R_A` for shells on atoms A != B (`d = R_A - R_B`).
pub fn gamma_derivative(eta_a: f64, eta_b: f64, d: [f64; 3]) -> [f64; 3] {
    let g = 2.0 / (1.0 / eta_a + 1.0 / eta_b);
    let t = dot(d, d) + 1.0 / (g * g);
    let f = -1.0 / (t * t.sqrt());
    d.map(|v| f * v)
}
