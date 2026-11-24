//! The non-self-consistent electronic part of GFN0-xTB, as in xtb and gfn0-rs
//! (`src/hamiltonian.rs`, `src/electronic.rs`, `src/smearing.rs`):
//!
//! - H0: self-energies `H_l - k_CN CN - k_q q - k_q2 q^2` on the diagonal and
//!   `H_mn = S_mn K_ll' (H_m + H_n) / 2 (1 + k_A x)(1 + k_B x)`, `x = sqrt(r / (R_A + R_B))`, with
//!   the EN, Slater-exponent, and pair scalings of `K`;
//! - one diagonalization `H C = S C e` (Loewdin orthogonalization) and Fermi-Dirac filling of
//!   each spin channel at 300 K (chemical potential by bisection, centred in gaps);
//! - the free energy `sum f e + kT sum [f ln f + (1 - f) ln(1 - f)]`.
//!
//! The gradient is `Tr(P dH/dx) - Tr(W dS/dx)` (`W` energy-weighted), with the self-energy
//! derivatives handed back as `dE/dCN` and `dE/dq` per atom.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::basis::{pair_overlap_gradient, Basis, BasisShell};
use super::params::{pair_scale, shell_k_en, BOLTZMANN_HARTREE_PER_K};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{mul, mul3, symmetric_eigen};
use crate::parameters::gfn0_xtb as g;

/// Electronic temperature (K).
pub const TEMPERATURE_K: f64 = 300.0;

/// `K_ll'` times its scalings, and the radial polynomial of a shell pair.
pub(super) struct PairScale {
    pub(super) km: f64,
    pub(super) poly: f64,
    /// `d poly / d x_A` (`x_A - x_B` direction).
    pub(super) dpoly: [f64; 3],
}

pub(super) fn pair_scale_of(a: &BasisShell, b: &BasisShell, d: [f64; 3]) -> MoptResult<PairScale> {
    let (ka, ena) = shell_k_en(a.l);
    let (kb, enb) = shell_k_en(b.l);
    let kscale = 0.5 * (ka + kb);
    let enscale = g::GFN0_EN_SCALE_PREFACTOR * (ena + enb);
    let (ea, eb) = (a.exponent.max(1.0e-8), b.exponent.max(1.0e-8));
    let zeta = 2.0 * (ea * eb).sqrt() / (ea + eb);
    let den = a.en - b.en;
    let den2 = den * den;
    let enp = 1.0 + enscale * den2 + g::GFN0_ENSCALE4 * enscale * den2 * den2;
    let mut km = kscale * pair_scale(a.z, b.z) * zeta * enp;
    if a.valence != b.valence {
        km *= g::GFN0_KDIFF;
    }
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if r <= 1.0e-12 {
        return Err(MoptError::calc(
            "coincident atoms in the GFN0-xTB Hamiltonian",
        ));
    }
    let x = (r / (a.atomic_radius + b.atomic_radius)).sqrt();
    let ga = 0.01 * a.poly * x;
    let gb = 0.01 * b.poly * x;
    let poly = (1.0 + ga) * (1.0 + gb);
    let dpoly_dr = 0.5 * (ga * (1.0 + gb) + (1.0 + ga) * gb) / r;
    Ok(PairScale {
        km,
        poly,
        dpoly: d.map(|v| dpoly_dr * v / r),
    })
}

/// Shell self-energies.
pub fn self_energies(basis: &Basis, cn: &[f64], q: &[f64]) -> Vec<f64> {
    basis
        .shells
        .iter()
        .map(|s| {
            let qa = q[s.atom];
            s.level - s.kcn * cn[s.atom] - s.kq * qa - s.kqat2 * qa * qa
        })
        .collect()
}

fn diff(x: &[[f64; 3]], a: usize, b: usize) -> [f64; 3] {
    [x[a][0] - x[b][0], x[a][1] - x[b][1], x[a][2] - x[b][2]]
}

fn sq(d: [f64; 3]) -> f64 {
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// H0.
pub fn hamiltonian(
    basis: &Basis,
    x: &[[f64; 3]],
    s: &Mat<f64>,
    eps: &[f64],
) -> MoptResult<Mat<f64>> {
    let n = basis.len();
    let mut h = Mat::<f64>::zeros(n, n);
    for (k, sh) in basis.shells.iter().enumerate() {
        for mu in sh.first_ao..sh.first_ao + sh.nao {
            h[(mu, mu)] = eps[k];
        }
    }
    let ns = basis.shells.len();
    for a in 0..ns {
        for b in a + 1..ns {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            if sa.atom == sb.atom || (!sa.valence && !sb.valence) {
                continue;
            }
            let d = diff(x, sa.atom, sb.atom);
            if basis.negligible(a, b, sq(d)) {
                continue;
            }
            let ps = pair_scale_of(sa, sb, d)?;
            let scale = 0.5 * ps.km * (eps[a] + eps[b]) * ps.poly;
            for mu in sa.first_ao..sa.first_ao + sa.nao {
                for nu in sb.first_ao..sb.first_ao + sb.nao {
                    let v = s[(mu, nu)] * scale;
                    h[(mu, nu)] = v;
                    h[(nu, mu)] = v;
                }
            }
        }
    }
    Ok(h)
}

/// The filled electronic structure.
#[derive(Debug, Clone)]
pub struct Electronic {
    /// Free energy.
    pub energy: f64,
    /// Orbital energies.
    pub eigenvalues: Vec<f64>,
    /// Total occupations.
    pub occupations: Vec<f64>,
    /// Density matrix.
    pub density: Mat<f64>,
    /// Energy-weighted density matrix.
    pub weighted: Mat<f64>,
    /// Orbitals in the Loewdin-orthogonalized atomic orbitals, `S^(1/2) C` (columns).
    pub lowdin: Mat<f64>,
    /// Orbitals in the atomic orbitals, `C` (columns).
    pub coefficients: Mat<f64>,
    /// Occupations of the alpha and of the beta spin.
    pub spin_occupations: [Vec<f64>; 2],
}

fn fermi(x: f64) -> f64 {
    if x >= 50.0 {
        0.0
    } else if x <= -50.0 {
        1.0
    } else {
        1.0 / (x.exp() + 1.0)
    }
}

fn entropy_term(x: f64) -> f64 {
    let f = fermi(x);
    if f > 1.0e-15 && 1.0 - f > 1.0e-15 {
        f * f.ln() + (1.0 - f) * (1.0 - f).ln()
    } else {
        0.0
    }
}

/// Bisection for the chemical potential, centred on gaps (as gfn0-rs).
fn chemical_potential(lower: f64, upper: f64, target: f64, count: impl Fn(f64) -> f64) -> f64 {
    let edge = |inclusive: bool| {
        let (mut lo, mut hi) = (lower, upper);
        for _ in 0..240 {
            let mid = 0.5 * (lo + hi);
            let v = count(mid);
            if (inclusive && v <= target) || (!inclusive && v < target) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let (a, b) = (edge(false), edge(true));
    let centred = 0.5 * (a + b);
    if (count(centred) - target).abs() <= 1.0e-9 {
        centred
    } else {
        a
    }
}

/// Occupations and entropy correction of one spin channel with `ne` electrons at the
/// thermal energy `kt` (Hartree).
fn fill(e: &[f64], ne: usize, kt: f64) -> MoptResult<(Vec<f64>, f64)> {
    let mut f = vec![0.0; e.len()];
    if ne == 0 || ne == e.len() {
        f.iter_mut().take(ne).for_each(|v| *v = 1.0);
        return Ok((f, 0.0));
    }
    let target = ne as f64;
    let lo = e.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let count = |mu: f64| e.iter().map(|&v| fermi((v - mu) / kt)).sum::<f64>();
    let mu = chemical_potential(
        lo - 100.0 * kt - 10.0,
        hi + 100.0 * kt + 10.0,
        target,
        count,
    );
    let mut ts = 0.0;
    for (o, &v) in f.iter_mut().zip(e) {
        let x = (v - mu) / kt;
        *o = fermi(x);
        ts += kt * entropy_term(x);
    }
    let residual = target - f.iter().sum::<f64>();
    if residual.abs() > 1.0e-8 {
        return Err(MoptError::calc(format!(
            "Fermi filling did not conserve the electron count (residual {residual:.3e})"
        )));
    }
    Ok((f, ts))
}

/// Diagonalize and fill: `electrons` valence electrons, `unpaired` of them unpaired, at the
/// electronic temperature of GFN0-xTB.
pub fn solve(
    h: &Mat<f64>,
    s: &Mat<f64>,
    electrons: usize,
    unpaired: usize,
) -> MoptResult<Electronic> {
    let xh = lowdin_orthogonalizer(s)?;
    solve_at(
        h,
        &xh,
        electrons,
        unpaired,
        BOLTZMANN_HARTREE_PER_K * TEMPERATURE_K,
    )
}

/// The symmetric orthogonalizer `S^(-1/2)` of an overlap matrix.
pub fn lowdin_orthogonalizer(s: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let n = s.nrows();
    let (sv, su) = symmetric_eigen(s)?;
    if sv.iter().any(|&v| v <= 1.0e-10) {
        return Err(MoptError::linalg(
            "the overlap matrix of the tight-binding basis is not positive definite",
        ));
    }
    let xs = Mat::from_fn(n, n, |i, j| su[(i, j)] / sv[j].sqrt());
    Ok(mul(xs.as_ref(), su.transpose()))
}

/// [`solve`] with the orthogonalizer `xh = S^(-1/2)` (from [`lowdin_orthogonalizer`]) and
/// the thermal energy `kt` (Hartree) of the Fermi smearing.
pub fn solve_at(
    h: &Mat<f64>,
    xh: &Mat<f64>,
    electrons: usize,
    unpaired: usize,
    kt: f64,
) -> MoptResult<Electronic> {
    solve_with(h, xh, electrons, unpaired, kt, true)
}

/// [`solve_at`] without the energy-weighted density (`weighted` is `0 x 0`), for the
/// iterations of a self-consistent field, which need the density only.
pub fn solve_density_at(
    h: &Mat<f64>,
    xh: &Mat<f64>,
    electrons: usize,
    unpaired: usize,
    kt: f64,
) -> MoptResult<Electronic> {
    solve_with(h, xh, electrons, unpaired, kt, false)
}

/// `C diag(w) C^T` over the columns of `c` with a non-zero weight, as one product of matrices
/// (new; the same sum as the loop over all columns, up to rounding).
pub fn weighted_outer(c: &Mat<f64>, w: &[f64]) -> Mat<f64> {
    let n = c.nrows();
    let cols: Vec<usize> = (0..w.len()).filter(|&k| w[k] != 0.0).collect();
    let scaled = Mat::from_fn(n, cols.len(), |i, k| w[cols[k]] * c[(i, cols[k])]);
    let plain = Mat::from_fn(n, cols.len(), |i, k| c[(i, cols[k])]);
    mul(scaled.as_ref(), plain.transpose())
}

fn solve_with(
    h: &Mat<f64>,
    xh: &Mat<f64>,
    electrons: usize,
    unpaired: usize,
    kt: f64,
    weighted: bool,
) -> MoptResult<Electronic> {
    let n = h.nrows();
    if unpaired > electrons || !(electrons - unpaired).is_multiple_of(2) || electrons > 2 * n {
        return Err(MoptError::config(format!(
            "{electrons} electrons with {unpaired} unpaired do not fit {n} tight-binding orbitals"
        )));
    }
    let ht = mul3(xh.transpose(), h.as_ref(), xh.as_ref());
    let ht = Mat::from_fn(n, n, |i, j| 0.5 * (ht[(i, j)] + ht[(j, i)]));
    let (ev, u) = symmetric_eigen(&ht)?;
    let c = mul(xh.as_ref(), u.as_ref());
    let e: Vec<f64> = ev.iter().copied().collect();
    let (fa, tsa) = fill(&e, (electrons + unpaired) / 2, kt)?;
    let (fb, tsb) = fill(&e, (electrons - unpaired) / 2, kt)?;
    let occ: Vec<f64> = fa.iter().zip(&fb).map(|(a, b)| a + b).collect();
    let band: f64 = e.iter().zip(&occ).map(|(v, o)| v * o).sum();
    let density = weighted_outer(&c, &occ);
    let weighted = if weighted {
        let ew: Vec<f64> = occ.iter().zip(&e).map(|(o, v)| o * v).collect();
        weighted_outer(&c, &ew)
    } else {
        Mat::zeros(0, 0)
    };
    Ok(Electronic {
        energy: band + tsa + tsb,
        eigenvalues: e,
        occupations: occ,
        density,
        weighted,
        lowdin: u,
        coefficients: c,
        spin_occupations: [fa, fb],
    })
}

/// Gradient of the electronic free energy through H0 and S: the Cartesian part is added to
/// `grad`; `dE/d eps` per shell is returned (the self-energy route to CN and q).
pub fn gradient(
    basis: &Basis,
    x: &[[f64; 3]],
    s: &Mat<f64>,
    eps: &[f64],
    el: &Electronic,
    grad: &mut [f64],
) -> MoptResult<Vec<f64>> {
    gradient_with(basis, x, s, eps, &el.density, &el.weighted, grad)
}

/// [`gradient`] with the symmetric `p` in the place of the density and `w` in the place of
/// the energy-weighted density (the relaxed difference density of an excited state and its
/// weights of the overlap, `pes::native::tda_gradient`).
pub fn gradient_with(
    basis: &Basis,
    x: &[[f64; 3]],
    s: &Mat<f64>,
    eps: &[f64],
    p: &Mat<f64>,
    w: &Mat<f64>,
    grad: &mut [f64],
) -> MoptResult<Vec<f64>> {
    let ns = basis.shells.len();
    let mut de_deps = vec![0.0; ns];
    for (k, sh) in basis.shells.iter().enumerate() {
        for mu in sh.first_ao..sh.first_ao + sh.nao {
            de_deps[k] += p[(mu, mu)];
        }
    }
    for a in 0..ns {
        for b in a + 1..ns {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            if sa.atom == sb.atom {
                continue;
            }
            let d = diff(x, sa.atom, sb.atom);
            if basis.negligible(a, b, sq(d)) {
                continue;
            }
            let ds = pair_overlap_gradient(basis, a, b, x[sa.atom], x[sb.atom]);
            let has_h = sa.valence || sb.valence;
            let ps = if has_h {
                Some(pair_scale_of(sa, sb, d)?)
            } else {
                None
            };
            let mut g = [0.0; 3];
            let mut de_dhav = 0.0;
            for (row, mu) in (sa.first_ao..sa.first_ao + sa.nao).enumerate() {
                for (col, nu) in (sb.first_ao..sb.first_ao + sb.nao).enumerate() {
                    let dsv = ds[row * sb.nao + col];
                    // Both triangles: factor 2.
                    let mut coef = -2.0 * w[(mu, nu)];
                    if let Some(ps) = &ps {
                        let hav = 0.5 * ps.km * (eps[a] + eps[b]);
                        coef += 2.0 * p[(mu, nu)] * hav * ps.poly;
                        let sv = s[(mu, nu)];
                        for (gk, dk) in g.iter_mut().zip(ps.dpoly) {
                            *gk += 2.0 * p[(mu, nu)] * hav * sv * dk;
                        }
                        de_dhav += 2.0 * p[(mu, nu)] * sv * ps.poly;
                    }
                    for (gk, dk) in g.iter_mut().zip(dsv) {
                        *gk += coef * dk;
                    }
                }
            }
            if let Some(ps) = &ps {
                de_deps[a] += 0.5 * ps.km * de_dhav;
                de_deps[b] += 0.5 * ps.km * de_dhav;
            }
            for k in 0..3 {
                grad[3 * sa.atom + k] += g[k];
                grad[3 * sb.atom + k] -= g[k];
            }
        }
    }
    Ok(de_deps)
}
