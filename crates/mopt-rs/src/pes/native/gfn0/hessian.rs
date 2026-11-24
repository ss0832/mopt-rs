//! The analytic Hessian of the GFN0-xTB ground state.
//!
//! GFN0-xTB is not self-consistent: its electronic free energy is that of the orbitals of H0,
//! whose shell self energies `eps_a = h_a - k_CN CN'_A - k_q q_A - k_q2 q_A^2` depend on the
//! geometry through the capped erf coordination numbers `CN'` and the EEQ charges `q`. With the
//! gradient `G_x = tr(P dH0/dx) - tr(W dS/dx)`,
//!
//! ```text
//! H_xy = tr(P d2H0/dx dy) - tr(W d2S/dx dy) + tr(dH0/dx P^y) - tr(dS/dx W^y)
//! ```
//!
//! where `P^y`, `W^y` are the response of the Fermi-smeared density to `dH0/dy`, `dS/dy`
//! ([`crate::pes::native::tb_response`]). `d2H0` holds the second derivatives of the overlaps
//! and of the distance polynomial, their cross terms with the self energies, and
//! `d2 eps_a = -k_CN d2CN' - (k_q + 2 k_q2 q) d2q - 2 k_q2 dq dq`.
//!
//! The EEQ charges solve the bordered system `K [q; mu] = [X; Q]`; their derivatives
//! `K [q_x; .] = [X_x - A_x q; 0]`, and a weighted sum of their second derivatives is one
//! adjoint solve, `K lambda = [w; 0]`:
//! `w . q_xy = lambda . (X_xy - A_xy q - A_x q_y - A_y q_x)`. The electrostatic energy
//! `E = q (A q / 2 - X)` is stationary in `q` under the constraint, so
//! `E_xy = q A_xy q / 2 - q X_xy + q_y . (A_x q - X_x)`. The second derivatives of `X`, of the
//! short-range basis correction and of D4 go through those of the coordination numbers, which
//! are contracted with their weights rather than formed per atom. D4 uses C6 with its second
//! derivatives in the covalent CN and the EEQ-2019 charges (`D4Model::c6_second`).
//!
//! The plan follows the analytic Hessian of gfn0-rs (GPL-3.0-or-later); the code is written
//! here from the energy expression of [`super`] and checked against central differences of
//! the analytic gradient in the tests.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::basis::{pair_overlap_gradient, pair_overlap_hessian};
use super::cn::smooth_cutoff;
use super::eeq::EeqParams;
use super::electronic::{pair_scale_of, TEMPERATURE_K};
use super::params::{ElementParams, BOLTZMANN_HARTREE_PER_K};
use super::Gfn0Xtb;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::mul3;
use crate::math::stable::{logistic, softplus};
use crate::parameters::gfn0_xtb as g;
use crate::pes::dispersion::d4::D4Model;
use crate::pes::native::tb_response::OrbitalResponse;

type M3 = [[f64; 3]; 3];

const SQRT_PI: f64 = 1.772_453_850_905_516;
const CN_MAX: f64 = 8.0;
const SRB_CUTOFF: f64 = 14.142_135_623_730_95;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The second derivative with respect to `R_A` of `F(|R_A - R_B|)` (`F'` = `f1`, `F''` =
/// `f2`, `d = R_A - R_B`).
fn radial(d: [f64; 3], f1: f64, f2: f64) -> M3 {
    let r2 = dot(d, d);
    let r = r2.sqrt();
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            let dd = d[i] * d[j] / r2;
            let delta = if i == j { 1.0 } else { 0.0 };
            *v = f2 * dd + f1 / r * (delta - dd);
        }
    }
    out
}

/// Adds `scale` times the Hessian of a function of `R_A - R_B` with the block `block`.
fn add_pair(h: &mut Mat<f64>, a: usize, b: usize, block: &M3, scale: f64) {
    for i in 0..3 {
        for j in 0..3 {
            let v = scale * block[i][j];
            h[(3 * a + i, 3 * a + j)] += v;
            h[(3 * b + i, 3 * b + j)] += v;
            h[(3 * a + i, 3 * b + j)] -= v;
            h[(3 * b + i, 3 * a + j)] -= v;
        }
    }
}

/// `h += s (u v^T + v u^T)`. Where `u` is sparse (the coordinates of a pair of atoms), a row
/// `x` with `u_x = 0` is `s v_x u^T`, non-zero only where `u` is: those rows visit the
/// non-zero entries of `u` alone (the entries left out would add zeros), so that the cost is
/// that of the non-zeros of `u` times the length rather than the length squared.
fn add_sym_outer(h: &mut Mat<f64>, u: &[f64], v: &[f64], s: f64) {
    let nonzero: Vec<usize> = (0..u.len()).filter(|&y| u[y] != 0.0).collect();
    for x in 0..u.len() {
        let (ux, vx) = (s * u[x], s * v[x]);
        if ux == 0.0 && vx == 0.0 {
            continue;
        }
        if ux != 0.0 {
            for y in 0..u.len() {
                h[(x, y)] += ux * v[y] + vx * u[y];
            }
        } else {
            for &y in &nonzero {
                h[(x, y)] += ux * v[y] + vx * u[y];
            }
        }
    }
}

/// `h += s u u^T`.
fn add_square(h: &mut Mat<f64>, u: &[f64], s: f64) {
    if s == 0.0 {
        return;
    }
    for x in 0..u.len() {
        let ux = s * u[x];
        if ux == 0.0 {
            continue;
        }
        for y in 0..u.len() {
            h[(x, y)] += ux * u[y];
        }
    }
}

/// `d|R_A - R_B| / dR` as a coordinate vector.
fn distance_gradient(dof: usize, a: usize, b: usize, d: [f64; 3]) -> Vec<f64> {
    let r = dot(d, d).sqrt();
    let mut out = vec![0.0; dof];
    for k in 0..3 {
        out[3 * a + k] = d[k] / r;
        out[3 * b + k] = -d[k] / r;
    }
    out
}

/// The smooth cutoff of [`smooth_cutoff`] with its second radial derivative.
fn cutoff_second(r: f64, on: f64, off: f64) -> (f64, f64, f64) {
    let (f, df) = smooth_cutoff(r, on, off);
    if r <= on || r >= off {
        return (f, df, 0.0);
    }
    let w = off - on;
    let t = (r - on) / w;
    let u = 1.0 - t;
    let x1 = 1.0 / (t * t) + 1.0 / (u * u);
    let x2 = -2.0 / (t * t * t) + 2.0 / (u * u * u);
    let d2 = -((1.0 - 2.0 * f) * df * x1 / w + f * (1.0 - f) * x2 / (w * w));
    (f, df, d2)
}

/// One pair of the coordination numbers.
struct CnPair {
    i: usize,
    j: usize,
    d: [f64; 3],
    /// `d count / dr` and `d2 count / dr2` of the pair (with the cutoff).
    c1: f64,
    c2: f64,
    /// The EN weight of the covalent CN.
    den: f64,
}

/// The coordination numbers of [`super::cn`] with their first derivatives.
struct CnSecond {
    pairs: Vec<CnPair>,
    /// `dCN_i/dx` before the cap, of the capped CN', and of the covalent CN (atoms by
    /// coordinates).
    jraw: Vec<Vec<f64>>,
    jerf: Vec<Vec<f64>>,
    jcov: Vec<Vec<f64>>,
    /// `dCN'/dCN` and `d2CN'/dCN2` of the cap.
    s1: Vec<f64>,
    s2: Vec<f64>,
    erf: Vec<f64>,
    covalent: Vec<f64>,
}

fn en_weight(a: &ElementParams, b: &ElementParams) -> f64 {
    let d = (a.pauling_en - b.pauling_en).abs();
    4.10451 * (-((d + 19.08857) * (d + 19.08857)) / (2.0 * 11.28174 * 11.28174)).exp()
}

impl CnSecond {
    fn new(el: &[ElementParams], x: &[[f64; 3]]) -> Self {
        let n = x.len();
        let dof = 3 * n;
        let mut raw = vec![0.0; n];
        let mut covalent = vec![0.0; n];
        let mut pairs = Vec::new();
        let mut jraw = vec![vec![0.0; dof]; n];
        let mut jcov = vec![vec![0.0; dof]; n];
        for i in 0..n {
            for j in 0..i {
                let d = sub(x[i], x[j]);
                let r = dot(d, d).sqrt();
                if r >= 40.0 || r <= 1.0e-12 {
                    continue;
                }
                let (cut, dcut, d2cut) = cutoff_second(r, 35.0, 40.0);
                let rc = (el[i].covalent_radius + el[j].covalent_radius) * (4.0 / 3.0);
                let arg = -7.5 * (r - rc) / rc;
                let count = 0.5 * (1.0 + libm::erf(arg));
                let e = (-(arg * arg)).exp();
                let dcount = -7.5 / (SQRT_PI * rc) * e;
                let d2count = -112.5 * arg / (SQRT_PI * rc * rc) * e;
                let base = count * cut;
                let c1 = dcount * cut + count * dcut;
                let c2 = d2count * cut + 2.0 * dcount * dcut + count * d2cut;
                let den = en_weight(&el[i], &el[j]);
                raw[i] += base;
                raw[j] += base;
                covalent[i] += den * base;
                covalent[j] += den * base;
                for k in 0..3 {
                    let gk = c1 * d[k] / r;
                    for atom in [i, j] {
                        jraw[atom][3 * i + k] += gk;
                        jraw[atom][3 * j + k] -= gk;
                        jcov[atom][3 * i + k] += den * gk;
                        jcov[atom][3 * j + k] -= den * gk;
                    }
                }
                pairs.push(CnPair {
                    i,
                    j,
                    d,
                    c1,
                    c2,
                    den,
                });
            }
        }
        let cap = softplus(CN_MAX);
        let erf: Vec<f64> = raw.iter().map(|&c| cap - softplus(CN_MAX - c)).collect();
        let s1: Vec<f64> = raw.iter().map(|&c| logistic(CN_MAX - c)).collect();
        let s2: Vec<f64> = s1.iter().map(|&s| -s * (1.0 - s)).collect();
        let jerf = (0..n)
            .map(|a| jraw[a].iter().map(|v| s1[a] * v).collect())
            .collect();
        Self {
            pairs,
            jraw,
            jerf,
            jcov,
            s1,
            s2,
            erf,
            covalent,
        }
    }

    /// `h += sum_i w_erf_i d2CN'_i + w_cov_i d2CN^cov_i`.
    fn add_second(&self, h: &mut Mat<f64>, w_erf: &[f64], w_cov: &[f64]) {
        for p in &self.pairs {
            let s = w_erf[p.i] * self.s1[p.i]
                + w_erf[p.j] * self.s1[p.j]
                + (w_cov[p.i] + w_cov[p.j]) * p.den;
            add_pair(h, p.i, p.j, &radial(p.d, p.c1, p.c2), s);
        }
        for (a, w) in w_erf.iter().enumerate() {
            add_square(h, &self.jraw[a], w * self.s2[a]);
        }
    }
}

/// One pair of the EEQ kernel: `A_ij(r)` with `dA/dr`, `d2A/dr2`.
struct KernelPair {
    i: usize,
    j: usize,
    d: [f64; 3],
    a1: f64,
    a2: f64,
}

/// A solved EEQ system of [`super::eeq`] with the first derivatives of its charges.
struct EeqSecond {
    q: Vec<f64>,
    /// `dq_i/dx` (atoms by coordinates).
    jq: Vec<Vec<f64>>,
    /// The inverse of the bordered matrix.
    kinv: Mat<f64>,
    pairs: Vec<KernelPair>,
    /// `dX_i/dCN'_i`, `d2X_i/dCN'_i^2`.
    f1: Vec<f64>,
    f2: Vec<f64>,
}

/// `dX/dCN` and `d2X/dCN2` of `X = kappa CN / (sqrt(CN + 1e-12) + 1e-14)` (zero for
/// `CN <= 0`, where the right-hand side is clamped).
fn rhs_derivatives(cn: f64, kappa: f64) -> (f64, f64) {
    if cn <= 0.0 {
        return (0.0, 0.0);
    }
    let root = (cn + 1.0e-12).sqrt();
    let den = root + 1.0e-14;
    let first = kappa * (den - cn / (2.0 * root)) / (den * den);
    let second = kappa
        * (cn / (4.0 * root * root * root * den * den) - 1.0 / (root * den * den)
            + cn / (2.0 * root * root * den * den * den));
    (first, second)
}

impl EeqSecond {
    fn new(
        p: &[EeqParams],
        x: &[[f64; 3]],
        cn: &CnSecond,
        charge: f64,
        scale: f64,
    ) -> MoptResult<Self> {
        let n = x.len();
        let dof = 3 * n;
        let widths: Vec<f64> = p.iter().map(|v| v[3].max(1.0e-12)).collect();
        let cself = (2.0 / SQRT_PI) * std::f64::consts::FRAC_1_SQRT_2;
        let mut kkt = Mat::<f64>::zeros(n + 1, n + 1);
        let mut pairs = Vec::new();
        for i in 0..n {
            kkt[(i, i)] = p[i][1] + scale * cself / widths[i];
            kkt[(i, n)] = 1.0;
            kkt[(n, i)] = 1.0;
            for j in 0..i {
                let d = sub(x[i], x[j]);
                let r = dot(d, d).sqrt();
                if r <= 1.0e-12 {
                    return Err(MoptError::calc("coincident atoms in the EEQ model"));
                }
                let gamma = (widths[i] * widths[i] + widths[j] * widths[j]).sqrt();
                let s = r / gamma;
                let e = (-(s * s)).exp();
                let erf = libm::erf(s);
                kkt[(i, j)] = scale * erf / r;
                kkt[(j, i)] = kkt[(i, j)];
                let pre = 2.0 / (SQRT_PI * gamma);
                let a1 = scale * (pre * e / r - erf / (r * r));
                let a2 = scale
                    * (-2.0 * pre * e / (gamma * gamma) - 2.0 * pre * e / (r * r)
                        + 2.0 * erf / (r * r * r));
                pairs.push(KernelPair { i, j, d, a1, a2 });
            }
        }
        let lu = crate::math::linalg::DenseLu::new(kkt.as_ref())?;
        let eye = Mat::<f64>::identity(n + 1, n + 1);
        let kinv = lu.solve(eye.as_ref());
        if kinv.col_iter().any(|c| c.iter().any(|v| !v.is_finite())) {
            return Err(MoptError::linalg("EEQ linear system is singular"));
        }
        let rhs_x: Vec<f64> = (0..n)
            .map(|i| {
                let c = cn.erf[i].max(0.0);
                -p[i][0] + p[i][2] * c / ((c + 1.0e-12).sqrt() + 1.0e-14)
            })
            .collect();
        let mut b = Col::<f64>::zeros(n + 1);
        for (i, v) in rhs_x.iter().enumerate() {
            b[i] = *v;
        }
        b[n] = charge;
        let q: Vec<f64> = (0..n)
            .map(|i| (0..=n).map(|k| kinv[(i, k)] * b[k]).sum())
            .collect();
        let (f1, f2): (Vec<f64>, Vec<f64>) =
            (0..n).map(|i| rhs_derivatives(cn.erf[i], p[i][2])).unzip();
        // The right-hand sides X_x - A_x q of the charge derivatives.
        let mut rx: Vec<Vec<f64>> = (0..n)
            .map(|i| cn.jerf[i].iter().map(|v| f1[i] * v).collect())
            .collect();
        for kp in &pairs {
            let r = dot(kp.d, kp.d).sqrt();
            for k in 0..3 {
                let gk = kp.a1 * kp.d[k] / r;
                rx[kp.i][3 * kp.i + k] -= gk * q[kp.j];
                rx[kp.j][3 * kp.i + k] -= gk * q[kp.i];
                rx[kp.i][3 * kp.j + k] += gk * q[kp.j];
                rx[kp.j][3 * kp.j + k] += gk * q[kp.i];
            }
        }
        let jq = (0..n)
            .map(|t| {
                (0..dof)
                    .map(|xx| (0..n).map(|i| kinv[(t, i)] * rx[i][xx]).sum())
                    .collect()
            })
            .collect();
        Ok(Self {
            q,
            jq,
            kinv,
            pairs,
            f1,
            f2,
        })
    }

    /// Adds `s sum_ij dA_ij/dx (u_i dq_j/dy + u_j dq_i/dy)` to `h` (and its transpose with
    /// `transpose`).
    fn add_kernel_cross(&self, h: &mut Mat<f64>, u: &[f64], s: f64, transpose: bool) {
        let dof = h.nrows();
        for kp in &self.pairs {
            let r = dot(kp.d, kp.d).sqrt();
            let uy: Vec<f64> = (0..dof)
                .map(|y| s * (u[kp.i] * self.jq[kp.j][y] + u[kp.j] * self.jq[kp.i][y]))
                .collect();
            for k in 0..3 {
                let gk = kp.a1 * kp.d[k] / r;
                let (xi, xj) = (3 * kp.i + k, 3 * kp.j + k);
                for (y, v) in uy.iter().enumerate() {
                    let c = gk * v;
                    h[(xi, y)] += c;
                    h[(xj, y)] -= c;
                    if transpose {
                        h[(y, xi)] += c;
                        h[(y, xj)] -= c;
                    }
                }
            }
        }
    }

    /// Adds `sum_t w_t d2q_t/dx dy` to `h`, except for the part through the second
    /// derivatives of the CN', whose weights are added to `w_erf`.
    fn add_charge_second(&self, cn: &CnSecond, w: &[f64], h: &mut Mat<f64>, w_erf: &mut [f64]) {
        let n = w.len();
        let lam: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|t| self.kinv[(i, t)] * w[t]).sum())
            .collect();
        for (i, w) in w_erf.iter_mut().enumerate().take(n) {
            *w += lam[i] * self.f1[i];
            add_square(h, &cn.jerf[i], lam[i] * self.f2[i]);
        }
        for kp in &self.pairs {
            let s = -(lam[kp.i] * self.q[kp.j] + lam[kp.j] * self.q[kp.i]);
            add_pair(h, kp.i, kp.j, &radial(kp.d, kp.a1, kp.a2), s);
        }
        self.add_kernel_cross(h, &lam, -1.0, true);
    }

    /// Adds the Hessian of the electrostatic energy `q (A q / 2 - X)` to `h`, except for the
    /// part through the second derivatives of the CN', whose weights are added to `w_erf`.
    fn add_energy_second(&self, cn: &CnSecond, h: &mut Mat<f64>, w_erf: &mut [f64]) {
        let n = self.q.len();
        let dof = h.nrows();
        for kp in &self.pairs {
            let s = self.q[kp.i] * self.q[kp.j];
            add_pair(h, kp.i, kp.j, &radial(kp.d, kp.a1, kp.a2), s);
        }
        for (i, w) in w_erf.iter_mut().enumerate().take(n) {
            *w -= self.q[i] * self.f1[i];
            add_square(h, &cn.jerf[i], -self.q[i] * self.f2[i]);
        }
        // q_y . (A_x q - X_x): symmetric as a whole, added once.
        self.add_kernel_cross(h, &self.q, 1.0, false);
        for i in 0..n {
            for xx in 0..dof {
                let c = self.f1[i] * cn.jerf[i][xx];
                if c == 0.0 {
                    continue;
                }
                for y in 0..dof {
                    h[(xx, y)] -= c * self.jq[i][y];
                }
            }
        }
    }
}

/// A shell pair of H0 and S on two atoms.
struct ShellPair {
    a: usize,
    b: usize,
    ia: usize,
    ib: usize,
    first: (usize, usize),
    size: (usize, usize),
    s: Vec<f64>,
    ds: Vec<[f64; 3]>,
    d2s: Vec<M3>,
    /// `(K, poly, dpoly/dR_A, d2poly/dR_A2)` where H0 couples the pair.
    h: Option<(f64, f64, [f64; 3], M3)>,
}

impl Gfn0Xtb {
    /// The analytic Hessian (Hartree/Bohr^2) of the ground state.
    pub fn analytic_hessian(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        let setup = self.setup(sys)?;
        let el = &setup.el;
        let x = &setup.x;
        let n = x.len();
        let dof = 3 * n;
        let basis = &setup.basis;
        let ns = basis.shells.len();
        let nao = basis.len();
        let elec = &setup.elec;
        let (p, w) = (&elec.density, &elec.weighted);
        let eps = &setup.eps;

        let cn = CnSecond::new(el, x);
        let gfn0_p: Vec<EeqParams> = el.iter().map(|e| [e.xi, e.gam, e.kappa, e.alpg]).collect();
        let eeq = EeqSecond::new(&gfn0_p, x, &cn, setup.charge, g::GFN0_EEQ_JSCALE)?;
        let q = &eeq.q;

        // d eps_a / dx.
        let jeps: Vec<Vec<f64>> = basis
            .shells
            .iter()
            .map(|sh| {
                let at = sh.atom;
                let kq = sh.kq + 2.0 * sh.kqat2 * q[at];
                (0..dof)
                    .map(|y| -sh.kcn * cn.jerf[at][y] - kq * eeq.jq[at][y])
                    .collect()
            })
            .collect();

        let mut pairs = Vec::new();
        for a in 0..ns {
            for b in a + 1..ns {
                let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
                if sa.atom == sb.atom {
                    continue;
                }
                let d = sub(x[sa.atom], x[sb.atom]);
                if basis.negligible(a, b, dot(d, d)) {
                    continue;
                }
                let h = if sa.valence || sb.valence {
                    let ps = pair_scale_of(sa, sb, d)?;
                    let r = dot(d, d).sqrt();
                    let xr = (r / (sa.atomic_radius + sb.atomic_radius)).sqrt();
                    let (ka, kb) = (0.01 * sa.poly, 0.01 * sb.poly);
                    let x1 = xr / (2.0 * r);
                    let x2 = -xr / (4.0 * r * r);
                    let lin = ka * (1.0 + kb * xr) + kb * (1.0 + ka * xr);
                    let d2poly = radial(d, lin * x1, 2.0 * ka * kb * x1 * x1 + lin * x2);
                    Some((ps.km, ps.poly, ps.dpoly, d2poly))
                } else {
                    None
                };
                let mut s = Vec::with_capacity(sa.nao * sb.nao);
                for mu in sa.first_ao..sa.first_ao + sa.nao {
                    for nu in sb.first_ao..sb.first_ao + sb.nao {
                        s.push(setup.s[(mu, nu)]);
                    }
                }
                pairs.push(ShellPair {
                    a,
                    b,
                    ia: sa.atom,
                    ib: sb.atom,
                    first: (sa.first_ao, sb.first_ao),
                    size: (sa.nao, sb.nao),
                    s,
                    ds: pair_overlap_gradient(basis, a, b, x[sa.atom], x[sb.atom]),
                    d2s: pair_overlap_hessian(basis, a, b, x[sa.atom], x[sb.atom]),
                    h,
                });
            }
        }

        let mut hm = Mat::<f64>::zeros(dof, dof);
        // The explicit part at fixed P, W, with dE/d eps_a for the self-energy chain.
        let mut de_deps: Vec<f64> = basis
            .shells
            .iter()
            .map(|sh| {
                (sh.first_ao..sh.first_ao + sh.nao)
                    .map(|mu| p[(mu, mu)])
                    .sum()
            })
            .collect();
        for pr in &pairs {
            let ((fa, fb), (na, nb)) = (pr.first, pr.size);
            let (mut t1, mut dt1, mut d2t1) = (0.0, [0.0; 3], [[0.0; 3]; 3]);
            let mut d2t2 = [[0.0; 3]; 3];
            for r in 0..na {
                for c in 0..nb {
                    let (mu, nu) = (fa + r, fb + c);
                    let k = r * nb + c;
                    let pw = 2.0 * p[(mu, nu)];
                    let ww = 2.0 * w[(mu, nu)];
                    t1 += pw * pr.s[k];
                    for i in 0..3 {
                        dt1[i] += pw * pr.ds[k][i];
                        for j in 0..3 {
                            d2t1[i][j] += pw * pr.d2s[k][i][j];
                            d2t2[i][j] += ww * pr.d2s[k][i][j];
                        }
                    }
                }
            }
            let mut bl = d2t2.map(|row| row.map(|v| -v));
            if let Some((km, poly, dpoly, d2poly)) = pr.h {
                let hav = 0.5 * km * (eps[pr.a] + eps[pr.b]);
                for i in 0..3 {
                    for j in 0..3 {
                        bl[i][j] += hav
                            * (d2poly[i][j] * t1
                                + dpoly[i] * dt1[j]
                                + dt1[i] * dpoly[j]
                                + poly * d2t1[i][j]);
                    }
                }
                let half = 0.5 * km;
                de_deps[pr.a] += half * poly * t1;
                de_deps[pr.b] += half * poly * t1;
                // Cross terms of the pair geometry with the self energies.
                let mut geo = vec![0.0; dof];
                for k in 0..3 {
                    let v = half * (dpoly[k] * t1 + poly * dt1[k]);
                    geo[3 * pr.ia + k] += v;
                    geo[3 * pr.ib + k] -= v;
                }
                let both: Vec<f64> = (0..dof).map(|y| jeps[pr.a][y] + jeps[pr.b][y]).collect();
                add_sym_outer(&mut hm, &geo, &both, 1.0);
            }
            add_pair(&mut hm, pr.ia, pr.ib, &bl, 1.0);
        }
        // sum_a dE/d eps_a d2 eps_a.
        let mut w_erf = vec![0.0; n];
        let mut w_cov = vec![0.0; n];
        let mut w_q = vec![0.0; n];
        for (a, sh) in basis.shells.iter().enumerate() {
            let at = sh.atom;
            w_erf[at] -= de_deps[a] * sh.kcn;
            w_q[at] -= de_deps[a] * (sh.kq + 2.0 * sh.kqat2 * q[at]);
            add_square(&mut hm, &eeq.jq[at], -2.0 * de_deps[a] * sh.kqat2);
        }

        // The response of the density.
        let kt = BOLTZMANN_HARTREE_PER_K * TEMPERATURE_K;
        let [fa_occ, fb_occ] = &elec.spin_occupations;
        let orb = OrbitalResponse::new(
            elec.coefficients.clone(),
            elec.eigenvalues.clone(),
            [fa_occ.as_slice(), fb_occ.as_slice()],
            kt,
        );
        // One column per coordinate, independent of the others: in parallel, each in its own
        // place, at most as many at once as fit in the memory budget.
        let per_task = crate::core_infra::memory::f64_bytes(12.0 * (nao * nao) as f64);
        let response_column = |y: usize| -> Vec<f64> {
            let (atom, k) = (y / 3, y % 3);
            let mut fy = Mat::<f64>::zeros(nao, nao);
            let mut sy = Mat::<f64>::zeros(nao, nao);
            for (a, sh) in basis.shells.iter().enumerate() {
                for mu in sh.first_ao..sh.first_ao + sh.nao {
                    fy[(mu, mu)] += jeps[a][y];
                }
            }
            for pr in &pairs {
                let ((fa, fb), (na, nb)) = (pr.first, pr.size);
                let sign = if atom == pr.ia {
                    1.0
                } else if atom == pr.ib {
                    -1.0
                } else {
                    0.0
                };
                let deps = jeps[pr.a][y] + jeps[pr.b][y];
                if sign == 0.0 && (pr.h.is_none() || deps == 0.0) {
                    continue;
                }
                for r in 0..na {
                    for c in 0..nb {
                        let kk = r * nb + c;
                        let (mu, nu) = (fa + r, fb + c);
                        let mut f = 0.0;
                        if let Some((km, poly, dpoly, _)) = pr.h {
                            let hav = 0.5 * km * (eps[pr.a] + eps[pr.b]);
                            f += 0.5 * km * deps * poly * pr.s[kk];
                            f += sign * hav * (poly * pr.ds[kk][k] + dpoly[k] * pr.s[kk]);
                        }
                        fy[(mu, nu)] += f;
                        fy[(nu, mu)] += f;
                        let ds = sign * pr.ds[kk][k];
                        sy[(mu, nu)] += ds;
                        sy[(nu, mu)] += ds;
                    }
                }
            }
            let r = orb.respond(&orb.to_orbitals(&fy), Some(&orb.to_orbitals(&sy)));
            // tr(dH0/dx P^y) - tr(dS/dx W^y) for every x.
            let mut col = vec![0.0; dof];
            let mut deps_y: Vec<f64> = basis
                .shells
                .iter()
                .map(|sh| {
                    (sh.first_ao..sh.first_ao + sh.nao)
                        .map(|mu| r.p[(mu, mu)])
                        .sum()
                })
                .collect();
            for pr in &pairs {
                let ((fa, fb), (na, nb)) = (pr.first, pr.size);
                let mut lp = [0.0; 3];
                let mut ps = 0.0;
                for rr in 0..na {
                    for c in 0..nb {
                        let kk = rr * nb + c;
                        let (mu, nu) = (fa + rr, fb + c);
                        let p2 = 2.0 * r.p[(mu, nu)];
                        let w2 = 2.0 * r.w[(mu, nu)];
                        ps += p2 * pr.s[kk];
                        for (kx, l) in lp.iter_mut().enumerate() {
                            *l -= w2 * pr.ds[kk][kx];
                            if let Some((km, poly, dpoly, _)) = pr.h {
                                let hav = 0.5 * km * (eps[pr.a] + eps[pr.b]);
                                *l += p2 * hav * (poly * pr.ds[kk][kx] + dpoly[kx] * pr.s[kk]);
                            }
                        }
                    }
                }
                for kx in 0..3 {
                    col[3 * pr.ia + kx] += lp[kx];
                    col[3 * pr.ib + kx] -= lp[kx];
                }
                if let Some((km, poly, _, _)) = pr.h {
                    let half = 0.5 * km * poly * ps;
                    deps_y[pr.a] += half;
                    deps_y[pr.b] += half;
                }
            }
            for (xx, c) in col.iter_mut().enumerate() {
                let chain: f64 = (0..ns).map(|a| jeps[a][xx] * deps_y[a]).sum();
                *c += chain;
            }
            col
        };
        let columns =
            crate::pes::native::fd_hessian::parallel_within_memory(dof, per_task, response_column);
        for (y, col) in columns.iter().enumerate() {
            for (xx, c) in col.iter().enumerate() {
                hm[(xx, y)] += c;
            }
        }

        // The charges of H0 (weights w_q) and the electrostatic energy.
        eeq.add_charge_second(&cn, &w_q, &mut hm, &mut w_erf);
        eeq.add_energy_second(&cn, &mut hm, &mut w_erf);

        // Repulsion with its smooth cutoff.
        for i in 0..n {
            for j in i + 1..n {
                let d = sub(x[i], x[j]);
                let r = dot(d, d).sqrt();
                if r >= 40.0 || r <= 1.0e-12 {
                    continue;
                }
                let den = el[i].en - el[j].en;
                let den2 = den * den;
                let alpha = (el[i].repa * el[j].repa).max(1.0e-18).sqrt()
                    * (1.0 + (0.01 * den2 + 0.01 * den2 * den2) * g::GFN0_RENSCALE);
                let v = (-alpha * r.powf(1.5)).exp() * el[i].repb * el[j].repb / r;
                let l1 = -1.5 * alpha * r.sqrt() - 1.0 / r;
                let l2 = -0.75 * alpha / r.sqrt() + 1.0 / (r * r);
                let (v1, v2) = (v * l1, v * (l1 * l1 + l2));
                let (cut, dcut, d2cut) = cutoff_second(r, 35.0, 40.0);
                let e1 = v1 * cut + v * dcut;
                let e2 = v2 * cut + 2.0 * v1 * dcut + v * d2cut;
                add_pair(&mut hm, i, j, &radial(d, e1, e2), 1.0);
            }
        }

        // The short-range basis correction: V(r - r0(CN'_i, CN'_j)).
        for i in 0..n {
            for j in i + 1..n {
                let (zi, zj) = (el[i].z, el[j].z);
                if !(5..=9).contains(&zi) || !(5..=9).contains(&zj) || zi == zj {
                    continue;
                }
                let d = sub(x[i], x[j]);
                let r = dot(d, d).sqrt();
                if r >= SRB_CUTOFF {
                    continue;
                }
                let (si, sj) = (el[i].srb, el[j].srb);
                let (ri, rj) = (period_row(zi), period_row(zj));
                let dn = (si[2] - sj[2]).abs();
                let k1 = 0.005 * (g::GFN0_SRB_P[ri][0] + g::GFN0_SRB_P[rj][0]);
                let k2 = 0.005 * (g::GFN0_SRB_P[ri][1] + g::GFN0_SRB_P[rj][1]);
                let ff = 1.0 - k1 * dn - k2 * dn * dn;
                let ra = si[0] + si[1] * cn.erf[i] + g::GFN0_SRBSHIFT;
                let rb = sj[0] + sj[1] * cn.erf[j] + g::GFN0_SRBSHIFT;
                let r0 = (ra + rb) * ff;
                let pen = si[3] - sj[3];
                let expo = g::GFN0_SRBEXP * (1.0 + g::GFN0_SRBKEN * pen * pen);
                let u = r - r0;
                let v = g::GFN0_SRBPRE * (-expo * u * u).exp();
                let er = -2.0 * expo * u * v;
                let err = (-2.0 * expo + 4.0 * expo * expo * u * u) * v;
                let (ci, cj) = (ff * si[1], ff * sj[1]);
                // E_xy = err (r_x - r0_x)(r_y - r0_y) + er (r_xy - r0_xy).
                add_pair(&mut hm, i, j, &radial(d, er, err), 1.0);
                w_erf[i] -= er * ci;
                w_erf[j] -= er * cj;
                let dr = distance_gradient(dof, i, j, d);
                let r0x: Vec<f64> = (0..dof)
                    .map(|k| ci * cn.jerf[i][k] + cj * cn.jerf[j][k])
                    .collect();
                add_sym_outer(&mut hm, &dr, &r0x, -err);
                add_square(&mut hm, &r0x, err);
            }
        }

        // D4 with the covalent CN and the EEQ-2019 charges: E = -C6 g(r).
        let d4p: Vec<EeqParams> = el.iter().map(|e| e.eeq2019).collect();
        let eeq4 = EeqSecond::new(&d4p, x, &cn, setup.charge, g::GFN0_EEQ_JSCALE)?;
        let model = D4Model::new(&el.iter().map(|e| e.z).collect::<Vec<_>>())?;
        let c6 = model.c6_second(&cn.covalent, &eeq4.q);
        let mut w_q4 = vec![0.0; n];
        // sum over the pairs of g C6_ml J_m J_l^T (J the Jacobians of the covalent CN and the
        // charges of the two atoms) as V^T M V with V the stacked Jacobians: the coefficients
        // are collected per pair, the products formed once (new; a loop over the coordinates
        // per pair costs N^2 times the square of the coordinates).
        let mut coefficients = Mat::<f64>::zeros(2 * n, 2 * n);
        for i in 0..n {
            for j in i + 1..n {
                let d = sub(x[i], x[j]);
                let r = dot(d, d).sqrt();
                if r >= 60.0 || r <= 1.0e-12 {
                    continue;
                }
                let c8 = 3.0 * model.q4(i) * model.q4(j);
                let r0 = g::GFN0_A1 * c8.max(0.0).sqrt() + g::GFN0_A2;
                // t_m = 1 / (r^m + r0^m) with its radial derivatives.
                let term = |m: i32| {
                    let t = 1.0 / (r.powi(m) + r0.powi(m));
                    let mf = f64::from(m);
                    let t1 = -mf * r.powi(m - 1) * t * t;
                    let t2 = -mf * (mf - 1.0) * r.powi(m - 2) * t * t
                        + 2.0 * mf * mf * r.powi(2 * m - 2) * t * t * t;
                    (t, t1, t2)
                };
                let (s6, s8) = (g::GFN0_S6, g::GFN0_S8 * c8);
                let (t6, a6, b6) = term(6);
                let (t8, a8, b8) = term(8);
                let (f, f1, f2) = (s6 * t6 + s8 * t8, s6 * a6 + s8 * a8, s6 * b6 + s8 * b8);
                let (cut, dcut, d2cut) = cutoff_second(r, 55.0, 60.0);
                let gv = f * cut;
                let g1 = f1 * cut + f * dcut;
                let g2 = f2 * cut + 2.0 * f1 * dcut + f * d2cut;
                let cs = &c6[i * n + j];
                // E_xy = -C6 g_xy - g' (r_x C6_y + C6_x r_y) - g C6_xy.
                add_pair(&mut hm, i, j, &radial(d, -cs.c6 * g1, -cs.c6 * g2), 1.0);
                w_cov[i] -= cs.d[0] * gv;
                w_cov[j] -= cs.d[1] * gv;
                w_q4[i] -= cs.d[2] * gv;
                w_q4[j] -= cs.d[3] * gv;
                let jv: [&Vec<f64>; 4] = [&cn.jcov[i], &cn.jcov[j], &eeq4.jq[i], &eeq4.jq[j]];
                let lin: Vec<f64> = (0..dof)
                    .map(|k| (0..4).map(|m| cs.d[m] * jv[m][k]).sum())
                    .collect();
                let dr = distance_gradient(dof, i, j, d);
                add_sym_outer(&mut hm, &dr, &lin, -g1);
                let slot = [i, j, n + i, n + j];
                for m in 0..4 {
                    for l in 0..4 {
                        coefficients[(slot[m], slot[l])] += gv * cs.dd[m][l];
                    }
                }
            }
        }
        let stacked = Mat::from_fn(2 * n, dof, |r, k| {
            if r < n {
                cn.jcov[r][k]
            } else {
                eeq4.jq[r - n][k]
            }
        });
        hm -= mul3(stacked.transpose(), coefficients.as_ref(), stacked.as_ref());
        eeq4.add_charge_second(&cn, &w_q4, &mut hm, &mut w_erf);

        // The second derivatives of the coordination numbers, collected from every term.
        cn.add_second(&mut hm, &w_erf, &w_cov);
        Ok(Mat::from_fn(dof, dof, |i, j| {
            0.5 * (hm[(i, j)] + hm[(j, i)])
        }))
    }
}

/// The row of the SRB polynomial (xtb's `itr`: every element beyond argon in the fourth).
fn period_row(z: u8) -> usize {
    match z {
        1..=2 => 0,
        3..=10 => 1,
        11..=18 => 2,
        _ => 3,
    }
}
