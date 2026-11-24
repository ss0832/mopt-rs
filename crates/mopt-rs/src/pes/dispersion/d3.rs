//! DFT-D3 dispersion with Becke-Johnson (rational) damping, two-body part (S. Grimme,
//! J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys. 2010, 132, 154104; S. Grimme, S. Ehrlich,
//! L. Goerigk, J. Comput. Chem. 2011, 32, 1456), following simple-dftd3:
//!
//! `E = -sum_{i<j} C6_ij (s6 / (r^6 + R0^6) + s8 C8_ij / C6_ij / (r^8 + R0^8))` with
//! `C8 = 3 C6 Q_i Q_j`, `R0 = a1 sqrt(3 Q_i Q_j) + a2`, `Q = sqrt(0.5 <r^4>/<r^2> sqrt(Z))`, and
//! `C6_ij(CN_i, CN_j)` interpolated between reference systems with Gaussian weights
//! (`exp(-4 (CN - CN_ref)^2)`, normalized) of the coordination numbers
//! `CN_i = sum_j 1 / (1 + exp(-16 (R_cov,ij / r_ij - 1)))` (D3 covalent radii, 4/3 of the 2009
//! radii). Cutoffs: 60 Bohr for the dispersion energy, 40 Bohr for the coordination numbers;
//! periodic images are summed in periodic cells.
//!
//! Energy, gradient, and the exact Hessian (including the coupling through the coordination
//! numbers) are computed; the Hessian is assembled from pair terms and the coordination-number
//! Jacobian `J`: `H = H_pair + K J + J^T K^T + J^T M J + sum_k dE/dCN_k d2CN_k`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (genuine DFT-D3(BJ); MultiOptPy's approximation is
//!  in `legacy::fischerd3` etc.).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::{AtomId, Element, EvalOutput, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_grad_hessian, HyperDual, Scalar};
use crate::math::linalg::{mul, mul3};
use crate::math::stable::{exp_shifted, logistic, normalized_exp};
use crate::parameters::d3_reference::{
    D3_C6, D3_C6_OFFSET, D3_MAX_ELEM, D3_NREF, D3_R4_OVER_R2, D3_REF_CN,
};
use crate::parameters::model_hessian::covalent_radius_2009_bohr;
use crate::pes::{PesTerm, TermContext};

/// D3(BJ) damping parameters and cutoffs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct D3Bj {
    /// `s6`.
    pub s6: f64,
    /// `s8`.
    pub s8: f64,
    /// `a1`.
    pub a1: f64,
    /// `a2` (Bohr).
    pub a2: f64,
    /// Dispersion cutoff (Bohr).
    pub cutoff: f64,
    /// Coordination-number cutoff (Bohr).
    pub cn_cutoff: f64,
}

impl D3Bj {
    /// PBE-D3(BJ): `s6 = 1`, `s8 = 0.7875`, `a1 = 0.4289`, `a2 = 4.4407`.
    pub const PBE: Self = Self {
        s6: 1.0,
        s8: 0.7875,
        a1: 0.4289,
        a2: 4.4407,
        cutoff: 60.0,
        cn_cutoff: 40.0,
    };
}

fn z_of(el: Element) -> MoptResult<usize> {
    let z = el.atomic_number() as usize;
    if (1..=D3_MAX_ELEM).contains(&z) {
        Ok(z)
    } else {
        Err(MoptError::unsupported(format!(
            "DFT-D3 has no reference data for element {}",
            el.symbol()
        )))
    }
}

/// `sqrt(0.5 <r^4>/<r^2> sqrt(Z))`.
fn q_value(z: usize) -> f64 {
    (0.5 * D3_R4_OVER_R2[z - 1] * (z as f64).sqrt()).sqrt()
}

/// `C6(CN_a, CN_b)` for elements `za`, `zb`.
fn c6_of<T: Scalar>(za: usize, zb: usize, ca: T, cb: T) -> T {
    let (hi, lo, chi, clo) = if za >= zb {
        (za, zb, ca, cb)
    } else {
        (zb, za, cb, ca)
    };
    let (nh, nl) = (D3_NREF[hi - 1] as usize, D3_NREF[lo - 1] as usize);
    let off = D3_C6_OFFSET[lo - 1 + hi * (hi - 1) / 2] as usize;
    // Gaussian weights exp(-4 (CN_ref - CN)^2), normalized with the largest exponent shifted to
    // zero (simple-dftd3 subtracts the smallest square for the same reason).
    let weights = |z: usize, n: usize, c: T| -> Vec<T> {
        let x: Vec<T> = (0..n)
            .map(|k| {
                let d = T::from_f64(D3_REF_CN[z - 1][k]) - c;
                T::from_f64(-4.0) * d * d
            })
            .collect();
        normalized_exp(&x)
    };
    let (wh, wl) = (weights(hi, nh, chi), weights(lo, nl, clo));
    let mut c6 = T::zero();
    for (a, &x) in wh.iter().enumerate() {
        for (b, &y) in wl.iter().enumerate() {
            c6 += x * y * T::from_f64(D3_C6[off + a * nl + b]);
        }
    }
    c6
}

/// The largest number of D3 references of an element.
const MAX_REF: usize = 7;

/// The Gaussian weights of [`c6_of`] of element `z` at `cn` (`n` references) and their
/// derivatives with respect to `cn`, in closed form without allocations (the weights are the
/// same numbers as [`normalized_exp`]'s).
fn weights_gradient(z: usize, n: usize, cn: f64) -> ([f64; MAX_REF], [f64; MAX_REF]) {
    let mut x = [0.0; MAX_REF];
    let mut dx = [0.0; MAX_REF];
    for k in 0..n {
        let d = D3_REF_CN[z - 1][k] - cn;
        x[k] = -4.0 * d * d;
        dx[k] = 8.0 * d;
    }
    let m = x[..n].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut w = [0.0; MAX_REF];
    let mut dw = [0.0; MAX_REF];
    if !m.is_finite() {
        return (w, dw);
    }
    let mut norm = 0.0;
    for k in 0..n {
        w[k] = exp_shifted(x[k], m);
        norm += w[k];
    }
    let mut mean = 0.0;
    for k in 0..n {
        w[k] /= norm;
        mean += w[k] * dx[k];
    }
    for k in 0..n {
        dw[k] = w[k] * (dx[k] - mean);
    }
    (w, dw)
}

/// `C6(CN_a, CN_b)` of [`c6_of`] with its derivatives `(dC6/dCN_a, dC6/dCN_b)`.
fn c6_gradient(za: usize, zb: usize, ca: f64, cb: f64) -> (f64, [f64; 2]) {
    let swapped = za < zb;
    let (hi, lo, chi, clo) = if swapped {
        (zb, za, cb, ca)
    } else {
        (za, zb, ca, cb)
    };
    let (nh, nl) = (D3_NREF[hi - 1] as usize, D3_NREF[lo - 1] as usize);
    let off = D3_C6_OFFSET[lo - 1 + hi * (hi - 1) / 2] as usize;
    let (wh, dwh) = weights_gradient(hi, nh, chi);
    let (wl, dwl) = weights_gradient(lo, nl, clo);
    let (mut c6, mut dhi, mut dlo) = (0.0, 0.0, 0.0);
    for a in 0..nh {
        for b in 0..nl {
            let r = D3_C6[off + a * nl + b];
            c6 += wh[a] * wl[b] * r;
            dhi += dwh[a] * wl[b] * r;
            dlo += wh[a] * dwl[b] * r;
        }
    }
    if swapped {
        (c6, [dlo, dhi])
    } else {
        (c6, [dhi, dlo])
    }
}

fn norm3(d: [f64; 3]) -> f64 {
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Value, gradient, and Hessian of a function of a 3-vector.
type Local = (f64, [f64; 3], [[f64; 3]; 3]);

fn local3(d: [f64; 3], f: impl Fn([HyperDual; 3]) -> HyperDual) -> Local {
    let (v, g, h) = eval_value_grad_hessian(&d, None, |x| f([x[0], x[1], x[2]]));
    (
        v,
        [g[0], g[1], g[2]],
        std::array::from_fn(|p| std::array::from_fn(|q| h[(p, q)])),
    )
}

/// One pair image: atoms `i > j` (or `i == j` for a self image) and `d = r_i - r_j - T`.
struct Image {
    i: usize,
    j: usize,
    d: [f64; 3],
}

fn images(sys: &MolecularSystem, cutoff: f64) -> Vec<Image> {
    let n = sys.n_atoms();
    let cell = sys.active_cell();
    let pos: Vec<[f64; 3]> = sys
        .atoms()
        .map(|a| {
            let p = sys.position(a).value();
            cell.map_or(p, |c| c.wrap_point(p))
        })
        .collect();
    let shifts = cell.map_or_else(
        || vec![[0.0; 3]],
        |c| c.lattice_shifts_within_cutoff(cutoff),
    );
    let c2 = cutoff * cutoff;
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            for t in &shifts {
                let d: [f64; 3] = std::array::from_fn(|k| pos[i][k] - pos[j][k] - t[k]);
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 > c2 || r2 < 1e-12 {
                    continue;
                }
                out.push(Image { i, j, d });
            }
        }
    }
    out
}

fn add_pair_block(h: &mut Mat<f64>, i: usize, j: usize, b: &[[f64; 3]; 3], w: f64) {
    for (p, row) in b.iter().enumerate() {
        for (q, &v) in row.iter().enumerate() {
            let v = w * v;
            h[(3 * i + p, 3 * i + q)] += v;
            h[(3 * j + p, 3 * j + q)] += v;
            h[(3 * i + p, 3 * j + q)] -= v;
            h[(3 * j + p, 3 * i + q)] -= v;
        }
    }
}

impl D3Bj {
    /// Coordination numbers.
    pub fn coordination_numbers(&self, sys: &MolecularSystem) -> MoptResult<Vec<f64>> {
        Ok(self.cn_terms(sys)?.0)
    }

    fn rcov(sys: &MolecularSystem, a: usize) -> f64 {
        4.0 / 3.0 * covalent_radius_2009_bohr(sys.element(AtomId::new(a)))
    }

    /// Coordination numbers and the per-image counting-function derivatives.
    #[allow(clippy::type_complexity)]
    fn cn_terms(&self, sys: &MolecularSystem) -> MoptResult<(Vec<f64>, Vec<(Image, Local)>)> {
        let n = sys.n_atoms();
        let mut cn = vec![0.0; n];
        let mut terms = Vec::new();
        for im in images(sys, self.cn_cutoff) {
            let rc = Self::rcov(sys, im.i) + Self::rcov(sys, im.j);
            let l = local3(im.d, |x| {
                let r = Scalar::norm3(x);
                logistic(
                    HyperDual::from_f64(16.0) * (HyperDual::from_f64(rc) / r - HyperDual::one()),
                )
            });
            cn[im.i] += l.0;
            if im.i != im.j {
                cn[im.j] += l.0;
            }
            terms.push((im, l));
        }
        Ok((cn, terms))
    }

    /// Energy and gradient with the first derivatives in closed form (the path of every
    /// gradient; the Hessian path differentiates the same functions with hyper-dual numbers).
    /// The coordination-number chain is contracted over the pairs of the counting function
    /// (new; the same numbers up to rounding).
    fn energy_gradient(&self, sys: &MolecularSystem) -> MoptResult<EvalOutput> {
        let n = sys.n_atoms();
        let z: Vec<usize> = sys
            .atoms()
            .map(|a| z_of(sys.element(a)))
            .collect::<MoptResult<_>>()?;
        let q: Vec<f64> = z.iter().map(|&zz| q_value(zz)).collect();
        // Coordination numbers and the derivatives of the counts.
        let mut cn = vec![0.0; n];
        let mut cn_pairs: Vec<(usize, usize, [f64; 3])> = Vec::new();
        for im in images(sys, self.cn_cutoff) {
            let rc = Self::rcov(sys, im.i) + Self::rcov(sys, im.j);
            let r = norm3(im.d);
            let x = 16.0 * (rc / r - 1.0);
            let f = logistic(x);
            cn[im.i] += f;
            if im.i != im.j {
                cn[im.j] += f;
                // d f / d d = f(1 - f) 16 (-rc / r^2) d / r.
                let slope = logistic(x) * logistic(-x) * 16.0 * (-rc / (r * r)) / r;
                cn_pairs.push((im.i, im.j, im.d.map(|v| slope * v)));
            }
        }
        let mut energy = 0.0;
        let mut de_dcn = vec![0.0; n];
        let mut grad = Col::<f64>::zeros(3 * n);
        // The images come pair by pair: the C6 of a pair once for all its images.
        let mut last: Option<(usize, usize, f64, [f64; 2])> = None;
        for im in images(sys, self.cutoff) {
            let (i, j) = (im.i, im.j);
            let (cv, cg) = match last {
                Some((a, b, cv, cg)) if (a, b) == (i, j) => (cv, cg),
                _ => {
                    let (cv, cg) = c6_gradient(z[i], z[j], cn[i], cn[j]);
                    last = Some((i, j, cv, cg));
                    (cv, cg)
                }
            };
            let rr = 3.0 * q[i] * q[j];
            let r0 = self.a1 * rr.sqrt() + self.a2;
            let (r06, r08) = (r0.powi(6), r0.powi(8));
            let (s6, s8) = (self.s6, self.s8 * rr);
            let r2 = im.d[0] * im.d[0] + im.d[1] * im.d[1] + im.d[2] * im.d[2];
            let r6 = r2 * r2 * r2;
            let (t6, t8) = (r6 + r06, r6 * r2 + r08);
            let gv = s6 / t6 + s8 / t8;
            let w = if i == j { 0.5 } else { 1.0 };
            energy -= w * cv * gv;
            if i == j {
                // Self image: geometry-independent, but the C6 depends on CN_i.
                de_dcn[i] -= w * (cg[0] + cg[1]) * gv;
                continue;
            }
            de_dcn[i] -= cg[0] * gv;
            de_dcn[j] -= cg[1] * gv;
            let slope = -6.0 * s6 * r2 * r2 / (t6 * t6) - 8.0 * s8 * r6 / (t8 * t8);
            for c in 0..3 {
                let gc = cv * slope * im.d[c];
                grad[3 * i + c] -= gc;
                grad[3 * j + c] += gc;
            }
        }
        for &(i, j, fg) in &cn_pairs {
            let s = de_dcn[i] + de_dcn[j];
            for c in 0..3 {
                grad[3 * i + c] += fg[c] * s;
                grad[3 * j + c] -= fg[c] * s;
            }
        }
        Ok(EvalOutput {
            energy: Quantity::new(energy),
            gradient_bohr: grad,
            hessian_bohr: None,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }

    /// Energy, gradient, and (when `need_hessian`) Hessian.
    pub fn evaluate(&self, sys: &MolecularSystem, need_hessian: bool) -> MoptResult<EvalOutput> {
        if !need_hessian {
            return self.energy_gradient(sys);
        }
        let n = sys.n_atoms();
        let dof = 3 * n;
        let z: Vec<usize> = sys
            .atoms()
            .map(|a| z_of(sys.element(a)))
            .collect::<MoptResult<_>>()?;
        let q: Vec<f64> = z.iter().map(|&zz| q_value(zz)).collect();
        let (cn, cn_terms) = self.cn_terms(sys)?;
        // C6 and its CN derivatives for every atom pair (j <= i).
        let mut c6 = vec![(0.0, [0.0; 2], [[0.0; 2]; 2]); n * n];
        for i in 0..n {
            for j in 0..=i {
                let (v, g, h) =
                    eval_value_grad_hessian(&[cn[i], cn[j]], None, |x: &[HyperDual]| {
                        c6_of(z[i], z[j], x[0], x[1])
                    });
                c6[i * n + j] = (
                    v,
                    [g[0], g[1]],
                    [[h[(0, 0)], h[(0, 1)]], [h[(1, 0)], h[(1, 1)]]],
                );
            }
        }
        let mut energy = 0.0;
        let mut de_dcn = vec![0.0; n];
        let mut grad = Col::<f64>::zeros(dof);
        let mut hess = need_hessian.then(|| Mat::<f64>::zeros(dof, dof));
        let mut k = need_hessian.then(|| Mat::<f64>::zeros(dof, n));
        let mut m = need_hessian.then(|| Mat::<f64>::zeros(n, n));
        for im in images(sys, self.cutoff) {
            let (i, j) = (im.i, im.j);
            let rr = 3.0 * q[i] * q[j];
            let r0 = self.a1 * rr.sqrt() + self.a2;
            let (r06, r08) = (r0.powi(6), r0.powi(8));
            let (s6, s8) = (self.s6, self.s8 * rr);
            let (gv, gg, gh) = local3(im.d, |x| {
                let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                let r6 = r2 * r2 * r2;
                HyperDual::from_f64(s6) / (r6 + HyperDual::from_f64(r06))
                    + HyperDual::from_f64(s8) / (r6 * r2 + HyperDual::from_f64(r08))
            });
            let (cv, cg, ch) = c6[i * n + j];
            let w = if i == j { 0.5 } else { 1.0 };
            energy -= w * cv * gv;
            if i == j {
                // Self image: geometry-independent, but the C6 depends on CN_i.
                let d1 = cg[0] + cg[1];
                de_dcn[i] -= w * d1 * gv;
                if let Some(m) = m.as_mut() {
                    m[(i, i)] -= w * (ch[0][0] + 2.0 * ch[0][1] + ch[1][1]) * gv;
                }
                continue;
            }
            de_dcn[i] -= cg[0] * gv;
            de_dcn[j] -= cg[1] * gv;
            for c in 0..3 {
                grad[3 * i + c] -= cv * gg[c];
                grad[3 * j + c] += cv * gg[c];
            }
            if let (Some(h), Some(k), Some(m)) = (hess.as_mut(), k.as_mut(), m.as_mut()) {
                add_pair_block(h, i, j, &gh, -cv);
                for c in 0..3 {
                    k[(3 * i + c, i)] -= cg[0] * gg[c];
                    k[(3 * j + c, i)] += cg[0] * gg[c];
                    k[(3 * i + c, j)] -= cg[1] * gg[c];
                    k[(3 * j + c, j)] += cg[1] * gg[c];
                }
                m[(i, i)] -= ch[0][0] * gv;
                m[(j, j)] -= ch[1][1] * gv;
                m[(i, j)] -= ch[0][1] * gv;
                m[(j, i)] -= ch[1][0] * gv;
            }
        }
        // Coordination-number Jacobian and its contributions.
        let mut jac = Mat::<f64>::zeros(n, dof);
        for (im, (_, fg, _)) in &cn_terms {
            if im.i == im.j {
                continue;
            }
            for c in 0..3 {
                for &a in &[im.i, im.j] {
                    jac[(a, 3 * im.i + c)] += fg[c];
                    jac[(a, 3 * im.j + c)] -= fg[c];
                }
            }
        }
        let chain = mul(jac.transpose(), Col::from_fn(n, |a| de_dcn[a]).as_mat());
        for k in 0..dof {
            grad[k] += chain[(k, 0)];
        }
        let hessian = match (hess, k, m) {
            (Some(mut h), Some(k), Some(m)) => {
                let kj = mul(k.as_ref(), jac.as_ref());
                h += &kj + kj.transpose();
                h += mul3(jac.transpose(), m.as_ref(), jac.as_ref());
                for (im, (_, _, fh)) in &cn_terms {
                    if im.i != im.j {
                        add_pair_block(&mut h, im.i, im.j, fh, de_dcn[im.i] + de_dcn[im.j]);
                    }
                }
                crate::math::linalg::symmetrize(&mut h);
                Some(h)
            }
            _ => None,
        };
        Ok(EvalOutput {
            energy: Quantity::new(energy),
            gradient_bohr: grad,
            hessian_bohr: hessian,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }
}

impl PesTerm for D3Bj {
    fn name(&self) -> &str {
        "d3bj"
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        _ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        D3Bj::evaluate(self, sys, need_hessian)
    }

    /// A sum over interatomic distances (minimum images and periodic images in a cell).
    fn frame(&self) -> crate::pes::term::Frame {
        crate::pes::term::Frame::Invariant
    }

    fn history(&self) -> crate::pes::term::History {
        crate::pes::term::History::Static
    }

    /// The Hessian is assembled from exact pair and coordination-number derivatives.
    fn analytic_hessian(&self) -> bool {
        true
    }
}
