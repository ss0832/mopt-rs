//! DFT-D4 dispersion (E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher,
//! C. Bannwarth, S. Grimme, J. Chem. Phys. 2019, 150, 154122), following dftd4 with the
//! EEQ-2019 charge model of multicharge:
//!
//! - covalent coordination numbers with electronegativity weighting,
//!   `CN_i = sum_j k4 exp(-(|EN_i - EN_j| + k5)^2 / k6) (1 + erf(-7.5 (r - R_ij) / R_ij)) / 2`;
//! - EEQ charges from `A q = b`, `A_ij = erf(r sqrt(gamma_ij)) / r`,
//!   `A_ii = eta_i + sqrt(2/pi) / sigma_i`, `b_i = -chi_i + kappa_i sqrt(CN'_i)` (erf CN capped
//!   smoothly at 8) and the total-charge constraint;
//! - reference weights `sum_g exp(-6 g (CN - CN_ref)^2)` normalized and scaled by the charge
//!   function `zeta`; C6 from Casimir-Polder integration of the reference polarizabilities;
//! - two-body rational damping and the Axilrod-Teller-Muto three-body term (zero charges, `alpha
//!   = 16`).
//!
//! Energy and gradient are analytic (the charge response through one adjoint solve); the Hessian
//! comes from central differences of the gradient, as in dftd4. Periodic systems need the Ewald
//! form of the EEQ model and are not supported yet.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (genuine DFT-D4; MultiOptPy's approximation is in
//!  `legacy::*d4`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::{Col, Mat};

use crate::domain::{EvalOutput, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{local_value_grad, Dual, HyperDual, Scalar};
use crate::math::stable::{exp_shifted, logistic, softplus, EXP_MAX};
use crate::parameters::d4_reference::{
    D4_ALPHAIW, D4_ASCALE, D4_HARDNESS, D4_HCOUNT, D4_NREF, D4_PAULING_EN, D4_R4_OVER_R2,
    D4_REFSYS, D4_REF_CN, D4_REF_COVCN, D4_REF_H, D4_REF_Q, D4_SECAIW, D4_SSCALE, D4_ZEFF, EEQ_CHI,
    EEQ_ETA, EEQ_KCNCHI, EEQ_RAD,
};
use crate::parameters::model_hessian::covalent_radius_2009_bohr;
use crate::pes::{PesTerm, TermContext};

/// D4 damping parameters (rational damping plus ATM) and cutoffs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct D4 {
    /// `s6`.
    pub s6: f64,
    /// `s8`.
    pub s8: f64,
    /// `s9` (three-body).
    pub s9: f64,
    /// `a1`.
    pub a1: f64,
    /// `a2` (Bohr).
    pub a2: f64,
    /// Three-body damping exponent.
    pub alp: f64,
}

impl D4 {
    /// PBE-D4 with the three-body term.
    pub const PBE: Self = Self {
        s6: 1.0,
        s8: 0.95948085,
        s9: 1.0,
        a1: 0.38574991,
        a2: 4.80688534,
        alp: 16.0,
    };
}

const GA: f64 = 3.0;
const GC: f64 = 2.0;
const WF: f64 = 6.0;
const CUT_CN: f64 = 30.0;
const CUT_DISP2: f64 = 60.0;
const CUT_DISP3: f64 = 40.0;
const CUT_EEQ_CN: f64 = 25.0;
const EEQ_KCN: f64 = 7.5;
const EEQ_CN_MAX: f64 = 8.0;
const K4: f64 = 4.10451;
const K5: f64 = 19.08857;
const NFREQ: usize = 23;
const FREQ: [f64; NFREQ] = [
    0.000001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5,
    3.0, 4.0, 5.0, 7.5, 10.0,
];

fn trapzd(pol: &[f64; NFREQ]) -> f64 {
    let mut s = 0.0;
    for k in 0..NFREQ {
        let lo = if k > 0 { FREQ[k] - FREQ[k - 1] } else { 0.0 };
        let hi = if k + 1 < NFREQ {
            FREQ[k + 1] - FREQ[k]
        } else {
            0.0
        };
        s += pol[k] * 0.5 * (lo + hi);
    }
    s
}

/// Charge scaling `exp(a (1 - exp(c (1 - q_ref / q))))` (dftd4 `zeta`), with its limits taken
/// explicitly where the literal form overflows or divides by zero: `exp(a)` for `q <= 0` and when
/// the inner exponent underflows, zero when it overflows.
fn zeta<T: Scalar>(a: f64, c: f64, qref: f64, qmod: T) -> T {
    if qmod.val() <= 0.0 {
        return T::from_f64(a.exp());
    }
    let t = T::from_f64(c) * (T::one() - T::from_f64(qref) / qmod);
    if t.val() < -EXP_MAX {
        return T::from_f64(a.exp());
    }
    if t.val() > EXP_MAX {
        return T::zero();
    }
    let outer = T::from_f64(a) * (T::one() - t.exp());
    if outer.val() < -EXP_MAX {
        return T::zero();
    }
    outer.exp()
}

/// Element data of one atom kind.
#[derive(Debug, Clone)]
struct Species {
    /// The atomic number.
    z: usize,
    nref: usize,
    refcn: [f64; 7],
    refq: [f64; 7],
    ngw: [usize; 7],
    alpha: Vec<[f64; NFREQ]>,
    zeff: f64,
    gi: f64,
    q4: f64,
    rcov: f64,
    en: f64,
    chi: f64,
    eta: f64,
    kcn: f64,
    rad: f64,
}

impl Species {
    fn new(z: usize) -> MoptResult<Self> {
        if z == 0 || z > 103 {
            return Err(MoptError::unsupported(format!(
                "DFT-D4 with EEQ charges supports Z = 1..=103, got {z}"
            )));
        }
        let i = z - 1;
        let nref = D4_NREF[i] as usize;
        // Number of Gaussian weights per reference (dftd4 `set_refgw`).
        let mut cnc = [0usize; 20];
        cnc[0] = 1;
        for r in 0..nref {
            cnc[(D4_REF_CN[i][r].round() as usize).min(19)] += 1;
        }
        let mut ngw = [1usize; 7];
        for (r, n) in ngw.iter_mut().enumerate().take(nref) {
            let c = cnc[(D4_REF_CN[i][r].round() as usize).min(19)];
            *n = c * (c + 1) / 2;
        }
        let alpha = (0..nref)
            .map(|r| {
                let mut a = [0.0; NFREQ];
                let is = D4_REFSYS[i][r] as usize;
                if is == 0 {
                    return a;
                }
                let iz = D4_ZEFF[is - 1];
                let sec = zeta(GA, D4_HARDNESS[is - 1] * GC, iz, D4_REF_H[i][r] + iz);
                for (k, v) in a.iter_mut().enumerate() {
                    let aiw = D4_SSCALE[is - 1] * D4_SECAIW[is - 1][k] * sec;
                    *v = (D4_ASCALE[i][r] * (D4_ALPHAIW[7 * i + r][k] - D4_HCOUNT[i][r] * aiw))
                        .max(0.0);
                }
                a
            })
            .collect();
        let el = crate::domain::Element::from_atomic_number(z as u8)?;
        Ok(Self {
            z,
            nref,
            refcn: D4_REF_COVCN[i],
            refq: D4_REF_Q[i],
            ngw,
            alpha,
            zeff: D4_ZEFF[i],
            gi: D4_HARDNESS[i] * GC,
            q4: (0.5 * D4_R4_OVER_R2[i] * (z as f64).sqrt()).sqrt(),
            rcov: 4.0 / 3.0 * covalent_radius_2009_bohr(el),
            en: D4_PAULING_EN[i],
            chi: EEQ_CHI[i],
            eta: EEQ_ETA[i],
            kcn: EEQ_KCNCHI[i],
            rad: EEQ_RAD[i],
        })
    }

    /// Reference weights at `cn`, `q` (generic for derivatives). The Gaussian exponents are
    /// shifted by their maximum before exponentiation, so the normalization cannot underflow to
    /// zero (dftd4 instead falls back to the reference with the largest CN when it does).
    fn weights<T: Scalar>(&self, cn: T, q: T) -> Vec<T> {
        let expo = |r: usize, g: usize| {
            let d = cn - T::from_f64(self.refcn[r]);
            T::from_f64(-WF * g as f64) * d * d
        };
        let shift = (0..self.nref)
            .flat_map(|r| (1..=self.ngw[r]).map(move |g| (r, g)))
            .map(|(r, g)| expo(r, g).val())
            .fold(f64::NEG_INFINITY, f64::max);
        let gw = |r: usize, g: usize| exp_shifted(expo(r, g), shift);
        let mut norm = T::zero();
        for r in 0..self.nref {
            for g in 1..=self.ngw[r] {
                norm += gw(r, g);
            }
        }
        let maxcn = self.refcn[..self.nref]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (0..self.nref)
            .map(|r| {
                let mut expw = T::zero();
                for g in 1..=self.ngw[r] {
                    expw += gw(r, g);
                }
                let mut w = expw / norm;
                if !w.val().is_finite() || norm.val() <= f64::MIN_POSITIVE.sqrt() {
                    w = T::from_f64(if (maxcn - self.refcn[r]).abs() < 1e-12 {
                        1.0
                    } else {
                        0.0
                    });
                }
                w * zeta(
                    GA,
                    self.gi,
                    self.refq[r] + self.zeff,
                    q + T::from_f64(self.zeff),
                )
            })
            .collect()
    }
}

/// The reference C6 coefficients `3 / pi int alpha_a(iw) alpha_b(iw) dw` (trapezoids over the
/// frequencies) of the references of every pair of elements of a set of atoms, computed once
/// for each pair of elements present instead of for each pair of atoms (new; the same
/// numbers).
#[derive(Debug, Clone)]
struct ReferenceC6 {
    /// The element slot of every atom.
    kind: Vec<usize>,
    /// The number of elements present.
    kinds: usize,
    /// The number of references of each slot.
    nref: Vec<usize>,
    /// For each pair of slots `(a, b)` (`a * kinds + b`), the C6 of references
    /// `ra * nref_b + rb`.
    table: Vec<Vec<f64>>,
}

impl ReferenceC6 {
    fn new(sp: &[Species]) -> Self {
        let mut slot_of_z: Vec<(usize, usize)> = Vec::new();
        let mut first: Vec<usize> = Vec::new();
        let kind: Vec<usize> = sp
            .iter()
            .enumerate()
            .map(|(i, s)| match slot_of_z.iter().find(|(z, _)| *z == s.z) {
                Some(&(_, k)) => k,
                None => {
                    let k = first.len();
                    slot_of_z.push((s.z, k));
                    first.push(i);
                    k
                }
            })
            .collect();
        let kinds = first.len();
        let nref: Vec<usize> = first.iter().map(|&i| sp[i].nref).collect();
        let mut table = Vec::with_capacity(kinds * kinds);
        for &i in &first {
            for &j in &first {
                let mut block = Vec::with_capacity(sp[i].nref * sp[j].nref);
                for a in 0..sp[i].nref {
                    for b in 0..sp[j].nref {
                        let mut prod = [0.0; NFREQ];
                        for (k, p) in prod.iter_mut().enumerate() {
                            *p = sp[i].alpha[a][k] * sp[j].alpha[b][k];
                        }
                        block.push(3.0 / PI * trapzd(&prod));
                    }
                }
                table.push(block);
            }
        }
        Self {
            kind,
            kinds,
            nref,
            table,
        }
    }

    /// The C6 of reference `a` of atom `i` and reference `b` of atom `j`.
    fn get(&self, i: usize, j: usize, a: usize, b: usize) -> f64 {
        let (ki, kj) = (self.kind[i], self.kind[j]);
        self.table[ki * self.kinds + kj][a * self.nref[kj] + b]
    }
}

/// Weights and their derivatives with respect to CN and q.
struct Weighted {
    w: Vec<f64>,
    dcn: Vec<f64>,
    dq: Vec<f64>,
}

fn weighted(s: &Species, cn: f64, q: f64) -> Weighted {
    let a = s.weights(Dual { re: cn, eps: 1.0 }, Dual::from_f64(q));
    let b = s.weights(Dual::from_f64(cn), Dual { re: q, eps: 1.0 });
    Weighted {
        w: a.iter().map(|x| x.val()).collect(),
        dcn: a.iter().map(|x| x.eps).collect(),
        dq: b.iter().map(|x| x.eps).collect(),
    }
}

/// Weights with their first and second derivatives in `(CN, q)`.
struct WeightedSecond {
    w: Vec<f64>,
    d: Vec<[f64; 2]>,
    dd: Vec<[[f64; 2]; 2]>,
}

fn weighted_second(s: &Species, cn: f64, q: f64) -> WeightedSecond {
    let v = [cn, q];
    let seed = |k: usize, a: usize, b: usize| HyperDual {
        f0: v[k],
        f1: if k == a { 1.0 } else { 0.0 },
        f2: if k == b { 1.0 } else { 0.0 },
        f12: 0.0,
    };
    let run = |a: usize, b: usize| s.weights(seed(0, a, b), seed(1, a, b));
    let (cc, cq, qq) = (run(0, 0), run(0, 1), run(1, 1));
    WeightedSecond {
        w: cc.iter().map(|x| x.f0).collect(),
        d: cc.iter().zip(&qq).map(|(a, b)| [a.f1, b.f1]).collect(),
        dd: (0..cc.len())
            .map(|r| [[cc[r].f12, cq[r].f12], [cq[r].f12, qq[r].f12]])
            .collect(),
    }
}

/// `C6_ij` of a pair with its derivatives in `(CN_i, CN_j, q_i, q_j)`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct C6Second {
    /// `C6`.
    pub c6: f64,
    /// First derivatives.
    pub d: [f64; 4],
    /// Second derivatives.
    pub dd: [[f64; 4]; 4],
}

/// The D4 reference model of a set of atoms, for methods that bring their own coordination
/// numbers and charges (GFN0-xTB's dispersion term).
#[derive(Debug, Clone)]
pub(crate) struct D4Model {
    sp: Vec<Species>,
    refc6: ReferenceC6,
}

impl D4Model {
    /// The model of atoms with atomic numbers `z`.
    pub(crate) fn new(z: &[u8]) -> MoptResult<Self> {
        let sp: Vec<Species> = z
            .iter()
            .map(|&z| Species::new(z as usize))
            .collect::<MoptResult<_>>()?;
        let refc6 = ReferenceC6::new(&sp);
        Ok(Self { sp, refc6 })
    }

    /// `sqrt(0.5 <r^4>/<r^2> sqrt(Z))` of atom `i` (`C8 = 3 q4_i q4_j C6`).
    pub(crate) fn q4(&self, i: usize) -> f64 {
        self.sp[i].q4
    }

    /// `C6_ij` and its derivatives `(C6, dC6/dCN_i, dC6/dCN_j, dC6/dq_i, dC6/dq_j)` for all
    /// ordered pairs (`i * n + j`), from covalent coordination numbers `cn` and charges `q`.
    pub(crate) fn c6(&self, cn: &[f64], q: &[f64]) -> Vec<[f64; 5]> {
        let sp = &self.sp;
        let n = sp.len();
        let w: Vec<Weighted> = (0..n).map(|i| weighted(&sp[i], cn[i], q[i])).collect();
        let mut out = vec![[0.0; 5]; n * n];
        for i in 0..n {
            for j in 0..n {
                let o = &mut out[i * n + j];
                for a in 0..sp[i].nref {
                    for b in 0..sp[j].nref {
                        let r = self.refc6.get(i, j, a, b);
                        o[0] += w[i].w[a] * w[j].w[b] * r;
                        o[1] += w[i].dcn[a] * w[j].w[b] * r;
                        o[2] += w[i].w[a] * w[j].dcn[b] * r;
                        o[3] += w[i].dq[a] * w[j].w[b] * r;
                        o[4] += w[i].w[a] * w[j].dq[b] * r;
                    }
                }
            }
        }
        out
    }

    /// [`Self::c6`] with the second derivatives, for the pairs `i < j` (`i * n + j`; the other
    /// entries are zero), from covalent coordination numbers `cn` and charges `q`.
    pub(crate) fn c6_second(&self, cn: &[f64], q: &[f64]) -> Vec<C6Second> {
        let sp = &self.sp;
        let n = sp.len();
        let w: Vec<WeightedSecond> = (0..n)
            .map(|i| weighted_second(&sp[i], cn[i], q[i]))
            .collect();
        // The variables of atom i (CN_i, q_i) and of atom j (CN_j, q_j) in (CN_i, CN_j, q_i, q_j).
        const OF_I: [usize; 2] = [0, 2];
        const OF_J: [usize; 2] = [1, 3];
        let mut out = vec![C6Second::default(); n * n];
        for i in 0..n {
            for j in i + 1..n {
                let o = &mut out[i * n + j];
                let (wi, wj) = (&w[i], &w[j]);
                for a in 0..sp[i].nref {
                    for b in 0..sp[j].nref {
                        let r = self.refc6.get(i, j, a, b);
                        o.c6 += wi.w[a] * wj.w[b] * r;
                        for u in 0..2 {
                            o.d[OF_I[u]] += wi.d[a][u] * wj.w[b] * r;
                            o.d[OF_J[u]] += wi.w[a] * wj.d[b][u] * r;
                            for v in 0..2 {
                                o.dd[OF_I[u]][OF_I[v]] += wi.dd[a][u][v] * wj.w[b] * r;
                                o.dd[OF_J[u]][OF_J[v]] += wi.w[a] * wj.dd[b][u][v] * r;
                                let cross = wi.d[a][u] * wj.d[b][v] * r;
                                o.dd[OF_I[u]][OF_J[v]] += cross;
                                o.dd[OF_J[v]][OF_I[u]] += cross;
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

impl D4 {
    fn check_isolated(sys: &MolecularSystem) -> MoptResult<()> {
        if sys.has_pbc() {
            return Err(MoptError::unsupported(
                "DFT-D4 in periodic cells needs the Ewald EEQ model, which is not available yet",
            ));
        }
        Ok(())
    }

    /// EEQ-2019 charges.
    pub fn charges(&self, sys: &MolecularSystem) -> MoptResult<Vec<f64>> {
        Self::check_isolated(sys)?;
        let sp = species(sys)?;
        Ok(eeq(sys, &sp)?.q)
    }

    /// Energy and gradient.
    pub fn energy_gradient(&self, sys: &MolecularSystem) -> MoptResult<(f64, Col<f64>)> {
        Self::check_isolated(sys)?;
        let sp = species(sys)?;
        let n = sys.n_atoms();
        let pos: Vec<[f64; 3]> = sys.atoms().map(|a| sys.position(a).value()).collect();
        let mut grad = Col::<f64>::zeros(3 * n);
        let add = |g: &mut Col<f64>, a: usize, v: [f64; 3], s: f64| {
            for c in 0..3 {
                g[3 * a + c] += s * v[c];
            }
        };
        // Covalent CN.
        let mut cn = vec![0.0; n];
        let mut cn_pairs = Vec::new();
        for i in 0..n {
            for j in 0..i {
                let d = sub(pos[i], pos[j]);
                let r = norm(d);
                if r > CUT_CN {
                    continue;
                }
                let rc = sp[i].rcov + sp[j].rcov;
                let den = K4
                    * (-((sp[i].en - sp[j].en).abs() + K5).powi(2) / (2.0 * 11.28174_f64.powi(2)))
                        .exp();
                let x = -7.5 * (r - rc) / rc;
                let f = den * 0.5 * (1.0 + libm::erf(x));
                let df = den * (-7.5 / rc) * (-x * x).exp() / PI.sqrt();
                cn[i] += f;
                cn[j] += f;
                cn_pairs.push((i, j, d.map(|v| v / r), df));
            }
        }
        let e = eeq(sys, &sp)?;
        let w: Vec<Weighted> = (0..n).map(|i| weighted(&sp[i], cn[i], e.q[i])).collect();
        let w0: Vec<Weighted> = (0..n).map(|i| weighted(&sp[i], cn[i], 0.0)).collect();
        let refc6 = ReferenceC6::new(&sp);
        let c6ref = |i: usize, j: usize, a: usize, b: usize| refc6.get(i, j, a, b);
        // C6 with derivatives: (c6, dc6/dcn_i, dc6/dcn_j, dc6/dq_i, dc6/dq_j).
        let c6_of = |ws: &[Weighted], i: usize, j: usize| {
            let mut out = [0.0; 5];
            for a in 0..sp[i].nref {
                for b in 0..sp[j].nref {
                    let r = c6ref(i, j, a, b);
                    out[0] += ws[i].w[a] * ws[j].w[b] * r;
                    out[1] += ws[i].dcn[a] * ws[j].w[b] * r;
                    out[2] += ws[i].w[a] * ws[j].dcn[b] * r;
                    out[3] += ws[i].dq[a] * ws[j].w[b] * r;
                    out[4] += ws[i].w[a] * ws[j].dq[b] * r;
                }
            }
            out
        };
        let mut c6 = vec![[0.0; 5]; n * n];
        let mut c60 = vec![[0.0; 5]; n * n];
        for i in 0..n {
            for j in 0..n {
                c6[i * n + j] = c6_of(&w, i, j);
                c60[i * n + j] = c6_of(&w0, i, j);
            }
        }
        let mut energy = 0.0;
        let mut de_dcn = vec![0.0; n];
        let mut de_dq = vec![0.0; n];
        // Two-body.
        for i in 0..n {
            for j in 0..i {
                let d = sub(pos[i], pos[j]);
                let r = norm(d);
                if r > CUT_DISP2 {
                    continue;
                }
                let rr = 3.0 * sp[i].q4 * sp[j].q4;
                let r0 = self.a1 * rr.sqrt() + self.a2;
                let (r2, r06, r08) = (r * r, r0.powi(6), r0.powi(8));
                let (t6, t8) = (1.0 / (r2.powi(3) + r06), 1.0 / (r2.powi(4) + r08));
                let g = self.s6 * t6 + self.s8 * rr * t8;
                let dg =
                    -6.0 * self.s6 * r.powi(5) * t6 * t6 - 8.0 * self.s8 * rr * r.powi(7) * t8 * t8;
                let c = c6[i * n + j];
                energy -= c[0] * g;
                de_dcn[i] -= c[1] * g;
                de_dcn[j] -= c[2] * g;
                de_dq[i] -= c[3] * g;
                de_dq[j] -= c[4] * g;
                let u = d.map(|v| v / r);
                add(&mut grad, i, u, -c[0] * dg);
                add(&mut grad, j, u, c[0] * dg);
            }
        }
        // Three-body (zero charges).
        if self.s9 != 0.0 {
            let c2 = CUT_DISP3 * CUT_DISP3;
            for i in 0..n {
                for j in 0..i {
                    for k in 0..j {
                        let (vij, vik, vjk) = (
                            sub(pos[j], pos[i]),
                            sub(pos[k], pos[i]),
                            sub(pos[k], pos[j]),
                        );
                        let r2 = [vij, vik, vjk].map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                        if r2.iter().any(|&x| x > c2) {
                            continue;
                        }
                        let (cij, cik, cjk) = (c60[i * n + j], c60[i * n + k], c60[j * n + k]);
                        let prod = cij[0] * cik[0] * cjk[0];
                        let c9 = -self.s9 * prod.abs().sqrt();
                        let r0 = |a: usize, b: usize| {
                            self.a1 * (3.0 * sp[a].q4 * sp[b].q4).sqrt() + self.a2
                        };
                        let r0p = r0(i, j) * r0(i, k) * r0(j, k);
                        let alp3 = self.alp / 3.0;
                        let f = |x: &[[Dual; 3]; 3]| {
                            let d2 = |a: [Dual; 3], b: [Dual; 3]| {
                                let v = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                                v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
                            };
                            let (rij2, rik2, rjk2) =
                                (d2(x[0], x[1]), d2(x[0], x[2]), d2(x[1], x[2]));
                            let rr2 = rij2 * rik2 * rjk2;
                            let r1 = rr2.sqrt();
                            let r3 = rr2 * r1;
                            let r5 = r3 * rr2;
                            let fd = Dual::one()
                                / (Dual::one()
                                    + Dual::from_f64(6.0) * (Dual::from_f64(r0p) / r1).powf(alp3));
                            let ang = Dual::from_f64(0.375)
                                * (rij2 + rjk2 - rik2)
                                * (rij2 - rjk2 + rik2)
                                * (rjk2 + rik2 - rij2)
                                / r5
                                + Dual::one() / r3;
                            ang * fd
                        };
                        let l = local_value_grad(&[pos[i], pos[j], pos[k]], f);
                        // E3 = -c9 * ang * fdmp.
                        energy -= c9 * l.value;
                        for (s, &a) in [i, j, k].iter().enumerate() {
                            add(&mut grad, a, l.gradient[s], -c9);
                        }
                        if prod.abs() > 0.0 {
                            // c9 = -s9 sqrt|P|, P = C6_ij C6_ik C6_jk; E3 = -c9 L. The derivative
                            // d sqrt|P| / d C6_ab = sign(C6_ab) sqrt|P| / (2 |C6_ab|) does not divide
                            // by sqrt|P|.
                            let dc9 = |cab: f64| {
                                if cab == 0.0 {
                                    0.0
                                } else {
                                    -self.s9 * 0.5 * cab.signum() * (prod.abs().sqrt() / cab.abs())
                                }
                            };
                            let (dij, dik, djk) = (dc9(cij[0]), dc9(cik[0]), dc9(cjk[0]));
                            let l3 = -l.value;
                            de_dcn[i] += l3 * (dij * cij[1] + dik * cik[1]);
                            de_dcn[j] += l3 * (dij * cij[2] + djk * cjk[1]);
                            de_dcn[k] += l3 * (dik * cik[2] + djk * cjk[2]);
                        }
                    }
                }
            }
        }
        // Coordination-number chain.
        for &(i, j, u, df) in &cn_pairs {
            let s = (de_dcn[i] + de_dcn[j]) * df;
            add(&mut grad, i, u, s);
            add(&mut grad, j, u, -s);
        }
        // Charge response: lambda = A^-1 dE/dq; dE = lambda . (db - dA q).
        let mut rhs = Col::<f64>::zeros(n + 1);
        for i in 0..n {
            rhs[i] = de_dq[i];
        }
        let lambda = e.solve(&rhs)?;
        for &(i, j, u, dcn_dr, dcnp_i, dcnp_j) in &e.cn_pairs {
            // b_i depends on CN'_i.
            let s = lambda[i] * e.db_dcnp[i] * dcnp_i * dcn_dr
                + lambda[j] * e.db_dcnp[j] * dcnp_j * dcn_dr;
            add(&mut grad, i, u, s);
            add(&mut grad, j, u, -s);
        }
        for i in 0..n {
            for j in 0..i {
                let d = sub(pos[i], pos[j]);
                let r = norm(d);
                let gam = 1.0 / (sp[i].rad.powi(2) + sp[j].rad.powi(2));
                let sg = gam.sqrt();
                let da =
                    (2.0 * sg / PI.sqrt() * (-gam * r * r).exp() * r - libm::erf(sg * r)) / (r * r);
                let s = -(lambda[i] * e.q[j] + lambda[j] * e.q[i]) * da;
                let u = d.map(|v| v / r);
                add(&mut grad, i, u, s);
                add(&mut grad, j, u, -s);
            }
        }
        Ok((energy, grad))
    }

    /// Energy, gradient, and (when `need_hessian`) Hessian from central differences of the
    /// gradient (step 1e-4 Bohr).
    pub fn evaluate(&self, sys: &MolecularSystem, need_hessian: bool) -> MoptResult<EvalOutput> {
        let (energy, gradient) = self.energy_gradient(sys)?;
        let hessian = if need_hessian {
            // Central differences (dftd4 has no analytic Hessian either); the displaced
            // structures in parallel.
            Some(crate::pes::native::fd_hessian::structure_hessian(
                sys,
                1e-4,
                true,
                |at| Ok(self.energy_gradient(at)?.1),
            )?)
        } else {
            None
        };
        Ok(EvalOutput {
            energy: Quantity::new(energy),
            gradient_bohr: gradient,
            hessian_bohr: hessian,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }
}

fn species(sys: &MolecularSystem) -> MoptResult<Vec<Species>> {
    sys.atoms()
        .map(|a| Species::new(sys.element(a).atomic_number() as usize))
        .collect()
}

/// Solved EEQ model with what the gradient needs.
struct Eeq {
    q: Vec<f64>,
    /// The factorised EEQ matrix (once per structure: the charges and the response solve
    /// with it).
    lu: crate::math::linalg::DenseLu,
    /// `d b_i / d CN'_i`.
    db_dcnp: Vec<f64>,
    /// `(i, j, unit vector, d count / dr, dCN'_i/dCN_i, dCN'_j/dCN_j)`.
    cn_pairs: Vec<(usize, usize, [f64; 3], f64, f64, f64)>,
}

impl Eeq {
    fn solve(&self, rhs: &Col<f64>) -> MoptResult<Col<f64>> {
        let out = self.lu.solve_col(rhs);
        if out.iter().all(|v| v.is_finite()) {
            Ok(out)
        } else {
            Err(MoptError::linalg("EEQ linear system is singular"))
        }
    }
}

fn eeq(sys: &MolecularSystem, sp: &[Species]) -> MoptResult<Eeq> {
    let n = sys.n_atoms();
    let pos: Vec<[f64; 3]> = sys.atoms().map(|a| sys.position(a).value()).collect();
    let mut cn = vec![0.0; n];
    let mut raw = Vec::new();
    for i in 0..n {
        for j in 0..i {
            let d = sub(pos[i], pos[j]);
            let r = norm(d);
            if r > CUT_EEQ_CN {
                continue;
            }
            let rc = sp[i].rcov + sp[j].rcov;
            let x = -EEQ_KCN * (r - rc) / rc;
            cn[i] += 0.5 * (1.0 + libm::erf(x));
            cn[j] += 0.5 * (1.0 + libm::erf(x));
            let df = 0.5 * 2.0 / PI.sqrt() * (-x * x).exp() * (-EEQ_KCN / rc);
            raw.push((i, j, d.map(|v| v / r), df));
        }
    }
    // Smooth cap: CN' = ln(1 + e^cmax) - ln(1 + e^(cmax - CN)).
    let cmax = EEQ_CN_MAX;
    let cnp: Vec<f64> = cn
        .iter()
        .map(|&c| softplus(cmax) - softplus(cmax - c))
        .collect();
    let dcnp: Vec<f64> = cn.iter().map(|&c| logistic(cmax - c)).collect();
    let reg = 1e-14;
    let mut a = Mat::<f64>::zeros(n + 1, n + 1);
    let mut b = Col::<f64>::zeros(n + 1);
    let mut db = vec![0.0; n];
    for i in 0..n {
        a[(i, i)] = sp[i].eta + (2.0 / PI).sqrt() / sp[i].rad;
        b[i] = -sp[i].chi + sp[i].kcn * cnp[i] / (cnp[i] + reg).sqrt();
        db[i] = sp[i].kcn * (0.5 * cnp[i] + reg) / (cnp[i] + reg).powf(1.5);
        for j in 0..i {
            let r = norm(sub(pos[i], pos[j]));
            let gam = 1.0 / (sp[i].rad.powi(2) + sp[j].rad.powi(2));
            let v = libm::erf((r * r * gam).sqrt()) / r;
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
        a[(n, i)] = 1.0;
        a[(i, n)] = 1.0;
    }
    b[n] = f64::from(sys.spin_charge.charge);
    let mut e = Eeq {
        q: Vec::new(),
        lu: crate::math::linalg::DenseLu::new(a.as_ref())?,
        db_dcnp: db,
        cn_pairs: raw
            .into_iter()
            .map(|(i, j, u, df)| (i, j, u, df, dcnp[i], dcnp[j]))
            .collect(),
    };
    let q = e.solve(&b)?;
    e.q = (0..n).map(|i| q[i]).collect();
    Ok(e)
}

impl PesTerm for D4 {
    fn name(&self) -> &str {
        "d4"
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        _ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        D4::evaluate(self, sys, need_hessian)
    }

    /// Charges, coordination numbers, and dispersion are functions of interatomic distances;
    /// the Hessian (central differences of the gradient) is not analytic.
    fn frame(&self) -> crate::pes::term::Frame {
        crate::pes::term::Frame::Invariant
    }

    fn history(&self) -> crate::pes::term::History {
        crate::pes::term::History::Static
    }
}
