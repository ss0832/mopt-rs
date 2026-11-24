//! GFN-FF hydrogen bonds, halogen bonds, bonded three-body (ATM) terms, and torsions, ported
//! from xtb's `gfnff_hbset` (`src/gfnff/gfnff_ini2.F90`) and `abhgfnff_eg1`,
//! `abhgfnff_eg2new`, `abhgfnff_eg2_rnr`, `abhgfnff_eg3`, `egbend_nci_mul`, `egtors_nci_mul`,
//! `rbxgfnff_eg`, `batmgfnff_eg`, `egtors` (`src/gfnff/gfnff_eg.f90`; LGPL-3.0-or-later).
//!
//! The energies are written once, generic over [`Scalar`], and differentiated with forward-mode
//! dual numbers over the few atoms of each term for the gradient, and twice with hyper-dual
//! numbers for the Hessian ([`TermSink`]; xtb differentiates by hand; the energies are the
//! same). Dihedral angles are evaluated with `atan2`, which equals xtb's `acos` of the
//! normalized cross products but stays differentiable at planar geometries. The topology charges
//! `qa` and the HB/XB parameters are constants of the terms, as in xtb.
//!
//! The HB and XB lists are those of the current structure at every evaluation, so that the
//! energy is a function of the structure alone. xtb keeps its lists from a reference
//! structure until the atoms have moved by more than 0.3 Bohr (RMSD) and then builds them
//! anew: its energy is continuous between two rebuilds and steps at a rebuild, and it depends
//! on the structures seen before. A list with a sharp threshold rebuilt at every structure
//! has that step wherever a pair crosses a threshold (about 1e-9 Hartree per term, enough to
//! stall a line search near convergence). [`HbCutoff::Switched`] (the default) takes each
//! term to zero over the last share of its threshold with the infinitely smooth
//! [`switch_off`]; [`HbCutoff::Sharp`] drops it at the threshold, which is xtb's energy of a
//! single point.
//!
//! In a periodic cell every term sees the images of its atoms: the torsions and bonded ATM
//! triples those of the topology, the HB pairs every pair of cells of the HB translations
//! (`sqrt(hbthr2)`) with H in the home cell, the neighbours of B relative to the image of B,
//! and the C of a carbonyl acceptor and its R neighbours at cells summed with xtb's `fTrSum`
//! (a term whose C is out of the set is dropped, an R out of the set left out of the
//! product). The strain derivative of a term is `sum r_img (x) dE/dr_img` over its images (the
//! terms do not change under a common translation).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use super::constants as k;
use super::elements::Atom;
use super::lattice::{is_central, negate, Cell, Frame, Translations, HOME};
use super::nblists::NonBonded;
use super::neighbors::V3;
use super::periodic_eg::Sigma;
use super::topology::Setup;
use super::torsions::Torsion;
use crate::math::ad::{eval_value_and_gradient, Dual, HyperDual, Scalar};
use crate::pes::native::switching::switch_off;

/// Share of a threshold (of the squared distances `hbthr1`, `hbthr2`) over which the HB and
/// XB terms are switched off by default: the last 10 %, 15.0 to 15.8 Bohr for `hbthr1`.
pub const HB_SWITCH_SHARE: f64 = 0.1;

/// How the HB and XB terms end at the thresholds of their lists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HbCutoff {
    /// Taken to zero smoothly over this share of the threshold below it.
    Switched(f64),
    /// Dropped at the threshold (a step in the energy where a pair crosses it).
    Sharp,
}

impl Default for HbCutoff {
    fn default() -> Self {
        Self::Switched(HB_SWITCH_SHARE)
    }
}

impl HbCutoff {
    /// The factor of a term at the value `q` of the quantity its threshold limits.
    fn factor<T: Scalar>(self, q: T, threshold: f64) -> T {
        match self {
            Self::Sharp => T::one(),
            Self::Switched(share) => {
                let share = share.clamp(1.0e-6, 1.0);
                switch_off(q, threshold * (1.0 - share), threshold)
            }
        }
    }

    /// Factor for the squared distance of two atoms against `threshold`.
    fn pair<T: Scalar>(self, a: P<T>, b: P<T>, threshold: f64) -> T {
        self.factor(square(sub(a, b)), threshold)
    }

    /// Factor for the sum of the three squared distances of A, B, and H against `hbthr2`.
    fn triple<T: Scalar>(self, a: P<T>, b: P<T>, h: P<T>) -> T {
        let sum = square(sub(a, b)) + square(sub(h, a)) + square(sub(h, b));
        self.factor(sum, k::HBTHR2)
    }
}

fn square<T: Scalar>(a: P<T>) -> T {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

type P<T> = [T; 3];

fn sub<T: Scalar>(a: P<T>, b: P<T>) -> P<T> {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm<T: Scalar>(a: P<T>) -> T {
    T::norm3(a)
}

fn c<T: Scalar>(v: f64) -> T {
    T::from_f64(v)
}

/// Unsigned dihedral angle `i-j-k-l` (xtb's `valijklff`).
fn dihedral<T: Scalar>(pi: P<T>, pj: P<T>, pk: P<T>, pl: P<T>) -> T {
    let ra = sub(pj, pi);
    let rb = sub(pk, pj);
    let rc = sub(pl, pk);
    let na = T::cross3(ra, rb);
    let nb = T::cross3(rb, rc);
    let y = norm(rb) * T::dot3(ra, nb);
    let x = T::dot3(na, nb);
    y.atan2(x).abs()
}

/// xtb's `omegaPBC`: inversion angle of `l` over the plane `i`-`j`-`k`.
fn omega<T: Scalar>(pi: P<T>, pj: P<T>, pk: P<T>, pl: P<T>) -> T {
    let re = sub(pi, pj);
    let rd = sub(pk, pj);
    let rv = sub(pl, pi);
    let rn = T::cross3(re, rd);
    let (nn, nv) = (norm(rn), norm(rv));
    (T::dot3(rn, rv) / (nn * nv)).asin()
}

/// `gfnffdampt`: torsion distance damping.
fn dampt<T: Scalar>(a: &Atom, b: &Atom, r2: T) -> T {
    let rcut = k::ATCUTT * (a.rcov + b.rcov).powi(2);
    let rr = (r2 / c(rcut)).powi(2);
    T::one() / (T::one() + rr)
}

/// Where the terms of a few atoms go: into a gradient and a strain derivative (forward-mode
/// dual numbers), or into a Hessian (hyper-dual numbers). The energies are written once,
/// generic over [`Scalar`], and each sink evaluates them with its own scalar.
pub trait TermSink {
    /// The scalar the terms are evaluated with.
    type S: Scalar;

    /// Add the derivatives of the term `f` over the images `atoms` of `fr`; its value.
    fn term<F>(&mut self, atoms: &[(usize, Cell)], fr: &Frame<'_>, f: F) -> f64
    where
        F: Fn(&[P<Self::S>]) -> Self::S;
}

/// Gradient (Hartree/Bohr per atom) and strain derivative of the terms.
pub struct GradientSink<'a> {
    /// The gradient the terms are added to.
    pub g: &'a mut [V3],
    /// The strain derivative the terms are added to.
    pub sigma: &'a mut Sigma,
}

impl TermSink for GradientSink<'_> {
    type S = Dual;

    fn term<F>(&mut self, atoms: &[(usize, Cell)], fr: &Frame<'_>, f: F) -> f64
    where
        F: Fn(&[P<Dual>]) -> Dual,
    {
        ad_term(atoms, fr, self.g, self.sigma, f)
    }
}

/// Hessian (Hartree/Bohr^2) of the terms with respect to the positions (`3N x 3N`), or with
/// respect to the positions and the three lattice vectors (`(3N + 9) x (3N + 9)`, the lattice
/// vectors after the atoms): the image of atom `a` in cell `n` is at `x_a + sum_k n_k a_k`.
pub struct HessianSink<'a> {
    /// The Hessian the terms are added to.
    pub h: &'a mut Mat<f64>,
    /// The number of atoms (where the lattice vectors begin).
    pub atoms: usize,
    /// Whether the lattice vectors are variables.
    pub lattice: bool,
}

impl TermSink for HessianSink<'_> {
    type S = HyperDual;

    fn term<F>(&mut self, atoms: &[(usize, Cell)], fr: &Frame<'_>, f: F) -> f64
    where
        F: Fn(&[P<HyperDual>]) -> HyperDual,
    {
        let sites: Vec<Vec<(usize, f64)>> = atoms
            .iter()
            .map(|&(a, n)| {
                let mut c = vec![(a, 1.0)];
                if self.lattice {
                    for (k, &nk) in n.iter().enumerate() {
                        if nk != 0 {
                            c.push((self.atoms + k, f64::from(nk)));
                        }
                    }
                }
                c
            })
            .collect();
        ad_hessian(atoms, fr, self.h, &sites, f)
    }
}

/// Value of a term over the images `atoms` of `fr`, with its Hessian over the `3K`
/// coordinates of the images added into `h`: the image `s` is the combination `sites[s]` of
/// the variables (an atom and, for a cell, the lattice vectors with the coefficients of its
/// cell), each a 3-vector of `h`, so an atom seen at several images gets the sum of their
/// blocks.
pub fn ad_hessian<F>(
    atoms: &[(usize, Cell)],
    fr: &Frame<'_>,
    h: &mut Mat<f64>,
    sites: &[Vec<(usize, f64)>],
    f: F,
) -> f64
where
    F: Fn(&[P<HyperDual>]) -> HyperDual,
{
    let mut p: Vec<P<HyperDual>> = atoms
        .iter()
        .map(|&(a, n)| fr.at(a, n).map(HyperDual::constant))
        .collect();
    let m = 3 * atoms.len();
    if m == 0 {
        return f(&p).f0;
    }
    let mut local = vec![0.0; m * m];
    let mut value = 0.0;
    for u in 0..m {
        p[u / 3][u % 3].f1 = 1.0;
        for v in u..m {
            p[v / 3][v % 3].f2 = 1.0;
            let out = f(&p);
            if u == 0 && v == 0 {
                value = out.f0;
            }
            local[u * m + v] = out.f12;
            local[v * m + u] = out.f12;
            p[v / 3][v % 3].f2 = 0.0;
        }
        p[u / 3][u % 3].f1 = 0.0;
    }
    for (s, cs) in sites.iter().enumerate() {
        for (t, ct) in sites.iter().enumerate() {
            for &(a, wa) in cs {
                for &(b, wb) in ct {
                    let w = wa * wb;
                    for c in 0..3 {
                        for d in 0..3 {
                            h[(3 * a + c, 3 * b + d)] += w * local[(3 * s + c) * m + 3 * t + d];
                        }
                    }
                }
            }
        }
    }
    value
}

/// Value and gradient of a term over the images `atoms` of `fr`, added into `g`, and its
/// strain derivative added into `sigma`.
pub fn ad_term<F>(
    atoms: &[(usize, Cell)],
    fr: &Frame<'_>,
    g: &mut [V3],
    sigma: &mut Sigma,
    f: F,
) -> f64
where
    F: Fn(&[P<Dual>]) -> Dual,
{
    // Terms of up to this many images are differentiated without allocations (the same
    // passes as `eval_value_and_gradient`, one per coordinate).
    const ON_STACK: usize = 8;
    if atoms.len() <= ON_STACK {
        let k = atoms.len();
        let mut positions = [[0.0; 3]; ON_STACK];
        let mut p = [[Dual::constant(0.0); 3]; ON_STACK];
        for (s, &(a, n)) in atoms.iter().enumerate() {
            positions[s] = fr.at(a, n);
            p[s] = positions[s].map(Dual::constant);
        }
        let mut e = 0.0;
        let mut grad = [[0.0; 3]; ON_STACK];
        for s in 0..k {
            for cc in 0..3 {
                p[s][cc].eps = 1.0;
                let out = f(&p[..k]);
                if s == 0 && cc == 0 {
                    e = out.re;
                }
                grad[s][cc] = out.eps;
                p[s][cc].eps = 0.0;
            }
        }
        if k == 0 {
            e = f(&[]).re;
        }
        for (s, &(a, _)) in atoms.iter().enumerate() {
            for cc in 0..3 {
                g[a][cc] += grad[s][cc];
                for b in 0..3 {
                    sigma[b][cc] += positions[s][b] * grad[s][cc];
                }
            }
        }
        return e;
    }
    let positions: Vec<V3> = atoms.iter().map(|&(a, n)| fr.at(a, n)).collect();
    let flat: Vec<f64> = positions.iter().flatten().copied().collect();
    let (e, grad) = eval_value_and_gradient(&flat, |v| {
        let p: Vec<P<Dual>> = v.chunks(3).map(|q| [q[0], q[1], q[2]]).collect();
        f(&p)
    });
    for (s, &(a, _)) in atoms.iter().enumerate() {
        for cc in 0..3 {
            g[a][cc] += grad[3 * s + cc];
            for b in 0..3 {
                sigma[b][cc] += positions[s][b] * grad[3 * s + cc];
            }
        }
    }
    e
}

/// One torsion or out-of-plane term (`egtors`); `p` in the order of `Torsion::atoms`.
fn torsion_energy<T: Scalar>(t: &Torsion, el: &[Atom], p: &[P<T>]) -> T {
    let [i, j, kk, l] = t.atoms;
    let (pi, pj, pk, pl) = (p[0], p[1], p[2], p[3]);
    if t.nrot > 0 {
        let damp = dampt(&el[i], &el[j], T::dot3(sub(pi, pj), sub(pi, pj)))
            * dampt(&el[kk], &el[j], T::dot3(sub(pj, pk), sub(pj, pk)))
            * dampt(&el[kk], &el[l], T::dot3(sub(pk, pl), sub(pk, pl)));
        let phi = dihedral(pi, pj, pk, pl);
        let c1 = c::<T>(f64::from(t.nrot)) * (phi - c(t.phi0)) + c(PI);
        (T::one() + c1.cos()) * c(t.fc) * damp
    } else {
        let vab = sub(pj, pi);
        let vcb = sub(pj, pk);
        let vdc = sub(pj, pl);
        let damp = dampt(&el[i], &el[j], T::dot3(vab, vab))
            * dampt(&el[kk], &el[j], T::dot3(vcb, vcb))
            * dampt(&el[j], &el[l], T::dot3(vdc, vdc));
        let phi = omega(pi, pj, pk, pl);
        let et = if t.nrot == 0 {
            (T::one() + (phi - c(t.phi0) + c(PI)).cos()) * c(t.fc)
        } else {
            c::<T>(t.fc) * (phi.cos() - c(t.phi0.cos())).powi(2)
        };
        et * damp
    }
}

/// Torsion and out-of-plane energy.
pub fn torsions<K: TermSink>(tors: &[Torsion], el: &[Atom], fr: &Frame<'_>, sink: &mut K) -> f64 {
    tors.iter()
        .map(|t| {
            let atoms: Vec<(usize, Cell)> = (0..4).map(|q| (t.atoms[q], t.cells[q])).collect();
            sink.term(&atoms, fr, |p| torsion_energy(t, el, p))
        })
        .sum()
}

/// Bonded ATM energy of `(i, j, k)` (`batmgfnff_eg`).
fn batm_energy<T: Scalar>(qa: [f64; 3], zb: [f64; 3], p: &[P<T>]) -> T {
    const FQQ: f64 = 3.0;
    let f = |q: f64| (1.0 - FQQ * q).clamp(-4.0, 4.0);
    let ff = f(qa[0]) * f(qa[1]) * f(qa[2]);
    let c9 = ff * zb[0] * zb[1] * zb[2];
    let dij = sub(p[0], p[1]);
    let dik = sub(p[0], p[2]);
    let djk = sub(p[1], p[2]);
    let (r2ij, r2ik, r2jk) = (T::dot3(dij, dij), T::dot3(dik, dik), T::dot3(djk, djk));
    let mijk = -r2ij + r2jk + r2ik;
    let imjk = r2ij - r2jk + r2ik;
    let ijmk = r2ij + r2jk - r2ik;
    let rijk3 = r2ij * r2jk * r2ik;
    let rav3 = rijk3 * r2ij.sqrt() * r2jk.sqrt() * r2ik.sqrt();
    let ang = c::<T>(0.375) * ijmk * imjk * mijk / rijk3;
    c::<T>(c9) * (ang + T::one()) / rav3
}

/// Bonded ATM energy (a triple whose third atom has no cell is left out).
pub fn batm<K: TermSink>(s: &Setup, nbl: &NonBonded, fr: &Frame<'_>, sink: &mut K) -> f64 {
    nbl.b3list
        .iter()
        .filter_map(|b| b.ck.map(|ck| (b, ck)))
        .map(|(b, ck)| {
            let (i, j, kk) = (b.i, b.j, b.k);
            let qa = [s.qa[i], s.qa[j], s.qa[kk]];
            let zb = [i, j, kk].map(|a| k::zb3atm(s.el[a].z()));
            sink.term(&[(i, HOME), (j, b.cj), (kk, ck)], fr, |p| {
                batm_energy(qa, zb, p)
            })
        })
        .sum()
}

/// Logistic charge factor `e / (e + sf)` with `e = exp(st q)`.
fn qfac(st: f64, sf: f64, q: f64) -> f64 {
    let e = (st * q).exp();
    e / (e + sf)
}

/// Shared distances and damping of the HB terms.
struct HbCore<T> {
    rab: T,
    rab2: T,
    outl: T,
    damp: T,
}

/// `None` when the out-of-line exponent exceeds 15 (xtb returns a zero energy).
fn hb_core<T: Scalar>(radab: f64, pa: P<T>, pb: P<T>, ph: P<T>, short6: bool) -> Option<HbCore<T>> {
    let rab = norm(sub(pa, pb));
    let rab2 = rab * rab;
    let rah = norm(sub(pa, ph));
    let rbh = norm(sub(pb, ph));
    let rahprbh = rah + rbh + c(1.0e-12);
    let expo = c::<T>(k::HBACUT / radab) * (rahprbh / rab - T::one());
    if expo.val() > 15.0 {
        return None;
    }
    let outl = c::<T>(2.0) / (T::one() + expo.exp());
    let ratio1 = (rab2 / c(k::HBLONGCUT)).powf(k::HBALP);
    let dampl = T::one() / (T::one() + ratio1);
    let shortcut = k::HBSCUT * radab;
    let ratio3 = if short6 {
        (c::<T>(shortcut) / rab2).powi(6)
    } else {
        (c::<T>(shortcut) / rab2).powf(k::HBALP)
    };
    let damps = T::one() / (T::one() + ratio3);
    Some(HbCore {
        rab,
        rab2,
        outl,
        damp: damps * dampl,
    })
}

/// Whether every A-H...B term of A, B, and H is zero here: the out-of-line exponent of
/// [`hb_core`] passes 15, where xtb drops the term. Such a term and its derivatives are left
/// out without evaluating it (the same arithmetic as the term, so the same decision).
fn hb_vanishes(radab: f64, pa: V3, pb: V3, ph: V3) -> bool {
    hb_core::<f64>(radab, pa, pb, ph, false).is_none()
}

/// Mixed `R_BH^-3` / `R_AB^-3` damping of the A-H...B cases 2 and 3.
fn rdamp_mixed<T: Scalar>(core: &HbCore<T>, pb: P<T>, ph: P<T>) -> T {
    let p_bh = 1.0 + k::HBABMIX;
    let p_ab = -k::HBABMIX;
    let rbh = norm(sub(pb, ph));
    let rbh2 = rbh * rbh;
    core.damp * (c::<T>(p_bh) / rbh2 / rbh) + core.damp * (c::<T>(p_ab) / core.rab2 / core.rab)
}

/// Out-of-line factor of `A...nb-B` (`2 / (1 + exp(-expo)) - 1`).
fn outl_nb<T: Scalar>(cut: f64, radab: f64, pa: P<T>, pb: P<T>, pn: P<T>, rab: T) -> T {
    let s = norm(sub(pa, pn)) + norm(sub(pb, pn)) + c(1.0e-12);
    let expo = c::<T>(cut / radab) * (s / rab - T::one());
    c::<T>(2.0) / (T::one() + (-expo).exp()) - T::one()
}

/// HB constants of one triple.
#[derive(Debug, Clone, Copy)]
struct HbConst {
    radab: f64,
    qh: f64,
    qa: f64,
    qb: f64,
    bas: [f64; 2],
    aci: [f64; 2],
}

fn hb_const(s: &Setup, nbl: &NonBonded, a: usize, b: usize, h: usize) -> HbConst {
    HbConst {
        radab: s.el[a].raw.rad + s.el[b].raw.rad,
        qh: qfac(k::HBST, k::HBSF, s.qa[h]),
        qa: qfac(-k::HBST, k::HBSF, s.qa[a]),
        qb: qfac(-k::HBST, k::HBSF, s.qa[b]),
        bas: [nbl.hbbas[a], nbl.hbbas[b]],
        aci: [nbl.hbaci[a], nbl.hbaci[b]],
    }
}

/// Case 1, H bonded to neither A nor B (`abhgfnff_eg1`); `p = [A, B, H]`.
fn hb1<T: Scalar>(q: HbConst, p: &[P<T>]) -> T {
    let (pa, pb, ph) = (p[0], p[1], p[2]);
    let Some(core) = hb_core(q.radab, pa, pb, ph, false) else {
        return T::zero();
    };
    let rdamp = core.damp / core.rab2 / core.rab;
    let rah = norm(sub(pa, ph));
    let rbh = norm(sub(pb, ph));
    let (rah2, rbh2) = (rah * rah, rbh * rbh);
    let (rah4, rbh4) = (rah2 * rah2, rbh2 * rbh2);
    let denom = T::one() / (rah4 + rbh4);
    let caa = q.qa * q.bas[0];
    let cbb = q.qb * q.bas[1];
    let bas = (c::<T>(caa) * rah4 + c::<T>(cbb) * rbh4) * denom;
    let aci = (c::<T>(q.aci[1]) * rah4 + c::<T>(q.aci[0]) * rbh4) * denom;
    bas * (-aci * rdamp * c(q.qh) * core.outl)
}

/// Case 2, A-H...B with the neighbours of B (`abhgfnff_eg2new`); `p = [A, B, H, nb(B)...]`.
fn hb2<T: Scalar>(q: HbConst, nbcut: f64, p: &[P<T>]) -> T {
    let (pa, pb, ph) = (p[0], p[1], p[2]);
    let Some(core) = hb_core(q.radab, pa, pb, ph, false) else {
        return T::zero();
    };
    let mut tot = T::one();
    for &pn in &p[3..] {
        tot = tot * outl_nb(nbcut, q.radab, pa, pb, pn, core.rab);
    }
    let rdamp = rdamp_mixed(&core, pb, ph);
    let cst = q.aci[0] * q.qa * q.bas[1] * q.qb * k::XHACI_GLOBABH;
    -rdamp * (c::<T>(q.qh) * core.outl * tot) * c(cst)
}

/// Case 2 for a two-coordinated N acceptor with its lone pair (`abhgfnff_eg2_rnr`);
/// `p = [A, B, H, nb1(B), nb2(B)]`.
fn hb2_rnr<T: Scalar>(q: HbConst, repz_b: f64, p: &[P<T>]) -> T {
    const HBLPCUT: f64 = 56.0;
    let (pa, pb, ph) = (p[0], p[1], p[2]);
    let lp_dist = 0.50 - 0.018 * repz_b;
    let mut v = [T::zero(); 3];
    for &pn in &p[3..] {
        let d = sub(pn, pb);
        v = [v[0] + d[0], v[1] + d[1], v[2] + d[2]];
    }
    let vnorm = norm(v);
    let (lp, use_nb) = if vnorm.val() > 1.0e-10 {
        let s = c::<T>(lp_dist) / vnorm;
        ([pb[0] - s * v[0], pb[1] - s * v[1], pb[2] - s * v[2]], true)
    } else {
        (pb, false)
    };
    let Some(core) = hb_core(q.radab, pa, pb, ph, false) else {
        return T::zero();
    };
    let rblp = norm(sub(pb, lp));
    let ralp = norm(sub(pa, lp));
    let expo_lp = c::<T>(HBLPCUT / q.radab) * ((ralp + rblp + c(1.0e-12)) / core.rab - T::one());
    let outl_lp = c::<T>(2.0) / (T::one() + expo_lp.exp());
    let mut tot = T::one();
    if use_nb {
        for &pn in &p[3..] {
            tot = tot * outl_nb(k::HBNBCUT, q.radab, pa, pb, pn, core.rab);
        }
    }
    let rdamp = rdamp_mixed(&core, pb, ph);
    let cst = q.aci[0] * q.qa * q.bas[1] * q.qb * k::XHACI_GLOBABH;
    -rdamp * (c::<T>(q.qh) * core.outl * tot * outl_lp) * c(cst)
}

/// Case 3, A-H...O=C or A-H...O=N (`abhgfnff_eg3`); `p = [A, B, H, C, R...]` with the other
/// neighbours `R` of `C`.
fn hb3<T: Scalar>(q: HbConst, p: &[P<T>]) -> T {
    let (pa, pb, ph, pc) = (p[0], p[1], p[2], p[3]);
    let Some(core) = hb_core(q.radab, pa, pb, ph, true) else {
        return T::zero();
    };
    let tot = outl_nb(k::HBNBCUT, q.radab, pa, pb, pc, core.rab);
    let rdamp = rdamp_mixed(&core, pb, ph);
    // Torsions R-B-C-H (product over R).
    let tshift = k::TORS_HB;
    let fc = (1.0 - tshift) / 2.0;
    let mut etors = T::one();
    for &pr in &p[4..] {
        let phi = dihedral(pr, pb, pc, ph);
        let c1 = c::<T>(2.0) * (phi - c(PI / 2.0)) + c(PI);
        etors = etors * ((T::one() + c1.cos()) * c(fc) + c(tshift));
    }
    // Angle C-B...H (egbend_nci_mul, 120 degrees).
    let c0 = 120.0 * PI / 180.0;
    let kijk = (1.0 - k::BEND_HB) / (1.0 - c0.cos()).powi(2);
    let vab = sub(pc, pb);
    let vcb = sub(ph, pb);
    let (al, bl) = (norm(vab), norm(vcb));
    let cosa = if al.val() > 0.0 && bl.val() > 0.0 {
        T::dot3(vab, vcb) / (al * bl)
    } else {
        T::zero()
    };
    let cosa = if cosa.val() > 1.0 {
        T::one()
    } else if cosa.val() < -1.0 {
        -T::one()
    } else {
        cosa
    };
    let eangl = T::one() - c::<T>(kijk) * (cosa - c(c0.cos())).powi(2);
    let cst = q.aci[0] * q.qa * q.bas[1] * q.qb * k::XHACI_COH;
    -rdamp * (c::<T>(q.qh) * core.outl * tot) * eangl * etors * c(cst)
}

/// Halogen bond A-X...B (`rbxgfnff_eg`); `p = [A, B, X]`.
fn xb<T: Scalar>(radab: f64, qb: f64, qx: f64, cx: f64, p: &[P<T>]) -> T {
    let (pa, pb, px) = (p[0], p[1], p[2]);
    let (expo, rbx2, rbx) = xb_exponent(pa, pb, px);
    if expo.val() > 15.0 {
        return T::zero();
    }
    let outl = c::<T>(2.0) / (T::one() + expo.exp());
    let ratio1 = (rbx2 / c(k::HBLONGCUT_XB)).powf(k::HBALP);
    let dampl = T::one() / (T::one() + ratio1);
    let shortcut = k::XBSCUT * radab;
    let ratio3 = (c::<T>(shortcut) / rbx2).powf(k::HBALP);
    let damps = T::one() / (T::one() + ratio3);
    let rdamp = damps * dampl / rbx2 / rbx;
    let cst = qb * cx * qx;
    -rdamp * outl * c(cst)
}

/// The out-of-line exponent of a halogen bond A-X...B, with `R_BX^2` and `R_BX`.
fn xb_exponent<T: Scalar>(pa: P<T>, pb: P<T>, px: P<T>) -> (T, T, T) {
    let drab = sub(pa, pb);
    let drax = sub(pa, px);
    let drbx = sub(pb, px);
    let rab = T::dot3(drab, drab).sqrt();
    let rax = T::dot3(drax, drax).sqrt() + c(1.0e-12);
    let rbx2 = T::dot3(drbx, drbx);
    let rbx = rbx2.sqrt() + c(1.0e-12);
    let expo = c::<T>(k::XBACUT) * ((rax + rbx) / rab - T::one());
    (expo, rbx2, rbx)
}

/// The HB hydrogens (home cell) sorted into cubes, to find those near a point without a loop
/// over all of them (new).
struct HydrogenGrid {
    /// The corner of the grid.
    origin: V3,
    /// The number of cubes along each axis.
    dims: [usize; 3],
    /// Edge of a cube (Bohr): [`Self::EDGE`], or larger where the hydrogens are spread so far
    /// that cubes of that edge would be more than [`Self::CUBES_PER_POINT`] per hydrogen.
    edge: f64,
    /// The start of the entries of each cube in `entries` (one more than the cubes).
    start: Vec<usize>,
    /// `(index into hb_h, position)` cube by cube.
    entries: Vec<(usize, V3)>,
}

impl HydrogenGrid {
    /// Edge of a cube (Bohr).
    const EDGE: f64 = 5.0;
    /// At most this many cubes per hydrogen (and [`Self::SPARE_CUBES`] more), as the cell
    /// lists of `pes::native::neighbors`: hydrogens far apart get larger cubes instead of a
    /// grid of empty ones. The hydrogens found near a point are the same for any edge (the
    /// distance test is exact).
    const CUBES_PER_POINT: usize = 4;
    /// See [`Self::CUBES_PER_POINT`].
    const SPARE_CUBES: usize = 64;

    /// The number of cubes along each axis of the span `lo..hi` with cubes of `edge`.
    fn dims_of(lo: V3, hi: V3, edge: f64) -> [f64; 3] {
        [0, 1, 2].map(|c| ((hi[c] - lo[c]) / edge).floor() + 1.0)
    }

    fn new(fr: &Frame<'_>, hb_h: &[usize]) -> Self {
        let points: Vec<V3> = hb_h.iter().map(|&h| fr.x[h]).collect();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &points {
            for c in 0..3 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
        if points.is_empty() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let cap = (Self::CUBES_PER_POINT * points.len() + Self::SPARE_CUBES) as f64;
        let mut edge = Self::EDGE;
        while Self::dims_of(lo, hi, edge).iter().product::<f64>() > cap {
            edge *= 2.0;
        }
        let dims = Self::dims_of(lo, hi, edge).map(|d| d as usize);
        let cube = |p: &V3| -> usize {
            let idx = [0, 1, 2]
                .map(|c| (((p[c] - lo[c]) / edge).floor().max(0.0) as usize).min(dims[c] - 1));
            (idx[0] * dims[1] + idx[1]) * dims[2] + idx[2]
        };
        let ncubes = dims[0] * dims[1] * dims[2];
        let mut start = vec![0usize; ncubes + 1];
        for p in &points {
            start[cube(p) + 1] += 1;
        }
        for c in 0..ncubes {
            start[c + 1] += start[c];
        }
        let mut fill = start.clone();
        let mut entries = vec![(0, [0.0; 3]); points.len()];
        for (kh, p) in points.iter().enumerate() {
            let c = cube(p);
            entries[fill[c]] = (kh, *p);
            fill[c] += 1;
        }
        Self {
            origin: lo,
            dims,
            edge,
            start,
            entries,
        }
    }

    /// Append to `out` the indices of the hydrogens with `|h - centre|^2 <= reach2`.
    fn within(&self, centre: V3, reach2: f64, out: &mut Vec<usize>) {
        if self.entries.is_empty() || reach2 < 0.0 {
            return;
        }
        let reach = reach2.sqrt();
        let mut range = [(0usize, 0usize); 3];
        for c in 0..3 {
            let lo = ((centre[c] - reach - self.origin[c]) / self.edge).floor();
            let hi = ((centre[c] + reach - self.origin[c]) / self.edge).floor();
            if hi < 0.0 || lo > (self.dims[c] - 1) as f64 {
                return;
            }
            range[c] = (lo.max(0.0) as usize, (hi as usize).min(self.dims[c] - 1));
        }
        for a in range[0].0..=range[0].1 {
            for b in range[1].0..=range[1].1 {
                let row = (a * self.dims[1] + b) * self.dims[2];
                let (from, to) = (
                    self.start[row + range[2].0],
                    self.start[row + range[2].1 + 1],
                );
                for &(kh, p) in &self.entries[from..to] {
                    let d = [0, 1, 2].map(|c| p[c] - centre[c]);
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= reach2 {
                        out.push(kh);
                    }
                }
            }
        }
    }
}

/// HB and XB energies `(E_HB, E_XB)` with the lists of the current structure; `tr` are the
/// translations of the HB lists (the home cell for a molecule), `cut` how the terms end at
/// the thresholds.
pub fn hb_xb<K: TermSink>(
    s: &Setup,
    nbl: &NonBonded,
    fr: &Frame<'_>,
    tr: &Translations,
    cut: HbCutoff,
    sink: &mut K,
) -> (f64, f64) {
    let z = |a: usize| s.el[a].z();
    let d2 = |p: V3, q: V3| (0..3).map(|cc| (p[cc] - q[cc]).powi(2)).sum::<f64>();
    let grid = HydrogenGrid::new(fr, &nbl.hb_h);
    let mut candidates = Vec::new();
    let mut ehb = 0.0;
    for &(i, j) in &nbl.hb_ab {
        for &ci in &tr.cells {
            for &cj in &tr.cells {
                let (pi, pj) = (fr.at(i, ci), fr.at(j, cj));
                let rab = d2(pi, pj);
                if rab > k::HBTHR1 {
                    continue;
                }
                // The image of j relative to the image of i.
                let nonbond = match tr.sum(negate(ci), cj) {
                    Some(d) if is_central(d) => s.bpair.get(j, i, d) != Some(1),
                    _ => true,
                };
                // The H atoms that can form a term, in the order of `hb_h`: those bonded to
                // i or j (the second case), and those near enough for
                // `rab + rih + rjh < hbthr2`, that is (with the midpoint m of the pair)
                // `|h - m|^2 = (rih + rjh) / 2 - rab / 4 < hbthr2 / 2 - 3 rab / 4`.
                candidates.clear();
                candidates.extend_from_slice(&nbl.hb_h_bonded[i]);
                candidates.extend_from_slice(&nbl.hb_h_bonded[j]);
                let mid = [0, 1, 2].map(|cc| 0.5 * (pi[cc] + pj[cc]));
                let reach2 = 0.5 * k::HBTHR2 - 0.75 * rab;
                grid.within(mid, reach2 * (1.0 + 1e-9) + 1e-9, &mut candidates);
                candidates.sort_unstable();
                candidates.dedup();
                for &kh in &candidates {
                    let nh = nbl.hb_h[kh];
                    let ph = fr.x[nh];
                    let (rih, rjh) = (d2(ph, pi), d2(ph, pj));
                    let a_h = |a: usize, ca: Cell| {
                        is_central(ca) && s.bpair.get(a, nh, ca) == Some(1) && nonbond
                    };
                    let case2 = if a_h(i, ci) {
                        Some(((i, ci), (j, cj)))
                    } else if a_h(j, cj) {
                        Some(((j, cj), (i, ci)))
                    } else {
                        None
                    };
                    if let Some((a, b)) = case2 {
                        ehb += hb_case2(s, nbl, fr, tr, cut, sink, a, b, nh);
                    } else if rab + rih + rjh < k::HBTHR2 {
                        let q = hb_const(s, nbl, i, j, nh);
                        if hb_vanishes(q.radab, pi, pj, ph) {
                            continue;
                        }
                        ehb += sink.term(&[(i, ci), (j, cj), (nh, HOME)], fr, |p| {
                            hb1(q, p)
                                * cut.pair(p[0], p[1], k::HBTHR1)
                                * cut.triple(p[0], p[1], p[2])
                        });
                    }
                }
            }
        }
    }
    let mut exb = 0.0;
    for &(a, b, xx, cb, cx) in &nbl.xb_abx {
        if d2(fr.at(b, cb), fr.x[a]) > k::HBTHR2 {
            continue;
        }
        let radab = s.el[a].raw.rad + s.el[b].raw.rad;
        let qb = qfac(-k::XBST, k::XBSF, s.qa[b]);
        let qx = qfac(k::XBST, k::XBSF, s.qa[xx]);
        let cxb = k::xbaci(z(xx));
        if xb_exponent(fr.x[a], fr.at(b, cb), fr.at(xx, cx)).0 > 15.0 {
            continue;
        }
        exb += sink.term(&[(a, HOME), (b, cb), (xx, cx)], fr, |p| {
            xb(radab, qb, qx, cxb, p) * cut.pair(p[0], p[1], k::HBTHR2)
        });
    }
    (ehb, exb)
}

/// The image of a neighbour `(atom, cell relative to its atom)` of the image `(b, cb)`.
fn shifted(nb: (usize, Cell), cb: Cell) -> (usize, Cell) {
    (nb.0, [nb.1[0] + cb[0], nb.1[1] + cb[1], nb.1[2] + cb[2]])
}

/// The case-2 dispatch of `gfnff_eg` for A-H...B with H (home cell) bonded to the image A.
#[allow(clippy::too_many_arguments)]
fn hb_case2<K: TermSink>(
    s: &Setup,
    nbl: &NonBonded,
    fr: &Frame<'_>,
    tr: &Translations,
    cut: HbCutoff,
    sink: &mut K,
    (a, ca): (usize, Cell),
    (b, cb): (usize, Cell),
    h: usize,
) -> f64 {
    let z = |i: usize| s.el[i].z();
    let nb = &s.nb.nb;
    let q = hb_const(s, nbl, a, b, h);
    if hb_vanishes(q.radab, fr.at(a, ca), fr.at(b, cb), fr.x[h]) {
        return 0.0;
    }
    let core = [(a, ca), (b, cb), (h, HOME)];
    // The first two atoms of every term are A and B, whose distance the list limits.
    let end = |p: &[P<K::S>]| cut.pair(p[0], p[1], k::HBTHR1);
    if z(b) == 8 && nb[b].len() == 1 {
        let (cc, c_rel) = nb[b][0];
        if (z(cc) == 6 || z(cc) == 7) && nb[cc].len() > 1 {
            // The cell of C, and of the R neighbours of C, by xtb's sums of translations.
            let Some(c_abs) = tr.sum(c_rel, cb) else {
                return 0.0;
            };
            let mut atoms = core.to_vec();
            atoms.push((cc, c_abs));
            for &(r, r_rel) in nb[cc].iter().filter(|&&(r, _)| r != b) {
                if let Some(r_abs) = tr.sum(r_rel, c_abs) {
                    atoms.push((r, r_abs));
                }
            }
            return sink.term(&atoms, fr, |p| hb3(q, p) * end(p));
        }
    }
    if z(b) == 7 && nb[b].len() == 2 {
        let mut atoms = core.to_vec();
        atoms.push(shifted(nb[b][0], cb));
        atoms.push(shifted(nb[b][1], cb));
        let repz = s.el[b].raw.repz;
        return sink.term(&atoms, fr, |p| hb2_rnr(q, repz, p) * end(p));
    }
    let nbcut = if z(b) == 7 && nb[b].len() == 1 {
        2.0
    } else {
        k::HBNBCUT
    };
    let mut atoms = core.to_vec();
    atoms.extend(nb[b].iter().map(|&n| shifted(n, cb)));
    sink.term(&atoms, fr, |p| hb2(q, nbcut, p) * end(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hydrogens within `reach2` of `centre` by a loop over all of them.
    fn by_loop(points: &[V3], centre: V3, reach2: f64) -> Vec<usize> {
        (0..points.len())
            .filter(|&k| {
                let d = [0, 1, 2].map(|c| points[k][c] - centre[c]);
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= reach2
            })
            .collect()
    }

    #[test]
    fn far_apart_hydrogens_get_larger_cubes_and_the_same_neighbours() {
        // Three clusters of hydrogens 1e5 Bohr apart along every axis: cubes of 5 Bohr would
        // be about 8e12 of them.
        let mut x: Vec<V3> = Vec::new();
        for corner in [0.0, 5.0e4, 1.0e5] {
            for k in 0..6 {
                let t = k as f64;
                x.push([corner + 1.3 * t, corner - 0.7 * t, corner + 0.4 * t * t]);
            }
        }
        let hb_h: Vec<usize> = (0..x.len()).collect();
        let grid = HydrogenGrid::new(&Frame::molecule(&x), &hb_h);
        let cubes = grid.dims.iter().product::<usize>();
        let cap = HydrogenGrid::CUBES_PER_POINT * x.len() + HydrogenGrid::SPARE_CUBES;
        assert!(cubes <= cap, "{cubes} cubes for {} hydrogens", x.len());
        assert_eq!(grid.start.len(), cubes + 1);
        for (centre, reach2) in [(x[2], 30.0), ([5.0e4; 3], 400.0), ([7.0e4; 3], 1.0)] {
            let mut found = Vec::new();
            grid.within(centre, reach2, &mut found);
            found.sort_unstable();
            assert_eq!(found, by_loop(&x, centre, reach2));
        }
        // A compact molecule keeps cubes of 5 Bohr.
        let small: Vec<V3> = (0..8).map(|k| [k as f64, 0.5 * k as f64, 0.0]).collect();
        let grid = HydrogenGrid::new(&Frame::molecule(&small), &(0..8).collect::<Vec<_>>());
        assert_eq!(grid.edge, HydrogenGrid::EDGE);
        assert_eq!(grid.dims, [2, 1, 1]);
    }
}
