//! Analytic Hessian of GFN-FF, for a molecule and for a periodic cell, the derivative of the
//! analytic gradient of [`super::eg`] term by term: with respect to the positions, and in a
//! cell optionally also with respect to the three lattice vectors (the second derivatives of
//! the strain).
//!
//! - Pair terms (non-bonded and bonded repulsion, the dispersion at fixed C6, the EEQ matrix
//!   at fixed charges, the radial part of the bond stretches): `f(r)` with `f'' u u^T +
//!   (f' / r) (1 - u u^T)` in the blocks of the two atoms (and of the lattice vectors, with
//!   the cells of the images: [`super::hessian_core`]).
//! - Bends, torsions and out-of-plane terms, bonded ATM, hydrogen and halogen bonds: the
//!   energies of the gradient, generic over [`Scalar`], differentiated twice with hyper-dual
//!   numbers over the images of each term ([`super::eg_hb::HessianSink`]). The bends are
//!   written with `cos(theta)`; the linear ones (`theta_0 = pi`) with the power series of
//!   `acos(-cos theta)^2` in `1 + cos(theta)`, which is smooth through the linear geometry
//!   where `theta` itself is not.
//! - The coordination numbers: the D4 weights of the dispersion, the bond lengths, and the
//!   electronegativities of the EEQ depend on the log-capped CN. Every such term gives
//!   `dE/dCN` (`w`), `d2E/dCN dCN` (`M`), and the derivative of `dE/dCN` at fixed CN (`V`);
//!   with the CN Jacobian `J` and the second derivatives of the CN,
//!   `H += sum_i w_i d2CN_i + J^T M J + J^T V + V^T J` ([`Chain`]). `J` is sparse (the erf
//!   count underflows a few covalent radii out), so the cost is `O(N^2 k)` for `k` neighbours.
//! - The HB coordination numbers of the softened A-H bonds (no log cap), within the bond
//!   term.
//! - The EEQ charges respond to the positions (and the cell): [`super::hessian_eeq`],
//!   `O(N^3)`.
//!
//! The Hessian is that of the gradient (and of the strain derivative): where xtb leaves a
//! derivative out (the dependence of the Ewald splitting parameter, of the images of the
//! pairs, and of the sets of translations on the positions and the cell), so does this.
//! Where the energy is not smooth, it is the Hessian of the side the structure is on: the
//! hydrogen-bond terms drop to zero where their out-of-line exponent passes 15 (as in xtb; a
//! jump of about 1e-10 Hartree/Bohr in the gradient), and a bend with `theta_0 = pi` has a
//! cusp at `theta = 0`, next to which xtb's topology of some metal complexes puts a bend (at
//! N between C and the metal of the hexacyanides of Fe and Co, at 0.01 to 0.1 degrees, where
//! the curvature is 1e2 Hartree/Bohr^2).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;
use std::f64::consts::PI;

use faer::Mat;

use super::angles::Angle;
use super::constants as k;
use super::eg_disp::{table_index, weights, ReferenceC6};
use super::eg_hb::{batm, hb_xb, torsions, HbCutoff, HessianSink, TermSink};
use super::elements::{rab_estimate_shifted, rab_slopes, Atom};
use super::hessian_core::{
    erf_count, merged, radial_block, square_block, Acc, Chain, CnSecond, PairSum, Vars,
};
use super::hessian_eeq;
use super::lattice::{is_central, Cell, Frame, Lattice, Translations, HOME};
use super::neighbors::V3;
use super::periodic_eg::{translation_vectors, LATTICE_SUM_CUTOFF};
use super::GfnffTopology;
use crate::error::{MoptError, MoptResult};
use crate::math::ad::Scalar;
use crate::parameters::gfnff::{GFNFF_DISP_NREF, GFNFF_DISP_REFCN};

/// A cell as the coefficients of the lattice vectors.
fn coefficients(n: Cell) -> V3 {
    n.map(f64::from)
}

/// `exp(-alpha r^1.5) A / r`: its first and second derivatives in `r`.
fn repulsion_derivatives(a: f64, alpha: f64, r: f64) -> (f64, f64) {
    let u = alpha * r.powf(1.5);
    let f = (-u).exp() * a / r;
    let s = 1.5 * u + 1.0;
    (-f * s / r, f * (s * s + 1.0 - 0.75 * u) / (r * r))
}

/// Non-bonded repulsion of a molecule.
fn nonbonded_repulsion(t: &GfnffTopology, x: &[V3], acc: &mut Acc) {
    let el = &t.setup.el;
    for i in 0..x.len() {
        for j in 0..i {
            let rv = [0, 1, 2].map(|c| x[i][c] - x[j][c]);
            let r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
            if !(1.0e-8..=k::REPTHR).contains(&r2) || t.setup.bpair.home(i, j) == 1 {
                continue;
            }
            let alpha = t
                .nonbonded
                .alphanb(&t.setup, i, j, Some(t.setup.bpair.home(i, j)));
            let a = el[i].raw.repz * el[j].raw.repz * k::REPSCALN;
            let (d1, d2) = repulsion_derivatives(a, alpha, r2.sqrt());
            acc.pair_block(i, j, &radial_block(rv, d1, d2));
        }
    }
}

/// Non-bonded repulsion of a cell over the translations of `tr` (an atom and its own images
/// only through the lattice vectors).
fn periodic_nonbonded_repulsion(
    t: &GfnffTopology,
    x: &[V3],
    lattice: &Lattice,
    tr: &Translations,
    acc: &mut Acc,
) {
    let s = &t.setup;
    let with_lattice = acc.vars.lattice;
    for i in 0..x.len() {
        let upper = if with_lattice { i + 1 } else { i };
        for j in 0..upper {
            let mut sum = PairSum::default();
            for &cell in &tr.cells {
                let tv = lattice.translation(cell);
                let rv = [0, 1, 2].map(|c| x[i][c] + tv[c] - x[j][c]);
                let r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
                if !(1.0e-8..=k::REPTHR).contains(&r2) {
                    continue;
                }
                let bp = if is_central(cell) {
                    s.bpair.get(i, j, cell)
                } else {
                    None
                };
                if bp == Some(1) {
                    continue;
                }
                let alpha = t.nonbonded.alphanb(s, i, j, bp);
                let a = s.el[i].raw.repz * s.el[j].raw.repz * k::REPSCALN;
                let (d1, d2) = repulsion_derivatives(a, alpha, r2.sqrt());
                // x_i + L n - x_j: the image of i in cell n.
                sum.add(coefficients(cell), &radial_block(rv, d1, d2), with_lattice);
            }
            acc.pair_sum(i, j, &sum, 1.0);
        }
    }
}

/// Bonded repulsion.
fn bonded_repulsion(t: &GfnffTopology, f: &Frame<'_>, acc: &mut Acc) {
    let el = &t.setup.el;
    for &(jat, iat, cell) in &t.setup.bonds {
        let pj = f.at(jat, cell);
        let rv = [0, 1, 2].map(|c| f.x[iat][c] - pj[c]);
        let r = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
        let alpha = (el[iat].raw.repa * el[jat].raw.repa).sqrt();
        let a = el[iat].raw.repz * el[jat].raw.repz * k::REPSCALB;
        let (d1, d2) = repulsion_derivatives(a, alpha, r);
        let dn = coefficients(cell).map(|c| -c);
        acc.pair_cell(iat, jat, dn, &radial_block(rv, d1, d2));
    }
}

/// The second CN derivatives of the D4 reference weights (those of [`weights`]).
fn weight_second(t: usize, cn: f64) -> [f64; 7] {
    const WF: f64 = 4.0;
    let nref = GFNFF_DISP_NREF[t];
    let refcn = &GFNFF_DISP_REFCN[t];
    let mut e = [0.0; 7];
    let mut e1 = [0.0; 7];
    let mut e2 = [0.0; 7];
    for r in 0..nref {
        let val = -WF * (cn - refcn[r]).powi(2);
        let w = if val < -200.0 { 0.0 } else { val.exp() };
        let a = 2.0 * WF * (refcn[r] - cn);
        e[r] = w;
        e1[r] = a * w;
        e2[r] = (a * a - 2.0 * WF) * w;
    }
    let total: f64 = e.iter().sum();
    if total <= 1.0e-80 {
        return [0.0; 7];
    }
    let (n1, n2): (f64, f64) = (e1.iter().sum(), e2.iter().sum());
    let inv = 1.0 / total;
    let mut out = [0.0; 7];
    for r in 0..nref {
        out[r] = e2[r] * inv - 2.0 * e1[r] * n1 * inv * inv - e[r] * n2 * inv * inv
            + 2.0 * e[r] * n1 * n1 * inv * inv * inv;
    }
    out
}

/// Dispersion over the translation vectors `trans` of the cells `cells` (the origin alone
/// for a molecule, with the pairs `j < i` within `DISPTHR`; a cell: `j <= i` within 60 Bohr).
#[allow(clippy::too_many_arguments)]
fn dispersion(
    el: &[Atom],
    x: &[V3],
    trans: &[V3],
    cells: &[V3],
    periodic: bool,
    cn: &CnSecond,
    zetac6: &[Vec<f64>],
    acc: &mut Acc,
    chain: &mut Chain,
) -> MoptResult<()> {
    let n = x.len();
    let vars = acc.vars;
    let tab: Vec<usize> = el.iter().map(table_index).collect::<MoptResult<_>>()?;
    let w: Vec<([f64; 7], [f64; 7], [f64; 7])> = (0..n)
        .map(|i| {
            let (g, d) = weights(tab[i], cn.cn[i]);
            (g, d, weight_second(tab[i], cn.cn[i]))
        })
        .collect();
    let c6tab = ReferenceC6::new(&tab);
    let cutoff2 = LATTICE_SUM_CUTOFF * LATTICE_SUM_CUTOFF;
    for i in 0..n {
        let upper = if periodic { i + 1 } else { i };
        for j in 0..upper {
            let (ti, tj) = (tab[i], tab[j]);
            let block = c6tab.get(ti, tj);
            let (mut c6, mut dc6i, mut dc6j) = (0.0, 0.0, 0.0);
            let (mut c6ii, mut c6jj, mut c6ij) = (0.0, 0.0, 0.0);
            for ri in 0..GFNFF_DISP_NREF[ti] {
                for rj in 0..GFNFF_DISP_NREF[tj] {
                    let refc6 = block[ri][rj];
                    c6 += w[i].0[ri] * w[j].0[rj] * refc6;
                    dc6i += w[i].1[ri] * w[j].0[rj] * refc6;
                    dc6j += w[i].0[ri] * w[j].1[rj] * refc6;
                    c6ii += w[i].2[ri] * w[j].0[rj] * refc6;
                    c6jj += w[i].0[ri] * w[j].2[rj] * refc6;
                    c6ij += w[i].1[ri] * w[j].1[rj] * refc6;
                }
            }
            let r4r2ij = 3.0 * el[i].sqrt_zr4r2 * el[j].sqrt_zr4r2;
            let r0 =
                (k::D3A1 * (el[i].sqrt_zr4r2 * el[j].sqrt_zr4r2 * 3.0).sqrt() + k::D3A2).powi(2);
            let (q6, q8) = (r0.powi(3), r0.powi(4));
            let zeta = zetac6[i][j];
            // Over the translations: the sum of disp, of its derivative along r (and its
            // lattice moments), and the blocks.
            let mut sum = 0.0;
            let mut grad = [0.0; 3];
            let mut grad_lattice = [[0.0; 3]; 3];
            let mut blocks = PairSum::default();
            for (tv, cell) in trans.iter().zip(cells) {
                let rv = [0, 1, 2].map(|c| x[i][c] - x[j][c] - tv[c]);
                let s = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
                let inside = if periodic {
                    s <= cutoff2 && s >= 1.0e-12
                } else {
                    s < k::DISPTHR
                };
                if !inside {
                    continue;
                }
                let t6 = 1.0 / (s.powi(3) + q6);
                let t8 = 1.0 / (s.powi(4) + q8);
                let t6s = -3.0 * s * s * t6 * t6;
                let t6ss = -6.0 * s * t6 * t6 + 18.0 * s.powi(4) * t6.powi(3);
                let t8s = -4.0 * s.powi(3) * t8 * t8;
                let t8ss = -12.0 * s * s * t8 * t8 + 32.0 * s.powi(6) * t8.powi(3);
                let disp = (t6 + 2.0 * r4r2ij * t8) * zeta;
                let fs = (t6s + 2.0 * r4r2ij * t8s) * zeta;
                let fss = (t6ss + 2.0 * r4r2ij * t8ss) * zeta;
                sum += disp;
                if i != j || vars.lattice {
                    let dn = cell.map(|c| -c);
                    blocks.add(dn, &square_block(rv, fs, fss), vars.lattice);
                    for a in 0..3 {
                        let g = 2.0 * fs * rv[a];
                        grad[a] += g;
                        for (p, gl) in grad_lattice.iter_mut().enumerate() {
                            gl[a] += dn[p] * g;
                        }
                    }
                }
            }
            if sum == 0.0 {
                continue;
            }
            acc.pair_sum(i, j, &blocks, -c6);
            chain.w[i] -= dc6i * sum;
            chain.w[j] -= dc6j * sum;
            chain.m[(i, i)] -= c6ii * sum;
            chain.m[(j, j)] -= c6jj * sum;
            chain.mix(i, j, -c6ij * sum);
            let d = vars.pair_sum_vector(i, j, grad, &grad_lattice);
            if !d.is_empty() {
                chain.cross(i, -dc6i, &d);
                chain.cross(j, -dc6j, &d);
            }
        }
    }
    Ok(())
}

/// The HB coordination number of an H atom (`dncoord_erf`, no cap) and its derivatives.
struct HbCn {
    value: f64,
    /// `(B, x_B - x_H, cell of B, d1, d2)` of each counted acceptor image.
    pairs: Vec<(usize, V3, V3, f64, f64)>,
    /// `d cn_H / dx` by variable.
    grad: Vec<(usize, V3)>,
}

fn hb_coordination(t: &GfnffTopology, f: &Frame<'_>, vars: Vars) -> HashMap<usize, HbCn> {
    const KN: f64 = 27.5;
    const RCOV_SCAL: f64 = 1.78;
    const SQRTPI: f64 = 1.772_453_850_91;
    let el = &t.setup.el;
    let mut out: HashMap<usize, HbCn> = HashMap::new();
    for ah in &t.bond_hb.ah {
        let h = ah.h;
        let entry = out.entry(h).or_insert_with(|| HbCn {
            value: 0.0,
            pairs: Vec::new(),
            grad: Vec::new(),
        });
        for &(b, cb) in &ah.b {
            let pb = f.at(b, cb);
            let rv = [0, 1, 2].map(|c| pb[c] - f.x[h][c]);
            let r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
            if r2 > 900.0 {
                continue;
            }
            let r = r2.sqrt();
            let rc = RCOV_SCAL * (el[h].rcov + el[b].rcov);
            let (v, d1, d2) = erf_count(KN, rc, r, SQRTPI);
            entry.value += v;
            let dn = coefficients(cb);
            entry.pairs.push((b, rv, dn, d1, d2));
            let d = rv.map(|c| d1 * c / r);
            entry.grad.extend(vars.pair_vector(b, h, dn, d));
        }
    }
    for e in out.values_mut() {
        e.grad = merged(std::mem::take(&mut e.grad));
    }
    out
}

/// Bond stretches: the radial part, the CN dependence of the reference lengths, and the HB
/// coordination of the softened A-H bonds.
fn bonds(t: &GfnffTopology, f: &Frame<'_>, cn: &CnSecond, acc: &mut Acc, chain: &mut Chain) {
    let el = &t.setup.el;
    let vars = acc.vars;
    let any_hb = t.bond_hb.nr_hb.iter().any(|&v| v > 0);
    let hb = if any_hb {
        hb_coordination(t, f, vars)
    } else {
        HashMap::new()
    };
    for (bi, &(jat, iat, cell)) in t.setup.bonds.iter().enumerate() {
        let p = &t.bonds[bi];
        let pj = f.at(jat, cell);
        let rv = [0, 1, 2].map(|c| f.x[iat][c] - pj[c]);
        let r = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
        let rho = rab_estimate_shifted(&el[iat], cn.cn[iat], &el[jat], cn.cn[jat], p.shift);
        let (ff, _, _) = rab_slopes(&el[iat], &el[jat]);
        let (sa, sb) = (ff * el[iat].raw.rab[2], ff * el[jat].raw.rab[2]);
        let mut t8 = p.alpha;
        let mut soft: Option<(&HbCn, f64)> = None;
        if t.bond_hb.nr_hb[bi] >= 1 {
            let h = if el[iat].z() == 1 {
                iat
            } else if el[jat].z() == 1 {
                jat
            } else {
                continue;
            };
            let t1 = 1.0 - k::VBOND_SCALE;
            let hc = hb.get(&h);
            let cnh = hc.map_or(0.0, |c| c.value);
            t8 = (-t1 * cnh + 1.0) * p.alpha;
            soft = hc.map(|c| (c, -p.alpha * t1));
        }
        let dr = r - rho;
        let dd = p.prefactor * (-t8 * dr * dr).exp();
        let d_r = -2.0 * t8 * dr * dd;
        let d_rr = (4.0 * t8 * t8 * dr * dr - 2.0 * t8) * dd;
        // x_iat - (x_jat + L n): the coefficients of the lattice vectors are -n.
        let dn = coefficients(cell).map(|c| -c);
        acc.pair_cell(iat, jat, dn, &radial_block(rv, d_r, d_rr));
        // rho: dD/drho = -D_r, d2D/drho2 = D_rr, d2D/dr drho = -D_rr.
        chain.w[iat] -= d_r * sa;
        chain.w[jat] -= d_r * sb;
        chain.m[(iat, iat)] += d_rr * sa * sa;
        chain.m[(jat, jat)] += d_rr * sb * sb;
        chain.mix(iat, jat, d_rr * sa * sb);
        let u = rv.map(|c| c / r);
        let gr = vars.pair_vector(iat, jat, dn, u);
        chain.cross(iat, -d_rr * sa, &gr);
        chain.cross(jat, -d_rr * sb, &gr);
        // The steepness t8 = alpha (1 - t1 cn_H): d t8 / d cn_H = gt.
        if let Some((hc, gt)) = soft {
            let d_t = -dr * dr * dd;
            let d_tt = dr.powi(4) * dd;
            let d_rt = (-2.0 * dr + 2.0 * t8 * dr.powi(3)) * dd;
            acc.outer(d_tt * gt * gt, &hc.grad, &hc.grad);
            let h = if el[iat].z() == 1 { iat } else { jat };
            for &(b, bv, bn, d1, d2) in &hc.pairs {
                let c = d_t * gt;
                acc.pair_cell(b, h, bn, &radial_block(bv, c * d1, c * d2));
            }
            acc.outer(d_rt * gt, &gr, &hc.grad);
            acc.outer(d_rt * gt, &hc.grad, &gr);
            chain.cross(iat, -d_rt * gt * sa, &hc.grad);
            chain.cross(jat, -d_rt * gt * sb, &hc.grad);
        }
    }
}

/// `gfnffdampa` for the generic bends.
fn dampa<T: Scalar>(a: &Atom, b: &Atom, r2: T) -> T {
    let rcut = k::ATCUTA * (a.rcov + b.rcov).powi(2);
    let rr = (r2 / T::from_f64(rcut)).powi(2);
    T::one() / (T::one() + rr)
}

/// `acos(1 - u)^2`, smooth through `u = 0` (a linear bend): the power series
/// `sum_n 2^(n+1) u^n / (n^2 C(2n, n))` below `u = 0.1`, else the closed form.
fn acos_square<T: Scalar>(u: T) -> T {
    if u.val() >= 0.1 {
        let g = (T::one() - u).acos();
        return g * g;
    }
    const TERMS: usize = 32;
    let mut coef = [0.0; TERMS + 1];
    coef[1] = 2.0;
    for nn in 1..TERMS {
        let m = nn as f64;
        coef[nn + 1] = coef[nn] * m * m / ((2.0 * m + 1.0) * (m + 1.0));
    }
    let mut acc = T::from_f64(coef[TERMS]);
    for nn in (1..TERMS).rev() {
        acc = acc * u + T::from_f64(coef[nn]);
    }
    acc * u
}

/// One bend (`egbend`) generic over the scalar, `p = [i, j, k]` with `i` the vertex.
fn bend_energy<T: Scalar>(a: &Angle, el: &[Atom], p: &[[T; 3]]) -> T {
    let (pi, pj, pk) = (p[0], p[1], p[2]);
    let vab = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
    let vcb = [pk[0] - pi[0], pk[1] - pi[1], pk[2] - pi[2]];
    let rab2 = T::dot3(vab, vab);
    let rcb2 = T::dot3(vcb, vcb);
    let damp = dampa(&el[a.j], &el[a.i], rab2) * dampa(&el[a.k], &el[a.i], rcb2);
    let cosa = if rab2.val() > 0.0 && rcb2.val() > 0.0 {
        T::dot3(vab, vcb) / (rab2 * rcb2).sqrt()
    } else {
        T::zero()
    };
    let (c0, kijk) = (a.theta0, a.fc);
    let ea = if PI - c0 < 1.0e-6 {
        // (theta - c0)^2 with pi - theta = acos(-cos theta).
        let u = T::one() + cosa;
        let delta = PI - c0;
        let sq = acos_square(u);
        if delta == 0.0 || u.val() < 1.0e-10 {
            (sq + T::from_f64(delta * delta)) * T::from_f64(kijk)
        } else {
            let g = (-cosa).acos();
            (sq - g * T::from_f64(2.0 * delta) + T::from_f64(delta * delta)) * T::from_f64(kijk)
        }
    } else {
        let d = cosa - T::from_f64(c0.cos());
        d * d * T::from_f64(kijk)
    };
    ea * damp
}

/// Bends with hyper-dual numbers.
fn bends<K: TermSink>(angles: &[Angle], el: &[Atom], f: &Frame<'_>, sink: &mut K) -> f64 {
    angles
        .iter()
        .map(|a| {
            sink.term(&[(a.i, HOME), (a.j, a.cj), (a.k, a.ck)], f, |p| {
                bend_energy(a, el, p)
            })
        })
        .sum()
}

/// The Hessian (Hartree/Bohr^2) of the energy of `x` (continuous with the positions of the
/// topology in a cell, as for [`super::eg::energy_gradient`]): `3N x 3N` in the positions,
/// or with `with_lattice` (a cell only) `(3N + 9) x (3N + 9)` in the positions and the
/// lattice vectors as independent variables (the images of atom `j` at `x_j + L n`; the
/// variable `3N + 3k + c` is component `c` of lattice vector `k`).
pub fn hessian(
    t: &GfnffTopology,
    x: &[V3],
    lattice: Option<&Lattice>,
    hb_cutoff: HbCutoff,
    with_lattice: bool,
) -> MoptResult<Mat<f64>> {
    let n = x.len();
    if n != t.setup.el.len() {
        return Err(MoptError::config(
            "GFN-FF: structure does not match the topology",
        ));
    }
    let vars = Vars {
        atoms: n,
        lattice: with_lattice && lattice.is_some(),
    };
    let el = &t.setup.el;
    let f = Frame { x, lattice };
    let mut acc = Acc::new(vars);
    let mut chain = Chain::new(vars);
    let (cn, hb_translations) = match lattice {
        None => {
            nonbonded_repulsion(t, x, &mut acc);
            let cn = CnSecond::molecule(el, x);
            dispersion(
                el,
                x,
                &[[0.0; 3]],
                &[[0.0; 3]],
                false,
                &cn,
                &t.nonbonded.zetac6,
                &mut acc,
                &mut chain,
            )?;
            hessian_eeq::molecule(t, x, &cn, &mut acc, &mut chain)?;
            (cn, Translations::molecule())
        }
        Some(l) => {
            let tr = Translations::new(l, k::REPTHR.sqrt())?;
            periodic_nonbonded_repulsion(t, x, l, &tr, &mut acc);
            let trv = translation_vectors(l, &tr);
            let cells: Vec<V3> = tr.cells.iter().map(|&c| coefficients(c)).collect();
            let cn = CnSecond::periodic(el, x, &trv, &cells, LATTICE_SUM_CUTOFF, vars);
            dispersion(
                el,
                x,
                &trv,
                &cells,
                true,
                &cn,
                &t.nonbonded.zetac6,
                &mut acc,
                &mut chain,
            )?;
            hessian_eeq::periodic(t, x, l, &cn, &mut acc, &mut chain)?;
            (cn, Translations::new(l, k::HBTHR2.sqrt())?)
        }
    };
    bonds(t, &f, &cn, &mut acc, &mut chain);
    bonded_repulsion(t, &f, &mut acc);
    {
        let mut sink = HessianSink {
            h: &mut acc.h,
            atoms: n,
            lattice: vars.lattice,
        };
        bends(&t.angles, el, &f, &mut sink);
        torsions(&t.torsions, el, &f, &mut sink);
        batm(&t.setup, &t.nonbonded, &f, &mut sink);
        hb_xb(
            &t.setup,
            &t.nonbonded,
            &f,
            &hb_translations,
            hb_cutoff,
            &mut sink,
        );
    }
    chain.apply(&cn, &mut acc);
    let dim = vars.dim();
    let h = &acc.h;
    Ok(Mat::from_fn(dim, dim, |i, j| 0.5 * (h[(i, j)] + h[(j, i)])))
}

/// The energies of the bends with the generic expression (for the tests against the
/// gradient's own).
#[cfg(test)]
pub(super) fn generic_bend_energy(t: &GfnffTopology, x: &[V3]) -> f64 {
    struct ValueOnly;
    impl TermSink for ValueOnly {
        type S = f64;
        fn term<F>(&mut self, atoms: &[(usize, Cell)], fr: &Frame<'_>, f: F) -> f64
        where
            F: Fn(&[[f64; 3]]) -> f64,
        {
            let p: Vec<[f64; 3]> = atoms.iter().map(|&(a, n)| fr.at(a, n)).collect();
            f(&p)
        }
    }
    bends(&t.angles, &t.setup.el, &Frame::molecule(x), &mut ValueOnly)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use faer::Col;

    use super::*;
    use crate::domain::{Element, MolecularSystem, SpinCharge};

    fn system(symbols: &[&str], x: &[f64]) -> MolecularSystem {
        let el: Vec<Element> = symbols
            .iter()
            .map(|s| Element::parse_token(s).unwrap())
            .collect();
        MolecularSystem::from_flat_bohr(
            el,
            &Col::from_fn(x.len(), |i| x[i]),
            SpinCharge {
                charge: 0,
                multiplicity: 1,
            },
            None,
        )
        .unwrap()
    }

    fn positions(sys: &MolecularSystem) -> Vec<V3> {
        let x = sys.coords_flat_bohr();
        (0..sys.n_atoms())
            .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
            .collect()
    }

    /// Largest deviation of the analytic Hessian from the five-point differences of the
    /// gradient.
    fn deviation(sys: &MolecularSystem) -> f64 {
        let topo = GfnffTopology::new(sys).unwrap();
        let cut = HbCutoff::default();
        let h = topo.hessian_with(sys, cut).unwrap();
        let x0 = sys.coords_flat_bohr();
        let step = 1.0e-3;
        let mut worst = 0.0_f64;
        for k in 0..x0.nrows() {
            let g = |s: f64| {
                let mut x = x0.clone();
                x[k] += s * step;
                topo.compute_with(&sys.with_flat_coords_bohr(&x).unwrap(), cut)
                    .unwrap()
                    .gradient
            };
            let (p1, m1, p2, m2) = (g(1.0), g(-1.0), g(2.0), g(-2.0));
            for i in 0..x0.nrows() {
                let fd = (8.0 * (p1[i] - m1[i]) - (p2[i] - m2[i])) / (12.0 * step);
                worst = worst.max((h[(i, k)] - fd).abs());
            }
        }
        worst
    }

    #[test]
    fn the_generic_bends_have_the_energy_of_the_gradient() {
        let sys = system(
            &["C", "O", "H", "H", "O", "H", "H"],
            &[
                -5.785, -1.923, -0.029, -5.578, 0.179, 0.724, -4.278, -2.748, -1.130, -7.479,
                -2.976, 0.395, -0.830, 2.435, -0.786, -0.742, 4.042, 0.091, -2.447, 1.694, -0.274,
            ],
        );
        let topo = GfnffTopology::new(&sys).unwrap();
        let x = positions(&sys);
        let (energy, ..) =
            super::super::eg::energy_gradient(&topo, &x, None, HbCutoff::default()).unwrap();
        let generic = generic_bend_energy(&topo, &x);
        assert!(
            (generic - energy.angle).abs() < 1.0e-13,
            "{generic} {}",
            energy.angle
        );
    }

    #[test]
    fn a_linear_bend_has_a_finite_hessian_at_the_linear_geometry() {
        // HCN exactly on the z axis: the bend at C is linear (theta_0 = pi).
        let sys = system(
            &["H", "C", "N"],
            &[0.0, 0.0, -2.01, 0.0, 0.0, 0.0, 0.0, 0.0, 2.18],
        );
        let topo = GfnffTopology::new(&sys).unwrap();
        assert!(topo.angles.iter().any(|a| a.theta0 == PI));
        let h = topo.hessian_with(&sys, HbCutoff::default()).unwrap();
        assert!((0..9).all(|i| (0..9).all(|j| h[(i, j)].is_finite())));
        let worst = deviation(&sys);
        assert!(worst < 1.0e-7, "{worst:.3e}");
        // Bent a little: the series and the closed form of acos^2 on either side of 0.1.
        let bent = system(
            &["H", "C", "N"],
            &[0.3, 0.0, -1.98, 0.0, 0.0, 0.0, 0.0, 0.1, 2.18],
        );
        let worst = deviation(&bent);
        assert!(worst < 1.0e-7, "{worst:.3e}");
    }

    #[test]
    fn the_power_series_of_acos_squared_meets_the_closed_form() {
        for u in [1.0e-6, 0.01, 0.05, 0.0999] {
            let series = acos_square(u);
            let closed = (1.0 - u).acos().powi(2);
            assert!((series - closed).abs() < 1.0e-15 * closed.max(1.0), "{u}");
        }
    }
}
