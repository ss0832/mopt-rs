//! GFN-FF energy and analytic gradient of a molecule, ported from xtb's `gfnff_eg`, `egbond`,
//! `egbond_hb`, `dncoord_erf`, `egbend`, `gfnffdampa`, and `goed_gfnff`
//! (`src/gfnff/gfnff_eg.f90`; LGPL-3.0-or-later). The HB, XB, bonded ATM, and torsion terms are
//! in [`super::eg_hb`], the dispersion in [`super::eg_disp`].
//!
//! Terms: non-bonded and bonded repulsion, EEQ electrostatics (charges from the current
//! structure; the energy is variational in the charges, so only the CN derivative of the
//! right-hand side enters the gradient), D4-like dispersion, CN-dependent bond stretches (with
//! the HB-coordination softening of A-H bonds), bends, torsions and out-of-plane terms, HB, XB,
//! and bonded ATM. xtb's triple-bond torsion `sTors` is not ported: its atom list stays empty in
//! xtb 6.7.1 (the C4 search tests the neighbour count of the central atom, which is 2, against 3),
//! so its energy is always zero.
//!
//! In a three-dimensional periodic cell the repulsion, the coordination numbers, the dispersion,
//! and the electrostatics are lattice sums ([`super::periodic_eg`], [`super::periodic_cn`]);
//! the bonded terms and the HB/XB terms see the images of the topology; every term also gives
//! its strain derivative `sigma` (xtb's `sigma`, not divided by the volume).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::{Col, Mat};

use super::angles::Angle;
use super::constants as k;
use super::eg_disp::dispersion;
use super::eg_hb::{batm, hb_xb, torsions, GradientSink, HbCutoff};
use super::elements::{rab_estimate_shifted, rab_slopes, Atom};
use super::lattice::{Frame, Translations};
use super::neighbors::{dist, V3};
use super::periodic_cn::{
    coordination_numbers, molecular_coordination_numbers, CoordinationNumbers,
};
use super::periodic_eg::{self, translation_vectors, Sigma, LATTICE_SUM_CUTOFF};
use super::GfnffTopology;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{mul, DenseLlt, DenseLu};
use crate::pes::native::parallel::{add_into, for_row_chunks};

/// GFN-FF energy terms (Hartree), named as in xtb's output.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GfnffEnergy {
    /// Total energy.
    pub total: f64,
    /// Bond stretches.
    pub bond: f64,
    /// Bends.
    pub angle: f64,
    /// Torsions and out-of-plane terms.
    pub torsion: f64,
    /// Bonded and non-bonded repulsion.
    pub repulsion: f64,
    /// EEQ electrostatics.
    pub electrostatic: f64,
    /// Dispersion.
    pub dispersion: f64,
    /// Hydrogen bonds.
    pub hbond: f64,
    /// Halogen bonds.
    pub xbond: f64,
    /// Bonded three-body (ATM) term.
    pub batm: f64,
}

/// Non-bonded repulsion of all pairs that are not bonded.
fn nonbonded_repulsion(t: &GfnffTopology, x: &[V3], g: &mut [V3]) -> f64 {
    let el = &t.setup.el;
    let n = x.len();
    // The pairs in fixed chunks of atoms (`pes::native::parallel`), each with its own sums.
    let chunk = |rows: std::ops::Range<usize>| {
        let mut e = 0.0;
        let mut gc = vec![[0.0; 3]; n];
        for i in rows {
            for j in 0..=i {
                let r = dist(x[i], x[j]);
                let r2 = r * r;
                if !(1.0e-8..=k::REPTHR).contains(&r2) {
                    continue;
                }
                if t.setup.bpair.home(i, j) == 1 {
                    continue;
                }
                let rab = r2.sqrt();
                let t16 = r2.powf(0.75);
                let t19 = t16 * t16;
                let t8 = t16
                    * t.nonbonded
                        .alphanb(&t.setup, i, j, Some(t.setup.bpair.home(i, j)));
                let t26 = (-t8).exp() * el[i].raw.repz * el[j].raw.repz * k::REPSCALN;
                e += t26 / rab;
                let t27 = t26 * (1.5 * t8 + 1.0) / t19;
                for c in 0..3 {
                    let r3 = (x[i][c] - x[j][c]) * t27;
                    gc[i][c] -= r3;
                    gc[j][c] += r3;
                }
            }
        }
        (e, gc)
    };
    let mut e = 0.0;
    for_row_chunks(n, chunk, |(ec, gc)| {
        e += ec;
        add_into(g, &gc);
    });
    e
}

/// Bonded repulsion of the bonds `(j, i, n)` (the image of `j` in cell `n`).
fn bonded_repulsion(t: &GfnffTopology, f: &Frame<'_>, g: &mut [V3], sigma: &mut Sigma) -> f64 {
    let el = &t.setup.el;
    let mut e = 0.0;
    for &(jat, iat, cell) in &t.setup.bonds {
        let pj = f.at(jat, cell);
        let d = [0, 1, 2].map(|c| f.x[iat][c] - pj[c]);
        let rab = dist(f.x[iat], pj);
        let r2 = rab * rab;
        let alpha = (el[iat].raw.repa * el[jat].raw.repa).sqrt();
        let repab = el[iat].raw.repz * el[jat].raw.repz * k::REPSCALB;
        let t16 = r2.powf(0.75);
        let t19 = t16 * t16;
        let t26 = (-alpha * t16).exp() * repab;
        e += t26 / rab;
        let t27 = t26 * (1.5 * alpha * t16 + 1.0) / t19;
        for c in 0..3 {
            g[iat][c] -= d[c] * t27;
            g[jat][c] += d[c] * t27;
            for b in 0..3 {
                sigma[b][c] -= d[b] * d[c] * t27;
            }
        }
    }
    e
}

/// EEQ charges and electrostatic energy (`goed_gfnff`), with the pair intermediates
/// `(gamma_ij, erf(gamma_ij r_ij))` for the gradient.
struct Electrostatics {
    q: Vec<f64>,
    energy: f64,
    pair: Vec<(f64, f64)>,
}

/// The index of the pair `j < i` in the lower triangle stored row by row.
pub(super) fn pair_index(i: usize, j: usize) -> usize {
    i * (i - 1) / 2 + j
}

/// The solution `[q; lambda]` of the bordered EEQ system `[[J, B], [B^T, 0]] [q; lambda] =
/// [b; Q]` held in `a` and `rhs` (`n` atoms, the charge constraints after them). `J` is
/// factorised by Cholesky, `J = L L^T`, and the constraints solved through the Schur
/// complement `S = B^T J^-1 B = W^T W` with `W = L^-1 B` (forward substitution alone):
/// `S lambda = W^T L^-1 b - Q`, then `q = J^-1 (b - B lambda)`. About half the work of an LU of
/// the whole matrix, the same solution up to rounding; when `J` is not positive definite, the
/// whole matrix by LU with partial pivoting (xtb's `dsysv`). `None` when the solution is not
/// finite. New (xtb solves the whole matrix).
pub(super) fn solve_bordered(a: &Mat<f64>, rhs: &Col<f64>, n: usize) -> Option<Vec<f64>> {
    let m = a.nrows();
    let nc = m - n;
    let out: Vec<f64> = match DenseLlt::new(a.submatrix(0, 0, n, n)) {
        Some(llt) => {
            // L^-1 b and L^-1 B in one forward substitution.
            let r = Mat::from_fn(
                n,
                nc + 1,
                |i, c| {
                    if c == 0 {
                        rhs[i]
                    } else {
                        a[(i, n + c - 1)]
                    }
                },
            );
            let w = llt.solve_lower(r.as_ref());
            let lambda: Vec<f64> = if nc == 0 {
                Vec::new()
            } else {
                let wb = w.as_ref().subcols(1, nc);
                let schur = mul(wb.transpose(), wb);
                let wt = mul(wb.transpose(), w.as_ref().subcols(0, 1));
                let t = Mat::from_fn(nc, 1, |p, _| wt[(p, 0)] - rhs[n + p]);
                let l = DenseLu::new(schur.as_ref()).ok()?.solve(t.as_ref());
                (0..nc).map(|p| l[(p, 0)]).collect()
            };
            let corrected = Mat::from_fn(n, 1, |i, _| {
                rhs[i]
                    - lambda
                        .iter()
                        .enumerate()
                        .map(|(c, l)| a[(i, n + c)] * l)
                        .sum::<f64>()
            });
            let q = llt.solve(corrected.as_ref());
            (0..n)
                .map(|i| q[(i, 0)])
                .chain(lambda.iter().copied())
                .collect()
        }
        None => {
            let sol = DenseLu::new(a.as_ref()).ok()?.solve(rhs.as_mat());
            (0..m).map(|i| sol[(i, 0)]).collect()
        }
    };
    out.iter().all(|v| v.is_finite()).then_some(out)
}

fn electrostatics(t: &GfnffTopology, x: &[V3], cn: &[f64]) -> MoptResult<Electrostatics> {
    const TSQRT2PI: f64 = 0.797_884_560_802_866;
    let s = &t.setup;
    let n = x.len();
    let (a, rhs, pair) = eeq_system(t, x, cn);
    let sol = solve_bordered(&a, &rhs, n)
        .ok_or_else(|| MoptError::calc("GFN-FF: singular EEQ system"))?;
    let mut q: Vec<f64> = sol[..n].to_vec();
    if n == 1 {
        q[0] = s.qfrag[0];
    }
    let mut energy = 0.0;
    for i in 0..n {
        for j in 0..i {
            energy += q[i] * q[j] * pair[pair_index(i, j)].1 / dist(x[i], x[j]);
        }
        energy -= q[i] * (s.chieeq[i] + s.el[i].raw.cnf * cn[i].sqrt());
        energy += q[i] * q[i] * 0.5 * (s.gameeq[i] + TSQRT2PI / s.alpeeq[i].sqrt());
    }
    Ok(Electrostatics { q, energy, pair })
}

/// The EEQ system of a molecule (`goed_gfnff`): the matrix with the fragment constraints,
/// the right-hand side, and the pair intermediates `(gamma_ij, erf(gamma_ij r_ij))` (the pairs
/// `j < i` row by row, [`pair_index`]).
pub(super) fn eeq_system(
    t: &GfnffTopology,
    x: &[V3],
    cn: &[f64],
) -> (Mat<f64>, Col<f64>, Vec<(f64, f64)>) {
    const TSQRT2PI: f64 = 0.797_884_560_802_866;
    let s = &t.setup;
    let n = x.len();
    let nfrag = s.fraglist.iter().copied().max().unwrap_or(0);
    let m = n + nfrag;
    let mut a = Mat::<f64>::zeros(m, m);
    let mut rhs = Col::<f64>::zeros(m);
    let mut pair = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    // The pair intermediates (the error functions) in fixed chunks of rows, then the matrix.
    let chunk = |rows: std::ops::Range<usize>| {
        let mut out = Vec::new();
        for i in rows {
            for j in 0..i {
                let r = dist(x[i], x[j]);
                let gammij = 1.0 / (s.alpeeq[i] + s.alpeeq[j]).sqrt();
                out.push((gammij, libm::erf(gammij * r), r));
            }
        }
        out
    };
    for_row_chunks(n, chunk, |part| pair.extend(part));
    for i in 0..n {
        rhs[i] = s.chieeq[i] + s.el[i].raw.cnf * cn[i].sqrt();
        a[(i, i)] = TSQRT2PI / s.alpeeq[i].sqrt() + s.gameeq[i];
        for j in 0..i {
            let (_, tmp, r) = pair[pair_index(i, j)];
            a[(j, i)] = tmp / r;
            a[(i, j)] = tmp / r;
        }
    }
    let pair: Vec<(f64, f64)> = pair.into_iter().map(|(g, e, _)| (g, e)).collect();
    for f in 0..nfrag {
        rhs[n + f] = s.qfrag[f];
        for j in 0..n {
            if s.fraglist[j] == f + 1 {
                a[(n + f, j)] = 1.0;
                a[(j, n + f)] = 1.0;
            }
        }
    }
    (a, rhs, pair)
}

/// Gradient of the electrostatic energy at fixed charges; `dE / dCN` of the right-hand side
/// is added to `dedcn` (for the chain rule of the caller).
fn electrostatic_gradient(
    t: &GfnffTopology,
    x: &[V3],
    es: &Electrostatics,
    cn: &[f64],
    g: &mut [V3],
    dedcn: &mut [f64],
) {
    let sqrtpi = PI.sqrt();
    let n = x.len();
    let q = &es.q;
    let chunk = |rows: std::ops::Range<usize>| {
        let mut gc = vec![[0.0; 3]; n];
        for i in rows {
            for j in 0..i {
                let d = [0, 1, 2].map(|c| x[i][c] - x[j][c]);
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let rab = r2.sqrt();
                let (gammij, erff) = es.pair[pair_index(i, j)];
                let dd = (2.0 * gammij * (-gammij * gammij * r2).exp() / (sqrtpi * r2)
                    - erff / (rab * r2))
                    * q[i]
                    * q[j];
                for c in 0..3 {
                    gc[i][c] += d[c] * dd;
                    gc[j][c] -= d[c] * dd;
                }
            }
        }
        gc
    };
    for_row_chunks(n, chunk, |gc| add_into(g, &gc));
    for i in 0..n {
        dedcn[i] -= q[i] * t.setup.el[i].raw.cnf / (2.0 * cn[i].sqrt() + 1.0e-16);
    }
}

/// The HB coordination numbers of the HB-modified bonds (`dncoord_erf`).
struct HbCoordination {
    /// `cn_H` of every atom (only H atoms of A-H bonds are non-zero).
    cn: Vec<f64>,
    /// `d cn_H / d x_H` of every H.
    self_d: Vec<V3>,
    /// `d cn_H / d x_B` of each `(B, H)` pair.
    pair_d: std::collections::HashMap<(usize, usize), V3>,
    /// `d cn_H / d strain` of every H.
    strain: Vec<Sigma>,
}

fn hb_coordination(t: &GfnffTopology, f: &Frame<'_>) -> HbCoordination {
    const KN: f64 = 27.5;
    const RCOV_SCAL: f64 = 1.78;
    const HLFOSQRTPI: f64 = 1.0 / 1.772_453_850_91;
    let n = f.len();
    let el = &t.setup.el;
    let mut out = HbCoordination {
        cn: vec![0.0; n],
        self_d: vec![[0.0; 3]; n],
        pair_d: std::collections::HashMap::new(),
        strain: vec![[[0.0; 3]; 3]; n],
    };
    for ah in &t.bond_hb.ah {
        let h = ah.h;
        for &(b, cb) in &ah.b {
            let pb = f.at(b, cb);
            let rij = [0, 1, 2].map(|c| pb[c] - f.x[h][c]);
            let r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
            if r2 > 900.0 {
                continue;
            }
            let r = r2.sqrt();
            let rcovij = RCOV_SCAL * (el[h].rcov + el[b].rcov);
            let tmp = 0.5 * (1.0 + libm::erf(-KN * (r - rcovij) / rcovij));
            let dtmp =
                -HLFOSQRTPI * KN * (-(KN * KN) * (r - rcovij).powi(2) / rcovij.powi(2)).exp()
                    / rcovij;
            out.cn[h] += tmp;
            let v = rij.map(|c| dtmp * c / r);
            let entry = out.pair_d.entry((b, h)).or_insert([0.0; 3]);
            for c in 0..3 {
                entry[c] += v[c];
                out.self_d[h][c] -= v[c];
                for a in 0..3 {
                    out.strain[h][a][c] += rij[a] * v[c];
                }
            }
        }
    }
    out
}

/// Bond stretches with CN-dependent reference lengths, over the bonds `(j, i, n)`; `dE / dCN`
/// is added to `dedcn` (for the chain rule of the caller).
fn bonds(
    t: &GfnffTopology,
    f: &Frame<'_>,
    cn: &CoordinationNumbers,
    g: &mut [V3],
    sigma: &mut Sigma,
    dedcn: &mut [f64],
) -> f64 {
    let el = &t.setup.el;
    let any_hb = t.bond_hb.nr_hb.iter().any(|&v| v > 0);
    let hb = if any_hb {
        Some(hb_coordination(t, f))
    } else {
        None
    };
    // xtb's `considered_ABH`: the B part of the HB-CN gradient once per (A, B, H).
    let mut considered = std::collections::HashSet::new();
    let mut e = 0.0;
    for (bi, &(jat, iat, cell)) in t.setup.bonds.iter().enumerate() {
        let p = &t.bonds[bi];
        let pj = f.at(jat, cell);
        let rab = dist(pj, f.x[iat]);
        let rij = rab_estimate_shifted(&el[iat], cn.cn[iat], &el[jat], cn.cn[jat], p.shift);
        let (ff, _, _) = rab_slopes(&el[iat], &el[jat]);
        let (cfi, cfj) = (el[iat].raw.rab[2], el[jat].raw.rab[2]);
        let mut t8 = p.alpha;
        let mut soft: Option<(usize, usize, f64)> = None;
        if t.bond_hb.nr_hb[bi] >= 1 {
            let (h, a) = if el[iat].z() == 1 {
                (iat, jat)
            } else if el[jat].z() == 1 {
                (jat, iat)
            } else {
                continue;
            };
            let t1 = 1.0 - k::VBOND_SCALE;
            let cnh = hb.as_ref().map_or(0.0, |c| c.cn[h]);
            t8 = (-t1 * cnh + 1.0) * p.alpha;
            soft = Some((h, a, t1));
        }
        let dr = rab - rij;
        let dum = p.prefactor * (-t8 * dr * dr).exp();
        e += dum;
        let yy = 2.0 * t8 * dr * dum;
        let vrab = [0, 1, 2].map(|c| f.x[iat][c] - pj[c]);
        for c in 0..3 {
            let dg = -yy * vrab[c] / rab;
            g[iat][c] += dg;
            g[jat][c] -= dg;
            for a in 0..3 {
                sigma[a][c] += vrab[a] * dg;
            }
        }
        dedcn[iat] += yy * ff * cfi;
        dedcn[jat] += yy * ff * cfj;
        if let (Some((h, a, t1)), Some(hb)) = (soft, hb.as_ref()) {
            let zz = dum * p.alpha * dr * dr * t1;
            for ah in t
                .bond_hb
                .ah
                .iter()
                .filter(|ah| ah.h == h && ah.a == a && ah.ca == cell)
            {
                for c in 0..3 {
                    g[h][c] += zz * hb.self_d[h][c];
                }
                for a2 in 0..3 {
                    for c in 0..3 {
                        sigma[a2][c] += zz * hb.strain[h][a2][c];
                    }
                }
                for &(b, _) in &ah.b {
                    if considered.insert((a, b, h)) {
                        if let Some(d) = hb.pair_d.get(&(b, h)) {
                            for c in 0..3 {
                                g[b][c] += zz * d[c];
                            }
                        }
                    }
                }
            }
        }
    }
    e
}

/// `gfnffdampa`: bend distance damping and its derivative factor.
fn dampa(a: &Atom, b: &Atom, r2: f64) -> (f64, f64) {
    let rcut = k::ATCUTA * (a.rcov + b.rcov).powi(2);
    let rr = (r2 / rcut).powi(2);
    (
        1.0 / (1.0 + rr),
        -2.0 * 2.0 * rr / (r2 * (1.0 + rr).powi(2)),
    )
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Bends (`egbend`), the outer atoms at their images.
fn bends(angles: &[Angle], el: &[Atom], f: &Frame<'_>, g: &mut [V3], sigma: &mut Sigma) -> f64 {
    let mut e = 0.0;
    for a in angles {
        let (i, j, kk) = (a.i, a.j, a.k);
        let (pi, pj, pk) = (f.x[i], f.at(j, a.cj), f.at(kk, a.ck));
        let vab = [0, 1, 2].map(|c| pj[c] - pi[c]);
        let vcb = [0, 1, 2].map(|c| pk[c] - pi[c]);
        let rab2 = vab[0] * vab[0] + vab[1] * vab[1] + vab[2] * vab[2];
        let rcb2 = vcb[0] * vcb[0] + vcb[1] * vcb[1] + vcb[2] * vcb[2];
        let vp = cross(vcb, vab);
        let tot = vp[0] * vp[0] + vp[1] * vp[1] + vp[2] * vp[2];
        let rp = if tot > 0.0 { tot.sqrt() } else { 0.0 } + 1.0e-14;
        let (al, bl) = (rab2.sqrt(), rcb2.sqrt());
        let dot = vab[0] * vcb[0] + vab[1] * vcb[1] + vab[2] * vcb[2];
        let cosa = if al > 0.0 && bl > 0.0 {
            dot / (al * bl)
        } else {
            0.0
        };
        let cosa = cosa.clamp(-1.0, 1.0);
        let theta = cosa.acos();
        let (dampij, damp2ij) = dampa(&el[j], &el[i], rab2);
        let (dampjk, damp2jk) = dampa(&el[kk], &el[i], rcb2);
        let damp = dampij * dampjk;
        let (c0, kijk) = (a.theta0, a.fc);
        let (ea, deddt) = if PI - c0 < 1.0e-6 {
            let dt = theta - c0;
            (kijk * dt * dt, 2.0 * kijk * dt)
        } else {
            (
                kijk * (cosa - c0.cos()).powi(2),
                2.0 * kijk * theta.sin() * (c0.cos() - cosa),
            )
        };
        e += ea * damp;
        let rmul1 = -deddt / (rab2 * rp);
        let rmul2 = deddt / (rcb2 * rp);
        let deda = cross(vab, vp).map(|v| v * rmul1);
        let dedc = cross(vcb, vp).map(|v| v * rmul2);
        for c in 0..3 {
            let dedb = deda[c] + dedc[c];
            let term1 = ea * damp2ij * dampjk * vab[c];
            let term2 = ea * damp2jk * dampij * vcb[c];
            let (gi, gj, gk) = (
                -dedb * damp - term1 - term2,
                deda[c] * damp + term1,
                dedc[c] * damp + term2,
            );
            g[i][c] += gi;
            g[j][c] += gj;
            g[kk][c] += gk;
            for b in 0..3 {
                sigma[b][c] += pi[b] * gi + pj[b] * gj + pk[b] * gk;
            }
        }
    }
    e
}

/// Energy terms, EEQ charges, gradient (`g`, Hartree/Bohr per atom), and for a periodic
/// cell the strain derivative of `x` (Bohr; in a cell, continuous with the positions of the
/// topology).
pub fn energy_gradient(
    t: &GfnffTopology,
    x: &[V3],
    lattice: Option<&super::lattice::Lattice>,
    hb_cutoff: HbCutoff,
) -> MoptResult<(GfnffEnergy, Vec<f64>, Vec<V3>, Option<Sigma>)> {
    let n = x.len();
    if n != t.setup.el.len() {
        return Err(MoptError::config(
            "GFN-FF: structure does not match the topology",
        ));
    }
    let el = &t.setup.el;
    let f = Frame { x, lattice };
    let mut g = vec![[0.0; 3]; n];
    let mut sigma: Sigma = [[0.0; 3]; 3];
    // dE / dCN of the dispersion, the electrostatics and the bonds, contracted with the
    // derivatives of the coordination numbers once at the end.
    let mut dedcn = vec![0.0; n];
    let (mut erep, es_energy, q, edisp, cn, hb_translations) = match lattice {
        None => {
            let erep = nonbonded_repulsion(t, x, &mut g);
            let cn = molecular_coordination_numbers(el, x);
            let es = electrostatics(t, x, &cn.cn)?;
            let edisp = dispersion(el, x, &cn.cn, &t.nonbonded.zetac6, &mut g, &mut dedcn)?;
            electrostatic_gradient(t, x, &es, &cn.cn, &mut g, &mut dedcn);
            (erep, es.energy, es.q, edisp, cn, Translations::molecule())
        }
        Some(lattice) => {
            let tr = Translations::new(lattice, k::REPTHR.sqrt())?;
            let erep = periodic_eg::nonbonded_repulsion(t, x, lattice, &tr, &mut g, &mut sigma);
            let trv = translation_vectors(lattice, &tr);
            let cn = coordination_numbers(el, x, &trv, LATTICE_SUM_CUTOFF);
            let es = periodic_eg::electrostatics(t, x, lattice, &cn)?;
            let edisp = periodic_eg::dispersion(
                el,
                x,
                &trv,
                &cn,
                &t.nonbonded.zetac6,
                &mut g,
                &mut sigma,
                &mut dedcn,
            )?;
            periodic_eg::electrostatic_gradient(t, x, &es, &cn, &mut g, &mut sigma, &mut dedcn);
            let hb = Translations::new(lattice, k::HBTHR2.sqrt())?;
            (erep, es.energy, es.q, edisp, cn, hb)
        }
    };
    let ebond = bonds(t, &f, &cn, &mut g, &mut sigma, &mut dedcn);
    cn.add_chain(&dedcn, &mut g, &mut sigma);
    erep += bonded_repulsion(t, &f, &mut g, &mut sigma);
    let eangl = bends(&t.angles, el, &f, &mut g, &mut sigma);
    let mut sink = GradientSink {
        g: &mut g,
        sigma: &mut sigma,
    };
    let etors = torsions(&t.torsions, el, &f, &mut sink);
    let ebatm = batm(&t.setup, &t.nonbonded, &f, &mut sink);
    let (ehb, exb) = hb_xb(
        &t.setup,
        &t.nonbonded,
        &f,
        &hb_translations,
        hb_cutoff,
        &mut sink,
    );
    let total = es_energy + edisp + erep + ebond + eangl + etors + ehb + exb + ebatm;
    let energy = GfnffEnergy {
        total,
        bond: ebond,
        angle: eangl,
        torsion: etors,
        repulsion: erep,
        electrostatic: es_energy,
        dispersion: edisp,
        hbond: ehb,
        xbond: exb,
        batm: ebatm,
    };
    Ok((energy, q, g, lattice.map(|_| sigma)))
}
