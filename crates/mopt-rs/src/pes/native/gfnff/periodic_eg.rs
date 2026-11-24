//! The terms of the periodic GFN-FF that differ from the molecule: the non-bonded repulsion
//! and the dispersion as lattice sums, and the EEQ electrostatics by Ewald summation, each
//! with its gradient and its strain derivative.
//!
//! As xtb 6.7.1 computes them (`gfnff_eg`, `goed_pbc_gfnff`, `es_grad_sigma` in
//! `src/gfnff/gfnff_eg.f90`; `d3_gradient_latp`, `disp_gradient_latp_inter/intra` in
//! `src/gfnff/gdisp0.f90`; LGPL-3.0-or-later):
//! - repulsion: every pair `j <= i` and every translation `T` of the set of `sqrt(repthr)`,
//!   `r = |x_i + T - x_j|` below `sqrt(repthr)`, without the bonded pairs inside the 27 cells;
//!   an atom and its own image are counted for `T` and `-T`, twice the lattice sum, as in xtb;
//! - dispersion: the same pairs within 60 Bohr, with the D4 weights of the periodic CN, the
//!   self-image pairs doubled in the energy and the strain (and zero in the gradient);
//! - electrostatics: `E = q^T A q / 2 - q.X + sum_f lambda_f Q_f` with the Ewald matrix
//!   ([`super::ewald`]), the total charge on the first constraint row and the fragment charges
//!   on the others, `X_i = chi_i + cnf_i sqrt(logCN_i)`; the charges are variational, so the
//!   gradient and the strain derivative are those of the matrix at fixed charges and of the
//!   CN in `X`.
//!
//! The strain derivative is `sigma_ab = sum r_a dE/dr_b` over the pair vectors (xtb's
//! `sigma`, not divided by the volume).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::constants as k;
use super::eg_disp::{table_index, weights};
use super::elements::Atom;
use super::ewald::{self, EwaldSetup, Images};
use super::lattice::{is_central, Lattice, Translations};
use super::neighbors::V3;
use super::periodic_cn::CoordinationNumbers;
use super::GfnffTopology;
use crate::error::{MoptError, MoptResult};
use crate::parameters::gfnff::GFNFF_DISP_NREF;

/// The strain derivative of the energy (3 x 3, Hartree).
pub type Sigma = [[f64; 3]; 3];

/// Cutoff of the periodic CN and dispersion (Bohr).
pub const LATTICE_SUM_CUTOFF: f64 = 60.0;

/// The vectors `L n` of the cells of `tr`.
pub fn translation_vectors(lattice: &Lattice, tr: &Translations) -> Vec<V3> {
    tr.cells.iter().map(|&n| lattice.translation(n)).collect()
}

/// Non-bonded repulsion over the translations of `tr`.
pub fn nonbonded_repulsion(
    t: &GfnffTopology,
    x: &[V3],
    lattice: &Lattice,
    tr: &Translations,
    g: &mut [V3],
    sigma: &mut Sigma,
) -> f64 {
    let s = &t.setup;
    let el = &s.el;
    let mut e = 0.0;
    for i in 0..x.len() {
        for j in 0..=i {
            for &cell in &tr.cells {
                let tv = lattice.translation(cell);
                let rv = [0, 1, 2].map(|c| x[i][c] + tv[c] - x[j][c]);
                let r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
                if !(1.0e-8..=k::REPTHR).contains(&r2) {
                    continue;
                }
                // bpair(i, j, T): bonds between x_j and the image of i.
                let bp = if is_central(cell) {
                    s.bpair.get(i, j, cell)
                } else {
                    None
                };
                if bp == Some(1) {
                    continue;
                }
                let alpha = t.nonbonded.alphanb(s, i, j, bp);
                let rab = r2.sqrt();
                let t16 = r2.powf(0.75);
                let t19 = t16 * t16;
                let t8 = t16 * alpha;
                let t26 = (-t8).exp() * el[i].raw.repz * el[j].raw.repz * k::REPSCALN;
                e += t26 / rab;
                let t27 = t26 * (1.5 * t8 + 1.0) / t19;
                for c in 0..3 {
                    let r3 = rv[c] * t27;
                    g[i][c] -= r3;
                    g[j][c] += r3;
                    for b in 0..3 {
                        sigma[b][c] -= r3 * rv[b];
                    }
                }
            }
        }
    }
    e
}

/// Dispersion over the translation vectors `trans` (the first the origin) within 60 Bohr,
/// with the periodic CN `cn`; `dE / dCN` is added to `dedcn` (for the chain rule of the
/// caller).
#[allow(clippy::too_many_arguments)]
pub fn dispersion(
    el: &[Atom],
    x: &[V3],
    trans: &[V3],
    cn: &CoordinationNumbers,
    zetac6: &[Vec<f64>],
    g: &mut [V3],
    sigma: &mut Sigma,
    dedcn: &mut [f64],
) -> MoptResult<f64> {
    const DISPSCALE: f64 = 1.0;
    let n = x.len();
    let tab: Vec<usize> = el.iter().map(table_index).collect::<MoptResult<_>>()?;
    let w: Vec<([f64; 7], [f64; 7])> = (0..n).map(|i| weights(tab[i], cn.cn[i])).collect();
    let c6tab = super::eg_disp::ReferenceC6::new(&tab);
    let cutoff2 = LATTICE_SUM_CUTOFF * LATTICE_SUM_CUTOFF;
    let mut energy = 0.0;
    for i in 0..n {
        for j in 0..=i {
            let (ti, tj) = (tab[i], tab[j]);
            let (mut c6, mut dc6i, mut dc6j) = (0.0, 0.0, 0.0);
            for ri in 0..GFNFF_DISP_NREF[ti] {
                for rj in 0..GFNFF_DISP_NREF[tj] {
                    let refc6 = c6tab.get(ti, tj)[ri][rj];
                    c6 += w[i].0[ri] * w[j].0[rj] * refc6;
                    dc6i += w[i].1[ri] * w[j].0[rj] * refc6;
                    dc6j += w[i].0[ri] * w[j].1[rj] * refc6;
                }
            }
            let r4r2ij = 3.0 * el[i].sqrt_zr4r2 * el[j].sqrt_zr4r2;
            let r0 =
                (k::D3A1 * (el[i].sqrt_zr4r2 * el[j].sqrt_zr4r2 * 3.0).sqrt() + k::D3A2).powi(2);
            let zeta = zetac6[i][j] * DISPSCALE;
            for (it, tv) in trans.iter().enumerate() {
                let rij = [0, 1, 2].map(|c| x[i][c] - x[j][c] - tv[c]);
                let r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let t6 = 1.0 / (r2.powi(3) + r0.powi(3));
                let t8 = 1.0 / (r2.powi(4) + r0.powi(4));
                let d6 = -6.0 * r2.powi(2) * t6 * t6;
                let d8 = -8.0 * r2.powi(3) * t8 * t8;
                let disp = (t6 + 2.0 * r4r2ij * t8) * zeta;
                let ddisp = (d6 + 2.0 * r4r2ij * d8) * zeta;
                let de = -c6 * disp * 0.5;
                let dg = rij.map(|v| -c6 * ddisp * v);
                energy += de;
                dedcn[i] -= dc6i * disp;
                for a in 0..3 {
                    for b in 0..3 {
                        sigma[a][b] += 0.5 * dg[b] * rij[a];
                    }
                }
                if i != j || it != 0 {
                    energy += de;
                    dedcn[j] -= dc6j * disp;
                    for c in 0..3 {
                        g[i][c] += dg[c];
                        g[j][c] -= dg[c];
                    }
                    for a in 0..3 {
                        for b in 0..3 {
                            sigma[a][b] += 0.5 * dg[b] * rij[a];
                        }
                    }
                }
            }
        }
    }
    Ok(energy)
}

/// The EEQ charges and the electrostatic energy in the cell, with what the gradient needs.
pub struct PeriodicElectrostatics {
    pub q: Vec<f64>,
    pub energy: f64,
    setup: EwaldSetup,
    images: Images,
}

/// EEQ charges and energy (`goed_pbc_gfnff`).
pub fn electrostatics(
    t: &GfnffTopology,
    x: &[V3],
    lattice: &Lattice,
    cn: &CoordinationNumbers,
) -> MoptResult<PeriodicElectrostatics> {
    let n = x.len();
    let system = eeq_system(t, x, lattice, &cn.cn);
    let m = system.matrix.nrows();
    let xs = super::eg::solve_bordered(&system.matrix, &system.rhs, n)
        .ok_or_else(|| MoptError::calc("GFN-FF: singular periodic EEQ system"))?;
    let q = xs[..n].to_vec();
    let mut energy = 0.0;
    for i in 0..n {
        let mut aq = 0.0;
        for j in 0..n {
            aq += system.matrix[(i, j)] * q[j];
        }
        energy += 0.5 * q[i] * aq - q[i] * system.rhs[i];
    }
    for f in n..m {
        energy += xs[f] * system.rhs[f];
    }
    Ok(PeriodicElectrostatics {
        q,
        energy,
        setup: system.setup,
        images: system.images,
    })
}

/// The EEQ system of a cell (`goed_pbc_gfnff`).
pub struct PeriodicEeqSystem {
    /// The Ewald matrix of the atoms bordered by the charge constraints.
    pub matrix: Mat<f64>,
    /// The right-hand side: `X` of the atoms, then the total and fragment charges.
    pub rhs: Col<f64>,
    /// The sums of the Ewald matrix.
    pub setup: EwaldSetup,
    /// The Wigner-Seitz images of the pairs.
    pub images: Images,
}

/// The EEQ system of the cell for the coordination numbers `cn`.
pub fn eeq_system(t: &GfnffTopology, x: &[V3], lattice: &Lattice, cn: &[f64]) -> PeriodicEeqSystem {
    let s = &t.setup;
    let n = x.len();
    let nfrag = s.fraglist.iter().copied().max().unwrap_or(0).max(1);
    let mean_alp = s.alpeeq.iter().sum::<f64>() / n as f64;
    let setup = EwaldSetup::new(lattice.a, mean_alp);
    let images = Images::new(x, &lattice.a);
    let amat = ewald::matrix(x, &setup, &images, &s.alpeeq, &s.gameeq);
    let m = n + nfrag;
    let mut a = Mat::<f64>::zeros(m, m);
    let mut rhs = Col::<f64>::zeros(m);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = amat[(i, j)];
        }
        rhs[i] = s.chieeq[i] + s.el[i].raw.cnf * cn[i].sqrt();
        // The first constraint row holds every atom.
        a[(n, i)] = 1.0;
        a[(i, n)] = 1.0;
    }
    for f in 0..nfrag {
        rhs[n + f] = s.qfrag.get(f).copied().unwrap_or(0.0);
        if f == 0 {
            continue;
        }
        for j in 0..n {
            if s.fraglist[j] == f + 1 {
                a[(n + f, j)] = 1.0;
                a[(j, n + f)] = 1.0;
            }
        }
    }
    PeriodicEeqSystem {
        matrix: a,
        rhs,
        setup,
        images,
    }
}

/// Gradient and strain derivative of the electrostatic energy at fixed charges; the `dE / dCN`
/// of `X` (`es_grad_sigma` and the `qtmp` of xtb) is added to `dedcn` (for the chain rule of
/// the caller).
pub fn electrostatic_gradient(
    t: &GfnffTopology,
    x: &[V3],
    es: &PeriodicElectrostatics,
    cn: &CoordinationNumbers,
    g: &mut [V3],
    sigma: &mut Sigma,
    dedcn: &mut [f64],
) {
    let s = &t.setup;
    ewald::derivatives(x, &es.q, &es.setup, &es.images, &s.alpeeq, g, sigma);
    for (i, de) in dedcn.iter_mut().enumerate().take(x.len()) {
        *de -= es.q[i] * s.el[i].raw.cnf / (2.0 * cn.cn[i].sqrt() + 1.0e-16);
    }
}
