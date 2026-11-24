//! GFN-FF dispersion, ported from xtb's `d3_gradient`, `weight_references_d4`, and
//! `get_atomic_c6_d4` (`src/gfnff/gdisp0.f90`; LGPL-3.0-or-later) with the reference C6 of
//! xtb's `newD3Model` (`src/disp/dftd4.F90`), precomputed for the elements by
//! `tools/params/gen_gfnff.py` into [`crate::parameters::gfnff`].
//!
//! The coordination number is GFN-FF's log-capped erf CN, the Gaussian weighting factor 4,
//! the pair damping rational with `R0^2 = (a1 sqrt(3 Q_i Q_j) + a2)^2`, `s6 = 1`, `s8 = 2`, and
//! each pair is scaled by the product of the topology-charge `zeta` factors.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::constants as k;
use super::elements::Atom;
use super::neighbors::V3;
use crate::error::{MoptError, MoptResult};
use crate::parameters::gfnff::{
    GFNFF_DISP_ALPHA, GFNFF_DISP_NREF, GFNFF_DISP_REFCN, GFNFF_DISP_TRAPZD, GFNFF_ELEMENTS,
};

/// Weighting factor of the reference CNs.
const WF: f64 = 4.0;
/// xtb's `dispscale` default.
const DISPSCALE: f64 = 1.0;

/// Index of an element in the generated tables (`Z - 1`).
pub(super) fn table_index(a: &Atom) -> MoptResult<usize> {
    let z = a.z();
    usize::from(z)
        .checked_sub(1)
        .filter(|&t| GFNFF_ELEMENTS.get(t).is_some_and(|e| e.z == z))
        .ok_or_else(|| {
            MoptError::unsupported(format!(
                "GFN-FF dispersion has D4 references for {} only, not Z = {z}",
                super::elements::ELEMENT_RANGE
            ))
        })
}

/// The reference C6 coefficients (Hartree Bohr^6) of the pairs of the elements of a structure,
/// from the reference polarizabilities by xtb's Casimir-Polder integration (`trapzd`):
/// `c6[ri][rj] = 3 / pi sum_k alpha_i[ri][k] alpha_j[rj][k] w_k`.
pub(super) struct ReferenceC6 {
    /// The slot of each element of the tables (`usize::MAX` for those not present).
    slot: Vec<usize>,
    /// The number of elements present.
    present: usize,
    /// The blocks of the pairs of present elements, row by row.
    blocks: Vec<[[f64; 7]; 7]>,
}

impl ReferenceC6 {
    /// The pairs of the elements of the table indices `types`.
    pub(super) fn new(types: &[usize]) -> Self {
        let mut slot = vec![usize::MAX; GFNFF_ELEMENTS.len()];
        let mut order = Vec::new();
        for &t in types {
            if slot[t] == usize::MAX {
                slot[t] = order.len();
                order.push(t);
            }
        }
        let thopi = 3.0 / std::f64::consts::PI;
        let mut blocks = Vec::with_capacity(order.len() * order.len());
        for &a in &order {
            for &b in &order {
                let mut block = [[0.0; 7]; 7];
                for (ra, row) in block.iter_mut().enumerate().take(GFNFF_DISP_NREF[a]) {
                    for (rb, value) in row.iter_mut().enumerate().take(GFNFF_DISP_NREF[b]) {
                        let mut sum = 0.0;
                        for k in 0..23 {
                            sum += GFNFF_DISP_ALPHA[a][ra][k]
                                * GFNFF_DISP_ALPHA[b][rb][k]
                                * GFNFF_DISP_TRAPZD[k];
                        }
                        *value = thopi * sum;
                    }
                }
                blocks.push(block);
            }
        }
        Self {
            slot,
            present: order.len(),
            blocks,
        }
    }

    /// The block of the elements of the table indices `a` and `b` (both present).
    pub(super) fn get(&self, a: usize, b: usize) -> &[[f64; 7]; 7] {
        &self.blocks[self.slot[a] * self.present + self.slot[b]]
    }
}

/// xtb's `weight_cn`.
fn weight_cn(cn: f64, cnref: f64) -> f64 {
    let val = -WF * (cn - cnref).powi(2);
    if val < -200.0 {
        0.0
    } else {
        val.exp()
    }
}

/// Reference weights and their CN derivatives of one atom (`weight_references_d4`).
pub(super) fn weights(t: usize, cn: f64) -> ([f64; 7], [f64; 7]) {
    let nref = GFNFF_DISP_NREF[t];
    let refcn = &GFNFF_DISP_REFCN[t];
    let mut norm = 0.0;
    let mut dnorm = 0.0;
    for &rc in refcn.iter().take(nref) {
        let gw = weight_cn(cn, rc);
        norm += gw;
        dnorm += 2.0 * WF * (rc - cn) * gw;
    }
    let norm = if norm > 1.0e-80 { 1.0 / norm } else { 0.0 };
    let maxcn = refcn[..nref]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut gw = [0.0; 7];
    let mut dgw = [0.0; 7];
    for r in 0..nref {
        let expw = weight_cn(cn, refcn[r]);
        let expd = 2.0 * WF * (refcn[r] - cn) * expw;
        let mut gwk = expw * norm;
        if norm == 0.0 {
            gwk = if (maxcn - refcn[r]).abs() < 1.0e-12 {
                1.0
            } else {
                0.0
            };
        }
        gw[r] = gwk;
        dgw[r] = expd * norm - expw * dnorm * norm * norm;
    }
    (gw, dgw)
}

/// Dispersion energy; adds the gradient at fixed coordination numbers into `g` and
/// `dE / dCN` into `dedcn` (for the chain rule of the caller).
pub fn dispersion(
    el: &[Atom],
    x: &[V3],
    cn: &[f64],
    zetac6: &[Vec<f64>],
    g: &mut [V3],
    dedcn: &mut [f64],
) -> MoptResult<f64> {
    let n = x.len();
    let t: Vec<usize> = el.iter().map(table_index).collect::<MoptResult<_>>()?;
    let w: Vec<([f64; 7], [f64; 7])> = (0..n).map(|i| weights(t[i], cn[i])).collect();
    let mut energy = 0.0;
    let c6tab = ReferenceC6::new(&t);
    // The pairs in fixed chunks of atoms (`pes::native::parallel`), each with its own sums.
    let chunk = |rows: std::ops::Range<usize>| {
        let mut energy = 0.0;
        let mut g = vec![[0.0; 3]; n];
        let mut dedcn = vec![0.0; n];
        for i in rows {
            for j in 0..i {
                let rij = [0, 1, 2].map(|c| x[i][c] - x[j][c]);
                let r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                if r2 >= k::DISPTHR {
                    continue;
                }
                let (ti, tj) = (t[i], t[j]);
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
                let r0 = (k::D3A1 * (el[i].sqrt_zr4r2 * el[j].sqrt_zr4r2 * 3.0).sqrt() + k::D3A2)
                    .powi(2);
                let t6 = 1.0 / (r2.powi(3) + r0.powi(3));
                let t8 = 1.0 / (r2.powi(4) + r0.powi(4));
                let d6 = -6.0 * r2.powi(2) * t6 * t6;
                let d8 = -8.0 * r2.powi(3) * t8 * t8;
                let zeta = zetac6[i][j] * DISPSCALE;
                let disp = (t6 + 2.0 * r4r2ij * t8) * zeta;
                let ddisp = (d6 + 2.0 * r4r2ij * d8) * zeta;
                energy += -c6 * disp;
                for c in 0..3 {
                    let dg = -c6 * ddisp * rij[c];
                    g[i][c] += dg;
                    g[j][c] -= dg;
                }
                dedcn[i] -= dc6i * disp;
                dedcn[j] -= dc6j * disp;
            }
        }
        (energy, g, dedcn)
    };
    crate::pes::native::parallel::for_row_chunks(n, chunk, |(e, gc, dc)| {
        energy += e;
        crate::pes::native::parallel::add_into(g, &gc);
        for (a, b) in dedcn.iter_mut().zip(&dc) {
            *a += b;
        }
    });
    Ok(energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Element;

    #[test]
    fn every_element_has_d4_references() {
        for z in 1..=86u8 {
            let a = super::super::elements::atom(Element::from_atomic_number(z).unwrap()).unwrap();
            let t = table_index(&a).unwrap();
            assert_eq!(t, usize::from(z) - 1);
            let nref = GFNFF_DISP_NREF[t];
            assert!((1..=7).contains(&nref), "Z = {z}: {nref}");
            let c6 = ReferenceC6::new(&[t]);
            // The C6 of the element with itself at its first reference is positive.
            assert!(c6.get(t, t)[0][0] > 0.0, "Z = {z}");
            let (w, _) = weights(t, 2.0);
            let sum: f64 = w.iter().take(nref).sum();
            assert!((sum - 1.0).abs() < 1e-12, "Z = {z}: {sum}");
        }
    }
}
