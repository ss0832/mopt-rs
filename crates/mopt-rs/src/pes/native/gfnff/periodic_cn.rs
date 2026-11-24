//! Coordination numbers of the periodic GFN-FF: xtb's lattice sum of the error-function count
//! with its log cap, and the derivatives with respect to the positions and the strain.
//!
//! As xtb 6.7.1 computes them (`getCoordinationNumberLP`, `ncoordLatP`,
//! `cutCoordinationNumber` in `src/gfnff/gfnff_eg.f90`; the lattice points of
//! `src/type/latticepoint.f90`; LGPL-3.0-or-later): for every pair `j <= i` and every
//! translation `T` of a set, `r = |x_i - x_j - T|` within the cutoff counts
//! `0.5 (1 + erf(-7.5 (r - R0) / R0))`, `R0` the sum of the D3 covalent radii, to `CN_i` and,
//! unless `j = i` and `T = 0`, to `CN_j`, so that a pair of an atom and its own image counts
//! twice (xtb's loop); the CN is then capped, `ln(1 + e^4.4) - ln(1 + e^(4.4 - CN))`. The setup
//! uses this CN over all lattice points within `max(60, |a1 + a2 - a3| + 1)` Bohr and a cutoff
//! of 40 Bohr for the metallic character; the energy uses the kept translations of
//! `sqrt(repthr)` and a cutoff of 60 Bohr.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::constants as k;
use super::elements::Atom;
use super::lattice::Lattice;
use super::neighbors::{dist, V3};
use crate::error::MoptResult;

/// Steepness of the error-function count.
const KCN: f64 = 7.5;

/// The coordination numbers and their derivatives, stored by pairs: a pair `(i, j, v)` of
/// atoms `i != j` (or of an atom and an image of another) whose count has the derivative `v`
/// with respect to `x_i` (and `-v` with respect to `x_j`) adds `scale_i v` to `dCN_i / dx_i`,
/// `-scale_i v` to `dCN_i / dx_j`, `-scale_j v` to `dCN_j / dx_j` and `scale_j v` to
/// `dCN_j / dx_i`, `scale` the derivative of the log cap. Pairs whose count has a zero
/// derivative, and the pairs of an atom with its own images (whose derivatives cancel), are
/// left out, so that the storage grows with the number of pairs within reach of the counting
/// function rather than with `N^2` (new; xtb keeps the dense `dlogCN(3, N, N)`).
#[derive(Debug, Clone)]
pub struct CoordinationNumbers {
    /// The capped CN of every atom.
    pub cn: Vec<f64>,
    /// The pairs `(i, j, d count / d x_i)`.
    pub pairs: Vec<(usize, usize, V3)>,
    /// `d CN_i / d CN_raw_i` of the log cap.
    pub scale: Vec<f64>,
    /// `strain[i] = d CN_i / d strain` (3 x 3, `[a][b] = sum r_a dCN/dr_b`).
    pub strain: Vec<[[f64; 3]; 3]>,
}

impl CoordinationNumbers {
    /// Add `sum_i dE/dCN_i dCN_i/dx` to `g` and `sum_i dE/dCN_i dCN_i/d strain` to `sigma`.
    pub fn add_chain(&self, dedcn: &[f64], g: &mut [V3], sigma: &mut [[f64; 3]; 3]) {
        let w: Vec<f64> = dedcn.iter().zip(&self.scale).map(|(d, s)| d * s).collect();
        for &(i, j, v) in &self.pairs {
            let s = w[i] + w[j];
            for c in 0..3 {
                g[i][c] += v[c] * s;
                g[j][c] -= v[c] * s;
            }
        }
        for (st, &de) in self.strain.iter().zip(dedcn) {
            for a in 0..3 {
                for b in 0..3 {
                    sigma[a][b] += st[a][b] * de;
                }
            }
        }
    }
}

/// The lattice points `L n` within `cutoff` (Bohr), the origin first, by distance.
pub fn lattice_points(lattice: &Lattice, cutoff: f64) -> Vec<V3> {
    let [a, b, c] = lattice.a;
    let cross = |p: V3, q: V3| {
        [
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        ]
    };
    let norm = |p: V3| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let volume = {
        let bc = cross(b, c);
        (a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]).abs()
    };
    // The heights of the cell along the normals bound the coefficients.
    let range = |p: V3, q: V3| (cutoff * norm(cross(p, q)) / volume).ceil() as i32;
    let (r0, r1, r2) = (range(b, c), range(c, a), range(a, b));
    let mut points: Vec<(f64, V3)> = Vec::new();
    for i in -r0..=r0 {
        for j in -r1..=r1 {
            for kk in -r2..=r2 {
                let t = lattice.translation([i, j, kk]);
                let d2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
                if d2 <= cutoff * cutoff {
                    points.push((d2, t));
                }
            }
        }
    }
    points.sort_by(|p, q| p.0.total_cmp(&q.0));
    points.into_iter().map(|(_, t)| t).collect()
}

/// The capped CN over the translations `trans` (the first the origin) within `cutoff`, with
/// its derivatives by pairs.
pub fn coordination_numbers(
    el: &[Atom],
    x: &[V3],
    trans: &[V3],
    cutoff: f64,
) -> CoordinationNumbers {
    let n = x.len();
    let mut cn = vec![0.0; n];
    let mut pairs = Vec::new();
    let mut strain = vec![[[0.0; 3]; 3]; n];
    let cutoff2 = cutoff * cutoff;
    let sqrtpi = PI.sqrt();
    for i in 0..n {
        for j in 0..=i {
            let rc = el[i].rcov + el[j].rcov;
            for (t, tv) in trans.iter().enumerate() {
                let rij = [0, 1, 2].map(|c| x[i][c] - (x[j][c] + tv[c]));
                let r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let r1 = r2.sqrt();
                let count = 0.5 * (1.0 + libm::erf(-KCN * (r1 - rc) / rc));
                let dc = -KCN / sqrtpi / rc * (-(KCN * KCN) * (r1 - rc).powi(2) / (rc * rc)).exp();
                let countd = rij.map(|v| dc * v / r1);
                let distinct = i != j || t != 0;
                cn[i] += count;
                if distinct {
                    cn[j] += count;
                }
                if i != j && countd != [0.0; 3] {
                    pairs.push((i, j, countd));
                }
                for a in 0..3 {
                    for b in 0..3 {
                        let s = countd[b] * rij[a];
                        strain[i][a][b] += s;
                        if distinct {
                            strain[j][a][b] += s;
                        }
                    }
                }
            }
        }
    }
    // The log cap.
    let cut = k::PARAM_CNMAX;
    let mut scale = vec![0.0; n];
    for i in 0..n {
        let f = cut.exp() / (cut.exp() + cn[i].exp());
        scale[i] = f;
        for row in strain[i].iter_mut() {
            *row = row.map(|c| c * f);
        }
        cn[i] = (1.0 + cut.exp()).ln() - (1.0 + (cut - cn[i]).exp()).ln();
    }
    CoordinationNumbers {
        cn,
        pairs,
        scale,
        strain,
    }
}

/// The log-capped erf CN of a molecule (`gfnff_dlogcoord`, with xtb's truncated `sqrt(pi)`)
/// with its derivatives by pairs (the same values as
/// [`super::neighbors::log_cn_derivatives`]; no strain).
pub fn molecular_coordination_numbers(el: &[Atom], x: &[V3]) -> CoordinationNumbers {
    const KN: f64 = -7.5;
    const SQRTPI: f64 = 1.772_453_850_91;
    let n = x.len();
    let thr = k::CNTHR.sqrt();
    let mut cn = vec![0.0; n];
    let mut pairs = Vec::new();
    // The pairs in fixed chunks of atoms (`pes::native::parallel`), each with its own sums.
    let chunk = |rows: std::ops::Range<usize>| {
        let mut cn = vec![0.0; n];
        let mut pairs = Vec::new();
        for i in rows {
            for j in 0..i {
                let r = dist(x[i], x[j]);
                if r > thr {
                    continue;
                }
                let r0 = el[i].rcov + el[j].rcov;
                let dr = (r - r0) / r0;
                let e = 0.5 * (1.0 + libm::erf(KN * dr));
                cn[i] += e;
                cn[j] += e;
                let der = KN / SQRTPI * (-(KN * KN) * dr * dr).exp() / r0;
                let v = [0, 1, 2].map(|c| der * (x[i][c] - x[j][c]) / r);
                if v != [0.0; 3] {
                    pairs.push((i, j, v));
                }
            }
        }
        (cn, pairs)
    };
    crate::pes::native::parallel::for_row_chunks(n, chunk, |(c, p)| {
        for (a, b) in cn.iter_mut().zip(&c) {
            *a += b;
        }
        pairs.extend(p);
    });
    let cnmax = k::PARAM_CNMAX;
    let scale = cn
        .iter()
        .map(|&c| cnmax.exp() / (cnmax.exp() + c.exp()))
        .collect();
    let cn = cn
        .iter()
        .map(|&c| (1.0 + cnmax.exp()).ln() - (1.0 + (cnmax - c).exp()).ln())
        .collect();
    CoordinationNumbers {
        cn,
        pairs,
        scale,
        strain: Vec::new(),
    }
}

/// The capped CN over `trans` within `cutoff` with the dense derivatives
/// `d[i][j] = d CN_i / d x_j`, accumulated as xtb does (for the metallic character of the
/// setup, whose thresholds must see the same values).
fn dense_coordination_numbers(
    el: &[Atom],
    x: &[V3],
    trans: &[V3],
    cutoff: f64,
) -> (Vec<f64>, Vec<Vec<V3>>) {
    let n = x.len();
    let mut cn = vec![0.0; n];
    let mut d = vec![vec![[0.0; 3]; n]; n];
    let cutoff2 = cutoff * cutoff;
    let sqrtpi = PI.sqrt();
    for i in 0..n {
        for j in 0..=i {
            let rc = el[i].rcov + el[j].rcov;
            for (t, tv) in trans.iter().enumerate() {
                let rij = [0, 1, 2].map(|c| x[i][c] - (x[j][c] + tv[c]));
                let r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let r1 = r2.sqrt();
                let count = 0.5 * (1.0 + libm::erf(-KCN * (r1 - rc) / rc));
                let dc = -KCN / sqrtpi / rc * (-(KCN * KCN) * (r1 - rc).powi(2) / (rc * rc)).exp();
                let countd = rij.map(|v| dc * v / r1);
                let distinct = i != j || t != 0;
                cn[i] += count;
                if distinct {
                    cn[j] += count;
                }
                for c in 0..3 {
                    d[i][i][c] += countd[c];
                    d[j][j][c] -= countd[c];
                    d[j][i][c] += countd[c];
                    d[i][j][c] -= countd[c];
                }
            }
        }
    }
    // The log cap.
    let cut = k::PARAM_CNMAX;
    for i in 0..n {
        let f = cut.exp() / (cut.exp() + cn[i].exp());
        for v in d[i].iter_mut() {
            *v = v.map(|c| c * f);
        }
        cn[i] = (1.0 + cut.exp()).ln() - (1.0 + (cut - cn[i]).exp()).ln();
    }
    (cn, d)
}

/// The metallic character of the setup (`gfnff_ini`): the CN over the lattice points within
/// `max(60, |a1 + a2 - a3| + 1)` Bohr with a cutoff of 40 Bohr, and
/// `exp(-0.005 EN^8) sum_j |dCN_i / dx_j| / (CN_i + 1)`.
pub fn setup_mchar(el: &[Atom], x: &[V3], lattice: &Lattice) -> MoptResult<Vec<f64>> {
    let [a, b, c] = lattice.a;
    let v = [0, 1, 2].map(|k| a[k] + b[k] - c[k]);
    let reach = ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() + 1.0).max(60.0);
    let trans = lattice_points(lattice, reach);
    let (cn, d) = dense_coordination_numbers(el, x, &trans, 40.0);
    Ok((0..x.len())
        .map(|i| {
            let s: f64 = d[i]
                .iter()
                .map(|g| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt())
                .sum();
            (-0.005 * el[i].raw.en.powi(8)).exp() * s / (cn[i] + 1.0)
        })
        .collect())
}
