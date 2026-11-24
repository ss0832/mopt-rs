//! The EEQ electrostatics of GFN-FF in a periodic cell: xtb's Ewald matrix, its splitting
//! parameter, the Wigner-Seitz images of the pairs, and the derivatives of the matrix with
//! respect to the positions and the strain.
//!
//! As xtb 6.7.1 computes it (`goed_pbc_gfnff`, `get_amat_3d`, `get_damat_3d`,
//! `es_grad_sigma`, `get_cf` in `src/gfnff/gfnff_eg.f90`, `generate_wsc` in
//! `src/generate_wsc.f90`; LGPL-3.0-or-later), three-dimensional cells only:
//! - fixed sums over the cells `L n`, `n` in `{-2..2}^3` (125 vectors with 0), and over the
//!   reciprocal vectors `2 pi L^-T m`, `m` in `{-2..2}^3` without 0 (124 vectors), whatever the
//!   size of the cell;
//! - the splitting parameter `alpha` from a golden-section search with the ratio
//!   `(1 + sqrt(2)) / 2` on `[1e-8, 2]` (tolerance 1e-4) that balances the leading real and
//!   reciprocal terms;
//! - each pair `(i, j)` over its Wigner-Seitz images: the translations `t` in `{-1, 0, 1}^3`
//!   that put `x_j + L t` within 0.01 Bohr of the closest to `x_i`, each with the weight
//!   `1 / count`;
//! - `A_ij = sum_t w [sum_R phi_ij(|v + R|) + (4 pi / V) sum_G cos(G.v) exp(-G^2/4a^2) / G^2]`,
//!   `v = x_i - x_j - L t`, `phi_ij(r) = (erf(gamma_ij r) - erf(alpha r)) / r`;
//!   `A_ii = sum_(R != 0) phi_ii(|R|) + (4 pi / V) sum_G exp(-G^2/4a^2) / G^2 + gam_i +
//!   sqrt(2/pi) / sqrt(alp_i) - 2 alpha / sqrt(pi)`; no `G = 0` (neutralizing background) term.
//!
//! The derivatives hold `alpha` and the images fixed (both are piecewise constant).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use super::neighbors::V3;
use crate::math::ad::{HyperDual, Scalar};

/// `(1 + sqrt(2)) / 2`, xtb's ratio of the golden-section search.
const GOLDEN: f64 = 1.207_106_781_186_547_5;
/// Bounds and tolerance of the search for `alpha`.
const ALPHA_LOW: f64 = 1.0e-8;
const ALPHA_HIGH: f64 = 2.0;
const ALPHA_TOLERANCE: f64 = 1.0e-4;
/// Images within this much (Bohr) of the closest count as minimum images.
const IMAGE_TOLERANCE: f64 = 0.01;
/// Distances below this (Bohr) are left out of the real-space sums.
const SMALLEST_DISTANCE: f64 = 1.0e-9;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// `n0 a0 + n1 a1 + n2 a2` of the lattice vectors `a`.
fn combine(a: &[V3; 3], n: [f64; 3]) -> V3 {
    [0, 1, 2].map(|c| n[0] * a[0][c] + n[1] * a[1][c] + n[2] * a[2][c])
}

/// The cells and reciprocal vectors of the sums, and the splitting parameter.
#[derive(Debug, Clone)]
pub struct EwaldSetup {
    /// The lattice vectors (Bohr).
    pub lattice: [V3; 3],
    /// The volume of the cell (Bohr^3).
    pub volume: f64,
    /// The cells `L n` of the real-space sum (125, with 0).
    pub real: Vec<V3>,
    /// The `n` of [`Self::real`].
    pub real_cells: Vec<V3>,
    /// The reciprocal vectors `G` (124).
    pub reciprocal: Vec<V3>,
    /// The `m` of [`Self::reciprocal`], `G = 2 pi L^-T m`.
    pub reciprocal_cells: Vec<V3>,
    /// `(4 pi / V) exp(-G^2 / (4 alpha^2)) / G^2` for each `G`.
    pub weight: Vec<f64>,
    /// The splitting parameter `alpha` (1/Bohr).
    pub alpha: f64,
}

impl EwaldSetup {
    /// The sums of the cell `lattice` (lattice vectors, Bohr) for EEQ widths of mean `mean_alp`
    /// (the `alpeeq` of the atoms).
    pub fn new(lattice: [V3; 3], mean_alp: f64) -> Self {
        let volume = dot(lattice[0], cross(lattice[1], lattice[2])).abs();
        let signed = dot(lattice[0], cross(lattice[1], lattice[2]));
        let recip = [
            cross(lattice[1], lattice[2]),
            cross(lattice[2], lattice[0]),
            cross(lattice[0], lattice[1]),
        ]
        .map(|b| b.map(|v| 2.0 * PI * v / signed));
        let mut real = Vec::with_capacity(125);
        let mut real_cells = Vec::with_capacity(125);
        let mut reciprocal = Vec::with_capacity(124);
        let mut reciprocal_cells = Vec::with_capacity(124);
        for i in -2..=2 {
            for j in -2..=2 {
                for k in -2..=2 {
                    let n = [f64::from(i), f64::from(j), f64::from(k)];
                    real.push(combine(&lattice, n));
                    real_cells.push(n);
                    if (i, j, k) != (0, 0, 0) {
                        reciprocal.push(combine(&recip, n));
                        reciprocal_cells.push(n);
                    }
                }
            }
        }
        let alpha = splitting_parameter(&real, &reciprocal, volume, mean_alp);
        let weight = reciprocal
            .iter()
            .map(|g| {
                let g2 = dot(*g, *g);
                4.0 * PI / volume * (-0.25 * g2 / (alpha * alpha)).exp() / g2
            })
            .collect();
        Self {
            lattice,
            volume,
            real,
            real_cells,
            reciprocal,
            reciprocal_cells,
            weight,
            alpha,
        }
    }
}

/// xtb's `get_cf`: the `alpha` in `[1e-8, 2]` at which the leading reciprocal term
/// `4 pi exp(-Gmin^2/4a^2) / (V Gmin^2)` and the leading real term
/// `(erf(gbar Rmin) - erf(a Rmin)) / Rmin` are closest, `gbar = 1/sqrt(2 mean_alp)`.
fn splitting_parameter(real: &[V3], reciprocal: &[V3], volume: f64, mean_alp: f64) -> f64 {
    let gbar = 1.0 / (2.0 * mean_alp).sqrt();
    let gmin = reciprocal
        .iter()
        .map(|g| norm(*g))
        .fold(f64::INFINITY, f64::min);
    let rmin = real
        .iter()
        .map(|r| norm(*r))
        .filter(|&r| r != 0.0)
        .fold(f64::INFINITY, f64::min);
    let gap = |a: f64| {
        let g = 4.0 * PI * (-(gmin * gmin) / (4.0 * a * a)).exp() / (volume * gmin * gmin);
        let r = -libm::erf(a * rmin) / rmin + libm::erf(gbar * rmin) / rmin;
        (g - r).abs()
    };
    let (mut x1, mut x4) = (ALPHA_LOW, ALPHA_HIGH);
    let mut x2 = x4 - (x4 - x1) / GOLDEN;
    let mut x3 = x1 + (x4 - x1) / GOLDEN;
    while x4 - x1 > ALPHA_TOLERANCE {
        if gap(x2) < gap(x3) {
            x4 = x3;
        } else {
            x1 = x2;
        }
        x2 = x4 - (x4 - x1) / GOLDEN;
        x3 = x1 + (x4 - x1) / GOLDEN;
    }
    0.5 * (x1 + x4)
}

/// The Wigner-Seitz images of the pairs: for `i > j`, the translations `L t` (`t` in
/// `{-1, 0, 1}^3`) within [`IMAGE_TOLERANCE`] of the closest image of `x_j` to `x_i`.
#[derive(Debug, Clone)]
pub struct Images {
    /// `shifts[i][j]` for `j < i`.
    pub shifts: Vec<Vec<Vec<V3>>>,
    /// The `t` of [`Self::shifts`].
    pub cells: Vec<Vec<Vec<V3>>>,
}

impl Images {
    /// The images of the pairs of `x` in the cell `lattice`.
    pub fn new(x: &[V3], lattice: &[V3; 3]) -> Self {
        let mut cells = Vec::with_capacity(27);
        let mut index = Vec::with_capacity(27);
        for a in -1..=1 {
            for b in -1..=1 {
                for c in -1..=1 {
                    let t = [f64::from(a), f64::from(b), f64::from(c)];
                    cells.push(combine(lattice, t));
                    index.push(t);
                }
            }
        }
        let chosen: Vec<Vec<Vec<usize>>> = (0..x.len())
            .map(|i| {
                (0..i)
                    .map(|j| {
                        let d: Vec<f64> = cells
                            .iter()
                            .map(|t| norm(sub(x[i], add(x[j], *t))))
                            .collect();
                        let closest = d.iter().copied().fold(f64::INFINITY, f64::min);
                        (0..cells.len())
                            .filter(|&k| (d[k] - closest).abs() < IMAGE_TOLERANCE)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let pick = |table: &[V3]| -> Vec<Vec<Vec<V3>>> {
            chosen
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|ks| ks.iter().map(|&k| table[k]).collect())
                        .collect()
                })
                .collect()
        };
        Self {
            shifts: pick(&cells),
            cells: pick(&index),
        }
    }
}

/// `phi(r) = (erf(gamma r) - erf(alpha r)) / r` and `phi'(r) / r`.
fn phi(gamma: f64, alpha: f64, r: f64) -> (f64, f64) {
    let r2 = r * r;
    let (eg, ea) = (libm::erf(gamma * r), libm::erf(alpha * r));
    let value = (eg - ea) / r;
    let sqrtpi = PI.sqrt();
    let derivative = 2.0 * gamma * (-gamma * gamma * r2).exp() / (sqrtpi * r2)
        - eg / (r2 * r)
        - 2.0 * alpha * (-alpha * alpha * r2).exp() / (sqrtpi * r2)
        + ea / (r2 * r);
    (value, derivative)
}

/// `phi'(r) / r` and `phi''(r)` of [`phi`].
fn phi_second(gamma: f64, alpha: f64, r: f64) -> (f64, f64) {
    let r2 = r * r;
    let sqrtpi = PI.sqrt();
    // f(r) = erf(g r) / r: f' / r and f''.
    let part = |g: f64| {
        let e = libm::erf(g * r);
        let gauss = 2.0 * g * (-g * g * r2).exp() / sqrtpi;
        (
            gauss / r2 - e / (r2 * r),
            -gauss * (2.0 * g * g + 2.0 / r2) + 2.0 * e / (r2 * r),
        )
    };
    let (dg, d2g) = part(gamma);
    let (da, d2a) = part(alpha);
    (dg - da, d2g - d2a)
}

/// The derivatives of the pair elements of the Ewald matrix at fixed `alpha` and images: for
/// every pair `j < i`, `f(i, j, d, k)` with the gradient `d = dA_ij / dx_i` and the Hessian
/// block `k = d^2 A_ij / dx_i dx_i` (`A_ij` depends on `x_i - x_j` only, so
/// `dA_ij / dx_j = -d` and the blocks of `x_j` are `k` and `-k`). The diagonal elements do not
/// depend on the positions.
pub fn pair_derivatives<F>(x: &[V3], setup: &EwaldSetup, images: &Images, alpeeq: &[f64], mut f: F)
where
    F: FnMut(usize, usize, V3, [[f64; 3]; 3]),
{
    for i in 0..x.len() {
        for j in 0..i {
            let gamma = 1.0 / (alpeeq[i] + alpeeq[j]).sqrt();
            let shifts = &images.shifts[i][j];
            let w = 1.0 / shifts.len() as f64;
            let mut d = [0.0; 3];
            let mut k = [[0.0; 3]; 3];
            for t in shifts {
                let v = sub(sub(x[i], x[j]), *t);
                for r in &setup.real {
                    let rv = add(v, *r);
                    let dd = norm(rv);
                    if dd < SMALLEST_DISTANCE {
                        continue;
                    }
                    let (d1r, d2) = phi_second(gamma, setup.alpha, dd);
                    let u = rv.map(|c| c / dd);
                    for a in 0..3 {
                        d[a] += w * d1r * rv[a];
                        for b in 0..3 {
                            let delta = if a == b { 1.0 } else { 0.0 };
                            k[a][b] += w * (d2 * u[a] * u[b] + d1r * (delta - u[a] * u[b]));
                        }
                    }
                }
                for (gv, weight) in setup.reciprocal.iter().zip(&setup.weight) {
                    let phase = dot(*gv, v);
                    let (s, c) = phase.sin_cos();
                    for a in 0..3 {
                        d[a] -= w * weight * s * gv[a];
                        for b in 0..3 {
                            k[a][b] -= w * weight * c * gv[a] * gv[b];
                        }
                    }
                }
            }
            f(i, j, d, k);
        }
    }
}

/// The parts of the real-space sum of the Ewald matrix that involve the lattice vectors, as
/// functions of the positions and the lattice vectors (images `x_j + L n`), at fixed `alpha`
/// and images.
#[derive(Debug, Clone, Default)]
pub struct LatticeMoments {
    /// `d1[k] = dA_ij / d a_k` (the derivative with respect to lattice vector `k`).
    pub d1: [V3; 3],
    /// `k1[k] = d2A_ij / dx_i d a_k`.
    pub k1: [[V3; 3]; 3],
    /// `k2[k][l] = d2A_ij / d a_k d a_l`.
    pub k2: [[[V3; 3]; 3]; 3],
}

/// The real-space [`LatticeMoments`] of every pair `j < i` and of every atom with its own
/// images (`f(i, i, ..)`, the diagonal element).
pub fn real_lattice_moments<F>(
    x: &[V3],
    setup: &EwaldSetup,
    images: &Images,
    alpeeq: &[f64],
    mut f: F,
) where
    F: FnMut(usize, usize, &LatticeMoments),
{
    let accumulate = |m: &mut LatticeMoments, w: f64, dn: V3, rv: V3, gamma: f64| {
        let d = norm(rv);
        if d < SMALLEST_DISTANCE {
            return;
        }
        let (d1r, d2) = phi_second(gamma, setup.alpha, d);
        let u = rv.map(|c| c / d);
        let mut k = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let delta = if a == b { 1.0 } else { 0.0 };
                k[a][b] = w * (d2 * u[a] * u[b] + d1r * (delta - u[a] * u[b]));
            }
        }
        for p in 0..3 {
            if dn[p] == 0.0 {
                continue;
            }
            for a in 0..3 {
                m.d1[p][a] += dn[p] * w * d1r * rv[a];
                for b in 0..3 {
                    m.k1[p][a][b] += dn[p] * k[a][b];
                }
            }
            for q in 0..3 {
                if dn[q] == 0.0 {
                    continue;
                }
                for a in 0..3 {
                    for b in 0..3 {
                        m.k2[p][q][a][b] += dn[p] * dn[q] * k[a][b];
                    }
                }
            }
        }
    };
    for i in 0..x.len() {
        for j in 0..i {
            let gamma = 1.0 / (alpeeq[i] + alpeeq[j]).sqrt();
            let shifts = &images.shifts[i][j];
            let w = 1.0 / shifts.len() as f64;
            let mut m = LatticeMoments::default();
            for (t, tc) in shifts.iter().zip(&images.cells[i][j]) {
                let v = sub(sub(x[i], x[j]), *t);
                for (r, rc) in setup.real.iter().zip(&setup.real_cells) {
                    accumulate(&mut m, w, sub(*rc, *tc), add(v, *r), gamma);
                }
            }
            f(i, j, &m);
        }
        let gamma = 1.0 / (2.0 * alpeeq[i]).sqrt();
        let mut m = LatticeMoments::default();
        for (r, rc) in setup.real.iter().zip(&setup.real_cells) {
            accumulate(&mut m, 1.0, *rc, *r, gamma);
        }
        f(i, i, &m);
    }
}

/// `G` (3) and `W = 4 pi exp(-G^2 / 4 alpha^2) / (V G^2)` of the reciprocal vector `m` as
/// functions of the lattice vectors `h[3 k + c]` (component `c` of vector `k`).
fn reciprocal_term<T: Scalar>(h: &[T; 9], m: V3, alpha: f64) -> [T; 4] {
    let e = |k: usize, c: usize| h[3 * k + c];
    // The rows of the matrix are the lattice vectors; its inverse has the reciprocal vectors
    // (without 2 pi) as columns.
    let cof = |r: usize, c: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
        e(r1, c1) * e(r2, c2) - e(r1, c2) * e(r2, c1)
    };
    let det = e(0, 0) * cof(0, 0) + e(0, 1) * cof(0, 1) + e(0, 2) * cof(0, 2);
    // inverse[c][k] = cof(k, c) / det.
    let two_pi = T::from_f64(2.0 * PI);
    let g: [T; 3] = std::array::from_fn(|c| {
        let mut sum = T::zero();
        for (k, &mk) in m.iter().enumerate() {
            if mk != 0.0 {
                sum += cof(k, c) * T::from_f64(mk);
            }
        }
        two_pi * sum / det
    });
    let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    let volume = if det.val() < 0.0 { -det } else { det };
    let w = T::from_f64(4.0 * PI) * (-g2 / T::from_f64(4.0 * alpha * alpha)).exp() / (volume * g2);
    [g[0], g[1], g[2], w]
}

/// The derivatives of the reciprocal part of `0.5 q^T A q` (and of `A q`) with respect to
/// the lattice vectors, at fixed `alpha`, from the structure factors
/// `S = sum_j q_j exp(i G.x_j)`: the reciprocal energy is `0.5 sum_G W |S|^2` (the
/// Wigner-Seitz weights of a pair sum to one and `G.L t` is a multiple of `2 pi`). The
/// variables are the positions and the lattice vectors `v = 3 k + c`.
#[derive(Debug, Clone)]
pub struct ReciprocalLattice {
    /// `xh[i][c][v] = d2E / dx_ic d a_v`.
    pub xh: Vec<[[f64; 9]; 3]>,
    /// `hh[v][w] = d2E / d a_v d a_w`.
    pub hh: [[f64; 9]; 9],
    /// `response[i][v] = d (A q)_i / d a_v` at fixed charges.
    pub response: Vec<[f64; 9]>,
}

/// [`ReciprocalLattice`] of the charges `q` at `x`.
pub fn reciprocal_lattice(x: &[V3], q: &[f64], setup: &EwaldSetup) -> ReciprocalLattice {
    let n = x.len();
    let mut out = ReciprocalLattice {
        xh: vec![[[0.0; 9]; 3]; n],
        hh: [[0.0; 9]; 9],
        response: vec![[0.0; 9]; n],
    };
    let flat: [f64; 9] = std::array::from_fn(|v| setup.lattice[v / 3][v % 3]);
    let dot3 = |a: V3, b: V3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for &m in &setup.reciprocal_cells {
        // G and W with their first and second derivatives in the lattice vectors.
        let mut val = [0.0; 4];
        let mut d1 = [[0.0; 9]; 4];
        let mut d2 = [[[0.0; 9]; 9]; 4];
        let mut hd: [HyperDual; 9] = flat.map(HyperDual::constant);
        for u in 0..9 {
            hd[u].f1 = 1.0;
            for v in u..9 {
                hd[v].f2 = 1.0;
                let r = reciprocal_term(&hd, m, setup.alpha);
                for o in 0..4 {
                    if u == 0 && v == 0 {
                        val[o] = r[o].f0;
                    }
                    if v == u {
                        d1[o][u] = r[o].f1;
                    }
                    d2[o][u][v] = r[o].f12;
                    d2[o][v][u] = r[o].f12;
                }
                hd[v].f2 = 0.0;
            }
            hd[u].f1 = 0.0;
        }
        let g = [val[0], val[1], val[2]];
        let w = val[3];
        let dg: [V3; 9] = std::array::from_fn(|v| [d1[0][v], d1[1][v], d1[2][v]]);
        let dw = d1[3];
        let ddw = d2[3];
        // The structure factor and its moments.
        let (mut c, mut sn) = (0.0, 0.0);
        let (mut cx, mut sx) = ([0.0; 3], [0.0; 3]);
        let mut trig = Vec::with_capacity(n);
        for (xi, &qi) in x.iter().zip(q) {
            let (si, ci) = dot3(g, *xi).sin_cos();
            trig.push((ci, si));
            c += qi * ci;
            sn += qi * si;
            for a in 0..3 {
                cx[a] += qi * ci * xi[a];
                sx[a] += qi * si * xi[a];
            }
        }
        let power = c * c + sn * sn;
        let mut y = [0.0; 3];
        let mut m1 = [[0.0; 3]; 3];
        for i in 0..n {
            let (ci, si) = trig[i];
            let t_i = ci * c + si * sn;
            let q_i = si * c - ci * sn;
            let xi = x[i];
            let a_vec = [0, 1, 2].map(|a| si * cx[a] - ci * sx[a]);
            let b_vec = [0, 1, 2].map(|a| ci * cx[a] + si * sx[a]);
            for a in 0..3 {
                y[a] += q[i] * q_i * xi[a];
                for b in 0..3 {
                    m1[a][b] += q[i] * t_i * xi[a] * xi[b];
                }
            }
            for v in 0..9 {
                let dvx = dot3(dg[v], xi);
                out.response[i][v] += dw[v] * t_i + w * (-q_i * dvx + dot3(dg[v], a_vec));
                let dq = t_i * dvx - dot3(dg[v], b_vec);
                for cc in 0..3 {
                    out.xh[i][cc][v] -=
                        q[i] * (dw[v] * q_i * g[cc] + w * dq * g[cc] + w * q_i * dg[v][cc]);
                }
            }
        }
        let mut m12 = m1;
        for a in 0..3 {
            for b in 0..3 {
                m12[a][b] -= cx[a] * cx[b] + sx[a] * sx[b];
            }
        }
        let dp: [f64; 9] = std::array::from_fn(|v| -2.0 * dot3(dg[v], y));
        for v in 0..9 {
            for u in 0..9 {
                let d2g = [d2[0][v][u], d2[1][v][u], d2[2][v][u]];
                let mut quad = 0.0;
                for a in 0..3 {
                    for b in 0..3 {
                        quad += dg[v][a] * m12[a][b] * dg[u][b];
                    }
                }
                let ddp = -2.0 * (dot3(d2g, y) + quad);
                out.hh[v][u] += 0.5 * (ddw[v][u] * power + dw[v] * dp[u] + dw[u] * dp[v] + w * ddp);
            }
        }
    }
    out
}

/// The Ewald matrix of the atoms (the `n x n` block of xtb's EEQ matrix) for the widths
/// `alpeeq` and hardnesses `gameeq`.
pub fn matrix(
    x: &[V3],
    setup: &EwaldSetup,
    images: &Images,
    alpeeq: &[f64],
    gameeq: &[f64],
) -> Mat<f64> {
    const TSQRT2PI: f64 = 0.797_884_560_802_866;
    let n = x.len();
    let mut a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..i {
            let gamma = 1.0 / (alpeeq[i] + alpeeq[j]).sqrt();
            let shifts = &images.shifts[i][j];
            let w = 1.0 / shifts.len() as f64;
            let mut value = 0.0;
            for t in shifts {
                let v = sub(sub(x[i], x[j]), *t);
                for r in &setup.real {
                    let d = norm(add(v, *r));
                    if d >= SMALLEST_DISTANCE {
                        value += w * phi(gamma, setup.alpha, d).0;
                    }
                }
                for (g, weight) in setup.reciprocal.iter().zip(&setup.weight) {
                    value += w * weight * dot(*g, v).cos();
                }
            }
            a[(i, j)] = value;
            a[(j, i)] = value;
        }
        let gamma = 1.0 / (2.0 * alpeeq[i]).sqrt();
        let mut value = 0.0;
        for r in &setup.real {
            let d = norm(*r);
            if d >= SMALLEST_DISTANCE {
                value += phi(gamma, setup.alpha, d).0;
            }
        }
        value += setup.weight.iter().sum::<f64>();
        a[(i, i)] = value + gameeq[i] + TSQRT2PI / alpeeq[i].sqrt() - 2.0 * setup.alpha / PI.sqrt();
    }
    a
}

/// The gradient (added to `g`) and the strain derivative `sigma_ab = dE/d eps_ab` (added to
/// `sigma`, not divided by the volume) of `0.5 q^T A q` at fixed charges.
pub fn derivatives(
    x: &[V3],
    q: &[f64],
    setup: &EwaldSetup,
    images: &Images,
    alpeeq: &[f64],
    g: &mut [V3],
    sigma: &mut [[f64; 3]; 3],
) {
    let n = x.len();
    for i in 0..n {
        for j in 0..i {
            let gamma = 1.0 / (alpeeq[i] + alpeeq[j]).sqrt();
            let shifts = &images.shifts[i][j];
            let w = 1.0 / shifts.len() as f64;
            let qq = q[i] * q[j];
            for t in shifts {
                let v = sub(sub(x[i], x[j]), *t);
                for r in &setup.real {
                    let rv = add(v, *r);
                    let d = norm(rv);
                    if d < SMALLEST_DISTANCE {
                        continue;
                    }
                    let f = w * qq * phi(gamma, setup.alpha, d).1;
                    for c in 0..3 {
                        g[i][c] += f * rv[c];
                        g[j][c] -= f * rv[c];
                        for b in 0..3 {
                            sigma[c][b] += f * rv[c] * rv[b];
                        }
                    }
                }
                for (gv, weight) in setup.reciprocal.iter().zip(&setup.weight) {
                    let phase = dot(*gv, v);
                    let f = -w * qq * weight * phase.sin();
                    let c2 = w * qq * weight * phase.cos();
                    let g2 = dot(*gv, *gv);
                    let k = 2.0 / g2 + 1.0 / (2.0 * setup.alpha * setup.alpha);
                    for c in 0..3 {
                        g[i][c] += f * gv[c];
                        g[j][c] -= f * gv[c];
                        for b in 0..3 {
                            let delta = if b == c { 1.0 } else { 0.0 };
                            sigma[c][b] += c2 * (k * gv[c] * gv[b] - delta);
                        }
                    }
                }
            }
        }
        // The self terms: the images of the atom itself and the reciprocal sum at v = 0.
        let gamma = 1.0 / (2.0 * alpeeq[i]).sqrt();
        let qq = 0.5 * q[i] * q[i];
        for r in &setup.real {
            let d = norm(*r);
            if d < SMALLEST_DISTANCE {
                continue;
            }
            let f = qq * phi(gamma, setup.alpha, d).1;
            for c in 0..3 {
                for b in 0..3 {
                    sigma[c][b] += f * r[c] * r[b];
                }
            }
        }
        for (gv, weight) in setup.reciprocal.iter().zip(&setup.weight) {
            let g2 = dot(*gv, *gv);
            let k = 2.0 / g2 + 1.0 / (2.0 * setup.alpha * setup.alpha);
            for c in 0..3 {
                for b in 0..3 {
                    let delta = if b == c { 1.0 } else { 0.0 };
                    sigma[c][b] += qq * weight * (k * gv[c] * gv[b] - delta);
                }
            }
        }
    }
}
