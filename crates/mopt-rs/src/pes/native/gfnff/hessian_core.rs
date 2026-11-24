//! The pieces the analytic GFN-FF Hessian is assembled from: the variables (the positions,
//! and in a cell optionally the three lattice vectors), the accumulator of the Hessian with
//! its pair blocks, the log-capped coordination numbers with their first and second
//! derivatives, and the chain rule through them.
//!
//! The variables are 3-vectors: the atoms `0..N` and, with the lattice, the lattice vectors
//! `a_k` as the pseudo-atoms `N + k`. The image of atom `a` in cell `n` is at
//! `x_a + sum_k n_k a_k`, so a function of `r = x_i - x_j + sum_k dn_k a_k` has the block
//! `c_u c_v K` between the variables `u` and `v`, with `K = d2f / dr dr` and the coefficients
//! `c = (1, -1, dn_0, dn_1, dn_2)` of `(x_i, x_j, a_0, a_1, a_2)`.
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
use super::neighbors::V3;

/// A 3 x 3 block.
pub type B3 = [[f64; 3]; 3];

/// The variables of the Hessian.
#[derive(Debug, Clone, Copy)]
pub struct Vars {
    /// The number of atoms.
    pub atoms: usize,
    /// Whether the three lattice vectors are variables (after the atoms).
    pub lattice: bool,
}

impl Vars {
    /// The number of 3-vector variables.
    pub fn vectors(&self) -> usize {
        self.atoms + if self.lattice { 3 } else { 0 }
    }

    /// The dimension of the Hessian.
    pub fn dim(&self) -> usize {
        3 * self.vectors()
    }

    /// The derivative of a function of `r = x_i - x_j + sum_k dn_k a_k` whose derivative along
    /// `r` is `d`, as a sparse vector of the variables (the atoms cancel for `i = j`).
    pub fn pair_vector(&self, i: usize, j: usize, dn: V3, d: V3) -> Vec<(usize, V3)> {
        let mut out = Vec::with_capacity(5);
        if i != j {
            out.push((i, d));
            out.push((j, d.map(|c| -c)));
        }
        if self.lattice {
            for (kk, &nk) in dn.iter().enumerate() {
                if nk != 0.0 {
                    out.push((self.atoms + kk, d.map(|c| nk * c)));
                }
            }
        }
        out
    }

    /// The sparse vector of a sum over images of one pair: `d0 = sum d`, `d1[k] = sum dn_k d`.
    pub fn pair_sum_vector(&self, i: usize, j: usize, d0: V3, d1: &[V3; 3]) -> Vec<(usize, V3)> {
        let mut out = Vec::with_capacity(5);
        if i != j {
            out.push((i, d0));
            out.push((j, d0.map(|c| -c)));
        }
        if self.lattice {
            for (kk, dk) in d1.iter().enumerate() {
                if dk.iter().any(|&v| v != 0.0) {
                    out.push((self.atoms + kk, *dk));
                }
            }
        }
        out
    }
}

/// The blocks of a pair summed over images: `k0 = sum K`, `k1[k] = sum dn_k K`,
/// `k2[k][l] = sum dn_k dn_l K` (the last two only with the lattice).
#[derive(Debug, Clone, Default)]
pub struct PairSum {
    /// `sum K`.
    pub k0: B3,
    /// `sum dn_k K`.
    pub k1: [B3; 3],
    /// `sum dn_k dn_l K`.
    pub k2: [[B3; 3]; 3],
}

impl PairSum {
    /// Add the block `k` of an image with the lattice coefficients `dn`.
    pub fn add(&mut self, dn: V3, kb: &B3, lattice: bool) {
        for a in 0..3 {
            for b in 0..3 {
                self.k0[a][b] += kb[a][b];
            }
        }
        if !lattice {
            return;
        }
        for p in 0..3 {
            if dn[p] == 0.0 {
                continue;
            }
            for a in 0..3 {
                for b in 0..3 {
                    self.k1[p][a][b] += dn[p] * kb[a][b];
                }
            }
            for q in 0..3 {
                if dn[q] == 0.0 {
                    continue;
                }
                for a in 0..3 {
                    for b in 0..3 {
                        self.k2[p][q][a][b] += dn[p] * dn[q] * kb[a][b];
                    }
                }
            }
        }
    }
}

/// The Hessian being assembled.
pub struct Acc {
    /// The Hessian (Hartree/Bohr^2), row and column `3 variable + component`.
    pub h: Mat<f64>,
    /// The variables.
    pub vars: Vars,
}

impl Acc {
    /// A zero Hessian of the variables `vars`.
    pub fn new(vars: Vars) -> Self {
        Self {
            h: Mat::zeros(vars.dim(), vars.dim()),
            vars,
        }
    }

    /// `H[a, b] += c k`.
    fn block(&mut self, a: usize, b: usize, c: f64, kb: &B3) {
        for x in 0..3 {
            for y in 0..3 {
                self.h[(3 * a + x, 3 * b + y)] += c * kb[x][y];
            }
        }
    }

    /// A term of `x_i - x_j` (no lattice part) with the symmetric block `k = d2E / dx_i dx_i`.
    pub fn pair_block(&mut self, i: usize, j: usize, kb: &B3) {
        if i == j {
            return;
        }
        self.block(i, i, 1.0, kb);
        self.block(j, j, 1.0, kb);
        self.block(i, j, -1.0, kb);
        self.block(j, i, -1.0, kb);
    }

    /// A term of `r = x_i - x_j + sum_k dn_k a_k` with the symmetric block `k = d2E / dr dr`.
    pub fn pair_cell(&mut self, i: usize, j: usize, dn: V3, kb: &B3) {
        self.pair_block(i, j, kb);
        if !self.vars.lattice {
            return;
        }
        let base = self.vars.atoms;
        for p in 0..3 {
            if dn[p] == 0.0 {
                continue;
            }
            if i != j {
                self.block(i, base + p, dn[p], kb);
                self.block(base + p, i, dn[p], kb);
                self.block(j, base + p, -dn[p], kb);
                self.block(base + p, j, -dn[p], kb);
            }
            for q in 0..3 {
                if dn[q] != 0.0 {
                    self.block(base + p, base + q, dn[p] * dn[q], kb);
                }
            }
        }
    }

    /// The pair `(i, j)` summed over images ([`PairSum`]), scaled by `c`.
    pub fn pair_sum(&mut self, i: usize, j: usize, s: &PairSum, c: f64) {
        if i != j {
            let k0 = s.k0.map(|row| row.map(|v| c * v));
            self.pair_block(i, j, &k0);
        }
        if !self.vars.lattice {
            return;
        }
        let base = self.vars.atoms;
        for p in 0..3 {
            if i != j {
                self.block(i, base + p, c, &s.k1[p]);
                self.block(base + p, i, c, &s.k1[p]);
                self.block(j, base + p, -c, &s.k1[p]);
                self.block(base + p, j, -c, &s.k1[p]);
            }
            for q in 0..3 {
                self.block(base + p, base + q, c, &s.k2[p][q]);
            }
        }
    }

    /// `c (g (x) h)` over the sparse vectors `g` and `h` (variables and their 3-vectors).
    pub fn outer(&mut self, c: f64, g: &[(usize, V3)], h: &[(usize, V3)]) {
        if c == 0.0 {
            return;
        }
        for &(a, ga) in g {
            for &(b, hb) in h {
                for x in 0..3 {
                    for y in 0..3 {
                        self.h[(3 * a + x, 3 * b + y)] += c * ga[x] * hb[y];
                    }
                }
            }
        }
    }
}

/// The block `f'' u u^T + (f' / r)(1 - u u^T)` of `f(|rv|)`, `f' = d1`, `f'' = d2`.
pub fn radial_block(rv: V3, d1: f64, d2: f64) -> B3 {
    let r = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
    let u = rv.map(|c| c / r);
    let mut kb = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            kb[a][b] = d2 * u[a] * u[b] + d1 / r * (delta - u[a] * u[b]);
        }
    }
    kb
}

/// The block `4 F'' rv rv^T + 2 F' 1` of `F(|rv|^2)`, `F' = fs`, `F'' = fss`.
pub fn square_block(rv: V3, fs: f64, fss: f64) -> B3 {
    let mut kb = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            kb[a][b] = 4.0 * fss * rv[a] * rv[b] + 2.0 * fs * delta;
        }
    }
    kb
}

/// `0.5 (1 + erf(-kn (r - r0) / r0))` with its first and second derivatives in `r`; the
/// first derivative with the value of `sqrt(pi)` the gradient uses.
pub fn erf_count(kn: f64, r0: f64, r: f64, sqrtpi: f64) -> (f64, f64, f64) {
    let dr = (r - r0) / r0;
    let value = 0.5 * (1.0 + libm::erf(-kn * dr));
    let d1 = -kn / (sqrtpi * r0) * (-(kn * kn) * dr * dr).exp();
    let d2 = d1 * (-2.0 * kn * kn * dr / r0);
    (value, d1, d2)
}

/// Merge the entries of the same variable of a sparse vector.
pub fn merged(mut v: Vec<(usize, V3)>) -> Vec<(usize, V3)> {
    v.sort_unstable_by_key(|e| e.0);
    let mut out: Vec<(usize, V3)> = Vec::with_capacity(v.len());
    for (a, d) in v {
        match out.last_mut() {
            Some(last) if last.0 == a => {
                for c in 0..3 {
                    last.1[c] += d[c];
                }
            }
            _ => out.push((a, d)),
        }
    }
    out
}

/// A pair of the coordination numbers: `count(|rv|)`,
/// `rv = x_i - x_j + sum_k dn_k a_k`, adds to the sums of `i` and `j` (twice to that of `i`
/// for an atom and its own image, `i = j`, whose only variables are the lattice vectors).
struct CnPair {
    i: usize,
    j: usize,
    rv: V3,
    dn: V3,
    d1: f64,
    d2: f64,
}

/// The log-capped coordination numbers with their first and second derivatives.
pub struct CnSecond {
    /// The capped CN.
    pub cn: Vec<f64>,
    /// `L'(s_i)` and `L''(s_i)` of the cap `L` of the sums `s_i`.
    lp: Vec<f64>,
    lpp: Vec<f64>,
    /// `d s_i / dx`, by variable.
    ds: Vec<Vec<(usize, V3)>>,
    pairs: Vec<CnPair>,
}

impl CnSecond {
    fn from_sums(vars: Vars, sums: Vec<f64>, pairs: Vec<CnPair>) -> Self {
        let cut = k::PARAM_CNMAX;
        let ec = cut.exp();
        let mut raw = vec![Vec::new(); vars.atoms];
        for p in &pairs {
            let r = (p.rv[0] * p.rv[0] + p.rv[1] * p.rv[1] + p.rv[2] * p.rv[2]).sqrt();
            let d = p.rv.map(|c| p.d1 * c / r);
            let v = vars.pair_vector(p.i, p.j, p.dn, d);
            raw[p.i].extend_from_slice(&v);
            raw[p.j].extend_from_slice(&v);
        }
        let ds = raw.into_iter().map(merged).collect();
        let lp = sums.iter().map(|&s| ec / (ec + s.exp())).collect();
        let lpp = sums
            .iter()
            .map(|&s| -ec * s.exp() / (ec + s.exp()).powi(2))
            .collect();
        let cn = sums
            .iter()
            .map(|&s| (1.0 + ec).ln() - (1.0 + (cut - s).exp()).ln())
            .collect();
        Self {
            cn,
            lp,
            lpp,
            ds,
            pairs,
        }
    }

    /// The CN of a molecule (`log_cn_derivatives`).
    pub fn molecule(el: &[Atom], x: &[V3]) -> Self {
        const KN: f64 = 7.5;
        const SQRTPI: f64 = 1.772_453_850_91;
        let n = x.len();
        let thr = k::CNTHR.sqrt();
        let mut sums = vec![0.0; n];
        let mut pairs = Vec::new();
        for i in 1..n {
            for j in 0..i {
                let rv = [0, 1, 2].map(|c| x[i][c] - x[j][c]);
                let r = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
                if r > thr {
                    continue;
                }
                let (e, d1, d2) = erf_count(KN, el[i].rcov + el[j].rcov, r, SQRTPI);
                sums[i] += e;
                sums[j] += e;
                if d1 != 0.0 || d2 != 0.0 {
                    pairs.push(CnPair {
                        i,
                        j,
                        rv,
                        dn: [0.0; 3],
                        d1,
                        d2,
                    });
                }
            }
        }
        let vars = Vars {
            atoms: n,
            lattice: false,
        };
        Self::from_sums(vars, sums, pairs)
    }

    /// The CN of a cell over the translations `trans` of the cells `cells` (the first the
    /// origin) within `cutoff` (`coordination_numbers`).
    pub fn periodic(
        el: &[Atom],
        x: &[V3],
        trans: &[V3],
        cells: &[V3],
        cutoff: f64,
        vars: Vars,
    ) -> Self {
        const KN: f64 = 7.5;
        let n = x.len();
        let cutoff2 = cutoff * cutoff;
        let mut sums = vec![0.0; n];
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in 0..=i {
                let rc = el[i].rcov + el[j].rcov;
                for (t, (tv, cell)) in trans.iter().zip(cells).enumerate() {
                    let rv = [0, 1, 2].map(|c| x[i][c] - (x[j][c] + tv[c]));
                    let r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
                    if r2 > cutoff2 || r2 < 1.0e-12 {
                        continue;
                    }
                    let (e, d1, d2) = erf_count(KN, rc, r2.sqrt(), PI.sqrt());
                    sums[i] += e;
                    if i != j || t != 0 {
                        sums[j] += e;
                    }
                    if (i != j || vars.lattice) && (d1 != 0.0 || d2 != 0.0) {
                        pairs.push(CnPair {
                            i,
                            j,
                            rv,
                            dn: cell.map(|c| -c),
                            d1,
                            d2,
                        });
                    }
                }
            }
        }
        Self::from_sums(vars, sums, pairs)
    }

    /// `dCN_i / dx` by variable.
    pub fn jacobian_row(&self, i: usize) -> impl Iterator<Item = (usize, V3)> + '_ {
        let l = self.lp[i];
        self.ds[i].iter().map(move |&(a, d)| (a, d.map(|c| l * c)))
    }
}

/// The derivatives of the energy with respect to the coordination numbers.
pub struct Chain {
    /// `dE / dCN_i`.
    pub w: Vec<f64>,
    /// `d2E / dCN_i dCN_j` (symmetric).
    pub m: Mat<f64>,
    /// `vt[(3 v + c, i)]`: the derivative of `dE / dCN_i` with respect to component `c` of
    /// variable `v` at fixed CN.
    pub vt: Mat<f64>,
}

impl Chain {
    /// No derivatives yet.
    pub fn new(vars: Vars) -> Self {
        Self {
            w: vec![0.0; vars.atoms],
            m: Mat::zeros(vars.atoms, vars.atoms),
            vt: Mat::zeros(vars.dim(), vars.atoms),
        }
    }

    /// Add the derivative of `dE/dCN_i` along the sparse vector `d` scaled by `c`.
    pub fn cross(&mut self, i: usize, c: f64, d: &[(usize, V3)]) {
        for &(a, v) in d {
            for x in 0..3 {
                self.vt[(3 * a + x, i)] += c * v[x];
            }
        }
    }

    /// `M[i, j] += c` and `M[j, i] += c`: the mixed derivative of a function of `CN_i` and
    /// `CN_j` (twice for `i = j`, a pair of an atom and its own image, whose function of
    /// `CN_i` has the second derivative `f_11 + 2 f_12 + f_22`).
    pub fn mix(&mut self, i: usize, j: usize, c: f64) {
        self.m[(i, j)] += c;
        self.m[(j, i)] += c;
    }

    /// `H += sum_i w_i d2CN_i + J^T M J + J^T V + V^T J`; `J^T V + V^T J` enters as
    /// `2 V^T J`, which the final symmetrization of the Hessian makes right.
    pub fn apply(&self, cn: &CnSecond, acc: &mut Acc) {
        let n = cn.cn.len();
        let dim = acc.vars.dim();
        for i in 0..n {
            acc.outer(self.w[i] * cn.lpp[i], &cn.ds[i], &cn.ds[i]);
        }
        for p in &cn.pairs {
            let c = self.w[p.i] * cn.lp[p.i] + self.w[p.j] * cn.lp[p.j];
            if c != 0.0 {
                acc.pair_cell(p.i, p.j, p.dn, &radial_block(p.rv, c * p.d1, c * p.d2));
            }
        }
        let jac: Vec<Vec<(usize, V3)>> = (0..n).map(|i| cn.jacobian_row(i).collect()).collect();
        // Z = 2 V^T + J^T M, then H += Z J.
        let mut z = Mat::from_fn(dim, n, |r, c| 2.0 * self.vt[(r, c)]);
        for kk in 0..n {
            let col = z.col_as_slice_mut(kk);
            for (i, row) in jac.iter().enumerate() {
                let mik = self.m[(i, kk)];
                if mik == 0.0 {
                    continue;
                }
                for &(a, d) in row {
                    for c in 0..3 {
                        col[3 * a + c] += mik * d[c];
                    }
                }
            }
        }
        for (kk, row) in jac.iter().enumerate() {
            let zk = z.col_as_slice(kk);
            for &(b, d) in row {
                for c in 0..3 {
                    let f = d[c];
                    if f == 0.0 {
                        continue;
                    }
                    let dst = acc.h.col_as_slice_mut(3 * b + c);
                    for (h, &zv) in dst.iter_mut().zip(zk) {
                        *h += f * zv;
                    }
                }
            }
        }
    }
}
