//! Extrapolation of Fock matrices for the self-consistent field of a tight-binding or
//! mean-field method: ADIIS first, CDIIS when ADIIS does not look like converging.
//!
//! Every iteration hands in the Fock matrix `F_i = F(P_i)` built from the density `P_i` (the
//! total density, both spins, in the atomic orbitals with overlap `S`) and gets back the
//! matrix to diagonalize next.
//!
//! - **CDIIS** (Pulay, Chem. Phys. Lett. 1980, 73, 393; J. Comput. Chem. 1982, 3, 556): the
//!   combination `sum_i c_i F_i`, `sum_i c_i = 1`, that minimizes the norm of `sum_i c_i e_i`
//!   with the commutators `e_i = X^T (F_i P_i S - S P_i F_i) X` in the orthonormal basis of
//!   `X` (`X^T S X = 1`), which vanish at self-consistency (also with fractional occupations,
//!   since then `P` is a function of `F`).
//! - **ADIIS** (Hu, Yang, J. Chem. Phys. 2010, 132, 054109): the convex combination
//!   (`c_i >= 0`, `sum_i c_i = 1`) that minimizes the second-order model of the energy about the
//!   newest density `P_n` (the augmented Roothaan-Hall energy),
//!   `f(c) = sum_i c_i <P_i - P_n | F_n> + 1/2 sum_ij c_i c_j <P_i - P_n | F_j - F_n>`,
//!   `<A | B> = sum_mu,nu A_mu,nu B_mu,nu`; written here for the total density, where
//!   `dE/dP = F` (Hu and Yang's factor 2 belongs to the density of one spin of a closed shell).
//!   The global minimum on the simplex comes from [`crate::math::simplex_qp::simplex_minimum`].
//!
//! The switch: ADIIS from the start, CDIIS from then on once ADIIS does not look like
//! converging: the largest element of the commutator has not reached a new low for `patience`
//! iterations (ADIIS stalls or oscillates), or over the last `window` iterations it has not
//! fallen to `rate^window` of its value (ADIIS converges, but too slowly: damped ADIIS can
//! creep down linearly for dozens of iterations where CDIIS needs two or three). One matrix
//! alone is returned as it is. The defaults are chosen by measurement on the GFN1-xTB
//! reference systems (see the module of GFN1-xTB).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; its DIIS family extrapolates geometries).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::Mat;

use crate::error::MoptResult;
use crate::math::linalg::mul3;
use crate::math::simplex_qp::{least_squares, simplex_minimum};

/// What one extrapolation did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiisStep {
    /// The newest Fock matrix as it is (one matrix kept).
    Plain,
    /// The ADIIS combination.
    Adiis,
    /// The CDIIS combination.
    Cdiis,
}

#[derive(Debug, Clone)]
struct Entry {
    fock: Mat<f64>,
    density: Mat<f64>,
    error: Mat<f64>,
}

/// The ADIIS-then-CDIIS extrapolation of Fock matrices.
#[derive(Debug, Clone)]
pub struct FockDiis {
    /// Matrices kept (the oldest is dropped beyond).
    pub depth: usize,
    /// Iterations without a new lowest commutator after which CDIIS takes over from ADIIS.
    pub patience: usize,
    /// The slowest mean fall of the commutator per iteration that ADIIS may keep (`None`: no
    /// such test).
    pub rate: Option<f64>,
    /// The iterations over which `rate` is judged.
    pub window: usize,
    history: VecDeque<Entry>,
    cdiis: bool,
    best: f64,
    since_best: usize,
    /// The largest commutator elements of the last `window + 1` iterations.
    recent: VecDeque<f64>,
}

/// `<A | B> = sum A_ij B_ij`.
fn inner(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut s = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            s += a[(i, j)] * b[(i, j)];
        }
    }
    s
}

/// The commutator `X^T (F P S - S P F) X` of a Fock matrix and its density.
pub fn commutator(fock: &Mat<f64>, density: &Mat<f64>, s: &Mat<f64>, x: &Mat<f64>) -> Mat<f64> {
    let fps = mul3(fock.as_ref(), density.as_ref(), s.as_ref());
    let e = &fps - fps.transpose();
    mul3(x.transpose(), e.as_ref(), x.as_ref())
}

/// The largest element of `m` in magnitude.
pub fn largest(m: &Mat<f64>) -> f64 {
    let mut worst = 0.0_f64;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            worst = worst.max(m[(i, j)].abs());
        }
    }
    worst
}

impl FockDiis {
    /// An empty history.
    pub fn new(depth: usize, patience: usize) -> Self {
        Self {
            depth: depth.max(1),
            patience: patience.max(1),
            rate: None,
            window: 3,
            history: VecDeque::new(),
            cdiis: false,
            best: f64::INFINITY,
            since_best: 0,
            recent: VecDeque::new(),
        }
    }

    /// With the test of the rate of ADIIS: CDIIS takes over when the commutator falls by less
    /// than `rate` per iteration on average over `window` iterations.
    pub fn with_rate(mut self, rate: Option<f64>, window: usize) -> Self {
        self.rate = rate;
        self.window = window.max(1);
        self
    }

    /// Whether CDIIS has taken over.
    pub fn in_cdiis(&self) -> bool {
        self.cdiis
    }

    /// Add `fock = F(density)` with its commutator `error` ([`commutator`]) and return the
    /// matrix to diagonalize next and what was done.
    pub fn extrapolate(
        &mut self,
        fock: Mat<f64>,
        density: Mat<f64>,
        error: Mat<f64>,
    ) -> MoptResult<(Mat<f64>, DiisStep)> {
        let size = largest(&error);
        self.recent.push_back(size);
        while self.recent.len() > self.window + 1 {
            self.recent.pop_front();
        }
        self.history.push_back(Entry {
            fock,
            density,
            error,
        });
        while self.history.len() > self.depth {
            self.history.pop_front();
        }
        let m = self.history.len();
        let newest = &self.history[m - 1];
        if m == 1 {
            return Ok((newest.fock.clone(), DiisStep::Plain));
        }
        if size < self.best {
            self.best = size;
            self.since_best = 0;
        } else {
            self.since_best += 1;
        }
        if !self.cdiis && self.since_best >= self.patience {
            self.cdiis = true;
        }
        if let (false, Some(rate)) = (self.cdiis, self.rate) {
            if self.recent.len() > self.window {
                let then = self.recent[0];
                if size > then * rate.powi(self.window as i32) {
                    self.cdiis = true;
                }
            }
        }
        let (c, step) = if self.cdiis {
            (self.cdiis_coefficients()?, DiisStep::Cdiis)
        } else {
            (self.adiis_coefficients()?, DiisStep::Adiis)
        };
        let Some(c) = c else {
            return Ok((self.history[m - 1].fock.clone(), DiisStep::Plain));
        };
        let n = self.history[0].fock.nrows();
        let mut out = Mat::<f64>::zeros(n, n);
        for (ci, e) in c.iter().zip(&self.history) {
            out += &e.fock * faer::Scale(*ci);
        }
        Ok((out, step))
    }

    /// The CDIIS coefficients: `[B 1; 1^T 0] [c; l] = [0; 1]`, `B_ij = <e_i | e_j>`, by least
    /// squares (the oldest matrices give way when `B` is nearly singular).
    fn cdiis_coefficients(&self) -> MoptResult<Option<Vec<f64>>> {
        let m = self.history.len();
        let mut a = Mat::<f64>::zeros(m + 1, m + 1);
        for i in 0..m {
            for j in 0..=i {
                let v = inner(&self.history[i].error, &self.history[j].error);
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
            a[(i, m)] = 1.0;
            a[(m, i)] = 1.0;
        }
        // Scale the error block to order one so the constraint row is not swamped.
        let top = (0..m).map(|i| a[(i, i)]).fold(0.0_f64, f64::max);
        if top > 0.0 {
            for i in 0..m {
                for j in 0..m {
                    a[(i, j)] /= top;
                }
            }
        }
        let mut b = vec![0.0; m + 1];
        b[m] = 1.0;
        let x = least_squares(&a, &b)?;
        let c: Vec<f64> = x[..m].to_vec();
        let sum: f64 = c.iter().sum();
        if c.iter().any(|v| !v.is_finite()) || (sum - 1.0).abs() > 1e-6 {
            return Ok(None);
        }
        Ok(Some(c))
    }

    /// The ADIIS coefficients on the simplex.
    fn adiis_coefficients(&self) -> MoptResult<Option<Vec<f64>>> {
        let m = self.history.len();
        let n = &self.history[m - 1];
        let dp: Vec<Mat<f64>> = self
            .history
            .iter()
            .map(|e| &e.density - &n.density)
            .collect();
        let df: Vec<Mat<f64>> = self.history.iter().map(|e| &e.fock - &n.fock).collect();
        let l: Vec<f64> = dp.iter().map(|d| inner(d, &n.fock)).collect();
        let mut q = Mat::<f64>::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                q[(i, j)] = 0.5 * (inner(&dp[i], &df[j]) + inner(&dp[j], &df[i]));
            }
        }
        let c = simplex_minimum(&q, &l)?;
        if c.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        Ok(Some(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::linalg::symmetric_eigen;

    /// A two-level model whose Fock matrix depends on its density: `F = H + U diag(P)`.
    fn model(p: &Mat<f64>) -> Mat<f64> {
        let h = Mat::from_fn(2, 2, |i, j| if i == j { [-1.0, -0.6][i] } else { -0.3 });
        Mat::from_fn(2, 2, |i, j| {
            h[(i, j)] + if i == j { 0.8 * p[(i, i)] } else { 0.0 }
        })
    }

    fn density(f: &Mat<f64>) -> Mat<f64> {
        let (_, v) = symmetric_eigen(f).unwrap();
        // Two electrons in the lowest orbital.
        Mat::from_fn(2, 2, |i, j| 2.0 * v[(i, 0)] * v[(j, 0)])
    }

    #[test]
    fn the_commutator_vanishes_at_self_consistency_and_the_extrapolation_finds_it() {
        let s = Mat::<f64>::identity(2, 2);
        let x = Mat::<f64>::identity(2, 2);
        // Start from the density of the bare Hamiltonian (which does not commute with F(P)).
        let mut p = density(&model(&Mat::<f64>::zeros(2, 2)));
        let mut diis = FockDiis::new(6, 4);
        let mut steps = Vec::new();
        let mut converged = false;
        for _ in 0..60 {
            let f = model(&p);
            let e = commutator(&f, &p, &s, &x);
            if largest(&e) < 1e-12 {
                converged = true;
                break;
            }
            let (next, step) = diis.extrapolate(f, p.clone(), e).unwrap();
            steps.push(step);
            p = density(&next);
        }
        assert!(converged, "{steps:?}");
        assert_eq!(steps[0], DiisStep::Plain);
        assert!(steps.contains(&DiisStep::Adiis) || steps.contains(&DiisStep::Cdiis));
        // The answer is a fixed point: its density gives itself back.
        let back = density(&model(&p));
        assert!(largest(&(&back - &p)) < 1e-9);
    }

    /// Commutators that fall steadily by `factor` per iteration (a new low every time, so the
    /// test of stalling never switches): the iteration at which CDIIS takes over.
    fn switch_with(factor: f64, rate: Option<f64>) -> Option<usize> {
        let mut diis = FockDiis::new(6, 2).with_rate(rate, 3);
        let p = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        for k in 0..12_usize {
            let size = factor.powi(k as i32);
            let f = Mat::from_fn(2, 2, |i, j| if i == j { -1.0 } else { 0.1 * size });
            let e = Mat::from_fn(2, 2, |i, j| if i == j { 0.0 } else { size });
            diis.extrapolate(f, p.clone(), e).unwrap();
            if diis.in_cdiis() {
                return Some(k);
            }
        }
        None
    }

    #[test]
    fn cdiis_takes_over_when_adiis_falls_too_slowly() {
        // x0.6 per iteration is slower than the rate 0.3: CDIIS after the window of 3.
        assert_eq!(switch_with(0.6, Some(0.3)), Some(3));
        // x0.1 per iteration is fast enough; and without the rate test, a steady fall never
        // switches.
        assert_eq!(switch_with(0.1, Some(0.3)), None);
        assert_eq!(switch_with(0.6, None), None);
    }

    #[test]
    fn adiis_coefficients_lie_on_the_simplex() {
        let mut diis = FockDiis::new(4, 100);
        let s = Mat::<f64>::identity(2, 2);
        let x = Mat::<f64>::identity(2, 2);
        let mut p = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.0 });
        for _ in 0..3 {
            let f = model(&p);
            let e = commutator(&f, &p, &s, &x);
            diis.extrapolate(f.clone(), p.clone(), e).unwrap();
            p = density(&f);
        }
        let c = diis.adiis_coefficients().unwrap().unwrap();
        assert!(c.iter().all(|v| *v >= -1e-12), "{c:?}");
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }
}
