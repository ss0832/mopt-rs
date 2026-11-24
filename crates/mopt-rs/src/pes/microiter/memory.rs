//! The limited-memory BFGS pairs of the relaxations.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::Col;

use crate::math::linalg::{col_dot, col_norm};

/// The pairs of steps and gradient changes of the limited-memory BFGS of the relaxations
/// (Nocedal, Math. Comput. 1980, 35, 773). A caller that relaxes similar problems one after
/// the other (the environment of an ONIOM optimization at the next outer step) keeps them, so
/// that a relaxation starts with the curvature the last one learned.
///
/// The pairs are in the variables of the relaxation that made them: Cartesian coordinates, or
/// the coordinates of an [`super::InnerCoordinates`] (whose generation is kept with them, so
/// that pairs of coordinates set up anew are not used).
#[derive(Debug, Clone, Default)]
pub struct MicroMemory {
    pairs: VecDeque<(Col<f64>, Col<f64>)>,
    /// The generation of the coordinates of the pairs (`None`: Cartesian).
    generation: Option<usize>,
    /// The lowest mode of the last check of the curvature (unit length, in the variables of
    /// the problem): the start of the next check.
    pub(super) mode: Option<Col<f64>>,
}

impl MicroMemory {
    /// `-H g` with the inverse Hessian of the pairs (two-loop recursion; the initial matrix is
    /// `s.y / y.y` of the newest pair, the identity without pairs).
    pub(super) fn direction(&self, g: &Col<f64>) -> Col<f64> {
        self.direction_with(g, &|_| None)
    }

    /// [`Self::direction`] with `h0(q)` as the initial inverse Hessian of the recursion where
    /// it gives one (a preconditioner, Nocedal and Wright section 7.2).
    pub(super) fn direction_with(
        &self,
        g: &Col<f64>,
        h0: &dyn Fn(&Col<f64>) -> Option<Col<f64>>,
    ) -> Col<f64> {
        let n = g.nrows();
        let mut q = g.clone();
        let mut factors = Vec::with_capacity(self.pairs.len());
        for (s, y) in self.pairs.iter().rev() {
            let rho = 1.0 / col_dot(y, s);
            let a = rho * col_dot(s, &q);
            q = Col::from_fn(n, |i| q[i] - a * y[i]);
            factors.push((a, rho));
        }
        let mut r = match h0(&q).filter(|r| r.nrows() == n) {
            // The inverse of the model scaled as the newest pair says, `s.y / y.P^-1 y`
            // (Nocedal and Wright, eq. 7.20 with the model in place of the identity), so that a
            // model of the wrong stiffness gives steps of the right length.
            Some(r) => {
                let scale = self
                    .pairs
                    .back()
                    .and_then(|(s, y)| {
                        h0(y)
                            .filter(|py| py.nrows() == n)
                            .map(|py| col_dot(s, y) / col_dot(y, &py))
                    })
                    .filter(|g| g.is_finite() && *g > 0.0)
                    .unwrap_or(1.0);
                Col::from_fn(n, |i| scale * r[i])
            }
            None => {
                let gamma = self
                    .pairs
                    .back()
                    .map_or(1.0, |(s, y)| col_dot(s, y) / col_dot(y, y));
                Col::from_fn(n, |i| gamma * q[i])
            }
        };
        for ((s, y), (a, rho)) in self.pairs.iter().zip(factors.iter().rev()) {
            let b = rho * col_dot(y, &r);
            r = Col::from_fn(n, |i| r[i] + (a - b) * s[i]);
        }
        Col::from_fn(n, |i| -r[i])
    }

    /// `H^-1 v` with the inverse Hessian of the pairs (the identity without pairs), a
    /// positive definite operator: a preconditioner for solves with the Hessian.
    pub fn apply_inverse(&self, v: &Col<f64>) -> Col<f64> {
        self.apply_inverse_with(v, &|_| None)
    }

    /// [`Self::apply_inverse`] with `h0` as the initial matrix (the two-loop recursion of
    /// `direction_with`).
    pub fn apply_inverse_with(
        &self,
        v: &Col<f64>,
        h0: &dyn Fn(&Col<f64>) -> Option<Col<f64>>,
    ) -> Col<f64> {
        let d = self.direction_with(v, h0);
        Col::from_fn(d.nrows(), |i| -d[i])
    }

    /// Keep the pair `(s, y)` when it has positive curvature, at most `limit` pairs.
    pub(super) fn push(&mut self, s: Col<f64>, y: Col<f64>, limit: usize) {
        if col_dot(&s, &y) > 1e-14 * col_norm(&s) * col_norm(&y) {
            self.pairs.push_back((s, y));
            while self.pairs.len() > limit.max(1) {
                self.pairs.pop_front();
            }
        }
    }

    /// Forget the pairs.
    pub(super) fn clear(&mut self) {
        self.pairs.clear();
    }

    /// Forget pairs of another number of variables or of other coordinates.
    pub(super) fn fit(&mut self, n: usize, generation: Option<usize>) {
        if self.generation != generation || self.pairs.front().is_some_and(|(s, _)| s.nrows() != n)
        {
            self.pairs.clear();
        }
        self.generation = generation;
    }

    /// The generation of the coordinates of the pairs (`None`: Cartesian).
    pub fn generation(&self) -> Option<usize> {
        self.generation
    }

    /// Pairs kept.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether no pair is kept.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}
