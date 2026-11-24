//! The step of high-index saddle dynamics with the optimizer's Hessian: `-opt hisd` and
//! `-opt ggad` with `-order k`.
//!
//! The step is `-beta (I - 2 V V^T) g`, the gradient reflected across `k` directions `V`
//! (J. Yin, L. Zhang, P. Zhang, SIAM J. Sci. Comput. 2019, 41, A3576), with Barzilai-Borwein
//! lengths `beta = s^T y / y^T y` (`s` the last step, `y` the change of the reflected
//! gradient; the first step and every step with `s^T y <= 0` of a fixed length), limited by
//! the optimizer's trust radius. The directions come from the Hessian of the optimizer (the
//! exact one with `-fc 1`, otherwise the updated one; the curvatures of an updated Hessian
//! are only as good as the update):
//!
//! - `hisd`: the eigenvectors of `k` of the lowest eigenvalues, the limit of fast
//!   direction dynamics. Of the `2k + 2` lowest eigenvectors those with the largest
//!   projection onto the directions of the last step (or onto the directions given by
//!   [`StepGenerator::seed_followed_modes`]) are taken, so that a soft positive mode that
//!   dips below an unstable one does not take its place;
//! - `ggad`: the directions of the generalized gentlest ascent dynamics (W. Quapp, J. M.
//!   Bofill, Theor. Chem. Acc. 2014, 133, 1510, eqs. 16-21), one Euler step of the deflated
//!   Rayleigh quotient flows per step with Schmidt orthogonalization, the first direction
//!   along the gradient at the start (their eq. 14).
//!
//! The search for saddle points from gradients alone, with products of the Hessian from
//! differences of gradients, is [`crate::path::hisd::HighIndexSaddleSearch`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{StepGenerator, StepInput};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen};
use crate::math::projection::{project_out_basis_mat, project_out_basis_vec};
use crate::path::hisd::settings::DirectionUpdate;
use crate::path::hisd::subspace::{euler_directions, gram_schmidt, orthonormal_to};

/// Which directions the step reflects across.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectedDirections {
    /// Eigenvectors of the Hessian, followed by their overlap.
    Hisd,
    /// GGAD's deflated flows.
    Ggad,
}

/// The reflected-gradient step.
#[derive(Debug, Clone)]
pub struct ReflectedGradient {
    directions: ReflectedDirections,
    mode_kill_order: Option<usize>,
    v: Vec<Col<f64>>,
    /// The reflected gradient of the last step.
    last: Option<Col<f64>>,
    /// The directions before the last GGAD update and their Euler directions.
    previous: Option<(Vec<Col<f64>>, Vec<Col<f64>>)>,
    /// Length of the first step (in the optimizer's coordinates).
    first_step: f64,
}

impl ReflectedGradient {
    /// The step of `-opt hisd` or `-opt ggad`.
    pub fn new(directions: ReflectedDirections) -> Self {
        Self {
            directions,
            mode_kill_order: None,
            v: Vec::new(),
            last: None,
            previous: None,
            first_step: 0.1,
        }
    }

    pub(crate) fn for_mode_kill(order: usize) -> Self {
        Self {
            mode_kill_order: Some(order),
            ..Self::new(ReflectedDirections::Hisd)
        }
    }

    fn gradient_step(&mut self, input: &StepInput<'_>, f: Col<f64>) -> Col<f64> {
        let first = self.first_step / col_norm(&f).max(1e-300);
        // A verification evaluation at the same point has no secant pair.
        // A small residual force must still produce a small proposed step there.
        let first = if self.mode_kill_order.is_some() {
            first.min(1.0)
        } else {
            first
        };
        let beta = match (&self.last, input.previous) {
            (Some(before), Some(previous)) if before.nrows() == f.nrows() => {
                let s = &previous.step;
                let y = &f - before;
                let (sy, yy) = (col_dot(s, &y), col_dot(&y, &y));
                if sy > 0.0 && yy > 0.0 {
                    sy / yy
                } else {
                    first
                }
            }
            _ => first,
        };
        self.last = Some(f.clone());
        f * -beta
    }

    /// Of the lowest eigenvectors, the `k` with the largest projection onto the current
    /// directions (all of the lowest `k` when there are none).
    fn followed(&self, values: &[f64], vectors: &[Col<f64>], k: usize) -> Vec<Col<f64>> {
        let pool = (2 * k + 2).min(vectors.len());
        let mut chosen: Vec<usize> = if self.v.len() == k {
            let mut scored: Vec<(usize, f64)> = (0..pool)
                .map(|j| {
                    let p: f64 = self.v.iter().map(|v| col_dot(v, &vectors[j]).powi(2)).sum();
                    (j, p)
                })
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            scored.iter().take(k).map(|s| s.0).collect()
        } else {
            (0..k.min(vectors.len())).collect()
        };
        chosen.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
        chosen.into_iter().map(|j| vectors[j].clone()).collect()
    }
}

impl StepGenerator for ReflectedGradient {
    fn name(&self) -> &str {
        match self.directions {
            ReflectedDirections::Hisd => "hisd",
            ReflectedDirections::Ggad => "ggad",
        }
    }

    fn uses_hessian(&self) -> bool {
        self.mode_kill_order != Some(0)
    }

    /// The steps at a saddle order read the bias Hessian whatever the mode kill does.
    fn uses_bias_hessian(&self) -> bool {
        true
    }

    fn learns_negative_curvature(&self) -> bool {
        true
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let k = input.saddle_order;
        if k == 0 && self.mode_kill_order == Some(0) {
            let gradient = match input.rigid_body_basis {
                Some(basis) => project_out_basis_vec(basis, input.gradient),
                None => input.gradient.clone(),
            };
            return Ok(self.gradient_step(input, gradient));
        }
        if k == 0 {
            return Err(MoptError::config(format!(
                "-opt {} looks for saddle points: give -order 1 or more",
                self.name()
            )));
        }
        let n = input.gradient.nrows();
        let mut h: Mat<f64> = input.hessian.clone();
        if let Some(b) = input.bias_hessian {
            h += b;
        }
        let mut g = input.gradient.clone();
        let mut shifted = h.clone();
        let mut dropped = 0;
        if let Some(d) = input
            .rigid_body_basis
            .filter(|d| d.nrows() == n && d.ncols() > 0)
        {
            h = project_out_basis_mat(d, &h);
            g = project_out_basis_vec(d, &g);
            let scale = (0..n).map(|i| h[(i, i)].abs()).fold(1.0, f64::max);
            shifted = &h + (d * d.transpose()) * (10.0 * scale * n as f64 + 1.0);
            dropped = d.ncols();
        }
        if k + dropped > n {
            return Err(MoptError::config(format!(
                "-opt {}: order {k} for {} free coordinates",
                self.name(),
                n - dropped
            )));
        }
        let (values, vectors) = symmetric_eigen(&shifted)?;
        let kept = n - dropped;
        let values: Vec<f64> = (0..kept).map(|j| values[j]).collect();
        let vectors: Vec<Col<f64>> = (0..kept).map(|j| vectors.col(j).to_owned()).collect();
        let project = |v: &Col<f64>| match input.rigid_body_basis {
            Some(d) if d.nrows() == n && d.ncols() > 0 => project_out_basis_vec(d, v),
            _ => v.clone(),
        };
        match self.directions {
            ReflectedDirections::Hisd => {
                self.v = self.followed(&values, &vectors, k);
            }
            ReflectedDirections::Ggad if self.v.len() != k => {
                let mut start: Vec<Col<f64>> = orthonormal_to(&[], &g).into_iter().collect();
                start.extend(self.v.iter().map(&project));
                start.extend(vectors.iter().cloned());
                let mut v = gram_schmidt(&start);
                v.truncate(k);
                self.v = v;
                self.previous = None;
            }
            ReflectedDirections::Ggad => {
                let v: Vec<Col<f64>> = self.v.iter().map(&project).collect();
                let v = gram_schmidt(&v);
                let u: Vec<Col<f64>> = v.iter().map(|x| project(&(&h * x))).collect();
                let d = euler_directions(DirectionUpdate::Ggad, &v, &u);
                let scale = v
                    .iter()
                    .zip(&u)
                    .fold(1e-12_f64, |m, (a, b)| m.max(col_dot(a, b).abs()));
                let moved: Vec<Col<f64>> = (0..v.len())
                    .map(|i| {
                        let mut gamma = 0.5 / scale;
                        if let Some((before, d_before)) = &self.previous {
                            if before.len() == v.len() && d_before.len() == v.len() {
                                let dv = &v[i] - &before[i];
                                let dd = &d[i] - &d_before[i];
                                let (a, b) = (col_dot(&dv, &dd).abs(), col_dot(&dd, &dd));
                                if a > 0.0 && b > 0.0 {
                                    gamma = a / b;
                                }
                            }
                        }
                        let length = col_norm(&d[i]);
                        if gamma * length > 0.5 {
                            gamma = 0.5 / length;
                        }
                        &v[i] + &d[i] * gamma
                    })
                    .collect();
                let mut next = gram_schmidt(&moved);
                if next.len() < k {
                    next = self.followed(&values, &vectors, k);
                }
                self.previous = Some((v, d));
                self.v = next;
            }
        }
        if self.v.len() != k {
            return Err(MoptError::config(format!(
                "-opt {}: no {k} directions to reflect",
                self.name()
            )));
        }
        let mut f = g.clone();
        for v in &self.v {
            f -= v * (2.0 * col_dot(v, &g));
        }
        Ok(self.gradient_step(input, f))
    }

    fn reset(&mut self) {
        self.last = None;
        self.previous = None;
    }

    fn seed_followed_modes(&mut self, directions: &[Col<f64>]) -> bool {
        let v = gram_schmidt(directions);
        if v.is_empty() {
            return false;
        }
        self.v = v;
        true
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::opt::stepper::Stepper;

    /// `E = -x^2 - y^2 + 2 z^2 + x^4 + y^4 / 2`: an index-2 saddle at the origin.
    fn surface(p: &Col<f64>) -> (f64, Col<f64>, Mat<f64>) {
        let (x, y, z) = (p[0], p[1], p[2]);
        let e = -x * x - y * y + 2.0 * z * z + x.powi(4) + 0.5 * y.powi(4) + 0.3 * x * z;
        let g = Col::from_fn(3, |i| {
            [
                -2.0 * x + 4.0 * x.powi(3) + 0.3 * z,
                -2.0 * y + 2.0 * y.powi(3),
                4.0 * z + 0.3 * x,
            ][i]
        });
        let h = Mat::from_fn(3, 3, |i, j| {
            [
                [-2.0 + 12.0 * x * x, 0.0, 0.3],
                [0.0, -2.0 + 6.0 * y * y, 0.0],
                [0.3, 0.0, 4.0],
            ][i][j]
        });
        (e, g, h)
    }

    #[test]
    fn hisd_and_ggad_steps_with_exact_hessians_reach_the_index_two_saddle() {
        for method in ["hisd", "ggad", "hisd_block_bofill"] {
            let mut s = Stepper::new(method, 3, 2).unwrap().with_max_step(0.1);
            let mut x = Col::from_fn(3, |i| [0.3, -0.25, 0.1][i]);
            for _ in 0..200 {
                let (e, g, h) = surface(&x);
                if col_norm(&g) < 1e-10 {
                    break;
                }
                s.set_hessian(h).unwrap();
                x += s.propose(&x, e, &g).unwrap();
            }
            assert!(col_norm(&x) < 1e-8, "{method}: {x:?}");
        }
        // Ordinary optimization selects this downhill flow only during mode kill.
        let mut s = Stepper::new("hisd", 3, 0).unwrap();
        let x = Col::from_fn(3, |i| [0.3, -0.25, 0.1][i]);
        let (e, g, _) = surface(&x);
        assert!(s.propose(&x, e, &g).is_err());
    }
}
