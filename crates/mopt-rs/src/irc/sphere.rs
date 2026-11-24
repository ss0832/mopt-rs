//! Minimization of the energy on a hypersphere `|q - c| = r` in mass-weighted coordinates.
//!
//! Two users, which differ in the centre only:
//! - GS2 ([`super::gs2`]): the centre is the pivot point half a step down the gradient from
//!   the last path point, the radius half the step size;
//! - the correction of the first step ([`super::first_step`]): the centre is the transition
//!   state itself, the radius the length of the first displacement.
//!
//! The minimization is local: it goes downhill on the sphere from its starting point and
//! ends in the minimum of that side. (A sphere around a transition state has a minimum on
//! either side of it; the lowest point of the quadratic model on the whole sphere, which a
//! multiplier below the lowest eigenvalue of the Hessian gives, can lie on the other side.)
//!
//! Each iteration takes the energy and the gradient at the current point `q = c + p`,
//! updates the Hessian `H` (Bofill, valid where the curvature is negative), and takes a
//! Newton step within the tangent space of the sphere: with the gradient `g_t` within the
//! sphere and the eigenvalues `b_i` of the Hessian projected onto the tangent space, the
//! curvatures of the energy on the sphere are `b_i - (g . p) / r^2`, and the step is
//! `-sum_i v_i g_i / (b_i - (g . p) / r^2 + mu)` with the smallest `mu >= 0` that makes all
//! denominators positive and the step no longer than the trust length (half the radius at
//! most). The new point is the step taken back to the sphere along the radius. A point
//! with a higher energy than the last one is not taken: the trust length is halved and the
//! step repeated. Modes without curvature (`|b_i| < 1e-8`: the rigid-body motion and the
//! radius itself) are not moved along. The minimization ends when the gradient within the
//! sphere is small, when the next step is short, or after the largest number of
//! evaluations.
//!
//! References: C. Gonzalez and H. B. Schlegel, J. Chem. Phys. 1989, 90, 2154 and J. Phys.
//! Chem. 1990, 94, 5523 (the constrained step of GS2).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::lqa::SMALL_EIGENVALUE;
use super::settings::SphereSettings;
use super::types::{dot, norm, rms};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;
use crate::opt::update::formulas::bofill;

/// Longest step within the sphere, as a share of the radius.
pub const TRUST_SHARE: f64 = 0.5;

/// State of a minimization after a point was told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SphereStatus {
    /// The next point is to be evaluated.
    Continue,
    /// The point just evaluated is the minimum on the sphere.
    Converged,
    /// The evaluations are used up; the point just evaluated is the result.
    Exhausted,
}

/// The best point so far.
#[derive(Debug, Clone)]
struct Accepted {
    q: Col<f64>,
    gradient: Col<f64>,
    energy: f64,
}

/// Minimization on a hypersphere, one point at a time.
#[derive(Debug, Clone)]
pub struct SphereMinimizer {
    center: Col<f64>,
    radius: f64,
    q: Col<f64>,
    hessian: Mat<f64>,
    settings: SphereSettings,
    evaluated: Option<(Col<f64>, Col<f64>)>,
    accepted: Option<Accepted>,
    trust: f64,
    iterations: usize,
    tangential_rms: f64,
    /// Whether the point to evaluate is the best one, asked for again at the end of the
    /// evaluations: its value ends the minimization whatever it is.
    revisiting: bool,
}

/// `v` scaled to the length `radius`.
fn on_sphere(v: &Col<f64>, radius: f64) -> MoptResult<Col<f64>> {
    let size = norm(v);
    if size == 0.0 {
        return Err(MoptError::config(
            "hypersphere minimization: the starting point is the centre of the sphere",
        ));
    }
    Ok(v * (radius / size))
}

impl SphereMinimizer {
    /// Minimize on the sphere of `radius` around `center`, starting from `start` (moved
    /// onto the sphere along the line from the centre) with the mass-weighted Hessian
    /// `hessian` of the surface that is followed.
    pub fn new(
        center: Col<f64>,
        radius: f64,
        start: &Col<f64>,
        hessian: Mat<f64>,
        settings: SphereSettings,
    ) -> MoptResult<Self> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(MoptError::config(
                "hypersphere minimization: the radius must be positive",
            ));
        }
        let q = &center + on_sphere(&(start - &center), radius)?;
        Ok(Self {
            center,
            radius,
            q,
            hessian,
            settings,
            evaluated: None,
            accepted: None,
            trust: TRUST_SHARE * radius,
            iterations: 0,
            tangential_rms: f64::NAN,
            revisiting: false,
        })
    }

    /// The point (mass-weighted) to evaluate next; after the end, the result.
    pub fn point(&self) -> &Col<f64> {
        &self.q
    }

    /// Points told so far.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// RMS of the gradient within the sphere at the best point so far.
    pub fn tangential_rms(&self) -> f64 {
        self.tangential_rms
    }

    /// The Hessian as updated so far.
    pub fn into_hessian(self) -> Mat<f64> {
        self.hessian
    }

    /// The energy and the mass-weighted gradient of the surface at [`Self::point`].
    pub fn tell(&mut self, energy: f64, gradient: &Col<f64>) -> MoptResult<SphereStatus> {
        let finite = energy.is_finite() && gradient.iter().all(|v| v.is_finite());
        if gradient.nrows() != self.q.nrows() || !finite {
            return Err(MoptError::calc(
                "hypersphere minimization: the surface is not finite or of another size",
            ));
        }
        self.iterations += 1;
        let revisited = std::mem::take(&mut self.revisiting);
        if revisited && self.accepted.as_ref().is_some_and(|a| energy > a.energy) {
            // The best point evaluated once more after the last evaluation, with a value a
            // little above the one it had (noise of the surface): it is the result all the
            // same, which would otherwise be rejected and asked for again without end. The
            // point stays the best one, and the value just told belongs to it. (A value that
            // is not higher goes on as before.)
            self.evaluated = Some((self.q.clone(), gradient.clone()));
            return Ok(SphereStatus::Exhausted);
        }
        if let Some((q0, g0)) = &self.evaluated {
            let s = &self.q - q0;
            let y = gradient - g0;
            if norm(&s) > 0.0 {
                let change = bofill(&self.hessian, &s, &y, 1.0);
                let finite = (0..change.ncols())
                    .all(|j| (0..change.nrows()).all(|i| change[(i, j)].is_finite()));
                if finite {
                    self.hessian += change;
                }
            }
        }
        self.evaluated = Some((self.q.clone(), gradient.clone()));
        let rejected = self.accepted.as_ref().is_some_and(|a| energy > a.energy);
        if rejected {
            self.trust *= 0.5;
        } else {
            if self.accepted.is_some() {
                self.trust = (2.0 * self.trust).min(TRUST_SHARE * self.radius);
            }
            self.accepted = Some(Accepted {
                q: self.q.clone(),
                gradient: gradient.clone(),
                energy,
            });
        }
        let Some(best) = self.accepted.clone() else {
            return Err(MoptError::config("hypersphere minimization: no point"));
        };
        let p = &best.q - &self.center;
        let radial = &p * (1.0 / self.radius);
        let slope = dot(&best.gradient, &radial);
        let tangential = &best.gradient - &radial * slope;
        self.tangential_rms = rms(&tangential);
        if !rejected && self.tangential_rms < self.settings.gradient_tolerance {
            return Ok(SphereStatus::Converged);
        }
        let step = self.tangent_step(&radial, &tangential, slope / self.radius)?;
        let moved = on_sphere(&(&p + &step), self.radius)?;
        let length = norm(&(&moved - &p));
        if !rejected && length < self.settings.step_share * self.radius {
            return Ok(SphereStatus::Converged);
        }
        if self.iterations >= self.settings.max_iterations {
            if rejected {
                // The result is the best point: it is evaluated once more, so that the
                // value of the last evaluation belongs to it.
                self.q = best.q;
                self.revisiting = true;
                return Ok(SphereStatus::Continue);
            }
            return Ok(SphereStatus::Exhausted);
        }
        if length == 0.0 {
            // Nothing to gain along any direction of the sphere.
            self.q = best.q;
            return Ok(if rejected {
                SphereStatus::Continue
            } else {
                SphereStatus::Converged
            });
        }
        self.q = &self.center + moved;
        Ok(SphereStatus::Continue)
    }

    /// The Newton step within the tangent space at the point with the unit radius vector
    /// `radial` and the gradient `tangential` within the sphere; `shift` is `(g . p) / r^2`,
    /// by which the curvature of the sphere lowers those of the surface.
    fn tangent_step(
        &self,
        radial: &Col<f64>,
        tangential: &Col<f64>,
        shift: f64,
    ) -> MoptResult<Col<f64>> {
        let n = radial.nrows();
        // P H P with P = 1 - radial radial^T.
        let hr = &self.hessian * radial;
        let rhr = dot(radial, &hr);
        let projected = Mat::from_fn(n, n, |i, j| {
            self.hessian[(i, j)] - radial[i] * hr[j] - hr[i] * radial[j]
                + rhr * radial[i] * radial[j]
        });
        let (values, vectors) = symmetric_eigen(&projected)?;
        let modes: Vec<(f64, f64, usize)> = (0..n)
            .filter(|&k| values[k].abs() >= SMALL_EIGENVALUE)
            .map(|k| {
                let slope: f64 = (0..n).map(|i| vectors[(i, k)] * tangential[i]).sum();
                (values[k] - shift, slope, k)
            })
            .collect();
        if modes.is_empty() {
            return Ok(Col::zeros(n));
        }
        let lowest = modes.iter().map(|m| m.0).fold(f64::INFINITY, f64::min);
        let length = |mu: f64| -> f64 {
            libm::sqrt(
                modes
                    .iter()
                    .map(|(c, g, _)| {
                        let d = g / (c + mu);
                        d * d
                    })
                    .sum::<f64>(),
            )
        };
        // The smallest shift that makes every curvature positive.
        let scale = lowest.abs().max(1e-6);
        let floor = if lowest > 0.0 {
            0.0
        } else {
            -lowest + 1e-6 * scale
        };
        let mut mu = floor;
        if length(mu) > self.trust {
            let mut high = floor.max(scale);
            let mut doublings = 0;
            while length(high) > self.trust && doublings < 200 {
                high *= 2.0;
                doublings += 1;
            }
            let mut low = floor;
            for _ in 0..100 {
                let mid = 0.5 * (low + high);
                if length(mid) > self.trust {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            mu = high;
        }
        let mut step = Col::<f64>::zeros(n);
        for (c, g, k) in &modes {
            let d = -g / (c + mu);
            for i in 0..n {
                step[i] += d * vectors[(i, *k)];
            }
        }
        Ok(step)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn matrix() -> Mat<f64> {
        Mat::from_fn(3, 3, |i, j| {
            [[-0.4, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.2, 1.5]][i][j]
        })
    }

    /// E = 0.5 x^T A x + b.x with a saddle.
    fn surface(q: &Col<f64>) -> (f64, Col<f64>) {
        let a = matrix();
        let b = Col::from_fn(3, |i| [0.02, -0.05, 0.03][i]);
        let g = &a * q + &b;
        (0.5 * dot(q, &(&a * q)) + dot(&b, q), g)
    }

    fn minimize(
        center: &Col<f64>,
        radius: f64,
        start: &Col<f64>,
        hessian: Mat<f64>,
    ) -> SphereMinimizer {
        let settings = SphereSettings {
            max_iterations: 80,
            gradient_tolerance: 1e-9,
            step_share: 1e-12,
        };
        let mut m = SphereMinimizer::new(center.clone(), radius, start, hessian, settings).unwrap();
        let mut status = SphereStatus::Continue;
        while status == SphereStatus::Continue {
            let (e, g) = surface(m.point());
            status = m.tell(e, &g).unwrap();
            assert!((norm(&(m.point() - center)) - radius).abs() < 1e-12);
        }
        assert_eq!(status, SphereStatus::Converged, "{} points", m.iterations());
        m
    }

    #[test]
    fn finds_the_minimum_of_its_side_of_the_sphere() {
        // Around the saddle of the surface: the energy falls along +x and along -x, and
        // the sphere has a minimum on either side. From a poor Hessian (the updates have to
        // find the curvature) each start ends on its own side.
        let saddle = {
            // A x + b = 0
            let (_, g0) = surface(&Col::zeros(3));
            let a = matrix();
            let inverse = crate::math::linalg::inverse_small(&a).unwrap();
            &inverse * (&g0 * -1.0)
        };
        let identity = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let mut ends = Vec::new();
        for sign in [1.0, -1.0] {
            let start = &saddle + Col::from_fn(3, |i| [sign * 0.35, 0.3, -0.2][i]);
            let m = minimize(&saddle, 0.5, &start, identity.clone());
            let end = m.point().clone();
            assert!((end[0] - saddle[0]) * sign > 0.3, "{sign}: {end:?}");
            // A minimum on the sphere: lower than its neighbours on the sphere.
            let (best, _) = surface(&end);
            assert!(best < surface(&start).0);
            for k in 0..3 {
                for h in [1e-3, -1e-3] {
                    let mut near = &end - &saddle;
                    near[k] += h;
                    let near = &saddle + on_sphere(&near, 0.5).unwrap();
                    assert!(surface(&near).0 >= best - 1e-12);
                }
            }
            ends.push(end);
        }
        assert!(norm(&(&ends[0] - &ends[1])) > 0.6);
    }

    #[test]
    fn a_best_point_that_evaluates_higher_again_ends_the_evaluations() {
        // Every evaluation a little higher than the surface: the steps after the best point are
        // rejected, and so is the best point evaluated once more at the end (the noise of a
        // surface). The evaluations end there, bounded here by a counter.
        let settings = SphereSettings {
            max_iterations: 5,
            gradient_tolerance: 1e-12,
            step_share: 1e-14,
        };
        let center = Col::<f64>::zeros(3);
        let start = Col::from_fn(3, |i| [0.2, 0.1, -0.1][i]);
        let mut m = SphereMinimizer::new(center, 0.3, &start, matrix(), settings).unwrap();
        let mut told: Vec<(Col<f64>, f64)> = Vec::new();
        let mut status = SphereStatus::Continue;
        while status == SphereStatus::Continue && told.len() < 50 {
            let (e, g) = surface(m.point());
            let noisy = e + 1e-9 * (told.len() as f64 + 1.0);
            told.push((m.point().clone(), noisy));
            status = m.tell(noisy, &g).unwrap();
        }
        assert_eq!(
            status,
            SphereStatus::Exhausted,
            "{} evaluations",
            told.len()
        );
        assert_eq!(told.len(), 6);
        // The result is the point of the lowest value of the evaluations before the last
        // (which is that point evaluated again), and the last evaluation belongs to it.
        let (best, _) = told[..told.len() - 1]
            .iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert_eq!(m.point(), best);
        assert_eq!(&told[told.len() - 1].0, best);
    }

    #[test]
    fn exact_hessian_converges_quickly_and_bad_input_fails() {
        let center = Col::<f64>::zeros(3);
        let start = Col::from_fn(3, |i| [0.2, 0.1, -0.1][i]);
        let m = minimize(&center, 0.3, &start, matrix());
        assert!(m.iterations() <= 8, "{}", m.iterations());
        let settings = SphereSettings::default();
        assert!(SphereMinimizer::new(center.clone(), 0.3, &center, matrix(), settings).is_err());
        assert!(SphereMinimizer::new(center.clone(), 0.0, &start, matrix(), settings).is_err());
        let mut m = SphereMinimizer::new(center, 0.3, &start, matrix(), settings).unwrap();
        assert!(m.tell(f64::NAN, &start).is_err());
    }
}
