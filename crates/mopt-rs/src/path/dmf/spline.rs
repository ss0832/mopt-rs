//! A path as a B-spline of its parameter `t` in `[0, 1]`.
//!
//! `x(t) = sum_b B_b(t) c_b` with the B-splines `B_b` of degree `p` on `segments` equal
//! intervals and knots repeated at both ends (the first and the last control point are the
//! ends of the path), as in the authors' program PyDMF (`dmf.py:VariationalPathOpt`:
//! degree 3, 4 segments, 7 control points).
//!
//! MultiOptPy: `MEP/pathopt_dmf_force.py` (not ported; this is written from the method).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_norm, safe_inverse};

/// The B-splines of a path.
#[derive(Debug, Clone, PartialEq)]
pub struct Basis {
    degree: usize,
    knots: Vec<f64>,
}

impl Basis {
    /// B-splines of degree `degree` on `segments` equal intervals of `[0, 1]`.
    pub fn new(degree: usize, segments: usize) -> MoptResult<Self> {
        if degree == 0 || segments == 0 {
            return Err(MoptError::config(
                "a spline path needs a degree and a number of segments of one at least",
            ));
        }
        let mut knots = vec![0.0; degree];
        knots.extend((0..=segments).map(|k| k as f64 / segments as f64));
        knots.extend(std::iter::repeat_n(1.0, degree));
        Ok(Self { degree, knots })
    }

    /// The number of B-splines (control points).
    pub fn len(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    /// Whether there is no B-spline (never, see [`Basis::new`]).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The Greville abscissae: where on the path every control point acts most,
    /// `(u_(b+1) + ... + u_(b+p)) / p`.
    pub fn greville(&self) -> Vec<f64> {
        let p = self.degree;
        (0..self.len())
            .map(|b| self.knots[b + 1..=b + p].iter().sum::<f64>() / p as f64)
            .collect()
    }

    /// The ends of the segments, ascending (the distinct knots).
    pub fn breaks(&self) -> Vec<f64> {
        self.knots[self.degree..self.knots.len() - self.degree].to_vec()
    }

    /// The B-splines of degree `degree` at `t` (Cox-de Boor; `0 / 0` counts as `0`).
    fn of_degree(&self, degree: usize, t: f64) -> Vec<f64> {
        let u = &self.knots;
        let t = t.clamp(0.0, 1.0);
        // The last interval of positive length includes its right end.
        let last = u.len() - 1 - self.degree;
        let mut values: Vec<f64> = (0..u.len() - 1)
            .map(|i| {
                let inside = u[i] <= t && (t < u[i + 1] || (i + 1 == last && t <= u[i + 1]));
                if inside && u[i] < u[i + 1] {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        for d in 1..=degree {
            let next: Vec<f64> = (0..u.len() - 1 - d)
                .map(|i| {
                    let mut value = 0.0;
                    let left = u[i + d] - u[i];
                    if left > 0.0 {
                        value += (t - u[i]) / left * values[i];
                    }
                    let right = u[i + d + 1] - u[i + 1];
                    if right > 0.0 {
                        value += (u[i + d + 1] - t) / right * values[i + 1];
                    }
                    value
                })
                .collect();
            values = next;
        }
        values
    }

    /// The values of the B-splines at `t`.
    pub fn values(&self, t: f64) -> Vec<f64> {
        self.of_degree(self.degree, t)
    }

    /// The first derivatives of the B-splines at `t`.
    pub fn derivatives(&self, t: f64) -> Vec<f64> {
        let (p, u) = (self.degree, &self.knots);
        let lower = self.of_degree(p - 1, t);
        (0..self.len())
            .map(|i| {
                let mut value = 0.0;
                let left = u[i + p] - u[i];
                if left > 0.0 {
                    value += p as f64 / left * lower[i];
                }
                let right = u[i + p + 1] - u[i + 1];
                if right > 0.0 {
                    value -= p as f64 / right * lower[i + 1];
                }
                value
            })
            .collect()
    }
}

/// A path: the B-splines and their control points.
#[derive(Debug, Clone)]
pub struct SplinePath {
    /// The B-splines.
    pub basis: Basis,
    /// The control points; the first and the last are the ends of the path.
    pub control: Vec<Col<f64>>,
}

impl SplinePath {
    /// The point of the path at `t`.
    pub fn at(&self, t: f64) -> Col<f64> {
        combine(&self.basis.values(t), &self.control)
    }

    /// The derivative of the path with respect to `t` at `t`.
    pub fn velocity(&self, t: f64) -> Col<f64> {
        combine(&self.basis.derivatives(t), &self.control)
    }

    /// The number of variables of a point.
    pub fn dimension(&self) -> usize {
        self.control.first().map_or(0, Col::nrows)
    }

    /// The path through the points `chain` (two at least): they are joined by straight
    /// lines, `4 segments + 1` points at equal distances along that line are taken, and
    /// the inner control points are the least-squares fit to them (as PyDMF starts).
    pub fn through(chain: &[Col<f64>], degree: usize, segments: usize) -> MoptResult<Self> {
        let (Some(first), Some(last)) = (chain.first(), chain.last()) else {
            return Err(MoptError::config("a path needs two points at least"));
        };
        if chain.len() < 2 {
            return Err(MoptError::config("a path needs two points at least"));
        }
        let basis = Basis::new(degree, segments)?;
        let count = basis.len();
        let samples = 4 * segments + 1;
        let mut lengths = vec![0.0; chain.len()];
        for k in 1..chain.len() {
            lengths[k] = lengths[k - 1] + col_norm(&(&chain[k] - &chain[k - 1]));
        }
        let total = lengths[chain.len() - 1];
        if total <= 0.0 {
            return Err(MoptError::config(
                "the two ends of the path are the same point",
            ));
        }
        let n = first.nrows();
        let inner = count - 2;
        // Normal equations of the fit of the inner control points.
        let mut normal = Mat::<f64>::zeros(inner, inner);
        let mut rhs: Vec<Col<f64>> = (0..inner).map(|_| Col::zeros(n)).collect();
        let mut segment = 0;
        for j in 0..samples {
            let t = j as f64 / (samples - 1) as f64;
            let target = t * total;
            while segment + 2 < chain.len() && lengths[segment + 1] < target {
                segment += 1;
            }
            let span = lengths[segment + 1] - lengths[segment];
            let share = if span > 0.0 {
                ((target - lengths[segment]) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let point = &chain[segment] * (1.0 - share) + &chain[segment + 1] * share;
            let b = basis.values(t);
            let rest = &point - first * b[0] - last * b[count - 1];
            for r in 0..inner {
                for c in 0..inner {
                    normal[(r, c)] += b[r + 1] * b[c + 1];
                }
                rhs[r] += &rest * b[r + 1];
            }
        }
        let inverse = safe_inverse(&normal, 1e-12)?;
        let mut control = Vec::with_capacity(count);
        control.push(first.clone());
        for r in 0..inner {
            let mut point = Col::<f64>::zeros(n);
            for c in 0..inner {
                point += &rhs[c] * inverse[(r, c)];
            }
            control.push(point);
        }
        control.push(last.clone());
        Ok(Self { basis, control })
    }

    /// `count` points of the path at equal steps of `t`, the ends among them.
    pub fn images(&self, count: usize) -> Vec<Col<f64>> {
        let count = count.max(2);
        (0..count)
            .map(|k| self.at(k as f64 / (count - 1) as f64))
            .collect()
    }
}

fn combine(weights: &[f64], points: &[Col<f64>]) -> Col<f64> {
    let n = points.first().map_or(0, Col::nrows);
    let mut out = Col::<f64>::zeros(n);
    for (weight, point) in weights.iter().zip(points) {
        if *weight != 0.0 {
            out += point * *weight;
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    #[test]
    fn the_splines_sum_to_one_and_their_derivatives_are_those_of_differences() {
        let basis = Basis::new(3, 4).unwrap();
        assert_eq!(basis.len(), 7);
        for k in 0..=40 {
            let t = f64::from(k) / 40.0;
            let values = basis.values(t);
            assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-14, "{t}");
            assert!(values.iter().all(|v| *v >= -1e-15));
            let derivatives = basis.derivatives(t);
            assert!(derivatives.iter().sum::<f64>().abs() < 1e-12, "{t}");
            if k > 0 && k < 40 {
                let (up, down) = (basis.values(t + 1e-6), basis.values(t - 1e-6));
                for b in 0..7 {
                    let numeric = (up[b] - down[b]) / 2e-6;
                    assert!((numeric - derivatives[b]).abs() < 1e-7, "{t} {b}");
                }
            }
        }
        // The ends of the path are the first and the last control point.
        assert_eq!(basis.values(0.0)[0], 1.0);
        assert_eq!(basis.values(1.0)[6], 1.0);
    }

    #[test]
    fn a_path_through_points_on_a_line_is_that_line_at_constant_speed() {
        let chain = [col(&[0.0, 0.0]), col(&[1.0, 2.0]), col(&[4.0, 8.0])];
        let path = SplinePath::through(&chain, 3, 4).unwrap();
        for k in 0..=10 {
            let t = f64::from(k) / 10.0;
            let point = path.at(t);
            assert!((point[0] - 4.0 * t).abs() < 1e-10 && (point[1] - 8.0 * t).abs() < 1e-10);
            let velocity = path.velocity(t);
            assert!((velocity[0] - 4.0).abs() < 1e-9 && (velocity[1] - 8.0).abs() < 1e-9);
        }
        let images = path.images(5);
        assert_eq!(images.len(), 5);
        assert!((images[2][0] - 2.0).abs() < 1e-10);
        // A bend is followed: the fitted path passes near the middle point.
        let bent = [col(&[0.0, 0.0]), col(&[1.0, 1.0]), col(&[2.0, 0.0])];
        let path = SplinePath::through(&bent, 3, 4).unwrap();
        let top = path.at(0.5);
        assert!((top[0] - 1.0).abs() < 1e-10 && top[1] > 0.7 && top[1] < 1.0);
    }
}
