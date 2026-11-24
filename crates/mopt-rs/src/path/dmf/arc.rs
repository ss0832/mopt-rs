//! Arc length along a spline path, the parameter where a share of the length is reached,
//! and how both change with the control points.
//!
//! `s(t) = integral_0^t |dx/du| du` by eight-point Gauss-Legendre quadrature on every
//! segment of the spline (the speed is smooth there), and
//! `ds(t)/dc_b = integral_0^t B_b'(u) dx/du / |dx/du| du` the same way.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::spline::SplinePath;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_norm;

/// Nodes of the eight-point Gauss-Legendre rule on `[-1, 1]` (the positive half).
const NODES: [f64; 4] = [
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
/// Their weights.
const WEIGHTS: [f64; 4] = [
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

/// The points and weights of the rule on `[a, b]`.
fn rule(a: f64, b: f64) -> [(f64, f64); 8] {
    let (middle, half) = (0.5 * (a + b), 0.5 * (b - a));
    let mut out = [(0.0, 0.0); 8];
    for k in 0..4 {
        out[2 * k] = (middle - half * NODES[k], half * WEIGHTS[k]);
        out[2 * k + 1] = (middle + half * NODES[k], half * WEIGHTS[k]);
    }
    out
}

/// The arc length of a path.
#[derive(Debug, Clone)]
pub struct ArcLength<'a> {
    path: &'a SplinePath,
    /// The ends of the segments of the spline.
    breaks: Vec<f64>,
    /// Arc length up to every break.
    cumulative: Vec<f64>,
}

impl<'a> ArcLength<'a> {
    /// The arc length of `path`.
    pub fn new(path: &'a SplinePath) -> MoptResult<Self> {
        let breaks = path.basis.breaks();
        let mut cumulative = vec![0.0];
        for pair in breaks.windows(2) {
            let piece: f64 = rule(pair[0], pair[1])
                .iter()
                .map(|(u, w)| w * col_norm(&path.velocity(*u)))
                .sum();
            let last = cumulative.last().copied().unwrap_or(0.0);
            cumulative.push(last + piece);
        }
        let found = Self {
            path,
            breaks,
            cumulative,
        };
        if !(found.length() > 0.0 && found.length().is_finite()) {
            return Err(MoptError::config(
                "the path has no length: its two ends are the same point",
            ));
        }
        Ok(found)
    }

    /// The length of the whole path.
    pub fn length(&self) -> f64 {
        self.cumulative.last().copied().unwrap_or(0.0)
    }

    /// The segment that contains `t`.
    fn segment(&self, t: f64) -> usize {
        let last = self.breaks.len() - 2;
        self.breaks[1..]
            .iter()
            .position(|end| t < *end)
            .unwrap_or(last)
            .min(last)
    }

    /// The arc length from the start to `t`.
    pub fn at(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        let j = self.segment(t);
        let start = self.breaks[j];
        self.cumulative[j]
            + rule(start, t)
                .iter()
                .map(|(u, w)| w * col_norm(&self.path.velocity(*u)))
                .sum::<f64>()
    }

    /// The parameter where the arc length is `share` of the whole length.
    pub fn parameter(&self, share: f64) -> f64 {
        let share = share.clamp(0.0, 1.0);
        if share <= 0.0 {
            return 0.0;
        }
        if share >= 1.0 {
            return 1.0;
        }
        let target = share * self.length();
        let j = self
            .cumulative
            .windows(2)
            .position(|pair| target <= pair[1])
            .unwrap_or(self.breaks.len() - 2);
        let (mut low, mut high) = (self.breaks[j], self.breaks[j + 1]);
        let span = self.cumulative[j + 1] - self.cumulative[j];
        let mut t = if span > 0.0 {
            low + (high - low) * (target - self.cumulative[j]) / span
        } else {
            0.5 * (low + high)
        };
        // Newton steps kept in the bracket, bisection where they leave it.
        for _ in 0..60 {
            let miss = self.at(t) - target;
            if miss.abs() <= 1e-14 * self.length() {
                break;
            }
            if miss > 0.0 {
                high = t;
            } else {
                low = t;
            }
            let speed = col_norm(&self.path.velocity(t));
            let newton = if speed > 0.0 {
                t - miss / speed
            } else {
                f64::NAN
            };
            t = if newton > low && newton < high {
                newton
            } else {
                0.5 * (low + high)
            };
        }
        t
    }

    /// `ds(t)/dc_b` for the inner control points `b`.
    pub fn derivative(&self, t: f64) -> Vec<Col<f64>> {
        let count = self.path.control.len();
        let n = self.path.dimension();
        let mut out: Vec<Col<f64>> = (0..count - 2).map(|_| Col::zeros(n)).collect();
        let t = t.clamp(0.0, 1.0);
        let j = self.segment(t);
        let mut add = |a: f64, b: f64| {
            for (u, w) in rule(a, b) {
                let velocity = self.path.velocity(u);
                let speed = col_norm(&velocity);
                if speed <= 0.0 {
                    continue;
                }
                let derivatives = self.path.basis.derivatives(u);
                for (k, of_point) in out.iter_mut().enumerate() {
                    let d = derivatives[k + 1];
                    if d != 0.0 {
                        *of_point += &velocity * (w * d / speed);
                    }
                }
            }
        };
        for k in 0..j {
            add(self.breaks[k], self.breaks[k + 1]);
        }
        add(self.breaks[j], t);
        out
    }

    /// How uneven the speed of the parameter is: the largest speed over the smallest at
    /// the points of the rule.
    pub fn unevenness(&self) -> f64 {
        let (low, high) = self
            .breaks
            .windows(2)
            .flat_map(|pair| rule(pair[0], pair[1]))
            .map(|(u, _)| col_norm(&self.path.velocity(u)))
            .fold((f64::INFINITY, 0.0_f64), |(l, h), v| (l.min(v), h.max(v)));
        if low > 0.0 {
            high / low
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    #[test]
    fn lengths_parameters_and_their_derivatives() {
        // A straight path at constant speed: length 5, parameter = share.
        let line = SplinePath::through(&[col(&[0.0, 0.0]), col(&[3.0, 4.0])], 3, 4).unwrap();
        let arc = ArcLength::new(&line).unwrap();
        assert!((arc.length() - 5.0).abs() < 1e-12);
        for share in [0.0, 0.1, 0.37, 0.5, 0.99, 1.0] {
            assert!((arc.parameter(share) - share).abs() < 1e-12, "{share}");
        }
        assert!((arc.unevenness() - 1.0).abs() < 1e-9);

        // A bent path whose control points are moved: lengths against fine sums of chords,
        // the parameter of a share, and the derivative against differences.
        let chain = [col(&[0.0, 0.0]), col(&[1.0, 1.2]), col(&[2.5, 0.3])];
        let mut path = SplinePath::through(&chain, 3, 4).unwrap();
        path.control[2][0] += 0.3;
        let arc = ArcLength::new(&path).unwrap();
        // Sums of chords fall short of the length by the square of their length: the
        // extrapolation from two of them.
        let sum = |count: u32| -> f64 {
            (0..count)
                .map(|k| {
                    let (a, b) = (
                        f64::from(k) / f64::from(count),
                        f64::from(k + 1) / f64::from(count),
                    );
                    col_norm(&(&path.at(b) - &path.at(a)))
                })
                .sum()
        };
        let chords = (4.0 * sum(8000) - sum(4000)) / 3.0;
        // Eight points of the rule on a segment: 4e-8 of the length on this bend.
        assert!(
            (arc.length() - chords).abs() < 1e-6 * chords,
            "{} {chords}",
            arc.length()
        );
        assert!(arc.unevenness() > 1.05);
        for share in [0.2, 0.5, 0.8] {
            let t = arc.parameter(share);
            assert!((arc.at(t) / arc.length() - share).abs() < 1e-12);
        }
        let t = 0.6;
        let derivative = arc.derivative(t);
        assert_eq!(derivative.len(), 5);
        for (b, of_point) in derivative.iter().enumerate() {
            for (k, analytic) in of_point.iter().enumerate() {
                let mut up = path.clone();
                up.control[b + 1][k] += 1e-6;
                let mut down = path.clone();
                down.control[b + 1][k] -= 1e-6;
                let numeric = (ArcLength::new(&up).unwrap().at(t)
                    - ArcLength::new(&down).unwrap().at(t))
                    / 2e-6;
                assert!(
                    (numeric - analytic).abs() < 1e-8,
                    "{b} {k}: {numeric} {analytic}"
                );
            }
        }
        assert!(
            ArcLength::new(&SplinePath::through(&[col(&[1.0]), col(&[2.0])], 3, 4).unwrap())
                .is_ok()
        );
    }
}
