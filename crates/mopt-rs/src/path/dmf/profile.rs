//! The energy along a path from the energies and their slopes at a few points, and where
//! the points of the next iteration belong.
//!
//! Between two points the energy is the cubic with their energies and slopes `dE/dt`
//! (`g . dx/dt`). Its highest point is the estimate of the transition state. The direct
//! MaxFlux method needs the energy near that point only (the share of a point in the
//! objective falls off as `exp(beta (E - E_max))`), so the points are moved towards it:
//! one to the top (odd numbers of points) and the others to where the energy is a
//! multiple of `spacing` below it, on both sides (PyDMF, `dmf.py:DirectMaxFlux`,
//! `update_teval`).
//!
//! MultiOptPy: no counterpart (its `-dmf` keeps the points at equal steps).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};
use crate::math::roots::polynomial_real_roots;

/// The energy along a path as cubic pieces.
#[derive(Debug, Clone)]
pub struct EnergyProfile {
    t: Vec<f64>,
    /// Coefficients of every piece in `u = (t - t_i) / (t_(i+1) - t_i)`, lowest first.
    pieces: Vec<[f64; 4]>,
    /// Where the energy is highest.
    pub t_max: f64,
    /// The energy there.
    pub e_max: f64,
}

impl EnergyProfile {
    /// The profile through the energies `energies` with the slopes `slopes` (`dE/dt`) at
    /// the points `t` (ascending, two at least).
    pub fn new(t: &[f64], energies: &[f64], slopes: &[f64]) -> MoptResult<Self> {
        let n = t.len();
        if n < 2 || energies.len() != n || slopes.len() != n {
            return Err(MoptError::config(
                "an energy profile needs an energy and a slope at two points at least",
            ));
        }
        if t.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(MoptError::config(
                "the points of an energy profile must ascend",
            ));
        }
        let pieces: Vec<[f64; 4]> = (0..n - 1)
            .map(|i| {
                let width = t[i + 1] - t[i];
                let (e0, e1) = (energies[i], energies[i + 1]);
                let (s0, s1) = (slopes[i] * width, slopes[i + 1] * width);
                [
                    e0,
                    s0,
                    3.0 * (e1 - e0) - 2.0 * s0 - s1,
                    2.0 * (e0 - e1) + s0 + s1,
                ]
            })
            .collect();
        // The highest point: the ends of the pieces and the maxima inside them.
        let mut t_max = t[0];
        let mut e_max = energies[0];
        for i in 0..n {
            if energies[i] > e_max {
                e_max = energies[i];
                t_max = t[i];
            }
        }
        for (i, c) in pieces.iter().enumerate() {
            for u in polynomial_real_roots(&[c[1], 2.0 * c[2], 3.0 * c[3]])? {
                if u > 0.0 && u < 1.0 && 2.0 * c[2] + 6.0 * c[3] * u < 0.0 {
                    let energy = value(c, u);
                    if energy > e_max {
                        e_max = energy;
                        t_max = t[i] + u * (t[i + 1] - t[i]);
                    }
                }
            }
        }
        Ok(Self {
            t: t.to_vec(),
            pieces,
            t_max,
            e_max,
        })
    }

    /// The energy at `t`.
    pub fn at(&self, t: f64) -> f64 {
        let last = self.pieces.len() - 1;
        let i = self
            .t
            .windows(2)
            .position(|pair| t < pair[1])
            .unwrap_or(last);
        let u = ((t - self.t[i]) / (self.t[i + 1] - self.t[i])).clamp(0.0, 1.0);
        value(&self.pieces[i], u)
    }

    /// Where the energy is `level`, on the way up to the highest point (the crossing
    /// nearest to it) and on the way down from it; `None` where the profile does not
    /// come down to `level` on that side.
    pub fn crossings(&self, level: f64) -> MoptResult<(Option<f64>, Option<f64>)> {
        let mut before: Option<f64> = None;
        let mut after: Option<f64> = None;
        for (i, c) in self.pieces.iter().enumerate() {
            for u in polynomial_real_roots(&[c[0] - level, c[1], c[2], c[3]])? {
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }
                let at = self.t[i] + u * (self.t[i + 1] - self.t[i]);
                if at < self.t_max {
                    before = Some(before.map_or(at, |b| b.max(at)));
                } else if at > self.t_max {
                    after = Some(after.map_or(at, |a| a.min(at)));
                }
            }
        }
        Ok((before, after))
    }

    /// Where `count` inner points belong: at the highest point (odd `count`) and where
    /// the energy is `spacing`, `2 spacing`, ... below it on both sides (even `count`:
    /// `spacing / 2`, `3 spacing / 2`, ...), ascending. `None` for a point whose level the
    /// profile does not reach on its side.
    pub fn targets(&self, count: usize, spacing: f64) -> MoptResult<Vec<Option<f64>>> {
        let half = count / 2;
        let mut before = Vec::with_capacity(half);
        let mut after = Vec::with_capacity(half);
        for k in 0..half {
            let steps = if count % 2 == 1 {
                (k + 1) as f64
            } else {
                k as f64 + 0.5
            };
            let (up, down) = self.crossings(self.e_max - steps * spacing)?;
            before.push(up);
            after.push(down);
        }
        let mut targets: Vec<Option<f64>> = before.into_iter().rev().collect();
        if count % 2 == 1 {
            targets.push(Some(self.t_max));
        }
        targets.extend(after);
        Ok(targets)
    }
}

fn value(c: &[f64; 4], u: f64) -> f64 {
    c[0] + u * (c[1] + u * (c[2] + u * c[3]))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_top_of_a_cubic_profile_is_found_and_the_points_gather_around_it() {
        // E(t) = t^2 (1.2 - t): a maximum of 0.256 at t = 0.8, exact for cubic pieces.
        let energy = |t: f64| t * t * (1.2 - t);
        let slope = |t: f64| 2.4 * t - 3.0 * t * t;
        let t = [0.0, 0.3, 0.55, 0.9, 1.0];
        let energies: Vec<f64> = t.iter().map(|&x| energy(x)).collect();
        let slopes: Vec<f64> = t.iter().map(|&x| slope(x)).collect();
        let profile = EnergyProfile::new(&t, &energies, &slopes).unwrap();
        assert!((profile.t_max - 0.8).abs() < 1e-10, "{}", profile.t_max);
        assert!((profile.e_max - 0.256).abs() < 1e-12);
        for x in [0.1, 0.42, 0.77, 0.95, 1.0] {
            assert!((profile.at(x) - energy(x)).abs() < 1e-12);
        }
        // Three points: the top and the two places 0.02 below it.
        let targets = profile.targets(3, 0.02).unwrap();
        assert_eq!(targets.len(), 3);
        let (low, top, high) = (
            targets[0].unwrap(),
            targets[1].unwrap(),
            targets[2].unwrap(),
        );
        assert!(low < top && top < high);
        assert!((top - 0.8).abs() < 1e-10);
        assert!((energy(low) - 0.236).abs() < 1e-10 && (energy(high) - 0.236).abs() < 1e-10);
        // Four points: half a spacing and one and a half below the top; the end at
        // t = 1 lies 0.056 below the top, so 0.09 below is reached on the way up only.
        let targets = profile.targets(4, 0.06).unwrap();
        assert_eq!(targets.len(), 4);
        assert!((energy(targets[0].unwrap()) - 0.166).abs() < 1e-10);
        assert!((energy(targets[1].unwrap()) - 0.226).abs() < 1e-10);
        assert!((energy(targets[2].unwrap()) - 0.226).abs() < 1e-10);
        assert!(targets[3].is_none());
        // A profile that only rises has its top at the end.
        let rising = EnergyProfile::new(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 1.0]).unwrap();
        assert_eq!(rising.t_max, 1.0);
        assert!(EnergyProfile::new(&[0.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]).is_err());
    }
}
