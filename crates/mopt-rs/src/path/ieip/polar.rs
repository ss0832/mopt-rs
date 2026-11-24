//! MultiOptPy's polar coordinates of a structure about a centre, as a coordinate system: the
//! hyperspherical coordinates of the whole `3N` displacement `d = x - c`, the length `r` and
//! `3N - 1` angles (MultiOptPy `cart2polar(x, c)`),
//!
//! `d_0 = r cos t_1`, `d_i = r sin t_1 ... sin t_i cos t_(i+1)` for `0 < i < n - 1`,
//! `d_(n-1) = r sin t_1 ... sin t_(n-2) sin t_(n-1)`,
//!
//! with `t_1 ... t_(n-2)` in `[0, pi]` and `t_(n-1)` in `[0, 2 pi)`. The relaxation of the
//! images of [`super`] keeps `r` (the distance from where an image was) and moves the angles.
//! The Wilson matrix is the analytic `dq/dx`; gradients and steps go through the shared
//! machinery of [`crate::coord::system::transform`], as for every other coordinate system.
//!
//! A temporary implementation within the iEIP module: to be replaced by
//! `coord::system::Hyperspherical` (centre: a given structure) at the merge.
//!
//! MultiOptPy: `Coordinate/polar_coordinate.py` (`cart2polar`, `polar2cart`). Its gradient
//! (`cart_grad_2_polar_grad`) takes the Jacobian at the coordinates of the structure about the
//! origin instead of the displacement from the centre, and builds the first column of that
//! Jacobian from `cart2polar` where `polar2cart` is meant; here the gradient is that of the
//! coordinates themselves (the chain rule through the Wilson matrix).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::{Col, Mat};

use crate::coord::system::CoordinateSystem;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

/// Length below which a vector counts as zero (its angles are all zero).
const ZERO_LENGTH: f64 = 1e-9;
/// Smallest length of a tail of the vector that is divided by.
const TINY_TAIL: f64 = 1e-12;

/// `(r, t_1, ..., t_(n-1))` of `d` (MultiOptPy `cart2polar`).
pub fn from_cartesian(d: &Col<f64>) -> Col<f64> {
    let n = d.nrows();
    let mut p = Col::<f64>::zeros(n);
    if n == 0 {
        return p;
    }
    let tail = tails(d);
    let r = tail[0];
    p[0] = r;
    if r < ZERO_LENGTH || n == 1 {
        return p;
    }
    for i in 0..n.saturating_sub(2) {
        p[i + 1] = if tail[i] == 0.0 {
            0.0
        } else {
            libm::acos((d[i] / tail[i]).clamp(-1.0, 1.0))
        };
    }
    let mut last = libm::atan2(d[n - 1], d[n - 2]);
    if last < 0.0 {
        last += 2.0 * PI;
    }
    p[n - 1] = last;
    p
}

/// The vector of the coordinates `p` (MultiOptPy `polar2cart`).
pub fn to_cartesian(p: &Col<f64>) -> Col<f64> {
    let n = p.nrows();
    let mut x = Col::<f64>::zeros(n);
    if n == 0 {
        return x;
    }
    let r = p[0];
    if r.abs() < ZERO_LENGTH {
        return x;
    }
    if n == 1 {
        x[0] = r;
        return x;
    }
    // prod = r sin t_1 ... sin t_i
    let mut prod = r;
    for i in 0..n - 2 {
        x[i] = prod * libm::cos(p[i + 1]);
        prod *= libm::sin(p[i + 1]);
    }
    x[n - 2] = prod * libm::cos(p[n - 1]);
    x[n - 1] = prod * libm::sin(p[n - 1]);
    x
}

/// Lengths of the tails `d_i .. d_(n-1)` (without overflow), with a zero at the end.
fn tails(d: &Col<f64>) -> Vec<f64> {
    let n = d.nrows();
    let mut tail = vec![0.0; n + 1];
    for i in (0..n).rev() {
        tail[i] = libm::hypot(tail[i + 1], d[i]);
    }
    tail
}

/// `dq/dd` of the coordinates of [`from_cartesian`] at `d`.
pub fn jacobian(d: &Col<f64>) -> Mat<f64> {
    let n = d.nrows();
    let mut b = Mat::<f64>::zeros(n, n);
    if n == 0 {
        return b;
    }
    let tail = tails(d);
    let r = tail[0].max(TINY_TAIL);
    for i in 0..n {
        b[(0, i)] = d[i] / r;
    }
    if n == 1 {
        return b;
    }
    // t_k = acos(d_(k-1) / rho), rho = |d_(k-1..)|, sigma = |d_(k..)|.
    for k in 1..n - 1 {
        let rho = tail[k - 1].max(TINY_TAIL);
        let sigma = tail[k].max(TINY_TAIL);
        b[(k, k - 1)] = -sigma / (rho * rho);
        for i in k..n {
            b[(k, i)] = d[k - 1] * d[i] / (rho * rho * sigma);
        }
    }
    // The last angle: atan2(d_(n-1), d_(n-2)).
    let h = (d[n - 2] * d[n - 2] + d[n - 1] * d[n - 1]).max(TINY_TAIL * TINY_TAIL);
    b[(n - 1, n - 2)] = -d[n - 1] / h;
    b[(n - 1, n - 1)] = d[n - 2] / h;
    b
}

/// The polar coordinates of a structure about a centre.
#[derive(Debug, Clone)]
pub struct PolarAboutCentre {
    centre: Col<f64>,
}

impl PolarAboutCentre {
    /// About the flat Cartesian coordinates `centre` (Bohr).
    pub fn new(centre: Col<f64>) -> Self {
        Self { centre }
    }

    fn displacement_of(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        let x = sys.coords_flat_bohr();
        if x.nrows() != self.centre.nrows() {
            return Err(MoptError::coord(
                "polar coordinates: the structure and the centre differ in size",
            ));
        }
        Ok(&x - &self.centre)
    }
}

impl CoordinateSystem for PolarAboutCentre {
    fn name(&self) -> &str {
        "hyperspherical"
    }

    fn initialize(&mut self, _sys: &MolecularSystem) -> MoptResult<()> {
        Ok(())
    }

    fn dim(&self, sys: &MolecularSystem) -> usize {
        sys.dof()
    }

    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        Ok(from_cartesian(&self.displacement_of(sys)?))
    }

    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        Ok(jacobian(&self.displacement_of(sys)?))
    }

    /// The last angle is taken the short way round.
    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        let mut d = self.values(to)? - self.values(from)?;
        let n = d.nrows();
        if n > 1 {
            let last = &mut d[n - 1];
            while *last > PI {
                *last -= 2.0 * PI;
            }
            while *last < -PI {
                *last += 2.0 * PI;
            }
        }
        Ok(d)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Col<f64> {
        Col::from_fn(n, |i| {
            libm::sin(1.3 * i as f64 + 0.4) * (1.0 + 0.1 * i as f64)
        })
    }

    #[test]
    fn round_trip_and_ranges() {
        for n in [1, 2, 3, 4, 7, 21] {
            let x = sample(n);
            let p = from_cartesian(&x);
            let back = to_cartesian(&p);
            for i in 0..n {
                assert!((back[i] - x[i]).abs() < 1e-12, "n {n}, i {i}");
            }
            for k in 1..n.saturating_sub(1) {
                assert!((0.0..=PI).contains(&p[k]));
            }
            if n > 1 {
                assert!((0.0..2.0 * PI).contains(&p[n - 1]));
            }
        }
    }

    #[test]
    fn jacobian_against_differences() {
        for n in [2, 3, 6, 9] {
            let d = sample(n);
            let b = jacobian(&d);
            for i in 0..n {
                let h = 1e-6;
                let mut up = d.clone();
                up[i] += h;
                let mut down = d.clone();
                down[i] -= h;
                let (pu, pd) = (from_cartesian(&up), from_cartesian(&down));
                for k in 0..n {
                    let numeric = (pu[k] - pd[k]) / (2.0 * h);
                    assert!(
                        (numeric - b[(k, i)]).abs() < 1e-6,
                        "n {n}, k {k}, i {i}: {numeric} vs {}",
                        b[(k, i)]
                    );
                }
            }
        }
    }
}
