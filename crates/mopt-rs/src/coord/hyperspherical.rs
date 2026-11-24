//! Hyperspherical (polar) coordinates of a point of `n >= 2` dimensions: the radius and
//! `n - 1` angles,
//!
//! ```text
//! x_0     = r cos q_1
//! x_a     = r sin q_1 ... sin q_a cos q_(a+1)      (1 <= a <= n - 2)
//! x_(n-1) = r sin q_1 ... sin q_(n-2) sin q_(n-1)
//! ```
//!
//! with `q_1 ... q_(n-2)` in `[0, pi]` and the last angle `q_(n-1)` in `[0, 2 pi)`
//! (MultiOptPy's convention; `n = 3` gives the usual `r, theta, phi` with `x_0` along the
//! polar axis). The angles come from `atan2` of a partial norm and a component, the partial
//! norms `rho_a = |(x_a, ..., x_(n-1))|` are accumulated from the end with `hypot`, and
//! both Jacobians are analytic:
//!
//! ```text
//! dr/dx_k      = x_k / r
//! dq_a/dx_(a-1) = -rho_a / rho_(a-1)^2,  dq_a/dx_k = x_(a-1) x_k / (rho_(a-1)^2 rho_a)  (k >= a)
//! dq_(n-1)/dx_(n-2) = -x_(n-1) / rho_(n-2)^2,  dq_(n-1)/dx_(n-1) = x_(n-2) / rho_(n-2)^2
//! ```
//!
//! and `dx/dq` by replacing the factor of the angle by its derivative in the products above.
//! The coordinates are singular where a partial norm vanishes (the angles of the components
//! that follow are not defined); the coordinate systems that use them
//! ([`super::system::spherical`]) turn the frame so that a structure is far from there.
//!
//! MultiOptPy: `Coordinate/polar_coordinate.py` (`cart2polar`, `polar2cart`,
//! `compute_analytical_jacobian`, `cart_grad_2_polar_grad`). Its Jacobian takes the
//! Cartesian point for its first column from `cart2polar` of the polar coordinates (the
//! inverse map), so `dx/dr` is wrong; and its angles come from `arccos` of a ratio, which
//! loses accuracy near `0` and `pi`. See `docs/source/migration_optimizers.md`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use crate::error::{MoptError, MoptResult};

fn check(n: usize) -> MoptResult<()> {
    if n < 2 {
        return Err(MoptError::coord(format!(
            "hyperspherical coordinates need two dimensions at least, got {n}"
        )));
    }
    Ok(())
}

/// Partial norms `rho_a = |(x_a, ..., x_(n-1))|`, `rho_0 = r`.
pub fn partial_norms(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut rho = vec![0.0; n];
    let mut running = 0.0_f64;
    for a in (0..n).rev() {
        running = libm::hypot(running, x[a]);
        rho[a] = running;
    }
    rho
}

/// The hyperspherical coordinates `(r, q_1, ..., q_(n-1))` of `x`.
pub fn to_hyperspherical(x: &[f64]) -> MoptResult<Vec<f64>> {
    let n = x.len();
    check(n)?;
    let rho = partial_norms(x);
    let mut q = vec![0.0; n];
    q[0] = rho[0];
    for a in 1..n - 1 {
        q[a] = libm::atan2(rho[a], x[a - 1]);
    }
    let last = libm::atan2(x[n - 1], x[n - 2]);
    q[n - 1] = if last < 0.0 { last + 2.0 * PI } else { last };
    Ok(q)
}

/// The point of the hyperspherical coordinates `q`.
pub fn from_hyperspherical(q: &[f64]) -> MoptResult<Vec<f64>> {
    let n = q.len();
    check(n)?;
    let mut x = vec![0.0; n];
    // P_a = r sin q_1 ... sin q_a.
    let mut p = q[0];
    for a in 0..n - 1 {
        x[a] = p * libm::cos(q[a + 1]);
        if a < n - 2 {
            p *= libm::sin(q[a + 1]);
        }
    }
    x[n - 1] = p * libm::sin(q[n - 1]);
    Ok(x)
}

/// `dq/dx` at the point `x` (`n x n`, rows the coordinates). Fails where a partial norm
/// that a derivative divides by vanishes.
pub fn wilson(x: &[f64]) -> MoptResult<Mat<f64>> {
    let n = x.len();
    check(n)?;
    let rho = partial_norms(x);
    let scale = rho[0].max(f64::MIN_POSITIVE);
    let tiny = 1e-12 * scale;
    if rho[0] <= tiny || rho[n - 2] <= tiny {
        return Err(MoptError::coord(
            "hyperspherical coordinates are singular here (the radius or the last plane vanishes)",
        ));
    }
    let mut b = Mat::<f64>::zeros(n, n);
    for k in 0..n {
        b[(0, k)] = x[k] / rho[0];
    }
    for a in 1..n - 1 {
        if rho[a] <= tiny {
            return Err(MoptError::coord(format!(
                "hyperspherical coordinates are singular here (angle {a} is at a pole)"
            )));
        }
        let outer = rho[a - 1] * rho[a - 1];
        b[(a, a - 1)] = -rho[a] / outer;
        for k in a..n {
            b[(a, k)] = x[a - 1] * x[k] / (outer * rho[a]);
        }
    }
    let plane = rho[n - 2] * rho[n - 2];
    b[(n - 1, n - 2)] = -x[n - 1] / plane;
    b[(n - 1, n - 1)] = x[n - 2] / plane;
    Ok(b)
}

/// `dx/dq` at the coordinates `q` (`n x n`, columns the coordinates).
pub fn jacobian(q: &[f64]) -> MoptResult<Mat<f64>> {
    let n = q.len();
    check(n)?;
    let (sin, cos): (Vec<f64>, Vec<f64>) = q.iter().map(|v| (libm::sin(*v), libm::cos(*v))).unzip();
    // The angle factor of x_a after the products: cos q_(a+1), or sin q_(n-1) for the last.
    let tail = |a: usize| if a < n - 1 { cos[a + 1] } else { sin[n - 1] };
    let mut j = Mat::<f64>::zeros(n, n);
    // d/dr: the products with r = 1.
    let mut p = 1.0;
    for a in 0..n {
        j[(a, 0)] = p * tail(a);
        if a + 1 < n - 1 {
            p *= sin[a + 1];
        }
    }
    // d/dq_c for c = 1 .. n-1.
    let mut before = q[0]; // r sin q_1 ... sin q_(c-1)
    for c in 1..n {
        // x_(c-1) = before cos q_c.
        j[(c - 1, c)] = -before * sin[c];
        if c == n - 1 {
            // x_(n-1) = before sin q_(n-1).
            j[(n - 1, c)] = before * cos[c];
        } else {
            // x_a for a >= c carries sin q_c: its derivative carries cos q_c.
            let mut d = before * cos[c];
            for a in c..n {
                j[(a, c)] = d * tail(a);
                if a + 1 < n - 1 {
                    d *= sin[a + 1];
                }
            }
            before *= sin[c];
        }
    }
    Ok(j)
}

/// The gradient with respect to the hyperspherical coordinates of `x` of a function whose
/// gradient with respect to `x` is `g`: `(dx/dq)^T g`.
pub fn gradient_to_hyperspherical(x: &[f64], g: &[f64]) -> MoptResult<Vec<f64>> {
    if g.len() != x.len() {
        return Err(MoptError::coord(format!(
            "{} gradient components for {} coordinates",
            g.len(),
            x.len()
        )));
    }
    let j = jacobian(&to_hyperspherical(x)?)?;
    let n = x.len();
    Ok((0..n)
        .map(|c| (0..n).map(|a| j[(a, c)] * g[a]).sum())
        .collect())
}

/// `to - from` of the last angle wrapped into `(-pi, pi]`.
pub fn last_angle_difference(to: f64, from: f64) -> f64 {
    let d = to - from;
    d - 2.0 * PI * ((d + PI) / (2.0 * PI)).floor()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const POINTS: [&[f64]; 4] = [
        &[0.3, -1.2],
        &[1.0, 0.4, -0.7],
        &[0.2, -0.5, 0.9, 1.3, -0.4, 0.6, -1.1],
        &[-2.0, 0.1, 0.3, -0.2, 0.05],
    ];

    #[test]
    fn the_map_and_its_inverse_and_the_usual_spherical_coordinates() {
        for x in POINTS {
            let q = to_hyperspherical(x).unwrap();
            let back = from_hyperspherical(&q).unwrap();
            for (a, b) in x.iter().zip(&back) {
                assert!((a - b).abs() < 1e-13, "{x:?} {back:?}");
            }
            for angle in &q[1..x.len() - 1] {
                assert!((0.0..=PI).contains(angle));
            }
            assert!((0.0..2.0 * PI).contains(&q[x.len() - 1]));
        }
        // n = 3: r, theta from the first axis, phi in the plane of the other two.
        let q = to_hyperspherical(&[0.0, 0.0, 2.0]).unwrap();
        assert!((q[0] - 2.0).abs() < 1e-15 && (q[1] - PI / 2.0).abs() < 1e-15);
        assert!((q[2] - PI / 2.0).abs() < 1e-15);
    }

    #[test]
    fn both_jacobians_against_differences_and_each_other() {
        let h = 1e-6;
        for x in POINTS {
            let n = x.len();
            let q = to_hyperspherical(x).unwrap();
            let b = wilson(x).unwrap();
            let j = jacobian(&q).unwrap();
            for k in 0..n {
                let mut up = x.to_vec();
                let mut down = x.to_vec();
                up[k] += h;
                down[k] -= h;
                let (qu, qd) = (
                    to_hyperspherical(&up).unwrap(),
                    to_hyperspherical(&down).unwrap(),
                );
                for a in 0..n {
                    let mut d = qu[a] - qd[a];
                    if a == n - 1 {
                        d = last_angle_difference(qu[a], qd[a]);
                    }
                    assert!((d / (2.0 * h) - b[(a, k)]).abs() < 1e-7, "B {n} {a} {k}");
                }
                let mut qu = q.clone();
                let mut qd = q.clone();
                qu[k] += h;
                qd[k] -= h;
                let (xu, xd) = (
                    from_hyperspherical(&qu).unwrap(),
                    from_hyperspherical(&qd).unwrap(),
                );
                for a in 0..n {
                    assert!(
                        ((xu[a] - xd[a]) / (2.0 * h) - j[(a, k)]).abs() < 1e-7,
                        "J {n} {a} {k}"
                    );
                }
            }
            // dq/dx dx/dq = 1.
            let product = &b * &j;
            for a in 0..n {
                for c in 0..n {
                    let expected = if a == c { 1.0 } else { 0.0 };
                    assert!((product[(a, c)] - expected).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn the_gradient_and_the_singular_points() {
        // f = |x|^2 / 2: df/dr = r, nothing along the angles.
        let x = [0.4, -0.3, 1.2, 0.5];
        let g = gradient_to_hyperspherical(&x, &x).unwrap();
        let r = partial_norms(&x)[0];
        assert!((g[0] - r).abs() < 1e-14);
        assert!(g[1..].iter().all(|v| v.abs() < 1e-14));
        assert!(wilson(&[0.0, 0.0, 0.0]).is_err());
        assert!(wilson(&[1.0, 0.0, 0.0]).is_err());
        assert!(to_hyperspherical(&[1.0]).is_err());
        assert!((last_angle_difference(0.1, 2.0 * PI - 0.1) - 0.2).abs() < 1e-14);
    }
}
