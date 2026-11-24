//! Verification of the AD drivers against finite differences.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.

use super::*;
use crate::domain::quantity::shorthand::bohr;
use crate::domain::PeriodicCell;

/// Pair potential across a periodic boundary, generic over the AD scalar.
fn pair_potential<T: Scalar>(x: &[T], cell: &PeriodicCell) -> T {
    let r1 = [x[0], x[1], x[2]];
    let r2 = [x[3], x[4], x[5]];
    let d = mic_delta_ad(r1, r2, Some(cell));
    let dist = T::norm3(d);
    let dr = dist - T::from_f64(1.8);
    T::from_f64(0.5) * dr * dr + (dist * T::from_f64(0.7)).cos()
}

fn cell() -> PeriodicCell {
    PeriodicCell::orthorhombic(bohr(8.0), bohr(8.0), bohr(8.0)).unwrap()
}

const X0: [f64; 6] = [0.4, 1.1, 7.5, 7.7, 1.3, 0.2];

#[test]
fn dual_gradient_matches_central_differences() {
    let c = cell();
    let (e, g) = eval_value_and_gradient(&X0, |x| pair_potential(x, &c));
    assert!((e - pair_potential(&X0, &c)).abs() < 1e-14);
    let eps = 1e-6;
    for i in 0..6 {
        let mut xp = X0;
        let mut xm = X0;
        xp[i] += eps;
        xm[i] -= eps;
        let fd = (pair_potential(&xp, &c) - pair_potential(&xm, &c)) / (2.0 * eps);
        assert!(
            (g[i] - fd).abs() < 1e-9,
            "gradient mismatch at {i}: {} vs {fd}",
            g[i]
        );
    }
}

#[test]
fn hyperdual_hessian_matches_differences_of_exact_gradient() {
    let c = cell();
    let (e, g, h) = eval_value_grad_hessian(&X0, None, |x| pair_potential(x, &c));
    let (e_ref, g_ref) = eval_value_and_gradient(&X0, |x| pair_potential(x, &c));
    assert!((e - e_ref).abs() < 1e-14);
    let eps = 1e-6;
    for i in 0..6 {
        assert!((g[i] - g_ref[i]).abs() < 1e-14);
        let mut xp = X0;
        let mut xm = X0;
        xp[i] += eps;
        xm[i] -= eps;
        let (_, gp) = eval_value_and_gradient(&xp, |x| pair_potential(x, &c));
        let (_, gm) = eval_value_and_gradient(&xm, |x| pair_potential(x, &c));
        for j in 0..6 {
            let fd = (gp[j] - gm[j]) / (2.0 * eps);
            assert!(
                (h[(i, j)] - fd).abs() < 1e-8,
                "Hessian mismatch at ({i},{j})"
            );
            assert_eq!(h[(i, j)], h[(j, i)]);
        }
    }
}

#[test]
fn restricted_active_set_leaves_other_entries_zero() {
    let c = cell();
    let (_, g, h) = eval_value_grad_hessian(&X0, Some(&[0, 3]), |x| pair_potential(x, &c));
    let (_, g_full, h_full) = eval_value_grad_hessian(&X0, None, |x| pair_potential(x, &c));
    for i in 0..6 {
        let active_i = i == 0 || i == 3;
        assert_eq!(g[i], if active_i { g_full[i] } else { 0.0 });
        for j in 0..6 {
            let active = active_i && (j == 0 || j == 3);
            assert_eq!(h[(i, j)], if active { h_full[(i, j)] } else { 0.0 });
        }
    }
}

#[test]
fn the_angle_and_its_derivatives_stay_accurate_near_0_and_180_degrees() {
    // a = (1, 0, 0) and b = (s, y, z) at z = 0: theta = atan2(rho, s) with rho = |(y, z)|,
    // d theta / dy = s / (s^2 + y^2), d2 theta / dy2 = -2 s y / (s^2 + y^2)^2, and across the
    // plane d2 theta / dz2 = s / ((s^2 + y^2) y), for s = +1 (near 0) and s = -1 (near pi).
    // Near the line the angle is a cone: its curvature grows as 1 / y, which sets the scale of
    // the rounding errors of the second derivatives.
    for s in [1.0_f64, -1.0] {
        for y in [1e-9_f64, 1e-6, 1e-3] {
            let f = |v: &[HyperDual]| {
                let a = [HyperDual::one(), HyperDual::zero(), HyperDual::zero()];
                let b = [HyperDual::from_f64(s), v[0], v[1]];
                angle_between(a, b)
            };
            let (theta, g, h) = eval_value_grad_hessian(&[y, 0.0], None, f);
            let want = y.atan2(s);
            let d1 = s / (s * s + y * y);
            let d2 = -2.0 * s * y / (s * s + y * y).powi(2);
            let across = s / ((s * s + y * y) * y);
            let scale = 1.0 / y;
            assert!(
                (theta - want).abs() < 1e-15 * want.abs().max(1e-3),
                "{s} {y}"
            );
            assert!((g[0] - d1).abs() < 1e-12, "{s} {y}: {} vs {d1}", g[0]);
            assert!(g[1].abs() < 1e-12, "{s} {y}: {}", g[1]);
            assert!(
                (h[(0, 0)] - d2).abs() < 1e-12 * scale,
                "{s} {y}: {} vs {d2}",
                h[(0, 0)]
            );
            assert!(
                (h[(1, 1)] - across).abs() < 1e-12 * scale,
                "{s} {y}: {} vs {across}",
                h[(1, 1)]
            );
        }
    }
}

#[test]
fn jacobian_of_vector_function() {
    let (val, jac) = eval_vector_and_jacobian(&[1.0, 2.0], 2, |x, out| {
        out[0] = x[0] * x[1];
        out[1] = x[0].sin();
    });
    assert_eq!(val[0], 2.0);
    assert_eq!((jac[(0, 0)], jac[(0, 1)]), (2.0, 1.0));
    assert!((jac[(1, 0)] - 1.0_f64.cos()).abs() < 1e-15);
    assert_eq!(jac[(1, 1)], 0.0);
}
