//! Differentiable geometric internal-coordinate primitives (`bond`, `angle`, `dihedral`,
//! `out_of_plane`) generic over `Scalar` (`f64`, `Dual`, `HyperDual`) with PBC-MIC support.
//!
//! Exact derivatives are obtained per term with `math::ad::local`.
//!
//! MultiOptPy: `Utils/calc_tools.py:calc_bond_length_from_vec`, `calc_angle_from_vec`,
//!  `calc_dihedral_angle_from_vec`, `calc_outofplain_angle_from_vec` (the `torch_*` versions are
//!  replaced by local AD).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use crate::domain::PeriodicCell;
use crate::math::ad::{mic_delta_ad, Scalar};

/// Wrap a signed angle difference `(a - b)` into `(-pi, pi]`.
#[inline]
pub fn wrap_angle_diff_rad(diff: f64) -> f64 {
    let mut d = diff % (2.0 * PI);
    if d <= -PI {
        d += 2.0 * PI;
    } else if d > PI {
        d -= 2.0 * PI;
    }
    d
}

/// Differentiable interatomic distance `||r_a - r_b||` (in Bohr) with optional PBC-MIC.
#[inline]
pub fn bond_distance<T: Scalar>(r_a: [T; 3], r_b: [T; 3], cell: Option<&PeriodicCell>) -> T {
    let d = mic_delta_ad(r_a, r_b, cell);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + T::from_f64(1e-30)).sqrt()
}

/// Differentiable valence bond angle `a - b - c` in radians (`[0, pi]`) with optional PBC-MIC.
#[inline]
pub fn bond_angle_rad<T: Scalar>(
    r_a: [T; 3],
    r_b: [T; 3],
    r_c: [T; 3],
    cell: Option<&PeriodicCell>,
) -> T {
    let u = mic_delta_ad(r_a, r_b, cell);
    let v = mic_delta_ad(r_c, r_b, cell);
    let cross = T::cross3(u, v);
    let sin_norm = (T::dot3(cross, cross) + T::from_f64(1e-30)).sqrt();
    let cos_dot = T::dot3(u, v);
    sin_norm.atan2(cos_dot)
}

/// Differentiable IUPAC signed dihedral / torsion angle `a - b - c - d` in radians (`(-pi, pi]`).
#[inline]
pub fn dihedral_angle_rad<T: Scalar>(
    r_a: [T; 3],
    r_b: [T; 3],
    r_c: [T; 3],
    r_d: [T; 3],
    cell: Option<&PeriodicCell>,
) -> T {
    let b1 = mic_delta_ad(r_b, r_a, cell);
    let b2 = mic_delta_ad(r_c, r_b, cell);
    let b3 = mic_delta_ad(r_d, r_c, cell);

    let b2_norm = (T::dot3(b2, b2) + T::from_f64(1e-30)).sqrt();
    let b2_unit = [b2[0] / b2_norm, b2[1] / b2_norm, b2[2] / b2_norm];

    let n1 = T::cross3(b1, b2);
    let n2 = T::cross3(b2, b3);
    let m1 = T::cross3(n1, b2_unit);

    let x = T::dot3(n1, n2);
    let y = T::dot3(m1, n2);
    y.atan2(x)
}

/// Differentiable Wilson out-of-plane pyramidalization angle for `center` bonded to `(a, b, c)` in radians.
#[inline]
pub fn out_of_plane_angle_rad<T: Scalar>(
    r_center: [T; 3],
    r_a: [T; 3],
    r_b: [T; 3],
    r_c: [T; 3],
    cell: Option<&PeriodicCell>,
) -> T {
    let e1 = mic_delta_ad(r_a, r_center, cell);
    let e2 = mic_delta_ad(r_b, r_center, cell);
    let e3 = mic_delta_ad(r_c, r_center, cell);

    let n1 = (T::dot3(e1, e1) + T::from_f64(1e-30)).sqrt();
    let n2 = (T::dot3(e2, e2) + T::from_f64(1e-30)).sqrt();
    let n3 = (T::dot3(e3, e3) + T::from_f64(1e-30)).sqrt();

    let u1 = [e1[0] / n1, e1[1] / n1, e1[2] / n1];
    let u2 = [e2[0] / n2, e2[1] / n2, e2[2] / n2];
    let u3 = [e3[0] / n3, e3[1] / n3, e3[2] / n3];

    let cross_23 = T::cross3(u2, u3);
    let denom = (T::dot3(cross_23, cross_23) + T::from_f64(1e-28)).sqrt();
    let sin_chi = T::dot3(cross_23, u1) / denom;
    sin_chi.asin()
}
