//! The starts of the upward and downward searches of the solution landscape (J. Yin, B. Yu,
//! L. Zhang, Sci. China Math. 2021, 64, 1801, Algorithms 1 and 2; J. Yin, Y. Wang,
//! J. Z. Y. Chen, P. Zhang, L. Zhang, Phys. Rev. Lett. 2020, 124, 090601).
//!
//! - Upward, from a stationary point of index `m` with its unstable directions `u_1..u_m`
//!   and a stable direction `w` (a positive curvature, the lowest by default): the point
//!   `x +- eps w` and the directions `[u_1, ..., u_m, w]` for a search of index `m + 1`. At
//!   the start the reflected gradient pushes away along `w` and the search climbs there.
//! - Downward, from a saddle point of index `m >= 1`: for each unstable direction `w` (or,
//!   for `m = 2`, a fan of `cos(t) u_1 + sin(t) u_2`, `t = pi j / N`, since an index-2
//!   saddle can have more lower neighbours than four), the point `x +- eps w` and the other
//!   unstable directions (the orthonormal complement of `w` in their span) for a search of
//!   index `m - 1` (a minimization for `m = 1`, which is not a saddle search).
//!
//! Directions and the displacement are in the coordinates of the given point and its
//! directions (in the mass-weighted metric if the directions are mass-weighted and the
//! point is converted by the caller).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::subspace::{gram_schmidt, orthonormal_to};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_norm;

/// The start of one search of the landscape.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchStart {
    /// The displaced point.
    pub point: Col<f64>,
    /// The directions of the search (the first `order` are reflected).
    pub directions: Vec<Col<f64>>,
    /// The index sought.
    pub order: usize,
    /// The direction of the displacement (unit).
    pub along: Col<f64>,
    /// The sign of the displacement (`+1` or `-1`).
    pub sign: f64,
}

fn unit(v: &Col<f64>, what: &str) -> MoptResult<Col<f64>> {
    let length = col_norm(v);
    if !(length.is_finite() && length > 0.0) {
        return Err(MoptError::config(format!("{what}: a direction is zero")));
    }
    Ok(v * (1.0 / length))
}

/// The starts of the upward searches from `point` (index `unstable.len()`), one pair
/// (`+` and `-`) per direction of `stable`.
pub fn upward_starts(
    point: &Col<f64>,
    unstable: &[Col<f64>],
    stable: &[Col<f64>],
    displacement: f64,
) -> MoptResult<Vec<SearchStart>> {
    if !(displacement.is_finite() && displacement > 0.0) {
        return Err(MoptError::config(
            "upward search: the displacement must be positive",
        ));
    }
    let up = gram_schmidt(unstable);
    if up.len() != unstable.len() {
        return Err(MoptError::config(
            "upward search: the unstable directions are not independent",
        ));
    }
    let mut out = Vec::new();
    for w in stable {
        let w = unit(w, "upward search")?;
        let Some(w_perp) = orthonormal_to(&up, &w) else {
            return Err(MoptError::config(
                "upward search: a stable direction lies in the span of the unstable ones",
            ));
        };
        let mut directions = up.clone();
        directions.push(w_perp.clone());
        for sign in [1.0, -1.0] {
            out.push(SearchStart {
                point: point + &w_perp * (sign * displacement),
                directions: directions.clone(),
                order: up.len() + 1,
                along: w_perp.clone(),
                sign,
            });
        }
    }
    Ok(out)
}

/// The starts of the downward searches from `point` (index `unstable.len() >= 1`): each
/// unstable direction, or with `fan >= 2` and two unstable directions `fan` directions of
/// their plane, both signs.
pub fn downward_starts(
    point: &Col<f64>,
    unstable: &[Col<f64>],
    displacement: f64,
    fan: usize,
) -> MoptResult<Vec<SearchStart>> {
    if !(displacement.is_finite() && displacement > 0.0) {
        return Err(MoptError::config(
            "downward search: the displacement must be positive",
        ));
    }
    let basis = gram_schmidt(unstable);
    if basis.is_empty() || basis.len() != unstable.len() {
        return Err(MoptError::config(
            "downward search: the unstable directions are missing or not independent",
        ));
    }
    let along: Vec<Col<f64>> = if basis.len() == 2 && fan >= 2 {
        (0..fan)
            .map(|j| {
                let t = std::f64::consts::PI * j as f64 / fan as f64;
                &basis[0] * libm::cos(t) + &basis[1] * libm::sin(t)
            })
            .collect()
    } else {
        basis.clone()
    };
    let mut out = Vec::new();
    for w in along {
        // The complement of `w` in the span: Gram-Schmidt of [w, basis...] without `w`.
        let mut with_w = vec![w.clone()];
        with_w.extend(basis.iter().cloned());
        let complement: Vec<Col<f64>> = gram_schmidt(&with_w).into_iter().skip(1).collect();
        for sign in [1.0, -1.0] {
            out.push(SearchStart {
                point: point + &w * (sign * displacement),
                directions: complement.clone(),
                order: basis.len() - 1,
                along: w.clone(),
                sign,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::math::linalg::col_dot;

    fn e(i: usize) -> Col<f64> {
        Col::from_fn(4, |k| if k == i { 1.0 } else { 0.0 })
    }

    #[test]
    fn upward_and_downward_starts() {
        let x = Col::<f64>::zeros(4);
        let up = upward_starts(&x, &[e(0)], &[e(1), e(2)], 0.1).unwrap();
        assert_eq!(up.len(), 4);
        assert_eq!(up[0].order, 2);
        assert!((up[1].point[1] + 0.1).abs() < 1e-15);
        assert_eq!(up[2].directions.len(), 2);
        assert!(upward_starts(&x, &[e(0)], &[e(0)], 0.1).is_err());
        let down = downward_starts(&x, &[e(0), e(1)], 0.1, 0).unwrap();
        assert_eq!(down.len(), 4);
        assert_eq!(down[0].order, 1);
        assert!(col_dot(&down[0].directions[0], &e(1)).abs() > 1.0 - 1e-12);
        let fan = downward_starts(&x, &[e(0), e(1)], 0.1, 4).unwrap();
        assert_eq!(fan.len(), 8);
        // The fan direction at 45 degrees keeps the other diagonal.
        let kept = &fan[2].directions[0];
        assert!((col_dot(kept, &fan[2].along)).abs() < 1e-12);
        assert!((kept[0].abs() - kept[1].abs()).abs() < 1e-12);
    }
}
