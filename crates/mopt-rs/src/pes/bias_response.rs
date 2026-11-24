//! Derivatives with respect to the constants of bias potentials (a force constant, an
//! equilibrium distance or angle, the collision energy of AFIR, a wall radius, ...), and the
//! response of a stationary point of the biased surface to them.
//!
//! At a structure `x` the derivative of the bias energy `dE/dp` and of its gradient `dg/dp`
//! come from central differences of the terms built with the constant moved up and down by a
//! step `h` (the terms are rebuilt from the constant, so every bias has its derivatives, those
//! given by users included); the error is of order `h^2`. A stationary point `x*(p)` of the
//! surface with the bias keeps `g(x*(p), p) = 0`, so that
//!
//! - `dx*/dp = -H^+ dg/dp` (`H` the Hessian of the whole surface at `x*`; the rigid motions are
//!   taken out where the surface does not depend on them),
//! - `dE*/dp = dE/dp` for the energy with the bias (its gradient vanishes at `x*`),
//! - `dE0*/dp = g0 . dx*/dp` for the energy without the bias (`g0` its gradient at `x*`): how
//!   much the energy of the optimized structure itself changes with the constant.
//!
//! The response is that of linear perturbation theory: it holds while the stationary point
//! moves smoothly with `p`; near a constant where the minimum disappears (a small eigenvalue of
//! `H`) it grows without bound, which [`StationaryResponse::smallest_curvature`] shows.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::math::linalg::symmetric_eigen;
use crate::pes::term::{PesTerm, TermContext};

/// Eigenvalues of the Hessian below this share of the largest one are left out of its
/// inverse (they belong to motions the surface does not depend on).
pub const RESPONSE_CUTOFF: f64 = 1e-10;

/// The derivatives of a bias energy and its gradient in one constant at a structure.
#[derive(Debug, Clone)]
pub struct ConstantDerivative {
    /// `dE/dp` (Hartree per unit of the constant).
    pub d_energy: f64,
    /// `dg/dp` (Hartree / Bohr per unit of the constant).
    pub d_gradient: Col<f64>,
    /// The step of the differences (unit of the constant).
    pub step: f64,
}

/// The energy and gradient of `terms` at `sys`.
pub fn terms_value(
    terms: &[Arc<dyn PesTerm>],
    sys: &MolecularSystem,
    ctx: &TermContext<'_>,
) -> MoptResult<(f64, Col<f64>)> {
    let mut energy = 0.0;
    let mut gradient = Col::<f64>::zeros(3 * sys.n_atoms());
    for term in terms {
        let out = term
            .evaluate(sys, ctx, false)
            .context(term.name().to_string())?;
        energy += out.energy.value();
        gradient += &out.gradient_bohr;
    }
    Ok((energy, gradient))
}

/// The step of the differences for a constant of `value`: 1e-4 of it, at least 1e-6.
pub fn default_step(value: f64) -> f64 {
    1e-4 * value.abs().max(1e-2)
}

/// The derivatives at `sys` in the constant whose value is `value`: `build(v)` makes the
/// terms with the constant at `v`.
pub fn constant_derivative(
    build: &dyn Fn(f64) -> MoptResult<Vec<Arc<dyn PesTerm>>>,
    value: f64,
    step: f64,
    sys: &MolecularSystem,
    ctx: &TermContext<'_>,
) -> MoptResult<ConstantDerivative> {
    if !(step.is_finite() && step > 0.0 && value.is_finite()) {
        return Err(MoptError::config(format!(
            "the derivative in a constant needs a finite value and a positive step (got {value} \
             and {step})"
        )));
    }
    let (up_e, up_g) = terms_value(&build(value + step)?, sys, ctx)?;
    let (down_e, down_g) = terms_value(&build(value - step)?, sys, ctx)?;
    let scale = 0.5 / step;
    Ok(ConstantDerivative {
        d_energy: (up_e - down_e) * scale,
        d_gradient: (&up_g - &down_g) * faer::Scale(scale),
        step,
    })
}

/// An orthonormal basis of the span of the columns of `vectors` (modified Gram-Schmidt;
/// columns within `1e-10` of the span of the earlier ones are left out). `None` when there are
/// none.
pub fn orthonormal_columns(vectors: &Mat<f64>) -> Option<Mat<f64>> {
    let n = vectors.nrows();
    let mut kept: Vec<Col<f64>> = Vec::new();
    for k in 0..vectors.ncols() {
        let mut v: Col<f64> = vectors.col(k).to_owned();
        let size = v.norm_l2();
        for _ in 0..2 {
            for q in &kept {
                let c = q.transpose() * &v;
                v -= q * faer::Scale(c);
            }
        }
        let left = v.norm_l2();
        if size > 0.0 && left > 1e-10 * size {
            kept.push(v * faer::Scale(1.0 / left));
        }
    }
    if kept.is_empty() {
        return None;
    }
    Some(Mat::from_fn(n, kept.len(), |i, j| kept[j][i]))
}

/// How a stationary point moves with a constant.
#[derive(Debug, Clone)]
pub struct StationaryResponse {
    /// `dx*/dp` (Bohr per unit of the constant).
    pub d_structure: Col<f64>,
    /// `dE*/dp` of the energy with the bias (Hartree per unit).
    pub d_energy: f64,
    /// `dE0*/dp` of the energy without the bias (Hartree per unit).
    pub d_unbiased_energy: f64,
    /// Whether the rigid motions were taken out.
    pub rigid_removed: bool,
    /// The smallest absolute eigenvalue of the Hessian that was kept (Hartree / Bohr^2).
    pub smallest_curvature: f64,
}

/// The response of the stationary point with the Hessian `hessian` of the whole surface and the
/// gradient `unbiased_gradient` of the surface without the bias there, to the constant of
/// `derivative`. The orthonormal columns of `removed` (the rigid motions) are taken out of the
/// Hessian and of `dg/dp` when given.
pub fn stationary_response(
    hessian: &Mat<f64>,
    derivative: &ConstantDerivative,
    unbiased_gradient: &Col<f64>,
    removed: Option<&Mat<f64>>,
) -> MoptResult<StationaryResponse> {
    let n = hessian.nrows();
    if hessian.ncols() != n || derivative.d_gradient.nrows() != n || unbiased_gradient.nrows() != n
    {
        return Err(MoptError::config(format!(
            "the response needs a square Hessian and gradients of its size ({}x{}, {}, {})",
            n,
            hessian.ncols(),
            derivative.d_gradient.nrows(),
            unbiased_gradient.nrows()
        )));
    }
    // P = I - R R^T on the Hessian and the derivative of the gradient.
    let project = |v: &Col<f64>| -> Col<f64> {
        match removed {
            Some(r) => {
                let c = r.transpose() * v;
                v - r * &c
            }
            None => v.clone(),
        }
    };
    let h = match removed {
        Some(r) => {
            let p = Mat::<f64>::identity(n, n) - r * r.transpose();
            &p * hessian * &p
        }
        None => hessian.clone(),
    };
    let sym = Mat::from_fn(n, n, |i, j| 0.5 * (h[(i, j)] + h[(j, i)]));
    let (values, vectors) = symmetric_eigen(&sym)?;
    let largest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let cutoff = RESPONSE_CUTOFF * largest.max(f64::MIN_POSITIVE);
    let dg = project(&derivative.d_gradient);
    let mut dx = Col::<f64>::zeros(n);
    let mut smallest = f64::INFINITY;
    for k in 0..n {
        let lambda = values[k];
        if lambda.abs() <= cutoff {
            continue;
        }
        let v = vectors.col(k);
        let along = v.transpose() * &dg;
        dx -= v * faer::Scale(along / lambda);
        smallest = smallest.min(lambda.abs());
    }
    if !smallest.is_finite() {
        return Err(MoptError::linalg(
            "the Hessian of the stationary point has no curvature to respond with",
        ));
    }
    let dx = project(&dx);
    let d_unbiased_energy = unbiased_gradient.transpose() * &dx;
    Ok(StationaryResponse {
        d_structure: dx,
        d_energy: derivative.d_energy,
        d_unbiased_energy,
        rigid_removed: removed.is_some(),
        smallest_curvature: smallest,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::AtomId;
    use crate::domain::{positions, Bohr, Element, SpinCharge};
    use crate::math::projection::build_tr_rot_basis;
    use crate::pes::bias::{AdBias, KeepDistance, Ramp};

    fn spring(k: f64, r0: f64) -> Arc<dyn PesTerm> {
        Arc::new(AdBias::new(KeepDistance {
            a: vec![AtomId::new(0)],
            b: vec![AtomId::new(1)],
            k: Ramp::constant(k),
            r0: Ramp::constant(r0),
        }))
    }

    fn dimer(r: f64) -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::H, Element::H],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    /// Two springs on one distance: the surface `K/2 (r - a)^2` and the bias `k/2 (r - r0)^2`,
    /// whose minimum `r* = (K a + k r0) / (K + k)` moves by `k / (K + k)` with `r0` and by
    /// `(r0 - r*) / (K + k)` with `k`.
    #[test]
    fn the_minimum_of_two_springs_moves_as_the_formula_says() {
        let (big_k, a, k, r0) = (0.4, 1.4, 0.1, 2.0);
        let r_star = (big_k * a + k * r0) / (big_k + k);
        let sys = dimer(r_star);
        let ctx = TermContext::at_iteration(0, &sys);
        let surface = spring(big_k, a);
        let surface_out = surface.evaluate(&sys, &ctx, true).unwrap();
        let bias_out = spring(k, r0).evaluate(&sys, &ctx, true).unwrap();
        let hessian = surface_out.hessian_bohr.unwrap() + bias_out.hessian_bohr.unwrap();
        let removed = build_tr_rot_basis(&sys, false);
        // In r0.
        let d = constant_derivative(
            &|v| Ok(vec![spring(k, v)]),
            r0,
            default_step(r0),
            &sys,
            &ctx,
        )
        .unwrap();
        // dE/dr0 = -k (r* - r0).
        assert!((d.d_energy + k * (r_star - r0)).abs() < 1e-10);
        let resp =
            stationary_response(&hessian, &d, &surface_out.gradient_bohr, Some(&removed)).unwrap();
        let dr = resp.d_structure[3] - resp.d_structure[0];
        assert!((dr - k / (big_k + k)).abs() < 1e-8, "{dr}");
        // dE0*/dr0 = K (r* - a) dr*/dr0.
        let want = big_k * (r_star - a) * k / (big_k + k);
        assert!((resp.d_unbiased_energy - want).abs() < 1e-8);
        // In k.
        let d = constant_derivative(&|v| Ok(vec![spring(v, r0)]), k, default_step(k), &sys, &ctx)
            .unwrap();
        assert!((d.d_energy - 0.5 * (r_star - r0).powi(2)).abs() < 1e-10);
        let resp =
            stationary_response(&hessian, &d, &surface_out.gradient_bohr, Some(&removed)).unwrap();
        let dr = resp.d_structure[3] - resp.d_structure[0];
        assert!((dr - (r0 - r_star) / (big_k + k)).abs() < 1e-8, "{dr}");
        assert!(resp.rigid_removed);
    }

    #[test]
    fn bad_input_is_refused() {
        let sys = dimer(1.4);
        let ctx = TermContext::at_iteration(0, &sys);
        assert!(constant_derivative(&|v| Ok(vec![spring(v, 1.0)]), 1.0, 0.0, &sys, &ctx).is_err());
        let d = constant_derivative(&|v| Ok(vec![spring(v, 1.0)]), 1.0, 1e-4, &sys, &ctx).unwrap();
        assert!(stationary_response(&Mat::zeros(3, 3), &d, &Col::zeros(6), None).is_err());
        assert!(stationary_response(&Mat::zeros(6, 6), &d, &Col::zeros(6), None).is_err());
    }
}
