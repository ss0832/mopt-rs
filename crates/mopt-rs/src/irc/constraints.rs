//! An IRC on the surface of constraints: the path of steepest descent in mass-weighted
//! coordinates within the structures that keep the constraints (fixed atoms, distances,
//! angles, dihedrals, expressions; those of `optmain`).
//!
//! In mass-weighted coordinates `q = M^(1/2) x` the constraints have the gradients
//! `A = J M^(-1/2)` (`J` their Cartesian gradients). The gradient and the Hessians at a
//! point are replaced by their parts on the surface: `P g_q` and `P (H_q - C_q) P` with
//! `P = 1 - Q Q^T` (`Q` an orthonormal basis of the rows of `A`), the curvature
//! `C = sum_k lambda_k d2c_k` of the constraints with the multipliers
//! `lambda = (J M^-1 J^T)^+ J M^-1 g` of the gradient, and a stiffness `s Q Q^T` across the
//! surface (`s = 100 max(max_i |(P H_q P)_ii|, 0.5)`, as the constrained optimizations), so
//! that no integrator steps off it. A step that leaves the curved surface is brought back
//! to it by Newton steps in the mass-weighted metric (the shortest way in `q`). Updates of
//! the Hessian along the path take the steps and the changes of the gradient on the surface
//! only and are put back on it ([`IrcConstraints::on_surface`]): the way back onto the
//! constraints is part of a step, and an update with it would bend the stiff directions
//! into the others.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's IRC ignores the constraints of
//! the optimization before it).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::types::PointValue;
use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::math::linalg::{orthonormalize_columns, symmetrize};
use crate::pes::constraint::ConstraintSet;

/// The stiffness across the surface relative to the largest diagonal element of the
/// Hessian (MultiOptPy's `hard_mult`, as `opt::constrained`).
const STIFFNESS: f64 = 100.0;

/// The constraints of an IRC with the masses of its variables.
#[derive(Debug, Clone)]
pub struct IrcConstraints {
    set: ConstraintSet,
    template: MolecularSystem,
    /// Inverse masses, one per variable.
    weights: Vec<f64>,
    /// Square roots of the masses, one per variable.
    sqrt_mass: Vec<f64>,
}

impl IrcConstraints {
    /// The constraints `set` of structures like `template` with the square roots of the
    /// masses `sqrt_mass` (one per Cartesian coordinate).
    pub fn new(set: ConstraintSet, template: MolecularSystem, sqrt_mass: &[f64]) -> Self {
        Self {
            set,
            template,
            weights: sqrt_mass.iter().map(|s| 1.0 / (s * s)).collect(),
            sqrt_mass: sqrt_mass.to_vec(),
        }
    }

    /// Number of scalar constraints.
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Whether the constraints fix the position or orientation in space.
    pub fn breaks_rigid_body(&self) -> bool {
        self.set.breaks_rigid_body()
    }

    /// The largest deviation of the structure at `x` from the constraints.
    pub fn largest_residual(&self, x: &Col<f64>) -> MoptResult<f64> {
        let sys = self.template.with_flat_coords_bohr(x)?;
        Ok(self
            .set
            .residuals(&sys)?
            .iter()
            .fold(0.0_f64, |m, r| m.max(r.abs())))
    }

    /// The point `x` brought back onto the constraints along the shortest way in
    /// mass-weighted coordinates.
    pub fn restore(&self, x: &Col<f64>) -> MoptResult<Col<f64>> {
        let sys = self.template.with_flat_coords_bohr(x)?;
        Ok(self
            .set
            .restore_in(&sys, Some(&self.weights))?
            .coords_flat_bohr())
    }

    /// The surface at `x` within the constraints: the gradients without their parts across
    /// the surface, the Hessians projected with the curvature of the constraints and stiff
    /// across the surface. The energies are kept.
    pub fn project(&self, x: &Col<f64>, value: PointValue) -> MoptResult<PointValue> {
        if self.set.is_empty() {
            return Ok(value);
        }
        let sys = self.template.with_flat_coords_bohr(x)?;
        let total = self
            .set
            .frame_in(&sys, &value.gradient, Some(&self.weights))?;
        let raw = self
            .set
            .frame_in(&sys, &value.raw_gradient, Some(&self.weights))?;
        let q = self.normals_of(&total.jacobian);
        let hessian = match &value.hessian {
            Some(h) => Some(self.projected(&(h - &raw.curvature), &q, true)),
            None => None,
        };
        let bias_hessian = match &value.bias_hessian {
            Some(b) => {
                let curvature = &total.curvature - &raw.curvature;
                Some(self.projected(&(b - curvature), &q, false))
            }
            None => None,
        };
        Ok(PointValue {
            energy: value.energy,
            gradient: total.project_gradient(&value.gradient),
            raw_energy: value.raw_energy,
            raw_gradient: raw.project_gradient(&value.raw_gradient),
            hessian,
            bias_hessian,
        })
    }

    /// An orthonormal basis (columns) of the directions across the surface at `x`, in
    /// mass-weighted coordinates.
    pub fn normals(&self, x: &Col<f64>) -> MoptResult<Mat<f64>> {
        let sys = self.template.with_flat_coords_bohr(x)?;
        let zero = Col::zeros(x.nrows());
        let frame = self.set.frame_in(&sys, &zero, Some(&self.weights))?;
        Ok(self.normals_of(&frame.jacobian))
    }

    fn normals_of(&self, jacobian: &Mat<f64>) -> Mat<f64> {
        let n = jacobian.ncols();
        let rows: Vec<Col<f64>> = (0..jacobian.nrows())
            .map(|k| Col::from_fn(n, |i| jacobian[(k, i)] / self.sqrt_mass[i]))
            .collect();
        orthonormalize_columns(n, &rows, 1e-10)
    }

    /// The mass-weighted Hessian `mw` on the surface with the normals `q`:
    /// `P mw P + s Q Q^T` (the stiffness `s` only with `stiff`).
    pub fn on_surface(mw: &Mat<f64>, q: &Mat<f64>, stiff: bool) -> Mat<f64> {
        let n = mw.nrows();
        let p = Mat::<f64>::identity(n, n) - q * q.transpose();
        let mut out = &p * mw * &p;
        if stiff && q.ncols() > 0 {
            // The stiffness from the diagonal of the Hessian before the projection, as
            // `opt::constrained::ConstraintProjection::hessian` takes it.
            let diagonal = (0..n).fold(0.0_f64, |m, i| m.max(mw[(i, i)].abs()));
            out += q * q.transpose() * faer::Scale(STIFFNESS * diagonal.max(0.5));
        }
        symmetrize(&mut out);
        out
    }

    /// `v` without its parts along the normals `q`.
    pub fn along_surface(v: &Col<f64>, q: &Mat<f64>) -> Col<f64> {
        if q.ncols() == 0 {
            return v.clone();
        }
        v - q * (q.transpose() * v)
    }

    /// The Cartesian Hessian `h` on the surface: `M^(1/2) on_surface(M^(-1/2) h M^(-1/2))
    /// M^(1/2)`.
    fn projected(&self, h: &Mat<f64>, q: &Mat<f64>, stiff: bool) -> Mat<f64> {
        let n = h.nrows();
        let s = &self.sqrt_mass;
        let mw = Mat::from_fn(n, n, |i, j| h[(i, j)] / (s[i] * s[j]));
        let out = Self::on_surface(&mw, q, stiff);
        Mat::from_fn(n, n, |i, j| out[(i, j)] * s[i] * s[j])
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{AtomId, Element, SpinCharge};
    use crate::pes::constraint::{Constraint, ConstraintKind};

    fn water() -> MolecularSystem {
        let x = Col::from_fn(9, |k| [0.0, 0.0, 0.2, 1.4, 0.0, -0.9, -1.4, 0.1, -0.9][k]);
        let elements = vec![
            Element::parse_token("O").unwrap(),
            Element::parse_token("H").unwrap(),
            Element::parse_token("H").unwrap(),
        ];
        MolecularSystem::from_flat_bohr(elements, &x, SpinCharge::default(), None).unwrap()
    }

    fn sqrt_masses(sys: &MolecularSystem) -> Vec<f64> {
        sys.masses_amu()
            .iter()
            .flat_map(|m| [m.sqrt(); 3])
            .collect()
    }

    #[test]
    fn the_gradient_and_the_hessian_keep_to_the_surface_and_a_step_comes_back() {
        let sys = water();
        let one = AtomId::from_one_based(1).unwrap();
        let two = AtomId::from_one_based(2).unwrap();
        let distance = Constraint::at_start(ConstraintKind::Distance(vec![one], vec![two]));
        let set = ConstraintSet::new(&[distance], &sys, None).unwrap();
        let c = IrcConstraints::new(set, sys.clone(), &sqrt_masses(&sys));
        let x = sys.coords_flat_bohr();
        let g = Col::from_fn(9, |i| 0.1 * (i as f64 + 1.0).sin());
        let h = Mat::from_fn(
            9,
            9,
            |i, j| if i == j { 1.0 + 0.1 * i as f64 } else { 0.02 },
        );
        let value = PointValue {
            hessian: Some(h),
            ..PointValue::plain(-1.0, g.clone(), None)
        };
        let projected = c.project(&x, value).unwrap();
        // The mass-weighted gradient has nothing along the gradient of the distance.
        let s = sqrt_masses(&sys);
        let along = |v: &Col<f64>| -> f64 {
            // d(distance)/dx of atoms 1 and 2, mass weighted.
            let (a, b) = (sys.position(one).value(), sys.position(two).value());
            let d = ((0..3).map(|k| (b[k] - a[k]).powi(2)).sum::<f64>()).sqrt();
            (0..3)
                .map(|k| {
                    let u = (b[k] - a[k]) / d;
                    (-u * v[k] / s[k]) / s[k] + (u * v[3 + k] / s[3 + k]) / s[3 + k]
                })
                .sum()
        };
        assert!(along(&projected.gradient).abs() < 1e-12);
        assert!(along(&projected.raw_gradient).abs() < 1e-12);
        assert!(along(&g).abs() > 1e-3);
        assert_eq!(projected.energy, -1.0);
        // The Hessian is symmetric and stiff across the surface.
        let hp = projected.hessian.unwrap();
        for i in 0..9 {
            for j in 0..9 {
                assert!((hp[(i, j)] - hp[(j, i)]).abs() < 1e-12);
            }
        }
        // A step off the surface comes back onto it.
        let mut moved = x.clone();
        moved[3] += 0.2;
        let back = c.restore(&moved).unwrap();
        assert!(c.largest_residual(&moved).unwrap() > 0.1);
        assert!(c.largest_residual(&back).unwrap() < 1e-9);
        // The way back is the shortest in mass-weighted coordinates: the light hydrogen
        // moves much more than the oxygen.
        let shift = |k: usize| (back[k] - moved[k]).abs();
        assert!(shift(3) > 10.0 * shift(0));
    }
}
