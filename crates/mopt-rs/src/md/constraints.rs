//! Holonomic constraints in molecular dynamics: SHAKE and RATTLE in matrix form.
//!
//! The constraints `c(x) = 0` are those of [`ConstraintSet`] (distances, angles, dihedrals,
//! Cartesian components, orientations, user functions; the atoms of `-fix` are held by an
//! inverse mass of zero instead) and linear rows `r . x = t` (the frozen modes of
//! [`super::freeze`]). After a drift from `x_old` the positions are moved onto the
//! constraints along `M^-1 J(x_old)^T` by Newton iterations on the multipliers (SHAKE;
//! Ryckaert, Ciccotti, Berendsen 1977; matrix form as in M-SHAKE), and the velocities by the
//! same displacement over the time of the drift. After a kick or a thermostat the
//! velocities are projected onto the tangent space in the metric of the masses (RATTLE;
//! Andersen 1983): `v <- v - M^-1 J^T (J M^-1 J^T)^+ J v`. The forces of the constraints
//! are therefore those of the Lagrange multipliers, and both positions and velocities stay
//! on the constraint surface.
//!
//! MultiOptPy: `Constraint/constraint_condition.py:SHAKE` (`shake_parser`, `run`) and
//! `ProjectOutConstrain` as used in `moleculardynamics.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::pseudo_inverse;
use crate::pes::constraint::ConstraintSet;

/// A linear constraint `row . x = target`.
#[derive(Debug, Clone)]
pub struct LinearRow {
    /// The gradient of the constraint.
    pub row: Col<f64>,
    /// The value held.
    pub target: f64,
}

/// Largest residual (Bohr, radians, or the unit of a user function) after SHAKE; the
/// iterations then go on to the rounding floor of the constraints (distances of a few Bohr:
/// about 1e-15). The constraints are hard: they hold their values at every step to machine
/// precision.
pub const DEFAULT_TOLERANCE: f64 = 1e-12;
/// A residual that stops falling below this is the rounding floor of a constraint function
/// with noise of its own (functions computed through a superposition carry about 1e-9 near
/// linear structures), not a failure of the iterations.
const STALLED_FLOOR: f64 = 1e-9;
/// Newton iterations of SHAKE before it gives up.
pub const MAX_ITERATIONS: usize = 100;

/// The holonomic constraints of a run.
#[derive(Debug, Clone, Default)]
pub struct Rattle {
    sets: Vec<ConstraintSet>,
    template: Option<MolecularSystem>,
    linear: Vec<LinearRow>,
    tolerance: f64,
    lincs: Option<super::lincs::Lincs>,
}

/// `(A + A^T) / 2` pseudo-inverse with eigenvalues below `1e-10` of the largest dropped, and
/// the number kept.
fn symmetric_pinv(a: &Mat<f64>) -> MoptResult<(Mat<f64>, usize)> {
    let n = a.nrows();
    let s = Mat::from_fn(n, n, |i, j| 0.5 * (a[(i, j)] + a[(j, i)]));
    crate::math::linalg::psd_pseudo_inverse(&s, 1e-10)
}

impl Rattle {
    /// The constraints of `sets`, evaluated on structures like `template` (elements,
    /// charge, cell), and none in linear form.
    pub fn new(sets: Vec<ConstraintSet>, template: Option<MolecularSystem>) -> MoptResult<Self> {
        if sets.iter().any(|s| !s.is_empty()) && template.is_none() {
            return Err(MoptError::config(
                "geometric constraints need atoms; plain variables can only freeze modes",
            ));
        }
        Ok(Self {
            sets: sets.into_iter().filter(|s| !s.is_empty()).collect(),
            template,
            linear: Vec::new(),
            tolerance: DEFAULT_TOLERANCE,
            lincs: None,
        })
    }

    /// Hold the bond lengths with `lincs` instead of SHAKE and RATTLE (while there are no
    /// linear rows; the constraint sets must be the same bonds).
    pub fn use_lincs(&mut self, lincs: super::lincs::Lincs) {
        self.lincs = Some(lincs);
    }

    /// The LINCS solver in use, when there is one and no linear rows.
    fn lincs(&self) -> Option<&super::lincs::Lincs> {
        self.lincs.as_ref().filter(|_| self.linear.is_empty())
    }

    /// Add the constraints of `set`.
    pub fn add_set(&mut self, set: ConstraintSet) -> MoptResult<()> {
        if set.is_empty() {
            return Ok(());
        }
        if self.template.is_none() {
            return Err(MoptError::config("geometric constraints need atoms"));
        }
        self.sets.push(set);
        Ok(())
    }

    /// Replace the linear rows.
    pub fn set_linear(&mut self, rows: Vec<LinearRow>) {
        self.linear = rows;
    }

    /// Set the largest residual left by SHAKE.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    /// Number of scalar constraints.
    pub fn len(&self) -> usize {
        self.sets.iter().map(ConstraintSet::len).sum::<usize>() + self.linear.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the geometric constraints pin the structure in space.
    pub fn breaks_rigid_body(&self) -> bool {
        self.sets.iter().any(ConstraintSet::breaks_rigid_body)
    }

    fn system(&self, x: &Col<f64>) -> MoptResult<Option<MolecularSystem>> {
        match &self.template {
            Some(t) if !self.sets.is_empty() => Ok(Some(t.with_flat_coords_bohr(x)?)),
            _ => Ok(None),
        }
    }

    /// Residuals `c(x)` of all constraints.
    pub fn residuals(&self, x: &Col<f64>) -> MoptResult<Col<f64>> {
        let mut out = Vec::with_capacity(self.len());
        if let Some(sys) = self.system(x)? {
            for set in &self.sets {
                out.extend(set.residuals(&sys)?.iter().copied());
            }
        }
        for r in &self.linear {
            out.push(r.row.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f64>() - r.target);
        }
        Ok(Col::from_fn(out.len(), |k| out[k]))
    }

    /// Gradients of the constraints (rows) at `x`.
    pub fn jacobian(&self, x: &Col<f64>) -> MoptResult<Mat<f64>> {
        let n = x.nrows();
        let mut rows: Vec<Col<f64>> = Vec::with_capacity(self.len());
        if let Some(sys) = self.system(x)? {
            let zero = Col::zeros(n);
            for set in &self.sets {
                let j = set.frame(&sys, &zero)?.jacobian;
                for k in 0..j.nrows() {
                    rows.push(Col::from_fn(n, |i| j[(k, i)]));
                }
            }
        }
        rows.extend(self.linear.iter().map(|r| r.row.clone()));
        Ok(Mat::from_fn(rows.len(), n, |k, i| rows[k][i]))
    }

    /// Largest absolute residual at `x`.
    pub fn max_residual(&self, x: &Col<f64>) -> MoptResult<f64> {
        Ok(self
            .residuals(x)?
            .iter()
            .fold(0.0_f64, |m, r| m.max(r.abs())))
    }

    /// Number of independent constraints at `x` with the inverse masses `w` (the degrees of
    /// freedom they take away).
    pub fn rank(&self, x: &Col<f64>, w: &[f64]) -> MoptResult<usize> {
        if self.is_empty() {
            return Ok(0);
        }
        let j = self.jacobian(x)?;
        Ok(symmetric_pinv(&gram(&j, &j, w))?.1)
    }

    /// SHAKE after a drift of duration `h` from `x_old`: `x` is moved onto the constraints
    /// along `M^-1 J(x_old)^T` and `v` corrected by the same displacement over `h`.
    /// Returns the Newton iterations taken.
    pub fn shake(
        &self,
        x_old: &Col<f64>,
        x: &mut Col<f64>,
        v: &mut Col<f64>,
        h: f64,
        w: &[f64],
    ) -> MoptResult<usize> {
        if self.is_empty() {
            return Ok(0);
        }
        if let Some(l) = self.lincs() {
            l.constrain(x_old, x, v, h, w)?;
            return Ok(1);
        }
        self.shake_matrix(x_old, x, v, h, w)
    }

    /// SHAKE by Newton iterations on the multipliers.
    fn shake_matrix(
        &self,
        x_old: &Col<f64>,
        x: &mut Col<f64>,
        v: &mut Col<f64>,
        h: f64,
        w: &[f64],
    ) -> MoptResult<usize> {
        let j_old = self.jacobian(x_old)?;
        let m = j_old.nrows();
        // G = M^-1 J_old^T (n x m).
        let g = Mat::from_fn(x.nrows(), m, |i, k| w[i] * j_old[(k, i)]);
        let free = x.clone();
        let mut lambda = Col::<f64>::zeros(m);
        let worst_of = |r: &Col<f64>| r.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let mut residual = self.residuals(x)?;
        let mut worst = worst_of(&residual);
        let mut iterations = 0;
        // Newton iterations until the residuals are below the tolerance and then until they
        // no longer fall (the rounding floor of evaluating the constraints: the iteration
        // converges quadratically, so one or two more steps reach it). A residual that stops
        // falling above the tolerance is that floor too (a function with rounding noise of
        // its own), below `STALLED_FLOOR`.
        let mut stalled = 0;
        loop {
            let below = worst <= self.tolerance;
            if (below || worst < STALLED_FLOOR) && stalled >= 1 {
                break;
            }
            if below && worst == 0.0 {
                break;
            }
            if iterations == MAX_ITERATIONS {
                if worst <= self.tolerance {
                    break;
                }
                return Err(MoptError::conv(format!(
                    "SHAKE did not converge in {MAX_ITERATIONS} iterations (largest residual \
                     {worst:.3e}); a smaller time step keeps the constraints"
                )));
            }
            let a = &self.jacobian(x)? * &g;
            let next_lambda = &lambda + &pseudo_inverse(&a)? * &residual;
            let next_x = &free - &g * &next_lambda;
            let next_residual = self.residuals(&next_x)?;
            let next_worst = worst_of(&next_residual);
            iterations += 1;
            if next_worst < 0.5 * worst {
                // Still falling: take it.
                stalled = 0;
                lambda = next_lambda;
                *x = next_x;
                residual = next_residual;
                worst = next_worst;
            } else {
                // No longer falling: the last point is at the floor.
                if next_worst < worst {
                    lambda = next_lambda;
                    *x = next_x;
                    residual = next_residual;
                    worst = next_worst;
                }
                stalled += 1;
            }
        }
        if iterations > 0 && h > 0.0 {
            *v -= &g * &lambda * faer::Scale(1.0 / h);
        }
        Ok(iterations)
    }

    /// RATTLE: `v` projected onto the tangent space of the constraints at `x` in the metric
    /// of the masses.
    pub fn project_velocities(&self, x: &Col<f64>, v: &mut Col<f64>, w: &[f64]) -> MoptResult<()> {
        if self.is_empty() {
            return Ok(());
        }
        if let Some(l) = self.lincs() {
            return l.project_velocities(x, v, w);
        }
        let j = self.jacobian(x)?;
        let (inverse, _) = symmetric_pinv(&gram(&j, &j, w))?;
        let mu = &inverse * (&j * &*v);
        for i in 0..v.nrows() {
            let push: f64 = (0..j.nrows()).map(|k| j[(k, i)] * mu[k]).sum();
            v[i] -= w[i] * push;
        }
        Ok(())
    }

    /// `x` moved onto the constraints along the directions of the masses (the start of a
    /// run with constraint values other than the starting ones).
    pub fn restore(&self, x: &mut Col<f64>, w: &[f64]) -> MoptResult<usize> {
        let old = x.clone();
        let mut v = Col::zeros(x.nrows());
        // The exact solver: the start may be far from the constraints.
        self.shake_matrix(&old, x, &mut v, 1.0, w)
    }
}

/// `A M^-1 B^T` for constraint gradients `A`, `B` and inverse masses `w`.
fn gram(a: &Mat<f64>, b: &Mat<f64>, w: &[f64]) -> Mat<f64> {
    Mat::from_fn(a.nrows(), b.nrows(), |k, l| {
        (0..a.ncols()).map(|i| a[(k, i)] * w[i] * b[(l, i)]).sum()
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, AtomId, Element, SpinCharge};
    use crate::pes::constraint::{Constraint, ConstraintKind};

    #[test]
    fn shake_and_rattle_keep_a_bond_and_its_rate() {
        let sys = MolecularSystem::new(
            vec![
                Element::parse_token("O").unwrap(),
                Element::parse_token("H").unwrap(),
            ],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let bond = Constraint::at_start(ConstraintKind::Distance(
            vec![AtomId::new(0)],
            vec![AtomId::new(1)],
        ));
        let set = ConstraintSet::new(&[bond], &sys, None).unwrap();
        let rattle = Rattle::new(vec![set], Some(sys.clone())).unwrap();
        let w: Vec<f64> = [16.0, 16.0, 16.0, 1.0, 1.0, 1.0]
            .iter()
            .map(|m| 1.0 / m)
            .collect();
        let x0 = sys.coords_flat_bohr();
        let mut v = Col::from_fn(6, |k| [0.0, 0.1, 0.0, 0.3, -0.2, 0.05][k]);
        rattle.project_velocities(&x0, &mut v, &w).unwrap();
        let j = rattle.jacobian(&x0).unwrap();
        assert!((0..6).map(|i| j[(0, i)] * v[i]).sum::<f64>().abs() < 1e-14);
        let mut x = &x0 + &v * faer::Scale(0.5);
        let iterations = rattle.shake(&x0, &mut x, &mut v, 0.5, &w).unwrap();
        assert!(iterations > 0);
        assert!(rattle.max_residual(&x).unwrap() < DEFAULT_TOLERANCE);
        // The heavy atom moved less than the light one.
        let moved = |k: usize| {
            (0..3)
                .map(|c| (x[3 * k + c] - x0[3 * k + c]).powi(2))
                .sum::<f64>()
        };
        assert!(moved(0) < moved(1));
        assert_eq!(rattle.rank(&x, &w).unwrap(), 1);
    }
}
