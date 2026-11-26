//! One optimization method used on its own: a step generator with its enhancers and Hessian
//! updater, driven by the caller on plain variables.
//!
//! [`Stepper::propose`] takes the current point, its energy and gradient, and returns the step
//! the method proposes. Nothing else of the optimizer state machines is involved (no trust
//! radius, no convergence test, no coordinate system), so that a method can be embedded in
//! another program or examined in isolation. The step actually taken may differ from the
//! proposal: the next call measures it from the new point.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::enhance::StepEnhancer;
use super::registry::{MethodContext, MethodRegistry};
use super::step::{PreviousPoint, StepGenerator, StepInput};
use super::trust::TrustSettings;
use super::update::{apply_update, project_out_rigid, HessianUpdater};
use crate::domain::Quantity;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::math::linalg::col_norm;

/// Input of a step proposal on plain variables (no atoms, no coordinate system).
pub fn plain_input<'a>(
    iteration: usize,
    x: &'a Col<f64>,
    energy: f64,
    gradient: &'a Col<f64>,
    hessian: &'a Mat<f64>,
    previous: Option<&'a PreviousPoint>,
    saddle_order: usize,
) -> StepInput<'a> {
    StepInput {
        iteration,
        cartesian: x,
        energy: Quantity::new(energy),
        gradient,
        raw_gradient: gradient,
        previous,
        hessian,
        bias_hessian: None,
        rigid_body_basis: None,
        project_updates: true,
        saddle_order,
        system: None,
        cartesian_coordinates: true,
        frame: None,
        cartesian_gradient: Some(gradient),
    }
}

/// A step generator with enhancers and a Hessian updater, used directly.
pub struct Stepper {
    generator: Box<dyn StepGenerator>,
    enhancers: Vec<Box<dyn StepEnhancer>>,
    updater: Option<Box<dyn HessianUpdater>>,
    hessian: Mat<f64>,
    saddle_order: usize,
    iteration: usize,
    /// Last point: position, energy, gradient.
    last: Option<(Col<f64>, f64, Col<f64>)>,
    max_step: Option<f64>,
    /// Orthonormal columns along which the function does not change (the rigid motions of
    /// variables that describe a structure); none by default.
    rigid_basis: Option<Mat<f64>>,
    /// Whether `rigid_basis` is taken out of the Hessian after every update.
    project_updates: bool,
}

impl Stepper {
    /// The method `name` (as `-opt`) on `n` variables, resolved in the default registry.
    pub fn new(name: &str, n: usize, saddle_order: usize) -> MoptResult<Self> {
        Self::with_registry(name, n, saddle_order, &MethodRegistry::default())
    }

    /// The method `name` on `n` variables, resolved in `registry`.
    pub fn with_registry(
        name: &str,
        n: usize,
        saddle_order: usize,
        registry: &MethodRegistry,
    ) -> MoptResult<Self> {
        let ctx = MethodContext {
            saddle_order,
            trust: TrustSettings::default(),
            ccqn: None,
        };
        let m = registry.resolve(name, &ctx)?;
        Ok(Self::from_parts(
            m.generator,
            m.enhancers,
            m.updater,
            n,
            saddle_order,
        ))
    }

    /// A stepper from its parts; the Hessian starts as the identity (times the scale the
    /// generator asks for).
    pub fn from_parts(
        generator: Box<dyn StepGenerator>,
        enhancers: Vec<Box<dyn StepEnhancer>>,
        updater: Option<Box<dyn HessianUpdater>>,
        n: usize,
        saddle_order: usize,
    ) -> Self {
        let scale = generator.initial_hessian_scale().unwrap_or(1.0);
        Self {
            generator,
            enhancers,
            updater,
            hessian: Mat::<f64>::identity(n, n) * scale,
            saddle_order,
            iteration: 0,
            last: None,
            max_step: None,
            rigid_basis: None,
            project_updates: true,
        }
    }

    /// Shorten every proposed step to at most this length (the role of the trust radius in
    /// the optimizers).
    pub fn with_max_step(mut self, max_step: f64) -> Self {
        self.max_step = Some(max_step);
        self
    }

    /// Whether the rigid motions of [`Self::set_rigid_basis`] are taken out of the Hessian
    /// after every update (on by default; without a basis nothing is taken out).
    pub fn with_project_updates(mut self, on: bool) -> Self {
        self.project_updates = on;
        self
    }

    /// The rigid motions at the point of the next [`Self::propose`], as orthonormal columns
    /// (`n` rows), or `None`: they are taken out of the Hessian after every update and of a
    /// Hessian given by [`Self::set_hessian`], and the step generator gets them as the
    /// optimizers give theirs. For variables that describe a structure the basis changes
    /// with the point and is set before every proposal.
    pub fn set_rigid_basis(&mut self, basis: Option<Mat<f64>>) -> MoptResult<()> {
        let n = self.hessian.nrows();
        if let Some(b) = &basis {
            if b.nrows() != n {
                return Err(MoptError::config(format!(
                    "the rigid motions have {} components, expected {n}",
                    b.nrows()
                )));
            }
        }
        self.rigid_basis = basis;
        Ok(())
    }

    /// Name of the step generator.
    pub fn name(&self) -> &str {
        self.generator.name()
    }

    /// Names of the enhancers in application order.
    pub fn enhancer_names(&self) -> Vec<String> {
        self.enhancers
            .iter()
            .map(|e| e.name().to_string())
            .collect()
    }

    /// Replace the Hessian (an exact or model Hessian); stored secant pairs are forgotten.
    pub fn set_hessian(&mut self, hessian: Mat<f64>) -> MoptResult<()> {
        let n = self.hessian.nrows();
        if hessian.nrows() != n || hessian.ncols() != n {
            return Err(MoptError::config(format!(
                "Hessian is {}x{}, expected {n}x{n}",
                hessian.nrows(),
                hessian.ncols()
            )));
        }
        self.hessian = hessian;
        project_out_rigid(&mut self.hessian, self.rigid_basis.as_ref());
        if let Some(up) = self.updater.as_mut() {
            up.reset();
        }
        Ok(())
    }

    /// Current Hessian.
    pub fn hessian(&self) -> &Mat<f64> {
        &self.hessian
    }

    /// Number of proposals made so far.
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    /// Step proposed at `x` with the given energy and gradient.
    pub fn propose(
        &mut self,
        x: &Col<f64>,
        energy: f64,
        gradient: &Col<f64>,
    ) -> MoptResult<Col<f64>> {
        let n = self.hessian.nrows();
        if x.nrows() != n || gradient.nrows() != n {
            return Err(MoptError::config(format!(
                "point and gradient must have {n} components"
            )));
        }
        let previous = self.last.take().map(|(px, pe, pg)| PreviousPoint {
            energy: Quantity::new(pe),
            gradient: pg.clone(),
            raw_gradient: pg,
            step: x - &px,
        });
        if let (Some(prev), Some(up)) = (&previous, self.updater.as_mut()) {
            let y = gradient - &prev.raw_gradient;
            // As in the optimizers: saddle searches feed pairs with `s . y <= 0` to the formulas
            // without a BFGS term.
            let negative = self.generator.learns_negative_curvature()
                || (self.saddle_order > 0 && up.learns_saddle_curvature());
            apply_update(up.as_mut(), &mut self.hessian, &prev.step, &y, negative)?;
            if self.project_updates {
                project_out_rigid(&mut self.hessian, self.rigid_basis.as_ref());
            }
        }
        let mut input = plain_input(
            self.iteration,
            x,
            energy,
            gradient,
            &self.hessian,
            previous.as_ref(),
            self.saddle_order,
        );
        input.rigid_body_basis = self.rigid_basis.as_ref();
        input.project_updates = self.project_updates;
        let mut step = self
            .generator
            .propose(&input)
            .context(self.generator.name().to_string())?;
        for e in &mut self.enhancers {
            step = e.enhance(&input, step).context(e.name().to_string())?;
        }
        if let Some(max) = self.max_step {
            let len = col_norm(&step);
            if len > max {
                step *= max / len;
            }
        }
        self.last = Some((x.clone(), energy, gradient.clone()));
        self.iteration += 1;
        Ok(step)
    }

    /// Forget all history (the Hessian is kept).
    pub fn reset(&mut self) {
        self.generator.reset();
        for e in &mut self.enhancers {
            e.reset();
        }
        if let Some(up) = self.updater.as_mut() {
            up.reset();
        }
        self.last = None;
        self.iteration = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_method_used_alone_minimizes_a_quadratic() {
        let a = Mat::from_fn(3, 3, |i, j| if i == j { [1.0, 2.0, 4.0][i] } else { 0.1 });
        let mut x = Col::from_fn(3, |i| [1.0, -1.0, 0.5][i]);
        let mut s = Stepper::new("rsirfo_bfgs", 3, 0)
            .unwrap()
            .with_max_step(0.5);
        assert_eq!(s.name(), "rsirfo");
        for _ in 0..100 {
            let g = &a * &x;
            let e = 0.5 * crate::math::linalg::col_dot(&x, &g);
            x += s.propose(&x, e, &g).unwrap();
        }
        assert!(col_norm(&(&a * &x)) < 1e-6, "{}", col_norm(&(&a * &x)));
        assert!(s.set_hessian(Mat::identity(2, 2)).is_err());
        // The same loop as the Python test of `StepMethod`.
        let mut s = Stepper::new("lbfgs_gdiis", 2, 0)
            .unwrap()
            .with_max_step(0.5);
        assert_eq!(
            (s.name(), s.enhancer_names()),
            ("lbfgs", vec!["gdiis".to_string()])
        );
        let mut x = Col::from_fn(2, |i| [1.0, -2.0][i]);
        for _ in 0..60 {
            let g = Col::from_fn(2, |i| [2.0, 8.0][i] * x[i]);
            let e = 0.5 * crate::math::linalg::col_dot(&x, &g);
            x += s.propose(&x, e, &g).unwrap();
        }
        assert!(x[0].abs() < 1e-6 && x[1].abs() < 1e-6, "{x:?}");
        s.reset();
        assert_eq!(s.iterations(), 0);
    }

    #[test]
    fn a_given_basis_is_taken_out_of_the_updates() {
        // A function of the differences of three variables: unchanged along (1, 1, 1).
        let f = |x: &Col<f64>| {
            let (a, b) = (x[0] - x[1], x[1] - x[2]);
            let e = a * a + 2.0 * b * b + 0.3 * a * a * b;
            let g = Col::from_fn(3, |i| match i {
                0 => 2.0 * a + 0.6 * a * b,
                1 => -2.0 * a - 0.6 * a * b + 4.0 * b + 0.3 * a * a,
                _ => -4.0 * b - 0.3 * a * a,
            });
            (e, g)
        };
        let d = Mat::from_fn(3, 1, |_, _| 1.0 / 3.0_f64.sqrt());
        for on in [true, false] {
            let mut s = Stepper::new("rsirfo_bfgs", 3, 0)
                .unwrap()
                .with_max_step(0.2)
                .with_project_updates(on);
            assert!(s.set_rigid_basis(Some(Mat::zeros(2, 1))).is_err());
            s.set_rigid_basis(Some(d.clone())).unwrap();
            let mut x = Col::from_fn(3, |i| [0.4, -0.3, 0.2][i]);
            for _ in 0..6 {
                let (e, g) = f(&x);
                x += s.propose(&x, e, &g).unwrap();
            }
            let along = col_norm(&(s.hessian() * d.col(0)));
            if on {
                assert!(along < 1e-12, "{along:e}");
            } else {
                assert!(along > 1e-3, "{along:e}");
            }
        }
    }
}
