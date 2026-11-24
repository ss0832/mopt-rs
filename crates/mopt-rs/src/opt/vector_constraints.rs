//! Equality constraints `c_k(x) = t_k` of an optimization of plain variables
//! ([`super::vector::VectorOptimizer`]).
//!
//! The same scheme as the constraints of structures ([`crate::pes::constraint`],
//! [`super::constrained`]): the gradients of the constraints `J` are projected out of the
//! gradient (`g - J^T lambda`, `lambda = (J J^T)^+ J g`), the step is taken with the Hessian
//! `P H P + s Q Q^T` (`Q` an orthonormal basis of the constraint gradients, `P = 1 - Q Q^T`,
//! `s` a stiffness along them), and after every step all constraints are restored together
//! by Newton projection, `x <- x - J^T (J J^T)^+ (c(x) - t)`. The quasi-Newton updates see
//! the gradients with the constraint forces removed, so that they learn the curvature of
//! the Lagrangian (that of the constraints included).
//!
//! A constraint is any function of the variables with its gradient ([`VectorConstraint`]):
//! one variable held at a value ([`FixedVariable`]), or a collective variable such as a
//! ReSPer coordinate ([`crate::analysis::resper`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy optimizes atomic coordinates only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, OnceLock};

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{symmetric_eigen, symmetrize};
use crate::pes::constraint::{
    evaluation_noise, held_to_text, rounding_floor, tolerance_note, GUARANTEE, NOISE_MARGIN,
    NOISE_STEP,
};

/// Stiffness along the constrained directions relative to the largest diagonal element of
/// the Hessian (as [`super::constrained`]).
const STIFFNESS: f64 = 100.0;
/// Newton steps of [`VectorConstraintSet::restore`] at most.
const RESTORE_STEPS: usize = 100;

/// A scalar function of the variables that an optimization holds at a value.
pub trait VectorConstraint: Send + Sync {
    /// Name for messages.
    fn name(&self) -> String;

    /// Value and gradient at `x`.
    fn evaluate(&self, x: &Col<f64>) -> MoptResult<(f64, Col<f64>)>;

    /// Period of the value (residuals are wrapped into half a period); `None` for most.
    fn period(&self) -> Option<f64> {
        None
    }

    /// The variable this constraint holds when it is one variable at a value (then the
    /// restore assigns it instead of solving for it, so that it is bit-identical to its
    /// target); `None` for every other function.
    fn fixed_variable(&self) -> Option<usize> {
        None
    }

    /// The largest residual the function is held to: [`GUARANTEE`] unless it was given a
    /// tolerance of its own.
    fn tolerance(&self) -> f64 {
        GUARANTEE
    }

    /// Whether a residual above [`Self::tolerance`] is a warning instead of an error (the
    /// ReSPer coordinates, computed through a superposition).
    fn warns_only(&self) -> bool {
        false
    }

    /// The Hessian of the function at `x` where it has an analytic one, for the curvature of
    /// the constraints in an exact Hessian ([`VectorConstraintSet::curvature`]); `None` (the
    /// default) leaves it to the quasi-Newton updates, which learn the curvature of the
    /// Lagrangian.
    fn hessian(&self, _x: &Col<f64>) -> MoptResult<Option<Mat<f64>>> {
        Ok(None)
    }
}

/// Variable `index` (0-based) held at a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedVariable {
    /// The variable.
    pub index: usize,
}

impl VectorConstraint for FixedVariable {
    fn name(&self) -> String {
        format!("variable {}", self.index + 1)
    }

    fn evaluate(&self, x: &Col<f64>) -> MoptResult<(f64, Col<f64>)> {
        if self.index >= x.nrows() {
            return Err(MoptError::config(format!(
                "constraint on variable {}: there are {} variables",
                self.index + 1,
                x.nrows()
            )));
        }
        let mut g = Col::zeros(x.nrows());
        g[self.index] = 1.0;
        Ok((x[self.index], g))
    }

    fn fixed_variable(&self) -> Option<usize> {
        Some(self.index)
    }
}

/// The constraints of a run with their targets.
#[derive(Clone, Default)]
pub struct VectorConstraintSet {
    rows: Vec<(Arc<dyn VectorConstraint>, f64)>,
    /// The repeatability of each function (`evaluation_noise`), measured at the first
    /// restore (zero for fixed variables).
    noise: OnceLock<Vec<f64>>,
    /// ReSPer coordinates left above [`GUARANTEE`]: (name, largest residual, warned).
    lenient: Arc<std::sync::Mutex<Vec<(String, f64, bool)>>>,
}

impl std::fmt::Debug for VectorConstraintSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows: Vec<(String, f64)> = self.rows.iter().map(|(c, t)| (c.name(), *t)).collect();
        f.debug_struct("VectorConstraintSet")
            .field("rows", &rows)
            .finish()
    }
}

/// Pseudo-inverse of the symmetric positive semi-definite `g`.
fn pinv(g: &Mat<f64>) -> MoptResult<Mat<f64>> {
    Ok(crate::math::linalg::psd_pseudo_inverse(g, 1e-12)?.0)
}

/// The constraint directions at one point.
#[derive(Debug, Clone)]
pub struct VectorConstraintFrame {
    /// Gradients of the constraints (rows).
    pub jacobian: Mat<f64>,
    /// Orthonormal basis of their span (columns).
    pub basis: Mat<f64>,
}

impl VectorConstraintFrame {
    /// `v` without its part along the constraint gradients.
    pub fn project(&self, v: &Col<f64>) -> Col<f64> {
        if self.basis.ncols() == 0 {
            return v.clone();
        }
        v - &self.basis * (self.basis.transpose() * v)
    }

    /// The Hessian for a step: `P H P + s Q Q^T`.
    pub fn step_hessian(&self, h: &Mat<f64>) -> Mat<f64> {
        let n = h.nrows();
        if self.basis.ncols() == 0 {
            return h.clone();
        }
        let p = Mat::<f64>::identity(n, n) - &self.basis * self.basis.transpose();
        let mut out = &p * h * &p;
        symmetrize(&mut out);
        let diag = (0..n).fold(0.0_f64, |a, i| a.max(h[(i, i)].abs()));
        out + &self.basis * self.basis.transpose() * faer::Scale(STIFFNESS * diag.max(0.5))
    }
}

impl VectorConstraintSet {
    /// No constraints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Hold `constraint` at `target`.
    pub fn push(&mut self, constraint: Arc<dyn VectorConstraint>, target: f64) -> MoptResult<()> {
        if !target.is_finite() {
            return Err(MoptError::config(format!(
                "constraint '{}': the target is not finite",
                constraint.name()
            )));
        }
        self.rows.push((constraint, target));
        self.noise = OnceLock::new();
        Ok(())
    }

    /// The repeatability of every function at `x` (measured once, at the first call).
    fn noise_at(&self, x: &Col<f64>) -> MoptResult<&[f64]> {
        if let Some(noise) = self.noise.get() {
            return Ok(noise);
        }
        let measured = self
            .rows
            .iter()
            .map(|(c, _)| {
                if c.fixed_variable().is_some() {
                    Ok(0.0)
                } else {
                    evaluation_noise(|y| c.evaluate(y), x)
                }
            })
            .collect::<MoptResult<Vec<f64>>>()?;
        Ok(self.noise.get_or_init(|| measured))
    }

    /// One warning line for every ReSPer coordinate a restore has left above [`GUARANTEE`]
    /// for the first time since the last call, and the largest residual of each so far
    /// (as [`crate::pes::constraint::ConstraintSet::lenient_notes`]).
    pub fn lenient_notes(&self) -> Vec<String> {
        let Ok(mut seen) = self.lenient.lock() else {
            return Vec::new();
        };
        seen.iter_mut()
            .filter(|(_, _, told)| !*told)
            .map(|(name, residual, told)| {
                *told = true;
                format!(
                    "Warning: constraint '{name}' restored to {residual:.2e}, above the \
                     {GUARANTEE:.0e} hard constraints are held to (its value is computed \
                     through a superposition); the run goes on."
                )
            })
            .collect()
    }

    /// The largest residual of every ReSPer coordinate left above [`GUARANTEE`] so far.
    pub fn lenient_summary(&self) -> Option<String> {
        let seen = self.lenient.lock().ok()?;
        (!seen.is_empty()).then(|| {
            format!(
                "Warning: largest residuals above {GUARANTEE:.0e} in this run: {}.",
                seen.iter()
                    .map(|(l, r, _)| format!("'{l}' {r:.2e}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }

    /// Lines for the log about the floors the functions are restored to: one per function
    /// whose value at `x` repeats only above the rounding floor.
    pub fn notes(&self, x: &Col<f64>) -> MoptResult<Vec<String>> {
        let noise = self.noise_at(x)?.to_vec();
        let mut notes = Vec::new();
        for ((c, _), noise) in self.rows.iter().zip(noise) {
            if c.fixed_variable().is_some() {
                continue;
            }
            if (c.tolerance() - GUARANTEE).abs() > f64::EPSILON * GUARANTEE {
                notes.push(tolerance_note(&format!("'{}'", c.name()), c.tolerance()));
            }
            let (value, g) = c.evaluate(x)?;
            let machine = rounding_floor(g.iter().copied().zip(x.iter().copied()), value);
            if NOISE_MARGIN * noise > machine {
                notes.push(format!(
                    "Constraint '{}': its value repeats to {noise:.1e} under displacements of \
                     {NOISE_STEP:.0e}, above the rounding floor {machine:.1e}; it is restored to \
                     {:.1e}.",
                    c.name(),
                    NOISE_MARGIN * noise
                ));
            }
        }
        Ok(notes)
    }

    /// Number of constraints.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Values minus targets at `x` (periodic ones wrapped into half a period).
    pub fn residuals(&self, x: &Col<f64>) -> MoptResult<Col<f64>> {
        let mut r = Col::zeros(self.rows.len());
        for (k, (c, target)) in self.rows.iter().enumerate() {
            let d = c.evaluate(x)?.0 - target;
            r[k] = match c.period() {
                Some(p) => d - p * (d / p).round(),
                None => d,
            };
        }
        Ok(r)
    }

    /// The constraint gradients at `x`.
    pub fn frame(&self, x: &Col<f64>) -> MoptResult<VectorConstraintFrame> {
        let n = x.nrows();
        let m = self.rows.len();
        let mut j = Mat::zeros(m, n);
        for (k, (c, _)) in self.rows.iter().enumerate() {
            let g = c.evaluate(x)?.1;
            if g.nrows() != n {
                return Err(MoptError::calc(format!(
                    "constraint '{}' returned {} gradient components for {n} variables",
                    c.name(),
                    g.nrows()
                )));
            }
            for i in 0..n {
                j[(k, i)] = g[i];
            }
        }
        // Orthonormal basis of the row space from the eigenvectors of J^T J.
        let (w, v) = symmetric_eigen(&(j.transpose() * &j))?;
        let largest = w.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        let keep: Vec<usize> = (0..n)
            .filter(|&k| w[k] > 1e-12 * largest.max(1e-300))
            .collect();
        let basis = Mat::from_fn(n, keep.len(), |i, c| v[(i, keep[c])]);
        Ok(VectorConstraintFrame { jacobian: j, basis })
    }

    /// The curvature of the constraints for the Hessian of the Lagrangian at `x`, `sum_k
    /// lambda_k d2c_k` with the multipliers `lambda = (J J^T)^+ J g` of the gradient
    /// `gradient` (as [`crate::pes::constraint::ConstraintFrame::curvature`]), over the
    /// functions that give an analytic Hessian ([`VectorConstraint::hessian`]); `None` when
    /// none does. An exact Hessian `H` of the objective becomes `H - curvature`.
    pub fn curvature(&self, x: &Col<f64>, gradient: &Col<f64>) -> MoptResult<Option<Mat<f64>>> {
        let hessians = self
            .rows
            .iter()
            .map(|(c, _)| c.hessian(x))
            .collect::<MoptResult<Vec<Option<Mat<f64>>>>>()?;
        if hessians.iter().all(Option::is_none) {
            return Ok(None);
        }
        let n = x.nrows();
        if gradient.nrows() != n {
            return Err(MoptError::config(format!(
                "the gradient has {} components for {n} variables",
                gradient.nrows()
            )));
        }
        let j = self.frame(x)?.jacobian;
        let lambda = pinv(&(&j * j.transpose()))? * (&j * gradient);
        let mut out = Mat::zeros(n, n);
        for (k, h) in hessians.iter().enumerate() {
            if let Some(h) = h {
                if h.nrows() != n || h.ncols() != n {
                    return Err(MoptError::calc(format!(
                        "constraint '{}' returned a Hessian of {}x{} for {n} variables",
                        self.rows[k].0.name(),
                        h.nrows(),
                        h.ncols()
                    )));
                }
                out += h * faer::Scale(lambda[k]);
            }
        }
        Ok(Some(out))
    }

    /// `x` moved back onto the constraints: fixed variables assigned (bit-identical to their
    /// targets), the other functions solved together by Newton projection for as long as the
    /// largest residual in units of its floor decreases, until every residual is at its floor
    /// (as [`crate::pes::constraint::ConstraintSet::restore_reported`]: the rounding floor
    /// `rounding_floor`, or twice the noise measured at the first restore).
    pub fn restore(&self, x: &Col<f64>) -> MoptResult<Col<f64>> {
        Ok(self.restore_reported(x)?.0)
    }

    /// [`Self::restore`] with the Newton steps taken and the largest residual in units of
    /// its floor at the end.
    pub fn restore_reported(&self, x: &Col<f64>) -> MoptResult<(Col<f64>, usize, f64)> {
        let mut x = x.clone();
        if self.rows.is_empty() {
            return Ok((x, 0, 0.0));
        }
        let n = x.nrows();
        let mut pinned = vec![false; n];
        for (c, target) in &self.rows {
            if let Some(i) = c.fixed_variable() {
                if i >= n {
                    return Err(MoptError::config(format!(
                        "constraint on variable {}: there are {n} variables",
                        i + 1
                    )));
                }
                x[i] = *target;
                pinned[i] = true;
            }
        }
        let solved: Vec<usize> = (0..self.rows.len())
            .filter(|&k| self.rows[k].0.fixed_variable().is_none())
            .collect();
        if solved.is_empty() {
            return Ok((x, 0, 0.0));
        }
        let noise = self.noise_at(&x)?.to_vec();
        // The best point so far: (largest residual in floors, the point, largest residual in
        // units of its tolerance).
        let mut best: Option<(f64, Col<f64>, f64)> = None;
        let mut steps = 0;
        for _ in 0..RESTORE_STEPS {
            let m = solved.len();
            let (mut r, mut floors, mut j) = (Col::zeros(m), Col::zeros(m), Mat::zeros(m, n));
            for (k, &row) in solved.iter().enumerate() {
                let (c, target) = &self.rows[row];
                let (value, g) = c.evaluate(&x)?;
                if g.nrows() != n {
                    return Err(MoptError::calc(format!(
                        "constraint '{}' returned {} gradient components for {n} variables",
                        c.name(),
                        g.nrows()
                    )));
                }
                let d = value - target;
                r[k] = match c.period() {
                    Some(p) => d - p * (d / p).round(),
                    None => d,
                };
                floors[k] = rounding_floor(g.iter().copied().zip(x.iter().copied()), value)
                    .max(NOISE_MARGIN * noise[row]);
                for i in 0..n {
                    // The assigned variables are left out of the Newton step.
                    j[(k, i)] = if pinned[i] { 0.0 } else { g[i] };
                }
            }
            let ratio = (0..m).fold(0.0_f64, |a, k| a.max(r[k].abs() / floors[k]));
            if !ratio.is_finite() {
                break;
            }
            let worst = (0..m).fold(0.0_f64, |a, k| {
                a.max(r[k].abs() / self.rows[solved[k]].0.tolerance())
            });
            match &best {
                // Stalled within the tolerances (above them, Newton steps go on).
                Some(b) if ratio >= b.0 && b.2 <= 1.0 => break,
                Some(b) if ratio >= b.0 => {}
                _ => best = Some((ratio, x.clone(), worst)),
            }
            if ratio <= 1.0 {
                break;
            }
            let dx = j.transpose() * (pinv(&(&j * j.transpose()))? * &r);
            steps += 1;
            x = &x - &dx;
        }
        match best {
            Some((ratio, x, worst)) if worst <= 1.0 => Ok((x, steps, ratio)),
            Some((ratio, x, _)) => {
                // Above the guarantee (or the tolerance of a function): a warning for the
                // functions that only warn (ReSPer coordinates), an error otherwise.
                let r = self.residuals(&x)?;
                let over: Vec<usize> = (0..r.nrows())
                    .filter(|&k| r[k].abs() > self.rows[k].0.tolerance())
                    .collect();
                if over.iter().all(|&k| self.rows[k].0.warns_only()) {
                    if let Ok(mut seen) = self.lenient.lock() {
                        for &k in &over {
                            let name = self.rows[k].0.name();
                            match seen.iter_mut().find(|(l, _, _)| *l == name) {
                                Some(e) => e.1 = e.1.max(r[k].abs()),
                                None => seen.push((name, r[k].abs(), false)),
                            }
                        }
                    }
                    return Ok((x, steps, ratio));
                }
                let excess = |k: usize| r[k].abs() / self.rows[k].0.tolerance();
                let k = over
                    .iter()
                    .copied()
                    .filter(|&k| !self.rows[k].0.warns_only())
                    .max_by(|&a, &b| excess(a).total_cmp(&excess(b)))
                    .unwrap_or(0);
                Err(MoptError::conv(format!(
                    "the constraints could not be restored: '{}' is off by {:.3e} after {steps} \
                     Newton steps, above the {}",
                    self.rows[k].0.name(),
                    r[k].abs(),
                    held_to_text(self.rows[k].0.tolerance())
                )))
            }
            None => Err(MoptError::conv(
                "the constraints could not be restored (their values are not finite)",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// `|x|^2` as a constraint (a sphere).
    struct Sphere;

    impl VectorConstraint for Sphere {
        fn name(&self) -> String {
            "sphere".to_string()
        }

        fn evaluate(&self, x: &Col<f64>) -> MoptResult<(f64, Col<f64>)> {
            let v = x.iter().map(|a| a * a).sum::<f64>();
            Ok((v, x * faer::Scale(2.0)))
        }

        fn hessian(&self, x: &Col<f64>) -> MoptResult<Option<Mat<f64>>> {
            Ok(Some(Mat::<f64>::identity(x.nrows(), x.nrows()) * 2.0))
        }
    }

    /// A noisy linear function, `x_0` plus about 1e-9 of noise.
    struct Noisy;

    impl VectorConstraint for Noisy {
        fn name(&self) -> String {
            "noisy".to_string()
        }

        fn evaluate(&self, x: &Col<f64>) -> MoptResult<(f64, Col<f64>)> {
            let hash = x.iter().fold(0u64, |h, v| {
                (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3)
            });
            let noise = 1e-9 * ((hash % 2001) as f64 / 1000.0 - 1.0);
            Ok((x[0] + noise, Col::from_fn(x.nrows(), |i| f64::from(i == 0))))
        }

        fn tolerance(&self) -> f64 {
            1e-8
        }
    }

    #[test]
    fn a_tolerance_of_its_own_and_the_curvature_of_the_lagrangian() {
        let mut set = VectorConstraintSet::new();
        set.push(Arc::new(Noisy), 0.4).unwrap();
        let x = Col::from_fn(3, |i| [0.3, -0.2, 0.1][i]);
        let y = set.restore(&x).unwrap();
        assert!(set.residuals(&y).unwrap()[0].abs() <= 1e-8);
        assert!(set.notes(&y).unwrap().iter().any(|n| n.contains("1.0e-8")));
        assert!(set.curvature(&y, &Col::zeros(3)).unwrap().is_none());
        // On the sphere |x|^2 = 4 with g = x: lambda = 1/2 and the curvature 2 lambda = 1.
        let mut sphere = VectorConstraintSet::new();
        sphere.push(Arc::new(Sphere), 4.0).unwrap();
        let on = Col::from_fn(3, |i| [2.0, 0.0, 0.0][i]);
        let c = sphere.curvature(&on, &on).unwrap().unwrap();
        for i in 0..3 {
            assert!((c[(i, i)] - 1.0).abs() < 1e-12, "{c:?}");
        }
    }

    #[test]
    fn constraints_are_restored_and_projected_out() {
        let mut set = VectorConstraintSet::new();
        set.push(Arc::new(Sphere), 4.0).unwrap();
        set.push(Arc::new(FixedVariable { index: 2 }), 1.0).unwrap();
        let x = Col::from_fn(3, |i| [0.3, -0.2, 0.0][i]);
        let y = set.restore(&x).unwrap();
        let r = set.residuals(&y).unwrap();
        assert!(r.iter().all(|v| v.abs() < 1e-10), "{r:?}");
        assert!((y[2] - 1.0).abs() < 1e-12);
        // The projected gradient has no part along the constraint gradients.
        let frame = set.frame(&y).unwrap();
        assert_eq!(frame.basis.ncols(), 2);
        let g = Col::from_fn(3, |i| [1.0, 2.0, 3.0][i]);
        let free = frame.project(&g);
        for k in 0..2 {
            let row = Col::from_fn(3, |i| frame.jacobian[(k, i)]);
            let along: f64 = row.iter().zip(free.iter()).map(|(a, b)| a * b).sum();
            assert!(along.abs() < 1e-12, "{along}");
        }
        assert!(set.push(Arc::new(Sphere), f64::NAN).is_err());
        assert!(FixedVariable { index: 5 }.evaluate(&x).is_err());
    }

    #[test]
    fn an_optimization_stays_on_its_constraints() {
        use crate::opt::state_machine::Termination;
        use crate::opt::vector::{VectorAction, VectorOptimizer, VectorSettings};
        // (x - 1)^2 + 2 (y - 2)^2 + (z + 1)^2 on the sphere |x|^2 = 4 with z = 0.5.
        let objective = |x: &Col<f64>| {
            let e = (x[0] - 1.0).powi(2) + 2.0 * (x[1] - 2.0).powi(2) + (x[2] + 1.0).powi(2);
            let g = Col::from_fn(3, |i| {
                [2.0 * (x[0] - 1.0), 4.0 * (x[1] - 2.0), 2.0 * (x[2] + 1.0)][i]
            });
            (e, g)
        };
        let mut set = VectorConstraintSet::new();
        set.push(Arc::new(Sphere), 4.0).unwrap();
        set.push(Arc::new(FixedVariable { index: 2 }), 0.5).unwrap();
        let mut opt = VectorOptimizer::new(
            VectorSettings::default(),
            Col::from_fn(3, |i| [1.0, 1.0, 0.0][i]),
        )
        .unwrap();
        opt.set_constraints(set.clone()).unwrap();
        while let VectorAction::Evaluate { x, .. } = opt.ask() {
            let r = set.residuals(&x).unwrap();
            assert!(r.iter().all(|v| v.abs() < 1e-9), "{r:?}");
            let (e, g) = objective(&x);
            opt.tell(e, g, None).unwrap();
        }
        assert_eq!(opt.termination(), Some(Termination::Converged));
        // At the constrained minimum the gradient is a combination of the constraint
        // gradients: it has no part across them.
        let x = opt.current().clone();
        let free = set.frame(&x).unwrap().project(&objective(&x).1);
        assert!(free.norm_l2() < 1e-3, "{free:?}");
        assert!((x[2] - 0.5).abs() < 1e-10);
    }
}
