//! Damped (quasi-)Newton step with a curvature-corrected Hessian, and its momentum variant.
//!
//! In the eigenbasis of the Hessian `H = sum_i l_i v_i v_i^T` the step is
//! `dx = -delta sum_i sigma_i (v_i . g) / |l_i| v_i` with `sigma_i = -1` for the `saddle_order`
//! lowest modes (which are maximized) and `+1` for the others: the Newton step
//! `-delta H^-1 g` whenever the Hessian has the wanted inertia, and a step with the right sign
//! along every mode otherwise (the modified Newton method of Greenstadt, Math. Comput. 1967, 21,
//! 360). Modes with `|l_i|` below `1e-8` (rigid-body motion) are left out. `delta = 0.5` as in
//! MultiOptPy.
//!
//! Momentum variant (`mnewton_<updater>`): the Hessian is updated with the secant pairs of the
//! exponentially averaged sequences `x~_k = b x~_{k-1} + (1 - b) x_k` and
//! `g~_k = b g~_{k-1} + (1 - b) g_k`, i.e. `(x~_k - x~_{k-1}, g~_k - g~_{k-1})`, which satisfy
//! the same secant equation for a quadratic and average out noise in the gradients
//! (MultiOptPy cites arXiv:2307.13744 for its momentum mode).
//!
//! MultiOptPy's class solves `H dx = g` directly, which fails for the singular Hessian of a
//! molecule and converges to whatever stationary point the inertia of `H` selects; its momentum
//! mode mixes the averaged displacement with the absolute geometry. The class is not reachable
//! from MultiOptPy's command line; mopt-rs registers it as `newton_<updater>` and
//! `mnewton_<updater>`.
//!
//! MultiOptPy: `Optimizer/newton.py:Newton`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;
use crate::math::linalg::{symmetric_eigen, symmetrize};
use crate::math::projection::{project_out_basis_mat, project_out_basis_vec};
use crate::opt::update::{apply_update, project_out_rigid, HessianUpdater};

/// Modes with `|lambda|` below this are null modes.
const NULL_MODE: f64 = 1e-8;

/// Newton step `-delta sum_i sigma_i (v_i . g) / |l_i| v_i` for a symmetric `hessian`; the
/// `saddle_order` lowest non-null modes are maximized.
pub fn modified_newton_step(
    hessian: &Mat<f64>,
    gradient: &Col<f64>,
    saddle_order: usize,
    delta: f64,
) -> MoptResult<Col<f64>> {
    let n = hessian.nrows();
    let (vals, vecs) = symmetric_eigen(hessian)?;
    let mut step: Col<f64> = Col::zeros(n);
    let mut maximized = 0;
    for k in 0..n {
        let lam = vals[k];
        if lam.abs() < NULL_MODE || !lam.is_finite() {
            continue;
        }
        let sign = if maximized < saddle_order {
            maximized += 1;
            -1.0
        } else {
            1.0
        };
        let gk: f64 = (0..n).map(|r| vecs[(r, k)] * gradient[r]).sum();
        let c = -delta * sign * gk / lam.abs();
        for r in 0..n {
            step[r] += c * vecs[(r, k)];
        }
    }
    Ok(step)
}

struct Momentum {
    beta: f64,
    updater: Box<dyn HessianUpdater>,
    hessian: Option<Mat<f64>>,
    /// Averaged position (relative to the first point) and gradient.
    average: Option<(Col<f64>, Col<f64>)>,
    /// Current position relative to the first point.
    position: Col<f64>,
}

/// Newton step generator.
pub struct Newton {
    /// Step factor `delta`.
    pub delta: f64,
    momentum: Option<Momentum>,
}

impl Newton {
    /// Newton steps with the optimizer's (quasi-Newton, model, or exact) Hessian.
    pub fn new(delta: f64) -> Self {
        Self {
            delta,
            momentum: None,
        }
    }

    /// Newton steps with a Hessian updated from exponentially averaged points and gradients
    /// (`beta` in `[0, 1)`; `0` is the plain secant update).
    pub fn with_momentum(delta: f64, beta: f64, updater: Box<dyn HessianUpdater>) -> Self {
        Self {
            delta,
            momentum: Some(Momentum {
                beta: beta.clamp(0.0, 1.0 - 1e-12),
                updater,
                hessian: None,
                average: None,
                position: Col::zeros(0),
            }),
        }
    }
}

impl Default for Newton {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl StepGenerator for Newton {
    fn set_step_factor(&mut self, delta: f64) -> bool {
        self.delta = delta;
        true
    }

    fn name(&self) -> &str {
        if self.momentum.is_some() {
            "mnewton"
        } else {
            "newton"
        }
    }

    fn uses_hessian(&self) -> bool {
        true
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let n = input.gradient.nrows();
        let base = match &mut self.momentum {
            None => input.hessian.clone(),
            Some(m) => {
                let mut h = m.hessian.take().unwrap_or_else(|| input.hessian.clone());
                m.position = match input.previous {
                    Some(prev) if m.position.nrows() == n => &m.position + &prev.step,
                    _ => Col::zeros(n),
                };
                let g = input.raw_gradient;
                let new = match m.average.take() {
                    Some((ax, ag)) if ax.nrows() == n => {
                        let nx = &ax * m.beta + &m.position * (1.0 - m.beta);
                        let ng = &ag * m.beta + g * (1.0 - m.beta);
                        apply_update(
                            m.updater.as_mut(),
                            &mut h,
                            &(&nx - &ax),
                            &(&ng - &ag),
                            false,
                        )?;
                        // The kept Hessian has no curvature along the rigid motions of this
                        // structure, updated or not.
                        if input.project_updates {
                            project_out_rigid(&mut h, input.rigid_body_basis);
                        }
                        (nx, ng)
                    }
                    _ => (m.position.clone(), g.clone()),
                };
                m.average = Some(new);
                m.hessian = Some(h.clone());
                h
            }
        };
        let gradient = match input.rigid_body_basis {
            Some(d) => project_out_basis_vec(d, input.gradient),
            None => input.gradient.clone(),
        };
        let mut h = match input.bias_hessian {
            Some(b) => &base + b,
            None => base,
        };
        if let Some(d) = input.rigid_body_basis {
            h = project_out_basis_mat(d, &h);
        }
        symmetrize(&mut h);
        modified_newton_step(&h, &gradient, input.saddle_order, self.delta)
    }

    fn reset(&mut self) {
        if let Some(m) = &mut self.momentum {
            m.hessian = None;
            m.average = None;
            m.position = Col::zeros(0);
            m.updater.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::linalg::col_norm;
    use crate::opt::step::PreviousPoint;
    use crate::opt::stepper::plain_input;
    use crate::opt::update::UpdaterRegistry;

    #[test]
    fn full_newton_step_reaches_the_stationary_point_of_any_inertia() {
        // Indefinite form: a first-order saddle at c.
        let a = Mat::from_fn(3, 3, |i, j| {
            [[-1.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 3.0]][i][j]
        });
        let c = Col::from_fn(3, |i| [0.3, -0.2, 0.5][i]);
        let x: Col<f64> = Col::zeros(3);
        let g = &a * (&x - &c);
        let s = modified_newton_step(&a, &g, 1, 1.0).unwrap();
        assert!(col_norm(&(&x + &s - &c)) < 1e-12);
        // As a minimizer on the same surface the step is downhill along every mode.
        let s = modified_newton_step(&a, &g, 0, 1.0).unwrap();
        assert!(crate::math::linalg::col_dot(&s, &g) < 0.0);
        // A singular Hessian (null mode) is no obstacle.
        let sing = Mat::from_fn(2, 2, |i, j| if i == 0 && j == 0 { 2.0 } else { 0.0 });
        let g = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let s = modified_newton_step(&sing, &g, 0, 1.0).unwrap();
        assert!((s[0] + 0.5).abs() < 1e-15 && s[1] == 0.0);
    }

    #[test]
    fn momentum_pairs_satisfy_the_secant_equation_of_a_quadratic() {
        let a = Mat::from_fn(3, 3, |i, j| if i == j { [1.0, 2.0, 4.0][i] } else { 0.2 });
        let updater = UpdaterRegistry::default().resolve("bfgs").unwrap();
        let mut newton = Newton::with_momentum(0.5, 0.5, updater);
        let h0 = Mat::<f64>::identity(3, 3);
        let mut x = Col::from_fn(3, |i| [1.0, -1.0, 0.5][i]);
        let mut prev: Option<PreviousPoint> = None;
        for it in 0..60 {
            let g = &a * &x;
            let s = newton
                .propose(&plain_input(it, &x, 0.0, &g, &h0, prev.as_ref(), 0))
                .unwrap();
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::zero(),
                gradient: g.clone(),
                raw_gradient: g,
                step: s.clone(),
            });
            x += &s;
        }
        assert!(col_norm(&(&a * &x)) < 1e-8, "{}", col_norm(&(&a * &x)));
        assert_eq!(newton.name(), "mnewton");
    }
}
