//! The relaxations of [`crate::pes::microiter`] by a method of the optimizer registry (the
//! `-opt` names), run by the vector optimizer of [`crate::opt::vector`] with the gradient
//! criterion of the relaxation: [`REGISTRY_METHODS`], which the code that builds the settings
//! of a relaxation hands in (`MicroSettings::methods`), since the surfaces are a layer below
//! the optimizers (`docs/source/architecture.md`). They are the methods of
//! `-oniom_opt micro_optimizer=NAME` and FIRE after a line search of the limited-memory BFGS
//! found no step.
//!
//! The vector optimizer keeps a dense Hessian of the variables, so this is for small
//! environments; larger ones are refused with the reason.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_max_abs;
use crate::opt::convergence::ConvergenceCriteria;
use crate::opt::state_machine::Termination;
use crate::opt::vector::{VectorAction, VectorOptimizer, VectorSettings};
use crate::pes::microiter::{
    is_converged, tell_step, MethodRelaxation, MicroEval, MicroProblem, MicroResult, MicroSettings,
    FALLBACK_STALL_STEPS,
};

/// Variables of a relaxation by a method with a dense Hessian at most (a Hessian of 18 MB,
/// whose eigenvectors a step of RS-I-RFO takes in about a second).
pub const DENSE_VARIABLES: usize = 1500;

/// The relaxations by the methods of the optimizer registry, for `MicroSettings::methods`.
pub const REGISTRY_METHODS: MethodRelaxation = MethodRelaxation(descend_method);

/// The default settings of a relaxation with the methods of the optimizer registry handed in,
/// as the commands have them: for code that builds the relaxations itself (for example
/// `MicroBias::with_settings(bias, opt::micro::settings())`), so that a failed line search goes
/// on with FIRE and `InnerMethod::Method` works, as in the commands.
pub fn settings() -> MicroSettings {
    MicroSettings {
        methods: Some(REGISTRY_METHODS),
        ..MicroSettings::default()
    }
}

/// One relaxation by the method `name` from `y` (no mode kill).
pub fn descend_method(
    problem: &dyn MicroProblem,
    y: Vec<f64>,
    s: &MicroSettings,
    name: &str,
) -> MoptResult<MicroResult> {
    let n = y.len();
    if n == 0 {
        let e = problem.evaluate(&y, false)?;
        return Ok(MicroResult {
            y,
            energy: e.energy,
            max_gradient: 0.0,
            iterations: 0,
            killed_modes: 0,
            converged: true,
            stalled: false,
            line_search_failed: false,
            fallbacks: 0,
        });
    }
    let settings = VectorSettings {
        methods: vec![name.to_string()],
        saddle_order: 0,
        max_iterations: s.max_iterations,
        // The gradient alone decides, as for the other minimizers.
        criteria: ConvergenceCriteria::new(
            s.gradient_tolerance,
            s.rms_tolerance.min(s.gradient_tolerance),
            f64::MAX,
            f64::MAX,
        ),
        ..VectorSettings::default()
    };
    // A method that keeps a Hessian keeps a dense one: small environments only. (The size
    // is tested on one variable, so that no large matrix is made for the test.)
    if n > DENSE_VARIABLES
        && VectorOptimizer::new(settings.clone(), Col::<f64>::zeros(1))?.keeps_hessian()
    {
        return Err(MoptError::config(format!(
            "microiteration: the method '{name}' keeps a dense Hessian of the {n} variables of \
             the relaxation, more than {DENSE_VARIABLES}; lbfgs, tn, and the methods that \
             move by the forces alone (fire) do not"
        )));
    }
    let mut opt = VectorOptimizer::new(settings, Col::from_fn(n, |i| y[i]))?;
    let exact = problem.has_hessian();
    // The point with the lowest largest gradient component: a method without line searches
    // (FIRE) does not end at its best point when it does not converge.
    let mut best: Option<(Vec<f64>, MicroEval, f64)> = None;
    let mut since = 0usize;
    let mut stalled = false;
    let termination = loop {
        match opt.ask() {
            VectorAction::Evaluate { x, need_hessian } => {
                let point: Vec<f64> = x.iter().copied().collect();
                let e = problem.evaluate(&point, need_hessian && exact)?;
                let hessian = if need_hessian && exact {
                    e.hessian.clone()
                } else {
                    None
                };
                opt.tell(e.energy, e.gradient.clone(), hessian)?;
                tell_step(problem, opt.iterations(), e.energy, &e.gradient);
                let gmax = col_max_abs(&e.gradient);
                let lowest = best.as_ref().map_or(f64::INFINITY, |b| b.2);
                if gmax < 0.9 * lowest {
                    since = 0;
                } else {
                    since += 1;
                }
                if gmax < lowest || is_converged(&e.gradient, s) {
                    best = Some((point, e, gmax));
                }
                if since >= FALLBACK_STALL_STEPS {
                    stalled = true;
                    break Termination::MaxIterations;
                }
            }
            VectorAction::Finished(t) => break t,
        }
    };
    let Some((y, e, max_gradient)) = best else {
        return Err(MoptError::calc(format!(
            "microiteration: the method '{name}' made no evaluation"
        )));
    };
    Ok(MicroResult {
        y,
        energy: e.energy,
        max_gradient,
        iterations: opt.iterations(),
        killed_modes: 0,
        converged: termination == Termination::Converged || is_converged(&e.gradient, s),
        stalled,
        line_search_failed: false,
        fallbacks: 0,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use faer::Col;

    use super::*;
    use crate::pes::microiter::{relax, InnerMethod, ModeKill};

    /// `E = sum_i k_i y_i^2 / 2` with its gradient (no Hessian).
    struct Bowl;

    impl MicroProblem for Bowl {
        fn dim(&self) -> usize {
            4
        }
        fn evaluate(&self, y: &[f64], _need_hessian: bool) -> MoptResult<MicroEval> {
            let k = |i: usize| 0.2 + 0.5 * i as f64;
            Ok(MicroEval {
                energy: 0.5 * (0..4).map(|i| k(i) * y[i] * y[i]).sum::<f64>(),
                gradient: Col::from_fn(4, |i| k(i) * y[i]),
                hessian: None,
            })
        }
        fn has_hessian(&self) -> bool {
            false
        }
    }

    fn settings(method: &str, methods: Option<MethodRelaxation>) -> MicroSettings {
        MicroSettings {
            gradient_tolerance: 1e-7,
            max_iterations: 5000,
            mode_kill: ModeKill {
                enabled: false,
                ..ModeKill::default()
            },
            method: InnerMethod::parse(method),
            methods,
            ..MicroSettings::default()
        }
    }

    #[test]
    fn a_method_of_the_registry_relaxes_when_it_is_handed_in() {
        let r = relax(
            &Bowl,
            &[1.0, -0.5, 0.3, 0.8],
            &settings("fire", Some(REGISTRY_METHODS)),
        )
        .unwrap();
        assert!(r.converged && r.max_gradient <= 1e-7, "{r:?}");
    }

    /// `E(y) = (y0^2 - 1)^2 + y1^2 + 0.1 y0 y1^2`: minima at `y0 = +-1`, a saddle at the
    /// origin, with its Hessian.
    struct DoubleWell;

    impl MicroProblem for DoubleWell {
        fn dim(&self) -> usize {
            2
        }
        fn evaluate(&self, y: &[f64], need_hessian: bool) -> MoptResult<MicroEval> {
            let (a, b) = (y[0], y[1]);
            Ok(MicroEval {
                energy: (a * a - 1.0).powi(2) + b * b + 0.1 * a * b * b,
                gradient: Col::from_fn(2, |i| {
                    if i == 0 {
                        4.0 * a * (a * a - 1.0) + 0.1 * b * b
                    } else {
                        2.0 * b + 0.2 * a * b
                    }
                }),
                hessian: need_hessian.then(|| {
                    faer::Mat::from_fn(2, 2, |i, j| match (i, j) {
                        (0, 0) => 12.0 * a * a - 4.0,
                        (1, 1) => 2.0 + 0.2 * a,
                        _ => 0.2 * b,
                    })
                }),
            })
        }
    }

    #[test]
    fn mode_kill_leaves_the_saddle_point_with_a_method_of_the_registry() {
        // Started exactly at the saddle point (zero gradient), as in the test of the
        // relaxation's own minimizers (`pes::microiter`).
        let s = MicroSettings {
            method: InnerMethod::parse("rsirfo_bfgs"),
            gradient_tolerance: 1e-8,
            methods: Some(REGISTRY_METHODS),
            ..MicroSettings::default()
        };
        let r = relax(&DoubleWell, &[0.0, 0.0], &s).unwrap();
        assert!(r.converged, "{r:?}");
        assert!((r.y[0].abs() - 1.0).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn a_method_of_the_registry_is_refused_without_the_optimizers() {
        let e = relax(&Bowl, &[1.0, -0.5, 0.3, 0.8], &settings("fire", None)).unwrap_err();
        assert!(e.to_string().contains("were not given"), "{e}");
        // The relaxation's own minimizer needs nothing.
        let r = relax(&Bowl, &[1.0, -0.5, 0.3, 0.8], &settings("lbfgs", None)).unwrap();
        assert!(r.converged, "{r:?}");
    }
}
