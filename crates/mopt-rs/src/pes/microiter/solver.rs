//! Inner minimizer and mode kill.
//!
//! MultiOptPy: the inner optimizers of `optimization.py:ONIOMHandler.compute`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::memory::MicroMemory;
use super::{InnerCoordinates, InnerMethod, MicroEval, MicroProblem, MicroResult, MicroSettings};
use crate::error::{MoptError, MoptResult};
use crate::math::davidson::{lowest_eigenpair_preconditioned, DavidsonSettings};
use crate::math::linalg::{col_dot, col_max_abs, col_norm, symmetric_eigen};

/// Relax `problem` from `y0`: damped Newton (or limited-memory BFGS) descent, then mode kill.
pub fn relax(
    problem: &dyn MicroProblem,
    y0: &[f64],
    settings: &MicroSettings,
) -> MoptResult<MicroResult> {
    relax_with(problem, y0, settings, &mut MicroMemory::default())
}

/// [`relax`] with the limited-memory BFGS pairs of `memory` at the start, and those it learns
/// left in it at the end.
pub fn relax_with(
    problem: &dyn MicroProblem,
    y0: &[f64],
    settings: &MicroSettings,
    memory: &mut MicroMemory,
) -> MoptResult<MicroResult> {
    relax_in(problem, y0, settings, memory, None)
}

/// [`relax_with`] in the coordinates `coords` (the limited-memory BFGS only; `None`: the
/// Cartesian variables of the problem). Mode kill finds and follows its modes in the
/// variables of the problem.
pub fn relax_in(
    problem: &dyn MicroProblem,
    y0: &[f64],
    settings: &MicroSettings,
    memory: &mut MicroMemory,
    mut coords: Option<&mut dyn InnerCoordinates>,
) -> MoptResult<MicroResult> {
    if y0.len() != problem.dim() {
        return Err(MoptError::calc(format!(
            "microiteration: {} starting values for {} variables",
            y0.len(),
            problem.dim()
        )));
    }
    if coords.is_some() && settings.method != InnerMethod::Lbfgs {
        return Err(MoptError::config(format!(
            "microiteration: other coordinates than the Cartesian ones are used with the \
             limited-memory BFGS, not with '{}'",
            settings.method.name()
        )));
    }
    let mut run = |y: Vec<f64>, memory: &mut MicroMemory| -> MoptResult<MicroResult> {
        let first = match coords.as_deref_mut() {
            Some(c) => {
                memory.fit(c.dim(), Some(c.generation()));
                super::internal::descend_in(problem, y, settings, memory, c)?
            }
            None => {
                memory.fit(problem.dim(), None);
                descend(problem, y, settings, memory)?
            }
        };
        let ended = |r: &MicroResult| {
            problem.progress(&super::MicroProgress::Ended {
                iterations: r.iterations,
                max_gradient: r.max_gradient,
                converged: r.converged,
            });
        };
        if first.converged || !first.line_search_failed {
            ended(&first);
            return Ok(first);
        }
        // A line search found no step that lowers the energy. Where the energy changes by
        // less than its noise (a lowest level in single precision near convergence), the
        // gradient still says where the minimum is: the limited-memory BFGS goes on with a
        // line search on the gradient alone, as FIRE would but with the steps of the model
        // (Cartesian variables only).
        let mut done = first;
        let mut fallbacks = 0;
        if coords.is_none() && settings.method == InnerMethod::Lbfgs {
            problem.progress(&super::MicroProgress::Fallback {
                iteration: done.iterations,
                method: "the limited-memory BFGS with a line search on the gradient alone",
            });
            fallbacks += 1;
            let after = descend_lbfgs_on(problem, done.y.clone(), settings, memory, true)?;
            done = MicroResult {
                iterations: done.iterations + after.iterations,
                fallbacks,
                ..after
            };
            if done.converged || !done.line_search_failed {
                ended(&done);
                return Ok(done);
            }
        }
        // Then the method without line searches, from the point reached, where the
        // optimizers were handed in. The pairs of the model that failed are not kept.
        let Some(methods) = settings.methods else {
            ended(&done);
            return Ok(done);
        };
        problem.progress(&super::MicroProgress::Fallback {
            iteration: done.iterations,
            method: "FIRE",
        });
        memory.clear();
        // FIRE refuses a point without a finite energy (a trial structure that the problem
        // rejects, such as one with an atom of an ONIOM environment within 0.5 Angstrom of
        // another): the relaxation then ends where the line searches left it, short of its
        // thresholds, and says why, instead of failing the run.
        let after = match (methods.0)(problem, done.y.clone(), settings, super::FALLBACK_METHOD) {
            Ok(after) => after,
            Err(e) => {
                problem.progress(&super::MicroProgress::FallbackFailed {
                    iteration: done.iterations,
                    method: super::FALLBACK_METHOD,
                    reason: if e.to_string().contains("not finite") {
                        "a trial point had no finite energy (a structure the problem refuses)"
                    } else {
                        "the method failed"
                    },
                });
                let kept = MicroResult {
                    converged: false,
                    stalled: true,
                    fallbacks: fallbacks + 1,
                    ..done
                };
                ended(&kept);
                return Ok(kept);
            }
        };
        let both = MicroResult {
            iterations: done.iterations + after.iterations,
            fallbacks: fallbacks + 1,
            ..after
        };
        ended(&both);
        Ok(both)
    };
    let mut best = run(y0.to_vec(), memory)?;
    let mk = settings.mode_kill;
    if !mk.enabled || problem.dim() == 0 {
        return Ok(best);
    }
    while best.killed_modes < mk.max_kills {
        // The pairs of the relaxation precondition the search where they are in the
        // variables of the problem (Cartesian), and the mode of the last check starts it.
        let cartesian = memory.generation().is_none();
        let (lowest, mode, products, residual, found) = if mk.guided {
            lowest_mode(
                problem,
                &best.y,
                cartesian.then_some(&*memory),
                memory.mode.as_ref(),
                true,
            )?
        } else {
            lowest_mode(problem, &best.y, None, None, false)?
        };
        memory.mode = Some(Col::from_fn(mode.len(), |i| mode[i]));
        let negative = lowest < -mk.threshold;
        problem.progress(&super::MicroProgress::Curvature {
            lowest,
            negative,
            products,
            residual,
            converged: found,
        });
        if !negative {
            break;
        }
        let mut next: Option<(MicroResult, MicroMemory)> = None;
        for sign in [1.0, -1.0] {
            let y: Vec<f64> = best
                .y
                .iter()
                .enumerate()
                .map(|(i, yi)| yi + sign * mk.displacement * mode[i])
                .collect();
            let mut m = memory.clone();
            let r = run(y, &mut m)?;
            if next.as_ref().is_none_or(|n| r.energy < n.0.energy) {
                next = Some((r, m));
            }
        }
        let Some((mut n, m)) = next else { break };
        *memory = m;
        let iterations = best.iterations + n.iterations;
        let lower = n.energy < best.energy;
        problem.progress(&super::MicroProgress::Left { lower });
        if !lower {
            // Both displacements relax back: the curvature is too weak to follow.
            best.iterations = iterations;
            break;
        }
        n.iterations = iterations;
        n.killed_modes = best.killed_modes + 1;
        n.fallbacks += best.fallbacks;
        best = n;
    }
    Ok(best)
}

/// Displacement of the central differences of the gradient along a unit vector.
pub(super) const CURVATURE_STEP: f64 = 1e-4;

/// The lowest curvature at `y`, its mode (unit length), the Hessian-vector products it
/// took, the norm of the residual, and whether that met the tolerance. With an exact Hessian, from its eigenvalues; without one, by Davidson's method on
/// Hessian-vector products from central differences of the gradient along unit vectors (two
/// gradients a product), so that the check costs tens of gradients instead of two per
/// variable. The value found is a Rayleigh quotient, an upper bound of the lowest
/// curvature: a negative one is a negative curvature.
///
/// `pairs`: the limited-memory BFGS of the relaxation, in the variables of the problem; its
/// inverse Hessian corrects the residual (a preconditioned inverse iteration: the soft
/// modes, which the plain search finds last in an environment with stiff bonds, come
/// first). `start`: the mode of the last check (the environment changes little from one
/// relaxation to the next, and so does its softest mode). `model`: the preconditioner of the
/// problem ([`MicroProblem::precondition`]) is the initial matrix of those pairs (or the
/// correction itself without pairs), and its inverse iteration from a fixed pattern is the
/// start without a mode of a last check.
fn lowest_mode(
    problem: &dyn MicroProblem,
    y: &[f64],
    pairs: Option<&MicroMemory>,
    start: Option<&Col<f64>>,
    model: bool,
) -> MoptResult<(f64, Vec<f64>, usize, f64, bool)> {
    let n = y.len();
    if problem.has_hessian() {
        if let Some(h) = problem.evaluate(y, true)?.hessian {
            let (w, v) = symmetric_eigen(&h)?;
            return Ok((w[0], (0..n).map(|i| v[(i, 0)]).collect(), 1, 0.0, true));
        }
    }
    let mut apply = |v: &Col<f64>| -> MoptResult<Col<f64>> {
        let moved = |sign: f64| -> Vec<f64> {
            y.iter()
                .zip(v.iter())
                .map(|(a, b)| a + sign * CURVATURE_STEP * b)
                .collect()
        };
        let gp = problem.evaluate(&moved(1.0), false)?.gradient;
        let gm = problem.evaluate(&moved(-1.0), false)?.gradient;
        Ok((&gp - &gm) * faer::Scale(0.5 / CURVATURE_STEP))
    };
    // The mode of the last check, else a start with every component (a fixed pattern, so
    // that runs repeat).
    let h0 = |q: &Col<f64>| if model { problem.precondition(q) } else { None };
    let start = match start {
        Some(mode) if mode.nrows() == n && col_norm(mode) > 0.0 => mode.clone(),
        _ => {
            let mut v = Col::from_fn(n, |i| 1.0 + 0.5 * ((i * 7919 % 13) as f64 / 13.0 - 0.5));
            // Two steps of inverse iteration with the model: its softest modes.
            for _ in 0..2 {
                match h0(&v) {
                    Some(w) if col_norm(&w) > 0.0 => v = &w * (1.0 / col_norm(&w)),
                    _ => break,
                }
            }
            v
        }
    };
    let settings = DavidsonSettings {
        max_iterations: CURVATURE_PRODUCTS.min(n.max(1)),
        tolerance: 1e-6,
        max_subspace: 20,
    };
    let preconditioned = pairs.filter(|m| !m.is_empty());
    let mut correct = |residual: &Col<f64>, _theta: f64| -> Col<f64> {
        match preconditioned {
            Some(m) => m.apply_inverse_with(residual, &h0),
            None => h0(residual).unwrap_or_else(|| residual.clone()),
        }
    };
    let pair = lowest_eigenpair_preconditioned(&mut apply, &mut correct, &start, &settings)?;
    Ok((
        pair.value,
        pair.vector.iter().copied().collect(),
        pair.applications,
        pair.residual,
        pair.converged,
    ))
}

/// Hessian-vector products of the check for negative curvature at most.
const CURVATURE_PRODUCTS: usize = 40;

/// Newton step with the absolute values of the Hessian eigenvalues (a descent direction that
/// also moves downhill along negative-curvature modes).
fn newton_step(h: &Mat<f64>, g: &Col<f64>) -> MoptResult<Col<f64>> {
    let (w, v) = symmetric_eigen(h)?;
    let n = g.nrows();
    let wmax = w.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    let floor = (1e-8 * wmax).max(1e-8);
    let mut p = Col::zeros(n);
    for k in 0..n {
        let c = (0..n).map(|i| v[(i, k)] * g[i]).sum::<f64>() / w[k].abs().max(floor);
        for i in 0..n {
            p[i] -= c * v[(i, k)];
        }
    }
    Ok(p)
}

/// One relaxation by the minimizer of `s` (without mode kill).
fn descend(
    problem: &dyn MicroProblem,
    y: Vec<f64>,
    s: &MicroSettings,
    memory: &mut MicroMemory,
) -> MoptResult<MicroResult> {
    match &s.method {
        InnerMethod::Lbfgs => descend_lbfgs(problem, y, s, memory),
        InnerMethod::TruncatedNewton => super::newton_cg::descend_tn(problem, y, s, memory),
        InnerMethod::Method(name) => match s.methods {
            Some(methods) => (methods.0)(problem, y, s, name),
            None => Err(MoptError::config(format!(
                "microiteration: the method '{name}' of the optimizer registry is run by the \
                 optimizers, which the settings of this relaxation were not given (lbfgs and \
                 tn are the relaxation's own)"
            ))),
        },
    }
}

fn descend_lbfgs(
    problem: &dyn MicroProblem,
    y: Vec<f64>,
    s: &MicroSettings,
    memory: &mut MicroMemory,
) -> MoptResult<MicroResult> {
    descend_lbfgs_on(problem, y, s, memory, false)
}

/// The limited-memory BFGS; `gradient_only`: a trial step is taken when the change of the
/// energy that the directional derivatives at its two ends give (the trapezoid rule,
/// `alpha (g0.p + g1.p) / 2`) meets the sufficient decrease, without the energy itself (the
/// gradient-only line search of Wilke, Kok, Snyman and Groenwold, Struct. Multidisc. Optim.
/// 2013, 47, 1).
fn descend_lbfgs_on(
    problem: &dyn MicroProblem,
    mut y: Vec<f64>,
    s: &MicroSettings,
    memory: &mut MicroMemory,
    gradient_only: bool,
) -> MoptResult<MicroResult> {
    let n = y.len();
    let exact = problem.has_hessian();
    let mut cur: MicroEval = problem.evaluate(&y, exact)?;
    let mut iterations = 0;
    let mut converged = super::is_converged(&cur.gradient, s);
    let mut stall = super::Stall::new(cur.energy, col_max_abs(&cur.gradient));
    let mut stalled = false;
    let mut line_search_failed = false;
    while !converged && iterations < s.max_iterations {
        iterations += 1;
        let g = cur.gradient.clone();
        let mut p = match (&cur.hessian, exact) {
            (Some(h), true) => newton_step(h, &g)?,
            _ => memory.direction_with(&g, &|q| problem.precondition(q)),
        };
        if col_dot(&g, &p) >= 0.0 {
            p = -g.clone();
        }
        let norm = col_norm(&p);
        if norm > s.max_step {
            p *= faer::Scale(s.max_step / norm);
        }
        let slope = col_dot(&g, &p);
        let roundoff = 1e-14 * cur.energy.abs().max(1.0);
        let mut alpha = 1.0;
        let mut accepted: Option<(Vec<f64>, MicroEval)> = None;
        for _ in 0..40 {
            let yn: Vec<f64> = y.iter().zip(p.iter()).map(|(a, b)| a + alpha * b).collect();
            let en = problem.evaluate(&yn, exact)?;
            let lower = if gradient_only {
                let end_slope = col_dot(&en.gradient, &p);
                en.energy.is_finite()
                    && end_slope.is_finite()
                    && 0.5 * (slope + end_slope) <= 1e-4 * slope
            } else {
                en.energy.is_finite() && en.energy <= cur.energy + 1e-4 * alpha * slope + roundoff
            };
            if lower {
                accepted = Some((yn, en));
                break;
            }
            alpha *= 0.5;
        }
        // No step along the direction lowers the energy (or only one at the round-off):
        // the surface does not follow its gradient here, and the relaxation goes on with a
        // method without line searches (`relax_in`).
        let Some((yn, en)) = accepted.filter(|_| alpha >= super::SMALLEST_STEP) else {
            line_search_failed = true;
            break;
        };
        if !exact {
            let sv = Col::from_fn(n, |i| yn[i] - y[i]);
            let yv = &en.gradient - &g;
            memory.push(sv, yv, s.memory);
        }
        y = yn;
        cur = en;
        super::tell_step(problem, iterations, cur.energy, &cur.gradient);
        converged = super::is_converged(&cur.gradient, s);
        if !converged && stall.stalled_after(cur.energy, col_max_abs(&cur.gradient)) {
            stalled = true;
            break;
        }
    }
    Ok(MicroResult {
        max_gradient: col_max_abs(&cur.gradient),
        energy: cur.energy,
        y,
        iterations,
        killed_modes: 0,
        converged,
        stalled,
        line_search_failed,
        fallbacks: 0,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::{effective_hessian, ModeKill};
    use super::*;

    /// `E(y) = (y0^2 - 1)^2 + y1^2 + 0.1 y0 y1^2`: minima at `y0 = +-1`, a saddle at the origin.
    struct DoubleWell {
        hessian: bool,
    }

    impl MicroProblem for DoubleWell {
        fn dim(&self) -> usize {
            2
        }
        fn evaluate(&self, y: &[f64], need_hessian: bool) -> MoptResult<MicroEval> {
            let (a, b) = (y[0], y[1]);
            let e = (a * a - 1.0).powi(2) + b * b + 0.1 * a * b * b;
            let g = Col::from_fn(2, |i| {
                if i == 0 {
                    4.0 * a * (a * a - 1.0) + 0.1 * b * b
                } else {
                    2.0 * b + 0.2 * a * b
                }
            });
            let h = (need_hessian && self.hessian).then(|| {
                Mat::from_fn(2, 2, |i, j| match (i, j) {
                    (0, 0) => 12.0 * a * a - 4.0,
                    (1, 1) => 2.0 + 0.2 * a,
                    _ => 0.2 * b,
                })
            });
            Ok(MicroEval {
                energy: e,
                gradient: g,
                hessian: h,
            })
        }
        fn has_hessian(&self) -> bool {
            self.hessian
        }
    }

    #[test]
    fn mode_kill_leaves_the_saddle_point() {
        for hessian in [true, false] {
            let p = DoubleWell { hessian };
            // Started exactly at the saddle point (zero gradient).
            let r = relax(&p, &[0.0, 0.0], &MicroSettings::default()).unwrap();
            assert!(r.converged, "{r:?}");
            assert_eq!(r.killed_modes, 1);
            assert!((r.y[0].abs() - 1.0).abs() < 1e-6, "{r:?}");
            // The other minimizer of the relaxation ends at the same minimum (a method of the
            // optimizer registry: `opt::micro`).
            let method = InnerMethod::TruncatedNewton;
            let s = MicroSettings {
                method: method.clone(),
                gradient_tolerance: 1e-8,
                ..MicroSettings::default()
            };
            let r = relax(&p, &[0.0, 0.0], &s).unwrap();
            assert!(r.converged, "{method:?}: {r:?}");
            assert!((r.y[0].abs() - 1.0).abs() < 1e-6, "{method:?}: {r:?}");
            let off = MicroSettings {
                mode_kill: ModeKill {
                    enabled: false,
                    ..ModeKill::default()
                },
                ..MicroSettings::default()
            };
            let stuck = relax(&p, &[0.0, 0.0], &off).unwrap();
            assert!(stuck.y[0].abs() < 1e-12);
        }
    }

    /// Stiff and soft modes in 24 variables around `c` (all components `shift`):
    /// `E = sum_k l_k (u_k . (y - c))^2 / 2 + 0.01 sum_i (y_i - c_i)^4`, with `u` the rows of
    /// a reflection, so that each mode has every variable. The products of every check of
    /// the curvature are kept.
    struct StiffAndSoft {
        shift: f64,
        products: std::cell::RefCell<Vec<(f64, usize)>>,
    }

    impl StiffAndSoft {
        const N: usize = 24;

        fn curvature(k: usize) -> f64 {
            if k.is_multiple_of(5) {
                2e-3 * (1.0 + k as f64)
            } else {
                0.8 + 0.02 * k as f64
            }
        }

        /// `u y` with `u = 1 - 2 w w^T / w^T w`.
        fn reflect(y: &[f64]) -> Vec<f64> {
            let w: Vec<f64> = (0..Self::N).map(|i| 1.0 + 0.3 * i as f64).collect();
            let ww: f64 = w.iter().map(|v| v * v).sum();
            let wy: f64 = w.iter().zip(y).map(|(a, b)| a * b).sum();
            y.iter()
                .zip(&w)
                .map(|(yi, wi)| yi - 2.0 * wi * wy / ww)
                .collect()
        }
    }

    impl MicroProblem for StiffAndSoft {
        fn dim(&self) -> usize {
            Self::N
        }
        fn evaluate(&self, y: &[f64], _need_hessian: bool) -> MoptResult<MicroEval> {
            let d: Vec<f64> = y.iter().map(|v| v - self.shift).collect();
            let q = Self::reflect(&d);
            let scaled: Vec<f64> = (0..Self::N).map(|k| Self::curvature(k) * q[k]).collect();
            let back = Self::reflect(&scaled);
            let energy = 0.5 * (0..Self::N).map(|k| scaled[k] * q[k]).sum::<f64>()
                + 0.01 * d.iter().map(|v| v.powi(4)).sum::<f64>();
            Ok(MicroEval {
                energy,
                gradient: Col::from_fn(Self::N, |i| back[i] + 0.04 * d[i].powi(3)),
                hessian: None,
            })
        }
        fn has_hessian(&self) -> bool {
            false
        }
        fn progress(&self, event: &super::super::MicroProgress) {
            if let super::super::MicroProgress::Curvature {
                lowest, products, ..
            } = event
            {
                self.products.borrow_mut().push((*lowest, *products));
            }
        }
    }

    /// A quadratic in 12 variables whose energy has a noise of 1e-8 Hartree from point to point
    /// (as a lowest level in single precision) and whose gradient is exact. A line search on
    /// the energy keeps the points whose noise happens to be low, until no step lowers it.
    struct Noisy;

    impl MicroProblem for Noisy {
        fn dim(&self) -> usize {
            12
        }
        fn evaluate(&self, y: &[f64], _need_hessian: bool) -> MoptResult<MicroEval> {
            let k = |i: usize| 0.05 + 0.3 * i as f64;
            let energy = 0.5 * (0..12).map(|i| k(i) * y[i] * y[i]).sum::<f64>();
            let bits = y.iter().fold(0x9e37_79b9_7f4a_7c15_u64, |h, v| {
                (h ^ v.to_bits()).wrapping_mul(0x100_0000_01b3)
            });
            let noise = 1e-8 * ((bits >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
            Ok(MicroEval {
                energy: energy + noise,
                gradient: Col::from_fn(12, |i| k(i) * y[i]),
                hessian: None,
            })
        }
        fn has_hessian(&self) -> bool {
            false
        }
    }

    #[test]
    fn a_noisy_energy_is_relaxed_with_the_gradient_alone() {
        let s = MicroSettings {
            gradient_tolerance: 1e-9,
            mode_kill: ModeKill {
                enabled: false,
                ..ModeKill::default()
            },
            ..MicroSettings::default()
        };
        let r = relax(&Noisy, &[1.0; 12], &s).unwrap();
        assert!(r.converged, "{r:?}");
        assert!(r.fallbacks >= 1, "{r:?}");
        assert!(r.max_gradient <= 1e-9 && r.iterations < 300, "{r:?}");
        println!("iterations {}, fallbacks {}", r.iterations, r.fallbacks);
    }

    #[test]
    fn the_guided_check_of_the_curvature_takes_fewer_products() {
        // Relaxations one after another, as along an outer optimization: the environment
        // moves a little between them.
        let mut totals = Vec::new();
        for guided in [false, true] {
            let mut p = StiffAndSoft {
                shift: 0.0,
                products: std::cell::RefCell::new(Vec::new()),
            };
            let settings = MicroSettings {
                gradient_tolerance: 1e-8,
                mode_kill: ModeKill {
                    guided,
                    ..ModeKill::default()
                },
                ..MicroSettings::default()
            };
            let mut memory = MicroMemory::default();
            let mut y = vec![0.3; StiffAndSoft::N];
            for step in 0..4 {
                p.shift = 0.05 * step as f64;
                let r = relax_with(&p, &y, &settings, &mut memory).unwrap();
                assert!(r.converged && r.killed_modes == 0, "{r:?}");
                y = r.y;
            }
            let checks = p.products.borrow().clone();
            assert_eq!(checks.len(), 4);
            for (lowest, _) in &checks {
                // The lowest curvature at the minimum (the quartic term adds nothing there).
                assert!((lowest - 2e-3).abs() < 1e-6, "{guided}: {lowest}");
            }
            totals.push(checks.iter().map(|c| c.1).sum::<usize>());
        }
        assert!(2 * totals[1] < totals[0], "{totals:?}");
    }

    #[test]
    fn effective_hessian_is_the_schur_complement() {
        // E(x, y) = x^2 + 2 x y + 3 y^2: min_y gives E* = x^2 - x^2 / 3.
        let h = Mat::from_fn(2, 2, |i, j| [[2.0, 2.0], [2.0, 6.0]][i][j]);
        let e = effective_hessian(&h, 1).unwrap();
        assert!((e[(0, 0)] - 4.0 / 3.0).abs() < 1e-14);
        // A flat environment variable is left out.
        let h = Mat::from_fn(2, 2, |i, j| [[2.0, 0.0], [0.0, 0.0]][i][j]);
        assert!((effective_hessian(&h, 1).unwrap()[(0, 0)] - 2.0).abs() < 1e-14);
    }
}
