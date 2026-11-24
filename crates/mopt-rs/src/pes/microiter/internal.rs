//! The limited-memory BFGS of a relaxation in other coordinates than the Cartesian ones.
//!
//! The steps, the gradient, and the pairs are in the coordinates `q` of an
//! [`InnerCoordinates`]; each trial point is the iterative back-transformation of the step
//! from the current point, and the pair of an accepted step is the change of `q` that was
//! reached (not the one asked for). The convergence test is on the gradient in the variables
//! of the problem, as in the Cartesian relaxation.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::memory::MicroMemory;
use super::{InnerCoordinates, MicroProblem, MicroResult, MicroSettings};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_max_abs, col_norm};

/// One relaxation in `coords` from `y` (no mode kill).
pub(super) fn descend_in(
    problem: &dyn MicroProblem,
    mut y: Vec<f64>,
    s: &MicroSettings,
    memory: &mut MicroMemory,
    coords: &mut dyn InnerCoordinates,
) -> MoptResult<MicroResult> {
    let n = y.len();
    if coords.renew_if_needed(&y)? {
        memory.fit(coords.dim(), Some(coords.generation()));
    }
    coords.prepare(&y)?;
    let mut cur = problem.evaluate(&y, false)?;
    let mut gq = coords.gradient_to_q(&cur.gradient);
    let mut iterations = 0;
    let mut converged = n == 0 || super::is_converged(&cur.gradient, s);
    let mut stall = super::Stall::new(cur.energy, col_max_abs(&cur.gradient));
    let mut stalled = false;
    let mut line_search_failed = false;
    while !converged && iterations < s.max_iterations {
        iterations += 1;
        let mut p = memory.direction(&gq);
        if col_dot(&gq, &p) >= 0.0 {
            p = -gq.clone();
        }
        let norm = col_norm(&p);
        if norm > s.max_step {
            p *= faer::Scale(s.max_step / norm);
        }
        let slope = col_dot(&gq, &p);
        let roundoff = 1e-14 * cur.energy.abs().max(1.0);
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let dq = Col::from_fn(p.nrows(), |i| alpha * p[i]);
            // A step the coordinates cannot take back counts as one that does not descend.
            if let Ok(yn) = coords.step(&y, &dq) {
                let en = problem.evaluate(&yn, false)?;
                if en.energy.is_finite()
                    && en.energy <= cur.energy + 1e-4 * alpha * slope + roundoff
                {
                    accepted = Some((yn, en));
                    break;
                }
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
        let reached = coords.displacement(&yn, &y);
        if coords.renew_if_needed(&yn)? {
            // New coordinates: the pairs of the old ones do not apply.
            memory.fit(coords.dim(), Some(coords.generation()));
            coords.prepare(&yn)?;
            gq = coords.gradient_to_q(&en.gradient);
        } else {
            coords.prepare(&yn)?;
            let gq_new = coords.gradient_to_q(&en.gradient);
            if let Ok(sq) = reached {
                memory.push(sq, &gq_new - &gq, s.memory);
            }
            gq = gq_new;
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
