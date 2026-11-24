//! Driver of the direct MaxFlux method: evaluates the points it asks for on the surface of
//! a chain and hands every iteration to the caller.
//!
//! The surfaces are those of the chain relaxations ([`super::chain::ChainSurface`]): a
//! calculator with bias potentials, two electronic states combined (`-mf`), an objective
//! of plain variables. A point is evaluated as the image of its number.
//!
//! MultiOptPy: `neb.py:NEB.execute` with `-dmf` (the loop over the images).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::chain::{ChainSurface, Flow};
use crate::error::{MoptResult, ResultExt};
use crate::path::dmf::{DirectMaxFlux, DmfAction, DmfEnd, DmfReport};

/// How a run ended.
#[derive(Debug, Clone)]
pub struct DmfOutcome {
    /// Iterations done.
    pub iterations: usize,
    /// How the method ended (`None`: the caller stopped it).
    pub end: Option<DmfEnd>,
    /// What the run leaves: the last iteration when it converged or was stopped, else
    /// the one with the lowest top of the path.
    pub last: Option<DmfReport>,
}

/// Runs `machine` on `surface` to its end. After every iteration `on_iteration` gets the
/// machine (its report is that of the iteration, its path the one after the step) and
/// what the calculator said meanwhile (warnings, `<S**2>`, the output of the program);
/// [`Flow::Stop`] ends the run there.
pub fn run_dmf(
    machine: &mut DirectMaxFlux,
    surface: &dyn ChainSurface,
    on_iteration: &mut dyn FnMut(&DirectMaxFlux, &[String]) -> MoptResult<Flow>,
) -> MoptResult<DmfOutcome> {
    let mut iteration = 0;
    loop {
        match machine.ask()? {
            DmfAction::Finished(end) => {
                return Ok(DmfOutcome {
                    iterations: iteration,
                    end: Some(end),
                    last: machine.result().cloned(),
                });
            }
            DmfAction::Evaluate(points) => {
                surface.begin_iteration(iteration);
                let mut values = Vec::with_capacity(points.len());
                let numbered: Vec<(usize, &faer::Col<f64>)> =
                    points.iter().map(|point| (point.index, &point.x)).collect();
                let all = surface.evaluate_all(&numbered, false);
                for (point, found) in points.iter().zip(all) {
                    let found = found.context(format!(
                        "point {} of the path (t = {:.4}), iteration {iteration}",
                        point.index, point.t
                    ))?;
                    values.push((found.energy, found.gradient));
                }
                machine.tell(values)?;
                iteration += 1;
                let notes = surface.take_notes();
                if on_iteration(machine, &notes)? == Flow::Stop {
                    return Ok(DmfOutcome {
                        iterations: iteration,
                        end: None,
                        last: machine.report().cloned(),
                    });
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::path::dmf::DmfSettings;
    use crate::pes::native::muller_brown::MB_MINIMA;
    use crate::pes::objective::MullerBrown2D;
    use crate::workflow::chain::ObjectiveSurface;
    use faer::Col;

    #[test]
    fn every_iteration_is_seen_and_a_stop_ends_the_run() {
        let objective = MullerBrown2D::default();
        let surface = ObjectiveSurface(&objective);
        let chain: Vec<Col<f64>> = [MB_MINIMA[0], MB_MINIMA[2]]
            .iter()
            .map(|p| Col::from_fn(2, |k| p[k]))
            .collect();
        let mut machine = DirectMaxFlux::new(DmfSettings::default(), &chain).unwrap();
        let mut seen = 0;
        let outcome = run_dmf(&mut machine, &surface, &mut |machine, notes| {
            let report = machine.report().unwrap();
            assert_eq!(report.iteration, seen);
            assert!(notes.is_empty());
            seen += 1;
            Ok(Flow::Continue)
        })
        .unwrap();
        assert!(matches!(
            outcome.end,
            Some(DmfEnd::Converged | DmfEnd::Settled)
        ));
        assert_eq!(outcome.iterations, seen);
        let mut machine = DirectMaxFlux::new(DmfSettings::default(), &chain).unwrap();
        let stopped = run_dmf(&mut machine, &surface, &mut |machine, _| {
            Ok(if machine.report().unwrap().iteration == 2 {
                Flow::Stop
            } else {
                Flow::Continue
            })
        })
        .unwrap();
        assert_eq!(stopped.end, None);
        assert_eq!(stopped.iterations, 3);
    }
}
