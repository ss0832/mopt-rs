//! A surface of a chain that answers some iterations from a Gaussian-process model of the
//! values it calculated.
//!
//! [`SurrogateSurface`] stands between a relaxation and any other surface (a calculator
//! with its bias potentials and constraints, a surface of electronic states, an objective
//! of plain variables), so every kind of force, step, redistribution, and added image of
//! [`crate::path::band`] works with it. What it does:
//!
//! - **Calculated iterations.** All images are calculated and learned
//!   ([`crate::path::band::surrogate`]). Before a value is learned, the model predicts it:
//!   the largest difference of a predicted and a calculated gradient component is the
//!   error of the model, measured anew in every such iteration.
//! - **Model iterations.** Between two calculated iterations the chain takes up to
//!   `between` iterations with the values of the model, as long as the measured error is
//!   below `error_share` of the largest force of the chain: steps that cost nothing while
//!   the forces are large against what the model gets wrong. An image the model is not
//!   sure of (far from every calculated point) is calculated in such an iteration, too.
//! - **Near the end** the error of the model is no longer small against the forces, and
//!   every iteration is calculated: the model cannot bring a chain to rest (the predicted
//!   gradients are off by more than the tolerances; see the measurement in
//!   `path::band::surrogate`).
//!
//! Three rules keep the result that of the surface itself: the first chain is calculated;
//! a chain that looks relaxed with predicted values is calculated before it counts as
//! relaxed ([`crate::workflow::chain::relax_chain`] asks for it); the chain of the last
//! iteration is calculated. Hessians are not predicted: an image whose Hessian is asked
//! for is calculated.
//!
//! MultiOptPy: `MEP/pathopt_gpneb_force.py` (unfinished there).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::Col;

use super::chain::ChainSurface;
use crate::error::{MoptError, MoptResult};
use crate::path::band::machine::ImageValue;
use crate::path::band::surrogate::{Surrogate, SurrogateSettings};

/// How often the surface calculated and how often it predicted.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SurrogateCounts {
    /// Images that were calculated.
    pub calculated: usize,
    /// Images that were answered from the model.
    pub predicted: usize,
    /// Iterations in which the model answered.
    pub model_iterations: usize,
    /// Chains that looked relaxed with predicted values and were calculated to check.
    pub checks: usize,
    /// The error of the model's gradients measured last (`None` before the first
    /// comparison).
    pub error: Option<f64>,
}

struct State {
    model: Surrogate,
    /// The images of the iteration at hand, as far as they were asked for.
    chain: Vec<Col<f64>>,
    /// Everything is calculated in this iteration.
    all: bool,
    /// Everything is calculated in the next iteration.
    all_next: bool,
    /// Whether a value of this iteration was predicted.
    predicted_now: bool,
    /// Model iterations since the last calculated one.
    since_calculated: usize,
    /// The error of the model measured in this iteration (largest difference of a
    /// predicted and a calculated gradient component), if anything was compared.
    error_now: Option<f64>,
    /// The largest force of the chain at the last calculated iteration.
    force: Option<f64>,
    counts: SurrogateCounts,
    /// The calculated and predicted images that the notes have told of.
    noted: (usize, usize),
}

/// `inner` with a model that answers some iterations.
pub struct SurrogateSurface<'a> {
    inner: &'a dyn ChainSurface,
    state: Mutex<State>,
}

impl<'a> SurrogateSurface<'a> {
    /// The surface `inner` behind a model with `settings`.
    pub fn new(inner: &'a dyn ChainSurface, settings: SurrogateSettings) -> Self {
        Self {
            inner,
            state: Mutex::new(State {
                model: Surrogate::new(settings),
                chain: Vec::new(),
                all: true,
                all_next: false,
                predicted_now: false,
                since_calculated: 0,
                error_now: None,
                force: None,
                counts: SurrogateCounts::default(),
                noted: (0, 0),
            }),
        }
    }

    fn state(&self) -> MoptResult<std::sync::MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| MoptError::calc("the model of the surface is not available"))
    }

    /// How often it calculated and predicted so far.
    pub fn counts(&self) -> SurrogateCounts {
        self.state().map(|s| s.counts).unwrap_or_default()
    }

    /// The number of points the model keeps.
    pub fn points(&self) -> usize {
        self.state().map(|s| s.model.len()).unwrap_or(0)
    }
}

impl ChainSurface for SurrogateSurface<'_> {
    fn description(&self) -> String {
        let settings = self
            .state()
            .map(|s| *s.model.settings())
            .unwrap_or_default();
        format!(
            "{}; up to {} iteration(s) between calculated ones are answered from a Gaussian \
             process of the {} nearest calculated points ({}, length scale {:.3}) while \
             its measured error is below {} of the largest force",
            self.inner.description(),
            settings.between,
            settings.neighbours,
            match settings.features {
                crate::path::band::surrogate::Features::Points => "seen as they are",
                crate::path::band::surrogate::Features::InverseDistances =>
                    "seen by their inverse interatomic distances",
            },
            settings.length_scale,
            settings.error_share
        )
    }

    fn evaluate(
        &self,
        point: &Col<f64>,
        index: usize,
        need_hessian: bool,
    ) -> MoptResult<ImageValue> {
        let expected = {
            let mut state = self.state()?;
            if state.chain.len() <= index {
                state.chain.resize(index + 1, point.clone());
            }
            state.chain[index] = point.clone();
            let found = state.model.predict(point)?;
            if let (false, false, Some(found)) = (need_hessian, state.all, &found) {
                if state.model.believes(found) {
                    state.predicted_now = true;
                    state.counts.predicted += 1;
                    return Ok(ImageValue::new(found.value, found.gradient.clone()));
                }
            }
            found
        };
        // Calculated without the lock: the surface behind may take long.
        let value = self.inner.evaluate(point, index, need_hessian)?;
        let mut state = self.state()?;
        if let Some(found) = expected {
            // What the model would have said, against what is so.
            let off = found
                .gradient
                .iter()
                .zip(value.gradient.iter())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            state.error_now = Some(state.error_now.map_or(off, |e| e.max(off)));
        }
        state.model.record(point, value.energy, &value.gradient)?;
        state.counts.calculated += 1;
        Ok(value)
    }

    fn begin_iteration(&self, iteration: usize) {
        self.inner.begin_iteration(iteration);
        if let Ok(mut state) = self.state() {
            let chain = std::mem::take(&mut state.chain);
            state.model.keep_near(&chain);
            if state.predicted_now {
                state.counts.model_iterations += 1;
            }
            if let Some(error) = state.error_now.take() {
                state.counts.error = Some(error);
            }
            let settings = *state.model.settings();
            // The model answers while it is good against the forces, a few iterations
            // in a row.
            let good = match (state.counts.error, state.force) {
                (Some(error), Some(force)) => error < settings.error_share * force,
                _ => false,
            };
            state.all = iteration == 0
                || state.all_next
                || !good
                || state.since_calculated >= settings.between;
            state.since_calculated = if state.all {
                0
            } else {
                state.since_calculated + 1
            };
            state.all_next = false;
            state.predicted_now = false;
        }
    }

    fn images_dropped(&self, dropped: &[usize]) {
        self.inner.images_dropped(dropped);
    }

    fn images_added(&self) {
        self.inner.images_added();
    }

    fn values_are_calculated(&self) -> bool {
        self.state().map(|s| !s.predicted_now).unwrap_or(true)
    }

    fn calculate_next(&self, check: bool) {
        if let Ok(mut state) = self.state() {
            state.all_next = true;
            if check {
                state.counts.checks += 1;
            }
        }
    }

    fn told(&self, largest_force: f64) {
        if let Ok(mut state) = self.state() {
            if !state.predicted_now {
                state.force = Some(largest_force);
            }
        }
    }

    fn take_notes(&self) -> Vec<String> {
        let mut notes = self.inner.take_notes();
        if let Ok(mut state) = self.state() {
            let (calculated, predicted) = (
                state.counts.calculated - state.noted.0,
                state.counts.predicted - state.noted.1,
            );
            state.noted = (state.counts.calculated, state.counts.predicted);
            if predicted > 0 {
                notes.push(format!(
                    "Gaussian process: {predicted} image(s) answered from the model, \
                     {calculated} calculated."
                ));
            }
        }
        notes
    }

    /// Those of the images that were calculated (a prediction has none).
    fn take_bare_energies(&self) -> Vec<(usize, f64)> {
        self.inner.take_bare_energies()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::path::band::machine::{ChainEnd, ChainRelaxation, ChainSettings};
    use crate::path::band::trust::FixedEnds;
    use crate::path::band::ChainKind;
    use crate::pes::objective::{MullerBrown2D, Objective};
    use crate::workflow::chain::relax_chain;

    /// The Muller-Brown surface, counting its calculations.
    struct Counted {
        surface: MullerBrown2D,
        calls: AtomicUsize,
    }

    impl ChainSurface for Counted {
        fn description(&self) -> String {
            "Muller-Brown".to_string()
        }

        fn evaluate(&self, point: &Col<f64>, _: usize, hessian: bool) -> MoptResult<ImageValue> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let found = self.surface.evaluate(point, hessian)?;
            Ok(ImageValue {
                energy: found.value,
                gradient: found.gradient,
                hessian: found.hessian,
            })
        }
    }

    fn straight(from: [f64; 2], to: [f64; 2], images: usize) -> Vec<Col<f64>> {
        (0..images)
            .map(|k| {
                let t = k as f64 / (images - 1) as f64;
                Col::from_fn(2, |c| from[c] + t * (to[c] - from[c]))
            })
            .collect()
    }

    #[test]
    fn the_same_chain_with_fewer_calculations() {
        let settings = ChainSettings {
            kind: ChainKind::Variables,
            max_iterations: 400,
            ends: FixedEnds::from_code(3),
            force_tolerance: Some(1e-3),
            ..ChainSettings::default()
        };
        let start = straight([-0.558, 1.442], [-0.050, 0.467], 9);
        let run = |model: Option<SurrogateSettings>| {
            let counted = Counted {
                surface: MullerBrown2D::default(),
                calls: AtomicUsize::new(0),
            };
            let mut machine = ChainRelaxation::new(settings.clone(), start.clone()).unwrap();
            let (outcome, counts) = match model {
                None => (relax_chain(&mut machine, &counted, &mut []).unwrap(), None),
                Some(model) => {
                    let surface = SurrogateSurface::new(&counted, model);
                    let outcome = relax_chain(&mut machine, &surface, &mut []).unwrap();
                    assert!(surface.description().contains("Gaussian process"));
                    (outcome, Some(surface.counts()))
                }
            };
            assert_eq!(outcome.end, Some(ChainEnd::Converged));
            let last = outcome.last.unwrap();
            let top = last
                .energies
                .iter()
                .fold(f64::NEG_INFINITY, |m, e| m.max(*e));
            (
                counted.calls.load(Ordering::SeqCst),
                outcome.iterations,
                top,
                last.largest_force,
                counts,
            )
        };
        let plain = run(None);
        // The surface is in units in which the length scale of the default is too long.
        let model = SurrogateSettings {
            length_scale: 0.25,
            error_share: 0.3,
            ..SurrogateSettings::default()
        };
        let modelled = run(Some(model));
        let counts = modelled.4.unwrap();
        println!(
            "calculations {} against {}, iterations {} against {}, checks {}",
            modelled.0, plain.0, modelled.1, plain.1, counts.checks
        );
        // What it calculated is what the surface behind was asked for.
        assert_eq!(counts.calculated, modelled.0);
        assert!(counts.predicted > 0 && counts.model_iterations > 0);
        assert!(counts.error.is_some());
        // The last chain was calculated and is relaxed: the same top of the profile, with
        // fewer calculations.
        assert!(modelled.3 < 1e-3);
        assert!(
            (modelled.2 - plain.2).abs() < 2e-3,
            "{} {}",
            modelled.2,
            plain.2
        );
        assert!(modelled.0 < plain.0, "{} {}", modelled.0, plain.0);
    }
}
