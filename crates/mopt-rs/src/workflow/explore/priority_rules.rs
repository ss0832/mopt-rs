//! More rules for the choice of the next minimum (see [`super::priority`]):
//!
//! - [`HeatedBoltzmann`]: MultiOptPy's selection in `confsearch` as it is, Boltzmann weights
//!   (with the visit penalty `exp(-alpha N_i)` of `-tabu`) whose temperature is ten times
//!   higher after every fifth trial (`i % 5 == 0`); registered as `boltzmann_heated` and
//!   `tabu_heated`;
//! - [`AdaptiveWaterFilling`]: the rising water of [`super::WaterFilling`] whose level rises
//!   to the energy of the next minimum above it (or of the `per_rise`-th next one), at least by
//!   `min_step` and at most by `max_step`. A fixed step submerges every minimum within the step
//!   at once, and each of them must then be sampled before the level rises again; where minima
//!   crowd at high energy the number of trials per rise grows with their density. With the
//!   adaptive rise every rise adds about `per_rise` minima; registered as
//!   `water_filling_adaptive`.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py` (`switch_conformer`,
//!  `switch_conformer_with_tabu`, the factor 10 at `i % 5 == 0`); the adaptive rise is new.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use super::priority::{boltzmann_log_weights, EqRecord, PriorityRegistry, SamplingPriority};
use crate::core_infra::rng::Rng;
use crate::domain::{Hartree, Quantity, Units};
use crate::error::{MoptError, MoptResult};

/// Boltzmann selection with a visit penalty and periodic heating.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatedBoltzmann {
    /// Temperature (K).
    pub temperature: f64,
    /// Visit penalty `alpha` (0: none).
    pub alpha: f64,
    /// Every this many trials the temperature is raised (0: never).
    pub every: usize,
    /// Factor of the raised temperature.
    pub factor: f64,
    trial: usize,
}

impl HeatedBoltzmann {
    /// MultiOptPy's selection at `temperature` with the visit penalty `alpha` (`None`: plain
    /// Boltzmann): ten times the temperature after trials `0, 5, 10, ...`.
    pub fn multioptpy(temperature: f64, alpha: Option<f64>) -> Self {
        Self {
            temperature,
            alpha: alpha.unwrap_or(0.0),
            every: 5,
            factor: 10.0,
            trial: 0,
        }
    }

    /// The temperature of the next selection.
    pub fn current_temperature(&self) -> f64 {
        if self.every > 0 && self.trial.is_multiple_of(self.every) {
            self.temperature * self.factor
        } else {
            self.temperature
        }
    }

    /// Probabilities of the next selection (for MultiOptPy's `tabu_selection_log.log`).
    pub fn probabilities(&self, pool: &[EqRecord]) -> Vec<f64> {
        let w = self.log_weights(pool);
        let top = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = w.iter().map(|l| (l - top).exp()).collect();
        let z: f64 = e.iter().sum();
        e.into_iter().map(|x| x / z).collect()
    }

    fn log_weights(&self, pool: &[EqRecord]) -> Vec<f64> {
        boltzmann_log_weights(pool, self.current_temperature())
            .into_iter()
            .zip(pool)
            .map(|(b, r)| b - self.alpha * r.visits as f64)
            .collect()
    }
}

impl SamplingPriority for HeatedBoltzmann {
    fn name(&self) -> &str {
        if self.alpha > 0.0 {
            "tabu_heated"
        } else {
            "boltzmann_heated"
        }
    }

    fn select(&mut self, pool: &[EqRecord], rng: &mut Rng) -> Option<usize> {
        self.select_with_weights(pool, rng).0
    }

    fn select_with_weights(
        &mut self,
        pool: &[EqRecord],
        rng: &mut Rng,
    ) -> (Option<usize>, Vec<f64>) {
        let weights = super::priority::relative_weights(&self.log_weights(pool));
        self.trial += 1;
        (super::priority::draw(&weights, rng), weights)
    }

    fn on_trial(&mut self, trial: usize) {
        self.trial = trial;
    }
}

/// Rising water with a rise adapted to the spacing of the minima.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveWaterFilling {
    /// Smallest rise.
    pub min_step: Quantity<Hartree>,
    /// Largest rise.
    pub max_step: Quantity<Hartree>,
    /// Minima above the level that one rise reaches (the level goes to the energy of the
    /// `per_rise`-th next one, within the bounds of the rise).
    pub per_rise: usize,
    /// Trials per minimum before the level rises.
    pub visits_per_level: usize,
    /// Level above the lowest minimum at which the search ends; unbounded when `None`.
    pub max_level: Option<Quantity<Hartree>>,
    level: f64,
}

impl AdaptiveWaterFilling {
    /// A rule rising by `min_step` to `max_step`, reaching `per_rise` further minima per rise.
    pub fn new(
        min_step: Quantity<Hartree>,
        max_step: Quantity<Hartree>,
        per_rise: usize,
        visits_per_level: usize,
    ) -> MoptResult<Self> {
        let (lo, hi) = (min_step.value(), max_step.value());
        if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && hi >= lo)
            || per_rise == 0
            || visits_per_level == 0
        {
            return Err(MoptError::config(
                "adaptive water filling needs 0 < min_step <= max_step, at least one minimum \
                 per rise, and at least one visit per level",
            ));
        }
        Ok(Self {
            min_step,
            max_step,
            per_rise,
            visits_per_level,
            max_level: None,
            level: 0.0,
        })
    }

    /// Stop once the level exceeds `max_level` above the lowest minimum.
    #[must_use]
    pub fn with_max_level(mut self, max_level: Quantity<Hartree>) -> Self {
        self.max_level = Some(max_level);
        self
    }

    /// Current level above the lowest minimum.
    pub fn level(&self) -> Quantity<Hartree> {
        Quantity::new(self.level)
    }
}

impl SamplingPriority for AdaptiveWaterFilling {
    fn name(&self) -> &str {
        "water_filling_adaptive"
    }

    fn select(&mut self, pool: &[EqRecord], _rng: &mut Rng) -> Option<usize> {
        let e_min = pool
            .iter()
            .map(|r| r.energy.value())
            .fold(f64::INFINITY, f64::min);
        loop {
            if self.max_level.is_some_and(|m| self.level > m.value()) {
                return None;
            }
            let pick = pool
                .iter()
                .enumerate()
                .filter(|(_, r)| r.energy.value() - e_min <= self.level + 1e-15)
                .filter(|(_, r)| r.visits < self.visits_per_level)
                .min_by(|(_, a), (_, b)| {
                    a.visits
                        .cmp(&b.visits)
                        .then(a.energy.value().total_cmp(&b.energy.value()))
                })
                .map(|(i, _)| i);
            if pick.is_some() {
                return pick;
            }
            let mut above: Vec<f64> = pool
                .iter()
                .map(|r| r.energy.value() - e_min)
                .filter(|d| *d > self.level + 1e-15)
                .collect();
            if above.is_empty() {
                return None;
            }
            above.sort_by(f64::total_cmp);
            let target = above[(self.per_rise - 1).min(above.len() - 1)];
            let rise = (target - self.level).clamp(self.min_step.value(), self.max_step.value());
            self.level += rise;
        }
    }
}

/// Register `boltzmann_heated`, `tabu_heated`, and `water_filling_adaptive` with MultiOptPy's
/// defaults (298.15 K, alpha = 0.5) and rises of 0.1 to 3 kcal/mol by one minimum, three
/// trials per minimum.
pub(crate) fn register_more(r: &mut PriorityRegistry) {
    r.register(
        "boltzmann_heated",
        Arc::new(|| {
            Box::new(HeatedBoltzmann::multioptpy(298.15, None)) as Box<dyn SamplingPriority>
        }),
    );
    r.register(
        "tabu_heated",
        Arc::new(|| {
            Box::new(HeatedBoltzmann::multioptpy(298.15, Some(0.5))) as Box<dyn SamplingPriority>
        }),
    );
    r.register(
        "water_filling_adaptive",
        Arc::new(|| {
            Box::new(AdaptiveWaterFilling {
                min_step: Quantity::new(0.1 * Units::KCAL_MOL_TO_HARTREE),
                max_step: Quantity::new(3.0 * Units::KCAL_MOL_TO_HARTREE),
                per_rise: 1,
                visits_per_level: 3,
                max_level: None,
                level: 0.0,
            }) as Box<dyn SamplingPriority>
        }),
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::workflow::explore::PenalizedBoltzmann;

    fn pool(kcal: &[f64], visits: &[usize]) -> Vec<EqRecord> {
        kcal.iter()
            .zip(visits)
            .map(|(e, v)| EqRecord {
                energy: Quantity::new(e * Units::KCAL_MOL_TO_HARTREE),
                visits: *v,
            })
            .collect()
    }

    #[test]
    fn multioptpy_heats_every_fifth_trial() {
        let mut rule = HeatedBoltzmann::multioptpy(298.15, Some(0.5));
        let p = pool(&[0.0, 3.0], &[0, 0]);
        rule.on_trial(0);
        let hot = rule.probabilities(&p);
        rule.on_trial(1);
        let cold = rule.probabilities(&p);
        // exp(-3 / (kT)) at 2981.5 K and 298.15 K.
        let kt: f64 = Units::BOLTZMANN_KCAL_MOL_K * 298.15;
        assert!((hot[1] / hot[0] - (-3.0 / (10.0 * kt)).exp()).abs() < 1e-12);
        assert!((cold[1] / cold[0] - (-3.0 / kt).exp()).abs() < 1e-12);
        assert_eq!(rule.name(), "tabu_heated");
    }

    #[test]
    fn large_visit_counts_do_not_end_the_selection() {
        // MultiOptPy: exp(-0.5 * 2000) underflows for every minimum and the sum is zero.
        let p = pool(&[0.0, 1.0, 2.0], &[2000, 2001, 2000]);
        let mut tabu = PenalizedBoltzmann {
            temperature: 298.15,
            alpha: 0.5,
        };
        let mut rng = Rng::from_seed(3);
        let picks: Vec<usize> = (0..200).filter_map(|_| tabu.select(&p, &mut rng)).collect();
        assert_eq!(picks.len(), 200);
        assert!(picks.iter().filter(|&&i| i == 0).count() > 150);
    }

    #[test]
    fn the_adaptive_level_takes_crowded_minima_one_by_one() {
        let kcal = Units::KCAL_MOL_TO_HARTREE;
        // Ten minima within 0.5 kcal/mol at 4 kcal/mol, one far above.
        let mut energies = vec![0.0];
        energies.extend((0..10).map(|k| 4.0 + 0.05 * k as f64));
        energies.push(20.0);
        let visits = vec![0; energies.len()];
        let mut p = pool(&energies, &visits);
        let mut fixed =
            crate::workflow::explore::WaterFilling::new(Quantity::new(kcal), 1).unwrap();
        let mut adaptive =
            AdaptiveWaterFilling::new(Quantity::new(0.01 * kcal), Quantity::new(3.0 * kcal), 1, 1)
                .unwrap();
        let mut rng = Rng::from_seed(0);
        let submerged = |level: f64| {
            energies
                .iter()
                .filter(|e| **e * kcal <= level + 1e-15)
                .count()
        };
        assert_eq!(fixed.select(&p, &mut rng), Some(0));
        p[0].visits = 1;
        p[1].visits = 1;
        // The fixed step of 1 kcal/mol submerges the other nine at once.
        let _ = fixed.select(&p, &mut rng);
        assert_eq!(submerged(fixed.level().value()), 11);
        // The adaptive one reaches the next minimum only.
        assert_eq!(adaptive.select(&p, &mut rng), Some(2));
        assert_eq!(submerged(adaptive.level().value()), 3);
        p[2].visits = 1;
        assert_eq!(adaptive.select(&p, &mut rng), Some(3));
        assert_eq!(submerged(adaptive.level().value()), 4);
        assert!(
            AdaptiveWaterFilling::new(Quantity::new(kcal), Quantity::new(0.5 * kcal), 1, 1)
                .is_err()
        );
        let reg = PriorityRegistry::default();
        for n in ["boltzmann_heated", "tabu_heated", "water_filling_adaptive"] {
            assert_eq!(reg.create(n).unwrap().name(), n);
        }
    }
}
