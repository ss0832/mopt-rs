//! Which local minimum (EQ) to sample next (extension point).
//!
//! Searches that collect EQs (conformation search, reaction-network exploration) repeatedly pick
//! one known EQ and start a new trial from it. A [`SamplingPriority`] makes that choice from the
//! energies and visit counts of the known EQs; any number of rules can be registered in a
//! [`PriorityRegistry`] and selected by name.
//!
//! Built in:
//! - [`Boltzmann`]: `P_i ∝ exp(-(E_i - E_min) / k_B T)` (MultiOptPy `confsearch`);
//! - [`PenalizedBoltzmann`]: `P_i ∝ exp(-(E_i - E_min) / k_B T) exp(-alpha N_i)` with visit
//!   counts `N_i` (MultiOptPy `confsearch -tabu`);
//! - [`WaterFilling`]: a deterministic "rising water" rule. A water level starts at the lowest
//!   EQ; the EQs below the level are sampled, least-visited first (lower energy breaks ties),
//!   until each has been sampled `visits_per_level` times; then the level rises by `step`. Low
//!   basins are thus filled before higher ones are explored, and no EQ under water is starved;
//! - in [`super::priority_rules`]: MultiOptPy's selection with its periodic heating
//!   (`boltzmann_heated`, `tabu_heated`) and the rising water with a step adapted to the
//!   spacing of the minima (`water_filling_adaptive`).
//!
//! Weights are drawn from their logarithms shifted by the largest, so that they never all
//! underflow (MultiOptPy's penalized weights do once every visit count is large).
//!
//! MultiOptPy: the choice of the next conformer in `Entrypoints/conformation_search.py`
//!  (`calc_boltzmann_distribution`, `-tabu`); the trait and the water-filling rule are new.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use crate::core_infra::rng::Rng;
use crate::domain::{Hartree, Quantity, Units};
use crate::error::{MoptError, MoptResult};

/// A known local minimum.
#[derive(Debug, Clone, PartialEq)]
pub struct EqRecord {
    /// Energy.
    pub energy: Quantity<Hartree>,
    /// How often a trial was started from this EQ.
    pub visits: usize,
}

/// Rule choosing the EQ to sample next.
pub trait SamplingPriority: Send {
    /// Canonical name.
    fn name(&self) -> &str;

    /// Index into `pool` of the EQ to sample next, or `None` when the rule considers the search
    /// finished. `pool` is never empty.
    fn select(&mut self, pool: &[EqRecord], rng: &mut Rng) -> Option<usize>;

    /// Select once and return the numerical priorities used, when the rule has them.
    fn select_with_weights(
        &mut self,
        pool: &[EqRecord],
        rng: &mut Rng,
    ) -> (Option<usize>, Vec<f64>) {
        (self.select(pool, rng), Vec::new())
    }

    /// The number of the trial (from 0) whose end the next [`Self::select`] follows, for rules
    /// that change with the trials (MultiOptPy heats every fifth); nothing by default.
    fn on_trial(&mut self, trial: usize) {
        let _ = trial;
    }
}

/// Logarithms of the Boltzmann weights `-(E_i - E_min) / k_B T`.
pub(crate) fn boltzmann_log_weights(pool: &[EqRecord], temperature: f64) -> Vec<f64> {
    let e_min = pool
        .iter()
        .map(|r| r.energy.value())
        .fold(f64::INFINITY, f64::min);
    pool.iter()
        .map(|r| {
            let de = (r.energy.value() - e_min) * Units::HARTREE_TO_KCAL_MOL;
            -de / (Units::BOLTZMANN_KCAL_MOL_K * temperature)
        })
        .collect()
}

fn boltzmann_weights(pool: &[EqRecord], temperature: f64) -> Vec<f64> {
    boltzmann_log_weights(pool, temperature)
        .into_iter()
        .map(f64::exp)
        .collect()
}

/// Shift the logarithms before exponentiating so the largest finite weight is one.
pub(super) fn relative_weights(log_weights: &[f64]) -> Vec<f64> {
    let top = log_weights
        .iter()
        .copied()
        .filter(|w| w.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return vec![0.0; log_weights.len()];
    }
    log_weights
        .iter()
        .map(|l| if l.is_finite() { (l - top).exp() } else { 0.0 })
        .collect()
}

/// Draw an index with probability proportional to `weights`.
pub(super) fn draw(weights: &[f64], rng: &mut Rng) -> Option<usize> {
    let total: f64 = weights.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let u = rng.next_f64() * total;
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += w;
        if u < acc {
            return Some(i);
        }
    }
    Some(weights.len() - 1)
}

/// Boltzmann selection (MultiOptPy `confsearch`, `-st` temperature).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boltzmann {
    /// Temperature (K); MultiOptPy's default is 298.15 K.
    pub temperature: f64,
}

impl SamplingPriority for Boltzmann {
    fn name(&self) -> &str {
        "boltzmann"
    }

    fn select(&mut self, pool: &[EqRecord], rng: &mut Rng) -> Option<usize> {
        self.select_with_weights(pool, rng).0
    }

    fn select_with_weights(
        &mut self,
        pool: &[EqRecord],
        rng: &mut Rng,
    ) -> (Option<usize>, Vec<f64>) {
        let weights = boltzmann_weights(pool, self.temperature);
        (draw(&weights, rng), weights)
    }
}

/// Boltzmann selection with a visit penalty `exp(-alpha N_i)` (MultiOptPy `confsearch -tabu`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PenalizedBoltzmann {
    /// Temperature (K).
    pub temperature: f64,
    /// Penalty coefficient `alpha` (MultiOptPy default 0.5).
    pub alpha: f64,
}

impl SamplingPriority for PenalizedBoltzmann {
    fn name(&self) -> &str {
        "tabu"
    }

    fn select(&mut self, pool: &[EqRecord], rng: &mut Rng) -> Option<usize> {
        self.select_with_weights(pool, rng).0
    }

    fn select_with_weights(
        &mut self,
        pool: &[EqRecord],
        rng: &mut Rng,
    ) -> (Option<usize>, Vec<f64>) {
        let w: Vec<f64> = boltzmann_log_weights(pool, self.temperature)
            .into_iter()
            .zip(pool)
            .map(|(b, r)| b - self.alpha * r.visits as f64)
            .collect();
        let weights = relative_weights(&w);
        (draw(&weights, rng), weights)
    }
}

/// Rising-water selection.
#[derive(Debug, Clone, PartialEq)]
pub struct WaterFilling {
    /// Rise of the water level per stage.
    pub step: Quantity<Hartree>,
    /// Trials per EQ before the level rises.
    pub visits_per_level: usize,
    /// Level (above the lowest EQ) at which the search ends; unbounded when `None`.
    pub max_level: Option<Quantity<Hartree>>,
    /// Current level above the lowest EQ.
    level: f64,
}

impl WaterFilling {
    /// Rule raising the level by `step` once every submerged EQ was sampled
    /// `visits_per_level` times.
    pub fn new(step: Quantity<Hartree>, visits_per_level: usize) -> MoptResult<Self> {
        if !step.value().is_finite() || step.value() <= 0.0 || visits_per_level == 0 {
            return Err(MoptError::config(
                "water filling needs a positive step and at least one visit per level",
            ));
        }
        Ok(Self {
            step,
            visits_per_level,
            max_level: None,
            level: 0.0,
        })
    }

    /// Stop once the level exceeds `max_level` above the lowest EQ.
    #[must_use]
    pub fn with_max_level(mut self, max_level: Quantity<Hartree>) -> Self {
        self.max_level = Some(max_level);
        self
    }

    /// Current water level above the lowest EQ.
    pub fn level(&self) -> Quantity<Hartree> {
        Quantity::new(self.level)
    }
}

impl SamplingPriority for WaterFilling {
    fn name(&self) -> &str {
        "water_filling"
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
            // Everything under water is done: raise the level, unless nothing is left above.
            let next = pool
                .iter()
                .map(|r| r.energy.value() - e_min)
                .filter(|d| *d > self.level + 1e-15)
                .fold(f64::INFINITY, f64::min);
            if !next.is_finite() {
                return None;
            }
            // Rise in whole steps until the next EQ is submerged.
            let steps = ((next - self.level) / self.step.value()).ceil().max(1.0);
            self.level += steps * self.step.value();
        }
    }
}

/// Factory of a sampling rule from its parameters `(temperature, alpha, ...)`.
pub type PriorityFactory = Arc<dyn Fn() -> Box<dyn SamplingPriority> + Send + Sync>;

/// Named sampling rules.
pub struct PriorityRegistry {
    entries: Vec<(String, PriorityFactory)>,
}

impl PriorityRegistry {
    /// Built-in rules with MultiOptPy's defaults (298.15 K, alpha = 0.5) and water filling in
    /// 1 kcal/mol steps with three trials per EQ.
    pub fn with_defaults() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register(
            "boltzmann",
            Arc::new(|| {
                Box::new(Boltzmann {
                    temperature: 298.15,
                }) as Box<dyn SamplingPriority>
            }),
        );
        r.register(
            "tabu",
            Arc::new(|| {
                Box::new(PenalizedBoltzmann {
                    temperature: 298.15,
                    alpha: 0.5,
                }) as Box<dyn SamplingPriority>
            }),
        );
        r.register(
            "water_filling",
            Arc::new(|| {
                Box::new(WaterFilling {
                    step: Quantity::new(Units::KCAL_MOL_TO_HARTREE),
                    visits_per_level: 3,
                    max_level: None,
                    level: 0.0,
                }) as Box<dyn SamplingPriority>
            }),
        );
        super::priority_rules::register_more(&mut r);
        r
    }

    /// Register (or replace) a rule.
    pub fn register(&mut self, name: &str, factory: PriorityFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, factory));
    }

    /// Registered names.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// New instance of the rule `name`.
    pub fn create(&self, name: &str) -> MoptResult<Box<dyn SamplingPriority>> {
        let key = name.to_ascii_lowercase();
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, f)| f())
            .ok_or_else(|| {
                MoptError::config(format!(
                    "unknown sampling priority '{name}' (available: {})",
                    self.names().join(", ")
                ))
            })
    }
}

impl Default for PriorityRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn pool(kcal: &[f64], visits: &[usize]) -> Vec<EqRecord> {
        kcal.iter()
            .zip(visits)
            .map(|(e, v)| EqRecord {
                energy: Quantity::new(e * Units::KCAL_MOL_TO_HARTREE),
                visits: *v,
            })
            .collect()
    }

    fn frequencies(rule: &mut dyn SamplingPriority, p: &[EqRecord], n: usize) -> Vec<f64> {
        let mut rng = Rng::from_seed(11);
        let mut c = vec![0usize; p.len()];
        for _ in 0..n {
            c[rule.select(p, &mut rng).unwrap()] += 1;
        }
        c.into_iter().map(|x| x as f64 / n as f64).collect()
    }

    #[test]
    fn boltzmann_and_tabu_match_multioptpy() {
        let p = pool(&[0.0, 0.5, 2.0], &[0, 4, 1]);
        let kt = Units::BOLTZMANN_KCAL_MOL_K * 298.15;
        let w: Vec<f64> = [0.0_f64, 0.5, 2.0]
            .iter()
            .map(|e| (-e / kt).exp())
            .collect();
        let z: f64 = w.iter().sum();
        let f = frequencies(
            &mut Boltzmann {
                temperature: 298.15,
            },
            &p,
            200_000,
        );
        for i in 0..3 {
            assert!(
                (f[i] - w[i] / z).abs() < 5e-3,
                "{i}: {} vs {}",
                f[i],
                w[i] / z
            );
        }
        let wp: Vec<f64> = w
            .iter()
            .zip([0.0, 4.0, 1.0])
            .map(|(b, n): (&f64, f64)| b * (-0.5 * n).exp())
            .collect();
        let zp: f64 = wp.iter().sum();
        let mut tabu = PenalizedBoltzmann {
            temperature: 298.15,
            alpha: 0.5,
        };
        let f = frequencies(&mut tabu, &p, 200_000);
        for i in 0..3 {
            assert!((f[i] - wp[i] / zp).abs() < 5e-3, "{i}");
        }
    }

    #[test]
    fn water_filling_fills_low_basins_first() {
        let mut rule = WaterFilling::new(Quantity::new(Units::KCAL_MOL_TO_HARTREE), 2)
            .unwrap()
            .with_max_level(Quantity::new(3.0 * Units::KCAL_MOL_TO_HARTREE));
        let mut rng = Rng::from_seed(1);
        let mut p = pool(&[0.0, 0.4, 1.5, 2.6, 7.0], &[0, 0, 0, 0, 0]);
        let mut order = Vec::new();
        while let Some(i) = rule.select(&p, &mut rng) {
            p[i].visits += 1;
            order.push(i);
        }
        // Level 0: EQ 0 twice. Level 1: EQ 1 twice. Level 2: EQ 2 twice. Level 3: EQ 3 twice.
        // EQ 4 (7 kcal/mol) stays above the maximum level.
        assert_eq!(order, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        assert_eq!(p[4].visits, 0);
        // A newly found low EQ is served before the level rises again.
        let mut rule = WaterFilling::new(Quantity::new(Units::KCAL_MOL_TO_HARTREE), 1).unwrap();
        let mut p = pool(&[0.0, 3.0], &[1, 0]);
        p.push(EqRecord {
            energy: Quantity::new(0.2 * Units::KCAL_MOL_TO_HARTREE),
            visits: 0,
        });
        assert_eq!(rule.select(&p, &mut rng), Some(2));
    }

    #[test]
    fn registry_creates_and_accepts_user_rules() {
        struct LowestFirst;
        impl SamplingPriority for LowestFirst {
            fn name(&self) -> &str {
                "lowest"
            }
            fn select(&mut self, pool: &[EqRecord], _rng: &mut Rng) -> Option<usize> {
                (0..pool.len())
                    .min_by(|&a, &b| pool[a].energy.value().total_cmp(&pool[b].energy.value()))
            }
        }
        let mut reg = PriorityRegistry::default();
        for n in ["boltzmann", "tabu", "water_filling"] {
            assert_eq!(reg.create(n).unwrap().name(), n);
        }
        reg.register(
            "lowest",
            Arc::new(|| Box::new(LowestFirst) as Box<dyn SamplingPriority>),
        );
        let mut r = reg.create("LOWEST").unwrap();
        let p = pool(&[1.0, -2.0, 0.0], &[0, 0, 0]);
        assert_eq!(r.select(&p, &mut Rng::from_seed(0)), Some(1));
        assert!(reg.create("nope").is_err());
    }
}
