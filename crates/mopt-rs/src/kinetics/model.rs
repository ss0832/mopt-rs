//! Kinetic models: how the evolution law is assembled from a network and its rate constants.
//! Built in: `canonical` (thermalized minima, first-order steps, with the bimolecular steps
//! pseudo-first order at the concentrations of their species: `dp/dt = K p` with the
//! generator of the [`RateSet`]), `mass_action` (the states and the species as
//! concentrations, non-linear, [`crate::kinetics::mass_action`]), `egme` (the energy-grained
//! master equation, [`crate::kinetics::egme`]), `two_grain` (hot intermediates as additional
//! states, [`crate::kinetics::hot`]), and `semi_markov` (residence times that need not be
//! exponential, [`crate::kinetics::semi_markov`]).
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue.pop` (the canonical model only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use super::conditions::Conditions;
use super::egme::{Egme, EgmeSettings};
use super::hot::{HotSettings, TwoGrain};
use super::mass_action::MassActionSystem;
use super::network::KineticNetwork;
use super::rate::{RateSet, RateSettings};
use super::semi_markov::semi_markov_of;
use super::solve::Evolution;
use crate::error::{MoptError, MoptResult};

/// What the models are made with.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelSettings {
    /// `mass_action`: the initial concentration (mol/L) of the states together.
    pub total_concentration: Option<f64>,
    /// `egme`: the grains and the collisions.
    pub egme: EgmeSettings,
    /// `two_grain`: the cooling rate and the threshold of `H`.
    pub hot: HotSettings,
    /// The settings of the rate theories (filled in by [`crate::kinetics::prepare`]): `egme`
    /// and `two_grain` count the states as `rrkm` does.
    pub rate: RateSettings,
    /// `--model auto`: the largest relative change of the partner of a pseudo-first-order step
    /// (`None`: [`crate::kinetics::order::DEFAULT_ORDER_TOLERANCE`]).
    pub order_tolerance: Option<f64>,
}

/// A kinetic model.
pub trait KineticModel: Send + Sync {
    /// Its name in the registry.
    fn name(&self) -> &str;

    /// Whether it takes the rate constants with every species at `c°` (a model of
    /// concentrations) instead of at the concentrations given (pseudo-first order).
    fn rates_at_standard_state(&self) -> bool {
        false
    }

    /// The evolution law of `net` with the rate constants `rates`.
    fn assemble(
        &self,
        net: &KineticNetwork,
        rates: &RateSet,
        c: &Conditions,
    ) -> MoptResult<Evolution>;
}

/// Thermalized minima with first-order steps.
#[derive(Debug, Clone, Copy, Default)]
pub struct Canonical;

impl KineticModel for Canonical {
    fn name(&self) -> &str {
        "canonical"
    }

    fn assemble(
        &self,
        _net: &KineticNetwork,
        rates: &RateSet,
        _c: &Conditions,
    ) -> MoptResult<Evolution> {
        Ok(Evolution::Linear(rates.generator()?))
    }
}

/// Mass action of the states and the species.
#[derive(Debug, Clone, Copy, Default)]
pub struct MassAction {
    /// The initial concentration (mol/L) of the states together.
    pub total_concentration: Option<f64>,
}

impl KineticModel for MassAction {
    fn name(&self) -> &str {
        "mass_action"
    }

    fn rates_at_standard_state(&self) -> bool {
        true
    }

    fn assemble(
        &self,
        _net: &KineticNetwork,
        rates: &RateSet,
        _c: &Conditions,
    ) -> MoptResult<Evolution> {
        Ok(Evolution::MassAction(MassActionSystem::new(
            rates,
            self.total_concentration,
        )?))
    }
}

/// Makes a model.
pub type ModelFactory =
    Arc<dyn Fn(&ModelSettings) -> MoptResult<Box<dyn KineticModel>> + Send + Sync>;

/// Names of the design that are refused, with the reason (none at present).
const NOT_IMPLEMENTED: &[(&str, &str)] = &[];

/// The energy-grained master equation.
impl KineticModel for Egme {
    fn name(&self) -> &str {
        "egme"
    }

    fn assemble(
        &self,
        net: &KineticNetwork,
        rates: &RateSet,
        c: &Conditions,
    ) -> MoptResult<Evolution> {
        Ok(Evolution::Resolved(self.resolve(net, rates, c)?))
    }
}

/// Hot intermediates as additional states.
impl KineticModel for TwoGrain {
    fn name(&self) -> &str {
        "two_grain"
    }

    fn assemble(
        &self,
        net: &KineticNetwork,
        rates: &RateSet,
        c: &Conditions,
    ) -> MoptResult<Evolution> {
        Ok(Evolution::Resolved(self.resolve(net, rates, c)?))
    }
}

/// A semi-Markov chain of the rate constants with gamma-distributed residence times.
#[derive(Debug, Clone, Copy, Default)]
pub struct SemiMarkovModel;

impl KineticModel for SemiMarkovModel {
    fn name(&self) -> &str {
        "semi_markov"
    }

    fn assemble(
        &self,
        net: &KineticNetwork,
        rates: &RateSet,
        _c: &Conditions,
    ) -> MoptResult<Evolution> {
        Ok(Evolution::SemiMarkov(semi_markov_of(net, rates)?))
    }
}

/// Models by name.
pub struct ModelRegistry {
    entries: Vec<(String, ModelFactory)>,
}

impl ModelRegistry {
    /// `canonical`, `mass_action`, `egme`, `two_grain`, and `semi_markov`.
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register(
            "canonical",
            Arc::new(|_: &ModelSettings| Ok(Box::new(Canonical) as Box<dyn KineticModel>)),
        );
        r.register(
            "mass_action",
            Arc::new(|s: &ModelSettings| {
                Ok(Box::new(MassAction {
                    total_concentration: s.total_concentration,
                }) as Box<dyn KineticModel>)
            }),
        );
        r.register(
            "egme",
            Arc::new(|s: &ModelSettings| {
                Ok(Box::new(Egme {
                    settings: s.egme,
                    rrkm: s.rate.rrkm,
                    symmetry: s.rate.qrrho.symmetry,
                }) as Box<dyn KineticModel>)
            }),
        );
        r.register(
            "two_grain",
            Arc::new(|s: &ModelSettings| {
                Ok(Box::new(TwoGrain {
                    settings: s.hot,
                    rrkm: s.rate.rrkm,
                    symmetry: s.rate.qrrho.symmetry,
                }) as Box<dyn KineticModel>)
            }),
        );
        r.register(
            "semi_markov",
            Arc::new(|_: &ModelSettings| Ok(Box::new(SemiMarkovModel) as Box<dyn KineticModel>)),
        );
        r
    }

    /// Register (or replace) a model.
    pub fn register(&mut self, name: &str, factory: ModelFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, factory));
    }

    /// The registered names.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// The model `name`.
    pub fn create(&self, name: &str, s: &ModelSettings) -> MoptResult<Box<dyn KineticModel>> {
        let key = name.to_ascii_lowercase();
        if let Some((_, f)) = self.entries.iter().find(|(k, _)| *k == key) {
            return f(s);
        }
        match NOT_IMPLEMENTED.iter().find(|(n, _)| *n == key) {
            Some((n, why)) => Err(MoptError::unsupported(format!(
                "the kinetic model '{n}' is not implemented: it needs {why}"
            ))),
            None => Err(MoptError::parse(format!(
                "unknown kinetic model '{name}' (available: {})",
                self.names().join(", ")
            ))),
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}
