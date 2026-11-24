//! Kinetics of reaction networks: rate constants from the stationary points of a network, the
//! master equation of the populations, and its solutions, among them the rate constant matrix
//! contraction (RCMC). Pure numerics without I/O; the network files are read and the outputs
//! written by [`crate::workflow::kinetics`] (`run_kinetics`, `run_rcmc`), and the mapper ranks
//! its minima with [`crate::workflow::mapper::priority`] (`priority=kinetics`).
//!
//! The parts, each a trait with a registry (`kinetics_beyond_tst.md`, section 5):
//!
//! - [`rate::RateTheory`] (`eyring`, `tst`, `rrkm` of [`rrkm`]) and [`rate::RateCorrection`]
//!   (tunnelling: `wigner`, `bell`, `eckart`, `skodje_truhlar`, `small_curvature`,
//!   `zero_curvature`; recrossing: `kappa`, `kramers`, `grote_hynes`, `reactive_flux`;
//!   crossings: `landau_zener`, `zhu_nakamura`, `weak_coupling`; `cvt`; the diffusion limit
//!   `collins_kimball`): the rate constants in flux form, [`rate::RateSet`];
//! - [`model::KineticModel`] (`canonical`, `mass_action`, `egme` of [`egme`], `two_grain` of
//!   [`hot`], `semi_markov` of [`semi_markov`]): the generator of `dp/dt = K p`,
//!   [`generator::Generator`] (on energy grains or hot grains for the resolved models), or the
//!   equations of mass action, [`mass_action::MassActionSystem`]; the bimolecular steps take up
//!   or give off the species of [`species`] (pseudo-first order at their concentrations in
//!   `canonical`);
//! - [`solve::KineticSolver`] (`propagate`, `rcmc`, `legacy_rcmc`, `expm`, `spectral`,
//!   `steady`, `rosenbrock`, `kmc`): populations at given times with diagnostics, built on the
//!   subtraction-free elimination of [`contract`](mod@contract), the matrix exponential of
//!   [`expm`](mod@expm), the Rosenbrock integrator of [`mass_action`], and Gillespie's sampling;
//! - [`observe`]: the scores of the states for an exploration (`population`,
//!   `max_population`, `residence`, `discovery_flux`); [`sensitivity`] and [`derivative`]:
//!   degrees of rate control; [`refine`]: refinement tasks ranked by sensitivity times
//!   uncertainty; [`uncertainty`]: propagation of energy errors; [`branching`]: branching
//!   fractions and transmission coefficients from trajectory pairs; [`msm`]: Markov state models
//!   and residence times from trajectories.
//!
//! [`KineticsSettings`] and [`prepare`] put the parts together from their names.
//!
//! MultiOptPy: `Utils/rcmc.py` (`RCMCQueue`, `main`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod branching;
pub mod conditions;
pub mod contract;
pub mod derivative;
pub mod egme;
pub mod expm;
pub mod generator;
pub mod hot;
pub mod mass_action;
pub mod model;
pub mod msm;
pub mod network;
pub mod observe;
pub mod order;
pub mod quadrature;
pub mod rate;
pub mod refine;
pub mod rrkm;
pub mod semi_markov;
pub mod sensitivity;
pub mod solve;
pub mod species;
pub mod uncertainty;

use std::sync::Arc;

pub use conditions::{Clamp, Conditions, EnergyKind};
pub use contract::{contract, greedy_order, stationary, Contraction, TimeRule};
pub use generator::Generator;
pub use mass_action::{MassActionSystem, Rosenbrock};
pub use model::{KineticModel, ModelRegistry, ModelSettings};
pub use network::{Bottleneck, BottleneckKind, KineticNetwork, KineticState, Partner, Species};
pub use observe::{scores, Score};
pub use rate::{
    RateCorrection, RateCorrectionRegistry, RateSet, RateSettings, RateTheory, RateTheoryRegistry,
};
pub use solve::{
    Closure, Evolution, KineticSolution, KineticSolver, SolverRegistry, SolverSettings,
};
pub use species::{Association, SpeciesSettings};

use crate::error::{MoptError, MoptResult};

/// The choices of a kinetic model and its solution, by name.
#[derive(Debug, Clone, PartialEq)]
pub struct KineticsSettings {
    /// Temperature, standard state, energies, clamp.
    pub conditions: Conditions,
    /// The rate theory (`eyring`).
    pub rate_theory: String,
    /// The corrections (none).
    pub corrections: Vec<String>,
    /// What the theories and corrections are made with.
    pub rate: RateSettings,
    /// The model (`canonical`).
    pub model: String,
    /// The solver (`propagate`).
    pub solver: String,
    /// What the solvers are made with.
    pub solver_settings: SolverSettings,
    /// The concentrations of the species and the treatment of their association.
    pub species: SpeciesSettings,
    /// What the models are made with.
    pub model_settings: ModelSettings,
}

impl Default for KineticsSettings {
    fn default() -> Self {
        Self {
            conditions: Conditions::default(),
            rate_theory: "eyring".to_string(),
            corrections: Vec::new(),
            rate: RateSettings::default(),
            model: "canonical".to_string(),
            solver: "propagate".to_string(),
            solver_settings: SolverSettings::default(),
            species: SpeciesSettings::default(),
            model_settings: ModelSettings::default(),
        }
    }
}

/// The registries the parts are taken from.
#[derive(Default)]
pub struct KineticsParts {
    /// Rate theories.
    pub theories: RateTheoryRegistry,
    /// Corrections.
    pub corrections: RateCorrectionRegistry,
    /// Models.
    pub models: ModelRegistry,
    /// Solvers.
    pub solvers: SolverRegistry,
}

/// The rate constants and the evolution law of a network.
#[derive(Debug, Clone)]
pub struct Prepared {
    /// The rate constants.
    pub rates: RateSet,
    /// The evolution law.
    pub evolution: Evolution,
}

/// The rate constants and the evolution law of `net` under `s`.
pub fn prepare(
    net: &KineticNetwork,
    s: &KineticsSettings,
    parts: &KineticsParts,
) -> MoptResult<Prepared> {
    let theory = parts.theories.create(&s.rate_theory, &s.rate)?;
    let corrections: Vec<Arc<dyn RateCorrection>> = s
        .corrections
        .iter()
        .map(|c| parts.corrections.create(c, &s.rate))
        .collect::<MoptResult<_>>()?;
    let model_settings = ModelSettings {
        rate: s.rate,
        ..s.model_settings
    };
    let model = parts.models.create(&s.model, &model_settings)?;
    let species = SpeciesSettings {
        at_standard: model.rates_at_standard_state(),
        symmetry: s.rate.qrrho.symmetry,
        ..s.species.clone()
    };
    let rates =
        RateSet::assemble_with(net, theory.as_ref(), &corrections, &s.conditions, &species)?;
    let evolution = model.assemble(net, &rates, &s.conditions)?;
    Ok(Prepared { rates, evolution })
}

/// The initial populations over the states of `rates` from `(node id, share)` pairs of the
/// network (normalized to 1).
pub fn initial_populations(
    net: &KineticNetwork,
    rates: &RateSet,
    shares: &[(usize, f64)],
) -> MoptResult<Vec<f64>> {
    let mut p = vec![0.0; rates.states.len()];
    for &(id, share) in shares {
        let Some(k) = net.index_of(id) else {
            return Err(MoptError::config(format!(
                "the initial population names the node {id}, which is not a state of the network"
            )));
        };
        let Some(i) = rates.index_of(k) else {
            return Err(MoptError::config(format!(
                "the initial population names {}, which is left out of the model ({})",
                net.states[k].label, rates.description
            )));
        };
        if !(share > 0.0 && share.is_finite()) {
            return Err(MoptError::config(format!(
                "the initial share of {} must be positive (got {share})",
                net.states[k].label
            )));
        }
        p[i] += share;
    }
    let total: f64 = p.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return Err(MoptError::config("the initial population is empty"));
    }
    for v in &mut p {
        *v /= total;
    }
    Ok(p)
}
