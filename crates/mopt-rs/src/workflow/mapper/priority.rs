//! Which node of the network is explored next (extension point): an [`ExplorationPriority`]
//! chooses among the minima that still have tasks. Every rule of the existing
//! [`crate::workflow::explore::PriorityRegistry`] (Boltzmann, tabu, water filling, ...) is one
//! through [`SamplingAdapter`], which passes the energies and visit counts of the candidates;
//! the kinetic rules of `kinetics_beyond_tst.md` (section 3.11) will be further
//! implementations that read the network. Rules are registered by name in an
//! [`ExplorationPriorityRegistry`].
//!
//! Built in: `boltzmann` (the default: `P_i` proportional to `exp(-(E_i - E_min) / k_B T)` at
//! the temperature of the run, MultiOptPy's `temperature_K`), `tabu` (the same with a visit
//! penalty), the other names of the existing registry, and `kinetics` ([`KineticPriority`]:
//! the minimum with the highest kinetic score, by default the population at the time of
//! interest as MultiOptPy's `--use_rcmc`, with a solver of [`crate::kinetics`]). The name
//! `rcmc` is refused with a pointer to `kinetics`, in which RCMC is one of the solvers.
//!
//! MultiOptPy: `Wrapper/mapper.py:ExplorationQueue`, `BoltzmannQueue` and
//!  `Utils/rcmc.py:RCMCQueue`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use super::network::{Level, Network};
use crate::core_infra::rng::Rng;
use crate::domain::Quantity;
use crate::error::{MoptError, MoptResult};
use crate::kinetics::conditions::{Clamp, EnergyKind};
use crate::kinetics::contract::TimeRule;
use crate::kinetics::observe::{scores_observed, Score};
use crate::kinetics::{initial_populations, prepare, KineticsParts, KineticsSettings};
use crate::workflow::explore::{
    Boltzmann, EqRecord, PenalizedBoltzmann, PriorityRegistry, SamplingPriority,
};
use crate::workflow::kinetics::kinetic_network;

/// A node that can be explored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The node.
    pub node: usize,
    /// Its energy (Hartree; the free energy where the node has one).
    pub energy: f64,
    /// Jobs started from it so far.
    pub visits: usize,
    /// Tasks it has left.
    pub untried: usize,
}

/// The choice of the node to explore next.
pub trait ExplorationPriority: Send {
    /// Its name in the registry.
    fn name(&self) -> &str;

    /// The candidate (index into `candidates`, which is never empty) to explore in job
    /// `job`, or `None` to end the search. A rule that scores candidates returns the one with
    /// the highest score.
    fn select(
        &mut self,
        network: &Network,
        candidates: &[Candidate],
        job: usize,
        rng: &mut Rng,
    ) -> MoptResult<Option<usize>>;

    /// Candidate weights or scores in their original order for the last choice.
    fn last_scores(&self) -> &[f64] {
        &[]
    }

    /// CSV of the last RCMC contraction used for a queue decision.
    fn contraction(&self) -> Option<&str> {
        None
    }

    /// Why the last [`Self::select`] chose what it chose, for the log (`None`: it does not
    /// say).
    fn reason(&self) -> Option<String> {
        None
    }
}

/// A rule of [`crate::workflow::explore`] as a priority of the mapper.
pub struct SamplingAdapter {
    rule: Box<dyn SamplingPriority>,
    weights: Vec<f64>,
}

impl SamplingAdapter {
    /// Choose with `rule`.
    pub fn new(rule: Box<dyn SamplingPriority>) -> Self {
        Self {
            rule,
            weights: Vec::new(),
        }
    }
}

impl ExplorationPriority for SamplingAdapter {
    fn name(&self) -> &str {
        self.rule.name()
    }

    fn select(
        &mut self,
        _network: &Network,
        candidates: &[Candidate],
        job: usize,
        rng: &mut Rng,
    ) -> MoptResult<Option<usize>> {
        let pool: Vec<EqRecord> = candidates
            .iter()
            .map(|c| EqRecord {
                energy: Quantity::new(c.energy),
                visits: c.visits,
            })
            .collect();
        self.rule.on_trial(job);
        let (pick, weights) = self.rule.select_with_weights(&pool, rng);
        self.weights = weights;
        Ok(pick)
    }

    fn last_scores(&self) -> &[f64] {
        &self.weights
    }
}

/// What a priority is made with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrioritySettings {
    /// Temperature (K) of the Boltzmann weights.
    pub temperature: f64,
}

/// Makes a priority.
pub type ExplorationPriorityFactory =
    Arc<dyn Fn(&PrioritySettings) -> MoptResult<Box<dyn ExplorationPriority>> + Send + Sync>;

/// Makes a priority that takes settings of its own (`KEY=VALUE`, e.g. the `kinetics_*` keys
/// of the mapper without the prefix).
pub type ExplorationPriorityOptionsFactory = Arc<
    dyn Fn(&PrioritySettings, &BTreeMap<String, String>) -> MoptResult<Box<dyn ExplorationPriority>>
        + Send
        + Sync,
>;

enum Factory {
    Plain(ExplorationPriorityFactory),
    WithOptions(ExplorationPriorityOptionsFactory),
}

/// Priorities by name.
pub struct ExplorationPriorityRegistry {
    entries: Vec<(String, Factory)>,
}

/// The message of the name `rcmc`.
pub const RCMC_RENAMED: &str = "the priority of the mapper from the kinetics is priority=kinetics \
     (MultiOptPy's --use_rcmc selects it with the solver rcmc), with the solver chosen by \
     kinetics_solver: propagate (the default of priority=kinetics), rcmc (the published RCMC), \
     legacy_rcmc (MultiOptPy's numbers), expm";

/// The kinetic priority (`priority=kinetics`): the minimum with the highest kinetic score in
/// the network as it stands, from the rate constants and solver of its settings. A network
/// without a transition state yet gives no kinetics; the lowest minimum is chosen then, as in
/// MultiOptPy.
pub struct KineticPriority {
    settings: KineticsSettings,
    score: Score,
    time: f64,
    start: Option<usize>,
    parts: KineticsParts,

    /// Why the last choice was made ([`ExplorationPriority::reason`]).
    last: Option<String>,
    last_scores: Vec<f64>,
    contraction: Option<String>,
}

/// The minima named with their scores in the reason of a kinetic choice.
const NAMED_SCORES: usize = 3;

/// The keys of the kinetic priority (without the prefix `kinetics_`).
pub const KINETIC_KEYS: &[&str] = &[
    "score",
    "solver",
    "time",
    "temperature",
    "start",
    "energy",
    "clamp",
    "time_rule",
    "tolerance",
    "rate_theory",
    "corrections",
    "kappa",
];

/// The kinetics that the `kinetics_*` settings choose: the settings of the rate constants and
/// the solver, the score, the time of interest, and the start (shared by `priority=kinetics`
/// and the kinetics at the end of a run, `kinetics=on`).
#[derive(Debug, Clone)]
pub struct KineticChoice {
    /// The rate theory, corrections, conditions, and solver.
    pub settings: KineticsSettings,
    /// The score of a minimum.
    pub score: Score,
    /// The time of interest (s).
    pub time: f64,
    /// The node the population starts in (`None`: the node of the first input).
    pub start: Option<usize>,
}

impl KineticChoice {
    /// From the temperature of the run and the `kinetics_*` options (without the prefix);
    /// unknown keys and names are refused.
    pub fn new(temperature: f64, options: &BTreeMap<String, String>) -> MoptResult<Self> {
        let mut settings = KineticsSettings::default();
        settings.conditions.temperature = temperature;
        let mut score = Score::Population;
        let mut time = 1.0;
        let mut start = None;
        let number = |key: &str, v: &str| -> MoptResult<f64> {
            v.trim().parse::<f64>().map_err(|_| {
                MoptError::parse(format!("kinetics_{key} must be a number, not '{v}'"))
            })
        };
        for (key, v) in options {
            match key.as_str() {
                "score" => score = Score::parse(v)?,
                "solver" => settings.solver = v.trim().to_ascii_lowercase(),
                "time" => time = number(key, v)?,
                "temperature" => settings.conditions.temperature = number(key, v)?,
                "start" => {
                    start = Some(v.trim().trim_start_matches("EQ").parse().map_err(|_| {
                        MoptError::parse(format!("kinetics_start is a node number, not '{v}'"))
                    })?)
                }
                "energy" => settings.conditions.energy = EnergyKind::parse(v)?,
                "clamp" => settings.conditions.clamp = Clamp::parse(v)?,
                "time_rule" => settings.solver_settings.time_rule = TimeRule::parse(v)?,
                "tolerance" => settings.solver_settings.tolerance = number(key, v)?,
                "rate_theory" => settings.rate_theory = v.trim().to_ascii_lowercase(),
                "corrections" => {
                    settings.corrections = v
                        .split(',')
                        .map(|c| c.trim().to_ascii_lowercase())
                        .filter(|c| !c.is_empty() && c != "none")
                        .collect()
                }
                "kappa" => {
                    settings.rate.kappa = Some(number(key, v)?);
                    if !settings.corrections.iter().any(|c| c == "kappa") {
                        settings.corrections.push("kappa".to_string());
                    }
                }
                other => {
                    return Err(MoptError::parse(format!(
                        "unknown setting kinetics_{other} of the kinetics of the mapper \
                         (kinetics_{})",
                        KINETIC_KEYS.join(", kinetics_")
                    )))
                }
            }
        }
        if !(time > 0.0 && time.is_finite()) {
            return Err(MoptError::parse(format!(
                "kinetics_time must be positive (got {time})"
            )));
        }
        settings.conditions.check()?;
        let parts = KineticsParts::default();
        // Unknown names are refused before the run starts.
        parts
            .solvers
            .create(&settings.solver, &settings.solver_settings)?;
        parts
            .theories
            .create(&settings.rate_theory, &settings.rate)?;
        for c in &settings.corrections {
            parts.corrections.create(c, &settings.rate)?;
        }
        Ok(Self {
            settings,
            score,
            time,
            start,
        })
    }
}

impl KineticPriority {
    /// From the settings of the run and the `kinetics_*` options (without the prefix).
    pub fn new(s: &PrioritySettings, options: &BTreeMap<String, String>) -> MoptResult<Self> {
        let choice = KineticChoice::new(s.temperature, options)?;
        Ok(Self {
            settings: choice.settings,
            score: choice.score,
            time: choice.time,
            start: choice.start,
            parts: KineticsParts::default(),
            last: None,
            last_scores: Vec::new(),
            contraction: None,
        })
    }

    /// The scores of the minima of `network` by node number.
    pub fn scores(&self, network: &Network, rng: &mut Rng) -> MoptResult<BTreeMap<usize, f64>> {
        self.score_network(network, rng, None)
    }

    fn score_network(
        &self,
        network: &Network,
        rng: &mut Rng,
        mut snapshot: Option<(usize, &mut Option<String>)>,
    ) -> MoptResult<BTreeMap<usize, f64>> {
        let net = kinetic_network(network, Level::Exploration);
        let mut out = BTreeMap::new();
        if net.bottlenecks.is_empty() || net.states.is_empty() {
            return Ok(out);
        }
        let prepared = prepare(&net, &self.settings, &self.parts)?;
        let rates = &prepared.rates;
        let lowest = (0..rates.states.len()).fold(0, |b, i| {
            if rates.state_energies[i] < rates.state_energies[b] {
                i
            } else {
                b
            }
        });
        let given = self
            .start
            .or_else(|| network.inputs.iter().find_map(|i| i.node))
            .and_then(|id| net.index_of(id))
            .and_then(|k| rates.index_of(k));
        let start_id = net.states[rates.states[given.unwrap_or(lowest)]].id;
        let p0 = initial_populations(&net, rates, &[(start_id, 1.0)])?;
        let mut settings = self.settings.solver_settings;
        settings.legacy_start_known = given.is_some();
        let mut solver = self
            .parts
            .solvers
            .create(&self.settings.solver, &settings)?;
        let visits: Vec<usize> = rates
            .states
            .iter()
            .map(|&k| network.visits(net.states[k].id))
            .collect();
        let capture_enabled =
            snapshot.is_some() && matches!(self.settings.solver.as_str(), "rcmc" | "legacy_rcmc");
        let mut capture = |solution: &crate::kinetics::solve::KineticSolution| {
            if let Some((job, output)) = snapshot.as_mut() {
                **output = crate::workflow::kinetics::contraction_csv(
                    solution,
                    &rates.labels,
                    self.settings.conditions.temperature,
                    *job,
                );
            }
        };
        let v = scores_observed(
            solver.as_mut(),
            &prepared.evolution,
            &p0,
            self.time,
            self.score,
            Some(&visits),
            rng,
            capture_enabled.then_some(&mut capture),
        )?;
        for (i, &k) in rates.states.iter().enumerate() {
            out.insert(net.states[k].id, v[i]);
        }
        Ok(out)
    }
}

impl ExplorationPriority for KineticPriority {
    fn name(&self) -> &str {
        "kinetics"
    }

    fn last_scores(&self) -> &[f64] {
        &self.last_scores
    }

    fn reason(&self) -> Option<String> {
        self.last.clone()
    }

    fn contraction(&self) -> Option<&str> {
        self.contraction.as_deref()
    }

    fn select(
        &mut self,
        network: &Network,
        candidates: &[Candidate],
        job: usize,
        rng: &mut Rng,
    ) -> MoptResult<Option<usize>> {
        let mut contraction = None;
        let score = self.score_network(network, rng, Some((job, &mut contraction)))?;
        self.contraction = contraction;
        self.last = Some(if score.is_empty() {
            "kinetics: no transition state yet, the lowest minimum".to_string()
        } else {
            let mut ranked: Vec<(&usize, &f64)> = score.iter().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(a.1).then(a.0.cmp(b.0)));
            let named: Vec<String> = ranked
                .iter()
                .take(NAMED_SCORES)
                .map(|(id, v)| {
                    let label = network
                        .node(**id)
                        .map_or_else(|_| format!("node {id}"), |n| n.label());
                    format!("{label} {v:.3e}")
                })
                .collect();
            format!(
                "kinetics by {} ({:?} after {:e} s): {}",
                self.settings.solver,
                self.score,
                self.time,
                named.join(", ")
            )
        });
        self.last_scores = candidates
            .iter()
            .map(|c| {
                if score.is_empty() {
                    -c.energy
                } else {
                    score.get(&c.node).copied().unwrap_or(0.0)
                }
            })
            .collect();
        // Keep the first maximum, including ties in the lowest-energy fallback.
        Ok((0..self.last_scores.len()).reduce(|best, i| {
            if self.last_scores[i] > self.last_scores[best] {
                i
            } else {
                best
            }
        }))
    }
}

impl ExplorationPriorityRegistry {
    /// `boltzmann` and `tabu` at the temperature of the run, the other rules of
    /// [`PriorityRegistry`] with their own settings, and `kinetics`.
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        let sampling = PriorityRegistry::default();
        for name in sampling.names() {
            let key = name.to_string();
            r.register(
                name,
                Arc::new(move |_: &PrioritySettings| {
                    Ok(Box::new(SamplingAdapter::new(
                        PriorityRegistry::default().create(&key)?,
                    )) as Box<dyn ExplorationPriority>)
                }),
            );
        }
        r.register(
            "boltzmann",
            Arc::new(|s: &PrioritySettings| {
                Ok(Box::new(SamplingAdapter::new(Box::new(Boltzmann {
                    temperature: s.temperature,
                }))) as Box<dyn ExplorationPriority>)
            }),
        );
        r.register(
            "tabu",
            Arc::new(|s: &PrioritySettings| {
                Ok(Box::new(SamplingAdapter::new(Box::new(PenalizedBoltzmann {
                    temperature: s.temperature,
                    alpha: 0.5,
                }))) as Box<dyn ExplorationPriority>)
            }),
        );
        r.register_with_options(
            super::goal::RrtPriority::NAME,
            Arc::new(|_: &PrioritySettings, o: &BTreeMap<String, String>| {
                Ok(Box::new(super::goal::RrtPriority::new(
                    super::goal::RrtSettings::parse(o)?,
                )) as Box<dyn ExplorationPriority>)
            }),
        );
        r.register_with_options(
            "kinetics",
            Arc::new(|s: &PrioritySettings, o: &BTreeMap<String, String>| {
                Ok(Box::new(KineticPriority::new(s, o)?) as Box<dyn ExplorationPriority>)
            }),
        );
        r
    }

    /// Register (or replace) a priority.
    pub fn register(&mut self, name: &str, factory: ExplorationPriorityFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, Factory::Plain(factory)));
    }

    /// Register (or replace) a priority that takes settings of its own.
    pub fn register_with_options(
        &mut self,
        name: &str,
        factory: ExplorationPriorityOptionsFactory,
    ) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, Factory::WithOptions(factory)));
    }

    /// Whether the priority `name` takes settings of its own (`None`: no such priority).
    pub fn takes_options(&self, name: &str) -> Option<bool> {
        let key = name.to_ascii_lowercase();
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, f)| matches!(f, Factory::WithOptions(_)))
    }

    /// The registered names.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// The priority `name` with `settings`.
    pub fn create(
        &self,
        name: &str,
        settings: &PrioritySettings,
    ) -> MoptResult<Box<dyn ExplorationPriority>> {
        self.create_with(name, settings, &BTreeMap::new())
    }

    /// The priority `name` with `settings` and its own `options`.
    pub fn create_with(
        &self,
        name: &str,
        settings: &PrioritySettings,
        options: &BTreeMap<String, String>,
    ) -> MoptResult<Box<dyn ExplorationPriority>> {
        if !(settings.temperature.is_finite() && settings.temperature > 0.0) {
            return Err(MoptError::parse(format!(
                "the temperature of the priority of the mapper must be positive (got {})",
                settings.temperature
            )));
        }
        let key = name.to_ascii_lowercase();
        let Some((_, factory)) = self.entries.iter().find(|(k, _)| *k == key) else {
            if key == "rcmc" {
                return Err(MoptError::unsupported(format!(
                    "priority=rcmc: {RCMC_RENAMED}"
                )));
            }
            return Err(MoptError::parse(format!(
                "unknown priority '{name}' of the mapper (available: {})",
                self.names().join(", ")
            )));
        };
        match factory {
            Factory::Plain(f) => {
                if let Some(k) = options.keys().next() {
                    return Err(MoptError::parse(format!(
                        "the priority '{name}' takes no settings of its own (kinetics_{k} \
                         belongs to priority=kinetics)"
                    )));
                }
                f(settings)
            }
            Factory::WithOptions(f) => f(settings, options),
        }
    }
}

impl Default for ExplorationPriorityRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::Units;

    #[test]
    fn the_registry_has_the_existing_rules_and_refuses_rcmc() {
        let reg = ExplorationPriorityRegistry::default();
        let s = PrioritySettings { temperature: 300.0 };
        for name in ["boltzmann", "tabu", "water_filling", "boltzmann_heated"] {
            assert_eq!(reg.create(name, &s).unwrap().name(), name);
        }
        let e = reg.create("rcmc", &s).err().unwrap();
        assert!(matches!(e, MoptError::Unsupported(_)), "{e}");
        assert!(e.to_string().contains("priority=kinetics"), "{e}");
        // The kinetic priority and its settings.
        let options = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        assert_eq!(reg.create("kinetics", &s).unwrap().name(), "kinetics");
        let ok = options(&[
            ("score", "residence"),
            ("solver", "rcmc"),
            ("time", "1e-6"),
            ("start", "EQ0"),
            ("corrections", "wigner,eckart"),
            ("kappa", "0.5"),
        ]);
        assert!(reg.create_with("kinetics", &s, &ok).is_ok());
        for bad in [
            options(&[("score", "nope")]),
            options(&[("solver", "nope")]),
            options(&[("time", "-1")]),
            options(&[("nope", "1")]),
            options(&[("corrections", "nope")]),
        ] {
            assert!(reg.create_with("kinetics", &s, &bad).is_err(), "{bad:?}");
        }
        // Settings of the kinetic priority are refused by the others.
        assert!(reg
            .create_with("boltzmann", &s, &options(&[("time", "1")]))
            .is_err());
        // Without a transition state the lowest candidate is chosen.
        let mut kinetic = reg.create("kinetics", &s).unwrap();
        let few = [
            Candidate {
                node: 0,
                energy: 0.0,
                visits: 0,
                untried: 1,
            },
            Candidate {
                node: 1,
                energy: -1.0,
                visits: 0,
                untried: 1,
            },
        ];
        let mut rng0 = Rng::from_seed(1);
        assert_eq!(
            kinetic
                .select(&Network::default(), &few, 0, &mut rng0)
                .unwrap(),
            Some(1)
        );
        assert!(reg.create("nope", &s).is_err());
        assert!(reg
            .create("boltzmann", &PrioritySettings { temperature: 0.0 })
            .is_err());
        // A low minimum is chosen far more often than one 3 kcal/mol above it.
        let mut rule = reg.create("boltzmann", &s).unwrap();
        let candidates = [
            Candidate {
                node: 0,
                energy: 0.0,
                visits: 0,
                untried: 1,
            },
            Candidate {
                node: 4,
                energy: 3.0 * Units::KCAL_MOL_TO_HARTREE,
                visits: 0,
                untried: 1,
            },
        ];
        let mut rng = Rng::from_seed(5);
        let net = Network::default();
        let low = (0..400)
            .filter(|k| rule.select(&net, &candidates, *k, &mut rng).unwrap() == Some(0))
            .count();
        assert!(low > 380, "{low}");
    }
}
