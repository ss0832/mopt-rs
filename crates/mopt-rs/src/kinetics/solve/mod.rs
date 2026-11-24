//! Solvers of a kinetic model: an [`Evolution`] (the generator), an initial population, and
//! output times give populations with diagnostics. Registered by name in a
//! [`SolverRegistry`]:
//!
//! - `propagate` (the default): contraction to superstates, then the exact propagation of the
//!   superstate generator by the matrix exponential (`kinetics_beyond_tst.md`, section 3.4);
//!   fewer states, down to none, are eliminated where the time scales are not separated, so
//!   that it degrades to the plain matrix exponential instead of to a wrong answer;
//! - `rcmc`: the published RCMC output of Type A (a snapshot `q = V p` at the last step whose
//!   reference time does not exceed the requested time);
//! - `legacy_rcmc`: MultiOptPy's `rcmc.py` as it is, for comparison (see `migration.md`);
//! - `expm`: `exp(t K) p0`, the reference;
//! - `spectral`: the eigen-decomposition of the symmetrized generator (diagnostics; small
//!   populations are lost to cancellation);
//! - `steady`: the stationary distribution reached from `p0` (`t -> inf`);
//! - `rosenbrock`: the L-stable Rosenbrock integrator ROS2 with step-size control, for the
//!   non-linear model `mass_action` (concentrations) and the linear ones
//!   ([`crate::kinetics::mass_action`]);
//! - `kmc`: kinetic Monte Carlo (Gillespie's direct method), for small non-stiff models and
//!   for semi-Markov chains with non-exponential residence times ([`kmc`]);
//! - `contract`: the contraction of mass-action networks of any reaction order, the extension
//!   of RCMC to non-linear kinetics ([`contract_mass`]);
//! - `windows`: mass action by pseudo-first-order windows, each propagated by the contraction
//!   with the species held and moved by stoichiometry in between ([`windows`]).
//!
//! Names of the design that are refused by name: `rcmc_b`, `ode`.
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue.pop` (one solver, `legacy_rcmc`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod contract_mass;
pub mod direct;
pub mod kmc;
pub mod legacy;
pub mod propagate;
pub mod rcmc;
pub mod windows;

use std::sync::Arc;

use faer::Mat;

use super::contract::{Contraction, TimeRule};
use super::generator::Generator;
use super::mass_action::{MassActionSystem, Rosenbrock, DEFAULT_ODE_TOLERANCE};
use super::quadrature::gauss_legendre;
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};

pub use contract_mass::ContractMassAction;
pub use direct::{Expm, Spectral, Steady};
pub use kmc::Kmc;
pub use legacy::LegacyRcmc;
pub use propagate::Propagate;
pub use rcmc::Rcmc;
pub use windows::Windows;

/// The evolution law of a kinetic model (one per solution, so the size of its variants does not
/// matter).
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum Evolution {
    /// `dp/dt = K p`.
    Linear(Generator),
    /// Concentrations under mass action (the model `mass_action`).
    MassAction(MassActionSystem),
    /// `dp/dt = K p` on finer states than those of the rate set (the energy grains of `egme`,
    /// the hot grains of `two_grain`), whose populations add up to those of the states.
    Resolved(Resolved),
    /// A semi-Markov chain (`semi_markov`): its Markov limit for the linear solvers, the
    /// residence-time distributions for `kmc`.
    SemiMarkov(SemiMarkov),
}

/// A linear model on finer states: every fine state belongs to one state of the rate set.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    /// The generator of the fine states.
    pub generator: Generator,
    /// The state (of the rate set) of every fine state.
    pub owner: Vec<usize>,
    /// The number of states of the rate set.
    pub states: usize,
    /// The model that made it.
    pub model: String,
    /// The phenomenological rate constants between the states, where the model gives them
    /// (`egme`).
    pub phenomenological: Option<crate::kinetics::egme::Phenomenological>,
    /// What the model says about itself.
    pub notes: Vec<String>,
}

impl Resolved {
    /// The fine populations of the populations `p` of the states: within a state, the
    /// equilibrium distribution over its fine states (`ln pi` of the generator; equal shares
    /// without it).
    pub fn expand(&self, p: &[f64]) -> Vec<f64> {
        let n = self.generator.n;
        let mut weights = vec![1.0; n];
        if let Some(ln_pi) = &self.generator.ln_pi {
            let mut top = vec![f64::NEG_INFINITY; self.states];
            for (f, &o) in self.owner.iter().enumerate() {
                top[o] = top[o].max(ln_pi[f]);
            }
            for (f, &o) in self.owner.iter().enumerate() {
                weights[f] = (ln_pi[f] - top[o]).exp();
            }
        }
        let mut total = vec![0.0; self.states];
        for (f, &o) in self.owner.iter().enumerate() {
            total[o] += weights[f];
        }
        (0..n)
            .map(|f| {
                let o = self.owner[f];
                if total[o] > 0.0 {
                    p.get(o).copied().unwrap_or(0.0) * weights[f] / total[o]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// The populations of the states from those of the fine states.
    pub fn fold(&self, p: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.states];
        for (v, &o) in p.iter().zip(&self.owner) {
            out[o] += v;
        }
        out
    }
}

/// A semi-Markov chain: the jumps of a generator with residence times that need not be
/// exponential.
#[derive(Debug, Clone, PartialEq)]
pub struct SemiMarkov {
    /// The Markov limit `K = (P - I) diag(1 / tau)`: the mean residence times `tau` and the
    /// jump probabilities `P` of the chain.
    pub generator: Generator,
    /// The shape of the gamma distribution of the residence time of every state (1: exponential).
    pub shapes: Vec<f64>,
    /// The first-order memory correction of the Markov limit, `K (I - K1)^-1` with
    /// `K1 = K diag(tau (cv^2 - 1) / 2)` (`kinetics_beyond_tst.md`, section 3.6), dense.
    pub memory: Option<Mat<f64>>,
    /// Where the residence times came from.
    pub notes: Vec<String>,
}

impl Evolution {
    /// The generator of a linear model (an error for the model `mass_action`, which only the
    /// solver `rosenbrock` integrates): of the fine states for a resolved model, the Markov
    /// limit for a semi-Markov chain.
    pub fn generator(&self) -> MoptResult<&Generator> {
        match self {
            Self::Linear(g) => Ok(g),
            Self::Resolved(r) => Ok(&r.generator),
            Self::SemiMarkov(s) => Ok(&s.generator),
            Self::MassAction(_) => Err(MoptError::unsupported(
                "this needs the generator of a linear model (canonical); the model mass_action \
                 is non-linear and is integrated by the solver rosenbrock",
            )),
        }
    }

    /// The populations of the states of the solvers from those of the states of the rate set
    /// (the fine states of a resolved model; the same populations otherwise).
    pub fn expand(&self, p: &[f64]) -> Vec<f64> {
        match self {
            Self::Resolved(r) => r.expand(p),
            _ => p.to_vec(),
        }
    }

    /// The populations of the states of the rate set from those of the solvers.
    pub fn fold(&self, p: &[f64]) -> Vec<f64> {
        match self {
            Self::Resolved(r) => r.fold(p),
            _ => p.to_vec(),
        }
    }
}

/// The closure of the superstate equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closure {
    /// `G = D diag(1^T M)^-1` (the original RCMC; subtraction-free).
    A,
    /// `G = D M^-1` (the better projection; its entries are not sign-definite).
    B,
}

impl Closure {
    /// `a` or `b`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "a" => Ok(Self::A),
            "b" => Ok(Self::B),
            other => Err(MoptError::parse(format!(
                "unknown closure '{other}' (a, b)"
            ))),
        }
    }
}

/// What the solvers are made with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverSettings {
    /// The reference time of a step of RCMC.
    pub time_rule: TimeRule,
    /// `propagate`: the eliminated states must have relaxed (`sigma_hat t >= ln(1 / tol)`)
    /// and the separation ratio `eps_hat` must not exceed it.
    pub tolerance: f64,
    /// The closure of `propagate`.
    pub closure: Closure,
    /// `legacy_rcmc`: whether the start node is known to MultiOptPy (`false` reproduces its
    /// mapper without `rcmc_start_node_id`, which starts the contracted equation from a
    /// uniform vector).
    pub legacy_start_known: bool,
    /// `rosenbrock`: the relative tolerance of the step-size control.
    pub ode_tolerance: f64,
    /// `kmc`: the number of trajectories.
    pub kmc_trajectories: usize,
    /// `kmc`: the largest number of events of a run.
    pub kmc_max_events: u64,
    /// `contract`: species are eliminated while their effective loss rate times `t` exceeds
    /// this.
    pub contract_theta: f64,
    /// `windows`: the largest relative change of a species over one window.
    pub window_tolerance: f64,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            time_rule: TimeRule::Gershgorin,
            tolerance: DEFAULT_TOLERANCE,
            closure: Closure::A,
            legacy_start_known: true,
            ode_tolerance: DEFAULT_ODE_TOLERANCE,
            kmc_trajectories: kmc::DEFAULT_TRAJECTORIES,
            kmc_max_events: kmc::DEFAULT_MAX_EVENTS,
            contract_theta: contract_mass::DEFAULT_THETA,
            window_tolerance: windows::DEFAULT_WINDOW_TOLERANCE,
        }
    }
}

/// The default tolerance of `propagate` (see `kinetics.md`: the contraction is used where its
/// error is below the accuracy of the matrix exponential it is checked against).
pub const DEFAULT_TOLERANCE: f64 = 1e-8;

/// The diagnostics of one output time.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SolveDiagnostics {
    /// The time.
    pub time: f64,
    /// States eliminated.
    pub eliminated: usize,
    /// `eps_hat = rho_hat(D) / sigma_hat(K_SS)` of the cut.
    pub eps_hat: Option<f64>,
    /// `sigma_hat(K_SS) t` (the initial transient is `exp(-sigma t)`).
    pub sigma_t: Option<f64>,
    /// `rho_hat(D) t`.
    pub rho_t: Option<f64>,
    /// `|1^T p(t) - 1^T p(0)|`.
    pub conservation: f64,
    /// The detailed-balance residual of the superstate generator, where it applies.
    pub detailed_balance: Option<f64>,
    /// The most negative population (0 when none is negative).
    pub most_negative: f64,
    /// What else the solver says.
    pub notes: Vec<String>,
}

/// The superstates of an output time.
#[derive(Debug, Clone)]
pub struct Superstates {
    /// The time.
    pub time: f64,
    /// The kept states (indices of the generator).
    pub kept: Vec<usize>,
    /// The states of each superstate, the kept state first.
    pub members: Vec<Vec<usize>>,
    /// The matrix of `rcmc_K_contracted.csv`.
    pub matrix: Mat<f64>,
    /// What that matrix is.
    pub matrix_name: String,
    /// The blocks of the subtraction-free contraction, where the solver has them.
    pub contraction: Option<Contraction>,
}

/// Populations at the output times with diagnostics.
#[derive(Debug, Clone)]
pub struct KineticSolution {
    /// The solver.
    pub solver: String,
    /// The output times.
    pub times: Vec<f64>,
    /// `populations[k][i]`: state `i` at `times[k]`.
    pub populations: Vec<Vec<f64>>,
    /// The diagnostics of every time.
    pub diagnostics: Vec<SolveDiagnostics>,
    /// The superstates of every time, where the solver contracts.
    pub superstates: Vec<Option<Superstates>>,
    /// Notes on the whole solution.
    pub notes: Vec<String>,
}

/// A solver.
pub trait KineticSolver: Send {
    /// Its name in the registry.
    fn name(&self) -> &str;

    /// Populations at `times` from `p0`.
    fn solve(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        times: &[f64],
        rng: &mut Rng,
    ) -> MoptResult<KineticSolution>;

    /// `int_0^t p(s) ds` of every state; by default by Gauss-Legendre panels on a logarithmic
    /// time axis (two per decade over sixteen decades below `t`) with the solutions of
    /// [`KineticSolver::solve`].
    fn residence(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        t: f64,
        rng: &mut Rng,
    ) -> MoptResult<Vec<f64>> {
        residence_by_quadrature(self, ev, p0, t, rng)
    }
}

/// The default of [`KineticSolver::residence`].
pub fn residence_by_quadrature<S: KineticSolver + ?Sized>(
    solver: &mut S,
    ev: &Evolution,
    p0: &[f64],
    t: f64,
    rng: &mut Rng,
) -> MoptResult<Vec<f64>> {
    let decades = 16.0;
    let panels = 32;
    let (x, w) = gauss_legendre(4);
    let (lo, hi) = (t.ln() - decades * std::f64::consts::LN_10, t.ln());
    let h = (hi - lo) / panels as f64;
    let mut times = Vec::new();
    let mut weights = Vec::new();
    for k in 0..panels {
        let mid = lo + (k as f64 + 0.5) * h;
        for (xi, wi) in x.iter().zip(&w) {
            let u = mid + 0.5 * h * xi;
            times.push(u.exp());
            weights.push(0.5 * h * wi * u.exp());
        }
    }
    let sol = solver.solve(ev, p0, &times, rng)?;
    let first = lo.exp();
    let mut tau: Vec<f64> = p0.iter().map(|p| p * first).collect();
    for (wk, p) in weights.iter().zip(&sol.populations) {
        for (t, v) in tau.iter_mut().zip(p) {
            *t += wk * v;
        }
    }
    Ok(tau)
}

/// Check `p0` and `times`.
pub fn check_input(n: usize, p0: &[f64], times: &[f64]) -> MoptResult<()> {
    if p0.len() != n {
        return Err(MoptError::config(format!(
            "the initial population has {} entries for {n} states",
            p0.len()
        )));
    }
    if p0.iter().any(|p| !(p.is_finite() && *p >= 0.0)) {
        return Err(MoptError::config(
            "the initial populations must be finite and not negative",
        ));
    }
    if times.iter().any(|t| !(t.is_finite() && *t >= 0.0)) {
        return Err(MoptError::config(
            "the output times must be finite and not negative",
        ));
    }
    Ok(())
}

/// Conservation and the most negative population of `p` against `p0`.
pub fn conservation(p: &[f64], p0: &[f64]) -> (f64, f64) {
    let total: f64 = p.iter().sum();
    let start: f64 = p0.iter().sum();
    ((total - start).abs(), p.iter().copied().fold(0.0, f64::min))
}

/// Makes a solver.
pub type SolverFactory =
    Arc<dyn Fn(&SolverSettings) -> MoptResult<Box<dyn KineticSolver>> + Send + Sync>;

/// Names of the design that are refused, with what they need.
const NOT_IMPLEMENTED: &[(&str, &str)] = &[
    ("rcmc_b", "the snapshot of Type B (V_TT = M^-1), which needs a projection onto the simplex in the pi-norm; closure b of propagate (--closure b) uses M^-1 without it"),
    ("ode", "a name that is not used: the L-stable integrator with step-size control is the solver rosenbrock (ROS2)"),
];

/// Solvers by name.
pub struct SolverRegistry {
    entries: Vec<(String, SolverFactory)>,
}

impl SolverRegistry {
    /// `propagate`, `rcmc`, `legacy_rcmc`, `expm`, `spectral`, `steady`, `rosenbrock`,
    /// `contract`, `windows`, `kmc`.
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register(
            "propagate",
            Arc::new(|s: &SolverSettings| {
                Ok(Box::new(Propagate::new(s.tolerance, s.closure)?) as Box<dyn KineticSolver>)
            }),
        );
        r.register(
            "rcmc",
            Arc::new(|s: &SolverSettings| {
                Ok(Box::new(Rcmc::new(s.time_rule)) as Box<dyn KineticSolver>)
            }),
        );
        r.register(
            "legacy_rcmc",
            Arc::new(|s: &SolverSettings| {
                Ok(Box::new(LegacyRcmc::new(s.legacy_start_known)) as Box<dyn KineticSolver>)
            }),
        );
        r.register(
            "expm",
            Arc::new(|_: &SolverSettings| Ok(Box::new(Expm) as Box<dyn KineticSolver>)),
        );
        r.register(
            "spectral",
            Arc::new(|_: &SolverSettings| Ok(Box::new(Spectral) as Box<dyn KineticSolver>)),
        );
        r.register(
            "steady",
            Arc::new(|_: &SolverSettings| Ok(Box::new(Steady) as Box<dyn KineticSolver>)),
        );
        r.register(
            "rosenbrock",
            Arc::new(|s: &SolverSettings| {
                if !(s.ode_tolerance > 0.0 && s.ode_tolerance < 1.0) {
                    return Err(MoptError::config(
                        "the tolerance of rosenbrock must be between 0 and 1",
                    ));
                }
                Ok(Box::new(Rosenbrock {
                    tolerance: s.ode_tolerance,
                }) as Box<dyn KineticSolver>)
            }),
        );
        r.register(
            "contract",
            Arc::new(|s: &SolverSettings| {
                Ok(Box::new(ContractMassAction {
                    theta: s.contract_theta,
                    tolerance: s.ode_tolerance,
                }) as Box<dyn KineticSolver>)
            }),
        );
        r.register(
            "windows",
            Arc::new(|s: &SolverSettings| {
                Ok(Box::new(Windows {
                    tolerance: s.window_tolerance,
                    contraction: s.tolerance,
                }) as Box<dyn KineticSolver>)
            }),
        );
        r.register(
            "kmc",
            Arc::new(|s: &SolverSettings| {
                Ok(Box::new(Kmc {
                    trajectories: s.kmc_trajectories,
                    max_events: s.kmc_max_events,
                }) as Box<dyn KineticSolver>)
            }),
        );
        r
    }

    /// Register (or replace) a solver.
    pub fn register(&mut self, name: &str, factory: SolverFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, factory));
    }

    /// The registered names.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// The solver `name`.
    pub fn create(&self, name: &str, s: &SolverSettings) -> MoptResult<Box<dyn KineticSolver>> {
        let key = name.to_ascii_lowercase();
        if let Some((_, f)) = self.entries.iter().find(|(k, _)| *k == key) {
            return f(s);
        }
        match NOT_IMPLEMENTED.iter().find(|(n, _)| *n == key) {
            Some((n, why)) => Err(MoptError::unsupported(format!(
                "the solver '{n}' is not implemented: {why}"
            ))),
            None => Err(MoptError::parse(format!(
                "unknown solver '{name}' (available: {})",
                self.names().join(", ")
            ))),
        }
    }
}

impl Default for SolverRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

/// `m x`.
pub fn mat_vec(m: &Mat<f64>, x: &[f64]) -> Vec<f64> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)] * x[j]).sum())
        .collect()
}

/// The members of every superstate of a contraction: each eliminated state goes to the kept
/// state its population enters first with the highest probability (`B`; the lower index on
/// ties).
pub fn members_of(c: &Contraction) -> Vec<Vec<usize>> {
    let (nt, ns) = (c.kept.len(), c.eliminated.len());
    let mut members: Vec<Vec<usize>> = c.kept.iter().map(|&t| vec![t]).collect();
    for q in 0..ns {
        let mut best = 0;
        for a in 1..nt {
            if c.b[a * ns + q] > c.b[best * ns + q] {
                best = a;
            }
        }
        if nt > 0 {
            members[best].push(c.eliminated[q]);
        }
    }
    members
}

/// The detailed-balance residual of the superstate generator `g` of closure A with the
/// weights `ln_pi` of the kept states and `1^T M` (`weights`).
pub fn superstate_balance(g: &Mat<f64>, ln_pi: &[f64], weights: &[f64]) -> f64 {
    let n = g.nrows();
    let ln_w: Vec<f64> = (0..n).map(|i| ln_pi[i] + weights[i].ln()).collect();
    let mut worst: f64 = 0.0;
    for i in 0..n {
        for k in 0..n {
            if i != k && (g[(i, k)] > 0.0 || g[(k, i)] > 0.0) {
                let r = (g[(i, k)].ln() + ln_w[k]) - (g[(k, i)].ln() + ln_w[i]);
                worst = worst.max(if r.is_finite() {
                    r.abs()
                } else {
                    f64::INFINITY
                });
            }
        }
    }
    worst
}
