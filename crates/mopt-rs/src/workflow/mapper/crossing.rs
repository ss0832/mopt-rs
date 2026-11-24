//! Crossing points between the electronic states of a run (`states=...`): every minimum of a
//! state PESi has, besides the tasks of the method on its own surface, crossing tasks for
//! every other state PESj. They find crossing points as transition states are found, by AFIR:
//! for every AFIR push of `afir_ts`, a path with the artificial force on the gradient
//! projection of the two states (`crossing_afir=gp`, the default; GP/AFIR of Harabuchi,
//! Taketsugu, and Maeda), which runs to and within the seam, or a path on a model function of
//! the two states whose minima lie near their crossing points (`model`: the seam model function
//! for an MECP, the Avoiding model function for an MECI, [`model_function_name`]). The
//! candidates of a path are its end and the local minima of the model function along it (for a
//! path on the gradient projection, of the seam model function evaluated along it).
//! Every candidate is optimized to the minimum-energy crossing point by the gradient
//! projection without the force (`opt_mesx` for states of different spin or charge, MECP;
//! `opt_meci` for two roots of one spin, MECI); from the crossing point a meta-IRC runs down
//! on each of the two states and its end is minimized. `crossing_direct` adds the optimization
//! from the minimum itself.
//! The crossing point becomes a node (`MECP0`, `MECI0`, ...) joined by a crossing edge to the
//! minima on its two sides; a minimum found so is a node of its state and is explored like any,
//! so that the network grows on every surface and between them (`docs/source/mapper_design.md`,
//! "Transition states and crossing points together").
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's mapper explores one surface).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use std::sync::Arc;

use super::afir::AfirTs;
use super::configuration::Configuration;
use super::method::{
    CrossingPath, ExplorationMethod, FoundCrossing, IrcBranch, Job, JobContext, JobOutcome,
    OptimizeRequest, PathPoint, Stage, Stationary, Task,
};
use super::network::{CrossingKind, ElectronicState, Network, Node, NodeKind};
use crate::core_infra::json::JsonValue;
use crate::error::{MoptError, MoptResult};

/// The prefix of the keys of the crossing tasks (`crossing:J`, `J` the index of the other
/// state).
pub const CROSSING_PREFIX: &str = "crossing:";

/// The kind of the crossing point of `a` and `b`: a conical intersection (MECI) of two states
/// of one charge and multiplicity, a crossing of states of different spin or charge (MECP)
/// otherwise.
pub fn crossing_kind(a: &ElectronicState, b: &ElectronicState) -> CrossingKind {
    if a.charge == b.charge && a.multiplicity == b.multiplicity {
        CrossingKind::Meci
    } else {
        CrossingKind::Mecp
    }
}

/// The index of `state` among `states`.
pub fn state_index(states: &[ElectronicState], state: &ElectronicState) -> Option<usize> {
    states.iter().position(|s| s == state)
}

/// `PES1`, `PES2`, ... for the index `i` of a state.
pub fn surface_name(i: usize) -> String {
    format!("PES{}", i + 1)
}

/// Which pairs of states have their crossing points searched (`crossing_pairs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrossingPairs {
    /// Every pair (`all`, the default): a minimum of PESi has a task for every other state.
    #[default]
    All,
    /// Neighbouring states only (`adjacent`): PESi with PESi-1 and PESi+1, n - 1 pairs of n
    /// states instead of n (n - 1) / 2.
    Adjacent,
}

impl CrossingPairs {
    /// `all` or `adjacent`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "all" => Ok(Self::All),
            "adjacent" => Ok(Self::Adjacent),
            other => Err(MoptError::parse(format!(
                "crossing_pairs is all (every pair of states, the default) or adjacent \
                 (neighbouring states only), not '{other}'"
            ))),
        }
    }

    /// Whether the crossing points of the states of indices `i` and `j` are searched.
    pub fn includes(self, i: usize, j: usize) -> bool {
        i != j && (self == Self::All || i.abs_diff(j) == 1)
    }
}

/// Which AFIR paths lead towards the crossing points of two states (`crossing_afir`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrossingAfir {
    /// The artificial force added to the gradient projection (`gp`, the default): the path
    /// ends at an approximate crossing point.
    #[default]
    Gp,
    /// The paths on the model functions (`model`): the seam model function for an MECP, the
    /// Avoiding model function for an MECI.
    Model,
    /// Both kinds of path (`both`).
    Both,
    /// No AFIR path (`off`).
    Off,
}

impl CrossingAfir {
    /// `gp`, `model`, `both` or `off` (`on`/`true` is `gp`, `false` is `off`).
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "gp" | "on" | "true" | "yes" => Ok(Self::Gp),
            "model" => Ok(Self::Model),
            "both" => Ok(Self::Both),
            "off" | "false" | "no" | "none" => Ok(Self::Off),
            other => Err(MoptError::parse(format!(
                "crossing_afir is gp (the artificial force on the gradient projection, the \
                 default), model (on the model functions), both, or off, not '{other}'"
            ))),
        }
    }

    /// The kinds of path it runs.
    pub fn paths(self) -> &'static [CrossingPath] {
        match self {
            Self::Gp => &[CrossingPath::Gp],
            Self::Model => &[CrossingPath::Model],
            Self::Both => &[CrossingPath::Gp, CrossingPath::Model],
            Self::Off => &[],
        }
    }
}

/// An AFIR push given by its two fragments (`crossing_pushes`).
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentPush {
    /// The force parameter (kJ/mol; negative: a pull).
    pub gamma: f64,
    /// The atoms of the first fragment (from 0).
    pub a: Vec<usize>,
    /// The atoms of the second fragment.
    pub b: Vec<usize>,
}

impl FragmentPush {
    /// The task of the push (the key and the parameters of an AFIR push, for
    /// [`super::afir`]).
    fn task(&self) -> Task {
        let numbers = |atoms: &[usize]| -> String {
            atoms
                .iter()
                .map(|i| (i + 1).to_string())
                .collect::<Vec<_>>()
                .join(".")
        };
        let list = |atoms: &[usize]| {
            JsonValue::Array(atoms.iter().map(|&i| JsonValue::Number(i as f64)).collect())
        };
        let mut p = BTreeMap::new();
        p.insert("gamma_kj_mol".to_string(), JsonValue::Number(self.gamma));
        p.insert("atoms_a".to_string(), list(&self.a));
        p.insert("atoms_b".to_string(), list(&self.b));
        Task {
            key: format!(
                "afir:{:+}:{}-{}",
                self.gamma,
                numbers(&self.a),
                numbers(&self.b)
            ),
            description: format!(
                "AFIR {} the fragments {} and {} ({} kJ/mol)",
                if self.gamma > 0.0 {
                    "push of"
                } else {
                    "pull of"
                },
                numbers(&self.a),
                numbers(&self.b),
                self.gamma
            ),
            parameters: JsonValue::Object(p),
        }
    }
}

/// Where the AFIR pushes of the paths towards crossing points come from.
enum Pushes {
    /// The pairs of atoms and the force of `afir_ts`.
    Pairs(Box<AfirTs>),
    /// The pushes given (`crossing_pushes`).
    Given(Vec<FragmentPush>),
}

/// A method with the crossing tasks added: the crossing tasks and the tasks of the method of a
/// minimum alternate, each run by its own.
pub struct WithCrossings {
    base: Box<dyn ExplorationMethod>,
    states: Vec<ElectronicState>,
    pairs: CrossingPairs,
    /// The AFIR pushes of the paths towards crossing points, and the kinds of path
    /// (`crossing_afir`).
    afir: Option<(Pushes, CrossingAfir)>,
    /// Only the minima of the inputs have crossing tasks (`crossing_from=inputs`).
    inputs_only: bool,
    /// The search from the minimum itself (`crossing_direct`).
    direct: bool,
    /// The states on which the method searches transition states (`ts_states`; `None`: all).
    ts_states: Option<Vec<usize>>,
    /// The Avoiding model function of an MECI is its lower adiabatic state, on which the seam
    /// is a ridge (`crossing_amf_branch=lower`).
    lower_avoiding: bool,
    /// Which extrema along an AFIR path are candidates.
    candidates: CandidateFilter,
    name: String,
}

/// The prefix of the keys of the tasks of AFIR on the gradient projection of two states
/// (`gpafir:J:<the key of the AFIR push>`).
pub const GP_AFIR_PREFIX: &str = "gpafir:";

/// The prefix of the keys of the tasks of AFIR on the model function of two states
/// (`xafir:J:<the key of the AFIR push>`).
pub const MODEL_AFIR_PREFIX: &str = "xafir:";

/// The candidates of one AFIR path that are refined: its end and the lowest local minima of
/// the model function along it, at most this many in all.
pub const MODEL_AFIR_CANDIDATES: usize = 3;

/// The model function whose minima lie near the crossing points of `kind`: the seam model
/// function for an MECP (states of different spin or charge), the Avoiding model function for an
/// MECI (two roots of one spin).
pub fn model_function_name(kind: CrossingKind) -> &'static str {
    match kind {
        CrossingKind::Mecp => "seam model function",
        CrossingKind::Meci => "Avoiding model function",
    }
}

/// The indices of the local minima of the energy along `path` (inner points not higher than
/// the point before and lower than the one after), lowest first.
pub fn local_minima(path: &[PathPoint]) -> Vec<usize> {
    let mut out: Vec<usize> = (1..path.len().saturating_sub(1))
        .filter(|&i| path[i].energy <= path[i - 1].energy && path[i].energy < path[i + 1].energy)
        .collect();
    out.sort_by(|&a, &b| path[a].energy.total_cmp(&path[b].energy));
    out
}

/// Which extrema of the function along an AFIR path are candidates of crossing points
/// (`crossing_candidate_prominence`, `crossing_candidate_distance`). On a surface that is not
/// smooth (numerical noise of the energies, a path that zigzags) the function has many small
/// extrema next to each other, which would all be refined to the same crossing point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateFilter {
    /// How far the function must rise on both sides of a minimum (fall, of a maximum) before
    /// it comes back to its value (Hartree; 0: every extremum counts).
    pub prominence: f64,
    /// The smallest root-mean-square displacement of the atoms between two candidates of one
    /// path, and between a candidate and a crossing point the job has found (Bohr; 0: none).
    pub distance: f64,
}

/// The prominence of an extremum that counts when `crossing_candidate_prominence` is not given
/// (Hartree, 0.5 kJ/mol); see `docs/source/mapper_design.md` for the measurement.
pub const DEFAULT_CANDIDATE_PROMINENCE: f64 = 0.5 * crate::domain::Units::KJ_MOL_TO_HARTREE;
/// The smallest displacement between two candidates when `crossing_candidate_distance` is not
/// given (Bohr; 0.05 Angstrom); see `docs/source/mapper_design.md` for the measurement.
pub const DEFAULT_CANDIDATE_DISTANCE: f64 = 0.05 * crate::domain::Units::ANGSTROM_TO_BOHR;

impl Default for CandidateFilter {
    fn default() -> Self {
        Self {
            prominence: DEFAULT_CANDIDATE_PROMINENCE,
            distance: DEFAULT_CANDIDATE_DISTANCE,
        }
    }
}

/// The prominence of the minimum `i` of `values`: how far the values rise on both sides
/// before a lower value (or the end of the path) is met, the smaller of the two sides.
pub fn prominence(values: &[f64], i: usize) -> f64 {
    let Some(&at) = values.get(i) else {
        return 0.0;
    };
    let rise = |side: &mut dyn Iterator<Item = &f64>| -> f64 {
        let mut top = at;
        for &v in side {
            if v < at {
                break;
            }
            top = top.max(v);
        }
        top - at
    };
    let left = rise(&mut values[..i].iter().rev());
    let right = rise(&mut values[i + 1..].iter());
    left.min(right)
}

/// The root-mean-square displacement of the atoms (of the variables, for a point of plain
/// variables) between two points of one job, without a superposition: the points of an
/// optimization are in one frame.
pub fn displacement(a: &Configuration, b: &Configuration) -> f64 {
    let (x, y) = (a.flat(), b.flat());
    if x.nrows() != y.nrows() || x.nrows() == 0 {
        return f64::INFINITY;
    }
    let parts = if a.is_variables() {
        x.nrows()
    } else {
        x.nrows() / 3
    };
    let sum: f64 = (0..x.nrows()).map(|i| (x[i] - y[i]).powi(2)).sum();
    (sum / parts.max(1) as f64).sqrt()
}

impl CandidateFilter {
    /// The extrema `extrema` of `path` (indices, the best first; minima, or maxima when
    /// `maxima`) that count: those of at least the prominence of the filter.
    pub fn prominent(&self, path: &[PathPoint], extrema: Vec<usize>, maxima: bool) -> Vec<usize> {
        if self.prominence <= 0.0 {
            return extrema;
        }
        let sign = if maxima { -1.0 } else { 1.0 };
        let values: Vec<f64> = path.iter().map(|p| sign * p.energy).collect();
        extrema
            .into_iter()
            .filter(|&i| prominence(&values, i) >= self.prominence)
            .collect()
    }

    /// Whether `point` is nearer than the distance of the filter to one of `others`.
    pub fn near<'a>(
        &self,
        point: &Configuration,
        mut others: impl Iterator<Item = &'a Configuration>,
    ) -> bool {
        self.distance > 0.0 && others.any(|o| displacement(point, o) < self.distance)
    }
}

/// `first` and `second` in turn, one of each while both last, then the rest.
fn alternate<T>(first: Vec<T>, second: Vec<T>) -> Vec<T> {
    let mut out = Vec::with_capacity(first.len() + second.len());
    let (mut a, mut b) = (first.into_iter(), second.into_iter());
    loop {
        match (a.next(), b.next()) {
            (None, None) => break,
            (x, y) => {
                out.extend(x);
                out.extend(y);
            }
        }
    }
    out
}

impl WithCrossings {
    /// `base` with the crossing points between the `pairs` of `states` (two or more).
    pub fn new(
        base: Box<dyn ExplorationMethod>,
        states: Vec<ElectronicState>,
        pairs: CrossingPairs,
    ) -> MoptResult<Self> {
        if states.len() < 2 {
            return Err(MoptError::config(
                "crossing points need two or more states (states=C,M;C,M)",
            ));
        }
        let name = format!("{}+crossings", base.name());
        Ok(Self {
            base,
            states,
            pairs,
            afir: None,
            inputs_only: false,
            direct: false,
            ts_states: None,
            lower_avoiding: false,
            candidates: CandidateFilter::default(),
            name,
        })
    }

    /// With the AFIR paths `kind` towards crossing points, for the pushes of `afir` (its pairs
    /// of atoms).
    #[must_use]
    pub fn with_afir(mut self, afir: AfirTs, kind: CrossingAfir) -> Self {
        self.afir = (kind != CrossingAfir::Off).then_some((Pushes::Pairs(Box::new(afir)), kind));
        self
    }

    /// With the AFIR paths `kind` towards crossing points for the pushes `pushes` alone
    /// (`crossing_pushes`).
    #[must_use]
    pub fn with_pushes(mut self, pushes: Vec<FragmentPush>, kind: CrossingAfir) -> Self {
        self.afir = (kind != CrossingAfir::Off).then_some((Pushes::Given(pushes), kind));
        self
    }

    /// With `only`, the minima of the inputs alone have crossing tasks (`crossing_from=inputs`):
    /// the run ends when the pushes from the inputs are done.
    #[must_use]
    pub fn with_inputs_only(mut self, only: bool) -> Self {
        self.inputs_only = only;
        self
    }

    /// With the search for a crossing point from every minimum itself.
    #[must_use]
    pub fn with_direct(mut self, direct: bool) -> Self {
        self.direct = direct;
        self
    }

    /// The method searches transition states only on the states `states` (their indices); the
    /// minima of the other states have their crossing tasks alone.
    #[must_use]
    pub fn with_ts_states(mut self, states: Option<Vec<usize>>) -> Self {
        self.ts_states = states;
        self
    }

    /// With `lower`, the Avoiding model function of an MECI is its lower adiabatic state: the
    /// seam is a ridge of it, so the crossing candidates of an AFIR path are its local maxima
    /// (lowest first) instead of its minima, and the end of the path comes after them.
    #[must_use]
    pub fn with_lower_avoiding(mut self, lower: bool) -> Self {
        self.lower_avoiding = lower;
        self
    }

    /// With `filter` deciding which extrema along an AFIR path are candidates.
    #[must_use]
    pub fn with_candidates(mut self, filter: CandidateFilter) -> Self {
        self.candidates = filter;
        self
    }

    /// The index of the other state of a task.
    fn other_state(&self, job: &Job) -> MoptResult<usize> {
        job.task
            .parameters
            .get("state")
            .and_then(JsonValue::as_f64)
            .map(|v| v as usize)
            .filter(|&j| j < self.states.len())
            .ok_or_else(|| {
                MoptError::parse(format!(
                    "crossings: the task '{}' has no state of the run",
                    job.task.key
                ))
            })
    }

    /// The crossing job from the minimum itself: the crossing point of the state of the job and
    /// the state `other` optimized from it, then followed down on each of the two.
    fn run_crossing(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let other = self.other_state(job)?;
        let here = job.state;
        if here == other {
            return Ok(JobOutcome::default());
        }
        let mut outcome = JobOutcome::default();
        let label = ctx.network.node(job.node)?.label();
        let kind = crossing_kind(&self.states[here], &self.states[other]);
        let what = format!(
            "the {} of {} and {} from {label}",
            kind.tag(),
            surface_name(here),
            surface_name(other)
        );
        if !self.refine(job, ctx, other, &job.start, &what, &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }

    /// The AFIR job towards the crossing points of the state of the job and the state `other`.
    /// The guesses of a path are its end and the local minima along it of the model function
    /// (on the gradient projection: of the seam model function of the two states, which only
    /// chooses them; on a model function, [`model_function_name`], of that function, its
    /// maxima on the lower Avoiding model function). Each is refined without the force, and
    /// every crossing point found is followed down on both states.
    fn run_afir(
        &self,
        job: &Job,
        ctx: &mut JobContext<'_>,
        path_kind: CrossingPath,
    ) -> MoptResult<JobOutcome> {
        let other = self.other_state(job)?;
        let here = job.state;
        if here == other {
            return Ok(JobOutcome::default());
        }
        let push = super::afir::push_of(&job.task)?;
        let atoms = job.start.structure().map_or(0, |s| s.n_atoms());
        if let Some(i) = push.a.iter().chain(&push.b).find(|&&i| i >= atoms) {
            return Err(MoptError::config(format!(
                "crossings: the AFIR push '{}' names atom {} of {atoms}",
                job.task.key,
                i + 1
            )));
        }
        let term = push.term()?;
        let kind = crossing_kind(&self.states[here], &self.states[other]);
        let label = ctx.network.node(job.node)?.label();
        let on = match path_kind {
            CrossingPath::Gp => "gradient projection".to_string(),
            CrossingPath::Model => model_function_name(kind).to_string(),
        };
        let path = ctx.jobs.crossing_path(
            &job.start,
            other,
            path_kind,
            &OptimizeRequest {
                label: format!(
                    "AFIR path on the {on} of {} and {}",
                    surface_name(here),
                    surface_name(other)
                ),
                stage: Stage::Biased,
                terms: vec![("AFIR_pot_0".to_string(), Arc::new(term))],
            },
            ctx.frames,
        )?;
        if path.interrupted {
            return Ok(JobOutcome::stopped());
        }
        let mut outcome = JobOutcome::default();
        let end = (path.system.clone(), "the end of the AFIR path".to_string());
        // Extrema along the path that are no candidates: too shallow, or next to another.
        let mut dropped = 0;
        let starts: Vec<(Configuration, String)> = match path_kind {
            // The path converges to a minimum of the mean AFIR function within the seam: its
            // end lies on the seam, displaced within it by the force. The guesses are that end
            // and the local minima of the seam model function of the two states along the
            // path (its energies), which are low only where the gap is small (Harabuchi,
            // Taketsugu, Maeda 2017, step c). The last point of a path that did not converge
            // is no guess.
            CrossingPath::Gp => {
                let mut starts = Vec::new();
                if path.converged {
                    starts.push(end);
                } else {
                    outcome.warnings.push(format!(
                        "the AFIR path on the {on} of {} and {} from {label} did not converge; \
                         its end is not taken as a crossing point",
                        surface_name(here),
                        surface_name(other)
                    ));
                }
                let minima = local_minima(&path.path);
                let found = minima.len();
                let minima = self.candidates.prominent(&path.path, minima, false);
                dropped += found - minima.len();
                for i in minima {
                    if starts.len() >= MODEL_AFIR_CANDIDATES {
                        break;
                    }
                    if self
                        .candidates
                        .near(&path.path[i].system, starts.iter().map(|s| &s.0))
                    {
                        dropped += 1;
                        continue;
                    }
                    starts.push((
                        path.path[i].system.clone(),
                        format!(
                            "the minimum of the seam model function at point {i} of the AFIR \
                             path on the {on}"
                        ),
                    ));
                }
                starts
            }
            // On the seam model function and the upper Avoiding model function the seam is a
            // valley: the end of the path, then the lowest local minima of the model function
            // along it. On the lower Avoiding model function it is a ridge, which a path
            // crosses at a maximum: the lowest local maxima, then the end. The other extrema
            // are artefacts of the penalty or the coupling of the model function, no
            // transition states of either state, and are not followed.
            CrossingPath::Model => {
                let ridge = kind == CrossingKind::Meci && self.lower_avoiding;
                let extrema = if ridge {
                    let mut m = super::afir::local_maxima(&path.path);
                    m.sort_by(|&a, &b| path.path[a].energy.total_cmp(&path.path[b].energy));
                    m
                } else {
                    local_minima(&path.path)
                };
                let found = extrema.len();
                let extrema = self.candidates.prominent(&path.path, extrema, ridge);
                dropped += found - extrema.len();
                let mut starts = Vec::new();
                if !ridge {
                    starts.push(end.clone());
                }
                for i in extrema {
                    if starts.len() + usize::from(ridge) >= MODEL_AFIR_CANDIDATES {
                        break;
                    }
                    if self
                        .candidates
                        .near(&path.path[i].system, starts.iter().map(|s| &s.0))
                    {
                        dropped += 1;
                        continue;
                    }
                    starts.push((
                        path.path[i].system.clone(),
                        format!(
                            "the {} of the {on} at point {i} of the AFIR path",
                            if ridge { "maximum" } else { "minimum" }
                        ),
                    ));
                }
                if ridge {
                    starts.push(end);
                }
                starts
            }
        };
        if dropped > 0 {
            ctx.jobs.note(&format!(
                "{dropped} extremum(s) of the function along the AFIR path are no candidates: \
                 less prominent than {:.3e} Hartree, or within {:.3} Bohr of another candidate.",
                self.candidates.prominence, self.candidates.distance
            ));
        }
        for (start, origin) in starts {
            // A candidate next to a crossing point this job has found would be refined to
            // it again.
            if self
                .candidates
                .near(&start, outcome.crossings.iter().map(|c| &c.point.system))
            {
                ctx.jobs.note(&format!(
                    "The candidate from {origin} lies within {:.3} Bohr of a crossing point \
                     found by this job; it is not refined.",
                    self.candidates.distance
                ));
                continue;
            }
            let what = format!(
                "the {} of {} and {} from {origin} from {label}",
                kind.tag(),
                surface_name(here),
                surface_name(other)
            );
            if !self.refine(job, ctx, other, &start, &what, &mut outcome)? {
                return Ok(JobOutcome::stopped());
            }
        }
        Ok(outcome)
    }

    /// The crossing point of the state of the job and `other` optimized from `start`, and from
    /// a converged one, on each of the two states, the meta-IRC (the path of steepest descent
    /// from the crossing point, which is no stationary point of either state) and the
    /// minimization of its end, pushed to `outcome`; `false` when the run was stopped.
    fn refine(
        &self,
        job: &Job,
        ctx: &mut JobContext<'_>,
        other: usize,
        start: &Configuration,
        what: &str,
        outcome: &mut JobOutcome,
    ) -> MoptResult<bool> {
        refine_crossing(&self.states, job, ctx, other, start, what, outcome)
    }
}

/// The crossing point of the state of `job` and the state `other` (indices among `states`,
/// the states of the run) optimized from `start`, and from a converged one, on each of the
/// two states, the meta-IRC and the minimization of its end, pushed to `outcome`; `false`
/// when the run was stopped. What every crossing job does with a guess, and what a crossing
/// point of another network is drawn again with ([`super::remap`]).
pub fn refine_crossing(
    states: &[ElectronicState],
    job: &Job,
    ctx: &mut JobContext<'_>,
    other: usize,
    start: &Configuration,
    what: &str,
    outcome: &mut JobOutcome,
) -> MoptResult<bool> {
    let here = job.state;
    let (Some(first), Some(second)) = (states.get(here), states.get(other)) else {
        return Err(MoptError::config(format!(
            "{what}: the states {} and {} of a run of {} states",
            here + 1,
            other + 1,
            states.len()
        )));
    };
    let kind = crossing_kind(first, second);
    let found = match ctx.jobs.crossing(start, other, what, ctx.frames) {
        Ok(found) => found,
        // A candidate the search cannot go on from (two states whose gradients are the same
        // there, which the gradient projection refuses) is left out; the job goes on with its
        // other candidates.
        Err(MoptError::Convergence(why)) => {
            outcome.warnings.push(format!(
                "{what} was left out: the search for a crossing point stopped ({why})"
            ));
            return Ok(true);
        }
        Err(e) => return Err(e),
    };
    if found.interrupted {
        return Ok(false);
    }
    if let Some(steps) = found.parallel {
        outcome.warnings.push(format!(
            "{what} was ended where the two states run parallel: for {steps} iterations \
                 their gap was closed ({:.3e} Hartree at the end) and the difference of their \
                 gradients small ({} Hartree/Bohr at the end); a region where the states are \
                 degenerate, no crossing point (crossing_parallel_stop)",
            found.gap.abs(),
            found
                .gradient_difference
                .map_or_else(|| "?".to_string(), |d| format!("{d:.3e}"))
        ));
        return Ok(true);
    }
    if !found.converged {
        // Both states were computed at the end of the search: where they are further apart
        // than a crossing point allows, the end is no crossing point, and nothing runs
        // down from it.
        outcome.warnings.push(if found.seam_check_failed {
            format!(
                "{what} is no crossing point: the check of the branching plane (gradients at \
                 displaced structures) still found the conical intersection not stationary on \
                 the seam after its last correction of the plane (seam_check_failed); no \
                 meta-IRC from it"
            )
        } else if found.stationary {
            format!(
                "{what} converged where the two states are {:.3e} Hartree ({:.2} kcal/mol) \
                     apart, more than crossing_gap ({:.3e} Hartree): no crossing point, and no \
                     meta-IRC from it",
                found.gap.abs(),
                found.gap.abs() * crate::domain::Units::HARTREE_TO_KCAL_MOL,
                found.gap_limit
            )
        } else {
            format!("{what} did not converge to a crossing point")
        });
        return Ok(true);
    }
    // The states in the order of the run, and the way down on each; the gap in that order.
    let (pair, gap) = if here < other {
        ([here, other], found.gap)
    } else {
        ([other, here], -found.gap)
    };
    // A point that is a saddle point on the seam is kept, but it is no minimum-energy crossing
    // point, and nothing descends along the seam from it.
    if let Some(seam) = found.seam.as_ref().filter(|s| s.is_saddle()) {
        outcome.warnings.push(format!(
            "{what} converged to a {} by the analysis within the seam: it is kept, but it is no \
             minimum-energy crossing point (nothing descends along the seam from it)",
            seam.describe()
        ));
    }
    // A check that was asked for and could not be done is said (one switched off is not).
    if let Some(super::network::SeamCheck::NotChecked(why)) = &found.seam {
        if !why.starts_with("switched off") {
            outcome.warnings.push(format!(
                "{what}: the check within the seam was not done ({why})"
            ));
        }
    }
    let mut ends: [Option<Stationary>; 2] = [None, None];
    let mut paths: [Option<IrcBranch>; 2] = [None, None];
    let mut settles: [Vec<PathPoint>; 2] = [Vec::new(), Vec::new()];
    for (side, &state) in pair.iter().enumerate() {
        let surface = surface_name(state);
        let down = format!("the meta-IRC of {what} on {surface}");
        let path = ctx
            .jobs
            .meta_irc_in(&found.point, state, &down, ctx.frames)?;
        if path.interrupted {
            return Ok(false);
        }
        let from = match path.branch {
            Some(mut branch) => {
                branch.side.name.clone_from(&surface);
                let last = branch.last.clone();
                paths[side] = Some(branch);
                last
            }
            None => {
                outcome.warnings.push(format!(
                    "{down} gave no path; its end is minimized from the crossing point"
                ));
                found.point.clone()
            }
        };
        let settle = format!("the end of {down}");
        let end = ctx.jobs.minimize_in(&from, state, &settle, ctx.frames)?;
        if end.interrupted {
            return Ok(false);
        }
        if end.converged {
            settles[side] = end.path;
            ends[side] = Some(Stationary {
                index_wavenumbers: end.index_wavenumbers,
                system: end.system,
                energy: end.energy,
                wavenumbers: end.wavenumbers,
                dissociation: None,
            });
        } else {
            outcome
                .warnings
                .push(format!("{settle} did not converge to a minimum"));
        }
    }
    outcome.crossings.push(FoundCrossing {
        point: Stationary {
            index_wavenumbers: None,
            system: found.point,
            energy: found.energy,
            wavenumbers: None,
            dissociation: None,
        },
        kind,
        states: pair,
        gap,
        gradient_difference: found.gradient_difference,
        seam: found.seam.clone(),
        ends,
        paths,
        settles,
    });
    Ok(true)
}

impl ExplorationMethod for WithCrossings {
    fn name(&self) -> &str {
        &self.name
    }

    fn tasks_follow_network(&self) -> bool {
        self.base.tasks_follow_network()
    }

    /// For a minimum of state PESi: its crossing tasks (for every other state PESj of the
    /// pairs searched, `crossing:J` with `crossing_direct`, and for every AFIR push
    /// `gpafir:J:...` and `xafir:J:...` as `crossing_afir` says) and the tasks of the method
    /// (where transition states are searched on PESi, `ts_states`), alternating: a crossing
    /// task, a task of the method, a crossing task, ..., so that neither kind waits for the
    /// other to run out.
    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        let mut crossing = Vec::new();
        let here = if node.kind == NodeKind::Minimum {
            state_index(&self.states, &node.state)
        } else {
            None
        };
        // With crossing_from=inputs, a minimum a job found has no crossing tasks.
        let crossed = !self.inputs_only || node.origin.job.is_none();
        if let Some(here) = here.filter(|_| crossed) {
            let others: Vec<usize> = (0..self.states.len())
                .filter(|j| self.pairs.includes(here, *j))
                .collect();
            let tag = |j: usize| crossing_kind(&self.states[here], &self.states[j]).tag();
            if self.direct {
                for &j in &others {
                    let mut p = BTreeMap::new();
                    p.insert("state".to_string(), JsonValue::Number(j as f64));
                    crossing.push(Task {
                        key: format!("{CROSSING_PREFIX}{j}"),
                        description: format!(
                            "the {} with {} from {}",
                            tag(j),
                            surface_name(j),
                            node.label()
                        ),
                        parameters: JsonValue::Object(p),
                    });
                }
            }
            if let Some((pushes, kind)) = &self.afir {
                let pushes = match pushes {
                    Pushes::Pairs(afir) => afir.tasks(node, network)?,
                    Pushes::Given(list) => list.iter().map(FragmentPush::task).collect(),
                };
                for push in pushes {
                    for &j in &others {
                        for path in kind.paths() {
                            let (prefix, on) = match path {
                                CrossingPath::Gp => {
                                    (GP_AFIR_PREFIX, "gradient projection".to_string())
                                }
                                CrossingPath::Model => (
                                    MODEL_AFIR_PREFIX,
                                    model_function_name(crossing_kind(
                                        &self.states[here],
                                        &self.states[j],
                                    ))
                                    .to_string(),
                                ),
                            };
                            let mut p = match &push.parameters {
                                JsonValue::Object(m) => m.clone(),
                                _ => BTreeMap::new(),
                            };
                            p.insert("state".to_string(), JsonValue::Number(j as f64));
                            crossing.push(Task {
                                key: format!("{prefix}{j}:{}", push.key),
                                description: format!(
                                    "{} on the {on} with {} (the {})",
                                    push.description,
                                    surface_name(j),
                                    tag(j)
                                ),
                                parameters: JsonValue::Object(p),
                            });
                        }
                    }
                }
            }
        }
        let searched = match (&self.ts_states, here) {
            (Some(states), Some(here)) => states.contains(&here),
            _ => true,
        };
        let own = if searched {
            self.base.tasks(node, network)?
        } else {
            Vec::new()
        };
        Ok(alternate(crossing, own))
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        if job.task.key.starts_with(CROSSING_PREFIX) {
            self.run_crossing(job, ctx)
        } else if job.task.key.starts_with(GP_AFIR_PREFIX) {
            self.run_afir(job, ctx, CrossingPath::Gp)
        } else if job.task.key.starts_with(MODEL_AFIR_PREFIX) {
            self.run_afir(job, ctx, CrossingPath::Model)
        } else {
            self.base.run(job, ctx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_alternate_and_the_rest_follows() {
        assert_eq!(alternate(vec![1, 3, 5, 7], vec![2, 4]), [1, 2, 3, 4, 5, 7]);
        assert_eq!(alternate(vec![1], vec![2, 4, 6]), [1, 2, 4, 6]);
        assert_eq!(alternate(Vec::<i32>::new(), vec![2]), [2]);
    }

    #[test]
    fn the_kinds_of_afir_path() {
        assert_eq!(
            CrossingAfir::parse("gp").unwrap().paths(),
            [CrossingPath::Gp]
        );
        assert_eq!(CrossingAfir::parse("both").unwrap().paths().len(), 2);
        assert!(CrossingAfir::parse("off").unwrap().paths().is_empty());
        assert!(CrossingAfir::parse("seam").is_err());
        assert!(CrossingPairs::Adjacent.includes(1, 2) && !CrossingPairs::Adjacent.includes(0, 2));
    }
}
