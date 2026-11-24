//! The mapper method `nanoreactor`: molecular dynamics of a minimum in the boundary potential
//! of the ab initio nanoreactor ([`crate::pes::bias::NanoReactor`]: a wide sphere and, for part
//! of every period, a narrow one that presses the atoms together), the reactions it shows,
//! and for each of them a transition state between its two minima.
//!
//! A minimum has `runs` tasks, `nanoreactor:1`, `nanoreactor:2`, ..., trajectories that differ
//! in the seed of their starting velocities and thermostat (from the seed of the run, the
//! number of the node, and the number of the trajectory, so that a restart runs the same).
//! Along a trajectory, the structure has left the basin it was in when
//!
//! - `detect=bonds`: the bonds (within 1.2 times the sums of the covalent radii) of `persist`
//!   frames in a row differ from those of the minimum of the basin; the first of them is
//!   optimized to the minimum of the new basin (a structure that falls back to the bonds of the
//!   old one is no reaction);
//! - `detect=quench`: every `quench_every`-th frame optimized to a minimum is another minimum
//!   than that of the basin (the comparison of structures of the run);
//!
//! and `auto` takes `bonds` where the minimum has bonds and `quench` where it has none (atomic
//! clusters). Each reaction is the pair of minima, the old and the new basin, with the frames
//! of the trajectory around it; the first `events` of a trajectory are kept. The path that the
//! dynamics took is relaxed: the old minimum, those frames, and the new minimum are the first
//! path of the two-point method `path` (of the registry of two-point methods, the chain
//! relaxation `neb` by default, as in step 2 of AutoTS; its settings `path_KEY=VALUE`), as Wang,
//! McGibbon, Pande, and Martinez refine the reactive events of the nanoreactor (J. Chem. Theory
//! Comput. 2016, 12, 638). With `segment=off` the method starts from the two minima alone.
//! Every maximum of the relaxed path is a guess (a reaction of several steps has several);
//! each is optimized, and the IRC of each new transition state runs with its two ends
//! optimized to minima ([`super::follow`]).
//!
//! Replica exchange (`replicas`, off with 1): a trajectory runs as that many replicas on a
//! ladder, geometric between the temperature of the thermostat and `replica_top` (twice it by
//! default) with `exchange=temperature`, or between the force constant `k2` of the narrow
//! sphere and `replica_top` times it (2 by default) with `exchange=walls` at one temperature.
//! The replicas run in segments of `exchange_every` fs, each continuing from the structure and
//! the velocities where its last segment ended, the clock of the walls going on; after every
//! segment neighbouring rungs, the even pairs and the odd pairs by turns, exchange with the
//! probability of Metropolis: `min(1, exp((1/kT_i - 1/kT_j) (E_i - E_j)))` of the energies
//! with the walls for temperatures (the velocities scaled to the new temperature),
//! `min(1, exp(-(U_i(x_j) + U_j(x_i) - U_i(x_i) - U_j(x_j)) / kT))` of the walls alone for
//! walls (the surface is the same and cancels). Each replica's own continuous trajectory is
//! searched for reactions as one without replicas is. Y. Sugita, Y. Okamoto, Chem. Phys.
//! Lett. 1999, 314, 141 (replica exchange molecular dynamics); Y. Sugita, A. Kitao, Y.
//! Okamoto, J. Chem. Phys. 2000, 113, 6042 (replicas of different Hamiltonians).
//!
//! Defaults (measured, `docs/source/mapper.md`): the wide phase is three quarters of a period
//! as in the paper (1.5 of 2.0 ps), and the force constant of the wide sphere is the paper's
//! (1.0 kcal/mol/Angstrom^2 per amu). The radii come from the size of the minimum: the wide
//! sphere lies 2 Angstrom outside its farthest atom, the narrow one at 0.45 of that atom's
//! distance. The narrow sphere presses with 5 kcal/mol/Angstrom^2 per amu. The period is
//! 250 fs, and the run is 4000 steps of 0.5 fs at 2000 K.
//!
//! The paper's narrow sphere (0.5 kcal/mol/Angstrom^2 per amu, at 8 of 14 Angstrom) did not
//! hold three hot acetylene molecules: they never came closer than 2.4 Angstrom, and no
//! reaction happened. The paper's spheres and its period of 2 ps are for mixtures of many
//! molecules. All of these are settings (`r1`, `r2` in Angstrom, `k1`, `k2`, `period` in fs,
//! `wide` as the share of the period).
//!
//! L.-P. Wang, A. Titov, R. McGibbon, F. Liu, V. S. Pande, T. J. Martinez, Nat. Chem. 2014,
//! 6, 1044.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's nanoreactor potential is the bias
//!  `-nrp` of its single runs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use super::afir::mix;
use super::configuration::Configuration;
use super::follow::{follow_guesses, Guess};
use super::method::{
    DynamicsRequest, ExplorationMethod, Job, JobContext, JobOutcome, OptimizeRequest,
    PairSearchRequest, Stage, Stationary, Task,
};
use super::network::{Network, Node, NodeKind};
use crate::core_infra::json::JsonValue;
use crate::core_infra::rng::Rng;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::pes::bias::{AdBias, NanoReactor};
use crate::pes::term::PesTerm;
use crate::workflow::explore::identity::{BondGraph, DEFAULT_BOND_SCALE};
use crate::workflow::pair_search::PairMethodRegistry;

/// How a trajectory is seen to leave a basin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detection {
    /// Bonds where the minimum has any, quenches otherwise.
    Auto,
    /// The bonds of the frames.
    Bonds,
    /// Optimizations of frames to minima.
    Quench,
}

/// What the replicas of a trajectory differ in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exchange {
    /// The temperature of the thermostat.
    Temperature,
    /// The force constant of the narrow sphere.
    Walls,
}

/// The settings of the method (`method_KEY=VALUE`).
#[derive(Debug, Clone, PartialEq)]
pub struct NanoreactorSettings {
    /// Trajectories per minimum.
    pub runs: usize,
    /// Steps of a trajectory.
    pub steps: usize,
    /// Time step (fs).
    pub time_step_fs: f64,
    /// Temperature of the thermostat (K).
    pub temperature: f64,
    /// Every how many steps a frame is kept.
    pub frame_every: usize,
    /// Radius of the wide sphere (Angstrom; `None`: 2 Angstrom outside the farthest atom).
    pub r1: Option<f64>,
    /// Radius of the narrow sphere (Angstrom; `None`: 0.45 of the distance of the farthest
    /// atom).
    pub r2: Option<f64>,
    /// Force constant of the wide sphere (kcal/mol/Angstrom^2 per amu).
    pub k1: f64,
    /// Force constant of the narrow sphere (kcal/mol/Angstrom^2 per amu).
    pub k2: f64,
    /// Period of the walls (fs).
    pub period_fs: f64,
    /// Share of the period the wide sphere holds.
    pub wide: f64,
    /// How a reaction is seen.
    pub detect: Detection,
    /// Frames in a row whose bonds must differ (`bonds`).
    pub persist: usize,
    /// Every how many frames one is optimized (`quench`).
    pub quench_every: usize,
    /// Reactions kept per trajectory.
    pub events: usize,
    /// The two-point method between the minima of a reaction.
    pub path: String,
    /// Its settings.
    pub path_settings: Vec<(String, String)>,
    /// Hand the method the frames of the trajectory around the reaction as the first path
    /// between the two minima (a method that takes a first path: `neb`, `dmf`); otherwise the
    /// two minima alone.
    pub segment: bool,
    /// Frames kept before the first frame of the new basin and after it (`bonds`; with
    /// `quench` the frames since the last quench that was still the old basin).
    pub segment_frames: usize,
    /// Replicas of a trajectory on a ladder (1: no replica exchange).
    pub replicas: usize,
    /// What the replicas differ in.
    pub exchange: Exchange,
    /// The top of the ladder: the highest temperature (K), or the factor of `k2` of the
    /// strongest narrow sphere (`None`: twice the temperature, or 2).
    pub replica_top: Option<f64>,
    /// The time between two exchanges (fs).
    pub exchange_every_fs: f64,
    /// Seed of the run.
    pub seed: u64,
}

impl Default for NanoreactorSettings {
    fn default() -> Self {
        Self {
            runs: 2,
            steps: 4000,
            time_step_fs: 0.5,
            temperature: 2000.0,
            frame_every: 10,
            r1: None,
            r2: None,
            k1: 1.0,
            k2: 5.0,
            period_fs: 250.0,
            wide: 0.75,
            detect: Detection::Auto,
            persist: 3,
            quench_every: 10,
            events: 3,
            path: "neb".to_string(),
            path_settings: Vec::new(),
            segment: true,
            segment_frames: 4,
            replicas: 1,
            exchange: Exchange::Temperature,
            replica_top: None,
            exchange_every_fs: 100.0,
            seed: 42,
        }
    }
}

/// Without `r2`, the narrow sphere lies at this share of the distance of the farthest atom
/// from the centre (measured: three acetylene molecules with GFN0-xTB, `docs/source/mapper.md`).
pub const NARROW_SHARE: f64 = 0.45;

/// The keys of the method with their meaning.
pub const KEYS: &[(&str, &str)] = &[
    ("runs", "trajectories per minimum (2)"),
    ("steps", "steps of a trajectory (4000)"),
    ("time_step", "time step in fs (0.5)"),
    ("temperature", "temperature of the thermostat in K (2000)"),
    ("frame_every", "every how many steps a frame is kept (10)"),
    (
        "r1",
        "radius of the wide sphere in Angstrom (2 outside the farthest atom)",
    ),
    (
        "r2",
        "radius of the narrow sphere in Angstrom (0.45 of the farthest atom)",
    ),
    (
        "k1",
        "force constant of the wide sphere, kcal/mol/Angstrom^2 per amu (1.0)",
    ),
    (
        "k2",
        "force constant of the narrow sphere, kcal/mol/Angstrom^2 per amu (5)",
    ),
    ("period", "period of the walls in fs (250)"),
    ("wide", "share of the period of the wide sphere (0.75)"),
    ("detect", "auto, bonds, or quench (auto)"),
    ("persist", "frames in a row with other bonds (3)"),
    (
        "quench_every",
        "every how many frames one is optimized, quench (10)",
    ),
    ("events", "reactions kept per trajectory (3)"),
    (
        "path",
        "two-point method between the minima of a reaction (neb)",
    ),
    ("path_KEY", "a setting of that method"),
    (
        "segment",
        "on: the frames around the reaction are the first path of the method (on)",
    ),
    (
        "segment_frames",
        "frames before and after the first frame of the new basin in that path (4)",
    ),
    (
        "replicas",
        "replicas of a trajectory exchanged along a ladder (1: no replica exchange)",
    ),
    (
        "exchange",
        "what the replicas differ in: temperature, or walls (the narrow sphere) (temperature)",
    ),
    (
        "replica_top",
        "the top of the ladder: the highest temperature in K, or the factor of k2 (twice the \
         temperature, 2)",
    ),
    ("exchange_every", "time between two exchanges in fs (100)"),
];

fn number(key: &str, v: &str) -> MoptResult<f64> {
    v.parse::<f64>()
        .ok()
        .filter(|x| x.is_finite())
        .ok_or_else(|| MoptError::parse(format!("method_{key}: '{v}' is not a number")))
}

fn whole(key: &str, v: &str) -> MoptResult<usize> {
    v.parse::<usize>()
        .map_err(|_| MoptError::parse(format!("method_{key}: '{v}' is not a whole number")))
}

impl NanoreactorSettings {
    /// From the `method_KEY=VALUE` settings `extra` (without the prefix) and the seed of the
    /// run.
    pub fn parse(extra: &BTreeMap<String, String>, seed: u64) -> MoptResult<Self> {
        let mut s = Self {
            seed,
            ..Self::default()
        };
        for (k, v) in extra {
            match k.as_str() {
                "runs" => s.runs = whole(k, v)?,
                "steps" => s.steps = whole(k, v)?,
                "time_step" => s.time_step_fs = number(k, v)?,
                "temperature" => s.temperature = number(k, v)?,
                "frame_every" => s.frame_every = whole(k, v)?,
                "r1" => s.r1 = Some(number(k, v)?),
                "r2" => s.r2 = Some(number(k, v)?),
                "k1" => s.k1 = number(k, v)?,
                "k2" => s.k2 = number(k, v)?,
                "period" => s.period_fs = number(k, v)?,
                "wide" => s.wide = number(k, v)?,
                "detect" => {
                    s.detect = match v.to_ascii_lowercase().as_str() {
                        "auto" => Detection::Auto,
                        "bonds" => Detection::Bonds,
                        "quench" => Detection::Quench,
                        _ => {
                            return Err(MoptError::parse(format!(
                                "method_detect: auto, bonds, or quench, got '{v}'"
                            )))
                        }
                    }
                }
                "persist" => s.persist = whole(k, v)?,
                "quench_every" => s.quench_every = whole(k, v)?,
                "events" => s.events = whole(k, v)?,
                "path" => s.path = v.clone(),
                "segment" => {
                    s.segment = match v.to_ascii_lowercase().as_str() {
                        "on" | "true" | "yes" | "1" => true,
                        "off" | "false" | "no" | "0" => false,
                        _ => {
                            return Err(MoptError::parse(format!(
                                "method_segment: on or off, got '{v}'"
                            )))
                        }
                    }
                }
                "segment_frames" => s.segment_frames = whole(k, v)?,
                "replicas" => s.replicas = whole(k, v)?,
                "exchange" => {
                    s.exchange = match v.to_ascii_lowercase().as_str() {
                        "temperature" | "temperatures" => Exchange::Temperature,
                        "walls" | "wall" => Exchange::Walls,
                        _ => {
                            return Err(MoptError::parse(format!(
                                "method_exchange: temperature or walls, got '{v}'"
                            )))
                        }
                    }
                }
                "replica_top" => s.replica_top = Some(number(k, v)?),
                "exchange_every" => s.exchange_every_fs = number(k, v)?,
                _ => match k.strip_prefix("path_") {
                    Some(key) if !key.is_empty() => {
                        s.path_settings.push((key.to_string(), v.clone()));
                    }
                    _ => {
                        return Err(MoptError::parse(format!(
                            "the mapper method nanoreactor: unknown setting method_{k} (keys: {})",
                            KEYS.iter()
                                .map(|(k, _)| format!("method_{k}"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )))
                    }
                },
            }
        }
        s.check()?;
        Ok(s)
    }

    /// Refuse settings that cannot run.
    pub fn check(&self) -> MoptResult<()> {
        let positive = |name: &str, x: f64| {
            if x.is_finite() && x > 0.0 {
                Ok(())
            } else {
                Err(MoptError::parse(format!(
                    "method_{name} must be positive (got {x})"
                )))
            }
        };
        if self.runs == 0 || self.steps == 0 || self.frame_every == 0 || self.events == 0 {
            return Err(MoptError::parse(
                "method_runs, method_steps, method_frame_every, and method_events must be at \
                 least 1",
            ));
        }
        if self.persist == 0 || self.quench_every == 0 {
            return Err(MoptError::parse(
                "method_persist and method_quench_every must be at least 1",
            ));
        }
        positive("time_step", self.time_step_fs)?;
        positive("temperature", self.temperature)?;
        positive("exchange_every", self.exchange_every_fs)?;
        if self.replicas == 0 {
            return Err(MoptError::parse("method_replicas must be at least 1"));
        }
        if let Some(top) = self.replica_top {
            let bottom = match self.exchange {
                Exchange::Temperature => self.temperature,
                Exchange::Walls => 1.0,
            };
            if !(top.is_finite() && top > bottom) {
                return Err(MoptError::parse(format!(
                    "method_replica_top must be above the bottom of the ladder ({bottom}; got \
                     {top})"
                )));
            }
        }
        positive("period", self.period_fs)?;
        positive("k1", self.k1)?;
        positive("k2", self.k2)?;
        if let Some(r) = self.r1 {
            positive("r1", r)?;
        }
        if let Some(r) = self.r2 {
            positive("r2", r)?;
        }
        if !(self.wide > 0.0 && self.wide < 1.0) {
            return Err(MoptError::parse(format!(
                "method_wide is a share of the period between 0 and 1 (got {})",
                self.wide
            )));
        }
        let method = PairMethodRegistry::default().get(&self.path)?;
        let ends = method.ends();
        if ends == crate::workflow::pair_search::Ends::One {
            return Err(MoptError::parse(format!(
                "method_path: {} starts from one structure; a reaction of the nanoreactor has \
                 two minima (a two-point method: neb, dmf, dsafir, gsm, ...)",
                method.name()
            )));
        }
        if self.segment && ends != crate::workflow::pair_search::Ends::Path {
            return Err(MoptError::parse(format!(
                "method_path: {} takes the two minima alone, not the frames of the trajectory \
                 as its first path (method_segment=on, the default): use neb or dmf, or \
                 method_segment=off",
                method.name()
            )));
        }
        let known = method.keys();
        for (k, _) in &self.path_settings {
            if !known.iter().any(|(key, _)| key == k) {
                return Err(MoptError::parse(format!(
                    "method_path_{k}: {} has no setting {k} (its keys: {})",
                    method.name(),
                    known.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", ")
                )));
            }
        }
        Ok(())
    }
}

/// The method `nanoreactor`.
#[derive(Debug, Clone)]
pub struct Nanoreactor {
    settings: NanoreactorSettings,
}

/// The centre of the atoms and the distance of the farthest one from it (Bohr).
fn centre_and_radius(sys: &MolecularSystem) -> ([f64; 3], f64) {
    let x = sys.coords_flat_bohr();
    let n = sys.n_atoms().max(1) as f64;
    let c: [f64; 3] =
        std::array::from_fn(|k| (0..sys.n_atoms()).map(|i| x[3 * i + k]).sum::<f64>() / n);
    let r = (0..sys.n_atoms())
        .map(|i| {
            (0..3)
                .map(|k| (x[3 * i + k] - c[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    (c, r)
}

/// One replica of a trajectory with replica exchange: where it is, on which rung, and the
/// frames of its own continuous trajectory.
struct Walker {
    point: Configuration,
    velocities: Option<faer::Col<f64>>,
    rung: usize,
    frames: Vec<MolecularSystem>,
    alive: bool,
    /// The energy with the walls where its last segment ended.
    energy: f64,
}

/// A bias potential whose clock starts at `offset` (atomic units): the walls of a segment of a
/// trajectory that goes on.
struct Clocked {
    inner: Arc<dyn PesTerm>,
    offset: f64,
}

impl PesTerm for Clocked {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        ctx: &crate::pes::term::TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<crate::domain::EvalOutput> {
        let shifted = crate::pes::term::TermContext {
            time_au: Some(ctx.time_au.unwrap_or(ctx.iteration as f64) + self.offset),
            ..*ctx
        };
        self.inner.evaluate(sys, &shifted, need_hessian)
    }

    fn atoms(&self) -> Option<Vec<crate::domain::AtomId>> {
        self.inner.atoms()
    }

    /// Those of the wrapped term: only its clock is shifted.
    fn frame(&self) -> crate::pes::term::Frame {
        self.inner.frame()
    }

    fn history(&self) -> crate::pes::term::History {
        self.inner.history()
    }

    fn fork(&self) -> Option<Arc<dyn PesTerm>> {
        let inner = self.inner.fork()?;
        Some(Arc::new(Self {
            inner,
            offset: self.offset,
        }))
    }

    fn analytic_hessian(&self) -> bool {
        self.inner.analytic_hessian()
    }

    fn analytic_stress(&self) -> bool {
        self.inner.analytic_stress()
    }
}

/// One reaction of a trajectory: the minima of the old and the new basin, and the frames of
/// the trajectory around it, in their order.
struct Reaction {
    from: Stationary,
    to: Stationary,
    frame: usize,
    segment: Vec<MolecularSystem>,
}

impl Nanoreactor {
    /// The registered name.
    pub const NAME: &'static str = "nanoreactor";

    /// The method with `settings`.
    pub fn new(settings: NanoreactorSettings) -> MoptResult<Self> {
        settings.check()?;
        Ok(Self { settings })
    }

    /// The boundary potential of the trajectories of `start`.
    pub fn walls(&self, start: &MolecularSystem) -> NanoReactor {
        let s = &self.settings;
        let (centre, radius) = centre_and_radius(start);
        let a = Units::ANGSTROM_TO_BOHR;
        let r1 = s.r1.map_or(radius + 2.0 * a, |r| r * a);
        let r2 = s.r2.map_or(NARROW_SHARE * radius, |r| r * a).min(r1);
        // kcal/mol/Angstrom^2 per amu to Hartree/Bohr^2 per amu.
        let k = |k: f64| k * Units::KCAL_MOL_TO_HARTREE / (a * a);
        let period = s.period_fs * Units::FS_TO_AU_TIME;
        NanoReactor {
            atoms: start.atoms().zip(start.masses_amu()).collect(),
            center: centre,
            r1,
            k1: k(s.k1),
            r2,
            k2: k(s.k2),
            tau: s.wide * period,
            period,
        }
    }

    /// The rungs of the ladder of the replicas: the temperature of each and the factor of
    /// the force constant of its narrow sphere, geometric from the bottom to the top.
    pub fn ladder(&self) -> Vec<(f64, f64)> {
        let s = &self.settings;
        let m = s.replicas.max(1);
        let rung = |k: usize, bottom: f64, top: f64| {
            if m == 1 {
                bottom
            } else {
                bottom * (top / bottom).powf(k as f64 / (m - 1) as f64)
            }
        };
        (0..m)
            .map(|k| match s.exchange {
                Exchange::Temperature => (
                    rung(
                        k,
                        s.temperature,
                        s.replica_top.unwrap_or(2.0 * s.temperature),
                    ),
                    1.0,
                ),
                Exchange::Walls => (s.temperature, rung(k, 1.0, s.replica_top.unwrap_or(2.0))),
            })
            .collect()
    }

    /// The trajectories of the replicas of trajectory `run` from the minimum of `job`, one per
    /// replica in its own continuous order (`None`: the run was stopped).
    fn replica_trajectories(
        &self,
        job: &Job,
        ctx: &mut JobContext<'_>,
        start: &MolecularSystem,
        run: u64,
        outcome: &mut JobOutcome,
    ) -> MoptResult<Option<Vec<Vec<MolecularSystem>>>> {
        let s = &self.settings;
        let ladder = self.ladder();
        let m = ladder.len();
        let dt = s.time_step_fs * Units::FS_TO_AU_TIME;
        let per = ((s.exchange_every_fs / s.time_step_fs).round() as usize).max(1);
        let base = self.walls(start);
        let wall_of = |rung: usize| {
            let mut w = base.clone();
            w.k2 *= ladder[rung].1;
            AdBias::new(w)
        };
        let seed = mix(mix(s.seed, job.node as u64), run);
        let mut walkers: Vec<Walker> = (0..m)
            .map(|r| Walker {
                point: job.start.clone(),
                velocities: None,
                rung: r,
                frames: Vec::new(),
                alive: true,
                energy: f64::NAN,
            })
            .collect();
        let (mut tried, mut accepted) = (vec![0_usize; m - 1], vec![0_usize; m - 1]);
        let (mut done, mut segment) = (0_usize, 0_usize);
        while done < s.steps {
            let steps = per.min(s.steps - done);
            let clock = done as f64 * dt;
            for (w, walker) in walkers.iter_mut().enumerate() {
                if !walker.alive {
                    continue;
                }
                let (temperature, _) = ladder[walker.rung];
                let walls: Arc<dyn PesTerm> = Arc::new(Clocked {
                    inner: Arc::new(wall_of(walker.rung)),
                    offset: clock,
                });
                let md = ctx.jobs.dynamics(
                    &walker.point,
                    &DynamicsRequest {
                        label: format!("nanoreactor trajectory {run} replica {}", w + 1),
                        terms: vec![("nano_reactor_potential".to_string(), walls)],
                        temperature,
                        time_step: dt,
                        steps,
                        seed: mix(seed, (w as u64) << 32 | segment as u64),
                        frame_every: s.frame_every,
                        velocities: walker.velocities.clone(),
                    },
                    ctx.frames,
                )?;
                if md.interrupted {
                    return Ok(None);
                }
                // A later segment starts where the one before ended (its first frame).
                let skip = usize::from(segment > 0);
                walker.frames.extend(
                    md.frames
                        .iter()
                        .skip(skip)
                        .filter_map(|p| p.system.structure().cloned()),
                );
                match md.end {
                    Some(end) if md.blew_up.is_none() => {
                        walker.point = end.point;
                        walker.velocities = Some(end.velocities);
                        walker.energy = end.biased_energy;
                    }
                    _ => {
                        walker.alive = false;
                        outcome.warnings.push(format!(
                            "replica {} of trajectory {run} blew up in the segment from step \
                             {done} (an atom jumped in one step); its frames before it are used",
                            w + 1
                        ));
                    }
                }
            }
            done += steps;
            segment += 1;
            if done >= s.steps {
                break;
            }
            // Exchanges of neighbouring rungs, the even pairs and the odd ones by turns.
            let mut rng = Rng::from_seed(mix(seed, 0x5eed_0000 + segment as u64));
            let clock = done as f64 * dt;
            for low in ((segment + 1) % 2..m - 1).step_by(2) {
                let find = |rung: usize| walkers.iter().position(|w| w.alive && w.rung == rung);
                let (Some(a), Some(b)) = (find(low), find(low + 1)) else {
                    continue;
                };
                tried[low] += 1;
                let exponent = match s.exchange {
                    Exchange::Temperature => {
                        let k = Units::BOLTZMANN_HARTREE_PER_K;
                        let (ta, tb) = (ladder[low].0, ladder[low + 1].0);
                        (1.0 / (k * ta) - 1.0 / (k * tb)) * (walkers[a].energy - walkers[b].energy)
                    }
                    Exchange::Walls => {
                        let u = |rung: usize, w: &Walker| -> MoptResult<f64> {
                            let sys = w.point.require_structure("a replica")?;
                            let term = wall_of(rung);
                            let ctx = crate::pes::term::TermContext {
                                iteration: 0,
                                initial: start,
                                time_au: Some(clock),
                            };
                            Ok(term.evaluate(sys, &ctx, false)?.energy.value())
                        };
                        let (wa, wb) = (&walkers[a], &walkers[b]);
                        let change = u(low, wb)? + u(low + 1, wa)? - u(low, wa)? - u(low + 1, wb)?;
                        -change / (Units::BOLTZMANN_HARTREE_PER_K * s.temperature)
                    }
                };
                if exponent >= 0.0 || rng.next_f64() < exponent.exp() {
                    accepted[low] += 1;
                    walkers[a].rung = low + 1;
                    walkers[b].rung = low;
                    if s.exchange == Exchange::Temperature {
                        // The velocities scaled to the temperature of the new rung.
                        let up = (ladder[low + 1].0 / ladder[low].0).sqrt();
                        if let Some(v) = walkers[a].velocities.as_mut() {
                            *v *= faer::Scale(up);
                        }
                        if let Some(v) = walkers[b].velocities.as_mut() {
                            *v *= faer::Scale(1.0 / up);
                        }
                    }
                }
            }
        }
        let rungs: Vec<String> = ladder
            .iter()
            .map(|(t, f)| match s.exchange {
                Exchange::Temperature => format!("{t:.0} K"),
                Exchange::Walls => format!("k2 x {f:.3}"),
            })
            .collect();
        let rates: Vec<String> = (0..m - 1)
            .map(|k| format!("{}/{}", accepted[k], tried[k]))
            .collect();
        ctx.jobs.note(&format!(
            "Job {}: {m} replicas of trajectory {run} on {} ({}); exchanges accepted between \
             neighbouring rungs: {}.",
            job.id,
            rungs.join(", "),
            match s.exchange {
                Exchange::Temperature => "temperatures",
                Exchange::Walls => "strengths of the narrow sphere",
            },
            rates.join(", ")
        ));
        Ok(Some(walkers.into_iter().map(|w| w.frames).collect()))
    }

    fn uses_bonds(&self, start: &MolecularSystem) -> bool {
        match self.settings.detect {
            Detection::Bonds => true,
            Detection::Quench => false,
            Detection::Auto => BondGraph::of(start, DEFAULT_BOND_SCALE)
                .adjacency
                .iter()
                .any(|n| !n.is_empty()),
        }
    }

    /// Optimize `frame` to a minimum (`None` when it did not converge; `Err` passes on).
    fn quench(
        &self,
        ctx: &mut JobContext<'_>,
        frame: &MolecularSystem,
        label: String,
    ) -> MoptResult<Option<Option<Stationary>>> {
        let m = ctx.jobs.optimize(
            &Configuration::Structure(frame.clone()),
            &OptimizeRequest {
                label,
                stage: Stage::Minimum,
                terms: Vec::new(),
            },
            ctx.frames,
        )?;
        if m.interrupted {
            return Ok(None);
        }
        Ok(Some(m.converged.then_some(Stationary {
            index_wavenumbers: m.index_wavenumbers,
            system: m.system,
            energy: m.energy,
            wavenumbers: m.wavenumbers,
            dissociation: None,
        })))
    }

    /// The reactions of a trajectory of `frames` from the minimum `start` (`None`: stopped).
    fn reactions(
        &self,
        ctx: &mut JobContext<'_>,
        start: Stationary,
        frames: &[MolecularSystem],
        warnings: &mut Vec<String>,
    ) -> MoptResult<Option<Vec<Reaction>>> {
        let s = &self.settings;
        let bonds = self.uses_bonds(start.system.require_structure("the nanoreactor (molecular dynamics in walls around the atoms; for plain variables method=addf)")?);
        let mut basin = start;
        let mut graph = BondGraph::of(
            basin.system.require_structure("the nanoreactor (molecular dynamics in walls around the atoms; for plain variables method=addf)")?,
            DEFAULT_BOND_SCALE,
        );
        let mut found = Vec::new();
        let mut since: Option<usize> = None;
        let mut after_last = 0;
        for (i, frame) in frames.iter().enumerate().skip(1) {
            if found.len() >= s.events {
                break;
            }
            let candidate = if bonds {
                let g = BondGraph::of(frame, DEFAULT_BOND_SCALE);
                if g.adjacency == graph.adjacency {
                    since = None;
                    continue;
                }
                let first = *since.get_or_insert(i);
                if i + 1 - first < s.persist {
                    continue;
                }
                since = None;
                first
            } else {
                if i % s.quench_every != 0 {
                    continue;
                }
                i
            };
            let Some(minimum) = self.quench(
                ctx,
                &frames[candidate],
                format!("frame {candidate} of the trajectory"),
            )?
            else {
                return Ok(None);
            };
            let Some(minimum) = minimum else {
                warnings.push(format!(
                    "frame {candidate} of the trajectory did not converge to a minimum"
                ));
                continue;
            };
            let minimum_structure = minimum.system.require_structure("the nanoreactor (molecular dynamics in walls around the atoms; for plain variables method=addf)")?;
            let same = if bonds {
                BondGraph::of(minimum_structure, DEFAULT_BOND_SCALE).adjacency == graph.adjacency
            } else {
                basin
                    .system
                    .same_as(basin.energy, &minimum.system, minimum.energy, ctx.matcher)?
            };
            if same {
                if !bonds {
                    after_last = candidate;
                }
                continue;
            }
            graph = BondGraph::of(minimum_structure, DEFAULT_BOND_SCALE);
            // The frames around the reaction, never before the last frame of the old basin
            // that is known (the last reaction, or with quenches the last quench that was
            // still the old basin).
            let (first, last) = if bonds {
                (
                    candidate.saturating_sub(s.segment_frames).max(after_last),
                    (candidate + s.segment_frames).min(frames.len() - 1),
                )
            } else {
                (after_last, candidate)
            };
            found.push(Reaction {
                from: basin.clone(),
                to: minimum.clone(),
                frame: candidate,
                segment: frames[first..=last].to_vec(),
            });
            after_last = candidate;
            basin = minimum;
        }
        Ok(Some(found))
    }
}

impl ExplorationMethod for Nanoreactor {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        if node.kind != NodeKind::Minimum {
            return Ok(Vec::new());
        }
        node.system.require_structure("the nanoreactor (molecular dynamics in walls around the atoms; for plain variables method=addf)")?;
        Ok((1..=self.settings.runs)
            .map(|k| {
                let mut p = BTreeMap::new();
                p.insert("run".to_string(), JsonValue::Number(k as f64));
                Task {
                    key: format!("nanoreactor:{k}"),
                    description: format!("nanoreactor trajectory {k}"),
                    parameters: JsonValue::Object(p),
                }
            })
            .collect())
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let s = &self.settings;
        let mut outcome = JobOutcome::default();
        let start_structure = job.start.require_structure("the nanoreactor (molecular dynamics in walls around the atoms; for plain variables method=addf)")?;
        if start_structure.has_pbc() {
            outcome.warnings.push(
                "the nanoreactor confines isolated structures; a periodic one is not explored by \
                 it"
                .to_string(),
            );
            return Ok(outcome);
        }
        let run = job
            .task
            .parameters
            .get("run")
            .and_then(JsonValue::as_f64)
            .ok_or_else(|| {
                MoptError::parse(format!(
                    "nanoreactor: the task '{}' has no run",
                    job.task.key
                ))
            })? as u64;
        let trajectories: Vec<Vec<MolecularSystem>> = if s.replicas > 1 {
            match self.replica_trajectories(job, ctx, start_structure, run, &mut outcome)? {
                Some(t) => t,
                None => return Ok(JobOutcome::stopped()),
            }
        } else {
            let walls: Arc<dyn PesTerm> = Arc::new(AdBias::new(self.walls(start_structure)));
            let md = ctx.jobs.dynamics(
                &job.start,
                &DynamicsRequest {
                    label: format!("nanoreactor trajectory {run}"),
                    terms: vec![("nano_reactor_potential".to_string(), walls)],
                    temperature: s.temperature,
                    time_step: s.time_step_fs * Units::FS_TO_AU_TIME,
                    steps: s.steps,
                    seed: mix(mix(s.seed, job.node as u64), run),
                    frame_every: s.frame_every,
                    velocities: None,
                },
                ctx.frames,
            )?;
            if md.interrupted {
                return Ok(JobOutcome::stopped());
            }
            if let Some(step) = md.blew_up {
                outcome.warnings.push(format!(
                    "trajectory {run} blew up at step {step} (an atom jumped in one step); the \
                     frames before it are used (a smaller method_time_step or a weaker \
                     method_k2 avoids it)"
                ));
            }
            vec![md
                .frames
                .into_iter()
                .filter_map(|p| p.system.structure().cloned())
                .collect()]
        };
        let start = Stationary {
            index_wavenumbers: None,
            system: job.start.clone(),
            energy: ctx
                .network
                .nodes
                .get(job.node)
                .map_or(f64::NAN, |n| n.energy),
            wavenumbers: None,
            dissociation: None,
        };
        let mut reactions = Vec::new();
        for frames in &trajectories {
            let Some(found) = self.reactions(ctx, start.clone(), frames, &mut outcome.warnings)?
            else {
                return Ok(JobOutcome::stopped());
            };
            reactions.extend(found);
        }
        if reactions.is_empty() {
            ctx.jobs.note(&format!(
                "Job {}: the {} showed no reaction ({} frames).",
                job.id,
                if trajectories.len() > 1 {
                    "replicas"
                } else {
                    "trajectory"
                },
                trajectories.iter().map(Vec::len).sum::<usize>()
            ));
            return Ok(outcome);
        }
        let mut guesses = Vec::new();
        for (k, r) in reactions.iter().enumerate() {
            // The first path: the old minimum, the frames around the reaction, the new
            // minimum (or the two minima alone).
            let mut structures = vec![r.from.system.clone()];
            if s.segment {
                structures.extend(r.segment.iter().cloned().map(Configuration::from));
            }
            structures.push(r.to.system.clone());
            ctx.jobs.note(&format!(
                "Job {}: reaction {} at frame {} ({:.4} -> {:.4} Hartree); {} from {}.",
                job.id,
                k + 1,
                r.frame,
                r.from.energy,
                r.to.energy,
                s.path,
                if s.segment {
                    format!(
                        "the {} frames of the trajectory between the two minima",
                        r.segment.len()
                    )
                } else {
                    "the two minima".to_string()
                }
            ));
            let search = ctx.jobs.pair_search(
                &PairSearchRequest {
                    label: format!("{} of reaction {}", s.path, k + 1),
                    job: job.id,
                    method: s.path.clone(),
                    structures,
                    settings: s.path_settings.clone(),
                    dissociation_scale: None,
                    dissociation_atoms: None,
                },
                ctx.frames,
            )?;
            if search.interrupted {
                return Ok(JobOutcome::stopped());
            }
            if search.estimates.is_empty() {
                outcome.warnings.push(format!(
                    "{} gave no transition-state estimate for reaction {} ({})",
                    s.path,
                    k + 1,
                    search.end
                ));
            }
            // Every maximum of the path is a guess (a reaction of several steps has several),
            // the highest first.
            let mut estimates = search.estimates;
            estimates.sort_by(|a, b| b.energy.total_cmp(&a.energy));
            for (m, e) in estimates.into_iter().enumerate() {
                guesses.push(Guess {
                    system: e.system,
                    origin: format!("{} maximum {} of reaction {}", s.path, m + 1, k + 1),
                });
            }
        }
        if !follow_guesses(job, ctx, &guesses, &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

#[cfg(test)]
#[path = "nanoreactor_tests.rs"]
mod tests;
