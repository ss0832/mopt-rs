//! The built-in way of exploring a minimum, `afir_ts`: an AFIR push between two atoms (the
//! single-component AFIR of MultiOptPy's mapper), the transition-state guesses on the path of
//! the push, their optimization to transition states, and the IRC of each new one with its two
//! ends optimized to minima.
//!
//! The tasks of a minimum are its pairs of atoms whose distance lies in the window
//! `[dist_lower, dist_upper]` (Angstrom) and that are not bonded (farther apart than
//! `covalent_margin` times the sum of their covalent radii), optionally only pairs of the
//! active atoms; each pair is pushed together with the collision energy `gamma` (kJ/mol) and,
//! with `negative_gamma`, also pulled apart with `-gamma`. The pairs are shuffled with a
//! generator seeded by the seed of the run and the number of the node, and at most `max_pairs`
//! are kept, so that the list is the same in every run with the same settings.
//!
//! On a variable cell (`variable_cell` in the steps) a minimum may also have tasks that push
//! the cell itself (`cell_pushes`): each periodic lattice vector, and the distance between the
//! tips of each two, compressed with `gamma` and stretched with `-gamma` (the artificial force
//! on the origin and the tips of the lattice vectors of `optmain -vc_afir`; Kondo, Hasegawa,
//! and Maeda push the translation vectors of molecular crystals the same way). The force is a
//! bias potential of the lattice vectors alone ([`CellPush`]); the atoms follow the cell.
//!
//! The optimization path of the push is relaxed without the bias by a chain method of the
//! registry of two-point methods (`method_relax`, `neb` by default: the chain relaxation of
//! `nebmain` with its defaults, Hessian-based steps from a model Hessian included; the
//! `step2_settings` of the settings file are its settings, as in the step 2 of AutoTS), and the
//! guesses are the maxima of the relaxed chain, the `top_n` highest of them. MultiOptPy's mapper
//! does the same through its whole AutoTS workflow. `method_relax=off` takes the guesses from
//! the AFIR path itself: its local maxima of the energy without the bias (the approximate
//! transition states of AFIR; Maeda et al., J. Comput. Chem. 2014, 35, 166), cheaper and
//! rougher.
//!
//! MultiOptPy: `Wrapper/mapper.py:PerturbationGenerator` (the pairs) and the four steps of
//!  `Wrapper/autots.py:AutoTSWorkflow` that `ReactionNetworkMapper._run_autots` runs.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use super::follow::Guess;
use super::method::{
    ExplorationMethod, Job, JobContext, JobOutcome, OptimizeRequest, PairSearchOutcome,
    PairSearchRequest, PathPoint, Stage, Task,
};
use super::network::{Network, Node, NodeKind};
use super::peak::follow_with_peak;
use crate::core_infra::json::JsonValue;
use crate::core_infra::rng::Rng;
use crate::domain::{AtomId, Units};
use crate::error::{MoptError, MoptResult};
use crate::parameters::covalent::covalent_radius_bohr;
use crate::pes::bias::afir::Afir;
use crate::workflow::pair_search::{Ends, PairMethodRegistry};

/// The AFIR pushes of a minimum.
#[derive(Debug, Clone, PartialEq)]
pub struct AfirSettings {
    /// Collision energy `gamma` (kJ/mol).
    pub gamma_kj_mol: f64,
    /// Pairs kept per minimum.
    pub max_pairs: usize,
    /// Smallest distance of a pair (Angstrom).
    pub dist_lower: f64,
    /// Largest distance of a pair (Angstrom).
    pub dist_upper: f64,
    /// Pairs closer than this times the sum of their covalent radii are bonded and left out.
    pub covalent_margin: f64,
    /// Only pairs of these atoms (0-based); all atoms when `None`.
    pub active_atoms: Option<Vec<usize>>,
    /// Also pull every pair apart with `-gamma`.
    pub negative_gamma: bool,
    /// Also push the cell itself (a variable cell): every periodic lattice vector and every
    /// distance between the tips of two, compressed and stretched.
    pub cell_pushes: bool,
    /// Push molecules rather than atoms (`afir_fragments`).
    pub fragments: AfirFragments,
    /// Seed of the order of the pairs.
    pub seed: u64,
}

impl Default for AfirSettings {
    /// MultiOptPy's defaults of `run_mapper` (`Entrypoints/mapper.py:merge_config`).
    fn default() -> Self {
        Self {
            gamma_kj_mol: 100.0,
            max_pairs: 9999,
            dist_lower: 1.5,
            dist_upper: 5.0,
            covalent_margin: 1.2,
            active_atoms: None,
            negative_gamma: false,
            cell_pushes: false,
            fragments: AfirFragments::Atoms,
            seed: 42,
        }
    }
}

/// What the AFIR pushes of the mapper act on (`afir_fragments`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfirFragments {
    /// Pairs of atoms (MultiOptPy's mapper).
    Atoms,
    /// Pairs of molecules (the connected parts of the bond graph, whole across a cell): each
    /// pushed or pulled as two fragments, with no force within a molecule, as the
    /// single-component AFIR of molecular crystals (Kondo, Hasegawa, Maeda; forces between
    /// automatically defined fragments, intramolecular forces switched off).
    Molecules,
}

impl AfirFragments {
    /// From `atoms` or `molecules`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "atoms" | "atom" => Ok(Self::Atoms),
            "molecules" | "molecule" => Ok(Self::Molecules),
            other => Err(MoptError::parse(format!(
                "afir_fragments: atoms or molecules, not '{other}'"
            ))),
        }
    }
}

impl AfirSettings {
    /// Refuse settings that cannot give pairs.
    pub fn check(&self) -> MoptResult<()> {
        if !(self.gamma_kj_mol.is_finite() && self.gamma_kj_mol > 0.0) {
            return Err(MoptError::parse(format!(
                "the AFIR gamma of the mapper must be positive (got {}); --negative_gamma adds \
                 the pulls",
                self.gamma_kj_mol
            )));
        }
        if self.max_pairs == 0 {
            return Err(MoptError::parse("max_pairs must be at least 1"));
        }
        if !(self.covalent_margin.is_finite() && self.covalent_margin >= 0.0) {
            return Err(MoptError::parse(format!(
                "pair_bond_scale must be 0 or more (got {})",
                self.covalent_margin
            )));
        }
        if !(self.dist_lower >= 0.0 && self.dist_lower < self.dist_upper) {
            return Err(MoptError::parse(format!(
                "the distance window of the AFIR pairs needs 0 <= dist_lower < dist_upper (got \
                 {} and {})",
                self.dist_lower, self.dist_upper
            )));
        }
        Ok(())
    }
}

/// Mix two numbers into a seed (SplitMix64 finalizer).
pub(crate) fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_add(0x9e37_79b9_7f4a_7c15).rotate_left(17);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// How the path of a push is relaxed before the transition-state guesses are taken from it.
#[derive(Debug, Clone, PartialEq)]
pub struct PathRelaxation {
    /// The chain method of the registry of two-point methods (`neb`, `dmf`); `None`: the
    /// guesses are the maxima of the AFIR path itself.
    pub method: Option<String>,
    /// Its settings (`method_relax_KEY=VALUE`).
    pub settings: Vec<(String, String)>,
}

/// The chain method that relaxes the AFIR paths by default.
pub const DEFAULT_RELAXATION: &str = "neb";

impl Default for PathRelaxation {
    fn default() -> Self {
        Self {
            method: Some(DEFAULT_RELAXATION.to_string()),
            settings: Vec::new(),
        }
    }
}

/// Another way to the first path of a pair of atoms than the AFIR push (`method_step1`): a
/// method of the registry of two-point methods that goes on from one end along the pair, so
/// that the job needs no second structure (user decision, 2026-11-19: a method of two ends
/// between the minimum and the end of the push adds little to the push itself, and is not
/// offered; its place is `method=connect` with a goal).
#[derive(Debug, Clone, PartialEq)]
pub struct Step1 {
    /// Its name in the registry.
    pub name: String,
    /// The setting of the method that takes the pair: `driving` (the driving coordinates of
    /// the single-ended growing string) or `bonds` (the bonds the Newton trajectory
    /// shortens).
    pub key: &'static str,
    /// Its settings (`method_step1_KEY=VALUE`).
    pub settings: Vec<(String, String)>,
}

impl Step1 {
    /// From the `method_KEY=VALUE` settings `extra` (without the prefix); `None` for the AFIR
    /// push itself (`method_step1=afir`, the default).
    pub fn parse(extra: &BTreeMap<String, String>) -> MoptResult<Option<Self>> {
        let settings: Vec<(String, String)> = extra
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("step1_")
                    .filter(|key| !key.is_empty())
                    .map(|key| (key.to_string(), v.clone()))
            })
            .collect();
        let name = extra.get("step1").map(|v| v.trim().to_ascii_lowercase());
        let name = match name.as_deref() {
            None | Some("afir") => {
                if let Some((key, _)) = settings.first() {
                    return Err(MoptError::parse(format!(
                        "method_step1_{key}: the AFIR push has no such setting (method_step1 \
                         names another method of step 1)"
                    )));
                }
                return Ok(None);
            }
            Some(name) => name.to_string(),
        };
        let method = PairMethodRegistry::default().get(&name)?;
        let known = method.keys();
        let has = |key: &str| known.iter().any(|(k, _)| *k == key);
        let key = match method.ends() {
            // A method that relaxes a path that exists is a method of step 2.
            Ends::Path => {
                return Err(MoptError::parse(format!(
                    "method_step1={}: it relaxes a path that exists and is a method of \
                     method_relax (step 2), not of step 1",
                    method.name()
                )))
            }
            Ends::One if has("driving") => "driving",
            Ends::OneOrTwo if has("bonds") => "bonds",
            Ends::One => {
                return Err(MoptError::parse(format!(
                    "method_step1={}: it explores a minimum as a whole and takes no pair of \
                     atoms (method={} is that way of exploring)",
                    method.name(),
                    method.name()
                )))
            }
            Ends::Two | Ends::OneOrTwo => {
                return Err(MoptError::parse(format!(
                    "method_step1={}: step 1 takes the methods that go on from one end along \
                     a pair of atoms (se_gsm, gnt); a method that needs a second structure \
                     runs between known minima and a goal (method=connect with goal=FILE and \
                     method_path={})",
                    method.name(),
                    method.name()
                )))
            }
        };
        for (k, _) in &settings {
            if k == key {
                return Err(MoptError::parse(format!(
                    "method_step1_{key}: the pair of atoms of the task is what {} goes along",
                    method.name()
                )));
            }
            if !has(k) {
                return Err(MoptError::parse(format!(
                    "method_step1_{k}: {} has no setting {k} (its keys: {})",
                    method.name(),
                    known.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", ")
                )));
            }
        }
        Ok(Some(Self {
            name: method.name().to_string(),
            key,
            settings,
        }))
    }
}

/// The `method_KEY` settings of `afir_ts`.
pub const KEYS: &[(&str, &str)] = &[
    (
        "relax",
        "the chain method that relaxes the AFIR path before the guesses are taken (neb, dmf), or \
         off: the maxima of the AFIR path itself (neb)",
    ),
    (
        "relax_KEY",
        "a setting of that method (over the step2_settings of the settings file)",
    ),
    (
        "step1",
        "the method that makes the first path of a pair of atoms: afir, the AFIR push (the \
         default), or a method that goes on from the minimum along the pair: se_gsm (the \
         single-ended growing string) or gnt (the Newton trajectory); neb and dmf are methods \
         of method_relax, and a method of two ends runs with method=connect and a goal",
    ),
    ("step1_KEY", "a setting of that method"),
];

impl PathRelaxation {
    /// From the `method_KEY=VALUE` settings `extra` (without the prefix) and the
    /// `step2_settings` of the settings file (`nebmain`'s settings, as compact JSON).
    pub fn parse(extra: &BTreeMap<String, String>, step2: Option<&str>) -> MoptResult<Self> {
        let mut r = Self::default();
        for (k, v) in extra {
            // (The settings of step 1 are read by [`Step1::parse`].)
            if k == "step1" || k.starts_with("step1_") {
                continue;
            }
            if k == "relax" {
                r.method = match v.to_ascii_lowercase().as_str() {
                    "off" | "none" | "false" | "no" | "0" => None,
                    name => Some(name.to_string()),
                };
                continue;
            }
            match k.strip_prefix("relax_") {
                Some(key) if !key.is_empty() => r.settings.push((key.to_string(), v.clone())),
                _ => {
                    return Err(MoptError::parse(format!(
                        "the mapper method afir_ts: unknown setting method_{k} (keys: {})",
                        KEYS.iter()
                            .map(|(k, _)| format!("method_{k}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )))
                }
            }
        }
        let Some(name) = &r.method else {
            return Ok(r);
        };
        let method = PairMethodRegistry::default().get(name)?;
        if method.ends() != Ends::Path {
            return Err(MoptError::parse(format!(
                "method_relax: {} does not take a first path; the AFIR path is relaxed by a chain \
                 method (neb, dmf), or method_relax=off",
                method.name()
            )));
        }
        let known = method.keys();
        for (k, _) in &r.settings {
            if !known.iter().any(|(key, _)| key == k) {
                return Err(MoptError::parse(format!(
                    "method_relax_{k}: {} has no setting {k} (its keys: {})",
                    method.name(),
                    known.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", ")
                )));
            }
        }
        let has_settings = known.iter().any(|(key, _)| *key == "settings");
        if let (Some(json), true) = (step2, has_settings) {
            if !r.settings.iter().any(|(k, _)| k == "settings") {
                r.settings
                    .insert(0, ("settings".to_string(), json.to_string()));
            }
        }
        Ok(r)
    }
}

/// The method `afir_ts`.
#[derive(Debug, Clone)]
pub struct AfirTs {
    settings: AfirSettings,
    top_n: usize,
    relax: PathRelaxation,
    /// Another method of step 1 than the AFIR push (`method_step1`).
    step1: Option<Step1>,
}

/// The parameters of a task: the AFIR push of two fragments (a pair of atoms for `afir_ts`).
pub(super) struct Push {
    /// The force parameter (kJ/mol; negative: a pull).
    pub(super) gamma: f64,
    /// The atoms of the first fragment (from 0).
    pub(super) a: Vec<usize>,
    /// The atoms of the second fragment.
    pub(super) b: Vec<usize>,
}

impl Push {
    /// The bias potential of the push.
    pub(super) fn term(&self) -> MoptResult<Afir> {
        let ids = |atoms: &[usize]| atoms.iter().map(|&i| AtomId::new(i)).collect();
        Afir::new(self.gamma, ids(&self.a), ids(&self.b))
    }
}

/// An artificial force on the cell itself as a bias potential: the energy of the
/// [`crate::opt::cell::CellForce`] of the lattice vectors of a structure (no gradient on the
/// atoms; its stress, which moves a variable cell, comes from its gradient with respect to the
/// lattice vectors).
pub struct CellPush {
    force: crate::opt::cell::CellForce,
    name: String,
}

impl CellPush {
    /// The force `gamma` (kJ/mol; positive compresses) on the distance `distance` of the cell
    /// (`a`, `b`, `c`, `ab`, `bc`, `ca`).
    pub fn new(gamma: f64, distance: &str) -> MoptResult<Self> {
        Ok(Self {
            force: crate::opt::cell::CellForce {
                gamma: crate::pes::bias::Ramp::constant(gamma),
                distances: vec![crate::opt::cell::CellDistance::parse(distance)?],
            },
            name: format!("cell_AFIR_{distance}"),
        })
    }
}

impl crate::pes::term::PesTerm for CellPush {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(
        &self,
        sys: &crate::domain::MolecularSystem,
        ctx: &crate::pes::term::TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<crate::domain::EvalOutput> {
        let cell = sys.active_cell().ok_or_else(|| {
            MoptError::config(format!("{}: the structure has no periodic cell", self.name))
        })?;
        let vectors: [[f64; 3]; 3] = std::array::from_fn(|k| cell.lattice_vector(k).value());
        let (energy, of_lattice) = self.force.evaluate(&vectors, ctx.iteration);
        let n = sys.dof();
        let mut out = crate::domain::EvalOutput::energy_gradient(
            crate::domain::Quantity::new(energy),
            faer::Col::zeros(n),
        );
        if need_hessian {
            out.hessian_bohr = Some(faer::Mat::zeros(n, n));
            // Its second derivatives along the strain (none with the atoms).
            out.strain_hessian = Some(crate::domain::StrainHessian {
                strain: self.force.strain_hessian(&vectors, ctx.iteration),
                mixed: faer::Mat::zeros(n, 9),
            });
        }
        // Its stress, from its gradient with respect to the lattice vectors.
        out.stress = Some(crate::pes::stress::lattice_stress(&of_lattice, cell));
        Ok(out)
    }

    fn atoms(&self) -> Option<Vec<AtomId>> {
        Some(Vec::new())
    }

    /// A function of the lattice vectors alone: no motion of the atoms changes it, and its
    /// force is constant.
    fn frame(&self) -> crate::pes::term::Frame {
        crate::pes::term::Frame::Invariant
    }

    fn history(&self) -> crate::pes::term::History {
        crate::pes::term::History::Static
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn analytic_stress(&self) -> bool {
        true
    }
}

/// The push of the cell of `task` (an `afir:cell:` task): its force and the distance it acts
/// on; `None` for a push of atoms.
fn cell_push_of(task: &Task) -> MoptResult<Option<CellPush>> {
    let Some(distance) = task.parameters.get("cell").and_then(JsonValue::as_str) else {
        return Ok(None);
    };
    let gamma = task
        .parameters
        .get("gamma_kj_mol")
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| {
            MoptError::parse(format!("afir_ts: the task '{}' has no gamma", task.key))
        })?;
    CellPush::new(gamma, distance).map(Some)
}

/// The AFIR push of `task` (an `afir:` task, or a task with its parameters): the fragments
/// `atoms_a` and `atoms_b`, or the pair of atoms `atom_a` and `atom_b`.
pub(super) fn push_of(task: &Task) -> MoptResult<Push> {
    let bad = || MoptError::parse(format!("afir_ts: the task '{}' has no AFIR push", task.key));
    let get = |k: &str| task.parameters.get(k).and_then(JsonValue::as_f64);
    let fragment = |many: &str, one: &str| -> Option<Vec<usize>> {
        match task.parameters.get(many).and_then(JsonValue::as_array) {
            Some(items) => items.iter().map(JsonValue::as_usize).collect(),
            None => get(one).map(|i| vec![i as usize]),
        }
    };
    Ok(Push {
        gamma: get("gamma_kj_mol").ok_or_else(bad)?,
        a: fragment("atoms_a", "atom_a").ok_or_else(bad)?,
        b: fragment("atoms_b", "atom_b").ok_or_else(bad)?,
    })
}

/// Indices of the local maxima of the energy along `path` (inner points higher than the point
/// before and not lower than the one after), highest first.
pub fn local_maxima(path: &[PathPoint]) -> Vec<usize> {
    let mut out: Vec<usize> = (1..path.len().saturating_sub(1))
        .filter(|&i| path[i].energy > path[i - 1].energy && path[i].energy >= path[i + 1].energy)
        .collect();
    out.sort_by(|&a, &b| path[b].energy.total_cmp(&path[a].energy).then(a.cmp(&b)));
    out
}

impl AfirTs {
    /// The registered name.
    pub const NAME: &'static str = "afir_ts";

    /// The method with `settings`, optimizing the `top_n` highest guesses of every push.
    pub fn new(settings: AfirSettings, top_n: usize) -> MoptResult<Self> {
        settings.check()?;
        if top_n == 0 {
            return Err(MoptError::parse("top_n_candidates (-n) must be at least 1"));
        }
        Ok(Self {
            settings,
            top_n,
            relax: PathRelaxation::default(),
            step1: None,
        })
    }

    /// The method with `relax` as the relaxation of the AFIR paths.
    pub fn with_relaxation(mut self, relax: PathRelaxation) -> Self {
        self.relax = relax;
        self
    }

    /// The method with `step1` in the place of the AFIR push (`None`: the push).
    pub fn with_step1(mut self, step1: Option<Step1>) -> Self {
        self.step1 = step1;
        self
    }

    /// The guesses of the search `search` of the method `name` (its estimates of transition
    /// states, the highest `top_n`) followed, with the peak of its path where the run keeps
    /// it.
    fn follow_search(
        &self,
        job: &Job,
        ctx: &mut JobContext<'_>,
        search: PairSearchOutcome,
        name: &str,
        mut outcome: JobOutcome,
    ) -> MoptResult<JobOutcome> {
        if search.interrupted {
            return Ok(JobOutcome::stopped());
        }
        if !search.converged {
            ctx.jobs.note(&format!(
                "Job {}: {name} ended before its criteria ({}); the maxima of its path are the \
                 guesses.",
                job.id, search.end
            ));
        }
        let mut estimates = search.estimates;
        estimates.sort_by(|a, b| b.energy.total_cmp(&a.energy));
        let guesses: Vec<Guess> = estimates
            .into_iter()
            .take(self.top_n)
            .enumerate()
            .map(|(m, e)| Guess {
                system: e.system,
                origin: format!("maximum {} of the path of {name}", m + 1),
            })
            .collect();
        if guesses.is_empty() {
            ctx.jobs.note(&format!(
                "Job {}: {name} gave no transition-state estimate ({}).",
                job.id, search.end
            ));
            return Ok(outcome);
        }
        if !follow_with_peak(
            job,
            ctx,
            &guesses,
            &search.path,
            name,
            search.converged,
            &mut outcome,
        )? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }

    /// Step 1 by a method that goes on from the minimum along the pair of the push (no AFIR
    /// push).
    fn run_driven(
        &self,
        job: &Job,
        ctx: &mut JobContext<'_>,
        push: &Push,
        step1: &Step1,
    ) -> MoptResult<JobOutcome> {
        let ([a], [b]) = (&push.a[..], &push.b[..]) else {
            return Err(MoptError::config(format!(
                "method_step1={}: it goes along a pair of atoms, and the task '{}' pushes \
                 fragments",
                step1.name, job.task.key
            )));
        };
        let value = match (step1.key, push.gamma > 0.0) {
            ("driving", true) => format!("ADD {} {}", a + 1, b + 1),
            ("driving", false) => format!("BREAK {} {}", a + 1, b + 1),
            (_, true) => format!("{}-{}", a + 1, b + 1),
            (_, false) => {
                // The Newton trajectory shortens bonds: it has no pull.
                ctx.jobs.note(&format!(
                    "Job {}: {} shortens the distance of a pair and has no pull; the pull of \
                     the atoms {} and {} is left out.",
                    job.id,
                    step1.name,
                    a + 1,
                    b + 1
                ));
                return Ok(JobOutcome::default());
            }
        };
        let mut settings = step1.settings.clone();
        settings.push((step1.key.to_string(), value));
        let label = format!("{} along the atoms {} and {}", step1.name, a + 1, b + 1);
        let search = ctx.jobs.pair_search(
            &PairSearchRequest {
                label: label.clone(),
                job: job.id,
                method: step1.name.clone(),
                structures: vec![job.start.clone()],
                settings,
                dissociation_scale: None,
                dissociation_atoms: None,
            },
            ctx.frames,
        )?;
        self.follow_search(job, ctx, search, &label, JobOutcome::default())
    }

    /// The guesses from the local maxima of the AFIR path itself.
    fn path_maxima(&self, path: &[PathPoint]) -> Vec<Guess> {
        local_maxima(path)
            .into_iter()
            .take(self.top_n)
            .map(|at| Guess {
                system: path[at].system.clone(),
                origin: format!("point {} of the AFIR path", at + 1),
            })
            .collect()
    }

    /// The pairs of molecules of `node` (their atoms, 0-based, ascending), in the order of the
    /// tasks: those with an active atom (all without `active_atoms`) whose closest atoms (the
    /// minimum image in a cell) are within the distance window. A molecule and its own
    /// images are no pair.
    pub fn molecule_pairs(&self, node: &Node) -> Vec<(Vec<usize>, Vec<usize>)> {
        let s = &self.settings;
        let Some(sys) = node.system.structure() else {
            return Vec::new();
        };
        let molecules: Vec<Vec<usize>> =
            super::channels::molecules(sys, crate::workflow::explore::identity::DEFAULT_BOND_SCALE)
                .into_iter()
                .filter(|m| {
                    s.active_atoms
                        .as_ref()
                        .is_none_or(|a| m.iter().any(|i| a.contains(i)))
                })
                .collect();
        let mut pairs = Vec::new();
        for (k, ma) in molecules.iter().enumerate() {
            for mb in &molecules[k + 1..] {
                let closest = ma
                    .iter()
                    .flat_map(|&i| {
                        mb.iter()
                            .map(move |&j| sys.distance(AtomId::new(i), AtomId::new(j)).value())
                    })
                    .fold(f64::INFINITY, f64::min);
                let d_ang = closest * Units::BOHR_TO_ANGSTROM;
                if d_ang >= s.dist_lower && d_ang <= s.dist_upper {
                    pairs.push((ma.clone(), mb.clone()));
                }
            }
        }
        Rng::from_seed(mix(s.seed, node.id as u64)).shuffle(&mut pairs);
        pairs.truncate(s.max_pairs);
        pairs
    }

    /// The pairs of atoms of `node` (0-based, `a < b`), in the order of the tasks.
    pub fn pairs(&self, node: &Node) -> Vec<(usize, usize)> {
        let s = &self.settings;
        let Some(sys) = node.system.structure() else {
            return Vec::new();
        };
        let active = |i: usize| s.active_atoms.as_ref().is_none_or(|a| a.contains(&i));
        let mut pairs = Vec::new();
        for i in 0..sys.n_atoms() {
            for j in (i + 1)..sys.n_atoms() {
                if !(active(i) && active(j)) {
                    continue;
                }
                let d = sys.distance(AtomId::new(i), AtomId::new(j)).value();
                let d_ang = d * Units::BOHR_TO_ANGSTROM;
                let bonded = d
                    <= s.covalent_margin
                        * (covalent_radius_bohr(sys.elements[i])
                            + covalent_radius_bohr(sys.elements[j]));
                if d_ang >= s.dist_lower && d_ang <= s.dist_upper && !bonded {
                    pairs.push((i, j));
                }
            }
        }
        Rng::from_seed(mix(s.seed, node.id as u64)).shuffle(&mut pairs);
        pairs.truncate(s.max_pairs);
        pairs
    }
}

impl ExplorationMethod for AfirTs {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        if node.kind != NodeKind::Minimum {
            return Ok(Vec::new());
        }
        node.system.require_structure(
            "afir_ts (AFIR pushes between pairs of atoms; for plain variables method=addf)",
        )?;
        let g = self.settings.gamma_kj_mol;
        let mut gammas = vec![g];
        if self.settings.negative_gamma {
            gammas.push(-g);
        }
        let mut tasks = Vec::new();
        if self.settings.fragments == AfirFragments::Molecules {
            let numbers = |atoms: &[usize]| {
                JsonValue::Array(atoms.iter().map(|&i| JsonValue::Number(i as f64)).collect())
            };
            for (a, b) in self.molecule_pairs(node) {
                for &gamma in &gammas {
                    let mut p = BTreeMap::new();
                    p.insert("gamma_kj_mol".to_string(), JsonValue::Number(gamma));
                    p.insert("atoms_a".to_string(), numbers(&a));
                    p.insert("atoms_b".to_string(), numbers(&b));
                    tasks.push(Task {
                        key: format!("afir:{gamma:+}:mol{}-mol{}", a[0] + 1, b[0] + 1),
                        description: format!(
                            "AFIR {} the molecules of atoms {} and {} ({gamma} kJ/mol)",
                            if gamma > 0.0 { "push of" } else { "pull of" },
                            a[0] + 1,
                            b[0] + 1
                        ),
                        parameters: JsonValue::Object(p),
                    });
                }
            }
        }
        let atom_pairs = match self.settings.fragments {
            AfirFragments::Atoms => self.pairs(node),
            AfirFragments::Molecules => Vec::new(),
        };
        for (a, b) in atom_pairs {
            for &gamma in &gammas {
                let mut p = BTreeMap::new();
                p.insert("gamma_kj_mol".to_string(), JsonValue::Number(gamma));
                p.insert("atom_a".to_string(), JsonValue::Number(a as f64));
                p.insert("atom_b".to_string(), JsonValue::Number(b as f64));
                tasks.push(Task {
                    key: format!("afir:{gamma:+}:{}-{}", a + 1, b + 1),
                    description: format!(
                        "AFIR {} atoms {} and {} ({gamma} kJ/mol)",
                        if gamma > 0.0 { "push of" } else { "pull of" },
                        a + 1,
                        b + 1
                    ),
                    parameters: JsonValue::Object(p),
                });
            }
        }
        // The cell itself: each periodic lattice vector and each distance between the tips
        // of two periodic ones, compressed and stretched.
        if self.settings.cell_pushes {
            if let Some(cell) = node.system.structure().and_then(|s| s.active_cell()) {
                let periodic = cell.periodic_axes;
                let mut distances = Vec::new();
                for (k, name) in ["a", "b", "c"].iter().enumerate() {
                    if periodic[k] {
                        distances.push((*name).to_string());
                    }
                }
                for (k, l, name) in [(0, 1, "ab"), (1, 2, "bc"), (2, 0, "ca")] {
                    if periodic[k] && periodic[l] {
                        distances.push(name.to_string());
                    }
                }
                for distance in distances {
                    for gamma in [g, -g] {
                        let mut p = BTreeMap::new();
                        p.insert("gamma_kj_mol".to_string(), JsonValue::Number(gamma));
                        p.insert("cell".to_string(), JsonValue::String(distance.clone()));
                        tasks.push(Task {
                            key: format!("afir:cell:{gamma:+}:{distance}"),
                            description: format!(
                                "AFIR {} of the cell along {distance} ({gamma} kJ/mol)",
                                if gamma > 0.0 {
                                    "compression"
                                } else {
                                    "stretch"
                                }
                            ),
                            parameters: JsonValue::Object(p),
                        });
                    }
                }
            }
        }
        Ok(tasks)
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        // A push of the cell (step 1 by another method is for pairs of atoms), or of atoms.
        let term: Arc<dyn crate::pes::term::PesTerm> = match cell_push_of(&job.task)? {
            Some(cell) => Arc::new(cell),
            None => {
                let push = push_of(&job.task)?;
                if let Some(step1) = &self.step1 {
                    return self.run_driven(job, ctx, &push, step1);
                }
                Arc::new(push.term()?)
            }
        };
        let path = ctx.jobs.optimize(
            &job.start,
            &OptimizeRequest {
                label: "AFIR path".to_string(),
                stage: Stage::Biased,
                terms: vec![("AFIR_pot_0".to_string(), term)],
            },
            ctx.frames,
        )?;
        if path.interrupted {
            return Ok(JobOutcome::stopped());
        }
        let mut outcome = JobOutcome::default();
        // The path the guesses are taken from, for its peak (`path_peaks`).
        let mut last_path: Option<(Vec<PathPoint>, String, bool)> = None;
        let guesses: Vec<Guess> = match &self.relax.method {
            Some(name) if path.path.len() >= 2 => {
                // The path without the bias, relaxed by the chain from the points of the push.
                let search = ctx.jobs.pair_search(
                    &PairSearchRequest {
                        label: format!("{name} of the AFIR path"),
                        job: job.id,
                        method: name.clone(),
                        structures: path.path.iter().map(|p| p.system.clone()).collect(),
                        settings: self.relax.settings.clone(),
                        dissociation_scale: None,
                        dissociation_atoms: None,
                    },
                    ctx.frames,
                )?;
                if search.interrupted {
                    return Ok(JobOutcome::stopped());
                }
                if !search.converged {
                    ctx.jobs.note(&format!(
                        "Job {}: the {name} relaxation of the AFIR path ended before its criteria \
                         ({}); the maxima of its last chain are the guesses.",
                        job.id, search.end
                    ));
                }
                if !search.path.is_empty() {
                    last_path = Some((
                        search.path,
                        format!("{name} of the AFIR path"),
                        search.converged,
                    ));
                }
                let mut estimates = search.estimates;
                estimates.sort_by(|a, b| b.energy.total_cmp(&a.energy));
                estimates
                    .into_iter()
                    .take(self.top_n)
                    .enumerate()
                    .map(|(m, e)| Guess {
                        system: e.system,
                        origin: format!("maximum {} of the AFIR path relaxed by {name}", m + 1),
                    })
                    .collect()
            }
            _ => self.path_maxima(&path.path),
        };
        let (peak_path, peak_method, peak_converged) = last_path
            .unwrap_or_else(|| (path.path.clone(), "AFIR path".to_string(), path.converged));
        if guesses.is_empty() {
            ctx.jobs.note(&format!(
                "Job {}: the energy along the {} has no maximum; no transition-state guess.",
                job.id,
                if self.relax.method.is_some() {
                    "relaxed AFIR path"
                } else {
                    "AFIR path"
                }
            ));
            return Ok(outcome);
        }
        if !follow_with_peak(
            job,
            ctx,
            &guesses,
            &peak_path,
            &peak_method,
            peak_converged,
            &mut outcome,
        )? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge};
    use crate::workflow::mapper::network::{ElectronicState, Level, Origin};

    #[test]
    fn pushes_of_the_cell_are_tasks_and_their_stress_is_that_of_their_energy() {
        use crate::pes::term::{PesTerm, TermContext};
        let ar = Element::from_atomic_number(18).unwrap();
        let cell = crate::domain::PeriodicCell::from_lattice_vectors(
            crate::domain::Vector3::<Angstrom>::new([5.0, 0.2, 0.0]),
            crate::domain::Vector3::<Angstrom>::new([0.3, 4.8, 0.1]),
            crate::domain::Vector3::<Angstrom>::new([0.0, -0.2, 5.5]),
            [true; 3],
        )
        .unwrap();
        let sys = MolecularSystem::new(
            vec![ar; 2],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap();
        let mut n = node(0);
        n.system = sys.clone().into();
        let s = AfirSettings {
            cell_pushes: true,
            active_atoms: Some(vec![0, 1]),
            ..AfirSettings::default()
        };
        let tasks = AfirTs::new(s, 1)
            .unwrap()
            .tasks(&n, &Network::default())
            .unwrap();
        let cells: Vec<&Task> = tasks
            .iter()
            .filter(|t| t.key.starts_with("afir:cell:"))
            .collect();
        // a, b, c, ab, bc, ca, each compressed and stretched.
        assert_eq!(cells.len(), 12);
        assert!(cells.iter().any(|t| t.key == "afir:cell:-100:ab"));
        // The stress of a push (a compression and a stretch) is the strain derivative of its
        // energy, and the push has no force on the atoms.
        let ctx = TermContext::at_iteration(0, &sys);
        for (gamma, distance) in [(100.0, "a"), (-100.0, "bc")] {
            let push = CellPush::new(gamma, distance).unwrap();
            let out = push.evaluate(&sys, &ctx, false).unwrap();
            let analytic = out.stress.unwrap();
            let numeric =
                crate::pes::stress::numerical_term_stress(&push, &sys, &ctx, 1e-5).unwrap();
            for a in 0..3 {
                for b in 0..3 {
                    assert!(
                        (analytic[a][b] - numeric[a][b]).abs() < 1e-10,
                        "{distance} {a}{b}: {} {}",
                        analytic[a][b],
                        numeric[a][b]
                    );
                }
            }
            assert!(out.gradient_bohr.iter().all(|g| *g == 0.0));
        }
    }

    fn node(id: usize) -> Node {
        // A water molecule and an argon atom.
        let sys = MolecularSystem::new(
            vec![
                Element::O,
                Element::H,
                Element::H,
                Element::from_atomic_number(18).unwrap(),
            ],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.117],
                [0.0, 0.757, -0.469],
                [0.0, -0.757, -0.469],
                [3.5, 0.0, 0.0],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let mut n = Node::new(
            NodeKind::Minimum,
            Level::Exploration,
            sys.clone().into(),
            0.0,
            ElectronicState::of(&sys, 0),
            Origin::default(),
        );
        n.id = id;
        n
    }

    #[test]
    fn pairs_skip_bonds_and_keep_the_window_and_their_order() {
        let method = AfirTs::new(AfirSettings::default(), 3).unwrap();
        let n = node(0);
        let mut pairs = method.pairs(&n);
        pairs.sort_unstable();
        // O-H are bonded; H-H 1.51 Angstrom is within the window and not bonded; Ar with all.
        assert_eq!(pairs, vec![(0, 3), (1, 2), (1, 3), (2, 3)]);
        // The same order every time, another one for another node.
        assert_eq!(method.pairs(&n), method.pairs(&n));
        let tasks = method.tasks(&n, &Network::default()).unwrap();
        assert_eq!(tasks.len(), 4);
        assert!(tasks.iter().all(|t| t.key.starts_with("afir:+100:")));
        let both = AfirTs::new(
            AfirSettings {
                negative_gamma: true,
                active_atoms: Some(vec![1, 2, 3]),
                max_pairs: 2,
                ..AfirSettings::default()
            },
            1,
        )
        .unwrap();
        let tasks = both.tasks(&n, &Network::default()).unwrap();
        assert_eq!(tasks.len(), 4);
        assert!(tasks[1].key.starts_with("afir:-100:"));
        assert!(AfirTs::new(
            AfirSettings {
                dist_lower: 6.0,
                ..AfirSettings::default()
            },
            1
        )
        .is_err());
    }

    #[test]
    fn molecules_are_pushed_as_fragments_without_forces_within_them() {
        // The water and the argon atom: one pair of molecules (closest atoms 3.5 Angstrom),
        // pushed and pulled as two fragments; no pair of atoms of the water.
        let method = AfirTs::new(
            AfirSettings {
                fragments: AfirFragments::Molecules,
                negative_gamma: true,
                ..AfirSettings::default()
            },
            1,
        )
        .unwrap();
        let n = node(0);
        assert_eq!(method.molecule_pairs(&n), vec![(vec![0, 1, 2], vec![3])]);
        let tasks = method.tasks(&n, &Network::default()).unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].key, "afir:+100:mol1-mol4");
        let push = push_of(&tasks[0]).unwrap();
        assert_eq!((push.a.clone(), push.b.clone()), (vec![0, 1, 2], vec![3]));
        assert!(push.term().is_ok());
        assert!(push_of(&tasks[1]).unwrap().gamma < 0.0);
        // Out of the window: no task.
        let far = AfirTs::new(
            AfirSettings {
                fragments: AfirFragments::Molecules,
                dist_upper: 3.0,
                ..AfirSettings::default()
            },
            1,
        )
        .unwrap();
        assert!(far.tasks(&n, &Network::default()).unwrap().is_empty());
        assert_eq!(
            AfirFragments::parse("Molecules").unwrap(),
            AfirFragments::Molecules
        );
        assert!(AfirFragments::parse("residues").is_err());
    }

    #[test]
    fn the_path_is_relaxed_by_neb_unless_turned_off() {
        let none = BTreeMap::new();
        let r = PathRelaxation::parse(&none, Some("{\"NSTEP\": 15}")).unwrap();
        assert_eq!(r.method.as_deref(), Some("neb"));
        assert_eq!(
            r.settings,
            vec![("settings".to_string(), "{\"NSTEP\": 15}".to_string())]
        );
        let mut extra = BTreeMap::new();
        extra.insert("relax".to_string(), "off".to_string());
        assert_eq!(PathRelaxation::parse(&extra, None).unwrap().method, None);
        extra.insert("relax".to_string(), "dmf".to_string());
        extra.insert("relax_iterations".to_string(), "50".to_string());
        let r = PathRelaxation::parse(&extra, None).unwrap();
        assert_eq!(r.method.as_deref(), Some("dmf"));
        assert_eq!(
            r.settings,
            vec![("iterations".to_string(), "50".to_string())]
        );
        // A method without a first path, a key the method does not have, a key of no one.
        for (k, v) in [("relax", "dsafir"), ("relax_springs", "1"), ("gamma", "1")] {
            let mut bad = BTreeMap::new();
            bad.insert(k.to_string(), v.to_string());
            assert!(PathRelaxation::parse(&bad, None).is_err(), "{k}={v}");
        }
    }

    #[test]
    fn step_1_takes_the_methods_that_make_a_path_and_not_those_that_relax_one() {
        let from = |pairs: &[(&str, &str)]| {
            let extra: BTreeMap<String, String> = pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect();
            // The relaxation leaves the settings of step 1 alone.
            assert!(PathRelaxation::parse(&extra, None).is_ok());
            Step1::parse(&extra)
        };
        assert_eq!(from(&[]).unwrap(), None);
        assert_eq!(from(&[("step1", "afir")]).unwrap(), None);
        // The single-ended string is driven by the pair, the Newton trajectory shortens it.
        let string = from(&[("step1", "se_gsm"), ("step1_nodes", "9")])
            .unwrap()
            .unwrap();
        assert_eq!((string.name.as_str(), string.key), ("se_gsm", "driving"));
        assert_eq!(
            string.settings,
            vec![("nodes".to_string(), "9".to_string())]
        );
        let newton = from(&[("step1", "GNT")]).unwrap().unwrap();
        assert_eq!((newton.name.as_str(), newton.key), ("gnt", "bonds"));
        // The methods of step 2, the methods that need a second structure, a method that
        // explores a minimum as a whole, a setting the method does not have, the pair itself
        // as a setting, a setting of the push.
        for (pairs, why) in [
            (vec![("step1", "neb")], "method_relax"),
            (vec![("step1", "dmf")], "method_relax"),
            (vec![("step1", "gsm")], "method=connect"),
            (vec![("step1", "fsm")], "method=connect"),
            (vec![("step1", "dsafir")], "method=connect"),
            (vec![("step1", "addf")], "as a whole"),
            (vec![("step1", "gnt"), ("step1_springs", "1")], "no setting"),
            (
                vec![("step1", "se_gsm"), ("step1_driving", "ADD 1 2")],
                "goes along",
            ),
            (vec![("step1", "gnt"), ("step1_bonds", "1-2")], "goes along"),
            (vec![("step1_nodes", "9")], "no such setting"),
            (vec![("step1", "nothing")], "no two-point method"),
        ] {
            let e = Step1::parse(
                &pairs
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                    .collect(),
            )
            .unwrap_err()
            .to_string();
            assert!(e.contains(why), "{pairs:?}: {e}");
        }
    }

    #[test]
    fn guesses_are_the_inner_maxima_highest_first() {
        let sys = node(0).system;
        let path: Vec<PathPoint> = [0.0, 1.0, 0.5, 2.0, 1.5, 3.0]
            .iter()
            .map(|&e| PathPoint {
                system: sys.clone(),
                energy: e,
                biased_energy: e,
            })
            .collect();
        assert_eq!(local_maxima(&path), vec![3, 1]);
    }
}
