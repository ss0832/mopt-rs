//! One registry of the methods that look for transition states from two structures (or
//! from one, along a direction or driving coordinates), so that a workflow (an automated
//! transition-state search, a reaction-network mapper) can pick any of them by name with
//! `KEY=VALUE` settings.
//!
//! Every method takes a [`PairInput`] (the end points; for the one-ended methods one point
//! and what drives it), the surface ([`SearchSurface`]: structures on a calculator from the
//! calculator registry, or an objective of plain variables), and [`PairSettings`]; every
//! method returns a [`PairResult`]: the estimates of the transition states ranked by
//! energy, the path where the method has one, the end points, and the counts of the
//! evaluations. The names, what they take, and their keys are in `docs/source/reaction_paths.md`
//! ("The registry of two-point methods") and in [`PairMethod::keys`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use faer::Col;

use super::relax_path::{
    check_relaxation, dsafir_alias, relax_path, PathRelaxation, DSAFIR_ALIAS_KEYS, RELAX_KEYS,
};
use super::surface::{SearchSurface, SurfaceValue};
use super::SearchRun;
use crate::domain::{Element, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::opt::convergence::ConvergenceCriteria;
use crate::path::gsm::DrivingCoordinate;
use crate::path::points::{PointRequest, StepMap};
use crate::path::string_nodes::PointGeometry;

/// Which end points a method takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ends {
    /// Two (a reactant and a product).
    Two,
    /// One (a minimum; with driving coordinates or bonds where the method needs them).
    One,
    /// One with a direction, or two (the direction towards the second).
    OneOrTwo,
    /// Two, or a whole first path: the ends and the points between them in their order (a
    /// trajectory, a path of another method).
    Path,
}

impl Ends {
    /// How it is written in the documentation.
    pub fn text(self) -> &'static str {
        match self {
            Self::Two => "two",
            Self::One => "one",
            Self::OneOrTwo => "one or two",
            Self::Path => "two, or a first path",
        }
    }
}

/// A test whether a point has fallen apart into fragments (structures).
pub type FragmentTest = Arc<dyn Fn(&Col<f64>) -> bool + Send + Sync>;

/// What a method is given besides its settings.
#[derive(Clone, Default)]
pub struct PairInput {
    /// The end points (one or two; Bohr for structures).
    pub points: Vec<Col<f64>>,
    /// Rigid-body motions and fixed variables of the points.
    pub geometry: PointGeometry,
    /// Whether the points are structures in Cartesian coordinates (lengths given in
    /// Angstrom are taken to Bohr; atoms can be named).
    pub structures: bool,
    /// The elements of the atoms (structures).
    pub elements: Option<Vec<Element>>,
    /// The first structure (structures; chain methods lay their images onto it).
    pub template: Option<MolecularSystem>,
    /// Driving coordinates of the single-ended growing string.
    pub driving: Vec<DrivingCoordinate>,
    /// The direction of the one-ended methods with one point (dimer, spring pair, Newton
    /// trajectory); `None`: towards the second point, or random.
    pub direction: Option<Col<f64>>,
    /// The test for points that fall apart (ADDF ends such paths).
    pub fragments: Option<FragmentTest>,
    /// The points are coordinates of structures in Bohr (and radians) without being
    /// Cartesian (`-coord`): lengths given in Angstrom are taken to Bohr.
    pub bohr_lengths: bool,
    /// The steps of iEIP and DS-AFIR through other coordinates (`-coord`).
    pub step_map: Option<StepMap>,
}

impl std::fmt::Debug for PairInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairInput")
            .field("points", &self.points.len())
            .field("structures", &self.structures)
            .field("driving", &self.driving)
            .finish_non_exhaustive()
    }
}

impl PairInput {
    /// A length given in Angstrom, in the unit of the points.
    pub fn length(&self, angstrom: f64) -> f64 {
        if self.structures || self.bohr_lengths {
            angstrom * Units::ANGSTROM_TO_BOHR
        } else {
            angstrom
        }
    }

    /// The first point.
    pub fn first(&self) -> MoptResult<&Col<f64>> {
        self.points
            .first()
            .ok_or_else(|| MoptError::config("a two-point method needs a point to start from"))
    }

    /// The second point.
    pub fn second(&self, method: &str) -> MoptResult<&Col<f64>> {
        self.points
            .get(1)
            .ok_or_else(|| MoptError::config(format!("{method} needs two end points")))
    }
}

/// `KEY=VALUE` settings of a method; [`PairSettings::finish`] refuses keys that were not
/// read.
#[derive(Debug, Clone, Default)]
pub struct PairSettings {
    pairs: Vec<(String, String)>,
    used: RefCell<BTreeSet<String>>,
}

impl PairSettings {
    /// From pairs.
    pub fn new(pairs: Vec<(String, String)>) -> Self {
        Self {
            pairs: pairs
                .into_iter()
                .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                .collect(),
            used: RefCell::new(BTreeSet::new()),
        }
    }

    /// From `KEY=VALUE` tokens.
    pub fn parse(tokens: &[String]) -> MoptResult<Self> {
        let mut pairs = Vec::with_capacity(tokens.len());
        for token in tokens {
            let (k, v) = token.split_once('=').ok_or_else(|| {
                MoptError::parse(format!("a setting is KEY=VALUE, got '{token}'"))
            })?;
            pairs.push((k.to_string(), v.to_string()));
        }
        Ok(Self::new(pairs))
    }

    /// The value of `key`, if given.
    pub fn text(&self, key: &str) -> Option<String> {
        self.used.borrow_mut().insert(key.to_string());
        self.pairs
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    }

    /// A number.
    pub fn f64(&self, key: &str, default: f64) -> MoptResult<f64> {
        match self.text(key) {
            None => Ok(default),
            Some(v) => v
                .parse::<f64>()
                .ok()
                .filter(|x| x.is_finite())
                .ok_or_else(|| MoptError::parse(format!("{key}: '{v}' is not a number"))),
        }
    }

    /// A whole number.
    pub fn usize(&self, key: &str, default: usize) -> MoptResult<usize> {
        match self.text(key) {
            None => Ok(default),
            Some(v) => v
                .parse::<usize>()
                .map_err(|_| MoptError::parse(format!("{key}: '{v}' is not a whole number"))),
        }
    }

    /// On or off.
    pub fn flag(&self, key: &str, default: bool) -> MoptResult<bool> {
        match self.text(key).map(|v| v.to_ascii_lowercase()) {
            None => Ok(default),
            Some(v) => match v.as_str() {
                "on" | "true" | "yes" | "1" => Ok(true),
                "off" | "false" | "no" | "0" => Ok(false),
                _ => Err(MoptError::parse(format!("{key}: on or off, got '{v}'"))),
            },
        }
    }

    /// Force thresholds by name (`normal`, `tight`, `loose`).
    pub fn criteria(&self, key: &str) -> MoptResult<ConvergenceCriteria> {
        match self.text(key).map(|v| v.to_ascii_lowercase()) {
            None => Ok(ConvergenceCriteria::NORMAL),
            Some(v) => match v.as_str() {
                "normal" => Ok(ConvergenceCriteria::NORMAL),
                "tight" => Ok(ConvergenceCriteria::TIGHT),
                "loose" => Ok(ConvergenceCriteria::LOOSE),
                _ => Err(MoptError::parse(format!(
                    "{key}: normal, tight, or loose, got '{v}'"
                ))),
            },
        }
    }

    /// The keys given.
    pub fn keys(&self) -> Vec<String> {
        self.pairs.iter().map(|(k, _)| k.clone()).collect()
    }

    /// Refuse keys that are not among `keys` (before a run) or that `method` did not read
    /// (after it).
    pub fn finish(&self, method: &str, keys: &[(&str, &str)]) -> MoptResult<()> {
        let used = self.used.borrow();
        let unknown: Vec<&str> = self
            .pairs
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| !used.contains(*k) && !keys.iter().any(|(known, _)| known == k))
            .collect();
        if unknown.is_empty() {
            return Ok(());
        }
        let known: Vec<&str> = keys.iter().map(|(k, _)| *k).collect();
        Err(MoptError::parse(format!(
            "{method}: unknown setting(s) {} (its keys: {})",
            unknown.join(", "),
            known.join(", ")
        )))
    }
}

/// Counts of the evaluations of a run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PairCounts {
    /// Evaluations with a gradient.
    pub gradients: usize,
    /// Evaluations with a Hessian.
    pub hessians: usize,
    /// Evaluations of the energy alone.
    pub energies: usize,
}

/// What a method found.
#[derive(Debug, Clone)]
pub struct PairResult {
    /// The name of the method.
    pub method: String,
    /// Whether it ended on its criteria.
    pub converged: bool,
    /// Why it ended, in words.
    pub end: String,
    /// The estimates of transition states with their energies, the lowest first.
    pub transition_states: Vec<(Col<f64>, f64)>,
    /// The path with its energies, where the method makes one.
    pub path: Option<Vec<(Col<f64>, f64)>>,
    /// The end points as the method left them (two-ended methods: laid onto the first).
    pub ends: Vec<Col<f64>>,
    /// Evaluations of the surface.
    pub counts: PairCounts,
    /// Whether a `STOP` file ended the run.
    pub interrupted: bool,
}

impl PairResult {
    /// Sort the estimates by energy, the lowest first.
    pub fn ranked(mut self) -> Self {
        self.transition_states.sort_by(|a, b| a.1.total_cmp(&b.1));
        self
    }
}

/// A method of the registry.
pub trait PairMethod: Send + Sync {
    /// Its name in the registry.
    fn name(&self) -> &'static str;

    /// What it is (one line, with the publication).
    fn summary(&self) -> &'static str;

    /// Which end points it takes.
    fn ends(&self) -> Ends;

    /// Its settings: key and meaning with the default.
    fn keys(&self) -> &'static [(&'static str, &'static str)];

    /// Run it.
    fn run(
        &self,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
    ) -> MoptResult<PairResult>;

    /// Run with an observer for completed NEB iterations.
    fn run_observed(
        &self,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
        _observer: Option<&mut dyn crate::workflow::chain::ChainObserver>,
    ) -> MoptResult<PairResult> {
        self.run(input, settings, surface, run)
    }
}

/// `inner` with its evaluations counted.
pub struct CountingSurface<'a> {
    inner: &'a dyn SearchSurface,
    gradients: AtomicUsize,
    hessians: AtomicUsize,
    energies: AtomicUsize,
}

impl<'a> CountingSurface<'a> {
    /// Count the evaluations of `inner`.
    pub fn new(inner: &'a dyn SearchSurface) -> Self {
        Self {
            inner,
            gradients: AtomicUsize::new(0),
            hessians: AtomicUsize::new(0),
            energies: AtomicUsize::new(0),
        }
    }

    /// The counts so far.
    pub fn counts(&self) -> PairCounts {
        PairCounts {
            gradients: self.gradients.load(Ordering::Relaxed),
            hessians: self.hessians.load(Ordering::Relaxed),
            energies: self.energies.load(Ordering::Relaxed),
        }
    }
}

impl SearchSurface for CountingSurface<'_> {
    fn description(&self) -> String {
        self.inner.description()
    }

    fn two_surfaces(&self) -> bool {
        self.inner.two_surfaces()
    }

    fn in_order(&self) -> bool {
        self.inner.in_order()
    }

    fn evaluate(
        &self,
        x: &Col<f64>,
        image: usize,
        request: &PointRequest,
    ) -> MoptResult<SurfaceValue> {
        let counter = if request.hessian {
            &self.hessians
        } else if request.gradient {
            &self.gradients
        } else {
            &self.energies
        };
        counter.fetch_add(1, Ordering::Relaxed);
        self.inner.evaluate(x, image, request)
    }

    fn structure(&self, x: &Col<f64>, image: usize) -> Option<MolecularSystem> {
        self.inner.structure(x, image)
    }

    fn take_notes(&self) -> Vec<String> {
        self.inner.take_notes()
    }

    fn begin_iteration(&self, iteration: usize) {
        self.inner.begin_iteration(iteration);
    }

    fn model_hessian(
        &self,
        x: &Col<f64>,
        image: usize,
        gradient: &Col<f64>,
        model: &dyn crate::model_hessian::ModelHessian,
    ) -> MoptResult<Option<faer::Mat<f64>>> {
        self.inner.model_hessian(x, image, gradient, model)
    }
}

/// The registry.
#[derive(Clone)]
pub struct PairMethodRegistry {
    methods: Vec<Arc<dyn PairMethod>>,
}

impl Default for PairMethodRegistry {
    fn default() -> Self {
        let mut r = Self {
            methods: Vec::new(),
        };
        for m in super::registry_methods::standard() {
            r.register(m);
        }
        r
    }
}

impl PairMethodRegistry {
    /// Register (or replace) a method under its name.
    pub fn register(&mut self, method: Arc<dyn PairMethod>) {
        self.methods.retain(|m| m.name() != method.name());
        self.methods.push(method);
    }

    /// The names, in the order of registration.
    pub fn names(&self) -> Vec<&'static str> {
        self.methods.iter().map(|m| m.name()).collect()
    }

    /// The methods.
    pub fn methods(&self) -> &[Arc<dyn PairMethod>] {
        &self.methods
    }

    /// The method called `name` (`-` and `_` alike, any case).
    pub fn get(&self, name: &str) -> MoptResult<Arc<dyn PairMethod>> {
        let key = name.trim().to_ascii_lowercase().replace('-', "_");
        self.methods
            .iter()
            .find(|m| m.name() == key)
            .cloned()
            .ok_or_else(|| {
                MoptError::config(format!(
                    "no two-point method called '{name}' ({})",
                    self.names().join(", ")
                ))
            })
    }

    /// Run the method called `name`: its counts are those of this run, its estimates
    /// ranked by energy.
    pub fn run(
        &self,
        name: &str,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
    ) -> MoptResult<PairResult> {
        self.run_observed(name, input, settings, surface, run, None)
    }

    /// Run through the same registry checks and emit completed chain iterations.
    pub fn run_observed(
        &self,
        name: &str,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
        observer: Option<&mut dyn crate::workflow::chain::ChainObserver>,
    ) -> MoptResult<PairResult> {
        let method = self.get(name)?;
        let n = match method.ends() {
            Ends::Two => (2, 2),
            Ends::One => (1, 2),
            Ends::OneOrTwo => (1, 2),
            Ends::Path => (2, usize::MAX),
        };
        if input.points.len() < n.0 || input.points.len() > n.1 {
            return Err(MoptError::config(format!(
                "{}: {} end point(s), {} given",
                method.name(),
                method.ends().text(),
                input.points.len()
            )));
        }
        // The relaxation of the path afterwards (every method; DS-AFIR's lup=N of earlier
        // versions names it with the LUP chain).
        let mut relaxation = PathRelaxation::from_settings(settings, input)?;
        if method.name() == "dsafir" {
            if let Some(alias) = dsafir_alias(settings, input, run)? {
                if relaxation.is_some() {
                    return Err(MoptError::config(
                        "dsafir: lup=N and method_relax name the same relaxation of the path; \
                         give one of them",
                    ));
                }
                relaxation = Some(alias);
            }
        }
        // Unknown keys are refused before anything is calculated.
        let mut known: Vec<(&str, &str)> = method.keys().to_vec();
        known.extend_from_slice(RELAX_KEYS);
        if method.name() == "dsafir" {
            known.extend_from_slice(DSAFIR_ALIAS_KEYS);
        }
        settings.finish(method.name(), &known)?;
        if relaxation.is_some() {
            check_relaxation(surface)?;
        }
        let counting = CountingSurface::new(surface);
        let mut result = method.run_observed(input, settings, &counting, run, observer)?;
        if let (Some(relaxation), Some(path), false) =
            (&relaxation, result.path.clone(), result.interrupted)
        {
            let relaxed = relax_path(relaxation, &path, input, &counting, run)?;
            super::string_runs::write_path(&counting, run, "relaxed", &relaxed.path)?;
            result.path = Some(relaxed.path);
            result.transition_states = relaxed.maxima;
        } else if relaxation.is_some() && result.path.is_none() {
            return Err(MoptError::unsupported(format!(
                "method_relax: {} gives no path between two structures to relax",
                method.name()
            )));
        }
        result.counts = counting.counts();
        Ok(result.ranked())
    }
}
