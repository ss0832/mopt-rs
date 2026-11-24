//! Conformer searches by moves of the structure itself: every trial changes the conformer it
//! starts from ([`ConformerMove::propose`]) and minimizes the result. The trials, the choice
//! of the next start ([`SamplingPriority`]), the judgement of the minimized structure (its
//! bond graph, duplicates, rotamers, the energy window), and the termination are those of the
//! AFIR sampling ([`super::afir`]); only the way a trial makes its start differs, and that
//! way is a [`ConformerMove`] with a name in a [`MoveRegistry`]: the built-in ones and those a
//! user adds (a Rust type that implements the trait, or a Python callable; see "Your own
//! search methods" in `docs/source/searches.md`).
//!
//! - **Torsions** (`torsion`): random turns about rotatable bonds, the Monte Carlo multiple
//!   minimum search (G. Chang, W. C. Guida, W. C. Still, J. Am. Chem. Soc. 1989, 111, 4379).
//!   A rotatable bond is one outside every ring with further atoms on both sides; one to
//!   `move_bonds` of them are turned by angles uniform in (-180, 180] degrees, the smaller
//!   side moving.
//! - **Kicks** (`kick`): every atom is moved to a random point within `move_radius` of its
//!   position (M. Saunders, J. Am. Chem. Soc. 1987, 109, 3150).
//! - **Low modes** (`lowmode`): a step along one of the `move_modes` softest vibrations of
//!   the conformer (either way), long enough that the atom that moves most moves by
//!   `move_step` (after I. Kolossváry, W. C. Guida, J. Am. Chem. Soc. 1996, 118, 5011, whose
//!   search follows the mode until the energy has passed a barrier; here the length of the
//!   step is fixed). Every trial costs the Hessian of its conformer, which is kept for the
//!   next trial from the same one.
//!
//! Which move finds the conformers of a molecule with the fewest gradients is measured in
//! `docs/source/searches.md` ("Conformation search").
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use faer::Col;

use super::search::{Search, Trials};
use super::{
    CommonSettings, Level, RelaxJob, SearchEvent, SearchJobs, SearchObserver, SearchResult,
};
use crate::coord::graph::MolecularGraph;
use crate::core_infra::rng::Rng;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::SamplingPriority;

/// What a trial is made from.
pub struct MoveContext<'a> {
    /// The trial (from 0).
    pub trial: usize,
    /// The number of the conformer the trial starts from, in the order of discovery.
    pub conformer: usize,
    /// The jobs of the search (a move may ask for the Hessian of the conformer).
    pub jobs: &'a dyn SearchJobs,
}

/// The start of a trial.
#[derive(Debug, Clone)]
pub struct Proposal {
    /// The structure that is minimized.
    pub start: MolecularSystem,
    /// What was done, for the log.
    pub what: String,
}

/// A way to make the start of a trial from a conformer. The search minimizes what it
/// proposes on the surface of the command, rejects a structure whose bonds differ from those
/// of the first conformer (in the mode of conformers), and keeps what is new.
pub trait ConformerMove: Send {
    /// The name of the move.
    fn name(&self) -> &str;

    /// Told the first conformer (the optimized input) and the scale of the covalent radii of
    /// the bond graph before the trials; a line for the log may be returned.
    fn prepare(
        &mut self,
        _first: &MolecularSystem,
        _bond_scale: f64,
    ) -> MoptResult<Option<String>> {
        Ok(None)
    }

    /// The start of the trial `context.trial` from `current`, made with the random numbers
    /// `rng` (those of the search: the same seed gives the same search); `None` when the move
    /// has nothing more to try, which ends the search.
    fn propose(
        &mut self,
        current: &MolecularSystem,
        context: &MoveContext<'_>,
        rng: &mut Rng,
    ) -> MoptResult<Option<Proposal>>;
}

/// The settings of a move as given (`-cs_opt move_KEY=VALUE`: the keys without `move_`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoveOptions(pub BTreeMap<String, String>);

impl MoveOptions {
    /// The number of `key`, or `default`.
    pub fn number(&self, key: &str, default: f64) -> MoptResult<f64> {
        match self.0.get(key) {
            None => Ok(default),
            Some(text) => text
                .trim()
                .parse::<f64>()
                .map_err(|_| MoptError::parse(format!("-cs_opt move_{key}={text}: not a number"))),
        }
    }

    /// The whole number of `key`, or `default`.
    pub fn whole(&self, key: &str, default: usize) -> MoptResult<usize> {
        match self.0.get(key) {
            None => Ok(default),
            Some(text) => text.trim().parse::<usize>().map_err(|_| {
                MoptError::parse(format!("-cs_opt move_{key}={text}: not a whole number"))
            }),
        }
    }

    /// Refuses keys other than `known` (a misspelt setting is not ignored).
    pub fn only(&self, name: &str, known: &[&str]) -> MoptResult<()> {
        match self.0.keys().find(|k| !known.contains(&k.as_str())) {
            None => Ok(()),
            Some(k) => Err(MoptError::parse(format!(
                "-cs_opt move_{k}: method={name} has the settings {}",
                if known.is_empty() {
                    "none".to_string()
                } else {
                    known
                        .iter()
                        .map(|k| format!("move_{k}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            ))),
        }
    }
}

/// Makes a move from its settings.
pub type MoveFactory =
    Arc<dyn Fn(&MoveOptions) -> MoptResult<Box<dyn ConformerMove>> + Send + Sync>;

/// Names of the strategies of the command, which no move takes.
const STRATEGIES: [&str; 6] = ["afir", "multioptpy", "imtd", "imtd-gc", "imtd_gc", "crest"];

/// The moves by name: the built-in ones and those a user registers.
#[derive(Clone)]
pub struct MoveRegistry {
    factories: BTreeMap<String, MoveFactory>,
}

impl Default for MoveRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl MoveRegistry {
    /// The built-in moves: `torsion`, `kick`, `lowmode`.
    pub fn builtin() -> Self {
        let mut factories: BTreeMap<String, MoveFactory> = BTreeMap::new();
        factories.insert(
            "torsion".to_string(),
            Arc::new(|o| Ok(Box::new(Torsions::new(o)?) as Box<dyn ConformerMove>)),
        );
        factories.insert(
            "kick".to_string(),
            Arc::new(|o| Ok(Box::new(Kick::new(o)?) as Box<dyn ConformerMove>)),
        );
        factories.insert(
            "lowmode".to_string(),
            Arc::new(|o| Ok(Box::new(LowModes::new(o)?) as Box<dyn ConformerMove>)),
        );
        Self { factories }
    }

    /// Add the move `name` (`-cs_opt method=NAME`); the names of the strategies of the command
    /// (`afir`, `imtd`) and of moves that are there already are refused.
    pub fn register(&mut self, name: &str, factory: MoveFactory) -> MoptResult<()> {
        let key = name.trim().to_ascii_lowercase();
        if key.is_empty() || STRATEGIES.contains(&key.as_str()) || self.factories.contains_key(&key)
        {
            return Err(MoptError::config(format!(
                "a conformer move cannot be registered as '{name}' (empty, a strategy of the \
                 command, or a name that is taken)"
            )));
        }
        self.factories.insert(key, factory);
        Ok(())
    }

    /// Whether `name` is a move.
    pub fn contains(&self, name: &str) -> bool {
        self.factories
            .contains_key(&name.trim().to_ascii_lowercase())
    }

    /// The names of the moves.
    pub fn names(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// The move `name` with the settings `options`.
    pub fn create(&self, name: &str, options: &MoveOptions) -> MoptResult<Box<dyn ConformerMove>> {
        let factory = self
            .factories
            .get(&name.trim().to_ascii_lowercase())
            .ok_or_else(|| {
                MoptError::parse(format!(
                    "-cs_opt method={name}: afir, imtd, or a move ({})",
                    self.names().join(", ")
                ))
            })?;
        factory(options)
    }
}

/// The trials and the termination of a search by moves (those of the AFIR sampling).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrialSettings {
    /// Most trials.
    pub max_samples: usize,
    /// The lowest conformers watched by the termination.
    pub rank: usize,
    /// Trials without a change of them that end the search.
    pub patience: usize,
    /// Every trial starts from the first conformer.
    pub no_stochastic: bool,
}

/// The bonds of `graph` outside every ring with further atoms on both sides, each with the
/// atoms of its smaller side (the side of the second atom of the pair).
pub fn rotatable_bonds(graph: &MolecularGraph) -> Vec<([usize; 2], Vec<usize>)> {
    let n = graph.n_atoms();
    let mut out = Vec::new();
    for [i, j] in graph.bonds() {
        if graph.neighbors(i).len() < 2 || graph.neighbors(j).len() < 2 {
            continue;
        }
        // The atoms reached from j without the bond to i.
        let mut seen = vec![false; n];
        seen[j] = true;
        let mut stack = vec![j];
        let mut ring = false;
        while let Some(a) = stack.pop() {
            for &b in graph.neighbors(a) {
                if a == j && b == i {
                    continue;
                }
                if b == i {
                    ring = true;
                }
                if !seen[b] {
                    seen[b] = true;
                    stack.push(b);
                }
            }
        }
        if ring {
            continue;
        }
        let side: Vec<usize> = (0..n).filter(|&a| seen[a]).collect();
        if 2 * side.len() <= n {
            out.push(([i, j], side));
        } else {
            let other: Vec<usize> = (0..n).filter(|&a| !seen[a]).collect();
            out.push(([j, i], other));
        }
    }
    out
}

/// `x` with the atoms of `side` turned by `angle` (rad) about the axis from atom `i` to atom
/// `j`.
fn turn(x: &mut Col<f64>, [i, j]: [usize; 2], side: &[usize], angle: f64) {
    let at = |x: &Col<f64>, a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
    let (pi, pj) = (at(x, i), at(x, j));
    let mut axis = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if length < 1e-12 {
        return;
    }
    for a in &mut axis {
        *a /= length;
    }
    let (c, s) = (angle.cos(), angle.sin());
    for &a in side {
        let p = at(x, a);
        let v = [p[0] - pj[0], p[1] - pj[1], p[2] - pj[2]];
        let dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
        let cross = [
            axis[1] * v[2] - axis[2] * v[1],
            axis[2] * v[0] - axis[0] * v[2],
            axis[0] * v[1] - axis[1] * v[0],
        ];
        // Rodrigues' rotation.
        for k in 0..3 {
            x[3 * a + k] = pj[k] + v[k] * c + cross[k] * s + axis[k] * dot * (1.0 - c);
        }
    }
}

/// Random turns about rotatable bonds (`torsion`).
struct Torsions {
    max_bonds: usize,
    bonds: Vec<([usize; 2], Vec<usize>)>,
}

impl Torsions {
    fn new(options: &MoveOptions) -> MoptResult<Self> {
        options.only("torsion", &["bonds"])?;
        Ok(Self {
            max_bonds: options.whole("bonds", 3)?.max(1),
            bonds: Vec::new(),
        })
    }
}

impl ConformerMove for Torsions {
    fn name(&self) -> &str {
        "torsion"
    }

    fn prepare(&mut self, first: &MolecularSystem, bond_scale: f64) -> MoptResult<Option<String>> {
        self.bonds = rotatable_bonds(&MolecularGraph::from_system(first, bond_scale));
        if self.bonds.is_empty() {
            return Err(MoptError::config(
                "method=torsion: the structure has no rotatable bond (a bond outside every \
                 ring with further atoms on both sides); method=kick and method=lowmode move \
                 such structures",
            ));
        }
        Ok(Some(format!(
            "Rotatable bonds: {} (one to {} turned in a trial)",
            self.bonds.len(),
            self.max_bonds.min(self.bonds.len())
        )))
    }

    fn propose(
        &mut self,
        current: &MolecularSystem,
        context: &MoveContext<'_>,
        rng: &mut Rng,
    ) -> MoptResult<Option<Proposal>> {
        // The side that turns holds no frozen atom: the other side where it does, and a bond
        // with frozen atoms on both sides is not turned.
        let frozen = context.jobs.frozen_atoms();
        let n = current.n_atoms();
        let bonds: Vec<([usize; 2], Vec<usize>)> = self
            .bonds
            .iter()
            .filter_map(|([i, j], side)| {
                let holds = |s: &[usize]| s.iter().any(|a| frozen.contains(a));
                if !holds(side) {
                    return Some(([*i, *j], side.clone()));
                }
                let other: Vec<usize> = (0..n)
                    .filter(|a| a != i && a != j && !side.contains(a))
                    .collect();
                (!holds(&other)).then_some(([*j, *i], other))
            })
            .collect();
        if bonds.is_empty() {
            return Err(MoptError::config(
                "method=torsion: every rotatable bond has frozen atoms (-fix) on both of its \
                 sides, so no turn leaves them in place",
            ));
        }
        let mut x = current.coords_flat_bohr();
        let count = 1 + rng.gen_range_usize(self.max_bonds.clamp(1, bonds.len()));
        let mut order: Vec<usize> = (0..bonds.len()).collect();
        rng.shuffle(&mut order);
        let mut turned = Vec::with_capacity(count);
        for &k in order.iter().take(count) {
            let (pair, side) = &bonds[k];
            let angle = rng.uniform(-std::f64::consts::PI, std::f64::consts::PI);
            turn(&mut x, *pair, side, angle);
            turned.push(format!(
                "{}-{} by {:.0} degrees",
                pair[0] + 1,
                pair[1] + 1,
                angle.to_degrees()
            ));
        }
        Ok(Some(Proposal {
            start: current.with_flat_coords_bohr(&x)?,
            what: format!("turns about {}", turned.join(", ")),
        }))
    }
}

/// Every atom moved within a radius (`kick`).
struct Kick {
    radius: f64,
}

impl Kick {
    fn new(options: &MoveOptions) -> MoptResult<Self> {
        options.only("kick", &["radius"])?;
        let angstrom = options.number("radius", 0.7)?;
        if !(angstrom > 0.0 && angstrom.is_finite()) {
            return Err(MoptError::parse(
                "-cs_opt move_radius: the radius of the kicks is positive (Angstrom)",
            ));
        }
        Ok(Self {
            radius: angstrom * Units::ANGSTROM_TO_BOHR,
        })
    }
}

impl ConformerMove for Kick {
    fn name(&self) -> &str {
        "kick"
    }

    fn propose(
        &mut self,
        current: &MolecularSystem,
        context: &MoveContext<'_>,
        rng: &mut Rng,
    ) -> MoptResult<Option<Proposal>> {
        let frozen = context.jobs.frozen_atoms();
        let mut x = current.coords_flat_bohr();
        for a in 0..current.n_atoms() {
            // The draws are made for every atom, so that the kicks of the others do not
            // depend on which atoms are frozen.
            let u = rng.unit_vector_3d();
            let r = self.radius * rng.next_f64().cbrt();
            if frozen.contains(&a) {
                continue;
            }
            for k in 0..3 {
                x[3 * a + k] += r * u[k];
            }
        }
        Ok(Some(Proposal {
            start: current.with_flat_coords_bohr(&x)?,
            what: format!(
                "kicks within {:.2} Angstrom",
                self.radius * Units::BOHR_TO_ANGSTROM
            ),
        }))
    }
}

/// A step along one of the softest vibrations (`lowmode`).
struct LowModes {
    modes: usize,
    step: f64,
    /// The softest vibrations of the conformers the search has started from.
    of: HashMap<usize, Vec<Col<f64>>>,
}

impl LowModes {
    fn new(options: &MoveOptions) -> MoptResult<Self> {
        options.only("lowmode", &["modes", "step"])?;
        let angstrom = options.number("step", 1.5)?;
        if !(angstrom > 0.0 && angstrom.is_finite()) {
            return Err(MoptError::parse(
                "-cs_opt move_step: the step along a vibration is positive (Angstrom)",
            ));
        }
        Ok(Self {
            modes: options.whole("modes", 6)?.max(1),
            step: angstrom * Units::ANGSTROM_TO_BOHR,
            of: HashMap::new(),
        })
    }
}

impl ConformerMove for LowModes {
    fn name(&self) -> &str {
        "lowmode"
    }

    fn propose(
        &mut self,
        current: &MolecularSystem,
        context: &MoveContext<'_>,
        rng: &mut Rng,
    ) -> MoptResult<Option<Proposal>> {
        if !self.of.contains_key(&context.conformer) {
            let hessian = context.jobs.hessian(current)?;
            let modes = crate::vib::normal_modes(current, &hessian, true)?;
            let soft: Vec<Col<f64>> = modes
                .displacements
                .iter()
                .zip(&modes.wavenumbers)
                .filter(|(_, w)| **w > 0.0)
                .take(self.modes)
                .map(|(d, _)| d.clone())
                .collect();
            self.of.insert(context.conformer, soft);
        }
        let soft = self
            .of
            .get(&context.conformer)
            .map_or(&[][..], Vec::as_slice);
        if soft.is_empty() {
            return Err(MoptError::calc(
                "method=lowmode: the conformer has no vibration to follow",
            ));
        }
        let k = rng.gen_range_usize(soft.len());
        let sign = if rng.next_f64() < 0.5 { 1.0 } else { -1.0 };
        // Frozen atoms do not move along the vibration.
        let frozen = context.jobs.frozen_atoms();
        let mut d = soft[k].clone();
        for &a in &frozen {
            for c in 0..3 {
                if 3 * a + c < d.nrows() {
                    d[3 * a + c] = 0.0;
                }
            }
        }
        let d = &d;
        let largest = (0..current.n_atoms())
            .map(|a| (0..3).map(|c| d[3 * a + c].powi(2)).sum::<f64>().sqrt())
            .fold(0.0_f64, f64::max);
        let mut x = current.coords_flat_bohr();
        if largest > 0.0 {
            for i in 0..x.nrows() {
                x[i] += sign * self.step / largest * d[i];
            }
        }
        Ok(Some(Proposal {
            start: current.with_flat_coords_bohr(&x)?,
            what: format!(
                "a step of {:+.2} Angstrom along the vibration {}",
                sign * self.step * Units::BOHR_TO_ANGSTROM,
                k + 1
            ),
        }))
    }
}

/// The search by the move `mover` from `input`; `None` when a `STOP` file ended it before
/// the first conformer.
pub fn run(
    input: &MolecularSystem,
    mover: &mut dyn ConformerMove,
    settings: &TrialSettings,
    common: &CommonSettings,
    priority: &mut dyn SamplingPriority,
    jobs: &dyn SearchJobs,
    observer: &mut dyn SearchObserver,
) -> MoptResult<Option<SearchResult>> {
    let rng = Rng::from_seed(common.seed);
    let first = RelaxJob::minimize("input", Level::Final);
    let Some(mut search) = Search::start(jobs, observer, common, input, &first)? else {
        return Ok(None);
    };
    let start = search.ensemble.members()[0].system.clone();
    if let Some(note) = mover.prepare(&start, common.bond_scale)? {
        search.note(note)?;
    }
    let mut trials = Trials::new(
        settings.rank,
        settings.patience,
        settings.max_samples,
        settings.no_stochastic,
        priority,
        rng,
    );
    let mut reason = "Max samples are reached.".to_string();
    for trial in 0..settings.max_samples {
        if jobs.stop_requested() {
            search.interrupted = true;
            reason = "The stop signal is detected.".to_string();
            break;
        }
        let current = trials.current;
        let from = search.ensemble.members()[current].system.clone();
        let context = MoveContext {
            trial,
            conformer: current,
            jobs,
        };
        let Some(proposal) = mover.propose(&from, &context, trials.rng())? else {
            reason = format!("The move {} has nothing more to try.", mover.name());
            break;
        };
        if proposal.start.elements != from.elements {
            return Err(MoptError::calc(format!(
                "the move {} proposed a structure with other atoms than the conformer",
                mover.name()
            )));
        }
        search.observer.on_event(&SearchEvent::Sample {
            trial,
            from: current,
            what: proposal.what.clone(),
        })?;
        let origin = format!("sample {trial}: {}", proposal.what);
        let low_before = search.lowest_energy();
        let relaxed = jobs.relax(
            &proposal.start,
            &RelaxJob::minimize(origin.clone(), Level::Final),
        );
        let fate = search.consider(relaxed, &origin)?;
        if search.interrupted {
            reason = "The stop signal is detected.".to_string();
            break;
        }
        if let Some(r) = trials.after(&mut search, trial, &fate, low_before)? {
            reason = r;
            break;
        }
    }
    Ok(Some(search.finish(reason)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::domain::{AtomId, SpinCharge};
    use crate::io::xyz::parse_xyz_str;

    const BUTANE: &str = "14\nbutane\nC -1.9 0.4 0.0\nC -0.6 -0.4 0.0\nC 0.6 0.4 0.0\n\
C 1.9 -0.4 0.0\nH -2.0 1.0 0.9\nH -2.0 1.0 -0.9\nH -2.8 -0.3 0.0\nH -0.6 -1.0 0.9\n\
H -0.6 -1.0 -0.9\nH 0.6 1.0 0.9\nH 0.6 1.0 -0.9\nH 2.0 -1.0 0.9\nH 2.0 -1.0 -0.9\nH 2.8 0.3 0.0\n";

    #[test]
    fn the_rotatable_bonds_are_those_outside_rings_and_a_turn_keeps_the_distances() {
        let sys = parse_xyz_str(BUTANE, SpinCharge::default(), None).unwrap();
        let graph = MolecularGraph::from_system(&sys, 1.1);
        let bonds = rotatable_bonds(&graph);
        // The three C-C bonds; the C-H bonds end at an atom with one neighbour.
        let mut pairs: Vec<[usize; 2]> = bonds
            .iter()
            .map(|(p, _)| [p[0].min(p[1]), p[0].max(p[1])])
            .collect();
        pairs.sort_unstable();
        assert_eq!(pairs, vec![[0, 1], [1, 2], [2, 3]]);
        // The smaller side of the middle bond: half of the molecule.
        let middle = bonds
            .iter()
            .find(|(p, _)| p.contains(&1) && p.contains(&2))
            .unwrap();
        assert_eq!(middle.1.len(), 7);
        // A turn about the middle bond moves the dihedral angle and no bonded distance.
        let mut x = sys.coords_flat_bohr();
        turn(&mut x, middle.0, &middle.1, 1.0);
        let turned = sys.with_flat_coords_bohr(&x).unwrap();
        for [i, j] in graph.bonds() {
            let (a, b) = (AtomId::new(i), AtomId::new(j));
            assert!((sys.distance(a, b).value() - turned.distance(a, b).value()).abs() < 1e-10);
        }
        let ends = (AtomId::new(0), AtomId::new(3));
        let before = sys.distance(ends.0, ends.1).value();
        let after = turned.distance(ends.0, ends.1).value();
        assert!((before - after).abs() > 0.1);
        // Cyclopropane has none.
        let ring = parse_xyz_str(
            "9\ncyclopropane\nC 0.0 0.87 0.0\nC 0.75 -0.43 0.0\nC -0.75 -0.43 0.0\n\
H 0.0 1.46 0.91\nH 0.0 1.46 -0.91\nH 1.26 -0.73 0.91\nH 1.26 -0.73 -0.91\n\
H -1.26 -0.73 0.91\nH -1.26 -0.73 -0.91\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert!(rotatable_bonds(&MolecularGraph::from_system(&ring, 1.1)).is_empty());
    }

    /// A move of a user: the first atom shifted along x, twice.
    struct Shift(f64);

    impl ConformerMove for Shift {
        fn name(&self) -> &str {
            "shift"
        }

        fn propose(
            &mut self,
            current: &MolecularSystem,
            context: &MoveContext<'_>,
            _rng: &mut Rng,
        ) -> MoptResult<Option<Proposal>> {
            if context.trial >= 2 {
                return Ok(None);
            }
            let mut x = current.coords_flat_bohr();
            x[0] += self.0;
            Ok(Some(Proposal {
                start: current.with_flat_coords_bohr(&x)?,
                what: format!("shift by {}", self.0),
            }))
        }
    }

    #[test]
    fn the_registry_has_the_built_in_moves_and_takes_those_of_users() {
        let mut registry = MoveRegistry::builtin();
        assert_eq!(registry.names(), vec!["kick", "lowmode", "torsion"]);
        // Settings: read by the move, and a misspelt one is refused.
        let mut options = MoveOptions::default();
        options.0.insert("radius".to_string(), "0.4".to_string());
        assert!(registry.create("kick", &options).is_ok());
        assert!(registry.create("torsion", &options).is_err());
        assert!(registry.create("nothing", &MoveOptions::default()).is_err());
        // A move of a user by name; the names of the strategies of the command stay theirs.
        registry
            .register(
                "shift",
                Arc::new(|o| {
                    o.only("shift", &["by"])?;
                    Ok(Box::new(Shift(o.number("by", 0.1)?)) as Box<dyn ConformerMove>)
                }),
            )
            .unwrap();
        assert!(registry.contains("Shift"));
        let made = registry.create("shift", &MoveOptions::default()).unwrap();
        assert_eq!(made.name(), "shift");
        let refused: MoveFactory = Arc::new(|_| Err(MoptError::config("not made")));
        assert!(registry.register("afir", Arc::clone(&refused)).is_err());
        assert!(registry.register("kick", refused).is_err());
    }
}
