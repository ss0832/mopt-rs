//! A calculator that stays with its electronic states from structure to structure.
//!
//! Without this, `EvalRequest::excited_state = N` is the state number `N` in the order of
//! the energies at every structure, so that a run changes to another state where two
//! states change places. [`FollowedStates`] gives the number another meaning: state `N` is
//! the state that was number `N` at the first structure it was asked for, and at every
//! later structure the state that it has become. What a state has become is decided by the
//! overlaps of the states of the new structure with those of the nearest structure seen
//! before ([`StateCharacters::squared_overlaps`]); all the states that are followed are
//! assigned at once (each to another state of the new structure, with the largest sum of
//! squared overlaps), so that two of them never end on the same state.
//!
//! Where two states are degenerate (at a conical intersection) the two are only defined
//! as a pair; either of them may then be given either number, and the pair stays the pair.
//! A squared overlap below one half is reported: the state was not found again with
//! certainty (it may have left the window of the lowest states that are looked at, or
//! mixed with another).
//!
//! The states of another charge or multiplicity are other states: every structure is
//! remembered with its charge and multiplicity, and a state is followed only among the
//! structures of its own (a singlet and a triplet asked for at the same coordinates keep
//! their own numbers).
//!
//! The calculator must hand out its states ([`Calculator::state_characters`]); the
//! built-in GFN0-xTB and PySCF (restricted Tamm-Dancoff) do.
//!
//! A restarted run takes its first evaluations from the journal, without the calculator.
//! The journal therefore records which state every evaluation was for and tells it back
//! ([`Calculator::followed_state`], [`Calculator::recall_state`]), so that the first
//! structure that is computed anew is compared with the last recorded one.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use faer::Mat;

use super::calculator::{Calculator, SiteTerms};
use super::point_charges::{EmbeddedOutput, PointCharges};
use super::states::StateCharacters;
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Orbitals, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::math::assignment::linear_sum_assignment;

/// Structures that are remembered as references.
const REMEMBERED: usize = 64;
/// States above the highest followed one that are looked at.
const WINDOW: usize = 6;
/// Squared overlap below which a state counts as not found again with certainty.
const FOUND: f64 = 0.5;
/// Share of the occupied orbitals that must stay occupied between two structures.
const KEPT: f64 = 0.9;
/// Two structures are the same when no coordinate differs by more than this (Bohr).
const SAME_BOHR: f64 = 1.0e-12;

/// A structure whose states are known, for one charge and multiplicity.
///
/// The states of another charge or multiplicity at the same coordinates are other states
/// (the excited states of a triplet are not those of the singlet): each pair has its own
/// records, and a number is followed only among the records of its own pair.
struct Seen {
    /// The charge and multiplicity the states are of.
    spin_charge: SpinCharge,
    x: Vec<f64>,
    /// `None` for a structure that the journal told about and that was not computed here.
    characters: Option<StateCharacters>,
    /// The index of the state (0: the lowest) of every followed number at this structure.
    roots: BTreeMap<usize, usize>,
}

#[derive(Default)]
struct Memory {
    seen: VecDeque<Seen>,
    notes: Vec<String>,
}

impl Memory {
    fn at(&mut self, spin_charge: SpinCharge, x: &[f64]) -> Option<&mut Seen> {
        self.seen.iter_mut().find(|s| {
            s.spin_charge == spin_charge
                && s.x.len() == x.len()
                && s.x.iter().zip(x).all(|(a, b)| (a - b).abs() <= SAME_BOHR)
        })
    }

    fn nearest(&self, spin_charge: SpinCharge, x: &[f64]) -> Option<usize> {
        let distance = |s: &Seen| -> f64 {
            if s.spin_charge != spin_charge || s.x.len() != x.len() {
                return f64::INFINITY;
            }
            s.x.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum()
        };
        (0..self.seen.len())
            .filter(|k| distance(&self.seen[*k]).is_finite())
            .min_by(|a, b| distance(&self.seen[*a]).total_cmp(&distance(&self.seen[*b])))
    }

    fn remember(&mut self, seen: Seen) {
        if self.seen.len() == REMEMBERED {
            self.seen.pop_front();
        }
        self.seen.push_back(seen);
    }

    fn highest(&self, spin_charge: SpinCharge, label: usize) -> usize {
        self.seen
            .iter()
            .filter(|s| s.spin_charge == spin_charge)
            .flat_map(|s| s.roots.keys().copied())
            .chain([label])
            .max()
            .unwrap_or(label)
    }
}

/// `inner` with its excited states followed from structure to structure.
pub struct FollowedStates {
    inner: Box<dyn Calculator>,
    memory: Mutex<Memory>,
}

impl FollowedStates {
    /// `inner`, whose states are followed.
    pub fn new(inner: Box<dyn Calculator>) -> Self {
        Self {
            inner,
            memory: Mutex::new(Memory::default()),
        }
    }

    fn states(&self, sys: &MolecularSystem, count: usize) -> MoptResult<StateCharacters> {
        self.inner.state_characters(sys, count)?.ok_or_else(|| {
            MoptError::unsupported(format!(
                "calculator '{}' does not hand out its excited states in a form that can be \
                 compared between structures; its states cannot be followed (they are counted \
                 by their energies without the option)",
                self.inner.name()
            ))
        })
    }

    /// The index (0: the lowest) of the state at `sys` that the followed state `label` is.
    fn root(&self, sys: &MolecularSystem, label: usize) -> MoptResult<usize> {
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let spin = sys.spin_charge;
        let mut memory = self
            .memory
            .lock()
            .map_err(|_| MoptError::calc("the memory of the followed states is not available"))?;
        if let Some(root) = memory
            .at(spin, &x)
            .and_then(|s| s.roots.get(&label).copied())
        {
            return Ok(root);
        }
        let count = memory.highest(spin, label) + WINDOW;
        let here = self.states(sys, count)?;
        if label > here.count() {
            return Err(MoptError::config(format!(
                "excited state {label}: the structure has {} states",
                here.count()
            )));
        }
        let mut roots = BTreeMap::new();
        if let Some(k) = memory.nearest(spin, &x) {
            if memory.seen[k].characters.is_none() {
                // A structure of the journal: its states are computed now.
                let flat = faer::Col::from_fn(memory.seen[k].x.len(), |i| memory.seen[k].x[i]);
                let old = sys.with_flat_coords_bohr(&flat)?;
                memory.seen[k].characters = Some(self.states(&old, count)?);
            }
            let reference = &memory.seen[k];
            let before = reference.characters.as_ref().ok_or_else(|| {
                MoptError::calc("the states of the reference structure are missing")
            })?;
            let (overlaps, kept) = before.squared_overlaps(&here)?;
            let labels: Vec<usize> = reference
                .roots
                .iter()
                .filter(|(_, root)| **root < before.count())
                .map(|(label, _)| *label)
                .collect();
            let mut notes = Vec::new();
            if kept.is_nan() || kept < KEPT {
                notes.push(format!(
                    "the occupied orbitals change between neighbouring structures (only \
                     {kept:.3} of them is kept): the reference of the excited states changes \
                     its character, and the states may not be the ones followed so far"
                ));
            }
            if !labels.is_empty() {
                let cost = Mat::from_fn(labels.len(), here.count(), |r, c| {
                    -overlaps[(reference.roots[&labels[r]], c)]
                });
                let columns = linear_sum_assignment(&cost)?;
                for (number, column) in labels.iter().zip(columns) {
                    let was = reference.roots[number];
                    let overlap = overlaps[(was, column)];
                    if overlap.is_nan() || overlap < FOUND {
                        notes.push(format!(
                            "electronic state {number} was not found again with certainty \
                             (squared overlap {overlap:.3} with state {} here)",
                            column + 1
                        ));
                    } else if column != was {
                        notes.push(format!(
                            "electronic state {number} (as counted where it was asked for \
                             first) is state {} in the order of the energies here (state {} at \
                             the structure before; squared overlap {overlap:.3})",
                            column + 1,
                            was + 1
                        ));
                    }
                    roots.insert(*number, column);
                }
            }
            memory.notes.extend(notes);
        }
        if !roots.contains_key(&label) {
            // Asked for the first time: counted by its energy here.
            if let Some((other, _)) = roots.iter().find(|(_, root)| **root == label - 1) {
                let other = *other;
                memory.notes.push(format!(
                    "electronic state {label} is asked for the first time where the followed \
                     state {other} is state {label} in the order of the energies: the two are \
                     the same state from here on"
                ));
            }
            roots.insert(label, label - 1);
        }
        let root = roots[&label];
        if let Some(seen) = memory.at(spin, &x) {
            seen.roots.extend(roots);
            seen.characters.get_or_insert(here);
        } else {
            memory.remember(Seen {
                spin_charge: spin,
                x,
                characters: Some(here),
                roots,
            });
        }
        Ok(root)
    }

    fn request(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalRequest> {
        if req.excited_state == 0 {
            return Ok(*req);
        }
        Ok(EvalRequest {
            excited_state: self.root(sys, req.excited_state)? + 1,
            ..*req
        })
    }
}

impl Calculator for FollowedStates {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> String {
        format!(
            "{}; excited states followed between structures by their overlaps",
            self.inner.description()
        )
    }

    fn citations(&self) -> Vec<String> {
        self.inner.citations()
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        self.inner.evaluate(sys, &self.request(sys, req)?)
    }

    fn supports_pbc(&self) -> bool {
        self.inner.supports_pbc()
    }

    fn is_rigid_body_invariant(&self) -> bool {
        self.inner.is_rigid_body_invariant()
    }

    fn supports_excited_states(&self) -> bool {
        self.inner.supports_excited_states()
    }

    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        self.inner.excitation_energies(sys)
    }

    fn state_characters(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<StateCharacters>> {
        self.inner.state_characters(sys, count)
    }

    fn transitions(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<Vec<crate::pes::states::Transition>>> {
        self.inner.transitions(sys, count)
    }

    fn followed_state(&self, sys: &MolecularSystem, label: usize) -> Option<usize> {
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let mut memory = self.memory.lock().ok()?;
        memory
            .at(sys.spin_charge, &x)
            .and_then(|s| s.roots.get(&label).copied())
    }

    fn recall_state(&self, sys: &MolecularSystem, label: usize, root: usize) {
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let Ok(mut memory) = self.memory.lock() else {
            return;
        };
        if let Some(seen) = memory.at(sys.spin_charge, &x) {
            seen.roots.insert(label, root);
        } else {
            memory.remember(Seen {
                spin_charge: sys.spin_charge,
                x,
                characters: None,
                roots: BTreeMap::from([(label, root)]),
            });
        }
    }

    fn evaluates_in_order(&self) -> bool {
        true
    }

    fn relaxed_environment(&self, sys: &MolecularSystem) -> Option<Vec<f64>> {
        self.inner.relaxed_environment(sys)
    }

    fn recall_environment(&self, sys: &MolecularSystem, environment: &[f64]) {
        self.inner.recall_environment(sys, environment);
    }

    fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Option<Orbitals>> {
        self.inner.orbitals(sys)
    }

    fn supports_point_charges(&self) -> bool {
        self.inner.supports_point_charges()
    }

    fn evaluate_in_charges(
        &self,
        sys: &MolecularSystem,
        req: &EvalRequest,
        charges: &PointCharges,
    ) -> MoptResult<EmbeddedOutput> {
        self.inner
            .evaluate_in_charges(sys, &self.request(sys, req)?, charges)
    }

    fn embedding_charges(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        self.inner.embedding_charges(sys)
    }

    fn uses_electrons(&self) -> bool {
        self.inner.uses_electrons()
    }

    fn checks_stability(&self) -> bool {
        self.inner.checks_stability()
    }

    fn take_warnings(&self) -> Vec<String> {
        let mut out = self.inner.take_warnings();
        if let Ok(mut memory) = self.memory.lock() {
            out.append(&mut memory.notes);
        }
        out
    }

    fn site_terms(&self, sys: &MolecularSystem) -> Option<MoptResult<SiteTerms>> {
        self.inner.site_terms(sys)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{Element, Quantity, SpinCharge};
    use faer::Col;

    /// Two atoms at the distance `d` with two excited states whose energies cross at
    /// `d = 2`: `A = 0.1 + 0.05 d` (the excitation into the first empty orbital) and `B =
    /// 0.3 - 0.05 d` (into the second).
    struct TwoStates;

    fn distance(sys: &MolecularSystem) -> f64 {
        let x = sys.coords_flat_bohr();
        x[3] - x[0]
    }

    fn energies(d: f64) -> [f64; 2] {
        [0.1 + 0.05 * d, 0.3 - 0.05 * d]
    }

    impl Calculator for TwoStates {
        fn name(&self) -> &str {
            "two_states"
        }

        fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
            let mut e = energies(distance(sys)).to_vec();
            e.sort_by(f64::total_cmp);
            let energy = match req.excited_state {
                0 => 0.0,
                k => e[k - 1],
            };
            Ok(EvalOutput::energy_gradient(
                Quantity::new(energy),
                Col::zeros(6),
            ))
        }

        fn state_characters(
            &self,
            sys: &MolecularSystem,
            count: usize,
        ) -> MoptResult<Option<StateCharacters>> {
            let [a, b] = energies(distance(sys));
            // The lower state first.
            let (first, values) = if a <= b {
                (0, vec![a, b])
            } else {
                (1, vec![b, a])
            };
            let amplitudes = Mat::from_fn(
                2,
                2,
                |p, k| {
                    if (k == 0) == (p == first) {
                        1.0
                    } else {
                        0.0
                    }
                },
            );
            Ok(Some(
                StateCharacters::new(Mat::<f64>::identity(3, 3), 1, values, amplitudes)?
                    .lowest(count),
            ))
        }
    }

    fn at(d: f64) -> MolecularSystem {
        let h = Element::parse_token("H").unwrap();
        let flat = Col::from_fn(6, |k| if k == 3 { d } else { 0.0 });
        MolecularSystem::from_flat_bohr(vec![h, h], &flat, SpinCharge::default(), None).unwrap()
    }

    fn state(k: usize) -> EvalRequest {
        EvalRequest {
            excited_state: k,
            ..EvalRequest::ENERGY_ONLY
        }
    }

    #[test]
    fn a_state_is_followed_where_two_change_places() {
        let followed = FollowedStates::new(Box::new(TwoStates));
        let energy =
            |d: f64, k: usize| followed.evaluate(&at(d), &state(k)).unwrap().energy.value();
        // Before the crossing state 1 is A, and it stays A behind it, where A is the
        // second state by energy; the ground state is not touched.
        assert!((energy(1.0, 1) - 0.15).abs() < 1e-14);
        assert!((energy(1.5, 1) - 0.175).abs() < 1e-14);
        assert!(followed.take_warnings().is_empty());
        assert!((energy(2.5, 1) - 0.225).abs() < 1e-14);
        assert_eq!(energy(2.5, 0), 0.0);
        let said = followed.take_warnings();
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(said[0].contains("is state 2 in the order of the energies here"));
        assert_eq!(followed.followed_state(&at(2.5), 1), Some(1));
        // Counted by the energy without the follower.
        let plain = TwoStates.evaluate(&at(2.5), &state(1)).unwrap();
        assert!((plain.energy.value() - 0.175).abs() < 1e-14);
        // A second state that is asked for later, where it is free.
        let followed = FollowedStates::new(Box::new(TwoStates));
        let energy =
            |d: f64, k: usize| followed.evaluate(&at(d), &state(k)).unwrap().energy.value();
        assert!((energy(1.0, 1) - 0.15).abs() < 1e-14);
        assert!((energy(1.0, 2) - 0.25).abs() < 1e-14);
        // Both are assigned at once behind the crossing: A stays 1, B stays 2.
        assert!((energy(3.0, 2) - 0.15).abs() < 1e-14);
        assert!((energy(3.0, 1) - 0.25).abs() < 1e-14);
        // More states than there are.
        assert!(followed.evaluate(&at(1.0), &state(3)).is_err());
    }

    #[test]
    fn a_restart_goes_on_from_what_the_journal_tells() {
        // The journal tells that state 1 was the second state by energy at d = 2.5; the
        // next structure is compared with that one.
        let followed = FollowedStates::new(Box::new(TwoStates));
        followed.recall_state(&at(2.5), 1, 1);
        let e = followed.evaluate(&at(2.6), &state(1)).unwrap();
        assert!((e.energy.value() - 0.23).abs() < 1e-14);
        // Without it the state is counted anew by its energy.
        let fresh = FollowedStates::new(Box::new(TwoStates));
        let e = fresh.evaluate(&at(2.6), &state(1)).unwrap();
        assert!((e.energy.value() - 0.17).abs() < 1e-14);
    }

    /// A calculator that does not hand out its states.
    struct Silent;

    impl Calculator for Silent {
        fn name(&self) -> &str {
            "silent"
        }

        fn evaluate(&self, _: &MolecularSystem, _: &EvalRequest) -> MoptResult<EvalOutput> {
            Ok(EvalOutput::energy_gradient(
                Quantity::new(0.0),
                Col::zeros(6),
            ))
        }
    }

    /// The singlet of [`TwoStates`] and a triplet whose two excited states never change
    /// places: `C = 0.5 + 0.01 d` below `E = 0.7 + 0.01 d`.
    struct TwoSpins;

    fn triplet(d: f64) -> [f64; 2] {
        [0.5 + 0.01 * d, 0.7 + 0.01 * d]
    }

    impl Calculator for TwoSpins {
        fn name(&self) -> &str {
            "two_spins"
        }

        fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
            if sys.spin_charge.multiplicity == 1 {
                return TwoStates.evaluate(sys, req);
            }
            let e = triplet(distance(sys));
            let energy = match req.excited_state {
                0 => 0.4,
                k => e[k - 1],
            };
            Ok(EvalOutput::energy_gradient(
                Quantity::new(energy),
                Col::zeros(6),
            ))
        }

        fn state_characters(
            &self,
            sys: &MolecularSystem,
            count: usize,
        ) -> MoptResult<Option<StateCharacters>> {
            if sys.spin_charge.multiplicity == 1 {
                return TwoStates.state_characters(sys, count);
            }
            Ok(Some(
                StateCharacters::new(
                    Mat::<f64>::identity(3, 3),
                    1,
                    triplet(distance(sys)).to_vec(),
                    Mat::<f64>::identity(2, 2),
                )?
                .lowest(count),
            ))
        }
    }

    fn with_spin(d: f64, multiplicity: u32) -> MolecularSystem {
        let mut sys = at(d);
        sys.spin_charge = SpinCharge::new(0, multiplicity);
        sys
    }

    #[test]
    fn the_states_of_two_multiplicities_are_followed_apart() {
        // The singlet state 1 is A, which becomes the second singlet behind d = 2; the
        // triplet state 1 is C at every distance. Both are asked for at the same structures.
        let followed = FollowedStates::new(Box::new(TwoSpins));
        let energy = |d: f64, m: u32, k: usize| {
            followed
                .evaluate(&with_spin(d, m), &state(k))
                .unwrap()
                .energy
                .value()
        };
        for d in [1.0, 1.5, 2.5, 3.0] {
            let a = energy(d, 1, 1);
            let c = energy(d, 3, 1);
            assert!((a - (0.1 + 0.05 * d)).abs() < 1e-14, "singlet at {d}: {a}");
            assert!((c - triplet(d)[0]).abs() < 1e-14, "triplet at {d}: {c}");
        }
        assert_eq!(followed.followed_state(&with_spin(3.0, 1), 1), Some(1));
        assert_eq!(followed.followed_state(&with_spin(3.0, 3), 1), Some(0));
        // What the journal tells of one multiplicity is not taken for the other.
        let fresh = FollowedStates::new(Box::new(TwoSpins));
        fresh.recall_state(&with_spin(2.5, 1), 1, 1);
        assert_eq!(fresh.followed_state(&with_spin(2.5, 3), 1), None);
        let c = fresh
            .evaluate(&with_spin(2.6, 3), &state(1))
            .unwrap()
            .energy
            .value();
        assert!((c - triplet(2.6)[0]).abs() < 1e-14, "{c}");
    }

    #[test]
    fn a_calculator_without_states_to_compare_is_refused_for_excited_states_only() {
        let followed = FollowedStates::new(Box::new(Silent));
        assert!(followed.evaluate(&at(1.0), &state(0)).is_ok());
        let e = followed.evaluate(&at(1.0), &state(1)).err().unwrap();
        assert!(e.to_string().contains("cannot be followed"), "{e}");
    }
}
