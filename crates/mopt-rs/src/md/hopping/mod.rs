//! Trajectory surface hopping by the Zhu-Nakamura theory (ZN-TSH) for the molecular dynamics
//! of [`super::MolecularDynamics`].
//!
//! The trajectory runs on one adiabatic state, the active one, of a set of electronic states
//! (roots counted as by `-es`: 0 the ground state). At every step the energies of all of them
//! are known, and the gradients of the active state and of its neighbours (the nearest
//! states of the set below and above it) whose gap to it is below [`HoppingSettings::gap`].
//! Following the review of Ishida, Nanbu, and Nakamura (section 3.1), a hop to a neighbour
//! is considered where the gap to it has a local minimum along the trajectory: at the middle
//! step of three consecutive ones (Yu et al., Fig. 1a), on which the active state did not
//! change. There the straight two-state model of the crossing and the hopping direction come
//! from the adiabatic energies and gradients alone ([`direction::gap_minimum`], Yu et al.,
//! eqs. 4-10; no nonadiabatic coupling vectors), and the Zhu-Nakamura formulas of
//! [`probability`] give the probability for the energy of the motion along the direction
//! (the potential energy of the active state and the kinetic energy along the direction at
//! the minimum). A uniform random number decides (the "ant-eater" procedure: the hop is made
//! when the probability exceeds it). The hop is made at the step after the minimum, the
//! first one at which the minimum is known: the velocity component along the direction
//! changes so that the total energy is conserved there ([`direction::adjust_velocities`]),
//! and the forces of the new state at that step continue the integration. When the kinetic
//! energy along the direction is too small for an upward hop, the hop is frustrated: the
//! velocities are kept (or the component along the direction is reversed, with
//! [`FrustratedHop::Reverse`]). Classically forbidden transitions are reported and not made
//! ([`probability`] gives the reasons).
//!
//! The random numbers come from a generator of their own, seeded from the seed of the run,
//! so that a run with surface hopping draws the same numbers for its thermostat as one
//! without. A restart replays the evaluations of the journal with the same seed and so
//! rebuilds the active state and the generator exactly.
//!
//! References: T. Ishida, S. Nanbu, H. Nakamura, Int. Rev. Phys. Chem. 2017, 36, 229 (the
//! authors' draft; section 3.1); L. Yu, C. Xu, Y. Lei, C. Zhu, Z. Wen, Phys. Chem. Chem.
//! Phys. 2014, 16, 25883 (section 2); C. Zhu, H. Kamisaka, H. Nakamura, J. Chem. Phys. 2001,
//! 115, 11036 (ZN-TSH; not read).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod direction;
pub mod probability;

use std::collections::VecDeque;

use faer::Col;

pub use direction::{adjust_velocities, gap_minimum, ForceModel, GapMinimum, StatePair};
pub use probability::{CrossingKind, CrossingModel, HopProbability};

use crate::core_infra::rng::Rng;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};

/// What a frustrated hop does to the velocities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrustratedHop {
    /// Nothing: the trajectory goes on as it was.
    Keep,
    /// The component along the hopping direction is reversed.
    Reverse,
}

impl FrustratedHop {
    /// The name in the settings and the log.
    pub fn name(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Reverse => "reverse",
        }
    }
}

/// Gap (eV) below which the gradients of the neighbours are computed and hops are
/// considered, without `-md_opt hopping_gap`. Measured on formaldehyde (`gfn0xtb`, 600 K): the
/// exponent of eq. 2.11 at the gap minima to the lower state scaled to 1 eV is 23 to 32 (the
/// probability below 1e-10); see the documentation of `mdmain`.
pub const DEFAULT_HOPPING_GAP_EV: f64 = 1.0;

/// The settings of surface hopping.
#[derive(Debug, Clone, PartialEq)]
pub struct HoppingSettings {
    /// The electronic states that take part (roots, ascending, without repetitions).
    pub states: Vec<usize>,
    /// The state the trajectory starts on.
    pub start: usize,
    /// Gap (Hartree) below which the gradients of the neighbours are computed and hops are
    /// considered.
    pub gap: f64,
    /// What a frustrated hop does.
    pub frustrated: FrustratedHop,
    /// How the diabatic forces of a gap minimum are taken.
    pub forces: ForceModel,
}

impl HoppingSettings {
    /// Surface hopping among `states` from `start`, with the default gap and treatments.
    pub fn new(states: Vec<usize>, start: usize) -> MoptResult<Self> {
        let s = Self {
            states,
            start,
            gap: DEFAULT_HOPPING_GAP_EV * Units::EV_TO_HARTREE,
            frustrated: FrustratedHop::Keep,
            forces: ForceModel::TwoState,
        };
        s.validate()?;
        Ok(s)
    }

    /// Check the values.
    pub fn validate(&self) -> MoptResult<()> {
        if self.states.len() < 2 {
            return Err(MoptError::config(
                "surface hopping needs at least two electronic states (hopping_states=0,1)",
            ));
        }
        if self.states.windows(2).any(|w| w[1] <= w[0]) {
            return Err(MoptError::config(
                "the states of surface hopping are given in ascending order, each once",
            ));
        }
        if !self.states.contains(&self.start) {
            return Err(MoptError::config(format!(
                "the starting state {} (-es) is not one of the states of surface hopping {:?}",
                self.start, self.states
            )));
        }
        if !(self.gap.is_finite() && self.gap > 0.0) {
            return Err(MoptError::config(
                "the gap of surface hopping must be positive",
            ));
        }
        Ok(())
    }
}

/// What the dynamics asks of the surface for surface hopping.
#[derive(Debug, Clone, PartialEq)]
pub struct StatesRequest {
    /// The active state (energy and gradient).
    pub active: usize,
    /// All states that take part (energies).
    pub states: Vec<usize>,
    /// The gradients of the neighbours of the active state are wanted when their gap to it
    /// is below this (Hartree).
    pub gradient_gap: f64,
    /// The neighbours whose gap was below `gradient_gap` at the step before: their gradients
    /// are asked for at once rather than after their energies.
    pub expected: Vec<usize>,
}

impl StatesRequest {
    /// The nearest states of `states` below and above `active`.
    pub fn neighbours(&self) -> Vec<usize> {
        neighbours(&self.states, self.active)
    }
}

/// The nearest states of `states` (ascending) below and above `active`.
pub fn neighbours(states: &[usize], active: usize) -> Vec<usize> {
    let below = states.iter().copied().filter(|&s| s < active).max();
    let above = states.iter().copied().filter(|&s| s > active).min();
    below.into_iter().chain(above).collect()
}

/// The energies of the states at one evaluation, and the gradients that were computed.
#[derive(Debug, Clone, PartialEq)]
pub struct StateValues {
    /// The states (those of [`StatesRequest::states`]).
    pub states: Vec<usize>,
    /// Their energies on the whole surface (with the bias potentials), Hartree.
    pub energies: Vec<f64>,
    /// Their gradients, where computed (the active state always).
    pub gradients: Vec<Option<Col<f64>>>,
}

impl StateValues {
    fn index(&self, state: usize) -> Option<usize> {
        self.states.iter().position(|&s| s == state)
    }

    /// The energy of `state`.
    pub fn energy(&self, state: usize) -> Option<f64> {
        self.index(state).map(|k| self.energies[k])
    }

    /// The gradient of `state`, when it was computed.
    pub fn gradient(&self, state: usize) -> Option<&Col<f64>> {
        self.index(state).and_then(|k| self.gradients[k].as_ref())
    }
}

/// How a candidate for a hop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HopOutcome {
    /// The trajectory hopped.
    Hopped,
    /// The random number exceeded the probability.
    Stayed,
    /// Accepted, but the kinetic energy along the direction could not pay for it.
    Frustrated,
    /// Classically forbidden: no hop is made.
    Forbidden,
    /// The two states had equal diabatic forces: no direction, no model.
    NoDirection,
}

impl HopOutcome {
    /// The name in the log and the files.
    pub fn name(self) -> &'static str {
        match self {
            Self::Hopped => "hop",
            Self::Stayed => "stay",
            Self::Frustrated => "frustrated",
            Self::Forbidden => "forbidden",
            Self::NoDirection => "no direction",
        }
    }
}

/// A candidate for a hop: a minimum of the gap between the active state and a neighbour.
#[derive(Debug, Clone, PartialEq)]
pub struct HopRecord {
    /// The step of the minimum.
    pub step: usize,
    /// Its time (atomic units).
    pub time: f64,
    /// The step at which the hop was decided (and made).
    pub decided: usize,
    /// The active state.
    pub from: usize,
    /// The neighbour.
    pub to: usize,
    /// The gap at the minimum (Hartree).
    pub gap: f64,
    /// The Zhu-Nakamura quantities (`None` without a direction).
    pub zn: Option<HopProbability>,
    /// The kinetic energy along the direction at the minimum (Hartree).
    pub kinetic_along: f64,
    /// The random number that decided (for the classically allowed ones).
    pub random: Option<f64>,
    /// How it ended.
    pub outcome: HopOutcome,
    /// The change of the potential energy in the hop (Hartree; 0 without one).
    pub potential_change: f64,
    /// The change of the kinetic energy in the hop (Hartree; 0 without one).
    pub kinetic_change: f64,
}

impl HopRecord {
    /// One line for the log.
    pub fn describe(&self) -> String {
        let ev = Units::HARTREE_TO_EV;
        let mut s = format!(
            "Hop candidate at step {} ({:.4} fs, decided at step {}): state {} -> {}, gap {:.6} \
             eV",
            self.step,
            self.time * Units::AU_TIME_TO_FS,
            self.decided,
            self.from,
            self.to,
            self.gap * ev
        );
        if let Some(zn) = &self.zn {
            s.push_str(&format!(
                ", {} crossing, {}, a^2 = {:.4e}, b^2 = {:.4e}, kinetic energy along the \
                 direction {:.6} eV",
                zn.kind.name(),
                if zn.allowed {
                    "classically allowed"
                } else {
                    "classically forbidden"
                },
                zn.a2,
                zn.b2,
                self.kinetic_along * ev
            ));
            if let Some(p) = zn.probability {
                s.push_str(&format!(", probability {p:.6e}"));
            }
            if let Some(t) = zn.transmission {
                s.push_str(&format!(", transmission P12 {t:.6e}"));
            }
        }
        if let Some(r) = self.random {
            s.push_str(&format!(", random number {r:.6}"));
        }
        s.push_str(&format!(": {}", self.outcome.name()));
        if self.outcome == HopOutcome::Hopped {
            s.push_str(&format!(
                " (potential energy {:+.6} eV, kinetic energy {:+.6} eV)",
                self.potential_change * ev,
                self.kinetic_change * ev
            ));
        }
        s
    }
}

/// What the dynamics keeps of a step for the hops.
#[derive(Debug, Clone)]
struct Frame {
    step: usize,
    time: f64,
    active: usize,
    v: Col<f64>,
    values: StateValues,
}

/// The states `a` and `b` of `frame` as the lower and the upper one.
fn state_pair(frame: &Frame, a: usize, b: usize) -> Option<StatePair<'_>> {
    let (ea, eb) = (frame.values.energy(a)?, frame.values.energy(b)?);
    let (ga, gb) = (frame.values.gradient(a)?, frame.values.gradient(b)?);
    Some(if ea <= eb {
        StatePair {
            lower: ea,
            upper: eb,
            lower_gradient: ga,
            upper_gradient: gb,
        }
    } else {
        StatePair {
            lower: eb,
            upper: ea,
            lower_gradient: gb,
            upper_gradient: ga,
        }
    })
}

/// The decision of a hop, to be carried out by the dynamics on its velocities and forces.
#[derive(Debug, Clone)]
pub struct HopPlan {
    /// The state to go to.
    pub to: usize,
    /// The mass-weighted unit direction (Cartesian velocity change of `sum m u^2 = 1`).
    pub direction: Col<f64>,
    /// The record, whose outcome the dynamics completes.
    pub record: HopRecord,
}

/// The state of surface hopping along a trajectory.
#[derive(Debug, Clone)]
pub struct SurfaceHopping {
    settings: HoppingSettings,
    active: usize,
    rng: Rng,
    frames: VecDeque<Frame>,
    hops: usize,
}

/// A mix of the seed of the run for the generator of the hops (so that it draws other
/// numbers than the thermostat).
const SEED_MIX: u64 = 0x5a4e_5453_485f_5a4e;

impl SurfaceHopping {
    /// Surface hopping with `settings` for a run of seed `seed`.
    pub fn new(settings: HoppingSettings, seed: u64) -> MoptResult<Self> {
        settings.validate()?;
        Ok(Self {
            active: settings.start,
            rng: Rng::from_seed(seed ^ SEED_MIX),
            settings,
            frames: VecDeque::new(),
            hops: 0,
        })
    }

    /// The settings.
    pub fn settings(&self) -> &HoppingSettings {
        &self.settings
    }

    /// The active state.
    pub fn active(&self) -> usize {
        self.active
    }

    /// Hops made so far.
    pub fn hops(&self) -> usize {
        self.hops
    }

    /// What the next evaluation must give.
    pub fn request(&self) -> StatesRequest {
        let expected = match self.frames.back().filter(|f| f.active == self.active) {
            Some(f) => neighbours(&self.settings.states, self.active)
                .into_iter()
                .filter(
                    |&j| match (f.values.energy(j), f.values.energy(self.active)) {
                        (Some(a), Some(b)) => (a - b).abs() < self.settings.gap,
                        _ => false,
                    },
                )
                .collect(),
            None => Vec::new(),
        };
        StatesRequest {
            active: self.active,
            states: self.settings.states.clone(),
            gradient_gap: self.settings.gap,
            expected,
        }
    }

    /// Check the states told for an evaluation.
    pub fn check(&self, values: &StateValues, n: usize) -> MoptResult<()> {
        if values.states != self.settings.states
            || values.energies.len() != values.states.len()
            || values.gradients.len() != values.states.len()
        {
            return Err(MoptError::calc(format!(
                "surface hopping asked for the states {:?} and was told {:?}",
                self.settings.states, values.states
            )));
        }
        if values.energies.iter().any(|e| !e.is_finite()) {
            return Err(MoptError::calc("an energy of the states is not finite"));
        }
        match values.gradient(self.active) {
            Some(g) if g.nrows() == n => Ok(()),
            _ => Err(MoptError::calc(format!(
                "surface hopping was told no gradient of the active state {}",
                self.active
            ))),
        }
    }

    /// Keep the completed step `step` (time `time`, velocities `v`), and decide on a hop at
    /// a gap minimum of the step before. `kinetic_along(direction)` is the kinetic energy
    /// along a mass-weighted unit direction at that minimum; `inverse_masses` per
    /// component. Returns the hop to make now, if any; the records of all candidates are
    /// in `records`.
    pub fn observe(
        &mut self,
        step: usize,
        time: f64,
        v: &Col<f64>,
        values: StateValues,
        inverse_masses: &[f64],
        masses: &[f64],
        records: &mut Vec<HopRecord>,
    ) -> Option<HopPlan> {
        self.frames.push_back(Frame {
            step,
            time,
            active: self.active,
            v: v.clone(),
            values,
        });
        while self.frames.len() > 3 {
            self.frames.pop_front();
        }
        if self.frames.len() < 3 || self.frames.iter().any(|f| f.active != self.active) {
            return None;
        }
        let (first, middle, last) = (&self.frames[0], &self.frames[1], &self.frames[2]);
        let active = self.active;
        let gap = |f: &Frame, j: usize| -> Option<f64> {
            Some((f.values.energy(j)? - f.values.energy(active)?).abs())
        };
        let mut candidates: Vec<(f64, usize)> = neighbours(&self.settings.states, active)
            .into_iter()
            .filter_map(|j| {
                let (g1, g2, g3) = (gap(first, j)?, gap(middle, j)?, gap(last, j)?);
                let below = [g1, g2, g3].iter().all(|&g| g < self.settings.gap);
                (g2 < g1 && g2 <= g3 && below).then_some((g2, j))
            })
            .collect();
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut plan = None;
        for (g2, j) in candidates {
            let record = |zn, kinetic_along, random, outcome| HopRecord {
                step: middle.step,
                time: middle.time,
                decided: last.step,
                from: active,
                to: j,
                gap: g2,
                zn,
                kinetic_along,
                random,
                outcome,
                potential_change: 0.0,
                kinetic_change: 0.0,
            };
            let mid = (|| {
                let (ea, ej) = (middle.values.energy(active)?, middle.values.energy(j)?);
                Some((ea.min(ej), ea.max(ej)))
            })();
            let (Some(p1), Some(mid), Some(p3)) = (
                state_pair(first, active, j),
                mid,
                state_pair(last, active, j),
            ) else {
                continue;
            };
            let model = gap_minimum(p1, mid, p3, inverse_masses, self.settings.forces);
            let Some(GapMinimum {
                model,
                direction: u,
            }) = model
            else {
                records.push(record(None, 0.0, None, HopOutcome::NoDirection));
                continue;
            };
            let c = direction::component(&middle.v, &u, masses);
            let kinetic_along = 0.5 * c * c;
            let energy = middle.values.energy(active).unwrap_or(f64::NAN) + kinetic_along;
            let Some(zn) = model.probability(energy) else {
                records.push(record(None, kinetic_along, None, HopOutcome::NoDirection));
                continue;
            };
            let Some(p) = zn.probability.filter(|_| zn.allowed) else {
                records.push(record(Some(zn), kinetic_along, None, HopOutcome::Forbidden));
                continue;
            };
            let xi = self.rng.next_f64();
            if p > xi {
                plan = Some(HopPlan {
                    to: j,
                    direction: u,
                    record: record(Some(zn), kinetic_along, Some(xi), HopOutcome::Hopped),
                });
                break;
            }
            records.push(record(
                Some(zn),
                kinetic_along,
                Some(xi),
                HopOutcome::Stayed,
            ));
        }
        plan
    }

    /// The hop of `plan` was made: the trajectory is on `plan.to` from now on.
    pub fn hopped(&mut self, to: usize) {
        self.active = to;
        self.hops += 1;
        self.frames.clear();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_settings_are_checked_and_the_neighbours_found() {
        assert!(HoppingSettings::new(vec![1], 1).is_err());
        assert!(HoppingSettings::new(vec![2, 1], 1).is_err());
        assert!(HoppingSettings::new(vec![1, 2], 0).is_err());
        let s = HoppingSettings::new(vec![0, 1, 3], 1).unwrap();
        assert_eq!(s.forces, ForceModel::TwoState);
        assert_eq!(neighbours(&s.states, 1), vec![0, 3]);
        assert_eq!(neighbours(&s.states, 3), vec![1]);
        assert_eq!(neighbours(&s.states, 0), vec![1]);
    }
}
