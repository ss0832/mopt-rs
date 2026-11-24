//! Conformers of a transition state.
//!
//! 1. The input is optimized to a saddle point of order 1, its Hessian must have exactly one
//!    imaginary frequency, and the IRC from it gives the two minima of its reaction.
//! 2. The saddle direction is held while the conformers are searched: the imaginary mode
//!    (mass-weighted normal mode, as Cartesian displacements) is written in the primitive
//!    internal coordinates of the transition state, `w = B dx`; the coordinates that carry at
//!    least `reactive_share` of its largest component form the reaction coordinate
//!    `c(x) = sum_k w_k (q_k(x) - q_k(TS))` (normalized), which every optimization holds at
//!    zero and the dynamics either hold by RATTLE or replace by mode freezing ([`Hold`]).
//!    Unlike the Cartesian mode, the coordinate does not depend on the orientation and stays
//!    attached to the reacting atoms as the rest of the molecule turns.
//! 3. The search of conformers runs with the coordinate held (any strategy); the constrained
//!    minima within `window` of the lowest are released and optimized to saddle points of
//!    order 1 (the methods of the command for `-order 1`), checked for exactly one imaginary
//!    frequency and for the same reaction (the bond graphs of the ends of their IRC are those
//!    of the ends of the input's), and freed of duplicates and rotamers. The lowest is the
//!    barrier of the reaction.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};
use rayon::prelude::*;

use super::ts_coordinate::{reaction_coordinate, reactive_bonds};
use super::{
    CommonSettings, DynamicsJob, Level, RelaxJob, Relaxed, Sampled, SearchEvent, SearchJobs,
    SearchObserver, SearchResult,
};
use crate::coord::graph::MolecularGraph;
use crate::domain::{AtomId, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::md::{FreezeSettings, ModeSelection};
use crate::pes::constraint::{Constraint, ConstraintFunction, ConstraintKind};
use crate::workflow::explore::{ConformerEnsemble, Verdict};

/// How the dynamics hold the saddle direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hold {
    /// The reaction coordinate held by RATTLE (no Hessians).
    Coordinate,
    /// No motion along the negative-curvature mode of the Hessian, computed again every
    /// `update` steps and followed by overlap (`md::freeze`).
    FrozenModes {
        /// Steps between Hessians.
        update: usize,
    },
}

/// Settings of the search of transition-state conformers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsSettings {
    /// How the dynamics hold the saddle direction.
    pub hold: Hold,
    /// Internal coordinates with less than this share of the largest component of the mode
    /// are left out of the reaction coordinate.
    pub reactive_share: f64,
    /// Also hold the lengths of the reactive bonds (the bonds with at least half the largest
    /// component of the mode among the bonds) at their values in the transition state, so
    /// that the reacting fragments cannot fall apart along the directions the reaction
    /// coordinate leaves free.
    pub hold_bonds: bool,
    /// Scale of the covalent radii of the primitive internal coordinates (partial bonds of
    /// a transition state are longer than ordinary ones).
    pub bond_scale: f64,
    /// Constrained minima within this of the lowest (Hartree) are optimized to saddle points.
    pub window: f64,
    /// At most this many of them, the lowest first.
    pub max_candidates: usize,
    /// Check the reaction of every saddle point by its IRC.
    pub check_reaction: bool,
}

impl Default for TsSettings {
    fn default() -> Self {
        Self {
            hold: Hold::Coordinate,
            reactive_share: 0.1,
            hold_bonds: true,
            bond_scale: 1.5,
            window: 6.0 * crate::domain::Units::KCAL_MOL_TO_HARTREE,
            max_candidates: 20,
            check_reaction: true,
        }
    }
}

/// A saddle point found from a constrained minimum.
#[derive(Debug, Clone)]
pub struct TsCandidate {
    /// How it was made.
    pub origin: String,
    /// The saddle point.
    pub system: MolecularSystem,
    /// Its energy.
    pub energy: f64,
    /// Imaginary frequencies.
    pub imaginary: usize,
    /// The imaginary wavenumber (cm^-1, negative), when there is one.
    pub wavenumber: Option<f64>,
    /// Whether its IRC ends at the minima of the input's reaction (`None`: not checked).
    pub same_reaction: Option<bool>,
    /// Energies of the ends of its IRC.
    pub ends: Option<[f64; 2]>,
    /// Its number in the ensemble of transition states, when it was kept as a new one.
    pub member: Option<usize>,
}

/// The end of the search of transition-state conformers.
#[derive(Debug, Clone)]
pub struct TsResult {
    /// The search with the saddle direction held (its members are constrained minima).
    pub constrained: SearchResult,
    /// The transition states kept: the input's first.
    pub ensemble: ConformerEnsemble,
    /// Every saddle-point optimization and what became of it.
    pub candidates: Vec<TsCandidate>,
    /// The bond graphs of the two ends of the input's reaction (when its IRC was followed).
    pub reaction: Option<[MolecularGraph; 2]>,
    /// The internal coordinates of the reaction coordinate with their weights.
    pub coordinate: Vec<(String, f64)>,
}

/// The search jobs of the command with the saddle direction held.
struct Held<'a> {
    inner: &'a dyn SearchJobs,
    /// The reaction coordinate (`None` when the held bonds carry all of the mode).
    constraint: Option<Constraint>,
    /// The reactive bonds (with `hold_bonds`).
    bonds: Vec<Constraint>,
    hold: Hold,
}

/// The atoms of a distance constraint between two single atoms.
fn pair_of(c: &Constraint) -> Option<[AtomId; 2]> {
    match &c.kind {
        ConstraintKind::Distance(a, b) if a.len() == 1 && b.len() == 1 => {
            Some([a[0].min(b[0]), a[0].max(b[0])])
        }
        _ => None,
    }
}

impl SearchJobs for Held<'_> {
    fn relax(&self, start: &MolecularSystem, job: &RelaxJob) -> MoptResult<Relaxed> {
        let mut job = job.clone();
        job.constraints.extend(self.constraint.iter().cloned());
        job.constraints.extend(self.bonds.iter().cloned());
        self.inner.relax(start, &job)
    }

    fn dynamics(&self, start: &MolecularSystem, job: &DynamicsJob) -> MoptResult<Sampled> {
        let mut job = job.clone();
        // A reactive bond that SHAKE holds already is not held twice.
        let shaken: Vec<[AtomId; 2]> = job.constraints.iter().filter_map(pair_of).collect();
        job.constraints.extend(
            self.bonds
                .iter()
                .filter(|c| pair_of(c).is_none_or(|p| !shaken.contains(&p)))
                .cloned(),
        );
        match self.hold {
            Hold::Coordinate => job.constraints.extend(self.constraint.iter().cloned()),
            Hold::FrozenModes { update } => {
                job.freeze = Some(FreezeSettings {
                    selection: ModeSelection::Negative,
                    update_interval: Some(update.max(1)),
                });
            }
        }
        self.inner.dynamics(start, &job)
    }

    fn hessian(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        self.inner.hessian(sys)
    }

    fn modes(&self, sys: &MolecularSystem) -> MoptResult<crate::vib::NormalModes> {
        self.inner.modes(sys)
    }

    fn reaction_ends(
        &self,
        ts: &MolecularSystem,
        label: &str,
    ) -> MoptResult<Option<[(MolecularSystem, f64); 2]>> {
        self.inner.reaction_ends(ts, label)
    }

    fn stop_requested(&self) -> bool {
        self.inner.stop_requested()
    }
}

/// Imaginary frequencies (by the one criterion of the crate,
/// `opt::mode_kill::IMAGINARY_CURVATURE`) and the first mode of `sys` on the surface of
/// `jobs`.
fn imaginary_modes(
    jobs: &dyn SearchJobs,
    sys: &MolecularSystem,
) -> MoptResult<(usize, Option<f64>, Col<f64>)> {
    let modes = jobs.modes(sys)?;
    let count = modes.imaginary_modes();
    let first = modes
        .curvatures
        .first()
        .filter(|k| crate::vib::is_imaginary_curvature(**k))
        .and_then(|_| modes.wavenumbers.first().copied());
    let mode = modes
        .displacements
        .first()
        .cloned()
        .ok_or_else(|| MoptError::calc("the Hessian has no modes"))?;
    Ok((count, first, mode))
}

fn graphs_of(ends: &[(MolecularSystem, f64); 2], scale: f64) -> [MolecularGraph; 2] {
    [
        MolecularGraph::from_system(&ends[0].0, scale),
        MolecularGraph::from_system(&ends[1].0, scale),
    ]
}

fn same_reaction(a: &[MolecularGraph; 2], b: &[MolecularGraph; 2]) -> bool {
    (a[0].is_isomorphic(&b[0]) && a[1].is_isomorphic(&b[1]))
        || (a[0].is_isomorphic(&b[1]) && a[1].is_isomorphic(&b[0]))
}

/// The search of the conformers of the transition state `input`; `search` runs the chosen
/// strategy on the jobs it is given (the saddle direction held) from the structure it is given.
pub fn run(
    input: &MolecularSystem,
    settings: &TsSettings,
    common: &CommonSettings,
    jobs: &dyn SearchJobs,
    observer: &mut dyn SearchObserver,
    search: &mut dyn FnMut(
        &MolecularSystem,
        &dyn SearchJobs,
        &mut dyn SearchObserver,
    ) -> MoptResult<Option<SearchResult>>,
) -> MoptResult<Option<TsResult>> {
    let note = |o: &mut dyn SearchObserver, t: String| o.on_event(&SearchEvent::Note(t));
    note(
        observer,
        "Transition state: optimization of the input to a saddle point of order 1.".into(),
    )?;
    let saddle = RelaxJob {
        saddle_order: 1,
        ..RelaxJob::minimize("transition state", Level::Final)
    };
    let ts = jobs.relax(input, &saddle)?;
    if ts.interrupted {
        return Ok(None);
    }
    if !ts.converged {
        return Err(MoptError::conv(
            "the input did not converge to a saddle point; start from a structure closer to \
             the transition state",
        ));
    }
    let (count, wavenumber, mode) = imaginary_modes(jobs, &ts.system)?;
    if count != 1 {
        return Err(MoptError::conv(format!(
            "the saddle point of the input has {count} imaginary frequencies, not one"
        )));
    }
    note(
        observer,
        format!(
            "Transition state: E = {:.10}, imaginary frequency {:.1} cm^-1",
            ts.energy,
            wavenumber.unwrap_or(f64::NAN)
        ),
    )?;
    let reaction = if settings.check_reaction {
        match jobs.reaction_ends(&ts.system, "transition state")? {
            Some(ends) => Some(graphs_of(&ends, common.bond_scale)),
            None => {
                note(
                    observer,
                    "The IRC of the input did not end at two minima: the reactions of the \
                     conformers are not checked."
                        .into(),
                )?;
                None
            }
        }
    } else {
        None
    };
    let n = mode.nrows() / 3;
    let size: Vec<f64> = (0..n)
        .map(|a| (0..3).map(|k| mode[3 * a + k].powi(2)).sum::<f64>().sqrt())
        .collect();
    let top = size.iter().copied().fold(f64::MIN_POSITIVE, f64::max);
    note(
        observer,
        format!(
            "Displacements of the atoms in the imaginary mode (share of the largest): {}",
            size.iter()
                .enumerate()
                .map(|(a, s)| format!("{}:{:.2}", a + 1, s / top))
                .collect::<Vec<_>>()
                .join(" ")
        ),
    )?;
    let bonds = if settings.hold_bonds {
        reactive_bonds(&ts.system, &mode, settings.bond_scale)?
    } else {
        Vec::new()
    };
    if !bonds.is_empty() {
        note(
            observer,
            format!(
                "Reactive bonds held at their lengths in the transition state: {}",
                bonds
                    .iter()
                    .map(|b| b.1.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )?;
    }
    let pairs: Vec<[AtomId; 2]> = bonds.iter().filter_map(|b| pair_of(&b.0)).collect();
    let coordinate = reaction_coordinate(
        &ts.system,
        &mode,
        settings.bond_scale,
        settings.reactive_share,
        &pairs,
    )?;
    let mut labels = Vec::new();
    let constraint = match coordinate {
        Some((coordinate, l)) => {
            note(
                observer,
                format!(
                    "Reaction coordinate held during the search: {}",
                    l.iter()
                        .map(|(name, w)| format!("{w:+.3} {name}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )?;
            labels = l;
            Some(Constraint {
                kind: ConstraintKind::Function(ConstraintFunction::new(Arc::new(coordinate))),
                target: Some(0.0),
                anchor: None,
            })
        }
        None => None,
    };
    let held = Held {
        inner: jobs,
        constraint,
        bonds: bonds.into_iter().map(|b| b.0).collect(),
        hold: settings.hold,
    };
    let Some(constrained) = search(&ts.system, &held, observer)? else {
        return Ok(None);
    };
    let mut ensemble =
        ConformerEnsemble::new(common.compare).comparing(common.compare_atoms.clone());
    ensemble.add(&ts.system, ts.energy, "input")?;
    let mut candidates = Vec::new();
    if constrained.interrupted {
        return Ok(Some(TsResult {
            constrained,
            ensemble,
            candidates,
            reaction,
            coordinate: labels,
        }));
    }
    let starts: Vec<usize> = constrained
        .ensemble
        .within(settings.window)
        .into_iter()
        .filter(|&i| i != 0)
        .take(settings.max_candidates)
        .collect();
    note(
        observer,
        format!(
            "{} constrained minima within the window are optimized to saddle points.",
            starts.len()
        ),
    )?;
    let saddles: Vec<MoptResult<Relaxed>> = starts
        .par_iter()
        .map(|&i| {
            jobs.relax(
                &constrained.ensemble.members()[i].system,
                &RelaxJob {
                    saddle_order: 1,
                    ..RelaxJob::minimize(format!("saddle point from EQ{i}"), Level::Final)
                },
            )
        })
        .collect();
    for (&i, result) in starts.iter().zip(saddles) {
        let origin = format!("saddle point from constrained EQ{i}");
        let r = match result {
            Ok(r) if r.interrupted => break,
            Ok(r) if r.converged => r,
            Ok(_) => {
                note(
                    observer,
                    format!("{origin}: the optimization did not converge."),
                )?;
                continue;
            }
            Err(e) => {
                note(observer, format!("{origin}: the optimization failed: {e}"))?;
                continue;
            }
        };
        let (imaginary, wavenumber, _) = imaginary_modes(jobs, &r.system)?;
        let mut candidate = TsCandidate {
            origin: origin.clone(),
            system: r.system.clone(),
            energy: r.energy,
            imaginary,
            wavenumber,
            same_reaction: None,
            ends: None,
            member: None,
        };
        if imaginary == 1 {
            if let Some(reference) = &reaction {
                if let Some(ends) = jobs.reaction_ends(&r.system, &origin)? {
                    candidate.ends = Some([ends[0].1, ends[1].1]);
                    candidate.same_reaction = Some(same_reaction(
                        reference,
                        &graphs_of(&ends, common.bond_scale),
                    ));
                } else {
                    candidate.same_reaction = Some(false);
                }
            }
            if candidate.same_reaction != Some(false) {
                let verdict = match ensemble.add(&r.system, r.energy, &origin)? {
                    Verdict::New(k) => {
                        candidate.member = Some(k);
                        format!("a new transition-state conformer, TS{k}")
                    }
                    Verdict::Duplicate { of, .. } => format!("the same as TS{of}"),
                    Verdict::Rotamer { of, .. } => format!("a rotamer of TS{of}"),
                };
                note(observer, format!("{origin}: {verdict}"))?;
            }
        }
        note(
            observer,
            format!(
                "{origin}: E = {:.10}, {imaginary} imaginary frequencies{}{}",
                r.energy,
                wavenumber.map_or(String::new(), |w| format!(" ({w:.1} cm^-1)")),
                match candidate.same_reaction {
                    Some(true) => ", the same reaction",
                    Some(false) => ", another reaction",
                    None => "",
                }
            ),
        )?;
        candidates.push(candidate);
    }
    Ok(Some(TsResult {
        constrained,
        ensemble,
        candidates,
        reaction,
        coordinate: labels,
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};

    /// Two H2 molecules side by side, each with the force constant of its own bond (Hartree /
    /// Bohr^2) along x: the stretches are orthogonal to the rigid-body motions, so their
    /// mass-weighted curvatures are those of the bonds.
    struct Pairs {
        k: [f64; 2],
    }

    impl SearchJobs for Pairs {
        fn relax(&self, start: &MolecularSystem, _job: &RelaxJob) -> MoptResult<Relaxed> {
            Ok(Relaxed {
                system: start.clone(),
                energy: 0.0,
                bare_energy: 0.0,
                converged: true,
                dissociated: false,
                iterations: 1,
                interrupted: false,
            })
        }

        fn dynamics(&self, _start: &MolecularSystem, _job: &DynamicsJob) -> MoptResult<Sampled> {
            Ok(Sampled::default())
        }

        fn hessian(&self, _sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
            let mut h = Mat::<f64>::zeros(12, 12);
            for (pair, k) in self.k.iter().enumerate() {
                let (a, b) = (6 * pair, 6 * pair + 3);
                h[(a, a)] = *k;
                h[(b, b)] = *k;
                h[(a, b)] = -k;
                h[(b, a)] = -k;
            }
            Ok(h)
        }

        fn reaction_ends(
            &self,
            _ts: &MolecularSystem,
            _label: &str,
        ) -> MoptResult<Option<[(MolecularSystem, f64); 2]>> {
            Ok(None)
        }

        fn stop_requested(&self) -> bool {
            false
        }
    }

    #[test]
    fn a_soft_second_imaginary_mode_above_the_old_cutoff_is_counted() {
        let sys = MolecularSystem::new(
            vec![Element::H; 4],
            &positions::<Bohr>(&[
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 6.0, 0.0],
                [1.4, 6.0, 0.0],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let mu = 0.5 * sys.masses_amu()[0];
        // The force constant of a bond whose wavenumber is -nu (cm^-1).
        let k = |nu: f64| -(nu / crate::domain::Units::EIGVAL_MW_TO_CM1).powi(2) * mu;
        // A transition state (-500 cm^-1) with a second, soft mode of -5 cm^-1: above the
        // former cutoff of -20 cm^-1, below the one criterion (about -0.5 cm^-1).
        let soft = Pairs {
            k: [k(500.0), k(5.0)],
        };
        let (count, first, _) = imaginary_modes(&soft, &sys).unwrap();
        assert_eq!(count, 2);
        assert!((first.unwrap() + 500.0).abs() < 1e-6, "{first:?}");
        // A flat second mode within the criterion is no imaginary mode.
        let flat = Pairs {
            k: [k(500.0), k(0.1)],
        };
        assert_eq!(imaginary_modes(&flat, &sys).unwrap().0, 1);
        // No imaginary mode: no wavenumber of one.
        let minimum = Pairs {
            k: [-k(500.0), -k(5.0)],
        };
        let (count, first, _) = imaginary_modes(&minimum, &sys).unwrap();
        assert_eq!((count, first), (0, None));
    }
}
