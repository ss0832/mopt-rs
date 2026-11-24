//! ONIOM microiterations: the optimizer moves the core of an ONIOM calculation, and the
//! environment is relaxed on the lowest level before every evaluation.
//!
//! The core is the outermost model region and the outer atoms of every link (`Q2`); the
//! environment is every other atom. With mechanical embedding the model systems do not depend
//! on the environment, so `E_ONIOM(x, y) = E_low(real; x, y) + const(x)` in the environment
//! `y`, and the relaxation of `y` needs only the lowest level on the real system. The surface
//! seen by the optimizer is `E*(x) = min_y E_ONIOM(x, y)`; its gradient is the ONIOM gradient
//! on the core at the relaxed environment (the environment gradient vanishes there). The
//! Hessian given for `E*` is its own, `H_xx - H_xy H_yy^-1 H_yx` (the effective Hessian:
//! central differences of the ONIOM gradient for `H_xx` and `H_yx`, `H_yy^-1` by conjugate
//! gradients on Hessian-vector products of the lowest level); [`CoreHessian`] can instead
//! give it with the inverse of the limited-memory BFGS, or the core block `H_xx` with the
//! environment held. Every relaxation runs to
//! convergence. The mode kill of [`crate::pes::microiter`] (the check of the lowest curvature
//! of the environment and the removal of its negative-curvature modes) is off by default: the
//! low layer is not mode-killed, the core is, by the mode kill of its optimizer (user,
//! 2026-11-20; on 5696 atoms with OpenMM the check took 40 Hessian-vector products per
//! relaxation). `micro_mode_kill=N` turns it on for the first relaxation and every Nth.
//!
//! Options of the relaxations:
//! - the minimizer ([`crate::pes::microiter::InnerMethod`]): the limited-memory BFGS whose
//!   pairs are kept from one relaxation to the next, a truncated Newton method, or a method
//!   of the optimizer registry;
//! - the coordinates of the limited-memory BFGS ([`EnvironmentCoordinates`]): Cartesian, or
//!   internal coordinates of each fragment of the environment ([`fragments`]);
//! - a start from the predicted response of the environment to the step of the core (the
//!   `response` module);
//! - a tolerance that follows the gradient of the core while it is large (`adaptive`, off by
//!   default; a last relaxation to the full tolerance at the end,
//!   [`RelaxedEnvironment::relax_fully`]).
//!
//! A relaxation starts from the environment of the nearest core structure evaluated so far.
//! Where rigid motions of the whole system are symmetries (no fixed atoms, no cell, both
//! levels invariant) that environment is carried along by the best-fit rigid motion of its
//! core onto the new one, and a core that is a rigid motion of a relaxed one takes the moved
//! environment without a relaxation: the optimizers recentre and rotate the core at every
//! step and the IRC puts its centre of mass at the origin, which would otherwise move the
//! core against an environment that stays where it was.
//!
//! A transition-state search, a vibrational analysis, and an IRC on this surface are those of
//! the core with the environment relaxed (the IRC is then the microiterative IRC of Polyak,
//! Hratchian, et al.). The full structures (core and relaxed environment) are kept for every
//! evaluated core structure, so that the files of a run hold the whole system
//! ([`RelaxedEnvironment::full_system`]).
//!
//! Several minima of the environment at once ([`replicas`], `micro_replicas=K`, off by
//! default): `K` replicas of the environment, each relaxed from its own environment of the core
//! before, are combined into the soft minimum `-(1/beta) ln sum_k exp(-beta E_k)` of their
//! surfaces, whose gradient is the weighted mean of their core gradients.
//!
//! References: T. Vreven, K. Morokuma, Ö. Farkas, H. B. Schlegel, M. J. Frisch, J. Comput.
//! Chem. 2003, 24, 760 (microiterations in ONIOM optimizations); I. Polyak, E. Boulanger,
//! K. Sen, W. Thiel, Phys. Chem. Chem. Phys. 2013, 15, 14188 (the microiterative IRC).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's ONIOM handler raises an exception).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod core_terms;
mod files;
pub mod fragments;
mod hessian;
mod precondition;
pub mod replicas;
mod response;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use faer::Col;

use super::layout::Region;
use crate::core_infra::log::Logger;
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::pes::calculator::Calculator;
use crate::pes::microiter::{
    relax_in, MicroEval, MicroMemory, MicroProblem, MicroProgress, MicroSettings,
};
use fragments::FragmentCoordinates;
use response::{Response, RESPONSE_ENTRIES};

pub use core_terms::CoreTerm;
pub use files::{write_full_structures, REPLICAS_SUFFIX};
pub use hessian::ResponseSolve;
pub use precondition::{
    PreconditionSettings, PRECONDITION_LARGEST, PRECONDITION_MODEL, PRECONDITION_SHIFT,
};
pub use replicas::{ReplicaSettings, ReplicaStart};

/// Memory for the full structures kept for the evaluated core structures (bytes; the oldest
/// are dropped first). A structure that is no longer kept is relaxed again where its whole
/// system is asked for ([`RelaxedEnvironment::full_system`]).
const KEPT_BYTES: usize = 256 << 20;

/// A structure read back from a file is a kept one below this largest difference (Bohr; the
/// files have ten decimals in Angstrom).
const SAME_IN_FILES: f64 = 1e-6;

/// The atoms that move in the optimizer and those that are relaxed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreSplit {
    /// Atoms of the core (0-based, ascending).
    pub core: Vec<usize>,
    /// Atoms of the environment that are relaxed (0-based, ascending).
    pub environment: Vec<usize>,
    /// Atoms of the environment held where they are (`-fix`; 0-based, ascending).
    pub fixed: Vec<usize>,
}

impl CoreSplit {
    /// The outermost model region and the outer atoms of every link, for `n_atoms` atoms; the
    /// atoms of `fixed` outside the core stay where they are.
    pub fn of(regions: &[Region], n_atoms: usize, fixed: &[usize]) -> MoptResult<Self> {
        Self::with_core(regions, n_atoms, fixed, &[])
    }

    /// [`Self::of`] with the atoms `promoted` in the core as well: atoms of the environment
    /// that a constraint or a bias potential acts on are moved by the optimizer (which holds
    /// the constraint and feels the bias), and the rest of the environment is relaxed around
    /// them. Any choice of the core gives the same stationary points of the whole surface.
    pub fn with_core(
        regions: &[Region],
        n_atoms: usize,
        fixed: &[usize],
        promoted: &[usize],
    ) -> MoptResult<Self> {
        let mut in_core = vec![false; n_atoms];
        for &a in promoted {
            if a < n_atoms {
                in_core[a] = true;
            }
        }
        for region in regions {
            for a in &region.atoms {
                in_core[a.index()] = true;
            }
            for link in &region.links {
                in_core[link.outer.index()] = true;
            }
        }
        let mut held = vec![false; n_atoms];
        for &a in fixed {
            if a < n_atoms {
                held[a] = true;
            }
        }
        let core: Vec<usize> = (0..n_atoms).filter(|&i| in_core[i]).collect();
        let environment: Vec<usize> = (0..n_atoms).filter(|&i| !in_core[i] && !held[i]).collect();
        let fixed: Vec<usize> = (0..n_atoms).filter(|&i| !in_core[i] && held[i]).collect();
        if environment.is_empty() {
            return Err(MoptError::config(
                "ONIOM microiterations: every atom is in a model region, holds a link atom, is \
                 fixed, or is held by a constraint or a bias potential; there is no \
                 environment to relax",
            ));
        }
        Ok(Self {
            core,
            environment,
            fixed,
        })
    }
}

/// The coordinates in which the limited-memory BFGS relaxes the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnvironmentCoordinates {
    /// Cartesian coordinates.
    #[default]
    Cartesian,
    /// Translation, rotation, and internal coordinates of each fragment ([`fragments`]).
    Fragments,
}

/// The Hessian of the core given to the optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreHessian {
    /// The Hessian of the relaxed surface, `H_xx - H_xy H_yy^-1 H_yx`, with `H_yy^-1` by
    /// conjugate gradients on Hessian-vector products of the lowest level.
    #[default]
    Effective,
    /// The same with `H_yy^-1` replaced by the inverse of the limited-memory BFGS of the
    /// relaxations (no gradients beyond those of `H_xx` and `H_yx`).
    Approximate,
    /// The core block `H_xx` with the environment held.
    Held,
}

/// Largest tolerance of an adaptive relaxation (Hartree/Bohr).
pub const ADAPTIVE_CAP: f64 = 1e-3;

/// Settings of the microiterations.
#[derive(Debug, Clone, PartialEq)]
pub struct MicroOniomSettings {
    /// The relaxation of the environment.
    pub inner: MicroSettings,
    /// Mode kill in the first relaxation and then in every this many (0: only the first).
    pub mode_kill_every: usize,
    /// Displacement of the central differences of the core Hessian (Bohr).
    pub hessian_step_bohr: f64,
    /// The Hessian of the core.
    pub core_hessian: CoreHessian,
    /// Start each relaxation from the predicted response of the environment (the `response`
    /// module).
    pub predict: bool,
    /// The coordinates of the limited-memory BFGS.
    pub coordinates: EnvironmentCoordinates,
    /// Tolerance of a relaxation as this share of the largest core gradient of the last
    /// evaluation, between the tolerance of `inner` and [`ADAPTIVE_CAP`] (0: always the
    /// tolerance of `inner`).
    pub adaptive: f64,
    /// The ONIOM surface has electronic embedding: the model systems see the charges of the
    /// environment, so the environment is relaxed on the whole surface (every inner step
    /// computes the higher levels too) instead of on the lowest level alone.
    pub embedded: bool,
    /// The environment block of a model Hessian as the preconditioner of the relaxations
    /// (`micro_precondition`; `None`: none).
    pub precondition: Option<PreconditionSettings>,
    /// The memory the Hessian of the core may take (GiB, `micro_hessian_memory`; `None`: half
    /// of the free memory of the machine, [`crate::core_infra::memory::budget_bytes`]). The
    /// analytic Hessian of the whole system is taken where its matrices fit (about
    /// `8 (3 N^2 + 2 m^2)` bytes for `N` coordinates of the whole system and `m` of the
    /// environment); the differences otherwise, which take a few `m x n` blocks.
    pub hessian_memory: Option<f64>,
    /// Only the atoms of the environment within this distance (Bohr) of an atom of the core
    /// respond in the Hessian of the core (`micro_hessian_shell`, given in Angstrom); the rest
    /// is held. `None`: all of them.
    pub hessian_shell: Option<f64>,
    /// How the response of the environment in the Hessian of the core is solved
    /// (`micro_hessian_solve`).
    pub response_solve: ResponseSolve,
    /// Correct the energy and the core gradient of a relaxation for the gradient left in the
    /// environment, by the dense response of the newest Hessian of the core
    /// (`micro_correct`; [`RelaxedEnvironment`]).
    pub correct: bool,
    /// The tolerance of the environment (largest component and RMS of its gradient,
    /// Hartree/Bohr) when a Hessian for a check of stability is taken
    /// ([`EvalRequest::for_stability`]: the mode kill at the convergence of an optimization,
    /// the start of an IRC, the minima and saddle points of the mapper, a vibrational analysis):
    /// one setting for every check, so that they all judge the same surface whatever the
    /// tolerance of the run that asks (`micro_check_tolerance`; `None`: that of `inner`).
    pub check_tolerance: Option<(f64, f64)>,
    /// Replicas of the environment combined by a soft minimum (`micro_replicas`; one by
    /// default: a single environment).
    pub replicas: ReplicaSettings,
}

impl Default for MicroOniomSettings {
    fn default() -> Self {
        Self {
            inner: MicroSettings {
                // Every relaxation runs to convergence, however many iterations it takes; this
                // only stops one that cannot converge (a calculator at its numerical floor).
                max_iterations: 200_000,
                // The thresholds of the forces of optmain's default criteria; a run gives
                // its own (`MicroOptions::settings_for`).
                gradient_tolerance: 3.0e-4,
                rms_tolerance: 2.0e-4,
                max_step: 0.3,
                // The low layer is not mode-killed by default (user, 2026-11-20): the core
                // is, by its optimizer; `micro_mode_kill=N` turns this one on.
                mode_kill: crate::pes::microiter::ModeKill {
                    enabled: false,
                    ..crate::pes::microiter::ModeKill::default()
                },
                ..MicroSettings::default()
            },
            // With the mode kill of the environment on: the first relaxation and every
            // this many.
            mode_kill_every: 1,
            hessian_step_bohr: 5.0e-3,
            core_hessian: CoreHessian::Effective,
            predict: false,
            coordinates: EnvironmentCoordinates::Cartesian,
            adaptive: 0.0,
            embedded: false,
            // Measured: the relaxations take 2.5 to 4 times fewer gradients and the check of the
            // curvature converges (docs/source/oniom.md); relaxations that go wrong with it
            // are made again without it.
            precondition: Some(PreconditionSettings {
                model: PRECONDITION_MODEL.to_string(),
                shift: PRECONDITION_SHIFT,
            }),
            hessian_memory: None,
            hessian_shell: None,
            response_solve: ResponseSolve::Auto,
            correct: true,
            check_tolerance: None,
            replicas: ReplicaSettings::default(),
        }
    }
}

/// What the relaxations did so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MicroCounts {
    /// Relaxations of the environment.
    pub relaxations: usize,
    /// Inner iterations of all relaxations.
    pub inner_iterations: usize,
    /// Negative-curvature modes of the environment removed.
    pub killed_modes: usize,
    /// Relaxations that did not meet the gradient threshold.
    pub unconverged: usize,
    /// Of those, relaxations that stopped where the energy no longer decreased.
    pub stalled: usize,
    /// Relaxations started from the predicted response of the environment.
    pub predicted: usize,
    /// Relaxations to a tolerance looser than the full one (`adaptive`).
    pub loose: usize,
    /// Relaxations that went on with the method without line searches after a line search
    /// failed ([`crate::pes::microiter::FALLBACK_METHOD`]).
    pub fallbacks: usize,
    /// Inner iterations after those fallbacks.
    pub fallback_iterations: usize,
    /// Gradients of the lowest level in the relaxations (every evaluation of the
    /// environment: the iterations, the trial points of the line searches, and the
    /// curvature checks).
    pub environment_gradients: usize,
    /// Checks of the curvature of the environment.
    pub curvature_checks: usize,
    /// Their Hessian-vector products (two gradients each).
    pub curvature_products: usize,
    /// Checks that ended at their limit of products with the residual above the tolerance.
    pub unconverged_checks: usize,
    /// Factorizations of the model of the preconditioner (`micro_precondition`).
    pub preconditioner_builds: usize,
    /// Environments given back by the journal of a restarted run.
    pub recalled: usize,
    /// Relaxations with the preconditioner made again without it (they failed or stopped
    /// short of the threshold).
    pub preconditioner_retries: usize,
    /// Trial structures of the relaxations with an atom of the environment within 0.5
    /// Angstrom of another atom (refused).
    pub contacts: usize,
    /// Relaxations that took the environment to another local minimum (an atom of it moved
    /// far more than the core did).
    pub jumps: usize,
    /// Hessians of the core whose environment stood on a saddle point of its own and was
    /// moved off it and relaxed again.
    pub environment_saddles: usize,
    /// Hessians taken with the environment still on a saddle point of its own after the
    /// attempts to leave it.
    pub environment_saddles_kept: usize,
}

impl MicroCounts {
    /// The counts of two surfaces together (the replicas of the environment).
    pub fn plus(&self, o: &Self) -> Self {
        Self {
            relaxations: self.relaxations + o.relaxations,
            inner_iterations: self.inner_iterations + o.inner_iterations,
            killed_modes: self.killed_modes + o.killed_modes,
            unconverged: self.unconverged + o.unconverged,
            stalled: self.stalled + o.stalled,
            predicted: self.predicted + o.predicted,
            loose: self.loose + o.loose,
            fallbacks: self.fallbacks + o.fallbacks,
            fallback_iterations: self.fallback_iterations + o.fallback_iterations,
            environment_gradients: self.environment_gradients + o.environment_gradients,
            curvature_checks: self.curvature_checks + o.curvature_checks,
            curvature_products: self.curvature_products + o.curvature_products,
            unconverged_checks: self.unconverged_checks + o.unconverged_checks,
            preconditioner_builds: self.preconditioner_builds + o.preconditioner_builds,
            recalled: self.recalled + o.recalled,
            preconditioner_retries: self.preconditioner_retries + o.preconditioner_retries,
            contacts: self.contacts + o.contacts,
            jumps: self.jumps + o.jumps,
            environment_saddles: self.environment_saddles + o.environment_saddles,
            environment_saddles_kept: self.environment_saddles_kept + o.environment_saddles_kept,
        }
    }
}

/// What one relaxation did, counted while it runs.
#[derive(Debug, Default)]
struct Tally {
    evaluations: std::cell::Cell<usize>,
    checks: std::cell::Cell<usize>,
    products: std::cell::Cell<usize>,
    unconverged_checks: std::cell::Cell<usize>,
    /// Inner iterations when the line search failed (the FIRE iterations are those after).
    fallback_at: std::cell::Cell<Option<usize>>,
    /// Trial structures refused by the contact guard.
    contacts: std::cell::Cell<usize>,
}

/// The last relaxation to the full tolerance after an adaptive run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullRelaxation {
    /// Change of the energy of the surface (Hartree).
    pub energy_change: f64,
    /// Largest gradient component of the core at the end (Hartree/Bohr).
    pub max_core_gradient: f64,
}

/// A core structure with an environment: relaxed for it, or the one of a structure of the
/// input ([`RelaxedEnvironment::remember`]).
struct Kept {
    core: Vec<f64>,
    environment: Vec<f64>,
    /// Whether the environment was relaxed to the full tolerance at this core.
    relaxed: bool,
    /// The tolerance of the largest gradient component it was relaxed to (infinite where it
    /// was not relaxed): a check of stability takes it only where that is its own or tighter.
    tolerance: f64,
}

/// How a relaxation of [`RelaxedEnvironment::relax_at`] is made, beyond the settings of the run.
#[derive(Clone, Copy, Default)]
struct Relax<'a> {
    /// The tolerance (largest component and RMS) in the place of that of the run (the one of
    /// the checks of stability).
    tolerance: Option<(f64, f64)>,
    /// The start in the place of the environment of the nearest kept core (an environment
    /// moved off a saddle point of its own).
    from: Option<&'a [f64]>,
}

/// How often an evaluation with a Hessian moves its environment off a saddle point of the
/// environment and relaxes it again before it takes the Hessian as it is (and does not judge a
/// check of stability on it).
const ENVIRONMENT_ESCAPES: usize = 3;

/// A core structure is the same as a kept one below this largest difference (Bohr).
const SAME_STRUCTURE: f64 = 1e-12;

/// A relaxation that moves an atom of the environment farther than this (Bohr, 0.5
/// Angstrom) and [`JUMP_RATIO`] times farther than the largest step of an atom of the core
/// took the environment to another local minimum.
const JUMP_BOHR: f64 = 0.945;
/// See [`JUMP_BOHR`].
const JUMP_RATIO: f64 = 10.0;

/// An atom of the environment this close to another atom (Bohr, 0.5 Angstrom) is a structure
/// the relaxation does not go to: no two atoms of a molecule are that close (H2 is 0.74
/// Angstrom), and an energy surface with a hole (point charges of electronic embedding next
/// to the atoms of the higher level) would pull an atom into it.
const CONTACT_BOHR: f64 = 0.944_863;

/// Factorized preconditioners kept at once at most (one per image of a chain of up to this
/// many images; the least recently used is dropped), within [`PRECONDITIONER_BYTES`].
const PRECONDITIONERS_KEPT: usize = 16;
/// Memory of the kept preconditioners (bytes): each is a dense factor of the environment
/// block, 32 MB for 2000 variables.
const PRECONDITIONER_BYTES: usize = 512 << 20;

/// How many preconditioners of an environment of `variables` are kept.
fn preconditioners_kept(variables: usize) -> usize {
    let each = 8 * variables.max(1) * variables.max(1);
    (PRECONDITIONER_BYTES / each).clamp(1, PRECONDITIONERS_KEPT)
}

/// A relaxation with the preconditioner that fails is made again without it; after this many
/// (and a quarter of the relaxations), the run goes on without the preconditioner.
const PRECONDITIONER_RETRIES: usize = 3;

/// A bond of the start (closer than this times the sum of the covalent radii) that ends
/// farther than [`BOND_BROKEN`] times the sum is broken; a pair that ends closer than
/// [`BOND_FORMED`] times the sum and was not a bond is formed. The margins keep a bond near the
/// cutoff from counting as a change.
const BOND: f64 = 1.25;
/// See [`BOND`].
const BOND_BROKEN: f64 = 1.5;
/// See [`BOND`].
const BOND_FORMED: f64 = 1.1;

/// The first bond of an atom of `environment` that `end` broke or formed against `start`
/// (atoms numbered from 1, for the log), if any.
fn bonds_changed(
    start: &MolecularSystem,
    end: &MolecularSystem,
    environment: &[bool],
) -> Option<String> {
    let radius: Vec<f64> = start
        .elements
        .iter()
        .map(|&e| crate::parameters::covalent::covalent_radius_bohr(e))
        .collect();
    let largest = radius.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = Quantity::new(2.0 * BOND_BROKEN * largest);
    let pairs = |sys: &MolecularSystem| {
        let mut out = std::collections::BTreeMap::new();
        for (i, list) in crate::pes::native::neighbors::neighbor_list(sys, cutoff)
            .iter()
            .enumerate()
        {
            for n in list {
                let j = n.atom.index();
                let involved = environment.get(i).copied().unwrap_or(false)
                    || environment.get(j).copied().unwrap_or(false);
                if j <= i || n.self_image || !involved {
                    continue;
                }
                let sum =
                    radius.get(i).copied().unwrap_or(0.0) + radius.get(j).copied().unwrap_or(0.0);
                let ratio = n.distance / sum.max(f64::MIN_POSITIVE);
                out.entry((i, j))
                    .and_modify(|r: &mut f64| *r = r.min(ratio))
                    .or_insert(ratio);
            }
        }
        out
    };
    let (before, after) = (pairs(start), pairs(end));
    for (&(i, j), &r0) in &before {
        let r1 = after.get(&(i, j)).copied().unwrap_or(f64::INFINITY);
        if r0 < BOND && r1 > BOND_BROKEN {
            return Some(format!("the bond of atoms {} and {} broke", i + 1, j + 1));
        }
    }
    for (&(i, j), &r1) in &after {
        let r0 = before.get(&(i, j)).copied().unwrap_or(f64::INFINITY);
        if r1 < BOND_FORMED && r0 >= BOND {
            return Some(format!("atoms {} and {} formed a bond", i + 1, j + 1));
        }
    }
    None
}

/// The pair closest below `CONTACT_BOHR` with an atom of `environment` in it, if any.
fn contact(sys: &MolecularSystem, environment: &[bool]) -> Option<(usize, usize, f64)> {
    let neighbors = crate::pes::native::neighbors::neighbor_list(sys, Quantity::new(CONTACT_BOHR));
    let mut closest: Option<(usize, usize, f64)> = None;
    for (i, list) in neighbors.iter().enumerate() {
        for n in list {
            let j = n.atom.index();
            let involved = environment.get(i).copied().unwrap_or(false)
                || environment.get(j).copied().unwrap_or(false);
            if n.self_image || !involved || j == i {
                continue;
            }
            if closest.is_none_or(|c| n.distance < c.2) {
                closest = Some((i, j, n.distance));
            }
        }
    }
    closest
}

struct State {
    /// The last relaxed environment (flat, Bohr), the start of a relaxation when no core
    /// structure was kept yet.
    environment: Vec<f64>,
    /// Evaluated core structures and their relaxed environments.
    kept: VecDeque<Kept>,
    /// Charge, multiplicity, and state of the relaxations so far, where the environment
    /// depends on them (a lowest level with electrons, electronic embedding).
    inner_state: Option<(SpinCharge, usize)>,
    counts: MicroCounts,
    /// The limited-memory BFGS pairs of the last relaxation, the start of the next one.
    memory: MicroMemory,
    /// The model of the response of the environment (with `predict`).
    response: Option<Response>,
    /// The dense response of the newest Hessian of the core, for the correction of the
    /// gradient of a relaxation that stopped short of a minimum.
    dense: Option<std::sync::Arc<hessian::DenseResponse>>,
    /// The coordinates of the limited-memory BFGS (`None`: Cartesian).
    fragments: Option<FragmentCoordinates>,
    /// The factorized models of the preconditioner, each kept while the free atoms stay near
    /// the structure it was built at (the most recently used last; at most
    /// [`PRECONDITIONERS_KEPT`], so that the images of a chain each keep theirs).
    preconditioners: VecDeque<std::sync::Arc<precondition::EnvironmentPreconditioner>>,
    /// The preconditioner could not be had (too large an environment, a model that cannot be
    /// built, relaxations that failed with it): said once, and the relaxations go on without it.
    preconditioner_refused: bool,
    /// The atoms of the environment, by atom of the whole system (the contact guard).
    environment_atoms: Vec<bool>,
    /// Largest gradient component of the core at the last evaluation (and its energy).
    last_core: Option<(f64, f64)>,
    /// Charge, multiplicity, and state of the last evaluation.
    last_state: Option<(SpinCharge, usize)>,
    /// Where the relaxations tell what they do, and every how many inner iterations they
    /// tell their gradient (`None`: nowhere).
    progress: Option<(Logger, usize)>,
    /// The Hessian of the last check of stability before its environment was moved to the
    /// minimum of its quadratic model (a replica: the error of the combined curvatures).
    check_first: Option<faer::Mat<f64>>,
}

/// The surface `E*(x) = min_y E_ONIOM(x, y)` of the core `x`, as a calculator of core
/// structures; with replicas of the environment, the soft minimum of theirs ([`replicas`]).
pub struct RelaxedEnvironment {
    oniom: Arc<dyn Calculator>,
    low: Arc<dyn Calculator>,
    template: MolecularSystem,
    split: CoreSplit,
    settings: MicroOniomSettings,
    name: String,
    state: Mutex<State>,
    /// Where every relaxation is counted besides [`Self::counts`] (the counters of the
    /// registry of the run).
    counters: Option<std::sync::Arc<crate::pes::counting::EvaluationCounters>>,
    /// How the Hessian of the core is made.
    routes: hessian::HessianRoutes,
    /// The terms of the model systems apart (mechanical embedding): with replicas, computed
    /// once per core for all of them.
    model_part: Option<crate::pes::oniom::ModelPart>,
    /// The replicas of the environment (`micro_replicas` above 1), each a surface of one
    /// environment on the same calculators; empty with one environment.
    replicas: Vec<RelaxedEnvironment>,
    /// This surface is replica `r` of `K` of another one.
    replica_of: Option<(usize, usize)>,
    /// The Hessians of the model systems shared by the replicas of a surface (`None`: one
    /// environment, computed at every Hessian as before).
    model_hessians: Option<replicas::ModelHessians>,
    /// What the replicas did (a surface with replicas).
    replica_record: Mutex<replicas::ReplicaRecord>,
}

/// The environment of the lowest level at a fixed core.
struct LowEnvironment<'a> {
    low: &'a dyn Calculator,
    template: &'a MolecularSystem,
    split: &'a CoreSplit,
    core: &'a Col<f64>,
    /// Charge and multiplicity of the whole structure.
    spin: SpinCharge,
    /// Energy and gradient, of the state the surface of the relaxation has.
    request: EvalRequest,
    /// Where the relaxation tells what it does: the log, every how many inner iterations
    /// the gradient is told, and the number of the relaxation.
    report: Option<(Logger, usize, usize)>,
    /// What the relaxation did.
    tally: Tally,
    /// The preconditioner of the relaxation.
    preconditioner: Option<&'a precondition::EnvironmentPreconditioner>,
    /// The atoms of the environment (by atom of the whole system), for the contact guard;
    /// `None`: no guard (the start itself has atoms that close).
    guard: Option<&'a [bool]>,
}

impl<'a> LowEnvironment<'a> {
    /// The same problem without the preconditioner, with a tally of its own.
    fn without_preconditioner(&self) -> Self {
        Self {
            low: self.low,
            template: self.template,
            split: self.split,
            core: self.core,
            spin: self.spin,
            request: self.request,
            report: self.report.clone(),
            tally: Tally::default(),
            preconditioner: None,
            guard: self.guard,
        }
    }
}

impl MicroProblem for LowEnvironment<'_> {
    fn dim(&self) -> usize {
        3 * self.split.environment.len()
    }

    fn evaluate(&self, y: &[f64], _need_hessian: bool) -> MoptResult<MicroEval> {
        let mut full = assemble(self.template, self.split, self.core, y)?;
        full.spin_charge = self.spin;
        if let Some(environment) = self.guard {
            if contact(&full, environment).is_some() {
                // Not a structure of the relaxation: no energy, and the step is shortened.
                self.tally.contacts.set(self.tally.contacts.get() + 1);
                return Ok(MicroEval {
                    energy: f64::INFINITY,
                    gradient: Col::zeros(y.len()),
                    hessian: None,
                });
            }
        }
        let out = self.low.evaluate(&full, &self.request)?;
        self.tally.evaluations.set(self.tally.evaluations.get() + 1);
        Ok(MicroEval {
            energy: out.energy.value(),
            gradient: gather(&out.gradient_bohr, &self.split.environment),
            hessian: None,
        })
    }

    fn has_hessian(&self) -> bool {
        false
    }

    fn precondition(&self, v: &Col<f64>) -> Option<Col<f64>> {
        self.preconditioner.and_then(|p| p.apply(v))
    }

    fn progress(&self, event: &MicroProgress) {
        let t = &self.tally;
        match *event {
            MicroProgress::Curvature {
                products,
                converged,
                ..
            } => {
                t.checks.set(t.checks.get() + 1);
                t.products.set(t.products.get() + products);
                if !converged {
                    t.unconverged_checks.set(t.unconverged_checks.get() + 1);
                }
            }
            MicroProgress::Fallback { iteration, .. } if t.fallback_at.get().is_none() => {
                t.fallback_at.set(Some(iteration));
            }
            _ => {}
        }
        let Some((log, every, k)) = &self.report else {
            return;
        };
        match *event {
            MicroProgress::Iteration {
                iteration,
                energy,
                max_gradient,
                rms_gradient,
            } => {
                if iteration % every == 0 {
                    log.info(&format!(
                        "  microiteration {k}: inner {iteration:>6}  E = {energy:.10}  largest \
                         gradient {max_gradient:.3e}  RMS {rms_gradient:.3e}"
                    ));
                }
            }
            MicroProgress::Fallback { iteration, method } => log.info(&format!(
                "  microiteration {k}: no step along the search direction lowers the energy \
                 after {iteration} inner iteration(s); on with {method}"
            )),
            MicroProgress::FallbackFailed {
                iteration,
                method,
                reason,
            } => log.warn(&format!(
                "ONIOM microiterations: relaxation {k}: {method} stopped after {iteration} inner \
                 iteration(s) because {reason}; the relaxation ends where the line searches left \
                 it, short of its thresholds"
            )),
            MicroProgress::Ended {
                iterations,
                max_gradient,
                converged,
            } => log.info(&format!(
                "  microiteration {k}: {} after {iterations} inner iteration(s), largest \
                 gradient {max_gradient:.3e}",
                if converged {
                    "converged"
                } else {
                    "stopped short of the thresholds"
                }
            )),
            MicroProgress::Curvature {
                lowest,
                negative,
                products,
                residual,
                converged,
            } => log.info(&format!(
                "  microiteration {k}: lowest curvature of the environment {lowest:+.3e} \
                 Hartree/Bohr^2 ({products} Hessian-vector products, residual {residual:.1e}{}): \
                 {}",
                if converged { "" } else { ", not converged" },
                if negative {
                    "a saddle point; its mode is followed downhill"
                } else {
                    "a minimum"
                }
            )),
            MicroProgress::Left { lower } => log.info(&format!(
                "  microiteration {k}: {}",
                if lower {
                    "relaxed again to a lower energy"
                } else {
                    "both displacements along the mode relaxed back (the curvature is too \
                     weak to follow)"
                }
            )),
        }
    }
}

/// The flat coordinates of `atoms` taken from `flat`.
fn gather(flat: &Col<f64>, atoms: &[usize]) -> Col<f64> {
    Col::from_fn(3 * atoms.len(), |k| flat[3 * atoms[k / 3] + k % 3])
}

/// Largest component of the difference of the kept core `k` and `x`.
fn distance(k: &[f64], x: &Col<f64>) -> f64 {
    k.iter()
        .zip(x.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
}

/// The kept core structure nearest to `x` (largest component of the difference; the later
/// one of two as near) with its environment.
fn nearest<'a>(kept: &'a VecDeque<Kept>, x: &Col<f64>) -> Option<&'a Kept> {
    kept.iter()
        .rev()
        .min_by(|a, b| distance(&a.core, x).total_cmp(&distance(&b.core, x)))
}

/// Points span a plane where the second moment along their second axis is more than this share
/// of that along the first (atoms on a line: zero).
const PLANAR_SHARE: f64 = 1e-6;

/// A rigid motion of points: `p -> R (p - from) + to` (Bohr).
struct RigidMotion {
    rotation: faer::Mat<f64>,
    from: [f64; 3],
    to: [f64; 3],
}

impl RigidMotion {
    /// The best fit (unweighted, Kabsch) of the points `mobile` onto `target` (flat). Points
    /// that do not span a plane (one atom, or atoms on one line) define no turn about their
    /// line, and the fit is the translation alone.
    fn fit(mobile: &[f64], target: &Col<f64>) -> Self {
        let n = mobile.len() / 3;
        let mut from = [0.0; 3];
        let mut to = [0.0; 3];
        for a in 0..n {
            for c in 0..3 {
                from[c] += mobile[3 * a + c] / n as f64;
                to[c] += target[3 * a + c] / n as f64;
            }
        }
        let mut cov = faer::Mat::<f64>::zeros(3, 3);
        let mut spread = faer::Mat::<f64>::zeros(3, 3);
        for a in 0..n {
            for r in 0..3 {
                for c in 0..3 {
                    cov[(r, c)] += (mobile[3 * a + r] - from[r]) * (target[3 * a + c] - to[c]);
                    spread[(r, c)] += (mobile[3 * a + r] - from[r]) * (mobile[3 * a + c] - from[c]);
                }
            }
        }
        // The second moment of the points along their second axis against the first.
        let planar = crate::math::linalg::symmetric_eigen(&spread)
            .map(|(values, _)| values[1] > PLANAR_SHARE * values[2].max(f64::MIN_POSITIVE))
            .unwrap_or(false);
        Self {
            rotation: if planar {
                crate::coord::alignment::proper_rotation_from_covariance(&cov)
            } else {
                faer::Mat::identity(3, 3)
            },
            from,
            to,
        }
    }

    /// The points `flat` moved.
    fn apply(&self, flat: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; flat.len()];
        for a in 0..flat.len() / 3 {
            let d = [
                flat[3 * a] - self.from[0],
                flat[3 * a + 1] - self.from[1],
                flat[3 * a + 2] - self.from[2],
            ];
            for r in 0..3 {
                out[3 * a + r] =
                    self.to[r] + (0..3).map(|c| self.rotation[(r, c)] * d[c]).sum::<f64>();
            }
        }
        out
    }
}

/// A kept core structure as the start at a new core: the kept core and its environment, both
/// moved by the rigid motion that fits the kept core best onto the new one where rigid
/// motions are symmetries, as they are otherwise.
struct Nearest {
    core: Vec<f64>,
    environment: Vec<f64>,
    relaxed: bool,
    /// The tolerance it was relaxed to ([`Kept::tolerance`]).
    tolerance: f64,
    /// Largest component of the difference of the (moved) kept core and the new one.
    distance: f64,
    /// Whether the rigid motion moved them.
    moved: bool,
}

/// The kept core structure nearest to `x` ([`nearest`]); with `rigid`, nearest after the
/// best-fit rigid motion of every kept core onto `x`, with that motion applied to the core
/// and its environment.
fn nearest_moved(kept: &VecDeque<Kept>, x: &Col<f64>, rigid: bool) -> Option<Nearest> {
    if !rigid {
        return nearest(kept, x).map(|k| Nearest {
            core: k.core.clone(),
            environment: k.environment.clone(),
            relaxed: k.relaxed,
            tolerance: k.tolerance,
            distance: distance(&k.core, x),
            moved: false,
        });
    }
    let mut best: Option<Nearest> = None;
    for k in kept.iter().rev() {
        let motion = RigidMotion::fit(&k.core, x);
        let core = motion.apply(&k.core);
        let d = distance(&core, x);
        if best.as_ref().is_some_and(|b| b.distance <= d) {
            continue;
        }
        best = Some(Nearest {
            moved: distance(&k.core, x) > SAME_STRUCTURE,
            environment: motion.apply(&k.environment),
            core,
            relaxed: k.relaxed,
            tolerance: k.tolerance,
            distance: d,
        });
    }
    best
}

/// The full structure of `template` with the core at `core` and the environment at `env`.
fn assemble(
    template: &MolecularSystem,
    split: &CoreSplit,
    core: &Col<f64>,
    env: &[f64],
) -> MoptResult<MolecularSystem> {
    let mut x = template.coords_flat_bohr();
    for (k, &a) in split.core.iter().enumerate() {
        for c in 0..3 {
            x[3 * a + c] = core[3 * k + c];
        }
    }
    for (k, &a) in split.environment.iter().enumerate() {
        for c in 0..3 {
            x[3 * a + c] = env[3 * k + c];
        }
    }
    template.with_flat_coords_bohr(&x)
}

impl RelaxedEnvironment {
    /// The microiterations of `oniom` on structures like `template`, with `low` the lowest
    /// level on the real system.
    pub fn new(
        oniom: crate::pes::oniom::WholeOniom,
        low: Box<dyn Calculator>,
        template: MolecularSystem,
        split: CoreSplit,
        settings: MicroOniomSettings,
    ) -> MoptResult<Self> {
        let (oniom, model) = oniom.under_microiterations();
        let oniom: Arc<dyn Calculator> = Arc::from(oniom);
        let low: Arc<dyn Calculator> = Arc::from(low);
        let count = settings.replicas.count;
        if count == 0 {
            return Err(MoptError::config(
                "-oniom_opt micro_replicas=0: one environment (1) or more",
            ));
        }
        let mut surface = Self::one(
            Arc::clone(&oniom),
            Arc::clone(&low),
            model.clone(),
            template.clone(),
            split.clone(),
            settings.clone(),
            None,
        )?;
        if count == 1 {
            return Ok(surface);
        }
        let r = &settings.replicas;
        if !(r.temperature.is_finite() && r.temperature > 0.0) {
            return Err(MoptError::config(format!(
                "-oniom_opt micro_replica_temperature={}: a temperature above 0 K",
                r.temperature
            )));
        }
        if r.start == ReplicaStart::Dynamics && r.dynamics.is_none() {
            return Err(MoptError::config(
                "-oniom_opt micro_replica_start=md: the molecular dynamics of the environment \
                 is handed in by the command, and this one gives none; use perturb or files",
            ));
        }
        let cache: replicas::ModelHessians = Arc::new(Mutex::new(VecDeque::new()));
        let mut one = settings.clone();
        one.replicas.count = 1;
        surface.replicas = (0..count)
            .map(|k| {
                let mut replica = Self::one(
                    Arc::clone(&oniom),
                    Arc::clone(&low),
                    model.clone(),
                    template.clone(),
                    split.clone(),
                    one.clone(),
                    Some((k, count)),
                )?;
                replica.model_hessians = Some(Arc::clone(&cache));
                Ok(replica)
            })
            .collect::<MoptResult<Vec<_>>>()?;
        surface.model_hessians = Some(cache);
        let frames = surface.read_replica_frames()?;
        surface.keep_replica_frames(frames);
        Ok(surface)
    }

    /// The surface of one environment (`replica_of`: replica `r` of `K` of another surface).
    fn one(
        oniom: Arc<dyn Calculator>,
        low: Arc<dyn Calculator>,
        model: Option<crate::pes::oniom::ModelPart>,
        template: MolecularSystem,
        split: CoreSplit,
        settings: MicroOniomSettings,
        replica_of: Option<(usize, usize)>,
    ) -> MoptResult<Self> {
        let x = template.coords_flat_bohr();
        let environment: Vec<f64> = gather(&x, &split.environment).iter().copied().collect();
        let name = format!("{} (microiterations)", oniom.name());
        let (m, n) = (3 * split.environment.len(), 3 * split.core.len());
        let response = (settings.predict && m * n <= RESPONSE_ENTRIES).then(|| Response::new(m, n));
        let fragments = match settings.coordinates {
            EnvironmentCoordinates::Cartesian => None,
            EnvironmentCoordinates::Fragments => Some(FragmentCoordinates::new(
                split
                    .environment
                    .iter()
                    .map(|&a| template.elements[a])
                    .collect(),
                &environment,
                template.cell.as_ref(),
            )?),
        };
        let mut environment_atoms = vec![false; template.n_atoms()];
        for &a in &split.environment {
            if let Some(f) = environment_atoms.get_mut(a) {
                *f = true;
            }
        }
        let shell = hessian::responding_rows(&template, &split, settings.hessian_shell).len();
        let model_part = model.clone().filter(|_| !settings.embedded);
        let routes = hessian::HessianRoutes::choose(
            oniom.as_ref(),
            model,
            template.dof(),
            (n, m, shell),
            &settings,
        );
        Ok(Self {
            routes,
            model_part,
            replicas: Vec::new(),
            replica_of,
            model_hessians: None,
            replica_record: Mutex::new(replicas::ReplicaRecord::default()),
            oniom,
            low,
            template,
            split,
            settings,
            name,
            state: Mutex::new(State {
                environment,
                kept: VecDeque::new(),
                inner_state: None,
                counts: MicroCounts::default(),
                memory: MicroMemory::default(),
                response,
                dense: None,
                fragments,
                preconditioners: VecDeque::new(),
                preconditioner_refused: false,
                environment_atoms,
                last_core: None,
                last_state: None,
                progress: None,
                check_first: None,
            }),
            counters: None,
        })
    }

    /// The free atoms of the environment within each of `radii` (Angstrom) of an atom of the
    /// core in the structure the surface was made from (minimum-image distances in a periodic
    /// cell): the atoms that `micro_hessian_shell` with that radius lets respond in the
    /// Hessian of the core, for the choice of the radius.
    pub fn shell_counts(&self, radii: &[f64]) -> Vec<(f64, usize)> {
        radii
            .iter()
            .map(|&r| {
                let rows = hessian::responding_rows(
                    &self.template,
                    &self.split,
                    Some(r * crate::domain::Units::ANGSTROM_TO_BOHR),
                );
                (r, rows.len() / 3)
            })
            .collect()
    }

    /// The same, with every relaxation also counted in `counters` (those of the registry of
    /// the run, which the command writes and its report shows).
    pub fn with_counters(
        mut self,
        counters: std::sync::Arc<crate::pes::counting::EvaluationCounters>,
    ) -> Self {
        self.replicas = std::mem::take(&mut self.replicas)
            .into_iter()
            .map(|r| r.with_counters(Arc::clone(&counters)))
            .collect();
        self.counters = Some(counters);
        self
    }

    /// What the settings ask for that the environment cannot have, for the log: the
    /// prediction of an environment too large for the response matrix, and the blocks of
    /// the fragment coordinates.
    pub fn setup_notes(&self) -> Vec<String> {
        let mut out = self.routes.notes.clone();
        out.extend(self.replica_notes());
        let Ok(state) = self.state.lock() else {
            return out;
        };
        if self.settings.predict && state.response.is_none() {
            out.push(format!(
                "ONIOM microiterations: no predicted response (the response matrix of {} x {} \
                 entries is above {RESPONSE_ENTRIES}).",
                3 * self.split.environment.len(),
                3 * self.split.core.len()
            ));
        }
        if let Some(f) = &state.fragments {
            let (internal, cartesian) = f.block_counts();
            out.push(format!(
                "ONIOM microiterations: the environment relaxed in the coordinates of its \
                 fragments, {internal} block(s) with internal coordinates and {cartesian} with \
                 Cartesian ones."
            ));
        }
        out
    }

    /// The core and the environment.
    pub fn split(&self) -> &CoreSplit {
        &self.split
    }

    /// The surface the environment is relaxed on: the lowest level on the whole structure,
    /// or the whole ONIOM surface with electronic embedding.
    fn inner(&self) -> &dyn Calculator {
        if self.settings.embedded {
            self.oniom.as_ref()
        } else {
            self.low.as_ref()
        }
    }

    /// The core of the starting structure, as the optimizer sees it.
    pub fn core_system(&self) -> MoptResult<MolecularSystem> {
        self.core_of(&self.template)
    }

    /// The core of a full structure.
    pub fn core_of(&self, full: &MolecularSystem) -> MoptResult<MolecularSystem> {
        let x = full.coords_flat_bohr();
        let elements = self.split.core.iter().map(|&a| full.elements[a]).collect();
        let mut core = MolecularSystem::from_flat_bohr(
            elements,
            &gather(&x, &self.split.core),
            full.spin_charge,
            full.cell.clone(),
        )?;
        core.comment.clone_from(&full.comment);
        Ok(core)
    }

    /// The whole structure `whole` with its environment relaxed to a minimum at its core, from
    /// the environment it has (for the runs that do not move the core, such as a vibrational
    /// analysis: the structure they compute is a minimum of the environment), with the
    /// largest displacement of a coordinate of the environment (Bohr). The tolerance is that
    /// of the checks of stability ([`MicroOniomSettings::check_tolerance`]): an analysis
    /// judges the structure as they do.
    pub fn relaxed_system(
        &self,
        whole: &MolecularSystem,
        excited_state: usize,
    ) -> MoptResult<(MolecularSystem, f64)> {
        self.remember(whole)?;
        let x = gather(&whole.coords_flat_bohr(), &self.split.core);
        let before = gather(&whole.coords_flat_bohr(), &self.split.environment);
        if !self.replicas.is_empty() {
            // The replicas relaxed to the tolerance of the checks (the soft minimum evaluated
            // as for a check), and the whole structure of the lowest.
            let mut core = self.core_of(whole)?;
            core.spin_charge = whole.spin_charge;
            self.relax_replicas_for_check(&core, excited_state)?;
            let mut full = self.full_system_of_replicas(&core)?;
            let env = gather(&full.coords_flat_bohr(), &self.split.environment);
            let moved = env
                .iter()
                .zip(before.iter())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            full.spin_charge = whole.spin_charge;
            full.comment.clone_from(&whole.comment);
            return Ok((full, moved));
        }
        let how = Relax {
            tolerance: Some(self.check_tolerance()),
            from: None,
        };
        let env = self.relax_at(&x, true, whole.spin_charge, excited_state, how)?;
        let moved = env
            .iter()
            .zip(before.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let mut full = assemble(&self.template, &self.split, &x, &env)?;
        full.spin_charge = whole.spin_charge;
        full.comment.clone_from(&whole.comment);
        Ok((full, moved))
    }

    /// From now on the relaxations write to `log` what they do: the energy and the gradient
    /// every `every` inner iterations (a relaxation has hundreds of them), the end of every
    /// descent, and the lowest curvature at its end (the test for a saddle point of the
    /// environment). The coordinates of the environment are not written.
    pub fn report_to(&self, log: Logger, every: usize) {
        for r in &self.replicas {
            r.report_to(log.clone(), every);
        }
        if let Ok(mut state) = self.state.lock() {
            state.progress = Some((log, every.max(1)));
        }
    }

    /// How many preconditioners of an environment of `variables` this surface keeps: the
    /// replicas of a surface share the memory of one ([`PRECONDITIONER_BYTES`]), each keeping
    /// one at least.
    fn preconditioners_kept(&self, variables: usize) -> usize {
        let share = self.replica_of.map_or(1, |(_, k)| k.max(1));
        (preconditioners_kept(variables) / share).max(1)
    }

    /// How many core structures are kept with their environments.
    fn kept_capacity(&self) -> usize {
        let per_structure = 8 * self.template.dof().max(1);
        // The replicas of a surface share the memory of one store.
        let share = self.replica_of.map_or(1, |(_, k)| k.max(1));
        (KEPT_BYTES / share / per_structure).clamp(256, 1 << 20)
    }

    /// The full structure of a core structure: the environment kept for it (a structure of
    /// the run, read back from a file or not, or a structure of the input). A core structure
    /// that is not kept (a frame of a mode, a structure of a restart that was replayed from
    /// the journal, one dropped from the memory of a long run) has its environment relaxed
    /// here, from that of the nearest kept one, so that every whole structure that is
    /// written has a relaxed environment of its own.
    pub fn full_system(&self, core: &MolecularSystem) -> MoptResult<MolecularSystem> {
        let x = core.coords_flat_bohr();
        if x.nrows() != 3 * self.split.core.len() {
            return Err(MoptError::calc(format!(
                "ONIOM microiterations: a structure of {} atoms for a core of {}",
                x.nrows() / 3,
                self.split.core.len()
            )));
        }
        if !self.replicas.is_empty() {
            return self.full_system_of_replicas(core);
        }
        let (kept, state_of_run) = {
            let state = self
                .state
                .lock()
                .map_err(|_| MoptError::calc("ONIOM microiterations: the state is poisoned"))?;
            (
                nearest_moved(&state.kept, &x, self.carries_environment())
                    .filter(|k| k.distance < SAME_IN_FILES)
                    .map(|k| k.environment),
                state.last_state,
            )
        };
        let env = match kept {
            Some(env) => env,
            None => {
                let (spin, excited_state) = state_of_run.unwrap_or((core.spin_charge, 0));
                self.relax_at(&x, true, spin, excited_state, Relax::default())?
            }
        };
        let mut full = assemble(&self.template, &self.split, &x, &env)?;
        full.comment.clone_from(&core.comment);
        Ok(full)
    }

    /// The full structure of a core structure with the environment of the nearest kept core
    /// as it is (moved with the best-fit rigid motion where rigid motions are symmetries), not
    /// relaxed: for the frames of an analysis (a core displaced along a mode), which show the
    /// displacement and are not structures of the surface. Where nothing is kept, the
    /// environment of the structure the surface was made from.
    pub fn nearest_full_system(&self, core: &MolecularSystem) -> MoptResult<MolecularSystem> {
        let x = core.coords_flat_bohr();
        if x.nrows() != 3 * self.split.core.len() {
            return Err(MoptError::calc(format!(
                "ONIOM microiterations: a structure of {} atoms for a core of {}",
                x.nrows() / 3,
                self.split.core.len()
            )));
        }
        if !self.replicas.is_empty() {
            return self.nearest_full_system_of_replicas(core);
        }
        let env = {
            let state = self
                .state
                .lock()
                .map_err(|_| MoptError::calc("ONIOM microiterations: the state is poisoned"))?;
            match nearest_moved(&state.kept, &x, self.carries_environment()) {
                Some(k) => k.environment,
                None => gather(&self.template.coords_flat_bohr(), &self.split.environment)
                    .iter()
                    .copied()
                    .collect(),
            }
        };
        let mut full = assemble(&self.template, &self.split, &x, &env)?;
        full.comment.clone_from(&core.comment);
        Ok(full)
    }

    /// Keep the environment of the whole structure `full` as the start of the relaxations of
    /// core structures near its core (the structures a chain is made from: each image starts
    /// from the environment of the structure it was made from rather than that of another
    /// image). Not an evaluation: the counts do not change.
    pub fn remember(&self, full: &MolecularSystem) -> MoptResult<()> {
        if full.n_atoms() != self.template.n_atoms() {
            return Err(MoptError::config(format!(
                "ONIOM microiterations: a structure of {} atoms for a system of {}",
                full.n_atoms(),
                self.template.n_atoms()
            )));
        }
        // With replicas the structures of the run are the start of the first one; the others
        // follow their own environments ([`replicas`]).
        if let Some(first) = self.replicas.first() {
            return first.remember(full);
        }
        let x = full.coords_flat_bohr();
        let core: Vec<f64> = gather(&x, &self.split.core).iter().copied().collect();
        let env: Vec<f64> = gather(&x, &self.split.environment)
            .iter()
            .copied()
            .collect();
        let mut state = self
            .state
            .lock()
            .map_err(|_| MoptError::calc("ONIOM microiterations: the state is poisoned"))?;
        if state.kept.len() >= self.kept_capacity() {
            state.kept.pop_front();
        }
        state.kept.push_back(Kept {
            core,
            environment: env,
            relaxed: false,
            tolerance: f64::INFINITY,
        });
        Ok(())
    }

    /// The request of the relaxations for an evaluation of the state `excited_state`: the
    /// ground state of the lowest level, or that state of the whole surface with electronic
    /// embedding.
    fn inner_request(&self, excited_state: usize) -> EvalRequest {
        EvalRequest {
            excited_state: if self.settings.embedded {
                excited_state
            } else {
                0
            },
            ..EvalRequest::ENERGY_GRAD
        }
    }

    /// Whether the environment of a kept core is carried along by the rigid motion of that
    /// core onto a new one: where rigid motions of the whole system are symmetries (no fixed
    /// atoms, no cell, both levels invariant).
    fn carries_environment(&self) -> bool {
        !self.template.has_pbc() && self.is_rigid_body_invariant()
    }

    /// What the relaxations did so far.
    pub fn counts(&self) -> MicroCounts {
        if !self.replicas.is_empty() {
            return self
                .replicas
                .iter()
                .fold(MicroCounts::default(), |sum, r| sum.plus(&r.counts()));
        }
        self.state.lock().map(|s| s.counts).unwrap_or_default()
    }

    /// After a run with an adaptive tolerance (`None` without one, or before any
    /// evaluation): relax the environment of the core `core` to the full tolerance, so that
    /// the final structure has its environment at a minimum as without the adaptive
    /// tolerance, and say how the energy and the core gradient changed.
    pub fn relax_fully(&self, core: &MolecularSystem) -> MoptResult<Option<FullRelaxation>> {
        let before = self.state.lock().ok().and_then(|s| s.last_core);
        let Some((_, energy_before)) = before.filter(|_| self.settings.adaptive > 0.0) else {
            return Ok(None);
        };
        if !self.replicas.is_empty() {
            // Every replica relaxed to the full tolerance, and the soft minimum again.
            for r in &self.replicas {
                r.relax_fully(core)?;
            }
            let (spin, excited_state) = self
                .state
                .lock()
                .ok()
                .and_then(|s| s.last_state)
                .unwrap_or((core.spin_charge, 0));
            let mut at = core.clone();
            at.spin_charge = spin;
            let req = EvalRequest {
                excited_state,
                ..EvalRequest::ENERGY_GRAD
            };
            let out = self.evaluate_replicas(&at, &req)?;
            return Ok(Some(FullRelaxation {
                energy_change: out.energy.value() - energy_before,
                max_core_gradient: crate::math::linalg::col_max_abs(&out.gradient_bohr),
            }));
        }
        // The state of the last evaluation of the run.
        let (spin, excited_state) = self
            .state
            .lock()
            .ok()
            .and_then(|s| s.last_state)
            .unwrap_or((core.spin_charge, 0));
        let x = core.coords_flat_bohr();
        let env = self.relax_at(&x, true, spin, excited_state, Relax::default())?;
        let mut full = assemble(&self.template, &self.split, &x, &env)?;
        full.spin_charge = spin;
        let request = EvalRequest {
            excited_state,
            ..EvalRequest::ENERGY_GRAD
        };
        let out = self.oniom.evaluate(&full, &request)?;
        let g = crate::math::linalg::col_max_abs(&gather(&out.gradient_bohr, &self.split.core));
        Ok(Some(FullRelaxation {
            energy_change: out.energy.value() - energy_before,
            max_core_gradient: g,
        }))
    }

    /// Relax the environment at the core `x`, from that of the nearest kept core (or from the
    /// predicted response). `full`: to the full tolerance whatever `adaptive` says. `how`: a
    /// tolerance of its own (a check of stability) and a start of its own (an environment
    /// moved off a saddle point of its own).
    ///
    /// The state is locked only to read what the relaxation starts from and to store what it
    /// reached: no lock is held across an evaluation of a calculator or the build of the
    /// preconditioner (the pairs of the limited-memory BFGS and the fragment coordinates are
    /// taken out of the state for the relaxation and put back after it).
    fn relax_at(
        &self,
        x: &Col<f64>,
        full: bool,
        spin: SpinCharge,
        excited_state: usize,
        how: Relax<'_>,
    ) -> MoptResult<Vec<f64>> {
        let wanted = how
            .tolerance
            .map_or(self.settings.inner.gradient_tolerance, |t| t.0);
        let poisoned = || MoptError::calc("ONIOM microiterations: the state is poisoned");
        let request = self.inner_request(excited_state);
        let rigid = self.carries_environment();
        let variables = 3 * self.split.environment.len();
        if !self.replicas.is_empty() {
            return Err(MoptError::calc(
                "ONIOM replicas: a relaxation of a surface with replicas is one of its replicas",
            ));
        }
        // What the relaxation starts from, read under the lock.
        let (inner, last, step, response, progress, number, environment_atoms, build) = {
            let mut guard = self.state.lock().map_err(|_| poisoned())?;
            let state = &mut *guard;
            // Where the surface of the relaxation depends on the electrons (a lowest level with
            // electrons, electronic embedding), states of another charge, multiplicity, or root
            // would each have an environment of their own, and a function of several states
            // would compare them at different structures.
            if self.settings.embedded || self.low.uses_electrons() {
                let key = (spin, request.excited_state);
                match state.inner_state {
                    None => state.inner_state = Some(key),
                    Some(first) if first != key => {
                        return Err(MoptError::unsupported(
                            "ONIOM microiterations: states of different charge, multiplicity, \
                             or root would relax the environment differently here (the lowest \
                             level has electrons, or the embedding is electronic); a function \
                             of several states (-mf) needs a lowest level without electrons \
                             and mechanical embedding",
                        ))
                    }
                    Some(_) => {}
                }
            }
            // A core structure whose environment is already relaxed (the same point asked
            // again: another state of -mf, a Hessian after the gradient) is not relaxed again,
            // nor one that is a rigid motion of it (a structure recentred or rotated by an
            // optimizer or the IRC), which takes the environment moved with it.
            // A check of stability takes it only where it was relaxed to its tolerance or a
            // tighter one.
            if !full && how.from.is_none() {
                if let Some(k) = nearest_moved(&state.kept, x, rigid) {
                    if k.relaxed && k.distance < SAME_STRUCTURE && k.tolerance <= wanted {
                        if k.moved {
                            if state.kept.len() >= self.kept_capacity() {
                                state.kept.pop_front();
                            }
                            state.kept.push_back(Kept {
                                core: x.iter().copied().collect(),
                                environment: k.environment.clone(),
                                relaxed: true,
                                tolerance: k.tolerance,
                            });
                            state.environment.clone_from(&k.environment);
                        }
                        return Ok(k.environment);
                    }
                }
            }
            let mut inner = self.settings.inner.clone();
            let n = state.counts.relaxations;
            inner.mode_kill.enabled = self.settings.inner.mode_kill.enabled
                && (n == 0
                    || (self.settings.mode_kill_every > 0
                        && n % self.settings.mode_kill_every == 0));
            if let Some((g, r)) = how.tolerance {
                inner.gradient_tolerance = g;
                inner.rms_tolerance = r;
            } else if self.settings.adaptive > 0.0 && !full {
                if let Some((g, _)) = state.last_core {
                    let loose = (self.settings.adaptive * g).min(ADAPTIVE_CAP);
                    if loose > inner.gradient_tolerance {
                        inner.gradient_tolerance = loose;
                        inner.rms_tolerance = f64::INFINITY;
                        state.counts.loose += 1;
                    }
                }
            }
            // From the environment of the nearest core structure evaluated so far: the last
            // one in an optimization, the image's own one of the iteration before in a chain
            // (whose images are evaluated one after another on the same surface), so that
            // every image follows its own environment; where rigid motions are symmetries,
            // moved with the best-fit rigid motion of that core onto this one. The step of the
            // core from there, for the prediction. A start given by `how` has no step.
            let (last, step) = match (how.from, nearest_moved(&state.kept, x, rigid)) {
                (Some(from), _) => (from.to_vec(), None),
                (None, Some(k)) => (
                    k.environment,
                    Some(Col::from_fn(x.nrows(), |i| x[i] - k.core[i])),
                ),
                (None, None) => (state.environment.clone(), None),
            };
            // The model of the preconditioner: one kept near these free atoms, else built
            // again (outside the lock, below).
            let build = match self
                .settings
                .precondition
                .as_ref()
                .filter(|_| !state.preconditioner_refused)
            {
                None => None,
                Some(pre) => {
                    let near = state
                        .preconditioners
                        .iter()
                        .position(|p| p.is_near(&self.template, &self.split, x, &last));
                    match near {
                        Some(k) => {
                            // The most recently used goes to the back.
                            if let Some(p) = state.preconditioners.remove(k) {
                                state.preconditioners.push_back(p);
                            }
                            None
                        }
                        None if variables > PRECONDITION_LARGEST => {
                            state.preconditioner_refused = true;
                            if let Some((log, _)) = &state.progress {
                                log.info(&format!(
                                    "ONIOM microiterations: no preconditioner for an \
                                     environment of {variables} variables (at most \
                                     {PRECONDITION_LARGEST}: its factor is a dense matrix); \
                                     -oniom_opt micro_precondition=none says so"
                                ));
                            }
                            None
                        }
                        None => Some(pre.clone()),
                    }
                }
            };
            (
                inner,
                last,
                step,
                state.response.clone(),
                state.progress.clone(),
                state.counts.relaxations + 1,
                state.environment_atoms.clone(),
                build,
            )
        };
        if let Some(pre) = build {
            let started = std::time::Instant::now();
            let built = precondition::EnvironmentPreconditioner::build(
                &pre,
                &self.template,
                &self.split,
                x,
                &last,
            );
            let seconds = started.elapsed().as_secs_f64();
            let mut guard = self.state.lock().map_err(|_| poisoned())?;
            let state = &mut *guard;
            match built {
                Ok(p) => {
                    if state.preconditioners.len() >= self.preconditioners_kept(variables) {
                        state.preconditioners.pop_front();
                    }
                    let note = p.describe();
                    state.preconditioners.push_back(std::sync::Arc::new(p));
                    state.counts.preconditioner_builds += 1;
                    if let Some((log, _)) = &state.progress {
                        log.info(&format!(
                            "  microiterations: the preconditioner of the environment ({}, \
                             {variables} variables{note}) built in {seconds:.2} s",
                            pre.model
                        ));
                    }
                }
                Err(e) => {
                    state.preconditioners.clear();
                    state.preconditioner_refused = true;
                    if let Some((log, _)) = &state.progress {
                        log.warn(&format!(
                            "ONIOM microiterations: the relaxations go on without a \
                             preconditioner: {e}"
                        ));
                    }
                }
            }
        }
        // What the relaxation takes along out of the state: the preconditioner (shared), the
        // pairs of the limited-memory BFGS, and the fragment coordinates (put back below,
        // whatever the relaxation gives).
        let (preconditioner, mut memory, mut fragments) = {
            let mut guard = self.state.lock().map_err(|_| poisoned())?;
            let state = &mut *guard;
            let preconditioner = if state.preconditioner_refused {
                None
            } else {
                state
                    .preconditioners
                    .iter()
                    .rev()
                    .find(|p| p.is_near(&self.template, &self.split, x, &last))
                    .cloned()
            };
            (
                preconditioner,
                std::mem::take(&mut state.memory),
                state.fragments.take(),
            )
        };
        let tolerance = (inner.gradient_tolerance, inner.rms_tolerance);
        let outcome = self.relax_unlocked(
            x,
            spin,
            request,
            &inner,
            &last,
            step.as_ref(),
            response.as_ref(),
            preconditioner.as_deref(),
            &environment_atoms,
            progress
                .as_ref()
                .map(|(log, every)| (log.clone(), *every, number)),
            &mut memory,
            &mut fragments,
        );
        let mut guard = self.state.lock().map_err(|_| poisoned())?;
        let state = &mut *guard;
        state.memory = memory;
        state.fragments = fragments;
        let (result, tallies, predicted, retry) = outcome?;
        if predicted {
            state.counts.predicted += 1;
        }
        if let Some(why) = retry {
            state.counts.preconditioner_retries += 1;
            let retries = state.counts.preconditioner_retries;
            if let Some((log, _)) = &state.progress {
                log.info(&format!(
                    "  microiteration {number}: the relaxation with the preconditioner was made \
                     again without it: {why}"
                ));
            }
            if retries >= PRECONDITIONER_RETRIES && 4 * retries > state.counts.relaxations + 1 {
                state.preconditioner_refused = true;
                if let Some((log, _)) = &state.progress {
                    log.warn(&format!(
                        "ONIOM microiterations: {retries} relaxations with the preconditioner had \
                         to be made again without it; the run goes on without the preconditioner"
                    ));
                }
            }
        }
        state.counts.relaxations += 1;
        state.counts.inner_iterations += result.iterations;
        if let Some(c) = &self.counters {
            c.record_micro(result.iterations);
        }
        state.counts.killed_modes += result.killed_modes;
        state.counts.fallbacks += result.fallbacks;
        for t in &tallies {
            state.counts.environment_gradients += t.evaluations.get();
            state.counts.curvature_checks += t.checks.get();
            state.counts.curvature_products += t.products.get();
            state.counts.unconverged_checks += t.unconverged_checks.get();
            state.counts.contacts += t.contacts.get();
            if let Some(at) = t.fallback_at.get() {
                state.counts.fallback_iterations += result.iterations.saturating_sub(at);
            }
        }
        if !result.converged {
            state.counts.unconverged += 1;
            if result.stalled {
                state.counts.stalled += 1;
            }
            // A relaxation that cannot reach its tolerance (a lowest level whose gradient has
            // too few digits, such as single-precision forces) says so instead of being taken
            // for a minimum.
            if let Some((log, _)) = &state.progress {
                log.warn(&format!(
                    "ONIOM microiterations: relaxation {number} ended at a largest gradient of \
                     {:.3e} Hartree/Bohr, above its tolerance {:.1e}{}; the energy and the \
                     gradient of the core there carry that error",
                    result.max_gradient,
                    tolerance.0,
                    if result.stalled {
                        " (the energy stopped falling: the precision of the lowest level)"
                    } else {
                        ""
                    }
                ));
            }
        }
        // An environment that went to another local minimum: an atom of the environment moved
        // much farther than any atom of the core (detection only; the Hessian of the core
        // and the response learned before belong to the other minimum).
        let jumped = step.as_ref().and_then(|dx| {
            let largest = |v: &dyn Fn(usize) -> f64, atoms: usize| {
                (0..atoms)
                    .map(|a| {
                        (v(3 * a).powi(2) + v(3 * a + 1).powi(2) + v(3 * a + 2).powi(2)).sqrt()
                    })
                    .fold(0.0_f64, f64::max)
            };
            let core = largest(&|k| dx[k], dx.nrows() / 3);
            let env = largest(&|k| result.y[k] - last[k], last.len() / 3);
            (env > JUMP_BOHR && env > JUMP_RATIO * core).then_some((env, core))
        });
        if let Some((env, core)) = jumped {
            state.counts.jumps += 1;
            state.dense = None;
            if let Some((log, _)) = &state.progress {
                log.warn(&format!(
                    "ONIOM microiterations: relaxation {number} moved an atom of the environment \
                     by {:.2} Angstrom for a step of the core of at most {:.3} Angstrom: the \
                     environment went to another local minimum, and the surface of the core is \
                     not continuous there (the Hessian of the core and its updates from before \
                     belong to the other minimum)",
                    env * crate::domain::Units::BOHR_TO_ANGSTROM,
                    core * crate::domain::Units::BOHR_TO_ANGSTROM
                ));
            }
        }
        if let (Some(model), Some(dx)) = (state.response.as_mut(), step.as_ref()) {
            // A relaxation that left a saddle point jumped to another basin: no response.
            if result.killed_modes == 0 && jumped.is_none() {
                let dy = Col::from_fn(last.len(), |i| result.y[i] - last[i]);
                model.update(dx, &dy);
            }
        }
        state.environment.clone_from(&result.y);
        if state.kept.len() >= self.kept_capacity() {
            state.kept.pop_front();
        }
        let loose = inner.gradient_tolerance > self.settings.inner.gradient_tolerance;
        let relaxed = result.converged && !loose;
        state.kept.push_back(Kept {
            core: x.iter().copied().collect(),
            environment: result.y.clone(),
            relaxed,
            tolerance: if relaxed {
                inner.gradient_tolerance
            } else {
                f64::INFINITY
            },
        });
        Ok(result.y)
    }

    /// The relaxation of [`Self::relax_at`] itself, without the state: from the prediction of
    /// `response` where it is lower than `last`, with `preconditioner`, and again without it
    /// where it failed with it. Gives the result, the tallies of the evaluations, whether the
    /// prediction was taken, and why the relaxation was made again (if it was).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn relax_unlocked(
        &self,
        x: &Col<f64>,
        spin: SpinCharge,
        request: EvalRequest,
        inner: &MicroSettings,
        last: &[f64],
        step: Option<&Col<f64>>,
        response: Option<&Response>,
        preconditioner: Option<&precondition::EnvironmentPreconditioner>,
        environment_atoms: &[bool],
        report: Option<(Logger, usize, usize)>,
        memory: &mut MicroMemory,
        fragments: &mut Option<FragmentCoordinates>,
    ) -> MoptResult<(
        crate::pes::microiter::MicroResult,
        Vec<Tally>,
        bool,
        Option<String>,
    )> {
        let problem = LowEnvironment {
            low: self.inner(),
            template: &self.template,
            split: &self.split,
            core: x,
            spin,
            request,
            report,
            tally: Tally::default(),
            preconditioner,
            guard: None,
        };
        let mut start = last.to_vec();
        let mut predicted = false;
        if let (Some(model), Some(dx)) = (response, step) {
            if model.is_known() {
                let dy = model.predict(dx);
                let guess: Vec<f64> = last.iter().zip(dy.iter()).map(|(a, b)| a + b).collect();
                let e_predicted = problem.evaluate(&guess, false)?.energy;
                let e_last = problem.evaluate(last, false)?.energy;
                if e_predicted < e_last {
                    start = guess;
                    predicted = true;
                }
            }
        }
        // The contact guard, unless the start itself has atoms that close (then it is the
        // structure given, and the relaxation goes on from it as it is).
        let start_full = assemble(&self.template, &self.split, x, &start)?;
        let problem = LowEnvironment {
            guard: contact(&start_full, environment_atoms)
                .is_none()
                .then_some(environment_atoms),
            ..problem
        };
        let memory_before = memory.clone();
        let coords = fragments
            .as_mut()
            .map(|f| f as &mut dyn crate::pes::microiter::InnerCoordinates);
        let first = relax_in(&problem, &start, inner, memory, coords);
        // A relaxation with the preconditioner that failed or stopped short is made again from
        // the same start without it (the pairs it learned are not kept).
        let changed = match (&first, problem.preconditioner.is_some()) {
            (Ok(r), true) => bonds_changed(
                &start_full,
                &assemble(&self.template, &self.split, x, &r.y)?,
                environment_atoms,
            ),
            _ => None,
        };
        let failed = problem.preconditioner.is_some()
            && match &first {
                Err(_) => true,
                Ok(r) => !r.converged || changed.is_some(),
            };
        if !failed {
            let result = first.map_err(|e| {
                e.context("ONIOM microiterations: the relaxation of the environment")
            })?;
            return Ok((result, vec![problem.tally], predicted, None));
        }
        let why = match (&first, &changed) {
            (Err(e), _) => format!("it failed ({e})"),
            (Ok(_), Some(bond)) => format!("the environment changed its bonds ({bond})"),
            (Ok(r), None) => format!(
                "it stopped short of the threshold (largest gradient {:.3e})",
                r.max_gradient
            ),
        };
        *memory = memory_before;
        let plain = problem.without_preconditioner();
        let coords = fragments
            .as_mut()
            .map(|f| f as &mut dyn crate::pes::microiter::InnerCoordinates);
        let again = relax_in(&plain, &start, inner, memory, coords)
            .map_err(|e| e.context("ONIOM microiterations: the relaxation of the environment"))?;
        Ok((
            again,
            vec![problem.tally, plain.tally],
            predicted,
            Some(why),
        ))
    }
}

impl RelaxedEnvironment {
    /// The tolerance of the environment (largest component and RMS of its gradient,
    /// Hartree/Bohr) for a Hessian of a check of stability
    /// ([`MicroOniomSettings::check_tolerance`], else that of the run), for the log of the
    /// commands that hand a structure on to such a check.
    pub fn check_tolerance(&self) -> (f64, f64) {
        self.settings.check_tolerance.unwrap_or((
            self.settings.inner.gradient_tolerance,
            self.settings.inner.rms_tolerance,
        ))
    }

    /// The energy and the gradient of the core at `x` with the environment at `env` (of the
    /// state of `req`), corrected for the gradient a relaxation left in the environment where
    /// there is a dense response for that, with the line of the log. `model`: the terms of the
    /// model systems computed already at this core (a replica of a surface with mechanical
    /// embedding), to which only the lowest level on the whole system is added here.
    fn surface_at(
        &self,
        x: &Col<f64>,
        env: &[f64],
        spin: SpinCharge,
        req: &EvalRequest,
        model: Option<&replicas::ModelValues>,
    ) -> MoptResult<(EvalOutput, Col<f64>)> {
        let mut full = assemble(&self.template, &self.split, x, env)?;
        full.spin_charge = spin;
        let state_request = EvalRequest {
            excited_state: req.excited_state,
            ..EvalRequest::ENERGY_GRAD
        };
        let out = match model {
            None => self.oniom.evaluate(&full, &state_request)?,
            Some(m) => {
                // `E = E_low(real) + sum_k s_k E_k(M_k)`: the lowest level is in its ground
                // state, as in the ONIOM surface.
                let mut low = crate::pes::calculator::evaluate_checked(
                    self.low.as_ref(),
                    &full,
                    &EvalRequest::ENERGY_GRAD,
                )
                .map_err(|e| e.context("ONIOM: the lowest level on the real system"))?;
                low.energy = Quantity::new(low.energy.value() + m.energy);
                low.gradient_bohr = &low.gradient_bohr + &m.gradient;
                low.orbitals = m.high.as_ref().and_then(|h| h.orbitals.clone());
                low
            }
        };
        let mut gradient = gather(&out.gradient_bohr, &self.split.core);
        let mut energy = out.energy.value();
        // The gradient left in the environment by a relaxation that stopped at its tolerance:
        // to first order the minimum lies `-H_yy^-1 g_y` away, and the energy and the core
        // gradient there differ by `-g_y^T H_yy^-1 g_y / 2` and `-H_xy H_yy^-1 g_y`. Corrected
        // with the dense response of the newest Hessian of the core (where there is one),
        // reported otherwise.
        let residual = gather(&out.gradient_bohr, &self.split.environment);
        let (residual_max, residual_rms) = crate::pes::microiter::gradient_norms(&residual);
        let (dense, progress) = match self.state.lock() {
            Ok(state) => (state.dense.clone(), state.progress.clone()),
            Err(_) => (None, None),
        };
        let attempt = dense
            .as_ref()
            .filter(|_| self.settings.correct)
            .map(|d| d.correction(&residual));
        let core_before = crate::math::linalg::col_max_abs(&gradient);
        let correction = attempt.as_ref().and_then(|a| a.as_ref().ok());
        if let Some((dg, de)) = correction {
            gradient = &gradient + dg;
            energy += de;
        }
        // Where rigid motions of the whole system are symmetries, `E*` does not change when the
        // core moves rigidly (the environment follows it), so its gradient has no component
        // along those motions. The gradient at a relaxed environment has one, of the size of
        // the residual of the relaxation (the net force and torque the environment still
        // exerts): the steps project it out, the test of convergence did not, and a
        // minimization of a water in a shell of water stood still for 1000 iterations with
        // 7e-4 Hartree/Bohr left along a rotation of the core (the residual of its
        // environment was 2.7e-4). It is taken off here.
        let rigid_part = if self.is_rigid_body_invariant() {
            let core = MolecularSystem::from_flat_bohr(
                self.split
                    .core
                    .iter()
                    .map(|&a| self.template.elements[a])
                    .collect(),
                x,
                spin,
                None,
            )?;
            let free = crate::math::projection::project_out_tr_rot_vec(&core, &gradient, false);
            let removed = crate::math::linalg::col_max_abs(&(&gradient - &free));
            gradient = free;
            Some(removed)
        } else {
            None
        };
        if let Some((log, _)) = &progress {
            log.info(&format!(
                "  microiterations: E* {energy:.10}, core gradient {:.3e} (largest) {:.3e} \
                 (RMS){}, environment residual {residual_max:.3e} (largest) {residual_rms:.3e} \
                 (RMS){}",
                crate::math::linalg::col_max_abs(&gradient),
                crate::math::linalg::col_norm(&gradient)
                    / ((gradient.nrows().max(1)) as f64).sqrt(),
                rigid_part.map_or_else(String::new, |r| format!(
                    " without its part along the rigid motions of the core ({r:.3e} largest)"
                )),
                match (&attempt, dense.as_ref()) {
                    (Some(Ok((dg, de))), Some(d)) => format!(
                        "; corrected for the residual by {de:+.3e} Hartree and {:.3e} in the \
                         core gradient (largest {core_before:.3e} before; response of {} \
                         coordinates)",
                        crate::math::linalg::col_max_abs(dg),
                        d.size()
                    ),
                    (Some(Err(reach)), _) => format!(
                        "; not corrected: the residual would move the environment by {reach:.2e} \
                         Bohr, beyond the {:.2} Bohr its quadratic model is trusted for",
                        hessian::CORRECTION_REACH
                    ),
                    _ if self.settings.correct => {
                        "; not corrected (no Hessian of the core with its response yet)".to_string()
                    }
                    _ => String::new(),
                }
            ));
        }
        if let Ok(mut state) = self.state.lock() {
            state.last_core = Some((crate::math::linalg::col_max_abs(&gradient), energy));
            state.last_state = Some((spin, req.excited_state));
        }
        let mut result = EvalOutput::energy_gradient(Quantity::new(energy), gradient);
        result.orbitals = out.orbitals;
        Ok((result, residual))
    }

    /// The start of a relaxation off the saddle point `saddle` of the environment `env` at the
    /// core `x`: `env` moved along the mode by the displacement of the mode kill of the
    /// environment ([`crate::pes::microiter::ModeKill::displacement`]), to the side of the
    /// lower energy of the relaxed surface (the gradient along the mode is near zero at a
    /// saddle point, so both sides are evaluated).
    fn off_saddle(
        &self,
        x: &Col<f64>,
        env: &[f64],
        saddle: &hessian::EnvironmentSaddle,
        spin: SpinCharge,
        excited_state: usize,
    ) -> MoptResult<Vec<f64>> {
        let length = self.settings.inner.mode_kill.displacement;
        let request = self.inner_request(excited_state);
        let mut best: Option<(f64, Vec<f64>)> = None;
        for sign in [1.0, -1.0] {
            let moved: Vec<f64> = env
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let along = if i < saddle.mode.nrows() {
                        saddle.mode[i]
                    } else {
                        0.0
                    };
                    v + sign * length * along
                })
                .collect();
            let mut full = assemble(&self.template, &self.split, x, &moved)?;
            full.spin_charge = spin;
            let energy = self.inner().evaluate(&full, &request)?.energy.value();
            if best.as_ref().is_none_or(|(e, _)| energy < *e) {
                best = Some((energy, moved));
            }
        }
        best.map(|(_, moved)| moved)
            .ok_or_else(|| MoptError::calc("ONIOM microiterations: no start off the saddle point"))
    }

    /// Keep `env` as the relaxed environment of the core `x` (to `tolerance`), the start of
    /// the relaxations near it and what the files of the run take for it.
    fn keep_relaxed(&self, x: &Col<f64>, env: &[f64], tolerance: f64) {
        if let Ok(mut state) = self.state.lock() {
            if state.kept.len() >= self.kept_capacity() {
                state.kept.pop_front();
            }
            state.kept.push_back(Kept {
                core: x.iter().copied().collect(),
                environment: env.to_vec(),
                relaxed: true,
                tolerance,
            });
            state.environment = env.to_vec();
        }
    }

    /// The masses of the coordinates of the core (amu) and, where rigid motions are symmetries
    /// of the relaxed surface, their directions at `x`: what the count of imaginary modes of
    /// the checks of stability takes (`opt::mode_kill`).
    fn core_masses_and_symmetries(&self, x: &Col<f64>) -> (Vec<f64>, Option<faer::Mat<f64>>) {
        let masses: Vec<f64> = self
            .split
            .core
            .iter()
            .flat_map(|&a| {
                let m = self.template.elements[a].mass_amu();
                [m, m, m]
            })
            .collect();
        let symmetries = if self.is_rigid_body_invariant() {
            MolecularSystem::from_flat_bohr(
                self.split
                    .core
                    .iter()
                    .map(|&a| self.template.elements[a])
                    .collect(),
                x,
                self.template.spin_charge,
                None,
            )
            .ok()
            .map(|core| crate::math::projection::build_tr_rot_basis(&core, false))
        } else {
            None
        };
        (masses, symmetries)
    }

    /// A Hessian for a check of stability with the error of its curvatures that the gradient
    /// left in the environment makes, reduced until the count of imaginary modes is resolved.
    ///
    /// The Hessian of the core is a function of the environment it is taken at, which a
    /// relaxation leaves anywhere within its tolerance of the minimum: the residual `g_y`
    /// moves the minimum by `-H_SS^-1 g_S` (the dense response of the Hessian just taken), and
    /// the curvatures of the core change with it. The environment is moved by that step and
    /// the Hessian taken again; the change of every mass-weighted eigenvalue is its error bar.
    /// Where the count of eigenvalues below minus the criterion of an imaginary mode
    /// ([`crate::opt::mode_kill::IMAGINARY_CURVATURE`]) is not the same at both ends of the
    /// error bars, the step is taken again from there (the residual falls by the square at
    /// every step), up to [`CHECK_ROUNDS`] times; the log says how it ended. The energy, the
    /// gradient, and the Hessian given back are those of the last environment, which is kept
    /// for the core (the files and the commands that follow take it).
    #[allow(clippy::too_many_arguments)]
    fn resolved_check(
        &self,
        x: &Col<f64>,
        mut env: Vec<f64>,
        mut result: EvalOutput,
        mut residual: Col<f64>,
        hessian: Option<faer::Mat<f64>>,
        spin: SpinCharge,
        req: &EvalRequest,
        model: Option<&replicas::ModelValues>,
    ) -> MoptResult<EvalOutput> {
        let Some(mut h) = hessian else {
            return Ok(result);
        };
        if self.replica_of.is_some() {
            if let Ok(mut state) = self.state.lock() {
                state.check_first = Some(h.clone());
            }
        }
        let (masses, symmetries) = self.core_masses_and_symmetries(x);
        let eigen = |h: &faer::Mat<f64>| {
            crate::opt::mode_kill::mass_weighted_eigenvalues(
                h,
                Some(masses.as_slice()),
                symmetries.as_ref(),
            )
        };
        let mut values = eigen(&h)?;
        let threshold = crate::opt::mode_kill::IMAGINARY_CURVATURE;
        let all = 3 * self.split.environment.len();
        let progress = self.state.lock().ok().and_then(|s| s.progress.clone());
        let mut resolved = false;
        let mut rounds = 0;
        while rounds < CHECK_ROUNDS {
            let dense = self.state.lock().ok().and_then(|s| s.dense.clone());
            let Some(step) = dense.and_then(|d| d.minimum_step(&residual, all)) else {
                if let Some((log, _)) = &progress {
                    log.info(
                        "  Hessian of the core for a check of stability: no step of the \
                         environment to its minimum within the reach of its quadratic model; the \
                         error of the curvatures is not estimated",
                    );
                }
                break;
            };
            rounds += 1;
            let moved: Vec<f64> = env
                .iter()
                .enumerate()
                .map(|(i, &v)| v + if i < step.nrows() { step[i] } else { 0.0 })
                .collect();
            let (next, next_residual) = self.surface_at(x, &moved, spin, req, model)?;
            let taken = self.core_hessian(x, &moved, spin, req.excited_state, true)?;
            let Some(next_h) = taken.hessian.filter(|_| taken.saddle.is_none()) else {
                break;
            };
            let next_values = eigen(&next_h)?;
            let errors: Vec<f64> = next_values
                .iter()
                .zip(&values)
                .map(|(a, b)| (a - b).abs())
                .collect();
            let count = |shift: f64| {
                next_values
                    .iter()
                    .zip(&errors)
                    .filter(|(v, e)| **v + shift * **e < -threshold)
                    .count()
            };
            resolved = count(-1.0) == count(1.0);
            if let Some((log, _)) = &progress {
                let lowest: Vec<String> = next_values
                    .iter()
                    .zip(&errors)
                    .take(6)
                    .map(|(v, e)| format!("{v:+.4e} +- {e:.1e}"))
                    .collect();
                log.info(&format!(
                    "  Hessian of the core for a check of stability: the environment moved to the \
                     minimum of its quadratic model (largest step {:.2e} Bohr) and the Hessian \
                     taken again (round {rounds} of {CHECK_ROUNDS}); lowest mass-weighted \
                     curvatures (Hartree/(Bohr^2 amu)) with the change as their error: {}; {} \
                     imaginary mode(s){}",
                    crate::math::linalg::col_max_abs(&step),
                    lowest.join(", "),
                    count(0.0),
                    if resolved {
                        ", the same count within the error"
                    } else {
                        ", the count changes within the error"
                    }
                ));
            }
            env = moved;
            result = next;
            residual = next_residual;
            h = next_h;
            values = next_values;
            self.keep_relaxed(x, &env, self.check_tolerance().0);
            if resolved {
                break;
            }
        }
        if !resolved && rounds > 0 {
            if let Some((log, _)) = &progress {
                log.warn(&format!(
                    "ONIOM microiterations: after {rounds} step(s) of the environment to its \
                     minimum the count of imaginary modes still changes within the error of the \
                     curvatures; the check judges the last Hessian"
                ));
            }
        }
        result.hessian_bohr = Some(h);
        Ok(result)
    }
}

/// Steps of the environment to the minimum of its quadratic model after a Hessian for a check
/// of stability, at most ([`RelaxedEnvironment::resolved_check`]).
const CHECK_ROUNDS: usize = 3;

/// The radii (Angstrom) of the shells whose numbers of free atoms of the environment the
/// description of the surface gives ([`RelaxedEnvironment::shell_counts`]).
const SHELL_RADII: [f64; 5] = [4.0, 5.0, 6.0, 8.0, 10.0];

impl Calculator for RelaxedEnvironment {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> String {
        let on = if self.settings.embedded {
            "the whole ONIOM surface (electronic embedding)".to_string()
        } else {
            format!("the lowest level ({})", self.low.name())
        };
        // What responds in the Hessian of the core, and how many free atoms of the
        // environment a shell of each radius would take (for the choice of
        // `micro_hessian_shell`).
        let shell = if self.split.environment.is_empty() {
            String::new()
        } else {
            let counts = self
                .shell_counts(&SHELL_RADII)
                .iter()
                .map(|(r, n)| format!("{r} A: {n}"))
                .collect::<Vec<String>>()
                .join(", ");
            let responding = match (self.settings.core_hessian, self.settings.hessian_shell) {
                (CoreHessian::Held, _) => {
                    "0 free environment atoms (micro_hessian=held)".to_string()
                }
                (_, Some(r)) => format!(
                    "the {} free environment atom(s) within {:.2} Angstrom of the core \
                     (micro_hessian_shell)",
                    hessian::responding_rows(&self.template, &self.split, Some(r)).len() / 3,
                    r * crate::domain::Units::BOHR_TO_ANGSTROM
                ),
                (_, None) => format!(
                    "all {} free environment atom(s)",
                    self.split.environment.len()
                ),
            };
            format!(
                "; the Hessian of the core takes the response of {responding} (free \
                 environment atoms within a distance of the core{}: {counts})",
                if self.template.active_cell().is_some() {
                    ", minimum image in the cell"
                } else {
                    ""
                }
            )
        };
        let replicas = if self.replicas.is_empty() {
            String::new()
        } else {
            let s = &self.settings.replicas;
            format!(
                "; {} replicas of the environment (start {}, seed {}) combined by a soft minimum \
                 at {} K",
                self.replicas.len(),
                s.start.name(),
                s.seed,
                s.temperature
            )
        };
        format!(
            "{} with microiterations: {} core atom(s) optimized, {} environment atom(s) \
             relaxed on {on}{shell}{replicas}",
            self.oniom.description(),
            self.split.core.len(),
            self.split.environment.len(),
        )
    }

    fn citations(&self) -> Vec<String> {
        let mut out = self.oniom.citations();
        out.push(
            "T. Vreven, K. Morokuma, Ö. Farkas, H. B. Schlegel, M. J. Frisch, J. Comput. Chem. \
             2003, 24, 760 (ONIOM microiterations)"
                .to_string(),
        );
        out
    }

    /// The relaxation of the environment starts from the environment of the nearest core
    /// evaluated before: which structure comes first decides where the next one starts.
    fn evaluates_in_order(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if req.need_stress {
            return Err(MoptError::unsupported(
                "ONIOM microiterations give the energy, the gradient, and the Hessian of a \
                 state; not a stress",
            ));
        }
        let x = sys.coords_flat_bohr();
        if x.nrows() != 3 * self.split.core.len() {
            return Err(MoptError::calc(format!(
                "ONIOM microiterations: a structure of {} atoms for a core of {}",
                x.nrows() / 3,
                self.split.core.len()
            )));
        }
        if !self.replicas.is_empty() {
            return self.evaluate_replicas(sys, req);
        }
        self.evaluate_one(sys, req, None, false)
    }

    /// With nothing holding the environment, `E*` does not change when the core moves
    /// rigidly: the environment relaxes to the moved core as a whole. Those motions are then
    /// zero modes of the effective Hessian, and an optimizer that does not project them out
    /// spends its steps on them, after which the environment has to follow by a slow
    /// collective motion. With fixed atoms in the environment `E*` is not invariant.
    fn is_rigid_body_invariant(&self) -> bool {
        self.split.fixed.is_empty()
            && self.oniom.is_rigid_body_invariant()
            && self.low.is_rigid_body_invariant()
    }

    /// Analytic where the Hessian of the core is the analytic Hessian of the whole system
    /// reduced to it (every level has one, and its matrices fit in the memory a Hessian may
    /// take).
    fn analytic_hessian(&self) -> bool {
        self.analytic_core_hessian()
    }

    fn supports_excited_states(&self) -> bool {
        self.oniom.supports_excited_states()
    }

    /// The environment relaxed at the core `sys` (one evaluated, and kept); with replicas,
    /// the environments of every replica one after the other.
    fn relaxed_environment(&self, sys: &MolecularSystem) -> Option<Vec<f64>> {
        if !self.replicas.is_empty() {
            return self.replica_environments(sys);
        }
        let x = sys.coords_flat_bohr();
        let state = self.state.lock().ok()?;
        state
            .kept
            .iter()
            .rev()
            .find(|k| k.relaxed && distance(&k.core, &x) <= SAME_STRUCTURE)
            .map(|k| k.environment.clone())
    }

    /// A restarted run replays the core at `sys`: its environment was `environment`, and the
    /// relaxations after the replay start from it as they would have in the first run.
    fn recall_environment(&self, sys: &MolecularSystem, environment: &[f64]) {
        if !self.replicas.is_empty() {
            self.recall_replica_environments(sys, environment);
            return;
        }
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        if environment.len() != 3 * self.split.environment.len()
            || x.len() != 3 * self.split.core.len()
        {
            return;
        }
        let capacity = self.kept_capacity();
        if let Ok(mut state) = self.state.lock() {
            state.environment = environment.to_vec();
            if state.kept.len() >= capacity {
                state.kept.pop_front();
            }
            // The journal does not say the tolerance: that of the relaxations of the run (a
            // check of stability relaxes it again to its own).
            state.kept.push_back(Kept {
                core: x,
                environment: environment.to_vec(),
                relaxed: true,
                tolerance: self.settings.inner.gradient_tolerance,
            });
            state.counts.recalled += 1;
        }
    }

    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        if !self.replicas.is_empty() {
            let full = self.full_system_of_replicas(sys)?;
            return self.oniom.excitation_energies(&full);
        }
        let x = sys.coords_flat_bohr();
        let env = self.relax_at(&x, false, sys.spin_charge, 0, Relax::default())?;
        let mut full = assemble(&self.template, &self.split, &x, &env)?;
        full.spin_charge = sys.spin_charge;
        self.oniom.excitation_energies(&full)
    }

    fn take_warnings(&self) -> Vec<String> {
        let mut out = self.oniom.take_warnings();
        out.extend(self.low.take_warnings());
        out
    }
}

impl RelaxedEnvironment {
    /// The surface of one environment at the core `sys` (the surface without replicas, and
    /// each replica of one with them). `model`: the terms of the model systems computed at this
    /// core already (replicas with mechanical embedding). `check`: the environment relaxed to
    /// the tolerance of the checks of stability even without a Hessian (the replicas of a
    /// check, whose energies decide their weights).
    fn evaluate_one(
        &self,
        sys: &MolecularSystem,
        req: &EvalRequest,
        model: Option<&replicas::ModelValues>,
        check: bool,
    ) -> MoptResult<EvalOutput> {
        let x = sys.coords_flat_bohr();
        // The state of the request: an excited state is that of the highest level (-es), and
        // the charge and multiplicity those of the structure (the states of -mf).
        let spin = sys.spin_charge;
        // A Hessian for a check of stability is taken with the environment relaxed to the
        // tolerance of the checks, whatever that of the run (`check_tolerance`).
        let tolerance =
            (check || (req.need_hessian && req.for_stability)).then(|| self.check_tolerance());
        let how = Relax {
            tolerance,
            from: None,
        };
        let mut env = self.relax_at(&x, false, spin, req.excited_state, how)?;
        let mut escapes = 0;
        loop {
            let (mut result, residual) = self.surface_at(&x, &env, spin, req, model)?;
            if !req.need_hessian {
                return Ok(result);
            }
            // For a check of stability the solves run to convergence; where they do not,
            // no Hessian is given and the check is not judged on one (the log says so).
            let taken = self.core_hessian(&x, &env, spin, req.excited_state, req.for_stability)?;
            match taken.saddle {
                // The environment stands on a saddle point of its own: its response is not
                // that of a minimum. It is moved off along the mode of that curvature,
                // downhill, and relaxed again; the energy, the gradient, and the Hessian are
                // those of the minimum it reaches.
                Some(saddle) if escapes < ENVIRONMENT_ESCAPES => {
                    escapes += 1;
                    let start = self.off_saddle(&x, &env, &saddle, spin, req.excited_state)?;
                    let progress = self.state.lock().ok().and_then(|mut s| {
                        s.counts.environment_saddles += 1;
                        s.progress.clone()
                    });
                    if let Some((log, _)) = &progress {
                        log.info(&format!(
                            "  microiterations: the environment of this Hessian stands on a \
                             saddle point of its own (curvature of its block {:+.3e} \
                             Hartree/Bohr^2); moved off along that mode by {:.2} Bohr, downhill, \
                             and relaxed again (attempt {escapes} of {ENVIRONMENT_ESCAPES})",
                            saddle.curvature, self.settings.inner.mode_kill.displacement
                        ));
                    }
                    let how = Relax {
                        tolerance,
                        from: Some(&start),
                    };
                    env = self.relax_at(&x, false, spin, req.excited_state, how)?;
                }
                Some(saddle) => {
                    let progress = self.state.lock().ok().and_then(|mut s| {
                        s.counts.environment_saddles_kept += 1;
                        s.progress.clone()
                    });
                    if let Some((log, _)) = &progress {
                        log.warn(&format!(
                            "ONIOM microiterations: the environment still stands on a saddle \
                             point of its own after {ENVIRONMENT_ESCAPES} attempt(s) to leave it \
                             (curvature of its block {:+.3e} Hartree/Bohr^2); {}",
                            saddle.curvature,
                            if req.for_stability {
                                "the stability is not judged on this Hessian"
                            } else {
                                "the Hessian takes its response over the positive curvatures"
                            }
                        ));
                    }
                    result.hessian_bohr = if req.for_stability {
                        None
                    } else {
                        taken.hessian
                    };
                    return Ok(result);
                }
                None if req.for_stability && taken.hessian.is_some() => {
                    return self.resolved_check(
                        &x,
                        env,
                        result,
                        residual,
                        taken.hessian,
                        spin,
                        req,
                        model,
                    );
                }
                None => {
                    result.hessian_bohr = taken.hessian;
                    return Ok(result);
                }
            }
        }
    }
}

/// A shared [`RelaxedEnvironment`] as a calculator, so that a run can keep it to write the
/// full structures after the optimizer is done with it.
impl Calculator for std::sync::Arc<RelaxedEnvironment> {
    fn name(&self) -> &str {
        self.as_ref().name()
    }

    fn description(&self) -> String {
        self.as_ref().description()
    }

    fn citations(&self) -> Vec<String> {
        self.as_ref().citations()
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        self.as_ref().evaluate(sys, req)
    }

    fn is_rigid_body_invariant(&self) -> bool {
        self.as_ref().is_rigid_body_invariant()
    }

    fn analytic_hessian(&self) -> bool {
        self.as_ref().analytic_hessian()
    }

    fn supports_excited_states(&self) -> bool {
        self.as_ref().supports_excited_states()
    }

    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        self.as_ref().excitation_energies(sys)
    }

    fn take_warnings(&self) -> Vec<String> {
        self.as_ref().take_warnings()
    }

    // The relaxations go from structure to structure, and the journal of a run records and
    // gives back the environments they reach: without these, a run that shares the surface
    // would evaluate in parallel, and a restart would relax its environments anew.
    fn evaluates_in_order(&self) -> bool {
        self.as_ref().evaluates_in_order()
    }

    fn relaxed_environment(&self, sys: &MolecularSystem) -> Option<Vec<f64>> {
        self.as_ref().relaxed_environment(sys)
    }

    fn recall_environment(&self, sys: &MolecularSystem, environment: &[f64]) {
        self.as_ref().recall_environment(sys, environment);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod replica_tests;
