//! Replicas of the environment: several minima of the environment at the same core, combined
//! into one surface of the core by a soft minimum (`-oniom_opt micro_replicas=K`; new in
//! mopt-rs, its own formulation).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! # Why
//!
//! With microiterations the optimizer moves on `E*(x) = min_y E(x, y)`. An environment of
//! hundreds of atoms has many minima near each other: at one structure of chorismate mutase
//! relaxations from perturbed starts ended in minima 5e-7 and 1.4e-4 Hartree apart, and the
//! structure was a first-order saddle point of `E*` in one of them and a second-order one in
//! the next (docs/source/oniom.md). Which minimum a relaxation finds then decides what the
//! stationary point is, and `E*` is only piecewise smooth: where two minima cross in energy
//! the lower one changes, and the gradient of `E*` jumps.
//!
//! # The surface
//!
//! `K` replicas `y_1 .. y_K` of the environment are kept, each relaxed at every core `x` from
//! its own environment of the core before (the store of the kept cores holds one environment
//! per replica): `E_k(x) = min over y near y_k of E(x, y)`. The surface of the core is their
//! soft minimum at the temperature `T` (`beta = 1 / (k_B T)`),
//!
//! ```text
//! E_soft(x) = -(1/beta) ln sum_k exp(-beta E_k(x))
//!           = E_min - (1/beta) ln sum_k exp(-beta (E_k - E_min)),
//! ```
//!
//! computed in the second form (log-sum-exp, [`crate::math::stable`]), with the weights
//! `w_k = exp(-beta (E_k - E_min)) / sum_l exp(-beta (E_l - E_min))`. Each replica is a minimum
//! in its own `y`, so `dE_k/dx` is the core gradient `g_k` at its relaxed environment (the
//! environment gradient vanishes there; the effective Hessian `H_k = H_xx - H_xy H_yy^-1 H_yx`
//! is the derivative of `g_k`), and
//!
//! ```text
//! g_soft = sum_k w_k g_k
//! H_soft = sum_k w_k H_k - beta sum_k w_k (g_k - g_soft)(g_k - g_soft)^T
//! ```
//!
//! (`dw_k/dx = -beta w_k (g_k - g_soft)`). The second term is written as the weighted
//! covariance of the replica gradients, which equals `sum_k w_k g_k g_k^T - g_soft g_soft^T` and
//! cannot lose its positive semidefiniteness to cancellation. Limits: one replica gives the
//! surface of one environment exactly (`K = 1` is the surface without replicas, the same code);
//! `T -> 0` gives the lowest replica, `E_min`; `E_min - k_B T ln K <= E_soft <= E_min`.
//! Where two replicas cross in energy the soft minimum is smooth: on a scale of `k_B T` the
//! weights pass from one to the other, and the covariance term is the curvature of that
//! passage (`-beta (g_1 - g_2)^2 / 4` at the crossing of two), so that the kink of `min_k E_k`
//! becomes a region of negative curvature of width `~ k_B T / |g_1 - g_2|`.
//!
//! What it is not: the replicas are `K` minima, not a thermal sample of the environment, and
//! `E_soft` is not a free energy of the environment (no entropy of the minima, no vibrations of
//! the environment). Energies and barriers of `E_soft` compare only between points evaluated
//! with the same replicas at the same temperature.
//!
//! # The replicas over a run
//!
//! - **Start** ([`ReplicaStart`]): replica 1 relaxes from the environment of the run (that of
//!   the input or of the nearest kept core); replica `k > 1` from that relaxed environment with
//!   every free atom moved by `micro_replica_amplitude` in a random direction (`perturb`, fixed
//!   seed), or after a short molecular dynamics of the environment at `T` with the core held
//!   (`md`, the dynamics of the crate handed in by the command), or from the environments of
//!   whole structures in a file (`files`, as written by a run with replicas).
//! - **Identity**: each replica relaxes from its own environment of the nearest kept core, as
//!   the single environment does. Structures that the run remembers (the input, the frames a
//!   chain is made from) are the start of replica 1 only; the others follow their own
//!   environments.
//! - **Two replicas in one minimum** ([`same_minimum`]): at the start a replica that relaxed
//!   into the minimum of another is started again from a larger perturbation (up to
//!   [`RESEEDS`] times; `perturb` and `md`), so that the run starts with distinct minima where
//!   they are found. During the run replicas that fall into the same minimum are reported and
//!   kept with their weights: removing one, or starting it anew, would change the surface
//!   between two steps of the optimizer by up to `k_B T ln 2` on top of the jump of the replica
//!   itself.
//! - **A replica on a saddle point of the environment** is moved off it as a single environment
//!   is (per replica, in its Hessians).
//! - **A replica far above the others** keeps relaxing (so that it keeps its identity), but its
//!   Hessian is not taken where its weight is below [`NEGLIGIBLE_WEIGHT`].
//! - **Cost**: every replica relaxes at every core. With mechanical embedding the model systems
//!   do not see the environment, so the higher levels are computed once per core (their energy,
//!   gradient, and Hessian) and only the lowest level per replica; with electronic embedding
//!   every replica has its own higher level, `K` times the cost of one environment. The replicas
//!   relax in parallel where their evaluations share no calculator with electrons (the order of
//!   the SCF guesses would otherwise depend on the threads), and one after the other in their
//!   order otherwise; the results are combined in a fixed order.
//!
//! The aim is that of the multistructural microiteration of K. Suzuki, K. Morokuma, S. Maeda
//! (J. Comput. Chem. 2017, 38, 2213) and K. Suzuki, S. Maeda (Phys. Chem. Chem. Phys. 2022, 24,
//! 16762): a surface of the core that accounts for several structures of the environment. This
//! module is not that method; its formulation is the one above.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::{Col, Mat};
use rayon::prelude::*;

use super::{assemble, distance, gather, Relax, RelaxedEnvironment, RigidMotion};
use crate::core_infra::rng::Rng;
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity, SpinCharge, Units};
use crate::error::{MoptError, MoptResult};
use crate::pes::calculator::Calculator;
use crate::pes::microiter::MicroProblem;

/// The temperature of the soft minimum without `micro_replica_temperature` (K): room
/// temperature, at which `k_B T` = 9.4e-4 Hartree. The minima of the environment of chorismate
/// mutase that changed the character of its transition state were 5e-7 Hartree apart (and
/// another one 1.4e-4), so at this temperature they weigh alike (weights 0.5 and 0.46 of two),
/// which is the case the replicas are for; a temperature far below their differences gives the
/// lowest replica alone, the single environment again (docs/source/oniom.md).
pub const REPLICA_TEMPERATURE: f64 = 298.15;

/// The displacement of every free atom of the environment that starts a replica
/// (`micro_replica_amplitude`, Angstrom): that of the measurements of the minima of the
/// environment of chorismate mutase, where 2 of 5 such starts relaxed into other minima.
pub const REPLICA_AMPLITUDE_ANGSTROM: f64 = 0.05;

/// The steps of the molecular dynamics of the environment that starts a replica
/// (`micro_replica_start=md`, `micro_replica_md_steps`): 100 fs at the time step of 0.5 fs.
pub const REPLICA_MD_STEPS: usize = 200;

/// The seed of the starts of the replicas (`micro_replica_seed`).
pub const REPLICA_SEED: u64 = 1;

/// A replica whose weight is below this has no Hessian taken: its share of the Hessian of the
/// soft minimum is below this times its own (`ln 1e12 = 27.6`, an energy 27.6 `k_B T` above the
/// lowest).
pub const NEGLIGIBLE_WEIGHT: f64 = 1e-12;

/// Two replicas are in the same minimum of the environment when no free atom differs by more
/// than this (Angstrom) and their energies by less than [`SAME_MINIMUM_HARTREE`].
pub const SAME_MINIMUM_ANGSTROM: f64 = 0.1;

/// See [`SAME_MINIMUM_ANGSTROM`] (Hartree).
pub const SAME_MINIMUM_HARTREE: f64 = 1e-5;

/// How often a replica that started in the minimum of another is started again at the start of
/// a run, each time from twice the displacement.
pub const RESEEDS: usize = 3;

/// The cores whose replica energies and weights are kept for the files of the run.
const SOFT_POINTS_KEPT: usize = 4096;

/// Where the replicas other than the first start.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ReplicaStart {
    /// From the relaxed environment of the first replica, every free atom moved by the
    /// amplitude in a random direction (`perturb`, the default).
    #[default]
    Perturb,
    /// From the relaxed environment of the first replica after a short molecular dynamics of
    /// the environment at the temperature of the replicas, with the core held (`md`).
    Dynamics,
    /// From the environments of the whole structures of a file, one per replica (the last
    /// `K` frames; `files`).
    Files(std::path::PathBuf),
    /// From the environments of the given whole structures (flat coordinates of the whole
    /// system, Bohr; one per replica), as `files` reads them.
    Structures(Vec<Vec<f64>>),
}

impl ReplicaStart {
    /// The name of the start, for the log and the description.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Perturb => "perturb",
            Self::Dynamics => "md",
            Self::Files(_) | Self::Structures(_) => "files",
        }
    }
}

/// What the molecular dynamics that starts a replica is asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicsRequest {
    /// Temperature (K).
    pub temperature: f64,
    /// Steps.
    pub steps: usize,
    /// Seed of its velocities and thermostat.
    pub seed: u64,
    /// Masses of the free atoms of the environment (amu), one per atom.
    pub masses: Vec<f64>,
}

/// A molecular dynamics of the environment (the coordinates of `problem`) from `start`,
/// giving the coordinates at its end. The dynamics are those of the crate, a layer above the
/// surfaces, so this layer takes the function from the code that builds the settings
/// ([`ReplicaSettings::dynamics`], `cli::oniom_micro::environment_dynamics`).
pub type EnvironmentDynamicsFn =
    fn(&dyn MicroProblem, &[f64], &DynamicsRequest) -> MoptResult<Vec<f64>>;

/// The dynamics of the environment, handed in with the settings.
#[derive(Clone, Copy)]
pub struct EnvironmentDynamics(pub EnvironmentDynamicsFn);

impl std::fmt::Debug for EnvironmentDynamics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EnvironmentDynamics")
    }
}

impl PartialEq for EnvironmentDynamics {
    /// The dynamics are one service of the code, not a setting.
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// Settings of the replicas of the environment.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplicaSettings {
    /// The number of replicas (`micro_replicas`; 1: one environment, no soft minimum).
    pub count: usize,
    /// The temperature of the soft minimum (K, `micro_replica_temperature`).
    pub temperature: f64,
    /// Where the replicas start (`micro_replica_start`).
    pub start: ReplicaStart,
    /// The displacement of the free atoms of a perturbed start (Bohr; given in Angstrom,
    /// `micro_replica_amplitude`).
    pub amplitude: f64,
    /// The seed of the perturbations and dynamics (`micro_replica_seed`).
    pub seed: u64,
    /// The steps of the dynamics of a start `md` (`micro_replica_md_steps`).
    pub dynamics_steps: usize,
    /// The dynamics of a start `md` (set by the command).
    pub dynamics: Option<EnvironmentDynamics>,
}

impl Default for ReplicaSettings {
    fn default() -> Self {
        Self {
            count: 1,
            temperature: REPLICA_TEMPERATURE,
            start: ReplicaStart::Perturb,
            amplitude: REPLICA_AMPLITUDE_ANGSTROM * Units::ANGSTROM_TO_BOHR,
            seed: REPLICA_SEED,
            dynamics_steps: REPLICA_MD_STEPS,
            dynamics: None,
        }
    }
}

/// The soft minimum of the energies of the replicas.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftMinimum {
    /// `E_soft` (Hartree).
    pub energy: f64,
    /// The weights `w_k` (they sum to one).
    pub weights: Vec<f64>,
    /// `beta = 1 / (k_B T)` (1/Hartree).
    pub beta: f64,
    /// The replica of the lowest energy (the first of equal ones).
    pub lowest: usize,
}

/// `E_soft = E_min - (1/beta) ln sum_k exp(-beta (E_k - E_min))` and the weights of `energies`
/// at `temperature` (K, above zero). The exponents are shifted by the lowest energy, so that
/// they are at most zero and the sum lies between 1 and `K`.
pub fn soft_minimum(energies: &[f64], temperature: f64) -> MoptResult<SoftMinimum> {
    if energies.is_empty() || energies.iter().any(|e| !e.is_finite()) {
        return Err(MoptError::calc(format!(
            "ONIOM replicas: the soft minimum needs finite energies, got {energies:?}"
        )));
    }
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(MoptError::config(format!(
            "-oniom_opt micro_replica_temperature={temperature}: a temperature above 0 K"
        )));
    }
    let beta = 1.0 / (Units::BOLTZMANN_HARTREE_PER_K * temperature);
    let (lowest, e_min) =
        energies
            .iter()
            .copied()
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |(k, m), (i, e)| {
                    if e < m {
                        (i, e)
                    } else {
                        (k, m)
                    }
                },
            );
    let exponents: Vec<f64> = energies.iter().map(|e| -beta * (e - e_min)).collect();
    let energy = e_min - crate::math::stable::log_sum_exp(&exponents) / beta;
    Ok(SoftMinimum {
        energy,
        weights: crate::math::stable::normalized_exp(&exponents),
        beta,
        lowest,
    })
}

/// `g_soft = sum_k w_k g_k`, summed in the order of the replicas.
pub fn soft_gradient(gradients: &[&Col<f64>], weights: &[f64]) -> Col<f64> {
    let n = gradients.first().map_or(0, |g| g.nrows());
    let mut out = Col::<f64>::zeros(n);
    for (g, &w) in gradients.iter().zip(weights) {
        for i in 0..n {
            out[i] += w * g[i];
        }
    }
    out
}

/// `H_soft = sum_k w_k H_k - beta sum_k w_k (g_k - g_soft)(g_k - g_soft)^T`. A replica without
/// a Hessian (`None`) must have a weight below [`NEGLIGIBLE_WEIGHT`]; its share is left out.
pub fn soft_hessian(
    hessians: &[Option<&Mat<f64>>],
    gradients: &[&Col<f64>],
    soft: &SoftMinimum,
) -> MoptResult<Mat<f64>> {
    let n = gradients.first().map_or(0, |g| g.nrows());
    let g = soft_gradient(gradients, &soft.weights);
    let mut out = Mat::<f64>::zeros(n, n);
    for (k, (h, &w)) in hessians.iter().zip(&soft.weights).enumerate() {
        match h {
            Some(h) => out += *h * faer::Scale(w),
            None if w < NEGLIGIBLE_WEIGHT => {}
            None => {
                return Err(MoptError::calc(format!(
                    "ONIOM replicas: replica {} has a weight of {w:.3e} and no Hessian",
                    k + 1
                )))
            }
        }
    }
    for (gk, &w) in gradients.iter().zip(&soft.weights) {
        let d = Col::from_fn(n, |i| gk[i] - g[i]);
        let s = soft.beta * w;
        for j in 0..n {
            for i in 0..n {
                out[(i, j)] -= s * d[i] * d[j];
            }
        }
    }
    Ok(out)
}

/// Whether the environments `a` and `b` (flat, Bohr) with the energies `ea` and `eb` are the
/// same minimum: no atom apart by more than [`SAME_MINIMUM_ANGSTROM`] and the energies within
/// [`SAME_MINIMUM_HARTREE`].
pub fn same_minimum(a: &[f64], b: &[f64], ea: f64, eb: f64) -> bool {
    (ea - eb).abs() < SAME_MINIMUM_HARTREE
        && largest_displacement(a, b) < SAME_MINIMUM_ANGSTROM * Units::ANGSTROM_TO_BOHR
}

/// The largest distance of an atom of `a` from the same atom of `b` (flat, Bohr).
pub fn largest_displacement(a: &[f64], b: &[f64]) -> f64 {
    a.chunks(3)
        .zip(b.chunks(3))
        .map(|(p, q)| {
            p.iter()
                .zip(q)
                .map(|(u, v)| (u - v) * (u - v))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0_f64, f64::max)
}

/// The seed of the start of replica `replica` (0-based) at the attempt `attempt`: a fixed
/// function of the seed of the run, the same whatever the threads.
fn start_seed(seed: u64, replica: usize, attempt: usize) -> u64 {
    let k = (replica as u64) * 64 + attempt as u64 + 1;
    seed ^ 0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(k)
}

/// `env` (flat, Bohr) with every atom moved by `amplitude` (Bohr) in a random direction (the
/// seed of replica `replica` at the attempt `attempt`).
pub fn perturbed(
    env: &[f64],
    amplitude: f64,
    seed: u64,
    replica: usize,
    attempt: usize,
) -> Vec<f64> {
    let mut rng = Rng::from_seed(start_seed(seed, replica, attempt));
    let mut out = env.to_vec();
    for atom in out.chunks_mut(3) {
        let d = rng.unit_vector_3d();
        for (c, v) in atom.iter_mut().enumerate() {
            *v += amplitude * d[c];
        }
    }
    out
}

/// What the soft minimum gave at a core: the files of the run write the lowest replica with
/// its weight.
#[derive(Debug, Clone)]
pub(super) struct SoftPoint {
    core: Vec<f64>,
    energies: Vec<f64>,
    weights: Vec<f64>,
}

/// The record of the replicas of a surface with more than one.
#[derive(Debug, Default)]
pub(super) struct ReplicaRecord {
    /// Whether the replicas have been started (or given back by the journal of a restart).
    generated: bool,
    /// The cores evaluated with their replica energies and weights (the newest last).
    points: VecDeque<SoftPoint>,
    /// The environments (flat, Bohr) and cores of the structures of `files`, one per replica.
    frames: Vec<(Vec<f64>, Vec<f64>)>,
    /// Pairs of replicas in the same minimum at the last core (reported when they change).
    same: Vec<(usize, usize)>,
    /// Replicas started again because they began in the minimum of another.
    pub(super) reseeds: usize,
    /// Replicas that stayed in the minimum of another after the restarts.
    pub(super) duplicates: usize,
    /// Hessians of replicas not taken because their weight was negligible.
    pub(super) skipped_hessians: usize,
    /// Evaluations of the soft minimum.
    pub(super) evaluations: usize,
}

/// The part of the surface that the environment does not change (mechanical embedding): the
/// energy of the terms of the model systems, their gradient on the real system, and the output
/// of the highest level.
pub(super) struct ModelValues {
    pub(super) energy: f64,
    pub(super) gradient: Col<f64>,
    pub(super) high: Option<EvalOutput>,
}

/// A Hessian of the terms of the model systems, kept for the replicas at the same core.
pub(super) struct ModelHessian {
    core: Vec<f64>,
    excited_state: usize,
    spin: SpinCharge,
    atoms: usize,
    hessian: Mat<f64>,
}

/// The Hessians of the model systems shared by the replicas of one surface (the last two: on
/// the core and on the whole system).
pub(super) type ModelHessians = std::sync::Arc<std::sync::Mutex<VecDeque<ModelHessian>>>;

impl RelaxedEnvironment {
    /// The number of replicas of the environment (1 without them).
    pub fn replica_count(&self) -> usize {
        self.replicas.len().max(1)
    }

    /// The temperature of the soft minimum of the replicas (K), `None` with one environment.
    pub fn replica_temperature(&self) -> Option<f64> {
        (!self.replicas.is_empty()).then_some(self.settings.replicas.temperature)
    }

    /// Whether the replicas relax in parallel: their evaluations share no calculator with
    /// electrons (mechanical embedding, a lowest level without electrons, and the higher levels
    /// computed once per core), whose SCF guesses would follow the order of the threads.
    fn replicas_in_parallel(&self) -> bool {
        self.model_part.is_some()
            && !self.settings.embedded
            && !self.low.uses_electrons()
            && (self.routes.whole || self.routes.model.is_some())
    }

    /// `f` for the replicas `which`, in parallel where [`Self::replicas_in_parallel`], in their
    /// order otherwise; the results in the order of `which`.
    fn each_of<T: Send>(
        &self,
        which: &[usize],
        f: impl Fn(usize, &RelaxedEnvironment) -> MoptResult<T> + Sync + Send,
    ) -> MoptResult<Vec<T>> {
        if self.replicas_in_parallel() {
            which
                .par_iter()
                .map(|&r| f(r, &self.replicas[r]))
                .collect::<Vec<MoptResult<T>>>()
                .into_iter()
                .collect()
        } else {
            which.iter().map(|&r| f(r, &self.replicas[r])).collect()
        }
    }

    /// [`Self::each_of`] for every replica.
    fn each<T: Send>(
        &self,
        f: impl Fn(usize, &RelaxedEnvironment) -> MoptResult<T> + Sync + Send,
    ) -> MoptResult<Vec<T>> {
        let all: Vec<usize> = (0..self.replicas.len()).collect();
        self.each_of(&all, f)
    }

    /// The part of the surface the environment does not change at the core `x` (mechanical
    /// embedding with the terms of the model systems apart), `None` otherwise.
    fn model_values(
        &self,
        x: &Col<f64>,
        spin: SpinCharge,
        excited_state: usize,
    ) -> MoptResult<Option<ModelValues>> {
        let Some(model) = self.model_part.as_ref().filter(|_| !self.settings.embedded) else {
            return Ok(None);
        };
        let env: Vec<f64> = gather(&self.template.coords_flat_bohr(), &self.split.environment)
            .iter()
            .copied()
            .collect();
        let mut full = assemble(&self.template, &self.split, x, &env)?;
        full.spin_charge = spin;
        let (energy, gradient, high) = model.energy_gradient(&full, excited_state)?;
        Ok(Some(ModelValues {
            energy,
            gradient,
            high,
        }))
    }

    /// The Hessian of the terms of the model systems at `full` on the atoms `atoms` (the
    /// split and the analytic routes of the Hessian of the core): computed once per core for
    /// the replicas of a surface, as it is for one environment otherwise.
    pub(super) fn model_hessian_on(
        &self,
        model: &crate::pes::oniom::ModelPart,
        full: &MolecularSystem,
        atoms: &[usize],
        excited_state: usize,
    ) -> MoptResult<Option<Mat<f64>>> {
        let Some(cache) = self.model_hessians.as_ref() else {
            return model.hessian_on(full, atoms, excited_state);
        };
        let core: Vec<f64> = gather(&full.coords_flat_bohr(), &self.split.core)
            .iter()
            .copied()
            .collect();
        let hit = |c: &ModelHessian| {
            c.atoms == atoms.len()
                && c.excited_state == excited_state
                && c.spin == full.spin_charge
                && c.core == core
        };
        if let Ok(kept) = cache.lock() {
            if let Some(c) = kept.iter().find(|c| hit(c)) {
                return Ok(Some(c.hessian.clone()));
            }
        }
        // Computed outside the lock; two replicas that compute it at once get the same matrix.
        let Some(hessian) = model.hessian_on(full, atoms, excited_state)? else {
            return Ok(None);
        };
        if let Some(c) = &self.counters {
            c.record_hessian();
        }
        if let Ok(mut kept) = cache.lock() {
            if !kept.iter().any(hit) {
                if kept.len() >= 2 {
                    kept.pop_front();
                }
                kept.push_back(ModelHessian {
                    core,
                    excited_state,
                    spin: full.spin_charge,
                    atoms: atoms.len(),
                    hessian: hessian.clone(),
                });
            }
        }
        Ok(Some(hessian))
    }

    /// The energy of the surface of the relaxations (the lowest level, or the whole surface
    /// with electronic embedding) at the core `x` with the environment `env`.
    fn inner_energy(
        &self,
        x: &Col<f64>,
        env: &[f64],
        spin: SpinCharge,
        excited_state: usize,
    ) -> MoptResult<f64> {
        let mut full = assemble(&self.template, &self.split, x, env)?;
        full.spin_charge = spin;
        Ok(self
            .inner()
            .evaluate(&full, &self.inner_request(excited_state))?
            .energy
            .value())
    }

    /// Read the structures of the start `files` (or take the given ones): one environment per
    /// replica, from the last `K` frames.
    pub(super) fn read_replica_frames(&self) -> MoptResult<Vec<(Vec<f64>, Vec<f64>)>> {
        let k = self.replicas.len();
        let wholes: Vec<Vec<f64>> = match &self.settings.replicas.start {
            ReplicaStart::Files(path) => {
                let frames = crate::io::xyz::read_xyz_trajectory_file(
                    path,
                    self.template.spin_charge,
                    self.template.cell.as_ref(),
                )
                .map_err(|e| {
                    e.context(format!("-oniom_opt micro_replica_files={}", path.display()))
                })?;
                frames
                    .frames
                    .iter()
                    .map(|f| f.system.coords_flat_bohr().iter().copied().collect())
                    .collect()
            }
            ReplicaStart::Structures(given) => given.clone(),
            ReplicaStart::Perturb | ReplicaStart::Dynamics => return Ok(Vec::new()),
        };
        if wholes.len() < k {
            return Err(MoptError::config(format!(
                "-oniom_opt micro_replica_start=files: {} structure(s) for {k} replicas \
                 (micro_replicas); the file needs one whole structure per replica",
                wholes.len()
            )));
        }
        let dof = self.template.dof();
        wholes[wholes.len() - k..]
            .iter()
            .map(|w| {
                if w.len() != dof {
                    return Err(MoptError::config(format!(
                        "-oniom_opt micro_replica_files: a structure of {} atoms for a system of \
                         {}",
                        w.len() / 3,
                        dof / 3
                    )));
                }
                let flat = Col::from_fn(dof, |i| w[i]);
                Ok((
                    gather(&flat, &self.split.core).iter().copied().collect(),
                    gather(&flat, &self.split.environment)
                        .iter()
                        .copied()
                        .collect(),
                ))
            })
            .collect()
    }

    /// Keep the structures of the start `files` for the first evaluation.
    pub(super) fn keep_replica_frames(&self, frames: Vec<(Vec<f64>, Vec<f64>)>) {
        if let Ok(mut record) = self.replica_record.lock() {
            record.frames = frames;
        }
    }

    /// The start of replica `replica` from the relaxed environment `env0` of the first one at
    /// the core `x`, at the attempt `attempt` (0: the first; each further one from twice the
    /// displacement of the one before).
    fn replica_start(
        &self,
        replica: usize,
        x: &Col<f64>,
        env0: &[f64],
        spin: SpinCharge,
        excited_state: usize,
        attempt: usize,
    ) -> MoptResult<Vec<f64>> {
        let s = &self.settings.replicas;
        let scale = f64::from(1u32 << attempt.min(16));
        match &s.start {
            ReplicaStart::Dynamics => {
                let dynamics = s.dynamics.ok_or_else(|| {
                    MoptError::config(
                        "-oniom_opt micro_replica_start=md: the command gives no molecular \
                         dynamics of the environment here",
                    )
                })?;
                let problem = super::LowEnvironment {
                    low: self.inner(),
                    template: &self.template,
                    split: &self.split,
                    core: x,
                    spin,
                    request: self.inner_request(excited_state),
                    report: None,
                    tally: super::Tally::default(),
                    preconditioner: None,
                    guard: None,
                };
                let request = DynamicsRequest {
                    temperature: s.temperature * scale,
                    steps: s.dynamics_steps,
                    seed: start_seed(s.seed, replica, attempt),
                    masses: self
                        .split
                        .environment
                        .iter()
                        .map(|&a| self.template.elements[a].mass_amu())
                        .collect(),
                };
                (dynamics.0)(&problem, env0, &request)
            }
            _ => Ok(perturbed(
                env0,
                s.amplitude * scale,
                s.seed,
                replica,
                attempt,
            )),
        }
    }

    /// The start of the replicas at the first evaluation (core `x`; `tolerance` that of the
    /// evaluation): every replica relaxed from its start, and a replica that relaxed into the
    /// minimum of another started again (not those of `files`). Nothing once started.
    fn start_replicas(
        &self,
        x: &Col<f64>,
        spin: SpinCharge,
        excited_state: usize,
        tolerance: Option<(f64, f64)>,
    ) -> MoptResult<()> {
        let frames = {
            let record = self
                .replica_record
                .lock()
                .map_err(|_| MoptError::calc("ONIOM replicas: the record is poisoned"))?;
            if record.generated {
                return Ok(());
            }
            record.frames.clone()
        };
        let k = self.replicas.len();
        let rigid = self.carries_environment();
        let mut notes = Vec::new();
        // The starts: the frames of the files (moved onto this core where rigid motions are
        // symmetries), or the first replica relaxed and the others from it.
        let (starts, env0): (Vec<Option<Vec<f64>>>, Option<Vec<f64>>) = if frames.is_empty() {
            let how = Relax {
                tolerance,
                from: None,
            };
            let env0 = self.replicas[0].relax_at(x, false, spin, excited_state, how)?;
            let starts = (0..k)
                .map(|r| {
                    if r == 0 {
                        Ok(None)
                    } else {
                        self.replica_start(r, x, &env0, spin, excited_state, 0)
                            .map(Some)
                    }
                })
                .collect::<MoptResult<Vec<_>>>()?;
            (starts, Some(env0))
        } else {
            let mut worst = 0.0_f64;
            let starts = frames
                .iter()
                .map(|(core, env)| {
                    let (core, env) = if rigid {
                        let motion = RigidMotion::fit(core, x);
                        (motion.apply(core), motion.apply(env))
                    } else {
                        (core.clone(), env.clone())
                    };
                    worst = worst.max(distance(&core, x));
                    Some(env)
                })
                .collect();
            if worst > super::SAME_IN_FILES {
                notes.push(format!(
                    "ONIOM replicas: the structures of micro_replica_files differ from this core \
                     by up to {:.3e} Bohr (a coordinate); their environments are the starts of \
                     the replicas all the same",
                    worst
                ));
            }
            (starts, None)
        };
        let mut envs = self.each(|r, c| {
            let how = Relax {
                tolerance,
                from: starts[r].as_deref(),
            };
            c.relax_at(x, false, spin, excited_state, how)
        })?;
        let mut energies = self.each(|r, c| c.inner_energy(x, &envs[r], spin, excited_state))?;
        let (mut reseeds, mut duplicates) = (0, 0);
        for j in 1..k {
            let mut attempt = 0;
            loop {
                let twin =
                    (0..j).find(|&i| same_minimum(&envs[i], &envs[j], energies[i], energies[j]));
                let Some(i) = twin else {
                    break;
                };
                let Some(env0) = env0.as_ref().filter(|_| attempt < RESEEDS) else {
                    duplicates += 1;
                    notes.push(format!(
                        "ONIOM replicas: replica {} relaxed into the minimum of replica {} at the \
                         start{}; both are kept with their weights",
                        j + 1,
                        i + 1,
                        if env0.is_some() {
                            format!(" after {RESEEDS} new starts")
                        } else {
                            " (structures of the file)".to_string()
                        }
                    ));
                    break;
                };
                attempt += 1;
                reseeds += 1;
                let start = self.replica_start(j, x, env0, spin, excited_state, attempt)?;
                let how = Relax {
                    tolerance,
                    from: Some(&start),
                };
                envs[j] = self.replicas[j].relax_at(x, false, spin, excited_state, how)?;
                energies[j] = self.replicas[j].inner_energy(x, &envs[j], spin, excited_state)?;
            }
        }
        let lowest = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let list: Vec<String> = energies
            .iter()
            .enumerate()
            .map(|(r, e)| format!("{}: {:+.3e}", r + 1, e - lowest))
            .collect();
        notes.push(format!(
            "ONIOM replicas: {k} environment(s) started ({}; {reseeds} new start(s) of a \
             replica that began in the minimum of another); their energies on the surface of \
             the relaxations relative to the lowest (Hartree): {}",
            self.settings.replicas.start.name(),
            list.join(", ")
        ));
        let progress = {
            let mut record = self
                .replica_record
                .lock()
                .map_err(|_| MoptError::calc("ONIOM replicas: the record is poisoned"))?;
            record.generated = true;
            record.reseeds += reseeds;
            record.duplicates += duplicates;
            self.state.lock().ok().and_then(|s| s.progress.clone())
        };
        if let Some((log, _)) = progress {
            for n in notes {
                log.info(&n);
            }
        }
        Ok(())
    }

    /// The surface of the soft minimum of the replicas at the core `sys` (the replicas
    /// evaluated, their Hessians where asked and their weight is not negligible, and combined).
    pub(super) fn evaluate_replicas(
        &self,
        sys: &MolecularSystem,
        req: &EvalRequest,
    ) -> MoptResult<EvalOutput> {
        self.evaluate_replicas_with(sys, req, req.need_hessian && req.for_stability)
    }

    /// The replicas at the core `core` relaxed to the tolerance of the checks of stability
    /// (the soft minimum evaluated as for a check, without its Hessian): the structure that a
    /// vibrational analysis takes.
    pub(super) fn relax_replicas_for_check(
        &self,
        core: &MolecularSystem,
        excited_state: usize,
    ) -> MoptResult<EvalOutput> {
        let req = EvalRequest {
            excited_state,
            ..EvalRequest::ENERGY_GRAD
        };
        self.evaluate_replicas_with(core, &req, true)
    }

    /// The Hessian of the model systems at the core `x` computed once before the replicas
    /// take their Hessians (in parallel they would each compute it), on the route the Hessian
    /// of the core takes.
    fn prefill_model_hessian(
        &self,
        x: &Col<f64>,
        spin: SpinCharge,
        excited_state: usize,
    ) -> MoptResult<()> {
        let Some(model) = self.model_part.as_ref() else {
            return Ok(());
        };
        let env: Vec<f64> = gather(&self.template.coords_flat_bohr(), &self.split.environment)
            .iter()
            .copied()
            .collect();
        let mut full = assemble(&self.template, &self.split, x, &env)?;
        full.spin_charge = spin;
        if self.settings.core_hessian != super::CoreHessian::Approximate
            && self.analytic_core_hessian()
        {
            let atoms: Vec<usize> = (0..full.n_atoms()).collect();
            self.model_hessian_on(model, &full, &atoms, excited_state)?;
        } else if self.routes.model.is_some() {
            self.model_hessian_on(model, &full, &self.split.core, excited_state)?;
        }
        Ok(())
    }

    /// [`Self::evaluate_replicas`]; `check`: the replicas relaxed to the tolerance of the
    /// checks of stability.
    fn evaluate_replicas_with(
        &self,
        sys: &MolecularSystem,
        req: &EvalRequest,
        check: bool,
    ) -> MoptResult<EvalOutput> {
        let x = sys.coords_flat_bohr();
        let spin = sys.spin_charge;
        let tolerance = check.then(|| self.check_tolerance());
        self.start_replicas(&x, spin, req.excited_state, tolerance)?;
        let model = self.model_values(&x, spin, req.excited_state)?;
        let k = self.replicas.len();
        let t = self.settings.replicas.temperature;
        let plain = EvalRequest {
            need_hessian: false,
            ..*req
        };
        let mut outs = self.each(|_, c| c.evaluate_one(sys, &plain, model.as_ref(), check))?;
        let energies = |outs: &[EvalOutput]| -> Vec<f64> {
            outs.iter().map(|o| o.energy.value()).collect::<Vec<f64>>()
        };
        let mut soft = soft_minimum(&energies(&outs), t)?;
        let mut hessians: Vec<Option<Mat<f64>>> = vec![None; k];
        let mut firsts: Vec<Option<Mat<f64>>> = vec![None; k];
        let mut judged = true;
        let mut skipped = 0;
        if req.need_hessian {
            let mut asked = vec![false; k];
            loop {
                let wanted: Vec<usize> = (0..k)
                    .filter(|&r| !asked[r] && soft.weights[r] >= NEGLIGIBLE_WEIGHT)
                    .collect();
                if wanted.is_empty() {
                    break;
                }
                self.prefill_model_hessian(&x, spin, req.excited_state)?;
                let taken = self.each_of(&wanted, |_, c| {
                    c.evaluate_one(sys, req, model.as_ref(), check)
                })?;
                for (&r, out) in wanted.iter().zip(taken) {
                    asked[r] = true;
                    if out.hessian_bohr.is_none() {
                        judged = false;
                    }
                    hessians[r] = out.hessian_bohr.clone();
                    firsts[r] = self.replicas[r]
                        .state
                        .lock()
                        .ok()
                        .and_then(|mut s| s.check_first.take());
                    outs[r] = out;
                }
                soft = soft_minimum(&energies(&outs), t)?;
            }
            skipped = asked.iter().filter(|a| !**a).count();
        }
        let gradients: Vec<&Col<f64>> = outs.iter().map(|o| &o.gradient_bohr).collect();
        let gradient = soft_gradient(&gradients, &soft.weights);
        let hessian = if req.need_hessian && judged {
            let refs: Vec<Option<&Mat<f64>>> = hessians.iter().map(Option::as_ref).collect();
            Some(soft_hessian(&refs, &gradients, &soft)?)
        } else {
            None
        };
        let progress = self.state.lock().ok().and_then(|s| s.progress.clone());
        // The error of the combined curvatures of a check: the combination of the Hessians of
        // the replicas before their environments were moved to the minima of their quadratic
        // models against the one after.
        if let (true, Some(h), Some((log, _))) = (check, hessian.as_ref(), progress.as_ref()) {
            if firsts.iter().any(Option::is_some) {
                let before: Vec<Option<&Mat<f64>>> = firsts
                    .iter()
                    .zip(&hessians)
                    .map(|(f, h)| f.as_ref().or(h.as_ref()))
                    .collect();
                if let Ok(h0) = soft_hessian(&before, &gradients, &soft) {
                    let (masses, symmetries) = self.core_masses_and_symmetries(&x);
                    let eig = |m: &Mat<f64>| {
                        crate::opt::mode_kill::mass_weighted_eigenvalues(
                            m,
                            Some(masses.as_slice()),
                            symmetries.as_ref(),
                        )
                    };
                    if let (Ok(a), Ok(b)) = (eig(&h0), eig(h)) {
                        let lowest: Vec<String> = b
                            .iter()
                            .zip(&a)
                            .take(6)
                            .map(|(v, u)| format!("{v:+.4e} +- {:.1e}", (v - u).abs()))
                            .collect();
                        log.info(&format!(
                            "  Hessian of the soft minimum for a check of stability: lowest \
                             mass-weighted curvatures (Hartree/(Bohr^2 amu)) with the change \
                             over the steps of the environments of the replicas as their error: \
                             {}",
                            lowest.join(", ")
                        ));
                    }
                }
            }
        }
        // Replicas in the same minimum here (reported where that changes).
        let envs: Vec<Option<Vec<f64>>> = self
            .replicas
            .iter()
            .map(|c| c.kept_environment(&x))
            .collect();
        let mut same = Vec::new();
        for j in 1..k {
            for i in 0..j {
                if let (Some(a), Some(b)) = (&envs[i], &envs[j]) {
                    if same_minimum(a, b, outs[i].energy.value(), outs[j].energy.value()) {
                        same.push((i, j));
                    }
                }
            }
        }
        {
            let mut record = self
                .replica_record
                .lock()
                .map_err(|_| MoptError::calc("ONIOM replicas: the record is poisoned"))?;
            record.evaluations += 1;
            record.skipped_hessians += skipped;
            if record.same != same {
                if let Some((log, _)) = &progress {
                    if same.is_empty() {
                        log.info("  microiterations: the replicas are in distinct minima again");
                    } else {
                        let pairs: Vec<String> = same
                            .iter()
                            .map(|(i, j)| format!("{} and {}", i + 1, j + 1))
                            .collect();
                        log.info(&format!(
                            "  microiterations: replicas {} are in the same minimum of the \
                             environment here; they keep their weights",
                            pairs.join(", ")
                        ));
                    }
                }
                record.same = same;
            }
            if record.points.len() >= SOFT_POINTS_KEPT {
                record.points.pop_front();
            }
            record.points.push_back(SoftPoint {
                core: x.iter().copied().collect(),
                energies: energies(&outs),
                weights: soft.weights.clone(),
            });
        }
        if let Some((log, _)) = &progress {
            let list: Vec<String> = outs
                .iter()
                .zip(&soft.weights)
                .enumerate()
                .map(|(r, (o, w))| {
                    format!(
                        "{}: {:+.3e} (weight {w:.3})",
                        r + 1,
                        o.energy.value() - outs[soft.lowest].energy.value()
                    )
                })
                .collect();
            log.info(&format!(
                "  microiterations: soft minimum of {k} replicas at {t} K: E {:.10}, the lowest \
                 replica {} at {:.10}; replica energies relative to it (Hartree): {}{}",
                soft.energy,
                soft.lowest + 1,
                outs[soft.lowest].energy.value(),
                list.join(", "),
                if skipped > 0 {
                    format!("; {skipped} Hessian(s) of replicas of negligible weight not taken")
                } else {
                    String::new()
                }
            ));
        }
        let lowest_out = &outs[soft.lowest];
        let orbitals = match model {
            Some(m) => m.high.and_then(|h| h.orbitals),
            None => lowest_out.orbitals.clone(),
        };
        let mut out = EvalOutput::energy_gradient(Quantity::new(soft.energy), gradient);
        out.hessian_bohr = hessian;
        out.orbitals = orbitals;
        if let Ok(mut state) = self.state.lock() {
            state.last_core = Some((
                crate::math::linalg::col_max_abs(&out.gradient_bohr),
                soft.energy,
            ));
            state.last_state = Some((spin, req.excited_state));
        }
        Ok(out)
    }

    /// The replica energies and weights kept for the core `x` (within [`super::SAME_IN_FILES`]),
    /// the newest of them.
    fn soft_point(&self, x: &Col<f64>, within: f64) -> Option<SoftPoint> {
        let record = self.replica_record.lock().ok()?;
        record
            .points
            .iter()
            .rev()
            .min_by(|a, b| distance(&a.core, x).total_cmp(&distance(&b.core, x)))
            .filter(|p| distance(&p.core, x) < within)
            .cloned()
    }

    /// The replica of the lowest energy at the core `core` and its weight, evaluating the soft
    /// minimum there where it was not evaluated.
    fn lowest_replica_at(&self, core: &MolecularSystem) -> MoptResult<(usize, f64, SoftPoint)> {
        let x = core.coords_flat_bohr();
        let point = match self.soft_point(&x, super::SAME_IN_FILES) {
            Some(p) => p,
            None => {
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
                self.evaluate_replicas(&at, &req)?;
                self.soft_point(&x, super::SAME_IN_FILES).ok_or_else(|| {
                    MoptError::calc("ONIOM replicas: no soft minimum at the structure")
                })?
            }
        };
        let r = point
            .energies
            .iter()
            .copied()
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |(k, m), (i, e)| if e < m { (i, e) } else { (k, m) },
            )
            .0;
        Ok((r, point.weights[r], point))
    }

    /// The whole structures of the replicas at the core `core` (each with its energy and
    /// weight), with the index of the lowest: what the files of a run with replicas write (the
    /// lowest in the place of the single environment, all of them beside it).
    pub fn replica_systems(
        &self,
        core: &MolecularSystem,
    ) -> MoptResult<Option<(usize, Vec<(MolecularSystem, f64, f64)>)>> {
        if self.replicas.is_empty() {
            return Ok(None);
        }
        let (lowest, _, point) = self.lowest_replica_at(core)?;
        let systems = self
            .replicas
            .iter()
            .zip(point.energies.iter().zip(&point.weights))
            .map(|(c, (&e, &w))| Ok((c.full_system(core)?, e, w)))
            .collect::<MoptResult<Vec<_>>>()?;
        Ok(Some((lowest, systems)))
    }

    /// The tag of the comment line of a whole structure written with the lowest replica.
    pub(super) fn replica_tag(&self, lowest: usize, weight: f64) -> String {
        format!(
            "micro_replica={}/{} weight={weight:.4} replica_temperature={}",
            lowest + 1,
            self.replicas.len(),
            self.settings.replicas.temperature
        )
    }

    /// The whole structure of the core `core` with the environment of its lowest replica.
    pub(super) fn full_system_of_replicas(
        &self,
        core: &MolecularSystem,
    ) -> MoptResult<MolecularSystem> {
        let (r, _, _) = self.lowest_replica_at(core)?;
        self.replicas[r].full_system(core)
    }

    /// The whole structure of a frame of an analysis with the environment of the lowest
    /// replica of the nearest evaluated core (as it is, not relaxed).
    pub(super) fn nearest_full_system_of_replicas(
        &self,
        core: &MolecularSystem,
    ) -> MoptResult<MolecularSystem> {
        let x = core.coords_flat_bohr();
        let r = self.soft_point(&x, f64::INFINITY).map_or(0, |p| {
            p.energies
                .iter()
                .copied()
                .enumerate()
                .fold(
                    (0, f64::INFINITY),
                    |(k, m), (i, e)| if e < m { (i, e) } else { (k, m) },
                )
                .0
        });
        self.replicas[r].nearest_full_system(core)
    }

    /// The relaxed environment of the core `x` kept by this surface (one environment).
    fn kept_environment(&self, x: &Col<f64>) -> Option<Vec<f64>> {
        let state = self.state.lock().ok()?;
        state
            .kept
            .iter()
            .rev()
            .find(|k| k.relaxed && distance(&k.core, x) <= super::SAME_STRUCTURE)
            .map(|k| k.environment.clone())
    }

    /// The relaxed environments of every replica at the core `sys`, one after the other (the
    /// record of the journal of a run).
    pub(super) fn replica_environments(&self, sys: &MolecularSystem) -> Option<Vec<f64>> {
        let x = sys.coords_flat_bohr();
        let mut out = Vec::new();
        for c in &self.replicas {
            out.extend(c.kept_environment(&x)?);
        }
        Some(out)
    }

    /// Given back by the journal of a restart: the environments of every replica at the core
    /// `sys` (one after the other), or that of the first one alone (a structure of a run with
    /// one environment).
    pub(super) fn recall_replica_environments(&self, sys: &MolecularSystem, environment: &[f64]) {
        let m = 3 * self.split.environment.len();
        let k = self.replicas.len();
        if environment.len() == k * m {
            for (c, env) in self.replicas.iter().zip(environment.chunks(m)) {
                c.recall_environment(sys, env);
            }
            if let Ok(mut record) = self.replica_record.lock() {
                record.generated = true;
            }
        } else if environment.len() == m {
            if let Some(first) = self.replicas.first() {
                first.recall_environment(sys, environment);
            }
        }
    }

    /// What the run log says about the replicas when the surface is set up, and whether the
    /// input structure was written by a run with replicas whose surface this one is not.
    pub(super) fn replica_notes(&self) -> Vec<String> {
        let mut out = Vec::new();
        let k = self.replicas.len();
        let s = &self.settings.replicas;
        if k > 1 {
            let start = match &s.start {
                ReplicaStart::Perturb => format!(
                    "perturb: every free atom moved by {:.3} Angstrom, seed {}",
                    s.amplitude * Units::BOHR_TO_ANGSTROM,
                    s.seed
                ),
                ReplicaStart::Dynamics => format!(
                    "md: {} steps of molecular dynamics of the environment at {} K with the core \
                     held, seed {}",
                    s.dynamics_steps, s.temperature, s.seed
                ),
                ReplicaStart::Files(p) => format!("files: {}", p.display()),
                ReplicaStart::Structures(_) => "files: the given structures".to_string(),
            };
            let cost = if self.model_part.is_some() && !self.settings.embedded {
                "the higher levels are computed once per core and the lowest level once per \
                 replica"
            } else {
                "with electronic embedding every replica computes the higher levels as well, \
                 so a step costs about K times one with one environment"
            };
            let order = if self.replicas_in_parallel() {
                "in parallel"
            } else {
                "one after the other (a calculator with electrons is evaluated per replica)"
            };
            out.push(format!(
                "ONIOM replicas (new): {k} replicas of the environment (start {start}) combined \
                 by the soft minimum -kT ln sum exp(-E_k / kT) at {} K; {cost}; the replicas \
                 relax {order}. The surface is not a free energy of the environment, lies below \
                 the lowest replica by at most kT ln K ({:.3e} Hartree), and its energies compare \
                 only with those of runs with the same replicas and temperature.",
                s.temperature,
                Units::BOLTZMANN_HARTREE_PER_K * s.temperature * (k as f64).ln()
            ));
        }
        // A structure written by a run with replicas carries their number and temperature.
        let written = self.template.comment.split_whitespace().find_map(|t| {
            t.strip_prefix("micro_replica=")
                .and_then(|v| v.split_once('/'))
                .and_then(|(_, n)| n.parse::<usize>().ok())
        });
        let temperature = self.template.comment.split_whitespace().find_map(|t| {
            t.strip_prefix("replica_temperature=")
                .and_then(|v| v.parse::<f64>().ok())
        });
        if let Some(n) = written {
            let same = k == n
                && matches!(
                    s.start,
                    ReplicaStart::Files(_) | ReplicaStart::Structures(_)
                )
                && temperature.is_none_or(|t| (t - s.temperature).abs() < 1e-9);
            if !same {
                out.push(format!(
                    "ONIOM replicas: the input structure was written by a run with {n} \
                     environment replicas{}, whose surface is their soft minimum; this run {}, so \
                     it is not on that surface, and its energies and curvatures are not those of \
                     that run. To go on on it: micro_replicas={n} micro_replica_start=files \
                     micro_replica_files=<the _replicas.xyz file written beside the structure>{}.",
                    temperature.map_or_else(String::new, |t| format!(" at {t} K")),
                    if k == 1 {
                        "has one environment".to_string()
                    } else if s.start.name() != "files" {
                        format!("starts its {k} replicas anew ({})", s.start.name())
                    } else {
                        format!("has {k} replicas at {} K", s.temperature)
                    },
                    temperature
                        .map_or_else(String::new, |t| format!(" micro_replica_temperature={t}"))
                ));
            }
        }
        out
    }

    /// What the replicas did so far, for the log of the run.
    pub fn replica_summary(&self) -> Option<String> {
        if self.replicas.is_empty() {
            return None;
        }
        let record = self.replica_record.lock().ok()?;
        Some(format!(
            "ONIOM replicas: {} environment replicas combined by a soft minimum at {} K in {} \
             evaluation(s); {} new start(s) of a replica that began in the minimum of another, {} \
             replica(s) left in the minimum of another at the start, {} Hessian(s) of replicas of \
             negligible weight not taken.",
            self.replicas.len(),
            self.settings.replicas.temperature,
            record.evaluations,
            record.reseeds,
            record.duplicates,
            record.skipped_hessians
        ))
    }
}
