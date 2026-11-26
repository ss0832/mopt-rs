//! A quasi-Newton step for every image of a chain, as MultiOptPy mixes it with the step
//! along the forces when the images have Hessians (`-modelhess`, `-fc`).
//!
//! Every image has its own Hessian and its own restricted-step RFO minimizer (the one of
//! `optmain`). In each iteration, for image `i` with its full gradient `g` (not the force
//! of the chain: the component along the chain is not removed):
//!
//! 1. an inner image gets the curvature of the energy along the chain added to its
//!    Hessian, `H <- H + gamma t t^T` ([`path_curvature`]; `t` is the component that the
//!    force rule removed from the gradient, not a unit vector), in every iteration, on top
//!    of what was added before;
//! 2. the Hessian is updated from the last step and the change of the gradient (the block
//!    Bofill formula for inner images, block FSB for the ends). MultiOptPy's block updates
//!    drop every pair right after using it, which makes them single-pair updates; here the
//!    pairs of the last iterations are kept and imposed together, and one pair as in
//!    MultiOptPy is `pairs = 1` ([`ChainRfoSettings::pairs`]; see
//!    `docs/source/migration_paths.md`, "Chain relaxation": both converge alike);
//! 3. the RFO step of a minimization is taken with this Hessian, without a trust radius,
//!    and shortened to 0.1 Bohr (0.2 for an end image).
//!
//! MultiOptPy asks its RS-I-RFO for a step restricted to a trust radius (0.2 Bohr for an
//! inner image, 0.5 for an end image), but the length of the step that its solver returns
//! does not depend on the parameter that is varied to meet the radius, so the step it gets
//! is the unrestricted one (an inner image of the reference data: 0.56 Bohr with a radius
//! of 0.2). What reaches the chain is that step cut to the lengths above, and this is
//! what is done here. The trust radius of MultiOptPy's minimizers, which they adjust with
//! the energy 0 they are told in every iteration, therefore has no effect either.
//!
//! MultiOptPy creates the minimizers of the inner images for minima (saddle order 0), so
//! that its branches for "transition state optimization of the highest image" and for
//! its NEB mode do nothing; they are not reproduced.
//!
//! # Steps across the chain
//!
//! A minimization with the full gradient takes every inner image down the valley towards
//! a minimum; only the redistribution of the images brings them back, and the chain never
//! comes to rest (on the Muller-Brown surface the forces stay at 0.03 to 0.1 of their
//! first values). The above is therefore what is done under `Compatibility::MultiOptPy`
//! only. Otherwise the quasi-Newton step of an inner image is a step across the chain,
//! the counterpart of the force of the chain (as in ase_qsm, whose quasi-Newton steps are
//! given the projected force and a Hessian projected the same way):
//!
//! - with `Q` the projection on the directions along the chain that the force rule removes
//!   (one unit vector per block) and `P = 1 - Q`, the step is the RFO step of the model
//!   with the gradient `P g` and the Hessian `P H P + Q`, without its component in `Q`;
//! - an image that climbs gets the model with one direction reversed, the tangent `t` of
//!   the whole image: gradient `g - 2 (g . t) t`, Hessian `P H P - Q H Q` with
//!   `Q = t t^T`, so that the step goes up along the chain and down across it;
//! - the curvature along the chain is not added to the Hessians (that direction is not
//!   part of the model), and the Hessians are updated with the full gradients as before:
//!   they stay Hessians of the surface;
//! - what the surface does not know is added to the model: the force of an image depends
//!   on its direction along the chain, which turns when the image moves. With the
//!   direction `t = (x' - x) / h` to the neighbour `x'` uphill, a step `d` across the
//!   chain turns `t` by `-d / h`, and the part of the gradient that is no longer removed
//!   is `(g . t) d / h`: the gradient across the chain rises by `G / h` per unit step,
//!   `G = g . t` being the slope along the chain. The model Hessian is therefore
//!   `P H P + sum_k (G_k / h_k) P_k + Q`, block by block (`G_k` and `h_k`: slope and
//!   distance of block `k`; blocks with a negative `G_k` add nothing, so that the model
//!   is never softer than the surface). Where the energy rises steeply along the chain
//!   this is as large as the curvature of the surface; without it the images step too
//!   far and neighbours swing against each other without end (Muller-Brown surface:
//!   forces of 0.03 to 0.07 in a cycle of two iterations). Images at an extremum of the
//!   profile, where the slope vanishes, and images that climb get no such term;
//! - the end images are minimized with their full gradients as before.
//!
//! On the ridge between two transition states the models are reversed along a transverse
//! mode as well ([`ChainRidge`]).
//!
//! # Where the Hessians are
//!
//! A chain has as many Hessians as images. MultiOptPy keeps them in files
//! (`tmp_hessian_<i>.npy` in the directory of the run) and loads the one of an image while
//! it works on that image; so does [`ChainRfo`] when it is given a directory
//! ([`ChainRfoSettings::directory`]), in the same format and under the same names, so that
//! one Hessian at a time is in memory. Without a directory they stay in memory.
//!
//! MultiOptPy: `Optimizer/rfo_neb.py:RFOOptimizer`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;

use faer::{Col, Mat};

use super::align::RigidMotion;
use super::force::{path_curvature, UpwindProjection};
use super::global::{GlobalRfoSettings, GlobalSolve};
use super::redistribute::ImageSource;
use super::ridge::TransverseMode;
use super::Compatibility;
use crate::domain::{MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::npy::{read_npy_matrix, write_npy_matrix};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen};
use crate::math::projection::{build_tr_rot_basis, project_out_basis_mat, project_out_basis_vec};
use crate::opt::step::rsirfo::{Rsirfo, RsirfoTrust};
use crate::opt::step::{PreviousPoint, StepGenerator, StepInput};
use crate::opt::update::{
    apply_update, project_out_rigid, BlockKind, BlockUpdater, HessianUpdater, UpdaterRegistry,
};

/// Settings of the quasi-Newton steps of a chain (MultiOptPy's values).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainRfoSettings {
    /// Share of the quasi-Newton step in the step of an inner image (`-rrs`).
    pub ratio: f64,
    /// Longest quasi-Newton step of an inner image (Bohr).
    pub inner_step: f64,
    /// Longest quasi-Newton step of an end image (Bohr).
    pub end_step: f64,
    /// Hessian update of the inner images (a name of the update registry).
    pub inner_update: String,
    /// Hessian update of the end images.
    pub end_update: String,
    /// Pairs of step and gradient change that the block updates keep (`None`: their own
    /// number; `Some(1)`: one pair, as MultiOptPy's block updates in effect do).
    pub pairs: Option<usize>,
    /// Directory in which the Hessians of the images are kept between the iterations, as
    /// `tmp_hessian_<i>.npy` (`None`: in memory).
    pub directory: Option<PathBuf>,
    /// One step for the whole chain instead of one per image ([`super::global`];
    /// experimental, not in MultiOptPy's form).
    pub global: Option<GlobalRfoSettings>,
    /// The images get their steps from the highest one down, and the step of the
    /// neighbour uphill enters the force of an image (see the module documentation; not
    /// in MultiOptPy's form).
    pub sweep: bool,
    /// For chains of structures: take the translations and rotations of an image out of its
    /// Hessian after every update (`-update_proj`; on by default).
    pub project_updates: bool,
}

impl Default for ChainRfoSettings {
    fn default() -> Self {
        Self {
            ratio: 0.5,
            inner_step: 0.1,
            end_step: 0.2,
            inner_update: "block_bofill".to_string(),
            end_update: "block_fsb".to_string(),
            pairs: None,
            directory: None,
            global: None,
            sweep: false,
            project_updates: true,
        }
    }
}

/// A chain on the ridge between two transition states.
///
/// The chain joins two transition states over a saddle point of index 2
/// (the Hessian form of the band on a ridge; [`super::ridge`] has the gradient-only form and
/// the reasons). Every inner image has a transverse mode `u`: the lowest eigenvector of the
/// model across the chain `P H P + Q` (the rigid-body motions and the direction of the chain
/// shifted out of it), or of its two lowest the one that continues the mode of the last
/// iteration. `u` is an eigenvector of the model, so the model of the image function is
/// exact: the gradient `g - 2 (g . u) u` and the Hessian `M - 2 lambda_u u u^T` (`lambda_u`
/// the curvature along `u`), and the RFO step of its minimization goes up along `u` and down
/// across the rest. A climbing image, whose model already reverses the direction of the
/// chain, gets the same reversal along `u`: its model has two directions reversed, and its
/// fixed point is the index-2 saddle. The forces of the chain get the same reversal
/// (`+ 2 (g . u) u`), so that the step along the forces agrees.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainRidge {
    /// The imaginary modes of the two end images.
    pub end_modes: [Col<f64>; 2],
    /// Which mode across the chain is maximized.
    pub transverse: TransverseMode,
    /// Below this overlap with the mode of the last iteration the mode counts as switched.
    pub switch_overlap: f64,
}

impl ChainRidge {
    /// The ridge of the two imaginary modes, the lowest mode across the chain maximized.
    pub fn new(end_modes: [Col<f64>; 2]) -> Self {
        Self {
            end_modes,
            transverse: TransverseMode::Lowest,
            switch_overlap: 0.5,
        }
    }
}

/// The transverse modes of the images of a chain on a ridge.
#[derive(Debug, Clone, Default)]
pub struct RidgeModes {
    /// The mode of every image (`None` at the ends).
    pub modes: Vec<Option<Col<f64>>>,
    /// The curvature along it (`NaN` at the ends).
    pub curvatures: Vec<f64>,
    /// The images whose mode switched (another of the two lowest than the last time, or a
    /// small overlap with the last one).
    pub switches: Vec<usize>,
}

/// The Hessian update `name`. The block Bofill and block FSB updates keep `pairs` pairs if
/// that is given, and apply MultiOptPy's thresholds to the pairs as they are if `legacy`.
fn updater(
    registry: &UpdaterRegistry,
    name: &str,
    pairs: Option<usize>,
    legacy: bool,
) -> MoptResult<Box<dyn HessianUpdater>> {
    let kind = match name.to_ascii_lowercase().as_str() {
        "block_bofill" => Some(BlockKind::Bofill),
        "block_fsb" => Some(BlockKind::Fsb),
        _ => None,
    };
    match kind {
        Some(kind) if pairs.is_some() || legacy => {
            let mut block = BlockUpdater::new(name, kind);
            if let Some(pairs) = pairs {
                block.max_window = pairs.max(1);
                block.block_size = block.block_size.min(block.max_window);
            }
            block.scale_pairs = !legacy;
            Ok(Box::new(block))
        }
        _ => registry.resolve(name),
    }
}

/// A trust radius that no step reaches.
const NO_RADIUS: f64 = 1e12;

/// What the force rule found along the chain, image by image.
#[derive(Debug, Clone, Copy)]
pub struct AlongChain<'a> {
    /// Components removed from the gradients.
    pub removed: &'a [Col<f64>],
    /// Directions of the chain (not normalized).
    pub directions: &'a [Col<f64>],
    /// Images that climb.
    pub climbing: &'a [usize],
}

/// The directions along the chain at one image: a unit vector per block.
pub(super) struct Along {
    /// The unit vectors as columns (blocks without a direction have none).
    pub(super) units: Mat<f64>,
    /// First coordinate of the block of every column.
    pub(super) starts: Vec<usize>,
    /// Length of the direction in the block of every column (the distance to the
    /// neighbour that the direction points to or from).
    lengths: Vec<f64>,
    /// Coordinates per block.
    pub(super) block: usize,
}

impl Along {
    /// The unit vectors along `direction`, one per block of `block` coordinates.
    pub(super) fn new(direction: &Col<f64>, block: usize) -> Self {
        let n = direction.nrows();
        let block = block.clamp(1, n.max(1));
        let (mut starts, mut lengths) = (Vec::new(), Vec::new());
        for start in (0..n).step_by(block) {
            let end = (start + block).min(n);
            let length = (start..end)
                .map(|k| direction[k] * direction[k])
                .sum::<f64>()
                .sqrt();
            if length > 1e-12 {
                starts.push(start);
                lengths.push(length);
            }
        }
        let mut units = Mat::<f64>::zeros(n, starts.len());
        for (column, (&start, &length)) in starts.iter().zip(&lengths).enumerate() {
            for k in start..(start + block).min(n) {
                units[(k, column)] = direction[k] / length;
            }
        }
        Self {
            units,
            starts,
            lengths,
            block,
        }
    }

    /// `Q v` for the projection `Q = U U^T` on the unit vectors.
    pub(super) fn part(&self, vector: &Col<f64>) -> Col<f64> {
        &self.units * (self.units.transpose() * vector)
    }

    /// What the turning of the direction adds to the curvature across the chain, block by
    /// block: `G_k / h_k` with the slope `G_k` of `gradient` towards the neighbour uphill
    /// (`uphill`: the direction points to it; otherwise away from it), zero where that is
    /// negative.
    pub(super) fn stiffening(&self, gradient: &Col<f64>, uphill: bool) -> Vec<f64> {
        let slopes = self.units.transpose() * gradient;
        let sign = if uphill { 1.0 } else { -1.0 };
        (0..self.starts.len())
            .map(|column| (sign * slopes[column] / self.lengths[column]).max(0.0))
            .collect()
    }

    /// `sum_k stiffening[k] P_k v`: the stiffening of every block across its direction.
    pub(super) fn stiffened(&self, stiffening: &[f64], v: &Col<f64>) -> Col<f64> {
        let n = v.nrows();
        let mut out = Col::<f64>::zeros(n);
        for (column, &s) in stiffening.iter().enumerate().take(self.starts.len()) {
            if s == 0.0 {
                continue;
            }
            let range = self.starts[column]..(self.starts[column] + self.block).min(n);
            let along: f64 = range.clone().map(|r| self.units[(r, column)] * v[r]).sum();
            for r in range {
                out[r] = s * (v[r] - along * self.units[(r, column)]);
            }
        }
        out
    }

    /// The Hessian of the model across the chain: `P H P + Q` plus `stiffening[k] P_k` in
    /// every block, or `P H P - Q H Q` for an image that climbs (`Q = U U^T`,
    /// `P = 1 - Q`).
    fn hessian(&self, hessian: &Mat<f64>, climbing: bool, stiffening: &[f64]) -> Mat<f64> {
        let n = hessian.nrows();
        let units = &self.units;
        let qh = units * (units.transpose() * hessian);
        let qhq = (&qh * units) * units.transpose();
        let q = units * units.transpose();
        let mut model = Mat::from_fn(n, n, |r, c| {
            let php = hessian[(r, c)] - qh[(r, c)] - qh[(c, r)] + qhq[(r, c)];
            let rest = if climbing {
                -0.5 * (qhq[(r, c)] + qhq[(c, r)])
            } else {
                q[(r, c)]
            };
            0.5 * (php + hessian[(c, r)] - qh[(c, r)] - qh[(r, c)] + qhq[(c, r)]) + rest
        });
        for (column, &extra) in stiffening.iter().enumerate().take(self.starts.len()) {
            let start = self.starts[column];
            for r in start..(start + self.block).min(n) {
                for c in start..(start + self.block).min(n) {
                    let same = if r == c { 1.0 } else { 0.0 };
                    model[(r, c)] += extra * (same - units[(r, column)] * units[(c, column)]);
                }
            }
        }
        model
    }
}

/// Where the Hessian of an image is between the iterations.
pub(super) enum Kept {
    Memory(Mat<f64>),
    File(PathBuf),
}

impl Kept {
    /// The Hessian where it is in memory (`None`: it is in a file).
    pub(super) fn in_memory(&self) -> Option<&Mat<f64>> {
        match self {
            Self::Memory(hessian) => Some(hessian),
            Self::File(_) => None,
        }
    }

    /// The Hessian, to be given back with [`Self::put`].
    pub(super) fn take(&mut self) -> MoptResult<Mat<f64>> {
        match self {
            Self::Memory(hessian) => Ok(std::mem::replace(hessian, Mat::zeros(0, 0))),
            Self::File(path) => read_npy_matrix(path),
        }
    }

    pub(super) fn put(&mut self, hessian: Mat<f64>) -> MoptResult<()> {
        match self {
            Self::Memory(kept) => {
                *kept = hessian;
                Ok(())
            }
            Self::File(path) => write_npy_matrix(path, &hessian),
        }
    }

    /// A copy of the Hessian.
    pub(super) fn copy(&self) -> MoptResult<Mat<f64>> {
        match self {
            Self::Memory(hessian) => Ok(hessian.clone()),
            Self::File(path) => read_npy_matrix(path),
        }
    }
}

/// The Hessian and the minimizer of one image.
pub(super) struct ImageRfo {
    generator: Rsirfo,
    pub(super) updater: Box<dyn HessianUpdater>,
    pub(super) hessian: Kept,
    /// Point and gradient of the last iteration.
    pub(super) last: Option<(Col<f64>, Col<f64>)>,
    pub(super) iteration: usize,
}

/// The minimizer of an image with the identity as its Hessian.
fn fresh_image(
    registry: &UpdaterRegistry,
    update: &str,
    pairs: Option<usize>,
    legacy: bool,
    n: usize,
) -> MoptResult<ImageRfo> {
    // No restriction by a trust radius (see the module documentation).
    let policy = RsirfoTrust {
        initial: NO_RADIUS,
        ..RsirfoTrust::for_saddle_order(0, Some(NO_RADIUS), Some(NO_RADIUS))
    };
    Ok(ImageRfo {
        generator: Rsirfo::new(policy),
        updater: updater(registry, update, pairs, legacy)?,
        hessian: Kept::Memory(Mat::identity(n, n)),
        last: None,
        iteration: 0,
    })
}

/// The quasi-Newton steps of the images of a chain.
pub struct ChainRfo {
    pub(super) settings: ChainRfoSettings,
    pub(super) images: Vec<ImageRfo>,
    /// For chains of structures: the atoms, to take translations and rotations out of
    /// gradients and Hessians.
    pub(super) template: Option<MolecularSystem>,
    /// The force rule of the chain: its blocks, and whether the steps are MultiOptPy's.
    pub(super) rule: UpwindProjection,
    /// Components of an image.
    pub(super) n: usize,
    /// What the last solve for the whole chain did.
    pub(super) solve: Option<GlobalSolve>,
    /// The transverse modes of the images on a ridge, for the steps of this iteration.
    pub(super) ridge: Vec<Option<Col<f64>>>,
    /// The transverse modes of the last iteration.
    ridge_previous: Vec<Option<Col<f64>>>,
}

impl ChainRfo {
    /// Minimizers for `images` images of `n` components, every Hessian the identity until
    /// it is set. `template` is given for chains of structures; `rule` is the force rule
    /// of the chain.
    pub fn new(
        settings: ChainRfoSettings,
        images: usize,
        n: usize,
        template: Option<MolecularSystem>,
        rule: UpwindProjection,
    ) -> MoptResult<Self> {
        if images < 2 {
            return Err(MoptError::config("a chain needs at least two images"));
        }
        if template.as_ref().is_some_and(|t| t.dof() != n) {
            return Err(MoptError::config(
                "the structure of the chain has another number of coordinates than its images",
            ));
        }
        let registry = UpdaterRegistry::default();
        let built = (0..images)
            .map(|i| {
                let end = i == 0 || i == images - 1;
                let update = if end {
                    &settings.end_update
                } else {
                    &settings.inner_update
                };
                fresh_image(
                    &registry,
                    update,
                    settings.pairs,
                    rule.compatibility == Compatibility::MultiOptPy,
                    n,
                )
            })
            .collect::<MoptResult<Vec<_>>>()?;
        let mut built = Self {
            settings,
            images: built,
            template,
            rule,
            n,
            solve: None,
            ridge: Vec::new(),
            ridge_previous: Vec::new(),
        };
        if let Some(directory) = built.settings.directory.clone() {
            std::fs::create_dir_all(&directory)
                .map_err(|e| MoptError::Io(format!("{}: {e}", directory.display())))?;
            for (i, slot) in built.images.iter_mut().enumerate() {
                let hessian = slot.hessian.take()?;
                slot.hessian = Kept::File(directory.join(format!("tmp_hessian_{i}.npy")));
                slot.hessian.put(hessian)?;
            }
        }
        Ok(built)
    }

    /// Number of images.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether there are no images (never: a chain has at least two).
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Share of the quasi-Newton step in the step of an inner image.
    pub fn ratio(&self) -> f64 {
        self.settings.ratio
    }

    /// Whether the steps of the images count on the moves of their neighbours.
    pub fn together(&self) -> bool {
        self.settings.global.is_some() || self.settings.sweep
    }

    /// The settings of the step for the whole chain, when that is asked for.
    pub fn global(&self) -> Option<GlobalRfoSettings> {
        self.settings.global
    }

    /// Replace the Hessian of `image` (a model or an exact Hessian). The update keeps its
    /// stored pairs, as in MultiOptPy.
    pub fn set_hessian(&mut self, image: usize, hessian: Mat<f64>) -> MoptResult<()> {
        let slot = self
            .images
            .get_mut(image)
            .ok_or_else(|| MoptError::config(format!("no image {image} in the chain")))?;
        let n = self.n;
        if hessian.nrows() != n || hessian.ncols() != n {
            return Err(MoptError::config(format!(
                "the Hessian of image {image} is {} x {}, expected {n} x {n}",
                hessian.nrows(),
                hessian.ncols()
            )));
        }
        slot.hessian.put(hessian)
    }

    /// The Hessian of `image` as it stands.
    pub fn hessian(&self, image: usize) -> MoptResult<Mat<f64>> {
        self.images
            .get(image)
            .ok_or_else(|| MoptError::config(format!("no image {image} in the chain")))?
            .hessian
            .copy()
    }

    /// The chain got images added: `sources` says where every image of the new chain
    /// comes from. An image of the chain before keeps what it has (its Hessian, under
    /// its new number when the Hessians are in files, and the pairs of its update); a new
    /// image starts with a copy of the Hessian of the image it is like.
    pub fn insert_images(&mut self, sources: &[ImageSource]) -> MoptResult<()> {
        let before = self.images.len();
        let kept: Vec<usize> = sources
            .iter()
            .filter_map(|s| match s {
                ImageSource::Old(i) => Some(*i),
                ImageSource::New { .. } => None,
            })
            .collect();
        let in_order = kept.len() == before && kept.iter().enumerate().all(|(k, i)| k == *i);
        let known = sources.iter().all(|s| match s {
            ImageSource::Old(_) => true,
            ImageSource::New { like } => *like < before,
        });
        if !(in_order && known) {
            return Err(MoptError::config(
                "added images: the images of the chain before must all stay, in their order",
            ));
        }
        let legacy = self.rule.compatibility == Compatibility::MultiOptPy;
        let registry = UpdaterRegistry::default();
        let directory = self.settings.directory.clone();
        let file = |dir: &std::path::Path, name: String| dir.join(name);
        let moved = |from: &std::path::Path, to: &std::path::Path| {
            std::fs::rename(from, to)
                .map_err(|e| MoptError::Io(format!("{} -> {}: {e}", from.display(), to.display())))
        };
        let mut old: Vec<Option<ImageRfo>> = std::mem::take(&mut self.images)
            .into_iter()
            .map(Some)
            .collect();
        // With files: the Hessians of the chain before under names that the new numbers
        // do not meet.
        if let Some(dir) = &directory {
            for (i, slot) in old.iter_mut().enumerate() {
                if let Some(slot) = slot {
                    let to = file(dir, format!("tmp_hessian_before_{i}.npy"));
                    if let Kept::File(from) = &slot.hessian {
                        moved(from, &to)?;
                    }
                    slot.hessian = Kept::File(to);
                }
            }
        }
        let missing = || MoptError::config("added images: an image of the chain before is gone");
        let mut images: Vec<Option<ImageRfo>> = Vec::with_capacity(sources.len());
        // The new images first: their sources are all still there.
        for (j, source) in sources.iter().enumerate() {
            match source {
                ImageSource::Old(_) => images.push(None),
                ImageSource::New { like } => {
                    let hessian = old[*like].as_ref().ok_or_else(missing)?.hessian.copy()?;
                    let mut image = fresh_image(
                        &registry,
                        &self.settings.inner_update,
                        self.settings.pairs,
                        legacy,
                        self.n,
                    )?;
                    image.hessian = match &directory {
                        Some(dir) => {
                            let mut kept = Kept::File(file(dir, format!("tmp_hessian_{j}.npy")));
                            kept.put(hessian)?;
                            kept
                        }
                        None => Kept::Memory(hessian),
                    };
                    images.push(Some(image));
                }
            }
        }
        for (j, source) in sources.iter().enumerate() {
            if let ImageSource::Old(i) = source {
                let mut image = old[*i].take().ok_or_else(missing)?;
                if let Some(dir) = &directory {
                    let to = file(dir, format!("tmp_hessian_{j}.npy"));
                    if let Kept::File(from) = &image.hessian {
                        moved(from, &to)?;
                    }
                    image.hessian = Kept::File(to);
                }
                images[j] = Some(image);
            }
        }
        self.images = images
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(missing)?;
        Ok(())
    }

    /// The images `dropped` (sorted, each once) were taken out of the chain: what is
    /// kept for them goes, and the Hessians of the others get their new numbers when
    /// they are in files.
    pub fn remove_images(&mut self, dropped: &[usize]) -> MoptResult<()> {
        let before = self.images.len();
        if dropped.windows(2).any(|w| w[0] >= w[1]) || dropped.last().is_some_and(|l| *l >= before)
        {
            return Err(MoptError::config(
                "images taken out: sorted numbers of images of the chain, each once",
            ));
        }
        let directory = self.settings.directory.clone();
        let mut left = Vec::with_capacity(before - dropped.len());
        for (i, mut image) in std::mem::take(&mut self.images).into_iter().enumerate() {
            if dropped.contains(&i) {
                if let Kept::File(path) = &image.hessian {
                    // The file of an image that is gone is in the way of the others.
                    let _ = std::fs::remove_file(path);
                }
                continue;
            }
            let j = left.len();
            if let (Some(dir), Kept::File(from)) = (&directory, &image.hessian) {
                let to = dir.join(format!("tmp_hessian_{j}.npy"));
                if *from != to {
                    std::fs::rename(from, &to).map_err(|e| {
                        MoptError::Io(format!("{} -> {}: {e}", from.display(), to.display()))
                    })?;
                    image.hessian = Kept::File(to);
                }
            }
            left.push(image);
        }
        self.images = left;
        Ok(())
    }

    /// The transverse modes of the images of a chain on a ridge from their Hessians as they
    /// stand, with `directions` the directions of the chain at the images; they are used by
    /// the next [`Self::steps`].
    pub fn ridge_modes(
        &mut self,
        chain: &[Col<f64>],
        directions: &[Col<f64>],
        ridge: &ChainRidge,
    ) -> MoptResult<RidgeModes> {
        let count = self.images.len();
        if chain.len() != count || directions.len() != count {
            return Err(MoptError::config(format!(
                "the transverse modes were set up for {count} images"
            )));
        }
        if ridge.end_modes.iter().any(|m| m.nrows() != self.n) {
            return Err(MoptError::config(
                "the modes of the ends differ in size from the images",
            ));
        }
        if self.ridge_previous.len() != count {
            // The modes of the ends, interpolated along the chain.
            let [a, b] = &ridge.end_modes;
            let sign = if col_dot(a, b) < 0.0 { -1.0 } else { 1.0 };
            self.ridge_previous = (0..count)
                .map(|i| {
                    let s = i as f64 / (count - 1) as f64;
                    Some(a * (1.0 - s) + b * (sign * s))
                })
                .collect();
        }
        let mut out = RidgeModes {
            modes: vec![None; count],
            curvatures: vec![f64::NAN; count],
            switches: Vec::new(),
        };
        let n = self.n;
        for i in 1..count - 1 {
            let surface = self.images[i].hessian.copy()?;
            let frame = Along::new(&directions[i], n);
            let mut model = frame.hessian(&surface, false, &[]);
            let scale = (0..n).map(|k| model[(k, k)].abs()).fold(1.0, f64::max);
            let shift = 10.0 * scale * n as f64 + 1.0;
            // The direction of the chain and the rigid-body motions above every curvature.
            model += (&frame.units * frame.units.transpose()) * shift;
            let basis = match &self.template {
                Some(template) => Some(build_tr_rot_basis(
                    &template.with_flat_coords_bohr(&chain[i])?,
                    false,
                )),
                None => None,
            };
            if let Some(d) = basis.as_ref().filter(|d| d.ncols() > 0) {
                model = project_out_basis_mat(d, &model);
                model += (d * d.transpose()) * shift;
            }
            let (values, vectors) = symmetric_eigen(&model)?;
            let previous = self.ridge_previous[i].clone().map(|p| {
                let p = match basis.as_ref() {
                    Some(d) => project_out_basis_vec(d, &p),
                    None => p,
                };
                &p - frame.part(&p)
            });
            let overlap = |j: usize| {
                previous.as_ref().map_or(1.0, |p| {
                    col_dot(&vectors.col(j).to_owned(), p) / col_norm(p).max(1e-300)
                })
            };
            let chosen = match ridge.transverse {
                TransverseMode::Lowest => 0,
                TransverseMode::Continued if n > 1 => {
                    if overlap(1).abs() > overlap(0).abs() {
                        1
                    } else {
                        0
                    }
                }
                TransverseMode::Continued => 0,
            };
            let mut u = vectors.col(chosen).to_owned();
            let o = overlap(chosen);
            if o < 0.0 {
                u = -u;
            }
            if chosen != 0 || o.abs() < ridge.switch_overlap {
                out.switches.push(i);
            }
            out.curvatures[i] = values[chosen];
            out.modes[i] = Some(u.clone());
            self.ridge_previous[i] = Some(u);
        }
        self.ridge = out.modes.clone();
        Ok(out)
    }

    /// Turn what is kept for `image` with the image: its Hessian, the point and the
    /// gradient of its last iteration, and the pairs of its Hessian update.
    pub fn move_image(&mut self, image: usize, motion: &RigidMotion) -> MoptResult<()> {
        for kept in [&mut self.ridge, &mut self.ridge_previous] {
            if let Some(Some(u)) = kept.get_mut(image) {
                *u = motion.vector(u);
            }
        }
        if let Some(slot) = self.images.get_mut(image) {
            let hessian = slot.hessian.take()?;
            slot.hessian.put(motion.matrix(&hessian))?;
            slot.last = slot
                .last
                .take()
                .map(|(x, g)| (motion.point(&x), motion.vector(&g)));
            slot.updater.map_pairs(&|v| motion.vector(v));
        }
        Ok(())
    }

    /// The quasi-Newton step of every image (the displacement to add), from the chain, its
    /// energies and full gradients, and what the force rule found `along` the chain.
    pub fn steps(
        &mut self,
        chain: &[Col<f64>],
        energies: &[f64],
        gradients: &[Col<f64>],
        along: AlongChain<'_>,
    ) -> MoptResult<Vec<Col<f64>>> {
        let count = self.images.len();
        if chain.len() != count
            || energies.len() != count
            || gradients.len() != count
            || along.removed.len() != count
            || along.directions.len() != count
        {
            return Err(MoptError::config(format!(
                "the quasi-Newton steps were set up for {count} images"
            )));
        }
        let legacy = self.rule.compatibility == Compatibility::MultiOptPy;
        // From the highest image down: the neighbour uphill of an image has its step by
        // the time the image gets its own (see `ChainRfoSettings::sweep`).
        let mut order: Vec<usize> = (0..count).collect();
        if self.settings.sweep && !legacy {
            order.sort_by(|a, b| energies[*b].total_cmp(&energies[*a]));
        }
        let mut steps: Vec<Option<Col<f64>>> = vec![None; count];
        for i in order {
            let end = i == 0 || i == count - 1;
            let transverse = if end {
                None
            } else {
                self.ridge.get(i).cloned().flatten()
            };
            let slot = &mut self.images[i];
            // The Hessian of the surface at this image, in memory while it is worked on.
            let mut surface = slot.hessian.take()?;
            if !end && legacy {
                // MultiOptPy's direction is minus the removed component; the sign matters
                // for the slope of the middle image in the fit.
                let direction = -&along.removed[i];
                let gamma = path_curvature(
                    [&chain[i - 1], &chain[i], &chain[i + 1]],
                    [energies[i - 1], energies[i], energies[i + 1]],
                    [&gradients[i - 1], &gradients[i], &gradients[i + 1]],
                    &direction,
                );
                let n = direction.nrows();
                for r in 0..n {
                    for c in 0..n {
                        surface[(r, c)] += gamma * direction[r] * direction[c];
                    }
                }
            }
            let previous = slot.last.take().map(|(x, g)| PreviousPoint {
                energy: Quantity::new(0.0),
                gradient: g.clone(),
                raw_gradient: g,
                step: &chain[i] - &x,
            });
            let basis = match &self.template {
                Some(template) => Some(build_tr_rot_basis(
                    &template.with_flat_coords_bohr(&chain[i])?,
                    false,
                )),
                None => None,
            };
            if let Some(prev) = &previous {
                let y = &gradients[i] - &prev.raw_gradient;
                // The update puts curvature along the rigid motions of the image; the kept
                // Hessian has none there, updated or not.
                apply_update(slot.updater.as_mut(), &mut surface, &prev.step, &y, false)?;
                if self.settings.project_updates {
                    project_out_rigid(&mut surface, basis.as_ref());
                }
            }
            // The model across the chain (see the module documentation).
            let climbing = along.climbing.contains(&i);
            let across = if end || legacy {
                None
            } else {
                let n = chain[i].nrows();
                let block = if climbing { n } else { self.rule.block };
                let frame = Along::new(&along.directions[i], block);
                let part = frame.part(&gradients[i]);
                let gradient = if climbing {
                    &gradients[i] - 2.0 * &part
                } else {
                    &gradients[i] - &part
                };
                // The direction points to the next image where the energy rises along
                // the chain, from the previous one where it falls.
                let rising = energies[i - 1] < energies[i] && energies[i] < energies[i + 1];
                let falling = energies[i - 1] > energies[i] && energies[i] > energies[i + 1];
                let stiffening = if climbing || !(rising || falling) {
                    Vec::new()
                } else {
                    frame.stiffening(&gradients[i], rising)
                };
                // The step of the neighbour uphill turns the direction of this image:
                // its force changes by the stiffening times that step across the chain.
                let uphill = if rising { i + 1 } else { i - 1 };
                let gradient = match steps.get(uphill).and_then(Option::as_ref) {
                    Some(step) if self.settings.sweep && !stiffening.is_empty() => {
                        gradient - frame.stiffened(&stiffening, step)
                    }
                    _ => gradient,
                };
                let hessian = frame.hessian(&surface, climbing, &stiffening);
                Some((frame, gradient, hessian))
            };
            // On a ridge: the image function, reversed along the transverse mode.
            let reversed = match (&across, &transverse) {
                (Some((_, gradient, hessian)), Some(u)) => {
                    let along = col_dot(gradient, u);
                    let curvature = col_dot(u, &(hessian * u));
                    let mut model = hessian.clone();
                    for r in 0..model.nrows() {
                        for c in 0..model.ncols() {
                            model[(r, c)] -= 2.0 * curvature * u[r] * u[c];
                        }
                    }
                    Some((gradient - u * (2.0 * along), model))
                }
                _ => None,
            };
            let (gradient, hessian) = match (&reversed, &across) {
                (Some((gradient, hessian)), _) => (gradient, hessian),
                (None, Some((_, gradient, hessian))) => (gradient, hessian),
                (None, None) => (&gradients[i], &surface),
            };
            let input = StepInput {
                iteration: slot.iteration,
                cartesian: &chain[i],
                energy: Quantity::new(0.0),
                gradient,
                raw_gradient: gradient,
                previous: previous.as_ref(),
                hessian,
                bias_hessian: None,
                rigid_body_basis: basis.as_ref(),
                project_updates: self.settings.project_updates,
                saddle_order: 0,
                system: None,
                cartesian_coordinates: true,
                frame: None,
                cartesian_gradient: Some(gradient),
            };
            let mut step = slot
                .generator
                .propose(&input)
                .context(format!("quasi-Newton step of image {i}"))?;
            if let Some((frame, _, _)) = &across {
                if !climbing {
                    step -= frame.part(&step);
                }
            }
            let limit = if end {
                self.settings.end_step
            } else {
                self.settings.inner_step
            };
            let length = col_norm(&step);
            if length > 1e-8 {
                step *= limit.min(length) / length;
            }
            slot.hessian.put(surface)?;
            slot.last = Some((chain[i].clone(), gradients[i].clone()));
            slot.iteration += 1;
            steps[i] = Some(step);
        }
        Ok(steps
            .into_iter()
            .map(|step| step.unwrap_or_else(|| Col::zeros(self.n)))
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "rfo_tests.rs"]
mod tests;
