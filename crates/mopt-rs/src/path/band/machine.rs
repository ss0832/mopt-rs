//! The relaxation of a chain as a state machine: it asks for the energies and gradients of
//! its images and is told them, and never evaluates anything itself.
//!
//! One iteration, in MultiOptPy's order:
//!
//! 1. forces of the images ([`UpwindProjection`]);
//! 2. a step along them (the rule MultiOptPy calls FIRE, or steepest descent from a given
//!    iteration on), limited so that no image passes its neighbours; images that are
//!    converged may be held. With Hessians of the images ([`ChainSettings::rfo`]) an inner
//!    image moves by `(1 - r)` times this step plus `r` times its quasi-Newton step
//!    ([`ChainRfo`]: a step across the chain; under `Compatibility::MultiOptPy` a
//!    minimization with the full gradient, the step rule along the forces starting anew
//!    in every iteration), and an end image by its quasi-Newton step alone;
//! 3. fixed end images go back to where they started, fixed components of every image to
//!    their values in the first chain;
//! 4. structures are centred and turned onto their neighbours in sequence (chains of atoms
//!    without fixed components), and what is kept for an image (its Hessian, its last
//!    gradient and step) is turned with it;
//! 5. every few iterations the images are put back along the chain (equal spacing, or
//!    towards the predicted maxima of the energy profile); images that climb stay where
//!    they are, and the others are spread between them.
//!
//! MultiOptPy runs a given number of iterations and has no test for the end. A tolerance
//! on the forces of the images that move (the inner images, and the end images that are
//! not fixed, which are minimized) ends the relaxation earlier here if it is set.
//!
//! The caller may replace the chain between iterations ([`ChainRelaxation::set_chain`]),
//! which is how extra processing after an iteration changes the path.
//!
//! MultiOptPy: `neb.py:NEB.execute`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::align::{align_chain_in_sequence, align_chain_onto_previous, centre_chain_at};
use super::diis::{ChainDiis, ChainDiisSettings};
use super::fire::{ChainFire, ChainFireSettings};
use super::force::UpwindProjection;
use super::images::{AddedImages, FailedImage, ImageFailure};
use super::peaks::PeakMoves;
use super::redistribute::{equal_spacing_between, gathered_at_predicted_maxima, Curve, Gathering};
use super::rfo::{AlongChain, ChainRfo, ChainRfoSettings, ChainRidge};
use super::springs::EnergyWeightedSprings;
use super::trust::{
    hold_converged, limit_steps, limit_steps_of, steepest_steps, FixedEnds, ImageCriteria,
};
use super::{check_chain, path_lengths, strict_maxima, strict_minima, ChainKind, Compatibility};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_max_abs, col_norm};

/// Settings of a [`ChainRelaxation`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChainSettings {
    /// What the points of the chain are.
    pub kind: ChainKind,
    /// Coordinates per block of the projection (`None`: the whole point, that is, the
    /// component of the gradient along the one direction of the chain is removed; under
    /// `Compatibility::MultiOptPy` an atom for chains of atoms, MultiOptPy's rule, which
    /// `Some(3)` chooses in the default form too). With one direction a chain of
    /// structures converged in half the iterations or fewer in the cases measured
    /// (`docs/source/migration_paths.md`, "Chain relaxation").
    pub block: Option<usize>,
    /// Iterations to run.
    pub max_iterations: usize,
    /// End images that stay where they are.
    pub ends: FixedEnds,
    /// Components of every image that keep their values of the first chain (the three
    /// coordinates of fixed atoms).
    pub fixed_components: Vec<usize>,
    /// Centre the structures and turn them onto each other after every step (chains of
    /// atoms without fixed components; off when positions or orientations are held by
    /// constraints).
    pub align: bool,
    /// For structures in a periodic cell, which does not turn with them: aligning shifts
    /// every structure along these orthonormal directions (the periodic ones,
    /// `align::free_shifts`) to the centroid of the first image of the first chain and
    /// turns nothing.
    pub periodic_shifts: Option<Vec<[f64; 3]>>,
    /// Springs between the images (the nudged elastic band with energy-weighted
    /// springs); none: the images are kept apart by the redistributions alone.
    pub springs: Option<EnergyWeightedSprings>,
    /// Images at maxima climb in iterations after this one.
    pub climb_after: Option<usize>,
    /// Steepest descent from this iteration on.
    pub steepest_from: Option<usize>,
    /// The step rule before that.
    pub fire: ChainFireSettings,
    /// Images that are converged are held.
    pub hold_converged: Option<ImageCriteria>,
    /// Equal spacing every this many iterations.
    pub equal_spacing_every: Option<usize>,
    /// Images towards the predicted maxima every this many iterations.
    pub predicted_maxima_every: Option<usize>,
    /// With that, the other images gathered around the maxima (`None`: spread evenly
    /// between the ends and the maxima, as always under `Compatibility::MultiOptPy`).
    pub gathering: Option<Gathering>,
    /// Quasi-Newton steps of the images mixed into the steps along the forces (`None`:
    /// the steps along the forces alone).
    pub rfo: Option<ChainRfoSettings>,
    /// With quasi-Newton steps: Hessians of the images are asked for in the first
    /// iteration and then every this many iterations (`None`: in the first only).
    pub hessian_every: Option<usize>,
    /// The relaxation ends when the largest force component of the inner images is below
    /// this (Hartree/Bohr; `None`: all iterations are run).
    pub force_tolerance: Option<f64>,
    /// Images are added on both sides of the images at maxima of the profile, to
    /// resolve the top of the barrier ([`AddedImages`]; `None`: the chain keeps its
    /// images).
    pub added: Option<AddedImages>,
    /// What is done with an image whose calculation failed.
    pub on_failure: FailedImage,
    /// Extrapolation over the chains of the last iterations ([`ChainDiis`]; `None`: the
    /// steps as the step rule proposes them). Not in MultiOptPy's form.
    pub diis: Option<ChainDiisSettings>,
    /// Where the direction of the chain at an image comes from.
    pub tangent: ChainTangent,
    /// The chain joins two transition states over an index-2 saddle: every inner image is
    /// maximized along its transverse mode ([`ChainRidge`]; needs the Hessians of the
    /// images, [`Self::rfo`], and the projection of whole images).
    pub ridge: Option<ChainRidge>,
    /// Form of the formulas where MultiOptPy's is inconsistent.
    pub compatibility: Compatibility,
}

/// The direction of the chain at an image, along which its gradient is removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChainTangent {
    /// Towards the higher neighbour (between the two at an extremum of the profile).
    #[default]
    HigherNeighbour,
    /// Carried outwards from the highest image by arcs and parabolas
    /// ([`super::tangents`]; the quadratic string form `-qsmv2`), while the profile has
    /// one maximum; towards the higher neighbour otherwise. For whole images only.
    CarriedFromTop,
}

impl Default for ChainSettings {
    fn default() -> Self {
        Self {
            kind: ChainKind::Atoms,
            block: None,
            max_iterations: 20,
            ends: FixedEnds::default(),
            fixed_components: Vec::new(),
            align: true,
            periodic_shifts: None,
            springs: None,
            climb_after: None,
            steepest_from: None,
            fire: ChainFireSettings::default(),
            hold_converged: None,
            equal_spacing_every: None,
            predicted_maxima_every: None,
            gathering: None,
            rfo: None,
            hessian_every: None,
            force_tolerance: None,
            added: None,
            on_failure: FailedImage::default(),
            diis: None,
            tangent: ChainTangent::default(),
            ridge: None,
            compatibility: Compatibility::Consistent,
        }
    }
}

impl ChainSettings {
    /// The way the chain is relaxed in the articles of MultiOptPy's author (`nebmain -ns 20
    /// -modelhess o1numhess -adpred 2`): 20 iterations, Hessians of the images at the
    /// start (and every 50 iterations), half a step along the forces and half a
    /// quasi-Newton step, and the images put towards the predicted maxima of the energy
    /// profile every second iteration.
    pub fn of_the_articles() -> Self {
        Self {
            max_iterations: 20,
            rfo: Some(ChainRfoSettings::default()),
            hessian_every: Some(50),
            predicted_maxima_every: Some(2),
            gathering: Some(Gathering {
                weight: 1.0,
                width: 1.0,
            }),
            ..Self::default()
        }
    }
}

/// Energy and gradient of one image.
#[derive(Debug, Clone)]
pub struct ImageValue {
    /// Energy.
    pub energy: f64,
    /// Gradient.
    pub gradient: Col<f64>,
    /// Hessian, when the relaxation asked for the Hessians of the images (an image without
    /// one keeps the Hessian it has).
    pub hessian: Option<Mat<f64>>,
}

impl ImageValue {
    /// Energy and gradient without a Hessian.
    pub fn new(energy: f64, gradient: Col<f64>) -> Self {
        Self {
            energy,
            gradient,
            hessian: None,
        }
    }
}

/// The images a relaxation wants evaluated.
#[derive(Debug, Clone)]
pub struct ChainRequest {
    /// The images, in the order of the values to tell.
    pub images: Vec<Col<f64>>,
    /// Whether Hessians of the images are wanted in this iteration (model Hessians or exact
    /// ones, as the caller has them).
    pub need_hessian: bool,
}

/// Why a relaxation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainEnd {
    /// All iterations were run.
    IterationLimit,
    /// The forces of the images that move are below the tolerance.
    Converged,
}

/// What a relaxation asks for.
#[derive(Debug, Clone)]
pub enum ChainAction {
    /// Energies and gradients (and Hessians, if asked for) of these images.
    Evaluate(ChainRequest),
    /// The relaxation is over; the chain is the last one evaluated.
    Finished(ChainEnd),
}

/// What happened in one iteration.
#[derive(Debug, Clone)]
pub struct ChainReport {
    /// Iteration (from zero).
    pub iteration: usize,
    /// The chain that was evaluated.
    pub chain: Vec<Col<f64>>,
    /// Energies of its images (those the chain is relaxed on: with the bias potentials).
    pub energies: Vec<f64>,
    /// Energies of its images without the bias potentials, when the surface has any
    /// (filled in by the driver; empty otherwise; `NaN` for a value the surface did not
    /// calculate, such as a prediction of a model).
    pub bare_energies: Vec<f64>,
    /// Length of the chain up to each image.
    pub lengths: Vec<f64>,
    /// Root mean square of the force on each image.
    pub force_rms: Vec<f64>,
    /// Largest component of the force on each image.
    pub force_max: Vec<f64>,
    /// Root mean square of the gradient of each image.
    pub gradient_rms: Vec<f64>,
    /// Cosine between force and gradient of each image.
    pub cosines: Vec<f64>,
    /// Distances of the inner images to their previous and next neighbours.
    pub to_previous: Vec<f64>,
    /// See `to_previous`.
    pub to_next: Vec<f64>,
    /// Inner images at maxima and minima of the energy profile.
    pub maxima: Vec<usize>,
    /// See `maxima`.
    pub minima: Vec<usize>,
    /// Images that climbed.
    pub climbing: Vec<usize>,
    /// Images held because their step was against their force.
    pub held: Vec<usize>,
    /// Images held because they are converged.
    pub converged: Vec<usize>,
    /// Whether the images were put back along the chain after the step.
    pub redistributed: bool,
    /// Images that were added to the chain at the end of this iteration
    /// ([`AddedImages`]).
    pub added: usize,
    /// Images whose calculation failed in this iteration, and what became of them (set
    /// by the driver).
    pub failures: Vec<ImageFailure>,
    /// What the surface said while the images were evaluated: warnings of the calculator,
    /// the output of a program, `<S**2>` (filled in by the driver; a text may have
    /// several lines, and one that begins with `Warning:` is a warning).
    pub notes: Vec<String>,
    /// Largest force component of the images that move (the inner ones, and the ends
    /// that are not fixed): what is compared with the tolerance.
    pub largest_force: f64,
    /// Iterations so far whose steps were extrapolated from the chains before
    /// ([`ChainDiis`]).
    pub extrapolated: usize,
    /// Time step of the step rule after this iteration.
    pub time_step: f64,
    /// On a ridge: the curvature along the transverse mode of every image (`NaN` at the
    /// ends; empty otherwise).
    pub ridge_curvatures: Vec<f64>,
    /// On a ridge: the images whose transverse mode switched in this iteration.
    pub ridge_switches: Vec<usize>,
}

/// The quasi-Newton settings of a chain of structures: fixed atoms pin the frame, and their
/// constraints take the place of the projection of the rigid motions out of the updated
/// Hessians (as in the optimizers and the IRC).
fn structure_rfo(settings: &ChainSettings) -> Option<ChainRfoSettings> {
    settings.rfo.clone().map(|mut rfo| {
        rfo.project_updates &= settings.fixed_components.is_empty();
        rfo
    })
}

/// The relaxation of a chain of images.
pub struct ChainRelaxation {
    pub(super) settings: ChainSettings,
    rule: UpwindProjection,
    pub(super) chain: Vec<Col<f64>>,
    pub(super) first: Vec<Col<f64>>,
    pub(super) fire: ChainFire,
    pub(super) previous_forces: Option<Vec<Col<f64>>>,
    /// The steps of the last iteration (`None` after the images were put back along the
    /// chain or replaced).
    pub(super) last_steps: Option<Vec<Col<f64>>>,
    pub(super) rfo: Option<ChainRfo>,
    /// The last moves of the images at maxima in the redistributions.
    pub(super) peak_moves: PeakMoves,
    /// Iterations since images were added (MultiOptPy's count).
    pub(super) since_added: usize,
    pub(super) diis: Option<ChainDiis>,
    template: Option<MolecularSystem>,
    hessian_due: bool,
    iteration: usize,
    pub(super) asked: bool,
    /// The chain as it was evaluated in the iteration before.
    pub(super) evaluated: Option<Vec<Col<f64>>>,
    end: Option<ChainEnd>,
}

impl ChainRelaxation {
    /// A relaxation of `chain` (at least two images of the same length).
    pub fn new(settings: ChainSettings, chain: Vec<Col<f64>>) -> MoptResult<Self> {
        let n = chain.first().map_or(0, Col::nrows);
        // Structures have three coordinates per atom whatever the blocks are.
        check_chain(&chain, settings.kind.block(n))?;
        let block = settings
            .block
            .unwrap_or_else(|| match settings.compatibility {
                Compatibility::Consistent => n,
                Compatibility::MultiOptPy => settings.kind.block(n),
            });
        check_chain(&chain, block)?;
        if let Some(&k) = settings.fixed_components.iter().find(|&&k| k >= n) {
            return Err(MoptError::config(format!(
                "fixed component {k} of images with {n} components"
            )));
        }
        if settings.ridge.is_some()
            && (settings.rfo.as_ref().is_none_or(|r| r.global.is_some())
                || block != n
                || settings.compatibility == Compatibility::MultiOptPy)
        {
            return Err(MoptError::config(
                "a chain on a ridge needs the Hessians of its images (quasi-Newton steps of \
                 every image, not one step for the whole chain) and the projection of whole \
                 images, in the form of mopt-rs",
            ));
        }
        if settings.tangent == ChainTangent::CarriedFromTop
            && (block != n || settings.compatibility == Compatibility::MultiOptPy)
        {
            return Err(MoptError::config(
                "the tangents carried from the highest image are those of whole images in \
                 the form of mopt-rs (not with the projection atom by atom or MultiOptPy's form)",
            ));
        }
        let rule = UpwindProjection {
            block,
            climb_after: settings.climb_after,
            compatibility: settings.compatibility,
        };
        let fire = ChainFire::new(settings.fire);
        let rfo = match &settings.rfo {
            Some(rfo) => Some(ChainRfo::new(rfo.clone(), chain.len(), n, None, rule)?),
            None => None,
        };
        let diis = match (settings.diis, settings.compatibility) {
            (Some(diis), Compatibility::Consistent) => Some(ChainDiis::new(diis)),
            _ => None,
        };
        Ok(Self {
            hessian_due: rfo.is_some(),
            diis,
            evaluated: None,
            peak_moves: PeakMoves::new(),
            since_added: 0,
            settings,
            rule,
            first: chain.clone(),
            chain,
            fire,
            previous_forces: None,
            last_steps: None,
            rfo,
            template: None,
            iteration: 0,
            asked: false,
            end: None,
        })
    }

    /// For chains of structures with quasi-Newton steps: the atoms of the structures, so
    /// that translations and rotations are taken out of the gradients and Hessians of the
    /// images.
    pub fn with_structure(mut self, template: MolecularSystem) -> MoptResult<Self> {
        let n = self.chain[0].nrows();
        if template.dof() != n {
            return Err(MoptError::config(format!(
                "a structure of {} coordinates for images of {n}",
                template.dof()
            )));
        }
        if let Some(rfo) = structure_rfo(&self.settings) {
            self.rfo = Some(ChainRfo::new(
                rfo,
                self.chain.len(),
                n,
                Some(template.clone()),
                self.rule,
            )?);
        }
        self.template = Some(template);
        Ok(self)
    }

    /// The current chain.
    pub fn chain(&self) -> &[Col<f64>] {
        &self.chain
    }

    /// Iterations done so far.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Why the relaxation ended (`None` while it runs).
    pub fn end(&self) -> Option<ChainEnd> {
        self.end
    }

    /// Replace the chain between two iterations (the same images, moved by the caller, or
    /// another number of images). With another number of images the history of the step
    /// rule and the first chain that fixed images go back to start anew.
    pub fn set_chain(&mut self, chain: Vec<Col<f64>>) -> MoptResult<()> {
        if self.asked {
            return Err(MoptError::config(
                "the chain cannot be replaced between ask() and tell()",
            ));
        }
        let n = check_chain(&chain, self.rule.block)?;
        if n != self.chain[0].nrows() {
            return Err(MoptError::config(
                "the new chain has images of another number of components",
            ));
        }
        if chain.len() != self.chain.len() {
            self.first = chain.clone();
            self.previous_forces = None;
            self.fire = ChainFire::new(self.settings.fire);
            if let Some(rfo) = structure_rfo(&self.settings) {
                // New images have no Hessians yet.
                self.rfo = Some(ChainRfo::new(
                    rfo,
                    chain.len(),
                    n,
                    self.template.clone(),
                    self.rule,
                )?);
                self.hessian_due = true;
            }
        }
        if chain.len() != self.chain.len() {
            self.evaluated = None;
        }
        self.chain = chain;
        self.last_steps = None;
        self.peak_moves.clear();
        Ok(())
    }

    /// Whether the iteration that is asked for next is the last one of the settings.
    pub fn is_last_iteration(&self) -> bool {
        self.iteration + 1 >= self.settings.max_iterations
    }

    /// A relaxation that ended because its forces were below the tolerance goes on: for
    /// values that were not those of the surface itself (predictions), which are replaced
    /// by calculated ones in the next iteration at the same chain.
    pub fn reopen(&mut self) {
        if self.end == Some(ChainEnd::Converged) {
            self.end = None;
        }
    }

    /// What is needed next.
    pub fn ask(&mut self) -> ChainAction {
        if let Some(end) = self.end {
            return ChainAction::Finished(end);
        }
        if self.iteration >= self.settings.max_iterations {
            self.end = Some(ChainEnd::IterationLimit);
            return ChainAction::Finished(ChainEnd::IterationLimit);
        }
        self.asked = true;
        ChainAction::Evaluate(ChainRequest {
            images: self.chain.clone(),
            need_hessian: self.rfo.is_some() && self.hessian_due,
        })
    }

    /// The values of the images of the last [`Self::ask`], in its order.
    pub fn tell(&mut self, values: Vec<ImageValue>) -> MoptResult<ChainReport> {
        if !self.asked {
            return Err(MoptError::config("tell() without a preceding ask()"));
        }
        let images = self.chain.len();
        let n = self.chain[0].nrows();
        if values.len() != images || values.iter().any(|v| v.gradient.nrows() != n) {
            return Err(MoptError::config(format!(
                "{images} images of {n} components need as many energies and gradients"
            )));
        }
        if values
            .iter()
            .any(|v| !v.energy.is_finite() || v.gradient.iter().any(|g| !g.is_finite()))
        {
            return Err(MoptError::calc(
                "an image of the chain has a non-finite energy or gradient",
            ));
        }
        self.asked = false;
        let evaluated = self.chain.clone();
        let iteration = self.iteration;
        let energies: Vec<f64> = values.iter().map(|v| v.energy).collect();
        let mut gradients = Vec::with_capacity(images);
        let fixed = &self.settings.fixed_components;
        for (i, value) in values.into_iter().enumerate() {
            if let (Some(rfo), Some(mut hessian)) = (self.rfo.as_mut(), value.hessian) {
                // A fixed component does not move: its rows and columns couple nothing
                // into the quasi-Newton steps of the others.
                for &k in fixed {
                    for j in 0..n {
                        hessian[(k, j)] = 0.0;
                        hessian[(j, k)] = 0.0;
                    }
                    hessian[(k, k)] = 1.0;
                }
                rfo.set_hessian(i, hessian)?;
            }
            // The gradient on a fixed component is a constraint force, which no step can
            // lower: it is left out of everything derived from the gradients (forces,
            // tangents, the test of the end, extrapolations, the files).
            let mut gradient = value.gradient;
            for &k in fixed {
                gradient[k] = 0.0;
            }
            gradients.push(gradient);
        }
        // Root mean squares over the components that move.
        let free = n.saturating_sub(fixed.len()).max(1) as f64;
        let rms = |v: &Col<f64>| libm::sqrt(col_dot(v, v) / free);
        self.hessian_due = false;
        // The tangents carried from the top assume one maximum of the profile.
        let carried = match self.settings.tangent {
            ChainTangent::CarriedFromTop if strict_maxima(&energies).len() <= 1 => {
                super::tangents::carried_from_the_top(&self.chain, &energies).map(|tangents| {
                    super::tangents::with_the_distance_uphill(tangents, &self.chain, &energies)
                })
            }
            _ => None,
        };
        let found = self.rule.forces_along(
            &self.chain,
            &energies,
            &gradients,
            iteration,
            carried.as_deref(),
        )?;
        let mut forces = found.forces;
        // On a ridge: every inner image is maximized along its transverse mode.
        let (mut ridge_curvatures, mut ridge_switches) = (Vec::new(), Vec::new());
        if let (Some(ridge), Some(rfo)) = (&self.settings.ridge, self.rfo.as_mut()) {
            let modes = rfo.ridge_modes(&self.chain, &found.directions, ridge)?;
            for (force, (u, g)) in forces.iter_mut().zip(modes.modes.iter().zip(&gradients)) {
                if let Some(u) = u {
                    *force += u * (2.0 * col_dot(g, u));
                }
            }
            ridge_curvatures = modes.curvatures;
            ridge_switches = modes.switches;
        }
        // Springs along the chain, where they were asked for.
        let pulls = self.settings.springs.map(|springs| {
            springs.pulls(
                &self.chain,
                &energies,
                &found.directions,
                &found.climbing,
                self.settings.kind,
                self.settings.compatibility,
            )
        });
        if let Some(pulls) = &pulls {
            for (force, pull) in forces.iter_mut().zip(pulls) {
                *force += &pull.force;
            }
        }
        // Fixed components of images that differ from image to image give the tangents a
        // part along them: the forces have none.
        for force in &mut forces {
            for &k in &self.settings.fixed_components {
                force[k] = 0.0;
            }
        }
        let along = AlongChain {
            removed: &found.removed,
            directions: &found.directions,
            climbing: &found.climbing,
        };

        let mut report = ChainReport {
            iteration,
            chain: self.chain.clone(),
            lengths: path_lengths(&self.chain, self.settings.kind),
            force_rms: forces.iter().map(rms).collect(),
            force_max: forces.iter().map(col_max_abs).collect(),
            gradient_rms: gradients.iter().map(rms).collect(),
            cosines: forces
                .iter()
                .zip(&gradients)
                .map(|(f, g)| {
                    let norms = col_norm(f) * col_norm(g);
                    if norms > 1e-20 {
                        col_dot(f, g) / norms
                    } else {
                        0.0
                    }
                })
                .collect(),
            to_previous: Vec::new(),
            to_next: Vec::new(),
            maxima: strict_maxima(&energies),
            minima: strict_minima(&energies),
            climbing: found.climbing.clone(),
            held: Vec::new(),
            converged: Vec::new(),
            redistributed: false,
            added: 0,
            failures: Vec::new(),
            notes: Vec::new(),
            largest_force: 0.0,
            extrapolated: 0,
            time_step: self.fire.time_step(),
            ridge_curvatures: std::mem::take(&mut ridge_curvatures),
            ridge_switches: std::mem::take(&mut ridge_switches),
            energies,
            bare_energies: Vec::new(),
        };
        // The images that move: the inner ones, and the ends that are not fixed.
        let ends = self.settings.ends;
        let moving_force = report
            .force_max
            .iter()
            .enumerate()
            .filter(|(i, _)| !((*i == 0 && ends.first) || (*i == images - 1 && ends.last)))
            .fold(0.0_f64, |m, (_, v)| m.max(*v));
        report.largest_force = moving_force;
        if self
            .settings
            .force_tolerance
            .is_some_and(|tolerance| images > 2 && moving_force < tolerance)
        {
            self.end = Some(ChainEnd::Converged);
            self.iteration += 1;
            return Ok(report);
        }

        let compatibility = self.settings.compatibility;
        let mut steps = match (&mut self.rfo, self.settings.steepest_from) {
            (Some(rfo), _) if compatibility == Compatibility::MultiOptPy => {
                // The quasi-Newton steps pass the limits of the chain as vectors that are
                // subtracted, with the gradients in the place of the forces (MultiOptPy).
                let newton = rfo.steps(&self.chain, &report.energies, &gradients, along)?;
                let subtracted: Vec<Col<f64>> = newton.iter().map(|d| -d).collect();
                let newton =
                    limit_steps(&self.chain, &gradients, &subtracted, ends, compatibility).steps;
                let mut fire = ChainFire::new(self.settings.fire);
                let proposed = fire.displacements(&forces, self.previous_forces.as_deref());
                let limited = limit_steps(&self.chain, &forces, &proposed, ends, compatibility);
                report.to_previous = limited.to_previous;
                report.to_next = limited.to_next;
                report.held = limited.held;
                report.time_step = fire.time_step();
                let ratio = rfo.ratio();
                (0..images)
                    .map(|i| {
                        if i == 0 || i == images - 1 {
                            -&newton[i]
                        } else {
                            (1.0 - ratio) * &limited.steps[i] - ratio * &newton[i]
                        }
                    })
                    .collect()
            }
            (Some(rfo), _) => {
                // Both parts of the step go across the chain; the step rule along the
                // forces keeps its time step, bounded by the curvature it meets.
                let mut newton = match rfo.global() {
                    Some(global) => rfo.global_steps(
                        &self.chain,
                        &report.energies,
                        &gradients,
                        along,
                        ends,
                        &global,
                    )?,
                    None => rfo.steps(&self.chain, &report.energies, &gradients, along)?,
                };
                if let Some(pulls) = &pulls {
                    // The quasi-Newton steps go across the chain; along it the springs
                    // are brought to balance.
                    for (step, pull) in newton.iter_mut().zip(pulls) {
                        *step += &pull.newton;
                    }
                }
                let together = rfo.together();
                let newton = limit_steps_of(
                    &self.chain,
                    &forces,
                    &newton,
                    ends,
                    compatibility,
                    !together,
                )
                .steps;
                if let (Some(before), Some(last)) = (&self.previous_forces, &self.last_steps) {
                    if before.len() == images && last.len() == images {
                        self.fire.observe_curvature(&forces, before, last);
                    }
                }
                let proposed = self
                    .fire
                    .displacements(&forces, self.previous_forces.as_deref());
                let limited = limit_steps(&self.chain, &forces, &proposed, ends, compatibility);
                report.to_previous = limited.to_previous;
                report.to_next = limited.to_next;
                report.held = limited.held;
                report.time_step = self.fire.time_step();
                let ratio = rfo.ratio();
                (0..images)
                    .map(|i| {
                        if i == 0 || i == images - 1 {
                            newton[i].clone()
                        } else {
                            (1.0 - ratio) * &limited.steps[i] + ratio * &newton[i]
                        }
                    })
                    .collect()
            }
            (None, Some(from)) if iteration >= from => steepest_steps(&self.chain, &forces, ends),
            (None, _) => {
                // The time step stays below what the curvature along the last steps allows
                // (not in MultiOptPy, whose time step grows until the steps overshoot).
                if let (Compatibility::Consistent, Some(before), Some(last)) =
                    (compatibility, &self.previous_forces, &self.last_steps)
                {
                    if before.len() == images && last.len() == images {
                        self.fire.observe_curvature(&forces, before, last);
                    }
                }
                let proposed = self
                    .fire
                    .displacements(&forces, self.previous_forces.as_deref());
                let limited = limit_steps(&self.chain, &forces, &proposed, ends, compatibility);
                report.to_previous = limited.to_previous;
                report.to_next = limited.to_next;
                report.held = limited.held;
                report.time_step = self.fire.time_step();
                limited.steps
            }
        };
        if let Some(diis) = self.diis.as_mut() {
            // The chains of the last iterations with what the step rule proposed there.
            steps = diis.steps(&self.chain, &forces, steps, moving_force);
            report.extrapolated = diis.used();
        }
        if let Some(criteria) = &self.settings.hold_converged {
            report.converged = hold_converged(&forces, &mut steps, criteria);
        }
        let mut next: Vec<Col<f64>> = self
            .chain
            .iter()
            .zip(&steps)
            .map(|(x, step)| x + step)
            .collect();
        if ends.first {
            next[0] = self.first[0].clone();
        }
        if ends.last {
            next[images - 1] = self.first[images - 1].clone();
        }
        for (image, start) in next.iter_mut().zip(&self.first) {
            for &k in &self.settings.fixed_components {
                image[k] = start[k];
            }
        }
        if self.settings.kind == ChainKind::Atoms
            && self.settings.align
            && self.settings.fixed_components.is_empty()
        {
            match (&self.settings.periodic_shifts, compatibility) {
                (Some(along), _) => {
                    let c = super::centroid(&self.first[0], super::BLOCK_ATOM);
                    centre_chain_at(&mut next, [c[0], c[1], c[2]], along);
                }
                (None, Compatibility::MultiOptPy) => align_chain_in_sequence(&mut next),
                (None, Compatibility::Consistent) => {
                    // What is kept for the images turns with them.
                    let motions = align_chain_onto_previous(&mut next);
                    for (k, motion) in motions.iter().enumerate() {
                        forces[k] = motion.vector(&forces[k]);
                        steps[k] = motion.vector(&steps[k]);
                        if let Some(rfo) = self.rfo.as_mut() {
                            rfo.move_image(k, motion)?;
                        }
                    }
                }
            }
        }
        let due = |every: Option<usize>| {
            every.is_some_and(|n| n > 0 && iteration > 0 && iteration.is_multiple_of(n))
        };
        if due(self.settings.equal_spacing_every) {
            // Climbing images stay on their saddles.
            next = equal_spacing_between(
                &next,
                self.settings.kind,
                &report.climbing,
                Curve::of(compatibility),
            );
            report.redistributed = true;
        }
        if due(self.settings.predicted_maxima_every) {
            next = gathered_at_predicted_maxima(
                &next,
                &report.energies,
                &gradients,
                compatibility,
                // MultiOptPy spreads the images evenly.
                match compatibility {
                    Compatibility::Consistent => self.settings.gathering,
                    Compatibility::MultiOptPy => None,
                },
                // ... and moves the image at a maximum by what the prediction says.
                match compatibility {
                    Compatibility::Consistent => Some(&mut self.peak_moves),
                    Compatibility::MultiOptPy => None,
                },
            );
            report.redistributed = true;
        }
        let (next, grown) = self.with_added_images(next, &mut report)?;
        self.evaluated = Some(evaluated);
        self.chain = next;
        self.previous_forces = (!grown).then_some(forces);
        self.last_steps = if report.redistributed || grown {
            None
        } else {
            Some(steps)
        };
        self.iteration += 1;
        self.hessian_due = self.rfo.is_some()
            && (self
                .settings
                .hessian_every
                .is_some_and(|every| every > 0 && self.iteration.is_multiple_of(every))
                // MultiOptPy computes the Hessians of all images anew after an addition.
                || (grown && compatibility == Compatibility::MultiOptPy));
        Ok(report)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "machine_tests.rs"]
mod tests;
