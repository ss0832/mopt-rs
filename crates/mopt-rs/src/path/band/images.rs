//! Images that are added to a chain or taken out of it while it is relaxed.
//!
//! - **Added** ([`AddedImages`], MultiOptPy's `-aneb N FREQ`): after every `FREQ`
//!   iterations, `N` images on either side of every image at a maximum of the energy
//!   profile, between it and its neighbour. An image that was there keeps what is kept
//!   for it (its Hessian and the pairs of its update); a new one starts with a copy of
//!   the Hessian of the image nearest to it. The history of the step rule starts anew,
//!   as in MultiOptPy; MultiOptPy also computes the Hessians of all images anew, which is
//!   done in its form only.
//! - **Failed** ([`FailedImage`]): an image whose calculation failed (no convergence of
//!   the wavefunction, a program that stopped). MultiOptPy leaves such an image out and
//!   goes on with the others. Here the step of the image is taken again first, shorter:
//!   half of it, a quarter, and at last none (the point at which the image was calculated
//!   in the iteration before); only when that fails too, or there is no such point, the
//!   image is taken out of the chain. The driver ([`crate::workflow::chain::relax_chain`])
//!   asks for these points between [`ChainRelaxation::ask`] and
//!   [`ChainRelaxation::tell`].
//!
//! MultiOptPy: `neb.py:NEB._exec_adaptive_neb`; the `except` branch of `calculate` in
//! `Calculator/*_calculation_tools.py` (the image is left out).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::fire::ChainFire;
use super::machine::{ChainRelaxation, ChainReport};
use super::redistribute::{with_images_around, Curve, ImageSource};
use super::{strict_maxima, Compatibility};
use crate::error::{MoptError, MoptResult};

/// The adaptive chain (P. Maragakis, S. A. Andreev, Y. Brumer, D. R. Reichman, and E.
/// Kaxiras, J. Chem. Phys. 2002, 117, 4651, as MultiOptPy has it with `-aneb N FREQ`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddedImages {
    /// Images added on either side of a maximum.
    pub images: usize,
    /// Iterations between two additions (MultiOptPy counts them from zero after an
    /// addition, so that the first addition follows iteration `every` and the later ones
    /// `every + 1` iterations after the one before).
    pub every: usize,
}

/// What is done with an image whose calculation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FailedImage {
    /// Its step is taken again, shorter; if that fails as well, it is taken out.
    #[default]
    Retry,
    /// It is taken out of the chain (MultiOptPy).
    Drop,
    /// The run ends with the error.
    Stop,
}

/// Shares of the step of an image that are tried after its calculation failed.
pub const RETRIED_SHARES: [f64; 3] = [0.5, 0.25, 0.0];

/// What became of an image whose calculation failed.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageFailure {
    /// The image, by its number in the chain that was asked for.
    pub image: usize,
    /// What the calculator said.
    pub message: String,
    /// The share of its step at which it could be calculated (`None`: at none).
    pub retaken_at: Option<f64>,
    /// Whether it was taken out of the chain.
    pub dropped: bool,
}

impl ChainRelaxation {
    /// What is done with images whose calculation fails.
    pub fn on_failure(&self) -> FailedImage {
        match self.settings.compatibility {
            Compatibility::Consistent => self.settings.on_failure,
            // MultiOptPy leaves the image out, unless the run is to end.
            Compatibility::MultiOptPy => match self.settings.on_failure {
                FailedImage::Stop => FailedImage::Stop,
                _ => FailedImage::Drop,
            },
        }
    }

    /// The point to try for `image` after its calculation failed for the `attempt`-th
    /// time (from 0): [`RETRIED_SHARES`] of the way from where the image was calculated
    /// in the iteration before to where it is now. `None`: nothing is left to try.
    pub fn retried_point(&self, image: usize, attempt: usize) -> Option<Col<f64>> {
        let share = *RETRIED_SHARES.get(attempt)?;
        let before = self.evaluated.as_ref()?;
        if !self.asked || before.len() != self.chain.len() {
            return None;
        }
        let (from, to) = (before.get(image)?, self.chain.get(image)?);
        Some(from + share * (to - from))
    }

    /// Put `image` at `point` between [`Self::ask`] and [`Self::tell`] (the point at
    /// which it could be calculated).
    pub fn replace_image(&mut self, image: usize, point: Col<f64>) -> MoptResult<()> {
        if !self.asked {
            return Err(MoptError::config(
                "an image is replaced between ask() and tell() only",
            ));
        }
        let n = self.chain[0].nrows();
        match self.chain.get_mut(image) {
            Some(slot) if point.nrows() == n => {
                *slot = point;
                Ok(())
            }
            _ => Err(MoptError::config(format!(
                "no image {image} of {n} components in the chain"
            ))),
        }
    }

    /// Take the images `dropped` out of the chain between [`Self::ask`] and
    /// [`Self::tell`]; `tell` then takes the values of the others.
    pub fn drop_images(&mut self, dropped: &[usize]) -> MoptResult<()> {
        if !self.asked {
            return Err(MoptError::config(
                "images are taken out between ask() and tell() only",
            ));
        }
        let images = self.chain.len();
        let mut out: Vec<usize> = dropped.to_vec();
        out.sort_unstable();
        out.dedup();
        if out.last().is_some_and(|last| *last >= images) {
            return Err(MoptError::config(format!(
                "the chain has {images} images; {out:?} cannot be taken out"
            )));
        }
        if images - out.len() < 2 {
            return Err(MoptError::calc(format!(
                "the calculation failed for {} of the {images} images of the chain; fewer \
                 than two are left",
                out.len()
            )));
        }
        let without = |all: &mut Vec<Col<f64>>| {
            for &i in out.iter().rev() {
                all.remove(i);
            }
        };
        without(&mut self.chain);
        without(&mut self.first);
        for kept in [
            &mut self.previous_forces,
            &mut self.last_steps,
            &mut self.evaluated,
        ] {
            match kept {
                Some(all) if all.len() == images => without(all),
                _ => *kept = None,
            }
        }
        if let Some(rfo) = self.rfo.as_mut() {
            rfo.remove_images(&out)?;
        }
        self.peak_moves.clear();
        if let Some(diis) = self.diis.as_mut() {
            diis.clear();
        }
        Ok(())
    }

    /// The end of an iteration of an adaptive chain: `next` with images added around the
    /// maxima when they are due (`report.added` says how many), and whether the chain
    /// grew.
    pub(super) fn with_added_images(
        &mut self,
        next: Vec<Col<f64>>,
        report: &mut ChainReport,
    ) -> MoptResult<(Vec<Col<f64>>, bool)> {
        let Some(added) = self.settings.added else {
            return Ok((next, false));
        };
        if !(self.since_added > 0 && self.since_added.is_multiple_of(added.every.max(1))) {
            self.since_added += 1;
            return Ok((next, false));
        }
        self.since_added = 0;
        let (longer, sources) = with_images_around(
            &next,
            &strict_maxima(&report.energies),
            added.images,
            Curve::of(self.settings.compatibility),
        );
        report.added = longer.len() - next.len();
        if report.added == 0 {
            return Ok((next, false));
        }
        // What is kept image by image follows the images; the history of the steps
        // starts anew, as in MultiOptPy.
        self.first = sources
            .iter()
            .zip(&longer)
            .map(|(source, point)| match source {
                ImageSource::Old(i) => self.first[*i].clone(),
                ImageSource::New { .. } => point.clone(),
            })
            .collect();
        if let Some(rfo) = self.rfo.as_mut() {
            rfo.insert_images(&sources)?;
        }
        self.fire = ChainFire::new(self.settings.fire);
        self.peak_moves.clear();
        if let Some(diis) = self.diis.as_mut() {
            diis.clear();
        }
        Ok((longer, true))
    }
}
