//! Improved elastic image pair method (iEIP): two images, one in each of two basins, move
//! towards each other and up to the ridge between the basins, where they meet near the
//! transition state.
//!
//! Y. Liu, H. Qi, M. Lei, J. Chem. Theory Comput. 2023, 19, 2410 (the improved method); the
//! weights of the attraction of the images follow DS-AFIR (S. Maeda, Y. Harabuchi, M.
//! Takagi, K. Saita, K. Suzuki, T. Ichino, Y. Sumiya, K. Sugiyama, Y. Ono, J. Comput. Chem.
//! 2018, 39, 233), in MultiOptPy's form.
//!
//! This is MultiOptPy's procedure, iteration by iteration ([`forces`] has the displacements
//! of one iteration): both images are evaluated, their displacements are summed and passed
//! through AdaBelief, the new images are laid onto each other (both centred, the first
//! turned onto the second), `beta` grows by 2 % whenever the images moved apart, and the
//! pair has converged when the images are closer than 0.15 Bohr after a step. With `-mi` the
//! images are relaxed on hyperspheres before the displacements of an iteration are made
//! ([`micro`]).
//!
//! [`ElasticImagePair`] is an ask/tell state machine over two points of `n` variables (the
//! `3N` Cartesian coordinates of two structures, or any other variables; the images are laid
//! onto each other only for structures on surfaces without a preferred orientation). It asks
//! for both images at a time and never evaluates anything itself.
//!
//! MultiOptPy: `OtherMethod/elastic_image_pair.py:ElasticImagePair.iteration`. MultiOptPy turns
//! the first evaluated structure onto the second without turning its gradient; here the
//! inputs are laid onto each other before the first evaluation, so that the structures that
//! are evaluated are those the displacements are made for (MultiOptPy lays every new pair
//! onto each other as well, so from the second iteration on the two agree).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod forces;
pub mod micro;
pub mod polar;
pub mod settings;
#[cfg(test)]
mod tests;

use faer::Col;

pub use forces::{aligned_pair, ImageValue, Moments, Parts};
pub use micro::{MicroCoordinateFactory, MicroCoordinates, MicroRelaxation, MicroStatus};
pub use polar::PolarAboutCentre;
pub use settings::{IeipSettings, MicroSettings, SEPARATION_TOLERANCE};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_norm;

/// Why an iEIP run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IeipEnd {
    /// The images came closer than the separation tolerance.
    Converged,
    /// The iterations are used up (MultiOptPy writes `not_converged.txt`).
    IterationLimit,
}

/// Next action of an [`ElasticImagePair`].
#[derive(Debug, Clone)]
pub enum IeipAction {
    /// Evaluate the two images (energy and gradient, with and without bias potentials).
    Evaluate([Col<f64>; 2]),
    /// The run has ended.
    Finished(IeipEnd),
}

/// The relaxation on the hyperspheres in an iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MicroSummary {
    /// Steps taken.
    pub steps: usize,
    /// Whether the forces fell below the tolerance.
    pub converged: bool,
    /// Length of the force in hyperspherical coordinates of each image at the end.
    pub forces: [f64; 2],
}

/// The step of an iteration.
#[derive(Debug, Clone)]
pub struct IeipStep {
    /// The displacements the step was made of.
    pub parts: Parts,
    /// The steps of AdaBelief.
    pub steps: [Col<f64>; 2],
    /// The images after the step (laid onto each other).
    pub new_images: [Col<f64>; 2],
    /// Their separation (MultiOptPy `delta_geometry`).
    pub separation: f64,
    /// `beta` of this iteration.
    pub beta: f64,
    /// The relaxation on the hyperspheres before the step.
    pub micro: Option<MicroSummary>,
}

/// What one `tell` did.
#[derive(Debug, Clone)]
pub struct IeipReport {
    /// Iteration (0 for the first).
    pub iteration: usize,
    /// Step of the relaxation on the hyperspheres (`None`: the images of the iteration).
    pub micro_step: Option<usize>,
    /// The images that were evaluated.
    pub images: [Col<f64>; 2],
    /// The surface there.
    pub values: [ImageValue; 2],
    /// The step of the iteration, once it was made.
    pub step: Option<IeipStep>,
    /// Why the run ended, if it did.
    pub end: Option<IeipEnd>,
}

/// Ask/tell state machine of the improved elastic image pair method.
pub struct ElasticImagePair {
    settings: IeipSettings,
    n: usize,
    images: [Col<f64>; 2],
    start: Option<[Col<f64>; 2]>,
    previous: Option<[Col<f64>; 2]>,
    moments: [Moments; 2],
    beta: f64,
    iteration: usize,
    separation: f64,
    micro: Option<MicroRelaxation>,
    end: Option<IeipEnd>,
    step_map: Option<super::points::StepMap>,
}

impl ElasticImagePair {
    /// Take the steps of the images through other coordinates: the displacements and the
    /// distance of the images stay those of MultiOptPy's procedure in the coordinates of the
    /// points, and `map` turns the step from `x` to `y` into the point it reaches.
    pub fn with_step_map(mut self, map: super::points::StepMap) -> Self {
        self.step_map = Some(map);
        self
    }

    /// A run between the points `first` and `second` (laid onto each other first when the
    /// settings say so).
    pub fn new(settings: IeipSettings, first: &Col<f64>, second: &Col<f64>) -> MoptResult<Self> {
        settings.validate()?;
        let n = first.nrows();
        if n == 0 || second.nrows() != n {
            return Err(MoptError::config(format!(
                "iEIP: the images have {n} and {} variables",
                second.nrows()
            )));
        }
        if first.iter().chain(second.iter()).any(|v| !v.is_finite()) {
            return Err(MoptError::config("iEIP: an image is not finite"));
        }
        if let Some(&k) = settings.frozen.iter().find(|&&k| k >= n) {
            return Err(MoptError::config(format!(
                "iEIP: frozen variable {k} of {n}"
            )));
        }
        if settings.align && (!settings.frozen.is_empty() || !n.is_multiple_of(3)) {
            return Err(MoptError::config(
                "iEIP: the images can be laid onto each other only for structures without \
                 fixed atoms",
            ));
        }
        if settings.micro.steps > 0 && !settings.frozen.is_empty() {
            return Err(MoptError::unsupported(
                "iEIP: the relaxation on hyperspheres (-mi) is not available with fixed atoms",
            ));
        }
        let images = if settings.align {
            let (a, b) = aligned_pair(first, second);
            [a, b]
        } else {
            [first.clone(), second.clone()]
        };
        if col_norm(&(&images[1] - &images[0])) == 0.0 {
            return Err(MoptError::config("iEIP: the two images coincide"));
        }
        Ok(Self {
            beta: settings.beta,
            settings,
            n,
            images,
            start: None,
            previous: None,
            moments: [Moments::zeros(n), Moments::zeros(n)],
            iteration: 0,
            separation: 0.0,
            micro: None,
            end: None,
            step_map: None,
        })
    }

    /// Number of variables of one image.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Iterations completed.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// `beta` of the next iteration.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Separation of the images after the last step (0 before the first).
    pub fn separation(&self) -> f64 {
        self.separation
    }

    /// Why the run ended, once it has.
    pub fn end(&self) -> Option<IeipEnd> {
        self.end
    }

    /// The images to evaluate next (after the end: the images after the last step).
    pub fn images(&self) -> [&Col<f64>; 2] {
        [&self.images[0], &self.images[1]]
    }

    /// The point halfway between the images: the estimate of the transition state.
    pub fn transition_state_estimate(&self) -> Col<f64> {
        Col::from_fn(self.n, |k| 0.5 * (self.images[0][k] + self.images[1][k]))
    }

    /// Next action.
    pub fn ask(&self) -> IeipAction {
        if let Some(end) = self.end {
            return IeipAction::Finished(end);
        }
        match &self.micro {
            Some(m) => {
                let [a, b] = m.points();
                IeipAction::Evaluate([a.clone(), b.clone()])
            }
            None => IeipAction::Evaluate(self.images.clone()),
        }
    }

    fn check(&self, values: &[ImageValue; 2]) -> MoptResult<()> {
        for v in values {
            let finite = v.energy.is_finite()
                && v.biased_energy.is_finite()
                && v.gradient
                    .iter()
                    .chain(v.biased_gradient.iter())
                    .all(|x| x.is_finite());
            if v.gradient.nrows() != self.n || v.biased_gradient.nrows() != self.n || !finite {
                return Err(MoptError::calc(
                    "iEIP: the surface at an image is not finite or of another size",
                ));
            }
        }
        Ok(())
    }

    /// The surface at the two images of [`Self::ask`].
    pub fn tell(&mut self, values: [ImageValue; 2]) -> MoptResult<IeipReport> {
        if self.end.is_some() {
            return Err(MoptError::config("tell() called after the iEIP run ended"));
        }
        self.check(&values)?;
        let iteration = self.iteration;
        if let Some(micro) = self.micro.as_mut() {
            let told = {
                let [a, b] = micro.points();
                [a.clone(), b.clone()]
            };
            let step = micro.steps();
            let status = micro.tell(
                [values[0].biased_energy, values[1].biased_energy],
                [&values[0].biased_gradient, &values[1].biased_gradient],
            )?;
            if let MicroStatus::Done { converged } = status {
                let summary = MicroSummary {
                    steps: micro.steps(),
                    converged,
                    forces: micro.forces(),
                };
                self.micro = None;
                return self.outer_step(told, values, Some(summary));
            }
            return Ok(IeipReport {
                iteration,
                micro_step: Some(step),
                images: told,
                values,
                step: None,
                end: None,
            });
        }
        let images = self.images.clone();
        if self.start.is_none() {
            self.start = Some(images.clone());
        }
        if let (Some(previous), true, Some(coordinates)) = (
            &self.previous,
            self.settings.micro.steps > 0,
            &self.settings.micro_coordinates,
        ) {
            let mut micro = MicroRelaxation::new(
                self.settings.micro,
                coordinates,
                previous.clone(),
                images.clone(),
            )?;
            let status = micro.tell(
                [values[0].biased_energy, values[1].biased_energy],
                [&values[0].biased_gradient, &values[1].biased_gradient],
            )?;
            if let MicroStatus::Done { converged } = status {
                let summary = MicroSummary {
                    steps: 0,
                    converged,
                    forces: micro.forces(),
                };
                return self.outer_step(images, values, Some(summary));
            }
            self.micro = Some(micro);
            return Ok(IeipReport {
                iteration,
                micro_step: None,
                images,
                values,
                step: None,
                end: None,
            });
        }
        self.outer_step(images, values, None)
    }

    /// The displacements of the iteration at the images `x` and the step.
    fn outer_step(
        &mut self,
        x: [Col<f64>; 2],
        values: [ImageValue; 2],
        micro: Option<MicroSummary>,
    ) -> MoptResult<IeipReport> {
        let start = self
            .start
            .clone()
            .ok_or_else(|| MoptError::calc("iEIP: the starting images are not known"))?;
        let mut parts = forces::displacements(
            self.iteration,
            [&x[0], &x[1]],
            [&values[0], &values[1]],
            [&start[0], &start[1]],
            self.beta,
        );
        for total in &mut parts.total {
            for &k in &self.settings.frozen {
                total[k] = 0.0;
            }
        }
        let steps = [
            self.moments[0].step(&parts.total[0]),
            self.moments[1].step(&parts.total[1]),
        ];
        let mut moved = [&x[0] + &steps[0], &x[1] + &steps[1]];
        if let Some(map) = &self.step_map {
            // The steps through other coordinates (`-coord`).
            moved = [map(&x[0], &moved[0])?, map(&x[1], &moved[1])?];
        }
        let new_images = if self.settings.align {
            let (a, b) = aligned_pair(&moved[0], &moved[1]);
            [a, b]
        } else {
            moved
        };
        let previous_separation = if self.iteration == 0 {
            0.0
        } else {
            self.separation
        };
        let separation = col_norm(&(&new_images[1] - &new_images[0]));
        self.separation = separation;
        let beta = self.beta;
        self.previous = Some(x.clone());
        if separation < SEPARATION_TOLERANCE {
            self.end = Some(IeipEnd::Converged);
        } else if separation > previous_separation {
            self.beta *= settings::BETA_GROWTH;
        }
        let iteration = self.iteration;
        self.iteration += 1;
        if self.end.is_none() && self.iteration >= self.settings.max_iterations {
            self.end = Some(IeipEnd::IterationLimit);
        }
        self.images = new_images.clone();
        Ok(IeipReport {
            iteration,
            micro_step: None,
            images: x,
            values,
            step: Some(IeipStep {
                parts,
                steps,
                new_images,
                separation,
                beta,
                micro,
            }),
            end: self.end,
        })
    }
}
