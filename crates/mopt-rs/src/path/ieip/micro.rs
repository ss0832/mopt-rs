//! The relaxation of the two images before the displacements of an iteration (`-mi`): from
//! the second iteration on, each image is moved downhill on the surface with the bias
//! potentials in a coordinate system about where it was in the iteration before, with some of
//! the coordinates kept. MultiOptPy's choice, the default ([`MicroCoordinates::polar`]), is
//! the polar coordinates of the displacement from that structure ([`super::polar`]) with the
//! length kept: the image moves on the hypersphere around where it was.
//!
//! The steps are L-BFGS steps in those coordinates (gradients and steps through the shared
//! machinery of [`crate::coord::system::transform`], so that any [`CoordinateSystem`] can be
//! given), limited by a trust radius in a length that weighs every third coordinate by 1 and
//! the others by 0.85 (the weight of the others shrinks while steps longer than 0.3 are made
//! in the coordinates `1, 4, 7, ...`), with MultiOptPy's constants: trust radius 0.015 at the
//! start, between 0.01 and 0.1, halved after a step whose energy change is less than 0.2 of
//! the predicted one, and raised by 1.25 after one of more than 0.85 at the trust radius. A
//! relaxation ends after `-mi` steps or when the force in these coordinates is below 0.01 for
//! both images.
//!
//! MultiOptPy: `OtherMethod/elastic_image_pair.py:ElasticImagePair.microiteration` and
//! `lbfgs_update`. Differences: MultiOptPy pairs steps in Cartesian coordinates with
//! differences of polar forces (of the opposite sign of a gradient), takes the polar gradient
//! at the coordinates of the structure about the origin (and with a wrong Jacobian), compares
//! the energy change of a step with the prediction of the next one, keeps the coordinates
//! `1, 4, 7, ...` within `[0, pi]` as if they were the polar angles of atoms, and returns the
//! structure after its last step with the energy and gradient of the structure before it.
//! Here the pairs are steps and gradient differences in the same coordinates, the gradient is
//! that of the coordinates, a step is judged by its own prediction, the ranges of the angles
//! are those of the coordinate system (through its displacements), and the relaxation ends
//! with a structure that was evaluated.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;
use std::sync::Arc;

use faer::Col;

use super::polar::PolarAboutCentre;
use super::settings::MicroSettings;
use crate::coord::system::transform::{back_transform, BackTransformOptions};
use crate::coord::system::{CoordinateFrame, CoordinateSystem};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm};

/// Added to divisors (MultiOptPy's `1e-10`).
const TINY: f64 = 1e-10;
/// Below this ratio of actual to predicted change a step counts as poor.
const POOR_RATIO: f64 = 0.2;
/// Above this ratio a step at the trust radius counts as excellent.
const EXCELLENT_RATIO: f64 = 0.85;
/// Factor of the trust radius after a poor step.
const SHRINK: f64 = 0.5;
/// Factor of the trust radius after an excellent one.
const GROW: f64 = 1.25;
/// A step longer than this in one of the coordinates `1, 4, 7, ...` shrinks their weight.
const LARGE_ANGLE: f64 = 0.3;

/// Makes the coordinate system of the relaxation of an image about its centre (the flat
/// Cartesian coordinates of the image in the iteration before).
pub type MicroCoordinateFactory =
    Arc<dyn Fn(&Col<f64>) -> MoptResult<Box<dyn CoordinateSystem>> + Send + Sync>;

/// The coordinates of the relaxation: the atoms, the coordinate system about a centre, and
/// the coordinates that are kept.
#[derive(Clone)]
pub struct MicroCoordinates {
    /// The atoms, charge, spin, and cell of the images.
    pub template: MolecularSystem,
    /// The coordinate system about a centre.
    pub factory: MicroCoordinateFactory,
    /// Coordinates that are kept (MultiOptPy: the length of the polar coordinates, 0).
    pub kept: Vec<usize>,
}

impl std::fmt::Debug for MicroCoordinates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MicroCoordinates")
            .field("atoms", &self.template.n_atoms())
            .field("kept", &self.kept)
            .finish_non_exhaustive()
    }
}

impl MicroCoordinates {
    /// MultiOptPy's: the polar coordinates of the displacement from the centre, the length
    /// kept.
    pub fn polar(template: MolecularSystem) -> Self {
        Self {
            template,
            factory: Arc::new(|centre| {
                Ok(Box::new(PolarAboutCentre::new(centre.clone())) as Box<dyn CoordinateSystem>)
            }),
            kept: vec![0],
        }
    }
}

/// What one image remembers of its last step.
struct LastStep {
    structure: MolecularSystem,
    gradient: Col<f64>,
    energy: f64,
    predicted: f64,
    weighted_length: f64,
    largest_angle: f64,
}

/// One image in its coordinates.
struct Image {
    coordinates: Box<dyn CoordinateSystem>,
    point: Col<f64>,
    trust: f64,
    memory: VecDeque<(Col<f64>, Col<f64>)>,
    last: Option<LastStep>,
}

/// State of a relaxation after the images were told.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicroStatus {
    /// The new points are to be evaluated.
    Continue,
    /// The points just evaluated are the result.
    Done {
        /// Whether the forces fell below the tolerance.
        converged: bool,
    },
}

/// The relaxation of both images.
pub struct MicroRelaxation {
    settings: MicroSettings,
    template: MolecularSystem,
    kept: Vec<usize>,
    images: [Image; 2],
    angular_weight: f64,
    steps: usize,
    forces: [f64; 2],
}

/// MultiOptPy's L-BFGS two-loop recursion: the step `-H g` from the pairs `(s, y)`.
fn lbfgs_direction(memory: &VecDeque<(Col<f64>, Col<f64>)>, gradient: &Col<f64>) -> Col<f64> {
    let k = memory.len();
    let Some((s_last, y_last)) = memory.back() else {
        return gradient * -1.0;
    };
    let rho: Vec<f64> = memory
        .iter()
        .map(|(s, y)| 1.0 / (col_dot(y, s) + TINY))
        .collect();
    let mut alpha = vec![0.0; k];
    let mut q = gradient.clone();
    for i in (0..k).rev() {
        let (s, y) = &memory[i];
        alpha[i] = rho[i] * col_dot(s, &q);
        q -= y * alpha[i];
    }
    let gamma = col_dot(s_last, y_last) / (col_dot(y_last, y_last) + TINY);
    let mut r = q * gamma;
    for i in 0..k {
        let (s, y) = &memory[i];
        let b = rho[i] * col_dot(y, &r);
        r += s * (alpha[i] - b);
    }
    r * -1.0
}

impl MicroRelaxation {
    /// A relaxation of the images now at `points` in the coordinates about `centres`.
    pub fn new(
        settings: MicroSettings,
        coordinates: &MicroCoordinates,
        centres: [Col<f64>; 2],
        points: [Col<f64>; 2],
    ) -> MoptResult<Self> {
        let [c1, c2] = centres;
        let [p1, p2] = points;
        let image = |centre: &Col<f64>, point: Col<f64>| -> MoptResult<Image> {
            let mut system = (coordinates.factory)(centre)?;
            system.initialize(&coordinates.template.with_flat_coords_bohr(&point)?)?;
            Ok(Image {
                coordinates: system,
                point,
                trust: settings.initial_trust,
                memory: VecDeque::new(),
                last: None,
            })
        };
        Ok(Self {
            angular_weight: settings.angular_weight,
            images: [image(&c1, p1)?, image(&c2, p2)?],
            template: coordinates.template.clone(),
            kept: coordinates.kept.clone(),
            settings,
            steps: 0,
            forces: [f64::NAN; 2],
        })
    }

    /// The points to evaluate next (after the end: the result).
    pub fn points(&self) -> [&Col<f64>; 2] {
        [&self.images[0].point, &self.images[1].point]
    }

    /// Steps taken so far.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Length of the force in the coordinates of each image at the last points.
    pub fn forces(&self) -> [f64; 2] {
        self.forces
    }

    /// Trust radii of the images.
    pub fn trust_radii(&self) -> [f64; 2] {
        [self.images[0].trust, self.images[1].trust]
    }

    fn keep(&self, mut v: Col<f64>) -> Col<f64> {
        for &k in &self.kept {
            if k < v.nrows() {
                v[k] = 0.0;
            }
        }
        v
    }

    /// The energies (with bias potentials) and their gradients at [`Self::points`].
    pub fn tell(
        &mut self,
        energies: [f64; 2],
        gradients: [&Col<f64>; 2],
    ) -> MoptResult<MicroStatus> {
        let mut states = Vec::with_capacity(2);
        for k in 0..2 {
            let structure = self.template.with_flat_coords_bohr(&self.images[k].point)?;
            let frame = CoordinateFrame::new(self.images[k].coordinates.as_ref(), &structure)?;
            let g = self.keep(frame.gradient_to_q(gradients[k]));
            let image = &mut self.images[k];
            if let Some(last) = image.last.take() {
                let s = image
                    .coordinates
                    .displacement(&structure, &last.structure)?;
                let mut s = s;
                for &j in &self.kept {
                    if j < s.nrows() {
                        s[j] = 0.0;
                    }
                }
                let y = &g - &last.gradient;
                if col_norm(&s) > TINY && col_norm(&y) > TINY && col_dot(&s, &y) > TINY {
                    image.memory.push_back((s, y));
                    while image.memory.len() > self.settings.memory {
                        image.memory.pop_front();
                    }
                }
                let ratio = (last.energy - energies[k]) / (last.predicted + TINY);
                if ratio < POOR_RATIO {
                    image.trust = (image.trust * SHRINK).max(self.settings.min_trust);
                } else if ratio > EXCELLENT_RATIO && last.weighted_length >= 0.9 * image.trust {
                    image.trust = (image.trust * GROW).min(self.settings.max_trust);
                }
                if last.largest_angle > LARGE_ANGLE {
                    self.angular_weight = (self.angular_weight * 0.9).max(0.1);
                }
            }
            self.forces[k] = col_norm(&g);
            states.push((structure, g));
        }
        let converged = self
            .forces
            .iter()
            .all(|f| *f < self.settings.force_tolerance);
        if converged || self.steps >= self.settings.steps {
            return Ok(MicroStatus::Done { converged });
        }
        for (k, (structure, g)) in states.into_iter().enumerate() {
            let mut d = self.keep(lbfgs_direction(&self.images[k].memory, &g));
            let weighted = col_norm(&Col::from_fn(d.nrows(), |j| {
                d[j] * if j % 3 == 0 {
                    self.settings.radial_weight
                } else {
                    self.angular_weight
                }
            }));
            let image = &mut self.images[k];
            if weighted > image.trust {
                d *= image.trust / weighted;
            }
            let predicted = -col_dot(&g, &d);
            let largest_angle = (1..d.nrows())
                .step_by(3)
                .map(|j| d[j].abs())
                .fold(0.0, f64::max);
            let dx = back_transform(
                image.coordinates.as_ref(),
                &structure,
                &d,
                &BackTransformOptions::default(),
            )?;
            if dx.iter().any(|v| !v.is_finite()) {
                return Err(MoptError::calc(
                    "iEIP: the step of the relaxation is not finite",
                ));
            }
            image.point = &image.point + &dx;
            image.last = Some(LastStep {
                structure,
                gradient: g,
                energy: energies[k],
                predicted,
                weighted_length: weighted,
                largest_angle,
            });
        }
        self.steps += 1;
        Ok(MicroStatus::Continue)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};

    /// E = sum_i a_i x_i^2 / 2 with a slope, its gradient.
    fn surface(x: &Col<f64>) -> (f64, Col<f64>) {
        let a = |i: usize| 0.4 + 0.3 * i as f64;
        let b = |i: usize| 0.05 * (i as f64 - 1.0);
        let e = (0..x.nrows())
            .map(|i| 0.5 * a(i) * x[i] * x[i] + b(i) * x[i])
            .sum();
        (e, Col::from_fn(x.nrows(), |i| a(i) * x[i] + b(i)))
    }

    fn template() -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::H, Element::H],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn stays_on_the_spheres_and_goes_downhill() {
        let settings = MicroSettings {
            steps: 300,
            ..MicroSettings::default()
        };
        let c1 = Col::from_fn(6, |i| [0.3, -0.2, 0.1, 0.4, 0.0, 0.2][i]);
        let c2 = Col::from_fn(6, |i| [-0.5, 0.2, 0.3, -0.1, 0.1, 0.0][i]);
        let p1 = &c1 + Col::from_fn(6, |i| [0.2, 0.1, -0.1, 0.05, 0.1, 0.0][i]);
        let p2 = &c2 + Col::from_fn(6, |i| [-0.1, 0.2, 0.15, 0.1, 0.0, -0.1][i]);
        let radii = [col_norm(&(&p1 - &c1)), col_norm(&(&p2 - &c2))];
        let start = [surface(&p1).0, surface(&p2).0];
        let coordinates = MicroCoordinates::polar(template());
        let mut m =
            MicroRelaxation::new(settings, &coordinates, [c1.clone(), c2.clone()], [p1, p2])
                .unwrap();
        let mut status = MicroStatus::Continue;
        let mut evaluations = 0;
        while status == MicroStatus::Continue {
            let [a, b] = m.points();
            let (ea, ga) = surface(a);
            let (eb, gb) = surface(b);
            status = m.tell([ea, eb], [&ga, &gb]).unwrap();
            evaluations += 1;
            let [a, b] = m.points();
            assert!((col_norm(&(a - &c1)) - radii[0]).abs() < 1e-5);
            assert!((col_norm(&(b - &c2)) - radii[1]).abs() < 1e-5);
        }
        assert_eq!(m.steps() + 1, evaluations);
        let [a, b] = m.points();
        assert!(surface(a).0 < start[0] && surface(b).0 < start[1]);
        assert!(
            matches!(status, MicroStatus::Done { converged: true }),
            "{status:?} after {} steps, forces {:?}, trust {:?}",
            m.steps(),
            m.forces(),
            m.trust_radii()
        );
    }

    #[test]
    fn two_loop_recursion_is_newton_on_a_quadratic() {
        // With the pairs of a quadratic surface the two-loop step is -A^-1 g on their span.
        let a = [2.0, 0.5];
        let mut memory = VecDeque::new();
        memory.push_back((
            Col::from_fn(2, |i| [1.0, 0.0][i]),
            Col::from_fn(2, |i| [a[0], 0.0][i]),
        ));
        memory.push_back((
            Col::from_fn(2, |i| [0.0, 1.0][i]),
            Col::from_fn(2, |i| [0.0, a[1]][i]),
        ));
        let g = Col::from_fn(2, |i| [0.4, -0.2][i]);
        let d = lbfgs_direction(&memory, &g);
        assert!(
            (d[0] + 0.2).abs() < 1e-8 && (d[1] - 0.4).abs() < 1e-8,
            "{d:?}"
        );
    }
}
