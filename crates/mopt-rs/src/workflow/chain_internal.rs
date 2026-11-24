//! Chains of structures relaxed in internal coordinates.
//!
//! [`InternalSurface`] turns a chain of structures into a chain of plain vectors: every
//! image is the vector of its internal coordinates relative to one reference structure,
//! and the relaxation of [`crate::path::band`] works on these vectors as on any other
//! variables. Differences of images, tangents, lengths along the chain, steps, the
//! Hessians of the images, the redistributions, and a Gaussian-process model
//! ([`super::chain_surrogate`]) are then all in internal coordinates, where a turning
//! group moves along a straight line and a stretching bond is one coordinate. Nothing in
//! the relaxation knows about it.
//!
//! When an image is evaluated, the structure with its coordinates is found by the
//! iterative back-transformation of [`crate::coord::system::transform`], starting from
//! the nearest structure that is already known; the surface behind (the calculator, with
//! everything on it) gets that structure, and its gradient and Hessian are taken to the
//! coordinates with the Wilson matrix there (`g_q = G^+ B g_x`, the Hessian without the
//! term of the second derivatives of the coordinates).
//!
//! The coordinates have to be independent, so that every vector the relaxation makes is
//! the vector of a structure: delocalized internal coordinates (`dic`), translation-
//! rotation internal coordinates (`tric`), and their combination with Cartesian
//! coordinates (`tric-cart`), all set up once at the reference structure. They stay as
//! they are for the whole run (a chain needs one set for all its images): where they
//! break down on the way (an angle that becomes straight in `dic`), the evaluation of
//! that image fails and is treated like a calculation that failed. `tric-cart` has no
//! such places.
//!
//! MultiOptPy: no counterpart (its chains are in Cartesian coordinates).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::Col;

use super::chain::ChainSurface;
use crate::coord::system::transform::{back_transform, BackTransformOptions, CoordinateFrame};
use crate::coord::system::CoordinateSystem;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_norm, orthonormalize_columns};
use crate::math::projection::build_tr_rot_basis;
use crate::path::band::machine::ImageValue;

/// Largest difference between the coordinates asked for and those of the structure found
/// for them, relative to the step from the structure the search started at, beyond what
/// the back-transformation leaves anyway ([`REACHED`]).
const NOT_REACHED: f64 = 1e-2;
/// A difference below this counts as reached whatever the step (the back-transformation
/// ends at `1e-6`).
const REACHED: f64 = 1e-4;

/// Curvature given to the coordinates of the position and orientation of the whole
/// structure in the Hessians of the images (Hartree per squared unit of the coordinates).
const RIGID_CURVATURE: f64 = 1.0;

/// A structure and its coordinates.
#[derive(Debug, Clone)]
struct Known {
    q: Col<f64>,
    x: Col<f64>,
}

struct State {
    coords: Box<dyn CoordinateSystem>,
    /// The structures of the chain evaluated before the one at hand.
    before: Vec<Known>,
    /// The structures evaluated in the iteration at hand, with the number of their image.
    now: Vec<(usize, Known)>,
}

/// `inner`, a surface of structures, as a surface of their internal coordinates.
pub struct InternalSurface<'a> {
    inner: &'a dyn ChainSurface,
    reference: MolecularSystem,
    name: String,
    dimension: usize,
    state: Mutex<State>,
}

impl<'a> InternalSurface<'a> {
    /// The surface `inner` in the coordinates `coords`, set up at `reference`, and the
    /// chain of structures `chain` (Cartesian, Bohr) as vectors of these coordinates
    /// relative to `reference`.
    pub fn new(
        inner: &'a dyn ChainSurface,
        mut coords: Box<dyn CoordinateSystem>,
        reference: MolecularSystem,
        chain: &[Col<f64>],
    ) -> MoptResult<(Self, Vec<Col<f64>>)> {
        coords.initialize(&reference)?;
        if coords.is_cartesian() {
            return Err(MoptError::config(
                "Cartesian coordinates need no change of coordinates",
            ));
        }
        let dimension = coords.dim(&reference);
        let mut before = Vec::with_capacity(chain.len());
        for (image, x) in chain.iter().enumerate() {
            let structure = reference.with_flat_coords_bohr(x)?;
            let q = coords.displacement(&structure, &reference).map_err(|e| {
                e.context(format!(
                    "image {image} of the first chain in the coordinates {}",
                    coords.name()
                ))
            })?;
            before.push(Known { q, x: x.clone() });
        }
        let points = before.iter().map(|known| known.q.clone()).collect();
        let name = coords.name().to_string();
        Ok((
            Self {
                inner,
                reference,
                name,
                dimension,
                state: Mutex::new(State {
                    coords,
                    before,
                    now: Vec::new(),
                }),
            },
            points,
        ))
    }

    /// The number of coordinates.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    fn state(&self) -> MoptResult<std::sync::MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| MoptError::calc("the coordinates of the chain are not available"))
    }

    /// The structures (Cartesian, Bohr) of the images `chain` (vectors of the
    /// coordinates): those that were evaluated in the iteration at hand as they were
    /// evaluated, the others (images a model answered for) found anew.
    pub fn structures(&self, chain: &[Col<f64>]) -> MoptResult<Vec<Col<f64>>> {
        let state = self.state()?;
        chain
            .iter()
            .enumerate()
            .map(|(image, q)| {
                let evaluated = state
                    .now
                    .iter()
                    .find(|(i, known)| *i == image && col_norm(&(&known.q - q)) < 1e-12);
                match evaluated {
                    Some((_, known)) => Ok(known.x.clone()),
                    None => self.structure_at(&state, q).map(|known| known.x),
                }
            })
            .collect()
    }

    /// The structure with the coordinates `q`, found from the nearest structure known.
    fn structure_at(&self, state: &State, q: &Col<f64>) -> MoptResult<Known> {
        let nearest = state
            .before
            .iter()
            .chain(state.now.iter().map(|(_, known)| known))
            .min_by(|a, b| col_norm(&(&a.q - q)).total_cmp(&col_norm(&(&b.q - q))))
            .ok_or_else(|| MoptError::calc("no structure of the chain is known"))?;
        let step = q - &nearest.q;
        let length = col_norm(&step);
        if length < 1e-14 {
            return Ok(nearest.clone());
        }
        let start = self.reference.with_flat_coords_bohr(&nearest.x)?;
        let dx = back_transform(
            state.coords.as_ref(),
            &start,
            &step,
            &BackTransformOptions::default(),
        )?;
        let x = &nearest.x + dx;
        let structure = self.reference.with_flat_coords_bohr(&x)?;
        let reached = state
            .coords
            .displacement(&structure, &self.reference)
            .map_err(|e| MoptError::calc(format!("the coordinates {}: {e}", self.name)))?;
        let off = col_norm(&(&reached - q));
        if !off.is_finite() || off > NOT_REACHED * length + REACHED {
            return Err(MoptError::calc(format!(
                "no structure has the coordinates asked for ({} coordinates {}; off by \
                 {off:.2e} after a step of {length:.2e}): the coordinates break down here",
                self.dimension, self.name
            )));
        }
        // The image is where the structure is: what is kept and what is told back are
        // the coordinates of the structure that was found.
        Ok(Known { q: reached, x })
    }
}

impl ChainSurface for InternalSurface<'_> {
    fn description(&self) -> String {
        format!(
            "{}; the chain is in {} internal coordinates ({})",
            self.inner.description(),
            self.dimension,
            self.name
        )
    }

    fn evaluate(
        &self,
        point: &Col<f64>,
        index: usize,
        need_hessian: bool,
    ) -> MoptResult<ImageValue> {
        if point.nrows() != self.dimension {
            return Err(MoptError::config(format!(
                "an image of {} coordinates in a chain of {}",
                point.nrows(),
                self.dimension
            )));
        }
        let known = {
            let state = self.state()?;
            self.structure_at(&state, point)?
        };
        // Calculated without the lock: the surface behind may take long.
        let value = self.inner.evaluate(&known.x, index, need_hessian)?;
        let mut state = self.state()?;
        let structure = self.reference.with_flat_coords_bohr(&known.x)?;
        let frame = CoordinateFrame::new(state.coords.as_ref(), &structure)
            .map_err(|e| MoptError::calc(format!("the coordinates {}: {e}", self.name)))?;
        let gradient = frame.gradient_to_q(&value.gradient);
        let mut hessian = value.hessian.as_ref().map(|h| frame.hessian_to_q(h));
        if let Some(h) = hessian.as_mut() {
            // Coordinates that contain the position and the orientation of the whole
            // structure (tric): the energy does not depend on them, and a Hessian with
            // zeros there would let the steps run along them. They are made stiff, as
            // the relaxation does for Cartesian coordinates.
            let rigid = build_tr_rot_basis(&structure, false);
            let b = state.coords.wilson_b(&structure)?;
            let image = &b * &rigid;
            let columns: Vec<Col<f64>> = (0..image.ncols())
                .map(|c| Col::from_fn(image.nrows(), |r| image[(r, c)]))
                .collect();
            let stiff = orthonormalize_columns(image.nrows(), &columns, 1e-6);
            for c in 0..stiff.ncols() {
                for i in 0..h.nrows() {
                    for j in 0..h.ncols() {
                        h[(i, j)] += RIGID_CURVATURE * stiff[(i, c)] * stiff[(j, c)];
                    }
                }
            }
        }
        state.now.retain(|(image, _)| *image != index);
        state.now.push((index, known));
        Ok(ImageValue {
            energy: value.energy,
            gradient,
            hessian,
        })
    }

    fn begin_iteration(&self, iteration: usize) {
        self.inner.begin_iteration(iteration);
        if let Ok(mut state) = self.state() {
            // What was evaluated takes the place of what was known for its image; an
            // image that was not evaluated keeps its structure.
            for (image, known) in std::mem::take(&mut state.now) {
                match state.before.get_mut(image) {
                    Some(slot) => *slot = known,
                    None => state.before.push(known),
                }
            }
        }
    }

    fn images_dropped(&self, dropped: &[usize]) {
        self.inner.images_dropped(dropped);
    }

    fn images_added(&self) {
        self.inner.images_added();
    }

    fn values_are_calculated(&self) -> bool {
        self.inner.values_are_calculated()
    }

    fn calculate_next(&self, check: bool) {
        self.inner.calculate_next(check);
    }

    fn told(&self, largest_force: f64) {
        self.inner.told(largest_force);
    }

    fn take_notes(&self) -> Vec<String> {
        self.inner.take_notes()
    }

    fn take_bare_energies(&self) -> Vec<(usize, f64)> {
        self.inner.take_bare_energies()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::coord::system::CoordinateRegistry;
    use crate::domain::SpinCharge;
    use crate::io::xyz::parse_xyz_str;

    /// Springs between all pairs of atoms, with their lengths at rest.
    struct Springs(Vec<f64>);

    impl ChainSurface for Springs {
        fn description(&self) -> String {
            "springs".to_string()
        }

        fn evaluate(&self, x: &Col<f64>, _: usize, _: bool) -> MoptResult<ImageValue> {
            let atoms = x.nrows() / 3;
            let mut energy = 0.0;
            let mut gradient = Col::<f64>::zeros(x.nrows());
            let mut pair = 0;
            for i in 0..atoms {
                for j in i + 1..atoms {
                    let v: Vec<f64> = (0..3).map(|k| x[3 * i + k] - x[3 * j + k]).collect();
                    let d = v.iter().map(|c| c * c).sum::<f64>().sqrt();
                    let off = d - self.0[pair];
                    pair += 1;
                    energy += 0.5 * off * off;
                    for k in 0..3 {
                        gradient[3 * i + k] += off * v[k] / d;
                        gradient[3 * j + k] -= off * v[k] / d;
                    }
                }
            }
            Ok(ImageValue::new(energy, gradient))
        }
    }

    #[test]
    fn the_gradient_in_the_coordinates_is_that_of_the_energy_along_them() {
        // Hydrogen peroxide, a little twisted: no three atoms on a line.
        let reference = parse_xyz_str(
            "4\n\nO 0.00 0.70 0.00\nO 0.00 -0.70 0.00\nH 0.85 0.95 0.35\nH -0.80 -0.95 0.45\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let springs = Springs(vec![2.5, 1.9, 3.4, 3.5, 1.7, 4.4]);
        let x0 = reference.coords_flat_bohr();
        for name in ["dic", "tric", "tric-cart"] {
            let coords = CoordinateRegistry::default().create(name).unwrap();
            let (surface, points) = InternalSurface::new(
                &springs,
                coords,
                reference.clone(),
                std::slice::from_ref(&x0),
            )
            .unwrap();
            let q0 = points[0].clone();
            // The reference has the coordinates zero and its own energy.
            assert!(col_norm(&q0) < 1e-12, "{name}");
            let here = surface.evaluate(&q0, 0, false).unwrap();
            let own = springs.evaluate(&x0, 0, false).unwrap();
            assert!((here.energy - own.energy).abs() < 1e-12, "{name}");
            let mut worst = 0.0_f64;
            for k in 0..surface.dimension() {
                let (mut up, mut down) = (q0.clone(), q0.clone());
                up[k] += 1e-4;
                down[k] -= 1e-4;
                let slope = (surface.evaluate(&up, 0, false).unwrap().energy
                    - surface.evaluate(&down, 0, false).unwrap().energy)
                    / 2e-4;
                worst = worst.max((slope - here.gradient[k]).abs());
            }
            println!(
                "{name}: {} coordinates, gradient off by {worst:.2e}",
                surface.dimension()
            );
            assert!(worst < 1e-5, "{name}: {worst}");
            // A step and back: the same structure again.
            let mut away = q0.clone();
            away[0] += 0.2;
            surface.begin_iteration(1);
            surface.evaluate(&away, 0, false).unwrap();
            surface.begin_iteration(2);
            let back = surface.evaluate(&q0, 0, false).unwrap();
            assert!(
                (back.energy - own.energy).abs() < 1e-8,
                "{name}: {}",
                back.energy - own.energy
            );
        }
    }
}
