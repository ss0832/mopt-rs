//! The force on the images of a chain: the gradient without its component along the chain.
//!
//! For an inner image `i` with gradient `g` and neighbours `i - 1`, `i + 1`, the direction
//! of the chain is taken towards the neighbour of higher energy,
//!
//! - `E(i-1) < E(i) < E(i+1)`: `d = x(i+1) - x(i)`,
//! - `E(i-1) > E(i) > E(i+1)`: `d = x(i) - x(i-1)`,
//! - otherwise (an extremum of the profile): the direction between the two,
//!   `d = a (x(i+1) - x(i)) + b (x(i) - x(i-1))` with
//!   `a = dE_max / (dE_max + dE_min + 1e-8)` and `b = dE_min / (dE_max + dE_min + 1e-8)`,
//!   the larger weight on the side of the higher neighbour (the weighted tangent of
//!   Henkelman and Jonsson). MultiOptPy removes the components along the two directions
//!   one after the other instead, `a` times the one and `b` times the other, which is not
//!   the removal of one direction where the chain has a kink: an image at a minimum of
//!   the profile then keeps a force however long it is relaxed (1.5e-5 Hartree/Bohr on
//!   the Muller-Brown surface scaled by 1e-3). That form is kept under
//!   [`Compatibility::MultiOptPy`],
//!
//! and the component of `g` along `d` is removed block by block: for every block (atom)
//! `k` with `u_k = d_k / |d_k|`, `g_k <- g_k - w u_k (u_k . g_k)`. With blocks of three
//! coordinates every atom loses the part of its gradient along its own displacement to
//! the next image, which is what MultiOptPy's default ("B-matrix constrained NEB") does;
//! with one block of all coordinates the same rule is the projection on the tangent of
//! the chain (the upwind tangent of Henkelman and Jonsson, J. Chem. Phys. 2000, 113,
//! 9978). The force is minus what is left; the end images keep minus their gradient.
//! There are no springs: the spacing of the images is kept by redistribution.
//!
//! Climbing: an image at a maximum of the profile gets `F = -g + 2 (g . t) t` with the
//! unit tangent `t = d / |d|` of the whole image, which takes it up along the chain and
//! down across it (Henkelman, Uberuaga, Jonsson, J. Chem. Phys. 2000, 113, 9901). One
//! direction is reversed, whatever the blocks of the projection are: a saddle has one
//! direction of negative curvature, not one per atom. MultiOptPy uses `F = -2 R` (`R`:
//! the component removed block by block), which moves the image down along the chain and
//! not across it, and scales the forces of the neighbours of a maximum by 0.001; both are
//! kept under [`Compatibility::MultiOptPy`] only.
//!
//! MultiOptPy: `MEP/pathopt_bneb_force.py:CaluculationBNEB`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{check_chain, strict_maxima, Compatibility};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_dot;

/// Squared length below which MultiOptPy's pseudo-inverse does not normalize a direction.
const DIRECTION_THRESHOLD: f64 = 1e-6;
/// What MultiOptPy adds to the length of a displacement before dividing by it.
const LENGTH_SHIFT: f64 = 1e-15;
/// Factor on the forces of the neighbours of a climbing image in MultiOptPy.
const NEIGHBOUR_SCALE: f64 = 0.001;

/// Forces of a chain and the components that were removed from its gradients.
#[derive(Debug, Clone)]
pub struct ChainForces {
    /// Force on every image.
    pub forces: Vec<Col<f64>>,
    /// Component of the gradient along the chain that was removed (zero at the ends);
    /// for an image that climbs, the component along its tangent.
    pub removed: Vec<Col<f64>>,
    /// Direction of the chain at every image, not normalized (zero at the ends).
    pub directions: Vec<Col<f64>>,
    /// Images that climbed.
    pub climbing: Vec<usize>,
}

/// The projection described in the module documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpwindProjection {
    /// Coordinates per block: 3 for the atom-wise rule, the number of coordinates of an
    /// image for the tangent of the whole chain.
    pub block: usize,
    /// Images at maxima climb in iterations after this one (`None`: never).
    pub climb_after: Option<usize>,
    /// Form of the climbing force.
    pub compatibility: Compatibility,
}

impl UpwindProjection {
    /// The atom-wise rule without climbing.
    pub fn atom_wise() -> Self {
        Self {
            block: super::BLOCK_ATOM,
            climb_after: None,
            compatibility: Compatibility::Consistent,
        }
    }

    /// The component of `gradient` along `from -> to`, block by block.
    fn along(&self, gradient: &Col<f64>, from: &Col<f64>, to: &Col<f64>) -> Col<f64> {
        self.along_direction(gradient, &(to - from))
    }

    /// The component of `gradient` along `direction`, block by block.
    fn along_direction(&self, gradient: &Col<f64>, direction: &Col<f64>) -> Col<f64> {
        let n = gradient.nrows();
        let mut out = Col::<f64>::zeros(n);
        for start in (0..n).step_by(self.block) {
            let range = start..start + self.block;
            let length = range
                .clone()
                .map(|k| direction[k].powi(2))
                .sum::<f64>()
                .sqrt();
            if length == 0.0 {
                continue;
            }
            // MultiOptPy divides by the length plus 1e-15 and inverts B^T B with a
            // threshold: directions that short are in effect not removed.
            let scaled = (length / (length + LENGTH_SHIFT)).powi(2);
            let weight = if scaled > DIRECTION_THRESHOLD {
                1.0
            } else {
                scaled * scaled
            };
            let dot: f64 = range
                .clone()
                .map(|k| direction[k] / length * gradient[k])
                .sum();
            for k in range {
                out[k] = weight * dot * direction[k] / length;
            }
        }
        out
    }

    /// The direction of the chain at an inner image (not normalized).
    pub(crate) fn direction(points: [&Col<f64>; 3], energies: [f64; 3]) -> Col<f64> {
        let [previous, current, next] = points;
        let [low, mid, high] = energies;
        if low < mid && mid < high {
            next - current
        } else if low > mid && mid > high {
            current - previous
        } else {
            let (forward, backward) = extremum_weights(energies);
            forward * (next - current) + backward * (current - previous)
        }
    }

    /// What is removed from the gradient of an inner image whose chain has `direction`.
    fn removed(
        &self,
        points: [&Col<f64>; 3],
        energies: [f64; 3],
        gradient: &Col<f64>,
        direction: &Col<f64>,
    ) -> Col<f64> {
        let [previous, current, next] = points;
        let [low, mid, high] = energies;
        let monotonic = (low < mid && mid < high) || (low > mid && mid > high);
        if !monotonic && self.compatibility == Compatibility::MultiOptPy {
            let (forward, backward) = extremum_weights(energies);
            forward * self.along(gradient, current, next)
                + backward * self.along(gradient, previous, current)
        } else {
            self.along_direction(gradient, direction)
        }
    }

    /// Forces of `chain` with its `energies` and `gradients` in `iteration`.
    pub fn forces(
        &self,
        chain: &[Col<f64>],
        energies: &[f64],
        gradients: &[Col<f64>],
        iteration: usize,
    ) -> MoptResult<ChainForces> {
        self.forces_along(chain, energies, gradients, iteration, None)
    }

    /// [`Self::forces`] with the directions of the chain at its images given from outside
    /// (`tangents`, one per image, as [`super::tangents`] makes them) in the place of the
    /// directions towards the higher neighbour.
    pub fn forces_along(
        &self,
        chain: &[Col<f64>],
        energies: &[f64],
        gradients: &[Col<f64>],
        iteration: usize,
        tangents: Option<&[Col<f64>]>,
    ) -> MoptResult<ChainForces> {
        let n = check_chain(chain, self.block)?;
        if tangents.is_some_and(|t| t.len() != chain.len() || t.iter().any(|d| d.nrows() != n)) {
            return Err(MoptError::config(
                "a chain needs one tangent of the size of an image per image",
            ));
        }
        let images = chain.len();
        if energies.len() != images || gradients.len() != images {
            return Err(MoptError::config(
                "a chain needs one energy and one gradient per image",
            ));
        }
        if gradients.iter().any(|g| g.nrows() != n) {
            return Err(MoptError::config(
                "a gradient of the chain has the wrong number of components",
            ));
        }
        let maxima = strict_maxima(energies);
        let climbs = self.climb_after.is_some_and(|after| iteration > after);
        let mut out = ChainForces {
            forces: Vec::with_capacity(images),
            removed: Vec::with_capacity(images),
            directions: Vec::with_capacity(images),
            climbing: Vec::new(),
        };
        for i in 0..images {
            if i == 0 || i == images - 1 {
                out.forces.push(-&gradients[i]);
                out.removed.push(Col::zeros(n));
                out.directions.push(Col::zeros(n));
                continue;
            }
            let points = [&chain[i - 1], &chain[i], &chain[i + 1]];
            let around = [energies[i - 1], energies[i], energies[i + 1]];
            let (direction, mut removed) = match tangents {
                Some(given) => (
                    given[i].clone(),
                    self.along_direction(&gradients[i], &given[i]),
                ),
                None => {
                    let direction = Self::direction(points, around);
                    let removed = self.removed(points, around, &gradients[i], &direction);
                    (direction, removed)
                }
            };
            let mut force = &removed - &gradients[i];
            if climbs {
                match self.compatibility {
                    Compatibility::Consistent => {
                        if maxima.contains(&i) {
                            let whole = Self { block: n, ..*self };
                            removed = whole.along_direction(&gradients[i], &direction);
                            force = 2.0 * &removed - &gradients[i];
                            out.climbing.push(i);
                        }
                    }
                    Compatibility::MultiOptPy => {
                        // MultiOptPy's list of maxima also holds index 0, which makes
                        // image 1 a "neighbour"; images 1 and n - 2 are left out again.
                        let beside = maxima.contains(&(i + 1)) || maxima.contains(&(i - 1));
                        if beside && i != 1 && i != images - 2 {
                            force *= NEIGHBOUR_SCALE;
                        } else if maxima.contains(&i) {
                            force = -2.0 * &removed;
                            out.climbing.push(i);
                        }
                    }
                }
            }
            out.forces.push(force);
            out.removed.push(removed);
            out.directions.push(direction);
        }
        Ok(out)
    }
}

/// Weights of the directions to the next and from the previous image at an extremum of
/// the profile: the larger one on the side of the higher neighbour.
fn extremum_weights(energies: [f64; 3]) -> (f64, f64) {
    let [low, mid, high] = energies;
    let (up, down) = ((high - mid).abs(), (mid - low).abs());
    let (largest, smallest) = (up.max(down), up.min(down));
    let a = largest / (largest + smallest + 1e-8);
    let b = smallest / (largest + smallest + 1e-8);
    if low < high {
        (a, b)
    } else {
        (b, a)
    }
}

/// Curvature of the energy along the chain at the middle one of three images: `E''(0)` of
/// the polynomial of fifth degree through the three energies and the three slopes, the
/// arc length measured from the middle image. The slopes of the outer images are their
/// gradients along the chords to the middle image; the slope of the middle image is its
/// gradient along `direction` (MultiOptPy passes the removed component of the gradient
/// there, which is not a unit vector). Zero for coinciding images or a singular fit.
///
/// MultiOptPy: `MEP/pathopt_bneb_force.py:CaluculationBNEB.calculate_gamma`.
pub fn path_curvature(
    points: [&Col<f64>; 3],
    energies: [f64; 3],
    gradients: [&Col<f64>; 3],
    direction: &Col<f64>,
) -> f64 {
    let [previous, current, next] = points;
    let back = current - previous;
    let ahead = next - current;
    let (d_back, d_ahead) = (col_dot(&back, &back).sqrt(), col_dot(&ahead, &ahead).sqrt());
    if d_back < 1e-6 || d_ahead < 1e-6 {
        return 0.0;
    }
    let s = [-d_back, 0.0, d_ahead];
    let slopes = [
        col_dot(gradients[0], &back) / d_back,
        col_dot(gradients[1], direction),
        col_dot(gradients[2], &ahead) / d_ahead,
    ];
    let mut a = Mat::<f64>::zeros(6, 6);
    let mut b = Mat::<f64>::zeros(6, 1);
    for (row, &x) in s.iter().enumerate() {
        for power in 0..6 {
            a[(row, power)] = x.powi(power as i32);
            if power > 0 {
                a[(row + 3, power)] = power as f64 * x.powi(power as i32 - 1);
            }
        }
        b[(row, 0)] = energies[row];
        b[(row + 3, 0)] = slopes[row];
    }
    let lu = a.partial_piv_lu();
    let solution = faer::linalg::solvers::Solve::solve(&lu, b.as_ref());
    let second = 2.0 * solution[(2, 0)];
    if second.is_finite() {
        second
    } else {
        0.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    #[test]
    fn the_component_towards_the_higher_neighbour_is_removed_atom_by_atom() {
        // Two atoms; the first moves along x from image to image, the second along y.
        let chain = vec![
            col(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            col(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            col(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
        ];
        let gradients = vec![
            col(&[1.0; 6]),
            col(&[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            col(&[1.0; 6]),
        ];
        let rule = UpwindProjection::atom_wise();
        let found = rule
            .forces(&chain, &[0.0, 1.0, 2.0], &gradients, 0)
            .unwrap();
        // Uphill towards the last image: x of atom 1 and y of atom 2 are removed.
        assert_eq!(found.removed[1], col(&[0.3, 0.0, 0.0, 0.0, 0.7, 0.0]));
        assert_eq!(found.forces[1], col(&[0.0, -0.4, -0.5, -0.6, 0.0, -0.8]));
        assert_eq!(found.forces[0], col(&[-1.0; 6]));
        assert!(found.climbing.is_empty());

        // One block of all coordinates: the tangent of the chain, (1, 0, 0, 0, 1, 0) / sqrt 2.
        let whole = UpwindProjection { block: 6, ..rule };
        let found = whole
            .forces(&chain, &[0.0, 1.0, 2.0], &gradients, 0)
            .unwrap();
        let along = 0.5 * (0.3 + 0.7);
        assert!((found.removed[1][0] - along).abs() < 1e-15);
        assert!((found.removed[1][4] - along).abs() < 1e-15);
        assert!(col_dot(&found.forces[1], &col(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])).abs() < 1e-15);

        // A maximum: both sides with weights by the energy differences; climbing turns the
        // removed component round.
        let peak = [0.0, 1.0, 0.5];
        let plain = rule.forces(&chain, &peak, &gradients, 3).unwrap();
        // The chain is straight here, so the direction between the two sides is theirs:
        // the whole component along it is removed.
        assert!((plain.removed[1][0] - 0.3).abs() < 1e-15);
        assert!((plain.removed[1][4] - 0.7).abs() < 1e-15);
        // MultiOptPy removes a share of each side, with weights by the energy differences
        // that do not quite add up to one.
        let a = 1.0 / (1.5 + 1e-8);
        let b = 0.5 / (1.5 + 1e-8);
        let shares = UpwindProjection {
            compatibility: Compatibility::MultiOptPy,
            ..rule
        };
        let plain_legacy = shares.forces(&chain, &peak, &gradients, 3).unwrap();
        assert!((plain_legacy.removed[1][0] - (a + b) * 0.3).abs() < 1e-15);
        let climbing = UpwindProjection {
            climb_after: Some(2),
            ..rule
        };
        let up = climbing.forces(&chain, &peak, &gradients, 3).unwrap();
        assert_eq!(up.climbing, vec![1]);
        // One direction is reversed, the tangent (1, 0, 0, 0, 1, 0) / sqrt 2 of the whole
        // image, not one direction per atom: (g . t) t = (0.5, 0, 0, 0, 0.5, 0).
        let along = col(&[0.5, 0.0, 0.0, 0.0, 0.5, 0.0]);
        let expected = 2.0 * &along - &gradients[1];
        assert!((&up.forces[1] - &expected).iter().all(|d| d.abs() < 1e-15));
        assert!((&up.removed[1] - &along).iter().all(|d| d.abs() < 1e-15));
        assert!((&up.directions[1] - col(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
            .iter()
            .all(|d| d.abs() < 1e-7));
        assert_eq!(up.directions[0], col(&[0.0; 6]));
        assert!(climbing
            .forces(&chain, &peak, &gradients, 2)
            .unwrap()
            .climbing
            .is_empty());
        let legacy = UpwindProjection {
            compatibility: Compatibility::MultiOptPy,
            ..climbing
        };
        let down = legacy.forces(&chain, &peak, &gradients, 3).unwrap();
        assert!((down.forces[1][0] + 2.0 * plain_legacy.removed[1][0]).abs() < 1e-15);
        assert_eq!(down.forces[1][1], 0.0);
    }

    #[test]
    fn at_a_kink_one_direction_between_the_sides_is_removed() {
        // A minimum of the profile at a right angle of the chain, equal energy differences:
        // the direction between the sides is the diagonal.
        let chain = vec![col(&[-1.0, 0.0]), col(&[0.0, 0.0]), col(&[0.0, 1.0])];
        let gradients = vec![col(&[0.0; 2]), col(&[0.3, 0.3]), col(&[0.0; 2])];
        let rule = UpwindProjection {
            block: 2,
            ..UpwindProjection::atom_wise()
        };
        let found = rule
            .forces(&chain, &[1.0, 0.0, 1.0], &gradients, 0)
            .unwrap();
        // A gradient along the diagonal is removed completely: no force is left.
        assert!(
            found.forces[1].iter().all(|f| f.abs() < 1e-9),
            "{:?}",
            found.forces[1]
        );
        // Half of the component along each side leaves half of the gradient.
        let shares = UpwindProjection {
            compatibility: Compatibility::MultiOptPy,
            ..rule
        };
        let left = shares
            .forces(&chain, &[1.0, 0.0, 1.0], &gradients, 0)
            .unwrap();
        assert!((left.forces[1][0] + 0.15).abs() < 1e-8 && (left.forces[1][1] + 0.15).abs() < 1e-8);
    }

    #[test]
    fn coinciding_images_remove_nothing_and_wrong_input_is_refused() {
        let chain = vec![col(&[0.0; 3]), col(&[0.0; 3]), col(&[1.0, 0.0, 0.0])];
        let gradients = vec![col(&[0.1; 3]), col(&[0.2, 0.3, 0.4]), col(&[0.1; 3])];
        let rule = UpwindProjection::atom_wise();
        // Downhill from the first image, which lies on the second: nothing to remove.
        let found = rule
            .forces(&chain, &[2.0, 1.0, 0.0], &gradients, 0)
            .unwrap();
        assert_eq!(found.forces[1], col(&[-0.2, -0.3, -0.4]));
        assert!(rule.forces(&chain, &[0.0, 1.0], &gradients, 0).is_err());
        assert!(rule
            .forces(&chain, &[0.0, 1.0, 2.0], &gradients[..2], 0)
            .is_err());
    }

    #[test]
    fn the_curvature_of_a_polynomial_profile_is_found() {
        // E(s) = 0.3 - 0.2 s + 0.7 s^2 + 0.1 s^3 along x, images at s = -0.4, 0, 0.5.
        let energy = |s: f64| 0.3 - 0.2 * s + 0.7 * s * s + 0.1 * s * s * s;
        let slope = |s: f64| -0.2 + 1.4 * s + 0.3 * s * s;
        let at = |s: f64| col(&[s, 0.0, 0.0]);
        let gradient = |s: f64| col(&[slope(s), 0.5, -0.5]);
        let (p, c, n) = (at(-0.4), at(0.0), at(0.5));
        let (gp, gc, gn) = (gradient(-0.4), gradient(0.0), gradient(0.5));
        let found = path_curvature(
            [&p, &c, &n],
            [energy(-0.4), energy(0.0), energy(0.5)],
            [&gp, &gc, &gn],
            &col(&[1.0, 0.0, 0.0]),
        );
        assert!((found - 1.4).abs() < 1e-10, "{found}");
        assert_eq!(
            path_curvature([&c, &c, &n], [0.0; 3], [&gp, &gc, &gn], &c),
            0.0
        );
    }
}
