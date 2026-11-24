//! One quasi-Newton step for the whole chain: all images as one variable (experimental).
//!
//! The steps of [`ChainRfo::steps`] are found image by image, each with the model across
//! the chain of that image alone; what an image does to its neighbours (its move turns
//! their directions along the chain) enters as a stiffening `G / h` of the image itself,
//! as if the neighbours stood still. That keeps neighbours from swinging against each
//! other, but it also holds back what the images do together: a stretch of the chain that
//! lies beside the path as a whole is moved with the curvature `H + G / h` although
//! nothing turns when its images move alike.
//!
//! Here the chain is one vector `X` of all images that move, with the model
//!
//! ```text
//! m(dX) = sum_i [ g_i . dx_i + dx_i . (P_i H_i P_i + Q_i) dx_i / 2 ]
//!       + sum_i (G_i / h_i) |P_i (dx_i - dx_u(i))|^2 / 2
//! ```
//!
//! (`g_i`: the gradient across the chain; `H_i`: the Hessian of the surface at image `i`;
//! `Q_i`, `P_i = 1 - Q_i`: the projections on and across the direction along the chain;
//! `u(i)`: the neighbour uphill of `i`, the one its direction points to; climbing images
//! have `P H P - Q H Q` and no second term, end images that are minimized `H`). The
//! second term is the turning of the directions written for the difference of the moves
//! of two neighbours: it vanishes when they move alike and is the stiffening of
//! [`ChainRfo::steps`] when the neighbour stands still. Written like this the model has
//! a symmetric Hessian (the forces of a chain are no gradient of a function, and their
//! own Jacobian is not symmetric), which a rational-function step needs.
//!
//! The step is that of the rational function: `dX = x / a` with the lowest eigenvector
//! `(x, a)` of the augmented matrix `[[H, g], [g^T, 0]]`. The Hessian of the chain has as
//! many rows as the chain has coordinates (images times atoms times three) and is never
//! formed: Davidson's method ([`crate::math::davidson`]) needs its action on a vector
//! only, which is computed image by image from the Hessians of the images. Those stay
//! where they are kept (in the files `tmp_hessian_<i>.npy` when the chain has a
//! directory); as many of them as fit into [`GlobalRfoSettings::memory_mb`] are held in
//! memory during one solve, the others are read for every application. The memory of a
//! solve is therefore the Hessians that are held, plus the vectors of the search (at most
//! 2 x 20 of the length of the chain).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::rfo::{Along, AlongChain, ChainRfo};
use super::trust::FixedEnds;
use crate::error::{MoptError, MoptResult};
use crate::math::davidson::{lowest_eigenpair, DavidsonSettings};
use crate::math::linalg::col_norm;
use crate::math::projection::build_tr_rot_basis;
use crate::opt::update::apply_update;

/// Settings of [`ChainRfo::global_steps`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalRfoSettings {
    /// Megabytes of Hessians of images that are held in memory during one solve (the
    /// others are read from their files for every application of the Hessian of the
    /// chain; without files all are in memory anyway).
    pub memory_mb: f64,
    /// Residual of the eigenpair at which the solve ends, as a share of the norm of the
    /// gradient of the chain.
    pub relative_tolerance: f64,
    /// Applications of the Hessian of the chain at most in one solve.
    pub max_applications: usize,
    /// Vectors of the search before it is collapsed.
    pub max_subspace: usize,
}

impl Default for GlobalRfoSettings {
    fn default() -> Self {
        Self {
            memory_mb: 256.0,
            relative_tolerance: 1e-4,
            max_applications: 100,
            max_subspace: 20,
        }
    }
}

/// What the last solve did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalSolve {
    /// Coordinates of the chain that move.
    pub dimension: usize,
    /// Applications of the Hessian of the chain.
    pub applications: usize,
    /// Whether the eigenpair was found to the tolerance.
    pub converged: bool,
    /// The lowest eigenvalue of the augmented matrix (the shift of the step).
    pub shift: f64,
    /// Hessians of images that were held in memory, and those that were read for every
    /// application.
    pub held: usize,
    /// See `held`.
    pub read: usize,
}

/// The model of one image for a solve (no Hessian).
struct ImageModel {
    /// Where the image starts in the vector of the chain (`None`: it does not move).
    offset: Option<usize>,
    /// The direction along the chain (inner images).
    frame: Option<Along>,
    climbing: bool,
    /// `G_k / h_k` of the blocks of the frame.
    stiffening: Vec<f64>,
    /// The neighbour the direction points to, when it moves and the image has a slope.
    uphill: Option<usize>,
    /// Translations and rotations of the structure.
    rigid: Option<Mat<f64>>,
    /// Gradient of the model.
    gradient: Col<f64>,
    /// Approximate diagonal of the Hessian of the model.
    diagonal: Col<f64>,
}

/// `v` without translations and rotations.
fn without_rigid(rigid: &Option<Mat<f64>>, v: &Col<f64>) -> Col<f64> {
    match rigid {
        Some(basis) => v - basis * (basis.transpose() * v),
        None => v.clone(),
    }
}

impl ImageModel {
    /// The Hessian of the model of this image alone applied to `v` (without the
    /// stiffening): `P H P + Q`, `P H P - Q H Q` for a climbing image, `H` for an end.
    fn apply(&self, hessian: &Mat<f64>, v: &Col<f64>) -> Col<f64> {
        let Some(frame) = &self.frame else {
            return hessian * v;
        };
        let along = frame.part(v);
        let across = v - &along;
        let image = hessian * &across;
        let mut out = &image - frame.part(&image);
        if self.climbing {
            out -= frame.part(&(hessian * &along));
        } else {
            out += along;
        }
        out
    }
}

impl ChainRfo {
    /// What the last call of [`Self::global_steps`] did (`None` before the first).
    pub fn last_solve(&self) -> Option<GlobalSolve> {
        self.solve
    }

    /// The quasi-Newton step of every image from one model of the whole chain (see the
    /// module documentation); arguments as for [`Self::steps`], and which ends are fixed.
    pub fn global_steps(
        &mut self,
        chain: &[Col<f64>],
        energies: &[f64],
        gradients: &[Col<f64>],
        along: AlongChain<'_>,
        ends: FixedEnds,
        settings: &GlobalRfoSettings,
    ) -> MoptResult<Vec<Col<f64>>> {
        let count = self.images.len();
        if chain.len() != count
            || energies.len() != count
            || gradients.len() != count
            || along.directions.len() != count
        {
            return Err(MoptError::config(format!(
                "the quasi-Newton steps were set up for {count} images"
            )));
        }
        let n = self.n;
        // The Hessians are updated, and the models of the images are set up.
        let mut models = Vec::with_capacity(count);
        let mut dimension = 0;
        for i in 0..count {
            let end = i == 0 || i == count - 1;
            let fixed = (i == 0 && ends.first) || (i == count - 1 && ends.last);
            let slot = &mut self.images[i];
            let mut surface = slot.hessian.take()?;
            if let Some((x, g)) = slot.last.take() {
                let step = &chain[i] - &x;
                let y = &gradients[i] - &g;
                apply_update(slot.updater.as_mut(), &mut surface, &step, &y, false)?;
            }
            let mut diagonal = Col::from_fn(n, |r| surface[(r, r)]);
            slot.hessian.put(surface)?;
            slot.last = Some((chain[i].clone(), gradients[i].clone()));
            slot.iteration += 1;
            let rigid = match &self.template {
                Some(template) => Some(build_tr_rot_basis(
                    &template.with_flat_coords_bohr(&chain[i])?,
                    false,
                )),
                None => None,
            };
            let climbing = along.climbing.contains(&i);
            let (frame, gradient, stiffening, uphill) = if end {
                (None, gradients[i].clone(), Vec::new(), None)
            } else {
                let block = if climbing { n } else { self.rule.block };
                let frame = Along::new(&along.directions[i], block);
                let part = frame.part(&gradients[i]);
                let gradient = if climbing {
                    &gradients[i] - 2.0 * &part
                } else {
                    &gradients[i] - &part
                };
                let rising = energies[i - 1] < energies[i] && energies[i] < energies[i + 1];
                let falling = energies[i - 1] > energies[i] && energies[i] > energies[i + 1];
                let (stiffening, uphill) = if climbing || !(rising || falling) {
                    (Vec::new(), None)
                } else {
                    (
                        frame.stiffening(&gradients[i], rising),
                        Some(if rising { i + 1 } else { i - 1 }),
                    )
                };
                for (column, &s) in stiffening.iter().enumerate().take(frame.starts.len()) {
                    for r in frame.starts[column]..(frame.starts[column] + frame.block).min(n) {
                        diagonal[r] += s;
                    }
                }
                (Some(frame), gradient, stiffening, uphill)
            };
            let offset = if fixed {
                None
            } else {
                dimension += n;
                Some(dimension - n)
            };
            models.push(ImageModel {
                offset,
                frame,
                climbing,
                stiffening,
                uphill,
                gradient: without_rigid(&rigid, &gradient),
                rigid,
                diagonal,
            });
        }
        // A neighbour that does not move leaves the stiffening on the image alone.
        for i in 0..count {
            if models[i].uphill.is_some_and(|u| models[u].offset.is_none()) {
                models[i].uphill = None;
            }
        }
        if dimension == 0 {
            return Ok(vec![Col::zeros(n); count]);
        }
        // Hessians that are held in memory during the solve.
        let bytes = (8 * n * n) as f64;
        let budget = (settings.memory_mb * 1e6 / bytes).floor().max(0.0) as usize;
        let mut cache: Vec<Option<Mat<f64>>> = Vec::with_capacity(count);
        let (mut held, mut read) = (0, 0);
        for (i, model) in models.iter().enumerate() {
            let slot = &self.images[i];
            if model.offset.is_none() || slot.hessian.in_memory().is_some() {
                cache.push(None);
                held += usize::from(model.offset.is_some());
            } else if held < budget {
                cache.push(Some(slot.hessian.copy()?));
                held += 1;
            } else {
                cache.push(None);
                read += 1;
            }
        }
        let mut gradient = Col::<f64>::zeros(dimension);
        let mut diagonal = Col::<f64>::zeros(dimension + 1);
        for model in &models {
            if let Some(offset) = model.offset {
                for r in 0..n {
                    gradient[offset + r] = model.gradient[r];
                    diagonal[offset + r] = model.diagonal[r];
                }
            }
        }
        let images = &self.images;
        let part = |v: &Col<f64>, model: &ImageModel| -> Col<f64> {
            match model.offset {
                Some(offset) => Col::from_fn(n, |r| v[offset + r]),
                None => Col::zeros(n),
            }
        };
        // The augmented matrix [[H, g], [g^T, 0]] applied to (x, a).
        let mut apply = |v: &Col<f64>| -> MoptResult<Col<f64>> {
            let scale = v[dimension];
            let raw: Vec<Col<f64>> = models.iter().map(|model| part(v, model)).collect();
            let moves: Vec<Col<f64>> = models
                .iter()
                .zip(&raw)
                .map(|(model, x)| without_rigid(&model.rigid, x))
                .collect();
            let mut out: Vec<Col<f64>> = vec![Col::zeros(n); count];
            for (i, model) in models.iter().enumerate() {
                if model.offset.is_none() {
                    continue;
                }
                let own = match (images[i].hessian.in_memory(), &cache[i]) {
                    (Some(hessian), _) | (None, Some(hessian)) => model.apply(hessian, &moves[i]),
                    (None, None) => model.apply(&images[i].hessian.copy()?, &moves[i]),
                };
                out[i] += own;
                if let Some(frame) = &model.frame {
                    let stiff = frame.stiffened(&model.stiffening, &moves[i]);
                    out[i] += &stiff;
                    if let Some(u) = model.uphill {
                        let other = frame.stiffened(&model.stiffening, &moves[u]);
                        out[i] -= &other;
                        out[u] += other - stiff;
                    }
                }
            }
            let mut result = Col::<f64>::zeros(dimension + 1);
            let mut last = 0.0;
            for (i, model) in models.iter().enumerate() {
                let Some(offset) = model.offset else {
                    continue;
                };
                // Translations and rotations keep their length: they get no step.
                let image = without_rigid(&model.rigid, &out[i]) + (&raw[i] - &moves[i]);
                for r in 0..n {
                    result[offset + r] = image[r] + scale * gradient[offset + r];
                    last += gradient[offset + r] * v[offset + r];
                }
            }
            result[dimension] = last;
            Ok(result)
        };
        let size = col_norm(&gradient);
        let start = Col::from_fn(dimension + 1, |r| {
            if r == dimension {
                1.0
            } else {
                -gradient[r] / diagonal[r].abs().max(1e-2)
            }
        });
        let pair = lowest_eigenpair(
            &mut apply,
            Some(&diagonal),
            &start,
            &DavidsonSettings {
                max_iterations: settings.max_applications.max(2),
                tolerance: (settings.relative_tolerance * size).max(1e-14),
                max_subspace: settings.max_subspace,
            },
        )?;
        self.solve = Some(GlobalSolve {
            dimension,
            applications: pair.applications,
            converged: pair.converged,
            shift: pair.value,
            held,
            read,
        });
        let scale = pair.vector[dimension];
        let mut steps = Vec::with_capacity(count);
        for model in &models {
            let Some(offset) = model.offset else {
                steps.push(Col::zeros(n));
                continue;
            };
            let mut step = if scale.abs() > 1e-10 {
                Col::from_fn(n, |r| pair.vector[offset + r] / scale)
            } else {
                // No component along the gradient: straight down the gradient instead.
                -&model.gradient
            };
            step = without_rigid(&model.rigid, &step);
            if let (Some(frame), false) = (&model.frame, model.climbing) {
                step -= frame.part(&step);
            }
            let limit = if model.frame.is_some() {
                self.settings.inner_step
            } else {
                self.settings.end_step
            };
            let length = col_norm(&step);
            if length > limit {
                step *= limit / length;
            }
            steps.push(step);
        }
        Ok(steps)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::path::band::force::UpwindProjection;
    use crate::path::band::rfo::ChainRfoSettings;
    use crate::path::band::Compatibility;

    const SLOPE: f64 = 0.3;
    const ACROSS: f64 = 2.0;

    /// A chain of five images along `x` in the valley `E = SLOPE x + ACROSS y^2 / 2`, all
    /// beside the floor by `off`: energies, gradients, directions to the next image, and
    /// the components of the gradients along them.
    #[allow(clippy::type_complexity)]
    fn beside(
        off: f64,
    ) -> (
        Vec<Col<f64>>,
        Vec<f64>,
        Vec<Col<f64>>,
        Vec<Col<f64>>,
        Vec<Col<f64>>,
    ) {
        let col = |x: f64, y: f64| Col::from_fn(2, |k| [x, y][k]);
        let chain: Vec<_> = (0..5).map(|i| col(i as f64, off)).collect();
        let energies = (0..5)
            .map(|i| SLOPE * i as f64 + 0.5 * ACROSS * off * off)
            .collect();
        let gradients = vec![col(SLOPE, ACROSS * off); 5];
        let directions = vec![col(1.0, 0.0); 5];
        let removed = vec![col(SLOPE, 0.0); 5];
        (chain, energies, gradients, directions, removed)
    }

    fn minimizers(settings: ChainRfoSettings) -> ChainRfo {
        let rule = UpwindProjection {
            block: 2,
            climb_after: None,
            compatibility: Compatibility::Consistent,
        };
        let mut rfo = ChainRfo::new(settings, 5, 2, None, rule).unwrap();
        let hessian = Mat::from_fn(2, 2, |i, j| if i == 1 && j == 1 { ACROSS } else { 0.0 });
        for i in 0..5 {
            rfo.set_hessian(i, hessian.clone()).unwrap();
        }
        rfo
    }

    #[test]
    fn images_that_lie_beside_the_path_together_are_moved_together() {
        let off = 1e-5;
        let (chain, energies, gradients, directions, removed) = beside(off);
        let along = AlongChain {
            removed: &removed,
            directions: &directions,
            climbing: &[],
        };
        let fixed = FixedEnds {
            first: true,
            last: true,
        };
        // One image at a time: every image has the curvature of the valley plus the
        // stiffening SLOPE / 1 of the turning of its direction.
        let mut apart = minimizers(ChainRfoSettings::default());
        let alone = apart.steps(&chain, &energies, &gradients, along).unwrap();
        let single = -ACROSS * off / (ACROSS + SLOPE);
        for step in &alone[1..4] {
            assert!(step[0].abs() < 1e-12 && (step[1] - single).abs() < 1e-9 * off);
        }
        // The whole chain: the stiffening acts on the difference of the moves of
        // neighbours. With the stiffness k = SLOPE between the images 1-2, 2-3 and from 3
        // to the fixed end: (c + k) s1 - k s2 = (c + 2k) s2 - k s1 - k s3
        //   = (c + 2k) s3 - k s2 = -c off.
        let mut whole = minimizers(ChainRfoSettings::default());
        let settings = GlobalRfoSettings {
            relative_tolerance: 1e-10,
            ..GlobalRfoSettings::default()
        };
        let together = whole
            .global_steps(&chain, &energies, &gradients, along, fixed, &settings)
            .unwrap();
        let (c, k) = (ACROSS, SLOPE);
        let matrix = Mat::from_fn(3, 3, |i, j| {
            [
                [c + k, -k, 0.0],
                [-k, c + 2.0 * k, -k],
                [0.0, -k, c + 2.0 * k],
            ][i][j]
        });
        let rhs = Mat::from_fn(3, 1, |_, _| -c * off);
        let solved = faer::linalg::solvers::Solve::solve(&matrix.partial_piv_lu(), rhs.as_ref());
        for i in 0..3 {
            let step = &together[i + 1];
            assert!(step[0].abs() < 1e-12, "{}", step[0]);
            assert!(
                (step[1] - solved[(i, 0)]).abs() < 1e-6 * off,
                "{} against {}",
                step[1],
                solved[(i, 0)]
            );
            // Nearer to the floor than one image at a time.
            assert!(step[1] < single);
        }
        assert_eq!(col_norm(&together[0]), 0.0);
        assert_eq!(col_norm(&together[4]), 0.0);
        let solve = whole.last_solve().unwrap();
        assert!(solve.converged && solve.dimension == 6 && solve.applications <= 8);
        assert!(solve.shift <= 0.0);
    }

    #[test]
    fn hessians_in_files_are_read_within_the_memory_given() {
        let off = 1e-3;
        let (chain, energies, gradients, directions, removed) = beside(off);
        let along = AlongChain {
            removed: &removed,
            directions: &directions,
            climbing: &[],
        };
        let free = FixedEnds {
            first: false,
            last: true,
        };
        let mut in_memory = minimizers(ChainRfoSettings::default());
        let settings = GlobalRfoSettings::default();
        let expected = in_memory
            .global_steps(&chain, &energies, &gradients, along, free, &settings)
            .unwrap();
        assert_eq!(in_memory.last_solve().unwrap().read, 0);
        let dir = std::env::temp_dir().join(format!("mopt_global_rfo_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // 2 x 2 Hessians of 32 bytes: room for two of the four images that move.
        for (memory_mb, held, read) in [(0.0, 0, 4), (64e-6, 2, 2), (1.0, 4, 0)] {
            let mut in_files = minimizers(ChainRfoSettings {
                directory: Some(dir.clone()),
                ..ChainRfoSettings::default()
            });
            let steps = in_files
                .global_steps(
                    &chain,
                    &energies,
                    &gradients,
                    along,
                    free,
                    &GlobalRfoSettings {
                        memory_mb,
                        ..settings
                    },
                )
                .unwrap();
            let solve = in_files.last_solve().unwrap();
            assert_eq!((solve.held, solve.read), (held, read));
            for (a, b) in steps.iter().zip(&expected) {
                assert!(col_norm(&(a - b)) < 1e-14);
            }
        }
        // The first image is an end that is minimized: it goes down the slope, no farther
        // than the longest step of an end.
        assert!(expected[0][0] < 0.0 && col_norm(&expected[0]) <= 0.2 + 1e-12);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_the_highest_image_down_the_step_of_the_neighbour_uphill_is_counted() {
        let off = 1e-5;
        let (chain, energies, gradients, directions, removed) = beside(off);
        let along = AlongChain {
            removed: &removed,
            directions: &directions,
            climbing: &[],
        };
        let mut swept = minimizers(ChainRfoSettings {
            sweep: true,
            ..ChainRfoSettings::default()
        });
        let steps = swept.steps(&chain, &energies, &gradients, along).unwrap();
        // The last image is an end that is minimized: across the valley it goes to the
        // floor (-off). The image below it counts on that move, which leaves its own
        // direction as it is, and goes to the floor as well, and so on down the chain:
        // (c + k) s_i = -c off + k s_(i+1) = -(c + k) off. One image at a time each
        // would go -c off / (c + k) only.
        for (i, step) in steps.iter().enumerate().take(4).skip(1) {
            assert!((step[1] + off).abs() < 1e-6 * off, "image {i}: {}", step[1]);
            assert!(step[0].abs() < 1e-12);
        }
        let mut apart = minimizers(ChainRfoSettings::default());
        let alone = apart.steps(&chain, &energies, &gradients, along).unwrap();
        assert!((alone[2][1] + ACROSS * off / (ACROSS + SLOPE)).abs() < 1e-6 * off);
    }
}
