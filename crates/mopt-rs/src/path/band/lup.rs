//! The locally updated planes method (LUP) as a variant of the chain relaxation: every inner
//! image moves in the hyperplane through it perpendicular to the path, down the true force
//! across the path, `F_i = -(g_i - (g_i . t_i) t_i)`, with no spring force and no climbing
//! image (R. Elber, I. Ulitsky, J. Chem. Phys. 1990, 92, 1510; R. Elber, C. Choi, J. Chem.
//! Phys. 1991, 94, 751). The planes are updated with the path: the tangent `t_i` is taken anew
//! in every iteration from the images as they are.
//!
//! In the chain machinery ([`super::machine::ChainRelaxation`]) that is the force assembly
//! without springs, without climbing, and with the projection of whole images: what
//! [`locally_updated_planes`] makes of any settings. The tangent is the upwind tangent of
//! Henkelman and Jonsson (the tangent of the machinery; Elber and Ulitsky take the chord
//! between the two neighbours, which agrees with it to first order in the spacing on a
//! smooth path), and the steps are those of the chain (the steps along the forces, mixed
//! with quasi-Newton steps across the path where the images have Hessians, which are
//! minimizations in the planes as well).
//!
//! The spacing of the images: the original method has no springs, and an image moves only
//! in its plane, so that images can slide along a curved path towards the basins (and
//! bunch up where the path bends). The chain machinery can put them back along the path
//! between iterations (its redistributions); whether `nebmain -lup` does so by default was
//! decided by measurement (`docs/source/reaction_paths.md`, "The locally updated planes").
//!
//! MultiOptPy: `-lup` is declared in `interface.py` (`LUP`), and its code has no
//! implementation of its own (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::machine::ChainSettings;
use crate::error::{MoptError, MoptResult};

/// `base` as the locally updated planes: no springs, no climbing image, no ridge, the
/// projection of whole images (the redistributions and the steps of `base` are kept).
pub fn locally_updated_planes(base: ChainSettings) -> ChainSettings {
    ChainSettings {
        springs: None,
        climb_after: None,
        ridge: None,
        block: None,
        ..base
    }
}

/// Refuse what the locally updated planes do not have, named by its option: springs
/// (`-ewbneb`), a climbing image (`-cineb`), a ridge, the projection atom by atom.
pub fn check_locally_updated_planes(settings: &ChainSettings) -> MoptResult<()> {
    let reason = if settings.springs.is_some() {
        Some("they have no springs (-ewbneb)")
    } else if settings.climb_after.is_some() {
        Some("no image climbs (-cineb)")
    } else if settings.ridge.is_some() {
        Some("they relax a path between minima, not along a ridge")
    } else if settings.block.is_some() {
        Some("the planes are those of whole images (not projection=atom, -qsm)")
    } else {
        None
    };
    match reason {
        Some(r) => Err(MoptError::config(format!(
            "-lup: in the locally updated planes {r}"
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use faer::Col;

    use super::*;
    use crate::math::linalg::{col_dot, col_norm};
    use crate::path::band::machine::{ChainAction, ChainEnd, ChainRelaxation, ImageValue};
    use crate::path::band::springs::EnergyWeightedSprings;
    use crate::path::band::trust::FixedEnds;
    use crate::path::band::ChainKind;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    /// A valley along the parabola `y = x^2` with a barrier at `x = 0` (the saddle at the
    /// origin): `E = 0.1 ((y - x^2)^2 + 0.3 exp(-4 x^2))`.
    fn valley(p: &Col<f64>) -> ImageValue {
        let (x, y) = (p[0], p[1]);
        let off = y - x * x;
        let bump = 0.3 * libm::exp(-4.0 * x * x);
        ImageValue::new(
            0.1 * (off * off + bump),
            col(&[0.1 * (-4.0 * x * off - 8.0 * x * bump), 0.1 * 2.0 * off]),
        )
    }

    #[test]
    fn the_planes_relax_onto_the_path_through_the_saddle() {
        let chain: Vec<Col<f64>> = (0..9)
            .map(|k| col(&[-1.0 + 0.25 * k as f64, 1.0]))
            .collect();
        for redistributed in [None, Some(5)] {
            let settings = locally_updated_planes(ChainSettings {
                kind: ChainKind::Variables,
                max_iterations: 2000,
                ends: FixedEnds::from_code(3),
                equal_spacing_every: redistributed,
                force_tolerance: Some(1e-7),
                springs: Some(EnergyWeightedSprings::default()),
                climb_after: Some(3),
                ..ChainSettings::default()
            });
            check_locally_updated_planes(&settings).unwrap();
            let mut m = ChainRelaxation::new(settings, chain.clone()).unwrap();
            let mut last = None;
            while let ChainAction::Evaluate(request) = m.ask() {
                last = Some(m.tell(request.images.iter().map(valley).collect()).unwrap());
            }
            assert_eq!(m.end(), Some(ChainEnd::Converged), "{redistributed:?}");
            let last = last.unwrap();
            // No force across the path is left at any inner image (the force of an image is
            // its true force across the path alone), and no image climbed.
            assert!(last.climbing.is_empty());
            assert!(last.largest_force < 1e-7, "{redistributed:?}");
            for k in 1..last.chain.len() - 1 {
                assert!(last.force_max[k] < 1e-7, "{redistributed:?} image {k}");
            }
            // The path passes through the saddle at the origin: the polygon of the images
            // comes within a small distance of it.
            let nearest = last
                .chain
                .windows(2)
                .map(|w| {
                    let d = &w[1] - &w[0];
                    let s = (-col_dot(&w[0], &d) / col_dot(&d, &d)).clamp(0.0, 1.0);
                    col_norm(&(&w[0] + &d * s))
                })
                .fold(f64::INFINITY, f64::min);
            assert!(nearest < 2e-2, "{redistributed:?}: {nearest}");
        }
        // What the planes do not have is refused by name.
        let springs = ChainSettings {
            springs: Some(EnergyWeightedSprings::default()),
            ..ChainSettings::default()
        };
        let err = check_locally_updated_planes(&springs)
            .unwrap_err()
            .to_string();
        assert!(err.contains("-ewbneb"), "{err}");
    }
}
