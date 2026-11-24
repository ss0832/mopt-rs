//! Entrance channels (`entrance_structures=N`): the structures a bimolecular reaction starts
//! from. The molecules of an input (the parts of its bond graph) are a dissociation channel;
//! `N` structures of that channel are made, every molecule kept rigid, turned by a rotation
//! drawn uniformly, and every molecule after the first placed in a direction drawn uniformly
//! from the centre of those placed before, at the distance at which its nearest atom to them is
//! at `contact` times the sum of the two van der Waals radii. Each structure is an input of the
//! run: minimized, it is a complex with the barrierless edge to the channel, or whatever it
//! falls to without a barrier.
//!
//! The generator is seeded by the seed of the run, so a restart makes the same structures.
//! The rules are in `docs/source/mapper_design.md` ("Entrance channels").
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::channels::molecules;
use crate::core_infra::rng::Rng;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::parameters::vdw::alvarez_vdw_radius_bohr;

/// How the structures of an entrance channel are made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntranceSettings {
    /// How many structures (`entrance_structures`; 0: none).
    pub structures: usize,
    /// The nearest atoms of two molecules are placed at this times the sum of their van der
    /// Waals radii (`entrance_contact`).
    pub contact: f64,
    /// Bonds within this factor times the sums of the covalent radii.
    pub bond_scale: f64,
    /// The seed of the rotations and directions.
    pub seed: u64,
}

/// The default of `entrance_contact`: the molecules touch (the sum of the van der Waals radii
/// of their nearest atoms).
pub const DEFAULT_ENTRANCE_CONTACT: f64 = 1.0;

/// The entrance of a network: the channel its jobs start from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entrance {
    /// The dissociation channel (its node).
    pub channel: usize,
    /// How many structures of it were made.
    pub structures: usize,
    /// The inputs that are those structures (indices into the inputs of the network).
    pub inputs: Vec<usize>,
}

/// The van der Waals radius of atom `i` (Bohr; 2 Angstrom for an element without one).
fn radius(system: &MolecularSystem, i: usize) -> f64 {
    alvarez_vdw_radius_bohr(system.elements[i]).unwrap_or(2.0 * Units::ANGSTROM_TO_BOHR)
}

fn centre(x: &[[f64; 3]], atoms: &[usize]) -> [f64; 3] {
    let n = atoms.len().max(1) as f64;
    std::array::from_fn(|c| atoms.iter().map(|&i| x[i][c]).sum::<f64>() / n)
}

/// The smallest ratio of a distance between an atom of `moved` (shifted by `shift`) and an atom
/// of `placed` to the sum of their van der Waals radii.
fn nearest_ratio(
    system: &MolecularSystem,
    x: &[[f64; 3]],
    moved: &[usize],
    shift: [f64; 3],
    placed: &[usize],
) -> f64 {
    let mut least = f64::INFINITY;
    for &i in moved {
        for &j in placed {
            let d: f64 = (0..3)
                .map(|c| (x[i][c] + shift[c] - x[j][c]).powi(2))
                .sum::<f64>()
                .sqrt();
            least = least.min(d / (radius(system, i) + radius(system, j)));
        }
    }
    least
}

/// The structures of the entrance channel of `system` (see the module text): none with
/// `structures` 0; an error when the structure is one molecule or lies in a periodic cell.
pub fn entrance_structures(
    system: &MolecularSystem,
    settings: &EntranceSettings,
) -> MoptResult<Vec<MolecularSystem>> {
    if settings.structures == 0 {
        return Ok(Vec::new());
    }
    if system.has_pbc() {
        return Err(MoptError::unsupported(
            "entrance_structures: an entrance channel of a periodic structure is not available",
        ));
    }
    if !(settings.contact.is_finite() && settings.contact > 0.0) {
        return Err(MoptError::parse(format!(
            "entrance_contact must be positive (got {})",
            settings.contact
        )));
    }
    let parts = molecules(system, settings.bond_scale);
    if parts.len() < 2 {
        return Err(MoptError::config(
            "entrance_structures: the input is one molecule (by its bonds); an entrance channel \
             needs the two or more molecules of a bimolecular reaction in one structure",
        ));
    }
    let flat = system.coords_flat_bohr();
    let start: Vec<[f64; 3]> = (0..system.n_atoms())
        .map(|i| std::array::from_fn(|c| flat[3 * i + c]))
        .collect();
    let mut rng = Rng::from_seed(settings.seed ^ 0x656e_7472_616e_6365);
    let mut out = Vec::with_capacity(settings.structures);
    for _ in 0..settings.structures {
        let mut x = start.clone();
        let mut placed: Vec<usize> = Vec::new();
        for (m, atoms) in parts.iter().enumerate() {
            // The molecule turned about its centre, the centre at the origin.
            let r = rng.rotation_matrix_3d();
            let c = centre(&start, atoms);
            for &i in atoms {
                let v: [f64; 3] = std::array::from_fn(|k| start[i][k] - c[k]);
                x[i] = std::array::from_fn(|k| (0..3).map(|l| r[k][l] * v[l]).sum());
            }
            if m > 0 {
                // Along a direction from the centre of the molecules placed so far: outwards
                // until nothing overlaps, then back to the contact by bisection.
                let from = centre(&x, &placed);
                let u = rng.unit_vector_3d();
                let at = |s: f64| -> [f64; 3] { std::array::from_fn(|k| from[k] + s * u[k]) };
                let ratio =
                    |s: f64| nearest_ratio(system, &x, atoms, at(s), &placed) - settings.contact;
                let mut high = 4.0 * Units::ANGSTROM_TO_BOHR;
                let mut doubled = 0;
                while ratio(high) < 0.0 && doubled < 40 {
                    high *= 2.0;
                    doubled += 1;
                }
                // The ratio need not grow all the way from the centre (a small molecule can
                // lie inside a hollow of the others): the outermost crossing is taken, from a
                // scan inwards in steps of a tenth of an Angstrom.
                let step = 0.1 * Units::ANGSTROM_TO_BOHR;
                let mut low = high;
                while low > 0.0 && ratio(low) >= 0.0 {
                    high = low;
                    low -= step;
                }
                let mut low = low.max(0.0);
                for _ in 0..60 {
                    let mid = 0.5 * (low + high);
                    if ratio(mid) >= 0.0 {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }
                let shift = at(high);
                for &i in atoms {
                    x[i] = std::array::from_fn(|k| x[i][k] + shift[k]);
                }
            }
            placed.extend(atoms.iter().copied());
        }
        let coords = Col::from_fn(3 * x.len(), |k| x[k / 3][k % 3]);
        out.push(system.with_flat_coords_bohr(&coords)?);
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};
    use crate::workflow::explore::identity::DEFAULT_BOND_SCALE;
    use crate::workflow::mapper::channels::{groups, ChannelSettings};

    fn water_and_ammonia() -> MolecularSystem {
        MolecularSystem::new(
            vec![
                Element::O,
                Element::H,
                Element::H,
                Element::N,
                Element::H,
                Element::H,
                Element::H,
            ],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.117],
                [0.0, 0.757, -0.469],
                [0.0, -0.757, -0.469],
                [8.0, 0.0, 0.0],
                [8.94, 0.0, -0.38],
                [7.53, 0.81, -0.38],
                [7.53, -0.81, -0.38],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    fn settings(structures: usize) -> EntranceSettings {
        EntranceSettings {
            structures,
            contact: DEFAULT_ENTRANCE_CONTACT,
            bond_scale: DEFAULT_BOND_SCALE,
            seed: 42,
        }
    }

    #[test]
    fn the_molecules_keep_their_shape_and_touch() {
        let sys = water_and_ammonia();
        let made = entrance_structures(&sys, &settings(12)).unwrap();
        assert_eq!(made.len(), 12);
        let parts = molecules(&sys, DEFAULT_BOND_SCALE);
        let distance = |s: &MolecularSystem, i: usize, j: usize| {
            let x = s.coords_flat_bohr();
            (0..3)
                .map(|c| (x[3 * i + c] - x[3 * j + c]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        for s in &made {
            // The same bonds, the same distances inside every molecule.
            assert_eq!(molecules(s, DEFAULT_BOND_SCALE), parts);
            for atoms in &parts {
                for &i in atoms {
                    for &j in atoms {
                        assert!((distance(s, i, j) - distance(&sys, i, j)).abs() < 1e-9);
                    }
                }
            }
            // The nearest atoms of the two molecules at the sum of their radii: in contact by
            // the test of the channels (one group).
            let x = s.coords_flat_bohr();
            let positions: Vec<[f64; 3]> = (0..s.n_atoms())
                .map(|i| std::array::from_fn(|c| x[3 * i + c]))
                .collect();
            let ratio = nearest_ratio(s, &positions, &parts[1], [0.0; 3], &parts[0]);
            assert!((ratio - 1.0).abs() < 1e-6, "{ratio}");
            let margin = ChannelSettings::default().margin;
            assert_eq!(groups(s, &parts, margin).len(), 1);
        }
        // Different orientations, and the same ones for the same seed.
        let first = made[0].coords_flat_bohr();
        let second = made[1].coords_flat_bohr();
        assert!((0..first.nrows()).any(|k| (first[k] - second[k]).abs() > 0.1));
        let again = entrance_structures(&sys, &settings(12)).unwrap();
        for (a, b) in made.iter().zip(&again) {
            let (xa, xb) = (a.coords_flat_bohr(), b.coords_flat_bohr());
            assert!((0..xa.nrows()).all(|k| xa[k] == xb[k]));
        }
        let other = entrance_structures(
            &sys,
            &EntranceSettings {
                seed: 43,
                ..settings(1)
            },
        )
        .unwrap();
        let xo = other[0].coords_flat_bohr();
        assert!((0..first.nrows()).any(|k| (first[k] - xo[k]).abs() > 1e-6));
    }

    #[test]
    fn one_molecule_has_no_entrance_channel() {
        let water = MolecularSystem::new(
            vec![Element::O, Element::H, Element::H],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.117],
                [0.0, 0.757, -0.469],
                [0.0, -0.757, -0.469],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert!(entrance_structures(&water, &settings(0))
            .unwrap()
            .is_empty());
        let e = entrance_structures(&water, &settings(3)).unwrap_err();
        assert!(e.to_string().contains("one molecule"), "{e}");
    }
}
