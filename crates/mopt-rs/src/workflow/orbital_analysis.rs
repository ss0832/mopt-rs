//! Orbital analyses that need a calculator: the orbitals of a structure and of structures
//! displaced along a direction are asked for, and `analysis::orbitals` does the rest.
//!
//! MultiOptPy: `WFAnalyzer/MO_analysis.py:NROAnalysis.run`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::analysis::orbitals::nro::{
    mass_weighted_displacement, natural_reaction_orbitals, response_matrix, NaturalReactionOrbitals,
};
use crate::analysis::orbitals::tracking::{follow, DEFAULT_DEGENERACY};
use crate::domain::{MolecularSystem, Orbitals};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_norm;
use crate::math::projection::build_tr_rot_basis;
use crate::pes::calculator::Calculator;

/// Displacement along the direction for the differences of the orbitals, in amu^(1/2)
/// Bohr. See `docs/source/migration_vibrations.md` ("Natural reaction orbitals") for the measurement.
pub const DEFAULT_ORBITAL_STEP: f64 = 0.01;

/// The natural reaction orbitals of one set of orbitals of a structure.
#[derive(Debug, Clone)]
pub struct ReactionOrbitalSet {
    /// `alpha`, `beta`, or `closed shell`.
    pub spin: &'static str,
    /// The pairs and their singular values (amu^(-1/2) Bohr^(-1)).
    pub orbitals: NaturalReactionOrbitals,
    /// The smallest overlap of an occupied orbital with what followed it at the displaced
    /// structures; well below 1, the differences are not to be trusted. (How the empty
    /// orbitals mix among themselves does not enter the pairs.)
    pub weakest_overlap: f64,
}

/// The orbitals of `sys` from `calc`, or an error that says which calculators have them.
pub fn orbitals_of(calc: &dyn Calculator, sys: &MolecularSystem) -> MoptResult<Orbitals> {
    calc.orbitals(sys)?.ok_or_else(|| {
        MoptError::unsupported(format!(
            "calculator '{}' does not hand out molecular orbitals (the built-in gfn0xtb, \
             PySCF, and tblite do)",
            calc.name()
        ))
    })
}

/// `direction` (a Cartesian vector) without the translations and rotations of `sys`, which
/// the orbitals of a structure follow rigidly; periodic structures keep theirs.
pub fn internal_direction(sys: &MolecularSystem, direction: &Col<f64>) -> MoptResult<Col<f64>> {
    let n = 3 * sys.n_atoms();
    if direction.nrows() != n {
        return Err(MoptError::config(format!(
            "a direction of {} components for {} atoms",
            direction.nrows(),
            sys.n_atoms()
        )));
    }
    if sys.has_pbc() {
        return Ok(direction.clone());
    }
    // In mass-weighted coordinates the rigid motions are orthogonal to the vibrations.
    let roots: Vec<f64> = sys
        .elements
        .iter()
        .map(|e| libm::sqrt(e.mass_amu()))
        .collect();
    let weighted = Col::from_fn(n, |k| roots[k / 3] * direction[k]);
    let basis = build_tr_rot_basis(sys, true);
    let along = basis.transpose() * &weighted;
    let cleaned = &weighted - &basis * along;
    if col_norm(&cleaned) <= 1e-8 * col_norm(&weighted) {
        return Err(MoptError::config(
            "the direction is a translation or rotation of the whole structure, along which the orbitals do not change",
        ));
    }
    Ok(Col::from_fn(n, |k| cleaned[k] / roots[k / 3]))
}

/// The natural reaction orbitals of `sys` for a displacement along `direction` (a
/// Cartesian vector: the tangent of a path, a step, a gradient; its length does not
/// matter, its rigid-body part is removed). The orbitals are differentiated with
/// displacements of `step` (amu^(1/2) Bohr) to both sides: three calculations.
pub fn natural_reaction_orbitals_at(
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    direction: &Col<f64>,
    step: f64,
) -> MoptResult<Vec<ReactionOrbitalSet>> {
    let direction = internal_direction(sys, direction)?;
    let masses = sys.masses_amu();
    let dx = mass_weighted_displacement(&masses, &direction, step)?;
    let x = sys.coords_flat_bohr();
    let reference = orbitals_of(calc, sys)?;
    let plus = orbitals_of(calc, &sys.with_flat_coords_bohr(&(&x + &dx))?)?;
    let minus = orbitals_of(calc, &sys.with_flat_coords_bohr(&(&x - &dx))?)?;
    let (sets, forward, backward) = (reference.sets(), plus.sets(), minus.sets());
    if forward.len() != sets.len() || backward.len() != sets.len() {
        return Err(MoptError::calc(
            "the displaced structures have another kind of orbitals (open against closed \
             shell) than the structure itself",
        ));
    }
    sets.iter()
        .zip(forward.iter().zip(&backward))
        .map(|((spin, set), ((_, up), (_, down)))| {
            let response =
                response_matrix(set, up, down, &reference.overlap, step, DEFAULT_DEGENERACY)?;
            let orbitals = natural_reaction_orbitals(set, &response.u)?;
            Ok(ReactionOrbitalSet {
                spin,
                weakest_overlap: response.weakest_overlap(&orbitals.occupied_indices),
                orbitals,
            })
        })
        .collect()
}

/// Positions of the points of a path along it: the summed lengths of the steps from frame
/// to frame in mass-weighted coordinates (amu^(1/2) Bohr), the first frame at 0. The frames
/// are taken as they are (not superposed).
pub fn arc_lengths(frames: &[MolecularSystem]) -> MoptResult<Vec<f64>> {
    let mut s = vec![0.0; frames.len()];
    for k in 1..frames.len() {
        if frames[k].elements != frames[0].elements {
            return Err(MoptError::config(
                "the frames of a path must have the same atoms in the same order",
            ));
        }
        let masses = frames[k].masses_amu();
        let (a, b) = (
            frames[k - 1].coords_flat_bohr(),
            frames[k].coords_flat_bohr(),
        );
        let step: f64 = (0..a.nrows())
            .map(|i| masses[i / 3] * (b[i] - a[i]).powi(2))
            .sum();
        s[k] = s[k - 1] + libm::sqrt(step);
    }
    Ok(s)
}

/// The direction of a path at each of its frames: the line through the neighbouring
/// frames (through the frame and its one neighbour at the ends), as a Cartesian vector.
pub fn path_directions(frames: &[MolecularSystem]) -> MoptResult<Vec<Col<f64>>> {
    let n = frames.len();
    if n < 2 {
        return Err(MoptError::config("a path needs two frames at least"));
    }
    Ok((0..n)
        .map(|k| {
            let before = frames[k.saturating_sub(1)].coords_flat_bohr();
            let after = frames[(k + 1).min(n - 1)].coords_flat_bohr();
            after - before
        })
        .collect())
}

/// The natural reaction orbitals at every frame of a path, for the displacement along the
/// path there ([`path_directions`]).
pub fn natural_reaction_orbitals_along(
    calc: &dyn Calculator,
    frames: &[MolecularSystem],
    step: f64,
) -> MoptResult<Vec<Vec<ReactionOrbitalSet>>> {
    let directions = path_directions(frames)?;
    frames
        .iter()
        .zip(&directions)
        .enumerate()
        .map(|(k, (sys, direction))| {
            natural_reaction_orbitals_at(calc, sys, direction, step)
                .map_err(|e| e.context(format!("frame {k}")))
        })
        .collect()
}

/// The energies of the orbitals of one spin along a path, each orbital followed from frame
/// to frame.
#[derive(Debug, Clone)]
pub struct FollowedEnergies {
    /// `alpha`, `beta`, or `closed shell`.
    pub spin: &'static str,
    /// `energies[k][i]`: orbital `i` (in the order of the first frame) at frame `k`.
    pub energies: Vec<Vec<f64>>,
    /// Occupations at the first frame.
    pub occupations: Vec<f64>,
    /// For each frame after the first, the smallest overlap of an orbital with the one
    /// that followed it from the frame before.
    pub weakest_overlaps: Vec<f64>,
}

/// The orbital energies along a path, with the orbitals followed from each frame to the
/// next by their overlaps (in the atomic orbitals of the earlier frame).
pub fn orbital_energies_along(
    calc: &dyn Calculator,
    frames: &[MolecularSystem],
) -> MoptResult<Vec<FollowedEnergies>> {
    let first = frames
        .first()
        .ok_or_else(|| MoptError::config("no frames"))?;
    let start = orbitals_of(calc, first)?;
    let mut out: Vec<FollowedEnergies> = start
        .sets()
        .iter()
        .map(|(spin, set)| FollowedEnergies {
            spin,
            energies: vec![set.energies.clone()],
            occupations: set.occupations.clone(),
            weakest_overlaps: Vec::new(),
        })
        .collect();
    // The orbitals of the previous frame in the order of the first.
    let mut previous: Vec<crate::domain::OrbitalSet> =
        start.sets().iter().map(|(_, set)| (*set).clone()).collect();
    let mut overlap = start.overlap.clone();
    for (k, sys) in frames.iter().enumerate().skip(1) {
        let here = orbitals_of(calc, sys)?;
        let sets = here.sets();
        if sets.len() != out.len() {
            return Err(MoptError::calc(format!(
                "frame {k} has another kind of orbitals (open against closed shell) than the first"
            )));
        }
        for (channel, (_, set)) in sets.iter().enumerate() {
            let followed = follow(&previous[channel], set, &overlap, DEFAULT_DEGENERACY)
                .map_err(|e| e.context(format!("frame {k}")))?;
            out[channel]
                .energies
                .push(followed.orbitals.energies.clone());
            out[channel].weakest_overlaps.push(followed.weakest());
            previous[channel] = followed.orbitals;
        }
        overlap = here.overlap.clone();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Element, SpinCharge};
    use crate::pes::native::Gfn0Xtb;

    fn molecule(symbols: &[&str], angstrom: &[[f64; 3]], multiplicity: u32) -> MolecularSystem {
        let elements = symbols
            .iter()
            .map(|s| Element::parse_token(s).unwrap())
            .collect();
        let scale = crate::domain::Units::ANGSTROM_TO_BOHR;
        let flat = Col::from_fn(3 * angstrom.len(), |k| angstrom[k / 3][k % 3] * scale);
        let spin = SpinCharge {
            charge: 0,
            multiplicity,
        };
        MolecularSystem::from_flat_bohr(elements, &flat, spin, None).unwrap()
    }

    fn water() -> MolecularSystem {
        molecule(
            &["O", "H", "H"],
            &[
                [0.0, 0.0, 0.119],
                [0.0, 0.763, -0.477],
                [0.0, -0.763, -0.477],
            ],
            1,
        )
    }

    /// Stretch of one O-H bond.
    fn stretch(sys: &MolecularSystem) -> Col<f64> {
        let x = sys.coords_flat_bohr();
        Col::from_fn(9, |k| match k / 3 {
            1 => x[k] - x[k - 3],
            _ => 0.0,
        })
    }

    #[test]
    fn orbitals_of_the_built_in_method_are_orthonormal() {
        let sys = water();
        let orbitals = orbitals_of(&Gfn0Xtb::default(), &sys).unwrap();
        assert!(orbitals.beta.is_none());
        let set = &orbitals.alpha;
        // Eight valence electrons in eight orbitals (2s and 2p of oxygen, 1s and 2s of each
        // hydrogen): four doubly occupied.
        assert_eq!((set.len(), set.n_basis()), (8, 8));
        assert!((set.occupations.iter().sum::<f64>() - 8.0).abs() < 1e-8);
        assert!(set.energies.windows(2).all(|w| w[0] <= w[1]));
        let unit = set.coefficients.transpose() * &orbitals.overlap * &set.coefficients;
        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((unit[(i, j)] - expected).abs() < 1e-10, "{i} {j}");
            }
        }
        // A doublet has the same orbitals for both spins, filled differently.
        let radical = molecule(&["O", "H"], &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]], 2);
        let open = orbitals_of(&Gfn0Xtb::default(), &radical).unwrap();
        let beta = open.beta.as_ref().unwrap();
        assert!((open.alpha.occupations.iter().sum::<f64>() - 4.0).abs() < 1e-8);
        assert!((beta.occupations.iter().sum::<f64>() - 3.0).abs() < 1e-8);
        // A force field has no orbitals.
        let argon = molecule(&["Ar", "Ar"], &[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], 1);
        assert!(orbitals_of(&crate::pes::native::LennardJones, &argon).is_err());
    }

    #[test]
    fn pairs_along_a_bond_stretch() {
        let sys = water();
        let calc = Gfn0Xtb::default();
        let direction = stretch(&sys);
        let sets =
            natural_reaction_orbitals_at(&calc, &sys, &direction, DEFAULT_ORBITAL_STEP).unwrap();
        assert_eq!(sets.len(), 1);
        let set = &sets[0];
        assert_eq!(set.spin, "closed shell");
        assert!(set.weakest_overlap > 0.999_99, "{}", set.weakest_overlap);
        let nro = &set.orbitals;
        // Four occupied and four empty orbitals: four pairs.
        assert_eq!(nro.singular_values.len(), 4);
        assert!(nro.singular_values[0] > 1e-3 && nro.sum_of_squares > 0.0);
        // The same pairs for the opposite direction and for a longer vector.
        let reversed =
            natural_reaction_orbitals_at(&calc, &sys, &(&direction * -3.0), DEFAULT_ORBITAL_STEP)
                .unwrap();
        for (a, b) in nro
            .singular_values
            .iter()
            .zip(&reversed[0].orbitals.singular_values)
        {
            assert!((a - b).abs() < 1e-8 * a.max(1.0), "{a} {b}");
        }
        // Half the step gives the same values: the differences have converged.
        let finer =
            natural_reaction_orbitals_at(&calc, &sys, &direction, 0.5 * DEFAULT_ORBITAL_STEP)
                .unwrap();
        for (a, b) in nro
            .singular_values
            .iter()
            .zip(&finer[0].orbitals.singular_values)
        {
            assert!((a - b).abs() < 1e-4 * a + 1e-9, "{a} {b}");
        }
    }

    #[test]
    fn rigid_motion_moves_no_electrons() {
        let sys = water();
        let calc = Gfn0Xtb::default();
        // A translation and a rotation about z: nothing is left of them, so there is no
        // direction to analyze.
        let translation = Col::from_fn(9, |k| if k % 3 == 0 { 1.0 } else { 0.0 });
        let x = sys.coords_flat_bohr();
        let rotation = Col::from_fn(9, |k| match k % 3 {
            0 => -x[k + 1],
            1 => x[k - 1],
            _ => 0.0,
        });
        for rigid in [translation, rotation] {
            assert!(internal_direction(&sys, &rigid).is_err());
            assert!(natural_reaction_orbitals_at(&calc, &sys, &rigid, 0.01).is_err());
        }
        // A stretch with a translation added gives the pairs of the stretch.
        let direction = stretch(&sys);
        let shifted = Col::from_fn(9, |k| direction[k] + if k % 3 == 2 { 5.0 } else { 0.0 });
        let plain = natural_reaction_orbitals_at(&calc, &sys, &direction, 0.01).unwrap();
        let moved = natural_reaction_orbitals_at(&calc, &sys, &shifted, 0.01).unwrap();
        for (a, b) in plain[0]
            .orbitals
            .singular_values
            .iter()
            .zip(&moved[0].orbitals.singular_values)
        {
            assert!((a - b).abs() < 1e-7 * a.max(1.0), "{a} {b}");
        }
    }

    /// Singular values against the step, for the documentation: `cargo test -p mopt-rs
    /// --lib orbital_step_measurement -- --nocapture`. Every run checks what the table
    /// showed: at the default step the first singular value is within 1e-4 of its value
    /// at a step a hundred times smaller.
    #[test]
    fn orbital_step_measurement() {
        let calc = Gfn0Xtb::default();
        let ethylene = molecule(
            &["C", "C", "H", "H", "H", "H"],
            &[
                [0.0, 0.0, 0.667],
                [0.0, 0.0, -0.667],
                [0.0, 0.923, 1.238],
                [0.0, -0.923, 1.238],
                [0.0, 0.923, -1.238],
                [0.0, -0.923, -1.238],
            ],
            1,
        );
        let twist = {
            let x = ethylene.coords_flat_bohr();
            // The two hydrogens of the first carbon turn about the C-C axis.
            Col::from_fn(18, |k| match (k / 3, k % 3) {
                (2 | 3, 0) => x[k + 1],
                _ => 0.0,
            })
        };
        let radical = molecule(&["O", "H"], &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]], 2);
        let bond = Col::from_fn(6, |k| if k == 5 { 1.0 } else { 0.0 });
        let cases: [(&str, &MolecularSystem, Col<f64>); 3] = [
            ("water, O-H stretch", &water(), stretch(&water())),
            ("ethylene, CH2 twist", &ethylene, twist),
            ("OH radical, stretch", &radical, bond),
        ];
        for (name, sys, direction) in cases {
            println!("{name}");
            let mut firsts = Vec::new();
            for step in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 1e-4, 1e-5] {
                let sets = natural_reaction_orbitals_at(&calc, sys, &direction, step).unwrap();
                firsts.push(sets[0].orbitals.singular_values[0]);
                let text: Vec<String> = sets
                    .iter()
                    .map(|s| {
                        format!(
                            "{}: sum {:.10} first {:.10} overlap {:.8}",
                            s.spin,
                            s.orbitals.sum_of_squares,
                            s.orbitals.singular_values[0],
                            s.weakest_overlap
                        )
                    })
                    .collect();
                println!("  step {step:7}: {}", text.join(" | "));
            }
            // The steps 0.01 (the default) and 1e-4.
            let (default, small) = (firsts[4], firsts[8]);
            assert!(
                (default - small).abs() < 1e-4 * small.abs(),
                "{name}: {default} {small}"
            );
        }
    }
}
