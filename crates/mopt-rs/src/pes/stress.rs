//! Stress of periodic cells: conventions, conversions, and a numerical reference.
//!
//! The stress is `sigma_ab = (1 / V) dE / d(eps_ab)` for a homogeneous strain `eps` that
//! takes every position and lattice vector `r` to `(1 + eps) r`, in `Hartree / Bohr^3`: the
//! convention of ASE, negative under compression. ASE's own form is the six components
//! `xx, yy, zz, yz, xz, xy` in `eV / Angstrom^3`.
//!
//! With the lattice vectors as the rows of `h` and the fractional coordinates of the atoms
//! kept, `dE / dh = V h^-T sigma`, the gradient a variable-cell optimization needs.
//!
//! Slabs and wires have a stress in this sense too: the strain derivative of the energy
//! divided by the volume of the stored cell, whose vectors along the directions that are
//! not periodic are arbitrary. Only `V sigma` and `dE / dh` mean something there.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::domain::{
    positions, Bohr, EvalRequest, MolecularSystem, PeriodicCell, StrainHessian, Units, Vector3,
};
use crate::error::{MoptError, MoptResult};
use crate::pes::calculator::Calculator;
use crate::pes::term::{PesTerm, TermContext};

/// Stress tensor in `Hartree / Bohr^3`.
pub type Stress = [[f64; 3]; 3];

/// Strain of the central differences of [`numerical_stress`].
pub const DEFAULT_STRAIN_STEP: f64 = 1e-4;

/// `Hartree / Bohr^3` in `eV / Angstrom^3`.
fn ev_per_cubic_angstrom() -> f64 {
    Units::HARTREE_TO_EV / Units::BOHR_TO_ANGSTROM.powi(3)
}

/// The six components of ASE (`xx, yy, zz, yz, xz, xy`, `eV / Angstrom^3`) of a stress; the
/// tensor is made symmetric first.
pub fn to_ase_voigt(stress: &Stress) -> [f64; 6] {
    let c = ev_per_cubic_angstrom();
    let s = |a: usize, b: usize| 0.5 * (stress[a][b] + stress[b][a]) * c;
    [s(0, 0), s(1, 1), s(2, 2), s(1, 2), s(0, 2), s(0, 1)]
}

/// The stress from the six components of ASE.
pub fn from_ase_voigt(voigt: &[f64; 6]) -> Stress {
    let c = 1.0 / ev_per_cubic_angstrom();
    let [xx, yy, zz, yz, xz, xy] = voigt.map(|v| v * c);
    [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
}

/// The cell of `sys` if it is periodic in all three directions.
pub fn full_cell(sys: &MolecularSystem) -> Option<&PeriodicCell> {
    sys.active_cell()
        .filter(|c| c.periodic_axes.iter().all(|p| *p))
}

/// `sys` with every position and lattice vector `r` taken to `(1 + strain) r`.
pub fn strained(sys: &MolecularSystem, strain: &Stress) -> MoptResult<MolecularSystem> {
    let cell = sys
        .active_cell()
        .ok_or_else(|| MoptError::config("a strain needs a periodic cell"))?;
    let deform = |v: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|a| v[a] + (0..3).map(|b| strain[a][b] * v[b]).sum::<f64>())
    };
    let vectors: [[f64; 3]; 3] = std::array::from_fn(|k| deform(cell.lattice_vector(k).value()));
    let deformed = PeriodicCell::from_lattice_vectors(
        Vector3::<Bohr>::new(vectors[0]),
        Vector3::<Bohr>::new(vectors[1]),
        Vector3::<Bohr>::new(vectors[2]),
        cell.periodic_axes,
    )?;
    let rows: Vec<[f64; 3]> = sys
        .atoms()
        .map(|a| deform(sys.position(a).value()))
        .collect();
    MolecularSystem::new(
        sys.elements.clone(),
        &positions::<Bohr>(&rows),
        sys.spin_charge,
        Some(deformed),
    )
}

/// The stress of the term `term` at `sys`, whose output there is `out`: the stress of the
/// output where the term gives it (analytically), central differences of its energy in the
/// six strain components otherwise.
pub fn term_stress(
    term: &dyn PesTerm,
    out: &crate::domain::EvalOutput,
    sys: &MolecularSystem,
    ctx: &TermContext<'_>,
    step: f64,
) -> MoptResult<Stress> {
    match out.stress {
        Some(stress) => Ok(stress),
        None => numerical_term_stress(term, sys, ctx, step),
    }
}

/// The stress of a function of the lattice vectors alone from its gradient `of_lattice` with
/// respect to the lattice vectors (the rows of `h`, Bohr) at `cell`: under the strain `1 + e`
/// every lattice vector `v_k` goes to `(1 + e) v_k`, so `dE/de_ab = sum_k (dE/dv_k)_a (v_k)_b`;
/// over the volume, made symmetric.
pub fn lattice_stress(of_lattice: &[[f64; 3]; 3], cell: &PeriodicCell) -> Stress {
    let volume = cell.volume_bohr3();
    let vectors: [[f64; 3]; 3] = std::array::from_fn(|k| cell.lattice_vector(k).value());
    let raw: Stress = std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            (0..3)
                .map(|k| of_lattice[k][a] * vectors[k][b])
                .sum::<f64>()
        })
    });
    std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (raw[a][b] + raw[b][a]) / volume))
}

/// The second derivatives along the strain ([`StrainHessian`]) of the energy and gradient
/// that `eval` gives at structures like `sys`, from central differences: of the energy for
/// the block of the strain (four energies for every pair of components, three on the
/// diagonal), of the gradient for the mixed block, all at the strain `step`. For checks of the
/// analytic ones.
pub fn numerical_strain_hessian(
    sys: &MolecularSystem,
    step: f64,
    eval: &dyn Fn(&MolecularSystem) -> MoptResult<(f64, Col<f64>)>,
) -> MoptResult<StrainHessian> {
    let at = |moves: &[(usize, f64)]| -> MoptResult<(f64, Col<f64>)> {
        let mut strain = [[0.0; 3]; 3];
        for (k, v) in moves {
            strain[k / 3][k % 3] += v;
        }
        eval(&strained(sys, &strain)?)
    };
    let mut out = StrainHessian::zeros(sys.dof());
    let e0 = at(&[])?.0;
    for x in 0..9 {
        let (plus, minus) = (at(&[(x, step)])?, at(&[(x, -step)])?);
        out.strain[x][x] = (plus.0 - 2.0 * e0 + minus.0) / (step * step);
        for i in 0..sys.dof() {
            out.mixed[(i, x)] = (plus.1[i] - minus.1[i]) / (2.0 * step);
        }
        for y in x + 1..9 {
            let corner = |sx: f64, sy: f64| -> MoptResult<f64> {
                Ok(at(&[(x, sx * step), (y, sy * step)])?.0)
            };
            let v = (corner(1.0, 1.0)? - corner(1.0, -1.0)? - corner(-1.0, 1.0)?
                + corner(-1.0, -1.0)?)
                / (4.0 * step * step);
            out.strain[x][y] = v;
            out.strain[y][x] = v;
        }
    }
    Ok(out)
}

/// The stress of `sys` from central differences of the energy of `calc` in the six strain
/// components (twelve energies).
pub fn numerical_stress(
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    step: f64,
) -> MoptResult<Stress> {
    numerical_stress_of_state(calc, sys, 0, step)
}

/// [`numerical_stress`] of the electronic state `excited_state` of `sys` (its charge and
/// multiplicity are those of `sys`).
pub fn numerical_stress_of_state(
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    excited_state: usize,
    step: f64,
) -> MoptResult<Stress> {
    let cell = sys
        .active_cell()
        .ok_or_else(|| MoptError::config("the stress needs a periodic cell"))?;
    let volume = cell.volume_bohr3();
    let request = EvalRequest {
        excited_state,
        ..EvalRequest::ENERGY_ONLY
    };
    let energy = |strain: &Stress| -> MoptResult<f64> {
        Ok(calc
            .evaluate(&strained(sys, strain)?, &request)?
            .energy
            .value())
    };
    let mut stress = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in a..3 {
            // A symmetric strain with eps_ab = eps_ba = step / 2 off the diagonal: the
            // difference gives (sigma_ab + sigma_ba) / 2.
            let mut plus = [[0.0; 3]; 3];
            let share = if a == b { 1.0 } else { 0.5 };
            plus[a][b] += share * step;
            plus[b][a] += if a == b { 0.0 } else { share * step };
            let minus = plus.map(|row| row.map(|x| -x));
            let value = (energy(&plus)? - energy(&minus)?) / (2.0 * step * volume);
            stress[a][b] = value;
            stress[b][a] = value;
        }
    }
    Ok(stress)
}

/// The contribution of an in-process term (a bias potential) to the stress of `sys`, from
/// central differences of its energy in the six strain components.
pub fn numerical_term_stress(
    term: &dyn PesTerm,
    sys: &MolecularSystem,
    ctx: &TermContext<'_>,
    step: f64,
) -> MoptResult<Stress> {
    let cell = sys
        .active_cell()
        .ok_or_else(|| MoptError::config("the stress needs a periodic cell"))?;
    let volume = cell.volume_bohr3();
    let energy = |strain: &Stress| -> MoptResult<f64> {
        Ok(term
            .evaluate(&strained(sys, strain)?, ctx, false)?
            .energy
            .value())
    };
    let mut stress = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in a..3 {
            let mut plus = [[0.0; 3]; 3];
            let share = if a == b { 1.0 } else { 0.5 };
            plus[a][b] += share * step;
            plus[b][a] += if a == b { 0.0 } else { share * step };
            let minus = plus.map(|row| row.map(|x| -x));
            let value = (energy(&plus)? - energy(&minus)?) / (2.0 * step * volume);
            stress[a][b] = value;
            stress[b][a] = value;
        }
    }
    Ok(stress)
}

/// The stress of `sys`: the calculator's own where it gives one, the numerical one
/// otherwise.
pub fn stress_of(calc: &dyn Calculator, sys: &MolecularSystem) -> MoptResult<Stress> {
    match calc.evaluate(sys, &EvalRequest::ENERGY_GRAD_STRESS)?.stress {
        Some(stress) => Ok(stress),
        None => numerical_stress(calc, sys, DEFAULT_STRAIN_STEP),
    }
}

/// `dE / dh` for the lattice vectors as the rows of `h` (Bohr), the fractional coordinates
/// of the atoms kept: `V h^-T sigma`, in `Hartree / Bohr`.
///
/// The contraction takes the stress as symmetric (`sigma_ba` for `sigma_ab`), which the
/// stress of every surface that does not depend on the orientation of the cell is. A term of
/// the Cartesian components of the lattice vectors (a `-ubias` or `-uc` expression with
/// `cell_ax()`, ...) has an antisymmetric part; its gradient is
/// `opt::cell::lattice_gradient`, which contracts `sigma_ab` itself and gives the bits of this
/// function for a symmetric stress.
pub fn cell_gradient(stress: &Stress, cell: &PeriodicCell) -> [[f64; 3]; 3] {
    let volume = cell.volume_bohr3();
    // Row i of the stored inverse is the reciprocal vector of lattice vector i
    // (b_i . a_j = delta_ij): dE / dh_ia = V sum_b (b_i)_b sigma_ba.
    let reciprocal = cell.h_inv_bohr;
    std::array::from_fn(|i| {
        std::array::from_fn(|a| {
            volume * (0..3).map(|b| reciprocal[i][b] * stress[b][a]).sum::<f64>()
        })
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::quantity::shorthand::bohr;
    use crate::domain::{Angstrom, Element, SpinCharge};
    use crate::pes::native::{Eam, Emt, Gfnff, Meam, Tersoff, TruncatedLennardJones};

    /// Four atoms in a skewed cell, off their sites.
    fn cell_of(symbol: &str, edge: f64) -> MolecularSystem {
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new([edge, 0.0, 0.0]),
            Vector3::<Angstrom>::new([0.06 * edge, 0.97 * edge, 0.0]),
            Vector3::<Angstrom>::new([-0.04 * edge, 0.05 * edge, 1.03 * edge]),
            [true, true, true],
        )
        .unwrap();
        let h = edge / 2.0;
        MolecularSystem::new(
            vec![Element::parse_token(symbol).unwrap(); 4],
            &positions::<Angstrom>(&[
                [0.05, 0.02, -0.03],
                [h + 0.04, h - 0.06, 0.02],
                [h - 0.03, 0.05, h + 0.07],
                [-0.02, h + 0.03, h - 0.05],
            ]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap()
    }

    fn close(found: &Stress, expected: &Stress, tolerance: f64) {
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (found[a][b] - expected[a][b]).abs() < tolerance,
                    "({a}, {b}): {} against {}",
                    found[a][b],
                    expected[a][b]
                );
            }
        }
    }

    /// The largest differences of the two blocks of `found` and `expected`, each against
    /// the largest entry of its block in `expected`.
    fn strain_differences(found: &StrainHessian, expected: &StrainHessian) -> (f64, f64) {
        let scale = expected
            .strain
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
            .max(1e-12);
        let strain = found
            .strain
            .iter()
            .flatten()
            .zip(expected.strain.iter().flatten())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
            / scale;
        let (mut worst, mut largest) = (0.0_f64, 1e-12_f64);
        for i in 0..expected.mixed.nrows() {
            for k in 0..9 {
                largest = largest.max(expected.mixed[(i, k)].abs());
                worst = worst.max((found.mixed[(i, k)] - expected.mixed[(i, k)]).abs());
            }
        }
        (strain, worst / largest)
    }

    #[test]
    fn the_hessian_in_the_positions_and_the_lattice_gives_the_strain_hessian() {
        // Two argon atoms in a skewed cell: the Hessian of the energy in the 6 positions and
        // the 9 components of the lattice vectors by central differences of the energy,
        // turned into the strain Hessian, against the analytic one of the potential.
        let argon = TruncatedLennardJones::new(bohr(12.0), Some(bohr(2.0))).unwrap();
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new([4.1, 0.0, 0.0]),
            Vector3::<Angstrom>::new([0.3, 3.9, 0.0]),
            Vector3::<Angstrom>::new([-0.2, 0.25, 4.3]),
            [true, true, true],
        )
        .unwrap();
        let sys = MolecularSystem::new(
            vec![Element::parse_token("Ar").unwrap(); 2],
            &positions::<Angstrom>(&[[0.1, 0.2, -0.1], [2.2, 1.7, 2.4]]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap();
        let n = sys.dof();
        let base: Vec<f64> = sys
            .coords_flat_bohr()
            .iter()
            .copied()
            .chain((0..3).flat_map(|k| sys.cell.as_ref().unwrap().lattice_vector(k).value()))
            .collect();
        let energy = |v: &[f64]| -> f64 {
            let vectors: [[f64; 3]; 3] =
                std::array::from_fn(|k| std::array::from_fn(|c| v[n + 3 * k + c]));
            let moved = PeriodicCell::from_lattice_vectors(
                Vector3::<Bohr>::new(vectors[0]),
                Vector3::<Bohr>::new(vectors[1]),
                Vector3::<Bohr>::new(vectors[2]),
                [true, true, true],
            )
            .unwrap();
            let rows: Vec<[f64; 3]> = (0..2)
                .map(|a| std::array::from_fn(|c| v[3 * a + c]))
                .collect();
            let s = MolecularSystem::new(
                sys.elements.clone(),
                &positions::<Bohr>(&rows),
                SpinCharge::default(),
                Some(moved),
            )
            .unwrap();
            argon
                .evaluate(&s, &EvalRequest::ENERGY_ONLY)
                .unwrap()
                .energy
                .value()
        };
        let h = 1e-3;
        let size = n + 9;
        let w = faer::Mat::from_fn(size, size, |u, v| {
            let at = |su: f64, sv: f64| {
                let mut x = base.clone();
                x[u] += su * h;
                x[v] += sv * h;
                energy(&x)
            };
            (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * h * h)
        });
        let converted = StrainHessian::from_cell_hessian(&sys, &w).unwrap();
        let own = argon
            .evaluate(
                &sys,
                &EvalRequest {
                    need_stress: true,
                    ..EvalRequest::ENERGY_GRAD_HESS
                },
            )
            .unwrap()
            .strain_hessian
            .unwrap();
        let (strain, mixed) = strain_differences(&converted, &own);
        assert!(strain < 1e-4 && mixed < 1e-4, "{strain:e} {mixed:e}");
        // The wrong size is refused.
        assert!(StrainHessian::from_cell_hessian(&sys, &faer::Mat::zeros(n, n)).is_err());
    }

    #[test]
    fn the_strain_hessian_of_the_potentials_is_the_second_strain_derivative_of_their_energy() {
        let argon = TruncatedLennardJones::new(bohr(16.0), Some(bohr(2.0))).unwrap();
        let element = |symbol: &str| Element::parse_token(symbol).unwrap();
        let mut slab = cell_of("Cu", 3.55);
        if let Some(cell) = slab.cell.as_mut() {
            cell.periodic_axes = [true, true, false];
        }
        let pressed = strained(
            &cell_of("Fe", 3.5),
            &[[-0.06, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.03]],
        )
        .unwrap();
        // Two water molecules in a skewed cell (GFN-FF: bonds, hydrogen bonds, Ewald sums).
        let waters = {
            let cell = PeriodicCell::from_lattice_vectors(
                Vector3::<Angstrom>::new([5.6, 0.0, 0.0]),
                Vector3::<Angstrom>::new([0.4, 5.3, 0.0]),
                Vector3::<Angstrom>::new([-0.3, 0.2, 5.9]),
                [true, true, true],
            )
            .unwrap();
            MolecularSystem::new(
                ["O", "H", "H", "O", "H", "H"]
                    .iter()
                    .map(|e| element(e))
                    .collect(),
                &positions::<Angstrom>(&[
                    [0.10, 0.05, 0.02],
                    [0.95, 0.30, -0.05],
                    [-0.20, 0.88, 0.10],
                    [2.80, 0.20, 0.30],
                    [3.30, -0.55, 0.05],
                    [3.20, 0.60, 1.05],
                ]),
                SpinCharge::default(),
                Some(cell),
            )
            .unwrap()
        };
        let gfnff = Gfnff::default();
        let cases: [(&str, &dyn Calculator, MolecularSystem); 11] = [
            ("gfnff waters", &gfnff, waters),
            (
                "meam Cu",
                &Meam::library(&[element("Cu")]).unwrap(),
                cell_of("Cu", 3.5),
            ),
            (
                "meam Fe, screened",
                &Meam::library(&[element("Fe")]).unwrap(),
                pressed,
            ),
            (
                "meam Si",
                &Meam::library(&[element("Si")]).unwrap(),
                cell_of("Si", 3.4),
            ),
            (
                "meam Ni, smooth cutoff",
                &Meam::library(&[element("Ni")])
                    .unwrap()
                    .with_smooth_cutoff(true),
                cell_of("Ni", 3.45),
            ),
            (
                "tersoff",
                &Tersoff::tersoff_1989(&[element("Si")]).unwrap(),
                cell_of("Si", 3.9),
            ),
            ("emt", &Emt::new(), cell_of("Cu", 3.55)),
            ("emt slab", &Emt::new(), slab),
            (
                "eam Cu",
                &Eam::zhou2004(&[element("Cu")]).unwrap(),
                cell_of("Cu", 3.5),
            ),
            (
                "eam Fe",
                &Eam::zhou2004(&[element("Fe")]).unwrap(),
                cell_of("Fe", 3.7),
            ),
            ("lj", &argon, cell_of("Ar", 5.2)),
        ];
        let request = EvalRequest {
            need_stress: true,
            ..EvalRequest::ENERGY_GRAD_HESS
        };
        for (name, calc, sys) in cases {
            let own = calc
                .evaluate(&sys, &request)
                .unwrap()
                .strain_hessian
                .unwrap_or_else(|| panic!("{name}: no strain Hessian"));
            // Differences at two steps, extrapolated (Richardson): their error falls as the
            // fourth power of the step, below that of the second derivatives of MEAM.
            let at = |step: f64| {
                numerical_strain_hessian(&sys, step, &|s| {
                    let out = calc.evaluate(s, &EvalRequest::ENERGY_GRAD)?;
                    Ok((out.energy.value(), out.gradient_bohr))
                })
                .unwrap()
            };
            let (coarse, fine) = (at(4e-4), at(2e-4));
            let mut reference = fine.clone().scaled(4.0 / 3.0, 4.0 / 3.0);
            reference.add(&coarse.scaled(-1.0 / 3.0, -1.0 / 3.0));
            let (strain, mixed) = strain_differences(&own, &reference);
            assert!(
                strain < 1e-5 && mixed < 1e-5,
                "{name}: {strain:e} {mixed:e}"
            );
            // Not without the stress asked for.
            assert!(calc
                .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
                .unwrap()
                .strain_hessian
                .is_none());
        }
    }

    #[test]
    fn the_stress_of_the_potentials_is_the_strain_derivative_of_their_energy() {
        let argon = TruncatedLennardJones::new(bohr(16.0), None).unwrap();
        let element = |symbol: &str| Element::parse_token(symbol).unwrap();
        // Silicon close enough for the Tersoff cutoff function to act on some pairs, and
        // iron pressed along an axis so that second neighbours are screened in part (MEAM).
        let silicon = cell_of("Si", 3.9);
        let pressed = strained(
            &cell_of("Fe", 3.5),
            &[[-0.06, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.03]],
        )
        .unwrap();
        let cases: [(&dyn Calculator, MolecularSystem); 9] = [
            (
                &Meam::library(&[element("Cu")]).unwrap(),
                cell_of("Cu", 3.5),
            ),
            (&Meam::library(&[element("Fe")]).unwrap(), pressed),
            (
                &Meam::library(&[element("Si")]).unwrap(),
                cell_of("Si", 3.4),
            ),
            (
                &Meam::library(&[element("Ni")])
                    .unwrap()
                    .with_smooth_cutoff(true),
                cell_of("Ni", 3.45),
            ),
            (&Tersoff::tersoff_1989(&[element("Si")]).unwrap(), silicon),
            (&Emt::new(), cell_of("Cu", 3.55)),
            (
                &Eam::zhou2004(&[Element::parse_token("Cu").unwrap()]).unwrap(),
                cell_of("Cu", 3.5),
            ),
            (
                &Eam::zhou2004(&[Element::parse_token("Fe").unwrap()]).unwrap(),
                cell_of("Fe", 3.7),
            ),
            (&argon, cell_of("Ar", 5.2)),
        ];
        for (calc, sys) in cases {
            let own = calc
                .evaluate(&sys, &EvalRequest::ENERGY_GRAD)
                .unwrap()
                .stress
                .unwrap();
            let reference = numerical_stress(calc, &sys, 1e-4).unwrap();
            let scale = reference
                .iter()
                .flatten()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            close(&own, &reference, 1e-5 * scale.max(1e-6));
            // Symmetric.
            let asymmetry = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| (own[a][b] - own[b][a]).abs())
                .fold(0.0_f64, f64::max);
            assert!(asymmetry < 1e-12 * scale.max(1e-6), "{asymmetry}");
            assert_eq!(stress_of(calc, &sys).unwrap(), own);
        }
        // A compressed crystal pushes outwards: negative stress, as in ASE.
        let squeezed = cell_of("Cu", 3.3);
        let stress = stress_of(&Emt::new(), &squeezed).unwrap();
        assert!(stress[0][0] < 0.0 && stress[1][1] < 0.0 && stress[2][2] < 0.0);
        // A slab: the strain derivative per volume of the stored cell, again that of the
        // energy.
        let mut slab = cell_of("Cu", 3.55);
        if let Some(cell) = slab.cell.as_mut() {
            cell.periodic_axes = [true, true, false];
        }
        let own = stress_of(&Emt::new(), &slab).unwrap();
        let reference = numerical_stress(&Emt::new(), &slab, 1e-4).unwrap();
        close(&own, &reference, 1e-8);
        let lone = MolecularSystem::new(
            vec![Element::parse_token("Cu").unwrap(); 2],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert!(Emt::new()
            .evaluate(&lone, &EvalRequest::ENERGY_GRAD)
            .unwrap()
            .stress
            .is_none());
        assert!(numerical_stress(&Emt::new(), &lone, 1e-4).is_err());
    }

    #[test]
    fn the_components_of_ase_and_the_gradient_of_the_cell() {
        let stress = [
            [1.0e-4, 2.0e-5, -3.0e-5],
            [2.0e-5, -4.0e-5, 5.0e-6],
            [-3.0e-5, 5.0e-6, 7.0e-5],
        ];
        let voigt = to_ase_voigt(&stress);
        // 1 Hartree / Bohr^3 = 183.63 eV / Angstrom^3.
        assert!((voigt[0] / stress[0][0] - 183.631_531).abs() < 1e-3);
        assert!((voigt[3] / stress[1][2] - voigt[0] / stress[0][0]).abs() < 1e-9);
        close(&from_ase_voigt(&voigt), &stress, 1e-18);
        // dE / dh against differences of the energy with the fractional coordinates kept.
        let calc = Emt::new();
        let sys = cell_of("Cu", 3.5);
        let cell = sys.active_cell().unwrap().clone();
        let gradient = cell_gradient(&stress_of(&calc, &sys).unwrap(), &cell);
        let fractions: Vec<[f64; 3]> = sys
            .atoms()
            .map(|a| cell.cart_to_frac(sys.position(a).value()))
            .collect();
        let energy = |i: usize, a: usize, shift: f64| -> f64 {
            let mut vectors: [[f64; 3]; 3] =
                std::array::from_fn(|k| cell.lattice_vector(k).value());
            vectors[i][a] += shift;
            let moved = PeriodicCell::from_lattice_vectors(
                Vector3::<Bohr>::new(vectors[0]),
                Vector3::<Bohr>::new(vectors[1]),
                Vector3::<Bohr>::new(vectors[2]),
                [true, true, true],
            )
            .unwrap();
            let rows: Vec<[f64; 3]> = fractions.iter().map(|f| moved.frac_to_cart(*f)).collect();
            let system = MolecularSystem::new(
                sys.elements.clone(),
                &positions::<Bohr>(&rows),
                sys.spin_charge,
                Some(moved),
            )
            .unwrap();
            calc.evaluate(&system, &EvalRequest::ENERGY_ONLY)
                .unwrap()
                .energy
                .value()
        };
        for (i, row) in gradient.iter().enumerate() {
            for (a, expected) in row.iter().enumerate() {
                let slope = (energy(i, a, 1e-4) - energy(i, a, -1e-4)) / 2e-4;
                assert!(
                    (slope - expected).abs() < 1e-8,
                    "({i}, {a}): {slope} against {expected}"
                );
            }
        }
    }
}
