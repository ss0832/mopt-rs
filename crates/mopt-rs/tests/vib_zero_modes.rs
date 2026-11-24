//! The zero modes of a vibrational analysis with frozen atoms: on a surface blind to rigid
//! motions, the rigid motions that keep the frozen atoms in place (rotations about one frozen
//! atom, the rotation about the line through two) are zero modes of the constrained surface;
//! they are projected out, so that only vibrations are left and the count of imaginary modes
//! (`vib::is_imaginary_curvature`, the one criterion of the crate) sees no zero mode. A
//! molecule, a linear molecule, and a slab, with one, two, and three frozen atoms.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::Mat;
use mopt_rs::cli::freqmain::constraint_surface;
use mopt_rs::domain::{
    positions, Angstrom, AtomId, Bohr, Element, EvalRequest, MolecularSystem, PeriodicCell,
    Quantity, SpinCharge, Vector3,
};
use mopt_rs::math::projection::rigid_motions_keeping;
use mopt_rs::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use mopt_rs::pes::native::TruncatedLennardJones;
use mopt_rs::pes::Calculator;
use mopt_rs::vib::{self, Direction};

fn argon(xyz: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); xyz.len()],
        &positions::<Angstrom>(xyz),
        SpinCharge::default(),
        cell,
    )
    .unwrap()
}

/// The modes left on the surface of `frozen` atoms of `sys` with the Lennard-Jones surface,
/// with the rigid motions that keep them projected out (`keep`) or not, and the number of
/// those motions.
fn constrained_modes(
    sys: &MolecularSystem,
    frozen: &[usize],
    keep: bool,
) -> (vib::NormalModes, usize) {
    let calc = TruncatedLennardJones::new(Quantity::<Bohr>::new(15.0), None).unwrap();
    let out = calc.evaluate(sys, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
    let hessian: Mat<f64> = out.hessian_bohr.unwrap();
    let atoms: Vec<AtomId> = frozen.iter().map(|&a| AtomId::new(a)).collect();
    let resolved = Constraint::at_start(ConstraintKind::Frozen(atoms))
        .resolve(sys, None)
        .unwrap();
    let set = ConstraintSet::new(&[resolved], sys, None).unwrap();
    let (lagrangian, directions) =
        constraint_surface(&set, sys, &out.gradient_bohr, &hessian, keep).unwrap();
    let kept = directions
        .iter()
        .filter(|d| matches!(d, Direction::Displacement(_)))
        .count();
    (
        vib::transverse_modes(sys, &lagrangian, false, &directions).unwrap(),
        kept,
    )
}

/// An argon tetrahedron near its minimum (edges 3.82 Angstrom, slightly uneven): the gradient
/// is small but not zero, as at a converged structure.
fn tetrahedron() -> MolecularSystem {
    let a = 3.82 / 2f64.sqrt();
    argon(
        &[
            [0.0, 0.0, 0.0],
            [a, a, 0.0],
            [a, 0.0, a + 0.002],
            [0.0, a - 0.001, a],
        ],
        None,
    )
}

#[test]
fn frozen_atoms_leave_no_zero_mode_among_the_vibrations() {
    let sys = tetrahedron();
    for (frozen, kept_expected) in [(vec![0], 3), (vec![0, 1], 1), (vec![0, 1, 2], 0)] {
        let (modes, kept) = constrained_modes(&sys, &frozen, true);
        let (bare, _) = constrained_modes(&sys, &frozen, false);
        eprintln!(
            "frozen {frozen:?}: kept {kept}, with: {:?}, without: {:?}",
            modes.curvatures, bare.curvatures
        );
        assert_eq!(kept, kept_expected, "{frozen:?}");
        // 3N - 3f - (the kept rigid motions) vibrations, all of them far from zero.
        assert_eq!(modes.wavenumbers.len(), 12 - 3 * frozen.len() - kept);
        assert!(
            modes.curvatures.iter().all(|k| k.abs() > 1e-6),
            "{:?}",
            modes.curvatures
        );
        assert_eq!(modes.imaginary_modes(), 0);
        // Without them the rotations about the frozen atoms are among the modes, with
        // curvatures that are zero up to the gradient.
        let flat = bare.curvatures.iter().filter(|k| k.abs() < 1e-6).count();
        assert_eq!(flat, kept_expected, "{:?}", bare.curvatures);
    }
}

#[test]
fn rigid_motions_keeping_a_linear_molecule_and_a_slab() {
    // A linear Ar3 with its first atom frozen: the two rotations about that atom.
    let linear = argon(&[[0.0; 3], [0.0, 0.0, 3.82], [0.0, 0.0, 7.64]], None);
    let n = linear.dof();
    let unit = |k: usize| faer::Col::from_fn(n, |i| f64::from(u8::from(i == k)));
    let rows: Vec<_> = (0..3).map(unit).collect();
    assert_eq!(rigid_motions_keeping(&linear, &rows).len(), 2);
    let (modes, kept) = constrained_modes(&linear, &[0], true);
    assert_eq!(kept, 2);
    // 9 - 3 - 2: two stretches and two bends (the bends of a straight Lennard-Jones chain are
    // soft but real: they come from the attraction of the outer pair).
    assert_eq!(modes.wavenumbers.len(), 4, "{:?}", modes.curvatures);
    assert_eq!(modes.imaginary_modes(), 0, "{:?}", modes.curvatures);
    // A slab (periodic in x and y) with one frozen atom: its three translations are all the
    // rigid motions, and the frozen atom keeps none of them.
    let d = 3.8;
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([d, 0.0, 0.0]),
        Vector3::<Angstrom>::new([0.0, d, 0.0]),
        Vector3::<Angstrom>::new([0.0, 0.0, 30.0]),
        [true, true, false],
    )
    .unwrap();
    let slab = argon(
        &[[0.0, 0.0, 5.0], [0.5 * d, 0.5 * d, 5.0 + d / 2f64.sqrt()]],
        Some(cell),
    );
    let (modes, kept) = constrained_modes(&slab, &[0], true);
    assert_eq!(kept, 0);
    assert_eq!(modes.wavenumbers.len(), 3);
    assert_eq!(modes.imaginary_modes(), 0, "{:?}", modes.wavenumbers);
}
