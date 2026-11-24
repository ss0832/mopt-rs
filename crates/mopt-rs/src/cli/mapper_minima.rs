//! Every minimum of `run_mapper` verified by its Hessian (`verify_minima`, on by default).
//!
//! At convergence the mapper computes the Hessian, projects out rigid motions where
//! appropriate, and returns the signed vibrational wavenumbers. Mode kill runs only when
//! requested. The saddle index uses the curvature tolerance of `opt::mode_kill` (about
//! 0.51 cm^-1). IRC endpoints use the wavenumbers of those same modes.
//! The wavenumbers of its vibrations (without the rigid motions where they are projected out)
//! go onto the node of the network; those of a transition state come the same way from the
//! Hessian of the start check of its IRC ([`vibrations`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's mapper takes every converged
//!  end of an IRC as a minimum).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::math::projection::build_tr_rot_basis;
use crate::opt::mode_kill::{examine, ModeKillSettings, ModeKillVerdict};
use crate::opt::state_machine::GeometryOptimizer;
use crate::pes::calculator::Calculator;
use crate::workflow::driver::CalculatorSet;

/// The mode kill of the mapper: `settings` (the defaults without them). Every count of
/// imaginary modes of the mapper and every mode kill of its optimizations takes the one
/// criterion of `opt::mode_kill`: eigenvalues of the mass-weighted Hessian below minus
/// [`crate::opt::mode_kill::IMAGINARY_CURVATURE`] (about 0.51 cm^-1), the same that gives the
/// wavenumbers on the nodes.
pub fn mapper_mode_kill(settings: Option<ModeKillSettings>) -> ModeKillSettings {
    settings.unwrap_or_default()
}

/// The masses of the coordinates of `sys` (amu, three per atom).
fn coordinate_masses(sys: &MolecularSystem) -> Vec<f64> {
    sys.masses_amu().iter().flat_map(|&m| [m, m, m]).collect()
}

/// What the Hessian at a converged structure says.
#[derive(Debug, Clone, PartialEq)]
pub enum MinimumCheck {
    /// A minimum, with the wavenumbers of its normal modes (cm^-1, imaginary ones negative).
    Minimum(Vec<f64>),
    /// The Hessian could not be had to convergence (the solves with the environment of an
    /// ONIOM core did not converge): the stability is not judged.
    Unjudged,
    /// Not a minimum: imaginary modes by the one criterion.
    Saddle {
        /// Imaginary modes.
        negative: usize,
        /// The lowest wavenumber (cm^-1, negative).
        lowest: f64,
        /// The wavenumbers of its vibrations, the imaginary ones negative.
        wavenumbers: Vec<f64>,
        /// Signed wavenumbers of exactly the modes counted by the index criterion.
        imaginary: Vec<f64>,
    },
}

/// Modes below the index tolerance, in the coordinates in which the check was made.
pub(super) fn index_modes(
    left: Vec<crate::opt::mode_kill::LeftMode>,
) -> Vec<crate::workflow::mapper::PointMode> {
    left.into_iter()
        .map(|mode| crate::workflow::mapper::PointMode {
            curvature: mode.curvature,
            wavenumber: Some(
                mode.wavenumber
                    .unwrap_or_else(|| crate::irc::eigenvalue_to_wavenumber(mode.curvature)),
            ),
            displacement: mode.direction,
        })
        .collect()
}

fn unstable_modes(verdict: ModeKillVerdict) -> Vec<crate::workflow::mapper::PointMode> {
    match verdict {
        ModeKillVerdict::Accept { .. } => Vec::new(),
        ModeKillVerdict::Displace { left, .. } => index_modes(left),
    }
}

/// The surface of the analysis at `sys`: the Hessian (that of the Lagrangian where `opt` holds
/// constraints, from the gradient and Hessian of its surface), whether the rigid motions are
/// left out, and the directions left out with them: the normals of the constraints and, where
/// they pin the frame, the rigid motions that keep them (the analysis of `freqmain` on the
/// surface of constraints, [`crate::cli::freqmain::constraint_surface`]).
fn analysed_surface(
    opt: &GeometryOptimizer,
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    gradient: &Col<f64>,
    hessian: Mat<f64>,
) -> MoptResult<(Mat<f64>, bool, Vec<crate::vib::Direction>)> {
    let rigid = calc.is_rigid_body_invariant();
    let resolved = opt.resolved_constraints();
    if resolved.is_empty() {
        return Ok((hessian, rigid, Vec::new()));
    }
    let set = crate::pes::constraint::ConstraintSet::new(resolved, sys, None)?;
    let (lagrangian, directions) =
        super::freqmain::constraint_surface(&set, sys, gradient, &hessian, rigid)?;
    // Constraints that pin the frame leave the rigid motions that keep them among the
    // directions; the others leave every rigid motion, projected as usual.
    Ok((lagrangian, rigid && !set.breaks_rigid_body(), directions))
}

/// The directions left out of the count of the imaginary modes (displacements, as
/// [`crate::opt::mode_kill::examine`] takes them): the rigid motions with `rigid`, and the
/// directions `held` of the constraints: a normal `n` (a gradient) as the displacement `M^-1 n`,
/// which is the normal `M^-1/2 n` of the mass-weighted space, a displacement as it is.
fn left_out(
    sys: &MolecularSystem,
    rigid: bool,
    held: &[crate::vib::Direction],
    masses: &[f64],
) -> Option<Mat<f64>> {
    let mut columns: Vec<Col<f64>> = Vec::new();
    if rigid {
        let basis = build_tr_rot_basis(sys, false);
        columns.extend((0..basis.ncols()).map(|k| Col::from_fn(basis.nrows(), |i| basis[(i, k)])));
    }
    columns.extend(held.iter().map(|d| match d {
        crate::vib::Direction::Gradient(n) => {
            Col::from_fn(n.nrows(), |i| n[i] / masses[i].max(1e-12))
        }
        crate::vib::Direction::Displacement(x) => x.clone(),
    }));
    let rows = sys.dof();
    (!columns.is_empty()).then(|| Mat::from_fn(rows, columns.len(), |i, k| columns[k][i]))
}

/// The wavenumbers of the vibrations on the surface of the constraints with the normals
/// `normals` ([`vibrations`] without any).
fn held_vibrations(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    rigid: bool,
    held: &[crate::vib::Direction],
) -> MoptResult<Vec<f64>> {
    if held.is_empty() {
        return vibrations(sys, hessian, rigid);
    }
    Ok(crate::vib::transverse_modes(sys, hessian, rigid, held)?.wavenumbers)
}

/// Check the converged structure `sys` of `opt` on its surface with `calc` (on the surface of
/// its constraints where it holds any: the Hessian of the Lagrangian in their tangent space).
pub fn check_minimum(
    opt: &mut GeometryOptimizer,
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    settings: &ModeKillSettings,
) -> MoptResult<MinimumCheck> {
    // One atom (a fragment of a dissociation channel) has no vibrations.
    if sys.n_atoms() == 1 && !sys.has_pbc() {
        return Ok(MinimumCheck::Minimum(Vec::new()));
    }
    let tasks = opt.hessian_check_tasks(sys)?;
    let results = CalculatorSet::single(calc).evaluate(&tasks)?;
    // A check of stability without a converged Hessian is not judged.
    if results.iter().any(|r| r.hessian_bohr.is_none()) {
        return Ok(MinimumCheck::Unjudged);
    }
    let (_, gradient, hessian) = opt.surface_hessian(sys, results)?;
    // In a cell the basis holds the translations along its periodic directions alone.
    let (hessian, rigid, normals) = analysed_surface(opt, calc, sys, &gradient, hessian)?;
    let masses = coordinate_masses(sys);
    let symmetries = left_out(sys, rigid, &normals, &masses);
    let unstable = unstable_modes(examine(
        &hessian,
        &gradient,
        0,
        settings,
        Some(&masses),
        symmetries.as_ref(),
    )?);
    let negative = unstable.len();
    let wavenumbers = held_vibrations(sys, &hessian, rigid, &normals)?;
    if negative > 0 {
        let imaginary: Vec<_> = unstable.iter().filter_map(|m| m.wavenumber).collect();
        let lowest = imaginary[0];
        return Ok(MinimumCheck::Saddle {
            negative,
            lowest,
            imaginary,
            wavenumbers,
        });
    }
    Ok(MinimumCheck::Minimum(wavenumbers))
}

/// The modes of the Hessian of the surface of `opt` at `sys` with `calc`, the lowest first:
/// the rigid motions taken out where the surface does not depend on them, and with `across`
/// the direction of the gradient as well; the number of imaginary modes by the one criterion
/// (`settings`), and the wavenumbers of the vibrations.
pub fn point_modes(
    opt: &mut GeometryOptimizer,
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    across: bool,
    settings: &ModeKillSettings,
) -> MoptResult<crate::workflow::mapper::PointModes> {
    if sys.n_atoms() == 1 && !sys.has_pbc() {
        return Ok(crate::workflow::mapper::PointModes::default());
    }
    let tasks = opt.hessian_check_tasks(sys)?;
    let results = CalculatorSet::single(calc).evaluate(&tasks)?;
    if results.iter().any(|r| r.hessian_bohr.is_none()) {
        return Err(crate::error::MoptError::conv(
            "the Hessian for the count of the imaginary modes could not be had to convergence \
             (the solves with the environment did not converge); the order of the point cannot \
             be judged",
        ));
    }
    let (_, gradient, hessian) = opt.surface_hessian(sys, results)?;
    // In a cell the basis holds the translations along its periodic directions alone.
    let (hessian, rigid, normals) = analysed_surface(opt, calc, sys, &gradient, hessian)?;
    let masses = coordinate_masses(sys);
    let symmetries = left_out(sys, rigid, &normals, &masses);
    let unstable = unstable_modes(examine(
        &hessian,
        &gradient,
        0,
        settings,
        Some(&masses),
        symmetries.as_ref(),
    )?);
    let negative = unstable.len();
    let mut directions = normals.clone();
    if across {
        directions.push(crate::vib::Direction::Gradient(gradient.clone()));
    }
    let modes = if rigid || !normals.is_empty() {
        crate::vib::transverse_modes(sys, &hessian, rigid, &directions)?
    } else {
        crate::vib::normal_modes_without(sys, &hessian, false, &directions)?
    };
    let mut list: Vec<crate::workflow::mapper::PointMode> = modes
        .curvatures
        .iter()
        .zip(&modes.wavenumbers)
        .zip(&modes.displacements)
        .map(|((&c, &w), d)| crate::workflow::mapper::PointMode {
            curvature: c,
            wavenumber: Some(w),
            displacement: d.clone(),
        })
        .collect();
    list.sort_by(|a, b| a.curvature.total_cmp(&b.curvature));
    Ok(crate::workflow::mapper::PointModes {
        unstable,
        modes: list,
        negative,
        wavenumbers: Some(held_vibrations(sys, &hessian, rigid, &normals)?),
    })
}

/// The modes of the Hessian `hessian` of an objective of plain variables at a point with the
/// gradient `gradient`, the lowest first (with `across` the direction of the gradient taken
/// out), and the number of imaginary modes by the one criterion with unit masses.
pub fn variable_modes(
    hessian: &Mat<f64>,
    gradient: &faer::Col<f64>,
    across: bool,
    settings: &ModeKillSettings,
) -> MoptResult<crate::workflow::mapper::PointModes> {
    let unstable = unstable_modes(examine(hessian, gradient, 0, settings, None, None)?);
    let negative = unstable.len();
    let list = coordinate_modes(hessian, gradient, across)?;
    // The wavenumbers of unit masses (the eigenvalues of the Hessian itself), as the IRC of a
    // point of plain variables gives its transition wavenumber.
    Ok(crate::workflow::mapper::PointModes {
        unstable,
        modes: list,
        negative,
        wavenumbers: Some(variable_wavenumbers(hessian)?),
    })
}

/// Eigenmodes in mass-weighted coordinates, optionally transverse to the weighted gradient.
pub(super) fn coordinate_modes(
    hessian: &Mat<f64>,
    gradient: &faer::Col<f64>,
    across: bool,
) -> MoptResult<Vec<crate::workflow::mapper::PointMode>> {
    let n = hessian.nrows();
    let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
    let unit = (across && norm > 0.0).then(|| faer::Col::from_fn(n, |i| gradient[i] / norm));
    let h = match &unit {
        // P H P with P = I - u u^T.
        Some(u) => {
            let p = Mat::from_fn(n, n, |i, j| f64::from(u8::from(i == j)) - u[i] * u[j]);
            &p * hessian * &p
        }
        None => hessian.clone(),
    };
    let (values, vectors) = crate::math::linalg::symmetric_eigen(&h)?;
    let mut list: Vec<crate::workflow::mapper::PointMode> = (0..n)
        .map(|k| crate::workflow::mapper::PointMode {
            curvature: values[k],
            wavenumber: Some(crate::irc::eigenvalue_to_wavenumber(values[k])),
            displacement: faer::Col::from_fn(n, |i| vectors[(i, k)]),
        })
        // The direction of the gradient itself is not a mode across the path.
        .filter(|m| {
            unit.as_ref().is_none_or(|u| {
                (0..n).map(|i| u[i] * m.displacement[i]).sum::<f64>().abs() < 0.9
            })
        })
        .collect();
    list.sort_by(|a, b| a.curvature.total_cmp(&b.curvature));
    Ok(list)
}

/// Signed vibrational wavenumbers for plain variables of unit mass, lowest first.
pub(super) fn variable_wavenumbers(hessian: &Mat<f64>) -> MoptResult<Vec<f64>> {
    let (values, _) = crate::math::linalg::symmetric_eigen(hessian)?;
    let mut wavenumbers: Vec<_> = values
        .iter()
        .map(|&v| crate::irc::eigenvalue_to_wavenumber(v))
        .collect();
    wavenumbers.sort_by(f64::total_cmp);
    Ok(wavenumbers)
}

/// The wavenumbers (cm^-1, imaginary ones negative) of the vibrations of `sys` from its
/// Cartesian Hessian: the 3N - 6 (3N - 5 for a linear structure) modes without the rigid
/// motions where the surface does not depend on them (`rigid`), as the rate theory `tst` takes
/// them; all 3N modes otherwise. One isolated atom has none.
pub fn vibrations(sys: &MolecularSystem, hessian: &Mat<f64>, rigid: bool) -> MoptResult<Vec<f64>> {
    if sys.n_atoms() == 1 && !sys.has_pbc() {
        return Ok(Vec::new());
    }
    let modes = if rigid {
        crate::vib::transverse_modes(sys, hessian, true, &[])?
    } else {
        crate::vib::normal_modes(sys, hessian, false)?
    };
    Ok(modes.wavenumbers)
}
