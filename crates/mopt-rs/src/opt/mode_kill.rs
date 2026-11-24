//! Mode kill at convergence: leave a stationary point that has more imaginary modes than were
//! asked for, and the one criterion of an imaginary mode.
//!
//! An optimization converges where the gradient vanishes, and a search for a saddle point of
//! order `k` (`k = 0`: a minimum) can end at a stationary point of higher order: by symmetry
//! (the gradient along a mode vanishes all the way), or because the Hessian in use never
//! showed the curvature of that mode. With this option the optimizer asks for the Hessian at
//! the converged structure, counts its imaginary modes, and when there are more than `k`
//! keeps the `k` lowest (the modes a saddle search climbs) and steps off along the others,
//! then goes on from there with that Hessian. The check is repeated at every convergence, so
//! the run ends at a stationary point of the order asked for, or after `max_rounds`
//! displacements.
//!
//! A mode is imaginary where the eigenvalue of the mass-weighted Hessian `M^(-1/2) H M^(-1/2)`
//! is below minus [`IMAGINARY_CURVATURE`] (MultiOptPy's `saddle_check`, about 0.51 cm^-1), with
//! the motions that are symmetries of the surface (translations and rotations) projected out
//! in the mass-weighted space. Every count of imaginary modes takes this criterion: the mode
//! kill of the optimizers, the minima and saddle points of the mapper, and the start of an IRC
//! (`crate::irc::settings`), so that what a report lists as an imaginary frequency is what the
//! checks count. A surface without masses (plain variables) has unit masses. It replaced a
//! threshold on the curvature without masses, scaled to the largest curvature of the surface,
//! which let collective modes of heavy fragments with imaginary frequencies of 10 cm^-1 pass as
//! minima (molecules of a crystal pulled apart; user, 2026-11-20).
//!
//! The gradient decides the side to which a mode is left when it has a component along it;
//! at an exactly symmetric point it has none, both sides are equivalent, and the positive
//! direction of the eigenvector is taken.
//!
//! MultiOptPy: `IRC/modekill.py:ModeKill` removes extra imaginary modes after an IRC, and
//!  `irc.py:IRC.saddle_check` counts imaginary modes with the same threshold; there is no
//!  counterpart for optimizations (new in mopt-rs). `pes::microiter` has its own mode kill for
//!  the inner relaxation of microiterations.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::MoptResult;
use crate::math::linalg::{orthonormalize_columns, symmetric_eigen, symmetrize};
use crate::math::projection::project_out_basis_mat;

/// Eigenvalues of the mass-weighted Hessian (Hartree / (Bohr^2 amu)) below minus this are
/// imaginary modes: MultiOptPy's `saddle_check` (`irc.py`), about 0.51 cm^-1. Flatter modes
/// are numerically flat and tolerated.
pub const IMAGINARY_CURVATURE: f64 = 1e-8;

/// Parameters of the mode kill at convergence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeKillSettings {
    /// Eigenvalues of the mass-weighted Hessian below minus this count as imaginary modes
    /// ([`IMAGINARY_CURVATURE`]).
    pub threshold: f64,
    /// Length of the step along each mode that is left (Bohr, or the unit of the variables).
    pub displacement: f64,
    /// Displacements at most in one run.
    pub max_rounds: usize,
    /// Judge a converged structure only by a Hessian asked for a check of stability
    /// ([`crate::domain::EvalRequest::for_stability`]), also where the converging iteration had
    /// an exact Hessian for its steps. On a surface whose Hessians for the steps and for the
    /// checks differ (the relaxed surface of ONIOM microiterations: the environment of a check
    /// is relaxed to a tighter tolerance) the verdict is then the one an IRC, a vibrational
    /// analysis, and the mapper reach at the same structure; elsewhere the two Hessians are the
    /// same and this costs one more Hessian where the convergence falls on an iteration with an
    /// exact one. Off by default (`cli::oniom_micro::verified_saddle_points` turns it on).
    pub checked_hessian: bool,
}

impl Default for ModeKillSettings {
    fn default() -> Self {
        Self {
            threshold: IMAGINARY_CURVATURE,
            displacement: 0.1,
            max_rounds: 10,
            checked_hessian: false,
        }
    }
}

/// A mode that a mode kill leaves.
#[derive(Debug, Clone)]
pub struct LeftMode {
    /// Eigenvalue of the mass-weighted Hessian (Hartree / (Bohr^2 amu); with unit masses the
    /// curvature in the units of the variables).
    pub curvature: f64,
    /// Its wavenumber (cm^-1, negative: imaginary), where the masses are those of atoms.
    pub wavenumber: Option<f64>,
    /// The unit direction of the step along it, in the coordinates of the Hessian.
    pub direction: Col<f64>,
}

/// What the Hessian at a converged point says.
#[derive(Debug, Clone)]
pub enum ModeKillVerdict {
    /// Not more imaginary modes than asked for (`negative` of them).
    Accept {
        /// Number of imaginary modes.
        negative: usize,
    },
    /// More than asked for: `step` leaves the point along the `killed` surplus modes.
    Displace {
        /// The step, in the coordinates of the Hessian.
        step: Col<f64>,
        /// Number of imaginary modes.
        negative: usize,
        /// Number of modes the step leaves.
        killed: usize,
        /// Each mode the step leaves.
        left: Vec<LeftMode>,
    },
}

/// The modes `left` by a mode kill for a log: the wavenumber of each where the masses are
/// those of atoms, its mass-weighted curvature otherwise.
pub fn describe_left(left: &[LeftMode]) -> String {
    left.iter()
        .map(|m| match m.wavenumber {
            Some(w) => format!("{w:.1} cm^-1"),
            None => format!("curvature {:+.3e}", m.curvature),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Count the imaginary modes of `hessian` (symmetric, in coordinates whose masses are
/// `masses`, one per coordinate; `None`: unit masses, a surface of plain variables) and
/// compare them with the order asked for; `gradient` is the gradient there, `symmetries` the
/// directions (columns, in the same coordinates) along which the surface does not change
/// (translations and rotations), projected out in the mass-weighted space.
pub fn examine(
    hessian: &Mat<f64>,
    gradient: &Col<f64>,
    saddle_order: usize,
    settings: &ModeKillSettings,
    masses: Option<&[f64]>,
    symmetries: Option<&Mat<f64>>,
) -> MoptResult<ModeKillVerdict> {
    Ok(examine_retaining(
        hessian,
        gradient,
        saddle_order,
        settings,
        masses,
        symmetries,
    )?
    .0)
}

/// Also return the unstable directions to follow when leaving the surplus ones.
pub(crate) fn examine_retaining(
    hessian: &Mat<f64>,
    gradient: &Col<f64>,
    saddle_order: usize,
    settings: &ModeKillSettings,
    masses: Option<&[f64]>,
    symmetries: Option<&Mat<f64>>,
) -> MoptResult<(ModeKillVerdict, Vec<Col<f64>>)> {
    let n = hessian.nrows();
    let atoms = masses.is_some_and(|m| m.len() == n);
    let (values, vectors, sqrt_m) = weighted_eigen(hessian, masses, symmetries)?;
    let negative = (0..n).filter(|&k| values[k] < -settings.threshold).count();
    if negative <= saddle_order {
        return Ok((ModeKillVerdict::Accept { negative }, Vec::new()));
    }
    // Eigenvalues ascend: the lowest `saddle_order` stay, the other imaginary ones are left,
    // along their directions in the coordinates (`M^(-1/2) y`, normalized).
    let direction = |k: usize| {
        let d = Col::from_fn(n, |i| vectors[(i, k)] / sqrt_m[i]);
        let norm = crate::math::linalg::col_norm(&d).max(1e-300);
        d * (1.0 / norm)
    };
    let retained = (0..saddle_order).map(direction).collect();
    let surplus: Vec<usize> = (saddle_order..negative).collect();
    let mut directions = Mat::<f64>::zeros(n, surplus.len());
    let mut left = Vec::with_capacity(surplus.len());
    for (c, &k) in surplus.iter().enumerate() {
        let d = direction(k);
        for i in 0..n {
            directions[(i, c)] = d[i];
        }
        left.push(LeftMode {
            curvature: values[k],
            wavenumber: atoms.then(|| crate::domain::units::eigenvalue_to_wavenumber(values[k])),
            direction: Col::from_fn(n, |i| directions[(i, c)]),
        });
    }
    let columns: Vec<usize> = (0..surplus.len()).collect();
    let step = downhill_step(&directions, &columns, gradient, settings.displacement);
    Ok((
        ModeKillVerdict::Displace {
            step,
            negative,
            killed: negative - saddle_order,
            left,
        },
        retained,
    ))
}

/// The eigenvalues (ascending) and eigenvectors of the mass-weighted Hessian that [`examine`]
/// counts, with the square roots of the masses (one per coordinate; unit masses where
/// `masses` does not fit).
fn weighted_eigen(
    hessian: &Mat<f64>,
    masses: Option<&[f64]>,
    symmetries: Option<&Mat<f64>>,
) -> MoptResult<(Col<f64>, Mat<f64>, Vec<f64>)> {
    let n = hessian.nrows();
    let sqrt_m: Vec<f64> = match masses.filter(|m| m.len() == n) {
        Some(m) => m.iter().map(|v| libm::sqrt(v.max(1e-12))).collect(),
        None => vec![1.0; n],
    };
    let mut weighted = Mat::from_fn(n, n, |i, j| hessian[(i, j)] / (sqrt_m[i] * sqrt_m[j]));
    if let Some(basis) = symmetries.filter(|b| b.nrows() == n && b.ncols() > 0) {
        // A displacement `dx` is `M^(1/2) dx` in the mass-weighted space.
        let columns: Vec<Col<f64>> = (0..basis.ncols())
            .map(|k| Col::from_fn(n, |i| basis[(i, k)] * sqrt_m[i]))
            .collect();
        let weighted_basis = orthonormalize_columns(n, &columns, 1e-8);
        if weighted_basis.ncols() > 0 {
            weighted = project_out_basis_mat(&weighted_basis, &weighted);
        }
    }
    symmetrize(&mut weighted);
    let (values, vectors) = symmetric_eigen(&weighted)?;
    Ok((values, vectors, sqrt_m))
}

/// The eigenvalues of the mass-weighted Hessian, ascending, with the symmetries projected out:
/// the values whose signs [`examine`] and [`imaginary_modes`] count (Hartree / (Bohr^2 amu)
/// with the masses of atoms).
pub fn mass_weighted_eigenvalues(
    hessian: &Mat<f64>,
    masses: Option<&[f64]>,
    symmetries: Option<&Mat<f64>>,
) -> MoptResult<Vec<f64>> {
    Ok(weighted_eigen(hessian, masses, symmetries)?
        .0
        .iter()
        .copied()
        .collect())
}

/// The number of imaginary modes that [`examine`] counts.
pub fn imaginary_modes(
    hessian: &Mat<f64>,
    masses: Option<&[f64]>,
    symmetries: Option<&Mat<f64>>,
) -> MoptResult<usize> {
    let n = hessian.nrows();
    Ok(
        match examine(
            hessian,
            &Col::zeros(n),
            0,
            &ModeKillSettings::default(),
            masses,
            symmetries,
        )? {
            ModeKillVerdict::Accept { negative } | ModeKillVerdict::Displace { negative, .. } => {
                negative
            }
        },
    )
}

/// The step of length `displacement` along each of the `columns` of `vectors`, towards the
/// side on which `gradient` descends (the positive direction of a vector along which the
/// gradient has no component). Shared with the ModeKill of the IRC module.
pub fn downhill_step(
    vectors: &Mat<f64>,
    columns: &[usize],
    gradient: &Col<f64>,
    displacement: f64,
) -> Col<f64> {
    let n = vectors.nrows();
    let mut step = Col::<f64>::zeros(n);
    for &k in columns {
        let slope: f64 = (0..n).map(|i| vectors[(i, k)] * gradient[i]).sum();
        let sign = if slope > 0.0 { -1.0 } else { 1.0 };
        for i in 0..n {
            step[i] += sign * displacement * vectors[(i, k)];
        }
    }
    step
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn surplus_imaginary_modes_are_left_and_the_lowest_are_kept() {
        let h = Mat::from_fn(4, 4, |i, j| {
            if i == j {
                [-0.5, -0.02, 0.3, 1.0][i]
            } else {
                0.0
            }
        });
        let g = Col::from_fn(4, |i| [0.0, 1e-6, 0.0, 0.0][i]);
        let s = ModeKillSettings::default();
        // A first-order saddle is asked for: the mode of -0.02 is left, downhill.
        match examine(&h, &g, 1, &s, None, None).unwrap() {
            ModeKillVerdict::Displace {
                step,
                negative,
                killed,
                left,
            } => {
                assert_eq!((negative, killed), (2, 1));
                assert!((step[1] + 0.1).abs() < 1e-12 && step[0].abs() < 1e-12);
                assert_eq!(left.len(), 1);
                assert!((left[0].curvature + 0.02).abs() < 1e-12);
                assert!(left[0].wavenumber.is_none());
            }
            other => panic!("{other:?}"),
        }
        // A minimum is asked for: both are left.
        match examine(&h, &g, 0, &s, None, None).unwrap() {
            ModeKillVerdict::Displace { step, killed, .. } => {
                assert_eq!(killed, 2);
                assert!((step[0].abs() - 0.1).abs() < 1e-12 && (step[1] + 0.1).abs() < 1e-12);
            }
            other => panic!("{other:?}"),
        }
        // Two or more are asked for: nothing to do.
        assert!(matches!(
            examine(&h, &g, 2, &s, None, None).unwrap(),
            ModeKillVerdict::Accept { negative: 2 }
        ));
    }

    #[test]
    fn a_mode_is_imaginary_by_its_mass_weighted_curvature() {
        // One coordinate of 40 amu (argon) and one of 1 amu, curvatures -1.2e-5 and -1e-8
        // (Hartree/Bohr^2): mass-weighted -3e-7 (about -2.8 cm^-1, imaginary) and -1e-8 (at
        // the threshold, flat).
        let h = Mat::from_fn(3, 3, |i, j| {
            if i == j {
                [-1.2e-5, -1e-8, 0.4][i]
            } else {
                0.0
            }
        });
        let masses = [40.0, 1.0, 12.0];
        match examine(
            &h,
            &Col::zeros(3),
            0,
            &ModeKillSettings::default(),
            Some(&masses),
            None,
        )
        .unwrap()
        {
            ModeKillVerdict::Displace { negative, left, .. } => {
                assert_eq!(negative, 1);
                let w = left[0].wavenumber.unwrap();
                let expected = -libm::sqrt(1.2e-5 / 40.0) * crate::domain::Units::EIGVAL_MW_TO_CM1;
                assert!((w - expected).abs() < 1e-9, "{w}");
                assert!((left[0].direction[0].abs() - 1.0).abs() < 1e-12);
                assert!(describe_left(&left).contains("-2.8 cm^-1"));
            }
            other => panic!("{other:?}"),
        }
        // A mode of 0.4 cm^-1 is flat: about 6e-9 mass-weighted.
        let flat = Mat::from_fn(
            2,
            2,
            |i, j| if i == j { [-6e-9 * 12.0, 0.4][i] } else { 0.0 },
        );
        assert_eq!(
            imaginary_modes(&flat, Some(&[12.0, 12.0]), None).unwrap(),
            0
        );
        // The same curvatures in other units of the masses count otherwise: the criterion
        // is that of the frequencies.
        assert_eq!(imaginary_modes(&flat, Some(&[0.1, 12.0]), None).unwrap(), 1);
    }

    #[test]
    fn symmetries_are_projected_out_in_the_mass_weighted_space() {
        // Two atoms on a line (one coordinate each), masses 1 and 4, a spring of 0.5 between
        // them: the translation (1, 1) is a symmetry. A negative curvature put along the
        // translation (noise of differences) is no imaginary mode once it is projected out.
        let k = 0.5;
        let noise = -1e-4;
        let h = Mat::from_fn(2, 2, |i, j| {
            let spring = if i == j { k } else { -k };
            spring + noise * 0.5
        });
        let masses = [1.0, 4.0];
        assert_eq!(imaginary_modes(&h, Some(&masses), None).unwrap(), 1);
        let translation = Mat::from_fn(2, 1, |_, _| 1.0);
        assert_eq!(
            imaginary_modes(&h, Some(&masses), Some(&translation)).unwrap(),
            0
        );
    }
}
