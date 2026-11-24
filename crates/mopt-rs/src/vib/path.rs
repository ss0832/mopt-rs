//! A reaction path as the reaction-path Hamiltonian sees it (Miller, Handy, Adams, J. Chem.
//! Phys. 1980, 72, 99): the path itself and the vibrations transverse to it.
//!
//! In mass-weighted coordinates `q = M^(1/2) x` with the arc length `s`:
//!
//! - the unit tangent `t(s)` (along the gradient; along the line through the neighbouring
//!   points where the gradient vanishes), oriented from the first point to the last;
//! - the curvature vector `k(s) = dt/ds`, by finite differences over the points, and the
//!   scalar curvature `|k|`;
//! - the generalized normal modes `L_k(s)` orthogonal to the path and to the rigid-body
//!   motions ([`super::transverse_modes`]), their wavenumbers, and the zero-point energy
//!   `sum_k h nu_k / 2` of those with real frequencies, which added to the energy gives the
//!   vibrationally adiabatic ground-state potential;
//! - the curvature couplings `B_kF = L_k . dt/ds`: how strongly motion along the path is
//!   coupled to mode `k` (`sum_k B_kF^2 = |k|^2`);
//! - the couplings between the modes `B_kl = L_k . dL_l/ds`, for which the modes of
//!   neighbouring points are matched by their overlaps and given a common sign.
//!
//! While all transverse curvatures are positive the path runs along the floor of a valley.
//! Where the lowest one passes through zero the valley becomes a ridge (a valley-ridge
//! inflection point): beyond it the path is unstable towards the two sides along that mode,
//! and trajectories part there (a bifurcation of the reaction path). Sign changes from
//! positive to negative are located by linear interpolation in the arc length.
//!
//! The points must be given in a common orientation (as an intrinsic reaction coordinate
//! gives them), since arc lengths, tangents, and overlaps compare Cartesian vectors.
//!
//! On a steepest-descent path the gradient is the tangent, and it is the better one (no
//! error from the spacing of the points). On another path (a scan, a band of images) the
//! gradient points elsewhere ([`gradient_alignment`] tells); there the tangent is taken
//! from the points ([`path_tangents`]), and the transverse frequencies are those of the
//! Hessian in the space orthogonal to the path, where the gradient does not vanish.
//!
//! MultiOptPy: `PESAnalyzer/calc_irc_curvature.py:calc_irc_curvature_properties` (tangent,
//!  curvature vector, scalar curvature, curvature coupling; there the curvature vector is the
//!  difference of the gradients instead of the unit tangents, and the couplings are taken
//!  with the modes of the unprojected Hessian). The transverse modes, the followed modes and
//!  their couplings, and the valley-ridge transitions have no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{transverse_modes_with_masses, Direction, NormalModes};
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::assignment::linear_sum_assignment;
use crate::math::linalg::{col_dot, col_norm};
use crate::math::projection::{build_tr_rot_basis_with_masses, project_out_basis_vec};

/// A point of a path.
#[derive(Debug, Clone)]
pub struct PathPoint {
    /// Structure.
    pub system: MolecularSystem,
    /// Tangent of the path there: the gradient, or a displacement along the path.
    pub tangent: Direction,
    /// Cartesian Hessian (Hartree/Bohr^2).
    pub hessian: Mat<f64>,
}

/// A point where the lowest transverse curvature changes from positive to negative.
#[derive(Debug, Clone)]
pub struct Bifurcation {
    /// Index of the last point with a positive curvature; the next one has a negative one.
    pub after: usize,
    /// Arc length at which the curvature is zero by linear interpolation
    /// (`sqrt(amu) Bohr`).
    pub arc_length: f64,
    /// Cartesian displacement of the mode that turns unstable, at the point after the
    /// change: the two branches leave along `+` and `-` this direction.
    pub mode: Col<f64>,
}

/// The path and its transverse vibrations, point by point.
#[derive(Debug, Clone)]
pub struct PathProfile {
    /// Arc length of every point from the first (`sqrt(amu) Bohr`).
    pub arc_length: Vec<f64>,
    /// Unit tangent in mass-weighted coordinates, from the first point towards the last.
    pub tangent: Vec<Col<f64>>,
    /// Curvature vector `dt/ds` in mass-weighted coordinates (`1 / (sqrt(amu) Bohr)`),
    /// without its components along the tangent and the rigid-body motions.
    pub curvature_vector: Vec<Col<f64>>,
    /// Its length: the scalar curvature of the path.
    pub curvature: Vec<f64>,
    /// Transverse modes at every point, ascending in their curvatures.
    pub modes: Vec<NormalModes>,
    /// Their wavenumbers (cm^-1; negative: imaginary).
    pub wavenumbers: Vec<Vec<f64>>,
    /// Curvature couplings `B_kF` of the transverse modes, in the order of `wavenumbers`.
    pub curvature_couplings: Vec<Vec<f64>>,
    /// Zero-point energy of the transverse modes with real frequencies (Hartree).
    pub zero_point_energy: Vec<f64>,
    /// Lowest transverse wavenumber at every point.
    pub lowest_wavenumber: Vec<f64>,
    /// Its curvature (Hartree / (Bohr^2 amu)).
    pub lowest_curvature: Vec<f64>,
    /// Cartesian displacement of that mode.
    pub lowest_mode: Vec<Col<f64>>,
    /// Absolute overlap of that mode with the lowest one of the previous point (1 for the
    /// first): near 1 while it is the same mode, small where another has become the lowest.
    pub overlap: Vec<f64>,
    /// For every point, the index (into `wavenumbers`) of each mode followed from the first
    /// point by the overlaps of neighbouring points.
    pub tracks: Vec<Vec<usize>>,
    /// Couplings `B_kl = L_k . dL_l/ds` between the followed modes (`tracks` order; an
    /// antisymmetric matrix at every point up to the error of the differences).
    pub mode_couplings: Vec<Mat<f64>>,
    /// Valley-ridge transitions found.
    pub bifurcations: Vec<Bifurcation>,
    /// The masses of the atoms (amu) the coordinates are weighted with.
    pub masses: Vec<f64>,
}

/// Weights of the derivative with respect to `s` at index `i` of a function given at the
/// abscissae `s`: the derivative of the parabola through three neighbouring points (of the
/// line through the two points of a path of two).
pub(crate) fn derivative_weights(s: &[f64], i: usize) -> Vec<(usize, f64)> {
    match s.len() {
        0 | 1 => Vec::new(),
        2 => {
            let h = s[1] - s[0];
            vec![(0, -1.0 / h), (1, 1.0 / h)]
        }
        n => {
            let a = i.saturating_sub(1).min(n - 3);
            let (x0, x1, x2, x) = (s[a], s[a + 1], s[a + 2], s[i]);
            vec![
                (a, (2.0 * x - x1 - x2) / ((x0 - x1) * (x0 - x2))),
                (a + 1, (2.0 * x - x0 - x2) / ((x1 - x0) * (x1 - x2))),
                (a + 2, (2.0 * x - x0 - x1) / ((x2 - x0) * (x2 - x1))),
            ]
        }
    }
}

/// Derivative of the scalars `values` given at `s`, at every point.
pub(crate) fn scalar_derivative(values: &[f64], s: &[f64]) -> Vec<f64> {
    (0..s.len())
        .map(|i| {
            derivative_weights(s, i)
                .iter()
                .map(|&(k, w)| w * values[k])
                .sum()
        })
        .collect()
}

/// Derivative of the vectors `values` given at `s`, at index `i`.
fn vector_derivative(values: &[Col<f64>], s: &[f64], i: usize) -> Col<f64> {
    let mut out = Col::zeros(values[i].nrows());
    for (k, w) in derivative_weights(s, i) {
        for c in 0..out.nrows() {
            out[c] += w * values[k][c];
        }
    }
    out
}

/// Mass-weighted positions of `systems`, the arc length of each from the first
/// (`sqrt(amu) Bohr`), and `sqrt(m)` of every coordinate.
fn weighted_path(
    systems: &[&MolecularSystem],
    masses: Option<&[f64]>,
) -> MoptResult<(Vec<Col<f64>>, Vec<f64>, Vec<f64>)> {
    let Some(first) = systems.first() else {
        return Err(MoptError::config("a path needs at least one point"));
    };
    let n = first.dof();
    if systems.iter().any(|s| s.dof() != n) {
        return Err(MoptError::config(
            "the points of a path must have the same atoms",
        ));
    }
    let masses = match masses {
        Some(m) if m.len() == first.n_atoms() && m.iter().all(|v| *v > 0.0 && v.is_finite()) => {
            m.to_vec()
        }
        Some(m) => {
            return Err(MoptError::config(format!(
                "a path of {} atoms with {} masses",
                first.n_atoms(),
                m.len()
            )))
        }
        None => first.masses_amu(),
    };
    let sqrt_mass: Vec<f64> = (0..n).map(|i| masses[i / 3].sqrt()).collect();
    let q: Vec<Col<f64>> = systems
        .iter()
        .map(|s| {
            let x = s.coords_flat_bohr();
            Col::from_fn(n, |i| sqrt_mass[i] * x[i])
        })
        .collect();
    let mut arc_length = vec![0.0; q.len()];
    for i in 1..q.len() {
        let step = col_norm(&(&q[i] - &q[i - 1]));
        if !(step > 0.0 && step.is_finite()) {
            return Err(MoptError::config(format!(
                "path points {} and {i} coincide",
                i - 1
            )));
        }
        arc_length[i] = arc_length[i - 1] + step;
    }
    Ok((q, arc_length, sqrt_mass))
}

/// Tangents of the path through `systems` from the points alone: the derivative of the
/// positions with respect to the arc length, from the parabola through three neighbouring
/// points. For a path that is not a steepest-descent path, and where the gradient vanishes.
pub fn path_tangents(systems: &[MolecularSystem]) -> MoptResult<Vec<Direction>> {
    path_tangents_with_masses(systems, None)
}

/// [`path_tangents`] with the arc length weighted by the masses `masses` of the atoms (amu;
/// `None`: the standard atomic weights).
pub fn path_tangents_with_masses(
    systems: &[MolecularSystem],
    masses: Option<&[f64]>,
) -> MoptResult<Vec<Direction>> {
    if systems.len() < 2 {
        return Err(MoptError::config(
            "tangents from the points need two points at least",
        ));
    }
    let refs: Vec<&MolecularSystem> = systems.iter().collect();
    let (_, arc_length, _) = weighted_path(&refs, masses)?;
    let x: Vec<Col<f64>> = systems.iter().map(|s| s.coords_flat_bohr()).collect();
    Ok((0..systems.len())
        .map(|i| Direction::Displacement(vector_derivative(&x, &arc_length, i)))
        .collect())
}

/// How far the gradient is the tangent of the path: the absolute overlap (0 to 1, in
/// mass-weighted coordinates) of each of `gradients` (Hartree/Bohr) with the tangent from
/// the points ([`path_tangents`]). 1 on a steepest-descent path up to the error of the
/// spacing; NaN where the gradient vanishes.
pub fn gradient_alignment(
    systems: &[MolecularSystem],
    gradients: &[Col<f64>],
) -> MoptResult<Vec<f64>> {
    gradient_alignment_with_masses(systems, gradients, None)
}

/// [`gradient_alignment`] in the coordinates weighted by the masses `masses` of the atoms
/// (amu; `None`: the standard atomic weights).
pub fn gradient_alignment_with_masses(
    systems: &[MolecularSystem],
    gradients: &[Col<f64>],
    masses: Option<&[f64]>,
) -> MoptResult<Vec<f64>> {
    let tangents = path_tangents_with_masses(systems, masses)?;
    if gradients.len() != systems.len() {
        return Err(MoptError::config(format!(
            "{} gradients for {} path points",
            gradients.len(),
            systems.len()
        )));
    }
    let mut out = Vec::with_capacity(systems.len());
    for (i, tangent) in tangents.iter().enumerate() {
        let Direction::Displacement(dx) = tangent else {
            return Err(MoptError::calc("path tangents are displacements"));
        };
        let g = &gradients[i];
        if g.nrows() != dx.nrows() {
            return Err(MoptError::config(format!(
                "the gradient of path point {i} has {} components, expected {}",
                g.nrows(),
                dx.nrows()
            )));
        }
        // (M^(1/2) dx) . (M^(-1/2) g) = dx . g; the lengths carry the masses.
        let masses = masses.map_or_else(|| systems[i].masses_amu(), <[f64]>::to_vec);
        if masses.len() * 3 != dx.nrows() {
            return Err(MoptError::config(format!(
                "path point {i}: {} masses for {} atoms",
                masses.len(),
                dx.nrows() / 3
            )));
        }
        let (mut along, mut dx2, mut g2) = (0.0, 0.0, 0.0);
        for c in 0..dx.nrows() {
            along += dx[c] * g[c];
            dx2 += masses[c / 3] * dx[c] * dx[c];
            g2 += g[c] * g[c] / masses[c / 3];
        }
        let lengths = (dx2 * g2).sqrt();
        out.push(if lengths > 0.0 {
            (along / lengths).abs().min(1.0)
        } else {
            f64::NAN
        });
    }
    Ok(out)
}

/// The path and its transverse vibrations at `points`; with `project_rigid`, translations
/// and rotations are removed from the tangent, the curvature, and the modes (not for a
/// surface in a fixed frame).
pub fn transverse_profile(points: &[PathPoint], project_rigid: bool) -> MoptResult<PathProfile> {
    transverse_profile_with_masses(points, project_rigid, None)
}

/// [`transverse_profile`] with the masses `masses` of the atoms (amu; `None`: the standard
/// atomic weights) in the arc length, the tangent, the curvature and the modes: those of the
/// IRC that traced the path (`vib::MassChoice`).
pub fn transverse_profile_with_masses(
    points: &[PathPoint],
    project_rigid: bool,
    masses: Option<&[f64]>,
) -> MoptResult<PathProfile> {
    let Some(first) = points.first() else {
        return Err(MoptError::config("a path needs at least one point"));
    };
    let n = first.system.dof();
    let count = points.len();
    let systems: Vec<&MolecularSystem> = points.iter().map(|p| &p.system).collect();
    let masses: Vec<f64> = masses.map_or_else(|| first.system.masses_amu(), <[f64]>::to_vec);
    let (q, arc_length, sqrt_mass) = weighted_path(&systems, Some(&masses))?;
    // Rigid-body motions at every point (mass-weighted, orthonormal).
    let rigid: Vec<Mat<f64>> = points
        .iter()
        .map(|p| {
            if project_rigid {
                build_tr_rot_basis_with_masses(&p.system, &masses)
            } else {
                Mat::zeros(n, 0)
            }
        })
        .collect();
    // Unit tangents, oriented along the sequence of the points.
    let mut tangent = Vec::with_capacity(count);
    for (i, point) in points.iter().enumerate() {
        let given = match &point.tangent {
            Direction::Gradient(g) | Direction::Displacement(g) => g,
        };
        if given.nrows() != n {
            return Err(MoptError::config(format!(
                "the tangent of path point {i} has {} components, expected {n}",
                given.nrows()
            )));
        }
        let weighted = match &point.tangent {
            Direction::Gradient(g) => Col::from_fn(n, |c| g[c] / sqrt_mass[c]),
            Direction::Displacement(dx) => Col::from_fn(n, |c| dx[c] * sqrt_mass[c]),
        };
        let raw = project_out_basis_vec(&rigid[i], &weighted);
        let length = col_norm(&raw);
        if !(length > 0.0 && length.is_finite()) {
            return Err(MoptError::config(format!(
                "path point {i} has no tangent: give the direction of the path where the \
                 gradient vanishes"
            )));
        }
        let along = if count > 1 {
            let (before, after) = (i.saturating_sub(1), (i + 1).min(count - 1));
            col_dot(&raw, &(&q[after] - &q[before]))
        } else {
            1.0
        };
        let sign = if along < 0.0 { -1.0 } else { 1.0 };
        tangent.push(Col::from_fn(n, |c| sign * raw[c] / length));
    }
    // Curvature vectors: dt/ds within the space of the transverse modes.
    let curvature_vector: Vec<Col<f64>> = (0..count)
        .map(|i| {
            let d = vector_derivative(&tangent, &arc_length, i);
            let along = col_dot(&d, &tangent[i]);
            let across = Col::from_fn(n, |c| d[c] - along * tangent[i][c]);
            project_out_basis_vec(&rigid[i], &across)
        })
        .collect();

    let mut profile = PathProfile {
        arc_length,
        curvature: curvature_vector.iter().map(col_norm).collect(),
        tangent,
        curvature_vector,
        modes: Vec::with_capacity(count),
        wavenumbers: Vec::with_capacity(count),
        curvature_couplings: Vec::with_capacity(count),
        zero_point_energy: Vec::with_capacity(count),
        lowest_wavenumber: Vec::with_capacity(count),
        lowest_curvature: Vec::with_capacity(count),
        lowest_mode: Vec::with_capacity(count),
        overlap: Vec::with_capacity(count),
        tracks: Vec::with_capacity(count),
        mode_couplings: Vec::with_capacity(count),
        bifurcations: Vec::new(),
        masses: masses.clone(),
    };
    // Mass-weighted unit vectors of the modes, in the order of the wavenumbers.
    let mut vectors: Vec<Vec<Col<f64>>> = Vec::with_capacity(count);
    for (i, point) in points.iter().enumerate() {
        let modes = transverse_modes_with_masses(
            &point.system,
            &point.hessian,
            project_rigid,
            std::slice::from_ref(&point.tangent),
            &masses,
        )?;
        if modes.wavenumbers.is_empty() {
            return Err(MoptError::calc(format!(
                "path point {i} has no vibration transverse to the path"
            )));
        }
        let here: Vec<Col<f64>> = modes
            .displacements
            .iter()
            .map(|d| {
                let v = Col::from_fn(n, |c| d[c] * sqrt_mass[c]);
                let length = col_norm(&v);
                Col::from_fn(n, |c| v[c] / length)
            })
            .collect();
        profile.curvature_couplings.push(
            here.iter()
                .map(|l| col_dot(l, &profile.curvature_vector[i]))
                .collect(),
        );
        profile
            .zero_point_energy
            .push(0.5 * Units::BOLTZMANN_HARTREE_PER_K * modes.temperatures.iter().sum::<f64>());
        // The lowest mode (the modes ascend) and the valley-ridge transitions.
        let overlap = match vectors.last() {
            Some(previous) => col_dot(&previous[0], &here[0]).abs(),
            None => 1.0,
        };
        let curvature = modes.curvatures[0];
        if let Some(&before) = profile.lowest_curvature.last() {
            // Negative by the one criterion of the crate (`vib::is_imaginary_curvature`).
            if !super::is_imaginary_curvature(before) && super::is_imaginary_curvature(curvature) {
                let (s_before, s_here) = (profile.arc_length[i - 1], profile.arc_length[i]);
                profile.bifurcations.push(Bifurcation {
                    after: i - 1,
                    arc_length: s_before + (s_here - s_before) * before / (before - curvature),
                    mode: modes.displacements[0].clone(),
                });
            }
        }
        profile.lowest_wavenumber.push(modes.wavenumbers[0]);
        profile.lowest_curvature.push(curvature);
        profile.lowest_mode.push(modes.displacements[0].clone());
        profile.overlap.push(overlap);
        profile.wavenumbers.push(modes.wavenumbers.clone());
        profile.modes.push(modes);
        vectors.push(here);
    }

    // Follow every mode of the first point through the path: at each point the modes are
    // matched with those of the point before so that the overlaps are largest in total, and
    // take the sign of their predecessors.
    let followed = vectors.iter().map(Vec::len).min().unwrap_or(0);
    let mut tracked: Vec<Vec<Col<f64>>> = Vec::with_capacity(count);
    for (i, here) in vectors.iter().enumerate() {
        let (order, signed) = match tracked.last() {
            None => (
                (0..followed).collect::<Vec<usize>>(),
                here[..followed].to_vec(),
            ),
            Some(previous) => {
                let cost = Mat::from_fn(followed, here.len(), |k, j| {
                    -col_dot(&previous[k], &here[j]).abs()
                });
                let order = linear_sum_assignment(&cost)
                    .map_err(|e| e.context(format!("following the modes at path point {i}")))?;
                let signed = order
                    .iter()
                    .enumerate()
                    .map(|(k, &j)| {
                        if col_dot(&previous[k], &here[j]) < 0.0 {
                            Col::from_fn(n, |c| -here[j][c])
                        } else {
                            here[j].clone()
                        }
                    })
                    .collect();
                (order, signed)
            }
        };
        profile.tracks.push(order);
        tracked.push(signed);
    }
    let series: Vec<Vec<Col<f64>>> = (0..followed)
        .map(|l| tracked.iter().map(|modes| modes[l].clone()).collect())
        .collect();
    for (i, here) in tracked.iter().enumerate() {
        let derivatives: Vec<Col<f64>> = series
            .iter()
            .map(|of_mode| vector_derivative(of_mode, &profile.arc_length, i))
            .collect();
        profile
            .mode_couplings
            .push(Mat::from_fn(followed, followed, |k, l| {
                col_dot(&here[k], &derivatives[l])
            }));
    }
    Ok(profile)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};

    fn atom(x: f64, y: f64, z: f64) -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::H],
            &positions::<Bohr>(&[[x, y, z]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    /// One atom on the surface `E = -x^3 / 3 + (1 - x) y^2 / 2 + z^2 / 2` along the path
    /// `y = z = 0`: a valley for `x < 1`, a ridge beyond.
    fn point(x: f64) -> PathPoint {
        let gradient = Col::from_fn(3, |i| if i == 0 { -x * x } else { 0.0 });
        let hessian = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => -2.0 * x,
            (1, 1) => 1.0 - x,
            (2, 2) => 1.0,
            _ => 0.0,
        });
        PathPoint {
            system: atom(x, 0.0, 0.0),
            tangent: Direction::Gradient(gradient),
            hessian,
        }
    }

    #[test]
    fn the_valley_ridge_transition_is_found_where_the_curvature_changes_sign() {
        let points: Vec<PathPoint> = (1..=9).map(|k| point(0.2 * k as f64)).collect();
        let profile = transverse_profile(&points, false).unwrap();
        let mass = points[0].system.masses_amu()[0];
        // Arc lengths are mass-weighted distances.
        assert!((profile.arc_length[8] - mass.sqrt() * 1.6).abs() < 1e-10);
        // Two transverse modes everywhere: the tangent is not among them.
        assert!(profile.wavenumbers.iter().all(|w| w.len() == 2));
        // The lowest transverse curvature is (1 - x) / m: along y throughout.
        for (k, curvature) in profile.lowest_curvature.iter().enumerate() {
            let x = 0.2 * (k + 1) as f64;
            assert!(
                (curvature - (1.0 - x) / mass).abs() < 1e-10,
                "{k}: {curvature}"
            );
            assert!(profile.lowest_mode[k][1].abs() > 0.9 * profile.lowest_mode[k][0].abs());
        }
        assert!(profile.overlap.iter().all(|o| (o - 1.0).abs() < 1e-10));
        // One transition, at x = 1.
        assert_eq!(profile.bifurcations.len(), 1);
        let found = &profile.bifurcations[0];
        assert!((found.arc_length - mass.sqrt() * 0.8).abs() < 1e-6 * mass.sqrt());
        assert!(found.mode[1].abs() > 1e-3 && found.mode[0].abs() < 1e-10);
        // A straight path: no curvature, no couplings; the tangent points along +x although
        // the gradient points along -x.
        assert!(profile.curvature.iter().all(|c| c.abs() < 1e-10));
        assert!(profile.tangent.iter().all(|t| (t[0] - 1.0).abs() < 1e-12));
        assert!(profile
            .curvature_couplings
            .iter()
            .all(|b| b.iter().all(|v| v.abs() < 1e-10)));
        // A path that stays in the valley has none, and an empty path is refused.
        let valley: Vec<PathPoint> = (1..=4).map(|k| point(0.2 * k as f64)).collect();
        assert!(transverse_profile(&valley, false)
            .unwrap()
            .bifurcations
            .is_empty());
        assert!(transverse_profile(&[], false).is_err());
        // So is a path that stays at one point.
        assert!(transverse_profile(&[point(0.2), point(0.2)], false).is_err());
        // The gradient is along the path.
        let systems: Vec<MolecularSystem> = points.iter().map(|p| p.system.clone()).collect();
        let gradients: Vec<Col<f64>> = points.iter().map(|p| vector(&p.tangent).clone()).collect();
        let alignment = gradient_alignment(&systems, &gradients).unwrap();
        assert!(alignment.iter().all(|a| (a - 1.0).abs() < 1e-12));
        assert!(gradient_alignment(&systems, &gradients[..3]).is_err());
        assert!(path_tangents(&systems[..1]).is_err());
    }

    fn vector(direction: &Direction) -> &Col<f64> {
        match direction {
            Direction::Gradient(v) | Direction::Displacement(v) => v,
        }
    }

    #[test]
    fn tangents_from_the_points_follow_a_curved_path() {
        // Points on a circle: the tangent from the points is the tangent of the circle, also
        // at the ends, and a gradient along the radius is orthogonal to it.
        let radius = 2.0;
        let angles: Vec<f64> = (0..9).map(|k| 0.1 * k as f64).collect();
        let systems: Vec<MolecularSystem> = circular_valley(radius, &angles)
            .into_iter()
            .map(|p| p.system)
            .collect();
        let tangents = path_tangents(&systems).unwrap();
        let mass = systems[0].masses_amu()[0];
        for (phi, tangent) in angles.iter().zip(&tangents) {
            let dx = vector(tangent);
            // dx/ds with s = sqrt(m) R phi.
            let (sin, cos) = phi.sin_cos();
            for (found, expected) in [(dx[0], -sin), (dx[1], cos), (dx[2], 0.0)] {
                assert!((found * mass.sqrt() - expected).abs() < 5e-3, "{found}");
            }
        }
        let radial: Vec<Col<f64>> = angles
            .iter()
            .map(|phi| Col::from_fn(3, |i| [phi.cos(), phi.sin(), 0.0][i]))
            .collect();
        let alignment = gradient_alignment(&systems, &radial).unwrap();
        assert!(alignment.iter().all(|a| *a < 5e-3), "{alignment:?}");
        // No gradient, no alignment.
        let none = vec![Col::<f64>::zeros(3); 9];
        assert!(gradient_alignment(&systems, &none).unwrap()[0].is_nan());
    }

    /// Points on the floor of the valley `E = (r - R)^2 / 2 + z^2`, the circle of radius `R`
    /// in the xy plane, at the angles `phi`. The gradient vanishes on the floor, so the
    /// tangent is given as a displacement.
    pub(crate) fn circular_valley(radius: f64, angles: &[f64]) -> Vec<PathPoint> {
        angles
            .iter()
            .map(|phi| {
                let (sin, cos) = phi.sin_cos();
                // Hessian at r = R: e_r e_r^T in the plane, 2 along z.
                let radial = [cos, sin, 0.0];
                let hessian = Mat::from_fn(3, 3, |i, j| {
                    radial[i] * radial[j] + if i == 2 && j == 2 { 2.0 } else { 0.0 }
                });
                PathPoint {
                    system: atom(radius * cos, radius * sin, 0.0),
                    tangent: Direction::Displacement(Col::from_fn(3, |i| [-sin, cos, 0.0][i])),
                    hessian,
                }
            })
            .collect()
    }

    #[test]
    fn a_circular_valley_has_the_curvature_of_its_radius_and_couples_to_the_radial_mode() {
        let radius = 2.0;
        let mass = atom(0.0, 0.0, 0.0).masses_amu()[0];
        let angles: Vec<f64> = (0..9).map(|k| 0.1 * k as f64).collect();
        let profile = transverse_profile(&circular_valley(radius, &angles), false).unwrap();
        // In mass-weighted coordinates the circle has the radius sqrt(m) R.
        let expected = 1.0 / (mass.sqrt() * radius);
        for i in 0..9 {
            assert!(
                (profile.curvature[i] - expected).abs() < 0.01 * expected,
                "{i}: {} against {expected}",
                profile.curvature[i]
            );
            // The path bends towards the centre: only the radial mode (curvature 1 / m, the
            // lower one) is coupled to it, with the whole curvature.
            let b = &profile.curvature_couplings[i];
            assert!((b[0].abs() - profile.curvature[i]).abs() < 1e-8);
            assert!(b[1].abs() < 1e-10);
            let sum: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((sum - profile.curvature[i]).abs() < 1e-10);
        }
        // Zero-point energy of the two transverse modes, constant along the path.
        let zpe = profile.zero_point_energy[0];
        assert!(zpe > 0.0);
        assert!(profile
            .zero_point_energy
            .iter()
            .all(|e| (e - zpe).abs() < 1e-12));
        // The radial mode turns with the path and stays orthogonal to z: the modes do not
        // couple to each other, and each is followed as itself.
        for i in 0..9 {
            assert_eq!(profile.tracks[i], vec![0, 1]);
            assert!(profile.mode_couplings[i][(0, 1)].abs() < 1e-8);
        }
    }

    #[test]
    fn modes_are_followed_through_a_crossing_and_their_coupling_is_the_rate_of_rotation() {
        // A straight path along x. The two transverse modes lie in the yz plane, turned by
        // the angle a(x) = x about the path, with the curvatures 1 + x and 2 - x: they cross
        // at x = 0.5 and rotate into each other at the rate da/ds.
        let mass = atom(0.0, 0.0, 0.0).masses_amu()[0];
        let points: Vec<PathPoint> = (0..10)
            .map(|k| {
                let x = 0.05 + 0.1 * k as f64;
                let (sin, cos) = x.sin_cos();
                let (u, v) = ([0.0, cos, sin], [0.0, -sin, cos]);
                let hessian = Mat::from_fn(3, 3, |i, j| {
                    (1.0 + x) * u[i] * u[j] + (2.0 - x) * v[i] * v[j]
                });
                PathPoint {
                    system: atom(x, 0.0, 0.0),
                    tangent: Direction::Displacement(Col::from_fn(3, |i| [1.0, 0.0, 0.0][i])),
                    hessian,
                }
            })
            .collect();
        let profile = transverse_profile(&points, false).unwrap();
        // Before the crossing the followed modes are in ascending order, after it swapped.
        assert_eq!(profile.tracks[2], vec![0, 1]);
        assert_eq!(profile.tracks[8], vec![1, 0]);
        // The lowest mode is another one after the crossing.
        assert!(profile.overlap[5] < 0.2, "{}", profile.overlap[5]);
        // |B_01| = da/ds = 1 / sqrt(m), antisymmetric.
        let expected = 1.0 / mass.sqrt();
        for i in 1..9 {
            let b = &profile.mode_couplings[i];
            assert!(
                (b[(0, 1)].abs() - expected).abs() < 0.01 * expected,
                "{i}: {}",
                b[(0, 1)]
            );
            assert!((b[(0, 1)] + b[(1, 0)]).abs() < 1e-3 * expected);
            assert!(b[(0, 0)].abs() < 1e-3 * expected);
        }
    }
}
