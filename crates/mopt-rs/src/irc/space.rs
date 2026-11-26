//! The variables an intrinsic reaction coordinate is followed in: mass weighting, and for
//! atoms the rigid-body motion.
//!
//! The path is defined in mass-weighted coordinates `q = M^(1/2) x`. For a structure the
//! weights are the atomic masses (amu; the standard atomic weights that the vibrational
//! analysis of mopt-rs uses, or given ones such as the masses of the most abundant isotopes
//! that MultiOptPy's IRC uses) and, when the surface does not change under
//! translations and rotations, exact Hessians are projected onto the vibrations and every new
//! point is moved back to the centre of mass, as MultiOptPy does. For plain variables the
//! masses are one (or given), nothing is projected, and nothing is moved.
//!
//! MultiOptPy: `IRC/lqa.py:LQA` (`get_mass_array`, `mass_weight_hessian`,
//! `mass_weight_coordinates`, `mass_weight_gradient`, `unmass_weight_step`) and
//! `Utils/calc_tools.py:Calculationtools` (`project_out_hess_tr_and_rot`,
//! `calc_center_of_mass`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use faer::{Col, Mat};

use super::constraints::IrcConstraints;
use super::types::PointValue;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::orthonormalize_columns;
use crate::math::projection::{build_tr_rot_basis, project_out_basis_mat};
use crate::parameters::isotope_masses::MOST_ABUNDANT_ISOTOPE_MASS_U;
use crate::pes::constraint::ConstraintSet;

/// Mass weighting and rigid-body handling of the variables of a path.
#[derive(Debug, Clone)]
pub struct IrcSpace {
    sqrt_mass: Col<f64>,
    atoms: Option<MolecularSystem>,
    project_rigid: bool,
    /// Whether the masses are other than the atomic weights of the structure.
    own_masses: bool,
    /// Constraints the path keeps (`None`: none).
    constraints: Option<Arc<IrcConstraints>>,
    /// Whether the surface does not change under translations and rotations (with
    /// constraints that fix the frame, `project_rigid` is off but the motions that keep
    /// the constraints are still projected out).
    rigid_surface: bool,
    /// Whether the rigid motions are taken out of an updated Hessian after every update, as
    /// out of an exact one ([`Self::project_hessian`]; `-update_proj`, on by default).
    project_updates: bool,
    /// The first restore onto the constraints that failed since the last evaluation
    /// ([`Self::placed`], [`Self::kept`]), returned by the next [`Self::constrained`]: a
    /// point off the constraints is never evaluated as if it were on them.
    restore_failure: Arc<Mutex<Option<MoptError>>>,
}

/// The masses of the most abundant isotopes of the atoms of `sys` (amu), which MultiOptPy's
/// IRC weights with.
pub fn most_abundant_isotope_masses(sys: &MolecularSystem) -> MoptResult<Vec<f64>> {
    sys.elements
        .iter()
        .map(|e| {
            MOST_ABUNDANT_ISOTOPE_MASS_U
                .get(e.atomic_number() as usize)
                .copied()
                .filter(|m| *m > 0.0)
                .ok_or_else(|| {
                    MoptError::config(format!("no isotope mass for element {}", e.symbol()))
                })
        })
        .collect()
}

impl IrcSpace {
    /// The Cartesian coordinates of `sys` (Bohr) with its atomic masses. With
    /// `project_rigid` the rigid-body motion is projected out of exact Hessians and new
    /// points are moved to the centre of mass (surfaces that do not change under
    /// translations and rotations).
    pub fn molecule(sys: &MolecularSystem, project_rigid: bool) -> Self {
        Self {
            sqrt_mass: sys.sqrt_masses_3n(),
            atoms: Some(sys.clone()),
            project_rigid,
            own_masses: false,
            constraints: None,
            rigid_surface: project_rigid,
            project_updates: true,
            restore_failure: Arc::default(),
        }
    }

    /// [`Self::molecule`] with the masses `masses_amu` of the atoms (isotopes; isolated
    /// structures only when rigid-body motion is projected out).
    pub fn molecule_with_masses(
        sys: &MolecularSystem,
        masses_amu: &[f64],
        project_rigid: bool,
    ) -> MoptResult<Self> {
        if masses_amu.len() != sys.n_atoms() {
            return Err(MoptError::config(format!(
                "{} masses for {} atoms",
                masses_amu.len(),
                sys.n_atoms()
            )));
        }
        if masses_amu.iter().any(|m| !(m.is_finite() && *m > 0.0)) {
            return Err(MoptError::config(
                "the masses of the atoms must be positive",
            ));
        }
        if project_rigid && sys.has_pbc() {
            return Err(MoptError::unsupported(
                "an IRC with given masses in a periodic cell is not available",
            ));
        }
        Ok(Self {
            sqrt_mass: Col::from_fn(sys.dof(), |k| libm::sqrt(masses_amu[k / 3])),
            atoms: Some(sys.clone()),
            project_rigid,
            own_masses: true,
            constraints: None,
            rigid_surface: project_rigid,
            project_updates: true,
            restore_failure: Arc::default(),
        })
    }

    /// `n` plain variables with unit masses.
    pub fn variables(n: usize) -> Self {
        Self {
            sqrt_mass: Col::from_fn(n, |_| 1.0),
            atoms: None,
            project_rigid: false,
            own_masses: false,
            constraints: None,
            rigid_surface: false,
            project_updates: true,
            restore_failure: Arc::default(),
        }
    }

    /// Plain variables with the given masses (one per variable, all positive).
    pub fn with_masses(masses: &[f64]) -> MoptResult<Self> {
        if masses.iter().any(|m| !(m.is_finite() && *m > 0.0)) {
            return Err(MoptError::config(
                "the masses of the variables must be positive",
            ));
        }
        Ok(Self {
            sqrt_mass: Col::from_fn(masses.len(), |i| libm::sqrt(masses[i])),
            atoms: None,
            project_rigid: false,
            own_masses: false,
            constraints: None,
            rigid_surface: false,
            project_updates: true,
            restore_failure: Arc::default(),
        })
    }

    /// The space with the constraints `set` on the structure: the path keeps them (see
    /// [`IrcConstraints`]). Constraints that fix the position or the orientation in space
    /// end the projection of the rigid-body motion. Plain variables have no constraints.
    pub fn with_constraints(mut self, set: ConstraintSet) -> MoptResult<Self> {
        let Some(atoms) = &self.atoms else {
            return Err(MoptError::unsupported(
                "constraints of an IRC need atoms; plain variables have none",
            ));
        };
        if set.is_empty() {
            return Ok(self);
        }
        let sqrt_mass: Vec<f64> = self.sqrt_mass.iter().copied().collect();
        let constraints = IrcConstraints::new(set, atoms.clone(), &sqrt_mass);
        self.project_rigid &= !constraints.breaks_rigid_body();
        self.constraints = Some(Arc::new(constraints));
        Ok(self)
    }

    /// The space with the rigid motions taken out of the updated Hessians after every update
    /// (`on`, the default) or left in them (`-update_proj off`, the Hessians of v2.1.0).
    pub fn with_update_projection(mut self, on: bool) -> Self {
        self.project_updates = on;
        self
    }

    /// Whether the rigid motions are taken out of the updated Hessians: with
    /// [`Self::with_update_projection`] on, where the rigid-body motion is projected out
    /// ([`Self::projects_rigid_body`]). Constraints that fix the frame are left to their own
    /// projection, as in the optimizers: taking the motions that keep them out of every update
    /// (rotations about a fixed atom) spoilt the path of HCN with its carbon atom fixed (17 and
    /// 28 steps of higher energy against 0 and 2, both sides ending at a rise of the energy).
    pub fn projects_updates(&self) -> bool {
        self.project_updates && self.projects_rigid_body()
    }

    /// An updated mass-weighted Hessian at `x` with the rigid motions taken out where
    /// [`Self::projects_updates`]; unchanged otherwise.
    pub fn project_updated_hessian(
        &self,
        mw_hessian: &Mat<f64>,
        x: &Col<f64>,
    ) -> MoptResult<Mat<f64>> {
        if self.projects_updates() {
            self.project_hessian(mw_hessian, x)
        } else {
            Ok(mw_hessian.clone())
        }
    }

    /// The constraints the path keeps.
    pub fn constraints(&self) -> Option<&IrcConstraints> {
        self.constraints.as_deref()
    }

    /// `x` on the constraints (the shortest way back in mass-weighted coordinates) and
    /// recentred; as it is where there are no constraints. Where the constraints cannot be
    /// met, `x` comes back as it is and the failure of the restore is recorded: the next
    /// [`Self::constrained`], the evaluation of the point, returns it as an error.
    pub fn placed(&self, x: &Col<f64>) -> Col<f64> {
        let kept = self.kept(x);
        self.recentered(&kept)
    }

    /// `x` on the constraints, not recentred (points of a hypersphere around a fixed
    /// centre); as it is without constraints. A failed restore is recorded as in
    /// [`Self::placed`].
    pub fn kept(&self, x: &Col<f64>) -> Col<f64> {
        match &self.constraints {
            Some(c) => match c.restore(x) {
                Ok(on) => on,
                Err(e) => {
                    if let Ok(mut failure) = self.restore_failure.lock() {
                        failure.get_or_insert(e);
                    }
                    x.clone()
                }
            },
            None => x.clone(),
        }
    }

    /// The value at `x` within the constraints (unchanged without constraints); the error of
    /// a restore that failed since the last evaluation ([`Self::placed`], [`Self::kept`]), so
    /// that a point off the constraints is never taken as one on them.
    pub fn constrained(&self, x: &Col<f64>, value: PointValue) -> MoptResult<PointValue> {
        let failure = match self.restore_failure.lock() {
            Ok(mut failure) => failure.take(),
            Err(_) => Some(MoptError::conv(
                "IRC: the record of the restores onto the constraints is unavailable",
            )),
        };
        if let Some(e) = failure {
            return Err(e.context("IRC: a point could not be brought onto the constraints"));
        }
        match &self.constraints {
            Some(c) => c.project(x, value),
            None => Ok(value),
        }
    }

    /// Number of variables.
    pub fn dimension(&self) -> usize {
        self.sqrt_mass.nrows()
    }

    /// The structure the variables belong to (`None` for plain variables).
    pub fn atoms(&self) -> Option<&MolecularSystem> {
        self.atoms.as_ref()
    }

    /// Whether rigid-body motion is projected out.
    pub fn projects_rigid_body(&self) -> bool {
        self.project_rigid && self.atoms.is_some()
    }

    /// Fail unless `x` has the length of the space.
    pub fn check(&self, x: &Col<f64>) -> MoptResult<()> {
        if x.nrows() == self.dimension() {
            Ok(())
        } else {
            Err(MoptError::config(format!(
                "expected {} variables, got {}",
                self.dimension(),
                x.nrows()
            )))
        }
    }

    /// `q = M^(1/2) x`.
    pub fn to_mw(&self, x: &Col<f64>) -> Col<f64> {
        Col::from_fn(x.nrows(), |i| x[i] * self.sqrt_mass[i])
    }

    /// `x = M^(-1/2) q` (points and displacements alike).
    pub fn from_mw(&self, q: &Col<f64>) -> Col<f64> {
        Col::from_fn(q.nrows(), |i| q[i] / self.sqrt_mass[i])
    }

    /// The gradient with respect to `q`: `M^(-1/2) g`.
    pub fn gradient_to_mw(&self, g: &Col<f64>) -> Col<f64> {
        Col::from_fn(g.nrows(), |i| g[i] / self.sqrt_mass[i])
    }

    /// The gradient with respect to `x` from the one with respect to `q`.
    pub fn gradient_from_mw(&self, g: &Col<f64>) -> Col<f64> {
        Col::from_fn(g.nrows(), |i| g[i] * self.sqrt_mass[i])
    }

    /// The Hessian with respect to `q`: `M^(-1/2) H M^(-1/2)`.
    pub fn hessian_to_mw(&self, h: &Mat<f64>) -> Mat<f64> {
        Mat::from_fn(h.nrows(), h.ncols(), |i, j| {
            h[(i, j)] / (self.sqrt_mass[i] * self.sqrt_mass[j])
        })
    }

    /// The Hessian with respect to `x` from the one with respect to `q`.
    pub fn hessian_from_mw(&self, h: &Mat<f64>) -> Mat<f64> {
        Mat::from_fn(h.nrows(), h.ncols(), |i, j| {
            h[(i, j)] * (self.sqrt_mass[i] * self.sqrt_mass[j])
        })
    }

    /// A mass-weighted Hessian with the translations and rotations of the structure at `x`
    /// projected out (unchanged for plain variables and surfaces that depend on the
    /// orientation).
    pub fn project_hessian(&self, mw_hessian: &Mat<f64>, x: &Col<f64>) -> MoptResult<Mat<f64>> {
        if let (Some(c), true) = (&self.constraints, self.rigid_surface && !self.project_rigid) {
            // Constraints that fix the frame: the rigid motions that keep them (rotations
            // about one fixed atom, about the line of two) still leave the energy as it
            // is; their curvatures are those of the noise of the Hessian.
            let kept = self.rigid_within(c, x)?;
            return Ok(if kept.ncols() > 0 {
                project_out_basis_mat(&kept, mw_hessian)
            } else {
                mw_hessian.clone()
            });
        }
        match (&self.atoms, self.project_rigid) {
            (Some(_), true) if self.own_masses => {
                Ok(project_out_basis_mat(&self.rigid_body_basis(x), mw_hessian))
            }
            (Some(atoms), true) => {
                let sys = atoms.with_flat_coords_bohr(x)?;
                Ok(project_out_basis_mat(
                    &build_tr_rot_basis(&sys, true),
                    mw_hessian,
                ))
            }
            _ => Ok(mw_hessian.clone()),
        }
    }

    /// Orthonormal basis (mass-weighted) of the rigid motions of the structure at `x` that
    /// keep the constraints `c`: the combinations of the translations and rotations
    /// without a part across the surface of the constraints.
    fn rigid_within(&self, c: &IrcConstraints, x: &Col<f64>) -> MoptResult<Mat<f64>> {
        let rigid = match (&self.atoms, self.own_masses) {
            (Some(_), true) => self.rigid_body_basis(x),
            (Some(atoms), false) => build_tr_rot_basis(&atoms.with_flat_coords_bohr(x)?, true),
            (None, _) => return Ok(Mat::zeros(x.nrows(), 0)),
        };
        let normals = c.normals(x)?;
        if rigid.ncols() == 0 || normals.ncols() == 0 {
            return Ok(rigid);
        }
        // Null space of N^T R within the span of R.
        let across = normals.transpose() * &rigid;
        let gram = across.transpose() * &across;
        let (values, vectors) = crate::math::linalg::symmetric_eigen(&gram)?;
        let kept: Vec<usize> = (0..values.nrows()).filter(|&k| values[k] < 1e-10).collect();
        let combinations = Mat::from_fn(rigid.ncols(), kept.len(), |i, j| vectors[(i, kept[j])]);
        Ok(&rigid * &combinations)
    }

    /// Orthonormal basis of the mass-weighted translations and rotations of the isolated
    /// structure at `x` with the masses of this space (MultiOptPy's vectors).
    fn rigid_body_basis(&self, x: &Col<f64>) -> Mat<f64> {
        let n = x.nrows();
        let centred = self.centred(x);
        let mut vectors = Vec::with_capacity(6);
        for c in 0..3 {
            vectors.push(Col::from_fn(n, |k| {
                if k % 3 == c {
                    self.sqrt_mass[k]
                } else {
                    0.0
                }
            }));
        }
        for axis in 0..3 {
            // The rotation about `axis`: e_axis x r.
            let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
            vectors.push(Col::from_fn(n, |k| {
                let atom = k / 3;
                let r = |c: usize| centred[3 * atom + c];
                let value = if k % 3 == a {
                    -r(b)
                } else if k % 3 == b {
                    r(a)
                } else {
                    0.0
                };
                value * self.sqrt_mass[k]
            }));
        }
        orthonormalize_columns(n, &vectors, 1e-10)
    }

    /// `x` with the centre of mass at the origin.
    fn centred(&self, x: &Col<f64>) -> Col<f64> {
        let mut total = 0.0;
        let mut com = [0.0_f64; 3];
        for k in 0..x.nrows() {
            let m = self.sqrt_mass[k] * self.sqrt_mass[k];
            com[k % 3] += m * x[k];
            if k % 3 == 0 {
                total += m;
            }
        }
        if total <= 0.0 {
            return x.clone();
        }
        Col::from_fn(x.nrows(), |k| x[k] - com[k % 3] / total)
    }

    /// The mass-weighted, projected Hessian of a Cartesian one at `x`.
    pub fn prepared_hessian(&self, hessian: &Mat<f64>, x: &Col<f64>) -> MoptResult<Mat<f64>> {
        let n = self.dimension();
        if hessian.nrows() != n || hessian.ncols() != n {
            return Err(MoptError::config(format!(
                "expected a {n}x{n} Hessian, got {}x{}",
                hessian.nrows(),
                hessian.ncols()
            )));
        }
        self.project_hessian(&self.hessian_to_mw(hessian), x)
    }

    /// The lowest curvature across the path at `x` (mass-weighted, Hartree / (amu Bohr^2)):
    /// the lowest eigenvalue of the mass-weighted, projected Hessian `hessian` (Cartesian)
    /// with the direction of the gradient `gradient` taken out as well. Negative where the
    /// valley the path runs in has turned into a ridge (past a valley-ridge inflection);
    /// zero or above otherwise (the directions taken out count as zero). `None` where the
    /// gradient is too small to give the direction of the path.
    pub fn curvature_across(
        &self,
        hessian: &Mat<f64>,
        x: &Col<f64>,
        gradient: &Col<f64>,
    ) -> MoptResult<Option<f64>> {
        let h = self.prepared_hessian(hessian, x)?;
        let g = self.gradient_to_mw(gradient);
        let length = super::types::norm(&g);
        if length.is_nan() || length <= 1e-8 {
            return Ok(None);
        }
        let t = Col::from_fn(g.nrows(), |i| g[i] / length);
        let ht = &h * &t;
        let tht = super::types::dot(&t, &ht);
        // (1 - t t^T) H (1 - t t^T).
        let across = Mat::from_fn(h.nrows(), h.ncols(), |i, j| {
            h[(i, j)] - t[i] * ht[j] - ht[i] * t[j] + tht * t[i] * t[j]
        });
        let (values, _) = crate::math::linalg::symmetric_eigen(&across)?;
        Ok(values.iter().copied().reduce(f64::min))
    }

    /// `x` moved so that the centre of mass is at the origin (isolated structures on
    /// surfaces that do not depend on the position; unchanged otherwise).
    pub fn recentered(&self, x: &Col<f64>) -> Col<f64> {
        let isolated = self
            .atoms
            .as_ref()
            .is_some_and(|a| self.project_rigid && !a.has_pbc());
        if isolated {
            self.centred(x)
        } else {
            x.clone()
        }
    }

    /// The point reached from `x` by the mass-weighted displacement `dq`, on the
    /// constraints and recentred ([`Self::placed`]).
    pub fn displaced(&self, x: &Col<f64>, dq: &Col<f64>) -> Col<f64> {
        let moved = Col::from_fn(x.nrows(), |i| x[i] + dq[i] / self.sqrt_mass[i]);
        self.placed(&moved)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{Element, SpinCharge};

    fn water() -> MolecularSystem {
        let x = Col::from_fn(9, |k| [0.0, 0.0, 0.2, 1.4, 0.0, -0.9, -1.4, 0.1, -0.9][k]);
        let elements = vec![
            Element::parse_token("O").unwrap(),
            Element::parse_token("H").unwrap(),
            Element::parse_token("H").unwrap(),
        ];
        MolecularSystem::from_flat_bohr(elements, &x, SpinCharge::default(), None).unwrap()
    }

    #[test]
    fn a_point_that_cannot_be_brought_onto_the_constraints_is_an_error() {
        // cos(r) never reaches 2: every restore fails.
        let sys = water();
        let impossible = crate::cli::constraints::user_constraint("cos(dist(1,2)) = 2").unwrap();
        let set = ConstraintSet::new(&[impossible], &sys, None).unwrap();
        let space = IrcSpace::molecule(&sys, false)
            .with_constraints(set)
            .unwrap();
        let x = sys.coords_flat_bohr();
        let value = || PointValue::plain(0.0, Col::zeros(9), None);
        for placed in [true, false] {
            let moved = if placed {
                space.placed(&x)
            } else {
                space.kept(&x)
            };
            // The point comes back as it is, and its evaluation is refused with the reason.
            assert_eq!(moved, x);
            let err = space.constrained(&moved, value()).unwrap_err().to_string();
            assert!(
                err.contains("onto the constraints") && err.contains("1e-10"),
                "{err}"
            );
        }
        // Once reported, a point on the constraints is evaluated again.
        let free = IrcSpace::molecule(&sys, false);
        assert!(free.constrained(&free.placed(&x), value()).is_ok());
    }

    #[test]
    fn mass_weighting_round_trips_and_recentres() {
        let sys = water();
        let space = IrcSpace::molecule(&sys, true);
        let x = sys.coords_flat_bohr();
        let back = space.from_mw(&space.to_mw(&x));
        for k in 0..9 {
            assert!((back[k] - x[k]).abs() < 1e-14);
        }
        let centred = space.recentered(&x);
        let m = sys.masses_amu();
        for c in 0..3 {
            let moment: f64 = (0..3).map(|i| m[i] * centred[3 * i + c]).sum();
            assert!(moment.abs() < 1e-12);
        }
        // A surface that depends on the position: nothing moves.
        let fixed = IrcSpace::molecule(&sys, false);
        assert_eq!(fixed.recentered(&x), x);
    }

    #[test]
    fn the_curvature_across_the_path_leaves_out_the_direction_of_the_path() {
        // Curvatures 2, -1, 3 along the axes.
        let space = IrcSpace::variables(3);
        let h = Mat::from_fn(3, 3, |i, j| if i == j { [2.0, -1.0, 3.0][i] } else { 0.0 });
        let x = Col::<f64>::zeros(3);
        let along = |k: usize| Col::from_fn(3, |i| if i == k { 0.5 } else { 0.0 });
        // A path along the first axis: the negative curvature of the second lies across it.
        let across = space.curvature_across(&h, &x, &along(0)).unwrap().unwrap();
        assert!((across + 1.0).abs() < 1e-12, "{across}");
        // A path along the negative direction itself: nothing negative across it.
        let across = space.curvature_across(&h, &x, &along(1)).unwrap().unwrap();
        assert!(across.abs() < 1e-12, "{across}");
        // Without a gradient there is no direction of the path.
        assert!(space
            .curvature_across(&h, &x, &Col::zeros(3))
            .unwrap()
            .is_none());
    }

    #[test]
    fn projected_hessian_has_six_zero_modes() {
        let sys = water();
        let space = IrcSpace::molecule(&sys, true);
        let h = Mat::from_fn(9, 9, |i, j| if i == j { 1.0 } else { 0.0 });
        let p = space.prepared_hessian(&h, &sys.coords_flat_bohr()).unwrap();
        let (values, _) = crate::math::linalg::symmetric_eigen(&p).unwrap();
        let zero = (0..9).filter(|&k| values[k].abs() < 1e-10).count();
        assert_eq!(zero, 6);
        let plain = IrcSpace::variables(9);
        assert_eq!(
            plain.prepared_hessian(&h, &sys.coords_flat_bohr()).unwrap(),
            h
        );
        assert!(IrcSpace::with_masses(&[1.0, -2.0]).is_err());
        // Given masses equal to the atomic weights give the same projection.
        let own = IrcSpace::molecule_with_masses(&sys, &sys.masses_amu(), true).unwrap();
        let q = own.prepared_hessian(&h, &sys.coords_flat_bohr()).unwrap();
        for i in 0..9 {
            for j in 0..9 {
                assert!((p[(i, j)] - q[(i, j)]).abs() < 1e-12);
            }
        }
        // The masses of the most abundant isotopes: 16O and 1H.
        let isotopes = most_abundant_isotope_masses(&sys).unwrap();
        assert!((isotopes[0] - 15.994_914_619_57).abs() < 1e-9 && isotopes[1] < 1.007_9);
        assert!(IrcSpace::molecule_with_masses(&sys, &isotopes[..2], true).is_err());
    }
}
