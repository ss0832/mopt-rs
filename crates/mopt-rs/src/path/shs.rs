//! Scaled normal coordinates of a minimum, the coordinates of the scaled hypersphere search
//! (SHS): with the eigenvalues `l_i` and eigenvectors `v_i` of the Hessian at the minimum
//! `x0` (rigid-body motions removed), `q_i = l_i^(1/2) v_i . (x - x0)`, so that the harmonic
//! energy is `|q|^2 / 2` and its levels are the hyperspheres `|q| = R`. The anharmonic
//! downward distortion (ADD) of a point is its energy less the harmonic one,
//! `ADD = E - E0 - R^2 / 2`; reaction paths leave the minimum where the ADD is largest
//! (most negative), at the energy minima on the spheres.
//!
//! The coordinates do not depend on the masses: any set of normal coordinates scaled this way
//! gives the same spheres and the same angles between points.
//!
//! Also here: the elimination function of the iterative optimization-elimination (IOE),
//! `-ADD_k cos^3(theta_k)` for `theta_k < pi/2` (`theta_k` the angle to the maximal ADD point
//! `k` in scaled coordinates), which is added to the energy so that a minimization on the
//! sphere does not end at an ADD already found.
//!
//! K. Ohno, S. Maeda, Chem. Phys. Lett. 2004, 384, 277 (SHS); S. Maeda, K. Ohno, J. Phys. Chem.
//! A 2005, 109, 5742; the elimination with `cos^3`, truncated at `theta = pi/2`, as the
//! reviews of the method describe it (S. Maeda, K. Ohno, K. Morokuma, Phys. Chem. Chem. Phys.
//! 2013, 15, 3683; S. Maeda et al., Bull. Chem. Soc. Jpn. 2014, 87, 1315).
//!
//! MultiOptPy: `OtherMethod/addf.py:ADDFlikeMethod` (`SQaxes`, `SQaxes_inv`,
//! `SuperSphere_cartesian`, `calctheta`, `IOE`, `IOE_grad`). MultiOptPy maps the scaled
//! coordinates back with `l_i^(1/2)` instead of `l_i^(-1/2)` (its spheres are not levels of the
//! harmonic energy, and its forward and backward maps are not inverse to each other) and
//! takes the gradient of the elimination function from differences; here the maps are the
//! inverse of each other and the gradient is analytic.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use faer::{Col, Mat};

use super::manifold::PointManifold;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen};
use crate::math::projection::project_out_basis_mat;

/// Eigenvalues below this share of the largest one belong to motions without restoring force
/// (the rigid-body motions taken out), which the coordinates leave out.
pub const ZERO_MODE_SHARE: f64 = 1e-7;

/// Newton iterations at most of the point of the hard constraints with given scaled
/// coordinates.
const PLACE_ITERATIONS: usize = 50;
/// Halvings at most of a Newton step that does not lower the residuals.
const PLACE_HALVINGS: usize = 8;
/// Placed points kept: the coordinates asked for again give the same point (a minimization
/// on a sphere evaluates its best point once more at its end), and the nearest one starts
/// the Newton iterations.
const PLACED_KEPT: usize = 256;
/// The Newton iterations end at this largest residual of the constraints (the restore then
/// takes the point to their floor).
const PLACE_TOLERANCE: f64 = 1e-12;
/// Where the Newton iterations stall above this largest residual, no point of the
/// constraints is near (the restore only polishes below it).
const PLACE_REACHED: f64 = 1e-8;

/// The coordinates asked for, the amounts along the normals at the minimum, and the point.
type PlacedPoint = (Col<f64>, Col<f64>, Col<f64>);

/// Scaled normal coordinates about a minimum.
#[derive(Debug, Clone)]
pub struct ScaledCoordinates {
    /// The minimum.
    origin: Col<f64>,
    /// The kept eigenvalues, ascending.
    values: Vec<f64>,
    /// Their eigenvectors as columns (`n x m`).
    vectors: Mat<f64>,
    /// The hard constraints the points lie on (`None`: the points are linear in the
    /// coordinates).
    manifold: Option<Arc<PointManifold>>,
    /// An orthonormal basis of the normals of the constraints at the minimum (`n x 0`
    /// without constraints).
    normals: Mat<f64>,
    /// The points placed last.
    last_placed: Arc<Mutex<Vec<PlacedPoint>>>,
}

impl ScaledCoordinates {
    /// The coordinates of the constrained minimum `origin` (on `manifold`), from the Hessian
    /// of the Lagrangian projected into the tangent space there (`hessian`; what the surfaces
    /// of the searches give, `P (H - C) P` plus a stiffness along the normals, which are left
    /// out here with the columns of `removed`).
    ///
    /// The derivation (new in mopt-rs; the SHS papers have no constraints): to second order
    /// about the constrained minimum `x0`, the energy on the manifold `M` is
    /// `E0 + 1/2 d^T P (H - C) P d` for a displacement `d` in `T_x0 M`, `C = sum_k lambda_k
    /// d2c_k` the curvature of the constraints weighted by their multipliers (the Hessian of the
    /// Lagrangian `E - lambda . c`). Its eigenvectors `v_i` in `T_x0 M` with the eigenvalues
    /// `l_i` give `q_i = l_i^(1/2) v_i . d`, and the harmonic energy on `M` is `|q|^2 / 2`: the
    /// spheres `|q| = R` are its levels and the ADD is `E - E0 - R^2 / 2` measured on `M`. A
    /// point of `M` with the coordinates `q` ([`Self::place`]) is `x(q) = x0 + W q + N0 c`
    /// (`W = V L^(-1/2)`, `N0` the normals of `M` at `x0`) with the amounts `c` along the
    /// normals solved so that the constraints hold: `M` as a graph over its tangent space at
    /// the minimum. As `V` is orthogonal to `N0`, the coordinates of `x(q)` are `q` exactly
    /// (up to the last restore to the floor of the constraints), so that every point
    /// evaluated on a sphere holds both the constraints and the radius. (Moving the linear
    /// point `x0 + W q` onto `M` by the restore of the constraints, or walking on `M` towards
    /// the coordinates, was tried first: along the soft modes the linear point lies far from
    /// `M`, and both then found no point for some coordinates of the test clusters.)
    pub fn on_manifold(
        origin: &Col<f64>,
        hessian: &Mat<f64>,
        removed: Option<&Mat<f64>>,
        manifold: Arc<PointManifold>,
    ) -> MoptResult<Self> {
        let n = origin.nrows();
        let normals = manifold.normals(origin)?;
        // The removed directions and the normals together, orthonormal.
        let mut columns: Vec<Col<f64>> = Vec::new();
        let given = removed.map_or(0, |d| d.ncols());
        for j in 0..given + normals.ncols() {
            let mut v = match removed {
                Some(d) if j < given => Col::from_fn(n, |i| d[(i, j)]),
                _ => Col::from_fn(n, |i| normals[(i, j - given)]),
            };
            for c in &columns {
                let d = col_dot(c, &v);
                v -= c * d;
            }
            let size = col_norm(&v);
            if size > 1e-10 {
                columns.push(v * (1.0 / size));
            }
        }
        let basis = Mat::from_fn(n, columns.len(), |i, j| columns[j][i]);
        let mut s = Self::new(origin, hessian, Some(&basis))?;
        s.manifold = Some(manifold);
        s.normals = normals;
        Ok(s)
    }

    /// The point of scaled coordinates `q`: [`Self::point`] without hard constraints, the
    /// point on them with these coordinates with them ([`Self::on_manifold`]): `x0 + W q +
    /// N0 c` with `c` solved by Newton's method so that the constraints hold (from the `c` of
    /// the nearest point placed before), then restored to their floor.
    pub fn place(&self, q: &Col<f64>) -> MoptResult<Col<f64>> {
        self.try_place(q)?.ok_or_else(|| {
            MoptError::calc(
                "scaled coordinates: no point on the constraints has these coordinates (the \
                 sphere has grown beyond where the constraints can be held across it)",
            )
        })
    }

    /// [`Self::place`], `None` where no point of the constraints has the coordinates `q`
    /// near the tangent space of the minimum (the constraints curve away from it: the
    /// spheres cannot be followed further on them).
    pub fn try_place(&self, q: &Col<f64>) -> MoptResult<Option<Col<f64>>> {
        let Some(m) = &self.manifold else {
            return Ok(Some(self.point(q)));
        };
        let placed = self
            .last_placed
            .lock()
            .map_err(|_| {
                MoptError::calc("scaled coordinates: the points placed are not available")
            })?
            .clone();
        // Coordinates placed before: the same point (a minimizer that evaluates its best
        // point once more must get the same energy).
        if let Some((_, _, x)) = placed.iter().rev().find(|(asked, _, _)| asked == q) {
            return Ok(Some(x.clone()));
        }
        let n0 = &self.normals;
        let mut c = placed
            .iter()
            .min_by(|a, b| col_norm(&(q - &a.0)).total_cmp(&col_norm(&(q - &b.0))))
            .map_or_else(|| Col::zeros(n0.ncols()), |p| p.1.clone());
        let base = self.point(q);
        let at = |c: &Col<f64>| &base + n0 * c;
        let size = |r: &Col<f64>| r.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let mut r = m.residuals(&at(&c))?;
        for _ in 0..PLACE_ITERATIONS {
            if size(&r) <= PLACE_TOLERANCE {
                break;
            }
            // Newton (least squares) on c: J(x) N0 dc = r.
            let b = m.jacobian(&at(&c))? * n0;
            // Singular: the normals of the minimum lie along the constraints here.
            let Ok(dc) = solve(&(b.transpose() * &b), &(b.transpose() * &r)) else {
                return Ok(None);
            };
            // Halved while it does not lower the residuals.
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..=PLACE_HALVINGS {
                let trial = &c - &dc * step;
                let rt = m.residuals(&at(&trial))?;
                if size(&rt) < size(&r) {
                    (c, r, accepted) = (trial, rt, true);
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                break;
            }
        }
        if size(&r) > PLACE_REACHED {
            // The Newton iterations stalled away from the constraints.
            return Ok(None);
        }
        // The floor of the constraints (and fixed variables bit-identical); the coordinates
        // stay those asked for to the precision of that last correction.
        let Ok(x) = m.restore(&at(&c)) else {
            return Ok(None);
        };
        self.keep((q.clone(), c, x.clone()));
        Ok(Some(x))
    }

    /// Keep `x`, a point on the constraints that the search knows (such as the second
    /// structure of 2PSHS), as a start of [`Self::place`] near its coordinates.
    pub fn remember(&self, x: &Col<f64>) {
        if self.manifold.is_some() {
            let q = self.coordinates(x);
            let c = self.normals.transpose() * (x - &self.point(&q));
            self.keep((q, c, x.clone()));
        }
    }

    /// Keep a placed point (the last [`PLACED_KEPT`]).
    fn keep(&self, point: PlacedPoint) {
        if let Ok(mut placed) = self.last_placed.lock() {
            placed.push(point);
            if placed.len() > PLACED_KEPT {
                placed.remove(0);
            }
        }
    }

    /// The gradient with respect to the scaled coordinates of the gradient `g` at the point
    /// `x` of [`Self::place`]: [`Self::gradient`] without hard constraints; with them the
    /// chain rule through the map `x(q) = x0 + W q + N0 c(q)` on the constraints, whose
    /// derivative is `D = W - N0 (J N0)^+ J W` (`J` the constraint gradients at `x`: `J D = 0`),
    /// so that `dE/dq = D^T g`.
    pub fn gradient_at(&self, x: &Col<f64>, g: &Col<f64>) -> MoptResult<Col<f64>> {
        let Some(m) = &self.manifold else {
            return Ok(self.gradient(g));
        };
        let (n, k) = (x.nrows(), self.values.len());
        let w = Mat::from_fn(n, k, |i, j| {
            self.vectors[(i, j)] / libm::sqrt(self.values[j])
        });
        let j = m.jacobian(x)?;
        let b = &j * &self.normals;
        let jw = &j * &w;
        let btb = b.transpose() * &b;
        let rhs = b.transpose() * &jw;
        let mut y = Mat::<f64>::zeros(self.normals.ncols(), k);
        for col in 0..k {
            let z = solve(&btb, &Col::from_fn(rhs.nrows(), |i| rhs[(i, col)]))?;
            for i in 0..z.nrows() {
                y[(i, col)] = z[i];
            }
        }
        let d = &w - &self.normals * &y;
        Ok(d.transpose() * g)
    }

    /// The coordinates of the minimum `origin` with the Hessian `hessian`; the directions of
    /// the columns of `removed` (orthonormal; the rigid-body motions, or none) are taken out.
    /// Fails when the Hessian has a negative eigenvalue beyond the zero modes (the point is
    /// not a minimum).
    pub fn new(
        origin: &Col<f64>,
        hessian: &Mat<f64>,
        removed: Option<&Mat<f64>>,
    ) -> MoptResult<Self> {
        let n = origin.nrows();
        if hessian.nrows() != n || hessian.ncols() != n || n == 0 {
            return Err(MoptError::config(format!(
                "scaled coordinates: a {}x{} Hessian for {n} variables",
                hessian.nrows(),
                hessian.ncols()
            )));
        }
        let projected = match removed {
            Some(d) if d.ncols() > 0 => project_out_basis_mat(d, hessian),
            _ => hessian.clone(),
        };
        let (values, vectors) = symmetric_eigen(&projected)?;
        let largest = (0..n).map(|k| values[k].abs()).fold(0.0, f64::max);
        if !(largest.is_finite() && largest > 0.0) {
            return Err(MoptError::calc(
                "scaled coordinates: the Hessian has no curvature",
            ));
        }
        let floor = ZERO_MODE_SHARE * largest;
        let lowest = (0..n).map(|k| values[k]).fold(f64::INFINITY, f64::min);
        if lowest < -floor {
            return Err(MoptError::config(format!(
                "scaled coordinates: the Hessian has a negative eigenvalue ({lowest:.3e}); the \
                 structure is not a minimum (optimize it first)"
            )));
        }
        let kept: Vec<usize> = (0..n).filter(|&k| values[k] > floor).collect();
        if kept.is_empty() {
            return Err(MoptError::calc("scaled coordinates: no vibrational mode"));
        }
        let mut order = kept;
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        Ok(Self {
            origin: origin.clone(),
            values: order.iter().map(|&k| values[k]).collect(),
            vectors: Mat::from_fn(n, order.len(), |i, j| vectors[(i, order[j])]),
            manifold: None,
            normals: Mat::zeros(n, 0),
            last_placed: Arc::default(),
        })
    }

    /// Number of scaled coordinates (vibrational modes).
    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    /// The minimum.
    pub fn origin(&self) -> &Col<f64> {
        &self.origin
    }

    /// The eigenvalues of the modes, ascending.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.values
    }

    /// The largest eigenvalue (the stiffest mode).
    pub fn stiffest(&self) -> f64 {
        self.values.last().copied().unwrap_or(0.0)
    }

    /// Radius of the sphere on which the displacement along the stiffest mode is `length`.
    pub fn radius_for_length(&self, length: f64) -> f64 {
        length * libm::sqrt(self.stiffest())
    }

    /// The point of scaled coordinates `q`.
    pub fn point(&self, q: &Col<f64>) -> Col<f64> {
        let n = self.origin.nrows();
        let mut x = self.origin.clone();
        for (j, &l) in self.values.iter().enumerate() {
            let c = q[j] / libm::sqrt(l);
            for i in 0..n {
                x[i] += c * self.vectors[(i, j)];
            }
        }
        x
    }

    /// The scaled coordinates of `x` (its part along the rigid-body motions is lost).
    pub fn coordinates(&self, x: &Col<f64>) -> Col<f64> {
        let d = x - &self.origin;
        Col::from_fn(self.values.len(), |j| {
            libm::sqrt(self.values[j])
                * (0..d.nrows())
                    .map(|i| self.vectors[(i, j)] * d[i])
                    .sum::<f64>()
        })
    }

    /// The gradient with respect to the scaled coordinates of the gradient `g` of `x`.
    pub fn gradient(&self, g: &Col<f64>) -> Col<f64> {
        Col::from_fn(self.values.len(), |j| {
            (0..g.nrows())
                .map(|i| self.vectors[(i, j)] * g[i])
                .sum::<f64>()
                / libm::sqrt(self.values[j])
        })
    }

    /// The scaled coordinates of a unit displacement along mode `k` (`sign` +1 or -1) on the
    /// sphere of radius `radius`.
    pub fn along_mode(&self, k: usize, sign: f64, radius: f64) -> Col<f64> {
        Col::from_fn(
            self.values.len(),
            |j| if j == k { sign * radius } else { 0.0 },
        )
    }
}

/// The solution of `a z = b` (an error where `a` is singular).
fn solve(a: &Mat<f64>, b: &Col<f64>) -> MoptResult<Col<f64>> {
    let k = b.nrows();
    let rhs = Mat::from_fn(k, 1, |i, _| b[i]);
    let lu = a.partial_piv_lu();
    let solution = faer::linalg::solvers::Solve::solve(&lu, rhs.as_ref());
    let out = Col::from_fn(k, |i| solution[(i, 0)]);
    if out.iter().any(|v| !v.is_finite()) {
        return Err(MoptError::calc(
            "scaled coordinates: the map onto the constraints is singular at this point",
        ));
    }
    Ok(out)
}

/// The angle between two points of scaled coordinates (0 if one is the origin).
pub fn angle(a: &Col<f64>, b: &Col<f64>) -> f64 {
    let (na, nb) = (col_norm(a), col_norm(b));
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    libm::acos((col_dot(a, b) / (na * nb)).clamp(-1.0, 1.0))
}

/// The elimination function of a maximal ADD point `center` (scaled coordinates) with the
/// ADD `add` (negative): `-add cos^3 theta` for `theta < pi/2`, else 0, and its gradient at
/// `q`.
pub fn elimination(q: &Col<f64>, center: &Col<f64>, add: f64) -> (f64, Col<f64>) {
    let m = q.nrows();
    let (nq, nc) = (col_norm(q), col_norm(center));
    if nq == 0.0 || nc == 0.0 || add >= 0.0 {
        return (0.0, Col::zeros(m));
    }
    let cos = (col_dot(q, center) / (nq * nc)).clamp(-1.0, 1.0);
    if cos <= 0.0 {
        return (0.0, Col::zeros(m));
    }
    // d cos / d q = c / (|q| |c|) - cos q / |q|^2
    let scale = -add * 3.0 * cos * cos;
    let gradient = Col::from_fn(m, |i| {
        scale * (center[i] / (nq * nc) - cos * q[i] / (nq * nq))
    });
    (-add * cos * cos * cos, gradient)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn hessian() -> Mat<f64> {
        Mat::from_fn(3, 3, |i, j| {
            [[2.0, 0.3, 0.0], [0.3, 0.5, 0.1], [0.0, 0.1, 1.2]][i][j]
        })
    }

    #[test]
    fn maps_are_inverse_and_spheres_are_harmonic_levels() {
        let x0 = Col::from_fn(3, |i| [0.1, -0.2, 0.3][i]);
        let s = ScaledCoordinates::new(&x0, &hessian(), None).unwrap();
        assert_eq!(s.dimension(), 3);
        let q = Col::from_fn(3, |i| [0.2, -0.1, 0.4][i]);
        let x = s.point(&q);
        let back = s.coordinates(&x);
        for i in 0..3 {
            assert!((back[i] - q[i]).abs() < 1e-12);
        }
        // Harmonic energy (x - x0)^T H (x - x0) / 2 = |q|^2 / 2.
        let d = &x - &x0;
        let harmonic = 0.5 * col_dot(&d, &(&hessian() * &d));
        assert!((harmonic - 0.5 * col_dot(&q, &q)).abs() < 1e-12);
        // The gradient of f(x) = a . x in scaled coordinates is the derivative along q.
        let a = Col::from_fn(3, |i| [0.4, 0.1, -0.3][i]);
        let gq = s.gradient(&a);
        for k in 0..3 {
            let mut up = q.clone();
            up[k] += 1e-6;
            let numeric = (col_dot(&a, &s.point(&up)) - col_dot(&a, &x)) / 1e-6;
            assert!((numeric - gq[k]).abs() < 1e-6);
        }
        // The displacement along the stiffest mode on a sphere.
        let r = s.radius_for_length(0.05);
        let along = s.point(&s.along_mode(2, 1.0, r));
        assert!((col_norm(&(&along - &x0)) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn points_of_the_constraints_have_their_coordinates_and_gradients() {
        use crate::domain::{positions, AtomId, Bohr, Element, MolecularSystem, SpinCharge};
        use crate::math::projection::build_tr_rot_basis;
        use crate::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
        let s = MolecularSystem::new(
            vec![Element::parse_token("Ar").unwrap(); 3],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [7.2, 0.0, 0.0], [3.6, 6.2, 0.3]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let c = vec![Constraint::at_start(ConstraintKind::Distance(
            vec![AtomId::new(0)],
            vec![AtomId::new(1)],
        ))];
        let set = ConstraintSet::new(&c, &s, None).unwrap();
        let m = Arc::new(PointManifold::new(set, s.clone()).unwrap());
        let x0 = s.coords_flat_bohr();
        let h = Mat::from_fn(
            9,
            9,
            |i, j| {
                if i == j {
                    1.0 + 0.1 * i as f64
                } else {
                    0.01
                }
            },
        );
        let rigid = build_tr_rot_basis(&s, false);
        let sc = ScaledCoordinates::on_manifold(&x0, &h, Some(&rigid), Arc::clone(&m)).unwrap();
        // 9 - 6 rigid - 1 constraint.
        assert_eq!(sc.dimension(), 2);
        let q = Col::from_fn(2, |i| [0.8, -0.5][i]);
        let x = sc.place(&q).unwrap();
        assert!(m.residuals(&x).unwrap()[0].abs() < 1e-10);
        assert!(col_norm(&(sc.coordinates(&x) - &q)) < 1e-10);
        // The gradient along the constraints by the chain rule, against differences.
        let a = Col::from_fn(9, |i| (0.3 * i as f64).sin());
        let g = sc.gradient_at(&x, &a).unwrap();
        for k in 0..2 {
            let mut up = q.clone();
            up[k] += 1e-5;
            let mut down = q.clone();
            down[k] -= 1e-5;
            let numeric = (col_dot(&a, &sc.place(&up).unwrap())
                - col_dot(&a, &sc.place(&down).unwrap()))
                / 2e-5;
            assert!((numeric - g[k]).abs() < 1e-6, "{k}: {numeric} {}", g[k]);
        }
    }

    #[test]
    fn rigid_directions_are_left_out_and_saddles_refused() {
        let x0 = Col::zeros(3);
        let e = Mat::from_fn(3, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        let s = ScaledCoordinates::new(&x0, &hessian(), Some(&e)).unwrap();
        assert_eq!(s.dimension(), 2);
        let saddle = Mat::from_fn(2, 2, |i, j| if i == j { [1.0, -0.5][i] } else { 0.0 });
        assert!(ScaledCoordinates::new(&Col::zeros(2), &saddle, None).is_err());
    }

    #[test]
    fn elimination_against_differences() {
        let c = Col::from_fn(3, |i| [1.0, 0.2, -0.1][i]);
        let q = Col::from_fn(3, |i| [0.8, 0.5, 0.1][i]);
        let (f, g) = elimination(&q, &c, -0.02);
        assert!(f > 0.0);
        for k in 0..3 {
            let mut up = q.clone();
            up[k] += 1e-6;
            let mut down = q.clone();
            down[k] -= 1e-6;
            let numeric = (elimination(&up, &c, -0.02).0 - elimination(&down, &c, -0.02).0) / 2e-6;
            assert!((numeric - g[k]).abs() < 1e-8);
        }
        // Beyond a right angle there is nothing.
        let away = Col::from_fn(3, |i| -c[i]);
        assert_eq!(elimination(&away, &c, -0.02).0, 0.0);
        assert!((angle(&c, &away) - std::f64::consts::PI).abs() < 1e-7);
    }
}
