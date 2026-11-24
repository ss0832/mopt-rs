//! Geodesic step: the Cartesian step is replaced by the end point of the geodesic that starts
//! along it, in the metric induced by the bond lengths.
//!
//! The internal coordinates `q(x)` (here: the distances of bonded atom pairs) pull their
//! Euclidean metric back to Cartesian space, `G = B^T B` with `B = dq/dx` (Zhu, Thompson, and
//! Martinez, J. Chem. Phys. 2019, 150, 164103). A geodesic `x(t)` of this metric satisfies
//! `B^T (B x'' + w) = 0`, `w_k = x'^T (d^2 q_k / dx^2) x'`,
//! i.e. the image curve `q(x(t))` has no acceleration within the manifold. Along it every bond
//! length changes linearly, `q_k(t) = q_k(0) + t (B x'(0))_k`, as far as the bonds are
//! independent, and the Cartesian acceleration `x'' = -B^+ w` is the smallest that achieves
//! this. A step that rotates a bond therefore follows the arc instead of the tangent and does
//! not stretch the bond. The equation is integrated from `t = 0` to `1` with the classical
//! Runge-Kutta method. For a bond `k` between atoms `i` and `j` with unit vector `u` and length
//! `r`: `w_k = (|v_ij|^2 - (u . v_ij)^2) / r` with `v_ij = v_i - v_j`.
//!
//! Bonds are the pairs closer than `bond_scale = 1.5` times the sum of the covalent radii at
//! the current structure. Without atoms, in periodic cells, in non-Cartesian coordinates, and
//! when there is no bond, the step passes unchanged.
//!
//! MultiOptPy integrates `q'' = -Gamma q' q'` in the space of the bond lengths with "Christoffel
//! symbols" that are not those of any metric (`Gamma_ijk = G^-1_ik sum_a dB_i,a. B_j`), and maps
//! the result back with the pseudo-inverse of `B`; see `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/geodesic_step.py:GeodesicStepper`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::StepEnhancer;
use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::math::simplex_qp::least_squares;
use crate::opt::step::StepInput;
use crate::parameters::covalent::sum_covalent_radii_bohr;

/// Geodesic step enhancer.
#[derive(Debug, Clone, Copy)]
pub struct GeodesicStep {
    /// Pairs closer than this factor times the sum of their covalent radii are bonds.
    pub bond_scale: f64,
    /// Number of Runge-Kutta steps.
    pub substeps: usize,
}

impl Default for GeodesicStep {
    fn default() -> Self {
        Self {
            bond_scale: 1.5,
            substeps: 20,
        }
    }
}

/// Bonded pairs of `sys` for the given scale of the covalent radii.
pub fn bonded_pairs(sys: &MolecularSystem, bond_scale: f64) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in sys.atoms() {
        for j in sys.atoms().skip(i.index() + 1) {
            if sys.element(i).is_dummy() || sys.element(j).is_dummy() {
                continue;
            }
            let limit = bond_scale * sum_covalent_radii_bohr(sys.element(i), sys.element(j));
            let r = sys.distance(i, j).value();
            if r > 1e-8 && r < limit {
                pairs.push((i.index(), j.index()));
            }
        }
    }
    pairs
}

/// Geodesic acceleration `x'' = -B^+ w` at position `x` with velocity `v` (flat `3N` vectors).
fn acceleration(x: &Col<f64>, v: &Col<f64>, pairs: &[(usize, usize)]) -> Option<Col<f64>> {
    let (m, n) = (pairs.len(), x.nrows());
    let mut b = Mat::<f64>::zeros(m, n);
    let mut w = vec![0.0; m];
    for (k, &(i, j)) in pairs.iter().enumerate() {
        let d = [
            x[3 * i] - x[3 * j],
            x[3 * i + 1] - x[3 * j + 1],
            x[3 * i + 2] - x[3 * j + 2],
        ];
        let r = libm::sqrt(d.iter().map(|c| c * c).sum::<f64>());
        if r < 1e-10 {
            continue;
        }
        let vij = [
            v[3 * i] - v[3 * j],
            v[3 * i + 1] - v[3 * j + 1],
            v[3 * i + 2] - v[3 * j + 2],
        ];
        let mut along = 0.0;
        for c in 0..3 {
            let u = d[c] / r;
            b[(k, 3 * i + c)] = u;
            b[(k, 3 * j + c)] = -u;
            along += u * vij[c];
        }
        let v2: f64 = vij.iter().map(|c| c * c).sum();
        w[k] = (v2 - along * along) / r;
    }
    // Minimum-norm solution of B a = -w: a = B^T (B B^T)^+ (-w).
    let gram = &b * b.transpose();
    let rhs: Vec<f64> = w.iter().map(|v| -v).collect();
    let lambda = least_squares(&gram, &rhs).ok()?;
    let mut a: Col<f64> = Col::zeros(n);
    for (k, l) in lambda.iter().enumerate() {
        for c in 0..n {
            a[c] += b[(k, c)] * l;
        }
    }
    let finite = a.iter().all(|v| v.is_finite());
    finite.then_some(a)
}

/// End point displacement `x(1) - x(0)` of the geodesic starting at `x0` with velocity `v0`
/// in the metric of the given bonds (`None` if the integration breaks down).
pub fn geodesic_displacement(
    x0: &Col<f64>,
    v0: &Col<f64>,
    pairs: &[(usize, usize)],
    substeps: usize,
) -> Option<Col<f64>> {
    let steps = substeps.max(1);
    let h = 1.0 / steps as f64;
    let (mut x, mut v) = (x0.clone(), v0.clone());
    for _ in 0..steps {
        let a1 = acceleration(&x, &v, pairs)?;
        let (x2, v2) = (&x + &v * (0.5 * h), &v + &a1 * (0.5 * h));
        let a2 = acceleration(&x2, &v2, pairs)?;
        let (x3, v3) = (&x + &v2 * (0.5 * h), &v + &a2 * (0.5 * h));
        let a3 = acceleration(&x3, &v3, pairs)?;
        let (x4, v4) = (&x + &v3 * h, &v + &a3 * h);
        let a4 = acceleration(&x4, &v4, pairs)?;
        x += (&v + &v2 * 2.0 + &v3 * 2.0 + &v4) * (h / 6.0);
        v += (&a1 + &a2 * 2.0 + &a3 * 2.0 + &a4) * (h / 6.0);
    }
    let out = x - x0;
    let finite = out.iter().all(|c| c.is_finite());
    finite.then_some(out)
}

impl StepEnhancer for GeodesicStep {
    fn name(&self) -> &str {
        "geodesic_step"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let Some(sys) = input.system else {
            return Ok(step);
        };
        if !input.cartesian_coordinates || sys.has_pbc() || step.nrows() != sys.dof() {
            return Ok(step);
        }
        let pairs = bonded_pairs(sys, self.bond_scale);
        if pairs.is_empty() {
            return Ok(step);
        }
        let x0 = sys.coords_flat_bohr();
        Ok(geodesic_displacement(&x0, &step, &pairs, self.substeps).unwrap_or(step))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};
    use crate::math::linalg::col_norm;
    use crate::opt::stepper::plain_input;

    fn dist(x: &Col<f64>, i: usize, j: usize) -> f64 {
        (0..3)
            .map(|c| (x[3 * i + c] - x[3 * j + c]).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn bond_lengths_change_linearly_along_the_geodesic() {
        // Water-like triangle: bonds O-H1 and O-H2.
        let x0 = Col::from_fn(9, |i| [0.0, 0.0, 0.0, 1.8, 0.0, 0.0, -0.5, 1.7, 0.0][i]);
        let pairs = [(0, 1), (0, 2)];
        // H1 moves perpendicular to its bond and stretches a little; H2 swings out of plane.
        let v = Col::from_fn(9, |i| [0.0, 0.0, 0.0, 0.1, 0.6, 0.0, 0.0, 0.0, 0.5][i]);
        let s = geodesic_displacement(&x0, &v, &pairs, 20).unwrap();
        let x1 = &x0 + &s;
        let straight = &x0 + &v;
        // Linearized changes: +0.1 for O-H1, 0 for O-H2.
        let want = [dist(&x0, 0, 1) + 0.1, dist(&x0, 0, 2)];
        let got = [dist(&x1, 0, 1), dist(&x1, 0, 2)];
        let line = [dist(&straight, 0, 1), dist(&straight, 0, 2)];
        for k in 0..2 {
            assert!(
                (got[k] - want[k]).abs() < 1e-6,
                "{k}: {} vs {}",
                got[k],
                want[k]
            );
            assert!(
                (line[k] - want[k]).abs() > 0.05,
                "the straight step stretches bond {k}"
            );
        }
        // A pure stretch is already a geodesic.
        let stretch = Col::from_fn(9, |i| [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0][i]);
        let s = geodesic_displacement(&x0, &stretch, &[(0, 1)], 20).unwrap();
        assert!(col_norm(&(&s - &stretch)) < 1e-12);
    }

    #[test]
    fn steps_pass_unchanged_without_bonds_or_atoms() {
        let h = Mat::<f64>::identity(6, 6);
        let g = Col::zeros(6);
        let step = Col::from_fn(6, |i| 0.1 * i as f64);
        let mut geo = GeodesicStep::default();
        let x = Col::zeros(6);
        let mut input = plain_input(0, &x, 0.0, &g, &h, None, 0);
        assert_eq!(geo.enhance(&input, step.clone()).unwrap(), step);
        // Two argon atoms at 7 Bohr: farther apart than 1.5 times the covalent radii.
        let ar = Element::parse_token("Ar").unwrap();
        let far = MolecularSystem::new(
            vec![ar, ar],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert!(bonded_pairs(&far, 1.5).is_empty());
        input.system = Some(&far);
        assert_eq!(geo.enhance(&input, step.clone()).unwrap(), step);
        // A bonded pair: a sideways step keeps the bond length.
        let near = MolecularSystem::new(
            vec![Element::H, Element::H],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        input.system = Some(&near);
        let side = Col::from_fn(6, |i| [0.0, 0.0, 0.0, 0.0, 0.5, 0.0][i]);
        let s = geo.enhance(&input, side).unwrap();
        let x1 = &near.coords_flat_bohr() + &s;
        assert!((dist(&x1, 0, 1) - 1.4).abs() < 1e-6);
    }
}
