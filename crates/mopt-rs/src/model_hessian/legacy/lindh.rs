//! MultiOptPy's Lindh model Hessian (`ModelHessian/lindh.py`), reproduced as it is, with the
//! internal-coordinate machinery it relies on (`Coordinate/redundant_coordinate.py`).
//!
//! MultiOptPy differs from Lindh et al. (Chem. Phys. Lett. 1995, 241, 423) in several ways that
//! are kept here:
//!
//! - the coordinates are the distances of all atom pairs; angle and dihedral force constants are
//!   added to the distances of their bonds, bond constants are divided by the reduced mass (u),
//!   and the reference distances are sums of covalent radii (not Lindh's `r_ref` table);
//! - non-bonded pairs get UFF Lennard-Jones and Coulomb (`664.12 q_i q_j / r^3`) curvatures;
//! - a gradient term `sum_k g_k d^2 q_k / dx^2` is added, where the "internal gradient" solves
//!   `(B B^T) g_int = B g` for the pair distances (singular from five atoms on, where the
//!   result depends on rounding) and its entries are paired with the bonds, angles, and
//!   dihedrals in order rather than with the distances they belong to.
//!
//! The golden test compares with MultiOptPy where its linear system is well posed.
//!
//! MultiOptPy: `ModelHessian/lindh.py:LindhApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{
    coords, cov_radius, mop_vdw_distance_bohr, mop_vdw_well_depth_hartree, norm, project_tr_rot,
    sub, Connectivity, V3,
};
use super::schlegel::{pair_distance_hessian, pair_index};
use crate::domain::{Element, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_grad_hessian, Scalar};
use crate::parameters::isotope_masses::MOST_ABUNDANT_ISOTOPE_MASS_U;
use crate::parameters::multioptpy_tables::{MOP_BOHR2ANGSTROMS, MOP_HARTREE2KCALMOL};
use crate::parameters::uff_effective_charge;

const ALPHA: [[f64; 3]; 3] = [
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.2800],
    [0.3949, 0.2800, 0.2800],
];

fn lindh_row(el: Element) -> usize {
    match el.atomic_number() {
        1..=2 => 0,
        3..=10 => 1,
        _ => 2,
    }
}

fn isotope_mass(el: Element) -> f64 {
    MOST_ABUNDANT_ISOTOPE_MASS_U
        .get(el.atomic_number() as usize)
        .copied()
        .unwrap_or(0.0)
}

/// `TorchDerivatives.distance`.
pub(crate) fn torch_distance<T: Scalar>(p: &[T]) -> T {
    let d = [p[0] - p[3], p[1] - p[4], p[2] - p[5]];
    T::dot3(d, d).sqrt()
}

/// `TorchDerivatives.angle` (with its `1e-15` in the denominator).
pub(crate) fn torch_angle<T: Scalar>(p: &[T]) -> T {
    let v1 = [p[0] - p[3], p[1] - p[4], p[2] - p[5]];
    let v2 = [p[6] - p[3], p[7] - p[4], p[8] - p[5]];
    let c =
        T::dot3(v1, v2) / (T::dot3(v1, v1).sqrt() * T::dot3(v2, v2).sqrt() + T::from_f64(1e-15));
    c.acos()
}

fn cross<T: Scalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `TorchDerivatives.dihedral_angle` (the unsigned angle, with MultiOptPy's normalization).
pub(crate) fn torch_dihedral<T: Scalar>(p: &[T]) -> T {
    let at = |k: usize| [p[3 * k], p[3 * k + 1], p[3 * k + 2]];
    let s = |a: [T; 3], b: [T; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let (a1, a2, a3) = (s(at(1), at(0)), s(at(2), at(1)), s(at(3), at(2)));
    let unit = |v: [T; 3]| {
        let n = T::dot3(v, v).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    };
    let v1 = unit(cross(a1, a2));
    let v2 = unit(cross(a2, a3));
    let s22 = T::dot3(v2, v2);
    let eps = T::from_f64(1e-15);
    let den =
        (v1[0] * v1[0] * s22 + eps) + (v1[1] * v1[1] * s22 + eps) + (v1[2] * v1[2] * s22 + eps);
    let c = T::dot3(v1, v2) / den.sqrt();
    c.acos().abs()
}

fn local_hessian(
    x: &[V3],
    atoms: &[usize],
    f: fn(&[crate::math::ad::HyperDual]) -> crate::math::ad::HyperDual,
) -> Mat<f64> {
    let flat: Vec<f64> = atoms.iter().flat_map(|&a| x[a]).collect();
    eval_value_grad_hessian(&flat, None, f).2
}

/// `RedundantInternalCoordinates.cartgrad2RICgrad`: `(B B^T)^-1 B g` for the pair distances,
/// by LU decomposition with partial pivoting (as `numpy.linalg.solve`).
pub(crate) fn pair_gradient(x: &[V3], g: &[f64]) -> MoptResult<Vec<f64>> {
    let n = x.len();
    let m = n * n.saturating_sub(1) / 2;
    let mut rows: Vec<(usize, usize, V3)> = Vec::with_capacity(m);
    for i in 0..n {
        for j in i + 1..n {
            let d = sub(x[i], x[j]);
            let r = norm(d);
            rows.push((i, j, [d[0] / r, d[1] / r, d[2] / r]));
        }
    }
    let bvec = |k: usize, dof: usize| -> f64 {
        let (i, j, u) = rows[k];
        let (a, c) = (dof / 3, dof % 3);
        if a == i {
            u[c]
        } else if a == j {
            -u[c]
        } else {
            0.0
        }
    };
    let mut a = vec![vec![0.0; m]; m];
    let mut rhs = vec![0.0; m];
    for p in 0..m {
        let (i, j, _) = rows[p];
        for dof in [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2] {
            rhs[p] += bvec(p, dof) * g[dof];
        }
        for q in 0..m {
            let (k, l, _) = rows[q];
            let mut s = 0.0;
            for dof in [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2] {
                let a_dof = dof / 3;
                if a_dof == k || a_dof == l {
                    s += bvec(p, dof) * bvec(q, dof);
                }
            }
            a[p][q] = s;
        }
    }
    // Gaussian elimination with partial pivoting.
    let mut perm: Vec<usize> = (0..m).collect();
    for col in 0..m {
        let piv = (col..m)
            .max_by(|&r1, &r2| {
                a[r1][col]
                    .abs()
                    .total_cmp(&a[r2][col].abs())
                    .then(r2.cmp(&r1))
            })
            .unwrap_or(col);
        if a[piv][col] == 0.0 {
            return Err(MoptError::calc(
                "legacy Lindh: singular internal-coordinate metric (MultiOptPy fails here too)",
            ));
        }
        a.swap(col, piv);
        rhs.swap(col, piv);
        perm.swap(col, piv);
        let (upper, lower) = a.split_at_mut(col + 1);
        let pivot_row = &upper[col];
        for (off, row) in lower.iter_mut().enumerate() {
            let r = col + 1 + off;
            let f = row[col] / pivot_row[col];
            if f != 0.0 {
                for (v, p) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                    *v -= f * p;
                }
                rhs[r] -= f * rhs[col];
            }
        }
    }
    let mut y = vec![0.0; m];
    for r in (0..m).rev() {
        let s: f64 = (r + 1..m).map(|c| a[r][c] * y[c]).sum();
        y[r] = (rhs[r] - s) / a[r][r];
    }
    Ok(y)
}

/// `RIChess2carthess`'s gradient term: `d^2 q / dx^2` of the bonds, angles, and dihedrals in
/// order, each times the internal gradient entry of the same running index.
pub(crate) fn gradient_term(x: &[V3], bc: &Connectivity, gint: &[f64]) -> MoptResult<Mat<f64>> {
    let n = x.len();
    let mut k = Mat::<f64>::zeros(3 * n, 3 * n);
    let mut count = 0usize;
    let mut add = |atoms: &[usize], h: Mat<f64>, count: &mut usize| -> MoptResult<()> {
        let w = *gint.get(*count).ok_or_else(|| {
            MoptError::calc("legacy model Hessian: more internal coordinates than pair gradients (IndexError in MultiOptPy)")
        })?;
        for (ia, &a) in atoms.iter().enumerate() {
            for (ib, &b) in atoms.iter().enumerate() {
                for p in 0..3 {
                    for q in 0..3 {
                        k[(3 * a + p, 3 * b + q)] += h[(3 * ia + p, 3 * ib + q)] * w;
                    }
                }
            }
        }
        *count += 1;
        Ok(())
    };
    for b in bc.bonds() {
        add(&b, local_hessian(x, &b, torch_distance), &mut count)?;
    }
    for a in bc.angles() {
        add(&a, local_hessian(x, &a, torch_angle), &mut count)?;
    }
    for d in bc.dihedrals() {
        add(&d, local_hessian(x, &d, torch_dihedral), &mut count)?;
    }
    Ok(k)
}

/// Pair constants of `guess_lindh_hessian`.
fn lindh_pair_constants(elements: &[Element], x: &[V3], bc: &Connectivity) -> Vec<f64> {
    let n = x.len();
    let mut f = vec![0.0; n * n.saturating_sub(1) / 2];
    let chain = |idx: &[usize], k0: f64| -> f64 {
        let mut k = k0;
        for w in idx.windows(2) {
            let (a, b) = (w[0], w[1]);
            let r0 = cov_radius(elements[a]) + cov_radius(elements[b]);
            let alpha = ALPHA[lindh_row(elements[a])][lindh_row(elements[b])];
            let r = norm(sub(x[a], x[b]));
            k *= (alpha * (r0 * r0 - r * r)).exp();
        }
        k
    };
    let bonds = bc.bonds();
    for b in &bonds {
        let k = chain(b, 0.45);
        let (m1, m2) = (isotope_mass(elements[b[0]]), isotope_mass(elements[b[1]]));
        f[pair_index(n, b[0], b[1])] += k / (m1 * m2 / (m1 + m2));
    }
    for a in bc.angles() {
        let k = chain(&a, 0.15);
        f[pair_index(n, a[0], a[1])] += k;
        f[pair_index(n, a[1], a[2])] += k;
    }
    for d in bc.dihedrals() {
        let k = chain(&d, 0.005);
        f[pair_index(n, d[0], d[1])] += k;
        f[pair_index(n, d[1], d[2])] += k;
        f[pair_index(n, d[2], d[3])] += k;
    }
    let es_unit = MOP_BOHR2ANGSTROMS * MOP_BOHR2ANGSTROMS / MOP_HARTREE2KCALMOL;
    for i in 0..n {
        for j in i + 1..n {
            if bonds.contains(&[i, j]) {
                continue;
            }
            let r = norm(sub(x[i], x[j]));
            let eps = (mop_vdw_well_depth_hartree(elements[i])
                * mop_vdw_well_depth_hartree(elements[j]))
            .sqrt();
            let sigma =
                (mop_vdw_distance_bohr(elements[i]) * mop_vdw_distance_bohr(elements[j])).sqrt();
            let lj = -12.0
                * eps
                * (-7.0 * (sigma.powi(6) / r.powi(8)) + 13.0 * (sigma.powi(12) / r.powi(14)));
            let es = 664.12
                * (uff_effective_charge(elements[i]) * uff_effective_charge(elements[j])
                    / r.powi(3))
                * es_unit;
            f[pair_index(n, i, j)] += lj + es;
        }
    }
    f
}

/// MultiOptPy's `LindhApproxHessian.main`.
pub(crate) fn lindh(sys: &MolecularSystem, gradient: &[f64]) -> MoptResult<Mat<f64>> {
    let x = coords(sys);
    let bc = Connectivity::new(&sys.elements, &x);
    let f = lindh_pair_constants(&sys.elements, &x, &bc);
    let gint = pair_gradient(&x, gradient)?;
    let mut h = pair_distance_hessian(&x, &f);
    h += gradient_term(&x, &bc, &gint)?;
    for i in 0..h.nrows() {
        for j in 0..h.ncols() {
            if h[(i, j)].is_nan() {
                h[(i, j)] = 0.0;
            }
        }
    }
    Ok(project_tr_rot(&h, &x))
}
