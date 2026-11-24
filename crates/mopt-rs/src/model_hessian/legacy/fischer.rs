//! MultiOptPy's Fischer model Hessian (`ModelHessian/fischer.py`), reproduced as it is.
//!
//! Differences from the published model (Fischer & Almlof, J. Phys. Chem. 1992, 96, 9768) kept
//! here: bonds, angles, and dihedrals come from MultiOptPy's connectivity (1.1 x covalent radii),
//! the bond count of a dihedral from a second criterion (1.3 x), and dihedrals with a bond angle
//! within about 1.8 degrees of linear are skipped.
//!
//! MultiOptPy: `ModelHessian/fischer.py:FischerApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{
    add_outer, bend2, coords, cov_radius, cross, dot, mirror_upper, norm, project_tr_rot, stretch2,
    sub, torsion2, Connectivity, V3,
};
use crate::domain::{Element, MolecularSystem};

fn sin_sq(x: &[V3], a: usize, b: usize, c: usize) -> f64 {
    let (v1, v2) = (sub(x[a], x[b]), sub(x[c], x[b]));
    let cr = cross(v1, v2);
    let (n1, n2) = (dot(v1, v1), dot(v2, v2));
    if n1 * n2 < 1e-12 {
        0.0
    } else {
        dot(cr, cr) / (n1 * n2)
    }
}

/// The unprojected Fischer Hessian (bonds, angles, dihedrals), upper triangle mirrored.
pub(crate) fn fischer_raw(elements: &[Element], x: &[V3]) -> Mat<f64> {
    let mut h = fischer_terms(elements, x);
    mirror_upper(&mut h);
    h
}

/// Bonds, angles, and dihedrals of `fischer.py` (also used by `fischerd4.py`).
pub(crate) fn fischer_terms(elements: &[Element], x: &[V3]) -> Mat<f64> {
    let n = x.len();
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    let rc = |i: usize, j: usize| cov_radius(elements[i]) + cov_radius(elements[j]);
    let bc = Connectivity::new(elements, x);
    for [i, j] in bc.bonds() {
        let r = norm(sub(x[i], x[j]));
        let k = 0.3601 * (-1.944 * (r - rc(i, j))).exp();
        let (_, b) = stretch2(x[i], x[j]);
        add_outer(&mut h, &[i, j], &b, k);
    }
    for [i, j, k] in bc.angles() {
        let (r_ij, r_jk) = (norm(sub(x[i], x[j])), norm(sub(x[j], x[k])));
        let (c_ij, c_jk) = (rc(i, j), rc(j, k));
        let val = c_ij * c_jk;
        let fc = if val.abs() < 1e-10 {
            0.0
        } else {
            0.089 + 0.11 / val.powf(-0.42) * (-0.44 * (r_ij + r_jk - c_ij - c_jk)).exp()
        };
        let (_, b) = bend2(x[i], x[j], x[k]);
        add_outer(&mut h, &[i, j, k], &b, fc);
    }
    let count = Connectivity::with_factor(elements, x, 1.3);
    for [i, j, k, l] in bc.dihedrals() {
        if sin_sq(x, i, j, k) < 1e-3 || sin_sq(x, j, k, l) < 1e-3 {
            continue;
        }
        let r = norm(sub(x[j], x[k]));
        let c = rc(j, k);
        let bonds = count.bonded[j].iter().filter(|&&v| v).count()
            + count.bonded[k].iter().filter(|&&v| v).count();
        let bond_sum = bonds as f64 - 2.0;
        let val = r * c;
        let fc = if val.abs() < 1e-10 {
            0.0
        } else {
            0.0015 + 14.0 * bond_sum.max(0.0).powf(0.57) / val.powi(4) * (-2.85 * (r - c)).exp()
        };
        let (_, b) = torsion2([x[i], x[j], x[k], x[l]]);
        if b.iter().flatten().all(|v| v.is_finite()) {
            add_outer(&mut h, &[i, j, k, l], &b, fc);
        }
    }
    h
}

/// MultiOptPy's `FischerApproxHessian.main`.
pub(crate) fn fischer(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    project_tr_rot(&fischer_raw(&sys.elements, &x), &x)
}
