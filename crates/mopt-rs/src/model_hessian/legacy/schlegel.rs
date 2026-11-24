//! MultiOptPy's Schlegel model Hessian (`ModelHessian/schlegel.py`), reproduced as it is.
//!
//! Unlike the published model (Schlegel, Theor. Chim. Acta 1984, 66, 333), MultiOptPy works on
//! the distances of all atom pairs: a bond adds `1.734 / (r - B)^3` to its distance, an angle
//! adds 0.160 (with a terminal hydrogen) or 0.250 to both of its bond distances, and a dihedral
//! adds `0.0023 - 0.07 (r - r_cov)` (central bond) to its three bond distances; the Cartesian
//! Hessian is `B^T diag(F) B` with the first-derivative B-matrix of the distances. Its row table
//! skips the elements Na-Ar (they get the last row).
//!
//! MultiOptPy: `ModelHessian/schlegel.py:SchlegelApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{coords, cov_radius, norm, project_tr_rot, sub, Connectivity, V3};
use crate::domain::{Element, MolecularSystem};

const B: [[f64; 7]; 7] = [
    [0.2573, 0.3401, 0.6937, 0.7126, 0.8335, 0.9491, 0.9491],
    [0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190, 1.7190],
    [0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185, 2.3185],
    [0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206, 2.5206],
    [0.8335, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110, 2.5110],
    [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110],
    [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110],
];

/// MultiOptPy's row of the B table: H-He 0, Li-Ne 1, K-Kr 2, Rb-Xe 3, Cs-Rn 4, others 5.
pub(crate) fn schlegel_row(el: Element) -> usize {
    match el.atomic_number() {
        1..=2 => 0,
        3..=10 => 1,
        19..=36 => 2,
        37..=54 => 3,
        55..=86 => 4,
        _ => 5,
    }
}

/// Index of the pair `(i, j)`, `i < j`, in `itertools.combinations(range(n), 2)` order.
pub(crate) fn pair_index(n: usize, i: usize, j: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    i * (2 * n - i - 1) / 2 + (j - i - 1)
}

/// Force constants of all pair distances.
pub(crate) fn schlegel_pair_constants(elements: &[Element], x: &[V3]) -> Vec<f64> {
    let n = x.len();
    let mut f = vec![0.0; n * (n.saturating_sub(1)) / 2];
    let bc = Connectivity::new(elements, x);
    for [i, j] in bc.bonds() {
        let r = norm(sub(x[i], x[j]));
        let b = B[schlegel_row(elements[i])][schlegel_row(elements[j])];
        f[pair_index(n, i, j)] += 1.734 / (r - b).powi(3);
    }
    for [i, j, k] in bc.angles() {
        let c = if elements[i] == Element::H || elements[k] == Element::H {
            0.160
        } else {
            0.250
        };
        f[pair_index(n, i, j)] += c;
        f[pair_index(n, j, k)] += c;
    }
    for [i, j, k, l] in bc.dihedrals() {
        let r = norm(sub(x[j], x[k]));
        let c = 0.0023 - 0.07 * (r - (cov_radius(elements[j]) + cov_radius(elements[k])));
        f[pair_index(n, i, j)] += c;
        f[pair_index(n, j, k)] += c;
        f[pair_index(n, k, l)] += c;
    }
    f
}

/// `B^T diag(f) B` for the distances of all pairs (NaN entries set to zero).
pub(crate) fn pair_distance_hessian(x: &[V3], f: &[f64]) -> Mat<f64> {
    let n = x.len();
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    for i in 0..n {
        for j in i + 1..n {
            let k = f[pair_index(n, i, j)];
            if k == 0.0 {
                continue;
            }
            let d = sub(x[i], x[j]);
            let r = norm(d);
            let u = [d[0] / r, d[1] / r, d[2] / r];
            for a in 0..3 {
                for b in 0..3 {
                    let v = k * u[a] * u[b];
                    h[(3 * i + a, 3 * i + b)] += v;
                    h[(3 * j + a, 3 * j + b)] += v;
                    h[(3 * i + a, 3 * j + b)] -= v;
                    h[(3 * j + a, 3 * i + b)] -= v;
                }
            }
        }
    }
    for i in 0..3 * n {
        for j in 0..3 * n {
            if h[(i, j)].is_nan() {
                h[(i, j)] = 0.0;
            }
        }
    }
    h
}

/// MultiOptPy's `SchlegelApproxHessian.main`.
pub(crate) fn schlegel(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let f = schlegel_pair_constants(&sys.elements, &x);
    project_tr_rot(&pair_distance_hessian(&x, &f), &x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_indices_follow_itertools_combinations() {
        let n = 5;
        let mut k = 0;
        for i in 0..n {
            for j in i + 1..n {
                assert_eq!(pair_index(n, i, j), k);
                assert_eq!(pair_index(n, j, i), k);
                k += 1;
            }
        }
    }
}
