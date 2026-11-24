//! MultiOptPy's Schlegel model Hessian with its D3 term (`ModelHessian/schlegeld3.py`),
//! reproduced as it is: the pair-distance form of `schlegel.py` with a seven-row B table that
//! classifies Na-Ar correctly, a damped `0.01 x 42 C6 / r^8` "D3" term (MultiOptPy's `r4r2`-based
//! `C6`) added to bond (x 1), angle (x 0.2), and dihedral (x 0.05) contributions, and C-N triple
//! bonds with stretch x 2, angle x 1.5, and torsion x 0.5.
//!
//! MultiOptPy: `ModelHessian/schlegeld3.py:SchlegelD3ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::common::{
    coords, cov_radius, mop_vdw_distance_bohr, norm, project_tr_rot, sub, Connectivity, V3,
};
use super::fischerd3::{d3_r4r2, A1, A2, S6};
use super::schlegel::{pair_distance_hessian, pair_index};
use super::swartd3::cyano_groups;
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

/// Rows: H-He 0, Li-Ne 1, Na-Ar 2, K-Kr 3, Rb-Xe 4, Cs-Rn 5, others 6.
fn row(el: Element) -> usize {
    match el.atomic_number() {
        1..=2 => 0,
        3..=10 => 1,
        11..=18 => 2,
        19..=36 => 3,
        37..=54 => 4,
        55..=86 => 5,
        _ => 6,
    }
}

/// `calc_d3_correction` (already scaled by 0.01).
pub(crate) fn d3_correction(r: f64, a: Element, b: Element) -> f64 {
    let (ra, rb) = (d3_r4r2(a), d3_r4r2(b));
    let c6 = (ra * ra * (rb * rb)).sqrt();
    let r0 = (mop_vdw_distance_bohr(a) * mop_vdw_distance_bohr(b)).sqrt();
    let damp = 1.0 / (1.0 + 6.0 * (r / (A1 * r0)).powf(A2));
    S6 * c6 * (42.0 / r.powi(8)) * damp * 0.01
}

/// Pair constants of `SchlegelD3ApproxHessian.guess_schlegel_hessian`.
fn constants(el: &[Element], x: &[V3]) -> Vec<f64> {
    let n = x.len();
    let mut f = vec![0.0; n * n.saturating_sub(1) / 2];
    let cyano = cyano_groups(el, x);
    let in_cn = |a: usize| cyano.iter().any(|&(c, nn)| a == c || a == nn);
    let dist = |a: usize, b: usize| norm(sub(x[a], x[b]));
    let bc = Connectivity::new(el, x);
    for [i, j] in bc.bonds() {
        let r = dist(i, j);
        let fb = 1.734 / (r - B[row(el[i])][row(el[j])]).powi(3);
        let is_cn = cyano
            .iter()
            .any(|&(c, nn)| (i == c && j == nn) || (i == nn && j == c));
        let d3 = d3_correction(r, el[i], el[j]);
        f[pair_index(n, i, j)] += if is_cn { 2.0 * fb } else { fb } + d3;
    }
    for [i, j, k] in bc.angles() {
        let fa = if el[i] == Element::H || el[k] == Element::H {
            0.160
        } else {
            0.250
        };
        let fa = if in_cn(i) || in_cn(j) || in_cn(k) {
            1.5 * fa
        } else {
            fa
        };
        f[pair_index(n, i, j)] += fa + d3_correction(dist(i, j), el[i], el[j]) * 0.2;
        f[pair_index(n, j, k)] += fa + d3_correction(dist(j, k), el[j], el[k]) * 0.2;
    }
    for [i, j, k, l] in bc.dihedrals() {
        let ft = 0.0023 - 0.07 * (dist(j, k) - (cov_radius(el[j]) + cov_radius(el[k])));
        let ft = if [i, j, k, l].iter().any(|&a| in_cn(a)) {
            0.5 * ft
        } else {
            ft
        };
        f[pair_index(n, i, j)] += ft + d3_correction(dist(i, j), el[i], el[j]) * 0.05;
        f[pair_index(n, j, k)] += ft + d3_correction(dist(j, k), el[j], el[k]) * 0.05;
        f[pair_index(n, k, l)] += ft + d3_correction(dist(k, l), el[k], el[l]) * 0.05;
    }
    f
}

/// MultiOptPy's `SchlegelD3ApproxHessian.main`.
pub(crate) fn schlegeld3(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let f = constants(&sys.elements, &x);
    project_tr_rot(&pair_distance_hessian(&x, &f), &x)
}
