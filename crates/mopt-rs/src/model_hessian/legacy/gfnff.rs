//! MultiOptPy's GFN-FF-like model Hessian, reproduced as it is.
//!
//! MultiOptPy: `ModelHessian/gfnff.py` (`GFNFFApproxHessian`) with the parameters of
//! `Parameters/gfnff_param.py` (`GFNFFParameters`), generated into
//! [`crate::parameters::multioptpy_gfn`].
//!
//! Provenance: this is MultiOptPy's own approximate model, inspired by GFN-FF (S. Spicher,
//! S. Grimme, Angew. Chem. Int. Ed. 2020, 59, 15665). It uses neither the GFN-FF energy
//! expression nor the published GFN-FF parameters: the bond, angle, torsion, hydrogen-bond, and
//! non-bonded force constants are MultiOptPy's estimates (some tables are labelled in MultiOptPy
//! as taken from `gfnff_param.f90`; they are reproduced as MultiOptPy has them). mopt-rs offers
//! it for the elements of MultiOptPy's element table, H to Rn
//! ([`super::GFNFF_LIKE_RANGE`]).
//!
//! Reproduced as in MultiOptPy, including:
//! - partial charges that count every bonded pair twice (both loop orders);
//! - hydrogen bonds detected only when the angle between D->H and H->A exceeds 120 degrees
//!   (bent arrangements; linear hydrogen bonds are not detected);
//! - bond parameters looked up with sorted symbols (so table entries whose key is not sorted,
//!   such as N-H, fall back to the covalent-radius estimate);
//! - after every torsion term the upper triangle is copied to the lower one, and the final
//!   matrix copies the lower triangle to the upper one.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use super::common::{coords, cross, dot, norm, project_tr_rot, sub, V3};
use crate::domain::{Element, MolecularSystem};
use crate::math::stable::logistic;
use crate::parameters::multioptpy_gfn as t;
use crate::parameters::multioptpy_tables::{MOP_BOHR2ANGSTROMS as B2A, MOP_HARTREE2KCALMOL};

const BOND_FACTOR: f64 = 1.3;

fn kcal2hartree() -> f64 {
    1.0 / MOP_HARTREE2KCALMOL
}

fn scale(v: V3, s: f64) -> V3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// An element table entry, or MultiOptPy's default when it has none.
fn table(values: &[f64], el: Element, default: f64) -> f64 {
    values
        .get(el.atomic_number() as usize)
        .copied()
        .filter(|v| !v.is_nan())
        .unwrap_or(default)
}

/// MultiOptPy: `GFNFFParameters.get_cov_radius` (Angstrom).
fn cov(el: Element) -> f64 {
    table(&t::GFNFF_COV_RADIUS_ANG, el, 1.0)
}

/// MultiOptPy: `GFNFFParameters.get_vdw_radius` (Angstrom).
fn vdw(el: Element) -> f64 {
    table(&t::GFNFF_VDW_RADIUS_ANG, el, 2.0)
}

/// MultiOptPy: `GFNFFParameters.get_electronegativity`.
fn en(el: Element) -> f64 {
    table(&t::GFNFF_ELECTRONEGATIVITY, el, 2.0)
}

/// MultiOptPy: `GFNFFParameters.get_polarizability`.
fn alpha(el: Element) -> f64 {
    table(&t::GFNFF_POLARIZABILITY, el, 10.0)
}

/// MultiOptPy: `GFNFFParameters.ref_cn.get(el, 1.0)`.
fn ref_cn(el: Element) -> f64 {
    table(&t::GFNFF_REF_CN, el, 1.0)
}

/// MultiOptPy: `GFNFFParameters.get_bond_params` ([r0 Bohr, k]).
fn bond_params(a: Element, b: Element) -> [f64; 2] {
    let (s, u) = (a.symbol(), b.symbol());
    let (s, u) = if s <= u { (s, u) } else { (u, s) };
    t::gfnff_bond(s, u).unwrap_or([(cov(a) + cov(b)) / B2A, t::GFNFF_DEFAULT_BOND_K])
}

/// MultiOptPy: `GFNFFParameters.get_angle_params` ([theta0 degrees, k]).
fn angle_params(a: Element, center: Element, c: Element) -> [f64; 2] {
    let (s, m, u) = (a.symbol(), center.symbol(), c.symbol());
    t::gfnff_angle(s, m, u)
        .or_else(|| t::gfnff_angle(u, m, s))
        .unwrap_or_else(|| {
            let theta0 = match m {
                "C" | "Si" => 109.5,
                "N" | "P" => 107.0,
                "O" | "S" => 104.5,
                _ => 120.0,
            };
            [theta0, t::GFNFF_DEFAULT_ANGLE_K]
        })
}

/// MultiOptPy: `GFNFFParameters.get_torsion_params` ([V1, V2, V3] Hartree).
fn torsion_params(e: [Element; 4]) -> [f64; 3] {
    let s = e.map(Element::symbol);
    let v = t::gfnff_torsion(s[0], s[1], s[2], s[3])
        .or_else(|| t::gfnff_torsion(s[3], s[2], s[1], s[0]))
        .unwrap_or(t::GFNFF_DEFAULT_TORSION_V);
    v.map(|x| x * kcal2hartree())
}

fn hbond_known(donor: &str, acceptor: &str) -> bool {
    t::gfnff_hbond(acceptor, "H", donor).is_some() || t::gfnff_hbond(donor, "H", acceptor).is_some()
}

/// MultiOptPy: `GFNFFParameters.get_hbond_params` ([r0 Bohr, k Hartree]).
fn hbond_params(donor: Element, acceptor: Element) -> [f64; 2] {
    let (d, a) = (donor.symbol(), acceptor.symbol());
    t::gfnff_hbond(d, "H", a)
        .or_else(|| t::gfnff_hbond(a, "H", d))
        .map_or([2.0 / B2A, 2.0 * kcal2hartree()], |[r0, k]| {
            [r0 / B2A, k * kcal2hartree()]
        })
}

struct Topology {
    bonds: Vec<[usize; 2]>,
    angles: Vec<[usize; 3]>,
    dihedrals: Vec<[usize; 4]>,
    nb_pairs: Vec<[usize; 2]>,
    hbonds: Vec<[usize; 3]>,
    cn: Vec<f64>,
}

/// MultiOptPy: `GFNFFApproxHessian.calculate_coordination_numbers`.
fn coordination_numbers(x: &[V3], el: &[Element]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let r_cov = cov(el[i]) + cov(el[j]);
                    let dist = norm(sub(x[i], x[j])) * B2A;
                    // 1 / (1 + exp(-16 (1.2 r_cov / r - 1)))
                    logistic(16.0 * (r_cov * 1.2 / dist - 1.0))
                })
                .sum()
        })
        .collect()
}

/// MultiOptPy: `GFNFFApproxHessian.get_bond_connectivity`.
// Index loops as in MultiOptPy, for side-by-side reading.
#[allow(clippy::needless_range_loop)]
fn connectivity(x: &[V3], el: &[Element]) -> Vec<Vec<bool>> {
    let n = x.len();
    let mut bonded = vec![vec![false; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let cov_sum = (cov(el[i]) + cov(el[j])) / B2A;
            let b = norm(sub(x[i], x[j])) <= cov_sum * BOND_FACTOR;
            bonded[i][j] = b;
            bonded[j][i] = b;
        }
    }
    bonded
}

/// MultiOptPy: `GFNFFApproxHessian.estimate_atomic_charges`.
fn charges(x: &[V3], el: &[Element], bonded: &[Vec<bool>]) -> Vec<f64> {
    let n = x.len();
    let cn = coordination_numbers(x, el);
    let mut q = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if i == j || !bonded[i][j] {
                continue;
            }
            let r = norm(sub(x[i], x[j])) * B2A;
            let en_i = en(el[i]) * (-0.1 * (cn[i] - ref_cn(el[i])).powi(2)).exp();
            let en_j = en(el[j]) * (-0.1 * (cn[j] - ref_cn(el[j])).powi(2)).exp();
            let transfer = 0.1 * (en_j - en_i) / (r * (en_i + en_j));
            q[i] += transfer;
            q[j] -= transfer;
        }
    }
    let mean = q.iter().sum::<f64>() / n as f64;
    q.iter_mut().for_each(|v| *v -= mean);
    q
}

const HB_ATOMS: [&str; 7] = ["O", "N", "F", "Cl", "Br", "I", "S"];

/// MultiOptPy: `GFNFFApproxHessian.detect_hydrogen_bonds` (donor, H, acceptor).
fn hydrogen_bonds(x: &[V3], el: &[Element], bonded: &[Vec<bool>], q: &[f64]) -> Vec<[usize; 3]> {
    let n = x.len();
    let mut out = Vec::new();
    for i in (0..n).filter(|&i| el[i].symbol() == "H") {
        let Some(donor) = (0..n).find(|&j| bonded[i][j] && HB_ATOMS.contains(&el[j].symbol()))
        else {
            continue;
        };
        for acc in 0..n {
            if acc == donor || bonded[i][acc] || !HB_ATOMS.contains(&el[acc].symbol()) {
                continue;
            }
            if !hbond_known(el[donor].symbol(), el[acc].symbol()) {
                continue;
            }
            let h_acc_dist = norm(sub(x[i], x[acc])) * B2A;
            let d_h = sub(x[i], x[donor]);
            let h_acc = sub(x[acc], x[i]);
            let (ld, la) = (norm(d_h), norm(h_acc));
            if ld > 0.0 && la > 0.0 {
                let c = (dot(d_h, h_acc) / (ld * la)).clamp(-1.0, 1.0);
                let angle = c.acos() * 180.0 / PI;
                let r0_hb = hbond_params(el[donor], el[acc])[0] * B2A;
                if h_acc_dist < 1.3 * r0_hb && angle > 120.0 && q[acc] < -0.05 {
                    out.push([donor, i, acc]);
                }
            }
        }
    }
    out
}

/// MultiOptPy: `GFNFFApproxHessian.build_topology`.
#[allow(clippy::needless_range_loop)]
fn topology(x: &[V3], el: &[Element]) -> Topology {
    let n = x.len();
    let bonded = connectivity(x, el);
    let q = charges(x, el, &bonded);
    let hbonds = hydrogen_bonds(x, el, &bonded, &q);
    let cn = coordination_numbers(x, el);
    let mut bonds = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if bonded[i][j] {
                bonds.push([i, j]);
            }
        }
    }
    let mut angles = Vec::new();
    for j in 0..n {
        let nb: Vec<usize> = (0..n).filter(|&i| bonded[i][j] && i != j).collect();
        for &i in &nb {
            for &k in &nb {
                if i < k {
                    angles.push([i, j, k]);
                }
            }
        }
    }
    let mut dihedrals = Vec::new();
    for &[j, k] in &bonds {
        let nj: Vec<usize> = (0..n)
            .filter(|&i| bonded[i][j] && i != j && i != k)
            .collect();
        let nk: Vec<usize> = (0..n)
            .filter(|&l| bonded[k][l] && l != k && l != j)
            .collect();
        for &i in &nj {
            for &l in &nk {
                if i != l {
                    dihedrals.push([i, j, k, l]);
                }
            }
        }
    }
    let mut graph = vec![Vec::new(); n];
    for &[i, j] in &bonds {
        graph[i].push(j);
        graph[j].push(i);
    }
    let mut nb_pairs = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if bonded[i][j] {
                continue;
            }
            let common: Vec<usize> = graph[i]
                .iter()
                .copied()
                .filter(|k| graph[j].contains(k))
                .collect();
            // MultiOptPy's 1-4 test: some neighbour l of j (l != k, l != i) is bonded to a
            // common neighbour k.
            let is_14 = common.iter().any(|&k| {
                graph[j]
                    .iter()
                    .any(|&l| l != k && graph[k].contains(&l) && l != i)
            });
            if common.is_empty() || !is_14 {
                nb_pairs.push([i, j]);
            }
        }
    }
    Topology {
        bonds,
        angles,
        dihedrals,
        nb_pairs,
        hbonds,
        cn,
    }
}

/// `+P` on the (i, i) and (j, j) blocks, `-P` on (i, j) and (j, i).
fn add_pair(h: &mut Mat<f64>, i: usize, j: usize, p: &[[f64; 3]; 3]) {
    for a in 0..3 {
        for b in 0..3 {
            h[(3 * i + a, 3 * i + b)] += p[a][b];
            h[(3 * j + a, 3 * j + b)] += p[a][b];
            h[(3 * i + a, 3 * j + b)] -= p[a][b];
            h[(3 * j + a, 3 * i + b)] -= p[a][b];
        }
    }
}

fn outer(u: V3, s: f64) -> [[f64; 3]; 3] {
    let mut p = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            p[a][b] = s * u[a] * u[b];
        }
    }
    p
}

/// MultiOptPy: `GFNFFApproxHessian.gfnff_bond_hessian`.
fn bond_terms(h: &mut Mat<f64>, x: &[V3], el: &[Element], top: &Topology) {
    let cn = &top.cn;
    for &[i, j] in &top.bonds {
        let r_vec = sub(x[j], x[i]);
        let r = norm(r_vec);
        let [r0, k] = bond_params(el[i], el[j]);
        let cn_factor = (-t::GFNFF_BOND_DECAY
            * ((cn[i] - ref_cn(el[i])).powi(2) + (cn[j] - ref_cn(el[j])).powi(2)))
        .exp();
        let fc = k * cn_factor * t::GFNFF_BOND_SCALING * (-2.0 * (r - r0).powi(2)).exp();
        let p = if r > 1e-10 {
            outer(scale(r_vec, 1.0 / r), fc)
        } else {
            let mut p = [[0.0; 3]; 3];
            (0..3).for_each(|a| p[a][a] = fc / 3.0);
            p
        };
        add_pair(h, i, j, &p);
    }
}

/// MultiOptPy: `GFNFFApproxHessian.gfnff_angle_hessian`.
fn angle_terms(h: &mut Mat<f64>, x: &[V3], el: &[Element], top: &Topology) {
    for &[i, j, k] in &top.angles {
        let r_ji = sub(x[i], x[j]);
        let r_jk = sub(x[k], x[j]);
        let (l_ji, l_jk) = (norm(r_ji), norm(r_jk));
        if l_ji < 1e-10 || l_jk < 1e-10 {
            continue;
        }
        let c = (dot(r_ji, r_jk) / (l_ji * l_jk)).clamp(-0.999999, 0.999999);
        let sin = c.acos().sin();
        let [_theta0, k_angle] = angle_params(el[i], el[j], el[k]);
        let fc = k_angle * (-0.1 * (top.cn[j] - ref_cn(el[j])).powi(2)).exp();
        let (u_ji, u_jk) = (scale(r_ji, 1.0 / l_ji), scale(r_jk, 1.0 / l_jk));
        let p_i = scale(sub(u_ji, scale(u_jk, c)), 1.0 / (l_ji * sin));
        let p_k = scale(sub(u_jk, scale(u_ji, c)), 1.0 / (l_jk * sin));
        let p_j = scale(add(p_i, p_k), -1.0);
        let atoms = [i, j, k];
        let d = [p_i, p_j, p_k];
        for m in 0..3 {
            for q in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        h[(3 * atoms[m] + a, 3 * atoms[q] + b)] += fc * d[m][a] * d[q][b];
                    }
                }
            }
        }
    }
}

/// MultiOptPy: `GFNFFApproxHessian.gfnff_torsion_hessian`.
fn torsion_terms(h: &mut Mat<f64>, x: &[V3], el: &[Element], top: &Topology) {
    let dim = h.nrows();
    for &[i, j, k, l] in &top.dihedrals {
        let r_ij = sub(x[j], x[i]);
        let r_jk = sub(x[k], x[j]);
        let r_kl = sub(x[l], x[k]);
        let n1 = cross(r_ij, r_jk);
        let n2 = cross(r_jk, r_kl);
        let (l1, l2, ljk) = (norm(n1), norm(n2), norm(r_jk));
        if l1 < 1e-10 || l2 < 1e-10 || ljk < 1e-10 {
            continue;
        }
        let cos_phi = (dot(n1, n2) / (l1 * l2)).clamp(-0.999999, 0.999999);
        let sin_phi = dot(cross(n1, n2), r_jk) / (l1 * l2 * ljk);
        let phi = sin_phi.atan2(cos_phi);
        let cn_factor = (-0.05
            * ((top.cn[j] - ref_cn(el[j])).powi(2) + (top.cn[k] - ref_cn(el[k])).powi(2)))
        .exp();
        let [v1, v2, v3] = torsion_params([el[i], el[j], el[k], el[l]]).map(|v| v * cn_factor);
        let force = -v1 * phi.sin() - 2.0 * v2 * (2.0 * phi).sin() - 3.0 * v3 * (3.0 * phi).sin();
        let k2 = v1 * phi.cos() + 4.0 * v2 * (2.0 * phi).cos() + 9.0 * v3 * (3.0 * phi).cos();
        let (lij, lkl) = (norm(r_ij), norm(r_kl));
        let e_ij = scale(r_ij, 1.0 / lij);
        let e_jk = scale(r_jk, 1.0 / ljk);
        let e_kl = scale(r_kl, 1.0 / lkl);
        let n1 = scale(n1, 1.0 / l1);
        let n2 = scale(n2, 1.0 / l2);
        let d_i = scale(cross(e_ij, n1), 1.0 / lij);
        let d_l = scale(cross(n2, e_kl), 1.0 / lkl);
        let d_j = sub(scale(d_i, -1.0), scale(cross(e_jk, n1), 1.0 / lij));
        let d_k = sub(scale(d_l, -1.0), scale(cross(n2, e_jk), 1.0 / lkl));
        let d = [d_i, d_j, d_k, d_l].map(|v| scale(v, force));
        let atoms = [i, j, k, l];
        for m in 0..4 {
            for q in m..4 {
                for a in 0..3 {
                    for b in 0..3 {
                        h[(3 * atoms[m] + a, 3 * atoms[q] + b)] += k2 * d[m][a] * d[q][b];
                    }
                }
            }
        }
        // MultiOptPy copies the upper triangle to the lower one after every torsion.
        for m in 0..dim {
            for q in 0..m {
                h[(m, q)] = h[(q, m)];
            }
        }
    }
}

/// MultiOptPy: `GFNFFApproxHessian.gfnff_hbond_hessian`.
fn hbond_terms(h: &mut Mat<f64>, x: &[V3], el: &[Element], top: &Topology) {
    for &[d, hy, a] in &top.hbonds {
        let [r0, k] = hbond_params(el[d], el[a]);
        let r_dh = sub(x[hy], x[d]);
        let r_ha = sub(x[a], x[hy]);
        let (l_dh, l_ha) = (norm(r_dh), norm(r_ha));
        if l_dh < 1e-10 || l_ha < 1e-10 {
            continue;
        }
        let c = (dot(r_dh, r_ha) / (l_dh * l_ha)).clamp(-0.999999, 0.999999);
        let angle = c.acos();
        let dist_factor = (-(l_ha - r0).powi(2) / (2.0 * 0.3_f64.powi(2))).exp();
        let angle_factor = (1.0 + (angle - PI).cos()).powi(2) / 4.0;
        let fc = k * dist_factor * angle_factor;
        add_pair(h, hy, a, &outer(scale(r_ha, 1.0 / l_ha), fc));
    }
}

/// MultiOptPy: `GFNFFApproxHessian.gfnff_nonbonded_hessian`.
fn nonbonded_terms(h: &mut Mat<f64>, x: &[V3], el: &[Element], top: &Topology) {
    for &[i, j] in &top.nb_pairs {
        let r_vec = sub(x[i], x[j]);
        let r = norm(r_vec);
        if r < 0.1 {
            continue;
        }
        let (ai, aj) = (alpha(el[i]), alpha(el[j]));
        let c6 = 2.0 * ai * aj / (ai / aj + aj / ai) * 0.05;
        let vdw_sum = vdw(el[i]) / B2A + vdw(el[j]) / B2A;
        let rep_const = 0.3;
        let rep_deriv =
            -12.0 * rep_const * (-(r / vdw_sum - 0.6) * 12.0).exp() * (1.0 / vdw_sum) / r;
        let r0 = 0.5 * vdw_sum;
        let (a1, a2) = (0.4, 3.0);
        let rr6 = (a1 * r0 + a2).powi(6);
        let damp = r.powi(6) / (r.powi(6) + rr6);
        let damp_deriv = 6.0 * r.powi(5) * rr6 / (r.powi(6) + rr6).powi(2);
        let disp_deriv = t::GFNFF_D4_S6 * c6 * (6.0 * damp / r.powi(7) - damp_deriv / r.powi(6));
        let factor = (rep_deriv + disp_deriv) / r + 0.2;
        add_pair(h, i, j, &outer(scale(r_vec, 1.0 / r), factor));
    }
}

/// MultiOptPy: `GFNFFApproxHessian.main`.
pub(crate) fn gfnff(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let el = &sys.elements;
    let dim = 3 * x.len();
    let mut h = Mat::<f64>::zeros(dim, dim);
    let top = topology(&x, el);
    bond_terms(&mut h, &x, el, &top);
    angle_terms(&mut h, &x, el, &top);
    torsion_terms(&mut h, &x, el, &top);
    hbond_terms(&mut h, &x, el, &top);
    nonbonded_terms(&mut h, &x, el, &top);
    for i in 0..dim {
        for j in 0..i {
            h[(j, i)] = h[(i, j)];
        }
    }
    project_tr_rot(&h, &x)
}
