//! MultiOptPy's GFN0-xTB-like model Hessian, reproduced as it is.
//!
//! MultiOptPy: `ModelHessian/gfn0xtb.py` (`GFN0XTBApproxHessian`) with the parameters of
//! `Parameters/gfn0xtb_param.py` (`GFN0Parameters`), generated into
//! [`crate::parameters::multioptpy_gfn`].
//!
//! Provenance: this is MultiOptPy's own approximate model, inspired by GFN0-xTB (P. Pracht,
//! E. Caldeweyher, S. Ehlert, S. Grimme, ChemRxiv 2019, DOI 10.26434/chemrxiv.8326202.v1). It
//! is not a tight-binding calculation and does not use the GFN0-xTB Hamiltonian: hybridization
//! is guessed from neighbour counts and bond lengths, and the bond, angle, torsion, and
//! non-bonded force constants are MultiOptPy's estimates (MultiOptPy labels the radii and
//! electronegativities as taken from `gfn0_param.f90`; they are reproduced as MultiOptPy has
//! them). mopt-rs offers it for the elements of MultiOptPy's radii and electronegativities, H
//! to Xe ([`super::GFN0_LIKE_RANGE`]).
//!
//! Reproduced as in MultiOptPy, including:
//! - bonds from the `rad` table read as Bohr (`distance < 1.3 (r_i + r_j)`), which detects few
//!   bonds between heavy atoms at usual distances;
//! - reference bond lengths looked up with sorted symbols (entries such as `("O", "H")` are
//!   never used) and only for the hybridization tests;
//! - torsion derivative vectors that are zero unless `sin(phi) > 1e-10`;
//! - the final matrix copies the lower triangle to the upper one.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;
use std::f64::consts::PI;

use faer::Mat;

use super::common::{coords, cross, dot, norm, project_tr_rot, sub, V3};
use crate::domain::{Element, MolecularSystem};
use crate::parameters::multioptpy_gfn as t;
use crate::parameters::multioptpy_tables::MOP_BOHR2ANGSTROMS as B2A;

const BOND_FACTOR: f64 = 1.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hyb {
    Sp,
    Sp2,
    Sp3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BondType {
    Single,
    Double,
    Triple,
}

impl BondType {
    fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Double => "double",
            Self::Triple => "triple",
        }
    }
}

fn scale(v: V3, s: f64) -> V3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn table(values: &[f64], el: Element, default: f64) -> f64 {
    values
        .get(el.atomic_number() as usize)
        .copied()
        .filter(|v| !v.is_nan())
        .unwrap_or(default)
}

/// MultiOptPy: `GFN0Parameters.get_radius` (Bohr).
fn radius(el: Element) -> f64 {
    table(&t::GFN0_RADIUS_BOHR, el, 1.0)
}

/// MultiOptPy: `GFN0Parameters.get_en`.
fn en(el: Element) -> f64 {
    table(&t::GFN0_EN, el, 2.0)
}

/// MultiOptPy: `GFN0Parameters.get_bond_length` (Angstrom).
fn bond_length(a: Element, b: Element, kind: BondType) -> f64 {
    let (s, u) = (a.symbol(), b.symbol());
    let (s, u) = if s <= u { (s, u) } else { (u, s) };
    let tagged = match kind {
        BondType::Single => None,
        other => t::gfn0_reference_bond_length(s, u, Some(other.name())),
    };
    tagged
        .or_else(|| t::gfn0_reference_bond_length(s, u, None))
        .unwrap_or_else(|| {
            let base = radius(a) + radius(b);
            match kind {
                BondType::Double => base * 0.85,
                BondType::Triple => base * 0.78,
                BondType::Single => base,
            }
        })
}

/// MultiOptPy: `GFN0Parameters.get_bond_force_constant`.
fn bond_force_constant(a: Element, b: Element, kind: BondType) -> f64 {
    let cn_pair = matches!((a.symbol(), b.symbol()), ("C", "N") | ("N", "C"));
    if cn_pair && kind == BondType::Triple {
        return t::gfn0_cnparams("kStretch").unwrap_or(0.9);
    }
    t::GFN0_KSTRETCHBASE * t::gfn0_bondforcefactors(kind.name()).unwrap_or(1.0)
}

struct Topology {
    bonded: Vec<Vec<bool>>,
    bonds: Vec<[usize; 2]>,
    bond_types: HashMap<(usize, usize), BondType>,
    angles: Vec<[usize; 3]>,
    dihedrals: Vec<[usize; 4]>,
    cyano: Vec<(usize, usize)>,
}

impl Topology {
    fn bond_type(&self, i: usize, j: usize) -> BondType {
        self.bond_types
            .get(&(i, j))
            .copied()
            .unwrap_or(BondType::Single)
    }
}

fn angle_deg(a: V3, b: V3) -> f64 {
    let c = (dot(a, b) / (norm(a) * norm(b))).clamp(-1.0, 1.0);
    c.acos() * 180.0 / PI
}

/// MultiOptPy: `analyze_molecular_structure` (hybridization of one atom).
fn hybridization(i: usize, x: &[V3], el: &[Element], bonded: &[Vec<bool>]) -> Hyb {
    let n = x.len();
    let neighbors: Vec<usize> = (0..n).filter(|&j| bonded[i][j]).collect();
    let dist = |j: usize| norm(sub(x[i], x[j]));
    match (el[i].symbol(), neighbors.len()) {
        ("C", 4) => Hyb::Sp3,
        ("C", 3) => Hyb::Sp2,
        ("C", 2) => {
            let angle = angle_deg(sub(x[neighbors[0]], x[i]), sub(x[neighbors[1]], x[i]));
            if angle > 160.0 {
                Hyb::Sp
            } else {
                Hyb::Sp2
            }
        }
        ("N", 3) | ("N", 2) => Hyb::Sp2,
        ("N", 1) => {
            let j = neighbors[0];
            if el[j].symbol() == "C" {
                let reference = bond_length(el[j], el[i], BondType::Triple) / B2A;
                if (dist(j) - reference).abs() < 0.15 {
                    Hyb::Sp
                } else {
                    Hyb::Sp2
                }
            } else {
                Hyb::Sp2
            }
        }
        ("O", 1) => {
            let j = neighbors[0];
            if el[j].symbol() == "C" {
                let reference = bond_length(el[j], el[i], BondType::Double) / B2A;
                if (dist(j) - reference).abs() < 0.15 {
                    Hyb::Sp2
                } else {
                    Hyb::Sp3
                }
            } else {
                Hyb::Sp3
            }
        }
        _ => Hyb::Sp3,
    }
}

/// MultiOptPy: `detect_bond_connectivity` and `analyze_molecular_structure`.
// Index loops as in MultiOptPy, for side-by-side reading.
#[allow(clippy::needless_range_loop)]
fn topology(x: &[V3], el: &[Element]) -> Topology {
    let n = x.len();
    let mut bonded = vec![vec![false; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let b = norm(sub(x[i], x[j])) < (radius(el[i]) + radius(el[j])) * BOND_FACTOR;
            bonded[i][j] = b;
            bonded[j][i] = b;
        }
    }
    let mut bonds = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if bonded[i][j] {
                bonds.push([i, j]);
            }
        }
    }
    let nc: Vec<usize> = bonded
        .iter()
        .map(|row| row.iter().filter(|b| **b).count())
        .collect();
    let hyb: Vec<Hyb> = (0..n).map(|i| hybridization(i, x, el, &bonded)).collect();
    let mut bond_types = HashMap::new();
    for &[i, j] in &bonds {
        let (hi, hj) = (hyb[i], hyb[j]);
        let kind = match (el[i].symbol(), el[j].symbol()) {
            ("C", "N") | ("N", "C") => {
                let triple = (hi == Hyb::Sp && hj == Hyb::Sp)
                    || (hi == Hyb::Sp && nc[i] == 2 && nc[j] == 1)
                    || (hj == Hyb::Sp && nc[j] == 2 && nc[i] == 1);
                if triple {
                    BondType::Triple
                } else {
                    BondType::Single
                }
            }
            ("C", "C") => match (hi, hj) {
                (Hyb::Sp, Hyb::Sp) => BondType::Triple,
                (Hyb::Sp2, Hyb::Sp2) => BondType::Double,
                _ => BondType::Single,
            },
            ("C", "O") | ("O", "C") => {
                let double = (hi == Hyb::Sp2 && hj == Hyb::Sp2)
                    || (hi == Hyb::Sp2 && nc[j] == 1)
                    || (hj == Hyb::Sp2 && nc[i] == 1);
                if double {
                    BondType::Double
                } else {
                    BondType::Single
                }
            }
            _ => BondType::Single,
        };
        bond_types.insert((i, j), kind);
        bond_types.insert((j, i), kind);
    }
    let mut cyano = Vec::new();
    for i in 0..n {
        if el[i].symbol() == "C" && hyb[i] == Hyb::Sp {
            let n_partners: Vec<usize> = (0..n)
                .filter(|&j| bonded[i][j] && el[j].symbol() == "N")
                .collect();
            let others = (0..n)
                .filter(|&j| bonded[i][j] && el[j].symbol() != "N")
                .count();
            if let Some(&n_idx) = n_partners.first() {
                if others == 1 && bond_types.get(&(i, n_idx)) == Some(&BondType::Triple) {
                    cyano.push((i, n_idx));
                }
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
    Topology {
        bonded,
        bonds,
        bond_types,
        angles,
        dihedrals,
        cyano,
    }
}

/// MultiOptPy: `compute_partial_charges`.
fn charges(el: &[Element], top: &Topology) -> Vec<f64> {
    let n = el.len();
    let mut q = vec![0.0; n];
    for &[i, j] in &top.bonds {
        let transfer = 0.05 * (0.2 * (en(el[j]) - en(el[i]))).tanh();
        q[i] += transfer;
        q[j] -= transfer;
    }
    for &(c, nn) in &top.cyano {
        q[nn] -= 0.3;
        q[c] += 0.3;
    }
    let mean = q.iter().sum::<f64>() / n as f64;
    q.iter_mut().for_each(|v| *v -= mean);
    q
}

fn add_pair(h: &mut Mat<f64>, i: usize, j: usize, u: V3, s: f64) {
    for a in 0..3 {
        for b in 0..3 {
            let p = s * u[a] * u[b];
            h[(3 * i + a, 3 * i + b)] += p;
            h[(3 * j + a, 3 * j + b)] += p;
            h[(3 * i + a, 3 * j + b)] -= p;
            h[(3 * j + a, 3 * i + b)] -= p;
        }
    }
}

fn add_all(h: &mut Mat<f64>, atoms: &[usize], d: &[V3]) {
    for (m, &am) in atoms.iter().enumerate() {
        for (q, &aq) in atoms.iter().enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    h[(3 * am + a, 3 * aq + b)] += d[m][a] * d[q][b];
                }
            }
        }
    }
}

/// MultiOptPy: `gfn0_bond_hessian`.
fn bond_terms(h: &mut Mat<f64>, x: &[V3], el: &[Element], top: &Topology) {
    for &[i, j] in &top.bonds {
        let r_vec = sub(x[j], x[i]);
        let r = norm(r_vec);
        let fc = bond_force_constant(el[i], el[j], top.bond_type(i, j));
        if r > 1e-10 {
            add_pair(h, i, j, scale(r_vec, 1.0 / r), fc);
        } else {
            for a in 0..3 {
                h[(3 * i + a, 3 * i + a)] += fc / 3.0;
                h[(3 * j + a, 3 * j + a)] += fc / 3.0;
                h[(3 * i + a, 3 * j + a)] -= fc / 3.0;
                h[(3 * j + a, 3 * i + a)] -= fc / 3.0;
            }
        }
    }
}

/// MultiOptPy: `gfn0_angle_hessian`.
fn angle_terms(h: &mut Mat<f64>, x: &[V3], el: &[Element], top: &Topology) {
    for &[i, j, k] in &top.angles {
        let r_ji = sub(x[i], x[j]);
        let r_jk = sub(x[k], x[j]);
        let (l_ji, l_jk) = (norm(r_ji), norm(r_jk));
        if l_ji < 1e-10 || l_jk < 1e-10 {
            continue;
        }
        let cyano_n = top.cyano.iter().find(|(c, _)| *c == j).map(|&(_, nn)| nn);
        let fc = match cyano_n {
            Some(nn) if i == nn || k == nn => t::gfn0_cnparams("kBend").unwrap_or(0.15),
            _ => {
                let factor = match el[j].symbol() {
                    "C" => 1.0,
                    "N" => 0.9,
                    "O" => 0.8,
                    _ => 0.7,
                };
                // MultiOptPy's theta0 (the natural angle of the hybridization) does not enter
                // the Hessian.
                t::GFN0_KANGLEBASE * factor
            }
        };
        let c = (dot(r_ji, r_jk) / (l_ji * l_jk)).clamp(-0.999999, 0.999999);
        let sin = c.acos().sin();
        if sin < 1e-10 {
            continue;
        }
        let d_i = scale(
            cross(cross(r_ji, r_jk), r_ji),
            1.0 / (l_ji * l_ji * l_jk * sin),
        );
        let d_k = scale(
            cross(cross(r_jk, r_ji), r_jk),
            1.0 / (l_ji * l_jk * l_jk * sin),
        );
        let d_j = scale([d_i[0] + d_k[0], d_i[1] + d_k[1], d_i[2] + d_k[2]], -1.0);
        let s = fc.sqrt();
        add_all(
            h,
            &[i, j, k],
            &[scale(d_i, s), scale(d_j, s), scale(d_k, s)],
        );
    }
}

/// MultiOptPy: `gfn0_torsion_hessian`.
fn torsion_terms(h: &mut Mat<f64>, x: &[V3], top: &Topology) {
    let v2_base = t::GFN0_V2BASE;
    let v3_base = t::GFN0_V3BASE;
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
        let cyano_bond = top
            .cyano
            .iter()
            .any(|&(c, nn)| (c, nn) == (j, k) || (nn, c) == (j, k));
        let (v2, v3) = if cyano_bond {
            let v = t::gfn0_cnparams("kTorsion").unwrap_or(0.002);
            (v, v)
        } else {
            match top.bond_type(j, k) {
                BondType::Triple => (0.001, 0.001),
                BondType::Double => (v2_base * 2.0, v3_base * 0.5),
                BondType::Single => (v2_base * 0.5, v3_base * 1.5),
            }
        };
        let d2v = 2.0 * v2 * (2.0 * phi).cos() - 4.5 * v3 * (3.0 * phi).cos();
        let (lij, lkl) = (norm(r_ij), norm(r_kl));
        let e_ij = if lij > 1e-10 {
            scale(r_ij, 1.0 / lij)
        } else {
            [0.0; 3]
        };
        let e_kl = if lkl > 1e-10 {
            scale(r_kl, 1.0 / lkl)
        } else {
            [0.0; 3]
        };
        let n1u = scale(n1, 1.0 / l1);
        let n2u = scale(n2, 1.0 / l2);
        let (g_i, g_l) = if sin_phi > 1e-10 {
            (
                scale(cross(e_ij, n1u), 1.0 / (lij * sin_phi)),
                scale(cross(e_kl, n2u), -1.0 / (lkl * sin_phi)),
            )
        } else {
            ([0.0; 3], [0.0; 3])
        };
        let g_j = scale(g_i, -1.0 - ljk / lij);
        let g_k = scale(g_l, -1.0 - ljk / lkl);
        let s = d2v.abs().sqrt();
        add_all(h, &[i, j, k, l], &[g_i, g_j, g_k, g_l].map(|g| scale(g, s)));
    }
}

/// MultiOptPy: `gfn0_nonbonded_hessian`.
fn nonbonded_terms(h: &mut Mat<f64>, x: &[V3], el: &[Element], top: &Topology) {
    let n = x.len();
    let q = charges(el, top);
    let bonded = &top.bonded;
    for i in 0..n {
        for j in i + 1..n {
            if bonded[i][j] || (0..n).any(|k| bonded[i][k] && bonded[j][k]) {
                continue;
            }
            let r_vec = sub(x[j], x[i]);
            let r = norm(r_vec);
            if r < 0.5 {
                continue;
            }
            let rep = 0.05 * ((radius(el[i]) + radius(el[j])) / r).powi(12);
            let elec = 0.1 * q[i] * q[j] / r;
            let factor = 12.0 * rep / (r * r) + 2.0 * elec / (r * r);
            add_pair(h, i, j, scale(r_vec, 1.0 / r), factor);
        }
    }
}

/// MultiOptPy: `GFN0XTBApproxHessian.main`.
pub(crate) fn gfn0xtb(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let el = &sys.elements;
    let dim = 3 * x.len();
    let mut h = Mat::<f64>::zeros(dim, dim);
    let top = topology(&x, el);
    bond_terms(&mut h, &x, el, &top);
    angle_terms(&mut h, &x, el, &top);
    torsion_terms(&mut h, &x, &top);
    nonbonded_terms(&mut h, &x, el, &top);
    for i in 0..dim {
        for j in 0..i {
            h[(j, i)] = h[(i, j)];
        }
    }
    project_tr_rot(&h, &x)
}
