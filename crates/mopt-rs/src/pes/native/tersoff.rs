//! Tersoff bond-order potential in the LAMMPS/ASE parameter convention, with exact gradient and
//! Hessian.
//!
//! `E = 1/2 sum_i sum_{j != i} f_c(r_ij) [A e^{-lambda1 r_ij} - b_ij B e^{-lambda2 r_ij}]`,
//! `b_ij = (1 + (beta zeta_ij)^n)^(-1/(2n))`,
//! `zeta_ij = sum_{k != i, j} f_c(r_ik) g(theta_ijk) exp[(lambda3 (r_ij - r_ik))^m]`,
//! `g = gamma (1 + c^2/d^2 - c^2 / (d^2 + (h - cos theta)^2))`.
//!
//! Parameters are keyed by element triples `(i, j, k)`: the pair entry `(i, j, j)` supplies
//! `A, B, lambda1, lambda2, beta, n, R, D` for `r_ij`; the triple entry `(i, j, k)` supplies
//! `m, gamma, lambda3, c, d, h, R, D` for `r_ik`. Built-in parameters are Tersoff's 1989 sets for
//! C, Si, and Ge with his mixing rules (Phys. Rev. B 39, 5566). Every bond term is
//! differentiated over the atoms it involves (forward-mode AD), and periodic images enter as
//! constant shifts.
//!
//! MultiOptPy's Tersoff energy and gradient are inconsistent; see `docs/source/migration_surfaces.md`.
//!
//! MultiOptPy: `Calculator/tersoff_calculation_tools.py:TersoffCore`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::neighbors::{Neighbor, VerletList};
use crate::domain::{
    AtomId, Bohr, Element, EvalOutput, EvalRequest, MolecularSystem, Quantity, StrainHessian, Units,
};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_and_gradient, eval_value_grad_hessian, Scalar};
use crate::pes::calculator::{Calculator, SiteTerms};

/// Parameters of one element triple (LAMMPS order; lengths in Angstrom, energies in eV).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TersoffParams {
    /// Exponent `m` of the `lambda3` term.
    pub m: f64,
    /// Angular prefactor `gamma`.
    pub gamma: f64,
    /// `lambda3` (1/Angstrom).
    pub lambda3: f64,
    /// Angular `c`.
    pub c: f64,
    /// Angular `d`.
    pub d: f64,
    /// Angular `h` (`cos theta_0`).
    pub h: f64,
    /// Bond-order exponent `n`.
    pub n: f64,
    /// Bond-order `beta`.
    pub beta: f64,
    /// Attractive decay `lambda2` (1/Angstrom).
    pub lambda2: f64,
    /// Attractive prefactor `B` (eV).
    pub b: f64,
    /// Cutoff center `R` (Angstrom).
    pub r: f64,
    /// Cutoff half-width `D` (Angstrom).
    pub d_cut: f64,
    /// Repulsive decay `lambda1` (1/Angstrom).
    pub lambda1: f64,
    /// Repulsive prefactor `A` (eV).
    pub a: f64,
}

/// Tersoff (1989) single-element parameters: `A, B, lambda, mu, beta, n, c, d, h, R, S`.
const TERSOFF_1989: [(&str, [f64; 11]); 3] = [
    (
        "C",
        [
            1393.6, 346.74, 3.4879, 2.2119, 1.5724e-7, 0.72751, 3.8049e4, 4.3484, -0.57058, 1.8,
            2.1,
        ],
    ),
    (
        "Si",
        [
            1830.8, 471.18, 2.4799, 1.7322, 1.1e-6, 0.78734, 1.0039e5, 16.217, -0.59825, 2.7, 3.0,
        ],
    ),
    (
        "Ge",
        [
            1769.0, 419.23, 2.4451, 1.7047, 9.0166e-7, 0.75627, 1.0643e5, 15.652, -0.43884, 2.8,
            3.1,
        ],
    ),
];

/// Tersoff (1989) `chi` for hetero pairs.
const CHI_1989: [(&str, &str, f64); 2] = [("Si", "C", 0.9776), ("Si", "Ge", 1.00061)];

/// Tersoff calculator.
#[derive(Debug, Clone)]
pub struct Tersoff {
    table: Vec<([Element; 3], TersoffParams)>,
    /// The pairs kept between evaluations.
    verlet: VerletList,
}

fn el(sym: &str) -> MoptResult<Element> {
    Element::parse_token(sym)
}

impl Tersoff {
    /// Parameters for arbitrary element triples (every triple of the elements used must be
    /// present).
    pub fn from_table(table: Vec<([Element; 3], TersoffParams)>) -> Self {
        Self {
            table,
            verlet: VerletList::default(),
        }
    }

    /// Tersoff's 1989 parameters and mixing rules for the given elements (C, Si, Ge; hetero
    /// pairs Si-C and Si-Ge).
    pub fn tersoff_1989(elements: &[Element]) -> MoptResult<Self> {
        let single = |e: Element| -> MoptResult<[f64; 11]> {
            TERSOFF_1989
                .iter()
                .find(|(s, _)| el(s).is_ok_and(|x| x == e))
                .map(|(_, p)| *p)
                .ok_or_else(|| MoptError::config(format!("no Tersoff (1989) parameters for {e}")))
        };
        let chi = |a: Element, b: Element| -> MoptResult<f64> {
            if a == b {
                return Ok(1.0);
            }
            CHI_1989
                .iter()
                .find(|(x, y, _)| {
                    let (x, y) = (el(x).ok(), el(y).ok());
                    (x == Some(a) && y == Some(b)) || (x == Some(b) && y == Some(a))
                })
                .map(|(_, _, c)| *c)
                .ok_or_else(|| {
                    MoptError::config(format!("no Tersoff (1989) mixing parameter for {a}-{b}"))
                })
        };
        let mut uniq: Vec<Element> = elements.to_vec();
        uniq.sort();
        uniq.dedup();
        let mut table = Vec::new();
        for &i in &uniq {
            for &j in &uniq {
                for &k in &uniq {
                    let (pi, pj, pk) = (single(i)?, single(j)?, single(k)?);
                    let r_ik = (pi[9] * pk[9]).sqrt();
                    let s_ik = (pi[10] * pk[10]).sqrt();
                    table.push((
                        [i, j, k],
                        TersoffParams {
                            m: 3.0,
                            gamma: 1.0,
                            lambda3: 0.0,
                            c: pi[6],
                            d: pi[7],
                            h: pi[8],
                            n: pi[5],
                            beta: pi[4],
                            lambda2: 0.5 * (pi[3] + pj[3]),
                            b: chi(i, j)? * (pi[1] * pj[1]).sqrt(),
                            r: 0.5 * (r_ik + s_ik),
                            d_cut: 0.5 * (s_ik - r_ik),
                            lambda1: 0.5 * (pi[2] + pj[2]),
                            a: (pi[0] * pj[0]).sqrt(),
                        },
                    ));
                }
            }
        }
        Ok(Self {
            table,
            verlet: VerletList::default(),
        })
    }

    /// Parse a LAMMPS `.tersoff` file (17 fields per entry: 3 elements and 14 parameters).
    pub fn from_lammps_str(text: &str) -> MoptResult<Self> {
        let tokens: Vec<&str> = text
            .lines()
            .map(|l| l.split('#').next().unwrap_or(""))
            .flat_map(str::split_whitespace)
            .collect();
        if tokens.is_empty() || !tokens.len().is_multiple_of(17) {
            return Err(MoptError::parse(
                "a LAMMPS Tersoff file has 17 fields per entry",
            ));
        }
        let mut table = Vec::new();
        for e in tokens.chunks(17) {
            let elems = [el(e[0])?, el(e[1])?, el(e[2])?];
            let v: Vec<f64> = e[3..]
                .iter()
                .map(|t| {
                    t.parse::<f64>()
                        .map_err(|err| MoptError::parse(format!("Tersoff parameter '{t}': {err}")))
                })
                .collect::<MoptResult<_>>()?;
            table.push((
                elems,
                TersoffParams {
                    m: v[0],
                    gamma: v[1],
                    lambda3: v[2],
                    c: v[3],
                    d: v[4],
                    h: v[5],
                    n: v[6],
                    beta: v[7],
                    lambda2: v[8],
                    b: v[9],
                    r: v[10],
                    d_cut: v[11],
                    lambda1: v[12],
                    a: v[13],
                },
            ));
        }
        Ok(Self {
            table,
            verlet: VerletList::default(),
        })
    }

    fn params(&self, i: Element, j: Element, k: Element) -> MoptResult<&TersoffParams> {
        self.table
            .iter()
            .find(|(t, _)| *t == [i, j, k])
            .map(|(_, p)| p)
            .ok_or_else(|| {
                MoptError::config(format!("no Tersoff parameters for the triple {i}-{j}-{k}"))
            })
    }

    fn cutoff(&self) -> f64 {
        self.table
            .iter()
            .map(|(_, p)| p.r + p.d_cut)
            .fold(0.0, f64::max)
    }
}

fn fc<T: Scalar>(r: T, p_r: f64, p_d: f64) -> T {
    let x = r.val();
    if x > p_r + p_d {
        T::zero()
    } else if x < p_r - p_d {
        T::one()
    } else {
        T::from_f64(0.5)
            * (T::one()
                - (T::from_f64(std::f64::consts::FRAC_PI_2 / p_d) * (r - T::from_f64(p_r))).sin())
    }
}

/// A neighbor reference inside a local term: local variable slot and constant image shift
/// (Angstrom) such that the image position is `x[slot] + shift`.
#[derive(Debug, Clone, Copy)]
struct Site {
    slot: usize,
    shift: [f64; 3],
}

/// One bond term `(i, j)` with its `k` neighbors.
struct BondTerm<'a> {
    pair: &'a TersoffParams,
    j: Site,
    ks: Vec<(Site, &'a TersoffParams)>,
    /// Where the nine components of a homogeneous strain follow the coordinates in the
    /// variables (`None`: no strain), for the stress.
    strain: Option<usize>,
}

/// Position of a site; with a strain `eps` (variables from `strain` on, row by row) it is
/// `(1 + eps)` times the position.
fn site_pos<T: Scalar>(x: &[T], s: Site, strain: Option<usize>) -> [T; 3] {
    let p: [T; 3] = std::array::from_fn(|c| x[3 * s.slot + c] + T::from_f64(s.shift[c]));
    match strain {
        None => p,
        Some(at) => std::array::from_fn(|a| {
            p[a] + x[at + 3 * a] * p[0] + x[at + 3 * a + 1] * p[1] + x[at + 3 * a + 2] * p[2]
        }),
    }
}

fn sub<T: Scalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

impl BondTerm<'_> {
    /// `1/2 f_c(r_ij) [A e^{-l1 r} - b_ij B e^{-l2 r}]` with slot 0 = atom i.
    fn energy<T: Scalar>(&self, x: &[T]) -> T {
        let ri = site_pos(
            x,
            Site {
                slot: 0,
                shift: [0.0; 3],
            },
            self.strain,
        );
        let rij_v = sub(site_pos(x, self.j, self.strain), ri);
        let rij = T::norm3(rij_v);
        let p = self.pair;
        let mut zeta = T::zero();
        for (k, pk) in &self.ks {
            let rik_v = sub(site_pos(x, *k, self.strain), ri);
            let rik = T::norm3(rik_v);
            if rik.val() > pk.r + pk.d_cut {
                continue;
            }
            let cos = T::dot3(rij_v, rik_v) / (rij * rik);
            let (c2, d2) = (pk.c * pk.c, pk.d * pk.d);
            let hc = T::from_f64(pk.h) - cos;
            let g = T::from_f64(pk.gamma)
                * (T::from_f64(1.0 + c2 / d2) - T::from_f64(c2) / (T::from_f64(d2) + hc * hc));
            let arg = (T::from_f64(pk.lambda3) * (rij - rik)).powi(pk.m as i32);
            zeta += fc(rik, pk.r, pk.d_cut) * g * arg.exp();
        }
        let bij = if zeta.val() > 0.0 {
            (T::one() + (zeta * T::from_f64(p.beta)).powf(p.n)).powf(-1.0 / (2.0 * p.n))
        } else {
            T::one()
        };
        let rep = T::from_f64(p.a) * (rij * T::from_f64(-p.lambda1)).exp();
        let att = T::from_f64(-p.b) * (rij * T::from_f64(-p.lambda2)).exp();
        T::from_f64(0.5) * fc(rij, p.r, p.d_cut) * (rep + bij * att)
    }
}

impl Tersoff {
    /// The bond terms of atom `i` with its neighbours `nbs` (at the variables `sites`).
    fn bond_terms<'a>(
        &'a self,
        sys: &MolecularSystem,
        i: AtomId,
        nbs: &[Neighbor],
        sites: &[Site],
        strain: Option<usize>,
    ) -> MoptResult<Vec<BondTerm<'a>>> {
        let ei = sys.element(i);
        let mut terms = Vec::with_capacity(nbs.len());
        for (jn, nb_j) in nbs.iter().enumerate() {
            let ej = sys.element(nb_j.atom);
            let pair = self.params(ei, ej, ej)?;
            if nb_j.distance * Units::BOHR_TO_ANGSTROM > pair.r + pair.d_cut {
                continue;
            }
            let ks = nbs
                .iter()
                .enumerate()
                .filter(|(kn, _)| *kn != jn)
                .map(|(kn, nb_k)| Ok((sites[kn], self.params(ei, ej, sys.element(nb_k.atom))?)))
                .collect::<MoptResult<Vec<_>>>()?;
            terms.push(BondTerm {
                pair,
                j: sites[jn],
                ks,
                strain,
            });
        }
        Ok(terms)
    }

    /// The energy of every atom (Hartree), `U_i` the sum of the bond terms of atom `i`, and
    /// the virials `W_k = sum_i (r_i - r_k) (x) dU_i/dr_k` (Hartree) over the neighbours `k`
    /// of every atom `i` and their periodic images (own images included), for the heat
    /// current (Fan et al., Phys. Rev. B 2015, 92, 094301). Every neighbour, image or not,
    /// is a variable of its own, so that each image gets its own derivative.
    fn site_terms_of(&self, sys: &MolecularSystem) -> MoptResult<SiteTerms> {
        let to_ang = Units::BOHR_TO_ANGSTROM;
        let e = Units::EV_TO_HARTREE;
        let nl = self
            .verlet
            .neighbors(sys, Quantity::<Bohr>::new(self.cutoff() / to_ang));
        let mut energies = vec![0.0; sys.n_atoms()];
        let mut virials = vec![[[0.0; 3]; 3]; sys.n_atoms()];
        for i in sys.atoms() {
            let nbs = &nl[i.index()];
            let pi = sys.position(i).value();
            let mut x: Vec<f64> = pi.iter().map(|c| c * to_ang).collect();
            for nb in nbs {
                x.extend((0..3).map(|c| (pi[c] + nb.vector[c]) * to_ang));
            }
            let sites: Vec<Site> = (0..nbs.len())
                .map(|k| Site {
                    slot: k + 1,
                    shift: [0.0; 3],
                })
                .collect();
            let mut u = 0.0;
            let mut g = vec![0.0; x.len()];
            for term in self.bond_terms(sys, i, nbs, &sites, None)? {
                let (value, g_loc) = eval_value_and_gradient(&x, |v| term.energy(v));
                u += value;
                for (s, gs) in g.iter_mut().enumerate() {
                    *gs += g_loc[s];
                }
            }
            energies[i.index()] = u * e;
            for (k, nb) in nbs.iter().enumerate() {
                let w = &mut virials[nb.atom.index()];
                for (a, row) in w.iter_mut().enumerate() {
                    for (b, value) in row.iter_mut().enumerate() {
                        // r_i - r_k = -vector (Angstrom) and dU_i/dr_k (eV/Angstrom).
                        *value -= nb.vector[a] * to_ang * g[3 * (k + 1) + b] * e;
                    }
                }
            }
        }
        Ok(SiteTerms { energies, virials })
    }

    /// Energy (eV), gradient (eV/Angstrom), Hessian, `dE / d(strain)` (eV) in a cell, and
    /// with `need_strain` (a Hessian in a cell) the second derivatives along the strain (eV,
    /// and eV/Angstrom for the mixed ones).
    #[allow(clippy::type_complexity)]
    fn assemble(
        &self,
        sys: &MolecularSystem,
        need_hessian: bool,
        need_strain: bool,
    ) -> MoptResult<(
        f64,
        Col<f64>,
        Option<Mat<f64>>,
        Option<[[f64; 3]; 3]>,
        Option<StrainHessian>,
    )> {
        let mut virial = sys.active_cell().map(|_| [[0.0; 3]; 3]);
        let mut strain_hessian = (need_hessian && need_strain && virial.is_some())
            .then(|| StrainHessian::zeros(sys.dof()));
        let to_ang = Units::BOHR_TO_ANGSTROM;
        let nl = self
            .verlet
            .neighbors(sys, Quantity::<Bohr>::new(self.cutoff() / to_ang));
        let n = sys.dof();
        let mut energy = 0.0;
        let mut grad = Col::zeros(n);
        let mut hess = need_hessian.then(|| Mat::zeros(n, n));
        for i in sys.atoms() {
            let nbs = &nl[i.index()];
            // Local variables: atom i first, then the distinct neighbor atoms.
            let mut atoms: Vec<AtomId> = vec![i];
            let mut sites: Vec<Site> = Vec::with_capacity(nbs.len());
            let pi = sys.position(i).value();
            for nb in nbs {
                let slot = match atoms.iter().position(|a| *a == nb.atom) {
                    Some(s) => s,
                    None => {
                        atoms.push(nb.atom);
                        atoms.len() - 1
                    }
                };
                sites.push(Site {
                    slot,
                    shift: image_shift(sys, i, nb, pi),
                });
            }
            let mut x: Vec<f64> = atoms
                .iter()
                .flat_map(|a| sys.position(*a).value().map(|c| c * to_ang))
                .collect();
            // The coordinates, and for the stress the nine components of a strain (zero).
            let coordinates: Vec<usize> = (0..x.len()).collect();
            let strain = virial.is_some().then_some(x.len());
            if strain.is_some() {
                x.extend([0.0; 9]);
            }
            for term in self.bond_terms(sys, i, nbs, &sites, strain)? {
                let (e, g_loc) = eval_value_and_gradient(&x, |v| term.energy(v));
                // With the strain Hessian: over the coordinates and the strain together.
                let active = match strain_hessian {
                    Some(_) => None,
                    None => Some(coordinates.as_slice()),
                };
                let h_loc =
                    need_hessian.then(|| eval_value_grad_hessian(&x, active, |v| term.energy(v)).2);
                if let (Some(sh), Some(hl), Some(at)) =
                    (strain_hessian.as_mut(), h_loc.as_ref(), strain)
                {
                    for (k, row) in sh.strain.iter_mut().enumerate() {
                        for (l, value) in row.iter_mut().enumerate() {
                            *value += hl[(at + k, at + l)];
                        }
                    }
                    // The coordinates of the variables are those before the strain, which
                    // moves them too: d/d eps_ab of dE/dx_p is that of dE/dr_p plus
                    // delta_pb dE/dr_a.
                    for (sa, a) in atoms.iter().enumerate() {
                        for q in 0..3 {
                            for k in 0..9 {
                                let (ka, kb) = (k / 3, k % 3);
                                let moved = if q == kb { g_loc[3 * sa + ka] } else { 0.0 };
                                sh.mixed[(a.dof(q), k)] += hl[(3 * sa + q, at + k)] - moved;
                            }
                        }
                    }
                }
                energy += e;
                if let (Some(w), Some(at)) = (virial.as_mut(), strain) {
                    for (a, row) in w.iter_mut().enumerate() {
                        for (b, value) in row.iter_mut().enumerate() {
                            *value += g_loc[at + 3 * a + b];
                        }
                    }
                }
                for (sa, a) in atoms.iter().enumerate() {
                    for c in 0..3 {
                        grad[a.dof(c)] += g_loc[3 * sa + c];
                    }
                }
                if let (Some(h), Some(hl)) = (hess.as_mut(), h_loc.as_ref()) {
                    for (sa, a) in atoms.iter().enumerate() {
                        for (sb, b) in atoms.iter().enumerate() {
                            for p in 0..3 {
                                for q in 0..3 {
                                    h[(a.dof(p), b.dof(q))] += hl[(3 * sa + p, 3 * sb + q)];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((energy, grad, hess, virial, strain_hessian))
    }
}

/// Constant shift (Angstrom) taking the neighbor atom's position to its image next to `i`.
fn image_shift(sys: &MolecularSystem, i: AtomId, nb: &Neighbor, pi: [f64; 3]) -> [f64; 3] {
    let pj = sys.position(nb.atom).value();
    let _ = i;
    std::array::from_fn(|c| (pi[c] + nb.vector[c] - pj[c]) * Units::BOHR_TO_ANGSTROM)
}

impl Calculator for Tersoff {
    fn analytic_hessian(&self) -> bool {
        true
    }

    fn citations(&self) -> Vec<String> {
        vec!["J. Tersoff, Phys. Rev. B 1989, 39, 5566 (Tersoff potential)".to_string()]
    }

    fn name(&self) -> &str {
        "tersoff"
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let (e_ev, g, h, virial, strain_hessian) =
            self.assemble(sys, req.need_hessian, req.need_stress)?;
        let stress = match (virial, sys.active_cell()) {
            (Some(w), Some(cell)) => {
                let volume = cell.volume_bohr3();
                Some(w.map(|row| row.map(|x| x * Units::EV_TO_HARTREE / volume)))
            }
            _ => None,
        };
        let to_ang = Units::BOHR_TO_ANGSTROM;
        let e = Units::EV_TO_HARTREE;
        Ok(EvalOutput {
            energy: Quantity::new(e_ev * e),
            gradient_bohr: g * (e * to_ang),
            hessian_bohr: h.map(|h| h * (e * to_ang * to_ang)),
            orbitals: None,
            stress,
            strain_hessian: strain_hessian.map(|sh| sh.scaled(e, e * to_ang)),
        })
    }

    fn site_terms(&self, sys: &MolecularSystem) -> Option<MoptResult<SiteTerms>> {
        Some(self.site_terms_of(sys))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "tersoff_tests.rs"]
mod tests;
