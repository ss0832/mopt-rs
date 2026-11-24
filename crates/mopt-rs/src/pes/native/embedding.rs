//! Shared engine for embedded-atom-type potentials (EMT, EAM, ...):
//!
//! `E = sum_i F_i(rho_i) + sum_i sum_{j != i} phi_ij(r_ij)`,  `rho_i = sum_{j != i} g_ij(r_ij)`
//!
//! (sums over a full neighbor list, so `phi_ij` is the energy of the ordered pair). The model
//! supplies the one-dimensional functions `g`, `F`, and `phi`, written once over [`Scalar`];
//! the engine assembles exact gradients and Hessians by the chain rule:
//!
//! `H = sum_i [F''(rho_i) grad(rho_i) grad(rho_i)^T + F'(rho_i) Hess(rho_i)] + Hess(phi terms)`,
//!
//! with radial second derivatives `d2r/dx dx = (I - u u^T) / r`. Periodic images are included
//! through [`neighbor_list`]. In a cell the stress comes from the strain derivatives of the
//! radial coordinates, and with a Hessian the second derivatives along the strain too
//! ([`crate::domain::StrainHessian`]): those of the radial terms, and `F''(rho_i)` along the
//! strain gradient of `rho_i`. A model that names the start of a switching range has `g` and
//! `phi` multiplied by [`switch_off`] from there to its cutoff, so that the energy and all
//! its derivatives are continuous where a neighbor crosses the cutoff.
//!
//! MultiOptPy: `Calculator/emt_calculation_tools.py` (shared embedded-atom engine).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::neighbors::{neighbor_list, Neighbor, VerletList};
use super::switching::switch_off;
use crate::domain::{
    AtomId, Bohr, Element, EvalRequest, Hartree, MolecularSystem, Quantity, StrainHessian,
};
use crate::error::MoptResult;
use crate::math::ad::{HyperDual, Scalar};

/// Radial and embedding functions of an embedded-atom-type model, in the model's own units.
pub trait EmbeddingModel: Send + Sync {
    /// Check that every element is parameterized.
    fn validate(&self, elements: &[Element]) -> MoptResult<()>;
    /// Neighbor cutoff in model length units.
    fn cutoff(&self) -> f64;
    /// Distance (model length units) from which density and pair energy are switched off
    /// towards the cutoff; `None`: they are cut as they are.
    fn switch_on(&self) -> Option<f64> {
        None
    }
    /// Contribution of neighbor `j` at distance `r` to the density at atom `i`.
    fn density<T: Scalar>(&self, i: Element, j: Element, r: T) -> T;
    /// Embedding energy of atom `i` at density `rho`.
    fn embedding<T: Scalar>(&self, i: Element, rho: T) -> T;
    /// Energy of the ordered pair `(i, j)` at distance `r`.
    fn pair<T: Scalar>(&self, i: Element, j: Element, r: T) -> T;
    /// The Verlet list the model keeps between evaluations; `None`: a fresh search every time.
    fn verlet_list(&self) -> Option<&VerletList> {
        None
    }
}

/// The neighbors of `sys` within `cutoff` from the Verlet list of `model`, or a fresh search.
fn neighbors_of<M: EmbeddingModel>(
    model: &M,
    sys: &MolecularSystem,
    cutoff: Quantity<Bohr>,
) -> Vec<Vec<Neighbor>> {
    match model.verlet_list() {
        Some(list) => list.neighbors(sys, cutoff),
        None => neighbor_list(sys, cutoff),
    }
}

/// Unit conversions between Bohr/Hartree and the model's units.
#[derive(Debug, Clone, Copy)]
pub struct ModelUnits {
    /// Model length units per Bohr.
    pub length_per_bohr: f64,
    /// Hartree per model energy unit.
    pub hartree_per_energy: f64,
}

/// Value, first, and second derivative of `f` at `x`.
fn d012(f: impl Fn(HyperDual) -> HyperDual, x: f64) -> (f64, f64, f64) {
    let y = f(HyperDual {
        f0: x,
        f1: 1.0,
        f2: 1.0,
        f12: 0.0,
    });
    (y.f0, y.f1, y.f12)
}

struct Accumulator {
    energy: f64,
    grad: Col<f64>,
    hess: Option<Mat<f64>>,
    /// `dE / d(strain)` in model units: `sum (dE/dr) r u u^T` over all radial coordinates,
    /// those to periodic images of the atom itself included.
    virial: [[f64; 3]; 3],
    /// The second derivatives along the strain, in model units (with a Hessian in a cell).
    strain_hessian: Option<StrainHessian>,
}

/// Energy, gradient, Hessian (on request), stress (structures in a cell), and the second
/// derivatives along the strain (with a Hessian and a stress asked for, in a cell) in Hartree
/// and Bohr.
pub type EmbeddingOutput = (
    Quantity<Hartree>,
    Col<f64>,
    Option<Mat<f64>>,
    Option<[[f64; 3]; 3]>,
    Option<StrainHessian>,
);

impl Accumulator {
    /// A radial coordinate `r` along `u` with `dE/dr = c1` under a homogeneous strain.
    fn strain(&mut self, r: f64, u: [f64; 3], c1: f64) {
        for a in 0..3 {
            for b in 0..3 {
                self.virial[a][b] += c1 * r * u[a] * u[b];
            }
        }
    }

    /// For the radial coordinate `r` of neighbor `nb` of atom `i` (unit vector `u`): add
    /// `c1 dr` to the gradient when `with_gradient`, and `c2 dr dr^T + c1 d2r` to the Hessian.
    #[allow(clippy::too_many_arguments)]
    fn radial(
        &mut self,
        i: AtomId,
        nb: &Neighbor,
        r: f64,
        u: [f64; 3],
        c1: f64,
        c2: f64,
        with_gradient: bool,
    ) {
        if let Some(sh) = self.strain_hessian.as_mut() {
            let atoms = (!nb.self_image).then_some((i.index(), nb.atom.index()));
            sh.add_radial(atoms, r, u, c1, c2);
        }
        if nb.self_image {
            return;
        }
        let j = nb.atom;
        if with_gradient {
            for (c, uc) in u.iter().enumerate() {
                self.grad[j.dof(c)] += c1 * uc;
                self.grad[i.dof(c)] -= c1 * uc;
            }
        }
        if let Some(h) = self.hess.as_mut() {
            for p in 0..3 {
                for q in 0..3 {
                    let id = if p == q { 1.0 } else { 0.0 };
                    let k = c2 * u[p] * u[q] + c1 * (id - u[p] * u[q]) / r;
                    h[(j.dof(p), j.dof(q))] += k;
                    h[(i.dof(p), i.dof(q))] += k;
                    h[(j.dof(p), i.dof(q))] -= k;
                    h[(i.dof(p), j.dof(q))] -= k;
                }
            }
        }
    }
}

/// Energy, gradient, (with `req.need_hessian`) Hessian, stress, and (with a Hessian and
/// `req.need_stress` in a cell) the second derivatives along the strain, in Hartree and Bohr.
pub fn evaluate_embedding<M: EmbeddingModel>(
    model: &M,
    units: ModelUnits,
    sys: &MolecularSystem,
    req: &EvalRequest,
) -> MoptResult<EmbeddingOutput> {
    model.validate(&sys.elements)?;
    let n = sys.dof();
    let cutoff = Quantity::<Bohr>::new(model.cutoff() / units.length_per_bohr);
    let nl = neighbors_of(model, sys, cutoff);
    let need_hessian = req.need_hessian;
    let need_strain = need_hessian && req.need_stress && sys.active_cell().is_some();
    let mut acc = Accumulator {
        energy: 0.0,
        grad: Col::zeros(n),
        hess: need_hessian.then(|| Mat::zeros(n, n)),
        virial: [[0.0; 3]; 3],
        strain_hessian: need_strain.then(|| StrainHessian::zeros(n)),
    };
    let s = units.length_per_bohr;
    let range = model.switch_on().map(|on| (on, model.cutoff()));
    let switched = |x: HyperDual, y: HyperDual| match range {
        Some((on, off)) => y * switch_off(x, on, off),
        None => y,
    };
    for i in sys.atoms() {
        let ei = sys.element(i);
        let nbs = &nl[i.index()];
        // Radial geometry in model units.
        let geo: Vec<(f64, [f64; 3])> = nbs
            .iter()
            .map(|nb| {
                let r = nb.distance * s;
                (r, nb.vector.map(|c| c * s / r))
            })
            .collect();
        let dens: Vec<(f64, f64, f64)> = nbs
            .iter()
            .zip(&geo)
            .map(|(nb, (r, _))| {
                d012(
                    |x| switched(x, model.density(ei, sys.element(nb.atom), x)),
                    *r,
                )
            })
            .collect();
        let rho: f64 = dens.iter().map(|d| d.0).sum();
        let (f, f1, f2) = d012(|x| model.embedding(ei, x), rho);
        acc.energy += f;

        // Sparse gradient of rho_i.
        let mut grad_rho: Vec<(AtomId, [f64; 3])> = Vec::new();
        let mut add = |a: AtomId, v: [f64; 3]| match grad_rho.iter_mut().find(|(b, _)| *b == a) {
            Some((_, g)) => (0..3).for_each(|c| g[c] += v[c]),
            None => grad_rho.push((a, v)),
        };
        for ((nb, (_, u)), (_, g1, _)) in nbs.iter().zip(&geo).zip(&dens) {
            if !nb.self_image {
                add(nb.atom, u.map(|c| g1 * c));
                add(i, u.map(|c| -g1 * c));
            }
        }
        for (a, v) in &grad_rho {
            for (c, vc) in v.iter().enumerate() {
                acc.grad[a.dof(c)] += f1 * vc;
            }
        }
        if let Some(h) = acc.hess.as_mut() {
            for (a, va) in &grad_rho {
                for (b, vb) in &grad_rho {
                    for p in 0..3 {
                        for q in 0..3 {
                            h[(a.dof(p), b.dof(q))] += f2 * va[p] * vb[q];
                        }
                    }
                }
            }
        }
        if let Some(sh) = acc.strain_hessian.as_mut() {
            // F'' along the strain gradient of rho_i (its own images included), with itself
            // and with the gradient of rho_i in the positions.
            let mut tau = [0.0; 9];
            for ((_, (r, u)), (_, g1, _)) in nbs.iter().zip(&geo).zip(&dens) {
                for (k, t) in tau.iter_mut().enumerate() {
                    *t += g1 * r * u[k / 3] * u[k % 3];
                }
            }
            for (x, row) in sh.strain.iter_mut().enumerate() {
                for (y, value) in row.iter_mut().enumerate() {
                    *value += f2 * tau[x] * tau[y];
                }
            }
            for (a, va) in &grad_rho {
                for (p, vp) in va.iter().enumerate() {
                    for (k, t) in tau.iter().enumerate() {
                        sh.mixed[(a.dof(p), k)] += f2 * vp * t;
                    }
                }
            }
        }
        for ((nb, (r, u)), (_, g1, g2)) in nbs.iter().zip(&geo).zip(&dens) {
            // The gradient part was added through grad(rho_i) above.
            acc.radial(i, nb, *r, *u, f1 * g1, f1 * g2, false);
            acc.strain(*r, *u, f1 * g1);
        }

        // Pair terms.
        for (nb, (r, u)) in nbs.iter().zip(&geo) {
            let (p0, p1, p2) = d012(|x| switched(x, model.pair(ei, sys.element(nb.atom), x)), *r);
            acc.energy += p0;
            acc.radial(i, nb, *r, *u, p1, p2, true);
            acc.strain(*r, *u, p1);
        }
    }
    finish(acc, units, sys)
}

/// The energy and the virials of the atoms (Hartree): `U_i = F_i(rho_i) + sum_j phi_ij`, and
/// `W_i = sum_j r_ij (x) dU_j/dr_ji` with `dU_j/dr_ji = (F_j'(rho_j) g_ji' + phi_ji') u_ji`
/// (`u_ji` from `j` to `i`), whose sum over the atoms is minus the virial of the stress.
/// The pairs of an atom with its own periodic images count as the other pairs (as in the
/// stress of the atoms of LAMMPS, `compute stress/atom`), so that the sum holds in any cell.
pub fn site_terms<M: EmbeddingModel>(
    model: &M,
    units: ModelUnits,
    sys: &MolecularSystem,
) -> MoptResult<crate::pes::calculator::SiteTerms> {
    model.validate(&sys.elements)?;
    let cutoff = Quantity::<Bohr>::new(model.cutoff() / units.length_per_bohr);
    let nl = neighbors_of(model, sys, cutoff);
    let (s, e) = (units.length_per_bohr, units.hartree_per_energy);
    let range = model.switch_on().map(|on| (on, model.cutoff()));
    let switched = |x: HyperDual, y: HyperDual| match range {
        Some((on, off)) => y * switch_off(x, on, off),
        None => y,
    };
    let n = sys.n_atoms();
    let mut energies = vec![0.0; n];
    let mut virials = vec![[[0.0; 3]; 3]; n];
    for j in sys.atoms() {
        let ej = sys.element(j);
        let nbs = &nl[j.index()];
        let dens: Vec<(f64, f64, f64)> = nbs
            .iter()
            .map(|nb| {
                d012(
                    |x| switched(x, model.density(ej, sys.element(nb.atom), x)),
                    nb.distance * s,
                )
            })
            .collect();
        let rho: f64 = dens.iter().map(|d| d.0).sum();
        let (f, f1, _) = d012(|x| model.embedding(ej, x), rho);
        let mut u = f;
        for (nb, (_, g1, _)) in nbs.iter().zip(&dens) {
            let (p0, p1, _) = d012(
                |x| switched(x, model.pair(ej, sys.element(nb.atom), x)),
                nb.distance * s,
            );
            u += p0;
            // Own images included: with them the virials of the atoms add up to that of
            // the cell, and the energy an atom exchanges with its images crosses the cell.
            // dU_j / dr (Hartree per Bohr) along u_ji = vector / distance.
            let c = (f1 * g1 + p1) * e * s;
            let w = &mut virials[nb.atom.index()];
            for (row, va) in w.iter_mut().zip(nb.vector) {
                for (x, vb) in row.iter_mut().zip(nb.vector) {
                    // r_ij = r_j - r_i = -vector.
                    *x -= va * c * vb / nb.distance;
                }
            }
        }
        energies[j.index()] = u * e;
    }
    Ok(crate::pes::calculator::SiteTerms { energies, virials })
}

fn finish(
    acc: Accumulator,
    units: ModelUnits,
    sys: &MolecularSystem,
) -> MoptResult<EmbeddingOutput> {
    let s = units.length_per_bohr;
    let e = units.hartree_per_energy;
    let g_scale = e * s;
    let h_scale = e * s * s;
    let grad = acc.grad * g_scale;
    let hess = acc.hess.map(|h| h * h_scale);
    let stress = sys.active_cell().map(|cell| {
        let volume = cell.volume_bohr3();
        acc.virial.map(|row| row.map(|x| x * e / volume))
    });
    // The strain has no unit: its second derivatives are energies, the mixed ones
    // energies per length.
    let strain = acc.strain_hessian.map(|sh| sh.scaled(e, g_scale));
    Ok((Quantity::new(acc.energy * e), grad, hess, stress, strain))
}
