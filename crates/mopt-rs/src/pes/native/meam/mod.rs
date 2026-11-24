//! Modified embedded-atom method (M. I. Baskes, Phys. Rev. B 1992, 46, 2727) with the
//! many-body screening of Baskes, Angelo, and Bisson (Modell. Simul. Mater. Sci. Eng. 1994,
//! 2, 505), in the conventions of the MEAM library of LAMMPS, whose parameters it uses.
//!
//! ```text
//! E = sum_i [ F_i(rhobar_i) + 1/2 sum_(j != i) S_ij phi_ij(r_ij) ]
//! F_i(rhobar) = A_i E_i x ln(x),   x = rhobar / rhobar_i^ref
//! rhobar_i = rho_i^(0) G(Gamma_i),   Gamma_i = sum_(l=1..3) t_i^(l) (rho_i^(l))^2 / (rho_i^(0))^2
//! rho_i^(0)     = sum_j S_ij rho_j^a(0)(r_ij)
//! (rho_i^(1))^2 = sum_a [sum_j S_ij rho_j^a(1) u_a]^2
//! (rho_i^(2))^2 = sum_ab [sum_j S_ij rho_j^a(2) u_a u_b]^2 - 1/3 [sum_j S_ij rho_j^a(2)]^2
//! (rho_i^(3))^2 = sum_abc [sum_j S_ij rho_j^a(3) u_a u_b u_c]^2 - 3/5 sum_a [sum_j S_ij rho_j^a(3) u_a]^2
//! rho_j^a(l)(r) = rho_0j exp(-beta_j^(l) (r / r_j - 1)),   u = (r_j - r_i) / r_ij
//! t_i^(l) = sum_j t_j^(l) S_ij rho_j^a(0) / rho_i^(0)
//! phi_ab(r) = [2 E_ab^u(r) - F_a(rhobar_a^ref(r)) - F_b(rhobar_b^ref(r))] / Z_ab
//! ```
//!
//! `E^u` is the universal equation of state of the reference structure, and
//! `rhobar^ref(r)` the background density of an atom in it at the nearest-neighbour
//! distance `r`, so that the reference structure has exactly that energy. Below
//! `alpha (r / r_e - 1) = -1` the pair function goes over into the screened repulsion of
//! the nuclei (reached at `-3`).
//!
//! `S_ij` is the product of the cutoff function of the distance,
//! `f_c((r_c - r_ij) / 0.1 Angstrom)`, and of the screening by every third atom `k`,
//! `f_c((C_ikj - C_min) / (C_max - C_min))`, where `C` is the parameter of the ellipse
//! through `k` with `i` and `j` at the ends of one axis: with `X_ik = (r_ik / r_ij)^2`,
//! `X_jk = (r_jk / r_ij)^2`, `C = [2 (X_ik + X_jk) - (X_ik - X_jk)^2 - 1] / [1 - (X_ik - X_jk)^2]`.
//!
//! Energies in eV, lengths in Angstrom. What is taken from LAMMPS is the set of
//! conventions that the parameters were made for (the published weight `t^(1)` is raised
//! by `3/5 t^(3)` to go with the `3/5` term of the third density; `G`, cutoff function, and
//! defaults `C_min = 2`, `C_max = 2.8`), not code; the pair function is evaluated in closed
//! form here, where LAMMPS interpolates a table. Second-neighbour MEAM is not included.
//!
//! Gradient and Hessian are exact, in isolated systems and in periodic cells: every pair is
//! differentiated over the atoms it involves (the two atoms and those that screen it in
//! part), and the embedding energy over the sums of each atom. The cutoff function `f_c`
//! of the model has a jump in its second derivative where a neighbour or a screening atom
//! comes in, so the Hessian is discontinuous there. [`Meam::with_smooth_cutoff`] puts an
//! infinitely smooth step in its place (the same limits, 1/2 half way), for both the
//! distance and the screening; the reference structures, whose neighbours are either not
//! screened or screened completely, keep their energies. Either way the Hessian grows like
//! the logarithm of the density where an atom gets its first neighbour.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod functions;
mod parameters;
#[cfg(test)]
mod tests;

use faer::{Col, Mat};

use self::functions::{
    density_factor, embedding, infinitely_smooth_step, rose, short_range_weight, smooth_step, zbl,
};
pub use self::parameters::{Lattice, MeamElement, MeamPair, ELEMENTS};
use super::neighbors::{Neighbor, VerletList};
use crate::domain::{
    AtomId, Bohr, Element, EvalOutput, EvalRequest, MolecularSystem, Quantity, StrainHessian, Units,
};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_and_gradient, eval_value_grad_hessian, HyperDual, Scalar};
use crate::pes::calculator::Calculator;

/// Cutoff of the interactions (Angstrom), the default of LAMMPS, unless
/// [`CUTOFF_PER_DISTANCE`] times the largest nearest-neighbour distance of the elements is
/// longer.
pub const DEFAULT_CUTOFF_ANGSTROM: f64 = 4.0;
/// See [`DEFAULT_CUTOFF_ANGSTROM`]: keeps the nearest neighbours of large atoms (4 Angstrom
/// is shorter than the nearest-neighbour distance of potassium) within the cutoff up to a
/// stretch of a fifth. Energies and elastic constants of the crystals do not depend on it
/// (`docs/source/migration_surfaces.md`, "Modified embedded-atom method").
pub const CUTOFF_PER_DISTANCE: f64 = 1.25;
/// Width of the cutoff function of the distance (Angstrom).
const CUTOFF_WIDTH: f64 = 0.1;
/// Default `C_min` of the screening.
pub const DEFAULT_CMIN: f64 = 2.0;
/// Default `C_max` of the screening.
pub const DEFAULT_CMAX: f64 = 2.8;

/// Sums over the neighbours of an atom: `rho^(0)`, its three `t`-weighted copies, and the
/// tensor components of the partial densities (3, 6 + trace, 10 + 3).
const SUMS: usize = 27;
/// The sums and half the pair energy.
const OUTPUTS: usize = SUMS + 1;
/// Multiplicities of the components `xx, xy, xz, yy, yz, zz`.
const SECOND_WEIGHTS: [f64; 6] = [1.0, 2.0, 2.0, 1.0, 2.0, 1.0];
/// Multiplicities of `xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz`.
const THIRD_WEIGHTS: [f64; 10] = [1.0, 3.0, 3.0, 3.0, 6.0, 3.0, 1.0, 3.0, 3.0, 1.0];

/// An element with what follows from its parameters.
#[derive(Debug, Clone)]
struct Species {
    p: MeamElement,
    /// Weights with `t^(1)` raised by `3/5 t^(3)`.
    t: [f64; 3],
    /// Background density of the reference structure at its equilibrium.
    reference_density: f64,
}

/// Modified embedded-atom method.
#[derive(Debug, Clone)]
pub struct Meam {
    species: Vec<Species>,
    /// Reference of every pair of species (the element's own on the diagonal).
    pairs: Vec<Vec<MeamPair>>,
    /// Pairs of different species whose reference was given.
    given: Vec<(usize, usize)>,
    cutoff: f64,
    cmin: f64,
    cmax: f64,
    nuclear_repulsion: bool,
    /// An infinitely smooth step in place of the cutoff function of the model.
    smooth_cutoff: bool,
    /// The pairs kept between evaluations.
    verlet: VerletList,
}

/// Value and derivatives of the outputs of one pair with respect to its variables.
struct PairDerivatives {
    value: [f64; OUTPUTS],
    /// `first[v][q]`.
    first: Vec<[f64; OUTPUTS]>,
    /// `second[v * n + w][q]` (empty without second derivatives).
    second: Vec<[f64; OUTPUTS]>,
}

/// Differentiate `f` at `x` with hyper-dual numbers.
fn differentiate(
    x: &[f64],
    second: bool,
    f: impl Fn(&[HyperDual]) -> [HyperDual; OUTPUTS],
) -> PairDerivatives {
    let n = x.len();
    let mut vars: Vec<HyperDual> = x.iter().map(|v| HyperDual::constant(*v)).collect();
    let mut out = PairDerivatives {
        value: [0.0; OUTPUTS],
        first: vec![[0.0; OUTPUTS]; n],
        second: if second {
            vec![[0.0; OUTPUTS]; n * n]
        } else {
            Vec::new()
        },
    };
    for v in 0..n {
        vars[v].f1 = 1.0;
        let partners = if second { v..n } else { v..v + 1 };
        for w in partners {
            if second {
                vars[w].f2 = 1.0;
            }
            let y = f(&vars);
            if w == v {
                for (q, yq) in y.iter().enumerate() {
                    out.value[q] = yq.f0;
                    out.first[v][q] = yq.f1;
                }
            }
            if second {
                for (q, yq) in y.iter().enumerate() {
                    out.second[v * n + w][q] = yq.f12;
                    out.second[w * n + v][q] = yq.f12;
                }
                vars[w].f2 = 0.0;
            }
        }
        vars[v].f1 = 0.0;
    }
    out
}

/// Energy of one atom and its derivatives over the atoms around it (eV, Angstrom).
struct LocalTerm {
    atoms: Vec<AtomId>,
    energy: f64,
    gradient: Vec<f64>,
    hessian: Option<Mat<f64>>,
    /// `dE / d(strain)` of the term (eV): its derivative with respect to every site times
    /// the position of the site relative to the atom.
    virial: [[f64; 3]; 3],
    /// Its second derivatives along the strain (eV) and the change of its gradient along
    /// the strain (eV/Angstrom, one row per local coordinate), where they were asked for.
    strain: Option<([[f64; 9]; 9], Vec<[f64; 9]>)>,
}

/// The strain parts of every output of the pair contributions of a term: `strain[q]` the
/// second derivatives of output `q` along the strain, `mixed[q][v]` those of its derivative by
/// local coordinate `v`.
struct StrainParts {
    strain: Vec<[[f64; 9]; 9]>,
    mixed: Vec<Vec<[f64; 9]>>,
}

fn positive(name: &str, value: f64) -> MoptResult<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(MoptError::config(format!(
            "meam: {name} must be a positive number, got {value}"
        )))
    }
}

impl Meam {
    /// The library parameters for the given elements (see [`ELEMENTS`]); pairs of different
    /// elements get [`MeamPair::mean_of`] until [`Meam::with_pair`] gives their reference.
    pub fn library(elements: &[Element]) -> MoptResult<Self> {
        let mut unique = elements.to_vec();
        unique.sort();
        unique.dedup();
        Self::new(
            unique
                .into_iter()
                .map(MeamElement::library)
                .collect::<MoptResult<Vec<_>>>()?,
        )
    }

    /// From the parameters of the elements.
    pub fn new(elements: Vec<MeamElement>) -> MoptResult<Self> {
        let mut species = Vec::with_capacity(elements.len());
        for (n, p) in elements.iter().enumerate() {
            if elements[..n].iter().any(|q| q.element == p.element) {
                return Err(MoptError::config(format!(
                    "meam: {} is given twice",
                    p.element
                )));
            }
            if matches!(p.lattice, Lattice::B1 | Lattice::B2) {
                return Err(MoptError::config(format!(
                    "meam: the reference structure of {} must be that of an element \
                     (fcc, bcc, hcp, dia)",
                    p.element
                )));
            }
            if ![0, 1, 3, 4].contains(&p.ibar) {
                return Err(MoptError::config(format!(
                    "meam: form {} of G(Gamma) of {} is not available (0, 1, 3, 4)",
                    p.ibar, p.element
                )));
            }
            for (name, value) in [
                ("the nearest-neighbour distance", p.re),
                ("the cohesive energy", p.ec),
                ("the density scale", p.rho0),
                ("alpha", p.alpha),
            ] {
                positive(name, value)?;
            }
            let neighbours = p.lattice.neighbours();
            // Without angular contributions in the reference structure, G = 1 (and it is
            // taken as 1 for the square-root form).
            let factor = if p.ibar <= 0 {
                1.0
            } else {
                let gamma = p.t[2] * p.lattice.third_shape() / (neighbours * neighbours);
                density_factor(gamma, p.ibar)
            };
            species.push(Species {
                p: *p,
                t: [p.t[0] + 0.6 * p.t[2], p.t[1], p.t[2]],
                reference_density: p.rho0 * neighbours * factor,
            });
        }
        let pairs = elements
            .iter()
            .map(|a| {
                elements
                    .iter()
                    .map(|b| {
                        if a.element == b.element {
                            MeamPair {
                                lattice: a.lattice,
                                ec: a.ec,
                                re: a.re,
                                alpha: a.alpha,
                            }
                        } else {
                            MeamPair::mean_of(a, b)
                        }
                    })
                    .collect()
            })
            .collect();
        let longest = elements.iter().map(|p| p.re).fold(0.0, f64::max);
        Ok(Self {
            species,
            pairs,
            given: Vec::new(),
            cutoff: DEFAULT_CUTOFF_ANGSTROM.max(CUTOFF_PER_DISTANCE * longest),
            cmin: DEFAULT_CMIN,
            cmax: DEFAULT_CMAX,
            nuclear_repulsion: true,
            smooth_cutoff: false,
            verlet: VerletList::default(),
        })
    }

    fn index_of(&self, element: Element) -> MoptResult<usize> {
        self.species
            .iter()
            .position(|s| s.p.element == element)
            .ok_or_else(|| {
                MoptError::config(format!("this MEAM calculator was set up without {element}"))
            })
    }

    /// Give the reference of the pair of different elements `a` and `b`.
    pub fn with_pair(mut self, a: Element, b: Element, pair: MeamPair) -> MoptResult<Self> {
        let (ia, ib) = (self.index_of(a)?, self.index_of(b)?);
        if ia == ib {
            return Err(MoptError::config(format!(
                "meam: a pair reference is for two different elements, got {a}-{b}"
            )));
        }
        for (name, value) in [
            ("the cohesive energy of a pair", pair.ec),
            ("the distance of a pair", pair.re),
            ("alpha of a pair", pair.alpha),
        ] {
            positive(name, value)?;
        }
        self.pairs[ia][ib] = pair;
        self.pairs[ib][ia] = pair;
        let key = (ia.min(ib), ia.max(ib));
        if !self.given.contains(&key) {
            self.given.push(key);
        }
        Ok(self)
    }

    /// Set the cutoff of the interactions (Angstrom).
    pub fn with_cutoff(mut self, angstrom: f64) -> MoptResult<Self> {
        self.cutoff = positive("the cutoff", angstrom)?;
        Ok(self)
    }

    /// Set the limits of the screening: a third atom with `C <= cmin` screens a pair
    /// completely, one with `C >= cmax` not at all.
    pub fn with_screening(mut self, cmin: f64, cmax: f64) -> MoptResult<Self> {
        if !(cmin.is_finite() && cmax.is_finite() && cmin >= 0.0 && cmax > cmin) {
            return Err(MoptError::config(format!(
                "meam: the screening needs 0 <= cmin < cmax, got {cmin} and {cmax}"
            )));
        }
        self.cmin = cmin;
        self.cmax = cmax;
        Ok(self)
    }

    /// Whether the pair function goes over into the repulsion of the nuclei at short
    /// distances (default) or stays the one derived from the equation of state.
    pub fn with_nuclear_repulsion(mut self, on: bool) -> Self {
        self.nuclear_repulsion = on;
        self
    }

    /// Use an infinitely smooth step in place of the cutoff function `[1 - (1 - x)^4]^2` of
    /// the model, for the distance and for the screening: energy, gradient, and Hessian are
    /// then continuous where neighbours and screening atoms come in. Not the default: the
    /// parameters were made with the function of the model.
    pub fn with_smooth_cutoff(mut self, on: bool) -> Self {
        self.smooth_cutoff = on;
        self
    }

    /// The cutoff function: 0 up to `x = 0`, 1 from `x = 1`.
    fn step<T: Scalar>(&self, x: T) -> T {
        if self.smooth_cutoff {
            infinitely_smooth_step(x)
        } else {
            smooth_step(x)
        }
    }

    /// Cutoff of the interactions (Angstrom).
    pub fn cutoff_angstrom(&self) -> f64 {
        self.cutoff
    }

    /// Largest `(r_ik / r_ij)^2` at which a third atom can lie on an ellipse with
    /// `C < C_max`: farther atoms cannot screen the pair.
    fn ellipse_bound(&self) -> f64 {
        if self.cmax >= 2.0 {
            self.cmax * self.cmax / (4.0 * (self.cmax - 1.0))
        } else {
            1.0
        }
    }

    /// Pair function of the species `a` and `b` at the distance `r`.
    fn pair_potential<T: Scalar>(&self, a: usize, b: usize, r: T) -> T {
        let (sa, sb) = (&self.species[a], &self.species[b]);
        let pair = &self.pairs[a][b];
        let neighbours = pair.lattice.neighbours();
        let shape = pair.lattice.third_shape();
        // Embedding energy of an atom of `own` in the reference structure, among atoms of
        // `other` at the distance `r`.
        let embedded = |own: &Species, other: &Species| -> T {
            let reduced = r / T::from_f64(other.p.re) - T::one();
            let partial = |l: usize| {
                (reduced * T::from_f64(-other.p.beta[l])).exp() * T::from_f64(other.p.rho0)
            };
            let rho0 = partial(0) * T::from_f64(neighbours);
            let third = partial(3);
            let gamma = if rho0.val() > 1e-14 {
                third * third * T::from_f64(shape * other.t[2]) / (rho0 * rho0)
            } else {
                T::zero()
            };
            let relative =
                rho0 * density_factor(gamma, own.p.ibar) / T::from_f64(own.reference_density);
            embedding(own.p.a * own.p.ec, relative)
        };
        let state = rose(pair.ec, pair.re, pair.alpha, r);
        let energy = (state * T::from_f64(2.0) - embedded(sa, sb) - embedded(sb, sa))
            * T::from_f64(1.0 / neighbours);
        if !self.nuclear_repulsion {
            return energy;
        }
        let reduced = (r / T::from_f64(pair.re) - T::one()) * T::from_f64(pair.alpha);
        if reduced.val() >= -1.0 {
            return energy;
        }
        let nuclear = zbl(
            f64::from(sa.p.element.atomic_number()),
            f64::from(sb.p.element.atomic_number()),
            r,
        );
        if reduced.val() <= -3.0 {
            return nuclear;
        }
        let weight = short_range_weight(reduced);
        energy * weight + nuclear * (T::one() - weight)
    }

    /// What the neighbour of species `b` adds to the sums of an atom of species `a`, and
    /// half their pair energy. `x`: the atom, the neighbour, and the atoms that screen the
    /// pair in part, three coordinates each.
    fn pair_terms<T: Scalar>(&self, a: usize, b: usize, x: &[T]) -> [T; OUTPUTS] {
        let at = |site: usize| -> [T; 3] { [x[3 * site], x[3 * site + 1], x[3 * site + 2]] };
        let sub = |p: [T; 3], q: [T; 3]| -> [T; 3] { [p[0] - q[0], p[1] - q[1], p[2] - q[2]] };
        let (centre, neighbour) = (at(0), at(1));
        let d = sub(neighbour, centre);
        let r2 = T::dot3(d, d);
        let r = r2.sqrt();
        let mut weight =
            self.step((T::from_f64(self.cutoff) - r) * T::from_f64(1.0 / CUTOFF_WIDTH));
        for site in 2..x.len() / 3 {
            let k = at(site);
            let (dik, djk) = (sub(k, centre), sub(k, neighbour));
            let xik = T::dot3(dik, dik) / r2;
            let xjk = T::dot3(djk, djk) / r2;
            let difference = xik - xjk;
            let width = T::one() - difference * difference;
            let c = ((xik + xjk) * T::from_f64(2.0) + width - T::from_f64(2.0)) / width;
            weight = weight
                * self.step(
                    (c - T::from_f64(self.cmin)) * T::from_f64(1.0 / (self.cmax - self.cmin)),
                );
        }
        let other = &self.species[b];
        let reduced = r / T::from_f64(other.p.re) - T::one();
        let partial = |l: usize| {
            (reduced * T::from_f64(-other.p.beta[l])).exp() * T::from_f64(other.p.rho0) * weight
        };
        let u = d.map(|c| c / r);
        let mut out = [T::zero(); OUTPUTS];
        let rho = partial(0);
        out[0] = rho;
        for (l, t) in other.t.iter().enumerate() {
            out[1 + l] = rho * T::from_f64(*t);
        }
        let first = partial(1);
        for (c, uc) in u.iter().enumerate() {
            out[4 + c] = first * *uc;
        }
        let second = partial(2);
        let mut slot = 7;
        for m in 0..3 {
            for n in m..3 {
                out[slot] = second * u[m] * u[n];
                slot += 1;
            }
        }
        out[13] = second;
        let third = partial(3);
        let mut slot = 14;
        for m in 0..3 {
            for n in m..3 {
                for p in n..3 {
                    out[slot] = third * u[m] * u[n] * u[p];
                    slot += 1;
                }
            }
        }
        for (c, uc) in u.iter().enumerate() {
            out[24 + c] = third * *uc;
        }
        out[SUMS] = self.pair_potential(a, b, r) * weight * T::from_f64(0.5);
        out
    }

    /// Embedding energy of an atom of species `a` from its sums.
    fn embedding_of<T: Scalar>(&self, a: usize, q: &[T]) -> T {
        let s = &self.species[a];
        let rho0 = q[0];
        if rho0.val() <= 0.0 {
            // No neighbour that is not screened.
            return T::zero();
        }
        let mut first = T::zero();
        for c in &q[4..7] {
            first += *c * *c;
        }
        let mut second = -(q[13] * q[13]) * T::from_f64(1.0 / 3.0);
        for (c, w) in q[7..13].iter().zip(SECOND_WEIGHTS) {
            second += *c * *c * T::from_f64(w);
        }
        let mut third = T::zero();
        for (c, w) in q[14..24].iter().zip(THIRD_WEIGHTS) {
            third += *c * *c * T::from_f64(w);
        }
        for c in &q[24..27] {
            third -= *c * *c * T::from_f64(0.6);
        }
        // The weights are those of the neighbours, averaged by their densities:
        // t^(l) = q[l] / rho^(0).
        let gamma = (q[1] * first + q[2] * second + q[3] * third) / (rho0 * rho0 * rho0);
        let relative = rho0 * density_factor(gamma, s.p.ibar) / T::from_f64(s.reference_density);
        embedding(s.p.a * s.p.ec, relative)
    }

    /// The neighbours (indices into `sites`, positions relative to the central atom) that
    /// screen the pair of the central atom and neighbour `j` in part; `None` if one of
    /// them screens it completely.
    fn screening(&self, sites: &[[f64; 3]], j: usize) -> Option<Vec<usize>> {
        let norm2 = |v: [f64; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let end = sites[j];
        let rij2 = norm2(end);
        let bound = self.ellipse_bound() * rij2;
        let mut partial = Vec::new();
        for (k, site) in sites.iter().enumerate() {
            if k == j {
                continue;
            }
            let rik2 = norm2(*site);
            if rik2 > bound {
                continue;
            }
            let rjk2 = norm2([site[0] - end[0], site[1] - end[1], site[2] - end[2]]);
            if rjk2 > bound {
                continue;
            }
            let (xik, xjk) = (rik2 / rij2, rjk2 / rij2);
            let width = 1.0 - (xik - xjk) * (xik - xjk);
            if width <= 0.0 {
                // Not between the two atoms.
                continue;
            }
            let c = (2.0 * (xik + xjk) + width - 2.0) / width;
            if c >= self.cmax {
                continue;
            }
            if c <= self.cmin {
                return None;
            }
            partial.push(k);
        }
        Some(partial)
    }

    /// Energy of atom `i` (embedding energy and half its pair energies); with `need_strain`
    /// (and a Hessian) its second derivatives along the strain.
    fn atom_term(
        &self,
        i: AtomId,
        kinds: &[usize],
        nbs: &[Neighbor],
        need_hessian: bool,
        need_strain: bool,
    ) -> LocalTerm {
        let to_angstrom = Units::BOHR_TO_ANGSTROM;
        let own = kinds[i.index()];
        // The atoms the term depends on: atom i first, then the atoms of its neighbours
        // (an atom can appear through several of its images).
        let mut atoms: Vec<AtomId> = vec![i];
        let slots: Vec<usize> = nbs
            .iter()
            .map(|nb| match atoms.iter().position(|a| *a == nb.atom) {
                Some(slot) => slot,
                None => {
                    atoms.push(nb.atom);
                    atoms.len() - 1
                }
            })
            .collect();
        let sites: Vec<[f64; 3]> = nbs
            .iter()
            .map(|nb| nb.vector.map(|c| c * to_angstrom))
            .collect();
        let size = 3 * atoms.len();
        let mut sums = [0.0; SUMS];
        let mut jacobian = Mat::<f64>::zeros(SUMS, size);
        let mut energy = 0.0;
        let mut gradient = vec![0.0; size];
        let mut hessian = need_hessian.then(|| Mat::<f64>::zeros(size, size));
        // Pairs with the columns of their variables, kept for the second derivatives of
        // the sums.
        let mut curved: Vec<(Vec<usize>, PairDerivatives)> = Vec::new();
        let mut pair_virial = [[0.0; 3]; 3];
        let mut sum_virial = vec![[[0.0; 3]; 3]; SUMS];
        let mut parts = (need_hessian && need_strain).then(|| StrainParts {
            strain: vec![[[0.0; 9]; 9]; OUTPUTS],
            mixed: vec![vec![[0.0; 9]; size]; OUTPUTS],
        });
        for (jn, nb) in nbs.iter().enumerate() {
            if nb.distance * to_angstrom >= self.cutoff {
                continue;
            }
            let Some(screens) = self.screening(&sites, jn) else {
                continue;
            };
            let mut x = vec![0.0; 3];
            x.extend_from_slice(&sites[jn]);
            let mut columns: Vec<usize> = (0..3).chain((0..3).map(|c| 3 * slots[jn] + c)).collect();
            for k in &screens {
                x.extend_from_slice(&sites[*k]);
                columns.extend((0..3).map(|c| 3 * slots[*k] + c));
            }
            let other = kinds[nb.atom.index()];
            let d = differentiate(&x, need_hessian, |v| self.pair_terms(own, other, v));
            // Under a homogeneous strain every site moves by the strain times its position
            // relative to the atom.
            for (v, row) in d.first.iter().enumerate() {
                let (a, site) = (v % 3, v / 3);
                for b in 0..3 {
                    let lever = x[3 * site + b];
                    pair_virial[a][b] += row[SUMS] * lever;
                    for (w, value) in sum_virial.iter_mut().zip(&row[..SUMS]) {
                        w[a][b] += value * lever;
                    }
                }
            }
            for (sum, value) in sums.iter_mut().zip(d.value) {
                *sum += value;
            }
            energy += d.value[SUMS];
            for (row, column) in d.first.iter().zip(&columns) {
                for (q, value) in row[..SUMS].iter().enumerate() {
                    jacobian[(q, *column)] += value;
                }
                gradient[*column] += row[SUMS];
            }
            if let Some(st) = parts.as_mut() {
                // A strain moves every site by the strain times its position relative to the
                // atom: dx'_w / d eps_ab = delta_(a, w mod 3) x_(site w, b), linear in eps.
                let n = columns.len();
                for (v, cv) in columns.iter().enumerate() {
                    let mut along = [[0.0; 9]; OUTPUTS];
                    for w in 0..n {
                        let (a, site) = (w % 3, w / 3);
                        let second = &d.second[v * n + w];
                        for b in 0..3 {
                            let lever = x[3 * site + b];
                            if lever != 0.0 {
                                for (row, value) in along.iter_mut().zip(second) {
                                    row[3 * a + b] += value * lever;
                                }
                            }
                        }
                    }
                    let (c, site) = (v % 3, v / 3);
                    for (q, row) in along.iter().enumerate() {
                        for (k, value) in row.iter().enumerate() {
                            st.mixed[q][*cv][k] += value;
                        }
                        for dd in 0..3 {
                            let lever = x[3 * site + dd];
                            if lever != 0.0 {
                                for (k, value) in row.iter().enumerate() {
                                    st.strain[q][k][3 * c + dd] += value * lever;
                                }
                            }
                        }
                    }
                }
            }
            if let Some(h) = hessian.as_mut() {
                let n = columns.len();
                for (v, cv) in columns.iter().enumerate() {
                    for (w, cw) in columns.iter().enumerate() {
                        h[(*cv, *cw)] += d.second[v * n + w][SUMS];
                    }
                }
                curved.push((columns, d));
            }
        }
        // The strain derivative once the slopes of the embedding energy are known.
        let strain_of = |slope: &Col<f64>| -> [[f64; 3]; 3] {
            let mut virial = pair_virial;
            for (q, w) in sum_virial.iter().enumerate() {
                for a in 0..3 {
                    for b in 0..3 {
                        virial[a][b] += slope[q] * w[a][b];
                    }
                }
            }
            virial
        };
        let virial;
        let mut strain = None;
        match hessian.as_mut() {
            None => {
                let (f, slope) = eval_value_and_gradient(&sums, |q| self.embedding_of(own, q));
                energy += f;
                virial = strain_of(&slope);
                for (column, g) in gradient.iter_mut().enumerate() {
                    *g += (0..SUMS)
                        .map(|q| slope[q] * jacobian[(q, column)])
                        .sum::<f64>();
                }
            }
            Some(h) => {
                let (f, slope, curvature) =
                    eval_value_grad_hessian(&sums, None, |q| self.embedding_of(own, q));
                energy += f;
                virial = strain_of(&slope);
                for (column, g) in gradient.iter_mut().enumerate() {
                    *g += (0..SUMS)
                        .map(|q| slope[q] * jacobian[(q, column)])
                        .sum::<f64>();
                }
                // F'' along the gradients of the sums, and F' times their second derivatives.
                let along = &curvature * &jacobian;
                *h += jacobian.transpose() * &along;
                if let Some(st) = parts.as_ref() {
                    // The pair energy, F' along the second strain derivatives of the sums, and
                    // F'' along their first ones (with themselves and with the gradient).
                    let first: Vec<[f64; 9]> = sum_virial
                        .iter()
                        .map(|w| std::array::from_fn(|k| w[k / 3][k % 3]))
                        .collect();
                    let mut k_term = st.strain[SUMS];
                    let mut m_term = st.mixed[SUMS].clone();
                    for q in 0..SUMS {
                        for (row, add) in k_term.iter_mut().zip(&st.strain[q]) {
                            for (value, part) in row.iter_mut().zip(add) {
                                *value += slope[q] * part;
                            }
                        }
                        for (row, add) in m_term.iter_mut().zip(&st.mixed[q]) {
                            for (value, part) in row.iter_mut().zip(add) {
                                *value += slope[q] * part;
                            }
                        }
                        for r in 0..SUMS {
                            let c = curvature[(q, r)];
                            if c == 0.0 {
                                continue;
                            }
                            for (x, row) in k_term.iter_mut().enumerate() {
                                for (y, value) in row.iter_mut().enumerate() {
                                    *value += c * first[q][x] * first[r][y];
                                }
                            }
                            for (column, row) in m_term.iter_mut().enumerate() {
                                let j = jacobian[(q, column)];
                                if j != 0.0 {
                                    for (value, f) in row.iter_mut().zip(&first[r]) {
                                        *value += c * j * f;
                                    }
                                }
                            }
                        }
                    }
                    strain = Some((k_term, m_term));
                }
                for (columns, d) in &curved {
                    let n = columns.len();
                    for (v, cv) in columns.iter().enumerate() {
                        for (w, cw) in columns.iter().enumerate() {
                            let second = &d.second[v * n + w];
                            h[(*cv, *cw)] += (0..SUMS).map(|q| slope[q] * second[q]).sum::<f64>();
                        }
                    }
                }
            }
        }
        LocalTerm {
            atoms,
            energy,
            gradient,
            hessian,
            virial,
            strain,
        }
    }

    /// Energy (eV), gradient (eV/Angstrom), Hessian (eV/Angstrom^2), `dE / d(strain)` (eV),
    /// and with `need_strain` (and a Hessian) the second derivatives along the strain (eV,
    /// eV/Angstrom).
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
        [[f64; 3]; 3],
        Option<StrainHessian>,
    )> {
        let mut virial = [[0.0; 3]; 3];
        let mut strain_hessian =
            (need_hessian && need_strain).then(|| StrainHessian::zeros(sys.dof()));
        let kinds = sys
            .elements
            .iter()
            .map(|e| self.index_of(*e))
            .collect::<MoptResult<Vec<usize>>>()?;
        // Atoms that screen a pair can lie a little beyond the cutoff of the pair.
        let reach = self.cutoff * self.ellipse_bound().sqrt();
        let nl = self
            .verlet
            .neighbors(sys, Quantity::<Bohr>::new(reach / Units::BOHR_TO_ANGSTROM));
        let n = sys.dof();
        let mut energy = 0.0;
        let mut grad = Col::zeros(n);
        let mut hess = need_hessian.then(|| Mat::zeros(n, n));
        for i in sys.atoms() {
            let term = self.atom_term(i, &kinds, &nl[i.index()], need_hessian, need_strain);
            if let (Some(sh), Some((k_term, m_term))) = (strain_hessian.as_mut(), &term.strain) {
                for (row, add) in sh.strain.iter_mut().zip(k_term) {
                    for (value, part) in row.iter_mut().zip(add) {
                        *value += part;
                    }
                }
                for (slot, atom) in term.atoms.iter().enumerate() {
                    for c in 0..3 {
                        for (k, value) in m_term[3 * slot + c].iter().enumerate() {
                            sh.mixed[(atom.dof(c), k)] += value;
                        }
                    }
                }
            }
            energy += term.energy;
            for (total, row) in virial.iter_mut().zip(&term.virial) {
                for (value, part) in total.iter_mut().zip(row) {
                    *value += part;
                }
            }
            for (slot, atom) in term.atoms.iter().enumerate() {
                for c in 0..3 {
                    grad[atom.dof(c)] += term.gradient[3 * slot + c];
                }
            }
            if let (Some(h), Some(local)) = (hess.as_mut(), term.hessian.as_ref()) {
                for (sa, a) in term.atoms.iter().enumerate() {
                    for (sb, b) in term.atoms.iter().enumerate() {
                        for p in 0..3 {
                            for q in 0..3 {
                                h[(a.dof(p), b.dof(q))] += local[(3 * sa + p, 3 * sb + q)];
                            }
                        }
                    }
                }
            }
        }
        Ok((energy, grad, hess, virial, strain_hessian))
    }
}

impl Calculator for Meam {
    fn analytic_hessian(&self) -> bool {
        true
    }

    fn citations(&self) -> Vec<String> {
        vec![
            "M. I. Baskes, Phys. Rev. B 1992, 46, 2727 (modified embedded-atom method)".to_string(),
            "M. I. Baskes, J. E. Angelo, C. L. Bisson, Modell. Simul. Mater. Sci. Eng. 1994, 2, 505 (screening)".to_string(),
        ]
    }

    fn name(&self) -> &str {
        "meam"
    }

    fn description(&self) -> String {
        let symbols: Vec<&str> = self.species.iter().map(|s| s.p.element.symbol()).collect();
        let mut text = format!(
            "meam: modified embedded-atom method (Baskes 1992) built into mopt-rs {} for {}; \
             cutoff {:.3} Angstrom, screening C_min {} C_max {}, {}",
            crate::VERSION,
            symbols.join(" "),
            self.cutoff,
            self.cmin,
            self.cmax,
            if self.smooth_cutoff {
                "infinitely smooth cutoff function"
            } else {
                "cutoff function of the model"
            }
        );
        let mut unfitted = Vec::new();
        for a in 0..self.species.len() {
            for b in a + 1..self.species.len() {
                if !self.given.contains(&(a, b)) {
                    unfitted.push(format!("{}-{}", symbols[a], symbols[b]));
                }
            }
        }
        if !unfitted.is_empty() {
            text.push_str(&format!(
                "; no reference given for {}: fcc with the means of the elements, not fitted \
                 to the alloy",
                unfitted.join(", ")
            ));
        }
        text
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let need_strain = req.need_stress && sys.active_cell().is_some();
        let (energy, gradient, hessian, virial, strain_hessian) =
            self.assemble(sys, req.need_hessian, need_strain)?;
        let stress = sys.active_cell().map(|cell| {
            let volume = cell.volume_bohr3();
            virial.map(|row| row.map(|x| x * Units::EV_TO_HARTREE / volume))
        });
        let to_angstrom = Units::BOHR_TO_ANGSTROM;
        let hartree = Units::EV_TO_HARTREE;
        Ok(EvalOutput {
            energy: Quantity::new(energy * hartree),
            gradient_bohr: gradient * (hartree * to_angstrom),
            hessian_bohr: hessian.map(|h| h * (hartree * to_angstrom * to_angstrom)),
            orbitals: None,
            stress,
            strain_hessian: strain_hessian.map(|sh| sh.scaled(hartree, hartree * to_angstrom)),
        })
    }
}
