//! Lennard-Jones type repulsive potentials (MultiOptPy `-rp`, `-rpv2`, `-rpg`, `-cp`).
//!
//! Per-atom parameters are UFF van der Waals distances and well depths (Rappe et al., J. Am.
//! Chem. Soc. 1992, 114, 10024), resolved when the term is built.
//!
//! MultiOptPy: `Potential/LJ_repulsive_potential.py` (`LJRepulsivePotentialScale`, `...Value`,
//!  `...v2Scale`, `...v2Value`, `...Gaussian`, `...Cone`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::engine::{norm, Clock, Geom, GeometricBias};
use crate::domain::{AtomId, Element, Units};
use crate::math::ad::{angle_between, Scalar};
use crate::parameters::forcefield::{uff_vdw_distance_bohr, uff_vdw_well_depth_hartree};
use crate::pes::term::{Frame, History};

/// `eps [(sigma/r)^12 - 2 (sigma/r)^6]`.
fn lj<T: Scalar>(r: T, eps: f64, sigma: f64) -> T {
    let s6 = (T::from_f64(sigma) / r).powi(6);
    T::from_f64(eps) * (s6 * s6 - T::from_f64(2.0) * s6)
}

fn element(elements: &[Element], a: AtomId) -> Element {
    elements.get(a.index()).copied().unwrap_or(Element::H)
}

/// How the LJ parameters of a pair are chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LjParameters {
    /// UFF combination scaled: `eps = s_eps sqrt(eps_i eps_j)`, `sigma = s_sigma sqrt(sigma_i
    /// sigma_j)` (MultiOptPy's `scale`).
    Scale {
        /// Well-depth scale.
        well: f64,
        /// Distance scale.
        distance: f64,
    },
    /// The same `eps` (Hartree) and `sigma` (Bohr) for every pair (MultiOptPy's `value`).
    Value {
        /// Well depth (Hartree).
        well: f64,
        /// Distance (Bohr).
        distance: f64,
    },
}

/// LJ potential between every atom of fragment 1 and every atom of fragment 2 (MultiOptPy `-rp`).
#[derive(Debug, Clone)]
pub struct LjRepulsive {
    /// `(a, b, eps, sigma)` per pair.
    pub pairs: Vec<(AtomId, AtomId, f64, f64)>,
}

impl LjRepulsive {
    /// Pairs between `a` and `b` with parameters from `params`.
    pub fn new(elements: &[Element], a: &[AtomId], b: &[AtomId], params: LjParameters) -> Self {
        let mut pairs = Vec::new();
        for &i in a {
            for &j in b.iter().filter(|&&j| j != i) {
                let (ei, ej) = (element(elements, i), element(elements, j));
                let (eps, sigma) = match params {
                    LjParameters::Scale { well, distance } => (
                        well * (uff_vdw_well_depth_hartree(ei) * uff_vdw_well_depth_hartree(ej))
                            .sqrt(),
                        distance * (uff_vdw_distance_bohr(ei) * uff_vdw_distance_bohr(ej)).sqrt(),
                    ),
                    LjParameters::Value { well, distance } => (well, distance),
                };
                pairs.push((i, j, eps, sigma));
            }
        }
        Self { pairs }
    }
}

impl GeometricBias for LjRepulsive {
    fn name(&self) -> &str {
        "repulsive_potential"
    }
    fn atoms(&self) -> Vec<AtomId> {
        self.pairs.iter().flat_map(|p| [p.0, p.1]).collect()
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        self.pairs.iter().fold(T::zero(), |e, &(a, b, eps, sigma)| {
            e + lj(norm(g.d(a, b)), eps, sigma)
        })
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        History::Static
    }
}

/// Generalized LJ potential centered at a point on the extension of a bond:
/// `eps [|A| (sigma/r)^n - |B| (sigma/r)^m]` for target atoms around
/// `c = r_1 + L (r_1 - r_0) / |r_1 - r_0|` (MultiOptPy `-rpv2`).
#[derive(Debug, Clone)]
pub struct LjCenter {
    /// Atoms `(r_0, r_1)` defining the center.
    pub center: [AtomId; 2],
    /// Distance `L` of the center beyond `r_1` (Bohr).
    pub length: f64,
    /// `(atom, eps, sigma)` per target.
    pub targets: Vec<(AtomId, f64, f64)>,
    /// Repulsive constant `A`.
    pub rep_const: f64,
    /// Attractive constant `B`.
    pub attr_const: f64,
    /// Repulsive order `n`.
    pub rep_order: f64,
    /// Attractive order `m`.
    pub attr_order: f64,
}

impl LjCenter {
    /// Target parameters as in MultiOptPy: with `Scale`, `eps = sqrt(s_eps eps_c eps_t)` and
    /// `sigma = sqrt(s_sigma sigma_c sigma_t)` (`c`: the atom `r_1`); with `Value`,
    /// `eps = sqrt(eps_v eps_t)` and `sigma = sqrt(sigma_v sigma_t)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        elements: &[Element],
        center: [AtomId; 2],
        length: f64,
        targets: &[AtomId],
        params: LjParameters,
        constants: [f64; 2],
        orders: [f64; 2],
    ) -> Self {
        let ec = element(elements, center[1]);
        let targets = targets
            .iter()
            .map(|&t| {
                let et = element(elements, t);
                let (wt, dt) = (uff_vdw_well_depth_hartree(et), uff_vdw_distance_bohr(et));
                let (eps, sigma) = match params {
                    LjParameters::Scale { well, distance } => (
                        (well * uff_vdw_well_depth_hartree(ec) * wt).sqrt(),
                        (distance * uff_vdw_distance_bohr(ec) * dt).sqrt(),
                    ),
                    LjParameters::Value { well, distance } => {
                        ((well * wt).sqrt(), (distance * dt).sqrt())
                    }
                };
                (t, eps, sigma)
            })
            .collect();
        Self {
            center,
            length,
            targets,
            rep_const: constants[0],
            attr_const: constants[1],
            rep_order: orders[0],
            attr_order: orders[1],
        }
    }
}

impl GeometricBias for LjCenter {
    fn name(&self) -> &str {
        "repulsive_potential_v2"
    }
    fn atoms(&self) -> Vec<AtomId> {
        let mut v = self.center.to_vec();
        v.extend(self.targets.iter().map(|t| t.0));
        v
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let [c0, c1] = self.center;
        let b = g.d(c1, c0);
        let s = T::from_f64(self.length) / norm(b);
        let mut e = T::zero();
        for &(t, eps, sigma) in &self.targets {
            let d = g.d(t, c1);
            let r = norm(std::array::from_fn(|k| d[k] - s * b[k]));
            let x = T::from_f64(sigma) / r;
            e += T::from_f64(eps)
                * (T::from_f64(self.rep_const.abs()) * x.powf(self.rep_order)
                    - T::from_f64(self.attr_const.abs()) * x.powf(self.attr_order));
        }
        e
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        History::Static
    }
}

/// LJ potential plus a Gaussian well between two fragments:
/// `eps_LJ [(s/r)^12 - 2 (s/r)^6] - eps_G exp(-(r - r_G)^2 / (0.03 b^2))` (MultiOptPy `-rpg`;
/// the factor 0.03 is MultiOptPy's `gau_range_const`).
#[derive(Debug, Clone)]
pub struct LjGaussian {
    /// Atom pairs.
    pub pairs: Vec<(AtomId, AtomId)>,
    /// LJ well depth (Hartree).
    pub lj_well: f64,
    /// LJ distance (Bohr).
    pub lj_distance: f64,
    /// Gaussian depth (Hartree).
    pub gauss_well: f64,
    /// Gaussian center (Bohr).
    pub gauss_distance: f64,
    /// Gaussian range `b` (Bohr).
    pub gauss_range: f64,
}

/// MultiOptPy's `gau_range_const`.
const GAUSS_RANGE_CONST: f64 = 0.03;

impl GeometricBias for LjGaussian {
    fn name(&self) -> &str {
        "repulsive_potential_gaussian"
    }
    fn atoms(&self) -> Vec<AtomId> {
        self.pairs.iter().flat_map(|p| [p.0, p.1]).collect()
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let w = GAUSS_RANGE_CONST * self.gauss_range * self.gauss_range;
        self.pairs.iter().fold(T::zero(), |e, &(a, b)| {
            let r = norm(g.d(a, b));
            let x = r - T::from_f64(self.gauss_distance);
            e + lj(r, self.lj_well, self.lj_distance)
                - T::from_f64(self.gauss_well) * (T::zero() - x * x / T::from_f64(w)).exp()
        })
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        History::Static
    }
}

/// Cone-shaped LJ repulsion around a center atom bonded to three atoms (ACS Catal. 2022, 12,
/// 3752; MultiOptPy `-cp`).
///
/// The apex lies 2.28 Angstrom from the center, opposite to the sum of the bonds to the three
/// atoms. A target seen at angle `theta` from the apex axis is at distance
/// `l = d sin(theta - alpha/2)` from the cone surface (`l = d` once `theta - alpha/2 > pi/2`), and
/// `E = 4 eps [(sigma / (l + sigma))^12 - (sigma / (l + sigma))^6]`.
#[derive(Debug, Clone)]
pub struct ConeRepulsive {
    /// Center atom.
    pub center: AtomId,
    /// The three atoms defining the axis.
    pub three: [AtomId; 3],
    /// `(atom, eps, sigma)` per target (`eps = sqrt(eps_v eps_t)`, `sigma = sqrt(sigma_v
    /// sigma_t)`).
    pub targets: Vec<(AtomId, f64, f64)>,
    /// Full cone angle `alpha` (rad).
    pub cone_angle: f64,
}

/// Apex distance from the center (MultiOptPy: 2.28 Angstrom).
const CONE_APEX_ANGSTROM: f64 = 2.28;

impl ConeRepulsive {
    /// Cone with MultiOptPy's parameter combination.
    pub fn new(
        elements: &[Element],
        center: AtomId,
        three: [AtomId; 3],
        targets: &[AtomId],
        well: f64,
        distance: f64,
        cone_angle: f64,
    ) -> Self {
        let targets = targets
            .iter()
            .map(|&t| {
                let et = element(elements, t);
                (
                    t,
                    (well * uff_vdw_well_depth_hartree(et)).sqrt(),
                    (distance * uff_vdw_distance_bohr(et)).sqrt(),
                )
            })
            .collect();
        Self {
            center,
            three,
            targets,
            cone_angle,
        }
    }
}

impl GeometricBias for ConeRepulsive {
    fn name(&self) -> &str {
        "cone_potential"
    }
    fn atoms(&self) -> Vec<AtomId> {
        let mut v = vec![self.center];
        v.extend(self.three);
        v.extend(self.targets.iter().map(|t| t.0));
        v
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let c = self.center;
        let mut s = [T::zero(); 3];
        for &k in &self.three {
            let d = g.d(k, c);
            s = std::array::from_fn(|i| s[i] + d[i]);
        }
        let apex = T::from_f64(CONE_APEX_ANGSTROM * Units::ANGSTROM_TO_BOHR) / norm(s);
        // Vectors from the apex: to the center (`ca`) and to a target (`ta`).
        let ca: [T; 3] = std::array::from_fn(|i| apex * s[i]);
        let half = T::from_f64(0.5 * self.cone_angle);
        let mut e = T::zero();
        for &(t, eps, sigma) in &self.targets {
            let dt = g.d(t, c);
            let ta: [T; 3] = std::array::from_fn(|i| dt[i] + ca[i]);
            let dist = norm(ta);
            let theta = angle_between(ca, ta);
            let l = if (theta - half).val() <= std::f64::consts::FRAC_PI_2 {
                dist * (theta - half).sin()
            } else {
                dist
            };
            let x = T::from_f64(sigma) / (l + T::from_f64(sigma));
            let x6 = x.powi(6);
            e += T::from_f64(4.0 * eps) * (x6 * x6 - x6);
        }
        e
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        History::Static
    }
}
