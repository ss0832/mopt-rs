//! Geometric restraints and simple external fields (MultiOptPy `-kp`, `-kpv2`, `-akp`, `-ka`,
//! `-kav2`, `-kda`, `-kdav2`, `-kdac`, `-kopa`, `-kopav2`, `-vpp`, `-up`, `-fp`, `-brp`,
//! `-lmefp`, `-lmefpv2`).
//!
//! Energies are in Hartree, lengths in Bohr, angles in radians internally. Atom groups
//! ("fragments") enter through their centroids. All terms use minimum-image geometry (see
//! [`super::engine`]).
//!
//! MultiOptPy: `Potential/keep_potential.py`, `anharmonic_keep_potential.py`,
//!  `keep_angle_potential.py`, `keep_dihedral_angle_potential.py`,
//!  `keep_outofplain_angle_potential.py`, `void_point_potential.py`, `universal_potential.py`,
//!  `flux_potential.py`, `value_range_potential.py`, `mechano_force_potential.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::engine::{norm, ramp_history, Clock, Geom, GeometricBias, Ramp};
use crate::domain::AtomId;
use crate::error::{MoptError, MoptResult};
use crate::math::ad::Scalar;
use crate::pes::term::{Frame, History};

fn dot<T: Scalar>(a: [T; 3], b: [T; 3]) -> T {
    T::dot3(a, b)
}

fn cross<T: Scalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    T::cross3(a, b)
}

fn scale<T: Scalar>(v: [T; 3], s: T) -> [T; 3] {
    v.map(|x| x * s)
}

fn flatten(groups: &[&[AtomId]]) -> Vec<AtomId> {
    groups.iter().flat_map(|g| g.iter().copied()).collect()
}

/// Harmonic distance between two atoms or two atom groups: `k/2 (r - r0)^2`
/// (MultiOptPy `-kp`, `-kpv2`).
#[derive(Debug, Clone)]
pub struct KeepDistance {
    /// Group 1 (one atom for `-kp`).
    pub a: Vec<AtomId>,
    /// Group 2.
    pub b: Vec<AtomId>,
    /// Spring constant (Hartree/Bohr^2).
    pub k: Ramp,
    /// Distance (Bohr).
    pub r0: Ramp,
}

impl GeometricBias for KeepDistance {
    fn name(&self) -> &str {
        "keep_pot"
    }
    fn atoms(&self) -> Vec<AtomId> {
        flatten(&[&self.a, &self.b])
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, clock: Clock) -> T {
        let it = clock.iteration;
        let r = norm(g.d_centroids(&self.a, &self.b));
        let d = r - T::from_f64(self.r0.at(it));
        T::from_f64(0.5 * self.k.at(it)) * d * d
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        ramp_history(&[&self.k, &self.r0])
    }
}

/// Morse-shaped distance restraint `De (1 - exp(-sqrt(k / 2De) (r - r0)))^2` (MultiOptPy `-akp`).
#[derive(Debug, Clone)]
pub struct MorseDistance {
    /// Atom pair.
    pub pair: [AtomId; 2],
    /// Well depth `De` (Hartree).
    pub depth: f64,
    /// Curvature at the minimum `k` (Hartree/Bohr^2).
    pub k: f64,
    /// Distance (Bohr).
    pub r0: f64,
}

impl MorseDistance {
    /// Check the constants: the range parameter `sqrt(k / 2De)` is real only for a positive
    /// well depth and a curvature that is not negative (otherwise every energy would be NaN).
    pub fn validate(&self) -> MoptResult<()> {
        if !(self.depth.is_finite() && self.depth > 0.0) {
            return Err(MoptError::config(format!(
                "-akp: the well depth must be positive, got {} (sqrt(k / 2 De) is not real)",
                self.depth
            )));
        }
        if !(self.k.is_finite() && self.k >= 0.0) {
            return Err(MoptError::config(format!(
                "-akp: the force constant must not be negative, got {} (sqrt(k / 2 De) is not \
                 real)",
                self.k
            )));
        }
        Ok(())
    }
}

impl GeometricBias for MorseDistance {
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        History::Static
    }
    fn name(&self) -> &str {
        "anharmonic_keep_pot"
    }
    fn atoms(&self) -> Vec<AtomId> {
        self.pair.to_vec()
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let r = norm(g.d(self.pair[0], self.pair[1]));
        let a = (self.k / (2.0 * self.depth)).sqrt();
        let x = T::one() - (T::zero() - T::from_f64(a) * (r - T::from_f64(self.r0))).exp();
        T::from_f64(self.depth) * x * x
    }
}

/// Coefficients of `arccos(1 - d)^2 = 2d + d^2/3 + 8d^3/45 + 4d^4/35 + 128d^5/1575 + ...`.
const ACOS_SQ: [f64; 5] = [128.0 / 1575.0, 4.0 / 35.0, 8.0 / 45.0, 1.0 / 3.0, 2.0];
/// Angle (rad) from 0 or pi below which the series is used.
const THETA_CUT: f64 = 1e-3;

/// `theta^2` near `theta = 0` as a series in `1 - cos(theta)`.
fn theta_sq_near_zero<T: Scalar>(u: T) -> T {
    let d = T::one() - u;
    let mut t = T::from_f64(ACOS_SQ[0]);
    for c in &ACOS_SQ[1..] {
        t = T::from_f64(*c) + d * t;
    }
    d * t
}

/// `(pi - theta)^2` near `theta = pi`.
fn theta_sq_near_pi<T: Scalar>(u: T) -> T {
    theta_sq_near_zero(T::zero() - u)
}

/// How the angle energy is continued where `arccos` is singular.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleContinuation {
    /// Series of `theta^2` (MultiOptPy `-ka`).
    Series,
    /// Quadratic extrapolation in `cos(theta)` (MultiOptPy `-kav2`).
    Quadratic,
}

/// `k/2 (theta - theta0)^2` from `u = cos(theta)`, following MultiOptPy's treatment of
/// `theta` near 0 and pi.
fn angle_energy<T: Scalar>(u: T, k: f64, theta0: f64, mode: AngleContinuation) -> T {
    let half_k = T::from_f64(0.5 * k);
    let (u_hi, u_lo) = (THETA_CUT.cos(), (PI - THETA_CUT).cos());
    let uv = u.val();
    let near_zero_target = theta0.abs() < 1e-8;
    let near_pi_target = (theta0 - PI).abs() < 1e-8;
    let quad = |th_cut: f64| {
        // E(theta(u)) expanded to second order in u around u = cos(th_cut). The d2theta/du2
        // term keeps the energy twice continuously differentiable at the cut (MultiOptPy omits
        // it, so its Hessian jumps there).
        let (s, c) = th_cut.sin_cos();
        let dth_du = -1.0 / s;
        let d2th_du2 = -c / (s * s * s);
        let val = 0.5 * k * (th_cut - theta0).powi(2);
        let d1 = k * (th_cut - theta0) * dth_du;
        let d2 = k * dth_du * dth_du + k * (th_cut - theta0) * d2th_du2;
        let du = u - T::from_f64(th_cut.cos());
        T::from_f64(val) + T::from_f64(d1) * du + T::from_f64(0.5 * d2) * du * du
    };
    let series0 = |u: T| {
        if near_zero_target {
            half_k * theta_sq_near_zero(u)
        } else {
            let th = theta_sq_near_zero(u).sqrt();
            let d = th - T::from_f64(theta0);
            half_k * d * d
        }
    };
    let series_pi = |u: T| {
        if near_pi_target {
            half_k * theta_sq_near_pi(u)
        } else {
            let th = T::from_f64(PI) - theta_sq_near_pi(u).sqrt();
            let d = th - T::from_f64(theta0);
            half_k * d * d
        }
    };
    if uv > u_hi {
        match mode {
            AngleContinuation::Quadratic if !near_zero_target => quad(THETA_CUT),
            _ => series0(u),
        }
    } else if uv < u_lo {
        match mode {
            AngleContinuation::Quadratic if !near_pi_target => quad(PI - THETA_CUT),
            _ => series_pi(u),
        }
    } else {
        let d = u.acos() - T::from_f64(theta0);
        half_k * d * d
    }
}

/// Harmonic angle between atoms or atom groups (vertex in the middle): `k/2 (theta - theta0)^2`
/// (MultiOptPy `-ka`, `-kav2`).
#[derive(Debug, Clone)]
pub struct KeepAngle {
    /// Groups 1, 2 (vertex), 3.
    pub groups: [Vec<AtomId>; 3],
    /// Spring constant (Hartree/rad^2).
    pub k: Ramp,
    /// Angle (rad).
    pub theta0: Ramp,
    /// Treatment near 0 and pi.
    pub continuation: AngleContinuation,
}

impl GeometricBias for KeepAngle {
    fn name(&self) -> &str {
        "keep_angle"
    }
    fn atoms(&self) -> Vec<AtomId> {
        flatten(&[&self.groups[0], &self.groups[1], &self.groups[2]])
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, clock: Clock) -> T {
        let it = clock.iteration;
        let v1 = g.d_centroids(&self.groups[0], &self.groups[1]);
        let v2 = g.d_centroids(&self.groups[2], &self.groups[1]);
        let n = norm(v1) * norm(v2);
        let u = dot(v1, v2) / n;
        angle_energy(u, self.k.at(it), self.theta0.at(it), self.continuation)
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        ramp_history(&[&self.k, &self.theta0])
    }
}

/// Smoothstep switch of a squared cross-product norm between 1e-10 and 1e-8 (MultiOptPy).
fn switch<T: Scalar>(x: T) -> T {
    let t = (x.val() - 1e-10) / (1e-8 - 1e-10);
    if t <= 0.0 {
        T::zero()
    } else if t >= 1.0 {
        T::one()
    } else {
        let t = (x - T::from_f64(1e-10)) / T::from_f64(1e-8 - 1e-10);
        t * t * (T::from_f64(3.0) - T::from_f64(2.0) * t)
    }
}

/// Squared norm of `b1 x b2` (Bohr^4) at and below which a dihedral is taken as undefined
/// (MultiOptPy's lower end of the collinearity switch).
pub(crate) const COLLINEAR_CROSS_SQ: f64 = 1e-10;

/// Dihedral angle (IUPAC sign) of the points `p1-p2-p3-p4` given the bond vectors, with the
/// collinearity switch factor. Where either plane is undefined (the squared norm of a cross
/// product at or below [`COLLINEAR_CROSS_SQ`]) both are zero: the restraints switch off
/// smoothly there by design (MultiOptPy).
pub(crate) fn dihedral<T: Scalar>(b1: [T; 3], b2: [T; 3], b3: [T; 3]) -> (T, T) {
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let (s1, s2) = (dot(n1, n1), dot(n2, n2));
    let sw = switch(s1) * switch(s2);
    if sw.val() == 0.0 {
        return (T::zero(), T::zero());
    }
    let n1h = scale(n1, T::one() / s1.sqrt());
    let n2h = scale(n2, T::one() / s2.sqrt());
    let b2h = scale(b2, T::one() / norm(b2));
    let x = dot(n1h, n2h);
    let y = dot(cross(n1h, n2h), b2h);
    (y.atan2(x), sw)
}

/// The dihedral angle of [`dihedral`] without the switch; `None` where it is undefined (three
/// consecutive points collinear, see [`COLLINEAR_CROSS_SQ`]).
pub(crate) fn dihedral_angle<T: Scalar>(b1: [T; 3], b2: [T; 3], b3: [T; 3]) -> Option<T> {
    let (n1, n2) = (cross(b1, b2), cross(b2, b3));
    if dot(n1, n1).val() <= COLLINEAR_CROSS_SQ || dot(n2, n2).val() <= COLLINEAR_CROSS_SQ {
        return None;
    }
    Some(dihedral(b1, b2, b3).0)
}

/// Shape of a dihedral restraint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DihedralForm {
    /// `k/2 (phi - phi0)^2` with `phi - phi0` wrapped into (-pi, pi] (MultiOptPy `-kda`,
    /// `-kdav2`).
    Harmonic,
    /// `k/2 (1 - cos(n phi - phi0))` (MultiOptPy `-kdac`, multiplicity `n`).
    Cosine {
        /// Multiplicity `n`.
        n: f64,
    },
}

/// Dihedral restraint between atoms or atom groups (MultiOptPy `-kda`, `-kdav2`, `-kdac`).
#[derive(Debug, Clone)]
pub struct KeepDihedral {
    /// Groups 1-4.
    pub groups: [Vec<AtomId>; 4],
    /// Force constant (Hartree/rad^2 for the harmonic form, Hartree for the cosine form).
    pub k: Ramp,
    /// Reference angle (rad).
    pub phi0: Ramp,
    /// Functional form.
    pub form: DihedralForm,
}

impl GeometricBias for KeepDihedral {
    fn name(&self) -> &str {
        "keep_dihedral_angle"
    }
    fn atoms(&self) -> Vec<AtomId> {
        flatten(&[
            &self.groups[0],
            &self.groups[1],
            &self.groups[2],
            &self.groups[3],
        ])
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, clock: Clock) -> T {
        let it = clock.iteration;
        let [a, b, c, d] = &self.groups;
        let b1 = g.d_centroids(b, a);
        let b2 = g.d_centroids(c, b);
        let b3 = g.d_centroids(d, c);
        let (phi, sw) = dihedral(b1, b2, b3);
        let (k, phi0) = (self.k.at(it), self.phi0.at(it));
        let raw = match self.form {
            DihedralForm::Harmonic => {
                let diff = phi.val() - phi0;
                let wrap = 2.0 * PI * (diff / (2.0 * PI)).round();
                let d = phi - T::from_f64(phi0 + wrap);
                T::from_f64(0.5 * k) * d * d
            }
            DihedralForm::Cosine { n } => {
                T::from_f64(0.5 * k) * (T::one() - (T::from_f64(n) * phi - T::from_f64(phi0)).cos())
            }
        };
        raw * sw
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        ramp_history(&[&self.k, &self.phi0])
    }
}

/// Angle of `a1` above the plane spanned by `a2` and `a3` (all from the center); `None` where
/// the plane is undefined (MultiOptPy's `|a2 x a3|^2 < 1e-8`).
pub(crate) fn out_of_plane<T: Scalar>(a1: [T; 3], a2: [T; 3], a3: [T; 3]) -> Option<T> {
    let n = cross(a2, a3);
    let nn = dot(n, n);
    if nn.val() < 1e-8 {
        return None;
    }
    let h = dot(a1, n) / nn.sqrt();
    let rp2 = dot(a1, a1) - h * h;
    let rp = if rp2.val() > 0.0 {
        rp2.sqrt()
    } else {
        T::zero()
    };
    Some(h.atan2(rp))
}

/// Out-of-plane angle of group 2 from the plane through group 1 (center) spanned by groups 3 and
/// 4: `k/2 (chi - chi0)^2` (MultiOptPy `-kopa`, `-kopav2`); zero where the plane is undefined.
#[derive(Debug, Clone)]
pub struct KeepOutOfPlane {
    /// Groups: center, out-of-plane, plane 1, plane 2.
    pub groups: [Vec<AtomId>; 4],
    /// Spring constant (Hartree/rad^2).
    pub k: Ramp,
    /// Angle (rad).
    pub chi0: Ramp,
}

impl GeometricBias for KeepOutOfPlane {
    fn name(&self) -> &str {
        "keep_out_of_plain_angle"
    }
    fn atoms(&self) -> Vec<AtomId> {
        flatten(&[
            &self.groups[0],
            &self.groups[1],
            &self.groups[2],
            &self.groups[3],
        ])
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, clock: Clock) -> T {
        let it = clock.iteration;
        let [c, i1, i2, i3] = &self.groups;
        let a1 = g.d_centroids(i1, c);
        let a2 = g.d_centroids(i2, c);
        let a3 = g.d_centroids(i3, c);
        let Some(chi) = out_of_plane(a1, a2, a3) else {
            return T::zero();
        };
        let d = chi - T::from_f64(self.chi0.at(it));
        T::from_f64(0.5 * self.k.at(it)) * d * d
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        ramp_history(&[&self.k, &self.chi0])
    }
}

/// `(1/p) k (|r - P| - r0)^p` for one atom and a fixed point `P` (MultiOptPy `-vpp`).
#[derive(Debug, Clone)]
pub struct VoidPoint {
    /// Atom.
    pub atom: AtomId,
    /// Point `P` (Bohr).
    pub point: [f64; 3],
    /// Constant `k`.
    pub k: f64,
    /// Distance `r0` (Bohr).
    pub r0: f64,
    /// Order `p`.
    pub order: f64,
}

fn power<T: Scalar>(x: T, p: f64) -> T {
    if p.fract() == 0.0 && p.abs() <= i32::MAX as f64 {
        x.powi(p as i32)
    } else {
        x.powf(p)
    }
}

/// Check an exponent applied to a quantity that can be negative: it must be a nonzero
/// integer (`opt` names the option, `what` the quantity).
fn check_signed_power(opt: &str, what: &str, p: f64) -> MoptResult<()> {
    if !p.is_finite() || p == 0.0 || p.fract() != 0.0 || p.abs() > i32::MAX as f64 {
        return Err(MoptError::config(format!(
            "{opt}: the order must be a nonzero integer, got {p} ({what} can be negative, and \
             a negative number has no real non-integer power)"
        )));
    }
    Ok(())
}

impl VoidPoint {
    /// Check the order: `(|r - P| - r0)^p` is negative inside the sphere `|r - P| < r0`,
    /// so `p` must be an integer, and `k / p` needs `p != 0`.
    pub fn validate(&self) -> MoptResult<()> {
        check_signed_power("-vpp", "|r - P| - r0", self.order)
    }
}

impl GeometricBias for VoidPoint {
    fn name(&self) -> &str {
        "void_point_pot"
    }
    fn atoms(&self) -> Vec<AtomId> {
        vec![self.atom]
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let r = norm(g.d_point(self.atom, self.point));
        T::from_f64(self.k / self.order) * power(r - T::from_f64(self.r0), self.order)
    }
    fn frame(&self) -> Frame {
        Frame::Absolute
    }
    fn history(&self) -> History {
        History::Static
    }
}

/// Gathers atoms toward their centroid: `c / N_pairs sum_i |r_i - centroid|` (MultiOptPy `-up`).
#[derive(Debug, Clone)]
pub struct Universal {
    /// Target atoms.
    pub atoms: Vec<AtomId>,
    /// Constant `c` (Hartree; given in kJ/mol on the command line).
    pub c: f64,
}

impl GeometricBias for Universal {
    fn name(&self) -> &str {
        "universal_pot"
    }
    fn atoms(&self) -> Vec<AtomId> {
        self.atoms.clone()
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let n = self.atoms.len();
        let pairs = (n * (n.saturating_sub(1)) / 2).max(1) as f64;
        let c = g.centroid(&self.atoms);
        let p0 = g.pos(self.atoms[0]);
        let mut e = T::zero();
        for &a in &self.atoms {
            // r_a - centroid through the image of a nearest to the first atom.
            let d0 = g.d(a, self.atoms[0]);
            let v: [T; 3] = std::array::from_fn(|k| p0[k] + d0[k] - c[k]);
            e += norm(v);
        }
        T::from_f64(self.c / pairs) * e
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        History::Static
    }
}

/// `sum_i sum_c k_c (x_ic - x0_c)^p_c` (MultiOptPy `-fp`).
#[derive(Debug, Clone)]
pub struct Flux {
    /// Target atoms.
    pub atoms: Vec<AtomId>,
    /// Constants per Cartesian component.
    pub k: [f64; 3],
    /// Orders per component.
    pub order: [f64; 3],
    /// Reference point (Bohr).
    pub point: [f64; 3],
}

impl GeometricBias for Flux {
    fn name(&self) -> &str {
        "flux_pot"
    }
    fn atoms(&self) -> Vec<AtomId> {
        self.atoms.clone()
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let mut e = T::zero();
        for &a in &self.atoms {
            let d = g.d_point(a, self.point);
            for ((dc, k), p) in d.iter().zip(self.k).zip(self.order) {
                e += T::from_f64(k) * power(*dc, p);
            }
        }
        e
    }
    fn frame(&self) -> Frame {
        Frame::Absolute
    }
    fn history(&self) -> History {
        History::Static
    }
}

impl Flux {
    /// Check the orders: `x_c - x0_c` takes both signs, so the order of every component with
    /// a nonzero constant must be an integer (zero is allowed here: `k x^0` is a constant).
    pub fn validate(&self) -> MoptResult<()> {
        for (k, p) in self.k.iter().zip(self.order) {
            if *k != 0.0 && p != 0.0 {
                check_signed_power("-fp", "x_c - x0_c", p)?;
            }
        }
        Ok(())
    }
}

/// Soft confinement of a group distance to `[lower, upper]`:
/// `ln[(1 + exp(cu (d - du))) (1 + exp(cl (dl - d)))]` (MultiOptPy `-brp`).
#[derive(Debug, Clone)]
pub struct BondRange {
    /// Group 1.
    pub a: Vec<AtomId>,
    /// Group 2.
    pub b: Vec<AtomId>,
    /// Upper steepness (1/Bohr).
    pub upper_const: f64,
    /// Lower steepness (1/Bohr).
    pub lower_const: f64,
    /// Upper distance (Bohr).
    pub upper: f64,
    /// Lower distance (Bohr).
    pub lower: f64,
}

impl GeometricBias for BondRange {
    fn name(&self) -> &str {
        "value_range_pot"
    }
    fn atoms(&self) -> Vec<AtomId> {
        flatten(&[&self.a, &self.b])
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let d = norm(g.d_centroids(&self.a, &self.b));
        // ln[(1 + e^u)(1 + e^l)] = softplus(u) + softplus(l), which does not overflow far outside
        // the range.
        let u = T::from_f64(self.upper_const) * (d - T::from_f64(self.upper));
        let l = T::from_f64(self.lower_const) * (T::from_f64(self.lower) - d);
        crate::math::stable::softplus(u) + crate::math::stable::softplus(l)
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        History::Static
    }
}

/// Constant pulling force `F` along atom pairs (external force is explicitly included, EFEI):
/// `-(F / n) sum_k |r_b,k - r_a,k|` over `n` pairs (MultiOptPy `-lmefp` with two pairs,
/// `-lmefpv2` with one).
#[derive(Debug, Clone)]
pub struct PullingForce {
    /// Atom pairs `(a, b)`.
    pub pairs: Vec<[AtomId; 2]>,
    /// Force (Hartree/Bohr); positive pulls the atoms of each pair apart.
    pub force: f64,
}

impl GeometricBias for PullingForce {
    fn name(&self) -> &str {
        "linear_mechano_force_pot"
    }
    fn atoms(&self) -> Vec<AtomId> {
        self.pairs.iter().flatten().copied().collect()
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let mut e = T::zero();
        for [a, b] in &self.pairs {
            e += norm(g.d(*b, *a));
        }
        T::from_f64(-self.force / self.pairs.len().max(1) as f64) * e
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        History::Static
    }
}
