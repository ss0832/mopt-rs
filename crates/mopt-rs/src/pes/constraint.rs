//! Geometric constraints of an optimization (MultiOptPy `-fix`, `-pc`, `-gfix`).
//!
//! A [`ConstraintSet`] holds equality constraints `c_k(x) = t_k` on the Cartesian coordinates:
//! distances between atoms or fragment centroids, angles, dihedrals, single Cartesian
//! components, frozen atoms, the orientation of a group of atoms (Eckart conditions with respect
//! to its starting structure), and fixed directions (a Hessian eigenvector or the relative motion
//! of an atom pair at the start). The optimizer
//!
//! 1. projects the constraint gradients `J` out of the gradient and Hessian in its own
//!    coordinates and adds the curvature of the constraints, `-sum_k lambda_k d2c_k`,
//!    `lambda = (J J^T)^+ J g` (the Hessian of the Lagrangian; MultiOptPy leaves it out), and
//! 2. after each step restores all constraints together by Newton projection,
//!    `x <- x - J^T (J J^T)^+ (c(x) - t)`, preceded by a rigid rotation of each oriented group
//!    back to its starting orientation.
//!
//! Distances, angles, and dihedrals use minimum images in periodic cells. Angles are taken as
//! `atan2(|a x b|, a . b)`, accurate up to 0 and 180 degrees; a dihedral constraint at a
//! structure where three consecutive atoms are collinear is an error naming the atoms (the
//! dihedral restraints instead switch off smoothly there, as in MultiOptPy).
//!
//! A [`Constraint`] that holds the value "at the start" is resolved once at a reference
//! structure ([`Constraint::resolve`], idempotent): its target, the positions of frozen atoms
//! and oriented groups, and the lab-frame direction of `atoms_pair` and eigenvector
//! constraints are fixed there and carried, so that every later stage of a run (follow-up
//! optimizations, conformer candidates, sampled starts) holds the same constraint instead of
//! the values of its own starting structure.
//!
//! MultiOptPy: `Constraint/constraint_condition.py:ProjectOutConstrain` and
//!  `optimization.py:ConstraintManager`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;
use std::sync::Arc;

use faer::{Col, Mat};

use crate::coord::alignment::proper_rotation_from_covariance;
use crate::domain::{AtomId, MolecularSystem, Position};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::math::ad::{angle_between, Scalar};
use crate::math::linalg::symmetric_eigen;
use crate::pes::bias::engine::norm;
use crate::pes::bias::restraints::dihedral_angle;
use crate::pes::bias::{AdBias, Clock, Geom, GeometricBias};
use crate::pes::term::{Frame, History};
use crate::pes::{PesTerm, TermContext};

/// Whether `constraints` pin the structure in space (frozen atoms, Cartesian components,
/// oriented groups, and the fixed lab-frame directions of `atoms_pair` and eigenvector
/// constraints: a rigid rotation changes `v . x` for a fixed `v`).
pub fn breaks_rigid_body(constraints: &[Constraint]) -> bool {
    constraints.iter().any(|c| kind_breaks_rigid_body(&c.kind))
}

fn kind_breaks_rigid_body(kind: &ConstraintKind) -> bool {
    match kind {
        ConstraintKind::Frozen(atoms) => !atoms.is_empty(),
        ConstraintKind::Coordinate(..)
        | ConstraintKind::Orientation(_)
        | ConstraintKind::EigenVector(_)
        | ConstraintKind::AtomPair(_) => true,
        ConstraintKind::Function(f) => f.fixes_frame,
        ConstraintKind::Distance(..) | ConstraintKind::Angle(_) | ConstraintKind::Dihedral(_) => {
            false
        }
    }
}

/// A user-defined constraint: the energy of any surface term is the constrained value
/// `c(x)`, with its gradient and Hessian (expressions, Python callbacks, or Rust terms).
#[derive(Clone)]
pub struct ConstraintFunction {
    /// The function.
    pub term: Arc<dyn PesTerm>,
    /// Period of the value (e.g. `2 pi` for an angle-like function); residuals are wrapped
    /// into half a period.
    pub period: Option<f64>,
    /// Whether the function pins the structure in space (depends on absolute positions or
    /// orientation), so that rigid-body motion must not be removed.
    pub fixes_frame: bool,
    /// The largest residual the function is held to in place of [`GUARANTEE`] (`-uc "...;
    /// tolerance = T"`: a function whose evaluation is noisier than the guarantee allows);
    /// `None`: the guarantee. A restore that cannot reach it is an error naming the function.
    pub tolerance: Option<f64>,
    /// Whether a residual above the tolerance is a warning instead of an error (the ReSPer
    /// coordinates, computed through a superposition with a noise of about 1e-9).
    pub warns_only: bool,
}

impl ConstraintFunction {
    /// A function of internal geometry only (no period), held to [`GUARANTEE`].
    pub fn new(term: Arc<dyn PesTerm>) -> Self {
        Self {
            term,
            period: None,
            fixes_frame: false,
            tolerance: None,
            warns_only: false,
        }
    }

    /// The same function with a residual above its tolerance a warning, not an error.
    #[must_use]
    pub fn warning_only(mut self) -> Self {
        self.warns_only = true;
        self
    }

    /// The residual the function is held to: its tolerance, or [`GUARANTEE`].
    pub fn held_to(&self) -> f64 {
        self.tolerance.unwrap_or(GUARANTEE)
    }
}

impl std::fmt::Debug for ConstraintFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintFunction")
            .field("term", &self.term.name())
            .field("period", &self.period)
            .field("fixes_frame", &self.fixes_frame)
            .field("tolerance", &self.tolerance)
            .field("warns_only", &self.warns_only)
            .finish()
    }
}

impl PartialEq for ConstraintFunction {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.term, &other.term)
            && self.period == other.period
            && self.fixes_frame == other.fixes_frame
            && self.tolerance == other.tolerance
            && self.warns_only == other.warns_only
    }
}

/// What a constraint holds.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    /// Distance between two atoms or fragment centroids (`bond`, `fbond`).
    Distance(Vec<AtomId>, Vec<AtomId>),
    /// Angle `a-b-c` at `b` (`angle`).
    Angle([AtomId; 3]),
    /// Dihedral `a-b-c-d`, IUPAC sign (`dihedral`).
    Dihedral([AtomId; 4]),
    /// One Cartesian component of an atom (`x`, `y`, `z`; axis 0, 1, 2).
    Coordinate(AtomId, usize),
    /// All coordinates of the atoms (`-fix`).
    Frozen(Vec<AtomId>),
    /// Orientation of a group of atoms (`rot`).
    Orientation(Vec<AtomId>),
    /// No motion along the `n`-th Hessian eigenvector at the start, counting from the lowest
    /// eigenvalue and skipping zero ones (`eigvec`).
    EigenVector(usize),
    /// No relative motion of an atom pair along their connecting line at the start
    /// (`atoms_pair`).
    AtomPair([AtomId; 2]),
    /// A user-defined function of the structure (`-uc`, Python callbacks, Rust terms).
    Function(ConstraintFunction),
}

/// What a constraint is held to besides its target value, fixed when the constraint is
/// resolved at a reference structure ([`Constraint::resolve`]) and carried from then on, so
/// that every stage of a run holds the same constraint. Everything is stored per atom, so a
/// resolved constraint can be renumbered onto a subsystem (the core of ONIOM
/// microiterations) by renumbering its atoms.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintAnchor {
    /// Reference positions (Bohr) of the atoms of a [`ConstraintKind::Frozen`] or
    /// [`ConstraintKind::Orientation`] constraint, one per atom in the order of the kind's
    /// atoms: frozen atoms return to them, an oriented group keeps their orientation.
    Positions(Vec<[f64; 3]>),
    /// The lab-frame direction `v` of the linear constraint `v . x = target` of an
    /// [`ConstraintKind::AtomPair`] or [`ConstraintKind::EigenVector`] constraint, as the
    /// components of `v` on every atom where it is not zero; the target is `v . x` at the
    /// reference.
    Direction(Vec<(AtomId, [f64; 3])>),
}

/// One constraint.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    /// What is held.
    pub kind: ConstraintKind,
    /// Value to hold (Bohr or rad, or the units of a user-defined function, or `v . x` of a
    /// fixed direction); `None`: the value at the structure where the constraint is resolved.
    /// Used by distances, angles, dihedrals, coordinates, user functions, and directions.
    pub target: Option<f64>,
    /// Positions or direction fixed at the reference structure (`None`: not resolved yet).
    /// Used by frozen atoms, oriented groups, and directions.
    pub anchor: Option<ConstraintAnchor>,
}

impl Constraint {
    /// Hold `kind` at its value where the constraint is resolved.
    pub fn at_start(kind: ConstraintKind) -> Self {
        Self {
            kind,
            target: None,
            anchor: None,
        }
    }

    /// Hold `kind` at `target`.
    pub fn with_target(kind: ConstraintKind, target: f64) -> Self {
        Self {
            kind,
            target: Some(target),
            anchor: None,
        }
    }

    /// Whether the constraint holds fixed values already (an explicit target, a target and
    /// anchor from an earlier [`Self::resolve`]), so that resolving it again changes nothing.
    pub fn is_resolved(&self) -> bool {
        match &self.kind {
            ConstraintKind::Distance(..)
            | ConstraintKind::Angle(_)
            | ConstraintKind::Dihedral(_)
            | ConstraintKind::Coordinate(..)
            | ConstraintKind::Function(_) => self.target.is_some(),
            ConstraintKind::Frozen(_) | ConstraintKind::Orientation(_) => {
                matches!(self.anchor, Some(ConstraintAnchor::Positions(_)))
            }
            ConstraintKind::EigenVector(_) | ConstraintKind::AtomPair(_) => {
                self.target.is_some() && matches!(self.anchor, Some(ConstraintAnchor::Direction(_)))
            }
        }
    }

    /// Whether the constraint cannot be resolved from a structure alone and waits for a
    /// Hessian (an eigenvector constraint that has not been resolved).
    pub fn is_pending(&self) -> bool {
        matches!(self.kind, ConstraintKind::EigenVector(_)) && !self.is_resolved()
    }

    /// The constraint with every value fixed at `reference`: targets of distances, angles,
    /// dihedrals, coordinates, and functions, positions of frozen atoms and oriented groups,
    /// and the lab-frame direction of `atoms_pair` constraints and of eigenvector constraints
    /// (from `hessian`, Cartesian, at `reference`). Idempotent: a resolved constraint
    /// ([`Self::is_resolved`]: an explicit `-pc manual` value, a relaxed-scan target, a
    /// previous resolve) comes back unchanged. An eigenvector constraint without a Hessian
    /// stays pending ([`Self::is_pending`]).
    pub fn resolve(
        &self,
        reference: &MolecularSystem,
        hessian: Option<&Mat<f64>>,
    ) -> MoptResult<Constraint> {
        if self.is_resolved() {
            return Ok(self.clone());
        }
        for a in c_atoms(&self.kind) {
            a.check(reference.n_atoms())?;
        }
        let x0 = reference.coords_flat_bohr();
        let mut out = self.clone();
        match &self.kind {
            ConstraintKind::Distance(a, b) => {
                out.target = Some(geo_value(GeoFn::Distance(a.clone(), b.clone()), reference)?)
            }
            ConstraintKind::Angle(a) => out.target = Some(geo_value(GeoFn::Angle(*a), reference)?),
            ConstraintKind::Dihedral(a) => {
                out.target = Some(geo_value(GeoFn::Dihedral(*a), reference)?)
            }
            ConstraintKind::Function(f) => out.target = Some(value_of(f.term.as_ref(), reference)?),
            ConstraintKind::Coordinate(atom, axis) => out.target = Some(x0[atom.dof(*axis)]),
            ConstraintKind::Frozen(atoms) | ConstraintKind::Orientation(atoms) => {
                out.anchor = Some(ConstraintAnchor::Positions(
                    atoms
                        .iter()
                        .map(|&a| reference.position(a).value())
                        .collect(),
                ))
            }
            ConstraintKind::AtomPair([a, b]) => {
                let (pa, pb) = (
                    reference.position(*a).value(),
                    reference.position(*b).value(),
                );
                let d: [f64; 3] = std::array::from_fn(|k| pb[k] - pa[k]);
                let nv = (2.0 * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).sqrt();
                if nv < 1e-10 {
                    return Err(MoptError::config(
                        "atoms_pair constraint: the atoms coincide",
                    ));
                }
                let u = d.map(|x| x / nv);
                let direction = vec![(*a, u), (*b, u.map(|x| -x))];
                out.target = Some(direction_dot(&direction, reference));
                out.anchor = Some(ConstraintAnchor::Direction(direction));
            }
            ConstraintKind::EigenVector(index) => {
                let Some(h) = hessian else {
                    return Ok(out);
                };
                let n = reference.dof();
                if h.nrows() != n || h.ncols() != n {
                    return Err(MoptError::config(format!(
                        "eigenvector constraint {index}: the Hessian is {}x{}, the structure has \
                         {n} coordinates",
                        h.nrows(),
                        h.ncols()
                    )));
                }
                let (w, vecs) = symmetric_eigen(h)?;
                let modes: Vec<usize> = (0..w.nrows()).filter(|&k| w[k].abs() > 1e-10).collect();
                let k = *modes.get(*index).ok_or_else(|| {
                    MoptError::config(format!(
                        "eigenvector constraint {index}: only {} nonzero modes",
                        modes.len()
                    ))
                })?;
                let direction: Vec<(AtomId, [f64; 3])> = reference
                    .atoms()
                    .map(|a| (a, std::array::from_fn(|c| vecs[(a.dof(c), k)])))
                    .filter(|(_, v): &(AtomId, [f64; 3])| v.iter().any(|x| *x != 0.0))
                    .collect();
                out.target = Some(direction_dot(&direction, reference));
                out.anchor = Some(ConstraintAnchor::Direction(direction));
            }
        }
        if let Some(t) = out.target {
            if !t.is_finite() {
                return Err(MoptError::config(format!(
                    "constraint {}: the value at the reference structure is not finite",
                    describe(&self.kind)
                )));
            }
        }
        Ok(out)
    }
}

/// `v . x` of a direction stored per atom.
fn direction_dot(direction: &[(AtomId, [f64; 3])], sys: &MolecularSystem) -> f64 {
    direction
        .iter()
        .map(|(a, v)| {
            let p = sys.position(*a).value();
            v[0] * p[0] + v[1] * p[1] + v[2] * p[2]
        })
        .sum()
}

/// The constraint in messages (atoms 1-based).
fn describe(kind: &ConstraintKind) -> String {
    let list = |a: &[AtomId]| {
        a.iter()
            .map(|x| x.one_based().to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    match kind {
        ConstraintKind::Distance(a, b) => format!("distance {} / {}", list(a), list(b)),
        ConstraintKind::Angle(a) => format!("angle {}", list(a)),
        ConstraintKind::Dihedral(a) => format!("dihedral {}", list(a)),
        ConstraintKind::Coordinate(a, k) => format!("coordinate {} of atom {}", k, a.one_based()),
        ConstraintKind::Frozen(a) => format!("frozen atoms {}", list(a)),
        ConstraintKind::Orientation(a) => format!("orientation of {}", list(a)),
        ConstraintKind::EigenVector(k) => format!("eigenvector {k}"),
        ConstraintKind::AtomPair(a) => format!("atoms_pair {}", list(a)),
        ConstraintKind::Function(f) => format!("'{}'", f.term.name()),
    }
}

/// A scalar geometric function.
#[derive(Debug, Clone, PartialEq)]
enum GeoFn {
    Distance(Vec<AtomId>, Vec<AtomId>),
    Angle([AtomId; 3]),
    Dihedral([AtomId; 4]),
}

impl GeometricBias for GeoFn {
    fn name(&self) -> &str {
        "constraint"
    }
    fn atoms(&self) -> Vec<AtomId> {
        match self {
            Self::Distance(a, b) => a.iter().chain(b).copied().collect(),
            Self::Angle(a) => a.to_vec(),
            Self::Dihedral(a) => a.to_vec(),
        }
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        match self {
            Self::Distance(a, b) => norm(g.d_centroids(a, b)),
            Self::Angle([a, b, c]) => angle_between(g.d(*a, *b), g.d(*c, *b)),
            // Undefined where three consecutive atoms are collinear: NaN, reported with the
            // atoms by `diagnose` (holding an undefined dihedral at a value is meaningless).
            Self::Dihedral([a, b, c, d]) => dihedral_angle(g.d(*b, *a), g.d(*c, *b), g.d(*d, *c))
                .unwrap_or(T::from_f64(f64::NAN)),
        }
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    fn history(&self) -> History {
        History::Static
    }
    fn diagnose(&self, g: &Geom<'_, f64>, _clock: Clock) -> Option<String> {
        match self {
            Self::Dihedral(at @ [a, b, c, d]) => {
                dihedral_angle(g.d(*b, *a), g.d(*c, *b), g.d(*d, *c))
                    .is_none()
                    .then(|| {
                        format!(
                            "dihedral constraint {}-{}-{}-{}: three consecutive atoms are \
                             collinear, so the dihedral is undefined",
                            at[0].one_based(),
                            at[1].one_based(),
                            at[2].one_based(),
                            at[3].one_based()
                        )
                    })
            }
            _ => None,
        }
    }
}

/// Value of a geometric function at `sys`.
fn geo_value(f: GeoFn, sys: &MolecularSystem) -> MoptResult<f64> {
    value_of(&AdBias::new(f), sys)
}

/// A resolved scalar constraint `c(x) = target`.
#[derive(Clone)]
enum Row {
    Geometric {
        f: Arc<dyn PesTerm>,
        target: f64,
        period: Option<f64>,
        /// The constraint in messages.
        label: String,
        /// Repeatability of the value measured at the start (`evaluation_noise`; zero for
        /// the built-in distances, angles, and dihedrals, whose floor is that of rounding).
        noise: f64,
        /// The largest residual it is held to ([`GUARANTEE`] unless the function gives its
        /// own tolerance).
        tolerance: f64,
        /// A residual above `tolerance` is a warning, not an error (ReSPer coordinates).
        lenient: bool,
    },
    /// `v . x = target`.
    Linear {
        v: Col<f64>,
        target: f64,
        label: String,
    },
    /// One Cartesian coordinate `x_dof = target` (frozen atoms, `-pc x|y|z`): assigned, never
    /// solved for, so that it is bit-identical to its target.
    Pinned { dof: usize, target: f64 },
}

impl Row {
    fn target(&self) -> f64 {
        match self {
            Self::Geometric { target, .. }
            | Self::Linear { target, .. }
            | Self::Pinned { target, .. } => *target,
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Geometric { label, .. } | Self::Linear { label, .. } => label.clone(),
            Self::Pinned { dof, .. } => {
                format!("coordinate {} of atom {}", dof % 3, dof / 3 + 1)
            }
        }
    }

    /// The largest residual the row is held to.
    fn tolerance(&self) -> f64 {
        match self {
            Self::Geometric { tolerance, .. } => *tolerance,
            Self::Linear { .. } | Self::Pinned { .. } => GUARANTEE,
        }
    }

    /// Whether a residual above its tolerance is a warning instead of an error.
    fn lenient(&self) -> bool {
        matches!(self, Self::Geometric { lenient: true, .. })
    }
}

impl std::fmt::Debug for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometric {
                f: t,
                target,
                period,
                noise,
                tolerance,
                lenient,
                ..
            } => f
                .debug_struct("Geometric")
                .field("term", &t.name())
                .field("target", target)
                .field("period", period)
                .field("noise", noise)
                .field("tolerance", tolerance)
                .field("lenient", lenient)
                .finish(),
            Self::Linear { target, label, .. } => f
                .debug_struct("Linear")
                .field("label", label)
                .field("target", target)
                .finish(),
            Self::Pinned { dof, target } => f
                .debug_struct("Pinned")
                .field("dof", dof)
                .field("target", target)
                .finish(),
        }
    }
}

/// Units in the last place a restored constraint value may be off by, relative to the
/// condition of its evaluation (see `rounding_floor`).
const FLOOR_ULPS: f64 = 4.0;

/// The rounding floor of a constraint value `c(x)` with the gradient `j` at `x`: the smallest
/// residual that Newton steps can be asked to reach in double precision.
///
/// Derivation. Every coordinate `x_i` a Newton step writes is rounded to the nearest double,
/// an error of at most `eps/2 |x_i|` (`eps` the machine epsilon), which changes the value by
/// `eps/2 sum_i |dc/dx_i| |x_i|` to first order. Evaluating `c` at the rounded coordinates
/// adds the rounding of its own arithmetic: the differences `x_a - x_b` of a distance, an
/// angle, or a dihedral carry `eps |x_a|` each (again `eps sum_i |dc/dx_i| |x_i|` through the
/// gradient), and the final operations (square root, `atan2`, the subtraction of the target)
/// `eps |c|`. Hence `floor = FLOOR_ULPS eps (sum_i |dc/dx_i| |x_i| + |c|)`, with a few units of
/// margin for the accumulation of these terms. Per kind, with coordinates of magnitude `X` and
/// bond lengths `d`: a distance `|dc/dx| = 1` per atom, so `floor ~ 4 eps (2 sqrt(3) X + d)`
/// (`~1e-13` Bohr for `X = 10`); an angle or a dihedral `|dc/dx| ~ 1/d`, so `floor ~ 4 eps (k
/// X / d + pi)`, a few `1e-14` rad; a centroid distance spreads `1/n` over the `n` atoms of each
/// fragment, as a distance; a Cartesian component or a frozen atom is assigned and exact; a
/// linear row `v . x` (Eckart conditions of an oriented group, fixed directions) has the same
/// bound with `|v_i|` for the gradient.
///
/// MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy stops at a fixed tolerance).
pub(crate) fn rounding_floor(j: impl Iterator<Item = (f64, f64)>, value: f64) -> f64 {
    let scale: f64 = j.map(|(d, x)| (d * x).abs()).sum();
    FLOOR_ULPS * f64::EPSILON * (scale + value.abs()).max(f64::MIN_POSITIVE)
}

/// Step of the displacements that measure the repeatability of a function (Bohr, or the unit
/// of the variables): small enough that the second-order change, `h^2/2 |d2c|`, stays below
/// `1e-17` for curvatures up to `1e1`, and large enough to be represented exactly against
/// coordinates of order `1e3`.
pub(crate) const NOISE_STEP: f64 = 1e-9;

/// The repeatability of `value_and_gradient` at `x`: the largest deviation of the value at
/// `x + y` from its first-order prediction `c(x) + g(x) . y`, over six displacements of
/// length `NOISE_STEP` along fixed directions. For a smooth function evaluated to machine
/// precision this is the rounding floor; for a function computed through an inner problem (a
/// superposition, as the ReSPer coordinates) it is the noise of that problem, below which
/// Newton steps cannot restore the function.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub(crate) fn evaluation_noise(
    value_and_gradient: impl Fn(&Col<f64>) -> MoptResult<(f64, Col<f64>)>,
    x: &Col<f64>,
) -> MoptResult<f64> {
    let n = x.nrows();
    let (v0, g0) = value_and_gradient(x)?;
    if g0.nrows() != n {
        return Ok(0.0);
    }
    let mut noise = 0.0_f64;
    for k in 0..3 {
        let u = Col::from_fn(n, |i| {
            ((i as f64 + 1.0) * (0.618_033_988_75 + 0.371 * k as f64)).sin()
        });
        let length = u.norm_l2().max(f64::MIN_POSITIVE);
        for sign in [1.0, -1.0] {
            let y = Col::from_fn(n, |i| x[i] + sign * NOISE_STEP * u[i] / length);
            // The displacement as it was represented, not as it was asked for.
            let linear: f64 = (0..n).map(|i| g0[i] * (y[i] - x[i])).sum();
            let v = value_and_gradient(&y)?.0;
            noise = noise.max((v - v0 - linear).abs());
        }
    }
    Ok(noise)
}

/// Constraint gradients, multipliers, and curvature at one structure.
#[derive(Debug, Clone)]
pub struct ConstraintFrame {
    /// Constraint gradients (rows) in Cartesian coordinates.
    pub jacobian: Mat<f64>,
    /// Lagrange multipliers `(J J^T)^+ J g` of the gradient `g`.
    pub multipliers: Col<f64>,
    /// `sum_k lambda_k d2c_k` (Cartesian).
    pub curvature: Mat<f64>,
}

impl ConstraintFrame {
    /// Gradient with the constraint directions removed, `g - J^T lambda`.
    pub fn project_gradient(&self, g: &Col<f64>) -> Col<f64> {
        g - self.jacobian.transpose() * &self.multipliers
    }
}

/// The constraints of a run, resolved against its starting structure.
#[derive(Debug, Clone)]
pub struct ConstraintSet {
    rows: Vec<Row>,
    /// Oriented groups and their starting positions (centered).
    oriented: Vec<(Vec<AtomId>, Vec<[f64; 3]>)>,
    breaks_rigid_body: bool,
    /// One line per user-defined function whose value repeats only above the rounding floor.
    notes: Vec<String>,
    /// ReSPer coordinates left above [`GUARANTEE`]: (label, largest residual, warned).
    lenient: Arc<std::sync::Mutex<Vec<(String, f64, bool)>>>,
}

/// `J W`: the rows of `j` scaled by the weights `w` of the variables (unchanged without).
fn weighted(j: &Mat<f64>, w: Option<&[f64]>) -> Mat<f64> {
    match w {
        Some(w) => Mat::from_fn(j.nrows(), j.ncols(), |k, i| j[(k, i)] * w[i]),
        None => j.clone(),
    }
}

/// Pseudo-inverse of `J W J^T` (`W` the diagonal of the weights `w`; `J J^T` without).
fn gram_pinv_in(j: &Mat<f64>, w: Option<&[f64]>) -> MoptResult<Mat<f64>> {
    let g = weighted(j, w) * j.transpose();
    Ok(crate::math::linalg::psd_pseudo_inverse(&g, 1e-12)?.0)
}

fn centroid(sys: &MolecularSystem, atoms: &[AtomId]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for &a in atoms {
        let p = sys.position(a).value();
        for k in 0..3 {
            c[k] += p[k] / atoms.len() as f64;
        }
    }
    c
}

impl ConstraintSet {
    /// The constraints at `start`: each one holds its target and anchor where it has them
    /// ([`Constraint::resolve`] gave them, or an explicit value), and is otherwise resolved at
    /// `start`. `hessian` (Cartesian, at `start`) is needed for eigenvector constraints that
    /// have not been resolved.
    pub fn new(
        constraints: &[Constraint],
        start: &MolecularSystem,
        hessian: Option<&Mat<f64>>,
    ) -> MoptResult<Self> {
        let n = start.dof();
        let mut set = Self {
            rows: Vec::new(),
            oriented: Vec::new(),
            breaks_rigid_body: false,
            notes: Vec::new(),
            lenient: Arc::default(),
        };
        for given in constraints {
            for a in c_atoms(&given.kind) {
                a.check(start.n_atoms())?;
            }
            if let ConstraintKind::Function(f) = &given.kind {
                check_function(f)?;
            }
            let c = given.resolve(start, hessian)?;
            if c.is_pending() {
                return Err(MoptError::config(
                    "an eigenvector constraint needs a Hessian at the start \
                     (an exact Hessian in the first iteration, e.g. -fc 1)",
                ));
            }
            set.breaks_rigid_body |= kind_breaks_rigid_body(&c.kind);
            let target = c.target.unwrap_or(0.0);
            let label = describe(&c.kind);
            match (&c.kind, &c.anchor) {
                (ConstraintKind::Distance(a, b), _) => {
                    set.push_geo(GeoFn::Distance(a.clone(), b.clone()), target, false, label)?
                }
                (ConstraintKind::Angle(a), _) => {
                    set.push_geo(GeoFn::Angle(*a), target, false, label)?
                }
                (ConstraintKind::Dihedral(a), _) => {
                    set.push_geo(GeoFn::Dihedral(*a), target, true, label)?
                }
                (ConstraintKind::Function(f), _) => {
                    let term = Arc::clone(&f.term);
                    let noise = evaluation_noise(
                        |y| {
                            let s = start.with_flat_coords_bohr(y)?;
                            let out =
                                term.evaluate(&s, &TermContext::at_iteration(0, &s), false)?;
                            Ok((out.energy.value(), out.gradient_bohr))
                        },
                        &start.coords_flat_bohr(),
                    )
                    .context(format!("constraint {label}: repeatability at the start"))?;
                    if let Some(tolerance) = f.tolerance {
                        set.notes.push(tolerance_note(&label, tolerance));
                    }
                    set.push_term(
                        Arc::clone(&f.term),
                        target,
                        f.period,
                        label,
                        noise,
                        (f.held_to(), f.warns_only),
                    )?;
                    set.note_noise(start)?;
                }
                (ConstraintKind::Coordinate(atom, axis), _) => {
                    set.rows.push(Row::Pinned {
                        dof: atom.dof(*axis),
                        target,
                    });
                }
                (ConstraintKind::Frozen(atoms), Some(ConstraintAnchor::Positions(at))) => {
                    check_anchor_len("frozen atoms", atoms.len(), at.len())?;
                    for (&atom, p) in atoms.iter().zip(at) {
                        for (axis, &value) in p.iter().enumerate() {
                            set.rows.push(Row::Pinned {
                                dof: atom.dof(axis),
                                target: value,
                            });
                        }
                    }
                }
                (ConstraintKind::Orientation(atoms), Some(ConstraintAnchor::Positions(at))) => {
                    check_anchor_len("orientation", atoms.len(), at.len())?;
                    let len = at.len().max(1) as f64;
                    let mut c0 = [0.0; 3];
                    for p in at {
                        for k in 0..3 {
                            c0[k] += p[k] / len;
                        }
                    }
                    let rel: Vec<[f64; 3]> = at
                        .iter()
                        .map(|p| [p[0] - c0[0], p[1] - c0[1], p[2] - c0[2]])
                        .collect();
                    // Eckart conditions sum_i a0_i x (r_i - c) = 0, linear in x.
                    for k in 0..3 {
                        let mut v = Col::zeros(n);
                        let mut target = 0.0;
                        for ((&atom, a), p) in atoms.iter().zip(&rel).zip(at) {
                            // d/dr of (a x r)_k = e_k x a (as a row: epsilon_klm a_l).
                            let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
                            v[atom.dof(k2)] += a[k1];
                            v[atom.dof(k1)] -= a[k2];
                            target += a[k1] * p[k2] - a[k2] * p[k1];
                        }
                        set.rows.push(Row::Linear {
                            v,
                            target,
                            label: format!("{label} (axis {k})"),
                        });
                    }
                    set.oriented.push((atoms.clone(), rel));
                }
                (
                    ConstraintKind::AtomPair(_) | ConstraintKind::EigenVector(_),
                    Some(ConstraintAnchor::Direction(direction)),
                ) => {
                    let mut v = Col::zeros(n);
                    for (atom, d) in direction {
                        atom.check(start.n_atoms())?;
                        for (axis, &value) in d.iter().enumerate() {
                            v[atom.dof(axis)] = value;
                        }
                    }
                    set.rows.push(Row::Linear { v, target, label });
                }
                (kind, _) => {
                    return Err(MoptError::config(format!(
                        "constraint {}: its anchor does not fit the kind",
                        describe(kind)
                    )))
                }
            }
        }
        Ok(set)
    }

    fn push_geo(&mut self, f: GeoFn, target: f64, periodic: bool, label: String) -> MoptResult<()> {
        let period = periodic.then_some(2.0 * PI);
        self.push_term(
            Arc::new(AdBias::new(f)),
            target,
            period,
            label,
            0.0,
            (GUARANTEE, false),
        )
    }

    /// Hold `f` at `target`; `held` is the residual it is held to and whether one above it
    /// is a warning only.
    fn push_term(
        &mut self,
        f: Arc<dyn PesTerm>,
        target: f64,
        period: Option<f64>,
        label: String,
        noise: f64,
        held: (f64, bool),
    ) -> MoptResult<()> {
        if !target.is_finite() {
            return Err(MoptError::config(format!(
                "constraint '{}': the target value is not finite",
                f.name()
            )));
        }
        self.rows.push(Row::Geometric {
            f,
            target,
            period,
            label,
            noise,
            tolerance: held.0,
            lenient: held.1,
        });
        Ok(())
    }

    /// The line of [`Self::notes`] for the function just pushed, when its measured noise
    /// lies above the rounding floor at `start`.
    fn note_noise(&mut self, start: &MolecularSystem) -> MoptResult<()> {
        let Some(Row::Geometric {
            f, label, noise, ..
        }) = self.rows.last()
        else {
            return Ok(());
        };
        let out = f.evaluate(start, &TermContext::at_iteration(0, start), false)?;
        let x = start.coords_flat_bohr();
        let machine = rounding_floor(
            out.gradient_bohr.iter().copied().zip(x.iter().copied()),
            out.energy.value(),
        );
        if NOISE_MARGIN * noise > machine {
            self.notes.push(format!(
                "Constraint {label}: its value repeats to {noise:.1e} under displacements of \
                 {NOISE_STEP:.0e} Bohr, above the rounding floor {machine:.1e}; it is restored to \
                 {:.1e}.",
                NOISE_MARGIN * noise
            ));
        }
        Ok(())
    }

    /// Lines for the log about the floors the constraints are restored to (a user-defined
    /// function whose evaluation noise lies above the rounding floor), one per such function.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// Number of scalar constraints.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether there are no constraints.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Whether the constraints fix the position or orientation in space (then the run is not
    /// invariant to rigid-body motions and the structure is not re-centered).
    pub fn breaks_rigid_body(&self) -> bool {
        self.breaks_rigid_body
    }

    /// Values minus targets (periodic values, such as dihedrals, wrapped into half a period).
    pub fn residuals(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        let x = sys.coords_flat_bohr();
        let mut r = Col::zeros(self.rows.len());
        for (k, row) in self.rows.iter().enumerate() {
            r[k] = match row {
                Row::Geometric {
                    f, target, period, ..
                } => wrap(value_of(f.as_ref(), sys)? - target, *period),
                Row::Linear { v, target, .. } => col_dot(v, &x) - target,
                Row::Pinned { dof, target } => x[*dof] - target,
            };
        }
        Ok(r)
    }

    /// Constraint gradients (rows), multipliers of `gradient`, and constraint curvature at
    /// `sys`.
    pub fn frame(&self, sys: &MolecularSystem, gradient: &Col<f64>) -> MoptResult<ConstraintFrame> {
        self.frame_in(sys, gradient, None)
    }

    /// [`Self::frame`] in the metric of the inverse weights `weights` (one per variable):
    /// the multipliers are `(J W J^T)^+ J W g`, so that `g - J^T lambda` is the gradient
    /// without its part along the constraints in the coordinates `W^(-1/2) x` (with `W` the
    /// inverse masses: the projection of the mass-weighted gradient of an IRC).
    pub fn frame_in(
        &self,
        sys: &MolecularSystem,
        gradient: &Col<f64>,
        weights: Option<&[f64]>,
    ) -> MoptResult<ConstraintFrame> {
        let n = sys.dof();
        if weights.is_some_and(|w| w.len() != n) {
            return Err(MoptError::config(format!(
                "constraints: {} weights for {n} degrees of freedom",
                weights.map_or(0, <[f64]>::len)
            )));
        }
        let m = self.rows.len();
        let mut j = Mat::zeros(m, n);
        let mut hessians: Vec<Option<Mat<f64>>> = Vec::with_capacity(m);
        let ctx = TermContext::at_iteration(0, sys);
        for (k, row) in self.rows.iter().enumerate() {
            match row {
                Row::Geometric { f, .. } => {
                    let out = f.evaluate(sys, &ctx, true)?;
                    if out.gradient_bohr.nrows() != n {
                        return Err(MoptError::calc(format!(
                            "constraint '{}' returned {} gradient components for {n} degrees of freedom",
                            f.name(),
                            out.gradient_bohr.nrows()
                        )));
                    }
                    for i in 0..n {
                        j[(k, i)] = out.gradient_bohr[i];
                    }
                    hessians.push(out.hessian_bohr);
                }
                Row::Linear { v, .. } => {
                    for i in 0..n {
                        j[(k, i)] = v[i];
                    }
                    hessians.push(None);
                }
                Row::Pinned { dof, .. } => {
                    j[(k, *dof)] = 1.0;
                    hessians.push(None);
                }
            }
        }
        let multipliers = gram_pinv_in(&j, weights)? * (weighted(&j, weights) * gradient);
        let mut curvature = Mat::zeros(n, n);
        for (k, h) in hessians.iter().enumerate() {
            if let Some(h) = h {
                curvature += h * faer::Scale(multipliers[k]);
            }
        }
        Ok(ConstraintFrame {
            jacobian: j,
            multipliers,
            curvature,
        })
    }

    /// `sys` moved back onto the constraints: oriented groups rotated rigidly to their starting
    /// orientation, frozen atoms reset, then Newton projection of all constraints.
    pub fn restore(&self, sys: &MolecularSystem) -> MoptResult<MolecularSystem> {
        self.restore_in(sys, None)
    }

    /// [`Self::restore`] with the Newton steps `W J^T (J W J^T)^+ r` in the metric of the
    /// inverse weights `weights` (one per variable): with inverse masses, the shortest way
    /// back in mass-weighted coordinates.
    pub fn restore_in(
        &self,
        sys: &MolecularSystem,
        weights: Option<&[f64]>,
    ) -> MoptResult<MolecularSystem> {
        Ok(self.restore_reported(sys, weights)?.system)
    }

    /// [`Self::restore_in`] with what it took: the Newton steps and the residuals reached.
    ///
    /// The constraints are restored to the rounding floor of double precision: oriented
    /// groups are turned back rigidly, frozen atoms and Cartesian components are assigned
    /// (bit-identical to their targets, and left out of the Newton steps, which do not move
    /// them), and the other constraints are solved together by Newton steps for as long as the
    /// largest residual, measured in units of its floor (`rounding_floor`, or the measured
    /// noise of a user-defined function, `evaluation_noise`), still decreases, until every
    /// residual is at its floor (stopping once they no longer decrease below [`GUARANTEE`]). A
    /// restore that leaves a residual above [`GUARANTEE`] is an error naming the constraint.
    pub fn restore_reported(
        &self,
        sys: &MolecularSystem,
        weights: Option<&[f64]>,
    ) -> MoptResult<RestoreOutcome> {
        let n = sys.dof();
        if weights.is_some_and(|w| w.len() != n) {
            return Err(MoptError::config(format!(
                "constraints: {} weights for {n} degrees of freedom",
                weights.map_or(0, <[f64]>::len)
            )));
        }
        let mut s = sys.clone();
        for (atoms, rel0) in &self.oriented {
            let c = centroid(&s, atoms);
            let mut cov = Mat::<f64>::zeros(3, 3);
            let rel: Vec<[f64; 3]> = atoms
                .iter()
                .map(|&a| {
                    let p = s.position(a).value();
                    [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
                })
                .collect();
            for (p, q) in rel.iter().zip(rel0) {
                for r in 0..3 {
                    for cc in 0..3 {
                        cov[(r, cc)] += p[r] * q[cc];
                    }
                }
            }
            let rot = proper_rotation_from_covariance(&cov);
            for (&a, p) in atoms.iter().zip(&rel) {
                let mut rp = c;
                for r in 0..3 {
                    for cc in 0..3 {
                        rp[r] += rot[(r, cc)] * p[cc];
                    }
                }
                s.set_position(a, Position::new(rp));
            }
        }
        // Assigned, not solved: frozen atoms and Cartesian components.
        let mut pinned = vec![false; n];
        {
            let mut x = s.coords_flat_bohr();
            for row in &self.rows {
                if let Row::Pinned { dof, target } = row {
                    x[*dof] = *target;
                    pinned[*dof] = true;
                }
            }
            s = s.with_flat_coords_bohr(&x)?;
        }
        let solved: Vec<usize> = (0..self.rows.len())
            .filter(|&k| !matches!(self.rows[k], Row::Pinned { .. }))
            .collect();
        let mut outcome = RestoreOutcome {
            system: s,
            newton_steps: 0,
            largest_residual: 0.0,
            largest_floor_ratio: 0.0,
        };
        if solved.is_empty() {
            return Ok(outcome);
        }
        // The best structure so far: (largest residual in floors, largest residual in units
        // of its tolerance, largest residual).
        let mut best: Option<(f64, f64, f64, MolecularSystem)> = None;
        let mut steps = 0;
        for _ in 0..RESTORE_STEPS {
            let s = &outcome.system;
            let x = s.coords_flat_bohr();
            let (r, floors, j) = self.solved_rows(s, &solved, &x)?;
            let (ratio, excess, worst) =
                (0..r.nrows()).fold((0.0_f64, 0.0_f64, 0.0_f64), |(a, e, w), k| {
                    (
                        a.max(r[k].abs() / floors[k]),
                        e.max(r[k].abs() / self.rows[solved[k]].tolerance()),
                        w.max(r[k].abs()),
                    )
                });
            if !ratio.is_finite() {
                break;
            }
            match &best {
                // Stalled within the tolerances: the residuals no longer decrease. Above them,
                // Newton steps on curved constraints need not decrease the largest residual at
                // every step, and they go on.
                Some(b) if ratio >= b.0 && b.1 <= 1.0 => break,
                Some(b) if ratio >= b.0 => {}
                _ => best = Some((ratio, excess, worst, s.clone())),
            }
            if ratio <= 1.0 {
                break;
            }
            // Newton step with the assigned coordinates left out: their columns of J are
            // zero, so the step leaves them exactly where they are.
            let mut j = j;
            for (i, _) in pinned.iter().enumerate().filter(|(_, p)| **p) {
                for k in 0..j.nrows() {
                    j[(k, i)] = 0.0;
                }
            }
            let dx = weighted(&j, weights).transpose() * (gram_pinv_in(&j, weights)? * &r);
            steps += 1;
            outcome.system = s.with_flat_coords_bohr(&(x - dx))?;
        }
        let Some((ratio, excess, worst, system)) = best else {
            return Err(MoptError::conv(
                "constraints could not be restored (the constraint values are not finite)",
            ));
        };
        // The guarantee: no hard constraint is left off by more than GUARANTEE, or the
        // tolerance it was given (the ReSPer coordinates, computed through a superposition,
        // are recorded and warned about).
        if excess > 1.0 {
            let x = system.coords_flat_bohr();
            let (r, _, _) = self.solved_rows(&system, &solved, &x)?;
            let over: Vec<usize> = (0..r.nrows())
                .filter(|&k| r[k].abs() > self.rows[solved[k]].tolerance())
                .collect();
            if over.iter().all(|&k| self.rows[solved[k]].lenient()) {
                for &k in &over {
                    self.record_lenient(&self.rows[solved[k]].label(), r[k].abs());
                }
                return Ok(RestoreOutcome {
                    system,
                    newton_steps: steps,
                    largest_residual: worst,
                    largest_floor_ratio: ratio,
                });
            }
            let k = over
                .iter()
                .copied()
                .filter(|&k| !self.rows[solved[k]].lenient())
                .max_by(|&a, &b| {
                    let row = |k: usize| r[k].abs() / self.rows[solved[k]].tolerance();
                    row(a).total_cmp(&row(b))
                })
                .unwrap_or(0);
            let row = &self.rows[solved[k]];
            return Err(MoptError::conv(format!(
                "constraints could not be restored: {} is off by {:.3e} after {steps} Newton \
                 steps, above the {}",
                row.label(),
                r[k].abs(),
                held_to_text(row.tolerance())
            )));
        }
        Ok(RestoreOutcome {
            system,
            newton_steps: steps,
            largest_residual: worst,
            largest_floor_ratio: ratio,
        })
    }

    /// The residuals, floors, and gradients (rows) of the constraints `solved` at `s` (one
    /// evaluation of each function, without its Hessian).
    fn solved_rows(
        &self,
        s: &MolecularSystem,
        solved: &[usize],
        x: &Col<f64>,
    ) -> MoptResult<(Col<f64>, Col<f64>, Mat<f64>)> {
        let n = s.dof();
        let m = solved.len();
        let (mut r, mut floors, mut j) = (Col::zeros(m), Col::zeros(m), Mat::zeros(m, n));
        let ctx = TermContext::at_iteration(0, s);
        for (k, &row) in solved.iter().enumerate() {
            let (value, target, period, noise) = match &self.rows[row] {
                Row::Geometric {
                    f,
                    target,
                    period,
                    noise,
                    ..
                } => {
                    let out = f.evaluate(s, &ctx, false)?;
                    if out.gradient_bohr.nrows() != n {
                        return Err(MoptError::calc(format!(
                            "constraint '{}' returned {} gradient components for {n} degrees of \
                             freedom",
                            f.name(),
                            out.gradient_bohr.nrows()
                        )));
                    }
                    for i in 0..n {
                        j[(k, i)] = out.gradient_bohr[i];
                    }
                    (out.energy.value(), *target, *period, *noise)
                }
                Row::Linear { v, target, .. } => {
                    for i in 0..n {
                        j[(k, i)] = v[i];
                    }
                    (col_dot(v, x), *target, None, 0.0)
                }
                Row::Pinned { .. } => continue,
            };
            r[k] = wrap(value - target, period);
            let machine = rounding_floor((0..n).map(|i| (j[(k, i)], x[i])), value);
            floors[k] = machine.max(NOISE_MARGIN * noise);
        }
        Ok((r, floors, j))
    }

    /// The largest residual of the constraints at `sys` in units of its rounding floor
    /// (`rounding_floor`, or the measured noise of a user-defined function); assigned
    /// coordinates count with their absolute residual (exactly zero after a restore).
    pub fn largest_floor_ratio(&self, sys: &MolecularSystem) -> MoptResult<f64> {
        let x = sys.coords_flat_bohr();
        let solved: Vec<usize> = (0..self.rows.len())
            .filter(|&k| !matches!(self.rows[k], Row::Pinned { .. }))
            .collect();
        let (r, floors, _) = self.solved_rows(sys, &solved, &x)?;
        let pinned = self
            .rows
            .iter()
            .filter_map(|row| match row {
                Row::Pinned { dof, target } => Some((x[*dof] - target).abs()),
                _ => None,
            })
            .fold(0.0_f64, f64::max);
        Ok((0..r.nrows())
            .map(|k| r[k].abs() / floors[k])
            .fold(pinned, f64::max))
    }
}

/// The bound a constraint is held to, for messages: "1e-10 every hard constraint is held to",
/// or the tolerance given with a user-defined function.
pub(crate) fn held_to_text(tolerance: f64) -> String {
    if (tolerance - GUARANTEE).abs() <= f64::EPSILON * GUARANTEE {
        format!("{GUARANTEE:.0e} every hard constraint is held to")
    } else {
        format!("{tolerance:.0e} it is held to (the tolerance given with it)")
    }
}

/// The line of the log for a user-defined function held to its own tolerance.
pub(crate) fn tolerance_note(label: &str, tolerance: f64) -> String {
    format!(
        "Constraint {label}: held to {tolerance:.1e} (the tolerance given with it) in place of \
         the {GUARANTEE:.0e} hard constraints are held to."
    )
}

/// Why the settings of a user-defined function cannot be used: a period or a tolerance that
/// is not positive and finite.
pub(crate) fn check_function(f: &ConstraintFunction) -> MoptResult<()> {
    for (what, value) in [("period", f.period), ("tolerance", f.tolerance)] {
        if let Some(v) = value {
            if !(v.is_finite() && v > 0.0) {
                return Err(MoptError::config(format!(
                    "constraint '{}': the {what} must be positive, got {v}",
                    f.term.name()
                )));
            }
        }
    }
    Ok(())
}

impl ConstraintSet {
    fn record_lenient(&self, label: &str, residual: f64) {
        if let Ok(mut seen) = self.lenient.lock() {
            match seen.iter_mut().find(|(l, _, _)| l == label) {
                Some(entry) => entry.1 = entry.1.max(residual),
                None => seen.push((label.to_string(), residual, false)),
            }
        }
    }

    /// One warning line for every ReSPer coordinate that a restore has left above
    /// [`GUARANTEE`] for the first time since the last call (once per constraint and run).
    pub fn lenient_notes(&self) -> Vec<String> {
        let Ok(mut seen) = self.lenient.lock() else {
            return Vec::new();
        };
        seen.iter_mut()
            .filter(|(_, _, told)| !*told)
            .map(|(label, residual, told)| {
                *told = true;
                format!(
                    "Warning: constraint {label} restored to {residual:.2e}, above the \
                     {GUARANTEE:.0e} hard constraints are held to (its value is computed \
                     through a superposition); the run goes on."
                )
            })
            .collect()
    }

    /// The largest residual of every ReSPer coordinate left above [`GUARANTEE`] in the run,
    /// for the end of the run (`None` when there was none).
    pub fn lenient_summary(&self) -> Option<String> {
        let seen = self.lenient.lock().ok()?;
        (!seen.is_empty()).then(|| {
            format!(
                "Warning: largest residuals above {GUARANTEE:.0e} in this run: {}.",
                seen.iter()
                    .map(|(l, r, _)| format!("{l} {r:.2e}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }
}

/// What a restore took ([`ConstraintSet::restore_reported`]).
#[derive(Debug, Clone)]
pub struct RestoreOutcome {
    /// The structure on the constraints.
    pub system: MolecularSystem,
    /// Newton steps taken (none when the structure was on the constraints already, or held
    /// only assigned coordinates).
    pub newton_steps: usize,
    /// Largest absolute residual of the solved constraints at the end (Bohr, rad, or the unit
    /// of a function).
    pub largest_residual: f64,
    /// Largest residual in units of its floor at the end (at most 1 when the floor was
    /// reached; somewhat more when the steps stalled at the rounding limit).
    pub largest_floor_ratio: f64,
}

/// Newton steps of a restore at most (quadratic convergence needs a handful).
const RESTORE_STEPS: usize = 100;

/// The largest residual (Bohr, rad, or the unit of a function) any hard constraint is left
/// with: a restore that cannot reach it is an error naming the constraint, never accepted
/// silently. The restore itself goes on to the rounding floor (about 1e-13), far below it,
/// so that values recomputed from files with 12 decimals stay within it as well.
pub const GUARANTEE: f64 = 1e-10;

/// The floor of a user-defined function is this multiple of its measured noise.
pub(crate) const NOISE_MARGIN: f64 = 2.0;

/// `d` wrapped into half a period.
fn wrap(d: f64, period: Option<f64>) -> f64 {
    match period {
        Some(p) => d - p * (d / p).round(),
        None => d,
    }
}

fn check_anchor_len(what: &str, atoms: usize, anchored: usize) -> MoptResult<()> {
    if atoms != anchored {
        return Err(MoptError::config(format!(
            "constraint on {what}: {atoms} atoms but {anchored} anchored positions"
        )));
    }
    Ok(())
}

fn col_dot(a: &Col<f64>, b: &Col<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn value_of(f: &dyn PesTerm, sys: &MolecularSystem) -> MoptResult<f64> {
    Ok(f.evaluate(sys, &TermContext::at_iteration(0, sys), false)?
        .energy
        .value())
}

fn c_atoms(kind: &ConstraintKind) -> Vec<AtomId> {
    match kind {
        ConstraintKind::Distance(a, b) => a.iter().chain(b).copied().collect(),
        ConstraintKind::Angle(a) => a.to_vec(),
        ConstraintKind::Dihedral(a) => a.to_vec(),
        ConstraintKind::Coordinate(a, _) => vec![*a],
        ConstraintKind::Frozen(a) | ConstraintKind::Orientation(a) => a.clone(),
        ConstraintKind::EigenVector(_) => Vec::new(),
        ConstraintKind::AtomPair(a) => a.to_vec(),
        ConstraintKind::Function(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::domain::{positions, Angstrom, Element, EvalOutput, Quantity, SpinCharge};

    fn id(i: usize) -> AtomId {
        AtomId::new(i)
    }

    /// Six atoms far from the origin (coordinates of about 20 Bohr).
    fn six() -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::from_atomic_number(18).unwrap(); 6],
            &positions::<Angstrom>(&[
                [10.0, 10.1, 9.8],
                [13.6, 10.3, 9.9],
                [11.9, 13.4, 10.2],
                [10.4, 11.2, 13.5],
                [14.1, 13.9, 12.8],
                [12.2, 9.0, 13.1],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    fn moved(s: &MolecularSystem, size: f64) -> MolecularSystem {
        let x = s.coords_flat_bohr();
        s.with_flat_coords_bohr(&Col::from_fn(x.nrows(), |i| {
            x[i] + size * ((i as f64) * 1.7 + 0.3).sin()
        }))
        .unwrap()
    }

    #[test]
    fn every_kind_is_restored_to_its_rounding_floor() {
        let start = six();
        let constraints = vec![
            Constraint::at_start(ConstraintKind::Distance(vec![id(0)], vec![id(1)])),
            Constraint::at_start(ConstraintKind::Distance(
                vec![id(0), id(1)],
                vec![id(2), id(3)],
            )),
            Constraint::at_start(ConstraintKind::Angle([id(0), id(1), id(2)])),
            Constraint::at_start(ConstraintKind::Dihedral([id(0), id(1), id(2), id(3)])),
            Constraint::at_start(ConstraintKind::Coordinate(id(4), 1)),
            Constraint::at_start(ConstraintKind::Frozen(vec![id(5)])),
            Constraint::at_start(ConstraintKind::Orientation(vec![id(2), id(3), id(4)])),
        ];
        let set = ConstraintSet::new(&constraints, &start, None).unwrap();
        for size in [1e-6, 1e-3, 0.1] {
            let out = set.restore_reported(&moved(&start, size), None).unwrap();
            let r = set.residuals(&out.system).unwrap();
            // Distances, angles, dihedrals, centroid distances, and the Eckart conditions of
            // the oriented group at about 1e-13 (Bohr or rad) for coordinates of 20 Bohr.
            assert!(r.iter().all(|v| v.abs() < 2e-13), "{size}: {r:?}");
            assert!(out.largest_residual <= GUARANTEE, "{out:?}");
            // Assigned: the same numbers.
            let (p, q) = (
                out.system.position(id(5)).value(),
                start.position(id(5)).value(),
            );
            assert_eq!(p, q);
            assert_eq!(
                out.system.position(id(4)).value()[1].to_bits(),
                start.position(id(4)).value()[1].to_bits()
            );
            // The oriented group has the orientation of the start: turning it back is the
            // identity to the floor.
            let back = set.restore(&out.system).unwrap();
            let shift = (out.system.coords_flat_bohr() - back.coords_flat_bohr()).norm_l2();
            assert!(shift < 1e-12, "{shift}");
        }
        assert!(set.notes().is_empty());
    }

    /// A distance with noise of about 1e-9 on every evaluation (as an inner superposition
    /// gives), a function of the structure alone.
    struct Noisy;

    impl PesTerm for Noisy {
        fn name(&self) -> &str {
            "noisy distance"
        }

        fn evaluate(
            &self,
            sys: &MolecularSystem,
            _ctx: &TermContext<'_>,
            _need_hessian: bool,
        ) -> MoptResult<EvalOutput> {
            let (a, b) = (sys.position(id(0)).value(), sys.position(id(1)).value());
            let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let hash = a.iter().chain(b.iter()).fold(0u64, |h, v| {
                (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3)
            });
            let noise = 1e-9 * ((hash % 2001) as f64 / 1000.0 - 1.0);
            let mut g = Col::zeros(sys.dof());
            for k in 0..3 {
                g[k] = -d[k] / r;
                g[3 + k] = d[k] / r;
            }
            Ok(EvalOutput::energy_gradient(Quantity::new(r + noise), g))
        }

        fn frame(&self) -> Frame {
            Frame::Invariant
        }

        fn history(&self) -> History {
            History::Static
        }
    }

    #[test]
    fn a_function_too_noisy_for_the_guarantee_is_an_error_naming_it() {
        let start = six();
        let noisy = Constraint::at_start(ConstraintKind::Function(ConstraintFunction::new(
            Arc::new(Noisy),
        )));
        let set = ConstraintSet::new(&[noisy], &start, None).unwrap();
        assert_eq!(set.notes().len(), 1, "{:?}", set.notes());
        assert!(
            set.notes()[0].contains("noisy distance"),
            "{:?}",
            set.notes()
        );
        // Its noise of 1e-9 is above the 1e-10 every hard constraint is held to: never
        // accepted silently: either a residual within it, or an error naming the constraint.
        for size in [0.05, 0.03, 0.02, 0.01] {
            match set.restore_reported(&moved(&start, size), None) {
                Ok(out) => {
                    let r = set.residuals(&out.system).unwrap()[0];
                    assert!(r.abs() <= GUARANTEE, "{size}: {r}");
                }
                Err(e) => {
                    let e = e.to_string();
                    assert!(e.contains("noisy distance") && e.contains("1e-10"), "{e}");
                }
            }
        }
        // With its own tolerance of 1e-8 the same function is held to it, with a note; a
        // tolerance that is not positive is refused.
        let tolerant = |tolerance: f64| {
            Constraint::at_start(ConstraintKind::Function(ConstraintFunction {
                tolerance: Some(tolerance),
                ..ConstraintFunction::new(Arc::new(Noisy))
            }))
        };
        let set = ConstraintSet::new(&[tolerant(1e-8)], &start, None).unwrap();
        assert!(
            set.notes()
                .iter()
                .any(|n| n.contains("noisy distance") && n.contains("1.0e-8")),
            "{:?}",
            set.notes()
        );
        for size in [0.05, 0.03, 0.02, 0.01] {
            let out = set.restore_reported(&moved(&start, size), None).unwrap();
            let r = set.residuals(&out.system).unwrap()[0];
            assert!(r.abs() <= 1e-8, "{size}: {r}");
        }
        let err = ConstraintSet::new(&[tolerant(-1.0)], &start, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tolerance must be positive"), "{err}");
        // Marked as a warning only (as the ReSPer coordinates are), it never fails: the
        // residuals above the guarantee are recorded and warned about once.
        let lenient = Constraint::at_start(ConstraintKind::Function(
            ConstraintFunction::new(Arc::new(Noisy)).warning_only(),
        ));
        let set = ConstraintSet::new(&[lenient], &start, None).unwrap();
        for size in [0.05, 0.03, 0.02, 0.01] {
            set.restore_reported(&moved(&start, size), None).unwrap();
        }
        let warned = set.lenient_notes();
        assert!(
            warned.iter().all(|w| w.contains("noisy distance")),
            "{warned:?}"
        );
        assert!(set.lenient_notes().is_empty());
        // A user function of plain distances has no note: its floor is that of rounding.
        let plain = Constraint::at_start(ConstraintKind::Function(ConstraintFunction::new(
            Arc::new(AdBias::new(GeoFn::Distance(vec![id(0)], vec![id(2)]))),
        )));
        let set = ConstraintSet::new(&[plain], &start, None).unwrap();
        assert!(set.notes().is_empty(), "{:?}", set.notes());
        let back = set.restore(&moved(&start, 0.05)).unwrap();
        assert!(set.residuals(&back).unwrap()[0].abs() < 2e-13);
    }
}
