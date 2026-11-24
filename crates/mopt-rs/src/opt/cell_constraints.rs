//! Geometric constraints of a variable-cell optimization (`optmain -vc` with `-fix`, `-pc`,
//! `-gfix`, `-uc`), as constraints of its vector of variables ([`super::cell::VariableCell`]:
//! the quasi-Cartesian coordinates `q_i = s_i h_0` of the atoms and the scaled lattice
//! vectors).
//!
//! - Frozen atoms (`-fix`) keep their fractional coordinates: their `q_i` are variables held
//!   at their values in the input ([`super::vector_constraints::FixedVariable`]).
//! - Distances, angles, dihedrals (minimum images, as [`crate::pes::constraint`] has them) and
//!   user-defined functions (`-uc`) are functions `c(x)` of the structure that the variables
//!   describe. Their gradient in the variables is mapped as the gradient of the energy is
//!   ([`super::cell::VariableCell::gradient_in_variables`]): the Cartesian gradient through
//!   `h_0^-1 h` for the atoms, and the derivative along the strain at fixed fractional
//!   coordinates (the "stress" of the function, analytic for the functions written over
//!   automatic differentiation) for the lattice.
//!
//! Refused, with the reason: a Cartesian component (`-pc x|y|z`), an oriented group (`-pc
//! rot`), the fixed lab-frame directions of `-pc eigvec` and `-pc atoms_pair`, and user
//! functions of Cartesian positions (`x()`, `y()`, `z()`): the lattice of a variable cell turns
//! freely and carries the atoms with it, so a fixed direction or point in space has no
//! meaning there.
//!
//! The optimizer holds the constraints as for plain variables
//! ([`super::vector_constraints`]): the constraint gradients are projected out of the
//! gradient and the step, and the structure is restored onto them after every step.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;
use std::sync::Arc;

use faer::{Col, Mat};

use super::cell::{lattice_gradient, VariableCell};
use super::vector_constraints::{FixedVariable, VectorConstraint, VectorConstraintSet};
use crate::domain::{AtomId, EvalOutput, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{angle_between, Scalar};
use crate::pes::bias::engine::norm;
use crate::pes::bias::restraints::dihedral_angle;
use crate::pes::bias::{AdBias, Clock, Geom, GeometricBias};
use crate::pes::constraint::{Constraint, ConstraintAnchor, ConstraintKind, GUARANTEE};
use crate::pes::stress::{term_stress, Stress, DEFAULT_STRAIN_STEP};
use crate::pes::term::{check_term_output, Frame, History, PesTerm, TermContext};

/// A distance, angle, or dihedral angle of a structure (minimum images in a cell).
///
/// MultiOptPy: no counterpart (new in mopt-rs; the same functions as the constraints of
/// structures in `pes::constraint`).
#[derive(Debug, Clone, PartialEq)]
enum Geometry {
    Distance(Vec<AtomId>, Vec<AtomId>),
    Angle([AtomId; 3]),
    Dihedral([AtomId; 4]),
}

impl GeometricBias for Geometry {
    fn name(&self) -> &str {
        match self {
            Self::Distance(..) => "distance constraint",
            Self::Angle(_) => "angle constraint",
            Self::Dihedral(_) => "dihedral constraint",
        }
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
            // Undefined where three consecutive atoms are collinear: NaN, refused by the
            // check of the output with the name of the constraint.
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
}

/// A function of the structure held at a value, as a constraint of the variables of a
/// variable cell.
struct CellFunction {
    map: VariableCell,
    term: Arc<dyn PesTerm>,
    period: Option<f64>,
    /// The residual it is held to ([`crate::pes::constraint::GUARANTEE`], or its own).
    tolerance: f64,
    /// A residual above it is a warning only (ReSPer coordinates).
    warns_only: bool,
}

impl CellFunction {
    /// The output of the term at the structure of `x`.
    fn output(
        &self,
        x: &Col<f64>,
        need_hessian: bool,
    ) -> MoptResult<(MolecularSystem, EvalOutput)> {
        let sys = self.map.system(x)?;
        let ctx = TermContext::at_iteration(0, &sys);
        let out = self.term.evaluate(&sys, &ctx, need_hessian)?;
        check_term_output(self.term.name(), &out)?;
        Ok((sys, out))
    }

    /// The stress of the function at `sys` (analytic for the functions over automatic
    /// differentiation, from differences of the value for a function that gives none).
    fn stress(&self, sys: &MolecularSystem, out: &EvalOutput) -> MoptResult<Stress> {
        let ctx = TermContext::at_iteration(0, sys);
        term_stress(self.term.as_ref(), out, sys, &ctx, DEFAULT_STRAIN_STEP)
    }
}

impl VectorConstraint for CellFunction {
    fn name(&self) -> String {
        self.term.name().to_string()
    }

    fn evaluate(&self, x: &Col<f64>) -> MoptResult<(f64, Col<f64>)> {
        let (sys, out) = self.output(x, false)?;
        // The derivative along the strain at fixed fractional coordinates (with its
        // antisymmetric part for a function of the orientation of the lattice).
        let stress = self.stress(&sys, &out)?;
        let gradient = self
            .map
            .gradient_in_variables(&sys, &out.gradient_bohr, &stress)?;
        Ok((out.energy.value(), gradient))
    }

    fn period(&self) -> Option<f64> {
        self.period
    }

    fn tolerance(&self) -> f64 {
        self.tolerance
    }

    fn warns_only(&self) -> bool {
        self.warns_only
    }

    /// The Hessian in the variables where the term gives its Cartesian Hessian and its second
    /// derivatives along the strain (the functions of the cell, [`CellExpression`]); `None`
    /// for the others.
    ///
    /// [`CellExpression`]: crate::pes::bias::CellExpression
    fn hessian(&self, x: &Col<f64>) -> MoptResult<Option<Mat<f64>>> {
        let (sys, out) = self.output(x, true)?;
        match (&out.hessian_bohr, &out.strain_hessian) {
            (Some(h), Some(strain)) => Ok(Some(self.map.hessian_in_variables(
                &sys,
                &out.gradient_bohr,
                h,
                strain,
            )?)),
            _ => Ok(None),
        }
    }
}

/// Why a constraint has no meaning in a variable cell, or `None` when it has one (the
/// commands that run variable-cell optimizations refuse such a constraint before they start).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn refused(kind: &ConstraintKind) -> Option<&'static str> {
    match kind {
        ConstraintKind::Coordinate(..) => Some(
            "-pc x|y|z holds a Cartesian component of an atom; the lattice of a variable cell \
             turns and carries the atoms, so a fixed coordinate in space has no meaning there \
             (-fix keeps the fractional coordinates of an atom)",
        ),
        ConstraintKind::Orientation(_) => Some(
            "-pc rot keeps the orientation of a group in space; the lattice of a variable cell \
             turns freely",
        ),
        ConstraintKind::EigenVector(_) => Some(
            "-pc eigvec holds a fixed direction of the Cartesian coordinates; the variables of \
             a variable cell are not Cartesian, and the lattice turns freely",
        ),
        ConstraintKind::AtomPair(_) => Some(
            "-pc atoms_pair holds the motion along a direction fixed in space; the lattice of a \
             variable cell turns freely (-pc bond holds the distance)",
        ),
        ConstraintKind::Function(f) if f.fixes_frame => Some(
            "a user-defined constraint of Cartesian positions (x(), y(), z()) depends on where \
             the structure is in space; the lattice of a variable cell turns freely",
        ),
        _ => None,
    }
}

/// The constraints `constraints` (resolved at the input, or resolved here at `start`) as
/// constraints of the variables of `map`, or why one of them cannot be held in a variable
/// cell.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn cell_constraints(
    map: &VariableCell,
    constraints: &[Constraint],
    start: &MolecularSystem,
) -> MoptResult<VectorConstraintSet> {
    let mut set = VectorConstraintSet::new();
    let n = start.n_atoms();
    // The functions of the cell alone, for the checks of consistency.
    let mut of_cell: Vec<Arc<CellFunction>> = Vec::new();
    for given in constraints {
        if let Some(why) = refused(&given.kind) {
            return Err(MoptError::unsupported(format!(
                "-vc with this constraint: {why}"
            )));
        }
        let c = given.resolve(start, None)?;
        let function = |term: Arc<dyn PesTerm>, period: Option<f64>| -> Arc<CellFunction> {
            Arc::new(CellFunction {
                map: map.clone(),
                term,
                period,
                tolerance: GUARANTEE,
                warns_only: false,
            })
        };
        let target = || {
            c.target.ok_or_else(|| {
                MoptError::config("a constraint of the variable cell has no target value")
            })
        };
        match (&c.kind, &c.anchor) {
            (ConstraintKind::Distance(a, b), _) => {
                check_atoms(a.iter().chain(b), n)?;
                let term = Arc::new(AdBias::new(Geometry::Distance(a.clone(), b.clone())));
                set.push(function(term, None), target()?)?;
            }
            (ConstraintKind::Angle(a), _) => {
                check_atoms(a.iter(), n)?;
                set.push(
                    function(Arc::new(AdBias::new(Geometry::Angle(*a))), None),
                    target()?,
                )?;
            }
            (ConstraintKind::Dihedral(a), _) => {
                check_atoms(a.iter(), n)?;
                set.push(
                    function(
                        Arc::new(AdBias::new(Geometry::Dihedral(*a))),
                        Some(2.0 * PI),
                    ),
                    target()?,
                )?;
            }
            (ConstraintKind::Function(f), _) => {
                crate::pes::constraint::check_function(f)?;
                let row = Arc::new(CellFunction {
                    map: map.clone(),
                    term: Arc::clone(&f.term),
                    period: f.period,
                    tolerance: f.held_to(),
                    warns_only: f.warns_only,
                });
                if f.term.atoms().is_some_and(|a| a.is_empty()) {
                    of_cell.push(Arc::clone(&row));
                }
                set.push(row, target()?)?;
            }
            (ConstraintKind::Frozen(atoms), Some(ConstraintAnchor::Positions(at))) => {
                if atoms.len() != at.len() {
                    return Err(MoptError::config(format!(
                        "-fix: {} atoms but {} anchored positions",
                        atoms.len(),
                        at.len()
                    )));
                }
                check_atoms(atoms.iter(), n)?;
                // The variables of the atoms are their fractional coordinates in the
                // starting cell times that cell: those of the anchored positions.
                let mut anchored = start.clone();
                for (&atom, p) in atoms.iter().zip(at) {
                    anchored.set_position(atom, crate::domain::Position::new(*p));
                }
                let q = map.variables(&anchored)?;
                for &atom in atoms {
                    for axis in 0..3 {
                        let index = atom.dof(axis);
                        set.push(Arc::new(FixedVariable { index }), q[index])?;
                    }
                }
            }
            (kind, _) => {
                return Err(MoptError::config(format!(
                    "-vc: the constraint {kind:?} was not resolved at the input"
                )))
            }
        }
    }
    check_cell(map, &of_cell, start)?;
    Ok(set)
}

/// Relative size below which a direction counts as a combination of others (the rows are
/// normalized; the gradients of independent functions of the lattice are far from it, a
/// dependent set reaches the rounding of the orthogonalization, about 1e-15).
const DEPENDENT: f64 = 1e-8;

/// The part of `v` orthogonal to the orthonormal `basis`, and its norm.
fn orthogonal_part(v: &[f64], basis: &[Vec<f64>]) -> (Vec<f64>, f64) {
    let mut out = v.to_vec();
    // Twice, for an orthogonality to the rounding (classical Gram-Schmidt, reorthogonalized).
    for _ in 0..2 {
        for b in basis {
            let overlap: f64 = out.iter().zip(b).map(|(x, y)| x * y).sum();
            for (x, y) in out.iter_mut().zip(b) {
                *x -= overlap * y;
            }
        }
    }
    let length = out.iter().map(|x| x * x).sum::<f64>().sqrt();
    (out, length)
}

/// The checks of the functions of the cell alone `rows` at the start, before the run:
///
/// - in a slab or a wire, a function of a lattice vector that is not a variable (the vector
///   along a direction that is not periodic) is refused: it has no meaning there (the
///   volume, the angles with that vector, its length or components);
/// - a function that does not change with the lattice variables, or that is a combination of
///   those before it (contradictory or redundant: the volume with the three lengths and
///   angles, a length held twice, more functions than the cell has degrees of freedom), is
///   refused with the names;
/// - functions that, with the rotations of the whole structure, leave the cell no freedom
///   (`shape` with a length, the six lattice parameters) are refused with the advice to drop
///   `-vc`.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
fn check_cell(
    map: &VariableCell,
    rows: &[Arc<CellFunction>],
    start: &MolecularSystem,
) -> MoptResult<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let x0 = map.variables(start)?;
    let dof = start.dof();
    let lattice = map.lattice_dimension();
    let names = ["a", "b", "c"];
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for row in rows {
        let name = row.name();
        let (sys, out) = row.output(&x0, false)?;
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("the structure has no periodic cell"))?;
        let of_lattice = lattice_gradient(&row.stress(&sys, &out)?, cell);
        let size = of_lattice
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        for k in (0..3).filter(|k| !map.variable_vectors().contains(k)) {
            if of_lattice[k].iter().any(|v| v.abs() > DEPENDENT * size) {
                return Err(MoptError::unsupported(format!(
                    "-vc with the constraint '{name}': it depends on the lattice vector {}, \
                     which is not a variable of this cell (only the periodic vectors change: \
                     hold their lengths, the angle between them, or their components)",
                    names[k]
                )));
            }
        }
        let gradient = row.evaluate(&x0)?.1;
        let part: Vec<f64> = (dof..dof + lattice).map(|i| gradient[i]).collect();
        let length = part.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(length > 0.0 && length.is_finite()) {
            return Err(MoptError::unsupported(format!(
                "-vc with the constraint '{name}': it does not change with the lattice \
                 variables of this cell"
            )));
        }
        let unit: Vec<f64> = part.iter().map(|v| v / length).collect();
        let (rest, left) = orthogonal_part(&unit, &basis);
        if left < DEPENDENT {
            return Err(MoptError::config(format!(
                "-vc: the constraints of the cell are contradictory or redundant: '{name}' is \
                 fixed by the ones before it ({}); the cell has {lattice} lattice variables, \
                 less its rotations",
                seen.iter()
                    .map(|s| format!("'{s}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        basis.push(rest.iter().map(|v| v / left).collect());
        seen.push(name);
    }
    // With the rotations of the whole structure that the variables describe, the cell has
    // no freedom left when the span is complete.
    let mut span = basis.clone();
    for mode in map.rigid_modes(&x0)?.into_iter().skip(3) {
        let part: Vec<f64> = (dof..dof + lattice).map(|i| mode[i]).collect();
        let length = part.iter().map(|v| v * v).sum::<f64>().sqrt();
        if length < DEPENDENT {
            continue;
        }
        let unit: Vec<f64> = part.iter().map(|v| v / length).collect();
        let (rest, left) = orthogonal_part(&unit, &span);
        if left >= DEPENDENT {
            span.push(rest.iter().map(|v| v / left).collect());
        }
    }
    if span.len() >= lattice {
        return Err(MoptError::unsupported(format!(
            "-vc: the constraints of the cell ({}) fix it completely (only its rotation is \
             left); optimize in the fixed cell instead: drop -vc and the constraints of the cell",
            seen.iter()
                .map(|s| format!("'{s}'"))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(())
}

/// Whether `constraints` take away the rotations of the lattice in a variable cell (a
/// function of the Cartesian components of the lattice vectors, `-uc "cell_cz()"`, `-vc_fix
/// c_vector`): the rotations are then no symmetry of the constrained problem, and are left in
/// its Hessian (the rotations about a held vector keep no curvature, as on any surface).
pub fn pins_rotations(constraints: &[Constraint]) -> bool {
    constraints.iter().any(
        |c| matches!(&c.kind, ConstraintKind::Function(f) if f.term.frame() != Frame::Invariant),
    )
}

fn check_atoms<'a>(atoms: impl Iterator<Item = &'a AtomId>, n: usize) -> MoptResult<()> {
    for a in atoms {
        a.check(n)?;
    }
    Ok(())
}

/// Whether `constraints` take away the translations of the atoms in a variable cell (frozen
/// atoms: their fractional coordinates are fixed); the rotations of the lattice, which carry
/// the atoms with their fractional coordinates, are left to them.
pub fn pins_translations(constraints: &[Constraint]) -> bool {
    constraints
        .iter()
        .any(|c| matches!(&c.kind, ConstraintKind::Frozen(a) if !a.is_empty()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::domain::{positions, Angstrom, Element, PeriodicCell, SpinCharge, Vector3};
    use crate::pes::constraint::ConstraintSet;

    fn crystal() -> MolecularSystem {
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new([5.1, 0.0, 0.0]),
            Vector3::<Angstrom>::new([0.4, 5.3, 0.0]),
            Vector3::<Angstrom>::new([-0.3, 0.2, 5.6]),
            [true; 3],
        )
        .unwrap();
        MolecularSystem::new(
            vec![Element::parse_token("Ar").unwrap(); 4],
            &positions::<Angstrom>(&[
                [0.3, 0.2, 0.1],
                [2.9, 2.4, 0.3],
                [2.6, 0.1, 2.9],
                [4.9, 2.8, 5.1],
            ]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap()
    }

    #[test]
    fn the_gradient_of_a_constraint_in_the_variables_is_that_of_its_value() {
        let sys = crystal();
        let map = VariableCell::new(&sys, 2.0, 0.0).unwrap();
        let atoms = |v: &[usize]| v.iter().map(|&a| AtomId::new(a)).collect::<Vec<_>>();
        // The pair 1-4 is closest through the boundary: the minimum image is followed.
        let constraints = vec![
            Constraint::at_start(ConstraintKind::Distance(atoms(&[0]), atoms(&[3]))),
            Constraint::at_start(ConstraintKind::Angle([
                AtomId::new(0),
                AtomId::new(1),
                AtomId::new(2),
            ])),
            Constraint::at_start(ConstraintKind::Dihedral([
                AtomId::new(0),
                AtomId::new(1),
                AtomId::new(2),
                AtomId::new(3),
            ])),
        ];
        let set = cell_constraints(&map, &constraints, &sys).unwrap();
        let x0 = map.variables(&sys).unwrap();
        // The targets are the values of the structure, as the constraints of structures have
        // them.
        let r = set.residuals(&x0).unwrap();
        assert!(r.iter().all(|v| v.abs() < 1e-12), "{r:?}");
        let reference = ConstraintSet::new(&constraints, &sys, None).unwrap();
        // Move the atoms and the cell a little: the residuals agree with those of the
        // structure.
        let x = Col::from_fn(x0.nrows(), |i| x0[i] + 0.01 * ((i as f64) * 0.7).sin());
        let moved = map.system(&x).unwrap();
        let (a, b) = (
            set.residuals(&x).unwrap(),
            reference.residuals(&moved).unwrap(),
        );
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-12, "{k}: {} {}", a[k], b[k]);
        }
        // Restored to the rounding floor from the moved structure.
        let back = set.restore(&x).unwrap();
        let r = set.residuals(&back).unwrap();
        assert!(r.iter().all(|v| v.abs() < 1e-12), "{r:?}");
        // The Jacobian against central differences of the values.
        let frame = set.frame(&x).unwrap();
        let h = 1e-5;
        for i in 0..x.nrows() {
            let at = |s: f64| {
                let mut y = x.clone();
                y[i] += s * h;
                set.residuals(&y).unwrap()
            };
            let (p, m) = (at(1.0), at(-1.0));
            for k in 0..3 {
                let numeric = (p[k] - m[k]) / (2.0 * h);
                let analytic = frame.jacobian[(k, i)];
                assert!(
                    (numeric - analytic).abs() < 1e-7 * (1.0 + analytic.abs()),
                    "row {k}, variable {i}: {analytic} vs {numeric}"
                );
            }
        }
    }

    fn fixed(items: &[&str]) -> Vec<Constraint> {
        let tokens: Vec<String> = items.iter().map(|s| s.to_string()).collect();
        crate::cli::constraints::cell_fix_constraints(&tokens).unwrap()
    }

    #[test]
    fn constraints_of_the_cell_have_exact_gradients_and_curvatures_in_the_variables() {
        let sys = crystal();
        let map = VariableCell::new(&sys, 2.0, 0.0).unwrap();
        let constraints = fixed(&["volume", "b", "gamma", "cz"]);
        let mut user = constraints.clone();
        user.push(crate::cli::constraints::user_constraint("dist(1,2) / cell_a()").unwrap());
        let set = cell_constraints(&map, &user, &sys).unwrap();
        let x0 = map.variables(&sys).unwrap();
        assert!(set.residuals(&x0).unwrap().iter().all(|r| r.abs() < 1e-12));
        let x = Col::from_fn(x0.nrows(), |i| x0[i] + 0.02 * ((i as f64) * 0.9).cos());
        let back = set.restore(&x).unwrap();
        let r = set.residuals(&back).unwrap();
        assert!(r.iter().all(|v| v.abs() < 1e-10), "{r:?}");
        // Gradients and Hessians against central differences.
        let frame = set.frame(&x).unwrap();
        let h = 1e-5;
        let n = x.nrows();
        for i in 0..n {
            let at = |s: f64| {
                let mut y = x.clone();
                y[i] += s * h;
                set.residuals(&y).unwrap()
            };
            let (p, m) = (at(1.0), at(-1.0));
            for k in 0..r.nrows() {
                let numeric = (p[k] - m[k]) / (2.0 * h);
                let analytic = frame.jacobian[(k, i)];
                assert!(
                    (numeric - analytic).abs() < 1e-6 * (1.0 + analytic.abs()),
                    "row {k}, variable {i}: {analytic} vs {numeric}"
                );
            }
        }
        // The curvature of the Lagrangian with all multipliers one is the sum of the
        // Hessians; that of the volume against differences of its gradient.
        // With the gradient of the objective equal to that of the volume, its multiplier is one.
        let volume = cell_constraints(&map, &fixed(&["volume"]), &sys).unwrap();
        let g = volume.frame(&x).unwrap().jacobian;
        let gradient = Col::from_fn(n, |i| g[(0, i)]);
        let curvature = volume.curvature(&x, &gradient).unwrap().unwrap();
        for i in 0..n {
            let at = |s: f64| {
                let mut y = x.clone();
                y[i] += s * h;
                volume.frame(&y).unwrap().jacobian
            };
            let (p, m) = (at(1.0), at(-1.0));
            for j in 0..n {
                let numeric = (p[(0, j)] - m[(0, j)]) / (2.0 * h);
                assert!(
                    (numeric - curvature[(i, j)]).abs() < 1e-5 * (1.0 + numeric.abs()),
                    "({i}, {j}): {} vs {numeric}",
                    curvature[(i, j)]
                );
            }
        }
        assert!(pins_rotations(&constraints));
        assert!(!pins_rotations(&fixed(&["volume", "a"])));
    }

    #[test]
    fn contradictory_redundant_and_complete_sets_of_the_cell_are_refused_by_name() {
        let sys = crystal();
        let map = VariableCell::new(&sys, 2.0, 0.0).unwrap();
        let refused = |items: &[&str]| {
            cell_constraints(&map, &fixed(items), &sys)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default()
        };
        let e = refused(&["a,b,c,alpha,beta,gamma,volume"]);
        assert!(e.contains("redundant") && e.contains("volume()"), "{e}");
        let e = refused(&["a", "a=5.2"]);
        assert!(e.contains("contradictory") && e.contains("cell_a()"), "{e}");
        let e = refused(&["shape", "a"]);
        assert!(
            e.contains("fix it completely") && e.contains("drop -vc"),
            "{e}"
        );
        let e = refused(&["a,b,c,alpha,beta,gamma"]);
        assert!(e.contains("fix it completely"), "{e}");
        let e = refused(&["c_vector", "shape"]);
        assert!(e.contains("fix it completely"), "{e}");
        // These leave freedom.
        for ok in [&["shape"][..], &["volume"], &["c_vector"], &["a,b,gamma"]] {
            assert!(cell_constraints(&map, &fixed(ok), &sys).is_ok(), "{ok:?}");
        }
        // A slab: its third vector is no variable.
        let mut slab = crystal();
        if let Some(cell) = slab.cell.as_mut() {
            cell.periodic_axes = [true, true, false];
        }
        let map = VariableCell::new(&slab, 2.0, 0.0).unwrap();
        let e = cell_constraints(&map, &fixed(&["volume"]), &slab)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            e.contains("lattice vector c") && e.contains("volume()"),
            "{e}"
        );
        assert!(cell_constraints(&map, &fixed(&["a", "gamma"]), &slab).is_ok());
    }

    #[test]
    fn frozen_atoms_keep_their_fractional_coordinates_and_fixed_directions_are_refused() {
        let sys = crystal();
        let map = VariableCell::new(&sys, 2.0, 0.0).unwrap();
        let frozen = [Constraint::at_start(ConstraintKind::Frozen(vec![
            AtomId::new(1),
        ]))];
        let set = cell_constraints(&map, &frozen, &sys).unwrap();
        assert_eq!(set.len(), 3);
        let x0 = map.variables(&sys).unwrap();
        let mut x = x0.clone();
        for i in 0..x.nrows() {
            x[i] += 0.05 * ((i as f64) * 1.3).cos();
        }
        let back = set.restore(&x).unwrap();
        // The variables of the frozen atom are assigned: the same numbers as at the start.
        for i in 3..6 {
            assert_eq!(back[i].to_bits(), x0[i].to_bits(), "variable {i}");
        }
        let (s, s0) = (map.system(&back).unwrap(), map.system(&x0).unwrap());
        let frac = |m: &MolecularSystem| {
            m.active_cell()
                .unwrap()
                .cart_to_frac(m.position(AtomId::new(1)).value())
        };
        let (a, b) = (frac(&s), frac(&s0));
        for c in 0..3 {
            assert!((a[c] - b[c]).abs() < 1e-12, "{a:?} {b:?}");
        }
        assert!(pins_translations(&frozen));
        for kind in [
            ConstraintKind::Coordinate(AtomId::new(0), 2),
            ConstraintKind::Orientation(vec![AtomId::new(0), AtomId::new(1)]),
            ConstraintKind::EigenVector(0),
            ConstraintKind::AtomPair([AtomId::new(0), AtomId::new(1)]),
        ] {
            let err = cell_constraints(&map, &[Constraint::at_start(kind)], &sys)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(err.contains("-vc") && err.contains("-pc"), "{err}");
        }
    }
}
