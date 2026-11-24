//! Constraint options (`-fix`, `-pc`, `-gfix`) to [`Constraint`]s.
//!
//! - `-fix atoms ...`: freeze atoms (`1,2,5-8`; every argument is used, MultiOptPy reads only
//!   the first).
//! - `-pc name atoms ...`: hold at the starting value; `-pc manual name atoms value ...`: hold at
//!   `value` (Angstrom for `bond`, `fbond`, `x`, `y`, `z`; degrees for `angle`, `dihedral`).
//!   Names: `bond` (2 atoms), `fbond` (atoms of both fragments followed by the number of atoms of
//!   the first), `angle` (3), `dihedral` (4), `x`/`y`/`z` (1), `rot` (a group keeps its
//!   orientation), `eigvec` (mode index instead of atoms; needs an exact Hessian in the first
//!   iteration), `atoms_pair` (2).
//! - `-gfix atoms ...`: no motion along the distances between the given atoms (MultiOptPy
//!   parses this option without using it; mopt-rs holds those distances at their starting
//!   values, which zeroes the gradient along them).
//! - `-uc "EXPR" ...` (mopt-rs): any function of the structure, written in the expression
//!   language of `-ubias` (atomic units), is held fixed. `"EXPR"` holds the starting value,
//!   `"EXPR = VALUE"` holds a value (`VALUE` may itself depend on the geometry, then
//!   `EXPR - VALUE = 0` is held), and a trailing `"; period = P"` wraps the residual into half
//!   a period (angle-like functions, e.g. `period = 2*pi`). Example:
//!   `-uc "dist(1,2) - dist(3,4)" "angle(1,2,3) = 109.5*deg" "dihedral(1,2,3,4) = 0; period = 2*pi"`.
//!
//! MultiOptPy: `interface.py:force_data_parser` (`-fix`, `-gfix`, `-pc`) and the option parsing of
//!  `Constraint/constraint_condition.py:ProjectOutConstrain`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};

use crate::pes::bias::ExpressionBias;
use crate::pes::constraint::{Constraint, ConstraintFunction, ConstraintKind};

fn atoms(t: &str) -> MoptResult<Vec<AtomId>> {
    parse_atom_indices_one_based(t)?
        .into_iter()
        .map(AtomId::from_one_based)
        .collect()
}

fn fixed<const N: usize>(name: &str, t: &str) -> MoptResult<[AtomId; N]> {
    atoms(t)?.try_into().map_err(|v: Vec<AtomId>| {
        MoptError::parse(format!(
            "-pc {name}: expected {N} atoms, got {} in '{t}'",
            v.len()
        ))
    })
}

fn constant(what: &str, source: &str, text: &str) -> MoptResult<f64> {
    ExpressionBias::parse(text)?
        .constant_value()
        .filter(|v| v.is_finite())
        .ok_or_else(|| {
            MoptError::parse(format!(
                "-uc '{source}': the {what} must be a finite constant, got '{text}'"
            ))
        })
}

/// One `-uc` constraint (`"EXPR"`, `"EXPR = VALUE"`, optionally followed by `"; period = P"`
/// and `"; tolerance = T"`).
pub fn user_constraint(source: &str) -> MoptResult<Constraint> {
    user_constraint_named(source, &format!("user_constraint '{}'", source.trim()))
}

/// [`user_constraint`] reported under the name `label` (the constraints of `-vc_fix`).
pub fn user_constraint_named(source: &str, label: &str) -> MoptResult<Constraint> {
    let mut parts = source.split(';');
    let body = parts.next().unwrap_or_default();
    let (mut period, mut tolerance) = (None, None);
    for setting in parts {
        let (key, value) = setting.split_once('=').ok_or_else(|| {
            MoptError::parse(format!(
                "-uc '{source}': expected '; period = P' or '; tolerance = T' after ';'"
            ))
        })?;
        let slot = match key.trim() {
            "period" => &mut period,
            "tolerance" => &mut tolerance,
            other => {
                return Err(MoptError::parse(format!(
                    "-uc '{source}': unknown setting '{other}' (period, tolerance)"
                )))
            }
        };
        let v = constant(key.trim(), source, value)?;
        if v <= 0.0 {
            return Err(MoptError::parse(format!(
                "-uc '{source}': the {} must be positive",
                key.trim()
            )));
        }
        if slot.replace(v).is_some() {
            return Err(MoptError::parse(format!(
                "-uc '{source}': '{}' is given twice",
                key.trim()
            )));
        }
    }
    let (expr, target) = match body.split_once('=') {
        None => (ExpressionBias::parse(body)?, None),
        Some((lhs, rhs)) => {
            let value = ExpressionBias::parse(rhs)?;
            match value.constant_value() {
                Some(v) if v.is_finite() => (ExpressionBias::parse(lhs)?, Some(v)),
                Some(_) => {
                    return Err(MoptError::parse(format!(
                        "-uc '{source}': the value is not finite"
                    )))
                }
                None => (
                    ExpressionBias::parse(&format!("({lhs}) - ({rhs})"))?,
                    Some(0.0),
                ),
            }
        }
    };
    if expr.constant_value().is_some() {
        return Err(MoptError::parse(format!(
            "-uc '{source}': the constraint does not depend on the structure"
        )));
    }
    let fixes_frame = expr.uses_cartesian();
    // An expression of the atoms alone over automatic differentiation in their coordinates;
    // one that uses the cell is differentiated along the lattice as well.
    let term = expr.named(label).into_term();
    Ok(Constraint {
        kind: ConstraintKind::Function(ConstraintFunction {
            term,
            period,
            fixes_frame,
            tolerance,
            warns_only: false,
        }),
        target,
        anchor: None,
    })
}

/// The `-uc` expressions that hold one item of `-vc_fix` (`NAME` or `NAME=VALUE`; values in
/// Angstrom, degrees, and Angstrom^3), with the name each is reported under: the dedicated
/// option is translated into these, so that `-vc_fix X` and `-uc` of the same expressions are
/// one and the same constraint.
///
/// | item | held |
/// |---|---|
/// | `a`, `b`, `c` | `cell_a()`, ... (lengths) |
/// | `alpha`, `beta`, `gamma` | `cell_alpha()`, ... (angles) |
/// | `volume` | `volume()` |
/// | `shape` | `cell_b()/cell_a()`, `cell_c()/cell_a()`, `cell_alpha()`, `cell_beta()`, `cell_gamma()` (only an isotropic scaling is left) |
/// | `a_vector`, `b_vector`, `c_vector` | the three Cartesian components of the vector |
/// | `ax`, `ay`, ..., `cz` | one Cartesian component |
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn cell_fix_expressions(item: &str) -> MoptResult<Vec<(String, String)>> {
    let (name, value) = match item.split_once('=') {
        Some((n, v)) => {
            let v = v
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    MoptError::parse(format!("-vc_fix {item}: the value must be a number"))
                })?;
            (n.trim().to_ascii_lowercase(), Some(v))
        }
        None => (item.trim().to_ascii_lowercase(), None),
    };
    let label = |expr: &str| format!("-vc_fix {} ({expr})", item.trim());
    let held = |function: &str, unit: &str| -> Vec<(String, String)> {
        let expr = match value {
            // `{v}` prints the shortest text that reads back as the same number.
            Some(v) => format!("{function} = {v}*{unit}"),
            None => function.to_string(),
        };
        vec![(label(&expr), expr)]
    };
    let no_value = |what: &str| -> MoptResult<()> {
        match value {
            Some(_) => Err(MoptError::parse(format!(
                "-vc_fix {item}: '{what}' holds the values of the input and takes no value"
            ))),
            None => Ok(()),
        }
    };
    Ok(match name.as_str() {
        "a" | "b" | "c" => held(&format!("cell_{name}()"), "ang"),
        "alpha" | "beta" | "gamma" => held(&format!("cell_{name}()"), "deg"),
        "volume" => held("volume()", "ang^3"),
        "shape" => {
            no_value("shape")?;
            [
                "cell_b()/cell_a()",
                "cell_c()/cell_a()",
                "cell_alpha()",
                "cell_beta()",
                "cell_gamma()",
            ]
            .iter()
            .map(|e| (label(e), e.to_string()))
            .collect()
        }
        v @ ("a_vector" | "b_vector" | "c_vector") => {
            no_value(v)?;
            let k = &v[..1];
            ["x", "y", "z"]
                .iter()
                .map(|c| {
                    let e = format!("cell_{k}{c}()");
                    (label(&e), e)
                })
                .collect()
        }
        c if c.len() == 2
            && matches!(&c[..1], "a" | "b" | "c")
            && matches!(&c[1..], "x" | "y" | "z") =>
        {
            held(&format!("cell_{c}()"), "ang")
        }
        other => {
            return Err(MoptError::parse(format!(
                "-vc_fix: unknown item '{other}' (a, b, c, alpha, beta, gamma, volume, shape, \
                 a_vector, b_vector, c_vector, ax, ay, ..., cz; NAME=VALUE in Angstrom, \
                 degrees, Angstrom^3)"
            )))
        }
    })
}

/// The constraints of `-vc_fix` (comma-separated items in every argument).
pub fn cell_fix_constraints(tokens: &[String]) -> MoptResult<Vec<Constraint>> {
    let mut out = Vec::new();
    for item in tokens.iter().flat_map(|t| t.split(',')) {
        if item.trim().is_empty() {
            continue;
        }
        for (label, expr) in cell_fix_expressions(item)? {
            out.push(user_constraint_named(&expr, &label)?);
        }
    }
    Ok(out)
}

/// One `-pc` constraint; `value` in the command-line units.
pub fn projection_constraint(
    name: &str,
    atoms_arg: &str,
    value: Option<f64>,
) -> MoptResult<Constraint> {
    let length = value.map(|v| v * Units::ANGSTROM_TO_BOHR);
    let angle = value.map(|v| v * Units::DEG_TO_RAD);
    let no_value = |kind: ConstraintKind| {
        if value.is_some() {
            Err(MoptError::parse(format!(
                "-pc manual: '{name}' takes no value"
            )))
        } else {
            Ok(Constraint::at_start(kind))
        }
    };
    match name.to_ascii_lowercase().as_str() {
        "bond" => {
            let [a, b] = fixed(name, atoms_arg)?;
            Ok(Constraint {
                kind: ConstraintKind::Distance(vec![a], vec![b]),
                target: length,
                anchor: None,
            })
        }
        "fbond" => {
            let mut list = atoms(atoms_arg)?;
            let split = list
                .pop()
                .map(AtomId::one_based)
                .filter(|&k| k >= 1 && k < list.len())
                .ok_or_else(|| {
                    MoptError::parse(format!(
                        "-pc fbond: atoms of both fragments followed by the size of the first, got '{atoms_arg}'"
                    ))
                })?;
            let second = list.split_off(split);
            Ok(Constraint {
                kind: ConstraintKind::Distance(list, second),
                target: length,
                anchor: None,
            })
        }
        "angle" => Ok(Constraint {
            kind: ConstraintKind::Angle(fixed(name, atoms_arg)?),
            target: angle,
            anchor: None,
        }),
        "dihedral" => Ok(Constraint {
            kind: ConstraintKind::Dihedral(fixed(name, atoms_arg)?),
            target: angle,
            anchor: None,
        }),
        axis @ ("x" | "y" | "z") => {
            let [a] = fixed(name, atoms_arg)?;
            let k = usize::from(axis == "y") + 2 * usize::from(axis == "z");
            Ok(Constraint {
                kind: ConstraintKind::Coordinate(a, k),
                target: length,
                anchor: None,
            })
        }
        "rot" => no_value(ConstraintKind::Orientation(atoms(atoms_arg)?)),
        "eigvec" => {
            let k = atoms_arg.trim().parse::<usize>().map_err(|e| {
                MoptError::parse(format!("-pc eigvec: mode index '{atoms_arg}': {e}"))
            })?;
            no_value(ConstraintKind::EigenVector(k))
        }
        "atoms_pair" => no_value(ConstraintKind::AtomPair(fixed(name, atoms_arg)?)),
        other => Err(MoptError::parse(format!(
            "-pc: unknown constraint '{other}' (bond, fbond, angle, dihedral, x, y, z, rot, \
             eigvec, atoms_pair)"
        ))),
    }
}

/// `-pc` arguments.
pub fn parse_projection_constraints(tokens: &[String]) -> MoptResult<Vec<Constraint>> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    if tokens[0].eq_ignore_ascii_case("manual") {
        if !tokens.len().is_multiple_of(4) {
            return Err(MoptError::parse(
                "-pc manual expects groups of: manual name atoms value",
            ));
        }
        tokens
            .chunks(4)
            .map(|c| {
                let v = c[3]
                    .parse::<f64>()
                    .map_err(|e| MoptError::parse(format!("-pc manual value '{}': {e}", c[3])))?;
                projection_constraint(&c[1], &c[2], Some(v))
            })
            .collect()
    } else {
        if !tokens.len().is_multiple_of(2) {
            return Err(MoptError::parse("-pc expects pairs: name atoms"));
        }
        tokens
            .chunks(2)
            .map(|c| projection_constraint(&c[0], &c[1], None))
            .collect()
    }
}

/// All constraint options of `p`.
pub fn parse_constraints(p: &ParsedArgs) -> MoptResult<Vec<Constraint>> {
    let mut out = Vec::new();
    let frozen: Vec<AtomId> = p
        .get_list("fix_atoms")
        .iter()
        .map(|t| atoms(t))
        .collect::<MoptResult<Vec<_>>>()?
        .concat();
    if !frozen.is_empty() {
        out.push(Constraint::at_start(ConstraintKind::Frozen(frozen)));
    }
    out.extend(parse_projection_constraints(
        p.get_list("projection_constrain"),
    )?);
    for t in p.get_list("gradient_fix_atoms") {
        let a = atoms(t)?;
        if a.len() < 2 {
            return Err(MoptError::parse(format!(
                "-gfix expects at least two atoms, got '{t}'"
            )));
        }
        for i in 0..a.len() {
            for j in i + 1..a.len() {
                out.push(Constraint::at_start(ConstraintKind::Distance(
                    vec![a[i]],
                    vec![a[j]],
                )));
            }
        }
    }
    for t in p.get_list("user_constraint") {
        out.push(user_constraint(t)?);
    }
    let cell = p.get_list("variable_cell_fix");
    let variable = p.get_bool("variable_cell", false);
    if !cell.is_empty() && !variable {
        return Err(MoptError::config(
            "-vc_fix holds the cell of a variable-cell optimization; it needs -vc",
        ));
    }
    out.extend(cell_fix_constraints(cell)?);
    // A function of the cell can only change, and so only be held, when the cell does.
    if !variable {
        if let Some(t) = p
            .get_list("user_constraint")
            .iter()
            .find(|t| uses_cell_anywhere(t))
        {
            return Err(MoptError::config(format!(
                "-uc '{t}': a function of the cell (cell_a(), volume(), ...) is held in a \
                 variable-cell optimization (-vc); in a fixed cell it cannot change"
            )));
        }
    }
    Ok(out)
}

/// Whether either side of `EXPR = VALUE` (before any `;` setting) uses a function of the cell.
pub(crate) fn uses_cell_anywhere(source: &str) -> bool {
    let body = source.split(';').next().unwrap_or_default();
    body.split('=')
        .any(|side| ExpressionBias::parse(side).is_ok_and(|e| e.uses_cell()))
}

/// The constraints of a run resolved once at its input structure `input`
/// ([`Constraint::resolve`]): every target, frozen position, oriented group, and `atoms_pair`
/// direction is fixed there, so that every stage and job of the run holds the same values.
/// Explicit targets (`-pc manual`, `-uc "... = VALUE"`) are kept. Eigenvector constraints stay
/// pending until a Hessian exists ([`pending_constraints`]); resolve them later with
/// [`Constraint::resolve`] and the Hessian.
pub fn resolve_for_run(
    constraints: &[Constraint],
    input: &MolecularSystem,
) -> MoptResult<Vec<Constraint>> {
    constraints.iter().map(|c| c.resolve(input, None)).collect()
}

/// The indices of the constraints that are still pending (eigenvector constraints waiting for
/// a Hessian, see [`Constraint::is_pending`]).
pub fn pending_constraints(constraints: &[Constraint]) -> Vec<usize> {
    constraints
        .iter()
        .enumerate()
        .filter_map(|(k, c)| c.is_pending().then_some(k))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, Position, SpinCharge};
    use crate::pes::constraint::ConstraintSet;

    /// Five atoms without symmetry.
    fn input() -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::C, Element::H, Element::O, Element::N, Element::H],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.0],
                [1.1, 0.1, -0.2],
                [-0.4, 1.3, 0.2],
                [0.3, -0.5, 1.2],
                [-1.0, -0.6, -0.7],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    /// `sys` with every atom moved (not rigidly).
    fn moved(sys: &MolecularSystem) -> MolecularSystem {
        let mut out = sys.clone();
        for (k, a) in sys.atoms().enumerate() {
            let p = sys.position(a).value();
            let s = 0.07 * (k as f64 + 1.0);
            out.set_position(a, Position::new([p[0] + s, p[1] - 0.5 * s, p[2] + 0.3 * s]));
        }
        out
    }

    fn all_kinds() -> Vec<Constraint> {
        let mut c = parse_projection_constraints(&s(&[
            "bond",
            "1,2",
            "angle",
            "2,1,3",
            "dihedral",
            "2,1,3,4",
            "x",
            "5",
            "rot",
            "1-3",
            "atoms_pair",
            "4,5",
            "eigvec",
            "0",
        ]))
        .unwrap();
        c.push(Constraint::at_start(ConstraintKind::Frozen(vec![
            AtomId::new(3),
            AtomId::new(4),
        ])));
        c.push(user_constraint("dist(1,4) - dist(2,5)").unwrap());
        c
    }

    #[test]
    fn resolving_twice_is_resolving_once() {
        let sys = input();
        let once = resolve_for_run(&all_kinds(), &sys).unwrap();
        let twice = resolve_for_run(&once, &moved(&sys)).unwrap();
        assert_eq!(once, twice);
        // Only the eigenvector constraint waits for a Hessian.
        assert_eq!(pending_constraints(&once), vec![6]);
        assert!(once
            .iter()
            .enumerate()
            .all(|(k, c)| k == 6 || c.is_resolved()));
        // With a Hessian it resolves, and then stays.
        let h = faer::Mat::from_fn(sys.dof(), sys.dof(), |i, j| {
            if i == j {
                1.0 + i as f64
            } else {
                0.01 * ((i + j) as f64).sin()
            }
        });
        let eig = once[6].resolve(&sys, Some(&h)).unwrap();
        assert!(eig.is_resolved() && !eig.is_pending());
        assert_eq!(
            eig.resolve(&moved(&sys), Some(&h.transpose().to_owned()))
                .unwrap(),
            eig
        );
    }

    #[test]
    fn explicit_targets_pass_through() {
        let manual = parse_projection_constraints(&s(&[
            "manual", "bond", "1,2", "1.7", "manual", "angle", "2,1,3", "95",
        ]))
        .unwrap();
        let resolved = resolve_for_run(&manual, &input()).unwrap();
        assert_eq!(resolved, manual);
        assert!((resolved[0].target.unwrap() - 1.7 * Units::ANGSTROM_TO_BOHR).abs() < 1e-14);
    }

    #[test]
    fn a_set_built_elsewhere_restores_onto_the_targets_of_the_input() {
        let sys = input();
        let kinds: Vec<Constraint> = all_kinds()
            .into_iter()
            .filter(|c| !matches!(c.kind, ConstraintKind::EigenVector(_)))
            .collect();
        let resolved = resolve_for_run(&kinds, &sys).unwrap();
        let other = moved(&sys);
        let set = ConstraintSet::new(&resolved, &other, None).unwrap();
        // At the input itself every residual vanishes; at the other structure they do not.
        let at_input = set.residuals(&sys).unwrap();
        assert!(at_input.iter().all(|r| r.abs() < 1e-12), "{at_input:?}");
        assert!(set
            .residuals(&other)
            .unwrap()
            .iter()
            .any(|r| r.abs() > 1e-3));
        let back = set.restore(&other).unwrap();
        // Frozen atoms return to the input positions.
        for a in [3, 4] {
            let (p, q) = (
                back.position(AtomId::new(a)).value(),
                sys.position(AtomId::new(a)).value(),
            );
            assert!((0..3).all(|k| (p[k] - q[k]).abs() < 1e-9), "atom {a}");
        }
        // Distances, angles, and dihedrals are those of the input.
        let d = |s: &MolecularSystem, a: usize, b: usize| {
            s.distance(AtomId::new(a), AtomId::new(b)).value()
        };
        assert!((d(&back, 0, 1) - d(&sys, 0, 1)).abs() < 1e-9);
        let fresh = ConstraintSet::new(&kinds, &sys, None).unwrap();
        let r = fresh.residuals(&back).unwrap();
        assert!(r.iter().all(|v| v.abs() < 1e-8), "{r:?}");
        // The same constraints unresolved hold the values of the other structure instead.
        let drifting = ConstraintSet::new(&kinds, &other, None).unwrap();
        assert!(drifting
            .residuals(&other)
            .unwrap()
            .iter()
            .all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn fixed_directions_break_the_rigid_body_invariance() {
        let sys = input();
        let pair = parse_projection_constraints(&s(&["atoms_pair", "1,2"])).unwrap();
        assert!(crate::pes::constraint::breaks_rigid_body(&pair));
        assert!(ConstraintSet::new(&pair, &sys, None)
            .unwrap()
            .breaks_rigid_body());
        let eig = parse_projection_constraints(&s(&["eigvec", "0"])).unwrap();
        assert!(crate::pes::constraint::breaks_rigid_body(&eig));
        let h = faer::Mat::from_fn(sys.dof(), sys.dof(), |i, j| {
            if i == j {
                1.0 + i as f64
            } else {
                0.0
            }
        });
        assert!(ConstraintSet::new(&eig, &sys, Some(&h))
            .unwrap()
            .breaks_rigid_body());
        let bond = parse_projection_constraints(&s(&["bond", "1,2"])).unwrap();
        assert!(!crate::pes::constraint::breaks_rigid_body(&bond));
    }

    #[test]
    fn a_dihedral_constraint_at_a_collinear_structure_names_its_atoms() {
        let line = MolecularSystem::new(
            vec![Element::C; 4],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.5, 1.2, 0.0],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let c = parse_projection_constraints(&s(&["dihedral", "1,2,3,4"])).unwrap();
        let err = ConstraintSet::new(&c, &line, None).unwrap_err().to_string();
        assert!(
            err.contains("1-2-3-4") && err.contains("collinear"),
            "{err}"
        );
    }

    #[test]
    fn the_angle_constraint_is_accurate_near_180_degrees() {
        // Atoms 1-2-3 with atom 3 at a small height y above the line: theta = pi - atan(y / l).
        let (l, y) = (1.3_f64, 2e-9_f64);
        let sys = MolecularSystem::new(
            vec![Element::C; 3],
            &positions::<crate::domain::Bohr>(&[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [l, y, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let c = parse_projection_constraints(&s(&["manual", "angle", "1,2,3", "170"])).unwrap();
        let set = ConstraintSet::new(&c, &sys, None).unwrap();
        let theta = set.residuals(&sys).unwrap()[0] + 170f64.to_radians();
        let want = std::f64::consts::PI - (y / l).atan();
        assert!((theta - want).abs() < 1e-14, "{theta} {want}");
        // d theta / d y_3 = -l / (l^2 + y^2).
        let jac = set.frame(&sys, &faer::Col::zeros(9)).unwrap().jacobian;
        let want = -l / (l * l + y * y);
        assert!((jac[(0, 7)] - want).abs() < 1e-9, "{} {want}", jac[(0, 7)]);
    }

    #[test]
    fn the_items_of_vc_fix_are_the_expressions_of_uc() {
        let exprs = |item: &str| -> Vec<String> {
            cell_fix_expressions(item)
                .unwrap()
                .into_iter()
                .map(|(_, e)| e)
                .collect()
        };
        assert_eq!(exprs("a"), ["cell_a()"]);
        assert_eq!(exprs("A=10.5"), ["cell_a() = 10.5*ang"]);
        assert_eq!(exprs("gamma=90"), ["cell_gamma() = 90*deg"]);
        assert_eq!(exprs("volume=270.5"), ["volume() = 270.5*ang^3"]);
        assert_eq!(exprs("shape").len(), 5);
        assert_eq!(exprs("c_vector"), ["cell_cx()", "cell_cy()", "cell_cz()"]);
        assert_eq!(exprs("bz=0"), ["cell_bz() = 0*ang"]);
        for bad in ["shape=1", "c_vector=2", "d", "volume=x", "ad"] {
            assert!(cell_fix_expressions(bad).is_err(), "{bad}");
        }
        // The targets are those of -uc with the same text.
        let a = cell_fix_constraints(&s(&["a=10.5,gamma"])).unwrap();
        let b = user_constraint("cell_a() = 10.5*ang").unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].target, b.target);
        assert!((b.target.unwrap() - 10.5 * Units::ANGSTROM_TO_BOHR).abs() < 1e-15);
    }

    #[test]
    fn a_user_function_takes_a_period_and_a_tolerance() {
        let c = user_constraint("dist(1,2); tolerance = 1e-8; period = 2*pi").unwrap();
        let (tolerance, period) = match &c.kind {
            ConstraintKind::Function(f) => (f.tolerance, f.period),
            _ => (None, None),
        };
        assert_eq!(tolerance, Some(1e-8));
        assert!((period.unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-15);
        for bad in [
            "dist(1,2); tolerance = 0",
            "dist(1,2); tolerance = 1e-8; tolerance = 1e-9",
            "dist(1,2); step = 1",
        ] {
            assert!(user_constraint(bad).is_err(), "{bad}");
        }
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_auto_and_manual_constraints() {
        let auto =
            parse_projection_constraints(&s(&["bond", "1,2", "fbond", "1,2,3,4,2", "eigvec", "0"]))
                .unwrap();
        assert_eq!(auto.len(), 3);
        assert_eq!(
            auto[1].kind,
            ConstraintKind::Distance(
                vec![AtomId::new(0), AtomId::new(1)],
                vec![AtomId::new(2), AtomId::new(3)]
            ),
            "the size of the first fragment is not an atom"
        );
        let manual =
            parse_projection_constraints(&s(&["manual", "angle", "1,2,3", "109.5"])).unwrap();
        assert!((manual[0].target.unwrap() - 109.5f64.to_radians()).abs() < 1e-14);
        assert!(parse_projection_constraints(&s(&["bond", "1,2,3"])).is_err());
        assert!(parse_projection_constraints(&s(&["manual", "rot", "1-3", "1.0"])).is_err());
        assert!(parse_projection_constraints(&s(&["torsion", "1,2,3,4"])).is_err());
        assert!(parse_projection_constraints(&s(&["fbond", "1,2,3,4,4"])).is_err());
    }
}
