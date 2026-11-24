//! The scanned coordinates of `relaxedscan` (`-scan KIND ATOMS START,END ...`): their kinds,
//! names, units, and the constraints that hold them at a value.
//!
//! The groups of three are the kind of coordinate, its atoms (1-based, as `-pc`), and the
//! range in the units of `-pc manual` (Angstrom for `bond`, `fbond`, `x`, `y`, `z`; degrees
//! for `angle`, `dihedral`). mopt-rs adds `expr` (an expression of the `-ubias` language in
//! atomic units in place of the atoms, as `-uc`), `var` (a variable of an objective of plain
//! variables, by its number from 1 or its name), `resper` (the ReSPer coordinate PCo1 or
//! PCo2 of the references of `-resper`), and `cell` (with `-vc`: a parameter of the cell by
//! its name in `-vc_fix`, `a`, `b`, `c` and `ax`, ..., `cz` in Angstrom, `alpha`, `beta`,
//! `gamma` in degrees, `volume` in Angstrom^3; e.g. `-scan cell volume 250,330` for an
//! energy-volume curve with the atoms and the shape of the cell relaxed at every volume).
//!
//! MultiOptPy: `Entrypoints/relaxed_scan.py:run_relaxedscan` (`make_scan_tgt`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::constraints::{
    cell_fix_expressions, projection_constraint, user_constraint, user_constraint_named,
};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::pes::constraint::Constraint;

/// What a scanned coordinate is.
#[derive(Debug, Clone, PartialEq)]
pub enum TargetKind {
    /// A coordinate of `-pc manual` (`bond`, `fbond`, `angle`, `dihedral`, `x`, `y`, `z`).
    Geometric {
        /// Its kind as `-pc` names it.
        name: String,
        /// Its atoms as given.
        atoms: String,
    },
    /// An expression of the `-ubias` language (atomic units).
    Expression(String),
    /// A plain variable, by number (1-based) or name.
    Variable(String),
    /// A ReSPer coordinate (0-based: PCo1 is 0).
    Resper(usize),
    /// A parameter of the cell, by its name in `-vc_fix` (`a`, `alpha`, `volume`, `cz`, ...).
    Cell(String),
}

/// The names of the parameters of the cell that a scan takes (one value each).
const CELL_TARGETS: &str = "a, b, c, alpha, beta, gamma, volume, ax, ay, ..., cz";

/// The `-uc` expression and the name of the constraint that hold the parameter `name` of the
/// cell at `value` (units of the command line), as `-vc_fix name=value` does.
fn cell_expression(name: &str, value: f64) -> MoptResult<(String, String)> {
    let mut held = cell_fix_expressions(&format!("{name}={value}"))?;
    match (held.pop(), held.is_empty()) {
        (Some(one), true) => Ok(one),
        _ => Err(MoptError::parse(format!(
            "-scan cell {name}: a scan takes one parameter of the cell ({CELL_TARGETS})"
        ))),
    }
}

/// One scanned coordinate with its range.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanTarget {
    /// What it is.
    pub kind: TargetKind,
    /// First value (units of the command line).
    pub start: f64,
    /// Last value.
    pub end: f64,
}

impl ScanTarget {
    /// The kind as MultiOptPy's energy profile names it (`bond`, ...).
    pub fn kind_name(&self) -> String {
        match &self.kind {
            TargetKind::Geometric { name, .. } => name.to_ascii_lowercase(),
            TargetKind::Expression(_) => "expr".into(),
            TargetKind::Variable(_) => "var".into(),
            TargetKind::Resper(_) => "resper".into(),
            TargetKind::Cell(_) => "cell".into(),
        }
    }

    /// The atoms (or what stands in their place) as given.
    pub fn atoms_text(&self) -> String {
        match &self.kind {
            TargetKind::Geometric { atoms, .. } => atoms.clone(),
            TargetKind::Expression(e) => e.clone(),
            TargetKind::Variable(v) | TargetKind::Cell(v) => v.clone(),
            TargetKind::Resper(k) => (k + 1).to_string(),
        }
    }

    /// The factor from the units of the command line to those of the constraints (Bohr,
    /// radian, Bohr^3; 1 for expressions, variables, and ReSPer coordinates).
    pub fn to_atomic(&self) -> f64 {
        let length = Units::ANGSTROM_TO_BOHR;
        match &self.kind {
            TargetKind::Geometric { name, .. } => match name.to_ascii_lowercase().as_str() {
                "angle" | "dihedral" => Units::DEG_TO_RAD,
                _ => length,
            },
            TargetKind::Cell(name) => match name.as_str() {
                "alpha" | "beta" | "gamma" => Units::DEG_TO_RAD,
                "volume" => length * length * length,
                _ => length,
            },
            _ => 1.0,
        }
    }

    /// The name of the coordinate in tables and figures, such as `bond(1,2)` or `PCo1`.
    pub fn label(&self) -> String {
        match &self.kind {
            TargetKind::Geometric { name, atoms } => {
                format!("{}({atoms})", name.to_ascii_lowercase())
            }
            TargetKind::Expression(e) => e.clone(),
            TargetKind::Variable(v) => v.clone(),
            TargetKind::Resper(k) => format!("PCo{}", k + 1),
            TargetKind::Cell(name) => format!("cell {name}"),
        }
    }

    /// The unit of the values (`plain`: an objective of plain variables).
    pub fn unit(&self, plain: bool) -> &'static str {
        match &self.kind {
            TargetKind::Geometric { name, .. } => match name.to_ascii_lowercase().as_str() {
                "angle" | "dihedral" => "degree",
                _ => "Angstrom",
            },
            TargetKind::Expression(_) => "a.u.",
            TargetKind::Variable(_) => "",
            TargetKind::Resper(_) if plain => "",
            TargetKind::Resper(_) => "Angstrom amu^0.5",
            TargetKind::Cell(name) => match name.as_str() {
                "alpha" | "beta" | "gamma" => "degree",
                "volume" => "Angstrom^3",
                _ => "Angstrom",
            },
        }
    }

    /// The constraint that holds a geometric coordinate, an expression, or a parameter of the
    /// cell at `value` (units of the command line); `None` for variables and ReSPer
    /// coordinates.
    pub fn constraint(&self, value: f64) -> MoptResult<Option<Constraint>> {
        match &self.kind {
            TargetKind::Geometric { name, atoms } => {
                projection_constraint(name, atoms, Some(value)).map(Some)
            }
            TargetKind::Expression(e) => user_constraint(&format!("{e} = {value}")).map(Some),
            TargetKind::Cell(name) => {
                let (label, expr) = cell_expression(name, value)?;
                user_constraint_named(&expr, &label).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// The tokens of `-pc manual` or `-uc` that hold it at `value`, for the settings of the
    /// optimization of a point.
    pub fn option_tokens(&self, value: f64) -> Option<(&'static str, Vec<String>)> {
        match &self.kind {
            TargetKind::Geometric { name, atoms } => Some((
                "projection_constrain",
                vec![
                    "manual".into(),
                    name.clone(),
                    atoms.clone(),
                    value.to_string(),
                ],
            )),
            TargetKind::Expression(e) => Some(("user_constraint", vec![format!("{e} = {value}")])),
            // The same constraint as -vc_fix NAME=VALUE.
            TargetKind::Cell(name) => Some(("variable_cell_fix", vec![format!("{name}={value}")])),
            _ => None,
        }
    }
}

fn range(text: &str) -> MoptResult<(f64, f64)> {
    let parts: Vec<&str> = text.split(',').collect();
    let numbers: Vec<f64> = parts
        .iter()
        .map(|p| p.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| MoptError::parse(format!("-scan: the range '{text}' is not START,END")))?;
    match numbers.as_slice() {
        [a, b] if a.is_finite() && b.is_finite() => Ok((*a, *b)),
        _ => Err(MoptError::parse(format!(
            "-scan: the range '{text}' is not START,END"
        ))),
    }
}

/// The coordinates of `-scan` (groups of KIND ATOMS START,END).
pub fn parse_targets(tokens: &[String]) -> MoptResult<Vec<ScanTarget>> {
    if tokens.is_empty() {
        return Err(MoptError::parse(
            "relaxedscan needs -scan KIND ATOMS START,END (e.g. -scan bond 1,2 1.0,2.0)",
        ));
    }
    if !tokens.len().is_multiple_of(3) {
        return Err(MoptError::parse(
            "-scan takes groups of three: KIND ATOMS START,END",
        ));
    }
    tokens
        .chunks(3)
        .map(|c| {
            let (start, end) = range(&c[2])?;
            let kind = match c[0].to_ascii_lowercase().as_str() {
                "expr" | "uc" => TargetKind::Expression(c[1].clone()),
                "cell" => {
                    let name = c[1].trim().to_ascii_lowercase();
                    // Checked here, with a value, as the points will hold it.
                    cell_expression(&name, start)?;
                    TargetKind::Cell(name)
                }
                "var" | "variable" => TargetKind::Variable(c[1].trim().to_string()),
                "resper" | "pco" => {
                    let k: usize = c[1].trim().parse().map_err(|_| {
                        MoptError::parse(format!(
                            "-scan resper: the coordinate is 1 or 2 (PCo1, PCo2), got '{}'",
                            c[1]
                        ))
                    })?;
                    if !(1..=2).contains(&k) {
                        return Err(MoptError::parse(
                            "-scan resper: the coordinate is 1 or 2 (PCo1, PCo2)",
                        ));
                    }
                    TargetKind::Resper(k - 1)
                }
                name => {
                    // Checked here, with a value, as the points will hold it.
                    projection_constraint(name, &c[1], Some(start))?;
                    TargetKind::Geometric {
                        name: name.to_string(),
                        atoms: c[1].clone(),
                    }
                }
            };
            let target = ScanTarget { kind, start, end };
            if let TargetKind::Expression(_) = target.kind {
                target.constraint(start)?;
            }
            Ok(target)
        })
        .collect()
}
