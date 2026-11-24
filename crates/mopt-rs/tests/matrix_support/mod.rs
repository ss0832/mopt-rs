//! Shared parts of the constraint and bias matrices of `confsearch`, `orientsearch`, and
//! `relaxedscan`: the constraint options of every kind, the constraints they hold resolved at
//! the input, and the checks of the structures the commands write.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(dead_code, clippy::unwrap_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::cli::constraints::resolve_for_run;
use mopt_rs::domain::{AtomId, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use mopt_rs::pes::TermContext;

/// The bound on a residual recomputed from a structure file: the guarantee every hard
/// constraint is held to. The xyz files have 12 decimals of Angstrom (a rounding of up to
/// 9.4e-13 Bohr per coordinate, a few 1e-12 on a distance or an angle).
pub const FROM_FILES: f64 = 1e-10;
/// The bound on a residual recomputed from a file of a variable cell: the guarantee as well
/// (the lattice is written with 15 decimals of Angstrom, `PeriodicCell::to_extxyz_header`).
pub const FROM_VARIABLE_CELL_FILES: f64 = 1e-10;
/// The bound on a frozen coordinate read back from a file (the rounding of 12 decimals).
pub const FROZEN_IN_FILES: f64 = 2e-12;
/// The bound on a fractional coordinate of a frozen atom in a variable cell read back from a
/// file (12 decimals of the positions, 15 of the lattice).
pub const FRACTIONAL_IN_FILES: f64 = 1e-12;

/// A fresh directory for a test.
pub fn scratch(prefix: &str, name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_{prefix}_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `text` written to `dir/name`.
pub fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p
}

/// The structure of an xyz file.
pub fn read(path: &Path) -> MolecularSystem {
    read_xyz_file(path, SpinCharge::default(), None).unwrap()
}

/// Every frame of an xyz file.
pub fn frames(path: &Path) -> Vec<MolecularSystem> {
    read_xyz_trajectory_file(path, SpinCharge::default(), None)
        .unwrap()
        .frames
        .into_iter()
        .map(|f| f.system)
        .collect()
}

/// The constraint kinds of the matrices as options, for the atoms `a`, `b`, `c`, `d` (from 1)
/// and two fragments: (name, options).
pub fn kinds(
    [a, b, c, d]: [usize; 4],
    fragments: (&[usize], &[usize]),
) -> Vec<(&'static str, Vec<String>)> {
    let list = |v: &[usize]| {
        v.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut fbond: Vec<usize> = fragments.0.to_vec();
    fbond.extend_from_slice(fragments.1);
    fbond.push(fragments.0.len());
    vec![
        ("frozen", vec!["-fix".into(), format!("{d}")]),
        (
            "bond",
            vec!["-pc".into(), "bond".into(), format!("{a},{b}")],
        ),
        ("fbond", vec!["-pc".into(), "fbond".into(), list(&fbond)]),
        (
            "angle",
            vec!["-pc".into(), "angle".into(), format!("{a},{b},{c}")],
        ),
        ("gfix", vec!["-gfix".into(), format!("{a},{b},{c}")]),
        (
            "uc",
            vec!["-uc".into(), format!("dist({a},{c}) - dist({b},{d})")],
        ),
    ]
}

/// The constraints `constraints` (as a command parsed them) resolved at `input`: those that
/// are not frozen atoms as a set, and the frozen atoms (from 0).
pub fn held(constraints: &[Constraint], input: &MolecularSystem) -> (ConstraintSet, Vec<usize>) {
    let resolved = resolve_for_run(constraints, input).unwrap();
    let mut frozen = Vec::new();
    let mut others = Vec::new();
    for c in resolved {
        match &c.kind {
            ConstraintKind::Frozen(a) => frozen.extend(a.iter().map(|x| x.index())),
            _ => others.push(c),
        }
    }
    (ConstraintSet::new(&others, input, None).unwrap(), frozen)
}

/// Every structure of `structures` holds the constraints at their values in `input` (within
/// `bound`), its frozen atoms where they are in `input` (within `frozen_bound`, or their
/// fractional coordinates within [`FRACTIONAL_IN_FILES`] with `fractional`).
pub fn check(
    cell: &str,
    structures: &[MolecularSystem],
    constraints: &[Constraint],
    input: &MolecularSystem,
    bound: f64,
    frozen_bound: f64,
    fractional: bool,
) {
    assert!(!structures.is_empty(), "{cell}: no structures");
    let (set, frozen) = held(constraints, input);
    let mut largest = 0.0_f64;
    for (k, sys) in structures.iter().enumerate() {
        assert_eq!(sys.n_atoms(), input.n_atoms(), "{cell}");
        let r = set.residuals(sys).unwrap();
        largest = r.iter().fold(largest, |a, v| a.max(v.abs()));
        assert!(
            r.iter().all(|v| v.abs() <= bound),
            "{cell} structure {k}: {r:?}"
        );
        for &atom in &frozen {
            let id = AtomId::new(atom);
            if fractional {
                let s = sys
                    .active_cell()
                    .unwrap()
                    .cart_to_frac(sys.position(id).value());
                let s0 = input
                    .active_cell()
                    .unwrap()
                    .cart_to_frac(input.position(id).value());
                for c in 0..3 {
                    assert!(
                        (s[c] - s0[c]).abs() <= FRACTIONAL_IN_FILES,
                        "{cell} structure {k}: {s:?} {s0:?}"
                    );
                }
            } else {
                let (p, q) = (sys.position(id).value(), input.position(id).value());
                for c in 0..3 {
                    assert!(
                        (p[c] - q[c]).abs() <= frozen_bound,
                        "{cell} structure {k}: atom {} at {p:?}, input {q:?}",
                        atom + 1
                    );
                }
            }
        }
    }
    eprintln!(
        "RESIDUAL {cell} (bound {bound:.0e}, {} structures): largest {largest:.2e}",
        structures.len()
    );
}

/// The distances between every pair of `atoms` (from 1) of `sys`.
pub fn distances(sys: &MolecularSystem, atoms: &[usize]) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, &a) in atoms.iter().enumerate() {
        for &b in &atoms[i + 1..] {
            out.push(sys.distance(AtomId::new(a - 1), AtomId::new(b - 1)).value());
        }
    }
    out
}

/// The bias options of the matrices for the atoms `a`, `b` (from 1) of `input`: AFIR (`-ma`),
/// a restraint (`-kp`), and an absolute-position bias on `a` pulling it to a point 0.6 Bohr
/// away from where it is: (name, flag, arguments, the point of the absolute bias).
pub fn biases(
    input: &MolecularSystem,
    a: usize,
    b: usize,
) -> Vec<(&'static str, String, Vec<String>, Option<[f64; 3]>)> {
    let p = input.position(AtomId::new(a - 1)).value();
    let goal = [p[0] + 0.5, p[1] - 0.3, p[2] + 0.2];
    vec![
        (
            "afir",
            "-ma".into(),
            vec!["50".into(), format!("{a}"), format!("{b}")],
            None,
        ),
        (
            "restraint",
            "-kp".into(),
            vec!["0.05".into(), "1.70".into(), format!("{a},{b}")],
            None,
        ),
        (
            "absolute",
            "-ubias".into(),
            vec![format!(
                "0.2 * ((x({a}) - {:.10})^2 + (y({a}) - {:.10})^2 + (z({a}) - {:.10})^2)",
                goal[0], goal[1], goal[2]
            )],
            Some(goal),
        ),
    ]
}

/// Asserts that atom `atom` (from 1) of `sys` is within `tolerance` (Bohr) of `goal`, up to a
/// lattice vector in a periodic cell (where the absolute bias sees the nearest image).
pub fn at_point(cell: &str, sys: &MolecularSystem, atom: usize, goal: [f64; 3], tolerance: f64) {
    let q = sys.position(AtomId::new(atom - 1)).value();
    let mut d: [f64; 3] = std::array::from_fn(|c| q[c] - goal[c]);
    if let Some(lattice) = sys.active_cell() {
        let f = lattice.cart_to_frac(d);
        d = lattice.frac_to_cart(f.map(|x| x - x.round()));
    }
    assert!(
        d.iter().all(|v| v.abs() < tolerance),
        "{cell}: atom at {q:?}, pulled to {goal:?} (Bohr)"
    );
}

/// The energy of the bias `flag arguments` at `sys` (Hartree).
pub fn bias_energy(flag: &str, arguments: &[String], sys: &MolecularSystem) -> f64 {
    let ctx = TermContext::at_iteration(0, sys);
    parse_bias_option(flag, arguments)
        .unwrap()
        .iter()
        .map(|spec| {
            spec.build(sys)
                .unwrap()
                .evaluate(sys, &ctx, false)
                .unwrap()
                .energy
                .value()
        })
        .sum()
}

/// The columns of a CSV table by name, and its rows.
pub fn csv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(str::to_string)
        .collect();
    // Quoted fields come last in these tables; split the leading ones only.
    let rows = lines
        .map(|l| {
            l.splitn(header.len(), ',')
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect();
    (header, rows)
}

/// The column `name` of a header.
pub fn column(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("no column {name} in {header:?}"))
}
