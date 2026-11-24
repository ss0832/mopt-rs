//! The analytic Hessian of GFN-FF against differences of its analytic gradient: the molecules
//! of `tests/golden/gfnff.json` and the cells of `tests/golden/gfnff_pbc_reference.json`, and
//! the Hessian of a cell in the positions and the lattice vectors against differences of the
//! gradient and of the derivative with respect to the lattice vectors (from the stress). The
//! differences are the five-point stencil `(8 (g(+h) - g(-h)) - (g(+2h) - g(-2h))) / 12h`
//! with `h = 1e-3` Bohr, over every coordinate of the small structures and an even sample of
//! those of the large ones; each case prints its largest deviation.
//!
//! A column that differs by more than [`REFINE`] is taken again with `h = 2.5e-4`, `1e-5`, and
//! `1e-6` Bohr, and the closest of the four counts. Two things need it, both of the
//! energy rather than of the Hessian:
//! - xtb's hydrogen-bond terms drop to zero where their out-of-line exponent passes 15, so the
//!   gradient has jumps of about 1e-10 Hartree/Bohr; a stencil across one is off by that over
//!   `12 h` (ice, the benzene crystal, taxol: 2e-8 to 1e-7 at `h = 1e-3`, 1e-13 once the
//!   stencil no longer crosses it);
//! - xtb's topology of the hexacyanides of Fe and Co bonds N to the metal, and the bend at N
//!   between C and the metal (theta = 0.01 and 0.1 degrees, theta_0 = 180 degrees) is at the
//!   cusp of `(theta - theta_0)^2` at `theta = 0`: its curvature (1e2 Hartree/Bohr^2) changes
//!   over `r theta`, 2.5e-4 Bohr for Fe and 2e-3 Bohr for Co, and the differences reach the
//!   analytic value only with steps below that (`1e-6` and `1e-5` Bohr).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Bohr, Element, MolecularSystem, PeriodicCell, SpinCharge, Vector3};
use mopt_rs::pes::calculator::Calculator;
use mopt_rs::pes::native::gfnff::{GfnffTopology, HbCutoff};
use mopt_rs::pes::native::Gfnff;

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn doc(name: &str) -> Vec<JsonValue> {
    let path = format!("{}/../../tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    let d = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    d.get("cases").unwrap().as_array().unwrap().to_vec()
}

fn elements(case: &JsonValue) -> Vec<Element> {
    case.get("symbols")
        .unwrap()
        .as_string_vec()
        .unwrap()
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect()
}

fn multiplicity(el: &[Element], charge: i32) -> u32 {
    if (el.iter().map(|e| e.atomic_number() as i32).sum::<i32>() - charge) % 2 == 0 {
        1
    } else {
        2
    }
}

fn molecule(case: &JsonValue) -> MolecularSystem {
    let el = elements(case);
    let x = floats(case.get("positions_bohr").unwrap());
    let charge = case.get("charge").unwrap().as_f64().unwrap() as i32;
    let multiplicity = multiplicity(&el, charge);
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge {
            charge,
            multiplicity,
        },
        None,
    )
    .unwrap()
}

fn crystal(case: &JsonValue) -> MolecularSystem {
    let el = elements(case);
    let x = floats(case.get("positions_bohr").unwrap());
    let l: Vec<Vec<f64>> = case
        .get("lattice_bohr")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(floats)
        .collect();
    let v = |k: usize| Vector3::<Bohr>::new([l[k][0], l[k][1], l[k][2]]);
    let cell = PeriodicCell::from_lattice_vectors(v(0), v(1), v(2), [true; 3]).unwrap();
    let charge = case.get("charge").unwrap().as_f64().unwrap() as i32;
    let multiplicity = multiplicity(&el, charge);
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge {
            charge,
            multiplicity,
        },
        Some(cell),
    )
    .unwrap()
}

/// The coordinates whose columns are compared: all up to `limit`, else `limit` spread evenly.
fn columns(dim: usize, limit: usize) -> Vec<usize> {
    if dim <= limit {
        return (0..dim).collect();
    }
    (0..limit).map(|k| k * dim / limit).collect()
}

/// Column `k` of the Hessian from the five-point stencil of the gradient with the step `h`.
fn difference_column(
    topo: &GfnffTopology,
    sys: &MolecularSystem,
    k: usize,
    cut: HbCutoff,
    h: f64,
) -> Col<f64> {
    let x0 = sys.coords_flat_bohr();
    let g = |s: f64| {
        let mut x = x0.clone();
        x[k] += s * h;
        topo.compute_with(&sys.with_flat_coords_bohr(&x).unwrap(), cut)
            .unwrap()
            .gradient
    };
    let (p1, m1, p2, m2) = (g(1.0), g(-1.0), g(2.0), g(-2.0));
    Col::from_fn(x0.nrows(), |i| {
        (8.0 * (p1[i] - m1[i]) - (p2[i] - m2[i])) / (12.0 * h)
    })
}

/// A column farther than this (Hartree/Bohr^2) from the differences of `h = 1e-3` is taken
/// again with smaller steps.
const REFINE: f64 = 1.0e-9;

/// Tolerance on the deviation relative to the largest element of the Hessian (at least 1
/// Hartree/Bohr^2).
const TOLERANCE: f64 = 1.0e-7;

/// What the comparison of one structure found.
struct Comparison {
    /// The largest deviation of the analytic Hessian from the differences over the columns.
    worst: f64,
    /// The largest element of the analytic Hessian.
    largest: f64,
    /// The columns taken again with smaller steps.
    refined: usize,
    /// The largest asymmetry of the analytic Hessian.
    asymmetry: f64,
}

fn deviation(sys: &MolecularSystem, cut: HbCutoff, cols: &[usize]) -> Comparison {
    let calc = Gfnff::default().with_hb_cutoff(cut);
    let topo = calc.topology(sys).unwrap();
    let h = topo.hessian_with(sys, cut).unwrap();
    let dim = h.nrows();
    let column_deviation = |fd: &Col<f64>, k: usize| {
        (0..dim)
            .map(|i| (h[(i, k)] - fd[i]).abs())
            .fold(0.0, f64::max)
    };
    let mut worst = 0.0_f64;
    let mut refined = 0;
    for &k in cols {
        let mut dev = column_deviation(&difference_column(&topo, sys, k, cut, 1.0e-3), k);
        if dev > REFINE {
            refined += 1;
            for step in [2.5e-4, 1.0e-5, 1.0e-6] {
                dev = dev.min(column_deviation(
                    &difference_column(&topo, sys, k, cut, step),
                    k,
                ));
            }
        }
        worst = worst.max(dev);
    }
    let pairs = || (0..dim).flat_map(|i| (0..dim).map(move |j| (i, j)));
    Comparison {
        worst,
        largest: pairs().map(|(i, j)| h[(i, j)].abs()).fold(0.0, f64::max),
        refined,
        asymmetry: pairs()
            .map(|(i, j)| (h[(i, j)] - h[(j, i)]).abs())
            .fold(0.0, f64::max),
    }
}

#[test]
fn analytic_hessian_of_the_molecules_matches_differences_of_the_gradient() {
    let mut failures = Vec::new();
    let mut overall = 0.0_f64;
    for case in doc("gfnff.json") {
        let name = case.get("molecule").unwrap().as_str().unwrap().to_string();
        let sys = molecule(&case);
        let dim = 3 * sys.n_atoms();
        let cols = columns(dim, if sys.n_atoms() > 50 { 12 } else { 24 });
        for cut in [HbCutoff::default(), HbCutoff::Sharp] {
            let c = deviation(&sys, cut, &cols);
            eprintln!(
                "{name} ({} atoms, {cut:?}, {} of {dim} columns, {} refined): max |H - FD| = \
                 {:.3e} (largest |H| {:.3e})",
                sys.n_atoms(),
                cols.len(),
                c.refined,
                c.worst,
                c.largest
            );
            overall = overall.max(c.worst);
            if c.worst > TOLERANCE * c.largest.max(1.0) || c.asymmetry > 0.0 || !c.worst.is_finite()
            {
                failures.push(format!("{name} {cut:?}: {:.3e}", c.worst));
            }
        }
    }
    eprintln!("molecules: largest deviation {overall:.3e}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn analytic_hessian_of_the_cells_matches_differences_of_the_gradient() {
    let mut failures = Vec::new();
    let mut overall = 0.0_f64;
    for case in doc("gfnff_pbc_reference.json") {
        let name = case.get("name").unwrap().as_str().unwrap().to_string();
        let sys = crystal(&case);
        let dim = 3 * sys.n_atoms();
        let cols = columns(dim, 12);
        let c = deviation(&sys, HbCutoff::default(), &cols);
        eprintln!(
            "{name} ({} atoms, {} of {dim} columns, {} refined): max |H - FD| = {:.3e} \
             (largest |H| {:.3e})",
            sys.n_atoms(),
            cols.len(),
            c.refined,
            c.worst,
            c.largest
        );
        overall = overall.max(c.worst);
        if c.worst > TOLERANCE * c.largest.max(1.0) || c.asymmetry > 0.0 || !c.worst.is_finite() {
            failures.push(format!("{name}: {:.3e}", c.worst));
        }
    }
    eprintln!("cells: largest deviation {overall:.3e}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_calculator_hessian_is_analytic_unless_asked_for_differences() {
    let case = doc("gfnff.json")
        .into_iter()
        .find(|c| c.get("molecule").unwrap().as_str() == Some("pyridine_water"))
        .unwrap();
    let sys = molecule(&case);
    let analytic = Gfnff::default();
    assert!(analytic.analytic_hessian());
    assert!(analytic.description().contains("Hessian: analytic"));
    let mut numerical = Gfnff::default();
    numerical.analytic_hessian = false;
    assert!(!numerical.analytic_hessian());
    let (ha, hn) = (
        analytic.hessian(&sys).unwrap(),
        numerical.hessian(&sys).unwrap(),
    );
    let dim = ha.nrows();
    let worst = (0..dim)
        .flat_map(|i| (0..dim).map(move |j| (i, j)))
        .map(|(i, j)| (ha[(i, j)] - hn[(i, j)]).abs())
        .fold(0.0, f64::max);
    // Central differences with the default step of 5e-3 Bohr.
    assert!(worst < 1.0e-4, "{worst:.3e}");
}

#[test]
fn the_registry_reads_the_kind_of_hessian() {
    use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

    let case = doc("gfnff.json")
        .into_iter()
        .find(|c| c.get("molecule").unwrap().as_str() == Some("h2o"))
        .unwrap();
    let sys = molecule(&case);
    let reg = CalculatorRegistry::native();
    let with = |value: &str| CalculatorOptions {
        extra: vec![("hessian".into(), value.into())],
        ..CalculatorOptions::default()
    };
    let default = reg
        .create("gfnff", &CalculatorOptions::default(), &sys)
        .unwrap();
    assert!(default.analytic_hessian());
    assert!(reg
        .create("gfnff", &with("analytic"), &sys)
        .unwrap()
        .analytic_hessian());
    let numerical = reg.create("gfnff", &with("numerical"), &sys).unwrap();
    assert!(!numerical.analytic_hessian());
    assert!(numerical.description().contains("central differences"));
    let err = reg
        .create("gfnff", &with("exact"), &sys)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("analytic or numerical"), "{err}");
}

/// The gradient in the positions and the lattice vectors of a cell (`3N + 9`): the forces'
/// negatives and `dE/da = A^-T (sigma - X^T G)` from the strain derivative `sigma`.
fn cell_gradient(topo: &GfnffTopology, sys: &MolecularSystem) -> Vec<f64> {
    let r = topo.compute_with(sys, HbCutoff::default()).unwrap();
    let cell = sys.active_cell().unwrap();
    let volume = cell.volume_bohr3();
    let stress = r.stress.unwrap();
    let x = sys.coords_flat_bohr();
    let n = sys.n_atoms();
    // a[k][c]: the lattice vectors are the columns of h_bohr.
    let a: [[f64; 3]; 3] = std::array::from_fn(|kk| std::array::from_fn(|c| cell.h_bohr[c][kk]));
    let mut m = [[0.0; 3]; 3];
    for p in 0..3 {
        for q in 0..3 {
            m[p][q] = stress[p][q] * volume
                - (0..n)
                    .map(|i| x[3 * i + p] * r.gradient[3 * i + q])
                    .sum::<f64>();
        }
    }
    // Solve A^T G_a = m column by column (Cramer's rule on A^T).
    let at: [[f64; 3]; 3] = std::array::from_fn(|p| std::array::from_fn(|kk| a[kk][p]));
    let det = |b: &[[f64; 3]; 3]| {
        b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
            - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
            + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0])
    };
    let d = det(&at);
    let mut ga = [[0.0; 3]; 3];
    for q in 0..3 {
        for kk in 0..3 {
            let mut b = at;
            for p in 0..3 {
                b[p][kk] = m[p][q];
            }
            ga[kk][q] = det(&b) / d;
        }
    }
    let mut out: Vec<f64> = (0..3 * n).map(|k| r.gradient[k]).collect();
    for row in &ga {
        out.extend_from_slice(row);
    }
    out
}

/// `sys` with the variable `v` of the positions and lattice vectors moved by `s` (the
/// Cartesian positions kept when a lattice vector moves).
fn moved_cell(sys: &MolecularSystem, v: usize, s: f64) -> MolecularSystem {
    let n = sys.n_atoms();
    if v < 3 * n {
        let mut x = sys.coords_flat_bohr();
        x[v] += s;
        return sys.with_flat_coords_bohr(&x).unwrap();
    }
    let h = sys.active_cell().unwrap().h_bohr;
    let (kk, c) = ((v - 3 * n) / 3, (v - 3 * n) % 3);
    let vector = |l: usize| {
        let mut a = [h[0][l], h[1][l], h[2][l]];
        if l == kk {
            a[c] += s;
        }
        Vector3::<Bohr>::new(a)
    };
    let mut out = sys.clone();
    out.cell = Some(
        PeriodicCell::from_lattice_vectors(vector(0), vector(1), vector(2), [true; 3]).unwrap(),
    );
    out
}

#[test]
fn analytic_cell_hessian_matches_differences_of_the_gradient_and_the_stress() {
    let mut failures = Vec::new();
    let mut overall = 0.0_f64;
    let mut structures: Vec<(String, MolecularSystem)> = doc("gfnff_pbc_reference.json")
        .iter()
        .map(|case| {
            let name = case.get("name").unwrap().as_str().unwrap().to_string();
            (name, crystal(case))
        })
        .collect();
    // Cubic ice with two atoms given at other images (moved by lattice vectors): the topology
    // takes them back, and the lattice vectors move them with their images.
    let ice = structures
        .iter()
        .find(|(name, _)| name == "water_ice_ic")
        .map(|(_, s)| s.clone())
        .unwrap();
    let h = ice.active_cell().unwrap().h_bohr;
    let mut x = ice.coords_flat_bohr();
    for c in 0..3 {
        x[c] += h[c][0];
        x[9 + c] -= h[c][1] + h[c][2];
    }
    structures.push((
        "water_ice_ic_images".into(),
        ice.with_flat_coords_bohr(&x).unwrap(),
    ));
    for (name, sys) in structures {
        let n = sys.n_atoms();
        // The 2x1x1 benzene supercell repeats the benzene crystal at twice the cost.
        if n > 64 {
            continue;
        }
        let calc = Gfnff::default();
        let topo = calc.topology(&sys).unwrap();
        let h = topo.cell_hessian_with(&sys, HbCutoff::default()).unwrap();
        let dim = 3 * n + 9;
        assert_eq!(h.nrows(), dim);
        // The block of the positions is the Hessian of the positions.
        let hx = topo.hessian_with(&sys, HbCutoff::default()).unwrap();
        let largest = (0..dim)
            .flat_map(|i| (0..dim).map(move |j| (i, j)))
            .map(|(i, j)| h[(i, j)].abs())
            .fold(0.0, f64::max);
        let block = (0..3 * n)
            .flat_map(|i| (0..3 * n).map(move |j| (i, j)))
            .map(|(i, j)| (h[(i, j)] - hx[(i, j)]).abs())
            .fold(0.0, f64::max);
        // The lattice columns and a few of the positions.
        let mut cols: Vec<usize> = (3 * n..dim).collect();
        cols.extend(columns(3 * n, 3));
        let column_at = |v: usize, step: f64| {
            let g = |s: f64| cell_gradient(&topo, &moved_cell(&sys, v, s * step));
            let (p1, m1, p2, m2) = (g(1.0), g(-1.0), g(2.0), g(-2.0));
            (0..dim)
                .map(|i| {
                    (h[(i, v)] - (8.0 * (p1[i] - m1[i]) - (p2[i] - m2[i])) / (12.0 * step)).abs()
                })
                .fold(0.0, f64::max)
        };
        let mut worst = 0.0_f64;
        let mut refined = 0;
        for &v in &cols {
            let mut dev = column_at(v, 1.0e-3);
            if dev > REFINE {
                refined += 1;
                for step in [2.5e-4, 1.0e-5] {
                    dev = dev.min(column_at(v, step));
                }
            }
            worst = worst.max(dev);
        }
        eprintln!(
            "{name} ({n} atoms, {} columns, {refined} refined): max |H - FD| = {worst:.3e}, \
             positions block vs hessian_with {block:.1e} (largest |H| {largest:.3e})",
            cols.len()
        );
        overall = overall.max(worst);
        if worst > TOLERANCE * largest.max(1.0) || block > 1.0e-10 || !worst.is_finite() {
            failures.push(format!("{name}: {worst:.3e} (block {block:.1e})"));
        }
    }
    eprintln!("cell Hessians: largest deviation {overall:.3e}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
