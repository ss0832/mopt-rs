//! Benchmark of Hessian updates in transition-state searches: the same searches with the
//! single-pair updates (Bofill, MSP), the block (multi-secant) Bofill forms, and GPMin, from
//! several guesses and with exact Hessians at several intervals, with the built-in GFN0-xTB.
//!
//! Cases:
//!
//! - `epoxidation`: ethylene and peroxyacetic acid, the spiro transition state in which the
//!   terminal oxygen of the peracid is handed to the double bond while the O-O bond breaks.
//!   Guesses differ in the O-O distance, the O-C distances, and a shift of the ethylene along
//!   its bond.
//! - `diels-alder`: butadiene and ethylene, guesses with the forming bonds between 2.0 and 2.4
//!   Angstrom.
//!
//! A run counts as having found the transition state when it converges to a point with one
//! negative curvature whose breaking and forming bonds are within the ranges of the case. The
//! table shows the iterations, `n` for a run that did not converge within the limit, and `x`
//! for one that converged elsewhere.
//!
//! ```text
//! cargo run --release -p mopt-rs --example bench_ts_updates                 # all cases
//! cargo run --release -p mopt-rs --example bench_ts_updates -- epoxidation  # one case
//! cargo run --release -p mopt-rs --example bench_ts_updates -- --probe      # where an exact-Hessian search ends
//! ```
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use mopt_rs::domain::{
    positions, Angstrom, AtomId, Element, EvalRequest, MolecularSystem, SpinCharge, Units,
};
use mopt_rs::error::MoptResult;
use mopt_rs::math::linalg::symmetric_eigen;
use mopt_rs::math::projection::project_out_tr_rot_hessian;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::pes::native::Gfn0Xtb;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::run_optimization;

type Row = (&'static str, [f64; 3]);

const METHODS: [&str; 6] = [
    "rsirfo_bofill",
    "rsirfo_msp",
    "rsirfo_block_bofill",
    "rsirfo_block_bofill_weighted",
    "rsirfo_block_cfd_bofill",
    "gpmin",
];
/// Iterations between exact Hessians (`1000`: at the start only).
const INTERVALS: [usize; 3] = [1000, 20, 10];
const MAX_ITERATIONS: usize = 200;

/// A bond (0-based atoms) and the range of its length (Angstrom) at the transition state.
type Bond = (usize, usize, f64, f64);

struct Case {
    name: &'static str,
    guesses: Vec<(String, Vec<Row>)>,
    bonds: Vec<Bond>,
}

/// Deterministic displacements of amplitude `amp` (Angstrom).
fn rattle(mut rows: Vec<Row>, amp: f64) -> Vec<Row> {
    for (i, r) in rows.iter_mut().enumerate() {
        for a in 0..3 {
            r.1[a] += amp * ((7 * i + 3 * a) as f64 * 1.37).sin();
        }
    }
    rows
}

/// Peroxyacetic acid (planar, the hydroxyl hydrogen towards the carbonyl oxygen) handing its
/// terminal oxygen to ethylene: O-O distance `oo`, distance `oc` from that oxygen to both
/// carbons. The ethylene bond is perpendicular to the plane of the acid (the spiro
/// arrangement) turned by `twist` degrees about the line of approach, and the ethylene is
/// moved by `shift` along its own bond.
fn epoxidation(oo: f64, oc: f64, shift: f64, twist: f64) -> Vec<Row> {
    // Direction from the inner to the terminal peroxide oxygen, and the one across it.
    let along = [0.985, 0.174, 0.0];
    let across = [-0.174, 0.985, 0.0];
    let (sin, cos) = twist.to_radians().sin_cos();
    // The ethylene bond and the direction of its hydrogens.
    let bond = [sin * across[0], sin * across[1], cos];
    let wide = [cos * across[0], cos * across[1], -sin];
    let inner = [0.675, -1.169, 0.0];
    let terminal = [inner[0] + oo * along[0], inner[1] + oo * along[1], 0.0];
    let half = 0.68;
    let reach = (oc * oc - half * half).max(0.0).sqrt();
    let middle: [f64; 3] =
        std::array::from_fn(|a| terminal[a] + reach * along[a] + shift * bond[a]);
    let mut rows = vec![
        ("C", [-1.50, 0.0, 0.0]),
        ("H", [-1.87, 1.03, 0.0]),
        ("H", [-1.87, -0.51, 0.89]),
        ("H", [-1.87, -0.51, -0.89]),
        ("C", [0.0, 0.0, 0.0]),
        ("O", [0.605, 1.048, 0.0]),
        ("O", inner),
        ("O", terminal),
        ("H", [terminal[0], terminal[1] + 0.98, 0.0]),
    ];
    for side in [1.0, -1.0] {
        let c: [f64; 3] = std::array::from_fn(|a| middle[a] + side * half * bond[a]);
        rows.push(("C", c));
        for hand in [1.0, -1.0] {
            rows.push((
                "H",
                std::array::from_fn(|a| {
                    c[a] + hand * 0.925 * wide[a] + side * 0.567 * bond[a] + 0.10 * along[a]
                }),
            ));
        }
    }
    rows
}

/// Butadiene and ethylene with the two forming bonds stretched from the guess of 2.20 Angstrom
/// by moving the ethylene by `away`.
fn diels_alder(away: f64) -> Vec<Row> {
    let mut rows: Vec<Row> = vec![
        ("C", [-1.42, 1.18, 0.0]),
        ("C", [-0.705, 0.0, 0.0]),
        ("C", [0.705, 0.0, 0.0]),
        ("C", [1.42, 1.18, 0.0]),
        ("C", [-0.695, 2.08, -1.8727]),
        ("C", [0.695, 2.08, -1.8727]),
        ("H", [-1.25, -0.944, 0.0]),
        ("H", [1.25, -0.944, 0.0]),
        ("H", [-2.50, 1.16, 0.15]),
        ("H", [-0.933, 2.066, 0.405]),
        ("H", [2.50, 1.16, 0.15]),
        ("H", [0.933, 2.066, 0.405]),
        ("H", [-1.24, 3.00, -2.0727]),
        ("H", [-1.24, 1.16, -2.0727]),
        ("H", [1.24, 3.00, -2.0727]),
        ("H", [1.24, 1.16, -2.0727]),
    ];
    // The ethylene (carbons 4 and 5 with their hydrogens) along the forming bonds.
    let direction = [0.0, 0.433, -0.901];
    for k in [4, 5, 12, 13, 14, 15] {
        for (value, d) in rows[k].1.iter_mut().zip(direction) {
            *value += away * d;
        }
    }
    rows
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "epoxidation",
            guesses: vec![
                (
                    "O-O 1.98, O-C 1.86, shifted 0.2".to_string(),
                    epoxidation(1.98, 1.86, 0.2, 0.0),
                ),
                (
                    "O-O 1.98, O-C 1.86, turned 30".to_string(),
                    epoxidation(1.98, 1.86, 0.0, 30.0),
                ),
                (
                    "O-O 1.98, O-C 1.86, turned 90".to_string(),
                    epoxidation(1.98, 1.86, 0.0, 90.0),
                ),
                (
                    "O-O 1.98, O-C 1.86, displaced".to_string(),
                    rattle(epoxidation(1.98, 1.86, 0.0, 0.0), 0.05),
                ),
                (
                    "O-O 1.90, O-C 2.00, shifted 0.2, turned 30".to_string(),
                    epoxidation(1.90, 2.00, 0.2, 30.0),
                ),
            ],
            // The breaking O-O bond and the two forming O-C bonds.
            bonds: vec![(6, 7, 1.55, 2.40), (7, 9, 1.70, 2.60), (7, 12, 1.70, 2.60)],
        },
        Case {
            name: "diels-alder",
            guesses: vec![
                ("forming bonds 2.20".to_string(), diels_alder(0.0)),
                ("forming bonds 2.00".to_string(), diels_alder(-0.20)),
                ("forming bonds 2.40".to_string(), diels_alder(0.20)),
                (
                    "2.20, displaced".to_string(),
                    rattle(diels_alder(0.0), 0.05),
                ),
            ],
            bonds: vec![(0, 4, 1.90, 2.50), (3, 5, 1.90, 2.50)],
        },
    ]
}

fn system(rows: &[Row]) -> MoptResult<MolecularSystem> {
    let elements = rows
        .iter()
        .map(|r| Element::parse_token(r.0))
        .collect::<MoptResult<Vec<_>>>()?;
    MolecularSystem::new(
        elements,
        &positions::<Angstrom>(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
        SpinCharge::default(),
        None,
    )
}

fn lengths(sys: &MolecularSystem, bonds: &[Bond]) -> Vec<f64> {
    bonds
        .iter()
        .map(|&(a, b, _, _)| {
            sys.distance(AtomId::new(a), AtomId::new(b)).value() * Units::BOHR_TO_ANGSTROM
        })
        .collect()
}

/// Negative curvatures of the surface at `sys` (rigid-body motions removed).
fn negative_curvatures(calc: &dyn Calculator, sys: &MolecularSystem) -> MoptResult<Vec<f64>> {
    let out = calc.evaluate(sys, &EvalRequest::ENERGY_GRAD_HESS)?;
    let Some(hessian) = out.hessian_bohr else {
        return Ok(Vec::new());
    };
    let (values, _) = symmetric_eigen(&project_out_tr_rot_hessian(sys, &hessian, false))?;
    Ok(values.iter().copied().filter(|v| *v < -1e-4).collect())
}

struct Outcome {
    converged: bool,
    iterations: usize,
    lengths: Vec<f64>,
    negative: Vec<f64>,
}

impl Outcome {
    fn found(&self, bonds: &[Bond]) -> bool {
        self.converged
            && self.negative.len() == 1
            && self
                .lengths
                .iter()
                .zip(bonds)
                .all(|(d, b)| *d >= b.2 && *d <= b.3)
    }

    fn cell(&self, bonds: &[Bond]) -> String {
        if self.found(bonds) {
            format!("{}", self.iterations)
        } else if self.converged {
            "x".to_string()
        } else {
            "n".to_string()
        }
    }
}

fn search(case: &Case, rows: &[Row], method: &str, interval: usize) -> MoptResult<Outcome> {
    let calc = Gfn0Xtb::default();
    let settings = OptimizerSettings {
        methods: vec![method.to_string()],
        saddle_order: 1,
        max_iterations: MAX_ITERATIONS,
        exact_hessian_interval: Some(interval),
        dissociation_distance: None,
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, system(rows)?)?;
    let out = run_optimization(&mut opt, &calc, &mut [])?;
    let end = opt.current();
    let converged = out.termination == Some(Termination::Converged);
    Ok(Outcome {
        converged,
        iterations: out.iterations,
        lengths: lengths(end, &case.bonds),
        negative: if converged {
            negative_curvatures(&calc, end)?
        } else {
            Vec::new()
        },
    })
}

fn main() -> MoptResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let probe = args.iter().any(|a| a == "--probe");
    let filters: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    for case in cases() {
        if !filters.is_empty() && !filters.iter().any(|f| case.name.contains(f.as_str())) {
            continue;
        }
        if probe {
            // Where a search with an exact Hessian at every iteration ends, per guess.
            println!("## {}: exact Hessian at every iteration", case.name);
            let calc = Gfn0Xtb::default();
            for (label, rows) in &case.guesses {
                let start = system(rows)?;
                let at_start = negative_curvatures(&calc, &start)?;
                let o = search(&case, rows, "rsirfo_bofill", 1)?;
                println!(
                    "- {label}: start {:.2?} A, negative curvatures {:.4?}; end after {} \
                     iterations (converged: {}) {:.3?} A, negative curvatures {:.4?}",
                    lengths(&start, &case.bonds),
                    at_start,
                    o.iterations,
                    o.converged,
                    o.lengths,
                    o.negative
                );
            }
            continue;
        }
        println!("## {}", case.name);
        println!();
        println!("| Guess | Hessian every | {} |", METHODS.join(" | "));
        println!("|---|---|{}", "---|".repeat(METHODS.len()));
        let mut totals = vec![(0usize, 0usize); METHODS.len()];
        for (label, rows) in &case.guesses {
            for interval in INTERVALS {
                let mut cells = Vec::new();
                for (k, method) in METHODS.iter().enumerate() {
                    let cell = match search(&case, rows, method, interval) {
                        Ok(o) => {
                            if o.found(&case.bonds) {
                                totals[k].0 += o.iterations;
                                totals[k].1 += 1;
                            }
                            o.cell(&case.bonds)
                        }
                        Err(e) => {
                            let short: String = e.to_string().chars().take(24).collect();
                            format!("error ({short})")
                        }
                    };
                    cells.push(cell);
                }
                let every = if interval >= MAX_ITERATIONS {
                    "start only".to_string()
                } else {
                    format!("{interval}")
                };
                println!("| {label} | {every} | {} |", cells.join(" | "));
            }
        }
        let runs = case.guesses.len() * INTERVALS.len();
        let found: Vec<String> = totals
            .iter()
            .map(|t| format!("{} of {runs}", t.1))
            .collect();
        let sums: Vec<String> = totals.iter().map(|t| format!("{}", t.0)).collect();
        println!("| **found** | | {} |", found.join(" | "));
        println!(
            "| **iterations of the runs that found it** | | {} |",
            sums.join(" | ")
        );
        println!();
    }
    Ok(())
}
