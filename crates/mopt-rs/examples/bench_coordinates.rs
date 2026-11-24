//! Benchmark behind the rule of `-coord auto`: the same optimizations in every coordinate
//! system (`cart`, `ric`, `dic`, `tric-p`, `tric`) for single molecules, complexes, clusters of
//! atoms, periodic solids, and transition-state searches, with the built-in calculators.
//!
//! For every case it prints the features the rule can use (atoms, bonded fragments, bonds per
//! atom, periodic network), and for every coordinate system the outcome: iterations, the
//! energy above the lowest one found when it is another minimum (more than 0.1 mHartree
//! higher), and the time per iteration. The table in `docs/source/coordinate_systems.md` is
//! this output.
//!
//! ```text
//! cargo run --release -p mopt-rs --example bench_coordinates            # all cases
//! cargo run --release -p mopt-rs --example bench_coordinates -- water   # cases containing "water"
//! cargo run --release -p mopt-rs --example bench_coordinates -- --method=lbfgs --modelhess=lindh
//! ```
//!
//! One optimization is given up after 500 iterations or 60 seconds (`--max=`, `--seconds=`) and
//! is then shown with `>`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::time::Instant;

use mopt_rs::coord::connectivity::BondConnectivity;
use mopt_rs::coord::system::auto::choose;
use mopt_rs::domain::{
    positions, Angstrom, Element, EvalRequest, MolecularSystem, PeriodicCell, Quantity, SpinCharge,
};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::opt::state_machine::{
    GeometryOptimizer, IterationReport, OptimizerSettings, Termination,
};
use mopt_rs::pes::native::{Emt, Gfn0Xtb, Gfnff, LennardJones, Tersoff};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::{run_optimization, Observer};

type Row = (&'static str, [f64; 3]);

const SYSTEMS: [&str; 6] = ["cart", "ric", "dic", "tric-p", "tric", "tric-cart"];

// --- geometry helpers ---------------------------------------------------------------------------

/// Deterministic displacements of amplitude `amp` (Angstrom).
fn rattle(mut rows: Vec<Row>, amp: f64) -> Vec<Row> {
    for (i, r) in rows.iter_mut().enumerate() {
        for a in 0..3 {
            r.1[a] += amp * ((7 * i + 3 * a) as f64 * 1.37).sin();
        }
    }
    rows
}

fn rotate(p: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let (s, c) = angle.sin_cos();
    let kxp = [
        k[1] * p[2] - k[2] * p[1],
        k[2] * p[0] - k[0] * p[2],
        k[0] * p[1] - k[1] * p[0],
    ];
    let kp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
    std::array::from_fn(|a| p[a] * c + kxp[a] * s + k[a] * kp * (1.0 - c))
}

/// `rows` turned about `axis` through the origin and moved by `shift`.
fn placed(rows: &[Row], axis: [f64; 3], angle: f64, shift: [f64; 3]) -> Vec<Row> {
    rows.iter()
        .map(|(s, p)| {
            let q = rotate(*p, axis, angle);
            (*s, [q[0] + shift[0], q[1] + shift[1], q[2] + shift[2]])
        })
        .collect()
}

fn water() -> Vec<Row> {
    vec![
        ("O", [0.0, 0.0, 0.117]),
        ("H", [0.0, 0.757, -0.469]),
        ("H", [0.0, -0.757, -0.469]),
    ]
}

fn benzene() -> Vec<Row> {
    let mut rows = Vec::new();
    for k in 0..6 {
        let t = std::f64::consts::FRAC_PI_3 * k as f64;
        rows.push(("C", [1.39 * t.cos(), 1.39 * t.sin(), 0.0]));
    }
    for k in 0..6 {
        let t = std::f64::consts::FRAC_PI_3 * k as f64;
        rows.push(("H", [2.48 * t.cos(), 2.48 * t.sin(), 0.0]));
    }
    rows
}

/// All-trans alkane with `n` carbons; the second value lists the atoms of every CHx group.
fn alkane(n: usize) -> (Vec<Row>, Vec<(usize, usize)>) {
    let (dx, dy) = (1.277, 0.861);
    let mut rows = Vec::new();
    let mut groups = Vec::new();
    for k in 0..n {
        let first = rows.len();
        let up = k % 2 == 1;
        let c = [dx * k as f64, if up { dy } else { 0.0 }, 0.0];
        let out = if up { 0.63 } else { -0.63 };
        let end = if up { -0.609 } else { 0.609 };
        rows.push(("C", c));
        rows.push(("H", [c[0], c[1] + out, 0.89]));
        rows.push(("H", [c[0], c[1] + out, -0.89]));
        if k == 0 {
            rows.push(("H", [c[0] - 0.904, c[1] + end, 0.0]));
        }
        if k == n - 1 {
            rows.push(("H", [c[0] + 0.904, c[1] + end, 0.0]));
        }
        groups.push((first, rows.len()));
    }
    (rows, groups)
}

/// The alkane with the part after carbon `k` turned by `angle` about the bond `k - (k+1)`.
fn twisted(mut rows: Vec<Row>, groups: &[(usize, usize)], k: usize, angle: f64) -> Vec<Row> {
    let (a, b) = (rows[groups[k].0].1, rows[groups[k + 1].0].1);
    let axis = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    for g in &groups[k + 1..] {
        for r in &mut rows[g.0..g.1] {
            let rel = [r.1[0] - b[0], r.1[1] - b[1], r.1[2] - b[2]];
            let q = rotate(rel, axis, angle);
            r.1 = [q[0] + b[0], q[1] + b[1], q[2] + b[2]];
        }
    }
    rows
}

/// An alkane with every third C-C bond turned by 50 degrees and all atoms displaced.
fn floppy_alkane(n: usize) -> Vec<Row> {
    let (mut rows, groups) = alkane(n);
    let mut k = 1;
    while k + 1 < n {
        rows = twisted(rows, &groups, k, 50.0_f64.to_radians());
        k += 3;
    }
    rattle(rows, 0.08)
}

fn biphenyl() -> Vec<Row> {
    // Ring A without the hydrogen at angle 0, ring B without the one at 180 degrees, 1.49
    // Angstrom apart and twisted by 70 degrees.
    let mut rows: Vec<Row> = benzene()
        .into_iter()
        .enumerate()
        .filter(|(k, _)| *k != 6)
        .map(|(_, r)| r)
        .collect();
    let ring_b: Vec<Row> = benzene()
        .into_iter()
        .enumerate()
        .filter(|(k, _)| *k != 9)
        .map(|(_, r)| r)
        .collect();
    rows.extend(placed(
        &ring_b,
        [1.0, 0.0, 0.0],
        70.0_f64.to_radians(),
        [4.27, 0.0, 0.0],
    ));
    rattle(rows, 0.05)
}

/// `n` waters on a loose spiral, each turned differently.
fn water_cluster(n: usize, spacing: f64) -> Vec<Row> {
    let mut rows = Vec::new();
    let per_side = (n as f64).cbrt().ceil() as usize;
    for k in 0..n {
        let (i, j, l) = (
            k % per_side,
            (k / per_side) % per_side,
            k / (per_side * per_side),
        );
        let shift = [
            spacing * i as f64 + 0.3 * (k as f64).sin(),
            spacing * j as f64 + 0.3 * (1.7 * k as f64).cos(),
            spacing * l as f64 + 0.3 * (2.3 * k as f64).sin(),
        ];
        let axis = [
            (k as f64).cos(),
            (2.0 * k as f64).sin(),
            0.5 + (3.0 * k as f64).cos().abs(),
        ];
        rows.extend(placed(&water(), axis, 0.9 * k as f64 + 0.4, shift));
    }
    rows
}

fn icosahedron(symbol: &'static str, d: f64) -> Vec<Row> {
    let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
    let s = d / (1.0 + phi * phi).sqrt();
    let mut rows = vec![(symbol, [0.0; 3])];
    for (a, b) in [(s, s * phi), (s, -s * phi), (-s, s * phi), (-s, -s * phi)] {
        rows.extend([
            (symbol, [0.0, a, b]),
            (symbol, [a, b, 0.0]),
            (symbol, [b, 0.0, a]),
        ]);
    }
    rows
}

fn fcc(symbol: &'static str, a: f64, cells: usize) -> Vec<Row> {
    let base = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    lattice(symbol, a, cells, &base)
}

fn diamond(symbol: &'static str, a: f64, cells: usize) -> Vec<Row> {
    let mut base: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let shifted: Vec<[f64; 3]> = base
        .iter()
        .map(|p| [p[0] + 0.25, p[1] + 0.25, p[2] + 0.25])
        .collect();
    base.extend(shifted);
    lattice(symbol, a, cells, &base)
}

fn lattice(symbol: &'static str, a: f64, cells: usize, base: &[[f64; 3]]) -> Vec<Row> {
    let mut rows = Vec::new();
    for i in 0..cells {
        for j in 0..cells {
            for k in 0..cells {
                for b in base {
                    rows.push((
                        symbol,
                        [
                            a * (b[0] + i as f64),
                            a * (b[1] + j as f64),
                            a * (b[2] + k as f64),
                        ],
                    ));
                }
            }
        }
    }
    rows
}

fn diels_alder() -> Vec<Row> {
    vec![
        ("C", [-1.42, 1.18, 0.0]),
        ("C", [-0.705, 0.0, 0.0]),
        ("C", [0.705, 0.0, 0.0]),
        ("C", [1.42, 1.18, 0.0]),
        ("C", [-0.695, 2.08, -1.872]),
        ("C", [0.695, 2.08, -1.872]),
        ("H", [-1.25, -0.944, 0.0]),
        ("H", [1.25, -0.944, 0.0]),
        ("H", [-2.50, 1.16, 0.15]),
        ("H", [-0.933, 2.066, 0.405]),
        ("H", [2.50, 1.16, 0.15]),
        ("H", [0.933, 2.066, 0.405]),
        ("H", [-1.24, 3.00, -2.072]),
        ("H", [-1.24, 1.16, -2.072]),
        ("H", [1.24, 3.00, -2.072]),
        ("H", [1.24, 1.16, -2.072]),
    ]
}

// --- cases --------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Engine {
    Gfnff,
    Gfn0,
    Lj,
    Emt,
    TersoffSi,
}

struct Case {
    name: &'static str,
    rows: Vec<Row>,
    /// Side of a cubic periodic cell (Angstrom).
    cell: Option<f64>,
    engine: Engine,
    method: &'static str,
    saddle_order: usize,
    exact_hessian: Option<usize>,
}

fn minimum(name: &'static str, rows: Vec<Row>, engine: Engine) -> Case {
    Case {
        name,
        rows,
        cell: None,
        engine,
        method: "rsirfo_bfgs",
        saddle_order: 0,
        exact_hessian: None,
    }
}

fn cases() -> Vec<Case> {
    use Engine::{Emt as E, Gfn0, Gfnff as Ff, Lj, TersoffSi as Si};
    let half = 70.0_f64.to_radians();
    let mut list = vec![
        // Single molecules.
        minimum("water", rattle(water(), 0.10), Ff),
        minimum("methane", rattle(alkane(1).0, 0.10), Ff),
        minimum("ethane", rattle(alkane(2).0, 0.10), Ff),
        minimum("butane (twisted)", floppy_alkane(4), Ff),
        minimum("octane (twisted)", floppy_alkane(8), Ff),
        minimum("hexadecane (twisted)", floppy_alkane(16), Ff),
        minimum("C32H66 (twisted)", floppy_alkane(32), Ff),
        minimum("C64H130 (twisted)", floppy_alkane(64), Ff),
        minimum("benzene", rattle(benzene(), 0.10), Ff),
        minimum("biphenyl (twisted)", biphenyl(), Ff),
        minimum(
            "CO2 (bent start)",
            vec![
                ("C", [0.0, 0.0, 0.0]),
                ("O", [1.18 * half.sin(), 1.18 * half.cos(), 0.0]),
                ("O", [-1.18 * half.sin(), 1.18 * half.cos(), 0.0]),
            ],
            Gfn0,
        ),
        minimum(
            "HCN (bent start)",
            vec![
                ("H", [-0.75, 0.75, 0.0]),
                ("C", [0.0, 0.0, 0.0]),
                ("N", [1.16, 0.0, 0.0]),
            ],
            Gfn0,
        ),
        // Complexes.
        minimum(
            "water dimer (apart)",
            {
                let mut r = water();
                r.extend(placed(&water(), [0.2, 1.0, 0.3], 0.8, [3.4, 0.5, 0.9]));
                r
            },
            Ff,
        ),
        minimum("water trimer", water_cluster(3, 3.0), Ff),
        minimum("water hexamer", water_cluster(6, 3.0), Ff),
        minimum("(H2O)20", water_cluster(20, 3.0), Ff),
        minimum(
            "benzene + water",
            {
                let mut r = benzene();
                r.extend(placed(&water(), [1.0, 0.3, 0.0], 2.2, [0.4, 0.3, 3.6]));
                r
            },
            Ff,
        ),
        minimum(
            "benzene dimer",
            {
                let mut r = benzene();
                r.extend(placed(&benzene(), [0.0, 0.0, 1.0], 0.4, [1.2, 0.6, 3.9]));
                r
            },
            Ff,
        ),
        minimum(
            "butane + water",
            {
                let mut r = floppy_alkane(4);
                r.extend(placed(&water(), [0.3, 1.0, 0.2], 1.1, [1.9, 0.4, 3.6]));
                r
            },
            Ff,
        ),
        minimum(
            "methane dimer",
            {
                let mut r = alkane(1).0;
                r.extend(placed(&alkane(1).0, [1.0, 1.0, 0.0], 0.7, [4.4, 0.3, 0.5]));
                r
            },
            Ff,
        ),
        // Clusters of atoms.
        minimum(
            "Ar7 (compressed)",
            vec![
                ("Ar", [1.95, 0.0, 0.0]),
                ("Ar", [0.6026, 1.8546, 0.03]),
                ("Ar", [-1.5776, 1.1462, 0.07]),
                ("Ar", [-1.5776, -1.1462, 0.11]),
                ("Ar", [0.6026, -1.8546, 0.15]),
                ("Ar", [0.0, 0.0, 2.1]),
                ("Ar", [0.2, 0.0, -2.0]),
            ],
            Lj,
        ),
        minimum("Ar13", rattle(icosahedron("Ar", 3.9), 0.25), Lj),
        minimum("Cu13", rattle(icosahedron("Cu", 2.55), 0.15), E),
    ];
    // Periodic solids.
    for (name, rows, side, engine) in [
        ("Cu fcc, 4 atoms", rattle(fcc("Cu", 3.61, 1), 0.12), 3.61, E),
        (
            "Cu fcc, 32 atoms",
            rattle(fcc("Cu", 3.61, 2), 0.12),
            7.22,
            E,
        ),
        (
            "Si diamond, 8 atoms",
            rattle(diamond("Si", 5.43, 1), 0.12),
            5.43,
            Si,
        ),
        (
            "Si diamond, 64 atoms",
            rattle(diamond("Si", 5.43, 2), 0.12),
            10.86,
            Si,
        ),
    ] {
        list.push(Case {
            cell: Some(side),
            ..minimum(name, rows, engine)
        });
    }
    // Transition state.
    list.push(Case {
        method: "rsirfo_bofill",
        saddle_order: 1,
        exact_hessian: Some(20),
        ..minimum("Diels-Alder TS", diels_alder(), Gfn0)
    });
    list
}

// --- running ------------------------------------------------------------------------------------

fn system(case: &Case) -> MoptResult<MolecularSystem> {
    let cell = match case.cell {
        Some(a) => {
            let side = Quantity::<Angstrom>::new(a);
            Some(PeriodicCell::orthorhombic(side, side, side)?)
        }
        None => None,
    };
    let elements = case
        .rows
        .iter()
        .map(|r| Element::parse_token(r.0))
        .collect::<MoptResult<Vec<_>>>()?;
    MolecularSystem::new(
        elements,
        &positions::<Angstrom>(&case.rows.iter().map(|r| r.1).collect::<Vec<_>>()),
        SpinCharge::default(),
        cell,
    )
}

fn calculator(engine: Engine, start: &MolecularSystem) -> MoptResult<Box<dyn Calculator>> {
    Ok(match engine {
        Engine::Gfnff => Box::new(Gfnff::starting_from(start)),
        Engine::Gfn0 => Box::new(Gfn0Xtb::default()),
        Engine::Lj => Box::new(LennardJones),
        Engine::Emt => Box::new(Emt::new()),
        Engine::TersoffSi => Box::new(Tersoff::tersoff_1989(&[Element::parse_token("Si")?])?),
    })
}

/// Stops a run that has used up its time (reported with `>` like one that ran out of
/// iterations).
struct Deadline {
    clock: Instant,
    seconds: f64,
}

impl Observer for Deadline {
    fn on_iteration(&mut self, _report: &IterationReport) -> MoptResult<()> {
        Ok(())
    }

    fn should_stop(&mut self, _current: &MolecularSystem) -> bool {
        self.clock.elapsed().as_secs_f64() > self.seconds
    }
}

/// Options of a benchmark run (`--method=`, `--modelhess=`, `--max=`, `--seconds=`,
/// `--systems=`, `--skip=`; other arguments select the cases whose names contain them).
struct Options {
    /// Method of the minimizations (the transition-state case keeps its own).
    method: Option<String>,
    model_hessian: Option<String>,
    max_iterations: usize,
    /// Time limit of one optimization (seconds).
    seconds: f64,
    /// Coordinate systems to compare (`--systems=cart,tric,tric-cart:0.3`).
    systems: Vec<String>,
    filters: Vec<String>,
    /// Cases left out (`--skip=C64H130,Cu fcc`).
    skip: Vec<String>,
}

impl Options {
    fn from_args() -> MoptResult<Self> {
        let mut o = Self {
            method: None,
            model_hessian: None,
            max_iterations: 500,
            seconds: 60.0,
            systems: SYSTEMS.iter().map(|s| (*s).to_string()).collect(),
            filters: Vec::new(),
            skip: Vec::new(),
        };
        for arg in std::env::args().skip(1) {
            if let Some(v) = arg.strip_prefix("--method=") {
                o.method = Some(v.to_string());
            } else if let Some(v) = arg.strip_prefix("--modelhess=") {
                o.model_hessian = Some(v.to_string());
            } else if let Some(v) = arg.strip_prefix("--max=") {
                o.max_iterations = v
                    .parse()
                    .map_err(|_| MoptError::parse(format!("--max: '{v}' is not a number")))?;
            } else if let Some(v) = arg.strip_prefix("--skip=") {
                o.skip = v.split(',').map(str::to_string).collect();
            } else if let Some(v) = arg.strip_prefix("--systems=") {
                o.systems = v.split(',').map(str::to_string).collect();
            } else if let Some(v) = arg.strip_prefix("--seconds=") {
                o.seconds = v
                    .parse()
                    .map_err(|_| MoptError::parse(format!("--seconds: '{v}' is not a number")))?;
            } else {
                o.filters.push(arg);
            }
        }
        Ok(o)
    }
}

struct Outcome {
    converged: bool,
    iterations: usize,
    energy: f64,
    seconds: f64,
}

fn run(
    case: &Case,
    start: &MolecularSystem,
    coordinates: &str,
    options: &Options,
) -> MoptResult<Outcome> {
    let calc = calculator(case.engine, start)?;
    let method = match (&options.method, case.saddle_order) {
        (Some(m), 0) => m.clone(),
        _ => case.method.to_string(),
    };
    let settings = OptimizerSettings {
        methods: vec![method],
        coordinates: coordinates.to_string(),
        saddle_order: case.saddle_order,
        exact_hessian_interval: case.exact_hessian,
        model_hessian: options
            .model_hessian
            .clone()
            .filter(|_| case.exact_hessian.is_none()),
        max_iterations: options.max_iterations,
        rigid_body_invariant: calc.is_rigid_body_invariant(),
        dissociation_distance: None,
        ..OptimizerSettings::default()
    };
    let clock = Instant::now();
    let mut opt = GeometryOptimizer::new(settings, start.clone())?;
    let mut deadline = Deadline {
        clock,
        seconds: options.seconds,
    };
    let out = run_optimization(&mut opt, calc.as_ref(), &mut [&mut deadline])?;
    let seconds = clock.elapsed().as_secs_f64();
    let energy = calc
        .evaluate(opt.current(), &EvalRequest::ENERGY_GRAD)?
        .energy
        .value();
    Ok(Outcome {
        converged: out.termination == Some(Termination::Converged),
        iterations: out.iterations,
        energy,
        seconds,
    })
}

fn main() -> MoptResult<()> {
    let options = Options::from_args()?;
    println!(
        "| Case | Atoms | Fragments | Bonds per atom | Periodic | {} | Fewest iterations | Rule |",
        options.systems.join(" | ")
    );
    println!(
        "|---|---|---|---|---|{}---|---|",
        "---|".repeat(options.systems.len())
    );
    for case in cases() {
        if !options.filters.is_empty()
            && !options
                .filters
                .iter()
                .any(|f| case.name.contains(f.as_str()))
        {
            continue;
        }
        if options.skip.iter().any(|s| case.name.contains(s.as_str())) {
            continue;
        }
        let start = system(&case)?;
        let conn = BondConnectivity::from_system(&start, 1.3);
        let bonds_per_atom = 2.0 * conn.bonds.len() as f64 / start.n_atoms() as f64;
        let outcomes: Vec<MoptResult<Outcome>> = options
            .systems
            .iter()
            .map(|c| run(&case, &start, c, &options))
            .collect();
        let lowest = outcomes
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .filter(|o| o.converged)
            .map(|o| o.energy)
            .fold(f64::INFINITY, f64::min);
        let mut cells = Vec::new();
        let mut best: Option<(usize, &str)> = None;
        for (name, outcome) in options.systems.iter().zip(&outcomes) {
            let o = match outcome {
                Ok(o) => o,
                Err(e) => {
                    let short: String = e.to_string().chars().take(28).collect();
                    cells.push(format!("error ({short})"));
                    continue;
                }
            };
            let per_step = 1e3 * o.seconds / o.iterations.max(1) as f64;
            // Another minimum: more than 0.1 mHartree above the lowest one found.
            let above = 1e3 * (o.energy - lowest);
            let other = o.converged && above > 0.1;
            cells.push(format!(
                "{}{}{} ({:.1} ms)",
                if o.converged { "" } else { ">" },
                o.iterations,
                if other {
                    format!(" [+{above:.2} mEh]")
                } else {
                    String::new()
                },
                per_step
            ));
            if o.converged && !other && best.is_none_or(|(n, _)| o.iterations < n) {
                best = Some((o.iterations, name.as_str()));
            }
        }
        println!(
            "| {} | {} | {} | {:.2} | {} | {} | {} | {} |",
            case.name,
            start.n_atoms(),
            conn.fragments.len(),
            bonds_per_atom,
            if start.has_pbc() { "yes" } else { "no" },
            cells.join(" | "),
            best.map_or("-".to_string(), |(n, s)| format!("{s} ({n})")),
            choose(&start, 1.3).name(),
        );
    }
    Ok(())
}
