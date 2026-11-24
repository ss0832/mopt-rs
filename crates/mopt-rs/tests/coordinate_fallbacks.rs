//! Coordinate systems that are set up anew during a run, and the automatic choice
//! (`-coord auto`): molecules that have to turn over (the rotation coordinates of TRIC would
//! leave their range), a bent molecule that becomes linear (an angle loses its derivative), and
//! structures for which different coordinates are chosen.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;
use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::coord::system::CoordinateRegistry;
use mopt_rs::domain::{
    positions, Angstrom, AtomId, Element, EvalOutput, EvalRequest, MolecularSystem, Quantity,
    SpinCharge,
};
use mopt_rs::error::MoptResult;
use mopt_rs::opt::state_machine::{
    GeometryOptimizer, IterationReport, OptimizerSettings, Termination,
};
use mopt_rs::pes::native::Gfn0Xtb;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::{run_optimization, Observer};

fn system(rows: &[(&str, [f64; 3])]) -> MolecularSystem {
    MolecularSystem::new(
        rows.iter()
            .map(|r| Element::parse_token(r.0).unwrap())
            .collect(),
        &positions::<Angstrom>(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

/// Rigid molecules in a uniform field: every pair distance inside a molecule is held by a
/// stiff spring, and the first atom of each molecule is pulled along `+z` relative to the
/// centroid of its molecule. The minimum has every molecule pointing its first atom up.
struct Field {
    molecules: Vec<Vec<usize>>,
    lengths: Vec<Vec<f64>>,
    stiffness: f64,
    strength: f64,
}

impl Field {
    fn new(sys: &MolecularSystem, molecules: Vec<Vec<usize>>) -> Self {
        let lengths = molecules
            .iter()
            .map(|m| {
                let mut out = Vec::new();
                for (k, &i) in m.iter().enumerate() {
                    for &j in &m[k + 1..] {
                        out.push(sys.distance(AtomId::new(i), AtomId::new(j)).value());
                    }
                }
                out
            })
            .collect();
        Self {
            molecules,
            lengths,
            stiffness: 0.5,
            strength: 0.02,
        }
    }
}

impl Calculator for Field {
    fn name(&self) -> &str {
        "field"
    }

    fn evaluate(&self, sys: &MolecularSystem, _req: &EvalRequest) -> MoptResult<EvalOutput> {
        let x = sys.coords_flat_bohr();
        let mut g = Col::<f64>::zeros(x.nrows());
        let mut e = 0.0;
        for (m, lengths) in self.molecules.iter().zip(&self.lengths) {
            let mut pair = 0;
            for (k, &i) in m.iter().enumerate() {
                for &j in &m[k + 1..] {
                    let d: [f64; 3] = std::array::from_fn(|a| x[3 * i + a] - x[3 * j + a]);
                    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let dr = r - lengths[pair];
                    e += self.stiffness * dr * dr;
                    for a in 0..3 {
                        let f = 2.0 * self.stiffness * dr * d[a] / r;
                        g[3 * i + a] += f;
                        g[3 * j + a] -= f;
                    }
                    pair += 1;
                }
            }
            // -strength * (z_first - z_centroid).
            let n = m.len() as f64;
            let zc: f64 = m.iter().map(|&i| x[3 * i + 2]).sum::<f64>() / n;
            e -= self.strength * (x[3 * m[0] + 2] - zc);
            g[3 * m[0] + 2] -= self.strength;
            for &i in m {
                g[3 * i + 2] += self.strength / n;
            }
        }
        Ok(EvalOutput::energy_gradient(Quantity::new(e), g))
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }
}

struct Notes(Vec<String>);

impl Observer for Notes {
    fn on_iteration(&mut self, r: &IterationReport) -> MoptResult<()> {
        self.0.extend(r.notes.iter().cloned());
        Ok(())
    }
}

struct Run {
    termination: Option<Termination>,
    iterations: usize,
    energy: f64,
    notes: Vec<String>,
    end: MolecularSystem,
}

fn optimize(
    calc: &dyn Calculator,
    start: &MolecularSystem,
    coordinates: &str,
    method: &str,
) -> Run {
    let settings = OptimizerSettings {
        methods: vec![method.to_string()],
        coordinates: coordinates.to_string(),
        max_iterations: 600,
        rigid_body_invariant: calc.is_rigid_body_invariant(),
        dissociation_distance: None,
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, start.clone()).unwrap();
    let mut notes = Notes(Vec::new());
    let out = run_optimization(&mut opt, calc, &mut [&mut notes]).unwrap();
    let end = opt.current().clone();
    let energy = calc
        .evaluate(&end, &EvalRequest::ENERGY_GRAD)
        .unwrap()
        .energy
        .value();
    Run {
        termination: out.termination,
        iterations: out.iterations,
        energy,
        notes: notes.0,
        end,
    }
}

fn anew(notes: &[String]) -> usize {
    notes
        .iter()
        .filter(|n| n.contains("were set up anew"))
        .count()
}

/// A water and a carbon monoxide that start upside down in the field: both have to turn by
/// about 170 degrees, far beyond the quarter turn at which TRIC takes a new reference, and
/// through the half turn where the rotation coordinates of a fixed reference end.
#[test]
fn molecules_that_turn_over_in_tric() {
    let start = system(&[
        ("O", [0.05, 0.0, -0.40]),
        ("H", [0.0, 0.757, 0.19]),
        ("H", [0.0, -0.757, 0.19]),
        ("C", [4.0, 0.1, -0.60]),
        ("O", [4.0, 0.0, 0.53]),
    ]);
    let field = Field::new(&start, vec![vec![0, 1, 2], vec![3, 4]]);
    let cart = optimize(&field, &start, "cart", "rsirfo_bfgs");
    assert_eq!(cart.termination, Some(Termination::Converged));
    for coordinates in ["tric", "tric-p", "auto"] {
        for method in ["rsirfo_bfgs", "lbfgs", "FIRE"] {
            let r = optimize(&field, &start, coordinates, method);
            assert_eq!(
                r.termination,
                Some(Termination::Converged),
                "{coordinates} {method}"
            );
            assert!(anew(&r.notes) >= 1, "{coordinates} {method}: {:?}", r.notes);
            assert!(
                (r.energy - cart.energy).abs() < 1e-5,
                "{coordinates} {method}: {}",
                r.energy
            );
            // Both molecules point their first atom up.
            let z = |i: usize| r.end.position(AtomId::new(i)).value()[2];
            assert!(
                z(0) > z(1) && z(0) > z(2) && z(3) > z(4),
                "{coordinates} {method}"
            );
            println!("{coordinates} {method}: {} iterations", r.iterations);
        }
    }
}

/// Carbon dioxide from a bent start (140 degrees) with GFN0-xTB: the angle passes 175 degrees
/// on the way to the linear minimum, where it has no derivative; the internal coordinates are
/// generated anew with linear-bend coordinates and the run converges to the Cartesian result.
#[test]
fn a_bent_molecule_that_becomes_linear() {
    let half = 70.0_f64.to_radians();
    let start = system(&[
        ("C", [0.0, 0.0, 0.0]),
        ("O", [1.18 * half.sin(), 1.18 * half.cos(), 0.0]),
        ("O", [-1.18 * half.sin(), 1.18 * half.cos(), 0.0]),
    ]);
    let calc = Gfn0Xtb::default();
    let cart = optimize(&calc, &start, "cart", "rsirfo_bfgs");
    assert_eq!(cart.termination, Some(Termination::Converged));
    for coordinates in ["ric", "dic", "tric", "tric-p", "auto"] {
        let r = optimize(&calc, &start, coordinates, "rsirfo_bfgs");
        assert_eq!(r.termination, Some(Termination::Converged), "{coordinates}");
        assert!(anew(&r.notes) >= 1, "{coordinates}: {:?}", r.notes);
        assert!(
            (r.energy - cart.energy).abs() < 1e-6,
            "{coordinates}: {} vs {}",
            r.energy,
            cart.energy
        );
        println!(
            "{coordinates}: {} iterations (cart {})",
            r.iterations, cart.iterations
        );
    }
}

/// `auto` from the registry and the command line: the choice is written into the log.
#[test]
fn automatic_choice_on_the_command_line() {
    assert_eq!(
        CoordinateRegistry::default().create("auto").unwrap().name(),
        "auto"
    );
    let dir = std::env::temp_dir().join(format!("mopt_auto_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cases = [
        (
            "ar7",
            "7\n\nAr 3.80 0.0 0.0\nAr 1.17 3.61 0.06\nAr -3.07 2.23 0.14\n\
             Ar -3.07 -2.23 0.21\nAr 1.17 -3.61 0.29\nAr 0.0 0.0 4.09\nAr 0.39 0.0 -3.90\n",
            "lj",
            "auto (cart)",
        ),
        (
            "water",
            "3\n\nO 0.0 0.0 0.12\nH 0.0 0.80 -0.47\nH 0.0 -0.70 -0.50\n",
            "gfnff",
            "auto (tric-cart)",
        ),
        (
            "dimer",
            "6\n\nO 0.0 0.0 0.12\nH 0.0 0.76 -0.47\nH 0.0 -0.76 -0.47\n\
             O 3.4 0.5 0.9\nH 3.1 1.3 1.35\nH 4.2 0.3 1.4\n",
            "gfnff",
            "auto (tric-cart)",
        ),
    ];
    for (stem, xyz, calculator, expect) in cases {
        let input = dir.join(format!("{stem}.xyz"));
        std::fs::write(&input, xyz).unwrap();
        let input = input.to_string_lossy();
        let cfg = config_from_cli(&[
            &input,
            "-os",
            calculator,
            "-opt",
            "rsirfo_bfgs",
            "-coord",
            "auto",
            "-ns",
            "500",
        ])
        .unwrap();
        let res = run(&cfg, &dir, None).unwrap();
        assert!(res[0].converged, "{stem}");
        let log = std::fs::read_to_string(res[0].directory.join("mopt.log")).unwrap();
        let head: String = log.lines().take(14).collect::<Vec<_>>().join(
            "
",
        );
        assert!(
            log.contains(&format!("Coordinate system: {expect}")),
            "{stem}: {expect}
{head}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
