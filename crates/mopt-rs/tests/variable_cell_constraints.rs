//! Constraints on the cell itself in variable-cell optimizations (`optmain -vc` with `-vc_fix`
//! or `-uc` of the functions of the cell): every kind (a length, an angle, the volume, the
//! shape, a lattice vector, one component) held to 1e-10 at every iteration in memory, the
//! stresses conjugate to the free degrees of freedom of the cell vanishing at convergence and
//! those conjugate to the held ones not, the constrained optimum against a stiff penalty, the
//! dedicated option against the expressions, the combinations (-fix, -pc, -vc_pressure,
//! -vc_afir, -vc_weight, -es, -mf, -order 1), and the refusals by name.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::{Col, Mat};
use mopt_rs::cli::constraints::{cell_fix_constraints, resolve_for_run, user_constraint};
use mopt_rs::cli::optmain::{self, config_from_cli, run_with};
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity, SpinCharge, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::opt::cell::{lattice_gradient, VariableCell};
use mopt_rs::opt::cell_constraints::cell_constraints;
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::Termination;
use mopt_rs::opt::vector::{VectorAction, VectorOptimizer, VectorSettings};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind};
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::pes::stress::{numerical_stress, Stress};
use mopt_rs::pes::{Calculator, PesTerm, TermContext};

/// A 2 x 2 x 2 fcc argon crystal (32 atoms) stretched to 5.40 Angstrom, every atom shifted.
fn fcc() -> String {
    let a = 5.40;
    let mut rows = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    let n = rows.len() as f64;
                    let shift = [
                        0.07 * (1.3 * n).sin(),
                        0.06 * (0.7 * n).cos(),
                        0.05 * (2.1 * n).sin(),
                    ];
                    let cell = [f64::from(i), f64::from(j), f64::from(k)];
                    rows.push(std::array::from_fn::<f64, 3, _>(|c| {
                        a * (cell[c] + f[c]) + shift[c]
                    }));
                }
            }
        }
    }
    // A little skewed, so that the angles have something to relax.
    let l = 2.0 * a;
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.15 {l} 0.0 -0.1 0.12 {l}\" pbc=\"T T T\"\n",
        rows.len()
    );
    for r in &rows {
        text.push_str(&format!("Ar {:.10} {:.10} {:.10}\n", r[0], r[1], r[2]));
    }
    text
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_vcfix_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn crystal(dir: &Path) -> (PathBuf, MolecularSystem) {
    let file = dir.join("fcc.xyz");
    std::fs::write(&file, fcc()).unwrap();
    let sys = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    (file, sys)
}

fn lj(sys: &MolecularSystem) -> Box<dyn Calculator> {
    CalculatorRegistry::native()
        .create("lj", &CalculatorOptions::default(), sys)
        .unwrap()
}

/// Energy, Cartesian gradient, and stress of `calc` at `sys`.
fn evaluate(calc: &dyn Calculator, sys: &MolecularSystem) -> (f64, Col<f64>, Stress) {
    let out = calc
        .evaluate(sys, &EvalRequest::ENERGY_GRAD_STRESS)
        .unwrap();
    let stress = match out.stress {
        Some(s) => s,
        None => numerical_stress(calc, sys, 1e-5).unwrap(),
    };
    (out.energy.value(), out.gradient_bohr, stress)
}

/// What an optimization through the API ended with.
struct Optimum {
    system: MolecularSystem,
    energy: f64,
    /// Largest residual of any constraint at any point the optimizer asked for.
    largest: f64,
    /// The gradient of the function minimized at the end, and its part across the
    /// constraints.
    gradient: Col<f64>,
    free: Col<f64>,
    stress: Stress,
    iterations: usize,
}

/// The optimization of `sys` with LJ in the variables of the cell, holding `constraints` (as
/// `optmain -vc` does), with a stiff penalty `k (f - target)^2` on a function `f` instead
/// when given.
fn optimize(
    sys: &MolecularSystem,
    constraints: &[Constraint],
    penalty: Option<(Arc<dyn PesTerm>, f64, f64)>,
) -> Optimum {
    let calc = lj(sys);
    let map = VariableCell::new(sys, 2.0, 0.0).unwrap();
    let resolved = resolve_for_run(constraints, sys).unwrap();
    let set = cell_constraints(&map, &resolved, sys).unwrap();
    let settings = VectorSettings {
        methods: vec!["rsirfo_bfgs".into()],
        max_iterations: 800,
        criteria: ConvergenceCriteria::TIGHT,
        ..VectorSettings::default()
    };
    let mut opt = VectorOptimizer::new(settings, map.variables(sys).unwrap()).unwrap();
    if !set.is_empty() {
        opt.set_constraints(set.clone()).unwrap();
    }
    let mut largest = 0.0_f64;
    let mut last = None;
    while let VectorAction::Evaluate { x, .. } = opt.ask() {
        if !set.is_empty() {
            let r = set.residuals(&x).unwrap();
            largest = r.iter().fold(largest, |m, v| m.max(v.abs()));
            assert!(
                r.iter().all(|v| v.abs() <= 1e-10),
                "iteration {}: {r:?}",
                opt.iterations()
            );
        }
        let s = map.system(&x).unwrap();
        let (e, g, stress) = evaluate(calc.as_ref(), &s);
        let (mut h, mut slope) = map.objective(&s, e, &g, &stress).unwrap();
        if let Some((term, target, k)) = &penalty {
            let out = term
                .evaluate(&s, &TermContext::at_iteration(0, &s), false)
                .unwrap();
            let d = out.energy.value() - target;
            let of = map
                .gradient_in_variables(&s, &out.gradient_bohr, &out.stress.unwrap())
                .unwrap();
            h += k * d * d;
            slope = &slope + &(of * faer::Scale(2.0 * k * d));
        }
        last = Some((s, e, slope.clone(), stress));
        opt.tell(h, slope, None).unwrap();
    }
    assert_eq!(opt.termination(), Some(Termination::Converged));
    let (system, energy, gradient, stress) = last.unwrap();
    let x = map.variables(&system).unwrap();
    let free = if set.is_empty() {
        gradient.clone()
    } else {
        set.frame(&x).unwrap().project(&gradient)
    };
    Optimum {
        system,
        energy,
        largest,
        gradient,
        free,
        stress,
        iterations: opt.iterations(),
    }
}

fn fixed(items: &[&str]) -> Vec<Constraint> {
    cell_fix_constraints(&items.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
}

fn lattice(sys: &MolecularSystem) -> [[f64; 3]; 3] {
    let cell = sys.active_cell().unwrap();
    std::array::from_fn(|k| cell.lattice_vector(k).value())
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[test]
fn every_kind_holds_at_every_iteration_and_the_free_stresses_vanish() {
    let dir = scratch("kinds");
    let (_, start) = crystal(&dir);
    let free = optimize(&start, &[], None);
    let v0 = start.active_cell().unwrap().volume_bohr3();
    for (kind, items) in [
        ("length", vec!["a"]),
        ("length with a target", vec!["a=10.6"]),
        ("angle", vec!["gamma"]),
        ("volume", vec!["volume"]),
        ("shape", vec!["shape"]),
        ("lattice vector", vec!["c_vector"]),
        ("component", vec!["cz"]),
        ("lengths and an angle", vec!["a,b,gamma"]),
    ] {
        let out = optimize(&start, &fixed(&items), None);
        eprintln!(
            "RESIDUAL cell {kind}: largest {:.2e} over {} iterations",
            out.largest, out.iterations
        );
        // The free part of the gradient vanishes; the cell has relaxed away from the start,
        // and does not reach the unconstrained optimum (the held part does not vanish).
        assert!(out.free.norm_l2() < 1e-4, "{kind}: {}", out.free.norm_l2());
        assert!(out.energy > free.energy - 1e-9, "{kind}");
        assert!(
            out.gradient.norm_l2() > 10.0 * out.free.norm_l2(),
            "{kind}: the held directions carry no force ({})",
            out.gradient.norm_l2()
        );
        let l = lattice(&out.system);
        let cell = out.system.active_cell().unwrap();
        let g = lattice_gradient(&out.stress, cell);
        let sigma = out.stress;
        let mean = (sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0;
        match kind {
            "volume" => {
                // Only the shape relaxed: no deviatoric stress, and the pressure is minus the
                // multiplier of the volume (the gradient of the enthalpy along that of V).
                assert!((cell.volume_bohr3() - v0).abs() < 1e-9 * v0);
                for a in 0..3 {
                    for b in 0..3 {
                        let deviatoric = sigma[a][b] - if a == b { mean } else { 0.0 };
                        assert!(deviatoric.abs() < 1e-7, "{a}{b}: {sigma:?}");
                    }
                }
                let map = VariableCell::new(&out.system, 2.0, 0.0).unwrap();
                let set = cell_constraints(
                    &map,
                    &resolve_for_run(&fixed(&["volume"]), &start).unwrap(),
                    &start,
                )
                .unwrap();
                let x = map.variables(&out.system).unwrap();
                let n = set.frame(&x).unwrap().jacobian;
                let dv = Col::from_fn(x.nrows(), |i| n[(0, i)]);
                // The gradient of the variables of the final cell is that of map.
                let (_, slope) = {
                    let calc = lj(&out.system);
                    let (e, gr, st) = evaluate(calc.as_ref(), &out.system);
                    map.objective(&out.system, e, &gr, &st).unwrap()
                };
                let lambda = (0..x.nrows()).map(|i| slope[i] * dv[i]).sum::<f64>()
                    / (0..x.nrows()).map(|i| dv[i] * dv[i]).sum::<f64>();
                let pressure = -mean;
                assert!(pressure.abs() > 1e-6, "no pressure: {sigma:?}");
                assert!(
                    (lambda + pressure).abs() < 1e-3 * pressure.abs(),
                    "multiplier {lambda} against the pressure {pressure}"
                );
            }
            "shape" => {
                // Only an isotropic scaling relaxed: no pressure.
                assert!(mean.abs() < 1e-7, "{sigma:?}");
            }
            "length" | "length with a target" => {
                // Only the force along a (on the length of a) is left.
                let u = l[0].map(|c| c / norm(l[0]));
                let along: f64 = (0..3).map(|c| g[0][c] * u[c]).sum();
                for k in 1..3 {
                    assert!(norm(g[k]) < 5e-5, "{kind} {k}: {g:?}");
                }
                assert!(norm(std::array::from_fn(|c| g[0][c] - along * u[c])) < 5e-5);
                assert!(along.abs() > 1e-4, "{kind}: {g:?}");
                if kind == "length with a target" {
                    let target = 10.6 * Units::ANGSTROM_TO_BOHR;
                    assert!((norm(l[0]) - target).abs() < 1e-10);
                }
            }
            "lattice vector" => {
                // The vector stays where it is in space; the others are free.
                let l0 = lattice(&start);
                for c in 0..3 {
                    assert!((l[2][c] - l0[2][c]).abs() < 1e-10);
                }
                assert!(norm(g[0]) < 5e-5 && norm(g[1]) < 5e-5, "{g:?}");
            }
            _ => {}
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_constrained_optimum_is_that_of_a_stiff_penalty() {
    let dir = scratch("penalty");
    let (_, start) = crystal(&dir);
    for (item, expr, target) in [
        ("a=10.6", "cell_a()", 10.6 * Units::ANGSTROM_TO_BOHR),
        (
            "volume=1150",
            "volume()",
            1150.0 * Units::ANGSTROM_TO_BOHR.powi(3),
        ),
    ] {
        let held = optimize(&start, &fixed(&[item]), None);
        let term = mopt_rs::pes::bias::ExpressionBias::parse(expr)
            .unwrap()
            .into_term();
        // Stiff enough that the force of the crystal moves the function by less than 1e-4
        // of its value.
        let k = if expr == "volume()" { 1e-5 } else { 1.0 };
        let spring = optimize(&start, &[], Some((Arc::clone(&term), target, k)));
        let value = |s: &MolecularSystem| {
            term.evaluate(s, &TermContext::at_iteration(0, s), false)
                .unwrap()
                .energy
                .value()
        };
        assert!((value(&held.system) - target).abs() < 1e-10 * target.max(1.0));
        assert!(
            (value(&spring.system) - target).abs() < 1e-4 * target,
            "{item}: {} vs {target}",
            value(&spring.system)
        );
        assert!(
            (held.energy - spring.energy).abs() < 2e-5,
            "{item}: {} vs {}",
            held.energy,
            spring.energy
        );
        let (lh, ls) = (lattice(&held.system), lattice(&spring.system));
        if expr == "volume()" {
            // At a constant volume the shape is soft: the volumes agree.
            let (vh, vs) = (
                held.system.active_cell().unwrap().volume_bohr3(),
                spring.system.active_cell().unwrap().volume_bohr3(),
            );
            assert!((vh - vs).abs() < 1e-4 * vh, "{item}: {vh} vs {vs}");
        } else {
            for k in 0..3 {
                assert!((norm(lh[k]) - norm(ls[k])).abs() < 2e-3, "{item} {k}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn run(dir: &Path, args: &[&str]) -> optmain::OptmainResult {
    let config = config_from_cli(args).unwrap();
    run_with(&config, dir, None, &CalculatorRegistry::native())
        .unwrap()
        .remove(0)
}

fn refusal(dir: &Path, args: &[&str]) -> String {
    match config_from_cli(args) {
        Err(e) => e.to_string(),
        Ok(config) => run_with(&config, dir, None, &CalculatorRegistry::native())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default(),
    }
}

#[test]
fn the_dedicated_option_and_the_expressions_are_the_same_constraints() {
    let dir = scratch("spellings");
    let (file, _) = crystal(&dir);
    let file = file.to_string_lossy().into_owned();
    let base = [
        file.as_str(),
        "-os",
        "lj",
        "-vc",
        "-opt",
        "rsirfo_bfgs",
        "-ns",
        "400",
    ];
    for (option, expressions) in [
        (vec!["-vc_fix", "volume"], vec!["volume()"]),
        (
            vec!["-vc_fix", "a=10.6,gamma"],
            vec!["cell_a() = 10.6*ang", "cell_gamma()"],
        ),
        (
            vec!["-vc_fix", "shape"],
            vec![
                "cell_b()/cell_a()",
                "cell_c()/cell_a()",
                "cell_alpha()",
                "cell_beta()",
                "cell_gamma()",
            ],
        ),
    ] {
        let mut a = base.to_vec();
        a.extend(option.iter().copied());
        let mut b = base.to_vec();
        b.push("-uc");
        b.extend(expressions.iter().copied());
        let (x, y) = (run(&dir, &a), run(&dir, &b));
        assert!(x.converged && y.converged, "{option:?}");
        let (x, y) = (x.summary.unwrap(), y.summary.unwrap());
        assert_eq!(x.iterations, y.iterations, "{option:?}");
        assert_eq!(
            x.energy.value().to_bits(),
            y.energy.value().to_bits(),
            "{option:?}"
        );
        assert_eq!(
            x.system.coords_flat_bohr(),
            y.system.coords_flat_bohr(),
            "{option:?}"
        );
        assert_eq!(lattice(&x.system), lattice(&y.system), "{option:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_of_the_cell_with_the_other_options() {
    let dir = scratch("combinations");
    let (file, start) = crystal(&dir);
    let file = file.to_string_lossy().into_owned();
    let v0 = start.active_cell().unwrap().volume_bohr3();
    let l0 = lattice(&start);
    let base = [
        file.as_str(),
        "-os",
        "lj",
        "-vc",
        "-opt",
        "rsirfo_bfgs",
        "-ns",
        "600",
    ];
    let with = |extra: &[&str]| {
        let mut args = base.to_vec();
        args.extend_from_slice(extra);
        let out = run(&dir, &args);
        assert!(out.converged, "{extra:?}");
        out.summary.unwrap().system
    };
    // Frozen atoms (fractional coordinates) and the volume.
    let s = with(&["-fix", "1,5", "-vc_fix", "volume"]);
    assert!((s.active_cell().unwrap().volume_bohr3() - v0).abs() < 1e-9 * v0);
    // A distance and a length.
    let s = with(&["-pc", "bond", "1,2", "-vc_fix", "a"]);
    assert!((norm(lattice(&s)[0]) - norm(l0[0])).abs() < 1e-10);
    let d = |s: &MolecularSystem| {
        s.distance(
            mopt_rs::domain::AtomId::new(0),
            mopt_rs::domain::AtomId::new(1),
        )
        .value()
    };
    assert!((d(&s) - d(&start)).abs() < 1e-10);
    // A pressure with the shape held: the enthalpy is minimized on the constraint surface,
    // where only the isotropic scaling is free, so the pressure of the crystal is the external
    // one.
    let s = with(&["-vc_pressure", "0.2", "-vc_fix", "shape"]);
    let (_, _, sigma) = evaluate(lj(&s).as_ref(), &s);
    let pressure = -(sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0;
    let gpa = pressure * mopt_rs::opt::cell::gigapascal_per_atomic_unit();
    assert!((gpa - 0.2).abs() < 2e-3, "{gpa} GPa");
    // An artificial force on a held length is taken up by the constraint (its multiplier):
    // the length stays, and the optimum is that of the constraint alone.
    let pushed = with(&["-vc_afir", "50", "a", "-vc_fix", "a"]);
    let alone = with(&["-vc_fix", "a"]);
    assert!((norm(lattice(&pushed)[0]) - norm(l0[0])).abs() < 1e-10);
    for k in 0..3 {
        assert!(
            (norm(lattice(&pushed)[k]) - norm(lattice(&alone)[k])).abs() < 1e-3,
            "{k}"
        );
    }
    // Another weight of the lattice variables.
    let s = with(&["-vc_weight", "1.0", "-vc_fix", "gamma"]);
    let angle = |l: [[f64; 3]; 3]| {
        let dot: f64 = (0..3).map(|c| l[0][c] * l[1][c]).sum();
        (dot / (norm(l[0]) * norm(l[1]))).acos()
    };
    assert!((angle(lattice(&s)) - angle(l0)).abs() < 1e-10);
    // Exact Hessians every iteration (the curvature of the constraints added).
    let s = with(&["-fc", "1", "-vc_fix", "volume"]);
    assert!((s.active_cell().unwrap().volume_bohr3() - v0).abs() < 1e-9 * v0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_cannot_be_held_is_refused_by_name() {
    let dir = scratch("refused");
    let (file, _) = crystal(&dir);
    let file = file.to_string_lossy().into_owned();
    let f = file.as_str();
    for (args, expected) in [
        (vec![f, "-os", "lj", "-vc_fix", "volume"], "needs -vc"),
        (vec![f, "-os", "lj", "-uc", "volume()"], "variable-cell"),
        (
            vec![f, "-os", "lj", "-vc", "-vc_fix", "a,b,c,alpha,beta,gamma"],
            "drop -vc",
        ),
        (
            vec![
                f,
                "-os",
                "lj",
                "-vc",
                "-vc_fix",
                "volume",
                "-uc",
                "volume() = 1e4",
            ],
            "contradictory",
        ),
        (
            vec![f, "-os", "lj", "-vc", "-vc_fix", "shape,b"],
            "fix it completely",
        ),
        (
            vec![f, "-os", "lj", "-vc", "-vc_fix", "shape=2"],
            "no value",
        ),
        (
            vec![
                f, "-os", "lj", "-vc", "-vc_fix", "a", "-irc", "0.1", "10", "lqa",
            ],
            "-irc after a constrained -vc",
        ),
    ] {
        let err = refusal(&dir, &args);
        assert!(err.contains(expected), "{args:?}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- Two states, an excited state, and a saddle point with the cell held ----

const K: f64 = 0.01;
const A: f64 = 0.05;
const KB: f64 = 0.02;
const L: [f64; 2] = [5.0, 6.0];
const R: [f64; 2] = [2.0, 2.5];
const C: [f64; 2] = [0.0, -0.002];

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Two atoms in a cell with two states (as `variable_cell_states`), and with `well` a double
/// well in the length of `b` (a saddle point at `|b| = 5.5`) in place of its harmonic term.
struct Model {
    well: bool,
}

impl Calculator for Model {
    fn name(&self) -> &str {
        "model"
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let s = usize::from(sys.spin_charge.multiplicity == 3 || req.excited_state == 1);
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("the model needs a periodic cell"))?;
        let a: [[f64; 3]; 3] = std::array::from_fn(|k| cell.lattice_vector(k).value());
        let mut e = C[s];
        for (k, v) in a.iter().enumerate() {
            let u = norm(*v) - L[s];
            e += if self.well && k == 1 {
                let w = norm(*v) - 5.5;
                0.05 * (w * w - 0.09).powi(2)
            } else {
                K * u * u
            };
        }
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            let c = dot(&a[i], &a[j]) / (norm(a[i]) * norm(a[j]));
            e += A * c * c;
        }
        let x = sys.coords_flat_bohr();
        let d = [x[3] - x[0], x[4] - x[1], x[5] - x[2]];
        let r = norm(d);
        let axis = a[0].map(|v| v / norm(a[0]));
        let parallel = dot(&d, &axis);
        let perpendicular: [f64; 3] = std::array::from_fn(|i| d[i] - parallel * axis[i]);
        e += KB * ((r - R[s]).powi(2) + dot(&perpendicular, &perpendicular));
        let f = 2.0 * KB * (r - R[s]) / r;
        let mut g = Col::<f64>::zeros(6);
        for c in 0..3 {
            g[3 + c] = f * d[c] + 2.0 * KB * perpendicular[c];
            g[c] = -g[3 + c];
        }
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        if req.need_hessian {
            let radial = 2.0 * KB * R[s] / r.powi(3);
            out.hessian_bohr = Some(Mat::from_fn(6, 6, |i, j| {
                let (row, col) = (i % 3, j % 3);
                let diagonal = if row == col { f + 2.0 * KB } else { 0.0 };
                let block = diagonal + radial * d[row] * d[col] - 2.0 * KB * axis[row] * axis[col];
                if i / 3 == j / 3 {
                    block
                } else {
                    -block
                }
            }));
        }
        Ok(out)
    }
}

fn model_registry(well: bool) -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    r.register("model", move |_| Ok(Box::new(Model { well })));
    r
}

fn pair(dir: &Path, edges: [f64; 3]) -> PathBuf {
    let [a, b, c] = edges.map(|e| e * Units::BOHR_TO_ANGSTROM);
    let r = 2.3 * Units::BOHR_TO_ANGSTROM;
    let path = dir.join("pair.xyz");
    std::fs::write(
        &path,
        format!(
            "2\nLattice=\"{a} 0 0 0.05 {b} 0 0 0.03 {c}\" pbc=\"T T T\"\nAr 0.3 0.4 0.5\nAr {} 0.4 0.5\n",
            0.3 + r
        ),
    )
    .unwrap();
    path
}

fn optimize_model(dir: &Path, input: &Path, well: bool, extra: &[&str]) -> MolecularSystem {
    let (converged, sys) = try_model(dir, input, well, extra);
    assert!(converged, "{extra:?} did not converge");
    sys
}

fn try_model(dir: &Path, input: &Path, well: bool, extra: &[&str]) -> (bool, MolecularSystem) {
    let input = input.to_string_lossy().into_owned();
    let mut args = vec![input.as_str(), "-os", "model", "-vc", "-ns", "400"];
    if !extra.contains(&"-order") {
        args.push("-tcc");
    }
    args.extend_from_slice(extra);
    let config = config_from_cli(&args).unwrap();
    let out = run_with(&config, dir, None, &model_registry(well)).unwrap();
    (out[0].converged, out[0].summary.clone().unwrap().system)
}

#[test]
fn two_states_an_excited_state_and_a_saddle_point_with_the_cell_held() {
    let dir = scratch("states");
    let input = pair(&dir, [5.3, 5.3, 5.3]);
    let start = read_xyz_file(&input, SpinCharge::default(), None).unwrap();
    let l0 = lattice(&start);
    // Near the crossing of the two states (the penalty of the seam) with the length of a
    // held.
    let s = optimize_model(
        &dir,
        &input,
        false,
        &["-mf", "seam", "0", "3", "-vc_fix", "a"],
    );
    assert!((norm(lattice(&s)[0]) - norm(l0[0])).abs() < 1e-10);
    let mut triplet = s.clone();
    triplet.spin_charge.multiplicity = 3;
    let model = Model { well: false };
    let e = |m: &MolecularSystem| {
        model
            .evaluate(m, &EvalRequest::ENERGY_ONLY)
            .unwrap()
            .energy
            .value()
    };
    let mut far = start.clone();
    far.spin_charge.multiplicity = 3;
    let (before, after) = (e(&start) - e(&far), e(&s) - e(&triplet));
    assert!(after.abs() < 0.25 * before.abs(), "gap {before} -> {after}");
    // A gradient projection with constraints in a variable cell is refused with the reason.
    let args = [
        input.to_str().unwrap(),
        "-os",
        "model",
        "-vc",
        "-mf",
        "opt_mesx",
        "0",
        "3",
        "-vc_fix",
        "a",
    ];
    let config = config_from_cli(&args).unwrap();
    let err = run_with(&config, &dir, None, &model_registry(false))
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("gradient projection") && err.contains("seam"),
        "{err}"
    );
    // The excited state with the volume held.
    let s = optimize_model(&dir, &input, false, &["-es", "1", "-vc_fix", "volume"]);
    let (v0, v1) = (
        start.active_cell().unwrap().volume_bohr3(),
        s.active_cell().unwrap().volume_bohr3(),
    );
    assert!((v1 - v0).abs() < 1e-9 * v0, "{v0} {v1}");
    // A saddle point of the length of b, with the length of a held away from its minimum.
    let input = pair(&dir, [5.3, 5.56, 5.0]);
    let start = read_xyz_file(&input, SpinCharge::default(), None).unwrap();
    let (free, sys) = try_model(
        &dir,
        &input,
        true,
        &["-order", "1", "-fc", "1", "-opt", "rsirfo_bofill"],
    );
    // The same search without constraints, for reference. With a held length the curvature
    // of the constraint must leave the rotations out (before that was done the constrained
    // search oscillated at steps of 7e-3 Bohr with forces of 1.3e-5 for 400 iterations).
    assert!(free && (norm(lattice(&sys)[1]) - 5.5).abs() < 1e-3);
    let s = optimize_model(
        &dir,
        &input,
        true,
        &[
            "-order",
            "1",
            "-fc",
            "1",
            "-opt",
            "rsirfo_bofill",
            "-vc_fix",
            "a",
        ],
    );
    let l = lattice(&s);
    assert!((norm(l[0]) - norm(lattice(&start)[0])).abs() < 1e-10);
    assert!((norm(l[1]) - 5.5).abs() < 1e-3, "|b| = {}", norm(l[1]));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_of_the_cell_need_a_variable_cell_in_a_scan_as_well() {
    let args = [
        "fcc.xyz",
        "-os",
        "lj",
        "-scan",
        "cell",
        "volume",
        "1100,1200",
    ];
    let err = mopt_rs::cli::relaxedscan::config_from_cli(&args)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("add -vc"), "{err}");
    let args = [
        "fcc.xyz", "-os", "lj", "-vc", "-scan", "cell", "shape", "1,2",
    ];
    assert!(mopt_rs::cli::relaxedscan::config_from_cli(&args).is_err());
    // A user constraint parsed directly holds the input value.
    let c = user_constraint("volume()").unwrap();
    assert!(matches!(c.kind, ConstraintKind::Function(_)) && c.target.is_none());
}
