//! Tests of the replicas of the environment and their soft minimum.
//!
//! The model: one atom of the core at `x` (the model region) and one of the environment at
//! `u + OFFSET` along the axis, with
//!
//! ```text
//! E = C4 (x^2 - 1)^2 + A (u^2 - 1)^2 + B x g(u) + K / 2 (y_1^2 + z_1^2 + y_2^2 + z_2^2),
//! g(u) = u (3 - u^2) / 2,
//! ```
//!
//! whose environment has two minima at `u = +1` and `u = -1` for every `x` (`g'(+-1) = 0`), of
//! the energies `E_+-(x) = C4 (x^2 - 1)^2 +- B x`: the wells cross at `x = 0`. The coupling
//! `B g'(u)` of `x` and `u` vanishes at both minima, so the effective Hessian of each is
//! `f''(x)`, and the soft minimum is `E_soft = f - kT ln(2 cosh(beta B x))` with the gradient
//! `f' - B tanh(beta B x)` and the curvature `f'' - beta B^2 sech^2(beta B x)` along `x`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::replicas::{soft_hessian, soft_minimum, ReplicaStart, NEGLIGIBLE_WEIGHT};
use super::*;
use crate::domain::{AtomId, Element, Units};
use crate::pes::oniom::{LinkSpec, MicroOptions, OniomSpec, RegionSpec};
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};

const C4: f64 = 0.002;
const A: f64 = 0.01;
const B: f64 = 1e-3;
const K: f64 = 0.05;
const OFFSET: f64 = 6.0;

/// The model surface above (one value for a single atom, the model system: zero).
struct DoubleWell;

impl Calculator for DoubleWell {
    fn name(&self) -> &str {
        "double_well"
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let n = sys.dof();
        let p = sys.coords_flat_bohr();
        let mut g = Col::<f64>::zeros(n);
        let mut h = Mat::<f64>::zeros(n, n);
        let mut e = 0.0;
        if sys.n_atoms() == 2 {
            let (x, u) = (p[0], p[3] - OFFSET);
            let gu = u * (3.0 - u * u) / 2.0;
            let dgu = 1.5 * (1.0 - u * u);
            e = C4 * (x * x - 1.0).powi(2) + A * (u * u - 1.0).powi(2) + B * x * gu;
            g[0] = 4.0 * C4 * x * (x * x - 1.0) + B * gu;
            g[3] = 4.0 * A * u * (u * u - 1.0) + B * x * dgu;
            h[(0, 0)] = 4.0 * C4 * (3.0 * x * x - 1.0);
            h[(0, 3)] = B * dgu;
            h[(3, 0)] = B * dgu;
            h[(3, 3)] = 4.0 * A * (3.0 * u * u - 1.0) - 3.0 * B * x * u;
            for i in [1, 2, 4, 5] {
                e += 0.5 * K * p[i] * p[i];
                g[i] = K * p[i];
                h[(i, i)] = K;
            }
        }
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        if req.need_hessian {
            out.hessian_bohr = Some(h);
        }
        Ok(out)
    }
}

fn registry() -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    r.register("double_well", |_| Ok(Box::new(DoubleWell)));
    r
}

fn whole(x: f64, u: f64) -> Vec<f64> {
    vec![x, 0.0, 0.0, OFFSET + u, 0.0, 0.0]
}

fn ar() -> Element {
    Element::parse_token("Ar").unwrap()
}

/// The surface of the model with the environment starting at `u0`.
fn model_surface(settings: MicroOniomSettings, u0: f64) -> RelaxedEnvironment {
    let sys = MolecularSystem::from_flat_bohr(
        vec![ar(), ar()],
        &Col::from_fn(6, |i| whole(0.3, u0)[i]),
        crate::domain::SpinCharge::default(),
        None,
    )
    .unwrap();
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: vec![AtomId::new(0)],
            links: LinkSpec::None,
            spin_charge: None,
        }],
        lower_levels: vec!["double_well".to_string()],
        micro: Some(MicroOptions::default()),
        ..OniomSpec::default()
    };
    let registry = registry();
    let options = CalculatorOptions {
        oniom: Some(spec.clone()),
        ..CalculatorOptions::default()
    };
    let oniom = registry
        .create_oniom("double_well", &options, &sys)
        .unwrap();
    let low = registry
        .create("double_well", &CalculatorOptions::default(), &sys)
        .unwrap();
    let split = CoreSplit::of(&spec.resolve(&sys).unwrap(), sys.n_atoms(), &[]).unwrap();
    RelaxedEnvironment::new(oniom, low, sys, split, settings).unwrap()
}

/// Settings of tight relaxations without a preconditioner, with `k` replicas at `t` K that
/// start in the two wells.
fn settings(k: usize, t: f64) -> MicroOniomSettings {
    let mut s = MicroOptions::default().settings_for((1e-10, f64::INFINITY));
    s.precondition = None;
    s.replicas.count = k;
    s.replicas.temperature = t;
    s.replicas.start = ReplicaStart::Structures(vec![whole(0.3, 1.0), whole(0.3, -1.0)]);
    s
}

fn core_at(x: f64) -> MolecularSystem {
    MolecularSystem::from_flat_bohr(
        vec![ar()],
        &Col::from_fn(3, |i| [x, 0.02, -0.01][i]),
        crate::domain::SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn beta(t: f64) -> f64 {
    1.0 / (Units::BOLTZMANN_HARTREE_PER_K * t)
}

fn f(x: f64) -> f64 {
    C4 * (x * x - 1.0).powi(2)
}

fn df(x: f64) -> f64 {
    4.0 * C4 * x * (x * x - 1.0)
}

fn d2f(x: f64) -> f64 {
    4.0 * C4 * (3.0 * x * x - 1.0)
}

/// `E_soft`, its derivative, and its curvature along `x` in closed form.
fn closed(x: f64, t: f64) -> (f64, f64, f64) {
    let b = beta(t);
    let tanh = (b * B * x).tanh();
    // ln(2 cosh(z)) = ln 2 + ln cosh(z).
    let e = f(x) - (std::f64::consts::LN_2 + crate::math::stable::ln_cosh(b * B * x)) / b
        + 0.5 * K * (0.02_f64.powi(2) + 0.01_f64.powi(2));
    (
        e,
        df(x) - B * tanh,
        d2f(x) - b * B * B * (1.0 - tanh * tanh),
    )
}

#[test]
fn the_soft_minimum_of_given_energies_has_its_limits() {
    let e = [-1.0, -1.0 + 4e-4, -0.9];
    let s = soft_minimum(&e, 298.15).unwrap();
    let kt = 1.0 / s.beta;
    let sum: f64 = s.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-15);
    assert_eq!(s.lowest, 0);
    // Below the lowest by at most kT ln K.
    assert!(s.energy <= -1.0 && s.energy >= -1.0 - kt * 3.0_f64.ln());
    // By hand, shifted by the lowest energy (unshifted, exp(1 / kT) overflows).
    let by_hand = -1.0 - kt * e.iter().map(|v| (-(v + 1.0) / kt).exp()).sum::<f64>().ln();
    assert!((s.energy - by_hand).abs() < 1e-12, "{} {by_hand}", s.energy);
    // One replica: its own energy, exactly.
    assert_eq!(soft_minimum(&[-3.25], 298.15).unwrap().energy, -3.25);
    // Equal energies: kT ln K below.
    let two = soft_minimum(&[-2.0, -2.0], 298.15).unwrap();
    assert!((two.energy - (-2.0 - kt * 2.0_f64.ln())).abs() < 1e-15);
    // A low temperature: the lowest, the others of no weight.
    let cold = soft_minimum(&e, 1e-3).unwrap();
    assert_eq!(cold.energy, -1.0);
    assert!(cold.weights[1] < NEGLIGIBLE_WEIGHT && cold.weights[0] == 1.0);
    // Energies far apart and large do not overflow.
    let far = soft_minimum(&[-5000.0, 0.0, 1e6], 10.0).unwrap();
    assert_eq!(far.energy, -5000.0);
    assert!(soft_minimum(&[], 300.0).is_err());
    assert!(soft_minimum(&[1.0], 0.0).is_err());
    assert!(soft_minimum(&[f64::NAN], 300.0).is_err());
    // The Hessian: a replica of negligible weight may have none, one of weight may not.
    let g = [Col::from_fn(1, |_| 0.1), Col::from_fn(1, |_| -0.2)];
    let gr: Vec<&Col<f64>> = g.iter().collect();
    let h = Mat::from_fn(1, 1, |_, _| 0.5);
    let s = soft_minimum(&[0.0, 1.0], 298.15).unwrap();
    assert!(soft_hessian(&[Some(&h), None], &gr, &s).is_ok());
    let s = soft_minimum(&[0.0, 0.0], 298.15).unwrap();
    assert!(soft_hessian(&[Some(&h), None], &gr, &s).is_err());
    // sum w H - beta (sum w g g^T - g g^T), the covariance form against the plain one.
    let both = soft_hessian(&[Some(&h), Some(&h)], &gr, &s).unwrap();
    let mean = 0.5 * (0.1 - 0.2);
    let plain = 0.5 - s.beta * (0.5 * (0.01 + 0.04) - mean * mean);
    assert!((both[(0, 0)] - plain).abs() < 1e-12 * plain.abs().max(1.0));
}

#[test]
fn two_replicas_of_a_double_well_environment_give_the_closed_forms() {
    let t = 298.15;
    let s = model_surface(settings(2, t), 1.0);
    assert_eq!(s.replica_count(), 2);
    let req = EvalRequest::ENERGY_GRAD_HESS;
    for x in [-1.3, -0.4, -0.05, 0.0, 0.07, 0.5, 1.1] {
        let out = s.evaluate(&core_at(x), &req).unwrap();
        let (e, g, h) = closed(x, t);
        assert!(
            (out.energy.value() - e).abs() < 1e-12,
            "x = {x}: E {} against {e}",
            out.energy.value()
        );
        assert!((out.gradient_bohr[0] - g).abs() < 1e-12, "x = {x}: g");
        assert!((out.gradient_bohr[1] - K * 0.02).abs() < 1e-12);
        let hess = out.hessian_bohr.unwrap();
        assert!(
            (hess[(0, 0)] - h).abs() < 1e-10,
            "x = {x}: H {} against {h}",
            hess[(0, 0)]
        );
        assert!((hess[(1, 1)] - K).abs() < 1e-10 && hess[(0, 1)].abs() < 1e-10);
    }
    // Central differences of E_soft and of its gradient, the environments relaxed again at
    // every displaced core.
    let h = 1e-4;
    for x in [-0.6, 0.02, 0.8] {
        let e = |x: f64| s.evaluate(&core_at(x), &EvalRequest::ENERGY_GRAD).unwrap();
        let (plus, minus, at) = (e(x + h), e(x - h), e(x));
        let numeric = (plus.energy.value() - minus.energy.value()) / (2.0 * h);
        assert!((numeric - at.gradient_bohr[0]).abs() < 1e-9, "x = {x}");
        let curvature = (plus.gradient_bohr[0] - minus.gradient_bohr[0]) / (2.0 * h);
        let analytic = s.evaluate(&core_at(x), &req).unwrap().hessian_bohr.unwrap()[(0, 0)];
        assert!((curvature - analytic).abs() < 1e-8, "x = {x}");
    }
}

#[test]
fn where_the_wells_cross_the_lowest_well_has_a_kink_and_the_soft_minimum_none() {
    // One environment in each well (the surfaces of one environment); their minimum is the
    // surface of whichever relaxation finds the lower well.
    let plus = model_surface(settings(1, 298.15), 1.0);
    let minus = model_surface(settings(1, 298.15), -1.0);
    let lowest = |x: f64| {
        let a = plus
            .evaluate(&core_at(x), &EvalRequest::ENERGY_GRAD)
            .unwrap();
        let b = minus
            .evaluate(&core_at(x), &EvalRequest::ENERGY_GRAD)
            .unwrap();
        if a.energy.value() <= b.energy.value() {
            a.gradient_bohr[0]
        } else {
            b.gradient_bohr[0]
        }
    };
    let soft = model_surface(settings(2, 298.15), 1.0);
    let smooth = |x: f64| {
        soft.evaluate(&core_at(x), &EvalRequest::ENERGY_GRAD)
            .unwrap()
            .gradient_bohr[0]
    };
    for d in [1e-3, 1e-5, 1e-7] {
        // The gradient of the lowest well jumps by 2 B across the crossing (besides the change
        // of f' over 2 d).
        let jump = lowest(d) - lowest(-d);
        assert!(
            (jump - (2.0 * d * d2f(0.0) - 2.0 * B)).abs() < 1e-9,
            "d = {d}: jump {jump}"
        );
        // That of the soft minimum goes to zero with the distance, as its curvature says.
        let step = smooth(d) - smooth(-d);
        let (_, _, h0) = closed(0.0, 298.15);
        assert!((step - 2.0 * d * h0).abs() < 1e-3 * d, "d = {d}: {step}");
    }
}

#[test]
fn at_a_low_temperature_the_soft_minimum_is_the_lower_well() {
    let s = model_surface(settings(2, 0.01), 1.0);
    for x in [-0.5, 0.3] {
        let out = s
            .evaluate(&core_at(x), &EvalRequest::ENERGY_GRAD_HESS)
            .unwrap();
        // The lower well: u = -1 for x > 0, E = f - B |x|.
        let e = f(x) - B * x.abs() + 0.5 * K * (0.02_f64.powi(2) + 0.01_f64.powi(2));
        assert!((out.energy.value() - e).abs() < 1e-14);
        assert!((out.gradient_bohr[0] - (df(x) - B * x.signum())).abs() < 1e-14);
        assert!((out.hessian_bohr.unwrap()[(0, 0)] - d2f(x)).abs() < 1e-10);
        // The upper replica has no weight, and its Hessian was not taken.
    }
    assert!(s
        .replica_summary()
        .unwrap()
        .contains("2 Hessian(s) of replicas of negligible weight not taken"));
}

/// Three waters (the first GFN0-xTB, all GFN-FF), as `tests.rs` has them.
fn trimer(settings: MicroOniomSettings) -> RelaxedEnvironment {
    let sys = crate::io::xyz::parse_xyz_str(
        "9\n0 1\nO 0.000 0.000 0.000\nH 0.960 0.000 0.000\nH -0.240 0.930 0.000\n\
         O 2.900 0.000 0.000\nH 3.200 0.900 0.000\nH 3.200 -0.400 0.800\nO 0.000 2.900 0.000\n\
         H 0.900 3.200 0.000\nH -0.400 3.200 0.800\n",
        crate::domain::SpinCharge::default(),
        None,
    )
    .unwrap();
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: (0..3).map(AtomId::new).collect(),
            links: LinkSpec::None,
            spin_charge: None,
        }],
        lower_levels: vec!["gfnff".to_string()],
        micro: Some(MicroOptions::default()),
        ..OniomSpec::default()
    };
    let registry = CalculatorRegistry::native();
    let options = CalculatorOptions {
        oniom: Some(spec.clone()),
        ..CalculatorOptions::default()
    };
    let oniom = registry.create_oniom("gfn0xtb", &options, &sys).unwrap();
    let low = registry
        .create("gfnff", &CalculatorOptions::default(), &sys)
        .unwrap();
    let split = CoreSplit::of(&spec.resolve(&sys).unwrap(), sys.n_atoms(), &[]).unwrap();
    RelaxedEnvironment::new(oniom, low, sys, split, settings).unwrap()
}

fn bits(out: &EvalOutput) -> Vec<u64> {
    let mut v = vec![out.energy.value().to_bits()];
    v.extend(out.gradient_bohr.iter().map(|g| g.to_bits()));
    if let Some(h) = &out.hessian_bohr {
        for j in 0..h.ncols() {
            for i in 0..h.nrows() {
                v.push(h[(i, j)].to_bits());
            }
        }
    }
    v
}

#[test]
fn one_replica_is_the_surface_of_one_environment_bit_for_bit() {
    let base = MicroOptions::default().settings_for((3e-4, 2e-4));
    // One replica with every other setting of the replicas changed: none of them is used.
    let mut one = base.clone();
    one.replicas.count = 1;
    one.replicas.temperature = 10.0;
    one.replicas.seed = 99;
    one.replicas.amplitude = 0.5;
    let run = |settings: MicroOniomSettings| {
        let s = trimer(settings);
        let core = s.core_system().unwrap();
        let mut out = Vec::new();
        for step in 0..3 {
            let x = core.coords_flat_bohr();
            let moved = core
                .with_flat_coords_bohr(&Col::from_fn(9, |k| {
                    x[k] + 0.02 * step as f64 * ((k % 4) as f64 - 1.5)
                }))
                .unwrap();
            let req = if step == 1 {
                EvalRequest::ENERGY_GRAD_HESS
            } else {
                EvalRequest::ENERGY_GRAD
            };
            out.push(bits(&s.evaluate(&moved, &req).unwrap()));
        }
        // An optimization of the core: the same iterations.
        let mut opt = crate::opt::state_machine::GeometryOptimizer::new(
            crate::opt::state_machine::OptimizerSettings {
                max_iterations: 15,
                ..Default::default()
            },
            core,
        )
        .unwrap();
        let outcome = crate::workflow::run_optimization(&mut opt, &s, &mut []).unwrap();
        out.push(vec![outcome.iterations as u64]);
        out.push(
            opt.current()
                .coords_flat_bohr()
                .iter()
                .map(|v| v.to_bits())
                .collect(),
        );
        (out, s.counts(), s.description())
    };
    let (a, ca, da) = run(base);
    let (b, cb, db) = run(one);
    assert_eq!(a, b);
    assert_eq!(ca, cb);
    assert_eq!(da, db);
}

#[test]
fn the_replicas_of_a_mechanical_embedding_compute_the_model_once_and_agree_with_the_surface() {
    let mut s = MicroOptions::default().settings_for((1e-6, f64::INFINITY));
    s.replicas.count = 3;
    s.replicas.amplitude = 0.3 * Units::ANGSTROM_TO_BOHR;
    let surface = trimer(s);
    let core = surface.core_system().unwrap();
    let out = surface
        .evaluate(&core, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap();
    let (lowest, systems) = surface.replica_systems(&core).unwrap().unwrap();
    assert_eq!(systems.len(), 3);
    // Each replica energy (the lowest level per replica and the model systems once) is that
    // of the whole ONIOM surface at its structure.
    let mut energies = Vec::new();
    for (full, e, _) in &systems {
        // The surface's own ONIOM calculator (a GFN-FF made anew would take its topology
        // from this structure).
        let direct = surface
            .oniom
            .evaluate(full, &EvalRequest::ENERGY_GRAD)
            .unwrap();
        // The corrections of the residual of the relaxations are of the order of 1e-12 here.
        assert!(
            (direct.energy.value() - e).abs() < 1e-9,
            "{} {e}",
            direct.energy.value()
        );
        energies.push(*e);
    }
    let soft = soft_minimum(&energies, super::replicas::REPLICA_TEMPERATURE).unwrap();
    assert_eq!(soft.lowest, lowest);
    assert!((out.energy.value() - soft.energy).abs() < 1e-12);
    // The Hessian is symmetric.
    let h = out.hessian_bohr.unwrap();
    for i in 0..9 {
        for j in 0..9 {
            assert!((h[(i, j)] - h[(j, i)]).abs() < 1e-9);
        }
    }
    // A restart: the journal records every replica, and a surface told them relaxes nothing.
    let env = surface.relaxed_environment(&core).unwrap();
    assert_eq!(env.len(), 3 * 18);
    let mut again_settings = MicroOptions::default().settings_for((1e-6, f64::INFINITY));
    again_settings.replicas.count = 3;
    again_settings.replicas.amplitude = 0.3 * Units::ANGSTROM_TO_BOHR;
    let again = trimer(again_settings);
    again.recall_environment(&core, &env);
    let replayed = again.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
    assert_eq!(again.counts().relaxations, 0);
    assert_eq!(again.counts().recalled, 3);
    assert!((replayed.energy.value() - out.energy.value()).abs() < 1e-12);
}

#[test]
fn the_replicas_are_the_same_for_any_number_of_threads() {
    let run = || -> Vec<Vec<u64>> {
        let mut s = MicroOptions::default().settings_for((3e-5, f64::INFINITY));
        s.replicas.count = 4;
        s.replicas.amplitude = 0.2 * Units::ANGSTROM_TO_BOHR;
        let surface = trimer(s);
        let core = surface.core_system().unwrap();
        let x = core.coords_flat_bohr();
        (0..3)
            .map(|step| {
                let moved = core
                    .with_flat_coords_bohr(&Col::from_fn(9, |k| {
                        x[k] + 0.03 * step as f64 * ((k % 3) as f64 - 1.0)
                    }))
                    .unwrap();
                let req = if step == 2 {
                    EvalRequest {
                        for_stability: true,
                        ..EvalRequest::ENERGY_GRAD_HESS
                    }
                } else {
                    EvalRequest::ENERGY_GRAD
                };
                bits(&surface.evaluate(&moved, &req).unwrap())
            })
            .collect()
    };
    let mut first: Option<Vec<Vec<u64>>> = None;
    for threads in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let got = pool.install(run);
        match &first {
            None => first = Some(got),
            Some(f) => assert_eq!(f, &got, "{threads} threads"),
        }
    }
}

#[test]
fn a_run_with_replicas_writes_every_replica_and_a_later_run_says_whose_surface_it_is_on() {
    let dir = std::env::temp_dir().join(format!("mopt_replica_files_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let s = model_surface(settings(2, 298.15), 1.0);
    let core = core_at(0.2);
    s.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
    std::fs::write(
        dir.join("optimized.xyz"),
        "1\nE= -1.5\nAr 0.10583544 0.01058354 -0.00529177\n",
    )
    .unwrap();
    assert_eq!(write_full_structures(&dir, &s).unwrap(), 1);
    let main = std::fs::read_to_string(dir.join("optimized.xyz")).unwrap();
    // The lowest replica at x > 0 is the well u = -1, the second; its weight is above a half.
    assert!(main.contains("micro_replica=2/2 weight=0.60"), "{main}");
    assert!(main.contains("E= -1.5"), "{main}");
    let all = std::fs::read_to_string(dir.join("optimized_replicas.xyz")).unwrap();
    assert_eq!(all.matches("micro_replica=").count(), 2, "{all}");
    let frames = crate::io::xyz::parse_xyz_trajectory_str(&all, Default::default(), None).unwrap();
    let u: Vec<f64> = frames
        .frames
        .iter()
        .map(|f| f.system.coords_flat_bohr()[3] - OFFSET)
        .collect();
    assert!(
        (u[0] - 1.0).abs() < 1e-6 && (u[1] + 1.0).abs() < 1e-6,
        "{u:?}"
    );
    // A run from the main file with one environment says that it is not on that surface;
    // one that takes the file of the replicas does not.
    let written = crate::io::xyz::parse_xyz_str(&main, Default::default(), None).unwrap();
    let notes_of = |settings: MicroOniomSettings| {
        let spec = OniomSpec {
            regions: vec![RegionSpec {
                atoms: vec![AtomId::new(0)],
                links: LinkSpec::None,
                spin_charge: None,
            }],
            lower_levels: vec!["double_well".to_string()],
            micro: Some(MicroOptions::default()),
            ..OniomSpec::default()
        };
        let registry = registry();
        let options = CalculatorOptions {
            oniom: Some(spec.clone()),
            ..CalculatorOptions::default()
        };
        let oniom = registry
            .create_oniom("double_well", &options, &written)
            .unwrap();
        let low = registry
            .create("double_well", &CalculatorOptions::default(), &written)
            .unwrap();
        let split =
            CoreSplit::of(&spec.resolve(&written).unwrap(), written.n_atoms(), &[]).unwrap();
        RelaxedEnvironment::new(oniom, low, written.clone(), split, settings)
            .unwrap()
            .setup_notes()
            .join("\n")
    };
    let one = notes_of(settings(1, 298.15));
    assert!(one.contains("not on that surface"), "{one}");
    let mut files = settings(2, 298.15);
    files.replicas.start = ReplicaStart::Files(dir.join("optimized_replicas.xyz"));
    let same = notes_of(files);
    assert!(!same.contains("not on that surface"), "{same}");
    let _ = std::fs::remove_dir_all(&dir);
}
