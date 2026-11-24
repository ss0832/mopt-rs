//! Combined surfaces: crossing points (MESX, MECI, seam penalty), EDEEL, and bias terms, checked
//! against analytic optima and finite differences.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use faer::{Col, Mat};
use mopt_rs::domain::{positions, AtomId, Bohr, EvalOutput, EvalRequest, Quantity};
use mopt_rs::domain::{Element, MolecularSystem, SpinCharge};
use mopt_rs::error::MoptResult;
use mopt_rs::math::ad::{eval_value_grad_hessian, Scalar};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::pes::combine::{
    edeel, edeel_with, model_function, two_states, EdeelLayout, EdeelStates, EdeelTarget,
    FunctionCombiner, QuadraticPenalty, TaskSpec,
};
use mopt_rs::pes::{Calculator, PesExpr, PesTerm, TermContext};
use mopt_rs::workflow::{run_optimization, CalculatorSet};

/// Evaluate `f` (a function of the flat coordinates) with exact derivatives.
fn ad_output<F>(sys: &MolecularSystem, req: &EvalRequest, f: F) -> EvalOutput
where
    F: FnMut(&[mopt_rs::math::ad::HyperDual]) -> mopt_rs::math::ad::HyperDual,
{
    let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
    let (e, g, h) = eval_value_grad_hessian(&x, None, f);
    EvalOutput {
        energy: Quantity::new(e),
        gradient_bohr: g,
        hessian_bohr: req.need_hessian.then_some(h),
        orbitals: None,
        stress: None,
        strain_hessian: None,
    }
}

/// One atom on two harmonic diabatic surfaces
/// `V1 = k/2 |r - a|^2`, `V2 = k/2 |r - b|^2 + c`, coupled by `V12 = lambda z`.
///
/// With `adiabatic = false` the state is chosen by the multiplicity (singlet: V1, triplet: V2),
/// as for a spin crossing. With `adiabatic = true` the electronic root selects the lower or upper
/// adiabatic surface, which meet at a conical intersection where `V1 = V2` and `z = 0`.
struct CrossingModel {
    k: f64,
    a: [f64; 3],
    b: [f64; 3],
    c: f64,
    lambda: f64,
    adiabatic: bool,
}

impl CrossingModel {
    fn energy<T: Scalar>(&self, state: usize, r: &[T]) -> T {
        let half_k = T::from_f64(0.5 * self.k);
        let sq = |p: [f64; 3]| {
            (0..3).fold(T::zero(), |acc, i| {
                let d = r[i] - T::from_f64(p[i]);
                acc + d * d
            })
        };
        let v1 = half_k * sq(self.a);
        let v2 = half_k * sq(self.b) + T::from_f64(self.c);
        if !self.adiabatic {
            return if state == 0 { v1 } else { v2 };
        }
        let mean = (v1 + v2) * T::from_f64(0.5);
        let half_gap = (v1 - v2) * T::from_f64(0.5);
        let v12 = r[2] * T::from_f64(self.lambda);
        // A tiny shift keeps the derivatives finite exactly at the intersection.
        let root = (half_gap * half_gap + v12 * v12 + T::from_f64(1e-16)).sqrt();
        if state == 0 {
            mean - root
        } else {
            mean + root
        }
    }
}

impl Calculator for CrossingModel {
    fn name(&self) -> &str {
        "crossing_model"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let state = if self.adiabatic {
            req.excited_state
        } else {
            usize::from(sys.spin_charge.multiplicity == 3)
        };
        Ok(ad_output(sys, req, |r| self.energy(state, r)))
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }

    fn supports_excited_states(&self) -> bool {
        self.adiabatic
    }
}

fn atom(r: [f64; 3]) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::H],
        &positions::<Bohr>(&[r]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn settings(methods: &[&str]) -> OptimizerSettings {
    OptimizerSettings {
        methods: methods.iter().map(|s| s.to_string()).collect(),
        max_iterations: 500,
        criteria: ConvergenceCriteria::TIGHT,
        rigid_body_invariant: false,
        dissociation_distance: None,
        ..OptimizerSettings::default()
    }
}

fn position(sys: &MolecularSystem) -> [f64; 3] {
    sys.position(AtomId::new(0)).value()
}

fn dist(p: [f64; 3], q: [f64; 3]) -> f64 {
    (0..3).map(|i| (p[i] - q[i]).powi(2)).sum::<f64>().sqrt()
}

fn spin_crossing() -> CrossingModel {
    CrossingModel {
        k: 1.0,
        a: [0.0, 0.0, 0.0],
        b: [1.0, 0.5, 0.0],
        c: 0.02,
        lambda: 0.0,
        adiabatic: false,
    }
}

/// Run `surface` with `calc` and return the termination, final position, and iterations.
fn optimize(
    methods: &[&str],
    start: [f64; 3],
    surface: PesExpr,
    calc: &dyn Calculator,
) -> (Termination, [f64; 3], usize) {
    let mut opt = GeometryOptimizer::new(settings(methods), atom(start)).unwrap();
    opt.set_surface(surface).unwrap();
    let out = run_optimization(&mut opt, calc, &mut []).unwrap();
    (
        out.termination.unwrap(),
        position(&out.system),
        out.iterations,
    )
}

#[test]
fn mesx_reaches_the_analytic_minimum_on_the_seam() {
    let m = spin_crossing();
    // Seam: n . (r - mid) = c / k with n = b - a; its lowest point is mid + n c / (k |n|^2).
    let n = [1.0, 0.5, 0.0];
    let n2 = 1.25;
    let mid = [0.5, 0.25, 0.0];
    let target: [f64; 3] = std::array::from_fn(|i| mid[i] + n[i] * m.c / (m.k * n2));
    for method in ["rsirfo_bfgs", "FIRELARS"] {
        let surface = two_states(
            model_function("opt_mesx", None, None).unwrap(),
            TaskSpec::whole("State2").spin_charge(SpinCharge::new(0, 3)),
        )
        .unwrap();
        let (t, p, it) = optimize(&[method], [0.1, -0.3, 0.2], surface, &m);
        assert_eq!(t, Termination::Converged, "{method}");
        assert!(
            dist(p, target) < 1e-4,
            "{method}: {p:?} vs {target:?} after {it}"
        );
    }
}

#[test]
fn seam_penalty_reaches_its_analytic_optimum() {
    let m = spin_crossing();
    let alpha = 0.05;
    let w = 1.0 / alpha;
    let n = [1.0, 0.5, 0.0];
    let n2 = 1.25;
    let mid = [0.5, 0.25, 0.0];
    // Stationary point of mean + w dE^2: r = mid - 2 w dE n with dE = -c / (1 + 2 w k |n|^2).
    let de = -m.c / (1.0 + 2.0 * w * m.k * n2);
    let target: [f64; 3] = std::array::from_fn(|i| mid[i] - 2.0 * w * de * n[i]);
    let surface = two_states(
        model_function("seam", Some(alpha), None).unwrap(),
        TaskSpec::whole("State2").spin_charge(SpinCharge::new(0, 3)),
    )
    .unwrap();
    let (t, p, it) = optimize(&["rsirfo_bfgs"], [0.1, -0.3, 0.2], surface, &m);
    assert_eq!(t, Termination::Converged);
    assert!(dist(p, target) < 1e-4, "{p:?} vs {target:?} after {it}");
}

#[test]
fn meci_reaches_the_conical_intersection() {
    let m = CrossingModel {
        k: 1.0,
        a: [0.0, 0.0, 0.3],
        b: [1.0, 0.0, 0.3],
        c: 0.02,
        lambda: 0.1,
        adiabatic: true,
    };
    // Intersection seam: x = 0.5 + c / k, z = 0; the mean energy is lowest at y = 0.
    let target = [0.52, 0.0, 0.0];
    let surface = two_states(
        model_function("opt_meci", None, None).unwrap(),
        TaskSpec::whole("State2").excited_state(1),
    )
    .unwrap();
    let mut opt =
        GeometryOptimizer::new(settings(&["rsirfo_bfgs"]), atom([0.3, 0.2, 0.15])).unwrap();
    opt.set_surface(surface).unwrap();
    let out = run_optimization(&mut opt, &m, &mut []).unwrap();
    let p = position(&out.system);
    assert_eq!(out.termination, Some(Termination::Converged));
    assert!(dist(p, target) < 1e-3, "{p:?} after {}", out.iterations);
}

/// Harmonic bonds whose strength and a uniform field depend on the charge and multiplicity, so
/// that the five EDEEL evaluations differ, plus a tether to the origin.
struct ChargedPairs;

impl ChargedPairs {
    fn energy<T: Scalar>(sc: SpinCharge, r: &[T]) -> T {
        let n = r.len() / 3;
        let q = f64::from(sc.charge);
        let s = 1.0 + 0.3 * q + 0.1 * f64::from(sc.multiplicity - 1);
        let mut e = T::zero();
        for i in 0..n {
            for j in i + 1..n {
                let d2 = (0..3).fold(T::zero(), |acc, c| {
                    let d = r[3 * i + c] - r[3 * j + c];
                    acc + d * d
                });
                let x = d2.sqrt() - T::from_f64(1.5);
                e += x * x * T::from_f64(0.05 * s);
            }
            e += r[3 * i] * T::from_f64(0.02 * q);
            for c in 0..3 {
                e += r[3 * i + c] * r[3 * i + c] * T::from_f64(0.01);
            }
        }
        e
    }
}

impl Calculator for ChargedPairs {
    fn name(&self) -> &str {
        "charged_pairs"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let sc = sys.spin_charge;
        Ok(ad_output(sys, req, |r| Self::energy(sc, r)))
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }
}

fn four_atoms(flat: &Col<f64>) -> MolecularSystem {
    MolecularSystem::from_flat_bohr(vec![Element::C; 4], flat, SpinCharge::new(0, 1), None).unwrap()
}

fn start_flat() -> Col<f64> {
    let r = [0.0, 0.0, 0.0, 1.4, 0.2, 0.0, 0.7, 1.3, 0.1, 0.6, 0.5, 1.2];
    Col::from_fn(12, |i| r[i])
}

/// Donor D(+1, doublet) / D(0, singlet), acceptor A(0, singlet) / A(-1, doublet); the complex
/// without the moving electron is D(+1) A(0), a doublet of charge +1.
fn charges() -> EdeelStates {
    EdeelStates {
        complex_without: SpinCharge::new(1, 2),
        donor_without: SpinCharge::new(1, 2),
        donor_with: SpinCharge::new(0, 1),
        acceptor_without: SpinCharge::new(0, 1),
        acceptor_with: SpinCharge::new(-1, 2),
    }
}

fn ids(v: &[usize]) -> Vec<AtomId> {
    v.iter().map(|&i| AtomId::new(i)).collect()
}

fn surface_at(expr: &mut PesExpr, flat: &Col<f64>, need_hessian: bool) -> EvalOutput {
    let sys = four_atoms(flat);
    let req = if need_hessian {
        EvalRequest::ENERGY_GRAD_HESS
    } else {
        EvalRequest::ENERGY_GRAD
    };
    let tasks = expr.tasks(&sys, &req).unwrap();
    let results = CalculatorSet::single(&ChargedPairs)
        .evaluate(&tasks)
        .unwrap();
    let ctx = TermContext::new(0, &sys, None);
    expr.evaluate(&sys, &req, results, &ctx).unwrap().total
}

fn check_against_fd(expr: &mut PesExpr, x: &Col<f64>) {
    let h = 1e-5;
    let at = surface_at(expr, x, true);
    let hess = at.hessian_bohr.clone().unwrap();
    for i in 0..x.nrows() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += h;
        xm[i] -= h;
        let p = surface_at(expr, &xp, false);
        let m = surface_at(expr, &xm, false);
        let fd = (p.energy.value() - m.energy.value()) / (2.0 * h);
        let g = at.gradient_bohr[i];
        assert!(
            (fd - g).abs() < 1e-7 * g.abs().max(1.0),
            "gradient {i}: {fd} vs {g}"
        );
        for j in 0..x.nrows() {
            let fd = (p.gradient_bohr[j] - m.gradient_bohr[j]) / (2.0 * h);
            let hv = hess[(j, i)];
            assert!(
                (fd - hv).abs() < 1e-6 * hv.abs().max(1.0),
                "Hessian ({j}, {i}): {fd} vs {hv}"
            );
        }
    }
}

#[test]
fn edeel_shares_the_complex_and_matches_finite_differences() {
    let (donor, acceptor) = (ids(&[0, 1]), ids(&[2, 3]));
    let x = start_flat();
    let sys = four_atoms(&x);
    for (target, n_tasks) in [
        (EdeelTarget::Initial, 3),
        (EdeelTarget::Final, 3),
        (EdeelTarget::Penalty { sigma: 2.0 }, 5),
        (EdeelTarget::Crossing, 5),
    ] {
        let mut expr = edeel(&donor, &acceptor, &charges(), target).unwrap();
        let tasks = expr.tasks(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        assert_eq!(tasks.len(), n_tasks, "the complex is evaluated once");
        assert_eq!(tasks[0].system.spin_charge, SpinCharge::new(1, 2));
        assert_eq!(tasks[1].system.n_atoms(), 2);
        // Gradient projection returns a pseudo-gradient, not the gradient of its energy.
        if target != EdeelTarget::Crossing {
            check_against_fd(&mut expr, &x);
        }
    }

    // Energies follow the diabatic definitions.
    let e = |atoms: &[usize], sc: SpinCharge| {
        let sub = sys.subsystem(&ids(atoms), sc).unwrap();
        ChargedPairs
            .evaluate(&sub, &EvalRequest::ENERGY_GRAD)
            .unwrap()
            .energy
            .value()
    };
    let c = charges();
    let ec = e(&[0, 1, 2, 3], c.complex_without);
    let v11 = ec - e(&[0, 1], c.donor_without) + e(&[0, 1], c.donor_with);
    let v22 = ec - e(&[2, 3], c.acceptor_without) + e(&[2, 3], c.acceptor_with);
    let mut expr = edeel(&donor, &acceptor, &c, EdeelTarget::Penalty { sigma: 2.0 }).unwrap();
    let total = surface_at(&mut expr, &x, false).energy.value();
    let expected = 0.5 * (v11 + v22) + 2.0 * (v11 - v22).powi(2);
    assert!((total - expected).abs() < 1e-12, "{total} vs {expected}");
}

fn run_edeel(target: EdeelTarget) -> f64 {
    run_edeel_from(target, None, None).0
}

/// EDEEL of the two pairs with the energy shifts `shifts` of `V11` and `V22`.
fn shifted_edeel(target: EdeelTarget, shifts: Option<[f64; 2]>) -> PesExpr {
    let (donor, acceptor) = (ids(&[0, 1]), ids(&[2, 3]));
    let mut layout = EdeelLayout::separate(&donor, &acceptor).unwrap();
    if let Some(shifts) = shifts {
        layout = layout.shifted(shifts);
    }
    edeel_with(&layout, &charges(), target).unwrap()
}

/// [`run_edeel`] from the model Hessian `model`, with the energy shifts `shifts` of the two
/// states: the gap and the iterations.
fn run_edeel_from(
    target: EdeelTarget,
    model: Option<&str>,
    shifts: Option<[f64; 2]>,
) -> (f64, usize) {
    let expr = shifted_edeel(target, shifts);
    let mut opt = GeometryOptimizer::new(
        OptimizerSettings {
            criteria: ConvergenceCriteria::NORMAL,
            model_hessian: model.map(str::to_string),
            ..settings(&["rsirfo_bfgs"])
        },
        four_atoms(&start_flat()),
    )
    .unwrap();
    opt.set_surface(expr).unwrap();
    let mut gap = Gap(f64::NAN);
    let out = run_optimization(&mut opt, &ChargedPairs, &mut [&mut gap]).unwrap();
    assert_eq!(
        out.termination,
        Some(Termination::Converged),
        "{target:?} (model Hessian {model:?}, shifts {shifts:?}): {} iterations, V11 - V22 = {}",
        out.iterations,
        gap.0
    );
    (gap.0, out.iterations)
}

#[test]
fn a_model_hessian_goes_through_the_projection_of_a_crossing_search() {
    // On the seam model of one atom: the model Hessian (here the unit matrix) of both states
    // becomes P H P + 2 |g1 - g2| x x^T, as an exact Hessian does.
    let m = spin_crossing();
    let surface = two_states(
        model_function("opt_mesx", None, None).unwrap(),
        TaskSpec::whole("State2").spin_charge(SpinCharge::new(0, 3)),
    )
    .unwrap();
    let sys = atom([0.3, -0.2, 0.1]);
    let tasks = surface.tasks(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
    let results = CalculatorSet::single(&m).evaluate(&tasks).unwrap();
    let ctx = TermContext::new(0, &sys, None);
    let unit = Mat::<f64>::identity(3, 3);
    let through = surface
        .model_hessian_of_states(&sys, results.clone(), &ctx, &unit)
        .unwrap()
        .expect("a crossing search");
    let g: Vec<Col<f64>> = results.iter().map(|r| r.gradient_bohr.clone()).collect();
    let d = &g[0] - &g[1];
    let norm = d.norm_l2();
    let x = Col::from_fn(3, |i| d[i] / norm);
    for i in 0..3 {
        for j in 0..3 {
            let p = f64::from(u8::from(i == j)) - x[i] * x[j];
            let want = p + 2.0 * norm * x[i] * x[j];
            assert!((through[(i, j)] - want).abs() < 1e-12, "{i} {j}");
        }
    }
    // A surface of one state has no combiner to go through.
    let plain = PesExpr::Task(TaskSpec::whole("State1"));
    let tasks = plain.tasks(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
    let results = CalculatorSet::single(&m).evaluate(&tasks).unwrap();
    assert!(plain
        .model_hessian_of_states(&sys, results, &ctx, &unit)
        .unwrap()
        .is_none());
}

#[test]
fn edeel_crossing_converges_from_a_model_hessian() {
    // The crossing search from Lindh's model Hessian, which goes through the projection.
    let (gap, iterations) = run_edeel_from(EdeelTarget::Crossing, Some("lindh"), None);
    assert!(
        gap.abs() < 1.5e-4,
        "V11 - V22 = {gap} after {iterations} iterations"
    );
}

#[test]
fn a_model_hessian_has_curvature_along_rigid_body_motions_of_a_surface_that_has_it() {
    // The pairs are tethered to the origin and pulled by a field, so the surface is not
    // invariant under translations and rotations, where Lindh's model has no curvature. From
    // it the optimization converges as from the unit matrix (before, the gradient along those
    // motions was never stepped on and 500 iterations did not converge).
    let plain = run_edeel_from(EdeelTarget::Initial, None, None).1;
    let lindh = run_edeel_from(EdeelTarget::Initial, Some("lindh"), None).1;
    assert!(
        lindh <= 2 * plain + 10,
        "{lindh} iterations from Lindh's model, {plain} from the unit matrix"
    );
}

#[test]
fn the_gradient_without_terms_is_taken_where_the_shifted_states_are() {
    // A constant shift of one state has no gradient, so the surface without its terms (what
    // the Hessian is updated with) has the gradient of the whole surface: the gap of the
    // crossing search and the weight of the penalty are those of the shifted states.
    let x = start_flat();
    let sys = four_atoms(&x);
    let shifts = [0.03, -0.012];
    for target in [EdeelTarget::Crossing, EdeelTarget::Penalty { sigma: 2.0 }] {
        let mut expr = shifted_edeel(target, Some(shifts));
        let tasks = expr.tasks(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        let results = CalculatorSet::single(&ChargedPairs)
            .evaluate(&tasks)
            .unwrap();
        let ctx = TermContext::new(0, &sys, None);
        let at = expr
            .evaluate(&sys, &EvalRequest::ENERGY_GRAD, results, &ctx)
            .unwrap();
        let difference = (&at.total.gradient_bohr - &at.raw.gradient_bohr).norm_l2();
        assert!(difference < 1e-14, "{target:?}: {difference}");
        assert!(at.total.gradient_bohr.norm_l2() > 1e-3);
        // The energy without the terms stays that of the states without their shifts.
        let mut plain = shifted_edeel(target, None);
        let unshifted = surface_at(&mut plain, &x, false).energy.value();
        assert!(
            (at.raw.energy.value() - unshifted).abs() < 1e-13,
            "{target:?}"
        );
        assert!(
            (at.total.energy.value() - unshifted).abs() > 1e-4,
            "{target:?}"
        );
    }
}

#[test]
fn a_crossing_of_shifted_states_converges_from_updated_hessians() {
    // The crossing of states with energy shifts (the solvent shifts of an electron transfer),
    // from the unit matrix and from Lindh's model: the Hessian is updated with the gradients of
    // the shifted crossing.
    let shifts = Some([0.004, -0.003]);
    let plain = run_edeel_from(EdeelTarget::Crossing, None, None).1;
    for model in [None, Some("lindh")] {
        let (gap, iterations) = run_edeel_from(EdeelTarget::Crossing, model, shifts);
        assert!(gap.abs() < 1.5e-4, "{model:?}: V11 - V22 = {gap}");
        assert!(
            iterations <= 2 * plain + 10,
            "{model:?}: {iterations} iterations ({plain} without shifts)"
        );
    }
}

#[test]
fn edeel_crossing_optimization_reaches_the_seam() {
    // The published method (gradient projection) converges onto V11 = V22; MultiOptPy's penalty
    // variant stops at a finite gap.
    let gp = run_edeel(EdeelTarget::Crossing);
    // Converged when the gap force 2 (V11 - V22) is below the force threshold (3e-4).
    assert!(gp.abs() < 1.5e-4, "gradient projection: V11 - V22 = {gp}");
    let penalty = run_edeel(EdeelTarget::Penalty { sigma: 2.0 });
    assert!(
        penalty.abs() > 10.0 * gp.abs() && penalty.abs() < 2e-2,
        "penalty: V11 - V22 = {penalty} (gradient projection: {gp})"
    );
}

/// Records `V11 - V22` from the iteration reports.
struct Gap(f64);

impl mopt_rs::workflow::Observer for Gap {
    fn on_iteration(&mut self, r: &mopt_rs::opt::state_machine::IterationReport) -> MoptResult<()> {
        let get = |l: &str| {
            r.components
                .iter()
                .find(|(k, _)| k == l)
                .map(|(_, e)| e.value())
        };
        if let (Some(a), Some(b)) = (get("V11"), get("V22")) {
            self.0 = a - b;
        }
        Ok(())
    }
}

/// Harmonic restraint `kappa/2 |r - b|^2` on every atom, as a user-defined term.
struct Restraint {
    b: [f64; 3],
    kappa: f64,
}

impl PesTerm for Restraint {
    fn name(&self) -> &str {
        "restraint"
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        _ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let req = if need_hessian {
            EvalRequest::ENERGY_GRAD_HESS
        } else {
            EvalRequest::ENERGY_GRAD
        };
        Ok(ad_output(sys, &req, |r| {
            let mut e = mopt_rs::math::ad::HyperDual::zero();
            for (i, v) in r.iter().enumerate() {
                let d = *v - mopt_rs::math::ad::HyperDual::from_f64(self.b[i % 3]);
                e += d * d * mopt_rs::math::ad::HyperDual::from_f64(0.5 * self.kappa);
            }
            e
        }))
    }
}

#[test]
fn bias_terms_shift_the_minimum_and_are_reported_separately() {
    let m = spin_crossing();
    let (b, kappa) = ([0.0, 1.0, 0.5], 3.0);
    // Minimum of k/2 |r - a|^2 + kappa/2 |r - b|^2.
    let target: [f64; 3] = std::array::from_fn(|i| (m.k * m.a[i] + kappa * b[i]) / (m.k + kappa));
    for method in ["rsirfo_bfgs", "FIRELARS"] {
        let mut opt = GeometryOptimizer::new(settings(&[method]), atom([0.4, -0.2, 0.1])).unwrap();
        opt.add_term("restraint", Arc::new(Restraint { b, kappa }))
            .unwrap();
        let out = run_optimization(&mut opt, &m, &mut []).unwrap();
        assert_eq!(out.termination, Some(Termination::Converged), "{method}");
        let p = position(&out.system);
        assert!(dist(p, target) < 1e-4, "{method}: {p:?}");
        let raw = 0.5 * m.k * dist(p, m.a).powi(2);
        let bias = 0.5 * kappa * dist(p, b).powi(2);
        assert!((out.energy.value() - raw).abs() < 1e-8);
        assert!((out.biased_energy.value() - raw - bias).abs() < 1e-8);
    }
}

#[test]
fn a_bias_on_one_state_of_a_combined_surface_is_consistent() {
    let term: Arc<dyn PesTerm> = Arc::new(Restraint {
        b: [0.5, 0.0, 0.0],
        kappa: 0.2,
    });
    let state1 = PesExpr::Linear(vec![
        (1.0, PesExpr::Task(TaskSpec::whole("State1"))),
        (1.0, PesExpr::term("restraint", term)),
    ]);
    let state2 = PesExpr::Task(TaskSpec::whole("State2").spin_charge(SpinCharge::new(1, 2)));
    let mut expr = PesExpr::combine(
        Box::new(FunctionCombiner::new(QuadraticPenalty::seam(0.05))),
        vec!["S1".into(), "S2".into()],
        vec![state1, state2],
    )
    .unwrap();
    check_against_fd(&mut expr, &start_flat());

    // The term part is separated: raw + term Hessian reproduces the total Hessian when the
    // combination is linear in the term.
    let sys = four_atoms(&start_flat());
    let mut linear = PesExpr::Task(TaskSpec::whole("pes")).plus_term(
        "restraint",
        Arc::new(Restraint {
            b: [0.5, 0.0, 0.0],
            kappa: 0.2,
        }),
    );
    let req = EvalRequest::ENERGY_GRAD_HESS;
    let tasks = linear.tasks(&sys, &req).unwrap();
    let results = CalculatorSet::single(&ChargedPairs)
        .evaluate(&tasks)
        .unwrap();
    let ctx = TermContext::new(0, &sys, None);
    let s = linear.evaluate(&sys, &req, results, &ctx).unwrap();
    let sum: Mat<f64> = s.raw.hessian_bohr.unwrap() + s.term_hessian.unwrap();
    let diff = sum - s.total.hessian_bohr.unwrap();
    assert!(diff.norm_max() < 1e-12);
    assert_eq!(s.components.len(), 2);
}

#[test]
fn inconsistent_labels_are_rejected() {
    let bad = PesExpr::Linear(vec![
        (1.0, PesExpr::Task(TaskSpec::whole("x"))),
        (1.0, PesExpr::Task(TaskSpec::whole("x").excited_state(1))),
    ]);
    assert!(bad.validate().is_err());
    assert!(two_states(
        model_function("seam", None, None).unwrap(),
        TaskSpec::whole("State1")
    )
    .is_err());
    assert!(model_function("nope", None, None).is_err());
    assert!(edeel(
        &ids(&[0, 1]),
        &ids(&[1, 2]),
        &charges(),
        EdeelTarget::Initial
    )
    .is_err());
}

#[test]
fn vibrations_within_the_seam_of_a_conical_intersection() {
    let m = CrossingModel {
        k: 1.0,
        a: [0.0, 0.0, 0.3],
        b: [1.0, 0.0, 0.3],
        c: 0.02,
        lambda: 0.1,
        adiabatic: true,
    };
    let surface = two_states(
        model_function("opt_meci", None, None).unwrap(),
        TaskSpec::whole("State2").excited_state(1),
    )
    .unwrap();
    let mut opt =
        GeometryOptimizer::new(settings(&["rsirfo_bfgs"]), atom([0.3, 0.2, 0.15])).unwrap();
    opt.set_surface(surface).unwrap();
    let out = run_optimization(&mut opt, &m, &mut []).unwrap();
    assert_eq!(out.termination, Some(Termination::Converged));
    // The two states with their Hessians and the branching plane of the last iteration.
    let tasks = opt.hessian_tasks(&out.system).unwrap();
    let results = CalculatorSet::single(&m).evaluate(&tasks).unwrap();
    let crossing = opt.crossing(&out.system, results).unwrap().unwrap();
    assert_eq!(crossing.labels, vec!["State1", "State2"]);
    assert_eq!(crossing.plane.len(), 2);
    let states = &crossing.states;
    let hessians = states.hessians.as_ref().unwrap();
    let seam = mopt_rs::vib::seam_analysis(
        &out.system,
        [states.energies[0], states.energies[1]],
        [&states.gradients[0], &states.gradients[1]],
        [&hessians[0], &hessians[1]],
        &crossing.plane,
        mopt_rs::vib::SecondDirection::FromHessians,
        false,
    )
    .unwrap();
    // The seam is the line x = 0.52, z = 0: one vibration, along y, with the force constant
    // k of the wells, although the Hessians of the two states are large next to the cusp.
    let mass = out.system.masses_amu()[0];
    assert_eq!(seam.modes.curvatures.len(), 1);
    assert!(
        (seam.modes.curvatures[0] * mass - m.k).abs() < 1e-3,
        "{}",
        seam.modes.curvatures[0] * mass
    );
    let mode = &seam.modes.displacements[0];
    assert!(mode[1].abs() > 100.0 * mode[0].abs().max(mode[2].abs()));
    let (largest, next, overlap) = seam.coupling_curvature.unwrap();
    assert!(largest.abs() > 100.0 * next.abs(), "{largest} {next}");
    // The direction from the Hessians agrees with that of the branching plane updating.
    assert!(overlap > 0.9, "{overlap}");
    assert!(seam.gap.abs() < 1e-4 && seam.seam_gradient < 1e-3);
    // A single surface is not a crossing.
    let plain = GeometryOptimizer::new(settings(&["rsirfo_bfgs"]), atom([0.3, 0.2, 0.15])).unwrap();
    let tasks = plain.hessian_tasks(&out.system).unwrap();
    let results = CalculatorSet::single(&m).evaluate(&tasks).unwrap();
    assert!(plain.crossing(&out.system, results).unwrap().is_none());
}

#[test]
fn seam_vibrations_from_the_commands() {
    use mopt_rs::cli::{freqmain, optmain};
    use mopt_rs::pes::registry::CalculatorRegistry;

    let dir = std::env::temp_dir().join(format!("mopt_seam_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let start = dir.join("atom.xyz");
    std::fs::write(&start, "1\n\nH 0.05 -0.15 0.1\n").unwrap();
    let start = start.to_string_lossy().into_owned();
    let mut registry = CalculatorRegistry::native();
    registry.register("crossing", |_| Ok(Box::new(spin_crossing())));
    // The crossing search with the analysis at its end.
    let cfg = optmain::config_from_cli(&[
        &start,
        "-os",
        "crossing",
        "-opt",
        "rsirfo_bfgs",
        "-tcc",
        "-mf",
        "opt_mesx",
        "0",
        "3",
        "-freq",
    ])
    .unwrap();
    let optimized = optmain::run_with(&cfg, &dir, None, &registry).unwrap();
    let report = std::fs::read_to_string(optimized[0].directory.join("seam.txt")).unwrap();
    assert!(report.contains("a minimum of the seam"), "{report}");
    assert!(
        report.contains("one direction in the branching plane"),
        "{report}"
    );
    assert!(!optimized[0].directory.join("thermochemistry.txt").exists());
    // The same for the structure alone.
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let cfg =
        freqmain::config_from_cli(&[&xyz, "-os", "crossing", "-mf", "opt_mesx", "0", "3"]).unwrap();
    let out = freqmain::run_with(&cfg, &dir, None, &registry).unwrap();
    let seam = out[0].seam.as_ref().unwrap();
    assert!(out[0].analysis.is_none());
    // The seam is a plane, and both wells have the force constant k: two vibrations with it.
    let mass = 1.008;
    assert_eq!(seam.modes.curvatures.len(), 2);
    for curvature in &seam.modes.curvatures {
        assert!((curvature * mass - 1.0).abs() < 1e-3, "{curvature}");
    }
    assert!(seam.gap.abs() < 1e-5 && seam.seam_gradient < 1e-4);
    assert!(seam.coupling_curvature.is_none());
    // The other analyses of a single surface are refused at a crossing.
    let cfg = freqmain::config_from_cli(&[
        &xyz,
        "-os",
        "crossing",
        "-mf",
        "opt_mesx",
        "0",
        "3",
        "-anharmonic",
    ])
    .unwrap();
    assert!(freqmain::run_with(&cfg, &dir, None, &registry).is_err());
}

#[test]
fn a_search_that_keeps_its_symmetry_is_sent_on_by_the_check_of_the_branching_plane() {
    // Wells in the plane z = 0, coupled by 0.1 z: the states meet on the line x = 0.52,
    // z = 0, lowest at y = 0. From a start in the plane z = 0 every gradient lies in that
    // plane, so the search never sees the coupling direction z: it takes the direction of
    // the mean gradient, y, for the second direction of the branching plane and stops
    // where the gap closes, wherever that is along y.
    let m = CrossingModel {
        k: 1.0,
        a: [0.0, 0.0, 0.0],
        b: [1.0, 0.0, 0.0],
        c: 0.02,
        lambda: 0.1,
        adiabatic: true,
    };
    let target = [0.52, 0.0, 0.0];
    let run = |check: bool| {
        let surface = two_states(
            model_function("opt_meci", None, None).unwrap(),
            TaskSpec::whole("State2").excited_state(1),
        )
        .unwrap();
        let mut search = settings(&["rsirfo_bfgs"]);
        if !check {
            search.crossing_check = None;
        }
        let mut opt = GeometryOptimizer::new(search, atom([0.3, 0.4, 0.0])).unwrap();
        opt.set_surface(surface).unwrap();
        let mut notes = Vec::new();
        while let mopt_rs::opt::state_machine::Action::Evaluate(tasks) = opt.ask() {
            let results = CalculatorSet::single(&m).evaluate(&tasks).unwrap();
            let report = opt.tell(results).unwrap();
            notes.extend(report.notes.iter().cloned());
        }
        assert_eq!(opt.termination(), Some(Termination::Converged));
        (position(opt.current()), notes)
    };
    let (unchecked, _) = run(false);
    assert!(
        (unchecked[0] - 0.52).abs() < 1e-3 && unchecked[1].abs() > 0.3,
        "{unchecked:?}"
    );
    let (checked, notes) = run(true);
    assert!(dist(checked, target) < 1e-3, "{checked:?}");
    // The first check finds the direction z, which the search did not have, and sends the
    // search on; the last one confirms the point.
    let checks: Vec<&String> = notes
        .iter()
        .filter(|n| n.contains("second direction from gradients"))
        .collect();
    assert!(checks.len() >= 2, "{notes:?}");
    assert!(checks[0].contains("overlaps 0.0000") && checks[0].contains("goes on"));
    assert!(checks[checks.len() - 1].contains("converged"));
}
