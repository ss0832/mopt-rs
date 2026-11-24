//! What the in-process terms declare about themselves (frame, history, copies), the analytic
//! stress of the automatically differentiated biases, the Hessians of the terms on request,
//! and the calculator with static biases (`pes::biased`).
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Instant;

use faer::Mat;
use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::domain::{
    positions, Angstrom, AtomId, Element, EvalRequest, MolecularSystem, PeriodicCell, Position,
    Quantity, SpinCharge, Vector3,
};
use mopt_rs::pes::bias::{
    AdBias, CollectiveVariable, KeepDistance, Metadynamics, MetadynamicsCv, Ramp, RmsdMetadynamics,
    VoidPoint,
};
use mopt_rs::pes::biased::BiasedCalculator;
use mopt_rs::pes::native::{LennardJones, TruncatedLennardJones};
use mopt_rs::pes::stress::{numerical_stress, numerical_term_stress};
use mopt_rs::pes::term::{fork_term, frame_invariant};
use mopt_rs::pes::{Calculator, CalculatorTerm, Frame, History, PesExpr, PesTerm, TermContext};
use mopt_rs::workflow::CalculatorSet;

/// Every built-in bias option with arguments for six atoms, and what its term must declare.
fn table() -> Vec<(&'static str, Vec<&'static str>, Frame, History)> {
    use Frame::{Absolute, Invariant};
    use History::{IterationRamp, Stateful, Static, TimeDependent};
    let limits = "0.05,0.1,0.15,0.2";
    vec![
        ("-ma", vec!["100", "1,2", "3,4"], Invariant, Static),
        (
            "-ma",
            vec!["100,200", "1,2", "3,4"],
            Invariant,
            IterationRamp,
        ),
        (
            "-rp",
            vec!["1.0", "1.0", "1,2", "3,4", "scale"],
            Invariant,
            Static,
        ),
        (
            "-rpv2",
            vec![
                "1.0", "1.0", "1.0", "1", "1", "12", "6", "1,2", "3,4", "scale",
            ],
            Invariant,
            Static,
        ),
        (
            "-rpg",
            vec!["1.0", "2.0", "0.5", "2.0", "1.0", "1,2", "3,4"],
            Invariant,
            Static,
        ),
        (
            "-cp",
            vec!["1.0", "1.0", "100", "1", "2,3,4", "5,6"],
            Invariant,
            Static,
        ),
        (
            "-fp",
            vec!["0.01,0.02,0.03", "2,2,3", "0,0,0", "1,2"],
            Absolute,
            Static,
        ),
        ("-kp", vec!["0.1", "1.2", "1,2"], Invariant, Static),
        (
            "-kp",
            vec!["0.1,0.2", "1.2", "1,2"],
            Invariant,
            IterationRamp,
        ),
        ("-kpv2", vec!["0.1", "1.5", "1,2", "3,4"], Invariant, Static),
        ("-akp", vec!["0.1", "0.3", "1.2", "1,2"], Invariant, Static),
        ("-ka", vec!["0.1", "100", "1,2,3"], Invariant, Static),
        (
            "-kav2",
            vec!["0.1", "100", "1", "2", "3,4"],
            Invariant,
            Static,
        ),
        ("-up", vec!["1.0", "1,2,3"], Invariant, Static),
        ("-kda", vec!["0.1", "30", "1,2,3,4"], Invariant, Static),
        (
            "-kda",
            vec!["0.1", "30,60", "1,2,3,4"],
            Invariant,
            IterationRamp,
        ),
        ("-kopa", vec!["0.1", "10", "1,2,3,4"], Invariant, Static),
        (
            "-kdav2",
            vec!["0.1", "30", "1", "2", "3", "4,5"],
            Invariant,
            Static,
        ),
        (
            "-kdac",
            vec!["0.1", "2", "30", "1", "2", "3", "4"],
            Invariant,
            Static,
        ),
        (
            "-kopav2",
            vec!["0.1", "10", "1", "2", "3", "4,5"],
            Invariant,
            Static,
        ),
        (
            "-vpp",
            vec!["0.1", "0.2", "0,0,0", "1,2", "2"],
            Absolute,
            Static,
        ),
        (
            "-brp",
            vec!["1.0", "1.0", "1.0", "0.8", "1", "2"],
            Invariant,
            Static,
        ),
        ("-wp", vec!["1.0", "1", "2", limits], Invariant, Static),
        ("-wwp", vec!["1.0", "z", limits, "1,2"], Absolute, Static),
        (
            "-vpwp",
            vec!["1.0", "0,0,0", limits, "1,2"],
            Absolute,
            Static,
        ),
        ("-awp", vec!["1.0", "1,2", limits, "3,4"], Invariant, Static),
        (
            "-smp",
            vec!["0.5", "1.0", "1.5", "2", "1,2"],
            Invariant,
            Stateful,
        ),
        (
            "-metad",
            vec!["bond", "1.0", "0.1", "1,2"],
            Invariant,
            Stateful,
        ),
        ("-lmefp", vec!["100", "1,2", "3,4"], Invariant, Static),
        ("-lmefpv2", vec!["100", "1,2"], Invariant, Static),
        (
            "-aerp",
            vec!["1.0", "1,1,1,1,1,1", "1.0", "1,2", "0"],
            Invariant,
            Stateful,
        ),
        (
            "-aerpv2",
            vec!["1.0", "1,1,1,1,1,1", "1.0", "1,2", "0"],
            Invariant,
            Stateful,
        ),
        (
            "-nrp",
            vec!["0.5", "1.0", "1.5", "0.5", "1.0", "0.5"],
            Absolute,
            TimeDependent,
        ),
        (
            "-ubias",
            vec!["0.1 * (dist(1,2) - 2)^2 + angle(1,2,3)"],
            Invariant,
            Static,
        ),
        ("-ubias", vec!["0.1 * x(1)^2"], Absolute, Static),
        (
            "-ubias",
            vec!["0.001 * it * dist(1,2)"],
            Invariant,
            IterationRamp,
        ),
        (
            "-ubias",
            vec!["0.001 * t * dist(1,2)"],
            Invariant,
            TimeDependent,
        ),
    ]
}

/// Six atoms without symmetry, in a cell when `cell` is given.
fn six(cell: Option<PeriodicCell>) -> MolecularSystem {
    MolecularSystem::new(
        vec![
            Element::C,
            Element::H,
            Element::O,
            Element::N,
            Element::H,
            Element::C,
        ],
        &positions::<Angstrom>(&[
            [0.31, 0.22, 0.41],
            [1.42, 0.35, 0.18],
            [0.05, 1.47, 0.63],
            [0.83, -0.21, 1.52],
            [-0.61, -0.32, -0.35],
            [2.12, 1.41, 0.97],
        ]),
        SpinCharge::default(),
        cell,
    )
    .unwrap()
}

fn terms_of(flag: &str, args: &[&str], sys: &MolecularSystem) -> Vec<Arc<dyn PesTerm>> {
    let tokens: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    parse_bias_option(flag, &tokens)
        .unwrap()
        .iter()
        .map(|spec| spec.build(sys).unwrap())
        .collect()
}

/// `sys` rotated about an axis through the origin and translated.
fn rigidly_moved(sys: &MolecularSystem) -> MolecularSystem {
    let (s, c) = 0.7_f64.sin_cos();
    let mut out = sys.clone();
    for a in sys.atoms() {
        let p = sys.position(a).value();
        out.set_position(
            a,
            Position::new([
                c * p[0] - s * p[1] + 1.3,
                s * p[0] + c * p[1] - 0.7,
                p[2] + 2.1,
            ]),
        );
    }
    out
}

#[test]
fn every_built_in_bias_declares_its_frame_and_history() {
    let sys = six(None);
    for (flag, args, frame, history) in table() {
        let terms = terms_of(flag, &args, &sys);
        assert!(!terms.is_empty(), "{flag}");
        for t in terms {
            assert_eq!(t.frame(), frame, "{flag} {args:?}");
            assert_eq!(t.history(), history, "{flag} {args:?}");
        }
    }
}

#[test]
fn invariant_terms_do_not_see_a_rigid_motion_and_absolute_ones_do() {
    let sys = six(None);
    let moved = rigidly_moved(&sys);
    let ctx = TermContext::at_iteration(5, &sys);
    for (flag, args, frame, history) in table() {
        // The inner minimizations of the stateful auxiliary biases start from structures in
        // the lab frame and may end in another minimum: their invariance is that of the
        // relaxed energy, not checked here.
        if matches!(flag, "-smp" | "-aerp" | "-aerpv2") {
            continue;
        }
        for t in terms_of(flag, &args, &sys) {
            if history == History::Stateful {
                // Something to be invariant about: one deposited Gaussian.
                t.commit(&sys, &ctx);
            }
            let e0 = t.evaluate(&sys, &ctx, false).unwrap().energy.value();
            let e1 = t.evaluate(&moved, &ctx, false).unwrap().energy.value();
            let change = (e1 - e0).abs();
            match frame {
                Frame::Invariant => assert!(
                    change < 1e-11 * (1.0 + e0.abs()),
                    "{flag} {args:?}: {e0} -> {e1}"
                ),
                _ => assert!(change > 1e-8, "{flag} {args:?}: {e0} -> {e1}"),
            }
        }
    }
}

#[test]
fn a_fork_has_its_own_state() {
    let sys = six(None);
    let ctx = TermContext::at_iteration(0, &sys);
    // Metadynamics: a deposit on the copy does not reach the original.
    let metad: Arc<dyn PesTerm> = Arc::new(AdBias::new(Metadynamics::new(vec![MetadynamicsCv {
        cv: CollectiveVariable::Bond([AtomId::new(0), AtomId::new(1)]),
        height: 0.01,
        width: 0.2,
    }])));
    let copy = fork_term(&metad).unwrap();
    assert!(!Arc::ptr_eq(&copy, &metad));
    copy.commit(&sys, &ctx);
    let e = |t: &Arc<dyn PesTerm>| t.evaluate(&sys, &ctx, false).unwrap().energy.value();
    assert!((e(&copy) - 0.01).abs() < 1e-12);
    assert_eq!(e(&metad), 0.0);
    // The copy keeps what was deposited before it was made, and goes its own way after.
    metad.commit(&sys, &ctx);
    let second = fork_term(&metad).unwrap();
    metad.commit(&sys, &ctx);
    assert!((e(&second) - 0.01).abs() < 1e-12 && (e(&metad) - 0.02).abs() < 1e-12);
    // RMSD metadynamics: references added to the copy stay there.
    let rmsd = RmsdMetadynamics::new(0.01, 1.0).unwrap();
    rmsd.add_reference(&sys).unwrap();
    let rmsd: Arc<dyn PesTerm> = Arc::new(rmsd);
    let copy = fork_term(&rmsd).unwrap();
    assert!((e(&copy) - e(&rmsd)).abs() < 1e-15);
    // Stateless terms are shared, stateful ones that cannot be copied are refused.
    let spring = terms_of("-kp", &["0.1", "1.2", "1,2"], &sys).remove(0);
    assert!(Arc::ptr_eq(&fork_term(&spring).unwrap(), &spring));
    struct Opaque;
    impl PesTerm for Opaque {
        fn name(&self) -> &str {
            "opaque"
        }
        fn evaluate(
            &self,
            sys: &MolecularSystem,
            _ctx: &TermContext<'_>,
            _need_hessian: bool,
        ) -> mopt_rs::error::MoptResult<mopt_rs::domain::EvalOutput> {
            Ok(mopt_rs::domain::EvalOutput::energy_gradient(
                Quantity::new(0.0),
                faer::Col::zeros(sys.dof()),
            ))
        }
    }
    let opaque: Arc<dyn PesTerm> = Arc::new(Opaque);
    assert_eq!(opaque.frame(), Frame::Absolute);
    assert_eq!(opaque.history(), History::Stateful);
    assert!(fork_term(&opaque)
        .err()
        .is_some_and(|e| e.to_string().contains("opaque")));
    // The auxiliary biases copy their last solution.
    let spacer = terms_of("-smp", &["0.5", "1.0", "1.5", "2", "1,2"], &sys).remove(0);
    let copy = fork_term(&spacer).unwrap();
    assert!(!Arc::ptr_eq(&copy, &spacer));
}

#[test]
fn the_rigid_body_invariance_of_a_surface_is_decided_in_one_place() {
    let sys = six(None);
    let spring = terms_of("-kp", &["0.1", "1.2", "1,2"], &sys).remove(0);
    let wall = terms_of("-vpp", &["0.1", "0.2", "0,0,0", "1,2", "2"], &sys).remove(0);
    assert!(frame_invariant(true, [spring.as_ref()]));
    assert!(!frame_invariant(false, [spring.as_ref()]));
    assert!(!frame_invariant(true, [spring.as_ref(), wall.as_ref()]));
    let expr = PesExpr::single().plus_term("kp", Arc::clone(&spring));
    assert!(expr.frame_invariant(true).unwrap());
    let expr = expr.plus_term("vpp", wall);
    assert!(!expr.frame_invariant(true).unwrap());
    let lj: Arc<dyn Calculator> = Arc::new(LennardJones);
    let term = CalculatorTerm::new(lj);
    assert_eq!(
        (term.frame(), term.history()),
        (Frame::Invariant, History::Static)
    );
}

fn skewed(axes: [bool; 3]) -> PeriodicCell {
    PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([6.1, 0.0, 0.0]),
        Vector3::<Angstrom>::new([0.9, 5.8, 0.0]),
        Vector3::<Angstrom>::new([-0.6, 0.7, 6.4]),
        axes,
    )
    .unwrap()
}

#[test]
fn the_stress_of_the_geometric_biases_is_the_strain_derivative_of_their_energy() {
    for axes in [[true, true, true], [true, true, false]] {
        let sys = six(Some(skewed(axes)));
        let ctx = TermContext::at_iteration(0, &sys);
        let mut checked = 0;
        for (flag, args, _, history) in table() {
            if matches!(flag, "-smp" | "-aerp" | "-aerpv2") {
                continue;
            }
            for t in terms_of(flag, &args, &sys) {
                if history == History::Stateful {
                    t.commit(&six(None), &ctx);
                }
                let out = t.evaluate(&sys, &ctx, false).unwrap();
                let analytic = out.stress.unwrap();
                let numeric = numerical_term_stress(t.as_ref(), &sys, &ctx, 1e-5).unwrap();
                let scale = numeric
                    .iter()
                    .flatten()
                    .fold(1e-6_f64, |m, v| m.max(v.abs()));
                for a in 0..3 {
                    for b in 0..3 {
                        assert!(
                            (analytic[a][b] - numeric[a][b]).abs() < 1e-6 * scale,
                            "{flag} {args:?} {axes:?} ({a}, {b}): {} vs {}",
                            analytic[a][b],
                            numeric[a][b]
                        );
                        assert!(
                            (analytic[a][b] - analytic[b][a]).abs() < 1e-14 * scale,
                            "{flag}: not symmetric"
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 30, "{checked}");
    }
}

/// An argon cluster of `n` atoms on a distorted grid.
fn argon(n: usize) -> MolecularSystem {
    let rows: Vec<[f64; 3]> = (0..n)
        .map(|k| {
            let (i, j, l) = (k % 3, (k / 3) % 3, k / 9);
            let w = 0.17 * (k as f64).sin();
            [
                3.7 * i as f64 + w,
                3.8 * j as f64 - 0.5 * w,
                3.6 * l as f64 + 0.3 * w,
            ]
        })
        .collect();
    MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); n],
        &positions::<Angstrom>(&rows),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

/// Several biases on the argon cluster.
fn cluster_biases(sys: &MolecularSystem) -> Vec<(String, Arc<dyn PesTerm>)> {
    let options: [(&str, &[&str]); 6] = [
        ("-kp", &["0.01", "3.5", "1,2", "0.01", "3.9", "4,8"]),
        ("-ka", &["0.01", "100", "1,2,5"]),
        ("-kda", &["0.01", "30", "2,1,4,5"]),
        ("-rp", &["1.0", "1.0", "1-9", "10-18", "scale"]),
        ("-ubias", &["0.001 * (dist([1-6],[7-12]) - 5)^2"]),
        ("-wp", &["1.0", "1-9", "10-18", "0.5,1.0,6.0,8.0"]),
    ];
    let mut out = Vec::new();
    for (flag, args) in options {
        for (k, t) in terms_of(flag, args, sys).into_iter().enumerate() {
            out.push((format!("{flag}_{k}"), t));
        }
    }
    out
}

#[test]
fn bias_hessians_follow_the_request() {
    let sys = argon(18);
    let terms = cluster_biases(&sys);
    let mut expr = PesExpr::single();
    for (label, t) in &terms {
        expr = expr.plus_term(label, Arc::clone(t));
    }
    let calc = LennardJones;
    let ctx = TermContext::at_iteration(0, &sys);
    let req = EvalRequest::ENERGY_GRAD;
    let tasks = expr.tasks(&sys, &req).unwrap();
    let results = || CalculatorSet::single(&calc).evaluate(&tasks).unwrap();
    let always = expr.evaluate(&sys, &req, results(), &ctx).unwrap();
    let asked = expr
        .evaluate_as_requested(&sys, &req, results(), &ctx)
        .unwrap();
    assert!(always.term_hessian.is_some() && asked.term_hessian.is_none());
    // The raw surface is the same; the biased gradient agrees to rounding (dual numbers
    // instead of hyper-dual numbers in the automatic derivatives of the terms).
    assert_eq!(always.raw.energy.value(), asked.raw.energy.value());
    assert_eq!(always.raw.gradient_bohr, asked.raw.gradient_bohr);
    assert!((always.total.energy.value() - asked.total.energy.value()).abs() < 1e-14);
    for i in 0..sys.dof() {
        let (a, b) = (always.total.gradient_bohr[i], asked.total.gradient_bohr[i]);
        assert!((a - b).abs() < 1e-15, "{i}: {a} {b}");
    }
    assert_eq!(always.components.len(), asked.components.len());
    // With a Hessian requested both are the same.
    let full = EvalRequest::ENERGY_GRAD_HESS;
    let tasks_h = expr.tasks(&sys, &full).unwrap();
    let results_h = || CalculatorSet::single(&calc).evaluate(&tasks_h).unwrap();
    let a = expr.evaluate(&sys, &full, results_h(), &ctx).unwrap();
    let b = expr
        .evaluate_as_requested(&sys, &full, results_h(), &ctx)
        .unwrap();
    assert_eq!(a.term_hessian, b.term_hessian);
    // Time per gradient-only evaluation of the surface (the calculator results are given).
    let reps = 20;
    let time = |as_requested: bool, expr: &mut PesExpr| {
        let start = Instant::now();
        for _ in 0..reps {
            let r = results();
            if as_requested {
                expr.evaluate_as_requested(&sys, &req, r, &ctx).unwrap();
            } else {
                expr.evaluate(&sys, &req, r, &ctx).unwrap();
            }
        }
        start.elapsed().as_secs_f64() / reps as f64
    };
    let before = time(false, &mut expr);
    let after = time(true, &mut expr);
    println!(
        "gradient-only surface evaluation, 18 argon atoms, {} bias terms: with term Hessians \
         {:.3} ms, as requested {:.3} ms",
        terms.len(),
        1e3 * before,
        1e3 * after
    );
}

fn compare(expr_out: (f64, &faer::Col<f64>, &Mat<f64>), calc: &mopt_rs::domain::EvalOutput) {
    let (e, g, h) = expr_out;
    assert!(
        (e - calc.energy.value()).abs() < 1e-12,
        "{e} {}",
        calc.energy.value()
    );
    let hc = calc.hessian_bohr.as_ref().unwrap();
    for i in 0..g.nrows() {
        assert!((g[i] - calc.gradient_bohr[i]).abs() < 1e-12);
        for j in 0..g.nrows() {
            assert!((h[(i, j)] - hc[(i, j)]).abs() < 1e-10, "({i}, {j})");
        }
    }
}

/// The calculator with terms against the surface expression with the same terms.
fn check_against_expr(
    inner: Arc<dyn Calculator>,
    sys: &MolecularSystem,
    terms: Vec<(String, Arc<dyn PesTerm>)>,
) {
    let biased = BiasedCalculator::new(Arc::clone(&inner), terms.clone()).unwrap();
    let mut expr = PesExpr::single();
    for (label, t) in &terms {
        expr = expr.plus_term(label, Arc::clone(t));
    }
    let req = EvalRequest::ENERGY_GRAD_HESS;
    let tasks = expr.tasks(sys, &req).unwrap();
    let results = CalculatorSet::single(inner.as_ref())
        .evaluate(&tasks)
        .unwrap();
    let ctx = TermContext::at_iteration(0, sys);
    let surf = expr.evaluate(sys, &req, results, &ctx).unwrap();
    let hessian = surf.raw.hessian_bohr.clone().unwrap() + surf.term_hessian.clone().unwrap();
    let parts = biased.evaluate_parts(sys, &req).unwrap();
    compare(
        (
            surf.total.energy.value(),
            &surf.total.gradient_bohr,
            &hessian,
        ),
        &parts.total,
    );
    assert!((parts.energy().value() - surf.raw.energy.value()).abs() < 1e-12);
    assert!((parts.biased_energy().value() - surf.total.energy.value()).abs() < 1e-12);
    assert_eq!(parts.components.len(), terms.len());
    if sys.active_cell().is_some() {
        let analytic = biased
            .evaluate(sys, &EvalRequest::ENERGY_GRAD_STRESS)
            .unwrap()
            .stress
            .unwrap();
        let numeric = numerical_stress(&biased, sys, 1e-5).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (analytic[a][b] - numeric[a][b]).abs() < 1e-9,
                    "({a}, {b}): {} vs {}",
                    analytic[a][b],
                    numeric[a][b]
                );
            }
        }
    }
}

#[test]
fn a_calculator_with_biases_equals_the_surface_expression() {
    let cluster = argon(18);
    check_against_expr(Arc::new(LennardJones), &cluster, cluster_biases(&cluster));
    let lj: Arc<dyn Calculator> = Arc::new(LennardJones);
    let biased = BiasedCalculator::new(Arc::clone(&lj), cluster_biases(&cluster)).unwrap();
    assert!(biased.is_rigid_body_invariant() && biased.analytic_hessian());

    // A periodic cell with a truncated LJ potential, a spring, AFIR, and a wall at a point.
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([8.1, 0.0, 0.0]),
        Vector3::<Angstrom>::new([0.7, 7.9, 0.0]),
        Vector3::<Angstrom>::new([-0.4, 0.6, 8.3]),
        [true; 3],
    )
    .unwrap();
    let periodic = MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); 4],
        &positions::<Angstrom>(&[
            [0.4, 0.3, 0.2],
            [3.9, 0.5, 0.6],
            [0.8, 3.6, 0.4],
            [4.4, 4.1, 3.9],
        ]),
        SpinCharge::default(),
        Some(cell),
    )
    .unwrap();
    let inner: Arc<dyn Calculator> =
        Arc::new(TruncatedLennardJones::new(Quantity::new(7.0 * 1.889_726_124_6), None).unwrap());
    let mut terms = Vec::new();
    for (flag, args) in [
        ("-kp", vec!["0.01", "3.0", "1,2"]),
        ("-ma", vec!["80", "1,2", "3,4"]),
        ("-vpp", vec!["0.01", "0.5", "1,1,1", "3", "2"]),
        ("-ubias", vec!["0.001 * angle(1,2,4)^2"]),
    ] {
        for (k, t) in terms_of(flag, &args, &periodic).into_iter().enumerate() {
            terms.push((format!("{flag}_{k}"), t));
        }
    }
    let biased = BiasedCalculator::new(Arc::clone(&inner), terms.clone()).unwrap();
    assert!(
        !biased.is_rigid_body_invariant(),
        "the wall is fixed in space"
    );
    assert!(biased.analytic_term_stress());
    check_against_expr(inner, &periodic, terms);
}

#[test]
fn a_calculator_refuses_biases_that_are_not_static() {
    let lj: Arc<dyn Calculator> = Arc::new(LennardJones);
    let metad: Arc<dyn PesTerm> = Arc::new(AdBias::new(Metadynamics::new(Vec::new())));
    let err = BiasedCalculator::new(Arc::clone(&lj), vec![("metad_0".to_string(), metad)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("metad_0") && err.contains("Stateful"), "{err}");
    let ramp: Arc<dyn PesTerm> = Arc::new(AdBias::new(KeepDistance {
        a: vec![AtomId::new(0)],
        b: vec![AtomId::new(1)],
        k: Ramp::parse("0.1,0.2").unwrap(),
        r0: Ramp::constant(3.0),
    }));
    let err = BiasedCalculator::new(Arc::clone(&lj), vec![("keep_pot_0".to_string(), ramp)])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("keep_pot_0") && err.contains("IterationRamp"),
        "{err}"
    );
    let wall: Arc<dyn PesTerm> = Arc::new(AdBias::new(VoidPoint {
        atom: AtomId::new(0),
        point: [0.0; 3],
        k: 0.1,
        r0: 1.0,
        order: 2.0,
    }));
    assert!(BiasedCalculator::new(lj, vec![("vpp_0".to_string(), wall)]).is_ok());
}
