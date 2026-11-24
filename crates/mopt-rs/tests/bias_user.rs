//! User-defined bias terms: expressions (`-ubias`) and registered options.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::domain::{positions, Angstrom, AtomId, Element, MolecularSystem, PeriodicCell};
use mopt_rs::domain::{SpinCharge, Vector3};
use mopt_rs::pes::bias::{AdBias, ExpressionBias};
use mopt_rs::pes::term_check::{check_term, energy_change_on_shift};
use mopt_rs::pes::{PesTerm, TermContext};

const POS: [[f64; 3]; 6] = [
    [0.2, 0.3, 0.4],
    [1.1, 0.5, 5.6],
    [4.6, 5.2, 0.6],
    [4.4, 0.1, 5.1],
    [2.4, 2.7, 3.0],
    [1.6, 1.1, 1.3],
];

fn system(pos: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::C; pos.len()],
        &positions::<Angstrom>(pos),
        SpinCharge::default(),
        cell,
    )
    .unwrap()
}

fn cell(axes: [bool; 3]) -> PeriodicCell {
    PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([5.0, 0.0, 0.0]),
        Vector3::<Angstrom>::new([1.2, 5.5, 0.0]),
        Vector3::<Angstrom>::new([0.4, -0.6, 6.0]),
        axes,
    )
    .unwrap()
}

fn energy(t: &dyn PesTerm, sys: &MolecularSystem) -> f64 {
    t.evaluate(sys, &TermContext::at_iteration(0, sys), false)
        .unwrap()
        .energy
        .value()
}

#[test]
fn expressions_pass_the_term_checks_in_all_cells() {
    let expr = "0.1 * (dist(1, 2) - 1.2*ang)^2 + 0.02 * (angle([1,6], 5, 3) - 100*deg)^2 \
                + 0.01 * cos(2 * dihedral(1, 6, 5, 4)) + 0.05 * exp(-dist([3-4], 6) / ang) \
                + 0.01 * tanh(oop(5, 1, 3, 6)) + 1e-3 * z(2)^2 / (1 + sqrt(dist(1, 3)))";
    let t = AdBias::new(ExpressionBias::parse(expr).unwrap());
    for axes in [
        None,
        Some([true, false, false]),
        Some([true, true, false]),
        Some([true, true, true]),
    ] {
        let sys = system(&POS, axes.map(cell));
        let c = check_term(&t, &sys, 1e-5).unwrap();
        assert!(c.passes(1e-6), "{axes:?}: {c:?}");
        let v = cell([true; 3]).lattice_vector(2).value();
        let de = energy_change_on_shift(&t, &sys, AtomId::new(0), v).unwrap();
        assert_eq!(
            de.abs() < 1e-12,
            axes.is_some_and(|a| a[2]),
            "{axes:?}: {de}"
        );
    }
}

#[test]
fn expressions_reproduce_built_in_biases() {
    let sys = system(&POS, None);
    let pairs = [
        (
            "-kp",
            vec!["0.3", "1.2", "1,2"],
            "0.15 * (dist(1, 2) - 1.2*ang)^2",
        ),
        (
            "-ka",
            vec!["0.2", "100", "1,6,5"],
            "0.1 * (angle(1, 6, 5) - 100*deg)^2",
        ),
        (
            "-kpv2",
            vec!["0.3", "1.0", "1,3", "2,4,5"],
            "0.15 * (dist([1,3], [2,4,5]) - ang)^2",
        ),
    ];
    for (opt, args, expr) in pairs {
        let args: Vec<String> = args.into_iter().map(String::from).collect();
        let built = parse_bias_option(opt, &args).unwrap()[0]
            .build(&sys)
            .unwrap();
        let user = AdBias::new(ExpressionBias::parse(expr).unwrap());
        let (a, b) = (energy(built.as_ref(), &sys), energy(&user, &sys));
        assert!((a - b).abs() < 1e-13, "{opt}: {a} vs {b}");
    }
}

#[test]
fn time_dependent_expressions_follow_the_clock() {
    let sys = system(&POS, None);
    let t = AdBias::new(ExpressionBias::parse("1e-3 * it * dist(1, 2) + t").unwrap());
    let r = sys.distance(AtomId::new(0), AtomId::new(1)).value();
    let mut ctx = TermContext::at_iteration(4, &sys);
    let e = t.evaluate(&sys, &ctx, false).unwrap().energy.value();
    assert!((e - (4e-3 * r + 4.0)).abs() < 1e-12);
    ctx.time_au = Some(100.0);
    let e = t.evaluate(&sys, &ctx, false).unwrap().energy.value();
    assert!((e - (4e-3 * r + 100.0)).abs() < 1e-12);
}

#[test]
fn user_expression_restrains_an_optimization_from_the_command_line() {
    let dir = std::env::temp_dir().join(format!("mopt_ubias_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let xyz = dir.join("ar3.xyz");
    std::fs::write(
        &xyz,
        "3\n\nAr 0.0 0.0 0.0\nAr 3.8 0.0 0.0\nAr 1.9 3.2 0.1\n",
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let out = run(
        &config_from_cli(&[
            &xyz,
            "-os",
            "lj",
            "-opt",
            "rsirfo_bfgs",
            "-ubias",
            "0.5 * (dist(1, 2) - 5.0*ang)^2",
        ])
        .unwrap(),
        &dir,
        None,
    )
    .unwrap();
    assert!(out[0].converged);
    let text = std::fs::read_to_string(&out[0].optimized).unwrap();
    let rows: Vec<Vec<f64>> = text
        .lines()
        .skip(2)
        .map(|l| {
            l.split_whitespace()
                .skip(1)
                .map(|x| x.parse().unwrap())
                .collect()
        })
        .collect();
    let d = (0..3)
        .map(|c| (rows[0][c] - rows[1][c]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((d - 5.0).abs() < 0.2, "restrained distance {d}");
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-ubias", "dist(1,"]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
