//! Tests of the molecular dynamics state machine on model surfaces.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::{Col, Mat};

use super::*;
use crate::pes::native::muller_brown::MB_SADDLES;
use crate::pes::objective::{MullerBrown2D, Objective};

/// Run `md` to its end on `f`, returning the reports.
fn drive(
    md: &mut MolecularDynamics,
    f: &dyn Fn(&Col<f64>, bool) -> (f64, Col<f64>, Option<Mat<f64>>),
) -> Vec<MdStepReport> {
    let mut out = Vec::new();
    while let MdAction::Evaluate(r) = md.ask() {
        let (e, g, h) = f(&r.x, r.need_hessian);
        let mut value = MdValue::plain(e, g);
        value.hessian = h;
        out.push(md.tell(value).unwrap());
    }
    out
}

fn quadratic(a: &Mat<f64>) -> impl Fn(&Col<f64>, bool) -> (f64, Col<f64>, Option<Mat<f64>>) + '_ {
    move |x: &Col<f64>, need: bool| {
        let g = a * x;
        let e = 0.5 * x.iter().zip(g.iter()).map(|(p, q)| p * q).sum::<f64>();
        (e, g, need.then(|| a.clone()))
    }
}

fn mb(x: &Col<f64>, need: bool) -> (f64, Col<f64>, Option<Mat<f64>>) {
    let v = MullerBrown2D::default().evaluate(x, need).unwrap();
    (v.value, v.gradient, v.hessian)
}

#[test]
fn velocity_verlet_conserves_the_energy_of_an_oscillator_to_second_order() {
    let a = Mat::from_fn(1, 1, |_, _| 1.0);
    let drift = |dt: f64| {
        let settings = MdSettings {
            time_step: dt,
            steps: (20.0 * std::f64::consts::TAU / dt) as usize,
            velocities: InitialVelocities::Given(Col::from_fn(1, |_| 1.0)),
            ..MdSettings::default()
        };
        let mut md = MolecularDynamics::new(
            settings,
            MdSpace::variables(1).unwrap(),
            Col::zeros(1),
            None,
        )
        .unwrap();
        let reports = drive(&mut md, &quadratic(&a));
        let e0 = reports[0].conserved;
        reports
            .iter()
            .map(|r| (r.conserved - e0).abs())
            .fold(0.0_f64, f64::max)
    };
    let (coarse, fine) = (drift(0.1), drift(0.05));
    // The energy of velocity Verlet oscillates with an amplitude of order dt^2.
    assert!(coarse < 2e-3, "{coarse}");
    assert!((coarse / fine - 4.0).abs() < 0.5, "{coarse} {fine}");
}

#[test]
fn the_same_seed_and_evaluations_give_the_same_run_bit_for_bit() {
    let a = Mat::from_fn(4, 4, |i, j| if i == j { 1.0 + i as f64 } else { 0.1 });
    let make = || {
        let settings = MdSettings {
            time_step: 0.05,
            steps: 200,
            temperature: 300.0,
            thermostat: "langevin".to_string(),
            integrator: "baoab".to_string(),
            thermostat_settings: ThermostatSettings {
                tau: 5.0,
                ..ThermostatSettings::default()
            },
            seed: 77,
            ..MdSettings::default()
        };
        MolecularDynamics::new(
            settings,
            MdSpace::variables(4).unwrap(),
            Col::zeros(4),
            None,
        )
        .unwrap()
    };
    let (mut one, mut two) = (make(), make());
    let first = drive(&mut one, &quadratic(&a));
    let second = drive(&mut two, &quadratic(&a));
    assert_eq!(first.len(), 201);
    for (p, q) in first.iter().zip(&second) {
        assert_eq!(p.x, q.x);
        assert_eq!(p.v, q.v);
        assert_eq!(p.conserved.to_bits(), q.conserved.to_bits());
    }
}

fn saddle() -> Col<f64> {
    let mut x = Col::from_fn(2, |i| MB_SADDLES[0][i]);
    for _ in 0..20 {
        let (_, g, h) = mb(&x, true);
        let h = h.unwrap();
        let det = h[(0, 0)] * h[(1, 1)] - h[(0, 1)] * h[(1, 0)];
        let dx = (h[(1, 1)] * g[0] - h[(0, 1)] * g[1]) / det;
        let dy = (h[(0, 0)] * g[1] - h[(1, 0)] * g[0]) / det;
        x[0] -= dx;
        x[1] -= dy;
    }
    x
}

#[test]
fn with_the_saddle_direction_frozen_the_dynamics_stays_at_the_muller_brown_saddle() {
    let start = saddle();
    let run = |freeze: bool| {
        let settings = MdSettings {
            time_step: 0.01,
            steps: 3000,
            temperature: 300.0,
            seed: 5,
            freeze: freeze.then(|| FreezeSettings {
                selection: ModeSelection::Negative,
                update_interval: None,
            }),
            ..MdSettings::default()
        };
        let mut md = MolecularDynamics::new(
            settings,
            MdSpace::variables(2).unwrap(),
            start.clone(),
            None,
        )
        .unwrap();
        let reports = drive(&mut md, &mb);
        (md.dof(), reports)
    };
    let (dof, frozen) = run(true);
    assert_eq!(dof, 1.0);
    let e0 = frozen[0].conserved;
    let mut farthest = 0.0_f64;
    for r in &frozen {
        assert!(
            (r.conserved - e0).abs() < 1e-6 * e0.abs(),
            "energy {} {}",
            r.conserved,
            e0
        );
        assert!(r.constraint_residual.unwrap() < 1e-10);
        farthest = farthest.max(((r.x[0] - start[0]).powi(2) + (r.x[1] - start[1]).powi(2)).sqrt());
    }
    assert!(farthest < 0.15, "{farthest}");
    let event = frozen[0].freeze.as_ref().unwrap();
    assert!(event.eigenvalues[0] < 0.0);
    // Without freezing the same start falls off the saddle.
    let (_, free) = run(false);
    let last = free.last().unwrap();
    let away = ((last.x[0] - start[0]).powi(2) + (last.x[1] - start[1]).powi(2)).sqrt();
    assert!(away > 0.2, "{away}");
}

#[test]
fn canonical_thermostats_give_the_mean_and_variance_of_the_temperature() {
    let n = 24;
    let a = Mat::from_fn(n, n, |i, j| {
        if i == j {
            1.0 + 0.1 * i as f64
        } else if i.abs_diff(j) == 1 {
            0.3
        } else {
            0.0
        }
    });
    for thermostat in ["bussi", "langevin", "nose_hoover_chain"] {
        let settings = MdSettings {
            time_step: 0.1,
            steps: 40_000,
            temperature: 500.0,
            thermostat: thermostat.to_string(),
            thermostat_settings: ThermostatSettings {
                tau: 5.0,
                ..ThermostatSettings::default()
            },
            seed: 9,
            ..MdSettings::default()
        };
        let mut md = MolecularDynamics::new(
            settings,
            MdSpace::variables(n).unwrap(),
            Col::zeros(n),
            None,
        )
        .unwrap();
        let reports = drive(&mut md, &quadratic(&a));
        let t: Vec<f64> = reports[5000..].iter().map(|r| r.temperature).collect();
        let mean = t.iter().sum::<f64>() / t.len() as f64;
        let var = t.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / t.len() as f64;
        let canonical = 2.0 * 500.0 * 500.0 / n as f64;
        assert!(
            (mean / 500.0 - 1.0).abs() < 0.05,
            "{thermostat}: mean {mean}"
        );
        assert!(
            (var / canonical - 1.0).abs() < 0.25,
            "{thermostat}: variance {var} vs {canonical}"
        );
    }
}
