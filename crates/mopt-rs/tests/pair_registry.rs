//! Every method of the registry of two-point methods runs by its name on the Muller-Brown
//! surface (the saddle between the minima B and C) through one entry point, and returns its
//! estimates ranked by energy with the counts of the evaluations.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use faer::Col;

use mopt_rs::core_infra::log::Logger;
use mopt_rs::path::gsm::DrivingCoordinate;
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::objective::MullerBrown2D;
use mopt_rs::workflow::pair_search::{
    Ends, PairInput, PairMethodRegistry, PairSettings, SearchRun, VariableSurface,
};

fn col(p: [f64; 2]) -> Col<f64> {
    Col::from_fn(2, |k| p[k])
}

/// Settings for the scale of the Muller-Brown surface (curvatures of some hundreds).
fn settings(name: &str) -> Vec<(&'static str, &'static str)> {
    match name {
        "gsm" | "se_gsm" => vec![
            ("max_step", "0.02"),
            ("first_scale", "0.001"),
            ("add_tolerance", "1.0"),
            ("climb_tolerance", "0.5"),
            ("criteria", "loose"),
            ("drive_step", "0.08"),
            ("drop", "0.5"),
        ],
        "fsm" => vec![("first_scale", "0.001"), ("max_component", "0.03")],
        "dsafir" => vec![("force", "100"), ("step", "0.01"), ("separation", "0.02")],
        "2pshs" => vec![("step", "0.01")],
        "gnt" => vec![("step", "0.02")],
        "dimer" => vec![("step", "0.02")],
        "spm" => vec![
            ("offset", "0.02"),
            ("length", "0.005"),
            ("drift", "5e-4"),
            ("spring", "2e-2"),
            ("climb", "5e-3"),
            ("tolerance", "1.0"),
            ("drift_steps", "50"),
            ("step", "0.01"),
            ("cycles", "3000"),
        ],
        "addf" => vec![("initial", "0.01"), ("step", "0.02"), ("paths", "0")],
        "lmm" => vec![("ray_step", "0.05"), ("criteria", "loose")],
        "neb" => vec![("images", "12"), ("iterations", "300")],
        "dmf" => vec![],
        "bitss" => vec![],
        _ => vec![],
    }
}

#[test]
fn every_registered_method_runs_by_its_name() {
    let registry = PairMethodRegistry::default();
    let names = registry.names();
    for expected in [
        "ieip", "gsm", "se_gsm", "fsm", "dsafir", "lmm", "2pshs", "gnt", "dimer", "spm", "addf",
        "neb", "dmf", "bitss",
    ] {
        assert!(names.contains(&expected), "{expected} is not registered");
    }
    let objective = MullerBrown2D::default();
    let surface = VariableSurface(&objective);
    let dir = std::env::temp_dir().join(format!("mopt_pair_registry_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let log = Logger::silent();
    let run = SearchRun {
        save_steps: true,
        dir: &dir,
        stems: ["b".to_string(), "c".to_string()],
        log: &log,
        figures: false,
        commands: &mopt_rs::cli::pair_settings::COMMAND_PARSERS,
    };
    let (b, c) = (col(MB_MINIMA[1]), col(MB_MINIMA[2]));
    let saddle = col(MB_SADDLES[1]);
    let mut failures = Vec::new();
    for method in registry.methods() {
        let name = method.name();
        let mut input = PairInput {
            points: match method.ends() {
                Ends::One => vec![c.clone()],
                _ => vec![b.clone(), c.clone()],
            },
            ..PairInput::default()
        };
        if name == "addf" {
            input.points = vec![b.clone()];
        }
        if name == "se_gsm" {
            input.points = vec![b.clone()];
            input.driving = vec![
                DrivingCoordinate::Variable(0, MB_MINIMA[2][0]),
                DrivingCoordinate::Variable(1, MB_MINIMA[2][1]),
            ];
        }
        let pairs = settings(name)
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        // The experimental methods are measured, not required to succeed.
        let experimental = name == "lmm";
        let mut fail = |line: String| {
            eprintln!("{line}");
            if !experimental {
                failures.push(line);
            }
        };
        let result = match registry.run(name, &input, &PairSettings::new(pairs), &surface, &run) {
            Ok(r) => r,
            Err(e) => {
                fail(format!("{name}: {e}"));
                continue;
            }
        };
        if matches!(name, "bitss" | "ieip" | "gsm" | "fsm" | "dsafir") {
            let path = result.path.as_deref().expect("both branches of the search");
            assert!(path.len() >= 4, "{name}: missing interior points");
            let distance = |x: &Col<f64>, y: &Col<f64>| mopt_rs::math::linalg::col_norm(&(x - y));
            assert!(distance(&path[0].0, &b) < 1e-8, "{name}: first input");
            assert!(
                distance(&path[path.len() - 1].0, &c) < 1e-8,
                "{name}: second input"
            );
            let interior = &path[1..path.len() - 1];
            let left = interior
                .iter()
                .position(|(x, _)| distance(x, &b) < distance(x, &c));
            let right = interior
                .iter()
                .rposition(|(x, _)| distance(x, &c) < distance(x, &b));
            assert!(
                matches!((left, right), (Some(i), Some(j)) if i < j),
                "{name}: missing or reversed branch"
            );
            for (x, energy) in path {
                use mopt_rs::pes::objective::Objective;
                let expected = objective.evaluate(x, false).unwrap().value;
                assert!(
                    (energy - expected).abs() < 1e-7,
                    "{name}: energy belongs to a different frame"
                );
            }
        }
        if result.transition_states.is_empty() {
            fail(format!("{name}: no estimate ({})", result.end));
            continue;
        }
        if matches!(name, "se_gsm" | "gnt") {
            // Both walks start at B and must retain the downhill part past their peak.
            // GNT uses C for its initial direction; reaching C is not its stopping rule.
            let path = result.path.as_ref().unwrap();
            assert!(path.len() >= 3, "{name}");
            assert!(
                mopt_rs::math::linalg::col_norm(&(&path[0].0 - &b)) < 1e-5,
                "{name}"
            );
            let peak = path
                .iter()
                .map(|(_, e)| *e)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(peak > path[0].1 && peak > path.last().unwrap().1, "{name}");
            let filename = if name == "se_gsm" {
                "gsm_path.csv"
            } else {
                "nt_points.csv"
            };
            let saved = std::fs::read_to_string(dir.join(filename)).unwrap();
            let rows: Vec<_> = saved.lines().skip(1).collect();
            assert_eq!(rows.len(), path.len(), "{name}");
            for ((x, energy), row) in path.iter().zip(rows) {
                let fields: Vec<f64> = row.split(',').map(|s| s.parse().unwrap()).collect();
                if name == "se_gsm" {
                    assert!((fields[0] - energy).abs() < 1e-10);
                    assert!((fields[1] - x[0]).abs() < 1e-10);
                    assert!((fields[2] - x[1]).abs() < 1e-10);
                } else {
                    assert!((fields[1] - energy).abs() < 1e-10);
                }
            }
        }
        for w in result.transition_states.windows(2) {
            assert!(w[0].1 <= w[1].1, "{name}: not ranked");
        }
        let nearest = result
            .transition_states
            .iter()
            .map(|(x, _)| mopt_rs::math::linalg::col_norm(&(x - &saddle)))
            .fold(f64::INFINITY, f64::min);
        let line = format!(
            "{name}: {} gradients, {} Hessians, {} energies; nearest estimate {nearest:.4} from \
             the saddle; {}",
            result.counts.gradients, result.counts.hessians, result.counts.energies, result.end
        );
        if result.counts.gradients + result.counts.energies == 0 || nearest >= 0.15 {
            fail(line);
        } else {
            eprintln!("{line}");
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    // The experimental translations of the dimer (descent-ascent steps): what they cost.
    for translation in ["lbfgs", "plain", "optimistic", "extragradient"] {
        let input = PairInput {
            points: vec![b.clone(), c.clone()],
            ..PairInput::default()
        };
        let pairs = vec![
            ("step".to_string(), "0.02".to_string()),
            ("translation".to_string(), translation.to_string()),
            ("rate_across".to_string(), "0.0005".to_string()),
            ("rate_along".to_string(), "0.0005".to_string()),
        ];
        match registry.run("dimer", &input, &PairSettings::new(pairs), &surface, &run) {
            Ok(r) => {
                let nearest = r.transition_states.first().map_or(f64::INFINITY, |(x, _)| {
                    mopt_rs::math::linalg::col_norm(&(x - &saddle))
                });
                eprintln!(
                    "dimer translation={translation}: {} gradients, converged {}, {nearest:.4} \
                     from the saddle",
                    r.counts.gradients, r.converged
                );
            }
            Err(e) => eprintln!("dimer translation={translation}: {e}"),
        }
    }
    // Unknown keys are refused before anything is calculated.
    let input = PairInput {
        points: vec![b, c],
        ..PairInput::default()
    };
    let bad = PairSettings::new(vec![("nodez".to_string(), "3".to_string())]);
    assert!(registry.run("gsm", &input, &bad, &surface, &run).is_err());
    assert!(registry
        .run(
            "no_such_method",
            &input,
            &PairSettings::default(),
            &surface,
            &run
        )
        .is_err());
}
