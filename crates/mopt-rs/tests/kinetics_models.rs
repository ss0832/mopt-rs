//! The kinetic models beyond the canonical one, checked against closed forms and limits:
//! the energy-grained master equation (`egme`: the strong-collision secular equation, the
//! high-pressure limit `rrkm`, the low-pressure limit proportional to the collision frequency,
//! the normalization of exponential down, detailed balance), the two-grain model of hot
//! intermediates (`two_grain`: its fast-cooling limit is the canonical model, its entries are
//! those of RRKM at `E*`), and the semi-Markov chain (`semi_markov` with `kmc`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use mopt_rs::core_infra::rng::Rng;
use mopt_rs::domain::{MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::parse_xyz_str;
use mopt_rs::kinetics::egme::{CollisionModel, EgmeSettings};
use mopt_rs::kinetics::hot::{HotSettings, TwoGrain};
use mopt_rs::kinetics::network::{Bottleneck, KineticNetwork, KineticState};
use mopt_rs::kinetics::rate::tst::SymmetryChoice;
use mopt_rs::kinetics::rrkm::RrkmSettings;
use mopt_rs::kinetics::solve::{Evolution, Expm, KineticSolver};
use mopt_rs::kinetics::{prepare, Conditions, KineticsParts, KineticsSettings, ModelSettings};

fn water() -> MolecularSystem {
    parse_xyz_str(
        "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn molecule(id: usize, label: &str, energy: f64, frequencies: &[f64]) -> KineticState {
    let mut s = KineticState::with_energy(id, label, energy);
    s.system = Some(water());
    s.frequencies = Some(frequencies.to_vec());
    s
}

fn saddle(id: usize, states: [usize; 2], energy: f64, frequencies: &[f64]) -> Bottleneck {
    let mut b = Bottleneck::saddle(id, &format!("TS{id}"), states, energy);
    b.system = Some(water());
    b.frequencies = Some(frequencies.to_vec());
    b.imaginary_wavenumber = frequencies.iter().copied().find(|w| *w < 0.0);
    b
}

/// A dissociation channel of an oxygen atom and a hydrogen molecule far below.
fn channel(id: usize, energy: f64) -> KineticState {
    let o = parse_xyz_str("1\n\nO 0.0 0.0 0.0\n", SpinCharge::default(), None).unwrap();
    let h2 = parse_xyz_str(
        "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n",
        SpinCharge::default(),
        None,
    )
    .unwrap();
    let mut a = KineticState::with_energy(0, "O", 0.5 * energy);
    a.system = Some(o);
    a.frequencies = Some(Vec::new());
    let mut b = KineticState::with_energy(1, "H2", 0.5 * energy);
    b.system = Some(h2);
    b.frequencies = Some(vec![4400.0]);
    let mut s = KineticState::with_energy(id, "DC1", energy);
    s.fragments = vec![a, b];
    s
}

/// A molecule that decays over a saddle point into a channel far below.
fn decay() -> KineticNetwork {
    KineticNetwork {
        states: vec![
            molecule(0, "EQ0", 0.0, &[1600.0, 3600.0, 3700.0]),
            channel(1, -0.4),
        ],
        bottlenecks: vec![saddle(0, [0, 1], 0.04, &[1200.0, 2800.0, -1500.0])],
        ..KineticNetwork::default()
    }
}

fn egme_settings(collision: CollisionModel, omega: f64, t: f64) -> KineticsSettings {
    KineticsSettings {
        conditions: Conditions {
            temperature: t,
            ..Conditions::default()
        },
        rate_theory: "rrkm".to_string(),
        model: "egme".to_string(),
        solver: "expm".to_string(),
        model_settings: ModelSettings {
            egme: EgmeSettings {
                grain: 200.0,
                above_kt: 40.0,
                collision: Some(collision),
                frequency: Some(omega),
                ..EgmeSettings::default()
            },
            ..ModelSettings::default()
        },
        ..KineticsSettings::default()
    }
}

fn resolved(net: &KineticNetwork, s: &KineticsSettings) -> mopt_rs::kinetics::solve::Resolved {
    let p = prepare(net, s, &KineticsParts::default()).unwrap();
    match p.evolution {
        Evolution::Resolved(r) => r,
        _ => panic!("egme gives a resolved model"),
    }
}

#[test]
fn strong_collisions_decay_as_the_secular_equation_says() {
    let net = decay();
    for omega in [1e10, 1e12, 1e14] {
        let r = resolved(&net, &egme_settings(CollisionModel::Strong, omega, 1500.0));
        let rates = r.generator.rates();
        let ln_pi = r.generator.ln_pi.clone().unwrap();
        let grains: Vec<usize> = (0..r.owner.len()).filter(|&f| r.owner[f] == 0).collect();
        let sink = (0..r.owner.len()).find(|&f| r.owner[f] == 1).unwrap();
        let top = grains
            .iter()
            .map(|&f| ln_pi[f])
            .fold(f64::NEG_INFINITY, f64::max);
        let q: f64 = grains.iter().map(|&f| (ln_pi[f] - top).exp()).sum();
        let f: Vec<f64> = grains.iter().map(|&a| (ln_pi[a] - top).exp() / q).collect();
        let k: Vec<f64> = grains.iter().map(|&a| rates.get(sink, a)).collect();
        // The collisions are those of the strong-collision operator.
        for (x, &b) in grains.iter().enumerate().take(5) {
            for (y, &a) in grains.iter().enumerate() {
                if a != b {
                    let want = omega * f[y];
                    assert!((rates.get(a, b) - want).abs() < 1e-9 * want, "{x} {y}");
                }
            }
        }
        // sum_a omega f_a / (omega + k_a - lambda) = 1, for lambda in (0, omega).
        let g = |l: f64| -> f64 {
            f.iter()
                .zip(&k)
                .map(|(fa, ka)| omega * fa / (omega + ka - l))
                .sum::<f64>()
                - 1.0
        };
        let (mut lo, mut hi) = (0.0, omega * (1.0 - 1e-15));
        for _ in 0..300 {
            let mid = 0.5 * (lo + hi);
            if g(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let lambda = 0.5 * (lo + hi);
        let p = r.phenomenological.as_ref().unwrap();
        assert!(
            p.eigenvalues[0].abs() < 1e-6 * lambda,
            "{:?}",
            p.eigenvalues
        );
        let eig = -p.eigenvalues[1];
        assert!(
            (eig / lambda - 1.0).abs() < 1e-6,
            "omega {omega}: {eig} {lambda}"
        );
        // The channel outweighs the molecule by exp(96): the eigenvectors cannot resolve the
        // molecule's part of the equilibrium mode, and the contraction is the result.
        assert!(p.eigen.is_none() && p.eigen_note.is_some());
        let ka = p.contraction_a[(1, 0)];
        assert!(
            (ka / lambda - 1.0).abs() < 2e-2,
            "{ka} {lambda} (gap {})",
            p.gap
        );
        assert!(p.gap > 1.0);
    }
}

#[test]
fn the_eigenpairs_and_the_contraction_agree_for_an_isomerization() {
    // Two wells of similar energy: the rate constants of Bartis and Widom and of the
    // contraction agree to the order of the gap and both keep detailed balance.
    let net = KineticNetwork {
        states: vec![
            molecule(0, "EQ0", 0.0, &[1600.0, 3600.0, 3700.0]),
            molecule(1, "EQ1", 0.002, &[1500.0, 3500.0, 3650.0]),
        ],
        bottlenecks: vec![saddle(0, [0, 1], 0.04, &[1200.0, 2800.0, -1500.0])],
        ..KineticNetwork::default()
    };
    let mut deviations = Vec::new();
    for omega in [1e11, 1e13, 1e15] {
        let r = resolved(&net, &egme_settings(CollisionModel::Strong, omega, 1500.0));
        let p = r.phenomenological.as_ref().unwrap();
        let ke = p
            .eigen
            .as_ref()
            .unwrap_or_else(|| panic!("{:?}", p.eigen_note));
        let ka = &p.contraction_a;
        let kb = &p.contraction_b;
        for (i, j) in [(0, 1), (1, 0)] {
            // The two closures bracket the result of the eigenpairs (as in the model of
            // kinetics_beyond_tst.md, section 3.7).
            assert!(
                ka[(i, j)] <= ke[(i, j)] * (1.0 + 1e-9) && ke[(i, j)] <= kb[(i, j)] * (1.0 + 1e-9),
                "{omega}: closure A {} eigen {} closure B {} (gap {})",
                ka[(i, j)],
                ke[(i, j)],
                kb[(i, j)],
                p.gap
            );
        }
        deviations.push(((ka[(1, 0)] / ke[(1, 0)] - 1.0).abs(), p.gap));
        // The constants of the eigenpairs balance with the populations of the wells (those of
        // the contraction with the populations of its superstates, which hold the grains above
        // the barrier partly in the other well).
        let k_inf = &p.high_pressure;
        let ratio_inf = k_inf[(1, 0)] / k_inf[(0, 1)];
        assert!((ke[(1, 0)] / ke[(0, 1)] / ratio_inf - 1.0).abs() < 1e-4);
        assert!(ke[(1, 0)] < k_inf[(1, 0)]);
    }
    // The deviation shrinks as the gap grows.
    for w in deviations.windows(2) {
        assert!(w[1].0 < w[0].0 && w[1].1 > w[0].1, "{deviations:?}");
    }
    assert!(deviations[2].0 < 1e-2, "{deviations:?}");
}

#[test]
fn the_fall_off_has_its_high_and_low_pressure_limits() {
    let net = decay();
    let alpha = CollisionModel::ExponentialDown { alpha: 300.0 };
    // High pressure: the canonical rate constant of rrkm (the hottest grains react at about
    // 1e15 1/s, so the collisions must be far faster for the grains to stay thermal).
    let r = resolved(&net, &egme_settings(alpha, 1e22, 1500.0));
    let p = r.phenomenological.as_ref().unwrap();
    let k_inf = p.high_pressure[(1, 0)];
    assert!(
        (p.contraction_a[(1, 0)] / k_inf - 1.0).abs() < 1e-6,
        "{} {k_inf}",
        p.contraction_a[(1, 0)]
    );
    assert!((p.contraction_b[(1, 0)] / k_inf - 1.0).abs() < 1e-6);
    // Low pressure: proportional to the collision frequency.
    let k1 = resolved(&net, &egme_settings(alpha, 1e4, 1500.0))
        .phenomenological
        .unwrap()
        .contraction_a[(1, 0)];
    let k2 = resolved(&net, &egme_settings(alpha, 2e4, 1500.0))
        .phenomenological
        .unwrap()
        .contraction_a[(1, 0)];
    assert!((k2 / k1 - 2.0).abs() < 2e-3, "{}", k2 / k1);
    assert!(k1 < 1e-3 * k_inf);
    // In between, k / k_inf rises with the pressure.
    let mut last = 0.0;
    for omega in [1e8, 1e10, 1e12, 1e14] {
        let k = resolved(&net, &egme_settings(alpha, omega, 1500.0))
            .phenomenological
            .unwrap()
            .contraction_a[(1, 0)];
        assert!(k > last && k < k_inf * (1.0 + 1e-9), "{omega}: {k}");
        last = k;
    }
}

#[test]
fn exponential_down_is_normalized_and_balanced() {
    let net = decay();
    let omega = 1e9;
    let r = resolved(
        &net,
        &egme_settings(
            CollisionModel::ExponentialDown { alpha: 300.0 },
            omega,
            1500.0,
        ),
    );
    assert!(r.generator.detailed_balance_residual().unwrap() < 1e-10);
    let rates = r.generator.rates();
    let grains: Vec<usize> = (0..r.owner.len()).filter(|&f| r.owner[f] == 0).collect();
    let unnormalized = r
        .notes
        .iter()
        .any(|n| n.contains("their downward part alone is normalized"));
    for &b in &grains {
        let out: f64 = grains
            .iter()
            .filter(|&&a| a != b)
            .map(|&a| rates.get(a, b))
            .sum();
        if !unnormalized {
            assert!(out <= omega * (1.0 + 1e-10), "{out}");
        }
        assert!(out > 0.0);
    }
    // The thermal distribution within the well is stationary under the collisions alone, and
    // the populations go to the channel.
    let p = prepare(
        &net,
        &egme_settings(
            CollisionModel::ExponentialDown { alpha: 300.0 },
            omega,
            1500.0,
        ),
        &KineticsParts::default(),
    )
    .unwrap();
    let p0 = p.evolution.expand(&[1.0, 0.0]);
    assert!((p0.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let mut rng = Rng::from_seed(1);
    let sol = Expm
        .solve(&p.evolution, &p0, &[1e-3, 1.0], &mut rng)
        .unwrap();
    let late = p.evolution.fold(&sol.populations[1]);
    assert!((late[1] - 1.0).abs() < 1e-9, "{late:?}");
}

/// a -> I -> b with a deep intermediate (the energies in Hartree).
fn activated() -> KineticNetwork {
    KineticNetwork {
        states: vec![
            molecule(0, "EQ0", 0.0, &[1600.0, 3600.0, 3700.0]),
            molecule(1, "EQ1", -0.05, &[1500.0, 3500.0, 3650.0]),
            molecule(2, "EQ2", -0.08, &[1550.0, 3550.0, 3600.0]),
        ],
        bottlenecks: vec![
            saddle(0, [0, 1], 0.03, &[1300.0, 3000.0, -1200.0]),
            saddle(1, [1, 2], 0.0, &[1250.0, 2900.0, -1400.0]),
        ],
        ..KineticNetwork::default()
    }
}

fn hot_settings(cooling: f64, model: &str) -> KineticsSettings {
    KineticsSettings {
        conditions: Conditions {
            temperature: 600.0,
            ..Conditions::default()
        },
        rate_theory: "rrkm".to_string(),
        model: model.to_string(),
        solver: "expm".to_string(),
        model_settings: ModelSettings {
            hot: HotSettings {
                cooling: Some(cooling),
                threshold: 0.1,
            },
            ..ModelSettings::default()
        },
        ..KineticsSettings::default()
    }
}

#[test]
fn a_hot_intermediate_cools_into_the_canonical_model() {
    let net = activated();
    let parts = KineticsParts::default();
    let canonical = prepare(&net, &hot_settings(1.0, "canonical"), &parts).unwrap();
    let times = [1e-12, 1e-9, 1e-6, 1e-3];
    let mut rng = Rng::from_seed(2);
    let want = Expm
        .solve(&canonical.evolution, &[1.0, 0.0, 0.0], &times, &mut rng)
        .unwrap();
    // Fast cooling (H is then small: the grain is forced with the threshold 0): the canonical
    // populations.
    let mut forced = hot_settings(1e24, "two_grain");
    forced.model_settings.hot.threshold = 0.0;
    let fast = prepare(&net, &forced, &parts).unwrap();
    let Evolution::Resolved(r) = &fast.evolution else {
        panic!("two_grain gives a resolved model");
    };
    assert_eq!(r.generator.n, 4, "{:?}", r.notes);
    assert!(r.generator.detailed_balance_residual().unwrap() < 1e-10);
    let p0 = fast.evolution.expand(&[1.0, 0.0, 0.0]);
    let got = Expm.solve(&fast.evolution, &p0, &times, &mut rng).unwrap();
    for (g, w) in got.populations.iter().zip(&want.populations) {
        let folded = fast.evolution.fold(g);
        for (a, b) in folded.iter().zip(w) {
            assert!((a - b).abs() < 1e-8, "{folded:?} {w:?}");
        }
    }
    // Slow cooling: the entries of the hot grain are those of RRKM at E*.
    let model = TwoGrain {
        settings: HotSettings {
            cooling: Some(1e6),
            threshold: 0.1,
        },
        rrkm: RrkmSettings::default(),
        symmetry: SymmetryChoice::Auto,
    };
    let c = Conditions {
        temperature: 600.0,
        ..Conditions::default()
    };
    let slow = prepare(&net, &hot_settings(1e6, "two_grain"), &parts).unwrap();
    let entrances = model.entrances(&net, &slow.rates, &c, 1e6).unwrap();
    let e = entrances
        .iter()
        .find(|h| h.from == 0 && h.intermediate == 1)
        .unwrap();
    assert!(e.grain && e.h > 1.0, "{e:?}");
    let Evolution::Resolved(r) = &slow.evolution else {
        panic!("two_grain gives a resolved model");
    };
    let k = r.generator.rates();
    let star = 3;
    assert!((k.get(0, star) / e.back - 1.0).abs() < 1e-10);
    let (_, kx) = e.exits[0];
    assert!((k.get(2, star) / kx - 1.0).abs() < 1e-10);
    assert!((e.h - kx / 1e6).abs() < 1e-10 * e.h);
    // The chemically activated branching of I*: k_x / (k_x + k_back + cooling).
    let out = k.get(0, star) + k.get(1, star) + k.get(2, star);
    let share = k.get(2, star) / out;
    let want = kx / (kx + e.back + k.get(1, star));
    assert!((share - want).abs() < 1e-12);
}

#[test]
fn a_semi_markov_chain_through_the_model_registry() {
    let mut net = activated();
    net.states[1].extra.residence_shape = Some(3.0);
    let s = KineticsSettings {
        conditions: Conditions {
            temperature: 600.0,
            ..Conditions::default()
        },
        rate_theory: "rrkm".to_string(),
        model: "semi_markov".to_string(),
        ..KineticsSettings::default()
    };
    let parts = KineticsParts::default();
    let p = prepare(&net, &s, &parts).unwrap();
    let Evolution::SemiMarkov(sm) = &p.evolution else {
        panic!("semi_markov gives a semi-Markov chain");
    };
    assert_eq!(sm.shapes, vec![1.0, 3.0, 1.0]);
    assert!(sm.memory.is_some());
    // The linear solvers see its Markov limit, the canonical generator.
    let canonical = prepare(
        &net,
        &KineticsSettings {
            model: "canonical".to_string(),
            ..s.clone()
        },
        &parts,
    )
    .unwrap();
    let mut rng = Rng::from_seed(3);
    let a = Expm
        .solve(&p.evolution, &[1.0, 0.0, 0.0], &[1e-9], &mut rng)
        .unwrap();
    let b = Expm
        .solve(&canonical.evolution, &[1.0, 0.0, 0.0], &[1e-9], &mut rng)
        .unwrap();
    assert_eq!(a.populations, b.populations);
    assert!(parts
        .models
        .create("semi_markov", &ModelSettings::default())
        .is_ok());
}
