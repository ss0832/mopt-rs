//! Tests of the ONIOM microiterations.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::*;
use crate::domain::AtomId;
use crate::io::xyz::parse_xyz_str;
use crate::pes::oniom::{LinkSpec, MicroOptions, OniomSpec, RegionSpec};
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};

/// Three water molecules; the first is the high layer (GFN0-xTB), all three are on the low
/// level (GFN-FF).
const TRIMER: &str = "9\n0 1\nO 0.000 0.000 0.000\nH 0.960 0.000 0.000\n\
                      H -0.240 0.930 0.000\nO 2.900 0.000 0.000\nH 3.200 0.900 0.000\n\
                      H 3.200 -0.400 0.800\nO 0.000 2.900 0.000\nH 0.900 3.200 0.000\n\
                      H -0.400 3.200 0.800\n";

fn surface() -> RelaxedEnvironment {
    surface_with(MicroOptions::default().settings_for((5.0e-6, f64::INFINITY)))
}

fn surface_with(settings: MicroOniomSettings) -> RelaxedEnvironment {
    let sys = parse_xyz_str(TRIMER, crate::domain::SpinCharge::default(), None).unwrap();
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: (1..=3)
                .map(|i| AtomId::from_one_based(i).unwrap())
                .collect(),
            links: LinkSpec::None,
            spin_charge: None,
        }],
        lower_levels: vec!["gfnff".to_string()],
        lower_options: Vec::new(),
        placement: Default::default(),
        traj_high_only: false,
        micro: Some(MicroOptions::default()),
        embedding: None,
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

#[test]
fn the_core_moves_on_the_relaxed_surface_of_its_environment() {
    let s = surface();
    assert_eq!(s.split().core, vec![0, 1, 2]);
    assert_eq!(s.split().environment, (3..9).collect::<Vec<_>>());
    // Fixed atoms of the environment are not relaxed.
    let held = CoreSplit::of(
        &[crate::pes::oniom::layout::Region {
            atoms: (0..3).map(AtomId::new).collect(),
            links: Vec::new(),
            spin_charge: crate::domain::SpinCharge::default(),
            placement: Default::default(),
            follows: None,
        }],
        9,
        &[4, 5, 1],
    )
    .unwrap();
    assert_eq!(held.environment, vec![3, 6, 7, 8]);
    assert_eq!(held.fixed, vec![4, 5]);
    let core = s.core_system().unwrap();
    let out = s.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
    assert_eq!(out.gradient_bohr.nrows(), 9);
    // The environment was relaxed: it moved from the start.
    let full = s.full_system(&core).unwrap();
    let moved = (9..27)
        .map(|k| (full.coords_flat_bohr()[k] - s.template.coords_flat_bohr()[k]).abs())
        .fold(0.0_f64, f64::max);
    assert!(moved > 1e-3, "the environment did not move ({moved})");
    // The gradient is that of the relaxed surface: central differences of E* along a
    // direction of the core (with a rigid motion in it, which the environment follows). The
    // environment is relaxed to 1e-8 Hartree/Bohr here: an energy at a looser tolerance is
    // off by about the tolerance squared over the softest curvature of the environment, and
    // those errors cancel in the difference only where both sides start from the same
    // environment.
    let tight = surface_with(MicroOptions::default().settings_for((1e-8, f64::INFINITY)));
    let at = tight.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
    let x = core.coords_flat_bohr();
    let d = Col::from_fn(9, |k| ((k * 7 + 3) % 5) as f64 - 2.0);
    let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    let h = 2e-3;
    let energy = |sign: f64| {
        let moved = core
            .with_flat_coords_bohr(&Col::from_fn(9, |k| x[k] + sign * h * d[k] / norm))
            .unwrap();
        tight
            .evaluate(&moved, &EvalRequest::ENERGY_GRAD)
            .unwrap()
            .energy
            .value()
    };
    let numeric = (energy(1.0) - energy(-1.0)) / (2.0 * h);
    let analytic = at
        .gradient_bohr
        .iter()
        .zip(d.iter())
        .map(|(g, v)| g * v / norm)
        .sum::<f64>();
    assert!(
        (numeric - analytic).abs() < 2e-5,
        "dE*/dd: numeric {numeric}, analytic {analytic}"
    );
    // The Hessian of the core with the environment held is symmetric and of the core.
    let hess = s
        .evaluate(&core, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap();
    assert_eq!((hess.nrows(), hess.ncols()), (9, 9));
    // The core and the two displaced cores were relaxed.
    assert!(s.counts().relaxations >= 1);
    assert!(tight.counts().relaxations >= 3);
}

#[test]
fn the_hessian_is_that_of_the_relaxed_surface() {
    // Tight relaxations, so that central differences of the gradient of E* (the
    // environment relaxed again at each displaced core) are a reference.
    let tight = |hessian: CoreHessian| {
        let mut s = MicroOptions::default().settings_for((5.0e-6, f64::INFINITY));
        s.inner.gradient_tolerance = 1e-10;
        s.core_hessian = hessian;
        surface_with(s)
    };
    let (s, held) = (tight(CoreHessian::Effective), tight(CoreHessian::Held));
    let approximate = tight(CoreHessian::Approximate);
    let core = s.core_system().unwrap();
    let x = core.coords_flat_bohr();
    let hessian = |surface: &RelaxedEnvironment| {
        surface
            .evaluate(&core, &EvalRequest::ENERGY_GRAD_HESS)
            .unwrap()
            .hessian_bohr
            .unwrap()
    };
    let (h_eff, h_held) = (hessian(&s), hessian(&held));
    // The pairs of the limited-memory BFGS come from a relaxation before the Hessian.
    approximate
        .evaluate(&core, &EvalRequest::ENERGY_GRAD)
        .unwrap();
    let h_approximate = hessian(&approximate);
    let h = 2e-3;
    let mut worst_effective = 0.0_f64;
    let mut worst_held = 0.0_f64;
    let mut worst_approximate = 0.0_f64;
    for k in [0, 2, 5] {
        let gradient_at = |sign: f64| {
            let moved = core
                .with_flat_coords_bohr(&Col::from_fn(9, |i| {
                    x[i] + if i == k { sign * h } else { 0.0 }
                }))
                .unwrap();
            s.evaluate(&moved, &EvalRequest::ENERGY_GRAD)
                .unwrap()
                .gradient_bohr
        };
        let (gp, gm) = (gradient_at(1.0), gradient_at(-1.0));
        for i in 0..9 {
            let numeric = (gp[i] - gm[i]) / (2.0 * h);
            worst_effective = worst_effective.max((numeric - h_eff[(i, k)]).abs());
            worst_held = worst_held.max((numeric - h_held[(i, k)]).abs());
            worst_approximate = worst_approximate.max((numeric - h_approximate[(i, k)]).abs());
        }
    }
    // The approximate one is between: some of the response, without its solves.
    assert!(
        worst_approximate <= worst_held,
        "approximate {worst_approximate}, held {worst_held}"
    );
    // The effective Hessian has the response of the environment that the held one lacks.
    assert!(
        worst_effective < 2e-4 && worst_effective < 0.5 * worst_held,
        "largest deviation from the relaxed surface: effective {worst_effective}, held \
         {worst_held}"
    );
}

#[test]
fn every_option_of_the_relaxations_reaches_the_same_surface() {
    // Three core structures in a row: each setting relaxes the environment to the same
    // minimum (to its tolerance), so that the energies and gradients of E* agree.
    let sequence = |settings: MicroOniomSettings| {
        let s = surface_with(settings);
        let core = s.core_system().unwrap();
        let x = core.coords_flat_bohr();
        let d = Col::from_fn(9, |k| 0.05 * (((k * 5 + 1) % 7) as f64 / 3.0 - 1.0));
        let mut last = core.clone();
        let out: Vec<(f64, Col<f64>)> = (0..3)
            .map(|step| {
                last = core
                    .with_flat_coords_bohr(&Col::from_fn(9, |k| x[k] + step as f64 * d[k]))
                    .unwrap();
                let e = s.evaluate(&last, &EvalRequest::ENERGY_GRAD).unwrap();
                (e.energy.value(), e.gradient_bohr)
            })
            .collect();
        // With an adaptive tolerance, the last structure relaxed to the full one.
        let full = s.relax_fully(&last).unwrap();
        (out, s.counts(), full)
    };
    // Tight relaxations: the soft rotations of the waters leave energies apart by g^2 / k
    // at the default tolerance.
    let mut base = MicroOptions::default().settings_for((5.0e-6, f64::INFINITY));
    base.inner.gradient_tolerance = 1e-9;
    let (reference, _, none) = sequence(base.clone());
    assert!(none.is_none());
    let with = |f: &dyn Fn(&mut MicroOniomSettings)| {
        let mut s = base.clone();
        f(&mut s);
        s
    };
    let settings = [
        (
            "fragments",
            with(&|s| s.coordinates = EnvironmentCoordinates::Fragments),
        ),
        (
            "truncated Newton",
            with(&|s| s.inner.method = crate::pes::microiter::InnerMethod::TruncatedNewton),
        ),
        ("predict", with(&|s| s.predict = true)),
        ("adaptive", with(&|s| s.adaptive = 0.1)),
        (
            "with a preconditioner",
            with(&|s| {
                s.precondition = Some(PreconditionSettings {
                    model: PRECONDITION_MODEL.to_string(),
                    shift: PRECONDITION_SHIFT,
                })
            }),
        ),
        (
            "the plain check of the curvature",
            with(&|s| {
                s.precondition = None;
                s.inner.mode_kill.guided = false;
            }),
        ),
    ];
    for (name, setting) in settings {
        let adaptive = setting.adaptive > 0.0;
        let (out, counts, full) = sequence(setting);
        assert_eq!(counts.unconverged, 0, "{name}");
        if adaptive {
            // Looser relaxations on the way; the last one to the full tolerance.
            assert!(counts.loose > 0, "{counts:?}");
            let full = full.unwrap();
            let e = out[2].0 + full.energy_change;
            assert!(
                (e - reference[2].0).abs() < 1e-9,
                "{name}: {e} against {}",
                reference[2].0
            );
            continue;
        }
        for ((e, g), (e0, g0)) in out.iter().zip(&reference) {
            assert!((e - e0).abs() < 1e-9, "{name}: energy {e} against {e0}");
            let worst = crate::math::linalg::col_max_abs(&(g - g0));
            assert!(worst < 1e-6, "{name}: gradient off by {worst}");
        }
    }
}

#[test]
fn a_rigid_motion_of_the_core_carries_its_environment() {
    let s = surface();
    let core = s.core_system().unwrap();
    let out = s.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
    let env = s.relaxed_environment(&core).unwrap();
    assert_eq!(s.counts().relaxations, 1);
    // The core turned by 30 degrees about z and moved, as an optimizer recentres and rotates
    // it: the environment moves with it, without a relaxation, at the same energy.
    let turn = std::f64::consts::FRAC_PI_6;
    let (c, sn) = (libm::cos(turn), libm::sin(turn));
    let turn = |p: [f64; 3]| {
        [
            c * p[0] - sn * p[1] + 1.3,
            sn * p[0] + c * p[1] - 0.7,
            p[2] + 2.1,
        ]
    };
    let x = core.coords_flat_bohr();
    let moved = Col::from_fn(x.nrows(), |k| {
        turn([x[3 * (k / 3)], x[3 * (k / 3) + 1], x[3 * (k / 3) + 2]])[k % 3]
    });
    let other = core.with_flat_coords_bohr(&moved).unwrap();
    let again = s.evaluate(&other, &EvalRequest::ENERGY_GRAD).unwrap();
    assert_eq!(s.counts().relaxations, 1);
    assert!((again.energy.value() - out.energy.value()).abs() < 1e-10);
    // The gradient turns with the structure.
    for a in 0..3 {
        let g = [0, 1, 2].map(|d| out.gradient_bohr[3 * a + d]);
        let turned = [c * g[0] - sn * g[1], sn * g[0] + c * g[1], g[2]];
        for (d, t) in turned.iter().enumerate() {
            assert!((again.gradient_bohr[3 * a + d] - t).abs() < 1e-8);
        }
    }
    let carried = s.relaxed_environment(&other).unwrap();
    for k in 0..env.len() / 3 {
        let p = turn([env[3 * k], env[3 * k + 1], env[3 * k + 2]]);
        for d in 0..3 {
            assert!((carried[3 * k + d] - p[d]).abs() < 1e-9);
        }
    }
    // A step that is not a rigid motion relaxes again, from the moved environment.
    let mut bent = moved.clone();
    bent[3] += 0.01;
    s.evaluate(
        &core.with_flat_coords_bohr(&bent).unwrap(),
        &EvalRequest::ENERGY_GRAD,
    )
    .unwrap();
    assert_eq!(s.counts().relaxations, 2);
}

#[test]
fn a_restart_gives_back_the_relaxed_environment() {
    let surface = || {
        let mut s = MicroOptions::default().settings_for((5.0e-6, f64::INFINITY));
        s.precondition = Some(PreconditionSettings {
            model: PRECONDITION_MODEL.to_string(),
            shift: PRECONDITION_SHIFT,
        });
        surface_with(s)
    };
    let first = surface();
    let core = first.core_system().unwrap();
    let out = first.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
    assert_eq!(first.counts().preconditioner_builds, 1);
    let env = first.relaxed_environment(&core).unwrap();
    assert_eq!(env.len(), 18);
    // A run that its journal tells the environment does not relax it again.
    let second = surface();
    second.recall_environment(&core, &env);
    assert_eq!(second.relaxed_environment(&core), Some(env.clone()));
    let again = second.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
    assert_eq!(second.counts().relaxations, 0);
    assert_eq!(second.counts().recalled, 1);
    assert!((again.energy.value() - out.energy.value()).abs() < 1e-12);
    // An environment of another length is not taken.
    second.recall_environment(&core, &env[..3]);
    assert_eq!(second.counts().recalled, 1);
}

#[test]
fn a_bond_of_the_environment_that_breaks_or_forms_is_found() {
    // Ethanol; the environment is the three hydrogens of the methyl group (atoms 5-7).
    let ethanol = "9

C 0.000 0.000 0.000
C 1.520 0.020 0.000
O 2.020 1.330 0.050
                   H 2.990 1.330 0.000
H -0.390 1.030 0.000
H -0.390 -0.510 0.890
                   H -0.390 -0.510 -0.890
H 1.910 -0.510 0.890
H 1.910 -0.510 -0.890
";
    let start = parse_xyz_str(ethanol, crate::domain::SpinCharge::default(), None).unwrap();
    let mut environment = vec![false; 9];
    for a in [4, 5, 6] {
        environment[a] = true;
    }
    assert_eq!(bonds_changed(&start, &start, &environment), None);
    let moved = |atom: usize, to: [f64; 3]| {
        let mut x = start.coords_flat_bohr();
        for c in 0..3 {
            x[3 * atom + c] = to[c] / 0.529_177_210_903;
        }
        start.with_flat_coords_bohr(&x).unwrap()
    };
    // H5 a little farther from C1 (1.25 Angstrom, within the margin): no change.
    assert_eq!(
        bonds_changed(&start, &moved(4, [-0.45, 1.17, 0.0]), &environment),
        None
    );
    // H5 off C1 (2.2 Angstrom): the bond of atoms 1 and 5 broke.
    let off = moved(4, [-0.80, 2.05, 0.0]);
    assert_eq!(
        bonds_changed(&start, &off, &environment).as_deref(),
        Some("the bond of atoms 1 and 5 broke")
    );
    // H6 onto the oxygen (1.0 Angstrom from it): it formed a bond.
    let onto = moved(5, [2.30, 2.25, 0.30]);
    let said = bonds_changed(&start, &onto, &environment).unwrap();
    assert!(said.contains("broke") || said.contains("formed"), "{said}");
}

#[test]
fn files_of_the_core_are_written_with_the_whole_system() {
    let s = surface();
    let core = s.core_system().unwrap();
    s.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
    let d = std::env::temp_dir().join(format!("mopt_oniom_micro_{}", std::process::id()));
    std::fs::create_dir_all(d.join("side")).unwrap();
    let x = core.coords_flat_bohr();
    let mut text = String::new();
    for frame in 0..2 {
        text.push_str(&format!("3\nframe {frame}\n"));
        for a in 0..3 {
            text.push_str(&format!(
                "{} {:.10} {:.10} {:.10}\n",
                core.elements[a].symbol(),
                x[3 * a] * crate::domain::Units::BOHR_TO_ANGSTROM,
                x[3 * a + 1] * crate::domain::Units::BOHR_TO_ANGSTROM,
                x[3 * a + 2] * crate::domain::Units::BOHR_TO_ANGSTROM
            ));
        }
    }
    std::fs::write(d.join("side").join("path.xyz"), &text).unwrap();
    // A file of another system is left alone.
    std::fs::write(d.join("other.xyz"), "1\n\nH 0 0 0\n").unwrap();
    assert_eq!(write_full_structures(&d, &s).unwrap(), 1);
    let written = std::fs::read_to_string(d.join("side").join("path.xyz")).unwrap();
    let headers: Vec<&str> = written.lines().filter(|l| l.trim() == "9").collect();
    assert_eq!(headers.len(), 2, "{written}");
    assert!(written.contains("frame 1"));
    assert_eq!(
        std::fs::read_to_string(d.join("other.xyz")).unwrap(),
        "1\n\nH 0 0 0\n"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_frames_of_an_analysis_take_the_nearest_environment_without_a_relaxation() {
    let s = surface();
    let core = s.core_system().unwrap();
    s.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
    let d = std::env::temp_dir().join(format!("mopt_oniom_analysis_{}", std::process::id()));
    let x = core.coords_flat_bohr();
    // Ten frames of a mode, each displaced from the evaluated core.
    let mut text = String::new();
    for frame in 0..10 {
        text.push_str(&format!("3\nMode 0 Frame {frame}\n"));
        for a in 0..3 {
            let shift = if a == 0 {
                0.01 * (frame as f64 + 1.0)
            } else {
                0.0
            };
            text.push_str(&format!(
                "{} {:.10} {:.10} {:.10}\n",
                core.elements[a].symbol(),
                x[3 * a] * crate::domain::Units::BOHR_TO_ANGSTROM + shift,
                x[3 * a + 1] * crate::domain::Units::BOHR_TO_ANGSTROM,
                x[3 * a + 2] * crate::domain::Units::BOHR_TO_ANGSTROM
            ));
        }
    }
    for sub in ["koopman_analysis", "path"] {
        std::fs::create_dir_all(d.join(sub)).unwrap();
        std::fs::write(d.join(sub).join("frames.xyz"), &text).unwrap();
    }
    let before = s.counts().relaxations;
    // The analysis file alone: no relaxation, every frame with the whole system.
    std::fs::remove_file(d.join("path").join("frames.xyz")).unwrap();
    assert_eq!(write_full_structures(&d, &s).unwrap(), 1);
    assert_eq!(s.counts().relaxations, before);
    let written = std::fs::read_to_string(d.join("koopman_analysis").join("frames.xyz")).unwrap();
    assert_eq!(written.lines().filter(|l| l.trim() == "9").count(), 10);
    // Frames that are structures of the run are relaxed where they are not kept.
    std::fs::write(d.join("path").join("frames.xyz"), &text).unwrap();
    std::fs::remove_dir_all(d.join("koopman_analysis")).unwrap();
    assert_eq!(write_full_structures(&d, &s).unwrap(), 1);
    assert!(s.counts().relaxations >= before + 10);
    let _ = std::fs::remove_dir_all(&d);
}

/// Three argon atoms (the core) in a triangle of 6 Angstrom and one of the environment at its
/// centre, on Lennard-Jones at both levels: by symmetry the environment has no gradient there,
/// and it stands on a saddle point of its own (3.46 Angstrom from each atom of the core, inside
/// the minimum of the pair at 3.87: out of the plane every distance grows towards it).
fn argon_on_a_saddle(settings: MicroOniomSettings) -> RelaxedEnvironment {
    let h = 3.0 * 3.0_f64.sqrt();
    let text = format!(
        "4\n0 1\nAr 0.0 0.0 0.0\nAr 6.0 0.0 0.0\nAr 3.0 {h:.15} 0.0\nAr 3.0 {:.15} 0.0\n",
        h / 3.0
    );
    let sys = parse_xyz_str(&text, crate::domain::SpinCharge::default(), None).unwrap();
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: (1..=3)
                .map(|i| AtomId::from_one_based(i).unwrap())
                .collect(),
            links: LinkSpec::None,
            spin_charge: None,
        }],
        lower_levels: vec!["lj".to_string()],
        lower_options: Vec::new(),
        placement: Default::default(),
        traj_high_only: false,
        micro: Some(MicroOptions::default()),
        embedding: None,
    };
    let registry = CalculatorRegistry::native();
    let options = CalculatorOptions {
        oniom: Some(spec.clone()),
        ..CalculatorOptions::default()
    };
    let oniom = registry.create_oniom("lj", &options, &sys).unwrap();
    let low = registry
        .create("lj", &CalculatorOptions::default(), &sys)
        .unwrap();
    let split = CoreSplit::of(&spec.resolve(&sys).unwrap(), sys.n_atoms(), &[]).unwrap();
    RelaxedEnvironment::new(oniom, low, sys, split, settings).unwrap()
}

#[test]
fn an_environment_on_a_saddle_point_of_its_own_is_left_before_a_hessian_is_taken() {
    for request in [
        EvalRequest::ENERGY_GRAD_HESS,
        EvalRequest::ENERGY_GRAD_HESS_CHECK,
    ] {
        let s = argon_on_a_saddle(MicroOptions::default().settings_for((1e-6, f64::INFINITY)));
        let core = s.core_system().unwrap();
        // The gradient alone: the relaxation has nothing to do, the environment stays on the
        // saddle point.
        let flat = s.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
        let on_saddle = s.full_system(&core).unwrap().coords_flat_bohr();
        assert!(on_saddle[11].abs() < 1e-9, "{}", on_saddle[11]);
        // A Hessian: the block of the environment has a negative curvature out of the plane;
        // the environment is moved off along it, relaxed again, and the Hessian is that of the
        // minimum (the check of stability judged, not refused).
        let out = s.evaluate(&core, &request).unwrap();
        assert!(out.hessian_bohr.is_some(), "{request:?}");
        assert_eq!(s.counts().environment_saddles, 1, "{request:?}");
        assert_eq!(s.counts().environment_saddles_kept, 0);
        let off = s.full_system(&core).unwrap().coords_flat_bohr();
        assert!(
            off[11].abs() > 0.5,
            "{request:?}: z of the environment {}",
            off[11]
        );
        assert!(
            out.energy.value() < flat.energy.value() - 1e-6,
            "{request:?}: {} against {} on the saddle point",
            out.energy.value(),
            flat.energy.value()
        );
    }
}

/// Eight copper atoms of a face-centred cubic fragment, three of them the core, on EMT at
/// both levels (both analytic).
fn copper_surface(settings: MicroOniomSettings) -> RelaxedEnvironment {
    let text = "8\n0 1\nCu 0.0 0.0 0.0\nCu 1.83 1.80 0.02\nCu 1.79 0.03 1.81\n\
                Cu 0.02 1.81 1.80\nCu 3.62 0.01 0.0\nCu 0.0 3.60 0.03\nCu 0.01 0.0 3.61\n\
                Cu 1.80 1.82 3.60\n";
    let sys = parse_xyz_str(text, crate::domain::SpinCharge::default(), None).unwrap();
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: (1..=3)
                .map(|i| AtomId::from_one_based(i).unwrap())
                .collect(),
            links: LinkSpec::None,
            spin_charge: None,
        }],
        lower_levels: vec!["emt".to_string()],
        lower_options: Vec::new(),
        placement: Default::default(),
        traj_high_only: false,
        micro: Some(MicroOptions::default()),
        embedding: None,
    };
    let registry = CalculatorRegistry::native();
    let options = CalculatorOptions {
        oniom: Some(spec.clone()),
        ..CalculatorOptions::default()
    };
    let oniom = registry.create_oniom("emt", &options, &sys).unwrap();
    let low = registry
        .create("emt", &CalculatorOptions::default(), &sys)
        .unwrap();
    let split = CoreSplit::of(&spec.resolve(&sys).unwrap(), sys.n_atoms(), &[]).unwrap();
    RelaxedEnvironment::new(oniom, low, sys, split, settings).unwrap()
}

#[test]
fn the_shells_of_the_environment_count_atoms_by_their_nearest_image_in_a_cell() {
    // A copper atom of the core near a face of a cubic cell of 12 Angstrom, an atom of the
    // environment 2.5 Angstrom from it through that face, one 2.5 Angstrom away within the
    // cell, and one 5.5 Angstrom away: within 3 Angstrom two atoms respond (one by its image),
    // within 6 Angstrom all three; without the cell the one through the face is 9.5 Angstrom
    // away.
    let text = "4\nLattice=\"12 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
                Cu 0.5 6.0 6.0\nCu 10.0 6.0 6.0\nCu 3.0 6.0 6.0\nCu 6.0 6.0 6.0\n";
    let sys = parse_xyz_str(text, crate::domain::SpinCharge::default(), None).unwrap();
    assert!(sys.active_cell().is_some());
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: vec![AtomId::from_one_based(1).unwrap()],
            links: LinkSpec::None,
            spin_charge: None,
        }],
        lower_levels: vec!["emt".to_string()],
        lower_options: Vec::new(),
        placement: Default::default(),
        traj_high_only: false,
        micro: Some(MicroOptions::default()),
        embedding: None,
    };
    let registry = CalculatorRegistry::native();
    let options = CalculatorOptions {
        oniom: Some(spec.clone()),
        ..CalculatorOptions::default()
    };
    let split = CoreSplit::of(&spec.resolve(&sys).unwrap(), sys.n_atoms(), &[]).unwrap();
    let surface = |settings| {
        let oniom = registry.create_oniom("emt", &options, &sys).unwrap();
        let low = registry
            .create("emt", &CalculatorOptions::default(), &sys)
            .unwrap();
        RelaxedEnvironment::new(oniom, low, sys.clone(), split.clone(), settings).unwrap()
    };
    let mut settings = copper_settings(None, Some(3.0 * crate::domain::Units::ANGSTROM_TO_BOHR));
    let s = surface(settings.clone());
    assert_eq!(
        s.shell_counts(&[2.0, 3.0, 6.0]),
        vec![(2.0, 0), (3.0, 2), (6.0, 3)]
    );
    let description = s.description();
    assert!(
        description.contains("2 free environment atom(s)"),
        "{description}"
    );
    assert!(
        description.contains("minimum image in the cell"),
        "{description}"
    );

    settings.core_hessian = CoreHessian::Held;
    let held = surface(settings).description();
    assert!(
        held.contains("0 free environment atoms (micro_hessian=held)"),
        "{held}"
    );
    assert!(held.contains("minimum image in the cell: 4 A: 2"), "{held}");
}

/// The settings of the copper tests: a tight relaxation, and `memory` GiB for the Hessian.
fn copper_settings(memory: Option<f64>, shell_bohr: Option<f64>) -> MicroOniomSettings {
    let mut settings = MicroOptions::default().settings_for((1e-9, f64::INFINITY));
    settings.hessian_memory = memory;
    settings.hessian_shell = shell_bohr;
    settings
}

/// The Hessian of the relaxed surface at the start of `s`.
fn core_hessian_at_start(s: &RelaxedEnvironment) -> faer::Mat<f64> {
    let core = s.core_system().unwrap();
    s.evaluate(&core, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap()
}

/// The largest difference of `a` and `b` against the largest diagonal entry of `a`.
fn relative_difference(a: &faer::Mat<f64>, b: &faer::Mat<f64>) -> f64 {
    let n = a.nrows();
    let scale = (0..n).map(|i| a[(i, i)].abs()).fold(0.0_f64, f64::max);
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    worst / scale
}

#[test]
fn the_split_route_gives_the_hessian_of_the_analytic_route() {
    let whole = copper_surface(copper_settings(None, None));
    assert!(whole.analytic_hessian());
    // 10.7 kB: the analytic Hessian of the whole system (17 kB) does not fit, the differences
    // (6 kB) do; EMT on the model system is analytic, so the split route is taken.
    let split = copper_surface(copper_settings(Some(1e-5), None));
    assert!(!split.analytic_hessian());
    assert!(split.routes.model.is_some());
    assert!(split
        .setup_notes()
        .iter()
        .any(|n| n.contains("the analytic Hessians of the model systems")));
    // The effective Hessians agree to the tolerance of the conjugate gradients of the split
    // route (a relative residual of 1e-3).
    let (a, b) = (core_hessian_at_start(&whole), core_hessian_at_start(&split));
    assert!(relative_difference(&a, &b) < 5e-3, "{a:?}\n{b:?}");
    // The core blocks (the model terms analytic, the lowest level by differences) agree to the
    // error of the central differences (2e-5 of the largest entry at the step of 5e-3 Bohr; a
    // term left out would be off by a tenth or more).
    let held = |memory: Option<f64>| {
        let mut settings = copper_settings(memory, None);
        settings.core_hessian = CoreHessian::Held;
        core_hessian_at_start(&copper_surface(settings))
    };
    let (a, b) = (held(None), held(Some(1e-5)));
    assert!(relative_difference(&a, &b) < 1e-4, "{a:?}\n{b:?}");
    // Nothing fits: the Hessian is refused with what it would take.
    let none = copper_surface(copper_settings(Some(1e-12), None));
    let core = none.core_system().unwrap();
    let refused = none.evaluate(&core, &EvalRequest::ENERGY_GRAD_HESS);
    let message = format!("{:?}", refused.err());
    assert!(message.contains("would take about"), "{message}");
}

#[test]
fn only_the_shell_of_the_environment_responds() {
    let full = core_hessian_at_start(&copper_surface(copper_settings(None, None)));
    // A shell that holds every atom is the whole environment.
    let wide = core_hessian_at_start(&copper_surface(copper_settings(None, Some(100.0))));
    assert!(relative_difference(&full, &wide) < 1e-9);
    // A shell that holds none is the core block with the environment held.
    let mut held_settings = copper_settings(None, None);
    held_settings.core_hessian = CoreHessian::Held;
    let held = core_hessian_at_start(&copper_surface(held_settings));
    let empty = core_hessian_at_start(&copper_surface(copper_settings(None, Some(0.01))));
    assert!(relative_difference(&held, &empty) < 1e-9);
    // A radius between the second and the third nearest atoms of the relaxed environment:
    // the reduction of the analytic Hessian on their rows alone, on the analytic route and on
    // the split route.
    let radius = {
        let s = copper_surface(copper_settings(None, None));
        let core = s.core_system().unwrap();
        s.evaluate(&core, &EvalRequest::ENERGY_GRAD).unwrap();
        let env = s.relaxed_environment(&core).unwrap();
        let x = core.coords_flat_bohr();
        let full = assemble(&s.template, &s.split, &x, &env).unwrap();
        let p = |a: usize| full.position(AtomId::new(a)).value();
        let mut nearest: Vec<f64> = s
            .split
            .environment
            .iter()
            .map(|&e| {
                s.split
                    .core
                    .iter()
                    .map(|&c| {
                        (0..3)
                            .map(|k| (p(e)[k] - p(c)[k]).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        nearest.sort_by(f64::total_cmp);
        0.5 * (nearest[1] + nearest[2])
    };
    let s = copper_surface(copper_settings(None, Some(radius)));
    let near = core_hessian_at_start(&s);
    let core = s.core_system().unwrap();
    let env = s.relaxed_environment(&core).unwrap();
    let x = core.coords_flat_bohr();
    let whole_sys = assemble(&s.template, &s.split, &x, &env).unwrap();
    let rows = hessian::responding_rows(&whole_sys, &s.split, Some(radius));
    assert_eq!(rows.len(), 6, "{radius} Bohr: {rows:?}");
    let h = s
        .oniom
        .evaluate(&whole_sys, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap();
    let at = |atoms: &[usize], k: usize| 3 * atoms[k / 3] + k % 3;
    let (c, e) = (&s.split.core, &s.split.environment);
    let (n, m) = (3 * c.len(), rows.len());
    let hyy = faer::Mat::from_fn(m, m, |i, j| h[(at(e, rows[i]), at(e, rows[j]))]);
    let hyx = faer::Mat::from_fn(m, n, |i, j| h[(at(e, rows[i]), at(c, j))]);
    let response = crate::math::linalg::pseudo_inverse(&hyy).unwrap() * &hyx;
    let expected = faer::Mat::from_fn(n, n, |i, j| {
        h[(at(c, i), at(c, j))] - (0..m).map(|k| hyx[(k, i)] * response[(k, j)]).sum::<f64>()
    });
    assert!(
        relative_difference(&expected, &near) < 1e-6,
        "{expected:?}\n{near:?}"
    );
    // The split route gives the same with the shell.
    let split_near =
        core_hessian_at_start(&copper_surface(copper_settings(Some(1e-5), Some(radius))));
    assert!(relative_difference(&near, &split_near) < 1e-3);
}

#[test]
fn the_dense_block_of_the_environment_gives_the_response_of_the_analytic_route() {
    // 10.7 kB of memory: no analytic Hessian of the whole system, but the dense block of the
    // environment (central differences of its gradient) and the conjugate gradients fit.
    let whole = core_hessian_at_start(&copper_surface(copper_settings(None, None)));
    let with = |solve: ResponseSolve| {
        let mut settings = copper_settings(Some(1e-5), None);
        settings.response_solve = solve;
        copper_surface(settings)
    };
    let dense_surface = with(ResponseSolve::Dense);
    let dense = core_hessian_at_start(&dense_surface);
    let cg = core_hessian_at_start(&with(ResponseSolve::ConjugateGradients));
    let auto = core_hessian_at_start(&with(ResponseSolve::Auto));
    // Against the analytic route both differ by the differences of the core block and the
    // coupling (2e-5 of the largest entry at 5e-3 Bohr, measured 1.3e-4 here); the dense block
    // is exact to its own differences, the conjugate gradients to their residual of 1e-3.
    let (d_dense, d_cg) = (
        relative_difference(&whole, &dense),
        relative_difference(&whole, &cg),
    );
    assert!(
        d_dense < 5e-4 && d_cg < 5e-3 && d_dense <= d_cg + 1e-6,
        "dense {d_dense:.3e}, conjugate gradients {d_cg:.3e}"
    );
    // A core of 3 atoms and 5 environment atoms: 30 gradients for the block against up to
    // 900 products of the conjugate gradients, so the dense block is chosen.
    assert!(relative_difference(&dense, &auto) < 1e-12);
    // The dense response is kept for the correction of the gradient.
    assert!(dense_surface.state.lock().unwrap().dense.is_some());
}

/// A lowest level that records whether the state of the microiterations was locked while it
/// computed (EMT underneath).
struct LockProbe {
    inner: Box<dyn Calculator>,
    surface: std::sync::Arc<std::sync::OnceLock<std::sync::Weak<RelaxedEnvironment>>>,
    locked: std::sync::atomic::AtomicUsize,
    calls: std::sync::atomic::AtomicUsize,
}

impl Calculator for LockProbe {
    fn name(&self) -> &str {
        "lock_probe"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        use std::sync::atomic::Ordering;
        if let Some(s) = self.surface.get().and_then(std::sync::Weak::upgrade) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if s.state.try_lock().is_err() {
                self.locked.fetch_add(1, Ordering::SeqCst);
            }
        }
        self.inner.evaluate(sys, req)
    }
}

#[test]
fn no_lock_is_held_while_the_lowest_level_computes() {
    let text = "8\n0 1\nCu 0.0 0.0 0.0\nCu 1.83 1.80 0.02\nCu 1.79 0.03 1.81\n\
                Cu 0.02 1.81 1.80\nCu 3.62 0.01 0.0\nCu 0.0 3.60 0.03\nCu 0.01 0.0 3.61\n\
                Cu 1.80 1.82 3.60\n";
    let sys = parse_xyz_str(text, crate::domain::SpinCharge::default(), None).unwrap();
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: (1..=3)
                .map(|i| AtomId::from_one_based(i).unwrap())
                .collect(),
            links: LinkSpec::None,
            spin_charge: None,
        }],
        lower_levels: vec!["emt".to_string()],
        lower_options: Vec::new(),
        placement: Default::default(),
        traj_high_only: false,
        micro: Some(MicroOptions::default()),
        embedding: None,
    };
    let registry = CalculatorRegistry::native();
    let options = CalculatorOptions {
        oniom: Some(spec.clone()),
        ..CalculatorOptions::default()
    };
    let oniom = registry.create_oniom("emt", &options, &sys).unwrap();
    let cell = std::sync::Arc::new(std::sync::OnceLock::new());
    let probe = std::sync::Arc::new(LockProbe {
        inner: registry
            .create("emt", &CalculatorOptions::default(), &sys)
            .unwrap(),
        surface: std::sync::Arc::clone(&cell),
        locked: std::sync::atomic::AtomicUsize::new(0),
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    struct Shared(std::sync::Arc<LockProbe>);
    impl Calculator for Shared {
        fn name(&self) -> &str {
            self.0.name()
        }
        fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
            self.0.evaluate(sys, req)
        }
    }
    let split = CoreSplit::of(&spec.resolve(&sys).unwrap(), sys.n_atoms(), &[]).unwrap();
    // The differences and the dense block of the environment (10.7 kB: no analytic route).
    let mut settings = copper_settings(Some(1e-5), None);
    settings.inner.gradient_tolerance = 1e-6;
    let s = std::sync::Arc::new(
        RelaxedEnvironment::new(
            oniom,
            Box::new(Shared(std::sync::Arc::clone(&probe))),
            sys,
            split,
            settings,
        )
        .unwrap(),
    );
    cell.set(std::sync::Arc::downgrade(&s)).unwrap();
    let core = s.core_system().unwrap();
    s.evaluate(&core, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
    let x = core.coords_flat_bohr();
    let moved = core
        .with_flat_coords_bohr(&Col::from_fn(x.nrows(), |i| {
            x[i] + 0.02 * ((i % 3) as f64 - 1.0)
        }))
        .unwrap();
    s.evaluate(&moved, &EvalRequest::ENERGY_GRAD).unwrap();
    use std::sync::atomic::Ordering;
    assert!(probe.calls.load(Ordering::SeqCst) > 10);
    assert_eq!(
        probe.locked.load(Ordering::SeqCst),
        0,
        "evaluations of the lowest level with the state locked"
    );
}

#[test]
fn the_analytic_hessian_of_the_whole_system_reduces_to_that_of_the_relaxed_surface() {
    let s = copper_surface(copper_settings(None, None));
    assert!(s.analytic_hessian());
    let core = s.core_system().unwrap();
    let analytic = s
        .evaluate(&core, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap();
    // Central differences of the gradient of the relaxed surface.
    let x = core.coords_flat_bohr();
    let step = 1e-3;
    let n = x.nrows();
    let mut numeric = faer::Mat::<f64>::zeros(n, n);
    for k in 0..n {
        let g = |sign: f64| {
            let moved = Col::from_fn(n, |i| x[i] + if i == k { sign * step } else { 0.0 });
            s.evaluate(
                &core.with_flat_coords_bohr(&moved).unwrap(),
                &EvalRequest::ENERGY_GRAD,
            )
            .unwrap()
            .gradient_bohr
        };
        let (plus, minus) = (g(1.0), g(-1.0));
        for i in 0..n {
            numeric[(i, k)] = (plus[i] - minus[i]) / (2.0 * step);
        }
    }
    let scale = (0..n)
        .map(|i| analytic[(i, i)].abs())
        .fold(0.0_f64, f64::max);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (analytic[(i, j)] - 0.5 * (numeric[(i, j)] + numeric[(j, i)])).abs() < 1e-3 * scale,
                "({i}, {j}): {} {}",
                analytic[(i, j)],
                numeric[(i, j)]
            );
        }
    }
}

#[test]
fn a_core_on_a_line_carries_its_environment_by_the_translation_alone() {
    // Three atoms on a line, moved along the line and sideways: no turn about the line is
    // defined, and a point beside it moves by the same translation.
    let mobile = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let shift = [0.5, 0.2, -0.1];
    let target = Col::from_fn(9, |k| mobile[k] + shift[k % 3]);
    let moved = RigidMotion::fit(&mobile, &target).apply(&[0.0, 1.0, 0.0]);
    for c in 0..3 {
        assert!((moved[c] - [0.5, 1.2, -0.1][c]).abs() < 1e-12, "{moved:?}");
    }
    // Three atoms in a plane turned by 90 degrees about z: the turn is found.
    let mobile = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
    let turned = Col::from_fn(9, |k| {
        let (x, y, z) = (
            mobile[3 * (k / 3)],
            mobile[3 * (k / 3) + 1],
            mobile[3 * (k / 3) + 2],
        );
        [-y, x, z][k % 3]
    });
    let moved = RigidMotion::fit(&mobile, &turned).apply(&[0.0, 0.0, 1.0]);
    let expected = [0.0, 0.0, 1.0];
    for c in 0..3 {
        assert!((moved[c] - expected[c]).abs() < 1e-9, "{moved:?}");
    }
}
