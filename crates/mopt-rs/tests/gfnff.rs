//! GFN-FF of mopt-rs against xtb 6.7.1: the formaldehyde-water reference of xtb's own unit test
//! (term by term), the molecules of `tests/golden/gfnff.json` (made by `tools/golden/gfnff.py`
//! with the xtb executable: every term and the gradient), and finite differences of the
//! gradient.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Element, MolecularSystem, SpinCharge};
use mopt_rs::pes::native::gfnff::HbCutoff;
use mopt_rs::pes::native::Gfnff;

fn system(symbols: &[&str], x: &[f64], charge: i32) -> MolecularSystem {
    let el: Vec<Element> = symbols
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect();
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge {
            charge,
            multiplicity: 1,
        },
        None,
    )
    .unwrap()
}

fn formaldehyde_water() -> MolecularSystem {
    system(
        &["C", "O", "H", "H", "O", "H", "H"],
        &[
            -5.78520874132429,
            -1.92328475821000,
            -0.02944611115854, //
            -5.57801768832583,
            0.17912532844037,
            0.72444143178660, //
            -4.27822256938673,
            -2.74845397256109,
            -1.13038073598642, //
            -7.47879539136783,
            -2.97570121473211,
            0.39488815557786, //
            -0.83005338399036,
            2.43458470560665,
            -0.78566331969245, //
            -0.74201439536855,
            4.04199055249898,
            0.09144422329636, //
            -2.44679415487233,
            1.69392751177087,
            -0.27417668699116,
        ],
        0,
    )
}

#[test]
fn formaldehyde_water_matches_xtb() {
    let r = Gfnff::default().compute(&formaldehyde_water()).unwrap();
    let e = r.energy;
    let reference = [
        ("bond", e.bond, -0.856707643549),
        ("angle", e.angle, 0.000579711773),
        ("torsion", e.torsion, 0.000000008811),
        ("repulsion", e.repulsion, 0.066881023899),
        ("electrostatic", e.electrostatic, -0.152313816530),
        ("dispersion", e.dispersion, -0.001251669186),
        ("hbond", e.hbond, -0.006894292371),
        ("xbond", e.xbond, 0.0),
        ("batm", e.batm, 0.0),
        ("total", e.total, -0.949706677154),
    ];
    let mut bad = Vec::new();
    for (name, got, want) in reference {
        if (got - want).abs() > 1.0e-11 {
            bad.push(format!("{name}: {got:.12} vs xtb {want:.12}"));
        }
    }
    let xtb_gradient = [
        2.0385312564599E-04,
        -2.3703834345239E-04,
        -1.6520674167641E-04, //
        2.7915165929489E-04,
        5.3894416502993E-04,
        8.4049498744631E-05, //
        2.3023302539725E-04,
        3.1926809269042E-04,
        2.6629833961815E-05, //
        1.5415295964773E-04,
        2.4891813148452E-04,
        2.1032555496691E-05, //
        -2.6519443430388E-04,
        -2.3661785046540E-04,
        5.4901595411924E-05, //
        -3.6823812440073E-04,
        -4.9735396104944E-04,
        7.7245926448093E-06, //
        -2.3395821128126E-04,
        -1.3612023423763E-04,
        -2.9131334583450E-05,
    ];
    let dg = (0..21)
        .map(|k| (r.gradient[k] - xtb_gradient[k]).abs())
        .fold(0.0, f64::max);
    if dg > 1.0e-10 {
        bad.push(format!("max |dg| = {dg:.3e}"));
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn gradient_matches_finite_differences() {
    let sys = formaldehyde_water();
    let calc = Gfnff::default();
    let g = calc.compute(&sys).unwrap().gradient;
    let x0 = sys.coords_flat_bohr();
    let h = 1.0e-4;
    let mut worst = 0.0_f64;
    for k in 0..x0.nrows() {
        let mut xp = x0.clone();
        xp[k] += h;
        let mut xm = x0.clone();
        xm[k] -= h;
        let ep = calc
            .compute(&sys.with_flat_coords_bohr(&xp).unwrap())
            .unwrap()
            .energy
            .total;
        let em = calc
            .compute(&sys.with_flat_coords_bohr(&xm).unwrap())
            .unwrap()
            .energy
            .total;
        worst = worst.max(((ep - em) / (2.0 * h) - g[k]).abs());
    }
    assert!(worst < 1.0e-8, "max |FD - analytic| = {worst:.3e}");
}

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn golden_molecules_match_xtb() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/gfnff.json");
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    let mut largest_shift = 0.0_f64;
    // The largest deviation of each term from xtb, and of the gradient, with the molecule.
    let mut worst = vec![("", 0.0_f64, String::new()); 10];
    let mut worst_gradient = (0.0_f64, String::new());
    let cases = doc.get("cases").unwrap().as_array().unwrap();
    assert_eq!(cases.len(), 173);
    for case in cases {
        let name = case.get("molecule").unwrap().as_str().unwrap();
        let symbols = case.get("symbols").unwrap().as_string_vec().unwrap();
        let sym: Vec<&str> = symbols.iter().map(String::as_str).collect();
        let x = floats(case.get("positions_bohr").unwrap());
        let charge = case.get("charge").unwrap().as_f64().unwrap() as i32;
        let sys = system(&sym, &x, charge);
        // xtb's energy of a single point: the terms of the hydrogen and halogen bonds
        // dropped at the thresholds of their lists.
        let sharp = Gfnff::default().with_hb_cutoff(HbCutoff::Sharp);
        let r = sharp.compute(&sys).unwrap();
        let e = r.energy;
        // The default switches them off smoothly below the thresholds; what that changes.
        let switched = Gfnff::default().compute(&sys).unwrap().energy.total;
        largest_shift = largest_shift.max((switched - e.total).abs());
        eprintln!(
            "{name}: switching the HB and XB terms off changes the energy by {:.3e} Hartree",
            switched - e.total
        );
        let terms = case.get("terms").unwrap();
        for (t, (key, got)) in [
            ("bond", e.bond),
            ("angle", e.angle),
            ("torsion", e.torsion),
            ("repulsion", e.repulsion),
            ("electrostatic", e.electrostatic),
            ("dispersion", e.dispersion),
            ("hbond", e.hbond),
            ("xbond", e.xbond),
            ("batm", e.batm),
            ("total", e.total),
        ]
        .into_iter()
        .enumerate()
        {
            let want = terms.get(key).unwrap().as_f64().unwrap();
            if (got - want).abs() > worst[t].1 {
                worst[t] = (key, (got - want).abs(), name.to_string());
            }
            if (got - want).abs() > 1.0e-10 {
                failures.push(format!("{name} {key}: {got:.12} vs xtb {want:.12}"));
            }
        }
        let g_ref = floats(case.get("gradient").unwrap());
        let dg = (0..g_ref.len())
            .map(|k| (r.gradient[k] - g_ref[k]).abs())
            .fold(0.0, f64::max);
        if dg > worst_gradient.0 {
            worst_gradient = (dg, name.to_string());
        }
        if dg > 1.0e-9 {
            let x0 = sys.coords_flat_bohr();
            let calc = Gfnff::default().with_hb_cutoff(HbCutoff::Sharp);
            let h = 1.0e-4;
            let mut worst_ours: f64 = 0.0;
            let mut worst_xtb: f64 = 0.0;
            for k in 0..x0.nrows() {
                let mut xp = x0.clone();
                xp[k] += h;
                let mut xm = x0.clone();
                xm[k] -= h;
                let ep = calc
                    .compute(&sys.with_flat_coords_bohr(&xp).unwrap())
                    .unwrap()
                    .energy
                    .total;
                let em = calc
                    .compute(&sys.with_flat_coords_bohr(&xm).unwrap())
                    .unwrap()
                    .energy
                    .total;
                let fd = (ep - em) / (2.0 * h);
                worst_ours = worst_ours.max((fd - r.gradient[k]).abs());
                worst_xtb = worst_xtb.max((fd - g_ref[k]).abs());
                if (r.gradient[k] - g_ref[k]).abs() > 1e-6 {
                    eprintln!(
                        "{name} k={k} ours {:.6e} xtb {:.6e} fd {fd:.6e}",
                        r.gradient[k], g_ref[k]
                    );
                }
            }
            failures.push(format!("{name}: max |dg| = {dg:.3e}; FD vs ours {worst_ours:.2e}, FD vs xtb {worst_xtb:.2e}"));
        }
    }
    for (key, d, name) in &worst {
        eprintln!("largest deviation of {key}: {d:.3e} Hartree ({name})");
    }
    eprintln!(
        "largest deviation of the gradient: {:.3e} Hartree/Bohr ({})",
        worst_gradient.0, worst_gradient.1
    );
    eprintln!("largest change by the switching: {largest_shift:.3e} Hartree");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // The switching changes the energy of these molecules by a few 1e-6 Hartree at most.
    assert!(largest_shift < 1.0e-5, "{largest_shift:.3e}");
}

#[test]
fn gradient_of_heavier_elements_matches_finite_differences() {
    // Records of `tests/golden/gfnff.json` with elements of the fifth and sixth periods:
    // 4d and 5d carbonyl, sandwich and eta complexes, lanthanide complexes (aqua, eta, xtb's
    // cerium complex), a noble-gas compound, a polyhalide anion, and hydrogen and halogen bonds
    // to Sb and Te.
    let names = [
        "elements_Mo(CO)6",
        "osmocene",
        "zeise_anion",
        "lanthanum_octaaqua",
        "tris_cyclopentadienyl_cerium",
        "Ce_0a7745_charged",
        "xenon_tetrafluoride",
        "triiodide",
        "water_stibine",
        "dimethyltelluride_ammonia",
    ];
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/gfnff.json");
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = doc.get("cases").unwrap().as_array().unwrap();
    let mut failures = Vec::new();
    for name in names {
        let case = cases
            .iter()
            .find(|c| c.get("molecule").unwrap().as_str() == Some(name))
            .unwrap_or_else(|| panic!("{name} is not in gfnff.json"));
        let symbols = case.get("symbols").unwrap().as_string_vec().unwrap();
        let sym: Vec<&str> = symbols.iter().map(String::as_str).collect();
        let x = floats(case.get("positions_bohr").unwrap());
        let charge = case.get("charge").unwrap().as_f64().unwrap() as i32;
        let sys = system(&sym, &x, charge);
        let calc = Gfnff::default();
        let g = calc.compute(&sys).unwrap().gradient;
        let x0 = sys.coords_flat_bohr();
        let h = 1.0e-4;
        let mut worst = 0.0_f64;
        for k in 0..x0.nrows() {
            let e = |s: f64| {
                let mut xs = x0.clone();
                xs[k] += s * h;
                calc.compute(&sys.with_flat_coords_bohr(&xs).unwrap())
                    .unwrap()
                    .energy
                    .total
            };
            worst = worst.max(((e(1.0) - e(-1.0)) / (2.0 * h) - g[k]).abs());
        }
        eprintln!("{name}: max |FD - analytic| = {worst:.3e}");
        if worst > 1.0e-7 {
            failures.push(format!("{name}: {worst:.3e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn optmain_with_gfnff_and_its_model_hessian() {
    use mopt_rs::cli::optmain::{config_from_cli, run};
    use mopt_rs::model_hessian::resolve;

    let dir = std::env::temp_dir().join(format!("mopt_gfnff_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let xyz = dir.join("methanol.xyz");
    std::fs::write(
        &xyz,
        "6\n\nC -0.046 0.663 0.0\nO -0.046 -0.756 0.0\nH -1.086 0.976 0.0\n\
         H 0.437 1.077 0.889\nH 0.437 1.077 -0.889\nH 0.866 -1.048 0.0\n",
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[
        &xyz,
        "-os",
        "gfnff",
        "-opt",
        "rsirfo_bfgs",
        "-modelhess",
        "gfnff",
    ])
    .unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    assert!(res[0].converged);
    let log = std::fs::read_to_string(res[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("Calculator: gfnff: GFN-FF built into mopt-rs"),
        "{log}"
    );
    let _ = std::fs::remove_dir_all(&dir);

    // The model Hessian is the GFN-FF Hessian: symmetric, consistent with the gradient, with the
    // 3N - 6 vibrations of formaldehyde-water positive or soft.
    let sys = formaldehyde_water();
    let h = resolve("gfnff")
        .unwrap()
        .hessian(&sys, &Col::zeros(21))
        .unwrap();
    for i in 0..21 {
        for j in 0..21 {
            assert!((h[(i, j)] - h[(j, i)]).abs() < 1e-7);
        }
    }
    let calc = Gfnff::default();
    let x0 = sys.coords_flat_bohr();
    let d = 1.0e-4;
    let mut xp = x0.clone();
    xp[4] += d;
    let mut xm = x0.clone();
    xm[4] -= d;
    let gp = calc
        .compute(&sys.with_flat_coords_bohr(&xp).unwrap())
        .unwrap()
        .gradient;
    let gm = calc
        .compute(&sys.with_flat_coords_bohr(&xm).unwrap())
        .unwrap()
        .gradient;
    for i in 0..21 {
        let fd = (gp[i] - gm[i]) / (2.0 * d);
        assert!(
            (fd - h[(i, 4)]).abs() < 1e-4 + 2e-3 * fd.abs(),
            "row {i}: {fd} vs {}",
            h[(i, 4)]
        );
    }
    let (vals, _) = mopt_rs::math::linalg::symmetric_eigen(&h).unwrap();
    let stiff = vals.iter().filter(|v| **v > 1e-2).count();
    assert!(stiff >= 9, "{vals:?}");
}

#[test]
fn topology_is_kept_from_the_first_structure() {
    let sys = formaldehyde_water();
    let calc = Gfnff::default();
    let t0 = calc.topology(&sys).unwrap();
    assert_eq!(t0.bonds(), vec![(0, 1), (0, 2), (0, 3), (4, 5), (4, 6)]);
    assert_eq!(t0.term_counts().0, 4);
    // Stretch the C=O bond far beyond bonding: the kept topology still has it.
    let mut x = sys.coords_flat_bohr();
    for c in 3..6 {
        x[c] += 2.0 * (x[c] - x[c - 3]);
    }
    let far = sys.with_flat_coords_bohr(&x).unwrap();
    let t1 = calc.topology(&far).unwrap();
    assert!(std::sync::Arc::ptr_eq(&t0, &t1));
    let fresh = Gfnff::refreshing();
    let t2 = fresh.topology(&far).unwrap();
    assert!(!t2.bonds().contains(&(0, 1)), "{:?}", t2.bonds());
    // Another charge sets the topology up anew.
    let cation = system(
        &["C", "O", "H", "H", "O", "H", "H"],
        &sys.coords_flat_bohr().iter().copied().collect::<Vec<_>>(),
        1,
    );
    let t3 = calc.topology(&cation).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&t0, &t3));
    // Elements beyond H to Rn are refused, with the range in the message.
    let fr = system(&["Fr", "Cl"], &[0.0, 0.0, 0.0, 0.0, 0.0, 5.0], 0);
    let err = Gfnff::default().compute(&fr).unwrap_err().to_string();
    assert!(
        err.contains("H to Rn (Z = 1 to 86)") && err.contains("not Fr (Z = 87)"),
        "{err}"
    );
}

/// Formaldehyde-water with the C=O bond stretched to 3 times its length.
fn broken_carbonyl() -> MolecularSystem {
    let sys = formaldehyde_water();
    let mut x = sys.coords_flat_bohr();
    for c in 3..6 {
        x[c] += 2.0 * (x[c] - x[c - 3]);
    }
    sys.with_flat_coords_bohr(&x).unwrap()
}

#[test]
fn a_changed_bonding_pattern_is_reported_once() {
    use mopt_rs::pes::Calculator;
    let calc = Gfnff::default();
    calc.compute(&formaldehyde_water()).unwrap();
    assert!(calc.take_warnings().is_empty());
    let far = broken_carbonyl();
    let t = calc.topology(&far).unwrap();
    let change = t.bonding_change(&far).unwrap();
    assert_eq!(change.broken, vec![(0, 1, [0, 0, 0])]);
    calc.compute(&far).unwrap();
    let w = calc.take_warnings();
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(w[0].contains("broken 1-2"), "{}", w[0]);
    calc.compute(&far).unwrap();
    assert!(calc.take_warnings().is_empty());
    calc.compute(&formaldehyde_water()).unwrap();
    let w = calc.take_warnings();
    assert!(w.len() == 1 && w[0].contains("match"), "{w:?}");
    // No check (and no warning) when the topology is set up at every structure.
    let fresh = Gfnff::refreshing();
    fresh.compute(&far).unwrap();
    assert!(fresh.take_warnings().is_empty());
}

#[test]
fn every_golden_structure_matches_its_own_topology() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/gfnff.json");
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    for case in doc.get("cases").unwrap().as_array().unwrap() {
        let name = case.get("molecule").unwrap().as_str().unwrap();
        let symbols = case.get("symbols").unwrap().as_string_vec().unwrap();
        let sym: Vec<&str> = symbols.iter().map(String::as_str).collect();
        let x = floats(case.get("positions_bohr").unwrap());
        let charge = case.get("charge").unwrap().as_f64().unwrap() as i32;
        let sys = system(&sym, &x, charge);
        let t = mopt_rs::pes::native::gfnff::GfnffTopology::new(&sys).unwrap();
        assert_eq!(t.bonding_change(&sys), None, "{name}");
    }
}

#[test]
fn registry_takes_the_topology_from_the_start_and_reads_settings() {
    use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
    use mopt_rs::workflow::{run_optimization, Observer};

    let reg = CalculatorRegistry::native();
    let start = formaldehyde_water();
    let calc = reg
        .create("gfnff", &CalculatorOptions::default(), &start)
        .unwrap();
    // The first evaluation is at another structure: the topology is still the start's.
    let far = broken_carbonyl();
    calc.evaluate(&far, &mopt_rs::domain::EvalRequest::ENERGY_GRAD)
        .unwrap();
    let w = calc.take_warnings();
    assert!(w.len() == 1 && w[0].contains("broken 1-2"), "{w:?}");

    let opts = CalculatorOptions {
        extra: vec![("refresh_topology".into(), "true".into())],
        ..CalculatorOptions::default()
    };
    let fresh = reg.create("gfnff", &opts, &start).unwrap();
    fresh
        .evaluate(&far, &mopt_rs::domain::EvalRequest::ENERGY_GRAD)
        .unwrap();
    assert!(fresh.take_warnings().is_empty());
    let bad = CalculatorOptions {
        extra: vec![("refresh".into(), "true".into())],
        ..CalculatorOptions::default()
    };
    let err = reg.create("gfnff", &bad, &start).err().unwrap().to_string();
    assert!(err.contains("unknown setting 'refresh'"), "{err}");

    // The driver hands the warnings to the observers.
    struct Collect(Vec<String>);
    impl Observer for Collect {
        fn on_iteration(
            &mut self,
            _: &mopt_rs::opt::state_machine::IterationReport,
        ) -> mopt_rs::error::MoptResult<()> {
            Ok(())
        }
        fn on_warning(&mut self, message: &str) -> mopt_rs::error::MoptResult<()> {
            self.0.push(message.to_string());
            Ok(())
        }
    }
    let calc = reg
        .create("gfnff", &CalculatorOptions::default(), &start)
        .unwrap();
    let settings = mopt_rs::opt::state_machine::OptimizerSettings {
        methods: vec!["fire".into()],
        max_iterations: 2,
        dissociation_distance: None,
        ..Default::default()
    };
    let mut opt = mopt_rs::opt::state_machine::GeometryOptimizer::new(settings, far).unwrap();
    let mut seen = Collect(Vec::new());
    run_optimization(&mut opt, calc.as_ref(), &mut [&mut seen]).unwrap();
    assert_eq!(seen.0.len(), 1, "{:?}", seen.0);
}

/// Two water molecules with their oxygen atoms `d` Bohr apart along x.
fn distant_waters(d: f64) -> MolecularSystem {
    let water = [[0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0]];
    let mut x = Vec::new();
    for shift in [0.0, d] {
        for atom in water {
            x.extend([atom[0] + shift, atom[1], atom[2]]);
        }
    }
    system(&["O", "H", "H", "O", "H", "H"], &x, 0)
}

#[test]
fn hydrogen_bond_terms_end_without_a_step() {
    // The threshold of the lists is 250 Bohr^2 for the squared distance of the two oxygen
    // atoms. Just inside and just outside it:
    let threshold = 250.0_f64.sqrt();
    let step_of = |cutoff: HbCutoff| {
        let calc = Gfnff::default().with_hb_cutoff(cutoff);
        let e = |d: f64| calc.compute(&distant_waters(d)).unwrap().energy.total;
        // The energy changes smoothly by its slope times the distance; what is left of the
        // difference across the threshold is the step.
        let h = 1.0e-5;
        let across = e(threshold + h) - e(threshold - h);
        let before = e(threshold - h) - e(threshold - 3.0 * h);
        (across - before).abs()
    };
    let sharp = step_of(HbCutoff::Sharp);
    let switched = step_of(HbCutoff::default());
    eprintln!("step of the energy at the threshold: sharp {sharp:.3e}, switched {switched:.3e}");
    assert!(sharp > 1.0e-11, "step of the sharp lists {sharp:.3e}");
    assert!(switched < 1.0e-13, "step with the switching {switched:.3e}");

    // Inside the range of the switching the gradient is that of the energy.
    let calc = Gfnff::default();
    let sys = distant_waters(15.4);
    let g = calc.compute(&sys).unwrap().gradient;
    let x0 = sys.coords_flat_bohr();
    let h = 1.0e-4;
    let mut worst = 0.0_f64;
    for k in 0..x0.nrows() {
        let mut xp = x0.clone();
        xp[k] += h;
        let mut xm = x0.clone();
        xm[k] -= h;
        let ep = calc
            .compute(&sys.with_flat_coords_bohr(&xp).unwrap())
            .unwrap()
            .energy
            .total;
        let em = calc
            .compute(&sys.with_flat_coords_bohr(&xm).unwrap())
            .unwrap()
            .energy
            .total;
        worst = worst.max(((ep - em) / (2.0 * h) - g[k]).abs());
    }
    assert!(worst < 1.0e-8, "max |FD - analytic| = {worst:.3e}");

    // Below the range the two give the same energy (xtb's energy of a single point).
    let near = distant_waters(5.5);
    let sharp = Gfnff::default().with_hb_cutoff(HbCutoff::Sharp);
    let a = sharp.compute(&near).unwrap().energy.total;
    let b = Gfnff::default().compute(&near).unwrap().energy.total;
    assert_eq!(a, b);

    // The setting of the registry.
    use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
    let reg = CalculatorRegistry::native();
    for (value, ok) in [("off", true), ("on", true), ("0.2", true), ("2", false)] {
        let opts = CalculatorOptions {
            extra: vec![("hb_switch".into(), value.into())],
            ..CalculatorOptions::default()
        };
        assert_eq!(reg.create("gfnff", &opts, &near).is_ok(), ok, "{value}");
    }
}
