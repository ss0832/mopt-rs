//! The translations and rotations taken out of the Hessians after every update
//! (`-update_proj`, on by default): for every path that keeps an updated Hessian of a
//! structure, the kept Hessian `H` and the rigid-body basis `D` of the structure it belongs to
//! satisfy `|H D| ~ 0` to round-off (non-linear, linear, a periodic slab, a variable cell, the
//! IRC, the images of a chain; BFGS, FSB, Bofill, and the block updates), and with the option
//! off they do not (the Hessians of 2.1.0). The log shows one spectrum per iteration: that of
//! the exact Hessian where there is one. The option is read from the command line and the
//! settings file of every command that has it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs 2.1.1).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use faer::{Col, Mat};
use mopt_rs::domain::{EvalRequest, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::parse_xyz_str;
use mopt_rs::irc::{HessianUpdate, IrcSpace, PointValue, SurfaceModel};
use mopt_rs::math::projection::build_tr_rot_basis;
use mopt_rs::opt::cell::VariableCell;
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{GeometryOptimizer, IterationReport, OptimizerSettings};
use mopt_rs::opt::vector::{VectorAction, VectorOptimizer, VectorSettings};
use mopt_rs::path::band::force::UpwindProjection;
use mopt_rs::path::band::rfo::{AlongChain, ChainRfo, ChainRfoSettings};
use mopt_rs::path::band::Compatibility;
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::pes::stress::numerical_stress;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::{run_optimization, Observer};

/// `|H D|` over `|H|` (largest entries): zero where `H` has no curvature along `D`.
fn along(h: &Mat<f64>, d: &Mat<f64>) -> f64 {
    let hd = h * d;
    let top = (0..hd.nrows())
        .flat_map(|i| (0..hd.ncols()).map(move |j| (i, j)))
        .fold(0.0_f64, |m, (i, j)| m.max(hd[(i, j)].abs()));
    let scale = (0..h.nrows())
        .flat_map(|i| (0..h.ncols()).map(move |j| (i, j)))
        .fold(0.0_f64, |m, (i, j)| m.max(h[(i, j)].abs()));
    top / scale.max(1e-300)
}

fn structure(text: &str) -> MolecularSystem {
    parse_xyz_str(text, SpinCharge::default(), None).unwrap()
}

/// Four argon atoms, no symmetry.
fn ar4() -> MolecularSystem {
    structure("4\n\nAr 0.00 0.00 0.00\nAr 3.95 0.20 -0.10\nAr 1.80 3.40 0.30\nAr 1.70 1.10 3.30\n")
}

/// Three argon atoms on a line (five rigid motions).
fn ar3_linear() -> MolecularSystem {
    structure("3\n\nAr 0.0 0.0 0.0\nAr 4.3 0.0 0.0\nAr 8.1 0.0 0.0\n")
}

/// Two argon atoms through the x boundary of a slab (periodic in x and y: three translations).
fn ar2_slab() -> MolecularSystem {
    structure(
        "2\nLattice=\"20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 30.0\" pbc=\"T T F\" \
         Properties=species:S:1:pos:R:3\nAr 1.0 10.0 15.0\nAr 18.5 10.0 15.2\n",
    )
}

/// The worst `|H D|` of the updated Hessians of a run, and how many there were.
#[derive(Default)]
struct Watch {
    worst: f64,
    updates: usize,
    exact: usize,
}

impl Observer for Watch {
    fn on_iteration(&mut self, _: &IterationReport) -> mopt_rs::error::MoptResult<()> {
        Ok(())
    }

    fn on_iteration_with_hessian(
        &mut self,
        r: &IterationReport,
        update: Option<(&str, &Mat<f64>)>,
    ) -> mopt_rs::error::MoptResult<()> {
        if r.hessian_eigenvalues.is_some() {
            self.exact += 1;
        }
        if let Some((_, h)) = update {
            let d = build_tr_rot_basis(&r.system, false);
            self.worst = self.worst.max(along(h, &d));
            self.updates += 1;
        }
        Ok(())
    }
}

fn optimize(
    sys: MolecularSystem,
    method: &str,
    order: usize,
    on: bool,
    fc: Option<usize>,
) -> Watch {
    let settings = OptimizerSettings {
        methods: vec![method.to_string()],
        saddle_order: order,
        max_iterations: 12,
        exact_hessian_interval: fc,
        criteria: ConvergenceCriteria::TIGHT,
        dissociation_distance: None,
        project_updates: on,
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, sys).unwrap();
    let mut watch = Watch::default();
    run_optimization(&mut opt, &LennardJones, &mut [&mut watch]).unwrap();
    watch
}

#[test]
fn the_updated_hessians_of_the_optimizer_have_no_rigid_curvature() {
    let methods = [
        "rsirfo_bfgs",
        "rsirfo_fsb",
        "rsirfo_bofill",
        "rsirfo_block_bfgs",
        "rsirfo_block_fsb",
        "rsirfo_block_bofill",
    ];
    for (name, sys) in [
        ("non-linear", ar4()),
        ("linear", ar3_linear()),
        ("slab", ar2_slab()),
    ] {
        for method in methods {
            let w = optimize(sys.clone(), method, 0, true, None);
            assert!(w.updates >= 2, "{name} {method}: {} updates", w.updates);
            assert!(
                w.worst < 1e-12,
                "{name} {method}: |H D| / |H| = {:e}",
                w.worst
            );
        }
    }
    // A saddle-point search (the guard on the inertia keeps the Hessian before an update that
    // gave too many negative curvatures; that one is projected as well), and exact Hessians
    // every third iteration with updates in between.
    for method in ["rsirfo_bofill", "rsirfo_block_bofill"] {
        let w = optimize(ar4(), method, 1, true, None);
        assert!(
            w.updates >= 2 && w.worst < 1e-12,
            "order 1 {method}: {:e}",
            w.worst
        );
    }
    let w = optimize(ar4(), "rsirfo_block_fsb", 0, true, Some(3));
    assert!(
        w.exact >= 2 && w.updates >= 2,
        "{} exact, {} updates",
        w.exact,
        w.updates
    );
    assert!(w.worst < 1e-12, "-fc 3: {:e}", w.worst);
}

#[test]
fn with_the_option_off_the_updates_are_kept_as_they_are() {
    for method in ["rsirfo_bfgs", "rsirfo_block_fsb"] {
        let w = optimize(ar4(), method, 0, false, None);
        assert!(w.updates >= 2, "{method}");
        assert!(w.worst > 1e-6, "{method}: |H D| / |H| = {:e}", w.worst);
    }
}

#[test]
fn the_variable_cell_takes_its_rigid_motions_out_of_the_updates() {
    // Eight argon atoms of fcc in a skewed cell, every atom shifted.
    let a = 5.40;
    let mut rows = Vec::new();
    for f in [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ] {
        for k in 0..2 {
            let n = rows.len() as f64;
            let shift = [
                0.07 * (1.3 * n).sin(),
                0.06 * (0.7 * n).cos(),
                0.05 * (2.1 * n).sin(),
            ];
            rows.push(std::array::from_fn::<f64, 3, _>(|c| {
                a * (f[c] + if c == 2 { f64::from(k) } else { 0.0 }) + shift[c]
            }));
        }
    }
    let mut text = format!(
        "{}\nLattice=\"{a} 0.0 0.0 0.15 {a} 0.0 -0.1 0.12 {}\" pbc=\"T T T\"\n",
        rows.len(),
        2.0 * a
    );
    for r in &rows {
        text.push_str(&format!("Ar {:.10} {:.10} {:.10}\n", r[0], r[1], r[2]));
    }
    let sys = structure(&text);
    let calc = CalculatorRegistry::native()
        .create("lj", &CalculatorOptions::default(), &sys)
        .unwrap();
    for on in [true, false] {
        let map = VariableCell::new(&sys, 2.0, 0.0).unwrap();
        let settings = VectorSettings {
            methods: vec!["rsirfo_bfgs".into()],
            max_iterations: 8,
            criteria: ConvergenceCriteria::TIGHT,
            project_updates: on,
            ..VectorSettings::default()
        };
        let mut opt = VectorOptimizer::new(settings, map.variables(&sys).unwrap()).unwrap();
        let modes = map.clone();
        opt.set_rigid_modes(Box::new(move |x| modes.rigid_modes(x)));
        let (mut worst, mut updates, mut told) = (0.0_f64, 0, 0);
        while let VectorAction::Evaluate { x, .. } = opt.ask() {
            let s = map.system(&x).unwrap();
            let out = calc.evaluate(&s, &EvalRequest::ENERGY_GRAD_STRESS).unwrap();
            let stress = match out.stress {
                Some(stress) => stress,
                None => numerical_stress(calc.as_ref(), &s, 1e-5).unwrap(),
            };
            let (h, slope) = map
                .objective(&s, out.energy.value(), &out.gradient_bohr, &stress)
                .unwrap();
            opt.tell(h, slope, None).unwrap();
            told += 1;
            // From the second point on the Hessian is updated with the pair that ends there.
            if told > 1 {
                let modes = map.rigid_modes(&x).unwrap();
                let d = Mat::from_fn(x.nrows(), modes.len(), |i, j| modes[j][i]);
                worst = worst.max(along(opt.hessian(), &d));
                updates += 1;
            }
        }
        assert!(updates >= 2, "{updates} updates");
        if on {
            assert!(worst < 1e-12, "|H D| / |H| = {worst:e}");
        } else {
            assert!(worst > 1e-6, "off: |H D| / |H| = {worst:e}");
        }
    }
}

#[test]
fn the_hessian_of_an_irc_side_has_no_rigid_curvature_after_its_updates() {
    let sys = ar4();
    let x0 = sys.coords_flat_bohr();
    let at = |x: &Col<f64>, hessian: bool| {
        let s = sys.with_flat_coords_bohr(x).unwrap();
        let request = if hessian {
            EvalRequest::ENERGY_GRAD_HESS
        } else {
            EvalRequest::ENERGY_GRAD
        };
        let out = LennardJones.evaluate(&s, &request).unwrap();
        PointValue::plain(out.energy.value(), out.gradient_bohr, out.hessian_bohr)
    };
    let direction = Col::from_fn(x0.nrows(), |k| 0.05 * (0.9 * k as f64).sin());
    for update in [
        HessianUpdate::Fsb,
        HessianUpdate::Bofill,
        HessianUpdate::GuardedBfgs,
    ] {
        for on in [true, false] {
            let space = IrcSpace::molecule(&sys, true).with_update_projection(on);
            let start = at(&x0, true);
            let h0 = space
                .prepared_hessian(start.hessian.as_ref().unwrap(), &x0)
                .unwrap();
            let mut model = SurfaceModel::new(h0, update);
            assert!(model.observe(&space, &x0, &start).unwrap());
            let mut worst = 0.0_f64;
            for k in 1..6 {
                let x = &x0 + &direction * (k as f64);
                assert!(!model.observe(&space, &x, &at(&x, false)).unwrap());
                let d = build_tr_rot_basis(&sys.with_flat_coords_bohr(&x).unwrap(), true);
                worst = worst.max(along(model.hessian(), &d));
            }
            if on {
                assert!(worst < 1e-12, "{update:?}: |H D| / |H| = {worst:e}");
            } else {
                assert!(worst > 1e-6, "{update:?} off: |H D| / |H| = {worst:e}");
            }
        }
    }
}

#[test]
fn the_images_of_a_chain_keep_hessians_without_rigid_curvature() {
    let a = ar4();
    let b = structure(
        "4\n\nAr 0.10 -0.20 0.00\nAr 4.10 0.00 0.20\nAr 1.50 3.70 -0.20\nAr 2.10 0.80 3.60\n",
    );
    let (xa, xb) = (a.coords_flat_bohr(), b.coords_flat_bohr());
    let n = xa.nrows();
    let images = 4;
    let rule = UpwindProjection {
        block: n,
        climb_after: None,
        compatibility: Compatibility::Consistent,
    };
    for (inner, end) in [
        ("block_bofill", "block_fsb"),
        ("bfgs", "fsb"),
        ("bofill", "block_bfgs"),
    ] {
        for on in [true, false] {
            let settings = ChainRfoSettings {
                inner_update: inner.into(),
                end_update: end.into(),
                project_updates: on,
                ..ChainRfoSettings::default()
            };
            let mut rfo = ChainRfo::new(settings, images, n, Some(a.clone()), rule).unwrap();
            let mut chain: Vec<Col<f64>> = (0..images)
                .map(|i| {
                    let t = i as f64 / (images - 1) as f64;
                    &xa * (1.0 - t) + &xb * t
                })
                .collect();
            let mut worst = 0.0_f64;
            for iteration in 0..5 {
                let mut energies = Vec::new();
                let mut gradients = Vec::new();
                for x in &chain {
                    let s = a.with_flat_coords_bohr(x).unwrap();
                    let out = LennardJones
                        .evaluate(&s, &EvalRequest::ENERGY_GRAD)
                        .unwrap();
                    energies.push(out.energy.value());
                    gradients.push(out.gradient_bohr);
                }
                let forces = rule
                    .forces(&chain, &energies, &gradients, iteration)
                    .unwrap();
                let along_chain = AlongChain {
                    removed: &forces.removed,
                    directions: &forces.directions,
                    climbing: &forces.climbing,
                };
                let steps = rfo
                    .steps(&chain, &energies, &gradients, along_chain)
                    .unwrap();
                if iteration > 0 {
                    for (i, x) in chain.iter().enumerate() {
                        let d = build_tr_rot_basis(&a.with_flat_coords_bohr(x).unwrap(), false);
                        worst = worst.max(along(&rfo.hessian(i).unwrap(), &d));
                    }
                }
                for (x, step) in chain.iter_mut().zip(&steps) {
                    *x += step;
                }
            }
            if on {
                assert!(worst < 1e-12, "{inner}/{end}: |H D| / |H| = {worst:e}");
            } else {
                assert!(worst > 1e-6, "{inner}/{end} off: |H D| / |H| = {worst:e}");
            }
        }
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_update_proj_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The blocks of the log of `dir` iteration by iteration.
fn iterations(dir: &Path) -> Vec<String> {
    let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap();
    log.split("# ITR. ").skip(1).map(str::to_string).collect()
}

#[test]
fn an_iteration_with_an_exact_hessian_shows_one_spectrum() {
    let dir = scratch("log");
    let xyz = dir.join("ar4.xyz");
    std::fs::write(
        &xyz,
        "4\n\nAr 0.00 0.00 0.00\nAr 3.95 0.20 -0.10\nAr 1.80 3.40 0.30\nAr 1.70 1.10 3.30\n",
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let config = mopt_rs::cli::optmain::config_from_cli(&[
        &xyz,
        "-os",
        "lj",
        "-opt",
        "rsirfo_block_fsb",
        "-fc",
        "2",
        "-ns",
        "8",
    ])
    .unwrap();
    let out = mopt_rs::cli::optmain::run(&config, &dir, None).unwrap();
    let blocks = iterations(&out[0].directory);
    assert!(blocks.len() >= 5, "{} iterations", blocks.len());
    let mut exact_with_update = 0;
    for (k, block) in blocks.iter().enumerate() {
        let spectra: Vec<&str> = block
            .lines()
            .filter(|l| l.starts_with("EIGENVALUES (NORMAL COORDINATE"))
            .collect();
        let updated = block.lines().any(|l| l.starts_with("Hessian update:"));
        // The exact Hessians of -fc 2 at the iterations 0, 2, 4, ...
        if k % 2 == 0 {
            assert_eq!(spectra.len(), 1, "iteration {k}:\n{block}");
            // Without the three translations and three rotations of the four atoms.
            assert_eq!(
                spectra[0], "EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES: 6):",
                "iteration {k}"
            );
            if updated {
                exact_with_update += 1;
            }
        } else {
            assert!(spectra.len() <= 1, "iteration {k}:\n{block}");
            if updated {
                assert_eq!(
                    spectra,
                    vec!["EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES: 6):"],
                    "iteration {k}"
                );
            }
        }
    }
    assert!(exact_with_update >= 2, "{exact_with_update}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_hessian_handed_on_has_no_rigid_curvature() {
    // The file a run leaves its Hessian in (relaxedscan hands it to the next point).
    let dir = scratch("final");
    let xyz = dir.join("ar4.xyz");
    std::fs::write(
        &xyz,
        "4\n\nAr 0.00 0.00 0.00\nAr 3.95 0.20 -0.10\nAr 1.80 3.40 0.30\nAr 1.70 1.10 3.30\n",
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    for (mode, projected) in [("on", true), ("off", false)] {
        let mut config = mopt_rs::cli::optmain::config_from_cli(&[
            &xyz,
            "-os",
            "lj",
            "-opt",
            "rsirfo_block_fsb",
            // Tight criteria: under the normal ones argon is converged at the start, before
            // any update.
            "-tcc",
            "-ns",
            "6",
            "-update_proj",
            mode,
        ])
        .unwrap();
        let kept = dir.join(format!("final_{mode}.npy"));
        config.final_hessian = Some(kept.clone());
        let out = mopt_rs::cli::optmain::run(&config, &dir, None).unwrap();
        let h = mopt_rs::io::npy::read_npy_matrix(&kept).unwrap();
        // The last structure evaluated, which the Hessian belongs to.
        let last = mopt_rs::io::xyz::read_xyz_file(&out[0].optimized, SpinCharge::default(), None)
            .unwrap();
        let worst = along(&h, &build_tr_rot_basis(&last, false));
        if projected {
            // The file holds 12 digits of Angstrom for the structure.
            assert!(worst < 1e-9, "{mode}: |H D| / |H| = {worst:e}");
        } else {
            assert!(worst > 1e-6, "{mode}: |H D| / |H| = {worst:e}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_option_is_read_by_every_command_that_has_it() {
    let opt = |args: &[&str]| mopt_rs::cli::optmain::config_from_cli(args);
    assert!(opt(&["a.xyz"]).unwrap().settings.project_updates);
    assert!(
        opt(&["a.xyz", "-update_proj", "on"])
            .unwrap()
            .settings
            .project_updates
    );
    assert!(
        !opt(&["a.xyz", "-update_proj", "off"])
            .unwrap()
            .settings
            .project_updates
    );
    assert!(
        !opt(&["a.xyz", "--update_projection", "OFF"])
            .unwrap()
            .settings
            .project_updates
    );
    assert!(opt(&["a.xyz", "-update_proj", "maybe"]).is_err());
    // The IRC after an optimization and ircmain.
    let irc = opt(&["a.xyz", "-irc", "0.1", "30", "lqa", "-update_proj", "off"]).unwrap();
    assert!(!irc.irc.unwrap().project_updates);
    let ircmain = |args: &[&str]| mopt_rs::cli::ircmain::config_from_cli(args).unwrap().irc;
    assert!(ircmain(&["a.xyz"]).project_updates);
    assert!(!ircmain(&["a.xyz", "-update_proj", "off"]).project_updates);
    // The quasi-Newton steps of the chains of nebmain (with model Hessians, the default).
    let neb = |args: &[&str]| {
        mopt_rs::cli::nebmain::config_from_cli(args)
            .unwrap()
            .chain
            .rfo
            .unwrap()
            .project_updates
    };
    assert!(neb(&["a.xyz"]));
    assert!(!neb(&["a.xyz", "-update_proj", "off"]));
    assert!(mopt_rs::cli::nebmain::config_from_cli(&["a.xyz", "-update_proj", "x"]).is_err());
    // ieipmain (DS-AFIR, the chains of its methods).
    let ieip = |args: &[&str]| mopt_rs::cli::ieipmain_settings::config_from_cli(args);
    assert!(ieip(&["a.xyz", "b.xyz"]).unwrap().project_updates);
    assert!(
        !ieip(&["a.xyz", "b.xyz", "-update_proj", "off"])
            .unwrap()
            .project_updates
    );
}

#[test]
fn the_settings_file_gives_it_and_the_command_line_wins() {
    let dir = scratch("settings");
    let file = dir.join("settings.json");
    std::fs::write(
        &file,
        "{\"INPUT\": [\"a.xyz\"], \"update_projection\": \"off\"}",
    )
    .unwrap();
    let dump = |extra: &[&str], name: &str| -> String {
        let out = dir.join(name);
        let mut argv: Vec<String> = ["mopt", "optmain", "--config"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        argv.push(file.to_string_lossy().into_owned());
        argv.extend(extra.iter().map(|s| s.to_string()));
        argv.push("--dump-config".into());
        argv.push(out.to_string_lossy().into_owned());
        assert_eq!(mopt_rs::cli::main_from_argv(&argv), 0);
        std::fs::read_to_string(out).unwrap()
    };
    // From the file; the command line over it.
    assert!(dump(&[], "file.json").contains("\"update_projection\": \"off\""));
    assert!(dump(&["-update_proj", "on"], "cli.json").contains("\"update_projection\": \"on\""));
    let _ = std::fs::remove_dir_all(&dir);
}
