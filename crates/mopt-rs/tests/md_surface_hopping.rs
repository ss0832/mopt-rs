//! Zhu-Nakamura surface hopping on a two-state model with a known probability: two atoms
//! whose two adiabatic states come from straight diabatic states along their distance
//! (constant forces and coupling, a sloped crossing), passed once from the lower state.
//! The share of the trajectories that end on the upper state is the probability of one
//! passage of eq. 2.11 of Ishida, Nanbu, and Nakamura for the straight model, and the
//! total energy is conserved across the hops, under thermostats and barostats too (the
//! model in a periodic cell has the stress of its pair).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;

use mopt_rs::domain::{
    Bohr, EvalOutput, EvalRequest, MolecularSystem, PeriodicCell, Quantity, SpinCharge, Units,
};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::io::xyz::parse_xyz_str;
use mopt_rs::kinetics::rate::zhu_nakamura::ZhuNakamuraCrossing;
use mopt_rs::md::hopping::{ForceModel, HopOutcome, HoppingSettings};
use mopt_rs::md::{InitialVelocities, MdSettings, MdSpace, MdStepReport, MolecularDynamics};
use mopt_rs::pes::combine::{PesExpr, TaskSpec};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::driver::CalculatorSet;
use mopt_rs::workflow::md::{run_md, FnMdObserver, MoleculeMdSurface};

/// Two straight diabatic states `V_k = -F_k (r - r_x)` of the distance `r` of two atoms
/// with the coupling `V`; state 0 is the lower adiabatic state, 1 the upper.
struct TwoStateModel {
    forces: [f64; 2],
    coupling: f64,
    crossing: f64,
}

impl TwoStateModel {
    /// The adiabatic energies and their derivatives along `r`.
    fn along(&self, r: f64) -> [(f64, f64); 2] {
        let x = r - self.crossing;
        let [f1, f2] = self.forces;
        let mean = -0.5 * (f1 + f2) * x;
        let half = -0.5 * (f1 - f2) * x;
        let root = (half * half + self.coupling * self.coupling).sqrt();
        let d_half = -0.5 * (f1 - f2);
        let d_root = half * d_half / root;
        [
            (mean - root, -0.5 * (f1 + f2) - d_root),
            (mean + root, -0.5 * (f1 + f2) + d_root),
        ]
    }
}

impl Calculator for TwoStateModel {
    fn name(&self) -> &str {
        "two_state_model"
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if req.excited_state > 1 {
            return Err(MoptError::config("the model has two states"));
        }
        let x = sys.coords_flat_bohr();
        let d: Vec<f64> = (0..3).map(|c| x[3 + c] - x[c]).collect();
        let r = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        let (e, slope) = self.along(r)[req.excited_state];
        let mut g = Col::<f64>::zeros(6);
        for c in 0..3 {
            g[3 + c] = slope * d[c] / r;
            g[c] = -slope * d[c] / r;
        }
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        // In a cell (much wider than the pair): sigma_ab = E'(r) d_a d_b / (r V).
        if let Some(cell) = sys.active_cell() {
            let volume = cell.volume_bohr3();
            out.stress = Some(std::array::from_fn(|a| {
                std::array::from_fn(|b| slope * d[a] * d[b] / (r * volume))
            }));
        }
        Ok(out)
    }
}

/// Two hydrogen atoms (reduced mass 0.504 amu), in `cell` if given.
fn pair_in(r: f64, cell: Option<&PeriodicCell>) -> MolecularSystem {
    let text = format!(
        "2\n\nH 0.0 0.0 0.0\nH {:.15} 0.0 0.0\n",
        r * Units::BOHR_TO_ANGSTROM
    );
    parse_xyz_str(&text, SpinCharge::default(), cell).unwrap()
}

fn pair(r: f64) -> MolecularSystem {
    pair_in(r, None)
}

/// The thermostat, integrator, and barostat of a run, and whether it runs in a periodic
/// cubic cell of 30 Bohr (always with a barostat).
#[derive(Debug, Clone, Copy)]
struct Ensemble {
    thermostat: &'static str,
    integrator: &'static str,
    barostat: Option<&'static str>,
    periodic: bool,
}

const NVE: Ensemble = Ensemble {
    thermostat: "none",
    integrator: "velocity_verlet",
    barostat: None,
    periodic: false,
};

const MODEL: TwoStateModel = TwoStateModel {
    forces: [0.02, 0.005],
    coupling: 0.002,
    crossing: 3.0,
};

/// The energy of the motion along the distance above the crossing energy (Hartree).
const ABOVE: f64 = 0.0055;

struct Passage {
    upper: bool,
    estimate: Option<f64>,
    largest_jump: f64,
    drift: f64,
    hops: usize,
    /// The step at which the hop (or the stay) was decided.
    decided: Option<usize>,
    conserved: Vec<f64>,
    /// The changes of the potential and the kinetic energy in the hop.
    hop_energies: Option<(f64, f64)>,
}

/// One passage from the lower state at `0.8 + offset` Bohr before the crossing, outwards,
/// with time step `dt_fs` (NVE).
fn passage(dt_fs: f64, offset: f64, seed: u64, forces: ForceModel) -> Passage {
    passage_in(dt_fs, offset, seed, forces, NVE)
}

/// [`passage`] in `ensemble` (at 300 K).
fn passage_in(
    dt_fs: f64,
    offset: f64,
    seed: u64,
    forces: ForceModel,
    ensemble: Ensemble,
) -> Passage {
    let r0 = MODEL.crossing - 0.8 - offset;
    let cell = (ensemble.periodic || ensemble.barostat.is_some()).then(|| {
        let side = Quantity::<Bohr>::new(30.0);
        PeriodicCell::orthorhombic(side, side, side).unwrap()
    });
    let sys = pair_in(r0, cell.as_ref());
    let masses = sys.masses_amu();
    let mu = masses[0] * masses[1] / (masses[0] + masses[1]) * Units::AMU_TO_AU_MASS;
    let kinetic = ABOVE - MODEL.along(r0)[0].0;
    let rdot = (2.0 * kinetic / mu).sqrt();
    // Opposite velocities of equal masses: the centre of mass at rest.
    let v = Col::<f64>::from_fn(6, |i| match i {
        0 => -0.5 * rdot,
        3 => 0.5 * rdot,
        _ => 0.0,
    });
    let hopping = HoppingSettings {
        forces,
        ..HoppingSettings::new(vec![0, 1], 0).unwrap()
    };
    let dt = dt_fs * Units::FS_TO_AU_TIME;
    // Long enough to go 0.8 Bohr past the crossing.
    let steps = (2.2 / (rdot * dt)).ceil() as usize + 3;
    let settings = MdSettings {
        time_step: dt,
        steps,
        thermostat: ensemble.thermostat.to_string(),
        integrator: ensemble.integrator.to_string(),
        barostat: ensemble.barostat.map(str::to_string),
        temperature: 300.0,
        velocities: InitialVelocities::Given(v),
        hopping: Some(hopping),
        seed,
        rigid: mopt_rs::md::RigidMotion::None,
        ..MdSettings::default()
    };
    let space = MdSpace::atoms(&masses).unwrap();
    let mut md =
        MolecularDynamics::new(settings, space, sys.coords_flat_bohr(), Some(sys.clone())).unwrap();
    let states = vec![
        (0, PesExpr::single()),
        (1, PesExpr::Task(TaskSpec::whole("pes").excited_state(1))),
    ];
    let surface = MoleculeMdSurface::new(
        CalculatorSet::single(&MODEL),
        PesExpr::single(),
        sys,
        vec![],
    )
    .unwrap()
    .with_states(states)
    .unwrap();
    let mut last: Option<MdStepReport> = None;
    let mut first_conserved = None;
    let mut previous: Option<f64> = None;
    let mut largest_jump = 0.0_f64;
    let mut estimate = None;
    let mut hops = 0;
    let mut decided = None;
    let mut conserved = Vec::new();
    let mut hop_energies = None;
    {
        let mut watch = FnMdObserver::new(|r: &MdStepReport| {
            conserved.push(r.conserved);
            first_conserved.get_or_insert(r.conserved);
            if let Some(p) = previous {
                largest_jump = largest_jump.max((r.conserved - p).abs());
            }
            previous = Some(r.conserved);
            for h in &r.hops {
                if let Some(zn) = h.zn {
                    estimate = estimate.or(zn.probability);
                }
                if h.outcome == HopOutcome::Hopped {
                    hops += 1;
                    hop_energies = Some((h.potential_change, h.kinetic_change));
                }
                decided = decided.or(Some(h.decided));
            }
            last = Some(r.clone());
            Ok(true)
        });
        run_md(&mut md, &surface, &mut [&mut watch]).unwrap();
    }
    let last = last.unwrap();
    Passage {
        upper: last.electronic_state == Some(1),
        estimate,
        largest_jump,
        drift: (last.conserved - first_conserved.unwrap()).abs(),
        hops,
        decided,
        conserved,
        hop_energies,
    }
}

/// The probability of one passage of the straight model at `ABOVE` (eq. 2.11).
fn exact() -> f64 {
    let m = pair(MODEL.crossing).masses_amu();
    let mu = m[0] * m[1] / (m[0] + m[1]);
    let [f1, f2] = MODEL.forces;
    let zn = ZhuNakamuraCrossing::from_diabatic(MODEL.coupling, f1 - f2, f1 * f2, mu).unwrap();
    (-zn.minus_ln_hop(zn.b2(ABOVE))).exp()
}

#[test]
fn the_share_of_hops_is_the_zhu_nakamura_probability_and_the_energy_is_kept() {
    let p = exact();
    assert!(p > 0.3 && p < 0.8, "{p}");
    let n = 400;
    let mut upper = 0;
    let mut estimates = Vec::new();
    let (mut jump, mut drift) = (0.0_f64, 0.0_f64);
    let mut pair: [Option<Passage>; 2] = [None, None];
    for k in 0..n {
        // The smallest gap falls anywhere within a step.
        let offset = 0.1 * k as f64 / n as f64;
        let run = passage(0.25, offset, 1000 + k as u64, ForceModel::TwoState);
        assert!(run.hops <= 1);
        upper += usize::from(run.upper);
        estimates.push(run.estimate.unwrap());
        jump = jump.max(run.largest_jump);
        drift = drift.max(run.drift);
        if k == 0 {
            let slot = usize::from(run.upper);
            pair[slot] = Some(run);
        }
    }
    eprintln!("largest change of the conserved quantity in a step {jump:.3e}, drift {drift:.3e}");
    // NVE (velocity Verlet, 0.25 fs): the energy is kept to the error of the integrator,
    // largest in the coupling region (measured: 6.6e-6 Hartree in a step, see the
    // documentation of mdmain).
    assert!(jump < 1e-5 && drift < 1e-5, "{jump} {drift}");
    let share = upper as f64 / n as f64;
    let sigma = (p * (1.0 - p) / n as f64).sqrt();
    eprintln!("share {share:.4}, ZN {p:.4}, sigma {sigma:.4}");
    assert!(
        (share - p).abs() < 3.0 * sigma,
        "share {share}, ZN {p}, sigma {sigma}"
    );
    let mean = estimates.iter().sum::<f64>() / n as f64;
    assert!((mean / p - 1.0).abs() < 0.03, "estimated {mean}, ZN {p}");
    // The hop itself keeps the energy exactly: a run that hops and one that stays (other
    // seeds, the same start) agree up to the step of the decision, the hop included.
    for seed in 1..200 {
        if pair.iter().all(Option::is_some) {
            break;
        }
        let run = passage(0.25, 0.0, seed, ForceModel::TwoState);
        let k = usize::from(run.upper);
        if pair[k].is_none() {
            pair[k] = Some(run);
        }
    }
    let [Some(stay), Some(hop)] = pair else {
        panic!("no pair of a hop and a stay");
    };
    let at = hop.decided.unwrap();
    assert_eq!(stay.decided, Some(at));
    for s in 0..=at {
        assert!(
            (hop.conserved[s] - stay.conserved[s]).abs() < 1e-13,
            "step {s}: {} {}",
            hop.conserved[s],
            stay.conserved[s]
        );
    }
}

#[test]
fn hops_keep_the_energy_under_thermostats_and_barostats() {
    // (thermostat, integrator, barostat, periodic, with a conserved quantity)
    let ensembles = [
        ("none", "velocity_verlet", None, true, true),
        ("nose_hoover_chain", "velocity_verlet", None, false, true),
        ("nose_hoover_chain", "velocity_verlet", None, true, true),
        ("langevin", "baoab", None, false, true),
        ("bussi", "velocity_verlet", None, true, true),
        (
            "nose_hoover_chain",
            "velocity_verlet",
            Some("mtk"),
            true,
            true,
        ),
        ("bussi", "velocity_verlet", Some("crescale"), true, false),
        (
            "berendsen",
            "velocity_verlet",
            Some("berendsen"),
            true,
            false,
        ),
    ];
    for (thermostat, integrator, barostat, periodic, conserving) in ensembles {
        let ensemble = Ensemble {
            thermostat,
            integrator,
            barostat,
            periodic,
        };
        let mut found = false;
        for seed in 1..60 {
            let run = passage_in(0.25, 0.0, seed, ForceModel::TwoState, ensemble);
            let Some((potential, kinetic)) = run.hop_energies else {
                continue;
            };
            // The hop moves energy between the potential and the kinetic energy only.
            assert!(potential > 0.0, "{ensemble:?}: an upward hop");
            assert!(
                (potential + kinetic).abs() < 1e-14,
                "{ensemble:?}: {potential} {kinetic}"
            );
            // Where the ensemble has a conserved quantity, the step of the hop changes it no
            // more than the integrator does elsewhere.
            if conserving {
                let at = run.decided.unwrap();
                let jump = (run.conserved[at] - run.conserved[at - 1]).abs();
                assert!(
                    jump < 1e-5 && run.largest_jump < 1e-5,
                    "{ensemble:?}: {jump}"
                );
            }
            found = true;
            break;
        }
        assert!(found, "{ensemble:?}: no hop in 59 seeds");
    }
}

#[test]
fn the_estimates_of_the_published_and_the_two_state_forces() {
    // The probability estimated at the gap minimum against that of the straight model, for
    // time steps that put the outer steps inside and outside the coupling region (measured
    // values in the documentation of mdmain).
    let p = exact();
    let mut measured = Vec::new();
    for dt in [0.1, 0.25, 0.5, 1.0] {
        let mut ratio = [0.0; 2];
        let n = 20;
        for k in 0..n {
            let offset = 0.2 * k as f64 / n as f64;
            for (j, forces) in [ForceModel::Published, ForceModel::TwoState]
                .into_iter()
                .enumerate()
            {
                let run = passage(dt, offset, 7, forces);
                ratio[j] += run.estimate.unwrap() / p / n as f64;
            }
        }
        eprintln!(
            "dt {dt} fs: published {:.4}, two_state {:.4}",
            ratio[0], ratio[1]
        );
        measured.push((dt, ratio));
    }
    for (dt, ratio) in measured {
        assert!((ratio[1] - 1.0).abs() < 0.05, "{dt} {ratio:?}");
        assert!(ratio[0] < ratio[1], "{dt} {ratio:?}");
    }
}

fn registry() -> mopt_rs::pes::registry::CalculatorRegistry {
    let mut r = mopt_rs::pes::registry::CalculatorRegistry::native();
    r.register("zn_model", |_| {
        Ok(Box::new(TwoStateModel {
            forces: MODEL.forces,
            coupling: MODEL.coupling,
            crossing: MODEL.crossing,
        }))
    });
    r
}

fn mdmain(xyz: &str, seed: u64) -> mopt_rs::cli::mdmain::MdmainConfig {
    let seed = format!("seed={seed}");
    mopt_rs::cli::mdmain::config_from_cli(&[
        xyz,
        "-os",
        "zn_model",
        "-mt",
        "nve",
        "-time",
        "60",
        "-ts",
        "0.25fs",
        "-temp",
        "300",
        "-md_opt",
        &seed,
        "write_every=1",
        "hopping_states=0,1",
    ])
    .unwrap()
}

#[test]
fn mdmain_logs_the_hops_and_a_stopped_run_restarts_to_the_same_end() {
    use mopt_rs::cli::mdmain::{run_with, run_with_observers};
    use mopt_rs::workflow::journal::JOURNAL_FILE;
    use mopt_rs::workflow::observers::STOP_FILE;

    let base = std::env::temp_dir().join(format!("mopt_md_zn_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let xyz = base.join("h2.xyz");
    std::fs::write(
        &xyz,
        format!(
            "2\n\nH 0.0 0.0 0.0\nH {:.10} 0.0 0.0\n",
            (MODEL.crossing - 0.8) * Units::BOHR_TO_ANGSTROM
        ),
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    // The first seed whose trajectory hops.
    let mut found = None;
    for seed in 1..40 {
        let runs = run_with(&mdmain(&xyz, seed), &base, None, &registry()).unwrap();
        let dir = runs[0].directory.clone();
        let hops = std::fs::read_to_string(dir.join("h2_hops.csv")).unwrap();
        if let Some(line) = hops.lines().find(|l| l.ends_with(",hop")) {
            let step: usize = line.split(',').nth(2).unwrap().parse().unwrap();
            found = Some((seed, runs, step));
            break;
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    let (seed, full, hop_step) = found.expect("a seed with a hop");
    let dir = full[0].directory.clone();
    let reference = std::fs::read_to_string(&full[0].trajectory).unwrap();
    let hops = std::fs::read_to_string(dir.join("h2_hops.csv")).unwrap();
    let states = std::fs::read_to_string(dir.join("h2_states.csv")).unwrap();
    // The active state per frame, in the trajectory and in the table of the states.
    assert!(reference.contains(" state=0") && reference.contains(" state=1"));
    assert!(states.starts_with("step,time_fs,active_state,energy_state_0_hartree"));
    let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap();
    assert!(log.contains("Hop candidate at step"), "{log}");
    assert!(log.contains("random number"), "{log}");
    assert!(
        log.contains("Surface hopping: 1 candidate(s), 1 hop(s)"),
        "{log}"
    );
    assert!(log.contains("Yu, C. Xu"), "{log}");

    // Stopped after the hop; the restart from the journal continues to the same end, with
    // the same active state and random numbers.
    let journal = dir.join(JOURNAL_FILE);
    std::fs::remove_file(&journal).unwrap();
    let mut cfg = mdmain(&xyz, seed);
    cfg.restart_dir = Some(dir.clone());
    let stop_at = dir.join(STOP_FILE);
    let mut writer = FnMdObserver::new(|r: &MdStepReport| {
        if r.step == hop_step + 2 {
            std::fs::write(&stop_at, "").unwrap();
        }
        Ok(true)
    });
    let stopped = run_with_observers(&cfg, &base, None, &registry(), &mut [&mut writer]).unwrap();
    assert!(stopped[0].interrupted);
    let done = run_with(&cfg, &base, None, &registry()).unwrap();
    assert!(!done[0].interrupted);
    assert_eq!(
        std::fs::read_to_string(&done[0].trajectory).unwrap(),
        reference
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("h2_hops.csv")).unwrap(),
        hops
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("h2_states.csv")).unwrap(),
        states
    );
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn what_surface_hopping_does_not_run_with_is_refused() {
    use mopt_rs::cli::mdmain::config_from_cli;
    let base = std::env::temp_dir().join(format!("mopt_md_zn_refuse_{}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    let xyz = base.join("h2.xyz");
    std::fs::write(&xyz, "2\n\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n").unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let refused = |extra: &[&str]| {
        let mut args = vec![xyz.as_str(), "-os", "gfn0xtb"];
        args.extend_from_slice(extra);
        config_from_cli(&args).unwrap_err().to_string()
    };
    let e = refused(&["-mf", "seam", "-md_opt", "hopping_states=0,1"]);
    assert!(e.contains("-mf"), "{e}");
    let e = refused(&["-es", "3", "-md_opt", "hopping_states=1,2"]);
    assert!(e.contains("starting state 3"), "{e}");
    let e = refused(&["-md_opt", "hopping_gap=0.5"]);
    assert!(e.contains("needs hopping_states"), "{e}");
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn mdmain_draws_the_states_and_the_populations_of_several_trajectories() {
    use mopt_rs::cli::mdmain::run_with;
    let base = std::env::temp_dir().join(format!("mopt_md_zn_pop_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let xyz = base.join("h2.xyz");
    std::fs::write(
        &xyz,
        format!(
            "2\n\nH 0.0 0.0 0.0\nH {:.10} 0.0 0.0\n",
            (MODEL.crossing - 0.8) * Units::BOHR_TO_ANGSTROM
        ),
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let mut config = mdmain(&xyz, 3);
    config.trajectories = 8;
    let runs = run_with(&config, &base, None, &registry()).unwrap();
    assert_eq!(runs.len(), 8);
    for r in &runs {
        assert!(r.directory.join("plot_surface_hopping_.png").exists());
    }
    let first = &runs[0].directory;
    let table = std::fs::read_to_string(first.join("surface_hopping_populations.csv")).unwrap();
    let rows: Vec<&str> = table.lines().collect();
    assert_eq!(rows[0], "time_fs,state_0,state_1");
    assert_eq!(rows[1], "0,1,0");
    // The shares add up to one at every step.
    for row in &rows[1..] {
        let f: Vec<f64> = row.split(',').map(|x| x.parse().unwrap()).collect();
        assert!((f[1] + f[2] - 1.0).abs() < 1e-12, "{row}");
    }
    assert!(first.join("plot_surface_hopping_populations_.png").exists());
    let log = std::fs::read_to_string(first.join("mopt.log")).unwrap();
    assert!(log.contains("Surface hopping over 8 trajectories"), "{log}");
    let _ = std::fs::remove_dir_all(&base);
}
