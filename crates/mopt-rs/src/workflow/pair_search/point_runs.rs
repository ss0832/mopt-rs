//! The drivers of the searches that evaluate a few points at a time: the dimer method, the
//! spring pair method, the two-point scaled hypersphere search, and the growing Newton
//! trajectory, with MultiOptPy's files where it has them. Every iteration shows its
//! structures in Cartesian coordinates and appends them to the trajectories of the run
//! (MultiOptPy's dimer writes a file per iteration into `dimer_structures/`).
//!
//! MultiOptPy: `OtherMethod/dimer.py:DimerMethod.run`,
//! `OtherMethod/spring_pair_method.py:SpringPairMethod.iteration`,
//! `OtherMethod/twopshs.py:twoPSHSlikeMethod.run`, `OtherMethod/newton_traj.py:NewtonTrajectory.main`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::files::{
    frame_text, local_extrema, progress_figure, single_plot, write_frames, write_indices,
    write_points, write_table,
};
use super::surface::{evaluate_all, SearchSurface};
use super::{log_notes, log_structures, SearchRun};
use crate::domain::{MolecularSystem, Units};
use crate::error::MoptResult;
use crate::math::linalg::{col_max_abs, col_norm};
use crate::path::dimer::{Dimer, DimerEnd};
use crate::path::newton_trajectory::{GrowingNewtonTrajectory, NtEnd};
use crate::path::points::{PointRequest, PointValue, SearchAction};
use crate::path::spring_pair::{SpringPair, SpringPairEnd};
use crate::path::twopshs::{TwoPointShs, TwoPshsEnd};
use crate::workflow::chain::Flow;
use crate::workflow::observers::StopFile;
use crate::workflow::trajectories::{GrowingTrajectory, PairTrajectories};

/// How such a search went.
#[derive(Debug, Clone)]
pub struct PointOutcome {
    /// Whether it ended on its criteria.
    pub converged: bool,
    /// Why it ended, in words (`None`: stopped by a hook or a `STOP` file).
    pub end: Option<String>,
    /// Whether a `STOP` file interrupted it.
    pub interrupted: bool,
    /// Iterations (translations, cycles, spheres, points of the trajectory).
    pub iterations: usize,
    /// The estimate of the transition state and its energy.
    pub transition_state: Option<(Col<f64>, f64)>,
}

/// One step of the shared loop: evaluate what `asked` asks for in the state of `image`.
pub(super) fn evaluate(
    surface: &dyn SearchSurface,
    run: &SearchRun,
    points: &[PointRequest],
    image_of: &(dyn Fn(usize) -> usize + Sync),
    iteration: usize,
) -> MoptResult<Vec<PointValue>> {
    surface.begin_iteration(iteration);
    let values = evaluate_all(surface, points, image_of)?;
    log_notes(surface, run.log);
    Ok(values
        .iter()
        .zip(points)
        .map(|(v, p)| v.point_value(p))
        .collect())
}

fn frame(
    surface: &dyn SearchSurface,
    x: &Col<f64>,
    image: usize,
    comment: String,
) -> Option<(MolecularSystem, String)> {
    surface.structure(x, image).map(|s| (s, comment))
}

/// Run a dimer search.
pub fn run_dimer(
    machine: &mut Dimer,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&crate::path::dimer::DimerReport) -> MoptResult<Flow>,
) -> MoptResult<PointOutcome> {
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let mut trajectory: Option<GrowingTrajectory> = None;
    let mut modes: Option<GrowingTrajectory> = None;
    let (mut energies, mut curvatures, mut forces) = (Vec::new(), Vec::new(), Vec::new());
    let mut rows = Vec::new();
    let mut points = Vec::new();
    let mut flow = Flow::Continue;
    let mut evaluations = 0;
    loop {
        let asked = match machine.ask() {
            SearchAction::Evaluate(p) => p,
            SearchAction::Finished => break,
        };
        evaluations += asked.len();
        let values = evaluate(surface, run, &asked, &|_| 0, machine.iterations())?;
        let r = machine.tell(values)?;
        if let Some(angle) = r.rotation {
            run.log.debug(&format!(
                "Rotation: {:.4} rad, curvature {:.6e}",
                angle, r.curvature
            ));
        }
        if r.step.is_none() && r.end.is_none() {
            continue;
        }
        let k = r.iteration;
        run.log.info(&format!(
            "Iteration {k}: Energy = {:.10}, Curvature = {:.6e}, Max Force = {:.6e}, RMS Force = \
             {:.6e}{}",
            r.energy,
            r.curvature,
            r.max_force,
            r.rms_force,
            r.step
                .map_or_else(String::new, |s| format!(", step {s:.4e}"))
        ));
        energies.push(r.energy);
        curvatures.push(r.curvature);
        forces.push(r.max_force);
        rows.push(vec![k as f64, r.energy, r.curvature]);
        points.push((r.energy, r.center.clone()));
        log_structures(
            surface,
            run.log,
            &[("Centre of the dimer".to_string(), &r.center, 0)],
        );
        if run.save_steps {
            if let Some(sys) = surface.structure(&r.center, 0) {
                let comment = format!(
                    "Dimer optimization - Iteration {k} - Energy {:.6} - Curvature {:.6}",
                    r.energy, r.curvature
                );
                let centres = match &mut trajectory {
                    Some(t) => t,
                    slot => slot.insert(GrowingTrajectory::start(
                        run.dir.join("dimer_trajectory.xyz"),
                    )?),
                };
                centres.push(&frame_text(&sys, &comment))?;
                let scale = 0.3 / col_max_abs(&r.axis).max(1e-10);
                if let Some(mode) =
                    surface.structure(&(&r.axis * (scale * Units::ANGSTROM_TO_BOHR)), 0)
                {
                    let axes = match &mut modes {
                        Some(t) => t,
                        slot => slot.insert(GrowingTrajectory::start(
                            run.dir.join("dimer_eigenmodes.xyz"),
                        )?),
                    };
                    axes.push(&frame_text(
                        &mode,
                        &format!(
                            "Dimer eigenmode - Iteration {k} - Curvature {:.6}",
                            r.curvature
                        ),
                    ))?;
                }
                if r.end == Some(DimerEnd::Converged) {
                    write_frames(&run.dir.join("dimer_saddle_point.xyz"), &[(sys, comment)])?;
                }
            }
            write_table(
                &run.dir.join("dimer_results.csv"),
                "iteration,energy,curvature",
                &rows,
            )?;
        }
        if run.figures {
            progress_figure(
                run.dir,
                "dimer_progress",
                "Dimer: energy at the centre",
                "Energy (Hartree)",
                &[("energy", &energies)],
            )?;
        }
        flow = hook(&r)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    if run.save_steps {
        if trajectory.is_none() {
            let p: Vec<(f64, &Col<f64>)> = points.iter().map(|(e, x)| (*e, x)).collect();
            write_points(&run.dir.join("dimer_points.csv"), "energy,variables", &p)?;
        }
        single_plot(
            run.dir,
            "dimer_energy",
            "dimer_energy",
            "Energy [Hartree]",
            &energies,
        )?;
        single_plot(
            run.dir,
            "dimer_curvature",
            "dimer_curvature",
            "Curvature",
            &curvatures,
        )?;
        single_plot(
            run.dir,
            "dimer_gradient",
            "dimer_gradient",
            "Max Force [a.u.]",
            &forces,
        )?;
    }
    let end = machine.end();
    run.log.info(&format!("Dimer: {evaluations} evaluations."));
    Ok(PointOutcome {
        converged: end == Some(DimerEnd::Converged),
        end: end.map(|e| match e {
            DimerEnd::Converged => "the dimer converged to a saddle point".to_string(),
            DimerEnd::IterationLimit => "the iterations are used up".to_string(),
        }),
        interrupted: stop.stopped() && flow == Flow::Continue,
        iterations: machine.iterations(),
        transition_state: Some((
            machine.center().clone(),
            energies.last().copied().unwrap_or(f64::NAN),
        )),
    })
}

/// The spring pair starts both particles at the same end. Follow the history of the
/// particle identified by the solver as its saddle estimate, without walking back
/// down the other particle's history to the starting minimum.
pub fn spring_pair_path(
    machine: &mut SpringPair,
    surface: &dyn SearchSurface,
    run: &SearchRun,
) -> MoptResult<(PointOutcome, Vec<(Col<f64>, f64)>)> {
    let mut paths = [Vec::new(), Vec::new()];
    let outcome = run_spring_pair(machine, surface, run, &mut |r| {
        if r.drift_step.is_none() {
            for (i, path) in paths.iter_mut().enumerate() {
                path.push((r.points[i].clone(), r.energies[i]));
            }
        }
        Ok(Flow::Continue)
    })?;
    let path = machine
        .saddle_estimate()
        .map(|(i, _)| std::mem::take(&mut paths[i]))
        .unwrap_or_default();
    Ok((outcome, path))
}

/// Run the spring pair method.
pub fn run_spring_pair(
    machine: &mut SpringPair,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&crate::path::spring_pair::SpringPairReport) -> MoptResult<Flow>,
) -> MoptResult<PointOutcome> {
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let (mut e1, mut e2, mut rows) = (Vec::new(), Vec::new(), Vec::new());
    let mut particles: Option<PairTrajectories> = None;
    let mut flow = Flow::Continue;
    let mut evaluations = 0;
    let mut last_energy = f64::NAN;
    loop {
        let asked = match machine.ask() {
            SearchAction::Evaluate(p) => p,
            SearchAction::Finished => break,
        };
        evaluations += asked.len();
        let values = evaluate(surface, run, &asked, &|_| 0, machine.cycles())?;
        let r = machine.tell(values)?;
        if let Some(d) = r.drift_step {
            run.log.debug(&format!(
                "Cycle {} drift {d}: force across the line {:.3e} {:.3e}, spring {:.4}",
                r.cycle, r.perpendicular[0], r.perpendicular[1], r.length
            ));
        } else {
            run.log.info(&format!(
                "Cycle {}: energies {:.10} {:.10}, largest forces {:.3e} {:.3e}, spring length {:.5} \
                 Bohr",
                r.cycle, r.energies[0], r.energies[1], r.max_force[0], r.max_force[1], r.length
            ));
            e1.push(r.energies[0]);
            e2.push(r.energies[1]);
            rows.push(vec![
                r.cycle as f64,
                r.energies[0],
                r.energies[1],
                r.max_force[0],
                r.max_force[1],
                r.length,
            ]);
            last_energy = if r.max_force[0] <= r.max_force[1] {
                r.energies[0]
            } else {
                r.energies[1]
            };
            let shown: Vec<(String, &Col<f64>, usize)> = r
                .points
                .iter()
                .enumerate()
                .map(|(i, x)| (format!("Particle {}", i + 1), x, 0))
                .collect();
            log_structures(surface, run.log, &shown);
            if run.save_steps {
                for (i, x) in r.points.iter().enumerate().take(2) {
                    if let Some((sys, comment)) = frame(
                        surface,
                        x,
                        0,
                        format!("SPM cycle {} particle {}", r.cycle, i + 1),
                    ) {
                        let files = match &mut particles {
                            Some(t) => t,
                            slot => slot.insert(PairTrajectories::start(
                                run.dir,
                                ["spm_particle_1_traj.xyz", "spm_particle_2_traj.xyz"],
                            )?),
                        };
                        files.push(i, &frame_text(&sys, &comment))?;
                    }
                }
                write_table(
                    &run.dir.join("spm_cycles.csv"),
                    "cycle,energy_1,energy_2,max_force_1,max_force_2,spring_length",
                    &rows,
                )?;
            }
            if run.figures {
                progress_figure(
                    run.dir,
                    "spm_energies",
                    "Spring pair: energies after the drift",
                    "Energy (Hartree)",
                    &[("particle 1", &e1), ("particle 2", &e2)],
                )?;
            }
        }
        flow = hook(&r)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    if let Some(files) = particles {
        // One trajectory at the end: the first particle in order, then the second backwards.
        files.join(&run.dir.join("spm_trajectory.xyz"), false, &|_, old| {
            old.to_string()
        })?;
    }
    run.log
        .info(&format!("Spring pair method: {evaluations} evaluations."));
    let end = machine.end();
    Ok(PointOutcome {
        converged: end == Some(SpringPairEnd::Converged),
        end: end.map(|e| match e {
            SpringPairEnd::Converged => {
                "one particle of the pair meets the force thresholds".to_string()
            }
            SpringPairEnd::CycleLimit => "the cycles are used up".to_string(),
        }),
        interrupted: stop.stopped() && flow == Flow::Continue,
        iterations: machine.cycles(),
        transition_state: machine
            .saddle_estimate()
            .map(|(_, x)| (x.clone(), last_energy)),
    })
}

/// Run the two-point scaled hypersphere search (the centre in the state of the first image,
/// the walk in that of the second, as MultiOptPy).
pub fn run_two_point_shs(
    machine: &mut TwoPointShs,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&crate::path::twopshs::WalkPoint) -> MoptResult<Flow>,
) -> MoptResult<PointOutcome> {
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let mut flow = Flow::Continue;
    let mut evaluations = 0;
    let mut energies = Vec::new();
    let mut setup = true;
    loop {
        let asked = match machine.ask() {
            SearchAction::Evaluate(p) => p,
            SearchAction::Finished => break,
        };
        evaluations += asked.len();
        let first = setup;
        let image_of = move |k: usize| if first && k == 0 { 0 } else { 1 };
        let values = evaluate(surface, run, &asked, &image_of, machine.path().len())?;
        setup = false;
        let Some(point) = machine.tell(values)? else {
            continue;
        };
        if point.sphere == 0 {
            if let Some(c) = machine.coordinates() {
                run.log.info(&format!(
                    "Scaled coordinates of the first structure: {} modes, eigenvalues {:.4e} .. \
                     {:.4e}; the second structure on the sphere of radius {:.5}",
                    c.dimension(),
                    c.eigenvalues().first().copied().unwrap_or(0.0),
                    c.stiffest(),
                    point.radius
                ));
                if run.save_steps {
                    let d = col_norm(&(&point.x - c.origin())) * Units::BOHR_TO_ANGSTROM;
                    std::fs::write(
                        run.dir.join("direction_info.csv"),
                        format!("direction,distance_between_sp1_sp2\nSP_1_to_SP_2,{d:.6}\n"),
                    )?;
                }
            }
        } else {
            run.log.info(&format!(
                "Sphere {}: radius {:.5}, energy {:.10}, {} evaluations, gradient within the \
                 sphere (RMS) {:.2e}",
                point.sphere, point.radius, point.energy, point.evaluations, point.tangential_rms
            ));
        }
        log_structures(
            surface,
            run.log,
            &[(
                format!(
                    "Point on sphere {} (radius {:.5})",
                    point.sphere, point.radius
                ),
                &point.x,
                1,
            )],
        );
        energies.push(point.energy);
        if run.save_steps {
            write_walk(machine, surface, run)?;
        }
        if run.figures {
            progress_figure(
                run.dir,
                "twopshs_energies",
                "2PSHS: energy minima on the spheres",
                "Energy (Hartree)",
                &[("energy", &energies)],
            )?;
        }
        flow = hook(&point)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    run.log.info(&format!("2PSHS: {evaluations} evaluations."));
    let end = machine.end();
    let ts = machine
        .transition_state_estimate()
        .map(|p| (p.x.clone(), p.energy));
    Ok(PointOutcome {
        converged: end.is_some() && ts.is_some(),
        end: end.map(|e| match e {
            TwoPshsEnd::ReachedCenter => {
                "the sphere has shrunk onto the first structure".to_string()
            }
            TwoPshsEnd::SphereLimit => "the spheres are used up".to_string(),
            TwoPshsEnd::LeftConstraints => "the spheres can no longer be followed on the \
                 constraints (the estimate is the highest point so far)"
                .to_string(),
        }),
        interrupted: stop.stopped() && flow == Flow::Continue,
        iterations: machine.path().len(),
        transition_state: ts,
    })
}

/// MultiOptPy's `ADD_0_trajectory.xyz` and `ADD_0_distances.csv` of the walk so far.
fn write_walk(
    machine: &TwoPointShs,
    surface: &dyn SearchSurface,
    run: &SearchRun,
) -> MoptResult<()> {
    let origin = machine.coordinates().map(|c| c.origin().clone());
    let mut frames = Vec::new();
    let mut rows = Vec::new();
    for p in machine.path() {
        let distance = origin.as_ref().map_or(f64::NAN, |o| {
            let atoms = (p.x.nrows() / 3).max(1) as f64;
            (col_norm(&(&p.x - o)).powi(2) / atoms).sqrt() * Units::BOHR_TO_ANGSTROM
        });
        rows.push(vec![p.sphere as f64, p.radius, distance, p.energy]);
        let comment = format!(
            "ADD_0 Iteration {} Radius {:.4} Distance_to_SP1 {distance:.6}",
            p.sphere, p.radius
        );
        if let Some(f) = frame(surface, &p.x, 1, comment) {
            frames.push(f);
        }
    }
    write_table(
        &run.dir.join("ADD_0_distances.csv"),
        "iteration,radius,distance_to_sp1,energy",
        &rows,
    )?;
    if frames.is_empty() {
        let points: Vec<(f64, &Col<f64>)> =
            machine.path().iter().map(|p| (p.energy, &p.x)).collect();
        write_points(
            &run.dir.join("twopshs_path.csv"),
            "energy,variables",
            &points,
        )
    } else {
        write_frames(&run.dir.join("ADD_0_trajectory.xyz"), &frames)
    }
}

/// Run the growing Newton trajectory.
pub fn run_newton_trajectory(
    machine: &mut GrowingNewtonTrajectory,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&crate::path::newton_trajectory::NtReport) -> MoptResult<Flow>,
) -> MoptResult<PointOutcome> {
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let mut flow = Flow::Continue;
    let mut evaluations = 0;
    loop {
        let asked = match machine.ask() {
            SearchAction::Evaluate(p) => p,
            SearchAction::Finished => break,
        };
        evaluations += asked.len();
        let values = evaluate(surface, run, &asked, &|_| 0, machine.iterations())?;
        let r = machine.tell(values)?;
        run.log.debug(&format!(
            "Corrector: energy {:.10}, RMS of the projected gradient {:.3e}",
            r.energy, r.projected_rms
        ));
        if let Some(p) = &r.added {
            run.log.info(&format!(
                "Newton trajectory point {}: energy {:.10}, RMS of the projected gradient {:.3e}, \
                 {} corrector evaluations{}{}",
                machine.points().len() - 1,
                p.energy,
                p.projected_rms,
                p.corrector_iterations,
                r.passed
                    .map_or_else(String::new, |what| format!("; passed a {what}")),
                if r.direction_updated {
                    "; new direction"
                } else {
                    ""
                }
            ));
            log_structures(
                surface,
                run.log,
                &[(
                    format!("Newton trajectory point {}", machine.points().len() - 1),
                    &p.x,
                    0,
                )],
            );
            if run.save_steps {
                write_trajectory(machine, surface, run)?;
            }
        }
        flow = hook(&r)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    if run.save_steps {
        let energies: Vec<f64> = machine
            .points()
            .iter()
            .map(|p| p.energy * Units::HARTREE_TO_KCAL_MOL)
            .collect();
        let gradients: Vec<f64> = machine
            .points()
            .iter()
            .map(|p| col_norm(&p.gradient))
            .collect();
        single_plot(
            run.dir,
            "nt_energy",
            "energy",
            "energy [kcal/mol]",
            &energies,
        )?;
        single_plot(
            run.dir,
            "nt_gradient",
            "gradient",
            "grad (RMS) [a.u.]",
            &gradients,
        )?;
        write_indices(run.dir, "approx_TS", &local_extrema(&energies, true))?;
        write_indices(run.dir, "approx_EQ", &local_extrema(&energies, false))?;
        write_indices(run.dir, "local_min_grad", &local_extrema(&gradients, false))?;
    }
    run.log
        .info(&format!("Newton trajectory: {evaluations} evaluations."));
    let end = machine.end();
    Ok(PointOutcome {
        converged: end == Some(NtEnd::PassedTransitionState),
        end: end.map(|e| match e {
            NtEnd::PassedTransitionState => "the trajectory passed a transition state".to_string(),
            NtEnd::IterationLimit => "the iterations are used up".to_string(),
        }),
        interrupted: stop.stopped() && flow == Flow::Continue,
        iterations: machine.iterations(),
        transition_state: machine
            .transition_state_estimate()
            .map(|p| (p.x.clone(), p.energy)),
    })
}

fn write_trajectory(
    machine: &GrowingNewtonTrajectory,
    surface: &dyn SearchSurface,
    run: &SearchRun,
) -> MoptResult<()> {
    let points = machine.points();
    let rows: Vec<Vec<f64>> = points
        .iter()
        .enumerate()
        .map(|(k, p)| vec![k as f64, p.energy, p.projected_rms])
        .collect();
    write_table(
        &run.dir.join("nt_points.csv"),
        "point,energy,projected_rms",
        &rows,
    )?;
    let frames: Vec<(MolecularSystem, String)> = points
        .iter()
        .enumerate()
        .filter_map(|(k, p)| {
            frame(
                surface,
                &p.x,
                0,
                format!("Newton trajectory point {k} energy {:.10}", p.energy),
            )
        })
        .collect();
    if !frames.is_empty() {
        write_frames(&run.dir.join("newton_trajectory.xyz"), &frames)?;
    }
    if run.figures {
        let e: Vec<f64> = points.iter().map(|p| p.energy).collect();
        progress_figure(
            run.dir,
            "nt_energies",
            "Newton trajectory: energies of its points",
            "Energy (Hartree)",
            &[("energy", &e)],
        )?;
    }
    Ok(())
}
