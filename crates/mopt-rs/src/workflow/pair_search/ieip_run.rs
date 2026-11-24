//! The driver of the improved elastic image pair method: the state machine
//! [`ElasticImagePair`] on a [`SearchSurface`], with MultiOptPy's report of every iteration
//! and the two images in Cartesian coordinates, its files, a figure of the energies of the
//! images drawn anew after every iteration, the `STOP` file, and hooks after every
//! iteration. Every iteration appends each image to a trajectory of its own
//! (`<stem>_image_1_traj.xyz`, `<stem>_image_2_traj.xyz`); at the end the two become
//! MultiOptPy's `<stem>_traj.xyz`.
//!
//! MultiOptPy: `OtherMethod/elastic_image_pair.py:ElasticImagePair` (`iteration`,
//! `print_info`, the files at its end).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::files::{
    image_frame, local_extrema, progress_figure, single_plot, write_indices, write_points,
};
use super::surface::{evaluate_all, SearchSurface};
use super::{log_notes, log_structures, SearchRun};
use crate::core_infra::log::python_float;
use crate::domain::Units;
use crate::error::MoptResult;
use crate::math::linalg::col_norm;
use crate::path::ieip::{ElasticImagePair, IeipAction, IeipEnd, IeipReport};
use crate::path::points::PointRequest;
use crate::workflow::chain::Flow;
use crate::workflow::observers::StopFile;
use crate::workflow::trajectories::PairTrajectories;

/// How an iEIP run went.
#[derive(Debug, Clone)]
pub struct IeipOutcome {
    /// Images evaluated on both inward walks, ordered from the first end to the second.
    pub path: Vec<(Col<f64>, f64)>,
    /// Iterations made.
    pub iterations: usize,
    /// Why it ended (`None`: stopped by a hook or a `STOP` file).
    pub end: Option<IeipEnd>,
    /// Whether a `STOP` file interrupted it.
    pub interrupted: bool,
    /// The images after the last step.
    pub images: [Col<f64>; 2],
    /// Energies (with bias potentials) of the images last evaluated.
    pub energies: [f64; 2],
    /// The midpoint of the last images: the estimate of the transition state.
    pub transition_state: Col<f64>,
}

/// What MultiOptPy plots and lists at the end, iteration by iteration.
#[derive(Default)]
struct Lists {
    energy: [Vec<f64>; 2],
    gradient: [Vec<f64>; 2],
    bias_energy: [Vec<f64>; 2],
    bias_gradient: [Vec<f64>; 2],
    separation: Vec<f64>,
    trajectories: Option<PairTrajectories>,
    points: [Vec<(f64, Col<f64>)>; 2],
}

/// MultiOptPy's `print_info`.
fn print_info(report: &IeipReport, run: &SearchRun) {
    let Some(step) = &report.step else {
        return;
    };
    let v = &report.values;
    let p = &step.parts;
    let pair = |a: f64, b: f64| format!("{}   {}", python_float(a), python_float(b));
    let n = |c: &Col<f64>| col_norm(c);
    let lines = [
        "[[opt information]]".to_string(),
        "                                                image_1               image_2".to_string(),
        format!(
            "energy  (normal)                       : {}",
            pair(v[0].energy, v[1].energy)
        ),
        format!(
            "energy  (bias)                         : {}",
            pair(v[0].biased_energy, v[1].biased_energy)
        ),
        format!(
            "gradient  (normal, RMS)                : {}",
            pair(n(&v[0].gradient), n(&v[1].gradient))
        ),
        format!(
            "gradient  (bias, RMS)                  : {}",
            pair(n(&v[0].biased_gradient), n(&v[1].biased_gradient))
        ),
        format!(
            "perpendicular_force (RMS)              : {}",
            pair(n(&p.perpendicular[0]), n(&p.perpendicular[1]))
        ),
        format!(
            "energy_difference_dependent_force (RMS): {}",
            pair(n(&p.energy_difference), n(&p.energy_difference))
        ),
        format!(
            "distance_dependent_force (RMS)         : {}",
            python_float(n(&p.attraction))
        ),
        format!(
            "Total_displacement (RMS)               : {}",
            pair(n(&p.total[0]), n(&p.total[1]))
        ),
        format!(
            "Image_distance                         : {}",
            python_float(step.separation)
        ),
        "[[threshold]]".to_string(),
        format!(
            "Image_distance                         :  {}",
            python_float(crate::path::ieip::SEPARATION_TOLERANCE)
        ),
    ];
    run.log.info(&lines.join("\n"));
    if let Some(m) = &step.micro {
        run.log.info(&format!(
            "Relaxation of the images (-mi): {} step(s), forces {:.3e} {:.3e}{}",
            m.steps,
            m.forces[0],
            m.forces[1],
            if m.converged {
                " (below the tolerance)"
            } else {
                ""
            }
        ));
    }
}

/// The names of the trajectories of the two images while the run goes on.
fn image_trajectories(run: &SearchRun) -> [String; 2] {
    [
        format!("{}_image_1_traj.xyz", run.stems[0]),
        format!("{}_image_2_traj.xyz", run.stems[1]),
    ]
}

/// Append the structures of the images of iteration `k` to their trajectories (or keep the
/// points of plain variables).
fn write_images(
    surface: &dyn SearchSurface,
    run: &SearchRun,
    k: usize,
    images: [&Col<f64>; 2],
    lists: &mut Lists,
    energies: [f64; 2],
) -> MoptResult<()> {
    if !run.save_steps {
        return Ok(());
    }
    for i in 0..2 {
        match surface.structure(images[i], i) {
            Some(sys) => {
                let trajectories = match &mut lists.trajectories {
                    Some(t) => t,
                    slot => {
                        let names = image_trajectories(run);
                        slot.insert(PairTrajectories::start(run.dir, [&names[0], &names[1]])?)
                    }
                };
                trajectories.push(i, &image_frame(&sys, &format!("Frame {k}")))?;
            }
            None => lists.points[i].push((energies[i], images[i].clone())),
        }
    }
    Ok(())
}

/// The files of MultiOptPy at the end of a run.
fn finish(run: &SearchRun, lists: &mut Lists, end: Option<IeipEnd>) -> MoptResult<()> {
    if !run.save_steps {
        return Ok(());
    }
    let joined =
        |l: &[Vec<f64>; 2]| -> Vec<f64> { l[0].iter().chain(l[1].iter().rev()).copied().collect() };
    let energy = joined(&lists.energy);
    let gradient = joined(&lists.gradient);
    single_plot(run.dir, "energy", "energy", "energy [kcal/mol]", &energy)?;
    single_plot(
        run.dir,
        "gradient",
        "gradient",
        "grad (RMS) [a.u.]",
        &gradient,
    )?;
    single_plot(
        run.dir,
        "energy",
        "bias_energy",
        "energy [kcal/mol]",
        &joined(&lists.bias_energy),
    )?;
    single_plot(
        run.dir,
        "gradient",
        "bias_gradient",
        "grad (RMS) [a.u.]",
        &joined(&lists.bias_gradient),
    )?;
    write_indices(run.dir, "approx_TS", &local_extrema(&energy, true))?;
    write_indices(run.dir, "approx_EQ", &local_extrema(&energy, false))?;
    write_indices(run.dir, "local_min_grad", &local_extrema(&gradient, false))?;
    match lists.trajectories.take() {
        None => {
            for i in 0..2 {
                let points: Vec<(f64, &Col<f64>)> =
                    lists.points[i].iter().map(|(e, x)| (*e, x)).collect();
                write_points(
                    &run.dir.join(format!("{}_images.csv", run.stems[i])),
                    "energy,variables",
                    &points,
                )?;
            }
        }
        // MultiOptPy's `make_traj_file_for_DM`: the first image in order, then the second
        // backwards, without the two starting structures, as `Frame m`.
        Some(trajectories) => {
            // The lattice of a periodic structure stays before the number of the frame.
            trajectories.join(
                &run.dir.join(format!("{}_traj.xyz", run.stems[0])),
                true,
                &|m, old| match old.rfind("Frame ") {
                    Some(at) => format!("{}Frame {m}", &old[..at]),
                    None => format!("Frame {m}"),
                },
            )?;
        }
    }
    if end == Some(IeipEnd::IterationLimit) {
        std::fs::write(
            run.dir.join("not_converged.txt"),
            "Reached maximum number of iterations. This is not converged.",
        )?;
        run.log
            .info("Reached maximum number of iterations. This is not converged.");
    }
    Ok(())
}

/// Run `machine` on `surface`; `hook` sees every report after the log and the files and can
/// end the run.
pub fn run_ieip(
    machine: &mut ElasticImagePair,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&IeipReport) -> MoptResult<Flow>,
) -> MoptResult<IeipOutcome> {
    let mut lists = Lists::default();
    let mut paths = super::ImagePaths::default();
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let mut energies = [f64::NAN; 2];
    let mut table: Vec<Vec<f64>> = Vec::new();
    let mut flow = Flow::Continue;
    {
        let [a, b] = machine.images();
        let (a, b) = (a.clone(), b.clone());
        write_images(surface, run, 0, [&a, &b], &mut lists, [f64::NAN; 2])?;
    }
    while let IeipAction::Evaluate([a, b]) = machine.ask() {
        let iteration = machine.iteration();
        surface.begin_iteration(iteration);
        let requests = [PointRequest::gradient(a), PointRequest::gradient(b)];
        let values = evaluate_all(surface, &requests, &|k| k)?;
        log_notes(surface, run.log);
        let report = machine.tell([values[0].image_value(), values[1].image_value()])?;
        energies = [values[0].biased_energy, values[1].biased_energy];
        if let Some(step) = report.micro_step {
            if step % 10 == 0 {
                run.log.info(&format!(
                    "# Microiteration {step}\nEnergy 1                 : {}\nEnergy 2                 : {}",
                    python_float(values[0].energy),
                    python_float(values[1].energy)
                ));
            }
        }
        if let Some(step) = &report.step {
            paths.push(
                &report.images,
                [
                    report.values[0].biased_energy,
                    report.values[1].biased_energy,
                ],
            );
            run.log.info(&format!("# ITR. {iteration}"));
            log_structures(
                surface,
                run.log,
                &[
                    ("Image 1".to_string(), &report.images[0], 0),
                    ("Image 2".to_string(), &report.images[1], 1),
                ],
            );
            print_info(&report, run);
            let kcal = Units::HARTREE_TO_KCAL_MOL;
            for i in 0..2 {
                let v = &report.values[i];
                lists.energy[i].push(v.energy * kcal);
                lists.bias_energy[i].push(v.biased_energy * kcal);
                lists.gradient[i].push(col_norm(&v.gradient));
                lists.bias_gradient[i].push(col_norm(&v.biased_gradient));
            }
            lists.separation.push(step.separation);
            let [na, nb] = &step.new_images;
            write_images(surface, run, iteration + 1, [na, nb], &mut lists, energies)?;
            table.push(vec![
                iteration as f64,
                report.values[0].energy,
                report.values[1].energy,
                report.values[0].biased_energy,
                report.values[1].biased_energy,
                step.separation,
                step.beta,
            ]);
            if run.save_steps {
                super::files::write_table(
                &run.dir.join("ieip_energies.csv"),
                "iteration,energy_1,energy_2,biased_energy_1,biased_energy_2,separation_after,beta",
                &table,
            )?;
            }
            if run.figures {
                let reference = lists.energy[0][0];
                let e1: Vec<f64> = lists.energy[0].iter().map(|e| e - reference).collect();
                let e2: Vec<f64> = lists.energy[1].iter().map(|e| e - reference).collect();
                progress_figure(
                    run.dir,
                    "ieip_energies",
                    "Energies of the images",
                    "E - E(first image at the start) (kcal/mol)",
                    &[("image 1", &e1), ("image 2", &e2)],
                )?;
                progress_figure(
                    run.dir,
                    "ieip_separation",
                    "Separation of the images after the step",
                    "Bohr",
                    &[("separation", &lists.separation)],
                )?;
            }
            if report.end == Some(IeipEnd::Converged) {
                run.log.info("Converged!!!");
            }
        }
        flow = hook(&report)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    finish(run, &mut lists, machine.end())?;
    let [a, b] = machine.images();
    Ok(IeipOutcome {
        path: paths.join(),
        iterations: machine.iteration(),
        end: machine.end(),
        interrupted: stop.stopped() && flow == Flow::Continue,
        images: [a.clone(), b.clone()],
        energies,
        transition_state: machine.transition_state_estimate(),
    })
}
