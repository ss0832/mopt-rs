//! The driver of the anharmonic downward distortion following ([`Addf`]) for `ieipmain
//! -addf`: the log with the structure of every point of a path in Cartesian coordinates,
//! MultiOptPy's files (`add_energy_list.csv`, `ADD_<k>_trajectory.xyz`, `TSpoints.xyz`; the
//! trajectories hold every point, so the directories `optimized_structures/ADD_<k>/` with a
//! file per point are not written), a table of the paths, a figure of their energies drawn
//! anew after every sphere, the `STOP` file, and a hook after every evaluation.
//!
//! MultiOptPy: `OtherMethod/addf.py:ADDFlikeMethod` (`run`, `detect_add`, `add_following`,
//! `save_optimized_structure`, `create_separate_xyz_files`, `write_ts_points`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;

use faer::Col;

use super::files::{progress_figure, write_frames, write_table};
use super::surface::{evaluate_all, SearchSurface};
use super::{log_notes, log_structures, SearchRun};
use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::path::points::SearchAction;
use crate::workflow::chain::Flow;
use crate::workflow::explore::addf::{Addf, AddfEnd, AddfEvent, PathState};
use crate::workflow::observers::StopFile;

/// How an ADDF run went.
#[derive(Debug, Clone)]
pub struct AddfOutcome {
    /// Why it ended (`None`: stopped by a hook or a `STOP` file).
    pub end: Option<AddfEnd>,
    /// Whether a `STOP` file interrupted it.
    pub interrupted: bool,
    /// Points evaluated.
    pub evaluations: usize,
    /// The estimates of transition states: path, point, energy.
    pub transition_states: Vec<(usize, Col<f64>, f64)>,
}

fn log_event(event: &AddfEvent, run: &SearchRun) {
    match event {
        AddfEvent::Started {
            energy,
            eigenvalues,
            first_radius,
            radius_step,
        } => run.log.info(&format!(
            "Initial structure is EQ (energy {energy:.10}). Scaled coordinates: {} modes, \
             eigenvalues {:.4e} .. {:.4e}; first sphere of radius {first_radius:.5}, the radius \
             grows by {radius_step:.5} per sphere.",
            eigenvalues.len(),
            eigenvalues.first().copied().unwrap_or(0.0),
            eigenvalues.last().copied().unwrap_or(0.0)
        )),
        AddfEvent::Scanned {
            candidates,
            followed,
        } => {
            let mut text = format!(
                "### Checking ADD energy. ### {} directions; the {followed} largest ADDs are \
                 followed:",
                candidates.len()
            );
            for (k, c) in candidates.iter().take(*followed).enumerate() {
                let _ = write!(
                    text,
                    "\nADD {k}: mode {} ({:+}), eigenvalue {:.4e}, ADD {:.8}",
                    c.mode, c.sign, c.eigenvalue, c.add
                );
            }
            run.log.info(&text);
        }
        AddfEvent::Point { path, point } => run.log.info(&format!(
            "ADD {path} on sphere {} (radius {:.4}): energy {:.10}, ADD {:.6}, with the \
             elimination {:.6}, {} evaluations",
            point.sphere, point.radius, point.energy, point.add, point.add_ioe, point.evaluations
        )),
        AddfEvent::Merged { path, into } => {
            run.log
                .info(&format!("ADD {path} ran into ADD {into} and is dropped."));
        }
        AddfEvent::Dissociated { path } => run.log.info(&format!(
            "ADD {path} leads to a dissociation channel (the structure fell apart); it is \
             followed no further."
        )),
        AddfEvent::Undefined { path } => run.log.info(&format!(
            "ADD {path} reached a point where the surface is not defined (an energy or a \
             gradient that is not finite); it is followed no further."
        )),
        AddfEvent::TransitionState { path, point } => run.log.info(&format!(
            "ADD {path} passed a maximum of its energy: the point on sphere {} (energy {:.10}) \
             estimates a transition state.",
            point.sphere, point.energy
        )),
        AddfEvent::SphereDone { sphere, active } => {
            run.log.info(&format!(
                "Sphere {sphere} done; {active} path(s) followed further."
            ));
        }
    }
}

fn structure(surface: &dyn SearchSurface, x: &Col<f64>) -> Option<MolecularSystem> {
    surface.structure(x, 0)
}

/// The files of the paths so far.
fn write_paths(machine: &Addf, surface: &dyn SearchSurface, run: &SearchRun) -> MoptResult<()> {
    if !run.save_steps {
        return Ok(());
    }
    let mut rows = Vec::new();
    for path in machine.paths() {
        let mut frames = Vec::new();
        for p in &path.points {
            rows.push(vec![
                path.id as f64,
                p.sphere as f64,
                p.radius,
                p.energy,
                p.add,
                p.add_ioe,
                p.evaluations as f64,
            ]);
            let comment = format!(
                "ADD_{} Sphere {} Radius {:.4} Energy {:.6}",
                path.id, p.sphere, p.radius, p.add
            );
            if let Some(sys) = structure(surface, &p.x) {
                frames.push((sys, comment));
            }
        }
        if !frames.is_empty() {
            write_frames(
                &run.dir.join(format!("ADD_{}_trajectory.xyz", path.id)),
                &frames,
            )?;
        }
    }
    write_table(
        &run.dir.join("addf_paths.csv"),
        "path,sphere,radius,energy,add,add_with_elimination,evaluations",
        &rows,
    )?;
    if run.figures {
        let series: Vec<(String, Vec<f64>)> = machine
            .paths()
            .iter()
            .take(4)
            .map(|p| {
                (
                    format!("ADD {}", p.id),
                    p.points
                        .iter()
                        .map(|q| q.energy - machine.minimum_energy())
                        .collect(),
                )
            })
            .collect();
        let refs: Vec<(&str, &[f64])> = series
            .iter()
            .map(|(l, v)| (l.as_str(), v.as_slice()))
            .collect();
        progress_figure(
            run.dir,
            "addf_energies",
            "ADDF: energies along the paths (the first four)",
            "E - E(minimum) (Hartree)",
            &refs,
        )?;
    }
    Ok(())
}

/// Run `machine` on `surface`.
pub fn run_addf(
    machine: &mut Addf,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&[AddfEvent]) -> MoptResult<Flow>,
) -> MoptResult<AddfOutcome> {
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let mut flow = Flow::Continue;
    let mut evaluations = 0;
    let mut calls = 0;
    loop {
        let asked = match machine.ask() {
            SearchAction::Evaluate(p) => p,
            SearchAction::Finished => break,
        };
        evaluations += asked.len();
        surface.begin_iteration(calls);
        calls += 1;
        let values = evaluate_all(surface, &asked, &|_| 0)?;
        log_notes(surface, run.log);
        let told = values
            .iter()
            .zip(&asked)
            .map(|(v, p)| v.point_value(p))
            .collect();
        let events = machine.tell(told)?;
        let mut changed = false;
        for event in &events {
            log_event(event, run);
            if let AddfEvent::Point { path, point } = event {
                log_structures(
                    surface,
                    run.log,
                    &[(
                        format!("ADD {path} on sphere {}", point.sphere),
                        &point.x,
                        0,
                    )],
                );
            }
            if run.save_steps {
                if let AddfEvent::Scanned { candidates, .. } = event {
                    let mut text = String::from(
                        "index_of_principal_axis,eigenvalue,direction,add_energy,abs_add_energy\n",
                    );
                    for c in candidates {
                        let _ = writeln!(
                            text,
                            "{},{},{},{},{}",
                            c.mode,
                            c.eigenvalue,
                            c.sign,
                            c.add,
                            c.add.abs()
                        );
                    }
                    std::fs::write(run.dir.join("add_energy_list.csv"), text)?;
                }
            }
            changed |= matches!(
                event,
                AddfEvent::SphereDone { .. } | AddfEvent::Scanned { .. }
            );
        }
        if changed {
            write_paths(machine, surface, run)?;
        }
        flow = hook(&events)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    write_paths(machine, surface, run)?;
    let transition_states: Vec<(usize, Col<f64>, f64)> = machine
        .transition_states()
        .into_iter()
        .map(|(k, p)| (k, p.x.clone(), p.energy))
        .collect();
    let frames: Vec<(MolecularSystem, String)> = transition_states
        .iter()
        .enumerate()
        .filter_map(|(i, (_, x, _))| {
            structure(surface, x).map(|s| (s, format!("TS candidate {}", i + 1)))
        })
        .collect();
    if run.save_steps && !frames.is_empty() {
        write_frames(&run.dir.join("TSpoints.xyz"), &frames)?;
        run.log.info(&format!(
            "Found {} potential transition states.",
            frames.len()
        ));
    }
    let unfinished = machine
        .paths()
        .iter()
        .filter(|p| p.state == PathState::SphereLimit)
        .count();
    if unfinished > 0 {
        run.log.info(&format!(
            "{unfinished} path(s) did not pass a transition state within the spheres (-addf_num)."
        ));
    }
    run.log.info(&format!(
        "### ADD Following is done. ### {evaluations} evaluations."
    ));
    Ok(AddfOutcome {
        end: machine.end(),
        interrupted: stop.stopped() && flow == Flow::Continue,
        evaluations,
        transition_states,
    })
}
