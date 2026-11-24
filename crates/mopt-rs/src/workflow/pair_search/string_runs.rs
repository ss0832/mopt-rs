//! The drivers of the string methods of `ieipmain` and of DS-AFIR: the growing string
//! (`-gsm`, `-se_gsm`), the freezing string (`-fsm`), and the double-sphere AFIR (`-dsafir`).
//! Each writes a table of its iterations, the path at the end (`<name>_path.xyz` for
//! structures, `<name>_path.csv` for plain variables) with its energies, and a figure of its
//! progress drawn anew after every iteration. While it runs, each shows the structures it
//! moved in Cartesian coordinates and appends them to a trajectory per side
//! (`<name>_side_1_traj.xyz`, `<name>_side_2_traj.xyz`: the reactant side and the product
//! side of a string, the two ends of DS-AFIR), so that an interrupted run leaves what it
//! reached; when the path is written at the end, the two files are removed.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::files::{frame_text, progress_figure, write_frames, write_points, write_table};
use super::point_runs::{evaluate, PointOutcome};
use super::surface::SearchSurface;
use super::{log_structures, SearchRun};
use crate::error::MoptResult;
use crate::path::dsafir::{DoubleSphereAfir, DsAfirEnd, DsAfirReport};
use crate::path::fsm::{FreezingString, FsmEnd, FsmReport};
use crate::path::gsm::{GrowingString, GsmEnd, GsmPhase, GsmReport};
use crate::path::lmm::{LmmEnd, LmmReport, LocalMinimax};
use crate::path::points::SearchAction;
use crate::workflow::chain::Flow;
use crate::workflow::observers::StopFile;
use crate::workflow::trajectories::PairTrajectories;

/// The trajectories of the two sides of a search while it runs: a point is appended to its
/// side when it is not the one appended there last.
struct Sides {
    name: &'static str,
    files: Option<PairTrajectories>,
    last: [Option<Col<f64>>; 2],
}

impl Sides {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            files: None,
            last: [None, None],
        }
    }

    /// Append `x` (energy `e`) to side `side` unless it is the point appended there last;
    /// plain variables have no such files. Returns whether the point was new.
    fn push(
        &mut self,
        surface: &dyn SearchSurface,
        run: &SearchRun,
        side: usize,
        x: &Col<f64>,
        e: f64,
    ) -> MoptResult<bool> {
        if self.last[side].as_ref() == Some(x) {
            return Ok(false);
        }
        self.last[side] = Some(x.clone());
        let Some(sys) = surface.structure(x, side) else {
            return Ok(true);
        };
        let files = match &mut self.files {
            Some(f) => f,
            slot => {
                let names = [
                    format!("{}_side_1_traj.xyz", self.name),
                    format!("{}_side_2_traj.xyz", self.name),
                ];
                slot.insert(PairTrajectories::start(run.dir, [&names[0], &names[1]])?)
            }
        };
        let k = files.frames()[side];
        let comment = format!("{} side {} point {k} energy {e:.12}", self.name, side + 1);
        files.push(side, &frame_text(&sys, &comment))?;
        Ok(true)
    }

    /// The path of the run is written: the files of the sides are removed.
    fn finish(self) -> MoptResult<()> {
        match self.files {
            Some(files) => files.remove(),
            None => Ok(()),
        }
    }
}

/// The path of a run: `<name>_path.xyz` (structures, with the energies in the comments) or
/// `<name>_path.csv`.
pub fn write_path(
    surface: &dyn SearchSurface,
    run: &SearchRun,
    name: &str,
    path: &[(Col<f64>, f64)],
) -> MoptResult<()> {
    let frames: Vec<_> = path
        .iter()
        .enumerate()
        .filter_map(|(k, (x, e))| {
            surface
                .structure(x, 0)
                .map(|s| (s, format!("{name} node {k} energy {e:.12}")))
        })
        .collect();
    if frames.len() == path.len() && !frames.is_empty() {
        write_frames(&run.dir.join(format!("{name}_path.xyz")), &frames)
    } else {
        let points: Vec<(f64, &Col<f64>)> = path.iter().map(|(x, e)| (*e, x)).collect();
        write_points(
            &run.dir.join(format!("{name}_path.csv")),
            "energy,variables",
            &points,
        )
    }
}

fn phase_name(p: GsmPhase) -> &'static str {
    match p {
        GsmPhase::Ends => "ends",
        GsmPhase::Growth => "growth",
        GsmPhase::LastNode => "last node",
        GsmPhase::Optimization => "optimization",
    }
}

/// Run a growing string (DE-GSM or SE-GSM).
pub fn run_gsm(
    machine: &mut GrowingString,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&GsmReport) -> MoptResult<Flow>,
) -> MoptResult<PointOutcome> {
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let (mut rows, mut highest, mut largest) = (Vec::new(), Vec::new(), Vec::new());
    let mut sides = Sides::new("gsm");
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
        let last = r.nodes.len().saturating_sub(1);
        // The nodes at the two frontiers while the string grows (the last node of a string
        // with one end), the climbing or the highest node once it is whole.
        let mut shown: Vec<(String, &Col<f64>, usize)> = Vec::new();
        if r.phase == GsmPhase::Optimization {
            let top = r
                .climbing
                .or_else(|| (1..last).max_by(|a, b| r.nodes[*a].1.total_cmp(&r.nodes[*b].1)));
            if let Some(k) = top {
                shown.push((
                    format!("Node {k} (the highest of the string)"),
                    &r.nodes[k].0,
                    0,
                ));
            }
            // The string as it is now, so that an interrupted run leaves it.
            write_path(surface, run, "gsm", &r.nodes)?;
        } else {
            let frontiers: Vec<(usize, usize)> = match r.gap {
                Some(g) if g >= 1 && g <= last => vec![(0, g - 1), (1, g)],
                _ => vec![(0, last)],
            };
            for (side, k) in frontiers {
                let (x, e) = &r.nodes[k];
                if sides.push(surface, run, side, x, *e)? {
                    shown.push((
                        format!(
                            "Node {k} (frontier of the {} side)",
                            if side == 0 { "reactant" } else { "product" }
                        ),
                        x,
                        0,
                    ));
                }
            }
        }
        let top = r
            .nodes
            .iter()
            .enumerate()
            .filter(|(k, _)| *k > 0 && *k < last)
            .map(|(_, n)| n.1)
            .fold(f64::NEG_INFINITY, f64::max);
        let across = r.across.iter().copied().fold(0.0_f64, f64::max);
        run.log.info(&format!(
            "GSM iteration {} ({}): {} nodes{}, highest node {:.10}, largest RMS gradient \
             across the string {:.3e}{}",
            r.iteration,
            phase_name(r.phase),
            r.nodes.len(),
            r.gap
                .map_or(String::new(), |g| format!(" (gap after node {g})")),
            top,
            across,
            r.climbing
                .map_or(String::new(), |c| format!(", node {c} climbing"))
        ));
        log_structures(surface, run.log, &shown);
        if top.is_finite() {
            highest.push(top);
            largest.push(across);
        }
        rows.push(vec![
            r.iteration as f64,
            r.nodes.len() as f64,
            top,
            across,
            r.climbing.map_or(-1.0, |c| c as f64),
        ]);
        write_table(
            &run.dir.join("gsm_iterations.csv"),
            "iteration,nodes,highest_energy,largest_rms_across,climbing_node",
            &rows,
        )?;
        if run.figures && !highest.is_empty() {
            progress_figure(
                run.dir,
                "gsm_energies",
                "Growing string: the highest node",
                "Energy (Hartree)",
                &[("highest node", &highest)],
            )?;
        }
        flow = hook(&r)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    let path: Vec<(Col<f64>, f64)> = machine
        .nodes()
        .into_iter()
        .filter_map(|(x, e)| e.map(|e| (x, e)))
        .collect();
    write_path(surface, run, "gsm", &path)?;
    sides.finish()?;
    run.log
        .info(&format!("Growing string: {evaluations} evaluations."));
    let end = machine.end();
    Ok(PointOutcome {
        converged: end == Some(GsmEnd::Converged),
        end: end.map(|e| match e {
            GsmEnd::Converged => "the climbing node meets the force thresholds".to_string(),
            GsmEnd::IterationLimit => "the iterations are used up".to_string(),
        }),
        interrupted: stop.stopped() && flow == Flow::Continue,
        iterations: machine.iterations(),
        transition_state: machine.transition_state_estimate(),
    })
}

/// Run a freezing string.
pub fn run_fsm(
    machine: &mut FreezingString,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&FsmReport) -> MoptResult<Flow>,
) -> MoptResult<PointOutcome> {
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let (mut rows, mut frozen) = (Vec::new(), Vec::new());
    let mut sides = Sides::new("fsm");
    let mut flow = Flow::Continue;
    let mut evaluations = 0;
    let mut nodes = 0;
    loop {
        let asked = match machine.ask() {
            SearchAction::Evaluate(p) => p,
            SearchAction::Finished => break,
        };
        evaluations += asked.len();
        let values = evaluate(surface, run, &asked, &|_| 0, nodes)?;
        let r = machine.tell(values)?;
        if r.side.is_none() {
            // The two ends, once they are evaluated.
            let path = machine.path();
            if let (Some(a), Some(b), true) = (path.first(), path.last(), path.len() >= 2) {
                sides.push(surface, run, 0, &a.0, a.1)?;
                sides.push(surface, run, 1, &b.0, b.1)?;
            }
        }
        if r.frozen {
            nodes += 1;
            frozen.push(r.energy);
            let side = r.side.unwrap_or(0).min(1);
            sides.push(surface, run, side, &r.x, r.energy)?;
            run.log.info(&format!(
                "FSM node {nodes} ({} side) frozen: energy {:.10}, interpolation left between \
                 the frontiers {:.4}",
                if r.side == Some(0) {
                    "reactant"
                } else {
                    "product"
                },
                r.energy,
                r.gap
            ));
            log_structures(surface, run.log, &[(format!("FSM node {nodes}"), &r.x, 0)]);
            rows.push(vec![
                nodes as f64,
                r.side.map_or(-1.0, |s| s as f64),
                r.energy,
                r.gap,
            ]);
            write_table(
                &run.dir.join("fsm_nodes.csv"),
                "node,side,energy,gap",
                &rows,
            )?;
            if run.figures {
                progress_figure(
                    run.dir,
                    "fsm_energies",
                    "Freezing string: the frozen nodes",
                    "Energy (Hartree)",
                    &[("frozen node", &frozen)],
                )?;
            }
        } else if r.side.is_none() {
            run.log.info(&format!(
                "FSM: step {:.4} along the interpolation of length {:.4}",
                machine.step_length(),
                r.gap
            ));
        }
        flow = hook(&r)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    write_path(surface, run, "fsm", &machine.path())?;
    sides.finish()?;
    run.log
        .info(&format!("Freezing string: {evaluations} evaluations."));
    let end = machine.end();
    Ok(PointOutcome {
        converged: end == Some(FsmEnd::Joined),
        end: end.map(|e| match e {
            FsmEnd::Joined => "the two sides met".to_string(),
            FsmEnd::Stalled => "the two sides do not approach each other".to_string(),
        }),
        interrupted: stop.stopped() && flow == Flow::Continue,
        iterations: nodes,
        transition_state: machine.transition_state_estimate(),
    })
}

/// Run the local minimax method.
pub fn run_lmm(
    machine: &mut LocalMinimax,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&LmmReport) -> MoptResult<Flow>,
) -> MoptResult<PointOutcome> {
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let (mut rows, mut energies, mut peaks) = (Vec::new(), Vec::new(), Vec::new());
    // Every point evaluated along the rays, written at the end (`lmm_rays_path.xyz`).
    let mut rays = Vec::new();
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
        rays.push((r.x.clone(), r.energy));
        if let Some(p) = &r.peak {
            let max = crate::math::linalg::col_max_abs(&p.gradient);
            run.log.info(&format!(
                "LMM iteration {}: peak at {:.4} along the ray, energy {:.10}, largest \
                 gradient {max:.3e}",
                r.iteration, p.t, p.energy
            ));
            log_structures(
                surface,
                run.log,
                &[(format!("Peak of iteration {}", r.iteration), &p.x, 0)],
            );
            rows.push(vec![r.iteration as f64, p.t, p.energy, max]);
            energies.push(p.energy);
            peaks.push((p.x.clone(), p.energy));
            // The peaks so far, so that an interrupted run leaves them.
            write_path(surface, run, "lmm_peaks", &peaks)?;
            write_table(
                &run.dir.join("lmm_peaks.csv"),
                "iteration,distance,energy,largest_gradient",
                &rows,
            )?;
            if run.figures {
                progress_figure(
                    run.dir,
                    "lmm_energies",
                    "Local minimax: the energy of the peak",
                    "Energy (Hartree)",
                    &[("peak", &energies)],
                )?;
            }
        }
        flow = hook(&r)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    write_path(surface, run, "lmm_peaks", &peaks)?;
    if run.save_steps {
        write_path(surface, run, "lmm_rays", &rays)?;
    }
    run.log
        .info(&format!("Local minimax method: {evaluations} evaluations."));
    let end = machine.end();
    Ok(PointOutcome {
        converged: end == Some(LmmEnd::Converged),
        end: end.map(|e| match e {
            LmmEnd::Converged => "the gradient at the peak meets the force thresholds".to_string(),
            LmmEnd::IterationLimit => "the iterations are used up".to_string(),
            LmmEnd::NoPeak => "a ray has no maximum".to_string(),
        }),
        interrupted: stop.stopped() && flow == Flow::Continue,
        iterations: machine.iterations(),
        transition_state: machine.peak().map(|p| (p.x.clone(), p.energy)),
    })
}

/// Run DS-AFIR.
pub fn run_dsafir(
    machine: &mut DoubleSphereAfir,
    surface: &dyn SearchSurface,
    run: &SearchRun,
    hook: &mut dyn FnMut(&DsAfirReport) -> MoptResult<Flow>,
) -> MoptResult<PointOutcome> {
    let mut stop = StopFile::new(run.dir, run.log.clone());
    let (mut rows, mut first, mut second, mut separation) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut sides = Sides::new("dsafir");
    let mut flow = Flow::Continue;
    let mut evaluations = 0;
    let mut iteration = 0;
    // Two surfaces: each end on its own.
    let two = surface.two_surfaces();
    if two {
        run.log.info(
            "The two structures are in different states: each end of DS-AFIR moves on the \
             surface of its own state, and the point where the ends meet estimates a \
             crossing of the two surfaces.",
        );
    }
    loop {
        let asked = match machine.ask() {
            SearchAction::Evaluate(p) => p,
            SearchAction::Finished => break,
        };
        evaluations += asked.len();
        let ends = machine.pending_ends();
        let image_of = |k: usize| {
            if two {
                ends.get(k).copied().unwrap_or(0)
            } else {
                0
            }
        };
        let values = evaluate(surface, run, &asked, &image_of, iteration)?;
        iteration += 1;
        let r = machine.tell(values)?;
        // No weights where no further step follows (the ends have met).
        let weights = r
            .weights
            .map_or(String::new(), |(x, y)| format!(", X {x:.4e}, Y {y:.4}"));
        run.log.info(&format!(
            "DS-AFIR step {}: end {} moved, energies {:.10} {:.10}, distance {:.4}{weights}",
            r.steps,
            r.moved.map_or(0, |m| m + 1),
            r.energies[0],
            r.energies[1],
            r.separation
        ));
        // The ends that moved (both at the start), on the screen and in their files.
        if let Some(ends) = machine.current_ends() {
            let mut shown: Vec<(String, &Col<f64>, usize)> = Vec::new();
            for (side, (x, e)) in ends.iter().enumerate() {
                if sides.push(surface, run, side, x, *e)? {
                    shown.push((format!("End {}", side + 1), x, side));
                }
            }
            log_structures(surface, run.log, &shown);
        }
        first.push(r.energies[0]);
        second.push(r.energies[1]);
        separation.push(r.separation);
        rows.push(vec![
            r.steps as f64,
            r.moved.map_or(-1.0, |m| m as f64),
            r.energies[0],
            r.energies[1],
            r.separation,
            r.weights.map_or(f64::NAN, |w| w.0),
            r.weights.map_or(f64::NAN, |w| w.1),
        ]);
        write_table(
            &run.dir.join("dsafir_steps.csv"),
            "step,moved_end,energy_1,energy_2,distance,X,Y",
            &rows,
        )?;
        if run.figures && r.steps.is_multiple_of(10) {
            progress_figure(
                run.dir,
                "dsafir_energies",
                "DS-AFIR: the energies of the two ends",
                "Energy (Hartree)",
                &[("end 1", &first), ("end 2", &second)],
            )?;
        }
        if r.end.is_some() || r.steps.is_multiple_of(50) {
            run.log.info(&format!(
                "DS-AFIR: {} steps, distance of the ends {:.4}",
                r.steps, r.separation
            ));
        }
        flow = hook(&r)?;
        if flow == Flow::Stop || stop.check() {
            break;
        }
    }
    write_path(surface, run, "dsafir", &machine.path())?;
    sides.finish()?;
    run.log
        .info(&format!("DS-AFIR: {evaluations} evaluations."));
    let end = machine.end();
    let estimate = if two {
        crossing_estimate(machine, surface, run)?
    } else {
        machine.transition_state_estimate()
    };
    Ok(PointOutcome {
        converged: end == Some(DsAfirEnd::Met),
        end: end.map(|e| match e {
            DsAfirEnd::Met => "the two ends met".to_string(),
            DsAfirEnd::StepLimit => "the steps are used up".to_string(),
        }),
        interrupted: stop.stopped() && flow == Flow::Continue,
        iterations: iteration,
        transition_state: estimate,
    })
}

/// With the ends on two surfaces: the point between them as the estimate of a crossing,
/// with the energies of the two states there (the higher one is the energy of the
/// estimate).
fn crossing_estimate(
    machine: &DoubleSphereAfir,
    surface: &dyn SearchSurface,
    run: &SearchRun,
) -> MoptResult<Option<(Col<f64>, f64)>> {
    let Some((x, at_ends)) = machine.meeting_point() else {
        return Ok(None);
    };
    let request = crate::path::points::PointRequest::energy(x.clone());
    let first = surface.evaluate(&x, 0, &request)?.biased_energy;
    let second = surface.evaluate(&x, 1, &request)?.biased_energy;
    super::log_notes(surface, run.log);
    run.log.info(&format!(
        "DS-AFIR on two surfaces: the ends are at {:.10} (state of the first structure) and \
         {:.10} (state of the second); at the point between them the two states are at \
         {first:.10} and {second:.10}, {:.3} kcal/mol apart.",
        at_ends[0],
        at_ends[1],
        (second - first).abs() * crate::domain::Units::HARTREE_TO_KCAL_MOL
    ));
    Ok(Some((x, first.max(second))))
}
