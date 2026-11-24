//! What every method of the mapper does with its transition-state guesses: each guess is
//! optimized to a transition state; a transition state already in the network, or one an
//! earlier guess of the job reached, is not followed again; the IRC of each new one is run,
//! and its two ends are optimized to minima. An end whose molecules are apart is a
//! dissociation channel instead and is not minimized, and a minimum of two or more molecules
//! (a complex) carries the channel of its molecules ([`super::dissociation`]). The reactions
//! found are what the job gives the driver ([`super::driver`]), which compares them with the
//! network.
//!
//! At a minimized IRC endpoint, an imaginary vibration at or above the configured magnitude threshold
//! triggers a search for another transition state (a TS–TS link: past a valley-ridge
//! inflection the path ran down a ridge into it, as it does where a symmetric reaction
//! bifurcates). That transition state is optimized at order 1 and followed in its turn, its
//! reaction goes onto the outcome first, and the end of the side is that transition state, not
//! a minimum, so the network has `EQ - TS1 - TS2` with the two products of `TS2`. Before that, a
//! shoulder of the side (a local minimum of the slope of the energy) is minimized once: a
//! minimum above the transition state at the end is a shallow one the path went past, and it is
//! the end instead.
//!
//! With `higher_saddles=on`, a guess whose search ends at a point with k ≥ 2 negative
//! curvatures, and a side of an IRC that ends at one, are optimized at order k and descended
//! from: each unstable mode is followed both ways from the point, each start optimized at order
//! k - 1 with the other unstable modes followed, down to transition states (followed as any,
//! their IRCs run) and minima. With `confirm_bifurcations=on`, a possible bifurcation of a side
//! of an IRC is tried: the lowest mode across the path at its point is followed both ways and
//! each start minimized (`docs/source/mapper_design.md`).
//!
//! The methods differ in where their guesses come from: the maxima of an AFIR path
//! ([`super::afir`]), the estimates of a two-point method ([`super::addf`],
//! [`super::nanoreactor`]).
//!
//! MultiOptPy: steps 3 and 4 of `Wrapper/autots.py:AutoTSWorkflow` that its mapper runs for
//!  every AFIR pair.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::configuration::Configuration;
use super::dissociation::dissociation_of;
use super::method::{
    Descent, DescentEnd, FoundReaction, FoundSaddle, IrcBranch, Job, JobContext, JobOutcome,
    OptimizeRequest, PathPoint, SaddleEnd, Stage, Stationary, DEFAULT_DESCENT_STEP,
};
use super::network::{Level, NodeKind};
use crate::error::MoptResult;
use crate::workflow::explore::identity::StructureMatcher;

/// A transition-state guess of a job.
#[derive(Debug, Clone)]
pub struct Guess {
    /// The point (a structure, or the values of the variables).
    pub system: Configuration,
    /// Where it comes from, for the warnings (`point 12 of the AFIR path`).
    pub origin: String,
}

/// `system` as a dissociation channel when its molecules are apart (the sum of the energies
/// of its fragments, each optimized on its own); `None` otherwise or when the run was stopped.
fn channel_end(
    ctx: &mut JobContext<'_>,
    system: &Configuration,
    warnings: &mut Vec<String>,
) -> MoptResult<Option<Stationary>> {
    Ok(
        dissociation_of(ctx, system, true, warnings)?.map(|d| Stationary {
            index_wavenumbers: None,
            system: system.clone(),
            energy: d.energy,
            wavenumbers: None,
            dissociation: Some(d),
        }),
    )
}

/// How many transition states a chain of TS–TS links may have beyond the first one
/// (TS1 -> TS2 -> TS3 -> TS4).
pub const MAX_LINKED_SADDLES: usize = 3;

/// A transition state this job reached, with the reaction it gave (an index of
/// [`JobOutcome::reactions`]; `None` when following it gave none).
struct Tried {
    ts: Stationary,
    reaction: Option<usize>,
}

/// The entry of `tried` that is the same transition state as `ts`.
fn tried_before<'t>(
    tried: &'t [Tried],
    ts: &Stationary,
    matcher: &StructureMatcher,
) -> MoptResult<Option<&'t Tried>> {
    for t in tried {
        if t.ts
            .system
            .same_as(t.ts.energy, &ts.system, ts.energy, matcher)?
        {
            return Ok(Some(t));
        }
    }
    Ok(None)
}

/// What following a transition state gave.
enum Followed {
    /// The run was stopped.
    Stopped,
    /// Nothing for the network (the reasons are in the warnings of the outcome).
    Nothing,
    /// A reaction, pushed onto the outcome at this index.
    Reaction(usize),
}

/// Follow `guesses` in their order (the most promising first) into `outcome`; returns
/// `false` when the run was stopped (the job is then [`JobOutcome::stopped`]).
pub fn follow_guesses(
    job: &Job,
    ctx: &mut JobContext<'_>,
    guesses: &[Guess],
    outcome: &mut JobOutcome,
) -> MoptResult<bool> {
    let mut tried: Vec<Tried> = Vec::new();
    for (k, guess) in guesses.iter().enumerate() {
        if ctx.jobs.stop_requested() {
            return Ok(false);
        }
        let ts = ctx.jobs.optimize(
            &guess.system,
            &OptimizeRequest {
                label: format!("TS guess {}", k + 1),
                stage: Stage::Saddle,
                terms: Vec::new(),
            },
            ctx.frames,
        )?;
        if ts.interrupted {
            return Ok(false);
        }
        if !ts.converged {
            outcome.warnings.push(format!(
                "the transition-state search from guess {} ({}) did not converge",
                k + 1,
                guess.origin
            ));
            continue;
        }
        let found = Stationary {
            index_wavenumbers: None,
            system: ts.system,
            energy: ts.energy,
            wavenumbers: None,
            dissociation: None,
        };
        if let Some(known) = ctx.network.find_same_in(
            &NodeKind::TransitionState,
            Level::Exploration,
            &found.system,
            found.energy,
            ctx.matcher,
            ctx.state.as_ref(),
        )? {
            ctx.jobs.note(&format!(
                "Job {}: TS guess {} converged to the known TS{known}.",
                job.id,
                k + 1
            ));
            outcome.known.push(known);
            continue;
        }
        // A guess that converged to a transition state of an earlier guess of this job (or that
        // an IRC of this job ended at) is not followed again, whatever came of the first one.
        if tried_before(&tried, &found, ctx.matcher)?.is_some() {
            continue;
        }
        let label = format!("guess {}", k + 1);
        if let Followed::Stopped = follow_ts(job, ctx, found, &label, 0, &mut tried, outcome)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The first point of a side of an IRC before its point `before`, after its start, at which
/// the slope of the energy along the path has a local minimum below a hundredth of its
/// largest value on the side: a near stop of the path, where it passed close to a stationary
/// point (another transition state that it ran off along the unstable direction of the ridge,
/// or a shallow minimum that lies off the path).
fn shoulder(arc_lengths: &[f64], energies: &[f64], before: usize) -> Option<usize> {
    let n = arc_lengths.len().min(energies.len()).min(before);
    if n < 5 {
        return None;
    }
    let slope: Vec<f64> = (1..n)
        .map(|k| {
            let ds = arc_lengths[k] - arc_lengths[k - 1];
            if ds > 0.0 {
                ((energies[k] - energies[k - 1]) / ds).abs()
            } else {
                f64::NAN
            }
        })
        .collect();
    let top = slope
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .fold(0.0_f64, f64::max);
    // slope[k] is between points k and k + 1; the last two slopes lead into the end.
    (1..slope.len().saturating_sub(2))
        .find(|&k| {
            slope[k].is_finite()
                && slope[k] < 0.01 * top
                && slope[k] <= slope[k - 1]
                && slope[k] <= slope[k + 1]
        })
        .map(|k| k + 1)
}

/// What came of the end of a side at another transition state.
enum LinkEnd {
    /// The run was stopped.
    Stopped,
    /// The end is a transition state: the stationary point and where it is.
    Saddle(Stationary, SaddleEnd),
    /// A shallow minimum at a shoulder of the side before it, which is the end instead (the
    /// transition state beyond it is followed as a reaction of its own).
    Minimum(Stationary),
    /// No transition state could be settled; the end is minimized as usual.
    Unsettled,
}

/// The end of `branch`, a side of the IRC of the transition state from `label` that reached
/// a point `at` (its point `index`: the end, or a near stop on the way) with one negative
/// curvature: another transition state (a TS–TS link), optimized at order 1 from there and
/// followed in its turn, or a known one; or, where the side has a shoulder before it that is a
/// shallow minimum, that minimum.
#[allow(clippy::too_many_arguments)]
fn link_end(
    job: &Job,
    ctx: &mut JobContext<'_>,
    branch: &IrcBranch,
    at: &Configuration,
    index: usize,
    label: &str,
    depth: usize,
    tried: &mut Vec<Tried>,
    outcome: &mut JobOutcome,
) -> MoptResult<LinkEnd> {
    let side = &branch.side.name;
    let ts = ctx.jobs.optimize(
        at,
        &OptimizeRequest {
            label: format!("TS at the {side} end of the IRC of the TS from {label}"),
            stage: Stage::Saddle,
            terms: Vec::new(),
        },
        ctx.frames,
    )?;
    if ts.interrupted {
        return Ok(LinkEnd::Stopped);
    }
    if !ts.converged {
        outcome.warnings.push(format!(
            "the {side} side of the IRC of the transition state from {label} ended at a saddle \
             point, but its optimization at order 1 did not converge"
        ));
        return Ok(LinkEnd::Unsettled);
    }
    let found = Stationary {
        index_wavenumbers: None,
        system: ts.system,
        energy: ts.energy,
        wavenumbers: None,
        dissociation: None,
    };
    // A shoulder before the end: a tight minimization from it; a minimum above the
    // transition state at the end is a shallow minimum the path went past.
    if let Some(k) = shoulder(&branch.side.arc_lengths, &branch.side.energies, index) {
        if let Some(frame) = branch.frames.get(k) {
            if ctx.jobs.stop_requested() {
                return Ok(LinkEnd::Stopped);
            }
            let m = ctx.jobs.optimize(
                frame,
                &OptimizeRequest {
                    label: format!("shoulder of the {side} side of the IRC of the TS from {label}"),
                    stage: Stage::Minimum,
                    terms: Vec::new(),
                },
                ctx.frames,
            )?;
            if m.interrupted {
                return Ok(LinkEnd::Stopped);
            }
            if m.converged && m.energy > found.energy {
                ctx.jobs.note(&format!(
                    "Job {}: the {side} side of the IRC of the TS from {label} passed a shallow \
                     minimum at a shoulder; it is the end, and the TS beyond it a reaction of its \
                     own.",
                    job.id
                ));
                let minimum = Stationary {
                    index_wavenumbers: m.index_wavenumbers,
                    system: m.system,
                    energy: m.energy,
                    wavenumbers: m.wavenumbers,
                    dissociation: None,
                };
                if ctx
                    .network
                    .find_same_in(
                        &NodeKind::TransitionState,
                        Level::Exploration,
                        &found.system,
                        found.energy,
                        ctx.matcher,
                        ctx.state.as_ref(),
                    )?
                    .is_none()
                    && tried_before(tried, &found, ctx.matcher)?.is_none()
                {
                    let beyond = format!("the TS beyond the shoulder of {label}");
                    if let Followed::Stopped =
                        follow_ts(job, ctx, found, &beyond, depth + 1, tried, outcome)?
                    {
                        return Ok(LinkEnd::Stopped);
                    }
                }
                return Ok(LinkEnd::Minimum(minimum));
            }
        }
    }
    if let Some(known) = ctx.network.find_same_in(
        &NodeKind::TransitionState,
        Level::Exploration,
        &found.system,
        found.energy,
        ctx.matcher,
        ctx.state.as_ref(),
    )? {
        return Ok(LinkEnd::Saddle(found, SaddleEnd::Known(known)));
    }
    if let Some(t) = tried_before(tried, &found, ctx.matcher)? {
        return Ok(match t.reaction {
            Some(r) => LinkEnd::Saddle(t.ts.clone(), SaddleEnd::Reaction(r)),
            None => LinkEnd::Unsettled,
        });
    }
    let next = format!("the {side} end of the IRC of the TS from {label}");
    match follow_ts(job, ctx, found, &next, depth + 1, tried, outcome)? {
        Followed::Stopped => Ok(LinkEnd::Stopped),
        Followed::Nothing => Ok(LinkEnd::Unsettled),
        Followed::Reaction(r) => Ok(LinkEnd::Saddle(
            outcome.reactions[r].ts.clone(),
            SaddleEnd::Reaction(r),
        )),
    }
}

/// What following a stationary point of order 2 or more gave.
enum Higher {
    /// The run was stopped.
    Stopped,
    /// Nothing for the network (the reasons are in the warnings of the outcome).
    Nothing,
    /// The saddle point, and where it is (a new one of the outcome, or a known node).
    Found(Box<Stationary>, SaddleEnd),
}

/// Follow the stationary point of order `order` (2 or more) near `start` (from `label`): it is
/// optimized at that order (following `follow` from the start, where given), checked by its
/// Hessian, matched with the known ones and those of this job, and descended along each of its
/// unstable modes: both signs of a displacement along the mode, each optimized at order
/// `order - 1` with the other unstable modes followed (`higher_saddles`).
#[allow(clippy::too_many_arguments)]
fn follow_higher(
    job: &Job,
    ctx: &mut JobContext<'_>,
    start: &Configuration,
    order: usize,
    follow: &[faer::Col<f64>],
    label: &str,
    depth: usize,
    tried: &mut Vec<Tried>,
    outcome: &mut JobOutcome,
) -> MoptResult<Higher> {
    let Some(settings) = ctx.jobs.higher_saddles() else {
        return Ok(Higher::Nothing);
    };
    if order > settings.max_order {
        outcome.warnings.push(format!(
            "{label} has {order} negative curvatures, more than max_saddle_order ({}); it is not \
             followed",
            settings.max_order
        ));
        return Ok(Higher::Nothing);
    }
    if ctx.jobs.stop_requested() {
        return Ok(Higher::Stopped);
    }
    let s = ctx.jobs.saddle(
        start,
        order,
        follow,
        &format!("saddle point of order {order} from {label}"),
        ctx.frames,
    )?;
    if s.optimized.interrupted {
        return Ok(Higher::Stopped);
    }
    let modes = match s.modes {
        Some(m) if s.optimized.converged && m.negative == order => m,
        Some(m) if s.optimized.converged => {
            outcome.warnings.push(format!(
                "the optimization at order {order} from {label} converged to a point with {} \
                 negative curvatures",
                m.negative
            ));
            return Ok(Higher::Nothing);
        }
        _ => {
            outcome.warnings.push(format!(
                "the optimization at order {order} from {label} did not converge"
            ));
            return Ok(Higher::Nothing);
        }
    };
    let point = Stationary {
        index_wavenumbers: Some(
            modes
                .unstable
                .iter()
                .filter_map(|mode| mode.wavenumber)
                .collect(),
        ),
        system: s.optimized.system,
        energy: s.optimized.energy,
        wavenumbers: modes.wavenumbers.clone(),
        dissociation: None,
    };
    let kind = NodeKind::of_order(order)?;
    if let Some(known) = ctx.network.find_same_in(
        &kind,
        Level::Exploration,
        &point.system,
        point.energy,
        ctx.matcher,
        ctx.state.as_ref(),
    )? {
        outcome.known.push(known);
        return Ok(Higher::Found(Box::new(point), SaddleEnd::Known(known)));
    }
    for (i, f) in outcome.saddles.iter().enumerate() {
        if f.order == order
            && f.point
                .system
                .same_as(f.point.energy, &point.system, point.energy, ctx.matcher)?
        {
            return Ok(Higher::Found(Box::new(point), SaddleEnd::Higher(i)));
        }
    }
    outcome.saddles.push(FoundSaddle {
        point: point.clone(),
        order,
        descents: Vec::new(),
    });
    let me = outcome.saddles.len() - 1;
    ctx.jobs.note(&format!(
        "Job {}: a saddle point of order {order} ({label}), E = {:.10} Hartree; descending \
         along its {order} unstable modes.",
        job.id, point.energy
    ));
    let unstable: Vec<_> = modes.unstable.iter().take(order).cloned().collect();
    for (i, mode) in unstable.iter().enumerate() {
        let others: Vec<faer::Col<f64>> = unstable
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, m)| m.displacement.clone())
            .collect();
        let mut ends = [DescentEnd::Unsettled, DescentEnd::Unsettled];
        for (k, sign) in [1.0, -1.0].into_iter().enumerate() {
            if ctx.jobs.stop_requested() {
                return Ok(Higher::Stopped);
            }
            let from =
                ctx.jobs
                    .displace_mode(&point.system, &mode.displacement, sign, settings.step)?;
            let what = format!(
                "the descent along mode {} ({}) of the saddle point of order {order} from {label}",
                i + 1,
                if sign > 0.0 { "+" } else { "-" }
            );
            match descend(
                job,
                ctx,
                &from,
                order - 1,
                &others,
                &what,
                depth,
                tried,
                outcome,
            )? {
                Some(end) => ends[k] = end,
                None => return Ok(Higher::Stopped),
            }
        }
        outcome.saddles[me].descents.push(Descent {
            mode: i,
            wavenumber: mode.wavenumber,
            ends,
        });
    }
    Ok(Higher::Found(Box::new(point), SaddleEnd::Higher(me)))
}

/// Follow the stationary point of order `order` (2 or more) near `start` as
/// the saddle points met by a search or an IRC are, from a method of its own (the upward
/// search): optimized at that
/// order following `follow`, matched, and descended from, into `outcome`. Returns `false` when
/// the run was stopped.
pub fn follow_saddle_point(
    job: &Job,
    ctx: &mut JobContext<'_>,
    start: &Configuration,
    order: usize,
    follow: &[faer::Col<f64>],
    label: &str,
    outcome: &mut JobOutcome,
) -> MoptResult<bool> {
    let mut tried = Vec::new();
    Ok(!matches!(
        follow_higher(job, ctx, start, order, follow, label, 0, &mut tried, outcome)?,
        Higher::Stopped
    ))
}

/// The end of a descent from `start` at order `order` (following `follow`): a minimum, a
/// transition state followed as any (its IRC), or a saddle point of higher order followed in
/// its turn; `None` when the run was stopped.
#[allow(clippy::too_many_arguments)]
fn descend(
    job: &Job,
    ctx: &mut JobContext<'_>,
    start: &Configuration,
    order: usize,
    follow: &[faer::Col<f64>],
    label: &str,
    depth: usize,
    tried: &mut Vec<Tried>,
    outcome: &mut JobOutcome,
) -> MoptResult<Option<DescentEnd>> {
    match order {
        0 => {
            let end = ctx.jobs.optimize(
                start,
                &OptimizeRequest {
                    label: label.to_string(),
                    stage: Stage::Minimum,
                    terms: Vec::new(),
                },
                ctx.frames,
            )?;
            if end.interrupted {
                return Ok(None);
            }
            if !end.converged {
                outcome
                    .warnings
                    .push(format!("{label} did not converge to a minimum"));
                return Ok(Some(DescentEnd::Unsettled));
            }
            let molecules = dissociation_of(ctx, &end.system, false, &mut outcome.warnings)?;
            let apart = molecules.as_ref().is_some_and(|d| d.separated);
            Ok(Some(DescentEnd::Minimum(Box::new(Stationary {
                index_wavenumbers: if apart { None } else { end.index_wavenumbers },
                energy: match &molecules {
                    Some(d) if apart => d.energy,
                    _ => end.energy,
                },
                system: end.system,
                wavenumbers: if apart { None } else { end.wavenumbers },
                dissociation: molecules,
            }))))
        }
        1 => {
            let s = ctx.jobs.saddle(start, 1, follow, label, ctx.frames)?;
            if s.optimized.interrupted {
                return Ok(None);
            }
            if !(s.optimized.converged && s.modes.as_ref().is_some_and(|m| m.negative == 1)) {
                outcome
                    .warnings
                    .push(format!("{label} did not converge to a transition state"));
                return Ok(Some(DescentEnd::Unsettled));
            }
            let found = Stationary {
                index_wavenumbers: None,
                system: s.optimized.system,
                energy: s.optimized.energy,
                wavenumbers: None,
                dissociation: None,
            };
            if let Some(known) = ctx.network.find_same_in(
                &NodeKind::TransitionState,
                Level::Exploration,
                &found.system,
                found.energy,
                ctx.matcher,
                ctx.state.as_ref(),
            )? {
                return Ok(Some(DescentEnd::Known(known)));
            }
            if let Some(t) = tried_before(tried, &found, ctx.matcher)? {
                return Ok(Some(match t.reaction {
                    Some(r) => DescentEnd::Reaction(r),
                    None => DescentEnd::Unsettled,
                }));
            }
            Ok(Some(
                match follow_ts(job, ctx, found, label, depth + 1, tried, outcome)? {
                    Followed::Stopped => return Ok(None),
                    Followed::Nothing => DescentEnd::Unsettled,
                    Followed::Reaction(r) => DescentEnd::Reaction(r),
                },
            ))
        }
        k => Ok(Some(
            match follow_higher(job, ctx, start, k, follow, label, depth + 1, tried, outcome)? {
                Higher::Stopped => return Ok(None),
                Higher::Nothing => DescentEnd::Unsettled,
                Higher::Found(_, SaddleEnd::Known(id)) => DescentEnd::Known(id),
                Higher::Found(_, SaddleEnd::Higher(i)) => DescentEnd::Higher(i),
                Higher::Found(_, SaddleEnd::Reaction(r)) => DescentEnd::Reaction(r),
            },
        )),
    }
}

/// Confirm the possible bifurcations of the sides of an IRC (`confirm_bifurcations`): at the
/// point where the curvature across the path was negative, the lowest mode across the path
/// (the gradient direction taken out) is followed both ways and each start minimized. Returns
/// the minima of each side; `None` when the run was stopped.
fn confirm_bifurcations(
    ctx: &mut JobContext<'_>,
    branches: &[IrcBranch],
    label: &str,
    warnings: &mut Vec<String>,
) -> MoptResult<Option<Vec<(usize, Vec<Stationary>)>>> {
    let step = ctx
        .jobs
        .higher_saddles()
        .map_or(DEFAULT_DESCENT_STEP, |s| s.step);
    let mut out = Vec::new();
    for (s, branch) in branches.iter().enumerate() {
        let Some(b) = &branch.possible_bifurcation else {
            continue;
        };
        let at = branch
            .side
            .arc_lengths
            .iter()
            .position(|&a| (a - b.at).abs() <= 1e-9 * b.at.abs().max(1.0));
        let Some(frame) = at.and_then(|k| branch.frames.get(k)) else {
            continue;
        };
        if ctx.jobs.stop_requested() {
            return Ok(None);
        }
        let Some(modes) = ctx.jobs.modes(frame, true)? else {
            continue;
        };
        let Some(mode) = modes.modes.first().filter(|m| m.curvature < 0.0).cloned() else {
            warnings.push(format!(
                "the possible bifurcation of the {} side of the IRC of the transition state \
                 from {label}: the Hessian there again has no negative curvature across the path",
                b.side
            ));
            continue;
        };
        let mut found = Vec::new();
        for sign in [1.0, -1.0] {
            if ctx.jobs.stop_requested() {
                return Ok(None);
            }
            let start = ctx
                .jobs
                .displace_mode(frame, &mode.displacement, sign, step)?;
            let end = ctx.jobs.optimize(
                &start,
                &OptimizeRequest {
                    label: format!(
                        "bifurcation of the {} side of the IRC of the TS from {label} ({})",
                        b.side,
                        if sign > 0.0 { "+" } else { "-" }
                    ),
                    stage: Stage::Minimum,
                    terms: Vec::new(),
                },
                ctx.frames,
            )?;
            if end.interrupted {
                return Ok(None);
            }
            if end.converged {
                let molecules = dissociation_of(ctx, &end.system, false, warnings)?;
                let apart = molecules.as_ref().is_some_and(|d| d.separated);
                found.push(Stationary {
                    index_wavenumbers: if apart { None } else { end.index_wavenumbers },
                    energy: match &molecules {
                        Some(d) if apart => d.energy,
                        _ => end.energy,
                    },
                    system: end.system,
                    wavenumbers: if apart { None } else { end.wavenumbers },
                    dissociation: molecules,
                });
            }
        }
        out.push((s, found));
    }
    Ok(Some(out))
}

/// Keep the saddle point `point` of order `order` (checked by `modes`) as a node of the outcome
/// without descending from it, unless the network or the outcome has it already. Returns where
/// it is.
fn keep_saddle(
    ctx: &mut JobContext<'_>,
    point: &Stationary,
    order: usize,
    modes: &super::method::PointModes,
    outcome: &mut JobOutcome,
) -> MoptResult<SaddleEnd> {
    let kind = NodeKind::of_order(order)?;
    if let Some(known) = ctx.network.find_same_in(
        &kind,
        Level::Exploration,
        &point.system,
        point.energy,
        ctx.matcher,
        ctx.state.as_ref(),
    )? {
        return Ok(SaddleEnd::Known(known));
    }
    for (i, f) in outcome.saddles.iter().enumerate() {
        if f.order == order
            && f.point
                .system
                .same_as(f.point.energy, &point.system, point.energy, ctx.matcher)?
        {
            return Ok(SaddleEnd::Higher(i));
        }
    }
    outcome.saddles.push(FoundSaddle {
        point: Stationary {
            index_wavenumbers: Some(
                modes
                    .unstable
                    .iter()
                    .filter_map(|mode| mode.wavenumber)
                    .collect(),
            ),
            wavenumbers: modes.wavenumbers.clone(),
            ..point.clone()
        },
        order,
        descents: Vec::new(),
    });
    Ok(SaddleEnd::Higher(outcome.saddles.len() - 1))
}

/// A search for a transition state from `label` that converged to `found`, a point with
/// `order` >= 2 negative curvatures, without `higher_saddles`: the point is kept as a `kSP`
/// node (its links are not followed), and the mode kill of the search goes on: the point is
/// displaced along its unstable modes but the lowest and optimized at order 1 again, following
/// the lowest; a transition state found so is followed as any.
#[allow(clippy::too_many_arguments)]
fn kill_to_transition_state(
    job: &Job,
    ctx: &mut JobContext<'_>,
    found: &Stationary,
    label: &str,
    depth: usize,
    tried: &mut Vec<Tried>,
    outcome: &mut JobOutcome,
) -> MoptResult<Followed> {
    let Some(modes) = ctx.jobs.modes(&found.system, false)? else {
        return Ok(Followed::Nothing);
    };
    let order = modes.negative;
    if order < 2 {
        return Ok(Followed::Nothing);
    }
    keep_saddle(ctx, found, order, &modes, outcome)?;
    ctx.jobs.note(&format!(
        "Job {}: the search from {label} converged to a saddle point of order {order}; it is \
         kept, and the mode kill goes on to a transition state.",
        job.id
    ));
    if ctx.jobs.stop_requested() {
        return Ok(Followed::Stopped);
    }
    let what = format!("mode kill of the saddle point of order {order} from {label}");
    let Some(s) = ctx
        .jobs
        .mode_kill_saddle(&found.system, &modes, &what, ctx.frames)?
    else {
        if ctx.jobs.stop_requested() {
            return Ok(Followed::Stopped);
        }
        outcome
            .warnings
            .push(format!("the {what} did not converge to a transition state"));
        return Ok(Followed::Nothing);
    };
    if s.optimized.interrupted {
        return Ok(Followed::Stopped);
    }
    if !(s.optimized.converged && s.modes.as_ref().is_some_and(|m| m.negative == 1)) {
        outcome
            .warnings
            .push(format!("the {what} did not converge to a transition state"));
        return Ok(Followed::Nothing);
    }
    let ts = Stationary {
        index_wavenumbers: None,
        system: s.optimized.system,
        energy: s.optimized.energy,
        wavenumbers: None,
        dissociation: None,
    };
    if ctx
        .network
        .find_same_in(
            &NodeKind::TransitionState,
            Level::Exploration,
            &ts.system,
            ts.energy,
            ctx.matcher,
            ctx.state.as_ref(),
        )?
        .is_some()
        || tried_before(tried, &ts, ctx.matcher)?.is_some()
    {
        return Ok(Followed::Nothing);
    }
    follow_ts(job, ctx, ts, &what, depth + 1, tried, outcome)
}

/// Run the IRC of the transition state `found` (from `label`, `guess 3`), take its ends to
/// minima or channels, follow an end at another transition state as a TS–TS link (up to
/// [`MAX_LINKED_SADDLES`] deep), and push the reaction onto `outcome`, after the reactions of
/// the transition states its ends reached.
#[allow(clippy::too_many_arguments)]
fn follow_ts(
    job: &Job,
    ctx: &mut JobContext<'_>,
    found: Stationary,
    label: &str,
    depth: usize,
    tried: &mut Vec<Tried>,
    outcome: &mut JobOutcome,
) -> MoptResult<Followed> {
    tried.push(Tried {
        ts: found.clone(),
        reaction: None,
    });
    let me = tried.len() - 1;
    let irc = ctx.jobs.irc(&found.system, ctx.frames)?;
    if irc.interrupted {
        return Ok(Followed::Stopped);
    }
    if !irc.transition_state {
        if irc.imaginary_modes >= 2 && ctx.jobs.higher_saddles().is_some() {
            // A saddle point of higher order: optimized at its order and descended from.
            let order = irc.imaginary_modes;
            if let Higher::Stopped = follow_higher(
                job,
                ctx,
                &found.system,
                order,
                &[],
                label,
                depth,
                tried,
                outcome,
            )? {
                return Ok(Followed::Stopped);
            }
            return Ok(Followed::Nothing);
        }
        if irc.imaginary_modes >= 2 {
            // Kept as a node; the mode kill goes on to a transition state.
            return kill_to_transition_state(job, ctx, &found, label, depth, tried, outcome);
        }
        outcome.warnings.push(if irc.imaginary_modes == 1 {
            format!(
                "the search from {label} converged to a structure with one imaginary mode, but \
                 its gradient at the start of the IRC is above the thresholds of a transition \
                 state (the log of the job has the numbers)"
            )
        } else {
            format!(
                "the search from {label} converged to a stationary point with {} imaginary \
                 modes, not a transition state",
                irc.imaginary_modes
            )
        });
        return Ok(Followed::Nothing);
    }
    if irc.branches.len() != 2 {
        outcome.warnings.push(format!(
            "the IRC of the transition state from {label} has {} side(s), not two",
            irc.branches.len()
        ));
        return Ok(Followed::Nothing);
    }
    let mut ends = Vec::new();
    // The minimizations of the ends of the sides (for the path file of the reaction).
    let mut settles: [Vec<PathPoint>; 2] = [Vec::new(), Vec::new()];
    let mut saddle_ends = [None, None];
    // Where a side reached the transition state of a link before it ran off: its data end
    // there.
    let mut cut: [Option<usize>; 2] = [None, None];
    'sides: for (s, branch) in irc.branches.iter().enumerate() {
        // An end whose molecules are apart is a dissociation channel; it is not minimized.
        if let Some(end) = channel_end(ctx, &branch.last, &mut outcome.warnings)? {
            ends.push(Some(end));
            continue;
        }
        if ctx.jobs.stop_requested() {
            return Ok(Followed::Stopped);
        }
        // Another transition state on the way: the side ran down a ridge into it (a TS–TS
        // link) and either ended there with a small gradient or, numerically off the ridge, ran
        // off it along its unstable direction after a near stop. A point with one negative
        // curvature there is that transition state.
        let mut places: Vec<(usize, &Configuration)> = Vec::new();
        if depth < MAX_LINKED_SADDLES {
            let points = branch.side.arc_lengths.len();
            if let Some(k) = shoulder(&branch.side.arc_lengths, &branch.side.energies, points) {
                if let Some(frame) = branch.frames.get(k) {
                    places.push((k, frame));
                }
            }
            if branch.side.end == "converged" {
                places.push((points, &branch.last));
            }
        }
        for (index, at) in places {
            if ctx.jobs.stop_requested() {
                return Ok(Followed::Stopped);
            }
            let negative = ctx.jobs.negative_curvatures(at)?;
            let higher = ctx.jobs.higher_saddles().is_some();
            if let Some(order) = negative.filter(|&k| k >= 2 && !higher) {
                // Without higher_saddles the saddle point is kept as a node (optimized at its
                // order) and the end of the side is minimized as before.
                let what = format!(
                    "the {} end of the IRC of the TS from {label}",
                    branch.side.name
                );
                let s = ctx.jobs.saddle(
                    at,
                    order,
                    &[],
                    &format!("saddle point of order {order} at {what}"),
                    ctx.frames,
                )?;
                if s.optimized.interrupted {
                    return Ok(Followed::Stopped);
                }
                if let Some(m) = s
                    .modes
                    .filter(|m| s.optimized.converged && m.negative == order)
                {
                    let point = Stationary {
                        index_wavenumbers: None,
                        system: s.optimized.system,
                        energy: s.optimized.energy,
                        wavenumbers: None,
                        dissociation: None,
                    };
                    keep_saddle(ctx, &point, order, &m, outcome)?;
                }
                break;
            }
            if let Some(order) = negative.filter(|&k| k >= 2 && higher) {
                // The side ran into a saddle point of higher order.
                let what = format!(
                    "the {} end of the IRC of the TS from {label}",
                    branch.side.name
                );
                match follow_higher(job, ctx, at, order, &[], &what, depth + 1, tried, outcome)? {
                    Higher::Stopped => return Ok(Followed::Stopped),
                    Higher::Found(point, which) => {
                        ctx.jobs.note(&format!(
                            "Job {}: the {} side of the IRC of the TS from {label} ends at a \
                             saddle point of order {order}.",
                            job.id, branch.side.name
                        ));
                        ends.push(Some(*point));
                        saddle_ends[s] = Some(which);
                        cut[s] = Some(index);
                        continue 'sides;
                    }
                    Higher::Nothing => {}
                }
                break;
            }
            if negative == Some(1) && index < branch.side.arc_lengths.len() {
                match link_end(job, ctx, branch, at, index, label, depth, tried, outcome)? {
                    LinkEnd::Stopped => return Ok(Followed::Stopped),
                    LinkEnd::Saddle(ts, which) => {
                        ctx.jobs.note(&format!(
                            "Job {}: the {} side of the IRC of the TS from {label} ends at \
                             another transition state (a TS–TS link).",
                            job.id, branch.side.name
                        ));
                        ends.push(Some(ts));
                        saddle_ends[s] = Some(which);
                        cut[s] = Some(index);
                        continue 'sides;
                    }
                    LinkEnd::Minimum(end) => {
                        ends.push(Some(end));
                        continue 'sides;
                    }
                    LinkEnd::Unsettled => {}
                }
                break;
            }
        }
        if ctx.jobs.stop_requested() {
            return Ok(Followed::Stopped);
        }
        let mut end = ctx.jobs.optimize(
            &branch.last,
            &OptimizeRequest {
                label: format!("IRC end ({})", branch.side.name),
                stage: Stage::IrcEnd,
                terms: Vec::new(),
            },
            ctx.frames,
        )?;
        if end.interrupted {
            return Ok(Followed::Stopped);
        }
        if end.converged {
            if let Some(slot) = settles.get_mut(s) {
                *slot = std::mem::take(&mut end.path);
            }
        }
        if !end.converged {
            // The molecules may have parted on the way down.
            if let Some(apart) = channel_end(ctx, &end.system, &mut outcome.warnings)? {
                ends.push(Some(apart));
                continue;
            }
            if ctx.jobs.stop_requested() {
                return Ok(Followed::Stopped);
            }
            // The end of this side is unknown; the transition state is kept with the other
            // (docs/source/mapper_design.md, edges).
            outcome.warnings.push(format!(
                "the {} end of the IRC of the transition state from {label} did not converge \
                 to a minimum (or its Hessian still has negative curvature: see the log of the \
                 job); that end is unknown, and the transition state is kept without it",
                branch.side.name
            ));
            ends.push(None);
            continue;
        }
        let mut imaginary = end
            .index_wavenumbers
            .as_ref()
            .or(end.wavenumbers.as_ref())
            .into_iter()
            .flatten()
            .copied()
            .filter(|&w| w < 0.0);
        if let Some(lowest) = imaginary.next() {
            let order = 1 + imaginary.count();
            if order > 1 {
                let what = format!(
                    "the optimized {} IRC endpoint from {label}",
                    branch.side.name
                );
                outcome.warnings.push(format!(
                    "{what} has {order} imaginary modes and is excluded from equilibria"
                ));
                if depth < MAX_LINKED_SADDLES
                    && matches!(
                        follow_higher(
                            job,
                            ctx,
                            &end.system,
                            order,
                            &[],
                            &what,
                            depth + 1,
                            tried,
                            outcome
                        )?,
                        Higher::Stopped
                    )
                {
                    return Ok(Followed::Stopped);
                }
                // Optional higher-order searches have their own nodes and descents.
                // This IRC did not establish an equilibrium on this side.
                ends.push(None);
                continue;
            }
            let side = &branch.side.name;
            let threshold = ctx.jobs.irc_end_threshold_cm1();
            if lowest <= -threshold {
                if depth < MAX_LINKED_SADDLES {
                    match link_end(
                        job,
                        ctx,
                        branch,
                        &end.system,
                        0,
                        label,
                        depth,
                        tried,
                        outcome,
                    )? {
                        LinkEnd::Stopped => return Ok(Followed::Stopped),
                        LinkEnd::Saddle(ts, which) => {
                            ctx.jobs.note(&format!(
                                "Job {}: the {side} IRC endpoint has a mode at {lowest:.2} cm^-1 (a TS–TS link).",
                                job.id
                            ));
                            ends.push(Some(ts));
                            saddle_ends[s] = Some(which);
                            continue;
                        }
                        LinkEnd::Minimum(minimum) => {
                            ends.push(Some(minimum));
                            continue;
                        }
                        LinkEnd::Unsettled => {}
                    }
                }
                outcome.warnings.push(format!(
                    "the optimized {side} IRC endpoint from {label} has a mode at {lowest:.2} cm^-1; its TS–TS link could not be resolved"
                ));
                ends.push(None);
                continue;
            }
            outcome.warnings.push(format!(
                "the optimized {side} IRC endpoint from {label} has a weak imaginary mode at {lowest:.2} cm^-1 (threshold {threshold} cm^-1); recorded as an equilibrium"
            ));
        }
        // A minimum of two or more molecules is a complex, with the channel of its
        // molecules (or itself a channel when they are apart).
        let molecules = dissociation_of(ctx, &end.system, false, &mut outcome.warnings)?;
        if ctx.jobs.stop_requested() {
            return Ok(Followed::Stopped);
        }
        let apart = molecules.as_ref().is_some_and(|d| d.separated);
        ends.push(Some(Stationary {
            index_wavenumbers: if apart { None } else { end.index_wavenumbers },
            energy: match &molecules {
                Some(d) if apart => d.energy,
                _ => end.energy,
            },
            system: end.system,
            wavenumbers: if apart { None } else { end.wavenumbers },
            dissociation: molecules,
        }));
    }
    let (Some(second), Some(first)) = (ends.pop(), ends.pop()) else {
        return Ok(Followed::Nothing);
    };
    for b in irc
        .branches
        .iter()
        .filter_map(|b| b.possible_bifurcation.as_ref())
    {
        let after = match b.after {
            Some(a) => format!("between arc lengths {a:.3} and {:.3}", b.at),
            None => format!("before arc length {:.3}", b.at),
        };
        outcome.warnings.push(format!(
            "the {} side of the IRC of the transition state from {label} may bifurcate {after} \
             amu^(1/2) Bohr: the curvature across the path is {:.0} cm^-1 there (from the exact \
             Hessians along the IRC; not confirmed, the IRC went on to one end)",
            b.side, b.wavenumber
        ));
    }
    let bifurcation_ends = if ctx.jobs.confirm_bifurcations() {
        match confirm_bifurcations(ctx, &irc.branches, label, &mut outcome.warnings)? {
            Some(found) => found,
            None => return Ok(Followed::Stopped),
        }
    } else {
        Vec::new()
    };
    // The transition state with the vibrations of the start check of its IRC.
    let ts = Stationary {
        index_wavenumbers: irc.imaginary_wavenumber.map(|w| vec![w]),
        wavenumbers: irc.wavenumbers,
        ..found
    };
    let mut branches = irc.branches;
    for (branch, settle) in branches.iter_mut().zip(settles) {
        branch.settle = settle;
    }
    for (branch, at) in branches.iter_mut().zip(cut) {
        let Some(k) = at else {
            continue;
        };
        if k + 1 < branch.side.arc_lengths.len() {
            branch.side.arc_lengths.truncate(k + 1);
            branch.side.energies.truncate(k + 1);
            branch.frames.truncate(k + 1);
            if let Some(last) = branch.frames.last() {
                branch.last = last.clone();
            }
            branch.side.end = "transition_state".to_string();
        }
    }
    outcome.reactions.push(FoundReaction {
        ts,
        imaginary_wavenumber: irc.imaginary_wavenumber,
        ends: [first, second],
        branches,
        saddle_ends,
        bifurcation_ends,
    });
    let index = outcome.reactions.len() - 1;
    tried[me].reaction = Some(index);
    Ok(Followed::Reaction(index))
}
