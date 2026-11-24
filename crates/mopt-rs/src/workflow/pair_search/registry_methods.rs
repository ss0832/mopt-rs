//! The members of the registry of two-point methods that run on points: iEIP, the dimer,
//! the spring pair, 2PSHS, the growing Newton trajectory, ADDF, the growing string (DE and
//! SE), the freezing string, and DS-AFIR. The chain methods (NEB, the direct MaxFlux
//! method, BITSS) are in [`super::registry_chains`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::point_runs::PointOutcome;
use super::registry::{Ends, PairInput, PairMethod, PairResult, PairSettings};
use super::registry_single::{AddfMember, DimerMember, SpmMember};
use super::registry_strings::{DsAfirMember, FsmMember, GsmMember};
use super::surface::SearchSurface;
use super::{run_ieip, run_newton_trajectory, run_two_point_shs, SearchRun};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};
use crate::irc::SphereSettings;
use crate::math::linalg::col_norm;
use crate::math::projection::build_tr_rot_basis;
use crate::path::ieip::{ElasticImagePair, IeipEnd, IeipSettings, MicroCoordinates, MicroSettings};
use crate::path::interp::FirstPath;
use crate::path::newton_trajectory::{GrowingNewtonTrajectory, NtDirection, NtSettings};
use crate::path::points::PointRequest;
use crate::path::twopshs::{TwoPointShs, TwoPshsSettings};
use crate::workflow::chain::Flow;

/// The members of the registry, in the order of the documentation.
pub fn standard() -> Vec<Arc<dyn PairMethod>> {
    let mut out: Vec<Arc<dyn PairMethod>> = vec![
        Arc::new(IeipMember),
        Arc::new(GsmMember { single: false }),
        Arc::new(GsmMember { single: true }),
        Arc::new(FsmMember),
        Arc::new(DsAfirMember),
        Arc::new(super::registry_experimental::LmmMember),
        Arc::new(TwoPshsMember),
        Arc::new(GntMember),
        Arc::new(DimerMember),
        Arc::new(SpmMember),
        Arc::new(AddfMember),
    ];
    out.extend(super::registry_chains::standard());
    out
}

pub(super) fn energy_at(surface: &dyn SearchSurface, x: &Col<f64>) -> MoptResult<f64> {
    Ok(surface
        .evaluate(x, 0, &PointRequest::energy(x.clone()))?
        .biased_energy)
}

/// A result from the outcome of a driver; an estimate without an energy is evaluated.
pub(super) fn from_outcome(
    name: &str,
    o: PointOutcome,
    path: Option<Vec<(Col<f64>, f64)>>,
    ends: Vec<Col<f64>>,
    surface: &dyn SearchSurface,
) -> MoptResult<PairResult> {
    let mut transition_states = Vec::new();
    if let Some((x, e)) = o.transition_state {
        let e = if e.is_finite() {
            e
        } else {
            energy_at(surface, &x)?
        };
        transition_states.push((x, e));
    }
    Ok(PairResult {
        method: name.to_string(),
        converged: o.converged,
        end: o.end.unwrap_or_else(|| "stopped".to_string()),
        transition_states,
        path,
        ends,
        counts: Default::default(),
        interrupted: o.interrupted,
    })
}

pub(super) fn go<R>(_: &R) -> MoptResult<Flow> {
    Ok(Flow::Continue)
}

/// The second point with a seeded random displacement of `noise` (structures: so that a
/// symmetric start can leave its symmetry), or the direction of a one-point input.
pub(super) fn direction(input: &PairInput, settings: &PairSettings) -> MoptResult<Col<f64>> {
    let first = input.first()?;
    let seed = settings.usize("seed", 1)? as u64;
    let noise = settings.f64("noise", if input.structures { 0.05 } else { 0.0 })?;
    let mut rng = Rng::from_seed(seed);
    let mut random = || {
        let v = Col::from_fn(first.nrows(), |_| rng.standard_normal());
        input.geometry.project(first, &v)
    };
    if let Some(d) = &input.direction {
        return Ok(d.clone());
    }
    match input.points.get(1) {
        Some(b) => {
            let b = input.geometry.laid_onto(b, first);
            let u = random();
            let size = col_norm(&u);
            let shifted = if noise > 0.0 && size > 0.0 {
                &b + &u * (noise / size)
            } else {
                b
            };
            Ok(&shifted - first)
        }
        None => Ok(random()),
    }
}

/// Directions left out of scaled coordinates: fixed variables, or rigid-body motions.
pub(super) fn removed(input: &PairInput) -> MoptResult<Option<Mat<f64>>> {
    let n = input.first()?.nrows();
    if !input.geometry.frozen.is_empty() {
        let f = &input.geometry.frozen;
        return Ok(Some(Mat::from_fn(n, f.len(), |i, j| {
            if i == f[j] {
                1.0
            } else {
                0.0
            }
        })));
    }
    Ok(input
        .geometry
        .rigid_body
        .as_ref()
        .and_then(|t| t.with_flat_coords_bohr(input.first().ok()?).ok())
        .map(|s| build_tr_rot_basis(&s, false)))
}

pub(super) fn sphere(settings: &PairSettings) -> MoptResult<SphereSettings> {
    Ok(SphereSettings {
        max_iterations: settings.usize("sphere_iterations", 30)?.max(1),
        gradient_tolerance: settings.f64("sphere_gradient", 1e-4)?,
        ..SphereSettings::default()
    })
}

pub(super) fn interpolation(
    settings: &PairSettings,
    input: &PairInput,
    default: FirstPath,
) -> MoptResult<FirstPath> {
    let method = match settings.text("interpolation") {
        Some(name) => FirstPath::from_name(&name)?,
        None => default,
    };
    if !input.structures && method != FirstPath::Linear {
        return Err(MoptError::config(
            "interpolation: plain variables are interpolated on the straight line",
        ));
    }
    Ok(method)
}

struct IeipMember;

impl PairMethod for IeipMember {
    fn name(&self) -> &'static str {
        "ieip"
    }
    fn summary(&self) -> &'static str {
        "improved elastic image pair method, MultiOptPy's procedure (Liu, Qi, Lei 2023)"
    }
    fn ends(&self) -> Ends {
        Ends::Two
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("iterations", "iterations at most (999)"),
            ("beta", "strength of the attraction of the images (1.0)"),
            (
                "micro",
                "relaxation steps on hyperspheres per iteration (0)",
            ),
            ("micro_tolerance", "force at which a relaxation ends (0.01)"),
        ]
    }
    fn run(
        &self,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
    ) -> MoptResult<PairResult> {
        let steps = settings.usize("micro", 0)?;
        let micro_coordinates = match (steps, &input.template) {
            (0, _) => None,
            (_, Some(t)) if input.structures => Some(MicroCoordinates::polar(t.clone())),
            _ => {
                return Err(MoptError::unsupported(
                    "ieip micro=: the relaxation on hyperspheres needs structures",
                ))
            }
        };
        let s = IeipSettings {
            max_iterations: settings.usize("iterations", 999)?.max(1),
            beta: settings.f64("beta", 1.0)?,
            micro: MicroSettings {
                steps,
                force_tolerance: settings.f64("micro_tolerance", 0.01)?,
                ..MicroSettings::default()
            },
            micro_coordinates,
            align: input.geometry.aligns(),
            frozen: input.geometry.frozen.clone(),
        };
        run.record_settings(self.name(), &s)?;
        let mut m = ElasticImagePair::new(s, input.first()?, input.second("ieip")?)?;
        if let Some(map) = &input.step_map {
            m = m.with_step_map(map.clone());
        }
        let o = run_ieip(&mut m, surface, run, &mut |r| go(r))?;
        let e = energy_at(surface, &o.transition_state)?;
        Ok(PairResult {
            method: self.name().to_string(),
            converged: o.end == Some(IeipEnd::Converged),
            end: match o.end {
                Some(IeipEnd::Converged) => format!("converged after {} iterations", o.iterations),
                Some(IeipEnd::IterationLimit) => "the iterations are used up".to_string(),
                None => "stopped".to_string(),
            },
            transition_states: vec![(o.transition_state.clone(), e)],
            path: Some(o.path),
            ends: o.images.to_vec(),
            counts: Default::default(),
            interrupted: o.interrupted,
        })
    }
}

struct TwoPshsMember;

impl PairMethod for TwoPshsMember {
    fn name(&self) -> &'static str {
        "2pshs"
    }
    fn summary(&self) -> &'static str {
        "two-point scaled hypersphere search into the first minimum (Maeda, Ohno 2005)"
    }
    fn ends(&self) -> Ends {
        Ends::Two
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            (
                "step",
                "contraction per sphere along the stiffest mode (0.05 Angstrom)",
            ),
            ("spheres", "spheres at most (300)"),
            (
                "sphere_iterations",
                "evaluations of a minimization on a sphere (30)",
            ),
            (
                "sphere_gradient",
                "RMS gradient within the sphere that ends it (1e-4)",
            ),
            (
                "noise",
                "random displacement of the second structure (0.05 Bohr; 0 for variables)",
            ),
            ("seed", "seed of that displacement (1)"),
        ]
    }
    fn run(
        &self,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
    ) -> MoptResult<PairResult> {
        let s = TwoPshsSettings {
            step: input.length(settings.f64("step", 0.05)?),
            max_spheres: settings.usize("spheres", 300)?.max(1),
            sphere: sphere(settings)?,
        };
        let first = input.first()?;
        let second = first + direction(input, settings)?;
        run.record_settings(self.name(), &s)?;
        let mut m = TwoPointShs::new(s, first, &second, removed(input)?)?;
        let o = run_two_point_shs(&mut m, surface, run, &mut |r| go(r))?;
        let path = m
            .path()
            .iter()
            .rev()
            .map(|p| (p.x.clone(), p.energy))
            .collect();
        from_outcome(
            self.name(),
            o,
            Some(path),
            vec![first.clone(), second],
            surface,
        )
    }
}

struct GntMember;

fn parse_bonds(text: &str) -> MoptResult<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    for pair in text.split([',', ';']).filter(|p| !p.trim().is_empty()) {
        let (a, b) = pair
            .split_once('-')
            .ok_or_else(|| MoptError::parse(format!("bonds: '{pair}' is not i-j")))?;
        let atom = |w: &str| -> MoptResult<usize> {
            w.trim()
                .parse::<usize>()
                .ok()
                .filter(|v| *v >= 1)
                .map(|v| v - 1)
                .ok_or_else(|| MoptError::parse(format!("bonds: '{w}' is not an atom (from 1)")))
        };
        out.push((atom(a)?, atom(b)?));
    }
    Ok(out)
}

impl PairMethod for GntMember {
    fn name(&self) -> &'static str {
        "gnt"
    }
    fn summary(&self) -> &'static str {
        "growing Newton trajectory towards the second structure or along bonds (Quapp 2005)"
    }
    fn ends(&self) -> Ends {
        Ends::OneOrTwo
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("step", "predictor step (0.5 Bohr)"),
            (
                "tolerance",
                "RMS of the projected gradient that ends the corrector (1.7e-3)",
            ),
            ("corrector", "corrector iterations per round (25)"),
            ("iterations", "iterations at most (999)"),
            (
                "bonds",
                "one-ended: bonds to shorten, i-j,k-l (atoms from 1)",
            ),
            (
                "noise",
                "random displacement of the second structure (0.05 Bohr; 0 for variables)",
            ),
            (
                "seed",
                "seed of that displacement or of a random direction (1)",
            ),
        ]
    }
    fn run(
        &self,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
    ) -> MoptResult<PairResult> {
        let s = NtSettings {
            step_length: settings.f64("step", 0.5)?,
            rms_tolerance: settings.f64("tolerance", 1.7e-3)?,
            corrector_iterations: settings.usize("corrector", 25)?.max(1),
            max_iterations: settings.usize("iterations", 999)?.max(1),
            align: input.geometry.aligns(),
            rigid_body: input.geometry.rigid_body.clone(),
            ..NtSettings::default()
        };
        let first = input.first()?;
        let source = match settings.text("bonds") {
            Some(text) if input.structures => NtDirection::Bonds(parse_bonds(&text)?),
            Some(_) => {
                return Err(MoptError::unsupported(
                    "gnt bonds=: plain variables have no atoms",
                ))
            }
            None if input.points.len() == 2 => {
                NtDirection::Toward(first + direction(input, settings)?)
            }
            None => NtDirection::Fixed(direction(input, settings)?),
        };
        run.record_settings(self.name(), &s)?;
        let mut m = GrowingNewtonTrajectory::new(s, first, source)?;
        if !input.geometry.frozen.is_empty() {
            m = m.with_frozen(input.geometry.frozen.clone())?;
        }
        let o = run_newton_trajectory(&mut m, surface, run, &mut |r| go(r))?;
        let path: Vec<(Col<f64>, f64)> =
            m.points().iter().map(|p| (p.x.clone(), p.energy)).collect();
        from_outcome(self.name(), o, Some(path), vec![first.clone()], surface)
    }
}
