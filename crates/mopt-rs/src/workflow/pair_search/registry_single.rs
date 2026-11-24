//! The members of the registry of two-point methods that start from one point with a
//! direction: the dimer, the spring pair, and ADDF.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::registry::{Ends, PairInput, PairMethod, PairResult, PairSettings};
use super::registry_methods::{direction, from_outcome, go, removed, sphere};
use super::surface::SearchSurface;
use super::{run_addf, run_dimer, SearchRun};
use crate::error::MoptResult;
use crate::path::dimer::{Dimer, DimerSettings};
use crate::path::spring_pair::{SpringPair, SpringPairSettings};
use crate::workflow::chain::Flow;
use crate::workflow::explore::addf::{Addf, AddfSettings};

pub(super) struct DimerMember;

impl PairMethod for DimerMember {
    fn name(&self) -> &'static str {
        "dimer"
    }
    fn summary(&self) -> &'static str {
        "dimer method (Henkelman, Jonsson 1999; Kastner, Sherwood 2008)"
    }
    fn ends(&self) -> Ends {
        Ends::OneOrTwo
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("separation", "half length of the dimer (1e-4 Bohr)"),
            ("trial_angle", "trial rotation (pi/32 rad)"),
            ("iterations", "translations at most (1000)"),
            ("rotations", "rotations per translation (8)"),
            (
                "rotation_angle",
                "rotation below which the rotations stop (1 degree)",
            ),
            ("step", "largest translation (0.1 Bohr)"),
            (
                "criteria",
                "force thresholds: normal, tight, loose (normal)",
            ),
            (
                "noise",
                "random displacement of the second structure (0.05 Bohr; 0 for variables)",
            ),
            ("seed", "seed of that displacement or of a random axis (1)"),
            (
                "translation",
                "lbfgs, or experimental: plain, optimistic, extragradient (lbfgs)",
            ),
            (
                "rate_across",
                "step per unit of the gradient across the axis (0.5)",
            ),
            (
                "rate_along",
                "step per unit of the gradient along the axis (0.5)",
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
        let d = DimerSettings::default();
        let s = DimerSettings {
            separation: settings.f64("separation", d.separation)?,
            trial_angle: settings.f64("trial_angle", d.trial_angle)?,
            max_iterations: settings.usize("iterations", d.max_iterations)?.max(1),
            max_rotations: settings.usize("rotations", d.max_rotations)?,
            rotation_tolerance: settings.f64("rotation_angle", 1.0)?.to_radians(),
            max_step: settings.f64("step", d.max_step)?,
            criteria: settings.criteria("criteria")?,
            translation: match settings.text("translation") {
                Some(name) => crate::path::dimer::DimerTranslation::from_name(&name)?,
                None => d.translation,
            },
            rate_across: settings.f64("rate_across", d.rate_across)?,
            rate_along: settings.f64("rate_along", d.rate_along)?,
            ..d
        };
        let first = input.first()?;
        let axis = direction(input, settings)?;
        run.record_settings(self.name(), &s)?;
        let mut m = Dimer::new(s, first, &axis)?.with_frozen(input.geometry.frozen.clone())?;
        if let Some(t) = &input.geometry.rigid_body {
            m = m.with_rigid_body(t.clone())?;
        }
        let mut path = Vec::new();
        let o = run_dimer(&mut m, surface, run, &mut |r| {
            path.push((r.center.clone(), r.energy));
            go(r)
        })?;
        from_outcome(self.name(), o, Some(path), vec![first.clone()], surface)
    }
}

pub(super) struct SpmMember;

impl PairMethod for SpmMember {
    fn name(&self) -> &'static str {
        "spm"
    }
    fn summary(&self) -> &'static str {
        "spring pair method (Cui, Jiang 2024)"
    }
    fn ends(&self) -> Ends {
        Ends::OneOrTwo
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("length", "natural length of the spring (0.1)"),
            (
                "offset",
                "distance of the second particle at the start (0.1)",
            ),
            ("drift", "rate of the drift (0.5)"),
            ("spring", "rate of the spring (1)"),
            ("climb", "rate of the climb (5)"),
            (
                "tolerance",
                "force across the line that ends a drift (2e-3)",
            ),
            ("drift_steps", "drift steps per cycle (200)"),
            ("cycles", "cycles at most (999)"),
            ("step", "largest step of a particle (0.1)"),
            (
                "criteria",
                "force thresholds: normal, tight, loose (normal)",
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
        let d = SpringPairSettings::default();
        let s = SpringPairSettings {
            spring_length: settings.f64("length", d.spring_length)?,
            offset: settings.f64("offset", d.offset)?,
            drift_rate: settings.f64("drift", d.drift_rate)?,
            spring_rate: settings.f64("spring", d.spring_rate)?,
            climb_rate: settings.f64("climb", d.climb_rate)?,
            drift_tolerance: settings.f64("tolerance", d.drift_tolerance)?,
            drift_steps: settings.usize("drift_steps", d.drift_steps)?.max(1),
            max_cycles: settings.usize("cycles", d.max_cycles)?.max(1),
            max_step: settings.f64("step", d.max_step)?,
            criteria: settings.criteria("criteria")?,
            align: input.geometry.aligns() && input.structures,
        };
        let first = input.first()?;
        let toward = direction(input, settings)?;
        run.record_settings(self.name(), &s)?;
        let mut m = SpringPair::new(s, first, &toward)?;
        if !input.geometry.frozen.is_empty() {
            m = m.with_frozen(input.geometry.frozen.clone())?;
        }
        let (o, path) = super::point_runs::spring_pair_path(&mut m, surface, run)?;
        from_outcome(self.name(), o, Some(path), vec![first.clone()], surface)
    }
}

pub(super) struct AddfMember;

impl PairMethod for AddfMember {
    fn name(&self) -> &'static str {
        "addf"
    }
    fn summary(&self) -> &'static str {
        "anharmonic downward distortion following from a minimum (Ohno, Maeda 2004)"
    }
    fn ends(&self) -> Ends {
        Ends::One
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            (
                "initial",
                "displacement along the stiffest mode on the first sphere (0.03 Angstrom)",
            ),
            (
                "step",
                "growth of the radius per sphere along the stiffest mode (0.1 Angstrom)",
            ),
            ("spheres", "spheres at most (300)"),
            ("paths", "largest ADDs followed (5; 0: all)"),
            ("angle", "angle within which two ADDs are one (0.01 rad)"),
            (
                "gradient",
                "largest gradient component of the minimum (1e-3)",
            ),
            (
                "sphere_iterations",
                "evaluations of a minimization on a sphere (30)",
            ),
            (
                "sphere_gradient",
                "RMS gradient within the sphere that ends it (1e-4)",
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
        let s = AddfSettings {
            initial_length: input.length(settings.f64("initial", 0.03)?),
            step_length: input.length(settings.f64("step", 0.1)?),
            max_spheres: settings.usize("spheres", 300)?.max(1),
            paths: settings.usize("paths", 5)?,
            sphere: sphere(settings)?,
            duplicate_angle: settings.f64("angle", 0.01)?,
            gradient_tolerance: settings.f64("gradient", 1e-3)?,
        };
        let first = input.first()?;
        run.record_settings(self.name(), &s)?;
        let mut m = Addf::new(s, first, removed(input)?)?;
        if let Some(test) = input.fragments.clone() {
            m = m.with_dissociation_test(Box::new(move |x| test(x)));
        }
        let o = run_addf(&mut m, surface, run, &mut |_| Ok(Flow::Continue))?;
        Ok(PairResult {
            method: self.name().to_string(),
            converged: !o.transition_states.is_empty(),
            end: format!("{} transition state estimate(s)", o.transition_states.len()),
            transition_states: o
                .transition_states
                .into_iter()
                .map(|(_, x, e)| (x, e))
                .collect(),
            path: None,
            ends: vec![first.clone()],
            counts: Default::default(),
            interrupted: o.interrupted,
        })
    }
}
