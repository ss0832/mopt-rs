//! Experimental members of the registry of two-point methods (new in mopt-rs, off unless
//! asked for by name): the local minimax method.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::registry::{Ends, PairInput, PairMethod, PairResult, PairSettings};
use super::registry_methods::{direction, from_outcome, go};
use super::surface::SearchSurface;
use super::{run_lmm, SearchRun};
use crate::error::MoptResult;
use crate::path::lmm::{LmmSettings, LocalMinimax};

/// The local minimax method from the first point towards the second (or along a direction).
pub(super) struct LmmMember;

/// The settings of the local minimax method from the keys of `lmm`.
pub fn lmm_settings(settings: &PairSettings, input: &PairInput) -> MoptResult<LmmSettings> {
    let d = LmmSettings::default();
    Ok(LmmSettings {
        max_iterations: settings.usize("iterations", d.max_iterations)?.max(1),
        first_step: settings.f64("first_step", d.first_step)?,
        sigma: settings.f64("sigma", d.sigma)?,
        rho: settings.f64("rho", d.rho)?,
        max_trials: settings.usize("trials", d.max_trials)?,
        barzilai_borwein: settings.flag("bb", d.barzilai_borwein)?,
        first_distance: match settings.text("first_distance") {
            Some(_) => input.length(settings.f64("first_distance", 1.0)?),
            None => match input.points.as_slice() {
                // Half way to the second point.
                [a, b, ..] => 0.5 * crate::math::linalg::col_norm(&(b - a)),
                _ => input.length(0.5),
            },
        },
        ray_step: match settings.text("ray_step") {
            Some(_) => input.length(settings.f64("ray_step", 0.15)?),
            None => input.length(0.15),
        },
        ray_tolerance: settings.f64("ray_tolerance", d.ray_tolerance)?,
        ray_evaluations: settings.usize("ray_evaluations", d.ray_evaluations)?.max(2),
        criteria: settings.criteria("criteria")?,
    })
}

impl PairMethod for LmmMember {
    fn name(&self) -> &'static str {
        "lmm"
    }
    fn summary(&self) -> &'static str {
        "local minimax method from a minimum (Li, Zhou 2001; experimental)"
    }
    fn ends(&self) -> Ends {
        Ends::OneOrTwo
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("iterations", "directions at most (200)"),
            ("first_step", "first trial step of the Armijo rule (0.1)"),
            ("sigma", "sigma of the Armijo rule (1e-4)"),
            ("rho", "rho of the Armijo rule (0.2)"),
            ("trials", "trials of the Armijo rule per iteration (8)"),
            ("bb", "trial steps of Barzilai and Borwein (on)"),
            (
                "first_distance",
                "first distance along the ray (half way to the second point)",
            ),
            (
                "ray_step",
                "step along a ray while the maximum is bracketed (0.15 Angstrom)",
            ),
            ("ray_tolerance", "|g . v| at which the peak is found (1e-4)"),
            ("ray_evaluations", "evaluations at most per ray (30)"),
            (
                "criteria",
                "force thresholds at the peak: normal, tight, loose (normal)",
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
        let s = lmm_settings(settings, input)?;
        let base = input.first()?;
        let v = direction(input, settings)?;
        run.record_settings(self.name(), &s)?;
        let mut m = LocalMinimax::new(s, base, &v, input.geometry.clone())?;
        let o = run_lmm(&mut m, surface, run, &mut |r| go(r))?;
        from_outcome(self.name(), o, None, vec![base.clone()], surface)
    }
}
