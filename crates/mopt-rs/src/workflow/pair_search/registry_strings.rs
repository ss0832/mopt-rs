//! The string members of the registry of two-point methods: the growing string (DE-GSM and
//! SE-GSM), the freezing string, and DS-AFIR.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::registry::{Ends, PairInput, PairMethod, PairResult, PairSettings};
use super::registry_methods::{from_outcome, go, interpolation};
use super::surface::SearchSurface;
use super::{run_dsafir, run_fsm, run_gsm, SearchRun};
use crate::error::{MoptError, MoptResult};
use crate::path::dsafir::{DoubleSphereAfir, DsAfirSettings, DsIntegrator};
use crate::path::fsm::{FreezingString, FsmSettings};
use crate::path::gsm::{Driving, DrivingCoordinate, GrowingString, GsmSettings};
use crate::path::interp::FirstPath;
use crate::path::string_nodes::NodeStepSettings;

/// The settings of the growing string from the keys of `gsm` (`single`: of `se_gsm`).
pub fn gsm_settings(
    settings: &PairSettings,
    input: &PairInput,
    single: bool,
) -> MoptResult<GsmSettings> {
    let defaults = GsmSettings::default();
    Ok(GsmSettings {
        nodes: settings.usize("nodes", if single { 20 } else { defaults.nodes })?,
        add_tolerance: settings.f64("add_tolerance", defaults.add_tolerance)?,
        node_steps: settings.usize("node_steps", defaults.node_steps)?.max(1),
        climb_tolerance: settings.f64("climb_tolerance", defaults.climb_tolerance)?,
        max_iterations: settings
            .usize("iterations", defaults.max_iterations)?
            .max(1),
        steps: NodeStepSettings {
            max_step: settings.f64("max_step", defaults.steps.max_step)?,
            first_scale: settings.f64("first_scale", defaults.steps.first_scale)?,
            ..defaults.steps
        },
        criteria: settings.criteria("criteria")?,
        interpolation: interpolation(settings, input, FirstPath::Linear)?,
        drive_step: settings.f64("drive_step", input.length(0.16))?,
        drop: settings.f64("drop", defaults.drop)?,
        final_steps: settings.usize("final_steps", defaults.final_steps)?,
    })
}

/// The settings of the freezing string from the keys of `fsm`.
pub fn fsm_settings(settings: &PairSettings, input: &PairInput) -> MoptResult<FsmSettings> {
    let defaults = FsmSettings::default();
    Ok(FsmSettings {
        nodes: settings.usize("nodes", defaults.nodes)?,
        gradients: settings.usize("gradients", defaults.gradients)?.max(1),
        steps: NodeStepSettings {
            max_component: settings.f64("max_component", input.length(0.3))?,
            first_scale: settings.f64("first_scale", defaults.steps.first_scale)?,
            ..defaults.steps
        },
        interpolation: interpolation(
            settings,
            input,
            if input.structures {
                FirstPath::Lst
            } else {
                FirstPath::Linear
            },
        )?,
        images: settings.usize("images", defaults.images)?.max(1),
    })
}

pub(super) struct GsmMember {
    pub(super) single: bool,
}

const GSM_KEYS: &[(&str, &str)] = &[
    (
        "nodes",
        "nodes of the string with its ends (9; SE-GSM: the most, 20)",
    ),
    (
        "add_tolerance",
        "RMS gradient across the string at which a frontier is settled (5e-3)",
    ),
    (
        "node_steps",
        "steps of a frontier before a node is added in any case (5)",
    ),
    (
        "climb_tolerance",
        "largest RMS gradient across the string at which the highest node climbs (2e-3)",
    ),
    ("iterations", "iterations at most (300)"),
    ("max_step", "largest step of a node (0.1 Bohr)"),
    (
        "first_scale",
        "length of the first step of a node per unit of gradient (1.0)",
    ),
    ("interpolation", "the gap of DE-GSM: linear or lst (linear)"),
    (
        "criteria",
        "force thresholds of the climbing node: normal, tight, loose (normal)",
    ),
    (
        "drive_step",
        "SE-GSM: growth step along the driving coordinates (0.16 Angstrom)",
    ),
    (
        "drop",
        "SE-GSM: the growth ends this far below the highest node (2e-3 Hartree)",
    ),
    (
        "final_steps",
        "SE-GSM: minimization steps of the last node (20)",
    ),
    (
        "driving",
        "SE-GSM: driving coordinates separated by ';' (ADD i j; BREAK i j; ...)",
    ),
];

impl PairMethod for GsmMember {
    fn name(&self) -> &'static str {
        if self.single {
            "se_gsm"
        } else {
            "gsm"
        }
    }
    fn summary(&self) -> &'static str {
        if self.single {
            "single-ended growing string along driving coordinates (Zimmerman 2015)"
        } else {
            "double-ended growing string with a climbing node (Peters et al. 2004; Zimmerman 2013)"
        }
    }
    fn ends(&self) -> Ends {
        if self.single {
            Ends::One
        } else {
            Ends::Two
        }
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        GSM_KEYS
    }
    fn run(
        &self,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
    ) -> MoptResult<PairResult> {
        let s = gsm_settings(settings, input, self.single)?;
        run.record_settings(self.name(), &s)?;
        let mut m = if self.single {
            let mut coordinates = input.driving.clone();
            if let Some(text) = settings.text("driving") {
                for part in text.split(';').filter(|p| !p.trim().is_empty()) {
                    coordinates.push(DrivingCoordinate::parse(part)?);
                }
            }
            let driving = Driving {
                coordinates,
                elements: input.elements.clone(),
            };
            GrowingString::single_ended(s, input.first()?, driving, input.geometry.clone())?
        } else {
            GrowingString::double_ended(
                s,
                input.first()?,
                input.second("gsm")?,
                input.geometry.clone(),
            )?
        };
        let o = run_gsm(&mut m, surface, run, &mut |r| go(r))?;
        let mut path = Vec::new();
        for (x, energy) in m.nodes() {
            // Redistribution can move a node after its last evaluation.
            let energy = match energy {
                Some(e) => e,
                None if o.interrupted => continue,
                None => super::registry_methods::energy_at(surface, &x)?,
            };
            path.push((x, energy));
        }
        let ends = match (path.first(), path.last()) {
            (Some(a), Some(b)) => vec![a.0.clone(), b.0.clone()],
            _ => Vec::new(),
        };
        from_outcome(self.name(), o, Some(path), ends, surface)
    }
}

pub(super) struct FsmMember;

impl PairMethod for FsmMember {
    fn name(&self) -> &'static str {
        "fsm"
    }
    fn summary(&self) -> &'static str {
        "freezing string (Behn, Zimmerman, Bell, Head-Gordon 2011)"
    }
    fn ends(&self) -> Ends {
        Ends::Two
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("nodes", "the interpolated path over the step (12)"),
            ("gradients", "evaluations per node (3)"),
            (
                "max_component",
                "largest component of a step (0.3 Angstrom)",
            ),
            (
                "first_scale",
                "length of the first step per unit of gradient (1.0)",
            ),
            (
                "interpolation",
                "lst for structures, linear (lst; plain variables: linear)",
            ),
            (
                "images",
                "images of an interpolation between the frontiers (10)",
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
        let s = fsm_settings(settings, input)?;
        run.record_settings(self.name(), &s)?;
        let mut m = FreezingString::new(
            s,
            input.first()?,
            input.second("fsm")?,
            input.geometry.clone(),
        )?;
        let o = run_fsm(&mut m, surface, run, &mut |r| go(r))?;
        let path = m.path();
        let ends = match (path.first(), path.last()) {
            (Some(a), Some(b)) => vec![a.0.clone(), b.0.clone()],
            _ => Vec::new(),
        };
        from_outcome(self.name(), o, Some(path), ends, surface)
    }
}

pub(super) struct DsAfirMember;

/// The DS-AFIR settings of `settings` on top of those of `variant` (`paper`: the method as
/// the paper states it; `default`: with the additions of mopt-rs that the measurements
/// chose).
pub fn dsafir_settings(settings: &PairSettings, input: &PairInput) -> MoptResult<DsAfirSettings> {
    let base = match settings.text("variant").as_deref() {
        None | Some("default") => DsAfirSettings::default(),
        Some("paper") => DsAfirSettings::paper(),
        Some(other) => {
            return Err(MoptError::parse(format!(
                "variant: paper or default, got '{other}'"
            )))
        }
    };
    Ok(DsAfirSettings {
        force: settings.f64("force", base.force)?,
        step: match settings.text("step") {
            Some(_) => input.length(settings.f64("step", 0.1)?),
            None => base.step,
        },
        separation: settings.f64("separation", input.length(0.12))?,
        max_steps: settings.usize("steps", base.max_steps)?.max(1),
        integrator: match settings.text("integrator").as_deref() {
            None => base.integrator,
            Some("lqa") => DsIntegrator::Lqa,
            Some("euler") => DsIntegrator::Euler,
            Some(other) => {
                return Err(MoptError::parse(format!(
                    "integrator: lqa or euler, got '{other}'"
                )))
            }
        },
        hessian_every: settings.usize("hessian_every", base.hessian_every)?,
        afir_curvature: settings.flag("afir_curvature", base.afir_curvature)?,
        attraction_only: settings.flag("attraction_only", base.attraction_only)?,
        align: settings.flag("align", base.align)?,
        alternate: settings.flag("alternate", base.alternate)?,
        quasi_newton: settings.flag("quasi_newton", base.quasi_newton)?,
        relax_steps: settings.usize("relax", base.relax_steps)?,
        relax_tolerance: settings.f64("relax_tolerance", base.relax_tolerance)?,
        relax_steps_settings: base.relax_steps_settings,
    })
}

impl PairMethod for DsAfirMember {
    fn name(&self) -> &'static str {
        "dsafir"
    }
    fn summary(&self) -> &'static str {
        "double-sphere AFIR between two structures (Maeda et al. 2018)"
    }
    fn ends(&self) -> Ends {
        Ends::Two
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            (
                "variant",
                "paper (as the paper states it) or default (with the measured additions)",
            ),
            (
                "force",
                "delta, the slope of the AFIR function along u (Hartree/Bohr)",
            ),
            ("step", "length of a step of an end (Bohr)"),
            (
                "integrator",
                "lqa (the local quadratic approximation, the paper's) or euler (fixed steps)",
            ),
            (
                "hessian_every",
                "with lqa: an exact Hessian at every N-th point of an end, updated in between \
                 (10; 0: at the first point only)",
            ),
            (
                "afir_curvature",
                "with lqa: the second derivatives of the distance terms in the Hessian",
            ),
            (
                "attraction_only",
                "leave the second term of the AFIR function out (Y = 1)",
            ),
            ("separation", "the ends meet below this (0.12 Angstrom)"),
            ("steps", "steps of the ends at most (2000)"),
            (
                "align",
                "lay the other end onto the moving one before every step",
            ),
            ("alternate", "move the two ends in turn"),
            (
                "quasi_newton",
                "L-BFGS steps down the AFIR function instead of Euler steps",
            ),
            (
                "relax",
                "steps of the moving end across its drive after each drive step (0)",
            ),
            (
                "relax_tolerance",
                "RMS gradient across the drive that ends them (1e-3)",
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
        let s = dsafir_settings(settings, input)?;
        run.record_settings(self.name(), &s)?;
        let mut m = DoubleSphereAfir::new(
            s,
            input.first()?,
            input.second("dsafir")?,
            input.geometry.clone(),
        )?;
        if let Some(map) = &input.step_map {
            m = m.with_step_map(map.clone());
        }
        let o = run_dsafir(&mut m, surface, run, &mut |r| go(r))?;
        let path = m.path();
        let ends = vec![
            input.first()?.clone(),
            input
                .geometry
                .laid_onto(input.second("dsafir")?, input.first()?),
        ];
        from_outcome(self.name(), o, Some(path), ends, surface)
    }
}
