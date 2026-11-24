//! `mdmain -md_opt velocities=wigner`: the structure and the velocities of the start drawn
//! from the harmonic Wigner distribution of the normal modes of the input structure (a
//! minimum, or a transition state for trajectories from it), at `-temp` or
//! `wigner_temperature=` (0: the vibrational ground state, for photodynamics). Settings:
//! `wigner_low=CUTOFF[:wigner|classical|zero]` (modes below CUTOFF cm^-1), and
//! `wigner_imaginary=boltzmann|forward|backward|zero` (the momentum along imaginary modes).
//! The Hessian is that of the calculator with the static bias potentials at the input
//! structure (in the journal of the run); the modes are those of the space the constraints
//! leave (resolved at the input structure, so that frozen atoms and the values held "at the
//! start" are those of the input, not of the sample), and rigid motions are left out only
//! where the surface is blind to them. Every trajectory of `-traj` draws its own sample from
//! its seed.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::Col;

use crate::core_infra::log::Logger;
use crate::core_infra::rng::Rng;
use crate::domain::{EvalRequest, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::md::wigner::{
    angular_frequency, mean_energy, sample, treatment, ImaginaryModes, LowModes, Treatment,
    WignerSettings,
};
use crate::pes::calculator::Calculator;
use crate::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use crate::pes::term::{check_term_output, History, PesTerm, TermContext};
use crate::vib::{transverse_modes, Direction};

/// The settings of the sampling from `-md_opt`, the temperature given or not.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WignerOption {
    /// The distribution (its temperature is replaced by `-temp` when none was given).
    pub settings: WignerSettings,
    /// Whether `wigner_temperature=` was given.
    pub temperature_given: bool,
}

fn number(key: &str, value: &str) -> MoptResult<f64> {
    match value.trim().parse::<f64>() {
        Ok(x) if x.is_finite() && x >= 0.0 => Ok(x),
        _ => Err(MoptError::parse(format!(
            "-md_opt {key}={value}: a number not below 0"
        ))),
    }
}

/// Apply the `wigner_*` setting `key=value`; whether `key` was one.
pub fn apply_wigner_key(w: &mut WignerOption, key: &str, value: &str) -> MoptResult<bool> {
    match key {
        "wigner_temperature" => {
            w.settings.temperature = number(key, value)?;
            w.temperature_given = true;
        }
        "wigner_low" => {
            let (cutoff, how) = value.split_once(':').unwrap_or((value, "classical"));
            w.settings.low_cutoff = number(key, cutoff)?;
            w.settings.low_modes = match how.trim().to_ascii_lowercase().as_str() {
                "wigner" => LowModes::Wigner,
                "classical" => LowModes::Classical,
                "zero" => LowModes::Zero,
                other => {
                    return Err(MoptError::parse(format!(
                        "-md_opt wigner_low=CUTOFF:{other}: wigner, classical, or zero"
                    )))
                }
            };
        }
        "wigner_imaginary" => {
            w.settings.imaginary = match value.trim().to_ascii_lowercase().as_str() {
                "boltzmann" => ImaginaryModes::Boltzmann,
                "forward" => ImaginaryModes::Forward,
                "backward" => ImaginaryModes::Backward,
                "zero" => ImaginaryModes::Zero,
                other => {
                    return Err(MoptError::parse(format!(
                        "-md_opt wigner_imaginary={other}: boltzmann, forward, backward, or zero"
                    )))
                }
            };
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn name(t: Treatment) -> &'static str {
    match t {
        Treatment::Wigner => "Wigner",
        Treatment::Classical => "classical",
        Treatment::Zero => "zero",
        Treatment::Imaginary => "imaginary",
    }
}

/// What the surface of the sampling is: the bias potentials on the calculator, the
/// constraints of the run (resolved at the input structure), and whether the surface is
/// blind to rigid motions of the whole structure (`pes::term::frame_invariant`).
pub struct WignerSurface<'a> {
    /// The bias potentials of the run; those that do not change with the time or with what
    /// they have seen are part of the Hessian of the sampling.
    pub terms: &'a [(String, Arc<dyn PesTerm>)],
    /// The constraints, resolved at the input structure.
    pub constraints: &'a [Constraint],
    /// Whether the calculator and the terms are blind to rigid motions.
    pub frame_invariant: bool,
}

/// The start drawn from the Wigner distribution of the modes of `start` on `calc` with the
/// static terms of `surface`, with the random numbers of `seed`: the structure and the
/// velocities. The modes are those of the space the constraints leave: frozen atoms are not
/// displaced and have no velocity, displacements and velocities have no part along the
/// gradients of the other constraints at `start` (the dynamics puts the structure back onto
/// them and takes the constrained part out of the velocities at its start), and rigid motions
/// are left out only where the surface is blind to them and no constraint pins the structure.
pub fn wigner_start(
    option: &WignerOption,
    temperature: f64,
    seed: u64,
    start: &MolecularSystem,
    calc: &dyn Calculator,
    surface: &WignerSurface<'_>,
    log: &Logger,
) -> MoptResult<(MolecularSystem, Col<f64>)> {
    let mut s = option.settings;
    if !option.temperature_given {
        s.temperature = temperature;
    }
    let out = calc.evaluate(start, &EvalRequest::ENERGY_GRAD_HESS)?;
    let mut hessian = out.hessian_bohr.ok_or_else(|| {
        MoptError::calc("Wigner sampling needs the Hessian, and the calculator gave none")
    })?;
    let mut g = out.gradient_bohr;
    let ctx = TermContext::at_iteration(0, start);
    let mut left_out = Vec::new();
    for (label, term) in surface.terms {
        if term.history() != History::Static {
            left_out.push(label.as_str());
            continue;
        }
        let bias = term.evaluate(start, &ctx, true).context(label.clone())?;
        check_term_output(label, &bias)?;
        let of_bias = bias.hessian_bohr.as_ref().ok_or_else(|| {
            MoptError::calc(format!("bias potential '{label}' returned no Hessian"))
        })?;
        hessian += of_bias;
        g += &bias.gradient_bohr;
    }
    if !left_out.is_empty() {
        log.info(&format!(
            "Wigner sampling: the Hessian without the bias potentials that change with the \
             time or with what they have seen ({}).",
            left_out.join(", ")
        ));
    }
    // The directions the constraints leave no motion along (the eigenvector constraints are
    // resolved by the dynamics at its first Hessian).
    let held: Vec<Constraint> = surface
        .constraints
        .iter()
        .filter(|c| !c.is_pending())
        .cloned()
        .collect();
    let mut directions = Vec::new();
    let mut pinned = false;
    if !held.is_empty() {
        let set = ConstraintSet::new(&held, start, None).context("constraints")?;
        pinned = set.breaks_rigid_body();
        let cf = set.frame(start, &g)?;
        g = cf.project_gradient(&g);
        let n = start.dof();
        for k in 0..cf.jacobian.nrows() {
            directions.push(Direction::Gradient(Col::from_fn(n, |i| {
                cf.jacobian[(k, i)]
            })));
        }
    }
    let rms = libm::sqrt(g.iter().map(|v| v * v).sum::<f64>() / g.nrows().max(1) as f64);
    if rms > 1e-3 {
        log.warn(&format!(
            "Wigner sampling: the input structure is not stationary (RMS gradient {rms:.2e} \
             Hartree/Bohr, the constraint forces left out); the harmonic distribution belongs \
             to a minimum or a saddle point."
        ));
    }
    let rigid = surface.frame_invariant && !pinned;
    let project = rigid && !start.has_pbc();
    if rigid && start.has_pbc() {
        // In a periodic cell only the translations are free.
        let n = start.dof();
        for axis in 0..3 {
            directions.push(Direction::Displacement(Col::from_fn(n, |i| {
                f64::from(u8::from(i % 3 == axis))
            })));
        }
    }
    let modes = transverse_modes(start, &hessian, project, &directions)?;
    log.info(&format!(
        "Wigner sampling: modes of the space left by {} constraint direction(s){}.",
        directions.len() - if rigid && start.has_pbc() { 3 } else { 0 },
        match (rigid, start.has_pbc()) {
            (false, _) => "; rigid motions kept (the surface or a constraint fixes the frame)",
            (true, false) => " and the rigid motions",
            (true, true) => " and the translations",
        }
    ));
    // A stream of its own, apart from that of the thermostat.
    let mut rng = Rng::from_seed(seed.rotate_left(17) ^ 0x5749_474e_4552);
    let mut drawn = sample(&start.coords_flat_bohr(), &modes, &s, &mut rng)?;
    // Frozen atoms stay exactly where they are, without velocity.
    let x0 = start.coords_flat_bohr();
    for c in &held {
        if let ConstraintKind::Frozen(atoms) = &c.kind {
            for atom in atoms {
                for k in 0..3 {
                    drawn.x[atom.dof(k)] = x0[atom.dof(k)];
                    drawn.v[atom.dof(k)] = 0.0;
                }
            }
        }
    }
    log.info(&format!(
        "Wigner sampling at {} K ({} modes; low modes below {} cm^-1: {:?}; imaginary modes: \
         {:?}); reference: M. Hillery, R. F. O'Connell, M. O. Scully, E. P. Wigner, Phys. Rep. \
         1984, 106, 121",
        s.temperature,
        modes.wavenumbers.len(),
        s.low_cutoff,
        s.low_modes,
        s.imaginary
    ));
    let kcal = Units::HARTREE_TO_KCAL_MOL;
    let mut total = (0.0, 0.0);
    for (k, &nu) in modes.wavenumbers.iter().enumerate() {
        let t = treatment(&modes, k, &s);
        let mean = match t {
            Treatment::Wigner => mean_energy(angular_frequency(modes.curvatures[k]), s.temperature),
            Treatment::Classical => Units::BOLTZMANN_HARTREE_PER_K * s.temperature,
            Treatment::Zero => 0.0,
            Treatment::Imaginary => 0.5 * Units::BOLTZMANN_HARTREE_PER_K * s.temperature,
        };
        total.0 += drawn.energies[k];
        total.1 += mean;
        log.info(&format!(
            "  mode {:>3} {:>10.2} cm^-1 {:<9} energy {:>10.4} kcal/mol (mean of the distribution {:.4})",
            k + 1,
            nu,
            name(t),
            drawn.energies[k] * kcal,
            mean * kcal
        ));
    }
    log.info(&format!(
        "Wigner sampling: vibrational energy {:.4} kcal/mol (mean of the distribution {:.4})",
        total.0 * kcal,
        total.1 * kcal
    ));
    Ok((start.with_flat_coords_bohr(&drawn.x)?, drawn.v))
}
