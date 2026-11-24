//! Effective settings used by the AutoTS stages, including command and solver defaults.
//!
//! MultiOptPy: `Entrypoints/autots.py` configures the stages. The effective-settings report
//! is specific to mopt-rs.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::autots::{AutotsConfig, FirstPathMethod};
use super::ieipmain::IeipmainConfig;
use super::nebmain::NebmainConfig;
use super::optmain::OptmainConfig;
use std::collections::BTreeMap;
use std::fmt::Write as _;

pub(super) fn workflow(c: &AutotsConfig, variables: bool) -> String {
    let first = match c.first_method {
        _ if c.skip_to_step4 || c.skip_step1 => "skipped",
        FirstPathMethod::Bitss => "BITSS",
        FirstPathMethod::Pair(m) => m.name,
        FirstPathMethod::Bias if variables => super::autots_variables::DEFAULT_FIRST_METHOD,
        FirstPathMethod::Bias => "bias-driven optimization",
    };
    format!("step1_method = {first}\nstep2_method = {:?}\nwork_dir = {}\ntop_n_candidates = {}\ncheck_frequencies = {}\nhtml_report = {}\nskip_step1 = {}\nrun_step4 = {}\nskip_to_step4 = {}\nindependent_paths = {}",
        c.path_method, c.work_dir.display(), c.top_n, c.check_frequencies, c.html_report,
        c.skip_step1, c.run_step4, c.skip_to_step4, c.workflow.as_ref().is_some_and(|w| w.independent_paths))
}

pub(super) fn text(settings: &BTreeMap<String, String>) -> String {
    settings
        .iter()
        .map(|(label, values)| format!("[{label}]\n{values}"))
        .collect::<Vec<_>>()
        .join("\n\n")
}

// Read from the typed configuration passed to the command. CLI defaults printed
// in --help can differ from the defaults selected for a particular calculation.
macro_rules! fields {
    ($config:ident; $($field:ident),+ $(,)?) => {{
        let mut out = String::new();
        $(let _ = writeln!(out, "{} = {:#?}", stringify!($field), $config.$field);)+
        out
    }};
}

pub(super) fn optimization(c: &OptmainConfig) -> String {
    let mut out = fields!(c; inputs, restart_dir, calculator, calculator_options, keep_full_path, state, settings, hisd_mode_kill, model_function, excited_state, initial_hessian, follow, shape_conditions, geometry_info, detect_negative_eigenvalues, debug, variable_cell, cmds, pca, koopman, threads, no_symmetry, frequency_analysis, force_frequency_analysis, frequency, project_gradient, ccqn_bonds, ccqn_product, ccqn_step, biases, bias_options, bias_derivatives, irc, stop_file, final_hessian);
    if c.model_function
        .first()
        .is_some_and(|m| m.eq_ignore_ascii_case("bitss"))
    {
        let _ = writeln!(out, "BITSS = {:#?}", super::bitss::effective_settings(c));
    }
    out
}

pub(super) fn chain(c: &NebmainConfig) -> String {
    fields!(c; inputs, restart_dir, calculator, calculator_options, state, excited_state, biases, constraints, model_function, chain, inserted, first_equal_spacing, spacing, bezier_spacing, model_hessian, exact_hessians, per_atom, figures, coordinates, surrogate, surrogate_length, surrogate_features, first_path, follow_up, dmf, threads)
}

pub(super) fn pair(c: &IeipmainConfig) -> String {
    fields!(c; input, inputs, restart_dir, method, surface_function, ignored, calculator, calculator_options, states, excited_states, max_iterations, beta, microiterations, gnt_bonds, gnt_step, gnt_corrector, addf_step, addf_spheres, addf_paths, twopshs_step, twopshs_spheres, dimer_separation, dimer_trial_angle, dimer_iterations, extra, biases, constraints, threads, debug, coordinates)
}
