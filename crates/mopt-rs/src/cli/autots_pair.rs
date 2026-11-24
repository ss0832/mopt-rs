//! Two parts of `run_autots` that look at more than one step: the searches between two
//! structures (`ieipmain`) as the method of step 1, and the comparison of the levels of
//! steps 3 and 4.
//!
//! **Step 1 between two structures** (`"step1_method": "gsm"` and so on). The input holds the
//! reactant and the product (its first and its last structure). The search runs with the
//! settings of step 1, which are settings of `ieipmain` then, and its path from the reactant
//! to the product (`<stem>_path.xyz`, `cli::ieipmain_path`) is the first path that step 2
//! relaxes. Every method of `ieipmain` between two structures can be named: iEIP, 2PSHS,
//! the growing Newton trajectory, the growing and the freezing string, DS-AFIR, the local
//! minimax method, the dimer, and the spring pair.
//!
//! **The levels of steps 3 and 4.** A transition state of step 3 is a stationary point of
//! the surface of step 3. When step 4 runs on another surface (another calculator, method,
//! basis set, state, model function, ONIOM setup, or bias potential), its IRC starts at a
//! point with a gradient: the run says so before it starts, and the report repeats it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's step 1 is an AFIR
//! optimization, and it does not compare the settings of its steps).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use super::autots_workflow::StepSettings;
use super::ieipmain::IeipmainConfig;
use super::optmain::OptmainConfig;
use super::spec::IEIPMAIN;
use crate::core_infra::json::JsonValue;
use crate::error::MoptResult;

/// A search of `ieipmain` between two structures that step 1 can use.
#[derive(Debug, PartialEq, Eq)]
pub struct PairMethod {
    /// Its name in `step1_method`.
    pub name: &'static str,
    /// The switch of `ieipmain` that selects it (`None`: iEIP, which runs without one).
    pub flag: Option<&'static str>,
    /// What it is, for the log and the report.
    pub title: &'static str,
}

impl PairMethod {
    /// What the second input contributes when it sets a direction instead of growing a branch.
    pub fn seed_note(&self) -> Option<&'static str> {
        match self.name {
            "dimer" | "spm" | "gnt" => Some("The search advances from the reactant towards the product. The product anchors the last segment of the initial band; this segment still needs relaxation."),
            "lmm" => Some("The initial band passes through the minimax estimate between the two inputs. The segments on either side still need relaxation."),
            _ => None,
        }
    }
}

/// The searches by their names.
pub const PAIR_METHODS: [PairMethod; 9] = [
    PairMethod {
        name: "ieip",
        flag: None,
        title: "improved elastic image pair method (ieipmain)",
    },
    PairMethod {
        name: "2pshs",
        flag: Some("use_2pshs"),
        title: "two-point scaled hypersphere search (ieipmain -2pshs)",
    },
    PairMethod {
        name: "gnt",
        flag: Some("use_gnt"),
        title: "growing Newton trajectory (ieipmain -gnt)",
    },
    PairMethod {
        name: "gsm",
        flag: Some("use_gsm"),
        title: "growing string method (ieipmain -gsm)",
    },
    PairMethod {
        name: "fsm",
        flag: Some("use_fsm"),
        title: "freezing string method (ieipmain -fsm)",
    },
    PairMethod {
        name: "dsafir",
        flag: Some("use_dsafir"),
        title: "double-sphere AFIR (ieipmain -dsafir)",
    },
    PairMethod {
        name: "lmm",
        flag: Some("use_lmm"),
        title: "local minimax method (ieipmain -lmm)",
    },
    PairMethod {
        name: "dimer",
        flag: Some("use_dimer"),
        title: "dimer method (ieipmain -use_dimer)",
    },
    PairMethod {
        name: "spm",
        flag: Some("use_spm"),
        title: "spring pair method (ieipmain -use_spm)",
    },
];

/// The search called `name` (letters in any case; `-` and `_` do not count, so `DS-AFIR`
/// is `dsafir`).
pub fn named(name: &str) -> Option<&'static PairMethod> {
    let plain: String = name
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .flat_map(char::to_lowercase)
        .collect();
    PAIR_METHODS.iter().find(|m| m.name == plain)
}

/// The names of the searches, for messages.
pub fn names() -> String {
    PAIR_METHODS
        .iter()
        .map(|m| m.name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Configure `method` between the structures in the files `reactant` and `product` with the
/// settings of a step, in `dir`.
pub fn configuration(
    method: &PairMethod,
    settings: &StepSettings,
    reactant: &Path,
    product: &Path,
) -> MoptResult<IeipmainConfig> {
    let mut extra = Vec::new();
    if let Some(flag) = method.flag {
        extra.push((flag, JsonValue::Bool(true)));
    }
    let args = super::autots::step_args(settings, &[reactant, product], &extra, &[], IEIPMAIN)?;
    IeipmainConfig::from_args(&args)
}

/// Run the configured pair search in its own directory.
pub fn search(
    method: &PairMethod,
    settings: &StepSettings,
    reactant: &Path,
    product: &Path,
    dir: &Path,
    console: Option<crate::core_infra::log::Level>,
    registry: &crate::pes::registry::CalculatorRegistry,
) -> MoptResult<super::ieipmain::IeipmainResult> {
    super::ieipmain::run_with(
        &configuration(method, settings, reactant, product)?,
        dir,
        console,
        registry,
    )
}

/// In what the surfaces of two optimizations differ (calculator and its settings, state,
/// model function, ONIOM, bias potentials), as words for a message; empty when they are
/// the same.
pub fn surface_differences(first: &OptmainConfig, second: &OptmainConfig) -> Vec<String> {
    let mut out = Vec::new();
    let mut differ = |what: &str, a: String, b: String| {
        if a != b {
            out.push(format!("{what} ({a} against {b})"));
        }
    };
    let shown = |text: &Option<String>| text.clone().unwrap_or_else(|| "the default".to_string());
    let (a, b) = (&first.calculator_options, &second.calculator_options);
    differ(
        "the calculator",
        first.calculator.clone(),
        second.calculator.clone(),
    );
    differ("the method (-func)", shown(&a.method), shown(&b.method));
    differ("the basis set (-bs)", shown(&a.basis), shown(&b.basis));
    differ(
        "the basis sets of elements (-sub_bs)",
        format!("{:?}", a.sub_basis),
        format!("{:?}", b.sub_basis),
    );
    differ(
        "the effective core potentials (-ecp)",
        format!("{:?}", a.ecp),
        format!("{:?}", b.ecp),
    );
    differ(
        "the unrestricted reference (-u)",
        a.unrestricted.to_string(),
        b.unrestricted.to_string(),
    );
    differ(
        "the integration grid (-grid)",
        format!("{:?}", a.grid_level),
        format!("{:?}", b.grid_level),
    );
    differ(
        "the solvation",
        format!("{:?}", a.solvation),
        format!("{:?}", b.solvation),
    );
    differ(
        "the settings of the calculator (-calc_opt)",
        format!("{:?}", a.extra),
        format!("{:?}", b.extra),
    );
    differ(
        "the ONIOM setup",
        format!("{:?}", a.oniom),
        format!("{:?}", b.oniom),
    );
    differ(
        "charge and multiplicity",
        format!("{:?} {:?}", first.state.charge, first.state.multiplicity),
        format!("{:?} {:?}", second.state.charge, second.state.multiplicity),
    );
    differ(
        "the electronic state (-es)",
        first.excited_state.to_string(),
        second.excited_state.to_string(),
    );
    differ(
        "the model function (-mf)",
        format!("{:?}", first.model_function),
        format!("{:?}", second.model_function),
    );
    differ(
        "the bias potentials",
        format!("{:?}", first.biases),
        format!("{:?}", second.biases),
    );
    differ(
        "the constraints (-fix, -pc, -gfix, -uc, -vc_fix)",
        super::mapper_report_data::describe_constraints(&first.settings.constraints),
        super::mapper_report_data::describe_constraints(&second.settings.constraints),
    );
    out
}

/// In what the surface of the chain of step 2 (`nebmain`) and that of an optimization differ:
/// the calculator and its settings, the state, the model function, the bias potentials and the
/// constraints ([`surface_differences`] for two optimizations).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn chain_surface_differences(
    chain: &super::nebmain_settings::NebmainConfig,
    second: &OptmainConfig,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut differ = |what: &str, a: String, b: String| {
        if a != b {
            out.push(format!("{what} ({a} against {b})"));
        }
    };
    differ(
        "the calculator",
        chain.calculator.clone(),
        second.calculator.clone(),
    );
    differ(
        "the settings of the calculator (-calc_opt)",
        format!("{:?}", chain.calculator_options.extra),
        format!("{:?}", second.calculator_options.extra),
    );
    differ(
        "the ONIOM setup",
        format!("{:?}", chain.calculator_options.oniom),
        format!("{:?}", second.calculator_options.oniom),
    );
    differ(
        "the electronic state (-es)",
        chain.excited_state.to_string(),
        second.excited_state.to_string(),
    );
    differ(
        "the model function (-mf)",
        format!("{:?}", chain.model_function),
        format!("{:?}", second.model_function),
    );
    differ(
        "the bias potentials",
        format!("{:?}", chain.biases),
        format!("{:?}", second.biases),
    );
    differ(
        "the constraints (-fix, -pc, -gfix, -uc, -vc_fix)",
        super::mapper_report_data::describe_constraints(&chain.constraints),
        super::mapper_report_data::describe_constraints(&second.settings.constraints),
    );
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cli::option::parser;
    use crate::cli::spec::OPTMAIN;

    #[test]
    fn the_searches_by_their_names() {
        assert_eq!(named("gsm").unwrap().flag, Some("use_gsm"));
        assert_eq!(named("DS-AFIR").unwrap().name, "dsafir");
        assert_eq!(named("ds_afir").unwrap().name, "dsafir");
        assert_eq!(named("2PSHS").unwrap().flag, Some("use_2pshs"));
        assert_eq!(named("iEIP").unwrap().flag, None);
        assert!(named("neb").is_none());
        assert!(names().starts_with("ieip, 2pshs, gnt, gsm, fsm, dsafir"));
        // Every switch is one of ieipmain.
        for method in &PAIR_METHODS {
            if let Some(flag) = method.flag {
                assert!(
                    IEIPMAIN.iter().any(|option| option.spec.key == flag),
                    "{flag}"
                );
            }
        }
    }

    #[test]
    fn what_differs_between_two_surfaces() {
        let config = |more: &[&str]| {
            let mut arguments = vec!["ts.xyz"];
            arguments.extend_from_slice(more);
            OptmainConfig::from_args(&parser("optmain", OPTMAIN).parse(arguments).unwrap()).unwrap()
        };
        let base = config(&["-os", "gfn1xtb", "-order", "1"]);
        // What only steers the optimization is no difference of the surface.
        let same = config(&[
            "-os", "gfn1xtb", "-order", "1", "-fc", "5", "-tcc", "-ns", "50",
        ]);
        assert!(surface_differences(&base, &same).is_empty());
        let other = config(&["-os", "gfn0xtb", "-elec", "1", "-ma", "100", "1", "2"]);
        let found = surface_differences(&base, &other);
        assert_eq!(found.len(), 3, "{found:?}");
        assert!(found[0].starts_with("the calculator (gfn1xtb against gfn0xtb)"));
        assert!(found[1].starts_with("charge and multiplicity"));
        assert!(found[2].starts_with("the bias potentials"));
        let level = config(&["-os", "pyscf", "-func", "b3lyp", "-bs", "def2-svp"]);
        let lower = config(&["-os", "pyscf", "-func", "hf", "-bs", "sto-3g"]);
        let found = surface_differences(&level, &lower);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(
            found[0].contains("b3lyp against hf") && found[1].contains("def2-svp against sto-3g")
        );
    }
}
