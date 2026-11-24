//! The settings of the members of the registry of two-point methods that are those of a
//! command (the chain relaxation of `nebmain` and `nebmain -dmf`, BITSS of `optmain -mf
//! bitss`), read from the keys of the member with the parsers of these commands
//! ([`crate::workflow::pair_search::CommandSettings`]): the command layer hands them to the
//! registry with each run, so that the workflow layer does not use the command line.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use super::nebmain_settings::{NebmainConfig, DEFAULT_FIRST_PATH};
use super::option::parser;
use super::settings::{merge, parse_settings};
use super::spec::NEBMAIN;
use crate::core_infra::arg_parser::ParsedArgs;
use crate::error::{MoptError, MoptResult};
use crate::path::bitss::BitssSettings;
use crate::workflow::pair_search::{ChainCommandSettings, CommandSettings, PairSettings};

/// The parsers of the commands.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommandParsers;

/// The parsers every run of the registry is given.
pub static COMMAND_PARSERS: CommandParsers = CommandParsers;

impl CommandSettings for CommandParsers {
    fn chain(
        &self,
        settings: &PairSettings,
        dir: &Path,
        dmf: bool,
    ) -> MoptResult<ChainCommandSettings> {
        let nebmain = parser("nebmain", NEBMAIN);
        // nebmain wants an input that exists: the run directory stands in (it is not read).
        let input = dir.display().to_string();
        let file = match settings.text("settings") {
            Some(text) => parse_settings(&text, NEBMAIN)
                .map_err(|e| MoptError::parse(format!("the settings of nebmain: {e}")))?,
            None => ParsedArgs::default(),
        };
        let mut tokens = vec![input];
        if let Some(iterations) = settings.text("iterations") {
            tokens.push("-ns".to_string());
            tokens.push(iterations);
        }
        if dmf {
            tokens.push("-dmf".to_string());
        }
        if let Some(path) = settings.text("path") {
            tokens.push("-neb_opt".to_string());
            tokens.push(format!("path={path}"));
        }
        if let Some(options) = settings.text("options") {
            tokens.extend(options.split_whitespace().map(str::to_string));
        }
        let p = merge(file, nebmain.parse(&tokens)?);
        let config = NebmainConfig::from_args(&p)?;
        Ok(ChainCommandSettings {
            chain: config.chain,
            dmf: config.dmf,
            model_hessian: config.model_hessian,
            exact_hessians: config.exact_hessians,
            first_path: config.first_path.unwrap_or(DEFAULT_FIRST_PATH),
            first_equal_spacing: config.first_equal_spacing,
        })
    }

    fn bitss(&self, tokens: &[String]) -> MoptResult<BitssSettings> {
        super::bitss::parse_settings(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step2_neb_uses_nebmain_limits_and_preserves_overrides() -> MoptResult<()> {
        let dir = Path::new(".");
        for (settings, iterations, tolerance) in [
            ("{}", 30, Some(3e-4)),
            (r#"{"-ns":7,"neb_options":["tolerance=0"]}"#, 7, None),
        ] {
            let settings = settings.to_string();
            let pair = PairSettings::new(vec![("settings".into(), settings.clone())]);
            let chain = COMMAND_PARSERS.chain(&pair, dir, false)?.chain;
            assert_eq!(chain.max_iterations, iterations);
            assert_eq!(chain.force_tolerance, tolerance);
            let pair = PairSettings::new(vec![
                ("settings".into(), settings),
                ("iterations".into(), "12".into()),
            ]);
            assert_eq!(
                COMMAND_PARSERS
                    .chain(&pair, dir, false)?
                    .chain
                    .max_iterations,
                12
            );
        }
        for (block, limit) in [("{}", 60), (r#"{"NSTEP":9}"#, 9)] {
            let pair = PairSettings::new(vec![("settings".into(), block.into())]);
            let parsed = COMMAND_PARSERS.chain(&pair, dir, true)?;
            assert_eq!(parsed.dmf.unwrap().max_iterations, limit);
        }
        Ok(())
    }
}
