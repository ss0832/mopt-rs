//! Command-line options with MultiOptPy's flags, defaults, and whether this version implements
//! them.
//!
//! MultiOptPy: the `argparse` tables of `interface.py` (`optimizeparser`, `nebparser`, ...).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;

use crate::core_infra::arg_parser::{ArgArity, CliParser, OptSpec, ParsedArgs};
use crate::error::{MoptError, MoptResult};

/// One option of a command.
#[derive(Debug, Clone)]
pub struct CliOption {
    /// Flags, destination key, arity, and help.
    pub spec: OptSpec,
    /// MultiOptPy's default, as text.
    pub default: &'static str,
    /// Whether this version implements the option.
    pub implemented: bool,
    /// The feature that is missing when it is not implemented (e.g. `IRC`).
    pub feature: &'static str,
}

/// Parser for `options`.
pub fn parser(command: &str, options: &[CliOption]) -> CliParser {
    let mut p = CliParser::new(command, "mopt-rs (successor implementation of MultiOptPy)");
    for o in options {
        p.add_opt(o.spec.clone());
    }
    p
}

/// `--help` text with defaults and what is not implemented yet.
pub fn help(command: &str, options: &[CliOption], implemented: bool, feature: &str) -> String {
    let mut s = format!(
        "usage: {command} [options]\n\nmopt-rs {} (successor implementation of MultiOptPy)\n",
        crate::VERSION
    );
    if !implemented {
        let _ = writeln!(
            s,
            "\n`{command}` ({feature}) is not implemented yet in this version; its options are parsed only."
        );
    }
    s.push_str("\noptions:\n  -h, --help\n        show this help message and exit\n");
    s.push_str(
        "  --config FILE\n        read options from a JSON settings file (keys are the long \
         option names; the command line overrides it)\n  --dump-config FILE\n        write the \
         options in effect as a settings file and exit\n  --restart RUN_DIR\n        continue \
         an interrupted run (create RUN_DIR/STOP to interrupt one) with the settings and \
         evaluations recorded in RUN_DIR\n",
    );
    for o in options {
        let arg = match o.spec.arity {
            ArgArity::Flag => "",
            ArgArity::Single => " VALUE",
            ArgArity::Optional(_) => " [VALUE]",
            ArgArity::ZeroOrMore => " [VALUE ...]",
            ArgArity::OneOrMore => " VALUE [VALUE ...]",
        };
        let _ = writeln!(s, "  {}{arg}", o.spec.flags.join(", "));
        let mut line = format!("        {}", o.spec.help);
        if !o.default.is_empty() && o.spec.arity != ArgArity::Flag {
            let _ = write!(line, " (default: {})", o.default);
        }
        if !o.implemented {
            let _ = write!(line, " [not implemented yet: {}]", o.feature);
        }
        let _ = writeln!(s, "{line}");
    }
    s
}

fn is_default(parsed: &ParsedArgs, o: &CliOption) -> bool {
    if !parsed.is_present(o.spec.key) {
        return true;
    }
    match o.spec.arity {
        ArgArity::Flag => !parsed.get_bool(o.spec.key, false),
        _ => {
            let given = parsed.get_list(o.spec.key).join(" ");
            let d = o.default;
            given == d
                || (given.is_empty() && (d == "None" || d.is_empty()))
                || matches!((given.parse::<f64>(), d.parse::<f64>()), (Ok(a), Ok(b)) if a == b)
        }
    }
}

/// Reject options that are not implemented yet when they are used with a non-default value.
pub fn reject_unavailable(parsed: &ParsedArgs, options: &[CliOption]) -> MoptResult<()> {
    let used: Vec<String> = options
        .iter()
        .filter(|o| !o.implemented && !is_default(parsed, o))
        .map(|o| format!("{} ({})", o.spec.flags.join("/"), o.feature))
        .collect();
    if used.is_empty() {
        Ok(())
    } else {
        Err(MoptError::unsupported(format!(
            "not available in this version of mopt-rs: {}",
            used.join(", ")
        )))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cli::spec::OPTMAIN;

    #[test]
    fn unimplemented_options_are_rejected_only_when_used() {
        let p = parser("optmain", OPTMAIN);
        let ok = p.parse(["x.xyz", "-opt", "rsirfo_bfgs", "-gi", "1", "-modelhess"]);
        // -modelhess without a value takes MultiOptPy's implicit model.
        let parsed = ok.unwrap();
        assert_eq!(
            parsed.get_opt_str("use_model_hessian"),
            Some("fischerd3old")
        );
        assert!(reject_unavailable(&parsed, OPTMAIN).is_ok());
        let parsed = p.parse(["x.xyz", "-sqm1"]).unwrap();
        assert!(reject_unavailable(&parsed, OPTMAIN).is_err());
        let parsed = p.parse(["x.xyz", "-xtb", "GFN2-xTB"]).unwrap();
        assert!(reject_unavailable(&parsed, OPTMAIN).is_ok());
        let parsed = p
            .parse(["x.xyz", "-opt", "FIRELARS", "-gi", "1", "-fc", "-1"])
            .unwrap();
        assert!(reject_unavailable(&parsed, OPTMAIN).is_ok());
        let parsed = p.parse(["x.xyz", "-kp", "1.0", "1.5", "1,2"]).unwrap();
        assert!(reject_unavailable(&parsed, OPTMAIN).is_ok());
        let parsed = p.parse(["x.xyz", "-sqm1"]).unwrap();
        let e = reject_unavailable(&parsed, OPTMAIN)
            .unwrap_err()
            .to_string();
        assert!(e.contains("-sqm1"), "{e}");
    }

    #[test]
    fn help_lists_every_option() {
        let h = help("optmain", OPTMAIN, true, "geometry optimization");
        for o in OPTMAIN {
            assert!(h.contains(o.spec.flags[0]), "{}", o.spec.flags[0]);
        }
    }
}
