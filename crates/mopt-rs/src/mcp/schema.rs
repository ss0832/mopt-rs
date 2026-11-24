//! The tool definitions: one tool per implemented command, generated at run time from the
//! option tables, so that the tools cannot differ from the command line; and the fixed tools
//! of the jobs and the components.
//!
//! The full schema of a command has one property per implemented option, named by the
//! option's long name (the key of settings files, `opt_method`), with the JSON type from its
//! arity (a flag: boolean; one value: string or number; an optional value: string, number, or
//! `true` for the flag alone; several values: array), the help text as description, and the
//! default. `INPUT` holds the positional inputs (paths relative to the root) and `files`
//! further files that options name. Options that are not implemented yet are not offered.
//!
//! `tools/list` gives the compact form by default ([`Listing::Compact`]): `INPUT`, `files`
//! and the few options of [`COMMON_OPTIONS`] that the command has, with every other option
//! accepted by its long name (the server checks each against the tables when the tool is
//! called); `command_help` gives the full schema and the help of each option. A server started
//! with `--full-schemas` lists the full schemas ([`Listing::Full`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use super::inputs;
use super::protocol::{object, text};
use crate::cli::option::CliOption;
use crate::core_infra::arg_parser::ArgArity;
use crate::core_infra::json::JsonValue;

/// Property of the positional inputs.
pub const INPUT_PROPERTY: &str = "INPUT";
/// Property of the further files copied into the job directory.
pub const FILES_PROPERTY: &str = "files";

/// A table of commands: `(command, implemented, feature, options)` as
/// [`crate::cli::spec::COMMANDS`].
pub type CommandTable<'a> = &'a [(&'a str, bool, &'a str, &'a [CliOption])];

/// Names of the tools of the jobs.
pub const JOB_TOOLS: [&str; 8] = [
    "job_status",
    "job_wait",
    "job_stop",
    "job_cancel",
    "job_restart",
    "job_result",
    "job_list",
    "mapper_steer",
];
/// Names of the tools of the components answered by the server itself (the others are those
/// of [`super::compute::TOOLS`]).
pub const COMPONENT_TOOLS: [&str; 4] = [
    "list_commands",
    "command_help",
    "list_calculators",
    "single_point",
];

/// The options that the compact schema of a command lists when the command has them (every
/// other implemented option is accepted by its long name all the same).
pub const COMMON_OPTIONS: [&str; 6] = [
    "othersoft",
    "opt_method",
    "NSTEP",
    "electronic_charge",
    "spin_multiplicity",
    "calc_option",
];

/// The longest `job_wait` (seconds).
pub const MAX_WAIT_SECONDS: f64 = 600.0;

/// How the tools of the commands are listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Listing {
    /// `INPUT`, `files` and [`COMMON_OPTIONS`]; the other options by name (`command_help`).
    #[default]
    Compact,
    /// One property per implemented option.
    Full,
}

/// Every tool of the server for the commands of `commands`.
pub fn all_tools(commands: CommandTable<'_>, listing: Listing) -> Vec<JsonValue> {
    let mut out = command_tools(commands, listing);
    out.extend(job_tools());
    out.extend(component_tools(commands));
    out.extend(super::compute::tools());
    out
}

/// The tools of the implemented commands of `commands`, in the order of the table.
pub fn command_tools(commands: CommandTable<'_>, listing: Listing) -> Vec<JsonValue> {
    commands
        .iter()
        .filter(|(_, implemented, _, _)| *implemented)
        .map(|(name, _, feature, options)| command_tool(name, feature, options, listing))
        .collect()
}

/// The tool that starts `command` as a job.
pub fn command_tool(
    command: &str,
    feature: &str,
    options: &[CliOption],
    listing: Listing,
) -> JsonValue {
    let mut properties = BTreeMap::new();
    let input = inputs::positional(command);
    let input_text = format!(
        "{} (the positional argument {} of the command). Paths relative to the server root \
         (or absolute inside it); each is copied into the job directory at the same relative \
         path and given to the command from there.",
        input.help.trim_end_matches('.'),
        input.name
    );
    let input_schema = if input.multiple {
        object([
            ("type", text("array")),
            ("items", object([("type", text("string"))])),
            ("minItems", JsonValue::Number(1.0)),
            ("description", text(input_text)),
        ])
    } else {
        object([("type", text("string")), ("description", text(input_text))])
    };
    properties.insert(INPUT_PROPERTY.to_string(), input_schema);
    properties.insert(
        FILES_PROPERTY.to_string(),
        object([
            ("type", text("array")),
            ("items", object([("type", text("string"))])),
            (
                "description",
                text(
                    "Further files or directories under the root that options name (an \
                     initial Hessian, a product structure, ...): copied into the job directory \
                     at the same relative paths, so that the paths of the options mean the same \
                     files there.",
                ),
            ),
        ]),
    );
    let offered = options
        .iter()
        .filter(|o| o.implemented)
        .filter(|o| listing == Listing::Full || COMMON_OPTIONS.contains(&o.spec.key));
    for o in offered {
        properties.insert(o.spec.key.to_string(), option_schema(o));
    }
    let implemented = options.iter().filter(|o| o.implemented).count();
    let (options_text, additional) = match listing {
        Listing::Full => (
            "The properties are the options of the command line by their long names (the keys \
             of settings files)."
                .to_string(),
            JsonValue::Bool(false),
        ),
        Listing::Compact => (
            format!(
                "Besides the properties listed, any of the {implemented} options of the command \
                 line is given by its long name (the key of settings files, e.g. opt_method, \
                 NSTEP): call command_help with command={command} for each option's type, \
                 default and help (options=[...] to look a few up)."
            ),
            object([
                (
                    "type",
                    JsonValue::Array(vec![
                        text("string"),
                        text("number"),
                        text("boolean"),
                        text("array"),
                    ]),
                ),
                (
                    "description",
                    text("Another option of the command by its long name (see command_help)."),
                ),
            ]),
        ),
    };
    object([
        ("name", text(command)),
        ("title", text(format!("{command}: {feature}"))),
        (
            "description",
            text(format!(
                "Start `{command}` ({feature}) as a job in a new directory under mcp_jobs/; \
                 returns its job_id and run directory at once. {options_text} INPUT names the \
                 inputs. Follow the job with job_status or job_wait, read it with job_result; \
                 job_stop writes STOP into the run directory, job_restart continues it with \
                 --restart, job_cancel ends it at once."
            )),
        ),
        (
            "inputSchema",
            object([
                ("type", text("object")),
                ("properties", JsonValue::Object(properties)),
                ("required", JsonValue::Array(vec![text(INPUT_PROPERTY)])),
                ("additionalProperties", additional),
            ]),
        ),
        ("annotations", annotations(false, false)),
    ])
}

/// Whether the default text of an option means "no value".
fn no_default(default: &str) -> bool {
    default.is_empty() || default == "None"
}

/// The JSON Schema of the option `o`.
pub fn option_schema(o: &CliOption) -> JsonValue {
    let flags = o.spec.flags.join(", ");
    let mut description = format!("{} [command line: {flags}", o.spec.help.trim());
    if o.spec.arity != ArgArity::Flag && !no_default(o.default) {
        description.push_str(&format!("; default: {}", o.default));
    }
    description.push(']');
    let scalar = || JsonValue::Array(vec![text("string"), text("number")]);
    let mut map = BTreeMap::new();
    match o.spec.arity {
        ArgArity::Flag => {
            map.insert("type".to_string(), text("boolean"));
            map.insert("default".to_string(), JsonValue::Bool(false));
        }
        ArgArity::Single => {
            map.insert("type".to_string(), scalar());
            if !no_default(o.default) {
                map.insert("default".to_string(), text(o.default));
            }
        }
        ArgArity::Optional(implicit) => {
            map.insert(
                "type".to_string(),
                JsonValue::Array(vec![text("string"), text("number"), text("boolean")]),
            );
            description.push_str(&if implicit.is_empty() {
                " true: the flag without a value.".to_string()
            } else {
                format!(" true: the flag without a value (which means {implicit}).")
            });
        }
        ArgArity::ZeroOrMore | ArgArity::OneOrMore => {
            map.insert("type".to_string(), text("array"));
            map.insert("items".to_string(), object([("type", scalar())]));
            if o.spec.arity == ArgArity::OneOrMore {
                map.insert("minItems".to_string(), JsonValue::Number(1.0));
            }
        }
    }
    map.insert("description".to_string(), text(description));
    JsonValue::Object(map)
}

fn annotations(read_only: bool, idempotent: bool) -> JsonValue {
    object([
        ("readOnlyHint", JsonValue::Bool(read_only)),
        ("destructiveHint", JsonValue::Bool(false)),
        ("idempotentHint", JsonValue::Bool(idempotent)),
        ("openWorldHint", JsonValue::Bool(false)),
    ])
}

/// A tool definition: `properties` of its input schema (an object without other
/// properties), the `required` ones, whether it only reads.
pub(crate) fn tool(
    name: &str,
    description: &str,
    properties: Vec<(&str, JsonValue)>,
    required: &[&str],
    read_only: bool,
) -> JsonValue {
    let props: BTreeMap<String, JsonValue> = properties
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let mut schema = BTreeMap::new();
    schema.insert("type".to_string(), text("object"));
    schema.insert("properties".to_string(), JsonValue::Object(props));
    if !required.is_empty() {
        schema.insert(
            "required".to_string(),
            JsonValue::Array(required.iter().map(text).collect()),
        );
    }
    schema.insert("additionalProperties".to_string(), JsonValue::Bool(false));
    object([
        ("name", text(name)),
        ("description", text(description)),
        ("inputSchema", JsonValue::Object(schema)),
        ("annotations", annotations(read_only, read_only)),
    ])
}

/// A property of `kind` with `description`.
pub(crate) fn prop(kind: &str, description: &str) -> JsonValue {
    object([("type", text(kind)), ("description", text(description))])
}

fn job_id() -> JsonValue {
    prop(
        "string",
        "The job_id that the tool of the command returned.",
    )
}

/// The tools of the jobs.
pub fn job_tools() -> Vec<JsonValue> {
    vec![
        tool(
            "job_status",
            "State of a job (running, finished, interrupted, failed, unknown), its exit code, \
             its run directories, the tail of mopt.log, the last iteration printed there, and \
             the tail of the console output.",
            vec![
                ("job_id", job_id()),
                (
                    "tail_lines",
                    object([
                        ("type", text("integer")),
                        ("minimum", JsonValue::Number(0.0)),
                        ("maximum", JsonValue::Number(500.0)),
                        ("default", JsonValue::Number(40.0)),
                        (
                            "description",
                            text("Lines of the end of mopt.log to return."),
                        ),
                    ]),
                ),
            ],
            &["job_id"],
            true,
        ),
        tool(
            "job_wait",
            "Wait until a job does not run any more or the time is up, then answer as \
             job_status (with waited_seconds and timed_out). While it waits, a request that \
             carries _meta.progressToken gets notifications/progress at every new iteration in \
             mopt.log; notifications/cancelled ends the wait without an answer.",
            vec![
                ("job_id", job_id()),
                (
                    "timeout_seconds",
                    object([
                        ("type", text("number")),
                        ("minimum", JsonValue::Number(0.0)),
                        ("maximum", JsonValue::Number(MAX_WAIT_SECONDS)),
                        ("default", JsonValue::Number(60.0)),
                        ("description", text("The longest wait (seconds).")),
                    ]),
                ),
                (
                    "tail_lines",
                    object([
                        ("type", text("integer")),
                        ("minimum", JsonValue::Number(0.0)),
                        ("maximum", JsonValue::Number(500.0)),
                        ("default", JsonValue::Number(40.0)),
                        (
                            "description",
                            text("Lines of the end of mopt.log to return."),
                        ),
                    ]),
                ),
            ],
            &["job_id"],
            true,
        ),
        tool(
            "job_stop",
            "Stop a running job cleanly: writes the file STOP into its run directories; the \
             command ends at its next check and can be continued with job_restart.",
            vec![("job_id", job_id())],
            &["job_id"],
            false,
        ),
        tool(
            "job_cancel",
            "End a running job of this server process at once (the process is killed; state \
             cancelled). What it wrote stays; job_restart continues it from its evaluation \
             journal. Prefer job_stop, which ends it cleanly at its next check.",
            vec![("job_id", job_id())],
            &["job_id"],
            false,
        ),
        tool(
            "job_restart",
            "Continue a stopped or killed job in its run directory (the command with --restart \
             RUN_DIR, which reads the settings and the evaluation journal kept there).",
            vec![
                ("job_id", job_id()),
                (
                    "run_directory",
                    prop(
                        "string",
                        "The run directory to continue (relative to the root), when the job \
                         has several.",
                    ),
                ),
                (
                    "force",
                    object([
                        ("type", text("boolean")),
                        ("default", JsonValue::Bool(false)),
                        (
                            "description",
                            text(
                                "Restart a job of another server process whose state is \
                                 unknown (make sure it does not run any more).",
                            ),
                        ),
                    ]),
                ),
            ],
            &["job_id"],
            false,
        ),
        tool(
            "job_result",
            "The files of a job's run directories with their sizes and resource URIs (read \
             them with resources/read), and the totals of the evaluations of the run.",
            vec![("job_id", job_id())],
            &["job_id"],
            true,
        ),
        tool(
            "job_list",
            "The jobs under mcp_jobs/ of the root, with their commands and states.",
            Vec::new(),
            &[],
            true,
        ),
        tool(
            "mapper_steer",
            "Steer a running run_mapper job: the lines are appended to steer.txt of its run \
             directory, which the run reads before its next choice of jobs (explore NAME: its \
             open tasks next; skip NAME / unskip NAME: no more jobs from that minimum, or again; \
             max_jobs N: the limit of jobs from now on). NAME is a label such as EQ5 or a \
             number; steer.log tells what became of each line.",
            vec![
                ("job_id", job_id()),
                (
                    "lines",
                    object([
                        ("type", text("array")),
                        ("items", object([("type", text("string"))])),
                        ("minItems", JsonValue::Number(1.0)),
                        ("maxItems", JsonValue::Number(100.0)),
                        (
                            "description",
                            text("Steering lines, e.g. explore EQ3, max_jobs 20."),
                        ),
                    ]),
                ),
            ],
            &["job_id", "lines"],
            false,
        ),
    ]
}

/// The tools of the components (`commands` for the names of `command_help`).
pub fn component_tools(commands: CommandTable<'_>) -> Vec<JsonValue> {
    let names: Vec<JsonValue> = commands.iter().map(|(n, _, _, _)| text(n)).collect();
    vec![
        tool(
            "list_commands",
            "The commands of mopt-rs with whether they are implemented, their features, their \
             numbers of options, and the name of the tool that starts each.",
            Vec::new(),
            &[],
            true,
        ),
        tool(
            "command_help",
            "The --help text of a command (its options with flags, defaults, and what is not \
             implemented yet), or, with options, the JSON Schema and help of the options named \
             (long names, flags or parts of them); schema=true adds the full input schema of \
             the command's tool, with every implemented option.",
            vec![
                (
                    "command",
                    object([
                        ("type", text("string")),
                        ("enum", JsonValue::Array(names)),
                        ("description", text("The command.")),
                    ]),
                ),
                (
                    "options",
                    object([
                        ("type", text("array")),
                        ("items", object([("type", text("string"))])),
                        (
                            "description",
                            text("Options to look up: long names, flags, or parts of them."),
                        ),
                    ]),
                ),
                (
                    "schema",
                    object([
                        ("type", text("boolean")),
                        ("default", JsonValue::Bool(false)),
                        (
                            "description",
                            text("Also the full input schema of the command's tool."),
                        ),
                    ]),
                ),
            ],
            &["command"],
            true,
        ),
        tool(
            "list_calculators",
            "The calculators this server can evaluate (the names of -os), the built-in \
             potentials among them, and the objectives of plain variables.",
            Vec::new(),
            &[],
            true,
        ),
        tool(
            "single_point",
            "Energy (Hartree) and gradient (Hartree/Bohr) of one structure with a calculator, \
             computed at once (for jobs, use the tools of the commands).",
            vec![
                (
                    "structure",
                    prop(
                        "string",
                        "An xyz file under the root (or give the text as xyz).",
                    ),
                ),
                (
                    "xyz",
                    prop("string", "The text of an xyz file (instead of structure)."),
                ),
                (
                    "calculator",
                    prop(
                        "string",
                        "Calculator name, as for -os (see list_calculators).",
                    ),
                ),
                (
                    "charge",
                    object([
                        ("type", text("integer")),
                        ("default", JsonValue::Number(0.0)),
                        ("description", text("Total charge.")),
                    ]),
                ),
                (
                    "multiplicity",
                    object([
                        ("type", text("integer")),
                        ("minimum", JsonValue::Number(1.0)),
                        ("default", JsonValue::Number(1.0)),
                        ("description", text("Spin multiplicity 2S + 1.")),
                    ]),
                ),
                (
                    "settings",
                    object([
                        ("type", text("object")),
                        (
                            "additionalProperties",
                            object([(
                                "type",
                                JsonValue::Array(vec![
                                    text("string"),
                                    text("number"),
                                    text("boolean"),
                                ]),
                            )]),
                        ),
                        (
                            "description",
                            text("Settings of the calculator, as KEY=VALUE of -calc_opt."),
                        ),
                    ]),
                ),
                (
                    "hessian",
                    object([
                        ("type", text("boolean")),
                        ("default", JsonValue::Bool(false)),
                        (
                            "description",
                            text("Also the Hessian (Hartree/Bohr^2, 3N x 3N, row by row)."),
                        ),
                    ]),
                ),
            ],
            &["calculator"],
            true,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::spec::COMMANDS;

    fn names(tools: &[JsonValue]) -> Vec<String> {
        tools
            .iter()
            .filter_map(|t| t.get("name").and_then(JsonValue::as_str).map(String::from))
            .collect()
    }

    #[test]
    fn a_command_implemented_in_the_table_is_a_tool_without_code_changes() {
        let mut table: Vec<(&str, bool, &str, &[CliOption])> = COMMANDS.to_vec();
        // A command that is not implemented (every command of the table may be: then the last
        // one is marked as not implemented, with its options, as gen_cli.py would).
        let flipped = match table
            .iter()
            .position(|(_, implemented, _, _)| !*implemented)
        {
            Some(k) => k,
            None => {
                let k = table.len() - 1;
                table[k].1 = false;
                let options: Vec<CliOption> = table[k]
                    .3
                    .iter()
                    .cloned()
                    .map(|mut o| {
                        o.implemented = false;
                        o
                    })
                    .collect();
                table[k].3 = Box::leak(options.into_boxed_slice());
                k
            }
        };
        let name = table[flipped].0;
        assert!(!names(&command_tools(&table, Listing::Full)).contains(&name.to_string()));
        table[flipped].1 = true;
        // Options of a command that is not implemented are marked as not implemented; the
        // flip of the table marks them too, as gen_cli.py does for an implemented command.
        let options: Vec<CliOption> = table[flipped]
            .3
            .iter()
            .cloned()
            .map(|mut o| {
                o.implemented = true;
                o
            })
            .collect();
        table[flipped].3 = Box::leak(options.into_boxed_slice());
        let tools = command_tools(&table, Listing::Full);
        let tool = tools
            .iter()
            .find(|t| t.get("name").and_then(JsonValue::as_str) == Some(name))
            .unwrap();
        let properties = tool
            .get("inputSchema")
            .and_then(|s| s.get("properties"))
            .and_then(JsonValue::as_object)
            .unwrap();
        for o in table[flipped].3 {
            assert!(
                properties.contains_key(o.spec.key),
                "{name}: {}",
                o.spec.key
            );
        }
        assert!(properties.contains_key(INPUT_PROPERTY));
    }

    #[test]
    fn arities_become_types() {
        let table = crate::cli::spec::OPTMAIN;
        let find = |key: &str| table.iter().find(|o| o.spec.key == key).unwrap();
        let flag = option_schema(find("unrestrict"));
        assert_eq!(flag.get("type"), Some(&text("boolean")));
        let single = option_schema(find("NSTEP"));
        assert_eq!(single.get("default"), Some(&text("1000")));
        let list = option_schema(find("opt_method"));
        assert_eq!(list.get("type"), Some(&text("array")));
        let optional = option_schema(find("use_model_hessian"));
        let d = optional
            .get("description")
            .and_then(JsonValue::as_str)
            .unwrap();
        assert!(d.contains("fischerd3old"), "{d}");
    }
}
