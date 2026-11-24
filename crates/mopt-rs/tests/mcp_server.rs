//! The MCP server end to end: `mopt mcp` as a child process, driven over stdio as a client
//! does it.
//!
//! The handshake of `2025-11-25` and the stateless requests of `2026-07-28`; the tools; an
//! `optmain` job on an argon cluster with the built-in `lj` calculator followed to its end and
//! read; a second job stopped by its STOP file and continued with `--restart`; the limit on
//! running jobs; the documentation inside the repository. The attacks are in `mcp_attacks.rs`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod mcp_client;

use std::path::Path;
use std::time::Duration;

use mcp_client::{argon_grid, get, scratch, Client, AR7, MODERN};
use mopt_rs::core_infra::json::JsonValue;

#[test]
fn a_client_runs_follows_stops_and_restarts_jobs() {
    let dir = scratch("jobs");
    let root = dir.join("root");
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    std::fs::create_dir_all(root.join("grid")).unwrap();
    std::fs::write(root.join("grid/ar.xyz"), argon_grid(4)).unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    assert_eq!(c.result("ping", "{}"), JsonValue::empty_object());

    // The tools of the implemented commands, of the jobs and of the components.
    let listed = c.result("tools/list", "{}");
    let names: Vec<String> = get(&listed, &["tools"])
        .as_array()
        .unwrap()
        .iter()
        .map(|t| get(t, &["name"]).as_str().unwrap().to_string())
        .collect();
    for n in [
        "optmain",
        "nebmain",
        "ircmain",
        "job_status",
        "job_stop",
        "job_restart",
        "job_result",
        "single_point",
        "run_mapper",
    ] {
        assert!(names.contains(&n.to_string()), "{n}: {names:?}");
    }
    // Every command is implemented now (mcp::schema tests that one that is not is no tool).
    for n in ["run_mapper", "run_kinetics", "run_rcmc"] {
        assert!(names.contains(&n.to_string()), "{n}: {names:?}");
    }
    let calculators = c.ok_tool("list_calculators", "{}");
    assert!(get(&calculators, &["calculators"])
        .as_string_vec()
        .unwrap()
        .contains(&"lj".to_string()));
    let commands = c.ok_tool("list_commands", "{}");
    assert!(get(&commands, &["commands"]).as_array().unwrap().len() >= 12);
    let help = c.ok_tool("command_help", r#"{"command":"optmain"}"#);
    assert!(get(&help, &["help"]).as_str().unwrap().contains("--NSTEP"));

    // A single point with a built-in calculator; the cluster starts compressed (UFF argon).
    let sp = c.ok_tool(
        "single_point",
        r#"{"structure":"ar7.xyz","calculator":"lj"}"#,
    );
    let e = get(&sp, &["energy_hartree"]).as_f64().unwrap();
    assert!(e.is_finite() && e > 0.0, "{e}");
    assert_eq!(
        get(&sp, &["gradient_hartree_per_bohr"])
            .as_array()
            .unwrap()
            .len(),
        7
    );

    // An optimization followed to its end.
    let started = c.ok_tool(
        "optmain",
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","opt_method":["rsirfo_bfgs"],"NSTEP":300}"#,
    );
    let job = get(&started, &["job_id"]).as_str().unwrap().to_string();
    assert!(mopt_rs::mcp::jobs::is_job_id(&job), "{job}");
    let status = c.wait(&job, Duration::from_secs(300));
    assert_eq!(
        get(&status, &["state"]).as_str(),
        Some("finished"),
        "{status:?}"
    );
    assert_eq!(get(&status, &["exit_code"]), &JsonValue::Number(0.0));
    assert!(
        get(&status, &["log_tail"])
            .as_str()
            .unwrap()
            .contains("converged"),
        "{status:?}"
    );
    assert!(get(&status, &["last_iteration"]).as_str().is_some());
    let result = c.ok_tool("job_result", &format!(r#"{{"job_id":"{job}"}}"#));
    let runs = get(&result, &["run_directories"]).as_array().unwrap();
    assert_eq!(runs.len(), 1);
    let files = get(&runs[0], &["files"]).as_array().unwrap();
    let optimized = files
        .iter()
        .find(|f| {
            get(f, &["path"])
                .as_str()
                .unwrap()
                .ends_with("ar7_optimized.xyz")
        })
        .expect("the optimized structure");
    assert!(files.iter().any(|f| get(f, &["path"])
        .as_str()
        .unwrap()
        .ends_with("settings.json")));
    let gradients = get(&result, &["evaluations", "gradients"])
        .as_f64()
        .unwrap();
    assert!(gradients > 5.0, "{result:?}");
    // Its files are resources.
    let uri = get(optimized, &["uri"]).as_str().unwrap().to_string();
    let read = c.result("resources/read", &format!(r#"{{"uri":"{uri}"}}"#));
    let content = get(&read, &["contents"]).as_array().unwrap()[0].clone();
    assert!(get(&content, &["text"])
        .as_str()
        .unwrap()
        .contains("OptimizedStructure"));
    let resources = c.result("resources/list", "{}");
    assert!(get(&resources, &["resources"])
        .as_array()
        .unwrap()
        .iter()
        .any(|r| get(r, &["uri"]).as_str() == Some(uri.as_str())));
    let text = c.refused_tool("job_stop", &format!(r#"{{"job_id":"{job}"}}"#));
    assert!(text.contains("not running"), "{text}");

    // A second job: stopped by its STOP file, continued with --restart in its run directory.
    let started = c.ok_tool(
        "optmain",
        r#"{"INPUT":["grid/ar.xyz"],"othersoft":"lj","NSTEP":40}"#,
    );
    let second = get(&started, &["job_id"]).as_str().unwrap().to_string();
    let run = get(&started, &["run_directory"])
        .as_str()
        .expect("the run directory")
        .to_string();
    assert!(run.contains("grid/ar_OPT_ASE_"), "{run}");
    let stopped = c.ok_tool("job_stop", &format!(r#"{{"job_id":"{second}"}}"#));
    assert!(get(&stopped, &["stop_files"]).as_array().unwrap().len() == 1);
    let status = c.wait(&second, Duration::from_secs(300));
    assert_eq!(
        get(&status, &["state"]).as_str(),
        Some("interrupted"),
        "{status:?}"
    );
    assert_eq!(get(&status, &["exit_code"]), &JsonValue::Number(3.0));
    let restarted = c.ok_tool("job_restart", &format!(r#"{{"job_id":"{second}"}}"#));
    assert_eq!(get(&restarted, &["attempt"]), &JsonValue::Number(2.0));
    assert_eq!(
        get(&restarted, &["run_directory"]).as_str(),
        Some(run.as_str())
    );
    let text = c.refused_tool("job_restart", &format!(r#"{{"job_id":"{second}"}}"#));
    assert!(text.contains("running"), "{text}");
    let status = c.wait(&second, Duration::from_secs(600));
    assert_eq!(
        get(&status, &["state"]).as_str(),
        Some("finished"),
        "{status:?}"
    );
    let listed = c.ok_tool("job_list", "{}");
    assert_eq!(get(&listed, &["jobs"]).as_array().unwrap().len(), 2);
    // The run continued in its directory, from the settings and the journal kept there.
    let settings = std::fs::read_to_string(root.join(&run).join("settings.json")).unwrap();
    assert!(settings.contains("grid/ar.xyz"), "{settings}");
    assert!(root.join(&run).join("evaluations.journal").is_file());
    c.close();

    // Another server process knows how the jobs ended.
    let mut again = Client::start(&root);
    again.initialize();
    let s = again.ok_tool("job_status", &format!(r#"{{"job_id":"{job}"}}"#));
    assert_eq!(get(&s, &["state"]).as_str(), Some("finished"));
    again.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_server_runs_at_most_max_jobs_at_a_time() {
    let dir = scratch("limit");
    let root = dir.join("root");
    std::fs::write(root.join("ar.xyz"), argon_grid(4)).unwrap();
    let mut c = Client::start_with(&root, &["--max-jobs", "1"]);
    c.initialize();
    let started = c.ok_tool(
        "optmain",
        r#"{"INPUT":["ar.xyz"],"othersoft":"lj","NSTEP":40}"#,
    );
    let job = get(&started, &["job_id"]).as_str().unwrap().to_string();
    let text = c.refused_tool(
        "optmain",
        r#"{"INPUT":["ar.xyz"],"othersoft":"lj","NSTEP":40}"#,
    );
    assert!(text.contains("--max-jobs"), "{text}");
    c.ok_tool("job_stop", &format!(r#"{{"job_id":"{job}"}}"#));
    c.wait(&job, Duration::from_secs(300));
    c.ok_tool(
        "optmain",
        r#"{"INPUT":["ar.xyz"],"othersoft":"lj","NSTEP":2}"#,
    );
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stateless_requests_and_errors_follow_the_2026_revision() {
    let dir = scratch("modern");
    let root = dir.join("root");
    let mut c = Client::start(&root);
    let meta = format!(
        r#""_meta":{{"io.modelcontextprotocol/protocolVersion":"{MODERN}","io.modelcontextprotocol/clientCapabilities":{{}}}}"#
    );
    let r = c.result("server/discover", &format!("{{{meta}}}"));
    assert_eq!(get(&r, &["resultType"]).as_str(), Some("complete"));
    assert!(get(&r, &["supportedVersions"])
        .as_string_vec()
        .unwrap()
        .contains(&MODERN.to_string()));
    assert_eq!(
        get(&r, &["_meta", "io.modelcontextprotocol/serverInfo", "name"]).as_str(),
        Some("mopt-rs")
    );
    let r = c.result("tools/list", &format!("{{{meta}}}"));
    assert_eq!(get(&r, &["resultType"]).as_str(), Some("complete"));
    assert!(get(&r, &["ttlMs"]).as_f64().is_some());
    assert!(get(&r, &["cacheScope"]).as_str().is_some());
    // No handshake and no _meta.
    let r = c.request("tools/list", "{}");
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
    // Capabilities missing.
    let r = c.request(
        "tools/list",
        &format!(r#"{{"_meta":{{"io.modelcontextprotocol/protocolVersion":"{MODERN}"}}}}"#),
    );
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
    // A version the server does not know.
    let r = c.request(
        "tools/list",
        r#"{"_meta":{"io.modelcontextprotocol/protocolVersion":"1900-01-01","io.modelcontextprotocol/clientCapabilities":{}}}"#,
    );
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32022.0));
    assert_eq!(
        get(&r, &["error", "data", "requested"]).as_str(),
        Some("1900-01-01")
    );
    // ping is gone in 2026-07-28; unknown methods are -32601.
    let r = c.request("ping", &format!("{{{meta}}}"));
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32601.0));
    let r = c.request("no/such", &format!("{{{meta}}}"));
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32601.0));
    // Resources outside the repository: the documentation embedded in the binary; a page
    // or a file that does not exist is -32602.
    let r = c.result(
        "resources/read",
        &format!(r#"{{"uri":"mopt://docs/quickstart",{meta}}}"#),
    );
    assert!(
        get(&get(&r, &["contents"]).as_array().unwrap()[0], &["text"])
            .as_str()
            .unwrap()
            .contains("mopt")
    );
    let r = c.request(
        "resources/read",
        &format!(r#"{{"uri":"mopt://docs/no_such_page",{meta}}}"#),
    );
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
    let r = c.request(
        "resources/read",
        &format!(r#"{{"uri":"mopt://root/no_such_file.xyz",{meta}}}"#),
    );
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
    let r = c.result("prompts/list", &format!("{{{meta}}}"));
    assert!(!get(&r, &["prompts"]).as_array().unwrap().is_empty());
    let r = c.result(
        "prompts/get",
        &format!(r#"{{"name":"optimize_structure","arguments":{{"structure":"a.xyz"}},{meta}}}"#),
    );
    assert!(get(&r, &["messages"]).as_array().unwrap().len() == 1);
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_documentation_of_the_repository_is_a_resource() {
    let docs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let mut c = Client::start(&docs);
    c.initialize();
    let list = c.result("resources/list", "{}");
    let uris: Vec<String> = get(&list, &["resources"])
        .as_array()
        .unwrap()
        .iter()
        .map(|r| get(r, &["uri"]).as_str().unwrap().to_string())
        .collect();
    assert!(
        uris.contains(&"mopt://docs/quickstart".to_string()),
        "{uris:?}"
    );
    assert!(uris.contains(&"mopt://docs/mcp".to_string()), "{uris:?}");
    let r = c.result("resources/read", r#"{"uri":"mopt://docs/mcp"}"#);
    let text = get(&r, &["contents"]).as_array().unwrap()[0].clone();
    assert!(get(&text, &["text"]).as_str().unwrap().contains("mopt mcp"));
    let r = c.request("resources/read", r#"{"uri":"mopt://docs/../../README"}"#);
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32002.0));
    let templates = c.result("resources/templates/list", "{}");
    assert_eq!(
        get(&templates, &["resourceTemplates"])
            .as_array()
            .unwrap()
            .len(),
        3
    );
    c.close();
}
