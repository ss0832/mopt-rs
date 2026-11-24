//! The features of the MCP server beyond the plain tool calls, end to end over stdio: the
//! compact tool list and `command_help`, pages of the lists, `job_wait` with progress and
//! cancellation, `job_cancel`, tasks of `2026-07-28` and of `2025-11-25`, subscriptions of
//! both, the component tools, the embedded documentation, and a short `run_mapper` on LJ7.
//! The attacks on the new parts are in `mcp_attacks.rs`.
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

const WATER: &str = "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469\n";

/// `_meta` of a stateless request, with the tasks extension or without.
fn meta(tasks: bool) -> String {
    let caps = if tasks {
        r#"{"extensions":{"io.modelcontextprotocol/tasks":{}}}"#
    } else {
        "{}"
    };
    format!(
        r#""_meta":{{"io.modelcontextprotocol/protocolVersion":"{MODERN}","io.modelcontextprotocol/clientCapabilities":{caps}}}"#
    )
}

/// The values of `field` of the entries of the list `key`.
fn fields(list: &JsonValue, key: &str, field: &str) -> Vec<String> {
    get(list, &[key])
        .as_array()
        .unwrap()
        .iter()
        .map(|t| get(t, &[field]).as_str().unwrap().to_string())
        .collect()
}

fn names(list: &JsonValue, key: &str) -> Vec<String> {
    fields(list, key, "name")
}

#[test]
fn the_list_is_compact_command_help_gives_the_options_and_lists_come_in_pages() {
    let dir = scratch("compact");
    let root = dir.join("root");
    let mut c = Client::start_with(&root, &["--page-size", "7"]);
    c.initialize();
    // Pages of seven, followed to the end.
    let mut all = Vec::new();
    let mut cursor: Option<String> = None;
    let mut pages = 0;
    loop {
        let params = match &cursor {
            Some(k) => format!(r#"{{"cursor":"{k}"}}"#),
            None => "{}".to_string(),
        };
        let page = c.result("tools/list", &params);
        let got = names(&page, "tools");
        assert!(got.len() <= 7);
        all.extend(got);
        pages += 1;
        match page.get("nextCursor").and_then(JsonValue::as_str) {
            Some(k) => cursor = Some(k.to_string()),
            None => break,
        }
    }
    assert!(pages >= 4, "{pages} pages");
    for n in [
        "optmain",
        "run_mapper",
        "job_wait",
        "job_cancel",
        "command_help",
        "model_hessian",
    ] {
        assert!(all.contains(&n.to_string()), "{n}: {all:?}");
    }
    let r = c.request("tools/list", r#"{"cursor":"1.0000000000000000"}"#);
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
    c.close();

    // The compact listing (one page here) is small; --full-schemas lists every option.
    let mut compact = Client::start(&root);
    compact.initialize();
    let small = compact.result("tools/list", "{}");
    assert!(small.get("nextCursor").is_none());
    let tools = get(&small, &["tools"]).as_array().unwrap().to_vec();
    let optmain = tools
        .iter()
        .find(|t| get(t, &["name"]).as_str() == Some("optmain"))
        .unwrap();
    let props = get(optmain, &["inputSchema", "properties"])
        .as_object()
        .unwrap();
    assert!(
        props.len() < 10 && props.contains_key("othersoft"),
        "{props:?}"
    );
    // An option that is not listed is taken by its long name.
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    let started = compact.ok_tool(
        "optmain",
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","opt_method":["rsirfo_bfgs"],"use_model_hessian":"lindh"}"#,
    );
    let job = get(&started, &["job_id"]).as_str().unwrap().to_string();
    let waited = compact.ok_tool(
        "job_wait",
        &format!(r#"{{"job_id":"{job}","timeout_seconds":300}}"#),
    );
    assert_eq!(
        get(&waited, &["state"]).as_str(),
        Some("finished"),
        "{waited:?}"
    );
    let text = compact.refused_tool("optmain", r#"{"INPUT":["ar7.xyz"],"no_such_option":1}"#);
    assert!(
        text.contains("no_such_option") && text.contains("command_help"),
        "{text}"
    );
    // command_help: the text, options looked up, the full schema.
    let help = compact.ok_tool("command_help", r#"{"command":"optmain"}"#);
    assert!(get(&help, &["help"]).as_str().unwrap().contains("--NSTEP"));
    let found = compact.ok_tool(
        "command_help",
        r#"{"command":"optmain","options":["modelhess"],"schema":true}"#,
    );
    let hit = &get(&found, &["matches"]).as_array().unwrap()[0];
    let options = get(hit, &["options"]).as_array().unwrap();
    assert!(options
        .iter()
        .any(|o| get(o, &["key"]).as_str() == Some("use_model_hessian")
            && get(o, &["help"]).as_str().unwrap().contains("gfn1xtb")));
    let schema_props = get(&found, &["inputSchema", "properties"])
        .as_object()
        .unwrap();
    assert!(schema_props.len() > 100, "{}", schema_props.len());
    let small_size = small.to_string_compact().len();
    compact.close();
    let mut full = Client::start_with(&root, &["--full-schemas"]);
    full.initialize();
    let big = full.result("tools/list", "{}");
    let big_size = big.to_string_compact().len();
    assert!(small_size * 5 < big_size, "{small_size} against {big_size}");
    eprintln!("tools/list: compact {small_size} bytes, full {big_size} bytes");
    full.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn job_wait_reports_progress_ends_when_cancelled_and_job_cancel_ends_a_job() {
    let dir = scratch("wait");
    let root = dir.join("root");
    std::fs::write(root.join("ar.xyz"), argon_grid(6)).unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    let started = c.ok_tool(
        "optmain",
        r#"{"INPUT":["ar.xyz"],"othersoft":"lj","NSTEP":100000}"#,
    );
    let job = get(&started, &["job_id"]).as_str().unwrap().to_string();
    // Waits with a progress token: notifications/progress before each answer at the timeout.
    let mut progress = Vec::new();
    for round in 0..12 {
        let id = c.next_id();
        c.send(&format!(
            r#"{{"jsonrpc":"2.0","id":{id},"method":"tools/call","params":{{"name":"job_wait","arguments":{{"job_id":"{job}","timeout_seconds":5}},"_meta":{{"progressToken":"p{round}"}}}}}}"#
        ));
        let answer = c.response_to(id);
        let waited = get(&answer, &["result", "structuredContent"]);
        assert_eq!(
            get(waited, &["state"]).as_str(),
            Some("running"),
            "{waited:?}"
        );
        assert_eq!(get(waited, &["timed_out"]), &JsonValue::Bool(true));
        assert!(get(waited, &["waited_seconds"]).as_f64().unwrap() >= 4.5);
        let heard = c.take_notifications("notifications/progress");
        assert!(heard
            .iter()
            .all(|p| get(p, &["params", "progressToken"]).as_str()
                == Some(format!("p{round}").as_str())));
        if !heard.is_empty() {
            progress = heard;
            break;
        }
    }
    assert!(!progress.is_empty(), "no progress");
    let values: Vec<f64> = progress
        .iter()
        .map(|p| get(p, &["params", "progress"]).as_f64().unwrap())
        .collect();
    assert!(values.windows(2).all(|w| w[1] > w[0]), "{values:?}");
    // A wait that the client cancels is never answered; the next request is.
    let cancelled = c.next_id();
    c.send(&format!(
        r#"{{"jsonrpc":"2.0","id":{cancelled},"method":"tools/call","params":{{"name":"job_wait","arguments":{{"job_id":"{job}","timeout_seconds":600}}}}}}"#
    ));
    c.send(&format!(
        r#"{{"jsonrpc":"2.0","method":"notifications/cancelled","params":{{"requestId":{cancelled},"reason":"test"}}}}"#
    ));
    let status = c.ok_tool("job_status", &format!(r#"{{"job_id":"{job}"}}"#));
    assert_eq!(get(&status, &["state"]).as_str(), Some("running"));
    // job_cancel ends the job at once; it can be continued from its journal.
    let ended = c.ok_tool("job_cancel", &format!(r#"{{"job_id":"{job}"}}"#));
    assert_eq!(
        get(&ended, &["state"]).as_str(),
        Some("cancelled"),
        "{ended:?}"
    );
    let text = c.refused_tool("job_cancel", &format!(r#"{{"job_id":"{job}"}}"#));
    assert!(text.contains("not running"), "{text}");
    // Nothing came for the cancelled wait meanwhile.
    while let Some(m) = c.read_within(Duration::from_millis(600)) {
        assert!(
            m.get("id").is_none(),
            "an answer to a cancelled request: {m:?}"
        );
    }
    c.close();
    // Another server process knows that it was cancelled.
    let mut again = Client::start(&root);
    again.initialize();
    let s = again.ok_tool("job_status", &format!(r#"{{"job_id":"{job}"}}"#));
    assert_eq!(get(&s, &["state"]).as_str(), Some("cancelled"));
    let restarted = again.ok_tool("job_restart", &format!(r#"{{"job_id":"{job}"}}"#));
    assert_eq!(get(&restarted, &["attempt"]), &JsonValue::Number(2.0));
    again.ok_tool("job_cancel", &format!(r#"{{"job_id":"{job}"}}"#));
    again.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn command_tools_run_as_tasks_in_both_revisions() {
    let dir = scratch("tasks");
    let root = dir.join("root");
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    std::fs::write(root.join("ar.xyz"), argon_grid(5)).unwrap();

    // 2026-07-28 with the tasks extension: the call answers with a task.
    let mut c = Client::start(&root);
    let discover = c.result("server/discover", &format!("{{{}}}", meta(false)));
    assert!(get(
        &discover,
        &[
            "capabilities",
            "extensions",
            "io.modelcontextprotocol/tasks"
        ]
    )
    .as_object()
    .is_some());
    let r = c.result(
        "tools/call",
        &format!(
            r#"{{"name":"optmain","arguments":{{"INPUT":["ar7.xyz"],"othersoft":"lj"}},{}}}"#,
            meta(true)
        ),
    );
    assert_eq!(get(&r, &["resultType"]).as_str(), Some("task"), "{r:?}");
    let task = get(&r, &["taskId"]).as_str().unwrap().to_string();
    assert!(mopt_rs::mcp::jobs::is_job_id(&task));
    assert_eq!(get(&r, &["status"]).as_str(), Some("working"));
    assert!(get(&r, &["createdAt"]).as_str().unwrap().ends_with('Z'));
    // Without the extension, the same call is a plain tool result.
    let plain = c.result(
        "tools/call",
        &format!(
            r#"{{"name":"single_point","arguments":{{"structure":"ar7.xyz","calculator":"lj"}},{}}}"#,
            meta(true)
        ),
    );
    assert_eq!(get(&plain, &["resultType"]).as_str(), Some("complete"));
    let mut done = JsonValue::Null;
    for _ in 0..600 {
        done = c.result(
            "tasks/get",
            &format!(r#"{{"taskId":"{task}",{}}}"#, meta(true)),
        );
        if get(&done, &["status"]).as_str() != Some("working") {
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    assert_eq!(
        get(&done, &["status"]).as_str(),
        Some("completed"),
        "{done:?}"
    );
    assert_eq!(get(&done, &["result", "isError"]), &JsonValue::Bool(false));
    assert!(
        !get(&done, &["result", "structuredContent", "run_directories"])
            .as_array()
            .unwrap()
            .is_empty()
    );
    // Cancel: cooperative (STOP); the task is cancelled at once, once.
    let r = c.result(
        "tools/call",
        &format!(
            r#"{{"name":"optmain","arguments":{{"INPUT":["ar.xyz"],"othersoft":"lj","NSTEP":100000}},{}}}"#,
            meta(true)
        ),
    );
    let long = get(&r, &["taskId"]).as_str().unwrap().to_string();
    // A stream that hears of the task.
    let listen = c.next_id();
    c.send(&format!(
        r#"{{"jsonrpc":"2.0","id":{listen},"method":"subscriptions/listen","params":{{"notifications":{{"taskIds":["{long}"]}},{}}}}}"#,
        meta(true)
    ));
    let ack = c.wait_notification(
        "notifications/subscriptions/acknowledged",
        Duration::from_secs(30),
    );
    assert_eq!(
        get(
            &ack,
            &["params", "_meta", "io.modelcontextprotocol/subscriptionId"]
        ),
        &JsonValue::Number(listen as f64)
    );
    std::thread::sleep(Duration::from_secs(3));
    let r = c.result(
        "tasks/cancel",
        &format!(r#"{{"taskId":"{long}",{}}}"#, meta(true)),
    );
    assert_eq!(get(&r, &["resultType"]).as_str(), Some("complete"));
    let heard = c.wait_notification("notifications/tasks", Duration::from_secs(60));
    assert_eq!(
        get(&heard, &["params", "taskId"]).as_str(),
        Some(long.as_str())
    );
    assert_eq!(
        get(&heard, &["params", "status"]).as_str(),
        Some("cancelled")
    );
    assert_eq!(
        get(
            &heard,
            &["params", "_meta", "io.modelcontextprotocol/subscriptionId"]
        ),
        &JsonValue::Number(listen as f64)
    );
    let r = c.result(
        "tasks/get",
        &format!(r#"{{"taskId":"{long}",{}}}"#, meta(true)),
    );
    assert_eq!(get(&r, &["status"]).as_str(), Some("cancelled"));
    for (method, code) in [
        ("tasks/cancel", -32602.0),
        ("tasks/update", -32602.0),
        ("tasks/result", -32601.0),
        ("tasks/list", -32601.0),
    ] {
        let r = c.request(method, &format!(r#"{{"taskId":"{long}",{}}}"#, meta(true)));
        assert_eq!(
            get(&r, &["error", "code"]),
            &JsonValue::Number(code),
            "{method}"
        );
    }
    // The job ends at its next check of STOP.
    c.send(&format!(
        r#"{{"jsonrpc":"2.0","method":"notifications/cancelled","params":{{"requestId":{listen}}}}}"#
    ));
    let mut legacy = Client::start(&root);
    let init = legacy.result(
        "initialize",
        r#"{"protocolVersion":"2025-11-25","capabilities":{"tasks":{}},"clientInfo":{"name":"t","version":"1"}}"#,
    );
    assert!(get(
        &init,
        &["capabilities", "tasks", "requests", "tools", "call"]
    )
    .as_object()
    .is_some());
    legacy.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
    let s = legacy.wait(&long, Duration::from_secs(300));
    assert_eq!(get(&s, &["state"]).as_str(), Some("interrupted"), "{s:?}");
    c.close();

    // 2025-11-25: execution.taskSupport, a task-augmented call, tasks/result, tasks/list,
    // notifications/tasks/status.
    let tools = legacy.result("tools/list", "{}");
    let optmain = get(&tools, &["tools"])
        .as_array()
        .unwrap()
        .iter()
        .find(|t| get(t, &["name"]).as_str() == Some("optmain"))
        .unwrap()
        .clone();
    assert_eq!(
        get(&optmain, &["execution", "taskSupport"]).as_str(),
        Some("optional")
    );
    let r = legacy.result(
        "tools/call",
        r#"{"name":"optmain","arguments":{"INPUT":["ar7.xyz"],"othersoft":"lj"},"task":{"ttl":60000}}"#,
    );
    let t = get(&r, &["task", "taskId"]).as_str().unwrap().to_string();
    assert_eq!(get(&r, &["task", "status"]).as_str(), Some("working"));
    let r = legacy.request("tools/call", r#"{"name":"job_list","task":{}}"#);
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
    let result = legacy.result("tasks/result", &format!(r#"{{"taskId":"{t}"}}"#));
    assert_eq!(get(&result, &["isError"]), &JsonValue::Bool(false));
    assert_eq!(
        get(
            &result,
            &["_meta", "io.modelcontextprotocol/related-task", "taskId"]
        )
        .as_str(),
        Some(t.as_str())
    );
    let status = legacy.wait_notification("notifications/tasks/status", Duration::from_secs(30));
    assert_eq!(
        get(&status, &["params", "taskId"]).as_str(),
        Some(t.as_str())
    );
    let listed = legacy.result("tasks/list", "{}");
    let ids: Vec<String> = get(&listed, &["tasks"])
        .as_array()
        .unwrap()
        .iter()
        .map(|x| get(x, &["taskId"]).as_str().unwrap().to_string())
        .collect();
    assert!(
        ids.contains(&t) && ids.contains(&task) && ids.contains(&long),
        "{ids:?}"
    );
    let r = legacy.result("tasks/get", &format!(r#"{{"taskId":"{long}"}}"#));
    assert_eq!(get(&r, &["status"]).as_str(), Some("cancelled"));
    let r = legacy.request("tasks/cancel", &format!(r#"{{"taskId":"{t}"}}"#));
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
    legacy.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn subscriptions_tell_when_resources_change() {
    let dir = scratch("subscribe");
    let root = dir.join("root");
    std::fs::write(root.join("watched.txt"), "one\n").unwrap();
    // The handshake revisions: resources/subscribe, notifications/resources/updated.
    let mut c = Client::start(&root);
    c.initialize();
    c.result(
        "resources/subscribe",
        r#"{"uri":"mopt://root/watched.txt"}"#,
    );
    c.result("resources/subscribe", r#"{"uri":"mopt://root/later.txt"}"#);
    std::thread::sleep(Duration::from_millis(600));
    std::fs::write(root.join("watched.txt"), "one\ntwo\n").unwrap();
    let n = c.wait_notification("notifications/resources/updated", Duration::from_secs(30));
    assert_eq!(
        get(&n, &["params", "uri"]).as_str(),
        Some("mopt://root/watched.txt")
    );
    std::fs::write(root.join("later.txt"), "new\n").unwrap();
    let n = c.wait_notification("notifications/resources/updated", Duration::from_secs(30));
    assert_eq!(
        get(&n, &["params", "uri"]).as_str(),
        Some("mopt://root/later.txt")
    );
    c.result(
        "resources/unsubscribe",
        r#"{"uri":"mopt://root/watched.txt"}"#,
    );
    std::fs::write(root.join("watched.txt"), "three\n").unwrap();
    std::thread::sleep(Duration::from_millis(1200));
    let r = c.result("ping", "{}");
    assert_eq!(r, JsonValue::empty_object());
    assert!(c
        .take_notifications("notifications/resources/updated")
        .is_empty());
    c.close();

    // 2026-07-28: subscriptions/listen; a URI that is refused is left out of the agreement.
    let mut m = Client::start(&root);
    let listen = m.next_id();
    m.send(&format!(
        r#"{{"jsonrpc":"2.0","id":{listen},"method":"subscriptions/listen","params":{{"notifications":{{"resourceSubscriptions":["mopt://root/watched.txt","mopt://root/../outside.txt","mopt://docs/mcp"],"toolsListChanged":true}},{}}}}}"#,
        meta(false)
    ));
    let ack = m.wait_notification(
        "notifications/subscriptions/acknowledged",
        Duration::from_secs(30),
    );
    let agreed = get(&ack, &["params", "notifications", "resourceSubscriptions"])
        .as_string_vec()
        .unwrap();
    assert_eq!(
        agreed,
        vec![
            "mopt://root/watched.txt".to_string(),
            "mopt://docs/mcp".to_string()
        ]
    );
    assert!(get(&ack, &["params", "notifications"])
        .get("toolsListChanged")
        .is_none());
    std::thread::sleep(Duration::from_millis(600));
    std::fs::write(root.join("watched.txt"), "four\n").unwrap();
    let n = m.wait_notification("notifications/resources/updated", Duration::from_secs(30));
    assert_eq!(
        get(
            &n,
            &["params", "_meta", "io.modelcontextprotocol/subscriptionId"]
        ),
        &JsonValue::Number(listen as f64)
    );
    // The client ends the stream: no notification after it, and no answer to it.
    m.send(&format!(
        r#"{{"jsonrpc":"2.0","method":"notifications/cancelled","params":{{"requestId":{listen}}}}}"#
    ));
    std::thread::sleep(Duration::from_millis(600));
    std::fs::write(root.join("watched.txt"), "five\n").unwrap();
    std::thread::sleep(Duration::from_millis(1200));
    let r = m.result("server/discover", &format!("{{{}}}", meta(false)));
    assert_eq!(get(&r, &["resultType"]).as_str(), Some("complete"));
    assert!(m
        .take_notifications("notifications/resources/updated")
        .is_empty());
    // resources/subscribe is of the handshake revisions only.
    let r = m.request(
        "resources/subscribe",
        &format!(r#"{{"uri":"mopt://root/watched.txt",{}}}"#, meta(false)),
    );
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32601.0));
    m.close();
    let _ = std::fs::remove_dir_all(&dir);
}

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn component_tools_answer_at_once() {
    let dir = scratch("components");
    let root = dir.join("root");
    std::fs::write(root.join("water.xyz"), WATER).unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    // The model Hessian of GFN1-xTB is the Hessian of the calculator gfn1xtb.
    let model = c.ok_tool(
        "model_hessian",
        r#"{"structure":"water.xyz","name":"gfn1xtb"}"#,
    );
    let point = c.ok_tool(
        "single_point",
        r#"{"structure":"water.xyz","calculator":"gfn1xtb","hessian":true}"#,
    );
    let a = floats(get(&model, &["hessian_hartree_per_bohr2"]));
    let b = floats(get(&point, &["hessian_hartree_per_bohr2"]));
    assert_eq!(a.len(), 81);
    let diff = a
        .iter()
        .zip(&b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(diff < 1e-10, "{diff}");
    let text = c.refused_tool(
        "model_hessian",
        r#"{"structure":"water.xyz","name":"nonsense"}"#,
    );
    assert!(text.contains("unknown model Hessian"), "{text}");
    // Normal modes from the calculator: three vibrations of water.
    let v = c.ok_tool(
        "vibrations",
        r#"{"structure":"water.xyz","calculator":"gfn1xtb"}"#,
    );
    let w = floats(get(&v, &["wavenumbers_cm1"]));
    assert_eq!(w.iter().filter(|x| **x > 500.0).count(), 3, "{w:?}");
    assert!(get(&v, &["zero_point_energy"]).as_f64().unwrap() > 0.0);
    let from_given = c.ok_tool(
        "vibrations",
        &format!(
            r#"{{"structure":"water.xyz","hessian":{}}}"#,
            get(&point, &["hessian_hartree_per_bohr2"]).to_string_compact()
        ),
    );
    let w2 = floats(get(&from_given, &["wavenumbers_cm1"]));
    assert!(w.iter().zip(&w2).all(|(x, y)| (x - y).abs() < 1e-6));
    let coords = c.ok_tool(
        "coordinates",
        r#"{"structure":"water.xyz","system":"redundant"}"#,
    );
    assert!(get(&coords, &["dimension"]).as_f64().unwrap() >= 3.0);
    let path = c.ok_tool(
        "interpolate",
        &format!(
            r#"{{"start":"water.xyz","end_xyz":{},"images":4,"method":"idpp"}}"#,
            JsonValue::String(
                WATER.replace("0.757 -0.469\nH 0.0 -0.757", "0.70 -0.52\nH 0.0 -0.80")
            )
            .to_string_compact()
        ),
    );
    assert_eq!(get(&path, &["frames"]), &JsonValue::Number(6.0));
    let s = c.ok_tool(
        "superpose",
        &format!(
            r#"{{"structure":"water.xyz","reference_xyz":{}}}"#,
            JsonValue::String(WATER.to_string()).to_string_compact()
        ),
    );
    assert!(get(&s, &["rmsd_angstrom"]).as_f64().unwrap() < 1e-8);
    let g = c.ok_tool("bond_graph", r#"{"structure":"water.xyz"}"#);
    assert_eq!(get(&g, &["bonds"]).as_array().unwrap().len(), 2);
    let d = c.ok_tool("dispersion", r#"{"structure":"water.xyz","model":"d4"}"#);
    assert!(get(&d, &["energy_hartree"]).as_f64().unwrap() < 0.0);
    let u = c.ok_tool(
        "hessian_update",
        r#"{"hessian":[1,0,0,1],"step":[0.1,0.0],"gradient_change":[0.3,0.0],"method":"sr1"}"#,
    );
    assert!((floats(get(&u, &["hessian"]))[0] - 3.0).abs() < 1e-12);
    let units = c.ok_tool("units", r#"{"value":1,"from":"ev","to":"kcal/mol"}"#);
    assert!((get(&units, &["value"]).as_f64().unwrap() - 23.06).abs() < 0.01);
    let e = c.ok_tool("elements", r#"{"elements":["Cu",79]}"#);
    assert_eq!(
        get(&get(&e, &["elements"]).as_array().unwrap()[1], &["symbol"]).as_str(),
        Some("Au")
    );
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_binary_serves_every_page_of_the_documentation_outside_the_repository() {
    let dir = scratch("docs");
    let root = dir.join("root");
    let mut c = Client::start(&root);
    c.initialize();
    let listed = c.result("resources/list", "{}");
    let uris = fields(&listed, "resources", "uri");
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/source");
    for page in std::fs::read_dir(&source).unwrap() {
        let page = page.unwrap().path();
        if page.extension().is_some_and(|e| e == "md") {
            let name = page.file_stem().unwrap().to_string_lossy().into_owned();
            let uri = format!("mopt://docs/{name}");
            assert!(uris.contains(&uri), "{uri} is not embedded in the binary");
            let r = c.result("resources/read", &format!(r#"{{"uri":"{uri}"}}"#));
            let content = get(&get(&r, &["contents"]).as_array().unwrap()[0], &["text"]);
            assert_eq!(
                content.as_str().unwrap(),
                std::fs::read_to_string(&page).unwrap(),
                "{name}"
            );
        }
    }
    let r = c.request("resources/read", r#"{"uri":"mopt://docs/no_such_page"}"#);
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32002.0));
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_short_reaction_network_of_lj7_runs_as_a_job() {
    let dir = scratch("mapper");
    let root = dir.join("root");
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    std::fs::write(
        root.join("mapper.json"),
        r#"{"mapper_settings":{"afir_gamma_kJmol":2.0,"dist_upper_ang":7.0,"max_iterations":2,"rng_seed":42},
"top_n_candidates":1,"step1_settings":{"othersoft":"lj","NSTEP":300},
"step3_settings":{"tight_convergence_criteria":true,"calc_exact_hess":1,"NSTEP":200},
"step4_settings":{"tight_convergence_criteria":true,"calc_exact_hess":1,"NSTEP":400,"intrinsic_reaction_coordinates":["0.5","200","lqa"]}}"#,
    )
    .unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    let started = c.ok_tool(
        "run_mapper",
        r#"{"INPUT":"ar7.xyz","config_file":"mapper.json","files":["mapper.json"]}"#,
    );
    let job = get(&started, &["job_id"]).as_str().unwrap().to_string();
    let mut status = c.ok_tool(
        "job_wait",
        &format!(r#"{{"job_id":"{job}","timeout_seconds":600}}"#),
    );
    while get(&status, &["state"]).as_str() == Some("running") {
        status = c.ok_tool(
            "job_wait",
            &format!(r#"{{"job_id":"{job}","timeout_seconds":600}}"#),
        );
    }
    assert_eq!(
        get(&status, &["state"]).as_str(),
        Some("finished"),
        "{status:?}"
    );
    let result = c.ok_tool("job_result", &format!(r#"{{"job_id":"{job}"}}"#));
    let files: Vec<String> = get(&result, &["run_directories"])
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            get(r, &["files"])
                .as_array()
                .unwrap()
                .iter()
                .map(|f| get(f, &["path"]).as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        files.iter().any(|f| f.ends_with("reaction_network.json")),
        "{files:?}"
    );
    // A run steered while it goes on: a limit of one job, read before its next choice.
    std::fs::write(
        root.join("long.json"),
        std::fs::read_to_string(root.join("mapper.json"))
            .unwrap()
            .replace(r#""max_iterations":2"#, r#""max_iterations":200"#),
    )
    .unwrap();
    let started = c.ok_tool(
        "run_mapper",
        r#"{"INPUT":"ar7.xyz","config_file":"long.json","files":["long.json"]}"#,
    );
    let long = get(&started, &["job_id"]).as_str().unwrap().to_string();
    let text = c.refused_tool(
        "mapper_steer",
        &format!(r#"{{"job_id":"{long}","lines":["max_jobs 1","rm -rf /"]}}"#),
    );
    assert!(text.contains("not a steering line"), "{text}");
    let steered = c.ok_tool(
        "mapper_steer",
        &format!(r#"{{"job_id":"{long}","lines":["max_jobs 1"]}}"#),
    );
    assert!(get(&steered, &["steer_file"])
        .as_str()
        .unwrap()
        .ends_with("steer.txt"));
    let mut status = c.ok_tool(
        "job_wait",
        &format!(r#"{{"job_id":"{long}","timeout_seconds":600}}"#),
    );
    while get(&status, &["state"]).as_str() == Some("running") {
        status = c.ok_tool(
            "job_wait",
            &format!(r#"{{"job_id":"{long}","timeout_seconds":600}}"#),
        );
    }
    assert_eq!(
        get(&status, &["state"]).as_str(),
        Some("finished"),
        "{status:?}"
    );
    let log_uri = get(&steered, &["log"]).as_str().unwrap().to_string();
    let log = c.result("resources/read", &format!(r#"{{"uri":"{log_uri}"}}"#));
    let log = get(&get(&log, &["contents"]).as_array().unwrap()[0], &["text"])
        .as_str()
        .unwrap()
        .to_string();
    assert!(log.contains("max_jobs 1"), "{log}");
    // Only a running run_mapper job is steered.
    let text = c.refused_tool(
        "mapper_steer",
        &format!(r#"{{"job_id":"{long}","lines":["max_jobs 1"]}}"#),
    );
    assert!(text.contains("not running"), "{text}");
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}
