//! Attacks on the MCP server: every way a client could reach outside the root, start another
//! program or load code, or end the server, must be refused, and the server must keep
//! answering.
//!
//! Directory traversal with `..` (also when it stays inside), absolute paths, drives, UNC
//! paths, device names, NUL characters; symbolic links inside the root that lead out (files,
//! directories, links inside a copied directory, a link swapped in after a check); encoded
//! separators in resource URIs; job ids that are paths or not made by the server; option
//! values and settings files that name files outside the root; options that start external
//! programs or load code; malformed and oversized messages.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod mcp_client;

use std::path::Path;

use mcp_client::{get, scratch, Client, AR7};
use mopt_rs::core_infra::json::JsonValue;

/// The number of job directories under the root (refused requests leave none).
fn jobs_in(root: &Path) -> usize {
    std::fs::read_dir(root.join("mcp_jobs"))
        .map(|d| d.count())
        .unwrap_or(0)
}

fn optmain(extra: &str) -> String {
    format!(r#"{{"INPUT":["ar7.xyz"],"othersoft":"lj","NSTEP":2{extra}}}"#)
}

#[test]
fn traversal_absolute_paths_and_special_names_are_refused() {
    let dir = scratch("attack_paths");
    let root = dir.join("root");
    std::fs::write(dir.join("outside.xyz"), AR7).unwrap();
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::write(root.join("sub/ar7.xyz"), AR7).unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    let outside = dir.join("outside.xyz").to_string_lossy().into_owned();
    let inside = root
        .join("ar7.xyz")
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let inputs = [
        "../outside.xyz".to_string(),
        "sub/../ar7.xyz".to_string(),
        "./../outside.xyz".to_string(),
        "..".to_string(),
        ".".to_string(),
        outside.clone(),
        "C:/outside.xyz".to_string(),
        "C:outside.xyz".to_string(),
        "//server/share/outside.xyz".to_string(),
        "\\\\server\\share\\outside.xyz".to_string(),
        "sub\\..\\..\\outside.xyz".to_string(),
        "NUL".to_string(),
        "con.xyz".to_string(),
        "sub/aux.xyz".to_string(),
        "ar7.xyz.".to_string(),
        "ar7.xyz:stream".to_string(),
        "ar7\u{0}.xyz".to_string(),
        "mcp_jobs".to_string(),
    ];
    for input in &inputs {
        let args = format!(
            r#"{{"INPUT":[{}],"othersoft":"lj"}}"#,
            JsonValue::String(input.clone()).to_string_compact()
        );
        c.refused_tool("optmain", &args);
        let args = format!(
            r#"{{"structure":{},"calculator":"lj"}}"#,
            JsonValue::String(input.clone()).to_string_compact()
        );
        if input != "mcp_jobs" && input != "." {
            c.refused_tool("single_point", &args);
        }
        let args = format!(
            r#"{{"INPUT":["ar7.xyz"],"othersoft":"lj","files":[{}]}}"#,
            JsonValue::String(input.clone()).to_string_compact()
        );
        c.refused_tool("optmain", &args);
    }
    assert_eq!(
        jobs_in(&root),
        0,
        "a refused request leaves no job directory"
    );
    // An absolute path inside the root is the same file as its relative path.
    let args = format!(
        r#"{{"structure":{},"calculator":"lj"}}"#,
        JsonValue::String(inside).to_string_compact()
    );
    c.ok_tool("single_point", &args);
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn option_values_and_settings_files_that_name_files_outside_are_refused() {
    let dir = scratch("attack_values");
    let root = dir.join("root");
    std::fs::write(dir.join("outside.xyz"), AR7).unwrap();
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    let outside = dir.join("outside.xyz").to_string_lossy().into_owned();
    for extra in [
        format!(
            r#","initial_hessian":{}"#,
            JsonValue::String(outside.clone()).to_string_compact()
        ),
        r#","initial_hessian":"/etc/passwd""#.to_string(),
        r#","initial_hessian":"../../../outside.xyz""#.to_string(),
        r#","initial_hessian":"~/.ssh/id_rsa""#.to_string(),
        r#","initial_hessian":"C:\\system\\win.ini""#.to_string(),
        r#","initial_hessian":"\\\\server\\share\\h.npy""#.to_string(),
        r#","calc_option":["param=/etc/passwd"]"#.to_string(),
        r#","calc_option":["param=@/etc/passwd"]"#.to_string(),
        r#","calc_option":["param=../../outside.xyz"]"#.to_string(),
        r#","fix_atoms":["1,../../x"]"#.to_string(),
        r#","coordinate_system":"hyperspherical:/etc/passwd""#.to_string(),
        r#","coordinate_system":"hyperspherical:../../outside.xyz""#.to_string(),
        r#","calc_option":["param=a /etc/passwd"]"#.to_string(),
    ] {
        c.refused_tool("optmain", &optmain(&extra));
    }
    // The same file named twice, or inside a directory that is copied too, is copied once.
    std::fs::create_dir_all(root.join("set")).unwrap();
    std::fs::write(root.join("set/a.xyz"), AR7).unwrap();
    let started = c.ok_tool(
        "optmain",
        r#"{"INPUT":["ar7.xyz","ar7.xyz"],"othersoft":"lj","NSTEP":2,"files":["ar7.xyz","set","set/a.xyz"]}"#,
    );
    let job_dir = root.join(get(&started, &["job_directory"]).as_str().unwrap());
    assert!(job_dir.join("set/a.xyz").is_file() && job_dir.join("ar7.xyz").is_file());
    let job = get(&started, &["job_id"]).as_str().unwrap().to_string();
    c.wait(&job, std::time::Duration::from_secs(300));
    std::fs::remove_dir_all(root.join("mcp_jobs")).unwrap();
    // A settings file of run_autots that names a file outside the root, or a calculator that
    // loads code, is refused when it is copied in.
    std::fs::write(
        root.join("autots.json"),
        r#"{"step1_settings": {"initial_hessian": "/etc/passwd"}}"#,
    )
    .unwrap();
    c.refused_tool(
        "run_autots",
        r#"{"INPUT":"ar7.xyz","config_file":"autots.json","files":["autots.json"]}"#,
    );
    std::fs::write(
        root.join("autots.json"),
        r#"{"step1_settings": {"othersoft": "ase:ase.calculators.emt:EMT"}}"#,
    )
    .unwrap();
    c.refused_tool(
        "run_autots",
        r#"{"INPUT":"ar7.xyz","config_file":"autots.json","files":["autots.json"]}"#,
    );
    assert_eq!(jobs_in(&root), 0);
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn external_programs_and_code_need_allow_external() {
    let dir = scratch("attack_external");
    let root = dir.join("root");
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    std::fs::write(root.join("software_path.conf"), "orca::/usr/bin/orca\n").unwrap();
    std::fs::write(
        root.join("mopt_profiles.json"),
        r#"{"x": {"factory": "os:system"}}"#,
    )
    .unwrap();
    std::fs::write(
        root.join("mopt_calculator_options.json"),
        r#"{"pyscf": {"verbose": 4}}"#,
    )
    .unwrap();
    std::fs::create_dir_all(root.join("conf")).unwrap();
    std::fs::write(
        root.join("conf/software_path.conf"),
        "orca::/usr/bin/orca\n",
    )
    .unwrap();
    let refused = [
        r#"{"INPUT":["ar7.xyz"],"othersoft":"ase:ase.calculators.emt:EMT"}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"orca"}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"multioptpy:gfn2"}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"openmm"}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","calc_option":["hook=os:system"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","calc_option":["hook.cmd=x"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","calc_option":["excited_setup=fake_ase:choose_root"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","calc_option":["state_reader=file"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","calc_option":["x=pkg.module:function"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","oniom_flag":["1-3","auto","ase:m:C"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","oniom_options":["low.system=model.xml"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","software_path_file":"software_path.conf"}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","files":["software_path.conf"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","files":["mopt_profiles.json"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","files":["mopt_calculator_options.json"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","files":["conf"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","oniom_options":["low.calc_option=hook=os:system"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","calc_option":["x=a.b:c,d.e:f"]}"#,
        r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","calc_option":["conv_tol=1e-9,hook=m:f"]}"#,
    ];
    let mut c = Client::start(&root);
    c.initialize();
    for args in refused {
        let text = c.refused_tool("optmain", args);
        assert!(text.contains("--allow-external"), "{args}: {text}");
    }
    let text = c.refused_tool(
        "single_point",
        r#"{"structure":"ar7.xyz","calculator":"ase:ase.calculators.emt:EMT"}"#,
    );
    assert!(text.contains("--allow-external"), "{text}");
    let text = c.refused_tool(
        "single_point",
        r#"{"structure":"ar7.xyz","calculator":"lj","settings":{"hook":"os:system"}}"#,
    );
    assert!(text.contains("--allow-external"), "{text}");
    assert_eq!(jobs_in(&root), 0);
    // Only the commands of the tables are tools; there is no tool that names a program.
    let r = c.request(
        "tools/call",
        r#"{"name":"sh","arguments":{"INPUT":["ar7.xyz"]}}"#,
    );
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
    let r = c.request("tools/call", r#"{"name":"run-job","arguments":{}}"#);
    assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
    c.close();

    // With the flag, the settings pass the server (whether the job can use them is the
    // command's matter: the Rust binary has no ASE).
    let mut open = Client::start_with(&root, &["--allow-external"]);
    open.initialize();
    open.ok_tool("optmain", r#"{"INPUT":["ar7.xyz"],"othersoft":"lj","calc_option":["cutoff=8.0"],"files":["software_path.conf"]}"#);
    open.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn job_ids_are_tokens_of_the_server_never_paths() {
    let dir = scratch("attack_ids");
    let root = dir.join("root");
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    let started = c.ok_tool("optmain", &optmain(""));
    let job = get(&started, &["job_id"]).as_str().unwrap().to_string();
    let job_dir = get(&started, &["job_directory"])
        .as_str()
        .unwrap()
        .to_string();
    // A directory made by hand with a record of its own is not a job of the server.
    std::fs::create_dir_all(root.join("mcp_jobs/fake")).unwrap();
    std::fs::write(
        root.join("mcp_jobs/fake/mcp_job.json"),
        r#"{"job_id":"fake","command":"optmain"}"#,
    )
    .unwrap();
    for id in [
        "../x".to_string(),
        "..".to_string(),
        "/etc".to_string(),
        "mcp_jobs".to_string(),
        job_dir.clone(),
        format!("{job}/.."),
        format!("../{job}"),
        job.to_uppercase(),
        "fake".to_string(),
        "job-../../../../etc/passwd".to_string(),
        String::new(),
    ] {
        for tool in ["job_status", "job_stop", "job_restart", "job_result"] {
            let args = format!(
                r#"{{"job_id":{}}}"#,
                JsonValue::String(id.clone()).to_string_compact()
            );
            c.refused_tool(tool, &args);
        }
    }
    // The run directory of a restart must be one of the job's.
    c.wait(&job, std::time::Duration::from_secs(300));
    for run in ["..", "/etc", "ar7.xyz", "mcp_jobs/fake"] {
        let args = format!(r#"{{"job_id":"{job}","run_directory":"{run}"}}"#);
        c.refused_tool("job_restart", &args);
    }
    let listed = c.ok_tool("job_list", "{}");
    assert_eq!(
        get(&listed, &["jobs"]).as_array().unwrap().len(),
        1,
        "the hand-made directory is no job"
    );
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resource_uris_with_encoded_or_climbing_paths_are_refused() {
    let dir = scratch("attack_uris");
    let root = dir.join("root");
    std::fs::write(dir.join("outside.xyz"), AR7).unwrap();
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    std::fs::write(root.join("big.txt"), vec![b'x'; (8 << 20) + 1]).unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    for uri in [
        "mopt://root/../outside.xyz",
        "mopt://root/%2e%2e/outside.xyz",
        "mopt://root/%2E%2E%2Foutside.xyz",
        "mopt://root/..%2foutside.xyz",
        "mopt://root/..%5coutside.xyz",
        "mopt://root/%252e%252e/outside.xyz",
        "mopt://root/ar7.xyz%00.txt",
        "mopt://root//etc/passwd",
        "mopt://root/C:/system/win.ini",
        "mopt://root/NUL",
        "mopt://root/big.txt",
        "file:///etc/passwd",
        "mopt://docs/../../README",
        "mopt://docs/%2e%2e%2fREADME",
    ] {
        let r = c.request(
            "resources/read",
            &format!(
                r#"{{"uri":{}}}"#,
                JsonValue::String(uri.into()).to_string_compact()
            ),
        );
        assert!(r.get("error").is_some(), "{uri}: {r:?}");
    }
    let r = c.result("resources/read", r#"{"uri":"mopt://root/ar7.xyz"}"#);
    assert!(get(&r, &["contents"]).as_array().unwrap()[0]
        .get("text")
        .is_some());
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn symbolic_links_inside_the_root_do_not_lead_out() {
    use std::os::unix::fs::symlink;
    let dir = scratch("attack_links");
    let root = dir.join("root");
    std::fs::write(dir.join("outside.xyz"), AR7).unwrap();
    std::fs::create_dir_all(dir.join("elsewhere")).unwrap();
    std::fs::write(dir.join("elsewhere/secret.xyz"), AR7).unwrap();
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    symlink(dir.join("outside.xyz"), root.join("link.xyz")).unwrap();
    symlink(dir.join("elsewhere"), root.join("updir")).unwrap();
    std::fs::create_dir_all(root.join("inputs")).unwrap();
    std::fs::write(root.join("inputs/a.xyz"), AR7).unwrap();
    symlink(dir.join("outside.xyz"), root.join("inputs/b.xyz")).unwrap();
    symlink(root.join("ar7.xyz"), root.join("inputs/c.xyz")).unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    for input in ["link.xyz", "updir/secret.xyz", "updir"] {
        c.refused_tool(
            "optmain",
            &format!(r#"{{"INPUT":["{input}"],"othersoft":"lj"}}"#),
        );
        c.refused_tool(
            "optmain",
            &format!(r#"{{"INPUT":["ar7.xyz"],"othersoft":"lj","files":["{input}"]}}"#),
        );
    }
    c.refused_tool(
        "single_point",
        r#"{"structure":"link.xyz","calculator":"lj"}"#,
    );
    for uri in [
        "mopt://root/link.xyz",
        "mopt://root/updir/secret.xyz",
        "mopt://root/updir",
    ] {
        let r = c.request("resources/read", &format!(r#"{{"uri":"{uri}"}}"#));
        assert!(r.get("error").is_some(), "{uri}: {r:?}");
    }
    // A directory listing names the links but does not follow them.
    let r = c.result("resources/read", r#"{"uri":"mopt://root/"}"#);
    let listing = get(&r, &["contents"]).as_array().unwrap()[0].clone();
    let listing = get(&listing, &["text"]).as_str().unwrap().to_string();
    assert!(!listing.contains("secret.xyz"), "{listing}");
    // A copied directory keeps its files and the links inside the root, never the links that
    // lead out.
    let started = c.ok_tool(
        "nebmain",
        r#"{"INPUT":["inputs"],"othersoft":"lj","NSTEP":1}"#,
    );
    let skipped = get(&started, &["skipped_links"]).as_string_vec().unwrap();
    assert_eq!(skipped, vec!["inputs/b.xyz".to_string()]);
    let job_dir = root.join(get(&started, &["job_directory"]).as_str().unwrap());
    assert!(job_dir.join("inputs/a.xyz").is_file());
    assert!(job_dir.join("inputs/c.xyz").is_file());
    assert!(
        !job_dir.join("inputs/c.xyz").is_symlink(),
        "copies are regular files"
    );
    assert!(!job_dir.join("inputs/b.xyz").exists());
    // A FIFO (which would block a reader) is never opened: refused as an input, a structure
    // and a resource, skipped inside a copied directory.
    let made = std::process::Command::new("mkfifo")
        .arg(root.join("pipe"))
        .status();
    if made.is_ok_and(|s| s.success()) {
        c.refused_tool("optmain", r#"{"INPUT":["pipe"],"othersoft":"lj"}"#);
        c.refused_tool("single_point", r#"{"structure":"pipe","calculator":"lj"}"#);
        let r = c.request("resources/read", r#"{"uri":"mopt://root/pipe"}"#);
        assert!(r.get("error").is_some(), "{r:?}");
        std::process::Command::new("mkfifo")
            .arg(root.join("inputs/pipe"))
            .status()
            .unwrap();
        let started = c.ok_tool(
            "nebmain",
            r#"{"INPUT":["inputs"],"othersoft":"lj","NSTEP":1}"#,
        );
        let skipped = get(&started, &["skipped_links"]).as_string_vec().unwrap();
        assert!(skipped.contains(&"inputs/pipe".to_string()), "{skipped:?}");
        let other = get(&started, &["job_id"]).as_str().unwrap().to_string();
        c.wait(&other, std::time::Duration::from_secs(300));
    } else {
        eprintln!("mkfifo is not available: the FIFO attacks were not run");
    }
    // The directory of the jobs may not be swapped for a link that leads out.
    let job = get(&started, &["job_id"]).as_str().unwrap().to_string();
    c.wait(&job, std::time::Duration::from_secs(300));
    std::fs::rename(root.join("mcp_jobs"), root.join("old_jobs")).unwrap();
    symlink(dir.join("elsewhere"), root.join("mcp_jobs")).unwrap();
    c.refused_tool("optmain", &optmain(""));
    c.refused_tool("job_status", &format!(r#"{{"job_id":"{job}"}}"#));
    assert_eq!(
        std::fs::read_dir(dir.join("elsewhere")).unwrap().count(),
        1,
        "nothing written there"
    );
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn malformed_and_oversized_messages_are_answered_and_the_server_goes_on() {
    let dir = scratch("attack_messages");
    let root = dir.join("root");
    let mut c = Client::start(&root);
    for (line, code) in [
        ("{not json".to_string(), -32700.0),
        ("[]".to_string(), -32600.0),
        (
            r#"[{"jsonrpc":"2.0","id":1,"method":"ping"}]"#.to_string(),
            -32600.0,
        ),
        ("42".to_string(), -32600.0),
        (
            r#"{"jsonrpc":"1.0","id":1,"method":"ping"}"#.to_string(),
            -32600.0,
        ),
        (
            r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#.to_string(),
            -32600.0,
        ),
        (
            r#"{"jsonrpc":"2.0","id":{"a":1},"method":"ping"}"#.to_string(),
            -32600.0,
        ),
        (
            r#"{"jsonrpc":"2.0","id":1,"method":7}"#.to_string(),
            -32600.0,
        ),
        (
            r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":"x"}"#.to_string(),
            -32600.0,
        ),
        ("[".repeat(200_000), -32600.0),
        (
            format!(
                r#"{{"jsonrpc":"2.0","id":1,"method":"ping","params":{{"x":"{}"}}}}"#,
                "a".repeat((16 << 20) + 10)
            ),
            -32600.0,
        ),
        ("\u{feff}{".to_string(), -32700.0),
    ] {
        c.send(&line);
        let r = c.read();
        assert_eq!(
            get(&r, &["error", "code"]),
            &JsonValue::Number(code),
            "{}: {r:?}",
            &line[..line.len().min(60)]
        );
    }
    // Invalid UTF-8 and an unterminated message.
    c.send_bytes(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"\xff\xfe\"}\n");
    let r = c.read();
    assert!(
        r.get("error").is_some() || r.get("result").is_none(),
        "{r:?}"
    );
    c.send(r#"{"jsonrpc":"2.0","id":2,"method":"ping""#);
    assert_eq!(
        get(&c.read(), &["error", "code"]),
        &JsonValue::Number(-32700.0)
    );
    // Wrong types inside a valid request.
    c.initialize();
    for params in [
        r#"{"name":5}"#,
        r#"{"name":"job_status","arguments":"x"}"#,
        r#"{"name":"job_status","arguments":{"job_id":5}}"#,
        r#"{"name":"optmain","arguments":{"INPUT":[1,2]}}"#,
        r#"{"name":"optmain","arguments":{"INPUT":["a.xyz"],"NSTEP":{"x":1}}}"#,
    ] {
        let r = c.request("tools/call", params);
        let failed = r.get("error").is_some()
            || r.get("result").and_then(|x| x.get("isError")) == Some(&JsonValue::Bool(true));
        assert!(failed, "{params}: {r:?}");
    }
    // A single point, which the server computes while it waits, is bounded in size.
    let big = JsonValue::String(mcp_client::argon_grid(13)).to_string_compact();
    let text = c.refused_tool(
        "single_point",
        &format!(r#"{{"xyz":{big},"calculator":"lj"}}"#),
    );
    assert!(text.contains("at most"), "{text}");
    let middle = JsonValue::String(mcp_client::argon_grid(7)).to_string_compact();
    let text = c.refused_tool(
        "single_point",
        &format!(r#"{{"xyz":{middle},"calculator":"lj","hessian":true}}"#),
    );
    assert!(text.contains("with the Hessian"), "{text}");
    // Still answering.
    assert_eq!(c.result("ping", "{}"), JsonValue::empty_object());
    c.close();
    let _ = std::fs::remove_dir_all(&dir);
}

/// `_meta` of a stateless request with the tasks extension.
fn modern_meta() -> String {
    r#""_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{"extensions":{"io.modelcontextprotocol/tasks":{}}}}"#.to_string()
}

fn quoted(s: &str) -> String {
    JsonValue::String(s.to_string()).to_string_compact()
}

#[test]
fn waits_tasks_subscriptions_cursors_and_component_tools_refuse_what_is_not_theirs() {
    let dir = scratch("attack_new");
    let root = dir.join("root");
    std::fs::write(dir.join("outside.xyz"), AR7).unwrap();
    std::fs::write(root.join("ar7.xyz"), AR7).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(dir.join("outside.xyz"), root.join("link.xyz")).unwrap();
    let mut c = Client::start(&root);
    c.initialize();
    let started = c.ok_tool("optmain", &optmain(""));
    let job = get(&started, &["job_id"]).as_str().unwrap().to_string();
    c.wait(&job, std::time::Duration::from_secs(300));
    std::fs::create_dir_all(root.join("mcp_jobs/fake")).unwrap();
    std::fs::write(
        root.join("mcp_jobs/fake/mcp_job.json"),
        r#"{"job_id":"fake","command":"optmain"}"#,
    )
    .unwrap();
    let bogus = [
        "../x".to_string(),
        "/etc".to_string(),
        "fake".to_string(),
        format!("{job}/.."),
        format!("../{job}"),
        job.to_uppercase(),
        "job-0123456789abcdef0123456789abcdef".to_string(),
        "job-../../../../etc/passwd".to_string(),
        String::new(),
    ];
    // Job ids of the new job tools, task ids, job resources.
    for id in &bogus {
        for tool in ["job_wait", "job_cancel"] {
            c.refused_tool(tool, &format!(r#"{{"job_id":{}}}"#, quoted(id)));
        }
        for method in ["tasks/get", "tasks/cancel", "tasks/result"] {
            let r = c.request(method, &format!(r#"{{"taskId":{}}}"#, quoted(id)));
            assert_eq!(
                get(&r, &["error", "code"]),
                &JsonValue::Number(-32602.0),
                "{method} {id}"
            );
        }
        for method in ["tasks/get", "tasks/cancel"] {
            let r = c.request(
                method,
                &format!(r#"{{"taskId":{},{}}}"#, quoted(id), modern_meta()),
            );
            assert_eq!(
                get(&r, &["error", "code"]),
                &JsonValue::Number(-32602.0),
                "{method} {id}"
            );
        }
        let uri = format!("mopt://jobs/{id}");
        let r = c.request("resources/read", &format!(r#"{{"uri":{}}}"#, quoted(&uri)));
        assert!(r.get("error").is_some(), "{uri}: {r:?}");
        let r = c.request(
            "resources/subscribe",
            &format!(r#"{{"uri":{}}}"#, quoted(&uri)),
        );
        assert!(r.get("error").is_some(), "subscribe {uri}: {r:?}");
    }
    // Waits: a negative or wrong timeout is refused; a huge one is bounded (answered at once
    // for a job that does not run).
    c.refused_tool(
        "job_wait",
        &format!(r#"{{"job_id":"{job}","timeout_seconds":-1}}"#),
    );
    c.refused_tool(
        "job_wait",
        &format!(r#"{{"job_id":"{job}","timeout_seconds":"x"}}"#),
    );
    let w = c.ok_tool(
        "job_wait",
        &format!(r#"{{"job_id":"{job}","timeout_seconds":1e300}}"#),
    );
    assert_eq!(get(&w, &["timed_out"]), &JsonValue::Bool(false));
    // Cursors that the server did not give out, on every list.
    let first = c.result("tools/list", "{}");
    assert!(first.get("nextCursor").is_none());
    for list in [
        "tools/list",
        "resources/list",
        "prompts/list",
        "resources/templates/list",
        "tasks/list",
    ] {
        for cursor in [
            quoted(""),
            quoted("0"),
            quoted("1.0000000000000000"),
            quoted("../../etc/passwd"),
            quoted(&"f".repeat(10_000)),
            "1".to_string(),
            "{}".to_string(),
        ] {
            let r = c.request(list, &format!(r#"{{"cursor":{cursor}}}"#));
            assert_eq!(
                get(&r, &["error", "code"]),
                &JsonValue::Number(-32602.0),
                "{list} {cursor}"
            );
        }
    }
    // Subscriptions: what resources/read refuses is refused here as well.
    for uri in [
        "mopt://root/../outside.xyz",
        "mopt://root/%2e%2e/outside.xyz",
        "mopt://root/..%2foutside.xyz",
        "mopt://root/sub/../../outside.xyz",
        "mopt://root//etc/passwd",
        "mopt://root/C:/x",
        "mopt://root/NUL",
        "mopt://root/a%00b",
        "file:///etc/passwd",
        "mopt://docs/../../README",
        "mopt://docs/no_such_page",
        "mopt://root/link.xyz",
    ] {
        let r = c.request(
            "resources/subscribe",
            &format!(r#"{{"uri":{}}}"#, quoted(uri)),
        );
        assert!(r.get("error").is_some(), "subscribe {uri}: {r:?}");
    }
    // A watched file swapped for a link that leads out: it is not read, and changes outside
    // are not told.
    #[cfg(unix)]
    {
        std::fs::write(root.join("watched.txt"), "inside\n").unwrap();
        c.result(
            "resources/subscribe",
            r#"{"uri":"mopt://root/watched.txt"}"#,
        );
        std::thread::sleep(std::time::Duration::from_millis(600));
        std::fs::remove_file(root.join("watched.txt")).unwrap();
        std::os::unix::fs::symlink(dir.join("outside.xyz"), root.join("watched.txt")).unwrap();
        c.wait_notification(
            "notifications/resources/updated",
            std::time::Duration::from_secs(30),
        );
        let r = c.request("resources/read", r#"{"uri":"mopt://root/watched.txt"}"#);
        assert!(r.get("error").is_some(), "{r:?}");
        std::fs::write(dir.join("outside.xyz"), format!("{AR7}\n\n")).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1500));
        c.result("ping", "{}");
        assert!(c
            .take_notifications("notifications/resources/updated")
            .is_empty());
        c.result(
            "resources/unsubscribe",
            r#"{"uri":"mopt://root/watched.txt"}"#,
        );
    }
    // At most 256 resources are watched.
    let mut refused = 0;
    for k in 0..300 {
        let r = c.request(
            "resources/subscribe",
            &format!(r#"{{"uri":"mopt://root/f{k}.txt"}}"#),
        );
        if r.get("error").is_some() {
            refused += 1;
        }
    }
    assert_eq!(refused, 300 - 256);
    // Structure arguments of the component tools are paths under the root, or text.
    let outside = dir.join("outside.xyz").to_string_lossy().into_owned();
    for path in [
        "../outside.xyz",
        outside.as_str(),
        "/etc/passwd",
        "C:/system/win.ini",
        "NUL",
        "sub/../ar7.xyz",
        "ar7.xyz\u{0}",
        "link.xyz",
    ] {
        for (tool, extra) in [
            ("model_hessian", r#","name":"lindh""#),
            ("vibrations", r#","calculator":"lj""#),
            ("coordinates", r#","system":"dic""#),
            ("bond_graph", ""),
            ("dispersion", ""),
        ] {
            c.refused_tool(tool, &format!(r#"{{"structure":{}{extra}}}"#, quoted(path)));
        }
        c.refused_tool(
            "superpose",
            &format!(r#"{{"structure":"ar7.xyz","reference":{}}}"#, quoted(path)),
        );
        c.refused_tool(
            "interpolate",
            &format!(r#"{{"start":"ar7.xyz","end":{}}}"#, quoted(path)),
        );
    }
    c.refused_tool(
        "bond_graph",
        &format!(r#"{{"structure":"ar7.xyz","xyz":{}}}"#, quoted(AR7)),
    );
    // A coordinate system whose argument names a file, programs and code in a calculator.
    for system in [
        "hyperspherical:../outside.xyz",
        "hyperspherical:/etc/passwd",
        "hyperspherical:ar7.xyz",
        "polar:x",
    ] {
        c.refused_tool(
            "coordinates",
            &format!(r#"{{"structure":"ar7.xyz","system":{}}}"#, quoted(system)),
        );
    }
    for calculator in [
        "ase:ase.calculators.emt:EMT",
        "orca",
        "multioptpy:gfn2xtb",
        "pm6_rs",
    ] {
        let text = c.refused_tool(
            "vibrations",
            &format!(
                r#"{{"structure":"ar7.xyz","calculator":{}}}"#,
                quoted(calculator)
            ),
        );
        assert!(text.contains("--allow-external"), "{calculator}: {text}");
    }
    let text = c.refused_tool(
        "vibrations",
        r#"{"structure":"ar7.xyz","calculator":"lj","settings":{"hook":"os:system"}}"#,
    );
    assert!(text.contains("--allow-external"), "{text}");
    // Steering lines are checked by their form, and only run_mapper jobs are steered.
    for line in [
        "explore ../../etc/passwd",
        "explore EQ1\nmax_jobs 0",
        "explore EQ1 # x",
        "max_jobs -1",
        "max_jobs 1e9",
        "skip",
        "shutdown",
        "explore EQ1\u{0}",
        "explore a/b",
    ] {
        let text = c.refused_tool(
            "mapper_steer",
            &format!(r#"{{"job_id":"{job}","lines":[{}]}}"#, quoted(line)),
        );
        assert!(
            text.contains("steering line") || text.contains("control characters"),
            "{line:?}: {text}"
        );
    }
    let text = c.refused_tool(
        "mapper_steer",
        &format!(r#"{{"job_id":"{job}","lines":["max_jobs 1"]}}"#),
    );
    assert!(text.contains("no run_mapper job"), "{text}");
    for id in &bogus {
        c.refused_tool(
            "mapper_steer",
            &format!(r#"{{"job_id":{},"lines":["max_jobs 1"]}}"#, quoted(id)),
        );
    }
    // Sizes are bounded.
    c.refused_tool(
        "interpolate",
        r#"{"start":"ar7.xyz","end":"ar7.xyz","images":1000000}"#,
    );
    c.refused_tool(
        "hessian_update",
        r#"{"hessian":[1,0,0],"step":[1,0],"gradient_change":[1,0]}"#,
    );
    let many = vec!["H"; 5000].join("\",\"");
    c.refused_tool("elements", &format!(r#"{{"elements":["{many}"]}}"#));
    c.close();
    // Streams of subscriptions/listen are bounded, and a refused URI is left out.
    let mut m = Client::start(&root);
    for k in 0..17 {
        let id = m.next_id();
        m.send(&format!(
            r#"{{"jsonrpc":"2.0","id":{id},"method":"subscriptions/listen","params":{{"notifications":{{"resourceSubscriptions":["mopt://root/../outside.xyz"]}},{}}}}}"#,
            modern_meta()
        ));
        if k < 16 {
            let ack = m.wait_notification(
                "notifications/subscriptions/acknowledged",
                std::time::Duration::from_secs(30),
            );
            assert!(get(&ack, &["params", "notifications"])
                .get("resourceSubscriptions")
                .is_none());
        } else {
            let r = m.response_to(id);
            assert_eq!(get(&r, &["error", "code"]), &JsonValue::Number(-32602.0));
        }
    }
    m.close();
    #[cfg(unix)]
    assert_eq!(
        std::fs::read_to_string(dir.join("outside.xyz")).unwrap(),
        format!("{AR7}\n\n"),
        "nothing outside was written"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
