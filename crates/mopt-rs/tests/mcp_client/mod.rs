//! A small MCP client for the tests: `mopt mcp` as a child process, one JSON-RPC message per
//! line.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use mopt_rs::core_infra::json::JsonValue;

/// An argon cluster (UFF argon, compressed).
pub const AR7: &str = "7
argon cluster
Ar   1.9500   0.0000   0.0000
Ar   0.6026   1.8546   0.0300
Ar  -1.5776   1.1462   0.0700
Ar  -1.5776  -1.1462   0.1100
Ar   0.6026  -1.8546   0.1500
Ar   0.0000   0.0000   2.1000
Ar   0.2000   0.0000  -2.0000
";

/// The stateless revision.
pub const MODERN: &str = "2026-07-28";

/// An argon grid far from its minimum: many iterations, each slower than those of AR7.
pub fn argon_grid(n: usize) -> String {
    let mut out = format!("{}\nargon grid\n", n * n * n);
    let mut k = 0usize;
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                k += 1;
                let shake = ((k * 7919) % 13) as f64 * 0.03;
                out.push_str(&format!(
                    "Ar {:.4} {:.4} {:.4}\n",
                    3.6 * i as f64 + shake,
                    3.6 * j as f64 - shake,
                    3.6 * l as f64 + 0.5 * shake
                ));
            }
        }
    }
    out
}

/// The server and its streams; a thread reads the output line by line.
pub struct Client {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    next_id: u64,
    /// The notifications read while waiting for responses, in order.
    pub notifications: Vec<JsonValue>,
}

/// How long a test waits for one message of the server.
pub const READ_TIMEOUT: Duration = Duration::from_secs(900);

impl Client {
    /// `mopt mcp --root ROOT` with further arguments.
    pub fn start_with(root: &Path, extra: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_mopt"))
            .args(["mcp", "--root"])
            .arg(root)
            .args(extra)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let (sender, lines) = channel();
        std::thread::spawn(move || loop {
            let mut line = String::new();
            match stdout.read_line(&mut line) {
                Ok(0) | Err(_) => return,
                Ok(_) => {
                    if sender.send(line).is_err() {
                        return;
                    }
                }
            }
        });
        Self {
            child,
            stdin,
            lines,
            next_id: 0,
            notifications: Vec::new(),
        }
    }

    /// `mopt mcp --root ROOT`.
    pub fn start(root: &Path) -> Self {
        Self::start_with(root, &[])
    }

    /// Write one line.
    pub fn send(&mut self, message: &str) {
        writeln!(self.stdin, "{message}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// Write raw bytes (with their own line end).
    pub fn send_bytes(&mut self, bytes: &[u8]) {
        self.stdin.write_all(bytes).unwrap();
        self.stdin.flush().unwrap();
    }

    /// The next message within `timeout`, if any.
    pub fn read_within(&mut self, timeout: Duration) -> Option<JsonValue> {
        match self.lines.recv_timeout(timeout) {
            Ok(line) => {
                assert!(line.ends_with('\n') && !line.trim().is_empty(), "{line:?}");
                Some(JsonValue::parse(line.trim()).unwrap())
            }
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => panic!("the server closed its output"),
        }
    }

    /// Read one message.
    pub fn read(&mut self) -> JsonValue {
        self.read_within(READ_TIMEOUT)
            .expect("no message from the server")
    }

    /// The next id of a request.
    pub fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// The response to the request `id`, keeping the notifications that come before it (any
    /// other response is a failure).
    pub fn response_to(&mut self, id: u64) -> JsonValue {
        loop {
            let m = self.read();
            if m.get("id").is_none() && m.get("method").is_some() {
                self.notifications.push(m);
                continue;
            }
            assert_eq!(m.get("id"), Some(&JsonValue::Number(id as f64)), "{m:?}");
            return m;
        }
    }

    /// A request; returns the whole response.
    pub fn request(&mut self, method: &str, params: &str) -> JsonValue {
        let id = self.next_id();
        self.send(&format!(
            r#"{{"jsonrpc":"2.0","id":{id},"method":"{method}","params":{params}}}"#
        ));
        self.response_to(id)
    }

    /// The notifications of `method` read so far, taken out of [`Self::notifications`].
    pub fn take_notifications(&mut self, method: &str) -> Vec<JsonValue> {
        let (taken, kept) = std::mem::take(&mut self.notifications)
            .into_iter()
            .partition(|n| n.get("method").and_then(JsonValue::as_str) == Some(method));
        self.notifications = kept;
        taken
    }

    /// Read notifications until one of `method` comes (within `timeout`).
    pub fn wait_notification(&mut self, method: &str, timeout: Duration) -> JsonValue {
        let begin = Instant::now();
        loop {
            if let Some(k) = self
                .notifications
                .iter()
                .position(|n| n.get("method").and_then(JsonValue::as_str) == Some(method))
            {
                return self.notifications.remove(k);
            }
            let left = timeout.saturating_sub(begin.elapsed());
            assert!(!left.is_zero(), "no {method} within {timeout:?}");
            if let Some(m) = self.read_within(left) {
                assert!(m.get("id").is_none(), "a response came unasked: {m:?}");
                self.notifications.push(m);
            }
        }
    }

    /// The result of a request.
    pub fn result(&mut self, method: &str, params: &str) -> JsonValue {
        let r = self.request(method, params);
        r.get("result")
            .cloned()
            .unwrap_or_else(|| panic!("{method}: {r:?}"))
    }

    /// A tool call: (structured content, is error, text).
    pub fn tool(&mut self, name: &str, arguments: &str) -> (JsonValue, bool, String) {
        let r = self.result(
            "tools/call",
            &format!(r#"{{"name":"{name}","arguments":{arguments}}}"#),
        );
        let is_error = r.get("isError").and_then(JsonValue::as_bool).unwrap();
        let text = r
            .get("content")
            .and_then(JsonValue::as_array)
            .and_then(|c| c.first())
            .and_then(|c| c.get("text"))
            .and_then(JsonValue::as_str)
            .unwrap()
            .to_string();
        let structured = r
            .get("structuredContent")
            .cloned()
            .unwrap_or(JsonValue::Null);
        (structured, is_error, text)
    }

    /// A tool call that must succeed.
    pub fn ok_tool(&mut self, name: &str, arguments: &str) -> JsonValue {
        let (s, is_error, text) = self.tool(name, arguments);
        assert!(!is_error, "{name}: {text}");
        s
    }

    /// A tool call that must fail; returns its message.
    pub fn refused_tool(&mut self, name: &str, arguments: &str) -> String {
        let (_, is_error, text) = self.tool(name, arguments);
        assert!(is_error, "{name} {arguments}: accepted: {text}");
        text
    }

    /// Poll `job_status` until the job does not run.
    pub fn wait(&mut self, job: &str, timeout: Duration) -> JsonValue {
        let begin = Instant::now();
        loop {
            let s = self.ok_tool("job_status", &format!(r#"{{"job_id":"{job}"}}"#));
            if s.get("state").and_then(JsonValue::as_str) != Some("running") {
                return s;
            }
            assert!(begin.elapsed() < timeout, "job {job} still runs: {s:?}");
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// The handshake of `2025-11-25`.
    pub fn initialize(&mut self) {
        let r = self.result(
            "initialize",
            r#"{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1"}}"#,
        );
        assert_eq!(get(&r, &["protocolVersion"]).as_str(), Some("2025-11-25"));
        assert_eq!(get(&r, &["serverInfo", "name"]).as_str(), Some("mopt-rs"));
        assert!(get(&r, &["capabilities", "tools"]).as_object().is_some());
        self.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
    }

    /// Close the input and wait for the server, which must end cleanly.
    pub fn close(mut self) {
        drop(self.stdin);
        let status = self.child.wait().unwrap();
        assert!(status.success(), "the server ended with {status}");
    }
}

/// A fresh directory with an empty `root/` in it.
pub fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mcp_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("root")).unwrap();
    dir
}

/// The value at `path` in `v`.
pub fn get<'a>(v: &'a JsonValue, path: &[&str]) -> &'a JsonValue {
    path.iter().fold(v, |v, k| {
        v.get(k).unwrap_or_else(|| panic!("no {k} in {v:?}"))
    })
}
