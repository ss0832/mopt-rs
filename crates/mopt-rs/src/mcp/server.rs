//! The server loop: one JSON-RPC message per line on standard input, one message per line on
//! standard output, and the routing of the methods.
//!
//! A request with `_meta["io.modelcontextprotocol/protocolVersion"]` is answered in that
//! revision (`2026-07-28`: with `resultType`, the cache fields, and the server identity in
//! `_meta`); `initialize` selects a revision with the handshake for the rest of the process.
//!
//! A thread reads the input, so that the loop can answer some requests later and send
//! notifications meanwhile:
//!
//! - `job_wait` and `tasks/result` (`2025-11-25`) are answered when their job ends (or the
//!   wait is up); a request that carries `_meta.progressToken` gets `notifications/progress`
//!   at every new iteration of the job, and `notifications/cancelled` with its id ends it
//!   without an answer;
//! - subscriptions: `subscriptions/listen` (`2026-07-28`; acknowledged at once, then
//!   `notifications/resources/updated` and `notifications/tasks` carrying its id, until
//!   `notifications/cancelled` ends it) and `resources/subscribe` / `resources/unsubscribe`
//!   with `notifications/resources/updated` (the revisions with the handshake);
//! - the status of tasks made by task-augmented calls of `2025-11-25`
//!   (`notifications/tasks/status`).
//!
//! What is watched is looked at every [`TICK`]; the loop sleeps on its input when nothing is.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::jobs::{JobManager, JobProgram, JobState};
use super::pagination::{Pager, DEFAULT_PAGE_SIZE};
use super::paths::Sandbox;
use super::policy::Policy;
use super::protocol::{
    self, error_response, get_str, negotiate_legacy, object, parse_message, result_response, text,
    Era, Incoming, RpcError, LEGACY_RESOURCE_NOT_FOUND, LEGACY_VERSIONS, META_CLIENT_CAPABILITIES,
    META_PROTOCOL_VERSION, META_SERVER_INFO, METHOD_NOT_FOUND,
};
use super::resources::{self, Docs, EmbeddedDocs, ResourceError, JOBS_PREFIX};
use super::runner::RUN_JOB;
use super::schema::{self, CommandTable, Listing, MAX_WAIT_SECONDS};
use super::tasks::{self, TaskStatus, LEGACY_TASKS_VERSION, META_RELATED_TASK, TASKS_EXTENSION};
use super::tools::{self, call_result, ToolContext};
use super::{prompts, SERVER_NAME};
use crate::cli::spec::COMMANDS;
use crate::core_infra::json::JsonValue;
use crate::pes::registry::CalculatorRegistry;

/// How long clients may keep the lists of tools and prompts (they do not change while a
/// server runs).
const TTL_STATIC_MS: f64 = 3_600_000.0;
/// How often waits, subscriptions and tasks are looked at.
pub const TICK: Duration = Duration::from_millis(250);
/// `_meta` key of the subscription a notification belongs to (`2026-07-28`).
pub const META_SUBSCRIPTION_ID: &str = "io.modelcontextprotocol/subscriptionId";
/// The most `subscriptions/listen` streams open at a time.
pub const MAX_LISTENS: usize = 16;
/// The most resources watched at a time (all subscriptions together).
pub const MAX_WATCHED: usize = 256;
/// The most requests waiting for jobs at a time.
pub const MAX_PENDING: usize = 64;

const INSTRUCTIONS: &str = "mopt-rs: geometry optimization, reaction paths, molecular dynamics, \
conformer and orientation searches, scans, IRC, vibrational analysis, AutoTS, reaction networks. \
Each implemented command (optmain, nebmain, run_mapper, ...) is a tool that starts a job under \
mcp_jobs/ of the server root and returns at once with a job_id (or a task, for clients with \
tasks): follow it with job_status or job_wait, read its run directory with job_result and \
resources/read, stop it with job_stop (a STOP file), end it with job_cancel, and continue it \
with job_restart (--restart). The arguments of a command tool are its command-line options by \
their long names, as in settings files (command_help lists them with their schemas); INPUT \
names the input files, with paths relative to the root. list_calculators names the \
calculators; single_point, model_hessian, vibrations, coordinates, interpolate, superpose, \
bond_graph, dispersion, hessian_update, units and elements answer at once.";

/// What a request is answered with.
enum Answer {
    /// An ordinary result.
    Plain(JsonValue),
    /// A result that clients may cache for this many milliseconds.
    Cacheable(JsonValue, f64),
    /// Nothing now: the answer comes later (a wait), or never (a subscription stream).
    Later,
}

/// What a request answered later waits for.
#[derive(Debug, Clone)]
enum Awaited {
    /// The end of a job (`job_wait`), with the lines of the log to return.
    Job { tail: usize },
    /// The end of a task (`tasks/result` of `2025-11-25`).
    TaskResult,
}

/// A request answered later.
#[derive(Debug, Clone)]
struct Pending {
    id: JsonValue,
    era: Era,
    job: String,
    awaited: Awaited,
    started: Instant,
    deadline: Option<Instant>,
    progress: Option<Progress>,
}

/// The progress notifications of a request.
#[derive(Debug, Clone)]
struct Progress {
    token: JsonValue,
    count: f64,
    seen: (u64, Option<String>),
}

/// A resource watched for a subscription, with what it was when last looked at.
#[derive(Debug, Clone)]
struct Watched {
    uri: String,
    fingerprint: String,
}

/// A stream of `subscriptions/listen`.
#[derive(Debug, Clone)]
struct Listen {
    id: JsonValue,
    resources: Vec<Watched>,
    tasks: Vec<(String, String)>,
}

/// The server: the command table, the jobs, the calculators, the documentation, and what is
/// waited for and watched.
#[derive(Debug)]
pub struct Server {
    commands: CommandTable<'static>,
    jobs: JobManager,
    registry: CalculatorRegistry,
    docs: Docs,
    legacy: Option<String>,
    listing: Listing,
    pager: Pager,
    pending: Vec<Pending>,
    listens: Vec<Listen>,
    subscribed: Vec<Watched>,
    legacy_tasks: Vec<(String, String)>,
    last_tick: Instant,
}

/// How a server is started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerOptions {
    /// Whether settings that start external programs or load code are allowed
    /// (`--allow-external`).
    pub allow_external: bool,
    /// How many jobs of the server may run at a time (`--max-jobs`).
    pub max_jobs: usize,
    /// How the tools of the commands are listed (`--full-schemas`: [`Listing::Full`]).
    pub listing: Listing,
    /// Entries per page of the lists (`--page-size`).
    pub page_size: usize,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            allow_external: false,
            max_jobs: 4,
            listing: Listing::Compact,
            page_size: DEFAULT_PAGE_SIZE,
        }
    }
}

/// One line of the input: its text, or `None` when it was longer than
/// [`protocol::MAX_MESSAGE_BYTES`] (the rest of it is read and dropped).
fn read_line<R: BufRead>(
    input: &mut R,
    buffer: &mut Vec<u8>,
) -> std::io::Result<Option<Option<String>>> {
    buffer.clear();
    let mut too_long = false;
    let mut any = false;
    loop {
        let available = input.fill_buf()?;
        if available.is_empty() {
            return Ok(if any {
                Some((!too_long).then(|| String::from_utf8_lossy(&buffer[..]).into_owned()))
            } else {
                None
            });
        }
        any = true;
        let (take, done) = match available.iter().position(|b| *b == b'\n') {
            Some(at) => (at + 1, true),
            None => (available.len(), false),
        };
        if !too_long {
            if buffer.len() + take > protocol::MAX_MESSAGE_BYTES {
                too_long = true;
                buffer.clear();
            } else {
                buffer.extend_from_slice(&available[..take]);
            }
        }
        input.consume(take);
        if done {
            return Ok(Some(
                (!too_long).then(|| String::from_utf8_lossy(&buffer[..]).into_owned()),
            ));
        }
    }
}

/// What the reading thread hands to the loop.
enum Input {
    /// A line (`None`: too long).
    Line(Option<String>),
    /// The input ended.
    End,
    /// The input failed.
    Failed(std::io::Error),
}

fn notification(method: &str, params: JsonValue) -> JsonValue {
    object([
        ("jsonrpc", text("2.0")),
        ("method", text(method)),
        ("params", params),
    ])
}

/// `value` (an object) with `key` set to `v`.
fn with_key(mut value: JsonValue, key: &str, v: JsonValue) -> JsonValue {
    if let Some(map) = value.as_object_mut() {
        map.insert(key.to_string(), v);
    }
    value
}

impl Server {
    /// The server of the directory `root`, whose jobs run this program (`mopt mcp run-job`),
    /// with the documentation `docs` (the pages embedded in the binary; empty: those of the
    /// repository that holds the root).
    pub fn new(root: &Path, options: &ServerOptions, docs: EmbeddedDocs) -> Result<Self, String> {
        let executable =
            std::env::current_exe().map_err(|e| format!("this program cannot be found: {e}"))?;
        let program = JobProgram {
            executable,
            prefix: vec!["mcp".to_string(), RUN_JOB.to_string()],
        };
        Self::with(
            root,
            COMMANDS,
            program,
            CalculatorRegistry::native(),
            options,
            docs,
        )
    }

    /// A server of `root` for the commands of `commands`, whose jobs run `program`, with the
    /// calculators of `registry` and the documentation `docs`.
    pub fn with(
        root: &Path,
        commands: CommandTable<'static>,
        program: JobProgram,
        registry: CalculatorRegistry,
        options: &ServerOptions,
        docs: EmbeddedDocs,
    ) -> Result<Self, String> {
        crate::core_infra::parallelism::deterministic_linear_algebra();
        let sandbox = Sandbox::new(root)?;
        let docs = Docs::of(sandbox.root(), docs);
        eprintln!(
            "mopt mcp: root {}; documentation {}; external programs and code {}; at most {} \
             jobs; {} tool schemas",
            sandbox.root().display(),
            docs.describe(),
            if options.allow_external {
                "allowed"
            } else {
                "refused"
            },
            options.max_jobs,
            match options.listing {
                Listing::Compact => "compact",
                Listing::Full => "full",
            }
        );
        let policy = Policy::new(options.allow_external);
        Ok(Self {
            commands,
            jobs: JobManager::new(sandbox, program, policy, options.max_jobs.max(1)),
            registry,
            docs,
            legacy: None,
            listing: options.listing,
            pager: Pager::new(options.page_size),
            pending: Vec::new(),
            listens: Vec::new(),
            subscribed: Vec::new(),
            legacy_tasks: Vec::new(),
            last_tick: Instant::now(),
        })
    }

    /// Whether anything is waited for or watched (the loop then looks every [`TICK`]).
    fn watching(&self) -> bool {
        !self.pending.is_empty()
            || !self.listens.is_empty()
            || !self.subscribed.is_empty()
            || !self.legacy_tasks.is_empty()
    }

    /// Answer the lines of `input` on `output` until the input ends.
    pub fn serve<R: BufRead + Send + 'static, W: Write>(
        &mut self,
        input: R,
        mut output: W,
    ) -> std::io::Result<()> {
        let (sender, receiver) = mpsc::channel::<Input>();
        std::thread::spawn(move || {
            let mut input = input;
            let mut buffer = Vec::new();
            loop {
                let next = match read_line(&mut input, &mut buffer) {
                    Ok(Some(line)) => Input::Line(line),
                    Ok(None) => Input::End,
                    Err(e) => Input::Failed(e),
                };
                let last = !matches!(next, Input::Line(_));
                if sender.send(next).is_err() || last {
                    return;
                }
            }
        });
        let mut write = |messages: Vec<JsonValue>| -> std::io::Result<()> {
            for m in messages {
                writeln!(output, "{}", m.to_string_compact())?;
            }
            output.flush()
        };
        loop {
            let next = if self.watching() {
                match receiver.recv_timeout(TICK) {
                    Ok(next) => Some(next),
                    Err(mpsc::RecvTimeoutError::Timeout) => None,
                    Err(mpsc::RecvTimeoutError::Disconnected) => Some(Input::End),
                }
            } else {
                Some(receiver.recv().unwrap_or(Input::End))
            };
            match next {
                Some(Input::Line(None)) => write(vec![error_response(
                    None,
                    &RpcError::new(
                        protocol::INVALID_REQUEST,
                        format!(
                            "Invalid request: a message is at most {} bytes",
                            protocol::MAX_MESSAGE_BYTES
                        ),
                    ),
                )])?,
                Some(Input::Line(Some(line))) if line.trim().is_empty() => {}
                Some(Input::Line(Some(line))) => write(self.handle_line(line.trim()))?,
                Some(Input::End) => return Ok(()),
                Some(Input::Failed(e)) => return Err(e),
                None => {}
            }
            if self.watching() && self.last_tick.elapsed() >= TICK {
                self.last_tick = Instant::now();
                write(self.tick())?;
            }
        }
    }

    /// The messages to send for one line: the response (none for notifications and for
    /// requests answered later) and notifications. A failure inside the server is an internal
    /// error of that request, never the end of the server.
    pub fn handle_line(&mut self, line: &str) -> Vec<JsonValue> {
        match parse_message(line) {
            Err(response) => vec![response],
            Ok(Incoming::Response) => Vec::new(),
            Ok(Incoming::Notification(method, params)) => {
                self.notified(&method, &params);
                Vec::new()
            }
            Ok(Incoming::Request(id, method, params)) => {
                let mut out = Vec::new();
                let answered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.request(&id, &method, &params, &mut out)
                }))
                .unwrap_or_else(|_| {
                    Err(RpcError::new(
                        protocol::INTERNAL_ERROR,
                        format!("Internal error while answering {method}"),
                    ))
                });
                match answered {
                    Ok(Some(result)) => out.insert(0, result_response(&id, result)),
                    Ok(None) => {}
                    Err(e) => out.insert(0, error_response(Some(&id), &e)),
                }
                out
            }
        }
    }

    fn notified(&mut self, method: &str, params: &JsonValue) {
        match method {
            "notifications/initialized" if self.legacy.is_none() => {
                eprintln!("mopt mcp: notifications/initialized before initialize; ignored");
            }
            "notifications/cancelled" => {
                let Some(id) = params.get("requestId") else {
                    return;
                };
                let before = self.pending.len() + self.listens.len();
                self.pending.retain(|p| p.id != *id);
                self.listens.retain(|l| l.id != *id);
                if self.pending.len() + self.listens.len() < before {
                    eprintln!(
                        "mopt mcp: request {} cancelled by the client",
                        id.to_string_compact()
                    );
                }
            }
            _ => {}
        }
    }

    fn era(&self, method: &str, params: &JsonValue) -> Result<Era, RpcError> {
        let meta = params.get("_meta");
        if let Some(version) = meta.and_then(|m| m.get(META_PROTOCOL_VERSION)) {
            let version = version.as_str().ok_or_else(|| {
                RpcError::invalid_params(format!("{META_PROTOCOL_VERSION} is a string"))
            })?;
            let era =
                Era::of_version(version).ok_or_else(|| RpcError::unsupported_version(version))?;
            if era == Era::Modern && meta.and_then(|m| m.get(META_CLIENT_CAPABILITIES)).is_none() {
                return Err(RpcError::invalid_params(format!(
                    "the request has no _meta {META_CLIENT_CAPABILITIES}"
                )));
            }
            return Ok(era);
        }
        if let Some(v) = &self.legacy {
            return Ok(Era::Legacy(v.clone()));
        }
        if method == "ping" {
            return Ok(Era::Legacy(LEGACY_VERSIONS[0].to_string()));
        }
        Err(RpcError::invalid_params(format!(
            "the request has no _meta {META_PROTOCOL_VERSION} (protocol {}), and no initialize \
             came before (protocols {})",
            protocol::MODERN_VERSION,
            LEGACY_VERSIONS.join(", ")
        ))
        .with_data(object([(
            "supported",
            JsonValue::Array(
                protocol::supported_versions()
                    .into_iter()
                    .map(text)
                    .collect(),
            ),
        )])))
    }

    fn capabilities(era: &Era) -> JsonValue {
        let mut caps = std::collections::BTreeMap::new();
        caps.insert(
            "tools".to_string(),
            object([("listChanged", JsonValue::Bool(false))]),
        );
        caps.insert(
            "resources".to_string(),
            object([
                ("listChanged", JsonValue::Bool(false)),
                ("subscribe", JsonValue::Bool(true)),
            ]),
        );
        caps.insert(
            "prompts".to_string(),
            object([("listChanged", JsonValue::Bool(false))]),
        );
        match era {
            Era::Modern => {
                caps.insert(
                    "extensions".to_string(),
                    object([(TASKS_EXTENSION, JsonValue::empty_object())]),
                );
            }
            Era::Legacy(v) if v == LEGACY_TASKS_VERSION => {
                caps.insert(
                    "tasks".to_string(),
                    object([
                        ("list", JsonValue::empty_object()),
                        ("cancel", JsonValue::empty_object()),
                        (
                            "requests",
                            object([("tools", object([("call", JsonValue::empty_object())]))]),
                        ),
                    ]),
                );
            }
            Era::Legacy(_) => {}
        }
        JsonValue::Object(caps)
    }

    fn server_info() -> JsonValue {
        object([
            ("name", text(SERVER_NAME)),
            ("title", text("mopt-rs")),
            ("version", text(crate::VERSION)),
        ])
    }

    /// A result in the form of `era`.
    fn wrap(era: &Era, answer: Answer) -> Option<JsonValue> {
        let (mut v, ttl) = match answer {
            Answer::Plain(v) => (v, None),
            Answer::Cacheable(v, ttl) => (v, Some(ttl)),
            Answer::Later => return None,
        };
        if *era == Era::Modern {
            if let Some(map) = v.as_object_mut() {
                map.entry("resultType".to_string())
                    .or_insert_with(|| text("complete"));
                map.insert(
                    "_meta".to_string(),
                    object([(META_SERVER_INFO, Self::server_info())]),
                );
                if let Some(ttl) = ttl {
                    map.insert("ttlMs".to_string(), JsonValue::Number(ttl));
                    map.insert("cacheScope".to_string(), text("private"));
                }
            }
        }
        Some(v)
    }

    fn request(
        &mut self,
        id: &JsonValue,
        method: &str,
        params: &JsonValue,
        out: &mut Vec<JsonValue>,
    ) -> Result<Option<JsonValue>, RpcError> {
        if method == "initialize"
            && params
                .get("_meta")
                .and_then(|m| m.get(META_PROTOCOL_VERSION))
                .is_none()
        {
            let version = negotiate_legacy(get_str(params, "protocolVersion"));
            self.legacy = Some(version.to_string());
            let era = Era::Legacy(version.to_string());
            return Ok(Some(object([
                ("protocolVersion", text(version)),
                ("capabilities", Self::capabilities(&era)),
                ("serverInfo", Self::server_info()),
                ("instructions", text(INSTRUCTIONS)),
            ])));
        }
        let era = self.era(method, params)?;
        let answer = self.answer(id, method, params, &era, out)?;
        Ok(Self::wrap(&era, answer))
    }

    /// A page of `items` of the list `list` (the `cursor` of `params`), under `key`.
    fn page(
        &self,
        list: &str,
        key: &str,
        items: Vec<JsonValue>,
        params: &JsonValue,
    ) -> Result<JsonValue, RpcError> {
        let (page, next) = self.pager.page(list, items, params.get("cursor"))?;
        let mut map = std::collections::BTreeMap::new();
        map.insert(key.to_string(), JsonValue::Array(page));
        if let Some(next) = next {
            map.insert("nextCursor".to_string(), text(next));
        }
        Ok(JsonValue::Object(map))
    }

    /// The tools in the form of `era` (`execution` for the experimental tasks of
    /// `2025-11-25`).
    fn tools_of(&self, era: &Era) -> Vec<JsonValue> {
        let mut tools = schema::all_tools(self.commands, self.listing);
        if *era == Era::Legacy(LEGACY_TASKS_VERSION.to_string()) {
            for t in &mut tools {
                let name = t
                    .get("name")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default();
                if self.is_command_tool(name) {
                    *t = with_key(
                        t.clone(),
                        "execution",
                        object([("taskSupport", text("optional"))]),
                    );
                }
            }
        }
        tools
    }

    fn is_command_tool(&self, name: &str) -> bool {
        self.commands
            .iter()
            .any(|(c, implemented, _, _)| *c == name && *implemented)
    }

    fn task_id(params: &JsonValue) -> Result<&str, RpcError> {
        get_str(params, "taskId").ok_or_else(|| RpcError::invalid_params("give the taskId"))
    }

    /// The job view of the task `id`, or `-32602` when there is no such task.
    fn task_view(&mut self, id: &str) -> Result<super::jobs::JobView, RpcError> {
        self.jobs
            .view(id)
            .map_err(|e| RpcError::invalid_params(format!("no task '{}': {e}", short(id))))
    }

    /// The `DetailedTask` of `2026-07-28` for the job `id`.
    fn detailed_task(&mut self, id: &str) -> Result<JsonValue, RpcError> {
        let view = self.task_view(id)?;
        let mut task = tasks::modern_task(id, &view);
        match TaskStatus::of(&view) {
            TaskStatus::Completed => {
                task = with_key(task, "result", call_result(self.jobs.result(id)));
            }
            TaskStatus::Failed => {
                task = with_key(task, "error", tasks::failure(id, &view));
            }
            _ => {}
        }
        Ok(task)
    }

    /// The `CallToolResult` of a terminal task `id` (`tasks/result` of `2025-11-25`).
    fn task_result(&mut self, id: &str) -> Result<JsonValue, RpcError> {
        let view = self.task_view(id)?;
        let status = TaskStatus::of(&view);
        let mut result = call_result(self.jobs.result(id));
        if status != TaskStatus::Completed {
            result = with_key(result, "isError", JsonValue::Bool(true));
            if let Some(JsonValue::Array(content)) =
                result.as_object_mut().and_then(|m| m.get_mut("content"))
            {
                content.insert(
                    0,
                    object([
                        ("type", text("text")),
                        ("text", text(tasks::status_message(&view))),
                    ]),
                );
            }
        }
        Ok(with_key(
            result,
            "_meta",
            object([(META_RELATED_TASK, object([("taskId", text(id))]))]),
        ))
    }

    /// Start the command tool `name` as a task of `era`.
    fn start_task(&mut self, name: &str, args: &JsonValue, era: &Era) -> Answer {
        let Some((command, _, _, options)) = self
            .commands
            .iter()
            .find(|(c, implemented, _, _)| *c == name && *implemented)
        else {
            return Answer::Plain(call_result(Err(format!("Unknown tool: {name}"))));
        };
        let started = self.jobs.start(command, options, args);
        let id = match started
            .as_ref()
            .ok()
            .and_then(|s| get_str(s, "job_id"))
            .map(String::from)
        {
            Some(id) => id,
            // A request the job could not start from is a tool result with isError.
            None => return Answer::Plain(call_result(started)),
        };
        let view = match self.jobs.view(&id) {
            Ok(v) => v,
            Err(e) => return Answer::Plain(call_result(Err(e))),
        };
        if *era == Era::Modern {
            Answer::Plain(with_key(
                tasks::modern_task(&id, &view),
                "resultType",
                text("task"),
            ))
        } else {
            self.legacy_tasks
                .push((id.clone(), TaskStatus::of(&view).name().to_string()));
            Answer::Plain(object([("task", tasks::legacy_task(&id, &view))]))
        }
    }

    fn progress_of(params: &JsonValue) -> Option<Progress> {
        let token = params.get("_meta").and_then(|m| m.get("progressToken"))?;
        let valid = match token {
            JsonValue::String(_) => true,
            JsonValue::Number(n) => n.fract() == 0.0,
            _ => false,
        };
        valid.then(|| Progress {
            token: token.clone(),
            count: 0.0,
            seen: (0, None),
        })
    }

    /// `job_wait`: the status now when the job does not run (or the wait is 0), else later.
    fn job_wait(
        &mut self,
        id: &JsonValue,
        args: &JsonValue,
        params: &JsonValue,
        era: &Era,
    ) -> Answer {
        let Some(job) = args.get("job_id").and_then(JsonValue::as_str) else {
            return Answer::Plain(call_result(Err("give the job_id".to_string())));
        };
        let timeout = match args.get("timeout_seconds") {
            None | Some(JsonValue::Null) => 60.0,
            Some(JsonValue::Number(n)) if n.is_finite() && *n >= 0.0 => n.min(MAX_WAIT_SECONDS),
            Some(_) => {
                return Answer::Plain(call_result(Err(format!(
                    "timeout_seconds is a number from 0 to {MAX_WAIT_SECONDS}"
                ))))
            }
        };
        let tail = args
            .get("tail_lines")
            .and_then(JsonValue::as_usize)
            .unwrap_or(40)
            .min(500);
        let view = match self.jobs.view(job) {
            Ok(v) => v,
            Err(e) => return Answer::Plain(call_result(Err(e))),
        };
        if view.state != JobState::Running || timeout == 0.0 {
            return Answer::Plain(self.wait_result(job, tail, 0.0));
        }
        if self.pending.len() >= MAX_PENDING {
            return Answer::Plain(call_result(Err(format!(
                "{MAX_PENDING} requests wait already; use job_status"
            ))));
        }
        let mut progress = Self::progress_of(params);
        if let Some(p) = &mut progress {
            p.seen = (view.log_bytes, view.last_iteration);
        }
        self.pending.push(Pending {
            id: id.clone(),
            era: era.clone(),
            job: job.to_string(),
            awaited: Awaited::Job { tail },
            started: Instant::now(),
            deadline: Some(Instant::now() + Duration::from_secs_f64(timeout)),
            progress,
        });
        Answer::Later
    }

    fn wait_result(&mut self, job: &str, tail: usize, waited: f64) -> JsonValue {
        let status = self.jobs.status(job, tail).map(|mut s| {
            let running = get_str(&s, "state") == Some("running");
            s = with_key(
                s,
                "waited_seconds",
                JsonValue::Number((waited * 10.0).round() / 10.0),
            );
            with_key(s, "timed_out", JsonValue::Bool(running))
        });
        call_result(status)
    }

    /// Watch the resource `uri` (checked first; job URIs name a job of the root).
    fn watched(&mut self, uri: &str) -> Result<Watched, String> {
        if let Some(job) = uri.strip_prefix(JOBS_PREFIX) {
            self.jobs.job_dir(job)?;
        } else {
            resources::check_watch(self.jobs.sandbox(), &self.docs, uri)?;
        }
        let fingerprint = self.fingerprint(uri);
        Ok(Watched {
            uri: uri.to_string(),
            fingerprint,
        })
    }

    fn watched_count(&self) -> usize {
        self.subscribed.len()
            + self
                .listens
                .iter()
                .map(|l| l.resources.len() + l.tasks.len())
                .sum::<usize>()
    }

    fn fingerprint(&mut self, uri: &str) -> String {
        match uri.strip_prefix(JOBS_PREFIX) {
            Some(job) => self.jobs.view(job).map_or_else(
                |_| "gone".to_string(),
                |v| format!("{}:{}:{}", v.state.name(), v.log_bytes, v.cancel_requested),
            ),
            None => resources::fingerprint(self.jobs.sandbox(), uri),
        }
    }

    /// `subscriptions/listen`: the acknowledgement now (in `out`), the notifications later.
    fn listen(
        &mut self,
        id: &JsonValue,
        params: &JsonValue,
        out: &mut Vec<JsonValue>,
    ) -> Result<Answer, RpcError> {
        if self.listens.len() >= MAX_LISTENS {
            return Err(RpcError::invalid_params(format!(
                "at most {MAX_LISTENS} subscriptions/listen streams are open at a time"
            )));
        }
        let wanted = params
            .get("notifications")
            .cloned()
            .unwrap_or_else(JsonValue::empty_object);
        let mut listen = Listen {
            id: id.clone(),
            resources: Vec::new(),
            tasks: Vec::new(),
        };
        let mut refused = Vec::new();
        for uri in wanted
            .get("resourceSubscriptions")
            .and_then(JsonValue::as_string_vec)
            .unwrap_or_default()
        {
            if self.watched_count() + listen.resources.len() + listen.tasks.len() >= MAX_WATCHED {
                refused.push(format!(
                    "{uri}: at most {MAX_WATCHED} resources are watched"
                ));
                continue;
            }
            match self.watched(&uri) {
                Ok(w) => listen.resources.push(w),
                Err(e) => refused.push(format!("{}: {e}", short(&uri))),
            }
        }
        for task in wanted
            .get("taskIds")
            .and_then(JsonValue::as_string_vec)
            .unwrap_or_default()
        {
            match self.jobs.view(&task) {
                Ok(v) => listen
                    .tasks
                    .push((task, TaskStatus::of(&v).name().to_string())),
                Err(e) => refused.push(format!("{}: {e}", short(&task))),
            }
        }
        for r in &refused {
            eprintln!("mopt mcp: subscription refused: {r}");
        }
        let mut agreed = std::collections::BTreeMap::new();
        if !listen.resources.is_empty() {
            agreed.insert(
                "resourceSubscriptions".to_string(),
                JsonValue::Array(listen.resources.iter().map(|w| text(&w.uri)).collect()),
            );
        }
        if !listen.tasks.is_empty() {
            agreed.insert(
                "taskIds".to_string(),
                JsonValue::Array(listen.tasks.iter().map(|(t, _)| text(t)).collect()),
            );
        }
        out.push(notification(
            "notifications/subscriptions/acknowledged",
            object([
                ("_meta", object([(META_SUBSCRIPTION_ID, id.clone())])),
                ("notifications", JsonValue::Object(agreed)),
            ]),
        ));
        self.listens.push(listen);
        Ok(Answer::Later)
    }

    fn answer(
        &mut self,
        id: &JsonValue,
        method: &str,
        params: &JsonValue,
        era: &Era,
        out: &mut Vec<JsonValue>,
    ) -> Result<Answer, RpcError> {
        let modern = *era == Era::Modern;
        let legacy_tasks = *era == Era::Legacy(LEGACY_TASKS_VERSION.to_string());
        match method {
            "server/discover" => Ok(Answer::Cacheable(
                object([
                    (
                        "supportedVersions",
                        JsonValue::Array(
                            protocol::supported_versions()
                                .into_iter()
                                .map(text)
                                .collect(),
                        ),
                    ),
                    ("capabilities", Self::capabilities(&Era::Modern)),
                    ("instructions", text(INSTRUCTIONS)),
                ]),
                TTL_STATIC_MS,
            )),
            "ping" if !modern => Ok(Answer::Plain(JsonValue::empty_object())),
            "tools/list" => {
                let tools = self.tools_of(era);
                Ok(Answer::Cacheable(
                    self.page("tools/list", "tools", tools, params)?,
                    TTL_STATIC_MS,
                ))
            }
            "tools/call" => {
                let name = get_str(params, "name").ok_or_else(|| {
                    RpcError::invalid_params("tools/call needs the name of the tool")
                })?;
                let args = match params.get("arguments") {
                    None | Some(JsonValue::Null) => JsonValue::empty_object(),
                    Some(a @ JsonValue::Object(_)) => a.clone(),
                    Some(_) => {
                        return Err(RpcError::invalid_params(
                            "the arguments of a tool are an object",
                        ))
                    }
                };
                let as_task = if modern {
                    tasks::client_has_tasks(
                        params
                            .get("_meta")
                            .and_then(|m| m.get(META_CLIENT_CAPABILITIES)),
                    ) && self.is_command_tool(name)
                } else {
                    match params.get("task") {
                        None | Some(JsonValue::Null) => false,
                        Some(JsonValue::Object(_)) if legacy_tasks => {
                            if !self.is_command_tool(name) {
                                return Err(RpcError::invalid_params(format!(
                                    "the tool {name} does not run as a task (the tools of the \
                                     commands do)"
                                )));
                            }
                            true
                        }
                        Some(_) => {
                            return Err(RpcError::invalid_params(
                                "task is an object, in protocol 2025-11-25",
                            ))
                        }
                    }
                };
                if as_task {
                    return Ok(self.start_task(name, &args, era));
                }
                if name == "job_wait" {
                    return Ok(self.job_wait(id, &args, params, era));
                }
                let mut ctx = ToolContext {
                    commands: self.commands,
                    jobs: &mut self.jobs,
                    registry: &self.registry,
                };
                tools::call(&mut ctx, name, &args).map(Answer::Plain)
            }
            "tasks/get" => {
                let task = Self::task_id(params)?.to_string();
                if modern {
                    Ok(Answer::Plain(self.detailed_task(&task)?))
                } else if legacy_tasks {
                    let view = self.task_view(&task)?;
                    Ok(Answer::Plain(tasks::legacy_task(&task, &view)))
                } else {
                    Err(Self::no_method(method))
                }
            }
            "tasks/cancel" => {
                if !(modern || legacy_tasks) {
                    return Err(Self::no_method(method));
                }
                let task = Self::task_id(params)?.to_string();
                let view = self.task_view(&task)?;
                let status = TaskStatus::of(&view);
                if status.terminal() {
                    return Err(RpcError::invalid_params(format!(
                        "task {task} is {} already",
                        status.name()
                    )));
                }
                self.jobs
                    .request_cancel(&task)
                    .map_err(RpcError::invalid_params)?;
                if modern {
                    Ok(Answer::Plain(JsonValue::empty_object()))
                } else {
                    let view = self.task_view(&task)?;
                    Ok(Answer::Plain(tasks::legacy_task(&task, &view)))
                }
            }
            "tasks/update" if modern => {
                let task = Self::task_id(params)?.to_string();
                self.task_view(&task)?;
                Err(RpcError::invalid_params(format!(
                    "task {task} asks for no input (the jobs of mopt-rs run without questions)"
                )))
            }
            "tasks/list" if legacy_tasks => {
                let mut list = Vec::new();
                for job in self.jobs.ids() {
                    if let Ok(view) = self.jobs.view(&job) {
                        list.push(tasks::legacy_task(&job, &view));
                    }
                }
                Ok(Answer::Plain(self.page(
                    "tasks/list",
                    "tasks",
                    list,
                    params,
                )?))
            }
            "tasks/result" if legacy_tasks => {
                let task = Self::task_id(params)?.to_string();
                let view = self.task_view(&task)?;
                if TaskStatus::of(&view).terminal() {
                    return Ok(Answer::Plain(self.task_result(&task)?));
                }
                if self.pending.len() >= MAX_PENDING {
                    return Err(RpcError::invalid_params(format!(
                        "{MAX_PENDING} requests wait already; use tasks/get"
                    )));
                }
                self.pending.push(Pending {
                    id: id.clone(),
                    era: era.clone(),
                    job: task,
                    awaited: Awaited::TaskResult,
                    started: Instant::now(),
                    deadline: None,
                    progress: None,
                });
                Ok(Answer::Later)
            }
            "subscriptions/listen" if modern => self.listen(id, params, out),
            "resources/subscribe" | "resources/unsubscribe" if !modern => {
                let uri = get_str(params, "uri")
                    .ok_or_else(|| RpcError::invalid_params(format!("{method} needs the uri")))?
                    .to_string();
                if method == "resources/unsubscribe" {
                    self.subscribed.retain(|w| w.uri != uri);
                    return Ok(Answer::Plain(JsonValue::empty_object()));
                }
                if self.subscribed.iter().any(|w| w.uri == uri) {
                    return Ok(Answer::Plain(JsonValue::empty_object()));
                }
                if self.watched_count() >= MAX_WATCHED {
                    return Err(RpcError::invalid_params(format!(
                        "at most {MAX_WATCHED} resources are watched at a time"
                    )));
                }
                let watched = self.watched(&uri).map_err(|e| {
                    RpcError::new(
                        LEGACY_RESOURCE_NOT_FOUND,
                        format!("Resource not found: {e}"),
                    )
                    .with_data(object([("uri", text(short(&uri)))]))
                })?;
                self.subscribed.push(watched);
                Ok(Answer::Plain(JsonValue::empty_object()))
            }
            "resources/list" => {
                let mut list = resources::list(self.jobs.sandbox(), &self.docs);
                for job in self.jobs.ids() {
                    list.push(object([
                        ("uri", text(format!("{JOBS_PREFIX}{job}"))),
                        ("name", text(format!("job {job}"))),
                        ("description", text("status of a job")),
                        ("mimeType", text("application/json")),
                    ]));
                }
                Ok(Answer::Cacheable(
                    self.page("resources/list", "resources", list, params)?,
                    0.0,
                ))
            }
            "resources/templates/list" => Ok(Answer::Cacheable(
                self.page(
                    "resources/templates/list",
                    "resourceTemplates",
                    resources::templates(),
                    params,
                )?,
                TTL_STATIC_MS,
            )),
            "resources/read" => {
                let uri = get_str(params, "uri")
                    .ok_or_else(|| RpcError::invalid_params("resources/read needs the uri"))?;
                let not_found = |message: String| {
                    let code = if modern {
                        protocol::INVALID_PARAMS
                    } else {
                        LEGACY_RESOURCE_NOT_FOUND
                    };
                    RpcError::new(code, format!("Resource not found: {message}"))
                        .with_data(object([("uri", text(short(uri)))]))
                };
                if let Some(job) = uri.strip_prefix(JOBS_PREFIX) {
                    let status = self.jobs.status(job, 40).map_err(not_found)?;
                    return Ok(Answer::Cacheable(
                        object([(
                            "contents",
                            JsonValue::Array(vec![object([
                                ("uri", text(uri)),
                                ("mimeType", text("application/json")),
                                ("text", text(status.to_string_pretty())),
                            ])]),
                        )]),
                        0.0,
                    ));
                }
                match resources::read(self.jobs.sandbox(), &self.docs, uri) {
                    Ok(contents) => Ok(Answer::Cacheable(
                        object([("contents", JsonValue::Array(contents))]),
                        0.0,
                    )),
                    Err(ResourceError::NotFound(message)) => Err(not_found(message)),
                    Err(ResourceError::Other(e)) => Err(e),
                }
            }
            "prompts/list" => Ok(Answer::Cacheable(
                self.page("prompts/list", "prompts", prompts::list(), params)?,
                TTL_STATIC_MS,
            )),
            "prompts/get" => {
                let name = get_str(params, "name")
                    .ok_or_else(|| RpcError::invalid_params("prompts/get needs the name"))?;
                let arguments = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(JsonValue::empty_object);
                prompts::get(name, &arguments).map(Answer::Plain)
            }
            other => Err(Self::no_method(other)),
        }
    }

    fn no_method(method: &str) -> RpcError {
        RpcError::new(METHOD_NOT_FOUND, format!("Method not found: {method}"))
    }

    /// Look at what is waited for and watched; the messages to send.
    pub fn tick(&mut self) -> Vec<JsonValue> {
        let mut out = Vec::new();
        // Requests answered later.
        let pending = std::mem::take(&mut self.pending);
        for mut p in pending {
            let view = self.jobs.view(&p.job);
            let done = match (&p.awaited, &view) {
                (_, Err(_)) => true,
                (Awaited::Job { .. }, Ok(v)) => v.state != JobState::Running,
                (Awaited::TaskResult, Ok(v)) => TaskStatus::of(v).terminal(),
            };
            let late = p.deadline.is_some_and(|d| Instant::now() >= d);
            if done || late {
                let result = match p.awaited {
                    Awaited::Job { tail } => {
                        Ok(self.wait_result(&p.job, tail, p.started.elapsed().as_secs_f64()))
                    }
                    Awaited::TaskResult => self.task_result(&p.job),
                };
                out.push(match result {
                    Ok(r) => match Self::wrap(&p.era, Answer::Plain(r)) {
                        Some(r) => result_response(&p.id, r),
                        None => continue,
                    },
                    Err(e) => error_response(Some(&p.id), &e),
                });
                continue;
            }
            if let (Some(progress), Ok(v)) = (&mut p.progress, &view) {
                let now = (v.log_bytes, v.last_iteration.clone());
                if now != progress.seen {
                    progress.seen = now;
                    progress.count += 1.0;
                    let message = v
                        .last_iteration
                        .clone()
                        .unwrap_or_else(|| format!("{} running", v.command));
                    out.push(notification(
                        "notifications/progress",
                        object([
                            ("progressToken", progress.token.clone()),
                            ("progress", JsonValue::Number(progress.count)),
                            ("message", text(message)),
                        ]),
                    ));
                }
            }
            self.pending.push(p);
        }
        // Subscriptions of the revisions with the handshake.
        let mut subscribed = std::mem::take(&mut self.subscribed);
        for w in &mut subscribed {
            let now = self.fingerprint(&w.uri);
            if now != w.fingerprint {
                w.fingerprint = now;
                out.push(notification(
                    "notifications/resources/updated",
                    object([("uri", text(&w.uri))]),
                ));
            }
        }
        self.subscribed = subscribed;
        // Streams of subscriptions/listen.
        let mut listens = std::mem::take(&mut self.listens);
        for l in &mut listens {
            let meta = object([(META_SUBSCRIPTION_ID, l.id.clone())]);
            for w in &mut l.resources {
                let now = self.fingerprint(&w.uri);
                if now != w.fingerprint {
                    w.fingerprint = now;
                    out.push(notification(
                        "notifications/resources/updated",
                        object([("uri", text(&w.uri)), ("_meta", meta.clone())]),
                    ));
                }
            }
            for (task, status) in &mut l.tasks {
                let Ok(view) = self.jobs.view(task) else {
                    continue;
                };
                let now = TaskStatus::of(&view).name().to_string();
                if now != *status {
                    *status = now;
                    if let Ok(detailed) = self.detailed_task(task) {
                        out.push(notification(
                            "notifications/tasks",
                            with_key(detailed, "_meta", meta.clone()),
                        ));
                    }
                }
            }
        }
        self.listens = listens;
        // The status of the tasks of 2025-11-25 made here.
        let mut legacy = std::mem::take(&mut self.legacy_tasks);
        legacy.retain_mut(|(task, status)| {
            let Ok(view) = self.jobs.view(task) else {
                return false;
            };
            let now = TaskStatus::of(&view);
            if now.name() != status {
                *status = now.name().to_string();
                out.push(notification(
                    "notifications/tasks/status",
                    tasks::legacy_task(task, &view),
                ));
            }
            !now.terminal()
        });
        self.legacy_tasks = legacy;
        out
    }
}

/// `s` shortened for messages (at most 120 characters).
fn short(s: &str) -> String {
    let mut out: String = s.chars().take(120).collect();
    if s.chars().count() > 120 {
        out.push_str("...");
    }
    out.replace('\0', "\\0")
}
