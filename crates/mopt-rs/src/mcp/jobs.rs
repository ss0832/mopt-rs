//! Jobs: a command run as a child process of the same program in its own directory
//! `ROOT/mcp_jobs/<job id>/`, so that a failing run does not take the server down.
//!
//! A job uses only what every command has: its inputs are copied into the job directory, the
//! command reads its options from `mcp_settings.json` (`--config`), writes its run directory
//! there, stops at a file `STOP` in the run directory, and continues with `--restart RUN_DIR`.
//! The console output goes to `console.log`; the child records its exit code in
//! `mcp_exit_code`, so that a server started later still knows how a job ended
//! (`mcp_job.json` keeps what was started, and marks an attempt that was killed by
//! `job_cancel` or asked to stop by a cancelled task).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use super::job_settings::{self, CopyBudget, CopyRules, SETTINGS_FILE};
use super::paths::Sandbox;
use super::policy::Policy;
use super::protocol::{count, object, text};
use super::run_files::{self, RUN_DEPTH, RUN_LOG};
use crate::cli::option::CliOption;
use crate::cli::settings::{CONFIG_FLAG, RESTART_FLAG};
use crate::cli::{EXIT_INTERRUPTED, EXIT_OK};
use crate::core_infra::clock::DateTime;
use crate::core_infra::json::JsonValue;
use crate::workflow::observers::STOP_FILE;

/// What the server keeps about a job.
pub const JOB_FILE: &str = "mcp_job.json";
/// The exit code of the last attempt, written by the child.
pub const EXIT_FILE: &str = "mcp_exit_code";
/// The console output of the child.
pub const CONSOLE_FILE: &str = "console.log";
/// How long `start` waits for the run directory to appear.
const START_WAIT: Duration = Duration::from_secs(3);

/// How a job stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    /// The child runs.
    Running,
    /// Exit code 0.
    Finished,
    /// Exit code 3: stopped by `STOP`; `job_restart` continues it.
    Interrupted,
    /// Any other exit code, or a signal.
    Failed,
    /// Started by another server process and not seen running; no exit code recorded.
    Unknown,
    /// Killed by `job_cancel` (no exit code); `job_restart` continues it from its journal.
    Cancelled,
}

impl JobState {
    /// The name in results.
    pub fn name(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Finished => "finished",
            Self::Interrupted => "interrupted",
            Self::Failed => "failed",
            Self::Unknown => "unknown",
            Self::Cancelled => "cancelled",
        }
    }

    fn of_code(code: Option<i32>) -> Self {
        match code {
            Some(EXIT_OK) => Self::Finished,
            Some(EXIT_INTERRUPTED) => Self::Interrupted,
            _ => Self::Failed,
        }
    }
}

/// The program that runs jobs: an executable and the arguments before the command.
#[derive(Debug, Clone)]
pub struct JobProgram {
    /// The executable.
    pub executable: PathBuf,
    /// The arguments before the command (`mcp run-job`).
    pub prefix: Vec<String>,
}

/// The jobs of a server.
#[derive(Debug)]
pub struct JobManager {
    sandbox: Sandbox,
    program: JobProgram,
    policy: Policy,
    max_jobs: usize,
    children: BTreeMap<String, Child>,
    counter: u64,
}

/// The largest file of the server's own (records, exit codes) that is read.
const RECORD_LIMIT: u64 = 1 << 20;
/// The largest log that is read whole (for the evaluation totals).
const LOG_LIMIT: u64 = 64 << 20;

fn read_json(path: &Path) -> Option<JsonValue> {
    JsonValue::parse(&run_files::read_text(path, RECORD_LIMIT)?).ok()
}

/// Open `path` for writing (appending or truncating), refusing anything there that is not a
/// regular file (a link put in its place).
fn open_plain(path: &Path, append: bool) -> Result<std::fs::File, String> {
    let shown = || {
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    match std::fs::symlink_metadata(path) {
        Ok(m) if !m.is_file() => return Err(format!("{} is not a regular file", shown())),
        Ok(_) => {}
        Err(_) => {
            return OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|e| format!("{}: {e}", shown()))
        }
    }
    OpenOptions::new()
        .write(true)
        .append(append)
        .truncate(!append)
        .open(path)
        .map_err(|e| format!("{}: {e}", shown()))
}

fn write_json(path: &Path, value: &JsonValue) -> Result<(), String> {
    use std::io::Write as _;
    let mut file = open_plain(path, false)?;
    file.write_all((value.to_string_pretty() + "\n").as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn stamp(t: &DateTime) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        t.year, t.month, t.day, t.hour, t.minute, t.second
    )
}

/// The time `t` (since the Unix epoch) as `YYYY-MM-DDTHH:MM:SSZ`.
fn stamp_of(t: std::time::SystemTime) -> String {
    let micros = t
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX));
    stamp(&DateTime::from_unix_micros(micros))
}

/// What a task needs to know about a job.
#[derive(Debug, Clone, PartialEq)]
pub struct JobView {
    /// The state of the latest attempt.
    pub state: JobState,
    /// Its exit code.
    pub exit_code: Option<i32>,
    /// Whether a cancelled task asked the latest attempt to stop.
    pub cancel_requested: bool,
    /// When the job was created (`YYYY-MM-DDTHH:MM:SSZ`).
    pub created: String,
    /// When the job last changed (its records, its exit code, its log).
    pub updated: String,
    /// The command.
    pub command: String,
    /// The last iteration printed in `mopt.log`.
    pub last_iteration: Option<String>,
    /// The size of `mopt.log` of its last run directory (bytes).
    pub log_bytes: u64,
}

fn strings_of(value: Option<&JsonValue>) -> Vec<String> {
    value.and_then(JsonValue::as_string_vec).unwrap_or_default()
}

/// Whether the process `pid` runs in `dir` (where the system tells; `/proc`).
fn alive_in(pid: u32, dir: &Path) -> bool {
    let cwd = Path::new("/proc").join(pid.to_string()).join("cwd");
    match std::fs::read_link(cwd) {
        Ok(c) => std::fs::canonicalize(c).is_ok_and(|c| c == dir),
        Err(_) => false,
    }
}

/// The most lines of one `mapper_steer` call.
pub const MAX_STEER_LINES: usize = 100;

fn text_value(s: impl AsRef<str>) -> JsonValue {
    text(s)
}

/// Why `line` is not a steering line of `run_mapper` (`None`: it is one): `explore NAME`,
/// `skip NAME`, `unskip NAME` (a label such as `EQ5` or a number) or `max_jobs N`, without
/// control characters or comments.
pub fn steer_line_refused(line: &str) -> Option<String> {
    let shown: String = line
        .chars()
        .take(80)
        .collect::<String>()
        .replace('\0', "\\0");
    if line.chars().any(char::is_control) || line.contains('#') || line.len() > 200 {
        return Some(format!(
            "'{shown}': one line without control characters or comments"
        ));
    }
    let words: Vec<&str> = line.split_whitespace().collect();
    let name = |w: &str| {
        !w.is_empty()
            && w.len() <= 32
            && w.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    };
    let valid = match words.as_slice() {
        ["explore" | "skip" | "unskip", target] => name(target),
        ["max_jobs", n] => n.parse::<u64>().is_ok(),
        _ => false,
    };
    (!valid).then(|| {
        format!(
            "'{shown}' is not a steering line (explore NAME, skip NAME, unskip NAME, max_jobs N)"
        )
    })
}

/// Whether `id` has the form of the job ids this server makes (`job-` and 32 hexadecimal
/// digits).
pub fn is_job_id(id: &str) -> bool {
    id.strip_prefix("job-").is_some_and(|h| {
        h.len() == 32
            && h.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

impl JobManager {
    /// Jobs under the root of `sandbox`, run by `program`, at most `max_jobs` at a time, with
    /// the settings that `policy` allows.
    pub fn new(sandbox: Sandbox, program: JobProgram, policy: Policy, max_jobs: usize) -> Self {
        Self {
            sandbox,
            program,
            policy,
            max_jobs,
            children: BTreeMap::new(),
            counter: 0,
        }
    }

    /// The sandbox.
    pub fn sandbox(&self) -> &Sandbox {
        &self.sandbox
    }

    /// The policy.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    fn rel(&self, path: &Path) -> String {
        self.sandbox.relative(path).unwrap_or_default()
    }

    /// A new opaque job id (`job-` and 32 hexadecimal digits from the random keys of the
    /// standard library's hasher, the time and the process).
    fn new_id(&mut self, jobs: &Path) -> String {
        use std::hash::BuildHasher;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        loop {
            self.counter += 1;
            let seed = (self.counter, nanos, std::process::id());
            let a = std::collections::hash_map::RandomState::new().hash_one(seed);
            let b = std::collections::hash_map::RandomState::new().hash_one((seed, a));
            let id = format!("job-{a:016x}{b:016x}");
            if !jobs.join(&id).exists() {
                return id;
            }
        }
    }

    /// The directory of the job `id`: the directory under `mcp_jobs/` whose record names this
    /// id. The id is compared, never used as a path.
    pub fn job_dir(&self, id: &str) -> Result<PathBuf, String> {
        if !is_job_id(id) {
            return Err(format!(
                "'{}' is not a job id of this server",
                id.chars().take(80).collect::<String>()
            ));
        }
        let jobs = self.sandbox.jobs_dir()?;
        for entry in run_files::sorted_entries(&jobs) {
            let Ok(meta) = std::fs::symlink_metadata(&entry) else {
                continue;
            };
            if !meta.is_dir() {
                continue;
            }
            let named = read_json(&entry.join(JOB_FILE)).and_then(|r| {
                r.get("job_id")
                    .and_then(JsonValue::as_str)
                    .map(String::from)
            });
            if named.as_deref() == Some(id) {
                let dir = std::fs::canonicalize(&entry).map_err(|e| e.to_string())?;
                if self.sandbox.contains(&dir) {
                    return Ok(dir);
                }
            }
        }
        Err(format!("no job '{id}'"))
    }

    /// The jobs of this server process that run (finished children keep their exit status).
    fn running(&mut self) -> usize {
        let mut running = 0;
        for child in self.children.values_mut() {
            if matches!(child.try_wait(), Ok(None)) {
                running += 1;
            }
        }
        running
    }

    /// Start `command` (with `options`) for the arguments `args` of its tool.
    pub fn start(
        &mut self,
        command: &str,
        options: &[CliOption],
        args: &JsonValue,
    ) -> Result<JsonValue, String> {
        let request = job_settings::request_of(command, options, args)?;
        self.policy.check_settings(&request.settings)?;
        if self.running() >= self.max_jobs {
            return Err(format!(
                "{} jobs of this server are running, the most it starts at a time (--max-jobs); \
                 wait for one to end or stop one",
                self.max_jobs
            ));
        }
        let jobs = self.sandbox.jobs_dir()?;
        // Everything is checked before anything is written.
        let mut sources = Vec::new();
        for p in request.inputs.iter().chain(&request.files) {
            sources.push(self.sandbox.existing(p)?);
        }
        let id = self.new_id(&jobs);
        let dir = jobs.join(&id);
        std::fs::create_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let dir = std::fs::canonicalize(&dir).map_err(|e| e.to_string())?;
        let mut budget = CopyBudget::default();
        let mut copied: Vec<String> = Vec::new();
        let prepared = (|| {
            job_settings::check_paths(&self.sandbox, &dir, &request.settings)?;
            let rules = CopyRules {
                sandbox: &self.sandbox,
                jobs_dir: &jobs,
                policy: &self.policy,
            };
            // Each file once; a file inside a directory that is copied comes with it.
            let mut done: Vec<&PathBuf> = Vec::new();
            for source in &sources {
                let covered = sources
                    .iter()
                    .any(|o| o != source && o.is_dir() && source.starts_with(o));
                if covered || done.contains(&source) {
                    continue;
                }
                copied.push(job_settings::copy_into(rules, source, &dir, &mut budget)?);
                done.push(source);
            }
            for doc in &budget.json {
                self.policy.check_json(doc)?;
                job_settings::check_json_paths(&self.sandbox, &dir, doc)?;
            }
            Ok::<(), String>(())
        })();
        if let Err(e) = prepared {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(e);
        }
        let inputs: Vec<String> = sources[..request.inputs.len()]
            .iter()
            .map(|s| self.rel(s))
            .collect();
        let document = job_settings::settings_document(&inputs, &request.settings);
        write_json(&dir.join(SETTINGS_FILE), &document)?;
        let record = object([
            ("job_id", text(&id)),
            ("command", text(command)),
            ("created", text(stamp(&DateTime::now_utc()))),
            (
                "inputs",
                JsonValue::Array(inputs.iter().map(text).collect()),
            ),
            (
                "copied",
                JsonValue::Array(copied.iter().map(text).collect()),
            ),
            ("attempts", JsonValue::Array(Vec::new())),
        ]);
        write_json(&dir.join(JOB_FILE), &record)?;
        let arguments = vec![CONFIG_FLAG.to_string(), SETTINGS_FILE.to_string()];
        self.spawn(&id, &dir, command, &arguments)?;
        let runs = self.wait_for_run_directory(&id, &dir);
        let mut out = self.status_of(&id, &dir, 10)?;
        if let Some(map) = out.as_object_mut() {
            map.insert(
                "run_directory".to_string(),
                runs.first().map_or(JsonValue::Null, |r| text(self.rel(r))),
            );
            if !budget.skipped.is_empty() {
                map.insert(
                    "skipped_links".to_string(),
                    JsonValue::Array(budget.skipped.iter().map(text).collect()),
                );
            }
            map.insert(
                "next".to_string(),
                text("call job_status with this job_id until the state is not running; then job_result"),
            );
        }
        Ok(out)
    }

    fn spawn(
        &mut self,
        id: &str,
        dir: &Path,
        command: &str,
        arguments: &[String],
    ) -> Result<(), String> {
        let _ = std::fs::remove_file(dir.join(EXIT_FILE));
        let console = open_plain(&dir.join(CONSOLE_FILE), true)?;
        let errors = console.try_clone().map_err(|e| e.to_string())?;
        let child = Command::new(&self.program.executable)
            .args(&self.program.prefix)
            .arg(command)
            .args(arguments)
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::from(console))
            .stderr(Stdio::from(errors))
            .spawn()
            .map_err(|e| format!("the job could not be started: {e}"))?;
        let mut record = read_json(&dir.join(JOB_FILE)).unwrap_or_else(JsonValue::empty_object);
        if let Some(map) = record.as_object_mut() {
            let attempt = object([
                ("started", text(stamp(&DateTime::now_utc()))),
                (
                    "arguments",
                    JsonValue::Array(
                        std::iter::once(command)
                            .chain(arguments.iter().map(String::as_str))
                            .map(text)
                            .collect(),
                    ),
                ),
                ("pid", JsonValue::Number(f64::from(child.id()))),
            ]);
            match map.get_mut("attempts") {
                Some(JsonValue::Array(a)) => a.push(attempt),
                _ => {
                    map.insert("attempts".to_string(), JsonValue::Array(vec![attempt]));
                }
            }
        }
        write_json(&dir.join(JOB_FILE), &record)?;
        eprintln!("mopt mcp: job {id}: {command} {}", arguments.join(" "));
        self.children.insert(id.to_string(), child);
        Ok(())
    }

    fn wait_for_run_directory(&mut self, id: &str, dir: &Path) -> Vec<PathBuf> {
        let begin = Instant::now();
        loop {
            let runs = self.run_directories(dir);
            if !runs.is_empty() || begin.elapsed() > START_WAIT {
                return runs;
            }
            if let Some(child) = self.children.get_mut(id) {
                if !matches!(child.try_wait(), Ok(None)) {
                    return runs;
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// The run directories of the job in `dir`.
    pub fn run_directories(&self, dir: &Path) -> Vec<PathBuf> {
        let record = read_json(&dir.join(JOB_FILE));
        let copied = strings_of(record.as_ref().and_then(|r| r.get("copied")));
        run_files::run_directories(dir, &job_settings::copied_paths(dir, &copied), RUN_DEPTH)
    }

    /// The latest attempt recorded for the job in `dir`.
    fn last_attempt(dir: &Path) -> Option<JsonValue> {
        read_json(&dir.join(JOB_FILE)).and_then(|r| {
            r.get("attempts")
                .and_then(JsonValue::as_array)
                .and_then(|a| a.last().cloned())
        })
    }

    /// Whether the latest attempt in `dir` carries the mark `key`.
    fn attempt_marked(dir: &Path, key: &str) -> bool {
        Self::last_attempt(dir)
            .and_then(|a| a.get(key).and_then(JsonValue::as_bool))
            .unwrap_or(false)
    }

    /// Mark the latest attempt of the job in `dir` with `key: true`.
    fn mark_attempt(dir: &Path, key: &str) -> Result<(), String> {
        let path = dir.join(JOB_FILE);
        let mut record = read_json(&path).ok_or_else(|| format!("{JOB_FILE} cannot be read"))?;
        if let Some(JsonValue::Array(attempts)) =
            record.as_object_mut().and_then(|m| m.get_mut("attempts"))
        {
            if let Some(JsonValue::Object(last)) = attempts.last_mut() {
                last.insert(key.to_string(), JsonValue::Bool(true));
            }
        }
        write_json(&path, &record)
    }

    /// The state and exit code of the job `id` in `dir`.
    pub fn state(&mut self, id: &str, dir: &Path) -> (JobState, Option<i32>) {
        let recorded = run_files::read_text(&dir.join(EXIT_FILE), RECORD_LIMIT)
            .and_then(|t| t.trim().parse::<i32>().ok());
        if let Some(child) = self.children.get_mut(id) {
            match child.try_wait() {
                Ok(None) => return (JobState::Running, None),
                Ok(Some(status)) => {
                    self.children.remove(id);
                    if recorded.is_none() && Self::attempt_marked(dir, "cancelled") {
                        return (JobState::Cancelled, None);
                    }
                    let code = recorded.or(status.code());
                    return (JobState::of_code(code), code);
                }
                Err(_) => {
                    self.children.remove(id);
                }
            }
        }
        if let Some(code) = recorded {
            return (JobState::of_code(Some(code)), Some(code));
        }
        let last = Self::last_attempt(dir);
        if last
            .as_ref()
            .and_then(|a| a.get("cancelled").and_then(JsonValue::as_bool))
            .unwrap_or(false)
        {
            return (JobState::Cancelled, None);
        }
        let pid = last
            .and_then(|a| a.get("pid").and_then(JsonValue::as_f64))
            .map(|p| p as u32);
        match pid {
            Some(p) if alive_in(p, dir) => (JobState::Running, None),
            _ => (JobState::Unknown, None),
        }
    }

    /// End the running job `id` of this server process at once (the process is killed).
    pub fn cancel(&mut self, id: &str) -> Result<JsonValue, String> {
        let dir = self.job_dir(id)?;
        let (state, _) = self.state(id, &dir);
        if state != JobState::Running {
            return Err(format!("job {id} is not running (state {})", state.name()));
        }
        let Some(child) = self.children.get_mut(id) else {
            return Err(format!(
                "job {id} was started by another server process, which this one cannot end; \
                 job_stop asks it to end at its next check"
            ));
        };
        // A child that ended on its own meanwhile keeps its exit code.
        let _ = child.kill();
        let _ = child.wait();
        self.children.remove(id);
        if !dir.join(EXIT_FILE).exists() {
            Self::mark_attempt(&dir, "cancelled")?;
        }
        let mut out = self.status_of(id, &dir, 10)?;
        if let Some(map) = out.as_object_mut() {
            map.insert(
                "next".to_string(),
                text("job_restart continues the run from its evaluation journal"),
            );
        }
        Ok(out)
    }

    /// Ask the job `id` to end, for a cancelled task: `STOP` in its run directories (the
    /// command ends at its next check, as with `job_stop`), or the end of the process when it
    /// has no run directory yet. The latest attempt is marked, so that the task stays
    /// cancelled.
    pub fn request_cancel(&mut self, id: &str) -> Result<(), String> {
        let dir = self.job_dir(id)?;
        let (state, _) = self.state(id, &dir);
        if state != JobState::Running {
            return Err(format!("job {id} is not running (state {})", state.name()));
        }
        Self::mark_attempt(&dir, "cancel_requested")?;
        if self.run_directories(&dir).is_empty() {
            self.cancel(id).map(|_| ())
        } else {
            self.stop(id).map(|_| ())
        }
    }

    /// What a task needs to know about the job `id`.
    pub fn view(&mut self, id: &str) -> Result<JobView, String> {
        let dir = self.job_dir(id)?;
        let (state, exit_code) = self.state(id, &dir);
        let record = read_json(&dir.join(JOB_FILE)).unwrap_or_else(JsonValue::empty_object);
        let runs = self.run_directories(&dir);
        let log = runs.last().map(|r| r.join(RUN_LOG));
        let log_bytes = log
            .as_ref()
            .and_then(|l| std::fs::symlink_metadata(l).ok())
            .filter(std::fs::Metadata::is_file)
            .map_or(0, |m| m.len());
        let last_iteration = log
            .as_ref()
            .and_then(|l| run_files::tail(l, 400))
            .as_deref()
            .and_then(run_files::last_iteration);
        let updated = [dir.join(JOB_FILE), dir.join(EXIT_FILE)]
            .into_iter()
            .chain(log)
            .filter_map(|f| std::fs::symlink_metadata(f).ok()?.modified().ok())
            .max()
            .map_or_else(|| stamp(&DateTime::now_utc()), stamp_of);
        let created = record
            .get("created")
            .and_then(JsonValue::as_str)
            .map_or_else(|| updated.clone(), String::from);
        Ok(JobView {
            state,
            exit_code,
            cancel_requested: Self::attempt_marked(&dir, "cancel_requested"),
            created,
            updated,
            command: record
                .get("command")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_string(),
            last_iteration,
            log_bytes,
        })
    }

    /// Append the steering lines `lines` to `steer.txt` in the run directory of the running
    /// `run_mapper` job `id` (the run reads them before its next choice of jobs and writes what
    /// became of them to `steer.log`). Every line is checked by its form first:
    /// [`steer_line_refused`].
    pub fn steer(&mut self, id: &str, lines: &[String]) -> Result<JsonValue, String> {
        if lines.is_empty() || lines.len() > MAX_STEER_LINES {
            return Err(format!("give 1 to {MAX_STEER_LINES} lines"));
        }
        for line in lines {
            if let Some(why) = steer_line_refused(line) {
                return Err(why);
            }
        }
        let dir = self.job_dir(id)?;
        let record = read_json(&dir.join(JOB_FILE)).unwrap_or_else(JsonValue::empty_object);
        if record.get("command").and_then(JsonValue::as_str) != Some("run_mapper") {
            return Err(format!("job {id} is no run_mapper job"));
        }
        let (state, _) = self.state(id, &dir);
        if state != JobState::Running {
            return Err(format!("job {id} is not running (state {})", state.name()));
        }
        let runs = self.run_directories(&dir);
        let Some(run) = runs.last() else {
            return Err(format!(
                "job {id} has no run directory yet; try again in a moment"
            ));
        };
        let file = run.join(crate::workflow::mapper::steer::STEER_FILE);
        let mut opened = open_plain(&file, true)?;
        let text: String = lines.iter().map(|l| format!("{}\n", l.trim())).collect();
        std::io::Write::write_all(&mut opened, text.as_bytes()).map_err(|e| e.to_string())?;
        let log = run.join(crate::workflow::mapper::steer::STEER_LOG);
        Ok(object([
            ("job_id", text_value(id)),
            ("steer_file", text_value(self.rel(&file))),
            (
                "lines",
                JsonValue::Array(lines.iter().map(|l| text_value(l.trim())).collect()),
            ),
            (
                "log",
                text_value(super::resources::root_uri(&self.rel(&log))),
            ),
            (
                "next",
                text_value(
                    "the run reads the lines before its next choice of jobs; steer.log (the log \
                     resource) tells what became of each",
                ),
            ),
        ]))
    }

    /// The ids of the jobs under the root, in the order of their directories.
    pub fn ids(&self) -> Vec<String> {
        let Ok(jobs) = self.sandbox.jobs_dir() else {
            return Vec::new();
        };
        run_files::sorted_entries(&jobs)
            .into_iter()
            .filter(|e| std::fs::symlink_metadata(e).is_ok_and(|m| m.is_dir()))
            .filter_map(|e| {
                read_json(&e.join(JOB_FILE))?
                    .get("job_id")
                    .and_then(JsonValue::as_str)
                    .map(String::from)
            })
            .filter(|id| is_job_id(id))
            .collect()
    }

    /// The status of the job `id`.
    pub fn status(&mut self, id: &str, tail_lines: usize) -> Result<JsonValue, String> {
        let dir = self.job_dir(id)?;
        self.status_of(id, &dir, tail_lines)
    }

    fn status_of(&mut self, id: &str, dir: &Path, tail_lines: usize) -> Result<JsonValue, String> {
        let (state, code) = self.state(id, dir);
        let record = read_json(&dir.join(JOB_FILE)).unwrap_or_else(JsonValue::empty_object);
        let runs = self.run_directories(dir);
        let log = runs.last().map(|r| r.join(RUN_LOG));
        let log_text = log.as_ref().and_then(|l| run_files::tail(l, 400));
        let lines = |t: &str, n: usize| {
            let all: Vec<&str> = t.lines().collect();
            all[all.len().saturating_sub(n)..].join("\n")
        };
        let attempts = record
            .get("attempts")
            .and_then(JsonValue::as_array)
            .map_or(0, <[JsonValue]>::len);
        let mut map = BTreeMap::new();
        map.insert("job_id".to_string(), text(id));
        map.insert(
            "command".to_string(),
            record.get("command").cloned().unwrap_or(JsonValue::Null),
        );
        map.insert("state".to_string(), text(state.name()));
        map.insert(
            "exit_code".to_string(),
            code.map_or(JsonValue::Null, |c| JsonValue::Number(f64::from(c))),
        );
        map.insert("attempt".to_string(), count(attempts));
        map.insert("job_directory".to_string(), text(self.rel(dir)));
        map.insert(
            "run_directories".to_string(),
            JsonValue::Array(runs.iter().map(|r| text(self.rel(r))).collect()),
        );
        map.insert(
            "log_tail".to_string(),
            log_text
                .as_deref()
                .map_or(JsonValue::Null, |t| text(lines(t, tail_lines))),
        );
        map.insert(
            "last_iteration".to_string(),
            log_text
                .as_deref()
                .and_then(run_files::last_iteration)
                .map_or(JsonValue::Null, text),
        );
        map.insert(
            "console_tail".to_string(),
            run_files::tail(&dir.join(CONSOLE_FILE), 20).map_or(JsonValue::Null, text),
        );
        if state == JobState::Unknown {
            map.insert(
                "note".to_string(),
                text(
                    "this job was started by another server process and no exit code was \
                     recorded: it may still run there, or it ended without recording one",
                ),
            );
        }
        Ok(JsonValue::Object(map))
    }

    /// Write `STOP` into the run directories of the running job `id`.
    pub fn stop(&mut self, id: &str) -> Result<JsonValue, String> {
        let dir = self.job_dir(id)?;
        let (state, _) = self.state(id, &dir);
        if !matches!(state, JobState::Running | JobState::Unknown) {
            return Err(format!("job {id} is not running (state {})", state.name()));
        }
        let runs = self.run_directories(&dir);
        if runs.is_empty() {
            return Err(format!(
                "job {id} has no run directory yet; try again in a moment"
            ));
        }
        for r in &runs {
            open_plain(&r.join(STOP_FILE), false)?;
        }
        Ok(object([
            ("job_id", text(id)),
            (
                "stop_files",
                JsonValue::Array(
                    runs.iter()
                        .map(|r| text(self.rel(&r.join(STOP_FILE))))
                        .collect(),
                ),
            ),
            (
                "next",
                text(
                    "the command ends at its next check of the STOP file; job_status shows \
                     state interrupted then, and job_restart continues the run",
                ),
            ),
        ]))
    }

    /// Continue the job `id` with `--restart` in its run directory `run` (needed when it has
    /// several).
    pub fn restart(
        &mut self,
        id: &str,
        run: Option<&str>,
        force: bool,
    ) -> Result<JsonValue, String> {
        let dir = self.job_dir(id)?;
        let (state, _) = self.state(id, &dir);
        match state {
            JobState::Running => return Err(format!("job {id} is running; stop it first")),
            JobState::Unknown if !force => {
                return Err(format!(
                    "the state of job {id} is unknown (another server process started it); \
                     give force=true once it surely does not run any more"
                ))
            }
            _ => {}
        }
        let runs = self.run_directories(&dir);
        let chosen = match run {
            Some(r) => {
                let wanted = self.sandbox.existing(r)?;
                runs.iter()
                    .find(|x| **x == wanted)
                    .cloned()
                    .ok_or_else(|| format!("{r} is not a run directory of job {id}"))?
            }
            None => match runs.as_slice() {
                [one] => one.clone(),
                [] => return Err(format!("job {id} has no run directory to continue")),
                _ => {
                    return Err(format!(
                        "job {id} has several run directories; name one as run_directory: {}",
                        runs.iter()
                            .map(|r| self.rel(r))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                }
            },
        };
        let record = read_json(&dir.join(JOB_FILE)).unwrap_or_else(JsonValue::empty_object);
        let command = record
            .get("command")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| format!("{JOB_FILE} of job {id} names no command"))?
            .to_string();
        let within: Vec<String> = chosen
            .strip_prefix(&dir)
            .map_err(|_| "the run directory is outside the job".to_string())?
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        let arguments = vec![RESTART_FLAG.to_string(), within.join("/")];
        self.spawn(id, &dir, &command, &arguments)?;
        let mut out = self.status_of(id, &dir, 10)?;
        if let Some(map) = out.as_object_mut() {
            map.insert("run_directory".to_string(), text(self.rel(&chosen)));
        }
        Ok(out)
    }

    /// The files of the run directories of the job `id` and the totals of its evaluations.
    pub fn result(&mut self, id: &str) -> Result<JsonValue, String> {
        let dir = self.job_dir(id)?;
        let (state, code) = self.state(id, &dir);
        let runs = self.run_directories(&dir);
        let mut run_entries = Vec::new();
        for r in &runs {
            let (files, more) = run_files::list_files(r, 2000);
            let entries: Vec<JsonValue> = files
                .iter()
                .map(|(f, size)| {
                    let rel = self.rel(f);
                    object([
                        ("path", text(&rel)),
                        ("size", JsonValue::Number(*size as f64)),
                        ("mimeType", text(run_files::mime_type(f))),
                        ("uri", text(super::resources::root_uri(&rel))),
                    ])
                })
                .collect();
            let log = run_files::read_text(&r.join(RUN_LOG), LOG_LIMIT).unwrap_or_default();
            run_entries.push(object([
                ("path", text(self.rel(r))),
                ("files", JsonValue::Array(entries)),
                ("more_files", JsonValue::Bool(more)),
                (
                    "evaluations",
                    run_files::evaluation_totals(&log).unwrap_or(JsonValue::Null),
                ),
            ]));
        }
        let console = run_files::read_text(&dir.join(CONSOLE_FILE), LOG_LIMIT).unwrap_or_default();
        Ok(object([
            ("job_id", text(id)),
            ("state", text(state.name())),
            (
                "exit_code",
                code.map_or(JsonValue::Null, |c| JsonValue::Number(f64::from(c))),
            ),
            ("run_directories", JsonValue::Array(run_entries)),
            (
                "evaluations",
                run_files::evaluation_totals(&console).unwrap_or(JsonValue::Null),
            ),
            (
                "job_files",
                JsonValue::Array(
                    [JOB_FILE, SETTINGS_FILE, CONSOLE_FILE, EXIT_FILE]
                        .iter()
                        .filter(|f| run_files::is_plain_file(&dir.join(f)))
                        .map(|f| text(super::resources::root_uri(&self.rel(&dir.join(f)))))
                        .collect(),
                ),
            ),
        ]))
    }

    /// The jobs under the root.
    pub fn list(&mut self) -> Result<JsonValue, String> {
        let jobs = self.sandbox.jobs_dir()?;
        let mut out = Vec::new();
        for entry in run_files::sorted_entries(&jobs) {
            if !std::fs::symlink_metadata(&entry).is_ok_and(|m| m.is_dir()) {
                continue;
            }
            let Some(record) = read_json(&entry.join(JOB_FILE)) else {
                continue;
            };
            let Some(id) = record
                .get("job_id")
                .and_then(JsonValue::as_str)
                .map(String::from)
            else {
                continue;
            };
            let Ok(dir) = std::fs::canonicalize(&entry) else {
                continue;
            };
            if !is_job_id(&id) || !self.sandbox.contains(&dir) {
                continue;
            }
            let (state, code) = self.state(&id, &dir);
            out.push(object([
                ("job_id", text(&id)),
                (
                    "command",
                    record.get("command").cloned().unwrap_or(JsonValue::Null),
                ),
                (
                    "created",
                    record.get("created").cloned().unwrap_or(JsonValue::Null),
                ),
                ("state", text(state.name())),
                (
                    "exit_code",
                    code.map_or(JsonValue::Null, |c| JsonValue::Number(f64::from(c))),
                ),
                (
                    "run_directories",
                    JsonValue::Array(
                        self.run_directories(&dir)
                            .iter()
                            .map(|r| text(self.rel(r)))
                            .collect(),
                    ),
                ),
            ]));
        }
        Ok(object([("jobs", JsonValue::Array(out))]))
    }
}
