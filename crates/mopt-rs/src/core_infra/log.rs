//! Leveled logging to the console and to files at the same time.
//!
//! A [`Logger`] writes every message to all of its sinks whose level admits it, e.g. the console
//! at `Info` and a job log file at `Debug`. Sinks are pluggable ([`LogSink`]); [`MemorySink`] keeps
//! lines for tests. Loggers are cheap to clone and can be shared between threads.
//!
//! MultiOptPy: `optimization.py:RunLogger` and the `print` calls of `Optimize._print_info`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::error::{MoptError, MoptResult};

/// Message severity, from most to least severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    /// Failures.
    Error,
    /// Problems that do not stop the run.
    Warn,
    /// Normal progress output.
    Info,
    /// Details for diagnosis.
    Debug,
}

impl Level {
    /// Parse `error`, `warn`/`warning`, `info`, or `debug` (case-insensitive).
    pub fn parse(s: &str) -> MoptResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "warn" | "warning" => Ok(Self::Warn),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            other => Err(MoptError::parse(format!("unknown log level '{other}'"))),
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::Error => "Error: ",
            Self::Warn => "Warning: ",
            Self::Info | Self::Debug => "",
        }
    }
}

/// Destination of log lines.
pub trait LogSink: Send {
    /// Write one line (without trailing newline).
    fn write_line(&mut self, level: Level, line: &str) -> std::io::Result<()>;

    /// Flush buffered output.
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Standard output (standard error for errors and warnings).
#[derive(Debug, Default)]
pub struct ConsoleSink;

impl LogSink for ConsoleSink {
    fn write_line(&mut self, level: Level, line: &str) -> std::io::Result<()> {
        if level <= Level::Warn {
            writeln!(std::io::stderr().lock(), "{}{line}", level.prefix())
        } else {
            writeln!(std::io::stdout().lock(), "{line}")
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stdout().flush()
    }
}

/// A log file (appended to).
#[derive(Debug)]
pub struct FileSink {
    out: BufWriter<File>,
}

impl FileSink {
    /// Open `path` for appending, creating parent directories.
    pub fn append(path: impl AsRef<Path>) -> MoptResult<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            out: BufWriter::new(file),
        })
    }
}

impl LogSink for FileSink {
    fn write_line(&mut self, level: Level, line: &str) -> std::io::Result<()> {
        // The logger flushes after every whole message (see `Logger::log`).
        writeln!(self.out, "{}{line}", level.prefix())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush()
    }
}

/// Lines kept in memory (tests, embedding applications).
#[derive(Debug, Clone, Default)]
pub struct MemorySink {
    lines: Arc<Mutex<Vec<(Level, String)>>>,
}

impl MemorySink {
    /// Empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Lines written so far.
    pub fn lines(&self) -> Vec<(Level, String)> {
        self.lines.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

impl LogSink for MemorySink {
    fn write_line(&mut self, level: Level, line: &str) -> std::io::Result<()> {
        if let Ok(mut l) = self.lines.lock() {
            l.push((level, line.to_string()));
        }
        Ok(())
    }
}

struct Entry {
    max_level: Level,
    sink: Box<dyn LogSink>,
}

/// Logger writing to several sinks.
#[derive(Clone, Default)]
pub struct Logger {
    entries: Arc<Mutex<Vec<Entry>>>,
}

impl std::fmt::Debug for Logger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.entries.lock().map(|e| e.len()).unwrap_or(0);
        f.debug_struct("Logger").field("sinks", &n).finish()
    }
}

impl Logger {
    /// Logger without sinks (discards everything).
    pub fn silent() -> Self {
        Self::default()
    }

    /// Logger printing to the console up to `level`.
    pub fn console(level: Level) -> Self {
        let l = Self::default();
        l.add_sink(level, Box::new(ConsoleSink));
        l
    }

    /// Add `sink` receiving messages up to `max_level`.
    pub fn add_sink(&self, max_level: Level, sink: Box<dyn LogSink>) {
        if let Ok(mut e) = self.entries.lock() {
            e.push(Entry { max_level, sink });
        }
    }

    /// Also write messages up to `max_level` to the file at `path` (appending).
    pub fn add_file(&self, path: impl AsRef<Path>, max_level: Level) -> MoptResult<()> {
        self.add_sink(max_level, Box::new(FileSink::append(path)?));
        Ok(())
    }

    /// Write `msg` (possibly several lines) at `level`.
    pub fn log(&self, level: Level, msg: &str) {
        let Ok(mut entries) = self.entries.lock() else {
            return;
        };
        for e in entries.iter_mut().filter(|e| level <= e.max_level) {
            for line in msg.lines() {
                // A failing sink (closed pipe, full disk) must not abort the computation.
                let _ = e.sink.write_line(level, line);
            }
            if msg.is_empty() {
                let _ = e.sink.write_line(level, "");
            }
            // Every message is in the file whole when the call returns (for users following a
            // running job, and for a run that is killed): one write for its lines instead of
            // one per line (the structure of an iteration is one message of many lines).
            let _ = e.sink.flush();
        }
    }

    /// `log(Error, msg)`.
    pub fn error(&self, msg: &str) {
        self.log(Level::Error, msg);
    }

    /// `log(Warn, msg)`.
    pub fn warn(&self, msg: &str) {
        self.log(Level::Warn, msg);
    }

    /// `log(Info, msg)`.
    pub fn info(&self, msg: &str) {
        self.log(Level::Info, msg);
    }

    /// `log(Debug, msg)`.
    pub fn debug(&self, msg: &str) {
        self.log(Level::Debug, msg);
    }

    /// Flush all sinks.
    pub fn flush(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            for e in entries.iter_mut() {
                let _ = e.sink.flush();
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn levels_and_multiple_sinks() {
        let (all, info) = (MemorySink::new(), MemorySink::new());
        let log = Logger::silent();
        log.add_sink(Level::Debug, Box::new(all.clone()));
        log.add_sink(Level::Info, Box::new(info.clone()));
        log.debug("detail");
        log.info("a\nb");
        log.warn("careful");
        assert_eq!(all.lines().len(), 4);
        assert_eq!(info.lines().len(), 3);
        assert_eq!(info.lines()[0], (Level::Info, "a".to_string()));
    }

    #[test]
    fn file_sink_appends() {
        let dir = std::env::temp_dir().join(format!("mopt_log_test_{}", std::process::id()));
        let path = dir.join("job.log");
        let _ = std::fs::remove_file(&path);
        let log = Logger::silent();
        log.add_file(&path, Level::Info).unwrap();
        log.info("first");
        log.warn("second");
        log.debug("hidden");
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, "first\nWarning: second\n");
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// A number as Python prints a float, which is how MultiOptPy's console shows numbers it
/// does not format: the shortest digits that give the number back, `.0` after a whole
/// number, and an exponent with a sign and two digits below `1e-4` and from `1e16` on
/// (`0.5`, `2.0`, `8.517104648778046e-07`).
pub fn python_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let magnitude = x.abs();
    if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
        let text = format!("{x:e}");
        let (mantissa, exponent) = text.split_once('e').unwrap_or((&text, "0"));
        let (sign, digits) = match exponent.strip_prefix('-') {
            Some(digits) => ('-', digits),
            None => ('+', exponent),
        };
        return format!("{mantissa}e{sign}{digits:0>2}");
    }
    let text = format!("{x}");
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    }
}

#[cfg(test)]
mod python_float_tests {
    use super::python_float;

    #[test]
    fn numbers_look_as_in_python() {
        for (x, expected) in [
            (0.5, "0.5"),
            (2.0, "2.0"),
            (0.0, "0.0"),
            (-3.0, "-3.0"),
            (0.1 + 0.2, "0.30000000000000004"),
            (8.517104648778046e-7, "8.517104648778046e-07"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (123456789012345.0, "123456789012345.0"),
            (f64::INFINITY, "inf"),
        ] {
            assert_eq!(python_float(x), expected);
        }
        assert_eq!(python_float(f64::NAN), "nan");
    }
}
