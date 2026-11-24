//! What every command leaves in its run directory, read the same way for every command: the
//! run directories of a job, the tail of `mopt.log`, its last iteration, the totals of the
//! evaluations, and the files with their sizes.
//!
//! A run directory is a directory with `settings.json` or `mopt.log`; directories inside a run
//! directory belong to it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::protocol::{count, object, text};
use crate::cli::settings::RUN_SETTINGS_FILE;
use crate::core_infra::json::JsonValue;

/// The log of a run directory.
pub const RUN_LOG: &str = "mopt.log";
/// How deep run directories are looked for below a job directory.
pub const RUN_DEPTH: usize = 6;
/// The end of a file that [`tail`] reads.
const TAIL_BYTES: u64 = 256 * 1024;

/// Whether `dir` is a run directory.
pub fn is_run_directory(dir: &Path) -> bool {
    dir.join(RUN_SETTINGS_FILE).is_file() || dir.join(RUN_LOG).is_file()
}

/// The run directories at most `depth` levels below `dir` (not `dir` itself), outside
/// `excluded`, sorted.
pub fn run_directories(dir: &Path, excluded: &[PathBuf], depth: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut level = vec![dir.to_path_buf()];
    for _ in 0..depth {
        let mut next = Vec::new();
        for d in &level {
            for child in sorted_entries(d) {
                let Ok(meta) = std::fs::symlink_metadata(&child) else {
                    continue;
                };
                if !meta.is_dir() || excluded.iter().any(|x| child.starts_with(x)) {
                    continue;
                }
                if is_run_directory(&child) {
                    out.push(child);
                } else {
                    next.push(child);
                }
            }
        }
        level = next;
    }
    out.sort();
    out
}

/// The entries of `dir`, sorted (none when it cannot be read).
pub fn sorted_entries(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|it| it.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    out.sort();
    out
}

/// Whether `path` is a regular file itself (not a link to one).
pub fn is_plain_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file())
}

/// The text of the regular file `path` (not a link), when it has at most `limit` bytes.
pub fn read_text(path: &Path, limit: u64) -> Option<String> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if !meta.is_file() || meta.len() > limit {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// The last `lines` lines of the text file `path` (a regular file, not a link; read from its
/// last 256 KiB).
pub fn tail(path: &Path, lines: usize) -> Option<String> {
    if !is_plain_file(path) {
        return None;
    }
    let mut file = std::fs::File::open(path).ok()?;
    let size = file.metadata().ok()?.len();
    let start = size.saturating_sub(TAIL_BYTES);
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    let content = String::from_utf8_lossy(&bytes);
    let all: Vec<&str> = content.lines().collect();
    // The first line may be cut when the file was read from the middle.
    let from = if start > 0 { 1 } else { 0 };
    let keep = &all[from.min(all.len())..];
    let first = keep.len().saturating_sub(lines);
    Some(keep[first..].join("\n"))
}

/// The last iteration printed in a log: from the last line with `ITR.` to the end (at most 60
/// lines).
pub fn last_iteration(log: &str) -> Option<String> {
    let all: Vec<&str> = log.lines().collect();
    let at = all.iter().rposition(|l| l.contains("ITR."))?;
    let end = (at + 60).min(all.len());
    Some(all[at..end].join("\n"))
}

/// The totals of the last line `Evaluations (...): N gradient calculation(s), M exact Hessian
/// calculation(s)[, K calculation(s) of the energy alone].` in `content`.
pub fn evaluation_totals(content: &str) -> Option<JsonValue> {
    let line = content
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with("Evaluations ("))?
        .trim();
    let numbers = line.split_once("): ").map_or("", |(_, rest)| rest);
    let mut gradients = None;
    let mut hessians = None;
    let mut energies = None;
    for part in numbers.trim_end_matches('.').split(", ") {
        let mut words = part.split_whitespace();
        let Some(n) = words.next().and_then(|w| w.parse::<usize>().ok()) else {
            continue;
        };
        let rest: Vec<&str> = words.collect();
        let rest = rest.join(" ");
        if rest.starts_with("gradient") {
            gradients = Some(n);
        } else if rest.starts_with("exact Hessian") {
            hessians = Some(n);
        } else if rest.contains("energy alone") {
            energies = Some(n);
        }
    }
    Some(object([
        ("line", text(line)),
        ("gradients", gradients.map_or(JsonValue::Null, count)),
        ("hessians", hessians.map_or(JsonValue::Null, count)),
        ("energies", count(energies.unwrap_or(0))),
    ]))
}

/// The regular files below `dir` (recursively, symbolic links not followed) with their sizes,
/// sorted, at most `max`; and whether there were more.
pub fn list_files(dir: &Path, max: usize) -> (Vec<(PathBuf, u64)>, bool) {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    let mut more = false;
    while let Some(d) = stack.pop() {
        let mut subdirs = Vec::new();
        for entry in sorted_entries(&d) {
            let Ok(meta) = std::fs::symlink_metadata(&entry) else {
                continue;
            };
            if meta.is_dir() {
                subdirs.push(entry);
            } else if meta.is_file() {
                if out.len() >= max {
                    more = true;
                    continue;
                }
                out.push((entry, meta.len()));
            }
        }
        stack.extend(subdirs.into_iter().rev());
    }
    out.sort();
    (out, more)
}

/// The media type of a file name.
pub fn mime_type(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "md" => "text/markdown",
        "log" | "txt" | "out" | "dat" | "inp" | "xyz" | "gjf" | "mol" | "mol2" | "pdb" => {
            "text/plain"
        }
        "csv" => "text/csv",
        "json" => "application/json",
        "html" | "htm" => "text/html",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "pdf" => "application/pdf",
        "sqlite" | "db" => "application/vnd.sqlite3",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluation_lines_are_read() {
        let t = "x\nEvaluations (in total): 12 gradient calculation(s), 1 exact Hessian \
                 calculation(s), 3 calculation(s) of the energy alone.\n";
        let e = evaluation_totals(t).unwrap();
        assert_eq!(e.get("gradients"), Some(&JsonValue::Number(12.0)));
        assert_eq!(e.get("hessians"), Some(&JsonValue::Number(1.0)));
        assert_eq!(e.get("energies"), Some(&JsonValue::Number(3.0)));
        let e = evaluation_totals(
            "Evaluations (in total): 4 gradient calculation(s), 0 exact Hessian calculation(s).",
        )
        .unwrap();
        assert_eq!(e.get("energies"), Some(&JsonValue::Number(0.0)));
        assert!(evaluation_totals("nothing").is_none());
    }

    #[test]
    fn the_last_iteration_is_found() {
        let log = "start\n# ITR. 0\na\n# ITR. 1\nb\nc\n";
        assert_eq!(last_iteration(log).as_deref(), Some("# ITR. 1\nb\nc"));
        assert!(last_iteration("no iterations").is_none());
    }
}
