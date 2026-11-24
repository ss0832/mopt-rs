//! Resources: the documentation pages, the files of run directories under the root, and the
//! status of jobs.
//!
//! - `mopt://docs/NAME`: a page of the documentation: the copy embedded in the `mopt` binary
//!   ([`Docs::Embedded`]), or `docs/source` of the repository that holds the root
//!   ([`Docs::Directory`], for servers made without the embedded copy). Only the names of the
//!   pages can be read.
//! - `mopt://root/PATH`: a file (or directory listing) under the root; `resources/list` lists
//!   the files of the run directories found there. Text files are returned as text, other
//!   files as Base64 up to [`READ_LIMIT`] bytes; larger ones are refused with their size.
//! - `mopt://jobs/JOB_ID`: the status of a job as JSON (as `job_status`), answered by the
//!   server ([`JOBS_PREFIX`]).
//!
//! Resources can be subscribed to: [`check_watch`] checks a URI as `resources/read` would,
//! and [`fingerprint`] tells when it changed (the metadata of the file it resolves to now,
//! which must still lie inside the root).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use super::paths::Sandbox;
use super::protocol::{count, object, text, RpcError, INVALID_PARAMS};
use super::run_files;
use crate::core_infra::json::JsonValue;

/// Scheme and authority of the documentation pages.
pub const DOCS_PREFIX: &str = "mopt://docs/";
/// Scheme and authority of the files under the root.
pub const ROOT_PREFIX: &str = "mopt://root/";
/// Scheme and authority of the status of jobs.
pub const JOBS_PREFIX: &str = "mopt://jobs/";

/// The pages of the documentation embedded in a binary: `(name without .md, text)`.
pub type EmbeddedDocs = &'static [(&'static str, &'static str)];

/// Where the documentation comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Docs {
    /// The pages embedded in the binary.
    Embedded(EmbeddedDocs),
    /// The `docs/source` directory of a repository.
    Directory(PathBuf),
    /// None.
    Missing,
}

impl Docs {
    /// The documentation of a server of `root`: `embedded` when it has pages, else
    /// `docs/source` of the repository that holds the root, if any.
    pub fn of(root: &Path, embedded: EmbeddedDocs) -> Self {
        if !embedded.is_empty() {
            return Self::Embedded(embedded);
        }
        find_docs(root).map_or(Self::Missing, Self::Directory)
    }

    /// Where the pages are, for the log.
    pub fn describe(&self) -> String {
        match self {
            Self::Embedded(pages) => format!("{} pages embedded in the binary", pages.len()),
            Self::Directory(d) => d.display().to_string(),
            Self::Missing => "none".to_string(),
        }
    }

    /// `(name, size, title)` of every page, sorted by name.
    pub fn pages(&self) -> Vec<(String, u64, String)> {
        let mut out: Vec<(String, u64, String)> = match self {
            Self::Embedded(pages) => pages
                .iter()
                .map(|(name, content)| {
                    (
                        name.to_string(),
                        content.len() as u64,
                        title_in(content).unwrap_or_else(|| name.to_string()),
                    )
                })
                .collect(),
            Self::Directory(d) => doc_pages(d)
                .into_iter()
                .map(|(name, path)| {
                    let size = std::fs::metadata(&path).map_or(0, |m| m.len());
                    let title = title_of(&path).unwrap_or_else(|| name.clone());
                    (name, size, title)
                })
                .collect(),
            Self::Missing => Vec::new(),
        };
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The text of the page `name` (by its name only, never as a path).
    pub fn text(&self, name: &str) -> Option<String> {
        match self {
            Self::Embedded(pages) => pages
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, content)| content.to_string()),
            Self::Directory(d) => doc_pages(d)
                .into_iter()
                .find(|(n, _)| n == name)
                .and_then(|(_, path)| std::fs::read_to_string(path).ok()),
            Self::Missing => None,
        }
    }
}
/// The largest file `resources/read` returns.
pub const READ_LIMIT: u64 = 8 << 20;
/// At most this many resources are listed.
pub const LIST_LIMIT: usize = 1000;
/// How deep run directories are looked for below the root.
const SCAN_DEPTH: usize = 8;
/// At most this many directories are visited when looking for run directories.
const SCAN_DIRECTORIES: usize = 20_000;

/// Why a resource could not be read.
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceError {
    /// No such resource.
    NotFound(String),
    /// Any other failure.
    Other(RpcError),
}

/// `rel` (relative to the root, `/` separators) as a resource URI.
pub fn root_uri(rel: &str) -> String {
    let mut out = String::from(ROOT_PREFIX);
    for b in rel.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Why the encoded path of a URI is refused before it is decoded: encoded dots, slashes,
/// backslashes and NUL characters (`%2e`, `%2f`, `%5c`, `%00`), which only serve to hide a
/// path from a check.
pub fn refused_encoding(encoded: &str) -> Option<&'static str> {
    let lower = encoded.to_ascii_lowercase();
    if ["%2e", "%2f", "%5c", "%00"]
        .iter()
        .any(|e| lower.contains(e))
    {
        Some("encoded dots, slashes, backslashes and NUL characters are refused")
    } else {
        None
    }
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut k = 0;
    while k < bytes.len() {
        if bytes[k] == b'%' {
            let hex = s.get(k + 1..k + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            k += 3;
        } else {
            out.push(bytes[k]);
            k += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// The documentation `docs/source` of the repository that holds `root` (the root or one of
/// its parents), if there is one.
pub fn find_docs(root: &Path) -> Option<PathBuf> {
    root.ancestors()
        .find(|d| {
            d.join("docs/source/index.rst").is_file()
                && d.join("crates/mopt-rs/Cargo.toml").is_file()
        })
        .map(|d| d.join("docs/source"))
}

/// The pages of the documentation directory `docs`: `(name, path)`, sorted.
pub fn doc_pages(docs: &Path) -> Vec<(String, PathBuf)> {
    run_files::sorted_entries(docs)
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "md") && p.is_file())
        .filter_map(|p| Some((p.file_stem()?.to_string_lossy().into_owned(), p)))
        .collect()
}

fn title_in(content: &str) -> Option<String> {
    content
        .lines()
        .find_map(|l| l.strip_prefix("# ").map(|t| t.trim().to_string()))
}

fn title_of(path: &Path) -> Option<String> {
    title_in(&std::fs::read_to_string(path).ok()?)
}

/// The run directories under the root (looked for at most `SCAN_DEPTH` levels deep, skipping
/// hidden directories and build output).
pub fn run_directories_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut level = vec![root.to_path_buf()];
    let mut visited = 0;
    for _ in 0..SCAN_DEPTH {
        let mut next = Vec::new();
        for d in &level {
            for child in run_files::sorted_entries(d) {
                let Ok(meta) = std::fs::symlink_metadata(&child) else {
                    continue;
                };
                let name = child
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if !meta.is_dir()
                    || name.starts_with('.')
                    || ["target", "node_modules", "__pycache__"].contains(&name.as_str())
                {
                    continue;
                }
                visited += 1;
                if visited > SCAN_DIRECTORIES {
                    return out;
                }
                if run_files::is_run_directory(&child) {
                    out.push(child);
                } else {
                    next.push(child);
                }
            }
        }
        level = next;
    }
    out
}

/// The resources: the documentation pages, then the files of the run directories.
pub fn list(sandbox: &Sandbox, docs: &Docs) -> Vec<JsonValue> {
    let mut out = Vec::new();
    for (name, size, title) in docs.pages() {
        out.push(object([
            ("uri", text(format!("{DOCS_PREFIX}{name}"))),
            ("name", text(format!("{name}.md"))),
            ("title", text(title)),
            ("description", text("documentation of mopt-rs")),
            ("mimeType", text("text/markdown")),
            ("size", JsonValue::Number(size as f64)),
        ]));
    }
    'runs: for run in run_directories_under(sandbox.root()) {
        let (files, _) = run_files::list_files(&run, LIST_LIMIT);
        for (f, size) in files {
            if out.len() >= LIST_LIMIT {
                break 'runs;
            }
            let Some(rel) = sandbox.relative(&f) else {
                continue;
            };
            out.push(object([
                ("uri", text(root_uri(&rel))),
                ("name", text(&rel)),
                ("mimeType", text(run_files::mime_type(&f))),
                ("size", JsonValue::Number(size as f64)),
            ]));
        }
    }
    out
}

/// The resource templates.
pub fn templates() -> Vec<JsonValue> {
    vec![
        object([
            ("uriTemplate", text(format!("{ROOT_PREFIX}{{+path}}"))),
            ("name", text("file under the root")),
            (
                "description",
                text("Any file or directory under the root of the server, by its path relative to the root."),
            ),
        ]),
        object([
            ("uriTemplate", text(format!("{DOCS_PREFIX}{{name}}"))),
            ("name", text("documentation page")),
            ("description", text("A page of the documentation of mopt-rs, by its file name without .md.")),
            ("mimeType", text("text/markdown")),
        ]),
        object([
            ("uriTemplate", text(format!("{JOBS_PREFIX}{{job_id}}"))),
            ("name", text("status of a job")),
            ("description", text("The status of a job as job_status gives it, by its job_id; subscribe to it to hear when the job changes.")),
            ("mimeType", text("application/json")),
        ]),
    ]
}

/// The path relative to the root of the URI `mopt://root/...` (decoded once and checked by
/// its text), or why it is refused.
fn root_path(uri: &str) -> Result<String, String> {
    let encoded = uri
        .strip_prefix(ROOT_PREFIX)
        .ok_or_else(|| format!("{uri}: not a file under the root"))?;
    if let Some(why) = refused_encoding(encoded) {
        return Err(format!("{uri}: {why}"));
    }
    let rel = percent_decode(encoded).ok_or_else(|| format!("{uri}: not a valid URI"))?;
    let rel = if rel.is_empty() { ".".to_string() } else { rel };
    if let Some(why) = super::paths::refused_path_text(&rel) {
        return Err(why);
    }
    Ok(rel)
}

/// Whether the URI `uri` can be subscribed to: a page of `docs`, or a path under the root that
/// is checked as `resources/read` checks it (it may not exist yet; when it exists it must lie
/// inside the root). Job URIs are checked by the server.
pub fn check_watch(sandbox: &Sandbox, docs: &Docs, uri: &str) -> Result<(), String> {
    if let Some(name) = uri.strip_prefix(DOCS_PREFIX) {
        return docs
            .text(name.trim_end_matches(".md"))
            .map(|_| ())
            .ok_or_else(|| format!("no documentation page '{name}'"));
    }
    let rel = root_path(uri)?;
    if let Ok(resolved) = std::fs::canonicalize(sandbox.root().join(&rel)) {
        if !sandbox.contains(&resolved) {
            return Err(format!("'{rel}' is outside the root of the server"));
        }
    }
    Ok(())
}

/// What tells that the resource `uri` (checked by [`check_watch`]) changed: the kind, size,
/// modification time and (for a directory) the number of entries of what it resolves to now.
/// A path that resolves outside the root now has a fixed fingerprint, so that nothing about
/// what lies outside is told.
pub fn fingerprint(sandbox: &Sandbox, uri: &str) -> String {
    if uri.starts_with(DOCS_PREFIX) {
        return "page".to_string();
    }
    let Ok(rel) = root_path(uri) else {
        return "refused".to_string();
    };
    match std::fs::canonicalize(sandbox.root().join(&rel)) {
        Ok(resolved) if sandbox.contains(&resolved) => match std::fs::metadata(&resolved) {
            Ok(m) => {
                let modified = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_nanos());
                let entries = if m.is_dir() {
                    std::fs::read_dir(&resolved).map_or(0, |r| r.take(100_000).count())
                } else {
                    0
                };
                format!("{}:{}:{modified}:{entries}", m.is_dir(), m.len())
            }
            Err(_) => "absent".to_string(),
        },
        Ok(_) => "outside".to_string(),
        Err(_) => "absent".to_string(),
    }
}

/// The contents of the resource `uri`.
pub fn read(sandbox: &Sandbox, docs: &Docs, uri: &str) -> Result<Vec<JsonValue>, ResourceError> {
    if let Some(name) = uri.strip_prefix(DOCS_PREFIX) {
        if *docs == Docs::Missing {
            return Err(ResourceError::NotFound(format!(
                "{uri}: this server has no documentation (the mopt binary embeds it; the \
                 Python package ships it: python -m mopt_rs.docs)"
            )));
        }
        let name = name.trim_end_matches(".md");
        let content = docs
            .text(name)
            .ok_or_else(|| ResourceError::NotFound(format!("no documentation page '{name}'")))?;
        return Ok(vec![object([
            ("uri", text(uri)),
            ("mimeType", text("text/markdown")),
            ("text", text(content)),
        ])]);
    }
    let Some(encoded) = uri.strip_prefix(ROOT_PREFIX) else {
        return Err(ResourceError::NotFound(format!(
            "{uri}: resources are {DOCS_PREFIX}NAME and {ROOT_PREFIX}PATH"
        )));
    };
    if let Some(why) = refused_encoding(encoded) {
        return Err(ResourceError::NotFound(format!("{uri}: {why}")));
    }
    // Decoded once; the result is checked as any path of a client.
    let rel = percent_decode(encoded)
        .ok_or_else(|| ResourceError::NotFound(format!("{uri}: not a valid URI")))?;
    let rel = if rel.is_empty() { ".".to_string() } else { rel };
    let path = sandbox.existing(&rel).map_err(ResourceError::NotFound)?;
    let meta = std::fs::metadata(&path).map_err(|e| ResourceError::NotFound(e.to_string()))?;
    if meta.is_dir() {
        let entries: Vec<JsonValue> = run_files::sorted_entries(&path)
            .iter()
            .filter_map(|e| {
                let m = std::fs::symlink_metadata(e).ok()?;
                let r = sandbox.relative(e)?;
                Some(object([
                    ("path", text(&r)),
                    ("uri", text(root_uri(&r))),
                    ("directory", JsonValue::Bool(m.is_dir())),
                    ("size", JsonValue::Number(m.len() as f64)),
                ]))
            })
            .collect();
        return Ok(vec![object([
            ("uri", text(uri)),
            ("mimeType", text("application/json")),
            (
                "text",
                text(object([("entries", JsonValue::Array(entries))]).to_string_pretty()),
            ),
        ])]);
    }
    if !meta.is_file() {
        return Err(ResourceError::NotFound(format!(
            "{rel} is not a regular file"
        )));
    }
    if meta.len() > READ_LIMIT {
        return Err(ResourceError::Other(
            RpcError::new(
                INVALID_PARAMS,
                format!(
                    "{rel} has {} bytes, more than the {READ_LIMIT} bytes read through MCP",
                    meta.len()
                ),
            )
            .with_data(object([
                ("uri", text(uri)),
                ("size", count(meta.len() as usize)),
            ])),
        ));
    }
    let (file, _) = sandbox
        .open_regular(&path)
        .map_err(ResourceError::NotFound)?;
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut std::io::Read::take(file, READ_LIMIT + 1), &mut bytes)
        .map_err(|e| ResourceError::NotFound(e.to_string()))?;
    if bytes.len() as u64 > READ_LIMIT {
        return Err(ResourceError::Other(RpcError::new(
            INVALID_PARAMS,
            format!("{rel} grew beyond the {READ_LIMIT} bytes read through MCP"),
        )));
    }
    let mime = run_files::mime_type(&path);
    let binary = (mime.starts_with("image/") && mime != "image/svg+xml")
        || mime == "application/pdf"
        || mime == "application/vnd.sqlite3";
    match (binary, String::from_utf8(bytes)) {
        (false, Ok(content)) => {
            let mime = if mime == "application/octet-stream" {
                "text/plain"
            } else {
                mime
            };
            Ok(vec![object([
                ("uri", text(uri)),
                ("mimeType", text(mime)),
                ("text", text(content)),
            ])])
        }
        (_, Ok(content)) => Ok(vec![object([
            ("uri", text(uri)),
            ("mimeType", text(mime)),
            ("blob", text(super::base64::encode(content.as_bytes()))),
        ])]),
        (_, Err(e)) => Ok(vec![object([
            ("uri", text(uri)),
            ("mimeType", text(mime)),
            ("blob", text(super::base64::encode(e.as_bytes()))),
        ])]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uris_round_trip() {
        let uri = root_uri("mcp_jobs/a b/x%.xyz");
        assert_eq!(uri, "mopt://root/mcp_jobs/a%20b/x%25.xyz");
        assert_eq!(
            percent_decode(uri.strip_prefix(ROOT_PREFIX).unwrap()).as_deref(),
            Some("mcp_jobs/a b/x%.xyz")
        );
        assert!(percent_decode("%zz").is_none());
    }
}
