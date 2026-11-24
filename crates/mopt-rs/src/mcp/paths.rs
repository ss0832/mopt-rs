//! The root of the server: every path of a request is checked by its text, resolved, and
//! checked again when it is used.
//!
//! A path from a client is refused by its text when it is empty, holds a NUL character, a
//! backslash or a colon, a `..` component, a component that ends in a dot or a space, a device
//! name reserved for devices (`CON`, `NUL`, `COM1`, ...), or starts like a drive (`C:`) or a UNC path
//! (`//server`). The rest is resolved by the operating system (`canonicalize`: symbolic links
//! and junctions as the system follows them) and must lie inside the canonical root. When a
//! file is opened, it is resolved again and the opened file is compared with the checked one
//! ([`Sandbox::open_regular`]), so that a link swapped in between is noticed. Paths shown to the
//! client are relative to the root, with `/` as the separator.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fs::{File, Metadata};
use std::path::{Path, PathBuf};

/// Directory of the jobs under the root.
pub const JOBS_DIR: &str = "mcp_jobs";

/// The reserved device names, which name devices in every directory on some systems.
const DEVICES: [&str; 24] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9", "CONIN$",
    "CONOUT$",
];

/// Why the text `text` cannot be a path of a client (`None`: it can).
pub fn refused_path_text(text: &str) -> Option<String> {
    let why = if text.is_empty() {
        "an empty path"
    } else if text.contains('\0') {
        "a NUL character"
    } else if text.contains('\\') {
        "a backslash (paths use /)"
    } else if text.starts_with("//") {
        "a UNC path"
    } else if text.contains(':') {
        "a colon (a drive, a device, or a stream)"
    } else if text.split('/').any(|c| c == "..") {
        "a '..' component"
    } else if text
        .split('/')
        .any(|c| c.len() > 1 && (c.ends_with('.') || c.ends_with(' ')))
    {
        "a component that ends in a dot or a space"
    } else if text.split('/').any(|c| {
        let stem = c
            .split('.')
            .next()
            .unwrap_or_default()
            .trim_end()
            .to_ascii_uppercase();
        DEVICES.contains(&stem.as_str())
    }) {
        "a device name"
    } else {
        return None;
    };
    Some(format!("'{}' is refused: {why}", text.replace('\0', "\\0")))
}

/// Why the part `text` of an option value cannot be given to a command (`None`: it can): an
/// absolute path, a drive, a UNC path, a home directory, a `..` component, a backslash or a NUL
/// character. A word that is not a path (`rsirfo_bfgs`, `1,2`, `tblite:GFN2-xTB`) passes.
pub fn refused_value_part(text: &str) -> Option<String> {
    let t = text.trim_start_matches('@');
    let b = t.as_bytes();
    let drive = b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':';
    let why = if t.contains('\0') {
        "a NUL character"
    } else if t.contains('\\') {
        "a backslash (paths use /)"
    } else if t.starts_with('/') {
        "an absolute path (paths are relative to the job directory)"
    } else if drive {
        "a drive"
    } else if t == "~" || t.starts_with("~/") {
        "a home directory"
    } else if t.split('/').any(|c| c == "..") {
        "a '..' component"
    } else {
        return None;
    };
    Some(format!("'{}' is refused: {why}", text.replace('\0', "\\0")))
}

/// The parts of an option value that may be paths: the value itself and the pieces between
/// `=`, `,`, `:`, `;` and white space (as in `-calc_opt key=FILE`, a list `a,b`, or a form
/// `hyperspherical:FILE` of an option).
pub fn path_like_parts(value: &str) -> Vec<&str> {
    let mut out = vec![value];
    for piece in value.split(|c: char| matches!(c, '=' | ',' | ':' | ';') || c.is_whitespace()) {
        if piece != value && !piece.is_empty() {
            out.push(piece);
        }
    }
    out
}

/// The directory the server may read and write.
#[derive(Debug, Clone)]
pub struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    /// The sandbox of the directory `root` (it must exist).
    pub fn new(root: &Path) -> Result<Self, String> {
        let root = std::fs::canonicalize(root)
            .map_err(|e| format!("the root {} cannot be resolved: {e}", root.display()))?;
        if !root.is_dir() {
            return Err(format!("the root {} is not a directory", root.display()));
        }
        Ok(Self { root })
    }

    /// The canonical root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether the canonical path `path` is the root or inside it.
    pub fn contains(&self, path: &Path) -> bool {
        path.starts_with(&self.root)
    }

    /// `path` (canonical, inside the root) relative to the root, with `/` separators (empty for
    /// the root itself).
    pub fn relative(&self, path: &Path) -> Option<String> {
        let rest = path.strip_prefix(&self.root).ok()?;
        let parts: Vec<String> = rest
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        Some(parts.join("/"))
    }

    /// The existing file or directory `user` (relative to the root, or absolute inside it),
    /// canonical.
    pub fn existing(&self, user: &str) -> Result<PathBuf, String> {
        if let Some(why) = refused_path_text(user) {
            return Err(why);
        }
        let joined = self.root.join(user);
        let resolved = std::fs::canonicalize(&joined)
            .map_err(|e| format!("'{user}' does not name a file under the root ({e})"))?;
        if !self.contains(&resolved) {
            return Err(format!("'{user}' is outside the root of the server"));
        }
        Ok(resolved)
    }

    /// Check the part `text` of an option value as a program started in the directory `base`
    /// (inside the root) sees it.
    pub fn check_value_part(&self, base: &Path, text: &str) -> Result<(), String> {
        if let Some(why) = refused_value_part(text) {
            return Err(why);
        }
        // An existing relative path must resolve inside the root as well (links).
        if let Ok(resolved) = std::fs::canonicalize(base.join(text.trim_start_matches('@'))) {
            if !self.contains(&resolved) {
                return Err(format!("'{text}' leads outside the root of the server"));
            }
        }
        Ok(())
    }

    /// Open the regular file `path` (a path inside the root, as checked before) for reading:
    /// resolved again now, checked to lie inside the root, opened, and compared with what was
    /// resolved, so that a link swapped in since the check is noticed.
    pub fn open_regular(&self, path: &Path) -> Result<(File, Metadata), String> {
        let shown = self
            .relative(path)
            .unwrap_or_else(|| path.display().to_string());
        let resolved = std::fs::canonicalize(path).map_err(|e| format!("{shown}: {e}"))?;
        if !self.contains(&resolved) {
            return Err(format!("{shown} leads outside the root of the server"));
        }
        // Checked before it is opened: opening a FIFO or a device could block or act.
        if !std::fs::metadata(&resolved).is_ok_and(|m| m.is_file()) {
            return Err(format!("{shown} is not a regular file"));
        }
        let file = File::open(&resolved).map_err(|e| format!("{shown}: {e}"))?;
        let opened = file.metadata().map_err(|e| format!("{shown}: {e}"))?;
        if !opened.is_file() {
            return Err(format!("{shown} is not a regular file"));
        }
        let now = std::fs::canonicalize(path).map_err(|e| format!("{shown}: {e}"))?;
        let at_path = std::fs::metadata(&now).map_err(|e| format!("{shown}: {e}"))?;
        if now != resolved || !same_file(&opened, &at_path) {
            return Err(format!("{shown} changed while it was opened"));
        }
        Ok((file, opened))
    }

    /// The directory of the jobs (created when missing), canonical and inside the root.
    pub fn jobs_dir(&self) -> Result<PathBuf, String> {
        let dir = self.root.join(JOBS_DIR);
        if let Ok(meta) = std::fs::symlink_metadata(&dir) {
            if !meta.is_dir() {
                return Err(format!("{JOBS_DIR} under the root is not a directory"));
            }
        }
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let resolved =
            std::fs::canonicalize(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        if !self.contains(&resolved) || resolved == self.root {
            return Err(format!("{JOBS_DIR} is outside the root of the server"));
        }
        Ok(resolved)
    }
}

#[cfg(unix)]
fn same_file(a: &Metadata, b: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}

#[cfg(not(unix))]
fn same_file(a: &Metadata, b: &Metadata) -> bool {
    a.len() == b.len() && a.modified().ok() == b.modified().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mopt_mcp_paths_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("root/sub")).unwrap();
        std::fs::write(dir.join("root/sub/a.xyz"), "1\n\nH 0 0 0\n").unwrap();
        std::fs::write(dir.join("outside.txt"), "secret").unwrap();
        dir
    }

    #[test]
    fn paths_are_refused_by_their_text() {
        for bad in [
            "",
            "a\0b",
            "..",
            "../x",
            "sub/../a.xyz",
            "a\\b",
            "C:/x",
            "C:x",
            "//server/share",
            "\\\\server\\share",
            "NUL",
            "con.xyz",
            "sub/aux",
            "com1.txt",
            "a.",
            "a ",
            "x:stream",
        ] {
            assert!(refused_path_text(bad).is_some(), "{bad:?}");
        }
        for good in [
            "a.xyz",
            "sub/a.xyz",
            "./a.xyz",
            ".hidden",
            "console.log",
            "nul_test.xyz",
        ] {
            assert!(refused_path_text(good).is_none(), "{good:?}");
        }
        for bad in [
            "/etc/passwd",
            "@/etc/passwd",
            "../x",
            "a/../../x",
            "C:\\x",
            "c:x",
            "~/x",
            "~",
            "a\\b",
            "a\0",
        ] {
            assert!(refused_value_part(bad).is_some(), "{bad:?}");
        }
        for good in [
            "rsirfo_bfgs",
            "1,2",
            "-1.5",
            "expr:(q1-1)^2",
            "1..5",
            "tblite:GFN2-xTB",
            "h.npy",
        ] {
            assert!(refused_value_part(good).is_none(), "{good:?}");
        }
        assert_eq!(
            path_like_parts("param=../x,1"),
            vec!["param=../x,1", "param", "../x", "1"]
        );
        assert!(path_like_parts("hyperspherical:/etc/passwd").contains(&"/etc/passwd"));
        assert!(path_like_parts("a /etc/passwd").contains(&"/etc/passwd"));
    }

    #[test]
    fn paths_stay_inside_the_root() {
        let dir = scratch("inside");
        let s = Sandbox::new(&dir.join("root")).unwrap();
        let a = s.existing("sub/a.xyz").unwrap();
        assert_eq!(s.relative(&a).as_deref(), Some("sub/a.xyz"));
        assert!(s.existing("../outside.txt").is_err());
        assert!(s
            .existing(&dir.join("outside.txt").to_string_lossy())
            .is_err());
        assert!(
            s.existing(&a.to_string_lossy()).is_ok(),
            "absolute inside the root"
        );
        assert!(s.existing("missing.xyz").is_err());
        let base = s.root().join("sub");
        assert!(s.check_value_part(&base, "a.xyz").is_ok());
        assert!(s.check_value_part(&base, "new/b.xyz").is_ok());
        assert!(s.check_value_part(&base, "../../outside.txt").is_err());
        assert!(s.open_regular(&a).is_ok());
        assert!(
            s.open_regular(s.root()).is_err(),
            "a directory is not a regular file"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_links_do_not_lead_out() {
        let dir = scratch("links");
        let s = Sandbox::new(&dir.join("root")).unwrap();
        std::os::unix::fs::symlink(dir.join("outside.txt"), dir.join("root/link.txt")).unwrap();
        std::os::unix::fs::symlink(dir.as_path(), dir.join("root/up")).unwrap();
        assert!(s.existing("link.txt").is_err());
        assert!(s.existing("up/outside.txt").is_err());
        assert!(s.check_value_part(s.root(), "link.txt").is_err());
        assert!(s.open_regular(&dir.join("root/link.txt")).is_err());
        std::os::unix::fs::symlink(dir.join("root/sub/a.xyz"), dir.join("root/inner.xyz")).unwrap();
        let inner = s.existing("inner.xyz").unwrap();
        assert_eq!(s.relative(&inner).as_deref(), Some("sub/a.xyz"));
        // A file checked and then swapped for a link that leads out is not opened.
        let checked = s.existing("sub/a.xyz").unwrap();
        std::fs::remove_file(&checked).unwrap();
        std::os::unix::fs::symlink(dir.join("outside.txt"), &checked).unwrap();
        assert!(s.open_regular(&checked).is_err());
        // The directory of the jobs may not be a link that leads out.
        std::os::unix::fs::symlink(dir.as_path(), dir.join("root").join(JOBS_DIR)).unwrap();
        assert!(s.jobs_dir().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
