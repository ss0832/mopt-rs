//! File names with wildcards, as MultiOptPy takes them through Python's `glob`: `*` (any
//! characters), `?` (one character), `[abc]`, `[a-z]`, `[!abc]` (one character of, or not
//! of, a set), within one component of a path.
//!
//! MultiOptPy: `optimization.py:Optimize.run` (`glob.glob(job_file)`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Component, Path, PathBuf};

use crate::error::MoptResult;

/// Whether `text` has a wildcard.
pub fn has_wildcard(text: &str) -> bool {
    text.contains(['*', '?', '['])
}

/// The set that opens at the start of `rest` (after `[`): whether it is negated, its
/// members as ranges, and what follows the closing bracket; `None` if it is never closed.
#[allow(clippy::type_complexity)]
fn set(rest: &[char]) -> Option<(bool, Vec<(char, char)>, &[char])> {
    let negated = rest.first() == Some(&'!');
    let body = if negated { &rest[1..] } else { rest };
    // A closing bracket right at the start is a member.
    let close = body.iter().skip(1).position(|c| *c == ']')? + 1;
    let (members, after) = (&body[..close], &body[close + 1..]);
    let mut ranges = Vec::new();
    let mut k = 0;
    while k < members.len() {
        if k + 2 < members.len() && members[k + 1] == '-' {
            ranges.push((members[k], members[k + 2]));
            k += 3;
        } else {
            ranges.push((members[k], members[k]));
            k += 1;
        }
    }
    Some((negated, ranges, after))
}

/// Whether `name` matches `pattern`.
pub fn matches(pattern: &[char], name: &[char]) -> bool {
    let Some((first, rest)) = pattern.split_first() else {
        return name.is_empty();
    };
    match first {
        '*' => (0..=name.len()).any(|k| matches(rest, &name[k..])),
        '?' => !name.is_empty() && matches(rest, &name[1..]),
        '[' => match (set(rest), name.split_first()) {
            (Some((negated, ranges, after)), Some((c, tail))) => {
                let inside = ranges.iter().any(|(low, high)| low <= c && c <= high);
                inside != negated && matches(after, tail)
            }
            (Some(_), None) => false,
            // Not closed: an ordinary character.
            (None, _) => name.first() == Some(&'[') && matches(rest, &name[1..]),
        },
        c => name.first() == Some(c) && matches(rest, &name[1..]),
    }
}

/// The existing paths that `pattern` stands for, sorted; `pattern` itself if it has no
/// wildcard (whether or not it exists).
pub fn expand(pattern: &str) -> MoptResult<Vec<PathBuf>> {
    if !has_wildcard(pattern) {
        return Ok(vec![PathBuf::from(pattern)]);
    }
    let mut found = vec![PathBuf::new()];
    for component in Path::new(pattern).components() {
        let text = match component {
            Component::Normal(name) => name.to_string_lossy().into_owned(),
            other => {
                for base in &mut found {
                    base.push(other.as_os_str());
                }
                continue;
            }
        };
        if !has_wildcard(&text) {
            for base in &mut found {
                base.push(&text);
            }
            continue;
        }
        let wanted: Vec<char> = text.chars().collect();
        let mut next = Vec::new();
        for base in &found {
            let dir = if base.as_os_str().is_empty() {
                Path::new(".")
            } else {
                base.as_path()
            };
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                // As in a shell, a leading dot is matched only by a dot.
                if name.starts_with('.') && !text.starts_with('.') {
                    continue;
                }
                if matches(&wanted, &name.chars().collect::<Vec<char>>()) {
                    next.push(base.join(&name));
                }
            }
        }
        found = next;
    }
    found.retain(|p| p.exists());
    found.sort();
    Ok(found)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn is_match(pattern: &str, name: &str) -> bool {
        matches(
            &pattern.chars().collect::<Vec<char>>(),
            &name.chars().collect::<Vec<char>>(),
        )
    }

    #[test]
    fn wildcards_match_as_in_a_shell() {
        assert!(is_match("*.xyz", "a.xyz") && is_match("*.xyz", ".xyz"));
        assert!(!is_match("*.xyz", "a.xyz.bak"));
        assert!(is_match("ts_?.xyz", "ts_1.xyz") && !is_match("ts_?.xyz", "ts_12.xyz"));
        assert!(is_match("ts_[0-9]*.xyz", "ts_12.xyz") && !is_match("ts_[0-9]*.xyz", "ts_a.xyz"));
        assert!(is_match("[!a]b", "cb") && !is_match("[!a]b", "ab"));
        assert!(is_match("a[b", "a[b") && is_match("[]x]", "]"));
        assert!(is_match("*a*b*", "xxaxxbxx") && !is_match("*a*b*", "xxbxxaxx"));
    }

    #[test]
    fn patterns_are_expanded_to_the_files_that_exist() {
        let dir = std::env::temp_dir().join(format!("mopt_glob_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        for name in ["a1.xyz", "a2.xyz", "b.xyz", ".hidden.xyz", "sub/c.xyz"] {
            std::fs::write(dir.join(name), "").unwrap();
        }
        let root = dir.to_string_lossy().into_owned();
        let names = |pattern: &str| -> Vec<String> {
            expand(&format!("{root}/{pattern}"))
                .unwrap()
                .iter()
                .map(|p| {
                    p.strip_prefix(&dir)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/")
                })
                .collect()
        };
        assert_eq!(names("a*.xyz"), ["a1.xyz", "a2.xyz"]);
        assert_eq!(names("?.xyz"), ["b.xyz"]);
        assert_eq!(names("[ab]*.xyz"), ["a1.xyz", "a2.xyz", "b.xyz"]);
        assert_eq!(names("*/c.xyz"), ["sub/c.xyz"]);
        assert!(names("z*.xyz").is_empty());
        // Without a wildcard the name is passed on as it is.
        assert_eq!(
            expand("not_there.xyz").unwrap(),
            [PathBuf::from("not_there.xyz")]
        );
    }
}
