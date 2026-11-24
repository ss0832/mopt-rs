//! Trajectory files that grow by one frame at a time, and the two trajectories of a pair of
//! structures that become one at the end of a run.
//!
//! A run appends the structure of every iteration to its trajectory, so that the structures
//! are on disk while it runs and after it was interrupted. A search with two structures
//! (two images, two ends of a path, two particles) keeps one such file per structure while it
//! runs; at its end the two become one trajectory that leads from the first structure to the
//! second: the frames of the first in their order, then those of the second backwards.
//!
//! MultiOptPy: `fileio.py:FileIO` (`make_psi4_input_file` wrote a directory
//! `samples_<stem>_<k>/` per iteration, and `make_traj_file`, `make_traj_file_for_DM` read
//! those directories at the end; these files take the place of the directories).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::error::{MoptError, MoptResult};

/// A trajectory file that grows by one frame at a time.
#[derive(Debug)]
pub struct GrowingTrajectory {
    path: PathBuf,
    frames: usize,
}

impl GrowingTrajectory {
    /// An empty file at `path` (what was there is replaced: a run that is started again
    /// writes its trajectory anew).
    pub fn start(path: impl Into<PathBuf>) -> MoptResult<Self> {
        let path = path.into();
        std::fs::write(&path, "")?;
        Ok(Self { path, frames: 0 })
    }

    /// Append one frame (its text as it stands in the file, with the line of the number of
    /// atoms and the comment).
    pub fn push(&mut self, frame: &str) -> MoptResult<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(frame.as_bytes())?;
        self.frames += 1;
        Ok(())
    }

    /// The file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Frames written so far.
    pub fn frames(&self) -> usize {
        self.frames
    }
}

/// The frames of the text of an XYZ trajectory: the comment and the lines of the atoms (with
/// their line ends) of each.
pub fn frames_of(text: &str) -> MoptResult<Vec<(String, String)>> {
    let mut lines = text.lines();
    let mut frames = Vec::new();
    while let Some(first) = lines.next() {
        if first.trim().is_empty() {
            continue;
        }
        let atoms: usize = first.trim().parse().map_err(|_| {
            MoptError::parse(format!(
                "a trajectory frame starts with '{first}' instead of the number of its atoms"
            ))
        })?;
        let comment = lines
            .next()
            .ok_or_else(|| MoptError::parse("a trajectory frame ends after its first line"))?;
        let mut body = String::new();
        for _ in 0..atoms {
            let line = lines
                .next()
                .ok_or_else(|| MoptError::parse("a trajectory frame has fewer lines than atoms"))?;
            let _ = writeln!(body, "{line}");
        }
        frames.push((comment.to_string(), body));
    }
    Ok(frames)
}

/// The trajectories of the two structures of a search while it runs.
#[derive(Debug)]
pub struct PairTrajectories {
    parts: [GrowingTrajectory; 2],
}

impl PairTrajectories {
    /// Two empty files `names[0]` and `names[1]` in `dir`.
    pub fn start(dir: &Path, names: [&str; 2]) -> MoptResult<Self> {
        Ok(Self {
            parts: [
                GrowingTrajectory::start(dir.join(names[0]))?,
                GrowingTrajectory::start(dir.join(names[1]))?,
            ],
        })
    }

    /// Append a frame to the trajectory of structure `which` (0 or 1).
    pub fn push(&mut self, which: usize, frame: &str) -> MoptResult<()> {
        self.parts[which.min(1)].push(frame)
    }

    /// The file of structure `which`.
    pub fn part(&self, which: usize) -> &Path {
        self.parts[which.min(1)].path()
    }

    /// Frames written for each structure.
    pub fn frames(&self) -> [usize; 2] {
        [self.parts[0].frames(), self.parts[1].frames()]
    }

    /// Remove the two files (a run that writes its path as one file of its own at its end).
    pub fn remove(self) -> MoptResult<()> {
        for part in &self.parts {
            std::fs::remove_file(part.path())?;
        }
        Ok(())
    }

    /// The one trajectory of the pair at `path`: the frames of the first structure in their
    /// order, then those of the second backwards; `inner_only` leaves the first and the last
    /// of them out (MultiOptPy's `make_traj_file_for_DM`), and `comment(m, old)` gives the
    /// comment of frame `m` from the one it had. The two files of the run are removed.
    /// Returns the number of frames written.
    pub fn join(
        self,
        path: &Path,
        inner_only: bool,
        comment: &dyn Fn(usize, &str) -> String,
    ) -> MoptResult<usize> {
        let first = frames_of(&std::fs::read_to_string(self.parts[0].path())?)?;
        let second = frames_of(&std::fs::read_to_string(self.parts[1].path())?)?;
        let all: Vec<&(String, String)> = first.iter().chain(second.iter().rev()).collect();
        let kept: &[&(String, String)] = match (inner_only, all.len()) {
            (false, _) => &all,
            (true, n) if n > 2 => &all[1..n - 1],
            (true, _) => &[],
        };
        let mut text = String::new();
        for (m, (old, body)) in kept.iter().enumerate() {
            let atoms = body.lines().count();
            let _ = write!(text, "{atoms}\n{}\n{body}", comment(m, old));
        }
        std::fs::write(path, text)?;
        for part in &self.parts {
            if part.path() != path {
                std::fs::remove_file(part.path())?;
            }
        }
        Ok(kept.len())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn frame(label: &str, x: f64) -> String {
        format!("2\n{label}\nH 0.0 0.0 0.0\nH {x} 0.0 0.0\n")
    }

    #[test]
    fn a_trajectory_grows_frame_by_frame_and_starts_anew() {
        let dir = std::env::temp_dir().join(format!("mopt_traj_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a_traj.xyz");
        let mut t = GrowingTrajectory::start(&path).unwrap();
        t.push(&frame("Frame 0", 0.7)).unwrap();
        // The frame is in the file before the next one is written.
        assert_eq!(
            frames_of(&std::fs::read_to_string(&path).unwrap())
                .unwrap()
                .len(),
            1
        );
        t.push(&frame("Frame 1", 0.8)).unwrap();
        assert_eq!(t.frames(), 2);
        let frames = frames_of(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(frames[1].0, "Frame 1");
        assert_eq!(frames[1].1, "H 0.0 0.0 0.0\nH 0.8 0.0 0.0\n");
        let t = GrowingTrajectory::start(&path).unwrap();
        assert_eq!(t.frames(), 0);
        assert!(std::fs::read_to_string(&path).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_two_trajectories_of_a_pair_become_one() {
        let dir = std::env::temp_dir().join(format!("mopt_pair_traj_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fill = |dir: &Path| {
            let mut pair = PairTrajectories::start(dir, ["p_1.xyz", "p_2.xyz"]).unwrap();
            for k in 0..3 {
                pair.push(0, &frame(&format!("first {k}"), 1.0 + k as f64))
                    .unwrap();
                pair.push(1, &frame(&format!("second {k}"), 9.0 - k as f64))
                    .unwrap();
            }
            assert_eq!(pair.frames(), [3, 3]);
            assert!(pair.part(0).is_file() && pair.part(1).is_file());
            pair
        };
        let joined = dir.join("p_traj.xyz");
        let n = fill(&dir)
            .join(&joined, false, &|m, old| format!("Frame {m} ({old})"))
            .unwrap();
        assert_eq!(n, 6);
        let frames = frames_of(&std::fs::read_to_string(&joined).unwrap()).unwrap();
        let comments: Vec<&str> = frames.iter().map(|f| f.0.as_str()).collect();
        assert_eq!(
            comments,
            [
                "Frame 0 (first 0)",
                "Frame 1 (first 1)",
                "Frame 2 (first 2)",
                "Frame 3 (second 2)",
                "Frame 4 (second 1)",
                "Frame 5 (second 0)"
            ]
        );
        assert!(!dir.join("p_1.xyz").exists() && !dir.join("p_2.xyz").exists());
        // Without the two starting structures, as MultiOptPy writes the trajectory of iEIP.
        let n = fill(&dir)
            .join(&joined, true, &|m, _| format!("Frame {m}"))
            .unwrap();
        assert_eq!(n, 4);
        let frames = frames_of(&std::fs::read_to_string(&joined).unwrap()).unwrap();
        assert_eq!(frames[0].1, "H 0.0 0.0 0.0\nH 2 0.0 0.0\n");
        assert_eq!(frames[3].1, "H 0.0 0.0 0.0\nH 8 0.0 0.0\n");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_broken_frame_is_an_error() {
        assert!(frames_of("2\ncomment\nH 0 0 0\n").is_err());
        assert!(frames_of("x\ncomment\n").is_err());
        assert!(frames_of("\n").unwrap().is_empty());
    }
}
