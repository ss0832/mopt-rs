//! Live job reports and the queue snapshot of run_mapper.
//!
//! MultiOptPy: Wrapper/mapper.py, _write_priority_log.
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::core_infra::rng::Rng;
use crate::error::MoptResult;
use crate::workflow::mapper::network::{JobStatus, Network};
use crate::workflow::mapper::observers::{job_files_dir, ExplorationEnd, MapperObserver, Source};
use crate::workflow::mapper::priority::{Candidate, ExplorationPriority};
use crate::workflow::mapper::store::write_whole;
use crate::workflow::mapper::Configuration;

/// Record exactly the candidate pool presented to the priority, including in parallel runs.
pub(super) struct PriorityLog {
    path: PathBuf,
    rule: Box<dyn ExplorationPriority>,
}

impl PriorityLog {
    pub(super) fn new(dir: &Path, rule: Box<dyn ExplorationPriority>) -> Self {
        Self {
            path: dir.join("queue_priority.log"),
            rule,
        }
    }
}

impl ExplorationPriority for PriorityLog {
    fn name(&self) -> &str {
        self.rule.name()
    }
    fn reason(&self) -> Option<String> {
        self.rule.reason()
    }

    fn select(
        &mut self,
        network: &Network,
        candidates: &[Candidate],
        job: usize,
        rng: &mut Rng,
    ) -> MoptResult<Option<usize>> {
        let pick = self.rule.select(network, candidates, job, rng)?;
        let scores = self.rule.last_scores();
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        if scores.len() == order.len() {
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        }
        let queued: usize = candidates.iter().map(|c| c.untried).sum();
        let reference = candidates
            .iter()
            .map(|c| c.energy)
            .fold(f64::INFINITY, f64::min);
        let mut log = format!("# Iter {job:06} | queued={queued} | finished_jobs={} | ref_energy={reference:.12} Ha | rule={}\n", network.jobs.len(), self.name());
        if let Some(reason) = self.reason() {
            let _ = writeln!(log, "# {reason}");
        }
        for i in order {
            let c = &candidates[i];
            let node = network.node(c.node)?;
            let value = scores
                .get(i)
                .map_or_else(|| "n/a".to_string(), |s| format!("{s:.12e}"));
            let _ = writeln!(log, "Node: {}{:06} | Priority: {value} | pending_pairs={} | visits={} | energy={:.12} Ha | selected={}",
                node.kind.tag(), node.number, c.untried, c.visits, c.energy, pick == Some(i));
        }
        if let Some(csv) = self.rule.contraction() {
            write_whole(
                &self.path.with_file_name("rcmc_K_contracted.csv"),
                csv.as_bytes(),
            )?;
        }
        write_whole(&self.path, log.as_bytes())?;
        Ok(pick)
    }
}

struct Running {
    task: String,
    csv: File,
    frames: usize,
    next_report: Instant,
}

impl Running {
    fn report(
        &mut self,
        dir: &Path,
        id: usize,
        status: &str,
        variables: bool,
        network: Option<&Network>,
    ) -> MoptResult<()> {
        let start = Instant::now();
        super::mapper_job_report::write(dir, id, &self.task, status, variables, network)?;
        self.next_report = Instant::now() + Duration::from_secs(30).max(start.elapsed() * 20);
        Ok(())
    }
}

/// The ordered frame table and the live report of each job.
pub(super) struct JobProgress {
    dir: PathBuf,
    variables: bool,
    running: BTreeMap<usize, Running>,
    pages: Vec<usize>,
}

impl JobProgress {
    pub(super) fn new(dir: &Path, variables: bool) -> Self {
        Self {
            dir: dir.to_path_buf(),
            variables,
            running: BTreeMap::new(),
            pages: Vec::new(),
        }
    }
}

impl MapperObserver for JobProgress {
    fn after_inputs(&mut self, network: &Network) -> MoptResult<()> {
        self.pages = network
            .jobs
            .iter()
            .map(|j| j.id)
            .filter(|&id| {
                self.dir
                    .join(job_files_dir(id))
                    .join("job_report.html")
                    .is_file()
            })
            .collect();
        super::mapper_job_report::index(&self.dir, &self.pages)
    }

    fn on_start(&mut self, source: &Source<'_>) -> MoptResult<()> {
        let Source::Job(job) = source else {
            return Ok(());
        };
        let dir = self.dir.join(job_files_dir(job.id));
        std::fs::create_dir_all(&dir)?;
        let mut csv = File::create(dir.join("steps.csv"))?;
        writeln!(csv, "frame,label,energy_hartree")?;
        let mut run = Running {
            task: job.task.description.clone(),
            csv,
            frames: 0,
            next_report: Instant::now(),
        };
        run.report(&dir, job.id, "Started", self.variables, None)?;
        self.running.insert(job.id, run);
        if !self.pages.contains(&job.id) {
            self.pages.push(job.id);
        }
        super::mapper_job_report::index(&self.dir, &self.pages)
    }

    fn on_frame(
        &mut self,
        source: &Source<'_>,
        _system: &Configuration,
        energy: f64,
        label: &str,
    ) -> MoptResult<()> {
        let Source::Job(job) = source else {
            return Ok(());
        };
        if let Some(run) = self.running.get_mut(&job.id) {
            run.frames += 1;
            writeln!(
                run.csv,
                "{},\"{}\",{energy:.12}",
                run.frames,
                label.replace('"', "\"\"")
            )?;
            if run.frames == 1 || Instant::now() >= run.next_report {
                run.report(
                    &self.dir.join(job_files_dir(job.id)),
                    job.id,
                    "In progress",
                    self.variables,
                    None,
                )?;
            }
        }
        Ok(())
    }

    fn on_finish(&mut self, _network: &Network, _end: &ExplorationEnd) -> MoptResult<()> {
        for (id, mut run) in std::mem::take(&mut self.running) {
            run.report(
                &self.dir.join(job_files_dir(id)),
                id,
                "Interrupted",
                self.variables,
                None,
            )?;
        }
        Ok(())
    }

    fn after_batch(&mut self, network: &Network, jobs: &[usize]) -> MoptResult<()> {
        for &id in jobs {
            let Some(mut run) = self.running.remove(&id) else {
                continue;
            };
            let status = network
                .jobs
                .iter()
                .rev()
                .find(|j| j.id == id)
                .map(|j| &j.status);
            let message = match status {
                Some(JobStatus::Failed(reason)) => format!("Failed: {reason}"),
                _ => "Finished".to_string(),
            };
            run.report(
                &self.dir.join(job_files_dir(id)),
                id,
                &message,
                self.variables,
                Some(network),
            )?;
        }
        Ok(())
    }
}
