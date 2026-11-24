//! How the jobs of the mapper are run: how many at the same time (`workers`), how many are
//! chosen together (`batch_size`), how their results are taken (`schedule`), and the time
//! limit of a job (MultiOptPy's `worker_timeout_s`).
//!
//! - `batch` (the default): up to `batch_size` jobs are chosen together from the network as it
//!   is, run with up to `workers` of them at the same time, and told in the order of their
//!   numbers once all of them have ended; then the next batch is chosen. The network follows
//!   from the settings, the seed, and `batch_size` alone: not from the number of workers, nor
//!   from the order in which the jobs end, nor from a restart. `batch_size` is `workers` when
//!   it is not given, so that every worker has a job; give it to have the network of a run
//!   with another number of workers.
//! - `rolling`: a job is chosen as soon as a worker is free, from the network with the results
//!   that have come in by then, and every result is told as it comes (MultiOptPy's mapper with
//!   `--max_workers` above 1). No worker waits for the slowest job of a batch, but the network
//!   depends on the order in which the jobs end, and so on the timing of the run.
//!
//! The calculations of every job are those of [`MapperWorkers::job`]: its own stop signal and
//! log, and its own calculator where one cannot be shared (a calculator that
//! [`crate::pes::Calculator::evaluates_in_order`]).
//!
//! MultiOptPy: `Wrapper/mapper.py:ReactionNetworkMapper` (`n_parallel`, `worker_timeout_s`,
//!  `_run_parallel_rolling`) and `Entrypoints/mapper.py` (`--max_workers`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::time::{Duration, Instant};

use super::method::{Job, MapperJobs};
use crate::error::{MoptError, MoptResult};

/// How the results of jobs that run at the same time are taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScheduleMode {
    /// Jobs chosen together, told in the order of their numbers when all of them have ended.
    #[default]
    Batch,
    /// A job chosen whenever a worker is free, its result told as it comes.
    Rolling,
}

impl ScheduleMode {
    /// The name of the mode in the settings.
    pub fn name(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Rolling => "rolling",
        }
    }

    /// The mode named `name`.
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "batch" => Ok(Self::Batch),
            "rolling" => Ok(Self::Rolling),
            other => Err(MoptError::parse(format!(
                "the schedule of the mapper is batch or rolling, not '{other}'"
            ))),
        }
    }
}

/// How the jobs of a run are run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {
    /// Jobs at the same time (at least 1).
    pub workers: usize,
    /// Jobs chosen together in the mode `batch` (at least 1).
    pub batch: usize,
    /// How the results are taken.
    pub mode: ScheduleMode,
    /// A job that runs longer is stopped and recorded as failed (`None`: no limit).
    pub time_limit: Option<Duration>,
}

impl Default for Schedule {
    /// One job at a time, as MultiOptPy's mapper without `--max_workers`.
    fn default() -> Self {
        Self {
            workers: 1,
            batch: 1,
            mode: ScheduleMode::Batch,
            time_limit: None,
        }
    }
}

impl Schedule {
    /// Refuse settings that cannot run.
    pub fn check(&self) -> MoptResult<()> {
        if self.workers == 0 {
            return Err(MoptError::parse(
                "workers (--max_workers, n_parallel) must be at least 1",
            ));
        }
        if self.batch == 0 {
            return Err(MoptError::parse("batch_size must be at least 1"));
        }
        if self.time_limit.is_some_and(|t| t.is_zero()) {
            return Err(MoptError::parse("worker_timeout_s must be positive"));
        }
        Ok(())
    }

    /// Whether only one job runs at a time and every job is told as soon as it ends (the
    /// loop of one job after the other).
    pub fn one_at_a_time(&self) -> bool {
        self.workers == 1 && (self.batch == 1 || self.mode == ScheduleMode::Rolling)
    }

    /// The jobs chosen together.
    pub fn chosen_together(&self) -> usize {
        match self.mode {
            ScheduleMode::Batch => self.batch,
            ScheduleMode::Rolling => 1,
        }
    }

    /// The end of a job that starts at `start`, if there is a time limit.
    pub fn deadline(&self, start: Instant) -> Option<Instant> {
        self.time_limit.map(|t| start + t)
    }

    /// A line for the log of the run.
    pub fn describe(&self) -> String {
        let limit = match self.time_limit {
            Some(t) => format!("; a job is stopped after {} s", t.as_secs_f64()),
            None => String::new(),
        };
        match (self.mode, self.workers) {
            (ScheduleMode::Batch, 1) if self.batch == 1 => {
                format!("Jobs: one at a time{limit}")
            }
            (ScheduleMode::Batch, w) => format!(
                "Jobs: {w} at a time, in batches of {} chosen together and kept in the order of \
                 their numbers (schedule batch: the network does not depend on the number of \
                 workers){limit}",
                self.batch
            ),
            (ScheduleMode::Rolling, w) => format!(
                "Jobs: {w} at a time, each chosen when a worker is free and kept when it ends \
                 (schedule rolling: the network can depend on the timing of the run){limit}"
            ),
        }
    }
}

/// The calculations of the jobs of a run that may run at the same time.
pub trait MapperWorkers: Sync {
    /// The calculations of the run itself: the inputs, the log of the run, the stop between
    /// jobs, and the evaluations.
    fn run(&self) -> &dyn MapperJobs;

    /// The calculations of `job` (made in the thread that runs it), with its own stop signal
    /// that also stops it at `deadline`, its own log, and a calculator it may use from this
    /// thread while other jobs run.
    fn job(&self, job: &Job, deadline: Option<Instant>) -> MoptResult<Box<dyn MapperJobs + '_>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_are_checked_and_described() {
        let one = Schedule::default();
        assert!(one.check().is_ok() && one.one_at_a_time());
        assert!(one.describe().contains("one at a time"));
        let four = Schedule {
            workers: 4,
            batch: 4,
            ..Schedule::default()
        };
        assert!(!four.one_at_a_time());
        assert_eq!(four.chosen_together(), 4);
        assert!(four.describe().contains("batches of 4"));
        let rolling = Schedule {
            mode: ScheduleMode::Rolling,
            ..four
        };
        assert_eq!(rolling.chosen_together(), 1);
        assert!(rolling.describe().contains("timing"));
        for bad in [
            Schedule {
                workers: 0,
                ..Schedule::default()
            },
            Schedule {
                batch: 0,
                ..Schedule::default()
            },
            Schedule {
                time_limit: Some(Duration::ZERO),
                ..Schedule::default()
            },
        ] {
            assert!(bad.check().is_err());
        }
        assert_eq!(
            ScheduleMode::parse("Rolling").ok(),
            Some(ScheduleMode::Rolling)
        );
        assert!(ScheduleMode::parse("eager").is_err());
    }
}
