//! The conformer search of a transition state for the mapper (`ts_conformers=on`): the search
//! of `confsearch -tsconf` ([`crate::workflow::confsearch::ts`]) with the calculator of the
//! job and the settings of the block `ts_conformer_settings` of the settings file (those of
//! `confsearch`). The saddle points it gives are handed back; the mapper follows each as a
//! transition-state guess, so the check of the reaction by the IRC inside the search is off.
//!
//! The search stops with the run (a `STOP` file in the run directory) and at the time limit of
//! its job: its stop signal asks that of the job. Its IRC files, where it writes any, go to the
//! directory of the job.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::CommandJobs;
use crate::cli::confsearch_jobs::{CommandJobs as SearchJobsOfCommand, JobAtoms, SharedStop};
use crate::core_infra::log::Logger;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::pes::calculator::Calculator;
use crate::workflow::confsearch::moves::MoveRegistry;
use crate::workflow::confsearch::ts;
use crate::workflow::confsearch::{
    CommonSettings, SearchEvent, SearchJobs, SearchObserver, SearchResult,
};
use crate::workflow::mapper::observers::job_files_dir;
use crate::workflow::mapper::{Configuration, Stationary, TsConformerSearch};

/// The notes of the search in the log of the job.
struct Notes<'a> {
    log: &'a Logger,
}

impl SearchObserver for Notes<'_> {
    fn on_event(&mut self, event: &SearchEvent<'_>) -> MoptResult<()> {
        if let SearchEvent::Note(text) = event {
            self.log.info(text);
        }
        Ok(())
    }
}

impl CommandJobs<'_> {
    /// [`crate::workflow::mapper::MapperJobs::ts_conformers`] of a structure with `calc`.
    pub(super) fn ts_conformers_with(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        job: usize,
        label: &str,
    ) -> MoptResult<TsConformerSearch> {
        let Some(config) = self.run_steps.ts_conformers.as_deref() else {
            return Err(MoptError::config(format!(
                "{label}: the run has no settings of the conformer search (ts_conformers=on)"
            )));
        };
        let Some(settings) = config.ts else {
            return Err(MoptError::config(format!(
                "{label}: the settings of the conformer search are not those of a transition \
                 state"
            )));
        };
        let dir = self.stop.dir().join(job_files_dir(job));
        if self.steps().save_job_steps {
            std::fs::create_dir_all(&dir).map_err(|e| MoptError::from(e).context(dir.display()))?;
        }
        // The search stops with the run and at the time limit of the job.
        let outer = self.stop.clone();
        let stop = SharedStop::following(&dir, Logger::silent(), move || outer.check());
        let jobs = SearchJobsOfCommand::new(
            config,
            JobAtoms {
                constraints: Vec::new(),
                terms: Vec::new(),
                fragments: Vec::new(),
                ccqn_bonds: Vec::new(),
                ccqn_product: None,
            },
            calc,
            &dir,
            self.log.clone(),
            stop.clone(),
        )?
        .with_irc_files(self.steps().save_job_steps);
        // The minima of the search are constrained minima (their Hessian has the imaginary
        // mode of the reaction): they are not verified as minima.
        let common = CommonSettings {
            verify: false,
            ..config.common.clone()
        };
        let moves = MoveRegistry::builtin();
        let mut strategy = |s: &MolecularSystem,
                            j: &dyn SearchJobs,
                            obs: &mut dyn SearchObserver|
         -> MoptResult<Option<SearchResult>> {
            crate::cli::confsearch::search_with(config, &common, &moves, &self.log, s, j, obs)
        };
        self.log.info(&format!(
            "{label}: the search of confsearch -tsconf, the saddle direction held."
        ));
        let mut notes = Notes { log: &self.log };
        let result = ts::run(
            start,
            &settings,
            &config.common,
            &jobs,
            &mut notes,
            &mut strategy,
        );
        if stop.stopped() || self.stop.check() {
            return Ok(TsConformerSearch {
                interrupted: true,
                ..TsConformerSearch::default()
            });
        }
        // What the search refuses about its start (a structure that is no saddle point of
        // order 1 on closer inspection) is a warning of the job, not its end.
        let result = match result {
            Ok(Some(r)) if !r.constrained.interrupted => r,
            Ok(_) => {
                return Ok(TsConformerSearch {
                    interrupted: true,
                    ..TsConformerSearch::default()
                })
            }
            Err(e) => {
                return Ok(TsConformerSearch {
                    warnings: vec![format!("{label}: the search did not run ({e})")],
                    ..TsConformerSearch::default()
                })
            }
        };
        // The saddle points other than the input (the first of the ensemble), lowest first.
        let members = result.ensemble.members();
        let conformers: Vec<Stationary> = result
            .ensemble
            .by_energy()
            .into_iter()
            .filter(|&i| i != 0)
            .filter_map(|i| members.get(i))
            .map(|m| Stationary {
                index_wavenumbers: None,
                system: Configuration::Structure(m.system.clone()),
                energy: m.energy,
                wavenumbers: None,
                dissociation: None,
            })
            .collect();
        self.log.info(&format!(
            "{label}: {} saddle point(s) from {} constrained minima; {} other conformer(s).",
            result.candidates.len(),
            result.constrained.ensemble.members().len(),
            conformers.len()
        ));
        Ok(TsConformerSearch {
            interrupted: false,
            conformers,
            warnings: Vec::new(),
        })
    }
}
