//! A calculator that tells the run log what its evaluations said.
//!
//! The commands that are not driven by an optimizer (`freqmain`: Hessians of a structure,
//! of displaced structures, of the points of a path, of a supercell) evaluate their
//! calculator in many places. Wrapped in [`Narrated`], every evaluation writes the
//! warnings of the calculator, `<S**2>`, and the output of the program itself
//! ([`crate::pes::calculator::said`]) to the log as it happens, numbered in the order of
//! the evaluations.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core_infra::log::Logger;
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Orbitals};
use crate::error::MoptResult;
use crate::pes::calculator::{said, Calculator};

/// A calculator whose evaluations are reported to a log.
pub struct Narrated<'a> {
    inner: &'a dyn Calculator,
    log: Logger,
    what: String,
    count: AtomicUsize,
}

impl<'a> Narrated<'a> {
    /// `inner`, with what its evaluations say written to `log`; the lines name the
    /// evaluations `what 1`, `what 2`, ...
    pub fn new(inner: &'a dyn Calculator, log: Logger, what: &str) -> Self {
        Self {
            inner,
            log,
            what: what.to_string(),
            count: AtomicUsize::new(0),
        }
    }

    /// The number of evaluations so far.
    pub fn evaluations(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

impl Calculator for Narrated<'_> {
    fn analytic_hessian(&self) -> bool {
        self.inner.analytic_hessian()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> String {
        self.inner.description()
    }

    fn citations(&self) -> Vec<String> {
        self.inner.citations()
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let number = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        let out = self.inner.evaluate(sys, req);
        // The warnings of an evaluation that failed are told as well.
        let of = format!("{} {number}", self.what);
        match &out {
            Ok(out) => {
                for line in said(self.inner, &of, out) {
                    self.log.info(&line);
                }
            }
            Err(_) => {
                for warning in self.inner.take_warnings() {
                    self.log.info(&format!("Warning: {of}: {warning}"));
                }
            }
        }
        out
    }

    fn supports_pbc(&self) -> bool {
        self.inner.supports_pbc()
    }

    fn is_rigid_body_invariant(&self) -> bool {
        self.inner.is_rigid_body_invariant()
    }

    fn evaluates_in_order(&self) -> bool {
        self.inner.evaluates_in_order()
    }

    fn state_characters(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<crate::pes::states::StateCharacters>> {
        self.inner.state_characters(sys, count)
    }

    fn transitions(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<Vec<crate::pes::states::Transition>>> {
        self.inner.transitions(sys, count)
    }

    fn followed_state(&self, sys: &MolecularSystem, label: usize) -> Option<usize> {
        self.inner.followed_state(sys, label)
    }

    fn recall_state(&self, sys: &MolecularSystem, label: usize, root: usize) {
        self.inner.recall_state(sys, label, root);
    }

    fn relaxed_environment(&self, sys: &MolecularSystem) -> Option<Vec<f64>> {
        self.inner.relaxed_environment(sys)
    }

    fn recall_environment(&self, sys: &MolecularSystem, environment: &[f64]) {
        self.inner.recall_environment(sys, environment);
    }

    fn supports_excited_states(&self) -> bool {
        self.inner.supports_excited_states()
    }

    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        self.inner.excitation_energies(sys)
    }

    fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Option<Orbitals>> {
        self.inner.orbitals(sys)
    }

    fn uses_electrons(&self) -> bool {
        self.inner.uses_electrons()
    }

    fn take_warnings(&self) -> Vec<String> {
        self.inner.take_warnings()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::core_infra::log::{Level, MemorySink};
    use crate::domain::{OrbitalData, SpinCharge};
    use crate::io::xyz::parse_xyz_str;
    use crate::pes::native::LennardJones;

    /// A calculator with a warning, a spin, and an output of its own.
    struct Talking;

    impl Calculator for Talking {
        fn name(&self) -> &str {
            "talking"
        }

        fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
            let mut out = LennardJones.evaluate(sys, req)?;
            out.orbitals = Some(OrbitalData {
                spin_square: Some(2.0031),
                program_output: Some("cycle 1\nconverged\n".to_string()),
                ..OrbitalData::default()
            });
            Ok(out)
        }

        fn take_warnings(&self) -> Vec<String> {
            vec!["the wavefunction was started anew".to_string()]
        }
    }

    #[test]
    fn every_evaluation_is_told_to_the_log_in_its_order() {
        let sys = parse_xyz_str(
            "2\n\nAr 0.0 0.0 0.0\nAr 3.9 0.0 0.0\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let sink = MemorySink::new();
        let log = Logger::silent();
        log.add_sink(Level::Info, Box::new(sink.clone()));
        let calc = Narrated::new(&Talking, log, "evaluation");
        let plain = LennardJones
            .evaluate(&sys, &EvalRequest::ENERGY_GRAD)
            .unwrap();
        for _ in 0..2 {
            let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
            // The values are those of the calculator.
            assert_eq!(out.energy.value(), plain.energy.value());
        }
        assert_eq!(calc.evaluations(), 2);
        let lines: Vec<String> = sink.lines().into_iter().map(|(_, line)| line).collect();
        assert_eq!(
            lines,
            [
                "Warning: evaluation 1: the wavefunction was started anew",
                "evaluation 1: <S**2> = 2.003100",
                "evaluation 1: output of the program",
                "cycle 1",
                "converged",
                "Warning: evaluation 2: the wavefunction was started anew",
                "evaluation 2: <S**2> = 2.003100",
                "evaluation 2: output of the program",
                "cycle 1",
                "converged",
            ]
        );
        assert_eq!(calc.name(), "talking");
    }
}
