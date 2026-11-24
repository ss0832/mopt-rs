//! Terms of the whole system on the surface of the core: a bias potential (AFIR, a
//! restraint) whose atoms are all in the core goes with the microiterations. Its energy does
//! not depend on the environment, so `min_y [E(x, y) + B(x)] = E*(x) + B(x)`: the term is
//! added to the relaxed surface as it is, evaluated at the core in the whole system. A term
//! on an atom of the environment would have to be felt by the relaxation of the environment
//! instead, and is not taken.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::CoreSplit;
use crate::domain::{AtomId, EvalOutput, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::pes::term::{PesTerm, TermContext};

/// A term of whole structures like `template` as a term of the core structures of `split`.
pub struct CoreTerm {
    term: Arc<dyn PesTerm>,
    template: MolecularSystem,
    core: Vec<usize>,
}

impl CoreTerm {
    /// `term` (of whole structures like `template`) on the core of `split`, or why it cannot
    /// be (an atom of the environment, or a term that does not say its atoms).
    pub fn new(
        term: Arc<dyn PesTerm>,
        template: &MolecularSystem,
        split: &CoreSplit,
    ) -> Result<Self, String> {
        let Some(atoms) = term.atoms() else {
            return Err(format!(
                "the bias potential '{}', which may depend on every atom",
                term.name()
            ));
        };
        if let Some(outside) = atoms.iter().find(|a| !split.core.contains(&a.index())) {
            return Err(format!(
                "the bias potential '{}' on atom {} of the environment, which the relaxation \
                 of the environment would have to feel",
                term.name(),
                outside.one_based()
            ));
        }
        Ok(Self {
            term,
            template: template.clone(),
            core: split.core.clone(),
        })
    }

    /// The whole structure with the core at `core` (the environment where it does not
    /// matter: at the start).
    fn whole(&self, core: &MolecularSystem) -> MoptResult<MolecularSystem> {
        let x = core.coords_flat_bohr();
        if x.nrows() != 3 * self.core.len() {
            return Err(MoptError::calc(format!(
                "bias potential '{}' on the core: a structure of {} atoms for a core of {}",
                self.term.name(),
                x.nrows() / 3,
                self.core.len()
            )));
        }
        let mut full = self.template.coords_flat_bohr();
        for (k, &a) in self.core.iter().enumerate() {
            for c in 0..3 {
                full[3 * a + c] = x[3 * k + c];
            }
        }
        self.template.with_flat_coords_bohr(&full)
    }
}

impl PesTerm for CoreTerm {
    fn name(&self) -> &str {
        self.term.name()
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let whole = self.whole(sys)?;
        let ctx = TermContext {
            initial: &self.template,
            ..*ctx
        };
        let mut out = self.term.evaluate(&whole, &ctx, need_hessian)?;
        let n = 3 * self.core.len();
        let index = |k: usize| 3 * self.core[k / 3] + k % 3;
        out.gradient_bohr = Col::from_fn(n, |k| out.gradient_bohr[index(k)]);
        out.hessian_bohr = out
            .hessian_bohr
            .as_ref()
            .map(|h| Mat::from_fn(n, n, |i, j| h[(index(i), index(j))]));
        Ok(out)
    }

    fn commit(&self, sys: &MolecularSystem, ctx: &TermContext<'_>) {
        if let Ok(whole) = self.whole(sys) {
            let ctx = TermContext {
                initial: &self.template,
                ..*ctx
            };
            self.term.commit(&whole, &ctx);
        }
    }

    /// That of the wrapped term: it depends on the core atoms alone, which move with the
    /// structure of the core.
    fn frame(&self) -> crate::pes::term::Frame {
        self.term.frame()
    }

    fn history(&self) -> crate::pes::term::History {
        self.term.history()
    }

    fn fork(&self) -> Option<Arc<dyn PesTerm>> {
        let term = self.term.fork()?;
        Some(Arc::new(Self {
            term,
            template: self.template.clone(),
            core: self.core.clone(),
        }))
    }

    fn analytic_hessian(&self) -> bool {
        self.term.analytic_hessian()
    }

    fn atoms(&self) -> Option<Vec<AtomId>> {
        let atoms = self.term.atoms()?;
        atoms
            .iter()
            .map(|a| {
                self.core
                    .iter()
                    .position(|&c| c == a.index())
                    .map(AtomId::new)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::domain::{Element, SpinCharge};
    use crate::pes::bias::{AdBias, KeepDistance, Ramp};

    fn chain_of_four() -> MolecularSystem {
        MolecularSystem::from_flat_bohr(
            vec![Element::C; 4],
            &Col::from_fn(12, |k| {
                [0.0, 0.3, 0.1, 2.8, -0.2, 0.4, 5.5, 0.6, -0.3, 8.1, 0.1, 0.9][k]
            }),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn a_term_on_the_core_is_that_of_the_whole_system_on_the_rows_of_the_core() {
        let template = chain_of_four();
        let split = CoreSplit {
            core: vec![1, 2],
            environment: vec![0, 3],
            fixed: vec![],
        };
        let keep = |a: usize, b: usize| KeepDistance {
            a: vec![AtomId::new(a)],
            b: vec![AtomId::new(b)],
            k: Ramp::constant(0.3),
            r0: Ramp::constant(2.0),
        };
        let bias = keep(1, 2);
        let whole_term: Arc<dyn PesTerm> = Arc::new(AdBias::new(bias));
        let on_core = CoreTerm::new(Arc::clone(&whole_term), &template, &split).unwrap();
        let core = MolecularSystem::from_flat_bohr(
            vec![Element::C; 2],
            &Col::from_fn(6, |k| [2.9, -0.1, 0.5, 5.2, 0.7, -0.4][k]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let ctx = TermContext::at_iteration(0, &core);
        let got = on_core.evaluate(&core, &ctx, true).unwrap();
        let whole = on_core.whole(&core).unwrap();
        let want = whole_term
            .evaluate(&whole, &TermContext::at_iteration(0, &template), true)
            .unwrap();
        assert!((got.energy.value() - want.energy.value()).abs() < 1e-14);
        for k in 0..6 {
            let w = want.gradient_bohr[3 + k];
            assert!((got.gradient_bohr[k] - w).abs() < 1e-14);
        }
        let (h, hw) = (got.hessian_bohr.unwrap(), want.hessian_bohr.unwrap());
        for i in 0..6 {
            for j in 0..6 {
                assert!((h[(i, j)] - hw[(3 + i, 3 + j)]).abs() < 1e-12);
            }
        }
        assert_eq!(
            on_core.atoms().unwrap(),
            vec![AtomId::new(0), AtomId::new(1)]
        );
        // On an atom of the environment: refused with the atom.
        let outside = keep(0, 2);
        let why = CoreTerm::new(Arc::new(AdBias::new(outside)), &template, &split)
            .err()
            .unwrap();
        assert!(why.contains("atom 1 of the environment"), "{why}");
    }
}
