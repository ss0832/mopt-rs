//! The `Calculator` trait: anything that returns an energy and its derivatives for a structure.
//!
//! Calculators work in Cartesian coordinates only: they receive the structure as Cartesian
//! positions (Bohr) and return the gradient and Hessian with respect to them (Hartree/Bohr,
//! Hartree/Bohr^2). The coordinate system an optimizer uses (Cartesian, RIC, DIC, user-defined)
//! never reaches a calculator; optimizers convert gradients and Hessians themselves. Any
//! coordinate system therefore works with any program, including QM programs that accept only
//! Cartesian geometries.
//!
//! MultiOptPy: `Calculator/*_calculation_tools.py:Calculation` (`single_point`,
//!  `calc_analytic_hessian`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Orbitals};
use crate::error::{MoptError, MoptResult};
use crate::pes::point_charges::{EmbeddedOutput, PointCharges};

/// Potential energy surface provider.
pub trait Calculator: Send + Sync {
    /// Short name used in logs and output directory names.
    fn name(&self) -> &str;

    /// Where the calculator comes from, for the run log (`lj: built into mopt-rs 2.1.1`;
    /// Python engines name their package, version, and file).
    fn description(&self) -> String {
        format!("{}: built into mopt-rs {}", self.name(), crate::VERSION)
    }

    /// The papers to cite for the method and the program behind the calculator, for the run
    /// log.
    fn citations(&self) -> Vec<String> {
        Vec::new()
    }

    /// Evaluate energy, gradient, and (on request) Hessian.
    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput>;

    /// Whether periodic cells are handled.
    fn supports_pbc(&self) -> bool {
        false
    }

    /// Whether the energy is invariant under rigid translations and rotations, so that
    /// rigid-body modes can be removed from steps (false for model surfaces in a fixed frame).
    fn is_rigid_body_invariant(&self) -> bool {
        true
    }

    /// Whether the Hessian of [`Self::evaluate`] is exact and costs little more than a
    /// gradient (analytic, or by automatic differentiation) rather than being made of
    /// differences of gradients: workflows ask for Hessians more often when they are (the IRC
    /// of `run_mapper` at every tenth point). Wrappers answer for what they wrap.
    fn analytic_hessian(&self) -> bool {
        false
    }

    /// Whether electronic states other than the ground state (`EvalRequest::excited_state > 0`)
    /// can be evaluated.
    fn supports_excited_states(&self) -> bool {
        false
    }

    /// Excitation energies (Hartree, ascending) of the excited states of `sys` that
    /// `EvalRequest::excited_state` counts, for reports; `None` if the calculator does not
    /// tell them.
    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        let _ = sys;
        Ok(None)
    }

    /// The lowest `count` excited states of `sys` with their excitation energies, transition
    /// dipoles, and changes of the atomic populations ([`crate::pes::states::Transition`]), for
    /// the couplings of an electron transfer (Mulliken-Hush); `None` if the calculator does not
    /// give them.
    fn transitions(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<Vec<crate::pes::states::Transition>>> {
        let _ = (sys, count);
        Ok(None)
    }

    /// The lowest `count` excited states of `sys` in the form that is compared between
    /// structures ([`crate::pes::states`]), so that a state can be followed
    /// ([`crate::pes::followed`]); `None` if the calculator does not hand them out.
    fn state_characters(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<crate::pes::states::StateCharacters>> {
        let _ = (sys, count);
        Ok(None)
    }

    /// Where states are followed ([`crate::pes::followed`]): the index (0: the lowest) of
    /// the state at `sys` that the followed state `label` is, once the structure has been
    /// evaluated; `None` otherwise, and where states are counted by their energies. The
    /// journal of a run records it.
    fn followed_state(&self, sys: &MolecularSystem, label: usize) -> Option<usize> {
        let _ = (sys, label);
        None
    }

    /// Told by the journal of a restarted run: the followed state `label` was the state
    /// with the index `root` at `sys`. Nothing where states are not followed.
    fn recall_state(&self, sys: &MolecularSystem, label: usize, root: usize) {
        let _ = (sys, label, root);
    }

    /// Where the calculator relaxes a part of the structure itself (the environment of ONIOM
    /// microiterations): the relaxed coordinates of that part (flat, Bohr) at `sys`, once the
    /// structure has been evaluated; `None` otherwise. The journal of a run records them.
    fn relaxed_environment(&self, sys: &MolecularSystem) -> Option<Vec<f64>> {
        let _ = sys;
        None
    }

    /// Told by the journal of a restarted run: at `sys` the relaxed part was `environment`
    /// ([`Self::relaxed_environment`]). Nothing where nothing is relaxed.
    fn recall_environment(&self, sys: &MolecularSystem, environment: &[f64]) {
        let _ = (sys, environment);
    }

    /// Whether the evaluations of a batch (the images of a chain, the states of a model
    /// function) must be made one after the other in their order, because a result depends
    /// on what was evaluated before (a calculator that follows its states: which structure
    /// comes first decides what a state is). Batches run in parallel otherwise.
    fn evaluates_in_order(&self) -> bool {
        false
    }

    /// The molecular orbitals of `sys` (coefficients, energies, occupations, and the overlap
    /// of the atomic orbitals), for analyses of the wavefunction; `None` if the calculator
    /// does not hand them out.
    fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Option<Orbitals>> {
        let _ = sys;
        Ok(None)
    }

    /// Whether the calculator treats electrons explicitly (quantum chemistry), so that the
    /// charge and multiplicity must be possible for the elements (see
    /// [`crate::domain::SpinCharge::check_electrons`]). Force fields and model surfaces ignore
    /// them.
    fn uses_electrons(&self) -> bool {
        false
    }

    /// Whether the calculator tests its SCF solutions for instabilities when asked to
    /// (`-stability`; see [`crate::pes::registry::CalculatorOptions::stability`]).
    fn checks_stability(&self) -> bool {
        false
    }

    /// Warnings gathered since the last call, for the run log (for example a bonding pattern
    /// that no longer matches a force field's fixed topology). Drivers call this after each
    /// batch of evaluations.
    fn take_warnings(&self) -> Vec<String> {
        Vec::new()
    }

    /// The energy of `sys` split over the atoms with the virials of the atoms (for the heat
    /// current of the thermal conductivity); `None` if the calculator cannot split it.
    fn site_terms(&self, sys: &MolecularSystem) -> Option<MoptResult<SiteTerms>> {
        let _ = sys;
        None
    }

    /// Whether [`Self::evaluate_in_charges`] is available (electronic embedding).
    fn supports_point_charges(&self) -> bool {
        false
    }

    /// Energy, gradient, and (on request) Hessian of `sys` in the field of the fixed point
    /// charges `charges` ([`crate::pes::point_charges`]): the energy holds the interaction of
    /// the system with the charges, not that of the charges among themselves, and the
    /// gradient with respect to the positions of the charges comes along.
    fn evaluate_in_charges(
        &self,
        sys: &MolecularSystem,
        req: &EvalRequest,
        charges: &PointCharges,
    ) -> MoptResult<EmbeddedOutput> {
        let _ = (sys, req, charges);
        Err(MoptError::unsupported(format!(
            "calculator '{}' takes no point charges (electronic embedding)",
            self.name()
        )))
    }

    /// Fixed atomic charges of `sys` (e) in which another calculation can be embedded, such as
    /// the charges of a force field; `None` if the calculator has none.
    fn embedding_charges(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        let _ = sys;
        Ok(None)
    }
}

/// The energy of a structure split over its atoms, `E = sum_i U_i`, and the virials of the
/// atoms `W_i = sum_j r_ij (x) dU_j / dr_ji` (`r_ij = r_j - r_i`; Fan et al., Phys. Rev. B
/// 2015, 92, 094301) over the neighbours `j` of `i` and their periodic images, an atom's own
/// images included, so that `sum_i W_i` is minus the virial of the stress in any cell; the
/// potential part of the heat current is `sum_i W_i v_i`.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteTerms {
    /// Site energies (Hartree).
    pub energies: Vec<f64>,
    /// Virials (Hartree), `[atom][a][b]`.
    pub virials: Vec<[[f64; 3]; 3]>,
}

/// What an evaluation said, as lines for the run log: the warnings of `calc` gathered
/// since they were taken last, `<S**2>`, and the output of the program itself. `of` names
/// the evaluation (`image 3`, `IRC point 12 (Path)`).
pub fn said(calc: &dyn Calculator, of: &str, out: &EvalOutput) -> Vec<String> {
    let mut lines: Vec<String> = calc
        .take_warnings()
        .into_iter()
        .map(|warning| format!("Warning: {of}: {warning}"))
        .collect();
    if let Some(orbitals) = &out.orbitals {
        if let Some(spin) = orbitals.spin_square {
            lines.push(format!("{of}: <S**2> = {spin:.6}"));
        }
        if let Some(text) = &orbitals.program_output {
            lines.push(format!(
                "{of}: output of the program\n{}",
                text.trim_end_matches('\n')
            ));
        }
    }
    lines
}

/// Evaluate after checking PBC support and the consistency of the result.
pub fn evaluate_checked(
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    req: &EvalRequest,
) -> MoptResult<EvalOutput> {
    if sys.has_pbc() && !calc.supports_pbc() {
        return Err(MoptError::unsupported(format!(
            "calculator '{}' does not support periodic boundary conditions",
            calc.name()
        )));
    }
    if req.excited_state > 0 && !calc.supports_excited_states() {
        return Err(MoptError::unsupported(format!(
            "calculator '{}' cannot evaluate excited state {}",
            calc.name(),
            req.excited_state
        )));
    }
    let out = calc.evaluate(sys, req)?;
    validate_output(&out, sys, req)
        .map_err(|e| e.context(format!("calculator '{}'", calc.name())))?;
    Ok(out)
}

/// Check shapes and finiteness of an evaluation result.
pub fn validate_output(
    out: &EvalOutput,
    sys: &MolecularSystem,
    req: &EvalRequest,
) -> MoptResult<()> {
    let n = sys.dof();
    if out.gradient_bohr.nrows() != n {
        return Err(MoptError::calc(format!(
            "gradient has length {}, expected {n}",
            out.gradient_bohr.nrows()
        )));
    }
    match &out.hessian_bohr {
        Some(h) if h.nrows() != n || h.ncols() != n => {
            return Err(MoptError::calc(format!(
                "Hessian is {}x{}, expected {n}x{n}",
                h.nrows(),
                h.ncols()
            )))
        }
        // A Hessian for a check of stability may be withheld where it cannot be had to
        // convergence; the check is then not judged (`EvalRequest::for_stability`).
        None if req.need_hessian && !req.for_stability => {
            return Err(MoptError::calc("Hessian requested but not returned"))
        }
        _ => {}
    }
    if !out.is_finite() {
        return Err(MoptError::calc(
            "energy, gradient, or Hessian contains non-finite values",
        ));
    }
    Ok(())
}
