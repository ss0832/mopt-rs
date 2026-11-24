//! A calculator that answers its first request for a Hessian with a Hessian that is given.
//!
//! An optimization can start from a Hessian that is already there (one from a vibrational
//! analysis, the Hessian that the relaxation of a chain kept for an image) without a
//! separate way into the optimizer: the optimizer asks the calculator for a Hessian at its
//! first point, as it does with `-fc`, and [`GivenHessian`] hands over the matrix it was
//! given, with the energy and the gradient of the calculator it wraps. Later requests go
//! to that calculator.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;
use std::sync::Mutex;

use faer::Mat;

use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Orbitals};
use crate::error::{MoptError, MoptResult};
use crate::io::hessian_file::read_hessian_file;
use crate::pes::calculator::Calculator;

/// Largest difference of two mirrored elements, relative to the largest element, at which
/// a matrix that is read counts as symmetric.
const ASYMMETRY: f64 = 1e-6;

/// `inner`, with the first Hessian that is asked for taken from a matrix (Cartesian
/// coordinates, Hartree/Bohr^2, the atoms in the order of the structure).
pub struct GivenHessian {
    inner: Box<dyn Calculator>,
    first: Mutex<Option<Mat<f64>>>,
}

/// `hessian` as a first Hessian is taken: square, finite, symmetric to 1e-6 of its
/// largest element, and made exactly symmetric (shared by [`GivenHessian::new`] and the
/// Python optimizer, which hands it over in its ask/tell loop).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn checked_hessian(hessian: Mat<f64>) -> MoptResult<Mat<f64>> {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n {
        return Err(MoptError::config(format!(
            "a Hessian is square; this one has {} rows and {} columns",
            hessian.nrows(),
            hessian.ncols()
        )));
    }
    let mut largest = 0.0_f64;
    let mut asymmetry = 0.0_f64;
    for r in 0..n {
        for c in 0..n {
            let value = hessian[(r, c)];
            if !value.is_finite() {
                return Err(MoptError::config(
                    "the Hessian has a value that is not finite",
                ));
            }
            largest = largest.max(value.abs());
            asymmetry = asymmetry.max((value - hessian[(c, r)]).abs());
        }
    }
    if asymmetry > ASYMMETRY * largest.max(f64::MIN_POSITIVE) {
        return Err(MoptError::config(format!(
            "the Hessian is not symmetric (elements differ by up to {asymmetry:.3e})"
        )));
    }
    Ok(Mat::from_fn(n, n, |r, c| {
        0.5 * (hessian[(r, c)] + hessian[(c, r)])
    }))
}

/// The Hessian of the file `path` (the kinds of files of [`crate::io::hessian_file`]) for
/// the structure `sys`: an error when its size is not that of the structure.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn read_for(path: &Path, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
    let hessian = read_hessian_file(path)?;
    if hessian.nrows() != sys.dof() {
        return Err(MoptError::config(format!(
            "{}: a Hessian of {} coordinates for a structure of {} atoms ({} coordinates)",
            path.display(),
            hessian.nrows(),
            sys.n_atoms(),
            sys.dof()
        )));
    }
    Ok(hessian)
}

impl GivenHessian {
    /// `inner` with `hessian` for its first Hessian. The matrix is made symmetric.
    pub fn new(inner: Box<dyn Calculator>, hessian: Mat<f64>) -> MoptResult<Self> {
        Ok(Self {
            inner,
            first: Mutex::new(Some(checked_hessian(hessian)?)),
        })
    }

    /// `inner` with the Hessian of the file `path` for its first Hessian (the kinds of
    /// files of [`crate::io::hessian_file`]); `sys` is the structure the optimization
    /// starts from.
    pub fn from_file(
        inner: Box<dyn Calculator>,
        path: &Path,
        sys: &MolecularSystem,
    ) -> MoptResult<Self> {
        Self::new(inner, read_for(path, sys)?)
    }
}

/// `calculator`, with the Hessian of `file` for its first Hessian when a file is named
/// (`-ih`).
pub fn wrap_if_given(
    calculator: Box<dyn Calculator>,
    file: Option<&Path>,
    sys: &MolecularSystem,
) -> MoptResult<Box<dyn Calculator>> {
    match file {
        Some(path) => Ok(Box::new(GivenHessian::from_file(calculator, path, sys)?)),
        None => Ok(calculator),
    }
}

impl Calculator for GivenHessian {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn citations(&self) -> Vec<String> {
        self.inner.citations()
    }

    fn description(&self) -> String {
        format!("{}; the first Hessian is given", self.inner.description())
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if !req.need_hessian {
            return self.inner.evaluate(sys, req);
        }
        let given = match self.first.lock() {
            Ok(mut first) => first.take().filter(|h| h.nrows() == sys.dof()),
            Err(_) => None,
        };
        let Some(hessian) = given else {
            return self.inner.evaluate(sys, req);
        };
        let plain = EvalRequest {
            need_gradient: true,
            need_hessian: false,
            ..*req
        };
        let mut out = self.inner.evaluate(sys, &plain)?;
        out.hessian_bohr = Some(hessian);
        Ok(out)
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
    use crate::domain::SpinCharge;
    use crate::io::xyz::parse_xyz_str;
    use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};

    #[test]
    fn the_first_hessian_is_the_one_given_and_the_next_the_calculators() {
        let sys = parse_xyz_str(
            "3\n\nAr 0.0 0.0 0.0\nAr 3.8 0.0 0.0\nAr 1.9 3.3 0.0\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let registry = CalculatorRegistry::native();
        let make = || {
            registry
                .create("lj", &CalculatorOptions::default(), &sys)
                .unwrap()
        };
        let given = Mat::from_fn(9, 9, |r, c| if r == c { 0.7 } else { 0.01 });
        let calculator = GivenHessian::new(make(), given.clone()).unwrap();
        let plain = make();
        let own = plain
            .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
            .unwrap();

        // Without a Hessian asked for nothing is used up.
        let only = calculator
            .evaluate(&sys, &EvalRequest::ENERGY_GRAD)
            .unwrap();
        assert!(only.hessian_bohr.is_none());
        let first = calculator
            .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
            .unwrap();
        assert!((first.energy.value() - own.energy.value()).abs() < 1e-14);
        let h = first.hessian_bohr.unwrap();
        for r in 0..9 {
            assert!((first.gradient_bohr[r] - own.gradient_bohr[r]).abs() < 1e-14);
            for c in 0..9 {
                assert!((h[(r, c)] - given[(r, c)]).abs() < 1e-15);
            }
        }
        let second = calculator
            .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
            .unwrap();
        let (h, expected) = (second.hessian_bohr.unwrap(), own.hessian_bohr.unwrap());
        for r in 0..9 {
            for c in 0..9 {
                assert!((h[(r, c)] - expected[(r, c)]).abs() < 1e-14);
            }
        }

        // What is no Hessian is refused.
        assert!(GivenHessian::new(make(), Mat::zeros(9, 6)).is_err());
        let mut skewed = given.clone();
        skewed[(0, 1)] = 0.5;
        assert!(GivenHessian::new(make(), skewed).is_err());
        let mut broken = given;
        broken[(2, 2)] = f64::NAN;
        assert!(GivenHessian::new(make(), broken).is_err());
    }
}
