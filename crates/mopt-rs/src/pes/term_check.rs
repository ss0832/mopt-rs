//! Consistency checks for in-process terms (bias potentials, user-defined terms).
//!
//! Every [`PesTerm`] must have an exact gradient and Hessian and, in periodic cells, depend on
//! positions only through minimum-image geometry along the periodic axes. These checks verify
//! both with finite differences and lattice translations, in isolated, 1D, 2D, and 3D cells;
//! the built-in bias potentials are tested with them and users can apply them to their own
//! terms.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{AtomId, MolecularSystem, Position};
use crate::error::{MoptError, MoptResult};
use crate::pes::term::{PesTerm, TermContext};

/// Largest deviations found by [`check_term`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TermCheck {
    /// Max |analytic - finite-difference| gradient component (Hartree/Bohr).
    pub gradient_error: f64,
    /// Max |analytic - finite-difference| Hessian element (Hartree/Bohr^2).
    pub hessian_error: f64,
    /// Max |analytic Hessian - its transpose|.
    pub asymmetry: f64,
    /// Max energy change when single atoms are moved by lattice vectors of periodic axes
    /// (zero for isolated systems).
    pub lattice_invariance_error: f64,
}

impl TermCheck {
    /// Whether all deviations are below `tol`.
    pub fn passes(&self, tol: f64) -> bool {
        self.gradient_error < tol
            && self.hessian_error < tol
            && self.asymmetry < tol
            && self.lattice_invariance_error < tol
    }
}

/// Check `term` at `sys` with central differences of step `step` (Bohr).
pub fn check_term(term: &dyn PesTerm, sys: &MolecularSystem, step: f64) -> MoptResult<TermCheck> {
    let ctx = TermContext::new(0, sys, None);
    let eval = |s: &MolecularSystem, h: bool| term.evaluate(s, &ctx, h);
    let at = eval(sys, true)?;
    let hess = at
        .hessian_bohr
        .clone()
        .ok_or_else(|| MoptError::calc(format!("term '{}' returned no Hessian", term.name())))?;
    let x = sys.coords_flat_bohr();
    let n = x.nrows();
    let (mut ge, mut he, mut asym) = (0.0_f64, 0.0_f64, 0.0_f64);
    for k in 0..n {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[k] += step;
        xm[k] -= step;
        let p = eval(&sys.with_flat_coords_bohr(&xp)?, false)?;
        let m = eval(&sys.with_flat_coords_bohr(&xm)?, false)?;
        let fd = (p.energy.value() - m.energy.value()) / (2.0 * step);
        ge = ge.max((fd - at.gradient_bohr[k]).abs());
        for l in 0..n {
            let fd = (p.gradient_bohr[l] - m.gradient_bohr[l]) / (2.0 * step);
            he = he.max((fd - hess[(l, k)]).abs());
            asym = asym.max((hess[(k, l)] - hess[(l, k)]).abs());
        }
    }
    let mut lie = 0.0_f64;
    if let Some(cell) = sys.active_cell() {
        let e0 = at.energy.value();
        for axis in (0..3).filter(|&a| cell.periodic_axes[a]) {
            let v = cell.lattice_vector(axis).value();
            for a in sys.atoms() {
                for sign in [1.0, -1.0] {
                    let mut moved = sys.clone();
                    let p = sys.position(a).value();
                    moved.set_position(
                        a,
                        Position::new([p[0] + sign * v[0], p[1] + sign * v[1], p[2] + sign * v[2]]),
                    );
                    lie = lie.max((eval(&moved, false)?.energy.value() - e0).abs());
                }
            }
        }
    }
    Ok(TermCheck {
        gradient_error: ge,
        hessian_error: he,
        asymmetry: asym,
        lattice_invariance_error: lie,
    })
}

/// Energy change when atom `atom` is moved by `shift` (Bohr); used to confirm that
/// non-periodic directions are not wrapped.
pub fn energy_change_on_shift(
    term: &dyn PesTerm,
    sys: &MolecularSystem,
    atom: AtomId,
    shift: [f64; 3],
) -> MoptResult<f64> {
    let ctx = TermContext::new(0, sys, None);
    let e0 = term.evaluate(sys, &ctx, false)?.energy.value();
    let mut moved = sys.clone();
    let p = sys.position(atom).value();
    moved.set_position(
        atom,
        Position::new([p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]),
    );
    Ok(term.evaluate(&moved, &ctx, false)?.energy.value() - e0)
}
