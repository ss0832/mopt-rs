//! `collins_kimball`: the diffusion limit of the bimolecular steps of a solution. A step
//! `a + S -> b` whose activated rate constant `k_act` (L/mol/s, the rate theory with the other
//! corrections) approaches the rate `k_D` at which `a` and `S` meet by diffusion is slowed to
//! Collins and Kimball's `1 / k = 1 / k_D + 1 / k_act`, i.e. the factor
//!
//! ```text
//! f = k_D / (k_D + k_act) = exp(-softplus(ln k_act - ln k_D))
//! ```
//!
//! on the flux of the step (both directions: the reverse step leaves the encounter pair through
//! the same diffusion, so the equilibrium constant is kept; `kinetics_beyond_tst.md`, section
//! 3.2: `1/F = 1/F_act + 1/F_diff`). `k_D = 4 pi R (D_a + D_S) N_A` is Smoluchowski's rate with
//! the Stokes-Einstein coefficients of the van der Waals radii of the two structures in a
//! solvent of the viscosity of `--viscosity` ([`super::encounter::diffusion_rate`]). The
//! diagnostic `k_act_over_k_d` says where it matters (above about 0.01).
//!
//! References: M. v. Smoluchowski, Z. Phys. Chem. 1917, 92, 129; F. C. Collins, G. E. Kimball,
//! J. Colloid Sci. 1949, 4, 425 (the forms of the textbooks, as in [`super::encounter`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::encounter::{diffusion_rate, EncounterPair};
use super::{AfterOthers, Diagnostic, RateCorrection, StepContext};
use crate::error::{MoptError, MoptResult};
use crate::kinetics::conditions::Conditions;
use crate::kinetics::network::{Bottleneck, KineticNetwork};
use crate::math::stable::softplus;

/// The diffusion limit of the bimolecular steps.
#[derive(Debug, Clone, Copy, Default)]
pub struct CollinsKimball;

impl RateCorrection for CollinsKimball {
    fn name(&self) -> &str {
        "collins_kimball"
    }

    fn ln_factor(
        &self,
        _b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        _c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        Ok(None)
    }

    fn after_others(&self) -> bool {
        true
    }

    fn ln_factor_after(
        &self,
        b: &Bottleneck,
        _step: &StepContext,
        net: &KineticNetwork,
        c: &Conditions,
        after: &AfterOthers,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        let (Some(p), Some(ln_conc)) = (&b.partner, after.ln_concentration) else {
            return Ok(None);
        };
        let Some(eta) = c.viscosity else {
            return Err(MoptError::config(
                "collins_kimball needs the viscosity of the solvent (--viscosity, Pa s)",
            ));
        };
        let free = p.free_side;
        let state = &net.states[b.states[free]];
        let species = &net.species[p.species];
        let (Some(a), Some(s)) = (&state.system, &species.data.system) else {
            return Err(MoptError::config(format!(
                "{}: collins_kimball needs the structures of {} and of the species {} (their \
                 radii)",
                b.label, state.label, species.name
            )));
        };
        let k_d = diffusion_rate(&EncounterPair::of(a, s), c.temperature, eta);
        let ln_k_act = after.ln_k[free] - ln_conc;
        let x = ln_k_act - k_d.ln();
        Ok(Some((
            -softplus(x),
            vec![
                Diagnostic::new("k_act_over_k_d", x.exp()),
                Diagnostic::new("k_d_l_mol_s", k_d),
            ],
        )))
    }
}
