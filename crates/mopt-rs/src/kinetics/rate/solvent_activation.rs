//! The solvent activation energy shift (SAES) of EDEEL for electron transfer within a molecule
//! of donor, bridge, and acceptor in an implicit solvent, its solvent reorganization energies,
//! and the decomposition of the barrier, after A. Mutsuji, PhD thesis, Hokkaido University
//! (2025), chapter 4 (doi:10.14943/doctoral.k16373). Chapter 4 of the thesis has not been
//! published in a peer-reviewed journal; equation numbers refer to the thesis.
//!
//! An implicit solvent relaxed to each diabatic state does not see the orientation of the
//! solvent molecules that the two states must share at a crossing. SAES adds to each diabatic
//! energy the free energy of a harmonic solvent coordinate `x` (0: relaxed to the initial
//! state, 1: to the final state), eqn (4.5) and (4.6):
//!
//! ```text
//! G_SA^i(x) = lambda_s^i x^2,    G_SA^f(x) = lambda_s^f (1 - x)^2,    x = alpha_SA x_s
//! ```
//!
//! with `x_s` the solvent coordinate at which the solvent alone brings the two states to the
//! same energy, eqn (4.8) (eqn (4.9) for `lambda_s^i = lambda_s^f`):
//!
//! ```text
//! x_s = (lambda_s^f - sqrt(lambda_s^i lambda_s^f - (lambda_s^f - lambda_s^i) dG_s0))
//!       / (lambda_s^f - lambda_s^i)
//! ```
//!
//! `alpha_SA` in `[0, 1]` shares the activation between the solvent (`1`: the solvent alone)
//! and the deformation of the donor and the acceptor (`0`: the inner sphere alone). Eqn (4.8)
//! is written here in the form `x_s = (lambda_s^f + dG_s0) / (lambda_s^f + sqrt(...))`, the
//! same number without the cancellation of eqn (4.8) when the two reorganization energies are
//! close, and with eqn (4.9) as its limit.
//!
//! The thesis does not define `dG_s0` beyond Fig. 4.3 (the solvent reaction energy at a fixed
//! structure of donor and acceptor). The crossing of the two shifted parabolas at `x_s` holds
//! for `dG_s0 = V22 - V11` at that structure; the thesis's statement that at `alpha_SA = 1` the
//! inner-sphere activation is zero (appendix 4.7) makes that structure the equilibrium of the
//! initial state, so [`SolventActivation::from_initial_equilibrium`] takes the vertical gap
//! there.
//!
//! The solvent reorganization energies come from nonequilibrium solvation of the fragments at
//! the initial equilibrium, eqn (4.10) and (4.11) ([`solvent_reorganization`]): the energy of
//! each fragment in the slow polarization of its other charge state less its equilibrium
//! energy, less the interaction of the two charges through the solvent that the separate
//! fragments leave out, `e^2 / (4 pi eps0) (1 / eps_inf - 1 / eps_s) / R_DA`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// The harmonic solvent coordinate of SAES for one pair of diabatic states (Hartree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolventActivation {
    /// `lambda_s^i`, the solvent reorganization energy of the initial state.
    pub lambda_initial: f64,
    /// `lambda_s^f`, the solvent reorganization energy of the final state.
    pub lambda_final: f64,
    /// `dG_s0`, the solvent reaction energy at the structure where the solvent alone
    /// activates the transfer.
    pub reaction_energy: f64,
}

impl SolventActivation {
    /// The coordinate of `lambda_initial`, `lambda_final`, and `reaction_energy` (Hartree).
    /// Both reorganization energies are positive.
    pub fn new(lambda_initial: f64, lambda_final: f64, reaction_energy: f64) -> MoptResult<Self> {
        let ok = |v: f64| v.is_finite() && v > 0.0;
        if !(ok(lambda_initial) && ok(lambda_final) && reaction_energy.is_finite()) {
            return Err(MoptError::config(format!(
                "SAES: the solvent reorganization energies must be positive ({lambda_initial}, \
                 {lambda_final}) and the solvent reaction energy finite ({reaction_energy})"
            )));
        }
        Ok(Self {
            lambda_initial,
            lambda_final,
            reaction_energy,
        })
    }

    /// The coordinate with `dG_s0` the vertical gap `V22 - V11` (Hartree) at the equilibrium
    /// of the initial state, where `alpha_SA = 1` puts the crossing (see the module text).
    pub fn from_initial_equilibrium(
        lambda_initial: f64,
        lambda_final: f64,
        v11: f64,
        v22: f64,
    ) -> MoptResult<Self> {
        Self::new(lambda_initial, lambda_final, v22 - v11)
    }

    /// `x_s`, eqn (4.8) and (4.9), in a form without cancellation (see the module text).
    pub fn crossing(&self) -> MoptResult<f64> {
        let (li, lf, dg) = (self.lambda_initial, self.lambda_final, self.reaction_energy);
        let radicand = li * lf - (lf - li) * dg;
        if radicand < 0.0 {
            return Err(MoptError::calc(format!(
                "SAES: the two solvent parabolas do not cross (lambda_s^i lambda_s^f - \
                 (lambda_s^f - lambda_s^i) dG_s0 = {radicand} < 0)"
            )));
        }
        Ok((lf + dg) / (lf + radicand.sqrt()))
    }

    /// The solvent coordinate `x = alpha x_s`, eqn (4.7).
    pub fn coordinate(&self, alpha: f64) -> MoptResult<f64> {
        if !alpha.is_finite() {
            return Err(MoptError::config("SAES: alpha_SA must be finite"));
        }
        Ok(alpha * self.crossing()?)
    }

    /// The shifts `[G_SA^i, G_SA^f]` (Hartree) of the initial and final diabatic states at
    /// `alpha`, eqn (4.5) and (4.6).
    pub fn shifts(&self, alpha: f64) -> MoptResult<[f64; 2]> {
        let x = self.coordinate(alpha)?;
        Ok([
            self.lambda_initial * x * x,
            self.lambda_final * (1.0 - x) * (1.0 - x),
        ])
    }

    /// The mean of the two solvent reorganization energies, the solvent share of the total
    /// reorganization energy of eqn (4.17).
    pub fn mean_reorganization(&self) -> f64 {
        0.5 * (self.lambda_initial + self.lambda_final)
    }
}

/// Energies (Hartree) of one fragment in nonequilibrium and equilibrium solvation, for
/// [`solvent_reorganization`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolvatedPair {
    /// The fragment in its state, in the slow polarization of its other charge state.
    pub nonequilibrium: f64,
    /// The fragment in its state, in the polarization of that state.
    pub equilibrium: f64,
}

/// A solvent reorganization energy from nonequilibrium solvation, eqn (4.10) or (4.11):
/// `lambda_s = sum (E(NonEq) - E(Eq)) - (1 / eps_inf - 1 / eps_s) / R_DA` (atomic units, `e = 1`)
/// for the two fragments of a state (the donor-bridge fragment and the bridge-acceptor
/// fragment), the distance `distance_bohr` between the centroids of the donor and the
/// acceptor, and the optical and static dielectric constants. In vacuum (both constants 1) it
/// is zero when the fragment energies do not depend on the solvation.
pub fn solvent_reorganization(
    donor_bridge: SolvatedPair,
    bridge_acceptor: SolvatedPair,
    distance_bohr: f64,
    optical: f64,
    static_constant: f64,
) -> MoptResult<f64> {
    let ok = |v: f64| v.is_finite() && v > 0.0;
    if !(ok(distance_bohr) && ok(optical) && ok(static_constant)) {
        return Err(MoptError::config(
            "solvent reorganization: the distance and the dielectric constants must be positive",
        ));
    }
    let fragments = (donor_bridge.nonequilibrium - donor_bridge.equilibrium)
        + (bridge_acceptor.nonequilibrium - bridge_acceptor.equilibrium);
    Ok(fragments - (1.0 / optical - 1.0 / static_constant) / distance_bohr)
}

/// The decomposition of a barrier of SAES (thesis appendix 4.6, eqn (4.S1)-(4.S3)), Hartree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarrierDecomposition {
    /// The solvent activation `G_SA^i` at the crossing.
    pub solvent: f64,
    /// The inner-sphere activation `V11(SX) - V11(EQ_i)`, eqn (4.S3).
    pub inner: f64,
    /// The activation of the donor `E^DB_{n+1}(SX) - E^DB_{n+1}(EQ_i)`, eqn (4.S1) (with the
    /// deformation of the bridge).
    pub donor: f64,
    /// The activation of the acceptor, `inner - donor`, eqn (4.S2).
    pub acceptor: f64,
    /// The whole barrier `solvent + inner`.
    pub total: f64,
}

/// The decomposition of the barrier at a crossing: `v11_sx` and `v11_eq` the initial diabatic
/// energy at the crossing and at the initial equilibrium, `donor_sx` and `donor_eq` the energy
/// of the donor fragment with the moving electron there, and `solvent` the shift `G_SA^i`.
pub fn decompose_barrier(
    v11_sx: f64,
    v11_eq: f64,
    donor_sx: f64,
    donor_eq: f64,
    solvent: f64,
) -> BarrierDecomposition {
    let inner = v11_sx - v11_eq;
    let donor = donor_sx - donor_eq;
    BarrierDecomposition {
        solvent,
        inner,
        donor,
        acceptor: inner - donor,
        total: solvent + inner,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_shifted_parabolas_cross_at_the_solvent_coordinate() {
        for (li, lf, dg) in [
            (0.03, 0.03, -0.01),
            (0.03, 0.031, 0.02),
            (0.02, 0.035, -0.005),
            (0.035, 0.02, 0.004),
        ] {
            let s = SolventActivation::new(li, lf, dg).unwrap();
            let x = s.crossing().unwrap();
            // V11 + lambda_i x^2 = V22 + lambda_f (1 - x)^2 with V22 - V11 = dG_s0.
            let gap = li * x * x - (dg + lf * (1.0 - x) * (1.0 - x));
            assert!(gap.abs() < 1e-15, "{li} {lf} {dg}: {gap}");
            // The root of eqn (4.8) itself, away from equal reorganization energies.
            if (lf - li).abs() > 1e-3 {
                let direct = (lf - (li * lf - (lf - li) * dg).sqrt()) / (lf - li);
                assert!((direct - x).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn equal_reorganization_energies_give_eqn_4_9() {
        let (l, dg) = (0.0305, -0.0123);
        let s = SolventActivation::new(l, l, dg).unwrap();
        let x = s.crossing().unwrap();
        assert!((x - 0.5 * (1.0 + dg / l)).abs() < 1e-15);
        // Nearly equal: no cancellation.
        let near = SolventActivation::new(l, l * (1.0 + 1e-13), dg).unwrap();
        assert!((near.crossing().unwrap() - x).abs() < 1e-12);
    }

    #[test]
    fn alpha_shares_the_activation() {
        let s = SolventActivation::new(0.03, 0.03, -0.004).unwrap();
        let [gi, gf] = s.shifts(1.0).unwrap();
        // The solvent alone: the two shifted states meet (dG_s0 = V22 - V11).
        assert!((gi - (gf - 0.004)).abs() < 1e-15);
        assert_eq!(s.shifts(0.0).unwrap(), [0.0, 0.03]);
        assert!((s.mean_reorganization() - 0.03).abs() < 1e-15);
    }

    #[test]
    fn the_reorganization_energy_subtracts_the_solvent_screened_coupling() {
        let pair = |noneq, eq| SolvatedPair {
            nonequilibrium: noneq,
            equilibrium: eq,
        };
        // Vacuum: nothing but the fragment differences, which are then zero.
        let l = solvent_reorganization(pair(-1.0, -1.0), pair(-2.0, -2.0), 20.0, 1.0, 1.0);
        assert_eq!(l.unwrap(), 0.0);
        // THF (Gaussian's constants) at R_DA = 20 Bohr.
        let l =
            solvent_reorganization(pair(-1.0, -1.02), pair(-2.0, -2.01), 20.0, 1.974025, 7.4257)
                .unwrap();
        let expected = 0.03 - (1.0 / 1.974025 - 1.0 / 7.4257) / 20.0;
        assert!((l - expected).abs() < 1e-15);
    }

    #[test]
    fn the_barrier_decomposes_into_solvent_donor_and_acceptor() {
        let d = decompose_barrier(-10.0, -10.012, -5.0, -5.004, 0.006);
        assert!((d.inner - 0.012).abs() < 1e-12);
        assert!((d.donor - 0.004).abs() < 1e-12);
        assert!((d.acceptor - 0.008).abs() < 1e-12);
        assert!((d.total - 0.018).abs() < 1e-12);
    }
}
