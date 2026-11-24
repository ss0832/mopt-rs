//! Electronic couplings of an electron transfer from excited states, and the maximization of
//! a rate constant along a scan.
//!
//! - **Mulliken-Hush** (thesis of A. Mutsuji, Hokkaido University 2025, eqn (4.18); N. S.
//!   Hush, Prog. Inorg. Chem. 1967, 8, 391): `V12 = E_ex |mu_ge| / (e r_ab)` from the excitation
//!   energy `E_ex` of the charge-transfer state, its transition dipole `mu_ge`, and the distance
//!   `r_ab` the electron moves.
//! - **Generalized Mulliken-Hush** (R. J. Cave, M. D. Newton, Chem. Phys. Lett. 1996, 249, 15):
//!   `V12 = |mu_12| E_ex / sqrt((mu_1 - mu_2)^2 + 4 mu_12^2)`, with the dipoles projected on the
//!   direction of the transfer; it does not need the two states to be localized.
//! - **The charge-transfer index** of T. Le Bahers, C. Adamo, I. Ciofini (J. Chem. Theory
//!   Comput. 2011, 7, 2498), on atoms: the change of the electron populations of the atoms from
//!   the ground to the excited state split into its gains and losses; the charge transferred
//!   `q_CT` is the sum of the gains, the distance `d_CT` that between the barycentres of the
//!   gains and of the losses, and `d_CT` is `r_ab` of Mulliken-Hush. The thesis takes as the
//!   charge-transfer state the lowest excited state with `q_CT` above 0.985
//!   ([`CT_THRESHOLD`]).
//! - **Maximization along a scan** ([`maximize_on_grid`]): the largest value, refined by the
//!   parabola through it and its two neighbours when it is inside the scan and the parabola
//!   opens downwards (chapter 3 of the thesis does this along the donor-acceptor distance with
//!   three rate constants).
//!
//! Atomic units throughout (`e = 1`: dipoles in e Bohr, distances in Bohr, energies in
//! Hartree).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// The share of an electron that a charge-transfer state moves at least (the thesis's
/// empirical threshold of section 4.2).
pub const CT_THRESHOLD: f64 = 0.985;

/// The Mulliken-Hush coupling (Hartree) of the excitation energy `excitation` (Hartree), the
/// transition dipole `transition_dipole` (e Bohr), and the transfer distance `distance` (Bohr).
pub fn mulliken_hush(
    excitation: f64,
    transition_dipole: [f64; 3],
    distance: f64,
) -> MoptResult<f64> {
    if !(distance.is_finite() && distance > 0.0 && excitation.is_finite()) {
        return Err(MoptError::config(
            "Mulliken-Hush: the transfer distance must be positive and the excitation finite",
        ));
    }
    let mu = transition_dipole.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(excitation.abs() * mu / distance)
}

/// The generalized Mulliken-Hush coupling (Hartree): the excitation energy, the transition
/// dipole, and the difference of the dipoles of the two adiabatic states (e Bohr), projected
/// on the direction of that difference.
pub fn generalized_mulliken_hush(
    excitation: f64,
    transition_dipole: [f64; 3],
    dipole_difference: [f64; 3],
) -> MoptResult<f64> {
    let dd = dipole_difference.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(dd.is_finite() && dd > 0.0) {
        return Err(MoptError::calc(
            "generalized Mulliken-Hush: the two states have the same dipole",
        ));
    }
    // The component of the transition dipole along the change of the dipole.
    let mu12 = transition_dipole
        .iter()
        .zip(&dipole_difference)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        / dd;
    Ok(mu12.abs() * excitation.abs() / (dd * dd + 4.0 * mu12 * mu12).sqrt())
}

/// The fragment-charge-difference coupling (Hartree) of A. A. Voityuk and N. Rosch (J. Chem.
/// Phys. 2002, 117, 5607): `V12 = E_ex |dq_12| / sqrt((dq_1 - dq_2)^2 + 4 dq_12^2)`, with
/// `dq_k` the difference of the electrons on the donor and on the acceptor in the state `k`
/// and `dq_12` that of the transition density. From the excitation energy, the populations of
/// the transition density on each atom, and the change of the electrons on each atom from the
/// ground to the excited state (`dq_1 - dq_2` is that change on the donor less on the
/// acceptor); `donor` and `acceptor` are the atoms (0-based) of the two, a bridge being
/// neither.
pub fn fragment_charge_difference(
    excitation: f64,
    transition_population: &[f64],
    population_change: &[f64],
    donor: &[usize],
    acceptor: &[usize],
) -> MoptResult<f64> {
    let n = transition_population.len();
    if population_change.len() != n || donor.is_empty() || acceptor.is_empty() {
        return Err(MoptError::config(
            "fragment charge difference: one transition population and one population change \
             per atom, and atoms of both the donor and the acceptor",
        ));
    }
    if let Some(&k) = donor.iter().chain(acceptor).find(|&&k| k >= n) {
        return Err(MoptError::config(format!(
            "fragment charge difference: atom {} of {n}",
            k + 1
        )));
    }
    if donor.iter().any(|k| acceptor.contains(k)) {
        return Err(MoptError::config(
            "fragment charge difference: the donor and the acceptor share atoms",
        ));
    }
    let on = |values: &[f64], atoms: &[usize]| atoms.iter().map(|&k| values[k]).sum::<f64>();
    let dq12 = on(transition_population, donor) - on(transition_population, acceptor);
    let dq = on(population_change, donor) - on(population_change, acceptor);
    let den = (dq * dq + 4.0 * dq12 * dq12).sqrt();
    if !(den.is_finite() && den > 0.0) {
        return Err(MoptError::calc(
            "fragment charge difference: the state moves no charge between the donor and the \
             acceptor",
        ));
    }
    Ok(excitation.abs() * dq12.abs() / den)
}

/// The charge transfer of an excitation on atoms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChargeTransfer {
    /// `q_CT` (electrons).
    pub charge: f64,
    /// `d_CT` (Bohr).
    pub distance: f64,
    /// The barycentre of the electrons gained (Bohr).
    pub gained_at: [f64; 3],
    /// The barycentre of the electrons lost (Bohr).
    pub lost_at: [f64; 3],
}

/// The charge-transfer index of the change `population_change` of the electron population of
/// each atom (excited less ground) at `positions` (Bohr).
pub fn charge_transfer(
    positions: &[[f64; 3]],
    population_change: &[f64],
) -> MoptResult<ChargeTransfer> {
    if positions.len() != population_change.len() || positions.is_empty() {
        return Err(MoptError::config(
            "charge transfer: one population change per atom",
        ));
    }
    let (mut gain, mut loss) = (0.0, 0.0);
    let (mut rg, mut rl) = ([0.0; 3], [0.0; 3]);
    for (p, &d) in positions.iter().zip(population_change) {
        if d > 0.0 {
            gain += d;
            for c in 0..3 {
                rg[c] += d * p[c];
            }
        } else if d < 0.0 {
            loss -= d;
            for c in 0..3 {
                rl[c] -= d * p[c];
            }
        }
    }
    if gain <= 0.0 || loss <= 0.0 {
        return Ok(ChargeTransfer {
            charge: 0.0,
            distance: 0.0,
            gained_at: rg,
            lost_at: rl,
        });
    }
    let rg = rg.map(|v| v / gain);
    let rl = rl.map(|v| v / loss);
    let distance = (0..3).map(|c| (rg[c] - rl[c]).powi(2)).sum::<f64>().sqrt();
    Ok(ChargeTransfer {
        charge: gain,
        distance,
        gained_at: rg,
        lost_at: rl,
    })
}

/// The largest of the values `y` at the points `x` of a scan (in the order of the scan), and
/// where it is: the vertex of the parabola through the largest value and its two neighbours
/// when the largest is inside the scan and the parabola opens downwards, the largest point
/// otherwise. Points whose value is not finite are left out.
pub fn maximize_on_grid(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let finite: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    let (best, &(xb, yb)) = finite
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))?;
    if best == 0 || best + 1 == finite.len() {
        return Some((xb, yb));
    }
    let (x0, y0) = finite[best - 1];
    let (x2, y2) = finite[best + 1];
    // The parabola through the three points.
    let d0 = (x0 - xb) * (x0 - x2);
    let d1 = (xb - x0) * (xb - x2);
    let d2 = (x2 - x0) * (x2 - xb);
    if d0 == 0.0 || d1 == 0.0 || d2 == 0.0 {
        return Some((xb, yb));
    }
    let a = y0 / d0 + yb / d1 + y2 / d2;
    if a >= 0.0 {
        return Some((xb, yb));
    }
    let b = -(y0 * (xb + x2) / d0 + yb * (x0 + x2) / d1 + y2 * (x0 + xb) / d2);
    let c = y0 * xb * x2 / d0 + yb * x0 * x2 / d1 + y2 * x0 * xb / d2;
    let xv = -b / (2.0 * a);
    // Inside the interval of the three points only.
    let (lo, hi) = (x0.min(x2), x0.max(x2));
    if !(lo..=hi).contains(&xv) {
        return Some((xb, yb));
    }
    Some((xv, a * xv * xv + b * xv + c))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn mulliken_hush_and_its_generalization_agree_for_localized_states() {
        // Localized states: the dipole difference is the transfer distance (e = 1) and the
        // transition dipole is small; GMH tends to Mulliken-Hush.
        let (e, r) = (0.05, 15.0);
        let mu = [0.0, 0.0, 0.02];
        let mh = mulliken_hush(e, mu, r).unwrap();
        let gmh = generalized_mulliken_hush(e, mu, [0.0, 0.0, r]).unwrap();
        assert!((mh - e * 0.02 / r).abs() < 1e-15);
        assert!((gmh - mh).abs() < 1e-5 * mh, "{gmh} vs {mh}");
        // A delocalized pair at the crossing (no dipole difference along the transfer from
        // the populations alone) is what GMH handles: V12 = E / 2 when mu_12 dominates.
        let v = generalized_mulliken_hush(e, [0.0, 0.0, 5.0], [0.0, 0.0, 1e-6]).unwrap();
        assert!((v - 0.5 * e).abs() < 1e-9);
    }

    #[test]
    fn gmh_and_fcd_give_the_coupling_of_two_mixed_states() {
        // Two diabatic states, the electron on the donor (atom 0) or on the acceptor (atom 1,
        // r Bohr away), with the energy gap eps and the coupling V. The adiabatic states mix
        // by the angle theta, tan 2 theta = 2 V / eps, and lie sqrt(eps^2 + 4 V^2) apart:
        // their populations, transition density, and dipoles give V back with both formulas.
        let (eps, v, r): (f64, f64, f64) = (0.01, 0.004, 12.0);
        let gap = (eps * eps + 4.0 * v * v).sqrt();
        let theta = 0.5 * (2.0 * v).atan2(eps);
        let (c, s) = (theta.cos(), theta.sin());
        // Ground state c |D> + s |A>, excited -s |D> + c |A>.
        let change = [s * s - c * c, c * c - s * s];
        let transition = [-c * s, c * s];
        let fcd = fragment_charge_difference(gap, &transition, &change, &[0], &[1]).unwrap();
        assert!((fcd - v).abs() < 1e-15, "{fcd} vs {v}");
        // Dipoles of the electron (charge -1) at 0 or at r along z.
        let dipole_change = [0.0, 0.0, -r * (c * c - s * s)];
        let mu12 = [0.0, 0.0, -r * c * s];
        let gmh = generalized_mulliken_hush(gap, mu12, dipole_change).unwrap();
        assert!((gmh - v).abs() < 1e-15, "{gmh} vs {v}");
        // A bridge atom takes part in neither fragment.
        let fcd3 = fragment_charge_difference(
            gap,
            &[transition[0], 0.0, transition[1]],
            &[change[0], 0.0, change[1]],
            &[0],
            &[2],
        )
        .unwrap();
        assert!((fcd3 - v).abs() < 1e-15);
        assert!(fragment_charge_difference(gap, &transition, &change, &[0], &[0]).is_err());
    }

    #[test]
    fn the_charge_transfer_index_moves_one_electron() {
        // Nearly one electron from an atom at the origin to two 10 and 11 Bohr away: 0.98 of
        // an electron, below the threshold of a charge-transfer state.
        let positions = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]];
        let change = [-0.98, 0.6, 0.38];
        let ct = charge_transfer(&positions, &change).unwrap();
        assert!((ct.charge - 0.98).abs() < 1e-12);
        let xg = (0.6 * 10.0 + 0.38 * 11.0) / 0.98;
        assert!((ct.distance - xg).abs() < 1e-12);
        assert!(ct.charge < CT_THRESHOLD);
        // One electron moved whole is one.
        let whole = charge_transfer(&positions, &[-1.0, 1.0, 0.0]).unwrap();
        assert!((whole.charge - 1.0).abs() < 1e-15 && whole.charge >= CT_THRESHOLD);
        assert!((whole.distance - 10.0).abs() < 1e-12);
    }

    #[test]
    fn the_maximum_of_a_scan_is_refined_by_a_parabola() {
        // log10 k = 7 - 20 (alpha - 0.43)^2 sampled every 0.1.
        let pts: Vec<(f64, f64)> = (0..=10)
            .map(|i| {
                let a = 1.0 - 0.1 * f64::from(i);
                (a, 7.0 - 20.0 * (a - 0.43).powi(2))
            })
            .collect();
        let (x, y) = maximize_on_grid(&pts).unwrap();
        assert!(
            (x - 0.43).abs() < 1e-12 && (y - 7.0).abs() < 1e-12,
            "{x} {y}"
        );
        // At the end of the scan: that point.
        let edge = [(1.0, 1.0), (0.9, 2.0), (0.8, 3.0)];
        assert_eq!(maximize_on_grid(&edge), Some((0.8, 3.0)));
        assert_eq!(maximize_on_grid(&[]), None);
    }
}
