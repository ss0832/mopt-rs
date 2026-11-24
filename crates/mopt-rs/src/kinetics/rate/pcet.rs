//! The parts of a proton-coupled electron transfer (PCET) rate in the nonadiabatic vibronic
//! picture: the vibrational states of the proton on a grid for the potential of each
//! electron-proton diabatic state, their overlaps and energies, and the rate as a sum over the
//! pairs of proton states.
//!
//! **Proton states** on a uniform grid by the discrete variable representation of D. T.
//! Colbert and W. H. Miller (J. Chem. Phys. 1992, 96, 1982): the kinetic energy of a particle of
//! mass `m` on the points `x_i = x_0 + i dx` is
//!
//! ```text
//! T_ii = hbar^2 pi^2 / (6 m dx^2),
//! T_ij = hbar^2 (-1)^(i-j) / (m dx^2 (i-j)^2)   (i != j),
//! ```
//!
//! and `H = T + diag(V(x_i))` gives the states as the eigenvectors (the values at the points
//! times `sqrt(dx)`), so that the overlap of two states of two potentials on the same grid is
//! the dot product of their vectors.
//!
//! **The rate** (the golden rule with the solvent classical and the proton states quantized,
//! the vibronic states `mu` of the reactant and `nu` of the product; R. I. Cukier, J. Phys.
//! Chem. 1996, 100, 15428; S. Hammes-Schiffer, A. A. Stuchebrukhov, Chem. Rev. 2010, 110, 6939,
//! the fixed proton donor-acceptor distance):
//!
//! ```text
//! k = sum_mu P_mu sum_nu (2 pi / hbar) |V_el S_mu,nu|^2 (4 pi lambda k_B T)^(-1/2)
//!     exp(-(dG_mu,nu + lambda)^2 / (4 lambda k_B T)),
//! dG_mu,nu = (E_nu^II - E_mu^I) + dG_0,
//! ```
//!
//! with the Boltzmann populations `P_mu` of the reactant's proton states, the electronic
//! coupling `V_el`, the overlaps `S_mu,nu` of the proton states, the solvent reorganization
//! energy `lambda`, and the energies `E` of the proton states in their potentials (which carry
//! the electronic energies of their diabatic states), shifted by the free energy `dG_0` that the
//! potentials do not hold (the solvent's). The sum is taken in logarithms. The fluctuation of
//! the proton donor-acceptor distance (the R mode) is not included.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;
use crate::math::stable::log_sum_exp;

/// The mass of the proton in atomic mass units.
pub const PROTON_MASS_AMU: f64 = 1.007_276_466_621;

/// The mass of the deuteron in atomic mass units.
pub const DEUTERON_MASS_AMU: f64 = 2.013_553_212_745;

/// The vibrational states of a particle in a potential on a uniform grid.
#[derive(Debug, Clone, PartialEq)]
pub struct ProtonStates {
    /// The energies (Hartree), lowest first.
    pub energies: Vec<f64>,
    /// The states, a column each: the values at the points times `sqrt(dx)` (unit length).
    pub vectors: Mat<f64>,
    /// The first point and the spacing of the grid (Bohr).
    pub grid: (f64, f64),
}

impl ProtonStates {
    /// The number of states.
    pub fn len(&self) -> usize {
        self.energies.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }

    /// The expectation value of the position in the state `k` (Bohr).
    pub fn mean_position(&self, k: usize) -> f64 {
        let (x0, dx) = self.grid;
        (0..self.vectors.nrows())
            .map(|i| self.vectors[(i, k)].powi(2) * (x0 + dx * i as f64))
            .sum()
    }
}

/// The lowest `count` states of a particle of mass `mass_amu` in the potential `potential`
/// (Hartree) at the points `x_0 + i dx` (Bohr).
pub fn proton_states(
    first: f64,
    spacing: f64,
    potential: &[f64],
    mass_amu: f64,
    count: usize,
) -> MoptResult<ProtonStates> {
    let n = potential.len();
    if n < 3 || !(spacing.is_finite() && spacing > 0.0) || !first.is_finite() {
        return Err(MoptError::config(
            "proton states: three or more points of a positive spacing",
        ));
    }
    if !(mass_amu.is_finite() && mass_amu > 0.0) || count == 0 || count > n {
        return Err(MoptError::config(format!(
            "proton states: a positive mass and 1 to {n} states"
        )));
    }
    if potential.iter().any(|v| !v.is_finite()) {
        return Err(MoptError::config(
            "proton states: the potential is not finite",
        ));
    }
    let mass = mass_amu * Units::AMU_TO_AU_MASS;
    let scale = 1.0 / (mass * spacing * spacing);
    let h = Mat::from_fn(n, n, |i, j| {
        if i == j {
            scale * PI * PI / 6.0 + potential[i]
        } else {
            let d = i as f64 - j as f64;
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            scale * sign / (d * d)
        }
    });
    let (values, vectors) = symmetric_eigen(&h)?;
    // A fixed sign: the largest value of each state positive.
    let mut out = Mat::<f64>::zeros(n, count);
    for k in 0..count {
        let col = vectors.col(k);
        let big = (0..n)
            .max_by(|&a, &b| col[a].abs().total_cmp(&col[b].abs()))
            .unwrap_or(0);
        let sign = if col[big] < 0.0 { -1.0 } else { 1.0 };
        for i in 0..n {
            out[(i, k)] = sign * col[i];
        }
    }
    Ok(ProtonStates {
        energies: (0..count).map(|k| values[k]).collect(),
        vectors: out,
        grid: (first, spacing),
    })
}

/// The overlaps `S_mu,nu` of the states `a` (rows) and `b` (columns) on the same grid.
pub fn proton_overlaps(a: &ProtonStates, b: &ProtonStates) -> MoptResult<Mat<f64>> {
    let same = a.vectors.nrows() == b.vectors.nrows()
        && (a.grid.0 - b.grid.0).abs() <= 1e-12 * a.grid.1
        && (a.grid.1 - b.grid.1).abs() <= 1e-12 * a.grid.1;
    if !same {
        return Err(MoptError::config(
            "proton overlaps: the two sets of states are on different grids",
        ));
    }
    Ok(a.vectors.transpose() * &b.vectors)
}

/// A PCET rate and the pairs of proton states that carry it.
#[derive(Debug, Clone, PartialEq)]
pub struct PcetRate {
    /// The rate (1/s).
    pub rate: f64,
    /// `(mu, nu, share)` of every pair of proton states, the largest share first.
    pub channels: Vec<(usize, usize, f64)>,
}

/// The nonadiabatic PCET rate (1/s) from the proton states of the reactant and of the product
/// (on the same grid, their energies with the electronic energies of the diabatic states), the
/// electronic coupling `coupling`, the solvent reorganization energy `reorganization`, the free
/// energy `delta_g0` the potentials do not hold (Hartree), at `kt` (Hartree).
pub fn pcet_rate(
    reactant: &ProtonStates,
    product: &ProtonStates,
    coupling: f64,
    reorganization: f64,
    delta_g0: f64,
    kt: f64,
) -> MoptResult<PcetRate> {
    if !(reorganization.is_finite() && reorganization > 0.0 && kt.is_finite() && kt > 0.0) {
        return Err(MoptError::config(
            "PCET rate: the reorganization energy and k_B T must be positive",
        ));
    }
    if reactant.is_empty() || product.is_empty() {
        return Err(MoptError::config("PCET rate: no proton states"));
    }
    let s = proton_overlaps(reactant, product)?;
    // ln P_mu, normalized over the states given.
    let lowest = reactant.energies[0];
    let weights: Vec<f64> = reactant
        .energies
        .iter()
        .map(|e| -(e - lowest) / kt)
        .collect();
    let ln_z = log_sum_exp(&weights);
    let prefactor = (2.0 * PI / Units::HBAR_HARTREE_S).ln() + 2.0 * coupling.abs().ln()
        - 0.5 * (4.0 * PI * reorganization * kt).ln();
    let mut terms = Vec::with_capacity(reactant.len() * product.len());
    for (mu, &w) in weights.iter().enumerate() {
        for nu in 0..product.len() {
            let overlap = s[(mu, nu)];
            if overlap == 0.0 {
                continue;
            }
            let dg = product.energies[nu] - reactant.energies[mu] + delta_g0;
            let ln_k = prefactor + (w - ln_z) + 2.0 * overlap.abs().ln()
                - (dg + reorganization).powi(2) / (4.0 * reorganization * kt);
            terms.push((mu, nu, ln_k));
        }
    }
    if terms.is_empty() || coupling == 0.0 {
        return Ok(PcetRate {
            rate: 0.0,
            channels: Vec::new(),
        });
    }
    let logs: Vec<f64> = terms.iter().map(|t| t.2).collect();
    let ln_rate = log_sum_exp(&logs);
    let mut channels: Vec<(usize, usize, f64)> = terms
        .iter()
        .map(|&(mu, nu, l)| (mu, nu, (l - ln_rate).exp()))
        .collect();
    channels.sort_by(|a, b| b.2.total_cmp(&a.2));
    Ok(PcetRate {
        rate: ln_rate.exp(),
        channels,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A harmonic potential `m w^2 (x - c)^2 / 2 + e0` on `n` points from `-l` to `l`.
    fn harmonic(mass_amu: f64, w: f64, c: f64, e0: f64, l: f64, n: usize) -> (f64, f64, Vec<f64>) {
        let m = mass_amu * Units::AMU_TO_AU_MASS;
        let dx = 2.0 * l / (n - 1) as f64;
        let v = (0..n)
            .map(|i| {
                let x = -l + dx * i as f64;
                0.5 * m * w * w * (x - c).powi(2) + e0
            })
            .collect();
        (-l, dx, v)
    }

    #[test]
    fn a_harmonic_proton_has_its_ladder() {
        // A proton at 3000 cm^-1.
        let w = 3000.0 * Units::WAVENUMBER_TO_HARTREE;
        let (x0, dx, v) = harmonic(PROTON_MASS_AMU, w, 0.0, 0.0, 2.0, 161);
        let st = proton_states(x0, dx, &v, PROTON_MASS_AMU, 6).unwrap();
        for (k, e) in st.energies.iter().enumerate() {
            let want = w * (k as f64 + 0.5);
            assert!((e - want).abs() < 1e-9, "{k}: {e} vs {want}");
        }
        // Unit length and orthogonal.
        let s = proton_overlaps(&st, &st).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s[(i, j)] - want).abs() < 1e-12);
            }
        }
        assert!(st.mean_position(0).abs() < 1e-9);
    }

    #[test]
    fn displaced_oscillators_have_their_franck_condon_factors() {
        // Two equal oscillators displaced by d: |<0|n'>|^2 = e^-S S^n / n!, S = m w d^2 / 2.
        let w = 2500.0 * Units::WAVENUMBER_TO_HARTREE;
        let d = 0.25;
        let (x0, dx, va) = harmonic(PROTON_MASS_AMU, w, 0.0, 0.0, 2.5, 201);
        let (_, _, vb) = harmonic(PROTON_MASS_AMU, w, d, 0.0, 2.5, 201);
        let a = proton_states(x0, dx, &va, PROTON_MASS_AMU, 4).unwrap();
        let b = proton_states(x0, dx, &vb, PROTON_MASS_AMU, 6).unwrap();
        let s = proton_overlaps(&a, &b).unwrap();
        let huang_rhys = 0.5 * PROTON_MASS_AMU * Units::AMU_TO_AU_MASS * w * d * d;
        let mut factorial = 1.0;
        for n in 0..6 {
            if n > 0 {
                factorial *= n as f64;
            }
            let want = (-huang_rhys).exp() * huang_rhys.powi(n as i32) / factorial;
            assert!((s[(0, n)].powi(2) - want).abs() < 1e-10, "{n}");
        }
        assert!((b.mean_position(0) - d).abs() < 1e-8);
    }

    #[test]
    fn one_channel_is_the_marcus_rate_and_deuterium_tunnels_less() {
        let kt = Units::BOLTZMANN_HARTREE_PER_K * 298.15;
        let w = 3000.0 * Units::WAVENUMBER_TO_HARTREE;
        let (coupling, lambda) = (0.002, 0.03);
        let states = |mass: f64, c: f64, e0: f64| {
            let (x0, dx, v) = harmonic(mass, w * (PROTON_MASS_AMU / mass).sqrt(), c, e0, 2.5, 201);
            proton_states(x0, dx, &v, mass, 4).unwrap()
        };
        // The same potential on both sides: only nu = mu couple, and the ground pair carries
        // the rate (the excited states are not populated): Marcus with the electronic coupling.
        let a = states(PROTON_MASS_AMU, 0.0, 0.0);
        let k = pcet_rate(&a, &a, coupling, lambda, 0.0, kt).unwrap();
        let marcus = super::super::inner_sphere::marcus_with_tunnelling(
            0.0,
            coupling,
            lambda,
            lambda / 4.0,
            kt,
        )
        .unwrap();
        assert!(
            (k.rate / marcus - 1.0).abs() < 1e-6,
            "{} vs {marcus}",
            k.rate
        );
        assert_eq!((k.channels[0].0, k.channels[0].1), (0, 0));
        // The proton moves by 0.6 Bohr: the overlaps shrink, more for the heavier deuteron
        // (the same force constant), so the rate of H exceeds that of D.
        let (h_r, h_p) = (
            states(PROTON_MASS_AMU, 0.0, 0.0),
            states(PROTON_MASS_AMU, 0.6, 0.0),
        );
        let (d_r, d_p) = (
            states(DEUTERON_MASS_AMU, 0.0, 0.0),
            states(DEUTERON_MASS_AMU, 0.6, 0.0),
        );
        let kh = pcet_rate(&h_r, &h_p, coupling, lambda, 0.0, kt)
            .unwrap()
            .rate;
        let kd = pcet_rate(&d_r, &d_p, coupling, lambda, 0.0, kt)
            .unwrap()
            .rate;
        assert!(kh > 2.0 * kd, "KIE {}", kh / kd);
        assert!(kh < k.rate);
        // Shares add up to one.
        let total: f64 = pcet_rate(&h_r, &h_p, coupling, lambda, 0.0, kt)
            .unwrap()
            .channels
            .iter()
            .map(|c| c.2)
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(proton_states(0.0, 0.1, &[0.0, 1.0], PROTON_MASS_AMU, 1).is_err());
    }
}
