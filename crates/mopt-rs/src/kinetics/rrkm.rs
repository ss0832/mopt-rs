//! RRKM theory: microcanonical rate constants from directly counted states, and the rate theory
//! `rrkm` whose canonical rate constants are their Boltzmann averages.
//!
//! **Counting.** The rovibrational states of a structure are counted in cells of width `delta`
//! (`--rrkm-cell`, 1 cm^-1) above its lowest level `E0 = E + ZPE` (electronic energy and the
//! harmonic zero-point energy): the classical density of the external rotations
//! (`2 sqrt(E) / (sigma sqrt(A B C))` for a non-linear molecule, `1 / (sigma B)` for a linear
//! one, integrated over every cell) is convolved with the harmonic vibrations by the algorithm
//! of Beyer and Swinehart (Commun. ACM 1973, 16, 379; with an initial array other than one
//! state at zero, the extension of Stein and Rabinovitch, J. Chem. Phys. 1973, 58, 2438):
//! for every vibration of `m` cells, `N[j] += N[j - m]`. The vibrations are rounded to whole
//! cells; modes at or below 10 cm^-1 and imaginary modes are left out, as in the
//! thermochemistry.
//!
//! **Microcanonical rate constants** (Marcus, J. Chem. Phys. 1952, 20, 359):
//! `k(E) = (g_ts / g) W_ts(E - E0_ts) / (h rho(E))`, `W` the number of states of the saddle
//! point up to an energy (the cells below plus half of its own, so that sums over cells are
//! midpoint rules), `rho` the density of the minimum, `g` the electronic degeneracies.
//!
//! **The canonical rate constant** is the Boltzmann average of `k(E)`; in flux form
//! (`kinetics_beyond_tst.md`, sections 2.4 and 3.7):
//!
//! ```text
//! pi_w = exp(-(G_trans + G_elec) / kT) sum_j N_j exp(-E_j / kT)          (E_j = E0 + j delta)
//! F    = exp(-(G_trans,ts + G_elec,ts) / kT) (1 / h) sum_j W_j delta exp(-E_j / kT)
//! ```
//!
//! so the theory `rrkm` gives every state the free energy `-kT ln pi_w` and every saddle point
//! `-kT ln(h F / kT)`, and its rate constants equal the averages of the `k(E)` that the
//! energy-grained master equation (`egme`) uses, to rounding: the high-pressure limit of
//! `egme` is `rrkm`. Against `tst` with harmonic vibrations (`--qrrho none`) it differs by the
//! rounding of the vibrations to whole cells and by `(x / 2) coth(x / 2)`, `x = delta / kT`
//! (2e-6 at 300 K for 1 cm^-1 cells); the translations and the electronic degeneracy are those
//! of the thermochemistry. Like `tst` it needs the frequencies and the structure of every state
//! and saddle point.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::conditions::Conditions;
use super::network::{Bottleneck, KineticState};
use super::rate::tst::{SymmetryChoice, SYMMETRY_TOLERANCE};
use super::rate::RateTheory;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::stable::log_sum_exp;
use crate::vib::{
    point_group, thermochemistry, NormalModes, QuasiHarmonic, RotorType, ThermoSettings,
};

/// Vibrations at or below this wavenumber (cm^-1) are left out, as in the thermochemistry.
const LOWEST_MODE: f64 = 10.0;

/// The counting of states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrkmSettings {
    /// The width of a cell (cm^-1).
    pub cell: f64,
    /// The states are counted up to this many `k_B T` above the lowest level of each structure
    /// (the canonical sums; `egme` counts up to its own highest grain).
    pub range_kt: f64,
}

impl Default for RrkmSettings {
    fn default() -> Self {
        Self {
            cell: 1.0,
            range_kt: 60.0,
        }
    }
}

/// The counted states of one structure.
#[derive(Debug, Clone, PartialEq)]
pub struct StateCount {
    /// The lowest level, electronic energy plus harmonic zero-point energy (Hartree).
    pub e0: f64,
    /// The width of a cell (Hartree).
    pub cell: f64,
    /// The rovibrational states in every cell: cell `j` holds the levels nearest to
    /// `e0 + j cell` (between `e0 + (j - 1/2) cell` and `e0 + (j + 1/2) cell`).
    pub counts: Vec<f64>,
    /// `G_trans + G_elec` (Hartree) at the temperature of the count: the factors that the
    /// rovibrational sum does not hold.
    pub outer: f64,
    /// The electronic degeneracy.
    pub degeneracy: f64,
}

impl StateCount {
    /// `ln sum_j N_j exp(-(e0 + j cell - reference) / kt)`.
    pub fn ln_sum(&self, kt: f64, reference: f64) -> f64 {
        let terms: Vec<f64> = self
            .counts
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0.0)
            .map(|(j, n)| n.ln() - (self.e0 + j as f64 * self.cell - reference) / kt)
            .collect();
        log_sum_exp(&terms)
    }

    /// The number of states up to each cell, `W_j = sum_{i < j} N_i + N_j / 2`.
    pub fn sum_of_states(&self) -> Vec<f64> {
        let mut below = 0.0;
        self.counts
            .iter()
            .map(|n| {
                let w = below + 0.5 * n;
                below += n;
                w
            })
            .collect()
    }

    /// The number of states with energies up to `e` (absolute, Hartree), the states of a cell
    /// spread evenly over it (`states_up_to(e0 + j cell)` is `W_j`).
    pub fn states_up_to(&self, e: f64) -> f64 {
        let x = (e - self.e0) / self.cell + 0.5;
        if x.is_nan() || x <= 0.0 {
            return 0.0;
        }
        let full = x.floor() as usize;
        let total: f64 = self.counts.iter().take(full).sum();
        total + self.counts.get(full).copied().unwrap_or(0.0) * (x - full as f64)
    }

    /// The density of states (per Hartree) at `e`, averaged over the window `[e - width / 2,
    /// e + width / 2]`.
    pub fn density(&self, e: f64, width: f64) -> f64 {
        let w = width.max(self.cell);
        (self.states_up_to(e + 0.5 * w) - self.states_up_to(e - 0.5 * w)) / w
    }
}

/// `ln(1 / h)` with `h` in Hartree s, consistent with the prefactor `k_B T / h` of the rate
/// theories ([`Conditions::ln_kt_over_h`] over [`Conditions::kt`]).
pub fn ln_inverse_planck(c: &Conditions) -> f64 {
    c.ln_kt_over_h() - c.kt().ln()
}

/// The microcanonical rate constant (1/s) over the saddle point `ts` out of the minimum
/// `well` at the absolute energy `e` (Hartree), the density taken over the window `width`.
pub fn rate_at_energy(
    ts: &StateCount,
    well: &StateCount,
    e: f64,
    width: f64,
    c: &Conditions,
) -> f64 {
    let rho = well.density(e, width);
    if rho.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return 0.0;
    }
    let w = ts.states_up_to(e);
    // W / (h rho) with rho per Hartree.
    ts.degeneracy / well.degeneracy * w / rho * ln_inverse_planck(c).exp()
}

/// The classical rotor states in `n` cells of width `cell` (Hartree) for the rotational
/// constants `b` (Hartree, `A >= B >= C`), the symmetry number `sigma`, and the kind of rotor:
/// the states between `(j - 1/2) cell` and `(j + 1/2) cell` go to cell `j` (rounded to the
/// nearest cell, as the vibrations are), so that the sums over the cells are midpoint rules.
fn rotor_counts(rotor: RotorType, b: [f64; 3], sigma: f64, cell: f64, n: usize) -> Vec<f64> {
    let edge = |j: usize| if j == 0 { 0.0 } else { j as f64 - 0.5 };
    match rotor {
        RotorType::Atom => {
            let mut v = vec![0.0; n];
            if n > 0 {
                v[0] = 1.0;
            }
            v
        }
        RotorType::Linear => {
            // The finite constant (the infinite one belongs to the molecular axis).
            let bl = b
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(0.0, f64::max);
            (0..n)
                .map(|j| (edge(j + 1) - edge(j)) * cell / (sigma * bl))
                .collect()
        }
        RotorType::Regular => {
            let scale = 4.0 / 3.0 * cell.powf(1.5) / (sigma * (b[0] * b[1] * b[2]).sqrt());
            (0..n)
                .map(|j| scale * edge(j + 1).powf(1.5) - scale * edge(j).powf(1.5))
                .collect()
        }
    }
}

/// Beyer and Swinehart: fold the vibrations `wavenumbers` (cm^-1) into `counts` (cells of
/// `cell_cm1`).
pub fn beyer_swinehart(counts: &mut [f64], wavenumbers: &[f64], cell_cm1: f64) {
    let n = counts.len();
    for &w in wavenumbers {
        if w.partial_cmp(&LOWEST_MODE) != Some(std::cmp::Ordering::Greater) {
            continue;
        }
        let m = ((w / cell_cm1).round() as usize).max(1);
        for j in m..n {
            counts[j] += counts[j - m];
        }
    }
}

/// The rate theory `rrkm`.
#[derive(Debug, Clone, Copy)]
pub struct Rrkm {
    settings: RrkmSettings,
    symmetry: SymmetryChoice,
}

impl Rrkm {
    /// With `settings`, the symmetry numbers chosen as for `tst`.
    pub fn new(settings: RrkmSettings, symmetry: SymmetryChoice) -> MoptResult<Self> {
        if !(settings.cell > 0.0 && settings.cell.is_finite()) {
            return Err(MoptError::config(format!(
                "rrkm: the cell width must be positive (got {} cm^-1)",
                settings.cell
            )));
        }
        if !(settings.range_kt >= 10.0 && settings.range_kt.is_finite()) {
            return Err(MoptError::config(format!(
                "rrkm: the states must be counted at least 10 kT high (got {})",
                settings.range_kt
            )));
        }
        Ok(Self { settings, symmetry })
    }

    /// The settings.
    pub fn settings(&self) -> RrkmSettings {
        self.settings
    }

    /// Count the states of a structure in `cells` cells (none: up to `range_kt` k_B T).
    #[allow(clippy::too_many_arguments)]
    pub fn count(
        &self,
        system: &MolecularSystem,
        frequencies: &[f64],
        energy: f64,
        multiplicity: u32,
        symmetry_number: Option<u32>,
        c: &Conditions,
        cells: Option<usize>,
    ) -> MoptResult<StateCount> {
        let sigma = match (symmetry_number, self.symmetry) {
            (Some(s), _) => s,
            (None, SymmetryChoice::Auto) => point_group(system, SYMMETRY_TOLERANCE).symmetry_number,
            (None, SymmetryChoice::None) => 1,
        }
        .max(1);
        let settings = ThermoSettings {
            temperature: c.temperature,
            pressure: c.pressure,
            quasi_harmonic: QuasiHarmonic::None,
            symmetry_number: sigma,
            multiplicity: multiplicity.max(1),
            concentration: c.concentration,
            ..ThermoSettings::default()
        };
        let modes = NormalModes {
            wavenumbers: frequencies.to_vec(),
            curvatures: Vec::new(),
            displacements: Vec::new(),
            reduced_masses: Vec::new(),
            force_constants: Vec::new(),
            temperatures: Vec::new(),
        };
        let thermo = thermochemistry(system, &modes, energy, &settings)?;
        let cell = self.settings.cell * Units::WAVENUMBER_TO_HARTREE;
        let n = match cells {
            Some(n) => n.max(1),
            None => ((self.settings.range_kt * c.kt() / cell).ceil() as usize).max(2),
        };
        let ghz = 1e9 * Units::PLANCK_J_S / Units::HARTREE_TO_JOULE;
        let b = thermo.rotational_constants_ghz.map(|x| x * ghz);
        let mut counts = rotor_counts(thermo.rotor, b, sigma as f64, cell, n);
        let real: Vec<f64> = frequencies.iter().copied().filter(|w| *w > 0.0).collect();
        beyer_swinehart(&mut counts, &real, self.settings.cell);
        let zpe = 0.5
            * real.iter().filter(|w| **w > LOWEST_MODE).sum::<f64>()
            * Units::WAVENUMBER_TO_HARTREE;
        Ok(StateCount {
            e0: energy + zpe,
            cell,
            counts,
            // The electronic part of the Gibbs energy holds the electronic energy, which `e0`
            // holds already: only -T S_elec.
            outer: thermo.gibbs.translational + thermo.gibbs.electronic - energy,
            degeneracy: multiplicity.max(1) as f64,
        })
    }

    /// The counted states of a state of the network (`None`: no frequencies or structure).
    pub fn count_state(
        &self,
        s: &KineticState,
        c: &Conditions,
        cells: Option<usize>,
    ) -> MoptResult<Option<StateCount>> {
        match (&s.system, &s.frequencies) {
            (Some(sys), Some(f)) => self
                .count(
                    sys,
                    f,
                    s.energy,
                    s.multiplicity,
                    s.symmetry_number,
                    c,
                    cells,
                )
                .map(Some),
            _ => Ok(None),
        }
    }

    /// The counted states of a saddle point (`None`: no frequencies or structure).
    pub fn count_bottleneck(
        &self,
        b: &Bottleneck,
        c: &Conditions,
        cells: Option<usize>,
    ) -> MoptResult<Option<StateCount>> {
        match (&b.system, &b.frequencies) {
            (Some(sys), Some(f)) => self
                .count(
                    sys,
                    f,
                    b.energy,
                    b.multiplicity,
                    b.symmetry_number,
                    c,
                    cells,
                )
                .map(Some),
            _ => Ok(None),
        }
    }
}

/// `-kT ln pi` (Hartree) of counted states.
pub fn state_free_energy(count: &StateCount, kt: f64) -> f64 {
    count.outer - kt * count.ln_sum(kt, 0.0)
}

/// `-kT ln(h F / kT)` (Hartree) of the counted states of a saddle point.
pub fn bottleneck_free_energy(count: &StateCount, kt: f64) -> f64 {
    let w = count.sum_of_states();
    let terms: Vec<f64> = w
        .iter()
        .enumerate()
        .filter(|(_, x)| **x > 0.0)
        .map(|(j, x)| x.ln() - (count.e0 + j as f64 * count.cell) / kt)
        .collect();
    count.outer - kt * (log_sum_exp(&terms) + (count.cell / kt).ln())
}

impl RateTheory for Rrkm {
    fn name(&self) -> &str {
        "rrkm"
    }

    fn state_energy(&self, s: &KineticState, c: &Conditions) -> MoptResult<Option<f64>> {
        Ok(self
            .count_state(s, c, None)?
            .map(|n| state_free_energy(&n, c.kt())))
    }

    fn bottleneck_energy(&self, b: &Bottleneck, c: &Conditions) -> MoptResult<Option<f64>> {
        Ok(self
            .count_bottleneck(b, c, None)?
            .map(|n| bottleneck_free_energy(&n, c.kt())))
    }

    fn describe(&self, _c: &Conditions) -> String {
        format!(
            "free energy from directly counted states (Beyer-Swinehart in {} cm^-1 cells, \
             classical rotors, symmetry numbers {})",
            self.settings.cell,
            match self.symmetry {
                SymmetryChoice::Auto => "from the point groups",
                SymmetryChoice::None => "1",
            }
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::SpinCharge;
    use crate::io::xyz::parse_xyz_str;
    use crate::kinetics::rate::tst::{QrrhoSettings, Tst};

    #[test]
    fn the_direct_count_equals_an_explicit_enumeration() {
        // Three oscillators of 2, 3, and 5 cells: count every (n1, n2, n3) with 2 n1 + 3 n2 + 5
        // n3 = j.
        let n = 40;
        let mut counts = vec![0.0; n];
        counts[0] = 1.0;
        beyer_swinehart(&mut counts, &[20.0, 30.0, 50.0], 10.0);
        for (j, c) in counts.iter().enumerate() {
            let mut want = 0;
            for a in 0..=j / 2 {
                for b in 0..=j / 3 {
                    for d in 0..=j / 5 {
                        if 2 * a + 3 * b + 5 * d == j {
                            want += 1;
                        }
                    }
                }
            }
            assert_eq!(*c, want as f64, "cell {j}");
        }
    }

    #[test]
    fn the_boltzmann_average_of_k_e_is_the_canonical_rate_and_close_to_tst() {
        let water = parse_xyz_str(
            "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let c = Conditions {
            temperature: 600.0,
            ..Conditions::default()
        };
        let kt = c.kt();
        let theory = Rrkm::new(RrkmSettings::default(), SymmetryChoice::Auto).unwrap();
        // Frequencies that are whole cells: the count is exact for the harmonic levels.
        let well = theory
            .count(&water, &[1595.0, 3657.0, 3756.0], 0.0, 1, None, &c, None)
            .unwrap();
        let ts = theory
            .count(&water, &[1200.0, 3000.0, -900.0], 0.03, 1, None, &c, None)
            .unwrap();
        // sum_j k(E_j) pi_j = F: with the densities of single cells (width = one cell) the
        // average is a sum over the cells of the well.
        let w = ts.sum_of_states();
        let mut terms = Vec::new();
        for (j, n) in w.iter().enumerate() {
            if *n > 0.0 {
                terms.push(
                    n.ln() + ts.cell.ln() + ln_inverse_planck(&c)
                        - (ts.e0 + j as f64 * ts.cell) / kt,
                );
            }
        }
        let ln_flux = log_sum_exp(&terms) - ts.outer / kt;
        let ln_k = ln_flux - (well.ln_sum(kt, 0.0) - well.outer / kt);
        let g_w = state_free_energy(&well, kt);
        let g_ts = bottleneck_free_energy(&ts, kt);
        let ln_k_theory = c.ln_kt_over_h() - (g_ts - g_w) / kt;
        assert!((ln_k - ln_k_theory).abs() < 1e-12, "{ln_k} {ln_k_theory}");
        // Against tst with harmonic oscillators: the free energies agree to the discretization.
        let tst = Tst::new(QrrhoSettings {
            method: QuasiHarmonic::None,
            ..QrrhoSettings::default()
        });
        let g_tst = tst
            .gibbs(&water, &[1595.0, 3657.0, 3756.0], 0.0, 1, None, &c)
            .unwrap();
        assert!((g_w - g_tst).abs() < 1e-3 * kt, "{} kT", (g_w - g_tst) / kt);
        let g_tst_ts = tst
            .gibbs(&water, &[1200.0, 3000.0, -900.0], 0.03, 1, None, &c)
            .unwrap();
        assert!(
            (g_ts - g_tst_ts).abs() < 1e-3 * kt,
            "{} kT",
            (g_ts - g_tst_ts) / kt
        );
        // k(E) is zero below the saddle point and grows above it.
        assert_eq!(
            rate_at_energy(&ts, &well, ts.e0 - 1e-4, 10.0 * well.cell, &c),
            0.0
        );
        let k1 = rate_at_energy(&ts, &well, ts.e0 + 0.01, 10.0 * well.cell, &c);
        let k2 = rate_at_energy(&ts, &well, ts.e0 + 0.02, 10.0 * well.cell, &c);
        assert!(k1 > 0.0 && k2 > k1);
    }
}
