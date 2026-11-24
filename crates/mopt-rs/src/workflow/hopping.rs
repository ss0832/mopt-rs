//! The rate of a charge hop between two molecules from one structure of the pair: the transfer
//! integral of the projection method at that structure, the four-point reorganization energy
//! from the equilibria of both molecules in both charges, the reaction energy, and the Marcus
//! rates both ways (`kinetics::rate::transfer_integral`; `etmain -et_method hopping`;
//! `docs/source/electron_transfer.md`).
//!
//! For a hop of the carrier from `a` to `b` (each molecule is neutral, or of its charge
//! without the carrier, in the pair; the carrier makes it one unit more positive for a hole and
//! more negative for an electron):
//!
//! ```text
//! lambda_ab = [E_a^n(R_a^c) - E_a^n(R_a^n)] + [E_b^c(R_b^n) - E_b^c(R_b^c)]
//! lambda_ba = [E_b^n(R_b^c) - E_b^n(R_b^n)] + [E_a^c(R_a^n) - E_a^c(R_a^c)]
//! dE_ab     = [E_a^n(R_a^n) + E_b^c(R_b^c)] - [E_a^c(R_a^c) + E_b^n(R_b^n)]
//! ```
//!
//! (`n` without the carrier, `c` with it, `R` the equilibrium of the molecule alone). The four
//! points of each direction differ for two different molecules; both rates take their mean,
//! so that `k_ab / k_ba = exp(-dE_ab / k_B T)` holds, and both are reported. The energies are
//! electronic (no thermal corrections), the molecules each alone.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{AtomId, MolecularSystem, SpinCharge, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::kinetics::rate::transfer_integral::{
    fragment_transfer_integral, hopping_rate, Carrier, TransferIntegral,
};
use crate::opt::state_machine::OptimizerSettings;
use crate::pes::combine::{PesExpr, TaskSpec, DEFAULT_CALCULATOR};
use crate::workflow::driver::CalculatorSet;
use crate::workflow::electron_transfer::{apparent_barrier, evaluate, optimize};
use crate::workflow::Observer;

/// The settings of a hop.
#[derive(Debug, Clone)]
pub struct HoppingSettings {
    /// Temperature (K).
    pub temperature: f64,
    /// The carrier.
    pub carrier: Carrier,
    /// The charges of the two molecules without the carrier.
    pub charges: [i32; 2],
    /// The multiplicities of `a` without and with the carrier, then of `b`; the lowest ones
    /// when `None`.
    pub multiplicities: Option<[u32; 4]>,
    /// The optimizations of the molecules.
    pub optimizer: OptimizerSettings,
}

/// One molecule in both charges: its equilibria and the energies (Hartree) of each charge at
/// both equilibria.
#[derive(Debug, Clone)]
pub struct HoppingMolecule {
    /// The atoms of the molecule in the pair.
    pub atoms: Vec<AtomId>,
    /// The charge and multiplicity without and with the carrier.
    pub states: [SpinCharge; 2],
    /// The equilibria without and with the carrier.
    pub equilibria: [MolecularSystem; 2],
    /// Whether each optimization converged.
    pub converged: [bool; 2],
    /// `[[E_n(R_n), E_n(R_c)], [E_c(R_n), E_c(R_c)]]`.
    pub energies: [[f64; 2]; 2],
}

/// The result of a hop.
#[derive(Debug, Clone)]
pub struct HoppingResult {
    /// The transfer integral at the structure of the pair.
    pub coupling: TransferIntegral,
    /// The molecules `a` and `b`.
    pub molecules: [HoppingMolecule; 2],
    /// The four-point reorganization energies of the hops from `a` to `b` and back.
    pub reorganization: [f64; 2],
    /// The reaction energy of the hop from `a` to `b` (the opposite back).
    pub reaction_energy: f64,
    /// The Marcus rates (1/s) from `a` to `b` and back, with the mean reorganization energy.
    pub rates: [f64; 2],
    /// The apparent free energies of activation of the two rates (Hartree).
    pub apparent_barriers: [f64; 2],
}

impl HoppingResult {
    /// The mean reorganization energy that the rates take.
    pub fn mean_reorganization(&self) -> f64 {
        0.5 * (self.reorganization[0] + self.reorganization[1])
    }
}

/// The lowest multiplicity of `electrons` electrons.
fn lowest(electrons: i64) -> u32 {
    if electrons.rem_euclid(2) == 0 {
        1
    } else {
        2
    }
}

/// The number of electrons of the atoms `atoms` of `sys` in the charge `charge`.
fn electrons(sys: &MolecularSystem, atoms: &[AtomId], charge: i32) -> i64 {
    atoms
        .iter()
        .map(|a| i64::from(sys.element(*a).atomic_number()))
        .sum::<i64>()
        - i64::from(charge)
}

/// The energy of `sys` (in its charge and multiplicity) with the default calculator.
fn energy(sys: &MolecularSystem, calcs: &CalculatorSet<'_>) -> MoptResult<f64> {
    let mut expr = PesExpr::Task(TaskSpec::whole("molecule"));
    Ok(evaluate(&mut expr, sys, calcs)?.total.energy.value())
}

/// `sys` in the charge and multiplicity `sc`.
fn in_state(sys: &MolecularSystem, sc: SpinCharge) -> MolecularSystem {
    let mut out = sys.clone();
    out.spin_charge = sc;
    out
}

/// The hop of `settings.carrier` from the molecule `a` to the molecule `b` of `pair` (the two
/// together are every atom of the pair, which is in the charges of the molecules without the
/// carrier).
pub fn run_hopping(
    pair: &MolecularSystem,
    a: &[AtomId],
    b: &[AtomId],
    settings: &HoppingSettings,
    calcs: &CalculatorSet<'_>,
    observers: &mut [&mut dyn Observer],
) -> MoptResult<HoppingResult> {
    if !(settings.temperature.is_finite() && settings.temperature > 0.0) {
        return Err(MoptError::config(
            "charge hopping: the temperature must be positive",
        ));
    }
    if a.is_empty() || b.is_empty() || a.iter().any(|x| b.contains(x)) {
        return Err(MoptError::config(
            "charge hopping: two molecules of their own atoms are needed",
        ));
    }
    if a.len() + b.len() != pair.n_atoms() {
        return Err(MoptError::config(format!(
            "charge hopping: the two molecules have {} of the {} atoms of the pair (every atom \
             belongs to one of them)",
            a.len() + b.len(),
            pair.n_atoms()
        )));
    }
    let [qa, qb] = settings.charges;
    if pair.spin_charge.charge != qa + qb {
        return Err(MoptError::config(format!(
            "charge hopping: the pair (charge {}) is in the charges of the two molecules without \
             the carrier ({qa} and {qb})",
            pair.spin_charge.charge
        )));
    }
    let step = match settings.carrier {
        Carrier::Hole => 1,
        Carrier::Electron => -1,
    };
    let m = settings.multiplicities.unwrap_or([
        lowest(electrons(pair, a, qa)),
        lowest(electrons(pair, a, qa + step)),
        lowest(electrons(pair, b, qb)),
        lowest(electrons(pair, b, qb + step)),
    ]);
    let states_a = [SpinCharge::new(qa, m[0]), SpinCharge::new(qa + step, m[1])];
    let states_b = [SpinCharge::new(qb, m[2]), SpinCharge::new(qb + step, m[3])];

    // The transfer integral at the structure of the pair, from the orbitals of the pair and of
    // each molecule alone at its place, without the carrier.
    let calc = calcs.get(DEFAULT_CALCULATOR)?;
    let orbitals = |sys: &MolecularSystem, what: &str| {
        calc.orbitals(sys)?.ok_or_else(|| {
            MoptError::unsupported(format!(
                "charge hopping: the calculator gives no orbitals of {what} (the projection \
                 method needs them)"
            ))
        })
    };
    let pair_orbitals = orbitals(pair, "the pair")?;
    let alone_a = pair.subsystem(a, states_a[0])?;
    let alone_b = pair.subsystem(b, states_b[0])?;
    let index = |atoms: &[AtomId]| atoms.iter().map(|x| x.index()).collect::<Vec<usize>>();
    let coupling = fragment_transfer_integral(
        &pair_orbitals,
        &orbitals(&alone_a, "the first molecule")?,
        &orbitals(&alone_b, "the second molecule")?,
        &index(a),
        &index(b),
        settings.carrier,
    )
    .context("the transfer integral")?;

    // Each molecule in both charges: the equilibria from its structure in the pair, and each
    // charge at both equilibria.
    let mut molecule = |atoms: &[AtomId],
                        alone: &MolecularSystem,
                        states: [SpinCharge; 2],
                        name: &str|
     -> MoptResult<HoppingMolecule> {
        let mut equilibria = Vec::with_capacity(2);
        let mut converged = [false; 2];
        for (k, sc) in states.iter().enumerate() {
            let (eq, ok) = optimize(
                PesExpr::Task(TaskSpec::whole("molecule")),
                &in_state(alone, *sc),
                &settings.optimizer,
                calcs,
                observers,
            )
            .context(format!(
                "the equilibrium of {name} {}",
                if k == 0 {
                    "without the carrier"
                } else {
                    "with the carrier"
                }
            ))?;
            equilibria.push(eq);
            converged[k] = ok;
        }
        let mut energies = [[0.0; 2]; 2];
        for (i, sc) in states.iter().enumerate() {
            for (j, eq) in equilibria.iter().enumerate() {
                energies[i][j] = energy(&in_state(eq, *sc), calcs)?;
            }
        }
        let [n, c] = [equilibria[0].clone(), equilibria[1].clone()];
        Ok(HoppingMolecule {
            atoms: atoms.to_vec(),
            states,
            equilibria: [n, c],
            converged,
            energies,
        })
    };
    let ma = molecule(a, &alone_a, states_a, "the first molecule")?;
    let mb = molecule(b, &alone_b, states_b, "the second molecule")?;

    // [charge][equilibrium], 0 without the carrier, 1 with it.
    let (ea, eb) = (ma.energies, mb.energies);
    let lambda_ab = (ea[0][1] - ea[0][0]) + (eb[1][0] - eb[1][1]);
    let lambda_ba = (eb[0][1] - eb[0][0]) + (ea[1][0] - ea[1][1]);
    let reaction = (ea[0][0] + eb[1][1]) - (ea[1][1] + eb[0][0]);
    let mean = 0.5 * (lambda_ab + lambda_ba);
    if !(mean.is_finite() && mean > 0.0) {
        return Err(MoptError::calc(format!(
            "charge hopping: the reorganization energy is {:.4} eV (the equilibria of the two \
             charges did not come out apart)",
            mean * Units::HARTREE_TO_EV
        )));
    }
    let kt = Units::BOLTZMANN_HARTREE_PER_K * settings.temperature;
    let rates = [
        hopping_rate(coupling.effective, mean, reaction, kt)?,
        hopping_rate(coupling.effective, mean, -reaction, kt)?,
    ];
    let apparent_barriers = [
        apparent_barrier(rates[0], settings.temperature),
        apparent_barrier(rates[1], settings.temperature),
    ];
    Ok(HoppingResult {
        coupling,
        molecules: [ma, mb],
        reorganization: [lambda_ab, lambda_ba],
        reaction_energy: reaction,
        rates,
        apparent_barriers,
    })
}
