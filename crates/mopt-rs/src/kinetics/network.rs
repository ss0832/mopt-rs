//! The kinetic network: states (minima) with what is known about them, and bottlenecks (saddle
//! points and crossing points of two electronic states) between two states. Pure data; the
//! readers of the network files are in [`crate::workflow::kinetics`].
//!
//! Energies are in Hartree, wavenumbers in cm^-1 (imaginary ones negative), couplings in
//! Hartree, gradient differences in Hartree/Bohr, reduced masses in amu.
//!
//! MultiOptPy: `Utils/rcmc.py:main` (`_Node`, `_Edge`: a node id and energies only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::rate::electron_transfer::TransferStep;
use super::rate::encounter::AssociationStep;
use super::species::Association;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

/// A state of the kinetic model: a minimum of the network.
#[derive(Debug, Clone)]
pub struct KineticState {
    /// Its number in the network file (`node_id`, or the node number of mopt-rs).
    pub id: usize,
    /// Its label (`EQ3`).
    pub label: String,
    /// Electronic energy.
    pub energy: f64,
    /// Gibbs free energy, where the file gives one.
    pub free_energy: Option<f64>,
    /// Vibrational wavenumbers, where they are known.
    pub frequencies: Option<Vec<f64>>,
    /// The structure, where the file gives it.
    pub system: Option<MolecularSystem>,
    /// Spin multiplicity.
    pub multiplicity: u32,
    /// Rotational symmetry number, where it is given (otherwise the rate theory finds it
    /// from the structure or takes 1).
    pub symmetry_number: Option<u32>,
    /// The fragments of a dissociation channel (empty for one molecule): the state is then
    /// the sum of its fragments, each a free molecule at the standard state.
    pub fragments: Vec<KineticState>,
    /// What the models and analyses beyond the canonical one need of the state.
    pub extra: StateExtra,
}

/// What a state may carry for the models and analyses that need more than its energies.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StateExtra {
    /// `semi_markov`: the shape `k` of the gamma distribution of its residence time (1: the
    /// exponential of a Markov state; the coefficient of variation is `1 / sqrt(k)`).
    pub residence_shape: Option<f64>,
    /// `egme`: the Lennard-Jones diameter (Angstrom) and well depth (K) of the molecule for
    /// its collisions with the bath gas.
    pub lennard_jones: Option<(f64, f64)>,
    /// The standard deviation of its energy (Hartree) for the ranking of refinement tasks.
    pub uncertainty: Option<f64>,
}

impl KineticState {
    /// A state with an electronic energy only.
    pub fn with_energy(id: usize, label: &str, energy: f64) -> Self {
        Self {
            id,
            label: label.to_string(),
            energy,
            free_energy: None,
            frequencies: None,
            system: None,
            multiplicity: 1,
            symmetry_number: None,
            fragments: Vec::new(),
            extra: StateExtra::default(),
        }
    }
}

/// The friction on the reaction coordinate at a barrier top (mass-weighted, so that `mu = 1`):
/// a Markovian friction coefficient for `kramers`, an exponential memory or a tabulated kernel
/// for `grote_hynes`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Friction {
    /// The friction coefficient `gamma` (1/s): Kramers' friction, or the integral of the
    /// kernel of an exponential memory.
    pub gamma: Option<f64>,
    /// The memory time `tau_c` (s) of the exponential kernel
    /// `zeta(t) = (gamma / tau_c) exp(-t / tau_c)`.
    pub memory: Option<f64>,
    /// A tabulated kernel `zeta(t)` (times in s, values in 1/s^2), e.g. the force
    /// autocorrelation of constrained dynamics at the barrier top.
    pub kernel: Option<(Vec<f64>, Vec<f64>)>,
    /// Where it came from.
    pub source: String,
}

/// The vibrationally adiabatic ground-state barrier along a reaction path and the effective
/// masses of small-curvature tunnelling, as `freqmain -path` writes them (`path_profile.csv`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TunnellingPath {
    /// Arc length (sqrt(amu) Bohr).
    pub arc_length: Vec<f64>,
    /// `V_a^G` at every point (Hartree).
    pub adiabatic: Vec<f64>,
    /// `mu_eff / mu` of small-curvature tunnelling at every point.
    pub mass_ratio: Vec<f64>,
    /// Where it was read from.
    pub source: String,
}

/// What kind of bottleneck a step has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckKind {
    /// A first-order saddle point on one surface.
    Saddle,
    /// A minimum-energy crossing point of two electronic states.
    Crossing,
    /// None: a barrierless dissociation of a complex into a dissociation channel, whose rate
    /// comes from the association ([`Bottleneck::association`]).
    Barrierless,
}

impl BottleneckKind {
    /// `saddle`, `crossing`, or `barrierless`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Saddle => "saddle",
            Self::Crossing => "crossing",
            Self::Barrierless => "barrierless",
        }
    }
}

/// A species outside the network that a bimolecular step takes up or gives off (a substrate,
/// a reagent, a product released): what is known about it, as for a state (its electronic
/// energy, its free energy at the standard state of the free energies, frequencies and
/// structure for the rate theory `tst`), and the treatment of its association, where it has
/// its own.
#[derive(Debug, Clone)]
pub struct Species {
    /// Its name.
    pub name: String,
    /// Its energies and structure (`energy` is NaN when only a free energy is given).
    pub data: KineticState,
    /// Its own treatment of the association entropy (`None`: that of the settings).
    pub association: Option<Association>,
}

impl Species {
    /// A species with an electronic energy and, where given, a free energy.
    pub fn new(name: &str, energy: Option<f64>, free_energy: Option<f64>) -> Self {
        let mut data = KineticState::with_energy(0, name, energy.unwrap_or(f64::NAN));
        data.free_energy = free_energy;
        Self {
            name: name.to_string(),
            data,
            association: None,
        }
    }
}

/// The partner of a bimolecular step: `state + species <-> other state`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partner {
    /// The species (index into [`KineticNetwork::species`]).
    pub species: usize,
    /// Which of the two states of the step is the one with the species free (0 or 1).
    pub free_side: usize,
}

/// A bottleneck between two states.
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Its number in the network file (`edge_id`, or the edge number of mopt-rs).
    pub id: usize,
    /// Its label (`TS12`, `MECP4`).
    pub label: String,
    /// Saddle point or crossing point.
    pub kind: BottleneckKind,
    /// The two states (indices into [`KineticNetwork::states`]).
    pub states: [usize; 2],
    /// Electronic energy.
    pub energy: f64,
    /// Gibbs free energy, where the file gives one.
    pub free_energy: Option<f64>,
    /// Vibrational wavenumbers (the imaginary one included), where they are known.
    pub frequencies: Option<Vec<f64>>,
    /// The imaginary wavenumber of the saddle point (negative), where it is known.
    pub imaginary_wavenumber: Option<f64>,
    /// The structure, where the file gives it.
    pub system: Option<MolecularSystem>,
    /// Spin multiplicity (of the lower state at a crossing).
    pub multiplicity: u32,
    /// Rotational symmetry number, where it is given.
    pub symmetry_number: Option<u32>,
    /// Coupling of the two states at a crossing (Hartree), where it is given.
    pub coupling: Option<f64>,
    /// Norm of the difference of the gradients of the two states at a crossing.
    pub gradient_difference: Option<f64>,
    /// Product of the slopes of the two states along the hopping coordinate at a crossing
    /// (Hartree^2/Bohr^2): positive for a sloped crossing, negative for a peaked one.
    pub slope_product: Option<f64>,
    /// Reduced mass along the hopping coordinate of a crossing (amu).
    pub reduced_mass: Option<f64>,
    /// A transmission coefficient given for this step.
    pub kappa: Option<f64>,
    /// The share of the flux through the bottleneck that goes between these two states (`None`:
    /// all of it): one effective step of an N-furcation, where the IRC of the saddle point runs
    /// into another saddle point and its flux divides among the products beyond. Both
    /// directions are multiplied by it, so the step keeps detailed balance.
    pub share: Option<f64>,
    /// The partner species of a bimolecular step (`None`: unimolecular).
    pub partner: Option<Partner>,
    /// An electron-transfer step (Marcus): its rate comes from the reorganization energy and
    /// the coupling instead of a saddle point.
    pub transfer: Option<TransferStep>,
    /// The reaction path through the saddle point for canonical variational TST.
    pub path: Option<CvtPath>,
    /// A barrierless step between a complex and a dissociation channel: its rate comes from
    /// the association of the two fragments instead of a bottleneck energy.
    pub association: Option<AssociationStep>,
    /// The friction at the barrier top (`kramers`, `grote_hynes`).
    pub friction: Option<Friction>,
    /// The transmission coefficient of the dividing surface at the saddle point from
    /// trajectories (`reactive_flux`): the value and its standard error.
    pub reactive_flux: Option<(f64, f64)>,
    /// The adiabatic barrier and effective masses along the path (`small_curvature`).
    pub tunnelling_path: Option<TunnellingPath>,
    /// The standard deviation of its energy (Hartree) for the ranking of refinement tasks.
    pub uncertainty: Option<f64>,
    /// The standard error of `share` where it was measured (trajectories from the saddle
    /// point); `None` for a share that is assumed.
    pub share_error: Option<f64>,
}

/// A reaction path through a saddle point for canonical variational TST: the arc length, the
/// energy, and either the transverse frequencies (the generalized free energy at any
/// temperature) or the generalized free energy at one temperature, as `freqmain -path` gives
/// them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CvtPath {
    /// Arc length (sqrt(amu) Bohr).
    pub arc_length: Vec<f64>,
    /// Energy along the path (Hartree).
    pub energies: Vec<f64>,
    /// Real wavenumbers of the transverse modes at every point (cm^-1).
    pub frequencies: Option<Vec<Vec<f64>>>,
    /// The generalized free energy at every point (Hartree) at `temperature`.
    pub free_energies: Option<Vec<f64>>,
    /// The temperature of `free_energies` (K).
    pub temperature: Option<f64>,
    /// Where it was read from.
    pub source: String,
}

impl Bottleneck {
    /// A saddle point between `states` with an electronic energy only.
    pub fn saddle(id: usize, label: &str, states: [usize; 2], energy: f64) -> Self {
        Self {
            id,
            label: label.to_string(),
            kind: BottleneckKind::Saddle,
            states,
            energy,
            free_energy: None,
            frequencies: None,
            imaginary_wavenumber: None,
            system: None,
            multiplicity: 1,
            symmetry_number: None,
            coupling: None,
            gradient_difference: None,
            slope_product: None,
            reduced_mass: None,
            kappa: None,
            share: None,
            partner: None,
            transfer: None,
            path: None,
            association: None,
            friction: None,
            reactive_flux: None,
            tunnelling_path: None,
            uncertainty: None,
            share_error: None,
        }
    }
}

/// The states and bottlenecks of a network, with what was left out when it was read.
#[derive(Debug, Clone, Default)]
pub struct KineticNetwork {
    /// The states.
    pub states: Vec<KineticState>,
    /// The bottlenecks.
    pub bottlenecks: Vec<Bottleneck>,
    /// The species that bimolecular steps take up or give off.
    pub species: Vec<Species>,
    /// What the network was read from.
    pub source: String,
    /// What was left out or assumed when it was read (for the log and the outputs).
    pub notes: Vec<String>,
}

impl KineticNetwork {
    /// The index of the state with the number `id` in the file.
    pub fn index_of(&self, id: usize) -> Option<usize> {
        self.states.iter().position(|s| s.id == id)
    }

    /// Check the indices and the energies.
    pub fn check(&self) -> MoptResult<()> {
        if self.states.is_empty() {
            return Err(MoptError::config("the kinetic network has no state"));
        }
        for s in &self.states {
            if !s.energy.is_finite() || s.free_energy.is_some_and(|g| !g.is_finite()) {
                return Err(MoptError::config(format!(
                    "{}: the energy is not a finite number",
                    s.label
                )));
            }
        }
        for b in &self.bottlenecks {
            let [x, y] = b.states;
            if x >= self.states.len() || y >= self.states.len() {
                return Err(MoptError::config(format!(
                    "{}: a state of the bottleneck is not in the network",
                    b.label
                )));
            }
            if x == y {
                return Err(MoptError::config(format!(
                    "{}: a bottleneck joins two different states",
                    b.label
                )));
            }
            if !b.energy.is_finite() || b.free_energy.is_some_and(|g| !g.is_finite()) {
                return Err(MoptError::config(format!(
                    "{}: the energy is not a finite number",
                    b.label
                )));
            }
            if let Some(p) = &b.partner {
                if p.species >= self.species.len() || p.free_side > 1 {
                    return Err(MoptError::config(format!(
                        "{}: the partner of the bimolecular step is not a species of the network",
                        b.label
                    )));
                }
            }
        }
        Ok(())
    }

    /// The index of the species `name`.
    pub fn species_index(&self, name: &str) -> Option<usize> {
        self.species.iter().position(|s| s.name == name)
    }
}
