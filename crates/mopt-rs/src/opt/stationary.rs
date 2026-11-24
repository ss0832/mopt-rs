//! Stationary points as a typestate.
//!
//! [`StationaryPoint<K>`] can only be obtained from an optimizer that converged with the
//! matching saddle order ([`GeometryOptimizer::stationary_point`]), so analyses that need a
//! minimum or a transition state (vibrational analysis, IRC, thermochemistry) take one as input
//! and cannot be handed an arbitrary, unoptimized structure by accident. Structures known to be
//! stationary from elsewhere enter through the explicitly named
//! [`StationaryPoint::assume_converged`].
//!
//! [`GeometryOptimizer::stationary_point`]: crate::opt::state_machine::GeometryOptimizer::stationary_point
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::marker::PhantomData;

use faer::Col;

use crate::domain::{Hartree, MolecularSystem, Quantity};

mod sealed {
    pub trait Sealed {}
}

/// Kind of stationary point.
pub trait StationaryKind: sealed::Sealed + Copy + std::fmt::Debug + Send + Sync + 'static {
    /// Number of negative Hessian eigenvalues (saddle order).
    const SADDLE_ORDER: usize;
    /// Human-readable name.
    const NAME: &'static str;
}

/// Local minimum (saddle order 0).
#[derive(Debug, Clone, Copy)]
pub struct Minimum;
/// First-order saddle point (transition state).
#[derive(Debug, Clone, Copy)]
pub struct TransitionState;

impl sealed::Sealed for Minimum {}
impl sealed::Sealed for TransitionState {}

impl StationaryKind for Minimum {
    const SADDLE_ORDER: usize = 0;
    const NAME: &'static str = "minimum";
}

impl StationaryKind for TransitionState {
    const SADDLE_ORDER: usize = 1;
    const NAME: &'static str = "transition state";
}

/// A converged stationary point of kind `K`.
///
/// The point is stationary on the surface that was optimized: the calculator with the bias
/// potentials. Both surfaces are kept, each with its own energy and gradient, so that a
/// consumer takes a consistent pair: [`Self::biased_energy`] with [`Self::gradient`] (the
/// surface of the optimization, for checks of stationarity, IRCs and analyses on that
/// surface), [`Self::energy`] with [`Self::raw_gradient`] (the calculator alone, for reported
/// energies and thermochemistry; the gradient there is the force the biases balance).
#[derive(Debug, Clone)]
pub struct StationaryPoint<K: StationaryKind> {
    system: MolecularSystem,
    energy: Quantity<Hartree>,
    biased_energy: Quantity<Hartree>,
    gradient: Col<f64>,
    raw_gradient: Col<f64>,
    iterations: usize,
    kind: PhantomData<K>,
}

impl<K: StationaryKind> StationaryPoint<K> {
    /// The point `system` with the energy and gradient of the calculator (`energy`,
    /// `raw_gradient`) and of the optimized surface (`biased_energy`, `gradient`).
    pub(crate) fn new(
        system: MolecularSystem,
        energy: Quantity<Hartree>,
        biased_energy: Quantity<Hartree>,
        gradient: Col<f64>,
        raw_gradient: Col<f64>,
        iterations: usize,
    ) -> Self {
        Self {
            system,
            energy,
            biased_energy,
            gradient,
            raw_gradient,
            iterations,
            kind: PhantomData,
        }
    }

    /// Wrap a structure the caller asserts to be a converged stationary point of kind `K`
    /// (e.g. read from a previous calculation) on a surface without bias potentials. No check
    /// is performed.
    pub fn assume_converged(
        system: MolecularSystem,
        energy: Quantity<Hartree>,
        gradient: Col<f64>,
    ) -> Self {
        Self::new(system, energy, energy, gradient.clone(), gradient, 0)
    }

    /// The structure.
    pub fn system(&self) -> &MolecularSystem {
        &self.system
    }

    /// Energy of the calculator alone (without bias potentials), the pair of
    /// [`Self::raw_gradient`].
    pub fn energy(&self) -> Quantity<Hartree> {
        self.energy
    }

    /// Energy of the optimized surface (with the bias potentials), the pair of
    /// [`Self::gradient`].
    pub fn biased_energy(&self) -> Quantity<Hartree> {
        self.biased_energy
    }

    /// Cartesian gradient of the optimized surface (with the bias potentials; Hartree/Bohr):
    /// the one that vanishes at the point.
    pub fn gradient(&self) -> &Col<f64> {
        &self.gradient
    }

    /// Cartesian gradient of the calculator alone (Hartree/Bohr).
    pub fn raw_gradient(&self) -> &Col<f64> {
        &self.raw_gradient
    }

    /// Number of optimizer iterations that produced the point (0 if assumed).
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Release the structure.
    pub fn into_system(self) -> MolecularSystem {
        self.system
    }
}
