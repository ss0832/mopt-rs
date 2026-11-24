//! Bias potentials (extension point).
//!
//! A bias potential is a [`PesTerm`]: an energy contribution with an exact gradient and Hessian
//! evaluated in-process. Bias potentials and multi-state combinations share one mechanism, the
//! surface expression [`PesExpr`](crate::pes::combine::PesExpr): a bias is a term leaf added to
//! the whole surface with [`PesExpr::plus_term`](crate::pes::combine::PesExpr::plus_term), or
//! placed on a single state or fragment of a combined surface. User-defined bias potentials only
//! need to implement [`PesTerm`]; geometric ones can implement [`GeometricBias`] instead and get
//! exact derivatives and periodic-cell support from [`AdBias`].
//!
//! MultiOptPy: `Potential/potential.py:BiasPotentialCalculation` and the files of `Potential/`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod afir;
pub mod auxiliary;
pub mod ellipsoid;
pub mod engine;
pub mod expression;
pub mod metadynamics;
pub mod nanoreactor;
pub mod repulsive;
pub mod restraints;
// Metadynamics with the RMSD to a set of structures as its variables (confsearch).
pub mod rmsd_metadynamics;
pub mod spacer;
pub mod wells;

pub use crate::pes::term::{PesTerm, TermContext};
pub use afir::{afir_alpha, Afir};
pub use auxiliary::{AuxiliaryBias, MicroBias};
pub use ellipsoid::{AsymmetricEllipsoids, Ellipsoid, EllipsoidVariant};
pub use engine::{AdBias, Clock, Geom, GeometricBias, Ramp};
pub use expression::{ExpressionBias, VariableExpression};
pub use metadynamics::{CollectiveVariable, Metadynamics, MetadynamicsCv};
pub use nanoreactor::NanoReactor;
pub use repulsive::{ConeRepulsive, LjCenter, LjGaussian, LjParameters, LjRepulsive};
pub use restraints::{
    AngleContinuation, BondRange, DihedralForm, Flux, KeepAngle, KeepDihedral, KeepDistance,
    KeepOutOfPlane, MorseDistance, PullingForce, Universal, VoidPoint,
};
pub use rmsd_metadynamics::RmsdMetadynamics;
pub use spacer::SpacerModel;
pub use wells::{Well, WellGeometry, WellPotential};
