//! Coordinate systems for optimization (extension point).
//!
//! A coordinate system only has to provide its values `q(x)`, the Wilson matrix
//! `B = dq/dx`, and the displacement between two geometries (which handles periodic
//! coordinates). Gradient / Hessian transformation and the iterative back-transformation of
//! internal steps to Cartesian displacements are shared machinery in [`transform`], so any
//! user-defined system (registered in [`CoordinateRegistry`]) works with every step generator.
//!
//! Built-in systems: `cart` ([`Cartesian`]), `ric` ([`RedundantInternal`]), `dic`
//! ([`DelocalizedInternal`]), `tric` / `tric-p` / `tric-cart` ([`Tric`]), `spherical`
//! ([`Spherical`]), `hyperspherical` / `polar` / `hyperspherical:FILE` ([`Hyperspherical`]),
//! and `auto` ([`Auto`], which chooses among cart, dic, tric, and tric-cart from the
//! structure). A system can ask to be set up anew during a run
//! ([`CoordinateSystem::needs_reinitialization`]).
//!
//! MultiOptPy: `Coordinate/redundant_coordinate.py:RedundantInternalCoordinates` (generalized to
//!  any coordinate system).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod auto;
pub mod cartesian;
pub mod delocalized;
pub mod redundant;
pub mod spherical;
pub mod transform;
pub mod tric;

use std::sync::Arc;

use faer::{Col, Mat};

pub use auto::{Auto, AutoChoice};
pub use cartesian::Cartesian;
pub use delocalized::DelocalizedInternal;
pub use redundant::RedundantInternal;
pub use spherical::{Centre, Hyperspherical, Spherical};
pub use transform::{back_transform, CoordinateFrame};
pub use tric::Tric;

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

/// Coordinate system in which an optimizer takes its steps.
pub trait CoordinateSystem: Send {
    /// Canonical name.
    fn name(&self) -> &str;

    /// Set up the system for the starting geometry (e.g. generate primitives / basis).
    fn initialize(&mut self, sys: &MolecularSystem) -> MoptResult<()>;

    /// Number of coordinates for `sys`.
    fn dim(&self, sys: &MolecularSystem) -> usize;

    /// Coordinate values `q(x)`.
    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>>;

    /// Names of the coordinates, in the order of [`Self::values`], where each is one internal
    /// coordinate (`R(1-2)`, `A(1-2-3)`, ...); `None` where they are combinations.
    fn labels(&self) -> Option<Vec<String>> {
        None
    }

    /// Wilson matrix `B = dq/dx` (`dim x 3N`).
    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>>;

    /// Displacement `q(to) - q(from)` (periodic coordinates wrapped).
    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        Ok(self.values(to)? - self.values(from)?)
    }

    /// Rigid-body modes to project out of steps, if the coordinates contain them
    /// (Cartesian coordinates of an isolated or periodic system).
    fn rigid_body_basis(&self, _sys: &MolecularSystem) -> Option<Mat<f64>> {
        None
    }

    /// Whether the coordinates should be set up anew at `sys` because their definition is
    /// about to break down there (TRIC: a fragment has turned far from its reference
    /// orientation). The optimizer then calls [`Self::initialize`] at `sys` and carries its
    /// Hessian over through Cartesian coordinates. Must not fail: a structure at which the
    /// coordinates cannot be evaluated at all answers `true`.
    fn needs_reinitialization(&self, _sys: &MolecularSystem) -> bool {
        false
    }

    /// `true` when `q = x` and `B = I`, enabling exact shortcuts.
    fn is_cartesian(&self) -> bool {
        false
    }
}

/// Factory creating a coordinate system.
pub type CoordinateFactory = Arc<dyn Fn() -> Box<dyn CoordinateSystem> + Send + Sync>;

/// Factory creating a coordinate system from the text after `:` in its name (`tric-cart:0.3`).
pub type CoordinateArgumentFactory =
    Arc<dyn Fn(&str) -> MoptResult<Box<dyn CoordinateSystem>> + Send + Sync>;

/// Name -> factory table of coordinate systems.
#[derive(Clone)]
pub struct CoordinateRegistry {
    entries: Vec<(String, CoordinateFactory)>,
    with_argument: Vec<(String, CoordinateArgumentFactory)>,
}

impl Default for CoordinateRegistry {
    fn default() -> Self {
        let mut reg = Self {
            entries: Vec::new(),
            with_argument: Vec::new(),
        };
        reg.register(
            "cart",
            Arc::new(|| Box::new(Cartesian) as Box<dyn CoordinateSystem>),
        );
        reg.register(
            "ric",
            Arc::new(|| Box::new(RedundantInternal::default()) as Box<dyn CoordinateSystem>),
        );
        reg.register(
            "dic",
            Arc::new(|| Box::new(DelocalizedInternal::default()) as Box<dyn CoordinateSystem>),
        );
        reg.register(
            "tric",
            Arc::new(|| Box::new(Tric::delocalized()) as Box<dyn CoordinateSystem>),
        );
        reg.register(
            "tric-p",
            Arc::new(|| Box::new(Tric::primitive()) as Box<dyn CoordinateSystem>),
        );
        reg.register(
            "tric-cart",
            Arc::new(|| {
                let mut c = Tric::delocalized();
                // The default weight is positive: the constructor cannot fail.
                if let Ok(hybrid) = Tric::with_cartesian(tric::DEFAULT_CARTESIAN_WEIGHT) {
                    c = hybrid;
                }
                Box::new(c) as Box<dyn CoordinateSystem>
            }),
        );
        reg.register_with_argument(
            "tric-cart",
            Arc::new(|weight| {
                let w: f64 = weight.trim().parse().map_err(|_| {
                    MoptError::config(format!(
                        "tric-cart: the weight '{weight}' is not a number (as in tric-cart:0.3)"
                    ))
                })?;
                Ok(Box::new(Tric::with_cartesian(w)?) as Box<dyn CoordinateSystem>)
            }),
        );
        reg.register(
            "spherical",
            Arc::new(|| Box::new(Spherical::default()) as Box<dyn CoordinateSystem>),
        );
        reg.register(
            "hyperspherical",
            Arc::new(|| {
                Box::new(Hyperspherical::new(Centre::Centroid)) as Box<dyn CoordinateSystem>
            }),
        );
        reg.register_with_argument(
            "hyperspherical",
            Arc::new(|file| {
                let centre = crate::io::xyz::read_xyz_file(
                    file.trim(),
                    crate::domain::SpinCharge::default(),
                    None,
                )
                .map_err(|e| {
                    MoptError::config(format!(
                        "hyperspherical:{file}: the centre could not be read: {e}"
                    ))
                })?;
                Ok(Box::new(Hyperspherical::new(Centre::Structure(
                    centre.coords_flat_bohr(),
                ))) as Box<dyn CoordinateSystem>)
            }),
        );
        reg.register(
            "auto",
            Arc::new(|| Box::new(Auto::default()) as Box<dyn CoordinateSystem>),
        );
        reg
    }
}

impl CoordinateRegistry {
    /// Register (or replace) a coordinate system under `name` (case-insensitive).
    pub fn register(&mut self, name: &str, factory: CoordinateFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, factory));
    }

    /// Register (or replace) a system that takes an argument: `name:ARGUMENT`.
    pub fn register_with_argument(&mut self, name: &str, factory: CoordinateArgumentFactory) {
        let key = name.to_ascii_lowercase();
        self.with_argument.retain(|(k, _)| *k != key);
        self.with_argument.push((key, factory));
    }

    /// Registered names.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// Create the coordinate system registered under `name`.
    pub fn create(&self, name: &str) -> MoptResult<Box<dyn CoordinateSystem>> {
        if let Some((head, argument)) = name.split_once(':') {
            let mut key = head.trim().to_ascii_lowercase();
            if key == "polar" {
                key = "hyperspherical".to_string();
            }
            return match self.with_argument.iter().find(|(k, _)| *k == key) {
                Some((_, f)) => f(argument),
                None => Err(MoptError::config(format!(
                    "coordinate system '{head}' takes no argument (got '{name}')"
                ))),
            };
        }
        let key = name.to_ascii_lowercase();
        let key = match key.as_str() {
            "cartesian" => "cart".to_string(),
            "redundant" => "ric".to_string(),
            "delocalized" => "dic".to_string(),
            "tricp" | "tric_p" => "tric-p".to_string(),
            "polar" => "hyperspherical".to_string(),
            _ => key,
        };
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, f)| f())
            .ok_or_else(|| {
                MoptError::config(format!(
                    "unknown coordinate system '{name}' (available: {})",
                    self.names().join(", ")
                ))
            })
    }
}
