//! Metadynamics (MultiOptPy `-metad`): Gaussians `h exp(-(s - s_k)^2 / 2w^2)` deposited along
//! collective variables `s` at every accepted evaluation.
//!
//! Collective variables: bond length, bond angle, dihedral angle, and out-of-plane angle. The
//! dihedral is periodic, so its Gaussians use the difference wrapped into `(-pi, pi]`
//! (MultiOptPy takes the angle modulo `2 pi` and subtracts directly, which puts a jump into the
//! bias at `phi = 0`); the out-of-plane angle lies in `[-pi/2, pi/2]` and is used as it is
//! (MultiOptPy's modulo moves negative angles next to `2 pi`).
//!
//! The other kind of collective variable, the RMSD to a set of structures (one variable per
//! deposited structure, each Gaussian at zero RMSD), is
//! [`super::rmsd_metadynamics::RmsdMetadynamics`]; `confsearch` runs it with the dynamics of
//! `md`.
//!
//! MultiOptPy: `Potential/gaussian_potential.py:GaussianPotential`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;
use std::sync::Mutex;

use super::engine::{norm, Clock, Geom, GeometricBias};
use super::restraints::{dihedral, out_of_plane};
use crate::domain::AtomId;
use crate::math::ad::{angle_between, Scalar};
use crate::pes::term::{Frame, History};

/// Collective variable of a metadynamics bias.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectiveVariable {
    /// `|r_a - r_b|`.
    Bond([AtomId; 2]),
    /// Angle `a-b-c` at `b`.
    Angle([AtomId; 3]),
    /// Dihedral `a-b-c-d` (IUPAC sign).
    Dihedral([AtomId; 4]),
    /// Angle of `b` above the plane of `c` and `d` around the center `a`.
    OutOfPlane([AtomId; 4]),
}

impl CollectiveVariable {
    /// Atoms of the variable.
    pub fn atoms(&self) -> Vec<AtomId> {
        match self {
            Self::Bond(a) => a.to_vec(),
            Self::Angle(a) => a.to_vec(),
            Self::Dihedral(a) | Self::OutOfPlane(a) => a.to_vec(),
        }
    }

    /// Value (Bohr or rad).
    pub fn value<T: Scalar>(&self, g: &Geom<'_, T>) -> T {
        match self {
            Self::Bond([a, b]) => norm(g.d(*a, *b)),
            Self::Angle([a, b, c]) => angle_between(g.d(*a, *b), g.d(*c, *b)),
            Self::Dihedral([a, b, c, d]) => dihedral(g.d(*b, *a), g.d(*c, *b), g.d(*d, *c)).0,
            Self::OutOfPlane([c, i1, i2, i3]) => {
                out_of_plane(g.d(*i1, *c), g.d(*i2, *c), g.d(*i3, *c)).unwrap_or(T::zero())
            }
        }
    }

    fn is_periodic(&self) -> bool {
        matches!(self, Self::Dihedral(_))
    }
}

/// One biased collective variable.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadynamicsCv {
    /// The variable.
    pub cv: CollectiveVariable,
    /// Gaussian height (Hartree).
    pub height: f64,
    /// Gaussian width (Bohr or rad).
    pub width: f64,
}

/// Metadynamics bias with its deposited Gaussians.
#[derive(Debug)]
pub struct Metadynamics {
    /// Biased variables.
    pub cvs: Vec<MetadynamicsCv>,
    centers: Mutex<Vec<Vec<f64>>>,
}

impl Metadynamics {
    /// Bias without Gaussians yet.
    pub fn new(cvs: Vec<MetadynamicsCv>) -> Self {
        let n = cvs.len();
        Self {
            cvs,
            centers: Mutex::new(vec![Vec::new(); n]),
        }
    }

    /// Deposited Gaussian centers, per variable.
    pub fn centers(&self) -> Vec<Vec<f64>> {
        self.centers.lock().map(|c| c.clone()).unwrap_or_default()
    }
}

impl GeometricBias for Metadynamics {
    fn name(&self) -> &str {
        "metadynamics"
    }
    fn atoms(&self) -> Vec<AtomId> {
        self.cvs.iter().flat_map(|c| c.cv.atoms()).collect()
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let centers = self.centers();
        let mut e = T::zero();
        for (cv, cs) in self.cvs.iter().zip(&centers) {
            let s = cv.cv.value(g);
            let w2 = 2.0 * cv.width * cv.width;
            for &c in cs {
                let mut d = s - T::from_f64(c);
                if cv.cv.is_periodic() {
                    d = d - T::from_f64(2.0 * PI * (d.val() / (2.0 * PI)).round());
                }
                e += T::from_f64(cv.height) * (T::zero() - d * d / T::from_f64(w2)).exp();
            }
        }
        e
    }
    fn commit(&self, g: &Geom<'_, f64>, _clock: Clock) {
        if let Ok(mut c) = self.centers.lock() {
            for (cv, cs) in self.cvs.iter().zip(c.iter_mut()) {
                cs.push(cv.cv.value(g));
            }
        }
    }
    fn frame(&self) -> Frame {
        Frame::Invariant
    }
    /// The deposited Gaussians are its state.
    fn history(&self) -> History {
        History::Stateful
    }
    /// The same variables with a copy of the Gaussians deposited so far.
    fn fork(&self) -> Option<Self> {
        Some(Self {
            cvs: self.cvs.clone(),
            centers: std::sync::Mutex::new(self.centers()),
        })
    }
}
