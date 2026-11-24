//! Electronic embedding of ONIOM (Vreven, Byun, Komáromi, Dapprich, Montgomery, Morokuma,
//! Frisch, J. Chem. Theory Comput. 2006, 2, 815): the model system of the higher level of a
//! region is computed in the field of the charges of the atoms around it, and the classical
//! Coulomb energy of the model atoms with the same charges is taken off with the lower level:
//!
//! ```text
//! E = E_low(real) + E_high(model; q_env) - E_low(model) - sum_(i in model, K) q_i q_K / r_iK
//! ```
//!
//! so that the interaction of the model with its environment is that of the higher level
//! instead of the charges of the lower one. The charges are fixed: those of the lowest level
//! (the topology charges of GFN-FF, a force field's charges; [`ChargeSource::Lowest`]) or given
//! ones, one per atom of the real system. The link atoms carry none.
//!
//! The atom `Q2` of a link (outside, replaced by the link atom) sits next to the link atom,
//! and its charge would polarize the model system there unphysically. [`BoundaryCharges`]:
//! `rc` (the default) moves it to the midpoints of the bonds of `Q2` to its other neighbours
//! in the environment, in equal shares (the redistributed charge scheme of Lin and Truhlar,
//! J. Phys. Chem. A 2005, 109, 3991; the total charge is kept); `zero` leaves it out; `keep`
//! leaves it where it is. The neighbours are those of the starting structure (1.2 times the
//! sum of the covalent radii).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::layout::Region;
use crate::domain::{AtomId, Element, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::parameters::covalent::covalent_radius_bohr;
use crate::pes::point_charges::PointCharges;

/// Bonded neighbours: closer than this many times the sum of the covalent radii.
const BOND_SCALE: f64 = 1.2;

/// What becomes of the charge of the atoms `Q2` of the links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryCharges {
    /// Moved to the midpoints of its bonds to the environment (redistributed charges).
    #[default]
    Redistributed,
    /// Left out.
    Zero,
    /// Kept where it is.
    Keep,
}

impl BoundaryCharges {
    /// The name of the option (`-oniom_opt boundary=`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Redistributed => "rc",
            Self::Zero => "zero",
            Self::Keep => "keep",
        }
    }

    /// The rule of `name`.
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "rc" | "redistributed" => Ok(Self::Redistributed),
            "zero" | "z1" => Ok(Self::Zero),
            "keep" => Ok(Self::Keep),
            other => Err(MoptError::parse(format!(
                "-oniom_opt boundary={other}: expected rc, zero, or keep"
            ))),
        }
    }
}

/// Where the charges of the environment come from.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ChargeSource {
    /// The lowest level's ([`crate::pes::Calculator::embedding_charges`]).
    #[default]
    Lowest,
    /// One per atom of the real system (e).
    Given(Vec<f64>),
}

/// The electronic embedding of an ONIOM calculation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ElectronicEmbedding {
    /// The charges.
    pub charges: ChargeSource,
    /// The charge of `Q2`.
    pub boundary: BoundaryCharges,
}

/// One point charge of the environment of a region: its charge and the atoms whose
/// positions place it (weights summing to 1: an atom, or the midpoint of a bond).
#[derive(Debug, Clone, PartialEq)]
struct Placed {
    charge: f64,
    element: Option<Element>,
    at: Vec<(usize, f64)>,
}

/// The charges around one region, set up once for the atoms of the real system.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionCharges {
    placed: Vec<Placed>,
    /// The charges of the model atoms (in the order of the model system, link atoms 0).
    model: Vec<f64>,
}

impl RegionCharges {
    /// The charges around `region` of `real` (atomic charges `q`, one per atom) with the
    /// boundary rule `boundary`.
    pub fn new(
        region: &Region,
        real: &MolecularSystem,
        q: &[f64],
        boundary: BoundaryCharges,
    ) -> MoptResult<Self> {
        if q.len() != real.n_atoms() {
            return Err(MoptError::config(format!(
                "ONIOM electronic embedding: {} charges for {} atoms",
                q.len(),
                real.n_atoms()
            )));
        }
        let n = real.n_atoms();
        let mut inside = vec![false; n];
        for a in &region.atoms {
            inside[a.index()] = true;
        }
        let mut is_q2 = vec![false; n];
        for link in &region.links {
            is_q2[link.outer.index()] = true;
        }
        let mut placed = Vec::new();
        for a in 0..n {
            if inside[a] {
                continue;
            }
            let element = Some(real.elements[a]);
            if !is_q2[a] || boundary == BoundaryCharges::Keep {
                placed.push(Placed {
                    charge: q[a],
                    element,
                    at: vec![(a, 1.0)],
                });
                continue;
            }
            if boundary == BoundaryCharges::Zero {
                continue;
            }
            let neighbours: Vec<usize> = (0..n)
                .filter(|&b| b != a && !inside[b] && bonded(real, a, b))
                .collect();
            if neighbours.is_empty() {
                // A Q2 with no other neighbour: nowhere to move its charge.
                continue;
            }
            let share = q[a] / neighbours.len() as f64;
            for b in neighbours {
                placed.push(Placed {
                    charge: share,
                    element: None,
                    at: vec![(a, 0.5), (b, 0.5)],
                });
            }
        }
        let mut model: Vec<f64> = region.atoms.iter().map(|a| q[a.index()]).collect();
        model.extend(std::iter::repeat_n(0.0, region.links.len()));
        Ok(Self { placed, model })
    }

    /// The point charges at the positions of `real`.
    pub fn at(&self, real: &MolecularSystem) -> MoptResult<PointCharges> {
        let x = real.coords_flat_bohr();
        let positions = self
            .placed
            .iter()
            .map(|p| {
                let mut r = [0.0; 3];
                for &(atom, w) in &p.at {
                    for c in 0..3 {
                        r[c] += w * x[3 * atom + c];
                    }
                }
                r
            })
            .collect();
        PointCharges::new(
            positions,
            self.placed.iter().map(|p| p.charge).collect(),
            self.placed.iter().map(|p| p.element).collect(),
        )
    }

    /// The gradient of the real system (flat) of a gradient with respect to the positions of
    /// the charges, added to `gradient`.
    pub fn add_to_real(&self, charge_gradient: &[[f64; 3]], gradient: &mut faer::Col<f64>) {
        for (p, g) in self.placed.iter().zip(charge_gradient) {
            for &(atom, w) in &p.at {
                for c in 0..3 {
                    gradient[3 * atom + c] += w * g[c];
                }
            }
        }
    }

    /// The charges of the model atoms (link atoms 0), in the order of the model system.
    pub fn model_charges(&self) -> &[f64] {
        &self.model
    }

    /// Number of point charges.
    pub fn len(&self) -> usize {
        self.placed.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.placed.is_empty()
    }
}

fn bonded(sys: &MolecularSystem, a: usize, b: usize) -> bool {
    let (pa, pb) = (
        sys.position(AtomId::new(a)).value(),
        sys.position(AtomId::new(b)).value(),
    );
    let d = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    r < BOND_SCALE * (covalent_radius_bohr(sys.elements[a]) + covalent_radius_bohr(sys.elements[b]))
}
