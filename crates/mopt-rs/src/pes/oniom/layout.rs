//! The regions of an ONIOM calculation, their link atoms, and the model systems made from a
//! structure, with the way back for gradients and Hessians.
//!
//! A region is a set of atoms of the real system. Where a bond of the real system crosses
//! its border (from `Q1` inside to `Q2` outside), the model system gets a hydrogen link atom
//! on the line from `Q1` to `Q2`:
//!
//! - [`LinkPlacement::Scaled`] (the default, as Gaussian; Dapprich et al., J. Mol. Struct.
//!   THEOCHEM 1999, 461, 1): `r_L = r_Q1 + g (r_Q2 - r_Q1)` with
//!   `g = (R(Q1) + R(H)) / (R(Q1) + R(Q2))` from covalent radii, fixed for the run. The map
//!   is linear: gradients and Hessians go back with its constant Jacobian.
//! - [`LinkPlacement::Distance`] (MultiOptPy's `separate_high_layer_and_low_layer`):
//!   `r_L = r_Q1 + d u`, `u = (r_Q2 - r_Q1) / |r_Q2 - r_Q1|`, `d = R(Q1) + R(H)`. The map is
//!   not linear: the Hessian gets the second-order term `sum_k (dE/dr_L,k) d2 r_L,k`.
//!
//! In a periodic cell, the model system is a cluster: the atoms of the region and the link
//! atoms are unwrapped along bonds, with disconnected components placed near the first atom.
//!
//! MultiOptPy: `Utils/oniom.py` (`specify_link_atom_pairs`,
//! `separate_high_layer_and_low_layer`, `link_number_high_layer_and_low_layer`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::{AtomId, Element, MolecularSystem, Position, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::parameters::covalent::covalent_radius_bohr;

/// Where the link atom of a bond across the border of a region sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinkPlacement {
    /// A fixed share `g` of the bond (Dapprich et al.; Gaussian).
    #[default]
    Scaled,
    /// A fixed distance from `Q1` (MultiOptPy).
    Distance,
}

impl LinkPlacement {
    /// `scaled` or `distance`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "scaled" | "g" | "factor" => Ok(Self::Scaled),
            "distance" | "fixed" | "multioptpy" => Ok(Self::Distance),
            other => Err(MoptError::parse(format!(
                "-oniom_opt link={other}: scaled (a share of the bond) or distance (MultiOptPy)"
            ))),
        }
    }
}

/// A bond across the border of a region: `inner` (`Q1`) is in the region, `outer` (`Q2`)
/// is not; the link atom replaces `outer` in the model system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Link {
    /// `Q1`.
    pub inner: AtomId,
    /// `Q2`.
    pub outer: AtomId,
    /// `g` of [`LinkPlacement::Scaled`] or `d` (Bohr) of [`LinkPlacement::Distance`].
    pub factor: f64,
}

/// A region of an ONIOM calculation with its link atoms.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// Its atoms, in the order of the model system.
    pub atoms: Vec<AtomId>,
    /// Its link atoms, after the atoms in the model system.
    pub links: Vec<Link>,
    /// Charge and multiplicity of the model system.
    pub spin_charge: SpinCharge,
    /// Where the link atoms sit.
    pub placement: LinkPlacement,
    /// Charge and multiplicity of the whole structure the region was set up for, when those
    /// of the model system follow it ([`model_state`]); `None` when they were given. A
    /// structure of another state (a state of `-mf`) then has the model system of that state.
    pub follows: Option<SpinCharge>,
}

/// Charge and multiplicity of a model system of `elements` (link atoms included) in a whole
/// structure of the state `real`: those of the whole structure, with the multiplicity changed
/// by one where the link atoms change the parity of the electrons. A singlet becomes a
/// doublet and a doublet a singlet; a high-spin state (triplet and above) keeps its unpaired
/// electrons and takes one more (a triplet becomes a quartet, not a doublet). An explicit
/// state of the layer (`-oniom_opt high_multiplicity=`, `middle_multiplicity=`) is taken
/// instead of this.
pub fn model_state(elements: &[Element], real: SpinCharge) -> SpinCharge {
    let electrons = elements
        .iter()
        .map(|e| i64::from(e.atomic_number()))
        .sum::<i64>()
        - i64::from(real.charge);
    let odd = electrons.rem_euclid(2) == 1;
    let real_odd = real.multiplicity.is_multiple_of(2);
    if odd == real_odd {
        return real;
    }
    let multiplicity = match real.multiplicity {
        0 | 1 => 2,
        2 => 1,
        m => m + 1,
    };
    SpinCharge::new(real.charge, multiplicity)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: [f64; 3]) -> f64 {
    libm::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
}

/// The covalent radius of hydrogen (Bohr).
fn hydrogen_radius() -> f64 {
    covalent_radius_bohr(Element::H)
}

impl Link {
    /// The link of the bond `inner`-`outer` of `sys` placed by `placement`.
    pub fn new(
        sys: &MolecularSystem,
        inner: AtomId,
        outer: AtomId,
        placement: LinkPlacement,
    ) -> Self {
        let (r1, r2) = (
            covalent_radius_bohr(sys.element(inner)),
            covalent_radius_bohr(sys.element(outer)),
        );
        let factor = match placement {
            LinkPlacement::Scaled => (r1 + hydrogen_radius()) / (r1 + r2),
            LinkPlacement::Distance => r1 + hydrogen_radius(),
        };
        Self {
            inner,
            outer,
            factor,
        }
    }
}

impl Region {
    /// The positions of the region and of its link atoms in `real` as a cluster (the
    /// bonded components are unwrapped in a periodic cell).
    fn positions(&self, real: &MolecularSystem) -> MoptResult<Vec<[f64; 3]>> {
        if self.atoms.is_empty() {
            return Err(MoptError::config("an ONIOM region has no atoms"));
        }
        let cell = real.active_cell();
        let mut out: Vec<[f64; 3]> = if cell.is_some() {
            let region = real.subsystem(&self.atoms, self.spin_charge)?;
            let mut bonds = Vec::new();
            for a in region.atoms() {
                for b in region.atoms().skip(a.index() + 1) {
                    if region_bond(&region, a, b, 1.2) {
                        bonds.push([a.index(), b.index()]);
                    }
                }
            }
            let numbers = region
                .atoms()
                .map(|a| region.element(a).atomic_number())
                .collect();
            let graph = crate::coord::graph::MolecularGraph::from_bonds(numbers, &bonds)?;
            let whole = crate::coord::graph::whole_molecules_at(
                &region,
                &graph,
                crate::coord::graph::FragmentAnchor::FirstAtom,
            )?;
            whole.atoms().map(|a| whole.position(a).value()).collect()
        } else {
            self.atoms
                .iter()
                .map(|&a| real.position(a).value())
                .collect()
        };
        let index: std::collections::HashMap<_, _> = self
            .atoms
            .iter()
            .enumerate()
            .map(|(i, &a)| (a, i))
            .collect();
        for link in &self.links {
            let inner = index
                .get(&link.inner)
                .ok_or_else(|| MoptError::config("ONIOM link starts outside its region"))?;
            let q1 = out[*inner];
            let outer = real.position(link.outer).value();
            let v = cell.map_or_else(|| sub(outer, q1), |c| c.mic_delta(outer, q1));
            let p = match self.placement {
                LinkPlacement::Scaled => std::array::from_fn(|k| q1[k] + link.factor * v[k]),
                LinkPlacement::Distance => {
                    let length = norm(v);
                    if length < 1e-8 {
                        return Err(MoptError::calc(
                            "ONIOM: the two atoms of a link are at the same place",
                        ));
                    }
                    std::array::from_fn(|k| q1[k] + link.factor * v[k] / length)
                }
            };
            out.push(p);
        }
        Ok(out)
    }

    /// The model system of the region in `real`: its atoms, then a hydrogen for every link.
    pub fn model(&self, real: &MolecularSystem) -> MoptResult<MolecularSystem> {
        let positions = self.positions(real)?;
        let mut elements: Vec<Element> = self.atoms.iter().map(|a| real.element(*a)).collect();
        elements.extend(std::iter::repeat_n(Element::H, self.links.len()));
        let placed: Vec<Position<crate::domain::Bohr>> =
            positions.iter().map(|p| Position::new(*p)).collect();
        // A whole structure of another charge or multiplicity than the one the region was
        // set up for (a state of -mf): the model system follows it, or is refused when its
        // state was given.
        let spin_charge = match self.follows {
            Some(built) if built == real.spin_charge => self.spin_charge,
            Some(_) => model_state(&elements, real.spin_charge),
            None => self.spin_charge,
        };
        MolecularSystem::new(elements, &placed, spin_charge, None)
    }

    /// The real-system gradient (`3N`) of a model gradient.
    pub fn gradient_to_real(
        &self,
        real: &MolecularSystem,
        model_gradient: &Col<f64>,
    ) -> MoptResult<Col<f64>> {
        let mut g = Col::zeros(real.dof());
        for (i, (_, block)) in self.blocks(real)?.into_iter().enumerate() {
            for (atom, m) in block {
                for r in 0..3 {
                    for c in 0..3 {
                        g[atom.dof(c)] += m[r][c] * model_gradient[3 * i + r];
                    }
                }
            }
        }
        Ok(g)
    }

    /// The real-system Hessian (`3N x 3N`) of a model Hessian (and, for links at a fixed
    /// distance, the second-order term from the model gradient).
    pub fn hessian_to_real(
        &self,
        real: &MolecularSystem,
        model_gradient: &Col<f64>,
        model_hessian: &Mat<f64>,
    ) -> MoptResult<Mat<f64>> {
        let all: Vec<usize> = (0..real.n_atoms()).collect();
        self.hessian_on_atoms(real, model_gradient, model_hessian, &all)
    }

    /// The block of [`Self::hessian_to_real`] on the real atoms `atoms` (0-based; its rows and
    /// columns are the coordinates of `atoms` in their order): the whole of it where `atoms`
    /// hold every atom of the region and both atoms of every link (the rest of the real
    /// system has no part in the term), had without a `3N x 3N` matrix. An atom of the region
    /// or of a link outside `atoms` is an error.
    pub fn hessian_on_atoms(
        &self,
        real: &MolecularSystem,
        model_gradient: &Col<f64>,
        model_hessian: &Mat<f64>,
        atoms: &[usize],
    ) -> MoptResult<Mat<f64>> {
        let mut slot = vec![None; real.n_atoms()];
        for (k, &a) in atoms.iter().enumerate() {
            if let Some(s) = slot.get_mut(a) {
                *s = Some(k);
            }
        }
        let at = |atom: AtomId, c: usize| -> MoptResult<usize> {
            slot.get(atom.index())
                .copied()
                .flatten()
                .map(|k| 3 * k + c)
                .ok_or_else(|| {
                    MoptError::calc(format!(
                        "ONIOM: atom {} of a region or of a link is not among the atoms of the \
                         block of the Hessian",
                        atom.one_based()
                    ))
                })
        };
        let n = 3 * atoms.len();
        let blocks = self.blocks(real)?;
        // J (3n_model x 3k) as dense columns of the atoms: small regions, so dense rows of
        // the model are fine.
        let rows = 3 * blocks.len();
        let mut j = Mat::<f64>::zeros(rows, n);
        for (i, (_, block)) in blocks.iter().enumerate() {
            for (atom, m) in block {
                let columns = [at(*atom, 0)?, at(*atom, 1)?, at(*atom, 2)?];
                for (r, row) in m.iter().enumerate() {
                    for (column, value) in columns.iter().zip(row) {
                        j[(3 * i + r, *column)] += value;
                    }
                }
            }
        }
        let mut h = j.transpose() * model_hessian * &j;
        if self.placement == LinkPlacement::Distance {
            let start = self.atoms.len();
            for (k, link) in self.links.iter().enumerate() {
                let (v, length) = self.link_vector(real, link)?;
                let u = v.map(|c| c / length);
                let w: [f64; 3] = std::array::from_fn(|r| model_gradient[3 * (start + k) + r]);
                let wu: f64 = (0..3).map(|r| w[r] * u[r]).sum();
                let scale = -link.factor / (length * length);
                let t: [[f64; 3]; 3] = std::array::from_fn(|a| {
                    std::array::from_fn(|b| {
                        let delta = if a == b { 1.0 } else { 0.0 };
                        scale * (w[a] * u[b] + w[b] * u[a] + wu * delta - 3.0 * wu * u[a] * u[b])
                    })
                });
                for (x, sx) in [(link.inner, -1.0), (link.outer, 1.0)] {
                    for (y, sy) in [(link.inner, -1.0), (link.outer, 1.0)] {
                        for a in 0..3 {
                            for b in 0..3 {
                                h[(at(x, a)?, at(y, b)?)] += sx * sy * t[a][b];
                            }
                        }
                    }
                }
            }
        }
        Ok(h)
    }

    /// `r_Q2 - r_Q1` (minimum image) and its length.
    fn link_vector(&self, real: &MolecularSystem, link: &Link) -> MoptResult<([f64; 3], f64)> {
        let (p1, p2) = (
            real.position(link.inner).value(),
            real.position(link.outer).value(),
        );
        let v = match real.active_cell() {
            Some(c) => c.mic_delta(p2, p1),
            None => sub(p2, p1),
        };
        let length = norm(v);
        if length < 1e-8 {
            return Err(MoptError::calc(
                "ONIOM: the two atoms of a link are at the same place",
            ));
        }
        Ok((v, length))
    }

    /// For every atom of the model system, the real atoms it depends on with the 3 x 3
    /// blocks `d r_model / d r_real`.
    #[allow(clippy::type_complexity)]
    fn blocks(
        &self,
        real: &MolecularSystem,
    ) -> MoptResult<Vec<(usize, Vec<(AtomId, [[f64; 3]; 3])>)>> {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut out: Vec<(usize, Vec<(AtomId, [[f64; 3]; 3])>)> = self
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (i, vec![(*a, identity)]))
            .collect();
        for (k, link) in self.links.iter().enumerate() {
            let index = self.atoms.len() + k;
            let (with_outer, with_inner) = match self.placement {
                LinkPlacement::Scaled => {
                    let g = link.factor;
                    (
                        identity.map(|row| row.map(|v| g * v)),
                        identity.map(|row| row.map(|v| (1.0 - g) * v)),
                    )
                }
                LinkPlacement::Distance => {
                    let (v, length) = self.link_vector(real, link)?;
                    let u = v.map(|c| c / length);
                    let a: [[f64; 3]; 3] = std::array::from_fn(|r| {
                        std::array::from_fn(|c| {
                            let delta = if r == c { 1.0 } else { 0.0 };
                            link.factor * (delta - u[r] * u[c]) / length
                        })
                    });
                    let b: [[f64; 3]; 3] =
                        std::array::from_fn(|r| std::array::from_fn(|c| identity[r][c] - a[r][c]));
                    (a, b)
                }
            };
            out.push((
                index,
                vec![(link.inner, with_inner), (link.outer, with_outer)],
            ));
        }
        Ok(out)
    }
}

fn region_bond(sys: &MolecularSystem, a: AtomId, b: AtomId, scale: f64) -> bool {
    sys.distance(a, b).value()
        < scale * (covalent_radius_bohr(sys.element(a)) + covalent_radius_bohr(sys.element(b)))
}

/// The bonds that cross the border of `atoms` in `sys`: pairs (inside, outside) closer
/// than `scale` times the sum of their covalent radii (MultiOptPy's threshold 1.2).
pub fn crossing_bonds(
    sys: &MolecularSystem,
    atoms: &[AtomId],
    scale: f64,
) -> Vec<(AtomId, AtomId)> {
    let inside: std::collections::HashSet<usize> = atoms.iter().map(|a| a.index()).collect();
    let mut out = Vec::new();
    for &a in atoms {
        for b in sys.atoms() {
            if inside.contains(&b.index()) {
                continue;
            }
            if region_bond(sys, a, b, scale) {
                out.push((a, b));
            }
        }
    }
    out
}

/// For every given outside atom (`Q2`), the nearest atom of `atoms` (`Q1`), as MultiOptPy
/// pairs them.
pub fn pair_with_nearest(
    sys: &MolecularSystem,
    atoms: &[AtomId],
    outside: &[AtomId],
) -> MoptResult<Vec<(AtomId, AtomId)>> {
    outside
        .iter()
        .map(|&q2| {
            atoms
                .iter()
                .map(|&q1| {
                    let (p1, p2) = (sys.position(q1).value(), sys.position(q2).value());
                    let d = match sys.active_cell() {
                        Some(c) => c.mic_distance(p2, p1),
                        None => norm(sub(p2, p1)),
                    };
                    (q1, d)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(q1, _)| (q1, q2))
                .ok_or_else(|| MoptError::config("an ONIOM region has no atoms"))
        })
        .collect()
}
