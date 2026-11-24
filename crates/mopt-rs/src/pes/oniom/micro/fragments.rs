//! Internal coordinates of each fragment of the environment, in the manner of the hybrid
//! delocalized internal coordinates (HDLC) of Billeter, Turner, and Thiel (Phys. Chem. Chem.
//! Phys. 2000, 2, 2177).
//!
//! The environment is cut into blocks: the connected fragments of its bond graph (a water, a
//! ligand), with fragments of more than [`BLOCK_ATOMS`] atoms cut into pieces along a
//! breadth-first order (HDLC cuts proteins into residues; a structure file has no residues,
//! so the pieces are of a size like theirs). Each block of three atoms or more has the
//! delocalized translation-rotation-internal coordinates of itself alone
//! ([`crate::coord::system::Tric`]: its translation, its rotation, and its internal
//! coordinates, `3 n` of them for `n` atoms); smaller blocks, and blocks whose coordinates do
//! not span their `3 n` motions, keep their Cartesian coordinates. The Wilson matrix is block
//! diagonal, so the transformations cost the sum over the blocks instead of the cube of the
//! whole environment: the stiff bonds and the soft motions of whole molecules are apart in
//! the variables of the limited-memory BFGS ([`crate::pes::microiter::relax_in`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::coord::connectivity::BondConnectivity;
use crate::coord::internal::PrimitiveOptions;
use crate::coord::system::transform::BackTransformOptions;
use crate::coord::system::{back_transform, CoordinateFrame, CoordinateSystem, Tric};
use crate::domain::{Element, MolecularSystem, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::pes::microiter::InnerCoordinates;

/// Atoms of a block at most (fragments above are cut into pieces of about a residue).
pub const BLOCK_ATOMS: usize = 30;

struct Block {
    /// Atoms of the block (indices into the environment).
    atoms: Vec<usize>,
    /// Its coordinates (`None`: Cartesian).
    coords: Option<Box<dyn CoordinateSystem>>,
    /// The frame of the transformations at the prepared point.
    frame: Option<CoordinateFrame>,
}

/// The coordinates of every block of an environment.
pub struct FragmentCoordinates {
    /// Elements of the environment atoms.
    elements: Vec<Element>,
    blocks: Vec<Block>,
    generation: usize,
}

/// The atoms of each fragment of `conn`, the larger ones cut into pieces of at most
/// [`BLOCK_ATOMS`] atoms along a breadth-first order from their first atom.
fn blocks_of(conn: &BondConnectivity) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for fragment in &conn.fragments {
        if fragment.len() <= BLOCK_ATOMS {
            out.push(fragment.iter().map(|a| a.index()).collect());
            continue;
        }
        let mut seen = vec![false; conn.n_atoms];
        let mut order = Vec::with_capacity(fragment.len());
        for &root in fragment {
            if seen[root.index()] {
                continue;
            }
            seen[root.index()] = true;
            let mut queue = std::collections::VecDeque::from([root]);
            while let Some(a) = queue.pop_front() {
                order.push(a.index());
                for &b in &conn.adjacency[a.index()] {
                    if !seen[b.index()] {
                        seen[b.index()] = true;
                        queue.push_back(b);
                    }
                }
            }
        }
        out.extend(order.chunks(BLOCK_ATOMS).map(<[usize]>::to_vec));
    }
    out
}

impl FragmentCoordinates {
    /// The blocks of the environment of `elements` at `y` (flat, Bohr). Not with a periodic
    /// cell: the blocks would need minimum-image fragments.
    pub fn new(
        elements: Vec<Element>,
        y: &[f64],
        cell: Option<&crate::domain::PeriodicCell>,
    ) -> MoptResult<Self> {
        if cell.is_some() {
            return Err(MoptError::unsupported(
                "-oniom_opt micro_coord=fragments: not in a periodic cell (micro_coord=cart is)",
            ));
        }
        let all = MolecularSystem::from_flat_bohr(
            elements.clone(),
            &Col::from_fn(y.len(), |i| y[i]),
            SpinCharge::default(),
            None,
        )?;
        let conn = BondConnectivity::from_system(&all, PrimitiveOptions::default().bond_scale);
        let mut out = Self {
            elements,
            blocks: blocks_of(&conn)
                .into_iter()
                .map(|atoms| Block {
                    atoms,
                    coords: None,
                    frame: None,
                })
                .collect(),
            generation: 0,
        };
        for k in 0..out.blocks.len() {
            out.set_up(k, y)?;
        }
        Ok(out)
    }

    /// Blocks with internal coordinates and blocks with Cartesian ones.
    pub fn block_counts(&self) -> (usize, usize) {
        let internal = self.blocks.iter().filter(|b| b.coords.is_some()).count();
        (internal, self.blocks.len() - internal)
    }

    /// The structure of block `k` at `y`.
    fn block_system(&self, k: usize, y: &[f64]) -> MoptResult<MolecularSystem> {
        let atoms = &self.blocks[k].atoms;
        MolecularSystem::from_flat_bohr(
            atoms.iter().map(|&a| self.elements[a]).collect(),
            &Col::from_fn(3 * atoms.len(), |i| y[3 * atoms[i / 3] + i % 3]),
            SpinCharge::default(),
            None,
        )
    }

    /// Set up the coordinates of block `k` at `y`: those of the block alone when they span
    /// its `3 n` motions, else Cartesian.
    fn set_up(&mut self, k: usize, y: &[f64]) -> MoptResult<()> {
        let n = self.blocks[k].atoms.len();
        self.blocks[k].coords = None;
        if n < 3 {
            return Ok(());
        }
        let sys = self.block_system(k, y)?;
        let mut tric = Tric::delocalized();
        if tric.initialize(&sys).is_ok() && tric.dim(&sys) == 3 * n {
            self.blocks[k].coords = Some(Box::new(tric));
        }
        Ok(())
    }

    /// The entries of block `k` of a vector over the environment.
    fn part(&self, k: usize, v: &Col<f64>) -> Col<f64> {
        let atoms = &self.blocks[k].atoms;
        Col::from_fn(3 * atoms.len(), |i| v[3 * atoms[i / 3] + i % 3])
    }

    /// Put `part` of block `k` into `v`.
    fn put(&self, k: usize, part: &Col<f64>, v: &mut Col<f64>) {
        for (i, &a) in self.blocks[k].atoms.iter().enumerate() {
            for c in 0..3 {
                v[3 * a + c] = part[3 * i + c];
            }
        }
    }

    fn dimension(&self) -> usize {
        3 * self.elements.len()
    }

    /// The first-order Cartesian displacement of the step `dq` at the prepared point
    /// (`B^T G^+ dq` in each block; `B^-1 dq` for the square Wilson matrices of the blocks).
    pub fn step_linear(&self, dq: &Col<f64>) -> Col<f64> {
        let mut out = Col::zeros(self.dimension());
        for (k, block) in self.blocks.iter().enumerate() {
            let part = self.part(k, dq);
            let dx = match &block.frame {
                Some(f) => f.step_to_cartesian_linear(&part),
                None => part,
            };
            self.put(k, &dx, &mut out);
        }
        out
    }
}

impl InnerCoordinates for FragmentCoordinates {
    fn dim(&self) -> usize {
        self.dimension()
    }

    fn prepare(&mut self, y: &[f64]) -> MoptResult<()> {
        for k in 0..self.blocks.len() {
            let frame = match &self.blocks[k].coords {
                Some(c) => Some(CoordinateFrame::new(c.as_ref(), &self.block_system(k, y)?)?),
                None => None,
            };
            self.blocks[k].frame = frame;
        }
        Ok(())
    }

    fn gradient_to_q(&self, g: &Col<f64>) -> Col<f64> {
        let mut out = Col::zeros(self.dimension());
        for (k, block) in self.blocks.iter().enumerate() {
            let part = self.part(k, g);
            let q = match &block.frame {
                Some(f) => f.gradient_to_q(&part),
                None => part,
            };
            self.put(k, &q, &mut out);
        }
        out
    }

    fn step(&self, y: &[f64], dq: &Col<f64>) -> MoptResult<Vec<f64>> {
        let mut out = y.to_vec();
        let options = BackTransformOptions::default();
        for (k, block) in self.blocks.iter().enumerate() {
            let part = self.part(k, dq);
            let dx = match &block.coords {
                Some(c) => back_transform(c.as_ref(), &self.block_system(k, y)?, &part, &options)?,
                None => part,
            };
            for (i, &a) in block.atoms.iter().enumerate() {
                for c in 0..3 {
                    out[3 * a + c] += dx[3 * i + c];
                }
            }
        }
        Ok(out)
    }

    fn displacement(&self, to: &[f64], from: &[f64]) -> MoptResult<Col<f64>> {
        let mut out = Col::zeros(self.dimension());
        for (k, block) in self.blocks.iter().enumerate() {
            let d = match &block.coords {
                Some(c) => {
                    c.displacement(&self.block_system(k, to)?, &self.block_system(k, from)?)?
                }
                None => Col::from_fn(3 * block.atoms.len(), |i| {
                    let j = 3 * block.atoms[i / 3] + i % 3;
                    to[j] - from[j]
                }),
            };
            self.put(k, &d, &mut out);
        }
        Ok(out)
    }

    fn renew_if_needed(&mut self, y: &[f64]) -> MoptResult<bool> {
        let mut renewed = false;
        for k in 0..self.blocks.len() {
            let stale = match &self.blocks[k].coords {
                Some(c) => c.needs_reinitialization(&self.block_system(k, y)?),
                None => false,
            };
            if stale {
                self.set_up(k, y)?;
                renewed = true;
            }
        }
        if renewed {
            self.generation += 1;
        }
        Ok(renewed)
    }

    fn generation(&self) -> usize {
        self.generation
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::io::xyz::parse_xyz_str;

    #[test]
    fn waters_are_blocks_with_their_own_coordinates() {
        let text = "7\n\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\nO 3 0 0\nH 3.3 0.9 0\n\
                    H 3.3 -0.4 0.8\nNa 0 4 0\n";
        let sys = parse_xyz_str(text, SpinCharge::default(), None).unwrap();
        let y: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let mut c = FragmentCoordinates::new(sys.elements.clone(), &y, None).unwrap();
        // Two waters with internal coordinates, the ion Cartesian.
        assert_eq!(c.block_counts(), (2, 1));
        assert_eq!(c.dim(), 21);
        c.prepare(&y).unwrap();
        // A step and its displacement agree, and a zero step stays put.
        let dq = Col::from_fn(21, |i| 1e-3 * ((i * 7 % 5) as f64 - 2.0));
        let moved = c.step(&y, &dq).unwrap();
        let reached = c.displacement(&moved, &y).unwrap();
        assert!(crate::math::linalg::col_norm(&(&reached - &dq)) < 1e-6);
        let still = c.step(&y, &Col::zeros(21)).unwrap();
        assert!(still.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-12));
    }
}
