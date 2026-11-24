//! Molecular structure: elements, coordinates, charge/multiplicity, and periodic cell.
//!
//! Coordinates are stored in Bohr. The typed API ([`Position`], [`Quantity`], [`AtomId`]) keeps
//! units and atom numbering explicit; flat `3N` vectors (`coords_flat_bohr`) are in Bohr by
//! contract and are what optimizers work with.
//!
//! MultiOptPy: `optimization.py:OptimizationState` (geometry, elements, charge and multiplicity)
//!  and `fileio.py:xyz2list`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::atom_id::{atom_ids, AtomId};
use crate::domain::element::Element;
use crate::domain::pbc::PeriodicCell;
use crate::domain::quantity::{Bohr, LengthDim, Position, Quantity, Unit, Vector3};
use crate::error::{MoptError, MoptResult};

/// Total charge and spin multiplicity `(2S + 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpinCharge {
    /// Total charge (e.g. `0`, `-1`, `+1`).
    pub charge: i32,
    /// Spin multiplicity `2S + 1` (`1` = singlet, `2` = doublet, `3` = triplet).
    pub multiplicity: u32,
}

impl SpinCharge {
    /// Create a `(charge, multiplicity)` pair from trusted values; a multiplicity of `0` is
    /// raised to `1`. User input goes through [`Self::checked`].
    pub fn new(charge: i32, multiplicity: u32) -> Self {
        Self {
            charge,
            multiplicity: multiplicity.max(1),
        }
    }

    /// `(charge, multiplicity)` from user input. The multiplicity is always `2S + 1`, for every
    /// calculator (programs that expect `2S`, such as PySCF's `mol.spin`, get
    /// [`Self::unpaired_electrons`] from their adapter).
    pub fn checked(charge: i64, multiplicity: i64) -> MoptResult<Self> {
        let charge = i32::try_from(charge)
            .map_err(|_| MoptError::config(format!("charge {charge} is out of range")))?;
        if multiplicity < 1 {
            return Err(MoptError::config(format!(
                "spin multiplicity must be 2S + 1 >= 1, got {multiplicity} (PySCF's mol.spin = 2S \
                 corresponds to multiplicity mol.spin + 1)"
            )));
        }
        let multiplicity = u32::try_from(multiplicity).map_err(|_| {
            MoptError::config(format!("spin multiplicity {multiplicity} is out of range"))
        })?;
        Ok(Self {
            charge,
            multiplicity,
        })
    }

    /// The same state from the number of unpaired electrons `2S` (PySCF's `mol.spin`).
    pub fn from_unpaired_electrons(charge: i32, unpaired: u32) -> Self {
        Self {
            charge,
            multiplicity: unpaired + 1,
        }
    }

    /// Number of unpaired electrons `2S = multiplicity - 1` (PySCF's `mol.spin`).
    pub fn unpaired_electrons(&self) -> u32 {
        self.multiplicity - 1
    }

    /// Number of electrons of `elements` with this charge (all-electron count).
    pub fn electron_count(&self, elements: &[Element]) -> i64 {
        elements
            .iter()
            .map(|e| i64::from(e.atomic_number()))
            .sum::<i64>()
            - i64::from(self.charge)
    }

    /// Check that the charge and multiplicity are possible for `elements`: a non-negative
    /// electron count with the parity of `multiplicity - 1`, and at least `multiplicity - 1`
    /// electrons.
    pub fn check_electrons(&self, elements: &[Element]) -> MoptResult<()> {
        let n = self.electron_count(elements);
        let unpaired = i64::from(self.unpaired_electrons());
        if n < 0 {
            return Err(MoptError::config(format!(
                "charge {} leaves {n} electrons",
                self.charge
            )));
        }
        if unpaired > n || (n - unpaired) % 2 != 0 {
            return Err(MoptError::config(format!(
                "{n} electrons cannot have spin multiplicity {} (2S = {unpaired} unpaired \
                 electrons); check -elec / -spin (the multiplicity is 2S + 1 for every calculator)",
                self.multiplicity
            )));
        }
        Ok(())
    }
}

impl Default for SpinCharge {
    fn default() -> Self {
        Self {
            charge: 0,
            multiplicity: 1,
        }
    }
}

/// Complete molecular state.
#[derive(Debug, Clone)]
pub struct MolecularSystem {
    /// Elements of the `N` atoms, in atom order.
    pub elements: Vec<Element>,
    /// Cartesian coordinates `(N, 3)` in Bohr.
    coords: Mat<f64>,
    /// Total charge and spin multiplicity.
    pub spin_charge: SpinCharge,
    /// 1D/2D/3D periodic cell (`None` for isolated systems).
    pub cell: Option<PeriodicCell>,
    /// Title or comment line.
    pub comment: String,
}

impl MolecularSystem {
    /// Construct from elements and positions in any length unit.
    pub fn new<U: Unit<Dim = LengthDim>>(
        elements: Vec<Element>,
        positions: &[Position<U>],
        spin_charge: SpinCharge,
        cell: Option<PeriodicCell>,
    ) -> MoptResult<Self> {
        if elements.len() != positions.len() {
            return Err(MoptError::coord(format!(
                "{} elements but {} positions",
                elements.len(),
                positions.len()
            )));
        }
        let bohr: Vec<[f64; 3]> = positions.iter().map(|p| p.to::<Bohr>().value()).collect();
        let coords = Mat::from_fn(bohr.len(), 3, |i, j| bohr[i][j]);
        Ok(Self {
            elements,
            coords,
            spin_charge,
            cell,
            comment: String::new(),
        })
    }

    /// Select atoms in the given order for a trajectory frame, preserving cell and comment.
    pub fn trajectory_atoms(&self, atoms: &[AtomId]) -> MoptResult<Self> {
        let mut elements = Vec::with_capacity(atoms.len());
        let mut positions = Vec::with_capacity(atoms.len());
        for &atom in atoms {
            atom.check(self.n_atoms())?;
            elements.push(self.element(atom));
            positions.push(self.position(atom));
        }
        let mut frame = Self::new(elements, &positions, self.spin_charge, self.cell.clone())?;
        frame.comment.clone_from(&self.comment);
        Ok(frame)
    }

    /// Construct from elements and a flat `3N` coordinate vector in Bohr.
    pub fn from_flat_bohr(
        elements: Vec<Element>,
        flat: &Col<f64>,
        spin_charge: SpinCharge,
        cell: Option<PeriodicCell>,
    ) -> MoptResult<Self> {
        if flat.nrows() != 3 * elements.len() {
            return Err(MoptError::coord(format!(
                "expected {} coordinates, got {}",
                3 * elements.len(),
                flat.nrows()
            )));
        }
        let coords = Mat::from_fn(elements.len(), 3, |i, j| flat[3 * i + j]);
        Ok(Self {
            elements,
            coords,
            spin_charge,
            cell,
            comment: String::new(),
        })
    }

    /// Number of atoms `N`.
    #[inline]
    pub fn n_atoms(&self) -> usize {
        self.elements.len()
    }

    /// Cartesian degrees of freedom `3N`.
    #[inline]
    pub fn dof(&self) -> usize {
        self.elements.len() * 3
    }

    /// Ids of all atoms.
    #[inline]
    pub fn atoms(&self) -> impl ExactSizeIterator<Item = AtomId> + Clone {
        atom_ids(self.n_atoms())
    }

    /// Element of `atom`.
    #[inline]
    pub fn element(&self, atom: AtomId) -> Element {
        self.elements[atom.index()]
    }

    /// Return `true` if a cell with at least one periodic axis is attached.
    #[inline]
    pub fn has_pbc(&self) -> bool {
        self.cell
            .as_ref()
            .is_some_and(PeriodicCell::is_any_periodic)
    }

    /// The attached cell, if it has at least one periodic axis.
    #[inline]
    pub fn active_cell(&self) -> Option<&PeriodicCell> {
        self.cell.as_ref().filter(|c| c.is_any_periodic())
    }

    /// Position of `atom`.
    #[inline]
    pub fn position(&self, atom: AtomId) -> Position<Bohr> {
        let i = atom.index();
        Position::new([
            self.coords[(i, 0)],
            self.coords[(i, 1)],
            self.coords[(i, 2)],
        ])
    }

    /// Set the position of `atom`.
    #[inline]
    pub fn set_position(&mut self, atom: AtomId, pos: Position<Bohr>) {
        for (c, v) in pos.value().into_iter().enumerate() {
            self.coords[(atom.index(), c)] = v;
        }
    }

    /// Positions of all atoms.
    pub fn positions(&self) -> Vec<Position<Bohr>> {
        self.atoms().map(|a| self.position(a)).collect()
    }

    /// Coordinates `(N, 3)` in Bohr.
    #[inline]
    pub fn coords_bohr(&self) -> &Mat<f64> {
        &self.coords
    }

    /// Flatten coordinates into a `3N` column vector (Bohr).
    pub fn coords_flat_bohr(&self) -> Col<f64> {
        Col::from_fn(self.dof(), |k| self.coords[(k / 3, k % 3)])
    }

    /// Clone with coordinates replaced by a flat `3N` vector (Bohr).
    pub fn with_flat_coords_bohr(&self, flat: &Col<f64>) -> MoptResult<Self> {
        let mut next = self.clone();
        next.set_flat_coords_bohr(flat)?;
        Ok(next)
    }

    /// Overwrite coordinates in place from a flat `3N` vector (Bohr).
    pub fn set_flat_coords_bohr(&mut self, flat: &Col<f64>) -> MoptResult<()> {
        if flat.nrows() != self.dof() {
            return Err(MoptError::coord(format!(
                "expected flat coordinate length {}, got {}",
                self.dof(),
                flat.nrows()
            )));
        }
        for k in 0..self.dof() {
            self.coords[(k / 3, k % 3)] = flat[k];
        }
        Ok(())
    }

    /// Minimum-image displacement `r_i - r_j` ([`PeriodicCell::mic_delta`] when periodic).
    #[inline]
    pub fn delta(&self, i: AtomId, j: AtomId) -> Vector3<Bohr> {
        let ri = self.position(i).value();
        let rj = self.position(j).value();
        Vector3::new(match self.active_cell() {
            Some(cell) => cell.mic_delta(ri, rj),
            None => [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]],
        })
    }

    /// Minimum-image distance `||r_i - r_j||`.
    #[inline]
    pub fn distance(&self, i: AtomId, j: AtomId) -> Quantity<Bohr> {
        self.delta(i, j).norm()
    }

    /// Atomic masses (`amu`), length `N`.
    pub fn masses_amu(&self) -> Vec<f64> {
        self.elements.iter().map(|e| e.mass_amu()).collect()
    }

    /// `sqrt(mass_amu)` repeated per Cartesian component (length `3N`) for mass weighting.
    pub fn sqrt_masses_3n(&self) -> Col<f64> {
        Col::from_fn(self.dof(), |k| libm::sqrt(self.elements[k / 3].mass_amu()))
    }

    /// Center of mass (meaningful for isolated or unwrapped coordinates).
    pub fn center_of_mass(&self) -> Position<Bohr> {
        let mut tot_m = 0.0;
        let mut com = [0.0; 3];
        for a in self.atoms() {
            let m = self.element(a).mass_amu();
            let p = self.position(a).value();
            tot_m += m;
            for c in 0..3 {
                com[c] += m * p[c];
            }
        }
        Position::new(if tot_m > 0.0 {
            com.map(|v| v / tot_m)
        } else {
            [0.0; 3]
        })
    }

    /// Unweighted centroid.
    pub fn centroid(&self) -> Position<Bohr> {
        let n = self.n_atoms().max(1) as f64;
        let mut c = [0.0; 3];
        for a in self.atoms() {
            let p = self.position(a).value();
            for k in 0..3 {
                c[k] += p[k] / n;
            }
        }
        Position::new(c)
    }

    /// Translate all atoms so that the center of mass is at the origin.
    pub fn recenter_com(&mut self) {
        let com = self.center_of_mass();
        for a in self.atoms() {
            let p = self.position(a) - com;
            self.set_position(a, p);
        }
    }

    /// Clone with every atom moved to the periodic image closest to the same atom in `reference`.
    ///
    /// Required for periodic path interpolation, NEB, GSM, FSM, DMF, DS-AFIR, BITSS, and IRC.
    pub fn unwrapped_to(&self, reference: &Self) -> MoptResult<Self> {
        if self.n_atoms() != reference.n_atoms() {
            return Err(MoptError::coord(
                "cannot unwrap structures with different atom counts",
            ));
        }
        let mut out = self.clone();
        if let Some(cell) = self.active_cell().or(reference.active_cell()) {
            for a in self.atoms() {
                let r_ref = reference.position(a).value();
                let d = cell.mic_delta(self.position(a).value(), r_ref);
                out.set_position(
                    a,
                    Position::new([r_ref[0] + d[0], r_ref[1] + d[1], r_ref[2] + d[2]]),
                );
            }
        }
        Ok(out)
    }

    /// Structure made of the listed atoms (in the given order), keeping the cell.
    pub fn subsystem(&self, atoms: &[AtomId], spin_charge: SpinCharge) -> MoptResult<Self> {
        let n = self.n_atoms();
        let ids = atoms
            .iter()
            .map(|a| a.check(n))
            .collect::<MoptResult<Vec<_>>>()?;
        let mut sub = Self::new(
            ids.iter().map(|a| self.element(*a)).collect(),
            &ids.iter().map(|a| self.position(*a)).collect::<Vec<_>>(),
            spin_charge,
            self.cell.clone(),
        )?;
        sub.comment = self.comment.clone();
        Ok(sub)
    }

    /// Minimum-image `3N` displacement `self - other` in Bohr.
    pub fn displacement_from(&self, other: &Self) -> MoptResult<Col<f64>> {
        if self.n_atoms() != other.n_atoms() {
            return Err(MoptError::coord(
                "cannot compute displacement between structures with different atom counts",
            ));
        }
        let cell = self.active_cell().or(other.active_cell());
        let mut diff = Col::zeros(self.dof());
        for a in self.atoms() {
            let ri = self.position(a).value();
            let rj = other.position(a).value();
            let d = match cell {
                Some(cell) => cell.mic_delta(ri, rj),
                None => [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]],
            };
            for c in 0..3 {
                diff[a.dof(c)] = d[c];
            }
        }
        Ok(diff)
    }
}
