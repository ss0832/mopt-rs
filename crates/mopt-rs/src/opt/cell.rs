//! Variable-cell optimization: the atoms and the lattice vectors as one set of variables on
//! which any of the optimization methods can work.
//!
//! The variables are those of Gubler, Krummenacher, Huber, and Goedecker, "Efficient
//! variable cell shape geometry optimization", J. Comput. Phys. X 2023, 17, 100131 (their
//! section 3.2 and section 4), chosen so that the Hessian is well conditioned whatever the
//! number of atoms and the shape of the cell:
//!
//! - atoms: quasi-Cartesian coordinates `q_i = s_i h_0`, the fractional coordinates `s_i`
//!   in the starting cell `h_0`. They do not change when the lattice changes, and they are
//!   lengths, unlike fractional coordinates, whose curvatures grow with the length of the
//!   lattice vectors;
//! - lattice: `A_k = gamma a_k / |a_k0|` for the three lattice vectors `a_k`, with
//!   `gamma = w sqrt(N)` for `N` atoms. Without the factor the curvatures of the lattice
//!   grow with the size of the cell (linearly with `N` for flat and long cells) while those
//!   of the atoms do not.
//!
//! With the gradient `g_i` of the atoms, the stress `sigma`, and the pressure `P`:
//!
//! ```text
//! H = E + P V                                   (enthalpy, the function that is minimized)
//! dH/dq_i = (h_0^-1 h) g_i
//! dH/dA_k = (|a_k0| / gamma) [V (h^-T sigma)_k + P V b_k]
//! ```
//!
//! with the lattice vectors as the rows of `h` and `b_k` the reciprocal vectors
//! (`b_k . a_l = delta_kl`). The lattice variables contain the rotations of the crystal,
//! along which the energy does not change.
//!
//! Slabs and wires follow Takagi et al. (Phys. Rev. B 2017, 95, 184110, section II.B):
//! the two lattice vectors of a slab vary within their plane (four lattice variables), the
//! one of a wire along its line (one); the other vectors of the cell stay as they are and
//! the atoms keep their fractional coordinates in the whole cell. With the translations
//! and rotations of the whole structure taken out ([`VariableCell::rigid_modes`]) this
//! leaves `3N + 3`, `3N`, and `3N - 3` degrees of freedom, as there. A pressure needs a
//! cell that is periodic in three directions.
//!
//! The Hessian in these variables comes from the analytic Hessian of the atoms and the
//! second derivatives along the strain where the calculator and the bias potentials give
//! them ([`VariableCell::lattice_columns`]), from differences of the gradient otherwise
//! ([`numerical_hessian`]); with it, exact-Hessian steps and searches for saddle points of the
//! atoms and the cell together work as for molecules.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::{
    positions, Bohr, MolecularSystem, PeriodicCell, StrainHessian, Units, Vector3,
};
use crate::error::{MoptError, MoptResult};
use crate::pes::bias::{afir_alpha, Ramp};
use crate::pes::stress::Stress;

/// `w` of `gamma = w sqrt(N)` (Bohr): the paper finds values between 1 and 2 suitable and
/// uses 2.
pub const DEFAULT_LATTICE_WEIGHT: f64 = 2.0;

/// `Hartree / Bohr^3` in GPa (`Units::GPA_PER_AU_PRESSURE`).
pub fn gigapascal_per_atomic_unit() -> f64 {
    Units::GPA_PER_AU_PRESSURE
}

/// `dF / dh` (the lattice vectors as the rows of `h`, Bohr; the fractional coordinates of the
/// atoms kept) of a function whose stress at `cell` is `stress`, `sigma_ab = (1/V) dF/d eps_ab`
/// under `r -> (1 + eps) r`: `dF/dh_ka = V sum_b (b_k)_b sigma_ab` with the reciprocal vectors
/// `b_k`. For a symmetric stress (any function that does not depend on the orientation in
/// space) this is [`crate::pes::stress::cell_gradient`]; the stress of a function of the Cartesian components of
/// the lattice vectors (`-uc "cell_cz()"`) has an antisymmetric part as well, which enters
/// here.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn lattice_gradient(stress: &Stress, cell: &PeriodicCell) -> [[f64; 3]; 3] {
    let volume = cell.volume_bohr3();
    let reciprocal = cell.h_inv_bohr;
    // sigma_ab directly: for a symmetric stress the same products and sums as
    // `cell_gradient` (which contracts sigma_ba), so the same bits.
    std::array::from_fn(|k| {
        std::array::from_fn(|a| {
            volume * (0..3).map(|b| reciprocal[k][b] * stress[a][b]).sum::<f64>()
        })
    })
}

/// The map between a periodic structure and the variables of its variable-cell
/// optimization.
#[derive(Debug, Clone)]
pub struct VariableCell {
    /// The starting structure (elements, state, and the cell `h_0`).
    start: MolecularSystem,
    /// Lengths of the starting lattice vectors.
    lengths: [f64; 3],
    /// `w sqrt(N)`.
    gamma: f64,
    /// The periodic directions, whose lattice vectors are variables.
    periodic: Vec<usize>,
    /// The lattice variables: a lattice vector and the unit direction along which it varies.
    directions: Vec<(usize, [f64; 3])>,
    /// External pressure (`Hartree / Bohr^3`).
    pressure: f64,
}

impl VariableCell {
    /// For the periodic structure `start`, the weight `w` of the lattice variables, and an
    /// external pressure in `Hartree / Bohr^3` (cells periodic in three directions only).
    pub fn new(start: &MolecularSystem, weight: f64, pressure: f64) -> MoptResult<Self> {
        let cell = start.active_cell().ok_or_else(|| {
            MoptError::config("a variable-cell optimization needs a periodic cell")
        })?;
        let periodic: Vec<usize> = (0..3).filter(|k| cell.periodic_axes[*k]).collect();
        if pressure != 0.0 && periodic.len() != 3 {
            return Err(MoptError::config(
                "a pressure needs a cell that is periodic in three directions",
            ));
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(MoptError::config(format!(
                "the weight of the lattice variables must be positive, got {weight}"
            )));
        }
        if !pressure.is_finite() {
            return Err(MoptError::config("the pressure must be a number"));
        }
        let lengths: [f64; 3] = std::array::from_fn(|k| {
            let v = cell.lattice_vector(k).value();
            libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        });
        // Where the lattice vectors may go: anywhere for a crystal, within the plane of the
        // two periodic vectors for a slab, along the periodic vector for a wire.
        let vector = |k: usize| cell.lattice_vector(k).value();
        let unit = |v: [f64; 3]| -> MoptResult<[f64; 3]> {
            let length = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if length < 1e-10 {
                return Err(MoptError::config(
                    "the periodic lattice vectors must be independent",
                ));
            }
            Ok(v.map(|c| c / length))
        };
        let frame: Vec<[f64; 3]> = match periodic.as_slice() {
            [k] => vec![unit(vector(*k))?],
            [k, l] => {
                let first = unit(vector(*k))?;
                let other = vector(*l);
                let along: f64 = (0..3).map(|c| first[c] * other[c]).sum();
                let second = unit(std::array::from_fn(|c| other[c] - along * first[c]))?;
                vec![first, second]
            }
            _ => vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let directions = periodic
            .iter()
            .flat_map(|k| frame.iter().map(move |e| (*k, *e)))
            .collect();
        Ok(Self {
            start: start.clone(),
            lengths,
            gamma: weight * libm::sqrt(start.n_atoms() as f64),
            periodic,
            directions,
            pressure,
        })
    }

    /// Number of variables: three per atom, and nine, four, or one of the lattice for a
    /// crystal, a slab, or a wire.
    pub fn dimension(&self) -> usize {
        self.start.dof() + self.directions.len()
    }

    /// Names of the variables: `q<atom><x|y|z>` for the atoms; for the lattice
    /// `A<a|b|c><x|y|z>` in a crystal and `A<a|b|c><1|2>` (the directions within the plane
    /// or along the line) in a slab or a wire.
    pub fn names(&self) -> Vec<String> {
        let axes = ["x", "y", "z"];
        let mut names = Vec::with_capacity(self.dimension());
        for atom in 1..=self.start.n_atoms() {
            names.extend(axes.iter().map(|c| format!("q{atom}{c}")));
        }
        let per_vector = self.directions.len() / self.periodic.len().max(1);
        for (slot, (k, _)) in self.directions.iter().enumerate() {
            let which = slot % per_vector.max(1);
            let tag = if self.periodic.len() == 3 {
                axes[which].to_string()
            } else {
                (which + 1).to_string()
            };
            names.push(format!("A{}{tag}", ["a", "b", "c"][*k]));
        }
        names
    }

    /// External pressure (`Hartree / Bohr^3`).
    pub fn pressure(&self) -> f64 {
        self.pressure
    }

    /// The lattice vectors that are variables (0, 1, 2 for `a`, `b`, `c`): all three for a
    /// crystal, the two periodic ones of a slab, the one of a wire.
    pub fn variable_vectors(&self) -> &[usize] {
        &self.periodic
    }

    /// Number of lattice variables (nine, four, or one).
    pub fn lattice_dimension(&self) -> usize {
        self.directions.len()
    }

    /// Masses of the variables for a path in them (an IRC of the atoms and the cell), one
    /// per variable: the atomic masses (amu) for the coordinates of the atoms. A change
    /// `dA` of a lattice variable of the vector `k` moves the vector by `|a_k0| / gamma dA`
    /// and every atom, which keeps its fractional coordinates, by `s_ik` times that; the
    /// mass of the variable is the diagonal of that kinetic metric about the centre of mass,
    /// `(|a_k0| / gamma)^2 sum_i m_i (s_ik - s_k)^2` (`s_k` the mass-weighted mean of the
    /// fractional coordinates), and not less than that of the mean atom spread evenly over
    /// the cell (a variance of 1/12). The couplings between the variables of the atoms and
    /// of the lattice in that metric are left out.
    pub fn masses(&self) -> MoptResult<Vec<f64>> {
        let reference = self.reference()?;
        let m = self.start.masses_amu();
        let total: f64 = m.iter().sum();
        let atoms = m.len().max(1) as f64;
        let fractions: Vec<[f64; 3]> = self
            .start
            .atoms()
            .map(|a| reference.cart_to_frac(self.start.position(a).value()))
            .collect();
        let spread: [f64; 3] = std::array::from_fn(|k| {
            let mean = m.iter().zip(&fractions).map(|(w, s)| w * s[k]).sum::<f64>() / total;
            let spread: f64 = m
                .iter()
                .zip(&fractions)
                .map(|(w, s)| w * (s[k] - mean) * (s[k] - mean))
                .sum();
            spread.max(total / atoms / 12.0)
        });
        let mut out: Vec<f64> = m.iter().flat_map(|w| [*w; 3]).collect();
        for (k, _) in &self.directions {
            let scale = self.lengths[*k] / self.gamma;
            out.push(spread[*k] * scale * scale);
        }
        Ok(out)
    }

    fn reference(&self) -> MoptResult<&PeriodicCell> {
        self.start
            .active_cell()
            .ok_or_else(|| MoptError::config("the starting structure has lost its cell"))
    }

    /// The variables of `sys` (the same atoms in a cell periodic in three directions).
    pub fn variables(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("the structure has no periodic cell"))?;
        if sys.n_atoms() != self.start.n_atoms() {
            return Err(MoptError::config(
                "the structure has another number of atoms than the starting one",
            ));
        }
        let reference = self.reference()?;
        let n = sys.dof();
        let mut x = Col::zeros(self.dimension());
        for atom in sys.atoms() {
            let q = reference.frac_to_cart(cell.cart_to_frac(sys.position(atom).value()));
            for (c, value) in q.iter().enumerate() {
                x[atom.dof(c)] = *value;
            }
        }
        for (slot, (k, e)) in self.directions.iter().enumerate() {
            let v = cell.lattice_vector(*k).value();
            let along: f64 = (0..3).map(|c| v[c] * e[c]).sum();
            x[n + slot] = self.gamma * along / self.lengths[*k];
        }
        Ok(x)
    }

    /// The structure of the variables `x`.
    pub fn system(&self, x: &Col<f64>) -> MoptResult<MolecularSystem> {
        if x.nrows() != self.dimension() {
            return Err(MoptError::config(format!(
                "{} variables given, {} expected",
                x.nrows(),
                self.dimension()
            )));
        }
        let reference = self.reference()?;
        let n = self.start.dof();
        // The vectors of the periodic directions from the variables (their components along
        // the directions they may vary in; any other component as at the start), the other
        // vectors as at the start.
        let mut vectors: [[f64; 3]; 3] =
            std::array::from_fn(|k| reference.lattice_vector(k).value());
        for (slot, (k, e)) in self.directions.iter().enumerate() {
            let before: f64 = (0..3)
                .map(|c| reference.lattice_vector(*k).value()[c] * e[c])
                .sum();
            let now = x[n + slot] * self.lengths[*k] / self.gamma;
            for c in 0..3 {
                vectors[*k][c] += (now - before) * e[c];
            }
        }
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Bohr>::new(vectors[0]),
            Vector3::<Bohr>::new(vectors[1]),
            Vector3::<Bohr>::new(vectors[2]),
            reference.periodic_axes,
        )?;
        let rows: Vec<[f64; 3]> = self
            .start
            .atoms()
            .map(|atom| {
                let q: [f64; 3] = std::array::from_fn(|c| x[atom.dof(c)]);
                cell.frac_to_cart(reference.cart_to_frac(q))
            })
            .collect();
        MolecularSystem::new(
            self.start.elements.clone(),
            &positions::<Bohr>(&rows),
            self.start.spin_charge,
            Some(cell),
        )
    }

    /// Add a gradient with respect to the lattice vectors (rows, the fractional
    /// coordinates of the atoms kept) to a gradient in the variables.
    pub fn add_lattice_gradient(&self, gradient: &mut Col<f64>, of_lattice: &[[f64; 3]; 3]) {
        let n = self.start.dof();
        for (slot, (k, e)) in self.directions.iter().enumerate() {
            let along: f64 = (0..3).map(|c| of_lattice[*k][c] * e[c]).sum();
            gradient[n + slot] += along * self.lengths[*k] / self.gamma;
        }
    }

    /// Whether the lattice vectors of the distances of `force` are all variables.
    pub fn takes(&self, force: &CellForce) -> bool {
        force
            .distances
            .iter()
            .flat_map(|d| d.vectors())
            .all(|k| self.periodic.contains(&k))
    }

    /// Orthonormal directions in the variables along which the structure of `x` moves as a
    /// whole, so that the energy does not change: the three translations of the atoms, and
    /// the rotations of the whole structure that the variables can describe. These are the
    /// three rotations of the lattice for a crystal; the rotation of the lattice about the
    /// normal of a slab if the vector that is not periodic lies along that normal (so that
    /// the atoms turn with the lattice); and the rotation of the atoms about the axis of a
    /// wire.
    pub fn rigid_modes(&self, x: &Col<f64>) -> MoptResult<Vec<Col<f64>>> {
        let sys = self.system(x)?;
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("the structure has no periodic cell"))?;
        let reference = self.reference()?;
        let n = self.start.dof();
        let size = self.dimension();
        let atoms = self.start.n_atoms().max(1) as f64;
        let mut modes: Vec<Col<f64>> = (0..3)
            .map(|c| {
                Col::from_fn(size, |i| {
                    if i < n && i % 3 == c {
                        1.0 / libm::sqrt(atoms)
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let cross = |u: [f64; 3], v: [f64; 3]| {
            [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
        };
        let norm = |v: [f64; 3]| libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        // Rotations of the lattice with the atoms following through their fractional
        // coordinates (the vectors that are not variables must lie along the axis).
        let lattice_axes: Vec<[f64; 3]> = match self.periodic.as_slice() {
            [_, _, _] => vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [k, l] => {
                let normal = cross(
                    cell.lattice_vector(*k).value(),
                    cell.lattice_vector(*l).value(),
                );
                let fixed = (0..3)
                    .find(|m| !self.periodic.contains(m))
                    .map(|m| cell.lattice_vector(m).value());
                match fixed {
                    Some(v) if norm(cross(normal, v)) < 1e-8 * norm(normal) * norm(v) => {
                        vec![normal.map(|c| c / norm(normal))]
                    }
                    _ => Vec::new(),
                }
            }
            _ => Vec::new(),
        };
        let mut candidates: Vec<Col<f64>> = Vec::new();
        for axis in lattice_axes {
            let mut mode = Col::zeros(size);
            for (slot, (k, e)) in self.directions.iter().enumerate() {
                let turned = cross(axis, cell.lattice_vector(*k).value());
                let along: f64 = (0..3).map(|c| turned[c] * e[c]).sum();
                mode[n + slot] = self.gamma * along / self.lengths[*k];
            }
            candidates.push(mode);
        }
        // A wire: the atoms turn about its axis; the lattice stays.
        if let [k] = self.periodic.as_slice() {
            let v = cell.lattice_vector(*k).value();
            let axis = v.map(|c| c / norm(v));
            let mut mode = Col::zeros(size);
            for atom in sys.atoms() {
                let turned = cross(axis, sys.position(atom).value());
                // Displacements map like positions: q = x h^-1 h_0.
                let q = reference.frac_to_cart(cell.cart_to_frac(turned));
                for (c, value) in q.iter().enumerate() {
                    mode[atom.dof(c)] = *value;
                }
            }
            candidates.push(mode);
        }
        for mut mode in candidates {
            // Orthogonal to the modes so far, then of unit length.
            for other in &modes {
                let overlap: f64 = (0..size).map(|i| other[i] * mode[i]).sum();
                for i in 0..size {
                    mode[i] -= overlap * other[i];
                }
            }
            let length = libm::sqrt((0..size).map(|i| mode[i] * mode[i]).sum::<f64>());
            if length > 1e-10 {
                modes.push(Col::from_fn(size, |i| mode[i] / length));
            }
        }
        Ok(modes)
    }

    /// [`Self::rigid_modes`] limited to the motions that `symmetries` keeps: the three
    /// translations of the atoms, the rotations of the whole structure, both, or none.
    pub fn rigid_modes_with(
        &self,
        x: &Col<f64>,
        symmetries: CellSymmetries,
    ) -> MoptResult<Vec<Col<f64>>> {
        let modes = self.rigid_modes(x)?;
        // The first three are the translations; the rest the rotations, orthogonal to them.
        Ok(modes
            .into_iter()
            .enumerate()
            .filter(|(k, _)| {
                if *k < 3 {
                    symmetries.translations
                } else {
                    symmetries.rotations
                }
            })
            .map(|(_, m)| m)
            .collect())
    }

    /// The gradient in the variables of a function of the structure `sys` (no `P V`): from its
    /// Cartesian gradient `gradient` and its derivative along the strain `stress` (`(1/V)
    /// dF/d eps`, the fractional coordinates of the atoms kept), mapped as
    /// [`Self::objective`] maps those of the energy. The gradient of a constraint of the
    /// variable cell ([`crate::opt::cell_constraints`]).
    pub fn gradient_in_variables(
        &self,
        sys: &MolecularSystem,
        gradient: &Col<f64>,
        stress: &Stress,
    ) -> MoptResult<Col<f64>> {
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("the structure has no periodic cell"))?;
        let n = sys.dof();
        if gradient.nrows() != n {
            return Err(MoptError::config("the gradient does not fit the structure"));
        }
        let map = self.atoms_map(cell)?;
        let mut out = Col::zeros(self.dimension());
        for atom in sys.atoms() {
            for a in 0..3 {
                out[atom.dof(a)] = (0..3)
                    .map(|b| map[a][b] * gradient[atom.dof(b)])
                    .sum::<f64>();
            }
        }
        let of_function = lattice_gradient(stress, cell);
        for (slot, (k, e)) in self.directions.iter().enumerate() {
            let along: f64 = (0..3).map(|c| of_function[*k][c] * e[c]).sum();
            out[n + slot] = along * self.lengths[*k] / self.gamma;
        }
        Ok(out)
    }

    /// The linear map from the variables of the atoms to their Cartesian coordinates at the
    /// cell `cell`: `r = q m` for every atom (row vectors), `m = h_0^-1 h`; the gradient of the
    /// variables of an atom is `m g` of its Cartesian gradient `g`, and the Hessian of two
    /// atoms at a fixed cell `m H m^T` of their Cartesian block.
    pub fn atoms_map(&self, cell: &PeriodicCell) -> MoptResult<[[f64; 3]; 3]> {
        let reference = self.reference()?;
        let lattice: [[f64; 3]; 3] = std::array::from_fn(|k| cell.lattice_vector(k).value());
        // (h_0^-1)_ak is the component a of the reciprocal vector k of the starting cell.
        Ok(std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                (0..3)
                    .map(|k| reference.h_inv_bohr[k][a] * lattice[k][b])
                    .sum::<f64>()
            })
        }))
    }

    /// The columns of the lattice variables of the Hessian of the enthalpy at `sys`, from
    /// the Cartesian gradient `gradient` of the atoms and the second derivatives `strain`
    /// along the strain (of the energy with its bias potentials and forces on the cell): the
    /// rows of the atoms (`3N x L`) and the block of the lattice (`L x L`, with the second
    /// derivative of `P V`).
    ///
    /// With the atoms at `r_i = q_i m`, `m = h_0^-1 h`, a change `dh` of the lattice at fixed
    /// fractional coordinates is the strain `eps_ab = sum_k (h^-1)_bk dh_ka`, linear in `dh`.
    /// So `d2E/(dh_kc dh_ld) = sum_bb' K_(cb)(db') (h^-1)_bk (h^-1)_b'l` with `K` the second
    /// derivatives along the strain, and the derivative of the gradient of the variables of
    /// the atoms, `(dE/dq_i)_a = sum_b m_ab g_ib`, is
    /// `(h_0^-1)_ak g_ic + sum_bb' m_ab M_(ib)(cb') (h^-1)_b'k` with `M` the mixed block. The
    /// lattice variables are linear in `h` (`dh_kc / dA = |a_k0| / gamma e_c`), and
    /// `d2V/(dh_kc dh_ld) = V [(h^-1)_ck (h^-1)_dl - (h^-1)_cl (h^-1)_dk]`.
    pub fn lattice_columns(
        &self,
        sys: &MolecularSystem,
        gradient: &Col<f64>,
        strain: &StrainHessian,
    ) -> MoptResult<(Mat<f64>, Mat<f64>)> {
        self.lattice_columns_at(sys, gradient, strain, self.pressure)
    }

    /// The Hessian in the variables of a function of the structure (no `P V`): its
    /// Cartesian gradient `gradient` and Hessian `hessian` and its second derivatives `strain`
    /// along the strain, mapped as [`Self::lattice_columns`] maps those of the energy (the
    /// block of the atoms `m H m^T`). The curvature of a constraint of the variable cell
    /// ([`crate::opt::cell_constraints`]).
    pub fn hessian_in_variables(
        &self,
        sys: &MolecularSystem,
        gradient: &Col<f64>,
        hessian: &Mat<f64>,
        strain: &StrainHessian,
    ) -> MoptResult<Mat<f64>> {
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("the structure has no periodic cell"))?;
        let n = sys.dof();
        if hessian.nrows() != n || hessian.ncols() != n {
            return Err(MoptError::config("the Hessian does not fit the structure"));
        }
        let m = self.atoms_map(cell)?;
        let (rows, block) = self.lattice_columns_at(sys, gradient, strain, 0.0)?;
        let dim = self.dimension();
        let mut out = Mat::<f64>::zeros(dim, dim);
        for i in 0..n / 3 {
            for j in 0..n / 3 {
                for a in 0..3 {
                    for c in 0..3 {
                        let mut v = 0.0;
                        for b in 0..3 {
                            for d in 0..3 {
                                v += m[a][b] * hessian[(3 * i + b, 3 * j + d)] * m[c][d];
                            }
                        }
                        out[(3 * i + a, 3 * j + c)] = v;
                    }
                }
            }
        }
        for l in n..dim {
            for i in 0..n {
                out[(i, l)] = rows[(i, l - n)];
                out[(l, i)] = rows[(i, l - n)];
            }
            for k in n..dim {
                out[(k, l)] = 0.5 * (block[(k - n, l - n)] + block[(l - n, k - n)]);
            }
        }
        Ok(out)
    }

    /// [`Self::lattice_columns`] with the pressure `pressure` in place of that of the cell.
    fn lattice_columns_at(
        &self,
        sys: &MolecularSystem,
        gradient: &Col<f64>,
        strain: &StrainHessian,
        pressure: f64,
    ) -> MoptResult<(Mat<f64>, Mat<f64>)> {
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("the structure has no periodic cell"))?;
        let n = sys.dof();
        if gradient.nrows() != n || strain.mixed.nrows() != n {
            return Err(MoptError::config(
                "the gradient or the strain Hessian does not fit the structure",
            ));
        }
        let reference = self.reference()?;
        let m = self.atoms_map(cell)?;
        // (h^-1)_bk = hinv[k][b], (h_0^-1)_ak = h0inv[k][a].
        let hinv = cell.h_inv_bohr;
        let h0inv = reference.h_inv_bohr;
        let volume = cell.volume_bohr3();
        let slots = self.directions.len();
        let scale = |k: usize| self.lengths[k] / self.gamma;
        let mut rows = Mat::<f64>::zeros(n, slots);
        for i in 0..n / 3 {
            // d (dE/dq_ia) / dh_kc.
            let of_lattice = |a: usize, k: usize, c: usize| -> f64 {
                let mut v = h0inv[k][a] * gradient[3 * i + c];
                for (b, mab) in m[a].iter().enumerate() {
                    for (b2, hk) in hinv[k].iter().enumerate() {
                        v += mab * strain.mixed[(3 * i + b, 3 * c + b2)] * hk;
                    }
                }
                v
            };
            for (slot, (k, e)) in self.directions.iter().enumerate() {
                for a in 0..3 {
                    rows[(3 * i + a, slot)] =
                        scale(*k) * (0..3).map(|c| e[c] * of_lattice(a, *k, c)).sum::<f64>();
                }
            }
        }
        let of_lattices = |k: usize, c: usize, l: usize, d: usize| -> f64 {
            let mut v = 0.0;
            for (b, hk) in hinv[k].iter().enumerate() {
                for (b2, hl) in hinv[l].iter().enumerate() {
                    v += strain.strain[3 * c + b][3 * d + b2] * hk * hl;
                }
            }
            v + pressure * volume * (hinv[k][c] * hinv[l][d] - hinv[l][c] * hinv[k][d])
        };
        let mut block = Mat::<f64>::zeros(slots, slots);
        for (s, (k, e)) in self.directions.iter().enumerate() {
            for (t, (l, f)) in self.directions.iter().enumerate() {
                let mut v = 0.0;
                for (c, ec) in e.iter().enumerate() {
                    for (d, fd) in f.iter().enumerate() {
                        v += ec * fd * of_lattices(*k, c, *l, d);
                    }
                }
                block[(s, t)] = scale(*k) * scale(*l) * v;
            }
        }
        Ok((rows, block))
    }

    /// Enthalpy `E + P V` of `sys` and its gradient in the variables, from the energy, the
    /// Cartesian gradient of the atoms, and the stress.
    pub fn objective(
        &self,
        sys: &MolecularSystem,
        energy: f64,
        gradient: &Col<f64>,
        stress: &Stress,
    ) -> MoptResult<(f64, Col<f64>)> {
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("the structure has no periodic cell"))?;
        let n = sys.dof();
        if gradient.nrows() != n {
            return Err(MoptError::config("the gradient does not fit the structure"));
        }
        let volume = cell.volume_bohr3();
        // x = q h_0^-1 h for row vectors: dE/dq_a = sum_b (h_0^-1 h)_ab g_b.
        let map = self.atoms_map(cell)?;
        let mut out = Col::zeros(self.dimension());
        for atom in sys.atoms() {
            for a in 0..3 {
                out[atom.dof(a)] = (0..3)
                    .map(|b| map[a][b] * gradient[atom.dof(b)])
                    .sum::<f64>();
            }
        }
        // For slabs and wires the stress refers to the volume of the stored cell, and
        // `lattice_gradient` multiplies by the same volume: only their product enters.
        let of_energy = lattice_gradient(stress, cell);
        for (slot, (k, e)) in self.directions.iter().enumerate() {
            // dV/da_k is V times the reciprocal vector k.
            let along: f64 = (0..3)
                .map(|c| {
                    (of_energy[*k][c] + self.pressure * volume * cell.h_inv_bohr[*k][c]) * e[c]
                })
                .sum();
            out[n + slot] = along * self.lengths[*k] / self.gamma;
        }
        Ok((energy + self.pressure * volume, out))
    }
}

/// Which motions of the whole structure leave the surface of a variable cell unchanged, so
/// that they are taken out of its Hessian ([`VariableCell::rigid_modes_with`]).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellSymmetries {
    /// The translations of the atoms (not with frozen atoms, nor on a surface fixed in space).
    pub translations: bool,
    /// The rotations of the lattice with the atoms (not on a surface fixed in space).
    pub rotations: bool,
}

impl CellSymmetries {
    /// Every motion of [`VariableCell::rigid_modes`].
    pub const ALL: Self = Self {
        translations: true,
        rotations: true,
    };
}

/// A distance between the points that span the cell: the origin and the tips of the
/// lattice vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellDistance {
    /// From the origin to the tip of a lattice vector (its length).
    Tip(usize),
    /// Between the tips of two lattice vectors.
    Tips(usize, usize),
}

impl CellDistance {
    /// From `a`, `b`, `c` (origin to tip) or `ab`, `bc`, `ca` (tip to tip).
    pub fn parse(text: &str) -> MoptResult<Self> {
        let index = |c: char| match c {
            'a' => Ok(0),
            'b' => Ok(1),
            'c' => Ok(2),
            other => Err(MoptError::config(format!(
                "a lattice vector is a, b, or c, not '{other}'"
            ))),
        };
        let letters: Vec<char> = text.trim().to_ascii_lowercase().chars().collect();
        match letters.as_slice() {
            [k] => Ok(Self::Tip(index(*k)?)),
            [k, l] if k != l => Ok(Self::Tips(index(*k)?, index(*l)?)),
            _ => Err(MoptError::config(format!(
                "a distance of the cell is a, b, c (origin to tip) or ab, bc, ca (tip to tip), \
                 not '{text}'"
            ))),
        }
    }

    fn vectors(self) -> Vec<usize> {
        match self {
            Self::Tip(k) => vec![k],
            Self::Tips(k, l) => vec![k, l],
        }
    }
}

/// Artificial force on the cell (the variable-cell part of the AFIR method for periodic
/// systems, Takagi et al., Phys. Rev. B 2017, 95, 184110): the AFIR function over distances
/// between dummy atoms at the origin and at the tips of the lattice vectors,
///
/// `E = alpha sum_p w_p r_p / sum_p w_p`, `w_p = r_p^-6`,
///
/// with `alpha` from the collision energy `gamma` as for atoms
/// ([`crate::pes::bias::afir_alpha`]). `gamma > 0` shortens the distances (compresses the
/// cell along them), `gamma < 0` lengthens them. The dummy atoms all have the same radius,
/// which cancels in the weights.
#[derive(Debug, Clone)]
pub struct CellForce {
    /// Collision energy in kJ/mol, possibly ramped over the iterations.
    pub gamma: Ramp,
    /// The distances the force acts on.
    pub distances: Vec<CellDistance>,
}

impl CellForce {
    /// Energy (Hartree) and its gradient with respect to the lattice vectors (rows, Bohr)
    /// at `iteration`.
    pub fn evaluate(&self, vectors: &[[f64; 3]; 3], iteration: usize) -> (f64, [[f64; 3]; 3]) {
        let alpha = afir_alpha(self.gamma.at(iteration));
        let mut gradient = [[0.0; 3]; 3];
        if alpha == 0.0 || self.distances.is_empty() {
            return (0.0, gradient);
        }
        // Each distance with its vector d (the gradient with respect to the first lattice
        // vector is d / r, with respect to the second -d / r).
        let spans: Vec<(CellDistance, [f64; 3], f64)> = self
            .distances
            .iter()
            .map(|p| {
                let d: [f64; 3] = match p {
                    CellDistance::Tip(k) => vectors[*k],
                    CellDistance::Tips(k, l) => {
                        std::array::from_fn(|c| vectors[*k][c] - vectors[*l][c])
                    }
                };
                let r = libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                (*p, d, r)
            })
            .collect();
        let weight = |r: f64| r.powi(-6);
        let s0: f64 = spans.iter().map(|(_, _, r)| weight(*r)).sum();
        let s1: f64 = spans.iter().map(|(_, _, r)| weight(*r) * r).sum();
        for (p, d, r) in &spans {
            // d(w r)/dr = -5 r^-6, dw/dr = -6 r^-7.
            let slope = alpha * (-5.0 * weight(*r) * s0 + 6.0 * weight(*r) / r * s1) / (s0 * s0);
            let (first, second) = match p {
                CellDistance::Tip(k) => (*k, None),
                CellDistance::Tips(k, l) => (*k, Some(*l)),
            };
            for c in 0..3 {
                gradient[first][c] += slope * d[c] / r;
                if let Some(l) = second {
                    gradient[l][c] -= slope * d[c] / r;
                }
            }
        }
        (alpha * s1 / s0, gradient)
    }

    /// The second derivatives of its energy along the strain of the cell at `iteration`
    /// ([`StrainHessian`]; the distances between the tips go to `(1 + eps) d` with the
    /// lattice): with `E = alpha s1 / s0`, `s0 = sum w`, `s1 = sum w r`, `w = r^-6`, the
    /// derivatives in the distances are those of a ratio, `s1'' = 30 r^-7` and
    /// `s0'' = 42 r^-8` on the diagonal.
    pub fn strain_hessian(&self, vectors: &[[f64; 3]; 3], iteration: usize) -> [[f64; 9]; 9] {
        let alpha = afir_alpha(self.gamma.at(iteration));
        let mut out = StrainHessian::zeros(0);
        if alpha == 0.0 || self.distances.is_empty() {
            return out.strain;
        }
        let spans: Vec<(Option<(usize, usize)>, f64, [f64; 3])> = self
            .distances
            .iter()
            .map(|p| {
                let d: [f64; 3] = match p {
                    CellDistance::Tip(k) => vectors[*k],
                    CellDistance::Tips(k, l) => {
                        std::array::from_fn(|c| vectors[*k][c] - vectors[*l][c])
                    }
                };
                let r = libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                (None, r, d.map(|c| c / r))
            })
            .collect();
        let r: Vec<f64> = spans.iter().map(|(_, r, _)| *r).collect();
        let s0: f64 = r.iter().map(|x| x.powi(-6)).sum();
        let s1: f64 = r.iter().map(|x| x.powi(-5)).sum();
        let d0: Vec<f64> = r.iter().map(|x| -6.0 * x.powi(-7)).collect();
        let d1: Vec<f64> = r.iter().map(|x| -5.0 * x.powi(-6)).collect();
        let first: Vec<f64> = (0..r.len())
            .map(|p| alpha * (d1[p] * s0 - s1 * d0[p]) / (s0 * s0))
            .collect();
        let second = |p: usize, q: usize| -> f64 {
            let (dd1, dd0) = if p == q {
                (30.0 * r[p].powi(-7), 42.0 * r[p].powi(-8))
            } else {
                (0.0, 0.0)
            };
            alpha
                * ((dd1 * s0 + d1[p] * d0[q] - d0[p] * d1[q] - s1 * dd0) / (s0 * s0)
                    - 2.0 * (d1[p] * s0 - s1 * d0[p]) * d0[q] / (s0 * s0 * s0))
        };
        out.add_radial_function(&spans, &first, &second);
        out.strain
    }
}

/// Displacement of the differences of [`numerical_hessian`] (Bohr; the variables are
/// lengths).
pub const DEFAULT_HESSIAN_STEP: f64 = 1e-3;

/// The Hessian at `x` from central differences of `gradient_at`, made symmetric (the serial
/// form of the one central-difference Hessian of the crate,
/// [`crate::pes::fd_hessian::serial_central_difference_hessian`]: the gradient of a cell may
/// start from what the calculator evaluated last).
pub fn numerical_hessian(
    x: &Col<f64>,
    step: f64,
    gradient_at: impl FnMut(&Col<f64>) -> MoptResult<Col<f64>>,
) -> MoptResult<Mat<f64>> {
    crate::pes::fd_hessian::serial_central_difference_hessian(x, step, gradient_at)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{Angstrom, Element, EvalRequest, SpinCharge};
    use crate::pes::calculator::Calculator;
    use crate::pes::native::Emt;
    use crate::pes::stress::stress_of;

    fn copper() -> MolecularSystem {
        let edge = 3.5;
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new([edge, 0.0, 0.0]),
            Vector3::<Angstrom>::new([0.05 * edge, 1.04 * edge, 0.0]),
            Vector3::<Angstrom>::new([-0.03 * edge, 0.06 * edge, 0.97 * edge]),
            [true, true, true],
        )
        .unwrap();
        let h = edge / 2.0;
        MolecularSystem::new(
            vec![Element::parse_token("Cu").unwrap(); 4],
            &positions::<Angstrom>(&[
                [0.05, 0.02, -0.03],
                [h + 0.04, h - 0.06, 0.02],
                [h - 0.03, 0.05, h + 0.07],
                [-0.02, h + 0.03, h - 0.05],
            ]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap()
    }

    #[test]
    fn the_variables_give_the_structure_back_and_the_gradient_is_that_of_the_enthalpy() {
        let sys = copper();
        let pressure = 3e-4;
        let map = VariableCell::new(&sys, 2.0, pressure).unwrap();
        let x = map.variables(&sys).unwrap();
        assert_eq!(x.nrows(), 21);
        // The atoms' variables are their positions at the start; the lattice variables have
        // the length gamma = 2 sqrt(4).
        for k in 0..12 {
            assert!((x[k] - sys.coords_flat_bohr()[k]).abs() < 1e-12);
        }
        let first = (x[12] * x[12] + x[13] * x[13] + x[14] * x[14]).sqrt();
        assert!((first - 4.0).abs() < 1e-12);
        let back = map.system(&x).unwrap();
        for k in 0..12 {
            assert!((back.coords_flat_bohr()[k] - sys.coords_flat_bohr()[k]).abs() < 1e-12);
        }
        // Differences of the enthalpy in every variable against the gradient.
        let calc = Emt::new();
        let enthalpy = |x: &Col<f64>| -> f64 {
            let s = map.system(x).unwrap();
            let volume = s.active_cell().unwrap().volume_bohr3();
            calc.evaluate(&s, &EvalRequest::ENERGY_ONLY)
                .unwrap()
                .energy
                .value()
                + pressure * volume
        };
        let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        let stress = stress_of(&calc, &sys).unwrap();
        let (h, gradient) = map
            .objective(&sys, out.energy.value(), &out.gradient_bohr, &stress)
            .unwrap();
        assert!((h - enthalpy(&x)).abs() < 1e-12);
        for k in 0..21 {
            let step = 1e-4;
            let moved =
                |sign: f64| Col::from_fn(21, |i| x[i] + if i == k { sign * step } else { 0.0 });
            let slope = (enthalpy(&moved(1.0)) - enthalpy(&moved(-1.0))) / (2.0 * step);
            assert!(
                (slope - gradient[k]).abs() < 1e-8,
                "variable {k}: {slope} against {}",
                gradient[k]
            );
        }
        // 1 Hartree/Bohr^3 is 29421 GPa.
        assert!((gigapascal_per_atomic_unit() - 29421.0).abs() < 1.0);
        assert!(VariableCell::new(&sys, 0.0, 0.0).is_err());
        assert_eq!(map.names()[0], "q1x");
        assert_eq!(map.names()[20], "Acz");
    }

    #[test]
    fn slabs_and_wires_vary_the_lattice_vectors_of_their_periodic_directions() {
        let calc = Emt::new();
        for (axes, variables) in [([true, true, false], 16), ([false, false, true], 13)] {
            let mut sys = copper();
            if let Some(cell) = sys.cell.as_mut() {
                cell.periodic_axes = axes;
            }
            let map = VariableCell::new(&sys, 2.0, 0.0).unwrap();
            assert_eq!(map.dimension(), variables);
            // No pressure without a volume.
            assert!(VariableCell::new(&sys, 2.0, 1e-4).is_err());
            let x = map.variables(&sys).unwrap();
            let energy = |x: &Col<f64>| -> f64 {
                calc.evaluate(&map.system(x).unwrap(), &EvalRequest::ENERGY_ONLY)
                    .unwrap()
                    .energy
                    .value()
            };
            let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
            let stress = stress_of(&calc, &sys).unwrap();
            let (_, gradient) = map
                .objective(&sys, out.energy.value(), &out.gradient_bohr, &stress)
                .unwrap();
            for k in 0..variables {
                let step = 1e-4;
                let moved = |sign: f64| {
                    Col::from_fn(variables, |i| x[i] + if i == k { sign * step } else { 0.0 })
                };
                let slope = (energy(&moved(1.0)) - energy(&moved(-1.0))) / (2.0 * step);
                assert!(
                    (slope - gradient[k]).abs() < 1e-8,
                    "{axes:?}, variable {k}: {slope} against {}",
                    gradient[k]
                );
            }
            // The other lattice vectors stay, and those that vary stay in their plane or
            // on their line.
            let moved = Col::from_fn(variables, |i| x[i] * (1.01 + 0.01 * (i % 3) as f64));
            let other = map.system(&moved).unwrap();
            let (before, after) = (sys.active_cell().unwrap(), other.active_cell().unwrap());
            let vectors = |cell: &PeriodicCell| -> Vec<[f64; 3]> {
                (0..3)
                    .filter(|k| axes[*k])
                    .map(|k| cell.lattice_vector(k).value())
                    .collect()
            };
            for (k, periodic) in axes.iter().enumerate() {
                let same = before.lattice_vector(k).value() == after.lattice_vector(k).value();
                assert_eq!(same, !periodic);
            }
            let (old, new) = (vectors(before), vectors(after));
            let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            let cross = |u: [f64; 3], v: [f64; 3]| {
                [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ]
            };
            if old.len() == 2 {
                let normal = cross(old[0], old[1]);
                assert!(dot(normal, new[0]).abs() < 1e-9 && dot(normal, new[1]).abs() < 1e-9);
            } else {
                let off = cross(old[0], new[0]);
                assert!(dot(off, off).sqrt() < 1e-9);
            }
        }
    }

    #[test]
    fn the_energy_does_not_change_along_the_rigid_modes() {
        let calc = Emt::new();
        for (axes, count) in [
            ([true, true, true], 6),
            ([true, true, false], 3),
            ([false, false, true], 4),
        ] {
            let mut sys = copper();
            if let Some(cell) = sys.cell.as_mut() {
                cell.periodic_axes = axes;
            }
            let map = VariableCell::new(&sys, 2.0, 0.0).unwrap();
            let x = map.variables(&sys).unwrap();
            let modes = map.rigid_modes(&x).unwrap();
            assert_eq!(modes.len(), count, "{axes:?}");
            let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
            let stress = stress_of(&calc, &sys).unwrap();
            let (_, gradient) = map
                .objective(&sys, out.energy.value(), &out.gradient_bohr, &stress)
                .unwrap();
            for (m, mode) in modes.iter().enumerate() {
                // Of unit length, orthogonal to the others, and to the gradient.
                for (k, other) in modes.iter().enumerate() {
                    let overlap: f64 = (0..x.nrows()).map(|i| mode[i] * other[i]).sum();
                    let expected = if k == m { 1.0 } else { 0.0 };
                    assert!((overlap - expected).abs() < 1e-12);
                }
                let slope: f64 = (0..x.nrows()).map(|i| mode[i] * gradient[i]).sum();
                assert!(slope.abs() < 1e-10, "{axes:?}, mode {m}: {slope}");
            }
        }
    }

    #[test]
    fn the_force_on_the_cell_has_the_gradient_of_its_energy() {
        let vectors = [[7.0, 0.2, -0.1], [0.5, 6.4, 0.3], [-0.4, 0.6, 8.1]];
        let force = CellForce {
            gamma: Ramp::constant(150.0),
            distances: ["a", "b", "c", "ab", "ca"]
                .iter()
                .map(|t| CellDistance::parse(t).unwrap())
                .collect(),
        };
        let (energy, gradient) = force.evaluate(&vectors, 0);
        assert!(energy > 0.0);
        for (k, row) in gradient.iter().enumerate() {
            for (c, expected) in row.iter().enumerate() {
                let at = |shift: f64| {
                    let mut moved = vectors;
                    moved[k][c] += shift;
                    force.evaluate(&moved, 0).0
                };
                let slope = (at(1e-5) - at(-1e-5)) / 2e-5;
                assert!(
                    (slope - expected).abs() < 1e-9,
                    "({k}, {c}): {slope} against {expected}"
                );
            }
        }
        // One distance: a constant pull of alpha along it; a negative gamma pushes.
        let single = CellForce {
            gamma: Ramp::constant(-150.0),
            distances: vec![CellDistance::Tip(0)],
        };
        let (energy, gradient) = single.evaluate(&vectors, 0);
        let length = (49.0_f64 + 0.04 + 0.01).sqrt();
        assert!((energy - afir_alpha(-150.0) * length).abs() < 1e-12 && energy < 0.0);
        assert!((gradient[0][0] - afir_alpha(-150.0) * 7.0 / length).abs() < 1e-12);
        assert!(CellDistance::parse("aa").is_err() && CellDistance::parse("d").is_err());
        assert_eq!(CellDistance::parse("BC").unwrap(), CellDistance::Tips(1, 2));
        // A slab has no force on its third vector.
        let mut slab = copper();
        if let Some(cell) = slab.cell.as_mut() {
            cell.periodic_axes = [true, true, false];
        }
        let map = VariableCell::new(&slab, 2.0, 0.0).unwrap();
        assert!(!map.takes(&force) && map.takes(&single));
        // Its second derivatives along the strain, against differences of its energy.
        let sys = copper();
        let lattice = |s: &MolecularSystem| -> [[f64; 3]; 3] {
            let cell = s.active_cell().unwrap();
            std::array::from_fn(|k| cell.lattice_vector(k).value())
        };
        let analytic = force.strain_hessian(&lattice(&sys), 0);
        let numeric = crate::pes::stress::numerical_strain_hessian(&sys, 2e-4, &|s| {
            Ok((force.evaluate(&lattice(s), 0).0, Col::zeros(s.dof())))
        })
        .unwrap();
        let scale = numeric
            .strain
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        for (a, b) in analytic
            .iter()
            .flatten()
            .zip(numeric.strain.iter().flatten())
        {
            assert!((a - b).abs() < 1e-5 * scale, "{a} {b}");
        }
    }

    #[test]
    fn a_slab_with_its_third_vector_along_the_normal_turns_as_a_whole() {
        let edge = 3.5 * Units::ANGSTROM_TO_BOHR;
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Bohr>::new([edge, 0.0, 0.0]),
            Vector3::<Bohr>::new([0.2 * edge, 1.1 * edge, 0.0]),
            Vector3::<Bohr>::new([0.0, 0.0, 6.0 * edge]),
            [true, true, false],
        )
        .unwrap();
        let sys = MolecularSystem::new(
            vec![Element::parse_token("Cu").unwrap(); 2],
            &positions::<Bohr>(&[[0.1, 0.2, 0.3], [0.5 * edge, 0.6 * edge, 0.4 * edge]]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap();
        let map = VariableCell::new(&sys, 2.0, 0.0).unwrap();
        assert_eq!(map.dimension(), 6 + 4);
        assert_eq!(map.names()[6..], ["Aa1", "Aa2", "Ab1", "Ab2"]);
        let x = map.variables(&sys).unwrap();
        // Three translations and the rotation about the normal: 3N of 3N + 4 are left.
        let modes = map.rigid_modes(&x).unwrap();
        assert_eq!(modes.len(), 4);
        let calc = Emt::new();
        let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        let stress = stress_of(&calc, &sys).unwrap();
        let (_, gradient) = map
            .objective(&sys, out.energy.value(), &out.gradient_bohr, &stress)
            .unwrap();
        for mode in &modes {
            let slope: f64 = (0..x.nrows()).map(|i| mode[i] * gradient[i]).sum();
            assert!(slope.abs() < 1e-10, "{slope}");
        }
    }

    #[test]
    fn the_lattice_gradient_of_a_symmetric_stress_is_that_of_the_cell() {
        let cell = PeriodicCell::from_lattice_vectors(
            crate::domain::Vector3::<crate::domain::Bohr>::new([9.1, 0.2, -0.3]),
            crate::domain::Vector3::new([1.4, 8.7, 0.5]),
            crate::domain::Vector3::new([-0.6, 0.9, 10.2]),
            [true; 3],
        )
        .unwrap();
        let symmetric: Stress = [
            [1.3e-4, -2.1e-5, 7.7e-6],
            [-2.1e-5, -4.4e-5, 3.3e-6],
            [7.7e-6, 3.3e-6, 9.9e-5],
        ];
        let a = lattice_gradient(&symmetric, &cell);
        let b = crate::pes::stress::cell_gradient(&symmetric, &cell);
        for k in 0..3 {
            for c in 0..3 {
                assert_eq!(a[k][c].to_bits(), b[k][c].to_bits());
            }
        }
        // With an antisymmetric part: V sum_b (b_k)_b sigma_ab.
        let mut skew = symmetric;
        skew[0][1] += 3e-5;
        let g = lattice_gradient(&skew, &cell);
        let v = cell.volume_bohr3();
        for (k, (row, reciprocal)) in g.iter().zip(&cell.h_inv_bohr).enumerate() {
            for (c, value) in row.iter().enumerate() {
                let want: f64 = (0..3).map(|b| reciprocal[b] * skew[c][b]).sum();
                assert!((value - v * want).abs() < 1e-15, "{k} {c}");
            }
        }
    }

    #[test]
    fn the_hessian_from_differences_of_the_gradient() {
        // f = x0^2 + 3 x0 x1 + 2 x1^3: Hessian [[2, 3], [3, 12 x1]].
        let x = Col::from_fn(2, |i| [0.4, -0.7][i]);
        let h = numerical_hessian(&x, 1e-4, |x| {
            Ok(Col::from_fn(2, |i| {
                [2.0 * x[0] + 3.0 * x[1], 3.0 * x[0] + 6.0 * x[1] * x[1]][i]
            }))
        })
        .unwrap();
        assert!((h[(0, 0)] - 2.0).abs() < 1e-8 && (h[(0, 1)] - 3.0).abs() < 1e-8);
        assert!((h[(1, 0)] - 3.0).abs() < 1e-8 && (h[(1, 1)] + 8.4).abs() < 1e-7);
    }

    /// The loop this module had before it used the shared central differences.
    fn former_numerical_hessian(
        x: &Col<f64>,
        step: f64,
        mut gradient_at: impl FnMut(&Col<f64>) -> MoptResult<Col<f64>>,
    ) -> MoptResult<Mat<f64>> {
        let n = x.nrows();
        let mut hessian = Mat::zeros(n, n);
        for k in 0..n {
            let moved =
                |sign: f64| Col::from_fn(n, |i| x[i] + if i == k { sign * step } else { 0.0 });
            let (plus, minus) = (gradient_at(&moved(1.0))?, gradient_at(&moved(-1.0))?);
            for i in 0..n {
                hessian[(i, k)] = (plus[i] - minus[i]) / (2.0 * step);
            }
        }
        Ok(Mat::from_fn(n, n, |i, j| {
            0.5 * (hessian[(i, j)] + hessian[(j, i)])
        }))
    }

    #[test]
    fn the_shared_central_differences_give_the_bits_of_the_former_loop() {
        // A gradient that keeps state (it counts its calls), as a calculator may.
        let gradient = |calls: &mut usize, x: &Col<f64>| -> MoptResult<Col<f64>> {
            *calls += 1;
            Ok(Col::from_fn(x.nrows(), |i| {
                (1.3 * x[i]).sin() * x[(i + 1) % x.nrows()].exp() + 0.2 * x[i] * x[i] * x[i]
            }))
        };
        for (n, step) in [(3, 1e-3), (9, 1e-4), (12, 3e-5)] {
            let x = Col::from_fn(n, |i| 0.37 * i as f64 - 1.1);
            let (mut a, mut b) = (0, 0);
            let new = numerical_hessian(&x, step, |y| gradient(&mut a, y)).unwrap();
            let old = former_numerical_hessian(&x, step, |y| gradient(&mut b, y)).unwrap();
            assert_eq!(a, b);
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(new[(i, j)].to_bits(), old[(i, j)].to_bits(), "{n} {i} {j}");
                }
            }
        }
    }
}
