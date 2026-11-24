//! Evaluation request and result types exchanged with a `Calculator`.
//!
//! MultiOptPy: the energy / gradient / Hessian values returned by
//!  `Calculator/*_calculation_tools.py:Calculation.single_point`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::quantity::{Hartree, Quantity};
use faer::{Col, Mat};

/// Derivatives requested from a calculator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalRequest {
    /// Request the gradient (`3N`, `Hartree / Bohr`).
    pub need_gradient: bool,
    /// Request the Hessian (`3N x 3N`, `Hartree / Bohr^2`).
    pub need_hessian: bool,
    /// Electronic state root (`0` = ground state).
    pub excited_state: usize,
    /// Request the stress of a periodic cell. Calculators that have it for free return it
    /// in any case; those for which it costs extra (programs behind ASE) compute it only
    /// when this is set.
    pub need_stress: bool,
    /// The Hessian is for a check of stability (the count of imaginary modes of the mode
    /// kill, of a minimum, of a transition state or a saddle point of higher order), not for
    /// the steps of an optimizer: a calculator whose Hessian is an approximation for the
    /// steps (the effective Hessian of ONIOM microiterations) computes it to convergence
    /// instead, and returns none where it cannot, so that the check is not judged on it.
    pub for_stability: bool,
}

impl EvalRequest {
    /// Energy only.
    pub const ENERGY_ONLY: Self = Self {
        need_gradient: false,
        need_hessian: false,
        excited_state: 0,
        need_stress: false,
        for_stability: false,
    };

    /// Energy and gradient.
    pub const ENERGY_GRAD: Self = Self {
        need_gradient: true,
        need_hessian: false,
        excited_state: 0,
        need_stress: false,
        for_stability: false,
    };

    /// Energy, gradient, and the stress of the cell.
    pub const ENERGY_GRAD_STRESS: Self = Self {
        need_gradient: true,
        need_hessian: false,
        excited_state: 0,
        need_stress: true,
        for_stability: false,
    };

    /// Energy, gradient, and Hessian.
    pub const ENERGY_GRAD_HESS: Self = Self {
        need_gradient: true,
        need_hessian: true,
        excited_state: 0,
        need_stress: false,
        for_stability: false,
    };

    /// Energy, gradient, and the Hessian for a check of stability ([`Self::for_stability`]).
    pub const ENERGY_GRAD_HESS_CHECK: Self = Self {
        need_gradient: true,
        need_hessian: true,
        excited_state: 0,
        need_stress: false,
        for_stability: true,
    };
}

impl Default for EvalRequest {
    fn default() -> Self {
        Self::ENERGY_GRAD
    }
}

/// Optional electronic-structure metadata (frontier orbital energies, atomic charges).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrbitalData {
    /// HOMO energy in Hartree.
    pub homo_hartree: Option<f64>,
    /// LUMO energy in Hartree.
    pub lumo_hartree: Option<f64>,
    /// Atomic partial charges (Mulliken, CM5, ...).
    pub atomic_charges: Option<Vec<f64>>,
    /// Expectation value `<S^2>` of the wavefunction, where the calculator tells it
    /// (unrestricted calculations).
    pub spin_square: Option<f64>,
    /// What the program printed during the evaluation (its own output), for the run log.
    pub program_output: Option<String>,
}

/// Result of a single calculator evaluation.
#[derive(Debug, Clone)]
pub struct EvalOutput {
    /// Potential energy.
    pub energy: Quantity<Hartree>,
    /// Gradient `+dE/dx` of length `3N` in `Hartree / Bohr`.
    pub gradient_bohr: Col<f64>,
    /// Cartesian Hessian `(3N, 3N)` in `Hartree / Bohr^2`, present only when requested.
    pub hessian_bohr: Option<Mat<f64>>,
    /// Orbital energies / atomic charges.
    pub orbitals: Option<OrbitalData>,
    /// Stress `sigma = (1 / V) dE / d(strain)` of a periodic cell in `Hartree / Bohr^3`
    /// (negative under compression, the sign of ASE), where the calculator gives it;
    /// `pes::stress` has a numerical one for the others. For slabs and wires `V` is the
    /// volume of the stored cell, so that only `V sigma` means something.
    pub stress: Option<[[f64; 3]; 3]>,
    /// The second derivatives along the strain of the cell ([`StrainHessian`]), where the
    /// calculator gives them: with a Hessian and a stress asked for (`need_hessian` and
    /// `need_stress`) at a structure with a cell.
    pub strain_hessian: Option<StrainHessian>,
}

/// The second derivatives of the energy of a periodic structure along the homogeneous strain
/// `eps` of its stress, which takes every position and lattice vector `r` to `(1 + eps) r`
/// (column vectors: `r'_a = r_a + sum_b eps_ab r_b`; the fractional coordinates kept), at no
/// strain. The component `eps_ab` has the index `3 a + b`.
///
/// With them and the Cartesian Hessian and gradient, the Hessian of the energy in any
/// coordinates of the atoms and of the lattice follows by the chain rule
/// ([`crate::opt::cell::VariableCell::lattice_columns`]).
#[derive(Debug, Clone, PartialEq)]
pub struct StrainHessian {
    /// `d2E / (d eps_ab d eps_cd)` at `[3 a + b][3 c + d]` (Hartree).
    pub strain: [[f64; 9]; 9],
    /// `d/d eps_ab` of `dE/dr_ip` at the strained structure, at `(3 i + p, 3 a + b)`
    /// (`3N x 9`, Hartree/Bohr): how the Cartesian gradient changes when the structure is
    /// strained.
    pub mixed: Mat<f64>,
}

impl StrainHessian {
    /// No second derivatives, for `dof` Cartesian coordinates.
    pub fn zeros(dof: usize) -> Self {
        Self {
            strain: [[0.0; 9]; 9],
            mixed: Mat::zeros(dof, 9),
        }
    }

    /// Add `other` (of the same structure).
    pub fn add(&mut self, other: &Self) {
        for (row, add) in self.strain.iter_mut().zip(&other.strain) {
            for (value, part) in row.iter_mut().zip(add) {
                *value += part;
            }
        }
        if self.mixed.nrows() == other.mixed.nrows() {
            self.mixed += &other.mixed;
        }
    }

    /// The same, times `factor` (a change of units).
    pub fn scaled(mut self, strain: f64, mixed: f64) -> Self {
        for row in &mut self.strain {
            for value in row.iter_mut() {
                *value *= strain;
            }
        }
        self.mixed *= faer::Scale(mixed);
        self
    }

    /// Add the strain derivatives of a radial coordinate `r = |d|` along `d = r u` (the
    /// vector from the atom `near` to the atom `far`, or to a periodic image; `None` for an
    /// image of the atom itself) of a function with `dE/dr = c1` and `d2E/dr2 = c2` there:
    /// `dr/d eps_ab = r u_a u_b`, `d2r/(d eps_ab d eps_cd) = r (delta_ac u_b u_d - u_a u_b u_c
    /// u_d)`, and the unit vector changes by `du_p/d eps_ab = delta_pa u_b - u_p u_a u_b`.
    pub fn add_radial(
        &mut self,
        atoms: Option<(usize, usize)>,
        r: f64,
        u: [f64; 3],
        c1: f64,
        c2: f64,
    ) {
        let t: [f64; 9] = std::array::from_fn(|k| r * u[k / 3] * u[k % 3]);
        for (x, row) in self.strain.iter_mut().enumerate() {
            let (a, b) = (x / 3, x % 3);
            for (y, value) in row.iter_mut().enumerate() {
                let (c, d) = (y / 3, y % 3);
                let same = if a == c { u[b] * u[d] } else { 0.0 };
                *value += c2 * t[x] * t[y] + c1 * r * (same - u[a] * u[b] * u[c] * u[d]);
            }
        }
        let Some((near, far)) = atoms else {
            return;
        };
        for p in 0..3 {
            for (x, tx) in t.iter().enumerate() {
                let (a, b) = (x / 3, x % 3);
                let turn = if p == a { u[b] } else { 0.0 } - u[p] * u[a] * u[b];
                let v = c2 * tx * u[p] + c1 * turn;
                self.mixed[(3 * far + p, x)] += v;
                self.mixed[(3 * near + p, x)] -= v;
            }
        }
    }
}

impl StrainHessian {
    /// The second derivatives along the strain of a structure `sys` in a cell from the Hessian
    /// `hessian` of its energy in the Cartesian positions and the lattice vectors as
    /// independent variables (`3N + 9` of them: the atoms, then the lattice vector `k`
    /// component `c` at `3N + 3k + c`; the images of an atom at `x_j + n h`). The strain moves
    /// both linearly, `dx_ia / d eps_cd = delta_ac x_id` and `dh_ka / d eps_cd = delta_ac h_kd`,
    /// so `K_(cd)(ef) = sum_PQ W_(Pc)(Qe) p_Pd p_Qf` over the points `p` (the atoms and the
    /// lattice vectors) and the mixed block is `M_(ip)(cd) = sum_Q W_(ip)(Qc) p_Qd`.
    pub fn from_cell_hessian(
        sys: &crate::domain::MolecularSystem,
        hessian: &Mat<f64>,
    ) -> crate::error::MoptResult<Self> {
        let cell = sys.active_cell().ok_or_else(|| {
            crate::error::MoptError::config("a Hessian in the lattice vectors needs a cell")
        })?;
        let n = sys.dof();
        if hessian.nrows() != n + 9 || hessian.ncols() != n + 9 {
            return Err(crate::error::MoptError::config(format!(
                "a Hessian in the positions and the lattice vectors of {} atoms has {} rows and \
                 columns, not {} x {}",
                sys.n_atoms(),
                n + 9,
                hessian.nrows(),
                hessian.ncols()
            )));
        }
        let points: Vec<[f64; 3]> = sys
            .atoms()
            .map(|a| sys.position(a).value())
            .chain((0..3).map(|k| cell.lattice_vector(k).value()))
            .collect();
        let mut out = Self::zeros(n);
        for (big_p, p) in points.iter().enumerate() {
            for (big_q, q) in points.iter().enumerate() {
                for c in 0..3 {
                    for e in 0..3 {
                        let w = hessian[(3 * big_p + c, 3 * big_q + e)];
                        if w == 0.0 {
                            continue;
                        }
                        for (d, pd) in p.iter().enumerate() {
                            for (f, qf) in q.iter().enumerate() {
                                out.strain[3 * c + d][3 * e + f] += w * pd * qf;
                            }
                        }
                    }
                }
            }
        }
        for row in 0..n {
            for (big_q, q) in points.iter().enumerate() {
                for c in 0..3 {
                    let w = hessian[(row, 3 * big_q + c)];
                    for (d, qd) in q.iter().enumerate() {
                        out.mixed[(row, 3 * c + d)] += w * qd;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Add the strain derivatives of a function of the radial coordinates of `pairs` (each
    /// with its atoms as in [`Self::add_radial`], its length, and its unit vector) whose
    /// first derivatives are `first` and second derivatives `second(k, l)`:
    /// `sum_kl E_kl (dr_k/d eps)(dr_l/d eps)^T + sum_k E_k d2r_k/d eps^2`, and for the
    /// gradient of the atoms `sum_l E_kl (dr_l/d eps) u_k + E_k du_k/d eps`.
    pub fn add_radial_function(
        &mut self,
        pairs: &[(Option<(usize, usize)>, f64, [f64; 3])],
        first: &[f64],
        second: &dyn Fn(usize, usize) -> f64,
    ) {
        let t: Vec<[f64; 9]> = pairs
            .iter()
            .map(|(_, r, u)| std::array::from_fn(|k| r * u[k / 3] * u[k % 3]))
            .collect();
        for ((atoms, r, u), e) in pairs.iter().zip(first) {
            self.add_radial(*atoms, *r, *u, *e, 0.0);
        }
        for (k, (atoms, _, u)) in pairs.iter().enumerate() {
            for (l, tl) in t.iter().enumerate() {
                let e = second(k, l);
                if e == 0.0 {
                    continue;
                }
                for (x, row) in self.strain.iter_mut().enumerate() {
                    for (y, value) in row.iter_mut().enumerate() {
                        *value += e * t[k][x] * tl[y];
                    }
                }
                if let Some((near, far)) = atoms {
                    for (p, up) in u.iter().enumerate() {
                        for (x, tx) in tl.iter().enumerate() {
                            let v = e * tx * up;
                            self.mixed[(3 * far + p, x)] += v;
                            self.mixed[(3 * near + p, x)] -= v;
                        }
                    }
                }
            }
        }
    }
}

impl EvalOutput {
    /// Build an energy + gradient result.
    pub fn energy_gradient(energy: Quantity<Hartree>, gradient_bohr: Col<f64>) -> Self {
        Self {
            energy,
            gradient_bohr,
            hessian_bohr: None,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        }
    }

    /// Return `true` if energy, gradient, and Hessian contain only finite values.
    pub fn is_finite(&self) -> bool {
        self.energy.is_finite()
            && self.gradient_bohr.iter().all(|v| v.is_finite())
            && self
                .hessian_bohr
                .as_ref()
                .is_none_or(|h| (0..h.ncols()).all(|j| h.col(j).iter().all(|v| v.is_finite())))
    }
}
