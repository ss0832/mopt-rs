//! Excited singlet states on tight-binding orbitals in the Tamm-Dancoff approximation: the
//! simplified form (sTDA) on GFN0-xTB or GFN1-xTB orbitals, and the Tamm-Dancoff approximation
//! of the linear response of the self-consistent charges of GFN1-xTB ([`Coupling`]).
//!
//! **sTDA.** GFN0-xTB has no excited states of its own. This module puts the simplified
//! Tamm-Dancoff approximation of Grimme (J. Chem. Phys. 2013, 138, 244104; sTDA) on the
//! orbitals of the GFN0-xTB Hamiltonian of a closed-shell molecule, in the form it takes
//! for a reference without exact exchange:
//!
//! ```text
//! A_(ia,jb) = d_ij d_ab (e_a - e_i) + 2 (ia|jb)'
//! (ia|jb)' = sum_AB q_ia^A g_AB q_jb^B,   g_AB = (R_AB^alpha + eta_AB^-alpha)^(-1/alpha)
//! ```
//!
//! with the Loewdin transition charges `q_ia^A = sum_(m on A) C'_mi C'_ma` (`C' = S^(1/2)
//! C`), `eta_AB` the mean of the chemical hardnesses of the two atoms, and the excitation
//! energies as the eigenvalues of `A` over all single excitations. The energy of an excited
//! state is the GFN0-xTB energy plus its excitation energy.
//!
//! **What this is and is not.** The form and the exponent `alpha = 1.42` are those of sTDA
//! for a fraction `a_x = 0` of exact exchange: the tight-binding Hamiltonian has none, so
//! the term `-(ij|ab)'` of sTDA, which carries the factor `a_x`, is absent (with the value
//! `a_x = 0.5` of sTDA-xTB, which belongs to another Hamiltonian, the lowest excitation
//! energy of formaldehyde came out negative). The hardness of an atom is its
//! self-interaction in the charge model of GFN0-xTB (the diagonal element of that model,
//! `gam + J sqrt(2 / pi) / alpha_g`). Nothing is fitted to excitation energies, and GFN0-xTB
//! orbital energies are not those of a density functional, so the excitation energies are
//! qualitative. The purpose is a built-in electronic structure with several states of one
//! spin whose surfaces cross as real ones do (conical intersections between excited
//! states), for the methods that work on such surfaces. As in every Tamm-Dancoff method the
//! ground state does not take part in the response, so a crossing of the first excited
//! state with the ground state is not a conical intersection here.
//!
//! **TDA of a self-consistent-charge method** ([`Coupling::Response`]). For GFN1-xTB the
//! coupling is the method's own: the second derivatives of its energy in the charges of its
//! shells, `K = gamma + 2 Gamma_A q_A` within each atom (the shell Coulomb kernel and the
//! third-order term), with the Mulliken transition charges of the shells,
//! `q_ia^s = 1/2 sum_(m on s) (C_mi (S C)_ma + C_ma (S C)_mi)`, in the same
//! `A = d (e_a - e_i) + 2 q K q^T` of singlets (Niehaus et al., Phys. Rev. B 2001, 63, 085108,
//! for the DFTB form of the response; here without the B matrix). Nothing in it is empirical
//! beyond the ground state. GFN1-xTB has no spin-dependent term, so its triplets would be the
//! orbital energy differences: only singlets are given. Both forms are for tests of the
//! methods that work on several states, not for photochemistry.
//!
//! **Gradients.** The gradients are analytic by default: that of the ground state, and that
//! of an excitation energy from one Z-vector for the response of the orbitals
//! (`pes::native::tda_gradient`). With `excited_gradient=numerical` that of an excitation
//! energy is taken by central differences in which the states are followed, not counted:
//! the amplitudes at a displaced structure are brought into the orbitals of the reference
//! by the overlaps of the Loewdin orbitals, `X' -> O_occ X' O_virt^T`, and the excitation
//! energy of reference state `k` there is the mean of the displaced ones weighted by the
//! squared overlaps with it. Away from a degeneracy this is the derivative of the
//! eigenvalue; where two states come close it is the derivative of the diagonal element of
//! their coupled problem in the basis of the reference states, which stays finite and is
//! what a search for a crossing needs, while differences of the ordered eigenvalues jump.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{symmetric_eigen, symmetrize};

/// Exponent of the damped Coulomb interaction of the transition charges (sTDA, `a_x = 0`).
pub(crate) const ALPHA: f64 = 1.42;
/// Largest number of single excitations for which the full matrix is diagonalized.
const MAX_EXCITATIONS: usize = 4000;

/// How the single excitations are coupled.
#[derive(Debug, Clone)]
pub(crate) enum Coupling {
    /// sTDA (`a_x = 0`): Loewdin transition charges on the atoms and the damped Coulomb
    /// interaction of the chemical hardnesses of the atoms.
    Simplified {
        /// Atom of every atomic orbital.
        ao_atom: Vec<usize>,
        /// Chemical hardness of every atom (Hartree).
        hardness: Vec<f64>,
        /// Positions (Bohr).
        x: Vec<[f64; 3]>,
    },
    /// The Tamm-Dancoff approximation of the response of a self-consistent-charge method.
    Response {
        /// Mulliken transition charges on the sites of the charges, at `(i * virtual + a,
        /// site)` ([`mulliken_transition_charges`]).
        charges: Mat<f64>,
        /// The second derivatives of the energy in the charges of the sites (Hartree).
        kernel: Mat<f64>,
    },
}

/// Orbitals of a tight-binding Hamiltonian as the response needs them.
#[derive(Debug, Clone)]
pub(crate) struct Reference {
    /// The method and its form of the response, for the messages (`GFN0-xTB sTDA`).
    pub method: &'static str,
    /// Orbital energies (Hartree), ascending.
    pub eps: Vec<f64>,
    /// Orbitals in the Loewdin-orthogonalized atomic orbitals (columns).
    pub lowdin: Mat<f64>,
    /// Number of doubly occupied orbitals.
    pub occupied: usize,
    /// How the excitations are coupled.
    pub coupling: Coupling,
}

/// Mulliken transition charges `q_ia^s = 1/2 sum_(m on s) (C_mi (S C)_ma + C_ma (S C)_mi)` of
/// the orbitals `c` (atomic orbitals by orbitals, the first `occupied` occupied) with the
/// overlap `s`, on the sites `site_of_ao` (one of `sites` for every atomic orbital), at
/// `(i * virtual + a, site)`.
pub(crate) fn mulliken_transition_charges(
    c: &Mat<f64>,
    s: &Mat<f64>,
    occupied: usize,
    site_of_ao: &[usize],
    sites: usize,
) -> Mat<f64> {
    let virtuals = c.ncols() - occupied;
    let sc = s * c;
    let mut q = Mat::<f64>::zeros(occupied * virtuals, sites);
    for (m, &site) in site_of_ao.iter().enumerate() {
        for i in 0..occupied {
            let (ci, sci) = (c[(m, i)], sc[(m, i)]);
            for a in 0..virtuals {
                let v = occupied + a;
                q[(i * virtuals + a, site)] += 0.5 * (ci * sc[(m, v)] + c[(m, v)] * sci);
            }
        }
    }
    q
}

/// Excited states of a structure.
#[derive(Debug, Clone)]
pub(crate) struct States {
    /// Excitation energies (Hartree), ascending.
    pub energies: Vec<f64>,
    /// Amplitudes: column `k` is state `k` over the excitations `(i, a)` at `i * virtual +
    /// a`.
    pub amplitudes: Mat<f64>,
}

/// The gradient of the energy of state `index` of `states` (those of `reference`, at `x0`):
/// `ground` (the gradient of the ground state) plus the central differences, with the
/// displacement `h`, of the excitation energy of the state followed into the displaced
/// structures (`reference_at` gives the reference of the displaced coordinates). No analytic
/// form is used here because this is the alternative to the analytic gradient of the response
/// that the engines offer. The one loop of GFN0-xTB and GFN1-xTB.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub(crate) fn followed_gradient(
    reference: &Reference,
    states: &States,
    index: usize,
    x0: &faer::Col<f64>,
    h: f64,
    ground: faer::Col<f64>,
    reference_at: impl Fn(&faer::Col<f64>) -> MoptResult<Reference>,
) -> MoptResult<faer::Col<f64>> {
    let mut gradient = ground;
    for k in 0..x0.nrows() {
        let mut slope = 0.0;
        for sign in [1.0, -1.0] {
            let mut moved = x0.clone();
            moved[k] += sign * h;
            let other = reference_at(&moved)?;
            let theirs = other.states()?;
            slope += sign * reference.followed(states, index, &other, &theirs)?;
        }
        gradient[k] += slope / (2.0 * h);
    }
    Ok(gradient)
}

impl Reference {
    fn virtuals(&self) -> usize {
        self.eps.len() - self.occupied
    }

    /// The Loewdin transition charges `q_ia^A` on the atoms at `(i * virtual + a, A)`.
    fn loewdin_charges(&self, ao_atom: &[usize], atoms: usize) -> Mat<f64> {
        let (occupied, virtuals) = (self.occupied, self.virtuals());
        let mut q = Mat::<f64>::zeros(occupied * virtuals, atoms);
        for (m, atom) in ao_atom.iter().enumerate() {
            for i in 0..occupied {
                let left = self.lowdin[(m, i)];
                for a in 0..virtuals {
                    q[(i * virtuals + a, *atom)] += left * self.lowdin[(m, occupied + a)];
                }
            }
        }
        q
    }

    /// All singlet excited states.
    pub fn states(&self) -> MoptResult<States> {
        let (occupied, virtuals) = (self.occupied, self.virtuals());
        let size = occupied * virtuals;
        let method = self.method;
        if size == 0 {
            return Err(MoptError::calc(format!(
                "{method} excited states: no occupied or no empty orbital"
            )));
        }
        if size > MAX_EXCITATIONS {
            return Err(MoptError::unsupported(format!(
                "{method} excited states: {size} single excitations (at most {MAX_EXCITATIONS}; \
                 the matrix is diagonalized in full)"
            )));
        }
        let (q, kernel) = match &self.coupling {
            Coupling::Simplified {
                ao_atom,
                hardness,
                x,
            } => {
                if hardness.iter().any(|eta| !(*eta > 0.0 && eta.is_finite())) {
                    return Err(MoptError::calc(format!(
                        "{method} excited states: an atom without a positive chemical hardness"
                    )));
                }
                let atoms = x.len();
                let coulomb = Mat::from_fn(atoms, atoms, |a, b| {
                    let d: [f64; 3] = std::array::from_fn(|c| x[a][c] - x[b][c]);
                    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let eta = 0.5 * (hardness[a] + hardness[b]);
                    (r.powf(ALPHA) + eta.powf(-ALPHA)).powf(-1.0 / ALPHA)
                });
                (self.loewdin_charges(ao_atom, atoms), coulomb)
            }
            Coupling::Response { charges, kernel } => {
                if charges.nrows() != size
                    || charges.ncols() != kernel.nrows()
                    || kernel.nrows() != kernel.ncols()
                {
                    return Err(MoptError::config(format!(
                        "{method} excited states: {}x{} transition charges and a {}x{} kernel \
                         for {size} excitations",
                        charges.nrows(),
                        charges.ncols(),
                        kernel.nrows(),
                        kernel.ncols()
                    )));
                }
                (charges.clone(), kernel.clone())
            }
        };
        let mut a = (&q * kernel) * q.transpose() * 2.0;
        for i in 0..occupied {
            for p in 0..virtuals {
                let at = i * virtuals + p;
                a[(at, at)] += self.eps[occupied + p] - self.eps[i];
            }
        }
        symmetrize(&mut a);
        let (values, vectors) = symmetric_eigen(&a)?;
        Ok(States {
            energies: values.iter().copied().collect(),
            amplitudes: vectors,
        })
    }

    /// The amplitudes `states` of the structure `other` in the orbitals of this one:
    /// `O_occ X' O_virt^T` with the overlaps of the Loewdin orbitals.
    fn carried(&self, other: &Self, states: &States) -> Mat<f64> {
        let overlap = self.lowdin.transpose() * &other.lowdin;
        crate::pes::states::carry_amplitudes(
            &overlap,
            self.occupied,
            self.virtuals(),
            &states.amplitudes,
        )
    }

    /// The excitation energy of state `root` (`0`: the lowest) of this structure, with the
    /// states `mine`, as found among the states `theirs` of the structure `other`: the mean
    /// of their excitation energies weighted by the squared overlaps with it.
    pub fn followed(
        &self,
        mine: &States,
        root: usize,
        other: &Self,
        theirs: &States,
    ) -> MoptResult<f64> {
        // The occupied orbitals must still be the occupied ones: where the gap between
        // occupied and empty orbitals closes, the closed-shell reference changes its
        // character and its excitations are no longer those of the reference structure.
        let kept: f64 = (0..self.occupied)
            .map(|i| {
                (0..self.occupied)
                    .map(|j| {
                        let overlap: f64 = (0..self.lowdin.nrows())
                            .map(|m| self.lowdin[(m, i)] * other.lowdin[(m, j)])
                            .sum();
                        overlap * overlap
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / self.occupied as f64;
        if kept.is_nan() || kept < 0.9 {
            return Err(MoptError::calc(format!(
                "{} excited states: the occupied orbitals change between neighbouring \
                 structures (only {kept:.3} of them is kept): the gap between occupied and \
                 empty orbitals has closed, and the closed-shell reference of the excited \
                 states is not valid here",
                self.method
            )));
        }
        let carried = self.carried(other, theirs);
        let size = carried.nrows();
        let (mut weight, mut energy) = (0.0, 0.0);
        for k in 0..carried.ncols() {
            let overlap: f64 = (0..size)
                .map(|p| mine.amplitudes[(p, root)] * carried[(p, k)])
                .sum();
            weight += overlap * overlap;
            energy += overlap * overlap * theirs.energies[k];
        }
        if weight.is_nan() || weight < 0.5 {
            return Err(MoptError::calc(format!(
                "{} excited states: state {} was not found again at a displaced structure \
                 (overlap {weight:.3})",
                self.method,
                root + 1
            )));
        }
        Ok(energy / weight)
    }
}
