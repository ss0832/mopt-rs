//! Charge hopping between two molecules: the transfer integral of their frontier orbitals by
//! the projection method, the reorganization energy of the four-point scheme, and the Marcus
//! rate of a hop.
//!
//! **Projection method** (E. F. Valeev, V. Coropceanu, D. A. da Silva Filho, S. Salman,
//! J.-L. Bredas, J. Am. Chem. Soc. 2006, 128, 9882; B. Baumeier, J. Kirkpatrick, D. Andrienko,
//! Phys. Chem. Chem. Phys. 2010, 12, 11103): the orbitals `phi_a`, `phi_b` of the two molecules,
//! each computed on its own at its place in the pair, written in the atomic orbitals of the
//! pair; the Fock matrix of the pair from its orbitals, `F = S C diag(e) C^T S` (exact when they
//! span its basis, as in a minimal basis); `e_a = phi_a F phi_a`, `e_b`, `J = phi_a F phi_b`,
//! `S_ab = phi_a S phi_b`; and after the symmetric orthogonalization of the two orbitals
//!
//! ```text
//! J_eff = (J - (e_a + e_b) S_ab / 2) / (1 - S_ab^2),
//! e_a,eff = ((e_a + e_b) / 2 - J S_ab + (e_a - e_b) sqrt(1 - S_ab^2) / 2) / (1 - S_ab^2),
//! ```
//!
//! `e_b,eff` with the opposite sign of the last term. Holes hop between the highest occupied
//! orbitals, electrons between the lowest empty ones. The orbitals are those of the neutral
//! molecules and pair (the frozen-orbital picture of the method). Where the frontier orbital
//! of a molecule is degenerate (within [`DEGENERATE`], as the pi orbitals of acetylene), its
//! choice within the degenerate set is arbitrary and so would be `J`: the combinations of each
//! set are taken that make the largest coupling (the singular vectors of the largest singular
//! value of the matrix of `J_eff` between the two sets), which do not depend on the rotation
//! within each set and are the pair that splits most in the dimer.
//!
//! **Four-point reorganization energy** (S. F. Nelsen, S. C. Blackstock, Y. Kim, J. Am. Chem.
//! Soc. 1987, 109, 677): for a hop from `a` to `b`, each molecule in the charge it ends with, at
//! the equilibrium of the charge it starts with, less at its own equilibrium:
//! `lambda = [E_a^end(R_a^start) - E_a^end(R_a^end)] + [E_b^end(R_b^start) - E_b^end(R_b^end)]`;
//! for two equal molecules the usual `[E_c(R_n) - E_c(R_c)] + [E_n(R_c) - E_n(R_n)]`.
//!
//! **The rate of a hop** (Marcus): `k = (2 pi / hbar) |J_eff|^2 (4 pi lambda k_B T)^(-1/2)
//! exp(-(dG + lambda)^2 / (4 lambda k_B T))`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::{OrbitalSet, Orbitals};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// Orbitals within this energy (Hartree, 0.27 meV) of the frontier orbital are degenerate with
/// it: above the splitting of a degeneracy by symmetry in the arithmetic (1e-15 Hartree in the
/// pi orbitals of acetylene with GFN1-xTB), below the couplings that are measured (meV).
pub const DEGENERATE: f64 = 1e-5;

/// The carrier that hops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carrier {
    /// A hole, between the highest occupied orbitals.
    Hole,
    /// An electron, between the lowest empty orbitals.
    Electron,
}

impl Carrier {
    /// `hole` or `electron`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "hole" | "h" | "p" => Ok(Self::Hole),
            "electron" | "e" | "n" => Ok(Self::Electron),
            other => Err(MoptError::parse(format!(
                "carrier '{other}': hole or electron"
            ))),
        }
    }

    /// `hole` or `electron`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Hole => "hole",
            Self::Electron => "electron",
        }
    }

    /// The index of the frontier orbital of this carrier among the orbitals `set`, which hold
    /// up to `per_orbital` electrons (2 for the spatial orbitals of a closed shell). The
    /// occupied orbitals are counted from the total occupation, as a smeared occupation (an
    /// electronic temperature) puts a little on the lowest empty orbitals.
    pub fn frontier(self, set: &OrbitalSet, per_orbital: f64) -> MoptResult<usize> {
        let total: f64 = set.occupations.iter().sum();
        let occupied = (total / per_orbital).round();
        if !occupied.is_finite() || occupied < 0.0 {
            return Err(MoptError::calc(format!(
                "transfer integral: an occupation of {total} electrons"
            )));
        }
        let occupied = occupied as usize;
        match self {
            Self::Hole if occupied > 0 && occupied <= set.energies.len() => Ok(occupied - 1),
            Self::Electron if occupied < set.energies.len() => Ok(occupied),
            _ => Err(MoptError::calc(format!(
                "transfer integral: no {} orbital in a set of {} orbitals with {occupied} \
                 occupied",
                if self == Self::Hole {
                    "occupied"
                } else {
                    "empty"
                },
                set.energies.len()
            ))),
        }
    }
}

/// The Fock matrix `S C diag(e) C^T S` of the orbitals `set` with the overlap `overlap` (the
/// matrix itself when the orbitals span the atomic orbitals).
pub fn fock_from_orbitals(overlap: &Mat<f64>, set: &OrbitalSet) -> Mat<f64> {
    let sc = overlap * &set.coefficients;
    let n = sc.nrows();
    let scaled = Mat::from_fn(n, sc.ncols(), |i, k| sc[(i, k)] * set.energies[k]);
    &scaled * sc.transpose()
}

/// The transfer integral of two orbitals and their energies (Hartree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferIntegral {
    /// `J = phi_a F phi_b`.
    pub coupling: f64,
    /// `e_a`, `e_b`.
    pub site_energies: [f64; 2],
    /// `S_ab`.
    pub overlap: f64,
    /// `J_eff` of the orthogonalized orbitals.
    pub effective: f64,
    /// The site energies of the orthogonalized orbitals.
    pub effective_site_energies: [f64; 2],
    /// The number of degenerate frontier orbitals of each molecule (1 and 1 for orbitals
    /// given as they are).
    pub degeneracy: [usize; 2],
}

/// The transfer integral of the orbitals `a` and `b` (in the atomic orbitals of the pair, each
/// normalized here in its metric) with the Fock matrix `fock` and the overlap `overlap` of the
/// pair.
pub fn transfer_integral(
    fock: &Mat<f64>,
    overlap: &Mat<f64>,
    a: &Col<f64>,
    b: &Col<f64>,
) -> MoptResult<TransferIntegral> {
    let n = fock.nrows();
    if fock.ncols() != n
        || overlap.nrows() != n
        || overlap.ncols() != n
        || a.nrows() != n
        || b.nrows() != n
    {
        return Err(MoptError::config(
            "transfer integral: the orbitals and matrices differ in their atomic orbitals",
        ));
    }
    let form = |m: &Mat<f64>, x: &Col<f64>, y: &Col<f64>| -> f64 {
        let my = m * y;
        (0..n).map(|i| x[i] * my[i]).sum()
    };
    let norm_a = form(overlap, a, a).sqrt();
    let norm_b = form(overlap, b, b).sqrt();
    if !(norm_a.is_finite() && norm_a > 0.0 && norm_b.is_finite() && norm_b > 0.0) {
        return Err(MoptError::calc("transfer integral: an orbital has no norm"));
    }
    let (ea, eb) = (
        form(fock, a, a) / (norm_a * norm_a),
        form(fock, b, b) / (norm_b * norm_b),
    );
    let j = form(fock, a, b) / (norm_a * norm_b);
    let s = form(overlap, a, b) / (norm_a * norm_b);
    let den = 1.0 - s * s;
    if den.is_nan() || den <= 0.0 {
        return Err(MoptError::calc(
            "transfer integral: the two orbitals are the same (overlap 1)",
        ));
    }
    let mean = 0.5 * (ea + eb);
    let half_gap = 0.5 * (ea - eb) * den.sqrt();
    Ok(TransferIntegral {
        coupling: j,
        site_energies: [ea, eb],
        overlap: s,
        effective: (j - mean * s) / den,
        effective_site_energies: [
            (mean - j * s + half_gap) / den,
            (mean - j * s - half_gap) / den,
        ],
        degeneracy: [1, 1],
    })
}

/// The transfer integral of the best-matched combinations of the orbitals `a` and of the
/// orbitals `b` (each set orthonormal in the metric of the pair, as the degenerate orbitals of a
/// molecule are): with the matrix `M_ij = J_eff(a_i, b_j)`, the combinations `sum_i u_i a_i` and
/// `sum_j v_j b_j` of the singular vectors of its largest singular value. One orbital each is
/// [`transfer_integral`] itself.
pub fn matched_transfer_integral(
    fock: &Mat<f64>,
    overlap: &Mat<f64>,
    a: &[Col<f64>],
    b: &[Col<f64>],
) -> MoptResult<TransferIntegral> {
    if a.is_empty() || b.is_empty() {
        return Err(MoptError::config("transfer integral: no orbitals"));
    }
    if a.len() == 1 && b.len() == 1 {
        return transfer_integral(fock, overlap, &a[0], &b[0]);
    }
    let mut m = Mat::<f64>::zeros(a.len(), b.len());
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            m[(i, j)] = transfer_integral(fock, overlap, x, y)?.effective;
        }
    }
    // The right singular vector of the largest singular value: the top eigenvector of M^T M.
    let (values, vectors) = symmetric_eigen(&(m.transpose() * &m))?;
    let top = values.nrows() - 1;
    let v = vectors.col(top).to_owned();
    let mv = &m * &v;
    let norm = mv.norm_l2();
    // No coupling at all: any combination will do.
    let u = if norm > 0.0 {
        mv * (1.0 / norm)
    } else {
        Col::from_fn(a.len(), |i| if i == 0 { 1.0 } else { 0.0 })
    };
    let combine = |set: &[Col<f64>], w: &Col<f64>| {
        let mut out = Col::<f64>::zeros(set[0].nrows());
        for (k, x) in set.iter().enumerate() {
            out += x * w[k];
        }
        out
    };
    let mut ti = transfer_integral(fock, overlap, &combine(a, &u), &combine(b, &v))?;
    ti.degeneracy = [a.len(), b.len()];
    Ok(ti)
}

/// The atomic orbitals of each of `atoms` atoms, in their order, from the atom of every atomic
/// orbital.
fn orbitals_of_atoms(atom_of_ao: &[usize], atoms: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); atoms];
    for (mu, &a) in atom_of_ao.iter().enumerate() {
        if a < atoms {
            out[a].push(mu);
        }
    }
    out
}

/// The orbital `orbital` of a molecule (in its own atomic orbitals, `atom_of_ao`) written in the
/// atomic orbitals of the pair (`pair_atom_of_ao`, `pair_atoms` atoms), the molecule's atom `k`
/// being atom `atoms[k]` of the pair. An atom has the same atomic orbitals, in the same order,
/// alone and in the pair (one basis per element).
pub fn embed(
    orbital: &Col<f64>,
    atom_of_ao: &[usize],
    atoms: &[usize],
    pair_atom_of_ao: &[usize],
    pair_atoms: usize,
) -> MoptResult<Col<f64>> {
    if orbital.nrows() != atom_of_ao.len() {
        return Err(MoptError::config(format!(
            "transfer integral: an orbital of {} atomic orbitals with the atoms of {}",
            orbital.nrows(),
            atom_of_ao.len()
        )));
    }
    let own = orbitals_of_atoms(atom_of_ao, atoms.len());
    let pair = orbitals_of_atoms(pair_atom_of_ao, pair_atoms);
    let mut out = Col::<f64>::zeros(pair_atom_of_ao.len());
    for (k, &a) in atoms.iter().enumerate() {
        let target = pair.get(a).ok_or_else(|| {
            MoptError::config(format!("transfer integral: no atom {} in the pair", a + 1))
        })?;
        if target.len() != own[k].len() {
            return Err(MoptError::config(format!(
                "transfer integral: atom {} has {} atomic orbitals alone and {} in the pair",
                a + 1,
                own[k].len(),
                target.len()
            )));
        }
        for (&mu, &nu) in own[k].iter().zip(target) {
            out[nu] = orbital[mu];
        }
    }
    Ok(out)
}

/// The atom of every atomic orbital of `o`, or why it cannot be used.
fn atoms_of(o: &Orbitals, what: &str) -> MoptResult<Vec<usize>> {
    o.atom_of_ao.clone().ok_or_else(|| {
        MoptError::unsupported(format!(
            "transfer integral: the orbitals of {what} do not say which atom each atomic \
             orbital is on"
        ))
    })
}

/// The orbitals of `o` the projection method takes and the electrons each holds: the spatial
/// orbitals of a closed shell, or the alpha orbitals.
fn spatial(o: &Orbitals) -> (&OrbitalSet, f64) {
    (&o.alpha, if o.beta.is_none() { 2.0 } else { 1.0 })
}

/// The transfer integral of the frontier orbitals of `carrier` of the molecules `a` and `b`
/// (each computed alone at its place in the pair) in the pair `pair`, their atoms being
/// `atoms_a` and `atoms_b` (0-based) of the pair. All three need the atom of each atomic
/// orbital; the closed shells of the neutral molecules are meant (of an open shell the alpha
/// orbitals are taken).
pub fn fragment_transfer_integral(
    pair: &Orbitals,
    a: &Orbitals,
    b: &Orbitals,
    atoms_a: &[usize],
    atoms_b: &[usize],
    carrier: Carrier,
) -> MoptResult<TransferIntegral> {
    let pair_atom_of = atoms_of(pair, "the pair")?;
    let a_atom_of = atoms_of(a, "the first molecule")?;
    let b_atom_of = atoms_of(b, "the second molecule")?;
    if atoms_a.iter().any(|x| atoms_b.contains(x)) {
        return Err(MoptError::config(
            "transfer integral: the two molecules share atoms",
        ));
    }
    let pair_atoms = pair_atom_of.iter().copied().max().map_or(0, |m| m + 1);
    // The frontier orbital and those degenerate with it, in the atomic orbitals of the pair.
    let frontier =
        |o: &Orbitals, atom_of: &[usize], atoms: &[usize]| -> MoptResult<Vec<Col<f64>>> {
            let (set, per) = spatial(o);
            let k = carrier.frontier(set, per)?;
            let e = set.energies[k];
            (0..set.energies.len())
                .filter(|&i| (set.energies[i] - e).abs() < DEGENERATE)
                .map(|i| {
                    let column = set.coefficients.col(i).to_owned();
                    embed(&column, atom_of, atoms, &pair_atom_of, pair_atoms)
                })
                .collect()
        };
    let phi_a = frontier(a, &a_atom_of, atoms_a)?;
    let phi_b = frontier(b, &b_atom_of, atoms_b)?;
    let fock = fock_from_orbitals(&pair.overlap, spatial(pair).0);
    matched_transfer_integral(&fock, &pair.overlap, &phi_a, &phi_b)
}

/// The four-point reorganization energy of a hop from `a` to `b` (Hartree): for each molecule
/// its energy in the charge it ends with, at the equilibrium of the charge it starts with and
/// at its own equilibrium (`[at_start, at_own]` for `a`, then for `b`).
pub fn four_point_reorganization(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - a[1]) + (b[0] - b[1])
}

/// The Marcus rate (1/s) of a hop with the transfer integral `coupling`, the reorganization
/// energy `reorganization`, and the free energy of the hop `delta_g` (Hartree) at `kt`
/// (Hartree).
pub fn hopping_rate(coupling: f64, reorganization: f64, delta_g: f64, kt: f64) -> MoptResult<f64> {
    if !(reorganization.is_finite() && reorganization > 0.0) {
        return Err(MoptError::config(
            "hopping rate: the reorganization energy must be positive",
        ));
    }
    super::inner_sphere::marcus_with_tunnelling(
        0.0,
        coupling,
        reorganization,
        (reorganization + delta_g).powi(2) / (4.0 * reorganization),
        kt,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A symmetric pair of one orbital each: `H = [[e, t], [t, e]]`, `S = [[1, s], [s, 1]]`,
    /// with its generalized eigenvectors `(1, +-1) / sqrt(2 (1 +- s))` and eigenvalues
    /// `(e +- t) / (1 +- s)`.
    fn pair(e: f64, t: f64, s: f64) -> (Mat<f64>, Mat<f64>, OrbitalSet) {
        let h = Mat::from_fn(2, 2, |i, j| if i == j { e } else { t });
        let overlap = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { s });
        let plus = 1.0 / (2.0 * (1.0 + s)).sqrt();
        let minus = 1.0 / (2.0 * (1.0 - s)).sqrt();
        let c = Mat::from_fn(2, 2, |i, k| match (i, k) {
            (_, 0) => plus,
            (0, _) => minus,
            _ => -minus,
        });
        let set = OrbitalSet::new(
            c,
            vec![(e + t) / (1.0 + s), (e - t) / (1.0 - s)],
            vec![2.0, 0.0],
        )
        .unwrap();
        (h, overlap, set)
    }

    #[test]
    fn the_fock_matrix_comes_back_from_a_complete_set_of_orbitals() {
        let (h, overlap, set) = pair(-0.4, -0.05, 0.2);
        let f = fock_from_orbitals(&overlap, &set);
        for i in 0..2 {
            for j in 0..2 {
                assert!((f[(i, j)] - h[(i, j)]).abs() < 1e-14, "{i} {j}");
            }
        }
    }

    #[test]
    fn the_orthogonalized_pair_has_the_eigenvalues_of_the_pair() {
        let (e, t, s) = (-0.4, -0.05, 0.2);
        let (h, overlap, _) = pair(e, t, s);
        let a = Col::from_fn(2, |i| if i == 0 { 1.0 } else { 0.0 });
        let b = Col::from_fn(2, |i| if i == 1 { 1.0 } else { 0.0 });
        let ti = transfer_integral(&h, &overlap, &a, &b).unwrap();
        assert!((ti.coupling - t).abs() < 1e-15 && (ti.overlap - s).abs() < 1e-15);
        assert!((ti.effective - (t - e * s) / (1.0 - s * s)).abs() < 1e-15);
        // In the orthogonalized basis the eigenvalues are e_eff +- J_eff.
        let [ea, eb] = ti.effective_site_energies;
        assert!((ea - eb).abs() < 1e-15);
        let roots = [ea + ti.effective, ea - ti.effective];
        let exact = [(e + t) / (1.0 + s), (e - t) / (1.0 - s)];
        for (r, x) in roots.iter().zip(exact) {
            assert!((r - x).abs() < 1e-14, "{r} {x}");
        }
        // Scaling an orbital changes nothing (each is normalized in the metric of the pair).
        let scaled = transfer_integral(&h, &overlap, &(&a * 3.0), &b).unwrap();
        assert!((scaled.effective - ti.effective).abs() < 1e-15);
    }

    #[test]
    fn unequal_sites_keep_the_trace_of_the_pair() {
        // H = [[e1, t], [t, e2]] with S = [[1, s], [s, 1]]: the trace of the orthogonalized
        // 2x2 problem is the trace of S^-1 H.
        let (e1, e2, t, s) = (-0.42, -0.37, -0.03, 0.15);
        let h = Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => e1,
            (1, 1) => e2,
            _ => t,
        });
        let overlap = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { s });
        let a = Col::from_fn(2, |i| if i == 0 { 1.0 } else { 0.0 });
        let b = Col::from_fn(2, |i| if i == 1 { 1.0 } else { 0.0 });
        let ti = transfer_integral(&h, &overlap, &a, &b).unwrap();
        let [ea, eb] = ti.effective_site_energies;
        let trace = (e1 + e2 - 2.0 * t * s) / (1.0 - s * s);
        assert!((ea + eb - trace).abs() < 1e-15);
        // ... and the determinant of S^-1 H.
        let det = (e1 * e2 - t * t) / (1.0 - s * s);
        assert!((ea * eb - ti.effective.powi(2) - det).abs() < 1e-15);
    }

    #[test]
    fn an_orbital_goes_to_the_atomic_orbitals_of_its_atoms_in_the_pair() {
        // A molecule of two atoms (one and two atomic orbitals) at atoms 2 and 0 of a pair of
        // three atoms (two, one, and one atomic orbitals).
        let orbital = Col::from_fn(3, |i| [0.5, -0.3, 0.8][i]);
        let own = [0, 1, 1];
        let pair_atom_of = [0, 0, 1, 2];
        let out = embed(&orbital, &own, &[2, 0], &pair_atom_of, 3).unwrap();
        assert_eq!(
            (0..4).map(|i| out[i]).collect::<Vec<_>>(),
            vec![-0.3, 0.8, 0.0, 0.5]
        );
        // Another number of atomic orbitals on an atom is refused.
        assert!(embed(&orbital, &own, &[0, 2], &pair_atom_of, 3).is_err());
    }

    #[test]
    fn degenerate_orbitals_couple_by_their_best_matched_combination() {
        // Two degenerate orbitals on a (atomic orbitals 0 and 1) and one on b (2), orthonormal,
        // with couplings t1 and t2 to b: the largest singular value is sqrt(t1^2 + t2^2),
        // whatever the rotation of the pair on a.
        let (e, t1, t2): (f64, f64, f64) = (-0.45, -0.004, 0.003);
        let fock = Mat::from_fn(3, 3, |i, j| match (i, j) {
            _ if i == j => e,
            (0, 2) | (2, 0) => t1,
            (1, 2) | (2, 1) => t2,
            _ => 0.0,
        });
        let overlap = Mat::<f64>::identity(3, 3);
        let unit = |k: usize| Col::from_fn(3, |i| if i == k { 1.0 } else { 0.0 });
        let want = (t1 * t1 + t2 * t2).sqrt();
        for angle in [0.0_f64, 0.3, 1.1] {
            let (c, s) = (angle.cos(), angle.sin());
            let a = [&unit(0) * c + &unit(1) * s, &unit(1) * c - &unit(0) * s];
            let ti = matched_transfer_integral(&fock, &overlap, &a, &[unit(2)]).unwrap();
            assert!((ti.effective.abs() - want).abs() < 1e-15, "{angle}");
            assert_eq!(ti.degeneracy, [2, 1]);
            assert!((ti.site_energies[0] - e).abs() < 1e-15);
        }
    }

    #[test]
    fn the_frontier_is_counted_from_the_total_occupation() {
        // A smeared closed shell of four electrons: the lowest empty orbital holds a little.
        let set = OrbitalSet::new(
            Mat::identity(4, 4),
            vec![-0.6, -0.4, -0.1, 0.2],
            vec![2.0, 1.999_999, 1.0e-6, 0.0],
        )
        .unwrap();
        assert_eq!(Carrier::Hole.frontier(&set, 2.0).unwrap(), 1);
        assert_eq!(Carrier::Electron.frontier(&set, 2.0).unwrap(), 2);
        assert_eq!(Carrier::parse("hole").unwrap(), Carrier::Hole);
        assert!(Carrier::parse("proton").is_err());
    }

    #[test]
    fn a_self_exchange_hops_over_a_quarter_of_the_reorganization() {
        let kt = 0.000_944; // about 298 K
        let (j, lambda) = (0.002, 0.01);
        let k = hopping_rate(j, lambda, 0.0, kt).unwrap();
        let direct =
            super::super::inner_sphere::marcus_with_tunnelling(0.0, j, lambda, lambda / 4.0, kt)
                .unwrap();
        assert!((k / direct - 1.0).abs() < 1e-12);
        // Activationless at -dG = lambda: the barrier vanishes.
        let top = hopping_rate(j, lambda, -lambda, kt).unwrap();
        assert!((top / k / (lambda / (4.0 * kt)).exp() - 1.0).abs() < 1e-12);
        assert!((four_point_reorganization([0.003, 0.0], [0.002, -0.001]) - 0.006).abs() < 1e-15);
        assert!(hopping_rate(j, 0.0, 0.0, kt).is_err());
    }
}
