//! The analytic gradient of an excitation energy of the Tamm-Dancoff approximation on
//! tight-binding orbitals (the excited states of `gfn0::excited`): what the derivatives of the
//! Hamiltonian, the overlap, and the kernels of a method are contracted with.
//!
//! For a closed-shell reference with orbitals `C` (`C^T S C = 1`, energies `e`, `o` occupied
//! and `v` empty) and a singlet state with the normalized amplitudes `X` (`o x v`),
//!
//! ```text
//! w = sum_pq T_pq F_pq + 2 Q^T K Q,   T_ab = (X^T X)_ab,  T_ij = -(X X^T)_ij,
//! Q_s = sum_ia X_ia q_s^ia
//! ```
//!
//! with the transition charges `q_s^pq` of the sites `s` (Mulliken charges of shells, or
//! Loewdin charges of atoms) and the kernel `K` of the sites. The excitation energy is
//! stationary in `X`, so its derivative is that of `w` at fixed `X` with the orbitals
//! following the structure, `dC = C U` (Furche, Ahlrichs, J. Chem. Phys. 2002, 117, 7433, for
//! the Lagrangian of TDDFT; Heringer, Niehaus, Frauenheim, J. Comput. Chem. 2007, 28, 2589,
//! for its tight-binding form). The blocks of `U` within the occupied and within the empty
//! orbitals are taken as `-S^x / 2` (a rotation within them changes only the representation of
//! the state); the occupied-empty block follows from the ground state staying converged, a
//! linear equation for every coordinate, which one Z-vector replaces:
//!
//! ```text
//! (e_a - e_i) Z_ai + 4 sum_bj (q^ai K_scc q^bj) Z_bj = L_ai
//! ```
//!
//! with `K_scc` the kernel of the self-consistent charges of the ground state (none for a
//! method without self-consistency, where `Z = L / (e_a - e_i)`). The gradient is then
//!
//! ```text
//! dw/dx = Tr(P~ dF/dx |_q) + Tr(M dS/dx) + sum_(s!=t) G_st dK_st/dx + sum_(s!=t) G'_st dgamma_st/dx
//! ```
//!
//! with the relaxed difference density `P~ = C T C^T - sym(C_v Z C_o^T)`, the overlap weights
//! `M` (the orthonormality of the orbitals, the overlap in the transition charges, and the
//! overlap in the Mulliken charges of the ground state), `G = 2 Q Q^T`, and `G'` the weights of
//! the derivatives of the kernel of the ground state at its fixed charges. [`terms`] computes
//! `P~`, `M`, `G` and `G'`; every method contracts them with its own derivatives as it does the
//! density and the energy-weighted density of its ground state.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// Occupied and empty orbitals closer than this (Hartree) leave the Z-vector undefined.
const DEGENERATE_GAP: f64 = 1.0e-8;
/// Relative residual at which the conjugate gradients of the Z-vector stop.
const Z_TOLERANCE: f64 = 1.0e-12;
/// Iterations of the conjugate gradients at most.
const Z_ITERATIONS: usize = 500;

/// How the transition charges are taken.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Charges<'a> {
    /// Mulliken charges of the sites (shells) of the atomic orbitals.
    Mulliken {
        /// Site of every atomic orbital.
        site_of_ao: &'a [usize],
        /// Number of sites.
        sites: usize,
    },
    /// Loewdin charges of the atoms (`C' = S^(1/2) C`).
    Lowdin {
        /// Atom of every atomic orbital.
        atom_of_ao: &'a [usize],
        /// Number of atoms.
        atoms: usize,
    },
}

/// The self-consistent charges of the ground state (GFN1-xTB).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SccCoupling<'a> {
    /// Shell of every atomic orbital.
    pub shell_of_ao: &'a [usize],
    /// The second derivatives of the energy in the shell charges (`dv/dq`).
    pub kernel: &'a Mat<f64>,
    /// The shell charges of the ground state.
    pub charges: &'a [f64],
    /// The density of the ground state (atomic orbitals).
    pub density: &'a Mat<f64>,
}

/// The third-order term of a Mulliken kernel on shells: `2 Gamma_A q_A` between the shells of
/// atom A, with `q_A` the charge of the ground state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ThirdOrder<'a> {
    /// Atom of every shell.
    pub atom_of_site: &'a [usize],
    /// `Gamma_A` of every atom.
    pub gamma3: &'a [f64],
}

/// What the gradient of one excitation energy is computed from.
pub(crate) struct Input<'a> {
    /// The orbitals in the atomic orbitals (columns, ascending energies).
    pub c: &'a Mat<f64>,
    /// Their energies.
    pub eps: &'a [f64],
    /// Doubly occupied orbitals.
    pub occupied: usize,
    /// The overlap of the atomic orbitals.
    pub s: &'a Mat<f64>,
    /// The normalized amplitudes of the state, `occupied x virtual`.
    pub x: &'a Mat<f64>,
    /// The transition charges.
    pub charges: Charges<'a>,
    /// The kernel of the sites of the transition charges.
    pub kernel: &'a Mat<f64>,
    /// The charge-dependent part of a Mulliken kernel of shells.
    pub third: Option<ThirdOrder<'a>>,
    /// The self-consistent charges of the ground state.
    pub scc: Option<SccCoupling<'a>>,
}

/// What the derivatives of a method are contracted with.
#[derive(Debug, Clone)]
pub(crate) struct Terms {
    /// `P~`: the weights of the derivatives of the Fock matrix at fixed ground-state charges.
    pub density: Mat<f64>,
    /// `M`: the weights of the derivatives of the overlap, `sum_mn M_mn dS_mn`.
    pub overlap: Mat<f64>,
    /// `G = 2 Q Q^T`: the weights of the derivatives of the kernel of the transition charges.
    pub kernel: Mat<f64>,
    /// `G'`: the weights of the derivatives of the kernel of the ground state at its fixed
    /// charges (`None` without self-consistent charges).
    pub scc: Option<Mat<f64>>,
}

fn sym(a: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| 0.5 * (a[(i, j)] + a[(j, i)]))
}

fn columns(c: &Mat<f64>, from: usize, to: usize) -> Mat<f64> {
    Mat::from_fn(c.nrows(), to - from, |m, k| c[(m, from + k)])
}

/// `w` of the sites as one value per atomic orbital.
fn per_ao(site_of_ao: &[usize], w: &[f64]) -> Vec<f64> {
    site_of_ao.iter().map(|&s| w[s]).collect()
}

/// `1/2 (diag(w) S + S diag(w))`: the matrix whose orbital elements are `sum_s w_s q_s^pq`
/// for Mulliken charges.
fn mulliken_matrix(s: &Mat<f64>, w_ao: &[f64]) -> Mat<f64> {
    Mat::from_fn(s.nrows(), s.ncols(), |m, n| {
        0.5 * (w_ao[m] + w_ao[n]) * s[(m, n)]
    })
}

/// The Mulliken charges `sum_(m on s) (P S)_mm` of the symmetric `p` on the sites.
fn mulliken_charges(p: &Mat<f64>, s: &Mat<f64>, site_of_ao: &[usize], sites: usize) -> Vec<f64> {
    let mut out = vec![0.0; sites];
    for (m, &site) in site_of_ao.iter().enumerate() {
        out[site] += (0..p.ncols()).map(|n| p[(m, n)] * s[(n, m)]).sum::<f64>();
    }
    out
}

/// `S^(1/2)` with the eigenvectors and the square roots of the eigenvalues of `S`.
struct Root {
    r: Mat<f64>,
    vectors: Mat<f64>,
    roots: Vec<f64>,
}

impl Root {
    fn of(s: &Mat<f64>) -> MoptResult<Self> {
        let (values, vectors) = symmetric_eigen(s)?;
        if values.iter().any(|v| *v <= 1.0e-10) {
            return Err(MoptError::linalg(
                "the overlap matrix of the tight-binding basis is not positive definite",
            ));
        }
        let roots: Vec<f64> = values.iter().map(|v| v.sqrt()).collect();
        let n = s.nrows();
        let scaled = Mat::from_fn(n, n, |i, k| vectors[(i, k)] * roots[k]);
        let r = &scaled * vectors.transpose();
        Ok(Self { r, vectors, roots })
    }

    /// `M~` with `sum_mn M_mn dR_mn = sum_mn M~_mn dS_mn` for `R = S^(1/2)` and a symmetric `M`
    /// (`dR R + R dR = dS`, solved in the eigenvectors of `S`).
    fn to_overlap(&self, m: &Mat<f64>) -> Mat<f64> {
        let v = &self.vectors;
        let inner = v.transpose() * m * v;
        let scaled = Mat::from_fn(inner.nrows(), inner.ncols(), |k, l| {
            inner[(k, l)] / (self.roots[k] + self.roots[l])
        });
        v * scaled * v.transpose()
    }
}

/// The weights of the gradient of the excitation energy of `input`.
pub(crate) fn terms(input: &Input<'_>) -> MoptResult<Terms> {
    let (c, s, x, eps) = (input.c, input.s, input.x, input.eps);
    let nao = c.nrows();
    let o = input.occupied;
    let mo = c.ncols();
    if o == 0 || o >= mo || x.nrows() != o || x.ncols() != mo - o || eps.len() != mo {
        return Err(MoptError::config(format!(
            "excited-state gradient: {}x{} amplitudes for {o} occupied of {mo} orbitals",
            x.nrows(),
            x.ncols()
        )));
    }
    let v = mo - o;
    let co = columns(c, 0, o);
    let cv = columns(c, o, mo);
    let delta = Mat::from_fn(v, o, |a, i| eps[o + a] - eps[i]);
    if (0..v).any(|a| (0..o).any(|i| delta[(a, i)] < DEGENERATE_GAP)) {
        return Err(MoptError::calc(
            "excited-state gradient: an empty orbital at or below an occupied one",
        ));
    }

    // The transition density, the charges of the state, and the matrices of site weights.
    let d = &co * x * cv.transpose();
    let ds = sym(&d);
    let root = match input.charges {
        Charges::Lowdin { .. } => Some(Root::of(s)?),
        Charges::Mulliken { .. } => None,
    };
    let (site_of_ao, sites) = match input.charges {
        Charges::Mulliken { site_of_ao, sites } => (site_of_ao, sites),
        Charges::Lowdin { atom_of_ao, atoms } => (atom_of_ao, atoms),
    };
    if input.kernel.nrows() != sites || input.kernel.ncols() != sites || site_of_ao.len() != nao {
        return Err(MoptError::config(
            "excited-state gradient: the kernel does not fit the sites of the charges",
        ));
    }
    let q: Vec<f64> = match (&root, input.charges) {
        (Some(r), _) => {
            let rdr = &r.r * &d * &r.r;
            let mut q = vec![0.0; sites];
            for (m, &site) in site_of_ao.iter().enumerate() {
                q[site] += rdr[(m, m)];
            }
            q
        }
        (None, _) => mulliken_charges(&ds, s, site_of_ao, sites),
    };
    // The orbital matrix `sum_s w_s q_s^pq` of the transition charges.
    let charge_matrix = |w: &[f64]| -> Mat<f64> {
        let w_ao = per_ao(site_of_ao, w);
        let ao = match &root {
            Some(r) => {
                let scaled = Mat::from_fn(nao, nao, |m, n| w_ao[m] * r.r[(m, n)]);
                &r.r * scaled
            }
            None => mulliken_matrix(s, &w_ao),
        };
        c.transpose() * ao * c
    };
    let lambda: Vec<f64> = (0..sites)
        .map(|s1| {
            4.0 * (0..sites)
                .map(|t| input.kernel[(s1, t)] * q[t])
                .sum::<f64>()
        })
        .collect();
    let big = charge_matrix(&lambda);
    let block = |m: &Mat<f64>, r0: usize, r1: usize, c0: usize, c1: usize| {
        Mat::from_fn(r1 - r0, c1 - c0, |i, j| m[(r0 + i, c0 + j)])
    };
    let l_oo = block(&big, 0, o, 0, o);
    let l_ov = block(&big, 0, o, o, mo);
    let l_vo = block(&big, o, mo, 0, o);
    let l_vv = block(&big, o, mo, o, mo);

    // The unrelaxed difference density.
    let xtx = x.transpose() * x;
    let xxt = x * x.transpose();
    let t_ao = &cv * &xtx * cv.transpose() - &co * &xxt * co.transpose();

    // The right side of the Z-vector equation.
    let mut l = &l_vv * x.transpose() - x.transpose() * &l_oo;
    let atom_charge = |third: &ThirdOrder<'_>| -> Vec<f64> {
        let mut qa = vec![0.0; third.gamma3.len()];
        for (site, &atom) in third.atom_of_site.iter().enumerate() {
            qa[atom] += q[site];
        }
        qa
    };
    if let Some(scc) = &input.scc {
        let shells = scc.kernel.nrows();
        let tau = mulliken_charges(&t_ao, s, scc.shell_of_ao, shells);
        let w: Vec<f64> = (0..shells)
            .map(|a| {
                4.0 * (0..shells)
                    .map(|b| scc.kernel[(a, b)] * tau[b])
                    .sum::<f64>()
            })
            .collect();
        let m = c.transpose() * mulliken_matrix(s, &per_ao(scc.shell_of_ao, &w)) * c;
        l += block(&m, o, mo, 0, o);
        if let Some(third) = &input.third {
            let qa = atom_charge(third);
            let w: Vec<f64> = (0..shells)
                .map(|a| {
                    let atom = third.atom_of_site[a];
                    -16.0 * third.gamma3[atom] * qa[atom] * qa[atom]
                })
                .collect();
            let m = c.transpose() * mulliken_matrix(s, &per_ao(scc.shell_of_ao, &w)) * c;
            l += block(&m, o, mo, 0, o);
        }
    }

    // The Z-vector: (e_a - e_i) Z + 4 q K_scc q^T Z = L.
    let z = match &input.scc {
        None => Mat::from_fn(v, o, |a, i| l[(a, i)] / delta[(a, i)]),
        Some(scc) => {
            let shells = scc.kernel.nrows();
            let apply = |vec: &Mat<f64>| -> Mat<f64> {
                let dv = sym(&(&cv * vec * co.transpose()));
                let rho = mulliken_charges(&dv, s, scc.shell_of_ao, shells);
                let w: Vec<f64> = (0..shells)
                    .map(|a| {
                        4.0 * (0..shells)
                            .map(|b| scc.kernel[(a, b)] * rho[b])
                            .sum::<f64>()
                    })
                    .collect();
                let m = cv.transpose() * mulliken_matrix(s, &per_ao(scc.shell_of_ao, &w)) * &co;
                Mat::from_fn(v, o, |a, i| delta[(a, i)] * vec[(a, i)] + m[(a, i)])
            };
            conjugate_gradients(&apply, &l, &delta)?
        }
    };

    // The relaxed difference density.
    let p = &t_ao - sym(&(&cv * &z * co.transpose()));

    // The weights of the derivatives of the overlap.
    let mut overlap = Mat::<f64>::zeros(nao, nao);
    // The orthonormality of the orbitals in the difference density: -1/2 T (e_p + e_q).
    let y_vv = Mat::from_fn(v, v, |a, b| -0.5 * xtx[(a, b)] * (eps[o + a] + eps[o + b]));
    let y_oo = Mat::from_fn(o, o, |i, j| 0.5 * xxt[(i, j)] * (eps[i] + eps[j]));
    overlap += &cv * y_vv * cv.transpose() + &co * y_oo * co.transpose();
    // The orthonormality in the response of the ground state: Z_ai e_i.
    let ze = Mat::from_fn(v, o, |a, i| z[(a, i)] * eps[i]);
    overlap += sym(&(&cv * ze * co.transpose()));
    // The overlap in the transition charges at fixed orbitals.
    let lambda_ao = per_ao(site_of_ao, &lambda);
    match &root {
        Some(r) => {
            let scaled = Mat::from_fn(nao, nao, |m, n| r.r[(m, n)] * lambda_ao[n]);
            let left = &d * &scaled;
            let scaled_left = Mat::from_fn(nao, nao, |m, n| lambda_ao[m] * r.r[(m, n)]);
            let right = scaled_left * &d;
            let n = sym(&(left + right));
            overlap += r.to_overlap(&n);
        }
        None => {
            overlap += Mat::from_fn(nao, nao, |m, n| {
                0.5 * (lambda_ao[m] + lambda_ao[n]) * ds[(m, n)]
            });
        }
    }
    // The orthonormality in the transition charges.
    let y1 = -(x.transpose() * &l_oo);
    let y2 = (&l_ov * x.transpose()) * -0.5;
    let y3 = (&l_vo * x) * -0.5;
    overlap += sym(&(&cv * y1 * co.transpose()))
        + sym(&(&co * y2 * co.transpose()))
        + sym(&(&cv * y3 * cv.transpose()));

    // The charges of the ground state in the kernel and the Fock matrix.
    let scc_weights = match &input.scc {
        None => None,
        Some(scc) => {
            let shells = scc.kernel.nrows();
            let pi = mulliken_charges(&p, s, scc.shell_of_ao, shells);
            let mut kappa: Vec<f64> = (0..shells)
                .map(|a| -(0..shells).map(|b| scc.kernel[(a, b)] * pi[b]).sum::<f64>())
                .collect();
            if let Some(third) = &input.third {
                let qa = atom_charge(third);
                for (a, k) in kappa.iter_mut().enumerate() {
                    let atom = third.atom_of_site[a];
                    *k += 4.0 * third.gamma3[atom] * qa[atom] * qa[atom];
                }
            }
            let kappa_ao = per_ao(scc.shell_of_ao, &kappa);
            let k_oo = co.transpose() * mulliken_matrix(s, &kappa_ao) * &co;
            overlap += &co * (k_oo * 2.0) * co.transpose();
            let dens = scc.density;
            overlap -= Mat::from_fn(nao, nao, |m, n| {
                0.5 * (kappa_ao[m] + kappa_ao[n]) * dens[(m, n)]
            });
            Some(Mat::from_fn(shells, shells, |a, b| {
                -0.5 * (pi[a] * scc.charges[b] + scc.charges[a] * pi[b])
            }))
        }
    };
    let kernel = Mat::from_fn(sites, sites, |a, b| 2.0 * q[a] * q[b]);
    Ok(Terms {
        density: p,
        overlap,
        kernel,
        scc: scc_weights,
    })
}

/// Solve the symmetric positive definite system `apply(Z) = rhs` by conjugate gradients
/// preconditioned with `diagonal`.
fn conjugate_gradients(
    apply: &dyn Fn(&Mat<f64>) -> Mat<f64>,
    rhs: &Mat<f64>,
    diagonal: &Mat<f64>,
) -> MoptResult<Mat<f64>> {
    let dot = |a: &Mat<f64>, b: &Mat<f64>| -> f64 {
        (0..a.nrows())
            .map(|i| (0..a.ncols()).map(|j| a[(i, j)] * b[(i, j)]).sum::<f64>())
            .sum()
    };
    let precondition =
        |r: &Mat<f64>| Mat::from_fn(r.nrows(), r.ncols(), |i, j| r[(i, j)] / diagonal[(i, j)]);
    let norm_rhs = dot(rhs, rhs).sqrt();
    if norm_rhs == 0.0 {
        return Ok(Mat::zeros(rhs.nrows(), rhs.ncols()));
    }
    let mut z = precondition(rhs);
    let mut r = rhs - apply(&z);
    let mut y = precondition(&r);
    let mut p = y.clone();
    let mut ry = dot(&r, &y);
    for _ in 0..Z_ITERATIONS {
        if dot(&r, &r).sqrt() <= Z_TOLERANCE * norm_rhs {
            return Ok(z);
        }
        let ap = apply(&p);
        let pap = dot(&p, &ap);
        if pap.is_nan() || pap <= 0.0 {
            return Err(MoptError::calc(
                "excited-state gradient: the response of the ground state is not positive \
                 definite (an unstable closed-shell reference)",
            ));
        }
        let alpha = ry / pap;
        z += &p * alpha;
        r -= &ap * alpha;
        y = precondition(&r);
        let next = dot(&r, &y);
        p = &y + &p * (next / ry);
        ry = next;
    }
    if dot(&r, &r).sqrt() <= 1.0e-8 * norm_rhs {
        return Ok(z);
    }
    Err(MoptError::calc(format!(
        "excited-state gradient: the Z-vector did not converge in {Z_ITERATIONS} iterations"
    )))
}

/// The derivative of the damped Coulomb interaction `g = (R^alpha + eta^-alpha)^(-1/alpha)`
/// of sTDA with respect to the position of the first atom, for the atoms at `xa` and `xb`.
pub(crate) fn damped_coulomb_derivative(
    xa: [f64; 3],
    xb: [f64; 3],
    eta: f64,
    alpha: f64,
) -> [f64; 3] {
    let d: [f64; 3] = std::array::from_fn(|k| xa[k] - xb[k]);
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if r < 1.0e-12 {
        return [0.0; 3];
    }
    let base = r.powf(alpha) + eta.powf(-alpha);
    let dg_dr = -r.powf(alpha - 1.0) * base.powf(-1.0 / alpha - 1.0);
    std::array::from_fn(|k| dg_dr * d[k] / r)
}

/// `sum_(A != B) w_AB dg_AB/dx` of the damped Coulomb interaction of sTDA of the atoms at `x`
/// with the hardnesses `hardness`, added to `grad`.
pub(crate) fn add_damped_coulomb_gradient(
    x: &[[f64; 3]],
    hardness: &[f64],
    alpha: f64,
    w: &Mat<f64>,
    grad: &mut [f64],
) {
    let n = x.len();
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            let eta = 0.5 * (hardness[a] + hardness[b]);
            let dg = damped_coulomb_derivative(x[a], x[b], eta, alpha);
            for k in 0..3 {
                // d g_AB / d x_A, and d g_AB / d x_B = -d g_AB / d x_A.
                grad[3 * a + k] += w[(a, b)] * dg[k];
                grad[3 * b + k] -= w[(a, b)] * dg[k];
            }
        }
    }
}
