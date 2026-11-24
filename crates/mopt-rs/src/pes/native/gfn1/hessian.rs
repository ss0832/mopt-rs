//! The analytic Hessian of GFN1-xTB.
//!
//! The gradient of [`super::Gfn1Xtb::compute_in`] is `G_x(R; P, W, q)`: the density `P`, the
//! energy-weighted density `W` and the shell charges `q` enter it at their self-consistent
//! values. Its total derivative is
//!
//! ```text
//! H_xy = dG_x/dy |_(P, W, q)  +  tr(Ft^(x) P^y) - tr(S^(x) W^y) + sum_c (u^x_c - (K m^x)_c) q^y_c
//! ```
//!
//! - the first part holds `P`, `W` and `q` (second derivatives of the overlaps, the distance
//!   polynomial, the self energies through the coordination numbers, the Coulomb kernel, the
//!   point charges, and the cross term `-sum_a m^x_a u^y_a` of the shell potentials);
//! - `Ft^(x) = dH0/dx - S^(x) (v_mu + v_nu) / 2` is how `G_x` depends on `P`,
//!   `m^x_a = sum_(mu in a) (P S^(x))_mu,mu`, `u^x = d(gamma q + V_ext)/dx` at fixed charges,
//!   and `K = d v / d q` (`gamma` and `2 Gamma_A q_A` within an atom);
//! - the responses `P^y`, `W^y` and `q^y` solve the coupled-perturbed SCC equations in the
//!   space of the shell charges: with the uncoupled susceptibility `chi_ab = dq_a/dv_b` and the
//!   uncoupled charge response `q0^y` to the perturbation (the Fock derivative at fixed charges
//!   and `S^(y)`), `(1 - chi K) q^y = q0^y`; the density responses come from the Fock
//!   derivative with the potential change `K q^y` added ([`crate::pes::native::tb_response`],
//!   Fermi smearing with the chemical potentials of the two spins kept).
//!
//! The classical terms: the repulsion pair by pair, the halogen bond by hyper-dual numbers per
//! triple, D3(BJ) by its own Hessian. The derivation is mopt-rs's own, written from the energy
//! expression of [`super`]; the split into a fixed-density part and a charge-space response
//! follows the plan of gfn1-rs (the user's GPL-3.0-or-later code), whose code was not copied.
//! Checked against central differences of the analytic gradient in the tests.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::hamiltonian::{bohr, diff, dot, gamma_derivative, pair_scale};
use super::params::{
    self, Gfn1Element, CN_CUTOFF, REP_KEXP, XB_ACCEPTORS, XB_CUTOFF2, XB_DAMP, XB_HALOGENS, XB_RAD,
};
use super::{
    charge_hardness, to_charge, Gfn1Xtb, BOLTZMANN_HARTREE_PER_K, ELECTRONIC_TEMPERATURE_K,
};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_grad_hessian, HyperDual, Scalar};
use crate::math::linalg::{mul, DenseLu};
use crate::pes::dispersion::d3::D3Bj;
use crate::pes::native::fd_hessian::parallel_within_memory;
use crate::pes::native::gfn0::basis::{pair_overlap_gradient, pair_overlap_hessian};
use crate::pes::native::tb_response::OrbitalResponse;
use crate::pes::point_charges::PointCharges;

type M3 = [[f64; 3]; 3];

/// The second derivative with respect to `R_A` of `F(|R_A - R_B|)` with `F'` = `f1` and
/// `F''` = `f2`, `d = R_A - R_B`.
fn radial(d: [f64; 3], f1: f64, f2: f64) -> M3 {
    let r2 = dot(d, d);
    let r = r2.sqrt();
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let dd = d[i] * d[j] / r2;
            out[i][j] = f2 * dd + f1 / r * (if i == j { 1.0 } else { 0.0 } - dd);
        }
    }
    out
}

/// Add the block of a function of `R_A - R_B` to the Hessian.
fn add_pair(h: &mut Mat<f64>, a: usize, b: usize, block: &M3) {
    for i in 0..3 {
        for j in 0..3 {
            let v = block[i][j];
            h[(3 * a + i, 3 * a + j)] += v;
            h[(3 * b + i, 3 * b + j)] += v;
            h[(3 * a + i, 3 * b + j)] -= v;
            h[(3 * b + i, 3 * a + j)] -= v;
        }
    }
}

/// Everything of a shell pair on two atoms.
struct Pair {
    a: usize,
    b: usize,
    ia: usize,
    ib: usize,
    /// `S`, `dS/dR_A`, `d2S/dR_A2` of the AO block (row-major `nao_a x nao_b`).
    s: Vec<f64>,
    ds: Vec<[f64; 3]>,
    d2s: Vec<M3>,
    km: f64,
    poly: f64,
    dpoly: [f64; 3],
    d2poly: M3,
}

/// `d2 Pi / dR_A2` of the distance polynomial of shells `sa`, `sb` (see
/// [`super::hamiltonian::pair_scale`]).
fn poly_second(
    sa: &crate::pes::native::gfn0::basis::BasisShell,
    sb: &crate::pes::native::gfn0::basis::BasisShell,
    d: [f64; 3],
) -> M3 {
    let r = dot(d, d).sqrt();
    let x = (r / (sa.atomic_radius + sb.atomic_radius)).sqrt();
    let (ka, kb) = (sa.poly, sb.poly);
    let x1 = x / (2.0 * r);
    let x2 = -x / (4.0 * r * r);
    let lin = ka * (1.0 + kb * x) + kb * (1.0 + ka * x);
    let p1 = lin * x1;
    let p2 = 2.0 * ka * kb * x1 * x1 + lin * x2;
    radial(d, p1, p2)
}

/// A pair of atoms of the coordination numbers: `f(r)`, `f'(r)`, `f''(r)`.
struct CnPair {
    i: usize,
    j: usize,
    d: [f64; 3],
    f1: f64,
    f2: f64,
}

fn cn_pairs(el: &[&Gfn1Element], x: &[[f64; 3]]) -> Vec<CnPair> {
    let n = x.len();
    let rcov: Vec<f64> = el.iter().map(|e| 4.0 / 3.0 * bohr(e.cov_rad_d3)).collect();
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..i {
            let d = diff(x, i, j);
            let r2 = dot(d, d);
            if r2 > CN_CUTOFF * CN_CUTOFF {
                continue;
            }
            let r = r2.sqrt();
            let rc = rcov[i] + rcov[j];
            // f = 1 / (1 + exp(-g)), g = 16 (rc / r - 1).
            let g = 16.0 * (rc / r - 1.0);
            let f = 1.0 / (1.0 + (-g).exp());
            let g1 = -16.0 * rc / r2;
            let g2 = 32.0 * rc / (r2 * r);
            let s = f * (1.0 - f);
            out.push(CnPair {
                i,
                j,
                d,
                f1: s * g1,
                f2: s * (1.0 - 2.0 * f) * g1 * g1 + s * g2,
            });
        }
    }
    out
}

/// The halogen-bond energy of one triple (halogen `xx`, its neighbour `xb`, acceptor `xa`),
/// as [`super::classical::halogen_bond`].
fn xb_triple<T: Scalar>(xx: [T; 3], xb: [T; 3], xa: [T; 3], cc: f64, r0: f64) -> T {
    let sub = |p: [T; 3], q: [T; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let (dxa, dxb, dba) = (sub(xa, xx), sub(xb, xx), sub(xa, xb));
    let (d2ax, d2bx, d2ab) = (T::dot3(dxa, dxa), T::dot3(dxb, dxb), T::dot3(dba, dba));
    let rax = d2ax.sqrt();
    let term = (d2bx + d2ax - d2ab) / (d2bx * d2ax).sqrt();
    let base = T::from_f64(0.5) - T::from_f64(0.25) * term;
    let aterm = base.powi(6);
    let t14 = (T::from_f64(r0) / rax).powi(6);
    let num = t14 * t14 - T::from_f64(XB_DAMP) * t14;
    let den = T::one() + t14 * t14;
    aterm * T::from_f64(cc) * num / den
}

/// The Hessian of the classical terms (repulsion, halogen bond) added to `h`.
fn classical_hessian(el: &[&Gfn1Element], x: &[[f64; 3]], h: &mut Mat<f64>) {
    let n = x.len();
    for i in 0..n {
        for j in 0..i {
            let alpha = (el[i].rep_alpha * el[j].rep_alpha).sqrt();
            let zz = el[i].rep_zeff * el[j].rep_zeff;
            let d = diff(x, i, j);
            let r = dot(d, d).sqrt();
            let u = alpha * r.powf(REP_KEXP);
            let e = zz * (-u).exp() / r;
            let ku1 = REP_KEXP * u + 1.0;
            let f1 = -e * ku1 / r;
            let f2 = e * (ku1 * ku1 - REP_KEXP * REP_KEXP * u + ku1) / (r * r);
            add_pair(h, i, j, &radial(d, f1, f2));
        }
    }
    for xa in 0..n {
        if !XB_HALOGENS.contains(&el[xa].z) || el[xa].xb_strength == 0.0 {
            continue;
        }
        let Some(b) = (0..n).filter(|&m| m != xa).min_by(|&p, &q| {
            let dp = diff(x, p, xa);
            let dq = diff(x, q, xa);
            dot(dp, dp).total_cmp(&dot(dq, dq))
        }) else {
            continue;
        };
        for aa in 0..n {
            if !XB_ACCEPTORS.contains(&el[aa].z) {
                continue;
            }
            let dxa = diff(x, aa, xa);
            if dot(dxa, dxa) >= XB_CUTOFF2 {
                continue;
            }
            let cc = el[xa].xb_strength;
            let r0 = XB_RAD * bohr(el[xa].atomic_rad + el[aa].atomic_rad);
            let atoms = [xa, b, aa];
            let flat: Vec<f64> = atoms.iter().flat_map(|&k| x[k]).collect();
            let (_, _, hb) = eval_value_grad_hessian(&flat, None, |v: &[HyperDual]| {
                xb_triple(
                    [v[0], v[1], v[2]],
                    [v[3], v[4], v[5]],
                    [v[6], v[7], v[8]],
                    cc,
                    r0,
                )
            });
            for (p, &ap) in atoms.iter().enumerate() {
                for (q, &aq) in atoms.iter().enumerate() {
                    for i in 0..3 {
                        for j in 0..3 {
                            h[(3 * ap + i, 3 * aq + j)] += hb[(3 * p + i, 3 * q + j)];
                        }
                    }
                }
            }
        }
    }
}

/// `d2 gamma / dR_A2` of `gamma = (r^2 + 1/g^2)^(-1/2)` (`g` from the hardnesses).
fn gamma_second(eta_a: f64, eta_b: f64, d: [f64; 3]) -> M3 {
    let g = 2.0 / (1.0 / eta_a + 1.0 / eta_b);
    let t = dot(d, d) + 1.0 / (g * g);
    let t32 = 1.0 / (t * t.sqrt());
    let t52 = t32 / t;
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = 3.0 * t52 * d[i] * d[j] - if i == j { t32 } else { 0.0 };
        }
    }
    out
}

impl Gfn1Xtb {
    /// The analytic Hessian (Hartree/Bohr^2) of the atoms in the field of `charges` (held
    /// fixed).
    pub fn analytic_hessian_in(
        &self,
        sys: &MolecularSystem,
        charges: Option<&PointCharges>,
    ) -> MoptResult<Mat<f64>> {
        let scc = self.scc(sys, charges)?;
        let x = &scc.x;
        let n = x.len();
        let dof = 3 * n;
        let basis = &scc.basis;
        let ns = basis.shells.len();
        let nao = basis.len();
        let (p, w) = (&scc.elec.density, &scc.elec.weighted);
        let (q, v) = (&scc.q_shell, &scc.potential);
        let s_mat = &scc.s;
        let dlevel: Vec<f64> = basis.shells.iter().map(|s| s.level * s.kcn).collect();
        let shell_of: Vec<usize> = {
            let mut out = vec![0; nao];
            for (a, sh) in basis.shells.iter().enumerate() {
                for slot in &mut out[sh.first_ao..sh.first_ao + sh.nao] {
                    *slot = a;
                }
            }
            out
        };

        // Shell pairs on two atoms.
        let mut pairs = Vec::new();
        for a in 0..ns {
            for b in a + 1..ns {
                let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
                if sa.atom == sb.atom {
                    continue;
                }
                let d = diff(x, sa.atom, sb.atom);
                if basis.negligible(a, b, dot(d, d)) {
                    continue;
                }
                let ps = pair_scale(sa, sb, d);
                let mut s = Vec::with_capacity(sa.nao * sb.nao);
                for mu in sa.first_ao..sa.first_ao + sa.nao {
                    for nu in sb.first_ao..sb.first_ao + sb.nao {
                        s.push(s_mat[(mu, nu)]);
                    }
                }
                pairs.push(Pair {
                    a,
                    b,
                    ia: sa.atom,
                    ib: sb.atom,
                    s,
                    ds: pair_overlap_gradient(basis, a, b, x[sa.atom], x[sb.atom]),
                    d2s: pair_overlap_hessian(basis, a, b, x[sa.atom], x[sb.atom]),
                    km: ps.km,
                    poly: ps.poly,
                    dpoly: ps.dpoly,
                    d2poly: poly_second(sa, sb, d),
                });
            }
        }
        // Loop helper over the AO pairs of a shell pair.
        let block = |pr: &Pair| {
            let (sa, sb) = (&basis.shells[pr.a], &basis.shells[pr.b]);
            (sa.first_ao, sa.nao, sb.first_ao, sb.nao)
        };

        // Coordination numbers and their Jacobian J[A][x].
        let cnp = cn_pairs(&scc.el, x);
        let mut jac = Mat::<f64>::zeros(n, dof);
        for c in &cnp {
            let r = dot(c.d, c.d).sqrt();
            for k in 0..3 {
                let g = c.f1 * c.d[k] / r;
                for atom in [c.i, c.j] {
                    jac[(atom, 3 * c.i + k)] += g;
                    jac[(atom, 3 * c.j + k)] -= g;
                }
            }
        }

        let mut h = Mat::<f64>::zeros(dof, dof);
        // dE/dCN_A at fixed P (the weights of the CN second derivatives).
        let mut de_dcn = vec![0.0; n];
        for (a, sh) in basis.shells.iter().enumerate() {
            let diag: f64 = (sh.first_ao..sh.first_ao + sh.nao)
                .map(|mu| p[(mu, mu)])
                .sum();
            de_dcn[sh.atom] += diag * dlevel[a];
        }
        // Pair terms at fixed self energies and the CN cross terms.
        for pr in &pairs {
            let (fa, na, fb, nb) = block(pr);
            let hav = 0.5 * pr.km * (scc.levels[pr.a] + scc.levels[pr.b]);
            let vv = v[pr.a] + v[pr.b];
            let (mut t1, mut dt1, mut d2t1) = (0.0, [0.0; 3], [[0.0; 3]; 3]);
            let (mut dt2, mut d2t2) = ([0.0; 3], [[0.0; 3]; 3]);
            for r in 0..na {
                for c in 0..nb {
                    let (mu, nu) = (fa + r, fb + c);
                    let k = r * nb + c;
                    let pw = 2.0 * p[(mu, nu)];
                    let ww = 2.0 * (w[(mu, nu)] + 0.5 * vv * p[(mu, nu)]);
                    t1 += pw * pr.s[k];
                    for i in 0..3 {
                        dt1[i] += pw * pr.ds[k][i];
                        dt2[i] += ww * pr.ds[k][i];
                        for j in 0..3 {
                            d2t1[i][j] += pw * pr.d2s[k][i][j];
                            d2t2[i][j] += ww * pr.d2s[k][i][j];
                        }
                    }
                }
            }
            let mut bl = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    bl[i][j] = hav
                        * (pr.d2poly[i][j] * t1
                            + pr.dpoly[i] * dt1[j]
                            + dt1[i] * pr.dpoly[j]
                            + pr.poly * d2t1[i][j])
                        - d2t2[i][j];
                }
            }
            add_pair(&mut h, pr.ia, pr.ib, &bl);
            // CN: the pair energy is linear in the self energies.
            let half = 0.5 * pr.km;
            de_dcn[pr.ia] += half * pr.poly * t1 * dlevel[pr.a];
            de_dcn[pr.ib] += half * pr.poly * t1 * dlevel[pr.b];
            let gcn: [f64; 3] = [0, 1, 2].map(|k| half * (pr.dpoly[k] * t1 + pr.poly * dt1[k]));
            for y in 0..dof {
                let wy = dlevel[pr.a] * jac[(pr.ia, y)] + dlevel[pr.b] * jac[(pr.ib, y)];
                if wy == 0.0 {
                    continue;
                }
                for k in 0..3 {
                    let c = gcn[k] * wy;
                    h[(3 * pr.ia + k, y)] += c;
                    h[(3 * pr.ib + k, y)] -= c;
                    h[(y, 3 * pr.ia + k)] += c;
                    h[(y, 3 * pr.ib + k)] -= c;
                }
            }
        }
        // CN second derivatives.
        for c in &cnp {
            let bl = radial(c.d, c.f1, c.f2);
            let wsum = de_dcn[c.i] + de_dcn[c.j];
            let scaled = bl.map(|row| row.map(|e| e * wsum));
            add_pair(&mut h, c.i, c.j, &scaled);
        }
        // The Coulomb kernel at fixed charges.
        for a in 0..ns {
            for b in a + 1..ns {
                let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
                if sa.atom == sb.atom {
                    continue;
                }
                let d = diff(x, sa.atom, sb.atom);
                let qq = q[a] * q[b];
                let bl = gamma_second(scc.eta[a], scc.eta[b], d).map(|row| row.map(|e| e * qq));
                add_pair(&mut h, sa.atom, sb.atom, &bl);
            }
        }
        // u^x = d(gamma q + V_ext)/dx at fixed charges (shells by coordinates).
        let mut u = Mat::<f64>::zeros(ns, dof);
        for a in 0..ns {
            for (b, qb) in q.iter().enumerate() {
                let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
                if sa.atom == sb.atom {
                    continue;
                }
                let d = diff(x, sa.atom, sb.atom);
                let dg = gamma_derivative(scc.eta[a], scc.eta[b], d);
                for k in 0..3 {
                    u[(a, 3 * sa.atom + k)] += dg[k] * qb;
                    u[(a, 3 * sb.atom + k)] -= dg[k] * qb;
                }
            }
        }
        if let Some(pc) = charges {
            let eta_k = charge_hardness(pc)?;
            for (a, sh) in basis.shells.iter().enumerate() {
                for (kk, &eta) in eta_k.iter().enumerate() {
                    let d = to_charge(x[sh.atom], pc.positions_bohr[kk]);
                    let dg = gamma_derivative(scc.eta[a], eta, d);
                    for k in 0..3 {
                        u[(a, 3 * sh.atom + k)] += dg[k] * pc.charges[kk];
                    }
                    let bl = gamma_second(scc.eta[a], eta, d);
                    let f = q[a] * pc.charges[kk];
                    for i in 0..3 {
                        for j in 0..3 {
                            h[(3 * sh.atom + i, 3 * sh.atom + j)] += f * bl[i][j];
                        }
                    }
                }
            }
        }
        // m^x_a = sum_(mu in a) (P S^(x))_mu,mu.
        let mut m = Mat::<f64>::zeros(ns, dof);
        for pr in &pairs {
            let (fa, na, fb, nb) = block(pr);
            for k in 0..3 {
                let mut t = 0.0;
                for r in 0..na {
                    for c in 0..nb {
                        t += p[(fa + r, fb + c)] * pr.ds[r * nb + c][k];
                    }
                }
                for sh in [pr.a, pr.b] {
                    m[(sh, 3 * pr.ia + k)] += t;
                    m[(sh, 3 * pr.ib + k)] -= t;
                }
            }
        }
        // The cross term of the shell potentials: -sum_a m^x_a u^y_a.
        let cross = mul(m.transpose(), u.as_ref());
        for xx in 0..dof {
            for y in 0..dof {
                h[(xx, y)] -= cross[(xx, y)];
            }
        }

        // The response.
        let kt = BOLTZMANN_HARTREE_PER_K * ELECTRONIC_TEMPERATURE_K;
        let [fa_occ, fb_occ] = &scc.elec.spin_occupations;
        let orb = OrbitalResponse::new(
            scc.elec.coefficients.clone(),
            scc.elec.eigenvalues.clone(),
            [fa_occ.as_slice(), fb_occ.as_slice()],
            kt,
        );
        let atom_of_shell: Vec<usize> = basis.shells.iter().map(|s| s.atom).collect();
        let gamma3: Vec<f64> = scc.el.iter().map(|e| e.gamma3).collect();
        // K = dv/dq.
        let kmat = Mat::from_fn(ns, ns, |a, b| {
            let (ia, ib) = (atom_of_shell[a], atom_of_shell[b]);
            scc.gamma[(a, b)]
                + if ia == ib {
                    2.0 * gamma3[ia] * scc.q_atom[ia]
                } else {
                    0.0
                }
        });
        // -1/2 S (dv_mu + dv_nu) of a change of the shell potentials.
        let potential_fock = |dv: &[f64]| {
            Mat::from_fn(nao, nao, |i, j| {
                -0.5 * s_mat[(i, j)] * (dv[shell_of[i]] + dv[shell_of[j]])
            })
        };
        // The diagonal of P1 S without the product: (P1 S)_ii = sum_j P1_ij S_ji.
        let mulliken = |p1: &Mat<f64>| -> Vec<f64> {
            let mut out = vec![0.0; ns];
            for i in 0..nao {
                let mut d = 0.0;
                for j in 0..nao {
                    d += p1[(i, j)] * s_mat[(j, i)];
                }
                out[shell_of[i]] -= d;
            }
            out
        };
        // The uncoupled susceptibility chi_ab = dq_a/dv_b, one column per shell potential (the
        // columns in parallel, each in its own place).
        // Each perturbation holds about a dozen matrices of the atomic orbitals at once.
        let per_task = crate::core_infra::memory::f64_bytes(12.0 * (nao * nao) as f64);
        let chi_columns: Vec<Vec<f64>> = parallel_within_memory(ns, per_task, |b| {
            let mut e = vec![0.0; ns];
            e[b] = 1.0;
            let r = orb.respond_density(&orb.to_orbitals(&potential_fock(&e)), None);
            mulliken(&r.p)
        });
        let chi = Mat::from_fn(ns, ns, |a, b| chi_columns[b][a]);
        let dielectric = Mat::<f64>::identity(ns, ns) - mul(chi.as_ref(), kmat.as_ref());
        let lu = DenseLu::new(dielectric.as_ref())?;
        // dq_c-coefficients of G_x: u^x_c - (K m^x)_c.
        let gq = &u - mul(kmat.as_ref(), m.as_ref());

        // The derivative matrices of one coordinate.
        let derivatives = |y: usize| -> (Mat<f64>, Mat<f64>, Mat<f64>) {
            // S^(y), Ft^(y) (no potential-change term), and the full Fock derivative at fixed
            // charges F^(y) = Ft^(y) - S (u^y + u^y) / 2.
            let mut sy = Mat::<f64>::zeros(nao, nao);
            let mut fty = Mat::<f64>::zeros(nao, nao);
            let (atom, k) = (y / 3, y % 3);
            let dh: Vec<f64> = (0..ns)
                .map(|a| dlevel[a] * jac[(atom_of_shell[a], y)])
                .collect();
            for (a, sh) in basis.shells.iter().enumerate() {
                for mu in sh.first_ao..sh.first_ao + sh.nao {
                    fty[(mu, mu)] += dh[a];
                }
            }
            for pr in &pairs {
                let (fa, na, fb, nb) = block(pr);
                let hav = 0.5 * pr.km * (scc.levels[pr.a] + scc.levels[pr.b]);
                let vv = v[pr.a] + v[pr.b];
                let sign = if atom == pr.ia {
                    1.0
                } else if atom == pr.ib {
                    -1.0
                } else {
                    0.0
                };
                let dhav = 0.5 * pr.km * (dh[pr.a] + dh[pr.b]);
                for r in 0..na {
                    for c in 0..nb {
                        let kk = r * nb + c;
                        let mut f = dhav * pr.poly * pr.s[kk];
                        let mut sv = 0.0;
                        if sign != 0.0 {
                            let dsk = pr.ds[kk][k];
                            sv = sign * dsk;
                            f += sign
                                * (hav * (pr.poly * dsk + pr.dpoly[k] * pr.s[kk]) - 0.5 * vv * dsk);
                        }
                        let (mu, nu) = (fa + r, fb + c);
                        fty[(mu, nu)] += f;
                        fty[(nu, mu)] += f;
                        sy[(mu, nu)] += sv;
                        sy[(nu, mu)] += sv;
                    }
                }
            }
            let uy: Vec<f64> = (0..ns).map(|a| u[(a, y)]).collect();
            let fy = &fty + potential_fock(&uy);
            (sy, fty, fy)
        };

        let dvq = |qy: &[f64]| -> Vec<f64> {
            (0..ns)
                .map(|a| (0..ns).map(|b| kmat[(a, b)] * qy[b]).sum())
                .collect()
        };
        // One column of the response per coordinate: independent of the others, so the
        // columns run in parallel, each written to its own place.
        let response_column = |y: usize| -> Vec<f64> {
            let (sy, _, fy) = derivatives(y);
            let f1 = orb.to_orbitals(&fy);
            let s1 = orb.to_orbitals(&sy);
            let r0 = orb.respond_density(&f1, Some(&s1));
            let mut q0 = mulliken(&r0.p);
            for (a, val) in q0.iter_mut().enumerate() {
                *val -= m[(a, y)];
            }
            let rhs = faer::Col::from_fn(ns, |a| q0[a]);
            let sol = lu.solve_col(&rhs);
            let qy: Vec<f64> = (0..ns).map(|a| sol[a]).collect();
            let ftot = &f1 + orb.to_orbitals(&potential_fock(&dvq(&qy)));
            let r = orb.respond(&ftot, Some(&s1));
            // Contract with every x: tr(Ft^(x) P^y) - tr(S^(x) W^y) + gq . q^y.
            let mut col = vec![0.0; dof];
            // The diagonal self energies (through the CN) and the pair blocks.
            let mut dlev = vec![0.0; ns];
            for (a, sh) in basis.shells.iter().enumerate() {
                dlev[a] += (sh.first_ao..sh.first_ao + sh.nao)
                    .map(|mu| r.p[(mu, mu)])
                    .sum::<f64>();
            }
            for pr in &pairs {
                let (fa, na, fb, nb) = block(pr);
                let hav = 0.5 * pr.km * (scc.levels[pr.a] + scc.levels[pr.b]);
                let vv = v[pr.a] + v[pr.b];
                let mut lp = [0.0; 3];
                let mut lw = [0.0; 3];
                let mut ps = 0.0;
                for rr in 0..na {
                    for c in 0..nb {
                        let kk = rr * nb + c;
                        let (mu, nu) = (fa + rr, fb + c);
                        let p2 = 2.0 * r.p[(mu, nu)];
                        let w2 = 2.0 * r.w[(mu, nu)];
                        ps += p2 * pr.s[kk];
                        for kx in 0..3 {
                            lp[kx] += p2
                                * (hav * (pr.poly * pr.ds[kk][kx] + pr.dpoly[kx] * pr.s[kk])
                                    - 0.5 * vv * pr.ds[kk][kx]);
                            lw[kx] += w2 * pr.ds[kk][kx];
                        }
                    }
                }
                for kx in 0..3 {
                    col[3 * pr.ia + kx] += lp[kx] - lw[kx];
                    col[3 * pr.ib + kx] -= lp[kx] - lw[kx];
                }
                let half = 0.5 * pr.km * pr.poly * ps;
                dlev[pr.a] += half;
                dlev[pr.b] += half;
            }
            for xx in 0..dof {
                let mut cn = 0.0;
                for a in 0..ns {
                    cn += dlevel[a] * jac[(atom_of_shell[a], xx)] * dlev[a];
                }
                let qterm: f64 = (0..ns).map(|c| gq[(c, xx)] * qy[c]).sum();
                let total = col[xx] + cn + qterm;
                col[xx] = total;
            }
            col
        };
        let columns: Vec<Vec<f64>> = parallel_within_memory(dof, per_task, response_column);
        let hresp = Mat::from_fn(dof, dof, |xx, y| columns[y][xx]);
        h += &hresp;

        classical_hessian(&scc.el, x, &mut h);
        let d3 = D3Bj {
            s6: params::D3_S6,
            s8: params::D3_S8,
            a1: params::D3_A1,
            a2: params::D3_A2,
            cutoff: params::D3_CUTOFF,
            cn_cutoff: params::CN_CUTOFF,
        };
        let disp = d3.evaluate(sys, true)?;
        let hd = disp
            .hessian_bohr
            .ok_or_else(|| MoptError::calc("GFN1-xTB: D3 gave no Hessian"))?;
        h += &hd;
        Ok(Mat::from_fn(dof, dof, |i, j| 0.5 * (h[(i, j)] + h[(j, i)])))
    }
}
