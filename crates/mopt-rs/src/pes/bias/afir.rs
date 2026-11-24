//! Artificial force induced reaction (AFIR) term, for isolated and periodic systems.
//!
//! Maeda et al. (Chem. Rec. 2016, 16, 2232; J. Comput. Chem. 2018, 39, 233; WIREs Comput. Mol.
//! Sci. 2021, 11, e1538): fragments A and B are pushed together (`gamma > 0`) or pulled apart
//! (`gamma < 0`) by
//!
//! `E = alpha sum_ij w_ij r_ij / sum_ij w_ij`, `w_ij = ((R_i + R_j) / r_ij)^p` (`p = 6`),
//!
//! over `i` in A and `j` in B with covalent radii `R`, and
//! `alpha = gamma / ((2^(-1/6) - (1 + sqrt(1 + |gamma| / eps))^(-1/6)) R0)` with
//! `R0 = 3.8164` Angstrom and `eps = 1.0061` kJ/mol, so that two argon atoms colliding with energy
//! `gamma` just overcome their Lennard-Jones repulsion. This reproduces MultiOptPy's
//! `AFIRPotential`.
//!
//! Under periodic boundary conditions the distances are minimum-image distances, as in the
//! periodic applications of AFIR (Takagi et al., Phys. Rev. B 2017, 95, 184110, atoms in the unit
//! cell; Sugiyama et al., Phys. Chem. Chem. Phys. 2019, 21, 14366, adsorbates on a frozen slab).
//! The cell itself is not a variable here; variable-cell AFIR (forces on the translation vectors)
//! belongs to variable-cell optimization.
//!
//! The gradient and Hessian are exact: the energy is a function of the pair distances, whose
//! derivatives are combined with those of the (minimum-image) distances.
//!
//! MultiOptPy: `Potential/AFIR_potential.py:AFIRPotential`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::{AtomId, EvalOutput, MolecularSystem, Quantity, Units};
use crate::error::{MoptError, MoptResult};
use crate::parameters::covalent::covalent_radius_bohr;
use crate::pes::term::{Frame, History, PesTerm, TermContext};

/// Exponent `p` of the weights.
pub const AFIR_P: f64 = 6.0;
/// `R0` of the argon-argon Lennard-Jones model (Angstrom).
pub const AFIR_R0_ANGSTROM: f64 = 3.8164;
/// `eps` of the argon-argon Lennard-Jones model (kJ/mol).
pub const AFIR_EPS_KJ_MOL: f64 = 1.0061;

/// `alpha` (Hartree/Bohr) for the collision energy `gamma` (kJ/mol).
pub fn afir_alpha(gamma_kj_mol: f64) -> f64 {
    if gamma_kj_mol == 0.0 {
        return 0.0;
    }
    let g = gamma_kj_mol * Units::KJ_MOL_TO_HARTREE;
    let eps = AFIR_EPS_KJ_MOL * Units::KJ_MOL_TO_HARTREE;
    let r0 = AFIR_R0_ANGSTROM * Units::ANGSTROM_TO_BOHR;
    g / ((2.0_f64.powf(-1.0 / 6.0) - (1.0 + (1.0 + g.abs() / eps).sqrt()).powf(-1.0 / 6.0)) * r0)
}

/// AFIR term between two fragments.
#[derive(Debug, Clone)]
pub struct Afir {
    /// Collision energy parameter `gamma` (kJ/mol).
    pub gamma_kj_mol: f64,
    /// Fragment A.
    pub fragment_a: Vec<AtomId>,
    /// Fragment B.
    pub fragment_b: Vec<AtomId>,
    /// Final `gamma` of a linear ramp over [`AFIR_RAMP_ITERATIONS`] iterations (MultiOptPy's
    /// `-ma gamma1,gamma2 ...`); constant `gamma` when `None`.
    pub gamma_final_kj_mol: Option<f64>,
}

/// Length of MultiOptPy's linear parameter ramp (`gradually_change_param`).
pub const AFIR_RAMP_ITERATIONS: f64 = 300.0;

impl Afir {
    /// Term pushing (`gamma > 0`) or pulling (`gamma < 0`) fragments A and B.
    pub fn new(
        gamma_kj_mol: f64,
        fragment_a: Vec<AtomId>,
        fragment_b: Vec<AtomId>,
    ) -> MoptResult<Self> {
        if fragment_a.is_empty() || fragment_b.is_empty() {
            return Err(MoptError::config("AFIR needs two non-empty fragments"));
        }
        if fragment_a.iter().any(|a| fragment_b.contains(a)) {
            return Err(MoptError::config("AFIR fragments must not share atoms"));
        }
        if !gamma_kj_mol.is_finite() {
            return Err(MoptError::config("AFIR gamma must be finite"));
        }
        Ok(Self {
            gamma_kj_mol,
            fragment_a,
            fragment_b,
            gamma_final_kj_mol: None,
        })
    }

    /// Ramp `gamma` linearly to `gamma_final` over [`AFIR_RAMP_ITERATIONS`] iterations.
    #[must_use]
    pub fn ramp_to(mut self, gamma_final_kj_mol: f64) -> Self {
        self.gamma_final_kj_mol = Some(gamma_final_kj_mol);
        self
    }

    /// `gamma` at optimizer iteration `iteration`.
    pub fn gamma_at(&self, iteration: usize) -> f64 {
        let (g1, Some(g2)) = (self.gamma_kj_mol, self.gamma_final_kj_mol) else {
            return self.gamma_kj_mol;
        };
        let g = g1 + (g2 - g1) / AFIR_RAMP_ITERATIONS * iteration as f64;
        if g1 < g2 {
            g.min(g2)
        } else {
            g.max(g2)
        }
    }
}

/// One pair `(i, j)` with its minimum-image displacement `r_i - r_j`.
struct Pair {
    i: usize,
    j: usize,
    d: [f64; 3],
    r: f64,
    w: f64,
}

impl PesTerm for Afir {
    fn name(&self) -> &str {
        "afir"
    }

    /// A function of the (minimum-image) pair distances.
    fn frame(&self) -> Frame {
        Frame::Invariant
    }

    /// Static, or ramped over the iterations with `gamma_final_kj_mol`.
    fn history(&self) -> History {
        if self.gamma_final_kj_mol.is_some() {
            History::IterationRamp
        } else {
            History::Static
        }
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn analytic_stress(&self) -> bool {
        true
    }

    fn atoms(&self) -> Option<Vec<AtomId>> {
        Some(
            self.fragment_a
                .iter()
                .chain(&self.fragment_b)
                .copied()
                .collect(),
        )
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let n = sys.n_atoms();
        let dof = sys.dof();
        let alpha = afir_alpha(self.gamma_at(ctx.iteration));
        let mut pairs = Vec::with_capacity(self.fragment_a.len() * self.fragment_b.len());
        for &a in &self.fragment_a {
            let a = a.check(n)?;
            for &b in &self.fragment_b {
                let b = b.check(n)?;
                let d = sys.delta(a, b).value();
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if r < 1e-8 {
                    return Err(MoptError::calc("AFIR: coincident atoms"));
                }
                let rr =
                    covalent_radius_bohr(sys.element(a)) + covalent_radius_bohr(sys.element(b));
                pairs.push(Pair {
                    i: a.index(),
                    j: b.index(),
                    d,
                    r,
                    w: (rr / r).powf(AFIR_P),
                });
            }
        }
        // E = alpha f / g with f = sum w r, g = sum w.
        let p = AFIR_P;
        let f: f64 = pairs.iter().map(|q| q.w * q.r).sum();
        let g: f64 = pairs.iter().map(|q| q.w).sum();
        let energy = alpha * f / g;
        // First derivatives with respect to the pair distances.
        let fk: Vec<f64> = pairs.iter().map(|q| (1.0 - p) * q.w).collect();
        let gk: Vec<f64> = pairs.iter().map(|q| -p * q.w / q.r).collect();
        let ek: Vec<f64> = (0..pairs.len())
            .map(|k| alpha * (fk[k] * g - f * gk[k]) / (g * g))
            .collect();

        let mut grad = Col::zeros(dof);
        for (k, q) in pairs.iter().enumerate() {
            for c in 0..3 {
                let v = ek[k] * q.d[c] / q.r;
                grad[3 * q.i + c] += v;
                grad[3 * q.j + c] -= v;
            }
        }

        // Second derivatives in pair-distance space (only the diagonal of f_kl, g_kl):
        // E_kl = alpha [ (f_kl g + f_k g_l - g_k f_l - f g_kl) / g^2
        //                - 2 (f_k g - f g_k) g_l / g^3 ].
        let fkk: Vec<f64> = pairs.iter().map(|q| (1.0 - p) * (-p * q.w / q.r)).collect();
        let gkk: Vec<f64> = pairs
            .iter()
            .map(|q| p * (p + 1.0) * q.w / (q.r * q.r))
            .collect();
        let second = |k: usize, l: usize| -> f64 {
            let mut e = alpha
                * ((fk[k] * gk[l] - gk[k] * fk[l]) / (g * g)
                    - 2.0 * (fk[k] * g - f * gk[k]) * gk[l] / (g * g * g));
            if k == l {
                e += alpha * (fkk[k] * g - f * gkk[k]) / (g * g);
            }
            e
        };
        let unit: Vec<[f64; 3]> = pairs.iter().map(|q| q.d.map(|x| x / q.r)).collect();
        let hessian = need_hessian.then(|| {
            let mut h = Mat::zeros(dof, dof);
            let m = pairs.len();
            let mut add =
                |a: usize, b: usize, sa: f64, sb: f64, ua: &[f64; 3], ub: &[f64; 3], w: f64| {
                    for x in 0..3 {
                        for y in 0..3 {
                            h[(3 * a + x, 3 * b + y)] += w * sa * sb * ua[x] * ub[y];
                        }
                    }
                };
            for k in 0..m {
                for l in 0..m {
                    let e = second(k, l);
                    if e == 0.0 {
                        continue;
                    }
                    let (qk, ql) = (&pairs[k], &pairs[l]);
                    let (uk, ul) = (&unit[k], &unit[l]);
                    // grad r_k = (+u_k on i_k, -u_k on j_k).
                    add(qk.i, ql.i, 1.0, 1.0, uk, ul, e);
                    add(qk.i, ql.j, 1.0, -1.0, uk, ul, e);
                    add(qk.j, ql.i, -1.0, 1.0, uk, ul, e);
                    add(qk.j, ql.j, -1.0, -1.0, uk, ul, e);
                }
            }
            // sum_k E_k Hess r_k, Hess r = (I - u u^T) / r in the [[+, -], [-, +]] pattern.
            for (k, q) in pairs.iter().enumerate() {
                let u = &unit[k];
                for x in 0..3 {
                    for y in 0..3 {
                        let v = ek[k] * ((if x == y { 1.0 } else { 0.0 }) - u[x] * u[y]) / q.r;
                        h[(3 * q.i + x, 3 * q.i + y)] += v;
                        h[(3 * q.j + x, 3 * q.j + y)] += v;
                        h[(3 * q.i + x, 3 * q.j + y)] -= v;
                        h[(3 * q.j + x, 3 * q.i + y)] -= v;
                    }
                }
            }
            h
        });
        // In a cell: the virial of the pair distances. Under a strain `1 + e` the
        // minimum-image vector `d` of a pair goes to `(1 + e) d`, so `dr/de_ab = d_a d_b / r`.
        let stress = sys.active_cell().map(|cell| {
            let volume = cell.volume_bohr3();
            let mut stress = [[0.0; 3]; 3];
            for (k, q) in pairs.iter().enumerate() {
                for (a, row) in stress.iter_mut().enumerate() {
                    for (b, value) in row.iter_mut().enumerate() {
                        *value += ek[k] * q.d[a] * q.d[b] / (q.r * volume);
                    }
                }
            }
            stress
        });
        // With a Hessian in a cell, the second derivatives along the strain: those of a
        // function of the pair distances (`d = r_i - r_j` runs from `j` to `i`).
        let strain_hessian = (need_hessian && sys.active_cell().is_some()).then(|| {
            let radial: Vec<(Option<(usize, usize)>, f64, [f64; 3])> = pairs
                .iter()
                .zip(&unit)
                .map(|(q, u)| (Some((q.j, q.i)), q.r, *u))
                .collect();
            let mut sh = crate::domain::StrainHessian::zeros(dof);
            sh.add_radial_function(&radial, &ek, &second);
            sh
        });
        Ok(EvalOutput {
            energy: Quantity::new(energy),
            gradient_bohr: grad,
            hessian_bohr: hessian,
            orbitals: None,
            stress,
            strain_hessian,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, PeriodicCell, SpinCharge};

    fn ortho(l: [f64; 3]) -> PeriodicCell {
        PeriodicCell::orthorhombic(
            Quantity::<Angstrom>::new(l[0]),
            Quantity::<Angstrom>::new(l[1]),
            Quantity::<Angstrom>::new(l[2]),
        )
        .unwrap()
    }

    fn ctx(sys: &MolecularSystem) -> TermContext<'_> {
        TermContext::new(0, sys, None)
    }

    fn system(pos: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
        let els = [
            Element::C,
            Element::H,
            Element::O,
            Element::N,
            Element::H,
            Element::C,
        ];
        MolecularSystem::new(
            els[..pos.len()].to_vec(),
            &positions::<Angstrom>(pos),
            SpinCharge::default(),
            cell,
        )
        .unwrap()
    }

    fn ids(v: &[usize]) -> Vec<AtomId> {
        v.iter().map(|&i| AtomId::new(i)).collect()
    }

    #[test]
    fn the_stress_in_a_cell_is_the_strain_derivative_of_the_energy() {
        // Keep each minimum-image pair away from a tie: the strain derivative is
        // undefined where shear switches between two equally near images.
        let sys = system(
            &[
                [0.2, 0.3, 0.1],
                [1.1, 0.2, 0.4],
                [5.6, 0.5, 0.3],
                [4.9, 1.2, 0.6],
                [2.5, 2.9, 3.0],
                [3.3, 3.5, 2.2],
            ],
            Some(ortho([6.0, 5.5, 5.0])),
        );
        let t = Afir::new(150.0, ids(&[0, 1, 4]), ids(&[2, 3, 5])).unwrap();
        let out = t.evaluate(&sys, &ctx(&sys), false).unwrap();
        let analytic = out.stress.unwrap();
        let numeric =
            crate::pes::stress::numerical_term_stress(&t, &sys, &ctx(&sys), 1e-5).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (analytic[a][b] - numeric[a][b]).abs() < 1e-9,
                    "{a}{b}: {} {}",
                    analytic[a][b],
                    numeric[a][b]
                );
            }
        }
        let analytic = t
            .evaluate(&sys, &ctx(&sys), true)
            .unwrap()
            .strain_hessian
            .unwrap();
        let numeric = crate::pes::stress::numerical_strain_hessian(&sys, 2e-4, &|s| {
            let out = t.evaluate(s, &ctx(s), false)?;
            Ok((out.energy.value(), out.gradient_bohr))
        })
        .unwrap();
        let scale = numeric
            .strain
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        for (a, b) in analytic
            .strain
            .iter()
            .flatten()
            .zip(numeric.strain.iter().flatten())
        {
            assert!((a - b).abs() < 1e-5 * scale, "{a} {b}");
        }
        for i in 0..sys.dof() {
            for k in 0..9 {
                let (a, b) = (analytic.mixed[(i, k)], numeric.mixed[(i, k)]);
                assert!((a - b).abs() < 1e-7, "({i}, {k}): {a} {b}");
            }
        }
        // Without a cell there is no stress.
        let free = system(&[[0.0, 0.0, 0.0], [1.7, 0.4, 0.0]], None);
        let t = Afir::new(100.0, ids(&[0]), ids(&[1])).unwrap();
        assert!(t
            .evaluate(&free, &ctx(&free), false)
            .unwrap()
            .stress
            .is_none());
    }

    #[test]
    fn alpha_and_single_pair_energy() {
        // gamma = 100 kJ/mol: alpha from the Ar-Ar model, computed independently.
        let g: f64 = 100.0 * Units::KJ_MOL_TO_HARTREE;
        let eps: f64 = 1.0061 * Units::KJ_MOL_TO_HARTREE;
        let r0 = 3.8164 * Units::ANGSTROM_TO_BOHR;
        let expected =
            g / ((2f64.powf(-1.0 / 6.0) - (1.0 + (1.0 + g / eps).sqrt()).powf(-1.0 / 6.0)) * r0);
        assert!((afir_alpha(100.0) - expected).abs() < 1e-15);
        assert!(afir_alpha(-100.0) < 0.0);
        // One pair: E = alpha r.
        let sys = system(&[[0.0, 0.0, 0.0], [1.7, 0.4, 0.0]], None);
        let t = Afir::new(100.0, ids(&[0]), ids(&[1])).unwrap();
        let e = t.evaluate(&sys, &ctx(&sys), false).unwrap().energy.value();
        let r = sys.distance(AtomId::new(0), AtomId::new(1)).value();
        assert!((e - afir_alpha(100.0) * r).abs() < 1e-14);
    }

    fn check_fd(sys: &MolecularSystem, t: &Afir) {
        let out = t.evaluate(sys, &ctx(sys), true).unwrap();
        let h = out.hessian_bohr.clone().unwrap();
        let x = sys.coords_flat_bohr();
        let step = 1e-5;
        for k in 0..x.nrows() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += step;
            xm[k] -= step;
            let sp = sys.with_flat_coords_bohr(&xp).unwrap();
            let sm = sys.with_flat_coords_bohr(&xm).unwrap();
            let op = t.evaluate(&sp, &ctx(&sp), false).unwrap();
            let om = t.evaluate(&sm, &ctx(&sm), false).unwrap();
            let fd = (op.energy.value() - om.energy.value()) / (2.0 * step);
            assert!(
                (fd - out.gradient_bohr[k]).abs() < 1e-8,
                "g{k}: {fd} vs {}",
                out.gradient_bohr[k]
            );
            for l in 0..x.nrows() {
                let fd = (op.gradient_bohr[l] - om.gradient_bohr[l]) / (2.0 * step);
                assert!(
                    (fd - h[(l, k)]).abs() < 1e-7,
                    "H({l},{k}): {fd} vs {}",
                    h[(l, k)]
                );
            }
        }
    }

    #[test]
    fn exact_derivatives_isolated_and_periodic() {
        let pos = [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [-0.6, 1.0, 0.3],
            [3.2, 0.4, 0.5],
            [3.9, 1.3, 0.2],
            [3.0, -1.1, 1.0],
        ];
        let t = Afir::new(150.0, ids(&[0, 1, 2]), ids(&[3, 4, 5])).unwrap();
        check_fd(&system(&pos, None), &t);
        let cell = ortho([4.5, 6.0, 7.0]);
        check_fd(&system(&pos, Some(cell)), &t);
        let pull = Afir::new(-80.0, ids(&[0, 2]), ids(&[4])).unwrap();
        check_fd(&system(&pos, None), &pull);
    }

    #[test]
    fn gamma_ramp_like_multioptpy() {
        let t = Afir::new(100.0, ids(&[0]), ids(&[1]))
            .unwrap()
            .ramp_to(400.0);
        assert!((t.gamma_at(0) - 100.0).abs() < 1e-12);
        assert!((t.gamma_at(150) - 250.0).abs() < 1e-12);
        assert!((t.gamma_at(1000) - 400.0).abs() < 1e-12);
        let down = Afir::new(300.0, ids(&[0]), ids(&[1])).unwrap().ramp_to(0.0);
        assert!((down.gamma_at(600) - 0.0).abs() < 1e-12);
    }

    fn cell(axes: [bool; 3]) -> PeriodicCell {
        // Skewed lattice vectors so that the minimum image is not axis-aligned.
        PeriodicCell::from_lattice_vectors(
            crate::domain::Vector3::<Angstrom>::new([5.0, 0.0, 0.0]),
            crate::domain::Vector3::<Angstrom>::new([1.2, 5.5, 0.0]),
            crate::domain::Vector3::<Angstrom>::new([0.4, -0.6, 6.0]),
            axes,
        )
        .unwrap()
    }

    #[test]
    fn isolated_1d_2d_and_3d_cells_pass_the_term_checks() {
        use crate::pes::term_check::{check_term, energy_change_on_shift};
        // Atoms near the cell faces so that minimum images matter along every periodic axis.
        let pos = [
            [0.2, 0.3, 0.4],
            [1.1, 0.5, 5.6],
            [4.6, 5.2, 0.6],
            [4.4, 0.1, 5.1],
            [0.5, 5.0, 5.4],
            [2.4, 2.7, 3.0],
        ];
        let t = Afir::new(150.0, ids(&[0, 1, 2]), ids(&[3, 4, 5])).unwrap();
        for axes in [
            None,
            Some([true, false, false]),
            Some([true, true, false]),
            Some([true, true, true]),
        ] {
            let sys = system(&pos, axes.map(cell));
            let c = check_term(&t, &sys, 1e-5).unwrap();
            assert!(c.passes(1e-6), "{axes:?}: {c:?}");
            // Moving an atom by the third lattice vector changes the energy unless that axis
            // is periodic.
            let v = cell([true; 3]).lattice_vector(2).value();
            let de = energy_change_on_shift(&t, &sys, AtomId::new(0), v).unwrap();
            let periodic_c = axes.is_some_and(|a| a[2]);
            assert_eq!(de.abs() < 1e-12, periodic_c, "{axes:?}: {de}");
        }
    }

    #[test]
    fn periodic_distances_use_the_minimum_image() {
        // Two atoms across the cell boundary: the minimum-image distance is 1.0 Angstrom.
        let cell = ortho([8.0, 8.0, 8.0]);
        let near = system(&[[0.5, 0.0, 0.0], [7.5, 0.0, 0.0]], Some(cell.clone()));
        let t = Afir::new(100.0, ids(&[0]), ids(&[1])).unwrap();
        let e = t
            .evaluate(&near, &ctx(&near), false)
            .unwrap()
            .energy
            .value();
        assert!((e - afir_alpha(100.0) * 1.0 * Units::ANGSTROM_TO_BOHR).abs() < 1e-12);
        // Translating an atom by a lattice vector changes nothing.
        let shifted = system(&[[0.5, 0.0, 0.0], [-0.5, 8.0, 0.0]], Some(cell));
        let e2 = t
            .evaluate(&shifted, &ctx(&shifted), false)
            .unwrap()
            .energy
            .value();
        assert!((e - e2).abs() < 1e-12);
    }
}
