//! Lennard-Jones potential with UFF noble-gas parameters (MultiOptPy
//! `Calculator/lj_calculation_tools.py`).
//!
//! `sigma = x_i / 2^(1/6)`, `epsilon = D_i` (kcal/mol), Lorentz-Berthelot combination.
//!
//! [`LennardJones`] sums over all pairs, as MultiOptPy does; in a periodic cell it takes the
//! nearest image of each pair, so the energy has a kink where a pair passes half the cell,
//! and time grows with the square of the number of atoms. [`TruncatedLennardJones`] takes
//! the pairs (all images) within a cutoff from a linked-cell search and multiplies the
//! potential by an infinitely smooth switching function below the cutoff: the energy and
//! all its derivatives are continuous, and time grows with the number of atoms.
//!
//! MultiOptPy: `Calculator/lj_calculation_tools.py:LennardJonesCore`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::AtomId;
use crate::domain::Quantity;
use faer::{Col, Mat};

use super::embedding::{evaluate_embedding, EmbeddingModel, ModelUnits};
use super::switching::switch_start;
use crate::domain::{Bohr, Element, EvalOutput, EvalRequest, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::Scalar;
use crate::pes::calculator::Calculator;

/// UFF `(x_i [Angstrom], D_i [kcal/mol])` (Rappe et al., J. Am. Chem. Soc. 1992, 114, 10024).
fn uff_noble_gas(el: Element) -> Option<(f64, f64)> {
    Some(match el.symbol() {
        "He" => (2.868, 0.0216),
        "Ne" => (3.087, 0.0731),
        "Ar" => (3.817, 0.237),
        "Kr" => (4.047, 0.357),
        "Xe" => (4.363, 0.507),
        "Rn" => (4.500, 0.635),
        _ => return None,
    })
}

/// Lennard-Jones cluster calculator.
#[derive(Debug, Clone, Copy, Default)]
pub struct LennardJones;

fn parameters(sys: &MolecularSystem) -> MoptResult<Vec<(f64, f64)>> {
    let conv = 1.0 / libm::pow(2.0, 1.0 / 6.0);
    sys.elements
        .iter()
        .map(|&e| {
            uff_noble_gas(e)
                .map(|(x, d)| (x * conv * Units::ANGSTROM_TO_BOHR, d / Units::HARTREE_TO_KCAL_MOL))
                .ok_or_else(|| {
                    MoptError::config(format!(
                        "Lennard-Jones parameters are available for He, Ne, Ar, Kr, Xe, Rn only (got {e})"
                    ))
                })
        })
        .collect()
}

impl Calculator for LennardJones {
    fn analytic_hessian(&self) -> bool {
        true
    }

    fn citations(&self) -> Vec<String> {
        vec![
            "A. K. Rappe, C. J. Casewit, K. S. Colwell, W. A. Goddard III, W. M. Skiff, J. Am. Chem. Soc. 1992, 114, 10024 (UFF parameters)".to_string(),
        ]
    }

    fn name(&self) -> &str {
        "lj"
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let params = parameters(sys)?;
        let n = sys.n_atoms();
        let mut energy = 0.0;
        let mut grad = Col::zeros(3 * n);
        let mut hess = req.need_hessian.then(|| Mat::zeros(3 * n, 3 * n));
        for a in 0..n {
            for b in (a + 1)..n {
                let d = sys.delta(AtomId::new(a), AtomId::new(b)).value();
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let sigma = 0.5 * (params[a].0 + params[b].0);
                let eps = libm::sqrt(params[a].1 * params[b].1);
                let sr6 = libm::pow(sigma * sigma / r2, 3.0);
                let sr12 = sr6 * sr6;
                energy += 4.0 * eps * (sr12 - sr6);
                // dV/dr / r
                let dv_r = -24.0 * eps / r2 * (2.0 * sr12 - sr6);
                for c in 0..3 {
                    grad[3 * a + c] += dv_r * d[c];
                    grad[3 * b + c] -= dv_r * d[c];
                }
                if let Some(h) = hess.as_mut() {
                    let d2v = 24.0 * eps / r2 * (26.0 * sr12 - 7.0 * sr6);
                    let t1 = (d2v - dv_r) / r2;
                    for p in 0..3 {
                        for q in 0..3 {
                            let block = t1 * d[p] * d[q] + if p == q { dv_r } else { 0.0 };
                            h[(3 * a + p, 3 * a + q)] += block;
                            h[(3 * b + p, 3 * b + q)] += block;
                            h[(3 * a + p, 3 * b + q)] -= block;
                            h[(3 * b + p, 3 * a + q)] -= block;
                        }
                    }
                }
            }
        }
        Ok(EvalOutput {
            energy: Quantity::new(energy),
            gradient_bohr: grad,
            hessian_bohr: hess,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }
}

/// Lennard-Jones interactions within a cutoff, switched off smoothly below it.
#[derive(Debug, Clone, Copy)]
pub struct TruncatedLennardJones {
    cutoff: f64,
    switch_on: Option<f64>,
}

impl TruncatedLennardJones {
    /// Pairs up to `cutoff`, switched off over the last `width` below it (a tenth of the
    /// cutoff without one; a width of zero cuts the potential as it is).
    pub fn new(cutoff: Quantity<Bohr>, width: Option<Quantity<Bohr>>) -> MoptResult<Self> {
        let cutoff = cutoff.value();
        if !(cutoff.is_finite() && cutoff > 0.0) {
            return Err(MoptError::config(format!(
                "lj: the cutoff must be a positive length, got {cutoff} Bohr"
            )));
        }
        Ok(Self {
            cutoff,
            switch_on: switch_start(cutoff, width.map(|w| w.value()))?,
        })
    }

    /// `(sigma [Bohr], epsilon [Hartree])` of an element (zero where it has none; `validate`
    /// refuses such elements before any is used).
    fn of(el: Element) -> (f64, f64) {
        let conv = 1.0 / libm::pow(2.0, 1.0 / 6.0);
        uff_noble_gas(el).map_or((0.0, 0.0), |(x, d)| {
            (
                x * conv * Units::ANGSTROM_TO_BOHR,
                d / Units::HARTREE_TO_KCAL_MOL,
            )
        })
    }
}

impl EmbeddingModel for TruncatedLennardJones {
    fn validate(&self, elements: &[Element]) -> MoptResult<()> {
        match elements.iter().find(|e| uff_noble_gas(**e).is_none()) {
            Some(e) => Err(MoptError::config(format!(
                "Lennard-Jones parameters are available for He, Ne, Ar, Kr, Xe, Rn only (got {e})"
            ))),
            None => Ok(()),
        }
    }

    fn cutoff(&self) -> f64 {
        self.cutoff
    }

    fn switch_on(&self) -> Option<f64> {
        self.switch_on
    }

    fn density<T: Scalar>(&self, _i: Element, _j: Element, _r: T) -> T {
        T::zero()
    }

    fn embedding<T: Scalar>(&self, _i: Element, _rho: T) -> T {
        T::zero()
    }

    fn pair<T: Scalar>(&self, i: Element, j: Element, r: T) -> T {
        // The engine sums over ordered pairs: half the pair potential each.
        let ((si, ei), (sj, ej)) = (Self::of(i), Self::of(j));
        let sr6 = (T::from_f64(0.5 * (si + sj)) / r).powi(6);
        (sr6 * sr6 - sr6) * T::from_f64(2.0 * libm::sqrt(ei * ej))
    }
}

impl Calculator for TruncatedLennardJones {
    fn analytic_hessian(&self) -> bool {
        true
    }

    fn citations(&self) -> Vec<String> {
        vec![
            "A. K. Rappe, C. J. Casewit, K. S. Colwell, W. A. Goddard III, W. M. Skiff, J. Am. Chem. Soc. 1992, 114, 10024 (UFF parameters)".to_string(),
        ]
    }

    fn name(&self) -> &str {
        "lj"
    }

    fn description(&self) -> String {
        let range = match self.switch_on {
            Some(on) => format!(
                "switched off from {:.4} Angstrom",
                on * Units::BOHR_TO_ANGSTROM
            ),
            None => "not switched".to_string(),
        };
        format!(
            "lj: built into mopt-rs {}; pairs within {:.4} Angstrom, {range}",
            crate::VERSION,
            self.cutoff * Units::BOHR_TO_ANGSTROM
        )
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let units = ModelUnits {
            length_per_bohr: 1.0,
            hartree_per_energy: 1.0,
        };
        let (energy, gradient_bohr, hessian_bohr, stress, strain_hessian) =
            evaluate_embedding(self, units, sys, req)?;
        Ok(EvalOutput {
            energy,
            gradient_bohr,
            hessian_bohr,
            orbitals: None,
            stress,
            strain_hessian,
        })
    }

    fn site_terms(
        &self,
        sys: &MolecularSystem,
    ) -> Option<MoptResult<crate::pes::calculator::SiteTerms>> {
        let units = ModelUnits {
            length_per_bohr: 1.0,
            hartree_per_energy: 1.0,
        };
        Some(super::embedding::site_terms(self, units, sys))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::quantity::shorthand::bohr;
    use crate::domain::{positions, Angstrom, PeriodicCell, SpinCharge};

    #[test]
    fn the_site_terms_add_up_to_the_energy_and_the_virial() {
        // Five argon atoms in a cell larger than twice the cutoff (no own images).
        let cell = PeriodicCell::orthorhombic(bohr(40.0), bohr(42.0), bohr(41.0)).unwrap();
        let sys = argon(
            &[
                [0.0, 0.0, 0.0],
                [3.7, 0.4, 0.1],
                [1.2, 3.6, -0.5],
                [19.0, 0.5, 1.0],
                [-1.5, -2.0, 3.3],
            ],
            Some(cell),
        );
        let calc = TruncatedLennardJones::new(bohr(16.0), Some(bohr(2.0))).unwrap();
        let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        let terms = calc.site_terms(&sys).unwrap().unwrap();
        let e: f64 = terms.energies.iter().sum();
        assert!((e - out.energy.value()).abs() < 1e-14);
        let volume = sys.active_cell().unwrap().volume_bohr3();
        let stress = out.stress.unwrap();
        for a in 0..3 {
            for b in 0..3 {
                let w: f64 = terms.virials.iter().map(|w| w[a][b]).sum();
                assert!((w + stress[a][b] * volume).abs() < 1e-12, "{a}{b}: {w}");
            }
        }
    }

    fn argon(points: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
        let el = Element::parse_token("Ar").unwrap();
        MolecularSystem::new(
            vec![el; points.len()],
            &positions::<Angstrom>(points),
            SpinCharge::default(),
            cell,
        )
        .unwrap()
    }

    #[test]
    fn a_cutoff_beyond_all_pairs_changes_nothing() {
        let sys = argon(
            &[
                [0.0, 0.0, 0.0],
                [3.9, 0.0, 0.0],
                [0.3, 3.1, 0.4],
                [2.0, 2.2, 3.3],
            ],
            None,
        );
        let exact = LennardJones
            .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
            .unwrap();
        // The switching begins beyond the longest distance.
        let cut = TruncatedLennardJones::new(bohr(40.0), None).unwrap();
        let found = cut.evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
        assert!((found.energy.value() - exact.energy.value()).abs() < 1e-15);
        let (h, k) = (found.hessian_bohr.unwrap(), exact.hessian_bohr.unwrap());
        for i in 0..12 {
            assert!((found.gradient_bohr[i] - exact.gradient_bohr[i]).abs() < 1e-15);
            for j in 0..12 {
                assert!((h[(i, j)] - k[(i, j)]).abs() < 1e-14);
            }
        }
        assert!(TruncatedLennardJones::new(bohr(-1.0), None).is_err());
        assert!(TruncatedLennardJones::new(bohr(10.0), Some(bohr(10.0))).is_err());
        let hydrogen = MolecularSystem::new(
            vec![Element::H; 2],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert!(cut.evaluate(&hydrogen, &EvalRequest::ENERGY_ONLY).is_err());
    }

    #[test]
    fn the_energy_has_no_step_at_the_cutoff_and_exact_derivatives_in_a_cell() {
        // Two atoms pulled through the cutoff: with the switching the energy and the force
        // reach zero there; without it the energy jumps by the potential at the cutoff.
        let rc = 8.0 * Units::BOHR_TO_ANGSTROM;
        let pair = |r: f64, calc: &TruncatedLennardJones| {
            let out = calc
                .evaluate(
                    &argon(&[[0.0, 0.0, 0.0], [r, 0.0, 0.0]], None),
                    &EvalRequest::ENERGY_GRAD,
                )
                .unwrap();
            (out.energy.value(), out.gradient_bohr[3])
        };
        let switched = TruncatedLennardJones::new(bohr(8.0), None).unwrap();
        let cut = TruncatedLennardJones::new(bohr(8.0), Some(bohr(0.0))).unwrap();
        let (inside, force) = pair(rc - 1e-7, &switched);
        assert!(
            inside.abs() < 1e-20 && force.abs() < 1e-14,
            "{inside} {force}"
        );
        assert_eq!(pair(rc + 1e-7, &switched), (0.0, 0.0));
        assert!(pair(rc - 1e-7, &cut).0.abs() > 1e-5);
        assert_eq!(pair(rc + 1e-7, &cut).0, 0.0);
        // A small cell (the cutoff is longer than the cell): differences against the
        // analytic gradient and Hessian, some pairs inside the switching range.
        let cell = PeriodicCell::orthorhombic(bohr(14.0), bohr(13.0), bohr(15.0)).unwrap();
        let sys = argon(
            &[[0.2, 0.1, 0.3], [3.8, 3.5, 4.1], [3.9, 0.2, 0.3]],
            Some(cell),
        );
        let calc = TruncatedLennardJones::new(bohr(16.0), Some(bohr(4.0))).unwrap();
        let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
        let h = out.hessian_bohr.clone().unwrap();
        let x0 = sys.coords_flat_bohr();
        let step = 1e-4;
        for i in 0..9 {
            let at = |sign: f64| {
                let mut x = x0.clone();
                x[i] += sign * step;
                calc.evaluate(
                    &sys.with_flat_coords_bohr(&x).unwrap(),
                    &EvalRequest::ENERGY_GRAD,
                )
                .unwrap()
            };
            let (plus, minus) = (at(1.0), at(-1.0));
            let slope = (plus.energy.value() - minus.energy.value()) / (2.0 * step);
            assert!((slope - out.gradient_bohr[i]).abs() < 1e-9, "gradient {i}");
            for j in 0..9 {
                let second = (plus.gradient_bohr[j] - minus.gradient_bohr[j]) / (2.0 * step);
                assert!((second - h[(i, j)]).abs() < 1e-8, "Hessian ({i}, {j})");
            }
        }
    }

    #[test]
    fn dimer_minimum_and_derivatives() {
        let el = Element::parse_token("Ar").unwrap();
        let rmin = 3.817; // UFF x_i is the minimum distance
        let sys = MolecularSystem::new(
            vec![el, el, el],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [rmin, 0.0, 0.0], [0.3, 3.1, 0.4]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let out = LennardJones
            .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
            .unwrap();
        let h = out.hessian_bohr.clone().unwrap();
        let x0 = sys.coords_flat_bohr();
        let step = 1e-5;
        for i in 0..9 {
            let mut xp = x0.clone();
            let mut xm = x0.clone();
            xp[i] += step;
            xm[i] -= step;
            let ep = LennardJones
                .evaluate(
                    &sys.with_flat_coords_bohr(&xp).unwrap(),
                    &EvalRequest::ENERGY_GRAD,
                )
                .unwrap();
            let em = LennardJones
                .evaluate(
                    &sys.with_flat_coords_bohr(&xm).unwrap(),
                    &EvalRequest::ENERGY_GRAD,
                )
                .unwrap();
            let fd = (ep.energy - em.energy).value() / (2.0 * step);
            assert!((fd - out.gradient_bohr[i]).abs() < 1e-9);
            for j in 0..9 {
                let fdh = (ep.gradient_bohr[j] - em.gradient_bohr[j]) / (2.0 * step);
                assert!((fdh - h[(i, j)]).abs() < 1e-7);
            }
        }
    }
}
