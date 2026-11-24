//! Effective medium theory (Jacobsen, Stoltze, Norskov, Surf. Sci. 1996, 366, 394), following
//! ASE's `ase.calculators.emt` (parameters, global cutoff, ASAP energy convention), with exact
//! analytic gradient and Hessian from the embedding engine.
//!
//! Per atom `i`: `sigma1_i = sum_j chi_ij exp(-eta2_j (r - beta s0_j)) w(r)`,
//! `s_i = -ln(sigma1_i / (12 gamma1_i)) / (beta eta2_i)`, and
//! `F_i = E0_i (1 + lambda_i s_i) exp(-lambda_i s_i) + 6 V0_i exp(-kappa_i s_i) - E0_i`;
//! the atomic-sphere pair term is
//! `phi_ij = (-V0_i/(4 gamma2_i)) chi_ij exp(-kappa_j (r/beta - s0_j)) w
//!         + (-V0_j/(4 gamma2_j)) exp(-kappa_i (r/beta - s0_i)) w / chi_ij`.
//!
//! ASE drops neighbors beyond the cutoff of its list, 0.5 Angstrom past the centre of the
//! Fermi function `w`, where `w` is about `1e-5`: the energy has a step of that relative
//! size there. [`Emt::with_switch_width`] switches the interactions off over the last part
//! of the list cutoff instead (not the default, which reproduces ASE).
//!
//! MultiOptPy's EMT double-counts the pair terms; see `docs/source/migration_surfaces.md`.
//!
//! MultiOptPy: `Calculator/emt_calculation_tools.py` (`EMT`, `EMTCore`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{Element, EvalOutput, EvalRequest, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::Scalar;
use crate::pes::calculator::Calculator;

use super::embedding::{evaluate_embedding, EmbeddingModel, ModelUnits};
use super::switching::switch_start;

/// Bohr in Angstrom as used by ASE (CODATA 2014) to convert the EMT parameter table. Kept on
/// purpose (not `Units::BOHR_TO_ANGSTROM`): the table is ASE's, stored in its Bohr-based units.
const ASE_BOHR: f64 = 0.529_177_210_563_841_1;
/// `beta = (16 pi / 3)^(1/3) / sqrt(2)`, rounded as in ASE.
const BETA: f64 = 1.809;

/// `(symbol, E0 [eV], s0 [Bohr], V0 [eV], eta2 [1/Bohr], kappa [1/Bohr], lambda [1/Bohr],
/// n0 [1/Bohr^3])` from ASE (Jacobsen et al. 1996; H, C, N, O are ASE's non-production extras).
const TABLE: [(&str, [f64; 7]); 11] = [
    ("Al", [-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700]),
    ("Cu", [-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910]),
    ("Ag", [-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547]),
    ("Au", [-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703]),
    ("Ni", [-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030]),
    ("Pd", [-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688]),
    ("Pt", [-5.85, 2.90, 4.067, 1.812, 3.145, 2.192, 0.00802]),
    ("H", [-3.21, 1.31, 0.132, 2.652, 2.790, 3.892, 0.00547]),
    ("C", [-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322]),
    ("N", [-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222]),
    ("O", [-4.60, 1.95, 0.332, 1.652, 2.790, 1.892, 0.00850]),
];

/// Per-element EMT parameters in Angstrom / eV.
#[derive(Debug, Clone, Copy)]
struct EmtParams {
    e0: f64,
    s0: f64,
    v0: f64,
    eta2: f64,
    kappa: f64,
    lambda: f64,
    n0: f64,
    inv12gamma1: f64,
    neg_half_v0_over_gamma2: f64,
}

/// EMT calculator.
#[derive(Debug, Clone)]
pub struct Emt {
    params: Vec<(Element, EmtParams)>,
    rc: f64,
    rc_list: f64,
    acut: f64,
    switch_on: Option<f64>,
}

impl Default for Emt {
    fn default() -> Self {
        Self::new()
    }
}

impl Emt {
    /// EMT with ASE's global cutoff (from the largest `s0` in the table).
    pub fn new() -> Self {
        let max_s0 = TABLE.iter().map(|(_, p)| p[1]).fold(0.0, f64::max) * ASE_BOHR;
        let r1nn = BETA * max_s0;
        let rc = r1nn * 0.5 * (3f64.sqrt() + 2.0);
        let r4nn = 2.0 * r1nn;
        let acut = libm::log(1.0 / 1e-4 - 1.0) / (r4nn - rc);
        let rc_list = rc + 0.5;
        let w = |r: f64| 1.0 / (1.0 + libm::exp(acut * (r - rc)));
        let params = TABLE
            .iter()
            .filter_map(|(sym, p)| {
                let el = Element::parse_token(sym).ok()?;
                let s0 = p[1] * ASE_BOHR;
                let (eta2, kappa) = (p[3] / ASE_BOHR, p[4] / ASE_BOHR);
                let (mut g1, mut g2) = (0.0, 0.0);
                for (n, k) in [(12.0, 1.0_f64), (6.0, 2.0), (24.0, 3.0)] {
                    let r = BETA * s0 * k.sqrt();
                    let x = n * w(r) / 12.0;
                    g1 += x * libm::exp(-eta2 * (r - BETA * s0));
                    g2 += x * libm::exp(-kappa / BETA * (r - BETA * s0));
                }
                Some((
                    el,
                    EmtParams {
                        e0: p[0],
                        s0,
                        v0: p[2],
                        eta2,
                        kappa,
                        lambda: p[5] / ASE_BOHR,
                        n0: p[6] / ASE_BOHR.powi(3),
                        inv12gamma1: 1.0 / (12.0 * g1),
                        neg_half_v0_over_gamma2: -0.5 * p[2] / g2,
                    },
                ))
            })
            .collect();
        Self {
            params,
            rc,
            rc_list,
            acut,
            switch_on: None,
        }
    }

    /// Switch the interactions off over the last `angstrom` below the cutoff of the neighbor
    /// list, so that the energy has no step there; 0: as in ASE.
    pub fn with_switch_width(mut self, angstrom: f64) -> MoptResult<Self> {
        self.switch_on = switch_start(self.rc_list, Some(angstrom))?;
        Ok(self)
    }

    fn p(&self, el: Element) -> EmtParams {
        // `validate` guarantees presence; the fallback is never used.
        self.params
            .iter()
            .find(|(e, _)| *e == el)
            .map(|(_, p)| *p)
            .unwrap_or(self.params[0].1)
    }

    fn cutoff_fn<T: Scalar>(&self, r: T) -> T {
        T::one() / (T::one() + (T::from_f64(self.acut) * (r - T::from_f64(self.rc))).exp())
    }
}

impl EmbeddingModel for Emt {
    fn validate(&self, elements: &[Element]) -> MoptResult<()> {
        match elements
            .iter()
            .find(|e| !self.params.iter().any(|(p, _)| p == *e))
        {
            Some(e) => Err(MoptError::config(format!("no EMT parameters for {e}"))),
            None => Ok(()),
        }
    }

    fn cutoff(&self) -> f64 {
        self.rc_list
    }

    fn switch_on(&self) -> Option<f64> {
        self.switch_on
    }

    fn density<T: Scalar>(&self, i: Element, j: Element, r: T) -> T {
        let (pi, pj) = (self.p(i), self.p(j));
        let chi = pj.n0 / pi.n0;
        (T::from_f64(-pj.eta2) * (r - T::from_f64(BETA * pj.s0))).exp()
            * T::from_f64(chi)
            * self.cutoff_fn(r)
    }

    fn embedding<T: Scalar>(&self, i: Element, sigma: T) -> T {
        let p = self.p(i);
        if sigma.val() <= 0.0 {
            // Isolated atom: ds -> infinity.
            return T::from_f64(-p.e0);
        }
        let ds = -(sigma * T::from_f64(p.inv12gamma1)).ln() / T::from_f64(BETA * p.eta2);
        let lds = ds * T::from_f64(p.lambda);
        T::from_f64(p.e0) * (T::one() + lds) * (-lds).exp()
            + T::from_f64(6.0 * p.v0) * (ds * T::from_f64(-p.kappa)).exp()
            - T::from_f64(p.e0)
    }

    fn pair<T: Scalar>(&self, i: Element, j: Element, r: T) -> T {
        let (pi, pj) = (self.p(i), self.p(j));
        let chi = pj.n0 / pi.n0;
        let w = self.cutoff_fn(r);
        let s2s = (T::from_f64(-pj.kappa) * (r / T::from_f64(BETA) - T::from_f64(pj.s0))).exp()
            * T::from_f64(chi)
            * w;
        let s2o = (T::from_f64(-pi.kappa) * (r / T::from_f64(BETA) - T::from_f64(pi.s0))).exp() * w
            / T::from_f64(chi);
        (s2s * T::from_f64(pi.neg_half_v0_over_gamma2)
            + s2o * T::from_f64(pj.neg_half_v0_over_gamma2))
            * T::from_f64(0.5)
    }
}

impl Calculator for Emt {
    fn analytic_hessian(&self) -> bool {
        true
    }

    fn citations(&self) -> Vec<String> {
        vec![
            "K. W. Jacobsen, P. Stoltze, J. K. Norskov, Surf. Sci. 1996, 366, 394 (effective medium theory)".to_string(),
            "A. H. Larsen et al., J. Phys.: Condens. Matter 2017, 29, 273002 (ASE, whose EMT parameters and conventions are used)".to_string(),
        ]
    }

    fn name(&self) -> &str {
        "emt"
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let units = ModelUnits {
            length_per_bohr: Units::BOHR_TO_ANGSTROM,
            hartree_per_energy: Units::EV_TO_HARTREE,
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
            length_per_bohr: Units::BOHR_TO_ANGSTROM,
            hartree_per_energy: Units::EV_TO_HARTREE,
        };
        Some(super::embedding::site_terms(self, units, sys))
    }
}
