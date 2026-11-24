//! The ONIOM energy surface as one calculator: N levels of theory on N nested regions,
//! combined by subtraction (Svensson, Humbel, Froese, Matsubara, Sieber, Morokuma,
//! J. Phys. Chem. 1996, 100, 19357; Dapprich, Komaromi, Byun, Morokuma, Frisch,
//! J. Mol. Struct. THEOCHEM 1999, 461, 1):
//!
//! ```text
//! E = E_LN(real) + sum_(k = 1 .. N-1) [ E_Lk(M_k) - E_L(k+1)(M_k) ]
//! ```
//!
//! with the regions `M_1` (innermost) to `M_(N-1)` and the levels `L_1` (highest) to `L_N`
//! (lowest; on the whole, the real system). Two layers: `E_low(real) + E_high(model) -
//! E_low(model)`. Gradients and Hessians of the model systems go back to the real atoms
//! through the link atoms ([`super::layout`]). By default this is mechanical embedding: every
//! model system is computed by itself, the environment acts through the lower level only.
//! With electronic embedding ([`OniomCalculator::with_electronic_embedding`],
//! [`super::embedding`]) the higher level of every region sees the charges of the atoms around
//! it, and the classical Coulomb energy of the model with them is taken off with the lower
//! level; the Hessian is then made of central differences of the gradient.
//!
//! Every term has a calculator of its own, made for the system it computes (a force field
//! sets up its topology from it).
//!
//! An excited state (`-es`) is that of the highest level on the innermost region; the other
//! terms are ground states. A periodic cell is the real system's: the lowest level sees it,
//! the model systems are clusters. A stress (a variable cell) is not available.
//!
//! MultiOptPy: `optimization.py:ONIOMHandler` (not working there: its `compute` starts by
//! raising an exception).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::embedding::{BoundaryCharges, ChargeSource, ElectronicEmbedding, RegionCharges};
use super::layout::Region;
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::pes::calculator::{evaluate_checked, Calculator};
use crate::pes::point_charges::PointCharges;

/// Displacement of the central differences of the Hessian with electronic embedding (Bohr).
const EMBEDDED_HESSIAN_STEP_BOHR: f64 = 5.0e-3;

/// The electronic embedding of a built ONIOM surface: the charges around every region.
struct Embedded {
    regions: Vec<RegionCharges>,
    description: String,
}

/// One term of the sum: a level on a region (or on the real system) with its sign.
#[derive(Clone)]
struct Term {
    calc: Arc<dyn Calculator>,
    /// `None`: the real system.
    region: Option<usize>,
    level: usize,
    sign: f64,
}

/// A structure of another charge or multiplicity than `real_state` (a state of -mf): the
/// model systems follow it where their state was not given; a given one cannot.
fn check_state(
    regions: &[Region],
    real_state: crate::domain::SpinCharge,
    sys: &MolecularSystem,
) -> MoptResult<()> {
    if sys.spin_charge != real_state && regions.iter().any(|r| r.follows.is_none()) {
        return Err(MoptError::unsupported(format!(
            "ONIOM: a structure of charge {} and multiplicity {} on a surface set up for \
             charge {} and multiplicity {}, with the charge and multiplicity of a model \
             system given (-oniom_opt high_charge, high_multiplicity): the model system \
             of the other state is not known",
            sys.spin_charge.charge,
            sys.spin_charge.multiplicity,
            real_state.charge,
            real_state.multiplicity
        )));
    }
    Ok(())
}

/// The terms of the model systems of a surface with mechanical embedding, apart from it: with
/// the lowest level on the real system they make the surface,
/// `E = E_LN(real) + sum_k s_k E_k(M_k)`. They depend on the atoms of the regions and of their
/// links alone (the core of the microiterations), so their Hessian is had there without the
/// Hessian of the real system ([`super::micro`] takes the rest from the lowest level).
#[derive(Clone)]
pub struct ModelPart {
    terms: Vec<Term>,
    regions: Vec<Region>,
    real_state: crate::domain::SpinCharge,
}

impl ModelPart {
    /// Whether every term has an analytic Hessian.
    pub fn analytic_hessian(&self) -> bool {
        self.terms.iter().all(|t| t.calc.analytic_hessian())
    }

    /// The real atoms the terms depend on (0-based, ascending): the atoms of the regions and
    /// both atoms of every link.
    pub fn atoms(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .regions
            .iter()
            .flat_map(|r| {
                r.atoms.iter().map(|a| a.index()).chain(
                    r.links
                        .iter()
                        .flat_map(|l| [l.inner.index(), l.outer.index()]),
                )
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The sum of the Hessians of the terms at `sys` on the real atoms `atoms` (which hold
    /// those of [`Self::atoms`]; rows and columns in their order), the highest level in the
    /// state `excited_state` and the others in their ground states, as the surface computes
    /// them. `None` where a term gives no Hessian.
    pub fn hessian_on(
        &self,
        sys: &MolecularSystem,
        atoms: &[usize],
        excited_state: usize,
    ) -> MoptResult<Option<Mat<f64>>> {
        check_state(&self.regions, self.real_state, sys)?;
        let n = 3 * atoms.len();
        let mut total = Mat::<f64>::zeros(n, n);
        for term in &self.terms {
            let Some(k) = term.region else {
                continue;
            };
            let region = &self.regions[k];
            let model = region.model(sys)?;
            let request = EvalRequest {
                excited_state: if term.level == 0 { excited_state } else { 0 },
                ..EvalRequest::ENERGY_GRAD_HESS
            };
            let out = evaluate_checked(term.calc.as_ref(), &model, &request).context(format!(
                "ONIOM level {} on region {}",
                term.level + 1,
                k + 1
            ))?;
            let Some(h) = &out.hessian_bohr else {
                return Ok(None);
            };
            total += region.hessian_on_atoms(sys, &out.gradient_bohr, h, atoms)?
                * faer::Scale(term.sign);
        }
        Ok(Some(total))
    }

    /// The sum of the energies of the terms at `sys` and their gradient on the real system
    /// (`sum_k s_k E_k(M_k)`), the highest level in the state `excited_state` and the others in
    /// their ground states, with the output of the highest level (its orbitals): the part of
    /// the surface that the environment does not change, computed once for every environment
    /// at the same core (the replicas of the environment, `super::micro`).
    pub fn energy_gradient(
        &self,
        sys: &MolecularSystem,
        excited_state: usize,
    ) -> MoptResult<(f64, Col<f64>, Option<EvalOutput>)> {
        check_state(&self.regions, self.real_state, sys)?;
        let mut energy = 0.0;
        let mut gradient = Col::<f64>::zeros(sys.dof());
        let mut high = None;
        for term in &self.terms {
            let Some(k) = term.region else {
                continue;
            };
            let region = &self.regions[k];
            let model = region.model(sys)?;
            let request = EvalRequest {
                excited_state: if term.level == 0 { excited_state } else { 0 },
                ..EvalRequest::ENERGY_GRAD
            };
            let out = evaluate_checked(term.calc.as_ref(), &model, &request).context(format!(
                "ONIOM level {} on region {}",
                term.level + 1,
                k + 1
            ))?;
            energy += term.sign * out.energy.value();
            gradient += region.gradient_to_real(sys, &out.gradient_bohr)? * faer::Scale(term.sign);
            if term.level == 0 {
                high = Some(out);
            }
        }
        Ok((energy, gradient, high))
    }
}

/// The ONIOM surface of N levels on N - 1 nested regions.
pub struct OniomCalculator {
    terms: Vec<Term>,
    regions: Vec<Region>,
    levels: usize,
    name: String,
    descriptions: Vec<String>,
    embedded: Option<Embedded>,
    /// Charge and multiplicity of the whole structure the surface was built for.
    real_state: crate::domain::SpinCharge,
}

impl OniomCalculator {
    /// The surface of `levels` levels (the highest is 0) on `regions` (innermost first; one
    /// fewer than the levels) for structures like `real`. `make(level, system, region)`
    /// makes the calculator of a level for a system: the real system (`region` is `None`)
    /// or the model system of a region.
    pub fn build(
        levels: usize,
        regions: Vec<Region>,
        real: &MolecularSystem,
        make: impl Fn(usize, &MolecularSystem, Option<&Region>) -> MoptResult<Box<dyn Calculator>>,
    ) -> MoptResult<Self> {
        if levels < 2 || regions.len() + 1 != levels {
            return Err(MoptError::config(format!(
                "ONIOM needs one region fewer than levels (two levels at least): {levels} \
                 level(s), {} region(s)",
                regions.len()
            )));
        }
        for pair in regions.windows(2) {
            if let Some(a) = pair[0].atoms.iter().find(|a| !pair[1].atoms.contains(a)) {
                return Err(MoptError::config(format!(
                    "ONIOM: atom {} of an inner region is not in the region around it",
                    a.one_based()
                )));
            }
        }
        let mut terms = Vec::with_capacity(2 * levels - 1);
        let last = levels - 1;
        terms.push(Term {
            calc: Arc::from(
                make(last, real, None).context("ONIOM: the lowest level on the real system")?,
            ),
            region: None,
            level: last,
            sign: 1.0,
        });
        for (k, region) in regions.iter().enumerate() {
            let model = region.model(real)?;
            for (level, sign) in [(k, 1.0), (k + 1, -1.0)] {
                terms.push(Term {
                    calc: Arc::from(make(level, &model, Some(region)).context(format!(
                        "ONIOM: level {} on region {}",
                        level + 1,
                        k + 1
                    ))?),
                    region: Some(k),
                    level,
                    sign,
                });
            }
        }
        let level_name = |level: usize| -> String {
            terms
                .iter()
                .find(|t| t.level == level)
                .map_or_else(String::new, |t| t.calc.name().to_string())
        };
        let names: Vec<String> = (0..levels).map(level_name).collect();
        let descriptions = (0..levels)
            .map(|level| {
                terms
                    .iter()
                    .find(|t| t.level == level)
                    .map_or_else(String::new, |t| t.calc.description())
            })
            .collect();
        Ok(Self {
            name: format!("oniom({})", names.join(":")),
            terms,
            regions,
            levels,
            descriptions,
            embedded: None,
            real_state: real.spin_charge,
        })
    }

    /// The same surface with electronic embedding (`embedding`) for structures like `real`:
    /// the higher level of every region in the charges of the atoms around it. Refused for a
    /// level that takes no point charges, a lowest level without charges when they are to
    /// come from it, and a periodic cell.
    pub fn with_electronic_embedding(
        mut self,
        embedding: &ElectronicEmbedding,
        real: &MolecularSystem,
    ) -> MoptResult<Self> {
        if real.active_cell().is_some() {
            return Err(MoptError::unsupported(
                "ONIOM electronic embedding in a periodic cell is not available; use the \
                 mechanical one (-oniom_opt embedding=mechanical)",
            ));
        }
        for term in self
            .terms
            .iter()
            .filter(|t| t.sign > 0.0 && t.region.is_some())
        {
            if !term.calc.supports_point_charges() {
                return Err(MoptError::unsupported(format!(
                    "ONIOM electronic embedding: level {} ({}) takes no point charges",
                    term.level + 1,
                    term.calc.name()
                )));
            }
        }
        let (charges, source) = match &embedding.charges {
            ChargeSource::Given(q) => (q.clone(), "the given charges".to_string()),
            ChargeSource::Lowest => {
                let lowest = self
                    .terms
                    .iter()
                    .find(|t| t.region.is_none())
                    .ok_or_else(|| MoptError::calc("ONIOM: no term of the real system"))?;
                let q = lowest.calc.embedding_charges(real)?.ok_or_else(|| {
                    MoptError::unsupported(format!(
                        "ONIOM electronic embedding: the lowest level ({}) has no charges; \
                         give them (-oniom_opt charges=FILE)",
                        lowest.calc.name()
                    ))
                })?;
                (q, format!("the charges of {}", lowest.calc.name()))
            }
        };
        let regions = self
            .regions
            .iter()
            .map(|r| RegionCharges::new(r, real, &charges, embedding.boundary))
            .collect::<MoptResult<Vec<_>>>()?;
        let boundary = match embedding.boundary {
            BoundaryCharges::Redistributed => {
                "the charge of each Q2 moved to the midpoints of its bonds (rc)"
            }
            BoundaryCharges::Zero => "the charges of the Q2 atoms left out (zero)",
            BoundaryCharges::Keep => "the charges of the Q2 atoms kept (keep)",
        };
        let description = format!(
            "electronic embedding: the higher level of each region in {source} ({} point \
             charges around region 1; {boundary}); Hessian: central differences of the \
             gradient",
            regions.first().map_or(0, RegionCharges::len)
        );
        self.embedded = Some(Embedded {
            regions,
            description,
        });
        Ok(self)
    }

    /// The regions (innermost first).
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// The terms of the model systems apart from the surface ([`ModelPart`]); `None` with
    /// electronic embedding, where they see the environment.
    pub fn model_part(&self) -> Option<ModelPart> {
        if self.embedded.is_some() {
            return None;
        }
        Some(ModelPart {
            terms: self
                .terms
                .iter()
                .filter(|t| t.region.is_some())
                .cloned()
                .collect(),
            regions: self.regions.clone(),
            real_state: self.real_state,
        })
    }

    /// The energy, gradient, and (when asked) Hessian of `term` at `sys`, in the real
    /// system, with what its calculator gave.
    fn evaluate_term(
        &self,
        term: &Term,
        sys: &MolecularSystem,
        req: &EvalRequest,
        charges: Option<(&RegionCharges, &PointCharges)>,
    ) -> MoptResult<(f64, Col<f64>, Option<Mat<f64>>, EvalOutput)> {
        let request = EvalRequest {
            excited_state: if term.level == 0 {
                req.excited_state
            } else {
                0
            },
            need_stress: false,
            ..*req
        };
        match term.region {
            None => {
                let out = evaluate_checked(term.calc.as_ref(), sys, &request)
                    .context(format!("ONIOM level {} on the real system", term.level + 1))?;
                Ok((
                    out.energy.value(),
                    out.gradient_bohr.clone(),
                    out.hessian_bohr.clone(),
                    out,
                ))
            }
            Some(k) => {
                let region = &self.regions[k];
                let model = region.model(sys)?;
                let on = || format!("ONIOM level {} on region {}", term.level + 1, k + 1);
                if let Some((around, pc)) = charges {
                    return self.embedded_term(term, sys, &model, &request, around, pc, k);
                }
                let out = evaluate_checked(term.calc.as_ref(), &model, &request).context(on())?;
                let g = region.gradient_to_real(sys, &out.gradient_bohr)?;
                let h = match &out.hessian_bohr {
                    Some(h) => Some(region.hessian_to_real(sys, &out.gradient_bohr, h)?),
                    None => None,
                };
                Ok((out.energy.value(), g, h, out))
            }
        }
    }

    /// A term of region `k` with electronic embedding: the higher level in the charges `pc`,
    /// the lower level by itself with the classical Coulomb energy of the model atoms with
    /// the same charges. The gradient includes that of the positions of the charges.
    #[allow(clippy::too_many_arguments)]
    fn embedded_term(
        &self,
        term: &Term,
        sys: &MolecularSystem,
        model: &MolecularSystem,
        request: &EvalRequest,
        around: &RegionCharges,
        pc: &PointCharges,
        k: usize,
    ) -> MoptResult<(f64, Col<f64>, Option<Mat<f64>>, EvalOutput)> {
        let region = &self.regions[k];
        let on = || format!("ONIOM level {} on region {}", term.level + 1, k + 1);
        let request = EvalRequest {
            need_hessian: false,
            ..*request
        };
        if term.sign > 0.0 {
            let embedded = term
                .calc
                .evaluate_in_charges(model, &request, pc)
                .context(on())?;
            let mut g = region.gradient_to_real(sys, &embedded.output.gradient_bohr)?;
            around.add_to_real(&embedded.charge_gradient_bohr, &mut g);
            return Ok((embedded.output.energy.value(), g, None, embedded.output));
        }
        let out = evaluate_checked(term.calc.as_ref(), model, &request).context(on())?;
        let x = model.coords_flat_bohr();
        let positions: Vec<[f64; 3]> = (0..model.n_atoms())
            .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
            .collect();
        let (coulomb, of_model, of_charges) =
            pc.coulomb_with(&positions, around.model_charges())?;
        let model_gradient = Col::from_fn(x.nrows(), |i| out.gradient_bohr[i] + of_model[i]);
        let mut g = region.gradient_to_real(sys, &model_gradient)?;
        around.add_to_real(&of_charges, &mut g);
        Ok((out.energy.value() + coulomb, g, None, out))
    }

    /// The Hessian of the embedded surface from central differences of its gradient.
    fn embedded_hessian(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<Mat<f64>> {
        let x0 = sys.coords_flat_bohr();
        let dim = x0.nrows();
        let h = EMBEDDED_HESSIAN_STEP_BOHR;
        let gradient_only = EvalRequest {
            need_hessian: false,
            ..*req
        };
        let mut out = Mat::<f64>::zeros(dim, dim);
        for k in 0..dim {
            let mut xp = x0.clone();
            xp[k] += h;
            let mut xm = x0.clone();
            xm[k] -= h;
            let gp = self
                .evaluate(&sys.with_flat_coords_bohr(&xp)?, &gradient_only)?
                .gradient_bohr;
            let gm = self
                .evaluate(&sys.with_flat_coords_bohr(&xm)?, &gradient_only)?
                .gradient_bohr;
            for i in 0..dim {
                out[(i, k)] = (gp[i] - gm[i]) / (2.0 * h);
            }
        }
        Ok(Mat::from_fn(dim, dim, |i, j| {
            0.5 * (out[(i, j)] + out[(j, i)])
        }))
    }
}

impl Calculator for OniomCalculator {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> String {
        let mut parts = vec![match &self.embedded {
            None => format!("ONIOM with {} levels (mechanical embedding)", self.levels),
            Some(e) => format!("ONIOM with {} levels ({})", self.levels, e.description),
        }];
        for (k, description) in self.descriptions.iter().enumerate() {
            let on = match self.regions.get(k) {
                Some(r) => format!(
                    "{} atoms and {} link atom(s), charge {} multiplicity {}",
                    r.atoms.len(),
                    r.links.len(),
                    r.spin_charge.charge,
                    r.spin_charge.multiplicity
                ),
                None => "the real system".to_string(),
            };
            parts.push(format!("level {}: {description} on {on}", k + 1));
        }
        parts.join("; ")
    }

    fn citations(&self) -> Vec<String> {
        let mut out = vec![
            "M. Svensson, S. Humbel, R. D. J. Froese, T. Matsubara, S. Sieber, K. Morokuma, \
             J. Phys. Chem. 1996, 100, 19357 (ONIOM)"
                .to_string(),
            "S. Dapprich, I. Komaromi, K. S. Byun, K. Morokuma, M. J. Frisch, J. Mol. Struct. \
             THEOCHEM 1999, 461, 1 (ONIOM gradients and link atoms)"
                .to_string(),
        ];
        if self.embedded.is_some() {
            out.push(
                "T. Vreven, K. S. Byun, I. Komáromi, S. Dapprich, J. A. Montgomery Jr., K. \
                 Morokuma, M. J. Frisch, J. Chem. Theory Comput. 2006, 2, 815 (ONIOM electronic \
                 embedding)"
                    .to_string(),
            );
            out.push(
                "H. Lin, D. G. Truhlar, J. Phys. Chem. A 2005, 109, 3991 (redistributed charges \
                 at the boundary)"
                    .to_string(),
            );
        }
        for term in &self.terms {
            for c in term.calc.citations() {
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        }
        out
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if req.need_stress {
            return Err(MoptError::unsupported(
                "ONIOM has no stress of the cell yet (-vc is not available with -oniom)",
            ));
        }
        check_state(&self.regions, self.real_state, sys)?;
        // With electronic embedding: the charges around every region at this structure, and
        // the Hessian from differences of the gradient.
        let around: Option<Vec<(&RegionCharges, PointCharges)>> = match &self.embedded {
            None => None,
            Some(e) => Some(
                e.regions
                    .iter()
                    .map(|r| Ok((r, r.at(sys)?)))
                    .collect::<MoptResult<_>>()?,
            ),
        };
        let embedded_hessian = match (&around, req.need_hessian) {
            (Some(_), true) => Some(self.embedded_hessian(sys, req)?),
            _ => None,
        };
        let req = &EvalRequest {
            need_hessian: req.need_hessian && around.is_none(),
            ..*req
        };
        let mut energy = 0.0;
        let mut gradient = Col::<f64>::zeros(sys.dof());
        let mut hessian = req
            .need_hessian
            .then(|| Mat::<f64>::zeros(sys.dof(), sys.dof()));
        let mut real_output = None;
        let mut high_output = None;
        for term in &self.terms {
            let charges = match (&around, term.region) {
                (Some(list), Some(k)) => list.get(k).map(|(r, pc)| (*r, pc)),
                _ => None,
            };
            let (e, g, h, out) = self.evaluate_term(term, sys, req, charges)?;
            energy += term.sign * e;
            gradient += g * faer::Scale(term.sign);
            if let Some(total) = hessian.as_mut() {
                let h = h.ok_or_else(|| {
                    MoptError::calc(format!(
                        "ONIOM: level {} returned no Hessian (every level needs one)",
                        term.level + 1
                    ))
                })?;
                *total += h * faer::Scale(term.sign);
            }
            if term.region.is_none() {
                real_output = Some(out);
            } else if term.level == 0 {
                high_output = Some(out);
            }
        }
        let mut out = real_output
            .ok_or_else(|| MoptError::calc("ONIOM: the real system was not computed"))?;
        out.energy = Quantity::new(energy);
        out.gradient_bohr = gradient;
        out.hessian_bohr = embedded_hessian.or(hessian);
        out.stress = None;
        // What the highest level says about its wavefunction (<S**2>, its output).
        if let Some(high) = high_output {
            out.orbitals = high.orbitals;
        }
        Ok(out)
    }

    fn supports_pbc(&self) -> bool {
        self.terms
            .iter()
            .filter(|t| t.region.is_none())
            .all(|t| t.calc.supports_pbc())
    }

    fn supports_excited_states(&self) -> bool {
        self.terms
            .iter()
            .filter(|t| t.level == 0)
            .all(|t| t.calc.supports_excited_states())
    }

    /// Those of the highest level on the innermost region (mechanical embedding; with
    /// electronic embedding they are not reported).
    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        if self.embedded.is_some() {
            return Ok(None);
        }
        let Some(term) = self
            .terms
            .iter()
            .find(|t| t.level == 0 && t.region.is_some())
        else {
            return Ok(None);
        };
        let Some(region) = term.region.and_then(|k| self.regions.get(k)) else {
            return Ok(None);
        };
        term.calc.excitation_energies(&region.model(sys)?)
    }

    fn followed_state(&self, sys: &MolecularSystem, label: usize) -> Option<usize> {
        let term = self
            .terms
            .iter()
            .find(|t| t.level == 0 && t.region.is_some())?;
        let region = term.region.and_then(|k| self.regions.get(k))?;
        term.calc.followed_state(&region.model(sys).ok()?, label)
    }

    fn recall_state(&self, sys: &MolecularSystem, label: usize, root: usize) {
        let highest = self
            .terms
            .iter()
            .find(|t| t.level == 0 && t.region.is_some());
        if let Some(term) = highest {
            let model = term
                .region
                .and_then(|k| self.regions.get(k))
                .and_then(|region| region.model(sys).ok());
            if let Some(model) = model {
                term.calc.recall_state(&model, label, root);
            }
        }
    }

    fn evaluates_in_order(&self) -> bool {
        self.terms.iter().any(|t| t.calc.evaluates_in_order())
    }

    fn uses_electrons(&self) -> bool {
        self.terms.iter().any(|t| t.calc.uses_electrons())
    }

    fn is_rigid_body_invariant(&self) -> bool {
        self.terms.iter().all(|t| t.calc.is_rigid_body_invariant())
    }

    /// Analytic where every level's Hessian is and the embedding is mechanical (with
    /// electronic embedding the Hessian comes from differences of the gradient).
    fn analytic_hessian(&self) -> bool {
        self.embedded.is_none() && self.terms.iter().all(|t| t.calc.analytic_hessian())
    }

    fn checks_stability(&self) -> bool {
        self.terms.iter().any(|t| t.calc.checks_stability())
    }

    fn take_warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        for term in &self.terms {
            let on = match term.region {
                Some(k) => format!("region {}", k + 1),
                None => "the real system".to_string(),
            };
            out.extend(
                term.calc
                    .take_warnings()
                    .into_iter()
                    .map(|w| format!("ONIOM level {} on {on}: {w}", term.level + 1)),
            );
        }
        out
    }
}
