//! The energies of the parts of a complex alone, for the minima of `orientsearch`.
//!
//! The interaction energy of a minimum is its energy minus those of its parts alone at their
//! geometries in it (one evaluation per part, `interaction=vertical`); the binding energy is
//! its energy minus those of the parts optimized alone from the input (`interaction=relaxed`).
//! Both use the calculator without bias potentials; they are left out with ONIOM (its layers
//! are given for the atoms of the whole complex), with `-mf` (two states), and for excited
//! states, and the charges of the parts must be known (`part_charges=` for a charged complex).
//! The evaluations go into the journal of the run directory, so a restart reads them.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use super::orientsearch::OrientsearchConfig;
use super::orientsearch_settings::Interaction;
use crate::core_infra::log::Logger;
use crate::domain::{AtomId, EvalRequest, MolecularSystem, SpinCharge};
use crate::error::MoptResult;
use crate::opt::state_machine::{GeometryOptimizer, Termination};
use crate::pes::calculator::evaluate_checked;
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};
use crate::workflow::explore::orient::{ComplexParts, OrientationSearch};
use crate::workflow::journal::RunJournal;
use crate::workflow::run_optimization;

/// The energies of the parts alone.
#[derive(Debug, Clone, Default)]
pub struct PartEnergies {
    /// Per minimum (in the order of the pool): the sum of the energies of its parts alone at
    /// their geometries in it.
    pub vertical: Vec<Option<f64>>,
    /// The sum of the energies of the parts optimized alone.
    pub relaxed: Option<f64>,
}

/// The states of the parts alone, or why they are not known.
fn part_states(
    config: &OrientsearchConfig,
    template: &MolecularSystem,
    parts: &ComplexParts,
) -> Result<Vec<SpinCharge>, String> {
    let n = parts.len();
    let charges: Vec<i32> = match &config.part_charges {
        Some(c) if c.len() == n => c.clone(),
        Some(c) => return Err(format!("part_charges has {} values for {n} parts", c.len())),
        None if template.spin_charge.charge == 0 => vec![0; n],
        None => return Err(
            "the complex is charged; give the charges of the parts (-orient_opt part_charges=...)"
                .to_string(),
        ),
    };
    if charges.iter().sum::<i32>() != template.spin_charge.charge {
        return Err(format!(
            "the charges of the parts add up to {}, the complex has {}",
            charges.iter().sum::<i32>(),
            template.spin_charge.charge
        ));
    }
    let multiplicities: Vec<u32> = match &config.part_multiplicities {
        Some(m) if m.len() == n => m.clone(),
        Some(m) => {
            return Err(format!(
                "part_multiplicities has {} values for {n} parts",
                m.len()
            ))
        }
        None => parts
            .atoms
            .iter()
            .zip(&charges)
            .map(|(atoms, c)| {
                let electrons: i64 = atoms
                    .iter()
                    .map(|&i| i64::from(template.elements[i].atomic_number()))
                    .sum::<i64>()
                    - i64::from(*c);
                if electrons % 2 == 0 {
                    1
                } else {
                    2
                }
            })
            .collect(),
    };
    Ok(charges
        .into_iter()
        .zip(multiplicities)
        .map(|(c, m)| SpinCharge::new(c, m))
        .collect())
}

impl PartEnergies {
    /// The energies the settings ask for; left out (with a note in the log) where they are not
    /// defined.
    pub fn compute(
        config: &OrientsearchConfig,
        registry: &CalculatorRegistry,
        template: &MolecularSystem,
        parts: &ComplexParts,
        search: &OrientationSearch,
        dir: &Path,
        log: &Logger,
    ) -> MoptResult<Self> {
        let mut out = Self {
            vertical: vec![None; search.pool().len()],
            relaxed: None,
        };
        if config.interaction == Interaction::Off || search.pool().is_empty() {
            return Ok(out);
        }
        let o = &config.optmain;
        let skip = if o.calculator_options.oniom.is_some() {
            Some("with -oniom the layers are given for the atoms of the whole complex")
        } else if !o.model_function.is_empty() {
            Some("-mf combines two states of the whole complex")
        } else if o.excited_state > 0 {
            Some("the excited states of the parts alone are other states")
        } else {
            None
        };
        if let Some(why) = skip {
            log.info(&format!(
                "Energies of the parts alone are not computed: {why}"
            ));
            return Ok(out);
        }
        let states = match part_states(config, template, parts) {
            Ok(s) => s,
            Err(why) => {
                log.info(&format!(
                    "Energies of the parts alone are not computed: {why}"
                ));
                return Ok(out);
            }
        };
        let plain = CalculatorOptions {
            oniom: None,
            ..o.calculator_options.clone()
        };
        let journal = RunJournal::open_with(dir, registry.replay())?;
        let subsystem = |sys: &MolecularSystem, k: usize| {
            let atoms: Vec<AtomId> = parts.atoms[k].iter().map(|&i| AtomId::new(i)).collect();
            sys.subsystem(&atoms, states[k])
        };
        if config.interaction.vertical() {
            for (m, minimum) in search.pool().minima().iter().enumerate() {
                let mut sum = 0.0;
                let mut failed = None;
                for k in 0..parts.len() {
                    let sub = subsystem(&minimum.system, k)?;
                    let energy = registry
                        .create(&o.calculator, &plain, &sub)
                        .and_then(|calc| {
                            let label = format!("part_{}", k + 1);
                            let j = journal.wrap(&label, calc.as_ref());
                            evaluate_checked(&j, &sub, &EvalRequest::ENERGY_ONLY)
                        });
                    match energy {
                        Ok(e) => sum += e.energy.value(),
                        Err(e) => failed = Some(e),
                    }
                }
                match failed {
                    None => out.vertical[m] = Some(sum),
                    Some(e) => log.warn(&format!(
                        "Part energies of the minimum of sample {}: {e}",
                        minimum.first_trial
                    )),
                }
            }
        }
        if config.interaction.relaxed() {
            if o.settings.saddle_order > 0 {
                log.info("Binding energies are not computed: the stationary points are saddle points (-order)");
            } else {
                let mut sum = 0.0;
                for k in 0..parts.len() {
                    let sub = subsystem(template, k)?;
                    let calc = registry.create(&o.calculator, &plain, &sub)?;
                    let label = format!("part_opt_{}", k + 1);
                    let j = journal.wrap(&label, calc.as_ref());
                    let mut settings = o.settings.clone();
                    settings.constraints.clear();
                    settings.fragments.clear();
                    settings.ccqn = None;
                    settings.rigid_body_invariant = calc.is_rigid_body_invariant();
                    let mut opt = GeometryOptimizer::new(settings, sub)?;
                    let outcome = run_optimization(&mut opt, &j, &mut [])?;
                    if outcome.termination != Some(Termination::Converged) {
                        log.warn(&format!(
                            "Part {} alone did not converge in {} iterations; its energy is that of the last structure",
                            k + 1,
                            outcome.iterations
                        ));
                    }
                    log.info(&format!(
                        "Part {} optimized alone: E = {:.10} Hartree ({} iterations)",
                        k + 1,
                        outcome.energy.value(),
                        outcome.iterations
                    ));
                    sum += outcome.energy.value();
                }
                out.relaxed = Some(sum);
            }
        }
        Ok(out)
    }
}
