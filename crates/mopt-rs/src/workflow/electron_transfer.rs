//! The rate constant of an electron transfer from one structure: EDEEL's diabatic states on
//! the fragments of a [`EdeelLayout`], the equilibria of the initial and final states, the
//! crossings of the two states along the solvent coordinate of SAES (or one crossing without a
//! solvent), and at each crossing the coupling, the inner sphere along the difference gradient,
//! the tunnelling factor of Scher and Holstein, and the rate constants of Marcus theory (thesis
//! eqn (4.1)) and of Zhu-Nakamura theory at the crossing; the largest rate constant of each
//! along the scan (`docs/source/electron_transfer.md`).
//!
//! **Barriers.** The barrier of the thesis is that of the electronic energies with the solvent
//! activation, `dE = V11(SX) - V11(EQ_i) + G_SA^i`. With the thermal corrections
//! ([`TransferSettings::thermo`]) the free energy of activation is `dG = G(SX) - G(EQ_i)`
//! as well: the harmonic free energy of the initial state at its equilibrium (its normal
//! modes) and of the crossing from the vibrations within the seam of the two states
//! ([`crate::vib::seam_analysis`], the direction across the seam left out, as for the
//! crossing points of nonadiabatic transition state theory), with the solvent shift at the
//! crossing. The rate constants are given with both; [`apparent_barrier`] turns a rate
//! constant into the free energy of activation that the Eyring equation gives it.
//!
//! The procedure is that of chapter 4 of A. Mutsuji's thesis (Hokkaido University 2025, steps 3
//! to 6 of its Fig. 4.6; not published in a peer-reviewed journal): the crossing at `alpha_SA =
//! 1` first, then each smaller `alpha_SA` from the crossing before. The energies are electronic
//! energies, as there. The Zhu-Nakamura rate is new in mopt-rs.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use faer::Mat;

use crate::domain::{EvalRequest, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::kinetics::rate::couplings::{
    charge_transfer, fragment_charge_difference, generalized_mulliken_hush, maximize_on_grid,
    mulliken_hush, CT_THRESHOLD,
};
use crate::kinetics::rate::inner_sphere::{
    inner_sphere, marcus_levich_jortner, marcus_with_tunnelling, scher_holstein,
    zhu_nakamura_at_crossing, InnerSphere, TunnellingFactor, ZhuNakamuraRate,
};
use crate::kinetics::rate::solvent_activation::{
    decompose_barrier, BarrierDecomposition, SolventActivation,
};
use crate::math::linalg::col_norm;
use crate::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use crate::pes::combine::{
    edeel_states, edeel_with, EdeelLayout, EdeelStates, EdeelTarget, PesExpr, SurfaceEvaluation,
};
use crate::pes::term::TermContext;
use crate::vib::{normal_modes, seam_analysis, thermochemistry, SecondDirection, ThermoSettings};
use crate::workflow::driver::{run_optimization_with, CalculatorSet};
use crate::workflow::Observer;

/// The `alpha_SA` of the thesis's scan: 1.0 down to 0.0 in steps of 0.1.
pub fn thesis_alphas() -> Vec<f64> {
    (0..=10).map(|k| 1.0 - 0.1 * f64::from(k)).collect()
}

/// Where the coupling of a crossing comes from.
pub trait CouplingSource {
    /// A short name for the log and the table.
    fn name(&self) -> String;

    /// The coupling (Hartree) at the crossing `crossing`, whose initial diabatic energy is
    /// `v11` (Hartree).
    fn coupling(
        &self,
        crossing: &MolecularSystem,
        v11: f64,
        calcs: &CalculatorSet<'_>,
    ) -> MoptResult<f64>;
}

/// A coupling given as a number (Hartree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GivenCoupling(pub f64);

impl CouplingSource for GivenCoupling {
    fn name(&self) -> String {
        "given".to_string()
    }

    fn coupling(&self, _: &MolecularSystem, _: f64, _: &CalculatorSet<'_>) -> MoptResult<f64> {
        Ok(self.0)
    }
}

/// The adiabatic coupling of EDEEL, `V12 = V11 - E^C_{n+m+1}` with the whole structure in the
/// state with the moving electron (RSC Adv. 2023, 13, 32097).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdeelCoupling {
    /// The charge and multiplicity of the states, for the task of the whole structure.
    pub states: EdeelStates,
    /// The multiplicity of the whole structure with the moving electron.
    pub multiplicity: u32,
}

impl CouplingSource for EdeelCoupling {
    fn name(&self) -> String {
        "edeel".to_string()
    }

    fn coupling(
        &self,
        crossing: &MolecularSystem,
        v11: f64,
        calcs: &CalculatorSet<'_>,
    ) -> MoptResult<f64> {
        let mut expr = PesExpr::Task(self.states.coupling_task(self.multiplicity));
        let e = evaluate(&mut expr, crossing, calcs)?.total.energy.value();
        Ok(v11 - e)
    }
}

/// The Mulliken-Hush coupling of thesis eqn (4.18) from the excited states of the whole
/// structure with the moving electron at the crossing: the lowest state among `count` that
/// moves at least [`CT_THRESHOLD`] electrons, with its excitation energy, transition dipole,
/// and the transfer distance of the charge-transfer index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MullikenHushCoupling {
    /// The charge and multiplicity of the states (the structure with the moving electron is
    /// the complex less one unit of charge).
    pub states: EdeelStates,
    /// The multiplicity of the structure with the moving electron.
    pub multiplicity: u32,
    /// The excited states searched.
    pub count: usize,
}

impl MullikenHushCoupling {
    /// The coupling at `crossing`, or `None` when the calculator gives no transitions.
    fn at(&self, crossing: &MolecularSystem, calcs: &CalculatorSet<'_>) -> MoptResult<Option<f64>> {
        let calc = calcs.get(crate::pes::combine::DEFAULT_CALCULATOR)?;
        let mut sys = crossing.clone();
        sys.spin_charge = crate::domain::SpinCharge::new(
            self.states.complex_without.charge - 1,
            self.multiplicity,
        );
        let Some(transitions) = calc.transitions(&sys, self.count)? else {
            return Ok(None);
        };
        let positions: Vec<[f64; 3]> = sys.atoms().map(|a| sys.position(a).value()).collect();
        for t in transitions {
            let ct = charge_transfer(&positions, &t.population_change)?;
            if ct.charge >= CT_THRESHOLD {
                return mulliken_hush(t.excitation, t.transition_dipole, ct.distance).map(Some);
            }
        }
        Err(MoptError::calc(format!(
            "Mulliken-Hush coupling: none of the {} lowest excited states moves {CT_THRESHOLD} \
             electrons",
            self.count
        )))
    }
}

impl CouplingSource for MullikenHushCoupling {
    fn name(&self) -> String {
        "mulliken_hush".to_string()
    }

    fn coupling(
        &self,
        crossing: &MolecularSystem,
        _v11: f64,
        calcs: &CalculatorSet<'_>,
    ) -> MoptResult<f64> {
        self.at(crossing, calcs)?.ok_or_else(|| {
            MoptError::unsupported(
                "Mulliken-Hush coupling: the calculator gives no transitions of its excited \
                 states (-et_coupling edeel or a value)",
            )
        })
    }
}

/// The formula of an [`ExcitedStateCoupling`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcitedStateFormula {
    /// Generalized Mulliken-Hush (Cave and Newton): the transition dipole and the change of
    /// the dipole.
    GeneralizedMullikenHush,
    /// Fragment charge difference (Voityuk and Rosch): the transition density and the change
    /// of the populations on the donor and on the acceptor.
    FragmentChargeDifference,
}

impl ExcitedStateFormula {
    /// `gmh` or `fcd`.
    pub fn name(self) -> &'static str {
        match self {
            Self::GeneralizedMullikenHush => "gmh",
            Self::FragmentChargeDifference => "fcd",
        }
    }
}

/// The generalized Mulliken-Hush or the fragment-charge-difference coupling from the excited
/// states of the whole structure with the moving electron at the crossing: the charge-transfer
/// state as for [`MullikenHushCoupling`] (the lowest among `count` that moves at least
/// [`CT_THRESHOLD`] electrons) in the formula. Neither needs the two states to be localized.
#[derive(Debug, Clone, PartialEq)]
pub struct ExcitedStateCoupling {
    /// The charge and multiplicity of the states.
    pub states: EdeelStates,
    /// The multiplicity of the structure with the moving electron.
    pub multiplicity: u32,
    /// The excited states searched.
    pub count: usize,
    /// The formula.
    pub formula: ExcitedStateFormula,
    /// The atoms (0-based) of the donor and of the acceptor (the fragment charge difference
    /// counts the electrons on them; a bridge is neither).
    pub fragments: [Vec<usize>; 2],
}

impl CouplingSource for ExcitedStateCoupling {
    fn name(&self) -> String {
        self.formula.name().to_string()
    }

    fn coupling(
        &self,
        crossing: &MolecularSystem,
        _v11: f64,
        calcs: &CalculatorSet<'_>,
    ) -> MoptResult<f64> {
        let what = self.formula.name();
        let calc = calcs.get(crate::pes::combine::DEFAULT_CALCULATOR)?;
        let mut sys = crossing.clone();
        sys.spin_charge = crate::domain::SpinCharge::new(
            self.states.complex_without.charge - 1,
            self.multiplicity,
        );
        let Some(transitions) = calc.transitions(&sys, self.count)? else {
            return Err(MoptError::unsupported(format!(
                "{what} coupling: the calculator gives no transitions of its excited states"
            )));
        };
        let positions: Vec<[f64; 3]> = sys.atoms().map(|a| sys.position(a).value()).collect();
        for t in transitions {
            if charge_transfer(&positions, &t.population_change)?.charge < CT_THRESHOLD {
                continue;
            }
            return match self.formula {
                ExcitedStateFormula::GeneralizedMullikenHush => {
                    let change = t.dipole_change.ok_or_else(|| {
                        MoptError::unsupported(
                            "gmh coupling: the calculator does not give the change of the \
                             dipole of its excited states",
                        )
                    })?;
                    generalized_mulliken_hush(t.excitation, t.transition_dipole, change)
                }
                ExcitedStateFormula::FragmentChargeDifference => {
                    let tp = t.transition_population.as_ref().ok_or_else(|| {
                        MoptError::unsupported(
                            "fcd coupling: the calculator does not give the populations of the \
                             transition densities of its excited states",
                        )
                    })?;
                    fragment_charge_difference(
                        t.excitation,
                        tp,
                        &t.population_change,
                        &self.fragments[0],
                        &self.fragments[1],
                    )
                }
            };
        }
        Err(MoptError::calc(format!(
            "{what} coupling: none of the {} lowest excited states moves {CT_THRESHOLD} \
             electrons",
            self.count
        )))
    }
}

/// Mulliken-Hush when the calculator gives the transitions of its excited states, EDEEL's
/// coupling otherwise; which one is settled at the first crossing.
#[derive(Debug)]
pub struct AutoCoupling {
    /// The Mulliken-Hush coupling.
    pub mulliken_hush: MullikenHushCoupling,
    /// EDEEL's coupling.
    pub edeel: EdeelCoupling,
    /// Whether the calculator gave transitions (`None` before the first crossing).
    chosen: std::cell::Cell<Option<bool>>,
    /// The crossings where the calculator refused its transitions and EDEEL's coupling was
    /// taken.
    fallbacks: std::cell::Cell<usize>,
}

impl AutoCoupling {
    /// Both sources, nothing settled yet.
    pub fn new(mulliken_hush: MullikenHushCoupling, edeel: EdeelCoupling) -> Self {
        Self {
            mulliken_hush,
            edeel,
            chosen: std::cell::Cell::new(None),
            fallbacks: std::cell::Cell::new(0),
        }
    }
}

impl CouplingSource for AutoCoupling {
    fn name(&self) -> String {
        match self.chosen.get() {
            Some(true) if self.fallbacks.get() > 0 => {
                let n = self.fallbacks.get();
                format!(
                    "auto: mulliken_hush, and edeel where the calculator refused its transitions \
                     ({n} {})",
                    if n == 1 { "crossing" } else { "crossings" }
                )
            }
            Some(true) => "auto: mulliken_hush".to_string(),
            Some(false) => "auto: edeel (the calculator gives no transitions)".to_string(),
            None => "auto".to_string(),
        }
    }

    fn coupling(
        &self,
        crossing: &MolecularSystem,
        v11: f64,
        calcs: &CalculatorSet<'_>,
    ) -> MoptResult<f64> {
        if self.chosen.get() != Some(false) {
            match self.mulliken_hush.at(crossing, calcs) {
                Ok(Some(v)) => {
                    self.chosen.set(Some(true));
                    return Ok(v);
                }
                Ok(None) => self.chosen.set(Some(false)),
                // The calculator gives transitions but not at this structure (fractional
                // occupations, say): EDEEL's coupling here, and the reason with it.
                Err(MoptError::Unsupported(why)) => {
                    self.chosen.set(Some(true));
                    self.fallbacks.set(self.fallbacks.get() + 1);
                    return self.edeel.coupling(crossing, v11, calcs).map_err(|e| {
                        e.context(format!(
                            "EDEEL's coupling where Mulliken-Hush was refused ({why})"
                        ))
                    });
                }
                Err(e) => {
                    self.chosen.set(Some(true));
                    return Err(e);
                }
            }
        }
        self.edeel.coupling(crossing, v11, calcs)
    }
}

/// The settings of a run.
#[derive(Debug, Clone)]
pub struct TransferSettings {
    /// Temperature (K); the thesis uses 296 K.
    pub temperature: f64,
    /// The solvent reorganization energies `[lambda_s^i, lambda_s^f]` (Hartree); `None` in the
    /// gas phase (one crossing without solvent activation).
    pub solvent: Option<[f64; 2]>,
    /// The `alpha_SA` of the scan, in its order (each crossing starts from the one before).
    pub alphas: Vec<f64>,
    /// The optimizations of the equilibria (and of the seam model function before a crossing).
    pub optimizer: OptimizerSettings,
    /// The searches of the crossings on the gradient projection (`etmain` gives them the
    /// largest trust radius of a gradient projection with RS-I-RFO, as `optmain` does).
    pub crossing_optimizer: OptimizerSettings,
    /// The thermal corrections of the free energies of activation (`None`: the electronic
    /// barriers only); their temperature is [`Self::temperature`].
    pub thermo: Option<ThermoSettings>,
    /// How the crossings are searched.
    pub crossing_search: CrossingSearch,
}

/// The search for the minimum of the seam of the two (shifted) states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossingSearch {
    /// Gradient projection (Bearpark, Robb, Schlegel) from the start.
    Projection,
    /// First the minimum of the seam model function `(V11 + V22) / 2 + w (V11 - V22)^2`
    /// (Sato et al., J. Phys. Chem. A 2009, 113, 1704; a surface with an energy, on which the
    /// quasi-Newton steps hold), then gradient projection from there to the seam.
    PenaltyThenProjection {
        /// `w` (1/Hartree).
        weight: f64,
    },
}

impl CrossingSearch {
    /// `projection`, or `penalty` (with `w` = 20/Hartree, Sato's `alpha` = 0.05 Hartree) or
    /// `penalty:W`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        let t = text.trim().to_ascii_lowercase();
        match t.as_str() {
            "projection" | "gp" => Ok(Self::Projection),
            "penalty" => Ok(Self::PenaltyThenProjection { weight: 20.0 }),
            other => match other.strip_prefix("penalty:").map(str::parse::<f64>) {
                Some(Ok(w)) if w.is_finite() && w > 0.0 => {
                    Ok(Self::PenaltyThenProjection { weight: w })
                }
                _ => Err(MoptError::parse(format!(
                    "crossing search '{text}': projection, penalty, or penalty:W (W in 1/Hartree)"
                ))),
            },
        }
    }
}

/// The free energy of activation (Hartree) that the Eyring equation gives the unimolecular
/// rate constant `rate` (1/s) at `temperature` (K): `-k_B T ln(k h / k_B T)`.
pub fn apparent_barrier(rate: f64, temperature: f64) -> f64 {
    let kt = Units::BOLTZMANN_HARTREE_PER_K * temperature;
    let frequency = Units::BOLTZMANN_J_PER_K * temperature / Units::PLANCK_J_S;
    -kt * (rate / frequency).ln()
}

/// One crossing of the scan and what is computed there (Hartree unless stated).
#[derive(Debug, Clone)]
pub struct CrossingResult {
    /// `alpha_SA` (`None` without a solvent).
    pub alpha: Option<f64>,
    /// The crossing structure.
    pub system: MolecularSystem,
    /// Whether its optimization converged.
    pub converged: bool,
    /// The shifts `[G_SA^i, G_SA^f]`.
    pub shifts: [f64; 2],
    /// The diabatic energies without the shifts.
    pub v11: f64,
    /// The final diabatic energy without its shift.
    pub v22: f64,
    /// The inner-sphere barrier `V11(SX) - V11(EQ_i)`.
    pub inner_barrier: f64,
    /// The whole barrier, the inner sphere and the solvent activation.
    pub barrier: f64,
    /// The inner sphere along the difference gradient (`None` without an inner barrier).
    pub inner: Option<InnerSphere>,
    /// The coupling.
    pub coupling: Option<f64>,
    /// Whether the coupling is the mean of those of the other crossings (none of its own was
    /// found, as the thesis does when no charge-transfer state is found at a crossing).
    pub coupling_from_others: bool,
    /// Why the crossing has no coupling of its own.
    pub coupling_note: Option<String>,
    /// The total reorganization energy of eqn (4.17).
    pub reorganization: Option<f64>,
    /// The tunnelling factor.
    pub tunnelling: Option<TunnellingFactor>,
    /// The Marcus rate constant with the tunnelling factor (1/s).
    pub marcus: Option<f64>,
    /// The Zhu-Nakamura rate constant.
    pub zhu_nakamura: Option<ZhuNakamuraRate>,
    /// The barrier split into the solvent, the donor, and the acceptor.
    pub decomposition: BarrierDecomposition,
    /// The free energy of activation `G(SX) - G(EQ_i)` (with the thermal corrections).
    pub free_energy_barrier: Option<f64>,
    /// The imaginary frequencies within the seam (the crossing is then not a minimum of it,
    /// and they are left out of its free energy).
    pub seam_imaginary: usize,
    /// The Marcus rate constant with the tunnelling factor and the free energy of activation.
    pub marcus_free: Option<f64>,
    /// The Zhu-Nakamura rate constant with the free energy of activation.
    pub zhu_nakamura_free: Option<ZhuNakamuraRate>,
    /// The Marcus-Levich-Jortner rate constant of the mode of the crossing, the mean solvent
    /// reorganization, and the reaction energy `V22(EQ_f) - V11(EQ_i)` (`None` without a
    /// solvent: it needs a classical bath).
    pub marcus_levich_jortner: Option<f64>,
}

/// The result of a run.
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// The equilibrium of the initial state and its `V11`.
    pub initial: (MolecularSystem, f64),
    /// The equilibrium of the final state and its `V22`.
    pub final_state: (MolecularSystem, f64),
    /// The solvent coordinate (`None` in the gas phase).
    pub solvent: Option<SolventActivation>,
    /// The crossings in the order of the scan.
    pub crossings: Vec<CrossingResult>,
    /// The largest Marcus rate constant along the scan: `(alpha_SA, log10 k)`.
    pub best_marcus: Option<(f64, f64)>,
    /// The largest Zhu-Nakamura rate constant along the scan: `(alpha_SA, log10 k)`.
    pub best_zhu_nakamura: Option<(f64, f64)>,
    /// The harmonic free energy of the initial state at its equilibrium.
    pub initial_free_energy: Option<f64>,
    /// The largest Marcus rate constant with the free energies of activation.
    pub best_marcus_free: Option<(f64, f64)>,
    /// The largest Zhu-Nakamura rate constant with the free energies of activation.
    pub best_zhu_nakamura_free: Option<(f64, f64)>,
    /// The largest Marcus-Levich-Jortner rate constant along the scan.
    pub best_marcus_levich_jortner: Option<(f64, f64)>,
}

/// The surface `expr` at `sys`: energies, gradients, and the energies of its leaves.
pub fn evaluate(
    expr: &mut PesExpr,
    sys: &MolecularSystem,
    calcs: &CalculatorSet<'_>,
) -> MoptResult<SurfaceEvaluation> {
    let request = EvalRequest::ENERGY_GRAD;
    let tasks = expr.tasks(sys, &request)?;
    let results = calcs.evaluate(&tasks)?;
    let ctx = TermContext::at_iteration(0, sys);
    expr.evaluate(sys, &request, results, &ctx)
}

/// The energy, gradient, and Hessian of the surface `expr` at `sys`.
fn with_hessian(
    expr: &mut PesExpr,
    sys: &MolecularSystem,
    calcs: &CalculatorSet<'_>,
) -> MoptResult<(f64, Col<f64>, Mat<f64>)> {
    let request = EvalRequest::ENERGY_GRAD_HESS;
    let tasks = expr.tasks(sys, &request)?;
    let results = calcs.evaluate(&tasks)?;
    let ctx = TermContext::at_iteration(0, sys);
    let e = expr.evaluate(sys, &request, results, &ctx)?;
    let mut hessian = e.raw.hessian_bohr.ok_or_else(|| {
        MoptError::calc(
            "electron transfer: the surface returned no Hessian for the free energies \
             (-et_free_energy off for the electronic barriers only)",
        )
    })?;
    if let Some(terms) = e.term_hessian {
        hessian += terms;
    }
    Ok((e.total.energy.value(), e.total.gradient_bohr, hessian))
}

/// The harmonic free energy of the crossing `sx` of the two states (shifted by `shifts`)
/// from the vibrations within their seam, and the number of its imaginary frequencies.
fn crossing_free_energy(
    plain: &EdeelLayout,
    states: &EdeelStates,
    sx: &MolecularSystem,
    shifts: [f64; 2],
    thermo: &ThermoSettings,
    calcs: &CalculatorSet<'_>,
) -> MoptResult<(f64, usize)> {
    let [mut s1, mut s2] = edeel_states(plain, states)?;
    let (e1, g1, h1) = with_hessian(&mut s1, sx, calcs).context("V11 with its Hessian")?;
    let (e2, g2, h2) = with_hessian(&mut s2, sx, calcs).context("V22 with its Hessian")?;
    let difference = &g1 - &g2;
    let length = col_norm(&difference);
    let plane = vec![Col::from_fn(difference.nrows(), |i| difference[i] / length)];
    let seam = seam_analysis(
        sx,
        [e1 + shifts[0], e2 + shifts[1]],
        [&g1, &g2],
        [&h1, &h2],
        &plane,
        SecondDirection::FromHessians,
        true,
    )?;
    let th = thermochemistry(sx, &seam.modes, e1 + shifts[0], thermo)?;
    Ok((th.gibbs.total, th.imaginary))
}

/// The energy of the leaf `label` of an evaluation.
fn component(eval: &SurfaceEvaluation, label: &str) -> MoptResult<f64> {
    eval.components
        .iter()
        .find(|(l, _)| l == label)
        .map(|(_, e)| e.value())
        .ok_or_else(|| MoptError::calc(format!("electron transfer: no evaluation '{label}'")))
}

/// The diabatic states `[V11, V22]` (without shifts) at `sys`, with the energy of the donor
/// fragment with the moving electron.
struct Diabatic {
    v11: f64,
    v22: f64,
    g11: Col<f64>,
    g22: Col<f64>,
    donor_with: f64,
}

fn diabatic(
    layout: &EdeelLayout,
    states: &EdeelStates,
    sys: &MolecularSystem,
    calcs: &CalculatorSet<'_>,
) -> MoptResult<Diabatic> {
    let mut plain = layout.clone();
    plain.shifts = None;
    let [mut v11, mut v22] = edeel_states(&plain, states)?;
    let e11 = evaluate(&mut v11, sys, calcs).context("V11")?;
    let e22 = evaluate(&mut v22, sys, calcs).context("V22")?;
    Ok(Diabatic {
        v11: e11.total.energy.value(),
        v22: e22.total.energy.value(),
        g11: e11.total.gradient_bohr.clone(),
        g22: e22.total.gradient_bohr.clone(),
        donor_with: component(&e11, "donor_with")?,
    })
}

/// The mass-weighted gradient: each component over the square root of its atom's mass (amu).
fn mass_weighted(sys: &MolecularSystem, g: &Col<f64>) -> Vec<f64> {
    let root = sys.sqrt_masses_3n();
    (0..g.nrows()).map(|k| g[k] / root[k]).collect()
}

/// The optimization of `expr` from `start`: the structure and whether it converged.
pub(crate) fn optimize(
    expr: PesExpr,
    start: &MolecularSystem,
    settings: &OptimizerSettings,
    calcs: &CalculatorSet<'_>,
    observers: &mut [&mut dyn Observer],
) -> MoptResult<(MolecularSystem, bool)> {
    let mut opt = GeometryOptimizer::new(settings.clone(), start.clone())?;
    opt.set_surface(expr)?;
    let out = run_optimization_with(&mut opt, calcs, observers)?;
    Ok((out.system, out.termination == Some(Termination::Converged)))
}

/// Run the procedure from `start` (the structure of the initial state; `final_start` that of
/// the final state, `start` when `None`).
#[allow(clippy::too_many_arguments)]
pub fn run_transfer(
    layout: &EdeelLayout,
    states: &EdeelStates,
    coupling: &dyn CouplingSource,
    start: &MolecularSystem,
    final_start: Option<&MolecularSystem>,
    settings: &TransferSettings,
    calcs: &CalculatorSet<'_>,
    observers: &mut [&mut dyn Observer],
) -> MoptResult<TransferResult> {
    if !(settings.temperature.is_finite() && settings.temperature > 0.0) {
        return Err(MoptError::config(
            "electron transfer: the temperature must be positive",
        ));
    }
    if !layout.is_bridged() {
        return Err(MoptError::unsupported(
            "electron transfer between separate molecules: the procedure of thesis chapter 3 \
             (the initial and final states as the isolated molecules, the crossings at fixed \
             donor-acceptor distances) is not built yet; this takes a molecule of donor, \
             bridge, and acceptor",
        ));
    }
    let kt = Units::BOLTZMANN_HARTREE_PER_K * settings.temperature;
    let thermo = settings.thermo.map(|t| ThermoSettings {
        temperature: settings.temperature,
        multiplicity: start.spin_charge.multiplicity,
        ..t
    });
    let mut plain = layout.clone();
    plain.shifts = None;
    // The equilibria of the two states.
    let (eq_i, _) = optimize(
        edeel_with(&plain, states, EdeelTarget::Initial)?,
        start,
        &settings.optimizer,
        calcs,
        observers,
    )
    .context("the equilibrium of the initial state")?;
    let (eq_f, _) = optimize(
        edeel_with(&plain, states, EdeelTarget::Final)?,
        final_start.unwrap_or(start),
        &settings.optimizer,
        calcs,
        observers,
    )
    .context("the equilibrium of the final state")?;
    let at_i = diabatic(&plain, states, &eq_i, calcs)?;
    let at_f = diabatic(&plain, states, &eq_f, calcs)?;
    let initial_free_energy = match &thermo {
        Some(t) => {
            let [mut v11, _] = edeel_states(&plain, states)?;
            let (e, _, h) = with_hessian(&mut v11, &eq_i, calcs)
                .context("the Hessian of the initial state at its equilibrium")?;
            let modes = normal_modes(&eq_i, &h, true)?;
            Some(thermochemistry(&eq_i, &modes, e, t)?.gibbs.total)
        }
        None => None,
    };
    let solvent = match settings.solvent {
        Some([li, lf]) => Some(SolventActivation::from_initial_equilibrium(
            li, lf, at_i.v11, at_i.v22,
        )?),
        None => None,
    };
    let alphas: Vec<Option<f64>> = match solvent {
        Some(_) => settings.alphas.iter().map(|&a| Some(a)).collect(),
        None => vec![None],
    };
    let mean_solvent = solvent.map_or(0.0, |s| s.mean_reorganization());
    let mut crossings = Vec::new();
    let mut from = eq_i.clone();
    for alpha in alphas {
        let shifts = match (solvent, alpha) {
            (Some(s), Some(a)) => s.shifts(a)?,
            _ => [0.0, 0.0],
        };
        let shifted = plain.clone().shifted(shifts);
        // At alpha_SA = 1 the solvent carries the whole activation: the shifts put the crossing
        // at the initial equilibrium (the definition of dG_s0 here), which is no search.
        let at_initial = matches!(alpha, Some(a) if (a - 1.0).abs() < 1e-12);
        let start = match settings.crossing_search {
            _ if at_initial => eq_i.clone(),
            CrossingSearch::Projection => from.clone(),
            CrossingSearch::PenaltyThenProjection { weight } => {
                optimize(
                    edeel_with(&shifted, states, EdeelTarget::Penalty { sigma: weight })?,
                    &from,
                    &settings.optimizer,
                    calcs,
                    observers,
                )
                .context("the minimum of the seam model function")?
                .0
            }
        };
        let (sx, converged) = if at_initial {
            (start, true)
        } else {
            optimize(
                edeel_with(&shifted, states, EdeelTarget::Crossing)?,
                &start,
                &settings.crossing_optimizer,
                calcs,
                observers,
            )
            .context(match alpha {
                Some(a) => format!("the crossing at alpha_SA = {a}"),
                None => "the crossing".to_string(),
            })?
        };
        from = sx.clone();
        let d = diabatic(&plain, states, &sx, calcs)?;
        let inner_barrier = d.v11 - at_i.v11;
        let barrier = inner_barrier + shifts[0];
        let inner = if inner_barrier > 1e-8 {
            Some(inner_sphere(
                &mass_weighted(&sx, &d.g11),
                &mass_weighted(&sx, &d.g22),
                inner_barrier,
            )?)
        } else {
            None
        };
        let (free_energy_barrier, seam_imaginary) = match (&thermo, initial_free_energy) {
            (Some(t), Some(g_i)) => {
                let (g_sx, imaginary) = crossing_free_energy(&plain, states, &sx, shifts, t, calcs)
                    .context("the free energy of the crossing")?;
                (Some(g_sx - g_i), imaginary)
            }
            _ => (None, 0),
        };
        let (v12, coupling_note) = match coupling.coupling(&sx, d.v11, calcs) {
            Ok(v) => (Some(v), None),
            Err(e) => (None, Some(e.to_string())),
        };
        let reorganization = inner.map_or(mean_solvent, |p| p.reorganization + mean_solvent);
        let reorganization = (reorganization > 0.0).then_some(reorganization);
        let tunnelling = match inner {
            Some(p) if p.reorganization > 0.0 => Some(scher_holstein(
                p.quantum,
                p.reaction_energy,
                p.reorganization,
                kt,
            )?),
            _ => None,
        };
        crossings.push(CrossingResult {
            alpha,
            system: sx,
            converged,
            shifts,
            v11: d.v11,
            v22: d.v22,
            inner_barrier,
            barrier,
            inner,
            coupling: v12,
            coupling_from_others: false,
            coupling_note,
            reorganization,
            tunnelling,
            marcus: None,
            zhu_nakamura: None,
            decomposition: decompose_barrier(
                d.v11,
                at_i.v11,
                d.donor_with,
                at_i.donor_with,
                shifts[0],
            ),
            free_energy_barrier,
            seam_imaginary,
            marcus_free: None,
            zhu_nakamura_free: None,
            marcus_levich_jortner: None,
        });
    }
    // A crossing without a coupling of its own takes the mean of the others; then the rates.
    let found: Vec<f64> = crossings.iter().filter_map(|c| c.coupling).collect();
    let mean = (!found.is_empty()).then(|| found.iter().sum::<f64>() / found.len() as f64);
    for c in &mut crossings {
        if c.coupling.is_none() {
            if let Some(m) = mean {
                c.coupling = Some(m);
                c.coupling_from_others = true;
            }
        }
        let Some(v) = c.coupling else { continue };
        let ln_gamma = c.tunnelling.map_or(0.0, |t| t.ln_gamma);
        if let (Some(p), true) = (c.inner, mean_solvent > 0.0) {
            c.marcus_levich_jortner = marcus_levich_jortner(
                v,
                p.quantum,
                p.reorganization,
                mean_solvent,
                at_f.v22 - at_i.v11,
                kt,
            )
            .ok();
        }
        if let Some(l) = c.reorganization {
            c.marcus = Some(marcus_with_tunnelling(ln_gamma, v, l, c.barrier, kt)?);
            if let Some(g) = c.free_energy_barrier {
                c.marcus_free = Some(marcus_with_tunnelling(ln_gamma, v, l, g, kt)?);
            }
        }
        if let Some(p) = c.inner {
            c.zhu_nakamura =
                zhu_nakamura_at_crossing(&p, v, c.barrier, c.inner_barrier, kt, 1.0).ok();
            if let Some(g) = c.free_energy_barrier {
                // The free energies hold the mode: no partition function of it divides.
                c.zhu_nakamura_free = c.zhu_nakamura.map(|z| ZhuNakamuraRate {
                    rate: z.with_free_energy(g, kt),
                    ..z
                });
            }
        }
    }
    let along = |f: &dyn Fn(&CrossingResult) -> Option<f64>| -> Option<(f64, f64)> {
        let points: Vec<(f64, f64)> = crossings
            .iter()
            .filter_map(|c| Some((c.alpha.unwrap_or(0.0), f(c)?.log10())))
            .collect();
        maximize_on_grid(&points)
    };
    let best_marcus = along(&|c| c.marcus);
    let best_zhu_nakamura = along(&|c| c.zhu_nakamura.map(|z| z.rate));
    let best_marcus_free = along(&|c| c.marcus_free);
    let best_zhu_nakamura_free = along(&|c| c.zhu_nakamura_free.map(|z| z.rate));
    let best_marcus_levich_jortner = along(&|c| c.marcus_levich_jortner);
    Ok(TransferResult {
        initial: (eq_i, at_i.v11),
        final_state: (eq_f, at_f.v22),
        solvent,
        crossings,
        best_marcus,
        best_zhu_nakamura,
        initial_free_energy,
        best_marcus_free,
        best_zhu_nakamura_free,
        best_marcus_levich_jortner,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{AtomId, Element, SpinCharge};
    use crate::pes::native::Gfn1Xtb;
    use crate::pes::oniom::layout::LinkPlacement;

    /// The radical anion of 1,4-pentadiene (Angstrom), slightly out of plane.
    fn pentadiene_anion() -> MolecularSystem {
        let rows: [(&str, [f64; 3]); 13] = [
            ("C", [-2.50, 1.00, 0.02]),
            ("C", [-1.25, 0.50, -0.01]),
            ("C", [0.00, 1.30, 0.00]),
            ("C", [1.25, 0.50, 0.01]),
            ("C", [2.50, 1.00, -0.02]),
            ("H", [-2.60, 2.08, 0.03]),
            ("H", [-3.42, 0.42, -0.02]),
            ("H", [-1.15, -0.58, 0.00]),
            ("H", [0.00, 1.95, 0.88]),
            ("H", [0.00, 1.95, -0.88]),
            ("H", [1.15, -0.58, 0.00]),
            ("H", [2.60, 2.08, -0.03]),
            ("H", [3.42, 0.42, 0.02]),
        ];
        let elements: Vec<Element> = rows
            .iter()
            .map(|(s, _)| Element::parse_token(s).unwrap())
            .collect();
        let x: Vec<f64> = rows
            .iter()
            .flat_map(|(_, p)| p.map(|c| c * Units::ANGSTROM_TO_BOHR))
            .collect();
        MolecularSystem::from_flat_bohr(
            elements,
            &Col::from_fn(x.len(), |i| x[i]),
            SpinCharge::new(-1, 2),
            None,
        )
        .unwrap()
    }

    fn atoms(list: &[usize]) -> Vec<AtomId> {
        list.iter()
            .map(|&k| AtomId::from_one_based(k).unwrap())
            .collect()
    }

    #[test]
    fn the_diabatic_states_of_a_bridged_molecule_have_the_gradients_of_their_energies() {
        let sys = pentadiene_anion();
        let layout = EdeelLayout::bridged(
            &sys,
            &atoms(&[1, 2, 6, 7, 8]),
            &atoms(&[3, 9, 10]),
            &atoms(&[4, 5, 11, 12, 13]),
            LinkPlacement::Scaled,
        )
        .unwrap();
        let states = EdeelStates {
            complex_without: SpinCharge::new(0, 1),
            donor_without: SpinCharge::new(0, 1),
            donor_with: SpinCharge::new(-1, 2),
            acceptor_without: SpinCharge::new(0, 1),
            acceptor_with: SpinCharge::new(-1, 2),
        };
        let calc = Gfn1Xtb::default();
        let calcs = CalculatorSet::single(&calc);
        let x0 = sys.coords_flat_bohr();
        let h = 1e-4;
        for (which, mut expr) in edeel_states(&layout, &states)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            let at = evaluate(&mut expr, &sys, &calcs).unwrap();
            let mut worst = 0.0_f64;
            // The atoms of the cut bonds (2, 3, 4) and their caps' neighbours, and others.
            for k in [3, 4, 6, 7, 8, 9, 10, 11, 1, 14, 25, 33] {
                let energy = |s: f64| {
                    let mut x = x0.clone();
                    x[k] += s;
                    let moved = sys.with_flat_coords_bohr(&x).unwrap();
                    let [v11, v22] = edeel_states(&layout, &states).unwrap();
                    let mut e = if which == 0 { v11 } else { v22 };
                    evaluate(&mut e, &moved, &calcs)
                        .unwrap()
                        .total
                        .energy
                        .value()
                };
                let numeric = (energy(h) - energy(-h)) / (2.0 * h);
                worst = worst.max((numeric - at.total.gradient_bohr[k]).abs());
            }
            assert!(
                worst < 2e-6,
                "V{}{}: largest difference {worst:e}",
                which + 1,
                which + 1
            );
        }
    }

    /// Two atoms, donor and acceptor, whose excited states are given: one that moves too little
    /// charge, then the charge-transfer state of a two-state model with the coupling `V`.
    struct TwoSites {
        extras: bool,
    }

    impl crate::pes::Calculator for TwoSites {
        fn name(&self) -> &str {
            "two_sites"
        }

        fn evaluate(
            &self,
            sys: &MolecularSystem,
            _req: &EvalRequest,
        ) -> MoptResult<crate::domain::EvalOutput> {
            Ok(crate::domain::EvalOutput::energy_gradient(
                crate::domain::Quantity::new(0.0),
                Col::zeros(sys.dof()),
            ))
        }

        fn transitions(
            &self,
            _sys: &MolecularSystem,
            _count: usize,
        ) -> MoptResult<Option<Vec<crate::pes::states::Transition>>> {
            let (eps, v, r): (f64, f64, f64) = (0.01, 0.004, 12.0);
            let gap = (eps * eps + 4.0 * v * v).sqrt();
            let theta = 0.5 * (2.0 * v).atan2(eps);
            let (c, s) = (theta.cos(), theta.sin());
            let local = crate::pes::states::Transition {
                excitation: 0.002,
                transition_dipole: [0.1, 0.0, 0.0],
                population_change: vec![-0.3, 0.3],
                dipole_change: Some([0.0, 0.0, 1.0]),
                transition_population: Some(vec![0.1, -0.1]),
            };
            // The charge-transfer state of the model moves c^2 - s^2 electrons; here it moves a
            // whole one (to pass the threshold) with the transition quantities of the model, so
            // the formulas give the values computed in the test, not V itself.
            let moved = 1.0;
            let ct = crate::pes::states::Transition {
                excitation: gap,
                transition_dipole: [0.0, 0.0, -r * c * s],
                population_change: vec![-moved, moved],
                dipole_change: self.extras.then_some([0.0, 0.0, -r * moved]),
                transition_population: self.extras.then(|| vec![-c * s, c * s]),
            };
            Ok(Some(vec![local, ct]))
        }
    }

    #[test]
    fn gmh_and_fcd_take_the_charge_transfer_state() {
        let sys = MolecularSystem::from_flat_bohr(
            vec![Element::C, Element::C],
            &Col::from_fn(6, |i| if i == 5 { 12.0 } else { 0.0 }),
            SpinCharge::new(-1, 2),
            None,
        )
        .unwrap();
        let states = EdeelStates {
            complex_without: SpinCharge::new(0, 1),
            donor_without: SpinCharge::new(0, 1),
            donor_with: SpinCharge::new(-1, 2),
            acceptor_without: SpinCharge::new(0, 1),
            acceptor_with: SpinCharge::new(-1, 2),
        };
        let (eps, v): (f64, f64) = (0.01, 0.004);
        let gap = (eps * eps + 4.0 * v * v).sqrt();
        let sin2 = 2.0 * v / gap;
        let model = TwoSites { extras: true };
        let calcs = CalculatorSet::single(&model);
        let source = |formula| ExcitedStateCoupling {
            states,
            multiplicity: 2,
            count: 4,
            formula,
            fragments: [vec![0], vec![1]],
        };
        // Only the second state passes the threshold. GMH: mu_12 = r c s along d mu, |d mu| =
        // r, so V = mu_12 E / sqrt(r^2 + 4 mu_12^2).
        let gmh = source(ExcitedStateFormula::GeneralizedMullikenHush)
            .coupling(&sys, 0.0, &calcs)
            .unwrap();
        let mu12 = 12.0 * 0.5 * sin2;
        let want = mu12 * gap / (144.0 + 4.0 * mu12 * mu12).sqrt();
        assert!((gmh - want).abs() < 1e-15, "{gmh} vs {want}");
        // FCD: dq_12 = sin 2 theta, dq_1 - dq_2 = 2.
        let fcd = source(ExcitedStateFormula::FragmentChargeDifference)
            .coupling(&sys, 0.0, &calcs)
            .unwrap();
        let want = gap * sin2 / (4.0 + 4.0 * sin2 * sin2).sqrt();
        assert!((fcd - want).abs() < 1e-15, "{fcd} vs {want}");
        // Without the dipole change or the transition populations: refused, and why.
        let bare = TwoSites { extras: false };
        let calcs = CalculatorSet::single(&bare);
        for formula in [
            ExcitedStateFormula::GeneralizedMullikenHush,
            ExcitedStateFormula::FragmentChargeDifference,
        ] {
            let e = source(formula).coupling(&sys, 0.0, &calcs).unwrap_err();
            assert!(matches!(e, MoptError::Unsupported(_)), "{e}");
        }
    }
}
