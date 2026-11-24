//! GFN1-xTB (S. Grimme, C. Bannwarth, P. Shushkov, J. Chem. Theory Comput. 2017, 13, 1989) for
//! H to Rn (every element of GFN1-xTB), built into mopt-rs.
//!
//! Provenance: the method and its parameters follow xtb 6.7.1 (grimme-lab/xtb,
//! LGPL-3.0-or-later; `src/xtb/gfn1.f90`, `hamiltonian.F90`, `basisset.f90`, `coulomb.f90`,
//! `thirdorder.f90`, `halogen.f90`, `repulsion.F90`, `src/scc_core.f90`, `src/disp/dftd3*`),
//! with tblite 0.4.0 (LGPL-3.0-or-later) as the numerical reference where the two differ
//! ([`params`]; `third_party/DISPERSION_AND_PARAMETERS.md` ("GFN1-xTB")). Other elements are refused.
//!
//! Energy: `E = sum P H0 + E2 + E3 - TS + E_rep + E_disp + E_XB`.
//!
//! - `E2 = 1/2 sum_ab q_a gamma_ab q_b` over Mulliken shell charges `q_a` (reference
//!   occupation minus population);
//! - `E3 = 1/3 sum_A Gamma_A q_A^3` over atomic charges;
//! - `-TS`: the electronic entropy of the Fermi smearing at 300 K, per spin channel;
//! - `E_rep`, `E_XB`: repulsion and the halogen-bond correction (module `classical`);
//!   `E_disp`: D3(BJ) (s8 = 2.4, a1 = 0.63, a2 = 5 Bohr, no
//!   three-body term) with mopt-rs's [`crate::pes::dispersion::d3`].
//!
//! The charges are made self-consistent (the Fock matrix
//! `F = H0 - 1/2 S (v_mu + v_nu)` with the shell potentials `v_a = sum_b gamma_ab q_b +
//! Gamma_A q_A^2`) from the charges of the last structure with the same atoms and state, and
//! from the superposition of atomic densities otherwise (every shell at its reference
//! occupation, the total charge spread over the atoms; [`SccGuess::Previous`], the default),
//! or always from the superposition ([`SccGuess::Sad`]: a result then does not depend on what
//! was computed before, at two to five times the iterations along an optimization). The Fock
//! matrices are extrapolated by ADIIS, and by CDIIS once ADIIS does not look like converging
//! (no new lowest commutator for `diis_patience` iterations, or a commutator that has not
//! fallen to `diis_rate^diis_window` of its value over the last `diis_window` iterations;
//! [`crate::pes::native::scf_diis`]), the input density damped while ADIIS runs
//! (`-calc_opt scc_guess=previous|sad`, `diis_patience=N`, `diis_depth=N`, `diis_damping=B`,
//! `diis_rate=R|off`, `diis_window=N`, `charge_limit=E`; the defaults are measured,
//! [`DIIS_DEPTH`], [`DIIS_RATE`]). A run that does not converge
//! or runs away to an atomic charge beyond `charge_limit` is repeated from the start with
//! stronger damping, and refused if it still does: the third-order term has no lower bound, and
//! without the guard POCl3 or BF4- can end far below their physical solutions at charges no
//! molecule has. With d-block metals the extrapolated Fock matrices can also settle on another
//! self-consistent solution (a titanium-ruthenium hydride: -6.5436 Hartree with Ti at -1.94 e,
//! where tblite has -6.7891 Hartree with Ti at +0.54 e; SrCl2: Sr at -2.10 e, 0.551 Hartree
//! above tblite's solution): for a structure with a d-block metal, a lanthanide, or Ca, Sr, Ba,
//! and for a run that fails or runs away, the shell charges are also mixed by the modified
//! Broyden method (D. D. Johnson, Phys. Rev. B 1988, 38, 12807; mixing 0.4 as tblite), and the
//! converged solution of lower energy is kept (`-calc_opt scc_broyden=auto|on|off`,
//! [`SccBroyden`]). Converged when no shell charge changes by more than `charge_tolerance` and no
//! element of the commutator exceeds [`COMMUTATOR_TOLERANCE`]. The gradient is analytic
//! (energy-weighted density matrix, overlap and Hamiltonian derivatives, the
//! coordination-number dependence of the self energies, the Coulomb kernel at fixed charges,
//! and the classical terms); so is the Hessian (module `hessian`: the derivatives at fixed
//! density and charges and the coupled-perturbed response of the charges), or central
//! differences of the gradient with `-calc_opt hessian=numerical`.
//!
//! Point charges (electronic embedding, [`Calculator::evaluate_in_charges`]) act on the shell
//! charges as in xtb's embedding of GFN1-xTB (`src/embedding.f90`, `jpot_pcem_gfn1`): every
//! shell `a` sees `V_a = sum_K q_K / sqrt(r_aK^2 + 1/g_aK^2)` with `g_aK = 2 / (1/eta_a +
//! 1/eta_K)`, `eta_a` the hardness of the shell and `eta_K` that of the element of the charge
//! (nitrogen for a charge of no element or of an element without parameters, as xtb's dummy
//! atom). The potential enters the Fock matrix with the others and the energy as
//! `sum_a q_a V_a`; its gradient goes to the atoms and to the charges.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy calls tblite for GFN1-xTB).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod classical;
mod elements;
mod hamiltonian;
mod hessian;
pub mod params;
#[cfg(test)]
mod tests;

use std::sync::Mutex;

use faer::{Col, Mat};

use self::hamiltonian::{diff, dot, gamma_derivative, pair_scale, Coordination};
use self::params::{Gfn1Element, BOLTZMANN_HARTREE_PER_K, ELECTRONIC_TEMPERATURE_K};
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, OrbitalSet, Orbitals, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::pes::calculator::Calculator;
use crate::pes::dispersion::d3::D3Bj;
use crate::pes::native::gfn0::basis::{overlap, pair_overlap_gradient, Basis};
use crate::pes::native::gfn0::electronic::{
    lowdin_orthogonalizer, solve_at, solve_density_at, Electronic,
};
use crate::pes::native::gfn0::excited;
use crate::pes::native::scf_diis::{commutator, largest, FockDiis};
use crate::pes::native::tda_gradient;
use crate::pes::point_charges::{EmbeddedOutput, PointCharges};

/// Fock matrices kept by the SCC extrapolation. The defaults of the depth, the patience and
/// the damping are measured on the 37 systems of `tests/golden/gfn1_xtb.json`: with depth 6,
/// patience 2 and damping 0.5 every system converges to tblite's solution in 1266 iterations
/// together (83 at most); without damping POCl3 runs away (P -16 e, 130 Hartree below the
/// physical solution), and ADIIS alone (no switch to CDIIS) fails for 7 of the 37.
pub const DIIS_DEPTH: usize = 6;
/// Iterations without a new lowest commutator after which CDIIS takes over.
pub const DIIS_PATIENCE: usize = 2;
/// Damping of the input density while ADIIS runs.
pub const DIIS_DAMPING: f64 = 0.5;
/// The slowest mean fall of the commutator per iteration that ADIIS may keep before CDIIS
/// takes over (`None`: only stalling switches). Measured on the same 37 systems with depth 6,
/// patience 2 and damping 0.5: damped ADIIS converges, but linearly (the commutator x0.6 per
/// iteration), so with stalling alone CDIIS hardly ever takes over (1266 iterations, 83 at
/// most; 35 for the Diels-Alder guess of butadiene and ethylene from SAD, 43 per warm start).
/// With the rate 0.3 over 3 iterations: 610 iterations, 36 at most, 16 from SAD and 19 per
/// warm start, every system on the same solution and BF4- on the physical one; rates 0.2
/// and 0.5 give 606 and 723, 0.7 gives 850. Over 2 iterations POCl3 converges to another
/// solution, 1.38 Hartree above.
pub const DIIS_RATE: Option<f64> = Some(0.3);
/// The iterations over which [`DIIS_RATE`] is judged.
pub const DIIS_WINDOW: usize = 3;
/// The stronger dampings of the repeated runs.
pub const RETRY_DAMPING: [f64; 2] = [0.7, 0.9];
/// The largest atomic charge (e) a converged SCC may have before it is taken as a runaway: a
/// safeguard, far beyond the charges of the reference systems (0.91 e at most).
pub const CHARGE_LIMIT: f64 = 4.0;

/// The symbols of the elements GFN1-xTB in mopt-rs has parameters for (H to Rn), in
/// the order of its table of elements.
pub fn element_symbols() -> Vec<&'static str> {
    params::ELEMENTS
        .iter()
        .filter_map(|e| crate::domain::Element::from_atomic_number(e.z).ok())
        .map(crate::domain::Element::symbol)
        .collect()
}

/// Energy terms (Hartree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gfn1Energy {
    /// Total energy.
    pub total: f64,
    /// Electronic free energy (`sum P H0 + E2 + E3 - TS`).
    pub electronic: f64,
    /// Second-order Coulomb energy.
    pub coulomb: f64,
    /// Third-order energy.
    pub third_order: f64,
    /// `-TS` of the Fermi smearing.
    pub entropy: f64,
    /// Repulsion.
    pub repulsion: f64,
    /// D3 dispersion.
    pub dispersion: f64,
    /// Halogen-bond correction.
    pub halogen_bond: f64,
    /// Interaction of the shell charges with point charges (`sum_a q_a V_a`; 0 without).
    pub embedding: f64,
}

/// A GFN1-xTB result.
#[derive(Debug, Clone)]
pub struct Gfn1Result {
    /// Energy terms.
    pub energy: Gfn1Energy,
    /// Gradient (Hartree/Bohr, 3N).
    pub gradient: Col<f64>,
    /// Mulliken atomic charges.
    pub charges: Vec<f64>,
    /// SCC iterations.
    pub iterations: usize,
    /// Gradient of the energy with respect to the positions of the point charges.
    pub charge_gradient: Vec<[f64; 3]>,
}

/// The GFN1-xTB calculator.
#[derive(Debug)]
pub struct Gfn1Xtb {
    /// Displacement of the central-difference Hessian (Bohr).
    pub hessian_step_bohr: f64,
    /// Largest change of a shell charge at convergence of the SCC.
    pub charge_tolerance: f64,
    /// Largest number of SCC iterations.
    pub max_iterations: usize,
    /// The analytic Hessian (true, the default) or central differences of the gradient.
    pub analytic_hessian: bool,
    /// Where the SCC starts.
    pub guess: SccGuess,
    /// Fock matrices kept by the extrapolation.
    pub diis_depth: usize,
    /// Iterations without a new lowest commutator after which CDIIS takes over from ADIIS.
    pub diis_patience: usize,
    /// While ADIIS runs, the share of the last input density kept in the next one (0: none).
    pub diis_damping: f64,
    /// The slowest mean fall of the commutator per iteration that ADIIS may keep before
    /// CDIIS takes over (`None`: only stalling switches).
    pub diis_rate: Option<f64>,
    /// The iterations over which `diis_rate` is judged.
    pub diis_window: usize,
    /// The largest atomic charge (e) of a converged SCC before it counts as a runaway.
    pub charge_limit: f64,
    /// When the shell charges are also mixed by the modified Broyden method.
    pub broyden: SccBroyden,
    /// The excited states (`EvalRequest::excited_state > 0`).
    pub excited: ExcitedStates,
    /// Displacement of the central differences of the excitation energies (Bohr).
    pub excitation_step_bohr: f64,
    /// The gradient of an excited state: analytic (true, the default; a Z-vector for the
    /// response of the orbitals, `pes::native::tda_gradient`) or central differences of the
    /// excitation energy that follow the state.
    pub analytic_excited: bool,
    /// The converged shell charges of the last structure, with its atomic numbers (the start
    /// of the next SCC with [`SccGuess::Previous`]).
    last: Mutex<Option<(Vec<u8>, i32, u32, Vec<f64>)>>,
}

/// When the SCC of GFN1-xTB also mixes the shell charges by the modified Broyden method and
/// keeps the converged solution of lower energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SccBroyden {
    /// For structures with a d-block metal, a lanthanide, or Ca, Sr, Ba, and when the
    /// extrapolation of the Fock matrices fails or runs away (the default).
    #[default]
    Auto,
    /// Always.
    On,
    /// Only when the extrapolation fails or runs away.
    Off,
}

impl SccBroyden {
    /// `auto`, `on`, or `off`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "always" | "true" => Ok(Self::On),
            "off" | "never" | "false" => Ok(Self::Off),
            other => Err(MoptError::config(format!(
                "gfn1xtb: scc_broyden must be auto, on, or off, got '{other}'"
            ))),
        }
    }
}

/// The mixing of the modified Broyden method (tblite's damping).
pub const BROYDEN_MIXING: f64 = 0.4;
/// The weight of the first iteration in the modified Broyden method (Johnson's `w0`).
const BROYDEN_W0: f64 = 0.01;

/// Whether `z` is a metal with d shells of the kind that `scc_broyden=auto` mixes by Broyden's
/// method: the d-block metals (Sc to Zn, Y to Cd, La to Hg with the lanthanides, which carry
/// 5d shells in GFN1-xTB) and the alkaline-earth metals Ca, Sr, Ba, whose empty d shells let
/// the extrapolated Fock matrices settle on a solution with the metal as an anion. Measured on
/// the 119 systems of `tests/golden/gfn1_xtb.json`: without Broyden mixing SrCl2 converges to
/// Sr at -2.10 e, 0.551 Hartree above tblite's solution (Sr +0.98 e), and HfCl4 0.604
/// Hartree above it; with Broyden mixing for every system all agree with tblite, at 7439
/// iterations in all instead of 5457. Mixing for these metals only costs the extra
/// iterations of the systems that have one.
fn is_d_block(z: u8) -> bool {
    matches!(z, 20..=30 | 38..=48 | 56..=80)
}

/// Where the SCC of GFN1-xTB starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SccGuess {
    /// Always the superposition of atomic densities: the reference occupations, the total
    /// charge spread over the atoms (the result does not depend on what was computed before).
    Sad,
    /// The converged charges of the last structure with the same atoms and state, the
    /// superposition otherwise (the default: fewer iterations along an optimization).
    #[default]
    Previous,
}

/// The largest commutator element at convergence of the SCC.
pub const COMMUTATOR_TOLERANCE: f64 = 1e-7;

/// The excited singlet states of GFN1-xTB (`-calc_opt excited=tda|stda`), for tests of the
/// methods that work on several states rather than for photochemistry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExcitedStates {
    /// The Tamm-Dancoff approximation of the linear response of its self-consistent charges:
    /// its own kernel, nothing empirical beyond the ground state (the default).
    #[default]
    Tda,
    /// The simplified Tamm-Dancoff approximation (sTDA, `a_x = 0`) on its orbitals, as for
    /// GFN0-xTB, with the chemical hardnesses of GFN1-xTB.
    Stda,
}

impl Clone for Gfn1Xtb {
    fn clone(&self) -> Self {
        Self {
            hessian_step_bohr: self.hessian_step_bohr,
            charge_tolerance: self.charge_tolerance,
            max_iterations: self.max_iterations,
            analytic_hessian: self.analytic_hessian,
            guess: self.guess,
            diis_depth: self.diis_depth,
            diis_patience: self.diis_patience,
            diis_damping: self.diis_damping,
            broyden: self.broyden,
            diis_rate: self.diis_rate,
            diis_window: self.diis_window,
            charge_limit: self.charge_limit,
            excited: self.excited,
            excitation_step_bohr: self.excitation_step_bohr,
            analytic_excited: self.analytic_excited,
            last: Mutex::new(None),
        }
    }
}

impl Default for Gfn1Xtb {
    fn default() -> Self {
        Self {
            hessian_step_bohr: 5.0e-3,
            charge_tolerance: 1.0e-10,
            max_iterations: 250,
            analytic_hessian: true,
            guess: SccGuess::Previous,
            diis_depth: DIIS_DEPTH,
            diis_patience: DIIS_PATIENCE,
            diis_damping: DIIS_DAMPING,
            broyden: SccBroyden::Auto,
            diis_rate: DIIS_RATE,
            diis_window: DIIS_WINDOW,
            charge_limit: CHARGE_LIMIT,
            excited: ExcitedStates::Tda,
            excitation_step_bohr: crate::pes::native::gfn0::DEFAULT_EXCITATION_STEP_BOHR,
            analytic_excited: true,
            last: Mutex::new(None),
        }
    }
}

/// Everything of a converged SCC.
struct Scc {
    el: Vec<&'static Gfn1Element>,
    x: Vec<[f64; 3]>,
    basis: Basis,
    s: Mat<f64>,
    h0: Mat<f64>,
    coord: Coordination,
    levels: Vec<f64>,
    eta: Vec<f64>,
    gamma: Mat<f64>,
    q_shell: Vec<f64>,
    q_atom: Vec<f64>,
    potential: Vec<f64>,
    /// The potential of the point charges at the shells (zeros without).
    external: Vec<f64>,
    elec: Electronic,
    entropy: f64,
    unpaired: usize,
    iterations: usize,
}

/// The hardness of every point charge: that of its element, nitrogen's when it has none or
/// one without parameters.
fn charge_hardness(charges: &PointCharges) -> MoptResult<Vec<f64>> {
    let nitrogen = params::element(crate::domain::Element::N)?.eta;
    Ok(charges
        .elements
        .iter()
        .map(|e| {
            e.and_then(|e| params::element(e).ok())
                .map_or(nitrogen, |p| p.eta)
        })
        .collect())
}

/// `r_a - r_K` of the atom of a shell and a point charge.
fn to_charge(x: [f64; 3], charge: [f64; 3]) -> [f64; 3] {
    [x[0] - charge[0], x[1] - charge[1], x[2] - charge[2]]
}

/// The potential of the point charges at the shells of `basis` (shell hardnesses `eta`).
fn external_potential(
    basis: &Basis,
    x: &[[f64; 3]],
    eta: &[f64],
    charges: Option<&PointCharges>,
) -> MoptResult<Vec<f64>> {
    let ns = basis.shells.len();
    let Some(pc) = charges else {
        return Ok(vec![0.0; ns]);
    };
    let eta_k = charge_hardness(pc)?;
    Ok((0..ns)
        .map(|a| {
            let atom = basis.shells[a].atom;
            (0..pc.len())
                .map(|k| {
                    let d = to_charge(x[atom], pc.positions_bohr[k]);
                    let g = 2.0 / (1.0 / eta[a] + 1.0 / eta_k[k]);
                    pc.charges[k] / (dot(d, d) + 1.0 / (g * g)).sqrt()
                })
                .sum()
        })
        .collect())
}

fn positions(sys: &MolecularSystem) -> Vec<[f64; 3]> {
    let x = sys.coords_flat_bohr();
    (0..sys.n_atoms())
        .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
        .collect()
}

impl Gfn1Xtb {
    fn scc(&self, sys: &MolecularSystem, charges: Option<&PointCharges>) -> MoptResult<Scc> {
        self.scc_with(sys, charges, None, true)
    }

    /// The SCC of `sys`, started from the shell charges `seed` when given (of the same
    /// shells), from [`Self::guess`] otherwise; `remember` keeps its converged charges as the
    /// start of the next SCC of [`SccGuess::Previous`].
    fn scc_with(
        &self,
        sys: &MolecularSystem,
        charges: Option<&PointCharges>,
        seed: Option<&[f64]>,
        remember: bool,
    ) -> MoptResult<Scc> {
        if sys.has_pbc() {
            return Err(MoptError::unsupported(
                "GFN1-xTB in mopt-rs is available for isolated systems only",
            ));
        }
        let el: Vec<&'static Gfn1Element> = sys
            .elements
            .iter()
            .map(|&e| params::element(e))
            .collect::<MoptResult<_>>()?;
        let x = positions(sys);
        let n = x.len();
        let charge = sys.spin_charge.charge;
        let valence: f64 =
            el.iter().map(|e| e.valence_electrons()).sum::<f64>() - f64::from(charge);
        let electrons = valence.round();
        if electrons < 0.0 {
            return Err(MoptError::config("negative electron count for GFN1-xTB"));
        }
        let electrons = electrons as usize;
        let mut unpaired = sys.spin_charge.multiplicity.saturating_sub(1) as usize;
        if unpaired == 0 && electrons % 2 == 1 {
            // As xtb and tblite: an odd-electron singlet request becomes a doublet.
            unpaired = 1;
        }

        let basis = hamiltonian::basis(&el)?;
        let s = overlap(&basis, &x);
        let xh = lowdin_orthogonalizer(&s)?;
        let coord = Coordination::new(&el, &x);
        let levels = hamiltonian::levels(&basis, &coord.cn);
        let h0 = hamiltonian::hamiltonian(&basis, &x, &s, &levels);
        let eta = hamiltonian::shell_hardness(&el, &basis)?;
        let gamma = hamiltonian::gamma(&basis, &x, &eta);
        let external = external_potential(&basis, &x, &eta, charges)?;
        let ns = basis.shells.len();
        let ref_occ: Vec<f64> = basis
            .shells
            .iter()
            .map(|sh| {
                el[sh.atom]
                    .shells
                    .iter()
                    .find(|p| p.l == sh.l && p.valence == sh.valence)
                    .map_or(0.0, |p| p.ref_occ)
            })
            .collect();
        let gamma3: Vec<f64> = el.iter().map(|e| e.gamma3).collect();
        let kt = BOLTZMANN_HARTREE_PER_K * ELECTRONIC_TEMPERATURE_K;
        let numbers: Vec<u8> = el.iter().map(|e| e.z).collect();

        // Start (SAD, the superposition of atomic densities): every shell at its reference
        // occupation, the total charge spread over the atoms by their reference occupations;
        // or, with `SccGuess::Previous`, the last converged charges of the same atoms and state.
        let sad: Vec<f64> = {
            let per_atom = f64::from(charge) / n as f64;
            basis
                .shells
                .iter()
                .enumerate()
                .map(|(k, sh)| {
                    let total = el[sh.atom].valence_electrons();
                    per_atom * ref_occ[k] / total
                })
                .collect()
        };
        let seeded = seed.filter(|q| q.len() == ns).map(<[f64]>::to_vec);
        let previous = match (seeded, self.guess) {
            (Some(q), _) => Some(q),
            (None, SccGuess::Sad) => None,
            (None, SccGuess::Previous) => self
                .last
                .lock()
                .ok()
                .and_then(|g| g.clone())
                .filter(|(z, c, m, q)| {
                    *z == numbers
                        && *c == charge
                        && *m == sys.spin_charge.multiplicity
                        && q.len() == ns
                })
                .map(|(_, _, _, q)| q),
        };
        let q = previous.unwrap_or(sad);

        let atomic = |q: &[f64]| -> Vec<f64> {
            let mut qa = vec![0.0; n];
            for (k, sh) in basis.shells.iter().enumerate() {
                qa[sh.atom] += q[k];
            }
            qa
        };
        let potential = |q: &[f64]| -> Vec<f64> {
            let qa = atomic(q);
            (0..ns)
                .map(|a| {
                    let atom = basis.shells[a].atom;
                    (0..ns).map(|b| gamma[(a, b)] * q[b]).sum::<f64>()
                        + gamma3[atom] * qa[atom] * qa[atom]
                        + external[a]
                })
                .collect()
        };
        let fock = |v: &[f64]| -> Mat<f64> {
            let mut v_ao = vec![0.0; basis.len()];
            for (k, sh) in basis.shells.iter().enumerate() {
                for slot in &mut v_ao[sh.first_ao..sh.first_ao + sh.nao] {
                    *slot = v[k];
                }
            }
            Mat::from_fn(basis.len(), basis.len(), |i, j| {
                h0[(i, j)] - 0.5 * s[(i, j)] * (v_ao[i] + v_ao[j])
            })
        };
        let charges_of = |p: &Mat<f64>| -> Vec<f64> {
            let mut q_out = ref_occ.clone();
            for (k, sh) in basis.shells.iter().enumerate() {
                for mu in sh.first_ao..sh.first_ao + sh.nao {
                    let pop: f64 = (0..basis.len()).map(|nu| p[(mu, nu)] * s[(nu, mu)]).sum();
                    q_out[k] -= pop;
                }
            }
            q_out
        };
        let solve = |q: &[f64], weighted: bool| -> MoptResult<(Electronic, Vec<f64>, Vec<f64>)> {
            let v = potential(q);
            let elec = if weighted {
                solve_at(&fock(&v), &xh, electrons, unpaired, kt)?
            } else {
                solve_density_at(&fock(&v), &xh, electrons, unpaired, kt)?
            };
            let q_out = charges_of(&elec.density);
            Ok((elec, q_out, v))
        };

        // The first density from the start, then Fock matrices extrapolated by ADIIS and, once
        // ADIIS stalls, CDIIS ([`crate::pes::native::scf_diis`]), the input density damped while
        // ADIIS runs. One run returns the charges, its iterations, and whether it converged.
        let start = q.clone();
        let run = |damping: f64| -> MoptResult<(Vec<f64>, usize, bool)> {
            let mut q = start.clone();
            let mut diis = FockDiis::new(self.diis_depth, self.diis_patience)
                .with_rate(self.diis_rate, self.diis_window);
            let mut density =
                solve_density_at(&fock(&potential(&q)), &xh, electrons, unpaired, kt)?.density;
            let mut q_now = charges_of(&density);
            let mut iterations = 1;
            while iterations < self.max_iterations {
                let f = fock(&potential(&q_now));
                let e = commutator(&f, &density, &s, &xh);
                let change = q_now
                    .iter()
                    .zip(&q)
                    .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                q = q_now;
                if change < self.charge_tolerance && largest(&e) < COMMUTATOR_TOLERANCE {
                    return Ok((q, iterations, true));
                }
                let kept = density.clone();
                let (next, _) = diis.extrapolate(f, density, e)?;
                iterations += 1;
                density = solve_density_at(&next, &xh, electrons, unpaired, kt)?.density;
                if !diis.in_cdiis() && damping > 0.0 {
                    density = Mat::from_fn(density.nrows(), density.ncols(), |i, j| {
                        (1.0 - damping) * density[(i, j)] + damping * kept[(i, j)]
                    });
                }
                q_now = charges_of(&density);
            }
            Ok((q, iterations, false))
        };
        // The modified Broyden mixing of the shell charges (Johnson 1988): the input charges
        // move by `alpha F` plus the corrections of the earlier steps, `F` the change of the
        // charges that one diagonalization gives.
        let broyden = || -> MoptResult<(Vec<f64>, usize, bool)> {
            let alpha = BROYDEN_MIXING;
            let mut q_in = start.clone();
            let mut d_f: Vec<Vec<f64>> = Vec::new();
            let mut u: Vec<Vec<f64>> = Vec::new();
            let mut last: Option<(Vec<f64>, Vec<f64>)> = None;
            let mut iterations = 0;
            while iterations < self.max_iterations {
                let v = potential(&q_in);
                let f = fock(&v);
                let density = solve_density_at(&f, &xh, electrons, unpaired, kt)?.density;
                iterations += 1;
                let q_out = charges_of(&density);
                let residual: Vec<f64> = q_out.iter().zip(&q_in).map(|(o, i)| o - i).collect();
                let change = residual.iter().fold(0.0_f64, |m, r| m.max(r.abs()));
                if change < self.charge_tolerance {
                    let e = commutator(&fock(&potential(&q_out)), &density, &s, &xh);
                    if largest(&e) < COMMUTATOR_TOLERANCE {
                        return Ok((q_out, iterations, true));
                    }
                }
                if let Some((q_prev, f_prev)) = &last {
                    let df: Vec<f64> = residual.iter().zip(f_prev).map(|(a, b)| a - b).collect();
                    let norm = df.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm > 1e-300 {
                        let df: Vec<f64> = df.iter().map(|x| x / norm).collect();
                        let dq: Vec<f64> = q_in
                            .iter()
                            .zip(q_prev)
                            .map(|(a, b)| (a - b) / norm)
                            .collect();
                        u.push(df.iter().zip(&dq).map(|(f, q)| alpha * f + q).collect());
                        d_f.push(df);
                    }
                }
                // gamma_n = sum_k c_k beta_kn with beta = (w0^2 + a)^-1, a_kn = dF_k . dF_n,
                // c_k = dF_k . F (unit weights for the later iterations).
                let m = d_f.len();
                let mut next: Vec<f64> = q_in
                    .iter()
                    .zip(&residual)
                    .map(|(q, r)| q + alpha * r)
                    .collect();
                if m > 0 {
                    let dot =
                        |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                    let a = Mat::from_fn(m, m, |k, n| {
                        dot(&d_f[k], &d_f[n]) + if k == n { BROYDEN_W0 * BROYDEN_W0 } else { 0.0 }
                    });
                    let c = Mat::from_fn(m, 1, |k, _| dot(&d_f[k], &residual));
                    let lu = a.full_piv_lu();
                    let gamma = faer::linalg::solvers::Solve::solve(&lu, c.as_ref());
                    for (n, un) in u.iter().enumerate() {
                        for (x, ui) in next.iter_mut().zip(un) {
                            *x -= gamma[(n, 0)] * ui;
                        }
                    }
                }
                last = Some((q_in, residual));
                q_in = next;
            }
            Ok((q_in, iterations, false))
        };
        // The electronic energy of converged shell charges, to choose between two solutions.
        let energy_of = |q: &[f64]| -> MoptResult<f64> {
            let (elec, q_shell, _) = solve(q, false)?;
            let p = &elec.density;
            let nao = basis.len();
            let mut e_h0 = 0.0;
            for i in 0..nao {
                for j in 0..nao {
                    e_h0 += p[(i, j)] * h0[(i, j)];
                }
            }
            let mut coulomb = 0.0;
            for a in 0..ns {
                for b in 0..ns {
                    coulomb += 0.5 * q_shell[a] * gamma[(a, b)] * q_shell[b];
                }
            }
            let third: f64 = atomic(&q_shell)
                .iter()
                .zip(&gamma3)
                .map(|(q, g)| g * q * q * q / 3.0)
                .sum();
            let band: f64 = elec
                .eigenvalues
                .iter()
                .zip(&elec.occupations)
                .map(|(e, o)| e * o)
                .sum();
            let embedding: f64 = q_shell.iter().zip(&external).map(|(q, v)| q * v).sum();
            Ok(e_h0 + coulomb + third + (elec.energy - band) + embedding)
        };
        let within =
            |q: &[f64]| atomic(q).iter().fold(0.0_f64, |m, v| m.max(v.abs())) <= self.charge_limit;
        // A run that does not converge, or that runs away to an atomic charge beyond
        // `charge_limit` (the third-order term `Gamma q^3 / 3` has no lower bound, so the SCC can
        // fall to charges no molecule has, far below the physical solution), is repeated with
        // stronger damping; the energy surface itself is not changed.
        let mut iterations = 0;
        let mut outcome = None;
        let mut runaway = None;
        for damping in [self.diis_damping, RETRY_DAMPING[0], RETRY_DAMPING[1]] {
            if damping < self.diis_damping {
                continue;
            }
            let (q_run, used, converged) = run(damping)?;
            iterations += used;
            if !converged {
                continue;
            }
            let worst = atomic(&q_run).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if worst > self.charge_limit {
                runaway = Some(worst);
                continue;
            }
            outcome = Some(q_run);
            break;
        }
        let metal = el.iter().any(|e| is_d_block(e.z));
        let mix = match self.broyden {
            SccBroyden::On => true,
            SccBroyden::Auto => metal || outcome.is_none(),
            SccBroyden::Off => outcome.is_none(),
        };
        if mix {
            let (q_mixed, used, converged) = broyden()?;
            iterations += used;
            if converged && within(&q_mixed) {
                outcome = match outcome {
                    Some(q_diis) if energy_of(&q_diis)? <= energy_of(&q_mixed)? => Some(q_diis),
                    _ => Some(q_mixed),
                };
            } else if converged && outcome.is_none() {
                runaway = Some(atomic(&q_mixed).iter().fold(0.0_f64, |m, v| m.max(v.abs())));
            }
        }
        let Some(converged_q) = outcome else {
            return Err(MoptError::calc(match runaway {
                Some(worst) => format!(
                    "GFN1-xTB: the self-consistent charges run away to an atomic charge of \
                     {worst:.2} e (beyond charge_limit {:.2} e), also with the damping raised to \
                     {} and with Broyden mixing; the structure is probably far from any molecule",
                    self.charge_limit, RETRY_DAMPING[1]
                ),
                None => format!(
                    "GFN1-xTB: the self-consistent charges did not converge in {} iterations, \
                     also with the damping raised to {} and with Broyden mixing",
                    self.max_iterations, RETRY_DAMPING[1]
                ),
            }));
        };
        let q = converged_q;
        // The final evaluation at the converged charges.
        let (elec, q_shell, _) = solve(&q, true)?;
        let q_atom = atomic(&q_shell);
        let pot = potential(&q_shell);
        let band: f64 = elec
            .eigenvalues
            .iter()
            .zip(&elec.occupations)
            .map(|(e, o)| e * o)
            .sum();
        let entropy = elec.energy - band;
        if remember {
            if let Ok(mut g) = self.last.lock() {
                *g = Some((
                    numbers,
                    charge,
                    sys.spin_charge.multiplicity,
                    q_shell.clone(),
                ));
            }
        }
        Ok(Scc {
            el,
            x,
            basis,
            s,
            h0,
            coord,
            levels,
            eta,
            gamma,
            q_shell,
            q_atom,
            potential: pot,
            external,
            elec,
            entropy,
            unpaired,
            iterations,
        })
    }

    /// Energy terms and gradient.
    pub fn compute(&self, sys: &MolecularSystem) -> MoptResult<Gfn1Result> {
        self.compute_in(sys, None)
    }

    /// Energy terms and gradient in the field of `charges` (none: [`Self::compute`]).
    pub fn compute_in(
        &self,
        sys: &MolecularSystem,
        charges: Option<&PointCharges>,
    ) -> MoptResult<Gfn1Result> {
        let scc = self.scc(sys, charges)?;
        self.result_of(sys, &scc, charges)
    }

    /// Energy terms and gradient of the converged `scc` (in the field of `charges`).
    fn result_of(
        &self,
        sys: &MolecularSystem,
        scc: &Scc,
        charges: Option<&PointCharges>,
    ) -> MoptResult<Gfn1Result> {
        let n = scc.x.len();
        let basis = &scc.basis;
        let ns = basis.shells.len();
        let (p, w) = (&scc.elec.density, &scc.elec.weighted);

        // Electronic energy.
        let nao = basis.len();
        let mut e_h0 = 0.0;
        for i in 0..nao {
            for j in 0..nao {
                e_h0 += p[(i, j)] * scc.h0[(i, j)];
            }
        }
        let mut coulomb = 0.0;
        for a in 0..ns {
            for b in 0..ns {
                coulomb += 0.5 * scc.q_shell[a] * scc.gamma[(a, b)] * scc.q_shell[b];
            }
        }
        let third_order: f64 = scc
            .el
            .iter()
            .zip(&scc.q_atom)
            .map(|(e, q)| e.gamma3 * q * q * q / 3.0)
            .sum();
        let electronic = e_h0 + coulomb + third_order + scc.entropy;
        let embedding: f64 = scc
            .q_shell
            .iter()
            .zip(&scc.external)
            .map(|(q, v)| q * v)
            .sum();

        // Gradient of the electronic energy: the density and the energy-weighted density, and
        // the Coulomb kernel at fixed charges (`1/2 q gamma q`).
        let half_qq = Mat::from_fn(ns, ns, |a, b| 0.5 * scc.q_shell[a] * scc.q_shell[b]);
        let mut grad = electronic_gradient(scc, p, w, &half_qq);
        // The point charges at fixed shell charges.
        let mut charge_gradient = vec![[0.0; 3]; charges.map_or(0, PointCharges::len)];
        if let Some(pc) = charges {
            let eta_k = charge_hardness(pc)?;
            for (a, sh) in basis.shells.iter().enumerate() {
                for k in 0..pc.len() {
                    let d = to_charge(scc.x[sh.atom], pc.positions_bohr[k]);
                    let dg = gamma_derivative(scc.eta[a], eta_k[k], d);
                    let f = scc.q_shell[a] * pc.charges[k];
                    for c in 0..3 {
                        grad[3 * sh.atom + c] += f * dg[c];
                        charge_gradient[k][c] -= f * dg[c];
                    }
                }
            }
        }
        let repulsion = classical::repulsion(&scc.el, &scc.x, &mut grad);
        let halogen_bond = classical::halogen_bond(&scc.el, &scc.x, &mut grad);
        let d3 = D3Bj {
            s6: params::D3_S6,
            s8: params::D3_S8,
            a1: params::D3_A1,
            a2: params::D3_A2,
            cutoff: params::D3_CUTOFF,
            cn_cutoff: params::CN_CUTOFF,
        };
        let disp = d3.evaluate(sys, false)?;
        let dispersion = disp.energy.value();
        for (k, g) in grad.iter_mut().enumerate() {
            *g += disp.gradient_bohr[k];
        }

        let energy = Gfn1Energy {
            total: electronic + embedding + repulsion + dispersion + halogen_bond,
            electronic,
            coulomb,
            third_order,
            entropy: scc.entropy,
            repulsion,
            dispersion,
            halogen_bond,
            embedding,
        };
        Ok(Gfn1Result {
            energy,
            gradient: Col::from_fn(3 * n, |k| grad[k]),
            charges: scc.q_atom.clone(),
            iterations: scc.iterations,
            charge_gradient,
        })
    }

    /// The orbitals of the converged SCC in the atomic orbitals.
    pub fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Orbitals> {
        let scc = self.scc(sys, None)?;
        let atoms = scc.basis.atom_of_ao();
        let elec = scc.elec;
        let [alpha, beta] = elec.spin_occupations;
        if scc.unpaired == 0 {
            let both: Vec<f64> = alpha.iter().zip(&beta).map(|(a, b)| a + b).collect();
            let set = OrbitalSet::new(elec.coefficients, elec.eigenvalues, both)?;
            return Orbitals::new(scc.s, set, None)?.with_atoms(atoms);
        }
        let up = OrbitalSet::new(elec.coefficients.clone(), elec.eigenvalues.clone(), alpha)?;
        let down = OrbitalSet::new(elec.coefficients, elec.eigenvalues, beta)?;
        Orbitals::new(scc.s, up, Some(down))?.with_atoms(atoms)
    }

    /// The orbitals of a closed-shell structure with integer occupations for its excited
    /// states, with the coupling of `excited`.
    fn reference(&self, sys: &MolecularSystem) -> MoptResult<excited::Reference> {
        self.reference_of(&self.scc(sys, None)?)
    }

    /// [`Self::reference`] of the converged `scc`.
    fn reference_of(&self, scc: &Scc) -> MoptResult<excited::Reference> {
        let occ = &scc.elec.occupations;
        let electrons: f64 = occ.iter().sum();
        let occupied = (electrons / 2.0).round() as usize;
        if scc.unpaired != 0 || (electrons - 2.0 * occupied as f64).abs() > 1e-6 {
            return Err(MoptError::unsupported(
                "GFN1-xTB excited states need a closed-shell singlet",
            ));
        }
        let integer = occ[..occupied].iter().all(|o| (o - 2.0).abs() < 1e-6)
            && occ[occupied..].iter().all(|o| o.abs() < 1e-6);
        if !integer {
            return Err(MoptError::unsupported(
                "GFN1-xTB excited states need integer occupations: the occupations at the \
                 electronic temperature are fractional here (the gap between occupied and empty \
                 orbitals is too small for a closed-shell reference)",
            ));
        }
        let shells = &scc.basis.shells;
        let nao = scc.basis.len();
        let mut shell_of_ao = vec![0; nao];
        let mut atom_of_ao = vec![0; nao];
        for (k, sh) in shells.iter().enumerate() {
            for mu in sh.first_ao..sh.first_ao + sh.nao {
                shell_of_ao[mu] = k;
                atom_of_ao[mu] = sh.atom;
            }
        }
        let (method, coupling) = match self.excited {
            ExcitedStates::Tda => {
                // The second derivatives of the energy in the shell charges: the Coulomb kernel
                // and the third-order term `Gamma_A q_A^3 / 3`, which gives `2 Gamma_A q_A`
                // between the shells of atom A.
                let ns = shells.len();
                let mut kernel = scc.gamma.clone();
                for a in 0..ns {
                    for b in 0..ns {
                        let atom = shells[a].atom;
                        if shells[b].atom == atom {
                            kernel[(a, b)] += 2.0 * scc.el[atom].gamma3 * scc.q_atom[atom];
                        }
                    }
                }
                let charges = excited::mulliken_transition_charges(
                    &scc.elec.coefficients,
                    &scc.s,
                    occupied,
                    &shell_of_ao,
                    ns,
                );
                (
                    "GFN1-xTB TDA",
                    excited::Coupling::Response { charges, kernel },
                )
            }
            ExcitedStates::Stda => (
                "GFN1-xTB sTDA",
                excited::Coupling::Simplified {
                    ao_atom: atom_of_ao,
                    hardness: scc.el.iter().map(|e| e.eta).collect(),
                    x: scc.x.clone(),
                },
            ),
        };
        Ok(excited::Reference {
            method,
            eps: scc.elec.eigenvalues.clone(),
            lowdin: scc.elec.lowdin.clone(),
            occupied,
            coupling,
        })
    }

    /// The lowest `count` excited states of `sys` in the Tamm-Dancoff approximation of the
    /// response of its self-consistent charges for a reference of any spin (the excitations of
    /// each spin coupled by the same kernel, [`crate::pes::native::unrestricted_tda`]), with
    /// their transition dipoles (dipole integrals of the basis) and the changes of the atomic
    /// populations. The occupations of each spin must be integers.
    pub fn transitions_of(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Vec<crate::pes::states::Transition>> {
        use crate::pes::native::unrestricted_tda::{lowest_states, SpinBlock};
        let scc = self.scc(sys, None)?;
        let integer = |f: &[f64]| -> MoptResult<usize> {
            let n = f.iter().filter(|&&v| v > 0.5).count();
            let ok = f[..n].iter().all(|v| (v - 1.0).abs() < 1e-6)
                && f[n..].iter().all(|v| v.abs() < 1e-6);
            if ok {
                Ok(n)
            } else {
                Err(MoptError::unsupported(
                    "GFN1-xTB transitions need integer occupations of each spin (the gap at \
                     the electronic temperature is too small here)",
                ))
            }
        };
        let [fa, fb] = &scc.elec.spin_occupations;
        let occupied = [integer(fa)?, integer(fb)?];
        let shells = &scc.basis.shells;
        let ns = shells.len();
        let nao = scc.basis.len();
        let mut shell_of_ao = vec![0; nao];
        let mut atom_of_ao = vec![0; nao];
        for (k, sh) in shells.iter().enumerate() {
            for mu in sh.first_ao..sh.first_ao + sh.nao {
                shell_of_ao[mu] = k;
                atom_of_ao[mu] = sh.atom;
            }
        }
        let c = &scc.elec.coefficients;
        let eps = &scc.elec.eigenvalues;
        let nmo = eps.len();
        let blocks: Vec<SpinBlock> = occupied
            .iter()
            .map(|&n| SpinBlock {
                occupied: eps[..n].to_vec(),
                virtuals: eps[n..].to_vec(),
                charges: excited::mulliken_transition_charges(c, &scc.s, n, &shell_of_ao, ns),
            })
            .collect();
        let kernel = Self::charge_kernel(&scc);
        let states = lowest_states(&blocks, &kernel, count)?;
        let d = crate::pes::native::gfn0::basis::dipole(&scc.basis, &scc.x, [0.0; 3]);
        let d_mo: Vec<Mat<f64>> = d.iter().map(|dc| c.transpose() * dc * c).collect();
        let n_atoms = scc.x.len();
        let mut out = Vec::with_capacity(states.energies.len());
        for (r, &excitation) in states.energies.iter().enumerate() {
            let mut mu = [0.0; 3];
            let mut dp = Mat::<f64>::zeros(nao, nao);
            // The transition density in the atomic orbitals, over both spins (the convention
            // of the transition dipole).
            let mut tp = Mat::<f64>::zeros(nao, nao);
            let mut offset = 0;
            for &n_occ in &occupied {
                let nv = nmo - n_occ;
                let x = Mat::from_fn(n_occ, nv, |i, a| {
                    states.amplitudes[(offset + i * nv + a, r)]
                });
                for (k, dm) in d_mo.iter().enumerate() {
                    for i in 0..n_occ {
                        for a in 0..nv {
                            mu[k] += x[(i, a)] * dm[(i, n_occ + a)];
                        }
                    }
                }
                let c_o = c.subcols(0, n_occ);
                let c_v = c.subcols(n_occ, nv);
                let gained = c_v * (x.transpose() * &x) * c_v.transpose();
                let lost = c_o * (&x * x.transpose()) * c_o.transpose();
                dp += gained - lost;
                tp += c_o * &x * c_v.transpose();
                offset += n_occ * nv;
            }
            let ps = &dp * &scc.s;
            let ts = &tp * &scc.s;
            let st = &scc.s * &tp;
            let mut change = vec![0.0; n_atoms];
            let mut transition_population = vec![0.0; n_atoms];
            for mu_ao in 0..nao {
                change[atom_of_ao[mu_ao]] += ps[(mu_ao, mu_ao)];
                // The Mulliken population of the symmetrized transition density.
                transition_population[atom_of_ao[mu_ao]] +=
                    0.5 * (ts[(mu_ao, mu_ao)] + st[(mu_ao, mu_ao)]);
            }
            // The dipole of the electrons (charge -1) of the difference density; the nuclei
            // do not move. The origin does not matter, as the difference density has no
            // electrons in all.
            let mut dipole_change = [0.0; 3];
            for (k, dk) in d.iter().enumerate() {
                let mut trace = 0.0;
                for i in 0..nao {
                    for j in 0..nao {
                        trace += dp[(i, j)] * dk[(j, i)];
                    }
                }
                dipole_change[k] = -trace;
            }
            out.push(crate::pes::states::Transition {
                excitation,
                transition_dipole: mu,
                population_change: change,
                dipole_change: Some(dipole_change),
                transition_population: Some(transition_population),
            });
        }
        Ok(out)
    }

    /// The kernel of the shell charges of `scc`: the Coulomb kernel and the third-order term
    /// `Gamma_A q_A^3 / 3`, which gives `2 Gamma_A q_A` between the shells of atom A.
    fn charge_kernel(scc: &Scc) -> Mat<f64> {
        let shells = &scc.basis.shells;
        let ns = shells.len();
        let mut kernel = scc.gamma.clone();
        for a in 0..ns {
            for b in 0..ns {
                let atom = shells[a].atom;
                if shells[b].atom == atom {
                    kernel[(a, b)] += 2.0 * scc.el[atom].gamma3 * scc.q_atom[atom];
                }
            }
        }
        kernel
    }

    /// Energy and analytic gradient of the excited singlet state `root` (1: the lowest): the
    /// ground state with its gradient, and the gradient of the excitation energy from the
    /// relaxed difference density of [`crate::pes::native::tda_gradient`] contracted with the
    /// derivatives of the Hamiltonian, the overlap, and the kernels.
    fn excited_analytic(&self, sys: &MolecularSystem, root: usize) -> MoptResult<(f64, Col<f64>)> {
        let scc = self.scc(sys, None)?;
        let ground = self.result_of(sys, &scc, None)?;
        let reference = self.reference_of(&scc)?;
        let states = reference.states()?;
        let index = root - 1;
        if index >= states.energies.len() {
            return Err(MoptError::config(format!(
                "GFN1-xTB excited states: state {root} of {}",
                states.energies.len()
            )));
        }
        let o = reference.occupied;
        let v = reference.eps.len() - o;
        let x = Mat::from_fn(o, v, |i, a| states.amplitudes[(i * v + a, index)]);
        let shells = &scc.basis.shells;
        let ns = shells.len();
        let nao = scc.basis.len();
        let mut shell_of_ao = vec![0; nao];
        let mut atom_of_ao = vec![0; nao];
        for (k, sh) in shells.iter().enumerate() {
            for mu in sh.first_ao..sh.first_ao + sh.nao {
                shell_of_ao[mu] = k;
                atom_of_ao[mu] = sh.atom;
            }
        }
        let atom_of_shell: Vec<usize> = shells.iter().map(|sh| sh.atom).collect();
        let gamma3: Vec<f64> = scc.el.iter().map(|e| e.gamma3).collect();
        let kernel = Self::charge_kernel(&scc);
        let coupling = tda_gradient::SccCoupling {
            shell_of_ao: &shell_of_ao,
            kernel: &kernel,
            charges: &scc.q_shell,
            density: &scc.elec.density,
        };
        let n = scc.x.len();
        let hardness: Vec<f64> = scc.el.iter().map(|e| e.eta).collect();
        let damped = Mat::from_fn(n, n, |a, b| {
            let d = diff(&scc.x, a, b);
            let r = dot(d, d).sqrt();
            let eta = 0.5 * (hardness[a] + hardness[b]);
            (r.powf(excited::ALPHA) + eta.powf(-excited::ALPHA)).powf(-1.0 / excited::ALPHA)
        });
        let input = match self.excited {
            ExcitedStates::Tda => tda_gradient::Input {
                c: &scc.elec.coefficients,
                eps: &reference.eps,
                occupied: o,
                s: &scc.s,
                x: &x,
                charges: tda_gradient::Charges::Mulliken {
                    site_of_ao: &shell_of_ao,
                    sites: ns,
                },
                kernel: &kernel,
                third: Some(tda_gradient::ThirdOrder {
                    atom_of_site: &atom_of_shell,
                    gamma3: &gamma3,
                }),
                scc: Some(coupling),
            },
            ExcitedStates::Stda => tda_gradient::Input {
                c: &scc.elec.coefficients,
                eps: &reference.eps,
                occupied: o,
                s: &scc.s,
                x: &x,
                charges: tda_gradient::Charges::Lowdin {
                    atom_of_ao: &atom_of_ao,
                    atoms: n,
                },
                kernel: &damped,
                third: None,
                scc: Some(coupling),
            },
        };
        let terms = tda_gradient::terms(&input)?;
        let mut weights = terms.scc.clone().unwrap_or_else(|| Mat::zeros(ns, ns));
        if self.excited == ExcitedStates::Tda {
            weights += &terms.kernel;
        }
        let w = &terms.overlap * -1.0;
        let mut grad = electronic_gradient(&scc, &terms.density, &w, &weights);
        if self.excited == ExcitedStates::Stda {
            tda_gradient::add_damped_coulomb_gradient(
                &scc.x,
                &hardness,
                excited::ALPHA,
                &terms.kernel,
                &mut grad,
            );
        }
        let gradient = Col::from_fn(3 * n, |k| ground.gradient[k] + grad[k]);
        Ok((ground.energy.total + states.energies[index], gradient))
    }

    /// Excitation energies (Hartree, ascending) of the singlet excited states of a
    /// closed-shell structure ([`ExcitedStates`]).
    pub fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Vec<f64>> {
        Ok(self.reference(sys)?.states()?.energies)
    }

    /// Energy and gradient of the excited singlet state `root` (1: the lowest): the ground
    /// state with its analytic gradient, the excitation energy by central differences that
    /// follow the state (as for GFN0-xTB).
    fn excited(&self, sys: &MolecularSystem, root: usize) -> MoptResult<(f64, Col<f64>)> {
        if self.analytic_excited {
            return self.excited_analytic(sys, root);
        }
        self.excited_numerical(sys, root)
    }

    /// [`Self::excited`] with the gradient of the excitation energy by central differences
    /// that follow the state.
    fn excited_numerical(&self, sys: &MolecularSystem, root: usize) -> MoptResult<(f64, Col<f64>)> {
        let ground = self.compute(sys)?;
        let reference = self.reference(sys)?;
        let states = reference.states()?;
        let index = root - 1;
        if index >= states.energies.len() {
            return Err(MoptError::config(format!(
                "GFN1-xTB excited states: state {root} of {}",
                states.energies.len()
            )));
        }
        let gradient = excited::followed_gradient(
            &reference,
            &states,
            index,
            &sys.coords_flat_bohr(),
            self.excitation_step_bohr,
            ground.gradient.clone(),
            |moved| self.reference(&sys.with_flat_coords_bohr(moved)?),
        )?;
        Ok((ground.energy.total + states.energies[index], gradient))
    }

    /// The Hessian of the excited state `root` from central differences of its gradient.
    ///
    /// The columns run one after the other in the order of the coordinates (the serial form
    /// of the shared central differences), as they always have here.
    fn excited_hessian(&self, sys: &MolecularSystem, root: usize) -> MoptResult<Mat<f64>> {
        crate::pes::fd_hessian::serial_central_difference_hessian(
            &sys.coords_flat_bohr(),
            self.hessian_step_bohr,
            |x| Ok(self.excited(&sys.with_flat_coords_bohr(x)?, root)?.1),
        )
    }

    /// The Hessian: analytic ([`Self::analytic_hessian_in`]), or central differences of the
    /// analytic gradient when `analytic_hessian` is off.
    pub fn hessian(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        self.hessian_in(sys, None)
    }

    /// The Hessian of the atoms (the charges held) in the field of `charges`.
    pub fn hessian_in(
        &self,
        sys: &MolecularSystem,
        charges: Option<&PointCharges>,
    ) -> MoptResult<Mat<f64>> {
        if self.analytic_hessian {
            return self.analytic_hessian_in(sys, charges);
        }
        self.numerical_hessian_in(sys, charges)
    }

    /// The Hessian from central differences of the analytic gradient.
    pub fn numerical_hessian_in(
        &self,
        sys: &MolecularSystem,
        charges: Option<&PointCharges>,
    ) -> MoptResult<Mat<f64>> {
        // The SCC at the structure itself first: it is the start of the next SCC, and every
        // displaced SCC starts from its charges (with `SccGuess::Previous`), so that the
        // displaced structures, evaluated in parallel, neither depend on the order in which they
        // run nor change the start kept for the next structure.
        let reference = self.scc_with(sys, charges, None, true)?;
        let seed = match self.guess {
            SccGuess::Sad => None,
            SccGuess::Previous => Some(reference.q_shell.clone()),
        };
        crate::pes::native::fd_hessian::structure_hessian(sys, self.hessian_step_bohr, true, |at| {
            let scc = self.scc_with(at, charges, seed.as_deref(), false)?;
            Ok(self.result_of(at, &scc, charges)?.gradient)
        })
    }
}

/// The gradient of `tr(p H) - tr(w S) + sum_(s != t) g_st gamma_st` with the charges and the
/// potential of `scc` held fixed: the derivatives of the Hamiltonian contracted with the
/// symmetric `p` (through the overlap, the
/// distance polynomial, and the coordination numbers of the levels), of the overlap with
/// `-w`, and of the Coulomb kernel with `g` (`sum_(s != t) g_st dgamma_st`). With the density,
/// the energy-weighted density, and `q q^T / 2` it is the gradient of the electronic energy;
/// the excited states give their own ([`crate::pes::native::tda_gradient`]).
fn electronic_gradient(scc: &Scc, p: &Mat<f64>, w: &Mat<f64>, g: &Mat<f64>) -> Vec<f64> {
    let basis = &scc.basis;
    let n = scc.x.len();
    let ns = basis.shells.len();
    let mut grad = vec![0.0; 3 * n];
    let mut de_dcn = vec![0.0; n];
    let dlevel: Vec<f64> = basis.shells.iter().map(|s| s.level * s.kcn).collect();
    for (a, sh) in basis.shells.iter().enumerate() {
        let diag: f64 = (sh.first_ao..sh.first_ao + sh.nao)
            .map(|mu| p[(mu, mu)])
            .sum();
        de_dcn[sh.atom] += diag * dlevel[a];
    }
    for a in 0..ns {
        for b in a + 1..ns {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            if sa.atom == sb.atom {
                continue;
            }
            let d = diff(&scc.x, sa.atom, sb.atom);
            if basis.negligible(a, b, dot(d, d)) {
                continue;
            }
            let ds = pair_overlap_gradient(basis, a, b, scc.x[sa.atom], scc.x[sb.atom]);
            let ps = pair_scale(sa, sb, d);
            let hav = 0.5 * ps.km * (scc.levels[a] + scc.levels[b]);
            let ht = hav * ps.poly;
            let vv = scc.potential[a] + scc.potential[b];
            let mut gr = [0.0; 3];
            let mut ps_sum = 0.0;
            for (row, mu) in (sa.first_ao..sa.first_ao + sa.nao).enumerate() {
                for (col, nu) in (sb.first_ao..sb.first_ao + sb.nao).enumerate() {
                    let pmn = p[(mu, nu)];
                    let fac = 2.0 * (pmn * ht - 0.5 * vv * pmn - w[(mu, nu)]);
                    let dsv = ds[row * sb.nao + col];
                    for (gk, dk) in gr.iter_mut().zip(dsv) {
                        *gk += fac * dk;
                    }
                    ps_sum += pmn * scc.s[(mu, nu)];
                }
            }
            for (gk, dk) in gr.iter_mut().zip(ps.dpoly) {
                *gk += 2.0 * ps_sum * hav * dk;
            }
            for k in 0..3 {
                grad[3 * sa.atom + k] += gr[k];
                grad[3 * sb.atom + k] -= gr[k];
            }
            let t = 2.0 * ps_sum * 0.5 * ps.km * ps.poly;
            de_dcn[sa.atom] += t * dlevel[a];
            de_dcn[sb.atom] += t * dlevel[b];
        }
    }
    // The Coulomb kernel.
    for a in 0..ns {
        for b in a + 1..ns {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            if sa.atom == sb.atom {
                continue;
            }
            let d = diff(&scc.x, sa.atom, sb.atom);
            let dg = gamma_derivative(scc.eta[a], scc.eta[b], d);
            let weight = g[(a, b)] + g[(b, a)];
            for k in 0..3 {
                grad[3 * sa.atom + k] += weight * dg[k];
                grad[3 * sb.atom + k] -= weight * dg[k];
            }
        }
    }
    scc.coord.contract(&de_dcn, &mut grad);
    grad
}

impl Calculator for Gfn1Xtb {
    fn citations(&self) -> Vec<String> {
        vec![
            "S. Grimme, C. Bannwarth, P. Shushkov, J. Chem. Theory Comput. 2017, 13, 1989 \
             (GFN1-xTB)"
                .to_string(),
        ]
    }

    fn name(&self) -> &str {
        "gfn1xtb"
    }

    fn analytic_hessian(&self) -> bool {
        self.analytic_hessian
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        Gfn1Xtb::excitation_energies(self, sys).map(Some)
    }

    fn transitions(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<Vec<crate::pes::states::Transition>>> {
        // The TDA of its own response; the sTDA has no transitions of open shells here.
        if self.excited != ExcitedStates::Tda {
            return Ok(None);
        }
        Ok(Some(self.transitions_of(sys, count)?))
    }

    fn state_characters(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<crate::pes::states::StateCharacters>> {
        let reference = self.reference(sys)?;
        let states = reference.states()?;
        Ok(Some(
            crate::pes::states::StateCharacters::new(
                reference.lowdin,
                reference.occupied,
                states.energies,
                states.amplitudes,
            )?
            .lowest(count),
        ))
    }

    fn description(&self) -> String {
        format!(
            "gfn1xtb: GFN1-xTB built into mopt-rs {} (xtb 6.7.1's parameters, tblite 0.4.0 as \
             the reference; {}); Hessian: analytic (central differences of the analytic \
             gradient with -calc_opt hessian=numerical); excited singlet states (-es), for \
             tests: {} (mopt-rs)",
            crate::VERSION,
            params::ELEMENT_RANGE,
            match self.excited {
                ExcitedStates::Tda =>
                    "the Tamm-Dancoff approximation of the response of its \
                                       self-consistent charges",
                ExcitedStates::Stda =>
                    "the simplified Tamm-Dancoff approximation (sTDA) on its \
                                        orbitals",
            }
        )
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let root = req.excited_state;
        if root > 0 {
            let (energy, gradient) = self.excited(sys, root)?;
            let mut out = EvalOutput::energy_gradient(Quantity::new(energy), gradient);
            if req.need_hessian {
                out.hessian_bohr = Some(self.excited_hessian(sys, root)?);
            }
            return Ok(out);
        }
        let r = self.compute(sys)?;
        let mut out = EvalOutput::energy_gradient(Quantity::new(r.energy.total), r.gradient);
        if req.need_hessian {
            out.hessian_bohr = Some(self.hessian(sys)?);
        }
        Ok(out)
    }

    fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Option<Orbitals>> {
        Gfn1Xtb::orbitals(self, sys).map(Some)
    }

    fn supports_point_charges(&self) -> bool {
        true
    }

    fn evaluate_in_charges(
        &self,
        sys: &MolecularSystem,
        req: &EvalRequest,
        charges: &PointCharges,
    ) -> MoptResult<EmbeddedOutput> {
        if req.excited_state > 0 {
            return Err(MoptError::unsupported(
                "GFN1-xTB excited states (-es) in point charges are not available",
            ));
        }
        let r = self.compute_in(sys, Some(charges))?;
        let mut output = EvalOutput::energy_gradient(Quantity::new(r.energy.total), r.gradient);
        if req.need_hessian {
            output.hessian_bohr = Some(self.hessian_in(sys, Some(charges))?);
        }
        Ok(EmbeddedOutput {
            output,
            charge_gradient_bohr: r.charge_gradient,
        })
    }

    fn uses_electrons(&self) -> bool {
        true
    }
}
