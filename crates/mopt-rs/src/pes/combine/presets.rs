//! Ready-made combined surfaces: MultiOptPy's model functions (`-mf`) and EDEEL.
//!
//! EDEEL comes in the two forms of A. Mutsuji's thesis (Hokkaido University, 2025,
//! doi:10.14943/doctoral.k16373): between separate donor and acceptor molecules (its chapter 3,
//! published as Mutsuji, Saita, Maeda, RSC Adv. 2023, 13, 32097), and within one molecule of
//! donor, bridge, and acceptor (its chapter 4, eqn (4.3) and (4.4); not published in a
//! peer-reviewed journal), where the fragments are the donor with the bridge and the bridge
//! with the acceptor, each with the bonds cut at its border capped by hydrogens.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use super::functions::{
    AvoidingModel, ConicalPenalty, FunctionCombiner, LegacyAvoidingModel, QuadraticPenalty,
};
use super::projection::GradientProjection;
use super::{Caps, PesExpr, StateCombiner, TaskSpec};
use crate::domain::{AtomId, MolecularSystem, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::pes::oniom::layout::LinkPlacement;
use crate::pes::term::EnergyShift;

/// Names accepted by [`model_function`].
pub const MODEL_FUNCTIONS: [&str; 8] = [
    "seam",
    "conical",
    "avoiding",
    "avoiding_lower",
    "legacy_avoiding",
    "opt_mesx",
    "opt_mesx_2",
    "opt_meci",
];

/// Two-state combiner by MultiOptPy name, with the optional `alpha` and `sigma` parameters.
///
/// Defaults follow MultiOptPy: `seam` alpha = 0.05; `conical` alpha = 0.025, sigma = 3.5;
/// `legacy_avoiding` (MultiOptPy's `avoiding`) alpha = 0.01. `avoiding` and `avoiding_lower`
/// are the Avoiding model function of Maeda, Ohno, and Morokuma (upper and lower adiabatic
/// state), alpha = 30 kJ/mol. `opt_mesx` and `opt_mesx_2` are the same method in MultiOptPy.
pub fn model_function(
    name: &str,
    alpha: Option<f64>,
    sigma: Option<f64>,
) -> MoptResult<Box<dyn StateCombiner>> {
    let positive = |v: Option<f64>, what: &str| -> MoptResult<Option<f64>> {
        match v {
            Some(x) if !(x.is_finite() && x > 0.0) => Err(MoptError::config(format!(
                "model function {what} must be positive, got {x}"
            ))),
            _ => Ok(v),
        }
    };
    let alpha = positive(alpha, "alpha")?;
    let sigma = positive(sigma, "sigma")?;
    let combiner: Box<dyn StateCombiner> = match name.to_ascii_lowercase().as_str() {
        "seam" => Box::new(FunctionCombiner::new(QuadraticPenalty::seam(
            alpha.unwrap_or(0.05),
        ))),
        "conical" => {
            let d = ConicalPenalty::default();
            Box::new(FunctionCombiner::new(ConicalPenalty {
                alpha: alpha.unwrap_or(d.alpha),
                sigma: sigma.unwrap_or(d.sigma),
            }))
        }
        "avoiding" | "avoiding_lower" => Box::new(FunctionCombiner::new(AvoidingModel {
            alpha: alpha.unwrap_or(AvoidingModel::default().alpha),
            lower: name.eq_ignore_ascii_case("avoiding_lower"),
        })),
        "legacy_avoiding" => Box::new(FunctionCombiner::new(LegacyAvoidingModel {
            alpha: alpha.unwrap_or(LegacyAvoidingModel::default().alpha),
        })),
        "opt_mesx" | "opt_mesx_2" | "mesx" => Box::new(GradientProjection::mesx()),
        "opt_meci" | "meci" => Box::new(GradientProjection::meci()),
        other => {
            return Err(MoptError::config(format!(
                "unknown model function '{other}' (known: {})",
                MODEL_FUNCTIONS.join(", ")
            )))
        }
    };
    Ok(combiner)
}

/// Two states of the whole structure combined by `combiner` (MultiOptPy `-mf NAME [charge
/// multiplicity]`): state 1 is the structure as given, state 2 is `state2`.
pub fn two_states(combiner: Box<dyn StateCombiner>, state2: TaskSpec) -> MoptResult<PesExpr> {
    two_states_of(combiner, TaskSpec::whole("State1"), state2)
}

/// The states `state1` and `state2` combined by `combiner`.
pub fn two_states_of(
    combiner: Box<dyn StateCombiner>,
    state1: TaskSpec,
    state2: TaskSpec,
) -> MoptResult<PesExpr> {
    if state2.label == state1.label {
        return Err(MoptError::config("the second state needs its own label"));
    }
    let labels = vec![state1.label.clone(), state2.label.clone()];
    PesExpr::combine(
        combiner,
        labels,
        vec![PesExpr::Task(state1), PesExpr::Task(state2)],
    )
}

/// Electronic states of the five EDEEL evaluations (Mutsuji, Saita, Maeda, RSC Adv. 2023, 13,
/// 32097, eqn (7) and (8)). The subscripts of the paper count electrons: the donor has `n`
/// electrons without and `n + 1` with the moving electron, the acceptor `m` and `m + 1`, and the
/// complex `n + m` (the donor-acceptor pair without the moving electron).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdeelStates {
    /// Complex without the moving electron, `E^C_{n+m}` (charge = donor + acceptor charges
    /// without the moving electron; e.g. +6 for a M(III)/M(III) pair of hexaaqua ions).
    pub complex_without: SpinCharge,
    /// Donor without the moving electron, `E^D_n`.
    pub donor_without: SpinCharge,
    /// Donor with the moving electron, `E^D_{n+1}`.
    pub donor_with: SpinCharge,
    /// Acceptor without the moving electron, `E^A_m`.
    pub acceptor_without: SpinCharge,
    /// Acceptor with the moving electron, `E^A_{m+1}`.
    pub acceptor_with: SpinCharge,
}

impl EdeelStates {
    /// Check that charges and multiplicities follow the electron counts `n`, `n + 1`, `m`,
    /// `m + 1`, and `n + m`.
    pub fn validate(&self) -> MoptResult<()> {
        self.validate_fragments()?;
        let bad = |what: &str| Err(MoptError::config(format!("EDEEL states: {what}")));
        if self.complex_without.charge != self.donor_without.charge + self.acceptor_without.charge {
            return bad(
                "the complex (without the moving electron) must carry the donor and acceptor \
                 charges without the moving electron",
            );
        }
        // Multiplicity 2S + 1 is odd for an even electron count and even for an odd one.
        let odd = |sc: SpinCharge| sc.multiplicity.is_multiple_of(2);
        if odd(self.donor_with) == odd(self.donor_without)
            || odd(self.acceptor_with) == odd(self.acceptor_without)
        {
            return bad("adding one electron must change the parity of the multiplicity");
        }
        if odd(self.complex_without) != (odd(self.donor_without) ^ odd(self.acceptor_without)) {
            return bad(
                "the complex multiplicity does not match the donor and acceptor electron counts",
            );
        }
        Ok(())
    }

    /// The checks that hold for both forms: the fragments with the moving electron are one
    /// unit more negative, and adding the electron changes the parity of the multiplicity.
    fn validate_fragments(&self) -> MoptResult<()> {
        let bad = |what: &str| Err(MoptError::config(format!("EDEEL states: {what}")));
        if self.donor_with.charge != self.donor_without.charge - 1 {
            return bad("the donor with the moving electron must be one unit more negative");
        }
        if self.acceptor_with.charge != self.acceptor_without.charge - 1 {
            return bad("the acceptor with the moving electron must be one unit more negative");
        }
        let odd = |sc: SpinCharge| sc.multiplicity.is_multiple_of(2);
        if odd(self.donor_with) == odd(self.donor_without)
            || odd(self.acceptor_with) == odd(self.acceptor_without)
        {
            return bad("adding one electron must change the parity of the multiplicity");
        }
        Ok(())
    }

    /// The checks for a molecule of donor, bridge, and acceptor (thesis, Table 4.1): the
    /// fragments as above, and the charge of the whole molecule that of the two fragments
    /// without the bridge, counted in both, of charge `bridge_charge`. The parities of the
    /// fragments depend on their caps; the calculators check them against their electrons.
    pub fn validate_bridged(&self, bridge_charge: i32) -> MoptResult<()> {
        self.validate_fragments()?;
        if self.complex_without.charge
            != self.donor_without.charge + self.acceptor_without.charge - bridge_charge
        {
            return Err(MoptError::config(format!(
                "EDEEL states: the molecule without the moving electron must carry the charges \
                 of the donor-bridge and bridge-acceptor fragments less that of the bridge \
                 ({bridge_charge})"
            )));
        }
        Ok(())
    }

    /// Evaluation of the complex with the moving electron, `E^C_{n+m+1}` (its adiabatic ground
    /// state for self-exchange), used for the coupling `V12 = V11 - E^C_{n+m+1}` at the crossing
    /// (`analysis::electron_transfer::edeel_coupling`).
    pub fn coupling_task(&self, multiplicity: u32) -> TaskSpec {
        TaskSpec::whole("complex_with").spin_charge(SpinCharge::new(
            self.complex_without.charge - 1,
            multiplicity,
        ))
    }
}

/// Surface optimized in an EDEEL run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdeelTarget {
    /// Initial diabatic state `V11` (equilibrium of the initial state).
    Initial,
    /// Final diabatic state `V22` (equilibrium of the final state).
    Final,
    /// Minimum-energy seam of crossing between `V11` and `V22` by gradient projection, as in the
    /// paper (step 2 of its workflow; Bearpark et al. 1994, Maeda et al. 2010).
    Crossing,
    /// MultiOptPy's penalty-function variant `(V11 + V22)/2 + sigma (V11 - V22)^2`, which is
    /// not part of the published method and stops short of the seam.
    Penalty {
        /// Penalty weight (1/Hartree); MultiOptPy uses 2.0.
        sigma: f64,
    },
}

/// A fragment of EDEEL: the atoms of the species that holds the moving electron in one diabatic
/// state, and the hydrogen caps of the bonds cut at its border (none for a separate molecule).
#[derive(Debug, Clone, PartialEq)]
pub struct EdeelFragment {
    /// Its atoms, in the order of its calculations.
    pub atoms: Vec<AtomId>,
    /// The caps of its cut bonds.
    pub caps: Option<Caps>,
    /// The electronic state of the fragment with the moving electron (`0`: its ground state).
    /// An excited state (TD-DFT in the thesis, section 4.2) gives the state of the moving
    /// electron in another orbital of the same species, for a final state that is an excited
    /// state of the acceptor.
    pub root: usize,
}

impl EdeelFragment {
    /// The separate molecule `atoms`.
    pub fn molecule(atoms: &[AtomId]) -> Self {
        Self {
            atoms: atoms.to_vec(),
            caps: None,
            root: 0,
        }
    }

    /// The moving electron in the state `root` of the fragment.
    #[must_use]
    pub fn in_root(mut self, root: usize) -> Self {
        self.root = root;
        self
    }
}

/// Where the moving electron sits in the two diabatic states of EDEEL, and the energy shifts
/// added to them.
#[derive(Debug, Clone, PartialEq)]
pub struct EdeelLayout {
    /// The donor (`D`, or the donor with the bridge `DB`).
    pub donor: EdeelFragment,
    /// The acceptor (`A`, or the bridge with the acceptor `BA`).
    pub acceptor: EdeelFragment,
    /// The atoms of the bridge, counted in both fragments (empty for separate molecules).
    pub bridge: Vec<AtomId>,
    /// The charge of the bridge (in the check of the charges of the fragments).
    pub bridge_charge: i32,
    /// Constant energies (Hartree) added to `V11` and `V22`: the solvent activation energy
    /// shift (SAES, thesis eqn (4.5) and (4.6)), or any other energy shift.
    pub shifts: Option<[f64; 2]>,
}

impl EdeelLayout {
    /// Separate donor and acceptor molecules (thesis chapter 3; RSC Adv. 2023).
    pub fn separate(donor: &[AtomId], acceptor: &[AtomId]) -> MoptResult<Self> {
        if donor.is_empty() || acceptor.is_empty() {
            return Err(MoptError::config("EDEEL needs donor and acceptor atoms"));
        }
        if donor.iter().any(|a| acceptor.contains(a)) {
            return Err(MoptError::config(
                "EDEEL donor and acceptor atoms must not overlap",
            ));
        }
        Ok(Self {
            donor: EdeelFragment::molecule(donor),
            acceptor: EdeelFragment::molecule(acceptor),
            bridge: Vec::new(),
            bridge_charge: 0,
            shifts: None,
        })
    }

    /// A molecule `sys` of `donor`, `bridge`, and `acceptor` (thesis chapter 4, eqn (4.3) and
    /// (4.4)): the fragments are the donor with the bridge and the bridge with the acceptor,
    /// and the bonds each cuts (bridge-acceptor and donor-bridge) are capped by hydrogens
    /// placed by `placement`. Every atom of `sys` belongs to one of the three parts.
    pub fn bridged(
        sys: &MolecularSystem,
        donor: &[AtomId],
        bridge: &[AtomId],
        acceptor: &[AtomId],
        placement: LinkPlacement,
    ) -> MoptResult<Self> {
        if donor.is_empty() || bridge.is_empty() || acceptor.is_empty() {
            return Err(MoptError::config(
                "EDEEL of a bridged molecule needs donor, bridge, and acceptor atoms",
            ));
        }
        let mut seen = vec![0usize; sys.n_atoms()];
        for a in donor.iter().chain(bridge).chain(acceptor) {
            a.check(sys.n_atoms())?;
            seen[a.index()] += 1;
        }
        if let Some(k) = seen.iter().position(|&c| c != 1) {
            return Err(MoptError::config(format!(
                "EDEEL of a bridged molecule: atom {} is in {} of the donor, the bridge, and \
                 the acceptor (every atom belongs to exactly one)",
                k + 1,
                seen[k]
            )));
        }
        let fragment = |first: &[AtomId], second: &[AtomId]| -> MoptResult<EdeelFragment> {
            let atoms: Vec<AtomId> = first.iter().chain(second).copied().collect();
            let caps = Caps::of_bonds(sys, &atoms, placement);
            if caps.links.is_empty() {
                return Err(MoptError::config(
                    "EDEEL of a bridged molecule: no bond joins the bridge to the donor or to \
                     the acceptor (for separate molecules use the intermolecular form)",
                ));
            }
            Ok(EdeelFragment {
                atoms,
                caps: Some(caps),
                root: 0,
            })
        };
        Ok(Self {
            donor: fragment(donor, bridge)?,
            acceptor: fragment(bridge, acceptor)?,
            bridge: bridge.to_vec(),
            bridge_charge: 0,
            shifts: None,
        })
    }

    /// With the energy shifts `shifts` (Hartree) added to `V11` and `V22`.
    #[must_use]
    pub fn shifted(mut self, shifts: [f64; 2]) -> Self {
        self.shifts = Some(shifts);
        self
    }

    /// Whether the molecule has a bridge (the intramolecular form).
    pub fn is_bridged(&self) -> bool {
        !self.bridge.is_empty()
    }
}

/// Energy decomposition and extrapolation-based electron localization (EDEEL; Mutsuji, Saita,
/// Maeda, RSC Adv. 2023, 13, 32097; doi:10.1039/D3RA05784D) for electron transfer from `donor` to
/// `acceptor`, separate molecules ([`edeel_with`] has the bridged form and energy shifts).
///
/// The diabatic energies are built from adiabatic energies (eqn (7) and (8)):
///
/// `V11(R) = E^C_{n+m}(R) - E^D_n(R^D) + E^D_{n+1}(R^D)`,
/// `V22(R) = E^C_{n+m}(R) - E^A_m(R^A) + E^A_{m+1}(R^A)`,
///
/// where `R` is the whole structure (the donor-acceptor complex) and `R^D`, `R^A` its donor and
/// acceptor parts; the moving electron is assumed to be localized on one species and not to
/// interact with the other. The complex is evaluated once and shared by both states; fragment
/// gradients and Hessians are scattered onto the complex. The paper optimizes the seam with the
/// donor-acceptor distance held fixed, which is a constraint of the optimization run.
pub fn edeel(
    donor: &[AtomId],
    acceptor: &[AtomId],
    states: &EdeelStates,
    target: EdeelTarget,
) -> MoptResult<PesExpr> {
    edeel_with(&EdeelLayout::separate(donor, acceptor)?, states, target)
}

/// The diabatic states `[V11, V22]` of EDEEL on `layout`, each with its energy shift.
///
/// `V11 = E^C_{n+m} - E^D_n + E^D_{n+1}` and `V22 = E^C_{n+m} - E^A_m + E^A_{m+1}`, with the
/// donor-bridge and bridge-acceptor fragments in place of `D` and `A` in a bridged molecule
/// (thesis eqn (4.3) and (4.4)).
pub fn edeel_states(layout: &EdeelLayout, states: &EdeelStates) -> MoptResult<[PesExpr; 2]> {
    if layout.is_bridged() {
        states.validate_bridged(layout.bridge_charge)?;
    } else {
        states.validate()?;
    }
    let complex =
        || PesExpr::Task(TaskSpec::whole("complex_without").spin_charge(states.complex_without));
    let fragment = |label: &str, part: &EdeelFragment, sc: SpinCharge, root: usize| {
        let mut task = TaskSpec::whole(label)
            .atoms(part.atoms.clone())
            .spin_charge(sc)
            .excited_state(root);
        if let Some(caps) = &part.caps {
            task = task.caps(caps.clone());
        }
        PesExpr::Task(task)
    };
    let (d, a) = (&layout.donor, &layout.acceptor);
    let mut v11 = vec![
        (1.0, complex()),
        (-1.0, fragment("donor_without", d, states.donor_without, 0)),
        (1.0, fragment("donor_with", d, states.donor_with, d.root)),
    ];
    let mut v22 = vec![
        (1.0, complex()),
        (
            -1.0,
            fragment("acceptor_without", a, states.acceptor_without, 0),
        ),
        (
            1.0,
            fragment("acceptor_with", a, states.acceptor_with, a.root),
        ),
    ];
    if let Some([s1, s2]) = layout.shifts {
        for (v, value, name) in [
            (&mut v11, s1, "shift_initial"),
            (&mut v22, s2, "shift_final"),
        ] {
            if !value.is_finite() {
                return Err(MoptError::config(format!(
                    "EDEEL: the energy shift {name} is not finite"
                )));
            }
            v.push((
                1.0,
                PesExpr::term(name, Arc::new(EnergyShift::new(name, value))),
            ));
        }
    }
    Ok([PesExpr::Linear(v11), PesExpr::Linear(v22)])
}

/// EDEEL on `layout` (separate molecules or a bridged molecule, with energy shifts), the
/// surface of `target`.
pub fn edeel_with(
    layout: &EdeelLayout,
    states: &EdeelStates,
    target: EdeelTarget,
) -> MoptResult<PesExpr> {
    let [v11, v22] = edeel_states(layout, states)?;
    let labels = vec!["V11".to_string(), "V22".to_string()];
    let expr = match target {
        EdeelTarget::Initial => v11,
        EdeelTarget::Final => v22,
        EdeelTarget::Crossing => {
            PesExpr::combine(Box::new(GradientProjection::mesx()), labels, vec![v11, v22])?
        }
        EdeelTarget::Penalty { sigma } => {
            if !(sigma.is_finite() && sigma > 0.0) {
                return Err(MoptError::config(format!(
                    "EDEEL sigma must be positive, got {sigma}"
                )));
            }
            PesExpr::combine(
                Box::new(FunctionCombiner::new(QuadraticPenalty { weight: sigma })),
                labels,
                vec![v11, v22],
            )?
        }
    };
    expr.validate()?;
    Ok(expr)
}
