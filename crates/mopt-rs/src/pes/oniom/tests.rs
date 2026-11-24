//! Tests of ONIOM: the bookkeeping (one level on every region gives the energy of the real
//! system), gradients and Hessians against differences with both kinds of link atoms, three
//! layers, and the link atoms that are found.
//!
//! MultiOptPy: no counterpart (its ONIOM does not run).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::Col;

use super::*;

#[test]
fn periodic_chain_and_link_remain_bonded_in_the_model() {
    let real=crate::io::xyz::parse_xyz_str("6\nLattice=\"10 0 0 0 10 0 0 0 10\"\nC 0 0 0\nC 1.4 0 0\nC 2.8 0 0\nC 4.2 0 0\nC 5.6 0 0\nC 7.0 0 0\n",SpinCharge::default(),None).unwrap();
    let region = Region {
        atoms: real.atoms().take(5).collect(),
        links: vec![Link {
            inner: AtomId::new(4),
            outer: AtomId::new(5),
            factor: 0.7,
        }],
        spin_charge: real.spin_charge,
        placement: LinkPlacement::Scaled,
        follows: None,
    };
    let model = region.model(&real).unwrap();
    for i in 0..5 {
        let d = model.distance(AtomId::new(i), AtomId::new(i + 1)).value()
            * crate::domain::Units::BOHR_TO_ANGSTROM;
        assert!((d - if i == 4 { 0.98 } else { 1.4 }).abs() < 1e-10);
    }
}
use crate::domain::{EvalRequest, SpinCharge};
use crate::io::xyz::parse_xyz_str;
use crate::pes::calculator::Calculator;
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};

/// Ethanol, a little out of its minimum.
const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

fn ethanol() -> MolecularSystem {
    parse_xyz_str(ETHANOL, SpinCharge::default(), None).unwrap()
}

fn ids(list: &[usize]) -> Vec<AtomId> {
    list.iter()
        .map(|&k| AtomId::from_one_based(k).unwrap())
        .collect()
}

fn oniom(names: &[&str], regions: Vec<RegionSpec>, placement: LinkPlacement) -> OniomCalculator {
    let sys = ethanol();
    let spec = OniomSpec {
        regions,
        lower_levels: names[1..].iter().map(|s| s.to_string()).collect(),
        lower_options: Vec::new(),
        placement,
        traj_high_only: false,
        micro: None,
        embedding: None,
    };
    let resolved = spec.resolve(&sys).unwrap();
    let registry = CalculatorRegistry::native();
    let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    OniomCalculator::build(names.len(), resolved, &sys, |level, system, _| {
        registry.create(&names[level], &CalculatorOptions::default(), system)
    })
    .unwrap()
}

fn hydroxyl() -> RegionSpec {
    RegionSpec {
        atoms: ids(&[2, 3, 4, 8, 9]),
        links: LinkSpec::Auto,
        spin_charge: None,
    }
}

#[test]
fn one_level_everywhere_is_the_real_system_and_links_are_found() {
    let sys = ethanol();
    let registry = CalculatorRegistry::native();
    let plain = registry
        .create("gfnff", &CalculatorOptions::default(), &sys)
        .unwrap();
    let real = plain.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
    for placement in [LinkPlacement::Scaled, LinkPlacement::Distance] {
        let calc = oniom(&["gfnff", "gfnff"], vec![hydroxyl()], placement);
        // The C-C bond crosses the border: one link atom in place of C1.
        assert_eq!(calc.regions()[0].links.len(), 1);
        assert_eq!(calc.regions()[0].links[0].outer.one_based(), 1);
        let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        assert!((out.energy.value() - real.energy.value()).abs() < 1e-10);
        let worst = (&out.gradient_bohr - &real.gradient_bohr)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(worst < 1e-10, "{worst}");
    }
    // Three layers of one level: the real system again.
    let middle = RegionSpec {
        atoms: ids(&[1, 2, 3, 4, 8, 9]),
        links: LinkSpec::Auto,
        spin_charge: None,
    };
    let three = oniom(
        &["gfnff", "gfnff", "gfnff"],
        vec![hydroxyl(), middle],
        LinkPlacement::Scaled,
    );
    let out = three.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
    assert!((out.energy.value() - real.energy.value()).abs() < 1e-10);
    // The model systems are closed shells: 5 atoms and a hydrogen in place of C1.
    let model = three.regions()[0].model(&sys).unwrap();
    assert_eq!(model.n_atoms(), 6);
    assert_eq!(three.regions()[0].spin_charge.multiplicity, 1);
}

fn difference_check(calc: &OniomCalculator, hessian: bool) {
    let sys = ethanol();
    let x = sys.coords_flat_bohr();
    let request = if hessian {
        EvalRequest::ENERGY_GRAD_HESS
    } else {
        EvalRequest::ENERGY_GRAD
    };
    let out = calc.evaluate(&sys, &request).unwrap();
    let h = 1e-4;
    for k in [0, 3, 5, 8, 10, 27 - 6] {
        let mut up = x.clone();
        up[k] += h;
        let mut down = x.clone();
        down[k] -= h;
        let (su, sd) = (
            sys.with_flat_coords_bohr(&up).unwrap(),
            sys.with_flat_coords_bohr(&down).unwrap(),
        );
        let (ou, od) = (
            calc.evaluate(&su, &EvalRequest::ENERGY_GRAD).unwrap(),
            calc.evaluate(&sd, &EvalRequest::ENERGY_GRAD).unwrap(),
        );
        let numeric = (ou.energy.value() - od.energy.value()) / (2.0 * h);
        assert!(
            (numeric - out.gradient_bohr[k]).abs() < 1e-6,
            "{k}: {numeric} {}",
            out.gradient_bohr[k]
        );
        if hessian {
            let column: Col<f64> = (&ou.gradient_bohr - &od.gradient_bohr) * faer::Scale(0.5 / h);
            let hess = out.hessian_bohr.as_ref().unwrap();
            for r in 0..x.nrows() {
                assert!(
                    (column[r] - hess[(r, k)]).abs() < 5e-5,
                    "{r} {k}: {} {}",
                    column[r],
                    hess[(r, k)]
                );
            }
        }
    }
}

#[test]
fn gradients_and_hessians_against_differences_with_both_kinds_of_link_atoms() {
    for placement in [LinkPlacement::Scaled, LinkPlacement::Distance] {
        let calc = oniom(&["gfn0xtb", "gfnff"], vec![hydroxyl()], placement);
        difference_check(&calc, false);
    }
    // Hessians, with the second-order term of the links at a fixed distance (the two levels
    // differ, so the terms of the model system do not cancel).
    for placement in [LinkPlacement::Scaled, LinkPlacement::Distance] {
        let calc = oniom(&["gfn0xtb", "gfnff"], vec![hydroxyl()], placement);
        difference_check(&calc, true);
    }
}

#[test]
fn given_link_atoms_pair_with_the_nearest_atom_and_are_checked() {
    let sys = ethanol();
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: ids(&[2, 3, 4, 8, 9]),
            links: LinkSpec::Given(ids(&[1])),
            spin_charge: Some(SpinCharge::new(0, 1)),
        }],
        lower_levels: vec!["gfnff".into()],
        lower_options: Vec::new(),
        placement: LinkPlacement::Scaled,
        traj_high_only: false,
        micro: None,
        embedding: None,
    };
    let regions = spec.resolve(&sys).unwrap();
    assert_eq!(regions[0].links[0].inner.one_based(), 2);
    // The link atom sits on the C-C bond at the share g.
    let model = regions[0].model(&sys).unwrap();
    let (c2, link) = (
        sys.position(AtomId::from_one_based(2).unwrap()).value(),
        model.position(AtomId::new(5)).value(),
    );
    let d = ((0..3).map(|k| (link[k] - c2[k]).powi(2)).sum::<f64>()).sqrt();
    let c1 = sys.position(AtomId::new(0)).value();
    let bond = ((0..3).map(|k| (c1[k] - c2[k]).powi(2)).sum::<f64>()).sqrt();
    assert!((d / bond - regions[0].links[0].factor).abs() < 1e-12);
    // A link atom inside the region is refused.
    let wrong = OniomSpec {
        regions: vec![RegionSpec {
            links: LinkSpec::Given(ids(&[3])),
            ..spec.regions[0].clone()
        }],
        ..spec
    };
    assert!(wrong.resolve(&sys).is_err());
}

/// Ethanol with GFN1-xTB on the hydroxymethyl group and GFN-FF on the whole molecule, with
/// electronic embedding by `boundary`.
fn embedded(boundary: embedding::BoundaryCharges) -> Box<dyn Calculator> {
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: ids(&[2, 3, 4, 8, 9]),
            links: LinkSpec::Auto,
            spin_charge: None,
        }],
        lower_levels: vec!["gfnff".into()],
        lower_options: Vec::new(),
        placement: LinkPlacement::Scaled,
        traj_high_only: false,
        micro: None,
        embedding: Some(embedding::ElectronicEmbedding {
            charges: embedding::ChargeSource::Lowest,
            boundary,
        }),
    };
    let options = CalculatorOptions {
        oniom: Some(spec),
        ..CalculatorOptions::default()
    };
    CalculatorRegistry::native()
        .create_oniom("gfn1xtb", &options, &ethanol())
        .unwrap()
        .at_given_structures()
}

#[test]
fn electronic_embedding_gradients_agree_with_differences_for_every_boundary_rule() {
    let sys = ethanol();
    let mechanical = {
        let spec = OniomSpec {
            regions: vec![RegionSpec {
                atoms: ids(&[2, 3, 4, 8, 9]),
                links: LinkSpec::Auto,
                spin_charge: None,
            }],
            lower_levels: vec!["gfnff".into()],
            lower_options: Vec::new(),
            placement: LinkPlacement::Scaled,
            traj_high_only: false,
            micro: None,
            embedding: None,
        };
        let options = CalculatorOptions {
            oniom: Some(spec),
            ..CalculatorOptions::default()
        };
        CalculatorRegistry::native()
            .create_oniom("gfn1xtb", &options, &sys)
            .unwrap()
            .at_given_structures()
    };
    let e_me = mechanical
        .evaluate(&sys, &EvalRequest::ENERGY_GRAD)
        .unwrap()
        .energy
        .value();
    for boundary in [
        embedding::BoundaryCharges::Redistributed,
        embedding::BoundaryCharges::Zero,
        embedding::BoundaryCharges::Keep,
    ] {
        let calc = embedded(boundary);
        assert!(calc.description().contains("electronic embedding"));
        let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        assert!((out.energy.value() - e_me).abs() > 1e-6, "{boundary:?}");
        let x0 = sys.coords_flat_bohr();
        let h = 1.0e-4;
        let e = |x: &Col<f64>| {
            calc.evaluate(
                &sys.with_flat_coords_bohr(x).unwrap(),
                &EvalRequest::ENERGY_GRAD,
            )
            .unwrap()
            .energy
            .value()
        };
        for k in 0..x0.nrows() {
            let mut xp = x0.clone();
            xp[k] += h;
            let mut xm = x0.clone();
            xm[k] -= h;
            let fd = (e(&xp) - e(&xm)) / (2.0 * h);
            let g = out.gradient_bohr[k];
            assert!((fd - g).abs() < 2e-6, "{boundary:?} {k}: {fd} {g}");
        }
    }
    // The Hessian from differences of the gradient is symmetric and matches a column of
    // differences.
    let calc = embedded(embedding::BoundaryCharges::Redistributed);
    let with = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
    let hess = with.hessian_bohr.unwrap();
    assert_eq!(hess.nrows(), sys.dof());
    for i in 0..sys.dof() {
        for j in 0..sys.dof() {
            assert!((hess[(i, j)] - hess[(j, i)]).abs() < 1e-12);
        }
    }
}

#[test]
fn electronic_embedding_is_refused_where_it_cannot_be() {
    // GFN0-xTB takes no point charges.
    let spec = OniomSpec {
        regions: vec![RegionSpec {
            atoms: ids(&[2, 3, 4, 8, 9]),
            links: LinkSpec::Auto,
            spin_charge: None,
        }],
        lower_levels: vec!["gfnff".into()],
        lower_options: Vec::new(),
        placement: LinkPlacement::Scaled,
        traj_high_only: false,
        micro: None,
        embedding: Some(embedding::ElectronicEmbedding::default()),
    };
    let options = CalculatorOptions {
        oniom: Some(spec.clone()),
        ..CalculatorOptions::default()
    };
    let registry = CalculatorRegistry::native();
    let refused = registry
        .create_oniom("gfn0xtb", &options, &ethanol())
        .err()
        .unwrap();
    assert!(
        refused.to_string().contains("takes no point charges"),
        "{refused}"
    );
    // A lowest level without charges (Lennard-Jones has none).
    let lj = OniomSpec {
        lower_levels: vec!["lj".into()],
        ..spec
    };
    let options = CalculatorOptions {
        oniom: Some(lj),
        ..CalculatorOptions::default()
    };
    let refused = registry
        .create_oniom("gfn1xtb", &options, &ethanol())
        .err()
        .unwrap();
    assert!(refused.to_string().contains("has no charges"), "{refused}");
}

#[test]
fn a_link_atom_that_changes_the_parity_keeps_a_high_spin_state() {
    // A methyl radical (CH3, 9 electrons) as the model of a larger structure whose parity
    // the link atom does not change, and one where it does (C2H5 with one more H: even).
    let odd = [Element::C, Element::H, Element::H, Element::H];
    let even = [Element::C, Element::H, Element::H, Element::H, Element::H];
    // Same parity: the state of the whole structure as it is.
    assert_eq!(
        layout::model_state(&odd, SpinCharge::new(0, 2)),
        SpinCharge::new(0, 2)
    );
    assert_eq!(
        layout::model_state(&odd, SpinCharge::new(0, 4)),
        SpinCharge::new(0, 4)
    );
    // Another parity: singlet <-> doublet, a triplet becomes a quartet (not a doublet), a
    // quartet a quintet.
    assert_eq!(
        layout::model_state(&odd, SpinCharge::new(0, 1)),
        SpinCharge::new(0, 2)
    );
    assert_eq!(
        layout::model_state(&even, SpinCharge::new(0, 2)),
        SpinCharge::new(0, 1)
    );
    assert_eq!(
        layout::model_state(&odd, SpinCharge::new(0, 3)),
        SpinCharge::new(0, 4)
    );
    assert_eq!(
        layout::model_state(&even, SpinCharge::new(0, 4)),
        SpinCharge::new(0, 5)
    );
    // The charge is that of the whole structure.
    assert_eq!(
        layout::model_state(&even, SpinCharge::new(1, 3)),
        SpinCharge::new(1, 4)
    );
    assert_eq!(
        layout::model_state(&odd, SpinCharge::new(-1, 3)),
        SpinCharge::new(-1, 3)
    );
}
