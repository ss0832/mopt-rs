//! The periodic GFN-FF of mopt-rs against xtb 6.7.1: every term, the gradient, and the strain
//! derivative of the cases of `tests/golden/gfnff_pbc_reference.json` (molecular crystals,
//! covalent networks, chains and sheets in a three-dimensional cell, molecules in a box, the
//! XeF2 crystal), and finite differences of the energy under strain.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;

use mopt_rs::domain::{Bohr, Element, MolecularSystem, PeriodicCell, SpinCharge, Vector3};
use mopt_rs::pes::native::gfnff::HbCutoff;
use mopt_rs::pes::native::Gfnff;

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn rows(v: &JsonValue) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(floats).collect()
}

/// The structure of a case (lattice rows are the lattice vectors).
fn system(case: &JsonValue) -> MolecularSystem {
    let symbols = case.get("symbols").unwrap().as_string_vec().unwrap();
    let el: Vec<Element> = symbols
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect();
    let x = floats(case.get("positions_bohr").unwrap());
    let l = rows(case.get("lattice_bohr").unwrap());
    let v = |k: usize| Vector3::<Bohr>::new([l[k][0], l[k][1], l[k][2]]);
    let cell = PeriodicCell::from_lattice_vectors(v(0), v(1), v(2), [true; 3]).unwrap();
    let charge = case.get("charge").unwrap().as_f64().unwrap() as i32;
    let multiplicity =
        if (el.iter().map(|e| e.atomic_number() as i32).sum::<i32>() - charge) % 2 == 0 {
            1
        } else {
            2
        };
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge {
            charge,
            multiplicity,
        },
        Some(cell),
    )
    .unwrap()
}

fn cases() -> Vec<JsonValue> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/gfnff_pbc_reference.json"
    );
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    doc.get("cases").unwrap().as_array().unwrap().to_vec()
}

#[test]
fn periodic_cases_match_xtb() {
    let mut failures = Vec::new();
    for case in cases() {
        let name = case.get("name").unwrap().as_str().unwrap().to_string();
        // Graphene 3x3 holds the Dirac point: its Hueckel levels at the Fermi level are
        // degenerate up to xtb's perturbation of 1e-9, so the pi bond orders depend on how an
        // eigensolver resolves that subspace (LAPACK in xtb), at the 1e-8 level of the bond
        // energy.
        let loose = if name == "graphene_3x3" { 100.0 } else { 1.0 };
        let sys = system(&case);
        // xtb's energy of a single point: the terms of the hydrogen and halogen bonds
        // dropped at the thresholds of their lists.
        let sharp = Gfnff::default().with_hb_cutoff(HbCutoff::Sharp);
        let r = match sharp.compute(&sys) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        let e = r.energy;
        // The default switches them off smoothly below the thresholds; a crystal has pairs
        // at every distance, so its energy changes more than that of a molecule (cubic ice:
        // 8.9e-6 Hartree, 1e-4 of its hydrogen bond energy).
        let switched = Gfnff::default().compute(&sys).unwrap().energy.total;
        if (switched - e.total).abs() > 2.0e-5 {
            failures.push(format!(
                "{name}: the switching changes the energy by {:.3e}",
                switched - e.total
            ));
        }
        let terms = case.get("terms").unwrap();
        let mut line = format!("{name:<24} switching {:+.1e}", switched - e.total);
        for (key, got) in [
            ("bond", e.bond),
            ("angle", e.angle),
            ("torsion", e.torsion),
            ("repulsion", e.repulsion),
            ("electrostatic", e.electrostatic),
            ("dispersion", e.dispersion),
            ("hbond", e.hbond),
            ("xbond", e.xbond),
            ("batm", e.batm),
            ("total", e.total),
        ] {
            let want = terms.get(key).unwrap().as_f64().unwrap();
            line.push_str(&format!(" {key} {:+.1e}", got - want));
            // The terms are printed by xtb with 12 decimals.
            if (got - want).abs() > 2.0e-10 * loose {
                failures.push(format!("{name} {key}: {got:.12} vs xtb {want:.12}"));
            }
        }
        let g_ref = floats(case.get("gradient").unwrap());
        let dg = (0..g_ref.len())
            .map(|k| (r.gradient[k] - g_ref[k]).abs())
            .fold(0.0, f64::max);
        line.push_str(&format!(" |dg| {dg:.1e}"));
        if dg > 1.0e-8 * loose {
            failures.push(format!("{name}: gradient off by {dg:.2e}"));
        }
        // xtb's sigma is dE/d strain; the calculator gives sigma / V.
        let s_ref = rows(case.get("sigma").unwrap());
        let volume = sys.active_cell().unwrap().volume_bohr3();
        let stress = r.stress.unwrap();
        let ds = (0..3)
            .flat_map(|a| (0..3).map(move |b| (a, b)))
            .map(|(a, b)| (stress[a][b] * volume - s_ref[a][b]).abs())
            .fold(0.0, f64::max);
        line.push_str(&format!(" |dsigma| {ds:.1e}"));
        if ds > 1.0e-8 * loose {
            failures.push(format!("{name}: sigma off by {ds:.2e}"));
        }
        let q_ref = floats(case.get("eeq").unwrap());
        let dq = q_ref
            .iter()
            .zip(&r.charges)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        line.push_str(&format!(" |dq| {dq:.1e}"));
        // The numbers of bonds, bends, and torsions of xtb's lists (xtb writes one empty
        // torsion entry when there are none).
        let topo = mopt_rs::pes::native::gfnff::GfnffTopology::new(&sys).unwrap();
        let (na, nt, _) = topo.term_counts();
        let counts = case.get("counts").unwrap();
        let want = |k: &str| counts.get(k).unwrap().as_f64().unwrap() as usize;
        for (what, got, n) in [
            ("bonds", topo.bonds().len(), want("blist")),
            ("bends", na, want("alist")),
            ("torsions", nt.max(1), want("tlist")),
        ] {
            if got != n {
                failures.push(format!("{name}: {got} {what}, xtb {n}"));
            }
        }
        eprintln!("{line}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_strain_derivative_is_that_of_the_energy() {
    // Central differences of the energy under a symmetric strain of positions and lattice.
    for name in [
        "water_ice_ic",
        "benzene_crystal",
        "diamond_conventional",
        "polyethylene_z1",
        "xenon_difluoride_crystal",
    ] {
        let case = cases()
            .into_iter()
            .find(|c| c.get("name").unwrap().as_str() == Some(name))
            .unwrap();
        let sys = system(&case);
        let r = Gfnff::default().compute(&sys).unwrap();
        let volume = sys.active_cell().unwrap().volume_bohr3();
        let strained = |a: usize, b: usize, e: f64| {
            let mut m = [[0.0; 3]; 3];
            for (k, row) in m.iter_mut().enumerate() {
                row[k] = 1.0;
            }
            m[a][b] += 0.5 * e;
            m[b][a] += 0.5 * e;
            let x = sys.coords_flat_bohr();
            let moved = Col::from_fn(x.nrows(), |k| {
                let (i, c) = (k / 3, k % 3);
                (0..3).map(|d| m[c][d] * x[3 * i + d]).sum()
            });
            let cell = sys.active_cell().unwrap();
            let h = cell.h_bohr;
            let v = |k: usize| {
                Vector3::<Bohr>::new([0, 1, 2].map(|c| (0..3).map(|d| m[c][d] * h[d][k]).sum()))
            };
            let mut s = sys.with_flat_coords_bohr(&moved).unwrap();
            s.cell = Some(PeriodicCell::from_lattice_vectors(v(0), v(1), v(2), [true; 3]).unwrap());
            Gfnff::default().compute(&s).unwrap().energy.total
        };
        let e = 1.0e-6;
        let stress = r.stress.unwrap();
        for (a, b) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)] {
            let numeric = (strained(a, b, e) - strained(a, b, -e)) / (2.0 * e);
            let analytic = 0.5 * (stress[a][b] + stress[b][a]) * volume;
            assert!(
                (numeric - analytic).abs() < 5.0e-7 * (1.0 + numeric.abs()),
                "{name} sigma[{a}][{b}]: numeric {numeric}, analytic {analytic}"
            );
        }
    }
}

#[test]
fn optmain_relaxes_a_crystal_with_its_cell() {
    // Cubic ice with its cell (-vc): the energy goes down and the cell is written back.
    let case = cases()
        .into_iter()
        .find(|c| c.get("name").unwrap().as_str() == Some("water_ice_ic"))
        .unwrap();
    let sys = system(&case);
    let start = Gfnff::default().compute(&sys).unwrap().energy.total;
    let dir = std::env::temp_dir().join(format!("mopt_gfnff_vc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("ice.xyz");
    std::fs::write(&input, mopt_rs::io::xyz::format_xyz_system(&sys, None)).unwrap();
    let argv: Vec<String> = [input.to_str().unwrap(), "-os", "gfnff", "-vc", "-ns", "6"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        mopt_rs::cli::run_command("optmain", &argv),
        mopt_rs::cli::EXIT_OK
    );
    let mut logs = Vec::new();
    let mut stack = vec![dir.clone()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap().flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "mopt.log") {
                logs.push(p);
            }
        }
    }
    let log = std::fs::read_to_string(&logs[0]).unwrap();
    let energies: Vec<f64> = log
        .lines()
        .filter(|l| l.starts_with("ENERGY  "))
        .filter_map(|l| l.split(':').nth(1))
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let last = *energies.last().unwrap();
    assert!(last < start - 1.0e-4, "start {start}, last {last}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Two water molecules in a cubic cell of 40 Bohr, their oxygen atoms `d` Bohr apart through
/// the boundary of the cell along x.
fn waters_across_the_boundary(d: f64) -> MolecularSystem {
    let side = 40.0;
    let water = [[0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0]];
    let mut x = Vec::new();
    for origin in [2.0, 2.0 + side - d] {
        for atom in water {
            x.extend([atom[0] + origin, atom[1] + 20.0, atom[2] + 20.0]);
        }
    }
    let v = |k: usize| {
        let mut a = [0.0; 3];
        a[k] = side;
        Vector3::<Bohr>::new(a)
    };
    let cell = PeriodicCell::from_lattice_vectors(v(0), v(1), v(2), [true; 3]).unwrap();
    let el = ["O", "H", "H", "O", "H", "H"]
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect();
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge {
            charge: 0,
            multiplicity: 1,
        },
        Some(cell),
    )
    .unwrap()
}

#[test]
fn hydrogen_bond_terms_of_images_end_without_a_step() {
    // The pair of the two oxygen atoms through the boundary crosses the threshold of the
    // lists (250 Bohr^2 for its squared distance).
    let threshold = 250.0_f64.sqrt();
    let step_of = |cutoff: HbCutoff| {
        // One topology for all distances, as in a run.
        let calc =
            Gfnff::starting_from(&waters_across_the_boundary(threshold)).with_hb_cutoff(cutoff);
        let e = |d: f64| {
            calc.compute(&waters_across_the_boundary(d))
                .unwrap()
                .energy
                .total
        };
        let h = 1.0e-5;
        let across = e(threshold + h) - e(threshold - h);
        let before = e(threshold - h) - e(threshold - 3.0 * h);
        (across - before).abs()
    };
    let sharp = step_of(HbCutoff::Sharp);
    let switched = step_of(HbCutoff::default());
    eprintln!("step of the energy at the threshold: sharp {sharp:.3e}, switched {switched:.3e}");
    assert!(sharp > 1.0e-11, "step of the sharp lists {sharp:.3e}");
    assert!(switched < 1.0e-12, "step with the switching {switched:.3e}");

    // Inside the range of the switching the stress is the strain derivative of the energy.
    let sys = waters_across_the_boundary(15.4);
    let calc = Gfnff::starting_from(&sys);
    let r = calc.compute(&sys).unwrap();
    let volume = sys.active_cell().unwrap().volume_bohr3();
    let stretched = |e: f64| {
        let x = sys.coords_flat_bohr();
        let moved = Col::from_fn(
            x.nrows(),
            |k| {
                if k % 3 == 0 {
                    (1.0 + e) * x[k]
                } else {
                    x[k]
                }
            },
        );
        let v = |k: usize| {
            let mut a = [0.0; 3];
            a[k] = if k == 0 { 40.0 * (1.0 + e) } else { 40.0 };
            Vector3::<Bohr>::new(a)
        };
        let mut s = sys.with_flat_coords_bohr(&moved).unwrap();
        s.cell = Some(PeriodicCell::from_lattice_vectors(v(0), v(1), v(2), [true; 3]).unwrap());
        calc.compute(&s).unwrap().energy.total
    };
    let e = 1.0e-6;
    let numeric = (stretched(e) - stretched(-e)) / (2.0 * e);
    let analytic = r.stress.unwrap()[0][0] * volume;
    assert!(
        (numeric - analytic).abs() < 5.0e-7 * (1.0 + numeric.abs()),
        "numeric {numeric}, analytic {analytic}"
    );
}
