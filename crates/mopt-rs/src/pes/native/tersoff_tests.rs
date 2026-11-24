//! Tests of the Tersoff potential (`tersoff.rs`): the parameter files, and the energies
//! and virials of the atoms against the energy, the stress, and the heat current.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::*;

#[test]
fn lammps_format_round_trip_and_missing_pairs() {
    let si = Element::parse_token("Si").unwrap();
    let t = Tersoff::tersoff_1989(&[si]).unwrap();
    let p = t.params(si, si, si).unwrap();
    assert!((p.r - 2.85).abs() < 1e-12 && (p.d_cut - 0.15).abs() < 1e-12);
    let text = "# Si\nSi Si Si 3.0 1.0 0.0 1.0039e5 16.217 -0.59825 0.78734 1.1e-6 1.7322 471.18 2.85 0.15 2.4799 1830.8\n";
    let parsed = Tersoff::from_lammps_str(text).unwrap();
    let q = parsed.params(si, si, si).unwrap();
    let fields = |t: &TersoffParams| {
        [
            t.m, t.gamma, t.lambda3, t.c, t.d, t.h, t.n, t.beta, t.lambda2, t.b, t.r, t.d_cut,
            t.lambda1, t.a,
        ]
    };
    for (x, y) in fields(p).iter().zip(fields(q)) {
        assert!((x - y).abs() <= 1e-12 * x.abs().max(1.0), "{x} vs {y}");
    }
    let c = Element::C;
    let ge = Element::parse_token("Ge").unwrap();
    assert!(Tersoff::tersoff_1989(&[c, ge]).is_err());
}

/// Silicon in the cell of `vectors` (Angstrom) at the `fractions`, moved off them.
fn silicon(vectors: [[f64; 3]; 3], fractions: &[[f64; 3]]) -> MolecularSystem {
    use crate::domain::{positions, Angstrom, PeriodicCell, SpinCharge, Vector3};
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new(vectors[0]),
        Vector3::<Angstrom>::new(vectors[1]),
        Vector3::<Angstrom>::new(vectors[2]),
        [true, true, true],
    )
    .unwrap();
    let points: Vec<[f64; 3]> = fractions
        .iter()
        .enumerate()
        .map(|(k, f)| {
            let k = k as f64;
            let f = [
                f[0] + 0.02 * k.sin(),
                f[1] + 0.015 * (2.0 * k).cos(),
                f[2] - 0.01 * (3.0 * k).sin(),
            ];
            std::array::from_fn(|c| {
                f[0] * vectors[0][c] + f[1] * vectors[1][c] + f[2] * vectors[2][c]
            })
        })
        .collect();
    let si = Element::parse_token("Si").unwrap();
    MolecularSystem::new(
        vec![si; points.len()],
        &positions::<Angstrom>(&points),
        SpinCharge::default(),
        Some(cell),
    )
    .unwrap()
}

#[test]
fn the_site_terms_add_up_to_the_energy_and_the_virial() {
    let base = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut diamond: Vec<[f64; 3]> = base.to_vec();
    diamond.extend(base.iter().map(|f| f.map(|c| c + 0.25)));
    let a = 5.431;
    let cubic = silicon([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], &diamond);
    // The primitive cell of two atoms so compressed that the atoms have their own
    // images within the cutoff (2.83 Angstrom apart, the cutoff 3.0).
    let h = 2.0;
    let primitive = silicon(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        &[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    );
    let si = Element::parse_token("Si").unwrap();
    let calc = Tersoff::tersoff_1989(&[si]).unwrap();
    for sys in [cubic, primitive] {
        let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        let terms = calc.site_terms(&sys).unwrap().unwrap();
        let e: f64 = terms.energies.iter().sum();
        assert!((e - out.energy.value()).abs() < 1e-12, "{e}");
        let volume = sys.active_cell().unwrap().volume_bohr3();
        let stress = out.stress.unwrap();
        for a in 0..3 {
            for b in 0..3 {
                let w: f64 = terms.virials.iter().map(|w| w[a][b]).sum();
                let expected = -stress[a][b] * volume;
                assert!((w - expected).abs() < 1e-10, "{a}{b}: {w} vs {expected}");
            }
        }
    }
}

#[test]
fn the_heat_current_of_a_cluster_is_the_rate_of_change_of_the_energy_moment() {
    // Without a cell, J = sum_i (e_i v_i + W_i v_i) is d/dt sum_i r_i e_i exactly, which
    // checks the virials of the single atoms, not only their sum.
    use crate::domain::{positions, Angstrom, SpinCharge};
    let si = Element::parse_token("Si").unwrap();
    let calc = Tersoff::tersoff_1989(&[si]).unwrap();
    let sys = MolecularSystem::new(
        vec![si; 5],
        &positions::<Angstrom>(&[
            [0.0, 0.0, 0.0],
            [2.3, 0.2, 0.1],
            [1.0, 2.1, -0.3],
            [1.2, 0.8, 2.0],
            [-1.1, 1.4, 1.5],
        ]),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    let n = sys.dof();
    let m = 28.0855 * Units::AMU_TO_AU_MASS;
    let x0 = sys.coords_flat_bohr();
    let v0 = Col::from_fn(n, |k| 1e-3 * ((k * 7 % 5) as f64 - 2.0));
    let g = calc
        .evaluate(&sys, &EvalRequest::ENERGY_GRAD)
        .unwrap()
        .gradient_bohr;
    let acc = Col::from_fn(n, |k| -g[k] / m);
    // sum_i r_i e_i and the heat current along the trajectory at time t (atomic units).
    let at = |t: f64| {
        let x = Col::from_fn(n, |k| x0[k] + v0[k] * t + 0.5 * acc[k] * t * t);
        let v = Col::from_fn(n, |k| v0[k] + acc[k] * t);
        let terms = calc
            .site_terms(&sys.with_flat_coords_bohr(&x).unwrap())
            .unwrap()
            .unwrap();
        let (mut moment, mut current) = ([0.0; 3], [0.0; 3]);
        for (i, (u, w)) in terms.energies.iter().zip(&terms.virials).enumerate() {
            let vi = [v[3 * i], v[3 * i + 1], v[3 * i + 2]];
            let e = 0.5 * m * (vi[0] * vi[0] + vi[1] * vi[1] + vi[2] * vi[2]) + u;
            for a in 0..3 {
                moment[a] += x[3 * i + a] * e;
                current[a] += e * vi[a] + (0..3).map(|b| w[a][b] * vi[b]).sum::<f64>();
            }
        }
        (moment, current)
    };
    let h = 0.5;
    let ((after, _), (before, _), (_, current)) = (at(h), at(-h), at(0.0));
    let size = libm::sqrt(current.iter().map(|c| c * c).sum::<f64>());
    for a in 0..3 {
        let rate = (after[a] - before[a]) / (2.0 * h);
        assert!(
            (rate - current[a]).abs() < 1e-5 * size,
            "{a}: {rate} vs {}",
            current[a]
        );
    }
}
