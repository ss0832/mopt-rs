//! Phonons of a periodic structure at any wave vector, from the force constants of a
//! supercell.
//!
//! The Hessian of a supercell of `n_1 x n_2 x n_3` primitive cells gives the force constants
//! `Phi(0 k, R k')` between atom `k` of one cell and atom `k'` of the cell at the lattice
//! vector `R`. The dynamical matrix at the wave vector `q` is
//!
//! ```text
//! D(q)_(k a, k' b) = (m_k m_k')^(-1/2) sum_R Phi(0 k a, R k' b) exp(i q . R)
//! ```
//!
//! and its eigenvalues are the squares of the phonon frequencies at `q` (a negative one: an
//! imaginary frequency, the structure is unstable against that wave).
//!
//! In a supercell the atom `k'` of cell `R` stands for all its periodic images `R + L`.
//! Each force constant is assigned to the image nearest to atom `k`; when several images are
//! equally near (the atom lies on the boundary of the Wigner-Seitz cell of the supercell) it
//! is shared between them. This keeps the dynamical matrix Hermitian and the dispersion
//! symmetric. The result is exact at the wave vectors commensurate with the supercell and an
//! interpolation between them; it converges with the size of the supercell as the force
//! constants decay.
//!
//! The acoustic sum rule `sum_(R k') Phi(0 k, R k') = 0` (no force on a uniform translation)
//! is imposed on the block of every atom with itself, which removes what numerical noise
//! leaves of the three acoustic frequencies at the zone centre.
//!
//! No correction for the splitting of longitudinal and transverse optical modes in polar
//! crystals is made (it needs Born effective charges and the dielectric tensor).
//!
//! From the frequencies on a mesh of wave vectors: the density of states (Gaussians of a
//! given width) and, per primitive cell, the zero-point energy, the free energy, the
//! entropy, and the heat capacity of the harmonic crystal.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::curvature_to_hz;
use crate::domain::{Bohr, MolecularSystem, PeriodicCell, Units, Vector3};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{symmetric_eigen, symmetrize};

/// The supercell of `sys` with `repeats` cells along the three lattice vectors: the atoms
/// of the cell at `(i, j, k)` follow those of the cell before it, the first index running
/// slowest, so that atom `a` of that cell has the index `((i n_2 + j) n_3 + k) N + a`.
pub fn supercell(sys: &MolecularSystem, repeats: [usize; 3]) -> MoptResult<MolecularSystem> {
    let cell = periodic_cell(sys)?;
    check_repeats(cell, repeats)?;
    let natoms = sys.n_atoms();
    let x = sys.coords_flat_bohr();
    let count = repeats[0] * repeats[1] * repeats[2];
    let mut elements = Vec::with_capacity(natoms * count);
    let mut flat = Vec::with_capacity(3 * natoms * count);
    for i in 0..repeats[0] {
        for j in 0..repeats[1] {
            for k in 0..repeats[2] {
                let shift = cell.frac_to_cart([i as f64, j as f64, k as f64]);
                for a in 0..natoms {
                    elements.push(sys.elements[a]);
                    flat.extend((0..3).map(|c| x[3 * a + c] + shift[c]));
                }
            }
        }
    }
    let vector = |axis: usize| -> Vector3<Bohr> {
        let scale = repeats[axis] as f64;
        Vector3::new(std::array::from_fn(|c| cell.h_bohr[c][axis] * scale))
    };
    let large =
        PeriodicCell::from_lattice_vectors(vector(0), vector(1), vector(2), cell.periodic_axes)?;
    MolecularSystem::from_flat_bohr(
        elements,
        &Col::from_fn(flat.len(), |i| flat[i]),
        sys.spin_charge,
        Some(large),
    )
}

fn periodic_cell(sys: &MolecularSystem) -> MoptResult<&PeriodicCell> {
    sys.cell
        .as_ref()
        .filter(|c| c.is_any_periodic())
        .ok_or_else(|| MoptError::config("phonons need a periodic cell"))
}

fn check_repeats(cell: &PeriodicCell, repeats: [usize; 3]) -> MoptResult<()> {
    for (axis, count) in repeats.iter().enumerate() {
        if *count == 0 || (!cell.periodic_axes[axis] && *count != 1) {
            return Err(MoptError::config(format!(
                "supercell: {count} cells along lattice vector {} (at least 1, and 1 along a \
                 direction that is not periodic)",
                axis + 1
            )));
        }
    }
    Ok(())
}

/// Force constants of a periodic structure in real space.
#[derive(Debug, Clone)]
pub struct ForceConstants {
    /// Atoms of the primitive cell.
    natoms: usize,
    /// `1 / sqrt(m)` of every coordinate of the primitive cell.
    inv_sqrt_mass: Vec<f64>,
    /// Terms `(k, k', lattice vector R in units of the lattice vectors, weight, 3 x 3
    /// block)`: the block times the weight belongs to the pair of atom `k` in the cell at
    /// the origin and atom `k'` in the cell at `R`.
    terms: Vec<(usize, usize, [f64; 3], f64, [[f64; 3]; 3])>,
    /// Largest correction made to impose the acoustic sum rule (Hartree/Bohr^2).
    pub sum_rule_residual: f64,
}

impl ForceConstants {
    /// From the Cartesian Hessian `hessian` (Hartree/Bohr^2) of [`supercell`]`(sys, repeats)`.
    pub fn from_supercell(
        sys: &MolecularSystem,
        repeats: [usize; 3],
        hessian: &Mat<f64>,
    ) -> MoptResult<Self> {
        let cell = periodic_cell(sys)?;
        check_repeats(cell, repeats)?;
        let natoms = sys.n_atoms();
        let count = repeats[0] * repeats[1] * repeats[2];
        let size = 3 * natoms * count;
        if hessian.nrows() != size || hessian.ncols() != size {
            return Err(MoptError::config(format!(
                "force constants: the Hessian is {}x{}, expected {size}x{size} for the supercell",
                hessian.nrows(),
                hessian.ncols()
            )));
        }
        let x = sys.coords_flat_bohr();
        let position = |a: usize| -> [f64; 3] { std::array::from_fn(|c| x[3 * a + c]) };
        let masses = sys.masses_amu();
        let mut terms = Vec::new();
        // Images of the supercell to look at: one period in each periodic direction.
        let reach = |axis: usize| -> Vec<i64> {
            if cell.periodic_axes[axis] {
                vec![-1, 0, 1]
            } else {
                vec![0]
            }
        };
        let mut on_site = vec![[[0.0_f64; 3]; 3]; natoms];
        for (index, (i, j, k)) in (0..repeats[0])
            .flat_map(|i| {
                (0..repeats[1]).flat_map(move |j| (0..repeats[2]).map(move |k| (i, j, k)))
            })
            .enumerate()
        {
            for (first, site) in on_site.iter_mut().enumerate() {
                for second in 0..natoms {
                    let block: [[f64; 3]; 3] = std::array::from_fn(|a| {
                        std::array::from_fn(|b| {
                            // Symmetrized over the two orders of differentiation.
                            let (r, c) = (3 * first + a, 3 * (index * natoms + second) + b);
                            0.5 * (hessian[(r, c)] + hessian[(c, r)])
                        })
                    });
                    for a in 0..3 {
                        for b in 0..3 {
                            site[a][b] += block[a][b];
                        }
                    }
                    // The nearest images of the second atom.
                    let mut images: Vec<([f64; 3], f64)> = Vec::new();
                    for l0 in reach(0) {
                        for l1 in reach(1) {
                            for l2 in reach(2) {
                                let lattice = [
                                    i as f64 + (l0 * repeats[0] as i64) as f64,
                                    j as f64 + (l1 * repeats[1] as i64) as f64,
                                    k as f64 + (l2 * repeats[2] as i64) as f64,
                                ];
                                let shift = cell.frac_to_cart(lattice);
                                let (p, q) = (position(first), position(second));
                                let d: [f64; 3] = std::array::from_fn(|c| q[c] + shift[c] - p[c]);
                                images.push((
                                    lattice,
                                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt(),
                                ));
                            }
                        }
                    }
                    let nearest = images.iter().fold(f64::INFINITY, |m, (_, d)| m.min(*d));
                    let tolerance = 1e-6 * (1.0 + nearest);
                    let shared: Vec<[f64; 3]> = images
                        .iter()
                        .filter(|(_, d)| *d <= nearest + tolerance)
                        .map(|(lattice, _)| *lattice)
                        .collect();
                    let weight = 1.0 / shared.len() as f64;
                    for lattice in shared {
                        terms.push((first, second, lattice, weight, block));
                    }
                }
            }
        }
        // The acoustic sum rule, on the block of every atom with itself at R = 0.
        let mut residual = 0.0_f64;
        for (atom, excess) in on_site.iter().enumerate() {
            for row in excess {
                for value in row {
                    residual = residual.max(value.abs());
                }
            }
            terms.push((
                atom,
                atom,
                [0.0; 3],
                1.0,
                std::array::from_fn(|a| std::array::from_fn(|b| -excess[a][b])),
            ));
        }
        Ok(Self {
            natoms,
            inv_sqrt_mass: (0..3 * natoms)
                .map(|c| 1.0 / masses[c / 3].sqrt())
                .collect(),
            terms,
            sum_rule_residual: residual,
        })
    }

    /// Number of phonon branches, `3 N`.
    pub fn branches(&self) -> usize {
        3 * self.natoms
    }

    /// Phonon wavenumbers (cm^-1, ascending; negative: imaginary) at the wave vector `q`
    /// in units of the reciprocal lattice vectors.
    pub fn wavenumbers(&self, q: [f64; 3]) -> MoptResult<Vec<f64>> {
        let n = self.branches();
        // The Hermitian matrix D = A + i B as the real symmetric [[A, -B], [B, A]].
        let mut real = Mat::<f64>::zeros(2 * n, 2 * n);
        for (first, second, lattice, weight, block) in &self.terms {
            let phase = 2.0
                * std::f64::consts::PI
                * (q[0] * lattice[0] + q[1] * lattice[1] + q[2] * lattice[2]);
            let (sin, cos) = phase.sin_cos();
            for (a, row) in block.iter().enumerate() {
                for (b, element) in row.iter().enumerate() {
                    let (r, c) = (3 * first + a, 3 * second + b);
                    let value = weight * element * self.inv_sqrt_mass[r] * self.inv_sqrt_mass[c];
                    real[(r, c)] += value * cos;
                    real[(n + r, n + c)] += value * cos;
                    real[(n + r, c)] += value * sin;
                    real[(r, n + c)] -= value * sin;
                }
            }
        }
        symmetrize(&mut real);
        let (values, _) = symmetric_eigen(&real)?;
        let to_wavenumber = curvature_to_hz() / Units::SPEED_OF_LIGHT_M_S * 1e-2;
        // Every eigenvalue of D appears twice.
        Ok((0..n)
            .map(|k| {
                let curvature = 0.5 * (values[2 * k] + values[2 * k + 1]);
                curvature.signum() * curvature.abs().sqrt() * to_wavenumber
            })
            .collect())
    }
}

/// Wave vectors along the straight lines between `corners` (in units of the reciprocal
/// lattice vectors) with `points` steps per line, and the distance of each from the first
/// along the path in those units.
pub fn path(corners: &[[f64; 3]], points: usize) -> MoptResult<Vec<(f64, [f64; 3])>> {
    if corners.len() < 2 || points == 0 {
        return Err(MoptError::config(
            "a path of wave vectors needs two points at least and one step per line",
        ));
    }
    let mut out = vec![(0.0, corners[0])];
    let mut travelled = 0.0;
    for pair in corners.windows(2) {
        let d: [f64; 3] = std::array::from_fn(|c| pair[1][c] - pair[0][c]);
        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        for step in 1..=points {
            let t = step as f64 / points as f64;
            out.push((
                travelled + t * length,
                std::array::from_fn(|c| pair[0][c] + t * d[c]),
            ));
        }
        travelled += length;
    }
    Ok(out)
}

/// The wave vectors of a uniform mesh with `mesh` points along each reciprocal lattice
/// vector, the zone centre included.
pub fn mesh(mesh: [usize; 3]) -> MoptResult<Vec<[f64; 3]>> {
    if mesh.contains(&0) {
        return Err(MoptError::config(
            "a mesh of wave vectors needs one point at least along every direction",
        ));
    }
    let mut out = Vec::with_capacity(mesh[0] * mesh[1] * mesh[2]);
    for i in 0..mesh[0] {
        for j in 0..mesh[1] {
            for k in 0..mesh[2] {
                out.push([
                    i as f64 / mesh[0] as f64,
                    j as f64 / mesh[1] as f64,
                    k as f64 / mesh[2] as f64,
                ]);
            }
        }
    }
    Ok(out)
}

/// Density of states from the wavenumbers at the wave vectors of a mesh: `(wavenumber,
/// states per cm^-1 and primitive cell)` on a grid of spacing `step` (cm^-1), each mode a
/// Gaussian of the standard deviation `width` (cm^-1). The density integrates to the number
/// of branches.
pub fn density_of_states(
    wavenumbers: &[Vec<f64>],
    width: f64,
    step: f64,
) -> MoptResult<Vec<(f64, f64)>> {
    let positive = |v: f64| v > 0.0 && v.is_finite();
    if wavenumbers.is_empty() || !positive(width) || !positive(step) {
        return Err(MoptError::config(
            "density of states: wavenumbers on a mesh, and a positive width and step",
        ));
    }
    let all = || wavenumbers.iter().flatten().copied();
    let lowest = all().fold(f64::INFINITY, f64::min) - 4.0 * width;
    let highest = all().fold(f64::NEG_INFINITY, f64::max) + 4.0 * width;
    let count = ((highest - lowest) / step).ceil() as usize + 1;
    let norm = 1.0 / (width * (2.0 * std::f64::consts::PI).sqrt() * wavenumbers.len() as f64);
    Ok((0..count)
        .map(|i| {
            let at = lowest + step * i as f64;
            let density: f64 = all()
                .map(|w| (-0.5 * ((at - w) / width).powi(2)).exp())
                .sum();
            (at, density * norm)
        })
        .collect())
}

/// Thermodynamic functions of the harmonic crystal per primitive cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhononThermo {
    /// Temperature (K).
    pub temperature: f64,
    /// Zero-point energy (Hartree).
    pub zero_point_energy: f64,
    /// Vibrational internal energy, zero-point energy included (Hartree).
    pub internal_energy: f64,
    /// Entropy (Hartree/K).
    pub entropy: f64,
    /// Helmholtz free energy (Hartree).
    pub free_energy: f64,
    /// Heat capacity at constant volume (Hartree/K).
    pub heat_capacity: f64,
    /// Modes that entered, summed over the mesh.
    pub modes: usize,
    /// Modes left out: at or below `lowest` in magnitude (the acoustic modes at the zone
    /// centre) or imaginary.
    pub left_out: usize,
    /// Imaginary modes (`vib::is_imaginary_curvature`) among those left out.
    pub imaginary: usize,
}

/// Thermodynamic functions per primitive cell at `temperature` (K) from the wavenumbers at
/// the wave vectors of a mesh; modes at or below `lowest` (cm^-1) in magnitude and imaginary
/// ones are left out.
pub fn thermodynamics(
    wavenumbers: &[Vec<f64>],
    temperature: f64,
    lowest: f64,
) -> MoptResult<PhononThermo> {
    if wavenumbers.is_empty() || !(temperature > 0.0 && temperature.is_finite()) {
        return Err(MoptError::config(
            "phonon thermodynamics: wavenumbers on a mesh and a positive temperature",
        ));
    }
    let k = Units::BOLTZMANN_HARTREE_PER_K;
    let to_theta = Units::HC_OVER_KB_CM_K;
    let weight = 1.0 / wavenumbers.len() as f64;
    let mut out = PhononThermo {
        temperature,
        zero_point_energy: 0.0,
        internal_energy: 0.0,
        entropy: 0.0,
        free_energy: 0.0,
        heat_capacity: 0.0,
        modes: 0,
        left_out: 0,
        imaginary: 0,
    };
    for w in wavenumbers.iter().flatten() {
        if *w <= lowest {
            out.left_out += 1;
            // Imaginary by the one criterion of the crate.
            out.imaginary += super::imaginary_wavenumbers(&[*w]);
            continue;
        }
        out.modes += 1;
        let theta = w * to_theta;
        let x = theta / temperature;
        // The terms of the oscillator of the molecular thermochemistry (decaying forms, finite
        // for small and large x).
        let (entropy, thermal) = super::thermo::oscillator(x);
        out.zero_point_energy += weight * 0.5 * k * theta;
        out.internal_energy += weight * k * (0.5 * theta + temperature * thermal);
        out.entropy += weight * k * entropy;
        out.heat_capacity += weight * k * super::thermo::oscillator_heat(x);
    }
    out.free_energy = out.internal_energy - temperature * out.entropy;
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Element, SpinCharge};

    /// A chain along x with the lattice constant `a` (Bohr) and the atoms `elements` at the
    /// fractional positions `fractions`, periodic along x only.
    fn chain(a: f64, elements: &[&str], fractions: &[f64]) -> MolecularSystem {
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Bohr>::new([a, 0.0, 0.0]),
            Vector3::<Bohr>::new([0.0, 20.0, 0.0]),
            Vector3::<Bohr>::new([0.0, 0.0, 20.0]),
            [true, false, false],
        )
        .unwrap();
        let points: Vec<[f64; 3]> = fractions.iter().map(|f| [f * a, 0.0, 0.0]).collect();
        MolecularSystem::new(
            elements
                .iter()
                .map(|e| Element::parse_token(e).unwrap())
                .collect(),
            &positions::<Bohr>(&points),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap()
    }

    /// Hessian of a ring of `n` atoms along x with the springs `springs[i]` between atom `i`
    /// and the next (the last to the first), and the spring `second` between every atom and
    /// its second neighbour.
    fn ring(springs: &[f64], second: f64) -> Mat<f64> {
        let n = springs.len();
        let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
        let mut couple = |i: usize, j: usize, k: f64| {
            let (r, c) = (3 * i, 3 * j);
            h[(r, r)] += k;
            h[(c, c)] += k;
            h[(r, c)] -= k;
            h[(c, r)] -= k;
        };
        for (i, k) in springs.iter().enumerate() {
            couple(i, (i + 1) % n, *k);
            if second != 0.0 {
                couple(i, (i + 2) % n, second);
            }
        }
        h
    }

    fn to_wavenumber(curvature: f64) -> f64 {
        curvature.sqrt() * curvature_to_hz() / Units::SPEED_OF_LIGHT_M_S * 1e-2
    }

    #[test]
    fn a_monatomic_chain_has_its_sine_dispersion() {
        // w^2 = (4 k / m) sin^2(pi q) + (4 k2 / m) sin^2(2 pi q) with second neighbours.
        let sys = chain(4.0, &["Cu"], &[0.0]);
        let (k, k2) = (0.05, 0.01);
        let large = supercell(&sys, [8, 1, 1]).unwrap();
        assert_eq!(large.n_atoms(), 8);
        assert!((large.coords_flat_bohr()[3 * 5] - 20.0).abs() < 1e-12);
        let constants =
            ForceConstants::from_supercell(&sys, [8, 1, 1], &ring(&[k; 8], k2)).unwrap();
        assert!(constants.sum_rule_residual < 1e-12);
        let mass = sys.masses_amu()[0];
        // Also at wave vectors that are not commensurate with the supercell: the force
        // constants do not reach beyond it.
        for q in [0.0, 0.125, 0.2, 0.37, 0.5] {
            let found = constants.wavenumbers([q, 0.0, 0.0]).unwrap();
            let (s1, s2) = (
                (std::f64::consts::PI * q).sin(),
                (2.0 * std::f64::consts::PI * q).sin(),
            );
            let expected = to_wavenumber(4.0 * (k * s1 * s1 + k2 * s2 * s2) / mass);
            // Two branches across the chain without a restoring force, one along it.
            assert!(found[0].abs() < 1e-3 && found[1].abs() < 1e-3);
            assert!(
                (found[2] - expected).abs() < 1e-3 + 1e-6 * expected,
                "q = {q}: {} against {expected}",
                found[2]
            );
        }
        // The dispersion is symmetric in q.
        let plus = constants.wavenumbers([0.3, 0.0, 0.0]).unwrap();
        let minus = constants.wavenumbers([-0.3, 0.0, 0.0]).unwrap();
        assert!((plus[2] - minus[2]).abs() < 1e-9);
    }

    #[test]
    fn a_diatomic_chain_has_an_acoustic_and_an_optical_branch() {
        // Masses m1, m2 alternating with one spring k: w^2 = k (1/m1 + 1/m2) -+
        // k sqrt((1/m1 + 1/m2)^2 - 4 sin^2(pi q) / (m1 m2)).
        let sys = chain(6.0, &["C", "O"], &[0.0, 0.5]);
        let k = 0.08;
        let hessian = ring(&[k; 8], 0.0);
        let constants = ForceConstants::from_supercell(&sys, [4, 1, 1], &hessian).unwrap();
        let m = sys.masses_amu();
        let sum = 1.0 / m[0] + 1.0 / m[1];
        for q in [0.0, 0.1, 0.25, 0.5] {
            let found = constants.wavenumbers([q, 0.0, 0.0]).unwrap();
            let s = (std::f64::consts::PI * q).sin();
            let root = (sum * sum - 4.0 * s * s / (m[0] * m[1])).sqrt();
            let acoustic = to_wavenumber((k * (sum - root)).max(0.0));
            let optical = to_wavenumber(k * (sum + root));
            assert_eq!(found.len(), 6);
            assert!((found[4] - acoustic).abs() < 1e-3 + 1e-6 * acoustic, "{q}");
            assert!((found[5] - optical).abs() < 1e-6 * optical, "{q}");
        }
        // A Hessian of another size is refused, and so is a structure without a cell.
        assert!(ForceConstants::from_supercell(&sys, [3, 1, 1], &hessian).is_err());
        assert!(supercell(&sys, [2, 2, 1]).is_err());
    }

    #[test]
    fn density_of_states_and_thermodynamics_of_einstein_modes() {
        // Every mode at 500 cm^-1 on a mesh of four wave vectors, three branches.
        let wavenumbers = vec![vec![500.0; 3]; 4];
        let dos = density_of_states(&wavenumbers, 10.0, 1.0).unwrap();
        let integral: f64 = dos.iter().map(|(_, d)| d).sum::<f64>() * 1.0;
        assert!((integral - 3.0).abs() < 1e-3, "{integral}");
        let peak = dos
            .iter()
            .fold((0.0, 0.0), |best, p| if p.1 > best.1 { *p } else { best });
        assert!((peak.0 - 500.0).abs() < 1.0);
        let t = 300.0;
        let found = thermodynamics(&wavenumbers, t, 10.0).unwrap();
        let k = Units::BOLTZMANN_HARTREE_PER_K;
        let theta = 500.0 * Units::HC_OVER_KB_CM_K;
        let x: f64 = theta / t;
        let occupation = 1.0 / (x.exp() - 1.0);
        assert!((found.zero_point_energy - 1.5 * k * theta).abs() < 1e-15);
        assert!((found.internal_energy - 3.0 * k * theta * (0.5 + occupation)).abs() < 1e-14);
        let free = 3.0 * (0.5 * k * theta + k * t * (1.0 - (-x).exp()).ln());
        assert!((found.free_energy - free).abs() < 1e-14);
        let heat = 3.0 * k * x * x * x.exp() / (x.exp() - 1.0).powi(2);
        assert!((found.heat_capacity - heat).abs() < 1e-14);
        assert_eq!((found.modes, found.left_out, found.imaginary), (12, 0, 0));
        // Acoustic modes at the zone centre and imaginary modes are left out and counted.
        let mixed = vec![vec![0.0, 0.0, 0.0, 500.0], vec![-80.0, 200.0, 300.0, 500.0]];
        let found = thermodynamics(&mixed, t, 10.0).unwrap();
        assert_eq!((found.modes, found.left_out, found.imaginary), (4, 4, 1));
        // Paths and meshes of wave vectors.
        let line = path(&[[0.0; 3], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]], 5).unwrap();
        assert_eq!(line.len(), 11);
        assert!((line[10].0 - 1.0).abs() < 1e-12 && line[5].1 == [0.5, 0.0, 0.0]);
        assert_eq!(mesh([2, 3, 1]).unwrap().len(), 6);
        assert!(mesh([0, 1, 1]).is_err() && path(&[[0.0; 3]], 5).is_err());
    }
}
