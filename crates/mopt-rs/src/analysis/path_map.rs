//! Maps of a series of structures in few dimensions: classical multidimensional scaling
//! of their mass-weighted distances, and principal component analysis of their
//! coordinates.
//!
//! Classical multidimensional scaling places `n` structures in a plane (or in `k`
//! dimensions) so that their distances there come as close as possible to the distances
//! `d_ij` of the structures themselves: the matrix `B = -J D2 J / 2` (`D2` the squared
//! distances, `J = 1 - 11^T / n`) is diagonalized and the coordinates are
//! `sqrt(lambda_k) v_k` for its largest eigenvalues. The distance of two structures is
//! `sqrt(sum_i m_i |r_i - r'_i|^2)` after the mass-weighted superposition of one on the
//! other. Applied to the structures of a reaction path this is the picture of Tsutsumi,
//! Ono, and Taketsugu (Chem. Commun. 2021, 57, 11734).
//!
//! Principal component analysis finds the directions of largest variance of a set of
//! variables over the structures; three sets are offered: mass-weighted Cartesian
//! coordinates (after superposition on the first structure), all interatomic distances, and
//! a chain of distances, angles, and dihedral angles (a Z-matrix in the order of the atoms).
//!
//! MultiOptPy: `PESAnalyzer/cmds_analysis.py:CMDSPathAnalysis`,
//! `PESAnalyzer/pca_analysis.py:PCAPathAnalysis`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;
use rayon::prelude::*;

use crate::coord::alignment::kabsch_align_bohr;
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// Eigenvalues below this share of the largest count as zero.
const RANK_TOLERANCE: f64 = 1e-12;

/// Distance of two structures of the same atoms: `sqrt(sum_i m_i |r_i - r'_i|^2)` after the
/// mass-weighted superposition of `a` on `b`, in Angstrom amu^(1/2). In a cell the atoms
/// are taken to the nearest images and only the drift is removed.
pub fn mass_weighted_distance(a: &MolecularSystem, b: &MolecularSystem) -> MoptResult<f64> {
    if a.elements != b.elements {
        return Err(MoptError::config(
            "structures of different atoms cannot be compared",
        ));
    }
    let (aligned, _) = kabsch_align_bohr(a, b, true)?;
    let sum: f64 = b
        .atoms()
        .map(|atom| {
            let p = aligned.position(atom).value();
            let q = b.position(atom).value();
            let d2: f64 = (0..3).map(|k| (p[k] - q[k]).powi(2)).sum();
            b.element(atom).mass_amu() * d2
        })
        .sum();
    Ok(libm::sqrt(sum) * Units::BOHR_TO_ANGSTROM)
}

/// The distances ([`mass_weighted_distance`]) between all pairs of `structures`.
pub fn distance_matrix(structures: &[MolecularSystem]) -> MoptResult<Mat<f64>> {
    let n = structures.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();
    let values: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| mass_weighted_distance(&structures[i], &structures[j]))
        .collect::<MoptResult<_>>()?;
    let mut d = Mat::<f64>::zeros(n, n);
    for (&(i, j), value) in pairs.iter().zip(values) {
        d[(i, j)] = value;
        d[(j, i)] = value;
    }
    Ok(d)
}

/// Points in few dimensions that reproduce a matrix of distances.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Coordinates: row `i` is point `i`, column `k` the `k`-th principal coordinate.
    pub coordinates: Mat<f64>,
    /// All eigenvalues of the centered matrix, largest first.
    pub eigenvalues: Vec<f64>,
    /// Share of each coordinate in the sum of the positive eigenvalues.
    pub contributions: Vec<f64>,
    /// Sum of all eigenvalues over the sum of the positive ones: 1 when the distances are
    /// those of points in a Euclidean space, less the more they deviate from that
    /// (MultiOptPy's "dimensional reproducibility").
    pub reproducibility: f64,
}

/// Classical multidimensional scaling of the symmetric matrix `distances` to `components`
/// dimensions. Each coordinate is given the sign that puts the first point on its negative
/// side, so that a path runs from left to right.
pub fn classical_mds(distances: &Mat<f64>, components: usize) -> MoptResult<Embedding> {
    let n = distances.nrows();
    if n != distances.ncols() {
        return Err(MoptError::config("a matrix of distances must be square"));
    }
    if n < 2 {
        return Err(MoptError::config(
            "multidimensional scaling needs at least two points",
        ));
    }
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (distances[(i, j)], distances[(j, i)]);
            if !a.is_finite() || a < 0.0 || (a - b).abs() > 1e-8 * a.abs().max(1.0) {
                return Err(MoptError::config(
                    "distances must be finite, not negative, and symmetric",
                ));
            }
        }
    }
    // B = -J D2 J / 2: subtract row and column means of the squared distances.
    let square = |i: usize, j: usize| distances[(i, j)] * distances[(i, j)];
    let row_mean: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| square(i, j)).sum::<f64>() / n as f64)
        .collect();
    let mean = row_mean.iter().sum::<f64>() / n as f64;
    let b = Mat::from_fn(n, n, |i, j| {
        -0.5 * (square(i, j) - row_mean[i] - row_mean[j] + mean)
    });
    let (values, vectors) = symmetric_eigen(&b)?;
    let order: Vec<usize> = (0..n).rev().collect();
    let eigenvalues: Vec<f64> = order.iter().map(|&k| values[k]).collect();
    let positive: f64 = eigenvalues.iter().filter(|v| **v > 0.0).sum();
    if positive <= 0.0 {
        return Err(MoptError::config(
            "all points coincide: there is nothing to map",
        ));
    }
    let k = components.min(n);
    let coordinates = Mat::from_fn(n, k, |i, c| {
        let column = order[c];
        let sign = if vectors[(0, column)] > 0.0 {
            -1.0
        } else {
            1.0
        };
        sign * libm::sqrt(values[column].max(0.0)) * vectors[(i, column)]
    });
    Ok(Embedding {
        coordinates,
        contributions: eigenvalues[..k]
            .iter()
            .map(|v| v.max(0.0) / positive)
            .collect(),
        reproducibility: eigenvalues.iter().sum::<f64>() / positive,
        eigenvalues,
    })
}

/// Principal components of variables over a series of structures.
#[derive(Debug, Clone)]
pub struct PrincipalComponents {
    /// Variances along the components, largest first (as many as there are components of
    /// nonzero variance, at most the number asked for).
    pub variances: Vec<f64>,
    /// Share of each component in the total variance.
    pub contributions: Vec<f64>,
    /// Unit vectors of the components: row `v` is variable `v`, column `k` component `k`.
    /// Each has the sign that makes its largest entry positive.
    pub vectors: Mat<f64>,
    /// The structures along the components: row `i` is structure `i`.
    pub scores: Mat<f64>,
    /// Means of the variables.
    pub means: Vec<f64>,
    /// Divisors of the variables (1 without standardization).
    pub scales: Vec<f64>,
}

/// Principal component analysis of `data` (row `i`: structure `i`; column `v`: variable
/// `v`). The variables are centered and, with `standardize`, divided by their standard
/// deviations (variables that do not vary are left as they are: they contribute nothing).
/// The work is done on the `n x n` matrix of products of the structures, so that any number
/// of variables is cheap.
pub fn principal_components(
    data: &Mat<f64>,
    standardize: bool,
    components: usize,
) -> MoptResult<PrincipalComponents> {
    let (n, p) = (data.nrows(), data.ncols());
    if n < 2 || p == 0 {
        return Err(MoptError::config(
            "principal component analysis needs at least two structures and one variable",
        ));
    }
    if (0..p).any(|v| (0..n).any(|i| !data[(i, v)].is_finite())) {
        return Err(MoptError::config("a variable is not finite"));
    }
    let means: Vec<f64> = (0..p)
        .map(|v| (0..n).map(|i| data[(i, v)]).sum::<f64>() / n as f64)
        .collect();
    let scales: Vec<f64> = (0..p)
        .map(|v| {
            if !standardize {
                return 1.0;
            }
            let variance = (0..n)
                .map(|i| (data[(i, v)] - means[v]).powi(2))
                .sum::<f64>()
                / n as f64;
            let deviation = libm::sqrt(variance);
            if deviation > 1e-12 * means[v].abs().max(1.0) {
                deviation
            } else {
                1.0
            }
        })
        .collect();
    let x = Mat::from_fn(n, p, |i, v| (data[(i, v)] - means[v]) / scales[v]);
    let gram = &x * x.transpose() * (1.0 / (n - 1) as f64);
    let (values, vectors) = symmetric_eigen(&gram)?;
    let total: f64 = values.iter().filter(|v| **v > 0.0).sum();
    if total <= 0.0 {
        return Err(MoptError::config(
            "the variables do not vary: there is nothing to analyze",
        ));
    }
    let largest = values[n - 1];
    let kept: Vec<usize> = (0..n)
        .rev()
        .filter(|&k| values[k] > RANK_TOLERANCE * largest)
        .take(components)
        .collect();
    let mut loadings = Mat::<f64>::zeros(p, kept.len());
    let mut scores = Mat::<f64>::zeros(n, kept.len());
    for (c, &k) in kept.iter().enumerate() {
        let norm = libm::sqrt((n - 1) as f64 * values[k]);
        // v = X^T u / sqrt((n - 1) lambda); the scores are X v = sqrt((n - 1) lambda) u.
        let mut column: Vec<f64> = (0..p)
            .map(|v| (0..n).map(|i| x[(i, v)] * vectors[(i, k)]).sum::<f64>() / norm)
            .collect();
        let peak = column
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| if b.abs() > a.abs() { b } else { a });
        let sign = if peak < 0.0 { -1.0 } else { 1.0 };
        for value in &mut column {
            *value *= sign;
        }
        for (v, value) in column.iter().enumerate() {
            loadings[(v, c)] = *value;
        }
        for i in 0..n {
            scores[(i, c)] = sign * norm * vectors[(i, k)];
        }
    }
    Ok(PrincipalComponents {
        variances: kept.iter().map(|&k| values[k]).collect(),
        contributions: kept.iter().map(|&k| values[k] / total).collect(),
        vectors: loadings,
        scores,
        means,
        scales,
    })
}

/// Variables of a series of structures with their names.
#[derive(Debug, Clone)]
pub struct Variables {
    /// Row `i`: structure `i`; column `v`: variable `v`.
    pub values: Mat<f64>,
    /// Names of the variables.
    pub names: Vec<String>,
}

fn same_atoms(structures: &[MolecularSystem]) -> MoptResult<&MolecularSystem> {
    let first = structures
        .first()
        .ok_or_else(|| MoptError::config("no structures"))?;
    if structures.iter().any(|s| s.elements != first.elements) {
        return Err(MoptError::config(
            "the structures of a series must have the same atoms",
        ));
    }
    Ok(first)
}

/// Mass-weighted Cartesian coordinates (Angstrom amu^(1/2)) of the structures, each
/// superposed on the first.
pub fn cartesian_variables(structures: &[MolecularSystem]) -> MoptResult<Variables> {
    let first = same_atoms(structures)?;
    let n_atoms = first.n_atoms();
    let aligned: Vec<MolecularSystem> = structures
        .par_iter()
        .map(|s| kabsch_align_bohr(s, first, true).map(|(a, _)| a))
        .collect::<MoptResult<_>>()?;
    let roots: Vec<f64> = first
        .elements
        .iter()
        .map(|e| libm::sqrt(e.mass_amu()) * Units::BOHR_TO_ANGSTROM)
        .collect();
    let values = Mat::from_fn(structures.len(), 3 * n_atoms, |i, v| {
        aligned[i].position(AtomId::new(v / 3)).value()[v % 3] * roots[v / 3]
    });
    let names = (0..3 * n_atoms)
        .map(|v| {
            format!(
                "{} {} - {}",
                v / 3 + 1,
                first.elements[v / 3].symbol(),
                ["x", "y", "z"][v % 3]
            )
        })
        .collect();
    Ok(Variables { values, names })
}

fn separation(sys: &MolecularSystem, i: usize, j: usize) -> [f64; 3] {
    let p = sys.position(AtomId::new(i)).value();
    let q = sys.position(AtomId::new(j)).value();
    match sys.active_cell() {
        Some(cell) => cell.mic_delta(p, q),
        None => [p[0] - q[0], p[1] - q[1], p[2] - q[2]],
    }
}

fn norm(v: [f64; 3]) -> f64 {
    libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// All interatomic distances (Angstrom; nearest images in a cell) of the structures.
pub fn distance_variables(structures: &[MolecularSystem]) -> MoptResult<Variables> {
    let first = same_atoms(structures)?;
    let n_atoms = first.n_atoms();
    let pairs: Vec<(usize, usize)> = (0..n_atoms)
        .flat_map(|i| (i + 1..n_atoms).map(move |j| (i, j)))
        .collect();
    if pairs.is_empty() {
        return Err(MoptError::config("one atom has no distances"));
    }
    let values = Mat::from_fn(structures.len(), pairs.len(), |s, v| {
        let (i, j) = pairs[v];
        norm(separation(&structures[s], i, j)) * Units::BOHR_TO_ANGSTROM
    });
    let names = pairs
        .iter()
        .map(|&(i, j)| {
            format!(
                "{} {} - {} {}",
                i + 1,
                first.elements[i].symbol(),
                j + 1,
                first.elements[j].symbol()
            )
        })
        .collect();
    Ok(Variables { values, names })
}

/// A chain of internal coordinates in the order of the atoms (a Z-matrix): for atom `i`
/// its distance to atom `i - 1` (Angstrom), the angle `i, i - 1, i - 2`, and the dihedral
/// angle `i, i - 1, i - 2, i - 3` (degrees). Dihedral angles are continued from structure
/// to structure, so that a rotation through 180 degrees is not a jump.
pub fn z_matrix_variables(structures: &[MolecularSystem]) -> MoptResult<Variables> {
    let first = same_atoms(structures)?;
    let n_atoms = first.n_atoms();
    if n_atoms < 2 {
        return Err(MoptError::config("one atom has no internal coordinates"));
    }
    let label = |i: usize| format!("{} {}", i + 1, first.elements[i].symbol());
    let mut names = Vec::new();
    let mut dihedral = Vec::new();
    for i in 1..n_atoms {
        names.push(format!("{} - {}", label(i), label(i - 1)));
        dihedral.push(false);
        if i >= 2 {
            names.push(format!(
                "{} - {} - {}",
                label(i),
                label(i - 1),
                label(i - 2)
            ));
            dihedral.push(false);
        }
        if i >= 3 {
            names.push(format!(
                "{} - {} - {} - {}",
                label(i),
                label(i - 1),
                label(i - 2),
                label(i - 3)
            ));
            dihedral.push(true);
        }
    }
    let row = |sys: &MolecularSystem| -> Vec<f64> {
        let mut out = Vec::with_capacity(names.len());
        for i in 1..n_atoms {
            // Bond vectors along the chain, from the atom towards the start.
            let b1 = separation(sys, i - 1, i);
            out.push(norm(b1) * Units::BOHR_TO_ANGSTROM);
            if i >= 2 {
                let b2 = separation(sys, i - 2, i - 1);
                let cosine = -dot(b1, b2) / (norm(b1) * norm(b2)).max(f64::MIN_POSITIVE);
                out.push(libm::acos(cosine.clamp(-1.0, 1.0)).to_degrees());
                if i >= 3 {
                    let b3 = separation(sys, i - 3, i - 2);
                    let (n1, n2) = (cross(b1, b2), cross(b2, b3));
                    let y = dot(cross(n1, n2), b2) / norm(b2).max(f64::MIN_POSITIVE);
                    // Zero for three atoms in a line, where the angle is not defined.
                    out.push(libm::atan2(y, dot(n1, n2)).to_degrees());
                }
            }
        }
        out
    };
    let mut rows: Vec<Vec<f64>> = structures.iter().map(row).collect();
    for s in 1..rows.len() {
        for v in 0..names.len() {
            if dihedral[v] {
                let previous = rows[s - 1][v];
                let turns = ((rows[s][v] - previous) / 360.0).round();
                rows[s][v] -= 360.0 * turns;
            }
        }
    }
    let values = Mat::from_fn(structures.len(), names.len(), |s, v| rows[s][v]);
    Ok(Variables { values, names })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Element, SpinCharge};
    use faer::Col;

    fn structure(symbols: &[&str], bohr: &[[f64; 3]]) -> MolecularSystem {
        let elements = symbols
            .iter()
            .map(|s| Element::parse_token(s).unwrap())
            .collect();
        let flat = Col::from_fn(3 * bohr.len(), |k| bohr[k / 3][k % 3]);
        MolecularSystem::from_flat_bohr(elements, &flat, SpinCharge::default(), None).unwrap()
    }

    #[test]
    fn scaling_recovers_points_of_a_plane_from_their_distances() {
        // Five points of a plane; the map must have their distances and nothing beyond two
        // dimensions.
        let points: [[f64; 2]; 5] = [[0.0, 0.0], [1.0, 0.2], [2.5, -0.4], [3.0, 1.5], [0.5, 2.0]];
        let d = Mat::from_fn(5, 5, |i, j| {
            (points[i][0] - points[j][0]).hypot(points[i][1] - points[j][1])
        });
        let map = classical_mds(&d, 3).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                let found = (0..3)
                    .map(|k| (map.coordinates[(i, k)] - map.coordinates[(j, k)]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!((found - d[(i, j)]).abs() < 1e-10, "{i} {j}");
            }
        }
        assert!(map.eigenvalues[2].abs() < 1e-10);
        assert!((map.contributions[0] + map.contributions[1] - 1.0).abs() < 1e-10);
        assert!((map.reproducibility - 1.0).abs() < 1e-10);
        // The first point is on the negative side of both coordinates.
        assert!(map.coordinates[(0, 0)] <= 0.0 && map.coordinates[(0, 1)] <= 0.0);
        // Distances that no points have (the triangle inequality is broken) lower the
        // reproducibility.
        let odd = Mat::from_fn(3, 3, |i, j| {
            if i == j {
                0.0
            } else {
                [1.0, 1.0, 5.0][i + j - 1]
            }
        });
        assert!(classical_mds(&odd, 2).unwrap().reproducibility < 1.0);
        assert!(classical_mds(&Mat::<f64>::zeros(3, 3), 2).is_err());
    }

    #[test]
    fn the_distance_ignores_where_and_how_a_structure_lies() {
        let water = [[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.0, -1.4, -0.9]];
        let a = structure(&["O", "H", "H"], &water);
        // Turned by 90 degrees about z and moved.
        let moved: Vec<[f64; 3]> = water
            .iter()
            .map(|p| [-p[1] + 3.0, p[0] - 1.0, p[2] + 0.5])
            .collect();
        let b = structure(&["O", "H", "H"], &moved);
        assert!(mass_weighted_distance(&a, &b).unwrap() < 1e-8);
        // One hydrogen pulled out along its bond: the oxygen hardly moves, so the distance
        // is close to sqrt(m_H) times the displacement.
        let mut stretched = water;
        stretched[1][1] += 0.3;
        stretched[1][2] -= 0.3 * 1.1 / 1.4;
        let c = structure(&["O", "H", "H"], &stretched);
        let d = mass_weighted_distance(&a, &c).unwrap();
        let moved_by = 0.3_f64.hypot(0.3 * 1.1 / 1.4) * Units::BOHR_TO_ANGSTROM;
        let light = 1.008_f64.sqrt() * moved_by;
        assert!(d < light && d > 0.9 * light, "{d} against {light}");
        let other = structure(&["O", "H", "F"], &water);
        assert!(mass_weighted_distance(&a, &other).is_err());
        let all = distance_matrix(&[a, b, c]).unwrap();
        assert!((all[(0, 2)] - d).abs() < 1e-12 && (all[(2, 0)] - d).abs() < 1e-12);
        assert!(all[(0, 1)] < 1e-8 && all[(1, 1)] == 0.0);
    }

    #[test]
    fn principal_components_of_points_along_a_line() {
        // Points on a line in three variables, with a little scatter across it.
        let direction = [0.6, 0.0, 0.8];
        let across = [0.8, 0.0, -0.6];
        let data = Mat::from_fn(7, 3, |i, v| {
            let t = i as f64 - 3.0;
            let wobble = 0.01 * [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0][i];
            5.0 + t * direction[v] + wobble * across[v]
        });
        let pca = principal_components(&data, false, 3).unwrap();
        assert_eq!(pca.variances.len(), 2);
        for (v, expected) in direction.iter().enumerate() {
            assert!((pca.vectors[(v, 0)] - expected).abs() < 1e-3, "{v}");
        }
        assert!(pca.contributions[0] > 0.999);
        assert!((pca.contributions.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // The scores are the centered data along the components, and their variances are
        // the eigenvalues.
        for i in 0..7 {
            let expected: f64 = (0..3)
                .map(|v| (data[(i, v)] - pca.means[v]) * pca.vectors[(v, 0)])
                .sum();
            assert!((pca.scores[(i, 0)] - expected).abs() < 1e-10);
        }
        let variance = (0..7).map(|i| pca.scores[(i, 0)].powi(2)).sum::<f64>() / 6.0;
        assert!((variance - pca.variances[0]).abs() < 1e-10);
        // Standardized: a variable that does not vary stays out, the other two count alike.
        let scaled = principal_components(&data, true, 3).unwrap();
        assert!((scaled.vectors[(0, 0)].abs() - scaled.vectors[(2, 0)].abs()).abs() < 1e-2);
        assert!(scaled.vectors[(1, 0)].abs() < 1e-12);
        assert!(principal_components(&Mat::<f64>::zeros(4, 3), false, 2).is_err());
    }

    #[test]
    fn variables_of_a_series() {
        // Hydrogen peroxide turned about its O-O bond in steps of 60 degrees.
        let frame = |degrees: f64| {
            let phi = degrees.to_radians();
            structure(
                &["H", "O", "O", "H"],
                &[
                    [1.7, 0.0, -0.4],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.7],
                    [1.7 * phi.cos(), 1.7 * phi.sin(), 3.1],
                ],
            )
        };
        let series: Vec<MolecularSystem> = (0..7).map(|k| frame(60.0 * k as f64)).collect();
        let z = z_matrix_variables(&series).unwrap();
        assert_eq!(z.names.len(), 6);
        assert_eq!(z.names[5], "4 H - 3 O - 2 O - 1 H");
        // Distances and angles stay; the dihedral angle goes on through 180 degrees.
        for s in 1..7 {
            for v in 0..5 {
                assert!((z.values[(s, v)] - z.values[(0, v)]).abs() < 1e-9);
            }
            let step = z.values[(s, 5)] - z.values[(s - 1, 5)];
            assert!((step.abs() - 60.0).abs() < 1e-9, "{step}");
        }
        assert!((z.values[(6, 5)] - z.values[(0, 5)]).abs() > 359.0);
        assert!((z.values[(0, 0)] - (1.7_f64.hypot(0.4)) * Units::BOHR_TO_ANGSTROM).abs() < 1e-12);

        let d = distance_variables(&series).unwrap();
        assert_eq!(d.names.len(), 6);
        assert_eq!(d.names[0], "1 H - 2 O");
        // Only the H-H distance changes with the turn.
        let pca = principal_components(&d.values, false, 2).unwrap();
        assert!(pca.vectors[(2, 0)].abs() > 0.999, "{:?}", pca.vectors);

        let c = cartesian_variables(&series).unwrap();
        assert_eq!(c.names.len(), 12);
        assert_eq!(c.names[4], "2 O - y");
        // Superposed with masses, the oxygens hardly move (their coordinates carry the
        // root of their mass, 4, the hydrogens' about 1).
        let moved = |v: usize| (c.values[(3, v)] - c.values[(0, v)]).abs();
        let oxygen = (moved(3) + moved(4) + moved(5)) / 4.0;
        assert!(oxygen < 0.2 * (moved(9) + moved(10) + moved(11)));
        assert!(cartesian_variables(&[]).is_err());
    }
}
