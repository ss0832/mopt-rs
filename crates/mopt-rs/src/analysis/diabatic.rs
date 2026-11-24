//! Adiabatic states along a path as diabatic states and their couplings.
//!
//! Where electronic states come close along a path (avoided crossings), the states counted
//! by their energies exchange their characters, and their energies have kinks or narrow
//! gaps. Diabatic states keep their characters: their energies (the diagonal of the
//! diabatic potential matrix `W`) cross, and what the calculation shows as a gap is their
//! coupling (the elements of `W` off the diagonal). The coupling and the slopes of the
//! diabatic energies at a crossing are what the Landau-Zener formula takes
//! ([`landau_zener_stay`]).
//!
//! [`from_overlaps`] is the propagative block-diagonalisation diabatisation (S. P.
//! Neville, I. Seidu, M. S. Schuurman, arXiv:1912.09553, eqs. 13 and 14; the
//! block-diagonalisation of T. Pacher, L. S. Cederbaum, H. Köppel) for any number of
//! states. It needs the overlaps of the adiabatic states of neighbouring points only. With
//! `A(n)` the adiabatic states of point `n` in the diabatic ones (the transpose of the
//! `U` of the paper) and `O(n)` the overlaps of the adiabatic states of point `n` with
//! those of point `n + 1`,
//!
//! ```text
//! S = A(n) O(n)        (the diabatic states of point n with the adiabatic ones of n + 1)
//! A(n + 1) = S (S^T S)^(-1/2)                  (the orthogonal matrix nearest to S)
//! W(n) = A(n) diag(E(n)) A(n)^T,               A(0) = 1
//! ```
//!
//! so the diabatic states are the adiabatic states of the first point, carried along with
//! the least change. Where the states that are followed are all the states that mix (the
//! overlap matrices are orthogonal), the result does not depend on the distance of the
//! points: it is the Hamiltonian in the states of the first point. Otherwise it is the
//! better the closer the points are, and the smallest singular value of `S` says how much
//! of the states a step keeps (`kept`). The signs of the states are free.
//!
//! [`from_gap`] needs no overlaps: the gap of two states near its minimum is fitted by the
//! gap of two straight diabatic energies with a constant coupling, `gap^2 = (dF (s -
//! s0))^2 + (2 H12)^2`; the coupling is half the smallest gap.
//!
//! Limits: the diabatic states depend on where the path starts (the first point should be
//! away from the crossings); states outside the followed ones that mix in are seen as
//! overlap that is lost; a crossing is looked at as one of two states (the Landau-Zener
//! formula is for two).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// The mass of one atomic mass unit in electron masses: a speed along a mass-weighted path
/// (amu^(1/2) Bohr per atomic unit of time) has the kinetic energy `AMU * v^2 / 2` Hartree.
const AMU_IN_ELECTRON_MASSES: f64 = Units::AMU_TO_AU_MASS;

/// A step that keeps less than this of a state has lost it: the nearest orthogonal matrix
/// is not defined.
const LOST: f64 = 1e-6;

/// Diabatic states along a path.
#[derive(Debug, Clone, PartialEq)]
pub struct DiabaticStates {
    /// The diabatic potential matrix at every point: the energies of the diabatic states
    /// on its diagonal (state `i` is the adiabatic state `i` of the first point), their
    /// couplings off it.
    pub potentials: Vec<Mat<f64>>,
    /// At every point, the share of the adiabatic state `j` in the diabatic state `i`
    /// (`[(i, j)]`; the unit matrix at the first point).
    pub weights: Vec<Mat<f64>>,
    /// For every step between two points, the smallest singular value of the overlaps of
    /// the diabatic states with the adiabatic states of the next point: one when the
    /// states of the next point are the states of this one, less when other states mix in
    /// or the points are far apart.
    pub kept: Vec<f64>,
}

impl DiabaticStates {
    /// The number of states.
    pub fn states(&self) -> usize {
        self.potentials.first().map_or(0, Mat::nrows)
    }

    /// The element `(i, j)` of the diabatic potential matrix along the path.
    pub fn element(&self, i: usize, j: usize) -> Vec<f64> {
        self.potentials.iter().map(|w| w[(i, j)]).collect()
    }
}

/// A crossing of two diabatic states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiabaticCrossing {
    /// The two states (their numbers in the list that was given, from 0).
    pub states: (usize, usize),
    /// Where on the path (the unit of the positions that were given).
    pub position: f64,
    /// The energy of the two diabatic states there.
    pub energy: f64,
    /// The size of their coupling there.
    pub coupling: f64,
    /// The size of the slope of the difference of their energies there (energy per unit
    /// of the positions).
    pub slope_difference: f64,
    /// The slopes of the energies of the two states there (energy per unit of the
    /// positions): of the same sign at a sloped crossing, of opposite signs at a peaked one.
    pub slopes: [f64; 2],
}

/// The orthogonal matrix nearest to `s`, `s (s^T s)^(-1/2)`, and the smallest singular
/// value of `s`.
fn nearest_orthogonal(s: &Mat<f64>) -> MoptResult<(Mat<f64>, f64)> {
    let (values, vectors) = symmetric_eigen(&(s.transpose() * s))?;
    let smallest = values
        .iter()
        .fold(f64::INFINITY, |m, v| m.min(*v))
        .max(0.0)
        .sqrt();
    if !smallest.is_finite() || smallest < LOST {
        return Err(MoptError::calc(format!(
            "diabatic states: a step keeps {smallest:.1e} of one of the states (the states of \
             the next point are other states)"
        )));
    }
    let n = s.nrows();
    let inverse_root = Mat::from_fn(n, n, |i, j| {
        (0..n)
            .map(|k| vectors[(i, k)] * vectors[(j, k)] / values[k].sqrt())
            .sum::<f64>()
    });
    Ok((s * inverse_root, smallest))
}

/// Diabatic states from the energies of the adiabatic states along a path (`energies[k]`:
/// the states of point `k` in ascending order, the same number at every point) and the
/// overlaps of the states of neighbouring points: `overlaps[k][(i, j)]` is the overlap of
/// state `i` at point `k` with state `j` at point `k + 1`. The signs of the states are
/// free (a state times -1 is the same state), as long as a point has the same states in
/// the overlaps with the point before it and with the point after it.
pub fn from_overlaps(energies: &[Vec<f64>], overlaps: &[Mat<f64>]) -> MoptResult<DiabaticStates> {
    let points = energies.len();
    let n = energies.first().map_or(0, Vec::len);
    if points < 2 || n < 2 || overlaps.len() + 1 != points {
        return Err(MoptError::config(format!(
            "diabatic states: {points} points of {n} states with {} overlap matrices (two \
             states and two points at least, and as many matrices as points less one)",
            overlaps.len()
        )));
    }
    if energies.iter().any(|e| e.len() != n)
        || overlaps.iter().any(|o| o.nrows() != n || o.ncols() != n)
    {
        return Err(MoptError::config(format!(
            "diabatic states: every point needs {n} energies and every step {n} x {n} overlaps"
        )));
    }
    let mut a = Mat::<f64>::identity(n, n);
    let mut out = DiabaticStates {
        potentials: Vec::with_capacity(points),
        weights: Vec::with_capacity(points),
        kept: Vec::with_capacity(points - 1),
    };
    for (k, energy) in energies.iter().enumerate() {
        if k > 0 {
            let (next, smallest) = nearest_orthogonal(&(&a * &overlaps[k - 1]))
                .map_err(|e| e.context(format!("the step from point {} to {k}", k - 1)))?;
            a = next;
            out.kept.push(smallest);
        }
        out.potentials.push(Mat::from_fn(n, n, |i, j| {
            (0..n).map(|m| a[(i, m)] * energy[m] * a[(j, m)]).sum()
        }));
        out.weights
            .push(Mat::from_fn(n, n, |i, j| a[(i, j)] * a[(i, j)]));
    }
    Ok(out)
}

/// The crossings of the diabatic states along the positions `arc` of the points: where
/// the energies of two of them change their order, by linear interpolation between the
/// two points. In the order of their positions.
pub fn crossings(arc: &[f64], states: &DiabaticStates) -> Vec<DiabaticCrossing> {
    let n = states.states();
    let points = arc.len().min(states.potentials.len());
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            for k in 0..points.saturating_sub(1) {
                let (w0, w1) = (&states.potentials[k], &states.potentials[k + 1]);
                let here = w0[(i, i)] - w0[(j, j)];
                let next = w1[(i, i)] - w1[(j, j)];
                let length = arc[k + 1] - arc[k];
                if (here == 0.0 && next == 0.0) || length <= 0.0 || here * next > 0.0 {
                    continue;
                }
                if next == 0.0 && k + 2 < points {
                    // Counted with the next step, where it is the first point.
                    continue;
                }
                let t = here / (here - next);
                let mix = |a: f64, b: f64| a + t * (b - a);
                out.push(DiabaticCrossing {
                    states: (i, j),
                    position: arc[k] + t * length,
                    energy: mix(w0[(i, i)], w1[(i, i)]),
                    coupling: mix(w0[(i, j)], w1[(i, j)]).abs(),
                    slope_difference: ((next - here) / length).abs(),
                    slopes: [
                        (w1[(i, i)] - w0[(i, i)]) / length,
                        (w1[(j, j)] - w0[(j, j)]) / length,
                    ],
                });
            }
        }
    }
    out.sort_by(|a, b| a.position.total_cmp(&b.position));
    out
}

/// The crossing of two straight diabatic states with a constant coupling whose adiabatic
/// gap is the gap `upper - lower` near its minimum (the parabola through `gap^2` at the
/// point of the smallest gap and its two neighbours). `None` when the smallest gap is at
/// an end of the path or the gap does not open on both sides. The states of the result
/// are `(0, 1)`.
pub fn from_gap(arc: &[f64], lower: &[f64], upper: &[f64]) -> Option<DiabaticCrossing> {
    let n = arc.len().min(lower.len()).min(upper.len());
    let gap = |k: usize| upper[k] - lower[k];
    let m = (0..n).min_by(|&a, &b| gap(a).total_cmp(&gap(b)))?;
    if m == 0 || m + 1 >= n {
        return None;
    }
    // The parabola y = c0 + c1 x + c2 x^2 through three points, x measured from arc[m].
    let (xa, xb) = (arc[m - 1] - arc[m], arc[m + 1] - arc[m]);
    let (ya, y0, yb) = (gap(m - 1).powi(2), gap(m).powi(2), gap(m + 1).powi(2));
    if xa >= 0.0 || xb <= 0.0 {
        return None;
    }
    let c2 = ((ya - y0) / xa - (yb - y0) / xb) / (xa - xb);
    let c1 = (ya - y0) / xa - c2 * xa;
    if c2 <= 0.0 {
        return None;
    }
    let x0 = -c1 / (2.0 * c2);
    let smallest = (y0 - c1 * c1 / (4.0 * c2)).max(0.0).sqrt();
    let mean = |v: &[f64]| {
        // The energy interpolated between the points next to x0.
        let (k, t) = if x0 >= 0.0 {
            (m, (x0 / xb).min(1.0))
        } else {
            (m - 1, (1.0 - x0 / xa).max(0.0))
        };
        v[k] + t * (v[k + 1] - v[k])
    };
    // The mean of the two adiabatic energies is that of the two straight states: its slope
    // between the neighbours of the smallest gap is the mean of their slopes.
    let mean_slope = (0.5 * (lower[m + 1] + upper[m + 1]) - 0.5 * (lower[m - 1] + upper[m - 1]))
        / (arc[m + 1] - arc[m - 1]);
    let half = 0.5 * c2.sqrt();
    Some(DiabaticCrossing {
        states: (0, 1),
        position: arc[m] + x0.clamp(xa, xb),
        energy: 0.5 * (mean(lower) + mean(upper)),
        coupling: 0.5 * smallest,
        slope_difference: c2.sqrt(),
        slopes: [mean_slope + half, mean_slope - half],
    })
}

/// The Landau-Zener probability that a passage through a crossing stays on its diabatic
/// state (that is, changes the adiabatic state): `exp(-2 pi H12^2 / (v |dF|))` in atomic
/// units (`hbar = 1`), for the coupling `coupling` (Hartree), the slope difference of the
/// diabatic energies along a mass-weighted path (Hartree per amu^(1/2) Bohr), and the
/// kinetic energy `kinetic` (Hartree) of the motion along the path at the crossing.
pub fn landau_zener_stay(coupling: f64, slope_difference: f64, kinetic: f64) -> f64 {
    if coupling == 0.0 {
        return 1.0;
    }
    let speed = (2.0 * kinetic.max(0.0) / AMU_IN_ELECTRON_MASSES).sqrt();
    let rate = speed * slope_difference.abs();
    if rate <= 0.0 {
        return 0.0;
    }
    (-2.0 * PI * coupling * coupling / rate).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The adiabatic states of the Hamiltonians `hamiltonian(s)` at the points `arc`:
    /// the lowest `kept` energies of every point, and the overlaps of the lowest `kept`
    /// states of neighbouring points, the states given the signs `sign(point, state)`.
    fn adiabatic(
        arc: &[f64],
        kept: usize,
        hamiltonian: impl Fn(f64) -> Mat<f64>,
        sign: impl Fn(usize, usize) -> f64,
    ) -> (Vec<Vec<f64>>, Vec<Mat<f64>>, Mat<f64>) {
        let solved: Vec<(faer::Col<f64>, Mat<f64>)> = arc
            .iter()
            .map(|s| symmetric_eigen(&hamiltonian(*s)).unwrap())
            .collect();
        let size = solved[0].1.nrows();
        let energies = solved
            .iter()
            .map(|(e, _)| (0..kept).map(|k| e[k]).collect())
            .collect();
        let overlaps = (0..arc.len() - 1)
            .map(|p| {
                Mat::from_fn(kept, kept, |i, j| {
                    sign(p, i)
                        * sign(p + 1, j)
                        * (0..size)
                            .map(|m| solved[p].1[(m, i)] * solved[p + 1].1[(m, j)])
                            .sum::<f64>()
                })
            })
            .collect();
        (energies, overlaps, solved[0].1.clone())
    }

    fn two_states(a: f64, v: f64) -> impl Fn(f64) -> Mat<f64> {
        move |s| {
            Mat::from_fn(2, 2, |i, j| {
                if i == j {
                    a * s * (1.0 - 2.0 * i as f64)
                } else {
                    v
                }
            })
        }
    }

    fn signs(point: usize, state: usize) -> f64 {
        if (point * 7 + state * 3).is_multiple_of(4) {
            -1.0
        } else {
            1.0
        }
    }

    #[test]
    fn the_coupling_and_the_slopes_of_a_model_crossing() {
        let (a, v) = (0.02, 0.004);
        let arc: Vec<f64> = (0..=400)
            .map(|k| -20.0 + 0.1 * f64::from(k) + 0.013)
            .collect();
        // The signs of the states as a program would give them: arbitrary.
        let (energies, overlaps, _) = adiabatic(&arc, 2, two_states(a, v), signs);
        let states = from_overlaps(&energies, &overlaps).unwrap();
        // Two states and nothing else: all of the overlap is kept.
        assert!(states.kept.iter().all(|k| (k - 1.0).abs() < 1e-12));
        let found = crossings(&arc, &states);
        assert_eq!(found.len(), 1, "{found:?}");
        let crossing = found[0];
        assert_eq!(crossing.states, (0, 1));
        // The diabatic states are the adiabatic states of the first point, which are the
        // straight ones up to the mixing angle there (v / (2 a |s|) = 0.005).
        assert!(crossing.position.abs() < 0.05, "{crossing:?}");
        assert!((crossing.coupling - v).abs() < 1e-3 * v, "{crossing:?}");
        assert!((crossing.slope_difference - 2.0 * a).abs() < 1e-3 * 2.0 * a);
        // A peaked crossing: the slopes +a and -a.
        assert!((crossing.slopes[0] - a).abs() < 1e-3 * a, "{crossing:?}");
        assert!((crossing.slopes[1] + a).abs() < 1e-3 * a, "{crossing:?}");
        assert!(crossing.energy.abs() < 1e-4);
        // The states have changed places at the end, and the diabatic energies are the
        // straight lines.
        assert!(states.weights[0][(0, 0)] == 1.0 && states.weights[400][(0, 1)] > 0.999);
        let k = 150;
        assert!((states.potentials[k][(0, 0)].abs() - a * arc[k].abs()).abs() < 1e-4);
        // Other signs: the same result up to the sign of the coupling.
        let (_, plain, _) = adiabatic(&arc, 2, two_states(a, v), |_, _| 1.0);
        let again = from_overlaps(&energies, &plain).unwrap();
        assert!(states
            .element(0, 1)
            .iter()
            .zip(again.element(0, 1))
            .all(|(x, y)| (x.abs() - y.abs()).abs() < 1e-12));

        // From the gap alone: the same crossing, exactly for this model.
        let lower: Vec<f64> = energies.iter().map(|e| e[0]).collect();
        let upper: Vec<f64> = energies.iter().map(|e| e[1]).collect();
        let fitted = from_gap(&arc, &lower, &upper).unwrap();
        assert!(fitted.position.abs() < 1e-9, "{fitted:?}");
        assert!((fitted.coupling - v).abs() < 1e-10);
        assert!((fitted.slope_difference - 2.0 * a).abs() < 1e-9);
        assert!((fitted.slopes[0] - a).abs() < 1e-9 && (fitted.slopes[1] + a).abs() < 1e-9);
    }

    /// Three states with slopes, offsets, and couplings of their own, and a fourth one
    /// far above that couples to them by `far`.
    fn three_and_one(far: f64) -> impl Fn(f64) -> Mat<f64> {
        move |s| {
            let diagonal = [
                0.02 * s,
                0.003 - 0.004 * s,
                -0.015 * s + 0.001,
                0.4 + 0.01 * s,
            ];
            let couplings = [
                [0.0, 0.002, 0.0015, far],
                [0.0, 0.0, 0.001, far],
                [0.0, 0.0, 0.0, far],
            ];
            Mat::from_fn(4, 4, |i, j| {
                if i == j {
                    diagonal[i]
                } else {
                    couplings[i.min(j)][i.max(j)]
                }
            })
        }
    }

    #[test]
    fn three_states_are_the_hamiltonian_in_the_states_of_the_first_point() {
        // The three states are all there is (the fourth does not couple): whatever the
        // distance of the points and the signs, the diabatic matrix is the Hamiltonian in
        // the adiabatic states of the first point.
        for step in [0.05, 0.5, 2.0] {
            let arc: Vec<f64> = (0..=(8.0 / step) as usize)
                .map(|k| -4.0 + step * k as f64)
                .collect();
            let (energies, overlaps, first) = adiabatic(&arc, 3, three_and_one(0.0), signs);
            let states = from_overlaps(&energies, &overlaps).unwrap();
            assert_eq!(states.states(), 3);
            assert!(states.kept.iter().all(|k| (k - 1.0).abs() < 1e-10));
            for (p, s) in arc.iter().enumerate() {
                let h = three_and_one(0.0)(*s);
                for i in 0..3 {
                    for j in 0..3 {
                        let exact: f64 = (0..4)
                            .flat_map(|m| (0..4).map(move |l| (m, l)))
                            .map(|(m, l)| first[(m, i)] * h[(m, l)] * first[(l, j)])
                            .sum();
                        let found = states.potentials[p][(i, j)];
                        // The sign of a coupling is that of the two states at the start.
                        let found = found * signs(0, i) * signs(0, j);
                        assert!(
                            (found - exact).abs() < 1e-11,
                            "step {step}, point {p}, ({i}, {j}): {found} {exact}"
                        );
                    }
                }
            }
            // Every pair of the three crosses on this path.
            let pairs: Vec<(usize, usize)> =
                crossings(&arc, &states).iter().map(|c| c.states).collect();
            for pair in [(0, 1), (0, 2), (1, 2)] {
                assert!(pairs.contains(&pair), "step {step}: {pairs:?}");
            }
        }
    }

    #[test]
    fn states_outside_the_followed_ones_cost_accuracy_that_closer_points_give_back() {
        // A fourth state couples to the three: the three followed states are not all that
        // mixes, a step loses a little of them, and the result depends on the distance of
        // the points. Against the result of very close points the error falls with it.
        let potentials = |step: f64| {
            let arc: Vec<f64> = (0..=(8.0 / step).round() as usize)
                .map(|k| -4.0 + step * k as f64)
                .collect();
            let (energies, overlaps, _) = adiabatic(&arc, 3, three_and_one(0.03), |_, _| 1.0);
            let states = from_overlaps(&energies, &overlaps).unwrap();
            let lost = 1.0 - states.kept.iter().copied().fold(1.0, f64::min);
            (states.potentials[arc.len() - 1].clone(), lost)
        };
        let (reference, _) = potentials(0.005);
        let error = |step: f64| {
            let (found, lost) = potentials(step);
            let worst = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .map(|(i, j)| (found[(i, j)].abs() - reference[(i, j)].abs()).abs())
                .fold(0.0, f64::max);
            (worst, lost)
        };
        let (coarse, lost_coarse) = error(1.0);
        let (middle, _) = error(0.25);
        let (fine, lost_fine) = error(0.0625);
        assert!(coarse > middle && middle > fine, "{coarse} {middle} {fine}");
        assert!(fine < 0.1 * coarse, "{coarse} {fine}");
        // What a step loses is reported, and it is less for closer points.
        assert!(
            lost_coarse > lost_fine && lost_coarse > 1e-9,
            "{lost_coarse} {lost_fine}"
        );
    }

    #[test]
    fn states_that_cross_without_a_coupling() {
        // A true crossing: the states change places between two points, with an overlap
        // matrix that is a permutation.
        let arc = [0.0, 1.0, 2.0, 3.0];
        let energies: Vec<Vec<f64>> = [0.3, 0.1, 0.1, 0.3].iter().map(|e| vec![-e, *e]).collect();
        let matrix = |rows: [[f64; 2]; 2]| Mat::from_fn(2, 2, |i, j| rows[i][j]);
        let same = matrix([[1.0, 0.0], [0.0, 1.0]]);
        let swapped = matrix([[0.0, -1.0], [1.0, 0.0]]);
        let states = from_overlaps(&energies, &[same.clone(), swapped, same.clone()]).unwrap();
        assert!(states.element(0, 1).iter().all(|v| v.abs() < 1e-15));
        assert!((states.potentials[3][(0, 0)] - 0.3).abs() < 1e-15);
        assert!((states.potentials[3][(1, 1)] + 0.3).abs() < 1e-15);
        let found = crossings(&arc, &states);
        assert_eq!(found.len(), 1);
        assert!((found[0].position - 1.5).abs() < 1e-12 && found[0].coupling < 1e-15);
        assert!((found[0].slope_difference - 0.4).abs() < 1e-12);
        // Overlap that is lost is reported; a state that is lost altogether is an error.
        let weak = matrix([[0.6, 0.0], [0.0, 0.9]]);
        let lost = from_overlaps(&energies, &[same.clone(), weak, same.clone()]).unwrap();
        assert!((lost.kept[1] - 0.6).abs() < 1e-12);
        let gone = matrix([[1.0, 0.0], [0.0, 0.0]]);
        let error = from_overlaps(&energies, &[same.clone(), gone, same.clone()]).unwrap_err();
        assert!(error.to_string().contains("point 1 to 2"), "{error}");
        assert!(from_overlaps(&energies, &[same]).is_err());
        // No minimum of the gap inside the path: no fit.
        assert!(from_gap(&arc, &[0.0, 0.0, 0.0, 0.0], &[0.1, 0.2, 0.3, 0.4]).is_none());
    }

    #[test]
    fn the_probability_of_staying_on_a_diabatic_state() {
        // No coupling: the passage stays; a strong coupling or a slow passage: it follows
        // the adiabatic state.
        assert_eq!(landau_zener_stay(0.0, 0.01, 1e-3), 1.0);
        assert!(landau_zener_stay(0.05, 0.01, 1e-3) < 1e-10);
        assert_eq!(landau_zener_stay(0.001, 0.01, 0.0), 0.0);
        // exp(-2 pi H12^2 / (v dF)) with v = sqrt(2 T / m).
        let (h, f, t) = (1e-4, 2e-3, 5e-4);
        let v = (2.0 * t / AMU_IN_ELECTRON_MASSES).sqrt();
        let expected = (-2.0 * PI * h * h / (v * f)).exp();
        assert!((landau_zener_stay(h, f, t) - expected).abs() < 1e-15);
        assert!(expected > 0.9 && expected < 1.0);
        // Faster: more of it stays.
        assert!(landau_zener_stay(h, f, 4.0 * t) > landau_zener_stay(h, f, t));
    }
}
