//! Dynamic modes of the last iterations of an optimization (extended dynamic mode
//! decomposition, an approximation of the Koopman operator of the map from one iteration
//! to the next).
//!
//! Each iteration is described by a vector of observables `psi` (the coordinates, their
//! powers, the interatomic distances, and the gradient). With the observables of `m + 1`
//! consecutive iterations as the columns of `X` (the first `m`) and `X'` (the last `m`),
//! the matrix `A` with `X' = A X` in the least-squares sense is examined through
//! `A~ = U^T X' V S^-1`, `X = U S V^T` being the singular value decomposition of `X` cut at
//! a rank `r`. The eigenvalues `lambda` of `A~` tell how the iterations move: `|lambda| <
//! 1` decays (the optimization converges along that mode), `|lambda| = 1` persists,
//! `|lambda| > 1` grows; a negative real part alternates in sign from iteration to
//! iteration, an imaginary part rotates. The modes are `X' V S^-1 W` with the eigenvectors
//! `W` of `A~`.
//!
//! The singular value decomposition is taken from the `m x m` matrix `X^T X`, since there
//! are few iterations and many observables.
//!
//! MultiOptPy: `PESAnalyzer/koopman_analysis.py:KoopmanAnalyzer`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// Iterations that enter an analysis besides the latest (MultiOptPy's window).
pub const DEFAULT_WINDOW: usize = 10;
/// Singular values of the observables below this are cut off (MultiOptPy's threshold).
pub const DEFAULT_RANK_THRESHOLD: f64 = 1e-5;
/// Highest power of the coordinates among the observables (MultiOptPy's degree).
pub const DEFAULT_DEGREE: usize = 2;

/// The observables of one iteration: the coordinates `x`, their powers up to `degree`, the
/// distances between all pairs of atoms (the coordinates are taken in threes), and the
/// gradient when there is one.
pub fn observables(positions: &[f64], gradient: Option<&[f64]>, degree: usize) -> Vec<f64> {
    let mut psi = Vec::new();
    for power in 1..=degree.max(1) {
        psi.extend(positions.iter().map(|x| x.powi(power as i32)));
    }
    let atoms = positions.len() / 3;
    for i in 0..atoms {
        for j in i + 1..atoms {
            let d2: f64 = (0..3)
                .map(|k| (positions[3 * i + k] - positions[3 * j + k]).powi(2))
                .sum();
            psi.push(libm::sqrt(d2));
        }
    }
    if let Some(g) = gradient {
        psi.extend_from_slice(g);
    }
    psi
}

/// Eigenvalues and modes of the map from one iteration to the next.
#[derive(Debug, Clone)]
pub struct DynamicModes {
    /// Rank at which the observables were cut.
    pub rank: usize,
    /// Eigenvalues as `[real part, imaginary part]`, largest modulus first.
    pub eigenvalues: Vec<[f64; 2]>,
    /// The modes in the space of the observables: `modes[k][v]` is observable `v` of mode
    /// `k` as `[real part, imaginary part]`.
    pub modes: Vec<Vec<[f64; 2]>>,
}

impl DynamicModes {
    /// Modulus of eigenvalue `k`.
    pub fn modulus(&self, k: usize) -> f64 {
        self.eigenvalues[k][0].hypot(self.eigenvalues[k][1])
    }
}

/// The dynamic modes of `snapshots`, the observables of consecutive iterations (at least
/// two, all of the same length). `None` when every singular value of the observables is
/// below `rank_threshold`.
pub fn dynamic_modes(
    snapshots: &[Vec<f64>],
    rank_threshold: f64,
) -> MoptResult<Option<DynamicModes>> {
    if snapshots.len() < 2 {
        return Err(MoptError::config(
            "dynamic modes need the observables of at least two iterations",
        ));
    }
    let p = snapshots[0].len();
    if p == 0 || snapshots.iter().any(|s| s.len() != p) {
        return Err(MoptError::config(
            "the observables of all iterations must have the same, nonzero length",
        ));
    }
    if snapshots.iter().flatten().any(|v| !v.is_finite()) {
        return Err(MoptError::config("an observable is not finite"));
    }
    let m = snapshots.len() - 1;
    let x = Mat::from_fn(p, m, |v, k| snapshots[k][v]);
    let x_next = Mat::from_fn(p, m, |v, k| snapshots[k + 1][v]);
    // X = U S V^T from the eigenvectors of X^T X.
    let gram = x.transpose() * &x;
    let (values, vectors) = symmetric_eigen(&gram)?;
    let kept: Vec<usize> = (0..m)
        .rev()
        .filter(|&k| libm::sqrt(values[k].max(0.0)) > rank_threshold)
        .collect();
    let r = kept.len();
    if r == 0 {
        return Ok(None);
    }
    // V S^-1, of which U = X V S^-1.
    let vs = Mat::from_fn(m, r, |i, c| {
        vectors[(i, kept[c])] / libm::sqrt(values[kept[c]])
    });
    let u = &x * &vs;
    let projected = &x_next * &vs;
    let reduced = u.transpose() * &projected;
    let evd = reduced
        .eigen()
        .map_err(|e| MoptError::linalg(format!("eigendecomposition failed: {e:?}")))?;
    let (s, w) = (evd.S(), evd.U());
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&a, &b| {
        let (ma, mb) = (s[a].re.hypot(s[a].im), s[b].re.hypot(s[b].im));
        mb.total_cmp(&ma)
    });
    let eigenvalues = order.iter().map(|&k| [s[k].re, s[k].im]).collect();
    let modes = order
        .iter()
        .map(|&k| {
            (0..p)
                .map(|v| {
                    let (mut re, mut im) = (0.0, 0.0);
                    for c in 0..r {
                        re += projected[(v, c)] * w[(c, k)].re;
                        im += projected[(v, c)] * w[(c, k)].im;
                    }
                    [re, im]
                })
                .collect()
        })
        .collect();
    Ok(Some(DynamicModes {
        rank: r,
        eigenvalues,
        modes,
    }))
}

/// What an eigenvalue says about the iterations, in MultiOptPy's words.
pub fn behavior(eigenvalue: [f64; 2]) -> &'static str {
    let modulus = eigenvalue[0].hypot(eigenvalue[1]);
    let size = if (modulus - 1.0).abs() < 1e-3 {
        "constant amplitude"
    } else if modulus < 1.0 {
        "damping"
    } else {
        "amplification"
    };
    let turning = eigenvalue[1].abs() > 1e-8 * modulus.max(1.0);
    match (size, eigenvalue[0] < 0.0, turning) {
        ("damping", false, false) => "damping, monotonic: converges straight",
        ("damping", true, false) => "damping, alternating sign: converges while oscillating",
        ("damping", false, true) => "damping, spiral: converges while rotating",
        ("damping", true, true) => "damping, reversal and spiral: converges",
        ("constant amplitude", false, false) => "constant amplitude: goes on in one direction",
        ("constant amplitude", true, false) => "constant amplitude, reversal: oscillates",
        ("constant amplitude", _, true) => "constant amplitude, rotation: persists",
        (_, false, false) => "amplification, monotonic: diverges",
        (_, true, false) => "amplification, reversal: diverges while oscillating",
        (_, _, true) => "amplification, spiral: diverges while rotating",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observables_have_coordinates_powers_distances_and_gradient() {
        let x = [0.0, 0.0, 0.0, 3.0, 4.0, 0.0];
        let g = [0.1, 0.2, 0.3, -0.1, -0.2, -0.3];
        let psi = observables(&x, Some(&g), 2);
        assert_eq!(psi.len(), 6 + 6 + 1 + 6);
        assert_eq!(&psi[..6], &x);
        assert!((psi[9] - 9.0).abs() < 1e-15 && (psi[10] - 16.0).abs() < 1e-15);
        assert!((psi[12] - 5.0).abs() < 1e-15);
        assert_eq!(&psi[13..], &g);
        assert_eq!(observables(&x, None, 1).len(), 7);
    }

    #[test]
    fn a_linear_map_gives_back_its_eigenvalues() {
        // x(k + 1) = A x(k) with a decaying rotation and a decaying, alternating direction.
        let (rho, phi) = (0.9_f64, 0.5_f64);
        let step = |x: [f64; 3]| {
            [
                rho * (phi.cos() * x[0] - phi.sin() * x[1]),
                rho * (phi.sin() * x[0] + phi.cos() * x[1]),
                -0.5 * x[2],
            ]
        };
        let mut x = [1.0, 0.3, 0.8];
        let mut snapshots = Vec::new();
        for _ in 0..9 {
            snapshots.push(x.to_vec());
            x = step(x);
        }
        // Three directions carry the motion; the cut removes the rounding noise beyond them.
        let modes = dynamic_modes(&snapshots, 1e-6).unwrap().unwrap();
        assert_eq!(modes.rank, 3);
        // The pair of the rotation first (modulus 0.9), then the real one (-0.5).
        assert!((modes.modulus(0) - rho).abs() < 1e-8 && (modes.modulus(1) - rho).abs() < 1e-8);
        assert!((modes.eigenvalues[0][0] - rho * phi.cos()).abs() < 1e-8);
        assert!((modes.eigenvalues[0][1].abs() - rho * phi.sin()).abs() < 1e-8);
        assert!((modes.eigenvalues[2][0] + 0.5).abs() < 1e-8);
        assert!(modes.eigenvalues[2][1].abs() < 1e-8);
        // The real mode is the third direction.
        let mode = &modes.modes[2];
        let size = mode.iter().map(|c| c[0].hypot(c[1])).fold(0.0, f64::max);
        assert!(mode[0][0].hypot(mode[0][1]) < 1e-6 * size);
        assert!(mode[2][0].hypot(mode[2][1]) > 0.999 * size);
        assert_eq!(
            behavior(modes.eigenvalues[2]),
            "damping, alternating sign: converges while oscillating"
        );
        assert_eq!(
            behavior(modes.eigenvalues[0]),
            "damping, spiral: converges while rotating"
        );
        assert_eq!(behavior([1.2, 0.0]), "amplification, monotonic: diverges");
    }

    #[test]
    fn a_series_that_does_not_move_has_no_modes_and_wrong_input_is_refused() {
        let still = vec![vec![0.0, 0.0]; 5];
        assert!(dynamic_modes(&still, 1e-5).unwrap().is_none());
        assert!(dynamic_modes(&[vec![1.0]], 1e-5).is_err());
        assert!(dynamic_modes(&[vec![1.0], vec![1.0, 2.0]], 1e-5).is_err());
        assert!(dynamic_modes(&[vec![f64::NAN], vec![1.0]], 1e-5).is_err());
    }
}
