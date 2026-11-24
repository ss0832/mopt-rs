//! Kabsch optimal proper `SO(3)` superposition (`det(R) = +1`) and RMSD.
//!
//! Rigid-body mode projectors live in `math::projection`.
//!
//! MultiOptPy: `Utils/calc_tools.py:Calculationtools.kabsch_algorithm`, `calc_center`,
//!  `calc_center_of_mass`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{AtomId, Position};
use faer::Mat;

use crate::domain::MolecularSystem;
use crate::error::MoptResult;

/// Compute the proper `SO(3)` rotation matrix `R` (`det(R) = +1`) and translation that minimize
/// `sum_i w_i || R (p_i - c_p) - (q_i - c_q) ||^2`, returning `(aligned_p, rmsd)`.
///
/// If `p` or `q` carries an active `PeriodicCell`, rotational alignment is forbidden by lattice
/// symmetry; in that case `p` is unwrapped onto `q` via MIC and only periodic translational drift
/// along periodic axes is removed.
pub fn kabsch_align_bohr(
    mobile: &MolecularSystem,
    reference: &MolecularSystem,
    mass_weighted: bool,
) -> MoptResult<(MolecularSystem, f64)> {
    let n = mobile.n_atoms();
    if n == 0 {
        return Ok((mobile.clone(), 0.0));
    }

    if mobile.has_pbc() || reference.has_pbc() {
        let mut unwrapped = mobile.unwrapped_to(reference)?;
        if let Some(cell) = mobile.active_cell().or(reference.active_cell()) {
            // Remove mean fractional drift along periodic axes only
            let mut mean_disp = [0.0; 3];
            let mut w_sum = 0.0;
            for i in 0..n {
                let w = if mass_weighted {
                    mobile.elements[i].mass_amu()
                } else {
                    1.0
                };
                let p = unwrapped.position(AtomId::new(i)).value();
                let q = reference.position(AtomId::new(i)).value();
                mean_disp[0] += w * (p[0] - q[0]);
                mean_disp[1] += w * (p[1] - q[1]);
                mean_disp[2] += w * (p[2] - q[2]);
                w_sum += w;
            }
            if w_sum > 0.0 {
                mean_disp = [
                    mean_disp[0] / w_sum,
                    mean_disp[1] / w_sum,
                    mean_disp[2] / w_sum,
                ];
                let mut frac_drift = cell.cart_to_frac(mean_disp);
                for (f, periodic) in frac_drift.iter_mut().zip(cell.periodic_axes) {
                    if !periodic {
                        *f = 0.0;
                    }
                }
                let cart_drift = cell.frac_to_cart(frac_drift);
                for i in 0..n {
                    let p = unwrapped.position(AtomId::new(i)).value();
                    unwrapped.set_position(
                        AtomId::new(i),
                        Position::new([
                            p[0] - cart_drift[0],
                            p[1] - cart_drift[1],
                            p[2] - cart_drift[2],
                        ]),
                    );
                }
            }
        }
        let rmsd = cartesian_rmsd_bohr(&unwrapped, reference)?;
        return Ok((unwrapped, rmsd));
    }

    // Isolated system: standard weighted Kabsch SVD alignment
    let weights: Vec<f64> = (0..n)
        .map(|i| {
            if mass_weighted {
                mobile.elements[i].mass_amu().max(1e-6)
            } else {
                1.0
            }
        })
        .collect();
    let w_tot: f64 = weights.iter().sum();

    let mut cp = [0.0; 3];
    let mut cq = [0.0; 3];
    for (i, w) in weights.iter().enumerate() {
        let p = mobile.position(AtomId::new(i)).value();
        let q = reference.position(AtomId::new(i)).value();
        for c in 0..3 {
            cp[c] += w * p[c] / w_tot;
            cq[c] += w * q[c] / w_tot;
        }
    }

    // Covariance H = sum_i w_i (p_i - cp) (q_i - cq)^T
    let mut cov = Mat::<f64>::zeros(3, 3);
    for (i, w) in weights.iter().enumerate() {
        let p = mobile.position(AtomId::new(i)).value();
        let q = reference.position(AtomId::new(i)).value();
        for r in 0..3 {
            for c in 0..3 {
                cov[(r, c)] += w * (p[r] - cp[r]) * (q[c] - cq[c]);
            }
        }
    }

    let rot = proper_rotation_from_covariance(&cov);
    let mut aligned = mobile.clone();
    let mut sq_sum = 0.0;
    for i in 0..n {
        let p = mobile.position(AtomId::new(i)).value();
        let dp = [p[0] - cp[0], p[1] - cp[1], p[2] - cp[2]];
        let mut rp = [cq[0], cq[1], cq[2]];
        for r in 0..3 {
            for c in 0..3 {
                rp[r] += rot[(r, c)] * dp[c];
            }
        }
        aligned.set_position(AtomId::new(i), Position::new(rp));
        let q = reference.position(AtomId::new(i)).value();
        for c in 0..3 {
            let d = rp[c] - q[c];
            sq_sum += d * d;
        }
    }

    let rmsd = libm::sqrt(sq_sum / (n as f64));
    Ok((aligned, rmsd))
}

/// MultiOptPy's `kabsch_algorithm` as applied to every new geometry: both structures are
/// centered on their unweighted centroids, `mobile` is rotated onto `reference`, and the result
/// is returned centered at the origin.
pub fn align_unweighted_centered(
    mobile: &MolecularSystem,
    reference: &MolecularSystem,
) -> MoptResult<MolecularSystem> {
    let n = mobile.n_atoms();
    if n != reference.n_atoms() {
        return Err(crate::error::MoptError::coord(
            "cannot align structures with different atom counts",
        ));
    }
    if n == 0 {
        return Ok(mobile.clone());
    }
    let centroid = |s: &MolecularSystem| {
        let mut c = [0.0; 3];
        for i in 0..n {
            let p = s.position(AtomId::new(i)).value();
            for k in 0..3 {
                c[k] += p[k] / n as f64;
            }
        }
        c
    };
    let (cp, cq) = (centroid(mobile), centroid(reference));
    let mut cov = Mat::<f64>::zeros(3, 3);
    for i in 0..n {
        let p = mobile.position(AtomId::new(i)).value();
        let q = reference.position(AtomId::new(i)).value();
        for r in 0..3 {
            for c in 0..3 {
                cov[(r, c)] += (p[r] - cp[r]) * (q[c] - cq[c]);
            }
        }
    }
    let rot = proper_rotation_from_covariance(&cov);
    let mut out = mobile.clone();
    for i in 0..n {
        let p = mobile.position(AtomId::new(i)).value();
        let d = [p[0] - cp[0], p[1] - cp[1], p[2] - cp[2]];
        let mut rp = [0.0; 3];
        for r in 0..3 {
            for c in 0..3 {
                rp[r] += rot[(r, c)] * d[c];
            }
        }
        out.set_position(AtomId::new(i), Position::new(rp));
    }
    Ok(out)
}

/// Compute proper `3 x 3` rotation `R = V diag(1, 1, det(V U^T)) U^T` from covariance `H = P^T W Q`.
pub fn proper_rotation_from_covariance(cov: &Mat<f64>) -> Mat<f64> {
    let Ok(svd) = cov.svd() else {
        return Mat::identity(3, 3);
    };
    let u = svd.U();
    let v = svd.V();

    // Candidate R = V * U^T
    let mut vu_t = Mat::<f64>::zeros(3, 3);
    for r in 0..3 {
        for c in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += v[(r, k)] * u[(c, k)];
            }
            vu_t[(r, c)] = acc;
        }
    }
    let det = vu_t[(0, 0)] * (vu_t[(1, 1)] * vu_t[(2, 2)] - vu_t[(1, 2)] * vu_t[(2, 1)])
        - vu_t[(0, 1)] * (vu_t[(1, 0)] * vu_t[(2, 2)] - vu_t[(1, 2)] * vu_t[(2, 0)])
        + vu_t[(0, 2)] * (vu_t[(1, 0)] * vu_t[(2, 1)] - vu_t[(1, 1)] * vu_t[(2, 0)]);

    let s2 = if det < 0.0 { -1.0 } else { 1.0 };
    let diag = [1.0, 1.0, s2];
    let mut rot = Mat::<f64>::zeros(3, 3);
    for r in 0..3 {
        for c in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += v[(r, k)] * diag[k] * u[(c, k)];
            }
            rot[(r, c)] = acc;
        }
    }
    rot
}

/// Horn's symmetric 4 x 4 matrix of the covariance `s` (`s_ab = sum_i p_i[a] q_i[b]`, rows
/// `xx xy xz / yx yy yz / zx zy zz`): its largest eigenvalue is `max_R sum_i q_i . R p_i` over
/// proper rotations, and its eigenvector is the quaternion of that rotation (Horn, J. Opt. Soc.
/// Am. A 1987, 4, 629). The one copy of the crate (internal coordinates of rotations, RMSD
/// metadynamics).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn horn_matrix(s: &[[f64; 3]; 3]) -> [[f64; 4]; 4] {
    let [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = *s;
    [
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
    ]
}

/// Direct MIC-aware root-mean-square atomic displacement in Bohr without rotation.
pub fn cartesian_rmsd_bohr(a: &MolecularSystem, b: &MolecularSystem) -> MoptResult<f64> {
    let diff = a.displacement_from(b)?;
    let n = a.n_atoms().max(1) as f64;
    let sq: f64 = diff.iter().map(|d| d * d).sum();
    Ok(libm::sqrt(sq / n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::quantity::shorthand::bohr;
    use crate::domain::{positions, Angstrom};
    use crate::domain::{Element, PeriodicCell, SpinCharge};

    fn h2o(rotated: bool) -> MolecularSystem {
        let base = [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ];
        let (c, s) = (0.6_f64.cos(), 0.6_f64.sin());
        let pos: Vec<[f64; 3]> = base
            .iter()
            .map(|p| {
                if rotated {
                    [
                        c * p[0] - s * p[1] + 1.0,
                        s * p[0] + c * p[1] - 2.0,
                        p[2] + 0.5,
                    ]
                } else {
                    *p
                }
            })
            .collect();
        MolecularSystem::new(
            vec![Element::O, Element::H, Element::H],
            &positions::<Angstrom>(&pos),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn kabsch_recovers_rigid_motion() {
        let (aligned, rmsd) = kabsch_align_bohr(&h2o(true), &h2o(false), true).unwrap();
        assert!(rmsd < 1e-10);
        assert!(cartesian_rmsd_bohr(&aligned, &h2o(false)).unwrap() < 1e-10);
    }

    #[test]
    fn periodic_alignment_only_unwraps_and_removes_drift() {
        let cell = PeriodicCell::orthorhombic(bohr(10.0), bohr(10.0), bohr(10.0)).unwrap();
        let mut a = h2o(false);
        a.cell = Some(cell.clone());
        let mut b = a.clone();
        for i in 0..3 {
            let p = b.position(AtomId::new(i)).value();
            b.set_position(
                AtomId::new(i),
                Position::new([p[0] + 10.0 + 0.3, p[1], p[2]]),
            );
        }
        let (aligned, rmsd) = kabsch_align_bohr(&b, &a, false).unwrap();
        assert!(rmsd < 1e-10);
        assert!(
            (aligned.position(AtomId::new(0)).value()[0] - a.position(AtomId::new(0)).value()[0])
                .abs()
                < 1e-10
        );
    }
}
