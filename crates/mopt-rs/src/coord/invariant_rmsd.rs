//! RMSD of two structures that does not depend on the numbering of equivalent atoms, and
//! optionally not on their handedness.
//!
//! Two structures of the same molecule are compared after the best superposition (Kabsch,
//! rotations only) and the best renumbering of the atoms within classes of equivalent atoms
//! (the classes of [`crate::coord::graph::MolecularGraph::classes`], or the elements): from a
//! few starting orientations (the numbering as it is, and the four proper alignments of the
//! principal axes), the atoms of every class are assigned to the nearest atoms of the same
//! class by the Hungarian method ([`crate::math::assignment`]) and the superposition is
//! repeated until the assignment no longer changes; the lowest RMSD wins. With `mirror`, the
//! mirror image of the first structure is tried as well (enantiomeric conformers of an achiral
//! molecule are then one conformer). This is the invariant RMSD in the sense of the CREST
//! program paper (Pracht et al., J. Chem. Phys. 2024, 160, 114110), implemented from that
//! description.
//!
//! Positions are compared as they are: molecules in a periodic cell are made whole first
//! ([`crate::coord::graph::whole_molecules`]).
//!
//! MultiOptPy: `Entrypoints/conformation_search.py:check_identical` (sorted distance matrices,
//!  which ignore the numbering and the handedness); the assignment is new.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::coord::alignment::proper_rotation_from_covariance;
use crate::domain::{AtomId, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::math::assignment::linear_sum_assignment;
use crate::math::linalg::symmetric_eigen;

/// What the comparison may change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsdOptions {
    /// Renumber atoms within their classes.
    pub permutations: bool,
    /// Also compare with the mirror image of the first structure.
    pub mirror: bool,
    /// Rotate (otherwise only the centroids are superposed).
    pub rotation: bool,
}

impl Default for RmsdOptions {
    fn default() -> Self {
        Self {
            permutations: true,
            mirror: false,
            rotation: true,
        }
    }
}

/// The best match of two structures.
#[derive(Debug, Clone, PartialEq)]
pub struct RmsdMatch {
    /// RMSD (Bohr).
    pub rmsd: f64,
    /// Atom `permutation[i]` of the second structure is matched with atom `i` of the first.
    pub permutation: Vec<usize>,
    /// Whether the mirror image of the first structure matched.
    pub mirrored: bool,
}

type Points = Vec<[f64; 3]>;

fn centered(sys: &MolecularSystem) -> Points {
    let n = sys.n_atoms().max(1) as f64;
    let raw: Points = sys.atoms().map(|a| sys.position(a).value()).collect();
    let mut c = [0.0; 3];
    for p in &raw {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    raw.iter()
        .map(|p| [p[0] - c[0], p[1] - c[1], p[2] - c[2]])
        .collect()
}

fn apply(r: &[[f64; 3]; 3], p: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2])
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Rotation taking `p` (centered) onto `q[perm]` (centered) best, or the identity.
fn kabsch(p: &[[f64; 3]], q: &[[f64; 3]], perm: &[usize], rotation: bool) -> [[f64; 3]; 3] {
    if !rotation {
        return IDENTITY;
    }
    let mut cov = Mat::<f64>::zeros(3, 3);
    for (i, a) in p.iter().enumerate() {
        let b = q[perm[i]];
        for r in 0..3 {
            for c in 0..3 {
                cov[(r, c)] += a[r] * b[c];
            }
        }
    }
    let rot = proper_rotation_from_covariance(&cov);
    std::array::from_fn(|i| std::array::from_fn(|j| rot[(i, j)]))
}

fn rmsd_of(p: &[[f64; 3]], q: &[[f64; 3]], perm: &[usize], r: &[[f64; 3]; 3]) -> f64 {
    let n = p.len().max(1) as f64;
    let sum: f64 = p
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let ra = apply(r, a);
            let b = q[perm[i]];
            (0..3).map(|k| (ra[k] - b[k]).powi(2)).sum::<f64>()
        })
        .sum();
    libm::sqrt(sum / n)
}

/// Kabsch RMSD (Bohr) of two structures of the same atoms in the same numbering (no
/// minimum images: molecules in a cell are made whole first).
pub fn direct_rmsd(a: &MolecularSystem, b: &MolecularSystem) -> MoptResult<f64> {
    if a.n_atoms() != b.n_atoms() {
        return Err(MoptError::coord(
            "cannot compare structures of different sizes",
        ));
    }
    let (p, q) = (centered(a), centered(b));
    let perm: Vec<usize> = (0..p.len()).collect();
    let r = kabsch(&p, &q, &perm, true);
    Ok(rmsd_of(&p, &q, &perm, &r))
}

/// Principal axes (columns, by ascending second moment) of centered points.
fn axes(p: &[[f64; 3]]) -> MoptResult<[[f64; 3]; 3]> {
    let mut m = Mat::<f64>::zeros(3, 3);
    for a in p {
        for r in 0..3 {
            for c in 0..3 {
                m[(r, c)] += a[r] * a[c];
            }
        }
    }
    let (_, v) = symmetric_eigen(&m)?;
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| v[(i, j)])))
}

/// The four proper rotations that take the principal axes of `p` onto those of `q`.
fn axis_starts(p: &[[f64; 3]], q: &[[f64; 3]]) -> MoptResult<Vec<[[f64; 3]; 3]>> {
    let (ea, eb) = (axes(p)?, axes(q)?);
    let mut out = Vec::new();
    for signs in [
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ] {
        // R = Eb S Ea^T
        let r: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..3).map(|k| eb[i][k] * signs[k] * ea[j][k]).sum())
        });
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        // The eigenvectors may form a left-handed frame: flip the last sign then.
        let r = if det < 0.0 {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    (0..3)
                        .map(|k| {
                            let s = if k == 2 { -signs[k] } else { signs[k] };
                            eb[i][k] * s * ea[j][k]
                        })
                        .sum()
                })
            })
        } else {
            r
        };
        out.push(r);
    }
    Ok(out)
}

/// Members of every class (atoms of `classes` with the same value), sorted by class.
fn groups(classes: &[usize]) -> Vec<Vec<usize>> {
    let mut keys: Vec<usize> = classes.to_vec();
    keys.sort_unstable();
    keys.dedup();
    keys.iter()
        .map(|k| (0..classes.len()).filter(|&i| classes[i] == *k).collect())
        .collect()
}

/// Assignment of the atoms of `p` (rotated by `r`) to those of `q` of the same class.
fn assign(
    p: &[[f64; 3]],
    q: &[[f64; 3]],
    r: &[[f64; 3]; 3],
    ga: &[Vec<usize>],
    gb: &[Vec<usize>],
) -> MoptResult<Vec<usize>> {
    let mut perm = vec![0usize; p.len()];
    for (a_atoms, b_atoms) in ga.iter().zip(gb) {
        if a_atoms.len() == 1 {
            perm[a_atoms[0]] = b_atoms[0];
            continue;
        }
        let rotated: Vec<[f64; 3]> = a_atoms.iter().map(|&i| apply(r, &p[i])).collect();
        let cost = Mat::from_fn(a_atoms.len(), b_atoms.len(), |i, j| {
            let b = q[b_atoms[j]];
            (0..3).map(|k| (rotated[i][k] - b[k]).powi(2)).sum::<f64>()
        });
        let cols = linear_sum_assignment(&cost)?;
        for (i, c) in cols.into_iter().enumerate() {
            perm[a_atoms[i]] = b_atoms[c];
        }
    }
    Ok(perm)
}

/// Iterations of assignment and superposition from one start.
const MAX_ROUNDS: usize = 30;

/// The best match of `a` to `b` with the classes `classes_a`, `classes_b` of their atoms (the
/// same class names for equivalent atoms; elements will do where nothing better is known).
pub fn invariant_rmsd(
    a: &MolecularSystem,
    b: &MolecularSystem,
    classes_a: &[usize],
    classes_b: &[usize],
    options: &RmsdOptions,
) -> MoptResult<RmsdMatch> {
    let n = a.n_atoms();
    if n != b.n_atoms() || classes_a.len() != n || classes_b.len() != n {
        return Err(MoptError::coord(
            "the structures and their classes of atoms do not match",
        ));
    }
    let (ga, gb) = (groups(classes_a), groups(classes_b));
    let sizes = |g: &[Vec<usize>], c: &[usize]| -> Vec<(usize, usize)> {
        g.iter().map(|m| (c[m[0]], m.len())).collect()
    };
    if sizes(&ga, classes_a) != sizes(&gb, classes_b) {
        return Err(MoptError::coord(
            "the structures have different classes of atoms (different bond graphs)",
        ));
    }
    let q = centered(b);
    let identity: Vec<usize> = (0..n).collect();
    let mut best: Option<RmsdMatch> = None;
    let mirrors: &[bool] = if options.mirror {
        &[false, true]
    } else {
        &[false]
    };
    for &mirrored in mirrors {
        let mut p = centered(a);
        if mirrored {
            for v in &mut p {
                v[0] = -v[0];
            }
        }
        let mut starts = vec![kabsch(&p, &q, &identity, options.rotation)];
        if options.permutations && options.rotation && n > 2 {
            starts.extend(axis_starts(&p, &q)?);
        }
        for start in starts {
            let (mut r, mut perm) = (start, identity.clone());
            if options.permutations {
                for _ in 0..MAX_ROUNDS {
                    let next = assign(&p, &q, &r, &ga, &gb)?;
                    let changed = next != perm;
                    perm = next;
                    r = kabsch(&p, &q, &perm, options.rotation);
                    if !changed {
                        break;
                    }
                }
            }
            let rmsd = rmsd_of(&p, &q, &perm, &r);
            if best.as_ref().is_none_or(|m| rmsd < m.rmsd - 1e-12) {
                best = Some(RmsdMatch {
                    rmsd,
                    permutation: perm,
                    mirrored,
                });
            }
        }
    }
    best.ok_or_else(|| MoptError::coord("no superposition was found"))
}

/// Whether `sys` differs from its mirror image by more than `threshold` (Bohr) after the
/// best rotation and renumbering within `classes`: a chiral structure, whose mirror image is
/// a second structure of the same energy.
pub fn is_chiral(sys: &MolecularSystem, classes: &[usize], threshold: f64) -> MoptResult<bool> {
    let mut mirror = sys.clone();
    for a in sys.atoms() {
        let p = sys.position(a).value();
        mirror.set_position(a, crate::domain::Position::new([-p[0], p[1], p[2]]));
    }
    let m = invariant_rmsd(sys, &mirror, classes, classes, &RmsdOptions::default())?;
    Ok(m.rmsd > threshold)
}

/// `sys` with its atoms renumbered: atom `i` of the result is atom `permutation[i]` of `sys`.
pub fn renumbered(sys: &MolecularSystem, permutation: &[usize]) -> MoptResult<MolecularSystem> {
    let atoms: Vec<AtomId> = permutation.iter().map(|&i| AtomId::new(i)).collect();
    sys.subsystem(&atoms, sys.spin_charge)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::coord::graph::{MolecularGraph, MULTIOPTPY_BOND_SCALE};
    use crate::domain::{Position, SpinCharge};
    use crate::io::xyz::parse_xyz_str;

    fn sys(text: &str) -> MolecularSystem {
        parse_xyz_str(text, SpinCharge::default(), None).unwrap()
    }

    const METHANOL: &str = "6\n\nC 0.000 0.000 0.000\nO 1.410 0.000 0.000\n\
H 1.730 0.900 0.000\nH -0.360 1.030 0.000\nH -0.360 -0.510 0.890\nH -0.360 -0.510 -0.890\n";

    fn rotate(s: &MolecularSystem, angle: f64, shift: [f64; 3]) -> MolecularSystem {
        let mut out = s.clone();
        let (c, si) = (angle.cos(), angle.sin());
        for a in s.atoms() {
            let p = s.position(a).value();
            out.set_position(
                a,
                Position::new([
                    c * p[0] - si * p[2] + shift[0],
                    p[1] + shift[1],
                    si * p[0] + c * p[2] + shift[2],
                ]),
            );
        }
        out
    }

    #[test]
    fn a_rotated_and_renumbered_copy_matches_and_the_numbering_is_found() {
        let a = sys(METHANOL);
        let classes = MolecularGraph::from_system(&a, MULTIOPTPY_BOND_SCALE).classes();
        // Swap two methyl hydrogens and move the copy.
        let perm = [0, 1, 2, 3, 5, 4];
        let b = rotate(&renumbered(&a, &perm).unwrap(), 0.7, [3.0, -1.0, 2.0]);
        let cb = MolecularGraph::from_system(&b, MULTIOPTPY_BOND_SCALE).classes();
        assert!(direct_rmsd(&a, &b).unwrap() > 0.3);
        let m = invariant_rmsd(&a, &b, &classes, &cb, &RmsdOptions::default()).unwrap();
        assert!(m.rmsd < 1e-8, "{}", m.rmsd);
        assert_eq!(m.permutation, vec![0, 1, 2, 3, 5, 4]);
        assert!(!m.mirrored);
        // Without renumbering the swapped hydrogens stay apart.
        let fixed = RmsdOptions {
            permutations: false,
            ..RmsdOptions::default()
        };
        assert!(invariant_rmsd(&a, &b, &classes, &cb, &fixed).unwrap().rmsd > 0.3);
    }

    #[test]
    fn mirror_images_match_only_when_allowed_and_chirality_is_seen() {
        // CHFClBr: a stereocentre.
        let a = sys("5\n\nC 0.0 0.0 0.0\nH 0.63 0.63 0.63\nF -0.8 -0.8 0.8\n\
Cl -1.0 1.0 -1.0\nBr 1.1 -1.1 -1.1\n");
        let classes: Vec<usize> = a
            .elements
            .iter()
            .map(|e| e.atomic_number() as usize)
            .collect();
        let mut b = a.clone();
        for at in a.atoms() {
            let p = a.position(at).value();
            b.set_position(at, Position::new([p[0], -p[1], p[2]]));
        }
        let plain = invariant_rmsd(&a, &b, &classes, &classes, &RmsdOptions::default()).unwrap();
        assert!(plain.rmsd > 0.2);
        let both = RmsdOptions {
            mirror: true,
            ..RmsdOptions::default()
        };
        let m = invariant_rmsd(&a, &b, &classes, &classes, &both).unwrap();
        assert!(m.rmsd < 1e-8 && m.mirrored);
        assert!(is_chiral(&a, &classes, 0.1).unwrap());
        let methanol = sys(METHANOL);
        let c = MolecularGraph::from_system(&methanol, MULTIOPTPY_BOND_SCALE).classes();
        assert!(!is_chiral(&methanol, &c, 0.1).unwrap());
    }
}
