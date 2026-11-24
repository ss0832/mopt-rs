//! Translation-rotation internal coordinates (TRIC; L.-P. Wang and C. Song, J. Chem. Phys.
//! 2016, 144, 214108).
//!
//! Every bonded fragment gets the primitive internal coordinates of its own atoms (bonds,
//! angles, dihedrals; no bonds between fragments) and, for its position and orientation as a
//! whole, three translations (the components of its centroid) and three rotations: the
//! exponential map `v = theta n` of the rotation that takes the fragment's geometry at the start
//! onto its current geometry in the least-squares sense (the quaternion of largest eigenvalue of
//! the 4 x 4 overlap matrix; Horn, J. Opt. Soc. Am. A 1987, 4, 629), multiplied by the radius of
//! gyration of the fragment so that it has the dimension of a length. Molecules in a complex,
//! solute and solvent, or an adsorbate and a surface therefore move relative to each other
//! along well-conditioned coordinates instead of along long, floppy inter-fragment bonds.
//!
//! Details:
//! - A fragment of one atom has translations only. A linear fragment (also every diatomic) has
//!   no orientation about its axis; as in Wang and Song's implementation, a dummy point
//!   `(x_last - x_first) x e0` with a fixed direction `e0` is added to its points, which makes
//!   the three rotations well defined (the one about the axis moves no atom and drops out of
//!   the active space).
//! - Periodic cells: a fragment is unwrapped along its bonds with minimum images at every
//!   evaluation, and translations are compared through the minimum-image displacements of the
//!   atoms, so wrapping atoms into the cell changes nothing. A fragment bonded to its own
//!   periodic image (a slab, a crystal) has translations only.
//! - `tric` delocalizes the set (the eigenvectors of `G = B B^T` with non-zero eigenvalues at
//!   the start, kept fixed, as for `dic`); `tric-p` uses the redundant primitive set.
//! - `tric-cart` stacks the Cartesian coordinates of all atoms, each times a weight `w`, under
//!   the TRIC primitives and delocalizes the stack (the idea of hybrid delocalized internal
//!   coordinates; Billeter, Turner, Thiel, Phys. Chem. Chem. Phys. 2000, 2, 2177). The metric
//!   of the transformation becomes `B^T B + w^2 I`: it always has full rank, its condition
//!   number is bounded by `(s_max^2 + w^2) / w^2`, and the coordinates go over into TRIC for
//!   `w -> 0` and into Cartesian coordinates for large `w`. `tric-cart:W` sets the weight.
//! - The rotation of a fragment is measured from a reference orientation (the one at the
//!   start), and the map is singular at 180 degrees. Before a fragment gets there, at 90
//!   degrees, the coordinates ask to be set up anew ([`CoordinateSystem::needs_reinitialization`]):
//!   the optimizer takes the current structure as the new reference and carries its Hessian
//!   over. A structure that jumps past the range in one step does not stop a run either: the
//!   back-transformation keeps its last valid iterate, and the next iteration starts from new
//!   coordinates.
//!
//! The derivatives of the rotations are exact: with `N q = lambda q` for the overlap matrix
//! `N`, which is linear in the positions, `dq = sum_k q_k (q_k^T dN q) / (lambda - lambda_k)`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::delocalized::{active_space, lost_rank};
use super::CoordinateSystem;
use crate::coord::connectivity::BondConnectivity;
use crate::coord::internal::{PrimitiveOptions, PrimitiveSet};
use crate::domain::{AtomId, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

type V3 = [f64; 3];

/// A fragment counts as linear when the second-largest principal value of its gyration tensor
/// is below this fraction of the largest.
const LINEAR_REL_TOL: f64 = 1e-3;
/// Weight of the Cartesian coordinates in `tric-cart` when none is given. Measured
/// (`docs/source/coordinate_systems.md`): over molecules and complexes, with and without a
/// model Hessian, the iterations in total are 2044 with 0.1 and 2079 with 0.3 (plain TRIC
/// 2125); 0.03 and 1 take more without a model Hessian, 1 by following the Cartesian
/// coordinates in complexes.
pub const DEFAULT_CARTESIAN_WEIGHT: f64 = 0.1;
/// Smallest scalar part of the rotation quaternion (cos of half the angle) that is accepted.
const MIN_QUATERNION_SCALAR: f64 = 0.05;
/// Scalar part below which new coordinates are asked for: a rotation by 90 degrees.
const REINITIALIZE_BELOW: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn centroid(p: &[V3]) -> V3 {
    let mut c = [0.0; 3];
    for v in p {
        c = add(c, *v);
    }
    let n = p.len().max(1) as f64;
    [c[0] / n, c[1] / n, c[2] / n]
}

fn centered(p: &[V3]) -> Vec<V3> {
    let c = centroid(p);
    p.iter().map(|v| sub(*v, c)).collect()
}

/// Horn's symmetric 4 x 4 matrix of `S_ab = sum_i y_i[a] c_i[b]`; its eigenvector of largest
/// eigenvalue is the quaternion of the rotation taking the points `y` onto the points `c`.
fn overlap_matrix(s: &[[f64; 3]; 3]) -> [[f64; 4]; 4] {
    crate::coord::alignment::horn_matrix(s)
}

/// `f(q0) = theta / sin(theta / 2)` with `theta = 2 acos(q0)`, and `df/dq0`, from the squared
/// norm `u2` of the vector part (`u2 = 1 - q0^2`); series for small angles.
fn exponential_map_factor(q0: f64, u2: f64) -> (f64, f64) {
    if u2 < 1e-8 {
        (
            2.0 * (1.0 + u2 / 6.0 + 3.0 * u2 * u2 / 40.0),
            -2.0 / 3.0 - 4.0 * u2 / 15.0,
        )
    } else {
        let u = libm::sqrt(u2);
        let f = 2.0 * libm::atan2(u, q0) / u;
        (f, (q0 * f - 2.0) / u2)
    }
}

/// Orientation reference of a fragment.
#[derive(Debug, Clone)]
struct RotationFrame {
    /// Centered reference points: the atoms, then the dummy point of a linear fragment.
    reference: Vec<V3>,
    /// Fixed direction of the dummy point of a linear fragment.
    e0: Option<V3>,
    /// Radius of gyration of the reference (Bohr).
    weight: f64,
}

/// The rotation vector (already weighted) and its derivative with respect to every point
/// (`jac[i][a][b] = d v_a / d p_i[b]`).
struct Rotation {
    vector: V3,
    jac: Vec<[[f64; 3]; 3]>,
}

impl RotationFrame {
    /// Reference for the unwrapped positions `p` of a fragment of at least two atoms.
    fn new(p: &[V3]) -> MoptResult<Self> {
        let c = centered(p);
        let weight = libm::sqrt(c.iter().map(|v| dot(*v, *v)).sum::<f64>() / c.len() as f64);
        let mut gyration = Mat::<f64>::zeros(3, 3);
        for v in &c {
            for a in 0..3 {
                for b in 0..3 {
                    gyration[(a, b)] += v[a] * v[b];
                }
            }
        }
        let (w, _) = symmetric_eigen(&gyration)?;
        let linear = w[1] < LINEAR_REL_TOL * w[2];
        let e0 = if linear {
            let axis = sub(p[p.len() - 1], p[0]);
            let n = libm::sqrt(dot(axis, axis));
            if n < 1e-8 {
                return Err(MoptError::coord("TRIC: two atoms of a fragment coincide"));
            }
            let axis = [axis[0] / n, axis[1] / n, axis[2] / n];
            // The Cartesian axis least parallel to the fragment.
            let k = (0..3).fold(0, |m, a| if axis[a].abs() < axis[m].abs() { a } else { m });
            let mut unit = [0.0; 3];
            unit[k] = 1.0;
            let e = cross(axis, unit);
            let en = libm::sqrt(dot(e, e));
            Some([e[0] / en, e[1] / en, e[2] / en])
        } else {
            None
        };
        let mut frame = Self {
            reference: Vec::new(),
            e0,
            weight,
        };
        frame.reference = centered(&frame.points(p));
        Ok(frame)
    }

    /// The points entering the superposition: the atoms and, for a linear fragment, the dummy.
    fn points(&self, p: &[V3]) -> Vec<V3> {
        let mut out = p.to_vec();
        if let Some(e0) = self.e0 {
            out.push(cross(sub(p[p.len() - 1], p[0]), e0));
        }
        out
    }

    /// Eigenvalues (ascending) and eigenvectors of the overlap matrix of the reference with the
    /// centered points `points`; the last eigenvector is the rotation quaternion.
    fn superposition(&self, points: &[V3]) -> MoptResult<(Col<f64>, Mat<f64>)> {
        let mut s = [[0.0; 3]; 3];
        for (y, c) in self.reference.iter().zip(points) {
            for a in 0..3 {
                for b in 0..3 {
                    s[a][b] += y[a] * c[b];
                }
            }
        }
        let n = overlap_matrix(&s);
        let (lam, vecs) = symmetric_eigen(&Mat::from_fn(4, 4, |i, j| n[i][j]))?;
        let scale = lam[3].abs().max(1e-300);
        if lam[3] - lam[2] < 1e-10 * scale {
            return Err(MoptError::coord(
                "TRIC: the orientation of a fragment is not defined (degenerate superposition)",
            ));
        }
        Ok((lam, vecs))
    }

    /// Cosine of half the rotation angle of the unwrapped positions `p` (`None`: not defined).
    fn half_angle_cosine(&self, p: &[V3]) -> Option<f64> {
        let (_, vecs) = self.superposition(&centered(&self.points(p))).ok()?;
        Some(vecs[(0, 3)].abs())
    }

    /// Rotation vector of the unwrapped positions `p`; with derivatives with respect to the
    /// atoms when `derivatives`.
    fn rotation(&self, p: &[V3], derivatives: bool) -> MoptResult<Rotation> {
        let points = centered(&self.points(p));
        let (lam, vecs) = self.superposition(&points)?;
        let sign = if vecs[(0, 3)] < 0.0 { -1.0 } else { 1.0 };
        let q: [f64; 4] = std::array::from_fn(|i| sign * vecs[(i, 3)]);
        if q[0] < MIN_QUATERNION_SCALAR {
            return Err(MoptError::coord(
                "TRIC: a fragment has turned by almost 180 degrees from its reference \
                 orientation, beyond the range of its rotation coordinates",
            ));
        }
        let u2 = q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        let (f, df) = exponential_map_factor(q[0], u2);
        let w = self.weight;
        let vector = [w * f * q[1], w * f * q[2], w * f * q[3]];
        let mut jac = Vec::new();
        if derivatives {
            let mut point_jac = vec![[[0.0; 3]; 3]; points.len()];
            for (i, y) in self.reference.iter().enumerate() {
                for b in 0..3 {
                    // d S_ab / d p_i[b] = y_i[a] (the centering drops out: sum_i y_i = 0).
                    let mut ds = [[0.0; 3]; 3];
                    for a in 0..3 {
                        ds[a][b] = y[a];
                    }
                    let dn = overlap_matrix(&ds);
                    let dn_q: [f64; 4] =
                        std::array::from_fn(|r| (0..4).map(|c| dn[r][c] * q[c]).sum());
                    let mut dq = [0.0; 4];
                    for k in 0..3 {
                        let g: f64 = (0..4).map(|r| vecs[(r, k)] * dn_q[r]).sum();
                        let coeff = g / (lam[3] - lam[k]);
                        for (r, d) in dq.iter_mut().enumerate() {
                            *d += coeff * vecs[(r, k)];
                        }
                    }
                    for a in 0..3 {
                        point_jac[i][a][b] = w * (df * dq[0] * q[a + 1] + f * dq[a + 1]);
                    }
                }
            }
            jac = point_jac[..p.len()].to_vec();
            if let Some(e0) = self.e0 {
                // d((x_last - x_first) x e0)_a / d(x_last)_b.
                let c = [
                    [0.0, e0[2], -e0[1]],
                    [-e0[2], 0.0, e0[0]],
                    [e0[1], -e0[0], 0.0],
                ];
                let dummy = point_jac[p.len()];
                let last = p.len() - 1;
                for a in 0..3 {
                    for b in 0..3 {
                        let t: f64 = (0..3).map(|m| dummy[a][m] * c[m][b]).sum();
                        jac[last][a][b] += t;
                        jac[0][a][b] -= t;
                    }
                }
            }
        }
        Ok(Rotation { vector, jac })
    }
}

/// One bonded fragment.
#[derive(Debug, Clone)]
struct Fragment {
    /// Atoms; the first is the root of the spanning tree.
    atoms: Vec<AtomId>,
    /// `(child, parent)` as indices into `atoms`, in an order in which parents come first.
    tree: Vec<(usize, usize)>,
    /// Orientation reference (`None`: translations only).
    rotation: Option<RotationFrame>,
}

/// Whether a bonded fragment of `sys` is bonded to its own periodic image (a crystal, a
/// slab, a chain through the cell).
pub(crate) fn has_periodic_network(sys: &MolecularSystem, conn: &BondConnectivity) -> bool {
    sys.has_pbc()
        && conn
            .fragments
            .iter()
            .any(|atoms| Fragment::spanning(conn, atoms).bonded_to_own_image(sys, conn))
}

impl Fragment {
    fn new(sys: &MolecularSystem, conn: &BondConnectivity, atoms: &[AtomId]) -> MoptResult<Self> {
        let mut frag = Self::spanning(conn, atoms);
        if atoms.len() >= 2 && !frag.bonded_to_own_image(sys, conn) {
            frag.rotation = Some(RotationFrame::new(&frag.unwrapped(sys))?);
        }
        Ok(frag)
    }

    /// The fragment with its spanning tree, without an orientation reference.
    fn spanning(conn: &BondConnectivity, atoms: &[AtomId]) -> Self {
        let index_of = |a: AtomId| atoms.iter().position(|x| *x == a);
        let mut tree = Vec::new();
        let mut seen = vec![false; atoms.len()];
        let mut queue = std::collections::VecDeque::from([0usize]);
        seen[0] = true;
        while let Some(parent) = queue.pop_front() {
            for &nb in &conn.adjacency[atoms[parent].index()] {
                if let Some(child) = index_of(nb) {
                    if !seen[child] {
                        seen[child] = true;
                        tree.push((child, parent));
                        queue.push_back(child);
                    }
                }
            }
        }
        Self {
            atoms: atoms.to_vec(),
            tree,
            rotation: None,
        }
    }

    /// Positions with every atom placed at the minimum image of its parent in the tree.
    fn unwrapped(&self, sys: &MolecularSystem) -> Vec<V3> {
        let mut p = vec![[0.0; 3]; self.atoms.len()];
        p[0] = sys.position(self.atoms[0]).value();
        for &(child, parent) in &self.tree {
            let d = sys.delta(self.atoms[child], self.atoms[parent]).value();
            p[child] = add(p[parent], d);
        }
        p
    }

    /// Whether a bond of the fragment closes a loop through a periodic boundary.
    fn bonded_to_own_image(&self, sys: &MolecularSystem, conn: &BondConnectivity) -> bool {
        if !sys.has_pbc() {
            return false;
        }
        let p = self.unwrapped(sys);
        let index_of = |a: AtomId| self.atoms.iter().position(|x| *x == a);
        self.atoms.iter().enumerate().any(|(i, &a)| {
            conn.adjacency[a.index()].iter().any(|&b| {
                index_of(b).is_some_and(|j| {
                    let direct = sub(p[j], p[i]);
                    let mic = sys.delta(b, a).value();
                    let d = sub(direct, mic);
                    dot(d, d) > 1e-8
                })
            })
        })
    }

    fn n_coords(&self) -> usize {
        if self.rotation.is_some() {
            6
        } else {
            3
        }
    }
}

/// Translation-rotation internal coordinates.
#[derive(Debug, Clone)]
pub struct Tric {
    /// Options used to generate the primitives inside the fragments (`connect_fragments` is
    /// ignored: fragments are never connected by bonds).
    pub options: PrimitiveOptions,
    delocalize: bool,
    primitives: PrimitiveSet,
    fragments: Vec<Fragment>,
    /// Active-space basis (`tric`, `tric-cart`).
    u: Option<Mat<f64>>,
    /// Weight of the stacked Cartesian coordinates (`tric-cart`).
    cartesian_weight: Option<f64>,
    n_atoms: usize,
}

impl Default for Tric {
    fn default() -> Self {
        Self::delocalized()
    }
}

impl Tric {
    /// Delocalized TRIC (`-coord tric`).
    pub fn delocalized() -> Self {
        Self {
            options: PrimitiveOptions::default(),
            delocalize: true,
            primitives: PrimitiveSet::default(),
            fragments: Vec::new(),
            u: None,
            cartesian_weight: None,
            n_atoms: 0,
        }
    }

    /// TRIC stacked with the Cartesian coordinates of all atoms, each times `weight`, and
    /// delocalized (`-coord tric-cart`).
    pub fn with_cartesian(weight: f64) -> MoptResult<Self> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(MoptError::config(format!(
                "the weight of the Cartesian coordinates must be positive, got {weight}"
            )));
        }
        Ok(Self {
            cartesian_weight: Some(weight),
            ..Self::delocalized()
        })
    }

    /// Weight of the stacked Cartesian coordinates (`None`: plain TRIC).
    pub fn cartesian_weight(&self) -> Option<f64> {
        self.cartesian_weight
    }

    /// Delocalized TRIC with `options` for the primitives inside the fragments.
    pub fn with_options(options: PrimitiveOptions) -> Self {
        Self {
            options,
            ..Self::delocalized()
        }
    }

    /// The redundant primitive set (`-coord tric-p`).
    pub fn primitive() -> Self {
        Self {
            delocalize: false,
            ..Self::delocalized()
        }
    }

    /// The primitive internal coordinates inside the fragments.
    pub fn primitives(&self) -> &PrimitiveSet {
        &self.primitives
    }

    /// The fragments (atoms of each), in the order of their translation and rotation
    /// coordinates.
    pub fn fragments(&self) -> Vec<Vec<AtomId>> {
        self.fragments.iter().map(|f| f.atoms.clone()).collect()
    }

    /// Whether fragment `k` has rotation coordinates.
    pub fn has_rotation(&self, k: usize) -> bool {
        self.fragments.get(k).is_some_and(|f| f.rotation.is_some())
    }

    /// Size of the primitive set: internal coordinates, then three translations and (where
    /// defined) three rotations per fragment.
    pub fn primitive_dim(&self) -> usize {
        self.primitives.len()
            + self.fragments.iter().map(Fragment::n_coords).sum::<usize>()
            + self.cartesian_weight.map_or(0, |_| 3 * self.n_atoms)
    }

    fn ensure_ready(&self) -> MoptResult<()> {
        if self.fragments.is_empty() {
            Err(MoptError::coord("TRIC coordinates are not initialized"))
        } else {
            Ok(())
        }
    }

    fn primitive_values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        self.ensure_ready()?;
        let mut q = Col::zeros(self.primitive_dim());
        let internal = self.primitives.values(sys);
        for k in 0..internal.nrows() {
            q[k] = internal[k];
        }
        let mut row = self.primitives.len();
        for frag in &self.fragments {
            let p = frag.unwrapped(sys);
            let c = centroid(&p);
            for a in 0..3 {
                q[row + a] = c[a];
            }
            row += 3;
            if let Some(rot) = &frag.rotation {
                let v = rot.rotation(&p, false)?.vector;
                for a in 0..3 {
                    q[row + a] = v[a];
                }
                row += 3;
            }
        }
        if let Some(w) = self.cartesian_weight {
            let x = sys.coords_flat_bohr();
            for k in 0..x.nrows() {
                q[row + k] = w * x[k];
            }
        }
        Ok(q)
    }

    fn primitive_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        self.ensure_ready()?;
        let mut b = Mat::<f64>::zeros(self.primitive_dim(), sys.dof());
        let internal = self.primitives.wilson_b(sys);
        for r in 0..internal.nrows() {
            for c in 0..internal.ncols() {
                b[(r, c)] = internal[(r, c)];
            }
        }
        let mut row = self.primitives.len();
        for frag in &self.fragments {
            let inv_n = 1.0 / frag.atoms.len() as f64;
            for atom in &frag.atoms {
                for a in 0..3 {
                    b[(row + a, 3 * atom.index() + a)] = inv_n;
                }
            }
            row += 3;
            if let Some(rot) = &frag.rotation {
                let r = rot.rotation(&frag.unwrapped(sys), true)?;
                for (atom, j) in frag.atoms.iter().zip(&r.jac) {
                    for a in 0..3 {
                        for c in 0..3 {
                            b[(row + a, 3 * atom.index() + c)] = j[a][c];
                        }
                    }
                }
                row += 3;
            }
        }
        if let Some(w) = self.cartesian_weight {
            for k in 0..sys.dof() {
                b[(row + k, k)] = w;
            }
        }
        Ok(b)
    }

    fn primitive_difference(
        &self,
        to: &MolecularSystem,
        from: &MolecularSystem,
    ) -> MoptResult<Col<f64>> {
        self.ensure_ready()?;
        let mut d = Col::zeros(self.primitive_dim());
        let internal = self.primitives.difference(to, from);
        for k in 0..internal.nrows() {
            d[k] = internal[k];
        }
        let cell = to.active_cell();
        let mut row = self.primitives.len();
        for frag in &self.fragments {
            // Mean of the minimum-image displacements of the atoms: the displacement of the
            // centroid, whatever periodic images the atoms are stored in.
            let mut shift = [0.0; 3];
            for &atom in &frag.atoms {
                let (a, b) = (to.position(atom).value(), from.position(atom).value());
                let step = match cell {
                    Some(c) => c.mic_delta(a, b),
                    None => sub(a, b),
                };
                shift = add(shift, step);
            }
            let inv_n = 1.0 / frag.atoms.len() as f64;
            for a in 0..3 {
                d[row + a] = shift[a] * inv_n;
            }
            row += 3;
            if let Some(rot) = &frag.rotation {
                let vt = rot.rotation(&frag.unwrapped(to), false)?.vector;
                let vf = rot.rotation(&frag.unwrapped(from), false)?.vector;
                for a in 0..3 {
                    d[row + a] = vt[a] - vf[a];
                }
                row += 3;
            }
        }
        if let Some(w) = self.cartesian_weight {
            // Minimum-image displacements, as for the translations.
            for atom in to.atoms() {
                let (a, b) = (to.position(atom).value(), from.position(atom).value());
                let step = match cell {
                    Some(c) => c.mic_delta(a, b),
                    None => sub(a, b),
                };
                for (k, s) in step.iter().enumerate() {
                    d[row + 3 * atom.index() + k] = w * s;
                }
            }
        }
        Ok(d)
    }
}

impl CoordinateSystem for Tric {
    fn name(&self) -> &str {
        match (self.cartesian_weight, self.delocalize) {
            (Some(_), _) => "tric-cart",
            (None, true) => "tric",
            (None, false) => "tric-p",
        }
    }

    fn initialize(&mut self, sys: &MolecularSystem) -> MoptResult<()> {
        if sys.n_atoms() == 0 {
            return Err(MoptError::coord("TRIC coordinates need at least one atom"));
        }
        self.n_atoms = sys.n_atoms();
        let conn = BondConnectivity::from_system(sys, self.options.bond_scale);
        let options = PrimitiveOptions {
            connect_fragments: false,
            ..self.options.clone()
        };
        self.primitives = if sys.n_atoms() >= 2 {
            PrimitiveSet::generate(sys, &options)?
        } else {
            PrimitiveSet::default()
        };
        self.fragments = conn
            .fragments
            .iter()
            .map(|atoms| Fragment::new(sys, &conn, atoms))
            .collect::<MoptResult<_>>()?;
        self.u = if self.delocalize {
            Some(active_space(&self.primitive_b(sys)?)?)
        } else {
            None
        };
        Ok(())
    }

    fn dim(&self, _sys: &MolecularSystem) -> usize {
        match &self.u {
            Some(u) => u.ncols(),
            None => self.primitive_dim(),
        }
    }

    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        let q = self.primitive_values(sys)?;
        Ok(match &self.u {
            Some(u) => u.transpose() * q,
            None => q,
        })
    }

    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        let b = self.primitive_b(sys)?;
        Ok(match &self.u {
            Some(u) => u.transpose() * b,
            None => b,
        })
    }

    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        let d = self.primitive_difference(to, from)?;
        Ok(match &self.u {
            Some(u) => u.transpose() * d,
            None => d,
        })
    }

    fn needs_reinitialization(&self, sys: &MolecularSystem) -> bool {
        let turned = self.fragments.iter().any(|frag| {
            frag.rotation.as_ref().is_some_and(|rot| {
                rot.half_angle_cosine(&frag.unwrapped(sys))
                    .is_none_or(|c| c < REINITIALIZE_BELOW)
            })
        });
        turned
            || self.primitives.needs_regeneration(sys)
            || self
                .u
                .as_ref()
                .is_some_and(|u| self.primitive_b(sys).map_or(true, |b| lost_rank(u, &b)))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::coord::system::{back_transform, transform::BackTransformOptions};
    use crate::domain::{positions, Angstrom, Bohr, Element, PeriodicCell, SpinCharge};
    use crate::math::linalg::col_norm;

    fn system(symbols: &[&str], rows: &[V3], cell: Option<PeriodicCell>) -> MolecularSystem {
        MolecularSystem::new(
            symbols
                .iter()
                .map(|s| Element::parse_token(s).unwrap())
                .collect(),
            &positions::<Angstrom>(rows),
            SpinCharge::default(),
            cell,
        )
        .unwrap()
    }

    const WATER: [V3; 3] = [
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469],
    ];

    /// Two waters, the second shifted and tilted.
    fn water_dimer() -> MolecularSystem {
        let mut rows = WATER.to_vec();
        for p in WATER {
            rows.push([p[0] + 2.9 + 0.2 * p[2], p[1] + 0.3, p[2] - 0.2 * p[0] + 0.4]);
        }
        system(&["O", "H", "H", "O", "H", "H"], &rows, None)
    }

    /// Rotation by `angle` about `axis` (Rodrigues).
    fn rotate(p: V3, axis: V3, angle: f64) -> V3 {
        let n = libm::sqrt(dot(axis, axis));
        let k = [axis[0] / n, axis[1] / n, axis[2] / n];
        let (s, c) = angle.sin_cos();
        let kxp = cross(k, p);
        let kp = dot(k, p);
        std::array::from_fn(|a| p[a] * c + kxp[a] * s + k[a] * kp * (1.0 - c))
    }

    fn finite_difference_b(coords: &Tric, sys: &MolecularSystem) -> Mat<f64> {
        let x = sys.coords_flat_bohr();
        let h = 1e-5;
        let m = coords.dim(sys);
        let mut b = Mat::<f64>::zeros(m, x.nrows());
        for k in 0..x.nrows() {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            let d = coords
                .displacement(
                    &sys.with_flat_coords_bohr(&xp).unwrap(),
                    &sys.with_flat_coords_bohr(&xm).unwrap(),
                )
                .unwrap();
            for r in 0..m {
                b[(r, k)] = d[r] / (2.0 * h);
            }
        }
        b
    }

    #[test]
    fn rotation_vector_of_a_rigid_rotation_is_angle_times_axis() {
        let sys = system(&["O", "H", "H"], &WATER, None);
        let mut tric = Tric::primitive();
        tric.initialize(&sys).unwrap();
        assert_eq!(tric.fragments().len(), 1);
        assert!(tric.has_rotation(0));
        // 3 internal + 3 translations + 3 rotations = 9 = 3N.
        assert_eq!(tric.primitive_dim(), 9);
        let p0: Vec<V3> = sys.positions().iter().map(|p| p.value()).collect();
        let c0 = centroid(&p0);
        let rg = libm::sqrt(
            p0.iter()
                .map(|p| dot(sub(*p, c0), sub(*p, c0)))
                .sum::<f64>()
                / 3.0,
        );
        for (axis, angle) in [
            ([0.0, 0.0, 1.0], 0.3),
            ([1.0, 2.0, -0.5], 1.1),
            ([0.3, -1.0, 0.2], 1e-7),
            ([-1.0, 0.4, 0.9], 2.6),
        ] {
            let shift = [0.4, -1.3, 2.2];
            let moved: Vec<V3> = p0
                .iter()
                .map(|p| add(add(rotate(sub(*p, c0), axis, angle), c0), shift))
                .collect();
            let to = sys
                .with_flat_coords_bohr(&Col::from_fn(9, |k| moved[k / 3][k % 3]))
                .unwrap();
            let d = tric.displacement(&to, &sys).unwrap();
            let n = libm::sqrt(dot(axis, axis));
            // Internal coordinates unchanged, centroid shifted, rotation = rg * angle * axis.
            for k in 0..3 {
                assert!(d[k].abs() < 1e-10, "internal {k}: {}", d[k]);
                assert!((d[3 + k] - shift[k]).abs() < 1e-10);
                let expect = rg * angle * axis[k] / n;
                assert!((d[6 + k] - expect).abs() < 1e-9 * rg.max(1.0), "{d:?}");
            }
        }
    }

    #[test]
    fn stacked_cartesian_coordinates_give_full_rank_and_both_limits() {
        // Two waters and an argon atom; also exactly linear CO2, where plain internal
        // coordinates have least to offer.
        let mut rows = WATER.to_vec();
        rows.extend(WATER.iter().map(|p| [p[0] + 3.1, p[1] + 0.4, p[2] - 0.2]));
        rows.push([0.5, 3.9, 0.3]);
        let sys = system(&["O", "H", "H", "O", "H", "H", "Ar"], &rows, None);
        let n = sys.dof();
        assert!(Tric::with_cartesian(0.0).is_err() && Tric::with_cartesian(f64::NAN).is_err());
        for w in [1e-3, 0.1, 1.0, 30.0] {
            let mut hybrid = Tric::with_cartesian(w).unwrap();
            hybrid.initialize(&sys).unwrap();
            assert_eq!(hybrid.name(), "tric-cart");
            assert_eq!(hybrid.dim(&sys), n);
            let b = hybrid.wilson_b(&sys).unwrap();
            let fd = finite_difference_b(&hybrid, &sys);
            for r in 0..n {
                for k in 0..n {
                    assert!((b[(r, k)] - fd[(r, k)]).abs() < 1e-7, "w = {w}");
                }
            }
            // The metric is that of TRIC plus w^2: its smallest eigenvalue is at least w^2.
            let metric = b.transpose() * &b;
            let (eig, _) = symmetric_eigen(&metric).unwrap();
            assert!(eig[0] >= w * w * (1.0 - 1e-9), "w = {w}: {}", eig[0]);
            // A step and back.
            let dq = Col::from_fn(n, |k| 0.01 * ((k + 1) as f64).sin());
            let dx = back_transform(&hybrid, &sys, &dq, &BackTransformOptions::default()).unwrap();
            let moved = sys
                .with_flat_coords_bohr(&(sys.coords_flat_bohr() + &dx))
                .unwrap();
            let got = hybrid.displacement(&moved, &sys).unwrap();
            assert!(col_norm(&(&got - &dq)) < 1e-6, "w = {w}");
        }
        // Large weight: the first-order step of a coordinate step is that of w x (Cartesian).
        let mut heavy = Tric::with_cartesian(1e4).unwrap();
        heavy.initialize(&sys).unwrap();
        let b = heavy.wilson_b(&sys).unwrap();
        let metric = b.transpose() * &b;
        for r in 0..n {
            for k in 0..n {
                let expect = if r == k { 1e8 } else { 0.0 };
                assert!((metric[(r, k)] - expect).abs() < 1e-5 * 1e8, "{r} {k}");
            }
        }
    }

    #[test]
    fn new_coordinates_are_asked_for_beyond_a_quarter_turn() {
        let start = water_dimer();
        let mut tric = Tric::delocalized();
        tric.initialize(&start).unwrap();
        assert!(!tric.needs_reinitialization(&start));
        let x0 = start.coords_flat_bohr();
        let p: Vec<V3> = (0..6)
            .map(|i| [x0[3 * i], x0[3 * i + 1], x0[3 * i + 2]])
            .collect();
        let c = centroid(&p[3..]);
        let turned = |angle: f64| {
            let mut x = x0.clone();
            for i in 3..6 {
                let r = add(rotate(sub(p[i], c), [0.3, -1.0, 0.5], angle), c);
                for a in 0..3 {
                    x[3 * i + a] = r[a];
                }
            }
            start.with_flat_coords_bohr(&x).unwrap()
        };
        assert!(!tric.needs_reinitialization(&turned(1.4)));
        assert!(tric.needs_reinitialization(&turned(1.7)));
        // Beyond the range of the map: values fail, the question is still answered.
        let far = turned(3.1);
        assert!(tric.values(&far).is_err());
        assert!(tric.needs_reinitialization(&far));
        // A step that would end there keeps a valid iterate instead of failing.
        let q_far = Col::from_fn(18, |k| if k % 2 == 0 { 3.0 } else { -3.0 });
        let dx = back_transform(&tric, &start, &q_far, &BackTransformOptions::default()).unwrap();
        assert!(dx.iter().all(|v| v.is_finite()));
        // New coordinates at the turned structure measure rotations from there.
        tric.initialize(&far).unwrap();
        assert!(!tric.needs_reinitialization(&far));
        assert_eq!(tric.dim(&far), 18);
        let d = tric.displacement(&turned(3.1 + 0.2), &far).unwrap();
        assert!(d.iter().all(|v| v.is_finite()) && col_norm(&d) > 1e-3);
    }

    #[test]
    fn wilson_matrix_matches_finite_differences() {
        // A tilted second water also tests rotations away from zero: move it first.
        let start = water_dimer();
        for delocalize in [false, true] {
            let mut tric = if delocalize {
                Tric::delocalized()
            } else {
                Tric::primitive()
            };
            tric.initialize(&start).unwrap();
            assert_eq!(tric.fragments().len(), 2);
            if delocalize {
                assert_eq!(tric.dim(&start), 18);
            } else {
                assert_eq!(tric.dim(&start), 2 * (3 + 6));
            }
            // At the start and at a distorted geometry (second water turned by 0.7 rad).
            let x0 = start.coords_flat_bohr();
            let p: Vec<V3> = (0..6)
                .map(|i| [x0[3 * i], x0[3 * i + 1], x0[3 * i + 2]])
                .collect();
            let c = centroid(&p[3..]);
            let mut x1 = x0.clone();
            for i in 3..6 {
                let r = add(rotate(sub(p[i], c), [0.2, 1.0, -0.4], 0.7), c);
                for a in 0..3 {
                    x1[3 * i + a] = r[a] + 0.05 * ((i + a) as f64).sin();
                }
            }
            for sys in [start.clone(), start.with_flat_coords_bohr(&x1).unwrap()] {
                let b = tric.wilson_b(&sys).unwrap();
                let fd = finite_difference_b(&tric, &sys);
                let mut worst = 0.0_f64;
                for r in 0..b.nrows() {
                    for k in 0..b.ncols() {
                        worst = worst.max((b[(r, k)] - fd[(r, k)]).abs());
                    }
                }
                assert!(worst < 1e-7, "delocalize={delocalize}: {worst}");
            }
        }
    }

    #[test]
    fn linear_fragments_and_single_atoms() {
        // CO2 (linear), N2 (diatomic), and an argon atom.
        let sys = system(
            &["C", "O", "O", "N", "N", "Ar"],
            &[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.16],
                [0.0, 0.0, -1.16],
                [3.5, 0.2, 0.1],
                [4.2, 0.9, 0.3],
                [0.5, 4.0, -1.0],
            ],
            None,
        );
        let mut tric = Tric::primitive();
        tric.initialize(&sys).unwrap();
        assert_eq!(tric.fragments().len(), 3);
        assert!(tric.has_rotation(0) && tric.has_rotation(1) && !tric.has_rotation(2));
        let b = tric.wilson_b(&sys).unwrap();
        let fd = finite_difference_b(&tric, &sys);
        for r in 0..b.nrows() {
            for k in 0..b.ncols() {
                assert!((b[(r, k)] - fd[(r, k)]).abs() < 1e-7, "row {r}, column {k}");
            }
        }
        // Delocalized: exactly 3N coordinates (the rotations about the molecular axes drop out).
        let mut tric = Tric::delocalized();
        tric.initialize(&sys).unwrap();
        assert_eq!(tric.dim(&sys), 18);
        // A step back and forth.
        let dq = Col::from_fn(18, |k| 0.02 * ((k + 1) as f64).cos());
        let dx = back_transform(&tric, &sys, &dq, &BackTransformOptions::default()).unwrap();
        let moved = sys
            .with_flat_coords_bohr(&(sys.coords_flat_bohr() + &dx))
            .unwrap();
        let got = tric.displacement(&moved, &sys).unwrap();
        assert!(
            col_norm(&(&got - &dq)) < 1e-6,
            "{}",
            col_norm(&(&got - &dq))
        );
    }

    #[test]
    fn periodic_cells_unwrap_fragments_and_keep_slabs_to_translations() {
        // A water across the boundary of a 6 Angstrom box and a chain bonded to its own image.
        let side = Quantity::<Angstrom>::new(6.0);
        let cell = PeriodicCell::orthorhombic(side, side, side).unwrap();
        let mut rows: Vec<V3> = WATER
            .iter()
            .map(|p| [p[0] + 5.9, p[1] + 3.0, p[2] + 3.0])
            .collect();
        // Hydrogen chain along x with period 6 Angstrom: 8 atoms 0.75 Angstrom apart.
        for k in 0..8 {
            rows.push([0.75 * k as f64, 0.5, 0.5]);
        }
        let mut symbols = vec!["O", "H", "H"];
        symbols.extend(["H"; 8]);
        let sys = system(&symbols, &rows, Some(cell));
        let mut tric = Tric::primitive();
        tric.initialize(&sys).unwrap();
        assert_eq!(tric.fragments().len(), 2);
        assert!(tric.has_rotation(0), "the water rotates as a whole");
        assert!(!tric.has_rotation(1), "the periodic chain only translates");
        let b = tric.wilson_b(&sys).unwrap();
        let fd = finite_difference_b(&tric, &sys);
        for r in 0..b.nrows() {
            for k in 0..b.ncols() {
                assert!((b[(r, k)] - fd[(r, k)]).abs() < 1e-7, "row {r}, column {k}");
            }
        }
        // Wrapping an atom into the cell is no displacement.
        let mut x = sys.coords_flat_bohr();
        let l = Quantity::<Angstrom>::new(6.0).to::<Bohr>().value();
        x[3] -= l;
        x[0] -= l;
        let wrapped = sys.with_flat_coords_bohr(&x).unwrap();
        let d = tric.displacement(&wrapped, &sys).unwrap();
        assert!(col_norm(&d) < 1e-9, "{}", col_norm(&d));
    }

    use crate::domain::Quantity;
}
