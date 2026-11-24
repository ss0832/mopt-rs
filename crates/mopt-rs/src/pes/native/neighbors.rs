//! Full neighbor lists including periodic images, by the linked-cell method.
//!
//! The atoms (wrapped into the cell along its periodic axes) and those of their periodic
//! images that lie within the cutoff of the box of the atoms are sorted into a grid of
//! cells no smaller than the cutoff; the neighbors of an atom are then looked for in its
//! cell and the 26 around it. Time and memory grow with the number of atoms, not with its
//! square. Every lattice translation that can bring an image within the cutoff is
//! included, so cutoffs larger than half the cell (small cells, metals) are handled
//! correctly. An atom can appear as its own neighbor through a nonzero lattice translation;
//! such displacements do not depend on the coordinates.
//!
//! [`VerletList`] keeps the pairs within the cutoff plus a skin between evaluations and gives
//! the same lists as a fresh search until an atom has moved half the skin.
//!
//! MultiOptPy: `Utils/pbc.py` (periodic images); the Verlet list is new in mopt-rs.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{AtomId, Bohr, MolecularSystem, PeriodicCell, Quantity};

/// One neighbor of a central atom.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    /// Neighbor atom (possibly the central atom itself, through a lattice translation).
    pub atom: AtomId,
    /// `r_neighbor_image - r_center` in Bohr.
    pub vector: [f64; 3],
    /// Distance in Bohr.
    pub distance: f64,
    /// Whether the displacement involves a nonzero lattice translation of the central atom
    /// onto itself (coordinate-independent).
    pub self_image: bool,
}

/// The grid holds at most this many cells per point (and [`SPARE_CELLS`] more): atoms far
/// apart in a large empty box get larger cells instead of a grid of empty ones.
const CELLS_PER_POINT: usize = 4;
/// See [`CELLS_PER_POINT`].
const SPARE_CELLS: usize = 64;
/// End of a chain of points in a cell.
const NONE: usize = usize::MAX;

/// Lattice translations that can bring an image of a point of the cell within `cutoff` of
/// another point of the cell, as integer multiples of the lattice vectors with their vectors
/// (Bohr); the zero translation comes first, the others in lexicographic order.
fn image_shifts(cell: &PeriodicCell, cutoff: f64) -> Vec<([i32; 3], [f64; 3])> {
    // Two points of the cell differ by less than one cell along an axis, so one more
    // translation than the cutoff spans is needed.
    cell.lattice_translations(cell.translation_ranges(cutoff, 1, 0))
}

/// The order of the neighbors of an atom: by atom, then the zero translation first and the
/// others in lexicographic order (the order of [`image_shifts`], whatever its ranges).
fn order_key(atom: usize, shift: [i32; 3]) -> (usize, bool, [i32; 3]) {
    (atom, shift != [0; 3], shift)
}

/// An atom or one of its images.
struct Point {
    atom: usize,
    /// Index of its lattice translation (0: the atom itself).
    shift: usize,
    position: [f64; 3],
}

/// Cells of edge `>= cutoff` over a box, with the points of each cell chained.
struct Grid {
    low: [f64; 3],
    counts: [usize; 3],
    edges: [f64; 3],
    /// First point of each cell.
    head: Vec<usize>,
    /// Next point of the same cell.
    next: Vec<usize>,
}

impl Grid {
    fn new(low: [f64; 3], high: [f64; 3], cutoff: f64, points: &[Point]) -> Self {
        let extent: [f64; 3] = std::array::from_fn(|k| (high[k] - low[k]).max(cutoff));
        let mut counts: [usize; 3] =
            std::array::from_fn(|k| (libm::floor(extent[k] / cutoff) as usize).max(1));
        let limit = CELLS_PER_POINT * points.len() + SPARE_CELLS;
        let total = counts.iter().map(|c| *c as f64).product::<f64>();
        if total > limit as f64 {
            let shrink = libm::cbrt(total / limit as f64);
            for c in &mut counts {
                *c = (libm::floor(*c as f64 / shrink) as usize).max(1);
            }
        }
        let edges: [f64; 3] = std::array::from_fn(|k| extent[k] / counts[k] as f64);
        let mut grid = Self {
            low,
            counts,
            edges,
            head: vec![NONE; counts[0] * counts[1] * counts[2]],
            next: vec![NONE; points.len()],
        };
        for (p, point) in points.iter().enumerate() {
            let cell = grid.index(grid.cell_of(point.position));
            grid.next[p] = grid.head[cell];
            grid.head[cell] = p;
        }
        grid
    }

    fn cell_of(&self, position: [f64; 3]) -> [usize; 3] {
        std::array::from_fn(|k| {
            let c = libm::floor((position[k] - self.low[k]) / self.edges[k]);
            (c.max(0.0) as usize).min(self.counts[k] - 1)
        })
    }

    fn index(&self, cell: [usize; 3]) -> usize {
        (cell[0] * self.counts[1] + cell[1]) * self.counts[2] + cell[2]
    }

    /// Call `visit` with every point in the cell of `position` and the cells around it.
    fn around(&self, position: [f64; 3], mut visit: impl FnMut(usize)) {
        let centre = self.cell_of(position);
        let range =
            |k: usize| centre[k].saturating_sub(1)..=(centre[k] + 1).min(self.counts[k] - 1);
        for a in range(0) {
            for b in range(1) {
                for c in range(2) {
                    let mut p = self.head[self.index([a, b, c])];
                    while p != NONE {
                        visit(p);
                        p = self.next[p];
                    }
                }
            }
        }
    }
}

/// The positions of the atoms wrapped into the cell along its periodic axes.
fn wrapped_positions(sys: &MolecularSystem) -> Vec<[f64; 3]> {
    let cell = sys.active_cell();
    sys.atoms()
        .map(|a| {
            let p = sys.position(a).value();
            cell.map_or(p, |c| c.wrap_point(p))
        })
        .collect()
}

/// The vector from `centre` to the image of `p` by `shift` and its length (Bohr): the same
/// operations for a fresh search and a kept list, so the same bits.
fn displacement(centre: [f64; 3], p: [f64; 3], shift: [f64; 3]) -> ([f64; 3], f64) {
    let position = [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
    let v: [f64; 3] = std::array::from_fn(|k| position[k] - centre[k]);
    (v, libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
}

/// A pair that [`search`] found.
struct Found {
    /// The translation, as multiples of the lattice vectors and as a vector (Bohr).
    key: [i32; 3],
    shift: [f64; 3],
    /// The neighbor (its distance 0 for a coincident atom, with `coincident`).
    neighbor: Neighbor,
}

/// The pairs within `rc` by the linked cells, from the wrapped positions `base`: for every
/// atom its neighbors in the order of [`order_key`]; with `coincident`, also those at
/// distance 0 (other atoms than itself at the same place, which a list that is kept must hold
/// since they can move apart).
fn search(sys: &MolecularSystem, base: &[[f64; 3]], rc: f64, coincident: bool) -> Vec<Vec<Found>> {
    let n = base.len();
    let mut out: Vec<Vec<Found>> = (0..n).map(|_| Vec::new()).collect();
    if n == 0 || rc.is_nan() || rc <= 0.0 {
        return out;
    }
    let shifts = sys
        .active_cell()
        .map_or_else(|| vec![([0; 3], [0.0; 3])], |c| image_shifts(c, rc));
    // The box of the atoms, widened by the cutoff: images outside it have no neighbor.
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for p in base {
        for k in 0..3 {
            low[k] = low[k].min(p[k] - rc);
            high[k] = high[k].max(p[k] + rc);
        }
    }
    let mut points = Vec::with_capacity(n);
    for (shift, (_, s)) in shifts.iter().enumerate() {
        for (atom, p) in base.iter().enumerate() {
            let position = [p[0] + s[0], p[1] + s[1], p[2] + s[2]];
            if (0..3).all(|k| position[k] >= low[k] && position[k] <= high[k]) {
                points.push(Point {
                    atom,
                    shift,
                    position,
                });
            }
        }
    }
    let grid = Grid::new(low, high, rc, &points);
    for (i, (centre, list)) in base.iter().zip(&mut out).enumerate() {
        grid.around(*centre, |p| {
            let point = &points[p];
            let self_image = point.atom == i;
            if self_image && point.shift == 0 {
                return;
            }
            let (key, shift) = shifts[point.shift];
            let (v, r) = displacement(*centre, base[point.atom], shift);
            if r < rc && (r > 0.0 || coincident) {
                list.push(Found {
                    key,
                    shift,
                    neighbor: Neighbor {
                        atom: AtomId::new(point.atom),
                        vector: v,
                        distance: r,
                        self_image,
                    },
                });
            }
        });
        list.sort_by_key(|f| order_key(f.neighbor.atom.index(), f.key));
    }
    out
}

/// Neighbors of every atom within `cutoff` (strictly closer than the cutoff), ordered by
/// atom and lattice translation.
pub fn neighbor_list(sys: &MolecularSystem, cutoff: Quantity<Bohr>) -> Vec<Vec<Neighbor>> {
    let base = wrapped_positions(sys);
    search(sys, &base, cutoff.value(), false)
        .into_iter()
        .map(|list| list.into_iter().map(|f| f.neighbor).collect())
        .collect()
}

/// The skin of the Verlet lists of the built-in potentials (Bohr): the pairs within the
/// cutoff plus this are kept, and the list serves until an atom has moved half of it
/// (its cost and the choice: `docs/source/benchmarks.md`, "Verlet lists of the metal potentials").
pub const VERLET_SKIN_BOHR: f64 = 1.0;

/// The lists a [`VerletList`] keeps at most (one per structure that is evaluated in turn, as
/// the images of a band or the structures of a finite-difference Hessian).
const VERLET_LISTS: usize = 8;

/// The candidates of a Verlet list: the pairs within the cutoff plus the skin at the
/// structure where it was built.
#[derive(Debug)]
struct Candidates {
    /// The cutoff it was built for (Bohr).
    cutoff: f64,
    /// The cell it was built in.
    cell: Option<PeriodicCell>,
    /// The wrapped positions it was built at, and the translations the wrapping applied.
    wrapped: Vec<[f64; 3]>,
    translations: Vec<[f64; 3]>,
    /// For every atom its candidates (atom, translation, vector of the translation) in the
    /// order of [`order_key`].
    pairs: Vec<Vec<(usize, bool, [f64; 3])>>,
}

impl Candidates {
    /// Whether the list holds every pair within `cutoff` of `sys` (wrapped positions `base`,
    /// translations `translations`): the same cutoff and cell, no atom wrapped by another
    /// translation, and none moved by half the skin or more.
    fn serves(
        &self,
        sys: &MolecularSystem,
        cutoff: f64,
        base: &[[f64; 3]],
        translations: &[[f64; 3]],
        skin: f64,
    ) -> bool {
        // The wrapping moves an atom by a lattice vector or not at all; rounding of the
        // translation is far below this.
        const SAME_TRANSLATION: f64 = 1e-6;
        let half = 0.5 * skin;
        self.cutoff.to_bits() == cutoff.to_bits()
            && self.wrapped.len() == base.len()
            && self.cell.as_ref() == sys.active_cell()
            && self
                .translations
                .iter()
                .zip(translations)
                .all(|(a, b)| (0..3).all(|k| (a[k] - b[k]).abs() < SAME_TRANSLATION))
            && self.wrapped.iter().zip(base).all(|(a, b)| {
                let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
                // Strictly less, with a margin for the rounding of the distances.
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < half * (1.0 - 1e-9)
            })
    }
}

/// A Verlet list: the pairs within the cutoff plus a skin, built by [`neighbor_list`]'s linked
/// cells and kept until an atom has moved half the skin, the cell has changed or an atom has
/// been wrapped through it. Every call gives exactly what [`neighbor_list`] gives (the same
/// neighbors in the same order, their vectors and distances computed by the same operations
/// from the same positions): a pair within the cutoff now was within the cutoff plus the skin
/// when the list was built, since neither of its atoms moved half the skin.
///
/// The lists are kept per instance (a clone starts empty) behind a lock that is held only to
/// look a list up or to store one, never while the neighbors are computed, so evaluations at
/// the same time never wait for each other's work, and the results never depend on which
/// list served (they are those of a fresh search).
#[derive(Debug)]
pub struct VerletList {
    skin: f64,
    lists: std::sync::Mutex<Vec<std::sync::Arc<Candidates>>>,
}

impl Default for VerletList {
    fn default() -> Self {
        Self::new(VERLET_SKIN_BOHR)
    }
}

impl Clone for VerletList {
    fn clone(&self) -> Self {
        Self::new(self.skin)
    }
}

impl VerletList {
    /// An empty list with the skin `skin` (Bohr; 0 searches afresh at every call).
    pub fn new(skin: f64) -> Self {
        Self {
            skin: skin.max(0.0),
            lists: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// The skin (Bohr).
    pub fn skin(&self) -> f64 {
        self.skin
    }

    /// The neighbors of every atom of `sys` within `cutoff`: those of
    /// [`neighbor_list`]`(sys, cutoff)`.
    pub fn neighbors(&self, sys: &MolecularSystem, cutoff: Quantity<Bohr>) -> Vec<Vec<Neighbor>> {
        let rc = cutoff.value();
        if self.skin <= 0.0 || rc.is_nan() || rc <= 0.0 {
            return neighbor_list(sys, cutoff);
        }
        let base = wrapped_positions(sys);
        let translations: Vec<[f64; 3]> = sys
            .atoms()
            .zip(&base)
            .map(|(a, w)| {
                let p = sys.position(a).value();
                std::array::from_fn(|k| w[k] - p[k])
            })
            .collect();
        let found = self.lists.lock().ok().and_then(|lists| {
            lists
                .iter()
                .rev()
                .find(|c| c.serves(sys, rc, &base, &translations, self.skin))
                .cloned()
        });
        let candidates = match found {
            Some(c) => c,
            None => {
                let wide = search(sys, &base, rc + self.skin, true);
                let built = std::sync::Arc::new(Candidates {
                    cutoff: rc,
                    cell: sys.active_cell().cloned(),
                    wrapped: base.clone(),
                    translations,
                    pairs: wide
                        .into_iter()
                        .map(|list| {
                            list.into_iter()
                                .map(|f| (f.neighbor.atom.index(), f.neighbor.self_image, f.shift))
                                .collect()
                        })
                        .collect(),
                });
                if let Ok(mut lists) = self.lists.lock() {
                    if lists.len() >= VERLET_LISTS {
                        lists.remove(0);
                    }
                    lists.push(std::sync::Arc::clone(&built));
                }
                built
            }
        };
        base.iter()
            .zip(&candidates.pairs)
            .map(|(centre, list)| {
                list.iter()
                    .filter_map(|&(atom, self_image, shift)| {
                        let (v, r) = displacement(*centre, base[atom], shift);
                        (r < rc && r > 0.0).then_some(Neighbor {
                            atom: AtomId::new(atom),
                            vector: v,
                            distance: r,
                            self_image,
                        })
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::quantity::shorthand::bohr;
    use crate::domain::{positions, Element, SpinCharge, Vector3};

    /// Every pair and every lattice translation, tried one by one.
    fn all_pairs(sys: &MolecularSystem, rc: f64) -> Vec<Vec<(usize, [f64; 3])>> {
        let shifts = match sys.active_cell() {
            Some(cell) => cell.lattice_shifts_within_cutoff(rc),
            None => vec![[0.0; 3]],
        };
        let mut out = vec![Vec::new(); sys.n_atoms()];
        for i in sys.atoms() {
            for j in sys.atoms() {
                let d0 = sys.delta(j, i).value();
                for s in &shifts {
                    if i == j && s.iter().all(|c| *c == 0.0) {
                        continue;
                    }
                    let v = [d0[0] + s[0], d0[1] + s[1], d0[2] + s[2]];
                    let r = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                    if r < rc && r > 0.0 {
                        out[i.index()].push((j.index(), v));
                    }
                }
            }
        }
        out
    }

    /// Pseudo-random numbers in [0, 1).
    fn stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn scattered(n: usize, size: f64, cell: Option<PeriodicCell>, seed: u64) -> MolecularSystem {
        let mut next = stream(seed);
        let points: Vec<[f64; 3]> = (0..n)
            .map(|_| std::array::from_fn(|_| size * (next() - 0.25) * 1.5))
            .collect();
        MolecularSystem::new(
            vec![Element::H; n],
            &positions::<Bohr>(&points),
            SpinCharge::default(),
            cell,
        )
        .unwrap()
    }

    fn same_lists(sys: &MolecularSystem, rc: f64) {
        let cells = neighbor_list(sys, bohr(rc));
        let pairs = all_pairs(sys, rc);
        for (i, (found, expected)) in cells.iter().zip(&pairs).enumerate() {
            assert_eq!(found.len(), expected.len(), "atom {i}");
            // The same displacements, whatever their order.
            for (atom, v) in expected {
                assert!(
                    found.iter().any(|nb| nb.atom.index() == *atom
                        && (0..3).all(|k| (nb.vector[k] - v[k]).abs() < 1e-9)),
                    "atom {i}: neighbor {atom} at {v:?} is missing"
                );
            }
            for nb in found {
                let r = nb.vector.iter().map(|c| c * c).sum::<f64>().sqrt();
                assert!((r - nb.distance).abs() < 1e-12 && r < rc);
                assert_eq!(nb.self_image, nb.atom.index() == i);
            }
        }
    }

    #[test]
    fn simple_cubic_counts_images() {
        // One atom in a 4 Bohr cubic cell: 6 neighbors at 4, 12 at 5.66, 8 at 6.93 Bohr.
        let cell = PeriodicCell::orthorhombic(bohr(4.0), bohr(4.0), bohr(4.0)).unwrap();
        let sys = MolecularSystem::new(
            vec![Element::H],
            &positions::<Bohr>(&[[0.3, 0.1, 3.9]]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap();
        let nl = neighbor_list(&sys, bohr(6.0));
        assert_eq!(nl[0].len(), 18);
        assert!(nl[0].iter().all(|n| n.self_image));
        assert_eq!(neighbor_list(&sys, bohr(7.0))[0].len(), 26);
    }

    #[test]
    fn the_cells_find_the_neighbors_of_a_search_over_all_pairs() {
        // Isolated atoms, some of them far outside the others.
        same_lists(&scattered(150, 30.0, None, 1), 6.5);
        let mut far = scattered(40, 12.0, None, 2);
        let mut x = far.coords_flat_bohr();
        x[0] += 4000.0;
        x[4] -= 2500.0;
        far = far.with_flat_coords_bohr(&x).unwrap();
        same_lists(&far, 5.0);
        // A skewed cell with atoms outside it (they are wrapped), cutoffs below and above
        // half its widths and beyond the whole cell.
        let skewed = PeriodicCell::from_lattice_vectors(
            Vector3::<Bohr>::new([11.0, 0.0, 0.0]),
            Vector3::<Bohr>::new([3.5, 9.0, 0.0]),
            Vector3::<Bohr>::new([-2.0, 4.0, 12.0]),
            [true, true, true],
        )
        .unwrap();
        for rc in [3.0, 6.5, 14.0] {
            same_lists(&scattered(60, 14.0, Some(skewed.clone()), 3), rc);
        }
        // A slab (two periodic axes) and a wire (one).
        for axes in [[true, true, false], [false, false, true]] {
            let mut partial = skewed.clone();
            partial.periodic_axes = axes;
            same_lists(&scattered(50, 14.0, Some(partial), 4), 7.0);
        }
        // One atom alone, and a cutoff of nothing.
        assert!(neighbor_list(&scattered(1, 1.0, None, 6), bohr(5.0))[0].is_empty());
        assert!(neighbor_list(&scattered(5, 1.0, None, 6), bohr(0.0))
            .iter()
            .all(Vec::is_empty));
    }

    /// The bits of a neighbor list.
    fn list_bits(nl: &[Vec<Neighbor>]) -> Vec<(usize, bool, [u64; 4])> {
        nl.iter()
            .flatten()
            .map(|nb| {
                (
                    nb.atom.index(),
                    nb.self_image,
                    [
                        nb.vector[0].to_bits(),
                        nb.vector[1].to_bits(),
                        nb.vector[2].to_bits(),
                        nb.distance.to_bits(),
                    ],
                )
            })
            .chain(nl.iter().map(|l| (l.len(), false, [0; 4])))
            .collect()
    }

    /// A kept list gives the bits of a fresh search at every step of a random walk of the
    /// atoms (through the faces of the cell, so that they are wrapped), with changes of the
    /// cell in between and two structures evaluated in turn; and it is rebuilt only now and
    /// then.
    #[test]
    fn a_verlet_list_gives_the_bits_of_a_fresh_search() {
        let mut next = stream(11);
        let size = 14.0;
        let mut cell = PeriodicCell::orthorhombic(bohr(size), bohr(size), bohr(size)).unwrap();
        let mut sys = scattered(80, size / 1.5, Some(cell.clone()), 9);
        let mut slab = cell.clone();
        slab.periodic_axes = [true, true, false];
        let other = scattered(30, 10.0, Some(slab), 5);
        let list = VerletList::default();
        let rc = bohr(6.0);
        for step in 0..120 {
            let mut x = sys.coords_flat_bohr();
            for v in x.iter_mut() {
                *v += 0.12 * (next() - 0.5);
            }
            sys = sys.with_flat_coords_bohr(&x).unwrap();
            if step % 40 == 39 {
                // A cell compressed by 1 %.
                let scale = 0.99;
                cell =
                    PeriodicCell::orthorhombic(bohr(size * scale), bohr(size * scale), bohr(size))
                        .unwrap();
                sys = MolecularSystem::from_flat_bohr(
                    sys.elements.clone(),
                    &sys.coords_flat_bohr(),
                    SpinCharge::default(),
                    Some(cell.clone()),
                )
                .unwrap();
            }
            assert_eq!(
                list_bits(&list.neighbors(&sys, rc)),
                list_bits(&neighbor_list(&sys, rc)),
                "step {step}"
            );
            assert_eq!(
                list_bits(&list.neighbors(&other, rc)),
                list_bits(&neighbor_list(&other, rc)),
                "step {step}, the other structure"
            );
        }
        let kept = list.lists.lock().map(|l| l.len()).unwrap_or(0);
        assert!((1..=VERLET_LISTS).contains(&kept), "{kept}");
        // Without a skin: a fresh search every time.
        assert_eq!(
            list_bits(&VerletList::new(0.0).neighbors(&sys, rc)),
            list_bits(&neighbor_list(&sys, rc))
        );
        // Molecules without a cell, and an atom on top of another one that moves away.
        let mut pair = MolecularSystem::new(
            vec![Element::H; 3],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let molecule = VerletList::default();
        for step in 0..5 {
            assert_eq!(
                list_bits(&molecule.neighbors(&pair, rc)),
                list_bits(&neighbor_list(&pair, rc)),
                "step {step}"
            );
            let mut x = pair.coords_flat_bohr();
            x[3] += 0.05;
            pair = pair.with_flat_coords_bohr(&x).unwrap();
        }
    }

    /// The cells against all pairs: the same neighbours for 1000 atoms on every run, and
    /// the times for 4000 and 16000 atoms with `MOPT_FULL_MEASUREMENTS` set (all pairs of
    /// 16000 atoms take minutes in a build without optimization).
    #[test]
    fn time_of_the_neighbor_search() {
        let sizes: &[usize] = if std::env::var_os("MOPT_FULL_MEASUREMENTS").is_some() {
            &[1000, 4000, 16000]
        } else {
            &[1000]
        };
        for &n in sizes {
            // Constant density: about 60 neighbors within the cutoff.
            let size = 4.0 * (n as f64).cbrt();
            let cell = PeriodicCell::orthorhombic(bohr(size), bohr(size), bohr(size)).unwrap();
            let sys = scattered(n, size / 1.5, Some(cell), 7);
            let start = std::time::Instant::now();
            let cells = neighbor_list(&sys, bohr(10.0));
            let with_cells = start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            let pairs = all_pairs(&sys, 10.0);
            let with_pairs = start.elapsed().as_secs_f64();
            let count: usize = cells.iter().map(Vec::len).sum();
            assert_eq!(count, pairs.iter().map(Vec::len).sum::<usize>());
            println!(
                "{n} atoms, {:.1} neighbors each: cells {with_cells:.4} s, all pairs {with_pairs:.4} s",
                count as f64 / n as f64
            );
        }
    }
}
