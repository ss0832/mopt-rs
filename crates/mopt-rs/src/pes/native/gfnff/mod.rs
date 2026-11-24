//! GFN-FF (S. Spicher, S. Grimme, "Robust Atomistic Modeling of Materials, Organometallic, and
//! Biochemical Systems", Angew. Chem. Int. Ed. 2020, 59, 15665, DOI 10.1002/anie.202004239) for
//! H to Rn (Z = 1 to 86), built into mopt-rs.
//!
//! Provenance: a port of the official implementation in grimme-lab/xtb 6.7.1 (`src/gfnff`,
//! LGPL-3.0-or-later; version `angewChem2020_2`, accuracy 0.1) for isolated molecules: the
//! topology setup of `gfnff_ini.f90` / `gfnff_ini2.F90` (neighbours and hybridization, rings,
//! topology EEQ charges, pi systems with the iterative Hueckel bond orders, bond / bend /
//! torsion / out-of-plane parameters, HB / XB / bonded-ATM lists) and the energy and gradient
//! of `gfnff_eg.f90` / `gdisp0.f90`. The parameters are generated from xtb's files by
//! `tools/params/gen_gfnff.py` into [`crate::parameters::gfnff`]; they are those of the xtb
//! 6.7.1 release, whose tables end at Rn. The branches of the setup for the main-group metals,
//! boron, the noble gases, the transition metals and the lanthanides (their hybridization,
//! topology charges, bonds by row and ligand, eta coordination, linear ligands, and the
//! one-sided bonds of `nbondmat_pbc`) are ported. Later states of the xtb source extend GFN-FF
//! to Lr with refitted lanthanide parameters, a bond-length factor of lanthanide and actinide
//! halides and the removal of H atoms bonded to them; the release, the reference of this port,
//! has none of these, so they are not included. Solvation is not included, and elements beyond
//! Rn are refused.
//!
//! As in xtb, the topology (bonds, hybridization, parameters, topology charges) is set up once,
//! from the first structure evaluated, and kept while the elements and the charge stay the same
//! ([`Gfnff::refresh_topology`] sets it up anew at every structure instead). The EEQ charges of
//! the electrostatics follow the structure. The gradient and the Hessian are analytic
//! ([`GfnffTopology::hessian_with`], the `hessian` and `hessian_eeq` modules: every term, with
//! the derivatives of the coordination numbers and the response of the EEQ charges); central
//! differences of the gradient remain selectable ([`Gfnff::analytic_hessian`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's GFN-FF-like model Hessian is
//! `model_hessian::legacy::gfnff`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

// The setup keeps xtb's index loops so that it can be compared with the Fortran line by line.
#![allow(clippy::needless_range_loop)]

mod angles;
mod bonds;
pub mod constants;
mod eg;
mod eg_disp;
mod eg_hb;
pub mod elements;
mod ewald;
mod hessian;
mod hessian_core;
mod hessian_eeq;
mod hueckel;
pub mod lattice;
mod nblists;
mod neighbors;
mod periodic_cn;
mod periodic_eg;
mod rings;
mod topology;
mod torsions;

use std::sync::{Arc, Mutex};

use faer::{Col, Mat};

use crate::domain::{Element, EvalOutput, EvalRequest, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::pes::calculator::Calculator;

pub use eg::GfnffEnergy;
pub use eg_hb::{HbCutoff, HB_SWITCH_SHARE};
pub use elements::ELEMENTS;

use angles::Angle;
use bonds::BondParams;
use lattice::{Cell, Lattice};
use nblists::{BondHb, NonBonded};
use neighbors::{dist, V3};
use topology::Setup;
use torsions::Torsion;

/// The fixed GFN-FF topology of a molecule: everything `gfnff_ini` sets up.
#[derive(Debug, Clone)]
pub struct GfnffTopology {
    elements: Vec<Element>,
    charge: i32,
    pub(crate) setup: Setup,
    pub(crate) nonbonded: NonBonded,
    pub(crate) pi: Vec<bool>,
    pub(crate) pibo: Vec<f64>,
    pub(crate) bonds: Vec<BondParams>,
    pub(crate) bond_hb: BondHb,
    pub(crate) angles: Vec<Angle>,
    pub(crate) torsions: Vec<Torsion>,
    /// In a periodic cell, the fractional coordinates of the wrapped positions of the setup:
    /// each structure is taken at the images of its atoms closest to them, so that the cells
    /// of the topology stay those of the atoms however a caller wraps them.
    reference: Option<Vec<V3>>,
}

/// Bonds formed and broken relative to the topology (0-based pairs `i <= j` and the cell of
/// the image of `j`, the home cell for a molecule).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BondingChange {
    /// Bonds found now that the topology does not have.
    pub formed: Vec<(usize, usize, Cell)>,
    /// Bonds of the topology not found now.
    pub broken: Vec<(usize, usize, Cell)>,
}

impl std::fmt::Display for BondingChange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let list = |v: &[(usize, usize, Cell)]| {
            if v.is_empty() {
                "none".to_string()
            } else {
                v.iter()
                    .map(|&(i, j, c)| {
                        if c == lattice::HOME {
                            format!("{}-{}", i + 1, j + 1)
                        } else {
                            format!("{}-{}{:?}", i + 1, j + 1, c)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        write!(
            f,
            "formed {}; broken {} (atoms numbered from 1)",
            list(&self.formed),
            list(&self.broken)
        )
    }
}

/// A GFN-FF result.
#[derive(Debug, Clone)]
pub struct GfnffResult {
    /// Energy terms.
    pub energy: GfnffEnergy,
    /// Gradient (Hartree/Bohr, 3N).
    pub gradient: Col<f64>,
    /// EEQ charges of the electrostatics at this structure.
    pub charges: Vec<f64>,
    /// In a periodic cell, the stress `(1 / V) dE / d strain` (Hartree/Bohr^3, ASE's sign).
    pub stress: Option<[[f64; 3]; 3]>,
}

fn positions(sys: &MolecularSystem) -> Vec<V3> {
    let x = sys.coords_flat_bohr();
    (0..sys.n_atoms())
        .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
        .collect()
}

/// The lattice of a structure: `None` for a molecule; three-dimensional cells only, as in
/// xtb 6.7.1, which refuses one- and two-dimensional periodicity.
fn lattice_of(sys: &MolecularSystem) -> MoptResult<Option<Lattice>> {
    let Some(cell) = sys.active_cell() else {
        return Ok(None);
    };
    if cell.periodic_axes != [true; 3] {
        return Err(MoptError::unsupported(
            "GFN-FF: periodic in one or two directions is not available (xtb 6.7.1 refuses it \
             too); a three-dimensional cell with vacuum along the other directions is",
        ));
    }
    let h = cell.h_bohr;
    Ok(Some(Lattice {
        a: [0, 1, 2].map(|k| [h[0][k], h[1][k], h[2][k]]),
    }))
}

/// Fractional coordinates of `x` in `lattice`.
fn fractional(lattice: &Lattice, x: &[V3]) -> MoptResult<Vec<V3>> {
    let inv = lattice.inverse()?;
    Ok(x.iter()
        .map(|p| [0, 1, 2].map(|r| inv[r][0] * p[0] + inv[r][1] * p[1] + inv[r][2] * p[2]))
        .collect())
}

impl GfnffTopology {
    /// Set up the topology from `sys` (a molecule, or a three-dimensional periodic cell, of
    /// H to Rn). In a cell the positions are wrapped into it first, as
    /// xtb does.
    pub fn new(sys: &MolecularSystem) -> MoptResult<Self> {
        if sys.n_atoms() == 0 {
            return Err(MoptError::config("GFN-FF: no atoms"));
        }
        let lattice = lattice_of(sys)?;
        let el = sys
            .elements
            .iter()
            .map(|&e| elements::atom(e))
            .collect::<MoptResult<Vec<_>>>()?;
        let x = match &lattice {
            Some(l) => l.wrap(&positions(sys))?,
            None => positions(sys),
        };
        let reference = match &lattice {
            Some(l) => Some(fractional(l, &x)?),
            None => None,
        };
        let charge = sys.spin_charge.charge;
        let mut setup = topology::setup(el, &x, lattice, f64::from(charge))?;
        let nonbonded = nblists::nonbonded(&setup, &x);
        let (pibo, pi) = hueckel::pi_bond_orders(&setup, &x)?;
        torsions::fix_hybridization(&mut setup, &x, &pi);
        let bonds = bonds::bond_params(&setup, &pibo, &pi);
        let bond_hb = nblists::bond_hb(&setup, &x, &nonbonded);
        // Per atom pair: the last bond of the pair (bonds through different cells overwrite
        // each other, as xtb's `pbo`).
        let index: std::collections::HashMap<(usize, usize), usize> = setup
            .bonds
            .iter()
            .enumerate()
            .map(|(b, &(i, j, _))| ((i.min(j), i.max(j)), b))
            .collect();
        let pbo = |a: usize, b: usize| index.get(&(a.min(b), a.max(b))).map_or(0.0, |&m| pibo[m]);
        let angles = angles::angles(&setup, &x, &pi, &pbo);
        let torsions = torsions::torsions(&setup, &x, &pi, &pibo, &bonds, &pbo);
        Ok(Self {
            elements: sys.elements.clone(),
            charge,
            setup,
            nonbonded,
            pi,
            pibo,
            bonds,
            bond_hb,
            angles,
            torsions,
            reference,
        })
    }

    /// Whether the topology belongs to the elements and charge of `sys` (and to a periodic
    /// cell when it was set up in one).
    pub fn matches(&self, sys: &MolecularSystem) -> bool {
        self.elements == sys.elements
            && self.charge == sys.spin_charge.charge
            && self.reference.is_some() == sys.has_pbc()
    }

    /// The positions of `sys` the terms see and its lattice. In a periodic cell (in the
    /// fractional coordinates of the current lattice, which may differ from that of the
    /// setup) the atoms of every group of bonded atoms are placed along its bonds, each bond
    /// vector as at the setup, and the group as a whole at the image nearest to its position
    /// at the setup by the mean of the drifts of its atoms. A group that moved by less than
    /// half a cell has every atom at its own image nearest to its setup position, as before;
    /// one that moved farther stays whole (with every atom at its own nearest image, an atom
    /// that had moved by more than half a cell went to another image than its neighbours, and
    /// its bonds became a lattice vector long).
    fn frame_of(&self, sys: &MolecularSystem) -> MoptResult<(Vec<V3>, Option<Lattice>)> {
        let x = positions(sys);
        let lattice = lattice_of(sys)?;
        let (Some(l), Some(reference)) = (lattice.as_ref(), self.reference.as_ref()) else {
            return Ok((x, lattice));
        };
        let f = fractional(l, &x)?;
        let n = x.len();
        // How far every atom is from its setup position, in fractional coordinates.
        let drift: Vec<V3> = (0..n)
            .map(|k| [0, 1, 2].map(|c| reference[k][c] - f[k][c]))
            .collect();
        let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(j, i, _) in &self.setup.bonds {
            if i < n && j < n && i != j {
                neighbours[i].push(j);
                neighbours[j].push(i);
            }
        }
        let mut shift: Vec<Option<Cell>> = vec![None; n];
        let mut queue = std::collections::VecDeque::new();
        for root in 0..n {
            if shift[root].is_some() {
                continue;
            }
            // The group of `root` placed along its bonds, relative to `root`: the difference
            // of the shifts of two bonded atoms is the difference of their drifts, rounded.
            let mut group = vec![root];
            shift[root] = Some([0; 3]);
            queue.push_back(root);
            while let Some(i) = queue.pop_front() {
                let Some(si) = shift[i] else { continue };
                for &j in &neighbours[i] {
                    if shift[j].is_some() {
                        continue;
                    }
                    shift[j] = Some(std::array::from_fn(|c| {
                        si[c] + (drift[j][c] - drift[i][c]).round() as i32
                    }));
                    group.push(j);
                    queue.push_back(j);
                }
            }
            // The whole group at the image nearest to its setup position.
            let size = group.len() as f64;
            let offset: Cell = std::array::from_fn(|c| {
                let mean = group
                    .iter()
                    .map(|&k| drift[k][c] - f64::from(shift[k].map_or(0, |s| s[c])))
                    .sum::<f64>()
                    / size;
                mean.round() as i32
            });
            for &k in &group {
                if let Some(s) = shift[k].as_mut() {
                    for c in 0..3 {
                        s[c] += offset[c];
                    }
                }
            }
        }
        let moved = x
            .iter()
            .zip(&shift)
            .map(|(p, s)| l.image(*p, s.unwrap_or([0; 3])))
            .collect();
        Ok((moved, lattice))
    }

    /// Bonds as atom pairs (0-based; in a cell a pair may appear once per image).
    pub fn bonds(&self) -> Vec<(usize, usize)> {
        self.setup.bonds.iter().map(|&(i, j, _)| (j, i)).collect()
    }

    /// How the bonds that xtb's neighbour search finds at `sys` differ from this topology
    /// (`None` when they are the same, or when the search cannot be made).
    pub fn bonding_change(&self, sys: &MolecularSystem) -> Option<BondingChange> {
        let (x, lattice) = self.frame_of(sys).ok()?;
        self.bonding_change_at(&x, lattice.as_ref())?.0
    }

    /// [`Self::bonding_change`] at the positions `x` of the terms in `lattice`, with the
    /// margin of [`topology::bonds_with_margin`]; `None` when the search cannot be made.
    fn bonding_change_at(
        &self,
        x: &[V3],
        lattice: Option<&Lattice>,
    ) -> Option<(Option<BondingChange>, f64)> {
        let frame = lattice::Frame { x, lattice };
        let (now, margin) = topology::bonds_with_margin(
            &self.setup.el,
            &frame,
            &self.setup.qa_neighbors,
            Some(&self.setup.mchar),
        )
        .ok()?;
        let mut kept: Vec<(usize, usize, Cell)> = self
            .setup
            .bonds
            .iter()
            .map(|&(j, i, c)| topology::bond_key(j, i, c))
            .collect();
        kept.sort_unstable();
        kept.dedup();
        // Both lists are sorted.
        let formed: Vec<_> = now
            .iter()
            .copied()
            .filter(|b| kept.binary_search(b).is_err())
            .collect();
        let broken: Vec<_> = kept
            .iter()
            .copied()
            .filter(|b| now.binary_search(b).is_err())
            .collect();
        let change = if formed.is_empty() && broken.is_empty() {
            None
        } else {
            Some(BondingChange { formed, broken })
        };
        Some((change, margin))
    }

    /// Hybridization of each atom (xtb's `hyb`: 0 isolated, 1 sp, 2 sp2, 3 sp3, 5 hypervalent).
    pub fn hybridization(&self) -> &[i32] {
        &self.setup.nb.hyb
    }

    /// Topology charges `qa`.
    pub fn topology_charges(&self) -> &[f64] {
        &self.setup.qa
    }

    /// Pi atoms after the Hueckel step.
    pub fn pi_atoms(&self) -> &[bool] {
        &self.pi
    }

    /// Hueckel pi bond orders of the bonds (in the order of [`Self::bonds`]).
    pub fn pi_bond_orders(&self) -> &[f64] {
        &self.pibo
    }

    /// Numbers of bend, torsion (including out-of-plane), and bonded ATM terms.
    pub fn term_counts(&self) -> (usize, usize, usize) {
        (
            self.angles.len(),
            self.torsions.len(),
            self.nonbonded.b3list.len(),
        )
    }

    /// Energy terms and gradient of `sys` with this topology (the hydrogen and halogen bond
    /// terms switched off smoothly at their thresholds, [`HbCutoff::default`]).
    pub fn compute(&self, sys: &MolecularSystem) -> MoptResult<GfnffResult> {
        self.compute_with(sys, HbCutoff::default())
    }

    /// The analytic Hessian (`3N x 3N`, Hartree/Bohr^2) of `sys` with this topology, the
    /// derivative of the gradient of [`Self::compute_with`] with the same `hb_cutoff`.
    pub fn hessian_with(&self, sys: &MolecularSystem, hb_cutoff: HbCutoff) -> MoptResult<Mat<f64>> {
        if !self.matches(sys) {
            return Err(MoptError::config(
                "GFN-FF: the structure has other elements or another charge than the topology",
            ));
        }
        let (x, lattice) = self.frame_of(sys)?;
        hessian::hessian(self, &x, lattice.as_ref(), hb_cutoff, false)
    }

    /// The analytic Hessian of a periodic cell in the positions and the lattice vectors, the
    /// derivative of the gradient and of the strain derivative of [`Self::compute_with`]:
    /// `(3N + 9) x (3N + 9)` (Hartree/Bohr^2), the Cartesian positions (Bohr) and the three
    /// lattice vectors (Bohr; variable `3N + 3k + c` is component `c` of lattice vector `k`)
    /// as independent variables, the images of atom `j` at `x_j + sum_k n_k a_k`. A change
    /// of the lattice vectors at fixed Cartesian positions moves the images only; the
    /// derivatives at fixed fractional coordinates or of a homogeneous strain follow from it
    /// by the chain rule (the strain derivative of [`GfnffResult::stress`] is
    /// `sum_i x_ia g_ib + sum_k a_ka (dE/da_k)_b`). The Ewald splitting parameter, the images
    /// of the pairs, and the sets of translations are held, as in the stress.
    pub fn cell_hessian_with(
        &self,
        sys: &MolecularSystem,
        hb_cutoff: HbCutoff,
    ) -> MoptResult<Mat<f64>> {
        if !self.matches(sys) {
            return Err(MoptError::config(
                "GFN-FF: the structure has other elements or another charge than the topology",
            ));
        }
        let (x, lattice) = self.frame_of(sys)?;
        let Some(l) = lattice.as_ref() else {
            return Err(MoptError::config(
                "GFN-FF: the Hessian of the lattice vectors needs a periodic cell",
            ));
        };
        // The cell of the image each atom was taken at: frame_of moves atoms by whole lattice
        // vectors only.
        let given = positions(sys);
        let shifts: Vec<V3> = x
            .iter()
            .zip(&given)
            .map(|(m, p)| [0, 1, 2].map(|c| m[c] - p[c]))
            .collect();
        let cells: Vec<V3> = fractional(l, &shifts)?
            .iter()
            .map(|f| f.map(f64::round))
            .collect();
        let mut h = hessian::hessian(self, &x, lattice.as_ref(), hb_cutoff, true)?;
        // The terms saw the atoms at the images x_a + L n_a: the lattice vectors move them.
        let n = x.len();
        let dim = h.nrows();
        let moved = |h: &mut Mat<f64>, columns: bool| {
            for (a, cell) in cells.iter().enumerate() {
                for (kk, &f) in cell.iter().enumerate() {
                    if f == 0.0 {
                        continue;
                    }
                    for c in 0..3 {
                        let (to, from) = (3 * (n + kk) + c, 3 * a + c);
                        for other in 0..dim {
                            if columns {
                                let v = h[(other, from)];
                                h[(other, to)] += f * v;
                            } else {
                                let v = h[(from, other)];
                                h[(to, other)] += f * v;
                            }
                        }
                    }
                }
            }
        };
        moved(&mut h, true);
        moved(&mut h, false);
        Ok(h)
    }

    /// Energy terms and gradient of `sys` with this topology; `hb_cutoff` says how the
    /// hydrogen and halogen bond terms end at the thresholds of their lists.
    pub fn compute_with(
        &self,
        sys: &MolecularSystem,
        hb_cutoff: HbCutoff,
    ) -> MoptResult<GfnffResult> {
        if !self.matches(sys) {
            return Err(MoptError::config(
                "GFN-FF: the structure has other elements or another charge than the topology",
            ));
        }
        let (x, lattice) = self.frame_of(sys)?;
        let (energy, charges, g, sigma) =
            eg::energy_gradient(self, &x, lattice.as_ref(), hb_cutoff)?;
        let stress = match (sigma, sys.active_cell()) {
            (Some(s), Some(cell)) => {
                let v = cell.volume_bohr3();
                Some(s.map(|row| row.map(|value| value / v)))
            }
            _ => None,
        };
        Ok(GfnffResult {
            energy,
            gradient: Col::from_fn(3 * g.len(), |k| g[k / 3][k % 3]),
            charges,
            stress,
        })
    }
}

/// The GFN-FF calculator.
///
/// With the kept topology, every structure is also checked with xtb's neighbour search; when
/// the bonds found differ from the topology (a bond formed or broken, as in a reaction), a
/// warning names them ([`Calculator::take_warnings`], shown in the run log), once for each new
/// bonding pattern. The energy keeps using the first topology, as in xtb.
#[derive(Debug)]
pub struct Gfnff {
    /// The analytic Hessian (the default); `false` makes it of central differences of the
    /// analytic gradient (`-calc_opt hessian=numerical`).
    pub analytic_hessian: bool,
    /// Displacement of the central-difference Hessian (Bohr); 0 means the default 5e-3.
    pub hessian_step_bohr: f64,
    /// Set up the topology anew at every structure instead of keeping the first one.
    pub refresh_topology: bool,
    /// How the hydrogen and halogen bond terms end at the thresholds of their lists
    /// (switched off smoothly by default, so that the energy has no steps).
    pub hb_cutoff: HbCutoff,
    cache: Mutex<Option<Arc<GfnffTopology>>>,
    state: Mutex<WarningState>,
}

/// The bonding pattern reported last, the warnings not yet taken, and the structures whose
/// bonds were searched last.
#[derive(Debug, Default, Clone)]
struct WarningState {
    last: Option<BondingChange>,
    pending: Vec<String>,
    checked: std::collections::VecDeque<CheckedFrame>,
}

/// The structures whose bonding is remembered: one per image of a chain evaluated with one
/// calculator, and some to spare.
const CHECKED_FRAMES: usize = 64;

/// A structure whose bonds were searched: its positions as the terms see them, its lattice,
/// the margin of the search ([`topology::bonds_with_margin`]) and what it found. Any
/// structure whose atoms are all within half the margin of these positions (in the same
/// lattice) has the same bonds.
#[derive(Debug, Clone)]
struct CheckedFrame {
    topology: Arc<GfnffTopology>,
    x: Vec<V3>,
    lattice: Option<[V3; 3]>,
    margin: f64,
    change: Option<BondingChange>,
}

impl CheckedFrame {
    /// Whether `x` in `lattice` has the bonds of this structure: every atom moved by less than
    /// half the margin (with a little room for the rounding of the distances).
    fn covers(&self, topo: &Arc<GfnffTopology>, x: &[V3], lattice: Option<[V3; 3]>) -> bool {
        if !Arc::ptr_eq(&self.topology, topo)
            || self.lattice != lattice
            || self.x.len() != x.len()
            || self.margin.is_nan()
            || self.margin <= 0.0
        {
            return false;
        }
        let largest = self
            .x
            .iter()
            .zip(x)
            .map(|(a, b)| dist(*a, *b))
            .fold(0.0_f64, f64::max);
        2.0 * largest < self.margin * (1.0 - 1e-9) - 1e-9
    }
}

impl Default for Gfnff {
    fn default() -> Self {
        Self {
            analytic_hessian: true,
            hessian_step_bohr: 0.0,
            refresh_topology: false,
            hb_cutoff: HbCutoff::default(),
            cache: Mutex::new(None),
            state: Mutex::new(WarningState::default()),
        }
    }
}

impl Clone for Gfnff {
    fn clone(&self) -> Self {
        let cache = self.cache.lock().map(|c| c.clone()).unwrap_or(None);
        let state = self.state.lock().map(|s| s.clone()).unwrap_or_default();
        Self {
            analytic_hessian: self.analytic_hessian,
            hessian_step_bohr: self.hessian_step_bohr,
            refresh_topology: self.refresh_topology,
            hb_cutoff: self.hb_cutoff,
            cache: Mutex::new(cache),
            state: Mutex::new(state),
        }
    }
}

impl Gfnff {
    /// A calculator that sets the topology up anew at every structure.
    pub fn refreshing() -> Self {
        Self {
            refresh_topology: true,
            ..Self::default()
        }
    }

    /// The calculator with its hydrogen and halogen bond terms ending as `cutoff` says.
    #[must_use]
    pub fn with_hb_cutoff(mut self, cutoff: HbCutoff) -> Self {
        self.hb_cutoff = cutoff;
        self
    }

    /// The topology used for `sys` (the cached one when it fits).
    pub fn topology(&self, sys: &MolecularSystem) -> MoptResult<Arc<GfnffTopology>> {
        if self.refresh_topology {
            return Ok(Arc::new(GfnffTopology::new(sys)?));
        }
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| MoptError::calc("GFN-FF: topology cache poisoned"))?;
        if let Some(t) = cache.as_ref() {
            if t.matches(sys) {
                return Ok(Arc::clone(t));
            }
        }
        let t = Arc::new(GfnffTopology::new(sys)?);
        *cache = Some(Arc::clone(&t));
        Ok(t)
    }

    /// A calculator whose topology is set up from `start` (the starting structure of a run, so
    /// that a restarted run uses the same topology); when `start` cannot have one (another
    /// subsystem is evaluated, or other elements), the first structure evaluated sets it up.
    pub fn starting_from(start: &MolecularSystem) -> Self {
        let calc = Self::default();
        if let Ok(t) = GfnffTopology::new(start) {
            if let Ok(mut c) = calc.cache.lock() {
                *c = Some(Arc::new(t));
            }
        }
        calc
    }

    /// A calculator whose topology is that of the structure in `file` (its first frame, with
    /// the charge and multiplicity of `system`; `-calc_opt topology=FILE`): the structure a
    /// workflow set its topology up from, so that a later step, or a command that starts from a
    /// result of the workflow (a transition state, an IRC end), evaluates the same force field
    /// and not the one of the bonds of its own input. Refused when the file cannot be read or
    /// has other atoms than `system`, or no topology can be set up from it.
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs; xtb keeps the topology file `gfnff_topo`
    /// of a run for the same purpose).
    pub fn from_reference_file(
        file: &std::path::Path,
        system: &MolecularSystem,
    ) -> MoptResult<Self> {
        let reference =
            crate::io::xyz::read_xyz_file(file, system.spin_charge, None).map_err(|e| {
                MoptError::config(format!(
                    "gfnff: topology={}: the reference structure cannot be read ({e})",
                    file.display()
                ))
            })?;
        Self::from_reference(&reference, system)
            .map_err(|e| MoptError::config(format!("gfnff: topology={}: {e}", file.display())))
    }

    /// [`Self::from_reference_file`] with the reference structure given.
    pub fn from_reference(
        reference: &MolecularSystem,
        system: &MolecularSystem,
    ) -> MoptResult<Self> {
        if reference.elements != system.elements {
            return Err(MoptError::config(
                "the reference structure of the topology has other atoms (or another order) \
                 than the structure evaluated",
            ));
        }
        let mut reference = reference.clone();
        reference.spin_charge = system.spin_charge;
        if reference.cell.is_none() {
            reference.cell.clone_from(&system.cell);
        }
        let topology = GfnffTopology::new(&reference)?;
        let calc = Self::default();
        if let Ok(mut c) = calc.cache.lock() {
            *c = Some(Arc::new(topology));
        }
        Ok(calc)
    }

    /// Energy terms and gradient (and, with the kept topology, the bonding check).
    pub fn compute(&self, sys: &MolecularSystem) -> MoptResult<GfnffResult> {
        let topo = self.topology(sys)?;
        let result = topo.compute_with(sys, self.hb_cutoff)?;
        if !self.refresh_topology {
            self.check_bonding(&topo, sys)?;
        }
        Ok(result)
    }

    /// Record a warning when the bonding pattern at `sys` is new and differs from `topo`. The
    /// neighbour search is skipped when `sys` is within half the margin of a structure
    /// searched before (its bonds are then those found there: no distance can have crossed
    /// its threshold), so that the warnings are those of a search at every structure.
    fn check_bonding(&self, topo: &Arc<GfnffTopology>, sys: &MolecularSystem) -> MoptResult<()> {
        let poisoned = || MoptError::calc("GFN-FF: warning state poisoned");
        let change = match topo.frame_of(sys) {
            Err(_) => None,
            Ok((x, lattice)) => {
                let key = lattice.as_ref().map(|l| l.a);
                let known = {
                    let st = self.state.lock().map_err(|_| poisoned())?;
                    st.checked
                        .iter()
                        .find(|c| c.covers(topo, &x, key))
                        .map(|c| c.change.clone())
                };
                match known {
                    Some(change) => change,
                    None => match topo.bonding_change_at(&x, lattice.as_ref()) {
                        None => None,
                        Some((change, margin)) => {
                            let mut st = self.state.lock().map_err(|_| poisoned())?;
                            // A margin of -inf (metals) bounds nothing; +inf (no pairs)
                            // bounds everything.
                            if margin > 0.0 {
                                st.checked.retain(|c| Arc::ptr_eq(&c.topology, topo));
                                if st.checked.len() >= CHECKED_FRAMES {
                                    st.checked.pop_front();
                                }
                                st.checked.push_back(CheckedFrame {
                                    topology: Arc::clone(topo),
                                    x,
                                    lattice: key,
                                    margin,
                                    change: change.clone(),
                                });
                            }
                            change
                        }
                    },
                }
            }
        };
        let mut st = self.state.lock().map_err(|_| poisoned())?;
        if change == st.last {
            return Ok(());
        }
        let msg = match &change {
            Some(c) => format!(
                "GFN-FF: the bonds at this structure differ from the topology set up from the \
                 first structure ({c}); the energy still uses the first topology \
                 (-calc_opt refresh_topology=true sets it up at every structure)"
            ),
            None => "GFN-FF: the bonds match the topology of the first structure again".into(),
        };
        st.pending.push(msg);
        st.last = change;
        Ok(())
    }

    /// The Hessian: analytic ([`GfnffTopology::hessian_with`]), or central differences of the
    /// analytic gradient when `analytic_hessian` is off.
    pub fn hessian(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        if self.analytic_hessian {
            return self.topology(sys)?.hessian_with(sys, self.hb_cutoff);
        }
        self.numerical_hessian(sys)
    }

    /// The analytic Hessian of a periodic cell in the positions and the lattice vectors
    /// ([`GfnffTopology::cell_hessian_with`]).
    pub fn cell_hessian(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        self.topology(sys)?.cell_hessian_with(sys, self.hb_cutoff)
    }

    /// Hessian from central differences of the analytic gradient (same topology throughout;
    /// the displaced structures in parallel, [`crate::pes::native::fd_hessian`]).
    pub fn numerical_hessian(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        let topo = self.topology(sys)?;
        let h = if self.hessian_step_bohr > 0.0 {
            self.hessian_step_bohr
        } else {
            5.0e-3
        };
        crate::pes::native::fd_hessian::structure_hessian(sys, h, true, |at| {
            Ok(topo.compute_with(at, self.hb_cutoff)?.gradient)
        })
    }
}

impl Calculator for Gfnff {
    fn citations(&self) -> Vec<String> {
        vec!["S. Spicher, S. Grimme, Angew. Chem. Int. Ed. 2020, 59, 15665 (GFN-FF)".to_string()]
    }

    fn name(&self) -> &str {
        "gfnff"
    }

    fn description(&self) -> String {
        format!(
            "gfnff: GFN-FF built into mopt-rs {} (port of xtb 6.7.1 src/gfnff, angewChem2020_2; \
             {}); Hessian: {}",
            crate::VERSION,
            elements::ELEMENT_RANGE,
            if self.analytic_hessian {
                "analytic"
            } else {
                "central differences of the analytic gradient"
            }
        )
    }

    fn analytic_hessian(&self) -> bool {
        self.analytic_hessian
    }

    fn take_warnings(&self) -> Vec<String> {
        self.state
            .lock()
            .map(|mut s| std::mem::take(&mut s.pending))
            .unwrap_or_default()
    }

    /// Three-dimensional cells, as xtb 6.7.1's periodic GFN-FF.
    fn supports_pbc(&self) -> bool {
        true
    }

    /// The topology charges, fixed with the topology (an embedding in them has no
    /// derivatives of the charges).
    fn embedding_charges(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        Ok(Some(self.topology(sys)?.topology_charges().to_vec()))
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if req.excited_state > 0 {
            return Err(MoptError::unsupported("GFN-FF excited states"));
        }
        let r = self.compute(sys)?;
        let mut out = EvalOutput::energy_gradient(Quantity::new(r.energy.total), r.gradient);
        out.stress = r.stress;
        // The EEQ charges of the electrostatics (for dipoles along a trajectory).
        out.orbitals = Some(crate::domain::OrbitalData {
            atomic_charges: Some(r.charges),
            ..Default::default()
        });
        if req.need_hessian {
            if self.analytic_hessian && req.need_stress && sys.active_cell().is_some() {
                // With the stress of a cell: the Hessian in the positions and the lattice
                // vectors once, its block of the positions as the Hessian, and the second
                // derivatives along the strain from all of it.
                let whole = self.cell_hessian(sys)?;
                let n = sys.dof();
                out.hessian_bohr = Some(faer::Mat::from_fn(n, n, |i, j| whole[(i, j)]));
                out.strain_hessian = Some(crate::domain::StrainHessian::from_cell_hessian(
                    sys, &whole,
                )?);
            } else {
                out.hessian_bohr = Some(self.hessian(sys)?);
            }
        }
        Ok(out)
    }
}
