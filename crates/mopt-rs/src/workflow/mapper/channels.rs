//! Dissociation channels (DC) of the reaction network: sets of fragments separated beyond the
//! interaction range (`docs/source/mapper_design.md`, the nodes and when two are the same).
//!
//! - The fragments of a structure are its groups of molecules in contact ([`groups`]): two
//!   molecules (connected parts of the bond graph) are in contact when two of their atoms are
//!   closer than the sum of their van der Waals radii and a margin. A structure of two or
//!   more groups is a DC (an atom that left a cluster: the cluster and the atom).
//! - A DC node keeps the structure the fragments were found in, and for every fragment its
//!   atoms and its record in [`Network::fragments`]: the fragment optimized on its own, once
//!   for every kind of fragment (bond graph, handedness of its tetrahedral centres, charge
//!   and multiplicity) and level, which is the cache of the energies. The energy of the DC is
//!   the sum of those of its fragments. Two enantiomers are two kinds of fragment, as two
//!   mirror images are two minima, unless mirror images count as the same
//!   ([`ChannelSettings::mirror_images`]): the complexes of the two with one partner do not
//!   fall apart into one channel.
//! - Two DCs are the same when they have the same multiset of fragment records: how the
//!   fragments are oriented, and the path that led there, do not matter.
//! - A minimum of exactly two molecules in contact (a bimolecular complex) has a barrierless
//!   [`DissociationEdge`] to the DC of the two. A minimum of three or more molecules in one
//!   group has none: its channels are stepwise, not the complete separation (an atomic
//!   cluster without covalent bonds is such a group of atoms).
//! - The states of the fragments ([`fragment_states`]): the charge 0 for every fragment of a
//!   neutral structure, or the whole charge on one fragment (each in turn); every fragment at
//!   the lowest multiplicity of its number of electrons, raised by 2 for as few fragments as
//!   the spins need to couple to the multiplicity of the whole. Of several candidates the
//!   lowest sum of energies is taken.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::network::{Level, Network, NodeKind};
use crate::domain::{MolecularSystem, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::parameters::vdw::alvarez_vdw_radius_bohr;
use crate::workflow::explore::identity::BondGraph;

/// A fragment of a dissociation channel, optimized on its own.
#[derive(Debug, Clone)]
pub struct FragmentRecord {
    /// Its number (index into [`Network::fragments`]).
    pub id: usize,
    /// The colour of its bond graph with its handedness ([`fragment_key`]; equal for the
    /// same molecule).
    pub key: u64,
    /// Its formula (`H2O`).
    pub formula: String,
    /// Its charge.
    pub charge: i32,
    /// Its spin multiplicity.
    pub multiplicity: u32,
    /// The level it was computed at.
    pub level: Level,
    /// The optimized structure (its own atoms, Bohr).
    pub system: MolecularSystem,
    /// Its energy (Hartree).
    pub energy: f64,
    /// Its vibrational wavenumbers (cm^-1), where a Hessian gave them.
    pub frequencies: Option<Vec<f64>>,
}

/// One fragment of a dissociation channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelPart {
    /// Its atoms in the structure of the channel (from 0, ascending).
    pub atoms: Vec<usize>,
    /// Its record (index into [`Network::fragments`]).
    pub fragment: usize,
}

/// A barrierless dissociation of a complex (a minimum of two molecules in contact) into the
/// channel of its two molecules.
#[derive(Debug, Clone)]
pub struct DissociationEdge {
    /// Its number (index into [`Network::dissociation_edges`]).
    pub id: usize,
    /// The level it was computed at.
    pub level: Level,
    /// The complex.
    pub minimum: usize,
    /// The channel.
    pub channel: usize,
    /// `E(channel) - E(complex)` (Hartree).
    pub energy_difference: f64,
    /// The job that found it (`None`: an input).
    pub job: Option<usize>,
    /// The relaxed scan from the complex towards the channel (`confirm_dissociations`; `None`:
    /// not scanned). With a barrier on it the dissociation is not barrierless.
    pub scan: Option<super::scan::DissociationScan>,
}

/// When the molecules of a structure are a dissociation channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelSettings {
    /// Whether channels are recognized at all (`channels`).
    pub enabled: bool,
    /// The margin over the sums of the van der Waals radii (Bohr; `channel_margin` in
    /// Angstrom).
    pub margin: f64,
    /// Bonds within this factor times the sums of the covalent radii.
    pub bond_scale: f64,
    /// Whether a fragment and its mirror image are one kind of fragment (`mirror`, as for the
    /// minima).
    pub mirror_images: bool,
}

/// The default margin of the separation test (Angstrom): 1, so that hydrogen-bonded and van
/// der Waals complexes (their nearest atoms at about the sum of the radii or closer) stay
/// complexes.
pub const DEFAULT_CHANNEL_MARGIN_ANGSTROM: f64 = 1.0;

impl Default for ChannelSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            margin: DEFAULT_CHANNEL_MARGIN_ANGSTROM * crate::domain::Units::ANGSTROM_TO_BOHR,
            bond_scale: crate::workflow::explore::identity::DEFAULT_BOND_SCALE,
            mirror_images: false,
        }
    }
}

/// The molecules of `system` (connected parts of its bond graph), atoms ascending.
pub fn molecules(system: &MolecularSystem, bond_scale: f64) -> Vec<Vec<usize>> {
    BondGraph::of(system, bond_scale).molecules
}

/// The van der Waals radius of atom `i` (Bohr; 2 Angstrom for an element without one).
fn radius(system: &MolecularSystem, i: usize) -> f64 {
    alvarez_vdw_radius_bohr(system.elements[i])
        .unwrap_or(2.0 * crate::domain::Units::ANGSTROM_TO_BOHR)
}

/// The groups of the `molecules` of `system` in contact: two molecules are in one group when
/// two of their atoms are closer than the sum of their van der Waals radii and `margin`
/// (Bohr), and groups are the connected parts of that relation. The atoms of every group,
/// ascending; the groups in the order of their first atoms.
///
/// The distances are those of the positions as they are, without minimum images: a structure
/// in a periodic cell never comes here. Its callers are the channels of an isolated structure
/// ([`super::dissociation::dissociation_of`] returns `None` in a cell), the scan of a
/// barrierless edge (which only such a channel has, [`super::scan`]) and the entrance
/// structures (refused in a cell); a periodic structure is judged by [`cell_contacts`].
pub fn groups(system: &MolecularSystem, molecules: &[Vec<usize>], margin: f64) -> Vec<Vec<usize>> {
    let x = system.coords_flat_bohr();
    let near = |ma: &[usize], mb: &[usize]| {
        ma.iter().any(|&i| {
            mb.iter().any(|&j| {
                let d: f64 = (0..3).map(|k| (x[3 * i + k] - x[3 * j + k]).powi(2)).sum();
                d.sqrt() <= radius(system, i) + radius(system, j) + margin
            })
        })
    };
    // Union-find over the molecules.
    let mut root: Vec<usize> = (0..molecules.len()).collect();
    fn find(root: &mut [usize], mut a: usize) -> usize {
        while root[a] != a {
            root[a] = root[root[a]];
            a = root[a];
        }
        a
    }
    for a in 0..molecules.len() {
        for b in a + 1..molecules.len() {
            if near(&molecules[a], &molecules[b]) {
                let (ra, rb) = (find(&mut root, a), find(&mut root, b));
                root[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut slot: Vec<Option<usize>> = vec![None; molecules.len()];
    for (m, atoms) in molecules.iter().enumerate() {
        let r = find(&mut root, m);
        let k = *slot[r].get_or_insert_with(|| {
            out.push(Vec::new());
            out.len() - 1
        });
        out[k].extend_from_slice(atoms);
    }
    for g in &mut out {
        g.sort_unstable();
    }
    out.sort();
    out
}

/// Whether the `molecules` of `system` are apart in two or more groups ([`groups`]).
pub fn separated(system: &MolecularSystem, molecules: &[Vec<usize>], margin: f64) -> bool {
    molecules.len() >= 2 && groups(system, molecules, margin).len() >= 2
}

/// How far the molecules in contact of a structure in a periodic cell reach ([`cell_contacts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellContacts {
    /// The fewest periodic directions that one group of molecules in contact spans with its
    /// images (3: a crystal; 2: layers; 1: chains; 0: separate molecules or clusters).
    pub lowest: usize,
    /// The periodic directions of the cell.
    pub periodic: usize,
    /// The groups of molecules in contact (with their images) in the cell.
    pub groups: usize,
}

impl CellContacts {
    /// The structure came apart in its cell: a group of its molecules in contact spans fewer
    /// directions than the cell is periodic in.
    pub fn apart(&self) -> bool {
        self.lowest < self.periodic
    }

    /// What it came apart into, for a log.
    pub fn describe(&self) -> String {
        let kind = match self.lowest {
            0 => "separate molecules or clusters",
            1 => "chains",
            _ => "layers",
        };
        format!(
            "its molecules in contact span {} of the {} periodic directions of its cell \
             ({kind}; {} group(s))",
            self.lowest, self.periodic, self.groups
        )
    }
}

/// The rank of integer lattice vectors (as reals; at most 3).
fn lattice_rank(vectors: &[[f64; 3]]) -> usize {
    let mut basis: Vec<[f64; 3]> = Vec::new();
    for v in vectors {
        let mut w = *v;
        for b in &basis {
            let dot: f64 = (0..3).map(|k| w[k] * b[k]).sum();
            for k in 0..3 {
                w[k] -= dot * b[k];
            }
        }
        let norm = libm::sqrt(w.iter().map(|x| x * x).sum());
        if norm > 1e-6 {
            basis.push(std::array::from_fn(|k| w[k] / norm));
            if basis.len() == 3 {
                break;
            }
        }
    }
    basis.len()
}

/// The reach of the molecules of `system` in its periodic cell (`None` without one): the
/// molecules (bond graph with `bond_scale`) made whole along their bonds, two molecules (or a
/// molecule and its own image) in contact when two of their atoms are closer than the sum of
/// their van der Waals radii and `margin` (Bohr), the criterion of the groups of [`groups`];
/// a group spans as many directions as the rank of the lattice translations of the closed
/// paths of its contacts and bonds (the dimension of the network, as for the components of a
/// crystal of low dimension). A molecular crystal spans all of them; one pulled apart by
/// artificial forces into layers with vacuum between them spans two.
pub fn cell_contacts(
    system: &MolecularSystem,
    bond_scale: f64,
    margin: f64,
) -> Option<CellContacts> {
    let cell = system.active_cell()?;
    let periodic = (0..3).filter(|&k| cell.periodic_axes[k]).count();
    let n = system.n_atoms();
    if n == 0 {
        return None;
    }
    let graph = BondGraph::of(system, bond_scale);
    let x: Vec<[f64; 3]> = (0..n)
        .map(|i| system.position(crate::domain::AtomId::new(i)).value())
        .collect();
    let frac_round = |v: [f64; 3]| -> [f64; 3] {
        let f = cell.cart_to_frac(v);
        std::array::from_fn(|k| {
            if cell.periodic_axes[k] {
                libm::round(f[k])
            } else {
                0.0
            }
        })
    };
    // Whole molecules: positions along the bonds by minimum images; a bond that closes a ring
    // through the cell (a polymer, a network) gives a lattice translation of the molecule.
    let mut whole = x.clone();
    let mut seen = vec![false; n];
    let mut molecule_of = vec![0_usize; n];
    let mut own: Vec<Vec<[f64; 3]>> = vec![Vec::new(); graph.molecules.len()];
    for (m, atoms) in graph.molecules.iter().enumerate() {
        let Some(&first) = atoms.first() else {
            continue;
        };
        seen[first] = true;
        let mut queue = std::collections::VecDeque::from([first]);
        while let Some(i) = queue.pop_front() {
            molecule_of[i] = m;
            for &j in &graph.adjacency[i] {
                let d = cell.mic_delta(x[j], x[i]);
                let at: [f64; 3] = std::array::from_fn(|k| whole[i][k] + d[k]);
                if seen[j] {
                    let gap: [f64; 3] = std::array::from_fn(|k| at[k] - whole[j][k]);
                    let t = frac_round(gap);
                    if t.iter().any(|v| *v != 0.0) {
                        own[m].push(t);
                    }
                } else {
                    seen[j] = true;
                    whole[j] = at;
                    queue.push_back(j);
                }
            }
        }
    }
    // Contacts between whole molecules and the images of others (and of themselves).
    let radii: Vec<f64> = (0..n).map(|i| radius(system, i)).collect();
    let largest = radii.iter().copied().fold(0.0_f64, f64::max);
    let reach = 2.0 * largest + margin;
    let widths = cell.perpendicular_widths_bohr();
    let span: [i64; 3] = std::array::from_fn(|k| {
        if cell.periodic_axes[k] && widths[k].is_finite() {
            libm::ceil(reach / widths[k]) as i64 + 1
        } else {
            0
        }
    });
    let mut edges: Vec<(usize, usize, [f64; 3])> = Vec::new();
    for i in 0..n {
        for j in i..n {
            let base: [f64; 3] = std::array::from_fn(|k| whole[j][k] - whole[i][k]);
            let centre = frac_round(base);
            let limit = radii[i] + radii[j] + margin;
            for a in -span[0]..=span[0] {
                for b in -span[1]..=span[1] {
                    for c in -span[2]..=span[2] {
                        let t = [
                            -centre[0] + a as f64,
                            -centre[1] + b as f64,
                            -centre[2] + c as f64,
                        ];
                        let (mi, mj) = (molecule_of[i], molecule_of[j]);
                        let zero = t.iter().all(|v| *v == 0.0);
                        // An atom with itself, and two atoms of one molecule in the same image,
                        // are no contact.
                        if mi == mj && zero {
                            continue;
                        }
                        let shift = cell.frac_to_cart(t);
                        let d: f64 = (0..3)
                            .map(|k| (base[k] + shift[k]).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        if d <= limit {
                            edges.push((mi, mj, t));
                        }
                    }
                }
            }
        }
    }
    // The groups and the lattice translations of their closed paths.
    let count = graph.molecules.len();
    let mut adjacency: Vec<Vec<(usize, [f64; 3])>> = vec![Vec::new(); count];
    for &(a, b, t) in &edges {
        adjacency[a].push((b, t));
        adjacency[b].push((a, [-t[0], -t[1], -t[2]]));
    }
    let mut offset: Vec<Option<[f64; 3]>> = vec![None; count];
    let mut lowest = usize::MAX;
    let mut groups = 0;
    for start in 0..count {
        if offset[start].is_some() {
            continue;
        }
        groups += 1;
        offset[start] = Some([0.0; 3]);
        let mut cycles: Vec<[f64; 3]> = Vec::new();
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(a) = queue.pop_front() {
            cycles.extend(own[a].iter().copied());
            let oa = offset[a].unwrap_or([0.0; 3]);
            for &(b, t) in &adjacency[a] {
                let at: [f64; 3] = std::array::from_fn(|k| oa[k] + t[k]);
                match offset[b] {
                    Some(ob) => {
                        let gap: [f64; 3] = std::array::from_fn(|k| at[k] - ob[k]);
                        if gap.iter().any(|v| *v != 0.0) {
                            cycles.push(gap);
                        }
                    }
                    None => {
                        offset[b] = Some(at);
                        queue.push_back(b);
                    }
                }
            }
        }
        lowest = lowest.min(lattice_rank(&cycles));
    }
    Some(CellContacts {
        lowest: if lowest == usize::MAX { 0 } else { lowest },
        periodic,
        groups,
    })
}

/// The formula of the atoms `atoms` of `system` (Hill order: C, H, then alphabetical).
pub fn formula(system: &MolecularSystem, atoms: &[usize]) -> String {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for &i in atoms {
        *counts.entry(system.elements[i].symbol()).or_insert(0) += 1;
    }
    let mut out = String::new();
    let mut put = |symbol: &str, n: usize| {
        out.push_str(symbol);
        if n > 1 {
            out.push_str(&n.to_string());
        }
    };
    let carbon = counts.remove("C");
    if let Some(n) = carbon {
        put("C", n);
        if let Some(h) = counts.remove("H") {
            put("H", h);
        }
    }
    for (symbol, n) in counts {
        put(symbol, n);
    }
    out
}

/// The structure of the atoms `atoms` of `system` with `charge` and `multiplicity`.
pub fn subsystem(
    system: &MolecularSystem,
    atoms: &[usize],
    charge: i32,
    multiplicity: u32,
) -> MoptResult<MolecularSystem> {
    let x = system.coords_flat_bohr();
    let elements = atoms.iter().map(|&i| system.elements[i]).collect();
    let coords = faer::Col::from_fn(3 * atoms.len(), |k| x[3 * atoms[k / 3] + k % 3]);
    MolecularSystem::from_flat_bohr(
        elements,
        &coords,
        SpinCharge {
            charge,
            multiplicity,
        },
        None,
    )
}

/// The colour of the bond graph of a fragment on its own, with the configurations of its
/// double bonds ([`crate::workflow::explore::smiles::double_bond_configuration`]: the E and Z
/// isomers are two fragments) and, unless `mirror_images` are the same, with the handedness of
/// its tetrahedral centres ([`crate::workflow::explore::smiles::handedness`]): the key of a
/// fragment without such a bond or centre is the colour alone.
pub fn fragment_key(fragment: &MolecularSystem, bond_scale: f64, mirror_images: bool) -> u64 {
    let graph = BondGraph::of(fragment, bond_scale);
    let colour = if graph.molecule_colors.len() == 1 {
        graph.molecule_colors[0]
    } else {
        // A fragment that fell apart on its own: the colours of its parts together.
        let mut c = graph.molecule_colors.clone();
        c.sort_unstable();
        c.iter().fold(c.len() as u64, |h, &x| h.rotate_left(7) ^ x)
    };
    use crate::workflow::explore::smiles::{double_bond_configuration, handedness};
    let colour = match double_bond_configuration(fragment, bond_scale) {
        0 => colour,
        sides => colour.rotate_left(29) ^ sides,
    };
    if mirror_images {
        return colour;
    }
    match handedness(fragment, bond_scale) {
        0 => colour,
        hand => colour.rotate_left(17) ^ hand,
    }
}

/// The candidate charges and multiplicities of the fragments `atoms` of `system`, the most
/// likely first (see the module text). Empty when no assignment is possible.
pub fn fragment_states(system: &MolecularSystem, atoms: &[Vec<usize>]) -> Vec<Vec<(i32, u32)>> {
    let total = system.spin_charge;
    let nuclear: Vec<i64> = atoms
        .iter()
        .map(|a| {
            a.iter()
                .map(|&i| i64::from(system.elements[i].atomic_number()))
                .sum()
        })
        .collect();
    let n = atoms.len();
    let charges: Vec<Vec<i32>> = if total.charge == 0 {
        vec![vec![0; n]]
    } else {
        (0..n)
            .map(|k| {
                (0..n)
                    .map(|i| if i == k { total.charge } else { 0 })
                    .collect()
            })
            .collect()
    };
    // Twice the spin of the whole.
    let whole = i64::from(total.multiplicity) - 1;
    let mut out = Vec::new();
    for q in charges {
        let electrons: Vec<i64> = nuclear
            .iter()
            .zip(&q)
            .map(|(z, c)| z - i64::from(*c))
            .collect();
        if electrons.iter().any(|e| *e < 0) {
            continue;
        }
        let lowest: Vec<i64> = electrons.iter().map(|e| e % 2).collect();
        let reachable = |spins: &[i64]| {
            let sum: i64 = spins.iter().sum();
            let largest = spins.iter().copied().max().unwrap_or(0);
            whole <= sum && whole >= (2 * largest - sum).max(0) && (sum - whole) % 2 == 0
        };
        // Raise as few fragments as needed by 2 (one more pair of unpaired electrons each).
        let mut found = false;
        for raised in 0..=n {
            for mask in 0u32..(1u32 << n) {
                if mask.count_ones() as usize != raised {
                    continue;
                }
                let spins: Vec<i64> = (0..n)
                    .map(|i| lowest[i] + if mask & (1 << i) != 0 { 2 } else { 0 })
                    .collect();
                // A fragment cannot have more unpaired electrons than electrons.
                if spins.iter().zip(&electrons).any(|(s, e)| s > e) || !reachable(&spins) {
                    continue;
                }
                out.push(
                    (0..n)
                        .map(|i| (q[i], (spins[i] + 1) as u32))
                        .collect::<Vec<_>>(),
                );
                found = true;
            }
            if found {
                break;
            }
        }
    }
    out
}

impl Network {
    /// The record of the fragment `key` with `charge` and `multiplicity` at `level`.
    pub fn find_fragment(
        &self,
        key: u64,
        charge: i32,
        multiplicity: u32,
        level: Level,
    ) -> Option<usize> {
        self.fragments
            .iter()
            .find(|f| {
                f.key == key
                    && f.charge == charge
                    && f.multiplicity == multiplicity
                    && f.level == level
            })
            .map(|f| f.id)
    }

    /// Add `record`, or keep the known record of the same fragment: the lower energy of the
    /// two stays, and the channels with that fragment get their energies again. Returns its
    /// number.
    pub fn add_fragment(&mut self, mut record: FragmentRecord) -> usize {
        match self.find_fragment(record.key, record.charge, record.multiplicity, record.level) {
            Some(id) => {
                if record.energy < self.fragments[id].energy {
                    record.id = id;
                    self.fragments[id] = record;
                    self.refresh_channels(id);
                }
                id
            }
            None => {
                record.id = self.fragments.len();
                self.fragments.push(record);
                self.fragments.len() - 1
            }
        }
    }

    /// The sum of the energies of the fragment records `records` (NaN when one is not known,
    /// as in a damaged network file).
    pub fn channel_energy(&self, records: &[usize]) -> f64 {
        records
            .iter()
            .map(|&r| self.fragments.get(r).map_or(f64::NAN, |f| f.energy))
            .sum()
    }

    /// The channel node of `level` with the multiset of fragment records `records`.
    pub fn find_channel(&self, level: Level, records: &[usize]) -> Option<usize> {
        let mut want = records.to_vec();
        want.sort_unstable();
        self.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Dissociated && n.level == level)
            .find(|n| {
                let mut have: Vec<usize> = n.parts.iter().map(|p| p.fragment).collect();
                have.sort_unstable();
                have == want
            })
            .map(|n| n.id)
    }

    /// Add `edge` (its number is set), unless the same complex and channel are joined
    /// already; returns its number.
    pub fn add_dissociation_edge(&mut self, mut edge: DissociationEdge) -> MoptResult<usize> {
        for id in [edge.minimum, edge.channel] {
            self.node(id)?;
        }
        if self.nodes[edge.channel].kind != NodeKind::Dissociated {
            return Err(MoptError::config(format!(
                "{} is not a dissociation channel",
                self.nodes[edge.channel].label()
            )));
        }
        if let Some(known) = self
            .dissociation_edges
            .iter()
            .find(|e| e.minimum == edge.minimum && e.channel == edge.channel)
        {
            return Ok(known.id);
        }
        edge.id = self.dissociation_edges.len();
        self.dissociation_edges.push(edge);
        Ok(self.dissociation_edges.len() - 1)
    }

    /// The energies of the channels with the fragment `fragment`, and of what depends on them
    /// (the barriers of the transition-state edges that end there, the energies of the
    /// dissociations), again.
    fn refresh_channels(&mut self, fragment: usize) {
        let changed: Vec<usize> = self
            .nodes
            .iter()
            .filter(|n| n.parts.iter().any(|p| p.fragment == fragment))
            .map(|n| n.id)
            .collect();
        for &id in &changed {
            let records: Vec<usize> = self.nodes[id].parts.iter().map(|p| p.fragment).collect();
            self.nodes[id].energy = self.channel_energy(&records);
        }
        for e in &mut self.ts_edges {
            for k in 0..2 {
                if changed.contains(&e.ends[k]) {
                    e.barriers[k] = self.nodes[e.ts].energy - self.nodes[e.ends[k]].energy;
                }
            }
        }
        for e in &mut self.open_ts_edges {
            for k in 0..2 {
                if let Some(end) = e.ends[k].filter(|end| changed.contains(end)) {
                    e.barriers[k] = Some(self.nodes[e.ts].energy - self.nodes[end].energy);
                }
            }
        }
        for e in &mut self.dissociation_edges {
            if changed.contains(&e.channel) {
                e.energy_difference = self.nodes[e.channel].energy - self.nodes[e.minimum].energy;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {

    /// Water molecules (O and two H) in a cubic cell of `side` Angstrom, `per_side` of them
    /// along each periodic axis, 3.1 Angstrom apart.
    fn water_cell(per_side: [usize; 3], side: [f64; 3]) -> MolecularSystem {
        let mut text = String::new();
        let mut atoms = 0;
        for a in 0..per_side[0] {
            for b in 0..per_side[1] {
                for c in 0..per_side[2] {
                    let (x, y, z) = (3.1 * a as f64, 3.1 * b as f64, 3.1 * c as f64);
                    text.push_str(&format!(
                        "O {x} {y} {z}\nH {} {} {z}\nH {} {} {z}\n",
                        x + 0.96,
                        y,
                        x - 0.24,
                        y + 0.93
                    ));
                    atoms += 3;
                }
            }
        }
        let xyz = format!(
            "{atoms}\nLattice=\"{} 0 0 0 {} 0 0 0 {}\" pbc=\"T T T\"\n{text}",
            side[0], side[1], side[2]
        );
        crate::io::xyz::parse_xyz_str(&xyz, SpinCharge::default(), None).unwrap()
    }

    #[test]
    fn a_crystal_spans_its_cell_and_one_pulled_apart_does_not() {
        let margin = DEFAULT_CHANNEL_MARGIN_ANGSTROM * crate::domain::Units::ANGSTROM_TO_BOHR;
        let scale = crate::workflow::explore::identity::DEFAULT_BOND_SCALE;
        // Two by two by two waters 3.1 Angstrom apart in a cell of 6.2: in contact with their
        // images along every axis.
        let crystal =
            cell_contacts(&water_cell([2, 2, 2], [6.2, 6.2, 6.2]), scale, margin).unwrap();
        assert_eq!(
            (crystal.lowest, crystal.periodic, crystal.groups),
            (3, 3, 1)
        );
        assert!(!crystal.apart());
        // The same layer with 20 Angstrom of vacuum along c: layers.
        let layers =
            cell_contacts(&water_cell([2, 2, 1], [6.2, 6.2, 26.2]), scale, margin).unwrap();
        assert_eq!((layers.lowest, layers.groups), (2, 1));
        assert!(layers.apart() && layers.describe().contains("layers"));
        // A chain along a, and one molecule alone in a large cell.
        let chain =
            cell_contacts(&water_cell([2, 1, 1], [6.2, 26.2, 26.2]), scale, margin).unwrap();
        assert_eq!(chain.lowest, 1);
        let alone =
            cell_contacts(&water_cell([1, 1, 1], [26.2, 26.2, 26.2]), scale, margin).unwrap();
        assert_eq!((alone.lowest, alone.groups), (0, 1));
        // A molecule outside a cell has no such count.
        let mut molecule = water_cell([1, 1, 1], [26.2, 26.2, 26.2]);
        molecule.cell = None;
        assert!(cell_contacts(&molecule, scale, margin).is_none());
    }

    use super::*;
    use crate::domain::{positions, Angstrom, Element};

    fn water_dimer(apart: f64) -> MolecularSystem {
        let o = Element::from_atomic_number(8).unwrap();
        let h = Element::from_atomic_number(1).unwrap();
        MolecularSystem::new(
            vec![o, h, h, o, h, h],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
                [2.9 + apart, 0.0, 0.0],
                [3.2 + apart, 0.9, 0.0],
                [3.2 + apart, -0.9, 0.0],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn molecules_close_together_are_a_complex_and_apart_a_channel() {
        let s = ChannelSettings::default();
        let near = water_dimer(0.0);
        let parts = molecules(&near, s.bond_scale);
        assert_eq!(parts, vec![vec![0, 1, 2], vec![3, 4, 5]]);
        assert!(!separated(&near, &parts, s.margin));
        let far = water_dimer(4.0);
        assert!(separated(&far, &molecules(&far, s.bond_scale), s.margin));
        assert_eq!(formula(&far, &parts[0]), "H2O");
        // The same molecule, the same key, wherever it is.
        let a = subsystem(&far, &parts[0], 0, 1).unwrap();
        let b = subsystem(&far, &parts[1], 0, 1).unwrap();
        assert_eq!(
            fragment_key(&a, s.bond_scale, false),
            fragment_key(&b, s.bond_scale, false)
        );
        assert_eq!(a.n_atoms(), 3);
    }

    #[test]
    fn an_atom_that_left_a_cluster_is_one_group_and_the_cluster_another() {
        let ar = Element::from_atomic_number(18).unwrap();
        // Three argon atoms in contact (no covalent bonds: three molecules) and one far away.
        let s = MolecularSystem::new(
            vec![ar; 4],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [1.9, 3.3, 0.0],
                [20.0, 0.0, 0.0],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let settings = ChannelSettings::default();
        let parts = molecules(&s, settings.bond_scale);
        assert_eq!(parts.len(), 4);
        assert_eq!(
            groups(&s, &parts, settings.margin),
            vec![vec![0, 1, 2], vec![3]]
        );
        assert!(separated(&s, &parts, settings.margin));
        // Without the far atom: one group of three, not a channel.
        let three = subsystem(&s, &[0, 1, 2], 0, 1).unwrap();
        let parts = molecules(&three, settings.bond_scale);
        assert_eq!(groups(&three, &parts, settings.margin).len(), 1);
        assert!(!separated(&three, &parts, settings.margin));
    }

    #[test]
    fn the_states_of_the_fragments_couple_to_the_whole() {
        let far = water_dimer(4.0);
        let parts = molecules(&far, 1.2);
        // A neutral singlet: two neutral singlets, nothing else.
        assert_eq!(fragment_states(&far, &parts), vec![vec![(0, 1), (0, 1)]]);
        // A neutral triplet: one of the two raised to a triplet.
        let mut triplet = far.clone();
        triplet.spin_charge.multiplicity = 3;
        assert_eq!(
            fragment_states(&triplet, &parts),
            vec![vec![(0, 3), (0, 1)], vec![(0, 1), (0, 3)]]
        );
        // A cation doublet: the charge on either water, which is then a doublet.
        let mut cation = far.clone();
        cation.spin_charge = SpinCharge {
            charge: 1,
            multiplicity: 2,
        };
        assert_eq!(
            fragment_states(&cation, &parts),
            vec![vec![(1, 2), (0, 1)], vec![(0, 1), (1, 2)]]
        );
    }

    #[test]
    fn hill_formulas() {
        let o = Element::from_atomic_number(8).unwrap();
        let c = Element::from_atomic_number(6).unwrap();
        let h = Element::from_atomic_number(1).unwrap();
        let s = MolecularSystem::new(
            vec![o, c, h, h],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [1.8, 0.9, 0.0],
                [1.8, -0.9, 0.0],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert_eq!(formula(&s, &[0, 1, 2, 3]), "CH2O");
        assert_eq!(formula(&s, &[0, 2]), "HO");
    }

    #[test]
    fn two_enantiomers_are_two_fragments() {
        // Bromochlorofluoromethane and its mirror image; `turn` moves the atoms round.
        let chiral = |sign: f64, turn: usize| {
            let xyz = [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.33],
                [1.03, 0.0, -0.36],
                [-0.84, 1.46, -0.59],
                [-0.91, -1.58, -0.64],
            ];
            let symbols = ["C", "F", "H", "Cl", "Br"];
            let order: Vec<usize> = (0..5).map(|k| (k + turn) % 5).collect();
            MolecularSystem::new(
                order
                    .iter()
                    .map(|&k| Element::parse_token(symbols[k]).unwrap())
                    .collect(),
                &positions::<Angstrom>(
                    &order
                        .iter()
                        .map(|&k| [sign * xyz[k][0], xyz[k][1], xyz[k][2]])
                        .collect::<Vec<_>>(),
                ),
                SpinCharge::default(),
                None,
            )
            .unwrap()
        };
        let s = ChannelSettings::default();
        assert!(!s.mirror_images);
        let right = fragment_key(&chiral(1.0, 0), s.bond_scale, false);
        let left = fragment_key(&chiral(-1.0, 0), s.bond_scale, false);
        assert_ne!(right, left);
        // The key does not depend on the numbering.
        assert_eq!(fragment_key(&chiral(1.0, 2), s.bond_scale, false), right);
        assert_eq!(fragment_key(&chiral(-1.0, 3), s.bond_scale, false), left);
        // With mirror images the same, the two are one fragment: the bond graph alone.
        assert_eq!(
            fragment_key(&chiral(1.0, 0), s.bond_scale, true),
            fragment_key(&chiral(-1.0, 0), s.bond_scale, true)
        );
        // A molecule without a centre has the key of its bond graph either way, and its
        // mirror image the same.
        let water = water_dimer(0.0);
        let one = subsystem(&water, &[0, 1, 2], 0, 1).unwrap();
        assert_eq!(
            fragment_key(&one, s.bond_scale, false),
            fragment_key(&one, s.bond_scale, true)
        );
    }
}
