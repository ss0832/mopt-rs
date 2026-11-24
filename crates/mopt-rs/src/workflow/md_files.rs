//! The files of a molecular dynamics run.
//!
//! Every step: a row of `md_properties.csv` (step, time, energies, temperature, conserved
//! quantity, pressure, volume) and of MultiOptPy's `energy_profile.csv`,
//! `gradient_profile.csv`, and `tot_energy_profile.csv`. Every `write_every` steps (and at
//! the last step): a frame of the trajectory `<stem>_traj.xyz` (extended XYZ with the cell
//! for periodic structures, molecules wrapped into the cell as wholes) or `<stem>_traj.csv`
//! for plain variables, and for atoms the velocities of the same frames in
//! `<stem>_velocities.xyz` (Angstrom per fs, in the layout of the coordinates; for the
//! velocity autocorrelation of the trajectory analysis). At the end: `energy_profile_kcalmol.csv`, `maximum_value.txt`,
//! `minimum_value.txt`, `local_min_grad.txt`, the figures `energy_plot_.png`,
//! `plot_gradient_.png`, `plot_temperature_.png`, `plot_tot_energy_.png`,
//! `plot_conserved_.png` (and `plot_pressure_.png`, `plot_volume_.png` with a cell), the
//! last structure `<stem>_final.xyz` (or `.txt`) and its velocities `final_velocities.txt`.
//! The files are written anew from step 0 on, so that a restarted run that replays its
//! journal leaves the same files.
//!
//! MultiOptPy: `moleculardynamics.py:MD.md` (the profile files, `FIO.make_traj_file`,
//! `FIO.argrelextrema_txt_save`, `Graph.double_plot`, `Graph.single_plot`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use super::md::MdObserver;
use super::observers::relative_extrema;
use crate::coord::graph::{whole_molecules, MolecularGraph};
use crate::core_infra::viz::{save_single_line_plot, PlotFigure, PlotSeries, PlotStyle, Rgb};
use crate::domain::{MolecularSystem, Units};
use crate::error::MoptResult;
use crate::md::barostat::gigapascal_per_atomic_unit;
use crate::md::{MdEnd, MdStepReport};

/// What the variables of a run are.
#[derive(Debug, Clone)]
pub enum MdLayout {
    /// The coordinates of the atoms of a structure like this one.
    Atoms(Box<MolecularSystem>),
    /// Plain variables with these names.
    Variables(Vec<String>),
}

/// Largest number of points drawn in a figure (the series are thinned evenly above it).
const FIGURE_POINTS: usize = 4000;

/// The files of a run.
#[derive(Debug)]
pub struct MdFiles {
    dir: PathBuf,
    stem: String,
    layout: MdLayout,
    write_every: usize,
    selected_atoms: Option<Vec<crate::domain::AtomId>>,
    /// The bonded fragments of a periodic structure: their atoms and their bond graphs (atoms
    /// numbered within the fragment).
    fragments: Option<Vec<(Vec<crate::domain::AtomId>, MolecularGraph)>>,
    times: Vec<f64>,
    raw: Vec<f64>,
    biased: Vec<f64>,
    total: Vec<f64>,
    conserved: Vec<f64>,
    temperature: Vec<f64>,
    gradient: Vec<f64>,
    pressure: Vec<f64>,
    volume: Vec<f64>,
    hopping: super::md_hopping_output::HoppingSeries,
    last: Option<MdStepReport>,
}

fn append(path: &Path, text: &str) -> MoptResult<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(text.as_bytes())?;
    Ok(())
}

fn rms(v: &faer::Col<f64>) -> f64 {
    if v.nrows() == 0 {
        return 0.0;
    }
    libm::sqrt(v.iter().map(|x| x * x).sum::<f64>() / v.nrows() as f64)
}

impl MdFiles {
    /// Files for input `stem` in `dir`; a trajectory frame every `write_every` steps.
    pub fn new(dir: &Path, stem: &str, layout: MdLayout, write_every: usize) -> MoptResult<Self> {
        fs::create_dir_all(dir)?;
        let fragments = match &layout {
            MdLayout::Atoms(sys) if sys.active_cell().is_some() => Some(fragment_graphs(sys)?),
            _ => None,
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            stem: stem.to_string(),
            layout,
            write_every: write_every.max(1),
            selected_atoms: None,
            fragments,
            times: Vec::new(),
            raw: Vec::new(),
            biased: Vec::new(),
            total: Vec::new(),
            conserved: Vec::new(),
            temperature: Vec::new(),
            gradient: Vec::new(),
            pressure: Vec::new(),
            volume: Vec::new(),
            hopping: Default::default(),
            last: None,
        })
    }

    /// Select corresponding atoms in the position and velocity trajectories.
    pub fn select_trajectory_atoms(&mut self, atoms: Option<Vec<crate::domain::AtomId>>) {
        self.selected_atoms = atoms;
    }

    /// The table of the electronic states of surface hopping (`<stem>_states.csv`), when
    /// the run has them.
    pub fn states_path(&self) -> Option<PathBuf> {
        let path = self.dir.join(format!("{}_states.csv", self.stem));
        (!self.hopping.is_empty()).then_some(path)
    }

    /// The trajectory file.
    pub fn trajectory_path(&self) -> PathBuf {
        let ext = match self.layout {
            MdLayout::Atoms(_) => "xyz",
            MdLayout::Variables(_) => "csv",
        };
        self.dir.join(format!("{}_traj.{ext}", self.stem))
    }

    /// The velocities of the frames of the trajectory (atoms only).
    pub fn velocities_path(&self) -> Option<PathBuf> {
        match self.layout {
            MdLayout::Atoms(_) => Some(self.dir.join(format!("{}_velocities.xyz", self.stem))),
            MdLayout::Variables(_) => None,
        }
    }

    /// The file of the last structure.
    pub fn final_path(&self) -> PathBuf {
        let ext = match self.layout {
            MdLayout::Atoms(_) => "xyz",
            MdLayout::Variables(_) => "txt",
        };
        self.dir.join(format!("{}_final.{ext}", self.stem))
    }

    fn start_files(&self) -> MoptResult<()> {
        fs::write(
            self.dir.join("md_properties.csv"),
            "step,time_fs,potential_energy,potential_energy_without_bias,kinetic_energy,\
             total_energy,conserved_quantity,temperature_K,target_temperature_K,pressure_GPa,\
             volume_angstrom3,heat_exchanged_hartree\n",
        )?;
        for name in ["dipoles", "pressure_tensor", "states", "hops"] {
            let _ = fs::remove_file(self.dir.join(format!("{}_{name}.csv", self.stem)));
        }
        fs::write(self.dir.join("energy_profile.csv"), "energy [hartree] \n")?;
        fs::write(
            self.dir.join("gradient_profile.csv"),
            "gradient (RMS) [hartree/Bohr] \n",
        )?;
        fs::write(
            self.dir.join("tot_energy_profile.csv"),
            "total energy [hartree] \n",
        )?;
        let header = match &self.layout {
            MdLayout::Atoms(_) => String::new(),
            MdLayout::Variables(names) => format!("step,time,{}\n", names.join(",")),
        };
        fs::write(self.trajectory_path(), header)?;
        if let Some(path) = self.velocities_path() {
            fs::write(path, "")?;
        }
        Ok(())
    }

    /// The series of every step for the correlation functions: the dipole
    /// (`<stem>_dipoles.csv`, e Bohr) when the calculator gives charges, and the pressure
    /// tensor `P = sum_i m_i v_i v_i^T / V - sigma` (`<stem>_pressure_tensor.csv`, Hartree /
    /// Bohr^3) when the stress is known in a cell.
    fn write_series(&self, r: &MdStepReport) -> MoptResult<()> {
        if let Some(mu) = r.dipole {
            let path = self.dir.join(format!("{}_dipoles.csv", self.stem));
            if !path.exists() {
                fs::write(&path, "step,time_fs,mu_x,mu_y,mu_z\n")?;
            }
            append(
                &path,
                &format!("{},{},{},{},{}\n", r.step, r.time_fs(), mu[0], mu[1], mu[2]),
            )?;
        }
        if let (Some(s), Some(v), MdLayout::Atoms(t)) = (r.stress, r.volume, &self.layout) {
            let path = self.dir.join(format!("{}_pressure_tensor.csv", self.stem));
            if !path.exists() {
                fs::write(
                    &path,
                    "step,time_fs,P_xx,P_yy,P_zz,P_xy,P_xz,P_yz,volume_bohr3\n",
                )?;
            }
            let masses: Vec<f64> = t
                .masses_amu()
                .iter()
                .map(|m| m * Units::AMU_TO_AU_MASS)
                .collect();
            let p = |a: usize, b: usize| {
                let k: f64 = masses
                    .iter()
                    .enumerate()
                    .map(|(i, m)| m * r.v[3 * i + a] * r.v[3 * i + b])
                    .sum();
                k / v - s[a][b]
            };
            append(
                &path,
                &format!(
                    "{},{},{},{},{},{},{},{},{v}\n",
                    r.step,
                    r.time_fs(),
                    p(0, 0),
                    p(1, 1),
                    p(2, 2),
                    p(0, 1),
                    p(0, 2),
                    p(1, 2)
                ),
            )?;
        }
        Ok(())
    }

    /// The electronic states of surface hopping at every step (`<stem>_states.csv`: the
    /// active state and the energies of all of them, Hartree) and the candidates for hops
    /// (`<stem>_hops.csv`).
    fn write_hopping(&self, r: &MdStepReport) -> MoptResult<()> {
        let Some(active) = r.electronic_state else {
            return Ok(());
        };
        let path = self.dir.join(format!("{}_states.csv", self.stem));
        if !path.exists() {
            let names: Vec<String> = r
                .state_energies
                .iter()
                .map(|(s, _)| format!("energy_state_{s}_hartree"))
                .collect();
            fs::write(
                &path,
                format!("step,time_fs,active_state,{}\n", names.join(",")),
            )?;
        }
        let energies: Vec<String> = r
            .state_energies
            .iter()
            .map(|(_, e)| format!("{e}"))
            .collect();
        append(
            &path,
            &format!(
                "{},{},{active},{}\n",
                r.step,
                r.time_fs(),
                energies.join(",")
            ),
        )?;
        let path = self.dir.join(format!("{}_hops.csv", self.stem));
        if !path.exists() {
            fs::write(
                &path,
                "step_of_minimum,time_fs,decided_at_step,from_state,to_state,gap_ev,crossing,\
                 allowed,a2,b2,kinetic_energy_along_ev,probability,transmission,random,outcome\n",
            )?;
        }
        let opt = |v: Option<f64>| v.map_or_else(String::new, |x| format!("{x}"));
        let ev = Units::HARTREE_TO_EV;
        for h in &r.hops {
            let zn = h.zn.as_ref();
            append(
                &path,
                &format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    h.step,
                    h.time * Units::AU_TIME_TO_FS,
                    h.decided,
                    h.from,
                    h.to,
                    h.gap * ev,
                    zn.map_or("", |z| z.kind.name()),
                    zn.map_or_else(String::new, |z| z.allowed.to_string()),
                    opt(zn.map(|z| z.a2)),
                    opt(zn.map(|z| z.b2)),
                    h.kinetic_along * ev,
                    opt(zn.and_then(|z| z.probability)),
                    opt(zn.and_then(|z| z.transmission)),
                    opt(h.random),
                    h.outcome.name()
                ),
            )?;
        }
        Ok(())
    }

    /// Append the frame of `r` to the trajectory (and its velocities).
    fn write_frame(&self, r: &MdStepReport) -> MoptResult<()> {
        append(
            &self.trajectory_path(),
            &self.frame(r, self.selected_atoms.as_deref())?,
        )?;
        if let (Some(path), MdLayout::Atoms(template)) = (self.velocities_path(), &self.layout) {
            let scale = Units::BOHR_TO_ANGSTROM / Units::AU_TIME_TO_FS;
            let mut text = format!(
                "{}\nFrame {} time={:.6} fs velocities (angstrom/fs)\n",
                self.selected_atoms
                    .as_ref()
                    .map_or(template.n_atoms(), Vec::len),
                r.step,
                r.time_fs()
            );
            let ids = self
                .selected_atoms
                .clone()
                .unwrap_or_else(|| template.atoms().collect());
            for atom in ids {
                let i = atom.index();
                let e = template.element(atom);
                text.push_str(&crate::io::xyz::multioptpy_row(
                    e.symbol(),
                    "  ",
                    [
                        r.v[3 * i] * scale,
                        r.v[3 * i + 1] * scale,
                        r.v[3 * i + 2] * scale,
                    ],
                ));
            }
            append(&path, &text)?;
        }
        Ok(())
    }

    fn frame(
        &self,
        r: &MdStepReport,
        atoms: Option<&[crate::domain::AtomId]>,
    ) -> MoptResult<String> {
        match &self.layout {
            MdLayout::Variables(_) => {
                let values: Vec<String> = r.x.iter().map(|v| format!("{v:.12e}")).collect();
                Ok(format!("{},{},{}\n", r.step, r.time, values.join(",")))
            }
            MdLayout::Atoms(template) => {
                let mut sys = template.with_flat_coords_bohr(&r.x)?;
                if r.cell.is_some() {
                    sys.cell.clone_from(&r.cell);
                }
                let sys = self.wrapped(sys)?;
                let sys = match atoms {
                    Some(ids) => sys.trajectory_atoms(ids)?,
                    None => sys,
                };
                let lattice = sys
                    .active_cell()
                    .map_or_else(String::new, |c| format!(" {}", c.to_extxyz_header()));
                let state = r
                    .electronic_state
                    .map_or_else(String::new, |s| format!(" state={s}"));
                let mut text = format!(
                    "{}\nFrame {} time={:.6} fs energy={:.12}{state}{lattice}\n",
                    sys.n_atoms(),
                    r.step,
                    r.time_fs(),
                    r.energy
                );
                text.push_str(&coordinate_lines(&sys));
                Ok(text)
            }
        }
    }

    /// Molecules moved into the cell as wholes: each fragment of bonded atoms is made whole
    /// along its bonds (`coord::graph::whole_molecules`, a breadth-first search over the bond
    /// graph, so a molecule longer than half the cell stays whole) and shifted so that its
    /// centre lies in the cell; a fragment whose bonds run through the boundaries into an
    /// infinite network (a crystal, a polymer) is wrapped atom by atom.
    fn wrapped(&self, mut sys: MolecularSystem) -> MoptResult<MolecularSystem> {
        let (Some(fragments), Some(cell)) = (&self.fragments, sys.active_cell().cloned()) else {
            return Ok(sys);
        };
        for (atoms, graph) in fragments {
            let alone = sys.trajectory_atoms(atoms)?;
            let Ok(whole) = whole_molecules(&alone, graph) else {
                for &a in atoms {
                    let p = cell.wrap_point(sys.position(a).value());
                    sys.set_position(a, crate::domain::Position::new(p));
                }
                continue;
            };
            let joined: Vec<[f64; 3]> = whole.atoms().map(|a| whole.position(a).value()).collect();
            let n = joined.len() as f64;
            let centre: [f64; 3] =
                std::array::from_fn(|c| joined.iter().map(|p| p[c]).sum::<f64>() / n);
            let inside = cell.wrap_point(centre);
            for (&a, p) in atoms.iter().zip(&joined) {
                let q: [f64; 3] = std::array::from_fn(|c| p[c] - centre[c] + inside[c]);
                sys.set_position(a, crate::domain::Position::new(q));
            }
        }
        Ok(sys)
    }

    fn write_profiles(&self) -> MoptResult<()> {
        let first = self.raw.first().copied().unwrap_or(0.0);
        let mut kcal = String::from("STEP,energy[kcal/mol]\n");
        for (i, e) in self.raw.iter().enumerate() {
            let _ = writeln!(kcal, "{i},{}", (e - first) * Units::HARTREE_TO_KCAL_MOL);
        }
        fs::write(self.dir.join("energy_profile_kcalmol.csv"), kcal)?;
        for (name, values, max) in [
            ("maximum_value", &self.raw, true),
            ("minimum_value", &self.raw, false),
            ("local_min_grad", &self.gradient, false),
        ] {
            let text: String = relative_extrema(values, max)
                .into_iter()
                .map(|i| format!("{i}\n"))
                .collect();
            fs::write(self.dir.join(format!("{name}.txt")), text)?;
        }
        Ok(())
    }

    fn write_figures(&self) -> MoptResult<()> {
        let stride = self.times.len().div_ceil(FIGURE_POINTS).max(1);
        let thin = |v: &[f64]| -> Vec<f64> { v.iter().step_by(stride).copied().collect() };
        let xs = thin(&self.times);
        let kcal = |v: &[f64]| -> Vec<f64> {
            thin(v)
                .iter()
                .map(|e| e * Units::HARTREE_TO_KCAL_MOL)
                .collect()
        };
        let mut fig = PlotFigure::new("Energy Profile", "time (fs)", "energy [kcal/mol]");
        for (label, values, color) in [
            ("energy", &self.raw, Rgb::EMERALD),
            ("energy with bias", &self.biased, Rgb::BLUE),
        ] {
            fig.add_series(PlotSeries {
                label: label.into(),
                xs: xs.clone(),
                ys: kcal(values),
                color,
                style: PlotStyle::LinePoints,
            });
        }
        fig.save(self.dir.join("energy_plot_.png"))?;
        if let Some(fig) = self.hopping.figure(FIGURE_POINTS) {
            fig.save(self.dir.join("plot_surface_hopping_.png"))?;
        }
        let mut singles = vec![
            ("gradient", "gradient (RMS) [a.u.]", &self.gradient),
            ("temperature", "temperature [K]", &self.temperature),
            ("tot_energy", "total energy [a.u.]", &self.total),
            ("conserved", "conserved quantity [a.u.]", &self.conserved),
        ];
        if !self.volume.is_empty() {
            singles.push(("volume", "volume [ang.^3]", &self.volume));
        }
        if !self.pressure.is_empty() {
            singles.push(("pressure", "pressure [GPa]", &self.pressure));
        }
        for (name, axis, values) in singles {
            let ys = thin(values);
            let n = ys.len().min(xs.len());
            save_single_line_plot(
                self.dir.join(format!("plot_{name}_.png")),
                name,
                "time (fs)",
                axis,
                &xs[..n],
                &ys[..n],
            )?;
        }
        Ok(())
    }

    fn write_final(&self, r: &MdStepReport) -> MoptResult<()> {
        match &self.layout {
            MdLayout::Variables(names) => {
                let values: Vec<f64> = r.x.iter().copied().collect();
                let text =
                    super::super::cli::variables::format_variables(names, &values, false, r.energy);
                fs::write(self.final_path(), text)?;
            }
            MdLayout::Atoms(_) => {
                let frame = self.frame(r, None)?;
                fs::write(self.final_path(), frame)?;
            }
        }
        let mut v =
            String::from("# velocities (Bohr per atomic unit of time), one row per particle\n");
        let group = match self.layout {
            MdLayout::Atoms(_) => 3,
            MdLayout::Variables(_) => 1,
        };
        for chunk in r.v.iter().copied().collect::<Vec<f64>>().chunks(group) {
            let row: Vec<String> = chunk.iter().map(|x| format!("{x:.12e}")).collect();
            let _ = writeln!(v, "{}", row.join(" "));
        }
        fs::write(self.dir.join("final_velocities.txt"), v)?;
        Ok(())
    }
}

/// The bonded fragments of `sys` (bonds within 1.2 times the sum of the covalent radii), each
/// with its atoms and its own bond graph.
fn fragment_graphs(
    sys: &MolecularSystem,
) -> MoptResult<Vec<(Vec<crate::domain::AtomId>, MolecularGraph)>> {
    let graph = MolecularGraph::from_system(sys, 1.2);
    graph
        .fragments()
        .into_iter()
        .map(|members| {
            let mut local = vec![usize::MAX; sys.n_atoms()];
            for (k, &i) in members.iter().enumerate() {
                local[i] = k;
            }
            let bonds: Vec<[usize; 2]> = members
                .iter()
                .flat_map(|&i| {
                    let local = &local;
                    graph
                        .neighbors(i)
                        .iter()
                        .filter(move |&&j| i < j)
                        .map(move |&j| [local[i], local[j]])
                })
                .collect();
            let numbers = members
                .iter()
                .map(|&i| sys.elements[i].atomic_number())
                .collect();
            let atoms = members
                .iter()
                .map(|&i| crate::domain::AtomId::new(i))
                .collect();
            Ok((atoms, MolecularGraph::from_bonds(numbers, &bonds)?))
        })
        .collect()
}

/// MultiOptPy's coordinate lines in Angstrom.
fn coordinate_lines(sys: &MolecularSystem) -> String {
    crate::io::xyz::multioptpy_rows(sys, "  ")
}

impl MdObserver for MdFiles {
    fn on_step(&mut self, r: &MdStepReport) -> MoptResult<()> {
        if r.step == 0 {
            self.start_files()?;
        }
        let a3 = Units::BOHR_TO_ANGSTROM.powi(3);
        let gpa = gigapascal_per_atomic_unit();
        let opt =
            |v: Option<f64>, scale: f64| v.map_or_else(String::new, |x| format!("{}", x * scale));
        append(
            &self.dir.join("md_properties.csv"),
            &format!(
                "{},{},{},{},{},{},{},{},{},{},{},{}\n",
                r.step,
                r.time_fs(),
                r.energy,
                r.raw_energy,
                r.kinetic,
                r.total_energy(),
                r.conserved,
                r.temperature,
                r.target_temperature,
                opt(r.pressure, gpa),
                opt(r.volume, a3),
                r.heat_exchanged
            ),
        )?;
        self.write_series(r)?;
        self.write_hopping(r)?;
        append(
            &self.dir.join("energy_profile.csv"),
            &format!("{}\n", r.raw_energy),
        )?;
        let g = rms(&r.raw_gradient);
        append(&self.dir.join("gradient_profile.csv"), &format!("{g}\n"))?;
        append(
            &self.dir.join("tot_energy_profile.csv"),
            &format!("{}\n", r.total_energy()),
        )?;
        if r.step.is_multiple_of(self.write_every) {
            self.write_frame(r)?;
        }
        self.times.push(r.time_fs());
        self.raw.push(r.raw_energy);
        self.biased.push(r.energy);
        self.total.push(r.total_energy());
        self.conserved.push(r.conserved);
        self.temperature.push(r.temperature);
        self.gradient.push(g);
        if let Some(p) = r.pressure {
            self.pressure.push(p * gpa);
        }
        if let Some(v) = r.volume {
            self.volume.push(v * a3);
        }
        self.hopping.push(r);
        self.last = Some(r.clone());
        Ok(())
    }

    fn on_finish(&mut self, _end: Option<MdEnd>) -> MoptResult<()> {
        let Some(last) = self.last.clone() else {
            return Ok(());
        };
        if !last.step.is_multiple_of(self.write_every) {
            self.write_frame(&last)?;
        }
        self.write_profiles()?;
        self.write_figures()?;
        self.write_final(&last)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, PeriodicCell, SpinCharge};

    #[test]
    fn a_chain_longer_than_half_the_cell_is_written_whole() {
        // Eight carbons 1.4 Angstrom apart along x (9.8 Angstrom) in a cube of 12 Angstrom,
        // starting at x = 8: the chain crosses the boundary, and its ends (2.2 Angstrom apart
        // through it) are not bonded. With a water molecule split by the boundary in y.
        let mut xyz: Vec<[f64; 3]> = (0..8)
            .map(|k| [(8.0 + 1.4 * k as f64) % 12.0, 6.0, 6.0])
            .collect();
        xyz.extend([[3.0, 11.8, 3.0], [3.0, 0.55, 3.0], [3.9, 11.6, 3.0]]);
        let mut elements = vec![Element::C; 8];
        elements.extend([Element::O, Element::H, Element::H]);
        let cell = PeriodicCell::parse_cli_spec("12,12,12").unwrap();
        let sys = MolecularSystem::new(
            elements,
            &positions::<Angstrom>(&xyz),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("mopt_md_files_whole_{}", std::process::id()));
        let files = MdFiles::new(&dir, "chain", MdLayout::Atoms(Box::new(sys.clone())), 1).unwrap();
        let out = files.wrapped(sys).unwrap();
        let at = |i: usize| out.position(crate::domain::AtomId::new(i)).value();
        let bohr = Units::ANGSTROM_TO_BOHR;
        let length = |i: usize, j: usize| {
            let (p, q) = (at(i), at(j));
            ((0..3).map(|c| (p[c] - q[c]).powi(2)).sum::<f64>()).sqrt() / bohr
        };
        // Bonded neighbours are next to each other without an image in between.
        for k in 0..7 {
            assert!(
                (length(k, k + 1) - 1.4).abs() < 1e-9,
                "{k}: {}",
                length(k, k + 1)
            );
        }
        assert!((length(8, 9) - 0.75).abs() < 1e-9 && length(8, 10) < 1.2);
        // The centre of the chain is in the cell.
        let centre = (0..8).map(|k| at(k)[0]).sum::<f64>() / 8.0 / bohr;
        assert!((0.0..12.0).contains(&centre), "{centre}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
