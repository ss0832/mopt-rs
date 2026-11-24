//! The files of a chain relaxation, written after every iteration: MultiOptPy's files
//! under MultiOptPy's names, so that what reads the output of its `nebmain` reads this.
//!
//! In the run directory, one line per iteration with one number per image:
//!
//! - `energy_plot.csv`: energies (Hartree; with the bias potentials: the energies the chain
//!   is relaxed on, and what the maxima, the candidates for transition states, and the
//!   optimizations after the chain are chosen by);
//! - `energy_plot_without_bias.csv` (mopt-rs; only with bias potentials): the energies of
//!   the images without them (`nan` for a value answered by a model, which has none);
//! - `path_length.csv`: lengths along the chain up to every image (Bohr);
//! - `bias_force_rms.csv`: root mean square of the gradient of every image;
//! - `orthogonality.csv`: cosine between force and gradient of every image;
//! - `perp_rms_gradient.csv`, `perp_max_gradient.csv`: root mean square and largest
//!   component of the force of every image.
//!
//! In `path_ITR_<iteration>_<name>/` for every iteration, the chain that was evaluated:
//!
//! - `<name>_<image>.xyz` (structures; Angstrom, the comment line has charge and
//!   multiplicity) and `<name>_path.xyz` with all of them, or, for chains of variables,
//!   `<name>_path.csv` with one line per image;
//! - `approx_TS_node.txt`, `approx_EQ_node.txt`: the images at maxima and minima of the
//!   energy profile; `local_min_bias_grad_node.txt`: those at minima of the gradient.
//!
//! After the last iteration, [`ChainFiles::candidates`] names the files of the images at
//! the maxima of the last profile (the highest inner image if there is no maximum), which
//! MultiOptPy lists as candidates for transition states.
//!
//! MultiOptPy: `neb.py:NEB._save_analysis_data`, `make_input_files`, `make_traj_file`,
//! `get_result_file_path`; `fileio.py:FileIO.argrelextrema_txt_save`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use faer::Col;

use super::chain::{ChainObserver, Flow};
use crate::core_infra::log::python_float;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::path::band::machine::{ChainRelaxation, ChainReport};
use crate::path::band::{strict_maxima, strict_minima};

/// The per-iteration files in the run directory.
const TABLES: [&str; 6] = [
    "energy_plot.csv",
    "path_length.csv",
    "bias_force_rms.csv",
    "orthogonality.csv",
    "perp_rms_gradient.csv",
    "perp_max_gradient.csv",
];

/// The energies of the images without the bias potentials (written when there are any).
pub const BARE_TABLE: &str = "energy_plot_without_bias.csv";

/// What the points of the chain are, for their files.
#[derive(Debug, Clone)]
pub enum ChainPoints {
    /// Structures with the atoms, charge, and multiplicity of this one.
    Structures(Box<MolecularSystem>),
    /// Plain variables with these names.
    Variables(Vec<String>),
}

#[derive(Debug)]
enum FilePoints {
    Standard(ChainPoints),
    VariableCell(Box<crate::opt::cell::VariableCell>),
}

/// Observer that writes the files.
#[derive(Debug)]
pub struct ChainFiles {
    directory: PathBuf,
    name: String,
    points: FilePoints,
    atoms: Option<Vec<crate::domain::AtomId>>,
    /// Directory and energies of the last iteration written.
    last: Option<(PathBuf, Vec<f64>)>,
}

/// The files of the candidates for transition states of a finished relaxation.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainCandidates {
    /// Images at the maxima of the last energy profile (the highest inner image if the
    /// profile has no maximum).
    pub images: Vec<usize>,
    /// Their files (structures only).
    pub files: Vec<PathBuf>,
    /// The file with the whole last chain.
    pub path: PathBuf,
}

impl ChainFiles {
    /// Files in `directory` for the chain called `name` (the input without its
    /// extension). Tables of an earlier run in the directory are started anew.
    pub fn new(directory: impl AsRef<Path>, name: &str, points: ChainPoints) -> MoptResult<Self> {
        Self::open(directory.as_ref(), name, FilePoints::Standard(points))
    }

    /// Write the lattice and Cartesian positions of each variable-cell image.
    pub fn variable_cell(
        directory: &Path,
        name: &str,
        map: &crate::opt::cell::VariableCell,
    ) -> MoptResult<Self> {
        Self::open(
            directory,
            name,
            FilePoints::VariableCell(Box::new(map.clone())),
        )
    }

    fn open(directory: &Path, name: &str, points: FilePoints) -> MoptResult<Self> {
        let directory = directory.to_path_buf();
        std::fs::create_dir_all(&directory)?;
        for table in TABLES.iter().chain([&BARE_TABLE]) {
            let path = directory.join(table);
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
        }
        Ok(Self {
            directory,
            name: name.to_string(),
            points,
            atoms: None,
            last: None,
        })
    }

    /// Select the atoms written to the XYZ files without changing the chain being optimized.
    pub fn with_atoms(mut self, atoms: Option<&[crate::domain::AtomId]>) -> Self {
        self.atoms = atoms.map(<[_]>::to_vec);
        self
    }

    fn append(&self, table: &str, values: &[f64]) -> MoptResult<()> {
        use std::io::Write as _;
        let line: Vec<String> = values.iter().map(|v| python_float(*v)).collect();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.directory.join(table))?;
        writeln!(file, "{}", line.join(","))?;
        Ok(())
    }

    /// The directory of `iteration`.
    pub fn iteration_directory(&self, iteration: usize) -> PathBuf {
        self.directory
            .join(format!("path_ITR_{iteration}_{}", self.name))
    }

    fn write_chain(&self, directory: &Path, chain: &[Col<f64>]) -> MoptResult<()> {
        match &self.points {
            FilePoints::Standard(ChainPoints::Structures(template)) => {
                // Charge and multiplicity as MultiOptPy writes them; in a periodic cell
                // the lattice (extended XYZ), without which the files could not be read
                // as what they are.
                let lattice = template.active_cell().map(|cell| {
                    format!("{} Properties=species:S:1:pos:R:3", cell.to_extxyz_header())
                });
                let state = lattice.clone().unwrap_or_else(|| {
                    format!(
                        "{} {}",
                        template.spin_charge.charge, template.spin_charge.multiplicity
                    )
                });
                let mut all = String::new();
                for (image, point) in chain.iter().enumerate() {
                    if point.nrows() != template.dof() {
                        return Err(MoptError::config(format!(
                            "an image of {} coordinates for {} atoms",
                            point.nrows(),
                            template.n_atoms()
                        )));
                    }
                    let count = self.atoms.as_ref().map_or(template.n_atoms(), Vec::len);
                    let mut one = format!("{count}\n{state}\n");
                    let frame = lattice.clone().unwrap_or_else(|| format!("Frame {image}"));
                    let _ = writeln!(all, "{count}\n{frame}");
                    for index in 0..count {
                        let atom = self.atoms.as_ref().map_or(index, |ids| ids[index].index());
                        let element = template.elements.get(atom).ok_or_else(|| {
                            MoptError::config("trajectory atom outside the chain image")
                        })?;
                        let at = |k: usize| point[3 * atom + k] * Units::BOHR_TO_ANGSTROM;
                        let symbol = element.symbol();
                        one.push_str(&crate::io::xyz::multioptpy_row(
                            symbol,
                            "   ",
                            [at(0), at(1), at(2)],
                        ));
                        let _ = writeln!(
                            all,
                            "{symbol:2}   {:17.12}  {:17.12}  {:17.12}",
                            at(0),
                            at(1),
                            at(2)
                        );
                    }
                    std::fs::write(directory.join(format!("{}_{image}.xyz", self.name)), one)?;
                }
                std::fs::write(directory.join(format!("{}_path.xyz", self.name)), all)?;
            }
            FilePoints::VariableCell(map) => {
                let mut all = String::new();
                for (image, point) in chain.iter().enumerate() {
                    let mut system = map.system(point)?;
                    if let Some(atoms) = &self.atoms {
                        system = system.trajectory_atoms(atoms)?;
                    }
                    system.comment = format!("NEB image {image}");
                    let frame = crate::io::xyz::format_xyz_system(&system, None);
                    std::fs::write(directory.join(format!("{}_{image}.xyz", self.name)), &frame)?;
                    all.push_str(&frame);
                }
                std::fs::write(directory.join(format!("{}_path.xyz", self.name)), all)?;
            }
            FilePoints::Standard(ChainPoints::Variables(names)) => {
                let mut all = format!("image,{}\n", names.join(","));
                for (image, point) in chain.iter().enumerate() {
                    let values: Vec<String> = point.iter().map(|v| python_float(*v)).collect();
                    let _ = writeln!(all, "{image},{}", values.join(","));
                }
                std::fs::write(directory.join(format!("{}_path.csv", self.name)), all)?;
            }
        }
        Ok(())
    }

    /// The points of a path that is not a chain of images (the points of the direct
    /// MaxFlux method), written as the chain of `iteration` with their energies and their
    /// lengths along the path (Bohr): the structures, the nodes at extrema,
    /// `energy_plot.csv`, `path_length.csv`, and the candidates after it.
    pub fn write_points(
        &mut self,
        iteration: usize,
        chain: &[Col<f64>],
        energies: &[f64],
        lengths: &[f64],
    ) -> MoptResult<()> {
        let directory = self.iteration_directory(iteration);
        std::fs::create_dir_all(&directory)?;
        self.write_chain(&directory, chain)?;
        node_file(&directory, "approx_TS_node", &strict_maxima(energies))?;
        node_file(&directory, "approx_EQ_node", &strict_minima(energies))?;
        self.append("energy_plot.csv", energies)?;
        self.append("path_length.csv", lengths)?;
        self.last = Some((directory, energies.to_vec()));
        Ok(())
    }

    /// The candidates for transition states after the last iteration (`None` before the
    /// first one).
    pub fn candidates(&self) -> Option<ChainCandidates> {
        let (directory, energies) = self.last.as_ref()?;
        let mut images = strict_maxima(energies);
        if images.is_empty() && energies.len() > 2 {
            // No maximum (the energy rises or falls all the way): the highest inner image.
            let inner = &energies[1..energies.len() - 1];
            let top = inner
                .iter()
                .enumerate()
                .fold(
                    (0, f64::NEG_INFINITY),
                    |m, (i, e)| if *e > m.1 { (i, *e) } else { m },
                );
            images.push(top.0 + 1);
        }
        let (files, path) = match &self.points {
            FilePoints::Standard(ChainPoints::Structures(_)) | FilePoints::VariableCell(_) => (
                images
                    .iter()
                    .map(|image| directory.join(format!("{}_{image}.xyz", self.name)))
                    .collect(),
                directory.join(format!("{}_path.xyz", self.name)),
            ),
            FilePoints::Standard(ChainPoints::Variables(_)) => (
                Vec::new(),
                directory.join(format!("{}_path.csv", self.name)),
            ),
        };
        Some(ChainCandidates {
            images,
            files,
            path,
        })
    }
}

fn node_file(directory: &Path, name: &str, images: &[usize]) -> MoptResult<()> {
    let mut text = String::new();
    for image in images {
        let _ = writeln!(text, "{image}");
    }
    std::fs::write(directory.join(format!("{name}.txt")), text)?;
    Ok(())
}

impl ChainObserver for ChainFiles {
    fn after_iteration(
        &mut self,
        report: &ChainReport,
        _machine: &mut ChainRelaxation,
    ) -> MoptResult<Flow> {
        let directory = self.iteration_directory(report.iteration);
        std::fs::create_dir_all(&directory)?;
        self.write_chain(&directory, &report.chain)?;
        node_file(&directory, "approx_TS_node", &report.maxima)?;
        node_file(&directory, "approx_EQ_node", &report.minima)?;
        node_file(
            &directory,
            "local_min_bias_grad_node",
            &strict_minima(&report.gradient_rms),
        )?;
        self.append("path_length.csv", &report.lengths)?;
        self.append("bias_force_rms.csv", &report.gradient_rms)?;
        self.append("orthogonality.csv", &report.cosines)?;
        self.append("perp_rms_gradient.csv", &report.force_rms)?;
        self.append("perp_max_gradient.csv", &report.force_max)?;
        self.append("energy_plot.csv", &report.energies)?;
        if !report.bare_energies.is_empty() {
            self.append(BARE_TABLE, &report.bare_energies)?;
        }
        self.last = Some((directory, report.energies.clone()));
        Ok(Flow::Continue)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::SpinCharge;
    use crate::io::xyz::{parse_xyz_str, read_xyz_trajectory_file};
    use crate::path::band::machine::{ChainAction, ChainSettings, ImageValue};
    use crate::path::band::ChainKind;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mopt_chain_files_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn read(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    /// Two iterations of a chain on `E = sum x^2 / 2` with the files written.
    fn run(files: &mut ChainFiles, chain: Vec<Col<f64>>, kind: ChainKind) -> Vec<ChainReport> {
        let settings = ChainSettings {
            kind,
            max_iterations: 2,
            ..ChainSettings::default()
        };
        let mut machine = ChainRelaxation::new(settings, chain).unwrap();
        let mut reports = Vec::new();
        while let ChainAction::Evaluate(request) = machine.ask() {
            let values = request
                .images
                .iter()
                .map(|x| {
                    let energy = 0.5 * x.iter().map(|v| v * v).sum::<f64>();
                    ImageValue::new(energy, x.clone())
                })
                .collect();
            let report = machine.tell(values).unwrap();
            files.after_iteration(&report, &mut machine).unwrap();
            reports.push(report);
        }
        reports
    }

    #[test]
    fn the_files_of_multioptpy_are_written_for_structures() {
        let dir = scratch("structures");
        let water = "3\n\nO 0.000 0.000 0.000\nH 0.000 0.757 0.586\nH 0.000 -0.757 0.586\n";
        let template = parse_xyz_str(water, SpinCharge::default(), None).unwrap();
        let base = template.coords_flat_bohr();
        // Three images: the molecule stretched more and more, so that the middle image
        // is neither a maximum nor a minimum of the energy.
        let chain: Vec<Col<f64>> = [1.0, 1.1, 1.3]
            .iter()
            .map(|s| Col::from_fn(base.nrows(), |k| s * base[k]))
            .collect();
        let mut files = ChainFiles::new(
            &dir,
            "water",
            ChainPoints::Structures(Box::new(template.clone())),
        )
        .unwrap();
        assert!(files.candidates().is_none());
        let reports = run(&mut files, chain.clone(), ChainKind::Atoms);

        // One line per iteration, one number per image, as Python writes floats.
        for table in TABLES {
            let text = read(&dir.join(table));
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(lines.len(), 2, "{table}");
            assert!(lines.iter().all(|l| l.split(',').count() == 3), "{table}");
        }
        let energies = read(&dir.join("energy_plot.csv"));
        assert_eq!(
            energies.lines().next().unwrap().split(',').next().unwrap(),
            python_float(reports[0].energies[0])
        );
        assert!(read(&dir.join("path_length.csv")).starts_with("0.0,"));

        // The chain of every iteration: one file per image and one with all of them.
        let first = dir.join("path_ITR_0_water");
        let image = read(&first.join("water_1.xyz"));
        let mut lines = image.lines();
        assert_eq!(lines.next(), Some("3"));
        assert_eq!(lines.next(), Some("0 1"));
        let oxygen = lines.next().unwrap();
        assert!(oxygen.starts_with("O    "), "{oxygen:?}");
        assert_eq!(oxygen.len(), 2 + 3 * (3 + 17));
        let all =
            read_xyz_trajectory_file(first.join("water_path.xyz"), SpinCharge::default(), None)
                .unwrap();
        assert_eq!(all.frames.len(), 3);
        let back = all.frames[2].system.coords_flat_bohr();
        assert!((&back - &chain[2]).iter().all(|d| d.abs() < 1e-10));
        assert!(read(&first.join("water_path.xyz")).contains("\nFrame 2\n"));
        assert!(dir.join("path_ITR_1_water").join("water_0.xyz").is_file());
        // No inner extremum: empty lists, and the highest inner image is the candidate.
        assert_eq!(read(&first.join("approx_TS_node.txt")), "");
        assert_eq!(read(&first.join("approx_EQ_node.txt")), "");
        let candidates = files.candidates().unwrap();
        assert_eq!(candidates.images, vec![1]);
        assert_eq!(
            candidates.files,
            vec![dir.join("path_ITR_1_water").join("water_1.xyz")]
        );
        assert!(candidates.path.ends_with("water_path.xyz"));

        // A new run in the same directory starts its tables anew.
        let mut again =
            ChainFiles::new(&dir, "water", ChainPoints::Structures(Box::new(template))).unwrap();
        run(&mut again, chain, ChainKind::Atoms);
        assert_eq!(read(&dir.join("energy_plot.csv")).lines().count(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn variable_cell_frames_keep_the_lattice_of_each_image_and_iteration() {
        let dir = scratch("variable_cell");
        let start = parse_xyz_str(
            "1\nLattice=\"4 0 0 0 4 0 0 0 4\"\nAr 0.5 0.5 0.5\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let map = crate::opt::cell::VariableCell::new(&start, 1.0, 0.0).unwrap();
        let x = map.variables(&start).unwrap();
        let mut y = x.clone();
        y[start.dof()] += 0.1;
        let mut files = ChainFiles::variable_cell(&dir, "neb", &map).unwrap();
        for (iteration, band) in [vec![x.clone(), y.clone()], vec![y, x]]
            .into_iter()
            .enumerate()
        {
            files
                .write_points(iteration, &band, &[0.0, 1.0], &[0.0, 0.1])
                .unwrap();
            let path = files.iteration_directory(iteration).join("neb_path.xyz");
            assert_eq!(read(&path).matches("Lattice=").count(), band.len());
            let traj = read_xyz_trajectory_file(path, SpinCharge::default(), None).unwrap();
            assert_eq!(traj.frames.len(), band.len());
            for (frame, expected) in traj.frames.iter().zip(&band) {
                let back = map.variables(&frame.system).unwrap();
                assert!((&back - expected).iter().all(|d| d.abs() < 1e-8));
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chains_of_variables_get_a_table_and_extrema_are_listed() {
        let dir = scratch("variables");
        let col = |values: &[f64]| Col::from_fn(values.len(), |k| values[k]);
        // The middle image is the highest (a maximum of the profile), the second the
        // lowest inner one.
        let chain = vec![
            col(&[1.0, 0.0]),
            col(&[0.2, 0.0]),
            col(&[2.0, 0.0]),
            col(&[0.5, 0.0]),
            col(&[0.6, 0.0]),
        ];
        let names = vec!["x".to_string(), "y".to_string()];
        let mut files = ChainFiles::new(&dir, "points", ChainPoints::Variables(names)).unwrap();
        run(&mut files, chain, ChainKind::Variables);
        let first = dir.join("path_ITR_0_points");
        let table = read(&first.join("points_path.csv"));
        assert_eq!(table.lines().next(), Some("image,x,y"));
        assert_eq!(table.lines().nth(3), Some("2,2.0,0.0"));
        assert_eq!(read(&first.join("approx_TS_node.txt")), "2\n");
        assert_eq!(read(&first.join("approx_EQ_node.txt")), "1\n3\n");
        let candidates = files.candidates().unwrap();
        assert!(candidates.files.is_empty());
        assert!(candidates.path.ends_with("points_path.csv"));
        assert!(!candidates.images.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
