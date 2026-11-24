//! Observers that analyze the structures of a run: maps of the path in two dimensions
//! (`-cmds`, `-pca`) when the run ends, and the dynamic modes of the last iterations at
//! every iteration (`-km`).
//!
//! MultiOptPy: the calls of `CMDSPathAnalysis`, `PCAPathAnalysis`, and `KoopmanAnalyzer` in
//! `optimization.py:Optimize`, and the files those classes write.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use super::driver::{Observer, Outcome};
use crate::analysis::koopman::{
    behavior, dynamic_modes, observables, DynamicModes, DEFAULT_DEGREE, DEFAULT_RANK_THRESHOLD,
    DEFAULT_WINDOW,
};
use crate::analysis::path_map::{
    cartesian_variables, classical_mds, distance_matrix, distance_variables, principal_components,
    z_matrix_variables, PrincipalComponents, Variables,
};
use crate::core_infra::log::{python_float, Logger};
use crate::core_infra::viz::{ColoredPoints, PlotFigure};
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::MoptResult;
use crate::opt::state_machine::IterationReport;

fn append(path: &Path, text: &str) -> MoptResult<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(text.as_bytes())?;
    Ok(())
}

/// A map of the structures colored by `values`, as a PNG and (with the numbers of the
/// axes and of the color bar) an SVG.
#[allow(clippy::too_many_arguments)]
fn save_map(
    dir: &Path,
    stem: &str,
    title: &str,
    axes: [&str; 2],
    xs: &[f64],
    ys: &[f64],
    values: &[f64],
    label: &str,
) -> MoptResult<()> {
    let mut figure = PlotFigure::new(title, axes[0], axes[1]);
    figure.set_colored_points(ColoredPoints {
        label: format!("{label} (kcal/mol)"),
        xs: xs.to_vec(),
        ys: ys.to_vec(),
        values: values.to_vec(),
    });
    figure.save(dir.join(format!("{stem}.png")))?;
    figure.save(dir.join(format!("{stem}.svg")))
}

/// `-cmds`, `-pca`: at the end of a run, the structures of all iterations in two
/// dimensions.
///
/// `-cmds` (classical multidimensional scaling of the mass-weighted distances between the
/// structures) writes `cmds_analysis_result.csv` and `cmds_result_visualization_energy`
/// / `_bias_energy` (`.png`, `.svg`). `-pca` (principal components of mass-weighted
/// Cartesian coordinates, of all interatomic distances, and of a chain of internal
/// coordinates) writes `pca_analysis_result.log`, `_redundant.log`, `_z_matrix.log` with
/// the components, the same names with `.csv` for the structures along them, and
/// `pca_result_visualization_*` figures.
#[derive(Debug)]
pub struct PathMaps {
    cmds: bool,
    pca: bool,
    dir: PathBuf,
    log: Logger,
    frames: Vec<MolecularSystem>,
    energies: Vec<f64>,
    biased: Vec<f64>,
}

impl PathMaps {
    /// Maps written into `dir`.
    pub fn new(cmds: bool, pca: bool, dir: &Path, log: Logger) -> Self {
        Self {
            cmds,
            pca,
            dir: dir.to_path_buf(),
            log,
            frames: Vec::new(),
            energies: Vec::new(),
            biased: Vec::new(),
        }
    }

    /// Energies relative to the first structure in kcal/mol.
    fn relative(values: &[f64]) -> Vec<f64> {
        let first = values.first().copied().unwrap_or(0.0);
        values
            .iter()
            .map(|v| (v - first) * Units::HARTREE_TO_KCAL_MOL)
            .collect()
    }

    fn write_cmds(&self, energies: &[f64], biased: &[f64]) -> MoptResult<()> {
        self.log
            .info("processing CMDS analysis to approx. reaction path ...");
        let map = classical_mds(&distance_matrix(&self.frames)?, 2)?;
        let share = |k: usize| map.contributions.get(k).copied().unwrap_or(0.0);
        let head = [
            ("dimensional reproducibility:", map.reproducibility),
            ("Percentage contribution 1:", share(0)),
            ("Percentage contribution 2:", share(1)),
        ];
        let mut text = String::new();
        for (name, value) in head {
            self.log.info(&format!("{name} {}", python_float(value)));
            let _ = writeln!(text, "{name},{}", python_float(value));
        }
        text.push_str("itr.,   PCo1,     PCo2,     energy[kcal/mol],    energy(bias)[kcal/mol]\n");
        let n = self.frames.len();
        let column = |k: usize| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    if k < map.coordinates.ncols() {
                        map.coordinates[(i, k)]
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let (xs, ys) = (column(0), column(1));
        for i in 0..n {
            let _ = writeln!(
                text,
                "{i}, {}, {},{}, {}",
                python_float(xs[i]),
                python_float(ys[i]),
                python_float(energies[i]),
                python_float(biased[i])
            );
        }
        fs::write(self.dir.join("cmds_analysis_result.csv"), text)?;
        let axes = ["PCo1 (ang. amu^0.5)", "PCo2 (ang. amu^0.5)"];
        for (name, values) in [("energy", energies), ("bias_energy", biased)] {
            save_map(
                &self.dir,
                &format!("cmds_result_visualization_{name}"),
                &format!("CMDS result ({name})"),
                axes,
                &xs,
                &ys,
                values,
                name,
            )?;
        }
        self.log.info("CMDS analysis completed...");
        Ok(())
    }

    /// One set of variables: the log of the components, the structures along the first two,
    /// and the two figures.
    #[allow(clippy::too_many_arguments)]
    fn write_pca(
        &self,
        heading: &[&str],
        suffix: &str,
        figure: &str,
        unit: &str,
        variables: &Variables,
        standardize: bool,
        energies: &[f64],
        biased: &[f64],
    ) -> MoptResult<()> {
        let n = self.frames.len();
        let pca = principal_components(&variables.values, standardize, n)?;
        let share = |k: usize| pca.contributions.get(k).copied().unwrap_or(0.0);
        for k in 0..3 {
            self.log.info(&format!(
                "Percentage contribution {}: {}",
                k + 1,
                python_float(share(k))
            ));
        }
        fs::write(
            self.dir.join(format!("pca_analysis_result{suffix}.log")),
            component_log(heading, &pca, &variables.names),
        )?;
        let column = |k: usize| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    if k < pca.scores.ncols() {
                        pca.scores[(i, k)]
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let (xs, ys) = (column(0), column(1));
        let mut text = String::from("itr.,PC1,PC2,energy[kcal/mol],energy(bias)[kcal/mol]\n");
        for i in 0..n {
            let _ = writeln!(
                text,
                "{i},{},{},{},{}",
                python_float(xs[i]),
                python_float(ys[i]),
                python_float(energies[i]),
                python_float(biased[i])
            );
        }
        fs::write(
            self.dir.join(format!("pca_analysis_result{suffix}.csv")),
            text,
        )?;
        let x_axis = format!("PC1 ({unit})");
        let y_axis = format!("PC2 ({unit})");
        for (name, values) in [("energy", energies), ("bias_energy", biased)] {
            save_map(
                &self.dir,
                &format!("pca_result_visualization_{figure}{name}"),
                &format!("PCA result ({figure}{name})"),
                [&x_axis, &y_axis],
                &xs,
                &ys,
                values,
                name,
            )?;
        }
        Ok(())
    }

    fn write_all_pca(&self, energies: &[f64], biased: &[f64]) -> MoptResult<()> {
        self.log
            .info("processing PCA analysis to approx. reaction path ...");
        self.log.info("### cartesian coordinate PCA analysis ###");
        self.write_pca(
            &["PCA analysis for approx. reaction path"],
            "",
            "",
            "ang. amu^0.5",
            &cartesian_variables(&self.frames)?,
            false,
            energies,
            biased,
        )?;
        if self.frames.first().is_some_and(|s| s.n_atoms() >= 2) {
            self.log
                .info("### redundant coordinate (only atom distance) PCA analysis ###");
            self.write_pca(
                &[
                    "PCA analysis for approx. reaction path",
                    "redundant coordinates",
                    "(only atom distances)",
                ],
                "_redundant",
                "redundant_",
                "ang.",
                &distance_variables(&self.frames)?,
                false,
                energies,
                biased,
            )?;
            self.log
                .info("### redundant coordinate (z-matrix) PCA analysis ###");
            self.write_pca(
                &[
                    "PCA analysis for approx. reaction path",
                    "redundant coordinates",
                    "(z-matrix)",
                ],
                "_z_matrix",
                "z_matrix_",
                "standardized",
                &z_matrix_variables(&self.frames)?,
                true,
                energies,
                biased,
            )?;
        }
        self.log.info("PCA analysis completed...");
        Ok(())
    }
}

/// The components of an analysis in the layout of MultiOptPy's logs: for each, its share
/// of the variance and every variable with its entry and the share of that entry in the
/// sum of the absolute entries, then the largest of those shares.
fn component_log(heading: &[&str], pca: &PrincipalComponents, names: &[String]) -> String {
    let width = 42;
    let stars = "*".repeat(width + 2);
    let mut text = format!("{stars}\n*{:width$}*\n", "");
    for line in heading {
        let _ = writeln!(text, "*{line:^width$}*");
    }
    let _ = writeln!(text, "*{:width$}*\n{stars}\n", "");
    for k in 0..pca.variances.len() {
        let entries: Vec<f64> = (0..names.len()).map(|v| pca.vectors[(v, k)]).collect();
        let total: f64 = entries.iter().map(|e| e.abs()).sum();
        let shares: Vec<f64> = entries
            .iter()
            .map(|e| if total > 0.0 { e.abs() / total } else { 0.0 })
            .collect();
        let _ = writeln!(
            text,
            "-----\nbasis {k}\ncontribution: {}\n  vector    contribution of vector",
            python_float(pca.contributions[k])
        );
        for (v, name) in names.iter().enumerate() {
            let _ = writeln!(
                text,
                "{v}  {name} : {}   {}",
                python_float(entries[v]),
                python_float(shares[v])
            );
        }
        let mut order: Vec<usize> = (0..names.len()).collect();
        order.sort_by(|&a, &b| shares[b].total_cmp(&shares[a]));
        let largest: Vec<String> = order
            .iter()
            .take(9)
            .map(|&v| format!("{v}  {}", python_float(shares[v])))
            .collect();
        let _ = writeln!(text, "{}", largest.join("  "));
    }
    text.push_str("-----\n");
    text
}

impl Observer for PathMaps {
    fn on_iteration(&mut self, r: &IterationReport) -> MoptResult<()> {
        self.record(&r.system, r.energy.value(), r.biased_energy.value());
        Ok(())
    }

    fn on_finish(&mut self, _outcome: &Outcome) -> MoptResult<()> {
        self.finish()
    }
}

impl PathMaps {
    /// One more structure with its energy without and with the bias potentials (Hartree).
    pub fn record(&mut self, sys: &MolecularSystem, energy: f64, biased: f64) {
        self.frames.push(sys.clone());
        self.energies.push(energy);
        self.biased.push(biased);
    }

    /// The maps of the structures recorded (not enough structures, or structures that
    /// cannot be mapped, are reported and are no error).
    pub fn finish(&mut self) -> MoptResult<()> {
        let energies = Self::relative(&self.energies);
        let biased = Self::relative(&self.biased);
        if self.frames.len() < 3 {
            if self.cmds {
                self.log
                    .info("There is not enough data to perform CMDS analysis...");
            }
            if self.pca {
                self.log
                    .info("There is not enough data to perform PCA analysis...");
            }
            return Ok(());
        }
        // An analysis that cannot be done (structures that all coincide, for example) is
        // reported and does not fail the run it follows.
        if self.cmds {
            if let Err(e) = self.write_cmds(&energies, &biased) {
                self.log.warn(&format!("CMDS analysis was not done: {e}"));
            }
        }
        if self.pca {
            if let Err(e) = self.write_all_pca(&energies, &biased) {
                self.log.warn(&format!("PCA analysis was not done: {e}"));
            }
        }
        Ok(())
    }
}

/// `-km`: at every iteration, the dynamic modes of the last iterations
/// ([`crate::analysis::koopman`]) in `koopman_analysis/`: `coordinates.csv` (coordinates in
/// Bohr and the gradient of every iteration), `koopman_eigs.csv` (the eigenvalues),
/// `koopman_modes.log` (the modes in the space of the observables), and
/// `koopman_mode_anim_<k>.xyz` (the latest structure moved along the part of mode `k` in
/// the coordinates).
#[derive(Debug)]
pub struct KoopmanLog {
    dir: PathBuf,
    log: Logger,
    window: usize,
    snapshots: Vec<Vec<f64>>,
    /// Frames of an animation and its amplitude in Bohr (MultiOptPy: 10 and 1.5).
    frames: usize,
    amplitude: f64,
}

impl KoopmanLog {
    /// Files in `koopman_analysis/` under `dir`.
    pub fn new(dir: &Path, log: Logger) -> MoptResult<Self> {
        let dir = dir.join("koopman_analysis");
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            log,
            window: DEFAULT_WINDOW,
            snapshots: Vec::new(),
            frames: 10,
            amplitude: 1.5,
        })
    }

    fn write_modes(
        &self,
        iteration: usize,
        modes: &DynamicModes,
        sys: &MolecularSystem,
    ) -> MoptResult<()> {
        let eigs = self.dir.join("koopman_eigs.csv");
        if !eigs.exists() {
            let mut header = String::from("iteration,rank");
            for k in 0..self.window {
                let _ = write!(header, ",mode_{k}_real,mode_{k}_imag,mode_{k}_norm");
            }
            header.push('\n');
            fs::write(&eigs, header)?;
        }
        let mut row = format!("{iteration},{}", modes.rank);
        for k in 0..self.window {
            match modes.eigenvalues.get(k) {
                Some(e) => {
                    let _ = write!(
                        row,
                        ",{},{},{}",
                        python_float(e[0]),
                        python_float(e[1]),
                        python_float(modes.modulus(k))
                    );
                }
                None => row.push_str(",,,"),
            }
        }
        row.push('\n');
        append(&eigs, &row)?;

        let mut text = format!("Iteration {iteration}:\n");
        for (k, mode) in modes.modes.iter().enumerate() {
            let e = modes.eigenvalues[k];
            let _ = writeln!(
                text,
                "Eigenvalue {k}: Real={:.6}, Imag={:.6}, Norm={:.6} ({})",
                e[0],
                e[1],
                modes.modulus(k),
                behavior(e)
            );
            for (title, part) in [("real", 0), ("imag", 1)] {
                let _ = writeln!(text, "Mode vector ({title} part):");
                for entry in mode {
                    let _ = writeln!(text, "{:.6}", entry[part]);
                }
            }
            text.push('\n');
        }
        text.push('\n');
        append(&self.dir.join("koopman_modes.log"), &text)?;

        // The part of each mode in the coordinates (its first 3N entries), normalized, as a
        // movement of the latest structure.
        let n3 = 3 * sys.n_atoms();
        for (k, mode) in modes.modes.iter().enumerate() {
            let part: Vec<f64> = mode.iter().take(n3).map(|c| c[0]).collect();
            let norm = libm::sqrt(part.iter().map(|v| v * v).sum::<f64>());
            let mut animation = String::new();
            for frame in 0..self.frames {
                let phase =
                    2.0 * std::f64::consts::PI * frame as f64 / (self.frames.max(2) - 1) as f64;
                let scale = if norm > 1e-9 {
                    self.amplitude * libm::sin(phase) / norm
                } else {
                    0.0
                };
                let _ = writeln!(animation, "{}\nMode {k} Frame {frame}", sys.n_atoms());
                for atom in sys.atoms() {
                    let i = atom.index();
                    let p = sys.position(AtomId::new(i)).value();
                    let moved: Vec<f64> = (0..3)
                        .map(|c| (p[c] + scale * part[3 * i + c]) * Units::BOHR_TO_ANGSTROM)
                        .collect();
                    let _ = writeln!(
                        animation,
                        "{} {:.6} {:.6} {:.6}",
                        sys.element(atom).symbol(),
                        moved[0],
                        moved[1],
                        moved[2]
                    );
                }
            }
            fs::write(
                self.dir.join(format!("koopman_mode_anim_{k}.xyz")),
                animation,
            )?;
        }
        Ok(())
    }
}

impl Observer for KoopmanLog {
    fn on_iteration(&mut self, r: &IterationReport) -> MoptResult<()> {
        let positions: Vec<f64> = r.system.coords_flat_bohr().iter().copied().collect();
        let gradient: Vec<f64> = r.gradient.iter().copied().collect();
        let coordinates = self.dir.join("coordinates.csv");
        if !coordinates.exists() {
            let mut header = String::from("iteration");
            for prefix in ["", "grad_"] {
                for atom in 1..=r.system.n_atoms() {
                    for axis in ["x", "y", "z"] {
                        let _ = write!(header, ",{prefix}{axis}{atom}");
                    }
                }
            }
            header.push('\n');
            fs::write(&coordinates, header)?;
        }
        let mut row = r.iteration.to_string();
        for value in positions.iter().chain(&gradient) {
            let _ = write!(row, ",{}", python_float(*value));
        }
        row.push('\n');
        append(&coordinates, &row)?;

        self.snapshots
            .push(observables(&positions, Some(&gradient), DEFAULT_DEGREE));
        if self.snapshots.len() > self.window + 1 {
            self.snapshots.remove(0);
        }
        if self.snapshots.len() < self.window + 1 {
            return Ok(());
        }
        match dynamic_modes(&self.snapshots, DEFAULT_RANK_THRESHOLD) {
            Ok(Some(modes)) => self.write_modes(r.iteration, &modes, &r.system),
            Ok(None) => {
                self.log
                    .info("Warning: All singular values are below threshold. No rank found.");
                Ok(())
            }
            Err(e) => {
                self.log
                    .warn(&format!("Koopman analysis was not done: {e}"));
                Ok(())
            }
        }
    }
}
