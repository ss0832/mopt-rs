//! The trajectory analysis of a molecular dynamics run: the files of a run of `mdmain` (or
//! any trajectory in XYZ, with velocities in the same layout) are read, the analyses of
//! [`crate::analysis::trajectory`] are run, and their results are written as CSV files and
//! figures (`md_analysis_output`), each named in the log.
//!
//! A run directory holds `<stem>_traj.xyz` (the frames, with `time=` in their comments),
//! `<stem>_velocities.xyz` (Angstrom per fs), and `md_properties.csv` (every step: energies,
//! temperature, pressure, volume); the statistics are those of `md_properties.csv`, or of
//! the energies of the frames when there is none.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fs;
use std::path::{Path, PathBuf};

use faer::Col;

use super::md_analysis_output as output;
use super::md_analysis_transport as transport;
use crate::analysis::correlation::spectrum::{QuantumCorrection, Window, DEFAULT_PADDING};
use crate::analysis::correlation::transport::DEFAULT_BLOCKS;
use crate::analysis::trajectory::{MdTrajectory, Measure, MsdSettings, RdfSettings, VdosSettings};
use crate::core_infra::log::Logger;
use crate::domain::{SpinCharge, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::read_xyz_trajectory_file;

/// One analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Analysis {
    /// Means with block-average errors of the energies, temperature, pressure, volume.
    Statistics,
    /// Radial distribution functions.
    Rdf,
    /// Mean squared displacements and diffusion coefficients.
    Msd,
    /// Velocity autocorrelation and vibrational density of states.
    Vacf,
    /// RMSD to the first frame and RMSF of every atom.
    Rmsd,
    /// Distances, angles, dihedral angles of chosen atoms.
    Geometry,
    /// Principal components of the motion.
    Pca,
    /// Autocorrelations and power spectra of the properties, the chosen distances and
    /// angles, and the orientations of chosen bonds.
    Correlation,
    /// Infrared spectrum from the dipole autocorrelation.
    Infrared,
    /// Transport coefficients: diffusion (Einstein and Green-Kubo), thermal conductivity
    /// (Green-Kubo of the heat current, reverse non-equilibrium), shear viscosity.
    Transport,
}

impl Analysis {
    /// Every analysis, in the order they run.
    pub const ALL: [Self; 10] = [
        Self::Statistics,
        Self::Rdf,
        Self::Msd,
        Self::Vacf,
        Self::Rmsd,
        Self::Geometry,
        Self::Pca,
        Self::Correlation,
        Self::Infrared,
        Self::Transport,
    ];

    /// The analysis called `name`.
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "statistics" | "stats" => Ok(Self::Statistics),
            "rdf" => Ok(Self::Rdf),
            "msd" | "diffusion" => Ok(Self::Msd),
            "vacf" | "vdos" => Ok(Self::Vacf),
            "rmsd" | "rmsf" => Ok(Self::Rmsd),
            "geometry" | "geom" => Ok(Self::Geometry),
            "pca" => Ok(Self::Pca),
            "acf" | "correlation" | "correlations" | "power" => Ok(Self::Correlation),
            "ir" | "infrared" => Ok(Self::Infrared),
            "transport" | "conductivity" | "viscosity" | "green_kubo" => Ok(Self::Transport),
            other => Err(MoptError::config(format!(
                "unknown trajectory analysis '{other}' (statistics, rdf, msd, vacf, rmsd, \
                 geometry, pca, acf, ir, transport, all)"
            ))),
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Statistics => "statistics",
            Self::Rdf => "rdf",
            Self::Msd => "msd",
            Self::Vacf => "vacf",
            Self::Rmsd => "rmsd",
            Self::Geometry => "geometry",
            Self::Pca => "pca",
            Self::Correlation => "acf",
            Self::Infrared => "ir",
            Self::Transport => "transport",
        }
    }
}

/// Settings of the spectra and the Green-Kubo integrals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectraSettings {
    /// Window of the correlations.
    pub window: Window,
    /// Zero padding of the transforms.
    pub padding: usize,
    /// Quantum correction of the infrared spectrum.
    pub correction: QuantumCorrection,
    /// Longest lag of the Green-Kubo integrals (fs; `None`: half a block).
    pub gk_lag_fs: Option<f64>,
    /// Blocks of the errors of the Green-Kubo integrals.
    pub gk_blocks: usize,
    /// Slabs and axis of a reverse non-equilibrium run.
    pub exchange: (usize, usize),
}

impl Default for SpectraSettings {
    fn default() -> Self {
        Self {
            window: Window::Hann,
            padding: DEFAULT_PADDING,
            correction: QuantumCorrection::Harmonic,
            gk_lag_fs: None,
            gk_blocks: DEFAULT_BLOCKS,
            exchange: (20, 2),
        }
    }
}

/// The series of every step a run writes besides its properties.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Series {
    /// `<stem>_dipoles.csv`.
    pub dipoles: Option<Properties>,
    /// `<stem>_heat_flux.csv`.
    pub heat_flux: Option<Properties>,
    /// `<stem>_pressure_tensor.csv`.
    pub pressure: Option<Properties>,
    /// `<stem>_cv.csv` (`-cv`).
    pub cvs: Option<Properties>,
}

impl Properties {
    /// These columns with the collective variables of `cvs` added (as `cv <name>`), when
    /// both have the same rows.
    pub fn with_cvs(&self, cvs: &Properties) -> Properties {
        let mut out = self.clone();
        let rows = self.columns.first().map_or(0, Vec::len);
        for (name, c) in cvs.names.iter().zip(&cvs.columns) {
            if name != "step" && name != "time_fs" && c.len() == rows {
                out.names.push(format!("cv {name}"));
                out.columns.push(c.clone());
            }
        }
        out
    }
}

/// Default number of principal components written.
pub const DEFAULT_COMPONENTS: usize = 3;

/// Settings of the analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisSettings {
    /// The analyses to run.
    pub analyses: Vec<Analysis>,
    /// Frames (and steps of the statistics) before this time are left out (fs).
    pub skip_fs: f64,
    /// Interval of frames without `time=` in their comments (fs).
    pub interval_fs: f64,
    /// Radial distribution functions.
    pub rdf: RdfSettings,
    /// Mean squared displacements.
    pub msd: MsdSettings,
    /// Share of the run the velocity autocorrelation reaches.
    pub correlation_share: f64,
    /// Density of states.
    pub vdos: VdosSettings,
    /// Distances, angles, dihedral angles.
    pub measures: Vec<Measure>,
    /// Principal components.
    pub components: usize,
    /// Extension of the figures (`png` or `svg`).
    pub figures: String,
    /// Spectra and Green-Kubo integrals.
    pub spectra: SpectraSettings,
    /// Bonds (pairs of atoms from 0) whose orientations are correlated.
    pub bonds: Vec<(usize, usize)>,
}

impl Default for AnalysisSettings {
    fn default() -> Self {
        Self {
            analyses: Analysis::ALL.to_vec(),
            skip_fs: 0.0,
            interval_fs: 1.0,
            rdf: RdfSettings::default(),
            msd: MsdSettings::default(),
            correlation_share: crate::analysis::trajectory::dynamics::DEFAULT_CORRELATION_SHARE,
            vdos: VdosSettings::default(),
            measures: Vec::new(),
            components: DEFAULT_COMPONENTS,
            figures: "png".to_string(),
            spectra: SpectraSettings::default(),
            bonds: Vec::new(),
        }
    }
}

/// The files of a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunFiles {
    /// The frames.
    pub trajectory: PathBuf,
    /// Their velocities, when there are.
    pub velocities: Option<PathBuf>,
    /// `md_properties.csv`, when there is one.
    pub properties: Option<PathBuf>,
    /// The dipoles, heat currents, pressure tensors, and collective variables of the steps,
    /// when there are.
    pub series: [Option<PathBuf>; 4],
}

impl RunFiles {
    /// The files of the series next to `trajectory` for its stem.
    pub fn series_of(dir: &Path, stem: &str) -> [Option<PathBuf>; 4] {
        ["dipoles", "heat_flux", "pressure_tensor", "cv"].map(|name| {
            let p = dir.join(format!("{stem}_{name}.csv"));
            p.is_file().then_some(p)
        })
    }

    /// The series read.
    pub fn read_series(&self) -> MoptResult<Series> {
        let read = |p: &Option<PathBuf>| p.as_deref().map(Properties::read).transpose();
        Ok(Series {
            dipoles: read(&self.series[0])?,
            heat_flux: read(&self.series[1])?,
            pressure: read(&self.series[2])?,
            cvs: read(&self.series[3])?,
        })
    }
}

impl RunFiles {
    /// The files of the run directory or trajectory file `path`.
    pub fn locate(path: &Path) -> MoptResult<Self> {
        let trajectory = if path.is_dir() {
            let mut found: Vec<PathBuf> = fs::read_dir(path)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.ends_with("_traj.xyz"))
                })
                .collect();
            found.sort();
            match found.len() {
                1 => found.remove(0),
                0 => {
                    return Err(MoptError::config(format!(
                        "{} holds no trajectory (<stem>_traj.xyz)",
                        path.display()
                    )))
                }
                _ => {
                    return Err(MoptError::config(format!(
                        "{} holds several trajectories; name one of them",
                        path.display()
                    )))
                }
            }
        } else if path.is_file() {
            path.to_path_buf()
        } else {
            return Err(MoptError::config(format!(
                "{} is neither a run directory nor a trajectory file",
                path.display()
            )));
        };
        let dir = trajectory
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let name = trajectory
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let stem = name
            .strip_suffix("_traj.xyz")
            .or_else(|| name.strip_suffix(".xyz"))
            .unwrap_or(&name);
        let velocities = dir.join(format!("{stem}_velocities.xyz"));
        let properties = dir.join("md_properties.csv");
        Ok(Self {
            velocities: velocities.is_file().then_some(velocities),
            properties: properties.is_file().then_some(properties),
            series: Self::series_of(&dir, stem),
            trajectory,
        })
    }
}

/// Velocities in the layout of XYZ coordinates (Angstrom per fs), as `mdmain` writes them:
/// one vector per frame in Bohr per atomic unit of time.
pub fn read_velocities(path: &Path, atoms: usize) -> MoptResult<Vec<Col<f64>>> {
    let text = fs::read_to_string(path)
        .map_err(MoptError::from)
        .context(path.display())?;
    let lines: Vec<&str> = text.lines().collect();
    let scale = Units::ANGSTROM_TO_BOHR * Units::AU_TIME_TO_FS;
    let mut frames = Vec::new();
    let mut k = 0;
    while k < lines.len() {
        if lines[k].trim().is_empty() {
            k += 1;
            continue;
        }
        let count: usize = lines[k].trim().parse().map_err(|_| {
            MoptError::parse(format!(
                "{}: line {}: not a count of atoms",
                path.display(),
                k + 1
            ))
        })?;
        if count != atoms || k + 2 + count > lines.len() {
            return Err(MoptError::parse(format!(
                "{}: a frame of {count} atoms at line {} (the trajectory has {atoms})",
                path.display(),
                k + 1
            )));
        }
        let mut v = Col::<f64>::zeros(3 * count);
        for i in 0..count {
            let fields: Vec<&str> = lines[k + 2 + i].split_whitespace().collect();
            for c in 0..3 {
                let value = fields
                    .get(c + 1)
                    .and_then(|f| f.parse::<f64>().ok())
                    .ok_or_else(|| {
                        MoptError::parse(format!(
                            "{}: line {}: three velocities expected",
                            path.display(),
                            k + 3 + i
                        ))
                    })?;
                v[3 * i + c] = value * scale;
            }
        }
        frames.push(v);
        k += 2 + count;
    }
    Ok(frames)
}

/// Columns of `md_properties.csv` (or any CSV of numbers with a header): names and values.
#[derive(Debug, Clone, PartialEq)]
pub struct Properties {
    /// Names of the columns.
    pub names: Vec<String>,
    /// Values, `[column][row]` (empty cells are NaN).
    pub columns: Vec<Vec<f64>>,
}

impl Properties {
    /// Read a CSV file with a header line.
    pub fn read(path: &Path) -> MoptResult<Self> {
        let text = fs::read_to_string(path)
            .map_err(MoptError::from)
            .context(path.display())?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let names: Vec<String> = lines
            .next()
            .ok_or_else(|| MoptError::parse(format!("{} is empty", path.display())))?
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let mut columns = vec![Vec::new(); names.len()];
        for line in lines {
            for (k, field) in line.split(',').enumerate().take(names.len()) {
                columns[k].push(field.trim().parse::<f64>().unwrap_or(f64::NAN));
            }
        }
        Ok(Self { names, columns })
    }

    /// The values of column `name`.
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|k| self.columns[k].as_slice())
    }
}

/// The trajectory of `files` (with its velocities, when they are there and match).
pub fn read_trajectory(
    files: &RunFiles,
    interval_fs: f64,
    log: &Logger,
) -> MoptResult<MdTrajectory> {
    let frames = read_xyz_trajectory_file(&files.trajectory, SpinCharge::default(), None)?;
    let mut traj = MdTrajectory::from_trajectory(&frames, interval_fs)?;
    if let Some(path) = &files.velocities {
        let v = read_velocities(path, traj.n_atoms())?;
        if v.len() == traj.len() {
            traj = traj.with_velocities(&v)?;
        } else {
            log.warn(&format!(
                "{} has {} frames, the trajectory {}: the velocities are not used.",
                path.display(),
                v.len(),
                traj.len()
            ));
        }
    }
    Ok(traj)
}

/// Run the analyses of `s` on the files of a run and write the results into `out`; the
/// files written.
pub fn analyze_files(
    files: &RunFiles,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let traj = read_trajectory(files, s.interval_fs, log)?;
    let properties = match &files.properties {
        Some(p) => Some(Properties::read(p)?),
        None => None,
    };
    log.info(&format!(
        "Trajectory analysis of {}{}{}",
        files.trajectory.display(),
        files
            .velocities
            .as_ref()
            .map_or_else(String::new, |p| format!(", velocities {}", p.display())),
        files
            .properties
            .as_ref()
            .map_or_else(String::new, |p| format!(", properties {}", p.display()))
    ));
    let series = files.read_series()?;
    let properties = match (properties, &series.cvs) {
        (Some(p), Some(c)) => Some(p.with_cvs(c)),
        (p, _) => p,
    };
    analyze(&traj, properties.as_ref(), &series, s, out, log)
}

/// The statistics of the steps of a run (`md_properties.csv`) alone, into `out`.
pub fn analyze_properties(
    properties: &Properties,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    fs::create_dir_all(out)?;
    let files = output::statistics(None, Some(properties), s, out, log)?;
    for f in &files {
        log.info(&format!(
            "Trajectory analysis (statistics): wrote {}",
            f.display()
        ));
    }
    Ok(files)
}

/// Run the analyses of `s` on `traj` (and the statistics on `properties`, or on the energies
/// of the frames) and write the results into `out`; the files written.
pub fn analyze(
    traj: &MdTrajectory,
    properties: Option<&Properties>,
    series: &Series,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    fs::create_dir_all(out)?;
    let traj = traj.from_time(s.skip_fs)?;
    let (traj, dropped) = traj.without_short_last_interval();
    log.info(&format!(
        "Trajectory analysis: {} frames from {} to {} fs, {} atoms{}{}",
        traj.len(),
        traj.times()[0],
        traj.times()[traj.len() - 1],
        traj.n_atoms(),
        if traj.is_periodic() {
            ", periodic cell"
        } else {
            ""
        },
        if dropped {
            " (the last frame, off the interval of the others, is left out of the analysis)"
        } else {
            ""
        }
    ));
    let mut written = Vec::new();
    for analysis in &s.analyses {
        let result = match analysis {
            Analysis::Statistics => output::statistics(Some(&traj), properties, s, out, log),
            Analysis::Rdf => output::rdf(&traj, s, out, log),
            Analysis::Msd => output::msd(&traj, s, out, log),
            Analysis::Vacf => output::vacf(&traj, s, out, log),
            Analysis::Rmsd => output::rmsd(&traj, s, out, log),
            Analysis::Geometry => output::geometry(&traj, s, out, log),
            Analysis::Pca => output::pca(&traj, s, out, log),
            Analysis::Correlation => transport::correlations(&traj, properties, s, out, log),
            Analysis::Infrared => transport::infrared(properties, series, s, out, log),
            Analysis::Transport => transport::transport(&traj, properties, series, s, out, log),
        };
        match result {
            Ok(files) => {
                for f in &files {
                    log.info(&format!(
                        "Trajectory analysis ({}): wrote {}",
                        analysis.name(),
                        f.display()
                    ));
                }
                written.extend(files);
            }
            Err(e) => log.warn(&format!(
                "Trajectory analysis ({}) not done: {e}",
                analysis.name()
            )),
        }
    }
    Ok(written)
}
