//! The trajectory analysis of `mdmain`: `-analysis [names] [key=value ...]` analyzes the
//! trajectory of a run when it ends, and `-analyze_only` analyzes the runs (directories)
//! or trajectory files given as inputs without a new run.
//!
//! Names: `statistics`, `rdf`, `msd`, `vacf` (with the density of states), `rmsd` (with the
//! RMSF), `geometry`, `pca`, `acf` (autocorrelations and power spectra), `ir` (infrared
//! spectrum of the dipoles), `transport` (diffusion, thermal conductivity, shear
//! viscosity), `all` (the default). Settings: `skip=<fs>` (frames and steps
//! before are left out), `interval=<fs>` (frames without times in their comments),
//! `rdf_bin=<ang.>`, `rdf_max=<ang.>`, `msd_fit=<from>,<to>` (shares of the longest lag),
//! `drift=true|false` (subtract the drift of the centre of mass from the MSD),
//! `vacf_share=<share>` (of the run the autocorrelations reach), `vdos_max=<cm^-1>`,
//! `window=hann|blackman|none`, `padding=<n>`, `qcf=harmonic|standard|schofield|none`,
//! `gk_lag=<fs>`, `gk_blocks=<n>`, `bonds=1,2;3,4` (orientational correlations),
//! `exchange_slabs=<n>`, `exchange_axis=a|b|c` (of a reverse non-equilibrium run),
//! `atoms=1,2;1,2,3;1,2,3,4` (distances, angles, dihedral angles, 1-based),
//! `components=<n>`, `figures=png|svg`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use super::mdmain::{MdmainConfig, MdmainResult};
use crate::analysis::correlation::spectrum::{QuantumCorrection, Window};
use crate::analysis::trajectory::Measure;
use crate::core_infra::log::{Level, Logger};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::workflow::md_analysis::{
    analyze_files, analyze_properties, Analysis, AnalysisSettings, Properties, RunFiles,
};

/// Directory of the results, in the run directory or next to a trajectory file.
pub const ANALYSIS_DIR: &str = "trajectory_analysis";

fn number(key: &str, value: &str) -> MoptResult<f64> {
    let v = value.trim();
    let v = v.strip_suffix("fs").unwrap_or(v).trim();
    match v.parse::<f64>() {
        Ok(x) if x.is_finite() => Ok(x),
        _ => Err(MoptError::config(format!(
            "-analysis {key}= needs a number, not '{value}'"
        ))),
    }
}

fn positive(key: &str, value: &str) -> MoptResult<f64> {
    let x = number(key, value)?;
    if x > 0.0 {
        Ok(x)
    } else {
        Err(MoptError::config(format!(
            "-analysis {key}= must be positive, not {value}"
        )))
    }
}

fn measures(value: &str) -> MoptResult<Vec<Measure>> {
    value
        .split([';', '/'])
        .filter(|g| !g.trim().is_empty())
        .map(|group| {
            let atoms: Vec<usize> = group
                .split([',', '-'])
                .filter(|t| !t.trim().is_empty())
                .map(|t| match t.trim().parse::<usize>() {
                    Ok(i) if i >= 1 => Ok(i - 1),
                    _ => Err(MoptError::config(format!(
                        "-analysis atoms=: '{t}' is not a 1-based atom number"
                    ))),
                })
                .collect::<MoptResult<_>>()?;
            Measure::of(&atoms)
        })
        .collect()
}

fn set(s: &mut AnalysisSettings, key: &str, value: &str) -> MoptResult<()> {
    match key {
        "skip" => s.skip_fs = number(key, value)?.max(0.0),
        "interval" => s.interval_fs = positive(key, value)?,
        "rdf_bin" => s.rdf.bin_width = positive(key, value)?,
        "rdf_max" => s.rdf.max = Some(positive(key, value)?),
        "msd_fit" => {
            let parts: Vec<&str> = value.split(',').collect();
            let [a, b] = parts.as_slice() else {
                return Err(MoptError::config(
                    "-analysis msd_fit= needs two shares, as msd_fit=0.1,0.9",
                ));
            };
            s.msd.fit = (number(key, a)?, number(key, b)?);
        }
        "drift" => {
            s.msd.remove_drift = match value.trim().to_ascii_lowercase().as_str() {
                "true" | "yes" | "1" | "on" => true,
                "false" | "no" | "0" | "off" => false,
                other => {
                    return Err(MoptError::config(format!(
                        "-analysis drift= is true or false, not '{other}'"
                    )))
                }
            }
        }
        "vacf_share" => s.correlation_share = positive(key, value)?.min(1.0),
        "vdos_max" => s.vdos.max = Some(positive(key, value)?),
        "atoms" => s.measures = measures(value)?,
        "components" => s.components = positive(key, value)?.round().max(1.0) as usize,
        "figures" => {
            let ext = value.trim().to_ascii_lowercase();
            if ext != "png" && ext != "svg" {
                return Err(MoptError::config(format!(
                    "-analysis figures= is png or svg, not '{value}'"
                )));
            }
            s.figures = ext;
        }
        other => return set_spectra(s, other, value),
    }
    Ok(())
}

/// The settings of the spectra, correlations, and transport coefficients.
fn set_spectra(s: &mut AnalysisSettings, key: &str, value: &str) -> MoptResult<()> {
    let sp = &mut s.spectra;
    match key {
        "window" => {
            sp.window = Window::parse(value)?;
            s.vdos.window = sp.window;
        }
        "padding" => {
            sp.padding = positive(key, value)?.round() as usize;
            s.vdos.padding = sp.padding;
        }
        "qcf" => sp.correction = QuantumCorrection::parse(value)?,
        "gk_lag" => sp.gk_lag_fs = Some(positive(key, value)?),
        "gk_blocks" => sp.gk_blocks = (positive(key, value)?.round() as usize).max(2),
        "exchange_slabs" => sp.exchange.0 = positive(key, value)?.round() as usize,
        "exchange_axis" => {
            sp.exchange.1 = match value.trim().to_ascii_lowercase().as_str() {
                "a" | "x" | "0" => 0,
                "b" | "y" | "1" => 1,
                "c" | "z" | "2" => 2,
                other => {
                    return Err(MoptError::config(format!(
                        "-analysis exchange_axis={other}: a, b, or c"
                    )))
                }
            }
        }
        "bonds" => {
            s.bonds = measures(value)?
                .into_iter()
                .map(|m| match m {
                    Measure::Distance(a, b) => Ok((a, b)),
                    _ => Err(MoptError::config(
                        "-analysis bonds= are pairs of atoms, as bonds=1,2;1,3",
                    )),
                })
                .collect::<MoptResult<_>>()?;
        }
        other => {
            return Err(MoptError::config(format!(
                "unknown -analysis setting '{other}' (skip, interval, rdf_bin, rdf_max, \
                 msd_fit, drift, vacf_share, vdos_max, atoms, components, figures, window, \
                 padding, qcf, gk_lag, gk_blocks, bonds, exchange_slabs, exchange_axis)"
            )))
        }
    }
    Ok(())
}

/// The settings of `-analysis` (`None` when it is not given and `-analyze_only` is not
/// either).
pub fn parse_analysis(given: bool, tokens: &[String]) -> MoptResult<Option<AnalysisSettings>> {
    if !given {
        return Ok(None);
    }
    let mut s = AnalysisSettings::default();
    let mut chosen: Vec<Analysis> = Vec::new();
    for token in tokens.iter().flat_map(|t| t.split_whitespace()) {
        if let Some((key, value)) = token.split_once('=') {
            set(&mut s, &key.trim().to_ascii_lowercase(), value)?;
            continue;
        }
        for name in token.split(',').filter(|n| !n.trim().is_empty()) {
            if name.trim().eq_ignore_ascii_case("all") {
                chosen.extend(Analysis::ALL);
            } else {
                chosen.push(Analysis::parse(name)?);
            }
        }
    }
    if !chosen.is_empty() {
        s.analyses = Analysis::ALL
            .into_iter()
            .filter(|a| chosen.contains(a))
            .collect();
    }
    if s.analyses == [Analysis::Geometry] && s.measures.is_empty() {
        return Err(MoptError::config(
            "-analysis geometry needs the atoms, as atoms=1,2;1,2,3",
        ));
    }
    Ok(Some(s))
}

/// The directory of the results of `input` (a run directory or a trajectory file).
pub fn output_dir(input: &Path) -> PathBuf {
    if input.is_dir() {
        input.join(ANALYSIS_DIR)
    } else {
        input
            .parent()
            .map_or_else(|| PathBuf::from(ANALYSIS_DIR), |p| p.join(ANALYSIS_DIR))
    }
}

/// `-analyze_only`: analyze every input without a new run.
pub fn analyze_inputs(
    config: &MdmainConfig,
    console: Option<Level>,
) -> MoptResult<Vec<MdmainResult>> {
    let settings = config.analysis.clone().unwrap_or_default();
    let mut results = Vec::new();
    for input in &config.inputs {
        let files = RunFiles::locate(input).context(input.display())?;
        let out = output_dir(input);
        std::fs::create_dir_all(&out)?;
        let log = console.map_or_else(Logger::silent, Logger::console);
        log.add_file(out.join("mopt.log"), Level::Debug)?;
        log.info(&format!(
            "mopt-rs {} mdmain -analyze_only: {}",
            crate::VERSION,
            input.display()
        ));
        analyze_files(&files, &settings, &out, &log).context(input.display())?;
        redraw_hopping(input, &out, &log)?;
        log.flush();
        results.push(MdmainResult {
            directory: out,
            last: files.trajectory.clone(),
            trajectory: files.trajectory,
            interrupted: false,
            steps: 0,
        });
    }
    Ok(results)
}

/// The figure of surface hopping of a run directory that has a `<stem>_states.csv`.
fn redraw_hopping(input: &Path, out: &Path, log: &Logger) -> MoptResult<()> {
    if !input.is_dir() {
        return Ok(());
    }
    let mut tables: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with("_states.csv"))
        .collect();
    tables.sort();
    for states in tables {
        if let Some(path) = crate::workflow::md_hopping_output::redraw(&states, out)? {
            log.info(&format!("Surface hopping: {}", path.display()));
        }
    }
    Ok(())
}

/// The analysis at the end of a run of plain variables: the statistics of its steps (the
/// other analyses are those of atoms).
pub fn analyze_variables(config: &MdmainConfig, dir: &Path, log: &Logger) -> MoptResult<()> {
    let Some(settings) = &config.analysis else {
        return Ok(());
    };
    if settings.analyses.iter().any(|a| *a != Analysis::Statistics) {
        log.info("Trajectory analysis of plain variables: the statistics only.");
    }
    let properties = Properties::read(&dir.join("md_properties.csv"))?;
    analyze_properties(&properties, settings, &dir.join(ANALYSIS_DIR), log)?;
    Ok(())
}

/// The analysis at the end of a run of the files in `dir`.
pub fn analyze_run(
    config: &MdmainConfig,
    trajectory: &Path,
    velocities: Option<PathBuf>,
    dir: &Path,
    log: &Logger,
) -> MoptResult<()> {
    let Some(settings) = &config.analysis else {
        return Ok(());
    };
    let mut settings = settings.clone();
    if let Some(e) = config.md.exchange {
        settings.spectra.exchange = (e.slabs, e.axis);
    }
    let properties = dir.join("md_properties.csv");
    let stem = trajectory
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix("_traj.xyz"))
        .unwrap_or_default();
    let files = RunFiles {
        trajectory: trajectory.to_path_buf(),
        velocities: velocities.filter(|p| p.is_file()),
        properties: properties.is_file().then_some(properties),
        series: RunFiles::series_of(dir, stem),
    };
    analyze_files(&files, &settings, &dir.join(ANALYSIS_DIR), log)?;
    Ok(())
}
