//! The correlation, infrared, and transport analyses of a run: autocorrelations and power
//! spectra of the properties, the chosen distances and angles, and the orientations of
//! chosen bonds; the infrared spectrum of the dipoles; the diffusion coefficients (Einstein
//! and Green-Kubo), the thermal conductivity (Green-Kubo of the heat current, reverse
//! non-equilibrium exchanges), and the shear viscosity (Green-Kubo of the off-diagonal
//! pressure), each with its error and written as CSV files and figures.
//!
//! | analysis | files |
//! |---|---|
//! | acf | `acf_properties.csv`, `power_properties.csv`, `acf_geometry.csv`, `orientation.csv`, `vacf_atoms.csv`, `acf.<ext>` |
//! | ir | `dipole_acf.csv`, `ir.csv`, `ir.<ext>` |
//! | transport | `transport.csv`, `green_kubo_<name>.csv`, `green_kubo_<name>.<ext>`, `nemd_profile.csv`, `nemd_profile.<ext>` |
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use super::md_analysis::{AnalysisSettings, Properties, Series};
use crate::analysis::correlation::spectrum::{cosine_transform, infrared as ir_spectrum};
use crate::analysis::correlation::transport::{
    diffusion_green_kubo, green_kubo, pascal_seconds, reverse_conductivity, watts_per_metre_kelvin,
    GreenKubo,
};
use crate::analysis::correlation::{autocorrelation, centered, orientational};
use crate::analysis::trajectory::{
    geometry_series, mean_squared_displacement, peaks, statistics, MdTrajectory,
};
use crate::core_infra::log::Logger;
use crate::core_infra::viz::{PlotFigure, PlotSeries, PlotStyle, Rgb};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};

const COLORS: [Rgb; 6] = [
    Rgb::BLUE,
    Rgb::RED,
    Rgb::EMERALD,
    Rgb::AMBER,
    Rgb::MAGENTA,
    Rgb::SLATE,
];

fn write(out: &Path, name: &str, text: &str) -> MoptResult<PathBuf> {
    let path = out.join(name);
    fs::write(&path, text)?;
    Ok(path)
}

fn figure(
    out: &Path,
    name: &str,
    s: &AnalysisSettings,
    (title, x, y): (&str, &str, &str),
    lines: &[(String, Vec<f64>, Vec<f64>)],
) -> MoptResult<PathBuf> {
    let mut fig = PlotFigure::new(title, x, y);
    for (k, (label, xs, ys)) in lines.iter().enumerate() {
        let stride = xs.len().div_ceil(4000).max(1);
        fig.add_series(PlotSeries {
            label: label.clone(),
            xs: xs.iter().step_by(stride).copied().collect(),
            ys: ys.iter().step_by(stride).copied().collect(),
            color: COLORS[k % COLORS.len()],
            style: PlotStyle::LinePoints,
        });
    }
    let path = out.join(format!("{name}.{}", s.figures));
    fig.save(&path)?;
    Ok(path)
}

/// A table with a first column and named columns.
fn table(first: (&str, &[f64]), columns: &[(String, Vec<f64>)]) -> String {
    let mut text = first.0.to_string();
    for (name, _) in columns {
        let _ = write!(text, ",{name}");
    }
    text.push('\n');
    for (k, x) in first.1.iter().enumerate() {
        let _ = write!(text, "{x}");
        for (_, c) in columns {
            let _ = write!(text, ",{}", c.get(k).copied().unwrap_or(f64::NAN));
        }
        text.push('\n');
    }
    text
}

/// The rows of `p` from `skip_fs` on: the times and the values of the columns `names`
/// (`None` when a column is missing or has gaps).
fn rows(p: &Properties, skip_fs: f64, names: &[&str]) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let times = p.column("time_fs")?;
    let keep: Vec<usize> = (0..times.len())
        .filter(|&k| times[k] >= skip_fs - 1e-9)
        .collect();
    let mut columns = Vec::new();
    for name in names {
        let c = p.column(name)?;
        let v: Vec<f64> = keep
            .iter()
            .map(|&k| c.get(k).copied().unwrap_or(f64::NAN))
            .collect();
        if v.len() < 4 || v.iter().any(|x| !x.is_finite()) {
            return None;
        }
        columns.push(v);
    }
    Some((keep.iter().map(|&k| times[k]).collect(), columns))
}

fn spacing(times: &[f64]) -> MoptResult<f64> {
    if times.len() < 2 {
        return Err(MoptError::config("a series of fewer than two steps"));
    }
    Ok((times[times.len() - 1] - times[0]) / (times.len() - 1) as f64)
}

fn normalized(c: &[f64]) -> Vec<f64> {
    let c0 = c.first().copied().unwrap_or(0.0);
    if c0 == 0.0 {
        return vec![0.0; c.len()];
    }
    c.iter().map(|v| v / c0).collect()
}

/// Integrated correlation time (fs) of a normalized correlation up to its first zero.
fn correlation_time(c: &[f64], dt: f64) -> f64 {
    let end = (1..c.len()).find(|&k| c[k] <= 0.0).unwrap_or(c.len());
    (1..end).map(|k| 0.5 * dt * (c[k - 1] + c[k])).sum()
}

/// Autocorrelations and power spectra of the properties, the measures, and the
/// orientations of bonds; the velocity autocorrelation of every atom.
pub(super) fn correlations(
    traj: &MdTrajectory,
    properties: Option<&Properties>,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    let share = s.correlation_share;
    if let Some(p) = properties {
        let names = [
            "potential_energy",
            "kinetic_energy",
            "total_energy",
            "temperature_K",
            "pressure_GPa",
            "volume_angstrom3",
        ];
        let mut acfs = Vec::new();
        let mut spectra = Vec::new();
        let mut lags = Vec::new();
        let mut grid = Vec::new();
        // The properties and the collective variables of -cv.
        let all: Vec<&str> = names
            .iter()
            .copied()
            .chain(
                p.names
                    .iter()
                    .map(String::as_str)
                    .filter(|n| n.starts_with("cv ")),
            )
            .collect();
        for name in all {
            let Some((times, cols)) = rows(p, s.skip_fs, &[name]) else {
                continue;
            };
            let dt = spacing(&times)?;
            let n = ((cols[0].len() as f64 * share) as usize).max(2);
            let c = autocorrelation(&centered(&cols[0]), n);
            if c[0] <= 0.0 {
                continue;
            }
            let spectrum = cosine_transform(&c, dt, s.spectra.window, s.spectra.padding, None)?;
            log.info(&format!(
                "  {name}: integrated correlation time {:.2} fs",
                correlation_time(&normalized(&c), dt)
            ));
            lags = (0..n).map(|k| k as f64 * dt).collect();
            grid = spectrum.wavenumbers.clone();
            acfs.push((name.to_string(), normalized(&c)));
            spectra.push((name.to_string(), spectrum.values));
        }
        if !acfs.is_empty() {
            files.push(write(
                out,
                "acf_properties.csv",
                &table(("lag_fs", &lags), &acfs),
            )?);
            files.push(write(
                out,
                "power_properties.csv",
                &table(("wavenumber_cm-1", &grid), &spectra),
            )?);
            let lines: Vec<(String, Vec<f64>, Vec<f64>)> = acfs
                .iter()
                .map(|(n, c)| (n.clone(), lags.clone(), c.clone()))
                .collect();
            files.push(figure(
                out,
                "acf",
                s,
                (
                    "Autocorrelations of the properties",
                    "lag (fs)",
                    "C(t) / C(0)",
                ),
                &lines,
            )?);
        }
    }
    files.extend(frame_correlations(traj, s, out, log)?);
    if files.is_empty() {
        return Err(MoptError::config(
            "nothing to correlate: no md_properties.csv, atoms=, bonds=, or velocities",
        ));
    }
    Ok(files)
}

/// The correlations of the frames: measures, bond orientations, velocities of the atoms.
fn frame_correlations(
    traj: &MdTrajectory,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    let Ok(dt) = traj.interval() else {
        return Ok(files);
    };
    let n = ((traj.len() as f64 * s.correlation_share) as usize).max(2);
    let lags: Vec<f64> = (0..n).map(|k| k as f64 * dt).collect();
    if !s.measures.is_empty() {
        let values = geometry_series(traj, &s.measures)?;
        let cols: Vec<(String, Vec<f64>)> = s
            .measures
            .iter()
            .zip(&values)
            .map(|(m, v)| (m.name(), normalized(&autocorrelation(&centered(v), n))))
            .collect();
        files.push(write(
            out,
            "acf_geometry.csv",
            &table(("lag_fs", &lags), &cols),
        )?);
    }
    if !s.bonds.is_empty() {
        let vectors: Vec<Vec<[f64; 3]>> = (0..traj.len())
            .map(|t| s.bonds.iter().map(|&(a, b)| traj.delta(t, a, b)).collect())
            .collect();
        let p1 = orientational(&vectors, 1, n)?;
        let p2 = orientational(&vectors, 2, n)?;
        log.info(&format!(
            "  bond orientations ({} bonds): correlation times {:.2} fs (P1), {:.2} fs (P2)",
            s.bonds.len(),
            correlation_time(&p1, dt),
            correlation_time(&p2, dt)
        ));
        let cols = vec![("P1".to_string(), p1), ("P2".to_string(), p2)];
        files.push(write(
            out,
            "orientation.csv",
            &table(("lag_fs", &lags), &cols),
        )?);
    }
    if let Some(v) = traj.velocities() {
        let masses = traj.masses();
        let cols: Vec<(String, Vec<f64>)> = (0..traj.n_atoms())
            .map(|i| {
                let mut c = vec![0.0; n];
                for k in 0..3 {
                    let x: Vec<f64> = v.iter().map(|row| row[i][k]).collect();
                    for (o, a) in c.iter_mut().zip(autocorrelation(&x, n)) {
                        *o += masses[i] * a;
                    }
                }
                (
                    format!("{}{}", traj.elements()[i].symbol(), i + 1),
                    normalized(&c),
                )
            })
            .collect();
        files.push(write(
            out,
            "vacf_atoms.csv",
            &table(("lag_fs", &lags), &cols),
        )?);
    }
    Ok(files)
}

/// Mean of the column `name` of `p` from `skip_fs` on.
fn mean_of(p: Option<&Properties>, skip_fs: f64, name: &str) -> Option<f64> {
    let (_, cols) = rows(p?, skip_fs, &[name])?;
    let v = &cols[0];
    Some(v.iter().sum::<f64>() / v.len() as f64)
}

/// The infrared spectrum of the dipoles.
pub(super) fn infrared(
    properties: Option<&Properties>,
    series: &Series,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let p = series.dipoles.as_ref().ok_or_else(|| {
        MoptError::config(
            "there are no dipoles (<stem>_dipoles.csv; the calculator must give atomic charges, \
             as gfnff and gfn0xtb do)",
        )
    })?;
    let (times, cols) = rows(p, s.skip_fs, &["mu_x", "mu_y", "mu_z"])
        .ok_or_else(|| MoptError::config("the dipoles have gaps or too few steps"))?;
    let dt = spacing(&times)?;
    let n = ((cols[0].len() as f64 * s.correlation_share) as usize).max(2);
    let mut c = vec![0.0; n];
    for x in &cols {
        for (o, v) in c.iter_mut().zip(autocorrelation(&centered(x), n)) {
            *o += v;
        }
    }
    let temperature = mean_of(properties, s.skip_fs, "temperature_K").unwrap_or(300.0);
    let sp = &s.spectra;
    let spectrum = ir_spectrum(
        &c,
        dt,
        temperature,
        sp.correction,
        sp.window,
        sp.padding,
        None,
    )?;
    let mut found = peaks(&spectrum.wavenumbers, &spectrum.values, 0.05);
    found.truncate(12);
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    let list: Vec<String> = found.iter().map(|(nu, _)| format!("{nu:.0}")).collect();
    log.info(&format!(
        "  infrared spectrum ({:?} correction at {temperature:.1} K, window {}, padding {}): \
         peaks at {} cm^-1",
        sp.correction,
        sp.window.name(),
        sp.padding,
        list.join(", ")
    ));
    let lags: Vec<f64> = (0..n).map(|k| k as f64 * dt).collect();
    Ok(vec![
        write(
            out,
            "dipole_acf.csv",
            &table(("lag_fs", &lags), &[("C_mu_e2bohr2".to_string(), c)]),
        )?,
        write(
            out,
            "ir.csv",
            &table(
                ("wavenumber_cm-1", &spectrum.wavenumbers),
                &[("intensity".to_string(), spectrum.values.clone())],
            ),
        )?,
        figure(
            out,
            "ir",
            s,
            (
                "Infrared spectrum",
                "wavenumber (cm^-1)",
                "intensity (arb. units)",
            ),
            &[("IR".to_string(), spectrum.wavenumbers, spectrum.values)],
        )?,
    ])
}

/// One row of `transport.csv` and its log line.
fn summary(
    rows: &mut String,
    log: &Logger,
    quantity: &str,
    method: &str,
    g: &GreenKubo,
    unit: &str,
) {
    let _ = writeln!(
        rows,
        "{quantity},{method},{},{},{unit},{},{},{}",
        g.value, g.value_error, g.lags[g.plateau], g.plateau_mean, g.crossed
    );
    log.info(&format!(
        "  {quantity} ({method}): {:.6e} +- {:.2e} {unit} (read at {:.1} fs, the first zero of \
         the correlation{}; mean from there to twice that lag {:.6e})",
        g.value,
        g.value_error,
        g.lags[g.plateau],
        if g.crossed {
            ""
        } else {
            " not reached: the last lag"
        },
        g.plateau_mean
    ));
}

fn gk_files(
    out: &Path,
    s: &AnalysisSettings,
    name: &str,
    g: &GreenKubo,
    unit: &str,
) -> MoptResult<Vec<PathBuf>> {
    let cols = vec![
        ("correlation".to_string(), g.correlation.clone()),
        (format!("integral_{unit}"), g.integral.clone()),
        ("error".to_string(), g.error.clone()),
    ];
    Ok(vec![
        write(
            out,
            &format!("green_kubo_{name}.csv"),
            &table(("lag_fs", &g.lags), &cols),
        )?,
        figure(
            out,
            &format!("green_kubo_{name}"),
            s,
            (&format!("Green-Kubo: {name}"), "lag (fs)", unit),
            &[
                ("integral".to_string(), g.lags.clone(), g.integral.clone()),
                (
                    "integral + error".to_string(),
                    g.lags.clone(),
                    g.integral
                        .iter()
                        .zip(&g.error)
                        .map(|(a, b)| a + b)
                        .collect(),
                ),
                (
                    "integral - error".to_string(),
                    g.lags.clone(),
                    g.integral
                        .iter()
                        .zip(&g.error)
                        .map(|(a, b)| a - b)
                        .collect(),
                ),
            ],
        )?,
    ])
}

/// The number of lags of a Green-Kubo integral of `n` samples `dt` apart.
fn gk_lags(s: &AnalysisSettings, n: usize, dt: f64) -> usize {
    match s.spectra.gk_lag_fs {
        Some(t) => ((t / dt) as usize).max(2),
        None => n / s.spectra.gk_blocks.max(2) / 2,
    }
}

/// Diffusion (Einstein and Green-Kubo), thermal conductivity, and shear viscosity.
pub(super) fn transport(
    traj: &MdTrajectory,
    properties: Option<&Properties>,
    series: &Series,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut text = String::from(
        "quantity,method,value,error,unit,read_at_fs,plateau_mean,correlation_crossed_zero\n",
    );
    let blocks = s.spectra.gk_blocks;
    let temperature = mean_of(properties, s.skip_fs, "temperature_K");
    // Diffusion: Einstein from the mean squared displacement, Green-Kubo from the velocities.
    if traj.velocities().is_some() && traj.len() > 40 {
        let dt = traj.interval()?;
        let all: Vec<usize> = (0..traj.n_atoms()).collect();
        let g = diffusion_green_kubo(traj, &all, gk_lags(s, traj.len(), dt), blocks)?;
        summary(
            &mut text,
            log,
            "self-diffusion",
            "Green-Kubo",
            &g,
            "angstrom2/ps",
        );
        files.extend(gk_files(out, s, "diffusion", &g, "angstrom2/ps")?);
        if let Ok(m) = mean_squared_displacement(traj, &s.msd) {
            let parts: Vec<f64> = (0..5)
                .filter_map(|b| {
                    let range = b * traj.len() / 5..(b + 1) * traj.len() / 5;
                    let part = MdTrajectory::new(
                        traj.systems()[range.clone()].to_vec(),
                        traj.times()[range].to_vec(),
                    )
                    .ok()?;
                    mean_squared_displacement(&part, &s.msd)
                        .ok()
                        .map(|m| m.total_diffusion.coefficient)
                })
                .collect();
            let error = statistics(&parts).map_or(f64::NAN, |st| st.naive_error);
            let _ = writeln!(
                text,
                "self-diffusion,Einstein,{},{error},angstrom2/ps,,,",
                m.total_diffusion.coefficient
            );
            log.info(&format!(
                "  self-diffusion (Einstein, mean squared displacement): {:.6e} +- {:.2e} \
                 angstrom2/ps (error from five parts of the run)",
                m.total_diffusion.coefficient, error
            ));
        }
    }
    files.extend(conductivity_and_viscosity(
        series,
        temperature,
        s,
        &mut text,
        out,
        log,
    )?);
    if let Some(r) = reverse(traj, properties, s, out, log, &mut text)? {
        files.extend(r);
    }
    if files.is_empty() {
        return Err(MoptError::config(
            "no transport coefficient: the run has no velocities, heat current (-md_opt \
             heat_flux=on), pressure tensor (-md_opt stress=on), or exchanges (-md_opt exchange=N)",
        ));
    }
    files.push(write(out, "transport.csv", &text)?);
    Ok(files)
}

fn conductivity_and_viscosity(
    series: &Series,
    temperature: Option<f64>,
    s: &AnalysisSettings,
    text: &mut String,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    let kt2 = |t: f64| Units::BOLTZMANN_HARTREE_PER_K * t * t;
    if let (Some(p), Some(t)) = (&series.heat_flux, temperature) {
        if let Some((times, cols)) = rows(p, s.skip_fs, &["J_x", "J_y", "J_z", "volume_bohr3"]) {
            let dt = spacing(&times)?;
            let volume = cols[3].iter().sum::<f64>() / cols[3].len() as f64;
            let components: Vec<Vec<f64>> = cols[..3].iter().map(|c| centered(c)).collect();
            let prefactor = watts_per_metre_kelvin() / (volume * kt2(t));
            let g = green_kubo(
                &components,
                dt,
                prefactor,
                gk_lags(s, times.len(), dt),
                s.spectra.gk_blocks,
            )?;
            summary(
                text,
                log,
                "thermal conductivity",
                "Green-Kubo",
                &g,
                "W/(m K)",
            );
            files.extend(gk_files(out, s, "thermal_conductivity", &g, "W/(m K)")?);
        }
    }
    if let (Some(p), Some(t)) = (&series.pressure, temperature) {
        if let Some((times, cols)) = rows(p, s.skip_fs, &["P_xy", "P_xz", "P_yz", "volume_bohr3"]) {
            let dt = spacing(&times)?;
            let volume = cols[3].iter().sum::<f64>() / cols[3].len() as f64;
            let components: Vec<Vec<f64>> = cols[..3].iter().map(|c| centered(c)).collect();
            let prefactor = volume / (Units::BOLTZMANN_HARTREE_PER_K * t) * pascal_seconds();
            let g = green_kubo(
                &components,
                dt,
                prefactor,
                gk_lags(s, times.len(), dt),
                s.spectra.gk_blocks,
            )?;
            summary(text, log, "shear viscosity", "Green-Kubo", &g, "Pa s");
            files.extend(gk_files(out, s, "shear_viscosity", &g, "Pa s")?);
        }
    }
    Ok(files)
}

/// The thermal conductivity of the velocity exchanges, when the run made them.
fn reverse(
    traj: &MdTrajectory,
    properties: Option<&Properties>,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
    text: &mut String,
) -> MoptResult<Option<Vec<PathBuf>>> {
    let Some((times, cols)) =
        properties.and_then(|p| rows(p, s.skip_fs, &["heat_exchanged_hartree"]))
    else {
        return Ok(None);
    };
    let exchanged = cols[0][cols[0].len() - 1] - cols[0][0];
    if exchanged.is_nan() || exchanged <= 0.0 || traj.velocities().is_none() {
        return Ok(None);
    }
    let duration = times[times.len() - 1] - times[0];
    let (slabs, axis) = s.spectra.exchange;
    let r = reverse_conductivity(traj, exchanged, duration, axis, slabs, s.spectra.gk_blocks)?;
    let _ = writeln!(
        text,
        "thermal conductivity,reverse non-equilibrium,{},{},W/(m K),,,",
        r.conductivity, r.conductivity_error
    );
    log.info(&format!(
        "  thermal conductivity (reverse non-equilibrium, {slabs} slabs): {:.6e} +- {:.2e} W/(m K) \
         (flux {:.4e} W/m^2, gradient {:.4} K/angstrom)",
        r.conductivity, r.conductivity_error, r.flux, r.gradient
    ));
    let cols = vec![
        ("temperature_K".to_string(), r.temperatures.clone()),
        ("error_K".to_string(), r.temperature_errors.clone()),
    ];
    Ok(Some(vec![
        write(
            out,
            "nemd_profile.csv",
            &table(("centre_angstrom", &r.centres), &cols),
        )?,
        figure(
            out,
            "nemd_profile",
            s,
            (
                "Temperature profile",
                "position along the axis (ang.)",
                "T (K)",
            ),
            &[("T".to_string(), r.centres, r.temperatures)],
        )?,
    ]))
}
