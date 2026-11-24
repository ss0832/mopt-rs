//! The files of the trajectory analysis: one CSV file and one figure (PNG or SVG) or more
//! per analysis, and a summary of each in the log.
//!
//! | analysis   | files                                                              |
//! |------------|--------------------------------------------------------------------|
//! | statistics | `statistics.csv`, `blocking.csv`                                   |
//! | rdf        | `rdf.csv`, `rdf.<ext>`                                             |
//! | msd        | `msd.csv`, `diffusion.csv`, `msd.<ext>`                            |
//! | vacf       | `vacf.csv`, `vdos.csv`, `vacf.<ext>`, `vdos.<ext>`                 |
//! | rmsd       | `rmsd.csv`, `rmsf.csv`, `rmsd.<ext>`, `rmsf.<ext>`                 |
//! | geometry   | `geometry.csv`, `geometry_<k>.<ext>`                               |
//! | pca        | `pca.csv`, `pca_components.csv`, `pca.<ext>`                       |
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use super::md_analysis::{AnalysisSettings, Properties};
use crate::analysis::trajectory::{
    fluctuations, geometry_series, mean_squared_displacement, peaks, radial_distribution,
    statistics as stats_of, trajectory_pca, velocity_autocorrelation,
    vibrational_density_of_states, MdTrajectory, Statistics,
};
use crate::core_infra::log::Logger;
use crate::core_infra::viz::{ColoredPoints, PlotFigure, PlotSeries, PlotStyle, Rgb};
use crate::error::{MoptError, MoptResult};

/// Largest number of points drawn of a series.
const FIGURE_POINTS: usize = 4000;

/// Colors of the series of a figure, in turn.
const COLORS: [Rgb; 6] = [
    Rgb::BLUE,
    Rgb::RED,
    Rgb::EMERALD,
    Rgb::AMBER,
    Rgb::MAGENTA,
    Rgb::SLATE,
];

fn series(label: &str, xs: &[f64], ys: &[f64], k: usize) -> PlotSeries {
    let stride = xs.len().div_ceil(FIGURE_POINTS).max(1);
    PlotSeries {
        label: label.to_string(),
        xs: xs.iter().step_by(stride).copied().collect(),
        ys: ys.iter().step_by(stride).copied().collect(),
        color: COLORS[k % COLORS.len()],
        style: PlotStyle::LinePoints,
    }
}

fn figure(
    out: &Path,
    name: &str,
    s: &AnalysisSettings,
    (title, x, y): (&str, &str, &str),
    lines: Vec<PlotSeries>,
) -> MoptResult<PathBuf> {
    let mut fig = PlotFigure::new(title, x, y);
    for line in lines {
        fig.add_series(line);
    }
    let path = out.join(format!("{name}.{}", s.figures));
    fig.save(&path)?;
    Ok(path)
}

/// A CSV table: the column `x`, the column `all`, and one column per element.
fn table(
    (x_name, xs): (&str, &[f64]),
    (all_name, all): (&str, &[f64]),
    names: &[&str],
    columns: &[Vec<f64>],
) -> String {
    let mut text = format!("{x_name},{all_name}");
    for name in names {
        let _ = write!(text, ",{name}");
    }
    text.push('\n');
    for (k, x) in xs.iter().enumerate() {
        let _ = write!(text, "{x},{}", all[k]);
        for v in columns {
            let _ = write!(text, ",{}", v[k]);
        }
        text.push('\n');
    }
    text
}

fn write(out: &Path, name: &str, text: &str) -> MoptResult<PathBuf> {
    let path = out.join(name);
    fs::write(&path, text)?;
    Ok(path)
}

/// The columns of `md_properties.csv` with statistics, their units.
const PROPERTY_UNITS: [(&str, &str); 8] = [
    ("potential_energy", "hartree"),
    ("potential_energy_without_bias", "hartree"),
    ("kinetic_energy", "hartree"),
    ("total_energy", "hartree"),
    ("conserved_quantity", "hartree"),
    ("temperature_K", "K"),
    ("pressure_GPa", "GPa"),
    ("volume_angstrom3", "angstrom^3"),
];

fn summary_row(name: &str, unit: &str, st: &Statistics) -> String {
    format!(
        "{name},{unit},{},{},{},{},{},{},{},{},{}\n",
        st.count,
        st.mean,
        st.deviation,
        st.error,
        st.block_size,
        st.inefficiency,
        st.converged,
        st.min,
        st.max
    )
}

/// The series of the statistics: name, unit, values (the steps from `skip_fs` on).
fn series_of(
    traj: Option<&MdTrajectory>,
    properties: Option<&Properties>,
    s: &AnalysisSettings,
) -> Vec<(String, String, Vec<f64>)> {
    let mut series_list: Vec<(String, String, Vec<f64>)> = Vec::new();
    match properties {
        Some(p) => {
            let times = p.column("time_fs");
            for (name, unit) in PROPERTY_UNITS {
                let Some(values) = p.column(name) else {
                    continue;
                };
                let kept: Vec<f64> = values
                    .iter()
                    .enumerate()
                    .filter(|(k, v)| {
                        v.is_finite()
                            && times
                                .is_none_or(|t| t.get(*k).is_some_and(|t| *t >= s.skip_fs - 1e-9))
                    })
                    .map(|(_, v)| *v)
                    .collect();
                if kept.len() >= 2 {
                    series_list.push((name.to_string(), unit.to_string(), kept));
                }
            }
            // The collective variables of -cv (atomic units).
            for (name, values) in p.names.iter().zip(&p.columns) {
                if !name.starts_with("cv ") {
                    continue;
                }
                let kept: Vec<f64> = values
                    .iter()
                    .enumerate()
                    .filter(|(k, v)| {
                        v.is_finite()
                            && times
                                .is_none_or(|t| t.get(*k).is_some_and(|t| *t >= s.skip_fs - 1e-9))
                    })
                    .map(|(_, v)| *v)
                    .collect();
                if kept.len() >= 2 {
                    series_list.push((name.clone(), "a.u.".to_string(), kept));
                }
            }
        }
        None => {
            let energies: Option<Vec<f64>> = traj
                .map_or(&[][..], |t| t.systems())
                .iter()
                .map(|sys| {
                    sys.comment
                        .split_whitespace()
                        .find_map(|t| t.strip_prefix("energy="))
                        .and_then(|v| v.parse::<f64>().ok())
                })
                .collect();
            if let Some(e) = energies.filter(|e| e.len() >= 2) {
                series_list.push(("potential_energy_of_frames".into(), "hartree".into(), e));
            }
        }
    }
    series_list
}

/// Means, deviations, and block-average errors of the properties of the steps.
pub(super) fn statistics(
    traj: Option<&MdTrajectory>,
    properties: Option<&Properties>,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let series_list = series_of(traj, properties, s);
    if series_list.is_empty() {
        return Err(MoptError::config(
            "there is no md_properties.csv and the frames carry no energies",
        ));
    }
    let mut summary = String::from(
        "quantity,unit,samples,mean,standard_deviation,error_of_mean,block_size,\
         statistical_inefficiency,converged,min,max\n",
    );
    let mut blocking = String::from("quantity,block_size,blocks,error,error_of_error\n");
    for (name, unit, values) in &series_list {
        let st = stats_of(values)?;
        summary.push_str(&summary_row(name, unit, &st));
        for l in &st.levels {
            let _ = writeln!(
                blocking,
                "{name},{},{},{},{}",
                l.block_size, l.blocks, l.error, l.error_of_error
            );
        }
        log.info(&format!(
            "  {name:<30} {:>18.10} +- {:<12.4e} {unit} (deviation {:.4e}, {} samples, block {}{})",
            st.mean,
            st.error,
            st.deviation,
            st.count,
            st.block_size,
            if st.converged {
                ""
            } else {
                ", too short for its correlation time"
            }
        ));
    }
    Ok(vec![
        write(out, "statistics.csv", &summary)?,
        write(out, "blocking.csv", &blocking)?,
    ])
}

/// Radial distribution functions with running coordination numbers.
pub(super) fn rdf(
    traj: &MdTrajectory,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let rdf = radial_distribution(traj, &s.rdf)?;
    let label = |p: &crate::analysis::trajectory::RdfPair| {
        format!("{}-{}", p.first.symbol(), p.second.symbol())
    };
    let mut text = String::from("r_angstrom");
    for p in &rdf.pairs {
        let _ = write!(text, ",g_{0},n_{0}", label(p));
    }
    text.push('\n');
    for (k, r) in rdf.r.iter().enumerate() {
        let _ = write!(text, "{r}");
        for p in &rdf.pairs {
            let _ = write!(text, ",{},{}", p.g[k], p.coordination[k]);
        }
        text.push('\n');
    }
    let kind = if rdf.periodic {
        "g(r)"
    } else {
        "distribution of distances (1/ang.)"
    };
    for p in &rdf.pairs {
        let found = peaks(&rdf.r, &p.g, 0.2);
        let mut by_r: Vec<(f64, f64)> = found.into_iter().take(4).collect();
        by_r.sort_by(|a, b| a.0.total_cmp(&b.0));
        let list: Vec<String> = by_r
            .iter()
            .map(|(r, g)| format!("{r:.3} ({g:.2})"))
            .collect();
        log.info(&format!(
            "  {} {kind}: peaks at {} ang.",
            label(p),
            list.join(", ")
        ));
    }
    let lines = rdf
        .pairs
        .iter()
        .enumerate()
        .map(|(k, p)| series(&label(p), &rdf.r, &p.g, k))
        .collect();
    Ok(vec![
        write(out, "rdf.csv", &text)?,
        figure(
            out,
            "rdf",
            s,
            ("Radial distribution", "r (ang.)", kind),
            lines,
        )?,
    ])
}

/// Mean squared displacements and diffusion coefficients.
pub(super) fn msd(
    traj: &MdTrajectory,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let m = mean_squared_displacement(traj, &s.msd)?;
    let columns: Vec<String> = m
        .species
        .iter()
        .map(|e| format!("msd_{}_angstrom2", e.symbol()))
        .collect();
    let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
    let text = table(
        ("lag_fs", &m.lags),
        ("msd_all_angstrom2", &m.total),
        &columns,
        &m.by_species,
    );
    let mut d = String::from(
        "atoms,D_angstrom2_per_ps,D_cm2_per_s,fit_from_fs,fit_to_fs,intercept_angstrom2,r_squared\n",
    );
    let names: Vec<String> = std::iter::once("all".to_string())
        .chain(m.species.iter().map(|e| e.symbol().to_string()))
        .collect();
    for (name, diff) in names
        .iter()
        .zip(std::iter::once(&m.total_diffusion).chain(&m.diffusion))
    {
        let _ = writeln!(
            d,
            "{name},{},{},{},{},{},{}",
            diff.coefficient,
            diff.cm2_per_s(),
            diff.window.0,
            diff.window.1,
            diff.intercept,
            diff.r_squared
        );
        log.info(&format!(
            "  diffusion coefficient ({name}): {:.6e} ang.^2/ps = {:.6e} cm^2/s (fit {:.1}-{:.1} fs, R^2 {:.4})",
            diff.coefficient,
            diff.cm2_per_s(),
            diff.window.0,
            diff.window.1,
            diff.r_squared
        ));
    }
    let mut lines = vec![series("all", &m.lags, &m.total, 0)];
    for (k, (e, v)) in m.species.iter().zip(&m.by_species).enumerate() {
        if m.species.len() > 1 {
            lines.push(series(e.symbol(), &m.lags, v, k + 1));
        }
    }
    Ok(vec![
        write(out, "msd.csv", &text)?,
        write(out, "diffusion.csv", &d)?,
        figure(
            out,
            "msd",
            s,
            ("Mean squared displacement", "lag (fs)", "MSD (ang.^2)"),
            lines,
        )?,
    ])
}

/// Velocity autocorrelation and vibrational density of states.
pub(super) fn vacf(
    traj: &MdTrajectory,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let c = velocity_autocorrelation(traj, s.correlation_share)?;
    let g = vibrational_density_of_states(&c, &s.vdos)?;
    let names: Vec<&str> = c.species.iter().map(|e| e.symbol()).collect();
    let vt = table(
        ("lag_fs", &c.lags),
        ("all", &c.total),
        &names,
        &c.by_species,
    );
    let gt = table(
        ("wavenumber_cm-1", &g.wavenumbers),
        ("all", &g.total),
        &names,
        &g.by_species,
    );
    let mut found = peaks(&g.wavenumbers, &g.total, 0.05);
    found.truncate(12);
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    let list: Vec<String> = found.iter().map(|(nu, _)| format!("{nu:.0}")).collect();
    log.info(&format!(
        "  vibrational density of states: peaks at {} cm^-1 (resolution about {:.0} cm^-1, \
         Nyquist {:.0} cm^-1)",
        list.join(", "),
        g.resolution,
        g.nyquist
    ));
    let mut vl = vec![series("all", &c.lags, &c.total, 0)];
    let mut gl = vec![series("all", &g.wavenumbers, &g.total, 0)];
    if c.species.len() > 1 {
        for (k, e) in c.species.iter().enumerate() {
            vl.push(series(e.symbol(), &c.lags, &c.by_species[k], k + 1));
            gl.push(series(e.symbol(), &g.wavenumbers, &g.by_species[k], k + 1));
        }
    }
    Ok(vec![
        write(out, "vacf.csv", &vt)?,
        write(out, "vdos.csv", &gt)?,
        figure(
            out,
            "vacf",
            s,
            ("Velocity autocorrelation", "lag (fs)", "C(t) / C(0)"),
            vl,
        )?,
        figure(
            out,
            "vdos",
            s,
            (
                "Vibrational density of states",
                "wavenumber (cm^-1)",
                "states per cm^-1",
            ),
            gl,
        )?,
    ])
}

/// RMSD to the first frame and RMSF of every atom.
pub(super) fn rmsd(
    traj: &MdTrajectory,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let f = fluctuations(traj)?;
    let mut rt = String::from("time_fs,rmsd_angstrom\n");
    for (t, r) in traj.times().iter().zip(&f.rmsd) {
        let _ = writeln!(rt, "{t},{r}");
    }
    let mut ft = String::from("atom,element,rmsf_angstrom\n");
    for (i, (e, r)) in traj.elements().iter().zip(&f.rmsf).enumerate() {
        let _ = writeln!(ft, "{},{},{r}", i + 1, e.symbol());
    }
    let largest = f
        .rmsf
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or((0, 0.0), |(i, r)| (i + 1, *r));
    log.info(&format!(
        "  RMSD to the first frame: last {:.4} ang., largest {:.4} ang.; largest RMSF {:.4} ang. (atom {})",
        f.rmsd.last().copied().unwrap_or(0.0),
        f.rmsd.iter().copied().fold(0.0, f64::max),
        largest.1,
        largest.0
    ));
    let atoms: Vec<f64> = (1..=f.rmsf.len()).map(|i| i as f64).collect();
    let mut stems = series("RMSF", &atoms, &f.rmsf, 0);
    stems.style = PlotStyle::Stem;
    Ok(vec![
        write(out, "rmsd.csv", &rt)?,
        write(out, "rmsf.csv", &ft)?,
        figure(
            out,
            "rmsd",
            s,
            ("RMSD to the first frame", "time (fs)", "RMSD (ang.)"),
            vec![series("RMSD", traj.times(), &f.rmsd, 0)],
        )?,
        figure(out, "rmsf", s, ("RMSF", "atom", "RMSF (ang.)"), vec![stems])?,
    ])
}

/// Distances, angles, dihedral angles of the chosen atoms.
pub(super) fn geometry(
    traj: &MdTrajectory,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    if s.measures.is_empty() {
        log.info("  geometry: no atoms chosen (atoms=1,2;1,2,3 for a distance and an angle).");
        return Ok(Vec::new());
    }
    let values = geometry_series(traj, &s.measures)?;
    let mut text = String::from("time_fs");
    for m in &s.measures {
        let _ = write!(text, ",{}", m.name());
    }
    text.push('\n');
    for (k, t) in traj.times().iter().enumerate() {
        let _ = write!(text, "{t}");
        for v in &values {
            let _ = write!(text, ",{}", v[k]);
        }
        text.push('\n');
    }
    let mut files = vec![write(out, "geometry.csv", &text)?];
    for (k, (m, v)) in s.measures.iter().zip(&values).enumerate() {
        if v.len() >= 2 {
            let st = stats_of(v)?;
            log.info(&format!(
                "  {}: mean {:.6} +- {:.2e} (deviation {:.4e})",
                m.name(),
                st.mean,
                st.error,
                st.deviation
            ));
        }
        files.push(figure(
            out,
            &format!("geometry_{}", k + 1),
            s,
            (&m.name(), "time (fs)", &m.name()),
            vec![series(&m.name(), traj.times(), v, k)],
        )?);
    }
    Ok(files)
}

/// The projection on the principal components of the motion.
pub(super) fn pca(
    traj: &MdTrajectory,
    s: &AnalysisSettings,
    out: &Path,
    log: &Logger,
) -> MoptResult<Vec<PathBuf>> {
    let p = trajectory_pca(traj, s.components)?;
    let k = p.scores.ncols();
    let mut text = String::from("time_fs");
    for c in 0..k {
        let _ = write!(text, ",PC{}", c + 1);
    }
    text.push('\n');
    for (i, t) in traj.times().iter().enumerate() {
        let _ = write!(text, "{t}");
        for c in 0..k {
            let _ = write!(text, ",{}", p.scores[(i, c)]);
        }
        text.push('\n');
    }
    let mut ct = String::from("component,variance_amu_angstrom2,contribution\n");
    for c in 0..k {
        let _ = writeln!(ct, "{},{},{}", c + 1, p.variances[c], p.contributions[c]);
    }
    let shares: Vec<String> = p
        .contributions
        .iter()
        .map(|c| format!("{:.1} %", 100.0 * c))
        .collect();
    log.info(&format!(
        "  principal components of the motion: {}",
        shares.join(", ")
    ));
    let mut files = vec![
        write(out, "pca.csv", &text)?,
        write(out, "pca_components.csv", &ct)?,
    ];
    if k >= 2 {
        let mut fig = PlotFigure::new(
            "Principal components",
            format!("PC1 ({:.1} %)", 100.0 * p.contributions[0]),
            format!("PC2 ({:.1} %)", 100.0 * p.contributions[1]),
        );
        fig.set_colored_points(ColoredPoints {
            label: "time (fs)".into(),
            xs: (0..traj.len()).map(|i| p.scores[(i, 0)]).collect(),
            ys: (0..traj.len()).map(|i| p.scores[(i, 1)]).collect(),
            values: traj.times().to_vec(),
        });
        let path = out.join(format!("pca.{}", s.figures));
        fig.save(&path)?;
        files.push(path);
    }
    Ok(files)
}
