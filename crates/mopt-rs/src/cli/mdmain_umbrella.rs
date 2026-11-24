//! `mdmain -umbrella KEY=VALUE ...`: umbrella sampling along a collective variable written in
//! the expression language of `-ubias` (atomic units; `ang` and `deg` convert), with the
//! free-energy profile by WHAM and MBAR.
//!
//! Keys: `cv` (the variable, e.g. `dist(1,2)`), `centers` (a list) or `from`, `to`, `windows`,
//! `k` (force constant, Hartree per unit of the variable squared), `equilibration` (steps of
//! every window that are not sampled; a fifth of `-time` without it), `bins` (50), `period`
//! (of an angle-like variable, e.g. `2*pi`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use crate::core_infra::log::{python_float, Logger};
use crate::core_infra::viz::{PlotFigure, PlotSeries, PlotStyle, Rgb};
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::md::{Cv, MdSettings};
use crate::pes::bias::ExpressionBias;
use crate::pes::calculator::Calculator;
use crate::pes::term::PesTerm;
use crate::workflow::md::MdObserver;
use crate::workflow::md_files::{MdFiles, MdLayout};
use crate::workflow::md_log::MdLog;
use crate::workflow::observers::StopFile;
use crate::workflow::umbrella::{run_umbrella, UmbrellaRun, UmbrellaSettings};

/// The settings of `-umbrella`.
#[derive(Debug, Clone, PartialEq)]
pub struct UmbrellaOption {
    /// The variable as written.
    pub cv: String,
    /// Centres of the windows.
    pub centers: Vec<f64>,
    /// Force constant.
    pub k: f64,
    /// Steps not sampled (`None`: a fifth of a window).
    pub equilibration: Option<usize>,
    /// Bins of the profile.
    pub bins: usize,
    /// Period of the variable.
    pub period: Option<f64>,
}

fn constant(key: &str, text: &str) -> MoptResult<f64> {
    ExpressionBias::parse(text)?
        .constant_value()
        .filter(|v| v.is_finite())
        .ok_or_else(|| MoptError::parse(format!("-umbrella {key}={text}: a number is needed")))
}

/// `-umbrella KEY=VALUE ...` (`None` without it).
pub fn parse_umbrella(tokens: &[String]) -> MoptResult<Option<UmbrellaOption>> {
    if tokens.is_empty() {
        return Ok(None);
    }
    let (mut cv, mut centers, mut k) = (None, None, None);
    let (mut from, mut to, mut windows) = (None, None, None);
    let (mut equilibration, mut bins, mut period) = (None, 50, None);
    for token in tokens {
        let (key, value) = token.split_once('=').ok_or_else(|| {
            MoptError::parse(format!("-umbrella expects KEY=VALUE, got '{token}'"))
        })?;
        let parse_count = |v: &str| {
            v.trim()
                .parse::<usize>()
                .map_err(|e| MoptError::parse(format!("-umbrella {key}={v}: {e}")))
        };
        match key.trim() {
            "cv" => cv = Some(value.trim().to_string()),
            "centers" => {
                centers = Some(
                    value
                        .split(',')
                        .map(|c| constant("centers", c))
                        .collect::<MoptResult<Vec<f64>>>()?,
                )
            }
            "from" => from = Some(constant("from", value)?),
            "to" => to = Some(constant("to", value)?),
            "windows" => windows = Some(parse_count(value)?),
            "k" => k = Some(constant("k", value)?),
            "equilibration" => equilibration = Some(parse_count(value)?),
            "bins" => bins = parse_count(value)?,
            "period" => period = Some(constant("period", value)?),
            other => {
                return Err(MoptError::parse(format!(
                    "-umbrella: unknown key '{other}' (cv, centers, from, to, windows, k, \
                     equilibration, bins, period)"
                )))
            }
        }
    }
    let cv = cv.ok_or_else(|| MoptError::parse("-umbrella needs cv=EXPRESSION"))?;
    let k = k.ok_or_else(|| MoptError::parse("-umbrella needs the force constant k="))?;
    let centers = match (centers, from, to, windows) {
        (Some(c), None, None, None) => c,
        (None, Some(a), Some(b), Some(n)) if n >= 2 => (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect(),
        _ => {
            return Err(MoptError::parse(
                "-umbrella needs centers=A,B,... or from=, to=, and windows= (two or more)",
            ))
        }
    };
    Ok(Some(UmbrellaOption {
        cv,
        centers,
        k,
        equilibration,
        bins,
        period,
    }))
}

/// Run the windows of `option` and write their files into `dir`: `umbrella_window_<k>/`
/// (the files of a run for every window), `umbrella_samples.csv`, `umbrella_windows.csv`,
/// `umbrella_profile.csv`, and `umbrella_profile.png`. Returns whether a `STOP` file
/// interrupted the run.
#[allow(clippy::too_many_arguments)]
pub fn run(
    option: &UmbrellaOption,
    md: &MdSettings,
    write_every: usize,
    calc: &dyn Calculator,
    terms: &[(String, Arc<dyn PesTerm>)],
    start: &MolecularSystem,
    stem: &str,
    dir: &Path,
    log: &Logger,
) -> MoptResult<UmbrellaRun> {
    // An expression with a period of its own, or any form of md::cv_spec.
    let cv = match option.period {
        Some(_) => Cv::expression(&option.cv, option.period)?,
        None => crate::md::cv_spec::parse_cv(&option.cv)?,
    };
    let settings = UmbrellaSettings {
        cv,
        centers: option.centers.clone(),
        k: option.k,
        equilibration: option.equilibration.unwrap_or(md.steps / 5),
        bins: option.bins,
    };
    log.info(&format!(
        "Umbrella sampling of {}: {} windows ({} to {}), k = {} Hartree, {} steps each, the \
         first {} not sampled",
        option.cv,
        settings.centers.len(),
        python_float(settings.centers[0]),
        python_float(settings.centers[settings.centers.len() - 1]),
        python_float(option.k),
        md.steps,
        settings.equilibration
    ));
    let mut make = |k: usize, u: &crate::md::Umbrella| -> MoptResult<Vec<Box<dyn MdObserver>>> {
        let sub = dir.join(format!("umbrella_window_{k}"));
        log.info(&format!("Window {k}: centre {}", python_float(u.center)));
        let files = MdFiles::new(
            &sub,
            stem,
            MdLayout::Atoms(Box::new(start.clone())),
            write_every,
        )?;
        let quiet = MdLog::new(log.clone(), md.steps.max(1), md.steps);
        Ok(vec![
            Box::new(files),
            Box::new(quiet),
            Box::new(StopFile::new(dir, log.clone())),
        ])
    };
    let run = run_umbrella(&settings, md, calc, terms, start, &mut make)?;
    write_results(&run, dir, log)?;
    Ok(run)
}

fn write_results(run: &UmbrellaRun, dir: &Path, log: &Logger) -> MoptResult<()> {
    let mut samples = String::from("window,center,value\n");
    let mut windows = String::from("window,center,k,samples,mean,standard_deviation\n");
    for (k, w) in run.windows.iter().enumerate() {
        for x in &w.samples {
            let _ = writeln!(samples, "{k},{},{x}", w.center);
        }
        let n = w.samples.len().max(1) as f64;
        let mean = w.samples.iter().sum::<f64>() / n;
        let sd = (w.samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
        let _ = writeln!(
            windows,
            "{k},{},{},{},{mean},{sd}",
            w.center,
            w.k,
            w.samples.len()
        );
    }
    std::fs::write(dir.join("umbrella_samples.csv"), samples)?;
    std::fs::write(dir.join("umbrella_windows.csv"), windows)?;
    let Some((w, m)) = &run.profiles else {
        return Ok(());
    };
    let kcal = Units::HARTREE_TO_KCAL_MOL;
    let mut text = String::from("value,free_energy_wham_kcal_mol,free_energy_mbar_kcal_mol\n");
    for b in 0..w.centers.len() {
        let _ = writeln!(
            text,
            "{},{},{}",
            w.centers[b],
            w.free_energy[b] * kcal,
            m.free_energy[b] * kcal
        );
    }
    std::fs::write(dir.join("umbrella_profile.csv"), text)?;
    let mut fig = PlotFigure::new(
        "Free-energy profile",
        "collective variable (a.u.)",
        "free energy [kcal/mol]",
    );
    for (label, p, color) in [("WHAM", w, Rgb::BLUE), ("MBAR", m, Rgb::RED)] {
        let (xs, ys): (Vec<f64>, Vec<f64>) = p
            .centers
            .iter()
            .zip(&p.free_energy)
            .filter(|(_, f)| f.is_finite())
            .map(|(x, f)| (*x, f * kcal))
            .unzip();
        fig.add_series(PlotSeries {
            label: label.into(),
            xs,
            ys,
            color,
            style: PlotStyle::LinePoints,
        });
    }
    fig.save(dir.join("umbrella_profile.png"))?;
    let top = |p: &crate::analysis::free_energy::Profile| {
        p.free_energy
            .iter()
            .filter(|f| f.is_finite())
            .fold(0.0_f64, |a, f| a.max(*f))
            * kcal
    };
    log.info(&format!(
        "Free-energy profile: range {:.3} kcal/mol (WHAM, {} iterations), {:.3} kcal/mol (MBAR, \
         {} iterations); umbrella_profile.csv",
        top(w),
        w.iterations,
        top(m),
        m.iterations
    ));
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn the_windows_are_read() {
        let u = parse_umbrella(&s(&[
            "cv=dist(1,2)",
            "from=3.5*ang",
            "to=4.5*ang",
            "windows=3",
            "k=0.02",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(u.centers.len(), 3);
        assert!((u.centers[1] - 4.0 * Units::ANGSTROM_TO_BOHR).abs() < 1e-12);
        let c = parse_umbrella(&s(&[
            "cv=angle(1,2,3)",
            "centers=1.5,2.0",
            "k=0.1",
            "bins=20",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!((c.centers.len(), c.bins), (2, 20));
        assert!(parse_umbrella(&[]).unwrap().is_none());
        assert!(parse_umbrella(&s(&["cv=dist(1,2)", "k=0.1"])).is_err());
        assert!(parse_umbrella(&s(&["cv=dist(1,2)", "centers=1", "k=x"])).is_err());
    }
}
