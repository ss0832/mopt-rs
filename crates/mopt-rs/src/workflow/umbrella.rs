//! Umbrella sampling: molecular dynamics in windows of harmonic restraints on a collective
//! variable, one window after the other (each from the last structure and velocities of the
//! one before), and the free-energy profile of the samples by WHAM and MBAR
//! ([`crate::analysis::free_energy`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::Col;

use super::md::{run_md, FnMdObserver, MdObserver, MoleculeMdSurface};
use crate::analysis::free_energy::{mbar, wham, Profile, ProfileSettings, Window};
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::md::{
    Cv, InitialVelocities, MdSettings, MdSpace, MdStepReport, MolecularDynamics, Umbrella,
};
use crate::pes::calculator::Calculator;
use crate::pes::term::PesTerm;

/// Settings of umbrella sampling.
#[derive(Debug, Clone)]
pub struct UmbrellaSettings {
    /// The collective variable.
    pub cv: Cv,
    /// Centres of the windows, in their order.
    pub centers: Vec<f64>,
    /// Force constant of the restraints (Hartree per unit of the variable squared).
    pub k: f64,
    /// Steps at the start of every window that are not sampled.
    pub equilibration: usize,
    /// Bins of the profile.
    pub bins: usize,
}

/// What umbrella sampling gave.
#[derive(Debug, Clone)]
pub struct UmbrellaRun {
    /// The samples of the windows that were run.
    pub windows: Vec<Window>,
    /// Profiles (`None` when the run was interrupted).
    pub profiles: Option<(Profile, Profile)>,
    /// Whether an observer stopped the run.
    pub interrupted: bool,
    /// The structure and velocities at the end.
    pub last: Option<(Col<f64>, Col<f64>)>,
}

/// Run the windows of `settings` with the dynamics `md` (the steps of one window) on `calc`
/// with the bias potentials `terms`, from `start`. `observers(k, umbrella)` gives the
/// further observers of window `k` (files, log, a `STOP` file).
pub fn run_umbrella(
    settings: &UmbrellaSettings,
    md: &MdSettings,
    calc: &dyn Calculator,
    terms: &[(String, Arc<dyn PesTerm>)],
    start: &MolecularSystem,
    observers: &mut dyn FnMut(usize, &Umbrella) -> MoptResult<Vec<Box<dyn MdObserver>>>,
) -> MoptResult<UmbrellaRun> {
    if settings.centers.is_empty() {
        return Err(MoptError::config("umbrella sampling needs windows"));
    }
    if settings.equilibration >= md.steps {
        return Err(MoptError::config(
            "the equilibration of a window must be shorter than the window (-time)",
        ));
    }
    let mut x = start.coords_flat_bohr();
    let mut v: Option<Col<f64>> = None;
    let mut windows = Vec::new();
    for (k, &center) in settings.centers.iter().enumerate() {
        let umbrella = Umbrella::new(settings.cv.clone(), center, settings.k)?;
        let mut window_terms = terms.to_vec();
        window_terms.push((
            format!("umbrella_{k}"),
            Arc::new(umbrella.clone()) as Arc<dyn PesTerm>,
        ));
        let mut window = md.clone();
        window.seed = md.seed.wrapping_add(k as u64);
        if let Some(previous) = &v {
            window.velocities = InitialVelocities::Given(previous.clone());
        }
        let first = start.with_flat_coords_bohr(&x)?;
        let surface = MoleculeMdSurface::with_terms(calc, first.clone(), window_terms)?;
        let space = MdSpace::atoms(&start.masses_amu())?;
        let mut dynamics = MolecularDynamics::new(window, space, x.clone(), Some(first.clone()))?;
        let mut samples = Vec::new();
        let mut last = None;
        let cv = &settings.cv;
        let skip = settings.equilibration;
        let mut collect = FnMdObserver::new(|r: &MdStepReport| {
            if r.step >= skip {
                samples.push(cv.value(&first.with_flat_coords_bohr(&r.x)?)?);
            }
            last = Some((r.x.clone(), r.v.clone()));
            Ok(true)
        });
        let mut further = observers(k, &umbrella)?;
        let end = {
            let mut all: Vec<&mut dyn MdObserver> = vec![&mut collect];
            for o in further.iter_mut() {
                all.push(o.as_mut());
            }
            run_md(&mut dynamics, &surface, &mut all)?
        };
        if let Some((lx, lv)) = last.clone() {
            x = lx;
            v = Some(lv);
        }
        windows.push(Window {
            center,
            k: settings.k,
            samples,
        });
        if end.is_none() {
            return Ok(UmbrellaRun {
                windows,
                profiles: None,
                interrupted: true,
                last,
            });
        }
    }
    let kt = Units::BOLTZMANN_HARTREE_PER_K * md.temperature;
    let analysis = ProfileSettings {
        bins: settings.bins,
        period: settings.cv.period(),
        ..ProfileSettings::at(kt)
    };
    let profiles = (wham(&windows, &analysis)?, mbar(&windows, &analysis)?);
    Ok(UmbrellaRun {
        windows,
        profiles: Some(profiles),
        interrupted: false,
        last: v.map(|v| (x, v)),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{AtomId, SpinCharge};
    use crate::io::xyz::parse_xyz_str;
    use crate::pes::native::LennardJones;

    #[test]
    fn the_windows_follow_each_other_and_give_a_profile() {
        let sys = parse_xyz_str(
            "2\n\nAr 0.0 0.0 0.0\nAr 3.8 0.0 0.0\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let cv = Cv::distance(AtomId::new(0), AtomId::new(1));
        let a = Units::ANGSTROM_TO_BOHR;
        let settings = UmbrellaSettings {
            cv,
            centers: (0..5).map(|i| (3.6 + 0.2 * i as f64) * a).collect(),
            k: 0.02,
            equilibration: 100,
            bins: 10,
        };
        let md = MdSettings {
            time_step: 10.0 * Units::FS_TO_AU_TIME,
            steps: 600,
            temperature: 40.0,
            thermostat: "langevin".to_string(),
            integrator: "baoab".to_string(),
            seed: 2,
            ..MdSettings::default()
        };
        let mut windows_seen = Vec::new();
        let run = run_umbrella(&settings, &md, &LennardJones, &[], &sys, &mut |k, u| {
            windows_seen.push((k, u.center));
            Ok(Vec::new())
        })
        .unwrap();
        assert_eq!(windows_seen.len(), 5);
        assert_eq!(run.windows[0].samples.len(), 501);
        let (w, m) = run.profiles.unwrap();
        // Each window samples around its centre.
        for window in &run.windows {
            let mean = window.samples.iter().sum::<f64>() / window.samples.len() as f64;
            assert!(
                (mean - window.center).abs() < 0.3 * a,
                "{mean} {}",
                window.center
            );
        }
        assert!(w.free_energy.contains(&0.0) && m.free_energy.contains(&0.0));
    }
}
