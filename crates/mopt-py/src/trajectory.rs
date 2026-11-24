//! Trajectory analysis from Python: [`TrajectoryAnalysis`] over frames in memory (or read
//! from the files of a run), the block-average statistics of a series, and the analysis of
//! a run directory as `mdmain -analyze_only` makes it.
//!
//! Positions and cells are in Angstrom, velocities in Angstrom per fs, times in fs; atoms
//! are counted from 0.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;

use faer::Col;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::analysis::correlation::spectrum::{self, QuantumCorrection, Window};
use mopt_rs::analysis::correlation::transport::{self, diffusion_green_kubo, GreenKubo};
use mopt_rs::analysis::correlation::{self, orientational};
use mopt_rs::analysis::trajectory::{
    fluctuations, geometry_series, mean_squared_displacement, radial_distribution, statistics,
    trajectory_pca, velocity_autocorrelation, vibrational_density_of_states, Diffusion,
    MdTrajectory, Measure, MsdSettings, RdfSettings, VdosSettings,
};
use mopt_rs::cli::mdmain_analysis::{output_dir, parse_analysis};
use mopt_rs::core_infra::log::Logger;
use mopt_rs::domain::{positions, AtomId, Vector3};
use mopt_rs::domain::{Angstrom, Element, MolecularSystem, PeriodicCell, SpinCharge, Units};
use mopt_rs::workflow::md_analysis::{analyze_files, read_trajectory, RunFiles};

use super::err;

fn cell(values: &[f64]) -> PyResult<PeriodicCell> {
    if values.len() != 9 {
        return Err(PyValueError::new_err(
            "a cell has 9 numbers (the vectors a, b, c in Angstrom)",
        ));
    }
    let v =
        |k: usize| Vector3::<Angstrom>::new([values[3 * k], values[3 * k + 1], values[3 * k + 2]]);
    PeriodicCell::from_lattice_vectors(v(0), v(1), v(2), [true, true, true]).map_err(err)
}

/// Frames of a trajectory and the analyses of `mdmain -analysis` on them.
#[pyclass(name = "TrajectoryAnalysis", module = "mopt_rs._core")]
pub struct PyTrajectory {
    traj: MdTrajectory,
}

fn diffusion<'py>(py: Python<'py>, d: &Diffusion) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("D_angstrom2_per_ps", d.coefficient)?;
    out.set_item("D_cm2_per_s", d.cm2_per_s())?;
    out.set_item("window_fs", d.window)?;
    out.set_item("intercept", d.intercept)?;
    out.set_item("r_squared", d.r_squared)?;
    Ok(out)
}

#[pymethods]
impl PyTrajectory {
    /// Frames of the atoms `symbols`: `positions_angstrom` (one list of `3 n` per frame) at
    /// `times_fs`, with `cells_angstrom` (one cell of 9 numbers for all frames, or one per
    /// frame) and `velocities_angstrom_per_fs` (one list of `3 n` per frame) when there are.
    #[new]
    #[pyo3(signature = (symbols, positions_angstrom, times_fs, *, cells_angstrom = None, velocities_angstrom_per_fs = None))]
    fn new(
        symbols: Vec<String>,
        positions_angstrom: Vec<Vec<f64>>,
        times_fs: Vec<f64>,
        cells_angstrom: Option<Vec<Vec<f64>>>,
        velocities_angstrom_per_fs: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        let elements = symbols
            .iter()
            .map(|s| Element::parse_token(s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        let n = elements.len();
        let cells: Vec<Option<PeriodicCell>> = match &cells_angstrom {
            None => vec![None; positions_angstrom.len()],
            Some(c) if c.len() == 1 => vec![Some(cell(&c[0])?); positions_angstrom.len()],
            Some(c) if c.len() == positions_angstrom.len() => c
                .iter()
                .map(|v| cell(v).map(Some))
                .collect::<PyResult<_>>()?,
            Some(_) => {
                return Err(PyValueError::new_err(
                    "cells_angstrom: one cell for all frames or one per frame",
                ))
            }
        };
        let mut systems = Vec::with_capacity(positions_angstrom.len());
        for (frame, c) in positions_angstrom.iter().zip(cells) {
            if frame.len() != 3 * n {
                return Err(PyValueError::new_err(format!(
                    "a frame of {n} atoms has {} coordinates, not {}",
                    3 * n,
                    frame.len()
                )));
            }
            let rows: Vec<[f64; 3]> = frame.chunks(3).map(|p| [p[0], p[1], p[2]]).collect();
            systems.push(
                MolecularSystem::new(
                    elements.clone(),
                    &positions::<Angstrom>(&rows),
                    SpinCharge::default(),
                    c,
                )
                .map_err(err)?,
            );
        }
        let mut traj = MdTrajectory::new(systems, times_fs).map_err(err)?;
        if let Some(v) = velocities_angstrom_per_fs {
            let scale = Units::ANGSTROM_TO_BOHR * Units::AU_TIME_TO_FS;
            let cols: Vec<Col<f64>> = v
                .iter()
                .map(|f| Col::from_fn(f.len(), |i| f[i] * scale))
                .collect();
            traj = traj.with_velocities(&cols).map_err(err)?;
        }
        Ok(Self { traj })
    }

    /// The trajectory of a run directory of `mdmain` or of a trajectory file (with the
    /// velocities next to it when there are); `interval_fs` for frames without times.
    #[staticmethod]
    #[pyo3(signature = (path, *, interval_fs = 1.0))]
    fn from_file(path: PathBuf, interval_fs: f64) -> PyResult<Self> {
        let files = RunFiles::locate(&path).map_err(err)?;
        let traj = read_trajectory(&files, interval_fs, &Logger::silent()).map_err(err)?;
        Ok(Self { traj })
    }

    fn __len__(&self) -> usize {
        self.traj.len()
    }

    /// The times of the frames (fs).
    #[getter]
    fn times_fs(&self) -> Vec<f64> {
        self.traj.times().to_vec()
    }

    /// The elements of the atoms.
    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.traj
            .elements()
            .iter()
            .map(|e| e.symbol().to_string())
            .collect()
    }

    /// Whether the frames have velocities.
    #[getter]
    fn has_velocities(&self) -> bool {
        self.traj.velocities().is_some()
    }

    /// Whether the frames are in a cell periodic along every axis.
    #[getter]
    fn periodic(&self) -> bool {
        self.traj.is_periodic()
    }

    /// The frames from `time_fs` on.
    fn skip(&self, time_fs: f64) -> PyResult<Self> {
        Ok(Self {
            traj: self.traj.from_time(time_fs).map_err(err)?,
        })
    }

    /// Radial distribution functions of the pairs of elements: `r` (Angstrom) and, for each
    /// pair `"A-B"`, `g` and the running coordination number.
    #[pyo3(signature = (bin_width = 0.02, r_max = None))]
    fn rdf<'py>(
        &self,
        py: Python<'py>,
        bin_width: f64,
        r_max: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let s = RdfSettings {
            bin_width,
            max: r_max,
        };
        let rdf = radial_distribution(&self.traj, &s).map_err(err)?;
        let out = PyDict::new(py);
        out.set_item("r", rdf.r.clone())?;
        out.set_item("bin_width", rdf.bin_width)?;
        out.set_item("periodic", rdf.periodic)?;
        out.set_item("frames", rdf.frames)?;
        let pairs = PyDict::new(py);
        for p in &rdf.pairs {
            let d = PyDict::new(py);
            d.set_item("g", p.g.clone())?;
            d.set_item("coordination", p.coordination.clone())?;
            pairs.set_item(format!("{}-{}", p.first.symbol(), p.second.symbol()), d)?;
        }
        out.set_item("pairs", pairs)?;
        Ok(out)
    }

    /// Mean squared displacements (Angstrom^2) at the lags (fs) and the diffusion
    /// coefficients fitted between the shares `fit` of the longest lag.
    #[pyo3(signature = (fit = (0.1, 0.9), remove_drift = true))]
    fn msd<'py>(
        &self,
        py: Python<'py>,
        fit: (f64, f64),
        remove_drift: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let m = mean_squared_displacement(&self.traj, &MsdSettings { fit, remove_drift })
            .map_err(err)?;
        let out = PyDict::new(py);
        out.set_item("lags_fs", m.lags.clone())?;
        out.set_item("total", m.total.clone())?;
        let species = PyDict::new(py);
        let diff = PyDict::new(py);
        diff.set_item("all", diffusion(py, &m.total_diffusion)?)?;
        for ((e, v), d) in m.species.iter().zip(&m.by_species).zip(&m.diffusion) {
            species.set_item(e.symbol(), v.clone())?;
            diff.set_item(e.symbol(), diffusion(py, d)?)?;
        }
        out.set_item("species", species)?;
        out.set_item("diffusion", diff)?;
        Ok(out)
    }

    /// The mass-weighted velocity autocorrelation up to `share` of the run, `C(t) / C(0)`.
    #[pyo3(signature = (share = 0.5))]
    fn vacf<'py>(&self, py: Python<'py>, share: f64) -> PyResult<Bound<'py, PyDict>> {
        let c = velocity_autocorrelation(&self.traj, share).map_err(err)?;
        let out = PyDict::new(py);
        out.set_item("lags_fs", c.lags.clone())?;
        out.set_item("total", c.total.clone())?;
        let species = PyDict::new(py);
        for (e, v) in c.species.iter().zip(&c.by_species) {
            species.set_item(e.symbol(), v.clone())?;
        }
        out.set_item("species", species)?;
        out.set_item("c0_hartree", c.c0)?;
        Ok(out)
    }

    /// The vibrational density of states (states per cm^-1, integrating to `3 n`) at the
    /// wavenumbers (cm^-1) up to `max_wavenumber` (the Nyquist one by default), with the
    /// `window` (`hann`, `blackman`, `none`) and the zero `padding` of the transform.
    #[pyo3(signature = (share = 0.5, max_wavenumber = None, window = "hann", padding = 4))]
    fn vdos<'py>(
        &self,
        py: Python<'py>,
        share: f64,
        max_wavenumber: Option<f64>,
        window: &str,
        padding: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let c = velocity_autocorrelation(&self.traj, share).map_err(err)?;
        let s = VdosSettings {
            max: max_wavenumber,
            window: Window::parse(window).map_err(err)?,
            padding,
        };
        let g = vibrational_density_of_states(&c, &s).map_err(err)?;
        let out = PyDict::new(py);
        out.set_item("wavenumbers", g.wavenumbers.clone())?;
        out.set_item("total", g.total.clone())?;
        let species = PyDict::new(py);
        for (e, v) in g.species.iter().zip(&g.by_species) {
            species.set_item(e.symbol(), v.clone())?;
        }
        out.set_item("species", species)?;
        out.set_item("nyquist", g.nyquist)?;
        out.set_item("resolution", g.resolution)?;
        Ok(out)
    }

    /// The self-diffusion coefficient of all atoms by Green-Kubo from their velocities
    /// (Angstrom^2 / ps), up to `lags` lags, errors from `blocks` blocks.
    #[pyo3(signature = (lags = None, blocks = 10))]
    fn diffusion_green_kubo<'py>(
        &self,
        py: Python<'py>,
        lags: Option<usize>,
        blocks: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let atoms: Vec<usize> = (0..self.traj.n_atoms()).collect();
        let lags = lags.unwrap_or(self.traj.len() / blocks.max(2) / 2);
        let g = diffusion_green_kubo(&self.traj, &atoms, lags, blocks).map_err(err)?;
        green_kubo_dict(py, &g)
    }

    /// The orientational correlation `<P_order(u(0) . u(t))>` (order 1 or 2) of the bonds
    /// (pairs of atoms from 0), up to `share` of the run.
    #[pyo3(signature = (bonds, order = 2, share = 0.5))]
    fn orientational(
        &self,
        bonds: Vec<(usize, usize)>,
        order: usize,
        share: f64,
    ) -> PyResult<Vec<f64>> {
        let n = self.traj.n_atoms();
        if bonds.iter().any(|&(a, b)| a >= n || b >= n || a == b) {
            return Err(PyValueError::new_err(
                "bonds are pairs of two different atoms",
            ));
        }
        let vectors: Vec<Vec<[f64; 3]>> = self
            .traj
            .systems()
            .iter()
            .map(|s| {
                bonds
                    .iter()
                    .map(|&(a, b)| {
                        let (p, q) = (
                            s.position(AtomId::new(a)).value(),
                            s.position(AtomId::new(b)).value(),
                        );
                        match s.active_cell() {
                            Some(c) => c.mic_delta(q, p),
                            None => [q[0] - p[0], q[1] - p[1], q[2] - p[2]],
                        }
                    })
                    .collect()
            })
            .collect();
        let lags = ((self.traj.len() as f64 * share) as usize).max(2);
        orientational(&vectors, order, lags).map_err(err)
    }

    /// RMSD of every frame to the first and RMSF of every atom (Angstrom).
    fn fluctuations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let f = fluctuations(&self.traj).map_err(err)?;
        let out = PyDict::new(py);
        out.set_item("rmsd", f.rmsd)?;
        out.set_item("rmsf", f.rmsf)?;
        Ok(out)
    }

    /// Distances (Angstrom), angles, and dihedral angles (degrees) of the atoms of each
    /// list (2, 3, or 4 atoms from 0) in every frame, by name.
    fn geometry<'py>(
        &self,
        py: Python<'py>,
        atoms: Vec<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let measures = atoms
            .iter()
            .map(|a| Measure::of(a))
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        let values = geometry_series(&self.traj, &measures).map_err(err)?;
        let out = PyDict::new(py);
        for (m, v) in measures.iter().zip(values) {
            out.set_item(m.name(), v)?;
        }
        Ok(out)
    }

    /// The principal components of the motion: the frames along them (`scores`, one row per
    /// frame), their variances (amu Angstrom^2), and their shares.
    #[pyo3(signature = (components = 3))]
    fn pca<'py>(&self, py: Python<'py>, components: usize) -> PyResult<Bound<'py, PyDict>> {
        let p = trajectory_pca(&self.traj, components).map_err(err)?;
        let out = PyDict::new(py);
        let scores: Vec<Vec<f64>> = (0..p.scores.nrows())
            .map(|i| (0..p.scores.ncols()).map(|c| p.scores[(i, c)]).collect())
            .collect();
        out.set_item("scores", scores)?;
        out.set_item("variances", p.variances)?;
        out.set_item("contributions", p.contributions)?;
        Ok(out)
    }
}

fn green_kubo_dict<'py>(py: Python<'py>, g: &GreenKubo) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("lags_fs", g.lags.clone())?;
    out.set_item("correlation", g.correlation.clone())?;
    out.set_item("integral", g.integral.clone())?;
    out.set_item("error", g.error.clone())?;
    out.set_item("value", g.value)?;
    out.set_item("value_error", g.value_error)?;
    out.set_item("read_at_fs", g.lags[g.plateau])?;
    out.set_item("plateau_mean", g.plateau_mean)?;
    out.set_item("crossed_zero", g.crossed)?;
    Ok(out)
}

/// The autocorrelation of `x` at lags `0..lags`, averaged over the time origins, by the
/// fast Fourier transform (`direct=True`: by the double sum).
#[pyfunction]
#[pyo3(signature = (x, lags, direct = false))]
fn autocorrelation(x: Vec<f64>, lags: usize, direct: bool) -> Vec<f64> {
    if direct {
        correlation::autocorrelation_direct(&x, lags)
    } else {
        correlation::autocorrelation(&x, lags)
    }
}

/// The power spectrum of the series `x` (samples `dt_fs` apart): the cosine transform of
/// its autocorrelation up to `share` of its length, with `window` and zero `padding`.
#[pyfunction]
#[pyo3(signature = (x, dt_fs, share = 0.5, window = "hann", padding = 4))]
fn power_spectrum<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    dt_fs: f64,
    share: f64,
    window: &str,
    padding: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let s = spectrum::power_spectrum(
        &x,
        dt_fs,
        share,
        Window::parse(window).map_err(err)?,
        padding,
    )
    .map_err(err)?;
    spectrum_dict(py, &s)
}

fn spectrum_dict<'py>(py: Python<'py>, s: &spectrum::Spectrum) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("wavenumbers", s.wavenumbers.clone())?;
    out.set_item("values", s.values.clone())?;
    out.set_item("resolution", s.resolution)?;
    out.set_item("nyquist", s.nyquist)?;
    Ok(out)
}

/// The infrared spectrum (largest value 1) from the dipole autocorrelation `c` (lags
/// `dt_fs` apart) at `temperature` with the quantum `correction` (`harmonic`, `standard`,
/// `schofield`, `none`).
#[pyfunction]
#[pyo3(signature = (c, dt_fs, temperature = 300.0, correction = "harmonic", window = "hann", padding = 4, max_wavenumber = None))]
#[allow(clippy::too_many_arguments)]
fn infrared_spectrum<'py>(
    py: Python<'py>,
    c: Vec<f64>,
    dt_fs: f64,
    temperature: f64,
    correction: &str,
    window: &str,
    padding: usize,
    max_wavenumber: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let s = spectrum::infrared(
        &c,
        dt_fs,
        temperature,
        QuantumCorrection::parse(correction).map_err(err)?,
        Window::parse(window).map_err(err)?,
        padding,
        max_wavenumber,
    )
    .map_err(err)?;
    spectrum_dict(py, &s)
}

/// `prefactor int_0^inf <X(0) X(t)> dt` of the series `components` (each one sample every
/// `dt_fs`; their correlations averaged), up to `lags` lags, with the errors of `blocks`
/// blocks and the value read at the first zero of the correlation.
#[pyfunction]
#[pyo3(signature = (components, dt_fs, prefactor = 1.0, lags = None, blocks = 10))]
fn green_kubo<'py>(
    py: Python<'py>,
    components: Vec<Vec<f64>>,
    dt_fs: f64,
    prefactor: f64,
    lags: Option<usize>,
    blocks: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let n = components.first().map_or(0, Vec::len);
    let lags = lags.unwrap_or(n / blocks.max(2) / 2);
    let g = transport::green_kubo(&components, dt_fs, prefactor, lags, blocks).map_err(err)?;
    green_kubo_dict(py, &g)
}

/// Mean, deviation, and the error of the mean from block averages of a correlated series
/// (with the blocking levels).
#[pyfunction]
fn block_average<'py>(py: Python<'py>, values: Vec<f64>) -> PyResult<Bound<'py, PyDict>> {
    let s = statistics(&values).map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("count", s.count)?;
    out.set_item("mean", s.mean)?;
    out.set_item("deviation", s.deviation)?;
    out.set_item("min", s.min)?;
    out.set_item("max", s.max)?;
    out.set_item("naive_error", s.naive_error)?;
    out.set_item("error", s.error)?;
    out.set_item("block_size", s.block_size)?;
    out.set_item("inefficiency", s.inefficiency)?;
    out.set_item("converged", s.converged)?;
    let levels: Vec<(usize, usize, f64, f64)> = s
        .levels
        .iter()
        .map(|l| (l.block_size, l.blocks, l.error, l.error_of_error))
        .collect();
    out.set_item("levels", levels)?;
    Ok(out)
}

/// The analysis of a run directory or trajectory file as `mdmain -analyze_only -analysis
/// OPTIONS...` makes it, into `out_dir` (`trajectory_analysis/` of the run by default);
/// the files written.
#[pyfunction]
#[pyo3(signature = (path, options = Vec::new(), out_dir = None))]
fn analyze_md_run(
    path: PathBuf,
    options: Vec<String>,
    out_dir: Option<PathBuf>,
) -> PyResult<Vec<String>> {
    let settings = parse_analysis(true, &options)
        .map_err(err)?
        .unwrap_or_default();
    let files = RunFiles::locate(&path).map_err(err)?;
    let out = out_dir.unwrap_or_else(|| output_dir(&path));
    let written = analyze_files(&files, &settings, &out, &Logger::silent()).map_err(err)?;
    Ok(written
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect())
}

/// Add the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrajectory>()?;
    m.add_function(wrap_pyfunction!(block_average, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_md_run, m)?)?;
    m.add_function(wrap_pyfunction!(autocorrelation, m)?)?;
    m.add_function(wrap_pyfunction!(power_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(infrared_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(green_kubo, m)?)?;
    Ok(())
}
