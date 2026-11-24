//! A vibrational analysis from a structure and its Hessian, and its reports: the table of
//! normal modes, the thermochemistry, and the animations of the modes, in MultiOptPy's
//! formats (`normal_modes.txt`, `thermochemistry.txt`,
//! `vibration_animation/mode_N_FREQ_wave_number.xyz`).
//!
//! The animations move the atoms along a mode so that the atom that moves most is displaced
//! by `amplitude` Angstrom at the turning points. (MultiOptPy multiplies the unnormalized
//! mode by 3 and adds it to the coordinates in Bohr, which gives another amplitude for every
//! mode.)
//!
//! MultiOptPy: `Thermo/normal_mode_analyzer.py:MolecularVibrations.print_normal_modes`,
//!  `print_thermochemistry`, `_VibrationalModeAnimator`,
//!  `optimization.py:Optimize._perform_vibrational_analysis`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;

use faer::Mat;

use super::{
    normal_modes_with_masses, point_group, thermochemistry, Direction, NormalModes, Parts,
    PointGroup, ThermoSettings, Thermochemistry,
};
use crate::domain::{MolecularSystem, Units};
use crate::error::MoptResult;

/// Choices of a vibrational analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencySettings {
    /// Conditions and low-frequency treatment; the symmetry number is taken from the point
    /// group unless `symmetry_tolerance` is `None`.
    pub thermo: ThermoSettings,
    /// Tolerance (Bohr) of the point-group analysis; `None` skips it (`-nosymm`: `C1`, and
    /// the symmetry number of `thermo`).
    pub symmetry_tolerance: Option<f64>,
    /// Project translations and rotations out of the Hessian (not for a surface in a fixed
    /// frame).
    pub project_rigid: bool,
    /// Modes below this wavenumber in magnitude (cm^-1) are left out of the table: 0.1.
    pub table_cutoff: f64,
    /// Modes below this wavenumber in magnitude (cm^-1) are not animated: 10.
    pub animation_cutoff: f64,
    /// Frames of an animation: 20.
    pub frames: usize,
    /// Largest displacement of an atom in an animation (Angstrom): 0.3.
    pub amplitude: f64,
}

impl Default for FrequencySettings {
    fn default() -> Self {
        Self {
            thermo: ThermoSettings::default(),
            symmetry_tolerance: Some(1e-2),
            project_rigid: true,
            table_cutoff: 0.1,
            animation_cutoff: 10.0,
            frames: 20,
            amplitude: 0.3,
        }
    }
}

/// Normal modes, point group, and thermochemistry of a structure.
#[derive(Debug, Clone)]
pub struct VibrationalAnalysis {
    /// Normal modes.
    pub modes: NormalModes,
    /// Point group.
    pub point_group: PointGroup,
    /// Thermodynamic functions.
    pub thermo: Thermochemistry,
}

/// Vibrational analysis of `sys` with its Cartesian Hessian and electronic energy.
pub fn analyze(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    electronic_energy: f64,
    settings: &FrequencySettings,
) -> MoptResult<VibrationalAnalysis> {
    analyze_without(sys, hessian, electronic_energy, settings, &[])
}

/// [`analyze`] with `directions` projected out of the modes as well (the tangent of a
/// reaction path).
pub fn analyze_without(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    electronic_energy: f64,
    settings: &FrequencySettings,
    directions: &[Direction],
) -> MoptResult<VibrationalAnalysis> {
    let masses = settings.thermo.masses.masses(sys)?;
    let modes =
        normal_modes_with_masses(sys, hessian, settings.project_rigid, directions, &masses)?;
    let mut thermo_settings = settings.thermo;
    let group = match settings.symmetry_tolerance {
        Some(tolerance) => {
            let group = point_group(sys, tolerance);
            thermo_settings.symmetry_number = group.symmetry_number;
            group
        }
        None => PointGroup::c1(),
    };
    let thermo = thermochemistry(sys, &modes, electronic_energy, &thermo_settings)?;
    Ok(VibrationalAnalysis {
        modes,
        point_group: group,
        thermo,
    })
}

/// The table of normal modes, three to a block: imaginary frequencies first (negative
/// wavenumbers), then the others in ascending order.
pub fn normal_modes_text(sys: &MolecularSystem, modes: &NormalModes, cutoff: f64) -> String {
    let shown = modes.vibrations(cutoff, true);
    let mut text = String::new();
    for block in shown.chunks(3) {
        let first = shown.iter().position(|k| *k == block[0]).unwrap_or(0);
        let row = |label: &str, value: &dyn Fn(usize) -> f64| {
            let cells: String = block
                .iter()
                .map(|&k| format!("{:20.4}", value(k)))
                .collect();
            format!("{label}{cells}\n")
        };
        let numbers: String = (first..first + block.len())
            .map(|i| format!("{i:20}"))
            .collect();
        let _ = writeln!(text, "Mode              {numbers}");
        text += &row("Freq [cm^-1]          ", &|k| modes.wavenumbers[k]);
        text += &row("Reduced mass [amu]    ", &|k| modes.reduced_masses[k]);
        text += &row("Force const [mDyne/A] ", &|k| modes.force_constants[k]);
        text += &row("Char temp [K]         ", &|k| modes.temperatures[k]);
        let _ = writeln!(
            text,
            "Normal mode            {}",
            "       x         y         z     ".repeat(block.len())
        );
        for (a, element) in sys.elements.iter().enumerate() {
            let cells: Vec<String> = block
                .iter()
                .map(|&k| {
                    let d = &modes.displacements[k];
                    format!(
                        "{:9.5}  {:9.5}  {:9.5}",
                        d[3 * a],
                        d[3 * a + 1],
                        d[3 * a + 2]
                    )
                })
                .collect();
            let _ = writeln!(
                text,
                "    {:>4}               {}",
                element.symbol(),
                cells.join("  ")
            );
        }
        text.push('\n');
    }
    text
}

/// The thermochemistry report.
pub fn thermochemistry_text(analysis: &VibrationalAnalysis) -> String {
    let t = &analysis.thermo;
    let mut text = String::new();
    let _ = writeln!(text, "Point group: {}", analysis.point_group.label);
    let _ = writeln!(text, "Temperature {:.4} [K]", t.temperature);
    let _ = writeln!(text, "Pressure {:.2} [Pa]", t.pressure);
    let _ = writeln!(text, "{}", t.masses.describe());
    if let Some(concentration) = t.concentration {
        let _ = writeln!(
            text,
            "Concentration {concentration} [mol/L] (translational entropy)"
        );
    }
    if t.scale != (1.0, 1.0) {
        let _ = writeln!(
            text,
            "Frequency scale factors: {} (thermal terms), {} (zero-point energy)",
            t.scale.0, t.scale.1
        );
    }
    let c = t.rotational_constants_ghz;
    let _ = writeln!(
        text,
        "Rotational constants [GHz] {:.5} {:.5} {:.5}",
        c[0], c[1], c[2]
    );
    let _ = writeln!(text, "Symmetry number {}", t.symmetry_number);
    let _ = writeln!(
        text,
        "Zero-point energy (ZPE) {:.10} [Eh]",
        t.zero_point_energy
    );
    let heads: Vec<String> = ["tot", "elec", "trans", "rot", "vib"]
        .iter()
        .map(|h| format!("{h:>20}"))
        .collect();
    let _ = writeln!(text, "                    {}", heads.join(" "));
    let values = |p: &Parts| {
        [
            p.total,
            p.electronic,
            p.translational,
            p.rotational,
            p.vibrational,
        ]
    };
    for (title, parts) in [
        ("Entropy [Eh/K]", &t.entropy),
        ("Cv [Eh/K]", &t.cv),
        ("Cp [Eh/K]", &t.cp),
    ] {
        let cells: Vec<String> = values(parts).iter().map(|v| format!("{v:20.10}")).collect();
        let _ = writeln!(text, "{title:<20} {}", cells.join(" "));
    }
    let wide = |parts: &Parts| -> String {
        let cells: Vec<String> = values(parts).iter().map(|v| format!("{v:20.10}")).collect();
        cells.join("    ")
    };
    let _ = writeln!(
        text,
        "Internal energy [Eh]     {}",
        wide(&t.internal_energy)
    );
    let _ = writeln!(
        text,
        "Total internal energy [Eh]  {:.10}",
        t.internal_energy.total
    );
    let _ = writeln!(
        text,
        "Electronic energy [Eh]      {:.10}",
        t.electronic_energy
    );
    let _ = writeln!(text, "Enthalpy [Eh]           {}", wide(&t.enthalpy));
    let _ = writeln!(text, "Total enthalpy [Eh]        {:.10}", t.enthalpy.total);
    let _ = writeln!(text, "Gibbs free energy [Eh]  {}", wide(&t.gibbs));
    let _ = writeln!(text, "Total Gibbs free energy [Eh]  {:.10}", t.gibbs.total);
    if let Some([s, h, g]) = t.uncorrected_vibration {
        let _ = writeln!(
            text,
            "Vibration without the low-frequency correction: S {s:.10} [Eh/K], H {h:.10} [Eh], \
             G {g:.10} [Eh]"
        );
    }
    if t.imaginary > 0 {
        let _ = writeln!(
            text,
            "Imaginary frequencies (not included): {}",
            t.imaginary
        );
    }
    if t.inverted > 0 {
        let _ = writeln!(text, "Imaginary frequencies taken as real: {}", t.inverted);
    }
    text
}

/// Animations of the modes beyond `settings.animation_cutoff`: file names
/// (`mode_N_FREQ_wave_number.xyz`, imaginary frequencies with a trailing `i`) and their
/// contents (xyz frames, Angstrom).
pub fn animations(
    sys: &MolecularSystem,
    modes: &NormalModes,
    settings: &FrequencySettings,
) -> Vec<(String, String)> {
    let n = sys.n_atoms();
    let com = sys.center_of_mass().value();
    let frames = settings.frames.max(2);
    modes
        .vibrations(settings.animation_cutoff, true)
        .into_iter()
        .enumerate()
        .map(|(number, k)| {
            let wavenumber = modes.wavenumbers[k];
            let frequency = if wavenumber < 0.0 {
                format!("{:.0}i", wavenumber.abs())
            } else {
                format!("{wavenumber:.0}")
            };
            let d = &modes.displacements[k];
            let largest = (0..n)
                .map(|a| (0..3).map(|c| d[3 * a + c].powi(2)).sum::<f64>().sqrt())
                .fold(0.0_f64, f64::max);
            let scale = if largest > 0.0 {
                settings.amplitude * Units::ANGSTROM_TO_BOHR / largest
            } else {
                0.0
            };
            let mut text = String::new();
            for frame in 0..frames {
                let phase = 2.0 * std::f64::consts::PI * frame as f64 / (frames - 1) as f64;
                let _ = writeln!(text, "{n}");
                let _ = writeln!(
                    text,
                    "Mode {}, Freq: {frequency} cm-1, Frame: {}/{frames}",
                    number + 1,
                    frame + 1
                );
                for (a, atom) in sys.atoms().enumerate() {
                    let r = sys.position(atom).value();
                    let p: [f64; 3] = std::array::from_fn(|c| {
                        (r[c] - com[c] + scale * phase.sin() * d[3 * a + c])
                            * Units::BOHR_TO_ANGSTROM
                    });
                    let _ = writeln!(
                        text,
                        "{} {:.6} {:.6} {:.6}",
                        sys.elements[a].symbol(),
                        p[0],
                        p[1],
                        p[2]
                    );
                }
            }
            (
                format!("mode_{}_{frequency}_wave_number.xyz", number + 1),
                text,
            )
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};

    /// Water with a diagonal-dominant model Hessian: three vibrations.
    fn water() -> (MolecularSystem, Mat<f64>) {
        let sys = MolecularSystem::new(
            vec![Element::O, Element::H, Element::H],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.117],
                [0.0, 0.757, -0.469],
                [0.0, -0.757, -0.469],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        // Springs along the two bonds and between the hydrogens.
        let mut h = Mat::<f64>::zeros(9, 9);
        for (i, j, k) in [(0, 1, 0.5), (0, 2, 0.5), (1, 2, 0.1)] {
            let d = sys
                .delta(crate::domain::AtomId::new(i), crate::domain::AtomId::new(j))
                .value();
            for a in 0..3 {
                for b in 0..3 {
                    let v = k * d[a] * d[b];
                    h[(3 * i + a, 3 * i + b)] += v;
                    h[(3 * j + a, 3 * j + b)] += v;
                    h[(3 * i + a, 3 * j + b)] -= v;
                    h[(3 * j + a, 3 * i + b)] -= v;
                }
            }
        }
        (sys, h)
    }

    #[test]
    fn reports_have_the_layout_of_multioptpy() {
        let (sys, h) = water();
        let analysis = analyze(&sys, &h, -76.4, &FrequencySettings::default()).unwrap();
        assert_eq!(analysis.point_group.label, "C2v");
        assert_eq!(analysis.thermo.symmetry_number, 2);
        assert_eq!(analysis.thermo.vibrations, 3);
        let table = normal_modes_text(&sys, &analysis.modes, 0.1);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(
            lines[0],
            "Mode                                 0                   1                   2"
        );
        assert!(lines[1].starts_with("Freq [cm^-1]          "));
        assert!(lines[5].starts_with("Normal mode                   x         y         z"));
        assert!(lines[6].starts_with("       O               "));
        assert_eq!(lines.len(), 10);
        let report = thermochemistry_text(&analysis);
        assert!(report
            .starts_with("Point group: C2v\nTemperature 298.1500 [K]\nPressure 101325.00 [Pa]\n"));
        assert!(report.contains("Symmetry number 2\n"));
        assert!(report.contains("Total Gibbs free energy [Eh]  -76."));
        // Without the point-group analysis: C1 and the symmetry number given.
        let plain = FrequencySettings {
            symmetry_tolerance: None,
            ..FrequencySettings::default()
        };
        let c1 = analyze(&sys, &h, -76.4, &plain).unwrap();
        assert_eq!(
            (c1.point_group.label.as_str(), c1.thermo.symmetry_number),
            ("C1", 1)
        );
        // Symmetry number 2 lowers the rotational entropy by k ln 2.
        let k = Units::BOLTZMANN_HARTREE_PER_K;
        let difference = c1.thermo.entropy.rotational - analysis.thermo.entropy.rotational;
        assert!((difference - k * 2.0_f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn animations_move_the_fastest_atom_by_the_amplitude() {
        let (sys, h) = water();
        let settings = FrequencySettings::default();
        let modes = crate::vib::normal_modes(&sys, &h, true).unwrap();
        let files = animations(&sys, &modes, &settings);
        assert_eq!(files.len(), 3);
        assert!(files[0].0.starts_with("mode_1_") && files[0].0.ends_with("_wave_number.xyz"));
        let lines: Vec<&str> = files[2].1.lines().collect();
        assert_eq!(lines.len(), 20 * 5);
        assert!(lines[1].starts_with("Mode 3, Freq: ") && lines[1].ends_with("Frame: 1/20"));
        // Frame 1 is the structure itself; a quarter period later an atom is 0.3 Angstrom away.
        let position = |frame: usize, atom: usize| -> [f64; 3] {
            let cells: Vec<f64> = lines[5 * frame + 2 + atom]
                .split_whitespace()
                .skip(1)
                .map(|v| v.parse().unwrap())
                .collect();
            [cells[0], cells[1], cells[2]]
        };
        let mut largest = 0.0_f64;
        for frame in 0..20 {
            for atom in 0..3 {
                let (p, q) = (position(frame, atom), position(0, atom));
                let d =
                    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                largest = largest.max(d);
            }
        }
        assert!((largest - 0.3).abs() < 0.01, "{largest}");
    }
}
