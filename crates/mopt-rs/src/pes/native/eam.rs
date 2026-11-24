//! Embedded-atom method with the analytic functions and parameters of Zhou, Johnson, and
//! Wadley (Phys. Rev. B 2004, 69, 144113) for sixteen metals and their alloys: Cu, Ag, Au,
//! Ni, Pd, Pt, Al, Pb, Fe, Mo, Ta, W, Mg, Co, Ti, Zr.
//!
//! With `x = r / r_e` of the element:
//!
//! ```text
//! f(r)      = f_e exp(-beta (x - 1)) / (1 + (x - lambda)^20)                 electron density
//! phi_aa(r) = A exp(-alpha (x - 1)) / (1 + (x - kappa)^20)
//!           - B exp(-beta (x - 1)) / (1 + (x - lambda)^20)                   pair potential
//! phi_ab(r) = 1/2 (f_b / f_a phi_aa + f_a / f_b phi_bb)                      alloy
//! F(rho)    = sum_k F_nk (rho / rho_n - 1)^k        rho < rho_n = 0.85 rho_e
//!           = sum_k F_k (rho / rho_e - 1)^k         rho_n <= rho < rho_0 = 1.15 rho_e
//!           = F_e (1 - eta ln(rho / rho_s)) (rho / rho_s)^eta   rho >= rho_0
//! E = sum_i F_i(rho_i) + 1/2 sum_(i != j) phi_ij(r_ij),   rho_i = sum_(j != i) f_j(r_ij)
//! ```
//!
//! (`rho_n = 0.25 rho_e` for Pt). Energies in eV, lengths in Angstrom. The interactions are
//! cut at `sqrt(5)` times the largest `r_e` of the elements present, which is the cutoff
//! `sqrt(10) / 2` lattice constants of the tabulated files in common use; the functions
//! have fallen to about `1e-6` of their values at `r_e` there. Over the last tenth of the
//! cutoff, density and pair potential are multiplied by a switching function
//! ([`super::switching`]) and reach zero with all derivatives continuous, so the energy
//! has no step where a neighbor crosses the cutoff ([`Eam::with_switch_width`]).
//!
//! Gradient and Hessian are exact (the embedding engine), in isolated systems and in
//! periodic cells.
//!
//! The parameters are those of the paper's table as distributed with the generator of
//! tabulated files written by X. W. Zhou (`tools/eam_database` of LAMMPS); see
//! `third_party/DISPERSION_AND_PARAMETERS.md`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::embedding::{evaluate_embedding, EmbeddingModel, ModelUnits};
use super::neighbors::VerletList;
use super::switching::switch_start;
use crate::domain::{Element, EvalOutput, EvalRequest, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::Scalar;
use crate::pes::calculator::Calculator;

/// `(symbol, [r_e, f_e, rho_e, rho_s, alpha, beta, A, B, kappa, lambda, F_n0, F_n1, F_n2,
/// F_n3, F_0, F_1, F_2, F_3, eta, F_e], rho_n / rho_e)`.
const TABLE: [(&str, [f64; 20], f64); 16] = [
    (
        "Cu",
        [
            2.556162, 1.554485, 21.175871, 21.175395, 8.127620, 4.334731, 0.396620, 0.548085,
            0.308782, 0.756515, -2.170269, -0.263788, 1.088878, -0.817603, -2.19, 0.0, 0.561830,
            -2.100595, 0.310490, -2.186568,
        ],
        0.85,
    ),
    (
        "Ag",
        [
            2.891814, 1.106232, 14.604100, 14.604144, 9.132010, 4.870405, 0.277758, 0.419611,
            0.339710, 0.750758, -1.729364, -0.255882, 0.912050, -0.561432, -1.75, 0.0, 0.744561,
            -1.150650, 0.783924, -1.748423,
        ],
        0.85,
    ),
    (
        "Au",
        [
            2.885034, 1.529021, 19.991632, 19.991509, 9.516052, 5.075228, 0.229762, 0.356666,
            0.356570, 0.748798, -2.937772, -0.500288, 1.601954, -0.835530, -2.98, 0.0, 1.706587,
            -1.134778, 1.021095, -2.978815,
        ],
        0.85,
    ),
    (
        "Ni",
        [
            2.488746, 2.007018, 27.562015, 27.562031, 8.383453, 4.471175, 0.429046, 0.633531,
            0.443599, 0.820658, -2.693513, -0.076445, 0.241442, -2.375626, -2.70, 0.0, 0.265390,
            -0.152856, 0.445470, -2.7,
        ],
        0.85,
    ),
    (
        "Pd",
        [
            2.750897, 1.595417, 21.335246, 21.940073, 8.697397, 4.638612, 0.406763, 0.598880,
            0.397263, 0.754799, -2.321006, -0.473983, 1.615343, -0.231681, -2.36, 0.0, 1.481742,
            -1.675615, 1.130000, -2.352753,
        ],
        0.85,
    ),
    (
        "Pt",
        [
            2.771916, 2.336509, 33.367564, 35.205357, 7.105782, 3.789750, 0.556398, 0.696037,
            0.385255, 0.770510, -1.455568, -2.149952, 0.528491, 1.222875, -4.17, 0.0, 3.010561,
            -2.420128, 1.450000, -4.145597,
        ],
        0.25,
    ),
    (
        "Al",
        [
            2.863924, 1.403115, 20.418205, 23.195740, 6.613165, 3.527021, 0.314873, 0.365551,
            0.379846, 0.759692, -2.807602, -0.301435, 1.258562, -1.247604, -2.83, 0.0, 0.622245,
            -2.488244, 0.785902, -2.824528,
        ],
        0.85,
    ),
    (
        "Pb",
        [
            3.499723, 0.647872, 8.450154, 8.450063, 9.121799, 5.212457, 0.161219, 0.236884,
            0.250805, 0.764955, -1.422370, -0.210107, 0.682886, -0.529378, -1.44, 0.0, 0.702726,
            -0.538766, 0.935380, -1.439436,
        ],
        0.85,
    ),
    (
        "Fe",
        [
            2.481987, 1.885957, 20.041463, 20.041463, 9.818270, 5.236411, 0.392811, 0.646243,
            0.170306, 0.340613, -2.534992, -0.059605, 0.193065, -2.282322, -2.54, 0.0, 0.200269,
            -0.148770, 0.391750, -2.539945,
        ],
        0.85,
    ),
    (
        "Mo",
        [
            2.728100, 2.723710, 29.354065, 29.354065, 8.393531, 4.476550, 0.708787, 1.120373,
            0.137640, 0.275280, -3.692913, -0.178812, 0.380450, -3.133650, -3.71, 0.0, 0.875874,
            0.776222, 0.790879, -3.712093,
        ],
        0.85,
    ),
    (
        "Ta",
        [
            2.860082, 3.086341, 33.787168, 33.787168, 8.489528, 4.527748, 0.611679, 1.032101,
            0.176977, 0.353954, -5.103845, -0.405524, 1.112997, -3.585325, -5.14, 0.0, 1.640098,
            0.221375, 0.848843, -5.141526,
        ],
        0.85,
    ),
    (
        "W",
        [
            2.740840, 3.487340, 37.234847, 37.234847, 8.900114, 4.746728, 0.882435, 1.394592,
            0.139209, 0.278417, -4.946281, -0.148818, 0.365057, -4.432406, -4.96, 0.0, 0.661935,
            0.348147, 0.582714, -4.961306,
        ],
        0.85,
    ),
    (
        "Mg",
        [
            3.196291, 0.544323, 7.132600, 7.132600, 10.228708, 5.455311, 0.137518, 0.225930, 0.5,
            1.0, -0.896473, -0.044291, 0.162232, -0.689950, -0.90, 0.0, 0.122838, -0.226010,
            0.431425, -0.899702,
        ],
        0.85,
    ),
    (
        "Co",
        [
            2.505979, 1.975299, 27.206789, 27.206789, 8.679625, 4.629134, 0.421378, 0.640107, 0.5,
            1.0, -2.541799, -0.219415, 0.733381, -1.589003, -2.56, 0.0, 0.705845, -0.687140,
            0.694608, -2.559307,
        ],
        0.85,
    ),
    (
        "Ti",
        [
            2.933872, 1.863200, 25.565138, 25.565138, 8.775431, 4.680230, 0.373601, 0.570968, 0.5,
            1.0, -3.203773, -0.198262, 0.683779, -2.321732, -3.22, 0.0, 0.608587, -0.750710,
            0.558572, -3.219176,
        ],
        0.85,
    ),
    (
        "Zr",
        [
            3.199978, 2.230909, 30.879991, 30.879991, 8.559190, 4.564902, 0.424667, 0.640054, 0.5,
            1.0, -4.485793, -0.293129, 0.990148, -3.202516, -4.51, 0.0, 0.928602, -0.981870,
            0.597133, -4.509025,
        ],
        0.85,
    ),
];

/// Elements with parameters.
pub const ELEMENTS: [&str; 16] = [
    "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "Al", "Pb", "Fe", "Mo", "Ta", "W", "Mg", "Co", "Ti", "Zr",
];

/// Parameters of one element (eV, Angstrom).
#[derive(Debug, Clone, Copy)]
struct Params {
    re: f64,
    fe: f64,
    rhoe: f64,
    rhos: f64,
    alpha: f64,
    beta: f64,
    a: f64,
    b: f64,
    kappa: f64,
    lambda: f64,
    low: [f64; 4],
    middle: [f64; 4],
    eta: f64,
    high: f64,
    rhon: f64,
}

/// The embedded-atom method of Zhou, Johnson, and Wadley.
#[derive(Debug, Clone)]
pub struct Eam {
    params: Vec<(Element, Params)>,
    cutoff: f64,
    /// Start of the switching range below the cutoff (Angstrom).
    switch_on: Option<f64>,
    /// The pairs kept between evaluations.
    verlet: VerletList,
}

impl Eam {
    /// The potential for a system with the elements `elements`.
    pub fn zhou2004(elements: &[Element]) -> MoptResult<Self> {
        let mut params: Vec<(Element, Params)> = Vec::new();
        for element in elements {
            if params.iter().any(|(e, _)| e == element) {
                continue;
            }
            let entry = TABLE
                .iter()
                .find(|(symbol, _, _)| Element::parse_token(symbol).ok() == Some(*element))
                .ok_or_else(|| {
                    MoptError::config(format!(
                        "no EAM parameters for {element} (available: {})",
                        ELEMENTS.join(" ")
                    ))
                })?;
            let p = entry.1;
            params.push((
                *element,
                Params {
                    re: p[0],
                    fe: p[1],
                    rhoe: p[2],
                    rhos: p[3],
                    alpha: p[4],
                    beta: p[5],
                    a: p[6],
                    b: p[7],
                    kappa: p[8],
                    lambda: p[9],
                    low: [p[10], p[11], p[12], p[13]],
                    middle: [p[14], p[15], p[16], p[17]],
                    eta: p[18],
                    high: p[19],
                    rhon: entry.2 * p[2],
                },
            ));
        }
        let largest = params.iter().map(|(_, p)| p.re).fold(0.0, f64::max);
        if params.is_empty() {
            return Err(MoptError::config("EAM needs one atom at least"));
        }
        let cutoff = 5.0_f64.sqrt() * largest;
        Ok(Self {
            params,
            cutoff,
            switch_on: switch_start(cutoff, None)?,
            verlet: VerletList::default(),
        })
    }

    /// Set the width (Angstrom) of the range below the cutoff over which the interactions
    /// are switched off; 0: they are cut as they are, as in the tabulated files.
    pub fn with_switch_width(mut self, angstrom: f64) -> MoptResult<Self> {
        self.switch_on = switch_start(self.cutoff, Some(angstrom))?;
        Ok(self)
    }

    fn p(&self, element: Element) -> Params {
        // `validate` guarantees presence; the fallback is never used.
        self.params
            .iter()
            .find(|(e, _)| *e == element)
            .map_or(self.params[0].1, |(_, p)| *p)
    }
}

/// `c exp(-k (x - 1)) / (1 + (x - shift)^20)`.
fn damped<T: Scalar>(c: f64, k: f64, shift: f64, x: T) -> T {
    (T::from_f64(-k) * (x - T::one())).exp() * T::from_f64(c)
        / (T::one() + (x - T::from_f64(shift)).powi(20))
}

fn density_of<T: Scalar>(p: &Params, r: T) -> T {
    damped(p.fe, p.beta, p.lambda, r / T::from_f64(p.re))
}

fn pair_of<T: Scalar>(p: &Params, r: T) -> T {
    let x = r / T::from_f64(p.re);
    damped(p.a, p.alpha, p.kappa, x) - damped(p.b, p.beta, p.lambda, x)
}

fn cubic<T: Scalar>(c: &[f64; 4], x: T) -> T {
    ((x * T::from_f64(c[3]) + T::from_f64(c[2])) * x + T::from_f64(c[1])) * x + T::from_f64(c[0])
}

impl EmbeddingModel for Eam {
    fn validate(&self, elements: &[Element]) -> MoptResult<()> {
        match elements
            .iter()
            .find(|e| !self.params.iter().any(|(p, _)| p == *e))
        {
            Some(e) => Err(MoptError::config(format!(
                "this EAM calculator was set up without {e}"
            ))),
            None => Ok(()),
        }
    }

    fn cutoff(&self) -> f64 {
        self.cutoff
    }

    fn switch_on(&self) -> Option<f64> {
        self.switch_on
    }

    fn verlet_list(&self) -> Option<&VerletList> {
        Some(&self.verlet)
    }

    fn density<T: Scalar>(&self, _i: Element, j: Element, r: T) -> T {
        density_of(&self.p(j), r)
    }

    fn embedding<T: Scalar>(&self, i: Element, rho: T) -> T {
        let p = self.p(i);
        let value = rho.val();
        // The low branch holds down to an atom without neighbors, whose energy is its value
        // at zero density (a few 1e-6 eV at most, not exactly zero in the published table):
        // no step when the first neighbor comes within the cutoff.
        if value < p.rhon {
            cubic(&p.low, rho / T::from_f64(p.rhon) - T::one())
        } else if value < 1.15 * p.rhoe {
            cubic(&p.middle, rho / T::from_f64(p.rhoe) - T::one())
        } else {
            let x = rho / T::from_f64(p.rhos);
            (T::one() - x.ln() * T::from_f64(p.eta)) * x.powf(p.eta) * T::from_f64(p.high)
        }
    }

    fn pair<T: Scalar>(&self, i: Element, j: Element, r: T) -> T {
        // The engine sums over ordered pairs: half the pair potential each.
        let (pi, pj) = (self.p(i), self.p(j));
        let half = T::from_f64(0.5);
        if i == j {
            return pair_of(&pi, r) * half;
        }
        let (fi, fj) = (density_of(&pi, r), density_of(&pj, r));
        (fj / fi * pair_of(&pi, r) + fi / fj * pair_of(&pj, r)) * half * half
    }
}

impl Calculator for Eam {
    fn analytic_hessian(&self) -> bool {
        true
    }

    fn citations(&self) -> Vec<String> {
        vec![
            "X. W. Zhou, R. A. Johnson, H. N. G. Wadley, Phys. Rev. B 2004, 69, 144113 (embedded-atom potentials)".to_string(),
        ]
    }

    fn name(&self) -> &str {
        "eam"
    }

    fn description(&self) -> String {
        format!(
            "eam: embedded-atom method of Zhou, Johnson, and Wadley (2004) built into mopt-rs {} \
             ({})",
            crate::VERSION,
            ELEMENTS.join(" ")
        )
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let units = ModelUnits {
            length_per_bohr: Units::BOHR_TO_ANGSTROM,
            hartree_per_energy: Units::EV_TO_HARTREE,
        };
        let (energy, gradient_bohr, hessian_bohr, stress, strain_hessian) =
            evaluate_embedding(self, units, sys, req)?;
        Ok(EvalOutput {
            energy,
            gradient_bohr,
            hessian_bohr,
            orbitals: None,
            stress,
            strain_hessian,
        })
    }

    fn site_terms(
        &self,
        sys: &MolecularSystem,
    ) -> Option<MoptResult<crate::pes::calculator::SiteTerms>> {
        let units = ModelUnits {
            length_per_bohr: Units::BOHR_TO_ANGSTROM,
            hartree_per_energy: Units::EV_TO_HARTREE,
        };
        Some(super::embedding::site_terms(self, units, sys))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use faer::Col;

    use super::*;
    use crate::domain::{positions, Angstrom, PeriodicCell, SpinCharge, Vector3};

    fn element(symbol: &str) -> Element {
        Element::parse_token(symbol).unwrap()
    }

    /// A crystal of `symbol` with the nearest-neighbour distance `d` (Angstrom): `fcc` (four
    /// atoms), `bcc` (two), or `hcp` with the ideal axial ratio (two).
    fn crystal(symbol: &str, lattice: &str, d: f64) -> MolecularSystem {
        let (vectors, fractions): ([[f64; 3]; 3], Vec<[f64; 3]>) = match lattice {
            "fcc" => {
                let a = d * 2.0_f64.sqrt();
                (
                    [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
                    vec![
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.0],
                        [0.5, 0.0, 0.5],
                        [0.0, 0.5, 0.5],
                    ],
                )
            }
            "bcc" => {
                let a = 2.0 * d / 3.0_f64.sqrt();
                (
                    [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
                    vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                )
            }
            _ => {
                let c = d * (8.0_f64 / 3.0).sqrt();
                (
                    [
                        [d, 0.0, 0.0],
                        [-0.5 * d, 0.5 * 3.0_f64.sqrt() * d, 0.0],
                        [0.0, 0.0, c],
                    ],
                    vec![[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.5]],
                )
            }
        };
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new(vectors[0]),
            Vector3::<Angstrom>::new(vectors[1]),
            Vector3::<Angstrom>::new(vectors[2]),
            [true, true, true],
        )
        .unwrap();
        let points: Vec<[f64; 3]> = fractions
            .iter()
            .map(|f| {
                std::array::from_fn(|c| {
                    f[0] * vectors[0][c] + f[1] * vectors[1][c] + f[2] * vectors[2][c]
                })
            })
            .collect();
        MolecularSystem::new(
            vec![element(symbol); points.len()],
            &positions::<Angstrom>(&points),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap()
    }

    fn energy_per_atom(sys: &MolecularSystem) -> f64 {
        let calc = Eam::zhou2004(&sys.elements).unwrap();
        calc.evaluate(sys, &EvalRequest::ENERGY_ONLY)
            .unwrap()
            .energy
            .value()
            / Units::EV_TO_HARTREE
            / sys.n_atoms() as f64
    }

    /// Lattices and the cohesive energies the potentials were fitted to (eV/atom).
    const CRYSTALS: [(&str, &str, f64); 16] = [
        ("Cu", "fcc", 3.54),
        ("Ag", "fcc", 2.85),
        ("Au", "fcc", 3.93),
        ("Ni", "fcc", 4.45),
        ("Pd", "fcc", 3.91),
        ("Pt", "fcc", 5.77),
        ("Al", "fcc", 3.58),
        ("Pb", "fcc", 2.02),
        ("Fe", "bcc", 4.29),
        ("Mo", "bcc", 6.81),
        ("Ta", "bcc", 8.09),
        ("W", "bcc", 8.76),
        ("Mg", "hcp", 1.55),
        ("Co", "hcp", 4.41),
        ("Ti", "hcp", 4.87),
        ("Zr", "hcp", 6.36),
    ];

    #[test]
    fn switching_removes_the_step_at_the_cutoff_and_hardly_changes_the_crystals() {
        let energy = |calc: &Eam, sys: &MolecularSystem| {
            calc.evaluate(sys, &EvalRequest::ENERGY_ONLY)
                .unwrap()
                .energy
                .value()
                / Units::EV_TO_HARTREE
        };
        // The crystals with and without the switching (eV/atom), for several widths of the
        // range (shares of the cutoff). The crystals have a shell of neighbors at the cutoff
        // itself, sqrt(5) r_e: without the switching, rounding decides which of its atoms
        // count. No other shell lies in the range, so its width makes no difference here.
        let mut changes = Vec::new();
        for share in [0.1, 0.05, 0.02] {
            let mut largest = (0.0_f64, "");
            for (symbol, lattice, _) in CRYSTALS {
                let re = TABLE.iter().find(|(s, _, _)| *s == symbol).unwrap().1[0];
                let sys = crystal(symbol, lattice, re);
                let cut = Eam::zhou2004(&sys.elements)
                    .unwrap()
                    .with_switch_width(0.0)
                    .unwrap();
                let switched = cut.clone().with_switch_width(share * cut.cutoff).unwrap();
                let change =
                    (energy(&switched, &sys) - energy(&cut, &sys)).abs() / sys.n_atoms() as f64;
                if change > largest.0 {
                    largest = (change, symbol);
                }
            }
            println!(
                "switching over {share} of the cutoff: largest change of a cohesive energy \
                 {:.2e} eV/atom ({})",
                largest.0, largest.1
            );
            changes.push(largest.0);
        }
        assert!(changes.iter().all(|c| *c < 2e-4), "{changes:?}");
        assert!((changes[0] - changes[2]).abs() < 1e-12, "{changes:?}");
        // Two atoms pulled through the cutoff.
        let switched = Eam::zhou2004(&[element("Cu")]).unwrap();
        let cut = switched.clone().with_switch_width(0.0).unwrap();
        let pair = |r: f64, calc: &Eam| {
            let sys = MolecularSystem::new(
                vec![element("Cu"); 2],
                &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
                SpinCharge::default(),
                None,
            )
            .unwrap();
            energy(calc, &sys)
        };
        let rc = switched.cutoff;
        let step = (pair(rc - 1e-7, &switched) - pair(rc + 1e-7, &switched)).abs();
        assert!(step < 1e-14, "{step}");
        let step = (pair(rc - 1e-7, &cut) - pair(rc + 1e-7, &cut)).abs();
        assert!(step > 1e-8, "{step}");
        println!("step of the energy of two copper atoms at the cutoff without it: {step:.2e} eV");
        assert!(switched.clone().with_switch_width(rc).is_err());
    }

    #[test]
    fn the_crystals_have_the_cohesive_energies_and_lattices_of_the_fit() {
        // At the nearest-neighbour distance r_e, where the energy is lowest.
        assert_eq!(CRYSTALS.len(), ELEMENTS.len());
        for (symbol, lattice, cohesive) in CRYSTALS {
            let re = TABLE.iter().find(|(s, _, _)| *s == symbol).unwrap().1[0];
            let at = |scale: f64| energy_per_atom(&crystal(symbol, lattice, re * scale));
            let (lower, middle, upper) = (at(0.99), at(1.0), at(1.01));
            assert!(
                (middle + cohesive).abs() < 2e-4,
                "{symbol}: {middle} eV/atom against -{cohesive}"
            );
            // Minimum of the parabola through the three scales.
            let minimum = 1.0 - 0.005 * (upper - lower) / (upper - 2.0 * middle + lower);
            assert!(
                (minimum - 1.0).abs() < 1e-3,
                "{symbol}: minimum at {minimum}"
            );
        }
    }

    /// Central differences of the energy and of the gradient against the analytic gradient
    /// and Hessian.
    fn check_derivatives(sys: &MolecularSystem) {
        let calc = Eam::zhou2004(&sys.elements).unwrap();
        let out = calc.evaluate(sys, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
        let hessian = out.hessian_bohr.unwrap();
        let x0 = sys.coords_flat_bohr();
        let n = x0.nrows();
        let h = 1e-4;
        for k in 0..n {
            let at = |sign: f64| {
                let moved = Col::from_fn(n, |i| x0[i] + if i == k { sign * h } else { 0.0 });
                calc.evaluate(
                    &sys.with_flat_coords_bohr(&moved).unwrap(),
                    &EvalRequest::ENERGY_GRAD,
                )
                .unwrap()
            };
            let (plus, minus) = (at(1.0), at(-1.0));
            let slope = (plus.energy.value() - minus.energy.value()) / (2.0 * h);
            assert!(
                (slope - out.gradient_bohr[k]).abs() < 1e-8,
                "gradient {k}: {slope} against {}",
                out.gradient_bohr[k]
            );
            for i in 0..n {
                let second = (plus.gradient_bohr[i] - minus.gradient_bohr[i]) / (2.0 * h);
                assert!(
                    (second - hessian[(i, k)]).abs() < 1e-7,
                    "Hessian ({i}, {k}): {second} against {}",
                    hessian[(i, k)]
                );
            }
        }
    }

    #[test]
    fn gradient_and_hessian_are_exact_for_alloys_and_in_a_cell() {
        // A cluster of four different metals: every pair is an alloy pair.
        let cluster = MolecularSystem::new(
            vec![element("Cu"), element("Ag"), element("Ni"), element("Fe")],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.0],
                [2.7, 0.1, 0.0],
                [1.2, 2.3, 0.2],
                [1.3, 0.9, 2.1],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        check_derivatives(&cluster);
        // A cell of copper with one atom replaced by gold and the atoms off their sites.
        let mut cell = crystal("Cu", "fcc", 2.6);
        cell.elements[1] = element("Au");
        let x = cell.coords_flat_bohr();
        let moved = Col::from_fn(x.nrows(), |i| {
            x[i] + 0.15 * (((i * 7 + 3) % 5) as f64 - 2.0) / 2.0
        });
        check_derivatives(&cell.with_flat_coords_bohr(&moved).unwrap());
    }

    #[test]
    fn the_site_terms_add_up_to_the_energy_and_the_virial_with_own_images() {
        // Four atoms of an alloy in the cubic cell of fcc copper (3.68 Angstrom, under the
        // cutoff): every atom has its own images among its neighbours.
        let mut cell = crystal("Cu", "fcc", 2.6);
        cell.elements[1] = element("Au");
        let x = cell.coords_flat_bohr();
        let moved = Col::from_fn(x.nrows(), |i| {
            x[i] + 0.15 * (((i * 7 + 3) % 5) as f64 - 2.0) / 2.0
        });
        let sys = cell.with_flat_coords_bohr(&moved).unwrap();
        let calc = Eam::zhou2004(&sys.elements).unwrap();
        let out = calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        let terms = calc.site_terms(&sys).unwrap().unwrap();
        let e: f64 = terms.energies.iter().sum();
        assert!((e - out.energy.value()).abs() < 1e-12, "{e}");
        let volume = sys.active_cell().unwrap().volume_bohr3();
        let stress = out.stress.unwrap();
        for a in 0..3 {
            for b in 0..3 {
                let w: f64 = terms.virials.iter().map(|w| w[a][b]).sum();
                let expected = -stress[a][b] * volume;
                assert!((w - expected).abs() < 1e-10, "{a}{b}: {w} vs {expected}");
            }
        }
    }

    #[test]
    fn an_atom_alone_has_no_energy_and_other_elements_are_refused() {
        let atom = MolecularSystem::new(
            vec![element("W")],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert!(energy_per_atom(&atom).abs() < 2e-6);
        // The low branch of the embedding function vanishes with the density.
        for (symbol, p, _) in &TABLE {
            let sum = p[10] - p[11] + p[12] - p[13];
            assert!(sum.abs() < 2e-6, "{symbol}: {sum}");
        }
        assert!(Eam::zhou2004(&[element("Cu"), element("C")]).is_err());
        // A calculator set up for copper does not take silver.
        let copper = Eam::zhou2004(&[element("Cu")]).unwrap();
        let silver = MolecularSystem::new(
            vec![element("Ag"); 2],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [2.9, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert!(copper.evaluate(&silver, &EvalRequest::ENERGY_GRAD).is_err());
    }
}
