//! Bias-potential options shared by the commands that take MultiOptPy's `parser_for_biasforce`
//! options (`-ma`, `-kp`, ..., `-nrp`).
//!
//! Parsing gives system-independent [`BiasSpec`]s (numbers in atomic units, atom ids); each is
//! built into a [`PesTerm`] for a concrete structure with [`BiasSpec::build`], because several
//! potentials take per-element parameters. Units follow MultiOptPy's help texts: distances in
//! Angstrom, angles in degrees, energies of LJ / well / metadynamics / AFIR terms in kJ/mol,
//! spring constants in atomic units, forces in pN. Terms whose constant is zero are skipped, as
//! in MultiOptPy.
//!
//! MultiOptPy: `interface.py:parser_for_biasforce`, `force_data_parser`, and
//!  `Potential/potential.py:make_bias_pot_obj_list`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::pes::bias::{
    AdBias, Afir, AngleContinuation, AsymmetricEllipsoids, BondRange, CollectiveVariable,
    ConeRepulsive, DihedralForm, Ellipsoid, EllipsoidVariant, ExpressionBias, Flux, KeepAngle,
    KeepDihedral, KeepDistance, KeepOutOfPlane, LjCenter, LjGaussian, LjParameters, LjRepulsive,
    Metadynamics, MetadynamicsCv, MicroBias, MorseDistance, NanoReactor, PullingForce, Ramp,
    SpacerModel, Universal, VoidPoint, Well, WellGeometry, WellPotential,
};
use crate::pes::PesTerm;

/// The settings of the relaxation of the auxiliary variables of a bias: the defaults, with the
/// methods of the optimizer registry for the fallback to FIRE (`crate::pes::microiter`).
fn auxiliary_relaxation() -> crate::pes::microiter::MicroSettings {
    crate::opt::micro::settings()
}

const A2B: f64 = Units::ANGSTROM_TO_BOHR;
const DEG: f64 = Units::DEG_TO_RAD;
const KJ: f64 = Units::KJ_MOL_TO_HARTREE;

/// One parsed bias option.
#[derive(Debug, Clone)]
pub struct BiasSpec {
    /// Option key (MultiOptPy's argparse destination), used in term labels.
    pub key: String,
    /// The potential.
    pub kind: BiasKind,
}

/// Builds a user-registered bias for a structure (see [`BiasRegistry`]).
pub trait BiasBuilder: Send + Sync + std::fmt::Debug {
    /// Term for `sys`.
    fn build(&self, sys: &MolecularSystem) -> MoptResult<Arc<dyn PesTerm>>;
}

/// Parser of a user-registered bias option: arguments to builders.
pub type CustomBiasParser =
    Arc<dyn Fn(&[String]) -> MoptResult<Vec<Arc<dyn BiasBuilder>>> + Send + Sync>;

/// Bias options by flag: the built-in ones and those registered by users.
#[derive(Clone, Default)]
pub struct BiasRegistry {
    custom: Vec<(String, String, CustomBiasParser)>,
}

impl std::fmt::Debug for BiasRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.custom.iter().map(|c| &c.0))
            .finish()
    }
}

impl BiasRegistry {
    /// Registry with the built-in options only.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register option `flag` (e.g. `"-mybias"`) with term labels `key_0`, `key_1`, ...
    pub fn register(&mut self, flag: &str, key: &str, parser: CustomBiasParser) -> MoptResult<()> {
        let taken = PARSERS.iter().any(|p| p.1 == flag || p.0 == key)
            || self.custom.iter().any(|c| c.0 == flag || c.1 == key);
        if taken || !flag.starts_with('-') || key.is_empty() {
            return Err(MoptError::config(format!(
                "cannot register bias option '{flag}' ('{key}'): taken or invalid"
            )));
        }
        self.custom
            .push((flag.to_string(), key.to_string(), parser));
        Ok(())
    }

    /// Flags and keys of all options.
    pub fn options(&self) -> Vec<(String, String)> {
        PARSERS
            .iter()
            .map(|p| (p.1.to_string(), p.0.to_string()))
            .chain(self.custom.iter().map(|c| (c.0.clone(), c.1.clone())))
            .collect()
    }

    /// Biases of option `flag` (or its key) with arguments `tokens`.
    pub fn parse_option(&self, flag: &str, tokens: &[String]) -> MoptResult<Vec<BiasSpec>> {
        let bare = flag.trim_start_matches('-');
        if let Some(c) = self.custom.iter().find(|c| c.0 == flag || c.1 == bare) {
            return Ok((c.2)(tokens)?
                .into_iter()
                .map(|b| BiasSpec {
                    key: c.1.clone(),
                    kind: BiasKind::Custom(b),
                })
                .collect());
        }
        let &(key, flag, parse) = PARSERS
            .iter()
            .find(|p| p.1 == flag || p.0 == bare)
            .ok_or_else(|| MoptError::parse(format!("unknown bias option '{flag}'")))?;
        Ok(parse(flag, tokens)?
            .into_iter()
            .map(|kind| BiasSpec {
                key: key.to_string(),
                kind,
            })
            .collect())
    }
}

/// Parsed parameters of a bias option (atomic units).
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum BiasKind {
    Afir(Afir),
    Distance(KeepDistance),
    Morse(MorseDistance),
    Angle(KeepAngle),
    Dihedral(KeepDihedral),
    OutOfPlane(KeepOutOfPlane),
    VoidPoint(VoidPoint),
    Universal(Universal),
    Flux(Flux),
    BondRange(BondRange),
    Pulling(PullingForce),
    Well(WellPotential),
    /// `-rp`: fragments and parameters (resolved with the elements).
    Lj {
        a: Vec<AtomId>,
        b: Vec<AtomId>,
        params: LjParameters,
    },
    /// `-rpv2`.
    LjCenter {
        center: [AtomId; 2],
        length: f64,
        targets: Vec<AtomId>,
        params: LjParameters,
        constants: [f64; 2],
        orders: [f64; 2],
    },
    LjGaussian(LjGaussian),
    /// `-cp`.
    Cone {
        center: AtomId,
        three: [AtomId; 3],
        targets: Vec<AtomId>,
        well: f64,
        distance: f64,
        angle: f64,
    },
    /// `-smp`.
    Spacer {
        targets: Vec<AtomId>,
        eps: f64,
        sigma: f64,
        cavity: f64,
        particles: usize,
    },
    /// `-metad` (all variables in one bias).
    Metadynamics(Vec<MetadynamicsCv>),
    /// `-aerp` / `-aerpv2` (all ellipsoids of an option in one bias).
    Ellipsoids(Vec<Ellipsoid>, EllipsoidVariant),
    /// `-nrp`; masses are taken from the structure.
    NanoReactor(NanoReactor),
    /// `-ubias`: an expression.
    Expression(ExpressionBias),
    /// A user-registered option.
    Custom(Arc<dyn BiasBuilder>),
}

fn num(opt: &str, t: &str) -> MoptResult<f64> {
    t.trim()
        .parse::<f64>()
        .map_err(|e| MoptError::parse(format!("{opt}: invalid number '{t}' ({e})")))
}

fn nums<const N: usize>(opt: &str, t: &str) -> MoptResult<[f64; N]> {
    let v = t
        .split(',')
        .map(|x| num(opt, x))
        .collect::<MoptResult<Vec<f64>>>()?;
    v.try_into().map_err(|v: Vec<f64>| {
        MoptError::parse(format!(
            "{opt}: expected {N} comma-separated values, got {}",
            v.len()
        ))
    })
}

fn atoms(t: &str) -> MoptResult<Vec<AtomId>> {
    parse_atom_indices_one_based(t)?
        .into_iter()
        .map(AtomId::from_one_based)
        .collect()
}

fn fixed<const N: usize>(opt: &str, t: &str) -> MoptResult<[AtomId; N]> {
    atoms(t)?.try_into().map_err(|v: Vec<AtomId>| {
        MoptError::parse(format!(
            "{opt}: expected {N} atoms, got {} in '{t}'",
            v.len()
        ))
    })
}

fn ramp(opt: &str, t: &str, scale: f64) -> MoptResult<Ramp> {
    let r = Ramp::parse(t).map_err(|e| MoptError::parse(format!("{opt}: {e}")))?;
    Ok(Ramp {
        start: r.start * scale,
        end: r.end.map(|e| e * scale),
    })
}

fn chunks<'a>(
    opt: &str,
    tokens: &'a [String],
    n: usize,
) -> MoptResult<std::slice::Chunks<'a, String>> {
    if tokens.len().is_multiple_of(n) {
        Ok(tokens.chunks(n))
    } else {
        Err(MoptError::parse(format!(
            "{opt} expects groups of {n} arguments, got {}",
            tokens.len()
        )))
    }
}

fn lj_params(opt: &str, mode: &str, well: f64, dist: f64) -> MoptResult<LjParameters> {
    match mode.to_ascii_lowercase().as_str() {
        "scale" => Ok(LjParameters::Scale {
            well,
            distance: dist,
        }),
        "value" => Ok(LjParameters::Value {
            well: well * KJ,
            distance: dist * A2B,
        }),
        other => Err(MoptError::parse(format!(
            "{opt}: last argument must be 'scale' or 'value', got '{other}'"
        ))),
    }
}

fn well(opt: &str, energy: &str, limits: &str) -> MoptResult<Well> {
    let l = nums::<4>(opt, limits)?;
    let w = Well {
        wall: num(opt, energy)? * KJ,
        limits: l.map(|x| x * A2B),
    };
    if w.is_valid() {
        Ok(w)
    } else {
        Err(MoptError::parse(format!(
            "{opt}: limits must satisfy a < b < c < d, got '{limits}'"
        )))
    }
}

/// Options in the order they are applied, with their parsers.
type Parser = fn(&'static str, &[String]) -> MoptResult<Vec<BiasKind>>;

const PARSERS: &[(&str, &str, Parser)] = &[
    ("manual_AFIR", "-ma", parse_afir),
    ("repulsive_potential", "-rp", parse_rp),
    ("repulsive_potential_v2", "-rpv2", parse_rpv2),
    ("repulsive_potential_gaussian", "-rpg", parse_rpg),
    ("cone_potential", "-cp", parse_cp),
    ("flux_potential", "-fp", parse_fp),
    ("keep_pot", "-kp", parse_kp),
    ("keep_pot_v2", "-kpv2", parse_kpv2),
    ("anharmonic_keep_pot", "-akp", parse_akp),
    ("keep_angle", "-ka", parse_ka),
    ("keep_angle_v2", "-kav2", parse_kav2),
    ("universal_potential", "-up", parse_up),
    ("keep_dihedral_angle", "-kda", parse_kda),
    ("keep_out_of_plain_angle", "-kopa", parse_kopa),
    ("keep_dihedral_angle_v2", "-kdav2", parse_kdav2),
    ("keep_dihedral_angle_cos", "-kdac", parse_kdac),
    ("keep_out_of_plain_angle_v2", "-kopav2", parse_kopav2),
    ("void_point_pot", "-vpp", parse_vpp),
    ("bond_range_potential", "-brp", parse_brp),
    ("well_pot", "-wp", parse_wp),
    ("wall_well_pot", "-wwp", parse_wwp),
    ("void_point_well_pot", "-vpwp", parse_vpwp),
    ("around_well_pot", "-awp", parse_awp),
    ("spacer_model_potential", "-smp", parse_smp),
    ("metadynamics", "-metad", parse_metad),
    ("linear_mechano_force_pot", "-lmefp", parse_lmefp),
    ("linear_mechano_force_pot_v2", "-lmefpv2", parse_lmefpv2),
    (
        "asymmetric_ellipsoidal_repulsive_potential",
        "-aerp",
        parse_aerp,
    ),
    (
        "asymmetric_ellipsoidal_repulsive_potential_v2",
        "-aerpv2",
        parse_aerpv2,
    ),
    ("nano_reactor_potential", "-nrp", parse_nrp),
    ("user_bias_expression", "-ubias", parse_ubias),
];

/// Option keys handled here.
pub fn bias_keys() -> impl Iterator<Item = &'static str> {
    PARSERS.iter().map(|p| p.0)
}

/// All built-in bias options of `p`.
pub fn parse_biases(p: &ParsedArgs) -> MoptResult<Vec<BiasSpec>> {
    let mut out = Vec::new();
    for &(key, flag, parse) in PARSERS {
        let tokens = p.get_list(key);
        if tokens.is_empty() {
            continue;
        }
        for kind in parse(flag, tokens)? {
            out.push(BiasSpec {
                key: key.to_string(),
                kind,
            });
        }
    }
    Ok(out)
}

/// The built-in bias options of `p` as written: flag and arguments, in the order they are
/// applied.
pub fn present_bias_options(p: &ParsedArgs) -> Vec<(String, Vec<String>)> {
    PARSERS
        .iter()
        .filter_map(|&(key, flag, _)| {
            let tokens = p.get_list(key);
            (!tokens.is_empty()).then(|| (flag.to_string(), tokens.to_vec()))
        })
        .collect()
}

/// Built-in bias option given by flag (`"-kp"`) or key and its tokens, e.g. from Python.
pub fn parse_bias_option(flag: &str, tokens: &[String]) -> MoptResult<Vec<BiasSpec>> {
    BiasRegistry::new().parse_option(flag, tokens)
}

impl BiasSpec {
    /// Term for `sys`.
    pub fn build(&self, sys: &MolecularSystem) -> MoptResult<Arc<dyn PesTerm>> {
        let el = &sys.elements;
        Ok(match &self.kind {
            BiasKind::Afir(t) => Arc::new(t.clone()),
            BiasKind::Distance(b) => Arc::new(AdBias::new(b.clone())),
            BiasKind::Morse(b) => {
                b.validate()?;
                Arc::new(AdBias::new(b.clone()))
            }
            BiasKind::Angle(b) => Arc::new(AdBias::new(b.clone())),
            BiasKind::Dihedral(b) => Arc::new(AdBias::new(b.clone())),
            BiasKind::OutOfPlane(b) => Arc::new(AdBias::new(b.clone())),
            BiasKind::VoidPoint(b) => {
                b.validate()?;
                Arc::new(AdBias::new(b.clone()))
            }
            BiasKind::Universal(b) => Arc::new(AdBias::new(b.clone())),
            BiasKind::Flux(b) => {
                b.validate()?;
                Arc::new(AdBias::new(b.clone()))
            }
            BiasKind::BondRange(b) => Arc::new(AdBias::new(b.clone())),
            BiasKind::Pulling(b) => Arc::new(AdBias::new(b.clone())),
            BiasKind::Well(b) => Arc::new(AdBias::new(b.clone())),
            BiasKind::Lj { a, b, params } => {
                Arc::new(AdBias::new(LjRepulsive::new(el, a, b, *params)))
            }
            BiasKind::LjCenter {
                center,
                length,
                targets,
                params,
                constants,
                orders,
            } => Arc::new(AdBias::new(LjCenter::new(
                el, *center, *length, targets, *params, *constants, *orders,
            ))),
            BiasKind::LjGaussian(b) => Arc::new(AdBias::new(b.clone())),
            BiasKind::Cone {
                center,
                three,
                targets,
                well,
                distance,
                angle,
            } => Arc::new(AdBias::new(ConeRepulsive::new(
                el, *center, *three, targets, *well, *distance, *angle,
            ))),
            BiasKind::Spacer {
                targets,
                eps,
                sigma,
                cavity,
                particles,
            } => Arc::new(MicroBias::with_settings(
                SpacerModel::new(el, targets, *eps, *sigma, *cavity, *particles),
                auxiliary_relaxation(),
            )),
            BiasKind::Metadynamics(cvs) => Arc::new(AdBias::new(Metadynamics::new(cvs.clone()))),
            BiasKind::Ellipsoids(list, variant) => Arc::new(MicroBias::with_settings(
                AsymmetricEllipsoids::new(el, list.clone(), *variant),
                auxiliary_relaxation(),
            )),
            BiasKind::NanoReactor(n) => {
                let mut n = n.clone();
                n.atoms = sys
                    .atoms()
                    .map(|a| (a, sys.element(a).mass_amu()))
                    .collect();
                Arc::new(AdBias::new(n))
            }
            // With cell functions (`volume()`, `cell_a()`, ...) the term that sees the lattice
            // and gives the stress (`pes::bias::CellExpression`); otherwise the term of atoms.
            BiasKind::Expression(e) => e.clone().into_term(),
            BiasKind::Custom(b) => b.build(sys)?,
        })
    }

    /// Label of the `k`-th term of this option.
    pub fn label(&self, k: usize) -> String {
        match self.key.as_str() {
            // MultiOptPy's name for AFIR terms.
            "manual_AFIR" => format!("AFIR_pot_{k}"),
            key => format!("{key}_{k}"),
        }
    }
}

fn parse_afir(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 3)? {
        let g = Ramp::parse(&c[0]).map_err(|e| MoptError::parse(format!("{opt}: {e}")))?;
        if g.has_zero() {
            continue;
        }
        let term = Afir::new(g.start, atoms(&c[1])?, atoms(&c[2])?)?;
        out.push(BiasKind::Afir(match g.end {
            Some(e) => term.ramp_to(e),
            None => term,
        }));
    }
    Ok(out)
}

fn parse_rp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 5)? {
        let (w, d) = (num(opt, &c[0])?, num(opt, &c[1])?);
        if w == 0.0 {
            continue;
        }
        out.push(BiasKind::Lj {
            a: atoms(&c[2])?,
            b: atoms(&c[3])?,
            params: lj_params(opt, &c[4], w, d)?,
        });
    }
    Ok(out)
}

fn parse_rpv2(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 10)? {
        let (w, d) = (num(opt, &c[0])?, num(opt, &c[1])?);
        if w == 0.0 {
            continue;
        }
        out.push(BiasKind::LjCenter {
            length: num(opt, &c[2])? * A2B,
            constants: [num(opt, &c[3])?, num(opt, &c[4])?],
            orders: [num(opt, &c[5])?, num(opt, &c[6])?],
            center: fixed(opt, &c[7])?,
            targets: atoms(&c[8])?,
            params: lj_params(opt, &c[9], w, d)?,
        });
    }
    Ok(out)
}

fn parse_rpg(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 7)? {
        let (a, b) = (atoms(&c[5])?, atoms(&c[6])?);
        let pairs = a
            .iter()
            .flat_map(|&i| b.iter().filter(move |&&j| j != i).map(move |&j| (i, j)))
            .collect();
        out.push(BiasKind::LjGaussian(LjGaussian {
            pairs,
            lj_well: num(opt, &c[0])? * KJ,
            lj_distance: num(opt, &c[1])? * A2B,
            gauss_well: num(opt, &c[2])? * KJ,
            gauss_distance: num(opt, &c[3])? * A2B,
            gauss_range: num(opt, &c[4])? * A2B,
        }));
    }
    Ok(out)
}

fn parse_cp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 6)? {
        let well = num(opt, &c[0])?;
        if well == 0.0 {
            continue;
        }
        out.push(BiasKind::Cone {
            well: well * KJ,
            distance: num(opt, &c[1])? * A2B,
            angle: num(opt, &c[2])? * DEG,
            center: fixed::<1>(opt, &c[3])?[0],
            three: fixed(opt, &c[4])?,
            targets: atoms(&c[5])?,
        });
    }
    Ok(out)
}

fn parse_fp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 4)? {
        let k = nums::<3>(opt, &c[0])?;
        if k.iter().all(|&x| x == 0.0) {
            continue;
        }
        let flux = Flux {
            k,
            order: nums(opt, &c[1])?,
            point: nums::<3>(opt, &c[2])?.map(|x| x * A2B),
            atoms: atoms(&c[3])?,
        };
        flux.validate()?;
        out.push(BiasKind::Flux(flux));
    }
    Ok(out)
}

fn distance(
    opt: &str,
    k: &str,
    r0: &str,
    a: Vec<AtomId>,
    b: Vec<AtomId>,
) -> MoptResult<Option<BiasKind>> {
    let k = ramp(opt, k, 1.0)?;
    if k.has_zero() && k.end.is_none() {
        return Ok(None);
    }
    Ok(Some(BiasKind::Distance(KeepDistance {
        a,
        b,
        k,
        r0: ramp(opt, r0, A2B)?,
    })))
}

fn parse_kp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 3)? {
        let [a, b] = fixed(opt, &c[2])?;
        out.extend(distance(opt, &c[0], &c[1], vec![a], vec![b])?);
    }
    Ok(out)
}

fn parse_kpv2(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 4)? {
        out.extend(distance(opt, &c[0], &c[1], atoms(&c[2])?, atoms(&c[3])?)?);
    }
    Ok(out)
}

fn parse_akp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 4)? {
        let depth = num(opt, &c[0])?;
        if depth == 0.0 {
            continue;
        }
        let morse = MorseDistance {
            depth,
            k: num(opt, &c[1])?,
            r0: num(opt, &c[2])? * A2B,
            pair: fixed(opt, &c[3])?,
        };
        morse.validate()?;
        out.push(BiasKind::Morse(morse));
    }
    Ok(out)
}

fn angle(
    opt: &str,
    k: &str,
    theta: &str,
    groups: [Vec<AtomId>; 3],
    c: AngleContinuation,
) -> MoptResult<Option<BiasKind>> {
    let k = ramp(opt, k, 1.0)?;
    if k.has_zero() && k.end.is_none() {
        return Ok(None);
    }
    Ok(Some(BiasKind::Angle(KeepAngle {
        groups,
        k,
        theta0: ramp(opt, theta, DEG)?,
        continuation: c,
    })))
}

fn parse_ka(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 3)? {
        let [a, b, d] = fixed(opt, &c[2])?;
        let g = [vec![a], vec![b], vec![d]];
        out.extend(angle(opt, &c[0], &c[1], g, AngleContinuation::Series)?);
    }
    Ok(out)
}

fn parse_kav2(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 5)? {
        let g = [atoms(&c[2])?, atoms(&c[3])?, atoms(&c[4])?];
        out.extend(angle(opt, &c[0], &c[1], g, AngleContinuation::Quadratic)?);
    }
    Ok(out)
}

fn parse_up(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 2)? {
        let v = num(opt, &c[0])?;
        if v == 0.0 {
            continue;
        }
        out.push(BiasKind::Universal(Universal {
            c: v * KJ,
            atoms: atoms(&c[1])?,
        }));
    }
    Ok(out)
}

fn dihedral(
    opt: &str,
    k: &str,
    phi: &str,
    groups: [Vec<AtomId>; 4],
    form: DihedralForm,
) -> MoptResult<Option<BiasKind>> {
    let k = ramp(opt, k, 1.0)?;
    if k.has_zero() && k.end.is_none() {
        return Ok(None);
    }
    Ok(Some(BiasKind::Dihedral(KeepDihedral {
        groups,
        k,
        phi0: ramp(opt, phi, DEG)?,
        form,
    })))
}

fn single4(opt: &str, t: &str) -> MoptResult<[Vec<AtomId>; 4]> {
    Ok(fixed::<4>(opt, t)?.map(|a| vec![a]))
}

fn parse_kda(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 3)? {
        out.extend(dihedral(
            opt,
            &c[0],
            &c[1],
            single4(opt, &c[2])?,
            DihedralForm::Harmonic,
        )?);
    }
    Ok(out)
}

fn groups4(c: &[String]) -> MoptResult<[Vec<AtomId>; 4]> {
    Ok([atoms(&c[0])?, atoms(&c[1])?, atoms(&c[2])?, atoms(&c[3])?])
}

fn parse_kdav2(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 6)? {
        out.extend(dihedral(
            opt,
            &c[0],
            &c[1],
            groups4(&c[2..])?,
            DihedralForm::Harmonic,
        )?);
    }
    Ok(out)
}

fn parse_kdac(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 7)? {
        let form = DihedralForm::Cosine {
            n: num(opt, &c[1])?,
        };
        out.extend(dihedral(opt, &c[0], &c[2], groups4(&c[3..])?, form)?);
    }
    Ok(out)
}

fn oop(opt: &str, k: &str, chi: &str, groups: [Vec<AtomId>; 4]) -> MoptResult<Option<BiasKind>> {
    let k = ramp(opt, k, 1.0)?;
    if k.has_zero() && k.end.is_none() {
        return Ok(None);
    }
    Ok(Some(BiasKind::OutOfPlane(KeepOutOfPlane {
        groups,
        k,
        chi0: ramp(opt, chi, DEG)?,
    })))
}

fn parse_kopa(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 3)? {
        out.extend(oop(opt, &c[0], &c[1], single4(opt, &c[2])?)?);
    }
    Ok(out)
}

fn parse_kopav2(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 6)? {
        out.extend(oop(opt, &c[0], &c[1], groups4(&c[2..])?)?);
    }
    Ok(out)
}

fn parse_vpp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 5)? {
        let k = num(opt, &c[0])?;
        if k == 0.0 {
            continue;
        }
        let (r0, point, order) = (
            num(opt, &c[1])? * A2B,
            nums::<3>(opt, &c[2])?.map(|x| x * A2B),
            num(opt, &c[4])?,
        );
        for atom in atoms(&c[3])? {
            let term = VoidPoint {
                atom,
                point,
                k,
                r0,
                order,
            };
            term.validate()?;
            out.push(BiasKind::VoidPoint(term));
        }
    }
    Ok(out)
}

fn parse_brp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 6)? {
        out.push(BiasKind::BondRange(BondRange {
            upper_const: num(opt, &c[0])?,
            lower_const: num(opt, &c[1])?,
            upper: num(opt, &c[2])? * A2B,
            lower: num(opt, &c[3])? * A2B,
            a: atoms(&c[4])?,
            b: atoms(&c[5])?,
        }));
    }
    Ok(out)
}

fn well_kind(well: Well, geometry: WellGeometry) -> BiasKind {
    BiasKind::Well(WellPotential { well, geometry })
}

fn parse_wp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    chunks(opt, t, 4)?
        .map(|c| {
            let g = WellGeometry::Fragments {
                a: atoms(&c[1])?,
                b: atoms(&c[2])?,
            };
            Ok(well_kind(well(opt, &c[0], &c[3])?, g))
        })
        .collect()
}

fn parse_wwp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    chunks(opt, t, 4)?
        .map(|c| {
            let axis = match c[1].trim().to_ascii_lowercase().as_str() {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                other => {
                    return Err(MoptError::parse(format!(
                        "{opt}: direction must be x, y, or z, got '{other}'"
                    )))
                }
            };
            let g = WellGeometry::Wall {
                axis,
                targets: atoms(&c[3])?,
            };
            Ok(well_kind(well(opt, &c[0], &c[2])?, g))
        })
        .collect()
}

fn parse_vpwp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    chunks(opt, t, 4)?
        .map(|c| {
            let g = WellGeometry::Point {
                point: nums::<3>(opt, &c[1])?.map(|x| x * A2B),
                targets: atoms(&c[3])?,
            };
            Ok(well_kind(well(opt, &c[0], &c[2])?, g))
        })
        .collect()
}

fn parse_awp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    chunks(opt, t, 4)?
        .map(|c| {
            let g = WellGeometry::Around {
                center: atoms(&c[1])?,
                targets: atoms(&c[3])?,
            };
            Ok(well_kind(well(opt, &c[0], &c[2])?, g))
        })
        .collect()
}

fn parse_smp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    chunks(opt, t, 5)?
        .map(|c| {
            let n = c[3]
                .trim()
                .parse::<usize>()
                .map_err(|e| MoptError::parse(format!("{opt}: particle number '{}': {e}", c[3])))?;
            Ok(BiasKind::Spacer {
                eps: num(opt, &c[0])? * KJ,
                sigma: num(opt, &c[1])? * A2B,
                cavity: num(opt, &c[2])?,
                particles: n,
                targets: atoms(&c[4])?,
            })
        })
        .collect()
}

fn parse_metad(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut cvs = Vec::new();
    for c in chunks(opt, t, 4)? {
        let height = num(opt, &c[1])? * KJ;
        let w = num(opt, &c[2])?;
        let (cv, width) = match c[0].trim().to_ascii_lowercase().as_str() {
            "bond" => (CollectiveVariable::Bond(fixed(opt, &c[3])?), w * A2B),
            "angle" => (CollectiveVariable::Angle(fixed(opt, &c[3])?), w * DEG),
            "dihedral" => (CollectiveVariable::Dihedral(fixed(opt, &c[3])?), w * DEG),
            "outofplain" | "outofplane" => {
                (CollectiveVariable::OutOfPlane(fixed(opt, &c[3])?), w * DEG)
            }
            other => {
                return Err(MoptError::parse(format!(
                    "{opt}: variable must be bond, angle, dihedral, or outofplain, got '{other}'"
                )))
            }
        };
        cvs.push(MetadynamicsCv { cv, height, width });
    }
    Ok(vec![BiasKind::Metadynamics(cvs)])
}

fn parse_lmefp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 3)? {
        let f = num(opt, &c[0])?;
        if f == 0.0 {
            continue;
        }
        out.push(BiasKind::Pulling(PullingForce {
            pairs: vec![fixed(opt, &c[1])?, fixed(opt, &c[2])?],
            force: f * Units::PN_TO_AU_FORCE,
        }));
    }
    Ok(out)
}

fn parse_lmefpv2(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let mut out = Vec::new();
    for c in chunks(opt, t, 2)? {
        let f = num(opt, &c[0])?;
        if f == 0.0 {
            continue;
        }
        out.push(BiasKind::Pulling(PullingForce {
            pairs: vec![fixed(opt, &c[1])?],
            force: f * Units::PN_TO_AU_FORCE,
        }));
    }
    Ok(out)
}

fn parse_ellipsoids(
    opt: &str,
    t: &[String],
    variant: EllipsoidVariant,
) -> MoptResult<Vec<BiasKind>> {
    let list = chunks(opt, t, 5)?
        .map(|c| {
            let [root, axis_atom] = fixed(opt, &c[3])?;
            Ok(Ellipsoid {
                root,
                axis_atom,
                eps: num(opt, &c[0])? * KJ,
                sigma: nums::<6>(opt, &c[1])?.map(|x| x * A2B),
                distance: num(opt, &c[2])? * A2B,
                off_targets: if c[4].trim().is_empty() || c[4].trim() == "0" {
                    Vec::new()
                } else {
                    atoms(&c[4])?
                },
            })
        })
        .collect::<MoptResult<Vec<_>>>()?;
    Ok(vec![BiasKind::Ellipsoids(list, variant)])
}

fn parse_aerp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    parse_ellipsoids(opt, t, EllipsoidVariant::Gnb)
}

fn parse_aerpv2(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    parse_ellipsoids(opt, t, EllipsoidVariant::Uff)
}

fn parse_ubias(_opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    t.iter()
        .map(|e| ExpressionBias::parse(e).map(BiasKind::Expression))
        .collect()
}

fn parse_nrp(opt: &'static str, t: &[String]) -> MoptResult<Vec<BiasKind>> {
    let [inner, outer, t1, t2, k1, k2] = t
        .iter()
        .map(|x| num(opt, x))
        .collect::<MoptResult<Vec<f64>>>()?
        .try_into()
        .map_err(|_| {
            MoptError::parse(format!(
                "{opt} expects 6 values: inner wall, outer wall (Angstrom), contraction and \
                 expansion times (ps), contraction and expansion force constants \
                 (kcal/mol/Angstrom^2)"
            ))
        })?;
    let ps = 1e-12 / Units::AU_TIME_TO_SEC;
    let k = Units::KCAL_MOL_TO_HARTREE / (A2B * A2B);
    Ok(vec![BiasKind::NanoReactor(NanoReactor {
        atoms: Vec::new(),
        center: crate::pes::bias::nanoreactor::MULTIOPTPY_CENTER,
        r1: outer * A2B,
        k1: k1 * k,
        r2: inner * A2B,
        k2: k2 * k,
        tau: t1 * ps,
        period: (t1 + t2) * ps,
    })])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_multioptpy_argument_groups() {
        let kp = parse_bias_option("-kp", &s(&["0.5", "1.2", "1,2", "0.0", "1.0", "2,3"])).unwrap();
        assert_eq!(kp.len(), 1, "zero spring constants are skipped");
        let BiasKind::Distance(d) = &kp[0].kind else {
            panic!("kp")
        };
        assert!((d.r0.start - 1.2 * A2B).abs() < 1e-14);
        assert_eq!(d.a, vec![AtomId::new(0)]);
        assert_eq!(kp[0].label(0), "keep_pot_0");
        assert!(parse_bias_option("-kp", &s(&["0.5", "1.2"])).is_err());
        assert!(parse_bias_option("-wp", &s(&["10", "1", "2", "3,2,1,0"])).is_err());
        let metad = parse_bias_option(
            "-metad",
            &s(&[
                "bond", "1.0", "0.1", "1,2", "dihedral", "1.0", "10", "1,2,3,4",
            ]),
        )
        .unwrap();
        assert_eq!(metad.len(), 1);
        let nrp = parse_bias_option("-nrp", &s(&["8", "14", "1.5", "0.5", "1.0", "0.5"])).unwrap();
        let BiasKind::NanoReactor(n) = &nrp[0].kind else {
            panic!("nrp")
        };
        assert!((n.r1 - 14.0 * A2B).abs() < 1e-12 && n.tau / n.period == 0.75);
        assert!(parse_bias_option("-xyz", &[]).is_err());
        let u = parse_bias_option("-ubias", &s(&["0.1 * dist(1, 2)^2", "x(3)"])).unwrap();
        assert_eq!(u.len(), 2);
        assert!(parse_bias_option("-ubias", &s(&["dist(1)"])).is_err());
    }

    /// Constants for which every energy would be NaN are refused when the option is read,
    /// with the option named.
    #[test]
    fn constants_without_a_real_energy_are_refused() {
        let err = |flag: &str, args: &[&str]| {
            parse_bias_option(flag, &s(args))
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default()
        };
        // -akp: sqrt(k / 2 De) is not real for a negative depth or a negative k.
        assert!(err("-akp", &["-0.1", "0.3", "1.0", "1,2"]).contains("-akp"));
        assert!(err("-akp", &["0.1", "-0.3", "1.0", "1,2"]).contains("-akp"));
        assert!(parse_bias_option("-akp", &s(&["0.1", "0.3", "1.0", "1,2"])).is_ok());
        // -vpp: (|r - P| - r0)^p inside the sphere needs an integer order.
        assert!(err("-vpp", &["0.1", "1.0", "0,0,0", "1", "2.5"]).contains("-vpp"));
        assert!(err("-vpp", &["0.1", "1.0", "0,0,0", "1", "0"]).contains("-vpp"));
        assert!(parse_bias_option("-vpp", &s(&["0.1", "1.0", "0,0,0", "1", "3"])).is_ok());
        // -fp: the components take both signs.
        assert!(err("-fp", &["0.1,0,0", "1.5,1,1", "0,0,0", "1"]).contains("-fp"));
        assert!(parse_bias_option("-fp", &s(&["0.1,0,0", "2,1.5,1", "0,0,0", "1"])).is_ok());
    }

    /// The parsed values that produced NaN before they were refused.
    #[test]
    fn the_refused_constants_did_give_nan() {
        use crate::domain::{positions, Angstrom, Element, SpinCharge};
        use crate::pes::TermContext;
        let sys = MolecularSystem::new(
            vec![Element::H, Element::H],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let ctx = TermContext::at_iteration(0, &sys);
        let morse = MorseDistance {
            pair: [AtomId::new(0), AtomId::new(1)],
            depth: -0.1,
            k: 0.3,
            r0: 1.0,
        };
        let void = VoidPoint {
            atom: AtomId::new(1),
            point: [0.0; 3],
            k: 0.1,
            r0: 2.0,
            order: 2.5,
        };
        for term in [
            Arc::new(AdBias::new(morse)) as Arc<dyn PesTerm>,
            Arc::new(AdBias::new(void)),
        ] {
            assert!(term.evaluate(&sys, &ctx, false).is_err(), "{}", term.name());
        }
    }

    #[derive(Debug)]
    struct Zero;

    impl BiasBuilder for Zero {
        fn build(&self, _sys: &MolecularSystem) -> MoptResult<Arc<dyn PesTerm>> {
            Ok(Arc::new(AdBias::new(
                ExpressionBias::parse("0 * dist(1, 2)").unwrap(),
            )))
        }
    }

    #[test]
    fn user_options_can_be_registered() {
        let mut r = BiasRegistry::new();
        let parser: CustomBiasParser =
            Arc::new(|t: &[String]| Ok(t.iter().map(|_| Arc::new(Zero) as _).collect()));
        r.register("-zero", "zero_pot", parser.clone()).unwrap();
        assert!(r.register("-kp", "x", parser.clone()).is_err());
        assert!(r.register("-zero", "y", parser).is_err());
        let specs = r.parse_option("-zero", &s(&["a", "b"])).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[1].label(1), "zero_pot_1");
        assert!(r.options().iter().any(|o| o.0 == "-zero"));
    }
}
