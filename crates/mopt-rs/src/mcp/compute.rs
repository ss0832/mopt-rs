//! The component tools that compute from structures given as text: model Hessians, normal
//! modes and thermochemistry of a given Hessian, coordinate systems, the first images between
//! two structures, superposition, the bond graph, dispersion, Hessian updates, units and
//! elements.
//!
//! Both servers offer them with the same definitions and the same numbers: the Rust server
//! calls [`compute`] directly, the server of the Python package through
//! `mopt_rs._core.mcp_component`. A structure argument is a file under the root (the property
//! named in [`structure_arguments`], read by the server through its sandbox) or its xyz text
//! (the property with the `_xyz` name, or `xyz` for the first structure); [`compute`] only
//! receives texts, so that nothing here opens a path.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use faer::{Col, Mat};

use super::protocol::{count, object, text};
use super::schema::{prop, tool};
use crate::coord::alignment::kabsch_align_bohr;
use crate::coord::connectivity::BondConnectivity;
use crate::coord::system::CoordinateRegistry;
use crate::core_infra::json::JsonValue;
use crate::domain::{AtomId, Element, MolecularSystem, SpinCharge, Units};
use crate::io::xyz::{format_xyz_system, parse_xyz_str};
use crate::model_hessian::{base_names, resolve};
use crate::opt::update::UpdaterRegistry;
use crate::parameters::covalent_radius_bohr;
use crate::path::interp::{self, FirstPath, PathFrame, PathSettings};
use crate::pes::dispersion::{D3Bj, D4};
use crate::vib::{self, FrequencySettings};

/// The tools of this module, in the order of `tools/list`.
pub const TOOLS: [&str; 10] = [
    "model_hessian",
    "vibrations",
    "coordinates",
    "interpolate",
    "superpose",
    "bond_graph",
    "dispersion",
    "hessian_update",
    "units",
    "elements",
];

/// The most atoms of a structure of these tools.
pub const MAX_ATOMS: usize = 2000;
/// The most atoms of the tools that answer with a Hessian (its `9 N^2` numbers).
pub const MAX_HESSIAN_ATOMS: usize = 300;
/// The most images of `interpolate`.
pub const MAX_IMAGES: usize = 200;

/// The structure arguments of the tool `name`: `(file property, text property)`.
pub fn structure_arguments(name: &str) -> &'static [(&'static str, &'static str)] {
    match name {
        "model_hessian" | "vibrations" | "coordinates" | "bond_graph" | "dispersion" => {
            &[("structure", "xyz")]
        }
        "superpose" => &[("structure", "xyz"), ("reference", "reference_xyz")],
        "interpolate" => &[("start", "start_xyz"), ("end", "end_xyz")],
        _ => &[],
    }
}

fn structure_props(name: &str) -> Vec<(&'static str, JsonValue)> {
    let mut out = Vec::new();
    for (file, xyz) in structure_arguments(name) {
        out.push((
            *file,
            prop(
                "string",
                "An xyz file under the root (relative path), or give its text instead.",
            ),
        ));
        out.push((
            *xyz,
            prop(
                "string",
                &format!("The text of an xyz file (instead of {file})."),
            ),
        ));
    }
    out
}

fn integer_prop(description: &str, default: f64) -> JsonValue {
    object([
        ("type", text("integer")),
        ("default", JsonValue::Number(default)),
        ("description", text(description)),
    ])
}

fn number_prop(description: &str, default: f64) -> JsonValue {
    object([
        ("type", text("number")),
        ("default", JsonValue::Number(default)),
        ("description", text(description)),
    ])
}

fn numbers_prop(description: &str) -> JsonValue {
    object([
        ("type", text("array")),
        ("items", object([("type", text("number"))])),
        ("description", text(description)),
    ])
}

fn charge_props() -> Vec<(&'static str, JsonValue)> {
    vec![
        ("charge", integer_prop("Total charge.", 0.0)),
        (
            "multiplicity",
            integer_prop("Spin multiplicity 2S + 1.", 1.0),
        ),
    ]
}

fn with(
    mut a: Vec<(&'static str, JsonValue)>,
    b: Vec<(&'static str, JsonValue)>,
) -> Vec<(&'static str, JsonValue)> {
    a.extend(b);
    a
}

/// The definitions of the tools of this module.
pub fn tools() -> Vec<JsonValue> {
    let models = base_names().join(", ");
    let systems = CoordinateRegistry::default().names().join(", ");
    vec![
        tool(
            "model_hessian",
            &format!(
                "A model (guess) Hessian of one structure, as -modelhess gives it (Hartree/Bohr^2, \
                 3N x 3N, row by row), with its lowest eigenvalues. Models: {models}; the prefix \
                 legacy_ selects MultiOptPy's version, the modifiers _ts, _sr, _clip apply to any \
                 (gfn1xtb: the analytic Hessian of the built-in GFN1-xTB). At most \
                 {MAX_HESSIAN_ATOMS} atoms."
            ),
            with(
                with(structure_props("model_hessian"), charge_props()),
                vec![(
                    "name",
                    prop(
                        "string",
                        "The model, as for -modelhess (lindh2007d3, gfn1xtb, ...).",
                    ),
                )],
            ),
            &["name"],
            true,
        ),
        tool(
            "vibrations",
            &format!(
                "Normal modes and thermochemistry of one structure (as freqmain): from a given \
                 Cartesian Hessian, or from the Hessian of a calculator (computed at once). \
                 Wavenumbers in cm^-1 (negative: imaginary), translations and rotations \
                 projected out; energies in Hartree. At most {MAX_HESSIAN_ATOMS} atoms."
            ),
            with(
                with(structure_props("vibrations"), charge_props()),
                vec![
                    (
                        "hessian",
                        numbers_prop("Cartesian Hessian (Hartree/Bohr^2), 3N x 3N row by row."),
                    ),
                    (
                        "calculator",
                        prop(
                            "string",
                            "A calculator whose Hessian to take instead (as for -os; see \
                             list_calculators).",
                        ),
                    ),
                    (
                        "energy",
                        number_prop(
                            "Electronic energy (Hartree) for the thermochemistry (that of the \
                             calculator when one is given).",
                            0.0,
                        ),
                    ),
                    ("temperature", number_prop("Temperature (K).", 298.15)),
                    ("pressure", number_prop("Pressure (Pa).", 101_325.0)),
                ],
            ),
            &[],
            true,
        ),
        tool(
            "coordinates",
            &format!(
                "The coordinates of a coordinate system (as -coord) at one structure: their \
                 number, labels where the system has them, and values (Bohr, radians). \
                 Systems: {systems}."
            ),
            with(
                structure_props("coordinates"),
                vec![(
                    "system",
                    prop(
                        "string",
                        "The coordinate system (as -coord), e.g. dic, redundant.",
                    ),
                )],
            ),
            &["system"],
            true,
        ),
        tool(
            "interpolate",
            &format!(
                "Images between two structures of the same atoms, made without a calculation \
                 (the first chain of nebmain): linear, lst, idpp, sidpp, geodesic, fbenm, \
                 cfbenm. Returns the path as a multi-frame xyz text with both ends. At most \
                 {MAX_IMAGES} images."
            ),
            with(
                structure_props("interpolate"),
                vec![
                    ("images", integer_prop("Images between the ends.", 8.0)),
                    (
                        "method",
                        object([
                            ("type", text("string")),
                            (
                                "enum",
                                JsonValue::Array(FirstPath::NAMES.iter().map(text).collect()),
                            ),
                            ("default", text("idpp")),
                            ("description", text("How the images are made.")),
                        ]),
                    ),
                ],
            ),
            &[],
            true,
        ),
        tool(
            "superpose",
            "Superpose a structure on a reference of the same atoms in the same order \
             (Kabsch): the RMSD (Angstrom) and the moved structure as xyz text.",
            with(
                structure_props("superpose"),
                vec![(
                    "mass_weighted",
                    object([
                        ("type", text("boolean")),
                        ("default", JsonValue::Bool(false)),
                        ("description", text("Weight the atoms by their masses.")),
                    ]),
                )],
            ),
            &[],
            true,
        ),
        tool(
            "bond_graph",
            "The bond graph of a structure: bonds (atoms numbered from 1; two atoms are \
             bonded within scale times the sum of their covalent radii), the bonded \
             fragments, and the numbers of bond angles and dihedrals.",
            with(
                structure_props("bond_graph"),
                vec![(
                    "scale",
                    number_prop("Factor on the sum of the covalent radii.", 1.3),
                )],
            ),
            &[],
            true,
        ),
        tool(
            "dispersion",
            "The DFT-D3(BJ) or DFT-D4 dispersion energy (Hartree) and gradient (Hartree/Bohr) \
             of a structure with the PBE parameters.",
            with(
                structure_props("dispersion"),
                vec![
                    (
                        "model",
                        object([
                            ("type", text("string")),
                            ("enum", JsonValue::Array(vec![text("d3"), text("d4")])),
                            ("default", text("d3")),
                            ("description", text("The dispersion model.")),
                        ]),
                    ),
                    ("charge", integer_prop("Total charge (D4 charges).", 0.0)),
                ],
            ),
            &[],
            true,
        ),
        tool(
            "hessian_update",
            "One quasi-Newton update of a Hessian with a step s and the change y of the \
             gradient (as -opt ..._NAME): B + dB, row by row. Methods: bfgs, sr1, psb, bofill, \
             fsb, msp, ts_bfgs, flowchart and the block_ forms.",
            vec![
                ("hessian", numbers_prop("The Hessian, n x n row by row.")),
                ("step", numbers_prop("The step s (n values).")),
                (
                    "gradient_change",
                    numbers_prop("The change of the gradient y along the step (n values)."),
                ),
                (
                    "method",
                    object([
                        ("type", text("string")),
                        ("default", text("bfgs")),
                        ("description", text("The update formula.")),
                    ]),
                ),
            ],
            &["hessian", "step", "gradient_change"],
            true,
        ),
        tool(
            "units",
            "The unit conversion factors of mopt-rs (CODATA 2018), and the \
             conversion of a value between hartree, ev, kcal/mol, kj/mol (energies) or bohr, \
             angstrom (lengths).",
            vec![
                ("value", prop("number", "A value to convert.")),
                ("from", prop("string", "Its unit.")),
                ("to", prop("string", "The unit to convert to.")),
            ],
            &[],
            true,
        ),
        tool(
            "elements",
            "Atomic number, symbol, mass (amu) and covalent radius (Angstrom) of elements given \
             by symbol or atomic number.",
            vec![(
                "elements",
                object([
                    ("type", text("array")),
                    (
                        "items",
                        object([(
                            "type",
                            JsonValue::Array(vec![text("string"), text("integer")]),
                        )]),
                    ),
                    ("minItems", JsonValue::Number(1.0)),
                    (
                        "description",
                        text("Symbols (Fe, fe) or atomic numbers (26)."),
                    ),
                ]),
            )],
            &["elements"],
            true,
        ),
    ]
}

fn integer(args: &JsonValue, key: &str, default: i64) -> Result<i64, String> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(default),
        Some(JsonValue::Number(n)) if n.fract() == 0.0 && n.abs() < 1e9 => Ok(*n as i64),
        Some(_) => Err(format!("{key} is an integer")),
    }
}

fn number(args: &JsonValue, key: &str, default: f64) -> Result<f64, String> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(default),
        Some(JsonValue::Number(n)) if n.is_finite() => Ok(*n),
        Some(_) => Err(format!("{key} is a number")),
    }
}

fn string<'a>(args: &'a JsonValue, key: &str, default: &'a str) -> Result<&'a str, String> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(default),
        Some(JsonValue::String(s)) => Ok(s.as_str()),
        Some(_) => Err(format!("{key} is a string")),
    }
}

fn numbers(args: &JsonValue, key: &str) -> Result<Option<Vec<f64>>, String> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                JsonValue::Number(n) if n.is_finite() => Ok(*n),
                _ => Err(format!("{key}: every entry is a finite number")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(format!("{key} is an array of numbers")),
    }
}

fn square(values: &[f64], n: usize, key: &str) -> Result<Mat<f64>, String> {
    if values.len() != n * n {
        return Err(format!(
            "{key} has {} entries; {n} x {n} = {} were expected",
            values.len(),
            n * n
        ));
    }
    Ok(Mat::from_fn(n, n, |i, j| values[i * n + j]))
}

fn flat(m: &Mat<f64>) -> JsonValue {
    JsonValue::Array(
        (0..m.nrows())
            .flat_map(|i| (0..m.ncols()).map(move |j| (i, j)))
            .map(|(i, j)| JsonValue::Number(m[(i, j)]))
            .collect(),
    )
}

fn column(c: &Col<f64>) -> JsonValue {
    JsonValue::Array(c.iter().map(|v| JsonValue::Number(*v)).collect())
}

fn floats(v: &[f64]) -> JsonValue {
    JsonValue::Array(v.iter().map(|x| JsonValue::Number(*x)).collect())
}

fn one_based(atoms: &[AtomId]) -> JsonValue {
    JsonValue::Array(atoms.iter().map(|a| count(a.index() + 1)).collect())
}

/// The structure `key` of `structures` with the charge and multiplicity of `args`.
fn structure(
    structures: &BTreeMap<String, String>,
    key: &str,
    args: &JsonValue,
    max_atoms: usize,
) -> Result<MolecularSystem, String> {
    let content = structures.get(key).ok_or_else(|| {
        format!("give the structure {key} (a file under the root) or its xyz text")
    })?;
    let state = SpinCharge::checked(
        integer(args, "charge", 0)?,
        integer(args, "multiplicity", 1)?,
    )
    .map_err(|e| e.to_string())?;
    let sys = parse_xyz_str(content, state, None).map_err(|e| format!("{key}: {e}"))?;
    if sys.n_atoms() > max_atoms {
        return Err(format!(
            "{key} has {} atoms; this tool takes at most {max_atoms} (run larger structures as \
             jobs)",
            sys.n_atoms()
        ));
    }
    Ok(sys)
}

fn lowest(h: &Mat<f64>, k: usize) -> Result<Vec<f64>, String> {
    let (w, _) = crate::math::linalg::symmetric_eigen(h).map_err(|e| e.to_string())?;
    Ok(w.iter().take(k).copied().collect())
}

/// Compute the tool `name` with the arguments `args` and the structure texts `structures`
/// (by the file property of [`structure_arguments`]).
pub fn compute(
    name: &str,
    args: &JsonValue,
    structures: &BTreeMap<String, String>,
) -> Result<JsonValue, String> {
    match name {
        "model_hessian" => {
            let sys = structure(structures, "structure", args, MAX_HESSIAN_ATOMS)?;
            let model_name = args
                .get("name")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| {
                    format!("give the name of the model ({})", base_names().join(", "))
                })?;
            let model = resolve(model_name).map_err(|e| e.to_string())?;
            let h = model
                .hessian(&sys, &Col::zeros(sys.dof()))
                .map_err(|e| e.to_string())?;
            let w = lowest(&h, 12)?;
            Ok(object([
                ("name", text(model.name())),
                ("atoms", count(sys.n_atoms())),
                ("hessian_hartree_per_bohr2", flat(&h)),
                ("lowest_eigenvalues", floats(&w)),
                (
                    "negative_eigenvalues",
                    count(w.iter().filter(|x| **x < -1e-8).count()),
                ),
            ]))
        }
        "vibrations" => {
            let sys = structure(structures, "structure", args, MAX_HESSIAN_ATOMS)?;
            let n = sys.dof();
            let values = numbers(args, "hessian")?
                .ok_or_else(|| "give the Hessian (or a calculator to compute it)".to_string())?;
            let h = square(&values, n, "hessian")?;
            let mut settings = FrequencySettings::default();
            settings.thermo.temperature = number(args, "temperature", 298.15)?;
            settings.thermo.pressure = number(args, "pressure", 101_325.0)?;
            if settings.thermo.temperature <= 0.0 || settings.thermo.pressure <= 0.0 {
                return Err("temperature and pressure are positive".to_string());
            }
            settings.thermo.multiplicity = sys.spin_charge.multiplicity;
            let energy = number(args, "energy", 0.0)?;
            let a = vib::analyze(&sys, &h, energy, &settings).map_err(|e| e.to_string())?;
            let t = &a.thermo;
            Ok(object([
                ("atoms", count(sys.n_atoms())),
                ("wavenumbers_cm1", floats(&a.modes.wavenumbers)),
                ("imaginary", count(a.modes.imaginary_modes())),
                ("point_group", text(&a.point_group.label)),
                ("temperature_k", JsonValue::Number(t.temperature)),
                ("pressure_pa", JsonValue::Number(t.pressure)),
                ("electronic_energy", JsonValue::Number(t.electronic_energy)),
                ("zero_point_energy", JsonValue::Number(t.zero_point_energy)),
                ("enthalpy", JsonValue::Number(t.enthalpy.total)),
                ("entropy", JsonValue::Number(t.entropy.total)),
                ("gibbs_energy", JsonValue::Number(t.gibbs.total)),
                ("text", text(vib::thermochemistry_text(&a))),
            ]))
        }
        "coordinates" => {
            let sys = structure(structures, "structure", args, MAX_ATOMS)?;
            let system = string(args, "system", "dic")?;
            // An argument names a file for some systems (hyperspherical:FILE): only numbers
            // are taken here, so that this tool opens no path.
            if let Some((_, argument)) = system.split_once(':') {
                if argument.trim().parse::<f64>().is_err() {
                    return Err(format!(
                        "'{system}': this tool takes a coordinate system with a number as its \
                         argument at most (give files to the commands, -coord)"
                    ));
                }
            }
            let mut coords = CoordinateRegistry::default()
                .create(system)
                .map_err(|e| e.to_string())?;
            coords.initialize(&sys).map_err(|e| e.to_string())?;
            let values = coords.values(&sys).map_err(|e| e.to_string())?;
            Ok(object([
                ("system", text(coords.name())),
                ("dimension", count(coords.dim(&sys))),
                (
                    "labels",
                    coords.labels().map_or(JsonValue::Null, |l| {
                        JsonValue::Array(l.iter().map(text).collect())
                    }),
                ),
                ("values", column(&values)),
            ]))
        }
        "interpolate" => {
            let start = structure(structures, "start", args, MAX_ATOMS)?;
            let end = structure(structures, "end", args, MAX_ATOMS)?;
            if start.elements != end.elements {
                return Err("start and end have different atoms".to_string());
            }
            let images = integer(args, "images", 8)?;
            if !(1..=MAX_IMAGES as i64).contains(&images) {
                return Err(format!("images is between 1 and {MAX_IMAGES}"));
            }
            let method =
                FirstPath::from_name(string(args, "method", "idpp")?).map_err(|e| e.to_string())?;
            let frame = PathFrame {
                elements: start.elements.clone(),
                cell: None,
                fixed_components: Vec::new(),
            };
            let made = interp::images_between(
                method,
                &start.coords_flat_bohr(),
                &end.coords_flat_bohr(),
                images as usize,
                &frame,
                &PathSettings::default(),
            )
            .map_err(|e| e.to_string())?;
            let mut path = format_xyz_system(&start, None);
            for image in &made.images {
                let sys = start
                    .with_flat_coords_bohr(image)
                    .map_err(|e| e.to_string())?;
                path.push_str(&format_xyz_system(&sys, None));
            }
            path.push_str(&format_xyz_system(&end, None));
            Ok(object([
                ("method", text(FirstPath::NAMES[method as usize])),
                ("images", count(made.images.len())),
                ("frames", count(made.images.len() + 2)),
                ("iterations", count(made.iterations)),
                ("converged", JsonValue::Bool(made.converged)),
                ("path_xyz", text(path)),
            ]))
        }
        "superpose" => {
            let mobile = structure(structures, "structure", args, MAX_ATOMS)?;
            let reference = structure(structures, "reference", args, MAX_ATOMS)?;
            if mobile.elements != reference.elements {
                return Err("the structure and the reference have different atoms".to_string());
            }
            let weighted = args
                .get("mass_weighted")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let (moved, rmsd) =
                kabsch_align_bohr(&mobile, &reference, weighted).map_err(|e| e.to_string())?;
            Ok(object([
                (
                    "rmsd_angstrom",
                    JsonValue::Number(rmsd * Units::BOHR_TO_ANGSTROM),
                ),
                ("aligned_xyz", text(format_xyz_system(&moved, None))),
            ]))
        }
        "bond_graph" => {
            let sys = structure(structures, "structure", args, MAX_ATOMS)?;
            let scale = number(args, "scale", 1.3)?;
            if !(0.1..=5.0).contains(&scale) {
                return Err("scale is between 0.1 and 5".to_string());
            }
            let conn = BondConnectivity::from_system(&sys, scale);
            Ok(object([
                (
                    "bonds",
                    JsonValue::Array(conn.bonds.iter().map(|b| one_based(b)).collect()),
                ),
                (
                    "fragments",
                    JsonValue::Array(conn.fragments.iter().map(|f| one_based(f)).collect()),
                ),
                ("angles", count(conn.angles.len())),
                ("dihedrals", count(conn.dihedrals.len())),
            ]))
        }
        "dispersion" => {
            let sys = structure(structures, "structure", args, MAX_ATOMS)?;
            let out = match string(args, "model", "d3")?.to_ascii_lowercase().as_str() {
                "d3" | "d3bj" => D3Bj::PBE.evaluate(&sys, false),
                "d4" => D4::PBE.evaluate(&sys, false),
                other => return Err(format!("the model is d3 or d4, not '{other}'")),
            }
            .map_err(|e| e.to_string())?;
            let g = &out.gradient_bohr;
            Ok(object([
                ("energy_hartree", JsonValue::Number(out.energy.value())),
                (
                    "gradient_hartree_per_bohr",
                    JsonValue::Array(
                        (0..sys.n_atoms())
                            .map(|i| floats(&[g[3 * i], g[3 * i + 1], g[3 * i + 2]]))
                            .collect(),
                    ),
                ),
            ]))
        }
        "hessian_update" => {
            let s = numbers(args, "step")?.ok_or("give the step")?;
            let y = numbers(args, "gradient_change")?.ok_or("give the gradient_change")?;
            let n = s.len();
            if n == 0 || y.len() != n {
                return Err("step and gradient_change have the same, positive length".to_string());
            }
            let b = square(
                &numbers(args, "hessian")?.ok_or("give the hessian")?,
                n,
                "hessian",
            )?;
            let method = string(args, "method", "bfgs")?;
            let mut updater = UpdaterRegistry::default()
                .resolve(method)
                .map_err(|e| e.to_string())?;
            let s = Col::from_fn(n, |i| s[i]);
            let y = Col::from_fn(n, |i| y[i]);
            let db = updater.correction(&b, &s, &y).map_err(|e| e.to_string())?;
            let updated = &b + &db;
            Ok(object([
                ("method", text(updater.name())),
                ("hessian", flat(&updated)),
            ]))
        }
        "units" => {
            let factors = object([
                (
                    "bohr_to_angstrom",
                    JsonValue::Number(Units::BOHR_TO_ANGSTROM),
                ),
                (
                    "angstrom_to_bohr",
                    JsonValue::Number(Units::ANGSTROM_TO_BOHR),
                ),
                ("hartree_to_ev", JsonValue::Number(Units::HARTREE_TO_EV)),
                (
                    "hartree_to_kcalmol",
                    JsonValue::Number(Units::HARTREE_TO_KCAL_MOL),
                ),
                (
                    "hartree_to_kjmol",
                    JsonValue::Number(Units::HARTREE_TO_KJ_MOL),
                ),
            ]);
            let mut map = BTreeMap::new();
            map.insert("factors".to_string(), factors);
            if let Some(v) = args.get("value") {
                let value = v
                    .as_f64()
                    .filter(|x| x.is_finite())
                    .ok_or("value is a number")?;
                let from = string(args, "from", "")?;
                let to = string(args, "to", "")?;
                let (a, kind_a) = unit(from)?;
                let (b, kind_b) = unit(to)?;
                if kind_a != kind_b {
                    return Err(format!(
                        "{from} and {to} are not units of the same quantity"
                    ));
                }
                map.insert("value".to_string(), JsonValue::Number(value * a / b));
                map.insert("unit".to_string(), text(to));
            }
            Ok(JsonValue::Object(map))
        }
        "elements" => {
            let given = match args.get("elements") {
                Some(JsonValue::Array(items)) if !items.is_empty() && items.len() <= 200 => items,
                _ => return Err("give 1 to 200 elements".to_string()),
            };
            let mut out = Vec::new();
            for g in given {
                let e = match g {
                    JsonValue::String(s) => Element::parse_token(s),
                    JsonValue::Number(z) if z.fract() == 0.0 && (1.0..=118.0).contains(z) => {
                        Element::from_atomic_number(*z as u8)
                    }
                    _ => return Err("an element is a symbol or an atomic number".to_string()),
                }
                .map_err(|e| e.to_string())?;
                out.push(object([
                    ("symbol", text(e.symbol())),
                    ("atomic_number", count(usize::from(e.atomic_number()))),
                    ("mass_amu", JsonValue::Number(e.mass_amu())),
                    (
                        "covalent_radius_angstrom",
                        JsonValue::Number(covalent_radius_bohr(e) * Units::BOHR_TO_ANGSTROM),
                    ),
                ]));
            }
            Ok(object([("elements", JsonValue::Array(out))]))
        }
        other => Err(format!("no component tool '{other}'")),
    }
}

/// The size of `name` in its base unit (Hartree or Bohr) and its quantity.
fn unit(name: &str) -> Result<(f64, &'static str), String> {
    match name.to_ascii_lowercase().replace(' ', "").as_str() {
        "hartree" | "eh" | "au" => Ok((1.0, "energy")),
        "ev" => Ok((Units::EV_TO_HARTREE, "energy")),
        "kcal/mol" | "kcalmol" | "kcal" => Ok((Units::KCAL_MOL_TO_HARTREE, "energy")),
        "kj/mol" | "kjmol" | "kj" => Ok((Units::KJ_MOL_TO_HARTREE, "energy")),
        "bohr" => Ok((1.0, "length")),
        "angstrom" | "a" => Ok((Units::ANGSTROM_TO_BOHR, "length")),
        other => Err(format!(
            "unknown unit '{other}' (hartree, ev, kcal/mol, kj/mol, bohr, angstrom)"
        )),
    }
}

/// The definitions of [`tools`] and the structure arguments of each, as JSON text (for
/// `mopt_rs._core.mcp_component_tools`).
pub fn tools_json() -> String {
    let structures: BTreeMap<String, JsonValue> = TOOLS
        .iter()
        .map(|n| {
            (
                n.to_string(),
                JsonValue::Array(
                    structure_arguments(n)
                        .iter()
                        .map(|(file, xyz)| JsonValue::Array(vec![text(file), text(xyz)]))
                        .collect(),
                ),
            )
        })
        .collect();
    object([
        ("tools", JsonValue::Array(tools())),
        ("structures", JsonValue::Object(structures)),
    ])
    .to_string_compact()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469\n";

    fn texts(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn vibrations_count_imaginary_modes_by_the_one_criterion() {
        const H2: &str = "2\nh2\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n";
        let sys = parse_xyz_str(H2, SpinCharge::default(), None).unwrap();
        let mu = 0.5 * sys.masses_amu()[0];
        // The Hessian of a bond along x whose wavenumber is -nu (cm^-1).
        let hessian = |nu: f64| {
            let k = -(nu / crate::domain::Units::EIGVAL_MW_TO_CM1).powi(2) * mu;
            let mut h = vec![0.0; 36];
            h[0] = k;
            h[3 * 6 + 3] = k;
            h[3] = -k;
            h[3 * 6] = -k;
            h
        };
        let imaginary = |nu: f64| {
            let args = object([("hessian", floats(&hessian(nu)))]);
            let v = compute("vibrations", &args, &texts(&[("structure", H2)])).unwrap();
            v.get("imaginary").and_then(JsonValue::as_f64).unwrap()
        };
        // -5 cm^-1: above the former cutoff of -10 cm^-1, an imaginary mode by the one
        // criterion (about -0.5 cm^-1); -0.1 cm^-1 is within it.
        assert_eq!(imaginary(5.0), 1.0);
        assert_eq!(imaginary(0.1), 0.0);
    }

    #[test]
    fn every_tool_has_a_definition_and_answers() {
        let names: Vec<String> = tools()
            .iter()
            .map(|t| {
                t.get("name")
                    .and_then(JsonValue::as_str)
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(
            names,
            TOOLS.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        let water = texts(&[("structure", WATER)]);
        let h = compute(
            "model_hessian",
            &JsonValue::parse(r#"{"name":"lindh2007d3"}"#).unwrap(),
            &water,
        )
        .unwrap();
        let flat = h.get("hessian_hartree_per_bohr2").unwrap().clone();
        assert_eq!(flat.as_array().unwrap().len(), 81);
        let args = object([("hessian", flat)]);
        let v = compute("vibrations", &args, &water).unwrap();
        assert_eq!(
            v.get("wavenumbers_cm1").unwrap().as_array().unwrap().len(),
            9
        );
        let c = compute(
            "coordinates",
            &JsonValue::parse(r#"{"system":"redundant"}"#).unwrap(),
            &water,
        )
        .unwrap();
        assert!(c.get("dimension").and_then(JsonValue::as_f64).unwrap() >= 3.0);
        let bent = WATER.replace("0.757 -0.469\nH 0.0 -0.757", "0.70 -0.52\nH 0.0 -0.80");
        let pair = texts(&[("start", WATER), ("end", bent.as_str())]);
        let p = compute(
            "interpolate",
            &JsonValue::parse(r#"{"images":3,"method":"linear"}"#).unwrap(),
            &pair,
        )
        .unwrap();
        assert_eq!(p.get("frames").and_then(JsonValue::as_f64), Some(5.0));
        let both = texts(&[("structure", bent.as_str()), ("reference", WATER)]);
        let s = compute("superpose", &JsonValue::empty_object(), &both).unwrap();
        assert!(s.get("rmsd_angstrom").and_then(JsonValue::as_f64).unwrap() > 0.0);
        let g = compute("bond_graph", &JsonValue::empty_object(), &water).unwrap();
        assert_eq!(g.get("bonds").unwrap().as_array().unwrap().len(), 2);
        let d = compute("dispersion", &JsonValue::empty_object(), &water).unwrap();
        assert!(d.get("energy_hartree").and_then(JsonValue::as_f64).unwrap() < 0.0);
        let u = compute(
            "hessian_update",
            &JsonValue::parse(r#"{"hessian":[1,0,0,1],"step":[0.1,0],"gradient_change":[0.2,0]}"#)
                .unwrap(),
            &BTreeMap::new(),
        )
        .unwrap();
        let updated = u.get("hessian").unwrap().as_array().unwrap();
        // BFGS reproduces the secant condition B s = y.
        assert!((updated[0].as_f64().unwrap() * 0.1 - 0.2).abs() < 1e-12);
        let c = compute(
            "units",
            &JsonValue::parse(r#"{"value":1,"from":"hartree","to":"kcal/mol"}"#).unwrap(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(
            (c.get("value").and_then(JsonValue::as_f64).unwrap() - 627.509_474_063).abs() < 1e-9
        );
        let e = compute(
            "elements",
            &JsonValue::parse(r#"{"elements":["Fe",8]}"#).unwrap(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(e.get("elements").unwrap().as_array().unwrap().len(), 2);
        assert!(compute(
            "units",
            &JsonValue::parse(r#"{"value":1,"from":"ev","to":"bohr"}"#).unwrap(),
            &BTreeMap::new()
        )
        .is_err());
        assert!(compute(
            "model_hessian",
            &JsonValue::empty_object(),
            &BTreeMap::new()
        )
        .is_err());
        assert!(JsonValue::parse(&tools_json()).is_ok());
    }
}
