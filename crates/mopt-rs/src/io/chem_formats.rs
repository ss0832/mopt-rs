//! Parsers and writers for Gaussian `.gjf`/`.com` (including ONIOM layer & link-atom tags),
//! MDL `.mol` (V2000), Tripos `.mol2`, and GAMESS / ORCA / Q-Chem `.inp` files.
//!
//! Fixes MultiOptPy's `_parse_qchem` iterator bug where `next(lines)` on a `list` raised `TypeError`.
//!
//! MultiOptPy: `fileio.py` (`_parse_gamess`, `_parse_orca`, `_parse_qchem`, `inp2list`,
//!  `mol2list`, `mol22list`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::Vector3;
use crate::domain::{positions, Angstrom, AtomId, Bohr};
use crate::domain::{Hartree, Quantity};
use std::fs;
use std::path::Path;

use crate::domain::{Element, MolecularSystem, PeriodicCell, SpinCharge, Units};
use crate::error::{MoptError, MoptResult};
use crate::io::xyz::{read_xyz_file, write_xyz_file};

/// Optional ONIOM layer metadata parsed from a Gaussian `.gjf` atom specification line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GjfOniomAtomMeta {
    /// Layer label (`"H"`, `"M"`, `"L"`, or arbitrary custom layer string).
    pub layer: String,
    /// Optional link-atom element symbol (e.g. `H`) when cutting a covalent bond.
    pub link_element: Option<Element>,
    /// Optional 0-based index of the host atom inside the higher layer.
    pub link_host_atom: Option<usize>,
    /// Optional freeze flag (`0` = free, `-1` = frozen).
    pub freeze_flag: i32,
}

/// Complete result of parsing a Gaussian `.gjf` / `.com` input file.
#[derive(Debug, Clone)]
pub struct ParsedGjf {
    /// Route section lines (starting with `#`).
    pub route_lines: Vec<String>,
    /// Parsed structure.
    pub system: MolecularSystem,
    /// Charge/multiplicity pairs in file order (one per ONIOM layer combination).
    pub layer_charges: Vec<SpinCharge>,
    /// Per-atom ONIOM layer metadata, if present.
    pub oniom_atoms: Option<Vec<GjfOniomAtomMeta>>,
}

/// Parse a Gaussian `.gjf` or `.com` input string, supporting standard geometries, freeze flags,
/// `El(Fragment=k)` / `El-AtomType-Charge` notation, and 2/3-layer ONIOM atom specifications.
pub fn parse_gjf_str(content: &str, override_cell: Option<&PeriodicCell>) -> MoptResult<ParsedGjf> {
    let raw_lines: Vec<&str> = content.lines().map(|l| l.trim_end_matches('\r')).collect();
    let mut idx = 0;
    let mut route_lines = Vec::new();

    // 1. Collect `%` link0 and `#` route lines until the blank line after the `#` section
    let mut seen_route = false;
    while idx < raw_lines.len() {
        let t = raw_lines[idx].trim();
        if t.starts_with('%') || t.starts_with('#') {
            if t.starts_with('#') {
                seen_route = true;
            }
            route_lines.push(t.to_string());
            idx += 1;
            continue;
        }
        if seen_route && t.is_empty() {
            idx += 1;
            break;
        }
        if seen_route {
            route_lines.push(t.to_string());
            idx += 1;
        } else if t.is_empty() {
            idx += 1;
        } else {
            break;
        }
    }

    // 2. Title section up to next blank line (if route section existed)
    let mut title_parts = Vec::new();
    if seen_route {
        while idx < raw_lines.len() {
            let t = raw_lines[idx].trim();
            idx += 1;
            if t.is_empty() {
                break;
            }
            title_parts.push(t);
        }
    }

    // Skip any extra blank lines before charge/multiplicity
    while idx < raw_lines.len() && raw_lines[idx].trim().is_empty() {
        idx += 1;
    }
    if idx >= raw_lines.len() {
        return Err(MoptError::parse(
            "GJF file missing charge/multiplicity line",
        ));
    }

    // 3. Parse `charge mult [charge2 mult2 ...]`
    let cm_tokens: Vec<i32> = raw_lines[idx]
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<i32>().map_err(|e| {
                MoptError::parse(format!("Invalid GJF charge/multiplicity token '{s}': {e}"))
            })
        })
        .collect::<MoptResult<Vec<i32>>>()?;
    if cm_tokens.len() < 2 || !cm_tokens.len().is_multiple_of(2) {
        return Err(MoptError::parse(
            "GJF charge/multiplicity line must contain pairs of integers",
        ));
    }
    let mut layer_charges = Vec::with_capacity(cm_tokens.len() / 2);
    for pair in cm_tokens.chunks_exact(2) {
        layer_charges.push(SpinCharge::new(pair[0], pair[1].unsigned_abs().max(1)));
    }
    idx += 1;

    // 4. Parse atom lines until blank line or EOF
    let mut elements = Vec::new();
    let mut pos_ang = Vec::new();
    let mut oniom_meta = Vec::new();
    let mut any_oniom = false;
    let mut cell = override_cell.cloned();
    let mut tv_vectors_ang: Vec<[f64; 3]> = Vec::new();

    while idx < raw_lines.len() {
        let t = raw_lines[idx].trim();
        idx += 1;
        if t.is_empty() {
            break;
        }
        let fields: Vec<&str> = t.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(MoptError::parse(format!("Malformed GJF atom line: '{t}'")));
        }

        // Extract base element symbol before `-` or `(` (e.g. `C-CA--0.12`, `O(Fragment=1)`, `TV`)
        let raw_head = fields[0];
        let head_clean = raw_head.split(['-', '(']).next().unwrap_or(raw_head).trim();

        // Check whether `fields[1]` is an integer freeze flag (`0`, `-1`, `1`) and coordinates start at `fields[2]`
        let (freeze_flag, coord_start) = if fields.len() >= 5 && fields[1].parse::<i32>().is_ok() {
            (fields[1].parse::<i32>().unwrap_or(0), 2usize)
        } else {
            (0, 1usize)
        };

        if fields.len() < coord_start + 3 {
            return Err(MoptError::parse(format!(
                "Insufficient coordinates in GJF atom line: '{t}'"
            )));
        }

        let x = fields[coord_start]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid GJF X float: {e}")))?;
        let y = fields[coord_start + 1]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid GJF Y float: {e}")))?;
        let z = fields[coord_start + 2]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid GJF Z float: {e}")))?;

        // Gaussian PBC translation vector `TV x y z`
        if head_clean.eq_ignore_ascii_case("TV") {
            tv_vectors_ang.push([x, y, z]);
            continue;
        }

        let el = Element::parse_token(head_clean)?;
        elements.push(el);
        pos_ang.push([x, y, z]);

        let trailing = &fields[coord_start + 3..];
        if !trailing.is_empty() {
            any_oniom = true;
            let layer = trailing[0].to_uppercase();
            let link_element = trailing.get(1).and_then(|tok| {
                let base = tok.split('-').next().unwrap_or(tok);
                Element::parse_token(base).ok()
            });
            let link_host_atom = trailing
                .get(2)
                .and_then(|tok| tok.parse::<usize>().ok())
                .and_then(|one_based| one_based.checked_sub(1));
            oniom_meta.push(GjfOniomAtomMeta {
                layer,
                link_element,
                link_host_atom,
                freeze_flag,
            });
        } else {
            oniom_meta.push(GjfOniomAtomMeta {
                layer: "H".to_string(),
                link_element: None,
                link_host_atom: None,
                freeze_flag,
            });
        }
    }

    if cell.is_none() && !tv_vectors_ang.is_empty() {
        cell = Some(build_cell_from_tv(&tv_vectors_ang)?);
    }

    let mut system = MolecularSystem::new(
        elements,
        &positions::<Angstrom>(&pos_ang),
        layer_charges[0],
        cell,
    )?;
    system.comment = title_parts.join(" ");

    Ok(ParsedGjf {
        route_lines,
        system,
        layer_charges,
        oniom_atoms: if any_oniom { Some(oniom_meta) } else { None },
    })
}

fn build_cell_from_tv(tv: &[[f64; 3]]) -> MoptResult<PeriodicCell> {
    match tv.len() {
        1 => PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new(tv[0]),
            Vector3::new([0.0, 50.0, 0.0]),
            Vector3::new([0.0, 0.0, 50.0]),
            [true, false, false],
        ),
        2 => PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new(tv[0]),
            Vector3::new(tv[1]),
            Vector3::new([0.0, 0.0, 50.0]),
            [true, true, false],
        ),
        _ => PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new(tv[0]),
            Vector3::new(tv[1]),
            Vector3::new(tv[2]),
            [true, true, true],
        ),
    }
}

/// Format a `MolecularSystem` as a standard Gaussian `.gjf` string.
pub fn write_gjf_str(system: &MolecularSystem, route: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str(route.unwrap_or("#p opt freq b3lyp/6-31g(d)"));
    out.push_str("\n\n");
    if system.comment.is_empty() {
        out.push_str(&format!("Generated by mopt-rs v{}\n\n", crate::VERSION));
    } else {
        out.push_str(&system.comment);
        out.push_str("\n\n");
    }
    out.push_str(&format!(
        "{} {}\n",
        system.spin_charge.charge, system.spin_charge.multiplicity
    ));
    let b2a = Units::BOHR_TO_ANGSTROM;
    for i in 0..system.n_atoms() {
        let p = system.position(AtomId::new(i)).value();
        out.push_str(&format!(
            "{:<3} {:>18.12} {:>18.12} {:>18.12}\n",
            system.elements[i].symbol(),
            p[0] * b2a,
            p[1] * b2a,
            p[2] * b2a
        ));
    }
    if let Some(cell) = &system.cell {
        for ax in 0..3 {
            if cell.periodic_axes[ax] {
                let v = cell.lattice_vector(ax).value();
                out.push_str(&format!(
                    "TV  {:>18.12} {:>18.12} {:>18.12}\n",
                    v[0] * b2a,
                    v[1] * b2a,
                    v[2] * b2a
                ));
            }
        }
    }
    out.push('\n');
    out
}

/// Parse an MDL `.mol` (V2000) string into a `MolecularSystem`.
pub fn parse_mol_str(
    content: &str,
    spin_charge: SpinCharge,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<MolecularSystem> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 4 {
        return Err(MoptError::parse(
            "MDL MOL file has fewer than 4 header lines",
        ));
    }
    let title = lines[0].trim().to_string();
    let counts_line = lines[3];
    // V2000 counts line has fixed-width 3-char atom count or whitespace-separated fields
    let n_atoms = counts_line
        .get(0..3)
        .and_then(|s| s.trim().parse::<usize>().ok())
        .or_else(|| {
            counts_line
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .ok_or_else(|| MoptError::parse("Failed to parse atom count in MDL MOL counts line"))?;

    if lines.len() < 4 + n_atoms {
        return Err(MoptError::parse("Truncated atom block in MDL MOL file"));
    }

    let mut elements = Vec::with_capacity(n_atoms);
    let mut pos_ang = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        let fields: Vec<&str> = lines[4 + i].split_whitespace().collect();
        if fields.len() < 4 {
            return Err(MoptError::parse(format!(
                "Malformed MOL atom line {}: '{}'",
                5 + i,
                lines[4 + i]
            )));
        }
        let x = fields[0]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid MOL X coordinate: {e}")))?;
        let y = fields[1]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid MOL Y coordinate: {e}")))?;
        let z = fields[2]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid MOL Z coordinate: {e}")))?;
        let el = Element::parse_token(fields[3])?;
        elements.push(el);
        pos_ang.push([x, y, z]);
    }

    let mut sys = MolecularSystem::new(
        elements,
        &positions::<Angstrom>(&pos_ang),
        spin_charge,
        override_cell.cloned(),
    )?;
    sys.comment = title;
    Ok(sys)
}

/// Parse a Tripos `.mol2` string (`@<TRIPOS>ATOM` section) into a `MolecularSystem`.
pub fn parse_mol2_str(
    content: &str,
    spin_charge: SpinCharge,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<MolecularSystem> {
    let mut in_atom_block = false;
    let mut elements = Vec::new();
    let mut pos_ang = Vec::new();

    for line in content.lines() {
        let t = line.trim();
        if t.starts_with("@<TRIPOS>") {
            in_atom_block = t.eq_ignore_ascii_case("@<TRIPOS>ATOM");
            continue;
        }
        if !in_atom_block || t.is_empty() || t.starts_with('#') {
            continue;
        }
        // Format: atom_id atom_name x y z atom_type [subst_id subst_name charge]
        let fields: Vec<&str> = t.split_whitespace().collect();
        if fields.len() < 6 {
            return Err(MoptError::parse(format!("Malformed MOL2 atom line: '{t}'")));
        }
        let x = fields[2]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid MOL2 X: {e}")))?;
        let y = fields[3]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid MOL2 Y: {e}")))?;
        let z = fields[4]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid MOL2 Z: {e}")))?;
        let sybyl_base = fields[5].split('.').next().unwrap_or(fields[5]);
        let el = Element::parse_token(sybyl_base).or_else(|_| Element::parse_token(fields[1]))?;
        elements.push(el);
        pos_ang.push([x, y, z]);
    }

    if elements.is_empty() {
        return Err(MoptError::parse(
            "No atoms found in @<TRIPOS>ATOM block of MOL2 input",
        ));
    }
    MolecularSystem::new(
        elements,
        &positions::<Angstrom>(&pos_ang),
        spin_charge,
        override_cell.cloned(),
    )
}

/// Parse a GAMESS `$DATA ... $END` input string.
pub fn parse_gamess_inp(
    content: &str,
    default_spin_charge: SpinCharge,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<MolecularSystem> {
    let lines: Vec<&str> = content.lines().collect();
    let mut idx = 0;
    while idx < lines.len()
        && !lines[idx]
            .trim_start()
            .to_ascii_uppercase()
            .starts_with("$DATA")
    {
        idx += 1;
    }
    if idx >= lines.len() {
        return Err(MoptError::parse("GAMESS input missing $DATA group"));
    }
    idx += 1; // title line
    let title = lines
        .get(idx)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    idx += 1;
    // Symmetry group line (e.g. `C1` or `CN 3` followed by blank/axis lines)
    if let Some(sym_line) = lines.get(idx) {
        let sym = sym_line.split_whitespace().next().unwrap_or("C1");
        idx += 1;
        if !sym.eq_ignore_ascii_case("C1") {
            // Non-C1 groups in GAMESS have symmetry frame lines until a blank line
            while idx < lines.len() && !lines[idx].trim().is_empty() {
                idx += 1;
            }
            if idx < lines.len() {
                idx += 1;
            }
        }
    }

    let mut elements = Vec::new();
    let mut pos_ang = Vec::new();
    while idx < lines.len() {
        let t = lines[idx].trim();
        idx += 1;
        if t.to_ascii_uppercase().starts_with("$END") {
            break;
        }
        if t.is_empty() {
            continue;
        }
        // Format: `ATOM_LABEL  ZNUC.0  X  Y  Z`
        let fields: Vec<&str> = t.split_whitespace().collect();
        if fields.len() < 5 {
            return Err(MoptError::parse(format!(
                "Malformed GAMESS $DATA atom line: '{t}'"
            )));
        }
        let znuc = fields[1]
            .parse::<f64>()
            .ok()
            .and_then(|z| Element::from_atomic_number(z.round() as u8).ok());
        let el = match znuc {
            Some(e) => e,
            None => Element::parse_token(fields[0])?,
        };
        let x = fields[2]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(e.to_string()))?;
        let y = fields[3]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(e.to_string()))?;
        let z = fields[4]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(e.to_string()))?;
        elements.push(el);
        pos_ang.push([x, y, z]);
    }
    let mut sys = MolecularSystem::new(
        elements,
        &positions::<Angstrom>(&pos_ang),
        default_spin_charge,
        override_cell.cloned(),
    )?;
    sys.comment = title;
    Ok(sys)
}

/// Parse an ORCA `* xyz charge mult ... *` input string.
pub fn parse_orca_inp(
    content: &str,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<MolecularSystem> {
    let lines: Vec<&str> = content.lines().collect();
    let mut idx = 0;
    let mut spin_charge = SpinCharge::default();
    let mut in_xyz = false;
    let mut is_bohr = false;
    let mut elements = Vec::new();
    let mut coords = Vec::new();

    while idx < lines.len() {
        let t = lines[idx].trim();
        idx += 1;
        if !in_xyz {
            let lower = t.to_ascii_lowercase();
            if lower.starts_with('*') && (lower.contains("xyz") || lower.contains("bohr")) {
                let cleaned = t.trim_start_matches('*').trim();
                let parts: Vec<&str> = cleaned.split_whitespace().collect();
                if parts.len() >= 3
                    && (parts[0].eq_ignore_ascii_case("xyz")
                        || parts[0].eq_ignore_ascii_case("bohr"))
                {
                    is_bohr = parts[0].eq_ignore_ascii_case("bohr");
                    let q = parts[1].parse::<i32>().unwrap_or(0);
                    let m = parts[2].parse::<u32>().unwrap_or(1);
                    spin_charge = SpinCharge::new(q, m);
                    in_xyz = true;
                }
            }
            continue;
        }
        if t.starts_with('*') {
            break;
        }
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = t.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(MoptError::parse(format!(
                "Malformed ORCA coordinate line: '{t}'"
            )));
        }
        let el = Element::parse_token(fields[0])?;
        let x = fields[1]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(e.to_string()))?;
        let y = fields[2]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(e.to_string()))?;
        let z = fields[3]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(e.to_string()))?;
        elements.push(el);
        coords.push([x, y, z]);
    }

    if elements.is_empty() {
        return Err(MoptError::parse(
            "No `* xyz charge mult ... *` block found in ORCA input",
        ));
    }
    if is_bohr {
        MolecularSystem::new(
            elements,
            &positions::<Bohr>(&coords),
            spin_charge,
            override_cell.cloned(),
        )
    } else {
        MolecularSystem::new(
            elements,
            &positions::<Angstrom>(&coords),
            spin_charge,
            override_cell.cloned(),
        )
    }
}

/// Parse a Q-Chem `$molecule ... $end` input string (bug-free indexed scan).
pub fn parse_qchem_inp(
    content: &str,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<MolecularSystem> {
    let lines: Vec<&str> = content.lines().collect();
    let mut idx = 0;
    while idx < lines.len() && !lines[idx].trim().eq_ignore_ascii_case("$molecule") {
        idx += 1;
    }
    if idx >= lines.len() {
        return Err(MoptError::parse("Q-Chem input missing $molecule block"));
    }
    idx += 1;
    while idx < lines.len() && (lines[idx].trim().is_empty() || lines[idx].trim().starts_with('!'))
    {
        idx += 1;
    }
    if idx >= lines.len() {
        return Err(MoptError::parse("Truncated Q-Chem $molecule block"));
    }
    let cm: Vec<&str> = lines[idx].split_whitespace().collect();
    if cm.len() < 2 {
        return Err(MoptError::parse("Invalid Q-Chem charge/multiplicity line"));
    }
    let spin_charge = SpinCharge::new(
        cm[0]
            .parse::<i32>()
            .map_err(|e| MoptError::parse(e.to_string()))?,
        cm[1]
            .parse::<u32>()
            .map_err(|e| MoptError::parse(e.to_string()))?,
    );
    idx += 1;

    let mut elements = Vec::new();
    let mut pos_ang = Vec::new();
    while idx < lines.len() {
        let t = lines[idx].trim();
        idx += 1;
        if t.eq_ignore_ascii_case("$end") {
            break;
        }
        if t.is_empty() || t.starts_with('!') {
            continue;
        }
        let fields: Vec<&str> = t.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(MoptError::parse(format!(
                "Malformed Q-Chem atom line: '{t}'"
            )));
        }
        let el = Element::parse_token(fields[0])?;
        let x = fields[1]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(e.to_string()))?;
        let y = fields[2]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(e.to_string()))?;
        let z = fields[3]
            .parse::<f64>()
            .map_err(|e| MoptError::parse(e.to_string()))?;
        elements.push(el);
        pos_ang.push([x, y, z]);
    }
    MolecularSystem::new(
        elements,
        &positions::<Angstrom>(&pos_ang),
        spin_charge,
        override_cell.cloned(),
    )
}

/// Auto-detect and read a `MolecularSystem` from `.xyz`, `.extxyz`, `.gjf`, `.com`, `.mol`, `.mol2`, or `.inp`.
pub fn read_molecular_file(
    path: impl AsRef<Path>,
    default_spin_charge: SpinCharge,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<MolecularSystem> {
    let p = path.as_ref();
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("xyz")
        .to_ascii_lowercase();

    match ext.as_str() {
        "xyz" | "extxyz" | "trj" => read_xyz_file(p, default_spin_charge, override_cell),
        "gjf" | "com" => {
            let content = fs::read_to_string(p)?;
            Ok(parse_gjf_str(&content, override_cell)?.system)
        }
        "mol" | "sdf" => {
            let content = fs::read_to_string(p)?;
            parse_mol_str(&content, default_spin_charge, override_cell)
        }
        "mol2" => {
            let content = fs::read_to_string(p)?;
            parse_mol2_str(&content, default_spin_charge, override_cell)
        }
        "inp" => {
            let content = fs::read_to_string(p)?;
            let upper = content.to_ascii_uppercase();
            if upper.contains("$DATA") {
                parse_gamess_inp(&content, default_spin_charge, override_cell)
            } else if upper.contains("$MOLECULE") {
                parse_qchem_inp(&content, override_cell)
            } else {
                parse_orca_inp(&content, override_cell)
            }
        }
        _ => read_xyz_file(p, default_spin_charge, override_cell),
    }
}

/// Write a `MolecularSystem` to disk based on its file extension (`.xyz`, `.gjf`, `.com`).
pub fn write_molecular_file(
    path: impl AsRef<Path>,
    system: &MolecularSystem,
    energy: Option<Quantity<Hartree>>,
) -> MoptResult<()> {
    let p = path.as_ref();
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("xyz")
        .to_ascii_lowercase();
    match ext.as_str() {
        "gjf" | "com" => {
            let txt = write_gjf_str(system, None);
            fs::write(p, txt)?;
            Ok(())
        }
        _ => write_xyz_file(p, system, energy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gjf_with_oniom_and_qchem_inp() {
        let gjf = "\
%chk=test.chk
#p oniom(b3lyp/6-31g(d):uff) opt

ONIOM test

0 1 0 1 0 1
O  0   0.000000  0.000000  0.000000 H
C  0   1.420000  0.000000  0.000000 L H 1

";
        let parsed = parse_gjf_str(gjf, None).unwrap();
        assert_eq!(parsed.system.n_atoms(), 2);
        let oniom = parsed.oniom_atoms.unwrap();
        assert_eq!(oniom[0].layer, "H");
        assert_eq!(oniom[1].layer, "L");
        assert_eq!(oniom[1].link_element, Some(Element::H));
        assert_eq!(oniom[1].link_host_atom, Some(0));

        let qchem = "\
$molecule
0 1
O  0.0  0.0  0.117
H  0.0  0.757 -0.469
H  0.0 -0.757 -0.469
$end
";
        let qsys = parse_qchem_inp(qchem, None).unwrap();
        assert_eq!(qsys.n_atoms(), 3);
    }
}
