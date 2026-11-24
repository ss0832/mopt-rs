//! Interruption and restart for every command: a journal of evaluations.
//!
//! Every evaluation a run makes is appended to `evaluations.journal` in the run directory as a
//! checksummed record (structure, request, result) and flushed before the run goes on. A run
//! restarted from the same settings rebuilds the same deterministic state machines, and
//! [`JournaledCalculator`] answers their evaluations from the journal instead of the program
//! as long as the requested structure matches a recorded one; from the first mismatch on (or
//! when the journal is used up) the program runs again and the journal grows. No algorithm
//! needs code of its own to be restartable, and a run killed at any moment loses at most the
//! evaluation in flight (a truncated last record is ignored).
//!
//! Records are matched by calculator label, electronic state, request, and positions (within
//! [`MATCH_TOLERANCE_BOHR`]), not by order, so evaluations made in parallel (path images) replay
//! correctly. Of the records that match, one of the identical structure comes first, and
//! among those the earliest, so that a run whose successive structures lie within the
//! tolerance of each other (an optimization near its convergence) replays exactly.
//!
//! Format: the header `MOPTJRN1`, then records `u32 length | payload | u32 CRC-32(payload)`,
//! little endian.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy cannot resume a run).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use faer::{Col, Mat};

use crate::core_infra::viz::checksum::crc32;
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Orbitals, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::pes::replay::{ReplaySource, Replayed};
use crate::pes::Calculator;

/// File name of the journal in a run directory.
pub const JOURNAL_FILE: &str = "evaluations.journal";
/// Largest coordinate difference (Bohr) for a recorded evaluation to answer a request.
pub const MATCH_TOLERANCE_BOHR: f64 = 1e-9;

const MAGIC: &[u8; 8] = b"MOPTJRN1";

/// One recorded evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalRecord {
    /// Calculator label (for runs with several calculators).
    pub calculator: String,
    /// Total charge.
    pub charge: i64,
    /// Multiplicity.
    pub multiplicity: i64,
    /// Electronic state root.
    pub excited_state: u64,
    /// Whether the stored gradient is available for replay.
    pub has_gradient: bool,
    /// Whether the Hessian was computed.
    pub has_hessian: bool,
    /// Positions (Bohr, flat).
    pub positions_bohr: Vec<f64>,
    /// Energy (Hartree).
    pub energy: f64,
    /// Gradient (Hartree/Bohr).
    pub gradient: Vec<f64>,
    /// Hessian (row-major), when computed.
    pub hessian: Option<Vec<f64>>,
    /// Lattice vectors (Bohr, row by row) of a structure in a cell.
    pub cell_bohr: Option<[f64; 9]>,
    /// Stress (Hartree/Bohr^3, row by row), when the calculator gave one.
    pub stress: Option<[f64; 9]>,
    /// Where states are followed between structures (`pes::followed`): the index (0: the
    /// lowest) of the state by energy that the evaluation was for.
    pub state_root: Option<u64>,
    /// Where the calculator relaxes a part of the structure (ONIOM microiterations): its
    /// relaxed coordinates (flat, Bohr).
    pub environment: Option<Vec<f64>>,
    /// The description of the calculator (method, basis, settings), so that the journals of a
    /// workflow, whose commands have calculators of other levels under the same label, answer
    /// a request only with an evaluation of the same calculator ([`ReplayPool`]).
    pub identity: Option<String>,
    /// The second derivatives along the strain of the cell, when the calculator gave them
    /// ([`crate::domain::StrainHessian`]): the 81 of the strain row by row, then the mixed
    /// block (`3N x 9`) row by row.
    pub strain_hessian: Option<Vec<f64>>,
}

/// The lattice vectors of `sys` as nine numbers.
fn cell_of(sys: &MolecularSystem) -> Option<[f64; 9]> {
    sys.cell.as_ref().map(|c| {
        let mut flat = [0.0; 9];
        for k in 0..3 {
            flat[3 * k..3 * k + 3].copy_from_slice(&c.lattice_vector(k).value());
        }
        flat
    })
}

impl EvalRecord {
    fn new(
        label: &str,
        sys: &MolecularSystem,
        req: &EvalRequest,
        out: &EvalOutput,
        state_root: Option<usize>,
        environment: Option<Vec<f64>>,
        identity: Option<String>,
    ) -> Self {
        let hessian = out.hessian_bohr.as_ref().map(|h| {
            let n = h.nrows();
            (0..n * n).map(|k| h[(k / n, k % n)]).collect()
        });
        Self {
            calculator: label.to_string(),
            charge: i64::from(sys.spin_charge.charge),
            multiplicity: i64::from(sys.spin_charge.multiplicity),
            excited_state: req.excited_state as u64,
            has_gradient: req.need_gradient || req.need_hessian,
            has_hessian: hessian.is_some(),
            positions_bohr: sys.coords_flat_bohr().iter().copied().collect(),
            energy: out.energy.value(),
            gradient: out.gradient_bohr.iter().copied().collect(),
            hessian,
            cell_bohr: cell_of(sys),
            stress: out.stress.map(|s| std::array::from_fn(|k| s[k / 3][k % 3])),
            state_root: state_root.map(|root| root as u64),
            environment,
            identity,
            strain_hessian: out.strain_hessian.as_ref().map(|h| {
                let mut flat: Vec<f64> = h.strain.iter().flatten().copied().collect();
                for i in 0..h.mixed.nrows() {
                    flat.extend((0..9).map(|k| h.mixed[(i, k)]));
                }
                flat
            }),
        }
    }

    /// Whether this record answers `req` at `sys` for calculator `label`.
    pub fn answers(&self, label: &str, sys: &MolecularSystem, req: &EvalRequest) -> bool {
        let x = sys.coords_flat_bohr();
        self.calculator == label
            && self.charge == i64::from(sys.spin_charge.charge)
            && self.multiplicity == i64::from(sys.spin_charge.multiplicity)
            && self.excited_state == req.excited_state as u64
            && (self.has_gradient || !req.need_gradient)
            && (self.has_hessian || !req.need_hessian)
            && match (&self.cell_bohr, cell_of(sys)) {
                (None, None) => true,
                (Some(a), Some(b)) => a
                    .iter()
                    .zip(&b)
                    .all(|(p, q)| (p - q).abs() <= MATCH_TOLERANCE_BOHR),
                _ => false,
            }
            && self.positions_bohr.len() == x.nrows()
            && self
                .positions_bohr
                .iter()
                .zip(x.iter())
                .all(|(a, b)| (a - b).abs() <= MATCH_TOLERANCE_BOHR)
    }

    fn output(&self, req: &EvalRequest) -> EvalOutput {
        let n = self.gradient.len();
        let mut out = EvalOutput::energy_gradient(
            Quantity::new(self.energy),
            Col::from_fn(n, |i| self.gradient[i]),
        );
        if req.need_hessian {
            out.hessian_bohr = self
                .hessian
                .as_ref()
                .map(|h| Mat::from_fn(n, n, |i, j| h[i * n + j]));
        }
        out.stress = self
            .stress
            .map(|s| std::array::from_fn(|a| std::array::from_fn(|b| s[3 * a + b])));
        if req.need_hessian && req.need_stress {
            out.strain_hessian = self.strain_hessian.as_ref().and_then(|flat| {
                (flat.len() == 81 + 9 * n).then(|| crate::domain::StrainHessian {
                    strain: std::array::from_fn(|x| std::array::from_fn(|y| flat[9 * x + y])),
                    mixed: Mat::from_fn(n, 9, |i, k| flat[81 + 9 * i + k]),
                })
            });
        }
        out
    }

    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64 + 8 * (2 * self.gradient.len()));
        let s = self.calculator.as_bytes();
        b.extend_from_slice(&(s.len() as u32).to_le_bytes());
        b.extend_from_slice(s);
        b.extend_from_slice(&self.charge.to_le_bytes());
        b.extend_from_slice(&self.multiplicity.to_le_bytes());
        b.extend_from_slice(&self.excited_state.to_le_bytes());
        b.push(u8::from(self.has_hessian));
        for v in [&self.positions_bohr, &self.gradient] {
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            v.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
        }
        b.extend_from_slice(&self.energy.to_le_bytes());
        if let Some(h) = &self.hessian {
            b.extend_from_slice(&(h.len() as u64).to_le_bytes());
            h.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
        }
        // The cell, the stress, the followed state, the relaxed environment, the identity, and
        // the strain Hessian follow as an extension: a byte of flags, then nine numbers for
        // each of the first two, one whole number for the third, a counted list of numbers for
        // the fourth, a counted text for the fifth, and a counted list for the sixth.
        // Bit 7 distinguishes explicit gradient presence (bit 6) from older records.
        {
            b.push(
                u8::from(self.cell_bohr.is_some())
                    | (u8::from(self.stress.is_some()) << 1)
                    | (u8::from(self.state_root.is_some()) << 2)
                    | (u8::from(self.environment.is_some()) << 3)
                    | (u8::from(self.identity.is_some()) << 4)
                    | (u8::from(self.strain_hessian.is_some()) << 5)
                    | (u8::from(self.has_gradient) << 6)
                    | 128,
            );
            for nine in [&self.cell_bohr, &self.stress].into_iter().flatten() {
                nine.iter()
                    .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
            }
            if let Some(root) = self.state_root {
                b.extend_from_slice(&root.to_le_bytes());
            }
            if let Some(env) = &self.environment {
                b.extend_from_slice(&(env.len() as u64).to_le_bytes());
                env.iter()
                    .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
            }
            if let Some(identity) = &self.identity {
                let text = identity.as_bytes();
                b.extend_from_slice(&(text.len() as u32).to_le_bytes());
                b.extend_from_slice(text);
            }
            if let Some(strain) = &self.strain_hessian {
                b.extend_from_slice(&(strain.len() as u64).to_le_bytes());
                strain
                    .iter()
                    .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
            }
        }
        b
    }

    fn decode(b: &[u8]) -> Option<Self> {
        let mut r = Reader { b, at: 0 };
        let n = r.u32()? as usize;
        let calculator = String::from_utf8(r.bytes(n)?.to_vec()).ok()?;
        let charge = r.i64()?;
        let multiplicity = r.i64()?;
        let excited_state = r.u64()?;
        let has_hessian = r.bytes(1)?[0] != 0;
        let positions_bohr = r.f64s()?;
        let gradient = r.f64s()?;
        let energy = r.f64()?;
        let hessian = if has_hessian { Some(r.f64s()?) } else { None };
        let (mut cell_bohr, mut stress, mut state_root, mut environment, mut identity) =
            (None, None, None, None, None);
        // Older records did not distinguish a zero gradient from an energy-only
        // placeholder. Only reuse a gradient whose presence can be established.
        let mut has_gradient = has_hessian || gradient.iter().any(|&g| g != 0.0);
        let mut strain_hessian = None;
        if r.at < b.len() {
            let flags = r.bytes(1)?[0];
            if flags & 128 != 0 {
                has_gradient = flags & 64 != 0;
            }
            if flags & 1 != 0 {
                cell_bohr = Some(r.nine()?);
            }
            if flags & 2 != 0 {
                stress = Some(r.nine()?);
            }
            if flags & 4 != 0 {
                state_root = Some(r.u64()?);
            }
            if flags & 8 != 0 {
                environment = Some(r.f64s()?);
            }
            if flags & 16 != 0 {
                let n = r.u32()? as usize;
                identity = Some(String::from_utf8(r.bytes(n)?.to_vec()).ok()?);
            }
            if flags & 32 != 0 {
                strain_hessian = Some(r.f64s()?);
            }
        }
        (r.at == b.len()).then_some(Self {
            calculator,
            charge,
            multiplicity,
            excited_state,
            has_hessian,
            has_gradient,
            positions_bohr,
            energy,
            gradient,
            hessian,
            cell_bohr,
            stress,
            state_root,
            environment,
            identity,
            strain_hessian,
        })
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn bytes(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(s)
    }
    fn arr<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.bytes(N)?.try_into().ok()
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.arr()?))
    }
    fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.arr()?))
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_le_bytes(self.arr()?))
    }
    fn f64s(&mut self) -> Option<Vec<f64>> {
        let n = usize::try_from(self.u64()?).ok()?;
        if n > self.b.len() / 8 {
            return None;
        }
        (0..n).map(|_| self.f64()).collect()
    }
    fn nine(&mut self) -> Option<[f64; 9]> {
        let mut out = [0.0; 9];
        for value in &mut out {
            *value = self.f64()?;
        }
        Some(out)
    }
}

/// The records of a journal file; a missing file is empty, a damaged tail is dropped.
/// Returns the records and the length of the valid part.
pub fn read_journal(path: &Path) -> MoptResult<(Vec<EvalRecord>, u64)> {
    let mut data = Vec::new();
    match File::open(path) {
        Ok(mut f) => {
            f.read_to_end(&mut data)?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(e.into()),
    }
    if data.is_empty() {
        return Ok((Vec::new(), 0));
    }
    if data.len() < MAGIC.len() || &data[..MAGIC.len()] != MAGIC {
        return Err(MoptError::config(format!(
            "{} is not a mopt-rs evaluation journal",
            path.display()
        )));
    }
    let mut records = Vec::new();
    let mut at = MAGIC.len();
    while let Some(len) = data
        .get(at..at + 4)
        .and_then(|b| b.try_into().ok())
        .map(|b: [u8; 4]| u32::from_le_bytes(b) as usize)
    {
        let (start, end) = (at + 4, at + 4 + len);
        let (Some(payload), Some(sum)) = (data.get(start..end), data.get(end..end + 4)) else {
            break;
        };
        let sum = u32::from_le_bytes([sum[0], sum[1], sum[2], sum[3]]);
        let Some(rec) = (crc32(payload) == sum)
            .then(|| EvalRecord::decode(payload))
            .flatten()
        else {
            break;
        };
        records.push(rec);
        at = end + 4;
    }
    Ok((records, at as u64))
}

/// The evaluations of an earlier run of a workflow, from every journal under its directory,
/// offered to the journals of the commands of its next run ([`RunJournal::open_with`]). A
/// workflow restarted from its directory runs its commands again; each asks for what it asked
/// before, and an evaluation of the same calculator (label and description), state, request,
/// and identical structure answers it. Records are kept by the exact coordinates, so a lookup
/// is a hash; each record answers once. Records written before the description was recorded
/// are not offered.
pub struct ReplayPool {
    records: Mutex<HashMap<u64, Vec<EvalRecord>>>,
    offered: usize,
    taken: AtomicUsize,
}

/// The key of a record: everything that must be identical for it to answer, but the Hessian.
fn replay_key(
    label: &str,
    identity: &str,
    charge: i64,
    multiplicity: i64,
    excited_state: u64,
    positions: &[f64],
    cell: Option<[f64; 9]>,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (label, identity, charge, multiplicity, excited_state).hash(&mut h);
    positions.len().hash(&mut h);
    positions.iter().for_each(|x| x.to_bits().hash(&mut h));
    cell.map(|c| c.map(f64::to_bits)).hash(&mut h);
    h.finish()
}

/// The journal files under `dir`, all levels (symbolic links are not followed).
fn journal_files(dir: &Path, out: &mut Vec<PathBuf>) -> MoptResult<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_dir() {
            journal_files(&path, out)?;
        } else if kind.is_file() && entry.file_name() == JOURNAL_FILE {
            out.push(path);
        }
    }
    Ok(())
}

impl ReplayPool {
    /// The records of every journal under `dir`.
    pub fn from_directory(dir: &Path) -> MoptResult<Self> {
        let mut files = Vec::new();
        journal_files(dir, &mut files)?;
        files.sort();
        let mut records: HashMap<u64, Vec<EvalRecord>> = HashMap::new();
        let mut offered = 0;
        for file in &files {
            for rec in read_journal(file)?.0 {
                let Some(identity) = rec.identity.as_deref() else {
                    continue;
                };
                let key = replay_key(
                    &rec.calculator,
                    identity,
                    rec.charge,
                    rec.multiplicity,
                    rec.excited_state,
                    &rec.positions_bohr,
                    rec.cell_bohr,
                );
                records.entry(key).or_default().push(rec);
                offered += 1;
            }
        }
        Ok(Self {
            records: Mutex::new(records),
            offered,
            taken: AtomicUsize::new(0),
        })
    }
}

impl ReplaySource for ReplayPool {
    fn take(
        &self,
        label: &str,
        identity: &str,
        sys: &MolecularSystem,
        req: &EvalRequest,
    ) -> Option<Replayed> {
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let key = replay_key(
            label,
            identity,
            i64::from(sys.spin_charge.charge),
            i64::from(sys.spin_charge.multiplicity),
            req.excited_state as u64,
            &x,
            cell_of(sys),
        );
        let mut map = self.records.lock().ok()?;
        let list = map.get_mut(&key)?;
        let fits = |r: &EvalRecord, kind: bool| {
            (!kind || r.has_hessian == req.need_hessian)
                && r.identity.as_deref() == Some(identity)
                && r.positions_bohr.iter().zip(&x).all(|(a, b)| a == b)
                && r.answers(label, sys, req)
        };
        let at = list
            .iter()
            .position(|r| fits(r, true))
            .or_else(|| list.iter().position(|r| fits(r, false)))?;
        let rec = list.remove(at);
        self.taken.fetch_add(1, Ordering::Relaxed);
        Some(Replayed {
            output: rec.output(req),
            state_root: rec.state_root.map(|root| root as usize),
            environment: rec.environment,
        })
    }

    fn counts(&self) -> (usize, usize) {
        (self.offered, self.taken.load(Ordering::Relaxed))
    }
}

/// An append-only journal file.
#[derive(Debug)]
pub struct Journal {
    path: PathBuf,
    file: File,
}

impl Journal {
    /// Open (or create) the journal at `path` for appending, dropping a damaged tail; returns
    /// it with the records already in it.
    pub fn open(path: impl Into<PathBuf>) -> MoptResult<(Self, Vec<EvalRecord>)> {
        let path = path.into();
        let (records, valid) = read_journal(&path)?;
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        if valid == 0 {
            file.set_len(0)?;
            file.write_all(MAGIC)?;
        } else {
            file.set_len(valid)?;
        }
        file.seek(SeekFrom::End(0))?;
        file.sync_data()?;
        Ok((Self { path, file }, records))
    }

    /// Where the journal is.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append `record` durably.
    pub fn append(&mut self, record: &EvalRecord) -> MoptResult<()> {
        let payload = record.encode();
        let len = u32::try_from(payload.len())
            .map_err(|_| MoptError::calc("evaluation record too large for the journal"))?;
        let mut buf = Vec::with_capacity(payload.len() + 8);
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&payload);
        buf.extend_from_slice(&crc32(&payload).to_le_bytes());
        self.file.write_all(&buf)?;
        self.file.sync_data()?;
        Ok(())
    }
}

struct JournalState {
    journal: Journal,
    pending: Vec<EvalRecord>,
    replayed: usize,
}

/// A calculator whose evaluations are recorded, and answered from earlier records on restart.
pub struct JournaledCalculator<'a> {
    inner: &'a dyn Calculator,
    label: String,
    state: &'a Mutex<JournalState>,
    replay: Option<&'a dyn ReplaySource>,
}

/// Journal shared by the calculators of one run.
pub struct RunJournal {
    state: Mutex<JournalState>,
    replay: Option<Arc<dyn ReplaySource>>,
}

impl RunJournal {
    /// The journal of run directory `dir` (records already there are replayed).
    pub fn open(dir: &Path) -> MoptResult<Self> {
        Self::open_with(dir, None)
    }

    /// [`Self::open`] with the evaluations of `replay` (those of an earlier run of a workflow,
    /// [`ReplayPool`]) taken where the journal has none for a request.
    pub fn open_with(dir: &Path, replay: Option<Arc<dyn ReplaySource>>) -> MoptResult<Self> {
        let (journal, pending) = Journal::open(dir.join(JOURNAL_FILE))?;
        Ok(Self {
            state: Mutex::new(JournalState {
                journal,
                pending,
                replayed: 0,
            }),
            replay,
        })
    }

    /// `calc` recording into (and replaying from) this journal under `label`.
    pub fn wrap<'a>(&'a self, label: &str, calc: &'a dyn Calculator) -> JournaledCalculator<'a> {
        JournaledCalculator {
            inner: calc,
            label: label.to_string(),
            state: &self.state,
            replay: self.replay.as_deref(),
        }
    }

    /// (records answered from the journal, records left unused).
    pub fn replay_counts(&self) -> (usize, usize) {
        self.state
            .lock()
            .map(|s| (s.replayed, s.pending.len()))
            .unwrap_or((0, 0))
    }
}

impl Calculator for JournaledCalculator<'_> {
    fn analytic_hessian(&self) -> bool {
        self.inner.analytic_hessian()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn citations(&self) -> Vec<String> {
        self.inner.citations()
    }

    fn description(&self) -> String {
        self.inner.description()
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        {
            let mut st = self
                .state
                .lock()
                .map_err(|_| MoptError::calc("journal lock poisoned"))?;
            // A structure may have been evaluated twice, without and with a Hessian (a
            // point that is tested first and becomes a point with a Hessian later): the
            // record of the same kind as the request is taken, so that the one with the
            // Hessian is left for the request that needs it. Near convergence, successive
            // structures can lie within the tolerance of each other: the record of the
            // identical structure comes first, then the earliest one within the tolerance
            // (the records keep their order, so that a restarted run takes them in the
            // order the first run made them).
            let x = sys.coords_flat_bohr();
            let identical = |r: &EvalRecord| {
                r.positions_bohr.len() == x.nrows()
                    && r.positions_bohr.iter().zip(x.iter()).all(|(a, b)| a == b)
            };
            let pick = |exact: bool, kind: bool| {
                st.pending.iter().position(|r| {
                    (!kind || r.has_hessian == req.need_hessian)
                        && (!exact || identical(r))
                        && r.answers(&self.label, sys, req)
                })
            };
            let found = pick(true, true)
                .or_else(|| pick(true, false))
                .or_else(|| pick(false, true))
                .or_else(|| pick(false, false));
            if let Some(k) = found {
                let rec = st.pending.remove(k);
                st.replayed += 1;
                if let Some(root) = rec.state_root {
                    // A calculator that follows its states goes on from the recorded one.
                    self.inner
                        .recall_state(sys, req.excited_state, root as usize);
                }
                if let Some(env) = &rec.environment {
                    // A calculator that relaxes an environment goes on from the recorded one.
                    self.inner.recall_environment(sys, env);
                }
                return Ok(rec.output(req));
            }
        }
        let identity = self.inner.description();
        if let Some(r) = self
            .replay
            .and_then(|source| source.take(&self.label, &identity, sys, req))
        {
            // An evaluation of the run before this one (a restarted workflow): recorded here
            // too, so that this run restarts from its own journal.
            if let Some(root) = r.state_root {
                self.inner.recall_state(sys, req.excited_state, root);
            }
            if let Some(env) = &r.environment {
                self.inner.recall_environment(sys, env);
            }
            let rec = EvalRecord::new(
                &self.label,
                sys,
                req,
                &r.output,
                r.state_root,
                r.environment,
                Some(identity),
            );
            let mut st = self
                .state
                .lock()
                .map_err(|_| MoptError::calc("journal lock poisoned"))?;
            st.journal.append(&rec)?;
            st.replayed += 1;
            return Ok(r.output);
        }
        let out = self.inner.evaluate(sys, req)?;
        let state_root = (req.excited_state > 0)
            .then(|| self.inner.followed_state(sys, req.excited_state))
            .flatten();
        let environment = self.inner.relaxed_environment(sys);
        let rec = EvalRecord::new(
            &self.label,
            sys,
            req,
            &out,
            state_root,
            environment,
            Some(identity),
        );
        self.state
            .lock()
            .map_err(|_| MoptError::calc("journal lock poisoned"))?
            .journal
            .append(&rec)?;
        Ok(out)
    }

    fn supports_pbc(&self) -> bool {
        self.inner.supports_pbc()
    }

    fn is_rigid_body_invariant(&self) -> bool {
        self.inner.is_rigid_body_invariant()
    }

    fn evaluates_in_order(&self) -> bool {
        self.inner.evaluates_in_order()
    }

    fn supports_excited_states(&self) -> bool {
        self.inner.supports_excited_states()
    }

    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        self.inner.excitation_energies(sys)
    }

    fn state_characters(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<crate::pes::states::StateCharacters>> {
        self.inner.state_characters(sys, count)
    }

    fn transitions(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<Vec<crate::pes::states::Transition>>> {
        self.inner.transitions(sys, count)
    }

    fn followed_state(&self, sys: &MolecularSystem, label: usize) -> Option<usize> {
        self.inner.followed_state(sys, label)
    }

    fn recall_state(&self, sys: &MolecularSystem, label: usize, root: usize) {
        self.inner.recall_state(sys, label, root);
    }

    fn relaxed_environment(&self, sys: &MolecularSystem) -> Option<Vec<f64>> {
        self.inner.relaxed_environment(sys)
    }

    fn recall_environment(&self, sys: &MolecularSystem, environment: &[f64]) {
        self.inner.recall_environment(sys, environment);
    }

    fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Option<Orbitals>> {
        self.inner.orbitals(sys)
    }

    fn uses_electrons(&self) -> bool {
        self.inner.uses_electrons()
    }

    fn take_warnings(&self) -> Vec<String> {
        self.inner.take_warnings()
    }

    fn site_terms(
        &self,
        sys: &MolecularSystem,
    ) -> Option<MoptResult<crate::pes::calculator::SiteTerms>> {
        self.inner.site_terms(sys)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn rec(e: f64, hess: bool) -> EvalRecord {
        EvalRecord {
            calculator: "main".into(),
            charge: -1,
            multiplicity: 2,
            excited_state: 0,
            has_hessian: hess,
            has_gradient: true,
            positions_bohr: vec![0.0, 1.0, 2.0],
            energy: e,
            gradient: vec![0.1, 0.2, 0.3],
            hessian: hess.then(|| (0..9).map(f64::from).collect()),
            // The records with a Hessian stand for structures in a cell, with a stress.
            cell_bohr: hess.then(|| std::array::from_fn(|k| if k % 4 == 0 { 7.5 } else { 0.0 })),
            stress: (hess && e < -2.5).then(|| std::array::from_fn(|k| 1e-4 * k as f64)),
            state_root: None,
            // The records without a Hessian stand for cores with a relaxed environment.
            environment: (!hess).then(|| vec![1.5, -0.25, 3.0, 0.125]),
            identity: (e < -1.5).then(|| "GFN1-xTB, 300 K".to_string()),
            // The records in a cell with a stress have the strain Hessian too.
            strain_hessian: (hess && e < -2.5)
                .then(|| (0..81 + 9 * 3).map(|k| 0.5 * k as f64).collect()),
        }
    }

    #[test]
    fn an_energy_only_placeholder_and_a_stationary_gradient_are_distinct() {
        let mut record = rec(-1.0, false);
        record.environment = None;
        record.gradient.fill(0.0);
        record.has_gradient = false;
        assert!(!EvalRecord::decode(&record.encode()).unwrap().has_gradient);
        record.has_gradient = true;
        let mut bytes = record.encode();
        assert!(EvalRecord::decode(&bytes).unwrap().has_gradient);
        bytes.pop(); // Legacy records without extension flags.
        assert!(!EvalRecord::decode(&bytes).unwrap().has_gradient);
    }

    #[test]
    fn the_followed_state_is_recorded() {
        // Without and with the other parts of the extension.
        let mut record = rec(-1.0, false);
        record.excited_state = 1;
        record.state_root = Some(2);
        assert_eq!(EvalRecord::decode(&record.encode()).unwrap(), record);
        let mut in_a_cell = rec(-3.0, true);
        in_a_cell.state_root = Some(0);
        assert_eq!(EvalRecord::decode(&in_a_cell.encode()).unwrap(), in_a_cell);
        // A record without it reads as before.
        assert_eq!(
            EvalRecord::decode(&rec(-1.0, false).encode())
                .unwrap()
                .state_root,
            None
        );
    }

    #[test]
    fn the_cell_is_part_of_what_a_record_answers() {
        use crate::domain::{positions, Angstrom, Element, PeriodicCell, SpinCharge, Vector3};
        let at = |a: f64| {
            let cell = PeriodicCell::from_lattice_vectors(
                Vector3::<Angstrom>::new([a, 0.0, 0.0]),
                Vector3::<Angstrom>::new([0.0, a, 0.0]),
                Vector3::<Angstrom>::new([0.0, 0.0, a]),
                [true, true, true],
            )
            .unwrap();
            MolecularSystem::new(
                vec![Element::parse_token("Cu").unwrap()],
                &positions::<Angstrom>(&[[0.0, 0.0, 0.0]]),
                SpinCharge::default(),
                Some(cell),
            )
            .unwrap()
        };
        let mut out = EvalOutput::energy_gradient(Quantity::new(-1.0), Col::zeros(3));
        out.stress = Some([[1e-4, 0.0, 0.0], [0.0, 2e-4, 0.0], [0.0, 0.0, 3e-4]]);
        let request = EvalRequest::ENERGY_GRAD_STRESS;
        let record = EvalRecord::new("main", &at(7.0), &request, &out, None, None, None);
        // One atom at the origin: the positions do not tell the two cells apart.
        assert!(record.answers("main", &at(7.0), &request));
        assert!(!record.answers("main", &at(7.1), &request));
        let back = EvalRecord::decode(&record.encode()).unwrap();
        assert_eq!(back, record);
        assert_eq!(back.output(&request).stress, out.stress);
    }

    #[test]
    fn records_survive_and_damaged_tails_are_dropped() {
        let dir = std::env::temp_dir().join(format!("mopt_journal_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(JOURNAL_FILE);
        {
            let (mut j, old) = Journal::open(&path).unwrap();
            assert!(old.is_empty());
            j.append(&rec(-1.0, false)).unwrap();
            j.append(&rec(-2.0, true)).unwrap();
        }
        // A record cut in the middle (killed while writing).
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() - 7]).unwrap();
        let (mut j, old) = Journal::open(&path).unwrap();
        assert_eq!(old, vec![rec(-1.0, false)]);
        j.append(&rec(-3.0, true)).unwrap();
        drop(j);
        let (all, _) = read_journal(&path).unwrap();
        assert_eq!(all, vec![rec(-1.0, false), rec(-3.0, true)]);
        std::fs::write(&path, b"garbage!").unwrap();
        assert!(read_journal(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A calculator that must not be called: every request is answered from the journal.
    struct Unused;

    impl Calculator for Unused {
        fn name(&self) -> &str {
            "unused"
        }
        fn evaluate(&self, _: &MolecularSystem, _: &EvalRequest) -> MoptResult<EvalOutput> {
            Err(MoptError::calc("not answered from the journal"))
        }
    }

    #[test]
    fn a_replay_takes_the_identical_structure_and_keeps_the_order() {
        use crate::domain::{positions, Angstrom, Element, SpinCharge};
        let dir = std::env::temp_dir().join(format!("mopt_journal_order_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let x = MolecularSystem::new(
            vec![Element::parse_token("H").unwrap()],
            &positions::<Angstrom>(&[[0.1, 0.2, 0.3]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let mut moved = x.coords_flat_bohr();
        moved[0] += 0.5 * MATCH_TOLERANCE_BOHR;
        let near = x.with_flat_coords_bohr(&moved).unwrap();
        let request = EvalRequest::ENERGY_GRAD;
        let out = |e: f64| EvalOutput::energy_gradient(Quantity::new(e), Col::zeros(3));
        {
            let (mut j, _) = Journal::open(dir.join(JOURNAL_FILE)).unwrap();
            // A structure within the tolerance of the next one, then the same structure
            // evaluated three times (with results told apart by their energies).
            j.append(&EvalRecord::new(
                "main",
                &near,
                &request,
                &out(-0.5),
                None,
                None,
                None,
            ))
            .unwrap();
            for e in [-1.0, -2.0, -3.0] {
                j.append(&EvalRecord::new(
                    "main",
                    &x,
                    &request,
                    &out(e),
                    None,
                    None,
                    None,
                ))
                .unwrap();
            }
        }
        let journal = RunJournal::open(&dir).unwrap();
        let calc = journal.wrap("main", &Unused);
        let energy = |s: &MolecularSystem| calc.evaluate(s, &request).unwrap().energy.value();
        // The identical structure is taken before the earlier one within the tolerance, and
        // the repeated evaluations come back in the order they were made.
        assert_eq!(energy(&x), -1.0);
        assert_eq!(energy(&x), -2.0);
        assert_eq!(energy(&x), -3.0);
        assert_eq!(energy(&near), -0.5);
        assert_eq!(journal.replay_counts(), (4, 0));
        assert!(calc.evaluate(&x, &request).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_pool_of_the_journals_under_a_directory_answers_its_own_calculator_once() {
        use crate::domain::{positions, Angstrom, Element, SpinCharge};
        let dir = std::env::temp_dir().join(format!("mopt_journal_pool_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let nested = dir.join("step2").join("run");
        std::fs::create_dir_all(&nested).unwrap();
        let x = MolecularSystem::new(
            vec![Element::parse_token("H").unwrap()],
            &positions::<Angstrom>(&[[0.1, 0.2, 0.3]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let request = EvalRequest::ENERGY_GRAD;
        let out = |e: f64| EvalOutput::energy_gradient(Quantity::new(e), Col::zeros(3));
        let identity = Unused.description();
        {
            let (mut j, _) = Journal::open(nested.join(JOURNAL_FILE)).unwrap();
            let rec = |e: f64, environment: Option<Vec<f64>>, who: Option<String>| {
                EvalRecord::new("main", &x, &request, &out(e), None, environment, who)
            };
            j.append(&rec(
                -1.0,
                Some(vec![0.5, 0.25, 0.125]),
                Some(identity.clone()),
            ))
            .unwrap();
            // Another calculator under the same label, and a record without a description
            // (written before descriptions were recorded): neither answers.
            j.append(&rec(-2.0, None, Some("another level".to_string())))
                .unwrap();
            j.append(&rec(-3.0, None, None)).unwrap();
        }
        let pool = Arc::new(ReplayPool::from_directory(&dir).unwrap());
        assert_eq!(pool.counts(), (2, 0));
        let fresh = dir.join("again");
        std::fs::create_dir_all(&fresh).unwrap();
        let journal = RunJournal::open_with(&fresh, Some(pool.clone())).unwrap();
        // Another label does not take it.
        assert!(journal
            .wrap("second", &Unused)
            .evaluate(&x, &request)
            .is_err());
        let calc = journal.wrap("main", &Unused);
        assert_eq!(calc.evaluate(&x, &request).unwrap().energy.value(), -1.0);
        assert!(calc.evaluate(&x, &request).is_err());
        assert_eq!(pool.counts(), (2, 1));
        // The new run keeps what it took in its own journal, with the environment.
        let (kept, _) = read_journal(&fresh.join(JOURNAL_FILE)).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].environment, Some(vec![0.5, 0.25, 0.125]));
        assert_eq!(kept[0].identity.as_deref(), Some(identity.as_str()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
