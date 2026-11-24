//! Whole commands give the same bits for any number of threads: every run is made inside rayon
//! pools of 1, 2, 4 and 8 threads with `-core` set to the same number, and what it reports
//! (the result it returns, with every energy, structure and count in it) and every file it
//! leaves (logs, trajectories, CSV tables, figures) are compared byte by byte, after the
//! directory of the run, the times of day in the names of the run directories and the
//! measured times have been taken out.
//!
//! The cluster of the minimum in Cartesian coordinates and of the vibrations is large enough
//! (375 coordinates) for faer's eigendecompositions to split their work into tasks when they
//! are allowed to; the other cases are small, so that each takes seconds.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mopt_rs::cli::option::parser;
use mopt_rs::cli::{
    autots, confsearch, freqmain, ieipmain, ircmain, mapper, mdmain, nebmain, optmain,
    orientsearch, relaxedscan, spec,
};
use mopt_rs::domain::Units;
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::native::muller_brown::MB_SADDLES;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::journal::{read_journal, JOURNAL_FILE};

/// The threads of the runs.
const THREADS: [usize; 4] = [1, 2, 4, 8];

fn failure(e: impl std::fmt::Display) -> MoptError {
    MoptError::config(e.to_string())
}

fn scratch(case: &str, run: usize) -> MoptResult<PathBuf> {
    let dir =
        std::env::temp_dir().join(format!("mopt_threads_{case}_{}_{run}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(failure)?;
    Ok(dir)
}

fn write(dir: &Path, name: &str, text: &str) -> MoptResult<String> {
    let path = dir.join(name);
    std::fs::write(&path, text).map_err(failure)?;
    Ok(path.to_string_lossy().into_owned())
}

/// `text` with every time of day of a run-directory name (`YYYY_MM_DD`, optionally followed
/// by `_HH_MM_SS_ffff`) replaced by `<time>`.
fn without_dates(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let digits = |at: usize, n: usize| {
            at + n <= bytes.len() && bytes[at..at + n].iter().all(u8::is_ascii_digit)
        };
        let starts_number = i == 0 || !bytes[i - 1].is_ascii_digit();
        if starts_number
            && digits(i, 4)
            && bytes.get(i + 4) == Some(&b'_')
            && digits(i + 5, 2)
            && bytes.get(i + 7) == Some(&b'_')
            && digits(i + 8, 2)
        {
            let mut j = i + 10;
            while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b'_') {
                j += 1;
            }
            out.push_str("<time>");
            i = j;
        } else {
            let c = text[i..].chars().next().unwrap_or(' ');
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// Whether a line of a log or a table is about the time a run took (its numbers are measured
/// times, not results).
fn is_about_time(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    [
        "elapsed",
        "wall",
        " took ",
        "seconds",
        "time:",
        "time (s)",
        "time_s",
        "duration",
        " s)",
        " s,",
        " s;",
        " s.",
        " s:",
        "ms)",
        "started at",
        "finished at",
    ]
    .iter()
    .any(|w| lower.contains(w))
        || lower.trim_end().ends_with(" s")
}

/// The words of `line` without its numbers (a line about a time).
fn without_numbers(line: &str) -> String {
    line.split(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == ')')
        .filter(|w| {
            let w = w.trim_end_matches(['s', ';', '.', ':']);
            w.parse::<f64>().is_err()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `text` with the number that follows each `key` (within a few characters: quotes, a colon,
/// a bracket) replaced by `#`: the number of threads the run was asked for, which its
/// settings and its result repeat.
fn without_value_of(text: &str, key: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(key) {
        let (head, tail) = rest.split_at(at + key.len());
        out.push_str(head);
        let start = tail
            .char_indices()
            .take(24)
            .find(|(_, c)| c.is_ascii_digit())
            .map(|(i, _)| i);
        match start {
            Some(s) => {
                let end = tail[s..]
                    .find(|c: char| !c.is_ascii_digit())
                    .map_or(tail.len(), |e| s + e);
                out.push_str(&tail[..s]);
                out.push('#');
                rest = &tail[end..];
            }
            None => rest = tail,
        }
    }
    out.push_str(rest);
    out
}

/// `text` with the run directory `base`, the number of threads asked for and the times
/// taken out.
fn normalized(text: &str, base: &Path) -> String {
    let base_text = base.to_string_lossy();
    let replaced = text.replace(base_text.as_ref(), "<base>");
    let replaced = replaced.replace(&base_text.replace('\\', "/"), "<base>");
    let replaced = without_dates(&replaced);
    let replaced = without_value_of(&replaced, "N_THREAD");
    let replaced = without_value_of(&replaced, "threads:");
    let replaced = without_value_of(&replaced, "\"-core\"");
    // The workers of the mapper, which its settings and its report repeat.
    let replaced = without_value_of(&replaced, "workers:");
    let replaced = without_value_of(&replaced, "workers=");
    let replaced = without_value_of(&replaced, "Jobs:");
    let replaced = without_value_of(&replaced, "threads =");
    replaced
        .lines()
        .map(|l| {
            if is_about_time(l) {
                without_numbers(l)
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every file under `dir`, by its path relative to `base` (dates taken out), normalized.
fn collect(dir: &Path, base: &Path, out: &mut BTreeMap<String, Vec<u8>>) -> MoptResult<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(failure)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect(&path, base, out)?;
            continue;
        }
        let relative = path
            .strip_prefix(base)
            .map_err(failure)?
            .to_string_lossy()
            .replace('\\', "/");
        if path.file_name().is_some_and(|f| f == JOURNAL_FILE) {
            // Evaluations that run at the same time are journaled as they end: the records
            // are compared as a set (a restart looks them up by their structure).
            let (records, _) = read_journal(&path)?;
            let mut lines: Vec<String> = records.iter().map(|r| format!("{r:?}")).collect();
            lines.sort();
            out.insert(without_dates(&relative), lines.join("\n").into_bytes());
            continue;
        }
        let bytes = std::fs::read(&path).map_err(failure)?;
        let content = match String::from_utf8(bytes) {
            Ok(text) => normalized(&text, base).into_bytes(),
            Err(e) => e.into_bytes(),
        };
        out.insert(without_dates(&relative), content);
    }
    Ok(())
}

/// The first line where `a` and `b` differ, for the message of a failure.
fn first_difference(a: &[u8], b: &[u8]) -> String {
    let (a, b) = (String::from_utf8_lossy(a), String::from_utf8_lossy(b));
    for (k, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            // Long lines (a result printed whole) from a little before the first difference.
            let at = x
                .char_indices()
                .zip(y.chars())
                .find(|((_, p), q)| p != q)
                .map_or(0, |((i, _), _)| i);
            let from = x[..at].char_indices().rev().nth(150).map_or(0, |(i, _)| i);
            let cut = |t: &str| {
                t.get(from..)
                    .unwrap_or(t)
                    .chars()
                    .take(400)
                    .collect::<String>()
            };
            return format!("line {}:\n  {}\n  {}", k + 1, cut(x), cut(y));
        }
    }
    format!("{} lines against {}", a.lines().count(), b.lines().count())
}

/// What a run reported and left behind.
struct Outcome {
    reported: String,
    files: BTreeMap<String, Vec<u8>>,
}

/// Run `case` (given its base directory and the threads, as the text of `-core`) in pools of
/// [`THREADS`] and require the same bits from every run; what the first run gave.
fn same_across_threads(
    case: &str,
    run: impl Fn(&Path, &str) -> MoptResult<String> + Sync,
) -> MoptResult<()> {
    outcome_across_threads(case, run).map(|_| ())
}

/// [`same_across_threads`], with the outcome of the first run.
fn outcome_across_threads(
    case: &str,
    run: impl Fn(&Path, &str) -> MoptResult<String> + Sync,
) -> MoptResult<Outcome> {
    let start = std::time::Instant::now();
    let mut first: Option<Outcome> = None;
    for (k, threads) in THREADS.into_iter().enumerate() {
        let base = scratch(case, k)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(failure)?;
        let reported = pool.install(|| run(&base, &threads.to_string()))?;
        let mut files = BTreeMap::new();
        collect(&base, &base, &mut files)?;
        let outcome = Outcome {
            reported: normalized(&reported, &base),
            files,
        };
        let _ = std::fs::remove_dir_all(&base);
        match &first {
            None => first = Some(outcome),
            Some(reference) => same_outcome(
                case,
                &format!("{threads} threads"),
                reference,
                &outcome,
                &|_| true,
            )?,
        }
    }
    eprintln!(
        "{case}: {} runs took {:.1} s",
        THREADS.len(),
        start.elapsed().as_secs_f64()
    );
    first.ok_or_else(|| MoptError::calc("no run"))
}

/// Require `outcome` to be `reference`, in the result reported and in the files that
/// `compared` admits.
fn same_outcome(
    case: &str,
    what: &str,
    reference: &Outcome,
    outcome: &Outcome,
    compared: &dyn Fn(&str) -> bool,
) -> MoptResult<()> {
    if reference.reported != outcome.reported {
        return Err(MoptError::calc(format!(
            "{case}: {what} report other results than 1 thread: {}",
            first_difference(reference.reported.as_bytes(), outcome.reported.as_bytes())
        )));
    }
    let names = |o: &Outcome| {
        o.files
            .keys()
            .filter(|n| compared(n))
            .cloned()
            .collect::<Vec<_>>()
    };
    if names(reference) != names(outcome) {
        return Err(MoptError::calc(format!(
            "{case}: {what} leave other files than 1 thread: {:?} against {:?}",
            names(reference),
            names(outcome)
        )));
    }
    for (name, bytes) in reference.files.iter().filter(|(n, _)| compared(n)) {
        if outcome.files.get(name) != Some(bytes) {
            return Err(MoptError::calc(format!(
                "{case}: {what} write another {name} than 1 thread: {}",
                first_difference(bytes, outcome.files.get(name).map_or(&[], Vec::as_slice))
            )));
        }
    }
    Ok(())
}

/// An argon cluster of `m^3` atoms on a slightly distorted simple cubic grid (Angstrom).
fn argon_cluster(m: usize) -> String {
    let spacing = 3.75;
    let mut text = format!("{}\nargon cluster\n", m * m * m);
    let mut k = 0usize;
    for i in 0..m {
        for j in 0..m {
            for l in 0..m {
                k += 1;
                let jitter = 0.12 * ((k as f64) * 0.731).sin();
                text.push_str(&format!(
                    "Ar {:.6} {:.6} {:.6}\n",
                    i as f64 * spacing + jitter,
                    j as f64 * spacing - 0.7 * jitter,
                    l as f64 * spacing + 0.4 * jitter
                ));
            }
        }
    }
    text
}

const HCN: &str =
    "3\nHCN\nC -0.025197409828 0.0 0.0\nN 1.115411466511 0.0 0.0\nH -1.090214056682 0.0 0.0\n";
const HNC: &str =
    "3\nHNC\nC -1.099032916795 0.0 0.0\nN 0.054338186952 0.0 0.0\nH 1.044694729843 0.0 0.0\n";

/// HCN, a bridged structure and HNC, for a band.
const HCN_PATH: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nbridged\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";

const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

const BUTANE: &str = "14\nn-butane all-anti\nC 0.000000 0.000000 0.000000\n\
C 1.268427 0.855565 0.000000\nC 2.536855 0.000000 0.000000\nC 3.805282 0.855565 0.000000\n\
H -0.875864 0.648816 0.000000\nH -0.013866 -0.629150 0.889988\nH -0.013866 -0.629150 -0.889988\n\
H 1.268427 1.484886 0.889975\nH 1.268427 1.484886 -0.889975\nH 2.536855 -0.629321 -0.889975\n\
H 2.536855 -0.629321 0.889975\nH 4.681146 0.206749 0.000000\nH 3.819148 1.484715 0.889988\n\
H 3.819148 1.484715 -0.889988\n";

fn optmain_case(base: &Path, threads: &str, m: usize, extra: &[&str]) -> MoptResult<String> {
    let input = write(base, "ar.xyz", &argon_cluster(m))?;
    let mut args = vec![input.as_str(), "-os", "lj", "-core", threads];
    args.extend_from_slice(extra);
    let config = optmain::config_from_cli(&args)?;
    let registry = CalculatorRegistry::native();
    let out = optmain::run_with(&config, base, None, &registry)?;
    Ok(format!("{out:?}\n{:?}", registry.counts()))
}

/// RS-I-RFO with exact Hessians, a restraint and a constraint.
const BIASED: [&str; 11] = [
    "-fc", "1", "-ns", "4", "-kp", "0.01", "4.0", "1,2", "-pc", "bond", "3,4",
];

#[test]
fn optmain_minimum_cartesian() -> MoptResult<()> {
    same_across_threads("opt_cart", |base, threads| {
        optmain_case(base, threads, 5, &BIASED)
    })
}

#[test]
fn optmain_minimum_dic() -> MoptResult<()> {
    same_across_threads("opt_dic", |base, threads| {
        let mut extra = BIASED.to_vec();
        extra.extend(["-coord", "dic"]);
        optmain_case(base, threads, 3, &extra)
    })
}

#[test]
fn optmain_saddle_cartesian_and_dic() -> MoptResult<()> {
    for coord in ["cart", "dic"] {
        same_across_threads(&format!("opt_order1_{coord}"), |base, threads| {
            let mut extra = BIASED.to_vec();
            extra.extend(["-order", "1", "-coord", coord]);
            optmain_case(base, threads, 3, &extra)
        })?;
    }
    Ok(())
}

#[test]
fn freqmain_of_a_cluster() -> MoptResult<()> {
    same_across_threads("freq", |base, threads| {
        let input = write(base, "ar125.xyz", &argon_cluster(5))?;
        let config = freqmain::config_from_cli(&[&input, "-os", "lj", "-core", threads])?;
        let registry = CalculatorRegistry::native();
        let out = freqmain::run_with(&config, base, None, &registry)?;
        Ok(format!("{out:?}\n{:?}", registry.counts()))
    })
}

#[test]
fn ircmain_of_a_model_surface() -> MoptResult<()> {
    same_across_threads("irc", |base, threads| {
        let [x, y] = MB_SADDLES[0];
        let ts = write(base, "ts.txt", &format!("x {x}\ny {y}\n"))?;
        let config = ircmain::config_from_cli(&[
            &ts, "-os", "mb2d", "-irc", "0.05", "200", "lqa", "-core", threads,
        ])?;
        let registry = CalculatorRegistry::native();
        let out = ircmain::run_with(&config, base, None, &registry)?;
        Ok(format!("{out:?}\n{:?}", registry.counts()))
    })
}

#[test]
fn nebmain_with_images_in_parallel() -> MoptResult<()> {
    same_across_threads("neb", |base, threads| {
        let input = write(base, "path.xyz", HCN_PATH)?;
        let config = nebmain::config_from_cli(&[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "6",
            "-p",
            "5",
            "-parallel",
            "4",
            "-core",
            threads,
        ])?;
        let registry = CalculatorRegistry::native();
        let out = nebmain::run_with(&config, base, None, &registry)?;
        Ok(format!("{out:?}\n{:?}", registry.counts()))
    })
}

#[test]
fn ieipmain_two_methods() -> MoptResult<()> {
    for method in ["-gnt", "-use_dimer"] {
        same_across_threads(&format!("ieip{method}"), |base, threads| {
            let pair = base.join("pair");
            std::fs::create_dir_all(&pair).map_err(failure)?;
            write(&pair, "hcn_A.xyz", HCN)?;
            write(&pair, "hnc_B.xyz", HNC)?;
            let pair = pair.to_string_lossy().into_owned();
            let config = ieipmain::config_from_cli(&[
                &pair, "-os", "gfn0xtb", method, "-ns", "20", "-core", threads,
            ])?;
            let registry = CalculatorRegistry::native();
            let out = ieipmain::run_with(&config, base, None, &registry)?;
            Ok(format!("{out:?}\n{:?}", registry.counts()))
        })?;
    }
    Ok(())
}

#[test]
fn mdmain_plain_and_thermostat() -> MoptResult<()> {
    for (kind, thermostat) in [("nve", "nve"), ("langevin", "langevin")] {
        same_across_threads(&format!("md_{kind}"), |base, threads| {
            let input = write(base, "ar64.xyz", &argon_cluster(4))?;
            let config = mdmain::config_from_cli(&[
                &input,
                "-os",
                "lj",
                "-mt",
                thermostat,
                "-time",
                "100",
                "-ts",
                "2fs",
                "-temp",
                "40",
                "-md_opt",
                "seed=3",
                "write_every=5",
                "structure_every=20",
                "-pc",
                "bond",
                "1,2",
                "-core",
                threads,
            ])?;
            let registry = CalculatorRegistry::native();
            let out = mdmain::run_with(&config, base, None, &registry)?;
            Ok(format!("{out:?}\n{:?}", registry.counts()))
        })?;
    }
    Ok(())
}

#[test]
fn confsearch_with_a_seed() -> MoptResult<()> {
    same_across_threads("confsearch", |base, threads| {
        let input = write(base, "butane.xyz", BUTANE)?;
        let config = confsearch::config_from_cli(&[
            &input, "-os", "gfnff", "-ms", "3", "-cs_opt", "seed=3", "-core", threads,
        ])?;
        let registry = CalculatorRegistry::native();
        let out = confsearch::run_with(&config, base, None, &registry)?;
        Ok(format!("{out:?}\n{:?}", registry.counts()))
    })
}

#[test]
fn orientsearch_with_a_seed() -> MoptResult<()> {
    same_across_threads("orientsearch", |base, threads| {
        let water = [
            ("O", [0.0, 0.0, 0.117]),
            ("H", [0.0, 0.757, -0.469]),
            ("H", [0.0, -0.757, -0.469]),
        ];
        let mut text = "6\nwaters\n".to_string();
        for k in 0..2 {
            for (s, p) in water {
                text.push_str(&format!(
                    "{s} {:.6} {:.6} {:.6}\n",
                    p[0] + 4.0 * f64::from(k),
                    p[1],
                    p[2]
                ));
            }
        }
        let input = write(base, "w2.xyz", &text)?;
        let config = orientsearch::config_from_cli(&[
            &input,
            "-os",
            "gfnff",
            "-nsample",
            "4",
            "-orient_opt",
            "seed=7",
            "-core",
            threads,
        ])?;
        let registry = CalculatorRegistry::native();
        let out = orientsearch::run_with(&config, base, None, &registry)?;
        Ok(format!("{out:?}\n{:?}", registry.counts()))
    })
}

#[test]
fn relaxedscan_of_a_bond() -> MoptResult<()> {
    same_across_threads("relaxedscan", |base, threads| {
        let input = write(base, "ethanol.xyz", ETHANOL)?;
        let config = relaxedscan::config_from_cli(&[
            &input,
            "-os",
            "gfnff",
            "-scan",
            "bond",
            "2,3",
            "1.35,1.55",
            "-nsample",
            "3",
            "-kp",
            "0.5",
            "1.2",
            "1,5",
            "-core",
            threads,
        ])?;
        let registry = CalculatorRegistry::native();
        let out = relaxedscan::run_with(&config, base, None, &registry)?;
        Ok(format!("{out:?}\n{:?}", registry.counts()))
    })
}

#[test]
fn oniom_optimization_with_microiterations() -> MoptResult<()> {
    same_across_threads("oniom", |base, threads| {
        let input = write(base, "ethanol.xyz", ETHANOL)?;
        let config = optmain::config_from_cli(&[
            &input, "-os", "gfn0xtb", "-oniom", "2-4,8,9", "auto", "gfnff", "-ns", "8", "-core",
            threads,
        ])?;
        let registry = CalculatorRegistry::native();
        let out = optmain::run_with(&config, base, None, &registry)?;
        Ok(format!("{out:?}\n{:?}", registry.counts()))
    })
}

/// Copper with a vacancy in a 2x2x2 cell of the fcc conventional cell (31 atoms), before and
/// after a nearest neighbour moves into it.
fn copper_vacancy() -> String {
    let a = 3.615;
    let mut sites = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for b in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    sites.push([
                        a * (f64::from(i) + b[0]),
                        a * (f64::from(j) + b[1]),
                        a * (f64::from(k) + b[2]),
                    ]);
                }
            }
        }
    }
    let start: Vec<[f64; 3]> = sites[1..].to_vec();
    let mut end = start.clone();
    end[0] = [0.0, 0.0, 0.0];
    let lattice = format!(
        "Lattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"",
        l = 2.0 * a
    );
    let mut text = String::new();
    for frame in [&start, &end] {
        text.push_str(&format!("{}\n{lattice}\n", frame.len()));
        for x in frame {
            text.push_str(&format!("Cu {:.6} {:.6} {:.6}\n", x[0], x[1], x[2]));
        }
    }
    text
}

#[test]
fn autots_of_a_vacancy_migration() -> MoptResult<()> {
    same_across_threads("autots", |base, threads| {
        let settings = write(
            base,
            "config.json",
            &r#"{
  "work_dir": "work",
  "top_n_candidates": 1,
  "step2_settings": {"othersoft": "eam", "NSTEP": 10},
  "step3_settings": {"othersoft": "eam", "calc_exact_hess": 5, "NSTEP": 15,
                     "tight_convergence_criteria": true},
  "step4_settings": {"othersoft": "eam", "calc_exact_hess": 10,
                     "tight_convergence_criteria": true,
                     "intrinsic_reaction_coordinates": ["0.3", "30", "lqa"]}
}"#
                // The threads of the run are those of its steps.
                .replace("\"othersoft\": \"eam\",", &format!("\"othersoft\": \"eam\", \"N_THREAD\": {threads},")),
        )?;
        let input = write(base, "vacancy.xyz", &copper_vacancy())?;
        let args = [
            input.as_str(),
            "-cfg",
            settings.as_str(),
            "--skip_step1",
            "--run_step4",
        ];
        let parsed = parser("run_autots", spec::RUN_AUTOTS).parse(args)?;
        let config = autots::AutotsConfig::from_args(&parsed)?;
        let registry = CalculatorRegistry::native();
        let out = autots::run_with(&config, base, None, &registry)?;
        Ok(format!("{out:?}\n{:?}", registry.counts()))
    })
}

/// LJ7 near its lowest minimum (Bohr).
const AR7: [[f64; 3]; 7] = [
    [6.023853759546511, 0.0, -0.15549746403767273],
    [1.962006769746685, 5.809518090796886, -0.061011157764781325],
    [-4.854674003698997, 3.590479638369873, 0.03347514850811009],
    [-4.999838497421851, -3.5904796383698727, 0.12796145478100152],
    [1.6599768546465556, -5.809518090796886, 0.2224477610538929],
    [-0.08463505395523435, 0.0, 3.794030138169188],
    [0.2933101711363313, 0.0, -3.961405880709738],
];

const MAPPER_CONFIG: &str = r#"{
  "mapper_settings": {"afir_gamma_kJmol": 2.0, "dist_upper_ang": 7.0, "rng_seed": 42},
  "top_n_candidates": 2,
  "step1_settings": {"othersoft": "lj", "NSTEP": 300},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 200},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "calc_exact_hess": 1,
    "NSTEP": 400,
    "intrinsic_reaction_coordinates": ["0.5", "200", "lqa"]
  }
}"#;

/// The mapper with the batch schedule: with one worker and with four, every run is the same
/// for any number of threads; and the network, the journal and the files of the jobs of four
/// workers are those of one (the log and the report of the run name the number of workers, so
/// they are compared across the threads only). The rolling schedule hands the jobs out as they
/// end, so its network depends on the timing by design and is not compared.
#[test]
fn mapper_batch_schedule_with_one_and_four_workers() -> MoptResult<()> {
    let run_with_workers = |workers: &str| {
        let count = format!("workers={workers}");
        outcome_across_threads(&format!("mapper_{workers}"), |base, threads| {
            let mut text = format!("{}\n0 1\n", AR7.len());
            for p in AR7 {
                text.push_str(&format!(
                    "Ar {:.10} {:.10} {:.10}\n",
                    p[0] * Units::BOHR_TO_ANGSTROM,
                    p[1] * Units::BOHR_TO_ANGSTROM,
                    p[2] * Units::BOHR_TO_ANGSTROM
                ));
            }
            let input = write(base, "ar7.xyz", &text)?;
            // The threads of the run are those of its steps.
            let settings = write(
                base,
                "config.json",
                &MAPPER_CONFIG.replace(
                    "\"NSTEP\": 300",
                    &format!("\"NSTEP\": 300, \"N_THREAD\": {threads}"),
                ),
            )?;
            let out = base.join("map").to_string_lossy().into_owned();
            let args = [
                input.as_str(),
                "-cfg",
                settings.as_str(),
                "--max_iter",
                "4",
                "--output_dir",
                out.as_str(),
                "-map_opt",
                count.as_str(),
                "batch_size=4",
                "schedule=batch",
            ];
            let parsed = parser("run_mapper", spec::RUN_MAPPER).parse(args)?;
            let config = mapper::MapperConfig::from_args(&parsed, None)?;
            let registry = CalculatorRegistry::native();
            let run = mapper::run_with(&config, base, None, &registry)?;
            Ok(format!("{:?}\n{:?}", run.network, registry.counts()))
        })
    };
    let one = run_with_workers("1")?;
    let four = run_with_workers("4")?;
    same_outcome("mapper", "4 workers", &one, &four, &|name| {
        !name.ends_with("mopt.log") && !name.ends_with(".html") && name != "map/settings.json"
    })
}
