# mopt-rs

**mopt-rs is the successor implementation of [MultiOptPy](https://github.com/ss0832/MultiOptPy).**
It re-implements MultiOptPy (v1.21.2) in Rust with a modular, plugin-oriented architecture and
exposes it to Python through PyO3.

> Status: `2.1.0`. Which MultiOptPy features are
> implemented is listed in `docs/source/coverage.md` and, per option, in
> `docs/source/cli_reference.md`; what is known not to work is in
> `docs/source/open_issues.md`.

## Highlights

- Pure safe Rust core (`#![forbid(unsafe_code)]`); external crates limited to `faer`, `rayon`,
  `libm`, and `pyo3`.
- No PyTorch dependency: exact derivatives through forward-mode automatic differentiation
  (`Dual` / `HyperDual`) applied per term.
- Pluggable step generators, Hessian updaters, step enhancers, and coordinate systems
  (Cartesian, redundant internal, delocalized internal, or user-defined).
- Optimizers are ask/tell state machines, so they can be driven either by Rust calculators or by
  Python calculators (PySCF, tblite, ASE, ...).
- Periodic boundary conditions through minimum-image displacements throughout.
- Built-in engines: GFN0-xTB, GFN1-xTB (analytic Hessians, excited states by the TDA), and
  GFN-FF, ports checked against xtb and tblite, with D3(BJ) and D4, and model potentials (LJ,
  EMT, EAM, MEAM, Tersoff); N-layer ONIOM with microiterations.
- Beyond MultiOptPy: the reaction-network mapper with kinetics beyond transition-state theory
  (`run_mapper`, `run_kinetics`, `docs/source/mapper.md`, `docs/source/kinetics.md`), searches
  of crossing points (`run_automecp`), vibrational analysis and thermochemistry (`freqmain`),
  and the rate constants of electron transfer (`etmain`, `docs/source/electron_transfer.md`).

## Quick start

### Install

```bash
pip install .
```

builds the Rust core with maturin (a Rust toolchain is needed) and installs the Python
package. The distribution is called `mopt-py` and is imported as `mopt_rs`
(`import mopt_rs`); it comes with the commands `optmain`, `nebmain`, `ieipmain`, `mdmain`,
`confsearch`, `relaxedscan`, `orientsearch`, `freqmain`, `ircmain`, `etmain`, `run_autots`,
`run_automecp`, `run_mapper`, `run_kinetics`, and `run_rcmc`, `mopt-py COMMAND ...` for any
of them, and `mopt-mcp` (the MCP server). The
engines built into mopt-rs need nothing else: `-os gfn1xtb`, `-os gfn0xtb`, and `-os gfnff`
(GFN1-xTB, GFN0-xTB, and GFN-FF for H to Rn), and the model potentials
(`lj`, `emt`, `eam`, `meam`, `tersoff`). The engines of other packages are optional: none
is installed with mopt-rs, and each is imported only when a calculation uses it. Install the
ones you need: `pip install ".[pyscf]"` (`-pyscf`), `".[tblite]"` (`-xtb`), `".[dxtb]"`
(`-dxtb`, with PyTorch), `".[ase]"` (`-os ase:module:Class`), or `".[qm]"` for PySCF, tblite
and ASE together. An engine whose package is missing says so when the run starts, with the
command that installs it, and

```bash
python -m mopt_rs.calculators
```

lists every engine with its package: the version installed, or the command to install it.

Without Python, `cargo build --release -p mopt-rs` makes `target/release/mopt`, which runs
the same commands with the built-in engines: `mopt optmain ...`.

### The examples of MultiOptPy, as they run here

The inputs of the examples in MultiOptPy's README are in `examples/quickstart/`:

```bash
cd examples/quickstart
```

Every command below was run as it is written here. They are MultiOptPy's command lines
with these changes:

- The scripts are commands: `optmain ...` for `python optmain.py ...`.
- `-spin` is the spin multiplicity `2S + 1` for every engine. MultiOptPy's README writes
  `-spin 0` with `-pyscf` (PySCF counts unpaired electrons); here that is `-spin 1`,
  `-unpaired 0`, or nothing (the default is a singlet). `-spin 0` is refused with a
  message that says so.
- A line `charge multiplicity` in a structure file (the `-1 1` in `SN2.xyz`) gives the
  state when the command line does not; MultiOptPy's files without a first line with the
  number of atoms (the directory `aldol_rxn/`) are read as MultiOptPy reads them.
- `-xtb GFN2-xTB` and `-pyscf` work as before when tblite and PySCF are installed. With
  `-os gfn1xtb` in their place the same commands run on the engine built into mopt-rs.

Minimum with an artificial force (AFIR), the SN2 reaction of fluoromethane with hydroxide
and an aldol reaction:

```bash
optmain SN2.xyz -ma 150 1 6 -pyscf -elec -1 -spin 1 -opt rsirfo_block_fsb -modelhess
optmain SN2.xyz -ma 150 1 6 -os gfn1xtb -opt rsirfo_block_fsb -modelhess
optmain aldol_rxn.xyz -ma 95 1 5 50 3 11 -xtb GFN2-xTB -opt rsirfo_block_fsb -modelhess
```

Transition state from a guess (add `-freq` for the vibrational analysis):

```bash
optmain aldol_rxn_PT.xyz -xtb GFN2-xTB -opt rsirfo_block_bofill -order 1 -fc 5
```

Path between the structures of a directory (`-qsmv2` and `-qsm` work as well):

```bash
nebmain aldol_rxn -xtb GFN2-xTB -ns 50 -adpred 1 -nd 0.5
```

Transition state between two structures (iEIP):

```bash
ieipmain ieip_test -xtb GFN2-xTB
```

Molecular dynamics (Nose-Hoover chain, 100000 steps of 0.5 fs):

```bash
mdmain aldol_rxn_PT.xyz -xtb GFN2-xTB -temp 298 -traj 1 -time 100000
```

Orientations of two molecules, conformers, and a relaxed scan:

```bash
orientsearch aldol_rxn.xyz -part 1-4 -ma 95 1 5 50 3 11 -nsample 5 -xtb GFN2-xTB -opt rsirfo_block_fsb -modelhess
confsearch s8_for_confomation_search_test.xyz -xtb GFN2-xTB -ns 2000
relaxedscan SN2.xyz -nsample 8 -scan bond 1,2 1.3,2.6 -xtb GFN2-xTB -opt crsirfo_block_fsb -modelhess
```

The workflow from a reactant complex to transition states and their paths (AFIR, path,
transition-state optimization, IRC), with MultiOptPy's settings file and with the same
settings on the built-in engine; the result is `autots_report.html` in the work directory:

```bash
run_autots aldol_rxn.xyz -cfg config_autots_run_xtb_test.json
run_autots aldol_rxn.xyz -cfg config_autots_builtin.json
```

`run_mapper` maps a reaction network from one or more structures with MultiOptPy's settings
file (AFIR pushes between pairs of atoms, transition states, IRCs); the network is in
`reaction_network.json` of its run directory, with `mapper_report.html`. It also takes a
network to another surface (`-remap`, `-single_points`), searches the crossing points of
several electronic states (`states`), and solves the kinetics of the network (`kinetics=on`,
or `run_kinetics` afterwards), on ONIOM surfaces (a core relaxed in its environment) and in
periodic cells, fixed or variable, with pushes between molecules and on the cell for molecular
crystals (`docs/source/mapper.md`). What each command writes, the settings behind the options,
and what was observed in these runs are in `docs/source/quickstart.md`; every option is listed
in `docs/source/cli_reference.md`.

## Agents (MCP)

`python -P -m mopt_rs.mcp --root DIR` (or `mopt mcp --root DIR` for the Rust binary) is a
Model Context Protocol server over stdio: every implemented command is a tool that starts a
job (or a task) under `DIR/mcp_jobs/`, with tools to follow, wait for, stop, cancel, continue
and read the jobs, tools for components that answer at once (calculators, single points, model
Hessians, normal modes, coordinates, first images, superposition, bond graphs, dispersion,
Hessian updates), and the documentation, run directories and jobs as resources that can be
subscribed to. Nothing outside `DIR` is read or written for a request, and
settings that start other programs or load code are refused unless the server runs with
`--allow-external`. See `docs/source/mcp.md`.

## Building

The Rust workspace builds with Cargo:

```bash
cargo build --release --workspace
```

The Python package is built with maturin (the distribution `mopt-py`, imported as
`mopt_rs`):

```bash
pip install .
```

## Documentation

The documents are the Markdown files of `docs/source`. They are installed with the Python
package (`python -m mopt_rs.docs` lists them, `python -m mopt_rs.docs running` prints one)
and build as a web site with Sphinx:

```bash
python -m pip install -r docs/requirements.txt
```

```bash
python -m sphinx -b html docs/source docs/build/html
```

Two sites are set up from the same sources: Read the Docs builds from `.readthedocs.yaml`
once the repository is imported there, and `.github/workflows/docs.yml` publishes to
GitHub Pages on every push to the default branch (in the repository settings, Pages >
Source must be "GitHub Actions").

## Relationship to MultiOptPy

mopt-rs keeps MultiOptPy's command-line entry points (`optmain`, `nebmain`, `ieipmain`, ...) and
their options, and reproduces its behavior. Places where mopt-rs intentionally differs (bug fixes
and corrected algorithms) are listed in `docs/source/migration.md`. A map of the code (layers,
how a run flows, where to change what) is in `docs/source/architecture.md`.

MultiOptPy, the predecessor of mopt-rs, is archived on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18811289.svg)](https://doi.org/10.5281/zenodo.18811289)
([10.5281/zenodo.18811289](https://doi.org/10.5281/zenodo.18811289)). Results obtained with
MultiOptPy, or with mopt-rs where they build on MultiOptPy's methods, can cite it as

```bibtex
@software{ss0832_multioptpy_2025,
  author    = {ss0832},
  title     = {MultiOptPy: Multifunctional geometry optimization tools for quantum chemical calculations},
  year      = 2026,
  publisher = {Zenodo},
  version   = {v1.21.1},
  doi       = {10.5281/zenodo.18811289},
  url       = {https://doi.org/10.5281/zenodo.18811289}
}
```

```
ss0832. (2026). MultiOptPy: Multifunctional geometry optimization tools for quantum chemical
calculations (v1.21.1). Zenodo. https://doi.org/10.5281/zenodo.18811289
```

## Generative AI used in development

mopt-rs was developed using generative AI, primarily Claude Opus 5.5 and Claude Fable 5.2 (both from Anthropic) and GPT-6.2 Astra (from OpenAI).

## License

GPL-3.0-or-later. Copyright (C) 2023-2026 ss0832.
