# Architecture: what is where, and how a run flows

This page is a map of the code for people who read or change it. The code is in two places:

- `crates/mopt-rs/` is the Rust library and the `mopt` binary: everything numerical.
- `python/mopt_rs/` with `crates/mopt-py/` (PyO3) is the Python package: the calculators that
  run Python programs, and Python access to the Rust parts. The bindings crate `mopt-py` (not
  published to a crate registry) builds the Python distribution of the same name, which is
  imported as `mopt_rs` and installs the commands `mopt-py`, `optmain`, ....

Every module header names its MultiOptPy counterpart (`MultiOptPy: ...`), and
[coverage.md](coverage.md) lists the same correspondence from the MultiOptPy side.

## Layers

Lower layers never use higher ones. A module may use any module below it.

```text
 L6  cli/            command lines and settings files -> typed configs -> jobs
       optmain.rs  settings.rs  calculator.rs  charge_spin.rs  constraints.rs  bias.rs
       variables.rs (optmain on plain variables: variables files, CSV output)
       ircmain.rs  irc.rs (the IRC of ircmain and of optmain -irc)
       mdmain.rs  mdmain_settings.rs  mdmain_umbrella.rs  mdmain_analysis.rs (molecular
       dynamics, its trajectory analysis)
       orientsearch.rs  orientsearch_settings.rs  orientsearch_output.rs (orientations of the
       parts of a complex: every sample an optmain run)

       confsearch.rs  confsearch_settings.rs  confsearch_jobs.rs (optimizations, dynamics,
       Hessians, IRCs on the surface of the command)  confsearch_files.rs (MultiOptPy's files,
       the ensemble)
       relaxedscan.rs (flow, restart, repair of jumps)  relaxedscan_settings.rs  _targets.rs
       (the scanned coordinates)  _scan.rs (one point: start, constraints, pre-image)
       _points.rs (a point with optmain or the optimizer of plain variables)  _resper.rs
       _outputs.rs  _followup.rs (optmain -order 1 from the candidates)  _report.rs (the page)
       oniom_micro.rs (ONIOM microiterations of optmain, ircmain, relaxedscan; constraints
       renumbered to the core)
       nebmain.rs  nebmain_settings.rs  nebmain_dmf.rs  nebmain_followup.rs (chains of
       images, DMF, optmain -order 1 from the maxima)  bitss.rs (optmain -mf bitss)
       ieipmain*.rs (iEIP and the other two-point methods by name, their states and paths)
       freqmain.rs (vibrational analysis, thermochemistry, VPT2, VSCF, VTST, URVA, phonons,
       -diabatize: diabatic_path.rs)  orbital_path.rs (orbitals along a path)
       autots*.rs (run_autots: the workflows of structures and of plain variables, their page)
       mapper*.rs (run_mapper: settings, jobs, minima, states, remap, merge, kinetics, the page)
       run_kinetics.rs (run_kinetics and run_rcmc)  automecp*.rs (run_automecp, its page)
       etmain.rs (electron transfer: EDEEL within a molecule, charge hopping)
       pair_settings.rs (nebmain's and BITSS's options for the registry of two-point methods)
       variable_cell.rs (-vc)  follow.rs (-follow: the direction a TS search starts along)
       oniom.rs (the ONIOM layers and their options)
       option.rs + spec.rs (generated option tables)   mod.rs (dispatch, exit codes)
     mcp/            the MCP server (`mopt mcp`): tools from the option tables, jobs, files
 ----------------------------------------------------------------------------------------
 L5  workflow/       drive the state machines with calculators; observers write output
       driver.rs (loop)  observers.rs (log, CSV, XYZ, plots, STOP file)
       journal.rs (evaluation journal, restart)
       explore/ (the exploration core: priority.rs choice of the next minimum, stop rules,
       identity.rs same minimum up to exchanged atoms and molecules, pool.rs distinct
       minima, the ensemble with duplicates and rotamers, crossing of conformers, ADDF,
       orient/ the orientation search)
       mapper/ (run_mapper: the network, its jobs and stores; the rules are those of
       mapper_design.md)  kinetics.rs (run_kinetics, run_rcmc)  kinetics_beyond.rs (the
       annotations, outputs, and MD-derived semi-Markov model of the models beyond the
       canonical one)  kinetics_dynamics.rs (trajectory pairs and the friction kernel at a
       saddle point for branching fractions, reactive flux, and Grote-Hynes)
       pair_search/ (the registry of two-point methods; the options of the members that are
       commands, nebmain's and BITSS's, come from cli with each run: SearchRun::commands)
       chain*.rs dmf.rs (the chain methods of nebmain)
       electron_transfer.rs (etmain -et_method edeel)  hopping.rs (etmain -et_method hopping)
       confsearch/ (the conformation search: MultiOptPy's AFIR sampling, iMTD-GC, the
       conformers of transition states; jobs and output through traits)
       irc.rs (IRC driver with observers)  irc_output.rs + irc_combined.rs (its files, log)
       md.rs (MD driver, surfaces, observers)  md_files.rs + md_log.rs  umbrella.rs
       md_analysis.rs + md_analysis_output.rs (trajectory analysis of a run: files)
       scan/ (the points of a scan and their order, rigid structures, stationary points of
       a scanned surface, jumps between branches, state and tables, figures)
     analysis/       results after a run (the rate formulas of electron transfer are in
                     kinetics/rate/, re-exported here as analysis::electron_transfer)
       orbitals/ (natural reaction orbitals, ROET, orbitals followed along a path)
       koopman.rs  diabatic.rs (diabatization)  free_energy.rs  disconnectivity.rs
       path_map.rs (classical multidimensional scaling, PCA)  resper/ (ReSPer coordinates
       of new structures by Gower's formula, their gradients and pre-images)
       trajectory/ (MD trajectories: statistics, RDF, MSD, VACF and VDOS, RMSD, PCA)
       correlation/ (time correlations, spectra, Green-Kubo and reverse NEMD transport)
 ----------------------------------------------------------------------------------------
 L4  opt/            optimizers as ask/tell state machines (no I/O)
       state_machine.rs  registry.rs (names -> methods)  convergence.rs  trust.rs
       vector.rs (the same methods on n plain variables, no atoms)  vector_constraints.rs
       (their equality constraints: fixed variables, ReSPer coordinates)
       step/ (one file per method: RS-I-RFO, P-RFO, CCQN, FIRE, L-BFGS, CG, Adam, GPMin, ...)
       update/ (BFGS, Bofill, ..., block)  enhance/ (LARS, DIIS family, line search, TRIM, ...)
       stepper.rs (one method used on its own)  constrained.rs  stationary.rs
       micro.rs (the methods of the registry for the relaxations of pes/microiter: cli hands
       them in with the settings, MicroSettings::methods)
     irc/            intrinsic reaction coordinate as an ask/tell state machine
       machine.rs  start.rs  registry.rs (names -> integrators)  space.rs (mass weighting)
       lqa.rs  hpc.rs + dwi.rs  dvv.rs  euler.rs  rk4.rs  gs2.rs  modekill.rs
       sphere.rs (minimization on a hypersphere)  first_step.rs  history.rs
     md/             molecular dynamics as an ask/tell state machine
       machine.rs  integrator.rs (splittings)  thermostat/  barostat/  space.rs (masses)
       constraints.rs (SHAKE, RATTLE)  freeze.rs (frozen modes)  cv.rs (variables, umbrella)
     path/           methods on pairs and chains of images as ask/tell state machines
       band/ (NEB and its relatives)  dmf/  gsm/  fsm.rs  dsafir/  bitss/  dimer/  ieip/
       hisd/ (high-index saddle dynamics)  interp/ (first paths)  shs.rs  twopshs.rs ...
     vib/            normal modes and thermochemistry (pure functions, as kinetics needs
                     them: kinetics_beyond_tst.md 5.1), seam vibrations, VPT2, VSCF, VTST,
                     URVA, phonons, local modes
     kinetics/       no I/O: networks, rate theories and corrections with their registries,
                     models, solvers (kinetics_beyond_tst.md 5.1); rate/ also holds the
                     formulas of electron transfer (Marcus, Zhu-Nakamura, inner sphere, SAES,
                     couplings, the transfer integral of the projection method, PCET);
                     rrkm.rs egme.rs hot.rs semi_markov.rs msm.rs branching.rs refine.rs
                     order.rs (RRKM counts, energy grains, hot intermediates, memory, Markov
                     state models, trajectory statistics, refinement tasks, the orders of the
                     steps); solve/kmc.rs solve/contract_mass.rs solve/windows.rs (Gillespie,
                     the contraction of mass action, pseudo-first-order windows)
 ----------------------------------------------------------------------------------------
 L3  pes/            potential-energy surfaces
       calculator.rs (the Calculator trait)  registry.rs (names -> calculators)
       native/ (LJ, EMT, EAM, MEAM, Tersoff, Muller-Brown, GFN0-xTB, GFN1-xTB with its TDA,
       GFN-FF)  dispersion/ (D3(BJ), D4)
       bias/ (AFIR and every bias potential)  combine/ (MECI, MECP, model functions, EDEEL)
       oniom/ (N-layer ONIOM, link atoms, embedding, microiterations)  point_charges.rs
       states.rs + followed.rs (excited states, followed from structure to structure)
       o1numhess/ (Hessians from further gradients)  stress.rs  replay.rs + counting.rs (the
       journal answers, the counts of evaluations)
       constraint.rs  microiter/ (ONIOM-type inner relaxation; the methods of the optimizer
       registry come from opt/micro.rs through its settings)  term.rs  term_check.rs
       objective.rs (functions of plain variables: the Objective trait, mb2d, expr:...)
     model_hessian/  model Hessians (published versions; legacy/ = MultiOptPy's as they are)
 ----------------------------------------------------------------------------------------
 L2  math/           ad/ (dual and hyper-dual numbers, local AD, minimum image)
                     linalg.rs  projection.rs (rigid-body modes)  stable.rs  roots.rs
                     line_search.rs  minimize.rs  gp.rs  simplex_qp.rs  assignment.rs  mlp.rs
                     fft.rs (Fourier transforms, autocorrelations)  special.rs (Airy)
     coord/          primitives (bond, angle, dihedral, out-of-plane)  connectivity
                     graph.rs (bond graphs: classes, isomorphism, molecules in a cell)
                     invariant_rmsd.rs  zmatrix.rs
                     internal coordinates  system/ (Cartesian, RIC, DIC, TRIC, transforms)
 ----------------------------------------------------------------------------------------
 L1  domain/         Element, MolecularSystem, SpinCharge, PeriodicCell, units, Quantity<U>,
                     AtomId, EvalRequest / EvalOutput, trajectories
     parameters/     element tables (radii, masses, D3/D4 references, GFN0-xTB, MultiOptPy's)
     io/             XYZ and other structure formats, CSV, software_path.conf
 ----------------------------------------------------------------------------------------
 L0  error.rs        MoptError with context        core_infra/  argument parser, JSON,
                                                    logger, clock, RNG, viz/ (SVG, PNG),
                                                    memory (the budget of dense matrices)
```

The Python side sits next to L3-L6:

```text
 python/mopt_rs/
   cli.py                 pip entry points (optmain, ...) -> Rust commands with the Python registry
   optimize.py            Optimizer (ask/tell, Rust), run(), Checkpoint
   workflows.py           chain relaxation, microiterations, state following and mapper networks
   variables.py           optimize_variables(), VectorOptimizer (plain variables, no atoms)
   orient.py              orientation_search() of a complex with any calculator
   calculators/
     base.py              Engine (a program) -> EngineCalculator (units, finite-difference
                          Hessians, cache, electron checks)
     registry.py          names -> engines: pyscf, psi4, tblite, dxtb, ase:..., profiles, plugins
     ase.py               any ASE calculator with no code (state keywords, Hessian route)
     pyscf.py psi4.py xtb.py bridges.py profiles.py program_options.py provenance.py
     programs.py          programs run through files (Gaussian, ORCA, ...): the table of
                          their settings and the engine; program_inputs.py builds their
                          ASE calculators and reads their Hessians
     option_files.py      defaults of a calculator kept in files, values read from files
     _bridge.py           what the Rust commands call to use Python engines
 crates/mopt-py/src/
   lib.rs                 the _core module: Optimizer, components, run_cli
   pycalc.rs              a Python engine behind the Rust Calculator trait, including state tracking and embedding
   chain.rs               Python callbacks for the Rust band and environment relaxation drivers
   network.rs             mapper methods, priorities and stores through the Rust registries
   components.rs callback.rs   standalone parts; Python callbacks as surface terms
   parts.rs steps.rs      coordinate systems, superposition, bond graph, priorities; step methods
   orient.rs              OrientationSearch (ask/tell), compare_structures, complex_parts
   variables.rs           VectorOptimizer and the built-in objectives
```

## How an `optmain` run flows

```text
 command line / --config file
        |  cli::settings (defaults < file < command line)   cli::option + spec (flags)
        v
 OptmainConfig  ------------------------------------------------------------------+
        |  cli::calculator: name + CalculatorOptions                               |
        v                                                                          |
 CalculatorRegistry.create(name, options, structure)                               |
        |  Rust names (lj, gfn0xtb, ...) -> pes::native                            |
        |  other names -> Python fallback (pip commands / mopt-py): pycalc -> _bridge|
        v                                                                          |
 Calculator  --wrapped by--> workflow::journal (records every evaluation)          |
        |                                                                          |
        v                                                                          |
 workflow::driver loop:                                                            |
     state machine (opt/) --ask--> EvalRequest(x) --> pes surface:                 |
                                         calculator + bias + constraints + combine |
                         <--tell-- energy, gradient, Hessian                       |
     observers (log, CSV, XYZ, plots, STOP) see every step  <---------------------+
        |
        v
 run directory: mopt.log, settings.json, evaluations.journal, *_traj.xyz,
                *_optimized.xyz, energy_profile.csv, plots
```

- Restart (`--restart RUN_DIR`) rebuilds the same configuration from `settings.json` and
  answers evaluations from the journal until the requested structures differ.
- The optimizer never sees the calculator: it asks for evaluations at Cartesian structures and
  is told the results. Coordinate systems (`coord/system`) convert gradients and Hessians on
  the optimizer side, so every calculator works with every coordinate system.

## How a calculation reaches a program

```text
 "-os NAME" ---> CalculatorRegistry (Rust) --- native? ---> pes::native::*
                         |
                         +-- otherwise (Python available) --> mopt_rs.calculators.registry
                                 pyscf / psi4 / tblite / dxtb engines
                                 ase:module:Class, profiles -> AseEngine (any ASE calculator)
                                 multioptpy:NAME (an installed MultiOptPy)
                                 entry-point plugins
 Every engine: charge and multiplicity 2S+1 (or -unpaired 2S) converted to the program's
 convention; program-specific settings through -calc_opt (keywords, prefix.path=value, hooks);
 Hessians from the program or central differences; the run log names the program's origin.
```

## Where to change things

| To add or change | Look at |
|---|---|
| a command-line option | `tools/cli/gen_cli.py` (regenerates `cli/spec.rs` and `cli_reference.md`), then the command in `cli/` |
| an optimizer | `opt/step/` (one file per method), registered in `opt/registry.rs` |
| a step enhancer | `opt/enhance/` (one file per method), registered in `opt/registry.rs` |
| an IRC integrator | `irc/` (one file per method), registered in `irc/registry.rs` |
| an MD integrator, thermostat, or barostat | `md/integrator.rs`, `md/thermostat/`, `md/barostat/` (a trait each, with its registry) |
| a collective variable | `md/cv.rs` (or any `PesTerm` through `Cv::from_term`) |
| a kind of scanned coordinate (`relaxedscan -scan`) | `cli/relaxedscan_targets.rs` (`TargetKind`), held by a constraint of `pes/constraint.rs` (structures) or `opt/vector_constraints.rs` (plain variables) |
| a Hessian update | `opt/update/formulas.rs`, `opt/update/block.rs` |
| a bias potential | `pes/bias/` (one file per kind; `term_check.rs` tests it) |
| a choice of the next minimum, a stop rule | `workflow/explore/priority*.rs` (`SamplingPriority`, `PriorityRegistry`), `workflow/explore/stop.rs` (`StopRule`) |
| a strategy of the conformation search | `workflow/confsearch/` (it uses `SearchJobs` and reports to a `SearchObserver`) |
| a rule of the reaction network (a kind of node or edge, an entrance, a default) | `mapper_design.md` first, then `workflow/mapper/` |
| a way of exploring a minimum, a priority, a stop rule of `run_mapper` | `workflow/mapper/method.rs` (`MethodRegistry`), `priority.rs`, `stop.rs` ([mapper.md](mapper.md), "Adding parts in Rust") |
| a two-point method (for `ieipmain`, `run_autots`, `run_mapper`) | `workflow/pair_search/registry*.rs` (`PairMethodRegistry`) |
| a rate theory, a correction, a kinetic model or solver | `kinetics/rate/`, `kinetics/model.rs`, `kinetics/solve/` (registries; [kinetics.md](kinetics.md), the last section) |
| a Rust calculator | `pes/native/`, registered in `pes/registry.rs` |
| an objective of plain variables | `pes/objective.rs` (`ObjectiveRegistry::builtin`) |
| a Python program | `python/mopt_rs/calculators/` (an `Engine`, or nothing at all for ASE calculators) |
| a model Hessian | `model_hessian/` (published) or `model_hessian/legacy/` (MultiOptPy's) |
| parameter tables | `tools/params/` generators (never edit generated tables by hand) |
| reference data for tests | `tools/golden/` (MultiOptPy, gfn0-rs, xtb) -> `tests/golden/` |
| what Python reaches of the Rust library | `tools/api/parity.json` (checked by `tests/test_api_parity.py`, `crates/mopt-rs/tests/api_parity.rs`) |
| an ASE program by name | a profile file (see running.md), or `python/mopt_rs/calculators/profiles.py` |
| MultiOptPy coverage | `tools/coverage/mapping.json` (+ `report.py` regenerates `coverage.md`) |

## Conventions that tie the code to MultiOptPy

- Every source file names its MultiOptPy counterpart in its header: `MultiOptPy:
  `Optimizer/fire.py:FIRE`.`, or `MultiOptPy: no counterpart (new in mopt-rs).` Generated
  files get the line from their generators.
- Every source file says which package it belongs to (`This file is part of mopt-rs.` or
  `mopt-py`); `tools/headers/add_headers.py` adds the line to new files, and
  `tests/test_headers.py` checks both header lines.
- MultiOptPy's numerical safeguards are taken over when they are sound and replaced when a
  better formulation exists; every difference is listed in [migration.md](migration.md)
  ("Numerical safeguards").

## Tests

- Rust: unit tests next to the code, integration tests in `crates/mopt-rs/tests/`
  (golden comparisons with MultiOptPy, gfn0-rs, and xtb, command-line runs, restart).
- Python: `tests/` (`tests/fake_ase.py` stands in for ASE when it is not installed).
- Everything is run locally: `cargo fmt`, `cargo clippy -D warnings`, `cargo test`, the
  Python tests in a plain environment and in one with PySCF, ASE, tblite, and the Rust engines,
  `ruff`, and `mypy`.
