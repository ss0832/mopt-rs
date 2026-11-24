# Differences from MultiOptPy

mopt-rs is the successor implementation of MultiOptPy (v1.21.2). It reproduces MultiOptPy's
behavior except where listed here. Every entry has a regression test in the Rust test suite.

To edit both code bases side by side: the header of every Rust and Python module has a
`MultiOptPy:` line naming the MultiOptPy file, class, and functions it corresponds to (or
"no counterpart (new in mopt-rs)"), and ported functions name their MultiOptPy function in
their documentation (for example `MultiOptPy: GFNFFApproxHessian.gfnff_bond_hessian`).
`docs/source/coverage.md` lists the same correspondence from the MultiOptPy side.

The differences are on the pages below, by topic; this page has the notes on the command line
and the methods that mopt-rs adds.

- [Optimization](migration_optimizers.md): Hessian updates, step methods and enhancers, trust
  radii, coordinate systems, periodic cells, the optimization loop, constraints, and model
  Hessians.
- [Calculators and surfaces](migration_surfaces.md): the built-in calculators, combined surfaces
  and bias potentials, crossings of two states, ONIOM, and periodic GFN-FF.
- [Reaction paths](migration_paths.md): chains of images (`nebmain`), the IRC, BITSS, and
  `ieipmain`.
- [Vibrational analysis and analyses of paths](migration_vibrations.md): normal modes,
  thermochemistry, and the curvature, orbitals, and maps of a path.
- [Dynamics, searches, the mapper, and kinetics](migration_workflows.md): `mdmain`,
  `orientsearch`, `confsearch`, `relaxedscan`, `run_mapper`, and the kinetics.
- [Numerical safeguards](migration_safeguards.md): the guards of both code bases compared.

## Command-line compatibility notes

### `-ih`: the first Hessian from a file

New in mopt-rs (`optmain`, `ircmain`, `freqmain`): MultiOptPy starts from a unit matrix,
a model Hessian, or the calculator's Hessian. The file is read by `io::hessian_file` (a
NumPy file, an ORCA `.hess` file, a Gaussian formatted checkpoint file or output, numbers
only), and a wrapper of the calculator (`pes::given_hessian`) answers the first request
for a Hessian with it, so the optimizer, the IRC, and the vibrational analysis are as they
were. The ORCA and Gaussian formats are read as those programs document them and as the
Python side of mopt-rs reads their outputs; they were tested with files written in these
formats by the tests, not with files from the programs.
`io::hessian_file::tests`, `pes::given_hessian::tests`, `tests/cli_optmain.rs::
a_run_starts_from_the_hessian_of_a_file`, `tests/cli_freqmain.rs::
the_vibrations_of_a_hessian_that_is_read`, `tests/cli_ircmain.rs::
the_path_starts_from_the_hessian_of_a_file`.

### The method without `-opt`

MultiOptPy uses `FIRELARS` when no method is named, whatever `-order` is. mopt-rs uses
`rsirfo_block_fsb` for a minimum and `rsirfo_block_bofill` for a saddle point (`-order 1`
and above): the restricted-step rational function method with the block updates, which is
what the author of MultiOptPy uses in the articles, and a first-order method cannot look
for a saddle point. `-opt FIRELARS` gives MultiOptPy's default. The comparisons with
MultiOptPy's trajectories name the method (`tests/golden_lj.rs`);
`tests/cli_optmain.rs::the_method_without_opt_follows_the_order`.

### `-pyscf`

MultiOptPy's default program is Psi4 and `-pyscf` switches to PySCF. In mopt-rs PySCF is
the default, so the flag changes nothing; it is accepted (the same as `-os pyscf`) so that
command lines written for MultiOptPy run as they are. Together with another choice of
program (`-os`, `-xtb`, `-dxtb`) it is an error; MultiOptPy lets `-pyscf` win.

### `-sc` shape conditions

`-sc` takes triples `value gt|lt atoms` (e.g. `-sc 2.5 gt 2,3 120 lt 1,2,3`); the run continues
while every condition holds and stops at the first violation, as in MultiOptPy. Dihedral
conditions use MultiOptPy's sign convention, which is opposite to the IUPAC convention used by
mopt-rs internal coordinates, so existing thresholds keep their meaning.

## Methods added in mopt-rs

### CCQN transition-state search (`-opt ccqn`)

Cone-shaped constrained quasi-Newton method of Wu and Wang (J. Chem. Theory Comput. 2025), not
present in MultiOptPy. While the quasi-Newton Hessian is positive definite, the step solves
eqn (9) (lowest point of the quadratic model on the sphere `||s|| = Delta` inside the cone around
the reaction-coordinate axis `e`) with the augmented Lagrangian method of Algorithm 1; once the
Hessian is indefinite, a saddle-point step is taken:

- `ccqn`: restricted-step P-RFO with the trust-radius rule of eqn (19), as in the paper;
- `ccqn_rsirfo`: RS-I-RFO (MultiOptPy's saddle optimizer).

The Hessian is updated with TS-BFGS (eqn (5), (6)) unless the method string names another
updater (`ccqn_bofill`, `ccqn_block_bofill`, `ccqn_rsirfo_block_bofill`, ...). The axis is
given by reactive bonds (Algorithm 2), by a midpoint toward the product (eqn (18)), or directly.
CCQN requires Cartesian coordinates.

Defaults follow the paper's Supporting Information (section 1): `cos(phi) = 0.1` for reactive
bonds and fixed directions and `0.5` for a midpoint axis; step sizes `0.1 <= Delta <= 0.2`
Angstrom (uphill steps of 0.1 Angstrom; the saddle-phase radius starts at 0.1 Angstrom and stays
within these bounds); initial Hessian `70 I` eV/Angstrom^2. The eqn (19) ratio thresholds and
factors are not given in the paper or the SI; Sella's saddle-search values are used
(`rho_inc = 1.035`, `rho_dec = 5`, `sigma_inc = 1.15`, `sigma_dec = 0.65`). All are settings.
CCQN controls its step length itself, so the optimizer's outer trust radius (0.1 Bohr for saddle
searches in MultiOptPy) applies only when `-tr` is given.

The uphill subproblem (eqn (9)) is solved exactly by default: its feasible set is a spherical
cap, whose minimizer is either a local minimizer of the quadratic on the sphere inside the cap or
the minimizer on the cap's boundary circle; both follow from secular equations in the
eigenbasis of `B` (`opt::step::cone`). The paper's augmented Lagrangian method (Algorithm 1)
returns a local solution and is available as `ConeSolver::Alm`; its tolerances are taken relative
to `Delta` and to the model gradient, and the penalty growth is bounded (the absolute tolerances
of a first version made single steps take seconds on LJ7). The inner minimization of Algorithm 1,
which the paper does not specify, uses L-BFGS with Armijo backtracking.

CCQN also runs in non-Cartesian coordinate systems (DIC, RIC, user-defined): the Cartesian axis
`e` is mapped with `dq = B dx` and the subproblem is solved in the optimizer's coordinates.

On Muller-Brown with a straight-line midpoint axis, the paper's `cos(phi) = 0.5` is too wide
(66 of 96 starting points succeed); 0.8-0.9 gives 92-96 of 96, the remedy the SI recommends for
coarse axes. RS-P-RFO and RS-I-RFO reach the same success rates; RS-P-RFO needs about as many or
fewer steps.

Quasi-Newton updates normally skip pairs with `s . y <= 0`, as MultiOptPy does. CCQN needs the
Hessian to acquire negative curvature, so for CCQN such pairs are used by every update whose
formula allows it (all except BFGS and block BFGS). Other methods are unaffected.

When the curvature test switches between the uphill and the saddle phase, the saddle-step
method keeps its trust radius and only discards the rating of the step taken by the other phase.

Tests: `tests/ccqn.rs`: an LJ7 first-order saddle (exactly one negative Hessian eigenvalue) from
the minimum in Cartesian and DIC coordinates with RS-P-RFO, RS-I-RFO, and block Bofill; both
Muller-Brown saddles from starting points 10%, 30%, and 60% of the
way into the wells, with all four saddle-step / updater combinations (`ccqn`, `ccqn_rsirfo`,
`ccqn_block_bofill`, `ccqn_rsirfo_block_bofill`), once with an exact Hessian every 5 iterations
and once from the identity Hessian, and a case where `rsirfo_block_bofill` alone fails;
`opt::step::cone::tests` and `opt::step::ccqn::tests` (exact and ALM cone steps against dense
scans), `opt::step::prfo::tests`,
`opt::update::formulas::tests::ts_bfgs_learns_negative_curvature`,
`opt::update::tests::negative_curvature_only_when_requested_and_supported`.

### User-defined bias potentials

MultiOptPy has a fixed set of bias options. mopt-rs adds expressions of geometric variables
(`-ubias`, also from Python), Python callbacks (`Optimizer.add_term`, checked with
`mopt_rs.check_term`), and registration of new options from Rust (`cli::bias::BiasRegistry`);
see [bias_potentials.md](bias_potentials.md). Expressions get exact derivatives and
periodic-cell support from the same engine as the built-in potentials.

Tests: `tests/bias_user.rs` (term checks in 0D-3D cells, agreement with `-kp`, `-ka`, `-kpv2`,
time dependence, a restrained `optmain` run), `cli::bias::tests::user_options_can_be_registered`,
`tests/test_python_api.py::test_python_terms_are_checked_and_restrain_an_optimization`.

### User-defined constraints

MultiOptPy constrains only its built-in coordinates. mopt-rs holds any function of the
structure: `-uc` expressions in the `-ubias` language (`"EXPR"`, `"EXPR = VALUE"`,
`"; period = 2*pi"`), Python callables (`constraint_functions=`), and any Rust `PesTerm`
(`ConstraintKind::Function`), in every optimizer and coordinate system.

Tests: `tests/constraints.rs::user_defined_constraints_from_rust_terms_and_expressions`,
`tests/constraints.rs::user_constraints_on_the_command_line`,
`tests/test_python_api.py::test_user_constraints_from_python`.

### Calculators

MultiOptPy implements every program as a module that repeats file handling, unit conversion,
charge/spin parsing, Hessian scheduling, rigid-body projection, and error flags, and every
ASE-based program as its own wrapper class. mopt-rs has one engine contract
(`mopt_rs.calculators.Engine`) and one adapter (`EngineCalculator`) that provides these
services. Any ASE calculator works without code of its own (`-os ase:module:Class`, or a
class, factory, or instance from Python): the charge and spin keywords and their convention,
the Hessian method, and an unrestricted reference for open shells are detected from the
calculator; `AseProfile` entries only add fixed keywords under a short name. The default
quantum-chemistry engine is PySCF instead of Psi4. Corrections:

| Program | MultiOptPy | mopt-rs |
|---|---|---|
| dxtb | float32 positions; `spin` = multiplicity; separate SCFs for energy and forces | float64; `spin` = 2S; one SCF with autograd |
| tblite | every calculation (and each of the 6N Hessian displacements) starts from scratch | SCF restarted from the previous result |
| PySCF | X2C only for unrestricted DFT; density fitting always on; no SCF guess reuse | X2C and density fitting optional for every reference; the previous density is the guess |
| PySCF | excited-state Hessians fail | finite differences of TDA gradients |
| PySCF | restricted and unrestricted wavefunctions only | also restricted open-shell (ROHF/ROKS, analytic gradients) and generalized (GHF/GKS, gradients from differences of energies since PySCF has none): `-calc_opt reference=ro` / `reference=g` |
| all | the calculator projects rigid-body motion out of Hessians, also for periodic or fixed-frame systems | projection is the optimizer's decision |
| Gaussian | on Linux `ASE_GAUSSIAN.run` uses names that are not defined there (`atom_obj`, `functional`, ...) and stops; the force constants are read by counting lines of the five-column layout | one code path (`mopt_rs.calculators.programs`); the route has `nosymm` so that forces and force constants share the frame of the input; the reader takes the last section and checks that it is complete |
| MOPAC, NWChem | `mult` is passed to the ASE calculator, which does not read it (ASE takes the spin from magnetic moments, NWChem from its `dft` block): open shells run as singlets | the multiplicity is set as magnetic moments (MOPAC: `DOUBLET UHF`, ...) and as `dft mult` with `odft` (NWChem) |
| GAMESS-US | `charge`, `mult`, and the basis name are passed as top-level keywords, which ASE's input writer takes for input groups | `$CONTRL ICHARG`, `MULT`, `SCFTYP`; the basis name is translated to `$BASIS` keywords |
| GPAW, Quantum ESPRESSO | modules without an option that reaches them | `-os gpaw`, `-os espresso` |
| file-based programs | fixed file names in one directory, also when structures are evaluated side by side | one calculation at a time per engine |
| file-based programs | the settings each wrapper has arguments for | every input group, block, and keyword through `-calc_opt` (`GROUP.KEY=`, `KEY=@FILE`), listed by `python -m mopt_rs.calculators.programs`; defaults per program in `mopt_calculator_options.json` |
| MOPAC | keywords other than the method cannot be given | `keywords="..."` is written to the keyword line (ASE's calculator writes only the method, the task, `RELSCF`, the charge, and the spin), `-core` writes `THREADS=N` |
| GPAW, Quantum ESPRESSO | a molecule gets no cell (`Atoms` without one) | a box around the first structure, kept for the run (`vacuum`); GPAW without point-group symmetry, which would stop the run at the first structure that breaks it |
| ASE calculators | no stress is used (MultiOptPy has no variable-cell optimization) | the stress of the calculator is used by `optmain -vc`, asked for only there and in one run with the energy and the forces (Quantum ESPRESSO: `tstress`); checked against ASE's EMT, GPAW, and Quantum ESPRESSO |
| GPAW, Quantum ESPRESSO | charge and spin are not passed | the charge (`charge`, `tot_charge`) and the multiplicity (initial magnetic moments; `nspin`, `tot_magnetization`) are set from `-elec` and `-spin` |

The spin is always the multiplicity `2S + 1` (`-spin`); each engine declares its own convention
(`SpinConvention.MULTIPLICITY`, `UNPAIRED` = 2S for PySCF `mol.spin`, xTB `uhf`, dxtb `spin`).

MultiOptPy supports only the program settings it has options for (`-cpcm`, `-alpb`, ...).
mopt-rs passes any setting through `-calc_opt`: engine keywords, `prefix.path=value` keys on the
program's objects (PySCF `scf.`, `mol.`, `apply=`; tblite `set.`, `add.`), and user hooks
(`hook=module:function`) on the program object, so solvent models, external fields, and other
features work without new options (`mopt_rs.calculators.program_options`).

MultiOptPy needs `software_path.conf` for many programs and does not log which program or
version it loaded. In mopt-rs the engines built into mopt-rs and the pip-installed packages are
the default, `software_path.conf` entries only override paths, and every run logs the
calculator's distribution, version, and file, and what was read from `software_path.conf`.

MOPAC 23.2.5, GPAW 26.7.0, and Quantum ESPRESSO 7.6 were run (`tests/test_programs_run.py`;
the table of what the runs check is in `calculators.md`, "Programs run through files").
Gaussian, ORCA, GAMESS-US, and NWChem were not run here; for them the tests check the input
they are given and the reading of the Hessians of Gaussian and ORCA
(`tests/test_programs.py`, `tests/test_program_settings.py`).

Tests: `tests/test_calculators.py`, `tests/test_programs.py`,
`tests/test_program_settings.py`, `tests/test_programs_run.py`,
`tests/cli_optmain.rs::software_path_conf_is_reported`.

### GFN0-xTB built in

MultiOptPy's `gfn0xtb` model Hessian is its own approximation (kept as `legacy_gfn0xtb`), and
GFN0-xTB itself needs an external program. mopt-rs has GFN0-xTB built in (`-os gfn0xtb`, and
`-modelhess gfn0xtb` for its Hessian) for H to Rn: the non-self-
consistent method of Pracht et al. with xtb's parameters, following gfn0-rs, with an analytic
gradient and an analytic Hessian of the ground state (the response of the Fermi-smeared
density to H0 and S, and the second derivatives of the EEQ charges by one adjoint solve per
weight; checked against central differences of the gradient). Energies and gradients agree
with gfn0-rs to round-off (`tests/golden/gfn0_xtb.json`, `crates/mopt-rs/tests/gfn0_xtb.rs`).

### GFN-FF built in

MultiOptPy's `gfnff` model Hessian is its own approximation (kept as `legacy_gfnff`), and the
force field itself needs xtb. mopt-rs has GFN-FF built in (`-os gfnff`, and `-modelhess gfnff`
for its Hessian) for H to Rn: a port of xtb 6.7.1's `src/gfnff`
(version `angewChem2020_2`) for isolated molecules, with the topology (bonds, hybridization,
parameters, topology charges) set up from the starting structure as in xtb (also after
`--restart`). Each structure is checked with xtb's own neighbour search, and when the bonds
found differ from the topology (a bond formed or broken), the run log shows a warning naming
them, once for each new bonding pattern (Python: a `RuntimeWarning`); `-calc_opt
refresh_topology=true` sets the topology up at every structure instead. Every energy term and
the gradient agree with xtb (`tests/golden/gfnff.json` from `tools/golden/gfnff.py`, and the
formaldehyde-water reference of xtb's unit test; `crates/mopt-rs/tests/gfnff.rs`). Differences
from xtb:

- The hydrogen and halogen bond terms end smoothly. xtb lists the pairs within thresholds
  (250 Bohr^2 for the squared distance of A and B, 450 Bohr^2 for the sum of the three squared
  distances of A, B, and H and for the squared distance of a halogen bond), keeps the lists
  until the structure has moved by more than 0.3 Bohr (RMSD), and then builds them anew: its
  energy is continuous between two rebuilds, steps at a rebuild, and depends on the
  structures it saw before. mopt-rs takes the lists of the structure at hand, so that the
  energy is a function of the structure alone (a restart, images evaluated in any order);
  with sharp thresholds that energy stepped wherever a pair crossed one (1.4e-8 Hartree for
  a pair of water molecules at 15.8 Bohr), which stalled line searches near convergence: the
  relaxation of 20 water molecules around a solute stood at a largest gradient of 3e-5
  Hartree/Bohr. Each term is now multiplied by the infinitely smooth switching function
  of `pes::native::switching` over the last 10 % of its threshold (15.0 to 15.8 Bohr for the
  first), so the energy has no step; below that range it is xtb's. `-calc_opt hb_switch=off`
  gives the sharp thresholds (xtb's energy of a single point at every structure),
  `hb_switch=SHARE` another range. With the switching the step at the threshold is 3e-15
  Hartree (molecule and periodic images alike), the gradient and the stress are those of the
  switched energy, and the energy differs from xtb's single point where pairs lie in the
  range: by nothing for 17 of the 20 reference molecules, 3e-10 Hartree (pdb 4qxx), 3.4e-7
  (rivaroxaban), and 2.9e-6 (taxol); in crystals, which have pairs at every distance, by
  8.9e-6 (cubic ice), 7.9e-6 (benzene), and 1.6e-5 Hartree (benzene 2x1x1). The comparisons
  with xtb's terms, gradients, and strain derivatives in the tests are made with
  `hb_switch=off` and hold as before (`tests/gfnff.rs::hydrogen_bond_terms_end_without_a_step`,
  `tests/gfnff_periodic.rs::hydrogen_bond_terms_of_images_end_without_a_step`).
  What it changes in a run (formaldehyde in 22 water molecules, three seeded clusters, ONIOM
  with GFN-FF at both levels, `-tcc`, 60 iterations): with the sharp thresholds a relaxation
  of the environment went on with FIRE after a line search found no step that lowers the
  energy in two of the three runs, with the switching in none; the gradients were 4264,
  5842, and 4120 against 5676, 2381, and 3867, and the one run of the six that converged
  within the 60 iterations was a switched one.
- The gradient of the HB, XB, bonded ATM, and torsion terms comes from forward-mode automatic
  differentiation of the energy instead of xtb's hand-written derivatives (the values agree).
- xtb's triple-bond torsion term `sTors` is left out: its atom list is never filled in xtb 6.7.1,
  so it contributes nothing there either.

### Optimization of plain variables

MultiOptPy optimizes atomic coordinates only; its two-variable Muller-Brown surface has to be
given as one atom whose third coordinate is tied down. In mopt-rs the optimizers
(`opt::vector::VectorOptimizer`: the same step generators, Hessian updates, step enhancers,
trust radius, method blending, and convergence test as for molecules) also run on any function
of `n` variables, with `n` not a multiple of three and nothing treated as an atom:

- command line: `optmain VARIABLES_FILE -os mb2d` or `-os "expr:..."` (text or JSON variables
  file; CSV trajectory and the optimized variables as output), see [running.md](running.md);
- Python: `mopt_rs.optimize_variables(fun_or_name, x0, ...)` and the ask/tell
  `mopt_rs.VectorOptimizer`;
- Rust: the `Objective` trait, `ObjectiveRegistry`, `VectorOptimizer`,
  `run_vector_optimization`.

On the Muller-Brown surface the two-variable run takes exactly the steps of the one-atom model
(`-os muller_brown`). CCQN takes a direction vector as its axis here (reactive bonds need
atoms). The L-BFGS step limit is the largest component for plain variables (the largest atomic
displacement for molecules). NEB, IRC, vibrational analysis, and molecular dynamics on plain
variables come with those commands.

Tests: `crates/mopt-rs/tests/optimize_variables.rs` (three minima and two saddle points of the
Muller-Brown surface with RS-I-RFO, CCQN, FIRE, L-BFGS; a five-variable quadratic form;
expressions), `crates/mopt-rs/tests/cli_variables.rs`, `tests/test_variables.py`.

### AutoTS with a workflow block

MultiOptPy's `run_autots` has two implementations, chosen by the settings file: the four
steps once each ("v1"), or what a `workflow` block lists ("v2",
`Wrapper/autots.py:AutoTSWorkflow_v2`), with `repeat`, `repeat_settings[k].param_override`,
`settings_key`, `enabled`, and for step 2 `mode` (`sequential` or `independent`). mopt-rs
reads the block with the same keys and the same meaning (`cli::autots_workflow`), and
writes the files of the runs under MultiOptPy's names (`<stem>_step1_run<k>_traj.xyz`,
`<stem>_step1_combined_traj.xyz`, `<stem>_step2_run<k>_init_path.xyz`,
`<stem>_step2_run<k>_final_path.xyz`, `<stem>_step2_run<k>_candidates/`,
`<stem>_s3_run<r>_ts_final_<k>.xyz`). Where MultiOptPy's implementation does something
that its settings do not ask for, mopt-rs does what they ask for:

| | MultiOptPy | mopt-rs |
|---|---|---|
| Candidates of step 2 that go to step 3 (`top_n_candidates`) | ranked by `energy_plot.csv` looked up with the number of the candidate file (`..._guess_<i>`) as if it were the number of the image; with `mode: independent` one file (that of the run with the most images) is used for the candidates of all runs | ranked by the energies of the images they are |
| Joining the trajectories of the runs of step 1 | the files one after the other, so the structure at which a run starts is there twice | that structure once |
| Several runs of step 3 | every run starts again from all candidates; only what the last run made is kept, so a transition state of an earlier run is lost when a later run does not converge | a later run takes up the candidates that have no transition state yet; the others keep theirs |
| The same step in two entries of the block | the second entry starts again from the first path, and its results are added to those of the first | the entries follow each other like the runs of one entry |
| `--skip_step1` | step 2 stops ("combined_path not found") | the input is the first path |
| Optimization of the ends of the IRC (step 4) | with the method of the IRC run (`opt_method` of `step4_settings`, a saddle method in MultiOptPy's examples); `step4b_opt_method` is not read | with `step4b_opt_method`, as in the four fixed steps |
| Several runs of step 4 | the same calculation again under another name | one run |
| A structure of step 3 that did not converge | left out | listed as a candidate with a warning, as in the four fixed steps |

MultiOptPy's example of the block (`test/config_autots_v2_test.json`) uses a neural network
potential that was not available here; the block was run with its structure (two runs of
step 1, two of step 2 with different methods) on the built-in engines.

New in mopt-rs: `step1_method` with a search of `ieipmain` between two structures
(`reaction_paths.md`, "Step 1 by a search between two structures"), and the warning when steps 3
and 4 run on different surfaces.

### Settings files, interruption, and restart

MultiOptPy reads JSON settings only in `run_autots` and `run_mapper`, and has no general way to
continue an interrupted run. Every mopt-rs command takes `--config FILE` (and
`--dump-config FILE`), keeps `settings.json` and an evaluation journal in its run directory,
stops cleanly on a `STOP` file, and continues with `--restart RUN_DIR` without recomputing
finished evaluations; see [running.md](running.md).

Tests: `cli::settings::tests`, `workflow::journal::tests`, `tests/restart.rs`,
`tests/test_calculators.py::test_settings_files`,
`tests/test_python_api.py::test_checkpoints_resume_interrupted_runs`.
