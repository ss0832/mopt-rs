# Running calculations

How a run is set up and what it leaves: what it prints, settings files, interruption and
restart, objective functions of plain variables, and the Rust interface. The surfaces, the
options of the optimizations, and the commands have pages of their own:
[Calculators](calculators.md), [ONIOM](oniom.md), [Excited states](excited_states.md), [Options
of the optimizations](optimization.md), [Vibrational analysis](vibrations.md), [Reaction paths
and transition states](reaction_paths.md), [Molecular dynamics](molecular_dynamics.md), and
[Conformations, orientations, and scans](searches.md).

## What a run prints

`optmain` writes to the screen and to `mopt.log` what MultiOptPy writes to the screen, in
the same order and with the same number formats:

```text
********************************
ar4.xyz
********************************
Use Lennard-Jones cluster potential.
Ar      0.000000000000      0.000000000000      0.000000000000      the structure, Angstrom
...
# ITR. 0
Ar     -1.925000000000     -1.175000000000     -0.875000000000      the structure evaluated
...
COORDINATES (ric, NUMBER OF VALUES: 3, unit a.u.):                   with -coord other than cart
R(1-2)                 :  1.834523412345
<S**2>                 :  0.781200000000  (S(S+1) = 0.7500; main)    where the calculator tells it
Energy ratio (actual/predicted): 0.9731...                           trust radius, when adapted
==================================================================================
trust radii (unit. Bohr):  0.5
step  radii (unit. Bohr):  0.4921047688193129
EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES: 6):                iterations with a Hessian
  0.00018431   0.00034346   0.00046667   0.00065936   0.00089065   0.00143207
==================================================================================
caluculation results (unit a.u.):
                         Value                     Threshold
ENERGY                 : -0.002204275365
...
```

Cartesian coordinates are in Angstrom (12 decimals); everything else is in atomic units
(Hartree, Bohr, radian). With a coordinate system other than Cartesian (`-coord ric`,
`dic`, `tric`, ...), the structure in those coordinates follows the Cartesian one at every
iteration: named (`R(1-2)` bond, `A(1-2-3)` angle, `L(...)` linear bend, `D(...)`
dihedral, atom numbers from 1) where each coordinate is one internal coordinate (`ric`),
numbered `q1, q2, ...` where they are combinations.

### Physical constants

Every command converts with one set of constants, CODATA 2018 (`mopt_rs::domain::Units`,
exported to Python as `mopt_rs._core.BOHR_TO_ANGSTROM` and friends): the exact SI values of
`h`, `c`, `e`, `k_B` and `N_A`, the thermochemical calorie (4.184 J), and the CODATA 2018 Bohr
radius (0.529177210903 Angstrom), Hartree energy (4.3597447222071e-18 J), electron mass and
atomic mass constant. Derived from these: 1 Hartree = 27.211386245988 eV = 627.509474063
kcal/mol = 2625.49963948 kJ/mol = 219474.6313632 cm^-1, `k_B` = 3.1668115635e-6 Hartree/K,
1 amu = 1822.888486217 electron masses, 1 Hartree/Bohr^3 = 29421.0157 GPa. MultiOptPy uses
older values (Bohr 0.52917721067 Angstrom, 627.509 kcal/mol, 2625.5 kJ/mol); they are kept only
in the `legacy_` model Hessians, which reproduce MultiOptPy, and in comparisons with MultiOptPy
reference data. Engines that carry their own constants for parameter compatibility (the GFN
methods use xtb's and tblite's, EMT uses ASE's) keep them, and the source names them.

Under the calculator line of a run come the papers to cite for the method and the program
(`Reference: ...`): for the potentials built into mopt-rs their original publications,
for PySCF, tblite, dxtb, Psi4, and ASE those of the programs. For a calculator reached
through ASE or through MultiOptPy, cite the program behind it as its authors ask; mopt-rs
cannot know it.

What a program prints itself is part of the stream: the output of PySCF (SCF iterations,
energies, `<S^2>`, gradients; `verbose=4` as in MultiOptPy, `-calc_opt verbose=N` to
change it) is shown under the heading of the iteration it belongs to and written to
`mopt.log`, once the evaluation is done.

`-debug` also shows on the screen what the optimization methods do inside a step: for
RS-I-RFO the lines of MultiOptPy (`RS-I-RFO Iteration`, gradient norm, condition number
of the Hessian, number of negative eigenvalues, the unrestricted step and whether it had
to be shortened, final step length, predicted energy change). They are always in
`mopt.log`.

Input names with `*`, `?`, or `[...]` stand for the files that match (as MultiOptPy's
`glob`), so `optmain "ts_*.xyz" ...` works without a shell as well. From Python,
`mopt_rs.run_optmain(["ts_*.xyz", "-os", "lj", ...])` runs the command and returns one
dictionary per input, in order (MultiOptPy's `Optimize.all_results`): `input_file`,
`folder_directory`, `final_file_directory`, `optimized_struct_file`, `traj_file`,
`element_list`, `final_geometry` (Bohr), `final_energy`, `final_bias_energy` (Hartree),
`symmetry`, `optimized_flag`, `interrupted`, `iterations`.

Small options of MultiOptPy:

| Option | What it does |
|---|---|
| `-d X` | step factor of the methods that have one (the learning rate of Adam, AdaBelief, RAdam, Eve, Prodigy; the factor on the steps of gradient descent, L-BFGS, Newton); other methods say that they have none |
| `-of 1-3 4,5` | each group of atoms gets the mean gradient of the group and moves as a whole (methods without a Hessian) |
| `-gi 1,2,5` | `geometry_info.csv` with the distances (Angstrom), angles, and dihedral angles (degrees) between the atoms at every iteration, and `plot_orthogonality_<atom>.png` with the cosine between the bias gradient and the gradient of the surface on each atom |
| `-negeigval` | with `-fc N` and `-order > 0`: stop at once if the first Hessian has no negative eigenvalue (`no_negative_eigenvalues_detected.txt`) |
| `-pyscf` | accepted; PySCF is the default calculator |

`<S**2>` is shown at every iteration, with `S(S+1)` of the multiplicity asked for, when
the calculator tells it: PySCF with an unrestricted wavefunction (open shells, or `-u`).
An engine of your own reports it as `EngineResult(extra={"spin_square": value})`; a Python
calculator in a loop of `ask` and `tell` as `Evaluation(..., spin_square=value)`, and it
comes back in the report as `report["spin_squares"]` (label, value, `S(S+1)`).

The other commands print the same on the screen and into `mopt.log` of the run directory,
as the evaluations happen: the warnings of the calculator, `<S**2>`, and the output of the
program itself, named after what was evaluated (`image 3` in `nebmain`, `IRC point 12
(Path)` in `ircmain`, `evaluation 7` in `freqmain`, where the Hessians of displaced
structures, of the points of a path, and of a supercell are evaluations as well).
`freqmain` also says which Hessian it calculates (`Hessian of the structure`, `Path point 3
of 21`, `Displaced Hessian 4 of 12`). The tests are `workflow::narrated` and
`workflow::irc` (`what_the_calculator_says_at_a_point_goes_to_the_log`).

### Counts of the evaluations

Every command ends its output with what its calculators were asked for, for each input (in
the log of the run directory as well) and in total:

```text
Evaluations (this input): 37 gradient calculation(s), 2 exact Hessian calculation(s).
Evaluations (in total): 74 gradient calculation(s), 4 exact Hessian calculation(s).
```

An evaluation that returned a Hessian is counted among the gradient calculations as well
(it returns both); evaluations of the energy alone are named when there are any. What is
counted is what the calculator was asked for: a Hessian that a calculator makes from
differences of its own gradients is one Hessian, and the gradients that `-modelhess
o1numhess` asks for are gradient calculations. Evaluations that a restarted run reads from
its journal are not made again and not counted. With ONIOM microiterations the line also names
the microiterations: the inner iterations of the relaxations of the environment, and how many
relaxations there were (`, 412 microiteration(s) in 37 relaxation(s) of the environment`); the
gradients of the lowest level in them are among the gradient calculations. From Python, `mopt_rs.cli.run_counted(
"optmain", [...])` returns the exit code and the total; in Rust,
`CalculatorRegistry::counts()` and `ObjectiveRegistry::counts()` (the calculators and
objectives a registry creates are counted, `pes::counting`).

## Settings files

Every command runs from flags, from a settings file, or both:

```bash
optmain water.xyz -os pyscf -func b3lyp -bs def2-svp -opt rsirfo_bfgs --dump-config water.json
optmain --config water.json              # the same run
optmain --config water.json -bs def2-tzvp  # the command line overrides the file
```

A settings file is a JSON object; keys are the long option names (`"opt_method"`,
`"othersoft"`; any flag spelling such as `"opt"` also works) plus `"INPUT"`; strings and
numbers are one value, arrays several, `true`/`false` switch flags. Each run directory keeps the
settings in effect as `settings.json`.

### Workflows: one calculator name, every step

Workflow settings (`run_autots`, `run_mapper`) contain one block of settings per step, in the
same format as the settings file of the command the step runs (`optmain`, `nebmain`, ...).
Step 2 uses the same iteration limits as `nebmain`: 30 for NEB and 60 for direct MaxFlux.
Both methods stop earlier when their convergence criteria are met.
Use `step2_settings.NSTEP` to change the limit. AutoTS builds a calculator for each step.
Specify the calculator and any ONIOM settings in each step that uses them.

```json
{
  "work_dir": "RE_tol_Pd_PMe3",
  "step1_settings": {"othersoft": "ase:pm6_rs.ase:PM6", "opt_method": ["rsirfo_block_fsb"],
                     "manual_AFIR": ["300", "4", "13"]},
  "step2_settings": {"othersoft": "ase:pm6_rs.ase:PM6", "NSTEP": 20},
  "step3_settings": {"othersoft": "ase:pm6_rs.ase:PM6", "calc_exact_hess": 5},
  "step4_settings": {"othersoft": "ase:pm6_rs.ase:PM6",
                     "intrinsic_reaction_coordinates": ["0.5", "200", "lqa"]}
}
```

Why this gives the same calculator in every step:

1. **One parser for all settings.** A step block is read by the same code as `--config` files
   and the command line (`cli::settings`), so `"othersoft"`, `"functional"`, `"calc_option"`,
   ... mean in a step exactly what they mean for the command.
2. **One translation into a calculator request.** `cli::calculator` turns the block into a
   name (`-os`, `-xtb`, `-dxtb`; default `pyscf`) and program-independent
   `CalculatorOptions` (method, basis, threads, memory, `-calc_opt` keywords). Identical
   settings give identical requests.
3. **One registry for all commands.** Every command, and every step of a workflow, receives
   the same `CalculatorRegistry` from the dispatcher (`cli::run_command_with`) and builds its
   calculator only through it. The registry knows the Rust potentials; run from the Python
   package (the pip-installed commands or `mopt-py`), it also has a fallback that hands every
   other name to `mopt_rs.calculators.create`.
4. **One adapter for ASE calculators.** `ase:pm6_rs.ase:PM6` is imported by path and wrapped in
   `AseEngine`, which inspects the class to find how it takes charge and spin, how it returns
   Hessians, and whether open shells need an unrestricted reference, and `EngineCalculator`
   converts its results to atomic units. Nothing in this chain depends on which step asked.
5. **The state comes with each request.** Charge and multiplicity travel with every structure
   (`-elec`/`-spin` of the step), so steps that change the state (for example the two ends of a
   path) are evaluated correctly with the same calculator.

`run_mapper` runs its steps on one calculator instance (with several jobs at a time, a copy
for each worker of a Python engine), so engines that restart their SCF from the previous
structure keep doing so across steps; the steps of `run_autots` each build their own instance
from the same settings. Each step of `run_autots` records its evaluations in its own journal,
so an interrupted workflow restarts step by step.
Settings written by MultiOptPy for these workflows (`run_autots -cfg config.json`) are read as
they are; only the calculator name changes when UMA is replaced by another program.

(`run_autots` follows rules 1–5; its steps do not share a calculator instance yet.
`run_autots --restart WORK_DIR` makes an interrupted or killed run again: see
[Interruption and restart](#interruption-and-restart). `run_mapper` runs every step on one
calculator with one journal and continues job by job, and with `-map_opt kinetics=on` it
computes the kinetics of its network when the exploration ends, also after a restart; see
[mapper.md](mapper.md). `run_automecp INPUT -cfg config.json -ma GAMMA FRAGMENT1 FRAGMENT2
-state2 CHARGE MULTIPLICITY [ROOT]`, the search for the minimum-energy crossing points of two
states from one structure, is a run of the mapper with the same rules; its section in
[mapper.md](mapper.md) has its settings file and its files.)

## Threads and reproducibility

`-core N` (`N_THREAD` in a settings file) is the number of threads of one calculation: a
command runs in a pool of that many threads (at least as many as `-parallel` structures it
evaluates at the same time), and the built-in engines and the dense linear algebra split
their work over it. The mapper takes it from its step settings and shares the threads out
among its workers (`-map_opt workers=N`).

**The results do not depend on the number of threads.** The energies, structures, gradients,
iteration counts and every file a run writes are the same bits with 1 thread and with 8, and
the network of the mapper with the batch schedule (`schedule=batch`, the default) is the same
with one worker and with four. The built-in engines split their work into chunks of a fixed
size and add them in a fixed order; their dense products, factorizations and
eigendecompositions use a fixed number of tasks, chosen from the size of the matrix alone; and
every other use of the linear-algebra library (in the optimizers, the coordinate systems, the
vibrations, the kinetics, ...) runs sequentially, which the commands, the Python module, the
MCP server and the native calculator registry set when they start (a Rust program that uses the
optimizers or the engines without any of them calls
`mopt_rs::core_infra::parallelism::deterministic_linear_algebra()` first). The library's own
default would size its work from the pool each call runs in, and so give other last bits in
pools of other sizes. `tests/thread_independence.rs` checks the engines and
`tests/thread_independence_commands.rs` whole commands in pools of 1, 2, 4 and 8 threads.

What is not covered, by design:

- the times in the logs and the order of the records of `evaluations.journal` when several
  structures are evaluated at the same time (each record is written as its evaluation ends; a
  restart looks the records up by their structure, so the order does not matter);
- the mapper with `schedule=rolling`, which hands out each job as soon as a worker is free:
  its network depends on which job ends first;
- external programs and Python calculators, which have their own threads.

## Interruption and restart

Every run records its evaluations in `RUN_DIR/evaluations.journal`. To stop a run cleanly,
create a file `RUN_DIR/STOP`; the run stops before its next evaluation (exit code 3). Continue
it, or a run that was killed, with

```bash
optmain --restart RUN_DIR
```

The settings come from `RUN_DIR/settings.json` (flags given with `--restart` override them).
The run is rebuilt and its evaluations are taken from the journal as long as the requested
structures match the recorded ones; then the calculations continue, so nothing that finished
is computed again (a record cut off by a kill is ignored). The result is the same as that of an
uninterrupted run. A record holds the cell of a periodic structure and the stress the
calculator gave; journals of periodic runs written before cells were recorded are not used
(the run computes again).

With ONIOM microiterations a record also holds the environment relaxed at that core
structure, and the replay gives it back: the relaxations after the replay start from the
environments the first run had, not from the input. (The pairs of the limited-memory BFGS of
the relaxations are not recorded, so the first relaxations after the replay can take other
steps to the same minimum within the tolerance.)

**A workflow** (`run_autots`) is made again with

```bash
run_autots --restart WORK_DIR
```

`WORK_DIR` is the working directory of the run (`work_dir` of its settings file). It holds the
settings the run was started with (`settings.json`, the settings file as it was read in
`autots_config.json`, and the copy of the input), so nothing else is needed and it works from
any directory. The run starts again from step 1; every evaluation of the run before, from the
journals of all its steps (every `evaluations.journal` under `WORK_DIR`), answers a request of
the new run that asks for it again: the same calculator (its label and its description: the
method, the basis, the settings), the same charge, multiplicity and state, and the identical
structure. So the steps that had ended take no new calculation, the step that was running goes
on from where it stopped, and the steps after it are computed. The new runs of the steps are
written next to the old ones, and the log ends with how many evaluations were taken. A run
that a kill cut off in the middle of an evaluation loses only that evaluation.

From Python, `mopt_rs.optimize.run(optimizer, calc, checkpoint="run.ckpt")` does the same for
the ask/tell API: run the same setup again with the same checkpoint file to continue.

## Python workflow components

`mopt_rs.relax_band(evaluate, images, ...)` relaxes a chain as `nebmain` does: the
same parser of its options, the same defaults, and the same set-up of the chain. A call
without options gives the band of `nebmain` without options on the same input (30
iterations, model Hessians `lindh2007d3`, images 0.5 Angstrom apart on the geodesic path,
redistribution towards the predicted maxima every second iteration). `evaluate` is a
function `f(x, image, need_hessian)` of the flat coordinates (Bohr for atoms) that returns
the energy, the gradient, and an optional flattened Hessian in atomic units; or the name of
a calculator of the commands (`"gfn1xtb"`, `"lj"`, an engine of `mopt_rs.calculators`, an
objective such as `"mb2d"`); or an `OniomCalculator`. `options` takes the options of
`nebmain` as on the command line, which select its methods by their names: `["-qsm"]`,
`["-qsmv2"]`, `["-lup"]`, `["-ewbneb"]`, `["-aneb", "1", "5"]`, `["-modelhess", "none"]`,
`["-neb_opt", "tangent=top", "gp=on"]`, and so on. The keyword arguments `max_iterations`
(`-ns`), `tolerance` (`-neb_opt tolerance=`), `spacing` (`-nd`, in Bohr for structures),
`inserted` (`-p`), `equal_spacing` (`-ad`), `coordinates` (`-coord`), and `surrogate`
(`-neb_opt gp=on`) override them. With `symbols` the images are structures: `charge`,
`multiplicity`, `cell_bohr` and `pbc` (as for `Optimizer`; every image in the cell of the
first, every atom following its nearest periodic image), and the constraints `fix_atoms`,
`constraints`, `user_constraints`, `constraint_functions` (the spelling of `Optimizer`) or
`-fix`, `-pc`, `-gfix`, `-uc` in `options`: every image holds its values in the first chain
(explicit values in all images) to 1e-10, frozen atoms exactly where the first image has
them. What `nebmain` refuses is refused with its message; `-dmf` is `mopt_rs.dmf_search`,
and `-neb_opt optimize=` is an `Optimizer` run on the images returned. `output_dir` writes
the files and figures of `nebmain`; an `observer(iteration, images, energies,
largest_force)` callback can return true to stop. Until this release `relax_band` had its
own defaults (20 iterations, steps along the forces only, no redistribution, no alignment,
the images on the straight line): pass `options=["-modelhess", "none", "-adpred", "0",
"-neb_opt", "path=linear"]` and `max_iterations=20` for steps like those.

`mopt_rs.irc_path` and `mopt_rs.Irc` take the same constraints (resolved once at the start,
held at every point to 1e-10) and a cell (`cell_bohr`, `pbc`).

`mopt_rs.Optimizer(..., initial_hessian=...)` starts from a Hessian: an array of 3N x 3N
numbers (Hartree/Bohr^2), a file of the kinds `optmain -ih` reads, or the name of a model
Hessian of `-modelhess`. A matrix answers the first request for a Hessian, as `-ih` does; the
calculator is not asked for it.

`mopt_rs.OniomCalculator(high_atoms, low, high_calculator=..., ...)` is the ONIOM
calculator of the commands: the layers, link atoms (`links`, `link_placement="scaled"` or
`"distance"`), the embedding, the charges and multiplicities of the model systems, and the
settings of `-oniom_opt` (`settings={"micro_tolerance": 1e-5, "micro_replicas": 4, ...}`),
read by the parser of `-oniom` (its `options` are the options it stands for). With
`Optimizer(..., calculator=oniom)` and `mopt_rs.optimize.run(optimizer, oniom)` the
optimizer moves the core and the environment is relaxed at every evaluation, as `optmain
-oniom` does; `whole_positions_bohr` gives the whole system.

`mopt_rs.read_xyz`, `read_xyz_frames`, `write_xyz`, and `append_xyz` read and write `.xyz`
and extended `.xyz` files with the reader and writer of the commands.

`mopt_rs.relax_environment(evaluate, initial, method="lbfgs")` minimizes the
environment coordinates. Its callback takes the coordinates and a Hessian request.
The result includes the final coordinates, energy, iteration count and convergence flag.

`mopt_rs.MapperNetwork(path.read_text())` opens a mapper network from JSON.
Its `tasks(node, method=...)` calls an exploration method, `select(priority,
candidates)` chooses among (node, visits, remaining tasks) tuples, and
`save(directory)` writes the network through a store observer.

For two-state optimization, pass `model_function` and `excited_state` to
`mopt_rs.Optimizer`. A `FollowedCalculator` wraps a Python engine that supplies
state characters. For electronic embedding, engines advertising
`Capabilities(point_charges=True)` implement `compute_embedded` and return
both the molecular derivatives and the gradients at the external charges.

## Plain variables (no atoms)

The optimizers also work on any function of `n` variables; `n` need not be a multiple of three
and nothing is treated as an atom (no rigid-body projection, no coordinate systems). The
calculator name is then an objective and the input a variables file:

```bash
optmain start.txt -os mb2d -opt rsirfo_bfgs                       # Muller-Brown surface, a minimum
optmain start.txt -os mb2d -opt rsirfo_bofill -order 1 -fc 1      # a saddle point
optmain start.json -os "expr:(q1-1)^2 + 2*(q2+0.5)^2 + q1*q3" -opt lbfgs
```

| Objective | Variables | Settings |
|---|---|---|
| `mb2d` (`muller_brown_2d`) | `x`, `y` | `-calc_opt energy_scale=S` (default `1e-3`) |
| `wolfe_quapp` | `x`, `y` | `x^4 + y^4 - 2x^2 - 4y^2 + xy + 0.3x + 0.1y` (the Wolfe-Quapp surface of the studies of valley-ridge inflections and the gentlest ascent dynamics; Bofill, Quapp, Caballero, Chem. Phys. Lett. 2013): 3 minima, 3 saddle points, 1 maximum; `energy_scale` (default 1) |
| `three_hole` | `x`, `y` | the three-hole potential (Park, Sener, Lu, Schulten, J. Chem. Phys. 2003, 119, 1313; Metzner, Schuette, Vanden-Eijnden, J. Chem. Phys. 2006, 125, 084110): minima at (+-1.048, -0.042) and (0, 1.537), saddle points at (+-0.617, 1.103) and (0, -0.316), a maximum at (0, 0.519); `energy_scale` |
| `himmelblau` | `x`, `y` | `(x^2 + y - 11)^2 + (x + y^2 - 7)^2` (Himmelblau 1972): 4 minima of value 0, 4 saddle points, a maximum at (-0.2708, -0.9230); `energy_scale` |
| `rosenbrock` | `x1`, `x2`, ... (any number from 2) | `sum 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2` (Rosenbrock, Comput. J. 1960, 3, 175); with 4 variables a second minimum at (-0.7757, 0.6131, 0.3821, 0.1460) (value 3.7014) and a saddle point between it and (1, 1, 1, 1) (value 3.7082); `energy_scale` |
| `styblinski_tang` | `x1`, `x2`, ... (any number from 1) | `sum (x_i^4 - 16 x_i^2 + 5 x_i) / 2` (Styblinski, Tang, Neural Networks 1990, 3, 467): every variable at one of the roots -2.9035 (lowest), 0.1567 (a maximum of its term), 2.7468, so `3^n` stationary points, `C(n, k) 2^(n-k)` of index `k`; `energy_scale` |
| `expr:EXPRESSION` | `q1`, `q2`, ... | the expression language of `-ubias` (`+ - * / ^`, `exp`, `ln`, `sqrt`, `sin`, `cos`, `tan`, `tanh`, `acos`, `abs`, `atan2`, `min`, `max`, `pi`), with the variables in place of the geometry functions; exact derivatives by automatic differentiation |

The variables file is text, one variable per line as `value` or `name value` (`#` starts a
comment), or JSON, `[1.0, 2.0]` or `{"variables": [1.0, 2.0], "names": ["a", "b"]}`.

```text
# start.txt
x  -0.4
y   1.3
```

The run directory holds `mopt.log`, `settings.json`, `input.txt`, the trajectory
`<stem>_traj.csv` (iteration, value, maximum and RMS gradient and step, trust radius, the
variables, the gradient), the plots `value_plot_.png` and `plot_gradient_.png`, and
`<stem>_optimized.<ext>`, the final variables in the format of the input (it can be used as
the next input). No XYZ files are written. `-opt`, `-order`, `-ns`, `-fc` (the objective's
exact Hessian every N iterations), `-tcc` / `-lcc`, and the trust-radius options apply;
thresholds, trust radii, and `-ccqn_step` are plain numbers in the units of the variables and
of the objective. CCQN takes its axis from `-ccqn_product VARIABLES_FILE` (the direction from
the start to that point). Options that need atoms (bias potentials, constraints, `-coord`,
`-modelhess`, `-mf`, `-sc`, `-elec`, `-spin`, `-ccqn_bonds`) are refused. A `STOP` file
interrupts the run; `--restart RUN_DIR` repeats it from the starting point, which costs
nothing for these objectives.

From Python, the objective is a function, a built-in name, or an expression:

```python
import numpy as np
from mopt_rs import VectorOptimizer, optimize_variables

def fun(x):                                  # -> (value, gradient) or (value, gradient, hessian)
    return float(np.sum((x - 1.0) ** 2)), 2.0 * (x - 1.0)

r = optimize_variables(fun, np.zeros(5), methods="rsirfo_bfgs")
r.converged, r.x, r.value, r.gradient, r.iterations

ts = optimize_variables("mb2d", [-0.8, 0.6], methods="rsirfo_bofill",
                        saddle_order=1, exact_hessian_interval=1)

opt = VectorOptimizer([0.0] * 7, methods=["lbfgs"])        # the ask/tell loop by hand
while (request := opt.ask()) is not None:
    x, need_hessian = request
    report = opt.tell(value_at(x), gradient_at(x))
```

When an exact Hessian is asked for (`exact_hessian_interval`) and the function returns none,
`hessian=` is called, and without it the Hessian comes from central differences of the
gradient. NEB, IRC, vibrational analysis, and molecular dynamics on plain variables come with
those commands.

## From Rust

The same interfaces in Rust; `crates/mopt-rs/tests/rust_interfaces.rs` runs every step below.

**1. A program behind the `Calculator` trait.** Calculators receive Cartesian structures (Bohr)
and return the energy (Hartree), the gradient (Hartree/Bohr), and, when requested, the Hessian
(Hartree/Bohr^2); every coordinate system, optimizer, bias, and constraint works with them.

```rust
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::Calculator;

struct Harmonic { x0: Vec<f64>, k: f64 }

impl Calculator for Harmonic {
    fn name(&self) -> &str { "harmonic" }
    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let x = sys.coords_flat_bohr();
        let d = faer::Col::from_fn(x.nrows(), |i| x[i] - self.x0[i]);
        let e = 0.5 * self.k * d.iter().map(|v| v * v).sum::<f64>();
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), d * faer::Scale(self.k));
        if req.need_hessian {
            out.hessian_bohr = Some(faer::Mat::identity(x.nrows(), x.nrows()) * faer::Scale(self.k));
        }
        Ok(out)
    }
    fn is_rigid_body_invariant(&self) -> bool { false } // a fixed point in space
}
```

**2. Evaluate and optimize.**

```rust
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings};
use mopt_rs::pes::evaluate_checked;
use mopt_rs::workflow::run_optimization;

let e = evaluate_checked(&calc, &start, &EvalRequest::ENERGY_GRAD)?;
let settings = OptimizerSettings {
    methods: vec!["rsirfo_bfgs".into()],
    rigid_body_invariant: calc.is_rigid_body_invariant(),
    ..OptimizerSettings::default()
};
let mut opt = GeometryOptimizer::new(settings, start)?;
let outcome = run_optimization(&mut opt, &calc, &mut [])?;   // observers: &mut [&mut dyn Observer]
```

**3. Register it so that every command finds it by name.** The factory receives the
program-independent options (`-func`, `-bs`, `-calc_opt key=value`, ...) and the part of the
name after `:`.

```rust
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

let mut registry = CalculatorRegistry::native();          // lj, emt, eam, meam, ...
registry.register("harmonic", move |req| {
    let k = req.options.extra.iter().find(|(key, _)| key == "k").map_or(0.5, |(_, v)| v.parse().unwrap_or(0.5));
    Ok(Box::new(Harmonic { x0: x0.clone(), k }))
});
let calc = registry.create("harmonic", &CalculatorOptions::default(), &start)?;

// Commands with this registry: `-os harmonic -calc_opt k=0.8`
let config = mopt_rs::cli::optmain::config_from_cli(&["water.xyz", "-os", "harmonic"])?;
mopt_rs::cli::optmain::run_with(&config, base_dir, None, &registry)?;
mopt_rs::cli::run_command_with("optmain", &args, &registry);   // what the binaries do
```

**4. Restartable evaluations in a driver of your own.**

```rust
use mopt_rs::workflow::journal::RunJournal;

let journal = RunJournal::open(&run_dir)?;          // replays what is already recorded
let calc = journal.wrap("main", &calc);             // records everything else
let outcome = run_optimization(&mut opt, &calc, &mut [])?;
let (replayed, unused) = journal.replay_counts();
```

**5. Plain variables.** An objective is anything that implements `Objective`
(`x -> value, gradient, Hessian`); `VectorOptimizer` is the ask/tell optimizer and
`run_vector_optimization` the loop.

```rust
use mopt_rs::opt::vector::{VectorOptimizer, VectorSettings};
use mopt_rs::pes::objective::{MullerBrown2D, ObjectiveRegistry};
use mopt_rs::workflow::run_vector_optimization;

let settings = VectorSettings { methods: vec!["rsirfo_bfgs".into()], ..VectorSettings::default() };
let mut opt = VectorOptimizer::new(settings, faer::Col::from_fn(2, |i| [-0.4, 1.3][i]))?;
let outcome = run_vector_optimization(&mut opt, &MullerBrown2D::default(), &mut |_report| Ok(()))?;

let objective = ObjectiveRegistry::builtin().create("expr:(q1-1)^2 + q2^2", &options)?;
```
