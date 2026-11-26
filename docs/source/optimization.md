# Options of the optimizations

The options of `optmain` that choose how a structure is optimized and what an optimization
leaves; the optimizations inside the other commands take the same ones. The coordinate systems
have a page of their own ([Coordinate systems](coordinate_systems.md)), and so do the bias
potentials ([Bias potentials](bias_potentials.md)).

## Optimization methods (`-opt`)

A method string names one step generator and any number of step enhancers; it is matched as in
MultiOptPy (exact names first, then quasi-Newton families `<family>_<updater>`, then substrings;
an unknown name falls back to FIRE with a warning). Two strings blend two methods by the RMS
force. Differences from MultiOptPy are listed in [migration_optimizers.md](migration_optimizers.md).

Without `-opt` a minimum is looked for with `rsirfo_block_fsb` and a saddle point (`-order 1`
and above) with `rsirfo_block_bofill`; the same holds for the optimizers of the library
when no method is named (Rust: `opt::registry::default_methods`; Python: `Optimizer` and
`VectorOptimizer` without `methods`). MultiOptPy's default, `FIRELARS`, is one of the names.

| Step generators | Names |
|---|---|
| inertial | `fire`, `fire2`, `abcfire` |
| steepest descent | `gradientdescent` (`steepest_descent`), `mwgradientdescent` (mass-weighted) |
| conjugate gradients (line search over the iterations) | `cg`, `cg_pr`, `cg_fr`, `cg_hs`, `cg_dy` |
| adaptive moments | `adam` (with Lookahead), `adamlars`, `adamlookahead`, `adamlookaheadlars`, `radam`, `ranger` (RAdam with Lookahead), `rangerlars`, `adabelief`, `eve`, `prodigy` |
| limited memory | `lbfgs`, `tr_lbfgs` (trust region) |
| Gaussian process, minima and saddle points (`-order`; saddle points need a Hessian: `-fc`, `-modelhess`) | `gpmin` |
| quasi-Newton, minima and saddle points (`-order`) | `rsirfo_<updater>`, `crsirfo_<updater>`, `rsprfo_<updater>`, `mf_rsirfo_<updater>` (mode following; `smf_`, `mwmf_`, `mwsmf_`; options `:index:ema0.5:grad0.3`), `newton_<updater>`, `mnewton_<updater>`, `ccqn` |
| high-index saddle dynamics, saddle points only (`-order 1` and above; the directions from the optimizer's Hessian, exact with `-fc 1`; block Bofill unless another updater is named; [high_index_saddles.md](high_index_saddles.md)) | `hisd`, `ggad` |

Updaters: `bfgs`, `fsb`, `bofill`, `msp`, `sr1`, `psb`, `flowchart`, their `cfd_`, `_dd`, and
`block_` forms (for example `rsirfo_block_fsb`).

| Step enhancers (substrings of the method string) | Effect |
|---|---|
| `lars`, `lookahead` | layer-wise rate scaling; slow/fast points |
| `linesearch` | strong Wolfe line search along the step, one trial point per iteration |
| `gdiis`, `c2diis`, `kdiis`, `ediis`, `adiis`, `gediis` | DIIS extrapolation and interpolation |
| `coordinate_locking`, `component_wise_scaling` | work on a subset of the coordinates; per-coordinate factors (first-order methods) |
| `geodesic_step` | follow the geodesic of the bond-length metric |
| `gpr_step`, `gan_step`, `rl_step` | step refinement on a Gaussian-process model; learned step scaling |
| `trim` | trust-region image minimization for steps beyond the trust radius (Hessian-based methods) |
| `perturbation` | annealed thermal noise |

For example `-opt rsirfo_block_fsb_gdiis`, `-opt lbfgs_linesearch`, `-opt cg_pr`,
`-opt fire2 rsprfo_bofill`. Every method can also be used on its own: `mopt_rs.StepMethod`
(Python) and `opt::stepper::Stepper` (Rust) propose steps for points supplied by the caller,
and the generators, enhancers, and their kernels are public types of `mopt_rs::opt` and
`mopt_rs::math`.

After each applied quasi-Newton update, `optmain` writes the update method
(`Hessian update: NAME`) and the eigenvalues of the updated Hessian to `mopt.log`, in
ascending order. These curvatures are in the optimizer's coordinates, before adding bias
curvature or applying step projections. Variable-cell runs include the scaled lattice
coordinates. The spectrum follows the step radius, with eight decimal places and the cutoff
`|λ| > 1e-10`. One spectrum is written per iteration: in an iteration with an exact Hessian
(`-fc N`) it is that of the exact Hessian with the translations and rotations taken out, and
the updated spectrum is not written (the update line still is); otherwise it is that of the
updated Hessian. (Up to 2.1.0 both were written one after the other under the same heading, the
updated one first and with the rigid-body curvatures in it.) Vibrational frequencies are
reported separately.

### The rigid-body motions of updated Hessians (`-update_proj`)

A quasi-Newton update fits the Hessian to the pair of the last step and the change of the
gradient. The pair joins two structures, and it is not orthogonal to the translations and
rotations of the newer one, so every update puts curvature along them, which a surface that
does not change under rigid motions does not have. With `-update_proj on` (the default) the
kept Hessian is projected after every update, `H <- P H P` with `P = I - D D^T` and `D` the
orthonormal rigid-body basis of the structure the update arrived at (`math::projection::
project_out_basis_mat`; `opt::update::project_out_rigid`). An exact Hessian is projected the
same way when it is taken over, whatever the option says. The Hessian that is projected is the
one that is kept: the next update starts from it, the trust radius predicts the energy change
with it, the log shows its spectrum, and later stages that take the Hessian of a run (the
optimizations after a chain, `relaxedscan`) get it. A Hessian whose update was skipped (a pair
with `s . y <= 0`, or one undone by the guard on the negative curvatures of a saddle search)
is projected with the newer structure as well. `-update_proj off` keeps the updated Hessians
as the formulas give them, as mopt-rs 2.1.0 and MultiOptPy do (the exact Hessians are still
projected).

The rigid-body basis is that of the step generators: three translations and three rotations,
two rotations for a linear structure, the translations only in a periodic cell (and the
rotation about the axis of a wire). The option applies to

- the optimizer of structures (`optmain` and everything built on it: TS searches, crossing
  points, ONIOM, `run_autots`, `run_mapper`, `ieipmain -mf`, `confsearch`, `relaxedscan`, ...)
  and the Hessian `mnewton` keeps for itself;
- variable-cell runs (`-vc`): the rigid motions of the atoms and the cell
  (`VariableCell::rigid_modes`, restricted as for the exact Hessian where `-fix` or `-vc_fix`
  pin translations or rotations);
- the quasi-Newton steps of the images of a chain (`nebmain`, the chains of the two-structure
  methods and of the mapper), each image with its own basis;
- the IRC (`ircmain`, `optmain -irc`): the Hessian of each side and that of the hypersphere
  minimizations of GS2 and of the first step, in mass-weighted coordinates;
- the LQA steps of DS-AFIR (the rigid motions and the fixed variables, as its steps take them
  out);
- `mopt_rs.StepMethod` / `opt::stepper::Stepper` when a basis is given (`rigid_basis=` of
  `propose`; none by default).

Nothing is taken out where there is nothing to take out or where it would be wrong:

- plain variables (Muller-Brown and the other objectives of `-os`), and surfaces that change
  under rigid motions (`Calculator::is_rigid_body_invariant` is false, or a bias potential is
  not invariant): their curvature along those directions is real;
- constraints that pin the frame (`-fix`, fixed Cartesian components, the frozen atoms of a
  chain): their own projection takes care of the frame, as it did before. In the IRC, taking the
  rotations about a fixed atom out of every update spoilt the path of HCN with its carbon atom
  fixed (17 and 28 steps of higher energy against 0 and 2), so the IRC follows the optimizers
  here;
- internal coordinates (`ric`, `dic`): they hold no rigid motions. `tric` has the translations
  and rotations of every fragment among its coordinates, but no basis of the rigid motions of
  the whole structure in them; its updated Hessians are kept as they are (an open issue, see
  [open_issues.md](open_issues.md));
- the `hessian_update` tool of the MCP server, which returns what a formula gives for the
  matrices it is handed.

The model Hessians are unchanged: the rigid-body curvature some of them add on purpose, and
their projection away from periodic cells, stay as they were; the first update projects them.
The initial identity matrix of a run without a model Hessian likewise stays until the first
update.

Measured on the IRC of HCN with GFN0-xTB (`ircmain -irc_step 0.1`, LQA, from the transition
state of `optmain -order 1 -fc 1 -tcc`): with the option on, no step was taken again on either
side (a point of higher energy, or one past the floor of the valley); with it off, 4 steps
without constraints and 5 with the C-N distance held (`-pc bond 1,2`). With the carbon atom
fixed (`-fix 1`) nothing is projected and both settings take the same path.

## Coordinate systems (`-coord`)

| Name | Coordinates |
|---|---|
| `cart` (default) | Cartesian |
| `ric`, `dic` | redundant and delocalized internal coordinates |
| `tric`, `tric-p` | translation-rotation internal coordinates (delocalized, and their primitive set): internal coordinates inside each fragment, translation and rotation of each fragment |
| `tric-cart`, `tric-cart:WEIGHT` | TRIC stacked with the Cartesian coordinates times a weight (0.1), delocalized |
| `auto` | `tric-cart`; Cartesian coordinates for atoms without bonds, periodic networks, dense bonding, and more than 300 atoms; chosen again when fragments join or part |

Calculators always receive Cartesian structures. Coordinates that break down during a run
(a fragment turned by 90 degrees, an angle that became linear, lost rank) are set up anew at
that structure. The measurements behind the default weight and the rule of `auto` are in
[coordinate_systems.md](coordinate_systems.md): with a model Hessian (`-modelhess`), internal
coordinates take about 40 % fewer iterations than Cartesian ones for molecules and complexes.

## Starting from a Hessian in a file (`-ih`)

```text
optmain  ts_guess.xyz -os ... -order 1 -ih guess.hess     # ORCA .hess
optmain  start.xyz    -os ... -ih hessian.npy             # numpy.save
ircmain  ts.xyz       -os ... -ih ts.fchk                 # Gaussian formatted checkpoint
freqmain x.xyz        -os ... -ih hessian.txt             # numbers only
```

`-ih FILE` gives the Hessian a run starts from: the first Hessian that is asked for (at
the first point of an optimization, at the saddle of `ircmain`, at the structure of
`freqmain`) is the one of the file, with the energy and the gradient from the calculator.
The file is a NumPy file of a square matrix, an ORCA `.hess` file, a Gaussian formatted
checkpoint file or output (a frequency job with `#P`), or numbers only (the square row by
row or its lower triangle; lines that begin with `#` are skipped). It is in Cartesian
coordinates, Hartree/Bohr^2, with the atoms in the order of the structure (what ORCA and
Gaussian write; a NumPy or text file has to be in these units), and for the structure the
run starts from: nothing checks that. In an optimization the Hessian is updated from
there; `-fc N` still asks the calculator every `N` iterations (not at the first point),
and a model Hessian (`-modelhess`) is not used together with `-ih`. Not with `-mf`,
`-vc`, or plain variables. From Rust: `io::hessian_file::read_hessian_file`,
`pes::given_hessian::GivenHessian` (any calculator, its first Hessian from a matrix).

## Following a given direction (`-follow`)

```text
optmain guess.xyz -os gfn1xtb -order 1 -fc 5 -follow tangent.xyz
```

`-follow FILE` (an xyz file of displacements, any length, Cartesian) makes a
transition-state search take the Hessian mode closest to that direction uphill and then
track it by overlap, instead of the lowest mode: the RS-I-RFO methods become their
mode-following variants (`mf_rsirfo_*`). The direction can be the tangent of a chain at the
image the search starts from. With ONIOM microiterations, a file of the whole system is cut
to the core.

It is off by default, also with ONIOM. Measured on the transition-state candidates of AutoTS,
with `-order 1 -fc 5 -tcc -tr 0.2 -ns 300` and the chain tangent at each candidate:

| candidate | without `-follow` | with `-follow` |
|---|---|---|
| aldol reaction, candidate 3 (GFN1-xTB) | 17 iterations, one imaginary frequency (-254 cm^-1) | 300 iterations, not converged |
| aldol reaction, candidates 1 and 2 | 300 iterations, not converged | 300 iterations, not converged |
| Claisen rearrangement in water, candidate 1 (ONIOM GFN1-xTB:GFN-FF) | 15 iterations, -417 cm^-1 | the same run (the tangent picks the lowest mode) |

## Stationary points of the order asked for (`-modekill`)

An optimization ends where the gradient vanishes. A search for a transition state (`-order
1`) can end at a saddle point of order 2, by symmetry or because the Hessian in use never
showed the second mode, and a minimization can end on a saddle point. With `-modekill` (off
by default) the Hessian is requested at the converged structure, its imaginary modes are
counted, and when there are more than `-order` asks for, the structure is moved by 0.1 Bohr
along the surplus modes (the lowest `-order` ones are kept) and the optimization goes on from
there with that Hessian. A mode is imaginary where the eigenvalue of the mass-weighted
Hessian (the masses of the atoms; translations and rotations projected out in the mass-weighted
space) is below minus 1e-8 Hartree/(Bohr^2 amu), MultiOptPy's `saddle_check`, about 0.51
cm^-1: the criterion of the IRC, of the mapper, and of the count of imaginary modes of `-freq`
(`normal_modes.txt`, the summary of a run, confsearch) as well, so a mode as flat as that is
tolerated and one of 2 cm^-1 is not. In coordinates other than Cartesian without constraints
the Cartesian Hessian of the surface is examined and the step goes in Cartesian coordinates;
plain variables have unit masses. The log names the wavenumber of every mode it leaves. This is repeated at every convergence, at most ten times. The log
has a line for every check that changes something. It needs Hessians from the calculator (its
own, finite differences, or `-modelhess o1numhess`); a structure with fewer negative
curvatures than asked for is reported and accepted. In Python: `Optimizer(...,
mode_kill=True)` and `optimize_variables(..., mode_kill=True)`.

For a standalone minimization, `optmain input.xyz -modekill` uses the selected
minimizer up to its first stationary point. If excess negative curvature is
detected, the displaced point descends with order-zero HiSD, then undergoes a
new Hessian check. An already verified minimum ends at the first check.
Use `-modekill -modekill_method classic` to displace along the same modes and
continue with the originally selected optimizer. Use `-modekill -modekill_method hisd`
to select HiSD explicitly. At positive order, the default is classic. Explicit HiSD follows the retained unstable directions after the displacement. The method option alone does not enable mode kill.
The mapper has its own classification of IRC endpoints.

Example (`tests/numerical_safeguards.rs`): ethylene placed symmetrically at peroxyacetic acid
converges, with GFN0-xTB and `-order 1`, to a saddle point of order 2 with both O-C distances
at 1.86 Angstrom; with `-modekill` the run goes on to the transition state of the epoxidation
(1.79 and 2.06 Angstrom, one negative curvature).

## Stress of periodic cells

In a periodic run, each Cartesian geometry in the optimization log is followed
by a blank line and the current lattice vectors `a`, `b` and `c`. Their Cartesian
components use the atomic coordinate columns and are in Å. The `pbc` flags
below them refer to those three vectors in order.

For a periodic cell the built-in potentials return the stress with every evaluation
(`EvalOutput::stress`; for slabs and wires per volume of the stored cell, so that only
`V sigma` means something): `sigma = (1 / V) dE / d(strain)` in
Hartree/Bohr^3, negative under compression, which is the convention of ASE. EMT, EAM, MEAM,
Tersoff, Lennard-Jones with a cutoff, and GFN-FF in a three-dimensional cell have it
analytically. A calculator for which the
stress costs extra is asked for it only where it is needed (`EvalRequest::need_stress`,
set by the variable-cell optimization): an ASE calculator is asked with `get_stress()`
before the energy and the forces, so that a program run through files computes all three
in one run, and Quantum ESPRESSO gets `tstress` in the input of those runs. From Python,
`EngineCalculator.evaluate_structure(structure, need_stress=True).stress`; an engine of
its own declares `compute(self, structure, hessian, stress=False)` (and, to give the energy
alone more cheaply, `gradient=True`, called with `gradient=False` for the energies of the
other states of surface hopping) and puts the stress in
`EngineResult.extra` (`stress_hartree_per_bohr3`, 3 x 3, or `stress_voigt_ev_per_a3`,
the six components of ASE). For a calculator without a stress
`pes::stress::stress_of` takes it from differences of the energy (twelve energies).
`pes::stress` converts to and from the six components of ASE (`xx, yy, zz, yz, xz, xy` in
eV/Angstrom^3) and gives `dE / dh` for the lattice vectors. From Python:
`_core.native_stress(name, symbols, positions_bohr, cell_bohr)`,
`_core.stress_to_ase_voigt`, `_core.stress_from_ase_voigt`.

## Variable cell (`optmain -vc`)

`-vc` optimizes the lattice vectors together with the atoms, with any method of `-opt`:

```text
optmain cu.xyz -os eam -vc -opt lbfgs                 # atoms and cell
optmain cu.xyz -os eam -vc -vc_pressure 20            # under 20 GPa: the enthalpy E + P V
optmain slab.xyz -os eam -vc                          # pbc="T T F": the two periodic vectors
optmain ts.xyz -os eam -vc -order 1 -fc 8 -opt rsirfo_bofill    # a saddle point
optmain cu.xyz -os eam -vc -vc_afir 200 a,b,c         # compress the cell (periodic AFIR)
```

The variables are those of Gubler, Krummenacher, Huber, and Goedecker (J. Comput. Phys. X
2023, 17, 100131): the atoms in quasi-Cartesian coordinates, `q_i = s_i h_0` (the
fractional coordinates `s_i` of the atom multiplied with the cell `h_0` of the start, so
that they are lengths and do not change when the lattice changes), and the lattice vectors
scaled to `A_k = w sqrt(N) a_k / |a_k0|`. `-vc_weight` sets `w` (default 2).

- The stress comes from the calculator (EMT, EAM, MEAM, Tersoff, Lennard-Jones with a
  cutoff have it analytically; any ASE calculator with a stress, for example
  `-os ase:ase.calculators.emt:EMT`, `-os gpaw` with plane waves, `-os espresso`) or from
  differences of the energy. The stress of a plane-wave program leaves out the change of
  the plane waves with the cell: with a low cutoff the cell that is found is not the one
  of lowest energy (GPAW, silicon: the stress and the slope of the energy differ by 2 %
  with 250 eV and 0.7 % with 400 eV; Quantum ESPRESSO, hydrogen with a hard
  pseudopotential: a factor of four with 30 Ry, 0.2 % with 120 Ry).
- Slabs and wires (`pbc="T T F"`, `"F F T"`): the two lattice vectors of a slab vary
  within their plane and the one of a wire along its line, as in Takagi et al. (Phys. Rev.
  B 2017, 95, 184110); the other vectors of the cell stay. A pressure needs three periodic
  directions.
- `-fc N` and `-order`: the Hessian in the variables of the atoms and the cell, with the
  translations and rotations of the whole structure taken out. Where the calculator and the
  bias potentials have analytic Hessians it is analytic: the block of the atoms from the
  Cartesian Hessian, and the columns of the lattice from the second derivatives along the
  strain of the cell (`domain::StrainHessian`: those of the energy in the strain, and those
  of the gradient; `opt::cell::VariableCell::lattice_columns`), which EMT, EAM, MEAM,
  Tersoff, the Lennard-Jones potential, AFIR, and the forces on the cell (`-vc_afir`) give.
  Where a part has a Cartesian Hessian but not those, the columns of the lattice come from
  differences of the gradient (two evaluations per lattice variable), and every column does
  without a Cartesian Hessian.
- Bias potentials (`-ma` and the others) act as usual; their share of the stress is their
  own where they give one (AFIR), from differences of their energy otherwise.
- Electronic states: `-es N` optimizes state `N` with its own stress (by differences of its
  energy where the calculator gives none). `-mf` combines two states in the variables of
  the cell: the enthalpy `E_k + P V` of every state and its gradient in the variables (from
  its energy, gradient, and stress) go into the model function as the energies and
  Cartesian gradients of the states do in a fixed cell, so that the cell feels the
  combined stress. The gradient projections (`-mf opt_mesx` for a crossing of two spin
  states, `-mf opt_meci` for a conical intersection) project the components of the lattice
  with those of the atoms (the difference gradient and the branching plane span all the
  variables, with the metric of the variables), and the penalty functions (`seam`,
  `conical`, `avoiding`) weigh the stresses of the states by their derivatives. `P V` is
  the same in both states, so the gap is that of the energies and the mean is the mean
  enthalpy. The log shows the energies of both states, their gap, and the norm of the
  difference of their gradients at every iteration, and the stress there is the mean of
  the two. The Hessian of two states comes from differences of the combined gradient. Not
  with `-vc`: `-projection_trust merit` (the steps are judged by the function that is
  minimized), the check of the branching plane at convergence (that of a fixed cell), and
  `-irc` after `-mf`. A crossing in a fixed periodic cell is the same `-mf` without `-vc`.
- `-vc_afir GAMMA DISTANCES` puts an artificial force on the cell: the AFIR function over
  distances between the origin and the tips of the lattice vectors (`a`, `b`, `c`) and
  between two tips (`ab`, `bc`, `ca`). A positive collision energy (kJ/mol) shortens them,
  a negative one lengthens them; ramps as for `-ma` (`0:300`). One distance per
  `-vc_afir` is the force on a pair of dummy atoms of Takagi et al.; several distances in
  one list are weighted as the atoms of two fragments are.
- The log shows the lattice vectors (Angstrom), the volume, the stress, and the pressure
  (atomic units, and GPa) at every iteration; the files of the run carry the cell in
  their comment lines (`Lattice="..."`). `BIAS ENERGY` is the function that is minimized:
  the energy with the bias potentials, the forces on the cell, and `P V`.
- Constraints (`-fix`, `-pc bond|fbond|angle|dihedral`, `-gfix`, `-uc`) are held in the
  variables (`opt::cell_constraints`): frozen atoms keep their fractional coordinates
  (their `q_i` are held), and distances, angles, dihedrals (minimum images), and `-uc`
  functions are functions of the structure the variables describe, whose gradient in the
  variables is mapped as that of the energy (the Cartesian gradient for the atoms, the
  derivative along the strain at fixed fractional coordinates for the lattice, analytic for
  the built-in functions). They are projected out of the gradient and the step and restored
  after every step, as for plain variables (frozen variables assigned, the others to their
  rounding floor, as below); a distance holds to 1e-12 Bohr while the cell relaxes, and the
  constrained optimum agrees with that of a stiff spring on the same
  distance (test `constraints_matrix_optmain`). With frozen atoms the translations of the
  atoms are no longer taken out of the Hessian (the rotations of the lattice, which carry the
  atoms with their fractional coordinates, still are); with a bias at a point in space
  neither is. Refused, with the reason: Cartesian components (`-pc x|y|z`), oriented groups
  (`-pc rot`), the fixed directions of `-pc eigvec` and `-pc atoms_pair`, `-uc` functions of
  `x()`, `y()`, `z()` (the lattice turns freely, so a point or direction in space has no
  meaning), and `-irc` after a constrained `-vc` (the IRC of the atoms and the cell has no
  constraints).

### Constraints on the cell

The cell itself can be held, in two spellings that are the same constraints: the dedicated
option `-vc_fix` is translated into the expressions of `-uc` below (`cli::constraints::
cell_fix_expressions`), so that `-vc_fix volume` and `-uc "volume()"` give bit-identical runs
(test `variable_cell_constraints`).

```text
optmain cu.xyz -os eam -vc -vc_fix volume             # constant volume: only the shape relaxes
optmain cu.xyz -os eam -vc -vc_fix shape              # constant shape: only the scale relaxes
optmain cu.xyz -os eam -vc -vc_fix a=3.70,gamma       # a at 3.70 Angstrom, gamma as in the input
optmain slab.xyz -os eam -vc -vc_fix c_vector,az,bz   # slab in a 3D cell: c and the plane kept
optmain cu.xyz -os eam -vc -uc "cell_a()/cell_b()"     # any function of the cell and the atoms
relaxedscan cu.xyz -os eam -vc -scan cell volume 40,52 -nsample 9    # energy-volume curve
```

| `-vc_fix` item | `-uc` expression | unit of `NAME=VALUE` |
|---|---|---|
| `a`, `b`, `c` | `cell_a()`, `cell_b()`, `cell_c()` (lengths) | Angstrom |
| `alpha`, `beta`, `gamma` | `cell_alpha()` (`b`, `c`), `cell_beta()` (`a`, `c`), `cell_gamma()` (`a`, `b`) | degree |
| `volume` | `volume()` | Angstrom^3 |
| `shape` | `cell_b()/cell_a()`, `cell_c()/cell_a()`, `cell_alpha()`, `cell_beta()`, `cell_gamma()` | (no value) |
| `a_vector`, `b_vector`, `c_vector` | `cell_ax()`, `cell_ay()`, `cell_az()`, ... (the lab-frame components) | (no value) |
| `ax`, `ay`, ..., `cz` | one component | Angstrom |

Without a value the item holds the value of the input cell. The expressions are in atomic
units (Bohr, radian, Bohr^3; `ang` and `deg` convert, `-uc "volume() = 1150*ang^3"`), and
they are functions of the language of `-ubias` (`pes::bias::expression`), so they combine
with the geometric functions of the atoms (`-uc "dist(1,2) / cell_c()"`) and are scan
targets of `relaxedscan` (`-scan cell NAME START,END` with the names and units of the table,
or `-scan expr "volume()" START,END` in atomic units).

- **Derivatives.** An expression that uses the cell is evaluated by `pes::bias::CellExpression`:
  forward-mode automatic differentiation in the coordinates of its atoms and the nine
  components of the strain together, so that the gradient at a fixed lattice, the stress
  (`sigma_ab = (1/V) dF/d eps_ab`), and with a Hessian the Cartesian Hessian and the second
  derivatives along the strain are exact. The stress is not made symmetric: a lab-frame
  component of a lattice vector depends on the orientation of the cell, and the antisymmetric
  part of its stress enters the gradient with respect to the lattice vectors
  (`opt::cell::lattice_gradient`, which is the gradient of the energy for a symmetric stress).
  The constraints are held in the variables of the cell as the other constraints of `-vc`
  (projected out of the gradient and the step, restored to their rounding floor after every
  step). With exact Hessians (`-fc`) the curvature of the constraints that give it (the
  functions of the cell) is subtracted from the Hessian, `H - sum_k lambda_k d2c_k`, the
  Hessian of the Lagrangian; the eigenvalues of the Hessian in the log are those on the
  constraint surface (the held directions taken out with the rigid motions).
- **Constant shape.** `shape` holds the two ratios of the lengths and the three angles, which
  determine the metric up to its scale: the gradients of these five functions are
  independent for any cell that is not degenerate (no angle near 0 or 180 degrees), and each
  is of the order of `1/|a|` per Bohr, as those of the lengths and angles are, so the
  constraint matrix is as well conditioned as that of the lattice parameters themselves.
  What is left is the isotropic scaling (and the positions of the atoms).
- **A lattice vector in space.** Its Cartesian components are held in the lab frame. The
  rotations of the lattice are then no symmetry of the constrained problem: they are not
  taken out of the Hessian (`opt::cell_constraints::pins_rotations`); the rotation about the
  held vector, which is still free, keeps the zero curvature it has on any surface, and the
  others are along the held directions, which the projection removes.
- **At convergence** the stresses conjugate to the free degrees of freedom of the cell
  vanish and those conjugate to the held ones need not: with constant volume the deviatoric
  stress vanishes and the pressure is minus the multiplier of the volume; with constant shape
  the pressure is the external one (`-vc_pressure`: the enthalpy is minimized on the
  constraint surface); with a length held only the force along that vector is left. The
  constrained optimum agrees with that of a stiff penalty on the same function (test
  `variable_cell_constraints`).
- **Combinations.** With the constraints of the atoms (`-fix`, `-pc`, `-gfix`, `-uc`), with
  `-vc_pressure`, `-vc_weight`, `-es`, the penalty functions of `-mf` (`seam`, `conical`,
  `avoiding`: near crossings with the cell held), `-order 1`, and `-fc`. The gradient
  projections of `-mf` (`opt_mesx`, `opt_meci`) with constraints under `-vc` are refused: the
  projection is formed in all the variables of the cell before the constraints are projected
  out, and its parts can cancel away from the seam (measured: `opt_mesx` with the length of
  `a` held stopped as converged with a gap of 2.4e-3 Hartree). An artificial force on a held length (`-vc_afir` on `a` with `-vc_fix a`) is
  taken up by the constraint (its multiplier); it is not refused, the length stays, and the
  force acts on the free part of the cell. `-irc` after a constrained `-vc` stays refused.
- **Refused, with the names.** Without `-vc` (`-vc_fix`, `-uc` of a function of the cell,
  `-scan cell`: in a fixed cell they cannot change); a set that is contradictory or
  redundant (a function whose gradient in the lattice variables is a combination of those
  before it: the volume with the three lengths and three angles, a length held twice, more
  functions than the cell has degrees of freedom); a set that leaves the cell no freedom
  apart from its rotation (`shape` with a length, the six lattice parameters, `c_vector` with
  `shape`), with the advice to drop `-vc`; in a slab or a wire a function of a lattice vector
  that is not a variable (the volume, `c`, the angles with `c`: only the periodic vectors
  change there); a value for `shape` or `*_vector`; `-scan cell` with `relax=off`.
- **Measured** (test `variable_cell_constraints`, 32 argon atoms with the Lennard-Jones
  potential, the largest residual at any point the optimizer asked for, 65 to 101 iterations):
  a length 7.1e-15 Bohr, an angle 0, the volume 2.0e-11 Bohr^3 (of 8.5e3), the shape 1.3e-15,
  a lattice vector 2.8e-17 Bohr, one component 0 (a component is linear in one variable).
  From the files (trajectory, final structure; test `constraints_matrix_optmain`): at most
  5.5e-12 Bohr^3 for the volume and 7.1e-15 for the others. The energy-volume curve of
  `-scan cell volume` (five points, 0.96 to 1.04 of the optimum) has its parabolic minimum at
  1219.67 Angstrom^3 against 1218.93 of the unconstrained `-vc` optimum (0.06 %), every point
  on its volume to 2.6e-11 Bohr^3 (test `constraints_matrix_relaxedscan`).

From Rust: `opt::cell::VariableCell` (the map between a structure and the variables, the
gradient of the enthalpy, the rigid modes), `opt::cell::CellForce`, and
`opt::cell::numerical_hessian`; the variables go to `opt::vector::VectorOptimizer`.

## Constraints and bias potentials in a run

- **Resolved once.** `optmain` resolves its constraints (`-fix`, `-pc`, `-gfix`, `-uc`) at
  the input structure: every target value, frozen position, and lab-frame direction is fixed
  there and every stage of the run holds the same values (the optimization and a restart of
  it, `-irc`, `-freq`, `-bias_derivatives`). Eigenvector constraints are resolved at the
  first Hessian of the optimization; `GeometryOptimizer::resolved_constraints` hands the
  resolved set to the stages that follow. A set that is already resolved (from another
  command) passes through unchanged.
- **Held to 1e-10.** A hard constraint holds the value of the input (or the `-pc manual`
  value) within 1e-10 Bohr or rad (`pes::constraint::GUARANTEE`) in every structure the
  optimizer hands out, and in practice to the rounding of double precision; restraints by
  bias potentials (`-kp` and the others) are harmonic and may deviate. A restore that cannot
  reach 1e-10 is an error naming the constraint and the residual, never accepted silently;
  the one exception is a ReSPer coordinate (`-scan resper`, computed through a superposition
  with a noise of about 1e-9), for which the run goes on with one warning line per
  constraint naming the residual reached and the bound, and the largest residual at the end
  of the run. After every step (its trust-radius cap, back-transformation from `-coord`, and
  the superposition on the previous structure, in that order) `ConstraintSet::restore` is
  the last change of the structure: oriented groups are turned back rigidly, frozen atoms
  and Cartesian components (`-fix`, `-pc x|y|z`) are assigned, bit-identical to their
  targets and left out of the Newton steps (their columns of the Jacobian are zero), and the
  other constraints are solved together by Newton steps until every residual is at its
  floor `4 eps (sum_i |dc/dx_i| |x_i| + |c|)` (`eps` the machine epsilon: the rounding of the
  coordinates a step writes and of the evaluation of `c`; about 1e-13 Bohr for a distance
  with coordinates of 10 Bohr, a few 1e-14 rad for an angle or a dihedral). Near the floor
  a step that no longer decreases the largest residual (in units of its floor) ends the
  restore once every residual is below 1e-10. A user function (`-uc`, Python, Rust terms)
  whose value repeats only to a larger noise has its noise measured once at the start (six
  displacements of 1e-9 Bohr against the first-order prediction) and twice that as its
  floor; the log has one line naming it and the floor (a floor above 1e-10 means an error at
  the first restore, except for ReSPer coordinates). Measured (13 argon atoms, ten constraints of every kind,
  test `constraints_floor`): every structure of RS-I-RFO, FIRE, `-coord dic`, and a periodic
  cell within 2.2e-14 of the targets (before: up to 9.9e-11, the old tolerance 1e-10); a
  restore from a displacement of 0.05 Bohr per coordinate takes 3 Newton steps and about
  0.2 ms (before: 0.74 ms, the Hessians of the constraints were computed at every step;
  now one evaluation of each function per step, without its Hessian).
- **Precision of the files.** Every structure file (the trajectories, the path files of
  chains, the final structures, the structures of `confsearch`, `orientsearch`,
  `relaxedscan`, and `ieipmain`, the IRC paths and ends, the Gaussian input of `io`) prints
  12 decimals of Angstrom (a rounding of up to 9.4e-13 Bohr per coordinate: a distance or an
  angle recomputed from a file agrees to a few 1e-12), and the lattice of extended xyz files
  15 decimals (5e-16 Angstrom; 12 would move the volume of a cell of 10^4 Bohr^3 by about
  1e-9 Bohr^3), so that the guarantee of 1e-10 holds for the values recomputed from the
  files as well (tests `constraints_matrix_*`). Files written with fewer decimals are read as
  before. The values in memory (the results of the API) are those of the restore.
- **A tolerance of its own.** `-uc "EXPR; tolerance = T"` holds one user-defined function to
  `T` instead of 1e-10 (a function computed through an inner problem whose noise is above
  the guarantee); the log says so once at the start, a restore that cannot reach `T` is an
  error naming the function, and without it a function noisier than 1e-10 stays an error.
  From Rust: `ConstraintFunction::tolerance`, and `ConstraintFunction::warns_only` for the
  functions whose residual above the bound is a warning only (the ReSPer coordinates set it;
  it used to be recognised by the name of the term).
- **`-gfix ATOMS ...`**: for every group, the distances between its atoms are held as
  `-pc bond` holds one (projection of the gradient and the Hessian, restore after every step).
  MultiOptPy parses it in nebmain and ieipmain only and does not use it; here it is also an
  option of `optmain` and of the commands built on it (`ircmain`, `freqmain`, `confsearch`,
  `relaxedscan`, `orientsearch`), so that the follow-up optimizations of a chain keep it.
- **Rigid-body motion.** The optimizer recentres and superposes every structure and takes
  the rigid modes out of its steps only where the surface does not see them: the calculator
  is invariant and so is every bias potential (`pes::term::frame_invariant`). A bias at a
  point or along a direction in space (`-vpp`, `-wwp`, `-vpwp`, `-fp`, `-nrp`, `-ubias` with
  `x()`, `y()`, `z()`) keeps the structure in the lab frame; constraints that pin the frame
  do the same. An optimizer created to remove rigid motion refuses a term that depends on
  the frame, with its name. Constraints that pin the frame need coordinates with a position
  and orientation (`cart`, `tric`): `-coord dic` with `-fix` is refused (the projection of a
  frozen atom in internal coordinates takes away its internal motions and the run stalls).
- **Bias Hessians.** The trust radius compares the change of the energy with the prediction
  of the model of the step, so the Hessian of the bias potentials is part of the prediction
  (argon trimer, a spring of 0.5 Hartree/Bohr^2, exact Hessians: the ratios of the first four steps were 0.64, 0.52, 0.62, 0.66 without it and are 1.000, 1.000, 0.96, 0.81 with it; test
  `optmain_constraint_bias_defects`). The Hessians of the terms are computed only where a
  method or an enhancer reads them (the methods with a Hessian, GPMin, TRIM) or the iteration
  asks for a Hessian anyway; FIRE, the gradient methods, and L-BFGS do without (FIRE on 27
  argon atoms with six bias terms, unoptimized test build: 124 ms per iteration with the term
  Hessians, 18.7 ms without; the trajectories agree to rounding, the gradients of the terms
  from dual instead of hyper-dual numbers).
- **Stationary points** keep two consistent pairs: the energy and gradient of the
  calculator alone (`StationaryPoint::energy`, `raw_gradient`) and of the surface that was
  optimized (`biased_energy`, `gradient`, the one that vanishes).
- **Criteria.** `-tcc` and `-lcc` together are refused. With `-pc` the displacement
  thresholds are four times larger (not with `-lcc`), as in MultiOptPy; the force thresholds
  are those of the criteria.

## Maps of a run and modes of its iterations (`-cmds`, `-pca`, `-km`)

`-cmds` and `-pca` draw the structures of all iterations of a run in two dimensions when
it ends, colored by the energy (and by the energy with the bias potentials) relative to
the first structure; `-km` examines at every iteration how the last ten moved.

- `-cmds`: classical multidimensional scaling. The distance of two structures is
  `sqrt(sum_i m_i |r_i - r'_i|^2)` after the mass-weighted superposition of one on the
  other (Angstrom amu^(1/2)); the structures are placed in a plane whose distances come as
  close to these as a plane allows (the picture of a reaction path of Tsutsumi, Ono, and
  Taketsugu, Chem. Commun. 2021, 57, 11734). `cmds_analysis_result.csv` has the two
  coordinates and the energies of every structure, headed by the shares of the two
  coordinates and by the "dimensional reproducibility" (1 when the distances are those of
  points in a Euclidean space); `cmds_result_visualization_energy` and `_bias_energy` are
  the figures.
- `-pca`: principal components of three sets of variables: mass-weighted Cartesian
  coordinates after superposition on the first structure, all interatomic distances, and a
  chain of distances, angles, and dihedral angles in the order of the atoms (standardized,
  since it mixes lengths and angles). For each set, `pca_analysis_result.log`
  (`_redundant.log`, `_z_matrix.log`) lists the components with the entry of every
  variable, the same name with `.csv` has the structures along the first two components,
  and `pca_result_visualization_*` are the figures.
- `-km`: extended dynamic mode decomposition of the last eleven iterations (coordinates,
  their squares, interatomic distances, and gradient as observables). The eigenvalues say
  how the optimization moves: a modulus below 1 decays, a negative real part alternates in
  sign from step to step, an imaginary part rotates. `koopman_analysis/koopman_eigs.csv`
  has them for every iteration, `koopman_modes.log` the modes with a line on what each
  eigenvalue means, `koopman_mode_anim_<k>.xyz` the latest structure moved along mode `k`,
  and `coordinates.csv` the coordinates and gradients of all iterations.

Every figure is written as a PNG and as an SVG; the PNG of mopt-rs has no lettering, so
the numbers of the axes and of the color bar are in the SVG (and in the `.csv`). A run of
fewer than three structures has no map. These options belong to an ordinary optimization;
`-vc`, `-mf bitss`, and plain variables do not take them.

In Python the parts are `mopt_rs._core.structure_distances`, `classical_mds`,
`path_variables`, `principal_components`, `koopman_observables`, and `dynamic_modes`. From
Rust: `analysis::path_map`, `analysis::koopman`.
