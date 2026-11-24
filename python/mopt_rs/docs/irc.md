# Intrinsic reaction coordinate (IRC)

The intrinsic reaction coordinate is the steepest-descent path in mass-weighted coordinates
`q = M^(1/2) x` from a transition state to the two minima it connects. mopt-rs follows it
with the integrators of MultiOptPy (LQA, HPC, DVV, Euler, RK4, and ModeKill for removing
imaginary modes) and with GS2, on any calculator and on objectives of plain variables.

Differences from MultiOptPy are listed in [migration_paths.md](migration_paths.md) ("IRC").

## Commands

```text
ircmain ts.xyz -os gfn0xtb                          # LQA, step 0.1, at most 300 steps
ircmain ts.xyz -os gfn0xtb -irc 0.1 300 gs2         # STEP_SIZE MAX_STEP METHOD, as optmain
ircmain ts.xyz -os gfn0xtb -irc_method hpc -irc_step 0.2 -fc 5
ircmain ts.xyz -os gfn0xtb -irc_opt initial=0.4 -path_analysis
ircmain ts.xyz -os gfn0xtb -irc_opt first_step=off energy_rise=off approach=off  # as MultiOptPy
ircmain point.txt -os mb2d -irc_step 0.05           # plain variables (unit masses)
optmain guess.xyz -os gfn0xtb -opt rsirfo_bofill -order 1 -fc 1 -irc 0.1 300 lqa
```

- `ircmain INPUT` starts from the given structure. `optmain ... -irc STEP_SIZE MAX_STEP
  METHOD` (MultiOptPy's entry) runs the same calculation from the last structure of the
  optimization, in the run directory of the optimization.
- The calculator options (`-os`, `-xtb`, `-func`, `-bs`, `-calc_opt`, ...), `-elec`,
  `-spin`, `-es`, the bias potentials (`-ma`, `-kp`, ...), `--config`, `--dump-config`, and
  `--restart` are those of `optmain`. Calculators come from the registry of the command, so
  Python engines (`ase:...`, PySCF, profiles) work with the pip commands.
- The Hessian of the starting structure is always computed (by the calculator, by
  differences of its gradients where it has none, or by O1NumHess with `-modelhess
  o1numhess`). `-fc N` asks for the exact Hessian at every N-th point of the path as well;
  without it the Hessian is updated along the path.
- Constraints (`-fix`, `-pc`, ...), `-mf`, and `-vc` are not available: the path is that of
  all atoms on the surface of one state in a fixed cell. `optmain -irc` after a constrained
  optimization follows the path without the constraints (as MultiOptPy) and says so.
- `ircmain` creates `<stem>_IRC_<time>/` next to the input. A `STOP` file in the run
  directory ends the run after the evaluation in progress (exit code 3);
  `ircmain --restart RUN_DIR` (`optmain --restart RUN_DIR`) continues it: the evaluations
  already made are read from `evaluations.journal`, and the files are written again from the
  start, so the result is the same as that of an uninterrupted run. Runs on objectives of
  plain variables are repeated from their start (they are cheap).

`ircmain` options (besides those of `optmain`):

| Option | Default | Meaning |
|---|---|---|
| `-irc_step` | 0.1 | step size in amu^(1/2) Bohr (the unit of the variables for plain variables) |
| `-irc_max` | 300 | `MAX_STEP` of each side |
| `-irc_method` | `lqa` | `lqa`, `hpc`, `dvv`, `euler`, `rk4`, `gs2`, `modekill[:MODES]` |
| `-irc STEP_SIZE MAX_STEP METHOD` | | the three together, as in `optmain` |
| `-irc_opt KEY=VALUE ...` | | further settings, see below (also for `optmain -irc`) |
| `-path_analysis` | off | analysis of the combined path as `freqmain -path` |

## What is done from the starting structure

The Hessian of the surface (with the bias potentials) is mass-weighted, translations and
rotations are projected out (for calculators whose energy does not depend on the position
and orientation), and the eigenvalues below `-1e-8` are counted:

- exactly one, and a gradient below the thresholds (largest component 0.0004, RMS 0.0001
  Hartree/Bohr; after `optmain` also the thresholds of the optimization that converged):
  **IRC**, two sides from `q_TS +- L v` along the eigenvector `v` of that mode, `L` = 0.1 of
  the step size (`initial=`). "Forward" is the side along which the largest component of `v`
  is positive.
- otherwise: **meta-IRC**, one path from `q - L g/|g|` down the gradient. The switch is
  automatic: a structure whose gradient is above the thresholds (a Franck-Condon point, a
  point on a seam, any structure) or which has no or several imaginary modes is followed
  down the gradient, and the log says `Execute meta-IRC`. On the surface of an excited
  state (`-es N`) it is the path of that state.
- `METHOD` = `modekill`: **ModeKill** at the structure itself.

### Constraints and combined surfaces

The constraints of `optmain` (`-fix`, `-pc`, `-gfix`, `-uc`) give the path of steepest
descent within them: in mass-weighted coordinates the gradient and the Hessians at every
point are replaced by their parts on the surface of the constraints (the gradient without
its part along the mass-weighted constraint gradients, the Hessian projected with the
curvature `sum_k lambda_k d2c_k` of the constraints and stiff across them, as in the
constrained optimizations), and every new point is brought back onto the constraints by
Newton steps in the mass-weighted metric (the shortest way in `q`; a light atom moves more
than a heavy one). The transition state is then that of the constrained surface, and the
ends are its minima. Constraints that fix the position or the orientation in space (fixed
atoms, Cartesian components, `rot`) end the projection of the rigid-body motion and the
recentring. `optmain -irc` keeps the constraints of the optimization before it, resolved
once at the input of the run: the path holds the targets, frozen positions, and directions
of the input, not the values at the optimized structure. Eigenvector constraints
(`-pc eigvec`) are resolved at the first Hessian of the optimization, which hands its
direction on, so that `optmain -irc` keeps them too; `ircmain -pc eigvec` has no Hessian of
its own to resolve them at the start and is refused there. `-gfix` (the distances between
the atoms of each group) is a constraint like `-pc bond` in `optmain` and `ircmain`.

The surface of the path is that of the last iteration of the optimization: a bias with a
ramp (`-ma 100,200`, `-kp 0.1,0.3`, `-ubias` with `it`) is held at the value it had where
the stationary point was found instead of starting over (`ircmain` holds it at its first
value). Terms whose value changes with the time or with what they have seen (`-ubias` with
`t`, the nanoreactor walls, metadynamics, the spacer and ellipsoid models with their inner
minimizations) are refused, with the name of the term: a path of steepest descent follows
one fixed surface. The rigid-body motion is projected out only where the calculator and every
bias potential are invariant under it (a bias at a point in space, `-vpp`, `-wwp`, `-ubias`
with `x()`, keeps it).

With constraints that fix the frame, the rigid motions that still keep the constraints
(the rotations about one fixed atom, the rotation about the line of two) are projected out
of the Hessians as the translations and rotations are otherwise: they leave the energy as
it is, and the noise of a Hessian from differences gave them eigenvalues of -4e-6 that
turned a transition state into a start with three imaginary modes.

`-mf NAME` follows a surface that combines two states (`seam`, `conical`, `avoiding`,
`avoiding_lower`, `legacy_avoiding`, the
functions of `optmain -mf`); the projections for crossings (`opt_mesx`, `opt_meci`) are
refused, since their gradients are not those of an energy and have no path of steepest
descent. The surface of one excited state is `-es N`.

### The atoms and the cell together (`-vc`)

`ircmain ts.xyz -vc` (and `optmain -vc ... -irc`) follows the path of steepest descent of
the enthalpy `E + P V` (`-vc_pressure`, GPa) in the variables of the variable-cell
optimization: the quasi-Cartesian coordinates of the atoms and the scaled lattice vectors
(`-vc_weight`, the same variables as `optmain -vc`). A path needs masses for all of them:
the atoms have theirs, and a lattice variable of the vector `k` has the diagonal of the
kinetic metric of a change of that vector with the atoms following through their
fractional coordinates, `(|a_k0| / gamma)^2 sum_i m_i (s_ik - s_k)^2` about the centre of
mass (at least that of the mean atom spread over the cell); the couplings of the atoms and
the lattice in that metric are left out (`opt::cell::VariableCell::masses`). Hessians are
those of `optmain -vc` (analytic where the calculator and the bias potentials give their
Hessians and their second derivatives along the strain, from differences of the gradient
in the variables otherwise), with the translations and rotations of the whole structure
projected out. The files of the
variables are those of an IRC of plain variables (`irc_<side>_points.csv`,
`<stem>_irc_endpoint_<n>.txt`); the structures with their cells are written beside them
(`irc_<side>_structures_cell.xyz`, and `<stem>_irc_endpoint_<n>.xyz`, the end points of
the run). `-vc_afir` (forces that push the cell) is refused, and so are `-mf` and the
constraints with `-vc`. `tests/irc_variable_cell.rs`: a vacancy of copper (EAM) hops to
the next site, from the saddle point of the atoms and the cell.

The masses are the standard atomic weights, the same as in the vibrational analysis
(`optmain -freq`, `freqmain`), so that the path and the analysis of it use the same
coordinates. `-irc_opt masses=isotope` takes the masses of the most abundant isotopes, as
MultiOptPy's IRC does. Plain variables have unit masses.

## Methods

| `METHOD` | What a step is | Evaluations per step | Counts `MAX_STEP` as |
|---|---|---|---|
| `lqa` | the steepest-descent path of the quadratic model at the point, followed for the step size (Page and McIver 1988, 1990) | 1 gradient | `MAX_STEP - 1` points |
| `hpc` | an LQA step predicts the next point; the step is taken again on the distance-weighted interpolant of the surface between the two points (Hratchian and Schlegel 2004, 2005) | 1 gradient | point 0 and `MAX_STEP - 1` points |
| `gs2` | the energy minimum on the hypersphere of radius `s/2` around the pivot `q - (s/2) g/|g|` (Gonzalez and Schlegel 1989, 1990) | a few gradients (1.6 to 3.3 on average in the measurements below) | `MAX_STEP` points |
| `rk4` | fourth-order Runge-Kutta for `dq/ds = -g/|g|` | 4 gradients | `MAX_STEP - 1` points |
| `euler` | the step size straight down the gradient | 1 gradient | `MAX_STEP` points |
| `dvv` | damped velocity Verlet with an adapted time step (Hratchian and Schlegel 2002); `STEP_SIZE` is not used | 1 gradient | `MAX_STEP - 1` points |
| `modekill` | the step size along the eigenvectors of the imaginary modes to remove | 1 gradient per cycle | `MAX_STEP` cycles |

The counting of `MAX_STEP` is MultiOptPy's (its loops are `range(1, max_step)` and, for
Euler, `range(1, max_step + 1)`).

Options of a method follow after colons: `dvv:v0=0.04:dt0=0.5:tol=0.003` (magnitude of the
velocity in amu^(1/2) Bohr/fs, first time step in fs, error tolerance in Bohr),
`modekill:1,2` (the modes with these numbers, counted from 0 in the order of the eigenvalues
of the mass-weighted Hessian; without numbers every imaginary mode below `nu_thresh` except
the lowest), `modekill:nu_thresh=-5` (cm^-1; by default the wavenumber of the one criterion of
an imaginary mode, about -0.51 cm^-1, which the start of an IRC counts by; MultiOptPy's -5
left modes between the two that the start then counted), `modekill:update=off` (the exact Hessian at
every cycle), `modekill:hess=off` (no Hessian of the last structure).

A side ends

- when the gradient is below the thresholds after more than 10 points (`Convergence
  reached. (IRC)`); LQA, Euler, and RK4 go on from there to the minimum first (the final
  approach, below),
- when the energy change reversed its sign 5 times in a row (MultiOptPy's oscillation rule;
  not while steps that raise the energy are taken again, which leaves nothing for it to
  catch),
- with `-irc_opt energy_rise=stop`: at the first point with a higher energy than the one
  before it,
- when a step raised the energy however short it was taken (ten halvings),
- when `MAX_STEP` is used up.

LQA and HPC shorten their steps by themselves near a minimum and arrive. GS2 halves its step
where the energy rises (down to 1/64 of the step size) and arrives or ends there. **DVV does
not arrive by itself** (it keeps its speed and circles the minimum, as in MultiOptPy), and
neither does RK4 with `approach=off` (its four gradients cancel at a minimum); both then run
until `MAX_STEP` unless `-irc_opt energy_rise=stop` is given.

### Settings of `-irc_opt`

| Key | Default | Meaning |
|---|---|---|
| `first_step` | `check` | test the first point of each side and correct it (below); `off`: MultiOptPy's first point as it is |
| `angle` | 10 | largest accepted angle (degrees) of the first-step test |
| `rho_min`, `rho_max` | 0.75, 1.25 | accepted range of the trust ratio of the first step |
| `initial` | 0.1 x step size | length `L` of the displacement from the transition state |
| `sides` | `both` | `forward` or `backward`: one side only |
| `masses` | `standard` | `isotope`: the most abundant isotopes (MultiOptPy) |
| `energy_rise` | `retry` | a step that raised the energy is taken again, shorter (LQA, Euler); `stop`: end a side where the energy rises; `off`: keep such points (MultiOptPy) |
| `overshoot` | `retry` | with `energy_rise=retry`: a step at whose end the gradient points back along it is taken again, shorter; `keep`: such points stay |
| `bend` | `off` | a share such as `0.25`: no step is longer than the one that loses this share of the energy it gains by cutting the bend of the valley (for `-fc 1`, see below) |
| `approach` | `on` | below `max_force` and `rms_force` a side goes on to the minimum (LQA, Euler, RK4); `off`: it ends there (MultiOptPy) |
| `approach_max_force`, `approach_rms_force` | 0.000015, 0.00001 | thresholds at which the final approach ends (those of `optmain -tcc`) |
| `approach_steps` | 100 | steps of the final approach at most |
| `max_force`, `rms_force` | 0.0004, 0.0001 | thresholds of the arrival (MultiOptPy's), and of the test whether the start is a stationary point |
| `min_points` | 10 | the arrival is tested after this many points |
| `oscillations` | 5 | reversals of the energy change in a row that end a side |
| `sphere_iterations` | 30 | evaluations at most in a minimization on a hypersphere |
| `sphere_gradient` | 1e-4 | RMS gradient within the sphere at which it ends |
| `sphere_step` | 1e-3 | ... or a next step below this share of the radius |

## The first step and its correction

Integrators without a constrained optimization leave the transition state by the straight
displacement `q_TS +- L v`. That point is on the path as far as the quadratic model of the
transition state holds over `L`. Two numbers are computed at the
trial point (which is the first point of the path anyway, so the test costs nothing):

- the **angle** between the downhill direction `-g` there and the displacement (zero on the
  path of the quadratic model, where `g = lambda L v`);
- the **trust ratio** `rho = dE_actual / dE_predicted` with
  `dE_predicted = g_TS . dq + lambda L^2 / 2` (`lambda`: the eigenvalue of the transition
  vector; the first term is zero at a converged transition state).

When the angle is above `angle` or `rho` outside `[rho_min, rho_max]`, the first point is
replaced by the energy minimum on the hypersphere `|q - q_TS| = L` **centred at the
transition state**, found from the trial point. It is the same minimizer as in GS2, whose
spheres are centred at pivot points half a step down the gradient from the last path point;
the difference is the centre only. The minimization is local (Newton steps within the
tangent space of the sphere with a trust length, Hessian updated by Bofill's formula), so it
ends in the minimum of the side it starts on: a sphere around a transition state has a
minimum on either side.

The numbers, whether the correction ran, its evaluations, how far the point moved, and the
energy it gained are written to the log and to `irc_first_step.csv`.

What the correction can and cannot do: the minimum on a sphere around the transition state
is on the path to second order in `L`, not exactly (this is why GS2 uses pivot points). The
measurements below show that it brings a point that left the floor of the valley back to
within a few percent of `L` of the path, and that it is not needed for the default `L`.

### Measurements behind the limits

Müller-Brown surface (two variables, energies scaled by 1e-3; `cargo test -p mopt-rs --test
irc_muller_brown -- --nocapture`). "Off the path" is the distance of the first
point from a reference path (RK4 in steps of 0.002).

| saddle | side | `L` | angle (deg) | rho | off the path | corrected | evaluations | off the path after |
|---|---|---|---|---|---|---|---|---|
| 1 | forward | 0.02 | 3.02 | 0.962 | 3.8e-4 | no | 0 | |
| 1 | forward | 0.05 | 8.31 | 0.898 | 2.4e-3 | no | 0 | |
| 1 | forward | 0.1 | 19.92 | 0.780 | 9.8e-3 | yes | 3 | 4.4e-3 |
| 1 | backward | 0.1 | 11.77 | 1.116 | 8.8e-3 | yes | 3 | 6.4e-3 |
| 1 | forward | 0.2 | 55.14 | 0.534 | 3.9e-2 | yes | 4 | 1.1e-2 |
| 1 | backward | 0.2 | 23.92 | 1.059 | 3.4e-2 | yes | 4 | 2.8e-2 |
| 1 | forward | 0.3 | 87.53 | 0.315 | 8.4e-2 | yes | 4 | 1.3e-2 |
| 1 | backward | 0.3 | 51.25 | 0.821 | 7.7e-2 | yes | 4 | 6.3e-2 |
| 2 | forward | 0.05 | 3.79 | 0.882 | 1.0e-3 | no | 0 | |
| 2 | forward | 0.1 | 13.95 | 0.726 | 5.8e-3 | yes | 3 | 4.0e-3 |
| 2 | backward | 0.1 | 1.02 | 1.087 | 1.9e-4 | no | 0 | |
| 2 | forward | 0.2 | 94.15 | 0.356 | 3.9e-2 | yes | 4 | 1.0e-2 |
| 2 | backward | 0.2 | 15.56 | 0.955 | 9.0e-3 | yes | 3 | 1.3e-2 |
| 2 | forward | 0.3 | 134.78 | -0.001 | 1.0e-1 | yes | 3 | 1.3e-3 |
| 2 | backward | 0.3 | 77.59 | 0.647 | 4.4e-2 | yes | 4 | 2.3e-2 |
| 2 | forward | 0.4 | 142.63 | -0.267 | 1.8e-1 | yes | 5 | 9.2e-2 |
| 2 | backward | 0.4 | 129.05 | 0.261 | 1.1e-1 | yes | 4 | 1.1e-2 |

HCN <-> HNC transition state, GFN0-xTB (`MOPT_FULL_MEASUREMENTS=1 cargo test -p mopt-rs
--test irc_hcn first_step_numbers -- --nocapture`; `L` in amu^(1/2) Bohr):

| `L` | side | angle (deg) | rho | corrected | evaluations | moved | energy gained (Hartree) |
|---|---|---|---|---|---|---|---|
| 0.01 | forward / backward | 0.14 / 0.12 | 0.995 / 1.005 | no | | | |
| 0.05 | forward / backward | 0.83 / 0.44 | 0.972 / 1.026 | no | | | |
| 0.1 | forward / backward | 2.11 / 0.68 | 0.943 / 1.050 | no | | | |
| 0.2 | forward / backward | 6.23 / 2.31 | 0.882 / 1.086 | no | | | |
| 0.3 | forward | 12.68 | 0.822 | yes | 2 | 0.014 | 2.3e-5 |
| 0.3 | backward | 6.19 | 1.103 | no | | | |
| 0.4 | forward | 21.72 | 0.762 | yes | 3 | 0.032 | 1.0e-4 |
| 0.4 | backward | 11.99 | 1.100 | yes | 3 | 0.019 | 5.4e-5 |
| 0.6 | forward | 45.71 | 0.646 | yes | 3 | 0.084 | 7.4e-4 |
| 0.6 | backward | 26.45 | 1.045 | yes | 4 | 0.054 | 4.8e-4 |
| 0.8 | forward | 67.90 | 0.535 | yes | 4 | 0.138 | 2.6e-3 |
| 0.8 | backward | 39.46 | 0.957 | yes | 4 | 0.100 | 1.7e-3 |

What the limits are taken from:

- Up to about 8 degrees and `rho` between 0.88 and 1.09 the straight point is off the path
  by at most 5% of `L` (Müller-Brown: 2.4e-3 at `L` = 0.05). From about 12 degrees on it is
  off by 4 to 45% of `L`, and the correction brings it closer in 17 of the 18 corrected
  Müller-Brown cases (lengths from 0.005 to 0.4 were measured, the table shows a selection;
  by factors of 1.2 to 80; in one, saddle 2 backward at `L` = 0.2 with 15.6 degrees, it is
  1.4 times farther, 1.3e-2 instead of 9.0e-3). The limit of 10 degrees lies between the
  two groups.
- `rho` alone is the weaker test: it stays within 0.75 to 1.25 for several points that are
  far off (saddle 1 backward, `L` = 0.2: 24 degrees, `rho` = 1.06). It catches the cases in
  which the energy does not fall as the model says although the gradient still points
  along the displacement. The range 0.75 to 1.25 is the one of trust-region methods for a
  model that is trusted; in the tables it is left together with or after the angle limit.
- With the default `L` (0.1 of the step size, 0.01 for the default step) both numbers are
  far inside the limits in all cases measured: the test is for large first displacements.
- The correction takes 2 to 5 evaluations with the default settings of the minimizer.

### Settings of the hypersphere minimizer

GS2 on the Müller-Brown surface with a step of 0.1 (both saddles; largest distance of a
path point from the reference path, evaluations of the whole calculation):

| `sphere_gradient` | saddle 1 | saddle 2 |
|---|---|---|
| 1e-3 | 1.80e-3, 102 | 2.19e-3, 75 |
| 1e-4 (default) | 1.94e-3, 98 | 1.69e-3, 83 |
| 1e-5 | 1.94e-3, 113 | 1.69e-3, 89 |

Below 1e-4 the path does not change (to three digits) and the cost rises by 7 to 15%; at
1e-3 the path of one saddle is off by 30% more. `sphere_step` between 1e-2 and 1e-4 changes
neither the path nor the cost by more than 8% (saddle 1: 106, 98, 98 evaluations); the
gradient test ends the minimizations before it does. A minimization takes 3.3 evaluations
on average on this surface (140 for the 43 points at a step of 0.05) and 1.6 for HCN <-> HNC
(232 for a path of 14.2 at a step of 0.1), far from the limit of 30.

## Steps that fail, and where a side ends

Along a path of steepest descent the energy falls from point to point, and the path ends in a
minimum. MultiOptPy's integrators keep a point whose energy is above that of the point before
it, and end a side at the first point whose gradient is below 0.0004 (largest component) and
0.0001 (RMS), which lies before the minimum. The defaults of mopt-rs for LQA (and Euler; RK4
has the final approach only) differ in four ways; `-irc_opt first_step=off energy_rise=off
approach=off` gives MultiOptPy's behaviour.

1. **A step that raised the energy is taken again** (`energy_rise=retry`). The point is
   dropped, and the step is taken from the last point of the path at half its length (with
   `-fc N`, first at the same length with the exact Hessian of that point). Up to ten
   halvings; then the side ends where it is. After every point that is kept the length
   doubles again, up to the step size.
2. **A step that went past the floor of the valley is taken again** (`overshoot=retry`). When
   the gradient at the end of a step points back along the step (`g . dq > 0`), the step
   crossed the floor: the energy may still be lower, but the next step would turn back, and
   the path would have a kink. Up to three halvings; then the point is kept.
3. **Zigzag**: a point at which the path turns back by more than a right angle halves the
   length of the following steps (down to 1/32 of the step size).
4. **The final approach** (`approach=on`). Below `max_force` and `rms_force` the side does
   not end but goes on until the gradient is below `approach_max_force` and
   `approach_rms_force` (the thresholds of `optmain -tcc`). The steps are those of LQA on the
   model with the magnitudes of its curvatures (so that they go down whatever sign the
   updated Hessian has along a mode), at most a step of the path long; near the minimum such
   a step ends in the minimum of the model. It begins only where the path lies in the bowl
   of the minimum: the model has no direction of negative curvature, and down to its minimum
   it has less than a tenth of the energy left to lose that the side has lost since the
   transition state. (A small gradient alone does not say that: on a flat surface it is
   below the thresholds next to the transition state, where the two energies are equal.)
   A side that reaches a stationary point with a direction of negative curvature (it ran
   down a symmetric ridge into a transition state, past a valley-ridge inflection) has
   arrived there as soon as its gradient is below `approach_max_force` and
   `approach_rms_force`, after `min_points` points and once it has lost energy. Before, such
   a side took steps of length zero there until its points were used up: the model, with an
   exact Hessian (`-fc N`), knows the negative curvature, so the side never was in a bowl.
   `run_mapper` tells such an end by its Hessian (a TS–TS link).
5. **Rounding across a ridge.** A part of the gradient in a mode of the Hessian below 1e-12 of
   its norm is left out of a step of LQA. Along a mode of negative curvature it would grow by
   `exp(|lambda| t)`, and near a stationary point, where the parameter `t` of the step is
   large, it carried a path that runs down a symmetric ridge off the ridge; steepest descent
   stays on it. MultiOptPy keeps it.

What the step control did is written to the log at the end of each side (points dropped for a
higher energy and for an overshoot, exact Hessians asked for, halvings, steps of the final
approach), and is part of `BranchSummary` (`counts`) and of the sides of
`mopt_rs.irc_path(...)` (`dropped_points`, `overshot_points`, `shortened_steps`,
`approach_steps`).

### Measurements

Transition states from chain relaxations, optimized with `optmain -opt rsirfo_bofill -order 1`,
built-in GFN0-xTB, LQA, step 0.1, Hessian updated along the path. "Rises": points with a
higher energy than the one before. "End": each end point of the IRC was optimized with
`optmain -tcc`; the columns are how far the energy of the IRC end was above that of the
optimized structure, the largest change of an interatomic distance, and the iterations of
the optimization (1: it had converged where it started). Sharp turns: angles below 150
degrees between two steps that are both longer than 0.02 amu^(1/2) Bohr.

| Transition state | Way | Points kept, both sides | Dropped (higher energy + overshoot) | Rises | Sharp turns (below 150 / below 90 degrees) | End above the optimized structure (Hartree) | Largest change of a distance (Angstrom) | Iterations of the optimization |
|---|---|---|---|---|---|---|---|---|
| HCN to HNC (1286i cm^-1) | defaults | 91 | 1 + 3 | 0 | 1 / 0 | < 1e-12 | < 1e-12 | 1, 1 |
| | MultiOptPy | 88 | | 1 | 2 / 0 | 2.0e-8, 5.7e-9 | 1.6e-4, 9.4e-5 | 33, 21 |
| CH3NC to CH3CN (329i) | defaults | 208 | 12 + 5 | 0 | 5 / 0 | < 1e-12 | < 1e-12 | 1, 1 |
| | MultiOptPy | 200 | | 2 | 8 / 1 | 9.2e-8, 9.4e-8 | 5.5e-4, 5.3e-4 | 38, 39 |
| CH3NC to CH3CN, another saddle (701i) | defaults | 196 | 17 + 3 | 0 | 0 / 0 | < 1e-12 | < 1e-12 | 1, 1 |
| | MultiOptPy | 187 | | 1 | 1 / 0 | 1.9e-7, 4.3e-9 | 4.4e-4, 6.0e-5 | 35, 19 |
| Allyl bromide, a saddle 45.8 kcal/mol above it (282i) | defaults | 327 | 16 + 6 | 0 | 7 / 0 | < 1e-12 | < 1e-12 | 1, 1 |
| | MultiOptPy | 304 | | 5 | 18 / 5 | 3.1e-5, 3.5e-5 | 7.2e-2, 7.6e-2 | 82, 86 |
| Water dimer, donor-acceptor interchange (21.8i) | defaults | 24 | 23 + 1 | 0 | 0 / 0 | < 1e-12 | < 1e-12 | 1, 1 |
| | MultiOptPy | 22 | | 4 | 0 / 0 | 2.0e-9, 1.3e-9 | 3.0e-5, 1.9e-5 | 16, 15 |

The defaults cost 8 to 16 % more evaluations than MultiOptPy's way on the first four (95
against 88, 225 against 200, 216 against 187, 349 against 304: the points that are dropped
and the final approach), and the optimization of the end points, 20 to 90 iterations each,
is no longer needed. Repeated runs differ by a point or two (the sums of the engine are
taken in parallel, and a difference in the last digit decides whether a step is kept in a
flat region).

Allyl bromide is the hard case among them. Its saddle here (bromine bridging two carbon
atoms while the CH2 group at the other end turns; both sides end in allyl bromide) was
optimized from an image of a chain that was not relaxed yet; it is not on the path of that
chain. The valleys down from it are long and flat, and MultiOptPy's way ends 0.07 Angstrom
and 3e-5 Hartree before the minima, with two and three points of higher energy on the way.
The relaxed chain itself (bromine from one end of the allyl group to the other) runs over
a plateau 20.0 kcal/mol above allyl bromide, where bromine is 3.8 to 4.5 Angstrom from the
carbon atoms; the highest image is on a saddle (an optimization from it changes the energy
by less than 1e-7 Hartree and the distances by less than 0.03 Angstrom; one imaginary
mode, 18i cm^-1), which lies 0.006 kcal/mol above the plateau, and the IRC from it ends
on the plateau within the thresholds, not in allyl bromide: with GFN0-xTB this path is a
dissociation and recombination, not one concerted step.

The water dimer shows the limit of every such rule: with GFN0-xTB the saddle of the
donor-acceptor interchange found here (21.8i cm^-1) lies 2e-7 Hartree above its
surroundings, the gradient is below the tight thresholds from the transition state on, and
both ways end within 0.2 amu^(1/2) Bohr of it; the end points are stationary within the
thresholds, not the minima of the dimer. The formic acid dimer has no barrier for the double
proton transfer with GFN0-xTB (the symmetric structure is a minimum), so it gave no test.

A model of the hard case is in `tests/irc_flat_valley.rs`: `E = a (x^2 - 1)^2 + K (y - c
x^2)^2 / 2` with `a = 1e-4`, `K = 1`, `c = 0.5`, a valley that is flat along the path (a
barrier of 1e-4 Hartree), a thousand times stiffer across it, and bent with a curvature of
1 per unit length. There a step of the full length leaves the floor by `s^2 / 2 = 0.005`,
which costs `K (s^2 / 2)^2 / 2 = 1.25e-5`, more than the 1e-5 the step gains: the energy
rises although the Hessian is exact. With the defaults no point rises, both sides end with
gradients below 1.5e-5 within 0.02 of the minima, and the points lie within 0.004 of the
steepest-descent path (integrated as a gradient flow for reference); MultiOptPy's way ends
next to the saddle, where the gradient is below its thresholds. Euler's straight steps are
stable in this valley only below a length of `2 |g| / K`, about 1e-4: at the step size every
step is halved several times, and the side creeps (it does not end falsely, and no point
rises).

### What was tried and is not used

Each of these was implemented and measured on the model and on the molecules above
(evaluations = points kept + points dropped, both sides):

| Idea | Result |
|---|---|
| Step length from the trust ratio `rho = dE_actual / dE_predicted` of the last step, continuously (`s <- s (eta / abs(1 - rho))^(1/3)`), instead of halving and doubling | evaluations 335 -> 340 (allyl bromide), 233 -> 231 (CH3NC), 92 -> 93 (HCN); on the model 123 -> 189. Not used. |
| Carrying the part of the gradient the quadratic model did not foresee over to the next step along the stiff modes (a third-order correction) | helps with the exact Hessian at every point (model: 81 -> 71 evaluations, points 2.5 times closer to the path); with updated Hessians it does harm (allyl bromide 335 -> 458), because the secant update has absorbed the same residual into the Hessian already. Not used. |
| Negative curvature of the model across the path taken as zero | next to the transition state the mode of the path itself holds less of the gradient than the stiff modes and is damped as well; the side does not leave. Not used. |
| The final approach as steps to the minimum of the model in one go (Newton) | the same end points; sharper turns in the path (allyl bromide: 29 angles below 150 degrees against 20, 4 against 1 below 90). Replaced by the LQA form. |
| `bend=SHARE`: a limit of the step length from the bend of the valley, `s^3 <= 8 SHARE abs(g) / (kappa . H . kappa)` with the curvature vector `kappa` of the path from the last three points (the energy a straight step loses by cutting the bend is `s^4 kappa.H.kappa / 8`: the curvature couples to the modes across the path, each with its force constant, the quantities of the reaction path Hamiltonian and of URVA) | with the exact Hessian at every point the points lie six times closer to the path on the model (5.6e-4 against 3.5e-3) at 20 % more evaluations, and fewer steps fail; with updated Hessians the estimate of `kappa . H . kappa` is too rough (model: 129 -> 183 evaluations), on the molecules it changes little (dropped points 14 -> 5 for allyl bromide, 17 -> 10 for CH3NC, evaluations within 3 %). Kept as an option, off by default; meant for `-fc 1`. |

A step control cannot make the energy fall with certainty from an updated Hessian; that is
why the points are tested and dropped, which costs 4 to 9 % of the evaluations on the first
four molecules.

## Accuracy and cost of the methods

The tables of this section were measured with `-irc_opt first_step=off energy_rise=off
approach=off` (every step of the step size, a side over at the first point below the
thresholds), so that they compare the integrators themselves; the defaults add the points
that are dropped and the final approach (the section above).

Müller-Brown surface, both saddles: largest distance of a path point from the reference
path (points closer than 0.15 to a minimum left out, where the reference itself ends short),
and the evaluations of both sides.

| method | step | saddle 1: distance, evaluations | saddle 2: distance, evaluations |
|---|---|---|---|
| `rk4` | 0.03 | 9.9e-6, 4787 (`MAX_STEP`) | 3.6e-5, 4787 (`MAX_STEP`) |
| `gs2` | 0.05 | 5.4e-4, 140 | 4.7e-4, 87 |
| `gs2` | 0.1 | 1.9e-3, 98 | 1.7e-3, 83 |
| `gs2` | 0.2 | 5.0e-3, 91 | 6.4e-3, 89 |
| `euler` | 0.01 | 2.8e-3, 206 | 1.4e-3, 107 |
| `lqa`, exact Hessians (`-fc 1`) | 0.02 | 6.0e-4, 103 | 3.7e-4, 53 |
| `lqa`, exact Hessians (`-fc 1`) | 0.05 | 3.5e-3, 46 | 2.6e-3, 27 |
| `lqa`, exact Hessians (`-fc 1`) | 0.1 | 1.3e-2, 27 | 7.9e-3, 23 |
| `lqa` | 0.02 | 6.0e-3, 103 | 1.3e-3, 56 |
| `lqa` | 0.05 | 7.9e-3, 46 | 5.6e-3, 28 |
| `lqa` | 0.1 | 3.0e-2, 30 | 1.9e-2, 23 |
| `lqa` | 0.2 | 9.3e-2, 23 | 7.6e-2, 23 |
| `lqa` | 0.5 | 3.5e-1, 23 | 3.0e-1, 23 |
| `hpc` | 0.05 | 1.1e-2, 40 | 2.3e-3, 25 |
| `hpc` | 0.1 | 2.2e-2, 25 | 7.2e-3, 25 |
| `hpc` | 0.2 | 5.4e-2, 25 | 1.8e-2, 25 |
| `dvv`, `energy_rise=stop` | | 1.3e-2, 177 | 1.3e-2, 65 |

LQA with exact Hessians is of second order in the step (1.3e-2, 3.5e-3, and 6.0e-4 at steps
of 0.1, 0.05, and 0.02: a factor of 3.7 for half the step and of 5.8 for a step 2.5 times
smaller); with updated Hessians it is limited by the update on this strongly anharmonic
surface (6e-3 at a step of 0.02). Among the methods that need gradients only, GS2 is the
most accurate per evaluation here; LQA and HPC are the cheapest.

HCN <-> HNC, GFN0-xTB: the largest difference of the energy from that of a reference path
(LQA, step 0.02, exact Hessians) at the same length of the path, and the evaluations of
both sides (path length 14.1 amu^(1/2) Bohr):

| method | step | evaluations | largest energy difference (Hartree) | ends |
|---|---|---|---|---|
| `lqa` | 0.05 | 164 | 1.5e-4 | converged, converged |
| `lqa` | 0.1 | 89 | 3.9e-4 | converged, converged |
| `lqa` | 0.2 | 55 | 2.5e-3 | converged, converged |
| `lqa` | 0.5 | 32 | 4.1e-3 | converged, converged |
| `hpc` | 0.1 | 80 | 2.2e-3 | converged, converged |
| `hpc` | 0.2 | 39 | 6.5e-3 | converged, converged |
| `gs2` | 0.1 | 232 | 4.3e-4 | converged, converged |
| `gs2` | 0.2 | 250 | 9.5e-4 | converged, energy rise |
| `rk4`, `energy_rise=stop` | 0.1 | 517 | 4.2e-4 | energy rise, energy rise |
| `euler` | 0.02 | 382 | 1.0e-5 | converged, converged |
| `dvv`, `energy_rise=stop` | | 412 | 3.4e-4 | energy rise, energy rise |

**The default step of `ircmain`, 0.1 amu^(1/2) Bohr with LQA**, is taken from this table:
the energy profile is within 4e-4 Hartree (0.25 kcal/mol) of the reference at 89
evaluations; half the step costs twice as much for 1.5e-4, and MultiOptPy's recommended
`0.5 300 lqa` is off by 4e-3 Hartree (2.6 kcal/mol) at 32 evaluations. `optmain -irc` has no
default (the three values are given, as in MultiOptPy). The default method is LQA, as
MultiOptPy recommends; 300 steps of 0.1 cover a side of 30 amu^(1/2) Bohr.

## Files

In the run directory (`irc_forward/` and `irc_backward/` for the two sides of a transition
state; a meta-IRC and ModeKill write into the run directory itself):

| File | Content |
|---|---|
| `irc_energies_gradients.csv` | `Step, Energy (Hartree), Bias Energy (Hartree), RMS Gradient, RMS Bias Gradient`, and `Time Step (fs), Damping Factor` (DVV) or `Path Length` (Euler); "Bias Energy" is the energy with the bias potentials |
| `irc_structures.xyz` | the points of the path (Angstrom), comment `IRC Step N` |
| `irc_curvature_properties.csv` | Euler and RK4: curvature of the path and its components along the modes of positive curvature |
| `plot_IRC_bending_0.png` | the bending angle of the path at every point |
| `plot_DVV_timesteps_0.png` | DVV: the time steps |
| `modekill_energies_gradients.csv`, `modekill_structures.xyz` | ModeKill, with the column `Imaginary Frequencies` |
| `irc_combined_path.xyz` | forward side from its end, the starting structure, backward side to its end |
| `irc_combined_data.csv` | the tables of both sides: backward from its end with negative steps, step 0 (the start), forward |
| `<stem>_irc_endpoint_1.xyz`, `_2.xyz` | the last structures of the forward and the backward side (a meta-IRC: `_1`) |
| `irc_profile.csv` (new) | one row per frame of `irc_combined_path.xyz`: signed mass-weighted length of the path from the start (positive on the forward side), energy, energy with bias potentials |
| `irc_first_step.csv` (new) | `first_step=check`: the test of the first point of each side |
| `path_analysis/` (new) | `-path_analysis`: the files of `freqmain -path` |

For plain variables the points are in `irc_points.csv`, `irc_combined_path.csv`, and
`<stem>_irc_endpoint_N.txt` (a variables file).

`irc_combined_path.xyz` is a path as `freqmain PATH.xyz -path` reads it: every frame,
the starting structure included, has its centre of mass at the origin and no rotation
between frames. `ircmain ... -path_analysis` runs that analysis with the same calculator
right after the IRC; `freqmain irc_combined_path.xyz -os ... -path` does the same later and
takes the options of the analysis (`-temp`, `-path_tangent`, `-phase_prominence`, ...).

## From Rust and Python

Rust: `mopt_rs::irc::Irc` is an ask/tell state machine (`ask()` gives the next point,
`tell()` takes the surface there, `drain_events()` what happened);
`mopt_rs::workflow::irc::run_irc` drives it on a surface with observers
(`IrcObserver`: the files, the `STOP` file, and any further processing after a step; an
observer can end the run), and `mopt_rs::cli::ircmain::run_with_observers` is the command
with further observers. Parts that work alone: `irc::lqa_step`, `irc::DwiSurface`,
`irc::SphereMinimizer`, `irc::test_first_step`, `irc::convergence_check`,
`irc::IntegratorRegistry` (integrators by name; further ones can be registered).

Python:

```python
import mopt_rs

def after_step(report):          # called after every point of the path
    print(report["branch"], report["step"], report["s"], report["energy"])
    return report["energy"] > -0.2      # False ends the run cleanly

result = mopt_rs.irc_path("mb2d", [-0.822, 0.624], step_size=0.05, callback=after_step)
result.kind                      # "transition_state"
result.side("forward").points    # the points of a side
result.path, result.path_energies, result.path_arc_lengths
```

`irc_path(objective, x, ...)` takes a function `x -> (value, gradient[, hessian])` or the
name of a built-in objective; `symbols=[...]` makes `x` the Cartesian coordinates (Bohr) of
a structure (atomic weights, or `masses=`), otherwise the variables have the `masses` given
(one each by default). `mopt_rs.Irc` is the state machine for any other way of evaluating
the points; `mopt_rs._core.run_cli("ircmain", [...])` and `mopt_rs.run_optmain` run the
commands with the Python calculators.

## Runs on molecules

Beyond the tests, `ircmain` was run on transition states that came out of a chain
relaxation (`examples/chain_molecule.rs`; the highest image, optimized with
`optmain -opt rsirfo_bofill -order 1 -fc 1`), with the built-in GFN0-xTB, LQA, step 0.1:

| Transition state | imaginary wave number | points of the path | ends |
|---|---|---|---|
| HCN to HNC | 1286i cm^-1 | 89 | HCN and HNC; barrier 68.285 kcal/mol from HCN, HNC 15.722 above HCN (the numbers of the chain) |
| CH3NC to CH3CN (from a chain of 6 images) | 329i | 201 | CH3NC and CH3CN; barrier 68.954 from CH3NC |
| CH3NC to CH3CN (from a chain of 10 images, another saddle) | 701i | | CH3NC and CH3CN |

`-path_analysis` was run for the first one (12 reaction phases reported).

In the second run, made while points of higher energy were still kept, the energy rose by
1.2 kcal/mol at point 27 of one side and fell back at the next point; the path bent by 95
and then 19 degrees there. The region is flat (the methyl group turns, a mode of 101 cm^-1
at the transition state), and LQA took a step that the surface did not follow. This is what
the step control above was written for; with it no point of these paths rises.

## Tests

- `tests/irc_flat_valley.rs`: the flat, stiff, bent valley (no point of higher energy, the
  ends in the minima, the distance from the steepest-descent path, the counts of the step
  control, the `bend` limit with exact Hessians, MultiOptPy's way for comparison, Euler).
- `crates/mopt-rs/src/irc/` (unit tests of every part), `tests/irc_muller_brown.rs` (every
  integrator from both saddles, distance from the reference path, the first step),
  `tests/irc_hcn.rs` (`optmain -irc` from a guess, every integrator, the first step, a bias
  potential, `-path_analysis`), `tests/cli_ircmain.rs` (files, `ircmain` and `optmain -irc`
  interrupted in the middle of the path and restarted with identical files, observers),
  `tests/test_irc.py`.
- `tests/golden_irc.rs` compares with MultiOptPy 1.21.2 where its result does not depend on
  an error that is corrected here (`tools/golden/irc_reference.py`): the LQA step (four
  Hessians and gradients, new structures within 1e-10 Bohr), the distance-weighted
  interpolant and its corrector (energies 1e-12, gradients 1e-11, end points 1e-10), and an
  RK4 run of 13 points on a Morse potential of three atoms (energies and RMS gradients
  within 1e-11 Hartree, structures within 5e-10 Bohr, the precision of MultiOptPy's file).
