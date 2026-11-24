# Bias potentials

Bias potentials are added to the surface being optimized (or to one state of a combined
surface). All built-in ones have exact gradients and Hessians (automatic differentiation, the
hand-written derivatives of AFIR, the Schur complement of the microiterated ones, and the
exact Hessian of the RMSD metadynamics of `confsearch` through Horn's quaternion matrix) and
work in isolated systems and in 1D, 2D, and 3D periodic cells: distances, angles, and fragment
centroids use minimum images along the periodic axes; absolute positions are taken from the
image nearest to the reference point. Each built-in potential is checked with
`pes::term_check` (finite differences, lattice translations of single atoms, no wrapping along
non-periodic axes) in skewed cells. Only user callbacks without a Hessian get one from
differences of their gradient (below).

Angles (`-ka` excepted, which follows MultiOptPy's series near 0 and 180 degrees, the angle
of `-ubias`, the angle variable of `-metad`, the cone of `-cp`, and angle constraints) are
computed as `atan2(|a x b|, a . b)`, which keeps the value and its derivatives accurate up to
the linear geometry; the arc cosine of the normalized dot product loses all digits there.

### Stress

In a periodic cell every bias built on `GeometricBias` (all options except `-ma`, `-smp`,
`-aerp`, `-aerpv2`) returns its stress analytically: under the homogeneous strain
`r -> (1 + e) r` every position and every minimum-image lattice shift scales, so the nine
strain components are differentiated by forward-mode automatic differentiation, symmetrized
and divided by the volume as in `pes::stress`. Fixed points of a bias (`-vpp`, `-fp`, `-vpwp`,
`-nrp`) stay where they are. AFIR has its own analytic stress and strain Hessian. The
microiterated biases have none; variable-cell runs difference their energy along the strain.
`tests/term_contracts.rs` compares the stress of every geometric option with central
differences of its energy in a triclinic cell and in a slab.

### What a term declares

Every in-process term (`pes::PesTerm`) says whether its value depends on where the structure
is in space (`frame`) and on more than the structure (`history`), and can be copied with its
own state (`fork`, `pes::term::fork_term`). `pes::term::frame_invariant` is the one place that
decides whether a surface may have its rigid-body motions removed; terms that do not say are
`Absolute` and `Stateful`.

| Options | Frame | History |
|---|---|---|
| `-ma`, `-rp`, `-rpv2`, `-rpg`, `-cp`, `-kp`, `-kpv2`, `-akp`, `-ka`, `-kav2`, `-up`, `-kda`, `-kdav2`, `-kdac`, `-kopa`, `-kopav2`, `-brp`, `-wp`, `-awp`, `-lmefp`, `-lmefpv2` | invariant | static; iteration ramp where a constant is given as `start,end` |
| `-vpp`, `-fp`, `-wwp`, `-vpwp` | absolute (a fixed point or a Cartesian axis) | static |
| `-nrp` | absolute (the origin, see below) | time-dependent |
| `-metad` | invariant | stateful (deposited Gaussians) |
| `-smp`, `-aerp`, `-aerpv2` | invariant | stateful (the inner minimization starts from the previous solution, which can decide the minimum where there are several) |
| `-ubias` | absolute with `x`, `y`, `z`, invariant otherwise | time-dependent with `t`, iteration ramp with `it`, static otherwise |

### A calculator with static biases

`pes::biased::BiasedCalculator` wraps a calculator and static terms into one `Calculator`
(energy, gradient, Hessian, and stress with the terms; `evaluate_parts` also reports the bare
energy, like `energy` and `biased_energy` of a job). It refuses any term whose history is not
static, naming it: a calculator is called without the iteration, the time, or the order of the
evaluations.

### Hessians of the terms on request

`PesExpr::evaluate` evaluates its terms with Hessians even for a gradient-only request, since
the optimizer adds the exact Hessian of its biases to its quasi-Newton Hessian at every
iteration. `PesExpr::evaluate_as_requested` evaluates them with Hessians only when the request
asks for one. Measured with `tests/term_contracts.rs`
(`bias_hessians_follow_the_request`, 18 argon atoms with seven bias terms, 20 repetitions of a
gradient-only evaluation of the surface from given calculator results, unoptimized test
build): 157 ms per evaluation with the Hessians of the terms, 11 ms as requested. The energies
and gradients agree to rounding (dual instead of hyper-dual numbers).

## Built-in options

The options, argument groups, and units are those of MultiOptPy (`optmain`, and later
`nebmain` and `ieipmain`). Several groups can follow one option; groups whose constant is zero
are skipped. Spring constants marked *ramp* accept `start,end`, changed linearly over 300
iterations as in MultiOptPy.

| Option | Arguments per group | Energy |
|---|---|---|
| `-ma` | gamma[,end] (kJ/mol), fragment 1, fragment 2 | AFIR |
| `-kp` | k (a.u., ramp), r0 (A, ramp), atom pair | `k/2 (r - r0)^2` |
| `-kpv2` | k, r0, fragment 1, fragment 2 | same, between centroids |
| `-akp` | De, k (a.u.), r0 (A), atom pair | Morse well |
| `-ka` / `-kav2` | k, theta0 (deg), 3 atoms / 3 fragments | `k/2 (theta - theta0)^2` |
| `-kda` / `-kdav2` | k, phi0 (deg), 4 atoms / 4 fragments | `k/2 (phi - phi0)^2` (wrapped) |
| `-kdac` | k, n, phi0 (deg), 4 fragments | `k/2 (1 - cos(n phi - phi0))` |
| `-kopa` / `-kopav2` | k, chi0 (deg), 4 atoms / fragments (center first) | out-of-plane angle |
| `-vpp` | k, r0 (A), point x,y,z (A), atoms, order p | `k/p (|r - P| - r0)^p` per atom |
| `-up` | c (kJ/mol), atoms | pulls atoms toward their centroid |
| `-fp` | k x,y,z, order x,y,z, point x,y,z (A), atoms | `sum k_c (x_c - x0_c)^p_c` |
| `-brp` | upper k, lower k (1/Bohr), upper, lower (A), fragments | soft distance range |
| `-lmefp` / `-lmefpv2` | F (pN), two pairs / one pair | constant pulling force |
| `-rp` | well scale, distance scale, fragment 1, fragment 2, `scale` or `value` | UFF LJ |
| `-rpv2` | well, distance, length (A), A, B, n, m, center pair, targets, `scale`/`value` | LJ around a point on a bond |
| `-rpg` | LJ well (kJ/mol), LJ distance (A), Gaussian well, center, range, fragments | LJ minus Gaussian |
| `-cp` | well (kJ/mol), distance (A), cone angle (deg), center, 3 atoms, targets | cone repulsion |
| `-wp` / `-wwp` / `-vpwp` / `-awp` | wall (kJ/mol), geometry, a,b,c,d (A), targets | flat-bottomed wells |
| `-smp` | well (kJ/mol), distance (A), cavity scale, particles, targets | spacer-model solvent (microiterations) |
| `-metad` | `bond`/`angle`/`dihedral`/`outofplain`, height (kJ/mol), width (A or deg), atoms | metadynamics |
| `-aerp` / `-aerpv2` | eps (kJ/mol), six semi-axes (A), distance (A), root,axis atoms, off-targets | asymmetric ellipsoids (microiterations) |
| `-nrp` | inner, outer wall (A), two times (ps), two force constants (kcal/mol/A^2) | nanoreactor (Wang et al. 2014) |
| `-ubias` | expression | user-defined, see below |

`-smp`, `-aerp`, and `-aerpv2` minimize auxiliary variables (particle positions, ellipsoid
rotations) at every evaluation with the microiteration machinery shared with ONIOM
(`pes::microiter`, with mode kill); the term is the relaxed energy, its gradient the envelope
gradient, and its Hessian the Schur complement over the auxiliary variables. Differences from
MultiOptPy are listed in [migration_surfaces.md](migration_surfaces.md).

Constants for which no energy is real are refused when the option is read, with the option
named: `-akp` needs a positive well depth and a force constant that is not negative
(`sqrt(k / 2 De)`), `-vpp` a nonzero integer order (`(|r - P| - r0)^p` is negative inside the
sphere), and `-fp` integer orders for the components it acts on (`x_c - x0_c` takes both
signs). MultiOptPy evaluates these as NaN.

The centre of the spheres of `-nrp` is the origin in every command: MultiOptPy's
`NanoReactorPotential.calc_energy` measures each atom by the norm of its position
(`torch.linalg.norm(geom_num_list, dim=1)`). The term is therefore fixed in space (absolute
frame), and optimizers neither recentre nor rotate a structure held by it. The nanoreactor
runs of `mapper` place the centre once at the centroid of their starting structure, also a
fixed point.

## User-defined bias potentials

### Expressions (`-ubias`, any interface)

```
optmain mol.xyz -ubias "0.15*(dist(1,2) - 1.5*ang)^2" "10*kjmol*exp(-dist([3-5],6)/ang)"
```

An expression gives the energy in Hartree from geometric variables in atomic units (Bohr,
radians), with exact derivatives and periodic cells like the built-in terms.

- Geometry (atoms `3` or groups `[1-3,5]`, entering by their centroids; 1-based):
  `dist(a,b)`, `angle(a,b,c)` (at `b`), `dihedral(a,b,c,d)` (IUPAC sign), `oop(c,a,b,d)`
  (angle of `a` above the plane of `b` and `d` around the center `c`), `x(a)`, `y(a)`, `z(a)`.
- Operators `+ - * /`, `^` or `**` (right-associative), parentheses.
- Functions `exp ln log sqrt sin cos tan tanh acos abs`, `atan2(y,x)`, `min(a,b)`, `max(a,b)`.
  `tanh` saturates at `+-1` for any argument. A power whose exponent is not a number is
  `exp(p ln b)` for a positive base, the real branch `(-1)^n |b|^p` where the exponent has an
  integer value `n`, and `0` for `0^p` with `p > 0`; anything else has no real value, and the
  evaluation stops with an error that names the operation and its arguments.
- Constants `pi`, unit factors `ang` (Bohr per Angstrom), `deg`, `kjmol`, `kcalmol`, `ev`.
- Variables `it` (optimizer iteration) and `t` (simulated time in a.u. in molecular dynamics).

### Python callbacks

```python
import numpy as np
import mopt_rs

def wall(positions, cell, iteration, need_hessian):
    x = np.asarray(positions).reshape(-1, 3)
    over = np.maximum(x[:, 2] - 10.0, 0.0)          # Bohr
    g = np.zeros_like(x)
    g[:, 2] = 0.1 * over
    return 0.05 * float(over @ over), g.ravel(), None   # no Hessian: finite differences

print(mopt_rs.check_term(wall, ["Ar"] * 2, [0, 0, 9.0, 0, 0, 11.0]))
# The wall stands at z = 10 Bohr: the structure must not be recentred.
opt = mopt_rs.Optimizer(["Ar"] * 2, [0, 0, 9.0, 0, 0, 11.0], methods=["rsirfo_bfgs"],
                        rigid_body_invariant=False)
opt.add_term("wall", wall)
```

A callable may say what its value depends on besides the positions with two attributes:
`frame` (`"invariant"` for a function of distances and angles, `"translation_only"`, or
`"absolute"`, the default) and `history` (`"static"`, `"iteration_ramp"`, `"time_dependent"`,
or `"stateful"`, the default). An optimizer that removes rigid-body motions (the default
`rigid_body_invariant=True`) takes only terms with `frame = "invariant"` and refuses the
others with their names, because recentring and superposing the structures at every step
would move them in a fixed frame; set `f.frame = "invariant"` on a function of the internal
geometry, or create the optimizer with `rigid_body_invariant=False`.

The callable receives the flat Cartesian positions (Bohr), the lattice vectors (nine numbers,
or `None`), the iteration, and whether a Hessian is needed, and returns the energy (Hartree),
the gradient, and the Hessian or `None` (flat lists or NumPy arrays). Without a Hessian, one is
built from central differences of the gradient (`6N` extra calls). `check_term` runs the same
checks as for the built-in potentials; `check_bias` does so for any option, e.g. an expression.
Built-in options are available from Python with `Optimizer.add_bias("-kp", [...])` and
`evaluate_bias(...)`.

### Rust

A potential that depends on atom positions implements `pes::bias::GeometricBias` (energy
written once over `Scalar`) and is wrapped in `AdBias`, which supplies exact derivatives and
minimum-image geometry; one with auxiliary variables implements `AuxiliaryBias` and is wrapped
in `MicroBias`. Any other term implements `pes::PesTerm` directly. Terms are added with
`GeometryOptimizer::add_term` or placed in a `PesExpr`. To make a potential available as an
option (for example from an embedding program or Python), register a parser with
`cli::bias::BiasRegistry::register("-mybias", "my_bias", parser)`; the parser turns the
option's arguments into `BiasBuilder`s, which build the term for a structure.

## Derivatives in the constants of the bias potentials (`-bias_derivatives`)

`optmain ... -kp 0.002 4.6 1,2 -bias_derivatives` computes, once the optimization has
converged, how the result depends on every constant `p` of the bias options as they are
written: the force constant 0.002 and the distance 4.6 here, the collision energy of `-ma`, the
radius of a wall, the limits of a well (each number of an argument with commas counts on its
own). A number is a constant when the option still reads it after it is moved a little; atom
numbers read only as whole numbers, so they are not constants. The numbers inside the
expression of `-ubias` are not found. A derivative is per unit of the option as it is written:
kJ/mol for the AFIR collision energy, Angstrom for distances, degrees for angles, atomic units
for spring constants (MultiOptPy's unit of each option).

At the converged structure `x*` the run evaluates the Hessian `H` of the whole surface (the
calculator and the bias potentials). For each constant it computes the derivative of the bias
energy and of its gradient by central differences (the terms are built again with the constant
moved by `1e-4` of its value), and from them the following quantities.

| Quantity | Formula | Meaning |
|---|---|---|
| `dE_bias_dp` | `dE/dp` | derivative of the energy with the bias of the optimized structure (its gradient vanishes at `x*`, so this is the whole change) |
| `dE_unbiased_dp` | `g0 . dx*/dp`, with `dx*/dp = -H^+ dg/dp` | how much the energy without the bias of the optimized structure changes (`g0` is its gradient at `x*`) |
| `largest_atom_shift_angstrom` | largest atom of `dx*/dp` | how far the structure moves per unit |

The rigid motions are taken out where the surface does not depend on them, and the directions
of constraints (`-fix`, `-pc`, `-gfix`, `-uc`) are taken out, with the Hessian of the
Lagrangian. The table is `bias_derivatives.csv` in the run directory, and the log has the same
lines. `bias_derivatives_iterations.csv` has `dE_bias/dp` of every constant at every
iteration (MultiOptPy prints `dE_bias_pot/d_param` of every bias at every iteration; here it
needs no extra evaluation of the calculator, since the bias terms are evaluated alone). The response is linear: near a constant where the minimum disappears, the smallest
curvature (in the last column) becomes small and the response grows without bound.

`-bias_derivatives` needs a fixed cell (not `-vc`) and the whole surface (not ONIOM
microiterations yet). From Rust the parts are `pes::bias_response` (the derivatives and the
response for any terms) and `cli::bias_constants` (the constants of the options). From Python,
`mopt_rs._core.bias_constant_derivatives(option, args, symbols, positions_bohr, hessian=...,
unbiased_gradient=..., removed=...)` gives the same per constant for one option at any
structure (with a Hessian and the gradient without the bias, the response too).

`tests/cli_bias_derivatives.rs` checks it against whole optimizations. It optimizes two argon
atoms (`lj`) with a spring `-kp 0.002 4.6 1,2` again at each constant moved up and down, and
compares the differences of the energies with and without the bias and of the distance with
the reported derivatives.

| Constant | Quantity | Reported (Hartree per unit) | Re-optimized |
|---|---|---|---|
| distance 4.6 Angstrom | `dE/dp` | 2.226068e-4 | 2.226067e-4 |
| distance 4.6 Angstrom | `dE0/dp` | 2.287054e-4 | 2.287053e-4 |
| force constant 0.002 | `dE/dp` | 1.734559e-3 | 1.734742e-3 |
| force constant 0.002 | `dE0/dp` | 3.564160e-3 | 3.564546e-3 |

The re-optimized side moves the force constant by 1 % in each direction, so its own error is
about 1e-4 of the value.
