# Excited states

Excited states of the built-in methods and of other programs, following one state from structure
to structure, conical intersections, and diabatic states. Vibrations within a crossing seam are
on [Vibrational analysis](vibrations.md).

## Excited states (`-es`) and conical intersections with the built-in GFN0-xTB and GFN1-xTB

`-es N` selects the electronic state `N` (0: the ground state) of a calculator that has
excited states, in `optmain` (the optimization runs on that state) and in `freqmain`. With
`-mf NAME` the first state is the state `N`; the second is the lowest state of the charge
and multiplicity given after the name (`-mf opt_mesx 0 3`: a crossing of states of
different spin) or, without them, the next state above the first (`-es 1 -mf opt_meci`: a
conical intersection of the first and the second excited state). The excitation energies
at the starting structure are written to `excitations.txt`.

A search on a gradient projection of the two states (`-mf opt_mesx`, `-mf opt_meci`) with an
RS-I-RFO method (the default methods are) has 0.1 Angstrom (0.189 Bohr) as its largest trust
radius unless `-tr` gives one: next to a seam the difference gradient, and with it the
projection, turns within a short distance, so the quadratic model of a step holds over a
shorter distance than on one surface (0.5 Bohr for a minimization there).

`-projection_trust merit` changes what a step of such a search is judged by when the trust
radius is adapted. The projected gradient `2 (E1 - E2) x + P (g1 + g2) / 2` is not the gradient
of the mean energy that the search reports, so the ratio of the change of that energy to the
prediction of the quadratic model (`energy`, the default, as on one surface) is no measure of
the model. With `merit` the change is that of the function the projected gradient does belong
to, the mean energy within the seam and `(E1 - E2)^2 / |g1 - g2|`
(`opt::state_machine::SeamPoint`). Measured on a model of three atoms with two surfaces
(tight criteria, `tests/cli_mapper_states.rs`): the searches from a minimum take 65 instead of
72 iterations for the MECP and 99 instead of 124 for the MECI; the refinements of the
candidates of AFIR paths are unchanged for the MECP, and for the MECI one more converges on
the upper Avoiding model function (5 of 12 instead of 4) and one fewer on the lower (10 of 11
instead of 11, 883 iterations instead of 507). The effect is not of one sign, so the default
stays `energy`.

The built-in `gfn0xtb` has excited singlet states of closed-shell molecules through a
simplified Tamm-Dancoff approximation on its orbitals, `A = (e_a - e_i) + 2 (ia|jb)'` with
Coulomb integrals of Loewdin transition charges, the form sTDA takes without exact
exchange. It is an addition of mopt-rs, not part of GFN0-xTB, and nothing in it is fitted
to excitation energies: they are qualitative. Its use is to have surfaces of several
states of one spin that cross as real ones do.

The built-in `gfn1xtb` has excited singlet states of closed-shell molecules with integer
occupations as well, also meant for tests of the methods that work on several states rather
than for photochemistry. By default they come from the Tamm-Dancoff approximation (TDA) of the
linear response of its own self-consistent charges: the same `A = (e_a - e_i) + 2 q K q^T`
with Mulliken transition charges of the shells and the kernel of GFN1-xTB itself (its shell
Coulomb kernel and the third-order term, `2 Gamma_A q_A` between the shells of an atom), so
nothing in it is empirical beyond the ground state. GFN1-xTB has no spin-dependent term, so
only singlets are given. `-calc_opt excited=stda` puts the sTDA form of `gfn0xtb` on its
orbitals instead, with its chemical hardnesses (sTDA and the TDA are different methods: sTDA
replaces the kernel by a damped Coulomb interaction of atomic hardnesses).
The gradients of the excited states of both are analytic: the excitation energy is
stationary in its amplitudes, and the response of the orbitals comes from one Z-vector
(Furche and Ahlrichs 2002 for the Lagrangian of TDDFT, Heringer, Niehaus and Frauenheim 2007
for its tight-binding form): `(e_a - e_i) Z + 4 q K q^T Z = L` with the kernel of the
self-consistent charges of `gfn1xtb`, `Z = L / (e_a - e_i)` for `gfn0xtb`, whose Hamiltonian
has no self-consistency. The relaxed difference density and the weights of the overlap and
of the kernels are contracted with the derivatives of the Hamiltonian as the ground state's
density is (`pes::native::tda_gradient`). Against the central differences that follow the
states, on formaldehyde and on butadiene without symmetry, the two lowest states of
`gfn0xtb` sTDA, `gfn1xtb` TDA and `gfn1xtb` sTDA agree to 1e-8 to 1e-7 Hartree/Bohr, and the
gradient takes 0.017 instead of 1.06 s for butadiene (60 times faster; 15 to 28 times for
formaldehyde). `-calc_opt excited_gradient=numerical` takes the central differences
instead, with the displacement `-calc_opt excitation_step=BOHR`.

```
optmain o3.xyz -os gfn0xtb -es 1 -mf opt_meci -opt rsirfo_bfgs -freq
```

Limits:

- Closed-shell singlets only, and all single excitations are diagonalized (at most 4000,
  a few tens of atoms).
- The gradient of an excitation energy is that of the state of its index, which is not
  defined where two states are degenerate (the eigenvector is not); near such a point
  `-calc_opt excited_gradient=numerical` takes central differences (two evaluations per
  coordinate) in which the states are followed by their overlaps, which stay meaningful
  where two states come close. A Hessian of an excited state is a difference of its
  gradients.
- As in every Tamm-Dancoff method the ground state takes no part in the response: a
  crossing of the first excited state with the ground state (`-es 0 -mf opt_meci`) is not
  a conical intersection here.
- Where the gap between occupied and empty orbitals closes (a breaking bond), the
  closed-shell reference fails, and the calculation stops with a message that says so.

**Check of the branching plane.** Gradient projection has no derivative coupling; it
builds the second direction of the branching plane from the steps of the search, starting
from the mean gradient. In a search that keeps the symmetry of its start the coupling
direction never appears in the gradients, and the search stops where the gap closes while
the energy still falls along the seam. When a search with `-mf opt_meci` has converged,
mopt-rs therefore evaluates both states at two structures displaced by 0.01 Bohr along a
direction without symmetry (and once more along the direction found): the change of the
difference of the two gradients lies in the branching plane and gives its second direction
without a Hessian. If the mean gradient within the seam does not meet the convergence
criteria with that direction, the search takes it as its branching plane and goes on (at
most five times). If the check still fails after the fifth correction, the run ends as not
converged (the termination `SeamCheckFailed`: `optmain` reports the run as not converged,
follow-up analyses that need a converged structure are not run, and the mapper treats the
crossing as a failed search) instead of accepting the structure. The log tells the overlap
of the two directions and the gradient. With constraints (`-fix`, `-pc`, `-uc`), the tested
gradient and the displacement directions are those within the constraints, as in the
iterations of the search (before, a frozen atom made the check fail every time). The check
costs four evaluations of both states; `-no_crossing_check` turns it off.

## Staying with one excited state (`-es_follow`)

`-es N` counts the states by their energies at every structure. Where two states change
places along an optimization, a path, or a chain of images, state `N` is then another state
than before: a minimization of the second excited state that reaches a crossing with the
first stays on the upper surface and ends at the crossing, although the state it started
on goes on below it.

With `-es_follow`, state `N` is the state that was number `N` at the first structure it
was asked for, and at every later structure the state that it has become. The states of a
new structure are compared with those of the nearest structure evaluated before: the
amplitudes of the single excitations are brought into the orbitals of that structure by
the overlaps of the orbitals, and the scalar products of the amplitudes are the overlaps of
the states. All the states that are followed (those of `-mf` as well) are assigned at once,
each to another state of the new structure. The log says where a followed state changes its
place in the order of the energies, with the squared overlap:

```
electronic state 2 (as counted where it was asked for first) is state 1 in the order of
the energies here (state 2 at the structure before; squared overlap 0.999)
```

and where a state is not found again with certainty (a squared overlap below one half: the
state has left the six states above the highest followed one that are looked at, or it has
mixed with another), or where less than 0.9 of the occupied orbitals stay occupied (the
reference of the excited states changes its character). The run goes on in both cases.

The option belongs to the calculator settings and works in every command (`optmain`,
`ircmain`, `nebmain`, `ieipmain`, `freqmain`, `mdmain`, ...). What a state is depends on
which structure comes first, so with the option the evaluations of a batch (the images of a
chain, the two structures of `ieipmain`) are made one after the other in their order and
not in parallel: the first image defines the states, and every other structure is compared
with the nearest one evaluated before it. It needs a calculator that hands out its states: the built-in `gfn0xtb`, and
PySCF with a restricted closed-shell reference (singlet Tamm-Dancoff states). The other
calculators say that their states cannot be followed. A calculator without excited states
says once that nothing is followed. With ONIOM the states of the highest level are
followed.

Restart: the journal records which state by energy every evaluation was for, and a
restarted run goes on from the last recorded one (one calculation of the states of that
structure is repeated). With ONIOM and microiterations the journal holds the evaluations
of the relaxed surface, not those of the highest level, and a restarted run counts the
states anew at the first structure it computes; this case is not covered yet.

What it is and is not:

- The comparison leaves out the change of the basis functions between the two structures
  (the orbitals are compared in the Loewdin orthogonalized atomic orbitals), as the
  gradients of the excitation energies of the built-in GFN0-xTB do. Between neighbouring
  structures of an optimization this is a small effect; for structures far apart the
  overlaps fall and the report of a state "not found again with certainty" appears.
- At a conical intersection the two states are degenerate and only defined as a pair.
  Either of them may then be given either number; the pair stays the pair. A search for a
  crossing (`-mf opt_meci`) with `-es_follow` therefore stays with the two states it
  started with, which is its use there.
- It follows the character (the diabatic state) through a crossing. Whether that or the
  state by energy is wanted depends on the question: the default is the state by energy,
  as before.
- PySCF computes the Tamm-Dancoff states once more per structure for the comparison (the
  SCF restarts from its converged density).

Measured on ozone with the built-in GFN0-xTB (the first and the second excited singlet
state cross on the line from the minimum of the first through a structure next to the
minimum of the second): along 17 points of that line the followed states changed places
once, with a squared overlap of 0.999; the largest second difference of the energy along
the line was below 1e-3 Hartree for the followed states and above 3e-3 for the states by
energy (the kink at the crossing). From a structure behind the crossing, `optmain -es 2
-es_follow` went through the crossing to the minimum of the state (the first by energy
there, within 1e-6 Hartree of the minimization of the first state); without the option the
run ended more than 0.01 Hartree above it. One molecule and one pair of states.

## Excited states of a program outside the package (`-calc_opt excited_setup=`, `state_reader=`)

The engines of the package that hand out their excited states are the built-in GFN0-xTB
and PySCF. For a program run through ASE (the programs of "Programs run through files",
any `ase:module:Class`) the package has the frame and no reader: which keywords choose a
state and where the amplitudes are written differs from program to program and from
version to version, and none of these programs was run here. Two functions of your own
fill the frame, both named on the command line:

```bash
optmain start.xyz -os orca -func b3lyp -bs def2-SVP -es 2 -es_follow -calc_opt excited_setup=my_orca:choose state_reader=my_orca:read
```

- `excited_setup=module:function`: `function(calculator, state, count)` is called before
  every calculation with the ASE calculator; it sets the keywords of the program so that
  the calculation is for the excited state `state` (`0`: the ground state) and computes at
  least `count` states. With it the engine takes `-es N` (energies and gradients of that
  state, as the program gives them).
- `state_reader=module:function`: `function(calculator, structure, count)` is called
  after a calculation of `count` states and returns what the program wrote as
  `mopt_rs.calculators.program_states.RawStates`: the coefficients of all orbitals in the
  basis functions of the program (a column per orbital, the doubly occupied ones first),
  the overlap of the basis functions (or `None` for an orthonormal basis), the number of
  occupied orbitals, the excitation energies in Hartree, and the amplitudes of the single
  excitations as `[state, occupied, empty]`. The package orthogonalizes and normalizes
  them. With it `-es_follow` works.
- `state_reader=file` reads `mopt_states.json` in the directory of the program instead
  (the keys `coefficients`, `overlap`, `occupied`, `energies`, `amplitudes`;
  `write_states_file` writes one), for a hook or a wrapper script that converts the output
  of the program after every run without a Python reader.

```python
# my_orca.py, somewhere on the search path of Python
from mopt_rs.calculators.program_states import RawStates

def choose(calculator, state, count):
    ...  # set the keywords of the program for this state and number of states

def read(calculator, structure, count):
    ...  # parse the files in calculator.directory
    return RawStates(coefficients, occupied, energies, amplitudes, overlap)
```

What the reader has to hand over is restricted closed-shell orbitals, all of them (frozen
or removed orbitals are refused), and singlet single excitations (the X of a Tamm-Dancoff
or a full calculation). A structure costs one calculation of the states and one of the
gradient of the state that is followed. The frame is tested with a model program that
writes such a file (two states that cross; the run follows its state through the
crossing); with a real program it is untested. Without a reader a run with `-es_follow`
ends with an error that says the states are not handed out, and the engine says which
setting is missing.

## States as diabatic states (`freqmain PATH.xyz -diabatize N M ...`)

Where excited states come close along a path, the states counted by their energies
exchange their characters: their energies have a kink (a crossing) or a narrow gap (an
avoided crossing). Diabatic states keep their characters; their energies cross, and what
the calculation shows as a gap is their coupling. The energies and the couplings are the
diagonal and the rest of the diabatic potential matrix `W`; the coupling and the slopes
at a crossing are what a Landau-Zener estimate of the passage takes.

```bash
freqmain path.xyz -os gfn0xtb -diabatize 1 2 -temp 298.15
```

```bash
freqmain path.xyz -os gfn0xtb -diabatize 1 2 3
```

`path.xyz` holds the frames of a path (an IRC, the last chain of `nebmain`, a scan); the
numbers are those of two or more states at the first frame. The method is the
propagative block-diagonalisation diabatisation (Neville, Seidu, Schuurman,
arXiv:1912.09553; the block-diagonalisation of Pacher, Cederbaum and Koppel), which needs
the overlaps of the adiabatic states of neighbouring frames only: the overlaps of the
diabatic states of a frame with the adiabatic states of the next are made orthogonal with
the least change, which gives the diabatic states of the next frame, and `W` is the
diagonal matrix of the adiabatic energies in them. The diabatic states are therefore the
states of the first frame, carried along; the first frame should be away from the
crossings. At every frame the states are those that hold most of the states of the frame
before (another state that passes through in energy is not taken for one of them). The
run directory gets

- `diabatic_states.csv`: for every frame the position (amu^(1/2) Bohr), the numbers and
  the energies of the adiabatic states, the upper triangle of `W` (`w_1_1`, `w_1_2`, ...,
  Hartree, named by the numbers of the states at the first frame), which adiabatic state
  every diabatic state mostly is and with which share, and how much of the states the
  next frame keeps;
- `diabatic_crossings.txt`: where two diabatic states cross, their coupling and the
  difference of their slopes there, and the Landau-Zener probability `exp(-2 pi H12^2 /
  (hbar v |dF|))` that one passage stays on its diabatic state, for the kinetic energy
  `kT/2` along the path at `-temp`; for every two states that are neighbours in energy
  the same from the gap alone (two straight states with a constant coupling fitted to the
  smallest gap: the coupling is half that gap);
- `diabatic_states.png` / `.svg`: the adiabatic energies as points and the diabatic ones
  as lines.

How exact it is: where the states that are followed are all the states that mix along
the path, the overlap matrices are orthogonal and `W` is the Hamiltonian in the states of
the first frame whatever the distance of the frames (three model states: the same to
1e-11 for steps from 0.05 to 2). Where a state outside them mixes in, a step keeps less
than all of the states, the result depends on the distance of the frames and gets better
with closer frames (a fourth model state coupled to the three: the error fell by more
than ten between steps of 1 and of 1/16), and the file says how much the weakest step
kept; below 0.9 the step is pointed out. More states in the list is the remedy for a
state that mixes in.

The calculator must hand out its excited states (the built-in GFN0-xTB, PySCF, a program
with a reader as above). Limits: the overlaps leave out the change of the basis functions
between two frames, as `-es_follow` does; the probability is the one-dimensional
Landau-Zener formula for two states, one passage, and one speed, an estimate and not a
rate constant; a ground state and an excited state are not treated (the excited states
come from one reference). From Python: `mopt_rs._core.diabatic_states`,
`landau_zener_stay`, and `state_overlaps(..., signed=True)` for the overlaps.

Measured on ozone with the built-in GFN0-xTB (33 frames on the line through the crossing
of the first and the second excited singlet state, as in the section before): on the line
the two states cross, with a coupling of 1.5e-5 Hartree at the crossing against gaps of
1e-3 Hartree at the frames next to it, and the largest second difference of the energies
along the line falls from 1.5e-3 Hartree (adiabatic) to 1.4e-4 (diabatic). With every atom
moved 0.05 Bohr off the line the crossing is avoided: coupling 1.430e-3 Hartree from the
overlaps, 1.426e-3 from the gap alone, and 1.431e-3 as half the smallest gap of the
frames. With the third state in the list the run goes through as well (the trace of `W`
is the sum of the three energies at every frame). One molecule and one line.
