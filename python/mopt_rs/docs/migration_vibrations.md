# Differences from MultiOptPy: vibrational analysis and analyses of paths

Normal modes and thermochemistry, the curvature of a path, the orbitals along a path, and the
maps of a path.

## Vibrational analysis and thermochemistry

`-freq` (and `-force_freq`, `-freq_cutoff`, `-freq_corr`, `-temp`, `-press`) works as in
MultiOptPy: after the optimization the Hessian of the whole surface (calculator and bias
potentials) at the final structure gives the normal modes and the thermodynamic functions of
the ideal gas, written to `thermochemistry.txt`, `normal_modes.txt`, and
`vibration_animation/`; `symmetry.txt` has the point group of the final structure (`vib`).

`freqmain` (new in mopt-rs) does the analysis for given structures without an optimization,
with the gradient projected out on request (`-project_gradient`) and, with `-path`, the
analysis of a reaction path (see running.md).

### Thermochemistry choices in the manner of GoodVibes

MultiOptPy pairs the treatments of the entropy and of the thermal energy (`grimme` with the
thermal energy of Li et al., `truhlar` in both) and has one cutoff. mopt-rs keeps this as the
default and adds, following GoodVibes (read, not run): separate treatments and cutoffs
(`-freq_corr_h`, `-freq_cutoff_h`), scale factors of the frequencies (`-freq_scale`,
`-zpe_scale`), the translational entropy at a concentration (`-conc`), small imaginary
frequencies taken as real (`-freq_invert`), the moment of the molecule as the bound of the
free rotor (`-freq_bav molecule`), and in `freqmain` a range of temperatures
(`-temp_scan`), electronic energies from another level of theory (`-spc_energy`), and
Boltzmann populations of several inputs (`-boltz`).

Not taken over: the free-volume correction of the solvent (`--freespace`; GoodVibes marks
it as deprecated), the database of scale factors by level of theory (the calculators here
are not those levels; give `-freq_scale`), and the tools for selectivities and reaction
profiles from many output files (`--ee`, `--pes`), which belong to the workflow commands.

Tests: `vib::thermo::tests::choices_in_the_manner_of_goodvibes` (1 mol/L against 1 atm
lowers `T S` by `k T ln 24.47` = 1.89 kcal/mol, the value GoodVibes documents; scale
factors; entropy and energy apart; inversion; the bound of the rotor),
`tests/cli_freqmain.rs` (the options, the scan, the populations of two inputs 1 mHartree
apart).

### Anharmonic vibrational analysis

New in mopt-rs (`vib::qff`, `vib::vpt2`, `vib::vscf`, `freqmain -anharmonic`); MultiOptPy
has the harmonic analysis only.

**Displacement.** Largest change of a VPT2 fundamental against a displacement of 0.02 in
the dimensionless normal coordinates (cm^-1), for structures optimized to the tight
criteria (measured with `freqmain -anharmonic -qff_step`):

| Step | water, GFN-FF | water, GFN0-xTB | formaldehyde, GFN-FF | formaldehyde, GFN0-xTB |
|---|---|---|---|---|
| 0.05 | 0.05 | 0.12 | 0.02 | 0.07 |
| 0.1 | 0.26 | 0.53 | 0.09 | 0.30 |
| 0.2 | 1.08 | 2.18 | 0.41 | 1.27 |
| 0.3 | 2.46 | 4.93 | 0.98 | 2.88 |
| 0.5 | 7.01 | 13.8 | 3.05 | 8.09 |
| 1.0 | 31.0 | 55.8 | 16.6 | 33.4 |

The error falls with the square of the step, as that of central differences does, and no
noise of the Hessians shows down to 0.02. The default is 0.05: about 0.1 cm^-1 from the
limit, and a factor of six in the square of the step away from the smallest step tried, as
a margin for calculators whose Hessians are less precise.

**VSCF basis.** Largest change of a VSCF fundamental against 24 functions per mode (cm^-1):

| Functions | water, GFN0-xTB | formaldehyde, GFN-FF | formaldehyde, GFN0-xTB |
|---|---|---|---|
| 6 | 4.71 | 3.64 | 0.89 |
| 8 | 0.50 | 3.76 | 0.08 |
| 12 | 0.00 | 0.34 | 0.00 |
| 16 | 0.00 | 0.02 | 0.00 |

The default is 16. For water with GFN-FF the quartic expansion is not bounded from below
(the bend has a quartic constant of -470 cm^-1): taking the modals by their energies gave
12 cm^-1 for the bend with 16 functions and nonsense with 24. The modals are therefore
followed by their overlap; the bend then comes out at 1454 to 1462 cm^-1 for 8 to 32
functions and the stretches within 0.3 cm^-1, and the result says that the force field is
not bounded.

**Checks.** The force constants of a Morse oscillator from its Hessians agree with the
analytic derivatives, and VPT2 on them gives its anharmonic constant `-w^2 / (4 D)`
(`vib::qff::tests`). For three coupled modes with a weak anharmonicity the VPT2 zero-point
energy, fundamentals, overtones, and a combination band agree within 0.02 to 0.04 cm^-1
with the diagonalization of the same force field in a product basis, where the shifts are
0.3 to 2 cm^-1; a Fermi resonance (`2 w_0 = 2000`, `w_1 = 2010`, `phi = 60` cm^-1) is taken
out, and its pair of levels agrees within 0.3 cm^-1 with the diagonalization; VSCF is exact
for one mode and an upper bound of the ground state for three (`vib::vscf::tests`).

Not checked: the Coriolis terms against a reference program, and the results for real
molecules against another implementation (none is installed). The state-independent terms
of the rotational kinetic energy are not in the zero-point energy.

### Partial Hessian analysis

New in mopt-rs (`vib::partial`, `freqmain -partial_atoms`): the vibrations of chosen atoms
from the block of their Hessian or from the effective Hessian with the other atoms relaxed,
with the translations and rotations of the chosen atoms projected out (see running.md).

The effective Hessian inverts the block of the other atoms. When the chosen atoms are a
single atom or lie on a line, that block has directions without a restoring force (the
other atoms turning about the chosen ones). In a computed Hessian their curvature is not
zero: for water optimized with GFN-FF to the tight criteria, with O and one H chosen, the
block of the other H has the eigenvalues `2.0e-6`, `0.047`, and `0.474` Hartree/Bohr^2,
the first being the rotation about the O-H axis (its size follows the residual gradient).
A relative threshold that separates such a value from soft vibrations of a large
environment cannot be given in general, so these rotations are constructed and projected
out explicitly; the threshold `-partial_free` (`1e-10` of the largest eigenvalue) is left
for exact zeros, which occur in model Hessians.

Tests: `vib::partial::tests` (the ends of a chain of two springs give the force constant of
springs in series; blocks and the frozen frame), `tests/cli_freqmain.rs` (the local O-H
stretch of water lies between the symmetric and the antisymmetric stretch, and the relaxed
force constant with the reduced mass of O-H gives its wavenumber).

### Local vibrational modes

New in mopt-rs (`vib::local`, `freqmain -local_modes`): local force constants, masses, and
wavenumbers of chosen internal coordinates after Konkoli and Cremer, and the shares of the
normal modes in each local mode (see running.md).

Tests: for a force field that is quadratic in two distances and an angle with couplings
between them, the local force constants are `1 / (F^-1)_nn` of its force-constant matrix
(not the diagonal elements), the local stretch lies between the two stretches of the
molecule, and the effective Hessian of the two atoms of a bond gives the same force
constant (`vib::local::tests`); `tests/cli_freqmain.rs` runs the command for water with
GFN-FF. The results have not been compared with another program for local mode analysis.

### Phonons

New in mopt-rs (`vib::phonon`, `freqmain -phonon_supercell`): force constants from the
Hessian of a supercell, the dynamical matrix at any wave vector, the dispersion along a
path, the density of states, and the thermodynamic functions of the harmonic crystal per
cell. A force constant between an atom and an atom of another cell is assigned to the
nearest periodic image of the supercell and shared between images that are equally near;
the acoustic sum rule is imposed on the block of every atom with itself.

Tests: `vib::phonon::tests` (a monatomic chain with first and second neighbour springs has
`w^2 = (4 k / m) sin^2(pi q) + (4 k_2 / m) sin^2(2 pi q)`, also at wave vectors that are not
commensurate with the supercell; a diatomic chain has its acoustic and optical branch;
Einstein modes give the textbook free energy and heat capacity), `tests/cli_freqmain.rs`
(copper with EMT in a 2 x 2 x 2 supercell of the cubic cell: the zone centre reproduces the
vibrations of the cell computed directly, the lattice is stable along the path, the highest
wavenumber is 260 cm^-1 against about 240 cm^-1 in experiment, and the heat capacity at
298 K is 95 % of the classical value). No comparison with a phonon program has been made.

### Vibrations at a crossing point

With `-mf opt_mesx` or `-mf opt_meci`, `-freq` no longer analyses the Hessian that the
gradient projection hands to the optimizer (`P H_mean P` plus the curvature of the gap
term along the difference gradient): one of its modes is the gap direction, and the
bending of the seam is missing. The vibrations within the seam are analysed instead
(`vib::seam`, also `freqmain -mf ...`; see running.md), from the Hessians of the two states
and the multiplier of the seam condition.

Tests: `vib::seam::tests` (two wells of different stiffness, whose seam is a sphere: the
curvature agrees with the mean energy followed along the seam and differs from that of the
mean Hessian; a conical intersection next to its cusp: one vibration with the force
constant of the wells although the Hessians of the states have elements of `1 / gap`),
`tests/combined_surfaces.rs` (after a search with `opt_meci`, and through `optmain -freq`
and `freqmain` with `opt_mesx`). No calculator with excited states or analytic Hessians of
them was available, so the conical case is tested on the model only.

### Curvature of a reaction path

MultiOptPy has functions for the curvature of an intrinsic reaction coordinate
(`PESAnalyzer/calc_irc_curvature.py`, written to `irc_curvature_properties.csv` by the Euler
and Runge-Kutta integrators; the call in the LQA integrator is commented out). mopt-rs has
them in `vib::path`, used by `freqmain -path`, with these corrections:

| | MultiOptPy | mopt-rs |
|---|---|---|
| Curvature vector | `(g_i - g_(i-1)) / step` with the mass-weighted gradients themselves: this is the rate of change of the gradient along the path (the Hessian applied to the tangent), not the curvature of the path | `dt/ds` with the unit tangents `t`, from the parabola through three neighbouring points, without its components along the tangent and the rigid-body motions |
| Arc length | the nominal step size of the integrator | the mass-weighted distance between the points |
| Curvature coupling | components of that vector along all eigenvectors of the unprojected mass-weighted Hessian with eigenvalues above `1e-8`, which include the motion along the path when its curvature is positive | `B_kF = L_k . dt/ds` with the modes orthogonal to the path and to the rigid-body motions, of any curvature |

New in mopt-rs (no counterpart in MultiOptPy): tangents from the points for paths that are
not steepest-descent paths (`-path_tangent points`), the vibrationally adiabatic potential,
followed modes and their couplings `B_kl`, valley-ridge transitions, the components of the
path direction and curvature along internal coordinates and the reaction phases (unified
reaction valley approach), the free-energy profile with the variational transition state,
and the transmission coefficients of zero- and small-curvature tunneling.

The small-curvature effective mass is implemented from the formula given in running.md; the
text of the original papers (Lu et al., Comput. Phys. Commun. 1992, 71, 235; Liu et al., J.
Am. Chem. Soc. 1993, 115, 2408) was not available when it was written, and the result has
not been compared with a reference program. The other quantities are tested against
analytic cases: a circular valley (curvature `1 / (sqrt(m) R)`, coupled to the radial mode
alone, effective mass), two modes that cross while turning into each other (`B_kl` is the
rate of rotation), a parabolic barrier (transmission coefficient `(u / 2) / sin(u / 2)`,
`u = hbar w / k T`), a mode that stiffens along the path (the variational transition state
moves off the saddle point), and a stretching bond (direction component
`sqrt(m_O / (m_O + m_H))`); `tests/cli_freqmain.rs` runs the command for a stretching bond
of water and for the umbrella inversion of ammonia with GFN-FF.

### Corrections to MultiOptPy's thermochemistry

| | MultiOptPy | mopt-rs |
|---|---|---|
| Rotational constants | `h / (4 pi I)`, which is `2 pi` too large (the formula it is taken from has `hbar`); the rotational entropy of a nonlinear molecule comes out `(3/2) R ln(2 pi)` = 2.76 R too small | `h / (8 pi^2 I)` |
| Centre of the moments of inertia | the coordinates are moved to the centre of mass, and the original centre of mass is subtracted again | the centre of mass |
| Electronic entropy | the multiplicity is `min(1, multiplicity)`, so it is always zero | `R ln(2S + 1)` |
| Symmetry numbers | 1 for the groups Th, O, and I | the number of proper rotations found (12, 24, 60) |
| Exponentials | `exp(-x) / (1 - exp(-x))` with the denominator floored at `1e-10` | `x / expm1(x)` and `ln1p(-exp(-x))`, accurate for all frequencies |
| Animations | the unnormalized mode times 3 is added to the coordinates in Bohr | the atom that moves most is displaced by 0.3 Angstrom |
| Single atom | point group `D∞h` | `Kh` |

What is called `grimme` interpolates the entropy between the harmonic oscillator and a free
rotor as in Grimme's paper, and the thermal energy between the oscillator and `R T / 2` as in
Li, Gomes, Sharada, Bell, and Head-Gordon (2015); the heat capacity is not corrected. This is
MultiOptPy's choice and is kept. The zero-point energy is harmonic in all treatments, and
modes at or below 10 cm^-1 and imaginary frequencies do not enter.

Tests: `vib::*::tests` (Sackur-Tetrode entropy of argon, entropy of N2 from its bond length
and frequency, the low-frequency treatments, the report layouts), `tests/cli_optmain.rs`
(`-freq` for water with GFN-FF).

### Point group without sampling axes

MultiOptPy tests rotations of every order up to a limit, reflections, and rotation-reflections
about candidate axes built from the atoms (principal axes, directions to atoms, and sums and
cross products of pairs of the first 50 atoms: several thousand axes), with a distance
tolerance of `1e-4` Bohr from the vibrational analysis.

mopt-rs uses that a symmetry operation is fixed by the images of two atoms (`vib::symmetry`):
with the atoms sorted into shells of equal element and distance from the centre of mass, the
candidates are the pairs of images of two reference atoms within their shells, each giving one
rotation and one rotation-reflection. There are about as many candidates as the smallest shell
has atoms (one for a molecule without symmetry), every operation is among them, and the
rotations found are counted for the symmetry number. The group is named from the orders of the
operations as permutations of the atoms, the inversion, and the mirror planes, which does not
depend on how accurately the operations hold. The operations are checked for closure, and the
tolerance is halved when they do not form a group.

The tolerance is 0.01 Bohr by default (`FrequencySettings::symmetry_tolerance`): a structure
from an optimization is symmetric only within the accuracy of the optimization, and with
MultiOptPy's `1e-4` most optimized structures come out as `C1`. `-nosymm` skips the analysis.

Tests: `vib::symmetry::tests` (20 groups from `C1` to `Ih`, linear molecules, tolerances).

## Orbitals and maps of a path

### Natural reaction orbitals and orbital energies along a path

MultiOptPy has `WFAnalyzer/MO_analysis.py:NROAnalysis`, but nothing calls it (there is no
option for it, and no calculator sets the `orbital_coefficients`, `overlap_matrix`, and
`orbital_energies` it reads). mopt-rs has the two analyses it combines as separate ones,
`freqmain -nro` and `freqmain -roet`, implemented from the papers:

| | MultiOptPy (`NROAnalysis.run`) | mopt-rs |
|---|---|---|
| what is decomposed | the whole matrix `C^T S dC/ds`, including the blocks within the occupied and within the empty orbitals | the block between empty and occupied orbitals (Eq. (17) of Ebisawa et al. 2022); mixing among occupied orbitals does not change the wavefunction |
| what is reported | the sum of the singular values, rescaled to 0..1 over the run | the singular values, their squares' sum (the quantity of the paper), the shares of the pairs, and the pairs themselves |
| which orbitals are subtracted | orbital `k` of one calculation from orbital `k` of the other, as they come | orbitals matched by their overlaps: order, rotation of degenerate ones, and sign (`analysis::orbitals::tracking`) |
| direction and step | the optimizer's step, `1e-4` Bohr in Cartesian coordinates | the path, a unit vector of mass-weighted coordinates without rigid motion; `0.01` amu^(1/2) Bohr |
| orbital energies | first and second differences between the three structures of one step | energies of orbitals followed along the whole path, derivatives with respect to the arc length, and the reactive orbitals |
| spins | not distinguished | alpha and beta separately |
| programs | tblite (as a comment says) | built-in GFN0-xTB, PySCF, tblite (`Calculator::orbitals`, `Engine.orbitals`) |

The step: singular values of water (O-H stretch), ethylene (CH2 twist), and the OH radical
with GFN0-xTB, relative to the limit of a small step:

| step (amu^(1/2) Bohr) | 0.1 | 0.05 | 0.02 | 0.01 | 0.005 | 0.001 |
|---|---|---|---|---|---|---|
| largest error of the sum of squares | 3.4e-3 | 8.6e-4 | 1.4e-4 | 3.5e-5 | 8.6e-6 | 3.5e-7 |

The error falls with the square of the step. `0.01` keeps it below `1e-4` while the
orbital coefficients of an iterative program, converged to about `1e-6`, still enter the
difference with a factor of 50 only; with GFN0-xTB (no iterations) smaller steps are
equally good. Not measured: the same table with PySCF or tblite.

The reactive orbitals follow Tsuneda and Taketsugu's statement of the theory
(arXiv:2412.11361): the occupied one is the most stabilized occupied orbital, by the
relative change `2 (e_prod - e_reac) / |e_prod + e_reac|` between the two ends. That text
gives no formula for the unoccupied one; mopt-rs takes the empty orbital with the largest
relative change in magnitude and says so in its report. The normalized reaction diagrams
of Tsuneda and Singh (2014) were not at hand and are not reproduced.

Not done: natural reaction orbitals from the coupled-perturbed equations (the differences
of the orbitals are used throughout, as the paper allows), restricted open-shell and
general (non-collinear) wavefunctions, pictures of the orbitals (the coefficients are
written), and a comparison with the authors' program.

Tests: `analysis::orbitals::*::tests`, `workflow::orbital_analysis::tests`,
`tests/orbital_path.rs`, `tests/test_orbitals.py`.

### Maps of a path and Koopman analysis (`-cmds`, `-pca`, `-km`)

MultiOptPy's `PESAnalyzer` classes are ported to `analysis::path_map` and
`analysis::koopman` with the same file names. What differs:

| | MultiOptPy | mopt-rs |
|---|---|---|
| structures | the first one of the run is left out, and the last row of the CMDS table is not written | all structures, energies relative to the first |
| superposition | Kabsch rotation of coordinates multiplied by `sqrt(m)`, centered on the plain mean of those | mass-weighted superposition (centers of mass, weights `m`), which is what minimizes `sum m |r - r'|^2`; the same for equal masses |
| CMDS eigenvalues | general eigenvalue solver on the symmetric matrix, complex parts dropped | symmetric solver; each coordinate has the sign that puts the first structure on its negative side |
| units of the axes | labeled "ang. / amu^0.5" | Angstrom amu^(1/2), which is what both compute |
| PCA variables | every variable divided by its standard deviation (a coordinate that does not move gives a division by zero), the covariance matrix of all variables is built (for distances, the square of `N (N - 1) / 2`), and the structures are projected without centering | Cartesian coordinates and distances are not rescaled (they share a unit); the chain of internal coordinates is standardized, constant variables staying out; the work is done on the matrix of the structures, whose size is the number of iterations; scores are those of the centered variables |
| PCA superposition | each structure on its successor | every structure on the first |
| Z-matrix | the angle of atom `i` is computed with the wrong distance in its denominator, and dihedral angles jump at 180 degrees | distance, angle, and dihedral angle of the chain `i, i - 1, i - 2, i - 3`; dihedral angles are continued from structure to structure |
| Koopman rank | the header of `koopman_eigs.csv` is written for the rank of the first analysis; later rows may have other lengths | ten columns of modes, empty where the rank is lower |
| Koopman animations | old animation files are looked for in the working directory, not where they are written | written into `koopman_analysis/`, replaced at every iteration |
| Koopman modes log | eigenvalues only | with what each eigenvalue means for the iterations (MultiOptPy has this as a table in a comment) |
| figures | PNG through matplotlib | PNG without lettering and SVG with it |

The observables, the window of ten iterations, the rank threshold `1e-5`, and the
animation (ten frames, amplitude 1.5 Bohr) of the Koopman analysis are MultiOptPy's. The
singular value decomposition is taken from the `m x m` matrix of the iterations.

Tests: `analysis::path_map::tests`, `analysis::koopman::tests`, `tests/path_analysis.rs`,
`tests/test_parts.py`.
