# Vibrational analysis and orbitals along a path

Normal modes, thermochemistry, and the analyses of `freqmain` and `optmain -freq`, and the
orbitals followed along a path. Excited states have a page of their own ([Excited
states](excited_states.md)).

## Vibrational analysis and thermochemistry (`-freq`)

`optmain ... -freq` analyses the converged structure (`-force_freq`: also when the run did
not converge): the Hessian of the surface there, normal modes and harmonic wavenumbers
(`normal_modes.txt`; imaginary frequencies are negative), animations of the modes
(`vibration_animation/`), and the thermodynamic functions of the ideal gas
(`thermochemistry.txt`).

| Option | Meaning | Default |
|---|---|---|
| `-temp` | temperature (K) | 298.15 |
| `-press` | pressure (Pa) | 101325 |
| `-freq_corr` | treatment of low frequencies: `grimme`, `truhlar`, `none` | `grimme` |
| `-freq_cutoff` | cutoff of that treatment (cm^-1) | 100 |
| `-nosymm` | no point-group analysis (symmetry number 1) | off |

| `-freq_lowest` | vibrations at or below this wavenumber (cm^-1) do not enter the thermochemistry | 10 |
| `-symm_tol` | distance (Bohr) within which atoms coincide in the point-group analysis | 0.01 |
| `-freq_corr_h` | treatment of the thermal energy apart from the entropy: `head-gordon`, `truhlar`, `none` | the one that goes with `-freq_corr` |
| `-freq_cutoff_h` | cutoff of that treatment (cm^-1) | `-freq_cutoff` |
| `-freq_bav` | moment of inertia that bounds the free rotor of `grimme`: `global` (1e-44 kg m^2) or `molecule` (the mean moment of the molecule) | `global` |
| `-freq_scale` | scale factor of the frequencies | 1 |
| `-zpe_scale` | scale factor of the frequencies in the zero-point energy | `-freq_scale` |
| `-freq_invert` | imaginary frequencies up to this magnitude (cm^-1) count as real ones | off |
| `-conc` | concentration (mol/L): translational entropy of a solute at this concentration instead of the gas at `-press` | off |
| `-project_gradient` | project the direction of the gradient out as well | off |

`freqmain structure.xyz -os ...` does the same analysis for a given structure without an
optimization (run directory `<date>/<stem>_FREQ_<timestamp>/`); it takes the options above
and those of `optmain` that describe the surface (calculator, charge and spin, bias
potentials, `-modelhess o1numhess` for the Hessian). A structure that is not a stationary
point is reported; `-project_gradient` then gives the modes orthogonal to the gradient. In a
periodic cell the modes are those at the zone centre (translations projected out), and the
thermodynamic functions of a molecule are not written.

Only `freqmain` has these:

| Option | Meaning |
|---|---|
| `-temp_scan START,END,STEP` | the thermodynamic functions at these temperatures (K) as well: `thermochemistry_scan.csv` |
| `-spc_energy E1 E2 ...` | electronic energies (Hartree) from another level of theory, one per input, in place of those of the calculator |
| `-boltz` | with several inputs (conformers): relative Gibbs free energies and Boltzmann populations at `-temp`, in `<date>/boltzmann_<time>.csv` |

The entropy treatments (`-freq_corr grimme|truhlar`), the thermal energy of Li et al.
(`head-gordon`), the scale factors, the concentration, and the inversion of small imaginary
frequencies follow GoodVibes (Luchini, Alegre-Requena, Funes-Ardoiz, Paton, F1000Research
2020, 9, 291), so that the numbers can be compared with it: `-freq_corr grimme -freq_corr_h
head-gordon` is its `-q`, `-freq_corr_h none` its default without `--qh`, `-freq_cutoff`
and `-freq_cutoff_h` its `--fs` and `--fh`, `-freq_scale` its `-v`, `-conc` its `--conc`,
`-freq_bav molecule` its `--bav conf`.

`freqmain path.xyz -os ... -path` takes the frames of the file as the points of a reaction
path in a common orientation (the points of an intrinsic reaction coordinate) and analyses
the path and the valley around it; see "Analysis of a reaction path" below.

The symmetry number comes from the point group of the structure, which is also written to
`symmetry.txt` after every converged optimization. The Hessian is the calculator's own,
finite differences, or `-modelhess o1numhess`. From Rust: `vib::analyze`, `vib::normal_modes`,
`vib::thermochemistry`, `vib::point_group`.

### Rigid motions, linearity, masses, and imaginary modes

**Imaginary modes.** A mode is imaginary when its eigenvalue of the mass-weighted Hessian is
below `-1e-8` Hartree / (Bohr^2 amu) (`opt::mode_kill::IMAGINARY_CURVATURE`, about
-0.51 cm^-1; `vib::is_imaginary_curvature`, `NormalModes::imaginary_modes`,
`vib::imaginary_wavenumbers`). This one criterion counts the imaginary modes of `-freq` (the
summary of a run, the table of minima of confsearch, `freqmain`), of the thermochemistry
("Imaginary frequencies (not included)"), of the seam analysis, of the anharmonic and local-mode
analyses (which need a minimum), and the valley-ridge transitions along a path; it is that of
the mode kill, the IRC and the mapper. (Before, `-freq` and confsearch counted only modes
below -20 cm^-1.) The projected rigid motions have curvatures of zero and never count;
`-freq_lowest` (10 cm^-1) only decides which modes enter the sums of the thermochemistry.

**One linearity test.** A structure is linear when the smallest principal moment of its atoms
(unit weights, about their centroid) is at most `1e-6` of the largest
(`math::projection::LINEAR_MOMENT_RATIO`, `math::projection::shape`): the RMS distance of the
atoms from their best line at most `1e-3` of their RMS spread along it. The projection (two
rotations instead of three), the rotor of the thermochemistry (linear or not), the number of
vibrations (`3N - 5` or `3N - 6`) and the point group (`D∞h`, `C∞v`) all take this answer.
Derivation: an optimized structure is accurate to its RMS displacement criterion, `1e-3` Bohr;
the smallest linear structures of three atoms or more spread over `z_rms` of 1.4 to 2 Bohr, so
their noise gives a ratio below `(1e-3 / 1.4)^2 = 5e-7`; a symmetric triatomic bent by `alpha`
has the ratio `alpha^2 / 12`, so the threshold is a bend of 3.5e-3 rad (0.2 degrees) and a
bend of one degree (2.5e-5) is far above it. Before, three tests disagreed: the projection
dropped the rotation about the axis below a vector norm of 1e-7 (orientation-dependent), the
thermochemistry called a rotor linear above a rotational constant of 1e8 GHz, and the point
group used its tolerance of 0.01 Bohr. Measured with CO2 and HCN bent at the carbon (test
`vib_rigid_consistency`, a valence model Hessian): from 1e-7 rad (3e-7 rad for a molecule along
no Cartesian axis) to 1e-3 (CO2) or 2e-3 rad (HCN) six directions were projected while the
rotor was linear, and up to 1e-2 rad the point group was `D∞h` or `C∞v` with a non-linear
rotor; in that window one bending mode was missing from the zero-point energy (0.88 kcal/mol
for the CO2 model, 0.95 for the HCN model). Now projection, rotor, count and point group agree
at every bend of the scan.

**Periodic structures.** In a cell with at least one periodic axis all three translations are
projected (a slab moves freely along its normal, a wire across its axis), and for a wire (one
periodic axis) also the rotation about its axis; the rotations of a slab or a crystal turn the
lattice and are not symmetries. Before, only the translations along the periodic axes were
projected, so a slab kept a zero "vibration" along its normal and a wire three (test
`vib_rigid_consistency`: an argon slab has 3 projected motions and `3N - 3` modes, a wire 4 and
`3N - 4`). Frozen atoms, frame-fixing constraints and frame-dependent terms switch the full
projection off as for molecules (`pes::term::frame_invariant`), and the rigid motions that keep
the constraints are projected as described above.

**Masses (`-freq_masses`).** `standard` (the default: the standard atomic weights, those of the
molecular dynamics) or `isotope` (the most abundant isotopes, those of MultiOptPy and of the IRC
option `masses=isotope`), for the normal modes, the rigid motions, the translation and the
rotation of the thermochemistry, and `-path`. `thermochemistry.txt` and the log name the
choice. `-anharmonic`, `-local_modes`, `-partial_atoms`, `-phonon_supercell` and `-mf` weight
with the standard weights only and are refused with `isotope`. The masses of other isotopes
are not tabulated in the crate. From Rust: `vib::MassChoice`, `vib::normal_modes_with_masses`,
`vib::transverse_profile_with_masses`.

**Low and high temperatures.** The vibrational terms are written in decaying forms
(`x e^-x / (1 - e^-x)`, `x^2 e^-x / (1 - e^-x)^2`, `ln(1 - e^-x)` through
`math::stable::log_diff_exp`), so they stay finite for any `x = theta / T`: the heat capacity
of a mode of 4000 cm^-1 was NaN below 8 K (`x` above 709, `0 * inf`), and at 1 K for any mode
above about 500 cm^-1. Checked against 400-digit values at 1, 8, 298.15 and 5000 K
(`vib::thermo` tests, `vib_rigid_consistency`); the phonon thermodynamics use the same terms.

**`-freq` after `-vc`.** The vibrational analysis of a variable-cell optimization is that of the
optimized atoms in the optimized cell with the cell held (the zone centre, as in a fixed cell,
with the constraints of the atoms on their surface; the lattice variables and the forces on the
cell are not part of it); the log and `frequency_cell.txt` say so. Before, `-freq` was
silently ignored after `-vc` (test `cli_freqmain_masses`).

### Vibrations at a constrained stationary point (`-fix`, `-pc`, `-gfix`, `-uc`)

After an optimization with constraints (`optmain -freq`), and in `freqmain` with the same
options, the structure is stationary within the constraints and not on the bare surface:
the gradient there is the constraint force, and the frequencies and the count of imaginary
modes of the bare Hessian (which the checks of transition states and the follow-up commands
read) are not those of the constrained surface. The analysis is then the one on the
constraint surface: the Hessian of the Lagrangian `H - sum_k lambda_k d2c_k`, with the
multipliers `lambda = (J J^T)^+ J g` of the gradient, mass-weighted, with the `m` constraint
gradients `M^-1/2 grad c_k` projected out with the rigid-body motions. That leaves `3N - 6 - m`
vibrations (`3N - 5 - m` for a linear structure), or `3N - m` when the constraints pin the
frame (frozen atoms, Cartesian components), less the rigid motions that keep the constraints
on a surface blind to rigid motions: the rotations about one frozen atom (3), the rotation
about the line through two (1), none with three atoms off one line
(`math::projection::rigid_motions_keeping`). Those are zero modes of the constrained surface
and are projected out with the constraint gradients, so they never appear among the
vibrations or in the count of imaginary modes (with a residual gradient they came out as
curvatures of about -1e-9 and were counted as imaginary; test `vib_zero_modes`). The
log says that the analysis is on the constraint surface, and `thermochemistry.txt` notes that
its thermal functions lack the constrained modes. The constraints are resolved at the input
of the run (eigenvector constraints at the first Hessian). `freqmain` with constraints
analyses one structure: `-path`, `-anharmonic`, `-local_modes`, `-partial_atoms`, the
phonons, the orbitals, `-mf`, and `-oniom` are refused with them.

Example: water with GFN0-xTB and one O-H bond held at 1.8 Angstrom (far beyond the
inflection of its curve). The bare Hessian has one imaginary frequency along the bond; on
the constraint surface the two vibrations left are real (`imaginary_modes` 0 instead of 1;
test `optmain_constraint_bias_defects`).

### Anharmonic vibrational analysis (`freqmain -anharmonic`)

`freqmain minimum.xyz -os ... -anharmonic` goes beyond the harmonic approximation for a
minimum. The structure is displaced along every normal mode in both directions, and the
Hessians there give the cubic force constants and the quartic ones with two equal indices
(two Hessians per vibration; they are evaluated through the journal, so an interrupted run
continues with `--restart`). On this force field:

- **VPT2** (second-order vibrational perturbation theory): anharmonic constants `x_ij`,
  fundamentals, first overtones, and the zero-point energy, with the Coriolis terms of an
  asymmetric top. Fermi resonances (`2 w_i ~ w_k`, `w_i + w_j ~ w_k`) are found by the test
  of Martin et al., their terms are left out, and the two coupled levels are listed from
  their 2x2 problem.
- **VSCF** (vibrational self-consistent field): every mode in the mean field of the others,
  each solved in harmonic oscillator functions; fundamentals from the separately converged
  one-quantum states.

| Option | Meaning | Default |
|---|---|---|
| `-qff_step` | displacement along every mode in its dimensionless normal coordinate | 0.05 |
| `-vscf_basis` | harmonic oscillator functions per mode in VSCF | 16 |
| `-fermi_gap` | largest frequency gap of a Fermi resonance (cm^-1) | 200 |
| `-fermi_threshold` | a term is a resonance when its correction beyond perturbation theory exceeds this (cm^-1) | 1 |
| `-freq_lowest` | modes at or below this wavenumber are not vibrations | 10 |

`anharmonic.txt` has the table of harmonic, VPT2, and VSCF fundamentals, the zero-point
energies, the overtones, the matrix `x_ij`, and the resonances; `force_field.txt` lists the
force constants in dimensionless normal coordinates (cm^-1).

What to keep in mind:

- The structure must be a minimum (a structure with imaginary frequencies is refused), and
  it should be converged tightly: the expansion is about the point where the gradient
  vanishes.
- VPT2 describes weakly anharmonic vibrations. A large-amplitude motion (a torsion, an
  inversion, a very soft bend) is outside its range; compare with VSCF, which treats the
  anharmonicity of each mode exactly but has no correlation between the modes. Where the
  two differ by much, neither is to be trusted without a closer look.
- The formulas are those of an asymmetric top. Degenerate modes (symmetric tops, linear
  molecules) are treated as the two nearly degenerate modes the Hessian gives; for a linear
  molecule the Coriolis terms are left out.
- A quartic expansion need not be bounded from below. VSCF then reports that the force
  field has states below the vibrations and follows the modals by their overlap; the
  wavenumbers of the affected modes then depend on the basis by a few cm^-1.

From Rust: `vib::Qff` (`ask` the next displaced structure, `tell` its Hessian),
`vib::ForceField`, `vib::vpt2`, `vib::vscf`.

### Vibrations of chosen atoms (`freqmain -partial_atoms`)

`freqmain structure.xyz -os ... -partial_atoms 1,4,7-9` analyses, besides the whole
structure, the vibrations of the listed atoms alone (numbers start at 1), as if they were a
molecule with the Hessian that the full Hessian leaves for them:

| `-partial_hessian` | Hessian of the chosen atoms `A` (`B`: the others) | Projection |
|---|---|---|
| `block` (default) | the block `H_AA`: the atoms `B` stay in place | translations and rotations of `A` |
| `effective` | `H_AA - H_AB H_BB^+ H_BA`: the atoms `B` follow so that no force acts on them | translations and rotations of `A` |
| `frozen` | the block `H_AA` | none: the atoms `B` are a fixed frame |

The results are in `partial_hessian/` of the run directory: `summary.txt`, `hessian.txt`
(the Hessian of the chosen atoms in the order given, Hartree/Bohr^2, one row per line),
`normal_modes.txt`, and `vibration_animation/`.

With `effective` and two atoms one vibration is left. Its force constant, printed in
`summary.txt`, is the relaxed force constant of the distance between the two atoms (the
reciprocal of the compliance constant), which is the force constant of the local stretching
mode of Konkoli and Cremer; its wavenumber is the local mode frequency. This is what
local-mode measures of bond strength such as the local Tolman electronic parameter use.

In the effective Hessian the atoms `B` are relaxed only in the directions in which they
have a restoring force: with one atom chosen, or atoms on a line, the rest can turn about
them as a rigid body, and these rotations are projected out of `H_BB` before it is
inverted. `-partial_free` (default `1e-10`) is the relative curvature below which any other
direction of `B` is left out; it guards against exact zeros.

From Rust: `vib::partial_hessian`, `vib::relaxed_stretch`, then `vib::normal_modes` on the
structure and Hessian returned.

### Local vibrational modes (`freqmain -local_modes`)

`freqmain minimum.xyz -os ... -local_modes 1-2 2-1-3 5-1-2-3` gives, for each internal
coordinate listed, its local vibrational mode in the sense of Konkoli and Cremer: the
motion that follows when that coordinate alone is changed and the rest of the molecule
relaxes. Coordinates are written with atom numbers starting at 1: `1-2` a distance, `1-2-3`
an angle (the middle atom is the vertex), `1-2-3-4` a dihedral; `bonds` stands for all
bonds of the structure and `auto` for all bonds, angles, and dihedrals.

`local_modes.txt` lists for every coordinate

- its value;
- the local force constant `k^a = 1 / (b f^+ b^T)` (mdyne/Angstrom for a distance,
  mdyne Angstrom/rad^2 for an angle), with the row `b` of the Wilson B matrix and the
  inverse `f^+` of the Hessian on the vibrations: the reciprocal of the compliance
  constant, that is the force constant with everything else relaxed;
- the local mode mass `1 / (b M^-1 b^T)` (for a distance the reduced mass of the two
  atoms);
- the local mode wavenumber, from `k^a` and that mass.

`local_mode_composition.csv` has the share of every normal mode (columns, headed by their
wavenumbers) in each local mode (rows); the shares of a row add up to 1. For a complete,
non-redundant set of `3N - 6` coordinates, dividing each column by its sum gives the
decomposition of the normal modes into local modes.

A local force constant belongs to one bond or angle and does not depend on the masses or
on how the normal modes happen to mix, so it can be compared between molecules as a
measure of bond strength; the local force constant of a metal-ligand or a C-O stretch is
the local-mode form of the Tolman electronic parameter. For a distance it is the same
number as the relaxed force constant from `-partial_atoms a,b -partial_hessian effective`.

The structure should be a minimum, converged tightly. With an imaginary frequency the
result carries a note (the compliance then has a negative part), and a vibration without
a restoring force is refused.

From Rust: `vib::local_modes` with `coord::internal::Primitive` coordinates.

### Phonons of a periodic structure (`freqmain -phonon_supercell`)

For a structure with a periodic cell, `freqmain crystal.xyz -os ...` alone gives the
vibrations at the zone centre (the cell moving in phase with all its images). With
`-phonon_supercell N1 N2 N3` the Hessian of the supercell of `N1 x N2 x N3` cells is
computed once, and its force constants give the phonons at any wave vector:

| Option | Meaning | Default |
|---|---|---|
| `-phonon_supercell N1 N2 N3` | cells of the supercell along the lattice vectors (1 along a direction that is not periodic) | none |
| `-phonon_path Q Q ...` | wave vectors `q1,q2,q3` in units of the reciprocal lattice vectors; the dispersion along the lines between them | none |
| `-phonon_points` | steps along every line | 40 |
| `-phonon_mesh M1 M2 M3` | mesh of wave vectors (zone centre included) for the density of states and the thermodynamic functions | none |
| `-phonon_dos_width` | width of the Gaussian of every mode in the density of states (cm^-1) | 5 |
| `-temp`, `-temp_scan`, `-freq_lowest` | temperature(s), and the wavenumber at or below which a mode is left out | as elsewhere |

```
freqmain cu.xyz -os emt -phonon_supercell 3 3 3 \
    -phonon_path 0,0,0 0.5,0,0 0.5,0.5,0 0,0,0 0.5,0.5,0.5 -phonon_mesh 12 12 12
```

| File | Content |
|---|---|
| `phonon_dispersion.csv` | distance along the path, wave vector, and the wavenumbers of all `3N` branches (negative: imaginary, the lattice is unstable against that wave) |
| `phonon_dos.csv` | density of states per cell (integrates to `3N`) |
| `phonon_thermochemistry.txt` | per cell at `-temp`: zero-point energy, vibrational internal energy, entropy, Helmholtz free energy, heat capacity at constant volume |
| `phonon_thermochemistry_scan.csv` | the same over `-temp_scan` |

The wave vectors refer to the cell given, not to a primitive cell found by symmetry: for
the conventional cubic cell of an fcc metal, `0.5,0,0` is the X point, folded onto the
zone boundary of that cell.

The force constants of a supercell reach half its width. The phonons are exact at the wave
vectors commensurate with the supercell and interpolated between them, so the supercell
must be large enough for the forces to have decayed; repeat with a larger one to see
whether the dispersion still changes. The width of the density of states and the number of
points of the path only set how the results are drawn.

Not included: the splitting of longitudinal and transverse optical modes of polar
crystals at the zone centre (it needs Born effective charges), and bias potentials (they
are defined for the cell given, not for its supercell).

From Rust: `vib::phonon::{supercell, ForceConstants, path, mesh, density_of_states,
thermodynamics}`.

### Vibrations within a crossing seam (`-mf opt_mesx`, `-mf opt_meci`)

A crossing point found by gradient projection (`optmain -mf opt_mesx [charge multiplicity]`
for states that do not interact, `-mf opt_meci` for a conical intersection) is a stationary
point of the mean energy on the seam where the two states have the same energy. `optmain
... -mf opt_mesx 0 3 -freq` analyses it at the end of the search, and `freqmain point.xyz
-os ... -mf opt_mesx 0 3` for a given structure: the Hessians of both states are evaluated,
and the vibrations within the seam are written to `normal_modes.txt` and
`vibration_animation/`, with a summary in `seam.txt`:

- the energies of the two states and their gap;
- the length of the difference gradient and the multiplier `l` of the seam condition
  (`grad E_mean = l (g_1 - g_2)`);
- the RMS of the mean gradient within the seam (zero at a stationary point of the seam);
- whether all curvatures within the seam are positive (a minimum of the seam) or not (a
  saddle point of the seam, from which a lower crossing point can be reached).

The Hessian within the seam is `W = H_mean - l (H_1 - H_2)`, projected onto the space
orthogonal to the branching plane and the rigid-body motions. The second term is the
bending of the seam; without it (the mean Hessian alone) the curvatures along a curved
seam are wrong. Thermodynamic functions are not written.

For states that do not interact (one direction in the branching plane) the result is
exact. For a conical intersection the adiabatic Hessians contain a term of the size
`1 / gap` whose range is the branching plane, so the plane must be projected out exactly.
Its second direction is taken from the gradients of the two states at displaced structures
(as in the check of the branching plane of the search, four more evaluations of both
states); `seam.txt` gives its overlap with the direction of the search and with the
eigenvector of `H_1 - H_2` of the largest eigenvalue, which is the same direction for
analytic Hessians and off by several degrees for Hessians made of finite differences that
reach across the cusp. The curvature of the coupling within the seam is not available and
is left out: take the frequencies of a conical intersection as an estimate.

`-anharmonic`, `-local_modes`, and `-partial_atoms` are not available at a crossing. From
Rust: `GeometryOptimizer::crossing` (the two states and the branching plane),
`vib::seam_analysis`, and `opt::crossing_check` (settings of the check; the optimizer asks
for the displaced evaluations itself, so a loop over `ask` and `tell` needs no change).

### Analysis of a reaction path (`freqmain -path`)

`freqmain path.xyz -os ... -path` evaluates energy, gradient, and Hessian at every frame and
describes the path in mass-weighted coordinates, as the reaction-path Hamiltonian of Miller,
Handy, and Adams does: the arc length `s`, the unit tangent `t(s)`, the curvature vector
`k(s) = dt/ds`, and the vibrations orthogonal to the path (the tangent is projected out with
the rigid-body motions).

| Option | Meaning | Default |
|---|---|---|
| `-path_tangent` | `gradient`: the tangent is the gradient (a steepest-descent path; the line through the neighbouring points is used where the gradient vanishes). `points`: that line everywhere, for a scan or a band of images | `gradient` |
| `-phase_prominence` | a minimum of the curvature divides two reaction phases when the curvature rises on both sides by more than this fraction of its largest value | 0 (every minimum) |
| `-temp`, `-press`, `-freq_corr`, `-freq_cutoff`, `-freq_lowest` | conditions and treatment of low frequencies for the free energies | as for `-freq` |

| File | Content |
|---|---|
| `path_profile.csv` | per point: arc length (`sqrt(amu) Bohr`), energy, RMS gradient, overlap of the gradient with the line through the neighbouring points, zero-point energy of the transverse modes, vibrationally adiabatic ground-state potential `V_a^G` (energy plus that zero-point energy), curvature of the path, effective mass of small-curvature tunneling, lowest transverse wavenumber with the overlap of its mode with that of the point before, all transverse wavenumbers |
| `bifurcations.txt` | where the lowest transverse curvature turns from positive to negative: the valley becomes a ridge (valley-ridge inflection) and the path bifurcates along that mode |
| `curvature_couplings.csv` | `B_kF = L_k . dt/ds` for the transverse modes `L_k` in ascending order of their wavenumbers: which vibrations the motion along the path drives where it bends (`sum_k B_kF^2` is the square of the curvature) |
| `mode_couplings.csv` | `B_kl = L_k . dL_l/ds` between the modes followed from the first point (matched from point to point by their overlaps, so a mode keeps its number when the order of the wavenumbers changes): how the modes turn into each other along the path |
| `urva_direction.csv`, `urva_curvature.csv` | components of the tangent and of the curvature vector along internal coordinates (unified reaction valley approach) |
| `reaction_phases.txt` | the stretches of the path between minima of the curvature, each with its curvature peak and the internal coordinates that contribute most there |
| `free_energy_profile.csv` | energy, `V_a^G`, and free energy of the generalized transition state at every point |
| `variational_tst.txt` | the variational transition state (maximum of the free energy), the ratio of the rate constants of canonical variational and conventional transition state theory, and the transmission coefficients of zero-curvature (ZCT) and small-curvature (SCT) tunneling |

**Which tangent.** On an intrinsic reaction coordinate the gradient is the tangent, and it
is the more accurate one. The column `gradient_alignment` and a line in the log tell how far
this holds (1 on a steepest-descent path). For a relaxed scan or a band of images it does
not; use `-path_tangent points`. The transverse frequencies are then those of the Hessian in
the space orthogonal to the path, where the gradient does not vanish, and are an
approximation. The curvature comes from finite differences of the tangents (the parabola
through three neighbouring points), so it needs points that resolve the bends of the path.

**Internal coordinates (URVA).** For an internal coordinate `q_n` with the row `b_n` of the
Wilson B matrix, `u_n = M^(-1/2) b_n^T / (b_n M^-1 b_n^T)^(1/2)` is the mass-weighted
direction along which `q_n` changes fastest. `t . u_n` tells how much the path moves along
the coordinate, `k . u_n` how much the coordinate takes part where the path bends (Zou,
Sexton, Kraka, Freindorf, Cremer, J. Chem. Theory Comput. 2016, 12, 650). The coordinates
are the bonds, angles, and dihedrals of all frames taken together, so a bond that exists
only at one end of the path is followed along all of it. The `u_n` are neither orthogonal
nor complete: the components are read one coordinate at a time and do not add up. The
curvature is small where the electronic structure changes steadily and peaks where bonds
start to break or form; its minima separate the reaction phases.

**Variational transition state.** The free energy of the dividing surface orthogonal to the
path at a point is the energy plus the thermal contributions of the transverse modes (and
of translation and rotation for a molecule), with the same treatment of low frequencies as
`-freq`. Its maximum is the transition state of canonical variational theory; the rate
constant is that of conventional theory (dividing surface at the point of the highest
energy) times `exp(-(G_max - G_saddle) / k T)`. The maximum is located by the parabola
through the highest point and its neighbours.

**Tunneling.** The transmission coefficient is the thermal average of the semiclassical
probability `1 / (1 + exp(2 theta(E)))` of passing the barrier `V_a^G(s)`, with the action
`theta(E) = int sqrt(2 mu_eff (V_a^G - E)) ds / hbar` between the turning points, divided
by the classical one. ZCT uses the mass of the mass-weighted coordinates. SCT lets the
tunneling path cut the corners of a curved path through a smaller effective mass,
`mu_eff / mu = min(exp(-2 a - a^2 + (dt/ds)^2), 1)` with `a = |k| t` and `t` the turning
point of the zero-point vibration along the curvature vector (the centrifugal-dominant
small-curvature method of Lu et al. 1992 and Liu et al. 1993; the formula is implemented as
stated here and has not been compared with a reference program). Energies below the higher
end of the path are not counted, so the path should reach down both sides of the barrier.

Within a set of degenerate modes the modes are defined only up to a rotation, and `B_kl`
among them has no meaning.

From Rust: `vib::transverse_modes` (modes with further directions projected out),
`vib::path_tangents`, `vib::gradient_alignment`, `vib::transverse_profile` (all of the above
per point), `vib::path_primitives`, `vib::internal_components`, `vib::reaction_phases`,
`vib::free_energy_profile`, `vib::effective_mass_ratio`, `vib::transmission`,
`vib::tunneling`.

From Python: `mopt_rs.path_profile` gives the transverse vibrations and the bifurcations of
a path from its points and their Hessians, or from the points and a calculator, which then
gives the gradients and energies too (`mopt_rs._core.transverse_profile` is the function
underneath, with flat lists). The tangent is chosen as `-path_tangent` does (`tangent=
"gradient"` or `"points"`, or `"given"` with `tangents`); `project=False` keeps translations
and rotations (a surface fixed in space). The result `PathProfile` has, as NumPy arrays,
`arc_length`, `energies`, `tangent`, `tangent_source`, `tangents_from_points`,
`gradient_alignment`, `curvature`, `curvature_vector`, `wavenumbers` and `curvatures` (one
array per point), `curvature_couplings`, `zero_point_energy`, `lowest_wavenumber`,
`lowest_curvature`, `lowest_mode`, `overlap`, `tracks`, `mode_couplings`, and `bifurcations`
(`PathBifurcation`: `after`, the index of the last point with a lowest curvature that is not
negative; `arc_length` of the zero by linear interpolation; `curvature_before`,
`curvature_after`; the `mode` that turns unstable). The numbers are those of
`path_profile.csv` and `bifurcations.txt` (tested along the inversion of ammonia with GFN-FF):

```python
import numpy as np
import mopt_rs
from mopt_rs.optimize import NativeCalculator

# One atom on E = -x^3/3 + (1 - x) y^2/2 + z^2/2 along y = z = 0: a valley for x < 1,
# a ridge beyond (Hartree, Bohr).
xs = [0.2 * k for k in range(1, 10)]
profile = mopt_rs.path_profile(
    ["H"],
    [[x, 0.0, 0.0] for x in xs],
    hessians=[np.diag([-2 * x, 1 - x, 1.0]) for x in xs],
    gradients=[[-x * x, 0.0, 0.0] for x in xs],
    project=False,
)
for b in profile.bifurcations:       # one, at x = 1: s = 0.8 sqrt(m_H) Bohr
    print(b.after, b.arc_length, b.mode.ravel())

# Along an IRC with a calculator (positions in Bohr, one array per point):
profile = mopt_rs.path_profile(symbols, irc_points, calculator=NativeCalculator("gfnff"))
print(profile.lowest_wavenumber, profile.bifurcations)
```

## Orbitals along a path (`freqmain -nro`, `-roet`)

`freqmain path.xyz -os gfn0xtb -nro -roet` analyzes the molecular orbitals along the
frames of `path.xyz` (the points of a reaction path, an intrinsic reaction coordinate for
example, in one file). The two analyses are independent; either option alone does its
own. Without `-path` nothing else is computed (no Hessians).

The calculator must hand out its orbitals: the built-in `gfn0xtb`, PySCF (restricted and
unrestricted Hartree-Fock and Kohn-Sham), and tblite do. Open shells are analyzed for the
alpha and the beta spin separately.

**`-nro`: natural reaction orbitals** (Ebisawa, Hasebe, Tsutsumi, Tsuneda, Taketsugu, Phys.
Chem. Chem. Phys. 2022, 24, 3532). At every frame the orbitals are differentiated along the
path (the line through the neighbouring frames, in mass-weighted coordinates, without its
translations and rotations), and the part of their change that mixes empty orbitals into
occupied ones is decomposed into pairs: an occupied and an empty natural reaction orbital
and a singular value, the rate at which the first takes up the second. The sum of the
squared singular values says how strongly the electrons respond to the motion; along a
reaction path it peaks where the electronic structure changes, near transition states
according to the paper.

- `nro.csv`: for every frame and spin the arc length (amu^(1/2) Bohr), the sum of the
  squared singular values (amu^(-1) Bohr^(-2)), the three largest singular values with
  their shares of that sum, and the smallest overlap of an occupied orbital with the one
  that follows it at the displaced structures.
- `nro_pairs.txt`: the leading pairs of every frame, written in the orbitals of the
  calculation (numbered from 1 by energy; parts above 5 %).
- `nro_orbitals_frame<k>_<spin>.csv`: the pairs in the atomic orbitals, at the frame where
  the sum is largest.
- `nro_sum_of_squares.png`, `.svg`: the sum along the path.
- `-orbital_step` (0.01 amu^(1/2) Bohr): the displacement to both sides for the
  differences; three calculations of orbitals per frame. A warning is written when an
  occupied orbital of a displaced structure overlaps with its original by less than 0.99.

**`-roet`: orbital energies along the path** (reactive orbital energy theory; Tsuneda,
Singh, Chattaraj, Phys. Chem. Chem. Phys. 2018, 20, 14211). The orbitals are followed from
frame to frame, so that an orbital keeps its column when the order of the energies
changes.

- `orbital_energies.csv`: the energy (Hartree) of every followed orbital at every frame.
- `1st_derivative_orbital_energy_plot.csv`, `2nd_derivative_orbital_energy_plot.csv`:
  their first and second derivatives along the path.
- `reactive_orbitals.txt`: every orbital with its energy at the two ends and its relative
  change `2 (last - first) / |last + first|`; the occupied reactive orbital (the occupied
  one that is stabilized most) and the unoccupied one (the empty one that changes most);
  the smallest overlap met while following.
- `orbital_energies.png`, `.svg`: the reactive orbitals and the two frontier orbitals.

How orbitals are followed: the orbitals of a neighbouring structure come in the order of
their energies, with arbitrary signs, and degenerate ones in an arbitrary rotation. Each
orbital is given its partner by the overlaps `C0^T S C1` (a linear assignment), degenerate
ones are rotated onto their partners, and signs are set so that overlaps are positive. The
frames must be close enough for this: overlaps well below 1 are reported.

From Python: `mopt_rs._core.follow_orbitals`, `natural_reaction_orbitals`,
`mass_weighted_displacement`, `reactive_orbitals`, and `EngineCalculator.orbitals(structure)`
for the orbitals of an engine. From Rust: `analysis::orbitals::{tracking, nro, roet}`,
`workflow::orbital_analysis`, and `Calculator::orbitals`.
