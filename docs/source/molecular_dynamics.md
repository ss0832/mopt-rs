# Molecular dynamics (`mdmain`)

```text
mdmain mol.xyz -os gfnff -time 20000 -temp 300                  # NVT, Nose-Hoover chain
mdmain mol.xyz -os gfnff -mt nve -ts 0.25fs -time 4000          # NVE
mdmain cu.xyz -os emt -mt npt -press 101.3 -ts 2fs              # NPT (Lattice= in the file)
mdmain mol.xyz -os gfnff -mt langevin -cc 1.09,1,2 -fix 3       # Langevin, a bond held, an atom fixed
mdmain ts.xyz -os gfnff -mt nve -md_opt freeze=negative          # no motion along the transition vector
mdmain start.txt -os mb2d -mt nve -ts 0.01                       # plain variables, unit masses
mdmain pair.xyz -os gfnff -umbrella cv="dist(1,2)" from=2.5*ang to=4.5*ang windows=11 k=0.02
```

The equations of motion are integrated in atomic units: positions in Bohr, masses in
electron masses (the atomic weights of the elements; one for plain variables), the time
step `-ts` in atomic units of time (`-ts 0.5fs` in femtoseconds), `-time` steps.
`-temp` is the temperature of the starting velocities (Maxwell-Boltzmann, scaled to it
exactly) and of the thermostat, `-ct STEP K ...` changes the target from those steps on.
Every step evaluates the surface once at the new positions: any calculator with its bias
potentials (time-dependent ones such as the nanoreactor's `-nrp` see the simulated time),
two electronic states combined by `-mf`, an excited state `-es`, or an objective of plain
variables.

`-mt` chooses the ensemble; `-md_opt KEY=VALUE` changes its parts one by one
(`thermostat=`, `integrator=`, `barostat=`, `tau=` in fs, `tau_p=`, `seed=`, ...; see the
reference of the option):

| `-mt` | Integrator | Thermostat | Barostat |
|---|---|---|---|
| `nve`, `velocityverlet` | velocity Verlet | none | |
| `nosehooverchain` (the default) | velocity Verlet | Nose-Hoover chain of three (Martyna, Tuckerman, Tobias, Klein) | |
| `nosehoover`, `nvt` | velocity Verlet | one Nose-Hoover thermostat | |
| `langevin`, `baoab` | BAOAB (Leimkuhler, Matthews) | Langevin | |
| `bussi`, `csvr` | velocity Verlet | velocity rescaling (Bussi, Donadio, Parrinello) | |
| `berendsen` | velocity Verlet | Berendsen | |
| `andersen` | velocity Verlet | Andersen | |
| `npt`, `npt_mtk` | velocity Verlet | Nose-Hoover chain | Martyna-Tobias-Klein, isotropic |
| `npt_crescale` | velocity Verlet | velocity rescaling | stochastic cell rescaling (Bernetti, Bussi) |
| `npt_berendsen` | velocity Verlet | Berendsen | Berendsen, isotropic |
| `npt_berendsen_aniso` | velocity Verlet | Berendsen | Berendsen, the cell deforms under the pressure tensor |
| `npt_aniso`, `npt_mtk_aniso` | velocity Verlet | Nose-Hoover chain | Martyna-Tobias-Klein, fully flexible cell |
| `npt_crescale_aniso` | velocity Verlet | velocity rescaling | stochastic cell rescaling, fully anisotropic (Del Tatto, Raiteri, Bernetti, Bussi) |

A step is a splitting around one evaluation of the forces (thermostat and barostat at both
ends for velocity Verlet, the thermostat in the middle of the drift for BAOAB). The
integrators, thermostats, and barostats are traits with registries
(`md::integrator::Integrator`, `md::thermostat::Thermostat`, `md::barostat::Barostat`), so
a new one is a new implementation, from Rust; the state machine
`md::MolecularDynamics` asks for the evaluations and is told their results
(`mopt_rs.MolecularDynamics` from Python, `mopt_rs.run_md(...)` with any function or
calculator).

The temperature counts `N_f` degrees of freedom: the coordinates that move, minus the
independent constraints, minus the frozen modes, minus the motions of the whole system the
dynamics conserves (`-md_opt rigid=auto`: the 3 translations and the rotations of an
isolated structure on a surface blind to them, the calculator and every bias potential (a
restraint of distances or angles is, a point or a wall in space is not:
`pes::term::frame_invariant`), the translations in a cell; none with Langevin or Andersen,
with held atoms, or with constraints that pin the structure; `rigid=full`, `translation`,
`none` to choose). The conserved motions are taken out of the starting velocities, drawn
or given (a Wigner sample), so that the count and the velocities agree; the log says so
for given ones. The log prints the conserved quantity of every step: the potential and
kinetic energies, the energy of the Nose-Hoover chains or the work the other thermostats did
on the velocities, and the energy of the barostat (`P V`, and the kinetic energy and chain of
Martyna-Tobias-Klein). On a surface that is a function of the structure alone it is
conserved up to the integration error for NVE, Nose-Hoover, and Martyna-Tobias-Klein, and it
is the effective energy of Bussi et al. for the stochastic thermostats. A bias potential that
changes with the time or the step (the nanoreactor `-nrp`, ramps of the constants, `t` and
`it` in `-ubias`) or with what it has seen (the deposits of `-metad`) does work on the
system that the conserved quantity does not contain: it drifts by that work, and the log
names those terms at the start (their `PesTerm::history`).

Constraints: the atoms of `-fix` have zero inverse mass (no velocity, no kinetic energy,
not counted); everything of `-pc`, `-gfix`, `-uc`, `-cc VALUE,A,B[,C[,D]]` (MultiOptPy's
SHAKE constraints: a distance in Angstrom or an angle in degrees), and Python constraint
functions is held by SHAKE (positions, along `M^-1 J^T` of the step before, Newton
iterations below 1e-12 and then on to the rounding floor of the constraints, about 1e-15
for a distance of a few Bohr: the constraints are hard and hold their values at every step
to machine precision; LINCS, `constraint_solver=lincs`, a truncated series, repeats its
correction until every bond is within 1e-12 Bohr or stops with an error above 1e-10, see
below) and RATTLE (velocities projected onto the tangent space in the metric
of the masses), so positions and velocities stay on the constraints and the forces of the
constraints are those of the Lagrange multipliers. The constraints are resolved at the
input structure: the frozen atoms and every value held "at the start" are those of the
input, also when the dynamics starts from a Wigner sample (`velocities=wigner`), which is
drawn from the modes of the space the constraints leave (frozen atoms not displaced, no
displacement or velocity along the gradients of the other constraints, rigid motions left
out only where the surface is blind to them and nothing pins the structure), with the
Hessian of the calculator and the static bias potentials.

Mode freezing (`-md_opt freeze=negative`, `below:VALUE`, `modes:0,1`, or `file:PATH` with
one Cartesian direction per line; `freeze_update=N`): the dynamics does not move along
chosen modes of the mass-weighted Hessian (the negative ones by default, rigid-body and
held motions left out). Their coordinates `u_k . M^(1/2) x` are held as linear
constraints, which takes their components out of the forces, the velocities, and the
displacements (`P = 1 - sum u_k u_k^T`) and out of the degrees of freedom; with
`freeze_update=N` the Hessian is computed again every `N` steps and every frozen mode is
followed to the new mode it overlaps most. On the Muller-Brown surface, NVE from the saddle
point with its negative mode frozen stays within 0.15 of the saddle with the conserved
quantity constant to 1e-6 and the constraint to 1e-10; without freezing it falls into a
minimum (`md::tests`).

Periodic cells come from `Lattice=` in the comment line of the structure or from `-pbc`
(`Lx,Ly,Lz` in Angstrom, the forms of `optmain`); the calculator sees the cell, the
dynamics is unwrapped, and the trajectory has the molecules wrapped into the cell as
wholes. The barostats need a cell periodic in three directions, the stress of the
calculator (the built-in potentials have it; ASE calculators that implement `get_stress`),
and no constraints. `-press` is in kPa (MultiOptPy's unit and default, 101.3).

`-vc` is the spelling of the other commands for a variable cell: with a constant-pressure
ensemble of `-mt` (or `-md_opt barostat=`) it is that ensemble; without `-mt` it is `-mt
npt_aniso` (the Martyna-Tobias-Klein barostat with a fully flexible cell and the Nose-Hoover
chain of the default ensemble), since under `-vc` the whole cell changes in the other
commands too. The pressure is `-vc_pressure` in GPa, 0 without it as in `optmain -vc`, or
`-press` in kPa (one of the two). `-mt npt*` without `-vc` works as before. Refused, with the
reason: `-vc` with an ensemble without a barostat, `-vc_weight` (the inertia of the cell is
`-md_opt tau_p=`), `-vc_fix` and `-uc` of functions of the cell (the barostat holds no
constraint of the cell), `-vc_afir`, plain variables. The trajectory has the lattice of every
frame, `md_properties.csv` the volume and the pressure of every step
(`tests/cli_mdmain_variable_cell.rs`).

Isotropic barostats (`berendsen`, `crescale`, `mtk`) scale the cell as a whole by the
scalar pressure and keep its shape; anisotropic ones (`berendsen_aniso`, `crescale_aniso`,
`mtk_aniso`, chosen by `-md_opt barostat=` or the `-mt` names above) let every length and
angle of the cell follow its component of the pressure tensor `P = sum m v v^T / V - sigma`
at the isotropic target pressure, for crystals whose cell changes its shape (phase
transitions, anisotropic thermal expansion). Of these, the anisotropic Berendsen barostat
does not sample the isothermal-isobaric ensemble; the other two do:

- `mtk_aniso` (`md::barostat::MtkAniso`): the Martyna-Tobias-Klein equations for a fully
  flexible cell, `dh/dt = v_g h`, `dv/dt = F/m - (v_g + Tr(v_g)/N_f) v`,
  `dp_g/dt = V (P - P0 I) + (2K/N_f) I` with a Nose-Hoover chain of its own on `p_g`; the
  pressure tensor is symmetrized, so that the cell does not rotate. The kicks and drifts
  are those of the isotropic barostat along the eigenvectors of the symmetric `v_g`. The mass
  of every component of `p_g` is a third of that of the isotropic barostat (a choice made
  here, so that the volume moves with the same period). The conserved quantity adds
  `W_g Tr(v_g^2) / 2`, `P0 V`, and the chain.
- `crescale_aniso` (`md::barostat::CellRescaleAniso`): the first-order stochastic equation
  of Del Tatto, Raiteri, Bernetti, and Bussi for the cell matrix,
  `dh = -(beta / 3 tau) [(P0 I - P) - (k_B T / V) I] h dt + sqrt(2 beta k_B T / (3 V tau)) dW h`,
  integrated as in their time-reversible scheme (a half step of the volume, the change of
  shape `exp(A_2 dt + b dW_2)` with the traceless parts, another half step of the volume);
  the rotation the noise brings is taken out by turning the cell back so that `a` lies along
  `x` and `b` in the `xy` plane, with the atoms and velocities turned with it. Positions go
  with the cell and the velocities are scaled by `mu^(-T)`. Like the isotropic `crescale`, it
  relaxes without oscillations and samples the volume fluctuations.

The log has, every step (`-md_opt print_every=`), the energies, MultiOptPy's
`hamiltonian :` and `Instantaneous_temperature:` lines, and the conserved quantity; and
every ten steps (`-md_opt structure_every=`, written to `settings.json`) the state of the
step: the Cartesian coordinates in Angstrom in MultiOptPy's layout (or the plain
variables), the velocities (atomic units, Bohr per atomic unit of time), and the energies
in Hartree: the kinetic energy, the energy of the calculator (`ENERGY`), with the bias
potentials (`BIAS  ENERGY`), the Hamiltonian (kinetic energy and `BIAS  ENERGY`), the terms
of the thermostat and the barostat, and the conserved quantity (their sum), whose drift
shows the error of the integration.

Files (MultiOptPy's names first): `energy_profile.csv` (energy without bias potentials),
`gradient_profile.csv`, `tot_energy_profile.csv`, `energy_profile_kcalmol.csv`,
`<stem>_traj.xyz` (every `write_every` steps, 10), `maximum_value.txt`, `minimum_value.txt`,
`local_min_grad.txt`, `energy_plot_.png`, `plot_gradient_.png`, `plot_temperature_.png`,
`plot_tot_energy_.png`; and `md_properties.csv` (step, time, energies, conserved quantity,
temperature, pressure, volume), `plot_conserved_.png` (`plot_pressure_.png`,
`plot_volume_.png` with a cell), `<stem>_final.xyz` and `final_velocities.txt`, and with
`-gi`, `-cmds`, `-pca` the files of `optmain` for the frames. `-traj N` runs N trajectories,
each in its own run directory with the seed one above the one before. A restart replays the
journal with the seed written in `settings.json`: the same trajectory, bit for bit.

Umbrella sampling (`-umbrella cv=EXPR ...`): windows of the harmonic restraint
`k / 2 (xi - xi_k)^2` on a variable written in the language of `-ubias` (atomic units),
run one after the other from the end of the one before (`-time` steps each, the first
fifth not sampled unless `equilibration=` says otherwise); `umbrella_profile.csv` has the
free-energy profile by WHAM and by MBAR, `umbrella_window_<k>/` the files of every window.
From Rust, `md::cv::Cv` has distances, angles, dihedral angles, coordination numbers, the
path variables `s` and `z`, expressions, and any term as variables,
`workflow::umbrella::run_umbrella` runs the windows, and `analysis::free_energy` has WHAM
and MBAR (`mopt_rs._core.wham`, `mopt_rs._core.mbar` from Python).

Not implemented yet: string methods in the space of collective variables (the string
method with restrained dynamics, the finite-temperature string method, and the string
method with swarms of trajectories). Their design on these parts: a string of images
`z_i` in the space of several variables, every image a short dynamics with the restraint
`sum_a k_a / 2 (xi_a - z_ia)^2` (one `Umbrella` per variable on the same surface), from
which the mean force `k (z_i - <xi>)` and the metric `<grad xi M^-1 grad xi^T>` are taken
(swarms: the mean drift of many short unrestrained trajectories from the image instead);
the images then move along the metric times the mean force and are put back at equal arc
length on the string (the reparametrization of `path::band::redistribute`), iteration by
iteration as an ask/tell state machine in `path::` with the dynamics of the images driven
by `workflow::md`.

## Coordinate systems (`-coord`)

`mdmain -coord NAME` takes the coordinate systems of `optmain` (any name of the
`CoordinateRegistry`). By default the dynamics runs in Cartesian coordinates and
`<stem>_coordinates.csv` has the values of the coordinates at the frames of the
trajectory; `-md_opt integrate=coordinates` runs the dynamics in the coordinates
themselves. The kinetic energy in coordinates `q` is `T = 1/2 p^T G(q) p` with
`G = B M^-1 B^T` (`B = dq/dx`, `p` the momenta conjugate to `q`), so that the dynamics in
`q` is not that of `q'' = -dV/dq`; whether it can run in them depends on the system:

- Non-redundant coordinates, whose `G` is regular: those that cover all `3N` motions
  (Cartesian coordinates, spherical coordinates of the atoms) and those that cover exactly
  the internal ones (`dic`, `3N - 6` of them, `3N - 5` for a linear molecule). The
  dynamics runs in them with the generalized leapfrog (`md::generalized`: Stormer-Verlet
  for a Hamiltonian that is not separable, symplectic and time-reversible; one evaluation
  of the surface per step, `dT/dq` from the change of `B` along the velocity, the
  structure of every `q` by the back-transformation of `optmain` to 1e-12). In internal
  coordinates this is the dynamics at zero linear and angular momentum, that of the
  Cartesian dynamics with the rigid motions taken out (the default of `mdmain` for an
  isolated molecule); it needs a surface that does not depend on the position and
  orientation in space (no bias potentials).
- Redundant coordinates: `G` is singular and there are no conjugate momenta, so the
  dynamics cannot run in them (`integrate=coordinates` is refused with exit code 2; the
  coordinates still serve for the output). Whether a set is redundant is decided at the
  start from the eigenvalues of `G` (smallest over largest below 1e-10): the primitive
  sets `ric` and `tric-p` of a molecule with more primitives than motions (ethanol: 33
  primitive internal coordinates for 21 internal motions) are redundant, while the
  delocalized sets (`dic`, `tric`, `tric-cart`) and the primitives of water (two bonds and
  an angle) are not.
- Constraints, frozen modes, a barostat, a periodic cell, an integrator other than velocity
  Verlet, umbrella sampling: these act on Cartesian coordinates and are refused with
  `integrate=coordinates`.

Thermostats act on `u = G^(1/2) p` (unit masses), so that every thermostat samples
`p ~ N(0, k_B T G^-1)`; the degrees of freedom are the coordinates (`3N - 6` for internal
ones, `3N` less the conserved rigid motions for a full set). From Rust,
`md::generalized::GeneralizedDynamics` takes any `CoordinateSystem` (also one of the
caller's own); from Python, `MolecularDynamics(..., coordinates="dic")` and
`run_md(..., coordinates="dic")`.

**Measurements** (`cargo run --release -p mopt-rs --example md_coordinates -- nve|nvt`).
NVE for 2 ps from the GFN-FF minimum with the velocities of 300 K (rigid motions taken
out), the largest deviation of the conserved quantity and the time per step (GFN-FF
evaluations included) and the evaluations of `B` per step; `spherical_atoms` is the
spherical coordinates of every atom about a point outside the molecule (`3N` curvilinear
coordinates, defined in the example):

| molecule | coordinates | time step (fs) | largest deviation (meV/atom) | time per step (us) | B per step |
|---|---|---|---|---|---|
| water | cart | 0.25 / 0.5 / 1 | 0.0947 / 0.3887 / 1.7375 | 19 / 11 / 9 | 0 |
| water | dic | 0.25 / 0.5 / 1 | 0.0947 / 0.3888 / 1.7377 | 173 / 197 / 157 | 14-16 |
| water | tric | 0.25 / 0.5 / 1 | 0.0947 / 0.3888 / 1.7377 | 452 / 580 / 657 | 14-17 |
| water | ric (not redundant) | 0.25 / 0.5 / 1 | 0.0947 / 0.3888 / 1.7377 | 172 / 201 / 192 | 14-16 |
| water | spherical_atoms | 0.25 / 0.5 / 1 | 0.0947 / 0.3887 / 1.7375 | 172 / 208 / 209 | 13 |
| HCN | cart | 0.25 / 0.5 / 1 | 0.0381 / 0.1549 / 0.6602 | 11 / 12 / 12 | 0 |
| HCN | dic | 0.25 / 0.5 / 1 | 0.0381 / 0.1550 / 0.6595 | 264 / 279 / 257 | 16-18 |
| HCN | tric | 0.25 / 0.5 / 1 | 0.0380 / 0.1549 / 0.6590 | 683 / 681 / 890 | 16-18 |
| HCN | spherical_atoms | 0.25 / 0.5 / 1 | 0.0381 / 0.1549 / 0.6602 | 184 / 216 / 250 | 12-13 |
| ethanol | cart | 0.25 / 0.5 / 1 | 0.0422 / 0.1662 / 0.6797 | 68 / 69 / 55 | 0 |
| ethanol | dic | 0.25 / 0.5 | 0.0421 / 0.1663 | 3667 / 3997 | 17-19 |
| ethanol | dic | 1 | broke down at step 1118 (the implicit half kick did not converge) | | |
| ethanol | tric | 0.25 | 0.0421 | 5741 | 17 |
| ethanol | tric | 0.5 / 1 | broke down (0.5 fs: a fragment turned beyond the range of the rotation coordinates of TRIC at step 2593; 1 fs: the implicit kick) | | |
| ethanol | ric | | redundant (33 coordinates), refused | | |
| ethanol | spherical_atoms | 0.25 / 0.5 / 1 | 0.0422 / 0.1662 / 0.6797 | 1212 / 1327 / 1385 | 13 |

The energy is conserved in every non-redundant system exactly as well as by velocity
Verlet in Cartesian coordinates at the same time step (the deviation is set by the
fastest vibration, whatever the coordinates), and the dynamics in internal coordinates
follows the Cartesian one without rigid motion (water at 0.05 fs: the O-H distances of the
two agree to 1e-7 Angstrom after 100 fs, `tests/md_coordinates.rs`). The cost is 10 to 60
times that of the Cartesian step with this cheap force field (13 to 19 evaluations of `B`
and the back-transformations per step; with a calculator that takes seconds per gradient
it would not matter), and the coordinates can break down where Cartesian dynamics does
not (ethanol at 1 fs; TRIC once a fragment turns too far). Since there is no gain in
accuracy or time step, the default is Cartesian dynamics, and the dynamics in the
coordinates is there for what needs it (`integrate=coordinates`).

NVT of water at 300 K, 0.5 fs, 40 ps (the first 4 ps left out): means with their
block-average errors and the standard deviations of the O-H distance, the angle, and the
instantaneous temperature:

| coordinates | thermostat | rigid motion | r(O-H) (ang.) | angle H-O-H (deg.) | T (K) | time per step (us) |
|---|---|---|---|---|---|---|
| cart | bussi | none (L = 0) | 0.97311 +- 0.00008 (sd 0.0219) | 106.337 +- 0.011 (sd 4.48) | 280 +- 16 (sd 227) | 13 |
| dic | bussi | none (internal) | 0.97339 +- 0.00009 (sd 0.0234) | 106.354 +- 0.011 (sd 4.59) | 292 +- 17 (sd 240) | 240 |
| spherical_atoms | bussi | none (L = 0) | 0.97319 +- 0.00008 (sd 0.0220) | 106.321 +- 0.011 (sd 4.51) | 287 +- 16 (sd 232) | 165 |
| cart | langevin | free | 0.97429 +- 0.00009 (sd 0.0237) | 106.223 +- 0.021 (sd 4.48) | 295 +- 8 (sd 143) | 12 |
| dic | langevin | none (internal) | 0.97343 +- 0.00009 (sd 0.0246) | 106.313 +- 0.016 (sd 4.23) | 286 +- 15 (sd 234) | 248 |
| spherical_atoms | langevin | free | 0.97443 +- 0.00010 (sd 0.0244) | 106.209 +- 0.022 (sd 4.59) | 304 +- 8 (sd 143) | 177 |

With Langevin dynamics, which thermalizes every mode, Cartesian and spherical
coordinates (both with free rotation) agree within one standard error in the distance and
the angle; the dynamics in internal coordinates samples the ensemble without rotation,
whose bonds are 0.0009 Angstrom shorter (no centrifugal stretching; the same in internal
coordinates with Langevin, 0.97343, and with Bussi, 0.97339). Bussi's thermostat with only
three degrees of freedom exchanges energy between the vibrations slowly, so that its runs
differ by two standard errors in the distance (0.97311 against 0.97339) and by 7 % in the
deviation; the temperature distributions have the canonical widths (`sqrt(2/3) T` = 245 K
for 3, `sqrt(2/9) T` = 141 K for 9 degrees of freedom).

## Trajectory analysis (`-analysis`, `-analyze_only`)

`-analysis` analyzes the trajectory when the run ends and writes the results into
`trajectory_analysis/` of the run directory, each file named in the log with a summary
(means and errors, peaks, diffusion coefficients):

| name | what | files |
|---|---|---|
| `statistics` | mean, deviation, and error of the mean from block averages of the energies, temperature, pressure, volume of every step (`md_properties.csv`) | `statistics.csv`, `blocking.csv` |
| `rdf` | radial distribution function `g(r)` of every pair of elements with the running coordination number (minimum image in a cell; without a cell the distribution of the distances) | `rdf.csv`, `rdf.png` |
| `msd` | mean squared displacement of every element and the diffusion coefficient `D = slope / 6` | `msd.csv`, `diffusion.csv`, `msd.png` |
| `vacf` | mass-weighted velocity autocorrelation and the vibrational density of states (cm^-1) | `vacf.csv`, `vdos.csv`, `vacf.png`, `vdos.png` |
| `rmsd` | RMSD to the first frame and RMSF of every atom | `rmsd.csv`, `rmsf.csv`, `rmsd.png`, `rmsf.png` |
| `geometry` | distances, angles, dihedral angles of `atoms=1,2;1,2,3;1,2,3,4` | `geometry.csv`, `geometry_<k>.png` |
| `pca` | projection on the principal components of the mass-weighted, superposed coordinates (`analysis::path_map`) | `pca.csv`, `pca_components.csv`, `pca.png` |

`-analysis` alone (or `all`) runs all of them; `KEY=VALUE` settings among the names:
`skip=<fs>` (the frames and steps before are left out, 0), `rdf_bin=` (0.02 Angstrom),
`rdf_max=` (half the smallest width of the cell, the reach of the minimum image),
`msd_fit=0.1,0.9` (the shares of the longest lag between which the straight line is
fitted), `drift=true` (the drift of the centre of mass is subtracted from the
displacements, as a Langevin or Andersen thermostat lets it wander), `vacf_share=0.5`,
`vdos_max=`, `vdos_step=1` (cm^-1), `components=3`, `figures=png|svg`, `interval=1` (fs
between frames whose comments carry no `time=`). The frames are those of `<stem>_traj.xyz`
(every `write_every` steps) with their velocities in `<stem>_velocities.xyz` (Angstrom per
fs); a correlation function needs evenly spaced frames, so the last frame of a run that
ends between two is left out of the analysis (and the log says so). `mdmain RUN_DIR
-analyze_only [-analysis ...]` analyzes a finished run (or any XYZ trajectory with
`time=` in its comments, and the velocities and `md_properties.csv` next to it when
there are) without a new run. For a run of plain variables the statistics are made.

How the results are made, and where the defaults come from:

- The error of a mean is that of the blocking transformation of Flyvbjerg and Petersen
  (J. Chem. Phys. 1989, 91, 461) at the block size chosen by the criterion of Lee et al.
  (Phys. Rev. E 2011, 83, 066706; pyblock); `blocking.csv` has every level. A series too
  short for its correlation time is marked `converged=false` (its error is then the
  largest of the levels with at least four blocks, a lower bound).
- `g(r)` is normalized with the volume of the cell of every frame (also at constant
  pressure) and `N_a (N_a - 1)` pairs for equal elements; the bin of 0.02 Angstrom is that
  of GROMACS (`gmx rdf -bin 0.002` nm). Distances beyond half the smallest width of the
  cell are refused (the minimum image does not reach them).
- The displacements are those of the positions with the jumps through the faces of the
  cell taken out frame by frame (the frames must be less than half the cell apart), and the
  mean squared displacement is averaged over all time origins by the fast Fourier
  transform (Kneller et al., Comput. Phys. Commun. 1995, 91, 191). The fit from 10 % to
  90 % of the longest lag is the default of GROMACS (`gmx msd -beginfit -endfit`).
- The velocity autocorrelation reaches half the run by default, so that every lag is
  averaged over at least half of the time origins; the density of states is its cosine
  transform with a Hann window, normalized to `3 N` states (the resolution, about the
  inverse of the longest lag, is in the log). The frames must be close enough for the
  fastest vibration: the Nyquist wavenumber `1 / (2 c dt_frame)` is 3336 cm^-1 for frames
  5 fs apart (0.5 fs and `write_every=10`), so that O-H and C-H stretches need
  `write_every=2` or less at 0.5 fs.
- Without a cell, every frame is superposed on the first (mass-weighted Kabsch) for the
  RMSD, the RMSF, and the principal components; in a cell only the drift of the centre of
  mass is removed.

From Rust, `analysis::trajectory` works on any frames in memory (`MdTrajectory::new`, or
`MdTrajectory::from_trajectory` of `io::xyz::read_xyz_trajectory_file`), and
`workflow::md_analysis::analyze_files` reads and writes the files of a run; from Python,
`mopt_rs.TrajectoryAnalysis` (frames in Angstrom and fs, or `TrajectoryAnalysis.from_file`
of a run directory), `mopt_rs.md.block_average`, and `mopt_rs.md.analyze_md_run` (the
files of `-analyze_only`).

The checks on real systems (`tests/md_trajectory_analysis.rs`):

| system | run | result | expected |
|---|---|---|---|
| Cu, 108 atoms, EAM (Zhou 2004), fcc a = 3.615 ang. | Bussi, 300 K, 2 fs, 1.2 ps (first 0.2 ps left out) | `g(r)` peaks at 2.55 and 3.63 ang., 12.00 neighbours within 3.1 ang.; `D` = -0.002 ang.^2/ps; T = 292 +- 5 K | a / sqrt 2 = 2.556, a = 3.615; 12; a crystal does not diffuse; 300 K |
| the same, cell 4 % wider | Bussi, 2500 K, 1 fs, 2.5 ps (first ps left out) | molten: `D` = 0.79 ang.^2/ps (7.9e-5 cm^2/s); first peak 2.43 ang. at g = 2.4 (crystal 8.3), second shell 0.73 (crystal 1.60) | a liquid diffuses (liquid metals near their melting points: a few 1e-5 cm^2/s), its shells beyond the first fade |
| Si, 64 atoms, Tersoff (1989), diamond a = 5.431 ang. | Bussi, 300 K, 1 fs, 0.4 ps | peaks at 2.35 and 3.85 ang., 4.00 neighbours within 2.9 ang.; bond 2.350 ang., angle 109.9 deg. | a sqrt 3 / 4 = 2.352, a / sqrt 2 = 3.840; 4; 109.47 deg. |
| water, GFN-FF | NVE from a structure off the minimum, velocities of 300 K, 0.25 fs, 2 ps, frames 0.5 fs apart | density of states peaks at 1618 and 3635 cm^-1 | `freqmain` at the minimum: 1633, 3634, 3637 cm^-1 |
| water, GFN0-xTB | the same | peaks at 1549 and 3420 cm^-1 | 1554, 3445, 3457 cm^-1 |

The peaks of the vibrations lie within 1 % of the harmonic frequencies, below them by the
classical anharmonic shift. A molecule that rotates (a Langevin or Andersen thermostat
lets it, since it does not conserve the angular momentum) has side bands at the
rotational frequencies around every vibration (with Langevin at 300 K: 3596 and 3690
cm^-1 around the stretches of 3635 cm^-1, and a band below 200 cm^-1), so a density of
states for comparison with the harmonic frequencies is best taken from NVE or a
momentum-conserving thermostat (`bussi`, `nosehooverchain`) with the rotation taken out
(`rigid=auto`, the default). The analysis of the files of a finished run with
`-analyze_only` gives the same files as at the end of the run.

## Initial conditions from the Wigner distribution (`-md_opt velocities=wigner`)

`-md_opt velocities=wigner` draws the structure and the velocities of the start from the
harmonic Wigner distribution of the normal modes of the input structure (a minimum, or a
transition state for trajectories from it; the Hessian is the calculator's, at the input
structure; `md::wigner`). Every vibration of angular frequency `omega` gets a Gaussian
normal coordinate and momentum with `<Q^2> = coth(omega / 2 k_B T) / (2 omega)` and
`<P^2> = omega coth(omega / 2 k_B T) / 2` (atomic units; Hillery, O'Connell, Scully, and
Wigner, Phys. Rep. 1984, 106, 121), so that its mean energy is that of the quantum
oscillator, `(omega / 2) coth(omega / 2 k_B T)`: the zero-point energy at
`wigner_temperature=0` (the vibrational ground state, the initial conditions of
photodynamics), `k_B T` in the classical limit. The temperature is that of `-temp` unless
`wigner_temperature=` is given. Imaginary modes get no displacement and a momentum
(`wigner_imaginary=boltzmann`, the default: from `N(0, k_B T)`; `forward`, `backward`: its
magnitude along or against the mode; `zero`); modes below a wavenumber
(`wigner_low=CUTOFF[:classical|zero|wigner]`, none by default) can be given the classical
Boltzmann distribution or nothing, since the quantum width of a soft mode is large. The log
lists every mode with its treatment, its drawn energy, and the mean of its distribution;
every trajectory of `-traj N` draws its own sample from its seed. Checks: the drawn
energies of a harmonic oscillator against `(omega / 2) coth(omega / 2 k_B T)` at 0, 3000,
and 30000 K within four standard errors of 40000 samples, the Cartesian structure and
velocities against the energy of the modes sample by sample (`md::wigner::tests`), and 40
trajectories of water (GFN-FF) from the ground state against its zero-point energy
(`tests/md_wigner.rs`).

## Collective variables (`-cv`, `-cv_bias`, `-umbrella cv=`)

One form of collective variables (`md::cv::Cv`, parsed by `md::cv_spec::parse_cv`) serves
every place `mdmain` takes them: `distance:1,2`, `distance:1,2,3/4,5` (centroids),
`angle:1,2,3`, `dihedral:1,2,3,4`, `coordination:1,2/3,4[:r0=ANG][:n=6][:m=12]` (the
rational switching function summed over the pairs of the groups, `r0` 2.0 Angstrom by
default), `path_s:FILE[:lambda=PER_ANG2]` and `path_z:FILE` (progress along and distance
from the path of the frames of an XYZ file, `lambda = 2.3 / <d^2>` of neighbouring frames by
default; Branduardi, Gervasio, and Parrinello, J. Chem. Phys. 2007, 126, 054103), and any
expression of the `-ubias` language. `-cv SPEC ...` writes them at every step to
`<stem>_cv.csv` (atomic units: Bohr, radians), and `-analysis statistics acf` gives their
means with block-average errors, autocorrelations, correlation times, and power spectra;
`-cv_bias "SPEC;center=C;k=K"` adds the harmonic restraint `k/2 (xi - C)^2` to the surface
(the restraint of a window of umbrella sampling, `md::cv::Umbrella`); `-umbrella cv=SPEC`
samples along any of them.

## LINCS (`-md_opt constraint_solver=lincs`)

LINCS (Hess, Bekker, Berendsen, and Fraaije, J. Comput. Chem. 1997, 18, 1463; `md::lincs`)
holds bond lengths without iterations: the positions are projected along the directions of
the bonds of the step before with `(I - A)^-1` expanded to `lincs_order=` terms (4), and
the lengthening by the rotation of the bonds is corrected `lincs_iterations=` times (1),
the defaults of the paper; the velocities are projected with the same expansion. Every hard
constraint is held to 1e-10 Bohr, so after those corrections LINCS repeats the correction
(the directions of the step kept, one series solve each) until the largest residual is below
1e-12 or no longer decreases, at most 200 more times; a bond still above 1e-10 Bohr is an
error naming it (a smaller time step, or SHAKE). It takes constraints of bond lengths only
(`-pc bond` of two atoms); with other constraints it is refused, and with frozen modes the
matrix SHAKE of the linear rows takes over.

The table below was measured with the corrections asked for only, before the repeated
corrections: its LINCS residuals are what those corrections leave (now continued to below
1e-12), and its LINCS times are those of the corrections asked for (each further correction
adds one series solve; not measured again here).

Measured on ethanol (GFN-FF, the six bonds to hydrogen held, velocities of 300 K, NVE for
2 ps; `examples/md_measurements.rs shake`; the time per step includes the force field):

| solver | time step (fs) | largest residual (Bohr) | largest energy deviation (meV/atom) | drift (meV/atom/ps) | time per step (us) |
|---|---|---|---|---|---|
| SHAKE/RATTLE | 0.5 | 1.0e-10 | 0.0460 | -0.0007 | 154 |
| SHAKE/RATTLE | 1 | 1.0e-10 | 0.1873 | -0.0021 | 151 |
| SHAKE/RATTLE | 2 | 8.9e-16 | 0.7891 | -0.0170 | 158 |
| LINCS, order 4, 1 correction | 0.5 | 5.2e-7 | 0.0460 | -0.0007 | 39 |
| LINCS, order 4, 1 correction | 1 | 4.3e-6 | 0.1870 | -0.0024 | 47 |
| LINCS, order 4, 1 correction | 2 | 3.1e-5 | 0.7887 | -0.0082 | 42 |
| LINCS, order 8, 2 corrections | 0.5 | 9.8e-10 | 0.0460 | -0.0007 | 42 |
| LINCS, order 8, 2 corrections | 1 | 1.6e-8 | 0.1873 | -0.0021 | 44 |
| LINCS, order 8, 2 corrections | 2 | 2.5e-7 | 0.7891 | -0.0170 | 48 |

The conservation of the energy is the same with both solvers at every time step (the
error is that of the integrator); LINCS takes a quarter to a third of the time of a step
here, since it needs no iterations and no linear solves, and with its defaults it holds
the bonds to 1e-7 to 1e-5 Bohr (the residual grows with the rotation of the bonds in a
step, so with the time step); order 8 with 2 corrections brings it to 1e-9 to 1e-7 Bohr
at no measurable cost. SHAKE/RATTLE stays the default: it holds every kind of
constraint, to the tolerance of its iterations.

## Surface hopping by the Zhu-Nakamura theory (`-md_opt hopping_states=`)

`-md_opt hopping_states=1,2` runs the trajectory on one adiabatic electronic state at a time
and lets it hop between the states of the list (roots counted as by `-es`: 0 the ground
state, 1 the first excited state), starting on the state of `-es` (`md::hopping`). It is off
without the key. Any calculator with excited states serves (`gfn0xtb` with its sTDA singlets,
programs behind ASE that give them, a model of one's own), with any thermostat and barostat,
in a periodic cell as well as for a molecule (a barostat needs the stress of the active
state from the calculator); `-mf`, `-umbrella`, plain variables, and `integrate=coordinates`
are refused with the reason.

**What happens at every step.** The surface gives the energies of all the states of the
list, the gradient of the active state, and the gradients of its neighbours (the nearest
states of the list below and above it) while their gap to it is below `hopping_gap=` (eV,
1). For `gfn0xtb` the energy of an excited state alone is one SCF and one sTDA; its
gradient is `6N` of them (central differences), so the gradients of the neighbours are
paid for only near a crossing. Following the review of Ishida, Nanbu, and Nakamura
(section 3.1), a hop to a neighbour is considered where the gap to it has a local minimum
along the trajectory: at the middle one of three consecutive steps on which the active
state did not change.

**The model of the crossing without coupling vectors.** The review takes the hopping
direction from the nonadiabatic coupling vector, or from the eigenvector of the Hessian of
the gap (its eq. 3.5). No built-in calculator gives coupling vectors, and the Hessian of the
gap needs the Hessians of both states at the hop: for `gfn0xtb` excited states those are
differences of difference gradients, `6N` gradients of `6N` excited-state calculations each
(`36 N^2` per state; 576 for formaldehyde) at every hop. The published approximation of Yu,
Xu, Lei, Zhu, and Wen (Phys. Chem. Chem. Phys. 2014, 16, 25883) needs only the gradients of
the two states at the outer steps: the diabatic forces at the middle step join the upper
adiabatic force at the first step to the lower at the last and the other way round (their
eqs. 7-8), the model of the crossing has `|F2 - F1|` and `F1 . F2` of the mass-weighted
forces (eqs. 4-5), the coupling is half the gap, and the hopping direction is that of
`F2 - F1` in mass-weighted coordinates (eqs. 9-10), the direction in which the probability is
largest. Two changes, both written in `md::hopping::direction`: the whole mass-weighted
vector is normalized (eq. 10 normalizes the vector of each nucleus), so that the kinetic
energy along the direction is that of the one coordinate of eqs. 4-5; and the connection of
eqs. 7-8 is taken in time over the equal steps, which is the same as the paper's connection
per Cartesian component for the straight motion of a step and is defined also for
components that do not move.

At the outer steps the adiabatic forces are those of the diabatic states only away from the
coupling region: in the straight two-state model their difference is
`sqrt(1 - (2V / Delta)^2)` times the diabatic one (`Delta` the gap), so that eqs. 7-8
shorten `|F2 - F1|` wherever the steps are short compared with the coupling region, and the
probability of a weakly coupled crossing (near 1) comes out too small. `hopping_forces=`
`two_state` (the default) divides the differences at the outer steps by that factor (at
least 0.1) and takes the coupling from the vertex of the parabola of the squared gap through
the three steps; both are exact for straight diabatic states passed at a constant velocity.
`hopping_forces=published` keeps eqs. 7-8 and half the gap at the middle step as printed.
Measured on the two-state model of `tests/md_surface_hopping.rs` (two hydrogen atoms,
straight diabatic states with forces 0.02 and 0.005 Hartree/Bohr and a coupling of 0.002
Hartree, passed 0.0055 Hartree above the crossing, where eq. 2.11 gives `p = 0.626`; the
mean of the estimated probability over 20 positions of the smallest gap within a step,
relative to `p`):

| time step (fs) | factor at the outer steps (about) | `published` | `two_state` |
|---|---|---|---|
| 0.1 | 0.06 | 0.045 | 0.977 |
| 0.25 | 0.16 | 0.203 | 1.000 |
| 0.5 | 0.30 | 0.430 | 1.000 |
| 1 | 0.53 | 0.674 | 0.998 |

The published forces fall short wherever the steps lie in the coupling region, which is
where a weak coupling makes the hop likely; the two-state form follows the straight model at
every step, and is the default.

**The probability and the decision.** For the energy of the motion along the direction (the
potential energy of the active state and the kinetic energy along the direction at the
minimum) the Zhu-Nakamura formulas of the kinetics are used (`kinetics.md`): a sloped
crossing (`F1 . F2 > 0`) is classically allowed from the upper adiabatic energy at the
minimum up and hops with the probability of one passage of eq. 2.11; a peaked one
(`F1 . F2 < 0`) is allowed from the bottom of the upper well up (`b^2 >= 1`) with eq. 2.26. A
uniform random number from a generator of its own (seeded from the seed of the run) decides:
the hop is made when the probability exceeds it (the "ant-eater" procedure of the review).
Classically forbidden candidates are reported and not made: below the well bottom of a
peaked crossing the upper state is closed and the log gives the overall transmission `P12`
of the lower state (eqs. 2.54-2.62); below the upper energy of a sloped crossing the
transition would end at the turning point of the upper state away from the point of the hop
(eqs. 2.41-2.53), which a hop at one point cannot do with the energy conserved.

**The hop.** It is made at the step after the minimum, the first at which the minimum is
known (Yu et al. hop at the minimum itself; the difference is one step). The velocity
component along the direction changes so that the total energy is conserved there, keeping
its sign (their eqs. 13-15), the perpendicular components are kept, and the forces of the
new state at that step continue the integration; the direction is first taken into the space
the velocities may move in (without the conserved motions of the whole system, projected
onto the constraints). When the kinetic energy along the direction cannot pay for an upward
hop, the hop is frustrated: the velocities are kept (`hopping_frustrated=keep`, the default)
or the component along the direction is reversed (`reverse`).

**Narrow crossings.** The step before a hop is integrated on the state the trajectory was on.
Where two states nearly touch, the force of an adiabatic state turns within a small part of
a step, and that step costs energy of the order of the difference of the forces times the
distance moved after the crossing; the hop itself moves energy only between the potential and
the kinetic energy. Measured on pyrazine (`gfn0xtb`, states 1 to 3 from state 2, 800 K, NVE,
0.25 fs, 100 fs, seed 7): three hops between states 1 and 2 at gaps of 12, 23, and 16 meV
with probabilities 0.9999, 0.995, and 0.9993 (near-diabatic passages), and two forbidden
candidates; the conserved quantity changed by 1.1e-3, 1.2e-4, and 2.1e-4 Hartree in the steps
of the three hops and by less than 2e-5 Hartree in every other step (1.2e-3 Hartree over the
run). A shorter time step shortens the distance moved past a crossing; how much it helps here
was not measured.

**What a run writes.** Every candidate goes to the log (time, states, gap, type of the
crossing, classically allowed or forbidden, `a^2`, `b^2`, the kinetic energy along the
direction, the probability or the transmission, the random number, and the outcome: hop,
stay, frustrated, forbidden) and to `<stem>_hops.csv`; the active state of every step and
the energies of all the states go to `<stem>_states.csv`, and every frame of the trajectory
carries `state=N` on its comment line. The end of the log counts the candidates, hops,
frustrated and forbidden ones. `plot_surface_hopping_.png` draws the energies of all the
states over time (eV above the lowest at the start), the state the trajectory is on (black
points), and the hops (red stems); with `-traj N` the share of the trajectories on every
state over time goes to `surface_hopping_populations.csv` and
`plot_surface_hopping_populations_.png` in the directory of the first trajectory
(`workflow::md_hopping_output`, also callable on the `<stem>_states.csv` of runs of one's
own); `mdmain RUN_DIR -analyze_only` draws the figure of a finished run again. A restart replays the evaluations of the journal with the
same seed, so the active state and the random numbers are rebuilt: a run stopped after a
hop and restarted writes the same trajectory, states, and hops as one without the stop
(`tests/md_surface_hopping.rs`).

**Any program, without a change of the library.** Surface hopping needs of the electronic
structure only the energies of the states and their gradients. Every calculator that gives
excited states through `EvalRequest::excited_state` serves (`gfn0xtb`; `pyscf`, the TDA
states of HF and DFT, and the other engines of `mopt_rs.calculators` that declare excited
states; a Rust `Calculator` of one's own). An engine whose `compute` declares a parameter
`gradient` (as `compute(self, structure, hessian, gradient=True)`) is called with
`gradient=False` for the energies of the other states and may leave the gradient out: PySCF
then skips the TDA gradients of those states. From Python, `run_md(None, x, ...,
options={"hopping_states": "0,1,2"}, ground=g, excited=e, electronic_state=1)` takes the
states from two functions of one's own, `g(x) -> (energy, gradient)` for the ground state
and `e(x, state, need_gradient) -> (energy, gradient or None)` for the excited ones, and
asks of them only what the step needs (`tests/test_md.py`: the two-state model as two
Python functions hops with the probability of eq. 2.11); the ask/tell machine underneath
gives `states` in `ask()` (the active state, all states, the neighbours, `gradient_gap`) and
takes `tell(..., state_energies=..., state_gradients=...)`. Measured with `pyscf`
(formaldehyde, B3LYP/STO-3G, TDA states 1 to 3 from state 2, 600 K, NVE, 10 fs): the
conserved quantity within 6.4e-5 Hartree, 82 gradients and 42 energies alone in 40 steps
(159 s; with the gradients of all states, 151 s on the same shared machine: for so small a
molecule the TDA gradient is not what costs; for larger ones, where it is, the saving is
expected to show, which was not measured).

**Checks.** The probabilities against eqs. 2.1-2.2, 2.11, and 2.26 written out and against
the functions of the kinetics, the limits (a true degeneracy hops with certainty; a vanishing
product of the forces gives Landau-Zener), the classically allowed regions, and the change
of the velocities (the kinetic energy changes by the potential energy, the perpendicular
part not at all, the sign along the direction kept) in `md::hopping`; on the two-state
model above with 400 trajectories from the lower state the share ending on the upper state
is 0.608 against `p = 0.626` (standard error 0.024), and the total energy changes by at most
6.7e-7 Hartree from one step to the next, across the hop as elsewhere (NVE, 0.25 fs); a run
that hops and one that stays agree to 1e-13 Hartree up to the step of the hop; under the
Nose-Hoover chain, Langevin (BAOAB), and Bussi thermostats, in a periodic cell, and at
constant pressure (Martyna-Tobias-Klein, the stochastic cell rescaling, Berendsen) the hop
moves energy between the potential and the kinetic energy only (to 1e-14 Hartree) and the
conserved quantity of the ensembles that have one changes at the hop step by less than
1e-5 Hartree;
formaldehyde on the `gfn0xtb` states 1 to 3 from state 2 for 10 fs (NVE, 0.25 fs) keeps the
conserved quantity within 2e-4 Hartree (`tests/cli_mdmain_hopping.rs`).
`examples/md_surface_hopping.rs` runs the model through `mdmain` over many trajectories and
writes its figures.

**The default gap.** `hopping_gap=1` eV. On formaldehyde
(`gfn0xtb`, states 1 to 3 from state 2, 600 K, NVE, 100 fs with every gap minimum taken)
the minima to state 3 lay 1.19 to 1.51 eV above and were classically forbidden, and those to
state 1, 1.98 to 2.18 eV, had `-ln p` of eq. 2.11 between 92 and 144 (`p` from 1e-40 to
1e-63). The exponent grows with the square of the gap at a given velocity and difference of
the forces (Landau-Zener's `pi Delta^2 / (4 v |F2 - F1|)`), so that these minima at 1 eV
would have `-ln p` from 23 to 32 (`p` below 1e-10), and at 0.5 eV from 6 to 8. Above 1 eV
hops do not matter for such motion; a run four times faster along the direction still has
`p` below 1e-2 there. The gradients of the neighbours (`6N` excited-state calculations each
for `gfn0xtb`) are paid for only below the gap: that run made 3 gradients per step with
every minimum taken, 1 with the default.

References: T. Ishida, S. Nanbu, H. Nakamura, Int. Rev. Phys. Chem. 2017, 36, 229 (the
authors' draft: the Zhu-Nakamura formulas from adiabatic potentials, eqs. 2.1-2.62, and
ZN-TSH, section 3.1); L. Yu, C. Xu, Y. Lei, C. Zhu, Z. Wen, Phys. Chem. Chem. Phys. 2014, 16,
25883 (the diabatic forces, the hopping direction, and the change of the momenta without
coupling vectors); C. Zhu, H. Nakamura, J. Chem. Phys. 1994, 101, 10630 (the theory; not
read); C. Zhu, H. Kamisaka, H. Nakamura, J. Chem. Phys. 2001, 115, 11036 (ZN-TSH; not read).

## Correlation functions, spectra, and transport coefficients (`-analysis acf ir transport`)

The correlation functions are those of `analysis::correlation`: the autocorrelation
`C(m dt) = <x(t) x(t + m dt)>` averaged over all `n - m` time origins (the unbiased
estimator), computed with the fast Fourier transform of `math::fft` (radix 2, the series
padded with zeros to a power of two of at least twice its length, so that the circular
correlation is the linear one; checked against the direct double sum to 1e-10), summed
over the components of vectors; `<P_l(u(0) . u(t))>` (`l` = 1, 2) of unit vectors for the
orientation of bonds. The runs write, besides the trajectory, every step of the series the
correlations need: `<stem>_dipoles.csv` (the dipole of the atomic charges, e Bohr; the
EEQ charges of `gfnff` and `gfn0xtb`), `<stem>_heat_flux.csv` (`-md_opt heat_flux=on`),
`<stem>_pressure_tensor.csv` (`-md_opt stress=on`), and the exchanged heat in
`md_properties.csv` (`-md_opt exchange=N`).

| analysis | what |
|---|---|
| `acf` | normalized autocorrelations, integrated correlation times, and power spectra of the energies, temperature, pressure, volume; autocorrelations of the measures of `atoms=`; `P1` and `P2` of the bonds of `bonds=1,2;3,4`; the mass-weighted velocity autocorrelation of every atom (`vacf_atoms.csv`) |
| `ir` | the dipole autocorrelation and the infrared spectrum |
| `transport` | the self-diffusion coefficient by Einstein (mean squared displacement) and by Green-Kubo (velocity autocorrelation), the thermal conductivity by Green-Kubo (heat current) and by the reverse non-equilibrium method, the shear viscosity by Green-Kubo (off-diagonal pressure) |

Spectra (`analysis::correlation::spectrum`): the one-sided cosine transform of a
correlation, `S(nu) = 4 c int_0^tau_max C(tau) w(tau) cos(2 pi c nu tau) d tau` (`nu` in
cm^-1), by the fast Fourier transform of the windowed, even-extended correlation padded with
zeros (`padding=4` by default: the grid is `1 / (P dt c)` with `P` the power of two at least
four times `2 L`; padding refines the grid, the resolution stays `1 / (c tau_max)`). Windows
(`window=`): Hann (`cos^2(pi x / 2)` of `x = tau / tau_max`, the default) and Blackman
(`0.42 + 0.5 cos(pi x) + 0.08 cos(2 pi x)`: side lobes below 1e-3 of the peak against 2.5e-2,
a peak 1.5 times as wide). The vibrational density of states (`vacf`) and the power spectrum
of any series (`power_spectrum`) are such transforms. The infrared absorption is
`alpha(omega) ~ omega (1 - exp(-beta hbar omega)) Q(omega) S_mu(omega)` with the quantum
correction `Q` of `qcf=` (Ramirez, Lopez-Ciudad, Kumar, and Marx, J. Chem. Phys. 2004, 121,
3973): `harmonic` (the default; `Q = beta hbar omega / (1 - exp(-beta hbar omega))`, so that
`alpha ~ omega^2 S_mu`, the classical form), `standard` (`Q = 2 / (1 + exp(-beta hbar omega))`),
`schofield` (`Q = exp(beta hbar omega / 2)`), or `none` (the dipole spectrum itself), at the
mean temperature of the run; the spectrum is scaled to a largest value of 1.

Green-Kubo integrals (`analysis::correlation::transport`): `L = A int_0^inf <X(0) X(t)> dt`
with the correlation averaged over the components:

| coefficient | `X` | `A` |
|---|---|---|
| self-diffusion `D` | velocity components `v_i,a` of all atoms | 1 |
| thermal conductivity `kappa` | `J_a = sum_i (e_i v_i + W_i v_i)_a` | `1 / (V k_B T^2)` |
| shear viscosity `eta` | `P_xy`, `P_xz`, `P_yz` | `V / (k_B T)` |

with `e_i = m_i v_i^2 / 2 + U_i` and `W_i = sum_j r_ij (x) dU_j/dr_ji` (the site energies
and virials of the atoms; Fan et al., Phys. Rev. B 2015, 92, 094301, eq. 13; the sum runs
over the neighbours `j` and their periodic images, an atom's own images included, as in
the stress of the atoms of LAMMPS), which the built-in `lj` with a cutoff, `eam`, `emt`,
and `tersoff` give (`Calculator::site_terms`; their sum is the energy, and the sum of the
virials is minus the virial of the stress, checked to 1e-10 also in cells smaller than
twice the cutoff); calculators that cannot split their energy over the atoms are refused for
`heat_flux=on` with this reason, and the reverse non-equilibrium method serves for them.
The integral is read at the first lag where the correlation crosses zero (the end of its
first decay; afterwards the integral of the noise only adds noise), and the mean of the
running integral from there to twice that lag is given as a check of the plateau; the error
is the standard error over 10 consecutive blocks of the run (`gk_blocks=`), and the
longest lag is half a block (`gk_lag=`). Without a zero crossing (a velocity
autocorrelation of a dilute fluid decays without one) the last lag is taken and the log
says so. In the reverse non-equilibrium method (Muller-Plathe, J. Chem. Phys. 1997, 106,
6082; `-md_opt exchange=N`, NVE in a cell), every `N` steps the atom of the cold slab (the
first of `exchange_slabs=20` along `exchange_axis=c`) with the largest kinetic energy and
the atom of the hot slab (the middle one) with the smallest, of the same mass, exchange
velocities; the conductivity is the exchanged heat over `2 A t` divided by the gradient of
the temperature profile fitted in both halves (the two slabs left out), with the error from
10 blocks of the run. That error is the scatter of the gradient over the blocks of one run,
the exchanged heat of the whole run in every block: it is not the spread of independent runs.
On the Lennard-Jones fluid below (216 atoms) it lies a factor 1.7 to 2 below that spread
(eight seeds, exchanges every 20 steps: `kappa*` 2.85 +- 0.36 over the runs, 0.12 to 0.21
reported by each; every 100 steps: 2.86 +- 1.1 over the runs, 0.29 to 1.6 reported). The
blocks (10 ps) are longer than the relaxation of the profile (about 1.6 ps for the slowest
mode of a cell of 7.1 `sigma`), so their correlation does not explain it; the fluctuation of
the flux, which the blocks do not see, explains about a factor 1.2; the rest is the
variation from trajectory to trajectory. For an error bar, run several seeds and take the
standard error of their mean. The Einstein diffusion coefficient has the error of five
parts of the run.

**Checks** (`tests/md_correlations.rs`; `cargo run --release -p mopt-rs --example
md_transport -- STEPS [ATOMS_PER_EDGE]` for longer runs). Water (GFN-FF, NVE from a
structure off the minimum at 300 K, 2 ps, frames 0.5 fs apart, Blackman window): the
density of states peaks at 1617 and 3636 cm^-1 and the infrared spectrum (harmonic
correction) at 1617 and 3632 cm^-1, against the harmonic 1633, 3634, and 3637 cm^-1 of
`freqmain` (the overtone of the stretches appears at 7268 cm^-1 with 4 % of the height); `P2`
of the O-H bonds starts at 1. The Lennard-Jones fluid of the built-in `lj` (UFF argon:
`sigma` = 3.4006 Angstrom, `epsilon / k_B` = 119.26 K) at `rho* = 0.6`, `T* = 1.35`, cut off
at `2.5 sigma` (switched off over the last 0.3 Angstrom), 216 atoms, 10 fs (`dt* = 0.0046`),
120 ps, the first 20 ps left out:

| quantity | method | result | reference |
|---|---|---|---|
| self-diffusion | Einstein (MSD) | 1.074 +- 0.020 Angstrom^2/ps (`D*` 0.200) | |
| self-diffusion | Green-Kubo (VACF; Bussi thermostat with a 5 ps coupling time) | 1.090 +- 0.019 Angstrom^2/ps (`D*` 0.203) | the two agree |
| thermal conductivity | Green-Kubo (heat current) | `kappa*` = 2.92 +- 0.34 | about 3.4 (Evans, Phys. Rev. A 1986, 34, 1449, Fig. 1) |
| thermal conductivity | reverse non-equilibrium (NVE, an exchange every 20 steps, 20 slabs; mean of eight seeds, standard error of the mean) | `kappa*` = 2.85 +- 0.13 (single runs 2.49 to 3.54) | the same |
| shear viscosity | Green-Kubo (off-diagonal pressure) | `eta*` = 0.857 +- 0.052 | |

With 512 atoms, NVE after 30 ps of equilibration with Bussi's thermostat, 400 ps each
(`md_transport 40000 8`, 1154 s for the three runs; the reverse run with the first 80 ps
left out):

| quantity | method | result |
|---|---|---|
| self-diffusion | Einstein | 1.182 +- 0.026 Angstrom^2/ps (`D*` 0.221) |
| self-diffusion | Green-Kubo | 1.186 +- 0.013 Angstrom^2/ps (`D*` 0.221) |
| thermal conductivity | Green-Kubo | 0.0703 +- 0.0080 W/(m K) (`kappa*` 3.74 +- 0.42) |
| thermal conductivity | reverse non-equilibrium | 0.0726 +- 0.0134 W/(m K) (`kappa*` 3.86 +- 0.71) |
| shear viscosity | Green-Kubo | 6.32e-5 +- 0.16e-5 Pa s (`eta*` 0.700 +- 0.018) |

(`kappa* = kappa sigma^2 sqrt(m / epsilon) / k_B`, `eta* = eta sigma^2 / sqrt(m epsilon)`,
`D* = D sqrt(m / epsilon) / sigma`.) Both thermal conductivities agree with each other and
with the reference within their errors; the errors of runs this short are 10 to 30 %. The
diffusion coefficient grows with the cell (0.200 with 216 atoms, 0.221 with 512), the known
finite-size effect of periodic cells (Yeh and Hummer, J. Phys. Chem. B 2004, 108, 15873); the
viscosity of the smaller run with the thermostat is higher by more than its error
(0.857 against 0.700), which these runs do not resolve further. A
reverse run must reach its steady state before it is averaged: in a run of 40 ps of 512
atoms with the first 8 ps left out, the temperature profile was still growing and the
conductivity came out at half the steady value (`kappa*` 2.0 +- 0.2), so that `skip=`
must cover the time `L^2 / (4 kappa / (rho c_v))` the heat needs to cross half the cell
(about 17 ps for 512 atoms here).

## Measurements behind the defaults

Script: `cargo run --release -p mopt-rs --example md_measurements -- nve|thermostats|npt|shake`.

**Time step.** NVE from the minimum (Ar13 from a relaxed icosahedron, ethanol relaxed with
GFN-FF) or the lattice (Cu), velocities at the temperature given; the largest deviation of
the conserved quantity from its start and the slope of a straight line through it:

| system | time step (fs) | largest deviation (meV/atom) | drift (meV/atom/ps) | mean T (K) |
|---|---|---|---|---|
| Ar13, lj, 30 K, 20 ps | 2 | 0.0001 | 0.0000 | 14.4 |
| | 10 | 0.0022 | 0.0000 | 14.4 |
| | 40 | 0.0365 | 0.0000 | 14.3 |
| Cu32 cell, emt, 300 K, 4 ps | 0.5 | 0.0027 | 0.0000 | 150.2 |
| | 2 | 0.0437 | 0.0001 | 150.2 |
| | 8 | 0.6996 | 0.0020 | 150.1 |
| ethanol, gfnff, 300 K, 2 ps | 0.1 | 0.0068 | 0.0000 | 197.1 |
| | 0.25 | 0.0422 | 0.0000 | 200.7 |
| | 0.5 | 0.1662 | 0.0000 | 204.0 |
| | 1 | 0.6797 | -0.0016 | 196.0 |
| | 2 | 3.4644 | 0.0209 | 210.8 |

Velocity Verlet loses energy with the square of the time step, as it must (a factor of 4
for each halving), with no drift below 1 fs. The rule for the default: the largest time
step whose largest deviation stays below 1 % of `k_B T` per atom at 300 K (0.26 meV): 0.5
fs for a molecule with hydrogen atoms, 2 fs for copper, 40 fs and more for argon. The
default is the one that holds for all of them, 0.5 fs; metals and noble gases can take the
larger steps of the table. For plain variables with unit masses (the Muller-Brown surface
with `k_B T = 1e-3`) the deviation is 1.1e-6 at 0.1 (0.1 % of `k_B T`) and 1.8e-5 at 0.4,
so MultiOptPy's 0.1 stays the default there.

**Thermostats.** Started at a tenth of the target temperature; the mean and the variance of
the instantaneous temperature over the last four fifths of 60000 steps (10 fs for Ar13, 600
ps; 0.5 fs for ethanol, 30 ps) against the canonical `var(T) = 2 T^2 / N_f`, and the time
until the mean over 50 fs is within 5 % of the target:

| system | thermostat | tau (fs) | mean T (K) | var(T) / canonical | time to 5 % (fs) |
|---|---|---|---|---|---|
| Ar13, lj, 30 K | berendsen | 20 | 30.0 | 0.015 | 250 |
| Ar13, lj, 30 K | berendsen | 100 | 30.0 | 0.167 | 1120 |
| Ar13, lj, 30 K | berendsen | 500 | 30.0 | 0.440 | 2190 |
| Ar13, lj, 30 K | bussi | 20 | 30.0 | 1.005 | 260 |
| Ar13, lj, 30 K | bussi | 100 | 30.0 | 0.990 | 470 |
| Ar13, lj, 30 K | bussi | 500 | 30.0 | 1.036 | 1530 |
| Ar13, lj, 30 K | langevin | 20 | 30.0 | 0.993 | 160 |
| Ar13, lj, 30 K | langevin | 100 | 30.0 | 0.996 | 240 |
| Ar13, lj, 30 K | langevin | 500 | 30.1 | 1.033 | 670 |
| Ar13, lj, 30 K | andersen | 20 | 30.0 | 0.973 | 80 |
| Ar13, lj, 30 K | andersen | 100 | 29.9 | 0.970 | 680 |
| Ar13, lj, 30 K | andersen | 500 | 30.1 | 0.997 | 3590 |
| Ar13, lj, 30 K | nose_hoover | 20 | 30.1 | 15.755 | 990 |
| Ar13, lj, 30 K | nose_hoover | 100 | 30.0 | 0.998 | 750 |
| Ar13, lj, 30 K | nose_hoover | 500 | 30.0 | 1.125 | 1330 |
| Ar13, lj, 30 K | nose_hoover_chain | 20 | failed: non-finite energy at step 5338 | | |
| Ar13, lj, 30 K | nose_hoover_chain | 100 | 30.0 | 1.016 | 530 |
| Ar13, lj, 30 K | nose_hoover_chain | 500 | 30.1 | 1.054 | 3070 |
| ethanol, gfnff, 300 K | berendsen | 20 | 300.0 | 0.195 | 76 |
| ethanol, gfnff, 300 K | berendsen | 100 | 300.0 | 0.358 | 394 |
| ethanol, gfnff, 300 K | berendsen | 500 | 300.1 | 0.524 | 1222 |
| ethanol, gfnff, 300 K | bussi | 20 | 297.1 | 0.947 | 76 |
| ethanol, gfnff, 300 K | bussi | 100 | 296.2 | 0.936 | 215 |
| ethanol, gfnff, 300 K | bussi | 500 | 290.0 | 0.841 | 1010 |
| ethanol, gfnff, 300 K | langevin | 20 | 297.4 | 0.960 | 73 |
| ethanol, gfnff, 300 K | langevin | 100 | 307.0 | 1.017 | 423 |
| ethanol, gfnff, 300 K | langevin | 500 | 311.0 | 1.109 | 1246 |
| ethanol, gfnff, 300 K | andersen | 20 | 297.0 | 0.933 | 82 |
| ethanol, gfnff, 300 K | andersen | 100 | 293.7 | 0.895 | 546 |
| ethanol, gfnff, 300 K | andersen | 500 | 284.1 | 0.749 | 2043 |
| ethanol, gfnff, 300 K | nose_hoover | 20 | 300.0 | 1.209 | 60 |
| ethanol, gfnff, 300 K | nose_hoover | 100 | 299.8 | 2.460 | 170 |
| ethanol, gfnff, 300 K | nose_hoover | 500 | 296.2 | 4.100 | 817 |
| ethanol, gfnff, 300 K | nose_hoover_chain | 20 | 303.7 | 1.061 | 102 |
| ethanol, gfnff, 300 K | nose_hoover_chain | 100 | 303.7 | 1.057 | 280 |
| ethanol, gfnff, 300 K | nose_hoover_chain | 500 | 310.1 | 1.175 | 1542 |

Berendsen keeps the mean but narrows the distribution (as known); one Nose-Hoover
thermostat is not ergodic for ethanol (1.2 to 4.1 times the canonical variance), and with
a coupling time of twice the time step it breaks down for Ar13 (20 fs at 10 fs); the
chain, Bussi's rescaling, Langevin, and Andersen are canonical within the statistical
error of runs of this length. The default coupling time is 100 fs: canonical for every
canonical thermostat on both systems, equilibrated within 0.2 to 0.7 ps, while 500 fs
takes 1 to 3.6 ps and 20 fs breaks the Nose-Hoover thermostats when the time step is
large. The default ensemble is the Nose-Hoover chain (deterministic, with a conserved
quantity, canonical in both tests), not MultiOptPy's single Nose-Hoover thermostat. A
coupling time below ten time steps is reported in the log.

**Barostats.** 32 copper atoms (EMT, bulk modulus 129.8 GPa from the pressure at two
volumes), 300 K, 0 GPa, 4 fs, 100 ps (the first 20 ps not used); the variance of the volume
against the isothermal-isobaric `k_B T V / B`, and the mean pressure with its standard
error from 20 blocks, and the largest component of the mean pressure tensor
`sum m v v^T / V - sigma` (the target is zero for every component):

| barostat | tau_p (fs) | compressibility (1/GPa) | seed | mean V (ang.^3) | var(V) / (kT V / B) | mean P (GPa) | largest mean P_ab (GPa) |
|---|---|---|---|---|---|---|---|
| berendsen | 500 | 0.0077 | 5 | 376.63 | 0.011 | -0.002 +- 0.005 | 0.067 |
| berendsen | 500 | 0.0077 | 6 | 376.73 | 0.012 | -0.001 +- 0.006 | 0.033 |
| berendsen | 1000 | 0.0077 | 5 | 376.62 | 0.006 | -0.003 +- 0.008 | 0.042 |
| berendsen | 1000 | 0.0100 | 5 | 376.63 | 0.007 | -0.003 +- 0.008 | 0.081 |
| crescale | 500 | 0.0077 | 5 | 376.12 | 0.965 | 0.198 +- 0.119 | 0.248 |
| crescale | 500 | 0.0077 | 6 | 377.05 | 0.905 | -0.097 +- 0.092 | 0.117 |
| crescale | 1000 | 0.0077 | 5 | 375.87 | 0.936 | 0.279 +- 0.136 | 0.347 |
| crescale | 1000 | 0.0100 | 5 | 375.95 | 0.948 | 0.245 +- 0.124 | 0.262 |
| mtk | 500 | 0.0077 | 5 | 376.77 | 1.108 | 0.001 +- 0.085 | 0.061 |
| mtk | 500 | 0.0077 | 6 | 376.90 | 1.021 | -0.000 +- 0.056 | 0.036 |
| mtk | 1000 | 0.0077 | 5 | 376.75 | 0.974 | -0.001 +- 0.015 | 0.030 |
| mtk | 1000 | 0.0100 | 5 | 376.75 | 0.974 | -0.001 +- 0.015 | 0.030 |
| berendsen_aniso | 500 | 0.0077 | 5 | 376.64 | 0.010 | -0.001 +- 0.005 | 0.018 |
| berendsen_aniso | 500 | 0.0077 | 6 | 376.80 | 0.010 | -0.001 +- 0.005 | 0.006 |
| berendsen_aniso | 1000 | 0.0077 | 5 | 376.65 | 0.006 | -0.003 +- 0.011 | 0.010 |
| berendsen_aniso | 1000 | 0.0100 | 5 | 376.65 | 0.007 | -0.003 +- 0.008 | 0.017 |
| crescale_aniso | 500 | 0.0077 | 5 | 376.14 | 1.120 | 0.292 +- 0.158 | 0.413 |
| crescale_aniso | 500 | 0.0077 | 6 | 377.07 | 0.976 | -0.014 +- 0.168 | 0.264 |
| crescale_aniso | 1000 | 0.0077 | 5 | 375.81 | 1.307 | 0.393 +- 0.212 | 0.535 |
| crescale_aniso | 1000 | 0.0100 | 5 | 375.96 | 1.228 | 0.353 +- 0.190 | 0.485 |
| mtk_aniso | 500 | 0.0077 | 5 | 377.05 | 1.137 | 0.001 +- 0.041 | 0.001 |
| mtk_aniso | 500 | 0.0077 | 6 | 377.10 | 1.042 | -0.001 +- 0.104 | 0.001 |
| mtk_aniso | 1000 | 0.0077 | 5 | 377.02 | 1.280 | 0.002 +- 0.053 | 0.003 |
| mtk_aniso | 1000 | 0.0100 | 5 | 377.02 | 1.280 | 0.002 +- 0.053 | 0.003 |

Berendsen holds the pressure but suppresses the fluctuations of the volume (1 % of the
canonical variance); the stochastic cell rescaling and Martyna-Tobias-Klein give the
canonical fluctuations. The mean pressure of the cell rescaling scatters by its larger
statistical error (two seeds: +0.20 and -0.10 GPa, each about one to two standard errors);
Martyna-Tobias-Klein, whose cell momentum integrates the pressure, holds it to a few MPa
over the run. The compressibility only sets the coupling of the first-order barostats;
their default, 0.01 /GPa, is that of a soft solid (copper: 0.0077 /GPa). The default
coupling time of the barostats is 1000 fs, the default barostat of `-mt npt` is
Martyna-Tobias-Klein.

The anisotropic barostats, whose cell changes its shape, behave as their isotropic forms:
the anisotropic Berendsen barostat suppresses the fluctuations of the volume (1 %); the
anisotropic cell rescaling gives the canonical fluctuations (0.98 to 1.31 of `k_B T V / B`)
with the mean pressure and its components scattering by their statistical errors as the
isotropic one does (about 0.2 GPa over 80 ps); the flexible Martyna-Tobias-Klein barostat
gives canonical fluctuations (1.04 to 1.28) and holds every component of the mean pressure
tensor within 3 MPa. The same run is `tests/md_dynamics.rs` for 12 ps (every component of
the mean pressure tensor within three of its standard errors; the lengths and the angles of
the cell move; the conserved quantity of the flexible barostat stays). A cell whose shape
is free of any stiffness (a gas) wanders to any shape under an anisotropic barostat; the
volume is not affected.

**SHAKE and RATTLE.** Ethanol (GFN-FF, 300 K, 2 ps) with its six bonds to hydrogen held:

| time step (fs) | largest residual (Bohr) | largest deviation (meV/atom) | drift (meV/atom/ps) |
|---|---|---|---|
| 0.5 | 1.0e-10 | 0.0460 | -0.0007 |
| 1 | 1.0e-10 | 0.1873 | -0.0021 |
| 2 | 8.9e-16 | 0.7891 | -0.0170 |

The constraints stay at their lengths to the tolerance of SHAKE (1e-10) at every step,
and the conserved quantity deviates about as much at 1 fs with the bonds held as without
them at 0.5 fs (0.19 against 0.17 meV/atom): holding the bonds to hydrogen allows twice
the time step.
