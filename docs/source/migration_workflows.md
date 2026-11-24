# Differences from MultiOptPy: dynamics, searches, the mapper, and kinetics

Molecular dynamics, the searches of orientations and conformations, relaxed scans, the
reaction-network mapper, and kinetics.

## Molecular dynamics (`mdmain`)

`mdmain` takes MultiOptPy's options (`-time`, `-ts`, `-temp`, `-mt`, `-ct`, `-cc`, `-fix`,
`-pc`, `-pbc`, `-press`, `-traj`, `-gi`, `-cmds`, `-pca`, the calculator options, all bias
potentials with `-nrp`) and writes its files (`energy_profile.csv`, `gradient_profile.csv`,
`tot_energy_profile.csv`, `energy_profile_kcalmol.csv`, `<stem>_traj.xyz`,
`maximum_value.txt`, `minimum_value.txt`, `local_min_grad.txt`, `energy_plot_.png`,
`plot_gradient_.png`, `plot_temperature_.png`, `plot_tot_energy_.png`). The dynamics is the
ask/tell state machine `md::MolecularDynamics` with replaceable integrators, thermostats,
and barostats ([molecular_dynamics.md](molecular_dynamics.md), "Molecular dynamics"). What differs, with the test
that shows it:

### Errors of MultiOptPy's dynamics that are not repeated

- **Units of the masses** (`MD/thermostat.py:Thermostat.__init__`): the masses are atomic
  mass units used as if they were atomic units (electron masses) with forces in
  Hartree/Bohr and the time step in atomic units, so the motion is
  `sqrt(1822.888) = 42.7` times faster than the time step says: `-ts 0.1` (its default)
  moves the atoms as a step of 4.27 a.u. (0.10 fs) would, and the times it prints (fs) are
  42.7 times too short. The masses are in electron masses, `-ts` is in atomic units as its
  help says (or femtoseconds with `fs` after the number, `-ts 0.5fs`), and the default is
  0.5 fs for atoms (measured; 0.1 for plain variables with unit masses).
  `tests/md_dynamics.rs::the_energy_error_of_velocity_verlet_falls_with_the_square_of_the_time_step`.
- **Velocity Verlet** (`Velocity_Verlet`): the first step adds half a kick with a zero
  "previous" force and then moves the positions with the full force term again, so the
  first displacement has twice the force term. Here the step is the standard one.
  `md::tests::velocity_verlet_conserves_the_energy_of_an_oscillator_to_second_order`.
- **Nose-Hoover and Nose-Hoover chain** (`Nose_Hoover_thermostat`,
  `Nose_Hoover_chain_thermostat`): the second half kick uses the force at the positions
  before the step (the force at the new positions is never used in that step), so the
  scheme is neither time reversible nor second order; the chain variables move by explicit
  Euler steps with fixed masses (`Q = 0.1` for one thermostat; `1, 2, 3, 6, 10, ..., 200`
  for a chain of ten) that do not depend on the temperature or the number of degrees of
  freedom. Here the chains are integrated as Martyna, Tuckerman, Tobias, and Klein (1996)
  do, with `Q_1 = N_f k_B T tau^2`, `Q_j = k_B T tau^2` from the coupling time `tau`, and the
  conserved quantity includes the energy of the chain.
  `md::tests::canonical_thermostats_give_the_mean_and_variance_of_the_temperature`.
- **Langevin (BAOAB)** (`Langevin_thermostat`): both half kicks use the force at the
  positions before the step (the B after the evaluation needs the new force), the friction
  is `0.01` in the time unit of the wrong masses (about 10 ps^-1), and the random numbers
  come from `numpy.random` without a seed (no run can be repeated). Here BAOAB has the new
  force in its last kick, the friction is `1 / tau` (100 fs, the same 10 ps^-1), and every
  random number comes from one generator seeded by `-md_opt seed=` (drawn and written to
  `settings.json` when it is not given), so a run can be repeated and restarted bit for
  bit. `md::tests::the_same_seed_and_evaluations_give_the_same_run_bit_for_bit`,
  `tests/cli_mdmain.rs::a_stochastic_run_restarts_from_its_journal_bit_for_bit`.
- **Degrees of freedom** (`Thermostat.g_value`): always `3N`. Here the temperature has
  `N_f` = moving coordinates minus independent constraints minus frozen modes minus the
  motions of the whole system that are conserved (3 translations, and 3 or 2 rotations of
  an isolated structure; none with stochastic thermostats, held atoms, or bias
  potentials, `-md_opt rigid=` to choose).
  `tests/cli_mdmain.rs::an_nve_run_of_a_cluster_writes_its_files_and_keeps_its_energy`.
- **Starting velocities** (`init_purtubation`): drawn at `-temp` without removing the
  motion of the whole system, and not scaled. Here the conserved motions and the components
  across the constraints are removed and the velocities are scaled to `-temp` exactly
  (`-md_opt velocities=maxwell` keeps them as drawn, as ASE does; GROMACS scales them).
- **Atoms of `-fix`** (`moleculardynamics.py:MD.md`): their positions are reset after every
  step, but their momenta stay, so they carry kinetic energy and the temperature is wrong.
  Here they have zero inverse mass: no velocity, no kinetic energy, not counted.
  `md::space::tests`, `tests/test_md.py::test_structures_with_a_built_in_calculator`.
- **`-cc` (SHAKE)** (`Constraint/constraint_condition.py:SHAKE.run`): distances follow
  Ryckaert et al. with a tolerance of 1e-5; angles and dihedral angles use factors of
  `1e+5` and `1e+7` without a derivation and overwrite the momenta of the atoms with the
  correction. Here every constraint (`-cc`, `-pc`, `-gfix`, `-uc`, Python functions) is held
  by SHAKE in matrix form (Newton iterations on the multipliers along `M^-1 J^T`, 1e-10)
  and RATTLE for the velocities. `md::constraints::tests`,
  `tests/md_dynamics.rs::bonds_to_hydrogen_stay_at_their_lengths_with_shake_and_rattle`.
- **`-pc`**: the gradient is projected and the positions are moved back after the step,
  but the velocities keep their components across the constraints. RATTLE takes them out.
- **`-pbc`**: the fragments are moved into the box after every step, and the calculator
  never sees a cell (no periodic interactions). Here the box is the cell of the structure
  (also from `Lattice=` of an extended XYZ file): the calculators see it, the dynamics is
  unwrapped, and the trajectory has the molecules wrapped into the cell as wholes.
- **`-press`**: converted to atomic units and never used. Here it is the target of the
  barostat of an NPT run (`-mt npt`, ...).
- **`-gi`**: the parser of `mdmain` replaces its value, so nothing is written. Here
  `geometry_info.csv` and the orthogonality plots of `optmain -gi` are written for the steps.
- **`-cmds`, `-pca`**: read and not used. Here they map the frames of the trajectory (at
  most 400, thinned evenly).
- **Nanoreactor potential** (`-nrp`; `Potential/potential.py` and
  `Potential/nanoreactor_potential.py`): the potential receives the iteration number as its
  time (in atomic units), so its periods of picoseconds are counted in steps; and its walls
  differ from the paper (see "Corrected bias potentials"). Here time-dependent bias
  potentials receive the simulated time of every evaluation
  (`pes::term::TermContext::time_au`, also `t` of `-ubias`).
  `tests/md_dynamics.rs::the_nanoreactor_compresses_on_its_period`,
  `workflow::md::tests`. The sphere is centred at the origin of the coordinates, as in
  the paper and in MultiOptPy. MultiOptPy also cites Meissner and Meisner (J. Chem. Theory
  Comput. 2025, 21, 218), whose subject is the acceleration of diffusion in the relaxation
  phase (DA-NMD); neither MultiOptPy nor mopt-rs implements it (its method is not described
  in the parts of the paper that could be read).
- **`tot_energy_profile.csv`**: from the second step on it has the total energy of the step
  before (one row fewer than the other profiles). Here every step has its own row.
- **Stop file**: `end.txt` in the run directory. Here the `STOP` file of every command, and
  `--restart RUN_DIR` continues the run from its journal.

### Defaults that differ

- `-mt` without a value: a Nose-Hoover chain of three (MultiOptPy: one Nose-Hoover
  thermostat). The measurements in [molecular_dynamics.md](molecular_dynamics.md) show why: one Nose-Hoover
  thermostat gives an ethanol molecule a variance of the temperature 1.2 to 4 times the
  canonical one; the chain stays within the statistical error. `-mt nosehoover` is still
  one thermostat.
- `-ts` without a value: 0.5 fs for atoms (MultiOptPy's 0.1 means 0.10 fs of real motion
  through the error of the masses above), 0.1 time units for plain variables.
- The trajectory has every tenth step (MultiOptPy keeps the input files of every tenth step
  and builds its trajectory from them); `-md_opt write_every=1` writes every step.

### Kept as in MultiOptPy

`-time` is the number of steps (default 100000), `-temp` the starting and target
temperature, `-ct STEP K ...` changes the target from the given steps on, `-traj N` runs
N trajectories one after the other (each in its own run directory; here with seeds one
apart), `-order` and `-addint` have no effect, the log has the `hamiltonian :` (potential
energy with bias potentials plus kinetic energy) and `Instantaneous_temperature:` lines.

### Added

- Ensembles and parts (`-mt`, `-md_opt`): velocity Verlet and BAOAB; Berendsen, Bussi's
  velocity rescaling, Langevin, Andersen, Nose-Hoover chains; NPT with the barostats of
  Berendsen (isotropic and anisotropic), Bernetti and Bussi (stochastic cell rescaling),
  and Martyna, Tobias, and Klein. All are traits with registries
  (`md::integrator::IntegratorRegistry`, `md::thermostat::ThermostatRegistry`,
  `md::barostat::BarostatRegistry`).
- Mode freezing (`-md_opt freeze=negative|below:VALUE|modes:...|file:PATH`,
  `freeze_update=N`): no motion along chosen modes of the mass-weighted Hessian.
  `md::tests::with_the_saddle_direction_frozen_the_dynamics_stays_at_the_muller_brown_saddle`,
  `tests/cli_mdmain.rs::plain_variables_freeze_the_saddle_direction_of_muller_brown`.
- Umbrella sampling (`-umbrella cv=EXPR ...`) with the free-energy profile by WHAM and
  MBAR (`analysis::free_energy`, `mopt_rs._core.wham`, `mopt_rs._core.mbar`), and
  collective variables (`md::cv::Cv`: distances, angles, dihedral angles, coordination
  numbers, path variables, expressions, any term). String methods in the space of
  collective variables are not implemented yet (their design is in
  [molecular_dynamics.md](molecular_dynamics.md)). `analysis::free_energy::tests`,
  `tests/cli_mdmain.rs::umbrella_sampling_of_a_distance_gives_a_profile`.
- Coordinate systems (`-coord`): `<stem>_coordinates.csv` has the coordinates at the
  frames; with `-md_opt integrate=coordinates` the dynamics runs in non-redundant
  generalized coordinates (`dic`, `tric`, coordinates that cover all motions) with the
  kinetic energy `1/2 p^T G p` and the generalized leapfrog
  (`md::generalized::GeneralizedDynamics`); it conserved the energy exactly as well as
  Cartesian dynamics at 10 to 60 times the cost, so Cartesian dynamics stays the default.
  `tests/md_coordinates.rs`.
- LINCS for bond lengths (`-md_opt constraint_solver=lincs`, `md::lincs`), initial
  conditions from the harmonic Wigner distribution of the normal modes
  (`-md_opt velocities=wigner`, `md::wigner`), and collective variables in one form for the
  output, the analysis, restraints, and umbrella sampling (`-cv`, `-cv_bias`,
  `md::cv_spec`). `tests/md_dynamics.rs`, `tests/md_wigner.rs`, `tests/cli_mdmain.rs`.
- Correlation functions, spectra, transport coefficients (`-analysis acf ir transport`,
  `analysis::correlation`): FFT autocorrelations with the direct sum for checking,
  orientational correlations, power spectra and the vibrational density of states with Hann
  or Blackman windows and zero padding, infrared spectra of the dipoles of the atomic
  charges with quantum corrections, Green-Kubo diffusion, thermal conductivity (heat current
  from the site energies and virials of `lj`, `eam`, `emt`, `tersoff`) and shear viscosity
  with block errors and a documented plateau, and Muller-Plathe's reverse non-equilibrium
  thermal conductivity (`-md_opt exchange=N`). `tests/md_correlations.rs`,
  `tests/test_md_analysis.py`, `pes::native::tersoff::tests`.
- Trajectory analysis (`-analysis`, at the end of a run; `-analyze_only`, of a finished
  run or any XYZ trajectory): statistics with block-average errors, radial distribution
  functions, mean squared displacements and diffusion coefficients, velocity
  autocorrelation and vibrational density of states, RMSD and RMSF, distances and angles,
  principal components (`analysis::trajectory`, `mopt_rs.TrajectoryAnalysis`). For it the
  velocities of the frames are written to `<stem>_velocities.xyz`.
  `tests/md_trajectory_analysis.rs` (copper as crystal and liquid, diamond silicon, water
  against `freqmain`), `tests/test_md_analysis.py`.
- Plain variables (`-os mb2d`, `-os expr:...` with a variables file), `-mf` (two
  electronic states), `-es`, the constraints of `optmain`, `md_properties.csv` (energies,
  conserved quantity, temperature, pressure, volume of every step), the last structure and
  its velocities, the conserved quantity in the log, `--config`, `--dump-config`,
  `--restart`, observers in Rust (`workflow::md::MdObserver`) and a callback in Python
  (`mopt_rs.run_md(..., callback=...)`), and the ask/tell machine from Python
  (`mopt_rs.MolecularDynamics`).

Tests: `crates/mopt-rs/src/md/`, `crates/mopt-rs/src/analysis/trajectory/`,
`tests/md_dynamics.rs`, `tests/cli_mdmain.rs`, `tests/md_trajectory_analysis.rs`,
`tests/md_coordinates.rs`, `tests/md_correlations.rs`,
`tests/test_md.py`, `tests/test_md_analysis.py`.

## Orientation search (`orientsearch`)

`orientsearch` takes MultiOptPy's options (`-nsample` 5, `-part`, `-dist` 5.0 Angstrom, and
every option of `optmain`) and writes the starts under MultiOptPy's names
(`<stem>_Sample_<k>.xyz`, comment line `Sample_<k>`, numbered from 0). Each start is optimized
by `optmain` with all of its options. The sampling is the ask/tell machine
`workflow::explore::orient::OrientationSearch` ([searches.md](searches.md), "Orientations of a
complex"). What differs, with the test that shows it:

### Errors of MultiOptPy's sampling that are not repeated

- **Polygon of three or more parts** (`make_random_orientation_xyz_file`): the centres are put
  at the radius `0.5 d tan((n - 2) pi / n)` from the origin. That is not the circumradius of a
  regular polygon with sides `d`. Neighbours end up `1.5 d` apart for three parts, `1.81 d` for
  five, `0.87 d` for six. For four parts `tan(pi / 2)` is about `1.6e16` in floating point, which
  puts the parts about 1e16 Angstrom apart. The scheme `multioptpy` uses the circumradius
  `d / (2 sin(pi / n))`, so neighbours are the drawn distance apart for every `n`.
  `workflow::explore::orient::placement::tests::multioptpy_polygon_has_neighbours_at_the_drawn_distance`.
- **The distance of two parts**: MultiOptPy draws `rand_dist` from `[d, 1.1 d]` for two parts
  as for more, but then puts the centres at `-d/2` and `+d/2`, so the draw has no effect.
  Here the drawn distance is used for every number of parts (same test).
- **Random numbers without a seed** (`random.uniform`, `random.sample`): no search can be
  repeated or continued. Here one generator is seeded by `-orient_opt seed=`. Without the
  option a seed is drawn and written to `settings.json`, so the search can be repeated and
  `--restart RUN_DIR` continues it from the journals of its samples.
  `workflow::explore::orient::search::tests::the_same_seed_gives_the_same_starts`,
  `tests/cli_orientsearch.rs::multioptpy_parts_restart_and_stop`.
- **Parts that are not checked**: an atom named in two lists is moved twice (the later part
  wins), and a number beyond the structure stops the run with an `IndexError`. Here both are
  refused with the atom named.
  `workflow::explore::orient::parts::tests::parts_from_lists_follow_multioptpy`.

### Defaults that differ

- **Without `-part`** MultiOptPy prints "No part is specified" and stops. Here the parts are the
  molecules of the bond graph (1.2 times the sums of the covalent radii), and a structure with
  one molecule is refused. `tests/cli_orientsearch.rs::the_water_dimer_with_gfnff_finds_its_hydrogen_bond`.
- **Sampling**: MultiOptPy draws three angles uniformly from `[0, 2 pi)` and rotates about x,
  y, and z in turn. These rotations are not uniform: `R_zz = cos a cos b` has the mean square
  1/4, where uniform rotations have 1/3. The centres go `d` apart (5 Angstrom by default,
  whatever the size of the molecules). The default here is the scheme `ring`: uniform
  rotations, and the centres on a regular polygon in a random plane, shrunk until the parts
  are at van der Waals contact. In a cell or with parts that stay, every part is brought from a
  random direction to contact with those placed before (`contact`). The measurements are in
  running.md. MultiOptPy's scheme, with the two errors above corrected, is
  `-orient_opt scheme=multioptpy`.
  `workflow::explore::orient::placement::tests::euler_angles_are_not_uniform_rotations`,
  `the_rotation_is_multioptpys_row_vector_product`.
- **Convergence of the samples**: MultiOptPy optimizes every sample with the default criteria
  of `optmain`. Here the samples get the tight criteria (`-tcc`) unless `-tcc` or `-lcc` is
  given; `-orient_opt criteria=normal` gives MultiOptPy's. With the default criteria the
  optimizations of the benzene dimer ended after about 4 gradients each, on the flat part of
  the surface 5 to 7.5 Angstrom apart. Copies of one minimum of the water dimer were up to
  0.4 Angstrom apart, so no RMSD separates them from its other stationary point
  (measurements in running.md). `tests/cli_orientsearch.rs::the_water_dimer_with_gfnff_finds_its_hydrogen_bond`
  (the criteria in the log).
- **Files**: MultiOptPy writes the starts next to the input and runs each optimization in its
  own run directory with a time stamp. Here everything goes into one run directory
  `<date>/<stem>_ORIENT_<time>/`: the starts, and each optimization in `<stem>_Sample_<k>_OPT/`
  (with `optmain`'s files, `<stem>_Sample_<k>_optimized.xyz` among them).

### Added

- **Distinct minima**: MultiOptPy leaves `-nsample` separate optimizations. Here results that
  are the same minimum are merged. That needs the same bonds, an RMSD below `rmsd=` after
  matching equivalent atoms and identical molecules (the Hungarian method, superpositions from
  the principal axes), and energies within `energy=`. The minima are ranked by energy with
  Boltzmann weights at `-temp`. Then come the interaction energies against the parts alone
  (`interaction=`), the files `<stem>_minimum_<rank>.xyz` and `<stem>_minima.xyz`, and the
  tables `orientsearch_minima.csv` and `orientsearch_samples.csv`.
  `workflow::explore::identity::tests`, `workflow::explore::pool::tests`.
- **Starts from known minima** (`refine=`, `priority=`), with the rules of
  `workflow::explore::PriorityRegistry`.
- **Fixed parts and periodic cells**: parts that stay (`fixed=`, and parts bonded to their own
  periodic images, such as a slab). In a cell the other parts come down onto the free side of
  a slab, or go to free points of a box. The schemes with distances between centres are
  refused in a cell, with the reason. `tests/cli_orientsearch.rs::a_molecule_comes_down_onto_a_periodic_slab`.
- **The surfaces of `optmain`**: ONIOM, `-mf`, bias potentials, constraints, `-coord`, and
  every calculator of the registry. `tests/cli_orientsearch.rs::oniom_gfn0xtb_in_gfnff_water`,
  `multioptpy_parts_restart_and_stop` (`-coord tric`).
- **The `STOP` file** of the run directory ends the search after the sample in progress;
  `--restart`, `--config`, and `--dump-config` work as for the other commands.
- From Python: `mopt_rs.OrientationSearch` (ask/tell), `mopt_rs.orientation_search` (with any
  calculator of the Python API), `mopt_rs.orient.compare_structures`, and
  `mopt_rs.orient.complex_parts` (`tests/test_orientsearch.py`).

Tests: `crates/mopt-rs/src/workflow/explore/`, `tests/cli_orientsearch.rs`,
`tests/test_orientsearch.py`.

## Conformation search (`confsearch`)

`confsearch` takes MultiOptPy's options (`-bf`, `-ms`, `-nl`, `-nr`, `-tgta`, `-st`, `-nost`,
`-pbc`, `-tabu`, `-alpha`) and every option of `optmain` for its optimizations, and writes
MultiOptPy's files with their names (`<stem>_EQ<k>.xyz`, `EQ_energy.dat`,
`visit_counts.dat`, `switch_conformer.log`, `no_update_count.log`, `search_atom_pairs.log`,
`tabu_search_settings.log`, `tabu_selection_log.log`, `REJECTED_CONFORMERS/`,
`reference_bond_table.dat`, `reference_bond_connectivity.log`, `EQ_summary.log`, `input.txt`)
in a run directory named as MultiOptPy names it (`<stem>_<bf>KJ_CS_<time>`). Its sampling
is MultiOptPy's by default (`-cs_opt method=afir`); see [searches.md](searches.md),
"Conformation search", for the other strategies. What differs, with the test that shows it:

### Errors of MultiOptPy's conformation search that are not repeated

- **The last atom pair** (`run_confsearch`): the loop ends when `len(atom_pair_list) <= i + 1`,
  so the last entry of the list is never sampled (with two entries, one sample is run).
  Here every entry is used.
  `tests/cli_confsearch.rs::afir_sampling_finds_the_gauche_conformer_and_writes_multioptpy_files`.
- **Units of the pair window** (`make_tgt_atom_pair`, `calc_normalized_distance_list`): the
  distances are in Angstrom and the covalent radii in Bohr, so the "normalized distance"
  is 1.89 times smaller than the ratio it stands for and the window `1 < d / (R_i + R_j) < 8`
  is in fact `1.89 < ratio < 15.1`: bonded and geminal pairs fall out, which is what the
  sampling needs. `-cs_opt pairs=multioptpy` reproduces the selection; `pairs=ratio` is
  the window in consistent units (which takes the bonded pairs), and `pairs=bonds:N` takes
  the pairs at least `N` bonds apart along the bond graph (the default with `N = 3`, below).
  `workflow::confsearch::afir::tests::the_pair_rules_choose_the_pairs_a_torsion_moves`.
- **The duplicate check** (`is_identical`, `check_identical`): once one known conformer is
  within 1e-4 Hartree, the sorted distance matrices of *all* known conformers are compared,
  so a conformer of another energy can make a new one a duplicate; and sorted distances
  ignore which atom is which and the handedness, with 0.1 Angstrom per distance rather than
  an RMSD. Here a structure is compared with the conformers within the energy threshold
  only (0.05 kcal/mol), by RMSD after superposition (0.125 Angstrom) and, where the
  rotational constants agree within 1 %, after the best renumbering of equivalent atoms (a
  rotamer, counted with its conformer), CREGEN's tests (`workflow::explore::ensemble`).
  `workflow::explore::ensemble::tests::duplicates_rotamers_and_new_conformers_are_told_apart`.
- **Weights that underflow** (`calc_penalized_boltzmann_distribution`,
  `get_index_from_distribution`): with large visit counts every `exp(-alpha N_i)` underflows,
  the sum is zero, and the probabilities are not numbers; `get_index_from_distribution` then
  raises (or returns `None` when the cumulative sum ends just below the random number).
  Here the weights are drawn from their logarithms shifted by the largest.
  `workflow::explore::priority_rules::tests::large_visit_counts_do_not_end_the_selection`.
- **Random numbers** (`random.shuffle`, `random.random`): not seeded, so no run can be
  repeated. Here one seeded generator (`-cs_opt seed=`, drawn and written to `settings.json`
  without it) gives the pairs, the choices, and the dynamics.
- **Continuing a run**: the folder of a run carries the time of its start, so the branch that
  reads `EQ_energy.dat` of an earlier run never runs. Here `--restart RUN_DIR` repeats the
  run from its journal.
  `tests/cli_confsearch.rs::a_restart_replays_the_journal_and_gives_the_same_ensemble`.

### Defaults that differ

- **Which minima are kept**: without `-pbc` MultiOptPy keeps every distinct minimum, whatever
  its bonds. The role of the command in mopt-rs is the conformers: minima whose bond graph
  is that of the optimized input up to a renumbering of the atoms (a proton moved between
  equivalent atoms keeps the molecule); the others go to `REJECTED_CONFORMERS/`.
  `-cs_opt mode=minima` keeps every minimum as MultiOptPy does, and `-pbc` compares the bonds
  in the numbering of the input as MultiOptPy's `-pbc` does.
  `tests/cli_confsearch.rs::the_minima_mode_keeps_what_the_conformer_mode_rejects`.
- **Convergence**: without `-tcc` or `-lcc` the optimizations of `confsearch` use the tight
  criteria. With the normal ones (MultiOptPy's) the flat torsion of gauche butane ends 2 to
  5 degrees and 0.02 to 0.06 kcal/mol apart from run to run (GFN-FF), and one conformer was
  counted three times; with the tight ones it is one ([searches.md](searches.md)).
- **Saddle points of symmetry**: an optimization can end exactly at a saddle point whose
  gradient vanishes by symmetry (anti butane with an eclipsed methyl group, 2.8 kcal/mol up,
  was kept by 4 of 6 runs of MultiOptPy's sampling); here every converged conformer is
  displaced by a seeded 0.02 Bohr per coordinate and optimized once more
  (`-cs_opt nudge=on`, the default), and the Hessian of every new conformer is checked: from
  a saddle point the optimization goes on along its imaginary mode (`-cs_opt verify=on`, the
  default). `tests/cli_confsearch.rs::the_optimized_input_is_checked_by_its_hessian`.
- **The start of a sample** when only one conformer is known: MultiOptPy starts from the input
  file as it was read; here from the optimized input (EQ0).
- **The atom pairs**: the pairs at least 3 bonds apart (`-cs_opt pairs=bonds:3`; the
  farthest pairs in a compact cluster, where every pair is within two bonds) instead of
  MultiOptPy's window (`pairs=multioptpy`). With the same seeds they found as many
  conformers or more in all nine runs of n-pentane, n-hexane, and Ace-Ala-NMe with GFN-FF
  (more in five; 4.0, 10.3, 10.3 against 3.7, 9.3, 9.0 on average) with about as many
  gradients ([searches.md](searches.md), "Measurements").
  `cli::confsearch_settings::tests::multioptpy_options_and_the_settings_of_mopt_rs_are_read`.

### Kept as in MultiOptPy

The AFIR force `-bf` on one pair per sample (both signs of every pair, shuffled), the
optimization without it from the biased structure, the termination (`-ms`; the `-nr`
lowest energies unchanged for more than `-nl` samples, counted once more than `-nr`
conformers are known; tested before the switch to the next start, while `-ms` ends the
search after it, so that `switch_conformer.log` has a line for the last sample), the
Boltzmann choice of the next start at `-st` with ten times the
temperature after every fifth sample (`i % 5 == 0`), the visit penalty `exp(-alpha N_i)` of
`-tabu`, `-nost`, the reference bonds of `-pbc` from the optimized input with a scale of 1.1
of the covalent radii, the stop file `end.txt` (with `STOP`), and the files.

### Added

- The iMTD-GC strategy (`-cs_opt method=imtd`), implemented from its papers: RMSD
  metadynamics (`pes::bias::RmsdMetadynamics`) run with the dynamics of `md`, multi-level
  optimization with energy windows, iterations from every lower conformer, the genetic
  crossing of conformers in Z-matrices (`workflow::explore::crossover`, `coord::zmatrix`), and
  short dynamics from the lowest conformers. `tests/cli_confsearch.rs::imtd_gc_finds_the_conformers_of_butane`.
- Conformers of transition states (`-tsconf`). `tests/cli_confsearch.rs::conformers_of_a_transition_state`.
- Searches by moves of the structure itself (`-cs_opt method=torsion | kick | lowmode`,
  `workflow::confsearch::moves`), each from its paper, and moves of a user by name (a Rust
  type that implements `ConformerMove`, or a Python callable given to
  `mopt_rs.confsearch.conformer_search`); the trials, the priority, and the termination are
  those of the AFIR sampling (`workflow::confsearch::search::Trials`, shared with it).
  `tests/cli_confsearch.rs::searches_by_moves_built_in_and_of_a_user`,
  `tests/test_confsearch.py::test_a_search_with_a_move_of_the_caller`.
- Sampling priorities of the exploration core for the choice of the next start
  (`-cs_opt priority=`: `boltzmann_heated` and `tabu_heated`, MultiOptPy's; `boltzmann`,
  `tabu`, `water_filling`, `water_filling_adaptive`, or a registered rule) and stop rules
  (`workflow::explore::stop`).
- The ensemble: `conformers.xyz` (sorted, with energies and Boltzmann weights),
  `rotamers.xyz`, `conformers.csv`, `ensemble.txt` (populations with the degeneracy of
  chiral conformers, conformational entropy, free energy), and with `-freq` a vibrational
  analysis of every conformer (`vibrations/EQ<k>/`).
- Every surface of `optmain`: `-mf`, `-es`, bias potentials, constraints, `-coord`, ONIOM
  (the conformers of the core, with microiterations in every optimization, step of the
  dynamics, and Hessian), molecules in a periodic cell (compared made whole; networks through the cell are refused),
  any calculator of the registry (ASE and the Python engines through the Python package).
  `tests/cli_confsearch.rs::oniom_with_gfn0_xtb_in_gfn_ff`,
  `tests/cli_confsearch.rs::a_cluster_in_a_periodic_box_and_a_network_that_is_refused`,
  `tests/test_confsearch.py::test_confsearch_with_an_ase_calculator`.
- Combinations that cannot make sense are refused with the reason: `-vc` (the cell would
  change with every optimization), `-ih` (one Hessian for many structures), `-mf bitss`,
  `-cmds`, `-pca`, `-km`, `-gi`, `-negeigval` (they describe one optimization), `-order`
  without `-tsconf`, objectives of plain variables (no bonds).
  `tests/cli_confsearch.rs::refused_combinations_say_why`.

Tests: `crates/mopt-rs/src/workflow/explore/`, `crates/mopt-rs/src/workflow/confsearch/`,
`crates/mopt-rs/src/coord/{graph,invariant_rmsd,zmatrix}.rs`,
`crates/mopt-rs/src/pes/bias/rmsd_metadynamics.rs`, `tests/cli_confsearch.rs`,
`tests/test_confsearch.py`.

## Relaxed scans (`relaxedscan`)

MultiOptPy's `Entrypoints/relaxed_scan.py:run_relaxedscan` takes `-scan KIND ATOMS
START,END ...`, makes `-nsample` values of every coordinate with `numpy.linspace`, and runs
`optimization.Optimize` once per value with `args.projection_constrain` set to `manual KIND
ATOMS VALUE` of every coordinate, each point from the optimized structure of the one before
(`-fo`: from the input). It writes `<stem>_scan_energy_profile_<date>.csv` (two header lines,
then the values, the energy, and the energy with the bias potentials) and
`<stem>_RelaxedScan_<k>.xyz`. mopt-rs keeps the options, the joint scan (every coordinate
moves together, MultiOptPy's meaning of several `-scan` groups), and the file names, and
differs where MultiOptPy loses information or writes outside its directory:

- **The constraints of the run hold.** MultiOptPy sets `args.projection_constrain = []`
  before adding the scan, so `-pc` of the command line is dropped without a word; mopt-rs
  keeps `-pc`, `-fix`, `-gfix`, and `-uc` and adds the scanned coordinates to them (a `-pc`
  without a value holds what the point starts from, which a scan carries from point to
  point). `tests/cli_relaxedscan.rs::the_constraints_of_the_run_hold_and_every_file_stays_in_the_scan`.
- **A point that did not converge** keeps the structure its energy belongs to. MultiOptPy
  records `final_energy` (the energy of the last evaluated structure) with `final_geometry`
  (the structure after the last step, never evaluated); mopt-rs records the last evaluated
  structure and its energy, and says in the log and `scan_points.csv` that the point did not
  converge. Same test.
- **Nothing outside the directory of the scan.** MultiOptPy copies the input to
  `<stem>_RelaxedScan_0.xyz` and every optimized structure to `<stem>_RelaxedScan_<k>.xyz` in
  the working directory as well, and every point's `Optimize` writes its run directory there;
  mopt-rs keeps all of them in `<stem>_RScan_<date>/` (the optimization of point `k` in
  `point_<k>/`). Same test.
- **`-scan` groups are checked.** MultiOptPy takes `int(len/3)` groups and drops what is left;
  mopt-rs refuses a count that is not a multiple of three, an unknown kind, and atoms that do
  not fit the kind. `cli::relaxedscan_settings::tests::targets_counts_and_options`.
- **Ranges that start with a negative number.** argparse takes `-180,180` for an unknown
  option and stops, so a dihedral cannot be scanned from -180 degrees; mopt-rs reads a list of
  numbers that begins with a minus sign as a value (for every command).
  `core_infra::arg_parser::tests::cli_parser_handles_negative_numbers_and_optional_flags`.
- **New in mopt-rs:** maps (`-grid`: every combination of the values, `-nsample N1,N2`), taken
  in snake order with each point started from its nearest optimized neighbour and the jumps
  between branches optimized again; rigid scans (`-rigid`); the kinds `expr` (any expression
  of `-ubias`), `var` (objectives of plain variables), and `resper` (ReSPer coordinates of
  reference structures, relaxed or rigid); the stationary points of the scanned surface
  (`stationary_points.csv`, `<stem>_ts_candidate_<n>.xyz`) and their optimization to
  transition states (`-scan_opt ts_optimize=on`); `scan_points.csv`, one trajectory per curve,
  figures (`relaxedscan_energy_profile`, `_map`, `_curves`, PNG and SVG), and a page of the scan
  (`relaxedscan_report.html`); `--restart` (the points done are kept, the interrupted one is
  replayed from its journal) and the `STOP` file of the scan; a point that fails is recorded
  and the scan goes on; every relaxed point starts from the last Hessian of the point it
  starts from (with ONIOM microiterations the Hessian of the core).
  `tests/cli_relaxedscan.rs`, `tests/cli_relaxedscan_maps.rs`,
  `tests/test_scan.py`.
- **ONIOM:** every point has microiterations: a scanned coordinate is renumbered to the core,
  an atom of the environment it names is moved with the core, and ReSPer coordinates are made
  of the atoms of the core; see the section on ONIOM microiterations in running.md. `tests/cli_oniom.rs::constraints_on_the_core_keep_the_microiterations`,
  `tests/cli_relaxedscan.rs::an_oniom_scan_keeps_its_microiterations`.

The rows of the energy profile of a map follow the order in which its points were taken (the
snake), as MultiOptPy's rows follow its loop; `scan_points.csv` lists every point by its grid
indices.

## Reaction-network mapper (`run_mapper`)

mopt-rs reads MultiOptPy's options and settings file of `run_mapper` (`mapper_settings`,
`step1_settings` to `step4_settings`, `top_n_candidates`) but does not take over the structure
of `Wrapper/mapper.py`. The network is a typed model (minima, transition states, and crossing
points as nodes with their level, state, and origin; edges with raw barriers and the IRC as
data), the exploration is an ask/tell driver whose parts are traits with named registries
(exploration method, priority, duplicate detection, stop rule, store), files and the report
are observers, and the settings are converted into typed values once, where they are read. See
[mapper.md](mapper.md).

### Behaviour that differs on purpose

- **One process, no AutoTS directories per job.** MultiOptPy runs its whole AutoTS workflow for
  every (minimum, pair) in a spawned process, in `runs/<iteration>/autots_workspace/`, and
  reads the results back from the files of AutoTS. mopt-rs runs the optimizations and IRCs of a
  job in the process of the command, on one calculator, and writes stage trajectories for each job
  (listed in `jobs/job_000003/trajectories.csv`), one log per job (`jobs/job_000003.log`), and the IRC of every
  new transition state (`paths/TS000007_irc.xyz`). The worker processes existed to isolate the
  `os.chdir` of AutoTS.
- **Several jobs at the same time in batches.** `--max_workers N` and `n_parallel` run up to
  `N` jobs at the same time in threads of the command (`-map_opt workers=N`). MultiOptPy takes
  every result as it comes and starts the next task at once, so that its network depends on
  which job ends first; mopt-rs by default chooses a batch of jobs from the network as it is and
  keeps their results in the order of their numbers when the batch has ended, so that the same
  settings and seed give the same network with any number of workers and after a restart
  (`batch_size`, the number of workers by default, decides the network).
  `-map_opt schedule=rolling` takes the results as they come, as MultiOptPy does. A job that
  fails or panics fails alone; MultiOptPy rebuilds its process pool when a worker dies.
- **`worker_timeout_s` stops the job, not the processes.** MultiOptPy kills every worker
  process when one job exceeds the limit (and marks the job `TIMEOUT`); mopt-rs stops that job at
  its next step and records it as failed, and the other jobs go on.
- **The NEB between the AFIR path and the guesses without the rest of AutoTS.** MultiOptPy's
  mapper runs its whole AutoTS workflow for a job; `afir_ts` runs the same four steps itself on
  the surface of the run (its journal, restarts, and counts): the AFIR push, the chain
  relaxation of the AFIR path (the two-point method `neb`, which is `nebmain`'s chain with
  `step2_settings` as its settings and nebmain's defaults, Hessian-based steps from the model
  Hessian `lindh2007d3` included, where MultiOptPy steps along the forces alone without
  `-modelhess`), the transition-state optimization of the maxima of the relaxed chain, and the
  IRC. `-map_opt method_relax=off` takes the guesses from the local maxima of the AFIR path
  itself, without the relaxation. `tests/cli_mapper.rs`.
- **The choice of the next job.** MultiOptPy keeps a heap of (minimum, pair) tasks ordered by
  `exp(-dE / k_B T)` and admits the pairs of a new minimum only with that probability (a
  minimum can thus be left out for good). mopt-rs draws the minimum among those that still have
  tasks with the Boltzmann weights at `temperature_K` (the default priority `boltzmann`; the
  other priorities of `workflow::explore` can be chosen) and takes its next task; every
  minimum keeps its tasks until they are run. The draws are seeded by the seed and the number
  of the job, so that a restarted run makes the same choices.
- **Duplicates.** MultiOptPy compares structures by an RMSD over rotations and permutations of
  equal elements, rejecting mirror images. mopt-rs uses the comparison of `workflow::explore`
  (the same bond graph, an RMSD after the best matching of equivalent atoms and identical
  molecules), with mirror images distinct as in MultiOptPy (`-map_opt mirror=true` merges
  them), and transition states are compared only with transition states, minima with minima.
- **Failed jobs are recorded.** MultiOptPy does not record a failed job and tries it again on
  `--resume`; mopt-rs records it (with the error, in the network and as a warning) and does not
  run it again: the calculations are deterministic, so a failure would repeat. A job stopped by
  a `STOP` file is not recorded and runs again on `--restart`, its evaluations from the journal.
- **The inputs and the ends of the IRCs are minimized with the same settings**
  (`step4_settings` with `step4b_opt_method`); MultiOptPy optimizes its seed with
  `step1_settings` without the AFIR keys and the IRC ends with those of AutoTS's step 4.
- **Several inputs** start one network; MultiOptPy takes one structure.
- **The surface is set once.** Steps 3 and 4 take the calculator settings of `step1_settings`
  where they give none, and steps whose surfaces differ are refused (a second level is the
  level-2 refinement).
- **`mapper_settings` is optional** and only the keys that are given count; MultiOptPy's
  `ReactionNetworkMapper` requires eight of its keys in the file although the command line
  gives them. Unknown keys are refused, as in MultiOptPy.
- **Excluded minima stay excluded.** MultiOptPy explores an excluded minimum when it is the only
  one; mopt-rs never explores it (the run ends when nothing else is left).
- **Results only in the run directory** (`--output_dir`), under fixed names; the transition
  states are numbered as nodes (`structures/TS000004.xyz`) instead of edges
  (`nodes/TS000000.xyz`).

### Options that are refused

- Because they only fit MultiOptPy's design: `-ma` (MultiOptPy replaces it in every job by the
  pair of the job), `--resume` (`--restart RUN_DIR` continues a run with the settings and the
  evaluations recorded in it), and `--log_file` (the log is `mopt.log` of the run directory).
- Not implemented yet: the parts named in mapper.md.

`--use_rcmc` and the `rcmc_*` settings work: they select the priority `kinetics`; see
[Kinetics](#kinetics-run_rcmc-run_kinetics-and-the-rcmc-priority) for what differs.

Tests: `workflow::mapper` (unit tests of the model, the JSON layout, the driver, the loop and
its schedules, the registries), `tests/cli_mapper.rs`, `tests/cli_mapper_parallel.rs` (several
jobs at the same time, stops and crashes at every phase), `tests/cli_mapper_kill.rs` (the
program killed and restarted), `tests/cli_mapper_methods.rs` (the methods `addf` and
`nanoreactor`).

## Kinetics (`run_rcmc`, `run_kinetics`, and the RCMC priority)

MultiOptPy's `Utils/rcmc.py` is one algorithm (its RCMC) used by `run_rcmc` and by the mapper's
`--use_rcmc`. In mopt-rs the kinetics are a module of parts (rate theories, corrections,
models, solvers, scores; [kinetics.md](kinetics.md)) in which RCMC is one solver. `run_rcmc`
keeps MultiOptPy's options and runs that code with the settings that reproduce MultiOptPy
(Eyring rates with MultiOptPy's clamp, the solver `legacy_rcmc`); `run_kinetics` has the other
choices, and the mapper's priority `kinetics` uses them. MultiOptPy has no counterpart of the
models and analyses beyond its RCMC (the rate theory `rrkm`, the models `egme`, `two_grain`,
`semi_markov`, the solver `kmc`, the corrections of recrossing, diffusion, curvature
tunnelling, and weak coupling, the trajectories from a saddle point, the refinement tasks):
nothing of MultiOptPy changes with them, and `run_rcmc` gives the same numbers as before.

### `legacy_rcmc` reproduces MultiOptPy's errors on purpose

The solver `legacy_rcmc` is MultiOptPy's `RCMCQueue.pop` as it is, for comparison with its
numbers (`tests/golden/rcmc_legacy.json`, made by running MultiOptPy's `rcmc.py`; populations
and the matrix `D` agree to 1e-9). It keeps the differences from the published method that
section 1.4 of [kinetics_beyond_tst.md](kinetics_beyond_tst.md) lists; the other solvers
(`propagate`, `rcmc`, `expm`, ...) do not have them:

1. The stop test comes after the contraction, so the step whose reference time exceeds the
   requested time is kept: a state that lives far longer than the requested time is taken as
   equilibrated.
2. With one state left (`rho(D) = 0`) the reference time is not updated, so the last step is
   never tested: a two-state network gives the equilibrium at any time.
3. The contracted equation `dp_T/dt = D p_T` is integrated from the superstate population
   without the factor `diag(1^T M)^-1`; superstates with many members gain and lose population
   too fast.
4. The fallback when RK4 would need more than 10^7 steps weights with `1 - K_TS K_SS^-1 1_S`
   instead of `1 - 1_S^T K_SS^-1 K_ST`.
5. The saddle-point energy is clamped by each minimum separately, which breaks detailed balance
   when the saddle point lies below the higher minimum (`--clamp legacy`; `run_kinetics` uses
   the consistent clamp `max(E_ts, E_a, E_b)` for both directions).
6. Without a start node (the mapper without `rcmc_start_node_id`) the populations start at the
   lowest node but the contracted equation from a uniform vector.
7. Inside MultiOptPy's mapper the electronic energies are always used.
8. `rcmc_K_contracted.csv` holds the Schur complement `D`, not the superstate matrix.
9. `K_SS` is factorized by LU after every step.

Measured on the three models of section 1.4 (`tests/cli_kinetics.rs`): the population left in
the start state is, exactly / `propagate` / `rcmc` / `legacy_rcmc`, 0.990 / 0.990 / 1.000 /
8e-10 (a start state with a lifetime of 100 s next to a fast pair, at 1 s), 0.670 / 0.670 /
1.000 / 3e-8 (decay at 1 1/s into ten degenerate conformers, at 0.4 s), and 1.000 / 1.000 /
1.000 / 0.119 (two states, 1e-9 1/s, at 1 s). The models differ in their details from those
of the table of section 1.4, so MultiOptPy's numbers differ from it; the pattern is the same.

### What differs on purpose

- **`run_rcmc`** reads mopt-rs's `reaction_network.json` (or the run directory of `run_mapper`)
  besides MultiOptPy's network file, prints MultiOptPy's table, and writes with `--output-dir`
  `rcmc_K_contracted.csv` as MultiOptPy does, with the other files of `run_kinetics`
  (`populations.csv`, `rates.csv`, `rcmc_superstates.json`, `diagnostics.json`, `mopt.log`,
  `settings.json`). `--log-level` changes what the log shows; the results are printed at every
  level, as in MultiOptPy.
- **`run_kinetics`** is new: the solver `propagate` by default (contraction, then the exact
  propagation of the superstates; it is the matrix exponential where the time scales are not
  separated), the consistent clamp, several output times, several start nodes, the rate theory
  `tst` and the corrections, `--validate`, the derivatives of the yields with respect to the
  energies of all nodes, `ln k` of all steps, and the temperature (`--sensitivity`, by dual
  numbers through the solver), energy errors, the exploration scores. Its `rcmc_K_contracted.csv` holds the superstate matrix
  `D diag(1^T M)^-1` (closure A) and says so in its header.
- **Bimolecular steps** (new; MultiOptPy's networks have unimolecular steps only): species and
  partner edges from `--annotations` or `--species`/`--partner`, pseudo-first order at
  `--concentration` (`k' = k_2 [S]`), the conversion of free energies of the gas to the
  standard concentration (`RT ln(c° R T / p)` per molecule, 1.894 kcal/mol at 298.15 K and
  1 atm; the standard state is that of `-conc`/`-press` of `freqmain`), optional treatments of
  the association entropy (`--association`), the apparent reaction orders `d ln p / d ln [S]`
  (`--sensitivity`), and the model `mass_action` with the solver `rosenbrock` (ROS2).
- **Rate theories of single edges** (new): Bell's tunnelling correction `bell` (the infinite
  and the truncated parabola), canonical variational TST `cvt` along a path of `freqmain
  -path`, and electron-transfer edges (Marcus non-adiabatic and adiabatic) with the EDEEL
  formulas, which moved from `analysis::electron_transfer` to `kinetics::rate::electron_transfer`
  (the old path re-exports them).
- **The mapper's `--use_rcmc`** selects the priority `kinetics` with the score `population`
  (MultiOptPy's queue: the minimum with the highest population at the reaction time). The
  solver is `kinetics_solver`, `propagate` by default; `kinetics_solver=legacy_rcmc` gives
  MultiOptPy's numbers (with `kinetics_clamp=legacy`). `rcmc_temperature_K`
  (`--rcmc_temperature`) defaults to the temperature of the run (MultiOptPy's command line
  sets 300 K whatever `temperature_K` is), `rcmc_reaction_time_s` (`--rcmc_time`) to 1 s,
  `rcmc_start_node_id` (`--rcmc_start_node`) to the node of the first input. A network without
  a transition state yet gives the lowest minimum, as in MultiOptPy. The energies are the
  electronic ones unless `kinetics_energy=free`. `priority=rcmc` is refused with a pointer to
  `priority=kinetics`.
- **The matrix exponential** of the reference solver `expm` multiplies non-negative numbers
  only (uniformization with squaring, the diagonal of a generator rebuilt from its columns):
  the scaling and squaring of a Pade approximant lost 5e-8 of the populations of a stiff
  network of the golden data at 100 s against a 90-digit reference (see kinetics.md).

Tests: `kinetics` (unit tests of the generator, the contraction, the matrix exponential, the
rate corrections, the quadrature), `tests/cli_kinetics.rs` (analytic solutions, the models of
section 1.4, MultiOptPy's numbers, the command lines, the mapper with `--use_rcmc`).
AutoTS records the resolved command settings of each dispatched stage, including
defaults and workflow overrides, in `effective_settings.txt` in its working
directory and in the Settings section of `autots_report.html`. For BITSS the
record includes the model function, its reference structure and the optimizer
used within its stages. The original settings remain available for restart.
