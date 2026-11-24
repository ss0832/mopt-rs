# Reaction networks (`run_mapper`)

`run_mapper` maps the reaction network of one or more structures: the minima (EQ), the
transition states (TS) between them, and the paths that connect them. It starts from the input
structures, optimizes each to a minimum, and then runs job after job: a job starts from a known
minimum with one of its tasks, finds transition states on the way out of the minimum, follows
their IRCs to the minima on both sides, and adds what is new to the network. The run ends when
no minimum has a task left, when `--max_iter` jobs have run, or when a further stop rule says
so.

```console
run_mapper ar7.xyz -cfg config.json
run_mapper "start_*.xyz" -cfg config.json --max_iter 100 --temperature 500
run_mapper --restart mapper_output
```

The command keeps MultiOptPy's options and its settings file (`-cfg config.json` with
`mapper_settings` and `step1_settings` to `step4_settings`), so that existing settings files
work. Its structure is new; the differences are listed in
[migration.md](migration_workflows.md#reaction-network-mapper-run_mapper). The rules of the network
(what a node and an edge are, when two are the same, how things enter it, what is built and
what is planned) are collected in [mapper_design.md](mapper_design.md).

## What a job does

The built-in way of exploring a minimum is `afir_ts`:

1. **The tasks of a minimum** are its pairs of atoms whose distance lies within
   `dist_lower_ang` to `dist_upper_ang` (Angstrom) and that are not bonded (farther apart than
   1.2 times the sum of their covalent radii), optionally only pairs of `active_atoms`. Each
   pair is pushed together by AFIR with the collision energy `afir_gamma_kJmol` and, with
   `include_negative_gamma`, also pulled apart. The pairs are put in an order drawn with the
   seed of the run and the number of the minimum, and at most `max_pairs` are kept.
2. **The AFIR path.** The minimum is optimized with the AFIR force between the two atoms
   (`step1_settings`). Every evaluated structure is a point of the path.
3. **Transition-state guesses.** The AFIR path is relaxed without the force by the chain
   relaxation of `nebmain`, as in the step 2 of AutoTS (`method_relax`, `neb` by default; the
   two-point method `neb` uses `step2_settings`, with `method_relax_KEY` taking precedence).
   Step 2 runs up to 30 NEB iterations, or 60 direct MaxFlux iterations with `method_relax=dmf`. The band stops earlier when the largest force on a
   moving image falls below 3e-4 Hartree/Bohr. Set `step2_settings.NSTEP` to change the limit.
   The points of the path, laid onto each other, are replaced by images 0.5 Angstrom apart along
   it, and every image takes Hessian-based steps from a model Hessian (`lindh2007d3`;
   `use_model_hessian` of `step2_settings` chooses another model, `calc_exact_hess` the
   calculator's, `"use_model_hessian": "none"` steps along the forces alone). The guesses are
   the maxima of the relaxed chain, the `top_n_candidates` (`-n`) highest. `method_relax=off`
   skips the relaxation and takes the local maxima of the energy without the force along the
   AFIR path itself (the approximate transition states of AFIR). Measured on LJ7 with the
   settings of `tests/cli_mapper.rs` (30 jobs, `priority=tabu`, an exact Hessian at every step
   of the transition-state optimizations): with the relaxation 6756 gradients and 2063 exact
   Hessians (119 s), without it 4001 gradients and 3211 exact Hessians (94 s); both runs found
   the four minima, with five and six transition states. The relaxation costs gradients and
   saves Hessians, since the optimizations start from better guesses; with a calculator whose
   Hessian costs many gradients it is the cheaper of the two.
4. **Transition states.** Each guess is optimized to a first-order saddle point
   (`step3_settings`, `-order 1`, exact Hessian every five steps by default). A guess that converges to a transition state known already
   is not followed again.
5. **IRC.** The Hessian at the transition state is checked (one imaginary mode, or the
   stationary point is reported and left out), and the IRC runs in both directions
   (`intrinsic_reaction_coordinates` of `step4_settings`; `0.1 300 lqa` without it, the IRC of
   `confsearch`). Both ends are optimized to minima (`step4b_opt_method`, `rsirfo_block_fsb`
   without it). The vibrations of the transition state come from the Hessian of that check.
   With `verify_minima` enabled, the optimized endpoint is classified by its vibrations.
   A single imaginary mode with magnitude at least `irc_end_threshold_cm1` (default 50 cm^-1)
   triggers a TS–TS link search and an IRC from the linked transition state. A single weaker
   imaginary mode is saved on the EQ with a warning. Two or more imaginary modes, or an
   endpoint that fails to converge, leave that side unknown. The mapper evaluates both sides
   independently and draws the TS beside any verified EQ. It saves the incomplete connection
   in `open_ts_edges` and excludes it from kinetics. Enable `higher_saddles` for optional
   higher-order saddle searches. Mode kill during minimization is opt-in.
   The IRC computes the exact Hessian at every tenth point
   (`irc_hessians`; `calc_exact_hess` of `step4_settings` goes first).
   - **A possible bifurcation.** Where the curvature across the path at one of those exact
     Hessians is below `-bifurcation_threshold_cm1` (50 cm^-1), the valley of the path has
     turned into a ridge (a valley-ridge inflection lies before that point). The edge records
     it (`irc.possible_bifurcations`: side, arc lengths, wavenumber) and the network warns. It
     is not confirmed: the IRC itself goes on to one end, and every tenth point can miss a
     short ridge. The threshold was measured with GFN1-xTB and the exact Hessian at every
     point of IRCs that do not bifurcate: the lowest curvature across the path was -0.0 cm^-1
     for the HCN isomerization (86 points), -17.4 cm^-1 for the Diels–Alder reaction of
     butadiene and ethylene (159 points, 25 of them below -10 cm^-1), and -0.0 cm^-1 for the
     Cope rearrangement of 1,5-hexadiene (258 points); the bifurcating model surface of
     `tests/cli_mapper_bifurcation.rs` goes below -100 cm^-1. 50 cm^-1 keeps about three times
     the largest value of the reactions that do not bifurcate.
   - **A TS–TS link.** An optimized endpoint with an imaginary vibration at or above
     `irc_end_threshold_cm1` is refined as a candidate TS2. A near stop along the path (a local
     minimum of the energy slope below a hundredth of its largest value) is also checked
     for a single negative curvature. In this case the IRC may have left a ridge along its
     unstable direction and the stored side is cut at TS2.
     TS2 is optimized at order 1, its IRC is run and its ends minimized, and the edge of TS1
     ends at TS2 instead of a minimum (`EQ - TS1 - TS2` with the two products of TS2). Before
     that, a shoulder of the side (a local minimum of the slope of the energy) is minimized
     once; a minimum above TS2 is a shallow one the path went past, and it is the end instead.
     The kinetics divides the flux of TS1 among the products beyond TS2 (equally, and says
     that the split is assumed); the report draws TS1 from its start to each of them.
6. **The network.** A minimum or a transition state is the same as a known one when its energy
   lies within `energy_tolerance_kcalmol` and the RMSD after the best matching of equivalent
   atoms is below `rmsd_threshold` (Angstrom); the bond graphs must agree, and mirror images
   are different structures. What is new is added: the two minima, the transition-state node,
   and the edge between the minima with the barriers from both sides, the imaginary frequency,
   and the IRC (the energy and the arc length of every point; the structures in
   `paths/TS000007_irc.xyz`).

### Other ways of exploring a minimum: `addf` and `nanoreactor`

`-map_opt method=addf` and `-map_opt method=nanoreactor` explore a minimum in other ways; steps
4 to 6 above are the same for every method (each transition-state guess is optimized, a known
transition state is not followed again, the IRC of each new one runs and its ends are
optimized). Their settings are `method_KEY=VALUE` (in `-map_opt` or `mapper_settings`), and
keys that a method does not know are refused before the run starts.

- **`addf`**: a minimum has one task, the anharmonic downward distortion following from it
  (Ohno, Maeda, Chem. Phys. Lett. 2004, 384, 277; the method `addf` of the two-point methods,
  [reaction_paths.md](reaction_paths.md)). Its paths end at the transition states they pass over, at
  dissociation channels (the structure falls apart into more fragments than the minimum), or
  after the largest number of spheres; the estimates, the lowest first, are the guesses. The
  settings are the keys of `addf` with the prefix: `method_paths` (the largest ADDs followed,
  5), `method_spheres` (300), `method_step`, `method_initial`, `method_angle`,
  `method_gradient`, `method_sphere_iterations`, `method_sphere_gradient`. The test for a
  dissociation channel has two settings of the mapper:
  - `method_dissociation`: the bond scale of the test (2.0, from twice the sums of the
    covalent radii, the definition the documentation of the method gives for growing spheres);
  - `method_dissociation_atoms`: limits the test to fragments that hold one of the listed atoms
    (`1,4-6`).

  With 1.5, the scale of `ieipmain`, a stretched C-H bond of acetaldehyde already counted as
  a lost hydrogen atom, and only the methyl rotation was found. Measured from acetaldehyde with
  GFN0-xTB, 8 jobs each (`method_paths=6 method_spheres=80`):

  | `method_dissociation` | Minima | Transition states | Gradients | Failed jobs |
  |---|---|---|---|---|
  | 1.5 | 1 | 1 | 2,452 | 0 |
  | 2.0 | 11 | 13 | 15,886 | 1 |
  | 2.5 | 11 | 14 | 17,416 | 2 |
  | 3.0 | 12 | 15 | 17,488 | 1 |

  A failed job starts from a minimum whose Hessian has a small negative eigenvalue: ADDF does
  not start there, the job is recorded as failed, and the run goes on.
- **`nanoreactor`**: a minimum has `method_runs` tasks (2), molecular dynamics in the boundary
  potential of the nanoreactor (Wang et al., Nat. Chem. 2014, 6, 1044: a wide sphere and, for
  part of every period, a narrow one that presses the atoms together; the bias `-nrp`). A
  trajectory leaves a basin when the bonds of `method_persist` frames in a row (3) differ from
  those of its minimum (`method_detect=bonds`; the first of them is optimized to the new
  minimum, one that falls back to the old bonds is no reaction), or when a frame optimized
  every `method_quench_every` frames (10) is another minimum (`quench`); `auto`, the default,
  takes `bonds` where the minimum has bonds and `quench` where it has none (atomic clusters).
  The first `method_events` reactions of a trajectory (3) are kept. The path the dynamics
  took is relaxed: the old minimum, the frames around the reaction, and the new minimum are the
  first path of the two-point method `method_path` (the chain relaxation `neb` by default, the
  step 2 of AutoTS; its settings `method_path_KEY`), laid one onto the other and replaced by
  images 0.5 Angstrom apart, as the reactive events of the nanoreactor are refined in Wang,
  McGibbon, Pande, Martinez, J. Chem. Theory Comput. 2016, 12, 638. Every maximum of the
  relaxed path is a guess (a reaction of several steps has several). `method_segment=off`
  starts the method from the two minima alone (then two-point methods such as `dsafir` can
  be used).

| Key of `nanoreactor` | Default | Meaning |
|---|---|---|
| `method_runs` | 2 | trajectories per minimum (seeds from the seed of the run, the minimum, and the trajectory) |
| `method_steps` | 4000 | steps of a trajectory |
| `method_time_step` | 0.5 | time step (fs) |
| `method_temperature` | 2000 | temperature of the thermostat (K; Bussi) |
| `method_frame_every` | 10 | every how many steps a frame is kept |
| `method_r1`, `method_r2` | from the minimum | radii of the wide and the narrow sphere (Angstrom): 2 Angstrom outside the farthest atom from the centre, and 0.45 of that atom's distance |
| `method_k1`, `method_k2` | 1.0, 5 | force constants of the walls (kcal/mol/Angstrom^2 per amu; the paper's wide one, and ten times its narrow one) |
| `method_period`, `method_wide` | 250, 0.75 | period of the walls (fs) and the share of it the wide sphere holds (the paper's share, 1.5 of 2 ps) |
| `method_detect`, `method_persist`, `method_quench_every` | `auto`, 3, 10 | how a reaction is seen |
| `method_events` | 3 | reactions kept per trajectory |
| `method_path`, `method_path_KEY` | `neb` | the two-point method between the minima of a reaction and its settings |
| `method_segment`, `method_segment_frames` | on, 4 | the frames around the reaction (4 before and after the first frame of the new basin; with quenches, those since the last quench of the old basin) as the first path; off: the two minima alone |
| `method_replicas` | 1 | replicas of a trajectory exchanged along a ladder (1: no replica exchange; below) |
| `method_exchange` | `temperature` | what the replicas differ in: `temperature` (of the thermostat), or `walls` (the force constant of the narrow sphere, at one temperature) |
| `method_replica_top` | twice the temperature, 2 | the top of the ladder: the highest temperature (K), or the factor of `k2` of the strongest narrow sphere |
| `method_exchange_every` | 100 | time between two exchanges (fs) |

**Replica exchange** (`method_replicas=3` or more): a trajectory runs as that many replicas on
a ladder, geometric from the temperature to `method_replica_top` (`method_exchange=temperature`)
or from `k2` to `method_replica_top` times it (`method_exchange=walls`, every replica at the
temperature). The replicas run in segments of `method_exchange_every` fs, each going on from
the structure and the velocities where its last segment ended, with the clock of the walls going
on. After every segment neighbouring rungs (the even pairs and the odd ones by turns) exchange
with the probability of Metropolis: of the energies with the walls and the two temperatures
for temperatures (the velocities scaled to the new temperature; Sugita and Okamoto 1999), of
the walls alone for walls (the surface cancels; Sugita, Kitao, and Okamoto 2000). Each
replica's own continuous trajectory is searched for reactions. The log of the job names the
rungs and the exchanges accepted between each two
(`3 replicas of trajectory 1 on 30 K, 52 K, 90 K (temperatures); exchanges accepted between
neighbouring rungs: 2/2, 1/2`); `tests/cli_mapper_methods.rs` runs both kinds on LJ7.

The paper's spheres of 14 and 8 Angstrom and its period of 2 ps are for mixtures of many
molecules, and runs of a few thousand steps would not reach the narrow phase. The defaults here
come from three acetylene molecules (GFN0-xTB, two trajectories of 4000 steps of 0.5 fs at
2000 K per minimum, at most 8 jobs; the two minima of each reaction joined by DS-AFIR in this
first measurement):

| Narrow sphere | Minima | Transition states | Gradients |
|---|---|---|---|
| `k2` 0.5 (the paper's), 0.8 of the farthest atom | 1 | 0 | 8,014 |
| `k2` 5, 0.8 of the farthest atom | 11 | 6 | 97,281 |
| `k2` 20, 0.8 of the farthest atom | 16 | 11 | 82,952 |
| `k2` 5, 1.2 Angstrom (0.45 of the farthest atom) | 23 | 15 | 58,752 |

With the paper's narrow force constant, the hot molecules spread to 4 to 6 Angstrom from the
centre and never came closer than 2.4 Angstrom (C to C), so no reaction happened.

The path step, measured the same way with the defaults above (8 jobs each):

| Path between the minima of a reaction | Minima | Transition states | Gradients | Transition states per 10,000 gradients |
|---|---|---|---|---|
| `neb` from the frames of the trajectory (the default) | 37 | 23 | 101,943 | 2.3 |
| `dsafir` from the two minima (`method_segment=off method_path=dsafir`) | 22 | 12 | 64,629 | 1.9 |

Relaxing the path the dynamics took found about twice the transition states, each at a lower
cost. Pressing
harder and narrower made them react. With `nanoreactor` a trajectory of a periodic structure
is not run: the job ends with a warning.

In that measurement a trajectory blew up: within 10 steps an atom was shot 10^7 Angstrom away,
most likely from two atoms pressed together. A structure from after it was taken as a minimum,
and DS-AFIR then ran its 2000 steps from that structure. Since then the mapper ends a
trajectory at a step that moves an atom more than 1 Bohr (more than ten times what a
hydrogen atom moves in a step of 0.5 fs at 2000 K) and uses the frames before it, with a
warning. Its searches also refuse structures with an atom more than 200 Bohr from their
centre (the job fails, with the reason). Every evaluated structure of a search and every kept frame of a trajectory go to the
trajectory of the job. The files and the full log of a search (every point with its structure)
go to the directory of the job, `jobs/job_000003/`. `tests/cli_mapper_methods.rs` runs both
methods on the argon cluster of seven atoms. `addf` found 3 of the 4 minima and 4 transition
states in 3 jobs (`method_paths=4 method_spheres=60`). `nanoreactor` found 2 minima and 1
transition state in 2 jobs (one trajectory of 1500 steps of 4 fs at 40 K, quenched every 5
frames). Both continue to the same network after a stop.

### Towards a goal: `priority=rrt` and `method=connect`

With `-map_opt goal=FILE` the run has a goal structure: the first frame of an xyz file (a
variables file for plain variables) with the atoms, their order, the charge and the
multiplicity of the inputs. It is optimized and verified like the inputs, enters the network
as an input marked `goal` (`"goal": true` in `reaction_network.json`, the column `goal` of the
database), and is never explored. The exploration ends when the goal is joined to a start by
edges of the network (`goal_stop=off` goes on); the log and `run.goal` of the network file say
when. With `goal_match=bonds` any minimum with the bond graph of the goal (another conformer
of it, which the identity test with its RMSD tells apart) ends the run when it is joined to a
start.

- **Goal-directed search** (`priority=rrt`, the priority when a goal is given): after Nakao,
  Harabuchi, Maeda, Tsuda (Phys. Chem. Chem. Phys. 2022, doi:10.1039/d2cp01079h), the rapidly
  exploring random tree with AFIR. Jobs with even numbers choose a group of minima of one
  connection pattern (bonds up to 1.1 times the sums of the covalent radii) with equal
  probability and then a member; jobs with odd numbers choose a group with a probability
  proportional to `exp(c s)` (`s` the best similarity of its members to the goal) and then a
  member by its own similarity (`rrt_goal_weight` = `c`, 1). The similarity compares the
  distances of every atom to the other atoms of each element, matched greedily
  (`rrt_locality`, 20 per Angstrom); it lies between 0 and the number of atoms. A new minimum
  is explored only with the probability `min(1, exp(-(E_PT - E_X) / kT))`, `E_PT` the highest
  transition state on the way of its job from the minimum `X` it started at, at
  `accept_temperature` (7500 K, the middle of the paper's 5000 to 10000 K); otherwise it is
  kept, its "not explored" column gives `p`, and a later job that reaches it again draws
  again; when no accepted minimum has a task left, the refused one with the highest `p` is
  explored (the filter orders the search, it never ends it). `accept_temperature` turns the filter on for
  any priority. The exploration method stays `afir_ts` (`addf` for plain variables).
- **Missing connections** (`method=connect`): after Carr, Trygubenko, Wales (J. Chem. Phys.
  2005, 122, 234903). The known minima form a complete graph: a pair joined by a transition
  state weighs 0, a pair tried `method_attempts` times (2) is not tried again, and any other
  pair weighs `D^2`, its distance after the best superposition. Dijkstra's method gives the
  path of least weight from the starts to the goal; every pair of it that is not joined yet
  (a gap) is a job: the two-point method `method_path` (`neb`; `method_path_KEY` its settings)
  between the two, every maximum of its path followed to a transition state and its IRC. A
  second attempt of a pair runs with twice the images. The path is computed again after
  every job.

```bash
run_mapper ar7.xyz -cfg config.json -map_opt goal=capped_octahedron.xyz
run_mapper ar7.xyz -cfg config.json -map_opt goal=capped_octahedron.xyz method=connect
```

**Measurements.** On the chain surface of `tests/cli_mapper_goal.rs` (an `expr` objective of
two variables: seven Gaussian wells 0.05 Hartree deep along y = 0 with barriers of about 0.02
Hartree, the start at x = 0, the goal at x = 6, and a deeper dead-end valley of four wells,
0.072 to 0.078 Hartree, branching off the second well; 11 minima and 11 transition states in
all), with ADDF (one job per minimum), ten seeds each; the shortest way needs 6 jobs:

| Setting | Jobs until the goal is joined (mean, range) | Gradients (mean) |
|---|---|---|
| `priority=rrt`, `accept_temperature=2500` | 7.0 (7 to 7) | 3292 |
| `priority=rrt`, `accept_temperature=7500` (the default) | 7.6 (7 to 9) | 3652 |
| `priority=rrt`, `accept_temperature=15000` | 8.3 (7 to 10) | 4010 |
| `priority=rrt` without the filter | 8.8 (6 to 10) | 4211 |
| `priority=boltzmann` | 10.0 (10 to 10) | 4900 |
| `method=connect`, `method_attempts=1` | 2.0 | 478 |

The Boltzmann priority explores the deep side valley first and finds the goal only when the
whole surface is known. The missing connections run one nudged elastic band from the start
to the goal, whose maxima gave five of the six transition states of the chain and the minima
between them, and a second one for the gap left between two of those minima. On LJ7 (`tests/cli_mapper_goal.rs`) from the pentagonal bipyramid to the
capped octahedron, `method=connect` took 1 job (319 gradients, 113 Hessians) and
`priority=rrt` with `afir_ts` 2 jobs (744 gradients, 340 Hessians); runs stopped at two and
three points and restarted gave the same networks.

On the urea synthesis of the paper of the goal-directed search (Nakao et al. 2022; NH4+ . OCN- . 2 H2O to urea . 2 H2O, 14 atoms; the goal by
its bonds, `goal_match=bonds`) with the built-in GFN1-xTB, `afir_ts` and the two best guesses
of a path (`top_n_candidates` 2), 40 jobs each:

| Setting | Seed | Minima (the goal among them) | Minima joined to the start | Minima the filter refused | Gradients | Hessians |
|---|---|---|---|---|---|---|
| `priority=rrt` without the filter | 1 | 31 | 17 | - | 127181 | 6028 |
| `priority=rrt`, `accept_temperature=7500` (the default) | 1 | 28 | 17 | 3 | 127021 | 6492 |
| `priority=boltzmann` | 1 | 17 | 14 | - | 137239 | 6589 |
| `priority=rrt` without the filter | 2 | 35 | 27 | - | 114896 | 5446 |
| `priority=rrt`, `accept_temperature=7500` (the default) | 2 | 33 | 9 | 1 | 106908 | 4974 |
| `priority=boltzmann` | 2 | 26 | 18 | - | 108989 | 6517 |
| `method=connect` | 1, 2 | 33 | 9 (and 10 to the goal) | - | 101907 | 9860 |

No run joined the start to the goal within its 40 jobs, and the filter refused one to three
minima of about thirty: a budget of this size does not show whether the filter helps on a
molecule, so the default of `accept_temperature` stays at the middle of the range of the paper
and is not a measured value. `method=connect` draws no random numbers (both seeds gave the
same network) and grows the network from both ends.

With 1000 jobs each (seed 1, six jobs at a time; the paper reports searches of this size), the
goal was still not joined to the start:

| Setting | Minima | Transition states | Species (by the SMILES strings of the minima) | Gradients | Hessians |
|---|---|---|---|---|---|
| `priority=rrt`, `accept_temperature=7500` (the default) | 260 | 581 | 8: carbamic acid + NH3 + H2O (113 conformers), HNCO + NH3 + 2 H2O (97), its iminol form + NH3 + H2O (38), ammonium cyanate + 2 H2O (4), and four of one to three minima | 2,615,019 | 112,877 |
| `priority=rrt` without the filter | 184 | 493 | 5: HNCO + NH3 + 2 H2O (100 conformers), cyanic acid HOCN + NH3 + 2 H2O (45), ammonium cyanate + 2 H2O (28), one of radicals (10), the goal (1) | 3,057,531 | 110,414 |

In both runs most transition states join two conformers of one species (543 of 581 with the
filter, 412 of 493 without it): with two water molecules and ammonia around a small molecule,
most jobs find another arrangement of the hydrogen bonds and not a reaction. The goal (urea + 2 H2O) is the one minimum of its
species in both networks, the input of the goal itself. So on this system a thousand jobs of
`afir_ts` from random pairs of atoms do not reach urea with the built-in GFN1-xTB, with the
filter or without it; what would change that is not the priority but the tasks (pushes
between the reacting atoms rather than all pairs) and a network of species rather than of
conformers. The filter changes which species are reached (carbamic acid with it, cyanic acid
without it), which one seed does not make a rule.

### Saddle points of higher order (experimental)

With `-map_opt higher_saddles=on` the mapper keeps the stationary points with k ≥ 2 negative
curvatures that it meets (a TS search that ends at one, an IRC that runs into one) as nodes
`2SP0`, `2SP1`, ..., `3SP0`, ... (up to `max_saddle_order`, 2). Each is optimized at its order
(the settings of `step3_settings` with `saddle_order` k), its index checked by the Hessian,
and descended from: every unstable mode both ways (0.1 amu^(1/2) Bohr, mass-weighted), each
start optimized at order k - 1 following the other unstable modes, down to transition states
(followed as any, with their IRCs) and minima. The descents are `saddle_edges` of the network
files and the database, and a table of the report. The transition state whose IRC ran into
the saddle point ends at it; the kinetics divides its flux among what lies below.

Without the setting (the default) such a point is still kept as a node, but its links are not
followed: a TS search that ends at one goes on by mode kill (the point displaced along its
unstable modes but the lowest and optimized at order 1 again) to a transition state, and an
IRC that runs into one keeps it and minimizes the end of the side as before.
Mode kill follows order-one HiSD from the displaced structure before local TS refinement.
On a variable cell the dynamics uses coupled atom and lattice coordinates and their masses.
For an index-two starting saddle it tries both signs of the surplus mode, then exchanges
the retained and surplus modes. It checks the coupled index again after optimization and rejects a return within the
original geometry’s convergence tolerances. Further attempts increase the displacement after trying the available branches. The job log gives the seed displacement,
final index and displacement from the original point for each converged attempt.

One criterion counts the imaginary modes of every point of the mapper (minima, transition
states, saddle points of higher order, the mode kill of its optimizations, the start of an
IRC): an eigenvalue of the mass-weighted Hessian below minus 1e-8 Hartree/(Bohr^2 amu),
MultiOptPy's `saddle_check`, about 0.51 cm^-1, with the rigid motions projected out in the
mass-weighted space where the surface does not depend on them (plain variables have unit
masses, a variable cell the masses of the variables of its IRC). The saddle tables and node details use the modes counted by this criterion.
In a variable cell these are effective wavenumbers in the coupled atom and strain metric.
The separate fixed-cell atomic spectrum supplies vibrational thermochemistry. Legacy periodic
checkpoints without saved index modes show the order with unavailable wavenumbers, unless an
IRC start recorded the transition wavenumber (`docs/source/mapper_design.md`, "Higher-order saddles").

`-map_opt upward=on` adds upward searches to the tasks of every minimum: from each of its
transition states, displaced along its softest stable modes (`upward_modes`, 2; both ways),
high-index saddle dynamics of index 2 (the shrinking-dimer form of `path::hisd`, gradients
only) climbs to a loose threshold and hands the point to the optimization at order 2; a
saddle point found so is descended from as above (`docs/source/high_index_saddles.md`).

`-map_opt confirm_bifurcations=on` tries every possible bifurcation of an IRC (the warning
"may bifurcate"): at that point the Hessian is computed again with the direction of the path
taken out, the point is displaced both ways along the lowest mode across the path, and both
are minimized. Minima other than the end of the side are recorded on the possible
bifurcation (`confirmed_ends`), with a warning, and the kinetics divides the flux of that
side among them.

On the surface `V = (x^2 - 1)^2 + cos(pi x) (y^2 + z^2) + y^4 + z^4 + 0.5 y^2 z^2` of
`tests/cli_mapper_saddles.rs` the IRC of the transition state at the origin runs into the two
index-2 saddle points (+-1, 0, 0); with `higher_saddles=on` the network has both `2SP` nodes,
their four descents to the eight transition states of the planes x = +-1, and the eight
minima, and runs stopped during the descents continue to the same network; with `upward=on`
and ADDF the upward search climbs from a transition state of the plane x = 1 to the `2SP`
at (1, 0, 0). On the bifurcating model of `tests/cli_mapper_bifurcation.rs` the confirmation
finds both products of the second transition state. On LJ7 a 16-job run with `upward=on`
found 8 `2SP` (among them the index-2 point at -14.723 epsilon of Quapp and Bofill), all four
minima, and 8 transition states; 12 of the 16 descents from the `2SP` reached a transition
state on both sides.

The report lists the saddle points of higher order in a table and draws them in the Graph and
the Map as squares above the saddle points their descents reach, joined by dashed lines. A
click on a square (or on its row of the table) shows the saddle point in the panel of the
chosen node, also over the whole screen: its order, energies, imaginary wavenumbers, the ends
of the descents along each of its unstable modes (a click on one goes on to it) and its
structure in a viewer, which is in the page as those of the other nodes. For
every saddle point (transition states and `kSP`) the tables and the details give the number
of imaginary modes and each imaginary wavenumber (`2: 512.3i, 120.4i`).

### Step 1 by another two-point method: `method_step1`

The first path of a pair of atoms of `afir_ts` comes from the AFIR push. `-map_opt
method_step1=NAME` takes it from a method of the registry of two-point methods that goes on
from the minimum along the pair, without a push and without a second structure
(`method_step1_KEY=VALUE` its settings):

| `method_step1` | What a job does |
|---|---|
| `afir` (the default) | the AFIR push, its path relaxed by `method_relax`, the maxima of the relaxed chain followed |
| `se_gsm` | the single-ended growing string from the minimum, driven by the pair (`ADD a b` for a push, `BREAK a b` for a pull with `--negative_gamma`); its maxima are followed |
| `gnt` | the growing Newton trajectory from the minimum, which shortens the distance of the pair (it has no pull); its maxima are followed |

`method_relax` does not run after such a method (it is the path method itself), and
`step2_settings` are not used. `neb` and `dmf` are refused as methods of step 1: they relax a
path that exists, and are the methods of `method_relax`. `addf` is refused (it explores a
minimum as a whole: `method=addf`), and so are the methods that need a second structure
(`gsm`, `fsm`, `dsafir`, ...): they run between the known minima and a structure that is
given, `method=connect` with `goal=FILE` and `method_path=NAME`
([above](#towards-a-goal-priorityrrt-and-methodconnect)).

Measured (`tests/cli_mapper_extras.rs`): for HCN with GFN0-xTB, the one job of the pair H, N
gave with the AFIR push and its chain no transition state (13 gradients), and with `se_gsm` the transition state to
HNC and HNC itself (2321 gradients, 40 Hessians). On the cluster of seven argon atoms, `gnt`
for two pairs gave 2 minima and 2 transition states (471 gradients, 375 Hessians).

### Conformers of the transition states: `ts_conformers`

`-map_opt ts_conformers=on` adds one job for every transition state of the network: the
conformer search of `confsearch -tsconf` from it, with its saddle direction held (the reaction
coordinate from its imaginary mode and its reactive bonds; [searches.md](searches.md),
"Conformers of a transition state").
Every saddle point the search finds is followed like any transition-state guess (optimized,
its IRC, the ends minimized), so the conformers of a transition state are transition states
of the network between the conformers of the minima. The settings of the search are the block
`ts_conformer_settings` of the settings file (the options of `confsearch`: `max_samples`,
`target_atoms`, `confsearch_options`, ...); it runs on the surface of `step1_settings` with
the seed of the run. A transition state that such a job found is not searched again.

```json
{
  "mapper_settings": {"ts_conformers": true},
  "step1_settings": {"othersoft": "gfn0xtb"},
  "ts_conformer_settings": {"max_samples": 10, "confsearch_options": ["pairs=ratio"]}
}
```

A search is many times a job of `afir_ts` (it samples, minimizes with the constraint, and
optimizes saddle points), so it is off by default. It is refused for plain variables, with
`-remap`, `-single_points`, and in a run of several states.

2-ethyl-1,5-hexadiene with GFN0-xTB (`tests/cli_mapper_conformers.rs`): from the chair
transition state of the Cope rearrangement with the ethyl group in its higher position, the
one search turned the group to a transition state 0.78 kcal/mol lower, with its own IRC; the
run took 786 gradients and 60 Hessians, and the search job about 2.3 times as long as the job
that followed the first guess.

### An entrance channel: `entrance_structures`

For a bimolecular reaction the input is the two (or more) molecules in one structure, and
`-map_opt entrance_structures=N` makes N structures of their dissociation channel: every
molecule kept as it is, turned by a rotation drawn uniformly, and placed in a direction drawn
uniformly at the distance at which its nearest atom to the others is at `entrance_contact`
(1.0) times the sum of the two van der Waals radii. They are further inputs of the run
(`FILE (entrance structure k)`), minimized like any input: complexes with the barrierless edge
to the channel, or whatever they fall to. The same seed gives the same structures.

The network records the entrance (`entrance`: the channel, the number of structures, the
inputs they are), and its kinetics starts with the whole population on the channel: a channel
is one node whatever the orientation of its molecules, so the equal shares of N structures
of one channel add up to the population of that node. `--start-node` and `--initial` of
`run_kinetics` say otherwise.

Two water molecules with GFN-FF (`tests/cli_mapper_extras.rs`): three structures of the
channel fell to complexes bound by 6.40 kcal/mol.

### Path peaks

A job whose path gave guesses, none of which led to a transition state, has found something
all the same: the highest point of that path, between the minima on its two sides. With
`path_peaks=failed` (the default) it is saved as a path peak: the nearest minima of the path
energy on both sides of the peak are minimized (minima of the network like any other), and
the record goes to `path_peaks` of the network file, `path_peaks.csv`, the structure
`structures/PP000000.xyz` and the path between the two minima `paths/PP000000_path.xyz`.
`path_peaks=all` saves the peak of every path, also of one that gave its transition state;
`off` saves none.

**A path peak is saved and is not in the network**: it is no node and no edge, the figures of
the report do not draw it (the page says how many were saved), the kinetics does not use it,
and it joins nothing. It is no stationary point and nothing was verified about it; it is
there for a later look or a later search from its structure. `path_peaks.csv` says for every
record which transition state joins its two ends by now, if one does. The single points of a
network (`-single_points`) leave the path peaks alone: they stay in the files of the first
run.

### Limits of the search: `max_bond_changes`, `max_species_changes`

By default every new minimum is explored. `-map_opt max_bond_changes=N` leaves out the minima
whose bonds changed more than N times on the way from their input, and `max_species_changes=N`
those whose chemical species (the SMILES string, so R/S and E/Z count) changed more than N
times. The way is the chain of the minima the jobs started from; a step counts when the bond
graph (the string) of the minimum found is not that of the minimum the job started at. Such
a minimum is kept as the end of its reaction and not explored (its reason says why). With
`0` the search stays among the conformers of the inputs: what they react to is found, and
nothing is explored from there. `inf` (or no setting) is no limit.

### The curvature across an IRC, and the scan of a dissociation

The IRC of the mapper computes an exact Hessian at every tenth point (`irc_hessians`; `irc_hessian_interval=N` sets another interval).
The lowest curvature across the path at each of them is kept with the path
(`transverse_cm1` of a side in the network file, the table `irc_transverse` of the database),
and a possible bifurcation says at which arc length that curvature changes sign
(`inflection_arc_length`, linear between the two Hessians around it): an estimate of where the
valley turns into a ridge. On the model surface of `tests/cli_mapper_bifurcation.rs`, whose
valley turns into a ridge 0.452 amu^(1/2) Bohr along the side, the estimate is
0.40 with the default interval of 10, and the same with an interval of 2. An exact Hessian at every point is not offered as a mode of its own: it
costs too much, and a Hessian in every few points shows the possibility.

A complex of two molecules is joined to its dissociation channel by an edge without a
bottleneck. `-map_opt confirm_dissociations=on` (off by default) tests that with a relaxed
scan, the scan of `relaxedscan`: the distance between the centres of the two molecules is
held at values `dissociation_scan_step` (0.3 Angstrom) apart from the complex outwards, every
point minimized under the constraint, until the molecules are apart by the test of the
channels. The curve is kept on the edge (`scan`); a maximum of it more than
`dissociation_scan_tolerance` (0.5 kcal/mol) above the channel is a barrier: the edge is
marked, a warning says so, and the kinetics leaves the edge out instead of taking it for a
capture. For the water dimer with GFN-FF the scan took 7 points to 4.89 Angstrom between the
centres, rose all the way, and ended 1.07 kcal/mol below the channel: no barrier.

### Several electronic states: transition states and crossing points together

`-map_opt states=C,M;C,M` (or `C,M,ROOT` for an excited state of the engine) lists the
electronic states of the run, PES1, PES2, ... in that order. Every minimum then has, besides
the tasks of the method on its own surface (its transition states), crossing tasks for every
other state. They find crossing points the way transition states are found, by AFIR: for
every AFIR push of `afir_ts`, a path from the minimum gives candidates, each candidate is
optimized to the minimum-energy crossing point without the force (MECP for states of
different spin or charge, MECI for two roots of one spin), and from the crossing point a
meta-IRC runs down on each of the two states, whose end is minimized. The crossing point is a
node (`MECP0`, `MECI0`, ...) joined to the two minima by a crossing edge with the two paths,
and a minimum found on another state is explored like any. The network thus holds the
transition states of every surface and the crossing points between them.

```bash
# a singlet and a triplet: transition states on both, MECP between them
run_mapper start.xyz -cfg config.json -map_opt "states=0,1;0,3"
# the ground state and the first excited state of the singlet: MECI between them
run_mapper start.xyz -cfg config.json -map_opt "states=0,1;0,1,1"
```

Any number of states can be given (PES3, PES4, ...): every minimum has a crossing task for
every other state, n (n - 1) / 2 pairs of n states; `crossing_pairs=adjacent` searches only
neighbouring states (PESk with PESk-1 and PESk+1, n - 1 pairs), where the crossings between
states far apart are not wanted or cost too much. The inputs lie on the first state of their
charge and multiplicity and are minimized on it (on an excited state when that is the first). Nodes are the same only on the same state. `crossings=off` explores
each surface on its own.

A state given by its root (`0,1,2`) is the state with that place in the order of the energies
at every structure. Where two roots change places between the minima, put `"follow_states":
true` into `step1_settings` (`-es_follow` of `optmain`; calculators that hand out their
states: the built-in GFN0-xTB, PySCF with a restricted reference): the states are then
followed by the overlaps of their amplitudes along every optimization and IRC and from node to
node, so that PES2 is one state over the whole network. Every node keeps the place of the
states in the order of the energies at its structure (`followed_states` in
`reaction_network.json`, shown in its panel of the report), and the jobs from it start there.

Two kinds of AFIR path lead to the candidates (`crossing_afir`):

- `gp` (the default): the artificial force on the gradient projection of the two states
  (GP/AFIR). The path runs to the seam and within it; its end lies on the seam, displaced
  within it by the force. The candidates are that end and the local minima along the path of
  the seam model function of the two states, which only chooses them.
- `model`: the path on a model function of the two states whose minima lie near their
  crossing points (`-mf seam` for an MECP, `-mf avoiding` for an MECI). One path gives several
  candidates, its end and the minima of the model function along it, so that the crossing
  points on the way of a reaction of several steps are found too; each is only approximate
  until it is refined.

| | `gp` | `model` |
|---|---|---|
| the path | the projected gradient with the force | the model function with the force |
| the candidates | the end and the local minima of the seam model function along the path | the end and the local minima of the model function along the path |
| the refinement | the gradient projection without the force | the same |

Both take several candidates from one path. They differ in the path: `gp` keeps the penalty of
the seam model function out of the optimization (Harabuchi, Taketsugu, and Maeda report 37 %
and 22 % fewer force calculations than on the seam model function for the S0/S1 MECIs of
benzene and naphthalene), `model` follows a scalar function and needs no projector.

On a surface that is not smooth (numerical noise of the energies, a path that zigzags) the
function has many small minima next to each other along a path, which would all be refined to
the same crossing point. Three rules keep them out: a minimum is a candidate only when the
function rises by `crossing_candidate_prominence` on both of its sides before it comes back
to its value; two candidates of one path are at least `crossing_candidate_distance` apart
(the root-mean-square displacement of the atoms; the end of the path and the lower minimum
stay); and a candidate within that distance of a crossing point that the job has found is not
refined. The log of the job says how many were left out. `0` switches a rule off.

`both` runs the two, `off` none; `crossing_direct=on` adds the optimization from the minimum
itself. On a variable cell (`variable_cell` in `step1_settings`) the crossing points, the AFIR
paths towards them, the meta-IRCs, and the minima of the other state move the atoms and the
cell together: the two states are combined in the variables of the cell as `optmain -vc -mf`
does it (the gradient projections and the model functions of their enthalpies; on the
gradient projection of a path the push stands with the enthalpy of each state), and the
energy of a crossing point is the mean enthalpy of the two states there. In a fixed periodic
cell the crossing points (MECP and MECI) are searched as in a molecule, with the
translations along the periodic directions as the only rigid motions (the check of the
branching plane included). `ts_states=1` searches transition states on PES1 only (the minima of the other states
have their crossing tasks alone). Within a minimum the crossing tasks and the tasks of the
method alternate; which minimum comes next is decided by the priority of the run
(`priority=kinetics` counts the crossing edges). With `kinetics=on` the
crossing edges are bottlenecks of the kinetics (a hopping probability of 1 by default). The
engine must give the states: the spin states of any engine with open shells, the roots of an
engine with excited states (`gfn1xtb` with its TDA excited states, PySCF and others through
Python). The report names the surface of every node (`EQ3 · PES2`, `MECP0 · PES1/PES2`), draws
crossing points as outlined triangles apart from the transition states, and its bar above the
figures shows PES1, PES2, ... one at a time, all of them, or all in layers (each surface in a
band, PES1 at the bottom, the crossing points between the bands). On a model of three atoms
with known seams (`tests/cli_mapper_states.rs`) the crossing points lie at the minima of the
seams within 1e-6 Hartree, with two and three states.

#### Any program for the electronic structure

The searches for crossing points take their energies and gradients from the calculator of the
run like every other command, so a program that mopt-rs does not know needs no change of
mopt-rs: it is given as a Python class (`-os engine:module:Class`, an engine of your own; see
[calculators.md](calculators.md)) or as an ASE calculator (`ase:module:Class`), named in
`step1_settings` (`"othersoft": "engine:my_module:MyEngine"`). What it has to hand over:

| For | The engine hands over |
|---|---|
| a crossing of states of different spin or charge (MECP; `states=0,1;0,3`) | `compute(structure, hessian)`: the energy and the gradient for the charge and multiplicity of `structure.state` |
| a conical intersection of two roots (MECI; `states=0,1;0,1,1`) | the same for the root `structure.excited_state` (0: the ground state), with `capabilities.excited_states = True` |
| the roots followed from structure to structure and from node to node (`"follow_states": true`) | `state_characters(structure, count)`: the orbitals, the excitation energies and the amplitudes of the lowest `count` states |
| exact Hessians (the refinement with `calc_exact_hess`, the check of every minimum) | nothing: they are made from the gradients by differences, unless the engine has its own (`capabilities.analytic_hessian`) |

An ASE calculator takes the charge and the multiplicity as its keywords or from `atoms.info`;
for its excited states two functions are named with `-calc_opt` (`excited_setup=module:function`
sets the keywords of the program for a root, `state_reader=module:function` reads the states
for following; running.md, "Excited states of a program outside the package").
`tests/test_crossing_plugin.py` runs `run_mapper` and `run_automecp` on a model of three atoms
given as such a class (`tests/crossing_engine.py`): the crossing points of two spin states, a
conical intersection of two roots, and the roots followed from node to node.

### One reaction: `run_automecp`

`run_automecp` is the counterpart of `run_autots` for crossing points: from one structure, a
few AFIR pushes, and a second state it gives the minimum-energy crossing points (MECP, or MECI
when the second state is another root of the same spin) with the paths down to the minima of
both states.

```bash
# the crossing points of the singlet and the triplet reached by pushing the fragments
# (atoms 1 and 4) and (atoms 7 and 8) together
run_automecp start.xyz -cfg config.json -ma 150 1,4 7,8 -state2 0 3
# with the first excited singlet (MECI), on the model function
run_automecp start.xyz -cfg config.json -ma 150 1,4 7,8 -state2 0 1 1 -afir model
```

The settings file has `step1_settings` (the AFIR paths; the calculator, and the charge and
multiplicity of the first state), `step2_settings` (the optimization of the crossing points),
and `step3_settings` (the meta-IRC, `intrinsic_reaction_coordinates`, and the minimizations),
each with the long option names of `optmain`; `manual_AFIR`, `state2` and `crossing_afir` can
stand in it instead of the command line, and `mapper_settings` takes further settings of the
mapper. The steps:

1. the input is minimized on its own state, and every push gives an AFIR path from that
   minimum towards the crossing points (`-afir gp`, the default, or `model`);
2. the candidates of the path are optimized to crossing points by the gradient projection
   without the force;
3. a meta-IRC runs down from every crossing point on each of the two states, and its end is
   minimized.

Nothing is computed by the command itself: it is a run of the mapper with the settings
`states`, `crossing_pushes` (the pushes of `-ma`), `crossing_from=inputs` (crossing tasks from
the input alone) and `ts_states=none` (no search for transition states), which `run_mapper`
takes as well. The run directory (`automecp_output`, `--output_dir`) has `settings.json`,
`mapper_config.json` (the settings of the mapper run), `automecp_summary.csv` (one line per
crossing edge, a crossing point with the minima of its two sides: its states, its energy, the
minima, how far it lies above each, the norm of the gradient difference, the file of its
paths, the gap, and the check within the seam), `automecp_report.html`, and
`mapper/`, the run of the mapper with `reaction_network.json`, `structures/`, `paths/`,
`mopt.log`, and `mapper_report.html`. `run_automecp --restart automecp_output` continues an
interrupted run.

`automecp_report.html` is the page of the search, made like that of `run_autots` (the same
style, structure viewer, charts and energy diagram; one file that loads nothing from
elsewhere), with the steps of this search in the place of those of a transition-state search:

- a summary: every crossing point with its energy, the norm of the gradient difference, its
  two minima and how far it lies above each; an energy diagram with each crossing point
  between the minimum of its first state and that of its second; the minima; the evaluations
  of the run;
- step 1: for every push the energies of the two states along the AFIR path, with the points
  taken as candidates, and the path as an animation;
- step 2: every candidate optimized to a crossing point, with the energies of the two states
  along the optimization, the gap and the gradient difference at its end; a candidate that did
  not converge has a warning;
- step 3: for every crossing point the energy along the meta-IRC of each state, and the whole
  way from the minimum of the first state through the crossing point to the minimum of the
  second as an animation (the file `mapper/paths/MECP000000_edge0_path.xyz`; every frame is
  named after its part and has the energy of the state it lies on).

It is written from the files of the mapper run at the end of every session of the run, so a
run that was stopped has a page of what it had done. On a variable cell (`variable_cell` in
`step1_settings`) its tables give enthalpies; the calculator it names is the one the mapper
run recorded.

#### The states, the settings the command makes, and a restart

The first state is that of the input: `electronic_charge` and `spin_multiplicity` (or
`unpaired_electrons`) of `step1_settings`, and its root `-root1 N` (or `excited_state` of
`step1_settings`; 0, the lowest state of the multiplicity, when neither is given), so that
`-root1 1 -state2 0 1 2` searches the conical intersections of the first and the second
excited singlet. Two different roots for the first state are refused. `states`,
`crossing_pushes`, `crossing_from`, `ts_states` and `crossing_afir` of the mapper run are made
by the command from its options: `-map_opt` refuses them, and in `mapper_settings` they are
refused unless they say the same. `run_automecp --restart RUN_DIR` goes on with the settings
the run recorded; every option given with it that differs from those the run started with
(the input, `-cfg`, `-ma`, `-state2`, `-root1`, `-afir`, `-map_opt`) is not used, and a
warning names it. A run that ends without a crossing point writes its summary and its page and
then ends with an error (exit code 1).

#### The search for a crossing point

These hold for `run_automecp` and for `run_mapper` with states alike.

- **The defaults of the optimization.** The crossing searches take the settings of the
  transition-state step (`step3_settings` of the mapper, `step2_settings` of
  `run_automecp`), so that without more a crossing point was optimized with
  `rsirfo_block_bofill` and two exact Hessians (one per state) every five iterations. A
  crossing point is a minimum of the projected surface, and where that step names neither
  `opt_method` nor `calc_exact_hess` it is now optimized with `rsirfo_block_fsb` and no exact
  Hessians (where it names either, with the settings of the step as they are; the
  transition-state searches of the step keep theirs). Measured on the two-state
  model surfaces of
  `tests/crossing_searches.rs` (an MECI of two coupled wells and an MESX of two spin wells,
  three starts each, the normal and the tight criteria; 24 searches, evaluations of one state
  counted):

  | method | exact Hessians | gradients | Hessians | not converged |
  |---|---|---|---|---|
  | `rsirfo_block_bofill` | every 5 iterations | 520 | 104 | 0 |
  | `rsirfo_block_bofill` | none | 448 | 0 | 0 |
  | `rsirfo_block_fsb` | every 5 iterations | 384 | 76 | 0 |
  | `rsirfo_block_fsb` | none | 434 | 0 | 0 |

  All four reach the crossing point; the minimizer with exact Hessians needs the fewest
  gradients, but its Hessians cost more than the 50 gradients it saves (a Hessian from
  differences of gradients is `6N` of them), so the default has none. A calculator with
  cheap analytic Hessians can ask for them (`"calc_exact_hess": 5`).
- **Two states with the same gradient.** The gradient projection needs the direction of
  `g_1 - g_2`. Where its norm is below 1e-8 Hartree/Bohr the two states run parallel, no
  direction closes their gap, and the search stops with an error that says so (before, the
  MECI projection took the whole mean gradient out there and reported its start converged,
  and the MESX projection went down the mean energy with the gap open). In a job of the
  mapper that candidate is left out with a warning, and the job goes on with its other
  candidates. Above the limit the
  multiplier of the seam condition, `(mean gradient . x) / |g_1 - g_2|`, is finite; it grows
  as the inverse of `|g_1 - g_2|`, and the trust radius holds the steps.
- **The gap.** The projected gradient has the component `2 (E_1 - E_2)` along the unit
  vector of the difference gradient, so its convergence bounds the gap only by
  `sqrt(3N) * threshold / 2` (a vector of `3N` coordinates whose root-mean-square value is
  below the threshold has a norm below `sqrt(3N)` times it): 1.1e-3 Hartree for 40 atoms with
  the normal criteria (RMS force 2e-4 Hartree/Bohr), above the `crossing_gap` of 1e-3. The
  gap is therefore a condition of its own: both states are computed at the end of the search,
  and a point whose gap is above `crossing_gap` is no crossing point, for an MECP and an MECI
  alike. The gap at the end (the energy of the first state of the pair minus that of the
  second, in the order of the states of the run) is kept with every crossing edge:
  `gap_hartree` in `reaction_network.json` and the database, `crossing_edges.csv`,
  `automecp_summary.csv`, and the tables of both pages.
- **The check of the branching plane** of an MECI that still finds the point not stationary
  on the seam after its last correction ends the search with a reason of its own
  (`seam_check_failed`), not as converged; the check works in the directions the constraints
  leave free (frozen atoms, held coordinates). The mapper takes no crossing point from such a
  search and says so in a warning of its own (not "did not converge").
- **The check within the seam (on by default).** Every crossing point that is accepted is
  analysed within the seam, as `freqmain -mf opt_mesx|opt_meci` does it at a structure: the
  Hessians of both states (and, for an MECI, the second direction of the branching plane from
  the gradients at four displaced structures), the vibrations of the mean energy within the
  seam, and the number of imaginary modes there by the one criterion of an imaginary mode
  (`opt::mode_kill::IMAGINARY_CURVATURE`). 0: a minimum on the seam, a minimum-energy
  crossing point. One or more: a saddle point on the seam; the point is kept in the network,
  but the tables, the pages and `automecp_summary.csv` call it a saddle point on the seam
  with the count, and a warning goes to the console, `mopt.log` and the notes of the report.
  Nothing descends along the seam from it: to go on, displace the structure along the
  imaginary mode within the seam (`freqmain` at the structure writes the mode) and optimize
  the crossing point again from there (`optmain -mf opt_mesx` or `opt_meci`). The result is
  `seam_imaginary_modes` and `seam_check` (`checked`, or why not) of the crossing edge in the
  network files, and the columns of the same names in the tables. The cost is the Hessians of
  both states at every accepted point, plus four evaluations of both states for an MECI; they
  are counted with the other evaluations. `crossing_seam_check=off` (`-map_opt` of either
  command) switches the check off: the points are then `not checked`, as where the analysis
  cannot be done (a variable cell; an error of the analysis, which is named).
- **Counting.** The console, the pages and the summary count crossing points as nodes. A
  crossing point reached with other minima at its ends has several crossing edges, and the
  tables (`crossing_edges.csv`, `automecp_summary.csv`) have a line per edge with the node
  named.

### An ONIOM surface

Every step with the same ONIOM layers (`oniom_flag`, and `oniom_middle`, `oniom_options` where
they are used; the steps run on one surface) makes a network of the **cores**: the atoms the
optimizer moves, the model region with the outer atoms of its links. Every calculation of the
run is on the relaxed surface of the core, as in every ONIOM run of mopt-rs but the
dynamics: the environment is relaxed to a minimum before every evaluation (the
microiterations, [oniom.md](oniom.md)). The AFIR pairs, `active_atoms`, and the other
numbers of atoms of the settings are those of the core, in the order of the whole structure.

```json
"step1_settings": {"othersoft": "gfn1xtb", "oniom_flag": ["1-6", "auto", "gfnff"]},
"step3_settings": {"othersoft": "gfn1xtb", "oniom_flag": ["1-6", "auto", "gfnff"]},
"step4_settings": {"othersoft": "gfn1xtb", "oniom_flag": ["1-6", "auto", "gfnff"]}
```

- **When two structures are one node** (`identity_scope`): `high`, the default, compares the
  cores alone (bond graph and RMSD; not the energies, which the environment moves);
  `active_site:R` adds the atoms of the environment within `R` Angstrom of the core in the
  first input, which the optimizer then moves with the core; `all` is refused (a relaxed
  environment is a function of its core in every calculation).
- **Environments.** The same core found again with another environment is the same node: the
  network lists the other environments with their energies (`environments` of the node; two
  within the energy tolerance of the identity, `energy_tolerance_kcalmol`, are one), and
  the node is the one of the lowest energy (its structure file is written again when a lower
  one comes). A TS edge keeps the barriers of its own IRC ends; an end more than 1 kcal/mol
  from its node is reported as environment heterogeneity. The report adds the number of other
  environments and their spread to the name of a node.
- **Files.** The structures of the nodes and the IRC paths hold the whole system, with the
  environment as the job relaxed it (`structures/`, `paths/`); `reaction_network.json` keeps
  the cores (`coords_bohr`) and the whole structures (`whole_coords_bohr`, with
  `whole_elements`); a job starts from the environment of its node, and a restart gives the
  same (the journal records the relaxed environments). The report draws the cores.
- **Fixed atoms** (`fix_atoms` in the steps, the `-fix` of `optmain`): atoms of the
  environment held where they are during the relaxations, such as the protein far from the
  active site of an enzyme; they leave the settings of the optimizations, which move the
  core. A fixed atom of the core is refused (the mapper moves the core freely), and so is
  any other constraint and `-fix` without ONIOM.
- **Not yet on an ONIOM surface** (refused with the reason): `-remap`, `-single_points`,
  `method=nanoreactor`, `entrance_structures`, `states`, `ts_conformers`. Dissociation
  channels are off (a core sits in its environment).

An example that the tests run (`tests/cli_mapper_oniom.rs`): vinyl alcohol on GFN1-xTB with
two water molecules on GFN-FF, one of them donating a hydrogen bond to the oxygen. The AFIR
push between the CH2 carbon and the hydrogen of the hydroxy group meets the rotation of the
hydroxy group first (syn to anti, a dihedral angle of about 90 degrees at the transition
state); its transition state and both ends of its IRC converge with the water relaxed at every
point, and the end on the side of the start is the starting minimum again at its energy
(within 3e-9 Hartree). The optimizers recentre and rotate the core at every step, and the IRC
puts its centre of mass at the origin; the environment moves with the core by the same rigid
motion before it is relaxed, so that those moves do not push the core against its
environment ([oniom.md](oniom.md), the microiterations).

### A periodic cell: fixed or variable

A structure with a cell (an extended XYZ file with `Lattice=`) makes a network of periodic
structures: the bonds, the AFIR pairs, the comparisons, and the paths take the nearest image of
every atom; the translations along the periodic directions are the symmetries that the
optimizations, the vibrations, and the IRC leave out. The nanoreactor is off in a cell.

A minimum that came apart in its cell is a dissociation channel (`DC`): where its molecules in
contact (two atoms closer than the sum of their van der Waals radii and `channel_margin`, with
the images of the cell) no longer span the periodic directions of the cell, as when a pull or a
stretch of the cell leaves layers, chains, or molecules with vacuum between them. The test is
the dimension of the network of contacts: the rank of the lattice translations of its closed
paths (3 for a molecular crystal, 2 for layers, 1 for chains, 0 for separate molecules). The
node keeps its structure and its enthalpy (its parts are periodic themselves and are not
optimized apart, unlike the fragments of an isolated system), says what it came apart into, is
the same node when the structure is found again, and is not explored. It is kept even where
imaginary modes remain (the minimization does not displace it along them; its wavenumbers,
imaginary ones included, go onto the node): such a structure is a minimum of the cell it was
pulled into, or a flat saddle point of it (user, 2026-11-20). `channels=false` keeps such
structures as minima.

With `"variable_cell": true` in `step1_settings` (with `variable_cell_pressure` in GPa and
`variable_cell_weight` where they are not the defaults; step 1 gives them to the other steps,
and steps that set other values are refused) the cell moves with the atoms in every step, on
the enthalpy `E + P V` in the variables of `optmain -vc`:

- the AFIR paths: the push between two atoms is a bias potential whose share of the stress
  moves the cell too (PBC-AFIR on a variable cell);
- the minima and the transition states: the optimizer of plain variables on the atoms and the
  cell; a minimum is checked with the Hessian of the atoms and the cell together. Displacement
  along a negative mode requires explicit mode kill. IRC endpoints use the coupled modes for the configurable 50 cm^-1 threshold. Two or more
  unstable modes exclude the endpoint from equilibria regardless of their magnitudes;
- the relaxation of an AFIR path and the other two-point searches: on the variables of the
  first structure, every image with its cell;
- the IRC: that of `ircmain -vc`, mass-weighted with the masses of the variables;
- the vibrations of the kinetics: those of the atoms in the cell of the structure.

The energies of the nodes are then enthalpies. Two structures in different cells are one node
only where a basis of the lattice of the first, turned rigidly, agrees with the lattice vectors
of the second within the RMSD tolerance in every component (the bases tried are the integer
combinations of the vectors with coefficients -1, 0, 1 and the determinant of a proper turn,
the turn the best fit of the three vectors): an optimization of the atoms and the cell may
turn the crystal as a whole, and the same lattice may come back with its vectors in another
order. The atoms of the second are then compared at their fractional coordinates in that
basis, in the cell of the first (copper with EMT on a variable cell: a vacancy that hopped
came back turned by 3 degrees, 0.7 Bohr apart in the components of its vectors and 0.0002
Bohr after the turn). In every
cell the comparison tries the point operations of the lattice (its rotations; the reflections
too with `mirror=true`) and the translations that put an atom of the rarest element onto each
atom of that element of the other structure: a crystal moved by a lattice translation of a
sublattice, or turned by an operation of its lattice, is the same crystal. A vacancy that
hopped to the next site is the minimum it left, and the hops to its neighbours are one
transition state from that minimum to itself (copper with EAM, 31 atoms: one minimum and the
hop at 0.528 eV in a fixed cell, 0.452 eV with the cell relaxed at zero pressure).

With `"cell_pushes": true` every minimum also has tasks that push the cell itself: each
periodic lattice vector (`a`, `b`, `c`), and the distance between the tips of each two (`ab`,
`bc`, `ca`), compressed with `afir_gamma_kJmol` and stretched with its negative: the artificial
force on the origin and the tips of the lattice vectors of `optmain -vc_afir` (Takagi et al.),
which Kondo, Hasegawa, and Maeda put on the translation vectors of molecular crystals beside
the pushes of the molecules. The atoms follow the cell; the path of the push is handled as that
of a push of atoms. Forces on the cell in the step settings (`-vc_afir`) are refused (the tasks
are the pushes), and a variable cell is refused on an ONIOM surface (ONIOM has no stress). In a
metal every neighbour lies within 1.2 times the sum of the covalent radii, so that
`pair_bond_scale` (below) has to be lower for the neighbours to be pushed.

With `"afir_fragments": "molecules"` the tasks of a minimum are pairs of molecules instead of
pairs of atoms: the connected parts of the bond graph (whole across the cell), each pair whose
closest atoms are within the distance window pushed (and with `include_negative_gamma` pulled)
as two fragments by the AFIR function of all their atom pairs, with no force within a
molecule. This is the single-component AFIR of Kondo, Hasegawa, and Maeda for molecular
crystals (artificial forces between automatically defined fragments, intramolecular forces
switched off so that only the packing changes; γ = 100 kJ/mol for the benzene crystal with
GAFF), and with `cell_pushes` the forces on the translation vectors come beside them.

The standard PBC mapper validation uses the `benzene_crystal` Pbca cell from
`tests/golden/gfnff_pbc_reference.json` (four benzene molecules, 48 atoms).
It uses GFN-FF with a variable cell, molecular fragments and lattice-vector pushes.

### A method of one's own in Python: `method=python:MODULE:CLASS`

A class written outside mopt-rs gives the tasks of a minimum and, for a job, what it proposes:
transition-state guesses and first paths. The mapper does the rest on the surface of the run,
as for its own methods: every path is relaxed by a chain method (`neb` by default;
`method_path=dmf` and `method_path_KEY=VALUE` choose another and its settings), its highest
`top_n_candidates` maxima join the guesses, and every guess is optimized to a transition state
whose IRC joins two minima. The other `method_KEY=VALUE` settings reach the class.

```python
# my_methods.py, importable from where run_mapper runs
class NearSaddle:
    def __init__(self, settings):           # {"offset": "0.03"} from method_offset=0.03
        self.offset = float(settings.get("offset", "0.03"))

    def tasks(self, node):                  # the same list in every run (a restart skips
        return [{"key": f"near:{node['id']}",   # the tasks that were done)
                 "description": "a guess and a path", "parameters": {}}]

    def propose(self, job):
        x, y = job["start"]["variables"]    # "symbols", "positions" (Angstrom), "cell" for atoms
        return {"guesses": [[-0.79, 0.60]],
                "paths": [[[x, y], [-0.4, 0.9], [-0.1, 0.5]]],
                "notes": ["one guess and one path"]}
```

```text
run_mapper start.txt -cfg config.json -map_opt method=python:my_methods:NearSaddle method_offset=0.03
```

A node, and the `start` of a job, is a dictionary: `id`, `label`, `energy` (Hartree; an
enthalpy on a variable cell), and `symbols`, `positions` (Angstrom), `charge`, `multiplicity`,
`cell` (the lattice vectors as rows, or `None`), or for plain variables `variables` and
`names`. A job has `id`, `node`, `task` (`key`, `description`, `parameters`), and `start`. A point
of a proposal is the positions of the atoms (Angstrom; the atoms, charge, multiplicity, and cell
of the start), a dictionary with `positions` and `cell`, or the values of the variables. The
module `mopt_rs.mapper_plugins` says the same. A proposer computes what it needs itself (with
the calculators of `mopt_rs.calculators`, for example); the calculations of the run, their
journal, and the restart are the mapper's. Methods of Python classes run from the commands of
the Python package; the command line of the Rust binary refuses them with that reason.
`tests/test_mapper_plugin.py` runs the class above on the Muller-Brown surface.

### Plain variables

With an objective of plain variables in the place of a calculator (`"othersoft": "mb2d"`, any
name of `optmain -os`: `mb2d`, `wolfe_quapp`, `three_hole`, `himmelblau`, `rosenbrock`,
`styblinski_tang`, `expr:...`), the nodes are points of the variables, of any number, and the
inputs are variables files (one point each, `name value` per line, as for `optmain`):

```bash
run_mapper start.txt -cfg config.json       # step1_settings: {"othersoft": "mb2d"}
```

- **Exploring**: the default method is `addf` (ADDF on the variables), since AFIR pushes need
  atoms; `afir_ts` and `nanoreactor` are refused with that reason, and dissociation channels
  do not arise. Methods registered by users take the points as they come
  (`Configuration::Variables`).
- **The steps**: the optimizer of plain variables (`optmain -os ...`) for the minima and the
  transition states, the IRC with unit masses, the exact Hessians of the objective (automatic
  differentiation) for the checks and every tenth point of the IRC. Options that need atoms
  (bias potentials, constraints, coordinate systems, model Hessians, charge and spin) are
  refused by the steps.
- **The same point**: two points are the same when their energies are within
  `energy_tolerance_kcalmol` and their Euclidean distance is below `variable_tolerance` (0.01,
  in the units of the variables): a hundredth of the smallest distance between two stationary
  points of the same kind of the built-in objectives (0.80, two minima of `mb2d`; the others
  1.2 and more), far above where two optimizations to the same point end.
- **Files**: `structures/EQ000000.txt` is a variables file (`# EQ0 energy=...` and the values),
  which `optmain -os ...` reads as its input; the trajectories of the jobs and the IRC paths
  are frames of the shape of xyz frames with `name value` lines (`jobs/job_000003/0001_afir_path.txt`,
  `paths/TS000000_irc.txt`); `reaction_network.json` lists the names of the variables
  (`"variables"`) and every node its values.
- **The report**: up to four variables, the map is the variables themselves (any two of
  them); with two, the surface is drawn on a grid behind the nodes in the colors of the energy
  scale, and every link runs along its IRC. A node shows its values in place of a structure,
  and energies are the values of the objective as they are (`-report_opt unit=` chooses a
  unit). More variables are placed by classical scaling of their Euclidean distances.
- The objective is shared by every job and is not journaled: a restart computes its values
  again (they are the same every time).

`tests/cli_mapper_variables.rs`: on `mb2d` the network has the three minima and the two
saddle points of Muller and Brown (3 jobs); on `rosenbrock` with four variables the two minima
(1, 1, 1, 1) and (-0.776, 0.613, 0.382, 0.146) and the saddle point between them; on
`wolfe_quapp` its three minima. One ADDF path on `mb2d` runs where the surface overflows (a
growing exponential far from its region): such a path ends there (`undefined` in the log of the
search) and the others go on.

Which minimum the next job starts from is the choice of the priority: by default `boltzmann`,
which draws a minimum that still has tasks with the probability `exp(-(E_i - E_min) / k_B T)`
at `temperature_K`; the job takes the next task of that minimum.

### A network on another surface: `-remap`

A network explored on a fast surface is drawn again on a better one by a run of its own:

```bash
run_mapper ar7.xyz -cfg fast.json --output_dir first
run_mapper -remap first -cfg better.json --output_dir second
```

`-remap RUN_DIR` takes the run directory of a run that ended or was stopped (its
`reaction_network.json`), or the database of a run (`reaction_network.sqlite`; the structure
and path files are read beside it). No input file is given: the minima of the first network
are the inputs, each minimized on the surface of the settings of this run (`-cfg`,
`-map_opt`), on the state it was on. Then every transition state, crossing point and saddle
point of higher order of the first network is one job:

| Item of the first network | Its job |
|---|---|
| a transition state | the minimum of its first end on the new surface, the structures of its path file (minimum to minimum, at most 60) and the minimum of its second end are the first chain of `method_path` (`neb`; `method_path_KEY` its settings, as for `method=connect`); every maximum of the relaxed chain is optimized to a transition state and followed by its IRC, the ends minimized. `-map_opt method_path=none`: no chain, its transition state is the guess (also when the chain gives no maximum) |
| a crossing point (MECP, MECI) | refined on the gradient projection of its two states from its structure, then the meta-IRC on each state and the minimization of its end |
| a saddle point of order 2 or more | optimized at its order from its structure and descended from (`higher_saddles=on`; without it these items are left out, with a warning) |

The first run is only read. The second run directory is that of any run: its network, its
report, its kinetics (`kinetics=on`), `--restart`. The run has no limit of jobs unless one is
given (`--max_iter`, `max_iterations`), and ends when every item has had its job. A network
of several states needs the same number of states in the same order (`-map_opt states=...`);
the searches for other crossing points (the AFIR paths, the searches from the minima) are not
run. What a job finds beyond its item (another transition state on its chain, a minimum the
first network did not have) is kept.

Not everything comes through on another surface, least of all on a real molecule: a minimum
that is no minimum there, a transition state whose barrier is gone, a crossing point of states
that no longer cross. What does not come through is left out and the run goes on (user
request, 2026-11-19):

- a minimum that does not converge to a minimum, or whose minimization fails (an SCF that
  does not converge): no node, a warning of the network, and the items it was an end of run
  from another minimum (the other end, or the first minimum that came through; for a crossing
  point a minimum on one of its two states);
- a transition state, crossing point or saddle point of higher order whose job gives none of
  its kind, or fails: no node, and a warning of the job.

`remap.csv` of the second run directory has a row for every minimum and item of the first
network: its label and energies there, its job, what became of it, the nodes of its kind that
the job added, and the node of its kind in the second network that is nearest to it, with the
distance (for structures the root-mean-square distance of the atoms after the best
superposition and numbering of identical atoms, in Angstrom) and its energies. What became of
it is one of:

| Status | Meaning |
|---|---|
| `minimized` | (a minimum) it is a node of the run |
| `added` | the job added a node of its kind |
| `known` | the job converged to a node that another job had found before it (the IRC of one transition state that runs into a saddle point of order 2, the descent from a saddle point that reaches a transition state): the nearest node says what it became |
| `left out` | it did not come through: the minimum did not converge, or the job gave nothing of its kind |
| `failed` | the job ended with an error (its message is in the warnings of the network); nothing of it is in the network |
| `not run`, `not minimized yet` | the run was stopped before it, or no minimum came through that could run it |

On the surface of `tests/cli_mapper_saddles.rs` (three variables; 8 minima, 9 transition
states, two saddle points of order 2), drawn again on the surface with a term that moves every
stationary point (`+ 0.05 (q1 - 0.3)^2`): 11 jobs, the same numbers of minima, transition
states and saddle points, and runs stopped at several points and restarted give the same
network. On the model of two states of `tests/cli_mapper_states.rs`, the two minima of the
first state, the minimum of the second, their transition state and the two MECP are found
again on the tilted surfaces in three jobs; on surfaces where the first state cannot be
computed beyond a wall, the minimum behind the wall, the transition state and the crossing
point that leads there are left out, and the run ends with the other two minima and their
crossing point.

On molecules, with the built-in programs (HCN and HNC joined by `method=connect` on GFN0-xTB,
then taken to GFN1-xTB; energies in kcal/mol above HCN):

| | HNC | Transition state | Barriers from HCN and from HNC | Cost on GFN1-xTB |
|---|---|---|---|---|
| the network of GFN0-xTB | 15.72 | 68.29 | 68.29, 52.56 | - |
| `-single_points` (GFN1-xTB at the structures of GFN0-xTB) | 21.29 | 58.08 | 58.08, 36.79 | 3 energies |
| `-remap` | 21.22 | 58.05 (1195i cm^-1; 1286i on GFN0-xTB) | 58.05, 36.83 | one job |
| `-remap`, `method_path=none` | 21.22 | 58.05 | 58.05, 36.83 | one job, without the chain |

Here the single points are within 0.1 kcal/mol of the remap, since the two surfaces have
nearly the same structures (the transition state moves by 0.03 Angstrom); they are what to
try first. A case where nothing comes through: the complexes of chloride with
1-chloro-1-fluoroethane (the SN2 network of GFN1-xTB above) are no minima on GFN0-xTB, where
the chloride leaves in the minimization; both minima are left out, the transition state has
no minimum to run from (`not run`), and the run ends with an empty network and the table
that says so, while the single points of the same network are computed as for any other.

### The energies of a network on another surface: `-single_points`

The cheaper way to a better surface keeps the structures: every node of a network gets the
energy of another surface at its structure (the layer written `2//1`: energies of the second
surface at structures of the first).

```bash
run_mapper -single_points first -cfg better.json --output_dir points --max_workers 8
```

`-single_points RUN_DIR` takes what `-remap` takes (a run directory, or the database of a
run), and no input file. Nothing is optimized and no Hessian is computed:

| Node | Its energy |
|---|---|
| a minimum, a transition state, a saddle point of higher order | one energy, on its state |
| a crossing point (MECP, MECI) | the energies of its two states; the node has their mean, and `single_points.csv` their gap. A crossing point of the first surface is in general none of the second: the gap says how far it is from one |
| a dissociation channel | every fragment on its own (its structure of the first run), the channel their sum |

The network of the run is the first one with these energies: the same nodes with their
numbers, labels, structures and frequencies; the barriers of the edges computed again; a free
energy shifted by the change of the energy (its thermal part is that of the first surface);
the energies along the IRCs left out (they are those of the first surface), the path files
copied as they are. The jobs listed are those that found the nodes in the first run. The run
directory is that of any run: the report, the kinetics (`kinetics=on`), the database
(`store=sqlite`). `--max_workers` energies are computed at a time, each evaluation goes to the
journal, and a run stopped on the way writes no network and continues with `--restart`. A
network of several states needs the same number of states in the same order.

`single_points.csv` has both energies of every node and fragment, relative to the lowest
minimum of each network, and the gap of the two states at every crossing point.

On the surface of `tests/cli_mapper_saddles.rs` (19 nodes) the energies on the second surface
are those of its formula at the structures of the first, with 19 evaluations (one per node)
and nothing else; on the model of two states of `tests/cli_mapper_states.rs` a crossing point
has the mean of its two states (8 evaluations for 6 nodes), and runs stopped after 2, 5 and 7
evaluations give the same network when they are continued.

### The networks of several runs as one: `-merge`

```bash
run_mapper -merge run_a run_b run_c -cfg config.json --output_dir merged
```

writes the networks of the run directories (or databases) as one, without a calculation and
without an input file. The first is taken as it is. Every node of a later one is matched with
the nodes so far by the identity test (`rmsd_threshold`, `energy_tolerance_kcalmol`, `mirror`;
a dissociation channel by its fragments) and added where it is new; an edge is added where the
merged network has none with the same transition state, crossing point or saddle point; the
jobs are appended with new numbers, and the inputs of every run are inputs of the merged
network. `merge.csv` says what every node of every run is in the merged network (`first`,
`added`, `known`). The path files of what was added are copied under the new numbers.

The runs must have the same atoms in the same order and the same states. That they ran on the
same surface is the word of the user: nothing in a network says more of its surface than the
name of its calculator. The merged directory is read like a run directory by `run_kinetics`,
`-remap`, `-single_points` and the report. It has no journal and is not continued by
`--restart`: a search from it starts as a new run from its structures.

## Settings

The command line goes over the settings file, the file over the defaults.

| `mapper_settings` | Option | Default | Meaning |
|---|---|---|---|
| `temperature_K` | `--temperature` | 300 | temperature of the Boltzmann priority (K) |
| `rmsd_threshold` | `--rmsd_threshold` | 0.125 | largest RMSD of the same structure (Angstrom); MultiOptPy's default is 0.30 |
| `energy_tolerance_kcalmol` | | 0.05 | largest energy difference of the same structure (kcal/mol); MultiOptPy's default is 1.0 |
| `max_iterations` | `--max_iter` | 50 | jobs of the run, 0: no limit |
| `afir_gamma_kJmol` | `--afir_gamma` | 100 | collision energy of the pushes (kJ/mol) |
| `max_pairs` | `--max_pairs` | 9999 | pairs per minimum |
| `dist_lower_ang`, `dist_upper_ang` | `--dist_lower`, `--dist_upper` | 1.5, 5.0 | distance window of the pairs (Angstrom) |
| `active_atoms` | `--active_atoms` | all | atoms of the pairs (from 1) |
| `include_negative_gamma` | `--negative_gamma` | false | also pull every pair apart |
| `pair_bond_scale` | | 1.2 | pairs closer than this times the sum of their covalent radii are bonded and not pushed (0: every pair in the window; a metal needs less than 1.2, where every neighbour is within it) |
| `afir_fragments` | | `atoms` | what the AFIR pushes act on: `atoms` (pairs of atoms, MultiOptPy's mapper) or `molecules` (pairs of molecules as fragments, no force within a molecule; [above](#a-periodic-cell-fixed-or-variable)) |
| `cell_pushes` | | false | on a variable cell, also push the cell itself: every periodic lattice vector and every distance between the tips of two, compressed and stretched ([below](#a-periodic-cell-fixed-or-variable)) |
| `excluded_node_ids` | `--exclude_nodes` | none | minima that are kept but never explored |
| `exclude_bond_rearrangement` | `--exclude_bond_rearrangement` | false | new minima whose bond graph is not that of their input are not explored |
| `rng_seed` | `--rng_seed` | 42 | seed of the order of the pairs and of the choices |
| `output_dir` | `--output_dir` | `mapper_output` | the run directory |
| `top_n_candidates` (top level) | `-n` | 3 | guesses per job |
| `n_parallel` | `--max_workers` | 1 | jobs at the same time (the same as `workers`) |
| `worker_timeout_s` | | no limit | a job that runs longer (seconds) is stopped and recorded as failed |

Settings of mopt-rs, in `mapper_settings` or as `-map_opt KEY=VALUE`:

| Key | Default | Meaning |
|---|---|---|
| `method` | `afir_ts` (`addf` for plain variables) | the way a minimum is explored: `afir_ts`, `addf`, `nanoreactor` ([above](#other-ways-of-exploring-a-minimum-addf-and-nanoreactor)), `connect` (the missing connections to the goal, [above](#towards-a-goal-priorityrrt-and-methodconnect)), `python:MODULE:CLASS` (a class of one's own, [above](#a-method-of-ones-own-in-python-methodpythonmoduleclass)), or a registered name |
| `priority` | `boltzmann` (`rrt` with a goal) | the choice of the next minimum: `boltzmann`, `tabu`, `water_filling`, `water_filling_adaptive`, `boltzmann_heated`, `tabu_heated`, `kinetics` (below), `rrt` (towards the goal), or a registered name |
| `goal` | none | the goal structure of `priority=rrt` and `method=connect` ([above](#towards-a-goal-priorityrrt-and-methodconnect)) |
| `goal_stop` | true | the exploration ends when the goal is joined to a start |
| `goal_match` | `structure` | `bonds`: a minimum with the bond graph of the goal counts as the goal |
| `accept_temperature` | 7500 with `priority=rrt`, off otherwise | K of the acceptance filter of new minima |
| `rrt_goal_weight`, `rrt_locality` | 1, 20 | `c` and `theta` (per Angstrom) of `priority=rrt` |
| `higher_saddles` | false | saddle points of order 2 or more as nodes, descended from ([above](#saddle-points-of-higher-order-experimental)) |
| `max_saddle_order` | 2 | the highest order followed |
| `upward` | false | upward searches by high-index saddle dynamics from the transition states of every minimum (turns `higher_saddles` on) |
| `upward_modes` | 2 | the softest stable modes of a transition state pushed along |
| `states` | (one state) | the electronic states of the run, `C,M[,ROOT]` with `;` between them: transition states on each and the crossing points between them ([above](#several-electronic-states-transition-states-and-crossing-points-together)) |
| `crossings` | true with `states` | the crossing points between the states searched from every minimum |
| `crossing_pairs` | `all` | which pairs of states: `all`, or `adjacent` (PESk with PESk-1 and PESk+1 only) |
| `crossing_afir` | `gp` with `states` | the AFIR paths towards crossing points, for every AFIR push of `afir_ts` and every pair of states: `gp` (the force on the gradient projection; the end of the path is the candidate), `model` (on `-mf seam` for an MECP, `-mf avoiding` for an MECI; the end and the minima along the path are the candidates), `both`, `off` |
| `crossing_direct` | false | the crossing point also optimized from every minimum itself |
| `ts_states` | all | the states (PES numbers, `,` between them) on which transition states are searched; `none`: on none |
| `crossing_pushes` | the pairs of atoms of `afir_ts` | the AFIR pushes of the paths towards crossing points, given by their fragments: `GAMMA:ATOMS:ATOMS` with `;` between them (kJ/mol; atoms from 1 with `,` and `-`), e.g. `150:1,4:7-8` |
| `crossing_from` | `all` | the minima that have crossing tasks: `all`, or `inputs` (the inputs alone) |
| `crossing_smf_alpha`, `crossing_amf_alpha` | 0.011426, 0.011426 | `alpha` of the seam and Avoiding model functions (Hartree; 30 kJ/mol); the seam model function also chooses the candidates along a `gp` path |
| `crossing_amf_branch` | `upper` | the adiabatic state of the Avoiding model function: `upper` or `lower` |
| `crossing_gap` | 0.001 | the largest gap of the two states at a crossing point (Hartree; 0.63 kcal/mol): both states are computed at the end of every crossing search, and a search that converged with a larger gap has found no crossing point; a warning says how far apart the states are, and no meta-IRC runs from its end |
| `crossing_candidate_prominence` | 0.00019 | how far the function along an AFIR path must rise on both sides of a minimum for it to be a candidate of a crossing point (Hartree; 0.5 kJ/mol; 0: every minimum) |
| `crossing_candidate_distance` | 0.05 | the smallest root-mean-square displacement of the atoms between two candidates of one path, and between a candidate and a crossing point the job has found (Angstrom; 0: none) |
| `crossing_parallel_stop` | `off` | `on`: a crossing search is ended where its two states run parallel, a region where they are degenerate and no point (fragments apart, for example): when the gap is at most `crossing_gap` and the difference of the two gradients below `crossing_parallel_gradient` for `crossing_parallel_steps` iterations in a row. A warning says so, and nothing runs down from its end |
| `crossing_parallel_gradient`, `crossing_parallel_steps` | 0.01, 10 | the difference of the two gradients below which the states count as parallel (Hartree/Bohr), and the iterations in a row that end the search |
| `crossing_seam_check` | `on` | every accepted crossing point analysed within the seam (the Hessians of both states): the number of imaginary modes there is kept, and a point with one or more is a saddle point on the seam, not a minimum-energy crossing point; `off`: not checked |
| `crossing_step` | 0.1 | the largest step of the AFIR paths towards crossing points and of the refinement of a crossing point (Angstrom) |
| `confirm_bifurcations` | false | possible bifurcations of the IRCs tried across the path |
| `irc_hessian_interval` | 10 | the points between the exact Hessians the IRC computes on its own when enabled (`irc_hessians`); the curvature across the path at each is kept |
| `confirm_dissociations` | false | a relaxed scan from every complex of two molecules towards its dissociation channel; `dissociation_scan_step` (0.3 Angstrom), `dissociation_scan_tolerance` (0.5 kcal/mol: a maximum of the curve higher above the channel is a barrier) |
| `path_peaks` | `failed` | the highest point of a path that gave no transition state is saved beside the network; `all`: that of every path; `off` |
| `ts_conformers` | false | one conformer search per transition state (`confsearch -tsconf`, the block `ts_conformer_settings` of the settings file) |
| `entrance_structures`, `entrance_contact` | 0, 1.0 | structures of the dissociation channel of the first input as further inputs, their molecules in contact at this times the sum of the van der Waals radii |
| `max_bond_changes`, `max_species_changes` | no limit | new minima whose bonds (chemical species) changed more than this many times on the way from their input are not explored; 0: the conformers of the inputs only |
| `method_step1`, `method_step1_KEY` | `afir` | `afir_ts`: the method of the first path of a pair of atoms (`se_gsm`, `gnt`: from the minimum along the pair, without a push) and its settings |
| `kinetics` | `off` | `on`: the kinetics of the network when the exploration ends ([below](#the-kinetics-at-the-end-of-a-run)) |
| `kinetics_*` | | the settings of `priority=kinetics` and of `kinetics=on` (below) |
| `stop` | none | further stop rules, `;` between them: `nothing_new:N` (no new minimum in `N` jobs in a row), `rank_unchanged:R,N`, `max_jobs:N` |
| `store` | none | further stores of the network, `,` between them |
| `html_report` | true | `mapper_report.html` |
| `save_job_steps` | true | save the input optimization and job trajectories, per-job logs, live pages, frame indices and completed NEB bands. Set `-map_opt save_job_steps=false` to keep the network, decision history, checkpoint and resulting reaction paths without intermediate job output |
| `mirror` | false | whether a mirror image is the same minimum (and the mirror image of a fragment of a dissociation channel the same fragment) |
| `verify_minima` | true | every minimization (inputs, ends of an IRC, quenched frames) checks the Hessian and stores the wavenumbers; mode kill is opt-in. At IRC endpoints, imaginary modes of magnitude at least 50 cm^-1 trigger a TS–TS search; weaker ones are kept on the EQ with a warning; `false` takes converged structures unchecked (`minima_verified` of the network file says which) |
| `irc_hessians` | true | the exact Hessian at every tenth point of every IRC (about a tenth more cost with GFN1-xTB), for the possible bifurcations; `false`: only where `calc_exact_hess` of `step4_settings` asks |
| `irc_end_threshold_cm1` | 50 | minimum imaginary-mode magnitude (cm^-1) at an optimized IRC endpoint for a TS–TS link; a single weaker imaginary mode gives an EQ with a warning. Two or more imaginary modes always exclude an EQ. Use a finite nonnegative value; zero follows any endpoint with exactly one imaginary mode. |
| `bifurcation_threshold_cm1` | 50 | how negative the curvature across the path at an exact Hessian of an IRC must be to record a possible bifurcation (cm^-1) |
| `variable_tolerance` | 0.01 | points of plain variables closer than this (in the units of the variables) and within the energy tolerance are the same ([plain variables](#plain-variables)) |
| `channels` | true | dissociation channels (`DC` nodes): an end of an IRC whose molecules are apart in two or more groups is a channel and is not minimized; a minimum of two molecules in contact (an input too) gets the channel of the two and a barrierless edge to it; every kind of fragment is optimized on its own once (`docs/source/mapper_design.md`); in a periodic cell a minimum that came apart in its cell is a channel with its own structure ([above](#a-periodic-cell-fixed-or-variable)); `false`: every end is minimized as before, and such a structure stays a minimum |
| `channel_margin` | 1.0 | Angstrom over the sums of the van der Waals radii (Alvarez) within which two molecules are in contact |
| `method_relax` | `neb` | `afir_ts`: the chain method that relaxes the AFIR path before the guesses are taken (`neb`, `dmf`), or `off` (the maxima of the AFIR path itself) |
| `method_relax_KEY` | | `afir_ts`: a setting of that method (`iterations`, `options`, ...; over `step2_settings`) |
| `method_KEY` | | a setting of a registered method |
| `workers` | 1 | jobs at the same time (`--max_workers` goes over it, it over `n_parallel`) |
| `batch_size` | `workers` | jobs chosen together in the schedule `batch` |
| `schedule` | `batch` | `batch` or `rolling` ([below](#several-jobs-at-the-same-time)) |

**The kinetic priority** (`priority=kinetics`, or MultiOptPy's `--use_rcmc` / `use_rcmc`)
explores, after every job, the minimum with the highest kinetic score in the network as it
stands: by default its population after `kinetics_time` (1 s; `rcmc_reaction_time_s`,
`--rcmc_time`) from the start node (`kinetics_start`, the node of the first input;
`rcmc_start_node_id`, `--rcmc_start_node`) at `kinetics_temperature` (the temperature of the run;
`rcmc_temperature_K`, `--rcmc_temperature`), with the Eyring rates of the electronic energies
and the solver `propagate` (with `use_rcmc`, the published RCMC: `rcmc`). For every job the
screen and `mopt.log` say why its minimum was chosen (`Job 7: chosen by kinetics by rcmc
(Population after 1e0 s): EQ2 6.1e-1, EQ0 3.9e-1, EQ4 2.0e-4`): the solver, the score, and the
highest scores of the minima. `kinetics_score` (`population`, `max_population`, `residence`, `discovery_flux`),
`kinetics_solver` (`rcmc`, `legacy_rcmc` for MultiOptPy's numbers, `expm`, `propagate`),
`kinetics_energy`, `kinetics_clamp`, `kinetics_time_rule`, `kinetics_tolerance`,
`kinetics_rate_theory`, `kinetics_corrections`, and `kinetics_kappa` choose the rest
([kinetics.md](kinetics.md)). Before the first transition state is known the lowest minimum is
explored, as in MultiOptPy. `run_kinetics RUN_DIR` computes the kinetics of a run directory.

### The kinetics at the end of a run

With `-map_opt kinetics=on` the run computes the kinetics of its network when the exploration
ends (by its stop rules, its job limit, or because no minimum has a task left), with the same
`kinetics_*` settings and defaults as the kinetic priority: Eyring rates of the electronic
energies at the temperature of the run, all population in the node of the first input
(`kinetics_start`), the solver `propagate`, and the populations at 13 times, the decades from
`10^-12 kinetics_time` up to `kinetics_time` (1 s). The files go to `kinetics/` of the run
directory: those of `run_kinetics` (`populations.csv`, `rates.csv`, `priority.csv` ranked by
`kinetics_score`, `rcmc_superstates.json`, `settings.json`, `mopt.log`) and
`kinetics_summary.json` (the settings, the populations of every state by node at every time,
the notes). The report has a section "Kinetics" with the populations against time of the states
that reach 1 % (at most eight) and the table of the states at the time of interest; the log of
the run ends with the largest populations. A network without a transition state has no
kinetics, and the summary says so.

An interrupted run (`STOP`, a kill) leaves no kinetics; `--restart RUN_DIR` computes it when the
continued run ends, and a restart of a run that has ended computes it again. `kinetics_*` keys
without `priority=kinetics` or `kinetics=on` are refused. For other settings (free energies,
corrections, sensitivities), `run_kinetics RUN_DIR` reads the same network.

```bash
run_mapper ar7.xyz -cfg config.json -map_opt kinetics=on kinetics_time=1e-6
```

The defaults are MultiOptPy's (`Entrypoints/mapper.py`), so that a settings file of MultiOptPy
gives the same search; the AFIR force of 100 kJ/mol is a usual collision energy of AFIR searches
inside one molecule for organic molecules, and a cluster of noble-gas atoms needs much less (the tests use 2 kJ/mol). The
search settings are neutral: no published protocol is the default by fiat.

The steps take the long option names of `optmain`, as in `run_autots`. The settings that make
the surface (`othersoft`, `electronic_charge`, `spin_multiplicity`, the method and basis set,
`excited_state`, and the others of the calculator) are given in `step1_settings`; steps 3 and 4
take them where they do not set them. All steps must run on the same surface: a run whose steps
differ there is refused with the differences.

```json
{
  "mapper_settings": {"afir_gamma_kJmol": 2.0, "dist_upper_ang": 7.0, "max_iterations": 30},
  "top_n_candidates": 2,
  "step1_settings": {"othersoft": "lj", "NSTEP": 300},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 200},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 400,
    "intrinsic_reaction_coordinates": ["0.5", "200", "lqa"]
  }
}
```

This is the setting of the test of the seven-atom argon cluster (`tests/cli_mapper.rs`, with
`-map_opt priority=tabu`). From the pentagonal bipyramid it finds the four minima (-16.505,
-15.935, -15.593, -15.533 epsilon; the fourth is chiral, so both mirror images are nodes) and
the transition states between them; the fourth minimum came after 21 jobs with `tabu` and after
41 with the default Boltzmann priority, and 100 jobs gave no further minimum (seed 42). Pulls
(`include_negative_gamma`) took the cluster apart into fragments around -9.1 epsilon, a larger
`afir_gamma_kJmol` (10) found the same network. The cluster's
wells are 0.24 kcal/mol deep, so its forces are small even against `-tcc`; with the
quasi-Newton Hessians of the minimizations an IRC end of the capped octahedron did not converge
in 400 iterations, and an end of the third minimum stopped at -15.585 instead of -15.593
epsilon. Exact Hessians in steps 3 and 4 (`calc_exact_hess: 1`, cheap for this potential) gave
every minimum to 1e-4 epsilon.

Several input structures (several files, the frames of one file, or a wildcard) start one
network; each is optimized to a minimum, and inputs that end in the same minimum are one node.
All inputs need the same atoms in the same order and the same charge and multiplicity.

The common settings of every command work as elsewhere: `--config FILE` and
`--dump-config FILE` for the options of the command, `--log_level` for the screen (the log
`mopt.log` of the run directory has every message, those of each job as one block).

### Constraints and bias potentials

The hard constraints (`-fix`, `-pc`, `-gfix`, `-uc`) and the static bias potentials (`-ma`,
`-kp`, `-ubias`, `-wwp`, ...: the options of `optmain`) are options shared by every step: on
the command line of `run_mapper` or in the block `shared_settings` of the settings file,
merged into `step1_settings` to `step4_settings` and `ts_conformer_settings` (a key that a
block sets itself wins, and the log says so; the steps must then still agree, or the run is
refused as one on several surfaces). The constraints are resolved once at the first input
and held in every stage of every job; the biases are part of the calculator of the run, under
the mapper's own AFIR pushes. Every node has the energy of the biased surface and, beside it,
the energy of the calculator alone (`bare_energy_hartree` in the network file, a column of
`nodes.csv`, both in the tables of the report), and the report names the constraints and
biases. What cannot hold them is refused by name (dissociation channels are off, and
`channels=on` is refused; a variable cell with constraints; the ONIOM core; plain variables;
`method=nanoreactor`; `entrance_structures`; `higher_saddles`; `method_relax=dmf`; several
states). The rules are in [mapper_design.md](mapper_design.md); the checks in
`tests/constraints_matrix_mapper.rs` and `tests/bias_matrix_mapper.rs`. With the built-in
GFN-FF the run writes `gfnff_topology.xyz`, the structure its topology comes from: `-calc_opt
topology=RUN_DIR/gfnff_topology.xyz` gives a command that starts from a result of the run
the same force field.

## Several jobs at the same time

`-map_opt workers=N` (MultiOptPy's `--max_workers N` or `mapper_settings.n_parallel`) runs up
to `N` jobs at the same time, each in a thread of the command with an `N`-th of the threads of
the command (`-core`, `N_THREAD` of `step1_settings`, 8 by default) for what it computes in
parallel:

```console
run_mapper ar7.xyz -cfg config.json --max_workers 4
run_mapper ar7.xyz -cfg config.json -map_opt workers=4 batch_size=8
```

- **The schedule `batch`** (the default) chooses `batch_size` jobs together from the network as
  it is (each job sees the ones chosen before it as running: their tasks are not offered again,
  and they count as visits of their minima), runs them, and when all of them have ended tells
  the network of them in the order of their numbers; then it chooses the next batch. The
  network therefore follows from the settings, the seed, and `batch_size` alone: not from the
  number of workers, not from which job ends first, and not from a restart. `batch_size` is the
  number of workers when it is not given, so that every worker has a job; a run with
  `batch_size=4` and one worker has the network of a run with four. A batch is not the same as
  jobs one after the other (a job of a batch is chosen without the results of the others), so
  `batch_size` above 1 gives another network than one job at a time.
- **The schedule `rolling`** chooses a job whenever a worker is free, from the network with
  the results that have come in by then, and tells every result as it comes, as MultiOptPy's
  mapper does with `--max_workers`. No worker waits for the slowest job of a batch, but the
  network depends on the order in which the jobs end, that is on the timing of the run: two
  runs with the same settings can differ.
- **The calculator.** The jobs share the calculator of the run where it can be shared; a
  calculator of the Python side then gets a copy per job in worker processes (`-parallel N`,
  unless `-parallel` is given). A calculator that evaluates in order (it follows excited states
  from structure to structure, `-es_follow`) is never used by two jobs: every job gets one of
  its own from the registry of the run, also with one worker, so that what one job evaluated
  does not change what another gets. A program that works in a fixed directory cannot run two
  copies of itself in one process; with such a program and `-es_follow`, keep one worker.
- **The logs.** What a job says (its optimizations, transition-state searches, and IRCs, what
  the calculator warns of or prints, the transition states it found known, its warnings) is
  kept for the job and goes to `mopt.log` as one block, headed `Job N (from EQk, task ...)`
  and closed by the line of what was kept of the job, when the job is kept: in the order of the
  jobs, as one worker writes them, whatever order the jobs end in (a block of the rolling
  schedule waits for those of the jobs before it), each block with one write, after the
  network of its batch is stored. A batch that is stopped writes no block, so that a stopped
  and restarted run has the blocks of an uninterrupted one in the same order, between its
  notes of the restart. The screen gets a line when a job starts (`Job N started: ...`) and
  ends (`Job N done after ...`), the warnings of the jobs, and the lines of the run (the
  batches, the end, the evaluations); the thread of a job writes nothing to the screen or to
  `mopt.log`. Every job also writes its lines to `jobs/job_000003.log` as it runs, to follow
  it. Each job has its own directory of stage trajectories, as with one worker.
- **`worker_timeout_s`** (seconds, MultiOptPy's time limit of the worker process of a job):
  a job that runs longer is stopped at its next step and recorded as failed, with the reason in
  the network and the warnings; the run goes on. A job is stopped between two evaluations, so a
  single evaluation that never returns is not ended by it. The time limit makes the network
  depend on the speed of the machine; leave it out unless a job can run away.
- **A job that fails or panics** is recorded as failed alone; the other jobs and the run go on.

Measured with the built-in `lj` on the clusters of seven and thirteen argon atoms (the settings
of the test above, `--max_iter 24`, seed 42; LJ13 with `--afir_gamma 10`), a release build on a
machine with 16 threads that ran other work at the same time (load 3 to 5), the run directory
in memory; wall time of two runs, evaluations, and the network:

| System | Threads (`N_THREAD`) | Schedule | 1 worker | 2 workers | 4 workers | Gradients, Hessians | Network |
|---|---|---|---|---|---|---|---|
| LJ7 | 8 (default) | one job at a time | 5.27, 5.07 s | | | 3178, 2548 | 3 minima, 4 TS |
| LJ7 | 8 | `batch_size=4` | 5.35, 5.20 s | 1.41, 1.31 s | 0.91, 0.98 s | 3321, 2769 | 3 minima, 4 TS, the same with 1, 2, 4 workers |
| LJ7 | 8 | rolling | | | 0.70, 0.66 s | 3159, 2606 | 3 minima, 3 TS; the two runs took other jobs |
| LJ7 | 1 | one job at a time | 1.20, 1.25 s | | | 3178, 2548 | as with 8 threads |
| LJ7 | 1 | `batch_size=4` | 1.59, 1.53 s | 1.09, 1.10 s | 0.93, 0.81 s | 3321, 2769 | the same with 1, 2, 4 workers |
| LJ7 | 1 | rolling | | | 0.55, 0.60 s | 3159, 3358 | 3 TS and 4 TS in the two runs |
| LJ13 | 8 | one job at a time | 13.54, 13.59 s | | | 6070, 4769 | 1 minimum |
| LJ13 | 8 | `batch_size=4` | 13.41, 13.56 s | 3.02, 3.05 s | 1.41, 1.42 s | 6070, 4769 | 1 minimum, the same jobs with 1, 2, 4 workers |
| LJ13 | 8 | rolling | | | 1.34, 0.97 s | 6070, 4769 | 1 minimum |
| LJ13 | 1 | one job at a time | 2.99, 3.04 s | | | 6070, 4769 | 1 minimum |
| LJ13 | 1 | `batch_size=4` | 2.83, 2.98 s | 1.87, 1.80 s | 1.13, 1.20 s | 6070, 4769 | the same with 1, 2, 4 workers |
| LJ13 | 1 | rolling | | | 0.89, 0.94 s | 6070, 4769 | 1 minimum |

- **The schedule `batch` gives the same network with any number of workers**: the networks of
  1, 2, and 4 workers with `batch_size=4` had the same jobs, nodes, and edges, the energies equal
  to 0.0 Hartree, and the same evaluations, in both runs. A batch of four is another network
  than one job at a time (3321 against 3178 gradients for LJ7, the same three minima). The
  rolling schedule took other jobs in two runs of the same settings.
- **The gain on these cheap surfaces.** With one thread per calculation, four workers took
  1.8 (LJ7) to 2.5 (LJ13) times less time than one worker with the same batches, and the
  rolling schedule 1.3 to 1.5 times less again (no worker waits for the slowest job of a batch). What is left is the
  work of the loop between the batches (the network, the files, the report) and the slowest job
  of every batch. With the default of 8 threads one job at a time was 4 to 5 times slower than
  with 1 thread: the linear algebra of these small optimizations spent its time waking the
  threads of the pool; with several workers every job has its share of them (8 / workers), which
  is why 2 workers took 4 times less time there. A calculation that is expensive (a quantum
  chemistry program) gains up to the number of workers, as long as the machine has the threads
  and the memory for that many calculations at once.
- **The default stays one worker**, as in MultiOptPy: the batch size follows the number of
  workers, so a default of several workers would give another network than MultiOptPy's settings
  give, and every worker runs a calculation of its own, so a default of several would multiply
  the threads and the memory a run takes without being asked. Give `workers` (and `batch_size`,
  to have the same network on another machine) where the calculations can run side by side.

Before the threads of the pool were shared out, every job of two workers woke the 16 threads of
the global pool: 2 workers took twice as long as one (30.8 against 14.5 s for LJ13) with 97 s of
processor time for 12 s of wall time. The run directory on the disk adds the flush of the
journal after every evaluation, one at a time under the lock of the journal (about 4 ms each on
the machine of the measurements): LJ7 on the disk took 13.9 and 14.2 s one job at a time, and
with `batch_size=4` 13.7 s with one worker, 25.5 s with two, 8.5 s with four, and 7.8 s rolling
(one run each, with the times on the disk varying by up to a factor of two between runs). A
surface this cheap is bound by the journal; an expensive one is not.

## Files

Everything of a run is in its run directory, under fixed names:

| File | Content |
|---|---|
| `reaction_network.json` | the network (see below), written after every batch of jobs that is kept (after every job when one job runs at a time) |
| `nodes.csv`, `ts_edges.csv`, `crossing_edges.csv`, `dissociation_edges.csv` | the nodes and edges as tables (`dissociation_edges.csv`: the complex, its channel, the energy of the channel above it, the fragments) |
| `structures/EQ000000.xyz`, `structures/TS000004.xyz` | the structure of every node (minima and transition states are numbered apart: EQ0, EQ1, ... and TS0, TS1, ...) |
| `paths/TS000004_irc.xyz` | the way of a transition state from minimum to minimum: the minimization of the backward end of its IRC reversed (`backward minimization`), the backward side reversed, the transition state (`TS`), the forward side, and the minimization of the forward end (`forward minimization`); a side that ends at a dissociation channel or at another saddle point has no minimization |
| `paths/MECP000000_edge0_path.xyz` | the way between the two minima of a crossing point (`states`): the minimum of the first state, the meta-IRC of that state up to the crossing point, the crossing point, the meta-IRC of the second state down to its minimum; each frame is labelled with its part and has the energy of its state. The report plays it in the panel of the crossing point |
| `jobs/input_000_traj.xyz` | trajectory of one input optimization |
| `jobs/job_000003/trajectories.csv` | ordered list of stage trajectory files in this job directory. AFIR, each TS refinement, each IRC branch and each endpoint minimization have separate trajectories. NEB evaluation trajectories are split by band iteration. The corresponding `jobs[].trajectory` entry points to this index. |
| `exploration_history.log` | one row per completed job in the checkpoint, including the starting node, method, task, pair or parameters, `DONE` or `FAILED` with its reason, and counts of new nodes and TS connections. A completed search can find zero new connections. Restarting preserves previous rows. |
| `queue_priority.log` | latest dispatch decision, in the legacy mapper's queue log format. Each eligible node has its priority, pending task count, visits, energy and selection flag. Boltzmann values are relative weights. Kinetics uses its actual scores, or negative energy when it falls back to the lowest minimum. Rules without a numerical score show `n/a`. |
| `rcmc_K_contracted.csv` | latest RCMC queue contraction in the run directory. Each RCMC decision replaces the matrix, superstate membership and transient EQ populations at the requested time. The header identifies the dispatch iteration. Available with `--use_rcmc` or kinetic priority using `rcmc` or `legacy_rcmc`, once a transition connects known minima. |
| `jobs/index.html`, `jobs/job_000003/job_report.html` | live index and per-job pages with the AutoTS structure viewer, stage energy profiles and links to complete trajectories. Each NEB iteration has its own band profile. Use `run_mapper --report RUN_DIR` to regenerate these pages from the saved stages. |
| `jobs/job_000003/steps.csv` | one row per frame across the ordered stage trajectories, numbered from 1, with the stage label and energy in Hartree. Contains the AFIR optimization, pair-search evaluations, TS optimizations and IRC branches as they run. Replaying an interrupted job starts its table and stage index together. |
| `jobs/job_000003/<search>_001/` | NEB energy, force and path-length tables and `path_ITR_<iteration>_neb/` files as in AutoTS. Each variable-cell image has its own lattice. Repeated searches keep separate numbered directories. |
| `jobs/job_000003.log` | the log of a job as it runs: its start, the optimizations and IRCs it ran, its end |
| `jobs/job_000003/` | with `addf` and `nanoreactor`: the files of the searches of the job and their full logs (`addf.log`, `addf_paths.csv`, `ADD_<k>_trajectory.xyz`, `TSpoints.xyz`; the files of the two-point method of a reaction) |
| `mapper_report.html` | the report, written after jobs (see "The report") |
| `remap.csv` | with `-remap`: what every minimum, transition state, crossing point and saddle point of higher order of the first network became in this run |
| `single_points.csv` | with `-single_points`: the energies of every node and fragment on both surfaces, and the gap of the two states at every crossing point |
| `path_peaks.csv`, `structures/PP000000.xyz`, `paths/PP000000_path.xyz` | the path peaks (`path_peaks`), where the run saved any: the table of the records, the structure of each, and the path between its two minima. Records beside the network |
| `merge.csv` | with `-merge`: what every node of every run is in the merged network |
| `steer.log` | every line of `steer.txt` that the run read, and what became of it |
| `reaction_network.sqlite` | with `-map_opt store=sqlite`: the network as an SQLite database (see "The database") |
| `mopt.log` | the log of the run: the lines of the run and the block of every job that is kept, in the order of the jobs |
| `settings.json`, `config_snapshot.json` | the options and the settings file of the run |
| `evaluations.journal` | every evaluation, for the restart |

`reaction_network.json` has `format` and `version` (the layout version, now 1), the units
(Hartree, Bohr, cm^-1), the elements, the inputs, and:

- `nodes`: every minimum (`EQ`), transition state (`TS`), and crossing point (`MECP`, `MECI`,
  with the pair of states), with its level (1: exploration, 2: refinement), its energy, its
  free energy and frequencies where they are known (a minimum from the Hessian that verified
  it, a transition state from the Hessian of the start check of its IRC, the imaginary
  wavenumber negative), its electronic state (charge, multiplicity,
  excited state), where it came from (the input of its lineage, the job, the node that job
  started from), and its coordinates;
- `ts_edges`: the transition state, the two minima of its IRC, the barriers from both (raw
  energy differences), the imaginary wavenumber, and the IRC as data;
- `open_ts_edges` stores a transition state when an IRC endpoint remains unresolved after
  minimization, vibration analysis, or a linked saddle search. Each resolved side retains
  its node and barrier. An unresolved side has `null` for both. Kinetics excludes the edge.
- `crossing_edges`: the crossing point, the minima on its two sides, and the gradient
  difference;
- `jobs`: every finished job with the task it ran and what it found; `warnings`: what did not
  converge or is not what it was searched for; `evaluations`: the totals of the run.

The edges keep raw quantities, no rate constants: rates are computed afterwards by a chosen
model (`run_kinetics`, [kinetics.md](kinetics.md); the theory in
[kinetics_beyond_tst.md](kinetics_beyond_tst.md)).

## The database

With `-map_opt store=sqlite` the network is also written, whole after every finished job, as
the SQLite database `reaction_network.sqlite`. It opens in any SQLite tool (Python's `sqlite3`,
the `sqlite3` shell, DB Browser for SQLite); mopt-rs writes it without a library. Layout
version 1 (ids are those of the network, from 0; energies in Hartree, coordinates in Bohr):

| Table | Columns |
|---|---|
| `format` | `key`, `value`: format, version, mopt_rs_version, units, run (the state of the run as JSON), and, where the network has them, entrance (the entrance channel), steering (what `steer.txt` changed) and variables (the names of the variables of a network of plain variables), as JSON |
| `elements` | `id` (the atom, from 0), `symbol` |
| `inputs` | `id`, `file`, `frame`, `node`, `goal` (1 for the goal structure) |
| `nodes` | `id`, `kind` (EQ, TS, MECP, MECI), `number`, `label`, `level`, `energy_hartree`, `free_energy_hartree`, `charge`, `multiplicity`, `excited_state`, `input`, `job`, `from_node`, `excluded`, `coords_bohr`, `json` |
| `ts_edges` | `id`, `ts`, `end_1`, `end_2`, `level`, `barrier_1_hartree`, `barrier_2_hartree`, `imaginary_wavenumber_cm1`, `job`, `irc_file`, `json` |
| `irc_sides`, `irc_points` | the sides of the IRCs (`edge`, `k`, `side`, `end_reason`) and their points (`edge`, `side`, `k`, `arc_length`, `energy_hartree`) |
| `irc_transverse` | the lowest curvature across the path at the exact Hessians of the IRCs: `edge`, `side`, `k`, `arc_length`, `wavenumber_cm1` |
| `path_peaks` | the path peaks (records beside the network): `id`, `label`, `level`, `energy_hartree`, `end_1`, `end_2`, `barrier_1_hartree`, `barrier_2_hartree`, `method`, `points`, `converged`, `searches`, `job`, `json` (with the structure) |
| `crossing_edges` | `id`, `crossing`, `end_1`, `end_2`, `level`, `barrier_1_hartree`, `barrier_2_hartree`, `gradient_difference`, `job`, `json` |
| `open_ts_edges` | the transition states with an end of the IRC that is not known: `id`, `ts`, `end_1`, `end_2` (empty where unknown), `level`, `barrier_1_hartree`, `barrier_2_hartree`, `imaginary_wavenumber_cm1`, `job`, `json` (with the IRC) |
| `jobs` | `id`, `node`, `method`, `task`, `description`, `status`, `error`, `found_new`, `found_lower`, `trajectory`, `json` |
| `warnings` | `id`, `job`, `message` |
| `evaluations` | `id`, `energies`, `gradients`, `hessians` |

`coords_bohr` holds the coordinates as little-endian 64-bit floats (x1, y1, z1, x2, ...); `json`
holds the whole entry of `reaction_network.json` (without the coordinates and the IRC points),
from which mopt-rs reads the network back exactly. For example, in Python:

```python
import sqlite3
import numpy as np

con = sqlite3.connect("mapper_output/reaction_network.sqlite")
for label, e in con.execute(
    "SELECT label, energy_hartree FROM nodes WHERE kind = 'EQ' ORDER BY energy_hartree LIMIT 5"
):
    print(label, e)
blob, = con.execute("SELECT coords_bohr FROM nodes WHERE label = 'TS0'").fetchone()
coords = np.frombuffer(blob, "<f8").reshape(-1, 3)
```

`run_mapper --report reaction_network.sqlite` writes the report from the database (next to it,
with the files of the run directory where they are there).

## The report

`mapper_report.html` is one file that loads nothing from elsewhere (light and dark themes). It
is written after the inputs, after jobs, and at the end; after a job only when writing it has
taken less than a twentieth of the time since the last one, so that the report costs at most
about 5 % of a run. `run_mapper --report RUN_DIR` writes it again from the files of a run that
ended, was stopped, or is still running, without computing anything; the settings of the page
are those of the run (`-report_opt` in `settings.json`) overridden by `-report_opt` of the
command line.

The page has:

- four figures of the network, one at a time:
  - **Map**: every structure at its place by multidimensional scaling of the mass-weighted
    distances between the structures (ReSPer). Any two of the first four principal coordinates
    can be shown, each with its share of the spread; when the two shown hold less than half of
    it, the page warns that structures far apart can lie close on the map. Above `map_exact`
    structures (1000) it is landmark scaling (de Silva and Tenenbaum 2004): classical scaling of
    `map_landmarks` structures (400) spread over the energies, every other structure placed by
    Gower's formula from its distances to them.
  - **Graph**: the minima as nodes and every link as an edge through its transition state,
    placed by a force-directed layout (Fruchterman and Reingold 1991, with a grid of cells so
    that a step costs O(n)); a minimum can be dragged and its transition states follow.
  - **Disconnectivity graph** (Becker and Karplus 1997) of the lowest `figure_minima` minima
    (300), with the merges of the whole network.
  - **Energy diagram** of the same minima.

  - **Circle**: the minima on a circle and every link as a chord through its transition
    state. With a lowest-barrier path chosen, its start is at the left, its end at the right
    and its minima along the upper half in their order; the other minima are along the lower
    half, each under the minimum of the path it is joined to in the fewest steps. Without a
    path the minima follow round the circle in the order of their steps from the lowest.
    With several species, "A circle per species" draws every species as a circle of its
    conformers (its size grows with their number, its name `S1`, `S2`, ... above it), the
    circles round a ring, the lowest species at the left: the links inside a species are
    chords of its circle, and the links between two species are lines from circle to circle.

  The map, the graph and the circle are drawn on a canvas with every node (zoom with Ctrl +
  wheel, a pinch, the buttons or the keys; drag to move; double-click to fit), sized to the
  window, so that no ratio of the window cuts them off; "Full screen" shows one over the
  whole screen with the panel of the chosen node at its right (its energies, links and
  structure, as in the page), where it is moved and zoomed as in the page and the wheel zooms
  without Ctrl; where the browser does not allow it (a page inside another page), the figure
  fills the window instead, and Esc or the button leaves it (the graph of the species has
  the same button). A click on a link chooses its transition state or crossing point, as a
  click on its node. With several species, "Species"
  above the figures shows the conformers of one species alone with the links between them (in
  the map, the graph and the circle), and "in the network" in the table of the species does
  the same. The disconnectivity graph and the energy diagram are then those of the conformers
  of that species (energies still above the lowest minimum of the whole network); the page
  holds them for the 12 lowest species, and shows those of the whole network for the others.
- the node chosen in any figure or table: its energies, population, the highest point on the
  way from the lowest minimum, its links, its structure in a viewer, and for a transition state
  the energy along its IRC and the IRC path as frames;
- the lowest-barrier path between two minima: the path of the minimum spanning tree with the
  energies of the transition states as weights, whose highest point is the lowest of all paths,
  drawn along the IRCs of its steps; "Play the path" shows its structures as one animation
  in the page and "Download the path" saves them as one xyz file: the start, then for every
  step the frames of its path file (turned round where the step goes against it) and the
  minimum it ends at, each piece laid on the frame before it by the best rotation, since the
  paths of different jobs are in different frames, and atoms of one element matched where two
  jobs numbered equivalent atoms differently. An IRC stops before its minimum, so the path
  file of a transition state goes on through the minimization of each end, and the pieces
  meet at the minima. Where two pieces still do not meet within 0.12 Angstrom (a path file
  written before the minimizations were in it; one minimum as two jobs found it, which may
  differ by up to the threshold of the same structure), frames on a straight line between
  them are put in, at most 15 at a join, so that the path is one motion: the page says so in
  a warning with their number, and the title of each says `interpolated (not computed)`.
  They are no part of a reaction path. (On a path of 32 steps of the urea synthesis above,
  from path files without the minimizations, an atom moved by more than 0.8 Angstrom at 6 of
  the 64 joins, and no renumbering of the atoms made these smaller.)
- the tables of the minima (Boltzmann populations, highest point on the way from the lowest),
  the transition states and the crossing points, paged and filtered by the script; a crossing
  point has the way between its two minima as an animation in its panel (the frames named
  after their part: the minimization and the meta-IRC of the first state, the crossing point,
  the meta-IRC and the minimization of the second state);
- the chemical species: the minima grouped by their SMILES strings, so that the conformers of
  one species stand together (their number, the lowest of them, the sum of their
  populations), and for every two species that transition states join, the lowest of those
  transition states with its height above the lowest conformer of each species and the number
  of transition states between the two, and a graph with a node per species and an edge per
  two species that transition states join (see "Chemical species" below);
- the warnings of the run, the jobs, the settings, and the evaluations of the run: gradient
  calculations, exact Hessian calculations, energies alone, and the microiterations of an
  ONIOM environment (or that none was relaxed); `evaluations` of `reaction_network.json` keeps
  them (`micro_relaxations`, `micro_iterations`).

Every transition state and crossing point in the energy window is shown, also one that joins no
two minima shown. The page holds the structure of every node and the path of every
transition state and crossing point (at most 30 frames each), so that each can be shown in
its panel and saved wherever the page is opened: those of the lowest `structures` nodes (400,
at most `structure_atoms` atoms in all) and the paths of the lowest transition states
(`irc_lines` atom lines in all) as text, and all the others packed, unpacked by the browser
when a node is chosen or a file saved. What is packed is written compactly before it is
compressed (the elements once, the coordinates as whole numbers of 10^-5 Angstrom, every frame
of a path as its difference from the frame before): about 9 bytes of the page per atom
line, a third of the text (the page of the urea run of 1000 jobs above, with 841 structures and
581 paths, is 6.3 MB). `structure_packed_atoms` and `irc_packed_lines` limit what is packed
(a number of atoms and of atom lines; `0`: nothing is packed, and what the page then does not
hold is marked with a dash). A packed structure has five decimals. The page has no links to
the files of the run directory, so that it can be moved or sent on its own.
The pages of `run_autots` and `run_automecp` hold all their structures in the same way.
Without scripts, the page keeps the figures and tables of the lowest nodes as drawn by mopt-rs.

### Chemical species

A network of a flexible molecule has many minima that are conformers of one compound. The page
groups the minima by a SMILES string written from each structure, and `nodes.csv` has the
string of every minimum in its column `smiles`:

1. the bonds are those of the identity test of the mapper (within 1.2 times the sums of the
   covalent radii); the connected parts are the molecules, joined by `.` in the string;
2. bond orders come from the usual valences (H and the halogens 1, B 3, C and Si 4, N and P 3
   or 5, O 2, S 2, 4 or 6): bonds between two atoms that lack one are raised, a maximum
   matching at a time, and N, P, S take two more bonds from terminal O or S atoms that still
   lack one (nitro, sulfone, phosphate);
3. an atom with one bond more than its valence is the usual ion (four bonds of N or P and
   three of O or S: +1; four of B or Al: -1), and what is left of the charge of the structure
   goes to atoms that still lack a bond (negative: O, S, N, the halogens, C; positive: C, Si,
   H, N, P, S, O); an atom that lacks bonds after that is written as a radical;
4. a tetrahedral centre of four different neighbours gets its mark (`@`, `@@`) from the
   coordinates, in the order its neighbours are written in;
5. the string is canonical: the heavy atoms are ordered by the colours of a Weisfeiler-Lehman
   refinement, every atom of a colour that several share is tried as the first of them, and
   the smallest string of all orders is taken (at most 720 orders per molecule).

The graph of the species has a node for every species, larger with more conformers and
coloured by the energy of its lowest conformer, and an edge between two species that
transition states join, with the energy of the lowest of them; a node chooses the lowest
conformer of its species and an edge its transition state. It is the network with the
conformers of every species drawn together (at most the 60 lowest species); a node of
the graph is dragged to another place, and its edges follow; a drag that starts elsewhere
moves the view, Ctrl + wheel zooms (the wheel alone over the whole screen), and a double
click shows the whole graph again.

A tetrahedral centre with four different substituents (four neighbours that the bond graph
tells apart) has `@` or `@@` from the coordinates, so that R and S have different strings and
two enantiomers are two species; a molecule that is not chiral has one string for a structure
and its mirror image. A double bond outside every ring whose two atoms each have one other
neighbour, or two that the bond graph tells apart, has `/` and `\` from the dihedral angle
about it, so that E and Z isomers have different strings and are two species (E-difluoroethene
`C(/F)=C\F`, Z `C(/F)=C/F`); the fragments of a dissociation channel are told apart the same
way. An allene, a ketene, and a double bond twisted to within 12 degrees of a right angle have
no mark. The string does not have the configuration of double bonds in rings, centres whose substituents differ only in their own configuration, centres with a
lone pair (sulfoxides, phosphines), or aromatic symbols (a ring is in a Kekule form), and its
bond orders are made without a calculation of the electrons: where the valence rules do not hold (metal complexes, unusual charges) it still
tells different bond graphs apart, and a bond order is a guess. Transition states and crossing
points have no string (their bonds are between two species). `workflow::explore::smiles` is
the function, usable on its own.

A reaction at a stereocentre, as a check: the identity SN2 reaction Cl- + CH3CHFCl with the
built-in GFN1-xTB, from the complex of the chloride with the S molecule to the complex after
the substitution (`run_mapper reactant.xyz -map_opt goal=product.xyz method=connect`, charge
-1). One job finds the transition state, 26.36 kcal/mol above both complexes (612.8i
cm^-1; the two barriers are equal, as the reaction is its own reverse), and the page has two
species, `C[C@@H](F)Cl.[Cl-]` (S) and `C[C@H](F)Cl.[Cl-]` (R): the substitution inverts the
carbon, and the product is the mirror image of the reactant. The two complexes have two
dissociation channels, since the fragments of a channel are told apart by their handedness
too (`mirror=true` makes a structure and its mirror image one minimum and one fragment).

With two centres every centre has its mark, and the string is the smallest over the ways of
writing a molecule whose atoms are equivalent, so that a meso form and its mirror image have
one string:

| Molecule | Strings |
|---|---|
| CHFCl-CHFCl | meso `[C@@H](Cl)([C@H](Cl)F)F`; the pair of enantiomers `[C@@H](Cl)([C@@H](Cl)F)F` and `[C@H](Cl)([C@H](Cl)F)F` |
| CHFCl-CHFBr (no meso form) | `[C@@H](F)([C@@H](Cl)F)Br` and its mirror image `[C@H](F)([C@H](Cl)F)Br`; `[C@H](F)([C@@H](Cl)F)Br` and its mirror image `[C@@H](F)([C@H](Cl)F)Br` |
| 1,2-dichlorocyclopropane | cis (meso) `C1[C@@H](Cl)[C@H]1Cl`; trans `C1[C@H](Cl)[C@H]1Cl` and `C1[C@@H](Cl)[C@@H]1Cl` |

The conformers of each have its string, and so has every numbering of its atoms (the tests
of `workflow::explore::smiles`). As fragments of a dissociation channel, a meso form and its
mirror image are one fragment and two enantiomers two.

The settings of the page (`-report_opt KEY=VALUE`): `unit` (kcal, kj, ev, hartree),
`temperature` (298.15 K), `energy_window` (none), `max_minima` (all), `figure_minima`,
`table_rows` (300), `structures`, `structure_atoms`, `structure_packed_atoms`, `irc_lines`,
`irc_packed_lines`,
`profile_points`,
`map_exact`, `map_landmarks`, `file` (the name of the page in the run directory).

Size and time, measured with made-up networks of 12-atom structures
(`examples/bench_mapper_report.rs`, release build):

| Minima | Nodes | Links | Page | Size |
|---|---|---|---|---|
| 1,000 | 2,333 | 1,333 | 0.8 s | 1.2 MB |
| 5,000 | 11,666 | 6,666 | 1.6 s | 2.0 MB |
| 20,000 | 46,666 | 26,666 | 5.5 s | 5.3 MB |

**Files that lack something.** `--report` writes the page whatever the files of the run lack: a
network file that is damaged is read as far as it goes (a node, edge, or job that cannot be read
is left out, the others are renumbered around it, and so is an edge whose node is left out); a
network file that is missing or not JSON is replaced by the structures of `structures/` and the
links of `ts_edges.csv` (without the energies along the IRCs); settings that cannot be read, a
missing settings file, a calculator that is not recorded, and IRC files that are missing are
left out. Every such thing is listed under "Missing information" at the top of the page and on
the screen. It fails only when the path is not a directory or the page cannot be written.
During a run, a page that cannot be written is a warning of the log and the run goes on.

## Interruption and restart

A `STOP` file in the run directory stops a run within a job; MultiOptPy's `stop.txt` stops it
before the next job. `run_mapper --restart RUN_DIR` reads the settings, the network, and the
journal of the run directory and goes on: the finished jobs are not run again, the job that was
stopped is run again, and its evaluations come back from the journal. The choices of the run
follow from the seed and the number of the job, so that the restarted run ends with the
network of a run that was never stopped: the same jobs, nodes, and edges, with the same
numbers. The evaluations of every job are recorded in the journal under the label of the job
and answer that job alone: the journal answers a request with a record of a structure within
1e-9 Bohr of it, and with one label for all jobs a restarted run could take the record of
another job whose search had converged to the same stationary point (two Newton searches end
within 1e-10 Bohr of each other, at structures that are not the same), which ended a search a
step earlier or later than in the run that was not stopped (found 2026-11-19 on the argon
cluster of the tests, where the next search then went to another stationary point). The totals of gradient and exact-Hessian evaluations
are given at the end of the screen output and the log, and in the network file over all
sessions of the run.

With several jobs at the same time the unit is the batch:

- **`STOP` within a batch** stops every job of the batch at its next step, and nothing of the
  batch is kept, also of the jobs that had ended: the network file is that after the last batch
  that was kept, and the restart chooses the same batch again (the network and the seeds are
  the same) and gets the evaluations of its jobs back from the journal.
- **`STOP` while a batch is kept** (between the jobs of the batch that are told, or while the
  report is written) keeps the batch whole and stops before the next one.
- **`stop.txt`** lets the jobs that run end, keeps them, and stops before the next batch (in
  the rolling schedule: before the next job, the jobs in flight are kept as they end).
- **A killed run** (the process killed at any moment) leaves files that are each whole or
  absent: every file of the network, the tables, the structures, the IRC paths, the report,
  and the settings is written next to its place and moved there, the settings first, and the
  journal drops a record that was cut off. A restart goes on from the network file, the network
  after the last batch that was kept, and removes what the killed session wrote beyond it (the
  structures and IRC paths of nodes the network does not have, and files that were being
  written). The evaluations the killed session made after it last wrote the network come back
  from the journal and are not computed again; the network file says how many records of the
  journal its totals count (`run.journal_records`), and the restart counts the records after
  them, so that every evaluation that was computed is counted once. (With a Hessian made from
  gradients, `o1numhess`, one record stands for several gradients, and those of a session that
  was killed are not counted.)
- **The trajectory and the log of a job** grow frame by frame and line by line (each written
  at once) and are written anew when a restart runs the job again.

In the rolling schedule a job that was stopped is dropped alone: the jobs that had ended are
kept, the restart chooses again from the network as it is, and the numbers of dropped jobs
below the last kept one are not given again (their trajectories and logs stay in `jobs/`).

The tests stop runs before a batch, with every job of a batch in flight, with some ended and
some in flight, while a batch is kept and while the report is written, a second time during
the restart, and while a job reaches its time limit; they end a run with an error in the
middle of keeping a batch (what a kill there leaves), and kill the program itself (SIGKILL) at
times spread over a run, when a job of a batch has ended, and while a batch is kept, the
restart killed again at some of the points (`tests/cli_mapper_parallel.rs`,
`tests/cli_mapper_kill.rs`, the unit tests of `workflow::mapper::driver`; with a release build
and `MOPT_MAPPER_KILL_POINTS=20`, 46 kills: for one worker 20 at random times and 2 while a job
was kept, for three workers 20 at random times, 2 while a batch was kept, and 2 when a job of a
batch had ended, the restart killed again at every second point). Every restart ends
with the network of the run that was not stopped, the blocks of its log whole and in order, and with totals of evaluations equal to the
records of the journal (every evaluation that was computed, once). After a `STOP` the totals
are those of the run that was not stopped (2269 gradients and 801 exact Hessians for each of
the three stops of the test with three workers, 2026-11-19). Before the evaluations of every
job had their own label in the journal they differed by a few either way (-1 to +7 of 1253),
because a request of one job could be answered by the record of another job.

### Steering a run: `steer.txt`

A run that goes on reads the file `steer.txt` of its run directory before every choice of jobs,
applies its lines, and removes it:

| Line | What it does |
|---|---|
| `explore EQ5` | the next jobs are the open tasks of that minimum, before the priority chooses again (several lines: in their order) |
| `skip EQ5` | no more jobs start from it |
| `unskip EQ5` | a minimum that a `skip` line left out is explored again |
| `max_jobs N` | the limit of jobs from now on (0: none) |

A minimum is named by its label or by its number in the network; what follows a `#` is a
comment. A line that cannot be read is told in the log and changes nothing. What became of
every line is written to `steer.log` with the number of jobs told by then, and what was
applied is kept in the network file (`steering`), so that a restart goes on with it. With
several jobs at a time the file is read between batches. A run that was steered is no longer
the run its settings alone give.

## Adding parts in Rust

Every part is a trait with a registry; `run_with_parts` of `cli::mapper` takes the registries
and further observers.

```rust
use std::sync::Arc;
use mopt_rs::cli::mapper::{run_with_parts, MapperParts};
use mopt_rs::workflow::mapper::{
    ExplorationMethod, Job, JobContext, JobOutcome, MethodSettings, Network, Node, Task,
};

struct MyMethod;

impl ExplorationMethod for MyMethod {
    fn name(&self) -> &str {
        "my_method"
    }
    fn tasks(&self, node: &Node, _network: &Network) -> mopt_rs::error::MoptResult<Vec<Task>> {
        // The tasks of a node, the same list in every run with the same settings.
        Ok(Vec::new())
    }
    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> mopt_rs::error::MoptResult<JobOutcome> {
        // ctx.jobs.optimize(...), ctx.jobs.irc(...), ctx.jobs.pair_search(...) (a method of
        // the registry of two-point methods) and ctx.jobs.dynamics(...) (molecular dynamics
        // with bias potentials) on the surface of the run; follow_guesses(job, ctx, ...)
        // optimizes transition-state guesses, runs the IRCs and optimizes their ends; the
        // reactions found go into the outcome, and the driver adds what is new.
        Ok(JobOutcome::default())
    }
}

let mut parts = MapperParts::default();
parts.methods.register(
    "my_method",
    Arc::new(|_: &MethodSettings| Ok(Box::new(MyMethod) as Box<dyn ExplorationMethod>)),
);
// run_with_parts(&config, base, console, &registry, &parts, &mut [])  with method=my_method
```

- **A method that only proposes**: implement `workflow::mapper::plugin::Proposer` (`tasks`,
  and `propose`, which returns guesses and first paths) and wrap it in `ProposalMethod`, which
  relaxes the paths and follows the guesses; `parts.methods.register_prefix("mine:", ...)`
  makes the method of every name with that prefix (the whole name is handed over), as the
  Python package does for `python:` (`tests/cli_mapper_plugin.rs`).
- **Priority**: implement `ExplorationPriority` (`select` returns the candidate to explore, or
  `None` to end the search) and register it in `parts.priorities`; any `SamplingPriority` of
  `workflow::explore` is one through `SamplingAdapter`.
- **Stop rule**: implement `workflow::explore::StopRule` and register a factory in
  `parts.stops` (`NAME:PARAMETERS` in `-map_opt stop=...`).
- **Store**: implement `NetworkStore` (`write` receives the network after every batch that is
  kept) and register it in `parts.stores` (`-map_opt store=NAME`).
- **Observer**: implement `MapperObserver` (every structure as it is computed, the network
  after every job, after every batch that is kept, and at the end) and pass it to
  `run_with_parts`. With several jobs at the same time the structures of the jobs come mixed
  (each with its job, those of one job in order) and the jobs are told in the order of their
  numbers; every call comes from the thread of the loop.
- **Your own loop**: `MapperDriver::ask` gives the next job, `MapperDriver::tell` takes what it
  found; `MapperDriver::ask_batch` gives jobs chosen together and `MapperDriver::tell_job`
  takes each by its number; `workflow::mapper::explore` (one job at a time) and
  `workflow::mapper::explore_with` (a `Schedule`, with a `MapperWorkers` that gives every job
  its calculations) are the loops of the command.

## Not done yet

These are named in the code and refused with a message (exit code 2) when they are asked for:

- model functions (`-mf`) and excited states of their own in the steps (the states of a run
  are given by `states`);
- on an ONIOM surface: what [An ONIOM surface](#an-oniom-surface) lists;
- bias potentials and constraints in the step settings;
- `ts_conformers=on` and `entrance_structures` in a run of several states (`states`).
