# Conformations, orientations, and scans

Searches over the structures of one system: conformers (`confsearch`), orientations of the parts
of a complex (`orientsearch`), and scans of coordinates (`relaxedscan`). The network of minima
and transition states is the mapper's ([Reaction-network mapper](mapper.md)).

## Conformation search (`confsearch`)

`confsearch INPUT -os NAME` enumerates the conformers of a structure: the minima that keep
its bond graph (no reactions), without duplicates, with rotamers counted, and with their
Boltzmann populations. It is the command for one molecule's conformers; `relaxedscan` scans
coordinates on a grid, and `run_mapper` maps reaction networks (it will use `confsearch` to
expand conformers). The input is optimized first (EQ0); its bond graph is the reference.

| strategy | option | what a trial does |
|---|---|---|
| AFIR sampling (MultiOptPy's, the default) | `-cs_opt method=afir` with `-bf`, `-ms`, `-nl`, `-nr`, `-tgta`, `-st`, `-nost`, `-tabu`, `-alpha` | an AFIR force of `-bf` kJ/mol pushes or pulls one pair of atoms in an optimization from the current conformer, the result is optimized without it; the next start is chosen by the sampling priority |
| iMTD-GC (CREST's workflow from its papers) | `-cs_opt method=imtd` | rounds of RMSD metadynamics, loose and final optimizations of their snapshots in an energy window, a new round from every lower conformer, then dynamics from the lowest conformers and the genetic crossing of the conformers |

The pairs of the AFIR sampling are the pairs of target atoms (`-tgta`) at least 3 bonds apart
along the bond graph (`-cs_opt pairs=bonds:3`, the default from the measurements below; in
a compact cluster, where every pair is within two bonds, the farthest pairs);
`pairs=multioptpy` takes MultiOptPy's window instead.

Every option of `optmain` applies to the optimizations: the calculator (any name of the
registry, ASE calculators and Python engines with the Python package), `-mf`, `-es`, the bias
potentials, the constraints (`-fix`, `-pc`, `-gfix`, `-uc`), `-coord` (the dynamics of
iMTD-GC run in Cartesian coordinates), ONIOM (`-oniom`), `-opt`, `-fc`, `-modelhess`,
`-freq` (a vibrational analysis of every conformer at the end, `vibrations/EQ<k>/`, as
`optmain -freq` makes it: with constraints the modes on the constraint surface, 3N - 6 - m
of them, so that a held coordinate is never an imaginary mode).
Without `-tcc` or `-lcc` the optimizations use the tight criteria (measured below). The
run directory, its settings file, `--restart RUN_DIR` from the journal, the `STOP` file
(and MultiOptPy's `end.txt`), and the counts of the evaluations at the end are those of every
command.

**Constraints** are resolved once, at the input (`cli::constraints::resolve_for_run`): every
candidate, optimization, dynamics, and IRC holds the values of the input (frozen atoms where
they are in the input, automatic `-pc` values of the input), not those of the structure a move
made. The moves leave frozen atoms where they are (kicks skip them, a torsion turns the side
without them and no bond with frozen atoms on both sides, a low-mode step has no component on
them), and every candidate is restored onto the constraints before it is optimized; one that
cannot be restored is discarded and counted (`off the constraints` in the line of the
candidates). Before, each job resolved its constraints at its own start, so that the frozen
atoms and automatic values of a kicked candidate stayed where the kick put them (test
`constraints_matrix_confsearch`).

**Bias potentials** are copied for every job (`pes::term::fork_term`): a term with state
(metadynamics deposits, the warm start of `-smp`, `-aerp`, `-aerpv2`) starts every job from the
same state, so that the energies do not depend on the order of the jobs or the number of
threads (test `bias_matrix_confsearch`: the same conformers with 1 and 4 threads). Terms that
change with the time or the iteration (ramps) are refused with the reason: every minimum would
carry a bias that depends on how long its optimization ran. The conformers are **ranked by the
energy with the bias potentials** (the surface they are minima of; `energy_hartree` of
`conformers.csv`); `bare_energy_hartree` is the energy without them, and `ensemble.txt` says
which one ranks. With a bias at a point in space the optimizations keep the lab frame
(`pes::term::frame_invariant`): the atom ends at the point the bias pulls it to.

### Which minima are kept, and when two are the same

- `-cs_opt mode=conformers` (the default): minima whose bond graph (covalent radii times 1.1,
  `bond_scale=`) is that of EQ0 up to a renumbering of the atoms (`graph=isomorphic`; `-pbc`
  compares in the numbering of the input as MultiOptPy does). The others are written to
  `REJECTED_CONFORMERS/` with the bonds formed and broken. `mode=minima` keeps every
  distinct minimum (MultiOptPy without `-pbc`).
- A new minimum is compared with the conformers within `ethr` (0.05 kcal/mol): an RMSD in
  the same numbering below `rthr` (0.125 Angstrom) makes it a duplicate; if the rotational
  constants differ by more than `bthr` (1 %) it is another conformer; otherwise an RMSD below
  `rthr` after the best renumbering of equivalent atoms (`permutations`; their classes of
  the bond graph by colour refinement, assigned by the Hungarian method) makes it a rotamer
  of that conformer, counted with it (`rotamers.xyz`). These are the tests and thresholds of
  CREGEN (Pracht, Bohle, Grimme, Phys. Chem. Chem. Phys. 2020, 22, 7169) with the rotamers
  decided by the renumbering itself. With `mirror=on` (the default) the mirror image of a
  conformer is the same conformer; a chiral conformer then has the degeneracy 2 in the
  populations. `compare_atoms=` compares a part only (a solute among solvent molecules).
- Molecules in a periodic cell are compared made whole, as one cluster; a structure whose
  bonds run through the cell into a network (a crystal, a surface, a polymer) has no
  conformers and is refused before anything is computed.
- An optimization can end exactly at a saddle point whose gradient vanishes by symmetry (an
  eclipsed methyl group, or a symmetric input). With `verify=on` (the default) the Hessian of
  every new conformer and of the optimized input is computed; at a saddle point (an
  imaginary mode by the one criterion of the crate, a mass-weighted curvature below
  `-IMAGINARY_CURVATURE`, about -0.5 cm^-1, with the zero modes projected as `optmain -freq`
  projects them) the structure is displaced by 0.2 Bohr along the imaginary mode
  (either way) and optimized again, and the lower minimum is judged instead. This costs one
  Hessian per conformer. With `nudge=on` (the default) every converged optimization is also
  displaced by a seeded random 0.02 Bohr per coordinate and optimized once more, which
  leaves most such points without a Hessian (measured below).

### The choice of the next start (`priority=`) and the end of the search

MultiOptPy's sampling starts every trial from a conformer chosen by a
`workflow::explore::SamplingPriority` (registered rules; any rule can be added in Rust):
`boltzmann_heated` (MultiOptPy's: Boltzmann weights at `-st`, ten times the temperature after
every fifth trial; with `-tabu` the visit penalty `exp(-alpha N)`, `tabu_heated`),
`boltzmann`, `tabu` (without the heating), `water_filling` (a rising level of 1 kcal/mol
steps; the conformers under it are sampled, least visited first, before it rises) and
`water_filling_adaptive` (the level rises to the next conformer above it, between 0.1 and 3
kcal/mol, so that a crowd of conformers at high energy is taken one by one and not all at
once). The search ends by the rules of `workflow::explore::stop`: `-ms` trials, the `-nr`
lowest energies unchanged for more than `-nl` trials (MultiOptPy's), or a rule that finds
nothing more to sample; iMTD-GC ends its rounds when one finds no conformer below the lowest
known before it (`NoLowerMinimum`) or after `rounds=` rounds.

### iMTD-GC

Implemented from Grimme (J. Chem. Theory Comput. 2019, 15, 2847), Pracht, Bohle, Grimme
(Phys. Chem. Chem. Phys. 2020, 22, 7169), Pracht et al. (J. Chem. Phys. 2024, 160, 114110), and
the documentation of CREST; no code of CREST is used.

- **RMSD metadynamics**: the bias `sum_i k exp(-alpha Delta_i^2)` of the RMSD `Delta_i` to
  reference structures (`pes::bias::RmsdMetadynamics`, the metadynamics of `pes/bias` with the
  RMSD to a set of structures as collective variable): the lowest known conformers
  (`known_references=`, 20) and a structure of the run deposited every `vbias=` fs (1000).
  Each round runs the dynamics of `md` (Bussi's thermostat at `mtd_temperature=`, 300 K;
  SHAKE on the bonds to hydrogen, `shake=`; hydrogen masses of 4 amu, `hmass=`; steps of
  `time_step=` fs) once for every pair `(k/N, alpha)` of `mtd_runs=` from the lowest
  conformer, in parallel. `mtd_runs=crest` (the default) is the 14 pairs printed in the
  example of CREST's documentation (`k/N` = 3.0, 1.5, 0.75 mEh with `alpha` = 1.300, 0.780,
  0.468, 0.281 Bohr^-2, and 1.0 mEh with 0.100, 5.0 mEh with 0.800), `k` scaled by the number
  of atoms of the bias (`bias_atoms=`).
- **Multi-level optimization**: a snapshot every `dump=` fs (100) is optimized with the loose
  criteria; those within `loose_window=` (12 kcal/mol) of the lowest are freed of
  duplicates and of known conformers, optimized with the criteria of the command, and judged.
- **Iterations**: a round that finds a conformer below the lowest known before starts the next
  from it.
- **Finishing**: dynamics without a bias from the `finish_conformers=` (4) lowest conformers at
  `finish_temperatures=` (400 and 500 K) for `finish_time=` ps (3), and the genetic crossing of
  the conformers within `crossover_window=` (6 kcal/mol): for every ordered pair `(i, j)` the
  structure `Z_ref + Z_i - Z_j` in the Z-matrix of the lowest conformer along its bonds
  (`coord::zmatrix`, `workflow::explore::crossover`; at most `crossover_max=` pairs), kept when
  its bonds are those of the reference and it is not within 0.25 Angstrom of another. When
  either finds a lower conformer, the rounds start again from it.

### Conformers of a transition state (`-tsconf`)

The input is optimized to a saddle point of order 1 (the methods of `-order 1` unless `-opt`
is given), its Hessian must have one imaginary frequency, and its IRC gives the minima of its
reaction. The imaginary mode is written in the primitive internal coordinates of the
transition state; those with at least `ts_share=` (0.1) of its largest component form the
reaction coordinate, which every optimization of the search holds at its value and the
dynamics hold by RATTLE (`ts_hold=coordinate`) or by freezing the negative-curvature mode of
the Hessian, computed again every `ts_update=` steps (`ts_hold=modes`, `md::freeze`). With
`ts_bonds=on` (the default) the reactive bonds (those with at least half the largest
component of the mode among the bonds) are held at their lengths as well and leave the
reaction coordinate: the mode alone leaves directions free along which the reacting fragments
fall apart (measured below). The
search (either strategy) then finds constrained minima; those within `ts_window=` (6 kcal/mol)
of the lowest (at most `ts_max=`) are released and optimized to saddle points of order 1,
checked for one imaginary frequency and for the same reaction (the bond graphs of the ends of
their IRC are those of the input's, `ts_irc=`), and freed of duplicates and rotamers.
`ts_conformers.xyz` and `ts_conformers.csv` list them with their energies relative to the
lowest and to the input, their imaginary wavenumber, whether the reaction is the same, and
the energies of the ends of their IRC (in `ts_irc/`); the lowest is the barrier of the
reaction. `conformers.xyz` and the other files of the search hold the constrained minima.

### What a run writes

MultiOptPy's files: `<stem>_EQ<k>.xyz` (every conformer in the order found), `EQ_energy.dat`,
`EQ_summary.log`, `switch_conformer.log`, `no_update_count.log`, `search_atom_pairs.log`,
with `-tabu` `visit_counts.dat`, `tabu_search_settings.log`, `tabu_selection_log.log`, and in
the conformer mode `reference_bond_table.dat`, `reference_bond_connectivity.log`, and
`REJECTED_CONFORMERS/` (`*_REJECTED<n>.xyz`, `*_REJECTED<n>_info.txt`, `rejected_count.dat`,
`rejected_energy.dat`, `rejected_summary.log`, `README.txt`). Besides: `conformers.xyz` (the
ensemble by energy, each with its energy, relative energy, and Boltzmann weight),
`rotamers.xyz` (with the rotamers), `conformers.csv` (rank, EQ number, energy, relative
energy, Boltzmann weight, degeneracy, RMSD to the lowest after the best renumbering,
rotational constants, rotamers, whether the bonds are those of EQ0, imaginary frequencies with
`-freq`, how it was found), `ensemble.txt` (the populations at `-temp`, the average energy,
the conformational entropy `S = -R sum p_i ln(p_i / g_i)`, and `-kT ln Z`), `mopt.log`,
`settings.json` (with the seed), and `evaluations.journal`.

From Rust the parts are `workflow::explore` (`ConformerEnsemble`, `SamplingPriority`,
`StopRules`, `crossover_candidates`), `coord::graph`, `coord::invariant_rmsd`,
`coord::zmatrix`, `pes::bias::RmsdMetadynamics`, and the strategies in `workflow::confsearch`
(they take any `SearchJobs` and `SearchObserver`); from Python `mopt_rs._core.bond_graph`,
`graph_isomorphism`, `invariant_rmsd`, `zmatrix`, `crossover_conformers`,
`ConformerEnsemble`, `RmsdMetadynamics`, `SamplingPriority`, and the command through
`run_cli("confsearch", [...])`, the `confsearch` entry point, or
`mopt_rs.confsearch.conformer_search` (which also takes search moves of your own, below).

### Searches by moves (`-cs_opt method=torsion | kick | lowmode`)

Besides MultiOptPy's AFIR sampling and iMTD-GC, a search can be made of trials that change
the conformer itself and minimize the result (`workflow::confsearch::moves`). The trials, the
choice of the next start (`priority=`), the judgement of the minimized structure, and the
termination (`-ms`, `-nr`, `-nl`) are those of the AFIR sampling; only the way a trial makes
its start differs:

- `method=torsion`: random turns about rotatable bonds, the Monte Carlo multiple minimum
  search (Chang, Guida, Still, J. Am. Chem. Soc. 1989, 111, 4379). A rotatable bond is one
  outside every ring with further atoms on both sides; one to `move_bonds` (3) of them are
  turned by angles uniform in (-180, 180] degrees, the smaller side moving.
- `method=kick`: every atom is moved to a random point within `move_radius` (0.7 Angstrom) of
  its position (Saunders, J. Am. Chem. Soc. 1987, 109, 3150).
- `method=lowmode`: a step along one of the `move_modes` (6) softest vibrations of the
  conformer, either way, so long that the atom that moves most moves by `move_step` (1.5
  Angstrom) (after Kolossváry and Guida, J. Am. Chem. Soc. 1996, 118, 5011, whose search
  follows the mode until the energy has passed a barrier; here the length of the step is
  fixed). Every trial costs the Hessian of its conformer, kept for later trials from it.

```text
confsearch molecule.xyz -os gfnff -cs_opt method=torsion move_bonds=2 -ms 60
```

A setting that the move does not have is refused, as is a `move_` setting with another
strategy.

### Your own search methods

A move is the part of a search that a user replaces: the proposal of the structure a trial
starts from. Everything else stays the search's, so that a method of your own has the
calculators (ONIOM with its microiterations, periodic cells, any engine), the constraints and
bias potentials of the command, the check of the bond graph, the removal of duplicates and
rotamers, the priority rules, the restart from the journal, and the files of `confsearch`.

**What the search does with a move.** (1) The input is optimized to the first conformer.
(2) The move is told that conformer (`prepare`). (3) For every trial the search hands the
move the conformer the trial starts from and gets a structure back; it minimizes that
structure on the surface of the command, rejects it when its bonds differ from those of the
first conformer (`mode=conformers`), leaves it out when it is a duplicate or a rotamer or
lies outside the energy window, and keeps it otherwise. (4) The priority rule chooses the
conformer of the next trial. (5) The search ends by `-ms` trials, by `-nl` trials without a
change among the `-nr` lowest conformers, or when the move returns nothing.

**From Rust.** Implement `workflow::confsearch::moves::ConformerMove` and register a factory
under a name:

```rust
use std::sync::Arc;
use mopt_rs::workflow::confsearch::moves::{
    ConformerMove, MoveContext, MoveRegistry, Proposal,
};

struct Stretch(f64);

impl ConformerMove for Stretch {
    fn name(&self) -> &str { "stretch" }

    fn propose(
        &mut self,
        current: &MolecularSystem,          // the conformer the trial starts from
        context: &MoveContext<'_>,          // trial, number of the conformer, the jobs
        rng: &mut Rng,                      // the random numbers of the search
    ) -> MoptResult<Option<Proposal>> {
        let mut x = current.coords_flat_bohr();          // 3N, Bohr
        let k = rng.gen_range_usize(x.nrows());
        x[k] += self.0;
        Ok(Some(Proposal {
            start: current.with_flat_coords_bohr(&x)?,   // the same atoms, moved
            what: format!("coordinate {} by {} Bohr", k + 1, self.0),
        }))
    }
}

let mut moves = MoveRegistry::builtin();
moves.register("stretch", Arc::new(|options| {
    options.only("stretch", &["by"])?;                   // -cs_opt move_by=...
    Ok(Box::new(Stretch(options.number("by", 0.5)?)) as Box<dyn ConformerMove>)
}))?;
let config = confsearch::config_from_cli(&["x.xyz", "-os", "gfnff", "-cs_opt", "method=stretch"])?;
let result = confsearch::run_with_moves(&config, base, None, &calculators, &moves)?;
```

- Positions are flat Cartesian coordinates in Bohr; the structure returned has the atoms of
  the one given, in their order (anything else is an error).
- Use `rng` for every random choice: the search is then repeated by `-cs_opt seed=`.
- `context.jobs` gives the Hessian of the surface (`hessian`), optimizations (`relax`), and
  dynamics (`dynamics`), should the move need them; they are counted and journaled like the
  search's own.
- `Ok(None)` ends the search (a move that enumerates and has run out); an `Err` stops the
  run with the message.
- `prepare(first, bond_scale)` is the place to set up what depends on the molecule (the
  built-in `torsion` finds its rotatable bonds there) and may return a line for the log.
- The settings `-cs_opt move_KEY=VALUE` arrive in the factory as `MoveOptions` (keys without
  `move_`); `only` refuses keys the move does not have.

**From Python.** Pass callables by name; `-cs_opt method=NAME` chooses one:

```python
import random
import mopt_rs.confsearch as cs

def shake(symbols, positions_bohr, trial, conformer, seed):
    r = random.Random(seed)                       # repeats with the search
    return [x + r.uniform(-0.8, 0.8) for x in positions_bohr], "a shake of 0.8 Bohr"

result = cs.conformer_search(
    ["x.xyz", "-os", "gfnff", "-cs_opt", "method=shake", "seed=3", "-ms", "40"],
    moves={"shake": shake},
)
print(result.directory, result.count, result.energies)
```

The callable gets the element symbols, the flat positions (3N, Bohr) of the conformer the
trial starts from, the number of the trial (from 0), the number of that conformer in the
order of discovery, and a seed drawn from the random numbers of the search. It returns the
positions to minimize (flat or nested), or `(positions, text)` with a line for the log, or
`None` when there is nothing more to try. The command runs with the GIL released and takes
it for every call. A Python move has no `move_` settings of the command line: it is an object
of the caller, who sets it up (a class with `__call__`, a closure).

With ONIOM microiterations the structures a move sees and returns are those of the core (the
atoms the optimizer moves); the environment is relaxed for whatever it proposes.

The reaction-path mapper (`run_mapper`) has the same shape in Rust: the methods that explore
a minimum, the priorities of the exploration, and the stop rules are traits with registries
by name, and the two-point methods are the registry of `workflow::pair_search`
([mapper.md](mapper.md), "Adding parts in Rust"). From Python the mapper runs as a command
(`run_mapper`); its parts cannot be given as Python callables yet, nor can the test for
duplicates be replaced.

The documentation itself is installed with the Python package: `python -m mopt_rs.docs`
lists the documents, `python -m mopt_rs.docs running` prints this one, and
`mopt_rs.docs.path()` is the directory that holds them.

### Measurements

Built-in GFN-FF, `tools/bench/confsearch_bench.py`, three seeds per setting, with
`verify=on`, `nudge=on`, and the tight criteria. The molecules: n-butane (input all anti),
n-pentane (input planar with one methyl group eclipsed, a saddle point by symmetry),
n-hexane, and Ace-Ala-NMe ("dipeptide"). The union is the set of conformers that any run of
the molecule found, merged by energy within 0.05 kcal/mol (an approximation: a run of the
dipeptide can hold two conformers that close). The AFIR settings are `-ms 50` with
MultiOptPy's pair window (`pairs=multioptpy`) and `priority=boltzmann_heated`
(`afir_boltzmann`, MultiOptPy's sampling as implemented here), `-tabu` (`afir_tabu`),
`priority=water_filling` and `water_filling_adaptive`, and `pairs=bonds:3` with
`boltzmann_heated` (`afir_bonds3`); `imtd_quick` is iMTD-GC with four metadynamics runs of 2
ps (`mtd_runs=3:1.3,1.5:0.78,0.75:0.468,3:0.281 mtd_time=2`). MultiOptPy itself was not
run; `afir_boltzmann` is its algorithm with the corrections listed in
[migration_workflows.md](migration_workflows.md).

| molecule | setting | conformers found (mean of seeds) | share of the union | gradients (mean) | Hessians (mean) | conformers per 1000 gradients |
|---|---|---|---|---|---|---|
| butane | afir_boltzmann | 2.0 | 1.00 (2) | 9985 | 2.0 | 0.20 |
| butane | afir_tabu | 2.0 | 1.00 (2) | 10241 | 2.0 | 0.20 |
| butane | afir_water | 2.0 | 1.00 (2) | 2640 | 2.0 | 0.76 |
| butane | afir_water_adaptive | 2.0 | 1.00 (2) | 2640 | 2.0 | 0.76 |
| butane | afir_bonds3 | 2.0 | 1.00 (2) | 10648 | 2.0 | 0.19 |
| butane | imtd_quick | 2.0 | 1.00 (2) | 29826 | 2.0 | 0.07 |
| pentane | afir_boltzmann | 3.7 | 0.92 (4) | 13299 | 6.0 | 0.28 |
| pentane | afir_tabu | 4.0 | 1.00 (4) | 13682 | 5.3 | 0.29 |
| pentane | afir_water | 3.0 | 0.75 (4) | 3482 | 4.3 | 0.86 |
| pentane | afir_water_adaptive | 3.0 | 0.75 (4) | 3534 | 4.3 | 0.85 |
| pentane | afir_bonds3 | 4.0 | 1.00 (4) | 14378 | 5.0 | 0.28 |
| pentane | imtd_quick | 4.0 | 1.00 (4) | 66753 | 5.0 | 0.06 |
| hexane | afir_boltzmann | 9.3 | 0.85 (11) | 15588 | 9.3 | 0.60 |
| hexane | afir_tabu | 9.7 | 0.88 (11) | 14536 | 9.7 | 0.67 |
| hexane | afir_water | 5.3 | 0.48 (11) | 6273 | 5.3 | 0.85 |
| hexane | afir_water_adaptive | 4.0 | 0.36 (11) | 4820 | 4.0 | 0.83 |
| hexane | afir_bonds3 | 10.3 | 0.94 (11) | 17074 | 10.3 | 0.61 |
| hexane | imtd_quick | 11.0 | 1.00 (11) | 89787 | 11.0 | 0.12 |
| dipeptide | afir_boltzmann | 9.0 | 0.30 (30) | 32440 | 9.0 | 0.28 |
| dipeptide | afir_tabu | 9.7 | 0.32 (30) | 31845 | 9.7 | 0.30 |
| dipeptide | afir_water | 11.3 | 0.38 (30) | 21496 | 11.3 | 0.53 |
| dipeptide | afir_water_adaptive | 12.3 | 0.41 (30) | 23140 | 12.3 | 0.53 |
| dipeptide | afir_bonds3 | 10.3 | 0.34 (30) | 28523 | 10.7 | 0.36 |
| dipeptide | imtd_quick | 27.3 | 0.91 (30) | 247229 | 28.7 | 0.11 |

The conformers of GFN-FF: pentane 0, 0.47, 0.83, and 2.36 kcal/mol (TT, TG, and two GG
conformers); hexane eleven between 0 and 2.85 kcal/mol; the dipeptide 30 up to 10 kcal/mol.
The AFIR runs of the dipeptide missed low ones as well (one with `pairs=bonds:3` found 11, up
to 7.1 kcal/mol, without those at 0.43, 1.58, and 1.59 kcal/mol). What the table says about
the defaults:

- **Strategy**: iMTD-GC found every conformer of pentane and hexane in every run and 17, 32,
  and 33 conformers of the dipeptide (the AFIR sampling 8 to 13), with 5 (pentane) to 8
  (dipeptide) times the gradients of the AFIR sampling. The AFIR sampling stays the default
  (the command's MultiOptPy counterpart, and the cheaper one); `method=imtd` is the one for
  flexible molecules when the ensemble has to be complete.
- **Metadynamics runs of iMTD-GC**: the default `mtd_runs=crest` (the 14 runs of 5 ps of
  CREST's documentation) found the same 4 conformers of pentane as the four runs of 2 ps of
  `imtd_quick`, with 195571 gradients (2.9 times; seed 1), and 36 conformers of the dipeptide
  with 862296 gradients (3.5 times the mean of `imtd_quick`, which found 17, 32, and 33): 26
  after the first round (with the optimized input), 6 more in the second, 1 from the
  finishing dynamics, and 3 from the crossing. The quick set is enough for the alkanes; the larger one, which found the most
  conformers of the dipeptide, stays the default.
- **Sampling priority**: `tabu_heated` and `boltzmann_heated` are within one conformer of
  each other. Water filling ends once every conformer under the water has been sampled
  three times and no new one turned up: a quarter to three quarters of the gradients, all
  of butane and more of the dipeptide than the Boltzmann choice (11.3 and 12.3 against 9.0),
  but 3 of the 4 of pentane and 5.3 and 4.0 of the 11 of hexane (2 to 8 per run; one run
  ended after 9 samples with the two lowest conformers only). `boltzmann_heated`
  (MultiOptPy's) stays the default.
- **Atom pairs**: with the same seeds, the pairs at least 3 bonds apart found as many
  conformers as MultiOptPy's window or more in all nine runs of pentane, hexane, and the
  dipeptide, more in five of them, with 0.9 to 1.1 times the gradients; `pairs=bonds:3` is the
  default, `pairs=multioptpy` MultiOptPy's selection.
- **Hessian check**: one Hessian per conformer and one for the optimized input (the column of
  Hessians). Without it and without `nudge` (an earlier build, AFIR sampling of butane with
  `boltzmann_heated` and `tabu_heated`), 4 of 6 runs kept the eclipsed saddle point of butane;
  with `nudge=on` alone the runs of pentane kept up to 6 "conformers" where 4 exist (saddle
  points of symmetry that the displacement of 0.02 Bohr did not leave); with `verify=on` no run
  kept more than 4.

**Convergence of the conformers.** n-Butane with GFN-FF, iMTD-GC with two runs of 2 ps
(seed 1): with the normal criteria of `optmain` the gauche conformer was found three times,
0.451, 0.470, and 0.514 kcal/mol above anti, with CCCC dihedral angles of 65.6, -67.7, and
70.5 degrees and rotational constants 1.1 % apart, so that the comparison could not merge
them; with the tight criteria the run has anti and gauche only (gauche 0.453 kcal/mol, with
the degeneracy 2). Hence the tight criteria without `-tcc` or `-lcc`. In the AFIR sampling of
butane (50 samples, priorities `boltzmann_heated` and `tabu_heated`, three seeds each) 4 of
the 6 runs also kept anti butane with one methyl group eclipsed (2.78 kcal/mol, a saddle
point whose gradient vanishes by symmetry); with `nudge=on` none did, and with `verify=on`
neither did pentane (the Hessian check above).

**Transition states.** GFN0-xTB. The Cope rearrangement of 1,5-hexadiene: the chair
transition state is 6.0 kcal/mol below the boat (the experimental difference is 5.8
kcal/mol); both have one imaginary frequency. From the boat, with the reaction coordinate
alone held (`ts_bonds=off`), the two allyl fragments separated along the symmetric stretch
the mode leaves free: 7 of 40 snapshots of two metadynamics of 2 ps dissociated and 12 more
lay 36 kcal/mol higher, and an AFIR pull of 400 kJ/mol between the central carbons pulled
the fragments apart. With the reactive bonds held (`ts_bonds=on`) no structure dissociated;
the chair was not reached in these short runs (it needs the ring that the two partial bonds
close to flip). 2-Ethyl-1,5-hexadiene at its chair transition state: written in all
primitive internal coordinates, the mode had components of 0.10 to 0.11 on the six
torsions about the bond to the ethyl group (its frame turns along the mode), so that the
reaction coordinate held the conformation of the group and 6 of 6 AFIR samples returned to
the start. Taking only the coordinates of the atoms that the mode moves (at least 0.1 of
the largest displacement: the carbons of the ethyl group move 0.01 and 0.02), the search
from the transition state with the ethyl group over the other fragment found the one 0.78
kcal/mol lower (-475 cm^-1, the same reaction by the IRC) in 6 samples: 3099 gradients and 4
Hessians in all.

## Orientations of a complex (`orientsearch`)

`orientsearch` places the parts of a complex (molecules or fragments) at random relative
orientations, optimizes every placement with `optmain`, and reports the distinct minima ranked
by energy:

```bash
orientsearch water_dimer.xyz -os gfnff -nsample 20                       # parts: the molecules
orientsearch complex.xyz -part 1-12 -os gfn0xtb -nsample 30 -orient_opt seed=4
orientsearch solvated.xyz -os gfn1xtb -oniom 1-3 none gfnff -nsample 10   # ONIOM
orientsearch co_on_cu.xyz -os emt -fix 1-18 -nsample 10                   # a slab in a cell
```

Every option of `optmain` applies to every sample:

- the calculator: any name of the registry, Python programs and ASE calculators included
  when the command is started from the Python package;
- `-oniom`, `-mf`, bias potentials, `-fix` and the other constraints, `-coord`, `-vc`;
- `-freq`: the imaginary frequencies of every minimum go into its table;
- `-modekill`: leave saddle points.

A sample is the run directory `<stem>_Sample_<k>_OPT/` of an `optmain` run from
`<stem>_Sample_<k>.xyz`. The samples converge with the tight criteria (`-tcc`) unless `-tcc`
or `-lcc` is given or `-orient_opt criteria=normal` asks for the default ones of `optmain` (the
measurements below show why).

**Parts.** `-part 1-4 9,10` makes the listed atoms parts and the remaining atoms one more part
(MultiOptPy). Without `-part`, the molecules of the bond graph are the parts.
`-orient_opt fixed=1` keeps part 1 where it is (a host). A part whose bonds reach its own
periodic images (a slab, a crystal, a polymer in a cell) always stays: it cannot be turned
without breaking the cell.

**Constraints** are resolved once, at the input, and every sample holds them: a part with
atoms the constraints pin in space (`-fix`, `-pc x|y|z|rot`) stays where it is in the input
(the log names it; the optimization still relaxes its free atoms), every placement is restored
onto the constraints before it is written and optimized (a placement that cannot be is
discarded with a line in the log), and a constraint between atoms of two parts (`-pc bond`,
`fbond`, `-gfix`, `-uc`) holds the value of the input, not that of the random placement.
Before, every sample resolved its constraints at its own start, so that a frozen atom of a
moved part was frozen where the placement put it and a distance between parts held the sampled
value (test `constraints_matrix_orientsearch`).

**Ranking.** The minima are ranked and weighted by their energy with the bias potentials,
the surface the samples are minimized on, as `confsearch` ranks; the bare energy is listed
beside it (`bare_energy_hartree`), and the interaction and binding energies are those of the
bare surface (the parts alone are calculated without the bias potentials). Before, the minima
were ranked by the bare energy (test `bias_matrix_orientsearch`).

**Schemes** (`-orient_opt scheme=`):

- `ring` (the default): every part is turned uniformly at random (Shoemake's quaternions).
  The centres go on a regular polygon in a random plane, in a random order, and the polygon
  shrinks until the closest pair of atoms of two parts is at `s (R_i + R_j)`. `R` are the van
  der Waals radii of Alvarez, and `s` is drawn from `contact=1.0,1.2`. Then every part that
  touches no other one is brought towards the centroid of the others until it does. For two
  parts this is `contact`. In a cell or with parts that stay it places as `contact`.
- `contact`: uniform rotations; the parts come one by one, in a random order, each from a
  random direction towards the centroid of those placed before, until they touch at `s (R_i +
  R_j)`. Parts that stay are placed first. In a cell whose parts that stay leave an empty
  layer of at least 8 Angstrom (a slab), the others come down onto the surface below the layer
  at a random point. In a cell without one (a box), they go to random points where they touch
  nothing closer than `s`.
- `uniform`: uniform rotations; every centre at the distance drawn from `[d, 1.1 d]` (`-dist
  d`) from the centre of a part placed before, in a uniformly random direction.
- `multioptpy`: MultiOptPy's scheme. Three angles from `[0, 2 pi)` rotate about x, y, and z in
  turn, and the centres are put on a regular polygon in the xy plane, neighbours the drawn
  distance apart. Two errors of MultiOptPy are corrected (migration.md).

`uniform` and `multioptpy` draw a sample again when two parts come closer than `clash=0.5`
of the sums of their radii (`clash=0` keeps every draw, as MultiOptPy does). Both are refused
in a periodic cell and with parts that stay, with the reason: in a cell the distance between
centres depends on the image.

**Starts from known minima** (`refine=SHARE`, 0 by default): after the first minimum, that
share of the samples starts from a known minimum, chosen by the rule `priority=` (`tabu`,
`boltzmann`, `water_filling`: `workflow::explore::PriorityRegistry`). One part is turned by up
to `angle=90` degrees and shifted by up to `shift=1.0` Angstrom, then moved out of any contact
closer than the lower end of the contact range.

**The same minimum.** Two converged samples are the same minimum when four things hold:

- Their bond graphs (1.2 times the sums of the covalent radii) agree up to a permutation that
  keeps the elements, compared by Weisfeiler-Lehman colours.
- After matching, the RMSD is below `rmsd=0.25` Angstrom. Identical molecules are assigned to
  each other and equivalent atoms within them by the Hungarian method. The superposition is
  tried from the 24 rotations between their principal axes, with turns about an axis whose
  moments are degenerate, and refined by alternating Kabsch fits and assignments.
- Their energies differ by less than `energy=1e-4` Hartree.
- `mirror=true` (the default): a mirror image counts as the same.

The lower of the two structures is kept. In a cell the structures are matched without
rotations, after the mean drift, with minimum-image distances.

**Energies of the parts alone** (`interaction=`): `vertical` (the default) evaluates every part
alone at its geometry in every minimum, `relaxed` optimizes every part alone once from the
input, `both`, `off`. The parts are neutral with their lowest multiplicity unless
`part_charges=` and `part_multiplicities=` say otherwise; a charged complex needs
`part_charges=`. They are left out, with a line in the log, with `-oniom` (its layers belong to
the atoms of the whole complex), with `-mf`, and for excited states.

**Files** in `<date>/<stem>_ORIENT_<time>/`:

- `<stem>_Sample_<k>.xyz` and `<stem>_Sample_<k>_OPT/`: the starts and their optimizations.
- `<stem>_minimum_<rank>.xyz` and `<stem>_minima.xyz`: the minima, lowest first, with their
  energies in the comment lines.
- `orientsearch_minima.csv` has one row per minimum:
  - rank, energy, and relative energy (kcal/mol);
  - the Boltzmann weight at `-temp` (minima only: not with `-order`, and not for structures
    in which `-freq` found imaginary frequencies; the log calls those saddle points);
  - the interaction and binding energies (kcal/mol);
  - how many samples ended there, the samples that found it, and how its first sample
    started;
  - the imaginary frequencies (`-freq`);
  - `contact_ratio`, the closest contact of two parts as a share of the sums of their van der
    Waals radii (above 1.5 the log says "parts apart": the parts did not find each other);
  - whether its bonds differ from those of the input.
- `orientsearch_samples.csv`: every sample with its start, convergence, iterations, energy,
  the minimum it gave, and its gradient and Hessian evaluations.
- `mopt.log`: a line per sample (as on the screen), the table of the minima, and the total of
  the evaluations. `settings.json` has the seed (drawn when `seed=` is not given).

A `STOP` file in the run directory ends the search after the sample in progress; a `STOP` file
in the directory of that sample ends its optimization at once. `--restart RUN_DIR` repeats the
samples with the same seed and takes their evaluations from their journals.

From Python, `mopt_rs.orientation_search(symbols, positions, calculator, ...)` runs the same
search with any calculator of the Python API, and `mopt_rs.OrientationSearch` is the ask/tell
machine. `mopt_rs.orient.compare_structures` and `mopt_rs.orient.complex_parts` are the
comparison and the parts. In Rust the pieces are in `workflow::explore`: `StructureMatcher`
and `BondGraph` (the same minimum), `MinimaPool` (the distinct minima, Boltzmann weights), and
`orient::{ComplexParts, Sampler, OrientationSearch}`:

```rust
use mopt_rs::workflow::explore::orient::{ComplexParts, OrientAction, OrientSettings, OrientationSearch, SampleOutcome};
use mopt_rs::workflow::explore::PriorityRegistry;

let parts = ComplexParts::molecules(&complex, &[], 1.2)?;
let settings = OrientSettings { samples: 20, ..OrientSettings::default() };
let mut search = OrientationSearch::new(&complex, &parts, settings, &PriorityRegistry::default())?;
while let OrientAction::Optimize(request) = search.ask()? {
    let (system, energy, converged) = my_optimizer(&request.start)?;   // any optimizer
    search.tell(Some(SampleOutcome { system, energy, converged }))?;
}
let lowest_first = search.pool().ranking();
```

### Measurements behind the defaults of `orientsearch`

`tools/bench/orientsearch_schemes.py` runs the searches below and makes the tables. They use
the built-in GFN-FF and GFN0-xTB, `-core 1`, and `interaction=off`. Each run has 12 samples,
and there are three runs per row (seeds 1 to 3) unless stated. The complexes start with the
molecules 8 Angstrom apart: water dimer, trimer, and tetramer, ammonia-water, and the benzene
dimer. The columns:

- *distinct*: minima per run that keep the bonds of the input and whose parts touch (contact
  ratio up to 1.5);
- *apart*: minima per run whose parts never came together;
- *gradients*: per run;
- *per 1000*: distinct minima per 1000 gradients;
- *found*: runs that reached the lowest minimum of all runs of the system and method;
- *in lowest*: the share of all samples that ended in that minimum.

**Convergence criteria.** With the default criteria of `optmain` (as MultiOptPy optimizes its
samples), GFN-FF gave:

| complex | scheme | distinct | apart | gradients | found | in lowest |
|---|---|---|---|---|---|---|
| water dimer | multioptpy | 2.0 | 3.0 | 1868 | 3/3 | 50% |
| water dimer | contact | 2.0 | 0.0 | 1258 | 3/3 | 67% |
| ammonia-water | multioptpy | 2.3 | 9.7 | 249 | 2/3 | 6% |
| ammonia-water | contact | 3.7 | 0.0 | 1159 | 3/3 | 44% |
| benzene dimer | multioptpy | 10.0 | 0.0 | 617 | 0/3 | 0% |
| benzene dimer | contact | 12.0 | 0.0 | 48 | 0/3 | 0% |

Three things go wrong with these criteria:

- With MultiOptPy's scheme the centres start 5 Angstrom apart. Of the minima per run, 9.7 of
  ammonia-water and 3.0 of the water dimer "converged" with the molecules apart.
- From van der Waals contact the benzene dimer took 4 gradients per sample and stopped wherever
  it was, 5 to 7.5 Angstrom between the ring centres. No run reached the stacked dimer, which
  the tight criteria find 0.47 kcal/mol below the T-shaped one.
- Copies of one minimum of the water dimer ended up to 0.4 Angstrom apart (RMSD after
  matching). No tolerance then tells them from the saddle point 0.043 kcal/mol above (one
  imaginary frequency with `-freq`).

With the tight criteria the copies of a minimum agree within 0.02 Angstrom (below). From
contact starts that costs 1.8 to 2.4 times the gradients (80 times for the benzene dimer, which
had stopped at once).
Hence `criteria=tight` is the default.

**Duplicate tolerances.** One run per complex had duplicate removal switched off (`rmsd=0
energy=0`, 16 samples, GFN-FF, tight criteria). The RMSD after matching of every pair of
converged samples:

| complex | copies: RMSD (dE, kcal/mol) | next pairs | then |
|---|---|---|---|
| water dimer | 79 pairs below 0.02 A (0.000) | 26 at 0.3-0.4 A: minimum and saddle point (0.043) | none |
| water trimer | 56 below 0.02 A (0.000) | none below 1.0 A | 22 at 1.0-1.5 A: ring and chain (1.2) |
| ammonia-water | 44 below 0.25 A (0.003): the free turn of NH3 | 12 at 0.3-0.5 A: two forms of the bond from ammonia to oxygen (0.021) | 64 at 0.7-1.0 A (4.8) |
| benzene dimer | 35 below 0.2 A (0.005) | 16 at 0.25-0.7 A: stacked dimers (0.010) | 54 at 1.0-1.5 A: stacked and T-shaped (0.48) |

Copies of one minimum stay below 0.25 Angstrom and other stationary points start at 0.3, so
`rmsd=0.25` is the default. The energies of copies differed by at most 0.005 kcal/mol (8e-6
Hartree). `energy=1e-4` Hartree (0.063 kcal/mol, the value of MultiOptPy's conformation
search) only keeps structures of distant energies from being compared. Stacked benzene dimers
0.25 to 0.7 Angstrom apart (0.005 to 0.010 kcal/mol) stay separate minima at this tolerance;
`rmsd=0.7` merges them.

**Schemes for two molecules** (tight criteria; `ring` places two parts as `contact` does):

| complex | method | scheme | distinct | gradients | per 1000 | found | in lowest |
|---|---|---|---|---|---|---|---|
| water dimer | GFN-FF | multioptpy | 1.7 | 3689 | 0.45 | 3/3 | 86% |
| | | uniform | 2.0 | 3221 | 0.62 | 3/3 | 92% |
| | | contact | 1.7 | 2292 | 0.73 | 3/3 | 92% |
| | | contact 0.8-1.0 | 1.3 | 2264 | 0.59 | 3/3 | 97% |
| | | contact 1.2-1.5 | 1.0 | 2876 | 0.35 | 3/3 | 100% |
| | | contact, refine 0.5 | 1.3 | 2213 | 0.60 | 3/3 | 94% |
| ammonia-water | GFN-FF | multioptpy | 2.0 | 3470 | 0.58 | 3/3 | 36% |
| | | uniform | 2.7 | 3681 | 0.72 | 3/3 | 31% |
| | | contact | 2.3 | 2301 | 1.01 | 3/3 | 44% |
| | | contact 0.8-1.0 | 3.0 | 1663 | 1.80 | 3/3 | 53% |
| | | contact 1.2-1.5 | 2.3 | 2432 | 0.96 | 3/3 | 47% |
| | | contact, refine 0.5 | 2.0 | 2003 | 1.00 | 3/3 | 53% |
| benzene dimer | GFN-FF | multioptpy | 4.0 | 3397 | 1.18 | 3/3 | 22% |
| | | uniform | 5.0 | 3071 | 1.63 | 3/3 | 19% |
| | | contact | 4.3 | 3830 | 1.13 | 3/3 | 36% |
| | | contact 0.8-1.0 | 3.7 | 3069 | 1.19 | 3/3 | 39% |
| | | contact 1.2-1.5 | 4.0 | 3566 | 1.12 | 3/3 | 22% |
| | | contact, refine 0.5 | 3.3 | 3090 | 1.08 | 3/3 | 33% |
| water dimer | GFN0-xTB | multioptpy | 2.0 | 4658 | 0.43 | 3/3 | 78% |
| | | uniform | 2.0 | 5381 | 0.37 | 3/3 | 75% |
| | | contact | 1.7 | 2779 | 0.60 | 3/3 | 92% |
| | | contact 0.8-1.0 | 1.0 | 2931 | 0.34 | 3/3 | 92% |
| | | contact, refine 0.5 | 1.7 | 2798 | 0.60 | 3/3 | 92% |
| ammonia-water | GFN0-xTB | multioptpy | 3.0 | 4140 | 0.72 | 3/3 | 56% |
| | | uniform | 2.7 | 3935 | 0.68 | 3/3 | 72% |
| | | contact | 3.0 | 1733 | 1.73 | 3/3 | 58% |
| | | contact 0.8-1.0 | 3.3 | 1436 | 2.32 | 3/3 | 56% |
| | | contact, refine 0.5 | 2.7 | 1955 | 1.36 | 3/3 | 78% |
| benzene dimer | GFN0-xTB | multioptpy | 2.0 | 3621 | 0.55 | 3/3 | 97% |
| | | uniform | 2.0 | 3276 | 0.61 | 3/3 | 100% |
| | | contact | 2.0 | 4079 | 0.49 | 3/3 | 97% |
| | | contact 0.8-1.0 | 2.0 | 3557 | 0.56 | 3/3 | 100% |
| | | contact, refine 0.5 | 2.7 | 3736 | 0.71 | 3/3 | 94% |

With GFN0-xTB, MultiOptPy's scheme also left 0.7 water dimers per run apart and gave 0.3
structures per run with a proton moved. Starting at contact 0.8 to 1.0 gave 0.3 such structures
per run for the water dimer and 0.7 for the trimer. The range 1.0 to 1.2 gave none in these
runs and 0.2 per run in the longer trimer runs below (`ring`: none).

**Schemes for three and four molecules** (16 samples, six runs per row, tight criteria):

| complex | method | scheme | distinct | gradients | per 1000 | found | in lowest |
|---|---|---|---|---|---|---|---|
| water trimer | GFN-FF | multioptpy | 4.7 | 8190 | 0.57 | 6/6 | 20% |
| | | uniform | 3.0 | 11054 | 0.27 | 3/6 | 3% |
| | | contact | 4.2 | 8166 | 0.51 | 2/6 | 3% |
| | | ring | 4.5 | 6530 | 0.69 | 6/6 | 30% |
| | | ring 0.8-1.0 | 4.5 | 5788 | 0.78 | 5/6 | 28% |
| water tetramer | GFN-FF | multioptpy | 2.8 | 7514 | 0.38 | 6/6 | 71% |
| | | uniform | 4.7 | 13389 | 0.35 | 6/6 | 14% |
| | | contact | 7.2 | 9718 | 0.74 | 6/6 | 29% |
| | | ring | 3.8 | 5046 | 0.76 | 6/6 | 47% |
| | | ring 0.8-1.0 | 3.8 | 4991 | 0.77 | 6/6 | 46% |
| water trimer | GFN0-xTB | multioptpy | 4.3 | 7649 | 0.57 | 6/6 | 17% |
| | | contact | 5.2 | 8624 | 0.60 | 4/6 | 4% |
| | | ring | 3.8 | 3778 | 1.01 | 6/6 | 16% |

What decided the defaults:

- **Scheme `ring`.** Parts added one by one (`contact`) and the uniform scheme rarely closed
  the ring of the water trimer (3 to 4% of the samples). Starting on a polygon did it in 16 to
  30%, from van der Waals contact (`ring`) with fewer gradients than from 5 Angstrom
  (`multioptpy`): 0.74 against 0.39 rings per 1000 gradients for the trimer with GFN-FF, and
  0.68 against 0.36 with GFN0-xTB. For the tetramer MultiOptPy's square reached the ring more
  often per sample (71% against 47%), at the same rate per gradient (1.51 against 1.49 per
  1000). For two molecules `ring` is `contact`. For the water dimer and ammonia-water it took
  0.42 to 0.66 of the gradients of MultiOptPy's scheme; for the benzene dimer 1.13 times as
  many. It never started with clashing parts or with parts out of reach.
- **Contact range 1.0 to 1.2.** Starting closer (0.8 to 1.0) saved up to 28% of the
  gradients (nothing for the water dimer), but with GFN0-xTB it gave structures with a proton
  moved. Starting farther (1.2 to 1.5) cost more and found no more.
- **No refinement by default.** Starting half of the samples from known minima (`refine=0.5`)
  changed the gradients by -31% to +13%. It found fewer distinct minima in six of eight
  searches (the same in one, more in one), and finding them is what the search is for.

**Starts.** Of 2000 starts of each scheme with `-dist 5`:

- MultiOptPy's put atoms of the two benzene molecules closer than half the sum of their radii
  in 8.3% of the starts (closer than 0.3 in 0.8%); the uniform scheme without the clash test in
  9.8%.
- For two water molecules, 96% of MultiOptPy's starts had no contact at all (every pair beyond
  1.5 times the sum of the radii).
- The contact scheme put every start at a ratio of 1.0 to 1.2.

**What the minima are.**

- *Water dimer*: the hydrogen-bonded dimer (H...O 1.96 Angstrom with GFN-FF, 1.63 with
  GFN0-xTB). A saddle point lies 0.043 (GFN-FF) or 0.118 (GFN0-xTB) kcal/mol above it, with
  one imaginary frequency (`-freq`). GFN0-xTB also gave structures with a proton moved (bonds
  `changed`); one was 3.8 kcal/mol below the hydrogen-bonded dimer, a failure of the method,
  not of the search.
- *Water trimer*: the ring with three hydrogen bonds (2.03 Angstrom with GFN-FF, 1.81 with
  GFN0-xTB). Chains with two lie 0.6 to 1.2 (GFN-FF) or 0.1 to 0.4 (GFN0-xTB) kcal/mol above.
- *Water tetramer* (GFN-FF): the ring with four hydrogen bonds (1.94 Angstrom); a second ring
  lies 0.26 kcal/mol above.
- *Ammonia-water*: with GFN-FF water donates to nitrogen (H...N 2.01 Angstrom) and the reverse
  bond lies 4.7 kcal/mol above. GFN0-xTB reverses the order: ammonia donates to oxygen (H...O
  1.58 Angstrom), and water to nitrogen lies 1.7 kcal/mol above.
- *Benzene dimer*: both methods put the stacked dimer lowest (3.50 Angstrom between the ring
  centres with GFN-FF, 3.42 with GFN0-xTB). The T-shaped dimer (4.75 Angstrom) lies 0.47
  kcal/mol above with GFN-FF; GFN0-xTB found no other minimum. Neither method gave a
  parallel-displaced minimum. Coupled-cluster references put the parallel-displaced and
  T-shaped dimers lowest and the stacked one higher, so these orders belong to the two
  methods.

## Scans of coordinates (`relaxedscan`)

```text
relaxedscan SN2.xyz -nsample 8 -scan bond 1,2 1.3,2.6 -elec -1 -os gfn1xtb
relaxedscan ethane.xyz -os gfnff -scan dihedral 3,1,2,6 -180,180 -nsample 13
relaxedscan pentane.xyz -os gfnff -scan dihedral 1,2,3,4 -180,180 dihedral 2,3,4,5 -180,180 \
    -grid -nsample 13,13 -scan_opt ts_optimize=on
relaxedscan ethane.xyz -os gfnff -scan dihedral 3,1,2,6 -180,180 -nsample 13 -rigid
relaxedscan ts.xyz -os gfn1xtb -resper irc_combined_path.xyz -scan resper 1 -0.9,0.9 -nsample 21
relaxedscan mb.txt -os mb2d -scan var x -1.5,1.1 var y -0.4,2.0 -grid -nsample 27,25
```

`relaxedscan` scans chosen coordinates and optimizes every point of the scan under
constraints that hold the coordinates at their values (or, with `-rigid`, calculates single
points). Its results are the energy curves and maps, the transition-state candidates and
minima on them, and the structure of every point. It builds no reaction network
(`run_mapper` does) and enumerates no conformers (`confsearch` does). Each point is a run of
`optmain`, so every option of `optmain` applies to the points: the calculator and its
settings, `-mf`, `-es`, the bias potentials, `-pc`, `-fix`, `-uc` (which hold together with
the scanned coordinates), `-coord`, `-opt`, `-fc`, `-modelhess`, the criteria, `-oniom` with
its microiterations, `-vc`; `-irc` is refused with the reason (the points are constrained
minima, not transition states). With `-vc` the cell relaxes at every point under the
constraints the variable cell holds (`bond`, `fbond`, `angle`, `dihedral`, `expr`, `cell`, `-fix` as
fixed fractional coordinates); `x`, `y`, `z`, `resper`, and `-pc rot|eigvec|atoms_pair` are
refused with the reason. The user's constraints are resolved once, at the input: every point
holds the values of the input whatever it starts from (a neighbour, a ReSPer pre-image), and
each point's `settings.json` says what it holds (an automatic `-pc` is written in the manual
form with the value of the input, followed by the scanned coordinates; where it has no value
to write, as `rot`, the scanned coordinates go to `-uc`). Bias potentials with state
(metadynamics) or of the simulated time are refused with the reason: every point is an
optimization of its own and would start them again. A ramp over the iterations restarts at
every point as well: every point sees the ramp from its first iteration, and its energy is
that of the ramp at the iteration where its optimization ended. `scan_points.csv` has the bare
and the biased energy and both relative energies, each against the lowest point of its own
surface (`relative_bare_kcal_mol`, `relative_biased_kcal_mol`); the candidates are found on the
biased surface. With ONIOM
every point has the environment relaxed to a minimum, the points of a rigid scan (`-rigid`)
too: there the core is where the scan puts it and only the environment moves.

**Coordinates** (`-scan KIND ATOMS START,END`, as many groups as there are coordinates):

| kind | atoms | unit of the values |
|---|---|---|
| `bond` | two atoms | Angstrom |
| `fbond` | the atoms of two groups and the size of the first (`1,2,3,4,2`): their centroids | Angstrom |
| `angle`, `dihedral` | three, four atoms | degrees |
| `x`, `y`, `z` | one atom | Angstrom |
| `expr` (mopt-rs) | an expression of `-ubias` in place of the atoms (`'dist(1,2)-dist(1,3)'`) | atomic units, as `-uc` |
| `var` (mopt-rs) | a variable of an objective of plain variables (`-os mb2d`), by number or name | the variables' |
| `resper` (mopt-rs) | `1` or `2`: the ReSPer coordinate PCo1 or PCo2 of the references of `-resper` | Angstrom amu^1/2 |
| `cell` (mopt-rs, with `-vc`) | a parameter of the cell as `-vc_fix` names it: `a`, `b`, `c`, `ax`, ..., `cz`; `alpha`, `beta`, `gamma`; `volume` | Angstrom; degrees; Angstrom^3 |

With `-scan cell volume START,END` every point holds the volume while the atoms and the shape of the cell relax: an energy-volume curve (`-vc_fix shape` in addition gives the curve of a fixed shape). Its minimum is at the volume of the unconstrained `-vc` optimum (32 argon atoms, Lennard-Jones: test `constraints_matrix_relaxedscan`). The functions of the cell are also expressions (`-scan expr 'volume()' START,END` in Bohr^3); a cell scan needs `-vc` and relaxed points (`relax=off` is refused: the cell of a point is brought onto its value by the optimization of the point).

A range may start with a negative number (`-180,180`). In periodic cells distances, angles,
and dihedral angles use the nearest images.

**Points.** `-nsample N` (10 by default, as MultiOptPy) values from `START` to `END` (both
included) for every coordinate. Without `-grid` the coordinates move together, as in
MultiOptPy (point `k` has the `k`-th value of each): one curve. With `-grid` every combination
is a point: `-nsample 13,13` makes a map of 169 points.

**Order and continuation.** A relaxed scan follows one valley as long as every point starts
from an optimized neighbour. The points of a map are taken in snake (boustrophedon) order: the
first coordinate runs forwards, then backwards at the next value of the second, and so on, so
that consecutive points are neighbours on the grid; each point starts from the done point
nearest to it on the grid (the point before it, in snake order). `-scan_opt order=raster`
takes every row in the same direction and starts each row from the first point of the row
before. A valley can still end inside a map, so that neighbouring points come to lie on
different branches; after the map, pairs of neighbours whose structures (mass-weighted after
superposition) differ by more than `jump_factor` (3) times the median of all neighbouring
pairs are such jumps, and the higher point of each is optimized again from the lower one and
the lower result kept (`-scan_opt repair=on`, the default; at most two rounds). A dihedral
scanned over a whole turn reaches -180 and 180 degrees, one structure, twice: where the two
energies differ by more than 1e-5 Hartree (converged copies agree to 2e-6 in the scans
measured), the higher is optimized again from the lower in the same way, also on a curve.
The scan therefore reports the lowest branch it found. `-fo` starts every point from the input
structure (MultiOptPy). A relaxed point also starts from the last Hessian of the point it
starts from (`-scan_opt hessian=continue`, the default where `-fc`, `-modelhess`, `-ih`, and
`-mf` give no first Hessian; with ONIOM microiterations it is the Hessian of the core, as
the optimizer of a point leaves it; every point leaves its Hessian in
`point_<k>/final_hessian.npy`,
and the next one reads it as `-ih`; a ReSPer point that starts from its interpolated
structure has no such Hessian; `hessian=off` starts every point as `optmain` does).

**Rigid scans** (`-rigid`): one evaluation per point at the input structure with the
coordinates set by moving the atoms on one side of each rigidly, as the scan of a Z-matrix
coordinate does: for a distance the side of the second atom (the smaller side where both are
possible) is translated along the bond, for an angle it is turned about the normal of the
plane, for a dihedral about the central bond. The sides come from the bond graph of the input
(1.2 times the covalent radii); where the bond graph does not divide the structure (a ring)
and for the other kinds, the atoms move by the shortest displacement that sets the value (the
Newton projection of the constraints). With ONIOM microiterations the environment is still
relaxed at every point.

**The page and the files.** `<stem>_RScan_<date>/` holds:

- `mopt.log` (one line per point: the values, the energy, how many iterations, the point it
  started from), `settings.json`, `scan_state.json` (what a restart needs);
- MultiOptPy's `<stem>_scan_energy_profile_<date>.csv` and `<stem>_RelaxedScan_<k>.xyz` (the
  structure of point `k`), `scan_points.csv` (every point with its grid indices, values,
  energies, kcal/mol above the lowest point, convergence, iterations, start, and flags);
- `<stem>_scan_trajectory.xyz` (a curve) or `<stem>_scan_curve_<j>.xyz` (every curve of a map
  along its first coordinate);
- `relaxedscan_energy_profile` (a curve), `relaxedscan_energy_map` and
  `relaxedscan_energy_curves` (a map), `.png` and `.svg`, with the energies with the bias
  potentials beside them when there are any (`_bias`); the map is drawn bilinearly between the
  grid points with a colour bar, missing points grey, the transition-state candidates
  (magenta) and minima (white) at the stationary points of their local models, and for
  ReSPer coordinates the reference path (black) and the grid points beyond the references
  or, where those are the majority, the points among them (dark grey) on top;
- `stationary_points.csv` and `<stem>_ts_candidate_<n>.xyz` (below);
- `point_<k>/`: the run of `optmain` of point `k` (its log, trajectory, journal);
- `relaxedscan_report.html` (`-scan_opt html=off` leaves it out): one file with the figures,
  the stationary points and their structures, the structures of all points played as frames,
  and the table of the points, in the style and with the structure viewer of the AutoTS page.

**Stationary points of the scanned surface.** At every interior point the energies (with the
bias potentials) are modelled by the quadratic of their central differences in grid steps; the
point holds a stationary point when the stationary point of its model lies within half a step
of it in every direction. No negative curvature is a minimum, one a transition-state
candidate (a maximum of a curve, a saddle point of a map); points on the edge and next to a
missing point are not judged, except along a dihedral angle scanned over one whole turn
(-180 to 180), whose neighbours wrap around (of the two copies of a point there the lower
counts). Candidates of one kind whose model stationary
points lie within one step of each other are one. `stationary_points.csv` lists them with the
position of the model's stationary point between the grid points, and every candidate's
structure is written as `<stem>_ts_candidate_<n>.xyz`, a start for `optmain -order 1`.
`-scan_opt ts_optimize=on` runs those optimizations at once (`ts_optimizations/`, the
Hessian of the calculator at the first point, a vibrational analysis that counts the
imaginary frequencies, candidates that end at the same structure marked as one), with the
results in `ts_optimizations.csv`.

**Restart and interruption.** A `STOP` file in the directory of the scan stops the point under
way at once and the scan after it; `relaxedscan --restart DIR` keeps the points done, replays
the interrupted one from its journal, and goes on. A point whose calculation fails (its
constraints cannot be reached, the calculator fails) is recorded as failed and the scan goes
on without it.

**Plain variables** (`-os mb2d`, `-os expr:...`): `-scan var NAME START,END` holds a variable;
the other variables are optimized by the optimizer of plain variables with equality
constraints (`opt::vector_constraints`: the constraint gradients projected out, the Newton
projection after every step, the quasi-Newton updates on the projected gradient). Every point
writes its trajectory and optimized variables in `point_<k>/`.

**From Python and Rust.** `mopt_rs.run_relaxedscan([...])` runs the command (with the Python
engines of the registry) and returns its points and stationary points; the parts are
`mopt_rs.ResperMap` (the map, `project` for the coordinates and gradients of a structure,
`preimage`), `VectorOptimizer.hold(variables=[(i, value)], resper=[(map, k, value)])`,
`_core.scan_points` (the points of a scan, their order and starts),
`_core.scan_stationary_points`, and `_core.rigid_scan_structure`. From Rust:
`cli::relaxedscan::run_with`, `analysis::resper` (`ResperMap`, `ResperCoordinate` as a
constraint function, `preimage`), `workflow::scan` (`ScanGrid`, `stationary_points`,
`rigid::set_rigidly`, `continuity::jumps`, `figures`), and `opt::vector_constraints`.

### ReSPer coordinates (`-scan resper`)

`-resper FILE` names reference structures, a path or a trajectory: the `<stem>_traj.xyz` of
`optmain -cmds`, the path of `nebmain`, `irc_combined_path.xyz` of an IRC, an MD trajectory
(for plain variables a CSV with a column per variable, such as `<stem>_traj.csv` of optmain
and `irc_combined_path.csv` of ircmain). Their map is that of `optmain -cmds`: classical
multidimensional scaling of the distances `sqrt(sum_a m_a |r_a - r'_a|^2)` after
mass-weighted superposition (`analysis::resper`; Tsutsumi, Ono, Taketsugu, Chem. Commun.
2021, 57, 11734), with the principal coordinates PCo1 and PCo2. A structure that is not a
reference gets its coordinates by Gower's formula for adding a point to the map (Biometrika
1968, 55, 582), `y_k = sum_i X_ik (B_ii - d_i^2) / (2 lambda_k)` with its distances `d_i` to
the references, and their gradient from those of the squared distances (at the optimal
superposition, `2 m_a (R_a - R'_a)`). The log gives the shares of the coordinates and the range
of the references on them, which is where a scan makes sense. `resper_atoms=` restricts the
distances to some atoms, `resper_mass=off` uses unit masses, `resper_stride=` and `resper_max=`
(200) thin a long trajectory.

There are two kinds of ReSPer scans:

- **Relaxed** (the default): every point is a constrained optimization that holds PCo1 (and
  PCo2) at the values of the point; the other coordinates relax. The first point starts from
  the structure for its values (below), the others from their optimized neighbours
  (`resper_start=preimage`: each from the structure for its own values). Second derivatives of
  the coordinates for the constraint curvature come from central differences of their
  gradients with `resper_curvature=on` (off by default: measured below).
- **Rigid** (`-rigid`): a single point at a structure for the values of the point. That
  structure comes from the references (the pre-image problem of the map,
  `analysis::resper::preimage`): the references are superposed in their order, and the
  structure is interpolated between those next to the point on the map: linearly between
  the two consecutive references whose coordinates enclose it (one coordinate), over the
  smallest triangle of nearby references that contains it, or along the nearest segment of
  consecutive references (two coordinates). Its own coordinates (Gower) are compared with the
  point: the distance is the error of the pre-image, written to `scan_points.csv`. With
  `resper_correct=on` (the default) the structure is then moved onto the point by Newton
  steps in the mass-weighted metric of the map. A grid point further from the nearest
  reference than `resper_outside_factor` (2) times the median spacing of the references on the
  map, or beyond their range or convex hull, is beyond the references: it is marked in the
  table and on the map (`resper_outside=skip` leaves it out).

Both kinds draw the same map (`relaxedscan_energy_map`, titled *Relaxed scan* or *Rigid scan
(single points)*) with the reference path on it.

### Measurements

All with the built-in calculators; gradients are those of the whole scan as the log counts
them (the times depend on the load of the machine and are left out).

**Torsion of ethane** (GFN-FF, `-scan dihedral 3,1,2,6 -180,180 -nsample 13`):

| scan | barrier (kcal/mol) | gradients |
|---|---|---|
| relaxed, `optmain`'s defaults, `hessian=off` | 2.664 | 227 |
| relaxed, `-modelhess fischerd3` | 2.664 | 127 |
| relaxed, `hessian=continue` (the default) | 2.664 | 129 |
| rigid (`-rigid`) | 2.772 | 13 |

The three candidates (0 and ±120 degrees) went to transition states with `ts_optimize=on` in
1 or 2 iterations each (one imaginary frequency each, 2.664 kcal/mol). The rigid barrier is
0.11 kcal/mol above the relaxed one (the methyl groups do not relax).

**Map of two dihedral angles of n-pentane** (GFN-FF, `-scan dihedral 1,2,3,4 -180,180
dihedral 2,3,4,5 -180,180 -grid -nsample 13,13`, 169 points): with `hessian=off` 8917
gradients in snake order and 9013 in raster order; with `hessian=continue` (the default)
3425 in snake order, and the energies of the points agree with those of `hessian=off` to
0.043 kcal/mol (the largest difference of the 169; all points converged in both). No jumps
between branches were found in either order. The first point (-180, -180 degrees), the
input structure, converged at once 2.78 kcal/mol above its copy at (180, 180), which the scan
reached from its neighbours; optimized again from the copies, four points at -180 degrees
came out lower (by 2.780, 0.015, 0.011, and 0.011 kcal/mol) for 5 gradients. The map then has
9 minima (the anti-anti structure, four with one gauche angle, two with two gauche angles of
one sign, two near (±90, ∓60) degrees) and 18 transition-state candidates;
`ts_optimize=on` took all 18 to transition states with one imaginary frequency each (2 to
32 iterations, 265 gradients and 36 Hessians in all), in five sets of symmetry-equivalent
ones at 2.837, 3.087, 3.267, 4.553, and 6.199 kcal/mol above the lowest point of the map.

**ReSPer coordinates of the Muller-Brown surface** (plain variables, references: the IRC
from both saddle points, A - S1 - C - S2 - B, 162 points; PCo1 90.7 %, PCo2 9.3 %). Barriers
in kcal/mol (energies times 627.5 with the surface scaled by 1e-3) over minimum A (S1) and
minimum C (S2); the IRC itself has 66.54 and 5.35:

| scan along PCo1 (41 points) | S1 over A | S2 over C | gradients |
|---|---|---|---|
| relaxed, each point from its optimized neighbour | 89.9 | 5.16 | 138 |
| relaxed, each point from its pre-image (the default) | 66.45 | 5.16 | 155 |
| rigid (pre-images, corrected) | 66.45 | 5.20 | 41 |

From its neighbour the relaxed scan stays on another branch of the level set of PCo1 (the
coordinate takes the same value at several places of the surface), 23 kcal/mol above the
saddle point; from the pre-images it follows the path, and its profile is the lowest energy
on each level set (it drops into the basin of A where the level sets of PCo1 cross it,
below the rigid profile). On the map of both coordinates (31 x 21 points) two coordinates
fix both variables, so the relaxed and the rigid scan with the correction coincide at every
point (|dE| = 0 at the 51 points among the references); without the correction the
interpolated structures miss their points by 0.019 (median; 0.038 at most; PCo1 spans 1.86)
among the references and by 0.20 (median; 0.61 at most) beyond them, their energies differ
by 0.13 kcal/mol (median; 1.4 at most), and the uneven map gives 18 spurious stationary
points (none with the correction). This is why the correction is on by default.

**ReSPer coordinates of HCN -> HNC** (GFN1-xTB; the transition state from `optmain -order 1
-fc 1`, one imaginary frequency of 1196i cm^-1; its IRC of 91 structures as the references;
PCo1 92.3 %, PCo2 7.6 %). Along PCo1 over the whole range of the references (21 points):

| scan | barrier HCN -> HNC (kcal/mol) | gradients |
|---|---|---|
| relaxed, from the neighbours | 57.910 | 104 |
| relaxed, from the neighbours, `resper_curvature=on` | 57.910 | 107 |
| relaxed, from the pre-images (the default) | 57.910 | 81 |
| rigid, corrected (pre-image error: median 1.1e-6, at most 6.3e-5 Angstrom amu^1/2) | 57.908 | 21 |
| rigid, not corrected | 57.908 | 21 |

The second derivatives of the coordinates do not save iterations (107 against 104 gradients),
so `resper_curvature` is off by default. On the map of both coordinates (15 x 11 points over
the box of the references) only 15 points lie among the references; the relaxed energies there
are 0.0003 to 0.58 kcal/mol below the rigid ones (the other atoms relax), and the pre-images
miss their points by 0.054 (median; 0.093 at most) Angstrom amu^1/2 before the correction.
Beyond the references both kinds give structures thousands of kcal/mol high (two coordinates
that no structure near the path has), and 3 relaxed points there (on the edge of the box)
could not reach their coordinates at all; the repair of jumps (two rounds, 24 and 33 pairs)
lowered eight relaxed points beyond the references, by 44 to 1507 kcal/mol, that had started
from a neighbour on another branch, and changed none of the 15 among them. The relaxed map
took 22860 gradients, nearly all of them beyond the references. Such points are marked as
beyond the references; `resper_outside=skip` leaves them out.

**Which kind is the default, and why.** The relaxed scan is the default for every
coordinate, ReSPer coordinates included: its profile is the lowest energy at each value
(the meaning of a relaxed scan) and it agrees with the IRC where the coordinate follows the
path (HCN, both barriers of Muller-Brown once its points start from their pre-images). The
rigid scan shows the energy along the reference path itself at a fifth of the cost; it is
the better picture where the level sets of the coordinates cut across neighbouring basins,
as near the first saddle point of Muller-Brown, where the relaxed profile drops into the
basin of A. The map figure is the same for both, titled *Relaxed scan* or *Rigid scan
(single points)*, with the reference path on it.

**ONIOM** (ethanol, GFN0-xTB on the hydroxymethyl group, GFN-FF on the whole, the C-C-O-H
dihedral angle in 13 points): with the microiterations the points at -180 and 180 degrees
(the same structure) agree to 0.001 kcal/mol and the barriers are 0.879 (0 degrees) and 2.020
kcal/mol (±120); without them the first point stays in another position of the methyl group
(3.919 kcal/mol at -180 against 1.124 at 180 degrees). The microiterations cost 4094
gradients against 326 here (their environment of three hydrogen atoms is small, and every
relaxation checks its lowest curvature); the point of them is larger environments, which
`optmain` measures (see the ONIOM section above).

**Periodic cell** (copper, 32 atoms in a 2 x 2 x 2 face-centred cubic supercell, EMT; the
distance of atom 1 and its nearest neighbour across the boundary of the cell, 2.2 to 3.2
Angstrom in 11 points): relaxed 67 gradients, rigid 11; both have their minimum at 2.6
Angstrom; at 3.2 Angstrom the rigid energy is 7.1 kcal/mol above the relaxed one.
