# Design of the reaction network (`run_mapper`): nodes, edges, entrances

This page is the one place where the rules of the network of `run_mapper` are written down.
That covers what a node is, when two are the same, what an edge joins, how things enter the
network, and how each kind reaches the kinetics and the report. Code that adds a kind of node
or edge, a way of exploring, or a store follows it, and a change of a rule is made here first.
[mapper.md](mapper.md) describes how to run the command; [kinetics.md](kinetics.md) describes
the kinetic models.

Each rule carries its state: **built** (in the code and tested), or **planned** (decided, not
built yet).

The files of a job are organized by stage. Each NEB iteration has its own band and
evaluation trajectory. The job record links to its ordered trajectory index. A queue
snapshot records the priority decision at dispatch, before the selected job runs. For RCMC the same dispatch publishes the latest superstate matrix, its EQ membership and the transient EQ populations in the run directory.

## Nodes

| Kind | Tag | What it is | State |
|---|---|---|---|
| minimum | `EQ` | a local minimum, **verified by its Hessian** (see below); a complex of fragments within interaction range is an ordinary `EQ` | built |
| transition state | `TS` | a first-order saddle point, with one imaginary mode checked by its Hessian | built |
| crossing point | `MECP`, `MECI` | minimum-energy crossing of two electronic states (other spin; same spin) | built (`states`; below) |
| dissociation channel | `DC` | a set of fragments separated beyond the interaction range, each optimized on its own (`workflow/mapper/channels.rs`) | built |
| path peak | (no node: a record `PP`, see below) | the highest point of a path that gave no verified TS: its structure and energy (no Hessian, no frequencies) and the minima on its two sides, saved beside the network and not a part of it ([below](#edges)) | built (`path_peaks=failed`, the default) |
| higher-order saddle | `2SP`, `3SP`, ... | a stationary point with k ≥ 2 negative curvatures, with its descending directions (± each unstable mode) followed to lower saddles or minima | built, experimental, off by default (`higher_saddles=on`) |

- **Every EQ is checked by vibrational analysis by default** (user decision, 2026-11-17:
  `-map_opt verify_minima=on`, the default; `off` takes minima without the check, and the
  network file says so in `minima_verified`). Built (`cli/mapper_minima.rs`), and so are the
  TS frequencies from the IRC start check:
  - Every minimization of the mapper (inputs, IRC ends, quenched frames, the ends of path
    peaks) computes the Hessian at convergence.
  - Mode kill is opt-in. Ordinary minima with significant negative curvature are rejected.
    At an optimized IRC endpoint, an imaginary vibration at or above `irc_end_threshold_cm1` (default 50 cm^-1)
    triggers a search for a linked TS and an IRC from it. If the link cannot be resolved,
    the endpoint is unknown. A single weaker imaginary mode is kept on the EQ with a warning. Two or more imaginary modes leave that side unknown. The network preserves the verified side in `open_ts_edges` and omits the incomplete reaction from kinetics.
  - The Step 3 TS search refreshes its exact Hessian every five steps by default. The Step 4
    IRC refreshes its exact Hessian every ten steps.
  - The frequencies of that Hessian (the 3N - 6 or 3N - 5 vibrations, without the rigid
    motions) are stored on the node, so the kinetics has the frequencies of every EQ without
    another calculation; the fragments of a DC have theirs the same way. Every TS has the
    Hessian of its IRC start check (one imaginary mode), and its frequencies come from it the
    same way (the imaginary one negative). Built.
- **Numbers.** Each kind is numbered on its own from 0 (`EQ0`, `EQ1`, ..., `TS0`, ...,
  `DC0`, ...). Files are named after the label (`structures/EQ000003.xyz`,
  `paths/TS000007_irc.xyz`; the file runs from minimum to minimum, with the minimization
  of each end of the IRC before and after it, as the path file of a crossing point does:
  user request, 2026-11-19). Built. Saddle points of order k ≥ 2 are numbered on their own
  for every order, the order before `SP` (`2SP0`, `2SP1`, ..., `3SP0`;
  `structures/2SP000003.xyz`; user, 2026-11-18): the order in front needs no separator from
  the number. `2SP` is what the literature writes SP2 (an index-2 saddle point). A saddle
  point of order 1 is always a `TS`. The files keep the kind as `2SP`, `3SP`, the order being
  the digits before `SP` (built). Crossing points are numbered the same way, each tag on its
  own (`MECP0`, `MECP1`, ..., `MECI0`, ...; `structures/MECP000000.xyz`), whatever their pair
  of states, which the node records (user question, 2026-11-18).
- **Level.** Every node is computed at one level: 1, the exploration, or 2, a refinement on
  another surface. The levels are never mixed in one node or edge. A network taken to another
  surface is a network of its own, in a run directory of its own, that names the network it
  comes from ([below](#a-network-taken-to-another-surface--remap-and--single_points)); the
  nodes of each are at the level of their own run.
- **Origin.** Every node records the input its lineage starts at, the job that found it, and
  the node that job started from. Built.
- **Raw quantities only.** Nodes and edges keep energies, frequencies, IRC energies and
  gradient differences, never rate constants: the kinetics computes rates afterwards from
  them. Built.

## When two nodes are the same

- **EQ and TS** are the same when the elements and bond graphs (bonds up to 1.2 times the sums
  of the covalent radii) agree, the energies lie within `energy_tolerance_kcalmol`, and the RMSD
  after the best matching of equivalent atoms lies within `rmsd_threshold`. Mirror images are
  different unless `mirror=true`. Distances are minimum-image distances in a cell. Built.
  - The thresholds are 0.125 Angstrom and 0.05 kcal/mol (those of the conformer ensembles of
    CREST: Pracht, Bohle, Grimme, Phys. Chem. Chem. Phys. 2020, 22, 7169), not MultiOptPy's
    0.30 Angstrom and 1 kcal/mol (user request, 2026-11-19: a sharper test of identity).
    Measured on the two runs of 1000 jobs of the urea synthesis with GFN1-xTB, from the last
    structure of every minimization of an IRC end in the trajectories of the jobs (1182 and
    1005 of them) against the minima of the network:

    | | `priority=rrt` with the filter | without it |
    |---|---|---|
    | ends that agree with a node to 0.005 Angstrom in every distance between two atoms (their energies agree to 7e-5 kcal/mol) | 1144 of 1182 | 969 of 1005 |
    | of the others, within 0.30 Angstrom and 1 kcal/mol of a node (taken for it) | 10 (one 0.003 Angstrom away, nine 0.14 to 0.30) | 4 (0.04, 0.09, 0.24 and 0.24 Angstrom away) |
    | of the others, within 0.125 Angstrom and 0.05 kcal/mol of a node | 1 (0.003 Angstrom, 0.017 kcal/mol) | 1 (0.04 Angstrom, 1e-4 kcal/mol) |

    (The RMSD of these rows is that of the atoms as they are numbered, an upper bound of the
    RMSD of the matching.) So one minimum found twice is the same to a few thousandths of an
    Angstrom, and what lies 0.09 to 0.30 Angstrom from a node is another minimum that the
    looser thresholds merged into it; one such pair (0.239 Angstrom, 0.0049 kcal/mol apart)
    was reached and merged in both runs. A surface with flatter minima or looser convergence
    may need looser thresholds; a minimum split in two costs jobs, one merged into another is
    lost.
  - Mirror images of one energy are a large part of a network of achiral molecules: of the
    260 and 184 minima of these runs, 79 and 78 pairs are a structure and its mirror image (the
    same distances between the atoms to 1e-4 Angstrom, the signed volumes with the other
    sign), and no pair is the same structure twice. They stay different nodes by default,
    since a mirror image may be the other enantiomer (user decision, 2026-11-19);
    `mirror=true` makes them one.
- **Several states** (`states`): nodes are the same only on the same state, crossing points
  only of the same pair of states. Built.
- **ONIOM** (built, 2026-11-20; `cli/mapper_oniom.rs`): the network holds the **cores** (the
  atoms the optimizer moves: the model region and the outer atoms of its links), and every
  calculation is on the relaxed surface of the core (the microiterations of every ONIOM run
  but the dynamics: the environment relaxed to a minimum before every evaluation).
  - `identity_scope=high` (the default): only the core decides identity, by its bond graph
    and RMSD; energies do not, since the environment moves them. `active_site:R` adds the
    atoms of the environment within R Angstrom of the core in the first input, which the
    optimizer then moves with the core. `all` is refused: a relaxed environment is a function
    of its core in every calculation, so environments are not kept apart as nodes.
  - Structures that differ only in the low layer are one node. Their environments are listed
    with energies as **environment conformers** (`environments` of the node: energy and job;
    two within the energy tolerance of the identity, 0.05 kcal/mol by default, are one, since
    two minimizations of one environment end a little apart):
    the lowest is the representative (the node takes its core and its whole structure, and
    its file is written again), and the report shows how many there are and their spread.
  - Environment conformers are not joined by TS edges.
  - An edge takes its barriers from the energies of its own IRC ends (consistent environments).
    A difference of more than 1 kcal/mol between such an end and its node is reported as
    environment heterogeneity (a warning; the threshold is a judgement, not yet measured).
  - The whole structures: a job starts from the environment of its node, and at its end the
    environment its calculator relaxed at every structure it found (which the journal
    records, so that a restart gives the same) makes the whole structure of that structure
    (`whole` of the node, `whole_coords_bohr` in the network file with `whole_elements`).
    The files of the nodes and the IRC paths hold the whole structures; the report and the
    database draw and keep the cores (and the whole structures in the JSON of the database).
  - Not on an ONIOM surface yet, refused by name with the reason: `-remap` and
    `-single_points`, `method=nanoreactor`, `entrance_structures`, several electronic states
    (`states`), `ts_conformers`. Dissociation channels are off: a core sits in its
    environment.
  - The environment moves with the core by the rigid motion that fits the nearest evaluated
    core onto the new one before it is relaxed (where rigid motions are symmetries), so that
    the recentring and rotation of the core by the optimizers and the IRC do not push it
    against its environment (built, 2026-11-20; checked on vinyl alcohol with two waters:
    the IRC end on the side of the start is the start again within 3e-9 Hartree).
- **A periodic cell** (built, 2026-11-20): minimum-image distances in the bonds, the AFIR
  pairs, the comparisons, and the paths (every atom of a point at the image nearest to that
  of the point before); the translations along the periodic directions are the symmetries
  left out of the optimizations, vibrations, and IRC. Dissociation channels, the nanoreactor,
  and entrance channels are off in a cell.
  - **Variable cell** (`variable_cell` in step 1, given to the other steps; `cli/mapper_jobs/
    cell.rs`): every step moves the cell with the atoms under the external pressure, on the
    enthalpy `E + P V` in the variables of `optmain -vc` (the AFIR push moves the cell through
    its share of the stress; minima and transition states by the optimizer of plain
    variables; two-point searches on the variables of their first structure; the IRC of
    `ircmain -vc`). Node energies are enthalpies; the vibrations of the kinetics are those of
    the atoms in the cell of the structure. Two structures in different cells are one node
    only where a basis of the lattice of the first (integer combinations of its vectors with
    coefficients -1, 0, 1), turned rigidly, agrees with the lattice vectors of the second
    within the RMSD threshold in every component, and their atoms are then compared at their
    fractional coordinates in that basis, in the cell of the first: a crystal turned as a
    whole, or with its vectors in another order, is the same node. Of the bases that fit,
    those whose fit differs by less than a hundredth of the threshold count as equally good
    and the one nearest the identity is tried first: an optimized cubic cell is cubic only to
    its convergence, every operation of its lattice fits it about as well as the identity,
    and the operations that the comparison of the atoms tries are those of the cell within a
    much tighter tolerance (a tetragonal cell then), so that a wrong basis was not undone and
    the same crossing point reached from two minima was two nodes (found 2026-11-20 on two
    spin states in a variable cell; RMSD 1.58 Bohr between structures 1e-5 Bohr apart).
    Forces on the cell itself (`cell_pushes`) are tasks of a minimum (built). With several
    states the crossing points, the AFIR paths towards them, the meta-IRCs, and the minima
    on the other state move the cell too: the two states are combined in the variables of
    the cell (`workflow::irc_cell::CellStates`, the enthalpies of the states and their
    gradients in the variables), on the gradient projection of a path the push stands with
    the enthalpy of each state, and the energy of a crossing point is the mean enthalpy of
    the two states (built 2026-11-20; before, the crossing searches held the cell of their
    start).
  - Planned: the identity of crystal structures independent of the choice of the cell beyond
    those bases (supercells, other reduced cells; stage 3).
- **DC** (built): the same DC is the same multiset of fragments, each given by its bond
  graph, composition, the handedness of its tetrahedral centres, and charge and multiplicity.
  How the fragments are oriented, and the path that led there, do not matter; a DC from
  another path is the same node. Two enantiomers are two fragments, as two mirror images are
  two minima (one with `mirror=true`): the handedness is that of the SMILES string of the
  fragment (`@`, `@@`), so a fragment without a tetrahedral centre of four different
  neighbours is its bond graph alone. (Found on the identity SN2 reaction Cl- + CH3CHFCl,
  2026-11-19: with the bond graph alone the complexes of the R and of the S molecule had one
  channel, which joined the two without a reaction, and a search for the way from one to the
  other ended before its first job.) The E and Z isomers of a double bond are two fragments
  too, with or without `mirror=true` (the marks `/` and `\` of the string; built, user,
  2026-11-19).
  - A fragment is a group of molecules in contact (connected parts of the bond graph whose
    nearest atoms are within the sum of their van der Waals radii and `channel_margin`, 1
    Angstrom by default). An atom that left a cluster gives the cluster and the atom.
  - Every kind of fragment (bond graph, handedness, charge, multiplicity, level) is optimized on its own
    once and kept in the network (`fragments`, the cache); a lower energy of the same fragment
    found later lowers every DC with it, and the barriers of its edges.
  - The charges and multiplicities of the fragments: 0 for every fragment of a neutral
    structure, or the whole charge on one fragment (each tried); every fragment at the lowest
    multiplicity of its electrons, raised by 2 for as few fragments as the spins need to
    couple to the multiplicity of the whole; of several candidates the lowest sum of energies.
- **Chemical species** (built for the report and `nodes.csv`; the high layer of an ONIOM
  network is built: the strings of its cores): the minima with one SMILES string are the
  conformers of one species.
  The string is written from the structure (`workflow::explore::smiles`: the bonds of the
  identity test, bond orders from the usual valences, the usual ions and the charge of the
  structure, canonical by individualization and refinement; `@` and `@@` at tetrahedral
  centres of four different neighbours, so that R and S are two species (user request,
  2026-11-19); `/` and `\` at the double bonds outside rings whose two ends each have
  neighbours that differ, so that E and Z are two species (user request, 2026-11-19); no
  aromatic symbols). It is a view over the nodes, not a kind of node: the network, its
  identity test and the exploration do not change, so two conformers stay two minima with
  their own transition states. The report lists the species (conformers, the lowest, the sum
  of the populations) and, for every two species joined by transition states, the lowest of
  them above the lowest conformer of each; it draws them as a graph (a node per species, an
  edge per two species that transition states join), and a node of the graph is dragged to another place
  with its edges; its figures show the conformers of
  one species alone when asked (the map, the graph, the circle, the disconnectivity graph and
  the energy diagram; the last two are made when the page is written, for the 12 lowest
  species). The circle has a second form with a circle per species and the links between
  species as lines from circle to circle (user request, 2026-11-19). Transition states, crossing points and saddle
  points have no string.

## Edges

Every edge has one bottleneck (a TS, a crossing point, or nothing) and a list of ends. **An end
may be unknown** (`None`), and **there may be more than two ends**, so nothing in the code may
assume two minima per edge.

| Edge | Joins | How it is made | State |
|---|---|---|---|
| TS edge | a TS and its IRC ends (EQ or DC) | IRC both ways from the TS, each end minimized or tested for separation | built |
| TS edge with an unknown end | a TS and the ends of its IRC that are known (none, one) | an endpoint unresolved by minimization, vibration analysis, or a linked saddle search has no node or barrier (`None`). The TS and any verified connection stay in the network | built (`open_ts_edges`; user decision, 2026-11-20) |
| crossing edge | a crossing point and the minima on its two sides | a minimization on each of the two states from the crossing point | built (`states`) |
| barrierless dissociation | an EQ and a DC | a complex of two molecules and the DC of the two (`dissociation_edges`) | built for complexes; from a path that rises without a maximum (an AFIR pull, an ADDF path) planned |
| TS–TS link (VRI) | TS1 and TS2 | an IRC side of TS1 that ends at a saddle (an imaginary mode of magnitude at least 50 cm^-1 at the optimized endpoint) instead of a minimum: TS2 is optimized at order 1 (or matched with a known TS) and followed in its turn, and the TS edge of TS1 has TS2 as that end | built (`workflow/mapper/follow.rs`) |
| path peak (a record, no edge of the network) | the peak of a path and the minima found by descent from it | a path whose guesses gave no verified TS | built (`path_peaks=failed`, the default) |

- **TS edge ends.** An IRC side is tested for separation before its end is minimized:
  - separated fragments → the end is a DC, not minimized (built); an end whose fragments
    part during its minimization is a DC too (built);
  - a minimum of two molecules in contact → a complex EQ, plus its barrierless dissociation
    edge (built). A minimum of three or more molecules in one group has none: its channels
    are stepwise, not the complete separation (an atomic cluster without covalent bonds is
    such a group);
  - At the optimized endpoint, any imaginary vibration at or above `irc_end_threshold_cm1`
    triggers refinement of TS2. A near stop along the IRC with a single negative curvature
    can also identify a ridge the path left before reaching its endpoint. In that case the
    stored side is cut at TS2. Chains TS1 → TS2 → TS3 are followed up to three links deep.
  - an end whose minimization does not converge (and whose fragments have not parted) →
    unknown (built; user decision, 2026-11-20). The other side is followed as well, the TS
    is a node of the network with its IRC and the end that is known, and the edge is kept
    apart from the TS edges (`open_ts_edges` of the network file and the database) with a
    warning. Before, the TS was dropped with the warning and only the peak of the path was
    kept. A TS found again later is the known node, so its IRC is not followed again.
  - The barriers of a TS edge with a DC end come from the energy of the DC (the sum of its
    fragments), not from the energy of the IRC end.
  - An input that is a complex of two molecules gets its DC and barrierless edge as well.
- **Shallow minimum or real TS–TS link.** Before a TS–TS link is recorded, a minimum of the
  gradient norm along the IRC (a shoulder) is optimized tightly once. If an EQ comes out,
  there are two ordinary edges instead. Built: the shoulder is the first local minimum of the
  slope of the energy along the side below a hundredth of its largest value (the slope is the
  gradient norm along an IRC), and the minimum counts when it lies above TS2; TS2 is then a
  reaction of its own.
- **Bifurcations and N-furcations.** One TS can lead to N minima through one or more VRIs
  along its IRC, chains TS1 → TS2 → TS3, or an IRC end at a saddle of order k (up to 2k
  descending directions, each minimized). These become one bottleneck with N ends.
- **Path peaks** (built; user decisions, 2026-11-19). `path_peaks=failed`, the default: the
  peak of a path is saved when the job got no transition state from it, for whatever reason,
  and not when it did; `all` saves the peak of every path, also of one that gave a
  transition state; `off` saves none. A saved peak costs the two minimizations of its
  sides. **They are saved and are not in the network**
  (user decision, 2026-11-19): a path peak is no node and no edge, the figures and graphs of
  the report do not draw it, the kinetics does not use it, and it joins nothing (the goal of
  a goal-directed search is joined by TS edges only, and `method=connect` still tries a pair
  between which only a path peak lies). What is kept is a record of what the job found, for a
  later look or a later search from it.
  - **When one is kept.** A job whose path gave guesses, none of which led to a transition
    state (a new one, or one the network knows; with `all`, every job with a path): the
    highest inner maximum of the energy along the path is the peak. The path is the last one the job has: the chain relaxed by
    `method_relax` (or the AFIR path itself with `method_relax=off`), the path of the
    two-point method of `method=connect`, of `method_step1`, or of a remap.
  - **Its ends** come from the nearest minima of the path energy on both sides of the peak
    (from the peak downwards along the path as long as the energy falls), each minimized and
    matched with the known minima and dissociation channels like the end of an IRC (a minimum
    found so is a node like any other). An end whose minimization does not converge is
    `None`. A peak whose two sides fall to the same minimum is not kept.
  - **The record** (`path_peaks` of the network file, `path_peaks.csv`): its number (`PP0`,
    `PP1`, ...), structure (`structures/PP000000.xyz`) and energy, its ends and the heights
    above them, the job, and its basis: the method of the path, the number of its points,
    whether the path converged, and how many transition-state searches from its maxima were
    tried and failed. The path between the two ends through the peak is
    `paths/PP000000_path.xyz` (the minimization of the first end backwards, the points of the
    path, the minimization of the second end). A peak that is the same structure as a
    recorded one (the identity test) is not recorded again. The table says for every record
    whether a TS edge joins its two ends by now.
- **Higher-order saddles** (experimental, `higher_saddles=on`, off by default; user decisions
  of 2026-11-18). After the solution landscape of Yin, Zhang and coworkers (a map of the
  stationary points of every index and their links, built by downward and upward searches
  between index k and k ± 1) and the work on index-2 saddles in chemistry (Heidrich and
  Quapp; Minyaev; Bofill and Quapp's gentlest ascent of any index):
  - **Where they come from, at no extra cost:** a TS search from a guess that ends with k ≥ 2
    imaginary modes (now only a warning), and a side of an IRC that ends, or comes to a near
    stop, at a point whose Hessian has k ≥ 2 negative curvatures. Each is optimized as a
    saddle point of order k (RS-I-RFO with `saddle_order` k, the settings of `step3_settings`)
    and matched with the known ones: a `kSP` node.
  - **Downward search:** for every unstable mode, the saddle point displaced by
    ±0.1 amu^(1/2) Bohr along it and optimized at order k - 1 (keeping the other unstable
    modes), down to transition states whose IRCs then run as for any TS, and to minima from
    an SP1. Each mode gives an edge from the `kSP` node to its two ends (saddle points of
    order k - 1, or minima), with the imaginary wavenumber of that mode: a `2SP` sits between
    two TSs (for example a synchronous saddle above two asynchronous transition states).
  - **The edge of the transition state whose IRC reached it** ends at the `kSP` node, as at
    TS2 of a TS–TS link.
  - **Upward search by ADDF** (an exploration method of its own, off by default; planned): from a
    transition state, ADDF on its dividing surface, the hypersphere minimizations of ADDF in
    the space of the other modes with the energy maximized along the mode of the transition
    state. The transition state is a minimum of that surface and a `2SP` one of its saddle
    points, so the anharmonic downward distortions of the dividing surface lead to `2SP` points, as
    those of the surface lead from a minimum to transition states. Checked on LJ7 (saddle
    points of higher index are known), the ring opening of cyclobutene (a `2SP`), and a model
    surface before use.
  - **The steps** (user decision, 2026-11-18), like those of AutoTS one index up; the methods
    of steps 1 and 2 are chosen by name from a registry, the defaults by measurement:
    | Step | What | Methods |
    |---|---|---|
    | 1 | exploration | ADDF on the dividing surface of a TS; the upward search of high-index saddle dynamics (HiSD, Yin, Zhang and coworkers: from an index-k point pushed along its next mode, HiSD with k + 1 unstable directions); and the `kSP` the mapper meets anyway (above) |
    | 2 | relaxation | HiSD (the shrinking-dimer form, gradients only) or the generalized gentlest ascent of Bofill and Quapp with the k unstable directions, stopped at a loose threshold, from the estimate of step 1; or, where step 1 gives a path between two transition states, NEB on the dividing surface (every image maximized along its unstable mode, the highest image the estimate) |
    | 3 | refinement | RS-I-RFO at `saddle_order` k with the tight criteria, the index checked by the Hessian (exactly k negative curvatures) |
    | 4 | descent | the downward search above: ± each unstable mode, optimized at order k - 1, down to transition states and their IRCs |
  - Cost: one optimization at order k and 2k optimizations at order k - 1 per `kSP`; with the
    setting off the points are treated as before.
  - **How it is built** (2026-11-18; experimental; `workflow/mapper/follow.rs`, `upward.rs`):
    - The node kind `HigherSaddle { order }`, tag `2SP`, `3SP`, ... (orders 2 to 9), numbered
      apart for every order; identity as for TS (bond graph, energy, RMSD) within one order.
    - The edge `saddle_edges` (network files, database table `saddle_edges`): the `kSP` node,
      the unstable mode (its number, lowest first, and its imaginary wavenumber), and its two
      ends from `+` and `-` (a saddle point of order k - 1 or, from k = 1 below, a minimum;
      `None` where the descent did not settle), the job.
    - `higher_saddles=on`: a TS guess whose search ends at a point with k ≥ 2 negative
      curvatures (by the start check of its IRC), and an IRC side that ends or comes to a near
      stop at a point whose Hessian has k ≥ 2 negative curvatures, are optimized at order k
      (the settings of `step3_settings` with `saddle_order` k); a point that keeps exactly k
      negative curvatures is a `kSP` node. Orders above `max_saddle_order` (2) are not
      followed (a warning says so).
    - **Without `higher_saddles`** (the default; user decision, 2026-11-18): a `kSP` the
      mapper meets is still kept as a node, but its links are not followed. A TS search that
      ends at a point with k ≥ 2 negative curvatures keeps the point and goes on by mode
      kill: the point is displaced by 0.1 along each of its unstable modes but the lowest and
      optimized at order 1 following the lowest; the transition state found is followed as
      any. For a variable cell the displacement and followed mode both come from the projected,
      mass-weighted atom/strain Hessian of the enthalpy. If refinement returns to the original
      structure or fails the index-one check, try the opposite displacement, then double the
      excursion from the original point, up to the mode-kill round limit. An IRC side that
      ends at such a point keeps it (optimized at order k) and the end
      of the side is minimized as before.
    - **A periodic cell that came apart** (built, user 2026-11-20): a minimum whose
      molecules in contact (with the images; the contact of the fragments of `channels`) span
      fewer directions than the cell is periodic in (the rank of the lattice translations of
      the closed paths of the contacts and bonds: layers, chains, molecules apart) is a `DC`
      node with its own structure and enthalpy and no fragments, the reason on the node, the
      structure matcher for its identity, not explored; kept even with imaginary modes (a flat
      saddle point of the cell it was pulled into is saved all the same).
    - **One criterion of an imaginary mode** (every count of the mapper: minima verified,
      transition states and the index of a saddle point, the mode kill of its optimizations,
      and the start of an IRC): an eigenvalue of the mass-weighted Hessian below minus `1e-8`
      Hartree/(Bohr^2 amu), MultiOptPy's `saddle_check`, about 0.51 cm^-1
      (`opt::mode_kill::IMAGINARY_CURVATURE`), with the rigid motions projected out in the
      mass-weighted space where the surface does not depend on them; plain variables have unit
      masses, a variable cell the masses of the variables of its IRC. The saddle report uses
      the modes counted by the index Hessian. For a variable cell these are coupled atom and
      strain modes. Vibrational thermochemistry uses the fixed-cell atomic spectrum.
      Modes flatter than 1 cm^-1 are tolerated (user, 2026-11-20), and the criteria of minima
      and saddle points are one (user, 2026-11-20: 「なぜ基準を分けるんだ」). It replaced a
      threshold on the curvature without masses (1e-4 Hartree/Bohr^2, or 1e-3 of the largest
      curvature magnitude at the point on soft surfaces), which let collective modes of whole
      molecules pass as minima: four structures of a benzene crystal pulled apart by GFN-FF kept
      imaginary frequencies of -2 to -11 cm^-1 as minima, while the transition states were
      counted with the threshold of the IRC. With masses the Ar7 transition states (curvatures
      -1.2e-5 to -9.2e-5 Hartree/Bohr^2 with `lj`, about -3 to -8 cm^-1) stay transition
      states.
    - **Descent** along each of the k unstable modes `u_i`: the starts `x ± 0.1 u_i`
      (amu^(1/2) Bohr, mass-weighted; the step of the IRC start), each optimized at order
      k - 1 with the other k - 1 unstable modes followed (RS-I-RFO with mode following,
      `mf_rsirfo_block_bofill`), so that the optimizer descends along `u_i` and keeps the
      others uphill. Ends at order 1 are transition states (a known one matched, a new one
      followed with its IRC); ends at order 0 are minimized and matched as IRC ends are; ends
      at order 2 or more are `kSP` nodes descended in their turn; every saddle point once per
      job.
    - **Upward search** (`upward=on`, which turns `higher_saddles` on): the tasks of every
      minimum get first, for each transition state joined to it by an IRC, its `upward_modes`
      (2) softest stable modes, both ways (`upward:TS:mode:sign`; the tasks follow the
      network, so they run as soon as a transition state is known, before the remaining tasks
      of the method).
      The job starts at the transition state displaced by 0.1 Bohr along the mode
      (`path::hisd::landscape::upward_starts`) and runs high-index saddle dynamics of index 2
      (HiOSD, the shrinking-dimer form, gradients only, mass-weighted, the rigid motions left
      out) with the unstable mode and the pushed one. Its force criteria are a hundredth of
      those of the TS search (`HISD_CRITERIA_SHARE`; with the default `NORMAL` thresholds every
      search on argon stopped at its first point with the wrong index), and it is handed over
      at that RMS force with the subspace turning by less than 0.1 (the value of
      `high_index_saddles.md`) to the optimization at order 2 with its directions followed;
      the `2SP` found is descended from as above. Steps 1 (HiSD upward), 2 (HiSD to the
      hand-off), 3 (RS-I-RFO at order k, index by the Hessian) and 4 (descent) of the table
      are built; ADDF on the dividing surface and NEB on it are planned.
    - **Confirmation of a possible bifurcation** (`confirm_bifurcations=on`): at the point of
      the IRC where the curvature across the path was negative the Hessian is computed again
      with the gradient direction taken out, the point displaced by ±0.1 amu^(1/2) Bohr along
      the lowest mode across the path, and both minimized and matched. Minima other than the
      end of the side are recorded on the possible bifurcation (`confirmed_ends`), with a
      warning.
    - Checked on `V = (x^2 - 1)^2 + cos(pi x) (y^2 + z^2) + y^4 + z^4 + 0.5 y^2 z^2`
      (`tests/cli_mapper_saddles.rs`: the IRC of the TS at the origin ends at the two `2SP`
      (±1, 0, 0); 4 descents to the 8 TSs of the planes x = ±1, 8 minima; restarts after stops
      at four points give the same network; the upward search from a TS of the plane x = 1
      reaches the `2SP` (1, 0, 0)) and on the bifurcating model of
      `tests/cli_mapper_bifurcation.rs` (the confirmation finds both products of TS2); a TS
      search whose guess is the `2SP` keeps it and reaches one transition state by the mode
      kill. On LJ7 (16 jobs of `afir_ts` with `upward=on` from the pentagonal bipyramid, debug
      build): the upward searches reached 8 `2SP` (-14.27 to -15.03 epsilon, among them
      the index-2 point of Quapp and Bofill at -14.723), one point of index 3 (warned), and one
      search left the index; of the 16 descents along their modes 12 reached a transition state
      on both sides, the others went down to a minimum or did not converge in 200 steps. The
      network has all 4 minima of LJ7 (the one at -15.533 twice, as its two mirror images) and
      8 transition states. Before the threshold was scaled the same searches reached the same
      points, and all were counted as index 0 or 1 (the refinement stayed there in 2 steps).
    - **Report**: the saddle points of higher order have a table (order, energy, the number of
      imaginary modes and their wavenumbers, the descents) and are drawn in the Graph and the
      Map as squares above what their descents reach, joined by dashed lines. Every saddle
      point (TS, `kSP`) shows the number of its imaginary modes and each wavenumber (user
      request, 2026-11-18).
- **Barriers** are stored from both ends of an edge, in Hartree, at the level of the edge.

## Entrances: how nodes come into the network

| Entrance | What it adds | State |
|---|---|---|
| inputs | one or more structures (files, frames, wildcards), each optimized; inputs ending in the same minimum are one node; atoms, order, charge and multiplicity must agree | built |
| exploration methods | the reactions a job finds from a minimum: `afir_ts` (AFIR pushes, each path relaxed by the chain of `nebmain` with model Hessians before the guesses are taken; `method_relax=off` skips it), `addf` (ADDF), `nanoreactor` (MD in the nanoreactor walls, then a path method), and methods registered by users | built |
| points of plain variables | a network of an objective of plain variables (`-os mb2d`, any number of variables): every node is a point of the variables (`Configuration::Variables`); two points are the same within `variable_tolerance` (Euclidean, 0.01) and the energy tolerance; bond graphs, AFIR pushes, dissociation channels and the nanoreactor need atoms and are refused or left out; the default method is `addf`; files are variables files and frames of `name value` lines; the report draws the variables (the surface behind them for two) | built |
| re-mapping on another surface (`-remap`) | the minima, transition states with their paths, crossing points and saddle points of higher order of an existing network (its run directory or database) taken to another surface: path relaxation, then TS optimization, then IRC, joined again, in a run directory of its own ([below](#a-network-taken-to-another-surface--remap-and--single_points)) | built |
| entrance channels | a DC as the source of bimolecular reactions (the separated reactants): `entrance_structures=N` makes N structures of the channel of the input (its molecules turned and placed at random, in contact) and every one is an input of the run | built (below) |
| conformers of the transition states | `ts_conformers=on`: one job per transition state, a conformer search with its saddle direction held (`confsearch -tsconf`), every saddle point found followed like any guess | built (below) |
| step 1 by another two-point method | `method_step1=NAME` of `afir_ts`: the path of a pair of atoms from a method that goes on from the minimum along the pair, without a push (`se_gsm`, `gnt`) | built (below) |
| networks of other runs | `-merge RUN_DIR ...`: the networks of several runs on one surface as one network, without a calculation | built (below) |
| goal-directed search | a given product structure (the goal): the network grows from the inputs towards it, alternating a fair random expansion (a node from a cluster of nodes with the same bond graph, chosen uniformly) and a goal-oriented one (a node by its similarity to the goal, local atomic environments matched greedily); a new node is explored only with the probability `min(1, exp(-(E_path max - E_node) / kT))`; the search stops when the goal is found. After Nakao, Harabuchi, Maeda, Tsuda, Phys. Chem. Chem. Phys. 2022 (DOI 10.1039/d2cp01079h, RRT combined with SC-AFIR) | built (`priority=rrt`, `goal=FILE`; below) |
| missing connections | two given minima far apart (a start and a goal, both inputs): every two known minima are joined in a complete graph; a pair joined by a TS edge costs nothing, an unjoined pair a monotonically increasing function of its distance after the best superposition (`D^2` in the paper), and a pair whose attempts are used up costs infinity; Dijkstra's method gives the chain of the least total missing connection; every gap of it is attempted with a method of the two-point registry (`pair_search`; the paper uses the doubly nudged elastic band), its estimates followed to TSs and IRCs as by the other methods; repeated until the start (or a minimum joined to it) and the goal are joined. After Carr, Trygubenko, Wales, J. Chem. Phys. 2005, 122, 234903 (arXiv:cond-mat/0504785) | built (`method=connect`, `goal=FILE`; below) |

- **The goal** (both entrances above; built, `workflow/mapper/driver.rs`, `network.rs`). `goal=FILE` gives the goal structure (the
  first frame of an xyz file, with the atoms, their order, the charge and the multiplicity of
  the inputs). It is optimized and verified like the inputs and enters the network as an
  input marked `goal` (`inputs[k].goal` in the network files), so that a minimum found later
  that is the same as it is that node. It is never explored (kept, with the reason "the goal").
  The run ends when the goal node is joined to the node of a start input by edges of the
  network (TS edges, TS–TS links, barrierless dissociations), unless `goal_stop=off`; the log
  and the network (`run.goal`) record it. A goal that optimizes to a start is joined at once.
  With `goal_match=bonds` a minimum with the bond graph of the goal (another conformer of it)
  joined to a start ends the run as well: the product an IRC reaches is often another
  conformer than the structure given, which the identity test (RMSD) tells apart.
- **Goal-directed search** (`priority=rrt`, which needs `goal`; built, `workflow/mapper/goal.rs`). After Nakao,
  Harabuchi, Maeda, Tsuda 2022. Option names do not name a program.
  - The jobs alternate by their number: even numbers take step A, odd numbers step B (with
    several jobs chosen together the alternation is that of their numbers).
  - **Step A, fair random expansion:** the minima that can be explored are grouped by their
    connection pattern (the bond graph with bonds up to 1.1 times the sums of the covalent
    radii, the scale of the paper, labelled by element, equal up to a renumbering of the
    atoms); a group is chosen with equal probability, then a member with equal probability.
  - **Step B, towards the goal:** each minimum has a similarity `z` to the goal (below); a
    group is chosen with probability proportional to `exp(c s)`, `s` the largest `z` of its
    members, then a member with probability proportional to `exp(c z)`; `c` is
    `rrt_goal_weight` (1, the paper). The weights are drawn as logarithms (log-sum-exp).
  - **Similarity of local environments:** for every element, every atom `i` of the first
    structure and `j` of the second (of that element) are compared by their distances to the
    other atoms: within each element `e`, the matrix `|d_ik - d_jl|` (`k`, `l` of element `e`,
    not `i`, `j`) gives its minimum value profile (the smallest entry, its row and column
    removed, repeated), and `S_ij` is the mean of `exp(-theta v)` over the values of the
    profiles of all elements (`theta` = `rrt_locality`, 20 per Angstrom, the paper). The
    similarity of the element is the sum of the maximum value profile of `S`; the similarity
    of the structures is the sum over the elements, between 0 and the number of atoms. The
    paper leaves which atoms `k`, `l` run over to its supporting information (not read here);
    here they are all other atoms, matched within their elements, so that the environment of
    an atom holds its distances to every other atom.
  - **Acceptance:** a new minimum found by a job from the minimum `X` is explored only with the
    probability `min(1, exp(-(E_PT - E_X) / (k T)))`, `E_PT` the energy of the highest
    transition state on the way of that job from `X` to it (the lowest such way when the job
    found several), `T` = `accept_temperature` (7500 K, the middle of the 5000 to 10000 K of the
    paper; the measurements in [mapper.md](mapper.md) on a chain of wells and a grid of 64
    wells disagree on whether the filter helps, and the urea synthesis of the paper with
    GFN1-xTB, 40 jobs and two seeds, did not decide it: no run reached the goal and the filter
    refused one to three minima of about thirty. The default is the convention of the paper,
    not a measured value; a run long enough to reach the goal is still to be measured). Otherwise it is kept but not explored (its reason gives
    `p`); a later job that reaches it again (through a transition state it adds or finds
    again) draws again, as the paper discards a refused node and finds it again later. Without
    the second draw a refused minimum on a chain would close the way behind it for good. The
    filter orders the search and never ends it: when no accepted minimum has a task left (a
    method with one task per minimum, ADDF, reaches nothing again), the refused minimum with
    the highest `p` (the lowest barrier) is explored, with a warning. The draw uses a generator seeded by the seed of the run, the job and the node, so a restart
    draws the same. `accept_temperature=T` turns the filter on with any priority; with
    `priority=rrt` it is on by default.
- **Missing connections** (`method=connect`, which needs `goal`; built,
  `workflow/mapper/connect.rs`). After Carr,
  Trygubenko, Wales 2005:
  - The known minima (level 1, not DC) form a complete graph. The weight of a pair is 0 when
    a TS edge joins them; infinite once `method_attempts` (`n_max`, 2) jobs have tried the
    pair, or when neither of the two may be explored (a job runs from one of them); otherwise
    `D^2`, `D` the distance after the best superposition and the best numbering of identical
    atoms (Bohr; atoms renumbered within their elements by the Hungarian method, as the paper
    minimizes over permutation-inversion isomers; no mirror image, which a path cannot reach;
    the Euclidean distance for plain variables). The two-point method of a gap runs to the far
    end with its atoms in that numbering, so that the path does not exchange identical atoms.
  - Dijkstra's method from the starts (the nodes of the inputs that are not the goal) gives
    the path of least weight to the goal; every pair of the path with a weight above 0 is a
    gap. The tasks of a minimum are the gaps of the current path that it runs (the end nearer
    the start, or the other end when that one may not be explored), `connect:A:B:k` (`A < B`,
    `k` the attempt). The path is computed again after every job, so a gap of an older path
    that is no longer on the path is not tried (the paper tries every gap of a path per
    cycle; with as many workers as gaps a batch is one such cycle).
  - A gap job runs the two-point method `method_path` of the registry `pair_search` (`neb`
    by default, its settings `method_path_KEY`; the paper uses the doubly nudged elastic
    band) between the two, and every maximum of its path is a guess followed to a TS and its
    IRC, as for the other methods. Attempt `k` of a pair runs the method with `2^(k-1)` times
    its images (`method_path_images`, 10 when not given) where it takes images, so that a
    repeated attempt is not the same calculation; the paper does not say how its attempts
    differ.
  - The run ends when the goal is joined to a start (the stop of the goal), or with no gap
    left that can be tried (every path has a pair whose attempts are used up).
- **Which minima are explored.** The priority chooses among minima that still have tasks. The
  user can narrow that with:
  - `excluded_node_ids` (built);
  - `exclude_bond_rearrangement` (built);
  - `max_bond_changes=N` and `max_species_changes=N` (built; user, 2026-11-19): a new minimum
    is not explored when, on the way from its input (the chain of the minima the jobs started
    from), the bonds changed more than N times, or the chemical species changed more than N
    times. A step counts for the bonds when the bond graphs of the minimum and of the
    minimum its job started from are not the same graph (whatever the numbering of the
    atoms), and for the species when their SMILES strings differ (so a change of R/S or E/Z
    counts there, and a rearrangement that gives the same molecule again does not count in
    either). With 0 nothing whose bonds (or species) changed is explored: the search stays
    among the conformers of the inputs, and what they react to is kept as the ends of their
    reactions. The count is that of the job that found the minimum first. Not for plain
    variables;
  - `explore_lowest=N`, only the N lowest minima (planned);
  - `explore_inputs_only` (planned).
  A DC is never explored.
- **Dissociation tests.**
  - ADDF ends a path when its structure has more fragments, with bonds up to
    `method_dissociation` (2.0) times the sums of the covalent radii. That is the definition
    the documentation of the method gives for growing spheres. `method_dissociation_atoms`
    limits the test to chosen atoms. Built.
  - The separation test for DC nodes: nearest atoms of two fragments beyond a margin over the
    sum of their van der Waals radii (Alvarez's radii; `channel_margin`, 1 Angstrom, so that
    hydrogen-bonded and van der Waals complexes stay complexes; `channels=off` turns DC nodes
    off). Built; the default margin is a judgement, not yet measured, and a test by the
    interaction energy is planned.
- **Guards.** These are built:
  - A trajectory that blows up (an atom moves more than 1 Bohr in one step) ends there, with
    a warning.
  - A search refuses structures with an atom more than 200 Bohr from their centre; the job
    fails and says why.
  - What a method refuses about its start (ADDF from a structure that is not quite a minimum)
    fails that job alone; the run goes on.

## After the goal-directed search

Decided (user, 2026-11-17), to be taken up after the goal-directed search and the missing
connections above; each item says its state (as of 2026-11-20). Defaults come from
measurement, as for the rest of this page.

- **Already decided:**
  - TS conformers: a conformer search that keeps the saddle direction constrained, the lowest
    TS conformer giving the barrier. Built (`ts_conformers=on`, below).
  - The search inside one molecule or one cell: the stochastic choice of the minimum to
    explore is the exploration priority (`priority`: `boltzmann`, `rrt`, `kinetics`; built);
    still to do are the end condition of such a search, the clustering of crystal structures,
    and a surface mode, which the periodic checks need (benzene crystal with GFN-FF, metal
    surfaces with EAM). Pushes between whole molecules as fragments are built
    (`afir_fragments = molecules`, 2026-11-20, for molecular crystals as in Kondo et al.);
    other fragments generated automatically (functional groups, parts of one molecule) are put
    off (user, 2026-11-19).
  - Replica exchange for the nanoreactor (temperatures or wall strengths). Built
    (2026-11-20; `method_replicas`, [mapper.md](mapper.md)): segments with exchanges of
    neighbouring rungs by Metropolis, each replica's own trajectory searched for reactions.
- **New:**
  1. A single-point level 2: the structures and frequencies of level 1 with the energies of a
     single point at level 2, for every node, the kinetics solved again with them. Built as
     `-single_points` ([above](#a-network-taken-to-another-surface--remap-and--single_points)):
     the mixed energies are a network of their own (`2//1`), so the rule that no node or edge
     mixes levels holds.
  2. A search towards a given product (a structure or a bond graph): minima nearest to it in
     bond-graph distance are explored first, and once one is close enough a two-point method
     of `pair_search` joins it to the product. Built in another form: `priority=rrt` and
     `method=connect` with `goal=FILE` ([above](#entrances-how-nodes-come-into-the-network);
     `goal_match=bonds` for a bond graph).
  3. An energy window: only minima whose lowest-barrier path from the inputs stays at or below
     `W` are explored, and an AFIR estimate above the window is not optimized to a TS.
     Planned (the `energy_window` of the report only limits what the page draws).
  4. Atom pairs related by a symmetry permutation (`vib::symmetry`) give the same reaction, so
     one pair of each set is pushed. Planned.
  5. An estimate of the minima not yet found from how often each EQ and TS was found again
     (Chao1, from the numbers found once and twice), in the log and the report and usable as
     a stop rule. AFIR does not sample at random, so the estimate is a guide; its error is
     measured on LJ7 before it is used. Planned.
  6. Level 2 for chosen edges only: those with the largest degree of rate control, or those on
     the lowest-barrier path between two given minima. Planned (`-remap` and
     `-single_points` take the whole network).
  7. The Hessians of the EQ and TS checks kept in files (moved to disk for large systems, as in
     NEB), so isotopologue frequencies and kinetic isotope effects (with Wigner or Bell) need
     no further electronic structure. Planned (the nodes keep their frequencies).
  8. Steering a running search (a file in the run directory or an MCP tool raises or stops a
     minimum) and merging runs made apart (other inputs or seeds) into one network with the
     same identity test. Built: `steer.txt` and `-merge` (below), and the MCP tool
     `mapper_steer`, which appends checked lines to `steer.txt` of a running job.
  9. Smaller items: a filter of reactive atom pairs (element pairs, functional groups,
     charges), and GraphML output. Planned.
  10. Saddle points of higher index with the nonadiabatic paths (user, 2026-11-18): the
      saddles of index k of `path::hisd` (HiSD, HiOSD) on each electronic state, together with
      the crossing points (MECP, MECI) and the Zhu-Nakamura surface hopping of `mdmain`, to map
      the paths a molecule takes after photoexcitation as a whole: which minima and crossings
      of the excited states connect through which saddles, and the topology of the surfaces
      around a conical intersection (the index-2 saddles that bound its funnel, the branches of
      a seam). Its nodes, edges and identity rules are written here before it is built.
      Planned.
- **Entrance channels** (built; user, 2026-11-19). A dissociation channel as the start of a
  bimolecular reaction. In published kinetics on such networks (the user's account; the
  papers are not named here) the start is not one structure of the separated reactants but
  many of one channel (of the order of a hundred, all the same channel by its fragments),
  each with the same share of the initial population (1 % each for a hundred).
  - `entrance_structures=N` (0, the default: off): the first input must have two or more
    molecules (by its bond graph). N structures are made from it: every molecule kept rigid,
    turned by a rotation drawn uniformly, and each molecule after the first placed in a
    direction drawn uniformly from the centre of those placed before, at the distance at
    which its nearest atom to them is at `entrance_contact` (1.0) times the sum of the two
    van der Waals radii. The generator is seeded by the seed of the run, so a restart makes
    the same structures. They are the inputs of the run after the given ones, named
    `FILE (entrance structure k)`.
  - Every one is minimized like an input: a complex (an `EQ` with the barrierless edge to the
    channel of its molecules, as for any input of several molecules), or whatever it falls
    to without a barrier. Inputs that end in the same minimum are one node.
  - The network records the entrance (`entrance`: the DC node, the number of structures and
    the minima they gave). One DC node stands for all the structures of the channel (a DC is
    its fragments, not their orientation), so the equal shares of the published scheme add up
    to the whole population on that node: the kinetics of such a network starts with all of
    it on the entrance channel unless `--start-node` or `--initial` says otherwise.
  - Not for plain variables, periodic cells, or a run of several states (refused).
- **Conformers of the transition states** (built; user, 2026-11-19: `ts_conformers=on`, off by
  default; one conformer search per transition state is many times the cost of a job).
  - Beside the tasks of the method every minimum has one task for every transition state of
    its TS edges of which it is the end with the lowest number (`tsconf:<node>`): the search
    of `confsearch -tsconf` from the transition state (the reaction coordinate from its
    imaginary mode held, with the reactive bonds, while the rest is sampled; the constrained
    minima within the window optimized to saddle points), with the settings of the block
    `ts_conformer_settings` of the settings file (the options of `confsearch`; the surface is
    that of `step1_settings`, the seed that of the run unless the block has one).
  - The check of the reaction by an IRC inside the search is off: every saddle point other
    than the start is handed back and followed like any transition-state guess (optimized,
    its IRC, the ends minimized), so the conformers of a transition state are transition
    states of their own between the conformers of the minima. The lowest of the transition
    states between two species is their barrier (the report lists it so already).
  - A transition state that such a job found is not searched again: its conformers are those
    of the search that found it.
  - What the search refuses about its start is a warning of the job. Not for plain variables,
    `-remap`, `-single_points`, or a run of several states (refused).
- **Step 1 by another two-point method** (built; user, 2026-11-19). The first path of a pair
  of atoms of `afir_ts` comes from the AFIR push by default; `method_step1=NAME`
  (`method_step1_KEY` its settings) takes it from a method of the registry of two-point
  methods that goes on from the minimum along the pair, without a push and without a second
  structure:
  - `se_gsm`, the single-ended growing string: the pair is its driving coordinate (`ADD a b`
    for a push, `BREAK a b` for a pull with `--negative_gamma`);
  - `gnt`, the growing Newton trajectory: the pair is the bond it shortens (it has no pull: a
    pull task is left out with a note).
  - The maxima of the path of the method (its estimates of transition states, the
    `top_n_candidates` highest) are the guesses; `method_relax` does not run after it (the
    method is the path method), and `step2_settings` are not used.
  - `neb` and `dmf` are refused here: they relax a path that exists and are the methods of
    `method_relax` (user, 2026-11-19). `addf` is refused too: it explores a minimum as a
    whole (`method=addf`).
  - A method of two ends (`gsm`, `fsm`, `dsafir`, ...) is refused as well (user decision,
    2026-11-19): between the minimum and the end of an AFIR push it adds little to the push
    itself. Its place is between known minima and a structure the user gives: `method=connect`
    with `goal=FILE` and `method_path=NAME`.
- **Networks of other runs** (built; user, 2026-11-19): `run_mapper -merge RUN_DIR ...
  --output_dir DIR` writes the networks of several runs as one, without a calculation.
  - The runs must have the same atoms in the same order and the same states; that they ran
    on the same surface is the user's word (the settings of the first are those of the
    merged directory, and a note names every run).
  - The first network is taken as it is. Every node of a later one is matched with the nodes
    so far by the identity test of the run (kind, state, bond graph, RMSD, energy) and added
    when it is new; an edge is added when the merged network has no edge of its kind with the
    same bottleneck and ends; the fragments of the dissociation channels are merged by their
    kind (the lower energy kept); the jobs are appended and renumbered, and the inputs of
    every run are inputs of the merged network.
  - Path files and structures are copied under the new numbers. The merged directory is read
    like a run directory by `run_kinetics`, `-remap`, `-single_points` and the report; it is
    not continued by `--restart` (its jobs come from different runs: start a new run from
    its structures, or draw it again with `-remap`).
- **Steering a run** (built; user, 2026-11-19). A file `steer.txt` in the run directory is
  read before every choice of jobs, applied, appended to `steer.log` with the number of jobs
  told by then, and removed. Its lines:
  - `explore EQ5` (or the number of the node): the next jobs are the open tasks of that
    minimum, before the priority chooses again (several lines: in their order);
  - `skip EQ5`: no more jobs start from it (its reason says "steered"); `unskip EQ5`;
  - `max_jobs N`: the limit of jobs from now on (0: none).
  (A run is ended by the files `STOP` and `stop.txt`, as before.) What was applied is kept in the network (`steering`), so a restart goes on with it. A run
  that was steered is no longer the run its settings alone give: the network says so, and
  the comparison of a restart with an uninterrupted run holds only with the same lines at
  the same jobs.
- **How the goal-directed search and the missing connections fit in:** a goal-oriented
  exploration priority; Dijkstra over the network giving gap jobs run through
  `workflow::pair_search`; growth from both ends with a `connected` stop rule; measured on LJ7
  and an organic reaction.

## Constraints and bias potentials of the user

Owner's ruling (2026-11-24): hard constraints and static bias potentials are part of the
surface in every stage of a run; a time-dependent or stateful bias is refused with the reason.

- **What the surface is.** The surface of a run is the calculator with the user's static
  bias potentials added (`pes::biased::BiasedCalculator`, made once where the run makes its
  calculator, so that every stage and every job inherits it), restricted to the manifold of
  the user's hard constraints. The mapper's own AFIR pushes are terms of the push stage only,
  on top of that surface; they are never a part of it. Built.
- **Where the options come from.** The options shared by all steps (`-fix`, `-pc`, `-gfix`,
  `-uc`, `-vc_fix` and the bias options, the spellings of `optmain`) are given on the command
  line of `run_mapper` or in the block `shared_settings` of the settings file (the command
  line wins over the file); they are merged into every step block (`step1_settings` to
  `step4_settings`, `ts_conformer_settings`). A key that a step block sets itself wins over
  the shared value, and the log says so by name. Built.
- **Resolved once.** The constraints are resolved once, at the first input structure
  (`cli::constraints::resolve_for_run`): every target, frozen position and fixed direction
  is fixed there, and every stage of every job holds those values to 1e-10, in memory and in
  every file the run writes. A further input or the goal is put onto the same values
  (`ConstraintSet::restore`), with a warning of how far it moved; one whose residual cannot be
  removed is refused by name. Eigenvector constraints (`-pc eigvec`) are refused: a network has
  no Hessian at its first input to resolve them at. Built.
- **Every stage holds them.** The AFIR push (an optimization with the constraints and the push
  term), the relaxation of the path (the chain of `nebmain` with `ChainConstraints`; the
  constraint keys of `step2_settings` are part of the run's constraints and are no longer
  dropped), the transition-state search and its verification (the Hessian projected onto the
  tangent space of the constraints, so that the index is that of the constrained surface),
  the IRC (`irc::constraints`) and the minimization of its ends, the two-point methods of
  `method=connect` and `addf` (`path::manifold::PointManifold`; the renumbering of identical
  atoms of `connect` keeps the atoms of the constraints), the mode kill of saddle points, the
  conformers of transition states (`confsearch -tsconf` with the run's constraints) and the
  remap. Built for an isolated system and a fixed periodic cell; checked by
  `tests/constraints_matrix_mapper.rs` for the AFIR method (the push, the relaxation, the
  search, the IRC).
- **Nodes.** A node is a stationary point of the constrained and biased surface. Its energy is
  that of the biased surface (what its barriers are taken from); the energy of the calculator
  alone at its structure is stored beside it (`bare_energy_hartree` in the network file and
  the database, a column of `nodes.csv`, both in the report). Built.
- **Identity.** With frozen atoms the structures share a frame: two structures are compared
  without superposition and without mirror images, and a frozen atom is matched only with
  itself (each frozen atom has a colour of its own in the bond graph); the other atoms are
  matched as before. Held distances, angles and functions need no rule of their own: they are
  equal in every structure of the run. The bond-change limits (`max_bond_changes`) count the
  bond graph as it is. Built.
- **Refused, by name, with the reason** (built):
  - with constraints or biases: dissociation channels are off by default (the fragments of a
    channel are optimized alone, where constraints and biases over the whole structure have
    no meaning), and `channels=on` is refused; `entrance_structures` (molecules placed at random
    break the held values); `method=nanoreactor` (the walls and the dynamics are a surface of
    their own); with constraints `higher_saddles` and `upward` (the high-index saddle dynamics
    holds none) and `method_relax=dmf` (the direct MaxFlux method holds none); several
    electronic states (`states`, the crossing searches: their plumbing is
    left to the crossing work); an objective of plain variables;
  - constraints with a variable cell (`variable_cell`): the IRC of the atoms and the cell holds
    no constraints (`workflow::irc_cell`, as `optmain -vc -irc` refuses it);
  - constraints on the core of an ONIOM surface (as before; `-fix` of environment atoms stays
    the hold of the relaxations), and biases on an ONIOM surface (the biased calculator does not
    hand on the relaxed environment of the microiterations);
  - biases whose value depends on more than the structure (`PesTerm::history`).

## Several electronic states: transition states and crossing points together

User request (2026-11-18): the transition states of every surface and the crossing points
between the surfaces (MECP, MECI) in one network, with the surfaces shown apart in the report.
Built (`workflow/mapper/crossing.rs`, `driver_states.rs`, `cli/mapper_states.rs`).

- **States.** `states=C,M[,ROOT];C,M[,ROOT];...` lists the electronic states of the run, PES1,
  PES2, ... in that order: the charge, the multiplicity, and the root of the engine (0, the
  ground state of that multiplicity, when it is not given), two or more and all different. The
  network records them (`states` of the network file, table `states` of the database); a
  restart with other states is refused. Without `states` the run has one state, as before.
  Plain variables have one surface and refuse `states`; a step that names an excited state or
  a model function of its own is refused with `states` (the states say which surfaces there
  are).
- **Every node lies on its state.** Minima, transition states, `kSP` and channels found by a
  job lie on the state of the job, which is that of its minimum (`Job::state`); every
  calculation of the job is on that surface (the structure carries the charge and
  multiplicity, the steps of the state the root). An input lies on the first state of its
  charge and multiplicity (an input on none is refused) and is minimized on that state, an
  excited root included (`MapperDriver::input_state_index`, `MapperJobs::minimize_in`). A crossing point lies on two states
  (`Crossing { kind, states }`, in the order of the run).
- **Identity:** with several states two nodes are the same only on the same state (two
  crossing points only of the same pair), besides the bond graph, energy and RMSD.
- **How crossing points are found** (decisions of the user, 2026-11-18). As transition
  states are, by AFIR: every minimum of PESi has, for every AFIR push of `afir_ts` (its pairs
  of atoms and `gamma`) and every other state PESj of the pairs searched, crossing tasks whose
  jobs run an AFIR path from the minimum and take crossing-point candidates from it. Two kinds
  of path, chosen by `crossing_afir`:
  - `gp` (the default with `states`), task `gpafir:J:<push>`: the artificial force on the
    gradient projection of the two states itself, the AFIR function in the place of the
    energies of both states, so that it cancels in the gap and acts within the seam space
    (GP/AFIR; Harabuchi, Taketsugu, Maeda, Chem. Phys. Lett. 2017, 674, 141). The penalty of
    the seam model function is taken out of the path, where it slows the convergence and
    raises the barriers along it, and kept only to choose the candidates: they report 5424
    force calculations against 8582 on the seam model function for the S0/S1 MECIs of benzene
    (37 % fewer) and 5889 against 7531 for naphthalene (22 % fewer), with the same MECIs
    found. The path converges to a
    minimum of the mean AFIR function within the seam: its end lies on the seam (the gap is
    zero) but is displaced within it by the force, so it is no crossing point of minimum
    energy yet. As in that paper (its section 2.3, steps b and c) the candidates are the local
    minima along the path of the seam model function `(E_1 + E_2) / 2 + (E_1 - E_2)^2 / beta`
    of the two states (their eq. 7, `beta = 30 kJ/mol`, `crossing_smf_alpha`), which is low
    only where the gap is small; the function only chooses the candidates and has no part in
    the path. The converged end of the path is taken as a candidate as well (on a path of
    discrete points it is the last minimum; not said in the paper); the last point of a path
    that did not converge is not (a warning). At most `MODEL_AFIR_CANDIDATES` (3) per path,
    the end first, then the lowest minima (the paper takes all; the limit bounds the cost of a
    path with many shallow minima; see "Which extrema are candidates" below). Measured on the
    model of three atoms: the ends of six
    GP/AFIR paths had gaps below 1e-6 Hartree and lay 1.4e-3 to 2.9e-3 Hartree above the
    crossing point of minimum energy that the refinement without the force reached in 5 to 7
    iterations.
  - `model`, task `xafir:J:<push>`: the path on a model function of the two states whose
    minima lie near their crossing points, `-mf seam` for an MECP (`(E_1 + E_2) / 2 +
    (E_1 - E_2)^2 / alpha`; Sato et al., J. Phys. Chem. A 2009, 113, 1704; with AFIR: Maeda,
    Saito, Morokuma, J. Phys. Chem. Lett. 2011, 2, 852) and `-mf avoiding` for an MECI (the
    Avoiding model function of Maeda, Ohno, Morokuma, Adv. Phys. Chem. 2012, 268124;
    `crossing_amf_branch=lower` for `-mf avoiding_lower`). One path gives several candidates,
    so that the crossing points on the way of a reaction of several steps are found too: on
    the seam model function and the upper Avoiding model function, where the seam is a valley,
    the end of the path and the lowest local minima of the model function along it
    (`MODEL_AFIR_CANDIDATES`, 3 in all); on the lower Avoiding model function, where it is a
    ridge that a path crosses at a maximum, the lowest local maxima and then the end. The
    other extrema of these paths are artefacts of the penalty or the coupling of the model
    function, not transition states of either state, and are not followed. `alpha` is 30
    kJ/mol for both model functions (`crossing_smf_alpha`, `crossing_amf_alpha`; `-mf seam`
    itself keeps MultiOptPy's 0.05 Hartree when it is called without a value). The model functions are those of
    `-mf` (`pes::combine`): the mapper has none of its own.
  - What the two kinds share and where they differ:
    | | `gp` | `model` |
    |---|---|---|
    | the path | the projected gradient with the force | the model function with the force |
    | the candidates | the end and the local minima of the seam model function along the path | the end and the local minima (maxima on the lower Avoiding model function) of the model function along the path |
    | the refinement | the gradient projection without the force | the same |
    The gradient projection needs the projector of the branching plane (for an MECI its
    updating); a model function is a scalar function that any minimization follows.
  - `both` runs the two kinds, `off` none. `crossing_direct=on` (off by default) adds the
    task `crossing:J`, the candidate being the minimum itself: on real molecules that alone
    does not reach the crossing points.
  The steps of the paths and of the refinement are at most 0.1 Angstrom (`crossing_step`):
  the projected gradient and the model functions change fast across the seam.
- **Refinement.** Every candidate, of either kind of path, is optimized to the minimum-energy
  crossing point without the force: the gradient projection of `opt_mesx` for states of
  different spin or charge (MECP), of `opt_meci` with the branching-plane updating for two
  roots of one spin (MECI); the settings of `step3_settings` at order 0, except the method
  and the exact Hessians: where `step3_settings` name neither `opt_method` nor
  `calc_exact_hess` the search takes `rsirfo_block_fsb` and no periodic exact Hessians (where
  they name either, the settings of the step as they are; a crossing point is a
  minimum of the projected surface; measured in `tests/crossing_searches.rs`, the same
  defaults as `run_automecp`, defined once in `cli::mapper_settings::CROSSING_METHOD`;
  2026-11-24, before the search took the Bofill update and two exact Hessians every five
  iterations of the transition-state step). The force displaces
  the end of a GP path within the seam, and a candidate of a model function still has a gap,
  so this step is always run. The end is a crossing point when the optimization converged and
  the gap is at most `crossing_gap` (1e-3 Hartree by default), for an MECP and an MECI
  alike, both being points where the two states are degenerate. The gap is a condition of
  its own: the convergence of the projected gradient bounds it only by `sqrt(3N)` times the
  force threshold over two (above 1e-3 Hartree from about 34 atoms with the default RMS
  threshold). Both states are computed at the end of the search for this, before
  anything runs down from it: with a larger gap the job gets a warning that says how far
  apart the two states are, the end is no node, and no meta-IRC runs from it (user request,
  2026-11-19). A conical intersection whose check of the branching plane still finds it not
  stationary on the seam after its last correction (`seam_check_failed`) is no crossing point
  either, with a warning of its own. A candidate whose two states have the same gradient
  (the gradient projection refuses them, `pes::combine::projection::PARALLEL_LIMIT`) is left
  out with a warning, and the job goes on with its other candidates. It is matched with the
  known crossing points of the pair or added
  (`MECP0`, `MECI0`, ...), its energy the mean of the two states, with the norm of the
  difference of the two gradients and the gap (the energy of the first state of the pair in
  the order of the states of the run minus that of the second; on the crossing edge, in the
  network files, the tables and the report).
- **The check on the seam (on by default; user decision, 2026-11-24).** Every crossing point
  that is accepted gets the vibrational analysis within the seam (`vib::seam_analysis`, as
  `freqmain` does it at a crossing: the Hessians of both states, for an MECI the second
  direction of the branching plane from the gradients at displaced structures): the number of
  imaginary modes on the seam by the one criterion `opt::mode_kill::IMAGINARY_CURVATURE`,
  kept on the crossing edge (`seam_imaginary_modes`; 0: a minimum on the seam). A point with
  one or more is kept in the network but is a saddle point on the seam, not a minimum-energy
  crossing point: so named in the tables and the report, with a warning. Nothing descends
  along the seam from it. `crossing_seam_check=off` switches the check off (`not checked`,
  as where it could not be done, with the reason: a variable cell, or an error of the
  analysis). It costs the Hessians of both states and, for an MECI, four evaluations of both
  states per accepted point; they are counted with the other evaluations.
- **Counting crossing points.** A crossing point is a node; the console, the page and the
  summaries count those nodes. A node can have several crossing edges (the same point reached
  with other minima at the ends), and the tables have a line per edge with the node named.
- **From a crossing point to the minima: meta-IRC.** On each of the two states the path of
  steepest descent in mass-weighted coordinates runs from the crossing point, which is no
  stationary point of either state (the IRC machine of `ircmain` with the IRC settings of the
  run, `cli::irc::run_structure_observed`; one side). Its end is minimized (the settings of
  the minimizations, verified by the Hessian as every minimum) and matched as IRC ends are on
  their state. The crossing edge joins the two minima, with the barriers from both, the
  gradient difference, and the two paths (`irc`: a side per state, named `PES1`, ...). The
  structures of the whole way between the two minima are in one file,
  `paths/MECP000000_edge0_path.xyz`: from the minimum of the first state (the minimization
  of the end of its meta-IRC backwards, then the meta-IRC backwards) up to the crossing
  point, then down the meta-IRC of the second state and the minimization of its end to the
  minimum of the second state; every frame is labelled with its part (`PES1 minimization`,
  `PES1`, `MECP`, `PES2`, `PES2 minimization`) and has the energy of the state it lies on.
  The report plays it in the panel of the crossing point. A known edge with the same ends is
  not added again. A
  minimum found so is explored like any, so that the network grows on every surface and
  between them.
- **Transition states of every surface.** By default the method runs from the minima of
  every state, on that state. `ts_states=1,3` lists the states (PES numbers) on which
  transition states are searched; the minima of the other states have their crossing tasks
  alone.
- **Order.** Which minimum is explored next is decided by the priority of the run, any of the
  registry: the kinetics of the network with its crossing edges (`priority=kinetics`), the
  Boltzmann populations, the goal (`priority=rrt`), or a registered one. Within a minimum the
  tasks alternate: a crossing task, a task of the method (transition states), a crossing
  task, and so on, so that neither kind waits for the other to run out.
- `crossings=off` explores each surface on its own (inputs of several states); on by default
  with `states`. `crossing_pairs=adjacent` searches only neighbouring states (PESk with PESk-1
  and PESk+1: n - 1 pairs of n states instead of n (n - 1) / 2); `all` is the default.
- **One reaction (`run_automecp`; user request, 2026-11-18).** The counterpart of
  `run_autots` for crossing points is a run of the mapper, not a second implementation: the
  command translates its settings (the pushes of `-ma`, the second state, the steps) into
  mapper settings and adds a summary file. For it the mapper takes AFIR pushes given by their
  fragments (`crossing_pushes=GAMMA:ATOMS:ATOMS;...` instead of the pairs of atoms of
  `afir_ts`), crossing tasks from the inputs alone (`crossing_from=inputs`), and no search for
  transition states (`ts_states=none`). The input is minimized on its own state first, as
  every input of the mapper (`run_autots` takes its input as it is). Checked on the model of
  three atoms: a push and a pull give the crossing points at minima of the seam with both
  meta-IRC paths, MECP and MECI, and `--restart` continues to the same network. Its page
  (`automecp_report.html`; user request, 2026-11-19) is that of `run_autots` with the steps
  of this search: the style, the viewer, the charts and the energy diagram are the same code
  (`cli::autots_report`), and the page is written from the files of the mapper run (the
  network, the trajectories and logs of the jobs, the path files), not from a second record
  of the search. For it the trajectory of a job has the energies of both states in every
  frame of a search on a surface of two states (`states=E1,E2` in the comment).
- **Which extrema are candidates (built; user request, 2026-11-19).** Where the surface is
  not smooth (numerical noise of the energies, a path that zigzags) the function along an
  AFIR path has many small extrema next to each other, and each would be refined to the same
  crossing point. Three rules (`workflow::mapper::CandidateFilter`), for both kinds of path:
  - *prominence*: an extremum is a candidate only when the function rises (falls, at a
    maximum of the lower Avoiding model function) by `crossing_candidate_prominence` on both
    of its sides before it comes back to its value (the smaller of the two sides; 0.5 kJ/mol);
  - *distance*: two candidates of one path are at least `crossing_candidate_distance` apart
    (the root-mean-square displacement of the atoms, no superposition: the points of one
    optimization are in one frame; 0.05 Angstrom); the end of the path and the better
    extremum stay;
  - *known crossing points*: a candidate within that distance of a crossing point that the
    same job has found is not refined.
  What is left out is said in the log of the job. Measured on the model of three atoms (the
  runs of `tests/cli_mapper_states.rs`, six AFIR paths each; refinements without and with the
  rules): 15 and 6 on the gradient projection for the MECP, 15 and 8 on the seam model
  function, 18 and 11 on the gradient projection for the MECI (one of its paths has more
  than 50 extrema that are no candidates), 18 and 12 on the upper Avoiding model function, 12 and 11
  on the lower; the crossing points found are the same. The two defaults are set from this
  model and are to be measured on molecules; `0` switches a rule off.
- **A search between parallel states is ended (built, off by default;
  `crossing_parallel_stop=on`; user decision, 2026-11-19).** Where two states are degenerate
  over a region (fragments apart) the gap is closed and the difference of the two gradients
  is small: the direction of the difference gradient is noise, and a search for a point of
  the seam cannot converge. With the rule a crossing search ends when the gap is at most
  `crossing_gap` and the difference of the two gradients below `crossing_parallel_gradient`
  (0.01 Hartree/Bohr) for `crossing_parallel_steps` (10) iterations in a row; a warning of the
  job says so with the gap and the difference at the end, the end is no node, and nothing
  runs down from it. The threshold: with the largest gap of a crossing point (1e-3 Hartree),
  a difference of 0.01 Hartree/Bohr leaves the gap closed over 0.1 Bohr, half a largest step
  of the search, so that the search cannot tell the seam from what lies beside it; at the
  crossing points of the model the difference is 0.11 Hartree/Bohr. The optimizer reports the
  gap and the difference of the gradients of a gradient projection at every iteration for this
  (`IterationReport::crossing`). It is off until the threshold has been measured on molecules
  whose states do run parallel.
- **What a step on the gradient projection is judged by (`projection_trust` in the steps,
  `-projection_trust merit`; built, the default is `energy`).** The trust radius of the
  optimizer compares the change of the energy with the prediction of its quadratic model.
  On a gradient projection the gradient is not that of the mean energy reported, and the
  ratio is no measure of the model (in a search of the model: -98, 284, -48 in consecutive
  steps, the radius shrinking at every step). `merit` compares the change of the function
  the projected gradient belongs to (`opt::state_machine::SeamPoint`). Measured on the model
  (tight criteria): searches from a minimum 65 against 72 iterations (MECP) and 99 against 124
  (MECI); refinements of the candidates of AFIR paths unchanged for the MECP; for the MECI
  one more converged on the upper Avoiding model function (5 of 12 against 4) and one fewer
  on the lower (10 of 11 against 11, 883 against 507 iterations). Not of one sign, so the
  default stays `energy`; molecules are to decide.
- **The pages can be moved (user request, 2026-11-19).** `mapper_report.html`,
  `automecp_report.html` and `autots_report.html` hold their structures and paths and have
  no links to the files of a run directory: what a page holds is saved from it, and what it
  does not hold is not offered.
- **Any program for the electronic structure (user request, 2026-11-19).** The crossing
  searches use the calculator of the run through the registry of calculators like every
  command, so a program is added without a change of mopt-rs: as a Python class
  (`engine:module:Class`) that hands over `compute` for a charge, a multiplicity and a root,
  and `state_characters` where the roots are to be followed, or as an ASE calculator with
  the two functions of `-calc_opt`. Tested with a model given as such a class
  (`tests/test_crossing_plugin.py`).
- **Following a state from structure to structure and from node to node (built).** A state
  named by its root (`states=0,1,1;0,1,2`) is by default the state with that place in the
  order of the energies at every structure, so that it changes its character where two roots
  change places. With `follow_states` in `step1_settings` (`-es_follow`) and a calculator that
  hands out its states (`pes::followed`: the built-in GFN0-xTB, PySCF with a restricted
  reference), the number means the state that had it at the first input, followed by the
  overlaps of its amplitudes:
  - along every optimization and IRC of a job, as in `optmain` (every job has a calculator
    of its own then, which evaluates in order);
  - from node to node: at the end of every optimization of a job the calculator is asked
    for the place, in the order of the energies there, of every excited state of the run
    that it has followed to that structure (`MapperJobs::take_followed`); a node keeps those
    of its structure (`Node::roots`, `followed_states` of `reaction_network.json`), and the
    calculator of a job starts from those of its node (`Calculator::recall_state`), so that
    PES2 at a node is the state that PES2 was at the node it was found from, whatever its
    place among the roots there.
  - A node that the calculator did not follow to (the end of a search that is not an
    optimization of the jobs, a node of a file of an earlier version) has none: a job from
    it counts the states by their energies there and follows them from there.
  - The identity of a node is its kind, its state, its structure and its energy, as before.
    The assignment by overlaps is made along a path, and two paths around a conical
    intersection can give one structure two names; such a structure is then two nodes with
    one energy. The report shows the followed states of the node chosen.
  - Checked on the model of three atoms with its two surfaces handed out in the order of
    their energies and with their characters (`tests/cli_mapper_states.rs`): followed, the
    network is that of the two surfaces named (the minima at R1 and R2 and their transition
    state on PES1, the minimum at RC on PES2, two conical intersections), and the nodes have
    the places of the states (at R1 the first state is the lowest excited state, at RC the
    second is); counted by their energies, the minimum at RC lies on PES1.
- **Not built yet:**
  - *The two kinds of path compared on a molecule* (how many crossing points per gradient),
    before the default is called measured here. On the model of three atoms (the runs of
    `tests/cli_mapper_states.rs`, six AFIR paths each, the same pushes and settings) the two
    kinds do not differ for an MECP: both find the two MECP, with 98 iterations of the paths
    and 84 of the refinements on the gradient projection (1233 gradients in the whole run)
    against 79 and 80 on the seam model function (1180 gradients). For an MECI the paths on
    the gradient projection take 779 iterations against 107 on the Avoiding model function,
    and with the tight criteria of these tests about half of the refinements stall with either
    (the point below), so that these runs say nothing about the cost of an MECI search; the
    lower Avoiding model function has 43 iterations of the paths and all 12 refinements
    converged (534 iterations). A model of three atoms whose roots are not coupled cannot
    stand for the saving reported for benzene and naphthalene.
  - *Saddle points within the seam* (a first-order saddle point of the energy restricted to
    the seam, between two crossing points along it): not searched and not planned for now
    (user decision, 2026-11-18). What they would be used for: a
    stationary point of the seam found by a crossing-point optimization can be a saddle point
    of the seam and not a minimum (second-order analysis of the seam: Paterson, Bearpark, Robb,
    Blancafort, J. Chem. Phys. 2004, 121, 11562); the path along the seam through such a saddle
    point shows whether two crossing points belong to one connected seam and how high the seam
    rises between them (Sicilia, Blancafort, Bearpark, Robb, J. Chem. Theory Comput. 2008, 4,
    257), which decides which part of the seam a molecule can reach at a given energy.
- **Kinetics:** the crossing edges are bottlenecks as written below (a hopping probability of
  1 by default, Landau-Zener or Zhu-Nakamura on request).
- **Report** (user request, 2026-11-18): crossing points are triangles with an outline of their
  own, apart from the transition states (diamonds), and have their own table; every node names
  its surface (`EQ3 · PES2`, `MECP0 · PES1/PES2`) in the tables, the tooltips and the details.
  A bar above the figures shows one surface at a time (PES1, PES2, ...: its nodes and the
  crossing points that leave it, each joined to the minimum of that surface), all of them,
  or all in layers (each surface in a band of its own, PES1 at the bottom, the crossing points
  between the bands of their states).
- **Checked** on a model of three atoms whose surfaces depend on the three distances, with
  the seam linear in one of them so that its minima are known
  (`tests/cli_mapper_states.rs`). With two spin states the network has the two minima of PES1
  and their TS, the minimum of PES2, and the two MECP at the minima of the seam (energies within
  1e-6 Hartree), each joined to a minimum of PES1 and that of PES2 with the meta-IRC of both
  states (each going down on its own state, its structures in the file of the edge); the same
  with two roots of one spin (MECI); with three states a crossing point of every pair (MECP,
  MECI, MECP), each at a minimum of the seam of its pair, and the kinetics of the network;
  `crossing_pairs=adjacent` leaves out the PES1/PES3 pair; the AFIR paths on the gradient
  projection (the default) and on both model functions (and the lower Avoiding model function)
  reach crossing points at the minima of the seam, MECP and MECI, with the tasks of a minimum
  alternating; `ts_states` leaves the transition states of the other states unsearched; runs
  stopped at four points continue to the same network.
- **The refinement of MECI candidates on the model: a stall under the tight criteria.** In
  the runs of these tests (tight criteria in `step3_settings`: largest force 1.5e-5
  Hartree/Bohr) every MECP candidate converges to the crossing point (6 of 6 from the paths
  on the gradient projection, 37 iterations in all, and 8 of 8 from those on the seam model
  function, 45 iterations; 15 of 15 each before the rules for the candidates), but of the MECI candidates
  only 6 of 11 (gradient projection) and 4 of 12 (upper Avoiding model function) converge
  within 300 iterations, and 11 of 11 on the lower one (9 of 18, 7 of 18 and 12 of 12
  before the rules). The others are reported as warnings, none becomes a node, and the
  crossing points that are found are the right ones. With the default criteria (largest
  force 4.5e-4) on the same runs all converge (11 of 11, 10 of 10, 11 of 11); their ends lie
  within 2e-5 Hartree of the minima of the seam, one of them 7.5e-4 Hartree above. What
  happens under the tight criteria, from the iterations of a search that did not converge:
  the gap closes to 1e-17 Hartree, the gradient difference stays at 0.107 Hartree/Bohr (the
  two states do cross there), the largest force stays at 1.83e-5, just above the threshold,
  and the step falls by a factor of about 100 per iteration to 1e-15 Bohr while the trust
  radius is still 0.05 Bohr; the search then repeats the same structure up to the limit. So
  the search is not wrong about the crossing point; a force of 2e-5 is left that its steps
  do not remove. Why they do not is not established (the two roots of the model are not
  coupled, so that its seam has one dimension more than a seam of conical intersections and
  the second vector of the branching plane has no direction to converge to; the gradient
  projection and its Hessian are `pes::combine::projection`). Judging the steps by the merit
  function (above) does not remove the stall.

## A network taken to another surface: `-remap` and `-single_points`

User requests, 2026-11-19: the paths of an existing network drawn again on another surface
(`-remap`; what these notes called the level-2 refinement), and the single points of a whole
network at once; both for transition states, crossing points (MECP, MECI) and saddle points
of higher order.

Rules of both.

- Both read a run directory (its `reaction_network.json`, or its database) and write a run
  directory of their own. The first run is only read. The second network is a network like
  any other: its files, its report, its kinetics, `--restart`.
- The surface of the second run is that of its settings (`-cfg`, `-map_opt`). A network of
  several states needs a run of as many states, in the same order (`states`).
- A table in the second run directory says what every minimum, edge and saddle point of the
  first network became (`remap.csv`, `single_points.csv`).

`-remap RUN_DIR`: the paths drawn again (built).

- The inputs of the run are the minima of the first network, each minimized on the second
  surface on its own state (its place among the states of the first network, not the first
  state of its charge and multiplicity). Two minima that become the same are one node.
- Then one job for every transition state, crossing point and saddle point of higher order of
  the first network, each once (the method `remap`; the job runs from the node of the
  minimum at one of its ends):
  - a transition state: the first chain is the minimum of its first end on the second
    surface, the structures of its path file (minimum to minimum on the first surface, at most
    60 of them), and the minimum of its second end. `method_path` (`neb`; `method_path_KEY`
    its settings) relaxes the chain on the second surface, and every maximum is followed as
    in any job: transition-state optimization, IRC, the ends minimized. `method_path=none`:
    no chain, the transition state of the first network is the guess, which it also is when
    the chain gives no maximum. An end that is no minimum (a dissociation channel, another
    saddle point) adds no structure to the chain.
  - a crossing point: refined on the gradient projection of its two states on the second
    surface from its structure, then the meta-IRC on each state and the minimization of its
    end, as in any crossing job. The job runs on one of the two states.
  - a saddle point of order k >= 2: optimized at order k from its structure and descended
    from, as in any job (`higher_saddles=on`; without it the item is left with a warning).
- Nothing else is explored: the run ends when these jobs are done (it has no limit of jobs
  unless one is given). What a job finds beyond its item (another transition state on the
  chain, another minimum at the end of an IRC) is kept, as in any job.
- The crossing searches of a run of several states (the AFIR paths towards crossing points,
  the searches from the minima) are not run: the crossing points of the first network are
  the items.
- What does not come through is left out, and the run goes on (user request, 2026-11-19: on
  real molecules some of every kind fail). A minimum that does not converge, or whose
  minimization fails, gives no node and a warning (`DriverSettings::skip_failed_inputs`; in
  any other run an input that fails ends the run); its items run from another minimum that
  came through. A transition state, crossing point or saddle point of higher order whose job
  gives none of its kind, or fails, gives no node and a warning of its job
  (`JobOutcome::known` tells a job that converged to a known node from one that found
  nothing).
- `remap.csv`: every minimum and item of the first network with its job, what became of it
  (`minimized`, `added`, `known`, `left out`, `failed`, `not run`), the nodes of its kind the
  job added, and the node of its kind in the second network nearest to it with the distance.

`-single_points RUN_DIR`: the energies of a whole network at its structures (built).

- Every node keeps its structure, its number, its label and its frequencies; its energy is
  that of the second surface at that structure: a minimum, a transition state and a saddle
  point of higher order one energy on its state; a crossing point the energies of its two
  states (the node has their mean, `single_points.csv` their gap: a crossing point of the
  first surface is in general none of the second, and the gap says how far it is from one); a
  dissociation channel every fragment record on its own, the channel their sum.
- No optimization and no Hessian. A free energy is shifted by the change of the energy (its
  thermal part is that of the first surface). The barriers of the edges are computed again
  from the energies of the nodes; the energies along the IRCs are those of the first surface
  and are left out.
- One evaluation per node (two for a crossing point), `--max_workers` at a time, every one
  in the journal: a stopped run writes no network and goes on with `--restart`.
- The network of the run is the first one with these energies and names its source
  (`run.source`, kind `single_points`); the title of the report says so. The mixed energies
  are thus a network of their own (`2//1`), never mixed into the first.
- The jobs of the first run stay in the network (they say how each node was found); the
  evaluations counted are those of the single points alone.

## Default on, or asked for

| Detection | Default | Why |
|---|---|---|
| TS–TS link at an IRC end | on (built) | the IRC end is examined anyway; it costs nothing more |
| possible bifurcation from the exact Hessians of the IRC | on (built; `irc_hessians`, `bifurcation_threshold_cm1`) | the lowest curvature across the path where the IRC computed an exact Hessian anyway: negative means a valley-ridge inflection may lie before that point. Recorded on the edge as a possibility (side, arc lengths, curvature) with a warning, never as confirmed ends. The mapper's IRC asks for an exact Hessian every 10 points by default (about a tenth more cost for GFN1). Every 10 points misses a ridge shorter than that |
| confirming a possible bifurcation (± along that mode, minimized) | off (built; `confirm_bifurcations=on`) | an extra Hessian and two minimizations per possible bifurcation |
| the curvature across the path at the exact Hessians of the IRC, and where it changes sign | on (built) | no cost of its own: the lowest curvature across the path at every exact Hessian the IRC computes anyway (every `irc_hessian_interval`-th point, 10 by default) is kept on the side (`transverse_cm1` of the path in the network file), and a possible bifurcation gets the arc length at which that curvature changes sign (`inflection`, linear between the two Hessians around it): an estimate of where the valley turns into a ridge. An exact Hessian at every point is not a mode of its own (user decision, 2026-11-19: it costs too much; the possible bifurcation is what one Hessian in every few points shows); `irc_hessian_interval` sets how many |
| relaxed scan to confirm a barrierless dissociation | off (built; `confirm_dissociations=on`) | for every barrierless edge of a complex of two molecules: a relaxed scan with the parts of `relaxedscan` (user, 2026-11-19). The distance between the centres of the two molecules (the constraint `fbond`) is held at values `dissociation_scan_step` (0.3 Angstrom) apart from the complex outwards, every point minimized under the constraint from the one before, until the molecules are a channel by the test of the channels (at most 40 points); the energies are kept on the edge (`scan`). The maxima of the curve are found as for the curve of a relaxed scan; one more than `dissociation_scan_tolerance` (0.5 kcal/mol) above the channel is a barrier: the edge is marked `barrier` with a warning, and the kinetics does not treat it as a capture |

## Kinetics of each kind

- **TS edge:** transition-state theory from the energies or free energies of the TS and its
  ends, with a correction chosen afterwards (`wigner`, `bell`, `eckart`, `skodje_truhlar`, a
  given `kappa`).
- **Crossing edge:** by default the crossing point as the bottleneck with a hopping
  probability of 1. Landau–Zener with a coupling is built, with the double passage of a
  peaked crossing (`1 - P^2`) or of a sloped one (`2 P (1 - P)`), told apart by the product of
  the slopes of the two states. Also built (user request, 2026-11-17):
  - Zhu–Nakamura (`zhu_nakamura`) for sloped crossings above the crossing energy, and for
    peaked crossings (the nonadiabatic-tunnelling type) at every energy from the higher
    minimum up, tunnelling below the crossing included (user request, 2026-11-17). It needs
    the coupling like Landau–Zener: for two states of the same spin it comes from the
    adiabatic gap (`freqmain -diabatize` writes the model from the adiabatic energies alone,
    sloped or peaked), for two spin states it is the spin–orbit coupling. The energies below
    a sloped crossing are planned (the formulas are in the same source);
  - couplings and slopes read from `diabatic_crossings.csv` of `freqmain -diabatize`
    (`diabatic_crossings` in the annotations) instead of typed in.
  Planned: spin–orbit couplings computed afterwards at a crossing point; the crossing edges of
  the mapper storing the product of the slopes.
- **DC:** a state whose free energy is the sum of its fragments, every fragment beyond the
  first one more free molecule at the standard concentration `c°` (1 M). Built. For a
  barrierless edge, association comes from a capture or diffusion model, and dissociation from
  detailed balance with ΔG° at `c°`; never from Eyring with ΔE alone. Built: the capture of
  hard spheres in the gas (the default), Collins–Kimball (diffusion with Stokes–Einstein and
  capture in series) with `run_kinetics --viscosity`; the radius of a fragment is that of the
  sphere of its van der Waals volume. A DC as an entrance channel (its population at the
  start) is planned.
- **N-furcation:**
  - The rate out of the minimum before TS1 comes from TS1.
  - One effective edge goes to each product reachable through TS–TS links. The search stops
    at every EQ, so later steps are not lumped in.
  - Built: a TS edge with a transition state at one end becomes one step from its other end
    to every state beyond (`Network::beyond_link`), each with a share of the flux of TS1
    (`Bottleneck::share`, a free energy `-kT ln share` on the bottleneck, so both directions
    are scaled and detailed balance holds), and the notes of the kinetics say the split is
    assumed. An edge with transition states at both ends is left out (planned).
  - The split among the products is a setting:
    - `equal` (the default, marked "assumed" in the report and the database; built);
    - `degeneracy`, symmetry-equivalent products, the only static weighting with a basis;
    - `geometric`, a static estimate: in the branching space of TS1 (its mode and the modes of
      the downstream TS2) in mass-weighted coordinates, the share of the thermal spread of
      TS1's transverse modes that falls in each product's cell, the cells bounded by the TS2
      ridges. Calibrated against trajectories before it can become the default;
    - `trajectories`, the ratio counted from short trajectories started at TS1 (Wigner
      sampling).
  - The sensitivity tools of the kinetics show how much the split matters.
- **Path peak:** not used (path peaks are records beside the network; user decision,
  2026-11-19).
- **TS edge with an unknown end:** not used (user decision, 2026-11-20): the kinetics has no
  state to send the flux to, and its notes name every such transition state. A TS–TS link
  that ends at such a transition state leads to nothing beyond it and is left out as well.
- **Higher-order saddle** (experimental): not a bottleneck of its own. It joins the
  N-furcation of the TS whose IRC reached it, and its descending directions are ends of that
  bottleneck (built: `Network::beyond_link` follows the descent edges and the transition
  states below them, and the flux is split as for TS–TS links). An edge whose IRC ends at
  saddle points on both sides is left out, as for TS–TS links on both sides.
- **Confirmed bifurcation** (`confirm_bifurcations`): the flux of the side divides equally
  among its end and the minima found across the path (`confirmed_ends`), marked assumed in
  the notes of the kinetics (built).
- **Kinetic parameters from the mapper.** The network supplies the frequencies of every EQ
  (from its Hessian check) and every TS (from its IRC start check), so `tst` needs no
  annotations (built). The `kinetics` key runs the kinetics at the end of a mapper run
  (user request; built, `cli/mapper_kinetics.rs`; on LJ7, `tests/cli_mapper_kinetics.rs`, the
  populations at 1 s are the Boltzmann weights of the joined minima within 1e-6, and runs
  stopped at three points and restarted give the same populations within 1e-9):
  - `kinetics=on` (off by default): when the exploration ends (its stop rules, its job limit,
    or no task left; not when it is interrupted), the kinetics of the level-1 network runs
    with the `kinetics_*` settings, the same keys and defaults as `priority=kinetics` (the
    temperature of the run, `kinetics_time` 1 s, the start `kinetics_start` or the node of the
    first input, Eyring rates of the electronic energies, the solver `propagate`). The keys
    `kinetics_*` are refused when neither `priority=kinetics` nor `kinetics=on` uses them.
  - The output times are the 13 decades from `10^-12 kinetics_time` to `kinetics_time`, so
    the report can draw the populations against time; the last is the time of interest.
  - Its files are those of `run_kinetics` in `RUN_DIR/kinetics/` (`populations.csv`,
    `rates.csv`, the superstates, `priority.csv` with the ranking by `kinetics_score`), and
    `kinetics/kinetics_summary.json` with the settings, the populations of every state (node id
    and label) at every output time, and the notes of the solver. A network without a transition state has no kinetics: the
    summary says so, and the run goes on to its end as before.
  - A restart of a run that has ended computes it again (it is cheap next to the run); an
    interrupted run computes none. The summary records the number of jobs of its network, and
    a report of a network with more jobs (a restart that goes on) says that the kinetics shown
    is that of the earlier network.
  - The report shows a section "Kinetics" from `kinetics_summary.json` (also when it is
    written again by `--report`): the settings, the populations against time of the states
    that reach at least 1 % (at most eight, on a logarithmic time axis), and the table of the
    states at the time of interest.

## Files and report

- `reaction_network.json` and `reaction_network.sqlite` store every kind and edge with its
  raw quantities and its warnings; a new kind adds columns or rows without changing the
  meaning of existing ones. Built for EQ, TS, crossing, DC (the `parts` of a DC node, the
  tables `fragments` and `dissociation_edges`; `dissociation_edges.csv`), and the TS edges
  with an unknown end (`open_ts_edges`, in the file only where there are any; the table
  `open_ts_edges`). `-merge` carries them; `-remap` does not take them (a path needs both
  minima) and says how many it left.
- The report draws every node and edge, whether connected or not: EQ and TS, numbered
  apart. A TS with an unknown end is drawn without a link, beside the minimum it is known
  to reach. Path peaks are not drawn (they are records beside the network; the page says how
  many the run kept and names `path_peaks.csv`); TS–TS links and N-furcation fans are drawn
  as such, and DCs have their own marker. Built for EQ and TS, a table of the DCs, and N-furcations as fans (a
  link from the start of TS1 to every minimum beyond its TS–TS link); the rest planned.
- The possible bifurcations of an IRC (`irc.possible_bifurcations` of a TS edge: side, arc
  lengths, wavenumber) are in both files and in the warnings of the network. Built.
- Every report a run writes is shown to the user who asked for it.

## Names

- **Option names.** Keys are snake_case words of mopt-rs and are not taken from other
  programs. Where another program's feature is followed, the documentation cites the papers,
  not the program.
- **Atoms.** Atoms are numbered from 1 on the command line and from 0 in the code.
