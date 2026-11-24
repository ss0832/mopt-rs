# Quick start: the examples of MultiOptPy

mopt-rs keeps the commands of MultiOptPy. This page takes the examples of MultiOptPy's
README (1.21.2) one by one, gives the form in which each runs with mopt-rs, and says what
the run did. Every command on this page was run as it is written; where a run was not
taken to its end, that is said.

## Install

```bash
pip install .
```

builds the Rust core with maturin (a Rust toolchain is needed) and installs the Python
package. The distribution is called `mopt-py` and is imported as `mopt_rs`
(`import mopt_rs`); it comes with the commands `optmain`, `nebmain`, `ieipmain`, `mdmain`,
`confsearch`, `relaxedscan`, `orientsearch`, `freqmain`, `ircmain`, `etmain`, `run_autots`,
`run_automecp`, `run_mapper`, `run_kinetics`, and `run_rcmc`, `mopt-py COMMAND ...` for any
of them, and `mopt-mcp` (the MCP server).

Engines:

- Built into mopt-rs, nothing else needed: `-os gfn1xtb` (GFN1-xTB), `-os gfn0xtb`
  (GFN0-xTB), `-os gfnff` (GFN-FF), for the elements H to Rn; the model
  potentials `lj`, `emt`, `eam`, `meam`, `tersoff`.
- Optional, installed only as needed (none comes with mopt-rs, and each is imported only when
  a calculation uses it): `-xtb GFN2-xTB` (and `GFN1-xTB`) uses the package tblite
  (`pip install ".[tblite]"`), `-pyscf` PySCF (`".[pyscf]"`), `-dxtb` dxtb (`".[dxtb]"`,
  with PyTorch), and `-os ase:module:Class` any ASE calculator (`".[ase]"`). `".[qm]"`
  installs PySCF, tblite, and ASE together. An engine whose package is missing says so when
  the run starts, with the command that installs it; `python -m mopt_rs.calculators` lists
  every engine with the version installed or the command to install it.

Without Python, `cargo build --release -p mopt-rs` makes `target/release/mopt`, which runs
the same commands with the built-in engines (`mopt optmain ...`).

## The inputs

`examples/quickstart/` holds the files of `test/` of MultiOptPy 1.21.2 that its README
uses:

| File | What it is |
|---|---|
| `SN2.xyz` | fluoromethane and hydroxide (charge -1, in its second line) |
| `aldol_rxn.xyz` | acetaldehyde and the enol of acetaldehyde |
| `aldol_rxn_PT.xyz` | a guess of a transition state of the aldol system |
| `aldol_rxn/` | 17 structures along a path of the aldol reaction (`_0.xyz` ... `_80.xyz`) |
| `ieip_test/` | HCN and HNC (`hcn_A.xyz`, `hcn_B.xyz`) |
| `s8_for_confomation_search_test.xyz` | an S8 ring |
| `config_autots_run_xtb_test.json` | MultiOptPy's settings of the AutoTS workflow (GFN2-xTB) |
| `config_autots_builtin.json` | the same settings with the built-in GFN1-xTB (made for this page) |

```bash
cd examples/quickstart
```

## What changes against MultiOptPy's command lines

1. **Commands, not scripts.** `optmain ...` for `python optmain.py ...`, and so on.
2. **`-spin` is the multiplicity `2S + 1` for every engine.** MultiOptPy takes the number
   of unpaired electrons there when PySCF is used, and its README writes `-pyscf ... -spin
   0`. Here that is `-spin 1`, or `-unpaired 0` (the number of unpaired electrons under
   its own name), or nothing (a singlet is the default). `-spin 0` stops with a message
   that says this.
3. **The state in the structure file.** A line of two whole numbers in a structure file is
   `charge multiplicity`, as in MultiOptPy (`-1 1` in `SN2.xyz`). It gives the state where
   the command line gives none; `-elec`, `-spin`, and `-unpaired` go over it value by
   value (in MultiOptPy the file wins). `optmain` says in its log what it took from the
   file. The runs that a command starts on files it writes itself (the points of a scan,
   the samples of a search, the steps of the workflow) get the same state.
4. **Files without the number of atoms.** The files of `aldol_rxn/` start with `0 1` and
   have no first line with the number of atoms. They are read as MultiOptPy reads them:
   every line `element x y z` is an atom, anything else separates structures.
5. **The engines** `-xtb` and `-pyscf` need their packages, as before. `-os gfn1xtb` in
   their place runs the same command on the built-in engine; GFN1-xTB and GFN2-xTB are
   different methods, so the numbers differ.
6. **Output.** The run directories have MultiOptPy's names (`<date>/<name>_OPT_.../`). The
   structures of the iterations are in `<name>_traj.xyz` (MultiOptPy: a directory per
   iteration); see `migration.md` for the rest.

## The examples

### A minimum with an artificial force (AFIR)

MultiOptPy: `optmain SN2.xyz -ma 150 1 6 -pyscf -elec -1 -spin 0 -opt rsirfo_block_fsb
-modelhess`.

```bash
optmain SN2.xyz -ma 150 1 6 -pyscf -elec -1 -spin 1 -opt rsirfo_block_fsb -modelhess
optmain SN2.xyz -ma 150 1 6 -xtb GFN2-xTB -opt rsirfo_block_fsb -modelhess
optmain SN2.xyz -ma 150 1 6 -os gfn1xtb -opt rsirfo_block_fsb -modelhess
```

`-ma 150 1 6` pushes atoms 1 and 6 together with 150 kJ/mol; the result is the structure
at the end of the pushed path, and `SN2_traj.xyz` is the path. With PySCF (B3LYP/def2-SVP,
the defaults) the run converged in 172 iterations; with GFN2-xTB in 145 and with the
built-in GFN1-xTB in 113 (charge -1 from the file).

MultiOptPy: `optmain aldol_rxn.xyz -ma 95 1 5 50 3 11 -pyscf -elec 0 -spin 0 -opt
rsirfo_block_fsb -modelhess`.

```bash
optmain aldol_rxn.xyz -ma 95 1 5 50 3 11 -xtb GFN2-xTB -opt rsirfo_block_fsb -modelhess
optmain aldol_rxn.xyz -ma 95 1 5 50 3 11 -os gfn1xtb -opt rsirfo_block_fsb -modelhess
```

Converged in 121 (GFN2-xTB) and 114 (GFN1-xTB) iterations. With `-pyscf -spin 1` in the
place of `-xtb GFN2-xTB` it is MultiOptPy's command; that form was run for `SN2.xyz` only.

### A transition state from a guess

```bash
optmain aldol_rxn_PT.xyz -xtb GFN2-xTB -opt rsirfo_block_bofill -order 1 -fc 5
optmain aldol_rxn_PT.xyz -os gfn1xtb -opt rsirfo_block_bofill -order 1 -fc 5
optmain aldol_rxn_PT.xyz -xtb GFN2-xTB -opt rsirfo_block_bofill -order 1 -fc 5 -freq
```

MultiOptPy's command as it is. Converged in 12 iterations with 3 exact Hessians (GFN2-xTB)
and in 13 with 3 (GFN1-xTB). With `-freq` the vibrational analysis follows: one imaginary
frequency (240i cm^-1 with GFN2-xTB), written to `normal_modes.txt` with the
thermochemistry in `thermochemistry.txt`.

### A path between structures

```bash
nebmain aldol_rxn -xtb GFN2-xTB -ns 50 -adpred 1 -nd 0.5
nebmain aldol_rxn -os gfn1xtb -ns 50 -adpred 1 -nd 0.5
```

MultiOptPy's command as it is: the 17 structures of the directory, put 0.5 Angstrom apart,
50 iterations. Both runs did their 50 iterations and named two candidates for transition
states (the images at the maxima of the energy profile); neither was relaxed to the
tolerance of mopt-rs by then (largest force 3e-3 and 7e-3 Hartree/Bohr against 3e-4), and
the log says how to go on (`nebmain --restart RUN_DIRECTORY -ns N`). MultiOptPy runs its
iterations without such a test. `-qsmv2` and `-qsm`, which MultiOptPy's settings of the
workflow use, are accepted as well (`reaction_paths.md`, "Reaction path between structures").

### A transition state between two structures (iEIP)

```bash
ieipmain ieip_test -xtb GFN2-xTB
ieipmain ieip_test -os gfn1xtb
```

MultiOptPy's command as it is. Converged after 88 iterations (GFN2-xTB) and 83 (GFN1-xTB);
the estimate is `hcn_A_ts_guess.xyz` in the run directory, and `optmain FILE -order 1`
from it finds the transition state (`-ieip_opt optimize=on` does that by itself).

### Molecular dynamics

```bash
mdmain aldol_rxn_PT.xyz -xtb GFN2-xTB -temp 298 -traj 1 -time 100000
```

MultiOptPy's command as it is: 100000 steps of 0.5 fs with a Nose-Hoover chain at 298 K.
It was started and stopped after five minutes; with `-time 2000` it ran to its end in 16
seconds (2001 gradients), and with `-os gfn1xtb -time 500` in 11 seconds.

### Orientations of two molecules

```bash
orientsearch aldol_rxn.xyz -part 1-4 -ma 95 1 5 50 3 11 -nsample 5 -xtb GFN2-xTB -opt rsirfo_block_fsb -modelhess
```

MultiOptPy's command (`python orientation_search.py ...`) as it is: five random
orientations of atoms 1-4 against the rest, each optimized with the artificial force. All
five converged, to two distinct structures. With `-os gfn1xtb -nsample 2` both samples
converged to one structure.

### Conformers

```bash
confsearch s8_for_confomation_search_test.xyz -xtb GFN2-xTB -ns 2000
confsearch s8_for_confomation_search_test.xyz -os gfn1xtb -ns 2000
```

MultiOptPy's command (`python conformation_search.py ...`) as it is. Finished after 31
candidates with 5 conformers (GFN2-xTB, under three minutes) and with 2 (GFN1-xTB, eight
minutes); `conformers.xyz` in the run directory is the ensemble.

### A relaxed scan

MultiOptPy: `relaxed_scan.py SN2.xyz -nsample 8 -scan bond 1,2 1.3,2.6 -elec -1 -spin 0
-xtb GFN2-xTB -opt crsirfo_block_fsb -modelhess`.

```bash
relaxedscan SN2.xyz -nsample 8 -scan bond 1,2 1.3,2.6 -xtb GFN2-xTB -opt crsirfo_block_fsb -modelhess
relaxedscan SN2.xyz -nsample 8 -scan bond 1,2 1.3,2.6 -os gfn1xtb -opt crsirfo_block_fsb -modelhess
```

Eight points of the C-F distance from 1.3 to 2.6 Angstrom, each optimized under the
constraint. The run lists the energies of the points, the minima along the scan, and the
candidates for transition states (`SN2_ts_candidate_1.xyz`). The charge is that of the
line `-1 1` of `SN2.xyz` at every point. With GFN2-xTB the scan names one candidate (the
point at 1.857 Angstrom, 60.3 kcal/mol above the lowest point; 466 gradients); with the
built-in GFN1-xTB the highest point is the first one, the end of the scan at 1.3 Angstrom,
and no candidate is named (175 gradients).

### The workflow to transition states (AutoTS)

```bash
run_autots aldol_rxn.xyz -cfg config_autots_run_xtb_test.json
run_autots aldol_rxn.xyz -cfg config_autots_builtin.json
```

MultiOptPy's quick start (`python run_autots.py aldol_rxn.xyz -cfg
config_autots_run_xtb_test.json`) as it is: step 1 pushes the two molecules together
(AFIR), step 2 relaxes the path of step 1 (`nebmain`), step 3 optimizes the highest images
to transition states with a vibrational analysis. `--run_step4` adds the IRC from every
transition state and the optimization of its ends. The work directory (`work_dir` of the
settings) gets `autots_report.html`, a page with the structures, the energy profile, and
the settings that opens in a browser, and `autots_summary.csv`.

What the runs did:

| Settings | Time | Step 3 |
|---|---|---|
| `config_autots_run_xtb_test.json` (GFN2-xTB, `QSMv2`) | 1.5 minutes | of three candidates one converged to a transition state with one imaginary frequency; two did not converge |
| `config_autots_builtin.json` (GFN1-xTB, `QSMv2`) | 2 minutes | three candidates converged, each with one imaginary frequency |

A candidate that does not converge is reported as such in the summary and in the report.
With GFN2-xTB and without `QSMv2` in step 2 (the default direction of the chain) the run
ended the same way: the same transition state from the second candidate (-17.526439
Hartree), the other two not converged.

The same workflow can be assembled step by step from the Python API, on `SN2.xyz`
with the built-in GFN1-xTB: see [python_workflows.md](python_workflows.md) and
`examples/python_autots/`.

### Reaction networks

`run_mapper` (MultiOptPy: `python run_mapper.py mapper_test.xyz -cfg config_mapper_test.json`)
takes the same settings file. It explores with AFIR pushes (between atoms, between molecules,
and on the cell), ADDF, and nanoreactor dynamics, finds the transition states and their IRCs,
takes a network to another surface (`-remap`, `-single_points`), maps crossing points of two
states, and solves the kinetics of the network, on isolated systems, ONIOM surfaces, and
periodic cells (fixed or variable); what it does not do on a surface is refused by name. See
[mapper.md](mapper.md).

## Where to go from here

- [running.md](running.md): how a run is set up and what it leaves (settings files,
  interruption and restart), with links to the pages of the calculators, the optimizations,
  the paths, the dynamics, and the searches; each with the measurements behind its defaults.
- [cli_reference.md](cli_reference.md): every option of every command, and which of
  MultiOptPy's options are not implemented yet.
- [migration.md](migration.md): where mopt-rs differs from MultiOptPy and why, by topic.
