# Open issues

What is known not to work, or not to be finished, in this release. Each entry says what
happens today and, where there is one, how to work around it. A combination that is refused
is refused with a message that names it; the entries marked *silent* are the ones where a run
completes and something is not as one would expect.

## Constraints and reaction coordinates

- **A frozen atom together with another hard constraint in an IRC fails.**
  `ircmain ... -fix 2 -pc bond 1,3` stops with "distance 1 / 3 is off by ... after 1 Newton
  steps", and with "energy is not finite" when the bond involves the frozen atom. Frozen atoms
  alone, or distances, angles and dihedrals alone, are held at every point. The same applies
  to the IRC stages of `run_autots` and `run_mapper`. Whether chains, variable-cell
  optimizations and molecular dynamics are free of the same combination has not been tested.
- **An IRC that starts from a very flat saddle point held by `-gfix` can fail** in the step
  that restores the constraint (seen on a Lennard-Jones cluster whose imaginary mode is about
  4i cm^-1). In `run_autots` such a failure ends the step for that transition state with a
  warning; the run goes on.
- **Hard constraints together with a variable cell are refused in `run_autots` and
  `run_mapper`** (the IRC of atoms and cell holds no constraints), and `-vc_fix` is refused in
  `run_autots`. `optmain`, `relaxedscan`, `nebmain` and `ieipmain` take them.

## Workflows

- **A network on a variable cell can depend on the last digits of the input** (*silent*). In
  the copper-vacancy example the push does not react, the guess for the saddle search lies at
  a point of mirror symmetry, and the search either climbs a soft mode of the cell to a
  transition state that leads to a second, strongly deformed minimum, or falls back to the
  start. Both networks are correct as far as they go; the second minimum is found by chance.
  The result does not depend on the number of threads.
- **Step 2 of `run_autots` with a variable cell** has not been compared with `nebmain -vc`,
  which was added last.
- **Following a run with stand-alone commands on the built-in GFN-FF.** The force field
  builds its topology from the structure each command is given, so a follow-up command can
  see another surface than the search did. `run_autots` hands the topology on by itself; with
  stand-alone commands pass `-calc_opt topology=FILE` (the structure of the first command).
  There is no warning when this is forgotten (*silent*).
- **`freqmain` cannot write the Hessian it computed** in the form `-ih` reads.

## Crossing points (`run_automecp`, `run_mapper` with states)

- The report matches the nodes of a path by their energies; on a variable cell nothing
  matches (*silent*).
- A conical intersection searched with the tight criteria can stall close to the cusp (the
  largest force stays near 2e-5 while the steps shrink) and end at the iteration limit; the
  normal criteria converge.
- A crossing point that the check within the seam finds to be a saddle point on the seam is
  reported as such, but it is not followed down along the seam automatically.

## ONIOM

- **Replicas of the environment** (`micro_replicas`): the number of replicas and the
  temperature have no measured recommendation. The option is off by default.

## Optimizers and options

- **Mode kill at a minimum.** The default continuation after removing an imaginary mode can
  advance very slowly along a soft mode (a thousand iterations without convergence in one
  measured case); `mode_kill_method classic` reached the minimum in 59. `run_autots` uses the
  classic form for the ends of its IRC.
- **`lup_max_step` and `method_relax_max_step`.** The relaxation after a two-point search can
  be asked for in two spellings; the older one limits the step of an image to 0.1 Angstrom by
  default, the newer one to the chain's own limit, so the two can take different steps.
- **Atom lists of `optmain`** are read in two places with slightly different rules
  (duplicates are kept by one of them).
- **`"save_pict": false` through the Python MCP server** does not switch the figures of
  `nebmain` off (*silent*). The settings file, the command line and the MCP server of the
  `mopt` binary take `false`.
- **Rigid-body motions of updated Hessians in `tric`** (`-update_proj`). The coordinates of
  `tric` have no basis of the translations and rotations of the whole structure, so its updated
  Hessians are kept as the formulas give them (Cartesian coordinates and `-vc` are projected;
  `ric` and `dic` hold no rigid motions). Likewise the updates of the IRC of a variable cell
  (`-irc` after `-vc`, whose space of plain variables has no atoms) and the Hessian `mnewton`
  keeps for itself in `-vc` (the optimizer of the cell gives its step generators no rigid
  basis) are not projected; the Hessians the cell optimizer keeps are.

## Not available

- Periodic boundary conditions for the built-in GFN0-xTB and GFN1-xTB, and the periodic D4
  dispersion correction.
- A workflow that repeats a transition-state search over several snapshots of an ONIOM
  environment and reports the spread of the barrier.
- A table of which command supports which kind of surface (molecule, fixed cell, variable
  cell, ONIOM) method by method; the pages of the commands name what each refuses.

## Performance

- The tables of a chain relaxation are opened and closed at every iteration.
- Several callers compute eigenvectors where eigenvalues alone would do.
