# Coordinate systems and the automatic choice

`-coord` selects the coordinates in which the optimizer takes its steps (calculators always
work in Cartesian coordinates): `cart`, `ric` (redundant internal), `dic` (delocalized
internal), `tric` and `tric-p` (translation-rotation internal coordinates, delocalized and
primitive), `tric-cart[:WEIGHT]` (TRIC stacked with the Cartesian coordinates times a weight
in the Wilson matrix, then delocalized), `spherical`, `hyperspherical` (also `polar`,
`hyperspherical:FILE`), and `auto`. This page records the measurements behind the default
weight of `tric-cart` and the rule of `auto`. A rule is changed only together with these
tables.

## Primitive internal coordinates at linear chains

The primitives of `ric`, `dic` and the internal part of `tric` (`coord::internal`) are bonds,
angles, the two linear-bend complements of every angle above 175 degrees, and dihedrals. A
dihedral through a linear angle has no value, so the twist about a linear chain had no
coordinate: a water dimer with a linear `O-H...O` (the link `H...O` makes the linear angle)
had 11 primitives of rank 11 instead of `3N - 6 = 12` after the set was generated anew at the
linear hydrogen bond, and allene (the twist of its two CH2 groups) lacked one as well. As in
the published schemes for linear chains (Bakken and Helgaker, J. Chem. Phys. 2002, 117, 9160;
the dihedrals of geomeTRIC that skip the linear atoms, Wang and Song, J. Chem. Phys. 2016, 144,
214108), every maximal chain `p0 - ... - pk` of atoms joined by linear angles now gets the
dihedrals `x - p0 - pk - y` about its axis from the neighbours of its ends off the line. The
Wilson matrix then has rank `3N - 6` (`3N - 5` for linear molecules) for the water dimer, HCN,
acetylene, allene, Ni(CO)4 and CO2...H2O (test
`coord::internal::tests::sets_with_linear_chains_span_the_internal_space`); sets of molecules
without linear angles are unchanged (`sets_without_linear_angles_do_not_change`).

## Spherical coordinates

Spherical coordinates are coordinate systems like the internal ones (values, Wilson
matrix, displacements with the periodic angle wrapped, and a request to be set up anew
where they break down), so every command and step generator works with them:

- `spherical`: every atom by its distance from the centroid of the structure and two
  angles. Each atom has its own polar axis, chosen when the coordinates are set up so that
  it starts on the equator; an atom within 0.2 Bohr of the centroid is described by its
  Cartesian displacement from it instead. The coordinates are set up anew when an atom
  comes within 5 degrees of a pole or within 0.1 Bohr of the centroid. They do not change
  under translations (3N coordinates of rank 3N - 3, treated as redundant).
- `hyperspherical` (`polar`): the whole structure as one point of 3N dimensions relative
  to the centroid of the structure, by its distance and 3N - 1 angles, MultiOptPy's polar
  coordinates (`Coordinate/polar_coordinate.py`, which MultiOptPy uses inside iEIP only).
  An orthogonal map set up with the coordinates turns the starting displacement onto the
  direction (1, ..., 1), where no partial norm vanishes; they are set up anew when an angle
  comes within 1e-3 rad of a pole. `hyperspherical:FILE` takes the structure of FILE as the
  centre: the radius is then the distance from it (a hypersphere around that structure is a
  surface of constant first coordinate), and the 3N coordinates are independent.

Chains (`nebmain -coord`) refuse spherical coordinates. The length of a change of an angle
grows with its partial norm, so distances and tangents along a chain in these coordinates
are far from those of the structures: HCN -> HNC in `hyperspherical:FILE` about a structure
off the path still had forces of 1 Hartree/Bohr after 150 iterations (GFN0-xTB), and about
a structure near the transition state images ran into the singular centre. Optimizations
are not affected: water and a distorted ethylene (GFN-FF, `-tcc`) reach the minima of
Cartesian coordinates within 1e-6 Hartree in `spherical`, `hyperspherical`, and `polar`
(water in 5 iterations each; ethylene 11 in Cartesian, 13 in `spherical`).

Periodic cells have no centroid; both systems refuse them. The n-dimensional functions are
`coord::hyperspherical` (`to_hyperspherical`, `from_hyperspherical`, both Jacobians, and the
gradient; from Python `mopt_rs._core.to_hyperspherical` and the others). Their measurements
in the benchmark below are not taken yet; `auto` does not choose them.

## How it was measured

`crates/mopt-rs/examples/bench_coordinates.rs` optimizes the same starting structures in
every coordinate system and counts the iterations to the default convergence criteria
(`rsirfo_bfgs`; GFN-FF for molecules, Lennard-Jones, EMT, and Tersoff for the others; the
transition state with `rsirfo_bofill`, GFN0-xTB, and an exact Hessian every 20 iterations):

```bash
cargo run --release -p mopt-rs --example bench_coordinates -- --systems=cart,dic,tric,tric-cart:0.03,tric-cart:0.1,tric-cart:0.3,tric-cart:1
```

```bash
cargo run --release -p mopt-rs --example bench_coordinates -- --modelhess=lindh2007d3 --systems=cart,dic,tric,tric-cart:0.1,tric-cart:0.3 --max=1000 --seconds=400
```

In the tables `>n` is a run stopped after `n` iterations (the iteration or time limit), and
`(+x)` a run that converged to a point `x` mHartree above the lowest one found for the case
(another minimum). The large cases of the first table were run on their own with longer
limits; C64H130 was not run to the end in any system, and the internal coordinates were not
run for the 32-atom copper cell.

## Without a model Hessian (the Hessian starts as the unit matrix)

| Case | Atoms | Fragments | cart | dic | tric | tric-cart:0.03 | tric-cart:0.1 | tric-cart:0.3 | tric-cart:1 |
|---|---|---|---|---|---|---|---|---|---|
| water | 3 | 1 | 9 | 10 | 10 | 10 | 10 | 10 | 9 |
| methane | 5 | 1 | 10 | 15 | 15 | 15 | 15 | 15 | 15 |
| ethane | 8 | 1 | 17 | 22 | 22 | 22 | 22 | 18 | 18 |
| butane (twisted) | 14 | 1 | 111 | 103 | 103 | 103 | 103 | 102 | 106 |
| octane (twisted) | 26 | 1 | 186 | 157 | 157 | 157 | 162 | 163 | 172 |
| hexadecane (twisted) | 50 | 1 | 299 | 324 | 324 | 270 | 258 | 255 | 289 |
| C32H66 (twisted) | 98 | 1 | 715 | 552 | 509 | >236 | 484 | 495 | >230 |
| C64H130 (twisted) | 194 | 1 | >175 | >50 | >25 | >34 | >32 | >34 | >32 |
| benzene | 12 | 1 | 26 | 25 | 25 | 25 | 25 | 25 | 26 |
| biphenyl (twisted) | 22 | 1 | 25 (+0.65) | 84 | 84 | 71 | 81 | 76 | 79 |
| CO2 (bent start) | 3 | 1 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| HCN (bent start) | 3 | 1 | 9 | 8 | 8 | 8 | 8 | 7 | 8 |
| water dimer (apart) | 6 | 2 | 100 | 68 | 49 | 57 | 58 | 58 | 74 |
| water trimer | 9 | 3 | 136 | 118 | 89 | 89 | 90 | 90 | 107 |
| water hexamer | 18 | 6 | 331 | >500 | 149 (+0.39) | 141 (+0.42) | 140 (+0.42) | 173 (+0.35) | 164 (+0.35) |
| (H2O)20 | 60 | 20 | >500 | >372 | 479 | >447 | 476 | 480 | >500 |
| benzene + water | 15 | 2 | 82 | 94 | 73 | 73 | 73 | 80 | 70 |
| benzene dimer | 24 | 2 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| butane + water | 17 | 2 | 341 | 122 (+0.13) | 76 (+0.58) | 76 (+0.58) | 77 (+0.58) | 77 (+0.62) | 327 |
| methane dimer | 10 | 2 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| Ar7 (compressed) | 7 | 3 | 27 | 62 | 28 | 28 | 28 | 28 | 30 |
| Ar13 | 13 | 13 | 33 | 70 | 33 | 33 | 33 | 33 | 33 |
| Cu13 | 13 | 1 | 26 | 56 | 33 | 33 | 33 | 33 | 33 |
| Cu fcc, 4 atoms | 4 | 1 | 10 | >500 | >500 | >500 | >500 | >500 | >500 |
| Diels-Alder TS | 16 | 2 | 22 | 25 | 29 | 28 | 27 | 23 | 24 |
| Si diamond, 8 atoms | 8 | 1 | 13 | 13 | 13 | 13 | 13 | 13 | 13 |
| Si diamond, 64 atoms | 64 | 1 | 24 | 26 | 26 | 26 | 26 | 26 | 26 |
| Cu fcc, 32 atoms | 32 | 1 | 20 | - | - | - | - | - | - |

## With a model Hessian (`-modelhess lindh2007d3`, every 50 iterations)

| Case | Atoms | Fragments | cart | dic | tric | tric-cart:0.1 | tric-cart:0.3 |
|---|---|---|---|---|---|---|---|
| water | 3 | 1 | 5 | 6 | 6 | 6 | 6 |
| methane | 5 | 1 | 7 | 7 | 7 | 7 | 7 |
| ethane | 8 | 1 | 16 | 12 | 12 | 12 | 12 |
| butane (twisted) | 14 | 1 | 29 | 19 | 19 | 19 | 20 |
| octane (twisted) | 26 | 1 | 55 | 26 | 26 | 29 | 31 |
| hexadecane (twisted) | 50 | 1 | 108 | 54 | 54 | 55 | 58 |
| C32H66 (twisted) | 98 | 1 | 267 | 153 | 150 | 151 | 157 |
| benzene | 12 | 1 | 29 | 26 | 26 | 26 | 26 |
| biphenyl (twisted) | 22 | 1 | 26 | 27 | 27 | 27 | 27 |
| CO2 (bent start) | 3 | 1 | 7 | 6 | 6 | 6 | 6 |
| HCN (bent start) | 3 | 1 | 7 | 6 | 6 | 6 | 6 |
| water dimer (apart) | 6 | 2 | 119 | 41 | 49 | 49 | 37 |
| water trimer | 9 | 3 | 112 | 45 | 82 | 74 | 81 |
| water hexamer | 18 | 6 | 136 (+0.39) | 682 (+1.87) | 123 | 131 | 87 |
| (H2O)20 | 60 | 20 | >1000 | 250 | >1000 | 848 (+0.28) | 249 (+13.20) |
| benzene + water | 15 | 2 | 67 | 40 | 35 | 38 | 34 |
| benzene dimer | 24 | 2 | 47 | 77 | 77 | 75 | 40 |
| butane + water | 17 | 2 | 91 | 78 | 38 | 39 | 44 |
| methane dimer | 10 | 2 | 193 | 38 | 38 | 42 | 85 |
| Ar7 (compressed) | 7 | 3 | 27 | 52 | 35 | 27 | 36 |
| Ar13 | 13 | 13 | 27 | 55 | 27 | 27 | 28 |
| Cu13 | 13 | 1 | 30 | 48 | 28 | 28 | 29 |
| Cu fcc, 4 atoms | 4 | 1 | 10 | 551 | >1000 | >1000 | >1000 |
| Si diamond, 8 atoms | 8 | 1 | 13 | 12 | 12 | 12 | 12 |
| Si diamond, 64 atoms | 64 | 1 | 19 | 20 | 20 | 20 | 20 |
| Diels-Alder TS | 16 | 2 | 22 | 25 | 29 | 27 | 23 |

## Iterations in total

Over the cases that every system finished at the lowest point found (the number of cases in
parentheses; the two tables do not share all of them):

| Kind | Hessian | cart | dic | tric | tric-cart:0.1 | tric-cart:0.3 |
|---|---|---|---|---|---|---|
| molecule (10) | unit matrix | 1389 | 1223 | 1180 | 1094 | 1097 |
| molecule (11) | model | 556 | 342 | 339 | 344 | 356 |
| complex (6) | unit matrix | 358 | 355 | 252 | 262 | 269 |
| complex (7) | model | 656 | 371 | 354 | 344 | 357 |
| **molecules and complexes** | both | **2959** | **2291** | **2125** | **2044** | **2079** |
| atom cluster (2) | unit matrix | 59 | 126 | 66 | 66 | 66 |
| atom cluster (2) | model | 57 | 103 | 55 | 55 | 57 |
| periodic (2) | unit matrix | 37 | 39 | 39 | 39 | 39 |
| periodic (2) | model | 32 | 32 | 32 | 32 | 32 |
| transition state (1) | exact, every 20 | 22 | 25 | 29 | 27 | 23 |

What the numbers say:

- With a model Hessian, internal coordinates need about 40 % fewer iterations than Cartesian
  coordinates for molecules and complexes; without one the gain is 10 to 30 %, and small rigid
  molecules (water, methane, ethane) are as fast or faster in Cartesian coordinates.
- `tric-cart:0.1` has the fewest iterations in total and is within 4 % of the best system in
  every row of molecules and complexes. Plain TRIC is best for complexes without a model
  Hessian (252 against 262) and behind for long chains without one (hexadecane 324 against
  258, C32H66 509 against 484).
- The weight: 0.1 and 0.3 are equal within 2 % in total. Without a model Hessian, 0.03 is as
  plain TRIC and 1 follows the Cartesian coordinates where those are slow (butane with water:
  327 iterations against 77; water dimer 74 against 58).
- Delocalized internal coordinates alone are erratic for complexes (water hexamer: more than
  500 iterations without a model Hessian, 682 to a higher minimum with one).
- Atoms without bonds, dense clusters, and periodic networks gain nothing from internal
  coordinates. For the 4-atom copper cell every internal system fails to converge.
- Water clusters have many minima, and the systems end in different ones ((H2O)20: between
  249 and more than 1000 iterations, to points up to 13 mHartree apart). These cases are not
  in the totals.
- One transition state is too few to rank the systems for saddle searches.

## Cost per iteration

Time of one iteration with the force fields above (the transformation dominates; with an
electronic-structure method it is negligible up to a few hundred atoms):

| Atoms | cart | dic | tric | tric-cart |
|---|---|---|---|---|
| 50 (hexadecane) | 0.02 s | 0.07 s | 0.09 s | 0.09 s |
| 98 (C32H66) | 0.06 s | 0.19 s | 0.35 s | 0.37 s |
| 194 (C64H130) | 0.5 s | 1.8 s | 3.7 s | 2.8 s |

## The defaults that follow

- `tric-cart` without a weight uses 0.1 (`coord::system::tric::DEFAULT_CARTESIAN_WEIGHT`).
- `auto` (`coord::system::auto`) chooses Cartesian coordinates for structures without bonds,
  for a fragment bonded to its own periodic image, for more than 5 bonds per atom on average,
  and for more than 300 atoms; `tric-cart` for everything else. The choice is made again
  during a run when fragments join or part or the structure moves into another of these
  classes.

How well each part of the rule is supported:

| Part | Support |
|---|---|
| `tric-cart` for molecules and complexes | 34 runs per system in the totals above |
| Cartesian without bonds | Ar13: the same iterations with and without a model Hessian |
| Cartesian for periodic networks | silicon cells tie; the 4-atom copper cell fails in internal coordinates |
| more than 5 bonds per atom | Cu13 (6.3): 26 against 33 iterations without a model Hessian, 30 against 28 with one; no case between 4 and 6.3 bonds per atom was measured |
| more than 300 atoms | the cost per iteration above, extrapolated; no optimization of that size was run |

The benchmark covers force fields only, alkanes and aromatic rings as molecules, and water as
the partner in most complexes. Before a rule is changed, the benchmark is run again and the
tables here are replaced.
