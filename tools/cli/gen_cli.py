"""Generate the Rust option tables and the CLI compatibility reference from the MultiOptPy spec.

Inputs: ``tools/cli/multioptpy_cli.json`` (from ``extract_multioptpy_cli.py``) and the status
tables below (which commands and options are implemented).
Outputs: ``crates/mopt-rs/src/cli/spec.rs``, ``docs/source/cli_reference.md``, and the same
tables as JSON with the positional inputs, ``python/mopt_rs/mcp/commands.json`` (the tools of
the MCP server of the Python package).

Usage:
    python -I tools/cli/gen_cli.py <repository root>

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# The commands, with the feature each provides; IMPLEMENTED_COMMANDS run, the others parse
# their options and report that the feature is not implemented yet.
COMMANDS = {
    "optmain": "geometry optimization",
    "nebmain": "reaction path methods (NEB and others)",
    "ieipmain": "iEIP and other two-structure methods",
    "mdmain": "molecular dynamics",
    "confsearch": "conformation search",
    "relaxedscan": "relaxed scan",
    "orientsearch": "orientation search",
    "run_autots": "automated transition-state search (AutoTS)",
    "run_mapper": "reaction network mapper",
    "run_rcmc": "rate constant matrix contraction (RCMC)",
    # Not in MultiOptPy: the analysis of `optmain -freq` for given structures.
    "freqmain": "vibrational analysis and thermochemistry",
    # Not in MultiOptPy: the IRC of `optmain -irc` for a given structure.
    "ircmain": "intrinsic reaction coordinate (IRC)",
    # Not in MultiOptPy: the kinetics of a network with a choice of rates and solvers (run_rcmc
    # is the same code with MultiOptPy's options and numbers).
    "run_kinetics": "kinetics of reaction networks (rate constants, populations; RCMC among the "
    "solvers)",
    # Not in MultiOptPy: the counterpart of run_autots for crossing points of two states.
    "run_automecp": "automated search for minimum-energy crossing points (MECP, MECI)",
    # Not in MultiOptPy: the rate constant of an electron transfer by EDEEL.
    "etmain": "electron-transfer rate constants (EDEEL diabatic states, SAES, Marcus with "
    "Scher-Holstein tunnelling, Zhu-Nakamura)",
}
IMPLEMENTED_COMMANDS = {"optmain", "freqmain", "ircmain", "nebmain", "ieipmain", "run_autots"}
# Molecular dynamics.
IMPLEMENTED_COMMANDS |= {"mdmain"}
# Orientations of the parts of a complex.
IMPLEMENTED_COMMANDS |= {"orientsearch"}
# Conformation search.
IMPLEMENTED_COMMANDS |= {"confsearch"}
# Relaxed and rigid scans of coordinates.
IMPLEMENTED_COMMANDS |= {"relaxedscan"}
# The reaction-network mapper (its parts that are not done yet are refused by name).
IMPLEMENTED_COMMANDS |= {"run_mapper"}
# The kinetics of networks (run_rcmc: MultiOptPy's options, the solver legacy_rcmc).
IMPLEMENTED_COMMANDS |= {"run_rcmc", "run_kinetics"}
# Crossing points of two states from one structure (a run of the mapper).
IMPLEMENTED_COMMANDS |= {"run_automecp"}
# Electron-transfer rate constants.
IMPLEMENTED_COMMANDS |= {"etmain"}

# run_mapper options (by destination) that are not implemented yet, with what is missing.
MAPPER_PENDING: dict[str, str] = {}
# run_mapper options that only fit the design of MultiOptPy's mapper, with what to use instead.
MAPPER_DROPPED = {
    "resume": "--restart RUN_DIR continues a run with the settings and evaluations recorded in it",
    "log_file": "the log of a run is mopt.log in its run directory",
}

# Commands that also accept every optmain option (MultiOptPy builds them on optimizeparser).
EXTENDS_OPTMAIN = {"confsearch", "relaxedscan", "orientsearch", "freqmain", "ircmain", "etmain"}

# optmain options (by destination) that work, besides the groups below.
OPTMAIN_IMPLEMENTED = {
    "INPUT",
    "oniom_flag",
    "NSTEP",
    "max_trust_radius",
    "min_trust_radius",
    "opt_method",
    "calc_exact_hess",
    "electronic_charge",
    "spin_multiplicity",
    "saddle_order",
    "othersoft",
    "pyscf",
    "tight_convergence_criteria",
    "loose_convergence_criteria",
    "shape_conditions",
    "dissociate_check",
    "model_function",
    "N_THREAD",
    "manual_AFIR",
    "no_symmetry_analyzer",
    "DELTA",
    "geom_info",
    "opt_fragment",
    "detect_negative_eigenvalues",
    # Maps of the path and dynamic modes of the iterations (analysis).
    "cmds",
    "pca",
    "koopman",
    # Vibrational analysis and thermochemistry (vib).
    "frequency_analysis",
    "frequency_analysis_cutoff",
    "frequency_correction_method",
    "force_frequency_analysis",
    "temperature",
    "pressure",
    # Electronic-structure settings, used by the calculators of mopt_rs.calculators.
    "basisset",
    "functional",
    "sub_basisset",
    "effective_core_potential",
    "dft_grid",
    "unrestrict",
    "SET_MEMORY",
    "excited_state",
    # IRC after the optimization (irc).
    "intrinsic_reaction_coordinates",
}

# The feature an option that is not implemented yet belongs to (for messages and the
# reference); unlisted ones are "this option".
# nebmain options (by destination) that are not implemented yet, with what is missing.
NEBMAIN_PENDING = {
    "OM": "the Onsager-Machlup action method",
    "BNEB": "the force of MultiOptPy's -bneb",
    "BNEB2": "the force of MultiOptPy's -bneb2",
    "DNEB": "the doubly nudged elastic band",
    "NESB": "the nudged elastic stiffness band",
    "align_distances_energy": "this redistribution",
    "align_distances_ritz_energy_predicted": "this redistribution",
    "align_distances_spline": "this redistribution",
    "align_distances_spline_ver2": "this redistribution",
    "align_distances_geodesic": "this redistribution",
    "align_distances_bernstein": "this redistribution",
    "align_distances_bernstein_energy": "this redistribution",
    "align_distances_adaptive_energy": "this redistribution",
    "align_distances_savgol": "this redistribution",
    "node_distance_spline": "this placement of the first chain",
    "node_distance_savgol": "this placement of the first chain",
    "conjugate_gradient": "conjugate-gradient steps of the chain",
    "memory_limited_BFGS": "L-BFGS steps of the chain",
    "climbing_image": "the relocation of an image to the maximum of the profile (-ci)",
}

# MultiOptPy options that mopt-rs does not provide (by decision), with what to use instead.
DROPPED_OPTIONS = {
    "sqm1": "SQM1 is not part of mopt-rs; use another engine (-xtb, -os)",
    "sqm2": "SQM2 is not part of mopt-rs; use another engine (-xtb, -os)",
}

OPTION_FEATURES = {
    "intrinsic_reaction_coordinates": "IRC",
    "frequency_analysis": "vibrational analysis",
    "frequency_analysis_cutoff": "vibrational analysis",
    "frequency_correction_method": "vibrational analysis",
    "force_frequency_analysis": "vibrational analysis",
    "temperature": "thermochemistry",
    "pressure": "thermochemistry",
    "detect_negative_eigenvalues": "vibrational analysis",
    "oniom_flag": "ONIOM",
}

# Option groups that work (by destination; optmain and the commands built on it).
BIAS_OPTIONS = [
    "manual_AFIR",
    "repulsive_potential",
    "repulsive_potential_v2",
    "repulsive_potential_gaussian",
    "cone_potential",
    "flux_potential",
    "keep_pot",
    "keep_pot_v2",
    "anharmonic_keep_pot",
    "keep_angle",
    "keep_angle_v2",
    "universal_potential",
    "keep_dihedral_angle",
    "keep_out_of_plain_angle",
    "keep_dihedral_angle_v2",
    "keep_dihedral_angle_cos",
    "keep_out_of_plain_angle_v2",
    "void_point_pot",
    "bond_range_potential",
    "well_pot",
    "wall_well_pot",
    "void_point_well_pot",
    "around_well_pot",
    "spacer_model_potential",
    "metadynamics",
    "linear_mechano_force_pot",
    "linear_mechano_force_pot_v2",
    "asymmetric_ellipsoidal_repulsive_potential",
    "asymmetric_ellipsoidal_repulsive_potential_v2",
    "nano_reactor_potential",
]
CONSTRAINT_OPTIONS = ["fix_atoms", "projection_constrain", "gradient_fix_atoms", "user_constraint"]
MODEL_HESSIAN_OPTIONS = ["use_model_hessian", "calc_model_hess"]
# Calculator selection and settings (cli::calculator; programs through mopt_rs.calculators).
CALCULATOR_OPTIONS = [
    "usextb",
    "usedxtb",
    "cpcm_solv_model",
    "alpb_solv_model",
    "software_path_file",
]
IMPLEMENTED = (
    set(BIAS_OPTIONS)
    | set(CONSTRAINT_OPTIONS)
    | set(MODEL_HESSIAN_OPTIONS)
    | set(CALCULATOR_OPTIONS)
)

# Options added by mopt-rs (not in MultiOptPy), appended to optmain and the commands built on it.
EXTRA_OPTMAIN = [
    {
        "names": ["-oniom_middle", "--oniom_middle"],
        "nargs": "*",
        "type": "str",
        "default": [],
        "help": "a middle layer of -oniom: ATOMS LINK CALCULATOR (as -oniom; ATOMS hold the atoms "
        "of the high layer and more): three-layer ONIOM, E_low(real) + E_middle(middle) - "
        "E_low(middle) + E_high(model) - E_middle(model)",
    },
    {
        "names": ["-oniom_opt", "--oniom_options"],
        "nargs": "*",
        "type": "str",
        "default": [],
        "help": "settings of -oniom as KEY=VALUE: low.KEY=VALUE and middle.KEY=VALUE (settings of "
        "the calculator of that level, which can be any calculator of -os, an ASE calculator "
        "among them: method, basis, and its own keys as -calc_opt takes them; -func, -bs, and "
        "-calc_opt belong to the highest level and are not handed down), "
        "link (scaled, the default: the link atom at the "
        "share g = (R(Q1) + R(H)) / (R(Q1) + R(Q2)) of the bond, as Gaussian; distance: at R(Q1) "
        "+ R(H) from Q1, as MultiOptPy), high_charge and high_multiplicity, middle_charge and "
        "middle_multiplicity (the model systems; by default the charge of the whole structure "
        "and its multiplicity, changed by one where the link atoms change the parity of the "
        "electrons), embedding (mechanical, the default; electronic: the higher level of every "
        "region in the charges of the atoms around it, the Coulomb energy of the model with them "
        "taken off with the lower level), charges (lowest, the default: the lowest level's, the "
        "topology charges of GFN-FF; or FILE: one per atom, in e), boundary (rc, the default: "
        "the charge of each Q2 moved to the midpoints of its bonds to the environment; zero; "
        "keep), micro (on by default: optmain, ircmain, and the chains of nebmain move the core, "
        "the outermost model region and the outer atoms of its links, and relax the rest on the "
        "lowest level (with electronic embedding the whole surface) to a minimum before every "
        "evaluation; Hessians are those of the core; the files hold the whole system; off: every "
        "atom moves), micro_mode_kill (off, the default: the environment, the low layer, is "
        "not mode-killed, the core is by its optimizer; N: the check for negative curvature of "
        "the environment after the first and every Nth relaxation, its modes followed down), "
        "micro_curvature (guided, the default: the check starts from the softest mode of the "
        "last one and takes the limited-memory BFGS of the relaxation as preconditioner; "
        "plain: from a fixed start without one), micro_precondition (lindh2007d3, the default, or "
        "another model Hessian of -modelhess, or none: its block of the environment, built at the "
        "free atoms and the fixed atoms within 4 Angstrom of the environment and shifted by "
        "micro_precondition_shift, default 1e-3 Hartree/Bohr^2, is the initial matrix of the "
        "limited-memory BFGS of the relaxations and the preconditioner of the curvature check; "
        "up to 16 are kept, one per structure they were built near; environments of at most 6000 "
        "variables; a relaxation that fails with it is made again without it), "
        "micro_tolerance (largest gradient component of the relaxed environment; by default "
        "micro_tolerance_factor times the thresholds of the forces of the run itself, the "
        "largest force and the RMS force of its criteria; a relaxation whose line search fails "
        "goes on with FIRE, one that ends above its tolerance says so), micro_tolerance_factor "
        "(1, the default, or a smaller share such as 0.1 of the thresholds of the forces of the run to which the "
        "environment is relaxed, not below 1e-5; up to 1), micro_check_tolerance (largest "
        "gradient component of the environment, Hartree/Bohr, for every Hessian of a check of "
        "stability: the verification of a converged saddle-point search, the start of an IRC, "
        "the stationary points of the mapper, a vibrational analysis; 1e-5 by default, never "
        "looser than the relaxations of the run; an environment that stands on a saddle point "
        "of its own is moved off it and relaxed again before any Hessian, and the error of the "
        "curvatures of a check is estimated by moving the environment to the minimum of its "
        "quadratic model and taking the Hessian again), micro_correct (on, the default, or off: correct the "
        "energy and the core gradient for the gradient left in the environment by the response "
        "of the newest Hessian of the core, where that moves the environment by at most 0.05 "
        "Bohr), micro_core_shell (Angstrom: the free atoms of the environment within that "
        "distance of the model region are moved by the optimizer with the core; by default "
        "none), micro_hessian (effective, the default: the Hessian of the relaxed "
        "surface; approximate: the same with the inverse of the limited-memory BFGS of the "
        "relaxations; held: the core block with the environment held), micro_hessian_solve "
        "(auto, the default: the response of the environment from its dense block, by the "
        "analytic Hessian of the lowest level or central differences of its gradient, where that "
        "costs fewer gradients than conjugate gradients and fits in the memory, conjugate "
        "gradients otherwise; dense; cg), micro_hessian_memory (GiB "
        "the Hessian of the core may take, by default half of the free memory: the analytic "
        "Hessian of the whole system where its matrices fit, central differences otherwise, "
        "refused where those do not fit either), micro_hessian_shell (Angstrom: only the "
        "environment within that distance of the core responds in the Hessian of the core, "
        "the rest is held; by default all of it), micro_optimizer (lbfgs, the "
        "default; tn: truncated Newton; or a method of -opt, for environments of at most 500 "
        "atoms), micro_coord (cart, the default; fragments: translation, rotation, and internal "
        "coordinates of each fragment of the environment), micro_predict (on or off, the "
        "default: start each relaxation from the predicted response of the environment), "
        "micro_adaptive (0, the default; ETA: the tolerance of a relaxation ETA times the "
        "largest core gradient while that is large, and a last relaxation to the full "
        "tolerance at the end), micro_replicas (1, the default: one environment; K: K replicas "
        "of the environment, each relaxed from its own environment of the core before, "
        "combined into the soft minimum -kT ln sum_k exp(-E_k / kT) of their surfaces, whose "
        "gradient is the weighted mean of theirs; not a free energy, at most kT ln K below the "
        "lowest replica; the files hold the lowest replica and NAME_replicas.xyz all of them), "
        "micro_replica_temperature (K of the soft minimum, 298.15 by default), "
        "micro_replica_start (perturb, the default: every free atom of the first relaxed "
        "environment moved by micro_replica_amplitude, 0.05 Angstrom, in a random direction "
        "of seed micro_replica_seed, 1; md: after micro_replica_md_steps, 200, steps of "
        "molecular dynamics of the environment at the temperature with the core held; files: "
        "the environments of the last K whole structures of micro_replica_files, such as the "
        "NAME_replicas.xyz of a run with replicas), traj_high_only (on or off, default off: write only the real "
        "high-layer atoms to optimization, MD and mapper XYZ trajectories)",
    },
    {
        "names": ["-stability", "--scf_stability"],
        "type": "int",
        "default": 0,
        "help": "test the SCF solution for instabilities every N evaluations and go on from a "
        "stabler one where it is unstable, repeating the evaluation there (0, the default: "
        "never; programs with an SCF that can test it, such as PySCF; the others say so once)",
    },
    {
        "names": ["-stability_external", "--scf_stability_external"],
        "type": "str",
        "default": "report",
        "help": "with -stability: an instability towards another kind of wavefunction "
        "(restricted to unrestricted): report (the default: it is said, the kind is kept), "
        "switch (go on with the unrestricted kind for the rest of the run), off (not tested)",
    },
    {
        "names": ["-parallel", "--parallel"],
        "type": "int",
        "default": 0,
        "help": "how many structures are calculated at the same time where a command has "
        "several that do not depend on each other (the images of a chain, the two "
        "structures of a search between two structures, the states of a function of "
        "several states): the threads of the command for the built-in calculators, copies "
        "in worker processes for the calculators of the Python side. -core stays the "
        "threads of one calculation (0: as many structures at a time as -core has threads "
        "for the built-in calculators, one for those of the Python side)",
    },
    {
        "names": ["-es_follow", "--follow_states"],
        "action": "store_true",
        "default": False,
        "help": "an excited state (-es N, the states of -mf) is the state that was number N "
        "at the first structure and is followed from structure to structure by the overlaps "
        "of its amplitudes, instead of the state number N in the order of the energies at "
        "every structure (calculators that hand out their states: the built-in GFN0-xTB and "
        "PySCF with a restricted reference; a state that is not found again with certainty "
        "is reported)",
    },
    {
        "names": ["-ih", "--initial_hessian"],
        "type": "str",
        "default": None,
        "help": "file with the Hessian the run starts from (Cartesian, Hartree/Bohr^2): a NumPy "
        "file, an ORCA .hess file, a Gaussian .fchk file or output, or numbers only (the "
        "square or its lower triangle); the first Hessian that is asked for is taken from it",
    },
    {
        "names": ["-follow", "--follow_direction"],
        "type": "str",
        "default": None,
        "help": "xyz file of displacements (any length): a transition-state search (-order 1) "
        "takes the Hessian mode closest to it uphill and then tracks that mode by overlap, "
        "instead of the lowest mode (RS-I-RFO becomes its mode-following variant mf_rsirfo_*); "
        "for example the tangent of a chain at the image it starts from; Cartesian coordinates",
    },
    {
        "names": ["-coord", "--coordinate_system"],
        "type": "str",
        "default": "cart",
        "help": "coordinate system of the optimization: cart, ric, dic, tric (translation-rotation "
        "internal coordinates, delocalized), tric-p (their primitive set), tric-cart[:WEIGHT] (TRIC "
        "stacked with weighted Cartesian coordinates), spherical (every atom by r, theta, phi "
        "about the centroid), hyperspherical or polar (the whole structure by its distance "
        "from the centroid and 3N-1 angles; hyperspherical:FILE about the structure of FILE; "
        "for optimizations, not chains), "
        "auto (tric-cart, or cart for atom clusters, dense or periodic networks and very large "
        "systems), or a registered name. Calculators always receive Cartesian structures.",
    },
    {
        "names": ["-ubias", "--user_bias_expression"],
        "nargs": "*",
        "type": "str",
        "default": [],
        "help": "user-defined bias potentials as expressions of geometric variables in atomic "
        "units (energy in Hartree), e.g. '0.15*(dist(1,2)-1.5*ang)^2'; functions dist, angle, "
        "dihedral, oop, x, y, z with atoms or groups [1-3]; see the bias documentation",
    },
    {
        "names": ["-unpaired", "--unpaired_electrons"],
        "nargs": "*",
        "type": "int",
        "default": [],
        "help": "the spin state as the number of unpaired electrons 2S instead of -spin (2S + 1), "
        "as the 'spin' of PySCF, xTB's uhf, and several Rust engines count it; one value, or one "
        "per state. It sets the state only: an unrestricted reference for singlets is -u",
    },
    {
        "names": ["-calc_opt", "--calc_option"],
        "nargs": "*",
        "type": "str",
        "default": [],
        "help": "program-specific calculator settings as key=value (Python literals), e.g. "
        "density_fit=True conv_tol=1e-10; for -os ase:module:Class they are constructor keywords",
    },
    {
        "names": ["-uc", "--user_constraint"],
        "nargs": "*",
        "type": "str",
        "default": [],
        "help": "user-defined constraints as expressions in the -ubias language: 'EXPR' holds "
        "the starting value, 'EXPR = VALUE' a value (atomic units; 'ang', 'deg' convert), "
        "'; period = 2*pi' wraps angle-like functions, '; tolerance = 1e-8' holds a noisy "
        "function to that residual instead of 1e-10, e.g. 'dist(1,2) - dist(3,4)'; with -vc "
        "the functions of the cell as well, e.g. 'volume()', 'cell_a() = 10.5*ang', "
        "'cell_a()/cell_b()'",
    },
    {
        # MultiOptPy has -gfix in the parsers of nebmain and ieipmain only (and drops it);
        # mopt-rs holds it in optmain and the commands built on it as well, so that the
        # follow-up optimizations of a chain keep it.
        "names": ["-gfix", "--gradient_fix_atoms"],
        "nargs": "*",
        "type": "str",
        "default": [],
        "help": "groups of atoms (ex.) 1,2,3 4,5 whose distances to each other are held at their "
        "values in the input (MultiOptPy: 'set the gradient of internal coordinates between "
        "atoms to zero'): every distance of a group is a constraint as -pc bond is, taken out of "
        "the gradient and of the Hessian and restored after every step",
    },
    {
        "names": ["-ccqn_bonds", "--ccqn_reactive_bonds"],
        "nargs": "*",
        "type": "str",
        "default": [],
        "help": "CCQN reaction-coordinate axis from reactive bonds, as atom pairs (ex.) 1,2 3,4",
    },
    {
        "names": ["-ccqn_product", "--ccqn_product"],
        "type": "str",
        "default": None,
        "help": "CCQN axis toward the midpoint of the straight line to this product structure (xyz)",
    },
    {
        "names": ["-ccqn_cosphi", "--ccqn_cos_phi"],
        "type": "float",
        "default": None,
        "help": "cosine of the CCQN cone half-angle (default: 0.1 for bonds, 0.5 for a product)",
    },
    {
        "names": ["-ccqn_step", "--ccqn_step"],
        "type": "float",
        "default": None,
        "help": "CCQN uphill step and initial saddle trust radius in Angstrom (default: 0.1)",
    },
    {
        "names": ["-freq_lowest", "--frequency_lowest"],
        "type": "float",
        "default": 10.0,
        "help": "vibrations at or below this wavenumber (cm^-1) are left out of the "
        "thermochemistry",
    },
    {
        "names": ["-freq_corr_h", "--frequency_correction_enthalpy"],
        "type": "str",
        "default": None,
        "help": "treatment of the thermal energy of low frequencies apart from that of the "
        "entropy (-freq_corr): head-gordon, truhlar, or none; without it, the one that goes "
        "with -freq_corr (grimme: head-gordon, truhlar: truhlar)",
    },
    {
        "names": ["-freq_cutoff_h", "--frequency_cutoff_enthalpy"],
        "type": "float",
        "default": None,
        "help": "cutoff (cm^-1) of the treatment of the thermal energy; without it, -freq_cutoff",
    },
    {
        "names": ["-conc", "--concentration"],
        "type": "float",
        "default": None,
        "help": "concentration (mol/L) for the translational entropy of a solute instead "
        "of the pressure of the gas",
    },
    {
        "names": ["-freq_scale", "--frequency_scale"],
        "type": "float",
        "default": 1.0,
        "help": "scale factor of the vibrational frequencies in the thermochemistry",
    },
    {
        "names": ["-zpe_scale", "--zero_point_scale"],
        "type": "float",
        "default": None,
        "help": "scale factor of the frequencies in the zero-point energy; without it, -freq_scale",
    },
    {
        "names": ["-freq_invert", "--frequency_invert"],
        "type": "float",
        "default": None,
        "help": "imaginary frequencies up to this magnitude (cm^-1) enter the "
        "thermochemistry as real ones",
    },
    {
        "names": ["-freq_bav", "--frequency_rotor_moment"],
        "type": "str",
        "default": "global",
        "help": "moment of inertia that bounds the free rotor of -freq_corr grimme: global "
        "(1e-44 kg m^2) or molecule (the mean moment of the molecule)",
    },
    {
        "names": ["-symm_tol", "--symmetry_tolerance"],
        "type": "float",
        "default": 0.01,
        "help": "distance (Bohr) within which atoms coincide in the point-group analysis",
    },
    {
        "names": ["-project_gradient", "--project_gradient"],
        "action": "store_true",
        "default": False,
        "help": "vibrational analysis: project the direction of the gradient out with the "
        "rigid-body motions (the modes orthogonal to a reaction path at a point that is not "
        "stationary)",
    },
    {
        "names": ["-bias_derivatives", "--bias_derivatives"],
        "action": "store_true",
        "default": False,
        "help": "after a converged optimization, the derivatives in every constant of the bias "
        "options (a force constant, an equilibrium value, the AFIR collision energy, ...; per "
        "unit of the option as it is written): of the bias energy, and how the stationary "
        "point and its energy without the bias move with the constant (dx/dp = -H^+ dg/dp with "
        "the Hessian of the whole surface); written to bias_derivatives.csv and the log",
    },
    {
        "names": ["-no_crossing_check", "--no_crossing_check"],
        "action": "store_true",
        "default": False,
        "help": "-mf opt_meci: do not check the branching plane with gradients at displaced "
        "structures when the search has converged (the check costs four evaluations of both "
        "states and sends the search on if the point is not stationary on the seam)",
    },
    {
        "names": ["-projection_trust", "--projection_trust"],
        "type": "str",
        "default": "energy",
        "help": "-mf opt_mesx, -mf opt_meci: what a step is judged by when the trust radius is "
        "adapted: energy (the change of the mean energy of the two states against the "
        "quadratic model of the projected gradient, as for a search on one surface) or merit "
        "(the change of the function the projected gradient belongs to: the mean energy within "
        "the seam and (E1 - E2)^2 / |g1 - g2|)",
    },
    {
        "names": ["-vc", "--variable_cell"],
        "action": "store_true",
        "default": False,
        "help": "optimize the cell together with the atoms (cells periodic in three "
        "directions; the stress comes from the calculator, or from differences of the energy "
        "where it has none); with -es the stress of that state, with -mf the two states "
        "combined in the variables of the cell (their enthalpies and their gradients in the "
        "variables: crossing points and conical intersections with the cell)",
    },
    {
        "names": ["-vc_pressure", "--variable_cell_pressure"],
        "type": "float",
        "default": 0.0,
        "help": "-vc: external pressure in GPa; the enthalpy E + P V is minimized",
    },
    {
        "names": ["-vc_afir", "--variable_cell_afir"],
        "nargs": "*",
        "default": [],
        "help": "-vc: artificial force on the cell (periodic AFIR): pairs of a collision "
        "energy in kJ/mol (positive: compress, negative: expand; a ramp as for -ma) and the "
        "distances it acts on, a list of a, b, c (origin to the tip of a lattice vector) and "
        "ab, bc, ca (between two tips), e.g. -vc_afir 200 a,b -vc_afir -100 ab",
    },
    {
        "names": ["-vc_fix", "--variable_cell_fix"],
        "nargs": "*",
        "type": "str",
        "default": [],
        "help": "-vc: hard constraints on the cell, comma-separated, at the values of the input "
        "or NAME=VALUE: a, b, c (lengths, Angstrom), alpha, beta, gamma (degrees), volume "
        "(Angstrom^3; only the shape relaxes), shape (b/a, c/a and the three angles; only an "
        "isotropic scaling relaxes), a_vector, b_vector, c_vector (the three Cartesian "
        "components of a lattice vector, e.g. the normal vector of a slab), ax, ay, ..., cz "
        "(one component, Angstrom); the same constraints as -uc 'cell_a()', 'volume()', ...; "
        "e.g. -vc_fix volume, -vc_fix c_vector, -vc_fix a=10.5,gamma",
    },
    {
        "names": ["-irc_opt", "--irc_options"],
        "nargs": "*",
        "default": [],
        "help": "-irc: further settings of the IRC as KEY=VALUE: first_step (check, the default: "
        "test the first point of each side and replace it by the energy minimum on a "
        "hypersphere around the transition state when it fails; off: MultiOptPy's first "
        "point), angle (largest angle in degrees between the downhill direction at the "
        "first point and the transition vector, 10), rho_min and rho_max (limits of actual "
        "over predicted energy change of the first step, 0.75 and 1.25), initial "
        "(mass-weighted length of the displacement from the transition state, 0.1 of the "
        "step size), sides (both, forward, backward), masses (standard: atomic weights, as "
        "the vibrational analysis; isotope: the most abundant isotopes, as MultiOptPy), "
        "energy_rise (retry, the default: a step that raised the energy is taken again at "
        "half its length, LQA and Euler; stop: end a side where the energy rises; off: keep "
        "such points, as MultiOptPy), overshoot (retry, the default, with energy_rise=retry: "
        "a step whose end has a gradient that points back along it is taken again at half "
        "its length; keep: such points stay), bend (off, the default; a share such as 0.25: "
        "no step is longer than the one that loses this share of the energy it gains by "
        "cutting the bend of the valley, from the curvature of the path and the Hessian "
        "across it; for exact Hessians along the path, -fc 1), approach (on, the default: "
        "below the force "
        "thresholds a side goes on to the minimum until the gradient is below "
        "approach_max_force "
        "and approach_rms_force, LQA, Euler, RK4; off: end at the first point below the "
        "thresholds, as MultiOptPy), approach_max_force (0.000015), approach_rms_force "
        "(0.00001), approach_steps (100), "
        "max_force (0.0004), rms_force (0.0001), min_points (10), oscillations (5), "
        "sphere_iterations (30), sphere_gradient (1e-4), sphere_step (1e-3)",
    },
    {
        "names": ["-bitss", "--bitss_options"],
        "nargs": "*",
        "default": [],
        "help": "-mf bitss REFERENCE: settings of the binary-image transition state search "
        "as KEY=VALUE: f (reduction of the separation per stage, 0.5), alpha (10), beta "
        "(0.1), interval (iterations between two settings of the coefficients, 100), points "
        "(points between the images for the barrier estimate, 3), min_separation (Bohr, "
        "0.01), stages (30), step_share (longest step as a share of the separation asked "
        "for, 0.5; none: no limit), midpoint_shift (end when the midpoint moved by less than "
        "this share of the reduction of a stage; none), midpoint_gradient (true)",
    },
    {
        "names": ["-vc_weight", "--variable_cell_weight"],
        "type": "float",
        "default": 2.0,
        "help": "-vc: weight w of the lattice variables, gamma = w sqrt(number of atoms) "
        "(Gubler et al. 2023; values between 1 and 2 are recommended there)",
    },
    {
        "names": ["-debug", "--debug"],
        "action": "store_true",
        "default": False,
        "help": "show on the screen what the optimization methods do inside a step (the "
        "messages of the inner solvers, such as those of RS-I-RFO); they are always written "
        "to mopt.log",
    },
    {
        "names": ["-modekill", "--mode_kill"],
        "action": "store_true",
        "default": False,
        "help": "at convergence, check the Hessian and leave a stationary point that has more "
        "negative curvatures than -order asks for (0: a minimum), then go on; needs Hessians "
        "from the calculator",
    },
    {
        "names": ["-modekill_method", "--mode_kill_method"],
        "type": "str",
        "default": None,
        "help": "optmain -modekill post-processing: hisd (default for minima) follows HiSD at the requested order after "
        "displacement; classic continues with the selected optimizer (default for saddle searches)",
    },
    {
        "names": ["-ccqn_solver", "--ccqn_cone_solver"],
        "type": "str",
        "default": "exact",
        "help": "solver of the CCQN uphill subproblem: exact (global) or alm (the paper's method)",
    },
]

NOTES = {
    (
        "optmain",
        "opt_method",
    ): "without this option mopt-rs uses rsirfo_block_fsb, and rsirfo_block_bofill with "
    "-order 1 and above (MultiOptPy: FIRELARS, which can be given)",
    (
        "optmain",
        "model_function",
    ): "the functions: seam, conical, avoiding (the Avoiding model function of Maeda, Ohno, "
    "and Morokuma, the upper adiabatic state; avoiding_lower the lower; legacy_avoiding "
    "MultiOptPy's avoiding), opt_mesx, opt_mesx_2, opt_meci",
    (
        "nebmain",
        "NSTEP",
    ): "mopt-rs defaults to at most 30 NEB iterations or 60 direct MaxFlux iterations",
    (
        "nebmain",
        "use_model_hessian",
    ): "mopt-rs uses the model Hessian lindh2007d3 without this option (the articles use "
    "o1numhess, a Hessian from further gradients, which can be given); "
    "-modelhess none: no Hessians, the steps along the forces alone, MultiOptPy's default",
    (
        "nebmain",
        "node_distance",
    ): "Angstrom; mopt-rs uses 0.5 for structures without this option (the settings of the "
    "articles); -nd 0 keeps the images as they are read, MultiOptPy's default",
    (
        "nebmain",
        "align_distances_energy_predicted",
    ): "mopt-rs uses 2 without this option (the settings of the articles); -adpred 0: never, "
    "MultiOptPy's default",
    ("nebmain", "fixedges"): "0 (default): both end images are minimized",
    (
        "nebmain",
        "DMF",
    ): "the direct MaxFlux method as its authors define it (a spline path, energies at a "
    "few points that move towards its top), not MultiOptPy's force on the images; -ns is "
    "its largest number of iterations, 60 without it",
    ("nebmain", "save_pict"): "on by default in mopt-rs, unlike MultiOptPy, which draws the "
    "figures only with -spng: the figures are drawn after every iteration; -spng off (or "
    "\"save_pict\": false in a settings file) writes none and the CSV data as before; -spng "
    "alone keeps meaning on",
    ("nebmain", "not_ts_optimization"): "has no effect (it has none in MultiOptPy either)",
    ("optmain", "manual_AFIR"): "minimum-image distances under PBC",
    (
        "optmain",
        "othersoft",
    ): "default `pyscf`; also `psi4`, `tblite`, `dxtb`, `lj`, `emt`, `eam`, `meam`, `tersoff`, "
    "`muller_brown`, ASE profiles, and `ase:module:Class` for any ASE calculator (programs "
    "other than the Rust potentials need the Python package: the pip-installed commands or "
    "`mopt-py`)",
    (
        "optmain",
        "max_trust_radius",
    ): "used in Bohr, as MultiOptPy does (its help text says Angstrom); without it, an RS-I-RFO "
    "search on a gradient projection of two states (`-mf opt_mesx`, `-mf opt_meci`) has 0.1 "
    "Angstrom (0.189 Bohr) as its largest trust radius",
    ("optmain", "dissociate_check"): "Angstrom",
    ("optmain", "no_symmetry_analyzer"): "the point-group analysis itself is not implemented yet",
    ("optmain", "N_THREAD"): "threads of the calculator and of the native thread pool",
    (
        "optmain",
        "intrinsic_reaction_coordinates",
    ): "methods lqa, hpc, dvv, euler, rk4, modekill[:MODES], and gs2 (mopt-rs); see irc.md",
}

# Notes of mdmain (molecular dynamics).
NOTES.update(
    {
        (
            "mdmain",
            "timestep",
        ): "atomic units with masses in electron masses (MultiOptPy takes the masses in amu, "
        "so that its steps are 42.7 times longer than the value says); 'fs' after the number "
        "gives femtoseconds; without it 0.5 fs for atoms and 0.1 for plain variables "
        "(measured, see molecular_dynamics.md)",
        ("mdmain", "NSTEP"): "the number of steps",
        (
            "mdmain",
            "mdtype",
        ): "nosehoover (nvt), nosehooverchain, langevin (baoab), velocityverlet (nve) as in "
        "MultiOptPy, and berendsen, bussi (csvr), andersen, npt (npt_mtk), npt_aniso "
        "(npt_mtk_aniso: a fully flexible cell), npt_berendsen, npt_berendsen_aniso, "
        "npt_crescale, npt_crescale_aniso; -md_opt sets the parts one by one; without this "
        "option mopt-rs uses nosehooverchain (measured, see molecular_dynamics.md)",
        (
            "mdmain",
            "pressure",
        ): "kPa; the target of the barostat of an NPT run (MultiOptPy reads it without using it)",
        (
            "mdmain",
            "change_temperature",
        ): "pairs of a step and a temperature (K): the target from that step on",
        (
            "mdmain",
            "constraint_condition",
        ): "distance (ang.), angle, or dihedral angle (deg.) with its atoms, held by SHAKE and "
        "RATTLE (as -pc manual)",
        (
            "mdmain",
            "periodic_boundary_condition",
        ): "a cell for the calculator (Lx,Ly,Lz in ang., or the forms of extended XYZ); "
        "an input with a Lattice header has its own",
        ("mdmain", "saddle_order"): "has no effect (it has none in MultiOptPy either)",
        ("mdmain", "additional_inputs"): "has no effect (it has none in MultiOptPy either)",
        (
            "mdmain",
            "geom_info",
        ): "atoms whose distances, angles, and dihedral angles are written at every step "
        "(MultiOptPy's mdmain replaces the value and writes nothing)",
        (
            "mdmain",
            "cmds",
        ): "map of the frames of the trajectory (MultiOptPy's mdmain reads it without using it)",
        (
            "mdmain",
            "pca",
        ): "principal components of the frames of the trajectory (MultiOptPy's mdmain reads it "
        "without using it)",
        ("mdmain", "TRAJECTORY"): "each trajectory in its own run directory, seeds one apart",
    }
)

# Notes of orientsearch (orientations of the parts of a complex).
NOTES.update(
    {
        (
            "orientsearch",
            "part",
        ): "every argument one part (1,2,3 or 1-3), the atoms no part names one more part; "
        "without -part the molecules of the bond graph (MultiOptPy stops without it)",
        (
            "orientsearch",
            "distance",
        ): "the distance between the centres of neighbouring parts of the multioptpy and uniform "
        "schemes, drawn from [d, 1.1 d] for every sample; the ring scheme (the default, "
        "measured, see searches.md) and the contact scheme place parts at van der Waals contact "
        "instead",
        (
            "orientsearch",
            "number_of_samples",
        ): "every sample is optimized by optmain in <stem>_Sample_<k>_OPT/ of the run directory",
    }
)

# Notes of confsearch (conformation search).
NOTES.update(
    {
        ("confsearch", "base_force"): "the collision energy of the AFIR force of a sample "
        "(-cs_opt method=afir, the default); it also names the run directory, as in MultiOptPy",
        ("confsearch", "max_samples"): "samples of -cs_opt method=afir; the last atom pair is "
        "used as well (MultiOptPy stops one pair early)",
        ("confsearch", "number_of_lowest"): "as in MultiOptPy: the search ends when the -nr "
        "lowest energies did not change for more than this many samples",
        ("confsearch", "number_of_rank"): "as in MultiOptPy",
        ("confsearch", "target_atoms"): "atoms of the pairs of the AFIR samples, and of nothing "
        "else",
        ("confsearch", "sampling_temperature"): "the temperature of the choice of the next "
        "minimum (ten times higher after every fifth sample, as in MultiOptPy); the "
        "populations of the ensemble use -temp",
        ("confsearch", "no_stochastic"): "as in MultiOptPy",
        ("confsearch", "preserve_bond_connectivity"): "the bonds are compared in the numbering "
        "of the input, as MultiOptPy does; without it mopt-rs keeps conformers whose bond "
        "graph is that of the input up to a renumbering of the atoms (MultiOptPy keeps every "
        "minimum without -pbc, -cs_opt mode=minima)",
        ("confsearch", "tabu_search"): "as in MultiOptPy, with the weights in logarithms so that "
        "large visit counts do not end the selection",
        ("confsearch", "tabu_alpha"): "as in MultiOptPy",
    }
)


def dest(o: dict[str, Any]) -> str:
    if "dest" in o:
        return str(o["dest"])
    names = [n for n in o["names"] if isinstance(n, str)]
    longs = [n for n in names if n.startswith("--")]
    if longs:
        return longs[0][2:].replace("-", "_")
    shorts = [n for n in names if n.startswith("-")]
    if shorts:
        return shorts[0][1:].replace("-", "_")
    return names[0]


def is_positional(o: dict[str, Any]) -> bool:
    return not any(isinstance(n, str) and n.startswith("-") for n in o["names"])


# Options whose arity and default mopt-rs changes, by destination (each exists in one
# command): the figures of nebmain are written by default, and -spng takes an optional switch
# word, so that `-spng off` turns them off while `-spng` alone keeps MultiOptPy's meaning (on).
ARITY_OVERRIDES = {"save_pict": ('ArgArity::Optional("true")', "true")}


def arity(o: dict[str, Any]) -> str:
    if dest(o) in ARITY_OVERRIDES:
        return ARITY_OVERRIDES[dest(o)][0]
    action = o.get("action")
    nargs = o.get("nargs")
    if action == "store_true":
        return "ArgArity::Flag"
    if nargs == "*":
        return "ArgArity::ZeroOrMore"
    if nargs == "+":
        return "ArgArity::OneOrMore"
    if nargs == "?":
        implicit = {"ModelhessAction": "fischerd3old", "CGAction": "PR"}
        a = action["expr"] if isinstance(action, dict) else ""
        return f'ArgArity::Optional("{implicit.get(a, "")}")'
    return "ArgArity::Single"


def default_text(o: dict[str, Any]) -> str:
    if dest(o) in ARITY_OVERRIDES:
        return ARITY_OVERRIDES[dest(o)][1]
    if o.get("action") == "store_true":
        return "false"
    d = o.get("default")
    if isinstance(d, dict):
        return d["expr"]
    if isinstance(d, list):
        return " ".join(str(x) for x in d)
    return "" if d is None else str(d)


def rust_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


# Help texts that mopt-rs changes for every command (by destination).
HELP_OVERRIDES = {
    "node_distance_bernstein": "the first chain on the Bezier curve of its images (Bernstein "
    "polynomials; the curve keeps the two ends and passes near the images between them): as "
    "many images as the distance (Angstrom) holds in the length of the chain. The distance of "
    "-nd is then applied (before it) only when -nd is given",
    "oniom_flag": "ONIOM: HIGH_ATOMS LINK LOW_CALCULATOR. The calculator of -os is the high "
    "level on the atoms HIGH_ATOMS (1-based, ranges as 1-5,8), LOW_CALCULATOR (any calculator "
    "name, such as gfnff or lj) the low level on the whole structure; LINK is the outside atoms "
    "whose bonds to the region get hydrogen link atoms (each with its nearest atom of the region, "
    "as MultiOptPy), auto (every bond across the border, 1.2 times the covalent radii), or none. "
    "Subtractive ONIOM, E_low(real) + E_high(model) - E_low(model), with mechanical embedding "
    "or electronic embedding (-oniom_opt embedding=electronic); -oniom_middle adds a middle "
    "layer, -oniom_opt sets the link atoms, the charges, and the embedding",
    "electronic_charge": "total (formal) charge; one value, or one per state where a command "
    "takes several structures",
    "spin_multiplicity": "spin multiplicity 2S + 1 for every calculator (MultiOptPy asked for 2S "
    "with PySCF; mopt-rs converts in the adapter); one value, or one per state",
    "othersoft": "calculator (default: pyscf): pyscf, psi4, tblite[:METHOD], dxtb[:METHOD], "
    "lj, emt, eam, meam, tersoff, muller_brown, an ASE profile (e.g. pm6_rs, mace_mp), "
    "ase:module:Class for any ASE calculator, or multioptpy:NAME through an installed MultiOptPy; "
    "optmain also takes an objective of plain variables (mb2d, expr:EXPRESSION) with a "
    "variables file as INPUT",
}


# Model Hessians of mopt-rs added to the examples in MultiOptPy's help of -modelhess.
MODEL_HESSIAN_EXAMPLES = ("gfn0xtb, fischer,", "gfn0xtb, gfn1xtb, fischer,")


def help_of(o: dict[str, Any]) -> str:
    """The help text of the option ``o`` in every table: MultiOptPy's, or mopt-rs's own where
    HELP_OVERRIDES has one, with the model Hessians of mopt-rs among the examples of
    -modelhess."""
    text = HELP_OVERRIDES.get(dest(o), str(o.get("help", "")))
    if dest(o) == "use_model_hessian":
        old, new = MODEL_HESSIAN_EXAMPLES
        if old not in text:
            raise SystemExit(f"the help of -modelhess no longer lists '{old}': {text}")
        text = text.replace(old, new)
    return text


def optmain_options(spec: dict[str, Any]) -> list[dict[str, Any]]:
    extra = [{**o, "origin": "mopt-rs", "line": 0} for o in EXTRA_OPTMAIN]
    return list(spec["commands"]["optmain"]["options"]) + extra


# Options of commands that MultiOptPy does not have.
EXTRA_COMMAND_OPTIONS = {
    "etmain": [
        {
            "names": ["-et_method", "--et_method"],
            "type": "str",
            "default": "edeel",
            "help": "edeel: the transfer within one molecule of donor, bridge, and acceptor by "
            "EDEEL's diabatic states, the crossings along the solvent coordinate of SAES, and "
            "the Marcus, Zhu-Nakamura, and Marcus-Levich-Jortner rates; hopping: the hop of a "
            "charge between two molecules (-et_donor, the molecule it starts on, and "
            "-et_acceptor), the transfer integral of the projection method at the input "
            "structure, the four-point reorganization energy from the equilibria of both "
            "molecules in both charges, and the Marcus rates both ways",
        },
        {
            "names": ["-et_carrier", "--et_carrier"],
            "type": "str",
            "default": "hole",
            "help": "the carrier of -et_method hopping: hole (between the highest occupied "
            "orbitals; the charged molecule one unit more positive) or electron (the lowest "
            "empty orbitals; one unit more negative)",
        },
        {
            "names": ["-et_donor", "--et_donor"],
            "type": "str",
            "default": None,
            "help": "the atoms of the donor (1-based; ranges such as 1-12,15): the species that "
            "holds the moving electron in the initial state",
        },
        {
            "names": ["-et_acceptor", "--et_acceptor"],
            "type": "str",
            "default": None,
            "help": "the atoms of the acceptor (as -et_donor)",
        },
        {
            "names": ["-et_bridge", "--et_bridge"],
            "type": "str",
            "default": "auto",
            "help": "auto: separate molecules when no bond joins the donor and the acceptor, "
            "else a molecule of donor, bridge (the other atoms), and acceptor, whose fragments "
            "(donor-bridge, bridge-acceptor) have their cut bonds capped by hydrogens; none; or "
            "the atoms of the bridge",
        },
        {
            "names": ["-et_charges", "--et_charges"],
            "nargs": 2,
            "type": "int",
            "default": [0, 0],
            "help": "the charges of the donor and the acceptor fragments without the moving "
            "electron (with their caps); the structure is the system with the moving electron. "
            "For -et_method hopping the charges of the two molecules without the carrier, and "
            "the structure is the pair in those charges",
        },
        {
            "names": ["-et_mult", "--et_mult"],
            "nargs": "*",
            "type": "int",
            "default": [],
            "help": "the multiplicities of the structure without the moving electron, the donor "
            "without and with it, and the acceptor without and with it (five values); for "
            "-et_method hopping those of the two molecules without and with the carrier (four "
            "values); the lowest ones when not given",
        },
        {
            "names": ["-et_lambda_s", "--et_lambda_s"],
            "nargs": "*",
            "type": "float",
            "default": [],
            "help": "the solvent reorganization energies of the initial and final states (eV) "
            "for the solvent activation energy shift (SAES); none: one crossing without solvent "
            "activation",
        },
        {
            "names": ["-et_alpha", "--et_alpha"],
            "type": "str",
            "default": "1,0,0.1",
            "help": "START,END,STEP of alpha_SA, the share of the solvent in the activation "
            "(each crossing starts from the one before)",
        },
        {
            "names": ["-et_coupling", "--et_coupling"],
            "type": "str",
            "default": "auto",
            "help": "the coupling at a crossing: mulliken_hush (the lowest excited state of the "
            "structure that moves at least 0.985 electrons), gmh (the generalized Mulliken-Hush "
            "of Cave and Newton with that state), fcd (the fragment charge difference of "
            "Voityuk and Rosch with that state, on the donor and the acceptor atoms), edeel (V11 "
            "less the energy of the structure with the moving electron), a value in eV, or auto "
            "(mulliken_hush when the calculator gives the transitions of its excited states, "
            "else edeel)",
        },
        {
            "names": ["-et_temperature", "--et_temperature"],
            "type": "float",
            "default": 298.15,
            "help": "temperature (K) of the rate constants",
        },
        {
            "names": ["-et_final", "--et_final"],
            "type": "str",
            "default": None,
            "help": "a structure (xyz) to start the equilibrium of the final state from (the "
            "input when not given)",
        },
        {
            "names": ["-et_free_energy", "--et_free_energy"],
            "type": "str",
            "default": "on",
            "help": "on: the free energies of activation as well, from the harmonic thermal "
            "corrections at the initial equilibrium and within the seam at each crossing "
            "(Hessians; -press, -freq_corr as for -freq), and the rate constants with them; "
            "off: the electronic barriers with the solvent shift only",
        },
        {
            "names": ["-et_crossing", "--et_crossing"],
            "type": "str",
            "default": "projection",
            "help": "how the crossings are searched: projection (gradient projection from the "
            "start), penalty (first the minimum of the seam model function (V11 + V22) / 2 + "
            "w (V11 - V22)^2 with w = 20/Hartree, then gradient projection from it), or "
            "penalty:W",
        },
        {
            "names": ["-et_states", "--et_states"],
            "type": "int",
            "default": 10,
            "help": "excited states searched for the charge-transfer state of Mulliken-Hush",
        },
    ],
    "nebmain": [
        {
            "names": ["-neb_opt", "--neb_options"],
            "nargs": "*",
            "default": [],
            "help": "further settings of the chain relaxation as KEY=VALUE: tolerance (end "
            "when the largest force of the images that move is below it, Hartree/Bohr; "
            "0.0003; 0: all -ns iterations are run, as MultiOptPy does and as form=multioptpy "
            "does without it), diis (chains of the last iterations that are extrapolated "
            "over, e.g. 5; off by default) and diis_start (largest force below which it "
            "begins, 0.005), sweep (on: the images get their quasi-Newton steps from the "
            "highest one down, and the step of the neighbour uphill enters the force of an "
            "image; off by default), global (on: one quasi-Newton step for all images at "
            "once, experimental; off by default) and global_memory (megabytes of Hessians "
            "of images held in memory during such a step, 256; the others are read from "
            "their files), failed (what is done with an image whose calculation fails: "
            "retry, the default: its step is taken again at half and a quarter of its length "
            "and the image is put back where it was calculated before, and it is taken out "
            "of the chain only if that fails; drop: it is taken out at once, as MultiOptPy "
            "does; stop: the run ends), gather (images gathered around the predicted maxima "
            "of -adpred, GATHER + 1 times as dense there; 1; 0: spread evenly), gather_width "
            "(width of the dense region in average distances of the images, 1), pairs (pairs "
            "of the Hessian updates of the images; 0: as many as the block updates keep, the "
            "default; 1: one, as MultiOptPy), projection (whole: the gradient along the "
            "one direction of the whole structure is removed, the default; atom: along the "
            "direction of every atom, as MultiOptPy does), tangent (higher: the direction "
            "of the chain at an image points to its higher neighbour, the default; top: it "
            "is carried outwards from the highest image by arcs and parabolas, as -qsmv2, "
            "while the profile has one maximum), form (consistent, the default; "
            "multioptpy: MultiOptPy's steps, alignment, and redistribution as they are), "
            "figures (on, the default; off), optimize (off, the default; on: optmain from "
            "the images of the relaxed chain, those at maxima to transition states, the ends "
            "and those at minima to minima; saddles, minima: one kind only), optimize_hessian "
            "(what such an optimization starts from: chain, the default: the Hessian the "
            "relaxation kept for the image; exact: the calculator's; none), "
            "optimize_settings (a settings file of optmain for these optimizations), "
            "optimize_check (which optimized structures get a vibrational analysis that "
            "counts their imaginary frequencies: saddles, the default; all; off), path "
            "(how the images between the structures that are read are made: geodesic, the "
            "default for structures; linear, MultiOptPy's way and the default with "
            "form=multioptpy; lst; idpp, also -idpp; sidpp; fbenm; cfbenm, also -cfbenm), "
            "gp (off, the default; on: between iterations that are calculated, a Gaussian "
            "process of the calculated points answers for up to gp_between iterations, 2, "
            "while its measured error is below gp_error, 0.1, of the largest force), "
            "gp_length (length scale of its kernel, Angstrom for structures, 0.4), "
            "gp_points (points used for one prediction, 12), gp_memory (points kept per "
            "image, 30), gp_reach (largest distance from a calculated point in length "
            "scales, 0.5), gp_uncertainty (largest posterior variance, 0.02), gp_features "
            "(what the model sees of a structure: points, the default; distances: its "
            "inverse interatomic distances, with gp_length in 1/Bohr, 0.1), spring_upper "
            "and spring_lower (the constants of the energy-weighted springs of -ewbneb at "
            "the highest image and at the energy of the higher end, Hartree/Bohr^2; 0.1 "
            "and 0.01); with -dmf: dmf_points (points that move towards the top of the "
            "energy profile, 3), dmf_guards (points between each end and those, 1), "
            "dmf_moves (on, the default; off: the points stay at equal shares of the "
            "length), dmf_beta (per eV, 10), dmf_segments (of the spline, 4), "
            "dmf_tolerance (largest component of the gradient of the objective, "
            "Hartree/Bohr, 7.8e-4), dmf_step (largest change of a coordinate of a control "
            "point in one iteration, Bohr, 0.3)",
        },
    ],
    "ieipmain": [
        {
            "names": ["-ieip_opt", "--ieip_options"],
            "nargs": "*",
            "default": [],
            "help": "further settings of the methods as KEY=VALUE: figures (on, the default; "
            "off), seed (of a random starting direction of -use_dimer and -use_spm without a "
            "second structure, 1), criteria (force thresholds of -use_dimer and -use_spm: "
            "normal, tight, loose), micro_tolerance (iEIP -mi: force in the coordinates of the "
            "relaxation at which it ends, 0.01), rotations (dimer rotations per translation, "
            "8), rotation_angle (degrees below which they stop, 1), step (largest translation "
            "of the dimer and largest step of the spring pair, Bohr, 0.1), spm_length (natural "
            "length of the spring, Bohr, 0.1), spm_offset (distance of the second particle at "
            "the start, Bohr, 0.1), spm_drift, spm_spring, spm_climb (rates of the drift, the "
            "spring, the climb; 0.5, 1, 5), spm_tolerance (largest force across the line that "
            "ends a drift, 2e-3), spm_drift_steps (200), addf_initial (Angstrom along the "
            "stiffest mode on the first sphere, 0.03), addf_angle (rad within which two ADDs "
            "are one, 0.01), addf_gradient (largest gradient component of the starting minimum, "
            "1e-3), sphere_iterations (evaluations of a minimization on a sphere of -addf and "
            "-2pshs, 30), sphere_gradient (RMS of the gradient within the sphere that ends it, "
            "1e-4), gnt_tolerance (RMS of the projected gradient that ends the corrector of "
            "-gnt, 1.7e-3), symmetry_noise (Bohr of a seeded random displacement of the second "
            "structure where it only gives a start or a direction, -use_dimer, -use_spm, -gnt, "
            "-2pshs, and of -dsafir, so that symmetric inputs can leave their symmetry; 0.05, "
            "0 for none); "
            "dimer_translation (lbfgs; experimental: plain, optimistic, extragradient), "
            "dimer_rate_across, dimer_rate_along (0.5 each); optimize (on: every estimate is "
            "optimized with optmain afterwards, a transition state with -order 1, a crossing "
            "of -dsafir between two states with -mf opt_meci or opt_mesx; off by default), "
            "optimize_hessian (exact, the default: a transition state search starts from the "
            "Hessian of the calculator; none), optimize_check (on: a vibrational or seam "
            "analysis of what is found; off), optimize_irc (on: from a crossing, a meta-IRC in "
            "the state of each structure, whose end is compared with it; off), "
            "optimize_settings (a settings file of optmain); the keys of -gsm, -se_gsm, -fsm, "
            "-dsafir, and -lmm with the prefixes gsm_, fsm_, dsafir_, lmm_ "
            "(the keys of the methods in the registry of two-point methods, see reaction_paths.md); "
            "method_relax (the relaxation of the path of the method afterwards by a chain of "
            "nebmain whose maxima become the estimates: lup, neb, or off, the default; every "
            "method with a path), method_relax_iterations (5), method_relax_nodes (0: by the "
            "spacing), method_relax_spacing (0.5 Angstrom), method_relax_max_step (Bohr), "
            "method_relax_options (further options of nebmain); dsafir_lup=N with dsafir_lup_nodes, "
            "dsafir_lup_spacing, dsafir_lup_max_step name the same step with the lup chain",
        },
        {
            "names": ["-gsm", "--use_gsm"],
            "action": "store_true",
            "help": "the double-ended growing string method between the two structures "
            "(Peters et al. 2004; Zimmerman 2013); new in mopt-rs",
        },
        {
            "names": ["-se_gsm", "--use_se_gsm"],
            "nargs": "+",
            "default": None,
            "help": "the single-ended growing string method from the first structure along the "
            "driving coordinates given, each as one argument: 'ADD i j', 'BREAK i j', "
            "'ANGLE i j k degrees', 'TORSION i j k l degrees' (atoms from 1), or 'VAR k value' "
            "for plain variables (Zimmerman 2015); new in mopt-rs",
        },
        {
            "names": ["-fsm", "--use_fsm"],
            "action": "store_true",
            "help": "the freezing string method between the two structures (Behn et al. "
            "2011); new in mopt-rs",
        },
        {
            "names": ["-lmm", "--use_lmm"],
            "action": "store_true",
            "help": "the local minimax method from the first structure towards the second "
            "(Li, Zhou 2001); experimental, new in mopt-rs",
        },
        {
            "names": ["-dsafir", "--use_dsafir"],
            "action": "store_true",
            "help": "the double-sphere artificial force induced reaction method between the "
            "two structures (Maeda et al. 2018); with the structures in two states (-elec, "
            "-spin, -es with one value each) every end moves on the surface of its own state "
            "and the point where they meet estimates a crossing; new in mopt-rs",
        },
    ],
    "ircmain": [
        {
            "names": ["-irc_step", "--irc_step_size"],
            "type": "float",
            "default": 0.1,
            "help": "step size in mass-weighted coordinates (amu^0.5 Bohr; the unit of the "
            "variables for an objective of plain variables); -irc STEP_SIZE MAX_STEP METHOD "
            "sets the three together as in optmain",
        },
        {
            "names": ["-irc_max", "--irc_max_step"],
            "type": "int",
            "default": 300,
            "help": "MAX_STEP: the largest number of steps of each side",
        },
        {
            "names": ["-irc_method", "--irc_method"],
            "type": "str",
            "default": "lqa",
            "help": "integrator: lqa, hpc, dvv, euler, rk4, gs2, or modekill[:MODES]; options "
            "of a method follow after colons (dvv:v0=0.04:dt0=0.5:tol=0.003, "
            "modekill:1,2:nu_thresh=-5)",
        },
        {
            "names": ["-path_analysis", "--path_analysis"],
            "action": "store_true",
            "default": False,
            "help": "after the IRC of a transition state: the analysis of freqmain -path for "
            "the combined path (vibrations orthogonal to the path, curvature and couplings, "
            "reaction phases, variational transition state, tunneling), in path_analysis/ of "
            "the run directory; the freqmain options of that analysis are not taken here "
            "(run freqmain irc_combined_path.xyz -path for them)",
        },
    ],
    "mdmain": [
        {
            "names": ["-md_opt", "--md_options"],
            "nargs": "*",
            "default": [],
            "help": "further settings of the dynamics as KEY=VALUE: thermostat (none, berendsen, "
            "bussi, langevin, andersen, nose_hoover, nose_hoover_chain; replaces the one of -mt), "
            "integrator (velocity_verlet, baoab), barostat (isotropic: berendsen, crescale, mtk; "
            "anisotropic, every length and angle of the cell following its component of the "
            "pressure tensor: berendsen_aniso, crescale_aniso, mtk_aniso; constant pressure -press "
            "in a periodic cell), tau (coupling time of the "
            "thermostat in fs, 100), friction (Langevin, 1/ps; 1000/tau), collision (Andersen, "
            "1/ps per particle; 1000/tau), chain (links of a Nose-Hoover chain, 3), yoshida (1, "
            "3, 5), substeps (1), tau_p (coupling time of the barostat in fs, 1000), "
            "compressibility (1/GPa, assumed by berendsen and crescale, 0.01), seed (of the "
            "random numbers; without it one is drawn and written to the settings of the run), "
            "velocities (exact: Maxwell-Boltzmann scaled to -temp, the default; maxwell: as "
            "drawn; zero), rigid (auto: the motions of the whole system that the surface "
            "conserves are taken out of the velocities and the degrees of freedom; none, "
            "translation, full), write_every (steps between frames of the trajectory, 10), "
            "print_every (steps between reports in the log, 1), structure_every (steps between "
            "the states in the log: the structure in MultiOptPy's layout, the velocities, and the "
            "energies; 10, written to the settings of the run), freeze (modes of the Hessian the "
            "dynamics does not move along: negative, below:EIGENVALUE in mass-weighted atomic "
            "units, modes:0,1 counted from the lowest, file:PATH with one Cartesian direction "
            "per line), freeze_update (steps between new modes, followed by overlap; none), "
            "shake_tolerance (1e-10), wrap (on: molecules of a periodic cell are written inside "
            "it; off), integrate (with -coord: cartesian, the default, or coordinates: the "
            "dynamics in the coordinates of -coord, refused where they are redundant or the run "
            "has constraints, frozen modes, a barostat, a cell, or bias potentials on internal "
            "coordinates), stress (on: the stress at every step, written as the pressure tensor "
            "<stem>_pressure_tensor.csv for the shear viscosity), heat_flux (on: the heat current "
            "of every step, <stem>_heat_flux.csv, for the thermal conductivity; lj with a cutoff, "
            "eam, emt, tersoff), exchange (steps between the velocity exchanges of Muller-Plathe's "
            "reverse non-equilibrium method, NVE in a cell), exchange_slabs (20), exchange_axis "
            "(c), constraint_solver (shake, the default, or lincs for bond lengths), lincs_order "
            "(4), lincs_iterations (1), velocities=wigner (the start from the harmonic Wigner "
            "distribution of the normal modes of the input structure) with wigner_temperature "
            "(K; -temp, 0 for the ground state), wigner_low (CUTOFF:classical|zero|wigner for "
            "modes below CUTOFF cm^-1), wigner_imaginary (boltzmann, forward, backward, zero), "
            "hopping_states (surface hopping by the Zhu-Nakamura theory among these electronic "
            "states, e.g. 1,2, from the state of -es; off without it) with hopping_gap (eV: the "
            "gradients of the neighbouring states are computed and hops considered below this "
            "gap, 1), hopping_frustrated (keep, the default, or reverse: the velocity along the "
            "hopping direction of a frustrated hop), hopping_forces (two_state, the default, or "
            "published: the diabatic forces of Yu et al. 2014 as printed)",
        },
        {
            "names": ["-coord", "--coordinate_system"],
            "type": "str",
            "default": "cart",
            "help": "coordinate system of the run, any name of -coord of optmain: the values of "
            "the coordinates at the frames go to <stem>_coordinates.csv, and the dynamics runs "
            "in Cartesian coordinates; with -md_opt integrate=coordinates a non-redundant "
            "system (dic, tric, or a registered one that covers all motions) is integrated in "
            "its coordinates with the kinetic energy 1/2 p^T G p (G = B M^-1 B^T) by the "
            "generalized leapfrog (redundant ones, with a singular G at the start such as ric "
            "and tric-p of a molecule with more primitives than motions, are refused)",
        },
        {
            "names": ["-umbrella", "--umbrella_sampling"],
            "nargs": "*",
            "default": [],
            "help": "umbrella sampling as KEY=VALUE: cv (the collective variable in the language "
            "of -ubias, atomic units, e.g. dist(1,2)), centers (A,B,...) or from, to, windows, k "
            "(force constant, Hartree per unit squared), equilibration (steps of every window "
            "that are not sampled; a fifth of -time), bins (50), period (of an angle-like "
            "variable, e.g. 2*pi); every window runs -time steps from the end of the one before, "
            "and the free-energy profile comes from WHAM and MBAR (umbrella_profile.csv)",
        },
        {
            "names": ["-analysis", "--trajectory_analysis"],
            "nargs": "*",
            "default": [],
            "help": "analyze the trajectory when the run ends, into trajectory_analysis/ of the "
            "run directory: names statistics (means with block-average errors of the energies, "
            "temperature, pressure, volume), rdf (radial distribution functions of the pairs of "
            "elements), msd (mean squared displacements, diffusion coefficients), vacf "
            "(velocity autocorrelation, vibrational density of states), rmsd (RMSD to the first "
            "frame, RMSF of the atoms), geometry (distances, angles, dihedral angles of atoms=), "
            "pca (principal components of the motion), acf (autocorrelations and power spectra "
            "of the properties and measures, orientations of bonds=, velocities of every atom), "
            "ir (infrared spectrum of the dipoles of the atomic charges), transport (diffusion "
            "by Einstein and Green-Kubo, thermal conductivity by Green-Kubo and reverse "
            "non-equilibrium, shear viscosity by Green-Kubo), all (the default); settings as "
            "KEY=VALUE: skip (fs left out at the start, 0), interval (fs between frames without "
            "times, 1), rdf_bin (angstrom, 0.02), rdf_max (angstrom; half the smallest width of "
            "the cell), msd_fit (shares of the longest lag of the fit, 0.1,0.9), drift (subtract "
            "the drift of the centre of mass, true), vacf_share (share of the run the "
            "autocorrelations reach, 0.5), vdos_max (cm^-1; the Nyquist wavenumber), window "
            "(hann, blackman, none), padding (zero padding of the transforms, 4), qcf (quantum "
            "correction of the infrared spectrum: harmonic, standard, schofield, none), gk_lag "
            "(fs, longest lag of the Green-Kubo integrals; half a block), gk_blocks (10), bonds "
            "(1-based pairs: 1,2;3,4), exchange_slabs, exchange_axis (of a reverse "
            "non-equilibrium run analyzed afterwards; 20, c), atoms (1-based: 1,2;1,2,3;1,2,3,4), "
            "components (3), figures (png, svg)",
        },
        {
            "names": ["-cv", "--collective_variables"],
            "nargs": "*",
            "default": [],
            "help": "collective variables written at every step to <stem>_cv.csv (atomic units) "
            "and analyzed with -analysis (statistics, autocorrelations): distance:1,2, "
            "distance:1,2,3/4,5 (centroids), angle:1,2,3, dihedral:1,2,3,4, "
            "coordination:1,2/3,4[:r0=ANG][:n=6][:m=12], path_s:FILE[:lambda=PER_ANG2], "
            "path_z:FILE, or an expression of -ubias (dist(1,2) - dist(2,3)); the same forms "
            "serve -cv_bias and -umbrella cv=",
        },
        {
            "names": ["-cv_bias", "--cv_bias"],
            "nargs": "*",
            "default": [],
            "help": "harmonic restraints k/2 (xi - center)^2 of collective variables, as "
            '"distance:1,2;center=3.0;k=0.1" (the centre in atomic units of the variable, k '
            "in Hartree per unit squared)",
        },
        {
            "names": ["-analyze_only", "--analyze_only"],
            "action": "store_true",
            "default": False,
            "help": "the inputs are run directories of mdmain or trajectory files (XYZ, with "
            "<stem>_velocities.xyz and md_properties.csv next to them when there are): analyze "
            "them with the settings of -analysis (all analyses without it) and run no dynamics",
        },
    ],
    "confsearch": [
        {
            "names": ["-cs_opt", "--confsearch_options"],
            "nargs": "*",
            "default": [],
            "help": "settings of the search as KEY=VALUE: method (afir, MultiOptPy's sampling and "
            "the default; imtd, the iMTD-GC workflow: RMSD metadynamics, loose and final "
            "optimizations, crossing of conformers, dynamics from the lowest ones), mode "
            "(conformers, the default: minima with the bond graph of the optimized input; "
            "minima: every distinct minimum), graph (isomorphic, the default; numbering, as -pbc), "
            "bond_scale (1.1), seed (without it one is drawn and written to the settings of the "
            "run), priority (the choice of the next minimum of method=afir: boltzmann_heated, "
            "MultiOptPy's and the default, tabu_heated with -tabu, boltzmann, tabu, "
            "water_filling, water_filling_adaptive, or a registered rule), pairs (the AFIR pairs: "
            "bonds[:N], pairs at least N bonds apart (the farthest ones in a compact cluster), 3, "
            "the default; multioptpy, MultiOptPy's window as it computes it; ratio, the same "
            "window in consistent units), ethr (kcal/mol, "
            "0.05), rthr (Angstrom, 0.125), bthr (relative difference of the rotational "
            "constants, 0.01), mirror (on: mirror images are one conformer), permutations (on), "
            "compare_atoms (atoms compared, e.g. a solute: 1-12), loose_window (kcal/mol, 12), "
            "skip_known (on), nudge (on: every converged conformer is displaced by 0.02 Bohr and "
            "optimized again, which leaves saddle points of symmetry), verify (on: the Hessian of "
            "every new conformer is checked, and from a saddle point the search goes on along "
            "its imaginary mode to a minimum), mtd_runs (crest, the 14 runs of CREST's documentation; or "
            "K_MEH_PER_ATOM:ALPHA,...), mtd_time (ps, 5), time_step (fs, 2), mtd_temperature "
            "(K, 300), dump (fs between snapshots, 100), vbias (fs between deposited "
            "references, 1000), known_references (20), shake (hydrogens, the default; all; "
            "off), hmass (amu of hydrogen in the dynamics, 4; 0: its mass), rounds (5), "
            "bias_atoms, crossover (on), crossover_window (kcal/mol, 6), crossover_max (300), "
            "finish (on), finish_conformers (4), finish_temperatures (400,500), finish_time "
            "(ps, 3); with -tsconf: ts_hold (coordinate, the default; modes), ts_update (steps "
            "between the Hessians of ts_hold=modes, 10), ts_window (kcal/mol, 6), ts_max (20), "
            "ts_irc (on), ts_share (0.1), ts_bonds (on: the reactive bonds held as well), "
            "ts_bond_scale (1.5)",
        },
        {
            "names": ["-tsconf", "--ts_conformers"],
            "action": "store_true",
            "default": False,
            "help": "the input is a transition state: search its conformers with the saddle "
            "direction held, optimize them to saddle points of order 1, keep those with one "
            "imaginary frequency and the same reaction (the ends of their IRC), and report the "
            "lowest (ts_conformers.xyz, ts_conformers.csv)",
        },
    ],
    "freqmain": [
        {
            "names": ["-path", "--path"],
            "action": "store_true",
            "default": False,
            "help": "the frames of INPUT are the points of a reaction path (in a common "
            "orientation): the vibrations orthogonal to the path at every point and where the "
            "lowest turns unstable (valley-ridge transitions, at which a path bifurcates), "
            "the curvature of the path and its couplings to the modes (reaction-path "
            "Hamiltonian), their components along internal coordinates and the reaction "
            "phases (unified reaction valley approach), the free-energy profile with the "
            "variational transition state at -temp, and tunneling (ZCT, SCT)",
        },
        {
            "names": ["-temp_scan", "--temperature_scan"],
            "type": "str",
            "default": None,
            "help": "START,END,STEP (K): the thermodynamic functions at these temperatures "
            "as well, in thermochemistry_scan.csv",
        },
        {
            "names": ["-spc_energy", "--single_point_energy"],
            "nargs": "*",
            "type": "float",
            "default": [],
            "help": "electronic energies (Hartree) from another level of theory, one per "
            "input, that replace those of the calculator in the thermochemistry",
        },
        {
            "names": ["-boltz", "--boltzmann"],
            "action": "store_true",
            "default": False,
            "help": "with several inputs: their relative Gibbs free energies and Boltzmann "
            "populations at -temp, in boltzmann_<time>.csv",
        },
        {
            "names": ["-anharmonic", "--anharmonic"],
            "action": "store_true",
            "default": False,
            "help": "anharmonic analysis of a minimum: cubic and quartic force constants "
            "from the Hessians at structures displaced along every normal mode (two per "
            "vibration), fundamentals and zero-point energy by second-order perturbation "
            "theory (VPT2) and by the vibrational self-consistent field (VSCF)",
        },
        {
            "names": ["-qff_step", "--qff_step"],
            "type": "float",
            "default": 0.05,
            "help": "-anharmonic: displacement along every normal mode in its dimensionless "
            "normal coordinate",
        },
        {
            "names": ["-vscf_basis", "--vscf_basis"],
            "type": "int",
            "default": 16,
            "help": "-anharmonic: harmonic oscillator functions per mode in VSCF",
        },
        {
            "names": ["-fermi_gap", "--fermi_gap"],
            "type": "float",
            "default": 200.0,
            "help": "-anharmonic: largest frequency gap (cm^-1) of a Fermi resonance",
        },
        {
            "names": ["-fermi_threshold", "--fermi_threshold"],
            "type": "float",
            "default": 1.0,
            "help": "-anharmonic: a term is a Fermi resonance when its correction beyond "
            "perturbation theory exceeds this (cm^-1)",
        },
        {
            "names": ["-phonon_supercell", "--phonon_supercell"],
            "nargs": "*",
            "type": "int",
            "default": [],
            "help": "N1 N2 N3: phonons of a periodic structure from the force constants of "
            "the supercell of N1 x N2 x N3 cells (its Hessian is computed); with "
            "-phonon_path the dispersion, with -phonon_mesh the density of states and the "
            "thermodynamic functions of the crystal",
        },
        {
            "names": ["-phonon_path", "--phonon_path"],
            "nargs": "*",
            "type": "str",
            "default": [],
            "help": "wave vectors in units of the reciprocal lattice vectors, e.g. 0,0,0 "
            "0.5,0,0 0.5,0.5,0: the phonon dispersion along the lines between them "
            "(phonon_dispersion.csv)",
        },
        {
            "names": ["-phonon_points", "--phonon_points"],
            "type": "int",
            "default": 40,
            "help": "-phonon_path: steps along every line",
        },
        {
            "names": ["-phonon_mesh", "--phonon_mesh"],
            "nargs": "*",
            "type": "int",
            "default": [],
            "help": "M1 M2 M3: mesh of wave vectors for the density of states "
            "(phonon_dos.csv) and the thermodynamic functions per cell "
            "(phonon_thermochemistry.txt)",
        },
        {
            "names": ["-phonon_dos_width", "--phonon_dos_width"],
            "type": "float",
            "default": 5.0,
            "help": "-phonon_mesh: width (cm^-1) of the Gaussian of every mode in the "
            "density of states",
        },
        {
            "names": ["-local_modes", "--local_modes"],
            "nargs": "*",
            "type": "str",
            "default": [],
            "help": "local vibrational modes (Konkoli-Cremer) of internal coordinates: the "
            "relaxed force constant and the frequency of each coordinate decoupled from the "
            "rest. Coordinates by atom numbers starting at 1: 1-2 (distance), 1-2-3 (angle), "
            "1-2-3-4 (dihedral); bonds (all bonds), auto (all bonds, angles, and dihedrals)",
        },
        {
            "names": ["-partial_atoms", "--partial_atoms"],
            "type": "str",
            "default": None,
            "help": "also analyse the vibrations of these atoms alone (numbers starting at 1, "
            "e.g. 1,4,7-9) from their part of the Hessian; see -partial_hessian",
        },
        {
            "names": ["-partial_hessian", "--partial_hessian"],
            "type": "str",
            "default": "block",
            "help": "-partial_atoms: block (the block of the chosen atoms, their translations "
            "and rotations projected out), effective (the other atoms relaxed: the Schur "
            "complement, then the same projection; for two atoms the relaxed force constant "
            "of the local stretching mode), or frozen (the block in the fixed frame of the "
            "other atoms, no projection)",
        },
        {
            "names": ["-partial_free", "--partial_free_direction"],
            "type": "float",
            "default": 1e-10,
            "help": "-partial_hessian effective: besides their free rotations about the "
            "chosen atoms, directions of the other atoms whose curvature is below this "
            "fraction of their largest one are left out (exact zeros)",
        },
        {
            "names": ["-path_tangent", "--path_tangent"],
            "type": "str",
            "default": "gradient",
            "help": "-path: direction of the path at its points: gradient (a steepest-descent "
            "path such as an intrinsic reaction coordinate; the points are used where the "
            "gradient vanishes) or points (the line through the neighbouring points, for a "
            "scan or a band of images)",
        },
        {
            "names": ["-freq_masses", "--freq_masses"],
            "type": "str",
            "default": "standard",
            "help": "masses of the normal modes, the thermochemistry and -path: standard (the "
            "standard atomic weights) or isotope (the most abundant isotopes, as the IRC "
            "option masses=isotope and MultiOptPy); named in thermochemistry.txt and the log. "
            "Not available with -anharmonic, -local_modes, -partial_atoms, -phonon_supercell "
            "and -mf, which weight with the standard atomic weights",
        },
        {
            "names": ["-nro", "--natural_reaction_orbitals"],
            "action": "store_true",
            "default": False,
            "help": "natural reaction orbitals along the frames of the input (a path): at "
            "every frame the pairs of an occupied and an empty orbital through which the "
            "electrons respond to the displacement along the path, and their singular values "
            "(Ebisawa et al., Phys. Chem. Chem. Phys. 2022, 24, 3532); needs a calculator that "
            "hands out orbitals (gfn0xtb, gfn1xtb, pyscf, tblite)",
        },
        {
            "names": ["-roet", "--reactive_orbital_energies"],
            "action": "store_true",
            "default": False,
            "help": "orbital energies along the frames of the input (a path), each orbital "
            "followed from frame to frame, their derivatives, and the orbitals whose energies "
            "change most (reactive orbital energy theory)",
        },
        {
            "names": ["-diabatize", "--diabatize"],
            "nargs": "*",
            "type": "int",
            "default": [],
            "help": "N M ...: two or more excited states (numbers at the first frame, 1: the "
            "lowest) along the frames of the input (a path) as diabatic states: the diabatic "
            "potential matrix (their energies and couplings), where two of them cross, and "
            "the Landau-Zener probability of a passage at -temp, from the overlaps of the "
            "states of neighbouring frames (propagative block-diagonalisation "
            "diabatisation); for calculators that hand out their excited states (gfn0xtb, "
            "pyscf)",
        },
        {
            "names": ["-orbital_step", "--orbital_step"],
            "type": "float",
            "default": 0.01,
            "help": "-nro: displacement along the path for the differences of the orbitals, "
            "in amu^0.5 Bohr",
        },
        {
            "names": ["-phase_prominence", "--phase_prominence"],
            "type": "float",
            "default": 0.0,
            "help": "-path: a minimum of the path curvature divides two reaction phases when "
            "the curvature rises on both sides by more than this fraction of its largest "
            "value (0: every minimum)",
        },
    ],
}


EXTRA_COMMAND_OPTIONS["orientsearch"] = [
    {
        "names": ["-orient_opt", "--orient_options"],
        "nargs": "*",
        "default": [],
        "help": "further settings of the orientation search as KEY=VALUE: scheme (ring, the "
        "default: uniform rotations, the centres on a regular polygon in a random plane shrunk "
        "to van der Waals contact; contact: every part brought from a random direction to "
        "contact with those placed before, also what ring does in a cell or with parts that "
        "stay; uniform: uniform rotations, centres -dist apart in random directions; "
        "multioptpy: MultiOptPy's Euler angles and polygon), seed "
        "(of the random numbers; without it one is drawn and written to the settings of the "
        "run), contact (LOW,HIGH: range of the ratio of the closest distance between two parts "
        "to the sum of their van der Waals radii, 1.0,1.2), clash (samples of multioptpy and "
        "uniform closer than this ratio are drawn again, 0.5; 0: never), attempts (draws per "
        "sample, 100), refine (share of the samples after the first minimum that start from a "
        "known minimum with one part turned and shifted, 0), priority (the rule that chooses "
        "that minimum: tabu, boltzmann, water_filling), angle (largest turn of such a part, "
        "degrees, 90), shift (largest shift, Angstrom, 1.0), rmsd (two minima with the same "
        "bonds are one when the RMSD after matching equivalent atoms and identical molecules "
        "is below this, Angstrom, 0.25), energy (and their energies differ by less than this, "
        "Hartree, 1e-4), mirror (true: a mirror image is the same minimum), bonds (bond "
        "scale of the comparison, 1.2), fixed (parts, from 1, that keep their place; parts "
        "bonded to their periodic images always do), interaction (vertical, the default: "
        "energies of the parts alone at their geometries in every minimum; relaxed: against "
        "the parts optimized alone; both; off), part_charges, part_multiplicities (of the "
        "parts alone, e.g. 0,0 and 1,2; without them neutral parts with the lowest "
        "multiplicity, which needs a neutral complex), criteria (convergence of the samples "
        "when neither -tcc nor -lcc is given: tight, the default, measured; normal: the default "
        "of optmain, as MultiOptPy; loose)",
    },
]

# Options of relaxedscan that MultiOptPy does not have.
EXTRA_COMMAND_OPTIONS["relaxedscan"] = [
    {
        "names": ["-grid", "--scan_grid"],
        "action": "store_true",
        "default": False,
        "help": "scan every combination of the values of the coordinates (a map of two "
        "coordinates, N1 x N2 points with -nsample N1,N2) instead of moving them together as "
        "MultiOptPy does",
    },
    {
        "names": ["-rigid", "--rigid_scan"],
        "action": "store_true",
        "default": False,
        "help": "single points without relaxation: the coordinates are set by moving the atoms "
        "on one side of each rigidly (the shortest displacement where the bonds do not divide "
        "the structure); ReSPer coordinates by a structure interpolated between the references",
    },
    {
        "names": ["-resper", "--resper_references"],
        "type": "str",
        "default": None,
        "help": "file of the reference structures of -scan resper (a path or a trajectory: "
        "xyz frames such as <stem>_traj.xyz of optmain -cmds, the path of nebmain, an IRC, an "
        "MD trajectory; for plain variables a CSV with a column per variable or numbers per "
        "line); their map by classical multidimensional scaling gives the coordinates PCo1 "
        "and PCo2 (Angstrom amu^0.5)",
    },
    {
        "names": ["-scan_opt", "--scan_options"],
        "nargs": "*",
        "default": [],
        "help": "further settings of the scan as KEY=VALUE: order (snake, the default: a map is "
        "taken in boustrophedon order, every point starting from the optimized point before "
        "it; raster), repair (on, the default for relaxed maps: neighbouring points whose "
        "structures differ by more than jump_factor, 3, times the median are on different "
        "branches, and the higher one is optimized again from the lower one; off), candidates "
        "(on, the default: transition-state candidates and minima of the scanned surface; off), "
        "ts_optimize (off, the default; on: optmain -order 1 from every transition-state "
        "candidate, with a vibrational analysis), hessian (continue, the default: a relaxed "
        "point starts from the last Hessian of the point it starts from, where -fc, -modelhess, "
        "-ih, and -mf give none (with ONIOM microiterations the Hessian of the core); off: from the start of "
        "optmain), figures (on; off), html (on; "
        "off), "
        "resper_atoms (the atoms of the distances of the ReSPer map; all, with ONIOM those of the core), resper_mass (on; "
        "off: unit masses), resper_stride (every Nth reference frame), resper_max (most "
        "references, 200), resper_curvature (off, the default; on: second derivatives of the "
        "ReSPer coordinates for the constrained optimizer), resper_correct (on, the default: "
        "a rigid scan moves the interpolated structure onto the grid point; off), "
        "resper_outside (mark, the default; skip: grid points beyond the references are not "
        "calculated), resper_outside_factor (2: beyond means further from the nearest "
        "reference than this many usual spacings of the references), resper_start (preimage, "
        "the default: a relaxed point of ReSPer coordinates starts from the structure "
        "interpolated between the references; neighbour: from its optimized neighbour)",
    },
]

# Options of run_mapper that MultiOptPy does not have.
EXTRA_COMMAND_OPTIONS["run_mapper"] = [
    {
        "names": ["-map_opt", "--mapper_options"],
        "nargs": "*",
        "default": [],
        "help": "further settings of the mapper as KEY=VALUE (the same keys may stand in "
        "mapper_settings of the settings file): method (the way a minimum is explored: afir_ts, "
        "the default, AFIR pushes between pairs of atoms; addf, the anharmonic downward "
        "distortion following from the minimum; nanoreactor, molecular dynamics in the "
        "nanoreactor walls and a two-point method between the minima of every reaction it "
        "shows; connect, the missing connections between the starts and the goal: the gaps of "
        "the path of least weight D^2 between the known minima, each filled by a two-point "
        "method; or a registered one), priority (the choice of the next minimum: boltzmann, "
        "the default, at temperature_K; tabu, water_filling, water_filling_adaptive, "
        "boltzmann_heated, tabu_heated; rrt, the default with goal, towards the goal: fair "
        "random choices among the groups of minima of one connection pattern alternating with "
        "choices by the similarity of the local atomic environments to the goal; or a "
        "registered one), goal (FILE: the goal structure, the first frame of an xyz file or a "
        "variables file, optimized like the inputs and never explored), goal_stop (true, the "
        "default: the exploration ends when the goal is joined to a start; false), goal_match "
        "(structure, the default: the goal structure itself; bonds: any minimum with its bond "
        "graph), "
        "accept_temperature (K: a new minimum is explored only with the probability min(1, "
        "exp(-(E_PT - E_X) / kT)), E_PT the highest transition state on the way of its job "
        "from the minimum X; 7500 by default with priority=rrt, off otherwise), "
        "rrt_goal_weight (c of the choices by similarity, 1), rrt_locality (per Angstrom, "
        "20), higher_saddles (false, the default: saddle points of order 2 or more that a "
        "search or an IRC meets are kept as nodes 2SP, 3SP, ... without following their links, "
        "and a search that ends at one goes on by mode kill to a transition state; true: they "
        "are optimized at their order and descended from along each unstable mode; "
        "experimental), max_saddle_order (2), upward (false; "
        "true: upward searches by high-index saddle dynamics from the transition states of every "
        "minimum to saddle points of order 2), upward_modes (2), confirm_bifurcations (false; "
        "true: a possible bifurcation of an IRC is tried both ways across the path and "
        "minimized), irc_hessian_interval (10: the points between the exact Hessians that "
        "the IRC computes on its own where the Hessian is analytic, irc_hessians; the lowest "
        "curvature across the path at each of them is kept, and a possible bifurcation says "
        "at which arc length that curvature changes sign), "
        "confirm_dissociations (false; true: a relaxed scan of the distance between the two "
        "molecules of every complex towards its dissociation channel, in steps of "
        "dissociation_scan_step, 0.3 Angstrom; a maximum more than "
        "dissociation_scan_tolerance, 0.5 kcal/mol, above the channel is a barrier), "
        "path_peaks (failed, the default: the highest point of a path that gave no "
        "transition state is saved with the minima on its two sides, as a record beside the "
        "network; all: that of every path; off), ts_conformers (false; true: the conformers "
        "of every transition state are searched with its saddle direction held, the search "
        "of confsearch -tsconf with the settings of the block ts_conformer_settings of the "
        "settings file, and every saddle point found is followed like any guess), "
        "entrance_structures (0; N: the molecules of the first input are an entrance "
        "channel, and N structures of it, the molecules turned and placed in contact at "
        "random, are further inputs of the run), entrance_contact (1.0: the nearest atoms of "
        "two molecules at this times the sum of their van der Waals radii), max_bond_changes "
        "and max_species_changes (no limit by default; N: a new minimum is not explored "
        "when its bonds, or its chemical species by the SMILES string, changed more than N "
        "times on the way from its input; 0: only what keeps the bonds, or the species, of "
        "the inputs is explored), method_step1 (afir_ts: the method of the first path of a pair of atoms; "
        "afir, the default, the AFIR push; se_gsm, the single-ended growing string driven by "
        "the pair, or gnt, the Newton trajectory that shortens it, grown from the minimum "
        "without a push; method_step1_KEY its settings; neb and dmf are methods of "
        "method_relax, and a method of two ends runs with method=connect and a goal), stop "
        "(further stop rules, ';' "
        "between them: nothing_new:N, rank_unchanged:R,N, max_jobs:N), store (further stores "
        "of the network, ',' between them, besides the files of the run directory: sqlite, the "
        "network as the SQLite database reaction_network.sqlite, or a registered one), "
        "save_job_steps (true by default: save stage trajectories, each NEB iteration and live job progress), "
        "html_report (true, the default; false), mirror (false, the default: mirror images are "
        "different minima; true), verify_minima (true, the default: every minimum is verified "
        "by its Hessian, mode kill is opt-in, and its "
        "wavenumbers are kept; false: converged structures are taken unchecked), irc_hessians "
        "(true, the default: the IRC computes the exact Hessian at "
        "every tenth point unless calc_exact_hess of step4_settings says otherwise, and a "
        "negative curvature across the path there is recorded as a possible bifurcation; false: "
        "only where calc_exact_hess asks), irc_end_threshold_cm1 (minimum magnitude of an "
        "imaginary vibration at an optimized IRC endpoint for a TS–TS link, cm^-1, 50), "
        "bifurcation_threshold_cm1 (how negative that "
        "curvature must be, cm^-1, 50), variable_tolerance (on an objective of plain variables, "
        "othersoft mb2d and the like: the largest distance of two points that are the same, in "
        "the units of the variables, 0.01; the method is then addf by default), channels (true, "
        "the default: an end of an IRC whose molecules are apart is a dissociation channel DC, "
        "its fragments optimized on their own once, and a minimum of two molecules in contact "
        "dissociates into its channel without a barrier; false: none), channel_margin "
        "(Angstrom over the sums of the van der Waals radii for molecules in contact, 1.0), "
        "method_KEY (a "
        "setting of the method: for afir_ts relax (neb, the default: the AFIR path is relaxed "
        "without the force by the chain relaxation of nebmain, with Hessian-based steps from a "
        "model Hessian and step2_settings as its settings, and the guesses are the maxima of the "
        "relaxed chain; dmf; off: the maxima of the AFIR path itself) and relax_KEY (a setting "
        "of that method); for addf the keys of "
        "addf of the two-point methods, such as method_paths=8, and dissociation (bond scale "
        "of the test for a dissociation channel, 2.0) and dissociation_atoms; for connect "
        "path (the two-point method of a gap, neb), path_KEY, and attempts (attempts of one "
        "pair, 2; attempt k runs with 2^(k-1) times the images); for nanoreactor "
        "runs, steps, time_step, temperature, frame_every, r1, r2, k1, k2, period, wide, "
        "detect, persist, quench_every, events, path (neb), path_KEY, segment (on: the frames "
        "around a reaction are the first path), and segment_frames; see mapper.md), workers (jobs "
        "at the same time, 1 by default; --max_workers and MultiOptPy's n_parallel set it too), "
        "batch_size (jobs chosen together, the number of workers by default: the network "
        "depends on it and not on the number of workers), schedule (batch, the default: the "
        "jobs of a batch are kept in the order of their numbers when all have ended; rolling: a "
        "job starts whenever a worker is free and is kept when it ends, so that the network can "
        "depend on the timing); kinetics (off, the default; on: when the exploration ends, not "
        "when it is interrupted, the kinetics of the network runs with the kinetics_* settings "
        "below, its files in kinetics/ of the run directory and a section in the report); with "
        "priority=kinetics (the kinetic priority, which --use_rcmc selects) and with "
        "kinetics=on: kinetics_score (population, the default as MultiOptPy; max_population, "
        "residence, discovery_flux), kinetics_solver (propagate, the default; rcmc, legacy_rcmc "
        "for MultiOptPy's numbers, expm), kinetics_time (s, 1.0; --rcmc_time), "
        "kinetics_temperature (K, the temperature of the run; --rcmc_temperature), "
        "kinetics_start (node of the start population, the first input; --rcmc_start_node), "
        "kinetics_energy (electronic, the default; free), kinetics_clamp (consistent, legacy, "
        "none), kinetics_time_rule (gershgorin, eigen, diag), kinetics_tolerance (of propagate, "
        "1e-8), kinetics_rate_theory (eyring, tst), kinetics_corrections (',' between them: "
        "wigner, bell, eckart, skodje_truhlar, kappa, landau_zener, zhu_nakamura), "
        "kinetics_kappa; states (the electronic states of the run, PES1, PES2, ...: "
        "CHARGE,MULTIPLICITY or CHARGE,MULTIPLICITY,ROOT with ';' between them, e.g. "
        "states=0,1;0,3: every minimum lies on its state, and its crossing tasks find the "
        "minimum-energy crossing points with every other state, MECP between spin states, "
        "MECI between two roots of one spin: an AFIR path gives candidates, each is "
        "optimized by the gradient projection without the force, and a meta-IRC runs down "
        "from the crossing point on both states), "
        "crossings (true, the default with states; false: each surface on its own), "
        "crossing_pairs (all, the default: every pair of states; adjacent: PESk with "
        "PESk-1 and PESk+1 only), crossing_afir (the AFIR paths, for every push of afir_ts: "
        "gp, the default, the force on the gradient projection, whose end and the minima of "
        "the seam model function along it are the candidates; "
        "model, on the seam model function, MECP, or the Avoiding model function, MECI, "
        "whose end and minima along the path are the candidates; both; off), "
        "crossing_direct (false; true: also from the minimum itself), crossing_pushes (the "
        "AFIR pushes of the paths by their fragments, GAMMA:ATOMS:ATOMS with ';' between them, "
        "instead of the pairs of atoms of afir_ts), crossing_from (all, the default; inputs: "
        "crossing tasks from the inputs alone), ts_states (none, or the "
        "states, PES numbers with ',' between them, on which transition states are "
        "searched; all by default), crossing_smf_alpha (Hartree, 30 kJ/mol), "
        "crossing_amf_alpha (Hartree, 30 kJ/mol), crossing_amf_branch (upper, lower), "
        "crossing_candidate_prominence (Hartree, 0.5 kJ/mol: how far the function along an "
        "AFIR path must rise on both sides of a minimum for it to be a candidate; 0: every "
        "minimum), crossing_candidate_distance (Angstrom, 0.05: the smallest root-mean-square "
        "displacement between two candidates of a path and between a candidate and a crossing "
        "point the job has found; 0: none), "
        "crossing_parallel_stop (off, the default; on: a crossing search is ended where its "
        "two states run parallel, the gap at most crossing_gap and the difference of the two "
        "gradients below crossing_parallel_gradient, 0.01 Hartree/Bohr, for "
        "crossing_parallel_steps, 10, iterations in a row), "
        "crossing_step (Angstrom, 0.1), crossing_gap (Hartree, 1e-3: the largest gap of the two "
        "states at the end of a converged crossing search; with a larger one the end is no "
        "crossing point, a warning says so, and no meta-IRC runs from it), "
        "crossing_seam_check (on, the default: every accepted crossing point is analysed "
        "within the seam with the Hessians of both states, and one with an imaginary mode "
        "there is kept as a saddle point on the seam, not a minimum-energy crossing point; "
        "off: not checked); the "
        "network of a run on another surface: -remap",
    },
    {
        "names": ["-report_opt", "--report_options"],
        "nargs": "*",
        "default": [],
        "help": "settings of the HTML report (mapper_report.html, written after jobs) as "
        "KEY=VALUE: unit (kcal, the default; kj, ev, hartree), temperature (298.15: K of the "
        "Boltzmann populations of the minima), energy_window (none, the default: every node; a "
        "number: only the nodes at most this far above the lowest minimum, in the unit), "
        "max_minima (all, the default; N: the lowest N minima shown, with the links between "
        "them), structures "
        "(400: the most nodes whose structure is in the page, the lowest first, for its viewer "
        "and downloads), structure_atoms (100000: the most atoms of those structures together), "
        "irc_lines (60000: the most atom lines of the IRC paths in the page, the lowest "
        "transition states first, at most 30 frames each), structure_packed_atoms and "
        "irc_packed_lines (every one, the default: the structures and paths beyond those are "
        "in the page packed, unpacked by the browser when one is shown or saved; a number "
        "limits them, 0: none), figure_minima (300: the minima of "
        "the disconnectivity graph and the energy diagram, and of the figures drawn without "
        "scripts), table_rows (300: rows of each table without scripts), profile_points "
        "(100000: points of the IRC energies in the page), map_exact (1000: up to this many "
        "structures the map is classical scaling of all distances, above it landmark scaling), "
        "map_landmarks (400), file (mapper_report.html: the name of the page in the run "
        "directory)",
    },
    {
        "names": ["--report"],
        "type": "str",
        "default": None,
        "help": "write the HTML report of the run directory RUN_DIR again from its files (a "
        "run that ended, was stopped, or is still running), or from the database of a run "
        "(reaction_network.sqlite of store=sqlite, the page written next to it), with the "
        "settings of the page of the run and those of -report_opt; no input is needed and "
        "nothing is computed; what the files lack is listed on the page instead of stopping it",
    },
    {
        "names": ["-remap", "--remap"],
        "type": "str",
        "default": None,
        "help": "draw the network of the run directory RUN_DIR (or of the database of a run) "
        "again on the surface of this run (-cfg, -map_opt), in a run directory of its own "
        "(--output_dir): its minima are the inputs (no input file is given), each minimized "
        "on its state, and every transition state, crossing point (MECP, MECI) and saddle "
        "point of higher order of it is one job: for a transition state the minima of its "
        "ends and the structures of its path file as the first chain of method_path (neb; "
        "method_path_KEY its settings; method_path=none: its transition state is the guess), "
        "the maxima followed to transition states and their IRCs; for a crossing point the "
        "refinement on the gradient projection of its two states and the meta-IRC on each; "
        "for a saddle point its optimization at its order and its descents "
        "(higher_saddles=on). The first run is only read; a network of several states needs "
        "the same number of states (states); without a limit of jobs given every item runs; "
        "remap.csv of the run directory says what each minimum and item became",
    },
    {
        "names": ["-single_points", "--single_points"],
        "type": "str",
        "default": None,
        "help": "the energies of every node of the network of the run directory RUN_DIR (or "
        "of the database of a run) on the surface of this run (-cfg, -map_opt), at the "
        "structures of that network, in a run directory of its own (--output_dir): no input "
        "file, no optimization and no Hessian. A minimum, a transition state and a saddle "
        "point of higher order have one energy on their state, a crossing point (MECP, MECI) "
        "the energies of its two states (the node has their mean, single_points.csv their "
        "gap), a dissociation channel the sum of its fragments. The network of the run is the "
        "first one with these energies (the same nodes, labels and frequencies, the barriers "
        "computed again; free energies shifted by the change of the energy); its report and "
        "kinetics (kinetics=on) are those of any run. --max_workers energies at a time; a "
        "stopped run continues with --restart; the first run is only read",
    },
    {
        "names": ["-merge", "--merge"],
        "nargs": "+",
        "default": [],
        "help": "the networks of the run directories RUN_DIR ... (or the databases of runs) "
        "written as one network in a run directory of its own (--output_dir), without a "
        "calculation and without an input file: the first as it is, the nodes of the others "
        "added where the merged network has no node that is the same (rmsd_threshold, "
        "energy_tolerance_kcalmol, mirror), their edges where it has none with the same "
        "transition state, crossing point or saddle point, their jobs and inputs appended. "
        "The runs must have the same atoms in the same order and the same states; that they "
        "ran on the same surface is the word of the user. merge.csv says what every node of "
        "every run is in the merged network; the directory is read by run_kinetics, -remap "
        "and -single_points and is not continued by --restart",
    },
]

# Options of run_automecp (a command of mopt-rs).
EXTRA_COMMAND_OPTIONS["run_automecp"] = [
    {
        "names": ["input_file"],
        "help": "the structure the search starts from (xyz); it is minimized on its own state "
        "first",
    },
    {
        "names": ["-cfg", "--config_file"],
        "default": "./config.json",
        "help": "the settings file: step1_settings (the AFIR paths; the calculator, the charge "
        "and multiplicity of the first state), step2_settings (the optimization of the "
        "crossing points), step3_settings (the meta-IRC and the minimizations), each with the "
        "long option names of optmain, and manual_AFIR, state2, crossing_afir, mapper_settings",
    },
    {
        "names": ["-ma", "--manual_AFIR"],
        "nargs": "*",
        "default": [],
        "help": "the AFIR pushes towards the crossing points, as -ma of optmain: triples of "
        "the force parameter (kJ/mol) and the atoms of two fragments, e.g. -ma 150 1,4 7,8",
    },
    {
        "names": ["-state2", "--state2"],
        "nargs": "*",
        "default": [],
        "help": "the second state: CHARGE MULTIPLICITY [ROOT] (root 0, the ground state of "
        "that multiplicity, when it is not given), e.g. -state2 0 3 for the crossing points "
        "with the triplet (MECP), -state2 0 1 1 with the first excited singlet (MECI); the "
        "first state is that of the input (electronic_charge, spin_multiplicity of "
        "step1_settings)",
    },
    {
        "names": ["-root1", "--state1_root"],
        "help": "the root of the first state (0: the lowest state of its charge and "
        "multiplicity; default: excited_state of step1_settings, else 0), e.g. -root1 1 "
        "-state2 0 1 2 for the conical intersections of the first and the second excited "
        "singlet; a root here and another in step1_settings are refused",
    },
    {
        "names": ["-afir", "--crossing_afir"],
        "help": "the kind of AFIR path: gp (the default, the force on the gradient projection "
        "of the two states), model (on the seam model function for an MECP, the Avoiding model "
        "function for an MECI), both",
    },
    {
        "names": ["-map_opt", "--mapper_options"],
        "nargs": "*",
        "default": [],
        "help": "further settings of the mapper run as KEY=VALUE (the keys of run_mapper "
        "-map_opt, e.g. crossing_step=0.05); states, crossing_pushes, crossing_from, "
        "ts_states and crossing_afir are made by the command from its own options and are "
        "refused here; in mapper_settings of the settings file they must agree with them",
    },
    {
        "names": ["--output_dir"],
        "default": "automecp_output",
        "help": "the run directory: settings.json, mapper_config.json (the settings of the "
        "mapper run), automecp_summary.csv (one line per crossing point), and mapper/ (the run "
        "of the mapper: reaction_network.json, structures/, paths/, mapper_report.html, "
        "mopt.log)",
    },
]

# Options of run_kinetics (a command of mopt-rs; run_rcmc keeps MultiOptPy's options).
EXTRA_COMMAND_OPTIONS["run_kinetics"] = [
    {
        "names": ["json_file"],
        "help": "the network: reaction_network.json of run_mapper, its database "
        "reaction_network.sqlite (store=sqlite), or its run directory (the JSON file, or the "
        "database when the directory has only that), or MultiOptPy's network file",
    },
    {
        "names": ["-T", "--temperature"],
        "type": "float",
        "default": 300.0,
        "help": "temperature (K)",
    },
    {
        "names": ["-t", "--time"],
        "type": "float",
        "default": 1.0,
        "help": "the time of interest (s): the output time without --times and --time-grid, and "
        "the horizon of --priority, --sensitivity, --uncertainty (the last output time)",
    },
    {
        "names": ["--times"],
        "nargs": "*",
        "default": [],
        "help": "output times (s), e.g. 1e-9 1e-6 1e-3",
    },
    {
        "names": ["--time-grid"],
        "nargs": "*",
        "default": [],
        "help": "TMIN TMAX N: N output times on a logarithmic grid (s)",
    },
    {
        "names": ["--start-node"],
        "type": "int",
        "default": None,
        "help": "node of the initial population (default: the node of the first input of a "
        "mapper run, node 0 of a network of MultiOptPy)",
    },
    {
        "names": ["--initial"],
        "nargs": "*",
        "default": [],
        "help": "initial population over several nodes as ID:SHARE (normalized), e.g. 0:0.7 3:0.3",
    },
    {
        "names": ["--output-dir"],
        "type": "str",
        "default": "kinetics_output",
        "help": "the run directory (populations.csv, rates.csv, rcmc_K_contracted.csv, "
        "rcmc_superstates.json, diagnostics.json, populations.svg, mopt.log, settings.json)",
    },
    {
        "names": ["--energy-type"],
        "type": "str",
        "default": "electronic",
        "help": "energies of the stationary points with --rate-theory eyring: electronic or "
        "free (the Gibbs energies of the file; stationary points without one are left out and "
        "listed)",
    },
    {
        "names": ["--solver"],
        "type": "str",
        "default": "propagate",
        "help": "propagate (contraction to superstates, then the exact propagation of their "
        "generator; no contraction where the time scales are not separated), rcmc (the "
        "published RCMC snapshot of Type A), legacy_rcmc (MultiOptPy's rcmc.py as it is), expm "
        "(the matrix exponential, the reference), spectral (eigenvectors of the symmetrized "
        "generator), steady (the stationary distribution), rosenbrock (the L-stable integrator "
        "ROS2, for mass_action and as a check), contract (the contraction of mass action of "
        "any reaction order, RCMC's extension: the fast species on their quasi-steady manifold, "
        "the rest propagated), windows (mass action by pseudo-first-order windows, each "
        "propagated by the contraction), kmc (kinetic Monte Carlo, Gillespie; small non-stiff "
        "models and "
        "semi_markov), or a registered name",
    },
    {
        "names": ["--time-rule"],
        "type": "str",
        "default": "gershgorin",
        "help": "reference time of a step of rcmc: gershgorin (ln 2 / sqrt(sigma rho) with the "
        "bounds of Iwata et al., as MultiOptPy), eigen (with the eigenvalues), diag (1 / |D_jj|)",
    },
    {
        "names": ["--tolerance"],
        "type": "float",
        "default": 1e-8,
        "help": "propagate: the eliminated states must have relaxed (sigma t >= ln(1/tol)) and "
        "the separation ratio eps must not exceed it",
    },
    {
        "names": ["--closure"],
        "type": "str",
        "default": "a",
        "help": "propagate: a (G = D diag(1^T M)^-1, the RCMC superstates) or b (G = D M^-1)",
    },
    {
        "names": ["--rate-theory"],
        "type": "str",
        "default": "eyring",
        "help": "eyring ((k_B T / h) exp(-dE / RT) with the energies of --energy-type), tst "
        "(free energies from the frequencies and structures of the states and saddle points, "
        "quasi-RRHO, symmetry numbers, the standard state), or rrkm (the Boltzmann average of "
        "RRKM k(E) from directly counted states, Beyer-Swinehart with classical rotors; the "
        "high-pressure limit of --model egme)",
    },
    {
        "names": ["--corrections"],
        "nargs": "*",
        "default": [],
        "help": "corrections of the flux of every step: wigner, bell (the infinite or the "
        "truncated parabola, --bell-truncation), eckart, skodje_truhlar (tunnelling from the "
        "imaginary frequency and the barriers), kappa (--kappa or a value per step of "
        "--annotations), landau_zener (crossings with a coupling; the double passage of a "
        "peaked or, with slope_product, a sloped crossing), zhu_nakamura (crossings with the "
        "slopes: sloped ones above the crossing energy; peaked ones from the higher minimum "
        "up, tunnelling below the crossing included), weak_coupling (the Airy form of the "
        "weak-coupling probability of a crossing with tunnelling below it; needs slope_product), "
        "cvt (canonical variational "
        "TST of the steps with a reaction path in --annotations; the others keep TST and say "
        "so), small_curvature and zero_curvature (tunnelling through the adiabatic barrier of "
        "path_profile.csv of freqmain -path, sct_profile_file), kramers (--friction or "
        "friction_s-1 per step), grote_hynes (a friction kernel: friction_memory_s or "
        "friction_kernel_file), reactive_flux (the transmission coefficient from trajectories "
        "of the saddle point, reactive_flux_kappa), collins_kimball (the diffusion limit of the "
        "bimolecular steps with --viscosity). Electron transfers are set per edge in "
        "--annotations (transfer)",
    },
    {
        "names": ["--bell-truncation"],
        "type": "str",
        "default": "auto",
        "help": "bell: auto (the infinite parabola (u/2)/sin(u/2) where the truncation changes "
        "kappa by less than 1e-8, the parabola truncated at the barrier near and past "
        "u = 2 pi and for low barriers), on (always truncated), off (always infinite; refused "
        "at u >= 2 pi)",
    },
    {
        "names": ["--bell-barrier"],
        "type": "str",
        "default": "higher",
        "help": "bell: the barrier that truncates the parabola: higher (from the higher "
        "minimum, the same factor from both sides, as skodje_truhlar), forward (from the "
        "first state of the step), reverse (from the second)",
    },
    {
        "names": ["--kappa"],
        "type": "float",
        "default": None,
        "help": "transmission coefficient of every step without its own",
    },
    {
        "names": ["--viscosity"],
        "type": "float",
        "default": None,
        "help": "viscosity of the solvent (Pa s; water at 25 C: 8.9e-4) for the association "
        "of the barrierless dissociations into channels of the mapper: diffusion "
        "(Smoluchowski with Stokes-Einstein) and capture on contact in series (Collins-Kimball); "
        "without it the gas (capture on contact alone)",
    },
    {
        "names": ["--clamp"],
        "type": "str",
        "default": "consistent",
        "help": "a saddle point below one of its minima: consistent (max(E_ts, E_a, E_b) for "
        "both directions, detailed balance kept), legacy (MultiOptPy: each direction by its "
        "own minimum), none",
    },
    {
        "names": ["--annotations"],
        "type": "str",
        "default": None,
        "help": "JSON file with data of nodes and edges by label or number: {\"nodes\": "
        "{\"EQ3\": {\"frequencies_cm1\": [...], \"free_energy_hartree\": ..., "
        "\"symmetry_number\": 2}}, \"edges\": {\"TS12\": {\"kappa\": 0.5, \"coupling_cm1\": "
        "50, \"gradient_difference\": 0.02, \"reduced_mass_amu\": 10, \"partner\": \"S\", "
        "\"partner_free_state\": \"EQ1\"}, \"TS4\": {\"transfer\": \"nonadiabatic\", "
        "\"reorganization_kcal_mol\": 20, \"coupling_cm1\": 30}, \"TS5\": "
        "{\"cvt_profile_file\": \"free_energy_profile.csv\"}}, \"species\": {\"S\": "
        "{\"energy_hartree\": ..., "
        "\"free_energy_hartree\": ..., \"frequencies_cm1\": [...], \"xyz\": \"...\", "
        "\"association\": \"factor:2\"}}}",
    },
    {
        "names": ["--model"],
        "type": "str",
        "default": "canonical",
        "help": "kinetic model: canonical (thermalized minima, first-order steps; bimolecular "
        "steps pseudo-first order at --concentration), mass_action (the states and the "
        "species as concentrations, non-linear, integrated by the solver rosenbrock; needs "
        "--total-concentration), auto (canonical where every bimolecular step is pseudo-first "
        "order, its partner changing by less than --order-tolerance with the states at "
        "--total-concentration; mass_action with the solver contract otherwise), egme (the "
        "energy-grained master equation with collisions; "
        "--rate-theory rrkm, --collision-model, --collision-frequency or --bath; "
        "phenomenological.csv), two_grain (hot intermediates as additional states; "
        "--cooling-rate), or semi_markov (gamma-distributed residence times of residence_shape "
        "in --annotations, or those of --md-trajectory; sampled by --solver kmc)",
    },
    {
        "names": ["-press", "--pressure"],
        "type": "float",
        "default": 101325.0,
        "help": "pressure of the standard state of the gas (Pa), as -press of freqmain: the "
        "translational entropy of tst, and the conversion RT ln(c R T / p) of the free energies "
        "of bimolecular steps to the standard concentration",
    },
    {
        "names": ["-conc", "--standard-concentration"],
        "type": "float",
        "default": None,
        "help": "the standard state of a solution (mol/L) instead of the gas, as -conc of "
        "freqmain: the free energies are taken there (tst computes them there) and it is the "
        "standard concentration of the bimolecular steps (default 1 mol/L, to which free "
        "energies of the gas are converted)",
    },
    {
        "names": ["--concentration"],
        "nargs": "*",
        "default": [],
        "help": "concentrations of the species of bimolecular steps as NAME:VALUE[UNIT] (mol/L "
        "by default; M, mM, uM, nM, pM, mol/m3, or a partial pressure in bar, atm, Pa): held "
        "fixed (pseudo-first order, k' = k2 [S]) with the model canonical, initial values with "
        "mass_action",
    },
    {
        "names": ["--species"],
        "nargs": "*",
        "default": [],
        "help": "species of bimolecular steps as NAME:E_HARTREE[:G_HARTREE] (electronic and "
        "free energy, '-' for none; or under \"species\" of --annotations with frequencies and "
        "structure)",
    },
    {
        "names": ["--partner"],
        "nargs": "*",
        "default": [],
        "help": "bimolecular edges as EDGE:SPECIES:STATE (the edge takes up the species from its "
        "state STATE, where the species is free; or \"partner\" and \"partner_free_state\" of an "
        "edge of --annotations)",
    },
    {
        "names": ["--association"],
        "type": "str",
        "default": "none",
        "help": "association entropy in solution: none (the conversion of the standard state "
        "only), wertz:F (every structure loses the fraction F of its translational and "
        "rotational entropy), free_volume:M:V or free_volume:SOLVENT (free volume of the "
        "solvent as GoodVibes), effective:C (an effective concentration in mol/L), factor:F",
    },
    {
        "names": ["--total-concentration"],
        "type": "float",
        "default": None,
        "help": "--model mass_action: the initial concentration of the states together (mol/L), "
        "shared by the initial population",
    },
    {
        "names": ["--ode-tolerance"],
        "type": "float",
        "default": 1e-7,
        "help": "--solver rosenbrock: relative tolerance of the step-size control",
    },
    {
        "names": ["--qrrho"],
        "type": "str",
        "default": "grimme",
        "help": "--rate-theory tst: low frequencies by grimme (interpolation to a free rotor), "
        "truhlar (raised to the cutoff), or none",
    },
    {
        "names": ["--qrrho-cutoff"],
        "type": "float",
        "default": 100.0,
        "help": "--rate-theory tst: the cutoff of --qrrho (cm^-1)",
    },
    {
        "names": ["--symmetry"],
        "type": "str",
        "default": "auto",
        "help": "--rate-theory tst: rotational symmetry numbers from the point groups (auto) "
        "or 1 (none), where the annotations give none",
    },
    {
        "names": ["--level"],
        "type": "int",
        "default": 1,
        "help": "the level of a network of mopt-rs (1: exploration)",
    },
    {
        "names": ["--sensitivity"],
        "nargs": "*",
        "default": [],
        "help": "derivatives of the yields of these nodes (without a node: every state) at every "
        "output time and at t -> infinity with respect to the energy of every node (minima, "
        "transition states, crossings; 1/Hartree), ln k of every step (both directions "
        "together, so detailed balance holds; d ln p / d ln k is the degree of rate control), "
        "and the temperature (1/K), by forward-mode dual numbers through the solver (propagate, "
        "rcmc, expm, steady): sensitivity.csv; the degrees of rate control of the effective rate "
        "constants between superstates, which sum to 1: rate_control.csv",
    },
    {
        "names": ["--uncertainty"],
        "type": "float",
        "default": None,
        "help": "standard deviation (kcal/mol) of independent errors of the energies of the "
        "stationary points: quantiles of the populations at the last time over --samples "
        "draws, in uncertainty.csv",
    },
    {
        "names": ["--samples"],
        "type": "int",
        "default": 100,
        "help": "draws of --uncertainty",
    },
    {
        "names": ["--seed"],
        "type": "int",
        "default": 42,
        "help": "seed of the draws of --uncertainty and of the solver kmc",
    },
    {
        "names": ["--validate"],
        "nargs": "?",
        "default": None,
        "help": "compare the populations with the matrix exponential and report the largest "
        "difference against TOL (default: the error estimate of the solver, max(eps_hat, "
        "exp(-sigma_hat t), 1e-10) for a contraction, 1e-10 otherwise)",
    },
    {
        "names": ["--priority"],
        "type": "str",
        "default": None,
        "help": "rank the states for an exploration: population, max_population, residence, "
        "discovery_flux (priority.csv)",
    },
    {
        "names": ["--lump-conformers"],
        "action": "store_true",
        "default": False,
        "help": "also report the populations of states with the same bond graph together "
        "(species.csv)",
    },
    {
        "names": ["--spectrum"],
        "action": "store_true",
        "default": False,
        "help": "the relaxation rates of the generator and their gaps (spectrum.csv)",
    },
    {
        "names": ["--refine"],
        "nargs": "*",
        "default": [],
        "help": "rank the refinement tasks (better energies of nodes and saddle points, the "
        "split of an N-furcation, the coupling of a crossing) by |d ln y / d theta| sigma for "
        "the populations of these nodes at the last output time (without a node: the most "
        "populated state); sigma from --uncertainty (1 kcal/mol without it) or "
        "uncertainty_kcal_mol in --annotations (refinement.csv)",
    },
    {
        "names": ["--friction"],
        "type": "float",
        "default": None,
        "help": "kramers, grote_hynes: the friction on the reaction coordinate (1/s) of every "
        "saddle point without its own (friction_s-1 in --annotations)",
    },
    {
        "names": ["--rrkm-cell"],
        "type": "float",
        "default": 1.0,
        "help": "rrkm, egme, two_grain: the width of the cells of the direct count of states "
        "(cm^-1)",
    },
    {
        "names": ["--egme-grain"],
        "type": "float",
        "default": 50.0,
        "help": "egme: the width of the energy grains (cm^-1)",
    },
    {
        "names": ["--egme-above"],
        "type": "float",
        "default": 40.0,
        "help": "egme: the grains reach this many k_B T above the highest saddle point or well",
    },
    {
        "names": ["--collision-model"],
        "type": "str",
        "default": "exponential_down",
        "help": "egme: exponential_down (with --energy-transfer, detailed balance, normalized "
        "from the top grain down) or strong",
    },
    {
        "names": ["--energy-transfer"],
        "type": "float",
        "default": None,
        "help": "egme, exponential_down: the mean energy transferred downwards in a collision "
        "(cm^-1)",
    },
    {
        "names": ["--collision-frequency"],
        "type": "float",
        "default": None,
        "help": "egme: the collision frequency (1/s); without it the Lennard-Jones frequency at "
        "--pressure with --bath and --lennard-jones",
    },
    {
        "names": ["--bath"],
        "type": "str",
        "default": None,
        "help": "egme: the bath gas as SIGMA:EPS:MASS (Angstrom, K, amu), e.g. 3.47:114:39.95 "
        "for argon",
    },
    {
        "names": ["--lennard-jones"],
        "type": "str",
        "default": None,
        "help": "egme: SIGMA:EPS (Angstrom, K) of the molecule for every well without its own "
        "(lennard_jones in --annotations)",
    },
    {
        "names": ["--cooling-rate"],
        "type": "float",
        "default": None,
        "help": "two_grain: the cooling rate of a hot intermediate (1/s; of the order of 1e11 "
        "to 1e12 in solution)",
    },
    {
        "names": ["--hot-threshold"],
        "type": "float",
        "default": 0.1,
        "help": "two_grain: a hot grain is added where H = k_exit(E*) / cooling rate exceeds this",
    },
    {
        "names": ["--kmc-trajectories"],
        "type": "int",
        "default": 10000,
        "help": "kmc: the number of trajectories",
    },
    {
        "names": ["--kmc-max-events"],
        "type": "float",
        "default": 1e8,
        "help": "kmc: the largest number of events of a run (more means the network is too "
        "stiff for kinetic Monte Carlo)",
    },
    {
        "names": ["--contract-theta"],
        "type": "float",
        "default": 100.0,
        "help": "contract: species are eliminated while their effective loss rate times the "
        "output time exceeds this",
    },
    {
        "names": ["--window-tolerance"],
        "type": "float",
        "default": 1e-3,
        "help": "windows: the largest relative change of a species over one pseudo-first-order "
        "window",
    },
    {
        "names": ["--order-tolerance"],
        "type": "float",
        "default": 1e-2,
        "help": "--model auto: a bimolecular step is pseudo-first order when its partner changes "
        "by less than this (with the states at --total-concentration)",
    },
    {
        "names": ["--md-trajectory"],
        "type": "str",
        "default": None,
        "help": "semi_markov: an XYZ trajectory whose frames are assigned to the minima of the "
        "network (bond graph, then RMSD; core sets): residence times and jumps for the model, "
        "and a Markov state model at --md-lag (msm.csv)",
    },
    {
        "names": ["--md-time-step"],
        "type": "float",
        "default": None,
        "help": "--md-trajectory: the time between two frames (fs)",
    },
    {
        "names": ["--md-lag"],
        "type": "int",
        "default": 1,
        "help": "--md-trajectory: the lag of the Markov state model (frames)",
    },
    {
        "names": ["--log-level"],
        "type": "str",
        "default": "INFO",
        "help": "what the screen shows (DEBUG, INFO, WARNING, ERROR); mopt.log has every message",
    },
]

# Notes of the run_mapper options (what mopt-rs does differently from MultiOptPy).
NOTES.update(
    {
        ("run_mapper", "config_file"): "MultiOptPy's settings file (mapper_settings and "
        "step1_settings to step4_settings); a restart reads the copy in the run directory",
        ("run_mapper", "top_n"): "the transition-state guesses of a job that are optimized (the "
        "highest maxima of the energy along its AFIR path)",
        ("run_mapper", "rmsd_threshold"): "after the best matching of equivalent atoms, with "
        "the same bond graph; mirror images are different minima. The default of mopt-rs is "
        "0.125 Angstrom, with 0.05 kcal/mol for the energies (energy_tolerance_kcalmol): "
        "MultiOptPy's 0.30 Angstrom and 1 kcal/mol took different minima for one",
        ("run_mapper", "output_dir"): "the run directory, which holds every file of the run",
        ("run_mapper", "log_level"): "of the screen, which has a short line when a job starts "
        "and ends; mopt.log has every message, those of each job as one block in the order of "
        "the jobs",
        ("run_mapper", "max_workers"): "jobs at the same time, in threads of the command (each "
        "with its own log and, where the calculator evaluates in order, its own calculator); "
        "the same as -map_opt workers=N; the jobs of a batch (batch_size) are kept in the order "
        "of their numbers, so that the network does not depend on which job ends first",
        ("run_mapper", "use_rcmc"): "the priority kinetics with the score population; the "
        "solver is kinetics_solver (propagate by default, legacy_rcmc for MultiOptPy's numbers)",
        ("run_mapper", "rcmc_temperature"): "kinetics_temperature (default: the temperature of "
        "the run)",
        ("run_mapper", "rcmc_time"): "kinetics_time",
        ("run_mapper", "rcmc_start_node"): "kinetics_start (default: the node of the first "
        "input)",
        ("run_mapper", "manual_AFIR"): "a static AFIR bias of the surface of every step, shared "
        "by all steps like the other bias options; the mapper makes its own AFIR pushes on top "
        "of it (MultiOptPy replaces this value in every job)",
        ("run_autots", "manual_AFIR"): "the AFIR push of step 1; with a workflow block it "
        "replaces the manual_AFIR of every run of step 1, and the log says so",
        ("run_rcmc", "json_file"): "also the reaction_network.json of run_mapper, its "
        "reaction_network.sqlite, or its run directory; run_kinetics has the other solvers "
        "and rate models",
        ("run_rcmc", "output_dir"): "rcmc_K_contracted.csv as MultiOptPy writes it, with the "
        "other files of run_kinetics",
        ("run_rcmc", "log_level"): "the results are printed at every level, as in MultiOptPy",
    }
)

# Notes of the relaxedscan options (what mopt-rs does differently from MultiOptPy).
NOTES.update(
    {
        (
            "relaxedscan",
            "number_of_samples",
        ): "one number for every coordinate, or one per coordinate with -grid (10,15)",
        (
            "relaxedscan",
            "scan_tgt",
        ): "kinds bond, fbond, x, y, z (Angstrom), angle, dihedral (degrees), expr (an "
        "expression of -ubias, atomic units; the expression in place of the atoms), var (a "
        "variable of an objective of plain variables by number or name), resper (PCo 1 or 2 of "
        "-resper), cell (with -vc: a parameter of the cell named as in -vc_fix in place of the "
        "atoms, a, b, c, ax, ..., cz in Angstrom, alpha, beta, gamma in degrees, volume in "
        "Angstrom^3, e.g. -scan cell volume 250,330); a range may start with a negative number "
        "(-180,180); the constraints of -pc and -fix hold as well (MultiOptPy replaces -pc by "
        "the scan)",
        (
            "relaxedscan",
            "first_only",
        ): "every point starts from the input structure (the start of a ReSPer point is its "
        "interpolated structure)",
    }
)

# Options of other commands that nebmain takes as well (MultiOptPy's nebmain has its own
# parser with the calculator, bias, -fix, and -pc options only): `(command, option)`.
NEBMAIN_SHARED = [
    ("ieipmain", "-gfix"),
    ("optmain", "-uc"),
    ("optmain", "-ubias"),
    ("optmain", "-calc_opt"),
    ("optmain", "-oniom"),
    ("optmain", "-oniom_middle"),
    ("optmain", "-oniom_opt"),
    ("optmain", "-stability"),
    ("optmain", "-stability_external"),
    ("optmain", "-es_follow"),
    ("optmain", "-parallel"),
    ("optmain", "-unpaired"),
    ("optmain", "-debug"),
    ("optmain", "-mf"),
    ("optmain", "-coord"),
    ("optmain", "-vc"),
    ("optmain", "-vc_pressure"),
    ("optmain", "-vc_weight"),
    ("optmain", "-vc_fix"),
    ("optmain", "-vc_afir"),
]

# Help texts of the variable-cell options in the commands that take them from optmain (by
# command and destination).
VARIABLE_CELL_HELP = {
    "nebmain": {
        "variable_cell": "relax the chain with the cell of every image as a part of it (cells "
        "periodic in three directions, or slabs and wires): the images are points of the "
        "variables of optmain -vc made from the first structure (quasi-Cartesian coordinates of "
        "the atoms and scaled lattice vectors), the same chain as the band of run_mapper on a "
        "variable cell; the ends may have different cells; the profile is the enthalpy E + P V",
        "variable_cell_pressure": "-vc: external pressure in GPa; the images are relaxed on "
        "the enthalpy E + P V",
        "variable_cell_afir": "-vc: refused for a chain (the artificial forces on the cell "
        "push it in a search; the chain is relaxed on the surface itself)",
    },
    "ieipmain": {
        "variable_cell": "search in the variables of the atoms and the cell (those of optmain "
        "-vc made from the first structure: quasi-Cartesian coordinates of the atoms and scaled "
        "lattice vectors), the vector of the two-point searches of run_mapper on a variable "
        "cell; the two structures may have different cells; the surface is the enthalpy E + P V",
        "variable_cell_pressure": "-vc: external pressure in GPa; the searches run on the "
        "enthalpy E + P V",
        "variable_cell_afir": "-vc: refused for the searches of ieipmain (the artificial forces "
        "on the cell push it in an optimization)",
    },
    "mdmain": {
        "variable_cell": "the cell as a dynamical variable at constant pressure, the common "
        "spelling of -mt npt*: with a constant-pressure ensemble of -mt (or -md_opt barostat=) "
        "its barostat, without -mt the fully flexible Martyna-Tobias-Klein barostat with the "
        "Nose-Hoover chain (-mt npt_aniso); an ensemble without a barostat is refused",
        "variable_cell_pressure": "-vc: external pressure in GPa (0 without it, as optmain "
        "-vc; -press gives it in kPa instead, not both)",
        "variable_cell_weight": "-vc: refused for the dynamics (the inertia of the cell is the "
        "time constant of the barostat, -md_opt tau_p=)",
        "variable_cell_fix": "-vc: refused for the dynamics (the barostat holds no constraint "
        "of the cell)",
        "variable_cell_afir": "-vc: refused for the dynamics (the artificial forces on the cell "
        "push it in an optimization)",
    },
}

# Options of other commands that ieipmain takes as well.
IEIPMAIN_SHARED = [
    ("optmain", "-fix"),
    ("optmain", "-pc"),
    ("optmain", "-uc"),
    ("optmain", "-ubias"),
    ("optmain", "-calc_opt"),
    ("optmain", "-oniom"),
    ("optmain", "-oniom_middle"),
    ("optmain", "-oniom_opt"),
    ("optmain", "-stability"),
    ("optmain", "-stability_external"),
    ("optmain", "-es_follow"),
    ("optmain", "-parallel"),
    ("optmain", "-unpaired"),
    ("optmain", "-debug"),
    ("optmain", "-coord"),
    ("optmain", "-vc"),
    ("optmain", "-vc_pressure"),
    ("optmain", "-vc_weight"),
    ("optmain", "-vc_fix"),
    ("optmain", "-vc_afir"),
]

# Notes of the ieipmain options (what mopt-rs does differently from MultiOptPy).
IEIPMAIN_NOTES = {
    "INPUT": "a directory with *_[A-Z].xyz (in the order of the letters), or one or two "
    "structure or variables files",
    "opt_method": "used by -mf and by the corrector of -gnt",
    "calc_exact_hess": "used by -mf",
    "model_function_mode": "optmain -mf at the first structure with the charges, "
    "multiplicities, and -es of the two inputs as the two states; with -dsafir, -gsm, "
    "-se_gsm, -fsm, or -lmm, that method on the function of the two states",
    "gradient_fix_atoms": "the distances between the atoms held exactly, as -gfix of optmain "
    "and mdmain (MultiOptPy parses it and drops it); see reaction_paths.md for the methods "
    "that hold constraints",
    "microiter": "the relaxation on hyperspheres with the gradient of its coordinates (see the "
    "migration notes)",
    "addf_step_size": "Angstrom along the stiffest mode per sphere, as the method defines its "
    "spheres (MultiOptPy: an increment of the square root of the harmonic energy)",
    "twoPshs_step_size": "Angstrom along the stiffest mode of the first structure per "
    "contraction of the sphere (MultiOptPy reads -addf_step instead)",
    "twoPshs_step_num": "used (MultiOptPy reads -addf_num instead)",
    "dimer_separation": "Bohr; used (MultiOptPy ignores the dimer options)",
    "dimer_trial_angle": "radians; used",
    "dimer_max_iterations": "used",
    "gnt_vec": "pairs of atoms whose bonds are shortened",
}


def nebmain_shared(spec: dict[str, Any]) -> list[dict[str, Any]]:
    taken = []
    for command, name in NEBMAIN_SHARED:
        options = (
            optmain_options(spec) if command == "optmain" else spec["commands"][command]["options"]
        )
        found = [o for o in options if name in o["names"]]
        if len(found) != 1:
            raise SystemExit(f"nebmain: {command} has {len(found)} options named {name}")
        taken.append({**found[0], "origin": "mopt-rs", "line": 0})
    return taken


def ieipmain_shared(spec: dict[str, Any]) -> list[dict[str, Any]]:
    taken = []
    for _, name in IEIPMAIN_SHARED:
        found = [o for o in optmain_options(spec) if name in o["names"]]
        if len(found) != 1:
            raise SystemExit(f"ieipmain: optmain has {len(found)} options named {name}")
        taken.append({**found[0], "origin": "mopt-rs", "line": 0})
    return taken


# Options of other commands that mdmain takes as well: `(command, option)`.
MDMAIN_SHARED = [
    ("ieipmain", "-gfix"),
    ("optmain", "-uc"),
    ("optmain", "-ubias"),
    ("optmain", "-calc_opt"),
    ("optmain", "-oniom"),
    ("optmain", "-oniom_middle"),
    ("optmain", "-oniom_opt"),
    ("optmain", "-stability"),
    ("optmain", "-stability_external"),
    ("optmain", "-es_follow"),
    ("optmain", "-parallel"),
    ("optmain", "-unpaired"),
    ("optmain", "-debug"),
    ("optmain", "-mf"),
    ("optmain", "-dxtb"),
    ("optmain", "-vc"),
    ("optmain", "-vc_pressure"),
    ("optmain", "-vc_weight"),
    ("optmain", "-vc_fix"),
    ("optmain", "-vc_afir"),
]


def mdmain_shared(spec: dict[str, Any]) -> list[dict[str, Any]]:
    taken = []
    for command, name in MDMAIN_SHARED:
        options = (
            optmain_options(spec) if command == "optmain" else spec["commands"][command]["options"]
        )
        found = [o for o in options if name in o["names"]]
        if len(found) != 1:
            raise SystemExit(f"mdmain: {command} has {len(found)} options named {name}")
        taken.append({**found[0], "origin": "mopt-rs", "line": 0})
    return taken


# Options of optmain that run_mapper and run_autots take as shared by every step: the hard
# constraints and the bias potentials, merged into every step block (a key of a step block
# wins over them).
WORKFLOW_SHARED = (
    CONSTRAINT_OPTIONS + ["variable_cell_fix", "user_bias_expression"] + BIAS_OPTIONS
)


def workflow_shared(spec: dict[str, Any], own: set[str]) -> list[dict[str, Any]]:
    taken = []
    for o in optmain_options(spec):
        if dest(o) in WORKFLOW_SHARED and dest(o) not in own:
            taken.append(
                {
                    **o,
                    "origin": "mopt-rs",
                    "line": 0,
                    "help": f"{o['help']} (shared by every step: merged into every step block "
                    "of the settings file, where a step that sets it itself wins; also the "
                    "block shared_settings of the settings file)",
                }
            )
    return taken


def own_options(spec: dict[str, Any], command: str) -> list[dict[str, Any]]:
    extra = [{**o, "origin": "mopt-rs", "line": 0} for o in EXTRA_COMMAND_OPTIONS.get(command, [])]
    options = list(spec["commands"].get(command, {"options": []})["options"])
    if command in ("run_mapper", "run_autots"):
        extra += workflow_shared(spec, {dest(o) for o in options + extra})
    if command == "nebmain":
        options = [
            {**o, "default": 30, "help": "Maximum iterations (60 with -dmf)."}
            if dest(o) == "NSTEP"
            else o
            for o in options
        ]
        extra += nebmain_shared(spec)
    if command == "ieipmain":
        extra += ieipmain_shared(spec)
    if command == "mdmain":
        # MultiOptPy's help texts give other units and defaults than the options have.
        mdmain_help = {
            "timestep": "time step in atomic units of time, or with 'fs' after the number "
            "(default: 0.5 fs for atoms, 0.1 for plain variables)",
            "pressure": "pressure of the barostat in kPa (default: 101.3 kPa; freqmain and "
            "optmain -freq take Pa, -vc_pressure GPa)",
        }
        options = [
            {**o, "help": mdmain_help[dest(o)]} if dest(o) in mdmain_help else o
            for o in options
        ]
        extra += mdmain_shared(spec)
    if command in VARIABLE_CELL_HELP:
        helps = VARIABLE_CELL_HELP[command]
        extra = [{**o, "help": helps[dest(o)]} if dest(o) in helps else o for o in extra]
    return options + extra


def options_of(spec: dict[str, Any], command: str) -> list[dict[str, Any]]:
    if command == "optmain":
        return optmain_options(spec)
    opts = own_options(spec, command)
    if command in EXTENDS_OPTMAIN:
        own = {dest(o) for o in opts}
        # The standalone post-processing choice does not change a workflow's endpoint policy.
        opts += [o for o in optmain_options(spec) if dest(o) not in own | {"mode_kill_method"}]
    return opts


def on_optmain(command: str, o: dict[str, Any]) -> bool:
    return command == "optmain" or (
        command in EXTENDS_OPTMAIN and o["origin"].startswith("interface.py")
    )


def implemented(command: str, o: dict[str, Any]) -> bool:
    d = dest(o)
    if d in DROPPED_OPTIONS:
        return False
    if o["origin"] == "mopt-rs":
        return True
    if on_optmain(command, o):
        return d in OPTMAIN_IMPLEMENTED or d in IMPLEMENTED
    if command == "nebmain" and d in NEBMAIN_PENDING:
        return False
    if command == "run_mapper" and (d in MAPPER_PENDING or d in MAPPER_DROPPED):
        return False
    return command in IMPLEMENTED_COMMANDS


def feature(command: str, o: dict[str, Any]) -> str:
    """What is missing when the option is not implemented yet."""
    if implemented(command, o):
        return ""
    if dest(o) in DROPPED_OPTIONS:
        return DROPPED_OPTIONS[dest(o)]
    if on_optmain(command, o):
        return OPTION_FEATURES.get(dest(o), "this option")
    if command == "nebmain" and dest(o) in NEBMAIN_PENDING:
        return NEBMAIN_PENDING[dest(o)]
    if command == "run_mapper" and dest(o) in MAPPER_PENDING:
        return MAPPER_PENDING[dest(o)]
    if command == "run_mapper" and dest(o) in MAPPER_DROPPED:
        return MAPPER_DROPPED[dest(o)]
    return COMMANDS[command]


def gen_rust(spec: dict[str, Any]) -> str:
    out = [
        "//! Option tables of the commands, generated by `tools/cli/gen_cli.py` from MultiOptPy",
        "//! 1.21.2 (`tools/cli/multioptpy_cli.json`). Do not edit by hand.",
        "//!",
        "//! MultiOptPy: `interface.py` (`optimizeparser`, `nebparser`, `ieipparser`, `mdparser`,",
        "//!  `parser_for_biasforce`) and the parsers of `Entrypoints/`.",
        "//!",
        "//! This file is part of mopt-rs.",
        "//! Copyright (C) 2023-2026 ss0832",
        "//! SPDX-License-Identifier: GPL-3.0-or-later",
        "",
        "use super::option::CliOption;",
        "use crate::core_infra::arg_parser::{ArgArity, OptSpec};",
        "",
    ]
    for command in COMMANDS:
        opts = [o for o in options_of(spec, command) if not is_positional(o)]
        out.append(f"/// Options of `{command}`.")
        out.append(f"pub const {command.upper()}: &[CliOption] = &[")
        for o in opts:
            flags = ", ".join(rust_str(n) for n in o["names"])
            help_text = help_of(o)
            out += [
                "    CliOption {",
                "        spec: OptSpec {",
                f"            key: {rust_str(dest(o))},",
                f"            flags: &[{flags}],",
                f"            arity: {arity(o)},",
                f"            help: {rust_str(help_text)},",
                "        },",
                f"        default: {rust_str(default_text(o))},",
                f"        implemented: {'true' if implemented(command, o) else 'false'},",
                f"        feature: {rust_str(feature(command, o))},",
                "    },",
            ]
        out.append("];")
        out.append("")
    out.append(
        "/// The positional input of each command that has one: `(command, name, several values,"
    )
    out.append("/// help)` (the MCP server offers it as the property `INPUT` of the command's tool).")
    out.append("pub const POSITIONALS: &[(&str, &str, bool, &str)] = &[")
    for command in COMMANDS:
        positional = next((o for o in options_of(spec, command) if is_positional(o)), None)
        if positional is None:
            continue
        name = rust_str(str(positional["names"][0]))
        multiple = "true" if positional.get("nargs") in ("*", "+") else "false"
        help_text = rust_str(str(positional.get("help", "")))
        out.append(f"    ({rust_str(command)}, {name}, {multiple}, {help_text}),")
    out.append("];")
    out.append("")
    out.append("/// `(command, implemented, feature, options)` for all commands.")
    out.append("pub const COMMANDS: &[(&str, bool, &str, &[CliOption])] = &[")
    for command, what in COMMANDS.items():
        done = "true" if command in IMPLEMENTED_COMMANDS else "false"
        out.append(f"    ({rust_str(command)}, {done}, {rust_str(what)}, {command.upper()}),")
    out.append("];")
    return "\n".join(out) + "\n"


def gen_json(spec: dict[str, Any]) -> str:
    """The tables of ``spec.rs`` with the positional inputs, as JSON (the tools of the MCP server
    of the Python package are built from it; ``crates/mopt-rs/tests/mcp_tables.rs`` checks that
    it agrees with ``spec.rs``)."""
    arities = {
        "ArgArity::Flag": "flag",
        "ArgArity::Single": "single",
        "ArgArity::ZeroOrMore": "zero_or_more",
        "ArgArity::OneOrMore": "one_or_more",
    }

    def plain(s: str) -> str:
        # As rust_str writes it, without the quoting.
        return s.replace("\n", " ")

    commands = []
    for command, what in COMMANDS.items():
        every = options_of(spec, command)
        positional = next((o for o in every if is_positional(o)), None)
        options = []
        for o in every:
            if is_positional(o):
                continue
            a = arity(o)
            optional = a.startswith("ArgArity::Optional")
            implicit = a[len('ArgArity::Optional("') : -2] if optional else None
            options.append(
                {
                    "key": dest(o),
                    "flags": [str(n) for n in o["names"]],
                    "arity": arities.get(a, "optional"),
                    "implicit": implicit,
                    "help": plain(help_of(o)),
                    "default": plain(default_text(o)),
                    "implemented": implemented(command, o),
                    "feature": plain(feature(command, o)),
                }
            )
        commands.append(
            {
                "name": command,
                "implemented": command in IMPLEMENTED_COMMANDS,
                "feature": what,
                "positional": None
                if positional is None
                else {
                    "name": str(positional["names"][0]),
                    "multiple": positional.get("nargs") in ("*", "+"),
                    "help": plain(str(positional.get("help", ""))),
                },
                "options": options,
            }
        )
    doc = {
        "about": "The option tables of the commands (as crates/mopt-rs/src/cli/spec.rs) with "
        "their positional inputs, generated by tools/cli/gen_cli.py. Do not edit by hand.",
        "commands": commands,
    }
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def table_text(text: str) -> str:
    """`text` as the plain text of a table cell: the characters that Markdown reads as markup
    (a cell separator, emphasis as in ``0.5*k*(r - r0)`` or ``<stem>_Sample_<k>_OPT``, a tag
    as in ``<stem>_traj.xyz``) are escaped outside code spans, so that GitHub and Sphinx show
    the text as it is; inside a code span only the cell separator is (a backslash there is
    shown as it is)."""
    out = []
    for k, part in enumerate(text.replace("\n", " ").split("`")):
        marks = ("|",) if k % 2 == 1 else ("|", "*", "<", "_")
        for mark in marks:
            part = part.replace(mark, "\\" + mark)
        out.append(part)
    return "`".join(out)


def gen_markdown(spec: dict[str, Any]) -> str:
    out = [
        "# Command-line reference",
        "",
        "mopt-rs provides the ten commands of MultiOptPy with the same options (extracted from",
        "MultiOptPy 1.21.2 by `tools/cli/extract_multioptpy_cli.py`; this page is generated by",
        "`tools/cli/gen_cli.py`). Each command is also available as a subcommand:",
        "`mopt-py optmain input.xyz ...` (Python) or `mopt optmain input.xyz ...` (Rust binary).",
        "",
        "The *Implemented* column says whether an option works in this version. Options that",
        "are not implemented yet are parsed and rejected with exit code 2 when they are given a",
        "value other than their default; commands that are not implemented yet parse their",
        "options and exit with code 2.",
        "",
        "| Command | Feature | Implemented |",
        "|---|---|---|",
    ]
    out += [
        f"| `{c}` | {w} | {'yes' if c in IMPLEMENTED_COMMANDS else 'not yet'} |"
        for c, w in COMMANDS.items()
    ]
    for command in COMMANDS:
        out += ["", f"## `{command}`", ""]
        if command in EXTENDS_OPTMAIN:
            out += ["Also accepts every `optmain` option.", ""]
        out += [
            "| Option | Arguments | Default | Implemented | Description |",
            "|---|---|---|---|---|",
        ]
        own = options_of(spec, "optmain") if command == "optmain" else own_options(spec, command)
        for o in own:
            names = ", ".join(f"`{n}`" for n in o["names"])
            a = {
                "ArgArity::Flag": "flag",
                "ArgArity::ZeroOrMore": "0 or more",
                "ArgArity::OneOrMore": "1 or more",
                "ArgArity::Single": "1",
            }.get(arity(o), "optional 1")
            if is_positional(o):
                a = "positional" + (" (0 or more)" if o.get("nargs") == "*" else "")
            note = NOTES.get((command, dest(o)), "")
            if command == "ieipmain" and not note:
                note = IEIPMAIN_NOTES.get(dest(o), "")
            desc = table_text(help_of(o))
            if o["origin"] == "mopt-rs":
                note = (note + "; " if note else "") + "mopt-rs extension"
            if note:
                desc += f" *({table_text(note)})*"
            d = default_text(o).replace("|", "\\|")
            if implemented(command, o):
                m = "yes"
            elif dest(o) in DROPPED_OPTIONS or (
                command == "run_mapper" and dest(o) in MAPPER_DROPPED
            ):
                m = f"no ({feature(command, o)})"
            else:
                m = f"not yet ({feature(command, o)})"
            out.append(f"| {names} | {a} | `{d}` | {m} | {desc} |")
    return "\n".join(out) + "\n"


def main() -> None:
    root = Path(sys.argv[1])
    spec = json.loads((root / "tools/cli/multioptpy_cli.json").read_text(encoding="utf-8"))
    rs = root / "crates/mopt-rs/src/cli/spec.rs"
    rs.parent.mkdir(parents=True, exist_ok=True)
    rs.write_text(gen_rust(spec), encoding="utf-8", newline="\n")
    md = root / "docs/source/cli_reference.md"
    md.write_text(gen_markdown(spec), encoding="utf-8", newline="\n")
    tables = root / "python/mopt_rs/mcp/commands.json"
    tables.parent.mkdir(parents=True, exist_ok=True)
    tables.write_text(gen_json(spec), encoding="utf-8", newline="\n")
    print(f"wrote {rs}, {md} and {tables}")


if __name__ == "__main__":
    main()
