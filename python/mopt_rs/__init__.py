"""mopt-rs: successor implementation of MultiOptPy (geometry optimization and reaction paths).

MultiOptPy: `multioptpy/__init__.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from mopt_rs._core import (
    OniomCalculator,
    Optimizer,
    ResperMap,
    StepMethod,
    __version__,
    check_bias,
    check_term,
    evaluate_bias,
    evaluate_native,
    run_optmain,
    run_relaxedscan,
)
from mopt_rs.bitss import Bitss, BitssResult, bitss_search
from mopt_rs.confsearch import ConformerSearchResult, conformer_move_names, conformer_search
from mopt_rs.dmf import DirectMaxFlux, DmfPath, DmfResult, dmf_search
from mopt_rs.ieip import (
    Addf,
    Dimer,
    DsAfir,
    ElasticImagePair,
    FreezingString,
    GrowingString,
    IeipJob,
    NewtonTrajectory,
    SearchResult,
    SpringPair,
    TwoPointShs,
    addf_search,
    dimer_search,
    dsafir_search,
    fsm_search,
    gsm_search,
    ieip_search,
    newton_trajectory,
    pair_methods,
    run_pair_method,
    spring_pair_search,
    two_point_shs,
)
from mopt_rs.irc import Irc, IrcResult, IrcSide, irc_path
from mopt_rs.md import MdResult, MolecularDynamics, TrajectoryAnalysis, run_md
from mopt_rs.optimize import Calculator, Evaluation, NativeCalculator, Result, Task, run
from mopt_rs.orient import OrientationSearch, OrientMinimum, OrientResult, orientation_search
from mopt_rs.path_analysis import PathBifurcation, PathProfile, path_profile
from mopt_rs.saddles import (
    FixedRidgeBand,
    HighIndexSaddleSearch,
    LowestModes,
    RidgeBand,
    SaddleResult,
    hisd_downward_starts,
    hisd_upward_starts,
    lowest_modes,
    ridge_band_search,
    saddle_search,
)
from mopt_rs.variables import (
    NativeObjective,
    VariableResult,
    VectorOptimizer,
    objective_names,
    optimize_variables,
)
from mopt_rs.workflows import FollowedCalculator, MapperNetwork, relax_band, relax_environment
from mopt_rs.xyz import XyzFrame, append_xyz, read_xyz, read_xyz_frames, write_xyz

__all__ = [
    # The ONIOM calculator of the commands (-oniom), for Optimizer and relax_band.
    "OniomCalculator",
    # Structure files (.xyz, extended .xyz) with the reader and writer of the commands.
    "XyzFrame",
    "append_xyz",
    "read_xyz",
    "read_xyz_frames",
    "write_xyz",
    "FollowedCalculator",
    "MapperNetwork",
    "relax_band",
    "relax_environment",
    "run_optmain",
    # Scans of coordinates and ReSPer coordinates (relaxedscan).
    "ResperMap",
    "run_relaxedscan",
    # The searches of ieipmain.
    "Addf",
    "Dimer",
    "DsAfir",
    "ElasticImagePair",
    "FreezingString",
    "GrowingString",
    "IeipJob",
    "NewtonTrajectory",
    "SearchResult",
    "SpringPair",
    "TwoPointShs",
    "addf_search",
    "dimer_search",
    "dsafir_search",
    "fsm_search",
    "gsm_search",
    "ieip_search",
    "newton_trajectory",
    "pair_methods",
    "run_pair_method",
    "spring_pair_search",
    "two_point_shs",
    # Saddle points of index k (high-index saddle dynamics, bands on a ridge).
    "FixedRidgeBand",
    "HighIndexSaddleSearch",
    "LowestModes",
    "RidgeBand",
    "SaddleResult",
    "hisd_downward_starts",
    "hisd_upward_starts",
    "lowest_modes",
    "ridge_band_search",
    "saddle_search",
    "Bitss",
    "BitssResult",
    "Calculator",
    "DirectMaxFlux",
    "DmfPath",
    "DmfResult",
    "Evaluation",
    "Irc",
    "IrcResult",
    "IrcSide",
    "MdResult",
    "MolecularDynamics",
    "NativeCalculator",
    "NativeObjective",
    "Optimizer",
    # The conformer search of confsearch, with search moves of the caller.
    "ConformerSearchResult",
    "conformer_move_names",
    "conformer_search",
    # The orientation search of orientsearch.
    "OrientMinimum",
    "OrientResult",
    "OrientationSearch",
    "orientation_search",
    # The vibrations transverse to a reaction path and its bifurcations (freqmain -path).
    "PathBifurcation",
    "PathProfile",
    "path_profile",
    "Result",
    "StepMethod",
    "Task",
    "TrajectoryAnalysis",
    "VariableResult",
    "VectorOptimizer",
    "__version__",
    "bitss_search",
    "check_bias",
    "check_term",
    "dmf_search",
    "evaluate_bias",
    "evaluate_native",
    "irc_path",
    "objective_names",
    "optimize_variables",
    "run",
    "run_md",
]
