# Searches of `ieipmain` (iEIP, dimer, spring pair, 2PSHS, ADDF, growing Newton trajectory)

The methods of `ieipmain` are written anew in Rust. Where they come from:

- Improved elastic image pair method, `crates/mopt-rs/src/path/ieip/`: MultiOptPy 1.21.2
  (`OtherMethod/elastic_image_pair.py`, GPL-3.0-or-later, the predecessor of mopt-rs), whose
  procedure is kept iteration by iteration (`tests/golden/ieip_lj_*.json`, made by
  `tools/golden/run_ieip_reference.py`), after
  > Y. Liu, H. Qi, M. Lei, J. Chem. Theory Comput. 2023, 19, 2410 (the improved elastic
  > image pair method).

  The weights of the attraction between the images follow DS-AFIR,
  > S. Maeda, Y. Harabuchi, M. Takagi, K. Saita, K. Suzuki, T. Ichino, Y. Sumiya, K.
  > Sugiyama, Y. Ono, J. Comput. Chem. 2018, 39, 233 (eqs. 6-10),

  in MultiOptPy's form (`Y = Z / (Z + 1) + 1/2`, where the paper has `Z / (Z + 1)`).
- Dimer method, `crates/mopt-rs/src/path/dimer/`:
  > G. Henkelman, H. Jonsson, J. Chem. Phys. 1999, 111, 7010;
  > A. Heyden, A. T. Bell, F. J. Keil, J. Chem. Phys. 2005, 123, 224101 (the rotation from
  > one gradient and the fit of the curvature);
  > J. Kastner, P. Sherwood, J. Chem. Phys. 2008, 128, 014106 (the extrapolated gradient and
  > the L-BFGS translation).
- Spring pair method, `crates/mopt-rs/src/path/spring_pair.rs`:
  > G. Cui, K. Jiang, Phys. Rev. E 2024, 110, 064123 (arXiv:2407.04373), Algorithm 1.
- Scaled hypersphere coordinates, two-point SHS, and ADDF,
  `crates/mopt-rs/src/path/shs.rs`, `crates/mopt-rs/src/path/twopshs.rs`,
  `crates/mopt-rs/src/workflow/explore/addf/`:
  > K. Ohno, S. Maeda, Chem. Phys. Lett. 2004, 384, 277 (SHS);
  > S. Maeda, K. Ohno, Chem. Phys. Lett. 2005, 404, 95 (2PSHS);
  > S. Maeda, K. Ohno, J. Phys. Chem. A 2005, 109, 5742 (ADDF);
  > S. Maeda, K. Ohno, K. Morokuma, Phys. Chem. Chem. Phys. 2013, 15, 3683, and S. Maeda et
  > al., Bull. Chem. Soc. Jpn. 2014, 87, 1315 (reviews: sizes of the spheres, the elimination
  > with `cos^3`, the large-ADD variant).
- Growing Newton trajectory, `crates/mopt-rs/src/path/newton_trajectory.rs`:
  > W. Quapp, J. Chem. Phys. 2005, 122, 174106;
  > W. Quapp, M. Hirsch, O. Imig, D. Heidrich, J. Comput. Chem. 1998, 19, 1087.

  The predictor, the threshold of the corrector, and the update of the direction follow the
  procedure of pysisyphus (`pysisyphus/cos/GrowingNT.py`, J. Steinmetzer, GPL-3.0), which
  MultiOptPy ports; no code of pysisyphus is included.

Methods of `ieipmain` that MultiOptPy does not have:

- Growing string, `crates/mopt-rs/src/path/gsm/`:
  > B. Peters, A. Heyden, A. T. Bell, A. Chakraborty, J. Chem. Phys. 2004, 120, 7877;
  > P. M. Zimmerman, J. Chem. Phys. 2013, 138, 184102 (in internal coordinates);
  > P. M. Zimmerman, J. Chem. Theory Comput. 2013, 9, 3043 (the climbing node and the search
  > for the transition state integrated with the string);
  > P. M. Zimmerman, J. Comput. Chem. 2015, 36, 601 (single-ended, driving coordinates).
- Freezing string, `crates/mopt-rs/src/path/fsm.rs`:
  > A. Behn, P. M. Zimmerman, A. T. Bell, M. Head-Gordon, J. Chem. Phys. 2011, 135, 224108.

  The number of gradients per node and of nodes follow the Q-Chem manual's description of
  its implementation (`FSM_NGRAD` 3, `FSM_NNODE` 12); the bound of 0.3 Angstrom per
  component of a step follows arXiv:2407.09763.
- Double-sphere AFIR, `crates/mopt-rs/src/path/dsafir.rs`:
  > S. Maeda, Y. Harabuchi, M. Takagi, K. Saita, K. Suzuki, T. Ichino, Y. Sumiya, K.
  > Sugiyama, Y. Ono, J. Comput. Chem. 2018, 39, 233 (eqs. 6-10, the procedure);
  > R. Elber, I. Ulitsky, J. Chem. Phys. 1990, 92, 1510, and R. Elber, C. Choi, J. Chem.
  > Phys. 1991, 94, 751 (the locally updated planes that relax the path).

  The constants the paper leaves open (the force `delta` 0.03 Hartree/Bohr, the Euler step
  of 0.1 Angstrom, the orientation of the second structure onto the first at the start, five
  relaxations of points 0.5 Angstrom apart) are those its authors' program documents for
  its defaults in its public manual; no code of that program was seen or run.

Experimental methods (off by default):

- Local minimax method, `crates/mopt-rs/src/path/lmm.rs`:
  > Y. Li, J. Zhou, SIAM J. Sci. Comput. 2001, 23, 840 (the method);
  > arXiv:2109.01865 (the normalized Armijo rule and the steps of Barzilai and Borwein as
  > it restates them).
- Descent-ascent steps of the dimer translation, `crates/mopt-rs/src/path/dimer/mod.rs`:
  > G. M. Korpelevich, Ekonomika i Matematicheskie Metody 1976, 12, 747 (extragradient);
  > optimistic steps as arXiv:1901.08511 analyses them for convex-concave problems.

What differs from MultiOptPy is in `docs/source/migration_paths.md`, "Transition states from two
structures or one (`ieipmain`)", and in the module documentation of each file.
