# Molecular dynamics: sources of the methods

The molecular dynamics of mopt-rs (`crates/mopt-rs/src/md/`, `mdmain`) is written from the
publications below; no code of other programs is included. MultiOptPy 1.21.2
(`moleculardynamics.py`, `MD/thermostat.py`, `Constraint/constraint_condition.py:SHAKE`,
`Potential/nanoreactor_potential.py`) gave the command, its options, and its files; where
its equations differ from the publications, mopt-rs follows the publications
(`docs/source/migration_workflows.md`, "Molecular dynamics").

| Part | Publication |
|---|---|
| velocity Verlet (`md::integrator::VelocityVerlet`) | W. C. Swope, H. C. Andersen, P. H. Berens, K. R. Wilson, J. Chem. Phys. 1982, 76, 637 |
| Trotter splitting of the thermostat and barostat stages; Nose-Hoover chains with Suzuki-Yoshida weights (`md::thermostat::nose_hoover`) | G. J. Martyna, M. E. Tuckerman, D. J. Tobias, M. L. Klein, Mol. Phys. 1996, 87, 1117 |
| Nose-Hoover chains, masses of the links | G. J. Martyna, M. L. Klein, M. Tuckerman, J. Chem. Phys. 1992, 97, 2635 |
| BAOAB (`md::integrator::Baoab`) | B. Leimkuhler, C. Matthews, Appl. Math. Res. Express 2013, 34; J. Chem. Phys. 2013, 138, 174102 |
| Berendsen thermostat and barostat | H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola, J. R. Haak, J. Chem. Phys. 1984, 81, 3684 |
| canonical velocity rescaling (`md::thermostat::Bussi`) | G. Bussi, D. Donadio, M. Parrinello, J. Chem. Phys. 2007, 126, 014101 |
| gamma deviates for the velocity rescaling | G. Marsaglia, W. W. Tsang, ACM Trans. Math. Softw. 2000, 26, 363 |
| Andersen thermostat | H. C. Andersen, J. Chem. Phys. 1980, 72, 2384 |
| stochastic cell rescaling (`md::barostat::CellRescale`) | M. Bernetti, G. Bussi, J. Chem. Phys. 2020, 153, 114107 |
| Martyna-Tobias-Klein barostat (`md::barostat::Mtk`) | G. J. Martyna, D. J. Tobias, M. L. Klein, J. Chem. Phys. 1994, 101, 4177; M. E. Tuckerman, J. Alejandre, R. Lopez-Rendon, A. L. Jochim, G. J. Martyna, J. Phys. A 2006, 39, 5629 |
| SHAKE (`md::constraints::Rattle::shake`) | J.-P. Ryckaert, G. Ciccotti, H. J. C. Berendsen, J. Comput. Phys. 1977, 23, 327; matrix form: V. Krautler, W. F. van Gunsteren, P. H. Hunenberger, J. Comput. Chem. 2001, 22, 501 |
| RATTLE (`md::constraints::Rattle::project_velocities`) | H. C. Andersen, J. Comput. Phys. 1983, 52, 24 |
| nanoreactor boundary potential (`pes::bias::NanoReactor`, time from the dynamics) | L.-P. Wang, A. Titov, R. McGibbon, F. Liu, V. S. Pande, T. J. Martinez, Nat. Chem. 2014, 6, 1044 |
