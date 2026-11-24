# Inputs of the quick start

The structure files and `config_autots_run_xtb_test.json` in this directory are the inputs
of the examples in the README of MultiOptPy, taken unchanged from the directory `test/` of
MultiOptPy 1.21.2 (by ss0832, GPL-3.0-or-later; mopt-rs is its successor by the same
author). `config_autots_builtin.json` is that settings file with the engine built
into mopt-rs (`"othersoft": "gfn1xtb"` in place of `"usextb": "GFN2-xTB"`) and another
work directory.

| File | What it is |
|---|---|
| `SN2.xyz` | fluoromethane and hydroxide (charge -1, in its second line) |
| `aldol_rxn.xyz` | acetaldehyde and the enol of acetaldehyde |
| `aldol_rxn_PT.xyz` | a guess of a transition state of the aldol system |
| `aldol_rxn/` | 17 structures along a path of the aldol reaction |
| `ieip_test/` | HCN and HNC |
| `s8_for_confomation_search_test.xyz` | an S8 ring |

The commands that use them are in the README at the top of the repository and, with what
each run did, in `docs/source/quickstart.md`.
