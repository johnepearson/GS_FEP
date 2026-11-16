"""Marginal r(I,E), PC1 partial r and screened fraction for each ensemble seed base.
Usage: python3 marginal_screened_ensembles.py            (seed bases SeedOne, SeedTwo, SeedThree of initialize.tex)
       python3 marginal_screened_ensembles.py 1000       (named seed bases: prints only, writes no macros)
Run from the directory holding ensemble_merged_seed*_worker*.npz (build/).

Writes results/marginal_screened.tex. For each ensemble k in One, Two, Three it defines the macros
Ens<k>N (organism-instances), Ens<k>Marginal (marginal r), Ens<k>Partial (PC1 partial r) and
Ens<k>Screened (percent screened), which the paper uses in Table 1."""
import sys
import numpy as np
from analyze_ensemble_pc1 import load
from partial_corr_utils import pc1_partial_corr
from params import load as load_params
from texout import num, write_macros

P = load_params("initialize.tex")
ORDINALS = ["One", "Two", "Three"]
defaults = [P["SeedOne"], P["SeedTwo"], P["SeedThree"]]
given = [int(a) for a in sys.argv[1:]]
seeds = given or defaults
macros = []
for s in seeds:
    out = load(s)
    if out is None:
        print(f"seed {s}: no files found"); continue
    I, E, fu, fv = out[:4]
    rm = np.corrcoef(I, E)[0, 1]
    rp = pc1_partial_corr(I, E, fu, fv)
    rp = rp[0] if isinstance(rp, tuple) else rp
    scr = 100 * (1 - rp ** 2 / rm ** 2)
    print(f"seed {s}: N={len(I)}  marginal r={rm:.4f}  PC1 partial r={rp:.4f}  screened={scr:.1f}%")
    if not given:
        k = ORDINALS[defaults.index(s)]
        macros += [(f"Ens{k}N", num(len(I), 0)), (f"Ens{k}Marginal", num(rm, 6)),
                   (f"Ens{k}Partial", num(rp, 6)), (f"Ens{k}Screened", num(scr, 3))]
if macros:
    write_macros("results/marginal_screened.tex", "marginal_screened_ensembles.py", macros)
