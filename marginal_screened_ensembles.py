"""Marginal r(I,E), PC1 partial r and screened fraction for each ensemble seed base.
Usage: python3 marginal_screened_ensembles.py            (seed bases 1000 9000 20000)
Run from the directory holding ensemble_merged_seed*_worker*.npz."""
import sys
import numpy as np
from analyze_ensemble_pc1 import load
from partial_corr_utils import pc1_partial_corr

seeds = [int(a) for a in sys.argv[1:]] or [1000, 9000, 20000]
for s in seeds:
    out = load(s)
    if out is None:
        print(f"seed {s}: no files found"); continue
    I, E, fu, fv = out[:4]
    rm = np.corrcoef(I, E)[0, 1]
    rp = pc1_partial_corr(I, E, fu, fv)
    rp = rp[0] if isinstance(rp, tuple) else rp
    print(f"seed {s}: N={len(I)}  marginal r={rm:.4f}  PC1 partial r={rp:.4f}  screened={100*(1-rp**2/rm**2):.1f}%")
