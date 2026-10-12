"""
Calibration of the CMI nulls on CURRENT data (reviewer request): do the two nulls used in the paper have the
right type-I error when I is conditionally independent of E given the blanket B = (flux_u, flux_v) by construction?

Construction (conditional surrogate): fit random-forest regressions mu_I(B), mu_E(B) to the real master-trajectory
data, then build synthetic data that keeps the real B values and the fitted conditional means but replaces the
residuals with two INDEPENDENTLY permuted draws of the real residual pools. I and E are then conditionally
independent given B exactly, with the real marginal shapes and the real (nonlinear) B-dependence.

Each replicate draws fresh permutations and an N=8000 subsample, z-scores I, E and both flux columns, and computes
    observed CMI, the 30-cluster null, and the local nearest-neighbour (Runge-style) null  (n-null trials each).
For a calibrated null the statistic z = (observed - null mean) / null SD should have mean ~0 and SD ~1 across
replicates, and the fraction of replicates with z > 1.645 should be ~5% (z > 1.96: ~2.5%).

Usage:
    python run_null_calibration.py --quick
    python run_null_calibration.py --n-rep 48 --jobs 12          # about 20-30 minutes
Writes null_calibration.npz and prints a summary. Resumable (null_calibration_cache.pkl).
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from conditional_mi import cmi_ksg
from run_u_enriched_cmi import zscore, cluster_null, local_null

CACHE = "null_calibration_cache.pkl"


def fit_surrogate(I, E, B, seed):
    rf_kw = dict(n_estimators=100, min_samples_leaf=50, n_jobs=1, random_state=seed)
    muI = RandomForestRegressor(**rf_kw).fit(B, I).predict(B)
    muE = RandomForestRegressor(**rf_kw).fit(B, E).predict(B)
    return muI, muE, I - muI, E - muE


def replicate(args):
    rep, B, muI, muE, epsI, epsE, n_sub, n_null = args
    r = np.random.default_rng(10_000 + rep)
    N = len(B)
    Isyn = muI + epsI[r.permutation(N)]
    Esyn = muE + epsE[r.permutation(N)]
    idx = r.choice(N, n_sub, replace=False)
    Ii, Ei = zscore(Isyn[idx])[:, 0], zscore(Esyn[idx])[:, 0]
    Z = zscore(B[idx])
    t0 = time.time()
    obs = cmi_ksg(Ii, Ei, Z)
    nc = cluster_null(Ii, Ei, Z, n_null, rep)
    nl = local_null(Ii, Ei, Z, n_null, rep)
    return rep, dict(obs=obs, cl_mean=nc.mean(), cl_sd=nc.std(), lo_mean=nl.mean(), lo_sd=nl.std(), secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--n-rep", type=int, default=48)
    ap.add_argument("--n-sub", type=int, default=8000)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_rep, a.n_sub, a.n_null = 4, 1500, 10
    d = np.load(a.data)
    I, E, fu, fv = d["I"], d["E"], d["fu"], d["fv"]
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    I, E, B = I[good], E[good], np.column_stack([fu[good], fv[good]])
    print(f"{good.sum():,} organism-instances; fitting random-forest surrogates")
    muI, muE, epsI, epsE = fit_surrogate(I, E, B, 0)
    print(f"surrogate: R^2 of I on B = {1-epsI.var()/I.var():.3f}, of E on B = {1-epsE.var()/E.var():.3f}")

    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    todo = [rep for rep in range(a.n_rep) if (rep, a.n_sub, a.n_null) not in cache]
    jobs = [(rep, B, muI, muE, epsI, epsE, a.n_sub, a.n_null) for rep in todo]
    print(f"{len(jobs)} replicates to run, {a.jobs} workers")
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        for rep, res in ex.map(replicate, jobs):
            cache[(rep, a.n_sub, a.n_null)] = res
            pickle.dump(cache, open(CACHE, "wb"))
            print(f"  replicate {rep} done  obs={res['obs']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    rows = [cache[(rep, a.n_sub, a.n_null)] for rep in range(a.n_rep)]
    out = {}
    print(f"\nType-I calibration over {len(rows)} replicates with I independent of E given B (N={a.n_sub}, {a.n_null} null trials)")
    for name, mk, sk in (("30-cluster null", "cl_mean", "cl_sd"), ("local NN null", "lo_mean", "lo_sd")):
        z = np.array([(r["obs"] - r[mk]) / r[sk] for r in rows])
        ex_ = np.array([r["obs"] - r[mk] for r in rows])
        print(f"  {name}: z mean {z.mean():+.2f}, z SD {z.std(ddof=1):.2f}; "
              f"fraction z>1.645: {(z > 1.645).mean():.3f}; z>1.96: {(z > 1.96).mean():.3f}; "
              f"mean excess {ex_.mean():+.4f} nats (SD {ex_.std(ddof=1):.4f})")
        out[name.replace(' ', '_') + "_z"] = z
        out[name.replace(' ', '_') + "_excess"] = ex_
    print("  (calibrated: z mean ~0, z SD ~1, fractions ~0.05 and ~0.025)")
    np.savez("null_calibration.npz", **out)
    print("wrote null_calibration.npz")


if __name__ == "__main__":
    main()
