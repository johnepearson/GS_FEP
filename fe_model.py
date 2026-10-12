"""Shared Friston generative-model constants, refit live from the merge-corrected primary ensemble
(ensemble_merged_seed1000_worker0..11.npz, marker spacing 24). Same procedure as rebuild_fig5_and_fe_model.py
(prior = mean/var of E; likelihood = OLS of (fu,fv) on E with residual covariance; posterior = OLS of E on I).
Cached in fe_model_constants.npz; delete that file to force a refit.
"""
import os
import numpy as np

N_WORKERS, SEED_BASE = 12, 1000
CACHE = "fe_model_constants.npz"


def refit():
    I, E, fu, fv = [], [], [], []
    for w in range(N_WORKERS):
        d = np.load(f"ensemble_merged_seed{SEED_BASE}_worker{w}.npz")
        I.append(d["I"]); E.append(d["E"]); fu.append(d["fu"]); fv.append(d["fv"])
    I, E, fu, fv = (np.concatenate(x) for x in (I, E, fu, fv))
    g = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    I, E, fu, fv = I[g], E[g], fu[g], fv[g]
    N = len(I)
    Bo = np.column_stack([np.ones(N), E])
    cu, *_ = np.linalg.lstsq(Bo, fu, rcond=None)
    cv, *_ = np.linalg.lstsq(Bo, fv, rcond=None)
    Sigma = np.cov(np.column_stack([fu - Bo @ cu, fv - Bo @ cv]).T)
    Bi = np.column_stack([np.ones(N), I])
    cq, *_ = np.linalg.lstsq(Bi, E, rcond=None)
    return dict(N=N, MU_P=E.mean(), SIGMA_P2=E.var(), A=np.array([cu[1], cv[1]]), B=np.array([cu[0], cv[0]]),
                SIGMA=Sigma, SLOPE=cq[1], INTERCEPT=cq[0], SIGMA_Q2=(E - Bi @ cq).var())


def load_constants():
    if os.path.exists(CACHE):
        c = dict(np.load(CACHE))
    else:
        c = refit()
        np.savez(CACHE, **c)
    print(f"[fe_model] generative model from N={int(c['N'])}: mu_p={float(c['MU_P']):.4f} "
          f"sigma_p2={float(c['SIGMA_P2']):.4e} slope={float(c['SLOPE']):.4f} sigma_q2={float(c['SIGMA_Q2']):.4e}")
    return (float(c["MU_P"]), float(c["SIGMA_P2"]), c["A"], c["B"], c["SIGMA"],
            float(c["SLOPE"]), float(c["INTERCEPT"]), float(c["SIGMA_Q2"]))
