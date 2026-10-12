"""
PC1-convention analysis of an independent ensemble (the paper's convention), for comparison across ensembles.

analyze_ensemble_merged.py prints the JOINT two-flux partial correlation; the paper's headline uses the single PC1 flux
regressor. This prints, for each requested seed base (default 1000, 9000, 20000), N, the partial r four ways
(PC1, joint, Phi_u only, Phi_v only), the per-realization PC1 distribution, and a realization-cluster bootstrap CI
for the pooled PC1 value (PC1 direction fixed from the pooled data).

Usage:  python3 analyze_ensemble_pc1.py            (seed bases 1000 9000 20000)
        python3 analyze_ensemble_pc1.py 20000
"""
import glob
import sys
import numpy as np
from partial_corr_utils import pc1_partial_corr, check_partial_corr_stability


def load(seed_base):
    files = sorted(glob.glob(f"ensemble_merged_seed{seed_base}_worker*.npz"))
    if not files:
        return None
    ds = [np.load(f) for f in files]
    g = lambda k: np.concatenate([d[k] for d in ds])
    I, E, fu, fv, rid = g("I"), g("E"), g("fu"), g("fv"), g("realization_id")
    ok = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    return I[ok], E[ok], fu[ok], fv[ok], rid[ok], len(files)


def pcorr_moments(n, S, Q):
    m = S / n
    C = Q / n - np.outer(m, m)
    return (C[0, 1] - C[0, 2] * C[1, 2] / C[2, 2]) / np.sqrt((C[0, 0] - C[0, 2] ** 2 / C[2, 2]) * (C[1, 1] - C[1, 2] ** 2 / C[2, 2]))


def report(seed_base):
    out = load(seed_base)
    if out is None:
        print(f"\nseed base {seed_base}: no files found")
        return
    I, E, fu, fv, rid, nfiles = out
    r_pc1, n, expl, (mean, std, v1) = pc1_partial_corr(I, E, fu, fv, return_direction=True)
    r_both, r_fu, r_fv, cond_raw, suspect = check_partial_corr_stability(I, E, fu, fv)
    print(f"\n=== seed base {seed_base}: {nfiles} worker files, {len(np.unique(rid))} realizations, N = {n:,} ===")
    print(f"  pooled partial r:  PC1 {r_pc1:+.4f}   joint {r_both:+.4f}   Phi_u only {r_fu:+.4f}   Phi_v only {r_fv:+.4f}")
    P = ((np.column_stack([fu, fv]) - mean) / std) @ v1
    X = np.column_stack([I, E, P])
    ids = np.unique(rid)
    nn = np.zeros(len(ids)); S = np.zeros((len(ids), 3)); Q = np.zeros((len(ids), 3, 3))
    per = []
    for k, r in enumerate(ids):
        x = X[rid == r]
        nn[k] = len(x); S[k] = x.sum(0); Q[k] = x.T @ x
        if len(x) >= 20:
            per.append(pcorr_moments(nn[k], S[k], Q[k]))
    per = np.array(per)
    print(f"  per-realization PC1 partial r: mean {per.mean():+.4f}, SD {per.std():.4f}, median {np.median(per):+.4f}, "
          f"IQR [{np.percentile(per,25):+.4f}, {np.percentile(per,75):+.4f}], same sign as pooled: {100*(np.sign(per)==np.sign(r_pc1)).mean():.1f}%")
    rng = np.random.default_rng(0)
    boot = np.empty(2000)
    for b in range(2000):
        idx = rng.integers(0, len(ids), len(ids))
        boot[b] = pcorr_moments(nn[idx].sum(), S[idx].sum(0), Q[idx].sum(0))
    print(f"  realization-cluster bootstrap 95% CI on pooled PC1 r: [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]")


if __name__ == "__main__":
    bases = [int(a) for a in sys.argv[1:]] or [1000, 9000, 20000]
    for b in bases:
        report(b)
